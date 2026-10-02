// npfc_py.cpp — pybind11 entry points for npfixedcomppy._core.
//
// The Python front-end (python/npfixedcomppy/npfc.py) calls one of:
//   npnormll / npnormcvm / npnormad / nptll / npnormcll / nppoisll
//   <same>_estpi0
// with the same positional payload the old PyO3 module accepted, and
// expects the same result dict back. All heavy computation lives in the
// C++/Eigen engine (npfc_engine.h) — there is no intermediate language.
//
// This file also carries the thin per-entry orchestration ported from
// npfixedcomppy/src/lib.rs (data transforms before the grid construction:
// sorting for CVM/AD, the atanh/tanh transform for the correlation family,
// the probability-integral transform for the t family, the count
// aggregation for the Poisson family, and the default grid/initial-mix
// construction from npfc_grid.h).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "npfc_engine.h"
#include "npfc_families.h"
#include "npfc_fam2d.h"
#include "npfc_grid.h"
#include "npfc_corrmatrix.h"

namespace py = pybind11;

using npfc::MixResult;
using npfc::MixSolver;

namespace {

// ---------------------------------------------------------------------------
// orchestration helpers (ported from src/lib.rs)
// ---------------------------------------------------------------------------

py::dict result_to_dict(const MixResult& r) {
    // Re-acquire the GIL: the entry points run under
    // `call_guard<gil_scoped_release>` so the heavy C++ compute happens without
    // the GIL, but constructing CPython objects (dict/list/str) REQUIRES it.
    // Without this, PyDict_New / PyList / str conversions run on a released GIL
    // — undefined behaviour that corrupts CPython state and later faults as a
    // garbage indirect call. gil_scoped_acquire re-takes the GIL for this
    // scope and releases it on return (nested acquire inside the released
    // zone is well-defined in pybind11).
    py::gil_scoped_acquire _acquire;
    py::dict d;
    d["pt"] = py::cast(r.pt);
    d["pr"] = py::cast(r.pr);
    d["beta"] = r.beta;
    d["family"] = r.family;
    d["min_gradient"] = r.min_gradient;
    d["ll"] = r.ll;
    d["flag"] = r.flag;
    d["iter"] = r.iter;
    d["convergence"] = r.convergence;
    return d;
}

// ---------------------------------------------------------------------------
// zero-copy numpy seams
// ---------------------------------------------------------------------------

// GIL-free copy from a `py::array_t<double>` argument into a std::vector.
// The by-value argument is converted to a C-contiguous float64 array by
// pybind11 while the GIL is still held, and the owning handle keeps it
// alive for the whole call; by the time this runs the call guard has
// released the GIL, so `size()` and `data()` (plain buffer-info reads,
// no CPython API) plus the memcpy are safe. The caller must not mutate
// the input array from another thread during the call.
inline std::vector<double> vec_from(const py::array_t<double>& a) {
    const std::size_t n = static_cast<std::size_t>(a.size());
    if (n == 0)
        return {};
    const double* p = static_cast<const double*>(a.data());
    return std::vector<double>(p, p + n);
}

// Build an owning numpy float64 array from a std::vector. Re-acquires the
// GIL first: CPython object construction requires it (the body otherwise
// runs with the GIL released by the call guard).
inline py::array_t<double> array_from(const std::vector<double>& v) {
    py::gil_scoped_acquire acq;
    py::array_t<double> out(static_cast<std::size_t>(v.size()));
    if (!v.empty())
        std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
    return out;
}

// R's `atanh(x) = 0.5 * log((1 + x)/(1 - x))` (for |x| < 1). Named
// `atanh_r` to avoid colliding with the C library `atanh` that MSVC exposes
// in the global namespace.
inline double atanh_r(double x) {
    return 0.5 * (std::log(1.0 + x) - std::log(1.0 - x));
}

// Stable `tanh(x)` (avoids overflow in the naive exp form).
inline double tanhv(double x) {
    if (x >= 0.0) {
        const double e2 = std::exp(-2.0 * x);
        return (1.0 - e2) / (1.0 + e2);
    }
    const double e2 = std::exp(2.0 * x);
    return (e2 - 1.0) / (e2 + 1.0);
}

std::vector<double> ones(std::size_t n) { return std::vector<double>(n, 1.0); }

// R's `table(v)`: group equal (integer-valued) observations and return
// `(unique values, counts)` sorted ascending.
std::vector<std::vector<double>>
aggregate_counts(const std::vector<double>& data) {
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end(),
              [](double a, double b) { return a < b; });
    std::vector<double> v, w;
    v.reserve(sorted.size());
    w.reserve(sorted.size());
    for (double x : sorted) {
        if (!v.empty() && v.back() == x) {
            w.back() += 1.0;
            continue;
        }
        v.push_back(x);
        w.push_back(1.0);
    }
    return {std::move(v), std::move(w)};
}

// The R wrapper's extended default grid for the CVM/AD families:
// `c(gridpoints.npnorm(v, beta), v[1] - 3*beta, v[n] + 3*beta)` on sorted
// `v`.
std::vector<double> default_grid_extended(const std::vector<double>& v_sorted,
                                          double beta) {
    const std::vector<double> w = ones(v_sorted.size());
    std::vector<double> g = npfc::grid::gridpoints_npnorm(v_sorted, w, beta, 100);
    g.push_back(v_sorted.front() - 3.0 * beta);
    g.push_back(v_sorted.back() + 3.0 * beta);
    return g;
}

// Weighted variant of `default_grid_extended` for the binned CVM/AD families:
// the histogram breaks are driven by the bin COUNTS (not unit weights) —
// the same `c(gridpoints.npnorm(v2), v[1] - 3*beta, v[n] + 3*beta)` the R
// `npnormcvmw`/`npnormadw` wrappers build on `v2 = npnorm(v1$v, w = v1$w)`.
std::vector<double> default_grid_extended_w(const std::vector<double>& v_sorted,
                                            const std::vector<double>& w,
                                            double beta) {
    std::vector<double> g = npfc::grid::gridpoints_npnorm(v_sorted, w, beta, 100);
    g.push_back(v_sorted.front() - 3.0 * beta);
    g.push_back(v_sorted.back() + 3.0 * beta);
    return g;
}

}  // namespace

// ---------------------------------------------------------------------------
// the six computemixdist entries
// ---------------------------------------------------------------------------

py::dict npnormll(py::array_t<double> data, py::array_t<double> mu0fixed,
                  py::array_t<double> pi0fixed, double beta,
                  py::array_t<double> initpt, py::array_t<double> initpr,
                  py::array_t<double> gridpoints, double tol, long maxit,
                  long verbose) {
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<double> w = ones(data_v.size());
    auto init =
        npfc::grid::initial_npnorm(data_v, w, beta, initpt_v, initpr_v);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? npfc::grid::gridpoints_npnorm(
                                         data_v, w, beta, 100)
                                   : gridpoints_v;
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormLL>(data_v, mu0fixed_v, pi0fixed_v,
                                              beta),
        mu0fixed_v, pi0fixed_v, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormll_estpi0(py::array_t<double> data, double beta, double val,
                         py::array_t<double> initpt,
                         py::array_t<double> initpr,
                         py::array_t<double> gridpoints, double tol,
                         long verbose, bool fast, bool relax,
                         double inner_tol) {
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<double> w = ones(data_v.size());
    auto init =
        npfc::grid::initial_npnorm(data_v, w, beta, initpt_v, initpr_v);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? npfc::grid::gridpoints_npnorm(
                                         data_v, w, beta, 100)
                                   : gridpoints_v;
    std::vector<double> mu0f{0.0}, pi0f{1.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormLL>(data_v, mu0f, pi0f, beta),
        mu0f, pi0f, std::get<1>(init),
        std::get<2>(init), grid, static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict npnormcvm(py::array_t<double> data_in, py::array_t<double> mu0fixed,
                   py::array_t<double> pi0fixed, double beta,
                   py::array_t<double> initpt, py::array_t<double> initpr,
                   py::array_t<double> gridpoints, double tol, long maxit,
                   long verbose) {
    std::vector<double> data = vec_from(data_in);
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints_v.empty() ? default_grid_extended(data, beta) : gridpoints_v;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt_v, initpr_v);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCVM>(data, mu0fixed_v, pi0fixed_v,
                                               beta),
        mu0fixed_v, pi0fixed_v, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormcvm_estpi0(py::array_t<double> data_in, double beta, double val,
                          py::array_t<double> initpt,
                          py::array_t<double> initpr,
                          py::array_t<double> gridpoints, double tol,
                          long verbose, bool fast, bool relax,
                          double inner_tol) {
    std::vector<double> data = vec_from(data_in);
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints_v.empty() ? default_grid_extended(data, beta) : gridpoints_v;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt_v, initpr_v);
    std::vector<double> mu0f{0.0}, pi0f{0.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCVM>(data, mu0f, pi0f, beta),
        mu0f, pi0f, std::get<1>(init),
        std::get<2>(init), grid, static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict npnormad(py::array_t<double> data_in, py::array_t<double> mu0fixed,
                  py::array_t<double> pi0fixed, double beta,
                  py::array_t<double> initpt, py::array_t<double> initpr,
                  py::array_t<double> gridpoints, double tol, long maxit,
                  long verbose) {
    std::vector<double> data = vec_from(data_in);
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints_v.empty() ? default_grid_extended(data, beta) : gridpoints_v;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt_v, initpr_v);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormAD>(data, mu0fixed_v, pi0fixed_v,
                                              beta),
        mu0fixed_v, pi0fixed_v, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormad_estpi0(py::array_t<double> data_in, double beta, double val,
                         py::array_t<double> initpt,
                         py::array_t<double> initpr,
                         py::array_t<double> gridpoints, double tol,
                         long verbose, bool fast, bool relax,
                         double inner_tol) {
    std::vector<double> data = vec_from(data_in);
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints_v.empty() ? default_grid_extended(data, beta) : gridpoints_v;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt_v, initpr_v);
    std::vector<double> mu0f{0.0}, pi0f{0.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormAD>(data, mu0f, pi0f, beta),
        mu0f, pi0f, std::get<1>(init),
        std::get<2>(init), grid, static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict npnormcll(py::array_t<double> data, py::array_t<double> mu0fixed,
                   py::array_t<double> pi0fixed, double beta,
                   py::array_t<double> initpt, py::array_t<double> initpr,
                   py::array_t<double> gridpoints, double tol, long maxit,
                   long verbose) {
    if (!(beta > 3.0))
        throw py::value_error(
            "npnormcll requires beta > 3 (the number of observations)");
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const double beta_n = 1.0 / std::sqrt(beta - 3.0);
    std::vector<double> v_atanh(data_v.size());
    for (std::size_t i = 0; i < data_v.size(); ++i)
        v_atanh[i] = atanh_r(data_v[i]);
    const std::vector<double> w = ones(v_atanh.size());
    std::vector<double> grid;
    if (gridpoints_v.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v_atanh, w, beta_n, 100);
        grid = g;
        for (double& z : grid)
            z = tanhv(z);
    } else {
        grid = gridpoints_v;
    }
    std::vector<double> initpt_t;
    for (double x : initpt_v)
        initpt_t.push_back(atanh_r(x));
    auto init = npfc::grid::initial_npnorm(v_atanh, w, beta_n, initpt_t, initpr_v);
    std::vector<double> initpt_n;
    for (double z : std::get<1>(init))
        initpt_n.push_back(tanhv(z));
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCLL>(data_v, mu0fixed_v, pi0fixed_v,
                                               beta),
        mu0fixed_v, pi0fixed_v, std::move(initpt_n), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormcll_estpi0(py::array_t<double> data, double beta, double val,
                          py::array_t<double> initpt,
                          py::array_t<double> initpr,
                          py::array_t<double> gridpoints, double tol,
                          long verbose, bool fast, bool relax,
                          double inner_tol) {
    if (!(beta > 3.0))
        throw py::value_error(
            "npnormcll requires beta > 3 (the number of observations)");
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const double beta_n = 1.0 / std::sqrt(beta - 3.0);
    std::vector<double> v_atanh(data_v.size());
    for (std::size_t i = 0; i < data_v.size(); ++i)
        v_atanh[i] = atanh_r(data_v[i]);
    const std::vector<double> w = ones(v_atanh.size());
    std::vector<double> grid;
    if (gridpoints_v.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v_atanh, w, beta_n, 100);
        grid = g;
        for (double& z : grid)
            z = tanhv(z);
    } else {
        grid = gridpoints_v;
    }
    std::vector<double> initpt_t;
    for (double x : initpt_v)
        initpt_t.push_back(atanh_r(x));
    auto init = npfc::grid::initial_npnorm(v_atanh, w, beta_n, initpt_t, initpr_v);
    std::vector<double> initpt_n;
    for (double z : std::get<1>(init))
        initpt_n.push_back(tanhv(z));
    std::vector<double> mu0f{0.0}, pi0f{1.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCLL>(data_v, mu0f, pi0f, beta),
        mu0f, pi0f,
        std::move(initpt_n), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict nppoisll(py::array_t<double> data, py::array_t<double> mu0fixed,
                  py::array_t<double> pi0fixed, double beta,
                  py::array_t<double> initpt, py::array_t<double> initpr,
                  py::array_t<double> gridpoints, double tol, long maxit,
                  long verbose) {
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<std::vector<double>> vw = aggregate_counts(data_v);
    const std::vector<double>& v = vw[0];
    const std::vector<double>& w = vw[1];
    std::vector<double> grid =
        gridpoints_v.empty() ? npfc::grid::gridpoints_nppois(v, beta, 100)
                             : gridpoints_v;
    auto init = npfc::grid::initial_nppois(v, w, beta, initpt_v, initpr_v);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpPoisLL>(v, w, mu0fixed_v, pi0fixed_v,
                                              beta),
        mu0fixed_v, pi0fixed_v, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict nppoisll_estpi0(py::array_t<double> data, double beta, double val,
                         py::array_t<double> initpt,
                         py::array_t<double> initpr,
                         py::array_t<double> gridpoints, double tol,
                         long verbose, bool fast, bool relax,
                         double inner_tol) {
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    const std::vector<std::vector<double>> vw = aggregate_counts(data_v);
    const std::vector<double>& v = vw[0];
    const std::vector<double>& w = vw[1];
    std::vector<double> grid =
        gridpoints_v.empty() ? npfc::grid::gridpoints_nppois(v, beta, 100)
                             : gridpoints_v;
    auto init = npfc::grid::initial_nppois(v, w, beta, initpt_v, initpr_v);
    std::vector<double> mu0f{0.0}, pi0f{1.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpPoisLL>(v, w, mu0f, pi0f, beta),
        mu0f, pi0f, std::get<1>(init),
        std::get<2>(init), grid, static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict nptll(py::array_t<double> data, py::array_t<double> mu0fixed,
               py::array_t<double> pi0fixed, double beta,
               py::array_t<double> initpt, py::array_t<double> initpr,
               py::array_t<double> gridpoints, double tol, long maxit,
               long verbose) {
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    // PIT to z-space: v1 = qnorm(pt(v, df = beta)). Identity when beta = Inf.
    std::vector<double> v1(data_v.size());
    for (std::size_t i = 0; i < data_v.size(); ++i)
        v1[i] = npfc::stats::qnorm(npfc::stats::pt(data_v[i], beta, true));
    const std::vector<double> w = ones(v1.size());
    std::vector<double> grid;
    if (gridpoints_v.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v1, w, 1.0, 100);
        grid = g;
        for (double& z : grid)
            z = npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta);
    } else {
        grid = gridpoints_v;
    }
    std::vector<double> initpt_z;
    for (double x : initpt_v)
        initpt_z.push_back(npfc::stats::qnorm(npfc::stats::pt(x, beta, true)));
    auto init = npfc::grid::initial_npnorm(v1, w, 1.0, initpt_z, initpr_v);
    std::vector<double> initpt_t;
    for (double z : std::get<1>(init))
        initpt_t.push_back(
            npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta));
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpTLL>(data_v, mu0fixed_v, pi0fixed_v,
                                           beta),
        mu0fixed_v, pi0fixed_v, std::move(initpt_t), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict nptll_estpi0(py::array_t<double> data, double beta, double val,
                      py::array_t<double> initpt, py::array_t<double> initpr,
                      py::array_t<double> gridpoints, double tol, long verbose,
                      bool fast, bool relax, double inner_tol) {
    const std::vector<double> data_v = vec_from(data);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    // PIT to z-space (same transform as `nptll`).
    std::vector<double> v1(data_v.size());
    for (std::size_t i = 0; i < data_v.size(); ++i)
        v1[i] = npfc::stats::qnorm(npfc::stats::pt(data_v[i], beta, true));
    const std::vector<double> w = ones(v1.size());
    std::vector<double> grid;
    if (gridpoints_v.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v1, w, 1.0, 100);
        grid = g;
        for (double& z : grid)
            z = npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta);
    } else {
        grid = gridpoints_v;
    }
    std::vector<double> initpt_z;
    for (double x : initpt_v)
        initpt_z.push_back(npfc::stats::qnorm(npfc::stats::pt(x, beta, true)));
    auto init = npfc::grid::initial_npnorm(v1, w, 1.0, initpt_z, initpr_v);
    std::vector<double> initpt_t;
    for (double z : std::get<1>(init))
        initpt_t.push_back(
            npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta));
    // estpi0nptll_ has no mu0/pi0: the fixed component is the (zero-weight)
    // point mass at 0 whose proportion is estimated by the solver.
    std::vector<double> mu0f{0.0}, pi0f{0.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpTLL>(data_v, mu0f, pi0f, beta),
        mu0f, pi0f,
        std::move(initpt_t), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

// ---------------------------------------------------------------------------
// the four binned ("...w") families: the caller pre-bins the observations
// into (bin centres `data`, bin counts `weights`) and passes `h = 10^order`
// (the R `bin(v, order)` contract). The C++ family receives the bin centres
// with count weights; the kernel and every loss/gradient/weight sweep are
// count-weighted accordingly.
// ---------------------------------------------------------------------------

// Sort the bin centres (and their aligned counts) ascending. The R `bin`
// helper already returns sorted centres with aligned counts, so this is a
// no-op on the normal path — it only guards the count-weighted empirical-cdf
// assumptions of the CVM/AD precompute against an unsorted caller.
static std::pair<std::vector<double>, std::vector<double>>
sort_centers_weights(std::vector<double> v, std::vector<double> w) {
    const std::size_t n = v.size();
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](std::size_t a, std::size_t b) { return v[a] < v[b]; });
    std::vector<double> vs(n), ws(n);
    for (std::size_t i = 0; i < n; ++i) {
        vs[i] = v[idx[i]];
        ws[i] = w[idx[i]];
    }
    return {std::move(vs), std::move(ws)};
}

py::dict npnormllw(py::array_t<double> data_in, py::array_t<double> weights_in,
                   py::array_t<double> mu0fixed, py::array_t<double> pi0fixed,
                   double beta, double h, py::array_t<double> initpt,
                   py::array_t<double> initpr,
                   py::array_t<double> gridpoints, double tol, long maxit,
                   long verbose) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? npfc::grid::gridpoints_npnorm(data, weights,
                                                                   beta, 100)
                                   : gridpoints_v;
    auto init =
        npfc::grid::initial_npnorm(data, weights, beta, initpt_v, initpr_v);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormLLW>(data, weights, mu0fixed_v,
                                               pi0fixed_v, beta, h),
        mu0fixed_v, pi0fixed_v, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormllw_estpi0(py::array_t<double> data_in,
                          py::array_t<double> weights_in, double beta, double h,
                          double val, py::array_t<double> initpt,
                          py::array_t<double> initpr,
                          py::array_t<double> gridpoints, double tol,
                          long verbose, bool fast, bool relax,
                          double inner_tol) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? npfc::grid::gridpoints_npnorm(data, weights,
                                                                   beta, 100)
                                   : gridpoints_v;
    auto init =
        npfc::grid::initial_npnorm(data, weights, beta, initpt_v, initpr_v);
    std::vector<double> mu0f{0.0}, pi0f{1.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormLLW>(data, weights, mu0f, pi0f, beta,
                                               h),
        mu0f, pi0f, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict npnormcvmw(py::array_t<double> data_in, py::array_t<double> weights_in,
                    py::array_t<double> mu0fixed, py::array_t<double> pi0fixed,
                    double beta, double h, py::array_t<double> initpt,
                    py::array_t<double> initpr,
                    py::array_t<double> gridpoints, double tol, long maxit,
                    long verbose) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? default_grid_extended_w(data, weights, beta)
                                   : gridpoints_v;
    auto init =
        npfc::grid::initial_npnorm(data, weights, beta, initpt_v, initpr_v);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCVMW>(data, weights, mu0fixed_v,
                                                pi0fixed_v, beta, h),
        mu0fixed_v, pi0fixed_v, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormcvmw_estpi0(py::array_t<double> data_in,
                           py::array_t<double> weights_in, double beta,
                           double h, double val, py::array_t<double> initpt,
                           py::array_t<double> initpr,
                           py::array_t<double> gridpoints, double tol,
                           long verbose, bool fast, bool relax,
                           double inner_tol) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? default_grid_extended_w(data, weights, beta)
                                   : gridpoints_v;
    auto init =
        npfc::grid::initial_npnorm(data, weights, beta, initpt_v, initpr_v);
    std::vector<double> mu0f{0.0}, pi0f{0.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCVMW>(data, weights, mu0f, pi0f, beta,
                                                h),
        mu0f, pi0f, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict npnormadw(py::array_t<double> data_in, py::array_t<double> weights_in,
                   py::array_t<double> mu0fixed, py::array_t<double> pi0fixed,
                   double beta, double h, py::array_t<double> initpt,
                   py::array_t<double> initpr,
                   py::array_t<double> gridpoints, double tol, long maxit,
                   long verbose) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? default_grid_extended_w(data, weights, beta)
                                   : gridpoints_v;
    auto init =
        npfc::grid::initial_npnorm(data, weights, beta, initpt_v, initpr_v);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormADW>(data, weights, mu0fixed_v,
                                               pi0fixed_v, beta, h),
        mu0fixed_v, pi0fixed_v, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormadw_estpi0(py::array_t<double> data_in,
                          py::array_t<double> weights_in, double beta, double h,
                          double val, py::array_t<double> initpt,
                          py::array_t<double> initpr,
                          py::array_t<double> gridpoints, double tol,
                          long verbose, bool fast, bool relax,
                          double inner_tol) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    std::vector<double> grid = gridpoints_v.empty()
                                   ? default_grid_extended_w(data, weights, beta)
                                   : gridpoints_v;
    auto init =
        npfc::grid::initial_npnorm(data, weights, beta, initpt_v, initpr_v);
    std::vector<double> mu0f{0.0}, pi0f{0.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormADW>(data, weights, mu0f, pi0f, beta,
                                               h),
        mu0f, pi0f, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict nptllw(py::array_t<double> data_in, py::array_t<double> weights_in,
                py::array_t<double> mu0fixed, py::array_t<double> pi0fixed,
                double beta, double h, py::array_t<double> initpt,
                py::array_t<double> initpr, py::array_t<double> gridpoints,
                double tol, long maxit, long verbose) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> mu0fixed_v = vec_from(mu0fixed);
    const std::vector<double> pi0fixed_v = vec_from(pi0fixed);
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    // PIT to z-space: vz = qnorm(pt(data, df = beta)). Identity when beta = inf
    // (the grid/init construction only; the family kernel is unchanged).
    std::vector<double> vz(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        vz[i] = npfc::stats::qnorm(npfc::stats::pt(data[i], beta, true));
    std::vector<double> grid;
    if (gridpoints_v.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(vz, weights, 1.0, 100);
        grid = g;
        for (double& z : grid)
            z = npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta);
    } else {
        grid = gridpoints_v;
    }
    std::vector<double> initpt_z;
    for (double x : initpt_v)
        initpt_z.push_back(npfc::stats::qnorm(npfc::stats::pt(x, beta, true)));
    auto init =
        npfc::grid::initial_npnorm(vz, weights, 1.0, initpt_z, initpr_v);
    std::vector<double> initpt_t;
    for (double z : std::get<1>(init))
        initpt_t.push_back(
            npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta));
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpTLLW>(data, weights, mu0fixed_v,
                                            pi0fixed_v, beta, h),
        mu0fixed_v, pi0fixed_v, std::move(initpt_t), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict nptllw_estpi0(py::array_t<double> data_in,
                       py::array_t<double> weights_in, double beta, double h,
                       double val, py::array_t<double> initpt,
                       py::array_t<double> initpr,
                       py::array_t<double> gridpoints, double tol, long verbose,
                       bool fast, bool relax, double inner_tol) {
    auto vw = sort_centers_weights(vec_from(data_in), vec_from(weights_in));
    const std::vector<double>& data = vw.first;
    const std::vector<double>& weights = vw.second;
    const std::vector<double> initpt_v = vec_from(initpt);
    const std::vector<double> initpr_v = vec_from(initpr);
    const std::vector<double> gridpoints_v = vec_from(gridpoints);
    // PIT to z-space (same transform as `nptllw`).
    std::vector<double> vz(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        vz[i] = npfc::stats::qnorm(npfc::stats::pt(data[i], beta, true));
    std::vector<double> grid;
    if (gridpoints_v.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(vz, weights, 1.0, 100);
        grid = g;
        for (double& z : grid)
            z = npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta);
    } else {
        grid = gridpoints_v;
    }
    std::vector<double> initpt_z;
    for (double x : initpt_v)
        initpt_z.push_back(npfc::stats::qnorm(npfc::stats::pt(x, beta, true)));
    auto init =
        npfc::grid::initial_npnorm(vz, weights, 1.0, initpt_z, initpr_v);
    std::vector<double> initpt_t;
    for (double z : std::get<1>(init))
        initpt_t.push_back(
            npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta));
    std::vector<double> mu0f{0.0}, pi0f{0.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpTLLW>(data, weights, mu0f, pi0f, beta, h),
        mu0f, pi0f, std::move(initpt_t), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

// ---------------------------------------------------------------------------
// posteriormean (R utility.R): for a fitted mixing distribution
// G = sum_j pr_j Delta_{pt_j}, the posterior mean of fun(pt) at each
// observation x_i under the family's kernel:
//
//   out[i] = sum_j K(x_i; pt_j, beta) * pr_j * fpt_j
//            ---------------------------------------
//            sum_j K(x_i; pt_j, beta) * pr_j
//
// where fpt_j = fun(pt_j) and K is the family density (normal /
// non-central-t / one-parameter normal / Poisson pmf). The function `fun`
// is applied only to the k support points (in Python), so the O(n * k)
// work below is pure C++. Accumulation is in ascending j, exactly the
// order of R's base `.rowSums` over the (n x k) column-major kernel matrix,
// so results agree with R to working precision.
py::array_t<double> posteriormean(const std::string& family,
                                  py::array_t<double> x,
                                  py::array_t<double> pt,
                                  py::array_t<double> pr, double beta,
                                  py::array_t<double> fpt) {
    const std::vector<double> x_v = vec_from(x);
    const std::vector<double> pt_v = vec_from(pt);
    const std::vector<double> pr_v = vec_from(pr);
    const std::vector<double> fpt_v = vec_from(fpt);
    if (pt_v.size() != pr_v.size() || pt_v.size() != fpt_v.size() ||
        pt_v.empty())
        throw std::invalid_argument(
            "posteriormean: pt/pr/fpt must be non-empty and equal length");
    const std::size_t n = x_v.size();
    const std::size_t m = pt_v.size();
    std::vector<double> K;
    if (family == "npnorm")
        npfc::kern::kmat_norm(x_v, pt_v, beta, K);
    else if (family == "npt")
        npfc::kern::kmat_t(x_v, pt_v, beta, K);
    else if (family == "npnormc")
        npfc::kern::kmat_normc(x_v, pt_v, beta, K);
    else if (family == "nppois")
        npfc::kern::kmat_pois(x_v, pt_v, K);
    else
        throw std::invalid_argument("posteriormean: unknown family " + family);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        double num = 0.0;
        double den = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            const double c = K[j * n + i] * pr_v[j];
            num += c * fpt_v[j];
            den += c;
        }
        out[i] = num / den;
    }
    return array_from(out);
}

// ---------------------------------------------------------------------------
// npnorm2Dll (R `computemixdist.npnorm2Dll` + `npnorm2Dll_`)
//
// The bivariate-normal mixing family under the maximum likelihood. Unlike the
// one-dimensional families, the support points are 2-vectors and the
// new-support-point search is a box-constrained L-BFGS-B run per grid cell,
// so this entry does NOT go through `MixSolver`; it orchestrates the
// standalone `npfc::fam::NpNorm2D` solver, building the default 2D grid and
// the (marginal) initial mixing distribution exactly as the R wrapper does:
// each marginal column gets its own `gridpoints.npnorm` / `initial.npnorm`
// (driven by the covariance's diagonal), then the two are combined into a
// tensor-product 2D grid / init mix. `data` is n x 2 (row-major).
py::dict npnorm2dll(py::array_t<double> data, py::array_t<double> mu0fixed,
                    py::array_t<double> pi0fixed, py::array_t<double> beta,
                    py::array_t<double> initpt, py::array_t<double> initpr,
                    py::array_t<double> gridpoints, double tol, long maxit,
                    long verbose) {
    const std::vector<double> flat = vec_from(data);
    const Eigen::Index n = static_cast<Eigen::Index>(flat.size() / 2);
    Eigen::MatrixXd D(n, 2);
    for (Eigen::Index i = 0; i < n; ++i) {
        D(i, 0) = flat[2 * static_cast<std::size_t>(i)];
        D(i, 1) = flat[2 * static_cast<std::size_t>(i) + 1];
    }

    const std::vector<double> mu0f_v = vec_from(mu0fixed);
    const Eigen::Index nf = static_cast<Eigen::Index>(mu0f_v.size() / 2);
    Eigen::MatrixXd mu0fixed_m(nf, 2);
    for (Eigen::Index i = 0; i < nf; ++i) {
        mu0fixed_m(i, 0) = mu0f_v[2 * static_cast<std::size_t>(i)];
        mu0fixed_m(i, 1) = mu0f_v[2 * static_cast<std::size_t>(i) + 1];
    }
    const std::vector<double> pi0f_v = vec_from(pi0fixed);
    Eigen::VectorXd pi0fixed_m(nf);
    for (Eigen::Index i = 0; i < nf; ++i)
        pi0fixed_m[i] = pi0f_v[static_cast<std::size_t>(i)];

    const std::vector<double> beta_v = vec_from(beta);
    Eigen::MatrixXd beta_m(2, 2);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            beta_m(r, c) = beta_v[r * 2 + c];

    // Marginal columns + unit weights (R: `v1 = npnorm(v[, 1])` → $v, $w = 1).
    std::vector<double> col1(n), col2(n), w(n, 1.0);
    for (Eigen::Index i = 0; i < n; ++i) {
        col1[static_cast<std::size_t>(i)] = D(i, 0);
        col2[static_cast<std::size_t>(i)] = D(i, 1);
    }

    // Default 2D grid: R's `cbind(sort(rep(g1, LLL)), sort(rep(g2, LLL)))`.
    Eigen::MatrixXd grid_m;
    const std::vector<double> gp_v = vec_from(gridpoints);
    if (gp_v.size() >= 4) {
        const Eigen::Index gm = static_cast<Eigen::Index>(gp_v.size() / 2);
        grid_m.resize(gm, 2);
        for (Eigen::Index i = 0; i < gm; ++i) {
            grid_m(i, 0) = gp_v[2 * static_cast<std::size_t>(i)];
            grid_m(i, 1) = gp_v[2 * static_cast<std::size_t>(i) + 1];
        }
    } else {
        const std::vector<double> g1 =
            npfc::grid::gridpoints_npnorm(col1, w, beta_m(0, 0), 100);
        const std::vector<double> g2 =
            npfc::grid::gridpoints_npnorm(col2, w, beta_m(1, 1), 100);
        const std::size_t LLL = std::max(g1.size(), g2.size());
        auto pad_sort = [](const std::vector<double>& g, std::size_t L) {
            std::vector<double> out(L);
            for (std::size_t t = 0; t < L; ++t)
                out[t] = g[t % g.size()];
            std::sort(out.begin(), out.end());
            return out;
        };
        const std::vector<double> g1s = pad_sort(g1, LLL);
        const std::vector<double> g2s = pad_sort(g2, LLL);
        grid_m.resize(static_cast<Eigen::Index>(LLL), 2);
        for (std::size_t i = 0; i < LLL; ++i) {
            grid_m(static_cast<Eigen::Index>(i), 0) = g1s[i];
            grid_m(static_cast<Eigen::Index>(i), 1) = g2s[i];
        }
    }

    // Initial mixing distribution. If the caller supplied a 2D `mix` (initpt
    // non-empty) use it directly; otherwise build it from the two marginals
    // (tensor product), mirroring R's `initial.npnorm(v1, beta[1,1], ...)` /
    // `initial.npnorm(v2, beta[2,2], ...)`.
    Eigen::MatrixXd initpt_m;
    Eigen::VectorXd initpr_m;
    const std::vector<double> ipt_v = vec_from(initpt);
    if (ipt_v.size() >= 4) {
        const Eigen::Index pm = static_cast<Eigen::Index>(ipt_v.size() / 2);
        initpt_m.resize(pm, 2);
        for (Eigen::Index i = 0; i < pm; ++i) {
            initpt_m(i, 0) = ipt_v[2 * static_cast<std::size_t>(i)];
            initpt_m(i, 1) = ipt_v[2 * static_cast<std::size_t>(i) + 1];
        }
        const std::vector<double> pr_v = vec_from(initpr);
        initpr_m.resize(static_cast<Eigen::Index>(pr_v.size()));
        for (std::size_t i = 0; i < pr_v.size(); ++i)
            initpr_m[static_cast<Eigen::Index>(i)] = pr_v[i];
    } else {
        auto r1 = npfc::grid::initial_npnorm(col1, w, beta_m(0, 0), {}, {});
        auto r2 = npfc::grid::initial_npnorm(col2, w, beta_m(1, 1), {}, {});
        const std::vector<double>& a = std::get<1>(r1);
        const std::vector<double>& pa = std::get<2>(r1);
        const std::vector<double>& b = std::get<1>(r2);
        const std::vector<double>& pb = std::get<2>(r2);
        const std::size_t len1 = a.size(), len2 = b.size();
        initpt_m.resize(static_cast<Eigen::Index>(len1 * len2), 2);
        initpr_m.resize(static_cast<Eigen::Index>(len1 * len2));
        for (std::size_t k = 0; k < len1 * len2; ++k) {
            const std::size_t i = k % len1;  // index into marginal 1
            const std::size_t j = k / len1;  // index into marginal 2
            initpt_m(static_cast<Eigen::Index>(k), 0) = a[i];
            initpt_m(static_cast<Eigen::Index>(k), 1) = b[j];
            initpr_m[static_cast<Eigen::Index>(k)] = pa[i] * pb[j];
        }
    }

    npfc::fam::NpNorm2D solver(D, mu0fixed_m, pi0fixed_m, beta_m, initpt_m,
                               initpr_m, grid_m, static_cast<int>(verbose));
    solver.computemixdist(tol, maxit);
    const npfc::fam::NpNorm2D::Ans a = solver.get_ans();

    py::gil_scoped_acquire _acquire;
    py::dict d;
    py::list ptlist;
    for (const auto& row : a.pt) {
        py::list r2;
        r2.append(row[0]);
        r2.append(row[1]);
        ptlist.append(r2);
    }
    d["pt"] = ptlist;
    d["pr"] = py::cast(a.pr);
    py::list betalist;
    for (const auto& row : a.beta) {
        py::list r2;
        r2.append(row[0]);
        r2.append(row[1]);
        betalist.append(r2);
    }
    d["beta"] = betalist;
    d["family"] = a.family;
    d["min_gradient"] = a.min_gradient;
    d["ll"] = a.ll;
    d["flag"] = a.flag;
    d["iter"] = static_cast<long>(a.iter);
    d["convergence"] = a.convergence;
    return d;
}

// ---------------------------------------------------------------------------
// Public ND normal kernels (R `dnpnormND` / `dnormNDarray_`). `x` is (m, dim),
// `mu0` is (k, dim), `pi0` length k, `Sigma` the (dim, dim) covariance. The
// primary use is the public `dnpnormND` distribution function; they also
// serve as the bit-for-bit diagnostic against the R reference kernel.
// ---------------------------------------------------------------------------
static Eigen::MatrixXd mat_from_2d(py::array_t<double> a, Eigen::Index r,
                                   Eigen::Index c) {
    const auto info = a.request();
    const double* p = static_cast<const double*>(info.ptr);
    Eigen::MatrixXd m(r, c);
    for (Eigen::Index i = 0; i < r; ++i)
        for (Eigen::Index j = 0; j < c; ++j)
            m(i, j) = p[static_cast<std::size_t>(i) * c + static_cast<std::size_t>(j)];
    return m;
}

static py::array_t<double> dnpnormND_py(py::array_t<double> x,
                                        py::array_t<double> mu0,
                                        py::array_t<double> pi0,
                                        py::array_t<double> Sigma, bool lg) {
    const auto xa = x.request();
    const auto ma = mu0.request();
    const auto pa = pi0.request();
    const Eigen::Index m = xa.shape[0];
    const Eigen::Index dim = xa.shape[1];
    const Eigen::Index k = ma.shape[0];
    const Eigen::MatrixXd X = mat_from_2d(x, m, dim);
    const Eigen::MatrixXd M = mat_from_2d(mu0, k, dim);
    const Eigen::MatrixXd S = mat_from_2d(Sigma, dim, dim);
    Eigen::VectorXd pi(k);
    const double* pp = static_cast<const double*>(pa.ptr);
    for (Eigen::Index j = 0; j < k; ++j)
        pi[j] = pp[static_cast<std::size_t>(j)];
    const Eigen::VectorXd out =
        npfc::kern::dnpnormND(X, M, pi, S, lg).reshaped();
    py::array_t<double> res(out.size());
    std::copy(out.data(), out.data() + out.size(), res.mutable_data());
    return res;
}

static py::array_t<double> dnormNDarray_py(py::array_t<double> x,
                                           py::array_t<double> mu0,
                                           py::array_t<double> Sigma, bool lg) {
    const auto xa = x.request();
    const auto ma = mu0.request();
    const Eigen::Index m = xa.shape[0];
    const Eigen::Index dim = xa.shape[1];
    const Eigen::Index k = ma.shape[0];
    const Eigen::MatrixXd X = mat_from_2d(x, m, dim);
    const Eigen::MatrixXd M = mat_from_2d(mu0, k, dim);
    const Eigen::MatrixXd S = mat_from_2d(Sigma, dim, dim);
    const Eigen::MatrixXd out = npfc::kern::dnormNDarray(X, M, S, lg);
    py::array_t<double> res({m, k});
    const double* op = out.data();
    double* rp = res.mutable_data();
    for (Eigen::Index i = 0; i < m; ++i)
        for (Eigen::Index j = 0; j < k; ++j)
            rp[static_cast<std::size_t>(i) * k + static_cast<std::size_t>(j)] =
                op[static_cast<std::size_t>(j) * m + static_cast<std::size_t>(i)];
    return res;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "npfixedcomppy C++/Eigen core (pybind11)";
    // A callable, matching the PyO3 `version()` entry point the Python
    // front-end (npfc.py) expects.
    m.def("version", [] { return std::string("0.2.1"); });

    const auto guard = py::call_guard<py::gil_scoped_release>();
    const auto kw = py::kw_only();

    m.def("npnormll", &npnormll, py::arg("data"), py::arg("mu0fixed"),
          py::arg("pi0fixed"), py::arg("beta"), py::arg("initpt"),
          py::arg("initpr"), py::arg("gridpoints"), py::arg("tol"),
          py::arg("maxit"), py::arg("verbose"), guard, kw);
    m.def("npnormcvm", &npnormcvm, py::arg("data"), py::arg("mu0fixed"),
          py::arg("pi0fixed"), py::arg("beta"), py::arg("initpt"),
          py::arg("initpr"), py::arg("gridpoints"), py::arg("tol"),
          py::arg("maxit"), py::arg("verbose"), guard, kw);
    m.def("npnormad", &npnormad, py::arg("data"), py::arg("mu0fixed"),
          py::arg("pi0fixed"), py::arg("beta"), py::arg("initpt"),
          py::arg("initpr"), py::arg("gridpoints"), py::arg("tol"),
          py::arg("maxit"), py::arg("verbose"), guard, kw);
    m.def("nptll", &nptll, py::arg("data"), py::arg("mu0fixed"),
          py::arg("pi0fixed"), py::arg("beta"), py::arg("initpt"),
          py::arg("initpr"), py::arg("gridpoints"), py::arg("tol"),
          py::arg("maxit"), py::arg("verbose"), guard, kw);
    m.def("npnormcll", &npnormcll, py::arg("data"), py::arg("mu0fixed"),
          py::arg("pi0fixed"), py::arg("beta"), py::arg("initpt"),
          py::arg("initpr"), py::arg("gridpoints"), py::arg("tol"),
          py::arg("maxit"), py::arg("verbose"), guard, kw);
    m.def("nppoisll", &nppoisll, py::arg("data"), py::arg("mu0fixed"),
          py::arg("pi0fixed"), py::arg("beta"), py::arg("initpt"),
          py::arg("initpr"), py::arg("gridpoints"), py::arg("tol"),
          py::arg("maxit"), py::arg("verbose"), guard, kw);

    m.def("npnormll_estpi0", &npnormll_estpi0, py::arg("data"), py::arg("beta"),
          py::arg("val"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("verbose"),
          py::arg("fast"), py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("npnormcvm_estpi0", &npnormcvm_estpi0, py::arg("data"), py::arg("beta"),
          py::arg("val"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("verbose"),
          py::arg("fast"), py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("npnormad_estpi0", &npnormad_estpi0, py::arg("data"), py::arg("beta"),
          py::arg("val"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("verbose"),
          py::arg("fast"), py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("nptll_estpi0", &nptll_estpi0, py::arg("data"), py::arg("beta"),
          py::arg("val"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("verbose"),
          py::arg("fast"), py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("npnormcll_estpi0", &npnormcll_estpi0, py::arg("data"), py::arg("beta"),
          py::arg("val"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("verbose"),
          py::arg("fast"), py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("nppoisll_estpi0", &nppoisll_estpi0, py::arg("data"), py::arg("beta"),
          py::arg("val"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("verbose"),
          py::arg("fast"), py::arg("relax"), py::arg("inner_tol"), guard, kw);

    // The four binned ("...w") computemixdist entries: the caller pre-bins
    // the observations into (bin centres `data`, bin counts `weights`) and
    // passes `h = 10^order`.
    m.def("npnormllw", &npnormllw, py::arg("data"), py::arg("weights"),
          py::arg("mu0fixed"), py::arg("pi0fixed"), py::arg("beta"),
          py::arg("h"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("maxit"),
          py::arg("verbose"), guard, kw);
    m.def("npnormcvmw", &npnormcvmw, py::arg("data"), py::arg("weights"),
          py::arg("mu0fixed"), py::arg("pi0fixed"), py::arg("beta"),
          py::arg("h"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("maxit"),
          py::arg("verbose"), guard, kw);
    m.def("npnormadw", &npnormadw, py::arg("data"), py::arg("weights"),
          py::arg("mu0fixed"), py::arg("pi0fixed"), py::arg("beta"),
          py::arg("h"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("maxit"),
          py::arg("verbose"), guard, kw);
    m.def("nptllw", &nptllw, py::arg("data"), py::arg("weights"),
          py::arg("mu0fixed"), py::arg("pi0fixed"), py::arg("beta"),
          py::arg("h"), py::arg("initpt"), py::arg("initpr"),
          py::arg("gridpoints"), py::arg("tol"), py::arg("maxit"),
          py::arg("verbose"), guard, kw);

    // The four binned ("...w") estpi0 entries.
    m.def("npnormllw_estpi0", &npnormllw_estpi0, py::arg("data"),
          py::arg("weights"), py::arg("beta"), py::arg("h"), py::arg("val"),
          py::arg("initpt"), py::arg("initpr"), py::arg("gridpoints"),
          py::arg("tol"), py::arg("verbose"), py::arg("fast"),
          py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("npnormcvmw_estpi0", &npnormcvmw_estpi0, py::arg("data"),
          py::arg("weights"), py::arg("beta"), py::arg("h"), py::arg("val"),
          py::arg("initpt"), py::arg("initpr"), py::arg("gridpoints"),
          py::arg("tol"), py::arg("verbose"), py::arg("fast"),
          py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("npnormadw_estpi0", &npnormadw_estpi0, py::arg("data"),
          py::arg("weights"), py::arg("beta"), py::arg("h"), py::arg("val"),
          py::arg("initpt"), py::arg("initpr"), py::arg("gridpoints"),
          py::arg("tol"), py::arg("verbose"), py::arg("fast"),
          py::arg("relax"), py::arg("inner_tol"), guard, kw);
    m.def("nptllw_estpi0", &nptllw_estpi0, py::arg("data"),
          py::arg("weights"), py::arg("beta"), py::arg("h"), py::arg("val"),
          py::arg("initpt"), py::arg("initpr"), py::arg("gridpoints"),
          py::arg("tol"), py::arg("verbose"), py::arg("fast"),
          py::arg("relax"), py::arg("inner_tol"), guard, kw);

    // The bivariate-normal mixing family (R `npnorm2Dll`). `data` is n x 2
    // (row-major), `beta` the 2 x 2 covariance (row-major), and the
    // remaining args match the one-dimensional entries.
    m.def("npnorm2Dll", &npnorm2dll, py::arg("data"), py::arg("mu0fixed"),
          py::arg("pi0fixed"), py::arg("beta"), py::arg("initpt"),
          py::arg("initpr"), py::arg("gridpoints"), py::arg("tol"),
          py::arg("maxit"), py::arg("verbose"), guard, kw);

    // Public ND normal kernels (R `dnpnormND` / `dnormNDarray_`).
    m.def("dnpnormND", &dnpnormND_py, py::arg("x"), py::arg("mu0"),
          py::arg("pi0"), py::arg("sigma"), py::arg("lg"));
    m.def("dnormNDarray", &dnormNDarray_py, py::arg("x"), py::arg("mu0"),
          py::arg("sigma"), py::arg("lg"));

    // Public 1D density / cdf wrappers (R `dnpnorm`/`pnpnorm`/`dnpnormc`/
    // `dnpt`/`pnpt`/`dnpdiscnorm`/`pnpdiscnorm`/`dnppois`/`pnppois`/
    // `dnpdisct`), each mirroring R's lg=FALSE (sum) and lg=TRUE
    // (sequential logspaceadd chain) paths.
    m.def("dnpnorm",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double stdev, bool lg) {
              return npfc::kern::dnpnorm_(x, mu0, pi0, stdev, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"),
          py::arg("stdev") = 1.0, py::arg("lg") = false);
    m.def("pnpnorm",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double stdev, bool lt, bool lg) {
              return npfc::kern::pnpnorm_(x, mu0, pi0, stdev, lt, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"),
          py::arg("stdev") = 1.0, py::arg("lt") = true, py::arg("lg") = false);
    m.def("dnpnormc",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double n, bool lg) {
              return npfc::kern::dnpnormc_(x, mu0, pi0, n, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("n"),
          py::arg("lg") = false);
    m.def("dnpt",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double df, bool lg) {
              return npfc::kern::dnpt_(x, mu0, pi0, df, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("df"),
          py::arg("lg") = false);
    m.def("pnpt",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double df, bool lt, bool lg) {
              return npfc::kern::pnpt_(x, mu0, pi0, df, lt, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("df"),
          py::arg("lt") = true, py::arg("lg") = false);
    m.def("dnpdiscnorm",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double stdev, double h, bool lg) {
              return npfc::kern::dnpdiscnorm_(x, mu0, pi0, stdev, h, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("stdev"),
          py::arg("h"), py::arg("lg") = false);
    m.def("pnpdiscnorm",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double stdev, double h, bool lt,
             bool lg) {
              return npfc::kern::pnpdiscnorm_(x, mu0, pi0, stdev, h, lt, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("stdev"),
          py::arg("h"), py::arg("lt") = true, py::arg("lg") = false);
    m.def("dnppois",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double stdev, bool lg) {
              return npfc::kern::dnppois_(x, mu0, pi0, stdev, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("stdev") = 1.0,
          py::arg("lg") = false);
    m.def("pnppois",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, bool lt, bool lg) {
              return npfc::kern::pnppois_(x, mu0, pi0, lt, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("lt") = true,
          py::arg("lg") = false);
    m.def("dnpdisct",
          [](std::vector<double> x, std::vector<double> mu0,
             std::vector<double> pi0, double df, double h, bool lg) {
              return npfc::kern::dnpdisct_(x, mu0, pi0, df, h, lg);
          },
          py::arg("x"), py::arg("mu0"), py::arg("pi0"), py::arg("df"),
          py::arg("h"), py::arg("lg") = false);

    m.def("posteriormean", &posteriormean, py::arg("family"),
          py::arg("x"), py::arg("pt"), py::arg("pr"), py::arg("beta"),
          py::arg("fpt"), guard, kw);

    m.def("correlationmatrixcpp",
          [](const std::vector<std::vector<double>>& G, double tau, double tol) {
              const std::size_t p = G.size();
              if (p == 0)
                  throw std::invalid_argument("correlationmatrixcpp: empty matrix");
              for (const auto& row : G)
                  if (row.size() != p)
                      throw std::invalid_argument(
                          "correlationmatrixcpp: rows must all have length p");
              Eigen::MatrixXd A(static_cast<Eigen::Index>(p),
                                static_cast<Eigen::Index>(p));
              for (std::size_t i = 0; i < p; ++i)
                  for (std::size_t j = 0; j < p; ++j)
                      A(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                          G[i][j];
              Eigen::MatrixXd X = npfc::corr::correlationmatrixcpp(A, tau, tol);
              std::vector<std::vector<double>> out(p);
              for (std::size_t i = 0; i < p; ++i) {
                  out[i].reserve(p);
                  for (std::size_t j = 0; j < p; ++j)
                      out[i].push_back(
                          X(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)));
              }
              return out;
          },
          py::arg("G"), py::arg("tau") = 0.0, py::arg("tol") = 1e-6, guard, kw);
}
