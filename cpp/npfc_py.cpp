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

PYBIND11_MODULE(_core, m) {
    m.doc() = "npfixedcomppy C++/Eigen core (pybind11)";
    // A callable, matching the PyO3 `version()` entry point the Python
    // front-end (npfc.py) expects.
    m.def("version", [] { return std::string("0.1.0-cpp"); });

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
