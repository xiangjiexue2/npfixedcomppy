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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "npfc_engine.h"
#include "npfc_families.h"
#include "npfc_grid.h"

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

py::dict npnormll(const std::vector<double>& data,
                  const std::vector<double>& mu0fixed,
                  const std::vector<double>& pi0fixed, double beta,
                  const std::vector<double>& initpt,
                  const std::vector<double>& initpr,
                  const std::vector<double>& gridpoints, double tol,
                  long maxit, long verbose) {
    const std::vector<double> w = ones(data.size());
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt, initpr);
    std::vector<double> grid =
        gridpoints.empty() ? npfc::grid::gridpoints_npnorm(data, w, beta, 100)
                           : gridpoints;
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormLL>(data, mu0fixed, pi0fixed, beta),
        mu0fixed, pi0fixed, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormll_estpi0(const std::vector<double>& data, double beta,
                         double val, const std::vector<double>& initpt,
                         const std::vector<double>& initpr,
                         const std::vector<double>& gridpoints, double tol,
                         long verbose, bool fast, bool relax,
                         double inner_tol) {
    const std::vector<double> w = ones(data.size());
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt, initpr);
    std::vector<double> grid =
        gridpoints.empty() ? npfc::grid::gridpoints_npnorm(data, w, beta, 100)
                           : gridpoints;
    std::vector<double> mu0f{0.0}, pi0f{1.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormLL>(data, mu0f, pi0f, beta),
        mu0f, pi0f, std::get<1>(init),
        std::get<2>(init), grid, static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict npnormcvm(const std::vector<double>& data_in,
                   const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed, double beta,
                   const std::vector<double>& initpt,
                   const std::vector<double>& initpr,
                   const std::vector<double>& gridpoints, double tol,
                   long maxit, long verbose) {
    std::vector<double> data = data_in;
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints.empty() ? default_grid_extended(data, beta) : gridpoints;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt, initpr);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCVM>(data, mu0fixed, pi0fixed, beta),
        mu0fixed, pi0fixed, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormcvm_estpi0(const std::vector<double>& data_in, double beta,
                          double val, const std::vector<double>& initpt,
                          const std::vector<double>& initpr,
                          const std::vector<double>& gridpoints, double tol,
                          long verbose, bool fast, bool relax,
                          double inner_tol) {
    std::vector<double> data = data_in;
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints.empty() ? default_grid_extended(data, beta) : gridpoints;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt, initpr);
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

py::dict npnormad(const std::vector<double>& data_in,
                  const std::vector<double>& mu0fixed,
                  const std::vector<double>& pi0fixed, double beta,
                  const std::vector<double>& initpt,
                  const std::vector<double>& initpr,
                  const std::vector<double>& gridpoints, double tol,
                  long maxit, long verbose) {
    std::vector<double> data = data_in;
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints.empty() ? default_grid_extended(data, beta) : gridpoints;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt, initpr);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormAD>(data, mu0fixed, pi0fixed, beta),
        mu0fixed, pi0fixed, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormad_estpi0(const std::vector<double>& data_in, double beta,
                         double val, const std::vector<double>& initpt,
                         const std::vector<double>& initpr,
                         const std::vector<double>& gridpoints, double tol,
                         long verbose, bool fast, bool relax,
                         double inner_tol) {
    std::vector<double> data = data_in;
    std::sort(data.begin(), data.end(), [](double a, double b) { return a < b; });
    const std::vector<double> w = ones(data.size());
    std::vector<double> grid =
        gridpoints.empty() ? default_grid_extended(data, beta) : gridpoints;
    auto init = npfc::grid::initial_npnorm(data, w, beta, initpt, initpr);
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

py::dict npnormcll(const std::vector<double>& data,
                   const std::vector<double>& mu0fixed,
                   const std::vector<double>& pi0fixed, double beta,
                   const std::vector<double>& initpt,
                   const std::vector<double>& initpr,
                   const std::vector<double>& gridpoints, double tol,
                   long maxit, long verbose) {
    if (!(beta > 3.0))
        throw py::value_error(
            "npnormcll requires beta > 3 (the number of observations)");
    const double beta_n = 1.0 / std::sqrt(beta - 3.0);
    std::vector<double> v_atanh(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        v_atanh[i] = atanh_r(data[i]);
    const std::vector<double> w = ones(v_atanh.size());
    std::vector<double> grid;
    if (gridpoints.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v_atanh, w, beta_n, 100);
        grid = g;
        for (double& z : grid)
            z = tanhv(z);
    } else {
        grid = gridpoints;
    }
    std::vector<double> initpt_t;
    for (double x : initpt)
        initpt_t.push_back(atanh_r(x));
    auto init = npfc::grid::initial_npnorm(v_atanh, w, beta_n, initpt_t, initpr);
    std::vector<double> initpt_n;
    for (double z : std::get<1>(init))
        initpt_n.push_back(tanhv(z));
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCLL>(data, mu0fixed, pi0fixed, beta),
        mu0fixed, pi0fixed, std::move(initpt_n), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict npnormcll_estpi0(const std::vector<double>& data, double beta,
                          double val, const std::vector<double>& initpt,
                          const std::vector<double>& initpr,
                          const std::vector<double>& gridpoints, double tol,
                          long verbose, bool fast, bool relax,
                          double inner_tol) {
    if (!(beta > 3.0))
        throw py::value_error(
            "npnormcll requires beta > 3 (the number of observations)");
    const double beta_n = 1.0 / std::sqrt(beta - 3.0);
    std::vector<double> v_atanh(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        v_atanh[i] = atanh_r(data[i]);
    const std::vector<double> w = ones(v_atanh.size());
    std::vector<double> grid;
    if (gridpoints.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v_atanh, w, beta_n, 100);
        grid = g;
        for (double& z : grid)
            z = tanhv(z);
    } else {
        grid = gridpoints;
    }
    std::vector<double> initpt_t;
    for (double x : initpt)
        initpt_t.push_back(atanh_r(x));
    auto init = npfc::grid::initial_npnorm(v_atanh, w, beta_n, initpt_t, initpr);
    std::vector<double> initpt_n;
    for (double z : std::get<1>(init))
        initpt_n.push_back(tanhv(z));
    std::vector<double> mu0f{0.0}, pi0f{1.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpNormCLL>(data, mu0f, pi0f, beta),
        mu0f, pi0f,
        std::move(initpt_n), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
}

py::dict nppoisll(const std::vector<double>& data,
                  const std::vector<double>& mu0fixed,
                  const std::vector<double>& pi0fixed, double beta,
                  const std::vector<double>& initpt,
                  const std::vector<double>& initpr,
                  const std::vector<double>& gridpoints, double tol,
                  long maxit, long verbose) {
    const std::vector<std::vector<double>> vw = aggregate_counts(data);
    const std::vector<double>& v = vw[0];
    const std::vector<double>& w = vw[1];
    std::vector<double> grid =
        gridpoints.empty() ? npfc::grid::gridpoints_nppois(v, beta, 100)
                           : gridpoints;
    auto init = npfc::grid::initial_nppois(v, w, beta, initpt, initpr);
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpPoisLL>(v, w, mu0fixed, pi0fixed, beta),
        mu0fixed, pi0fixed, std::get<1>(init), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict nppoisll_estpi0(const std::vector<double>& data, double beta,
                         double val, const std::vector<double>& initpt,
                         const std::vector<double>& initpr,
                         const std::vector<double>& gridpoints, double tol,
                         long verbose, bool fast, bool relax,
                         double inner_tol) {
    const std::vector<std::vector<double>> vw = aggregate_counts(data);
    const std::vector<double>& v = vw[0];
    const std::vector<double>& w = vw[1];
    std::vector<double> grid =
        gridpoints.empty() ? npfc::grid::gridpoints_nppois(v, beta, 100)
                           : gridpoints;
    auto init = npfc::grid::initial_nppois(v, w, beta, initpt, initpr);
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

py::dict nptll(const std::vector<double>& data,
               const std::vector<double>& mu0fixed,
               const std::vector<double>& pi0fixed, double beta,
               const std::vector<double>& initpt,
               const std::vector<double>& initpr,
               const std::vector<double>& gridpoints, double tol,
               long maxit, long verbose) {
    // PIT to z-space: v1 = qnorm(pt(v, df = beta)). Identity when beta = Inf.
    std::vector<double> v1(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        v1[i] = npfc::stats::qnorm(npfc::stats::pt(data[i], beta, true));
    const std::vector<double> w = ones(v1.size());
    std::vector<double> grid;
    if (gridpoints.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v1, w, 1.0, 100);
        grid = g;
        for (double& z : grid)
            z = npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta);
    } else {
        grid = gridpoints;
    }
    std::vector<double> initpt_z;
    for (double x : initpt)
        initpt_z.push_back(npfc::stats::qnorm(npfc::stats::pt(x, beta, true)));
    auto init =
        npfc::grid::initial_npnorm(v1, w, 1.0, initpt_z, initpr);
    std::vector<double> initpt_t;
    for (double z : std::get<1>(init))
        initpt_t.push_back(
            npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta));
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpTLL>(data, mu0fixed, pi0fixed, beta),
        mu0fixed, pi0fixed, std::move(initpt_t), std::get<2>(init), grid,
        static_cast<int>(verbose));
    solver->computemixdist(tol, maxit);
    return result_to_dict(solver->finish());
}

py::dict nptll_estpi0(const std::vector<double>& data, double beta,
                      double val, const std::vector<double>& initpt,
                      const std::vector<double>& initpr,
                      const std::vector<double>& gridpoints, double tol,
                      long verbose, bool fast, bool relax,
                      double inner_tol) {
    // PIT to z-space (same transform as `nptll`).
    std::vector<double> v1(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        v1[i] = npfc::stats::qnorm(npfc::stats::pt(data[i], beta, true));
    const std::vector<double> w = ones(v1.size());
    std::vector<double> grid;
    if (gridpoints.empty()) {
        const std::vector<double> g =
            npfc::grid::gridpoints_npnorm(v1, w, 1.0, 100);
        grid = g;
        for (double& z : grid)
            z = npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta);
    } else {
        grid = gridpoints;
    }
    std::vector<double> initpt_z;
    for (double x : initpt)
        initpt_z.push_back(npfc::stats::qnorm(npfc::stats::pt(x, beta, true)));
    auto init = npfc::grid::initial_npnorm(v1, w, 1.0, initpt_z, initpr);
    std::vector<double> initpt_t;
    for (double z : std::get<1>(init))
        initpt_t.push_back(
            npfc::stats::qt(npfc::stats::pnorm(z, 0.0, 1.0, true), beta));
    // estpi0nptll_ has no mu0/pi0: the fixed component is the (zero-weight)
    // point mass at 0 whose proportion is estimated by the solver.
    std::vector<double> mu0f{0.0}, pi0f{0.0};
    auto solver = std::make_unique<npfc::MixSolver>(
        std::make_unique<npfc::fam::NpTLL>(data, mu0f, pi0f, beta),
        mu0f, pi0f,
        std::move(initpt_t), std::get<2>(init), grid,
        static_cast<int>(verbose));
    if (fast)
        solver->estpi0_fast(val, tol, inner_tol, relax);
    else
        solver->estpi0(val, tol);
    return result_to_dict(solver->finish());
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
}
