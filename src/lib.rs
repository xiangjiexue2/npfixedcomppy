// Rust core for npfixedcomppy.
//
// Modules:
//   stats    - normal / t / poisson pdf & cdf (matching R's dnorm/pnorm/dt/pt/dpois)
//   nnls     - partial non-negative least squares (Lawson-Hanson pnnls / pnnlssum / pnnqp)
//   pretty   - exact port of R's R_pretty (histogram breaks)
//   misc     - small helpers (diff, index2num, ...)
//   grid     - initial mixing distribution & grid points (nspmix initial.* / gridpoints.*)
//   engine   - core computemixdist / estpi0 driver
//   families - per-family loss/gradient/mapping implementations (Family trait)
//   npfc_ffi - FFI into the C++/Eigen core (cpp/npfc_core.cpp): SIMD is chosen
//              by the compiler at build time, OpenMP is used when available
mod engine;
mod families;
mod grid;
mod misc;
mod nnls;
mod npfc_ffi;
mod pretty;
mod stats;

use crate::stats::{pnorm, pt, qnorm, qt};
use engine::{MixResult, MixSolver};
use families::{Family, NpNormAD, NpNormCLL, NpNormCVM, NpNormLL, NpPoisLL, NpTLL};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert a `MixResult` (the `get_ans` payload) into the dict that the
/// Python front-end (`npfc._to_npmix`) consumes.
fn result_to_py(py: Python, r: &MixResult) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("pt", r.pt.to_vec())?;
    d.set_item("pr", r.pr.to_vec())?;
    d.set_item("beta", r.beta)?;
    d.set_item("family", &r.family)?;
    d.set_item("min_gradient", r.min_gradient)?;
    d.set_item("ll", r.ll)?;
    d.set_item("flag", &r.flag)?;
    d.set_item("iter", r.iter)?;
    d.set_item("convergence", r.convergence)?;
    Ok(d.into())
}

fn sort_f64(v: &mut Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

fn ones(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// R's `atanh(x) = 0.5 * log((1 + x)/(1 - x))` (for |x| < 1).
fn atanh(x: f64) -> f64 {
    0.5 * ((1.0 + x).ln() - (1.0 - x).ln())
}

/// Stable `tanh(x)` (avoids overflow in the naive exp form).
fn tanhv(x: f64) -> f64 {
    if x >= 0.0 {
        let e2 = (-2.0 * x).exp();
        (1.0 - e2) / (1.0 + e2)
    } else {
        let e2 = (2.0 * x).exp();
        (e2 - 1.0) / (e2 + 1.0)
    }
}

/// R's `table(v)`: group equal (integer-valued) observations and return
/// `(unique values, counts)` sorted ascending.
fn aggregate_counts(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut sorted = data.to_vec();
    sort_f64(&mut sorted);
    let mut v: Vec<f64> = Vec::with_capacity(sorted.len());
    let mut w: Vec<f64> = Vec::with_capacity(sorted.len());
    for x in sorted {
        if let Some(last) = v.last() {
            if *last == x {
                *w.last_mut().unwrap() += 1.0;
                continue;
            }
        }
        v.push(x);
        w.push(1.0);
    }
    (v, w)
}

/// The R wrapper's extended default grid for the CVM/AD families:
/// `c(gridpoints.npnorm(v, beta), v[1] - 3*beta, v[n] + 3*beta)` on sorted `v`.
fn default_grid_extended(v_sorted: &[f64], beta: f64) -> Vec<f64> {
    let w = ones(v_sorted.len());
    let mut g = grid::gridpoints_npnorm(v_sorted, &w, beta, 100);
    g.push(v_sorted[0] - 3.0 * beta);
    g.push(*v_sorted.last().unwrap() + 3.0 * beta);
    g
}

/// `npnormll_` — normal mixing distribution by maximum likelihood.
///
/// Arguments mirror the R exported function `npnormll_(data, mu0fixed,
/// pi0fixed, beta, initpt, initpr, gridpoints, tol, maxit, verbose)`. When
/// `gridpoints` or `initpt`/`initpr` are empty the R-side defaults are
/// reproduced: `gridpoints.npnorm(data, beta)` and `initial.npnorm(data,
/// beta)`.
#[pyfunction]
fn npnormll(
    py: Python,
    data: Vec<f64>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    beta: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    maxit: i64,
    verbose: i32,
) -> PyResult<PyObject> {
    let w = vec![1.0f64; data.len()];
    let (_b, initpt, initpr) = grid::initial_npnorm(&data, &w, Some(beta), &initpt, &initpr);
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_npnorm(&data, &w, beta, 100)
    } else {
        gridpoints
    };
    let fam: Box<dyn Family> =
        Box::new(NpNormLL::new(data, mu0fixed.clone(), pi0fixed.clone(), beta));
    let mut solver = MixSolver::new(fam, mu0fixed, pi0fixed, initpt, initpr, gridpoints, verbose);
    solver.computemixdist(tol, maxit);
    result_to_py(py, &solver.finish())
}

/// `estpi0npnormll_` — estimate the point mass at 0 jointly with the mixing
/// distribution (normal MLE). The fixed component is `(mu0fixed = 0,
/// pi0fixed = 0)` internally, mirroring the C++ entry point.
#[pyfunction]
fn npnormll_estpi0(
    py: Python,
    data: Vec<f64>,
    beta: f64,
    val: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    verbose: i32,
    fast: bool,
    relax: bool,
    inner_tol: f64,
) -> PyResult<PyObject> {
    let w = vec![1.0f64; data.len()];
    let (_b, initpt, initpr) = grid::initial_npnorm(&data, &w, Some(beta), &initpt, &initpr);
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_npnorm(&data, &w, beta, 100)
    } else {
        gridpoints
    };
    let fam: Box<dyn Family> = Box::new(NpNormLL::new(data, vec![0.0], vec![1.0], beta));
    let mut solver =
        MixSolver::new(fam, vec![0.0], vec![1.0], initpt, initpr, gridpoints, verbose);
    if fast {
        solver.estpi0_fast(val, tol, inner_tol, relax);
    } else {
        solver.estpi0(val, tol);
    }
    result_to_py(py, &solver.finish())
}

/// `npnormcvm_` — normal mixing distribution, Cramer-von Mises distance.
/// Mirrors the R `computemixdist.npnormcvm` dispatch: the data are sorted and
/// the default grid is extended by `min - 3*beta` and `max + 3*beta`.
#[pyfunction]
fn npnormcvm(
    py: Python,
    data: Vec<f64>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    beta: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    maxit: i64,
    verbose: i32,
) -> PyResult<PyObject> {
    let mut data = data;
    sort_f64(&mut data);
    let w = ones(data.len());
    let gridpoints = if gridpoints.is_empty() {
        default_grid_extended(&data, beta)
    } else {
        gridpoints
    };
    let (_b, initpt, initpr) = grid::initial_npnorm(&data, &w, Some(beta), &initpt, &initpr);
    let fam: Box<dyn Family> =
        Box::new(NpNormCVM::new(data, mu0fixed.clone(), pi0fixed.clone(), beta));
    let mut solver = MixSolver::new(fam, mu0fixed, pi0fixed, initpt, initpr, gridpoints, verbose);
    solver.computemixdist(tol, maxit);
    result_to_py(py, &solver.finish())
}

/// `estpi0npnormcvm_` — Cramer-von Mises distance with an estimated point
/// mass at 0 (mirrors the R `estpi0.npnormcvm` dispatch).
#[pyfunction]
fn npnormcvm_estpi0(
    py: Python,
    data: Vec<f64>,
    beta: f64,
    val: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    verbose: i32,
    fast: bool,
    relax: bool,
    inner_tol: f64,
) -> PyResult<PyObject> {
    let mut data = data;
    sort_f64(&mut data);
    let w = ones(data.len());
    let gridpoints = if gridpoints.is_empty() {
        default_grid_extended(&data, beta)
    } else {
        gridpoints
    };
    let (_b, initpt, initpr) = grid::initial_npnorm(&data, &w, Some(beta), &initpt, &initpr);
    let fam: Box<dyn Family> = Box::new(NpNormCVM::new(data, vec![0.0], vec![0.0], beta));
    let mut solver =
        MixSolver::new(fam, vec![0.0], vec![0.0], initpt, initpr, gridpoints, verbose);
    if fast {
        solver.estpi0_fast(val, tol, inner_tol, relax);
    } else {
        solver.estpi0(val, tol);
    }
    result_to_py(py, &solver.finish())
}

/// `npnormad_` — normal mixing distribution, Anderson-Darling distance.
/// Mirrors the R `computemixdist.npnormad` dispatch (sorted data, extended
/// default grid).
#[pyfunction]
fn npnormad(
    py: Python,
    data: Vec<f64>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    beta: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    maxit: i64,
    verbose: i32,
) -> PyResult<PyObject> {
    let mut data = data;
    sort_f64(&mut data);
    let w = ones(data.len());
    let gridpoints = if gridpoints.is_empty() {
        default_grid_extended(&data, beta)
    } else {
        gridpoints
    };
    let (_b, initpt, initpr) = grid::initial_npnorm(&data, &w, Some(beta), &initpt, &initpr);
    let fam: Box<dyn Family> =
        Box::new(NpNormAD::new(data, mu0fixed.clone(), pi0fixed.clone(), beta));
    let mut solver = MixSolver::new(fam, mu0fixed, pi0fixed, initpt, initpr, gridpoints, verbose);
    solver.computemixdist(tol, maxit);
    result_to_py(py, &solver.finish())
}

/// `estpi0npnormad_` — Anderson-Darling distance with an estimated point
/// mass at 0 (mirrors the R `estpi0.npnormad` dispatch).
#[pyfunction]
fn npnormad_estpi0(
    py: Python,
    data: Vec<f64>,
    beta: f64,
    val: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    verbose: i32,
    fast: bool,
    relax: bool,
    inner_tol: f64,
) -> PyResult<PyObject> {
    let mut data = data;
    sort_f64(&mut data);
    let w = ones(data.len());
    let gridpoints = if gridpoints.is_empty() {
        default_grid_extended(&data, beta)
    } else {
        gridpoints
    };
    let (_b, initpt, initpr) = grid::initial_npnorm(&data, &w, Some(beta), &initpt, &initpr);
    let fam: Box<dyn Family> = Box::new(NpNormAD::new(data, vec![0.0], vec![0.0], beta));
    let mut solver =
        MixSolver::new(fam, vec![0.0], vec![0.0], initpt, initpr, gridpoints, verbose);
    if fast {
        solver.estpi0_fast(val, tol, inner_tol, relax);
    } else {
        solver.estpi0(val, tol);
    }
    result_to_py(py, &solver.finish())
}

/// `npnormcll_` — one-parameter normal mixing for sample correlation
/// coefficients, maximum likelihood. Mirrors the R `computemixdist.npnormcll`
/// dispatch: the grid and the initial support are built on the `atanh`-
/// transformed data with `beta_n = 1/sqrt(beta - 3)`, the initial support is
/// mapped back with `tanh`, and the family itself is fitted on the raw
/// correlations with the observation count `beta`.
#[pyfunction]
fn npnormcll(
    py: Python,
    data: Vec<f64>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    beta: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    maxit: i64,
    verbose: i32,
) -> PyResult<PyObject> {
    if !(beta > 3.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "npnormcll requires beta > 3 (the number of observations)",
        ));
    }
    let beta_n = 1.0 / (beta - 3.0).sqrt();
    let v_atanh: Vec<f64> = data.iter().map(|&x| atanh(x)).collect();
    let w = ones(v_atanh.len());
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_npnorm(&v_atanh, &w, beta_n, 100)
            .into_iter()
            .map(tanhv)
            .collect()
    } else {
        gridpoints
    };
    let initpt_t: Vec<f64> = if initpt.is_empty() {
        Vec::new()
    } else {
        initpt.iter().map(|&x| atanh(x)).collect()
    };
    let (_b, initpt_n, initpr) =
        grid::initial_npnorm(&v_atanh, &w, Some(beta_n), &initpt_t, &initpr);
    let initpt: Vec<f64> = initpt_n.into_iter().map(tanhv).collect();
    let fam: Box<dyn Family> =
        Box::new(NpNormCLL::new(data, mu0fixed.clone(), pi0fixed.clone(), beta));
    let mut solver = MixSolver::new(fam, mu0fixed, pi0fixed, initpt, initpr, gridpoints, verbose);
    solver.computemixdist(tol, maxit);
    result_to_py(py, &solver.finish())
}

/// `estpi0npnormcll_` — correlation MLE with an estimated point mass at 0
/// (mirrors the R `estpi0.npnormcll` dispatch).
#[pyfunction]
fn npnormcll_estpi0(
    py: Python,
    data: Vec<f64>,
    beta: f64,
    val: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    verbose: i32,
    fast: bool,
    relax: bool,
    inner_tol: f64,
) -> PyResult<PyObject> {
    if !(beta > 3.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "npnormcll requires beta > 3 (the number of observations)",
        ));
    }
    let beta_n = 1.0 / (beta - 3.0).sqrt();
    let v_atanh: Vec<f64> = data.iter().map(|&x| atanh(x)).collect();
    let w = ones(v_atanh.len());
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_npnorm(&v_atanh, &w, beta_n, 100)
            .into_iter()
            .map(tanhv)
            .collect()
    } else {
        gridpoints
    };
    let initpt_t: Vec<f64> = if initpt.is_empty() {
        Vec::new()
    } else {
        initpt.iter().map(|&x| atanh(x)).collect()
    };
    let (_b, initpt_n, initpr) =
        grid::initial_npnorm(&v_atanh, &w, Some(beta_n), &initpt_t, &initpr);
    let initpt: Vec<f64> = initpt_n.into_iter().map(tanhv).collect();
    let fam: Box<dyn Family> = Box::new(NpNormCLL::new(data, vec![0.0], vec![1.0], beta));
    let mut solver =
        MixSolver::new(fam, vec![0.0], vec![1.0], initpt, initpr, gridpoints, verbose);
    if fast {
        solver.estpi0_fast(val, tol, inner_tol, relax);
    } else {
        solver.estpi0(val, tol);
    }
    result_to_py(py, &solver.finish())
}

/// `nppoisll_` — Poisson mixing distribution, weighted maximum likelihood.
/// Mirrors the R `computemixdist.nppoisll` dispatch: the observations are
/// first grouped with `table(v)` into (value, count) pairs, which are passed
/// to the weighted C++ entry point.
#[pyfunction]
fn nppoisll(
    py: Python,
    data: Vec<f64>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    beta: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    maxit: i64,
    verbose: i32,
) -> PyResult<PyObject> {
    let (v, w) = aggregate_counts(&data);
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_nppois(&v, beta, 100)
    } else {
        gridpoints
    };
    let (_b, initpt, initpr) = grid::initial_nppois(&v, &w, Some(beta), &initpt, &initpr);
    let fam: Box<dyn Family> =
        Box::new(NpPoisLL::new(v, w, mu0fixed.clone(), pi0fixed.clone(), beta));
    let mut solver = MixSolver::new(fam, mu0fixed, pi0fixed, initpt, initpr, gridpoints, verbose);
    solver.computemixdist(tol, maxit);
    result_to_py(py, &solver.finish())
}

/// `estpi0nppoisll_` — Poisson MLE with an estimated point mass at 0
/// (mirrors the R `estpi0.nppoisll` dispatch).
#[pyfunction]
fn nppoisll_estpi0(
    py: Python,
    data: Vec<f64>,
    beta: f64,
    val: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    verbose: i32,
    fast: bool,
    relax: bool,
    inner_tol: f64,
) -> PyResult<PyObject> {
    let (v, w) = aggregate_counts(&data);
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_nppois(&v, beta, 100)
    } else {
        gridpoints
    };
    let (_b, initpt, initpr) = grid::initial_nppois(&v, &w, Some(beta), &initpt, &initpr);
    let fam: Box<dyn Family> = Box::new(NpPoisLL::new(v, w, vec![0.0], vec![1.0], beta));
    let mut solver =
        MixSolver::new(fam, vec![0.0], vec![1.0], initpt, initpr, gridpoints, verbose);
    if fast {
        solver.estpi0_fast(val, tol, inner_tol, relax);
    } else {
        solver.estpi0(val, tol);
    }
    result_to_py(py, &solver.finish())
}

/// `nptll_` — t mixing distribution, maximum likelihood. Mirrors the R
/// `computemixdist.nptll` dispatch: the data are probability-integral-
/// transformed to z-space (`qnorm(pt(v, df = beta))`), the initial support and
/// grid are built in z-space with `beta = 1`, both are mapped back to t-space
/// with `qt(pnorm(., df = beta))`, and the non-central-t family is fitted on
/// the raw data with `df = beta`. For `beta = Inf` the transforms are the
/// identity and the kernel reduces to the normal.
#[pyfunction]
fn nptll(
    py: Python,
    data: Vec<f64>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    beta: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    maxit: i64,
    verbose: i32,
) -> PyResult<PyObject> {
    // PIT to z-space: v1 = qnorm(pt(v, df = beta)). Identity when beta = Inf.
    let v1: Vec<f64> = data.iter().map(|&x| qnorm(pt(x, beta, true))).collect();
    let w = ones(v1.len());
    // Grid in z-space (beta = 1), mapped back to t-space with qt(pnorm(.), df).
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_npnorm(&v1, &w, 1.0, 100)
            .into_iter()
            .map(|z| qt(pnorm(z, 0.0, 1.0, true), beta))
            .collect()
    } else {
        gridpoints
    };
    // Initial support: if a mix is given, its (t-space) pt is mapped to
    // z-space before initial.npnorm; the returned support is mapped back.
    let initpt_z: Vec<f64> = if initpt.is_empty() {
        Vec::new()
    } else {
        initpt.iter().map(|&x| qnorm(pt(x, beta, true))).collect()
    };
    let (_b, initpt_z2, initpr) =
        grid::initial_npnorm(&v1, &w, Some(1.0), &initpt_z, &initpr);
    let initpt: Vec<f64> = initpt_z2
        .into_iter()
        .map(|z| qt(pnorm(z, 0.0, 1.0, true), beta))
        .collect();
    let fam: Box<dyn Family> =
        Box::new(NpTLL::new(data, mu0fixed.clone(), pi0fixed.clone(), beta));
    let mut solver = MixSolver::new(fam, mu0fixed, pi0fixed, initpt, initpr, gridpoints, verbose);
    solver.computemixdist(tol, maxit);
    result_to_py(py, &solver.finish())
}

/// `estpi0nptll_` — t MLE with an estimated point mass at 0 (mirrors the R
/// `estpi0.nptll` dispatch; same PIT transform as `nptll`).
#[pyfunction]
fn nptll_estpi0(
    py: Python,
    data: Vec<f64>,
    beta: f64,
    val: f64,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    tol: f64,
    verbose: i32,
    fast: bool,
    relax: bool,
    inner_tol: f64,
) -> PyResult<PyObject> {
    let v1: Vec<f64> = data.iter().map(|&x| qnorm(pt(x, beta, true))).collect();
    let w = ones(v1.len());
    let gridpoints = if gridpoints.is_empty() {
        grid::gridpoints_npnorm(&v1, &w, 1.0, 100)
            .into_iter()
            .map(|z| qt(pnorm(z, 0.0, 1.0, true), beta))
            .collect()
    } else {
        gridpoints
    };
    let initpt_z: Vec<f64> = if initpt.is_empty() {
        Vec::new()
    } else {
        initpt.iter().map(|&x| qnorm(pt(x, beta, true))).collect()
    };
    let (_b, initpt_z2, initpr) =
        grid::initial_npnorm(&v1, &w, Some(1.0), &initpt_z, &initpr);
    let initpt: Vec<f64> = initpt_z2
        .into_iter()
        .map(|z| qt(pnorm(z, 0.0, 1.0, true), beta))
        .collect();
    // estpi0nptll_ has no mu0/pi0: the fixed component is the (zero-weight)
    // point mass at 0 whose proportion is estimated by the solver.
    let fam: Box<dyn Family> = Box::new(NpTLL::new(data, vec![0.0], vec![0.0], beta));
    let mut solver =
        MixSolver::new(fam, vec![0.0], vec![0.0], initpt, initpr, gridpoints, verbose);
    if fast {
        solver.estpi0_fast(val, tol, inner_tol, relax);
    } else {
        solver.estpi0(val, tol);
    }
    result_to_py(py, &solver.finish())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The C++ core's OpenMP workers idle-spin by default (OMP_WAIT_POLICY=
    // active) and, when this module coexists with the rayon pool, the spin
    // burns cores the rayon tasks need (measured: solvegrad 20 -> 35 ms at
    // n=5000). Set passive *before* any C++ call so libomp initializes with
    // sleeping workers. Must happen at module import — libomp reads the
    // variable only once, at first thread-team creation.
    if std::env::var_os("OMP_WAIT_POLICY").is_none() {
        std::env::set_var("OMP_WAIT_POLICY", "passive");
    }
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(npnormll, m)?)?;
    m.add_function(wrap_pyfunction!(npnormll_estpi0, m)?)?;
    m.add_function(wrap_pyfunction!(npnormcvm, m)?)?;
    m.add_function(wrap_pyfunction!(npnormcvm_estpi0, m)?)?;
    m.add_function(wrap_pyfunction!(npnormad, m)?)?;
    m.add_function(wrap_pyfunction!(npnormad_estpi0, m)?)?;
    m.add_function(wrap_pyfunction!(npnormcll, m)?)?;
    m.add_function(wrap_pyfunction!(npnormcll_estpi0, m)?)?;
    m.add_function(wrap_pyfunction!(nppoisll, m)?)?;
    m.add_function(wrap_pyfunction!(nppoisll_estpi0, m)?)?;
    m.add_function(wrap_pyfunction!(nptll, m)?)?;
    m.add_function(wrap_pyfunction!(nptll_estpi0, m)?)?;
    Ok(())
}
