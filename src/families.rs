//! Per-family implementations of the mixing-distribution model.
//!
//! Each family plugs into the shared engine (see `engine::MixSolver`)
//! through the [`Family`] trait, mirroring the virtual interface of the C++
//! `npfixedcomp` base class in `npfixedcomp2/inst/include/npfixedcomp.h`.
//!
//! Implemented families:
//! - [`NpNormLL`] — normal mixing distribution, maximum likelihood (the C++
//!   `npnormll` class; flag `d1`).
//! - [`NpTLL`] — t mixing distribution, maximum likelihood (the C++ `nptll`
//!   class; non-central t kernel, flag `d0`).

use crate::nnls::{pnnlssum, pnnqp};
use crate::stats::{dnorm, dnt, gammln, pnorm, LN_SQRT_2PI};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Below this observation count the per-observation loops stay serial:
/// rayon's scheduling overhead outweighs the gain on small inputs.
///
/// The threshold can be overridden at runtime with the `NPFIXEDCOMPY_PAR_N`
/// environment variable (e.g. a huge value forces the serial path, which is
/// useful for benchmarking and exact-parity checks).
fn par_n() -> usize {
    static ONCE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| {
        std::env::var("NPFIXEDCOMPY_PAR_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048)
    })
}

/// Sum of `f(i)` over `i in 0..n`; parallel tree reduction for large `n`.
#[inline]
fn par_sum(n: usize, f: impl Fn(usize) -> f64 + Send + Sync) -> f64 {
    if n >= par_n() {
        (0..n).into_par_iter().map(f).sum()
    } else {
        let mut s = 0.0f64;
        for i in 0..n {
            s += f(i);
        }
        s
    }
}

/// Sum of a 3-tuple of per-index contributions, `f(i) = [a, b, c]`.
///
/// For `n < PAR_N` the summation is the plain sequential loop (bit-identical
/// to the original code); above the threshold a tree reduction is used
/// (reassociation changes the last ulps, far below the 1e-6 parity
/// tolerance).
#[inline]
fn par3(n: usize, f: impl Fn(usize) -> [f64; 3] + Send + Sync) -> [f64; 3] {
    if n >= par_n() {
        (0..n)
            .into_par_iter()
            .map(f)
            .reduce(
                || [0.0; 3],
                |mut a, b| {
                    a[0] += b[0];
                    a[1] += b[1];
                    a[2] += b[2];
                    a
                },
            )
    } else {
        let mut a = [0.0f64; 3];
        for i in 0..n {
            let b = f(i);
            a[0] += b[0];
            a[1] += b[1];
            a[2] += b[2];
        }
        a
    }
}

/// Single-pass accumulation of `k` sums over `i in 0..n`, where `f(i, acc)`
/// adds index `i`'s contribution to every one of the `k` accumulators.
///
/// The per-sum parallel reductions in `gradfunvec` launched one rayon pass
/// (and reduction) per accumulator, which dominated the grid-gradient hot
/// path for large `n`; this computes all `k` sums in one pass with
/// per-thread accumulators. Serial below `PAR_N` (bit-identical to the
/// original); above, a chunked pass with the same per-index arithmetic
/// (reassociation changes only the last ulps, far below the parity
/// tolerance).
fn par_acc(n: usize, k: usize, f: impl Fn(usize, &mut [f64]) + Send + Sync) -> Vec<f64> {
    if n < par_n() {
        let mut acc = vec![0.0f64; k];
        for i in 0..n {
            f(i, &mut acc);
        }
        return acc;
    }
    let threads = rayon::current_num_threads();
    let ch = (n / (threads * 4)).max(1024);
    (0..n)
        .into_par_iter()
        .chunks(ch)
        .map(|chunk| {
            let mut acc = vec![0.0f64; k];
            for i in chunk {
                f(i, &mut acc);
            }
            acc
        })
        .reduce(
            || Vec::new(),
            |mut a, b| {
                // `reduce` folds with the (empty) init as the left operand;
                // an empty accumulator must adopt the right one, otherwise
                // the element-wise zip is a no-op and the result stays empty.
                if a.is_empty() {
                    return b;
                }
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        )
}

/// Fill an (n x m) `DMatrix` from a per-row closure `f(i, row)` (row-major
/// rows), parallel over rows for large `n`, then transpose once into
/// column-major storage. Rows are independent, so each element is written
/// exactly once with the same arithmetic as the serial column loop.
fn mat_fill(n: usize, m: usize, f: impl Fn(usize, &mut [f64]) + Sync) -> DMatrix<f64> {
    let mut raw = vec![0.0f64; n * m];
    if n >= par_n() {
        raw.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
            f(i, row);
        });
    } else {
        for i in 0..n {
            f(i, &mut raw[i * m..(i + 1) * m]);
        }
    }
    DMatrix::<f64>::from_vec(m, n, raw).transpose()
}

/// The family interface used by the engine (mirrors the virtual methods of
/// the C++ `npfixedcomp` base class).
pub trait Family {
    /// Loss value given the mixture density `maps` (the fixed-component
    /// density is excluded; the family adds `precompute` internally).
    fn lossfunction(&self, maps: &[f64]) -> f64;

    /// Mixture density at the data for support points `mu0`/`pi0` (the fixed
    /// components are excluded).
    fn mapping(&self, mu0: &[f64], pi0: &[f64]) -> Vec<f64>;

    /// Gradient of the loss w.r.t. a single new support point `mu`.
    ///
    /// Returns `(ansd0, ansd1)`: the derivative in the probability direction
    /// and in the support-point direction; unrequested components are `0.0`
    /// (the C++ leaves them uninitialised and callers ignore them).
    fn gradfun(&self, mu: f64, dens: &[f64], d0: bool, d1: bool) -> (f64, f64);

    /// Vectorised gradient over support points `mu`.
    fn gradfunvec(&self, mu: &[f64], dens: &[f64], d0: bool, d1: bool) -> (Vec<f64>, Vec<f64>);

    /// Recompute the weights `pi0` given support points `mu0` and the current
    /// mixture density `dens` (constrained NNLS weight subproblem followed by
    /// the Armijo line search `checklossfun2`).
    fn computeweights(&self, mu0: &[f64], pi0: &mut Vec<f64>, dens: &[f64]);

    /// The precomputed fixed-component density (the C++ base class
    /// `precompute` member).
    fn precompute(&self) -> &[f64];

    /// The base-class `checklossfun2`: Armijo backtracking line search on the
    /// weight step `eta` (`sigma` halves from 2, `alpha = 0.3333`).
    fn checklossfun2(&self, diff: &[f64], pi0: &mut Vec<f64>, eta: &[f64], p: &[f64], dens: &[f64]) {
        let llorigin = self.lossfunction(dens);
        let mut con = 0.0f64;
        for j in 0..p.len().min(eta.len()) {
            con -= p[j] * eta[j];
        }
        let alpha = 0.3333;
        let mut sigma = 2.0;
        let mut ans = pi0.clone();
        loop {
            sigma *= 0.5;
            let mut td = Vec::with_capacity(dens.len());
            td.extend(dens.iter().zip(diff.iter()).map(|(&a, &b)| a + sigma * b));
            let lhs = self.lossfunction(&td);
            let rhs = llorigin + alpha * sigma * con;
            if lhs < rhs {
                for j in 0..pi0.len() {
                    ans[j] = pi0[j] + sigma * eta[j];
                }
                break;
            }
            if sigma < 0.001 {
                break;
            }
        }
        *pi0 = ans;
    }

    /// Extra term added to the loss for reporting. Zero for the un-binned
    /// likelihood families.
    fn extrafun(&self) -> f64 {
        0.0
    }

    /// Hypothesis statistic for `estpi0` (likelihood families: `ll - minloss`).
    fn hypofun(&self, ll: f64, minloss: f64) -> f64;

    /// The family (unmixed) density at a single point `x`.
    fn familydensity(&self, x: f64, mu0: &[f64], pi0: &[f64]) -> f64;

    /// Update the fixed components and recompute the precomputed fixed
    /// density (`setprecompute`).
    fn set_fixed(&mut self, mu0fixed: &[f64], pi0fixed: &[f64]);

    /// One-time, per-grid precomputation hook, called by the engine once the
    /// (sorted) grid is fixed. A family with an expensive kernel that is
    /// re-evaluated at every grid point on every iteration (the non-central-t
    /// `nptll` kernel) precomputes the full data×grid kernel matrix here, so
    /// each iteration's grid sweep becomes a cheap lookup. The kernel depends
    /// only on the data, `beta` and the grid — never on the current weights or
    /// the fixed components — so the matrix stays valid for the whole run,
    /// including across `estpi0`'s bisection (same family instance, same grid).
    /// Default: nothing to precompute.
    fn prepare(&mut self, _grid: &[f64]) {}

    fn family_name(&self) -> &str;
    fn flag(&self) -> &str;
    fn beta_value(&self) -> f64;
}

/// `ln N(x; mu, beta)` computed exactly like the C++ `dnormarray` (log-space
/// arithmetic; `exp` is applied by the caller):
/// `(x - mu)^2 / (-2 beta^2) - (ln sqrt(2 pi) + ln beta)`.
#[inline]
fn ln_dnorm(x: f64, mu: f64, beta: f64) -> f64 {
    let d = x - mu;
    (d * d) / (-2.0 * beta * beta) - (LN_SQRT_2PI + beta.ln())
}

/// `N(x; mu, beta)`, the non-log version.
#[inline]
fn dnormv(x: f64, mu: f64, beta: f64) -> f64 {
    ln_dnorm(x, mu, beta).exp()
}

/// `dnpnorm_` (non-log): the normal mixture density
/// `sum_j pi0[j] N(x_i; mu0[j], beta)` evaluated at all data points.
fn dnpnorm(data: &[f64], mu0: &[f64], pi0: &[f64], beta: f64) -> Vec<f64> {
    let n = data.len();
    if mu0.is_empty() {
        return vec![0.0; n];
    }
    let m = mu0.len();
    let mut out = vec![0.0f64; n];
    if m == 1 {
        let (mu, pj) = (mu0[0], pi0[0]);
        if n >= par_n() {
            out.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o = dnormv(data[i], mu, beta) * pj;
            });
        } else {
            for i in 0..n {
                out[i] = dnormv(data[i], mu, beta) * pj;
            }
        }
        return out;
    }
    for j in 0..m {
        let (mu, pj) = (mu0[j], pi0[j]);
        if n >= par_n() {
            out.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o += dnormv(data[i], mu, beta) * pj;
            });
        } else {
            for i in 0..n {
                out[i] += dnormv(data[i], mu, beta) * pj;
            }
        }
    }
    out
}

/// `dnpt_` (non-log): the non-central-t mixture density
/// `sum_j pi0[j] * dt(x_i, df = beta, ncp = mu0[j])` (the `nptll` kernel;
/// the support point is the non-centrality, and `beta = Inf` collapses to
/// the normal with mean `mu0[j]`).
fn dnpt(data: &[f64], mu0: &[f64], pi0: &[f64], beta: f64) -> Vec<f64> {
    let n = data.len();
    if mu0.is_empty() {
        return vec![0.0; n];
    }
    let m = mu0.len();
    let mut out = vec![0.0f64; n];
    if m == 1 {
        let (mu, pj) = (mu0[0], pi0[0]);
        if n >= par_n() {
            out.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o = dnt(data[i], beta, mu) * pj;
            });
        } else {
            for i in 0..n {
                out[i] = dnt(data[i], beta, mu) * pj;
            }
        }
        return out;
    }
    for j in 0..m {
        let (mu, pj) = (mu0[j], pi0[j]);
        if n >= par_n() {
            out.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o += dnt(data[i], beta, mu) * pj;
            });
        } else {
            for i in 0..n {
                out[i] += dnt(data[i], beta, mu) * pj;
            }
        }
    }
    out
}

/// Normal mixing distribution, maximum likelihood — a port of the C++
/// `npnormll` class (`npfixedcomp2/src/npnormll.cpp`).
pub struct NpNormLL {
    data: Vec<f64>,
    len: usize,
    beta: f64,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    precompute: Vec<f64>,
}

impl NpNormLL {
    pub fn new(data: Vec<f64>, mu0fixed: Vec<f64>, pi0fixed: Vec<f64>, beta: f64) -> Self {
        let len = data.len();
        let precompute = dnpnorm(&data, &mu0fixed, &pi0fixed, beta);
        Self {
            data,
            len,
            beta,
            mu0fixed,
            pi0fixed,
            precompute,
        }
    }

}

impl Family for NpNormLL {
    fn precompute(&self) -> &[f64] {
        &self.precompute
    }

    fn lossfunction(&self, maps: &[f64]) -> f64 {
        -par_sum(self.len, |i| (maps[i] + self.precompute[i]).ln())
    }

    fn mapping(&self, mu0: &[f64], pi0: &[f64]) -> Vec<f64> {
        dnpnorm(&self.data, mu0, pi0, self.beta)
    }

    fn gradfun(&self, mu: f64, dens: &[f64], d0: bool, d1: bool) -> (f64, f64) {
        if !d0 && !d1 {
            return (0.0, 0.0);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let [sum_temp, data_dot_temp, dens_dot_fullden] = par3(self.len, |i| {
            let fl = 1.0 / (dens[i] + self.precompute[i]);
            let t = dnormv(self.data[i], mu, self.beta) * scale * fl;
            [
                t,
                self.data[i] * t,
                if d0 { dens[i] * fl } else { 0.0 },
            ]
        });
        let a0 = if d0 { dens_dot_fullden - sum_temp } else { 0.0 };
        let a1 = if d1 {
            (sum_temp * mu - data_dot_temp) / (self.beta * self.beta)
        } else {
            0.0
        };
        (a0, a1)
    }

    fn gradfunvec(&self, mu: &[f64], dens: &[f64], d0: bool, d1: bool) -> (Vec<f64>, Vec<f64>) {
        let m = mu.len();
        let mut a0 = vec![0.0f64; m];
        let mut a1 = vec![0.0f64; m];
        if m == 0 {
            return (a0, a1);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let mut fullden = vec![0.0f64; n];
        let mut dens_dot_fullden = 0.0f64;
        for i in 0..n {
            let fl = 1.0 / (dens[i] + self.precompute[i]);
            fullden[i] = fl;
            dens_dot_fullden += dens[i] * fl;
        }
        // One pass over the data for large n (the previous per-point version
        // launched one rayon reduction per grid point); a tight per-point
        // serial loop with register accumulators for small n.
        let k0 = if d0 { 1 } else { 0 };
        let k1 = if d1 { 1 } else { 0 };
        let k = (k0 + k1) * m;
        if k == 0 {
            return (a0, a1);
        }
        if n >= par_n() {
            let off1 = k0 * m;
            let acc = par_acc(n, k, |i, acc| {
                let base = fullden[i];
                let xi = self.data[i];
                for j in 0..m {
                    let d = dnormv(xi, mu[j], self.beta) * base;
                    if k0 > 0 {
                        acc[j] += d;
                    }
                    if k1 > 0 {
                        acc[off1 + j] += (mu[j] - xi) * d;
                    }
                }
            });
            for j in 0..m {
                if d0 {
                    a0[j] = dens_dot_fullden - acc[j] * scale;
                }
                if d1 {
                    a1[j] = acc[off1 + j] * scale / self.beta / self.beta;
                }
            }
        } else {
            for j in 0..m {
                let mj = mu[j];
                let mut s0 = 0.0f64;
                let mut s1 = 0.0f64;
                for i in 0..n {
                    let d = dnormv(self.data[i], mj, self.beta) * fullden[i];
                    s0 += d;
                    s1 += (mj - self.data[i]) * d;
                }
                if d0 {
                    a0[j] = dens_dot_fullden - s0 * scale;
                }
                if d1 {
                    a1[j] = s1 * scale / self.beta / self.beta;
                }
            }
        }
        (a0, a1)
    }

    fn computeweights(&self, mu0: &[f64], pi0: &mut Vec<f64>, dens: &[f64]) {
        let m = mu0.len();
        if m == 0 || pi0.len() != m {
            return;
        }
        let n = self.len;
        let sum = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let mut fp = vec![0.0f64; n];
        for i in 0..n {
            fp[i] = dens[i] + self.precompute[i];
        }
        // tp = sp columnwise / fp, where sp = dnormarray(data, mu0, beta)
        let tp = mat_fill(n, m, |i, row| {
            let d = self.data[i];
            let fi = fp[i];
            for j in 0..m {
                row[j] = dnormv(d, mu0[j], self.beta) / fi;
            }
        });
        let nw = if n > 1000 {
            // C++: pnnqp_(tp^T tp, tp^T (precompute/fp - 2), sum)
            let tptp = tp.transpose() * &tp;
            let mut pv = DVector::zeros(n);
            for i in 0..n {
                pv[i] = self.precompute[i] / fp[i] - 2.0;
            }
            let ttpv = tp.transpose() * &pv;
            pnnqp(&tptp, &ttpv, sum)
        } else {
            // C++: pnnlssum_(tp, 2 - precompute/fp, sum)
            let mut bv = DVector::zeros(n);
            for i in 0..n {
                bv[i] = 2.0 - self.precompute[i] / fp[i];
            }
            pnnlssum(&tp, &bv, sum)
        };
        // diff = sp * nw - dens, where sp[i,j] = tp[i,j] * fp[i]
        let mut diff = vec![0.0f64; n];
        if n >= par_n() {
            diff.par_iter_mut().enumerate().for_each(|(i, d)| {
                let mut s = 0.0f64;
                for j in 0..m {
                    s += tp[(i, j)] * nw[j];
                }
                *d = fp[i] * s - dens[i];
            });
        } else {
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..m {
                    s += tp[(i, j)] * nw[j];
                }
                diff[i] = fp[i] * s - dens[i];
            }
        }
        let mut eta = vec![0.0f64; m];
        for j in 0..m {
            eta[j] = nw[j] - pi0[j];
        }
        let pcol: Vec<f64> = (0..m)
            .map(|j| {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += tp[(i, j)];
                }
                s
            })
            .collect();
        self.checklossfun2(&diff, pi0, &eta, &pcol, dens);
    }

    fn hypofun(&self, ll: f64, minloss: f64) -> f64 {
        ll - minloss
    }

    fn familydensity(&self, x: f64, mu0: &[f64], pi0: &[f64]) -> f64 {
        if mu0.is_empty() {
            return 0.0;
        }
        if mu0.len() == 1 {
            return dnormv(x, mu0[0], self.beta) * pi0[0];
        }
        let mut s = 0.0f64;
        for j in 0..mu0.len() {
            s += dnormv(x, mu0[j], self.beta) * pi0[j];
        }
        s
    }

    fn set_fixed(&mut self, mu0fixed: &[f64], pi0fixed: &[f64]) {
        self.mu0fixed = mu0fixed.to_vec();
        self.pi0fixed = pi0fixed.to_vec();
        self.precompute = dnpnorm(&self.data, &self.mu0fixed, &self.pi0fixed, self.beta);
    }

    fn family_name(&self) -> &str {
        "npnorm"
    }

    fn flag(&self) -> &str {
        "d1"
    }

    fn beta_value(&self) -> f64 {
        self.beta
    }
}

// ---------------------------------------------------------------------------
// Shared vectorised kernels (non-log) ported from `miscfuns.h`.
// ---------------------------------------------------------------------------

/// `dnormarray(x, mu0[], stdev)` (non-log): the (n x m) normal pdf matrix.
fn dnorm_m(x: &[f64], mu0: &[f64], beta: f64) -> DMatrix<f64> {
    let m = mu0.len();
    let b2 = beta * beta;
    let base = LN_SQRT_2PI + beta.ln();
    mat_fill(x.len(), m, |i, row| {
        let xi = x[i];
        for j in 0..m {
            let d = xi - mu0[j];
            row[j] = (d * d / (-2.0 * b2) - base).exp();
        }
    })
}

/// `pnormarray(x, mu0[], stdev, lower.tail=true)` (non-log): the (n x m)
/// normal cdf matrix.
fn pnorm_m(x: &[f64], mu0: &[f64], beta: f64) -> DMatrix<f64> {
    let m = mu0.len();
    mat_fill(x.len(), m, |i, row| {
        let xi = x[i];
        for j in 0..m {
            row[j] = pnorm(xi, mu0[j], beta, true);
        }
    })
}

/// `pnpnorm_(x, mu0[], pi0[], stdev)` (non-log): the (n) normal-cdf mixture
/// `sum_j pi0[j] Phi(x; mu0[j], beta)`.
fn pnpnorm(x: &[f64], mu0: &[f64], pi0: &[f64], beta: f64) -> Vec<f64> {
    let n = x.len();
    if mu0.is_empty() {
        return vec![0.0; n];
    }
    let m = mu0.len();
    let mut out = vec![0.0f64; n];
    for j in 0..m {
        let (pj, mj) = (pi0[j], mu0[j]);
        if n >= par_n() {
            out.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o += pnorm(x[i], mj, beta, true) * pj;
            });
        } else {
            for i in 0..n {
                out[i] += pnorm(x[i], mj, beta, true) * pj;
            }
        }
    }
    out
}

/// `dnormcarray(x, mu0[], n)` (non-log): the (n x m) one-parameter normal pdf
/// matrix (used for sample-correlation mixing; `n` is the observation count).
fn dnormc_m(x: &[f64], mu0: &[f64], n: f64) -> DMatrix<f64> {
    let len = x.len();
    let m = mu0.len();
    let sn = n.sqrt();
    mat_fill(len, m, |i, row| {
        let xi = x[i];
        for j in 0..m {
            let mj = mu0[j];
            let stdev = (1.0 - mj * mj) / sn;
            let s2 = stdev * stdev;
            let base = LN_SQRT_2PI + stdev.ln();
            let d = xi - mj;
            row[j] = (d * d / (-2.0 * s2) - base).exp();
        }
    })
}

/// `dnpnormc_(x, mu0[], pi0[], n)` (non-log): the (n) one-parameter normal pdf
/// mixture.
fn dnpnormc(x: &[f64], mu0: &[f64], pi0: &[f64], n: f64) -> Vec<f64> {
    let len = x.len();
    if mu0.is_empty() {
        return vec![0.0; len];
    }
    let m = mu0.len();
    let mut out = vec![0.0f64; len];
    let sn = n.sqrt();
    for j in 0..m {
        let pj = pi0[j];
        let mj = mu0[j];
        let stdev = (1.0 - mj * mj) / sn;
        let s2 = stdev * stdev;
        let base = LN_SQRT_2PI + stdev.ln();
        for i in 0..len {
            let d = x[i] - mj;
            out[i] += ((d * d / (-2.0 * s2) - base).exp()) * pj;
        }
    }
    out
}

/// C++ `dpoisarray` pmf, exactly replicated: the log-density is
/// `(x + mu > 0) ? x*ln(mu) - mu - lgamma(x+1) : 0`, then the whole matrix is
/// `.exp()`-ed — so the non-log else-branch is `exp(0) = 1`, not 0 (in
/// particular `dpois(0; 0) = 1`, the Poisson limit as mu -> 0). Matching this
/// matters: the estpi0 sp0 ratio divides by this, and a zero at mu = 0
/// corrupts the whole bisection.
#[inline]
fn pois_pmf_c(x: f64, mu: f64) -> f64 {
    if x + mu > 0.0 {
        (x * mu.ln() - mu - gammln(x + 1.0)).exp()
    } else {
        1.0
    }
}

/// `dpoisarray(x, mu0[])` (non-log): the (n x m) Poisson pmf matrix.
fn dpois_m(x: &[f64], mu0: &[f64]) -> DMatrix<f64> {
    let n = x.len();
    let m = mu0.len();
    let mut a = DMatrix::zeros(n, m);
    for j in 0..m {
        let mj = mu0[j];
        for i in 0..n {
            a[(i, j)] = pois_pmf_c(x[i], mj);
        }
    }
    a
}

/// `dnppois_(x, mu0[], pi0[])` (non-log): the (n) Poisson pmf mixture.
fn dnppois(x: &[f64], mu0: &[f64], pi0: &[f64]) -> Vec<f64> {
    let n = x.len();
    if mu0.is_empty() {
        return vec![0.0; n];
    }
    let m = mu0.len();
    let mut out = vec![0.0f64; n];
    for j in 0..m {
        let pj = pi0[j];
        let mj = mu0[j];
        for i in 0..n {
            out[i] += pois_pmf_c(x[i], mj) * pj;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// NpNormCVM: normal mixing, Cramer-von Mises distance.
// ---------------------------------------------------------------------------

/// Normal mixing distribution, Cramer-von Mises distance — a port of the C++
/// `npnormcvm` class (`npfixedcomp2/src/npnormcvm.cpp`). Flag `d1`.
pub struct NpNormCVM {
    data: Vec<f64>,
    len: usize,
    beta: f64,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    precompute: Vec<f64>,
}

impl NpNormCVM {
    /// CVM-specific precompute: the empirical midpoints `(i+0.5)/n` minus the
    /// fixed-component CDF.
    fn recompute_pre(&mut self) {
        let fixed = pnpnorm(&self.data, &self.mu0fixed, &self.pi0fixed, self.beta);
        let n = self.len as f64;
        self.precompute = (0..self.len)
            .zip(fixed)
            .map(|(i, f)| (i as f64 + 0.5) / n - f)
            .collect();
    }

    pub fn new(data: Vec<f64>, mu0fixed: Vec<f64>, pi0fixed: Vec<f64>, beta: f64) -> Self {
        let len = data.len();
        let mut s = Self {
            data,
            len,
            beta,
            mu0fixed,
            pi0fixed,
            precompute: Vec::new(),
        };
        s.recompute_pre();
        s
    }
}

impl Family for NpNormCVM {
    fn precompute(&self) -> &[f64] {
        &self.precompute
    }

    fn lossfunction(&self, maps: &[f64]) -> f64 {
        par_sum(self.len, |i| {
            let d = maps[i] - self.precompute[i];
            d * d
        })
    }

    fn mapping(&self, mu0: &[f64], pi0: &[f64]) -> Vec<f64> {
        pnpnorm(&self.data, mu0, pi0, self.beta)
    }

    fn gradfun(&self, mu: f64, dens: &[f64], d0: bool, d1: bool) -> (f64, f64) {
        if !d0 && !d1 {
            return (0.0, 0.0);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let sum_new = if d0 {
            if n >= par_n() {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let fl = dens[i] - self.precompute[i];
                        (pnorm(self.data[i], mu, self.beta, true) * scale - dens[i]) * fl
                    })
                    .sum()
            } else {
                let mut s = 0.0f64;
                for i in 0..n {
                    let fl = dens[i] - self.precompute[i];
                    s += (pnorm(self.data[i], mu, self.beta, true) * scale - dens[i]) * fl;
                }
                s
            }
        } else {
            0.0
        };
        let sum_d1 = if d1 {
            if n >= par_n() {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let fl = dens[i] - self.precompute[i];
                        dnorm(self.data[i], mu, self.beta) * fl
                    })
                    .sum()
            } else {
                let mut s = 0.0f64;
                for i in 0..n {
                    let fl = dens[i] - self.precompute[i];
                    s += dnorm(self.data[i], mu, self.beta) * fl;
                }
                s
            }
        } else {
            0.0
        };
        let a0 = if d0 { sum_new * 2.0 } else { 0.0 };
        let a1 = if d1 { sum_d1 * (-2.0 * scale) } else { 0.0 };
        (a0, a1)
    }

    fn gradfunvec(&self, mu: &[f64], dens: &[f64], d0: bool, d1: bool) -> (Vec<f64>, Vec<f64>) {
        let m = mu.len();
        let mut a0 = vec![0.0f64; m];
        let mut a1 = vec![0.0f64; m];
        if m == 0 {
            return (a0, a1);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let mut fullden = vec![0.0f64; n];
        let mut fd_dot_dens = 0.0f64;
        for i in 0..n {
            let fl = dens[i] - self.precompute[i];
            fullden[i] = fl;
            fd_dot_dens += fl * dens[i];
        }
        // One pass over the data for large n (the previous per-point version
        // launched one rayon reduction per grid point); a tight per-point
        // serial loop with register accumulators for small n.
        let k0 = if d0 { 1 } else { 0 };
        let k1 = if d1 { 1 } else { 0 };
        let k = (k0 + k1) * m;
        if k == 0 {
            return (a0, a1);
        }
        if n >= par_n() {
            let off1 = k0 * m;
            let acc = par_acc(n, k, |i, acc| {
                let base = fullden[i];
                let xi = self.data[i];
                for j in 0..m {
                    if k0 > 0 {
                        acc[j] += pnorm(xi, mu[j], self.beta, true) * base;
                    }
                    if k1 > 0 {
                        acc[off1 + j] += dnorm(xi, mu[j], self.beta) * base;
                    }
                }
            });
            for j in 0..m {
                if d0 {
                    a0[j] = acc[j] * 2.0 * scale - fd_dot_dens * 2.0;
                }
                if d1 {
                    a1[j] = acc[off1 + j] * (-2.0 * scale);
                }
            }
        } else {
            // Mirror the original per-point loop exactly, including skipping
            // the pnorm (CDF) sum when `d0` is off: the CDF is a special
            // function several times more expensive than dnorm, and the hot
            // grid call only asks for d1.
            for j in 0..m {
                let mj = mu[j];
                let mut s0 = 0.0f64;
                let mut s1 = 0.0f64;
                for i in 0..n {
                    let base = fullden[i];
                    if d0 {
                        s0 += pnorm(self.data[i], mj, self.beta, true) * base;
                    }
                    if d1 {
                        s1 += dnorm(self.data[i], mj, self.beta) * base;
                    }
                }
                if d0 {
                    a0[j] = s0 * 2.0 * scale - fd_dot_dens * 2.0;
                }
                if d1 {
                    a1[j] = s1 * (-2.0 * scale);
                }
            }
        }
        (a0, a1)
    }

    fn computeweights(&self, mu0: &[f64], pi0: &mut Vec<f64>, _dens: &[f64]) {
        let m = mu0.len();
        if m == 0 || pi0.len() != m {
            return;
        }
        let n = self.len;
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let fp = pnorm_m(&self.data, mu0, self.beta);
        let nw = if n > 1000 {
            // C++: pnnqp_(fp^T fp, fp^T precompute * -1, sum)
            let tptp = fp.transpose() * &fp;
            let pv: DVector<f64> =
                DVector::from_vec(self.precompute.iter().map(|v| -v).collect());
            let ttpv = fp.transpose() * &pv;
            pnnqp(&tptp, &ttpv, scale)
        } else {
            let bv = DVector::from_vec(self.precompute.clone());
            pnnlssum(&fp, &bv, scale)
        };
        let mut newpi = vec![0.0f64; m];
        for j in 0..m {
            newpi[j] = nw[j];
        }
        *pi0 = newpi;
    }

    fn extrafun(&self) -> f64 {
        // C++: `1 / 12 / this->len` — integer division, so always 0. R (ground
        // truth) reports the CVM ll without this term; match it exactly.
        0.0
    }

    fn hypofun(&self, ll: f64, _minloss: f64) -> f64 {
        ll
    }

    fn familydensity(&self, x: f64, mu0: &[f64], pi0: &[f64]) -> f64 {
        if mu0.is_empty() {
            return 0.0;
        }
        if mu0.len() == 1 {
            return dnormv(x, mu0[0], self.beta) * pi0[0];
        }
        let mut s = 0.0f64;
        for j in 0..mu0.len() {
            s += dnormv(x, mu0[j], self.beta) * pi0[j];
        }
        s
    }

    fn set_fixed(&mut self, mu0fixed: &[f64], pi0fixed: &[f64]) {
        self.mu0fixed = mu0fixed.to_vec();
        self.pi0fixed = pi0fixed.to_vec();
        self.recompute_pre();
    }

    fn family_name(&self) -> &str {
        "npnorm"
    }

    fn flag(&self) -> &str {
        "d1"
    }

    fn beta_value(&self) -> f64 {
        self.beta
    }
}

// ---------------------------------------------------------------------------
// NpNormAD: normal mixing, Anderson-Darling distance.
// ---------------------------------------------------------------------------

/// Normal mixing distribution, Anderson-Darling distance — a port of the C++
/// `npnormad` class (`npfixedcomp2/src/npnormad.cpp`). Flag `d1`.
pub struct NpNormAD {
    data: Vec<f64>,
    len: usize,
    beta: f64,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    precompute: Vec<f64>,
    w1: Vec<f64>,
    w2: Vec<f64>,
}

impl NpNormAD {
    fn recompute_pre(&mut self) {
        self.precompute = pnpnorm(&self.data, &self.mu0fixed, &self.pi0fixed, self.beta);
    }

    pub fn new(data: Vec<f64>, mu0fixed: Vec<f64>, pi0fixed: Vec<f64>, beta: f64) -> Self {
        let len = data.len();
        let n = len as f64;
        // w1 = LinSpaced(len, 1, 2len-1)/len = (2i+1)/len ; w2 = reversed.
        let mut w1 = vec![0.0f64; len];
        for i in 0..len {
            w1[i] = (2.0 * i as f64 + 1.0) / n;
        }
        let mut w2 = w1.clone();
        w2.reverse();
        let mut s = Self {
            data,
            len,
            beta,
            mu0fixed,
            pi0fixed,
            precompute: Vec::new(),
            w1,
            w2,
        };
        s.recompute_pre();
        s
    }
}

impl Family for NpNormAD {
    fn precompute(&self) -> &[f64] {
        &self.precompute
    }

    fn lossfunction(&self, maps: &[f64]) -> f64 {
        -par_sum(self.len, |i| {
            let t = maps[i] + self.precompute[i];
            self.w1[i] * t.ln() + self.w2[i] * (-t).ln_1p()
        })
    }

    fn mapping(&self, mu0: &[f64], pi0: &[f64]) -> Vec<f64> {
        pnpnorm(&self.data, mu0, pi0, self.beta)
    }

    fn gradfun(&self, mu: f64, dens: &[f64], d0: bool, d1: bool) -> (f64, f64) {
        if !d0 && !d1 {
            return (0.0, 0.0);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let mut s1: Vec<f64> = vec![0.0f64; n];
        if n >= par_n() {
            s1.par_iter_mut().enumerate().for_each(|(i, s)| {
                let fl = dens[i] + self.precompute[i];
                *s = self.w1[i] / fl - self.w2[i] / (1.0 - fl);
            });
        } else {
            for i in 0..n {
                let fl = dens[i] + self.precompute[i];
                s1[i] = self.w1[i] / fl - self.w2[i] / (1.0 - fl);
            }
        }
        let (s1_dot_new, s1_dot_d1) = if d0 {
            let s = if n >= par_n() {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        s1[i]
                            * (pnorm(self.data[i], mu, self.beta, true) * scale + self.precompute[i])
                    })
                    .sum()
            } else {
                let mut s = 0.0f64;
                for i in 0..n {
                    let new_i = pnorm(self.data[i], mu, self.beta, true) * scale + self.precompute[i];
                    s += s1[i] * new_i;
                }
                s
            };
            (s, 0.0)
        } else if d1 {
            let s = if n >= par_n() {
                (0..n)
                    .into_par_iter()
                    .map(|i| s1[i] * dnorm(self.data[i], mu, self.beta) * scale)
                    .sum()
            } else {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += s1[i] * dnorm(self.data[i], mu, self.beta) * scale;
                }
                s
            };
            (0.0, s)
        } else {
            (0.0, 0.0)
        };
        // C++: (s1.dot(new) + sum(w2/(1-fl))) * -1 + 2n   (note: SUM is POSITIVE inside the
        // negation, so the gradient is -s1_dot - sum_w2 + 2n).
        let sum_w2_term = if d0 {
            par_sum(n, |i| self.w2[i] / (1.0 - (dens[i] + self.precompute[i])))
        } else {
            0.0
        };
        let a0 = if d0 {
            (s1_dot_new + sum_w2_term) * -1.0 + 2.0 * self.len as f64
        } else {
            0.0
        };
        let a1 = if d1 { s1_dot_d1 } else { 0.0 };
        (a0, a1)
    }

    fn gradfunvec(&self, mu: &[f64], dens: &[f64], d0: bool, d1: bool) -> (Vec<f64>, Vec<f64>) {
        let m = mu.len();
        let mut a0 = vec![0.0f64; m];
        let mut a1 = vec![0.0f64; m];
        if m == 0 {
            return (a0, a1);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let sum_w2_term = if d0 {
            par_sum(n, |i| self.w2[i] / (1.0 - (dens[i] + self.precompute[i])))
        } else {
            0.0
        };
        // One pass over the data for large n (the previous per-point version
        // launched one rayon reduction per grid point); a tight per-point
        // serial loop with register accumulators for small n.
        let k0 = if d0 { 1 } else { 0 };
        let k1 = if d1 { 1 } else { 0 };
        let k = (k0 + k1) * m;
        if k == 0 {
            return (a0, a1);
        }
        if n >= par_n() {
            let off1 = k0 * m;
            let acc = par_acc(n, k, |i, acc| {
                let fl = dens[i] + self.precompute[i];
                let s1i = self.w1[i] / fl - self.w2[i] / (1.0 - fl);
                for j in 0..m {
                    if k0 > 0 {
                        acc[j] += (pnorm(self.data[i], mu[j], self.beta, true) * scale + self.precompute[i]) * s1i;
                    }
                    if k1 > 0 {
                        acc[off1 + j] += dnorm(self.data[i], mu[j], self.beta) * s1i;
                    }
                }
            });
            for j in 0..m {
                if d0 {
                    a0[j] = acc[j] * -1.0 + 2.0 * self.len as f64 - sum_w2_term;
                }
                if d1 {
                    a1[j] = acc[off1 + j] * scale;
                }
            }
        } else {
            // Mirror the original per-point loop exactly, including skipping
            // the pnorm (CDF) sum when `d0` is off: the CDF is a special
            // function several times more expensive than dnorm, and the hot
            // grid call only asks for d1.
            for j in 0..m {
                let mj = mu[j];
                let mut s0 = 0.0f64;
                let mut s1 = 0.0f64;
                for i in 0..n {
                    let fl = dens[i] + self.precompute[i];
                    let s1i = self.w1[i] / fl - self.w2[i] / (1.0 - fl);
                    if d0 {
                        s0 += (pnorm(self.data[i], mj, self.beta, true) * scale + self.precompute[i]) * s1i;
                    }
                    if d1 {
                        s1 += dnorm(self.data[i], mj, self.beta) * s1i;
                    }
                }
                if d0 {
                    a0[j] = s0 * -1.0 + 2.0 * self.len as f64 - sum_w2_term;
                }
                if d1 {
                    a1[j] = s1 * scale;
                }
            }
        }
        (a0, a1)
    }

    fn computeweights(&self, mu0: &[f64], pi0: &mut Vec<f64>, dens: &[f64]) {
        let m = mu0.len();
        if m == 0 || pi0.len() != m {
            return;
        }
        let n = self.len;
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let sf = pnorm_m(&self.data, mu0, self.beta);
        let sp: Vec<f64> = (0..n).map(|i| dens[i] + self.precompute[i]).collect();
        // S = sf / sp, U = sf / (sp - 1)
        let (s, u) = {
            let mut sraw = vec![0.0f64; n * m];
            let mut uraw = vec![0.0f64; n * m];
            if n >= par_n() {
                sraw.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
                    let spi = sp[i];
                    for j in 0..m {
                        row[j] = sf[(i, j)] / spi;
                    }
                });
                uraw.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
                    let spm1 = sp[i] - 1.0;
                    for j in 0..m {
                        row[j] = sf[(i, j)] / spm1;
                    }
                });
            } else {
                for i in 0..n {
                    let spi = sp[i];
                    let spm1 = sp[i] - 1.0;
                    for j in 0..m {
                        let v = sf[(i, j)];
                        sraw[i * m + j] = v / spi;
                        uraw[i * m + j] = v / spm1;
                    }
                }
            }
            let s = DMatrix::<f64>::from_vec(m, n, sraw).transpose();
            let u = DMatrix::<f64>::from_vec(m, n, uraw).transpose();
            (s, u)
        };
        let sw1: DVector<f64> = DVector::from_vec(self.w1.clone());
        let sw2: DVector<f64> = DVector::from_vec(self.w2.clone());
        let s2: DVector<f64> = s.transpose() * &sw1 + u.transpose() * &sw2;
        // q = S^T diag(w1) S + U^T diag(w2) U — a SINGLE power of the weights
        // (C++ `S.transpose() * w1.asDiagonal() * S`), i.e. row-scale by
        // sqrt(w), not w.
        let (sr, ur) = {
            let mut sraw = vec![0.0f64; n * m];
            let mut uraw = vec![0.0f64; n * m];
            if n >= par_n() {
                sraw.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
                    let w1 = self.w1[i].sqrt();
                    for j in 0..m {
                        row[j] = s[(i, j)] * w1;
                    }
                });
                uraw.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
                    let w2 = self.w2[i].sqrt();
                    for j in 0..m {
                        row[j] = u[(i, j)] * w2;
                    }
                });
            } else {
                for i in 0..n {
                    let w1 = self.w1[i].sqrt();
                    let w2 = self.w2[i].sqrt();
                    for j in 0..m {
                        sraw[i * m + j] = s[(i, j)] * w1;
                        uraw[i * m + j] = u[(i, j)] * w2;
                    }
                }
            }
            let sr = DMatrix::<f64>::from_vec(m, n, sraw).transpose();
            let ur = DMatrix::<f64>::from_vec(m, n, uraw).transpose();
            (sr, ur)
        };
        let q = sr.transpose() * &sr + ur.transpose() * &ur;
        // p = -2*S2 + S^T (precompute/sp o w1) + U^T ((1-precompute)/(1-sp) o w2)
        let mut p = DVector::zeros(m);
        for j in 0..m {
            p[j] = -2.0 * s2[j];
            if n >= par_n() {
                let t: f64 = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        s[(i, j)] * self.precompute[i] / sp[i] * self.w1[i]
                            + u[(i, j)] * (1.0 - self.precompute[i]) / (1.0 - sp[i]) * self.w2[i]
                    })
                    .sum();
                p[j] += t;
            } else {
                for i in 0..n {
                    p[j] += s[(i, j)] * self.precompute[i] / sp[i] * self.w1[i]
                        + u[(i, j)] * (1.0 - self.precompute[i]) / (1.0 - sp[i]) * self.w2[i];
                }
            }
        }
        let nw = pnnqp(&q, &p, scale);
        // diff = sf * nw - dens
        let mut diff = vec![0.0f64; n];
        if n >= par_n() {
            diff.par_iter_mut().enumerate().for_each(|(i, d)| {
                let mut v = 0.0f64;
                for j in 0..m {
                    v += sf[(i, j)] * nw[j];
                }
                *d = v - dens[i];
            });
        } else {
            for i in 0..n {
                let mut v = 0.0f64;
                for j in 0..m {
                    v += sf[(i, j)] * nw[j];
                }
                diff[i] = v - dens[i];
            }
        }
        let mut eta = vec![0.0f64; m];
        for j in 0..m {
            eta[j] = nw[j] - pi0[j];
        }
        // C++ passes S2 (= S^T w1 + U^T w2) as the Armijo `p` argument.
        let s2v: Vec<f64> = s2.iter().cloned().collect();
        self.checklossfun2(&diff, pi0, &eta, &s2v, dens);
    }

    fn extrafun(&self) -> f64 {
        -(self.len as f64)
    }

    fn hypofun(&self, ll: f64, _minloss: f64) -> f64 {
        ll
    }

    fn familydensity(&self, x: f64, mu0: &[f64], pi0: &[f64]) -> f64 {
        if mu0.is_empty() {
            return 0.0;
        }
        if mu0.len() == 1 {
            return dnormv(x, mu0[0], self.beta) * pi0[0];
        }
        let mut s = 0.0f64;
        for j in 0..mu0.len() {
            s += dnormv(x, mu0[j], self.beta) * pi0[j];
        }
        s
    }

    fn set_fixed(&mut self, mu0fixed: &[f64], pi0fixed: &[f64]) {
        self.mu0fixed = mu0fixed.to_vec();
        self.pi0fixed = pi0fixed.to_vec();
        self.recompute_pre();
    }

    fn family_name(&self) -> &str {
        "npnorm"
    }

    fn flag(&self) -> &str {
        "d1"
    }

    fn beta_value(&self) -> f64 {
        self.beta
    }
}

// ---------------------------------------------------------------------------
// NpNormCLL: one-parameter normal mixing (sample correlations), MLE.
// ---------------------------------------------------------------------------

/// One-parameter normal mixing distribution for sample correlation
/// coefficients, maximum likelihood — a port of the C++ `npnormcll` class
/// (`npfixedcomp2/src/npnormcll.cpp`). Flag `d0`; the structural parameter
/// `beta` is the number of observations.
pub struct NpNormCLL {
    data: Vec<f64>,
    len: usize,
    beta: f64,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    precompute: Vec<f64>,
}

impl NpNormCLL {
    pub fn new(data: Vec<f64>, mu0fixed: Vec<f64>, pi0fixed: Vec<f64>, beta: f64) -> Self {
        let len = data.len();
        let precompute = dnpnormc(&data, &mu0fixed, &pi0fixed, beta);
        Self {
            data,
            len,
            beta,
            mu0fixed,
            pi0fixed,
            precompute,
        }
    }
}

impl Family for NpNormCLL {
    fn precompute(&self) -> &[f64] {
        &self.precompute
    }

    fn lossfunction(&self, maps: &[f64]) -> f64 {
        -par_sum(self.len, |i| (maps[i] + self.precompute[i]).ln())
    }

    fn mapping(&self, mu0: &[f64], pi0: &[f64]) -> Vec<f64> {
        dnpnormc(&self.data, mu0, pi0, self.beta)
    }

    fn gradfun(&self, mu: f64, dens: &[f64], d0: bool, _d1: bool) -> (f64, f64) {
        if !d0 {
            return (0.0, 0.0);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let (sum_temp, dens_dot_fl) = if n >= par_n() {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let fl = 1.0 / (dens[i] + self.precompute[i]);
                    (
                        dnpnormc_single(self.data[i], mu, self.beta) * scale * fl,
                        dens[i] * fl,
                    )
                })
                .reduce(
                    || (0.0, 0.0),
                    |(a, b), (c, d)| (a + c, b + d),
                )
        } else {
            let mut sum_temp = 0.0f64;
            let mut dens_dot_fl = 0.0f64;
            for i in 0..n {
                let fl = 1.0 / (dens[i] + self.precompute[i]);
                let t = dnpnormc_single(self.data[i], mu, self.beta) * scale * fl;
                sum_temp += t;
                dens_dot_fl += dens[i] * fl;
            }
            (sum_temp, dens_dot_fl)
        };
        (dens_dot_fl - sum_temp, 0.0)
    }

    fn gradfunvec(&self, mu: &[f64], dens: &[f64], d0: bool, _d1: bool) -> (Vec<f64>, Vec<f64>) {
        let m = mu.len();
        let mut a0 = vec![0.0f64; m];
        let a1 = vec![0.0f64; m];
        if m == 0 {
            return (a0, a1);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let mut fullden = vec![0.0f64; n];
        let mut dens_dot_fl = 0.0f64;
        for i in 0..n {
            let fl = 1.0 / (dens[i] + self.precompute[i]);
            fullden[i] = fl;
            dens_dot_fl += dens[i] * fl;
        }
        // One pass over the data for large n (the previous per-point version
        // launched one rayon reduction per grid point); a tight per-point
        // serial loop with register accumulators for small n.
        let k = if d0 { m } else { 0 };
        if k == 0 {
            return (a0, a1);
        }
        if n >= par_n() {
            let acc = par_acc(n, k, |i, acc| {
                let base = fullden[i];
                let xi = self.data[i];
                for j in 0..m {
                    acc[j] += dnpnormc_single(xi, mu[j], self.beta) * base;
                }
            });
            for j in 0..m {
                if d0 {
                    a0[j] = dens_dot_fl - acc[j] * scale;
                }
            }
        } else {
            for j in 0..m {
                let mj = mu[j];
                let mut s0 = 0.0f64;
                for i in 0..n {
                    s0 += dnpnormc_single(self.data[i], mj, self.beta) * fullden[i];
                }
                if d0 {
                    a0[j] = dens_dot_fl - s0 * scale;
                }
            }
        }
        (a0, a1)
    }
    fn computeweights(&self, mu0: &[f64], pi0: &mut Vec<f64>, dens: &[f64]) {
        let m = mu0.len();
        if m == 0 || pi0.len() != m {
            return;
        }
        let n = self.len;
        let sum = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let mut fp = vec![0.0f64; n];
        for i in 0..n {
            fp[i] = dens[i] + self.precompute[i];
        }
        // tp = sp / fp, where sp = dnormcarray(data, mu0, beta)
        let tp = mat_fill(n, m, |i, row| {
            let d = self.data[i];
            let fi = fp[i];
            for j in 0..m {
                row[j] = dnpnormc_single(d, mu0[j], self.beta) / fi;
            }
        });
        let nw = if n > 1000 {
            let tptp = tp.transpose() * &tp;
            let mut pv = DVector::zeros(n);
            for i in 0..n {
                pv[i] = self.precompute[i] / fp[i] - 2.0;
            }
            let ttpv = tp.transpose() * &pv;
            pnnqp(&tptp, &ttpv, sum)
        } else {
            let mut bv = DVector::zeros(n);
            for i in 0..n {
                bv[i] = 2.0 - self.precompute[i] / fp[i];
            }
            pnnlssum(&tp, &bv, sum)
        };
        // diff = sp * nw - dens, where sp[i,j] = tp[i,j] * fp[i]
        let mut diff = vec![0.0f64; n];
        if n >= par_n() {
            diff.par_iter_mut().enumerate().for_each(|(i, d)| {
                let mut s = 0.0f64;
                for j in 0..m {
                    s += tp[(i, j)] * nw[j];
                }
                *d = fp[i] * s - dens[i];
            });
        } else {
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..m {
                    s += tp[(i, j)] * nw[j];
                }
                diff[i] = fp[i] * s - dens[i];
            }
        }
        let mut eta = vec![0.0f64; m];
        for j in 0..m {
            eta[j] = nw[j] - pi0[j];
        }
        let pcol: Vec<f64> = (0..m)
            .map(|j| {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += tp[(i, j)];
                }
                s
            })
            .collect();
        self.checklossfun2(&diff, pi0, &eta, &pcol, dens);
    }

    fn hypofun(&self, ll: f64, minloss: f64) -> f64 {
        ll - minloss
    }

    fn familydensity(&self, x: f64, mu0: &[f64], pi0: &[f64]) -> f64 {
        dnpnormc(&[x], mu0, pi0, self.beta)[0]
    }

    fn set_fixed(&mut self, mu0fixed: &[f64], pi0fixed: &[f64]) {
        self.mu0fixed = mu0fixed.to_vec();
        self.pi0fixed = pi0fixed.to_vec();
        self.precompute = dnpnormc(&self.data, &self.mu0fixed, &self.pi0fixed, self.beta);
    }

    fn family_name(&self) -> &str {
        "npnormc"
    }

    fn flag(&self) -> &str {
        "d0"
    }

    fn beta_value(&self) -> f64 {
        self.beta
    }
}

/// `dnormcarray(x, mu, n)` (non-log) for a single support point.
fn dnpnormc_single(x: f64, mu: f64, n: f64) -> f64 {
    let stdev = (1.0 - mu * mu) / n.sqrt();
    let d = (x - mu) / stdev;
    (-0.5 * d * d - LN_SQRT_2PI - stdev.ln()).exp()
}

// ---------------------------------------------------------------------------
// NpPoisLL: Poisson mixing, weighted MLE.
// ---------------------------------------------------------------------------

/// Poisson mixing distribution, maximum likelihood with observation weights
/// — a port of the C++ `nppoisll` class (`npfixedcomp2/src/nppois.cpp`). The
/// structural parameter `beta` is unused by the Poisson family (the R
/// wrapper still passes 1). Flag `d0`.
pub struct NpPoisLL {
    data: Vec<f64>,
    len: usize,
    weights: Vec<f64>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    precompute: Vec<f64>,
}

impl NpPoisLL {
    pub fn new(
        data: Vec<f64>,
        weights: Vec<f64>,
        mu0fixed: Vec<f64>,
        pi0fixed: Vec<f64>,
        _beta: f64,
    ) -> Self {
        let precompute = dnppois(&data, &mu0fixed, &pi0fixed);
        let len = data.len();
        Self {
            data,
            len,
            weights,
            mu0fixed,
            pi0fixed,
            precompute,
        }
    }
}

impl Family for NpPoisLL {
    fn precompute(&self) -> &[f64] {
        &self.precompute
    }

    fn lossfunction(&self, maps: &[f64]) -> f64 {
        -par_sum(self.len, |i| (maps[i] + self.precompute[i]).ln() * self.weights[i])
    }

    fn mapping(&self, mu0: &[f64], pi0: &[f64]) -> Vec<f64> {
        dnppois(&self.data, mu0, pi0)
    }

    fn gradfun(&self, mu: f64, dens: &[f64], d0: bool, _d1: bool) -> (f64, f64) {
        if !d0 {
            return (0.0, 0.0);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        // (dens - dnppois(data, mu, scale)).dot(weights/(dens+precompute))
        let acc = par_sum(n, |i| {
            let d = dens[i] + self.precompute[i];
            let fl = self.weights[i] / d;
            (dens[i] - pois_pmf_c(self.data[i], mu) * scale) * fl
        });
        (acc, 0.0)
    }

    fn gradfunvec(&self, mu: &[f64], dens: &[f64], d0: bool, _d1: bool) -> (Vec<f64>, Vec<f64>) {
        let m = mu.len();
        let mut a0 = vec![0.0f64; m];
        let a1 = vec![0.0f64; m];
        if m == 0 {
            return (a0, a1);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let mut fullden = vec![0.0f64; n];
        let mut dens_dot = 0.0f64;
        for i in 0..n {
            let d = dens[i] + self.precompute[i];
            fullden[i] = self.weights[i] / d;
            dens_dot += dens[i] * fullden[i];
        }
        // One pass over the data for large n (the previous per-point version
        // launched one rayon reduction per grid point); a tight per-point
        // serial loop with register accumulators for small n.
        let k = if d0 { m } else { 0 };
        if k == 0 {
            return (a0, a1);
        }
        if n >= par_n() {
            let acc = par_acc(n, k, |i, acc| {
                let base = fullden[i];
                let xi = self.data[i];
                for j in 0..m {
                    acc[j] += pois_pmf_c(xi, mu[j]) * base;
                }
            });
            for j in 0..m {
                if d0 {
                    a0[j] = dens_dot - acc[j] * scale;
                }
            }
        } else {
            for j in 0..m {
                let mj = mu[j];
                let mut s = 0.0f64;
                for i in 0..n {
                    s += pois_pmf_c(self.data[i], mj) * fullden[i];
                }
                if d0 {
                    a0[j] = dens_dot - s * scale;
                }
            }
        }
        (a0, a1)
    }

    fn computeweights(&self, mu0: &[f64], pi0: &mut Vec<f64>, dens: &[f64]) {
        let m = mu0.len();
        if m == 0 || pi0.len() != m {
            return;
        }
        let n = self.len;
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let fp: Vec<f64> = (0..n).map(|i| dens[i] + self.precompute[i]).collect();
        // tp = sp / fp, sp = dpoisarray(data, mu0)
        let tp = mat_fill(n, m, |i, row| {
            let d = self.data[i];
            let fi = fp[i];
            for j in 0..m {
                row[j] = pois_pmf_c(d, mu0[j]) / fi;
            }
        });
        // nw = pnnlssum(tp * sqrt(w) replicated, (2 - precompute/fp) * sqrt(w), scale)
        let wsq: Vec<f64> = self.weights.iter().copied().map(|w| w.sqrt()).collect();
        let tw = mat_fill(n, m, |i, row| {
            let w = wsq[i];
            for j in 0..m {
                row[j] = tp[(i, j)] * w;
            }
        });
        let mut bv = DVector::zeros(n);
        for i in 0..n {
            bv[i] = (2.0 - self.precompute[i] / fp[i]) * wsq[i];
        }
        let nw = pnnlssum(&tw, &bv, scale);
        // diff = sp * nw - dens, where sp[i,j] = tp[i,j] * fp[i]
        let mut diff = vec![0.0f64; n];
        if n >= par_n() {
            diff.par_iter_mut().enumerate().for_each(|(i, d)| {
                let mut v = 0.0f64;
                for j in 0..m {
                    v += tp[(i, j)] * nw[j];
                }
                *d = fp[i] * v - dens[i];
            });
        } else {
            for i in 0..n {
                let mut v = 0.0f64;
                for j in 0..m {
                    v += tp[(i, j)] * nw[j];
                }
                diff[i] = fp[i] * v - dens[i];
            }
        }
        let mut eta = vec![0.0f64; m];
        for j in 0..m {
            eta[j] = nw[j] - pi0[j];
        }
        // p = tp^T * weights
        let mut pcol = vec![0.0f64; m];
        for j in 0..m {
            let mut s = 0.0f64;
            for i in 0..n {
                s += tp[(i, j)] * self.weights[i];
            }
            pcol[j] = s;
        }
        self.checklossfun2(&diff, pi0, &eta, &pcol, dens);
    }

    fn hypofun(&self, ll: f64, minloss: f64) -> f64 {
        ll - minloss
    }

    fn familydensity(&self, x: f64, mu0: &[f64], pi0: &[f64]) -> f64 {
        dnppois(&[x], mu0, pi0)[0]
    }

    fn set_fixed(&mut self, mu0fixed: &[f64], pi0fixed: &[f64]) {
        self.mu0fixed = mu0fixed.to_vec();
        self.pi0fixed = pi0fixed.to_vec();
        self.precompute = dnppois(&self.data, &self.mu0fixed, &self.pi0fixed);
    }

    fn family_name(&self) -> &str {
        "nppois"
    }

    fn flag(&self) -> &str {
        "d0"
    }

    fn beta_value(&self) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// t mixing distribution, maximum likelihood (C++ `nptll`)
// ---------------------------------------------------------------------------

/// t mixing distribution, maximum likelihood — a port of the C++ `nptll`
/// class. The kernel is the non-central t with `df = beta` and the support
/// point as the non-centrality, so the fitted mixture is
/// `sum_j pi0[j] dt(x, df = beta, ncp = mu0[j])`; `beta = Inf` collapses to
/// the normal MLE. Only the probability-direction gradient is available
/// (`flag = "d0"`, as in C++), so the engine's support-point search uses the
/// derivative-free parabolic method. The loss is the same negative
/// log-likelihood form as the normal family.
pub struct NpTLL {
    data: Vec<f64>,
    len: usize,
    beta: f64,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    precompute: Vec<f64>,
    /// Cached grid (the sorted support-point candidates) and the precomputed
    /// data×grid kernel matrix, stored COLUMN-MAJOR: `kmat[j*n + i] =
    /// dnt(data[i], beta, kgrid[j])`. Column `j` is therefore the contiguous
    /// slice `kmat[j*n .. (j+1)*n]`, which the gradient sweep reads directly.
    /// Built once in `prepare`; empty until then.
    kgrid: Vec<f64>,
    kmat: Vec<f64>,
}

impl NpTLL {
    pub fn new(data: Vec<f64>, mu0fixed: Vec<f64>, pi0fixed: Vec<f64>, beta: f64) -> Self {
        let len = data.len();
        let precompute = dnpt(&data, &mu0fixed, &pi0fixed, beta);
        Self {
            data,
            len,
            beta,
            mu0fixed,
            pi0fixed,
            precompute,
            kgrid: Vec::new(),
            kmat: Vec::new(),
        }
    }
}

impl Family for NpTLL {
    fn prepare(&mut self, grid: &[f64]) {
        if grid.is_empty() {
            self.kgrid.clear();
            self.kmat.clear();
            return;
        }
        self.kgrid = grid.to_vec();
        let g = grid.len();
        let n = self.len;
        // Column-major: column j is contiguous, so the per-iteration sweep is
        // a cache-friendly sequential read. Parallel over columns (each chunk
        // is one contiguous column; elements are independent).
        let mut raw = vec![0.0f64; n * g];
        if n >= par_n() {
            raw.par_chunks_mut(n).enumerate().for_each(|(j, col)| {
                let mj = grid[j];
                for i in 0..n {
                    col[i] = dnt(self.data[i], self.beta, mj);
                }
            });
        } else {
            for i in 0..n {
                let xi = self.data[i];
                for j in 0..g {
                    raw[j * n + i] = dnt(xi, self.beta, grid[j]);
                }
            }
        }
        self.kmat = raw;
    }

    fn precompute(&self) -> &[f64] {
        &self.precompute
    }

    fn lossfunction(&self, maps: &[f64]) -> f64 {
        -par_sum(self.len, |i| (maps[i] + self.precompute[i]).ln())
    }

    fn mapping(&self, mu0: &[f64], pi0: &[f64]) -> Vec<f64> {
        dnpt(&self.data, mu0, pi0, self.beta)
    }

    /// Only `ansd0` is defined (the C++ class leaves `ansd1` uninitialised;
    /// the `d0` solver never reads it).
    fn gradfun(&self, mu: f64, dens: &[f64], d0: bool, d1: bool) -> (f64, f64) {
        if !d0 && !d1 {
            return (0.0, 0.0);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let a0 = if d0 {
            par_sum(self.len, |i| {
                let fl = 1.0 / (dens[i] + self.precompute[i]);
                dens[i] * fl - dnt(self.data[i], self.beta, mu) * fl * scale
            })
        } else {
            0.0
        };
        (a0, 0.0)
    }

    fn gradfunvec(&self, mu: &[f64], dens: &[f64], d0: bool, _d1: bool) -> (Vec<f64>, Vec<f64>) {
        let m = mu.len();
        let mut a0 = vec![0.0f64; m];
        let a1 = vec![0.0f64; m];
        if m == 0 || !d0 {
            return (a0, a1);
        }
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let n = self.len;
        let mut fullden = vec![0.0f64; n];
        let mut dens_dot_fullden = 0.0f64;
        for i in 0..n {
            let fl = 1.0 / (dens[i] + self.precompute[i]);
            fullden[i] = fl;
            dens_dot_fullden += dens[i] * fl;
        }
        // The full-grid sweep (`mu` == the cached grid) reads the
        // precomputed data×grid kernel columns; anything else (interior
        // solver points) falls back to direct `dnt`. Reading column `j`
        // reuses the exact same products, in the same accumulation order,
        // as `dnt(...)*base` below, so the cached result is bit-identical
        // to the uncached path.
        let g = self.kgrid.len();
        let cached = g == m
            && self.kmat.len() == n * g
            && mu
                .iter()
                .zip(self.kgrid.iter())
                .all(|(&a, &b)| a.is_finite() && a == b);
        if cached {
            for j in 0..m {
                let col = &self.kmat[j * n..(j + 1) * n];
                let s: f64 = col
                    .iter()
                    .zip(fullden.iter().copied())
                    .map(|(&k, f)| k * f)
                    .sum();
                a0[j] = dens_dot_fullden - s * scale;
            }
        } else if n >= par_n() {
            // One pass over the data for large n (the per-point version
            // launched one rayon reduction per grid point).
            let acc = par_acc(n, m, |i, acc| {
                let base = fullden[i];
                let xi = self.data[i];
                for j in 0..m {
                    acc[j] += dnt(xi, self.beta, mu[j]) * base;
                }
            });
            for j in 0..m {
                a0[j] = dens_dot_fullden - acc[j] * scale;
            }
        } else {
            for j in 0..m {
                let mj = mu[j];
                let mut s = 0.0f64;
                for i in 0..n {
                    s += dnt(self.data[i], self.beta, mj) * fullden[i];
                }
                a0[j] = dens_dot_fullden - s * scale;
            }
        }
        (a0, a1)
    }

    fn computeweights(&self, mu0: &[f64], pi0: &mut Vec<f64>, dens: &[f64]) {
        let m = mu0.len();
        if m == 0 || pi0.len() != m {
            return;
        }
        let n = self.len;
        let sum = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let mut fp = vec![0.0f64; n];
        for i in 0..n {
            fp[i] = dens[i] + self.precompute[i];
        }
        // tp = sp columnwise / fp, where sp = dtarray(data, mu0, beta)
        let tp = mat_fill(n, m, |i, row| {
            let d = self.data[i];
            let fi = fp[i];
            for j in 0..m {
                row[j] = dnt(d, self.beta, mu0[j]) / fi;
            }
        });
        let nw = if n > 1000 {
            let tptp = tp.transpose() * &tp;
            let mut pv = DVector::zeros(n);
            for i in 0..n {
                pv[i] = self.precompute[i] / fp[i] - 2.0;
            }
            let ttpv = tp.transpose() * &pv;
            pnnqp(&tptp, &ttpv, sum)
        } else {
            let mut bv = DVector::zeros(n);
            for i in 0..n {
                bv[i] = 2.0 - self.precompute[i] / fp[i];
            }
            pnnlssum(&tp, &bv, sum)
        };
        // diff = sp * nw - dens, where sp[i,j] = tp[i,j] * fp[i]
        let mut diff = vec![0.0f64; n];
        if n >= par_n() {
            diff.par_iter_mut().enumerate().for_each(|(i, d)| {
                let mut s = 0.0f64;
                for j in 0..m {
                    s += tp[(i, j)] * nw[j];
                }
                *d = fp[i] * s - dens[i];
            });
        } else {
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..m {
                    s += tp[(i, j)] * nw[j];
                }
                diff[i] = fp[i] * s - dens[i];
            }
        }
        let mut eta = vec![0.0f64; m];
        for j in 0..m {
            eta[j] = nw[j] - pi0[j];
        }
        let pcol: Vec<f64> = (0..m)
            .map(|j| {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += tp[(i, j)];
                }
                s
            })
            .collect();
        self.checklossfun2(&diff, pi0, &eta, &pcol, dens);
    }

    fn hypofun(&self, ll: f64, minloss: f64) -> f64 {
        ll - minloss
    }

    fn familydensity(&self, x: f64, mu0: &[f64], pi0: &[f64]) -> f64 {
        dnpt(&[x], mu0, pi0, self.beta)[0]
    }

    fn set_fixed(&mut self, mu0fixed: &[f64], pi0fixed: &[f64]) {
        self.mu0fixed = mu0fixed.to_vec();
        self.pi0fixed = pi0fixed.to_vec();
        self.precompute = dnpt(&self.data, &self.mu0fixed, &self.pi0fixed, self.beta);
    }

    fn family_name(&self) -> &str {
        "npt"
    }

    fn flag(&self) -> &str {
        "d0"
    }

    fn beta_value(&self) -> f64 {
        self.beta
    }
}
