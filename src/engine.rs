//! Core driver: `computemixdist` and `estpi0`, plus the shared mixing
//! utilities (simplify / collapse / sort, the Brent-like `Brmin` and the
//! successive-parabolic `Dfmin` support-point solvers) — a faithful port of
//! `npfixedcomp2/inst/include/npfixedcomp.h`.
//!
//! Per-family numerics live in :mod:`families` behind the `Family` trait.

use crate::families::Family;
use crate::misc::diff;

/// A converged mixing-distribution estimate (the `get_ans` payload).
#[derive(Debug, Clone)]
pub struct MixResult {
    pub pt: Vec<f64>,
    pub pr: Vec<f64>,
    pub beta: f64,
    pub family: String,
    pub min_gradient: f64,
    pub ll: f64,
    pub flag: String,
    pub iter: i64,
    pub convergence: i32,
}

/// Stable sort of `mu0`/`pi0` by `mu0` (port of `sortmix`).
pub fn sortmix(mu0: &mut Vec<f64>, pi0: &mut Vec<f64>) {
    let k = mu0.len();
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| mu0[a].partial_cmp(&mu0[b]).unwrap_or(std::cmp::Ordering::Equal));
    if idx.iter().enumerate().any(|(i, &j)| i != j) {
        let mut mu = vec![0.0f64; k];
        let mut pi = vec![0.0f64; k];
        for (i, &j) in idx.iter().enumerate() {
            mu[i] = mu0[j];
            pi[i] = pi0[j];
        }
        *mu0 = mu;
        *pi0 = pi;
    }
}

/// Remove points whose weight is `|pi| <= 1e-14` (port of `simplifymix`).
pub fn simplifymix(mu0: &mut Vec<f64>, pi0: &mut Vec<f64>) {
    if mu0.len() != 1 {
        let keep: Vec<usize> = (0..pi0.len()).filter(|&i| pi0[i].abs() > 1e-14).collect();
        if keep.len() != pi0.len() {
            let mut mu = Vec::with_capacity(keep.len());
            let mut pi = Vec::with_capacity(keep.len());
            for &i in &keep {
                mu.push(mu0[i]);
                pi.push(pi0[i]);
            }
            *mu0 = mu;
            *pi0 = pi;
        }
    }
}

/// Merge adjacent support points closer than `prec` (port of `collapsemix`).
pub fn collapsemix(mu0: &mut Vec<f64>, pi0: &mut Vec<f64>, prec: f64) {
    if mu0.len() > 1 {
        let mut foo = mu0
            .windows(2)
            .any(|w| w[1] - w[0] <= prec);
        while foo {
            let mut i = 0usize;
            while i + 1 < mu0.len() {
                if mu0[i + 1] - mu0[i] <= prec {
                    let temp = pi0[i] + pi0[i + 1];
                    mu0[i] = (mu0[i] * pi0[i] + mu0[i + 1] * pi0[i + 1]) / temp;
                    pi0[i + 1] = 0.0;
                    pi0[i] = temp;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            simplifymix(mu0, pi0);
            if mu0.len() <= 1 {
                foo = false;
            } else {
                foo = mu0.windows(2).any(|w| w[1] - w[0] <= prec);
            }
        }
    }
}

/// Successive-parabolic-interpolation candidate (port of `newmin`).
#[inline]
fn newmin(x: &[f64; 3], fx: &[f64; 3]) -> f64 {
    let p = (x[2] - x[0]) * (x[2] - x[0]) * (fx[2] - fx[1])
        - (x[2] - x[1]) * (x[2] - x[1]) * (fx[2] - fx[0]);
    let q = 2.0 * ((x[2] - x[0]) * (fx[2] - fx[1]) - (x[2] - x[1]) * (fx[2] - fx[0]));
    x[2] - p / q
}

/// The engine state shared by `computemixdist` and `estpi0`.
pub struct MixSolver {
    fam: Box<dyn Family>,
    mu0fixed: Vec<f64>,
    pi0fixed: Vec<f64>,
    initpt: Vec<f64>,
    initpr: Vec<f64>,
    gridpoints: Vec<f64>,
    iter: i64,
    convergence: i32,
    resultpt: Vec<f64>,
    resultpr: Vec<f64>,
    verbose: i32,
}

impl MixSolver {
    pub fn new(
        fam: Box<dyn Family>,
        mu0fixed: Vec<f64>,
        pi0fixed: Vec<f64>,
        initpt: Vec<f64>,
        initpr: Vec<f64>,
        gridpoints: Vec<f64>,
        verbose: i32,
    ) -> Self {
        // sort gridpoints (R does sort(gridpoints) before calling the C++ fn)
        let mut gridpoints = gridpoints;
        gridpoints.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut s = Self {
            fam,
            mu0fixed,
            pi0fixed,
            initpt,
            initpr,
            gridpoints,
            iter: 0,
            convergence: 0,
            resultpt: Vec::new(),
            resultpr: Vec::new(),
            verbose,
        };
        s.set_precompute();
        s.fam.prepare(&s.gridpoints);
        s
    }

    fn set_precompute(&mut self) {
        self.fam
            .set_fixed(&self.mu0fixed, &self.pi0fixed);
    }

    /// `Brmin`: improved Brent's method for the gradient's d1 (derivative
    /// available), called on a sign-change interval. Port of `Brmin`.
    fn brmin(&self, lb: f64, ub: f64, dens: &[f64], tol: f64) -> f64 {
        let (_duma, fa) = self.fam.gradfun(lb, dens, false, true);
        let (_dumb, fb0) = self.fam.gradfun(ub, dens, false, true);
        let mut a = lb;
        let mut b = ub;
        let mut fa = fa;
        let mut fb = fb0;
        let mut s = a;
        let mut fs = fa;
        let mut c = a;
        let mut fc = fa;
        let mut guard: i64 = 0;
        while fc.abs() > tol && fs.abs() > tol && (b - a).abs() > tol && guard < 1000 {
            guard += 1;
            c = (a + b) / 2.0;
            let (_dumc, fcc) = self.fam.gradfun(c, dens, false, true);
            fc = fcc;
            if fa != fc && fb != fc {
                s = a * fb * fc / (fa - fb) / (fa - fc)
                    + b * fa * fc / (fb - fa) / (fb - fc)
                    + c * fa * fb / (fc - fa) / (fc - fb);
            } else {
                s = b - fb * (b - a) / (fb - fa);
            }
            if s > a && s < b {
                let (_dums, fss) = self.fam.gradfun(s, dens, false, true);
                fs = fss;
            } else {
                s = c;
                fs = fc;
            }
            if c > s {
                std::mem::swap(&mut c, &mut s);
                std::mem::swap(&mut fc, &mut fs);
            }
            // a < c < s < b
            if fc * fs < 0.0 {
                a = c;
                fa = fc;
                b = s;
                fb = fs;
            } else if fs * fb < 0.0 {
                a = s;
                fa = fs;
            } else {
                b = c;
                fb = fc;
            }
        }
        if fc.abs() < tol {
            c
        } else {
            s
        }
    }

    /// `Dfmin`: derivative-free minimum via successive parabolic
    /// interpolation. Port of `Dfmin`.
    fn dfmin(&self, x1: [f64; 3], fx1: [f64; 3], dens: &[f64], tol: f64) -> f64 {
        // C++: `lb`/`ub` are taken from the ORIGINAL endpoints before the
        // reordering swaps; using the post-swap xx[0]/xx[2] narrows the
        // interval and changes the midpoint fallbacks (trajectory drift).
        let mut lb = x1[0];
        let mut ub = x1[2];
        let mut xx = x1;
        let mut fxx = fx1;
        if fxx[0] < fxx[1] {
            xx.swap(0, 1);
            fxx.swap(0, 1);
        }
        if fxx[1] < fxx[2] {
            xx.swap(1, 2);
            fxx.swap(1, 2);
        }
        let mut guard: i64 = 0;
        while ub - lb > tol && guard < 1000 {
            guard += 1;
            let mut newpoint = newmin(&xx, &fxx);
            if newpoint.is_nan() || newpoint < lb || newpoint > ub {
                if (xx[0] - xx[2]).abs() < (xx[1] - xx[2]).abs() {
                    newpoint = (xx[1] + xx[2]) / 2.0;
                } else {
                    newpoint = (xx[0] + xx[2]) / 2.0;
                }
            }
            let (fnewpoint, _dum) = self.fam.gradfun(newpoint, dens, true, false);
            if fnewpoint > fxx[2] {
                if newpoint > xx[2] {
                    ub = newpoint;
                    for i in 0..3 {
                        if xx[i] > newpoint {
                            xx[i] = newpoint;
                            fxx[i] = fnewpoint;
                        }
                    }
                } else {
                    lb = newpoint;
                    for i in 0..3 {
                        if xx[i] < newpoint {
                            xx[i] = newpoint;
                            fxx[i] = fnewpoint;
                        }
                    }
                }
            } else {
                if xx[2] > newpoint {
                    ub = xx[2];
                    for i in 0..3 {
                        if xx[i] > xx[2] {
                            xx[i] = xx[2];
                            fxx[i] = fxx[2];
                        }
                    }
                    xx[2] = newpoint;
                    fxx[2] = fnewpoint;
                } else {
                    lb = xx[2];
                    for i in 0..3 {
                        if xx[i] < xx[2] {
                            xx[i] = xx[2];
                            fxx[i] = fxx[2];
                        }
                    }
                    xx[2] = newpoint;
                    fxx[2] = fnewpoint;
                }
            }
        }
        if fxx[2] < 0.0 {
            xx[2]
        } else {
            f64::NAN
        }
    }

    /// New-support-point search when the gradient's derivative is available
    /// (flag `d1`). Port of `solvegradd1`.
    fn solvegradd1(&self, dens: &[f64], tol: f64) -> Vec<f64> {
        let gp = &self.gridpoints;
        let l = gp.len();
        if l < 2 {
            return Vec::new();
        }
        let (_pv, pg) = self.fam.gradfunvec(gp, dens, false, true);
        let idx: Vec<usize> = (0..l - 1)
            .filter(|&i| pg[i] < 0.0 && pg[i + 1] > 0.0)
            .collect();
        let mut ans: Vec<f64> = idx
            .iter()
            .map(|&i| self.brmin(gp[i], gp[i + 1], dens, tol))
            .collect();
        if !ans.is_empty() {
            let (vals, _g2) = self.fam.gradfunvec(&ans, dens, true, false);
            ans = ans
                .into_iter()
                .zip(vals.into_iter())
                .filter(|(_, v)| *v < 0.0)
                .map(|(x, _)| x)
                .collect();
        }
        let (pv2, _s2) = self.fam.gradfun(gp[0], dens, true, false);
        if pv2 < 0.0 && pg[0] > 0.0 {
            ans.push(gp[0]);
        }
        let last = l - 1;
        let (pv2, _s2) = self.fam.gradfun(gp[last], dens, true, false);
        if pv2 < 0.0 && pg[last] < 0.0 {
            ans.push(gp[last]);
        }
        ans
    }

    /// New-support-point search for derivative-free gradients (flag `d0`).
    /// Port of `solvegradd0`.
    fn solvegradd0(&self, dens: &[f64], tol: f64) -> Vec<f64> {
        let gp = &self.gridpoints;
        let l = gp.len();
        if l < 3 {
            return Vec::new();
        }
        let (pv, _g) = self.fam.gradfunvec(gp, dens, true, false);
        // crossings j: pv[j+1]-pv[j] < 0 and pv[j+2]-pv[j+1] > 0
        let idx: Vec<usize> = (0..l - 2)
            .filter(|&j| (pv[j + 1] - pv[j]) < 0.0 && (pv[j + 2] - pv[j + 1]) > 0.0)
            .collect();
        let mut ans: Vec<f64> = idx
            .iter()
            .map(|&j| self.dfmin([gp[j], gp[j + 1], gp[j + 2]], [pv[j], pv[j + 1], pv[j + 2]], dens, tol))
            .filter(|x| !x.is_nan())
            .collect();
        if pv[0] < 0.0 && (pv[1] - pv[0]) > 0.0 {
            ans.push(gp[0]);
        }
        if pv[l - 1] < 0.0 && (pv[l - 1] - pv[l - 2]) < 0.0 {
            ans.push(gp[l - 1]);
        }
        ans
    }

    fn solvegrad(&self, dens: &[f64], tol: f64) -> Vec<f64> {
        if self.fam.flag() == "d1" {
            self.solvegradd1(dens, tol)
        } else {
            self.solvegradd0(dens, tol)
        }
    }

    /// Port of `collapse` (always called with the fixed tolerance 1e-6, as in
    /// the C++ `computemixdist`).
    fn collapse(&self, mu0: &mut Vec<f64>, pi0: &mut Vec<f64>) {
        let tol: f64 = 1e-6;
        let dens = self.fam.mapping(mu0, pi0);
        let ll = self.fam.lossfunction(&dens);
        let ntol = (tol * 0.1).max(ll * 1e-16);
        let mut mu0new = mu0.clone();
        let mut pi0new = pi0.clone();
        loop {
            if mu0new.len() <= 1 {
                break;
            }
            let d = diff(&mu0new);
            let prec = 10.0 * d.iter().copied().fold(f64::INFINITY, |m, x| m.min(x));
            collapsemix(&mut mu0new, &mut pi0new, prec);
            let densn = self.fam.mapping(&mu0new, &pi0new);
            let nll = self.fam.lossfunction(&densn);
            if nll <= ll + ntol {
                *mu0 = mu0new.clone();
                *pi0 = pi0new.clone();
            } else {
                break;
            }
        }
        simplifymix(mu0, pi0);
    }

    /// Port of `computemixdist` (the outer iteration loop).
    pub fn computemixdist(&mut self, tol: f64, maxit: i64) {
        let mut mu0 = self.initpt.clone();
        let scale = 1.0 - self.pi0fixed.iter().sum::<f64>();
        let mut pi0: Vec<f64> = self.initpr.iter().map(|&p| p * scale).collect();
        self.iter = 0;
        let mut dens = self.fam.mapping(&mu0, &pi0);
        let mut closs = self.fam.lossfunction(&dens);
        let mut nloss = f64::NAN;
        // env-gated phase profiler (NPFIXEDCOMPY_PROFILE=1)
        let prof = std::env::var("NPFIXEDCOMPY_PROFILE").is_ok();
        let mut pt_map = 0.0f64;
        let mut pt_loss = 0.0f64;
        let mut pt_grad = 0.0f64;
        let mut pt_wt = 0.0f64;
        let mut pt_col = 0.0f64;
        let mut t_iter_start = std::time::Instant::now();
        loop {
            let t0 = std::time::Instant::now();
            let newpoints = self.solvegrad(&dens, tol);
            mu0.extend(newpoints.clone());
            pi0.extend(std::iter::repeat(0.0).take(newpoints.len()));
            sortmix(&mut mu0, &mut pi0);
            pt_grad += t0.elapsed().as_secs_f64();

            if self.verbose >= 1 {
                eprintln!("Iteration: {} with loss {}", self.iter, nloss);
                eprintln!("support points: {:?}", mu0);
                eprintln!("probabilities: {:?}", pi0);
            }
            if self.verbose >= 2 {
                let (gv, gg) = self.fam.gradfunvec(&newpoints, &dens, true, true);
                eprintln!("new points: {:?}", newpoints);
                eprintln!("gradient: {:?}", gv);
                if self.fam.flag() == "d1" {
                    eprintln!("gradient derivative: {:?}", gg);
                }
                let d2 = self.fam.mapping(&mu0, &pi0);
                eprintln!("loss:{}", self.fam.lossfunction(&d2));
            }

            let t0 = std::time::Instant::now();
            self.fam.computeweights(&mu0, &mut pi0, &dens);
            pt_wt += t0.elapsed().as_secs_f64();
            if self.verbose >= 2 {
                eprintln!("After computeweights");
                eprintln!("support points: {:?}", mu0);
                eprintln!("probabilities: {:?}", pi0);
                let d2 = self.fam.mapping(&mu0, &pi0);
                eprintln!("loss:{}", self.fam.lossfunction(&d2));
            }
            let t0 = std::time::Instant::now();
            self.collapse(&mut mu0, &mut pi0);
            pt_col += t0.elapsed().as_secs_f64();
            if self.verbose >= 2 {
                eprintln!("After collapse");
                eprintln!("support points: {:?}", mu0);
                eprintln!("probabilities: {:?}", pi0);
                let d2 = self.fam.mapping(&mu0, &pi0);
                eprintln!("loss:{}", self.fam.lossfunction(&d2));
            }
            self.iter += 1;
            let t0 = std::time::Instant::now();
            dens = self.fam.mapping(&mu0, &pi0);
            pt_map += t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            nloss = self.fam.lossfunction(&dens);
            pt_loss += t0.elapsed().as_secs_f64();

            if closs - nloss < tol {
                self.convergence = 0;
                break;
            }
            if self.iter > maxit {
                self.convergence = 1;
                break;
            }
            closs = nloss;
        }
        if prof {
            let tot = t_iter_start.elapsed().as_secs_f64();
            eprintln!(
                "PROFILE iters={} total={:.1}ms  solvegrad={:.1}  mapping={:.1}  loss={:.1}  weights={:.1}  collapse={:.1} (ms)",
                self.iter,
                tot * 1e3,
                pt_grad * 1e3,
                pt_map * 1e3,
                pt_loss * 1e3,
                pt_wt * 1e3,
                pt_col * 1e3
            );
        }
        self.resultpt = mu0;
        self.resultpr = pi0;
    }

    /// Port of `estpi0`: estimate the point mass at zero by thresholding the
    /// hypothesis statistic. Note (faithful to the C++): after the first
    /// `computemixdist(tol)`, the inner `computemixdist` calls all use the
    /// defaults `tol = 1e-6, maxit = 100, verbose = 0`.
    pub fn estpi0(&mut self, val: f64, tol: f64) {
        self.mu0fixed = vec![0.0];
        self.pi0fixed = vec![0.0];
        self.set_precompute();
        self.computemixdist(tol, 100);
        let densmin = self.fam.mapping(&self.resultpt, &self.resultpr);
        let minloss = self.fam.lossfunction(&densmin) + self.fam.extrafun();
        let dens0 = self.fam.mapping(&[0.0], &[1.0]);
        let stat0 = self.fam.lossfunction(&dens0) + self.fam.extrafun();
        if self.fam.hypofun(stat0, minloss) < val {
            self.resultpt = vec![0.0];
            self.resultpr = vec![1.0];
        } else {
            let mut lb: f64 = 0.0;
            let mut ub: f64 = 1.0;
            let mut flb = minloss;
            let mut fub = stat0;
            let sp0 = self
                .fam
                .familydensity(0.0, &self.resultpt, &self.resultpr)
                / self.fam.familydensity(0.0, &[0.0], &[1.0]);
            self.pi0fixed = vec![sp0];
            self.set_precompute();
            self.computemixdist(1e-6, 100);
            let mut ll = {
                let d = self.fam.mapping(&self.resultpt, &self.resultpr);
                self.fam.lossfunction(&d) + self.fam.extrafun()
            };
            let mut sp = sp0;
            let mut iter = 1;
            let mut guard: i64 = 0;
            while (self.fam.hypofun(ll, minloss) - val).abs() > tol
                && (ub - lb).abs() > tol
                && guard < 1000
            {
                guard += 1;
                if self.verbose >= 1 {
                    eprintln!("Iter: {} lower: {} upper: {}", iter, lb, ub);
                    eprintln!("current val: {} fval: {}", sp, ll);
                }
                let h = self.fam.hypofun(ll, minloss) - val;
                if h < 0.0 && sp > lb {
                    lb = sp;
                    flb = ll;
                }
                if h > 0.0 && sp < ub {
                    ub = sp;
                    fub = ll;
                }
                sp = (lb + ub) / 2.0;

                self.initpt = self.resultpt.clone();
                self.initpr = self.resultpr.clone();
                self.pi0fixed = vec![sp];
                self.set_precompute();
                self.computemixdist(1e-6, 100);
                ll = {
                    let d = self.fam.mapping(&self.resultpt, &self.resultpr);
                    self.fam.lossfunction(&d) + self.fam.extrafun()
                };

                // quadratic interpolation through (lb, sp, ub)
                let a00 = lb * lb;
                let a01 = lb;
                let a10 = sp * sp;
                let a11 = sp;
                let a20 = ub * ub;
                let a21 = ub;
                // solve A x = b for the quadratic coefficients (C++ uses
                // A.inverse() * b; an LU solve is equivalent to ~1e-15)
                let b0 = self.fam.hypofun(flb, minloss) - val;
                let b1 = self.fam.hypofun(ll, minloss) - val;
                let b2 = self.fam.hypofun(fub, minloss) - val;
                // Cramer's rule for the 3x3 (determinant computed explicitly).
                // A = [[lb^2, lb, 1], [sp^2, sp, 1], [ub^2, ub, 1]], b = (b0,b1,b2)
                // (column 0 holds the t^2 coefficients, column 1 the t, column 2 the
                // constant), so `x10/x11/x12` below are (c2, c1, c0).
                let det = a00 * (a11 - a21) - a01 * (a10 - a20) + (a10 * a21 - a11 * a20);
                let det0 = b0 * (a11 - a21) - a01 * (b1 - b2) + (b1 * a21 - a11 * b2);
                let det1 = a00 * (b1 - b2) - b0 * (a10 - a20) + (a10 * b2 - b1 * a20);
                let det2 = a00 * (a11 * b2 - b1 * a21)
                    - a01 * (a10 * b2 - b1 * a20)
                    + b0 * (a10 * a21 - a11 * a20);
                let x10 = det0 / det;
                let x11 = det1 / det;
                let x12 = det2 / det;
                let disc = x11 * x11 - 4.0 * x10 * x12;
                let mut spnew = if disc < 0.0 {
                    f64::NAN
                } else {
                    (-x11 + disc.sqrt()) / (2.0 * x10)
                };
                if (self.fam.hypofun(ll, minloss) - val) < 0.0 {
                    lb = sp;
                    flb = ll;
                }
                if (self.fam.hypofun(ll, minloss) - val) > 0.0 {
                    ub = sp;
                    fub = ll;
                }
                if spnew.is_nan() || spnew < lb || spnew > ub {
                    spnew = (lb + ub) / 2.0;
                }
                sp = spnew;

                self.initpt = self.resultpt.clone();
                self.initpr = self.resultpr.clone();
                self.pi0fixed = vec![sp];
                self.set_precompute();
                self.computemixdist(1e-6, 100);
                ll = {
                    let d = self.fam.mapping(&self.resultpt, &self.resultpr);
                    self.fam.lossfunction(&d) + self.fam.extrafun()
                };
                iter += 1;
            }
        }
    }

        /// Opt-in fast variant of `estpi0` (NOT bit-identical to the C++ port;
        /// enable with `estpi0(..., fast=True)`).
        ///
        /// Same setup and semantics as the legacy path — `pi0fixed = 0`, initial
        /// `computemixdist(tol)`, the same null test, the same `sp0` start and the
        /// same quadratic/bisection bookkeeping — but every refinement iteration
        /// evaluates the inner problem only ONCE (the legacy path spends two
        /// evaluations per iteration: one bisection midpoint plus one quadratic
        /// point, both at the strict inner tolerance 1e-6). Additional options:
        ///
        /// * `relax`: early stop. The statistic `hypofun(ll, minloss)` is
        ///   monotonically increasing in the fixed `pi0`, so once it has already
        ///   exceeded `val` with a small `pi0`, pushing `pi0` up only moves it
        ///   further past the threshold: in that case the bisection would be
        ///   tracking a root that no longer exists inside the bracket, and we
        ///   simply increase `pi0` and stop (a conservative "no point mass at
        ///   zero beyond the test" answer, in O(1) inner solves).
        /// * `inner_tol`: tolerance for the inner `computemixdist` calls
        ///   (legacy uses 1e-6; e.g. 1e-4 is usually ample and several times
        ///   cheaper per iteration).
        pub fn estpi0_fast(&mut self, val: f64, tol: f64, inner_tol: f64, relax: bool) {
            self.mu0fixed = vec![0.0];
            self.pi0fixed = vec![0.0];
            self.set_precompute();
            self.computemixdist(tol, 100);
            let densmin = self.fam.mapping(&self.resultpt, &self.resultpr);
            let minloss = self.fam.lossfunction(&densmin) + self.fam.extrafun();
            let dens0 = self.fam.mapping(&[0.0], &[1.0]);
            let stat0 = self.fam.lossfunction(&dens0) + self.fam.extrafun();
            if self.fam.hypofun(stat0, minloss) < val {
                self.resultpt = vec![0.0];
                self.resultpr = vec![1.0];
                return;
            }
            let mut lb: f64 = 0.0;
            let mut ub: f64 = 1.0;
            let flb = minloss;
            let fub = stat0;
            let mut sp0 = self
                .fam
                .familydensity(0.0, &self.resultpt, &self.resultpr)
                / self.fam.familydensity(0.0, &[0.0], &[1.0]);
            let mut lb_loss = flb;
            let mut ub_loss = fub;
            let mut guard: i64 = 0;
            while guard < 1000 {
                guard += 1;
                // 1) evaluate the current candidate
                self.initpt = self.resultpt.clone();
                self.initpr = self.resultpr.clone();
                self.pi0fixed = vec![sp0];
                self.set_precompute();
                self.computemixdist(inner_tol, 100);
                let ll = {
                    let d = self.fam.mapping(&self.resultpt, &self.resultpr);
                    self.fam.lossfunction(&d) + self.fam.extrafun()
                };
                let h = self.fam.hypofun(ll, minloss) - val;
                if self.verbose >= 1 {
                    eprintln!("[fast] iter {} sp {} ll {} h {}", guard, sp0, ll, h);
                }
                // 2) converged, or early-stop when the target is already beaten
                //    from below with a small pi0 (relax mode)
                if h.abs() <= tol || (ub - lb).abs() <= tol {
                    break;
                }
                if relax && h > 0.0 && sp0 <= 0.1 * ub {
                    // The statistic is monotonically increasing in pi0, so any
                    // larger pi0 stays past `val`: report that value and stop.
                    self.pi0fixed = vec![sp0 * 2.0];
                    self.set_precompute();
                    self.computemixdist(inner_tol, 100);
                    break;
                }
                // 3) next candidate: quadratic interpolation through the three
                //    current bracket points (lb, sp0, ub) — same Cramer's-rule
                //    quadratic as the legacy path; fall back to the midpoint
                let (a00, a01) = (lb * lb, lb);
                let (a10, a11) = (sp0 * sp0, sp0);
                let (a20, a21) = (ub * ub, ub);
                let b0 = self.fam.hypofun(lb_loss, minloss) - val;
                let b1 = h;
                let b2 = self.fam.hypofun(ub_loss, minloss) - val;
                let det = a00 * (a11 - a21) - a01 * (a10 - a20) + (a10 * a21 - a11 * a20);
                let det0 = b0 * (a11 - a21) - a01 * (b1 - b2) + (b1 * a21 - a11 * b2);
                let det1 = a00 * (b1 - b2) - b0 * (a10 - a20) + (a10 * b2 - b1 * a20);
                let det2 = a00 * (a11 * b2 - b1 * a21)
                    - a01 * (a10 * b2 - b1 * a20)
                    + b0 * (a10 * a21 - a11 * a20);
                let x10 = det0 / det;
                let x11 = det1 / det;
                let x12 = det2 / det;
                let disc = x11 * x11 - 4.0 * x10 * x12;
                let spnew = if disc < 0.0 {
                    f64::NAN
                } else {
                    (-x11 + disc.sqrt()) / (2.0 * x10)
                };
                // 4) update the bracket with the evaluated point
                if h < 0.0 {
                    lb = sp0;
                    lb_loss = ll;
                } else {
                    ub = sp0;
                    ub_loss = ll;
                }
                if spnew.is_nan() || spnew <= lb || spnew >= ub {
                    sp0 = (lb + ub) / 2.0;
                } else {
                    sp0 = spnew;
                }
            }
        }

    /// Port of `get_ans`.
    pub fn finish(self) -> MixResult {
        let mut mu0new = self.resultpt.clone();
        let mut pi0new = self.resultpr.clone();
        mu0new.extend(self.mu0fixed.clone());
        pi0new.extend(self.pi0fixed.clone());
        sortmix(&mut mu0new, &mut pi0new);
        let dens = self.fam.mapping(&self.resultpt, &self.resultpr);
        let (maxgrad, _g2) = self.fam.gradfunvec(&self.resultpt, &dens, true, false);
        let min_gradient = maxgrad
            .iter()
            .copied()
            .fold(f64::INFINITY, |m, x| m.min(x));
        MixResult {
            pt: mu0new,
            pr: pi0new,
            beta: self.fam.beta_value(),
            family: self.fam.family_name().to_string(),
            min_gradient,
            ll: self.fam.lossfunction(&dens) + self.fam.extrafun(),
            flag: self.fam.flag().to_string(),
            iter: self.iter,
            convergence: self.convergence,
        }
    }
}
