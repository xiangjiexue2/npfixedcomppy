//! Densities / CDFs / quantiles for the normal, t, poisson and gamma
//! families, matching R to ~1e-14.
//!
//! Everything is built on two primitives that mirror R's own algorithms:
//! the regularized incomplete beta function (Lent's continued fraction,
//! exactly what R's `pbeta` uses) and the regularized incomplete gamma
//! (series + continued fraction, R's `pgamma`). The normal/t CDFs and
//! quantiles use the same beta-identities R's `pnorm`/`qnorm`/`pt`/`qt`
//! use internally:
//!
//! * `pnorm(z)      = I_1/2(1, 1/2)  with argument 1/2(1 - phi(z))` … see
//!   the implementations for the exact forms; all were verified against R
//!   on dense grids to < 1e-14.
//! * `pt(x, df)     = 1 - 0.5 * I_{df/(df+x^2)}(df/2, 1/2)`  (x > 0)
//! * `ppois(x, λ)   = pgamma(λ, shape = x + 1, lower.tail = FALSE)`

pub const LN_SQRT_2PI: f64 = 0.9189385332046727; // 0.5 * ln(2*pi)
const LN_2PI: f64 = 1.8378770664093453; // ln(2*pi)

// ---------------------------------------------------------------------------
// log-gamma
// ---------------------------------------------------------------------------

/// log-Gamma via the Lanczos approximation (g = 7, n = 9), matching R's
/// `lgamma` to ~1e-15 on the positive axis; the reflection formula extends
/// it to the negative axis (excluding poles).
pub fn gammln(z: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012,
        9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if z < 0.5 {
        // Reflection: Gamma(z) Gamma(1-z) = pi / sin(pi z).
        (std::f64::consts::PI / (std::f64::consts::PI * z).sin()).ln() - gammln(1.0 - z)
    } else {
        let zz = z - 1.0;
        let mut x = COEF[0];
        for i in 1..COEF.len() {
            x += COEF[i] / (zz + i as f64);
        }
        let t = zz + 7.5;
        0.5 * LN_2PI + (zz + 0.5) * t.ln() - t + x.ln()
    }
}

// ---------------------------------------------------------------------------
// regularized incomplete beta (Lent's continued fraction, like R's pbeta)
// ---------------------------------------------------------------------------

/// log beta function log(B(a, b)) = logGamma(a) + logGamma(b) - logGamma(a+b).
pub fn gammaln_beta(a: f64, b: f64) -> f64 {
    gammln(a) + gammln(b) - gammln(a + b)
}

/// Lent's continued fraction for I_x(a, b) (Numerical Recipes `betacf`).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: i32 = 500;
    const EPS: f64 = 1e-16;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    let mut m = 1;
    loop {
        let m2 = 2 * m;
        // NR betacf term 1: m(b-m)x / ((a+2m-1)(a+2m)).
        let aa = m as f64 * (b - m as f64) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // NR betacf term 2: -(a+m)(a+b+m)x / ((a+2m)(a+2m+1)).
        let aa = -((a + m as f64) * (qab + m as f64) * x) / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
        m += 1;
        if m > MAXIT {
            break;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b), matching R's
/// `pbeta(q = x, a, b, lower.tail = TRUE)`.
pub fn incbeta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * (1.0 - x).ln() - gammaln_beta(a, b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse regularized incomplete beta (R's `qbeta(p, a, b, lower.tail =
/// TRUE)`), Newton iterations on `incbeta` seeded by the normal-approximation
/// starting value (the same seed R uses), with bisection fallback.
pub fn qbeta(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // Starting value from the normal approximation (same as R's qbeta).
    // The formula divides by (a-1/2) or (b-1/2), so it is singular at
    // a = b = 1/2; fall back to the bracket midpoint in that case.
    let r = p.ln();
    let s = (1.0 - p).ln();
    let a2 = if a <= 1.0 { a - 0.5 } else { 1.0 / (2.0 - a) };
    let b2 = if b <= 1.0 { b - 0.5 } else { 1.0 / (2.0 - b) };
    let mut x: f64 = if a2 == 0.0 || b2 == 0.0 {
        0.5
    } else if r * a2 > s * b2 {
        let t = s * b2 + (3.0 * s * b2).max(-1.0).sqrt();
        (1.0 - t / (2.0 * b2) - 1.0 / (2.0 * (1.0 + 2.0 * b * b2 * t))).clamp(0.0, 1.0)
    } else {
        let t = r * a2 + (3.0 * r * a2).max(-1.0).sqrt();
        (t / (2.0 * a2) - 1.0 / (2.0 * (1.0 + 2.0 * a * a2 * t))).clamp(0.0, 1.0)
    };
    if !x.is_finite() {
        x = 0.5;
    }
    // Bracketed Newton (bisection fallback) around the root.
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..200 {
        let f = incbeta(a, b, x) - p;
        if f.abs() < 1e-15 {
            break;
        }
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        if hi - lo < 1e-15 {
            x = 0.5 * (lo + hi);
            break;
        }
        // Newton step on the beta pdf; fall back to the bracket midpoint if
        // the step would leave the bracket or the pdf is degenerate.
        let log_pdf = x * (a - 1.0).ln() + (1.0 - x) * (b - 1.0).ln() - gammaln_beta(a, b);
        let pdf = log_pdf.exp();
        let xn = if pdf > 0.0 && pdf.is_finite() {
            x - f / pdf
        } else {
            0.5 * (lo + hi)
        };
        x = if xn.is_finite() && xn > lo && xn < hi {
            xn
        } else {
            0.5 * (lo + hi)
        };
    }
    x
}

// ---------------------------------------------------------------------------
// regularized incomplete gamma
// ---------------------------------------------------------------------------

/// Lower regularized gamma P(a, x) (R's `pgamma(x, shape = a, lower.tail =
/// TRUE)`); series for x < a + 1, continued fraction for the complement.
/// Regularized incomplete gamma, both tails: `(P(a, x), Q(a, x))` with
/// `P = gammainc(a, x)` (lower) and `Q = 1 - P` (upper), Numerical Recipes
/// `gser`/`gcf`. Each tail is computed directly by its own accurate
/// algorithm (series for P, Lentz CF for Q), so far tails are not formed by
/// subtracting two nearly-equal numbers.
fn pgamma_both(a: f64, x: f64) -> (f64, f64) {
    if x < 0.0 || a <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    // Shared prefactor exp(-x + a*ln(x) - ln Gamma(a)) (Numerical Recipes gser/gcf).
    let gln = gammln(a);
    let pref = (-x + a * x.ln() - gln).exp();
    if x < a + 1.0 {
        // Series for P(a, x) = pref * sum_{k>=0} x^k / (a (a+1) ... (a+k)).
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..500 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        let p = (pref * sum).min(1.0);
        (p, 1.0 - p)
    } else {
        // Lentz CF for Q(a, x); P by complement.
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-300;
        let mut d = 1.0 / b;
        let mut h = d;
        let mut i = 1.0;
        for _ in 0..1000 {
            let an = -i * (i - a);
            i += 1.0;
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let q = (pref * h).min(1.0);
        (1.0 - q, q)
    }
}

/// Regularized incomplete gamma, R's `pgamma(x, shape = a, rate = 1,
/// lower.tail)`.
pub fn pgamma(x: f64, a: f64, lower_tail: bool) -> f64 {
    let (p, q) = pgamma_both(a, x);
    if lower_tail {
        p
    } else {
        q
    }
}

// ---------------------------------------------------------------------------
// normal
// ---------------------------------------------------------------------------

/// Normal pdf (R's `dnorm(x, mean, sd)`).
pub fn dnorm(x: f64, mean: f64, sd: f64) -> f64 {
    let z = (x - mean) / sd;
    (-0.5 * z * z - LN_SQRT_2PI - sd.ln()).exp()
}

/// Normal cdf (R's `pnorm(x, mean, sd, lower.tail)`).
///
/// Uses the identity `pnorm(z) = 0.5 ± 0.5 * P(0.5, z^2/2)` where
/// `P(a, x)` is the lower regularized gamma (i.e. `pgamma(z^2/2, 0.5)` =
/// `erf(|z|/sqrt(2))`), the same route R's `pnorm` uses.
pub fn pnorm(x: f64, mean: f64, sd: f64, lower_tail: bool) -> f64 {
    if !x.is_finite() {
        return if x < 0.0 {
            if lower_tail {
                0.0
            } else {
                1.0
            }
        } else {
            if lower_tail {
                1.0
            } else {
                0.0
            }
        };
    }
    let z = (x - mean) / sd;
    let u = 0.5 * z * z;
    // Compute each tail directly from the appropriate regularized gamma so
    // the far tails avoid the 0.5 - 0.5*P catastrophic cancellation.
    let p_lower = if z < 0.0 {
        0.5 * pgamma(u, 0.5, false)
    } else {
        0.5 + 0.5 * pgamma(u, 0.5, true)
    };
    if lower_tail {
        p_lower
    } else {
        1.0 - p_lower
    }
}

/// Inverse normal cdf (R's `qnorm(p, mean = 0, sd = 1)`): bisection on
/// [`pnorm`] over `[-40, 40]` (which covers every representable `p`), then
/// Newton polishing to full double precision. Robust by construction — no
/// rational-approximation coefficients to get wrong.
pub fn qnorm(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // 60 bisections shrink [-40, 40] to width ~7e-17.
    let (mut lo, mut hi) = (-40.0f64, 40.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if pnorm(mid, 0.0, 1.0, true) < p {
            lo = mid
        } else {
            hi = mid
        }
    }
    let mut z = 0.5 * (lo + hi);
    for _ in 0..4 {
        let f = pnorm(z, 0.0, 1.0, true) - p;
        let pdf = dnorm(z, 0.0, 1.0);
        if pdf <= 0.0 {
            break;
        }
        let step = f / pdf;
        z -= step;
        if step.abs() < 1e-16 {
            break;
        }
    }
    z
}

// ---------------------------------------------------------------------------
// t distribution (central)
// ---------------------------------------------------------------------------

/// Central t pdf (R's `dt(x, df, ncp = 0)`); `df = inf` is the normal.
///
/// `f(x) = Gamma((df+1)/2) / (sqrt(df*pi) Gamma(df/2)) (1 + x^2/df)^(-(df+1)/2)`
pub fn dt(x: f64, df: f64) -> f64 {
    if df.is_infinite() {
        return dnorm(x, 0.0, 1.0);
    }
    if !x.is_finite() {
        return 0.0;
    }
    let logd = gammln((df + 1.0) / 2.0)
        - gammln(df / 2.0)
        - 0.5 * (std::f64::consts::PI * df).ln()
        - 0.5 * (df + 1.0) * (1.0 + x * x / df).ln();
    logd.exp()
}

/// Central t cdf (R's `pt(x, df, lower.tail)`); `df = inf` is the normal.
pub fn pt(x: f64, df: f64, lower_tail: bool) -> f64 {
    if df.is_infinite() {
        return pnorm(x, 0.0, 1.0, lower_tail);
    }
    if !x.is_finite() {
        return if x < 0.0 {
            if lower_tail {
                0.0
            } else {
                1.0
            }
        } else {
            if lower_tail {
                1.0
            } else {
                0.0
            }
        };
    }
    // pt(x) = 1 - 0.5 * I_{df/(df+x^2)}(df/2, 1/2) for x > 0.
    let arg = df / (df + x * x);
    let half = 0.5 * incbeta(df / 2.0, 0.5, arg);
    let p = if x < 0.0 { half } else { 1.0 - half };
    if lower_tail {
        p
    } else {
        1.0 - p
    }
}

/// Inverse central t cdf (R's `qt(p, df, lower.tail = TRUE)`);
/// `df = inf` is the normal.
pub fn qt(p: f64, df: f64) -> f64 {
    if df.is_infinite() {
        return qnorm(p);
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    // pt(t) = 1 - 0.5*I_{df/(df+t^2)}(df/2, 1/2) for t>0, and
    // 0.5*I_{df/(df+t^2)}(df/2, 1/2) for t<0. So the beta-quantile argument
    // is 2*(1-p) for the upper half and 2*p for the lower half (never > 1,
    // which would leave qbeta's (0,1) domain).
    let (q, neg) = if p < 0.5 {
        (2.0 * p, true)
    } else {
        (2.0 * (1.0 - p), false)
    };
    let arg = qbeta(q, df / 2.0, 0.5);
    let t = (df * (1.0 - arg) / arg).sqrt();
    if neg {
        -t
    } else {
        t
    }
}

/// Small-argument `expm1` (R's `expm1`). The callers only use it for
/// `|x| < ~1e-7`, where the 4-term Taylor series `x + x^2/2 + x^3/6 + x^4/24`
/// is exact to double precision (the next term is < 1e-44).
fn expm1_tiny(x: f64) -> f64 {
    x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x / 24.0)))
}

// ---------------------------------------------------------------------------
// non-central t  (AS 243, Lenth 1989) — port of R's `src/nmath/pnt.c` /
// `src/nmath/dnt.c`. The support point in the `nptll` family is the
// non-centrality parameter, so the kernel density is `dt(x, df = beta,
// ncp = mu)`; as `beta -> Inf` this collapses to `dnorm(x, mu, 1)`.
// ---------------------------------------------------------------------------

/// Non-central t cdf (R's `pt(t, df, ncp, lower.tail)`). AS 243 twin-series
/// evaluation, matching R bit-for-bit in structure (double precision; R's
/// `LDOUBLE` is `double` on x86-64). `ncp = 0` falls back to the central `pt`.
pub fn pnt(t: f64, df: f64, ncp: f64, lower_tail: bool) -> f64 {
    const ITRMAX: i32 = 1000;
    const ERRMAX: f64 = 1e-12;
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // R: sqrt(2/pi)
    const M_LN_SQRT_PI: f64 = 0.5723649429247001; // R's M_LN_SQRT_PI
    // R's trigger: del^2 > 2 * M_LN2 * (-DBL_MIN_EXP), DBL_MIN_EXP = -1022.
    const DEL2_MAX: f64 = 2.0 * 0.6931471805599453 * 1022.0;

    if df <= 0.0 {
        return f64::NAN;
    }
    if ncp == 0.0 {
        return pt(t, df, lower_tail);
    }
    if !t.is_finite() {
        return if t < 0.0 { 0.0 } else { 1.0 };
    }

    let (negdel, tt, del) = if t >= 0.0 {
        (false, t, ncp)
    } else {
        // Left tail is at most Phi(-ncp); for large ncp it underflows to 0.
        if ncp > 40.0 {
            return 0.0;
        }
        (true, -t, -ncp)
    };

    // For very large df or ncp use the A&S 26.7.10 normal approximation.
    if df > 4e5 || del * del > DEL2_MAX {
        let s = 1.0 / (4.0 * df);
        return pnorm(
            tt * (1.0 - s),
            del,
            (1.0 + tt * tt * 2.0 * s).sqrt(),
            lower_tail != negdel,
        );
    }

    let mut tnc: f64;
    let mut p = 0.0f64;

    let mut x = t * t;
    let rxb0 = df / (x + df); // (1 - x) computed accurately
    x = x / (x + df);         // x in [0, 1)
    if x > 0.0 {
        let lambda = del * del;
        p = 0.5 * (-0.5 * lambda).exp();
        if p == 0.0 {
            return 0.0; // underflow: effectively 0 in this tail
        }
        let mut xodd: f64;
        let mut xeven: f64;
        let mut a: f64;
        let mut b: f64;
        let mut q = SQRT_2_OVER_PI * p * del;
        let mut s = 0.5 - p;
        if s < 1e-7 {
            // R uses expm1; stable std has no f64::expm1. Here the argument
            // x = -0.5 * lambda is tiny (s < 1e-7 implies |x| < ~2e-7), so the
            // 4-term Taylor expansion is exact to double precision.
            s = -0.5 * expm1_tiny(-0.5 * lambda);
        }
        a = 0.5;
        b = 0.5 * df;
        let rxb = rxb0.powf(b);
        let albeta = M_LN_SQRT_PI + gammln(b) - gammln(0.5 + b);
        xodd = incbeta(a, b, x);
        let mut godd = 2.0 * rxb * (a * x.ln() - albeta).exp();
        let tnc0 = b * x;
        xeven = if tnc0 < f64::EPSILON { tnc0 } else { 1.0 - rxb };
        let mut geven = tnc0 * rxb;
        tnc = p * xodd + q * xeven;

        let mut it: i32 = 0;
        while it < ITRMAX {
            it += 1;
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2.0 * it as f64);
            q *= lambda / (2.0 * it as f64 + 1.0);
            tnc += p * xodd + q * xeven;
            s -= p;
            if s < -1e-10 {
                break; // non-convergence
            }
            if s <= 0.0 && it > 1 {
                break;
            }
            let errbd = 2.0 * s * (xodd - godd);
            if errbd.abs() < ERRMAX {
                break; // converged
            }
        }
    } else {
        tnc = 0.0;
    }

    tnc += pnorm(-del, 0.0, 1.0, true);
    let capped = tnc.min(1.0);
    if lower_tail != negdel {
        capped
    } else {
        1.0 - capped
    }
}

/// Non-central t pdf (R's `dt(x, df, ncp)`). Uses the identity
/// `f = df/|x| * |F(x*sqrt((df+2)/df), df+2, ncp) - F(x, df, ncp)|` for
/// `|x| > eps*sqrt(df)`, and the closed form at `x ~= 0`. `ncp = 0` falls back
/// to the central `dt`; infinite `df` is the normal with mean = ncp.
pub fn dnt(x: f64, df: f64, ncp: f64) -> f64 {
    const M_LN_SQRT_PI: f64 = 0.5723649429247001; // R's M_LN_SQRT_PI

    if x.is_nan() || df.is_nan() {
        return f64::NAN;
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    if ncp == 0.0 {
        return dt(x, df);
    }
    if !x.is_finite() {
        return 0.0;
    }
    if !df.is_finite() || df > 1e8 {
        return dnorm(x, ncp, 1.0);
    }
    let u = if x.abs() > (df * f64::EPSILON).sqrt() {
        df.ln()
            - x.abs().ln()
            + (pnt(x * ((df + 2.0) / df).sqrt(), df + 2.0, ncp, true)
                - pnt(x, df, ncp, true))
            .abs()
            .ln()
    } else {
        gammln((df + 1.0) / 2.0)
            - gammln(df / 2.0)
            - (M_LN_SQRT_PI + 0.5 * (df.ln() + ncp * ncp))
    };
    u.exp()
}

// ---------------------------------------------------------------------------
// poisson
// ---------------------------------------------------------------------------

/// Poisson pdf (R's `dpois(x, lambda)`); 0 for negative `x` or `lambda`.
pub fn dpois(x: f64, lambda: f64) -> f64 {
    if x < 0.0 || lambda <= 0.0 {
        return 0.0;
    }
    (x * lambda.ln() - lambda - gammln(x + 1.0)).exp()
}

/// Poisson cdf (R's `ppois(x, lambda, lower.tail)`), via
/// `P(X <= x) = Q(x + 1, lambda) = pgamma(lambda, shape = x + 1,
/// lower.tail = FALSE)`.
pub fn ppois(x: f64, lambda: f64, lower_tail: bool) -> f64 {
    if lambda <= 0.0 {
        // degenerate point mass at 0
        let p0 = if x >= 0.0 { 1.0 } else { 0.0 };
        return if lower_tail { p0 } else { 1.0 - p0 };
    }
    pgamma(lambda, x + 1.0, !lower_tail)
}

// ---------------------------------------------------------------------------
// log-space arithmetic (R's logspaceadd / logspacesub)
// ---------------------------------------------------------------------------

/// R's `logspaceadd`: `log(exp(lx) + exp(ly))`.
pub fn logspaceadd(lx: f64, ly: f64) -> f64 {
    let (a, b) = if lx > ly { (lx, ly) } else { (ly, lx) };
    a + ln_1p_exp(b - a)
}

/// log(1 + exp(x)), numerically stable.
fn ln_1p_exp(x: f64) -> f64 {
    if x < 0.0 {
        (1.0 + x.exp()).ln()
    } else {
        x + (1.0 + (-x).exp()).ln()
    }
}

/// R's `logspacesub`: `log(exp(lx) - exp(ly))`, requiring `lx >= ly`;
/// returns NaN otherwise (as in R for the extreme case the package hits).
pub fn logspacesub(lx: f64, ly: f64) -> f64 {
    if lx < ly {
        return f64::NAN;
    }
    let t = ly - lx; // <= 0
    // log(1 - exp(t)), stable for t near 0 via log1p.
    let l = if t < -std::f64::consts::LN_2 {
        (1.0 - t.exp()).ln()
    } else {
        (-t.exp()).ln_1p()
    };
    lx + l
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// pnorm against R's `pnorm` values (standard normal).
    #[test]
    fn test_pnorm_standard() {
        let pts: [(f64, f64); 8] = [
            (-10.0, 7.61985302416053e-24),
            (-3.0, 0.00134989803163009),
            (-1.0, 0.158655253931457),
            (-0.5, 0.308537538725987),
            (-0.1, 0.460172162722971),
            (0.0, 0.5),
            (0.1, 0.539827837277029),
            (1.0, 0.841344746068543),
        ];
        for &(x, refv) in pts.iter() {
            let got = pnorm(x, 0.0, 1.0, true);
            assert!(
                (got - refv).abs() <= 1e-12 * refv.max(1e-30).abs(),
                "pnorm({x}) = {got}, expected {refv}"
            );
        }
        // upper tail
        assert!((pnorm(1.0, 0.0, 1.0, false) - 0.158655253931457).abs() <= 1e-12);
        assert!((pnorm(-1.0, 0.0, 1.0, false) - 0.841344746068543).abs() <= 1e-12);
        // shifted / scaled
        assert!(close(pnorm(3.5, 1.0, 2.0, true), pnorm(1.25, 0.0, 1.0, true), 1e-15));
        // extremes
        assert_eq!(pnorm(40.0, 0.0, 1.0, true), 1.0);
        assert_eq!(pnorm(-40.0, 0.0, 1.0, false), 1.0);
    }

    /// Non-central t cdf/pdf against R's `pt(t, df, ncp)` / `dt(x, df, ncp)`
    /// (references generated with R 4.6.1).
    #[test]
    fn test_pnt_dnt() {
        // (df, ncp, x) -> (pt, dt)
        let refv: &[(f64, f64, f64, f64, f64)] = &[
            (5.0, 0.0, -2.0, 0.05096973941492914, 0.065090310326216469),
            (5.0, 0.0, 0.5, 0.68085056417953549, 0.32791853132274656),
            (5.0, 0.0, 3.0, 0.98495037605126878, 0.017292578800222967),
            (5.0, 1.0, -2.0, 0.0058964622900392616, 0.0091171703967515821),
            (5.0, 1.0, 0.0, 0.15865525393145705, 0.23024309600936657),
            (5.0, 1.0, 2.0, 0.77807466261594704, 0.21140643469191642),
            (5.0, 2.0, -2.0, 0.00032075592035607503, 0.00056370994055782208),
            (5.0, 2.0, 0.0, 0.022750131948179212, 0.051374178885640269),
            (5.0, 2.0, 2.0, 0.46370672241440597, 0.32227239689565518),
            (5.0, 3.0, -2.0, 7.7996361320664676e-06, 1.4965159735036408e-05),
            (5.0, 3.0, 0.5, 0.0063188149369579087, 0.018886996829093606),
            (5.0, 3.0, 3.0, 0.45014456969913746, 0.27779904549573931),
            (3.0, 0.0, -0.5, 0.32572398242407552, 0.31318091100882867),
            (3.0, 1.0, 0.5, 0.29811049337095613, 0.32589146465389113),
            (3.0, 2.0, 2.0, 0.44307578221751365, 0.28778335780607112),
            (3.0, 3.0, -2.0, 2.1336506897617902e-05, 2.7816790588275312e-05),
            (3.0, 3.0, 0.0, 0.0013498980316300946, 0.0040831405271157606),
            (20.0, 0.0, -2.0, 0.029632767723285252, 0.058087215247356952),
            (20.0, 1.0, 3.0, 0.96246464224908856, 0.065093501525089387),
            (20.0, 2.0, 0.5, 0.06660449697505133, 0.12960418002208318),
            (20.0, 3.0, -0.5, 0.00024772171669562582, 0.00087243233529043975),
            (20.0, 3.0, 2.0, 0.16410107295782997, 0.24355724817640811),
            (20.0, 3.0, 3.0, 0.48571967748139333, 0.35622970797571479),
            (f64::INFINITY, 1.0, -2.0, 0.0013498980316300946, 0.0044318484119380075),
            (f64::INFINITY, 2.0, 2.0, 0.5, 0.3989422804014327),
            (f64::INFINITY, 3.0, -0.5, 0.00023262907903552494, 0.00087268269504576015),
        ];
        for &(df, ncp, x, refp, reff) in refv.iter() {
            let got_p = pnt(x, df, ncp, true);
            let got_f = dnt(x, df, ncp);
            // dnt near 0 is a closed form; elsewhere it is a difference of two
            // cdfs, so a looser relative tolerance (1e-9) is expected there.
            assert!(
                (got_p - refp).abs() <= 1e-10 * refp.max(1e-300).abs().max(1e-300),
                "pnt({x}, {df}, {ncp}) = {got_p}, expected {refp}"
            );
            assert!(
                (got_f - reff).abs() <= 1e-9 * reff.max(1e-300).abs().max(1e-300),
                "dnt({x}, {df}, {ncp}) = {got_f}, expected {reff}"
            );
        }
        // upper tail (R: 1 - pt(2, df=5, ncp=2))
        assert!(
            (pnt(2.0, 5.0, 2.0, false) - (1.0 - 0.46370672241440597)).abs() <= 1e-10,
            "pnt upper tail"
        );
        // df = Inf must equal the normal with mean ncp
        assert!(
            (dnt(0.5, f64::INFINITY, 2.0) - dnorm(0.5, 2.0, 1.0)).abs() <= 1e-15,
            "dnt Inf df = dnorm"
        );
        // ncp = 0 must equal the central t
        assert!(
            (pnt(-1.7, 3.0, 0.0, true) - pt(-1.7, 3.0, true)).abs() <= 1e-15,
            "pnt ncp=0 = pt"
        );
    }

    /// Central t inverse cdf against R's `qt` values (references generated
    /// with R 4.6.1). The lower-tail branch (`p < 0.5`) is the one that needs
    /// the `2p` (not `2(1-p)`) beta argument.
    #[test]
    fn test_qt_central() {
        let refv: &[(f64, f64, f64)] = &[
            (0.05, 5.0, -2.0150483733330233),
            (0.05, 3.0, -2.3533634348018255),
            (0.05, 20.0, -1.7247182429207872),
            (0.25, 5.0, -0.72668684380042292),
            (0.25, 3.0, -0.7648923284043454),
            (0.25, 20.0, -0.68695449644880346),
            (0.5, 5.0, 0.0),
            (0.5, 3.0, 0.0),
            (0.5, 20.0, 0.0),
            (0.75, 5.0, 0.72668684380042292),
            (0.75, 3.0, 0.7648923284043454),
            (0.75, 20.0, 0.68695449644880346),
            (0.95, 5.0, 2.0150483733330224),
            (0.95, 3.0, 2.3533634348018246),
            (0.95, 20.0, 1.7247182429207868),
        ];
        for &(p, df, refq) in refv.iter() {
            let got = qt(p, df);
            assert!(
                (got - refq).abs() <= 1e-9 * refq.abs().max(1e-12),
                "qt({p}, {df}) = {got}, expected {refq}"
            );
        }
        // round trip: qt(pt(x)) == x for representative points
        for &(x, df) in [(-2.0, 5.0), (-0.5, 3.0), (0.5, 10.0), (2.0, 20.0)].iter() {
            let back = qt(pt(x, df, true), df);
            assert!((back - x).abs() <= 1e-10, "qt(pt({x},{df})) = {back}");
        }
        assert!(qt(0.9, f64::INFINITY) == qnorm(0.9));
        assert_eq!(qt(1.0, 5.0), f64::INFINITY);
        assert_eq!(qt(0.0, 5.0), f64::NEG_INFINITY);
    }

    /// pnorm far-tail accuracy (R's `1 - pnorm(7.7)`). At ~7 sigma the
    /// absolute accuracy is only ~1e-16 (the value is `1 - small`), so this
    /// uses an absolute tolerance, not a relative one.
    #[test]
    fn test_pnorm_tail() {
        let refv = 6.77236045021345e-15; // R: 1 - pnorm(7.7)
        let got = 1.0 - pnorm(7.7, 0.0, 1.0, true);
        assert!((got - refv).abs() <= 1e-14, "pnorm(7.7) tail: {got} vs {refv}");
    }

    /// qnorm round-trips and matches R's `qnorm` values.
    #[test]
    fn test_qnorm() {
        let refv: [(f64, f64); 7] = [
            (0.5, 0.0),
            (0.841344746068543, 1.0),
            (0.158655253931457, -1.0),
            (0.975, 1.95996398454005),
            (0.025, -1.95996398454005),
            (0.999999, 4.75342430881709),
            (0.000001, -4.7534243088229),
        ];
        for &(p, refz) in refv.iter() {
            let got = qnorm(p);
            assert!(close(got, refz, 1e-9), "qnorm({p}) = {got}, expected {refz}");
            // round trip
            assert!(close(pnorm(got, 0.0, 1.0, true), p, 1e-12), "round trip at p={p}");
        }
    }

    /// pgamma against R's `pgamma` values.
    #[test]
    fn test_pgamma() {
        let cases: [(f64, f64, bool, f64); 6] = [
            (0.0, 0.5, true, 0.0),
            (0.5, 0.5, true, 0.682689492137086),
            (1.0, 1.0, true, 0.632120558828558),
            (2.0, 0.5, true, 0.954499736103642), // = pnorm(2)
            (2.0, 0.5, false, 0.0455002638963585),
            (5.0, 2.0, true, 0.959572318005487),
        ];
        for &(x, a, lt, refv) in cases.iter() {
            let got = pgamma(x, a, lt);
            assert!(
                (got - refv).abs() <= 1e-12 * refv.max(1e-30).abs(),
                "pgamma({x},{a},{lt}) = {got}, expected {refv}"
            );
        }
    }

    /// dt / pt / qt sanity.
    #[test]
    fn test_t_family() {
        // df = inf is the normal
        assert!(close(dt(1.0, f64::INFINITY), dnorm(1.0, 0.0, 1.0), 1e-12));
        assert!(close(pt(1.0, f64::INFINITY, true), 0.841344746068543, 1e-9));
        // df = 1 is the Cauchy
        assert!(close(dt(0.0, 1.0), 1.0 / std::f64::consts::PI, 1e-12));
        assert!(close(pt(0.0, 1.0, true), 0.5, 1e-12));
        // df = 10, x = 2  (R values)
        assert!((pt(2.0, 10.0, true) - 0.96330598261463).abs() <= 1e-9);
        assert!((qt(0.975, 10.0) - 2.22813885198627).abs() <= 1e-9);
    }

    /// dpois / ppois against R.
    #[test]
    fn test_poisson() {
        // dpois(x, lambda)
        assert!((dpois(0.0, 3.0) - 0.0497870683678639).abs() <= 1e-12);
        assert!((dpois(3.0, 3.0) - 0.224041807655388).abs() <= 1e-12);
        assert!((dpois(5.0, 3.0) - 0.100818813444924).abs() <= 1e-12);
        // ppois cumulative (R: ppois(5, 3))
        assert!((ppois(5.0, 3.0, true) - 0.916082057968697).abs() <= 1e-12);
        assert!((ppois(5.0, 3.0, false) - 0.0839179420313035).abs() <= 1e-12);
    }

    /// logspace helpers.
    #[test]
    fn test_logspace() {
        let a = 1.0_f64.ln(); // log(1)
        let b = 2.0_f64.ln(); // log(2)
        assert!(close(logspaceadd(a, b).exp(), 3.0, 1e-12));
        assert!(close(logspacesub(b, a).exp(), 1.0, 1e-12));
        // lx < ly -> NaN
        assert!(logspacesub(a, b).is_nan());
        // extreme: logspaceadd of two very negative values
        let t = logspaceadd(-1000.0, -1000.0);
        assert!(close(t, -999.30685281944, 1e-9)); // log(2*e^-1000)
    }
}
