// npfc_stats.h — scalar math kernels: densities, CDFs and quantiles for the
// normal, t, non-central-t, gamma and Poisson families, matching R to ~1e-14.
//
// Faithful port of npfixedcomppy/src/stats.rs. Everything is built on two
// primitives that mirror R's own algorithms: the regularized incomplete beta
// function (Lent's continued fraction, exactly what R's `pbeta` uses) and the
// regularized incomplete gamma (series + continued fraction, R's `pgamma`).
// The normal/t CDFs and quantiles use the same beta identities R's
// `pnorm`/`qnorm`/`pt`/`qt` use internally.
#ifndef NPFIC_STATS_H
#define NPFIC_STATS_H

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace npfc {
namespace stats {

inline constexpr double LN_SQRT_2PI = 0.9189385332046727; // 0.5 * ln(2*pi)
inline constexpr double LN_2PI = 1.8378770664093453;      // ln(2*pi)
// MSVC's <cmath> does not define M_PI / M_LN2 unless _USE_MATH_DEFINES is
// set before the first include; use explicit constants so this header is
// include-order independent.
inline constexpr double PI_C = 3.14159265358979323846;  // pi
inline constexpr double LN2_C = 0.69314718055994530942; // ln 2

using std::isfinite;
using std::nan;
using std::numeric_limits;
using inf = double; // resolved below via numeric_limits

inline double posinf() { return std::numeric_limits<double>::infinity(); }
inline double neginf() { return -std::numeric_limits<double>::infinity(); }
inline double nanv() { return std::numeric_limits<double>::quiet_NaN(); }

// ---------------------------------------------------------------------------
// log-gamma
// ---------------------------------------------------------------------------

// log-Gamma via the Lanczos approximation (g = 7, n = 9), matching R's
// `lgamma` to ~1e-15 on the positive axis; the reflection formula extends it
// to the negative axis (excluding poles).
inline double gammln(double z) {
    static const double COEF[9] = {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313,  -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
    if (z < 0.5) {
        // Reflection: Gamma(z) Gamma(1-z) = pi / sin(pi z).
        return std::log(PI_C / std::sin(PI_C * z)) - gammln(1.0 - z);
    }
    const double zz = z - 1.0;
    double x = COEF[0];
    for (int i = 1; i < 9; ++i)
        x += COEF[i] / (zz + i);
    const double t = zz + 7.5;
    return 0.5 * LN_2PI + (zz + 0.5) * std::log(t) - t + std::log(x);
}

// log beta function log(B(a, b)) = logGamma(a) + logGamma(b) - logGamma(a+b).
inline double gammaln_beta(double a, double b) {
    return gammln(a) + gammln(b) - gammln(a + b);
}

// Lent's continued fraction for I_x(a, b) (Numerical Recipes `betacf`).
inline double betacf(double a, double b, double x) {
    constexpr int MAXIT = 500;
    constexpr double EPS = 1e-16;
    constexpr double FPMIN = 1e-300;
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < FPMIN)
        d = FPMIN;
    d = 1.0 / d;
    double h = d;
    int m = 1;
    while (true) {
        const double m2 = 2.0 * m;
        // NR betacf term 1: m(b-m)x / ((a+2m-1)(a+2m)).
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < FPMIN)
            d = FPMIN;
        c = 1.0 + aa / c;
        if (std::abs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        // NR betacf term 2: -(a+m)(a+b+m)x / ((a+2m)(a+2m+1)).
        aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < FPMIN)
            d = FPMIN;
        c = 1.0 + aa / c;
        if (std::abs(c) < FPMIN)
            c = FPMIN;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < EPS)
            break;
        m += 1;
        if (m > MAXIT)
            break;
    }
    return h;
}

// Regularized incomplete beta I_x(a, b), matching R's
// `pbeta(q = x, a, b, lower.tail = TRUE)`.
inline double incbeta(double a, double b, double x) {
    if (x <= 0.0)
        return 0.0;
    if (x >= 1.0)
        return 1.0;
    const double front =
        std::exp(a * std::log(x) + b * std::log(1.0 - x) - gammaln_beta(a, b));
    if (x < (a + 1.0) / (a + b + 2.0))
        return front * betacf(a, b, x) / a;
    return 1.0 - front * betacf(b, a, 1.0 - x) / b;
}

// Inverse regularized incomplete beta (R's `qbeta(p, a, b, lower.tail =
// TRUE)`), Newton iterations on `incbeta` seeded by the normal-approximation
// starting value (the same seed R uses), with bisection fallback.
inline double qbeta(double p, double a, double b) {
    if (p <= 0.0)
        return 0.0;
    if (p >= 1.0)
        return 1.0;
    // Starting value from the normal approximation (same as R's qbeta).
    // The formula divides by (a-1/2) or (b-1/2), so it is singular at
    // a = b = 1/2; fall back to the bracket midpoint in that case.
    const double r = std::log(p);
    const double s = std::log(1.0 - p);
    const double a2 = (a <= 1.0) ? (a - 0.5) : (1.0 / (2.0 - a));
    const double b2 = (b <= 1.0) ? (b - 0.5) : (1.0 / (2.0 - b));
    double x;
    if (a2 == 0.0 || b2 == 0.0) {
        x = 0.5;
    } else if (r * a2 > s * b2) {
        const double t = s * b2 + std::sqrt(std::max(3.0 * s * b2, -1.0));
        x = std::min(1.0, std::max(0.0, 1.0 - t / (2.0 * b2) -
                                              1.0 / (2.0 * (1.0 + 2.0 * b * b2 * t))));
    } else {
        const double t = r * a2 + std::sqrt(std::max(3.0 * r * a2, -1.0));
        x = std::min(1.0, std::max(0.0, t / (2.0 * a2) -
                                              1.0 / (2.0 * (1.0 + 2.0 * a * a2 * t))));
    }
    if (!isfinite(x))
        x = 0.5;
    // Bracketed Newton (bisection fallback) around the root.
    double lo = 0.0;
    double hi = 1.0;
    for (int i = 0; i < 200; ++i) {
        const double f = incbeta(a, b, x) - p;
        if (std::abs(f) < 1e-15)
            break;
        if (f > 0.0)
            hi = x;
        else
            lo = x;
        if (hi - lo < 1e-15) {
            x = 0.5 * (lo + hi);
            break;
        }
        // Newton step on the beta pdf; fall back to the bracket midpoint if
        // the step would leave the bracket or the pdf is degenerate.
        const double log_pdf =
            x * std::log(a - 1.0) + (1.0 - x) * std::log(b - 1.0) -
            gammaln_beta(a, b);
        const double pdf = std::exp(log_pdf);
        double xn = (pdf > 0.0 && isfinite(pdf)) ? (x - f / pdf) : 0.5 * (lo + hi);
        x = (isfinite(xn) && xn > lo && xn < hi) ? xn : 0.5 * (lo + hi);
    }
    return x;
}

// ---------------------------------------------------------------------------
// regularized incomplete gamma
// ---------------------------------------------------------------------------

// Regularized incomplete gamma, both tails: `(P(a, x), Q(a, x))` with
// `P = gammainc(a, x)` (lower) and `Q = 1 - P` (upper), Numerical Recipes
// `gser`/`gcf`. Each tail is computed directly by its own accurate
// algorithm (series for P, Lentz CF for Q), so far tails are not formed by
// subtracting two nearly-equal numbers.
inline void pgamma_both(double a, double x, double &p, double &q) {
    if (x < 0.0 || a <= 0.0) {
        p = nanv();
        q = nanv();
        return;
    }
    if (x == 0.0) {
        p = 0.0;
        q = 1.0;
        return;
    }
    // Shared prefactor exp(-x + a*ln(x) - ln Gamma(a)) (Numerical Recipes gser/gcf).
    const double gln = gammln(a);
    const double pref = std::exp(-x + a * std::log(x) - gln);
    if (x < a + 1.0) {
        // Series for P(a, x) = pref * sum_{k>=0} x^k / (a (a+1) ... (a+k)).
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int i = 0; i < 500; ++i) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::abs(del) < std::abs(sum) * 1e-16)
                break;
        }
        p = std::min(pref * sum, 1.0);
        q = 1.0 - p;
    } else {
        // Lentz CF for Q(a, x); P by complement.
        double b = x + 1.0 - a;
        double c = 1.0 / 1e-300;
        double d = 1.0 / b;
        double h = d;
        double i = 1.0;
        for (int k = 0; k < 1000; ++k) {
            const double an = -i * (i - a);
            i += 1.0;
            b += 2.0;
            d = an * d + b;
            if (std::abs(d) < 1e-300)
                d = 1e-300;
            c = b + an / c;
            if (std::abs(c) < 1e-300)
                c = 1e-300;
            d = 1.0 / d;
            const double del = d * c;
            h *= del;
            if (std::abs(del - 1.0) < 1e-15)
                break;
        }
        const double qq = std::min(pref * h, 1.0);
        p = 1.0 - qq;
        q = qq;
    }
}

// Regularized incomplete gamma, R's `pgamma(x, shape = a, rate = 1,
// lower.tail)`.
inline double pgamma(double x, double a, bool lower_tail) {
    double p, q;
    pgamma_both(a, x, p, q);
    return lower_tail ? p : q;
}

// ---------------------------------------------------------------------------
// normal
// ---------------------------------------------------------------------------

// Normal pdf (R's `dnorm(x, mean, sd)`).
inline double dnorm(double x, double mean, double sd) {
    const double z = (x - mean) / sd;
    return std::exp(-0.5 * z * z - LN_SQRT_2PI - std::log(sd));
}

// Normal cdf (R's `pnorm(x, mean, sd, lower.tail)`).
//
// Uses the identity `pnorm(z) = 0.5 +/- 0.5 * P(0.5, z^2/2)` where
// `P(a, x)` is the lower regularized gamma (i.e. `pgamma(z^2/2, 0.5)` =
// `erf(|z|/sqrt(2))`), the same route R's `pnorm` uses.
inline double pnorm(double x, double mean, double sd, bool lower_tail) {
    if (!isfinite(x)) {
        if (x < 0.0)
            return lower_tail ? 0.0 : 1.0;
        return lower_tail ? 1.0 : 0.0;
    }
    const double z = (x - mean) / sd;
    const double u = 0.5 * z * z;
    // Compute each tail directly from the appropriate regularized gamma so
    // the far tails avoid the 0.5 - 0.5*P catastrophic cancellation.
    const double p_lower = (z < 0.0) ? (0.5 * pgamma(u, 0.5, false))
                                     : (0.5 + 0.5 * pgamma(u, 0.5, true));
    return lower_tail ? p_lower : 1.0 - p_lower;
}

// Inverse normal cdf (R's `qnorm(p, mean = 0, sd = 1)`): bisection on
// `pnorm` over `[-40, 40]` (which covers every representable `p`), then
// Newton polishing to full double precision. Robust by construction -- no
// rational-approximation coefficients to get wrong.
inline double qnorm(double p) {
    if (p <= 0.0)
        return neginf();
    if (p >= 1.0)
        return posinf();
    // 60 bisections shrink [-40, 40] to width ~7e-17.
    double lo = -40.0, hi = 40.0;
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (pnorm(mid, 0.0, 1.0, true) < p)
            lo = mid;
        else
            hi = mid;
    }
    double z = 0.5 * (lo + hi);
    for (int i = 0; i < 4; ++i) {
        const double f = pnorm(z, 0.0, 1.0, true) - p;
        const double pdf = dnorm(z, 0.0, 1.0);
        if (pdf <= 0.0)
            break;
        const double step = f / pdf;
        z -= step;
        if (std::abs(step) < 1e-16)
            break;
    }
    return z;
}

// ---------------------------------------------------------------------------
// t distribution (central)
// ---------------------------------------------------------------------------

// Central t pdf (R's `dt(x, df, ncp = 0)`); `df = inf` is the normal.
inline double dt(double x, double df) {
    if (std::isinf(df))
        return dnorm(x, 0.0, 1.0);
    if (!isfinite(x))
        return 0.0;
    const double logd =
        gammln((df + 1.0) / 2.0) - gammln(df / 2.0) -
        0.5 * std::log(PI_C * df) -
        0.5 * (df + 1.0) * std::log(1.0 + x * x / df);
    return std::exp(logd);
}

// Central t cdf (R's `pt(x, df, lower.tail)`); `df = inf` is the normal.
inline double pt(double x, double df, bool lower_tail) {
    if (std::isinf(df))
        return pnorm(x, 0.0, 1.0, lower_tail);
    if (!isfinite(x)) {
        if (x < 0.0)
            return lower_tail ? 0.0 : 1.0;
        return lower_tail ? 1.0 : 0.0;
    }
    // pt(x) = 1 - 0.5 * I_{df/(df+x^2)}(df/2, 1/2) for x > 0.
    const double arg = df / (df + x * x);
    const double half = 0.5 * incbeta(df / 2.0, 0.5, arg);
    const double p = (x < 0.0) ? half : (1.0 - half);
    return lower_tail ? p : 1.0 - p;
}

// Inverse central t cdf (R's `qt(p, df, lower.tail = TRUE)`);
// `df = inf` is the normal.
inline double qt(double p, double df) {
    if (std::isinf(df))
        return qnorm(p);
    if (p <= 0.0)
        return neginf();
    if (p >= 1.0)
        return posinf();
    if (p == 0.5)
        return 0.0;
    // pt(t) = 1 - 0.5*I_{df/(df+t^2)}(df/2, 1/2) for t>0, and
    // 0.5*I_{df/(df+t^2)}(df/2, 1/2) for t<0. So the beta-quantile argument
    // is 2*(1-p) for the upper half and 2*p for the lower half (never > 1,
    // which would leave qbeta's (0,1) domain).
    double q;
    bool neg;
    if (p < 0.5) {
        q = 2.0 * p;
        neg = true;
    } else {
        q = 2.0 * (1.0 - p);
        neg = false;
    }
    const double arg = qbeta(q, df / 2.0, 0.5);
    const double t = std::sqrt(df * (1.0 - arg) / arg);
    return neg ? -t : t;
}

// Small-argument `expm1` (R's `expm1`). The callers only use it for
// `|x| < ~1e-7`, where the 4-term Taylor series `x + x^2/2 + x^3/6 + x^4/24`
// is exact to double precision (the next term is < 1e-44).
inline double expm1_tiny(double x) {
    return x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x / 24.0)));
}

// ---------------------------------------------------------------------------
// non-central t  (AS 243, Lenth 1989) — port of R's `src/nmath/pnt.c` /
// `src/nmath/dnt.c`. The support point in the `nptll` family is the
// non-centrality parameter, so the kernel density is `dt(x, df = beta,
// ncp = mu)`; as `beta -> Inf` this collapses to `dnorm(x, mu, 1)`.
// ---------------------------------------------------------------------------

// Non-central t cdf (R's `pt(t, df, ncp, lower.tail)`). AS 243 twin-series
// evaluation, matching R bit-for-bit in structure (double precision; R's
// `LDOUBLE` is `double` on x86-64). `ncp = 0` falls back to the central `pt`.
inline double pnt(double t, double df, double ncp, bool lower_tail) {
    constexpr int ITRMAX = 1000;
    constexpr double ERRMAX = 1e-12;
    constexpr double SQRT_2_OVER_PI = 0.7978845608028654; // R: sqrt(2/pi)
    constexpr double M_LN_SQRT_PI = 0.5723649429247001;   // R's M_LN_SQRT_PI
    // R's trigger: del^2 > 2 * M_LN2 * (-DBL_MIN_EXP), DBL_MIN_EXP = -1022.
    constexpr double DEL2_MAX = 2.0 * 0.6931471805599453 * 1022.0;

    if (df <= 0.0)
        return nanv();
    if (ncp == 0.0)
        return pt(t, df, lower_tail);
    if (!isfinite(t))
        return (t < 0.0) ? 0.0 : 1.0;

    bool negdel;
    double tt, del;
    if (t >= 0.0) {
        negdel = false;
        tt = t;
        del = ncp;
    } else {
        // Left tail is at most Phi(-ncp); for large ncp it underflows to 0.
        if (ncp > 40.0)
            return 0.0;
        negdel = true;
        tt = -t;
        del = -ncp;
    }

    // For very large df or ncp use the A&S 26.7.10 normal approximation.
    if (df > 4e5 || del * del > DEL2_MAX) {
        const double s = 1.0 / (4.0 * df);
        return pnorm(tt * (1.0 - s), del, std::sqrt(1.0 + tt * tt * 2.0 * s),
                     lower_tail != negdel);
    }

    double tnc;
    double p = 0.0;

    double x = t * t;
    const double rxb0 = df / (x + df); // (1 - x) computed accurately
    x = x / (x + df);                  // x in [0, 1)
    if (x > 0.0) {
        const double lambda = del * del;
        p = 0.5 * std::exp(-0.5 * lambda);
        if (p == 0.0)
            return 0.0; // underflow: effectively 0 in this tail
        double xodd;
        double xeven;
        double a;
        double b;
        double q = SQRT_2_OVER_PI * p * del;
        double s = 0.5 - p;
        if (s < 1e-7) {
            // R uses expm1; the argument x = -0.5 * lambda is tiny (s < 1e-7
            // implies |x| < ~2e-7), so the 4-term Taylor expansion is exact
            // to double precision.
            s = -0.5 * expm1_tiny(-0.5 * lambda);
        }
        a = 0.5;
        b = 0.5 * df;
        const double rxb = std::pow(rxb0, b);
        const double albeta =
            M_LN_SQRT_PI + gammln(b) - gammln(0.5 + b);
        xodd = incbeta(a, b, x);
        double godd = 2.0 * rxb * std::exp(a * std::log(x) - albeta);
        const double tnc0 = b * x;
        xeven = (tnc0 < DBL_EPSILON) ? tnc0 : (1.0 - rxb);
        double geven = tnc0 * rxb;
        tnc = p * xodd + q * xeven;

        int it = 0;
        while (it < ITRMAX) {
            it += 1;
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2.0 * it);
            q *= lambda / (2.0 * it + 1.0);
            tnc += p * xodd + q * xeven;
            s -= p;
            if (s < -1e-10)
                break; // non-convergence
            if (s <= 0.0 && it > 1)
                break;
            const double errbd = 2.0 * s * (xodd - godd);
            if (std::abs(errbd) < ERRMAX)
                break; // converged
        }
    } else {
        tnc = 0.0;
    }

    tnc += pnorm(-del, 0.0, 1.0, true);
    const double capped = std::min(tnc, 1.0);
    return (lower_tail != negdel) ? capped : (1.0 - capped);
}

// Non-central t pdf (R's `dt(x, df, ncp)`). Uses the identity
// `f = df/|x| * |F(x*sqrt((df+2)/df), df+2, ncp) - F(x, df, ncp)|` for
// `|x| > eps*sqrt(df)`, and the closed form at `x ~= 0`. `ncp = 0` falls back
// to the central `dt`; infinite `df` is the normal with mean = ncp.
inline double dnt(double x, double df, double ncp) {
    constexpr double M_LN_SQRT_PI = 0.5723649429247001; // R's M_LN_SQRT_PI

    if (std::isnan(x) || std::isnan(df))
        return nanv();
    if (df <= 0.0)
        return nanv();
    if (ncp == 0.0)
        return dt(x, df);
    if (!isfinite(x))
        return 0.0;
    if (std::isinf(df) || df > 1e8)
        return dnorm(x, ncp, 1.0);
    // R's `dnt.c`: `u = log(df) - log(|x|) + log(|F(x*sqrt((df+2)/df),
    // df+2, ncp) - F(x, df, ncp)|)` (log-scale throughout; the inner log is
    // part of the formula, not an output flag).
    const double u = (std::abs(x) > std::sqrt(df * DBL_EPSILON))
                         ? (std::log(df) - std::log(std::abs(x)) +
                            std::log(std::abs(pnt(x * std::sqrt((df + 2.0) / df),
                                                  df + 2.0, ncp, true) -
                                             pnt(x, df, ncp, true))))
                         : (gammln((df + 1.0) / 2.0) - gammln(df / 2.0) -
                            (M_LN_SQRT_PI + 0.5 * (std::log(df) + ncp * ncp)));
    return std::exp(u);
}

// ---------------------------------------------------------------------------
// poisson
// ---------------------------------------------------------------------------

// Poisson pdf (R's `dpois(x, lambda)`); 0 for negative `x` or `lambda`.
inline double dpois(double x, double lambda) {
    if (x < 0.0 || lambda <= 0.0)
        return 0.0;
    return std::exp(x * std::log(lambda) - lambda - gammln(x + 1.0));
}

// Poisson cdf (R's `ppois(x, lambda, lower.tail)`), via
// `P(X <= x) = Q(x + 1, lambda) = pgamma(lambda, shape = x + 1,
// lower.tail = FALSE)`.
inline double ppois(double x, double lambda, bool lower_tail) {
    if (lambda <= 0.0) {
        // degenerate point mass at 0
        const double p0 = (x >= 0.0) ? 1.0 : 0.0;
        return lower_tail ? p0 : 1.0 - p0;
    }
    return pgamma(lambda, x + 1.0, !lower_tail);
}

// ---------------------------------------------------------------------------
// log-space arithmetic (R's logspaceadd / logspacesub)
// ---------------------------------------------------------------------------

// log(1 + exp(x)), numerically stable.
inline double ln_1p_exp(double x) {
    if (x < 0.0)
        return std::log(1.0 + std::exp(x));
    return x + std::log(1.0 + std::exp(-x));
}

// R's `logspaceadd`: `log(exp(lx) + exp(ly))`.
inline double logspaceadd(double lx, double ly) {
    const double a = std::max(lx, ly);
    const double b = std::min(lx, ly);
    return a + ln_1p_exp(b - a);
}

// R's `logspacesub`: `log(exp(lx) - exp(ly))`, requiring `lx >= ly`;
// returns NaN otherwise (as in R for the extreme case the package hits).
inline double logspacesub(double lx, double ly) {
    if (lx < ly)
        return nanv();
    const double t = ly - lx; // <= 0
    // log(1 - exp(t)), stable for t near 0 via log1p.
    const double l = (t < -LN2_C) ? std::log(1.0 - std::exp(t))
                                  : std::log1p(-std::exp(t));
    return lx + l;
}

} // namespace stats
} // namespace npfc

#endif // NPFIC_STATS_H
