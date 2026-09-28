// npfc_pretty.h — R's `pretty()`: an exact port of `R_pretty`
// (`src/appl/pretty.c`) plus the `pretty.default` wrapper and the
// `seq.int(length.out = k)` arithmetic from `src/main/seq.c`. A faithful
// port of npfixedcomppy/src/pretty.rs. The histogram-based grid /
// initial-mix construction (nspmix) depends on these exact values.
#ifndef NPFIC_PRETTY_H
#define NPFIC_PRETTY_H

#include <cmath>
#include <limits>
#include <vector>

namespace npfc {
namespace pretty {

constexpr double DBL_EPS = 2.220446049250313e-16;  // double epsilon
constexpr double ROUNDING_EPS = 1e-10;
constexpr double INT_MAX_F = 2147483647.0;
constexpr double INT_MIN_F = -2147483648.0;

// Exact port of `R_pretty` (`src/appl/pretty.c`).
//
// Constructs the pretty-unit covering `[lo, up]` with about `*ndiv`
// intervals, adjusting `*lo`/`*up` (when `return_bounds != 0`) so the result
// covers the original range, and setting `*ndiv` to the actual number of
// intervals. Returns the unit.
//
// `h`, `h5`, `f_min` are the `high_u_fact` array `(high.u.bias, u5.bias,
// f.min)` of `pretty.default`: `(1.5, 0.5 + 1.5*1.5, 2^-20)`.
inline double r_pretty(double& lo, double& up, long& ndiv, long min_n,
                       double shrink_sml, double h, double h5, double f_min,
                       long eps_correction, long return_bounds) {
    const double lo_ = lo;
    const double up_ = up;
    const double dx = up_ - lo_;
    double cell;
    bool i_small;
    if (dx == 0.0 && up_ == 0.0) {
        cell = 1.0;
        i_small = true;
    } else {
        const double c0 = std::max(std::abs(lo_), std::abs(up_));
        double u = 1.0 + (h5 >= 1.5 * h + 0.5 ? 1.0 / (1.0 + h)
                                              : 1.5 / (1.0 + h5));
        u *= std::max(1L, ndiv) * DBL_EPS;
        i_small = dx < c0 * u * 3.0;
        cell = c0;
    }

    if (i_small) {
        if (cell > 10.0)
            cell = 9.0 + cell / 10.0;
        cell *= shrink_sml;
        if (min_n > 1)
            cell /= static_cast<double>(min_n);
    } else {
        cell = dx;
        if (std::isfinite(dx)) {
            if (ndiv > 1)
                cell /= static_cast<double>(ndiv);
        } else {
            // up - lo = +Inf (overflow; both finite)
            cell = up_ / static_cast<double>(ndiv) -
                   lo_ / static_cast<double>(ndiv);
        }
    }

    constexpr double MAX_F = 1.25;
    double subsmall = f_min * std::numeric_limits<double>::min();
    if (subsmall == 0.0)
        subsmall = std::numeric_limits<double>::min();
    if (cell < subsmall)
        cell = subsmall;
    else if (cell > std::numeric_limits<double>::max() / MAX_F)
        cell = std::numeric_limits<double>::max() / MAX_F;

    // base <= cell < 10*base
    const double base = std::pow(10.0, std::floor(std::log10(cell)));

    // unit in {1,2,5,10}*base, biased by h / h5
    double unit = base;
    if ((2.0 * base) - cell < h * (cell - unit))
        unit = 2.0 * base;
    if ((5.0 * base) - cell < h5 * (cell - unit))
        unit = 5.0 * base;
    if ((10.0 * base) - cell < h * (cell - unit))
        unit = 10.0 * base;

    double ns = std::floor(lo_ / unit + ROUNDING_EPS);
    double nu = std::ceil(up_ / unit - ROUNDING_EPS);

    if (eps_correction != 0 && (eps_correction > 1 || !i_small)) {
        // pretty.default always calls with eps.correct = 0; the correction is
        // implemented faithfully for completeness.
        const double e = DBL_EPS;
        const double d_max = std::numeric_limits<double>::max() * (1.0 - e);
        if (lo_ < 0.0)
            lo *= 1.0 + e;
        else if (lo_ > 0.0)
            lo *= 1.0 - e;
        else
            lo = -std::min(unit, std::numeric_limits<double>::min());
        if (up_ < 0.0)
            up *= 1.0 - e;
        else if (up_ > 0.0) {
            if (up_ < d_max)
                up *= 1.0 + e;
        } else {
            up = std::min(unit, std::numeric_limits<double>::min());
        }
    }

    // safety cap: in degenerate cases (inf/nan) the C loops could spin
    // forever
    long guard = 0;
    while (ns * unit > lo + ROUNDING_EPS * unit && guard < 1000000) {
        ns -= 1.0;
        guard++;
    }
    guard = 0;
    while (!std::isfinite(ns * unit) && guard < 1000000) {
        ns += 1.0;
        guard++;
    }
    guard = 0;
    while (nu * unit < up - ROUNDING_EPS * unit && guard < 1000000) {
        nu += 1.0;
        guard++;
    }
    guard = 0;
    while (!std::isfinite(nu * unit) && guard < 1000000) {
        nu -= 1.0;
        guard++;
    }

    const long k = static_cast<long>(0.5 + nu - ns);  // (int)(0.5 + nu - ns)
    if (k < min_n) {
        const long add = min_n - k;
        if (lo_ == 0.0 && ns == 0.0 && up_ != 0.0) {
            nu += static_cast<double>(add);
        } else if (up_ == 0.0 && nu == 0.0 && lo_ != 0.0) {
            ns -= static_cast<double>(add);
        } else if (ns >= 0.0) {
            nu += static_cast<double>(add / 2);
            ns -= static_cast<double>((add / 2) + (add % 2));
        } else {
            ns -= static_cast<double>(add / 2);
            nu += static_cast<double>((add / 2) + (add % 2));
        }
        ndiv = min_n;
    } else {
        ndiv = k;
    }

    if (return_bounds != 0) {
        if (ns * unit < lo)
            lo = ns * unit;
        if (nu * unit > up)
            up = nu * unit;
    } else {
        lo = ns;
        up = nu;
    }
    return unit;
}

// `seq.int(from, to, length.out = lout)` from `src/main/seq.c` (the
// `by == R_MissingArg` branch). Replicates R's exact arithmetic, including
// the integer fast path.
inline std::vector<double> seq_int_len(double from, double to,
                                       std::size_t lout) {
    if (lout == 0)
        return {};
    double rfrom = from, rto = to;
    double rby = 0.0;
    bool finite_del = false;
    if (lout > 2) {
        const double nint = static_cast<double>(lout - 1);
        const double del = rto - rfrom;
        if (std::isfinite(del)) {
            rby = del / nint;
            finite_del = true;
        } else {
            rby = rto / nint - rfrom / nint;
        }
    }
    const bool int_ok = rfrom <= INT_MAX_F && rfrom >= INT_MIN_F &&
                        rto <= INT_MAX_F && rto >= INT_MIN_F &&
                        std::trunc(rfrom) == rfrom &&
                        (lout <= 1 || std::trunc(rto) == rto) &&
                        (lout <= 2 || std::trunc(rby) == rby);
    std::vector<double> ans(lout, 0.0);
    if (int_ok) {
        ans[0] = static_cast<double>(static_cast<long long>(rfrom));
        if (lout > 1)
            ans[lout - 1] = static_cast<double>(static_cast<long long>(rto));
        for (std::size_t i = 1; i + 1 < lout; ++i)
            ans[i] = static_cast<double>(static_cast<long long>(
                rfrom + static_cast<double>(i) * rby));
    } else {
        ans[0] = rfrom;
        if (lout > 1)
            ans[lout - 1] = rto;
        if (lout > 2) {
            if (finite_del) {
                for (std::size_t i = 1; i + 1 < lout; ++i)
                    ans[i] = rfrom + static_cast<double>(i) * rby;
            } else {
                rfrom /= 4.0;
                rby /= 4.0;
                for (std::size_t i = 1; i + 1 < lout; ++i)
                    ans[i] = (rfrom + static_cast<double>(i) * rby) * 4.0;  // ldexp(y, 2)
            }
        }
    }
    return ans;
}

// R's `pretty(x, n, min.n = n %/% 3, shrink.sml = 0.75, high.u.bias = 1.5,
// u5.bias = 0.5 + 1.5*high.u.bias, f.min = 2^-20, eps.correct = 0,
// bounds = TRUE)` for a numeric vector `x`.
//
// This is the function `hist.default` uses for integer `breaks`, and hence
// what `nspmix::whist` (and `gridpoints.npnorm` / `initial.npnorm`) rely on.
inline std::vector<double> pretty(const std::vector<double>& x, long n) {
    std::vector<double> finite;
    for (double v : x)
        if (std::isfinite(v))
            finite.push_back(v);
    if (finite.empty())
        return {};
    double lo = std::numeric_limits<double>::infinity();
    double up = -std::numeric_limits<double>::infinity();
    for (double v : finite) {
        lo = std::min(lo, v);
        up = std::max(up, v);
    }
    const long min_n = n / 3;
    long ndiv = n;
    const double h = 1.5;
    const double h5 = 0.5 + 1.5 * h;
    const double f_min = std::pow(2.0, -20);
    r_pretty(lo, up, ndiv, min_n, 0.75, h, h5, f_min, 0, 1);

    // s = seq.int(z$l, z$u, length.out = n + 1L)
    std::vector<double> s = seq_int_len(lo, up, static_cast<std::size_t>(ndiv + 1));
    // eps.correct == 0 zero-correction
    if (ndiv != 0) {
        const double delta = (up - lo) / static_cast<double>(ndiv);
        for (double& v : s)
            if (std::abs(v) < 1e-14 * delta)
                v = 0.0;
    }
    return s;
}

// Convenience: `pretty(c(lo, hi), n, min.n = 1)` as used by
// `hist.default`'s integer-breaks branch.
inline std::vector<double> pretty_range(double lo, double hi, long n,
                                        long min_n) {
    long ndiv = n;
    const double h = 1.5;
    const double h5 = 0.5 + 1.5 * h;
    const double f_min = std::pow(2.0, -20);
    r_pretty(lo, hi, ndiv, min_n, 0.75, h, h5, f_min, 0, 1);
    std::vector<double> s =
        seq_int_len(lo, hi, static_cast<std::size_t>(ndiv + 1));
    if (ndiv != 0) {
        const double delta = (hi - lo) / static_cast<double>(ndiv);
        for (double& v : s)
            if (std::abs(v) < 1e-14 * delta)
                v = 0.0;
    }
    return s;
}

}  // namespace pretty
}  // namespace npfc

#endif  // NPFIC_PRETTY_H
