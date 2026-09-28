// npfc_grid.h — grid / initial-mix construction: exact ports of
// `nspmix::whist`, `nspmix:::gridpoints.npnorm` and `nspmix:::initial.npnorm`,
// plus the `hist.default` integer-breaks path they rely on. A faithful port
// of npfixedcomppy/src/grid.rs. These are R-exact: the support grid and the
// starting mixing distribution are what make the final estimate reproduce
// the R package.
#ifndef NPFIC_GRID_H
#define NPFIC_GRID_H

#include "npfc_pretty.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace npfc {
namespace grid {

// Stable ascending sort on f64, NaN "equal" (matches Rust's
// `partial_cmp().unwrap_or(Equal)`).
inline bool f64_less(double a, double b) { return a < b; }

// `diff(x)`: successive differences.
inline std::vector<double> diff(const std::vector<double>& x) {
    if (x.size() < 2)
        return {};
    std::vector<double> out(x.size() - 1);
    for (std::size_t i = 0; i + 1 < x.size(); ++i)
        out[i] = x[i + 1] - x[i];
    return out;
}

inline double median_sorted(std::vector<double> h) {
    if (h.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(h.begin(), h.end(), f64_less);
    const std::size_t m = h.size() / 2;
    if (h.size() % 2 == 1)
        return h[m];
    return 0.5 * (h[m - 1] + h[m]);
}

// Histogram binning with explicit (numeric) breaks — the shared core of
// `hist.default` for both the integer-`breaks` and vector-`breaks` forms
// (diddle rule: `median(h)` when `nB > 5`, `range(x)` when `nB <= 3`,
// `min(h)` otherwise; bin `i` is `(breaks[i] - diddle, breaks[i+1] + diddle]`).
struct HistOut {
    std::vector<double> breaks, mids, density;
    std::vector<long> counts;
};

inline HistOut hist_from_breaks(const std::vector<double>& x,
                                const std::vector<double>& breaks_in,
                                std::size_t n) {
    if (breaks_in.size() <= 1) {
        // degenerate data: one unique value
        const double m = x[0];
        return HistOut{{m - 0.5, m + 0.5}, {m}, {1.0 / static_cast<double>(n)},
                       {static_cast<long>(n)}};
    }
    const std::size_t nb = breaks_in.size();
    const std::vector<double> h = diff(breaks_in);
    double hmin = std::numeric_limits<double>::infinity();
    for (double v : h)
        hmin = std::min(hmin, v);
    const double hmed = median_sorted(h);
    const double minx = x[0], maxx = x[n - 1];
    const double diddle =
        1e-7 * (nb > 5 ? hmed : (nb <= 3 ? (maxx - minx) : hmin));
    std::vector<long> counts(nb - 1, 0);
    for (double xi : x) {
        // bin i: (breaks[i]-diddle, breaks[i+1]+diddle], 1-based i
        std::size_t b = 0;
        while (b + 1 < nb && xi > breaks_in[b + 1] + diddle)
            b += 1;
        // include.lowest: xi == breaks[0] goes to bin 0
        counts[b] += 1;
    }
    std::vector<double> dens(nb - 1), mids(nb - 1);
    for (std::size_t i = 0; i + 1 < nb; ++i) {
        dens[i] = static_cast<double>(counts[i]) /
                  (static_cast<double>(n) * h[i]);
        mids[i] = 0.5 * (breaks_in[i] + breaks_in[i + 1]);
    }
    return HistOut{breaks_in, mids, dens, std::move(counts)};
}

// Port of `hist.default(x, breaks = K, plot = FALSE)` for an *integer*
// `breaks = K` (the only form `whist` ever uses). Returns
// `(breaks, mids, density, counts)`, with `density = counts / (n * h)`.
inline HistOut hist_int(const std::vector<double>& x, long breaks_scalar) {
    // x is already finite & sorted by whist
    const std::size_t n = x.size();
    const double minx = x[0], maxx = x[n - 1];
    const std::vector<double> breaks =
        pretty::pretty_range(minx, maxx, breaks_scalar, 1);
    return hist_from_breaks(x, breaks, n);
}

// Weighted counts over explicit breaks (the `whist_breaks` shared core).
inline void weighted_counts(const std::vector<double>& xs,
                            const std::vector<double>& ws,
                            const std::vector<double>& breaks_in,
                            std::vector<double>& wcounts) {
    const std::size_t nb = breaks_in.size();
    wcounts.assign(nb - 1, 0.0);
    if (nb <= 1)
        return;
    const std::vector<double> h = diff(breaks_in);
    double hmin = std::numeric_limits<double>::infinity();
    for (double v : h)
        hmin = std::min(hmin, v);
    const double hmed = median_sorted(h);
    const double diddle =
        1e-7 * (nb > 5 ? hmed : (nb <= 3 ? (xs.back() - xs.front()) : hmin));
    for (std::size_t j = 0; j < xs.size(); ++j) {
        std::size_t b = 0;
        while (b + 1 < nb && xs[j] > breaks_in[b + 1] + diddle)
            b += 1;
        wcounts[b] += ws[j];
    }
}

// `whist` with explicit numeric breaks (as `initial.nppois` uses): returns
// `(breaks, mids, density)` with the weighted density.
inline std::tuple<std::vector<double>, std::vector<double>,
                  std::vector<double>>
whist_breaks(const std::vector<double>& x, const std::vector<double>& w,
             const std::vector<double>& breaks_in) {
    const std::size_t n = x.size();
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&x](std::size_t a, std::size_t b) {
                         return f64_less(x[a], x[b]);
                     });
    std::vector<double> xs(n), ws(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = x[idx[i]];
        ws[i] = w[idx[i]];
    }
    const HistOut hh = hist_from_breaks(xs, breaks_in, n);
    double wsum = 0.0;
    for (double v : ws)
        wsum += v;
    std::vector<double> wcounts;
    weighted_counts(xs, ws, breaks_in, wcounts);
    const std::size_t nb = hh.breaks.size();
    const std::vector<double> h = diff(hh.breaks);
    std::vector<double> dens(nb - 1);
    for (std::size_t i = 0; i + 1 < nb; ++i)
        dens[i] = wcounts[i] / (wsum * h[i]);
    return {hh.breaks, hh.mids, dens};
}

// Port of `nspmix::whist(x, w, breaks = K, plot = FALSE, freq = FALSE)`.
// Returns `(breaks, mids, density)` where `density = wcount / (W * h)`,
// `wcount` is the sum of the weights per bin and `W = sum(w)`.
inline std::tuple<std::vector<double>, std::vector<double>,
                  std::vector<double>>
whist(const std::vector<double>& x, const std::vector<double>& w,
      long breaks_scalar) {
    const std::size_t n = x.size();
    // order(x) / order(w) — stable ascending sort, ties in original order
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&x](std::size_t a, std::size_t b) {
                         return f64_less(x[a], x[b]);
                     });
    std::vector<double> xs(n), ws(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = x[idx[i]];
        ws[i] = w[idx[i]];
    }
    const HistOut hh = hist_int(xs, breaks_scalar);
    double wsum = 0.0;
    for (double v : ws)
        wsum += v;
    std::vector<double> wcounts;
    weighted_counts(xs, ws, hh.breaks, wcounts);
    const std::size_t nb = hh.breaks.size();
    const std::vector<double> h = diff(hh.breaks);
    std::vector<double> dens(nb - 1);
    for (std::size_t i = 0; i + 1 < nb; ++i)
        dens[i] = wcounts[i] / (wsum * h[i]);
    return {hh.breaks, hh.mids, dens};
}

// Port of `nspmix:::disc(pt, pr, sort = TRUE, collapse = FALSE)`: sorts by
// `pt` (stable) and renormalises `pr` to sum to 1. (Declared early; the
// `initial.*` constructors call it.)
void disc(std::vector<double>& pt, std::vector<double>& pr);

// Port of `nspmix:::gridpoints.npnorm(x, beta, grid = 100)`.
//
// `v` is the `disc`-style data: `v` (support points) and `w` (weights).
inline std::vector<double> gridpoints_npnorm(const std::vector<double>& v,
                                             const std::vector<double>& w,
                                             double beta, long grid) {
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    for (double x : v) {
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);
    }
    const double brange = diff({minv, maxv})[0];
    const long breaks_scalar =
        static_cast<long>(std::max(std::ceil(brange / (5.0 * beta)), 5.0));
    auto [br, mids, dens] = whist(v, w, breaks_scalar);
    (void)mids;
    // i = r$density != 0
    const std::size_t nb = dens.size();
    std::vector<bool> i0(nb), i(nb);
    for (std::size_t k = 0; k < nb; ++k) {
        i0[k] = dens[k] != 0.0;
        const bool prev = k > 0 ? i0[k - 1] : false;
        const bool next = k + 1 < nb ? i0[k + 1] : false;
        i[k] = i0[k] || prev || next;
    }
    long m = 0;
    for (bool b : i)
        m += b ? 1 : 0;
    const double ratio = m > 0 ? static_cast<double>(grid) / static_cast<double>(m)
                               : std::nan("");
    const long k = static_cast<long>(std::max(std::ceil(ratio), 10.0));
    const double d = br[1] - br[0];
    // s = r$breaks[-length(r$breaks)][i]: the first (nb) breaks, filtered by
    // i
    std::vector<double> out;
    for (std::size_t q = 0; q < nb; ++q) {
        if (!i[q])
            continue;
        const double s = br[q];
        for (long j = 0; j < k; ++j)
            out.push_back(s + d * (static_cast<double>(j) + 1.0 - 0.5) /
                              static_cast<double>(k));
    }
    std::vector<double> vout;
    vout.reserve(2 + out.size());
    vout.push_back(minv);
    vout.insert(vout.end(), out.begin(), out.end());
    vout.push_back(maxv);
    return vout;
}

// Port of `nspmix:::initial.npnorm(x, beta = NULL, mix = NULL, kmax = NULL)`.
//
// `v`/`w` are the data points/weights; `mix_pt`/`mix_pr` are the
// user-supplied starting mixing distribution (empty `mix_pt` for the
// default). Returns `(beta, pt, pr)`.
inline std::tuple<double, std::vector<double>, std::vector<double>>
initial_npnorm(const std::vector<double>& v, const std::vector<double>& w,
               double beta, const std::vector<double>& mix_pt,
               const std::vector<double>& mix_pr) {
    const bool have_mix = !mix_pt.empty();
    if (have_mix) {
        std::vector<double> pt = mix_pt;
        std::vector<double> pr =
            mix_pr.empty()
                ? std::vector<double>(mix_pt.size(), 1.0 /
                                                             static_cast<double>(mix_pt.size()))
                : mix_pr;
        disc(pt, pr);
        return {beta, std::move(pt), std::move(pr)};
    }
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    for (double x : v) {
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);
    }
    const double brange = diff({minv, maxv})[0];
    const long breaks_scalar =
        static_cast<long>(std::max(std::ceil(brange / (5.0 * beta)), 5.0));
    auto [br, mids, dens] = whist(v, w, breaks_scalar);
    (void)br;
    std::vector<double> pt, pr;
    for (std::size_t q = 0; q < dens.size(); ++q)
        if (dens[q] != 0.0) {
            pt.push_back(mids[q]);
            pr.push_back(dens[q]);
        }
    disc(pt, pr);
    return {beta, std::move(pt), std::move(pr)};
}

// Port of `nspmix:::gridpoints.nppois(x, beta, grid = 100)`. For the
// Poisson family the support space is `[0, Inf)`, so this reduces to the
// squared `seq` over the square-root range of the data:
// `seq(sqrt(min v), sqrt(max v), length = grid)^2`.
inline std::vector<double> gridpoints_nppois(const std::vector<double>& v,
                                             double /*beta*/, long grid) {
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    for (double x : v) {
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);
    }
    minv = std::max(minv, 0.0);
    maxv = std::max(maxv, 0.0);
    const double lo = std::sqrt(minv);
    const double hi = std::sqrt(maxv);
    const long g = std::max(grid, 2L);
    const double step = (hi - lo) / (static_cast<double>(g) - 1.0);
    std::vector<double> out(static_cast<std::size_t>(g));
    for (long i = 0; i < g; ++i) {
        const double t = lo + step * static_cast<double>(i);
        out[static_cast<std::size_t>(i)] = t * t;
    }
    return out;
}

// Port of `nspmix:::initial.nppois(x, beta = NULL, mix = NULL, kmax = NULL)`
// for the default `kmax = NULL` path (the only form `nppoisll` uses).
// Returns `(beta, pt, pr)`.
inline std::tuple<double, std::vector<double>, std::vector<double>>
initial_nppois(const std::vector<double>& v, const std::vector<double>& w,
               double beta, const std::vector<double>& mix_pt,
               const std::vector<double>& mix_pr) {
    if (!mix_pt.empty()) {
        std::vector<double> pt = mix_pt;
        std::vector<double> pr =
            mix_pr.empty()
                ? std::vector<double>(mix_pt.size(), 1.0 /
                                                             static_cast<double>(mix_pt.size()))
                : mix_pr;
        disc(pt, pr);
        return {beta, std::move(pt), std::move(pr)};
    }
    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    for (double x : v) {
        minv = std::min(minv, x);
        maxv = std::max(maxv, x);
    }
    minv = std::max(minv, 0.0);
    maxv = std::max(maxv, 0.0);
    const long long mi = std::max(static_cast<long long>(std::floor(std::sqrt(minv)) - 1.0),
                                  0LL);
    const long long ma = std::max(static_cast<long long>(std::ceil(std::sqrt(maxv))),
                                  1LL);
    // breaks = (mi:ma)^2
    std::vector<double> breaks;
    for (long long k = mi; k <= ma; ++k)
        breaks.push_back(static_cast<double>(k) * static_cast<double>(k));
    auto [br, mids, dens] = whist_breaks(v, w, breaks);
    (void)br;
    std::vector<double> pt, pr;
    for (std::size_t q = 0; q < dens.size(); ++q)
        if (dens[q] != 0.0) {
            pt.push_back(mids[q]);
            pr.push_back(dens[q]);
        }
    disc(pt, pr);
    return {beta, std::move(pt), std::move(pr)};
}

inline void disc(std::vector<double>& pt, std::vector<double>& pr) {
    const std::size_t k = std::max(pt.size(), pr.size());
    while (pt.size() < k)
        pt.push_back(pt.empty() ? 0.0 : pt.back());
    while (pr.size() < k)
        pr.push_back(1.0);
    std::vector<std::size_t> idx(k);
    for (std::size_t i = 0; i < k; ++i)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&pt](std::size_t a, std::size_t b) {
                         return f64_less(pt[a], pt[b]);
                     });
    std::vector<double> spt(k), spr(k);
    for (std::size_t i = 0; i < k; ++i) {
        spt[i] = pt[idx[i]];
        spr[i] = pr[idx[i]];
    }
    double s = 0.0;
    for (double v : spr)
        s += v;
    for (std::size_t j = 0; j < k; ++j) {
        pt[j] = spt[j];
        pr[j] = s != 0.0 ? spr[j] / s : spr[j];
    }
}

}  // namespace grid
}  // namespace npfc

#endif  // NPFIC_GRID_H
