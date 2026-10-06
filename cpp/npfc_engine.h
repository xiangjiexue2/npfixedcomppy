// npfc_engine.h — the mixing-distribution driver: `computemixdist` and
// `estpi0`, plus the shared mixing utilities (sort / simplify / collapse,
// the Brent-like `Brmin` and the successive-parabolic `Dfmin` support-point
// solvers). A faithful port of npfixedcomppy/src/engine.rs, which is itself a
// faithful port of `npfixedcomp2/inst/include/npfixedcomp.h`.
//
// Per-family numerics live behind the abstract `Family` interface (mirroring
// the virtual methods of the C++ `npfixedcomp` base class); the concrete
// families are defined in the npfc_fam_*.h headers.
#ifndef NPFIC_ENGINE_H
#define NPFIC_ENGINE_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "npfc_kernels.h"
#include "npfc_nnls.h"

namespace npfc {

// ---------------------------------------------------------------------------
// small numeric helpers
// ---------------------------------------------------------------------------

inline bool f64_isnan(double x) { return std::isnan(x); }
inline double f64_nan() { return std::numeric_limits<double>::quiet_NaN(); }

// ---------------------------------------------------------------------------
// Per-solve gradient scratch. The `dens` vector of a solvegrad call is fixed
// for the WHOLE support-point search (it changes only once the outer loop
// recomputes the mapping), so the per-family invariant arrays that every
// gradfun/gradfunvec call recomputed from scratch (e.g. the reciprocal
// `1 / (dens + pre)` and its dot product) are computed ONCE per solve by
// `prepare_solve` (defined per-family in npfc_families.h) and reused by every
// candidate evaluation inside the search.
// ---------------------------------------------------------------------------
struct SolveCtx {
    const std::vector<double>* dens = nullptr;
};

// Stable-sort comparator matching Rust's `a.partial_cmp(b).unwrap_or(Equal)`:
// NaNs compare "equal" (never less), so a plain `a < b` is equivalent.
inline bool f64_less(double a, double b) { return a < b; }

// `diff_(x)`: successive differences, length `x.size() - 1`.
inline std::vector<double> diff(const std::vector<double>& x) {
    if (x.size() < 2)
        return {};
    std::vector<double> out(x.size() - 1);
    for (std::size_t i = 0; i + 1 < x.size(); ++i)
        out[i] = x[i + 1] - x[i];
    return out;
}

// ---------------------------------------------------------------------------
// Family interface (the virtual methods of the C++ `npfixedcomp` base class)
// ---------------------------------------------------------------------------

class Family {
public:
    virtual ~Family() = default;

    // Loss value given the mixture density `maps` (the fixed-component
    // density is excluded; the family adds `precompute` internally).
    virtual double lossfunction(const std::vector<double>& maps) const = 0;

    // Mixture density at the data for support points `mu0`/`pi0` (the fixed
    // components are excluded).
    virtual std::vector<double> mapping(const std::vector<double>& mu0,
                                        const std::vector<double>& pi0) const = 0;

    // Gradient of the loss w.r.t. a single new support point `mu`. Fills
    // (a0, a1): the derivative in the probability direction (a0) and in the
    // support-point direction (a1); unrequested components are left 0.0.
    // `ctx` carries the per-solve invariant cache built by `prepare_solve`
    // (see `SolveCtx`); a ctx whose `dens` does not match the family's
    // cache makes the family recompute on the fly (identical arithmetic).
    virtual void gradfun(double mu, const std::vector<double>& dens,
                         SolveCtx& ctx, bool d0, bool d1, double& a0,
                         double& a1) const = 0;

    // Vectorised gradient over support points `mu`.
    virtual void gradfunvec(const std::vector<double>& mu,
                            const std::vector<double>& dens, SolveCtx& ctx,
                            bool d0, bool d1, std::vector<double>& a0,
                            std::vector<double>& a1) const = 0;

    // Per-solve invariant preparation, called ONCE at the start of each
    // `solvegrad` with the `dens` that the whole support-point search uses.
    // A family caches the arrays every gradfun/gradfunvec call inside the
    // search would otherwise recompute (e.g. `1 / (dens + precompute)`).
    // Default: nothing to prepare.
    virtual void prepare_solve(const std::vector<double>& /*dens*/) {}

    // Optional bound `M >= -inf a0''` for the weight-direction gain `a0`
    // (d1 families only), consumed by the 1-D crossing certificate in
    // `solvegradd1`: the sweep's endpoint values (a0 and a0' = a1) plus M
    // bound a0 from below on the whole interval. A family returning a bound
    // must return it for the a0 whose FIRST DERIVATIVE IS a1 (the d1
    // convention). Default: -1 = no bound (the certificate never applies).
    virtual double gain_curv_bound() const { return -1.0; }

    // Recompute the weights `pi0` given support points `mu0` and the current
    // mixture density `dens` (constrained NNLS weight subproblem followed by
    // the Armijo line search `checklossfun2`).
    virtual void computeweights(const std::vector<double>& mu0,
                                std::vector<double>& pi0,
                                const std::vector<double>& dens) const = 0;

    // The precomputed fixed-component density (the C++ base class
    // `precompute` member).
    virtual const std::vector<double>& precompute() const = 0;

    // The base-class `checklossfun2`: Armijo backtracking line search on the
    // weight step `eta` (`sigma` halves from 2, `alpha = 0.3333`).
    virtual void checklossfun2(const std::vector<double>& diff,
                               std::vector<double>& pi0,
                               const std::vector<double>& eta,
                               const std::vector<double>& p,
                               const std::vector<double>& dens) const {
        const double llorigin = lossfunction(dens);
        double con = 0.0;
        const std::size_t k = std::min(p.size(), eta.size());
        for (std::size_t j = 0; j < k; ++j)
            con -= p[j] * eta[j];
        const double alpha = 0.3333;
        double sigma = 2.0;
        std::vector<double> ans = pi0;
        while (true) {
            sigma *= 0.5;
            std::vector<double> td(dens.size());
            for (std::size_t i = 0; i < dens.size(); ++i)
                td[i] = dens[i] + sigma * diff[i];
            const double lhs = lossfunction(td);
            const double rhs = llorigin + alpha * sigma * con;
            if (lhs < rhs) {
                for (std::size_t j = 0; j < pi0.size(); ++j)
                    ans[j] = pi0[j] + sigma * eta[j];
                break;
            }
            if (sigma < 0.001)
                break;
        }
        pi0 = std::move(ans);
    }

    // Extra term added to the loss for reporting. Zero for the un-binned
    // likelihood families.
    virtual double extrafun() const { return 0.0; }

    // Hypothesis statistic for `estpi0` (likelihood families: `ll - minloss`).
    virtual double hypofun(double ll, double minloss) const = 0;

    // The family (unmixed) density at a single point `x`.
    virtual double familydensity(double x, const std::vector<double>& mu0,
                                 const std::vector<double>& pi0) const = 0;

    // Update the fixed components and recompute the precomputed fixed
    // density (`setprecompute`).
    virtual void set_fixed(const std::vector<double>& mu0fixed,
                           const std::vector<double>& pi0fixed) = 0;

    // One-time, per-grid precomputation hook, called by the engine once the
    // (sorted) grid is fixed. A family with an expensive kernel that is
    // re-evaluated at every grid point on every iteration (the non-central-t
    // `nptll` kernel, or the normal `npnormll` kernel) precomputes the full
    // data x grid kernel matrix here so each iteration's grid sweep becomes a
    // cheap lookup. The kernel depends only on the data, beta and the grid —
    // never on the current weights or the fixed components — so the matrix
    // stays valid for the whole run, including across `estpi0`'s bisection.
    virtual void prepare(const std::vector<double>& /*grid*/) {}

    virtual const char* family_name() const = 0;
    virtual const char* flag() const = 0;
    virtual double beta_value() const = 0;

    // Diagnostics: the (number, wall ms) of kernel columns this family had
    // to EVALUATE during the run (fresh, i.e. not already cached) — the
    // `NPFIXEDCOMPY_PROFILE=1` line reports it. Zero for families without a
    // column cache.
    virtual std::pair<std::size_t, double> kernel_fresh() const {
        return {0, 0.0};
    }
};

// ---------------------------------------------------------------------------
// mixing utilities (sortmix / simplifymix / collapsemix / newmin)
// ---------------------------------------------------------------------------

// Stable sort of `mu0`/`pi0` by `mu0` (port of `sortmix`).
inline void sortmix(std::vector<double>& mu0, std::vector<double>& pi0) {
    const std::size_t k = mu0.size();
    std::vector<std::size_t> idx(k);
    for (std::size_t i = 0; i < k; ++i)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&mu0](std::size_t a, std::size_t b) {
        return f64_less(mu0[a], mu0[b]);
    });
    bool moved = false;
    for (std::size_t i = 0; i < k; ++i)
        if (i != idx[i]) {
            moved = true;
            break;
        }
    if (moved) {
        std::vector<double> mu(k), pi(k);
        for (std::size_t i = 0; i < k; ++i) {
            mu[i] = mu0[idx[i]];
            pi[i] = pi0[idx[i]];
        }
        mu0 = std::move(mu);
        pi0 = std::move(pi);
    }
}

// Remove points whose weight is `|pi| <= 1e-14` (port of `simplifymix`).
inline void simplifymix(std::vector<double>& mu0, std::vector<double>& pi0) {
    if (mu0.size() != 1) {
        std::vector<std::size_t> keep;
        for (std::size_t i = 0; i < pi0.size(); ++i)
            if (std::abs(pi0[i]) > 1e-14)
                keep.push_back(i);
        if (keep.size() != pi0.size()) {
            std::vector<double> mu, pi;
            for (std::size_t i : keep) {
                mu.push_back(mu0[i]);
                pi.push_back(pi0[i]);
            }
            mu0 = std::move(mu);
            pi0 = std::move(pi);
        }
    }
}

// Merge adjacent support points closer than `prec` (port of `collapsemix`).
inline void collapsemix(std::vector<double>& mu0, std::vector<double>& pi0,
                        double prec) {
    if (mu0.size() > 1) {
        bool foo = false;
        for (std::size_t i = 1; i < mu0.size(); ++i)
            if (mu0[i] - mu0[i - 1] <= prec) {
                foo = true;
                break;
            }
        while (foo) {
            std::size_t i = 0;
            while (i + 1 < mu0.size()) {
                if (mu0[i + 1] - mu0[i] <= prec) {
                    const double temp = pi0[i] + pi0[i + 1];
                    mu0[i] = (mu0[i] * pi0[i] + mu0[i + 1] * pi0[i + 1]) / temp;
                    pi0[i + 1] = 0.0;
                    pi0[i] = temp;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            simplifymix(mu0, pi0);
            if (mu0.size() <= 1)
                foo = false;
            else {
                foo = false;
                for (std::size_t j = 1; j < mu0.size(); ++j)
                    if (mu0[j] - mu0[j - 1] <= prec) {
                        foo = true;
                        break;
                    }
            }
        }
    }
}

// Successive-parabolic-interpolation candidate (port of `newmin`).
inline double newmin(const std::array<double, 3>& x,
                     const std::array<double, 3>& fx) {
    const double p = (x[2] - x[0]) * (x[2] - x[0]) * (fx[2] - fx[1]) -
                     (x[2] - x[1]) * (x[2] - x[1]) * (fx[2] - fx[0]);
    const double q = 2.0 * ((x[2] - x[0]) * (fx[2] - fx[1]) -
                            (x[2] - x[1]) * (fx[2] - fx[0]));
    return x[2] - p / q;
}

// ---------------------------------------------------------------------------
// MixResult (the `get_ans` payload)
// ---------------------------------------------------------------------------

struct MixResult {
    std::vector<double> pt;
    std::vector<double> pr;
    double beta = 0.0;
    std::string family;
    double min_gradient = 0.0;
    // Minimum gain at the GRID points (grid-level certificate; see
    // `finish`). NaN when never computed.
    double grid_gain = std::numeric_limits<double>::quiet_NaN();
    double ll = 0.0;
    std::string flag;
    long iter = 0;
    int convergence = 0;
};

// ---------------------------------------------------------------------------
// MixSolver
// ---------------------------------------------------------------------------

class MixSolver {
public:
    MixSolver(std::unique_ptr<Family> fam, std::vector<double> mu0fixed,
              std::vector<double> pi0fixed, std::vector<double> initpt,
              std::vector<double> initpr, std::vector<double> gridpoints,
              int verbose)
        : fam_(std::move(fam)),
          mu0fixed_(std::move(mu0fixed)),
          pi0fixed_(std::move(pi0fixed)),
          initpt_(std::move(initpt)),
          initpr_(std::move(initpr)),
          gridpoints_(std::move(gridpoints)),
          iter_(0),
          convergence_(0),
          verbose_(verbose) {
        // A/B knob (default -1 = unlimited, i.e. the shipped behaviour): cap
        // the refinement steps each candidate runs inside `brmin`/`dfmin` —
        // "fewer steps per candidate" experiments (less per step, the outer
        // loop then needs more iterations to converge).
        if (const char* e = std::getenv("NPFIC_REFINE_STEPS"))
            refine_steps_ = std::atol(e);
        // 1-D crossing certificate (default on; NPFIC_1D_CERT=0 disables):
        // for a sign-change interval both of whose endpoint gains are
        // non-negative, the sweep's free endpoint values (a0 = `pv`,
        // a0' = a1 = `pg`) plus the family's `gain_curv_bound` (M >= -inf
        // a0'') give a tangent-parabola lower bound on a0 over the whole
        // interval; positive => a0 > 0 in it, so the CNM one-evaluation
        // check and `brmin` are guaranteed to find no valid root — both are
        // skipped at zero cost. Sound => the accepted-point set (and hence
        // the whole trajectory and the goldens) is unchanged.
        cert_1d_ = true;
        if (const char* e = std::getenv("NPFIC_1D_CERT"))
            cert_1d_ = std::atol(e) != 0;
        // Support-point hot start (always on): the grid is fixed for the
        // solver's life, so each sign-change interval (d1) / triple (d0)
        // has a stable index. The PREVIOUS call's refined root for an
        // interval is re-verified against the NEW gradient with ONE
        // gradient-value evaluation and accepted when still negative,
        // skipping the exact `brmin`/`dfmin` search — the CNM working-set
        // re-verification (column-generation hot start; Wang 2007,
        // JRSS-B — full references in docs/PERF.md §7). Measured A/B (docs/PERF.md
        // §4b): 1.03–2.39× faster; on families whose support roots drift
        // between iterations (e.g. npnormll/npnormad) it can land on a
        // slightly different local optimum — the goldens in tests/ are
        // recorded on this trajectory.
        // sort gridpoints (R does sort(gridpoints) before calling the C++ fn)
        std::sort(gridpoints_.begin(), gridpoints_.end(), f64_less);
        warm_root_d1_.assign(gridpoints_.size(), f64_nan());
        warm_root_d0_.assign(gridpoints_.size(), f64_nan());
        set_precompute();
        fam_->prepare(gridpoints_);
    }

    // -----------------------------------------------------------------------
    // the support-point solvers
    // -----------------------------------------------------------------------

    // `Brmin`: improved Brent's method for the gradient's d1 (derivative
    // available), called on a sign-change interval. Port of `Brmin`.
    // Stops early as soon as any evaluated point has a NEGATIVE gain
    // (a0 < 0): that point is already a valid new support point — the
    // exact minimum of the gain is not needed (the caller's sign filter
    // accepts any negative point).
    double brmin(double lb, double ub, const std::vector<double>& dens,
                 SolveCtx& ctx, double tol) const {
        double _duma, fa;
        ++grad_evals_;
        fam_->gradfun(lb, dens, ctx, false, true, _duma, fa);
        double _dumb, fb;
        ++grad_evals_;
        fam_->gradfun(ub, dens, ctx, false, true, _dumb, fb);
        double a = lb, b = ub;
        double s = a, fs = fa;
        double c = a, fc = fa;
        long guard = 0;
        while (std::abs(fc) > tol && std::abs(fs) > tol &&
               std::abs(b - a) > tol && guard < 1000 &&
               (refine_steps_ < 0 || guard < refine_steps_)) {
            guard++;
            c = (a + b) / 2.0;
            double a0c, fcc;
            ++grad_evals_;
            fam_->gradfun(c, dens, ctx, true, true, a0c, fcc);
            // Negative-gain early stop (a0 is piggybacked on the same
            // kernel column the d1 part needs).
            if (a0c < 0.0)
                return c;
            fc = fcc;
            if (fa != fc && fb != fc) {
                s = a * fb * fc / (fa - fb) / (fa - fc) +
                    b * fa * fc / (fb - fa) / (fb - fc) +
                    c * fa * fb / (fc - fa) / (fc - fb);
            } else {
                s = b - fb * (b - a) / (fb - fa);
            }
            if (s > a && s < b) {
                double a0s, fss;
                ++grad_evals_;
                fam_->gradfun(s, dens, ctx, true, true, a0s, fss);
                if (a0s < 0.0)
                    return s;
                fs = fss;
            } else {
                s = c;
                fs = fc;
            }
            if (c > s) {
                std::swap(c, s);
                std::swap(fc, fs);
            }
            // a < c < s < b
            if (fc * fs < 0.0) {
                a = c;
                fa = fc;
                b = s;
                fb = fs;
            } else if (fs * fb < 0.0) {
                a = s;
                fa = fs;
            } else {
                b = c;
                fb = fc;
            }
        }
        if (std::abs(fc) < tol)
            return c;
        return s;
    }

    // `Dfmin`: derivative-free minimum via successive parabolic
    // interpolation. Port of `Dfmin`. Stops early as soon as any
    // evaluated point has a NEGATIVE gain (a0 < 0): that point is
    // already a valid new support point (the function returns NaN when
    // it exhausts the search without finding one, as before).
    double dfmin(const std::array<double, 3>& x1, const std::array<double, 3>& fx1,
                 const std::vector<double>& dens, SolveCtx& ctx, double tol) const {
        // C++: `lb`/`ub` are taken from the ORIGINAL endpoints before the
        // reordering swaps; using the post-swap xx[0]/xx[2] narrows the
        // interval and changes the midpoint fallbacks (trajectory drift).
        double lb = x1[0];
        double ub = x1[2];
        std::array<double, 3> xx = x1;
        std::array<double, 3> fxx = fx1;
        if (fxx[0] < fxx[1]) {
            std::swap(xx[0], xx[1]);
            std::swap(fxx[0], fxx[1]);
        }
        if (fxx[1] < fxx[2]) {
            std::swap(xx[1], xx[2]);
            std::swap(fxx[1], fxx[2]);
        }
        long guard = 0;
        while (ub - lb > tol && guard < 1000 &&
               (refine_steps_ < 0 || guard < refine_steps_)) {
            guard++;
            double newpoint = newmin(xx, fxx);
            if (f64_isnan(newpoint) || newpoint < lb || newpoint > ub) {
                if (std::abs(xx[0] - xx[2]) < std::abs(xx[1] - xx[2]))
                    newpoint = (xx[1] + xx[2]) / 2.0;
                else
                    newpoint = (xx[0] + xx[2]) / 2.0;
            }
            double fnewpoint, _dum;
            ++grad_evals_;
            fam_->gradfun(newpoint, dens, ctx, true, false, fnewpoint, _dum);
            // Negative-gain early stop (a0 is what is evaluated here).
            if (fnewpoint < 0.0)
                return newpoint;
            if (fnewpoint > fxx[2]) {
                if (newpoint > xx[2]) {
                    ub = newpoint;
                    for (int i = 0; i < 3; ++i)
                        if (xx[i] > newpoint) {
                            xx[i] = newpoint;
                            fxx[i] = fnewpoint;
                        }
                } else {
                    lb = newpoint;
                    for (int i = 0; i < 3; ++i)
                        if (xx[i] < newpoint) {
                            xx[i] = newpoint;
                            fxx[i] = fnewpoint;
                        }
                }
            } else {
                if (xx[2] > newpoint) {
                    ub = xx[2];
                    for (int i = 0; i < 3; ++i)
                        if (xx[i] > xx[2]) {
                            xx[i] = xx[2];
                            fxx[i] = fxx[2];
                        }
                    xx[2] = newpoint;
                    fxx[2] = fnewpoint;
                } else {
                    lb = xx[2];
                    for (int i = 0; i < 3; ++i)
                        if (xx[i] < xx[2]) {
                            xx[i] = xx[2];
                            fxx[i] = fxx[2];
                        }
                    xx[2] = newpoint;
                    fxx[2] = fnewpoint;
                }
            }
        }
        if (fxx[2] < 0.0)
            return xx[2];
        return f64_nan();
    }

    // New-support-point search when the gradient's derivative is available
    // (flag `d1`). Port of `solvegradd1`.
    std::vector<double> solvegradd1(const std::vector<double>& dens,
                                    double tol) const {
        const std::vector<double>& gp = gridpoints_;
        const int l = static_cast<int>(gp.size());
        if (l < 2)
            return {};
        // Per-solve invariant cache (built once, reused by every candidate
        // evaluation below — the family recomputes the identical arithmetic).
        const_cast<Family*>(fam_.get())->prepare_solve(dens);
        SolveCtx ctx;
        ctx.dens = &dens;
        std::vector<double> pv, pg;
        // `pv` (the gains at the grid) is gained for free on the cached
        // grid kernel (the d1 part needs the same columns); it feeds the
        // zero-cost endpoint acceptance below.
        grad_evals_ += static_cast<long>(gp.size());
        fam_->gradfunvec(gp, dens, ctx, true, true, pv, pg);
        std::vector<int> idx;
        for (int i = 0; i < l - 1; ++i)
            if (pg[i] < 0.0 && pg[i + 1] > 0.0)
                idx.push_back(i);
        std::vector<double> ans;
        ans.reserve(idx.size());
        for (int i : idx) {
            // Cheapest first: an endpoint whose gain is already negative
            // (from the sweep's free `pv`) is a valid support point at
            // zero cost — no search, and the interval's warm root is
            // KEPT (it may still verify in a later call).
            double root = f64_nan();
            if (pv[i] < 0.0) {
                root = gp[i];
            } else if (pv[i + 1] < 0.0) {
                root = gp[i + 1];
            } else {
                // 1-D crossing certificate (default on, NPFIC_1D_CERT=0 off):
                // the sweep already gave a0 (pv) and a0' (= a1 = pg) at both
                // endpoints; with M >= -inf a0'' from the family, the
                // tangent-parabola lower bound is concave, so its interval
                // minimum sits at an endpoint:
                //   min a0 >= min( a0(a), a0(b),
                //                  a0(a) + a0'(a) h - (M/2) h^2,
                //                  a0(b) - a0'(b) h - (M/2) h^2 ).
                // Positive => a0 > 0 on the WHOLE interval, so the CNM
                // re-verification and `brmin` below are guaranteed to find
                // no valid root — skip both at zero cost. The warm root is
                // KEPT (the certificate is state-dependent: a later dens may
                // make the old root verifiable again).
                if (cert_1d_) {
                    const double M = fam_->gain_curv_bound();
                    if (M >= 0.0) {
                        const double h = gp[i + 1] - gp[i];
                        const double half = 0.5 * M * h * h;
                        const double lb = std::min(
                            {pv[i], pv[i + 1], pv[i] + pg[i] * h - half,
                             pv[i + 1] - pg[i + 1] * h - half});
                        if (lb > 0.0) {
                            ++cert_skips_;
                            continue;
                        }
                    }
                }
                // CNM working-set re-verification: re-verify the PREVIOUS
                // call's refined root for this interval with ONE
                // gradient-value evaluation. Still negative => a valid
                // new support point; no `brmin` needed.
                const double r = warm_root_d1_[i];
                if (!f64_isnan(r) && r > gp[i] && r < gp[i + 1]) {
                    double v, _s;
                    ++grad_evals_;
                    fam_->gradfun(r, dens, ctx, true, false, v, _s);
                    if (v < 0.0)
                        root = r;
                }
                if (f64_isnan(root))
                    root = brmin(gp[i], gp[i + 1], dens, ctx, tol);
                warm_root_d1_[i] = root;
            }
            ans.push_back(root);
        }
        if (!ans.empty()) {
            std::vector<double> vals, g2;
            grad_evals_ += static_cast<long>(ans.size());
            fam_->gradfunvec(ans, dens, ctx, true, false, vals, g2);
            std::vector<double> filt;
            filt.reserve(ans.size());
            for (std::size_t i = 0; i < ans.size(); ++i)
                if (vals[i] < 0.0)
                    filt.push_back(ans[i]);
            ans = std::move(filt);
        }
        double pv2, _s2;
        ++grad_evals_;
        fam_->gradfun(gp[0], dens, ctx, true, false, pv2, _s2);
        if (pv2 < 0.0 && pg[0] > 0.0)
            ans.push_back(gp[0]);
        const int last = l - 1;
        ++grad_evals_;
        fam_->gradfun(gp[last], dens, ctx, true, false, pv2, _s2);
        if (pv2 < 0.0 && pg[last] < 0.0)
            ans.push_back(gp[last]);
        return ans;
    }

    // New-support-point search for derivative-free gradients (flag `d0`).
    // Port of `solvegradd0`.
    std::vector<double> solvegradd0(const std::vector<double>& dens,
                                    double tol) const {
        const std::vector<double>& gp = gridpoints_;
        const int l = static_cast<int>(gp.size());
        if (l < 3)
            return {};
        const_cast<Family*>(fam_.get())->prepare_solve(dens);
        SolveCtx ctx;
        ctx.dens = &dens;
        std::vector<double> pv, _g;
        grad_evals_ += static_cast<long>(gp.size());
        fam_->gradfunvec(gp, dens, ctx, true, false, pv, _g);
        std::vector<double> ans;
        for (int j = 0; j < l - 2; ++j) {
            if ((pv[j + 1] - pv[j]) < 0.0 && (pv[j + 2] - pv[j + 1]) > 0.0) {
                // Cheapest first: the trigger makes `pv[j+1]` the triple's
                // discrete minimum; if it is already negative, `gp[j+1]`
                // is a valid support point at zero cost (no `dfmin`). The
                // triple's warm root is KEPT (see `solvegradd1`).
                double r = f64_nan();
                if (pv[j + 1] < 0.0) {
                    r = gp[j + 1];
                } else {
                    // CNM working-set re-verification: the previous call's
                    // refined root for this triple, verified with ONE
                    // gradient-value evaluation. `dfmin` is the expensive
                    // path (fresh kernel-column evaluations); a verified
                    // root skips it entirely.
                    const double rw = warm_root_d0_[j];
                    if (!f64_isnan(rw) && rw > gp[j] && rw < gp[j + 2]) {
                        double v, _s;
                        ++grad_evals_;
                        fam_->gradfun(rw, dens, ctx, true, false, v, _s);
                        if (v < 0.0)
                            r = rw;
                    }
                    if (f64_isnan(r))
                        r = dfmin({gp[j], gp[j + 1], gp[j + 2]},
                                  {pv[j], pv[j + 1], pv[j + 2]}, dens, ctx, tol);
                    warm_root_d0_[j] = r;
                }
                if (!f64_isnan(r))
                    ans.push_back(r);
            }
        }
        if (pv[0] < 0.0 && (pv[1] - pv[0]) > 0.0)
            ans.push_back(gp[0]);
        if (pv[l - 1] < 0.0 && (pv[l - 1] - pv[l - 2]) < 0.0)
            ans.push_back(gp[l - 1]);
        return ans;
    }

    std::vector<double> solvegrad(const std::vector<double>& dens,
                                  double tol) const {
        if (std::string(fam_->flag()) == "d1")
            return solvegradd1(dens, tol);
        return solvegradd0(dens, tol);
    }

    // Port of `collapse` (always called with the fixed tolerance 1e-6, as in
    // the C++ `computemixdist`).
    void collapse(std::vector<double>& mu0, std::vector<double>& pi0) const {
        const double tol = 1e-6;
        const std::vector<double> dens = fam_->mapping(mu0, pi0);
        const double ll = fam_->lossfunction(dens);
        const double ntol = std::max(tol * 0.1, ll * 1e-16);
        std::vector<double> mu0new = mu0;
        std::vector<double> pi0new = pi0;
        while (true) {
            if (mu0new.size() <= 1)
                break;
            const std::vector<double> d = diff(mu0new);
            double mn = std::numeric_limits<double>::infinity();
            for (double v : d)
                mn = std::min(mn, v);
            const double prec = 10.0 * mn;
            collapsemix(mu0new, pi0new, prec);
            const std::vector<double> densn = fam_->mapping(mu0new, pi0new);
            const double nll = fam_->lossfunction(densn);
            if (nll <= ll + ntol) {
                mu0 = mu0new;
                pi0 = pi0new;
            } else {
                break;
            }
        }
        simplifymix(mu0, pi0);
    }

    // -----------------------------------------------------------------------
    // the outer drivers
    // -----------------------------------------------------------------------

    // Port of `computemixdist` (the outer iteration loop).
    void computemixdist(double tol, long maxit) {
        std::vector<double> mu0 = initpt_;
        double scale = 1.0;
        for (double v : pi0fixed_)
            scale -= v;
        std::vector<double> pi0;
        pi0.reserve(initpr_.size());
        for (double p : initpr_)
            pi0.push_back(p * scale);
        iter_ = 0;
        std::vector<double> dens = fam_->mapping(mu0, pi0);
        double closs = fam_->lossfunction(dens);
        double nloss = f64_nan();

        // env-gated phase profiler (NPFIXEDCOMPY_PROFILE=1)
        const bool prof = std::getenv("NPFIXEDCOMPY_PROFILE") != nullptr;
        double pt_map = 0.0, pt_loss = 0.0, pt_grad = 0.0, pt_wt = 0.0,
               pt_col = 0.0;
        const auto t_iter_start = std::chrono::steady_clock::now();
        while (true) {
            auto t0 = std::chrono::steady_clock::now();
            const std::vector<double> newpoints = solvegrad(dens, tol);
            mu0.insert(mu0.end(), newpoints.begin(), newpoints.end());
            pi0.resize(pi0.size() + newpoints.size(), 0.0);
            sortmix(mu0, pi0);
            pt_grad +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t0)
                    .count();

            if (verbose_ >= 1) {
                std::fprintf(stderr, "Iteration: %ld with loss %g\n", iter_,
                             nloss);
                std::fprintf(stderr, "support points: ");
                for (size_t i = 0; i < mu0.size(); ++i)
                    std::fprintf(stderr, "%g%s", mu0[i],
                                 i + 1 < mu0.size() ? " " : "\n");
                std::fprintf(stderr, "probabilities: ");
                for (size_t i = 0; i < pi0.size(); ++i)
                    std::fprintf(stderr, "%g%s", pi0[i],
                                 i + 1 < pi0.size() ? " " : "\n");
            }
            if (verbose_ >= 2) {
                std::vector<double> gv, gg;
                SolveCtx vctx;
                vctx.dens = &dens;
                fam_->gradfunvec(newpoints, dens, vctx, true, true, gv, gg);
                std::fprintf(stderr, "new points: ");
                for (size_t i = 0; i < newpoints.size(); ++i)
                    std::fprintf(stderr, "%g%s", newpoints[i],
                                 i + 1 < newpoints.size() ? " " : "\n");
                std::fprintf(stderr, "gradient: ");
                for (size_t i = 0; i < gv.size(); ++i)
                    std::fprintf(stderr, "%g%s", gv[i],
                                 i + 1 < gv.size() ? " " : "\n");
                if (std::string(fam_->flag()) == "d1") {
                    std::fprintf(stderr, "gradient derivative: ");
                    for (size_t i = 0; i < gg.size(); ++i)
                        std::fprintf(stderr, "%g%s", gg[i],
                                     i + 1 < gg.size() ? " " : "\n");
                }
                const std::vector<double> d2 = fam_->mapping(mu0, pi0);
                std::fprintf(stderr, "loss:%g\n", fam_->lossfunction(d2));
            }

            t0 = std::chrono::steady_clock::now();
            fam_->computeweights(mu0, pi0, dens);
            pt_wt +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t0)
                    .count();
            if (verbose_ >= 2) {
                std::fprintf(stderr, "After computeweights\n");
                std::fprintf(stderr, "support points: ");
                for (size_t i = 0; i < mu0.size(); ++i)
                    std::fprintf(stderr, "%g%s", mu0[i],
                                 i + 1 < mu0.size() ? " " : "\n");
                std::fprintf(stderr, "probabilities: ");
                for (size_t i = 0; i < pi0.size(); ++i)
                    std::fprintf(stderr, "%g%s", pi0[i],
                                 i + 1 < pi0.size() ? " " : "\n");
                const std::vector<double> d2 = fam_->mapping(mu0, pi0);
                std::fprintf(stderr, "loss:%g\n", fam_->lossfunction(d2));
            }
            t0 = std::chrono::steady_clock::now();
            collapse(mu0, pi0);
            pt_col +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t0)
                    .count();
            if (verbose_ >= 2) {
                std::fprintf(stderr, "After collapse\n");
                std::fprintf(stderr, "support points: ");
                for (size_t i = 0; i < mu0.size(); ++i)
                    std::fprintf(stderr, "%g%s", mu0[i],
                                 i + 1 < mu0.size() ? " " : "\n");
                std::fprintf(stderr, "probabilities: ");
                for (size_t i = 0; i < pi0.size(); ++i)
                    std::fprintf(stderr, "%g%s", pi0[i],
                                 i + 1 < pi0.size() ? " " : "\n");
                const std::vector<double> d2 = fam_->mapping(mu0, pi0);
                std::fprintf(stderr, "loss:%g\n", fam_->lossfunction(d2));
            }
            iter_ += 1;
            t0 = std::chrono::steady_clock::now();
            dens = fam_->mapping(mu0, pi0);
            pt_map +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t0)
                    .count();
            t0 = std::chrono::steady_clock::now();
            nloss = fam_->lossfunction(dens);
            pt_loss +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t0)
                    .count();

            if (closs - nloss < tol) {
                convergence_ = 0;
                break;
            }
            if (iter_ > maxit) {
                convergence_ = 1;
                break;
            }
            closs = nloss;
        }
        if (prof) {
            const double tot = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_iter_start)
                                   .count();
            const std::pair<std::size_t, double> kf = fam_->kernel_fresh();
            std::fprintf(stderr,
                         "PROFILE iters=%ld total=%.1fms  solvegrad=%.1f "
                         "mapping=%.1f  loss=%.1f  weights=%.1f  collapse=%.1f "
                         "evals=%ld certskips=%ld freshcols=%zu freshms=%.1f "
                         "(ms)\n",
                         iter_, tot * 1e3, pt_grad * 1e3, pt_map * 1e3,
                         pt_loss * 1e3, pt_wt * 1e3, pt_col * 1e3,
                         grad_evals_, cert_skips_, kf.first, kf.second);
        }
        resultpt_ = std::move(mu0);
        resultpr_ = std::move(pi0);
    }

    // Port of `estpi0`: estimate the point mass at zero by thresholding the
    // hypothesis statistic. Note (faithful to the C++): after the first
    // `computemixdist(tol)`, the inner `computemixdist` calls all use the
    // defaults `tol = 1e-6, maxit = 100, verbose = 0`.
    void estpi0(double val, double tol) {
        mu0fixed_ = {0.0};
        pi0fixed_ = {0.0};
        set_precompute();
        computemixdist(tol, 100);
        const std::vector<double> densmin =
            fam_->mapping(resultpt_, resultpr_);
        const double minloss = fam_->lossfunction(densmin) + fam_->extrafun();
        const std::vector<double> dens0 = fam_->mapping({0.0}, {1.0});
        const double stat0 = fam_->lossfunction(dens0) + fam_->extrafun();
        if (fam_->hypofun(stat0, minloss) < val) {
            resultpt_ = {0.0};
            resultpr_ = {1.0};
        } else {
            double lb = 0.0, ub = 1.0;
            double flb = minloss, fub = stat0;
            const double sp0 =
                fam_->familydensity(0.0, resultpt_, resultpr_) /
                fam_->familydensity(0.0, {0.0}, {1.0});
            pi0fixed_ = {sp0};
            set_precompute();
            computemixdist(1e-6, 100);
            double ll = [this] {
                const std::vector<double> d =
                    fam_->mapping(resultpt_, resultpr_);
                return fam_->lossfunction(d) + fam_->extrafun();
            }();
            double sp = sp0;
            long iter = 1;
            long guard = 0;
            while (std::abs(fam_->hypofun(ll, minloss) - val) > tol &&
                   std::abs(ub - lb) > tol && guard < 1000) {
                guard++;
                if (verbose_ >= 1) {
                    std::fprintf(stderr, "Iter: %ld lower: %g upper: %g\n",
                                 iter, lb, ub);
                    std::fprintf(stderr, "current val: %g fval: %g\n", sp, ll);
                }
                const double h = fam_->hypofun(ll, minloss) - val;
                if (h < 0.0 && sp > lb) {
                    lb = sp;
                    flb = ll;
                }
                if (h > 0.0 && sp < ub) {
                    ub = sp;
                    fub = ll;
                }
                sp = (lb + ub) / 2.0;

                initpt_ = resultpt_;
                initpr_ = resultpr_;
                pi0fixed_ = {sp};
                set_precompute();
                computemixdist(1e-6, 100);
                ll = [this] {
                    const std::vector<double> d =
                        fam_->mapping(resultpt_, resultpr_);
                    return fam_->lossfunction(d) + fam_->extrafun();
                }();

                // quadratic interpolation through (lb, sp, ub)
                const double a00 = lb * lb;
                const double a01 = lb;
                const double a10 = sp * sp;
                const double a11 = sp;
                const double a20 = ub * ub;
                const double a21 = ub;
                const double b0 = fam_->hypofun(flb, minloss) - val;
                const double b1 = fam_->hypofun(ll, minloss) - val;
                const double b2 = fam_->hypofun(fub, minloss) - val;
                // Cramer's rule for the 3x3 (determinant computed explicitly).
                const double det =
                    a00 * (a11 - a21) - a01 * (a10 - a20) + (a10 * a21 - a11 * a20);
                const double det0 =
                    b0 * (a11 - a21) - a01 * (b1 - b2) + (b1 * a21 - a11 * b2);
                const double det1 =
                    a00 * (b1 - b2) - b0 * (a10 - a20) + (a10 * b2 - b1 * a20);
                const double det2 = a00 * (a11 * b2 - b1 * a21) -
                                    a01 * (a10 * b2 - b1 * a20) +
                                    b0 * (a10 * a21 - a11 * a20);
                const double x10 = det0 / det;
                const double x11 = det1 / det;
                const double x12 = det2 / det;
                const double disc = x11 * x11 - 4.0 * x10 * x12;
                double spnew = disc < 0.0 ? f64_nan()
                                          : (-x11 + std::sqrt(disc)) / (2.0 * x10);
                if (fam_->hypofun(ll, minloss) - val < 0.0) {
                    lb = sp;
                    flb = ll;
                }
                if (fam_->hypofun(ll, minloss) - val > 0.0) {
                    ub = sp;
                    fub = ll;
                }
                if (f64_isnan(spnew) || spnew < lb || spnew > ub)
                    spnew = (lb + ub) / 2.0;
                sp = spnew;

                initpt_ = resultpt_;
                initpr_ = resultpr_;
                pi0fixed_ = {sp};
                set_precompute();
                computemixdist(1e-6, 100);
                ll = [this] {
                    const std::vector<double> d =
                        fam_->mapping(resultpt_, resultpr_);
                    return fam_->lossfunction(d) + fam_->extrafun();
                }();
                iter += 1;
            }
        }
    }

    // Opt-in fast variant of `estpi0` (NOT bit-identical to the legacy path;
    // enable with `estpi0(..., fast=True)`).
    void estpi0_fast(double val, double tol, double inner_tol, bool relax) {
        mu0fixed_ = {0.0};
        pi0fixed_ = {0.0};
        set_precompute();
        computemixdist(tol, 100);
        const std::vector<double> densmin =
            fam_->mapping(resultpt_, resultpr_);
        const double minloss = fam_->lossfunction(densmin) + fam_->extrafun();
        const std::vector<double> dens0 = fam_->mapping({0.0}, {1.0});
        const double stat0 = fam_->lossfunction(dens0) + fam_->extrafun();
        if (fam_->hypofun(stat0, minloss) < val) {
            resultpt_ = {0.0};
            resultpr_ = {1.0};
            return;
        }
        double lb = 0.0, ub = 1.0;
        const double flb = minloss, fub = stat0;
        double sp0 = fam_->familydensity(0.0, resultpt_, resultpr_) /
                     fam_->familydensity(0.0, {0.0}, {1.0});
        double lb_loss = flb;
        double ub_loss = fub;
        long guard = 0;
        while (guard < 1000) {
            guard++;
            // 1) evaluate the current candidate
            initpt_ = resultpt_;
            initpr_ = resultpr_;
            pi0fixed_ = {sp0};
            set_precompute();
            computemixdist(inner_tol, 100);
            double ll = [this] {
                const std::vector<double> d =
                    fam_->mapping(resultpt_, resultpr_);
                return fam_->lossfunction(d) + fam_->extrafun();
            }();
            const double h = fam_->hypofun(ll, minloss) - val;
            if (verbose_ >= 1)
                std::fprintf(stderr, "[fast] iter %ld sp %g ll %g h %g\n", guard,
                             sp0, ll, h);
            // 2) converged, or early-stop when the target is already beaten
            //    from below with a small pi0 (relax mode)
            if (std::abs(h) <= tol || std::abs(ub - lb) <= tol)
                break;
            if (relax && h > 0.0 && sp0 <= 0.1 * ub) {
                pi0fixed_ = {sp0 * 2.0};
                set_precompute();
                computemixdist(inner_tol, 100);
                break;
            }
            // 3) next candidate: quadratic interpolation through the three
            //    current bracket points (lb, sp0, ub)
            const double a00 = lb * lb, a01 = lb;
            const double a10 = sp0 * sp0, a11 = sp0;
            const double a20 = ub * ub, a21 = ub;
            const double b0 = fam_->hypofun(lb_loss, minloss) - val;
            const double b1 = h;
            const double b2 = fam_->hypofun(ub_loss, minloss) - val;
            const double det =
                a00 * (a11 - a21) - a01 * (a10 - a20) + (a10 * a21 - a11 * a20);
            const double det0 =
                b0 * (a11 - a21) - a01 * (b1 - b2) + (b1 * a21 - a11 * b2);
            const double det1 =
                a00 * (b1 - b2) - b0 * (a10 - a20) + (a10 * b2 - b1 * a20);
            const double det2 = a00 * (a11 * b2 - b1 * a21) -
                                a01 * (a10 * b2 - b1 * a20) +
                                b0 * (a10 * a21 - a11 * a20);
            const double x10 = det0 / det;
            const double x11 = det1 / det;
            const double x12 = det2 / det;
            const double disc = x11 * x11 - 4.0 * x10 * x12;
            double spnew = disc < 0.0 ? f64_nan()
                                      : (-x11 + std::sqrt(disc)) / (2.0 * x10);
            // 4) update the bracket with the evaluated point
            if (h < 0.0) {
                lb = sp0;
                lb_loss = ll;
            } else {
                ub = sp0;
                ub_loss = ll;
            }
            if (f64_isnan(spnew) || spnew <= lb || spnew >= ub)
                sp0 = (lb + ub) / 2.0;
            else
                sp0 = spnew;
        }
    }

    // Port of `get_ans`.
    MixResult finish() {
        std::vector<double> mu0new = resultpt_;
        std::vector<double> pi0new = resultpr_;
        mu0new.insert(mu0new.end(), mu0fixed_.begin(), mu0fixed_.end());
        pi0new.insert(pi0new.end(), pi0fixed_.begin(), pi0fixed_.end());
        sortmix(mu0new, pi0new);
        const std::vector<double> dens = fam_->mapping(resultpt_, resultpr_);
        std::vector<double> maxgrad, _g2;
        SolveCtx fctx;
        fctx.dens = &dens;
        fam_->gradfunvec(resultpt_, dens, fctx, true, false, maxgrad, _g2);
        double min_gradient = std::numeric_limits<double>::infinity();
        for (double v : maxgrad)
            min_gradient = std::min(min_gradient, v);
        // Grid-level certificate: the gains at the FINAL (possibly
        // converged) dens across the whole grid. A negative one would
        // mean a direction outside both the support AND the grid (the
        // refined roots only certify sub-intervals), so it is tracked
        // separately — the support-gradient `min_gradient` stays the
        // strict KKT number.
        std::vector<double> gg, _g3;
        fam_->gradfunvec(gridpoints_, dens, fctx, true, false, gg, _g3);
        double gmin = std::numeric_limits<double>::infinity();
        for (double v : gg)
            gmin = std::min(gmin, v);
        grid_gain_ = gmin;
        MixResult r;
        r.pt = std::move(mu0new);
        r.pr = std::move(pi0new);
        r.beta = fam_->beta_value();
        r.family = fam_->family_name();
        r.min_gradient = min_gradient;
        r.grid_gain = gmin;
        r.ll = fam_->lossfunction(dens) + fam_->extrafun();
        r.flag = fam_->flag();
        r.iter = iter_;
        r.convergence = convergence_;
        return r;
    }

private:
    void set_precompute() { fam_->set_fixed(mu0fixed_, pi0fixed_); }

    std::unique_ptr<Family> fam_;
    std::vector<double> mu0fixed_, pi0fixed_;
    std::vector<double> initpt_, initpr_;
    std::vector<double> gridpoints_;
    long iter_;
    int convergence_;
    std::vector<double> resultpt_, resultpr_;
    int verbose_;
    long refine_steps_ = -1;  // -1 = unlimited (the shipped behaviour)
    // Previous-call refined roots (the always-on CNM working-set hot
    // start; see the constructor), indexed by grid interval (d1:
    // sign-change pair `i`; d0: triple `j`). NaN = no candidate yet.
    // `mutable`: the search methods are const like the rest of the solver.
    mutable std::vector<double> warm_root_d1_, warm_root_d0_;
    // Grid-level certificate of the last `finish()` call (see there).
    mutable double grid_gain_ = std::numeric_limits<double>::quiet_NaN();
    // Count of (mu, dens) gradient evaluations inside the loop (scalar
    // gradfun = 1 each, vectorised gradfunvec = one per point); the A/B
    // "less work per step" knob is read against this (PROFILE line only).
    mutable long grad_evals_ = 0;
    // 1-D crossing certificate (see the constructor and `solvegradd1`): the
    // gate + the per-run count of intervals certified (and skipped); the
    // count is reported on the PROFILE line only.
    bool cert_1d_ = false;
    mutable long cert_skips_ = 0;
};

}  // namespace npfc

#endif  // NPFIC_ENGINE_H
