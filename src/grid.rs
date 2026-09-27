//! Grid / initial-mix construction — exact ports of `nspmix::whist`,
//! `nspmix:::gridpoints.npnorm` and `nspmix:::initial.npnorm`, plus the
//! `hist.default` integer-breaks path they rely on. These are R-exact: the
//! support grid and the starting mixing distribution are what make the final
//! estimate reproduce the R package.

use crate::misc::diff;
use crate::pretty::pretty_range;

/// Port of `hist.default(x, breaks = K, plot = FALSE)` for an *integer*
/// `breaks = K` (the only form `whist` ever uses). Returns
/// `(breaks, mids, density, counts)`, with `density = counts / (n * h)` (the
/// `freq`-agnostic branch: for a numeric scalar `breaks` the `equidist` test
/// decides `freq`, but `density` is `counts/(n*h)` either way in the
/// histogram object that `whist` consumes).
///
/// Binning matches R's `C_BinCount` with `right = TRUE`,
/// `include.lowest = TRUE`, `fuzz = 1e-7` and the `diddle` rule
/// (`median(h)` when `nB > 5`, `diff(range(x))` when `nB <= 3`,
/// `min(h[h > 0])` otherwise): bin `i` is
/// `(breaks[i] - diddle, breaks[i+1] + diddle]`.
pub fn hist_int(x: &[f64], breaks_scalar: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    // x is already finite & sorted by whist
    let n = x.len();
    let (minx, maxx) = (x[0], x[n - 1]);
    let breaks = pretty_range(minx, maxx, breaks_scalar as i32, 1);
    hist_from_breaks(x, &breaks, n)
}

/// Histogram binning with explicit (numeric) breaks — the shared core of
/// `hist.default` for both the integer-`breaks` and vector-`breaks` forms
/// (diddle rule: `median(h)` when `nB > 5`, `range(x)` when `nB <= 3`,
/// `min(h)` otherwise; bin `i` is `(breaks[i] - diddle, breaks[i+1] + diddle]`).
fn hist_from_breaks(x: &[f64], breaks: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    if breaks.len() <= 1 {
        // degenerate data: one unique value
        let m = x[0];
        return (
            vec![m - 0.5, m + 0.5],
            vec![m],
            vec![1.0 / n as f64],
            vec![n],
        );
    }
    let nb = breaks.len();
    let h = diff(breaks);
    let hmin = h.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let hmed = median_sorted(&h);
    let (minx, maxx) = (x[0], x[n - 1]);
    let diddle = 1e-7 * if nb > 5 {
        hmed
    } else if nb <= 3 {
        maxx - minx
    } else {
        hmin
    };
    let mut counts = vec![0usize; nb - 1];
    for &xi in x {
        // bin i: (breaks[i]-diddle, breaks[i+1]+diddle], 1-based i
        let mut b = 0usize;
        while b + 1 < nb && xi > breaks[b + 1] + diddle {
            b += 1;
        }
        // include.lowest: xi == breaks[0] goes to bin 0
        counts[b] += 1;
    }
    let dens: Vec<f64> = counts
        .iter()
        .zip(h.iter())
        .map(|(&c, &hw)| c as f64 / (n as f64 * hw))
        .collect();
    let mids: Vec<f64> = (0..nb - 1).map(|i| 0.5 * (breaks[i] + breaks[i + 1])).collect();
    (breaks.to_vec(), mids, dens, counts)
}

/// `whist` with explicit numeric breaks (as `initial.nppois` uses): returns
/// `(breaks, mids, density)` with the weighted density.
pub fn whist_breaks(x: &[f64], w: &[f64], breaks_in: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let xs: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
    let ws: Vec<f64> = idx.iter().map(|&i| w[i]).collect();
    let (breaks, mids, _dens, _counts) = hist_from_breaks(&xs, breaks_in, n);
    let wsum: f64 = ws.iter().sum();
    let nb = breaks.len();
    let h = diff(&breaks);
    let hmin = h.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let diddle = 1e-7 * if nb > 5 {
        median_sorted(&h)
    } else if nb <= 3 {
        xs[n - 1] - xs[0]
    } else {
        hmin
    };
    let mut wcounts = vec![0.0f64; nb - 1];
    for j in 0..n {
        let mut b = 0usize;
        while b + 1 < nb && xs[j] > breaks[b + 1] + diddle {
            b += 1;
        }
        wcounts[b] += ws[j];
    }
    let dens: Vec<f64> = wcounts
        .iter()
        .zip(h.iter())
        .map(|(&c, &hw)| c / (wsum * hw))
        .collect();
    (breaks, mids, dens)
}

fn median_sorted(h: &[f64]) -> f64 {
    if h.is_empty() {
        return f64::NAN;
    }
    let mut s = h.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = s.len() / 2;
    if s.len() % 2 == 1 {
        s[m]
    } else {
        0.5 * (s[m - 1] + s[m])
    }
}

/// Port of `nspmix::whist(x, w, breaks = K, plot = FALSE, freq = FALSE)`.
/// Returns `(breaks, mids, density)` where `density = wcount / (W * h)`,
/// `wcount` is the sum of the weights per bin and `W = sum(w)`.
pub fn whist(x: &[f64], w: &[f64], breaks_scalar: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len();
    // order(x) / order(w) — stable ascending sort, ties in original order
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let xs: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
    let ws: Vec<f64> = idx.iter().map(|&i| w[i]).collect();
    let (breaks, mids, _dens, _counts) = hist_int(&xs, breaks_scalar);
    let wsum: f64 = ws.iter().sum();
    // weighted bin counts (same bins as hist_int above)
    let nb = breaks.len();
    let h = diff(&breaks);
    let hmin = h.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let diddle = 1e-7 * if nb > 5 {
        median_sorted(&h)
    } else if nb <= 3 {
        xs[n - 1] - xs[0]
    } else {
        hmin
    };
    let mut wcounts = vec![0.0f64; nb - 1];
    for j in 0..n {
        let mut b = 0usize;
        while b + 1 < nb && xs[j] > breaks[b + 1] + diddle {
            b += 1;
        }
        wcounts[b] += ws[j];
    }
    let dens: Vec<f64> = wcounts
        .iter()
        .zip(h.iter())
        .map(|(&c, &hw)| c / (wsum * hw))
        .collect();
    (breaks, mids, dens)
}

/// Port of `nspmix:::gridpoints.npnorm(x, beta, grid = 100)`.
///
/// `x` is the `disc`-style data: `v` (support points) and `w` (weights).
pub fn gridpoints_npnorm(v: &[f64], w: &[f64], beta: f64, grid: usize) -> Vec<f64> {
    let minv = v.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let maxv = v.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let breaks_scalar = (diff(&[minv, maxv])[0] / (5.0 * beta)).ceil().max(5.0) as usize;
    let (br, _mids, dens) = whist(v, w, breaks_scalar);
    // i = r$density != 0
    let i0: Vec<bool> = dens.iter().map(|&d| d != 0.0).collect();
    // i = i0 | c(i0[-1], FALSE) | c(FALSE, i0[-length(i0)])
    let nb = i0.len();
    let mut i = vec![false; nb];
    for k in 0..nb {
        let prev = if k > 0 { i0[k - 1] } else { false };
        let next = if k + 1 < nb { i0[k + 1] } else { false };
        i[k] = i0[k] | prev | next;
    }
    let m = i.iter().filter(|&&f| f).count() as usize;
    let k = (grid as f64 / m as f64).ceil().max(10.0) as usize;
    let d = br[1] - br[0];
    // s = r$breaks[-length(r$breaks)][i]
    // s = r$breaks[-length(r$breaks)][i]: the first (nb) breaks, filtered by i
    let out: Vec<f64> = (0..nb)
        .filter(|&q| i[q])
        .flat_map(|q| {
            let s = br[q];
            (0..k).map(move |j| s + d * ((j as f64 + 1.0) - 0.5) / k as f64)
        })
        .collect();
    let mut vout = Vec::with_capacity(2 + out.len());
    vout.push(minv);
    vout.extend(out);
    vout.push(maxv);
    vout
}

/// Port of `nspmix:::initial.npnorm(x, beta = NULL, mix = NULL, kmax = NULL)`.
///
/// `v`/`w` are the data points/weights; `mix` is the user-supplied starting
/// mixing distribution (`None` for the default). Returns `(beta, pt, pr)`.
pub fn initial_npnorm(
    v: &[f64],
    w: &[f64],
    beta: Option<f64>,
    mix_pt: &[f64],
    mix_pr: &[f64],
) -> (f64, Vec<f64>, Vec<f64>) {
    let beta = beta.unwrap_or(1.0);
    let have_mix = !mix_pt.is_empty();
    if have_mix {
        let mut pt = mix_pt.to_vec();
        let mut pr = if mix_pr.is_empty() {
            vec![1.0 / mix_pt.len() as f64; mix_pt.len()]
        } else {
            mix_pr.to_vec()
        };
        disc(&mut pt, &mut pr);
        return (beta, pt, pr);
    }
    let minv = v.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let maxv = v.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let breaks_scalar = (diff(&[minv, maxv])[0] / (5.0 * beta)).ceil().max(5.0) as usize;
    let (_br, mids, dens) = whist(v, w, breaks_scalar);
    let nz: Vec<usize> = dens
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d != 0.0)
        .map(|(q, _)| q)
        .collect();
    let mut pt: Vec<f64> = nz.iter().map(|&q| mids[q]).collect();
    let mut pr: Vec<f64> = nz.iter().map(|&q| dens[q]).collect();
    disc(&mut pt, &mut pr);
    (beta, pt, pr)
}

/// Port of `nspmix:::gridpoints.nppois(x, beta, grid = 100)`. For the
/// Poisson family the support space is `[0, Inf)`, so this reduces to the
/// squared `seq` over the square-root range of the data:
/// `seq(sqrt(min v), sqrt(max v), length = grid)^2`.
pub fn gridpoints_nppois(v: &[f64], _beta: f64, grid: usize) -> Vec<f64> {
    let minv = v.iter().copied().fold(f64::INFINITY, |a, b| a.min(b)).max(0.0);
    let maxv = v.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b)).max(0.0);
    let lo = minv.sqrt();
    let hi = maxv.sqrt();
    let g = grid.max(2);
    let step = (hi - lo) / (g as f64 - 1.0);
    (0..g).map(|i| {
        let t = lo + step * i as f64;
        t * t
    }).collect()
}

/// Port of `nspmix:::initial.nppois(x, beta = NULL, mix = NULL, kmax = NULL)`
/// for the default `kmax = NULL` path (the only form `nppoisll` uses).
/// Returns `(beta, pt, pr)`.
pub fn initial_nppois(
    v: &[f64],
    w: &[f64],
    beta: Option<f64>,
    mix_pt: &[f64],
    mix_pr: &[f64],
) -> (f64, Vec<f64>, Vec<f64>) {
    let beta = beta.unwrap_or(1.0);
    if !mix_pt.is_empty() {
        let mut pt = mix_pt.to_vec();
        let mut pr = if mix_pr.is_empty() {
            vec![1.0 / mix_pt.len() as f64; mix_pt.len()]
        } else {
            mix_pr.to_vec()
        };
        disc(&mut pt, &mut pr);
        return (beta, pt, pr);
    }
    let minv = v.iter().copied().fold(f64::INFINITY, |a, b| a.min(b)).max(0.0);
    let maxv = v.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b)).max(0.0);
    let mi = (minv.sqrt().floor() - 1.0).max(0.0) as i64;
    let ma = maxv.sqrt().ceil().max(1.0) as i64;
    // breaks = (mi:ma)^2
    let breaks: Vec<f64> = (mi..=ma).map(|k| (k as f64) * (k as f64)).collect();
    let (_br, mids, dens) = whist_breaks(v, w, &breaks);
    let nz: Vec<usize> = dens
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d != 0.0)
        .map(|(q, _)| q)
        .collect();
    let mut pt: Vec<f64> = nz.iter().map(|&q| mids[q]).collect();
    let mut pr: Vec<f64> = nz.iter().map(|&q| dens[q]).collect();
    disc(&mut pt, &mut pr);
    (beta, pt, pr)
}

/// Port of `nspmix:::disc(pt, pr, sort = TRUE, collapse = FALSE)`: sorts by
/// `pt` (stable) and renormalises `pr` to sum to 1.
pub fn disc(pt: &mut Vec<f64>, pr: &mut Vec<f64>) {
    let k = pt.len().max(pr.len());
    while pt.len() < k {
        pt.push(*pt.last().unwrap_or(&0.0));
    }
    while pr.len() < k {
        pr.push(1.0);
    }
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| pt[a].partial_cmp(&pt[b]).unwrap_or(std::cmp::Ordering::Equal));
    let spt: Vec<f64> = idx.iter().map(|&i| pt[i]).collect();
    let spr: Vec<f64> = idx.iter().map(|&i| pr[i]).collect();
    let s: f64 = spr.iter().sum();
    for j in 0..k {
        pt[j] = spt[j];
        pr[j] = if s != 0.0 { spr[j] / s } else { spr[j] };
    }
    pt.truncate(k);
    pr.truncate(k);
}

#[cfg(test)]
mod tests {
    use super::*;

    // reference (trace_initial.R, set.seed(123), rnorm(500, c(0, 2))):
    // range -2.465898 5.24104, K = 5, breaks -4 -2 0 2 4 6,
    // mids -3 -1 1 3 5.
    #[test]
    fn test_gridpoints_shape() {
        // a single point dataset: range 0, K = 5
        let v = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let w = vec![1.0; 11];
        let g = gridpoints_npnorm(&v, &w, 1.0, 100);
        // range 0..10, breaks = pretty(0,10,n=5) = 0,2,4,6,8,10 ; m = 5 (all nonzero)
        // k = max(ceiling(100/5),10) = 20 ; d = 2
        // points: 0, then s in {0,2,4,6,8} + 2*(1:20-0.5)/20, then 10
        assert_eq!(g[0], 0.0);
        assert_eq!(*g.last().unwrap(), 10.0);
        assert_eq!(g.len(), 2 + 5 * 20);
        // second gridpoint: 0 + 2*(0.5/20) = 0.05
        assert!((g[1] - 0.05).abs() < 1e-12);
    }
}
