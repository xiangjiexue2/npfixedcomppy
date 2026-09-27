//! R's `pretty()` — an exact port of `R_pretty` (`src/appl/pretty.c`) plus the
//! `pretty.default` wrapper and the `seq.int(length.out = k)` arithmetic from
//! `src/main/seq.c`. The histogram-based grid/initial-mix construction in
//! `nspmix` (and therefore `gridpoints.npnorm` / `initial.npnorm`) depends on
//! these exact values, so the port reproduces R's floating-point arithmetic
//! line-for-line.

const DBL_EPS: f64 = f64::EPSILON; // 2.220446049250313e-16
const ROUNDING_EPS: f64 = 1e-10;
const INT_MAX: f64 = 2147483647.0;
const INT_MIN: f64 = -2147483648.0;

/// Exact port of `R_pretty` (`src/appl/pretty.c`).
///
/// Constructs the pretty-unit covering `[lo, up]` with about `*ndiv`
/// intervals, adjusting `*lo`/`*up` (when `return_bounds != 0`) so the result
/// covers the original range, and setting `*ndiv` to the actual number of
/// intervals. Returns the unit.
///
/// `h`, `h5`, `f_min` are the `high_u_fact` array `(high.u.bias, u5.bias,
/// f.min)` of `pretty.default`: `(1.5, 0.5 + 1.5*1.5, 2^-20)`.
pub fn r_pretty(
    lo: &mut f64,
    up: &mut f64,
    ndiv: &mut i32,
    min_n: i32,
    shrink_sml: f64,
    h: f64,
    h5: f64,
    f_min: f64,
    eps_correction: i32,
    return_bounds: i32,
) -> f64 {
    let lo_ = *lo;
    let up_ = *up;
    let dx = up_ - lo_;
    let (mut cell, i_small) = if dx == 0.0 && up_ == 0.0 {
        (1.0, true)
    } else {
        let c0 = lo_.abs().max(up_.abs());
        let mut u = 1.0 + if h5 >= 1.5 * h + 0.5 {
            1.0 / (1.0 + h)
        } else {
            1.5 / (1.0 + h5)
        };
        u *= (1i32).max(*ndiv) as f64 * DBL_EPS;
        let i_small = dx < c0 * u * 3.0;
        (c0, i_small)
    };

    if i_small {
        if cell > 10.0 {
            cell = 9.0 + cell / 10.0;
        }
        cell *= shrink_sml;
        if min_n > 1 {
            cell /= min_n as f64;
        }
    } else {
        cell = dx;
        if dx.is_finite() {
            if *ndiv > 1 {
                cell /= *ndiv as f64;
            }
        } else {
            // up - lo = +Inf (overflow; both finite)
            cell = up_ / *ndiv as f64 - lo_ / *ndiv as f64;
        }
    }

    const MAX_F: f64 = 1.25;
    let mut subsmall = f_min * f64::MIN;
    if subsmall == 0.0 {
        subsmall = f64::MIN;
    }
    if cell < subsmall {
        cell = subsmall;
    } else if cell > f64::MAX / MAX_F {
        cell = f64::MAX / MAX_F;
    }

    // base <= cell < 10*base
    let base = 10f64.powf(cell.log10().floor());

    // unit in {1,2,5,10}*base, biased by h / h5
    let mut unit = base;
    if (2.0 * base) - cell < h * (cell - unit) {
        unit = 2.0 * base;
    }
    if (5.0 * base) - cell < h5 * (cell - unit) {
        unit = 5.0 * base;
    }
    if (10.0 * base) - cell < h * (cell - unit) {
        unit = 10.0 * base;
    }

    let mut ns = (lo_ / unit + ROUNDING_EPS).floor();
    let mut nu = (up_ / unit - ROUNDING_EPS).ceil();

    if eps_correction != 0 && (eps_correction > 1 || !i_small) {
        // pretty.default always calls with eps.correct = 0; the correction is
        // implemented faithfully for completeness.
        let e = DBL_EPS;
        let d_max = f64::MAX * (1.0 - e);
        if lo_ < 0.0 {
            *lo *= 1.0 + e;
        } else if lo_ > 0.0 {
            *lo *= 1.0 - e;
        } else {
            *lo = -unit.min(f64::MIN);
        }
        if up_ < 0.0 {
            *up *= 1.0 - e;
        } else if up_ > 0.0 {
            if up_ < d_max {
                *up *= 1.0 + e;
            }
        } else {
            *up = unit.min(f64::MIN);
        }
    }

    // safety cap: in degenerate cases (inf/nan) the C loops could spin forever
    let mut guard: i32 = 0;
    while ns * unit > *lo + ROUNDING_EPS * unit && guard < 1_000_000 {
        ns -= 1.0;
        guard += 1;
    }
    guard = 0;
    while !(ns * unit).is_finite() && guard < 1_000_000 {
        ns += 1.0;
        guard += 1;
    }
    guard = 0;
    while nu * unit < *up - ROUNDING_EPS * unit && guard < 1_000_000 {
        nu += 1.0;
        guard += 1;
    }
    guard = 0;
    while !(nu * unit).is_finite() && guard < 1_000_000 {
        nu -= 1.0;
        guard += 1;
    }

    let k = (0.5 + nu - ns) as i32; // (int)(0.5 + nu - ns)
    if k < min_n {
        let add = min_n - k;
        if lo_ == 0.0 && ns == 0.0 && up_ != 0.0 {
            nu += add as f64;
        } else if up_ == 0.0 && nu == 0.0 && lo_ != 0.0 {
            ns -= add as f64;
        } else if ns >= 0.0 {
            nu += (add / 2) as f64;
            ns -= ((add / 2) + (add % 2)) as f64;
        } else {
            ns -= (add / 2) as f64;
            nu += ((add / 2) + (add % 2)) as f64;
        }
        *ndiv = min_n;
    } else {
        *ndiv = k;
    }

    if return_bounds != 0 {
        if ns * unit < *lo {
            *lo = ns * unit;
        }
        if nu * unit > *up {
            *up = nu * unit;
        }
    } else {
        *lo = ns;
        *up = nu;
    }
    unit
}

/// `seq.int(from, to, length.out = lout)` from `src/main/seq.c` (the
/// `by == R_MissingArg` branch). Replicates R's exact arithmetic, including
/// the integer fast path.
pub fn seq_int_len(from: f64, to: f64, lout: usize) -> Vec<f64> {
    if lout == 0 {
        return Vec::new();
    }
    let (mut rfrom, rto) = (from, to);
    let mut rby = 0.0;
    let mut finite_del = false;
    if lout > 2 {
        let nint = (lout - 1) as f64;
        let del = rto - rfrom;
        if del.is_finite() {
            rby = del / nint;
            finite_del = true;
        } else {
            rby = rto / nint - rfrom / nint;
        }
    }
    let int_ok = rfrom <= INT_MAX
        && rfrom >= INT_MIN
        && rto <= INT_MAX
        && rto >= INT_MIN
        && rfrom.fract() == 0.0
        && (lout <= 1 || rto.fract() == 0.0)
        && (lout <= 2 || rby.fract() == 0.0);
    let mut ans = vec![0.0f64; lout];
    if int_ok {
        ans[0] = rfrom as i64 as f64;
        if lout > 1 {
            ans[lout - 1] = rto as i64 as f64;
        }
        for i in 1..lout.saturating_sub(1) {
            ans[i] = (rfrom + i as f64 * rby) as i64 as f64;
        }
    } else {
        ans[0] = rfrom;
        if lout > 1 {
            ans[lout - 1] = rto;
        }
        if lout > 2 {
            if finite_del {
                for i in 1..lout - 1 {
                    ans[i] = rfrom + i as f64 * rby;
                }
            } else {
                rfrom /= 4.0;
                rby /= 4.0;
                for i in 1..lout - 1 {
                    ans[i] = (rfrom + i as f64 * rby) * 4.0; // ldexp(y, 2)
                }
            }
        }
    }
    ans
}

/// R's `pretty(x, n, min.n = n %/% 3, shrink.sml = 0.75, high.u.bias = 1.5,
/// u5.bias = 0.5 + 1.5*high.u.bias, f.min = 2^-20, eps.correct = 0,
/// bounds = TRUE)` for a numeric vector `x`.
///
/// This is the function `hist.default` uses for integer `breaks`, and hence
/// what `nspmix::whist` (and `gridpoints.npnorm` / `initial.npnorm`) rely on.
pub fn pretty(x: &[f64], n: i32) -> Vec<f64> {
    let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Vec::new();
    }
    let mut lo = finite.iter().copied().fold(f64::INFINITY, |a, b| a.min(b));
    let mut up = finite.iter().copied().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let min_n = n / 3;
    let mut ndiv = n;
    let h = 1.5;
    let h5 = 0.5 + 1.5 * h;
    let f_min = 2f64.powi(-20);
    r_pretty(&mut lo, &mut up, &mut ndiv, min_n, 0.75, h, h5, f_min, 0, 1);

    // s = seq.int(z$l, z$u, length.out = n + 1L)
    let mut s = seq_int_len(lo, up, (ndiv + 1) as usize);
    // eps.correct == 0 zero-correction
    if ndiv != 0 {
        let delta = (up - lo) / ndiv as f64;
        for v in s.iter_mut() {
            if v.abs() < 1e-14 * delta {
                *v = 0.0;
            }
        }
    }
    s
}

/// Convenience: `pretty(c(lo, hi), n, min.n = 1)` as used by
/// `hist.default`'s integer-breaks branch.
pub fn pretty_range(lo: f64, hi: f64, n: i32, min_n: i32) -> Vec<f64> {
    let mut lo = lo;
    let mut up = hi;
    let mut ndiv = n;
    let h = 1.5;
    let h5 = 0.5 + 1.5 * h;
    let f_min = 2f64.powi(-20);
    r_pretty(&mut lo, &mut up, &mut ndiv, min_n, 0.75, h, h5, f_min, 0, 1);
    let mut s = seq_int_len(lo, up, (ndiv + 1) as usize);
    if ndiv != 0 {
        let delta = (up - lo) / ndiv as f64;
        for v in s.iter_mut() {
            if v.abs() < 1e-14 * delta {
                *v = 0.0;
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-15 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn test_pretty_basics() {
        // values verified against R 4.6.1
        assert_eq!(pretty(&[0.0, 1.0, 15.0], 5), vec![0.0, 5.0, 10.0, 15.0]);
        assert_eq!(pretty(&[0.0, 20.0], 5), vec![0.0, 5.0, 10.0, 15.0, 20.0]);
        assert_eq!(pretty(&[-2.465898, 5.24104], 5), vec![-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]);
        // min.n = 1 path (used by hist for integer breaks)
        assert_eq!(
            pretty_range(-2.465898, 5.24104, 5, 1),
            vec![-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn test_pretty_unit_selection() {
        // unit = 0.1, with seq.int length.out arithmetic (not k*unit)
        let v = pretty_range(-0.01, 3.9, 5, 1);
        assert_eq!(v.len(), 6);
        for (a, b) in v.iter().zip([&-1.0, &0.0, &1.0, &2.0, &3.0, &4.0].iter()) {
            assert!(close(*a, **b), "{a} vs {b}");
        }
    }
}
