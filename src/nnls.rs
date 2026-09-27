//! Non-negative least squares (Lawson & Hanson), a faithful port of the
//! `k = 0` case of the R package's `pnnls.f` (netlib `NNLS`), plus the
//! `pnnlssum_` / `pnnqp_` weight-subproblem wrappers from `miscfuns.h`.
//!
//! `nalgebra`'s `DMatrix` is column-major with leading dimension = number of
//! rows, matching the Fortran `A(MDA, N)` layout, so `a[(r, c)]` mirrors
//! `A(r+1, c+1)`. Counters (`nsetp`, `npp1`, `iz1`, …) are kept 1-based to
//! track the Fortran line-for-line; only the matrix/vector indexing converts.

use nalgebra::{DMatrix, DVector};

/// G1: orthogonal rotation mapping `(a, b) -> (sig, 0)`.
#[inline]
fn g1(a: f64, b: f64) -> (f64, f64, f64) {
    if a.abs() > b.abs() {
        let xr = b / a;
        let yr = (1.0 + xr * xr).sqrt();
        let c = a.signum() / yr;
        let s = c * xr;
        (c, s, a.abs() * yr)
    } else if b != 0.0 {
        let xr = a / b;
        let yr = (1.0 + xr * xr).sqrt();
        let s = b.signum() / yr;
        let c = s * xr;
        (c, s, b.abs() * yr)
    } else {
        // Fortran G1 zero case: C=1, S=0 (identity), SIG=0.
        (1.0, 0.0, 0.0)
    }
}

/// H12 (construct, NCV = 0): build a Householder reflection from the pivot
/// vector `col[lpivot0..m)`, setting `*up` and `col[lpivot0] = cl`.
fn hh_construct(col: &mut [f64], lpivot0: usize, m: usize, up: &mut f64) {
    let l1 = lpivot0 + 1; // first row zeroed (0-based)
    let mut cl = col[lpivot0].abs();
    for j in l1..m {
        cl = cl.max(col[j].abs());
    }
    if cl <= 0.0 {
        return;
    }
    let clinv = 1.0 / cl;
    let mut sm = (col[lpivot0] * clinv) * (col[lpivot0] * clinv);
    for j in l1..m {
        let v = col[j] * clinv;
        sm += v * v;
    }
    cl = cl * sm.sqrt();
    // H12: `IF (U(1,LPIVOT) <= 0) GOTO 50; CL=-CL` — negate only when the
    // pivot element is strictly positive (numerically safe Householder).
    if col[lpivot0] > 0.0 {
        cl = -cl;
    }
    *up = col[lpivot0] - cl;
    col[lpivot0] = cl;
}

/// H12 (apply): apply the reflection `(up, sub)` starting at row `lpivot0`,
/// where `cl` is the post-construct pivot value `col[lpivot0]`.
#[inline]
fn hh_apply(target: &mut [f64], sub: &[f64], lpivot0: usize, m: usize, up: f64, cl: f64) {
    let l1 = lpivot0 + 1;
    let b = up * cl;
    if b >= 0.0 {
        return;
    }
    let b = 1.0 / b;
    let mut sm = target[lpivot0] * up;
    for i in l1..m {
        sm += target[i] * sub[i - l1];
    }
    if sm == 0.0 {
        return;
    }
    sm *= b;
    target[lpivot0] += sm * up;
    for i in l1..m {
        target[i] += sm * sub[i - l1];
    }
}

/// Back-substitution on the implicit upper-triangular factor (label 400).
/// 1-based `nsetp`; `index` maps a set position -> 1-based column number.
fn solve_tri(a: &DMatrix<f64>, index: &[i32], nsetp: usize, zz: &mut DVector<f64>) {
    let mut jj: i32 = 0;
    for l in 1..=nsetp {
        let ip = nsetp + 1 - l;
        if l != 1 {
            // Fortran label 400: DO 410 II=1,IP (INCLUSIVE: II runs 1..ip);
            // ZZ(IP+1) in 1-based is zz[ip] in 0-based.
            for ii in 1..=ip {
                zz[ii - 1] -= a[(ii - 1, jj as usize - 1)] * zz[ip];
            }
        }
        jj = index[ip - 1];
        zz[ip - 1] /= a[(ip - 1, jj as usize - 1)];
    }
}

/// Standard non-negative least squares (the `k = 0` case of `pnnls.f`).
/// Solves `min ||A x - b||_2  s.t.  x >= 0`. `a` is `m x n`, `b` length `m`;
/// returns `x` length `n`. `a` and `b` are consumed (overwritten internally).
pub fn nnls(mut a: DMatrix<f64>, mut b: DVector<f64>, n: usize) -> DVector<f64> {
    let m = a.nrows();
    let mut x: DVector<f64> = DVector::zeros(n);
    let mut index: Vec<i32> = (1..=n as i32).collect();
    let mut w = DVector::zeros(n);
    let mut zz: DVector<f64>;
    let mut iter = 0;
    let itmax = 3 * n;
    let mut iz1 = 1;
    let iz2 = n;
    let mut nsetp = 0;
    let mut npp1 = 1;
    // Fortran `GO TO 60` after a rejection skips the dual recomputation;
    // mirror that with a flag so a rejected column is not re-selected.
    let mut skip_dual = false;

    loop {
        if iz1 > iz2 || nsetp >= m {
            break;
        }
        // dual vector for set Z (skipped right after a rejection)
        if !skip_dual {
            for iz in iz1..=iz2 {
                let j = index[iz - 1];
                let j0 = (j - 1) as usize;
                let mut sm = 0.0;
                for l in npp1..=m {
                    sm += a[(l - 1, j0)] * b[l - 1];
                }
                w[j0] = sm;
            }
        }
        skip_dual = false;
        // largest positive dual
        let (mut wmax, mut izmax) = (0.0f64, 0usize);
        for iz in iz1..=iz2 {
            let j = index[iz - 1];
            if w[(j - 1) as usize] > wmax {
                wmax = w[(j - 1) as usize];
                izmax = iz;
            }
        }
        if wmax <= 0.0 {
            break;
        }
        let iz = izmax;
        let j = index[iz - 1];
        let j0 = (j - 1) as usize;
        let asave = a[(npp1 - 1, j0)];
        // linear-independence test
        let mut unorm = 0.0;
        for l in 1..=nsetp {
            unorm += a[(l - 1, j0)] * a[(l - 1, j0)];
        }
        unorm = unorm.sqrt();
        // construct the Householder reflection on column j
        let (up, sub) = {
            let mut col = a.column_mut(j0);
            let cslice = col.as_mut_slice();
            let mut up = 0.0;
            hh_construct(cslice, npp1 - 1, m, &mut up);
            let sub = cslice[npp1..m].to_vec();
            (up, sub)
        };
        let cl = a[(npp1 - 1, j0)];
        let mut accept = false;
        if unorm + cl.abs() * 0.01 - unorm > 0.0 {
            let mut zzl = b.clone();
            hh_apply(zzl.as_mut_slice(), &sub, npp1 - 1, m, up, cl);
            let ztest = zzl[npp1 - 1] / cl;
            accept = ztest > 0.0;
            if accept {
                b = zzl;
            }
        }
        if !accept {
            // REJECT: restore the column, zero its dual, and re-scan the
            // existing W without recomputing it (Fortran `GO TO 60`).
            // Recomputing the dual here would restore the rejected column's
            // positive value, re-select it, and loop forever on
            // near-dependent columns (the CLL hang).
            a[(npp1 - 1, j0)] = asave;
            w[j0] = 0.0;
            skip_dual = true;
            continue;
        }
        // commit: move j from Z to P
        index[iz - 1] = index[iz1 - 1];
        index[iz1 - 1] = j;
        iz1 += 1;
        nsetp = npp1;
        npp1 += 1;
        for jz in iz1..=iz2 {
            let jj = index[jz - 1];
            let jj0 = (jj - 1) as usize;
            hh_apply(a.column_mut(jj0).as_mut_slice(), &sub, npp1 - 2, m, up, cl);
        }
        for l in npp1..=m {
            a[(l - 1, j0)] = 0.0;
        }
        w[j0] = 0.0;
        // Fortran label 400 reads ZZ() as the work area holding the
        // transformed B (Q*B) at this point.
        zz = b.clone();
        solve_tri(&a, &index, nsetp, &mut zz);
        // secondary loop
        'sec: loop {
            iter += 1;
            if iter > itmax {
                break;
            }
            let (mut alpha, mut jj) = (2.0f64, 0usize);
            for ip in 1..=nsetp {
                let l = index[ip - 1];
                let l0 = (l - 1) as usize;
                if zz[ip - 1] <= 0.0 {
                    let t = -x[l0] / (zz[ip - 1] - x[l0]);
                    if alpha > t {
                        alpha = t;
                        jj = ip;
                    }
                }
            }
            if alpha == 2.0 {
                for ip in 1..=nsetp {
                    let i = index[ip - 1];
                    x[(i - 1) as usize] = zz[ip - 1];
                }
                break 'sec;
            }
            for ip in 1..=nsetp {
                let l = index[ip - 1];
                let l0 = (l - 1) as usize;
                x[l0] += alpha * (zz[ip - 1] - x[l0]);
            }
            // label 260: move an infeasible coefficient from P back to Z
            'mv: loop {
                let i = index[jj - 1];
                let i0 = (i - 1) as usize;
                x[i0] = 0.0;
                if jj != nsetp {
                    jj += 1;
                    for j2 in jj..=nsetp {
                        let ii = index[j2 - 1];
                        let ii0 = (ii - 1) as usize;
                        index[j2 - 2] = index[j2 - 1];
                        // Fortran: CALL G1 (A(J-1,II), A(J,II), CC, SS, A(J-1,II)) — the
                        // fifth argument stores SIG back into A(J-1,II), i.e. the rotated
                        // pivot element; A(J,II) is then zeroed explicitly.
                        let (cc, ss, sig) = g1(a[(j2 - 2, ii0)], a[(j2 - 1, ii0)]);
                        a[(j2 - 2, ii0)] = sig;
                        a[(j2 - 1, ii0)] = 0.0;
                        for l in 0..n {
                            if l != ii0 {
                                let temp = a[(j2 - 2, l)];
                                a[(j2 - 2, l)] = cc * temp + ss * a[(j2 - 1, l)];
                                a[(j2 - 1, l)] = -ss * temp + cc * a[(j2 - 1, l)];
                            }
                        }
                        let temp = b[j2 - 2];
                        b[j2 - 2] = cc * temp + ss * b[j2 - 1];
                        b[j2 - 1] = -ss * temp + cc * b[j2 - 1];
                    }
                }
                npp1 = nsetp;
                nsetp -= 1;
                // Fortran: IZ1=IZ1-1; INDEX(IZ1)=I  (1-based → 0-based iz1-1)
                iz1 -= 1;
                index[iz1 - 1] = i;
                let mut ok = true;
                for j2 in 1..=nsetp {
                    let i2 = index[j2 - 1];
                    if x[(i2 - 1) as usize] <= 0.0 {
                        jj = j2;
                        ok = false;
                        break;
                    }
                }
                if ok {
                    break 'mv;
                }
            }
            zz = b.clone();
            solve_tri(&a, &index, nsetp, &mut zz);
        }
    }
    x
}

/// `pnnlssum_` from `miscfuns.h`: NNLS with an appended homogeneous
/// sum-to-one row and `sum` scaling, then normalized so the solution sums to
/// exactly `sum`. This is the weight subproblem used by `computeweights`.
pub fn pnnlssum(a: &DMatrix<f64>, b: &DVector<f64>, sum: f64) -> DVector<f64> {
    let n = a.ncols();
    let rows = a.nrows();
    // AA = [ (A*sum).colwise() - b ; 1 1 ... 1 ]  with shape (rows+1) x n
    let mut aa = DMatrix::zeros(rows + 1, n);
    for j in 0..n {
        for i in 0..rows {
            aa[(i, j)] = a[(i, j)] * sum - b[i];
        }
        aa[(rows, j)] = 1.0;
    }
    let mut bb = DVector::zeros(rows + 1);
    bb[rows] = 1.0;
    let x = nnls(aa, bb, n);
    let s = x.sum();
    if s != 0.0 && s.is_finite() {
        x / s * sum
    } else {
        DVector::from_vec(vec![sum / n as f64; n])
    }
}

/// `pnnqp_` from `miscfuns.h`: the large-data (`len > 1000`) weight
/// subproblem, which diagonalizes the positive-semidefinite Gram matrix `q`
/// and reduces to a `pnnlssum` call on the significant eigen-directions.
pub fn pnnqp(q: &DMatrix<f64>, p: &DVector<f64>, sum: f64) -> DVector<f64> {
    use nalgebra::SymmetricEigen;
    let eig = SymmetricEigen::new(q.clone());
    let m = q.nrows();
    // C++ `pnnqp_` uses Eigen's `SelfAdjointEigenSolver`, which returns
    // eigenvalues in ascending order; nalgebra's may be unsorted, so sort the
    // (eval, evec-column) pairs ascending to match.
    let mut pairs: Vec<(f64, usize)> = (0..m)
        .map(|i| (eig.eigenvalues[i], i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // `index = count(eigval > eigval.last()*1e-15)`, then the LAST `index`
    // (largest) eigen-directions are kept.
    let threshold = pairs[m - 1].0 * 1e-15;
    let index = pairs.iter().filter(|(ev, _)| *ev > threshold).count();
    let start = m - index;
    if index == 0 {
        return DVector::from_vec(vec![sum / m as f64; m]);
    }
    let mut a = DMatrix::zeros(index, m);
    for jj in 0..index {
        let (ev, col) = pairs[start + jj];
        let r = ev.sqrt();
        for i in 0..m {
            a[(jj, i)] = eig.eigenvectors[(i, col)] * r;
        }
    }
    let mut b = DVector::zeros(index);
    for jj in 0..index {
        let (ev, col) = pairs[start + jj];
        let mut s = 0.0;
        for i in 0..m {
            s += -eig.eigenvectors[(i, col)] * p[i];
        }
        b[jj] = s / ev.sqrt();
    }
    pnnlssum(&a, &b, sum)
}
