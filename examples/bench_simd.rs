//! Hot-path shape benchmark for the nalgebra + rayon strategy.
//!
//! Measures, on the stress shape n = 5000 observations, k = 200 support
//! grid points, the two production patterns of npfixedcomppy:
//!
//!   A. kernel-matrix fill K[i][j] = kernel(data[i], mu[j])
//!        A1 serial row-loop
//!        A2 rayon row-loop           <- production (mat_fill)
//!   B. Gram matrix  G = K^T K   (k x k, 2*n*k*k flops)
//!        B1 nalgebra (LLVM auto-vec, serial)
//!        B3 nalgebra, rayon row-blocks (block Grams summed)
//!   C. mat-vec  s = K^T v (k, 2*n*k flops)
//!        C1 nalgebra
//!        C3 nalgebra, rayon row-blocks
//!   D. end-to-end fill + Gram (normal kernel)
//!        D1 current production (A2 + B1)
//!
//! History: a SIMD GEMM library (faer, Par::Seq, no internal threads) was
//! prototyped here and measured ~12 % SLOWER than B1 at k = 200 — the
//! tall-and-thin shape is streaming-dominated, and faer's dispatch +
//! column-major packing lose to nalgebra's contiguous row-major inner
//! loop. In production the GEMM dimension is the current support-point
//! count (1-10), even further in that regime. faer was therefore removed
//! from the dependency tree; the recorded numbers live in docs/PERF.md.
//!
//! Kernels: normal (exp) and Poisson (log dpois with Lanczos lgamma).
//! Median of 5 runs.
//!
//! Run:  cargo run --release --example bench_simd

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

const N: usize = 5000; // observations
const K: usize = 200;  // support grid points
const REPS: u32 = 5;

// ---------------------------------------------------------------------------
// deterministic pseudo-data
// ---------------------------------------------------------------------------
fn mkdata(kind: u8) -> Vec<f64> {
    let mut v = Vec::with_capacity(N);
    for i in 0..N {
        let x = (i * 7919 % 10007) as f64 / 10007.0 * 6.0 - 2.0;
        let y = (i * 104729 % 9973) as f64 / 9973.0 * 4.0 + 0.2;
        v.push(if kind == 0 { x } else { y.round().max(0.0) });
    }
    v
}

fn grid() -> Vec<f64> {
    let (lo, hi) = (-2.0, 4.0);
    (0..K).map(|j| lo + (hi - lo) * j as f64 / (K - 1) as f64).collect()
}

// ---------------------------------------------------------------------------
// kernels (identical arithmetic in every variant)
// ---------------------------------------------------------------------------
#[inline]
fn knorm(x: f64, mu: f64) -> f64 {
    let z = x - mu;
    0.3989422804014327 * (-0.5 * z * z).exp()
}

const G7: [f64; 9] = [
    0.99999999999980993, 676.5203681218851, -1259.1392230685285, 773.8210538231854,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.993063493509122e-6,
    5.769497221165198e-8,
];
#[inline]
fn gammln(z: f64) -> f64 {
    let zz = z - 1.0;
    let mut x = G7[0];
    for i in 1..9 {
        x += G7[i] / (zz + i as f64);
    }
    let t = zz + 7.5;
    0.9189385332046727 + (zz + 0.5) * t.ln() - t + x.ln()
}
#[inline]
fn kpois(x: f64, mu: f64) -> f64 {
    if mu <= 0.0 {
        return if x == 0.0 { 1.0 } else { 0.0 };
    }
    (x * mu.ln() - mu - gammln(x + 1.0)).exp()
}

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------
fn timeit<F: FnMut()>(mut f: F) -> f64 {
    let mut ts = Vec::with_capacity(REPS as usize);
    for _ in 0..REPS {
        let t0 = std::time::Instant::now();
        f();
        ts.push(t0.elapsed().as_secs_f64());
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[REPS as usize / 2]
}

// ---------------------------------------------------------------------------
// fill (the production mat_fill pattern; serial + rayon)
// ---------------------------------------------------------------------------
fn fill_naive(out: &mut [f64], d: &[f64], mu: &[f64], k: impl Fn(f64, f64) -> f64) {
    for (i, xi) in d.iter().enumerate() {
        let row = &mut out[i * K..(i + 1) * K];
        for (j, mj) in mu.iter().enumerate() {
            row[j] = k(*xi, *mj);
        }
    }
}

fn fill_naive_par(out: &mut [f64], d: &[f64], mu: &[f64], k: impl Fn(f64, f64) -> f64 + Sync) {
    out.par_chunks_mut(K).enumerate().for_each(|(i, row)| {
        let xi = d[i];
        for (j, mj) in mu.iter().enumerate() {
            row[j] = k(xi, *mj);
        }
    });
}

fn main() {
    let data = mkdata(0);
    let pois = mkdata(1);
    let mu = grid();

    let mut out = vec![0.0f64; N * K];
    let vvec: Vec<f64> = (0..N).map(|i| (i as f64).sin() * 0.5 + 1.0).collect();
    let w = DVector::from_vec(vvec.clone());

    // pre-built K (untimed) as the common GEMM input
    fill_naive(&mut out, &data, &mu, knorm);
    let k_na = DMatrix::<f64>::from_vec(N, K, out.clone());

    println!(
        "n={N}  k={K}  threads={}  (median of {REPS})",
        rayon::current_num_threads()
    );

    println!("\n== A. kernel fill ({N} x {K} = {} elts) ==", N * K);
    for (name, d, k) in [
        ("normal ", &data, knorm as fn(f64, f64) -> f64),
        ("poisson", &pois, kpois as fn(f64, f64) -> f64),
    ] {
        let t = timeit(|| fill_naive(&mut out, d, &mu, k));
        println!("  A1 serial row-loop   [{name}] {:>8.2} ms", t * 1e3);
        let t = timeit(|| fill_naive_par(&mut out, d, &mu, k));
        println!("  A2 rayon row-loop    [{name}] {:>8.2} ms   <- production (mat_fill)", t * 1e3);
    }

    let gf_gram = 2.0 * N as f64 * K as f64 * K as f64 / 1e9;
    println!("\n== B. Gram G = K^T K  ({K}x{N} x {N}x{K} = {gf_gram:.2} GFlop) ==");
    let t = timeit(|| {
        let _ = &k_na.transpose() * &k_na;
    });
    println!("  B1 nalgebra (auto-vec, serial) {:>8.2} ms  ({:.1} GFlops)  <- production", t * 1e3, gf_gram / t);
    // B3: rayon over row blocks; each block runs the (LLVM auto-vectorized)
    // nalgebra kernel on its sub-matrix, block Grams are summed.
    let t = timeit(|| {
        let threads = rayon::current_num_threads();
        let ch = (N / (threads * 4)).max(128);
        let _ = (0..N)
            .step_by(ch)
            .collect::<Vec<usize>>()
            .par_iter()
            .map(|&start| {
                let end = (start + ch).min(N);
                let kb = k_na.view((start, 0), (end - start, K));
                kb.transpose() * &kb
            })
            .reduce(
                || DMatrix::<f64>::zeros(K, K),
                |mut a, b| {
                    a += &b;
                    a
                },
            );
    });
    println!("  B3 nalgebra, rayon row-blocks  {:>8.2} ms  ({:.1} GFlops)  <- SIMD (auto-vec) + outer rayon", t * 1e3, gf_gram / t);

    let gf_mv = 2.0 * N as f64 * K as f64 / 1e9;
    println!("\n== C. mat-vec s = K^T v  ({gf_mv:.2} GFlop) ==");
    let t = timeit(|| {
        let _ = &k_na.transpose() * &w;
    });
    println!("  C1 nalgebra (auto-vec, serial) {:>8.2} ms  <- production", t * 1e3);
    // C3: rayon over row blocks, same idea as B3
    let t = timeit(|| {
        let threads = rayon::current_num_threads();
        let ch = (N / (threads * 4)).max(128);
        let _ = (0..N)
            .step_by(ch)
            .collect::<Vec<usize>>()
            .par_iter()
            .map(|&start| {
                let end = (start + ch).min(N);
                let kb = k_na.view((start, 0), (end - start, K));
                let wb = DVector::<f64>::from_vec(vvec[start..end].to_vec());
                kb.transpose() * &wb
            })
            .reduce(
                || DVector::<f64>::zeros(K),
                |mut a, b| {
                    a += &b;
                    a
                },
            );
    });
    println!("  C3 nalgebra, rayon row-blocks  {:>8.2} ms  <- SIMD (auto-vec) + outer rayon", t * 1e3);

    println!("\n== D. end-to-end fill + Gram (normal kernel) ==");
    let t = timeit(|| {
        fill_naive_par(&mut out, &data, &mu, knorm);
        let tp = DMatrix::<f64>::from_vec(N, K, out.clone());
        let _ = &tp.transpose() * &tp;
    });
    println!("  D1 current   (rayon fill + nalgebra gram)  {:>8.2} ms", t * 1e3);

    println!("\nNOTE: faer (SIMD GEMM, Par::Seq) was prototyped here and measured");
    println!("      ~12 % slower than B1 at this shape (docs/PERF.md, section 3)");
    println!("      and was removed from the dependency tree.");
}
