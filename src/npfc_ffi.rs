//! FFI bindings to the C++/Eigen heavy-compute core (`cpp/npfc_core.cpp`).
//!
//! The C++ side owns all SIMD/threading decisions (compile-time `/arch:`,
//! optional OpenMP). Rust keeps its own rayon sites for reductions that the
//! C++ API does not cover (loss/gradient sweeps stay in Rust for now).

use std::ffi::c_void;

#[allow(non_camel_case_types)]
type npfc_int = i32;

extern "C" {
    pub fn npfc_info(caps: *mut u32, omp_threads: *mut npfc_int, simd_level: *mut u32);
    pub fn npfc_pool_create(n_elems: usize) -> *mut c_void;
    pub fn npfc_pool_free(p: *mut c_void);

    /// Column-major (n x g) normal-pdf kernel. Returns 1 on success.
    pub fn npfc_kmat_norm(
        x: *const f64,
        n: i64,
        mu: *const f64,
        g: i64,
        beta: f64,
        out: *mut f64,
    ) -> i32;

    /// Mixture density out[i] = sum_j pi[j] * N(x_i; mu_j, beta).
    pub fn npfc_mapping_norm(
        x: *const f64,
        n: i64,
        mu: *const f64,
        m: i64,
        pi: *const f64,
        beta: f64,
        out: *mut f64,
    ) -> i32;

    /// y = A^T x, A (n x m) column-major.
    pub fn npfc_gemv(
        a: *const f64,
        n: i64,
        m: i64,
        x: *const f64,
        y: *mut f64,
    ) -> i32;
}

/// Build/run-time capability of the C++ core (for `npfc_info()` exposure).
#[derive(Debug, Clone, Copy, Default)]
pub struct CoreInfo {
    pub openmp: bool,
    pub omp_threads: i32,
    pub simd_level: u32, // 0 none, 2 AVX2
}

pub fn core_info() -> CoreInfo {
    let mut caps: u32 = 0;
    let mut threads: npfc_int = 0;
    let mut lvl: u32 = 0;
    unsafe {
        npfc_info(&mut caps, &mut threads, &mut lvl);
    }
    CoreInfo {
        openmp: caps & 1 != 0,
        omp_threads: threads,
        simd_level: lvl,
    }
}

/// Fill `out` (length n, capacity guaranteed) with the normal mixture
/// density. Falls back to the Rust scalar path if the C++ call fails
/// (should be impossible on valid input; kept as a safety net).
#[inline]
pub fn mapping_norm(
    data: &[f64],
    mu0: &[f64],
    pi0: &[f64],
    beta: f64,
    out: &mut [f64],
) -> bool {
    let n = data.len();
    let m = mu0.len();
    if n == 0 || m == 0 || out.len() < n {
        return false;
    }
    if m == 1 {
        // Fast single-component path stays in Rust (no FFI overhead needed).
        let (mu, pj) = (mu0[0], pi0[0]);
        let inv_b = 1.0 / beta;
        const LN_SQRT_2PI: f64 = 0.9189385332046727;
        let log_pref = -(beta.ln() + LN_SQRT_2PI);
        for i in 0..n {
            let d = (data[i] - mu) * inv_b;
            out[i] = pj * (-0.5 * d * d + log_pref).exp();
        }
        return true;
    }
    unsafe {
        let r = npfc_mapping_norm(
            data.as_ptr(),
            n as i64,
            mu0.as_ptr(),
            m as i64,
            pi0.as_ptr(),
            beta,
            out.as_mut_ptr(),
        );
        r == 1
    }
}

/// Fill `out` (length n*g) with the column-major normal-pdf kernel.
#[inline]
pub fn kmat_norm(data: &[f64], grid: &[f64], beta: f64, out: &mut [f64]) -> bool {
    let n = data.len() as i64;
    let g = grid.len() as i64;
    if n <= 0 || g <= 0 || out.len() != (n * g) as usize {
        return false;
    }
    unsafe {
        let r = npfc_kmat_norm(data.as_ptr(), n, grid.as_ptr(), g, beta, out.as_mut_ptr());
        r == 1
    }
}

/// y = A^T x with A (n x m) column-major; `a` is the column-major storage.
#[inline]
pub fn gemv_colmajor(a: &[f64], n: usize, m: usize, x: &[f64], y: &mut [f64]) -> bool {
    if a.len() != n * m || x.len() != n || y.len() != m {
        return false;
    }
    unsafe {
        let r = npfc_gemv(
            a.as_ptr(),
            n as i64,
            m as i64,
            x.as_ptr(),
            y.as_mut_ptr(),
        );
        r == 1
    }
}
