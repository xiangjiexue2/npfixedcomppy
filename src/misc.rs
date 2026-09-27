//! Small numeric helpers ported from the `miscfuns.h` Eigen extensions used by
//! the engine: `diff_`, `index2num` and a few vector reductions.

/// `diff_(x)`: successive differences, length `x.len() - 1`.
pub fn diff(v: &[f64]) -> Vec<f64> {
    if v.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(v.len() - 1);
    for w in v.windows(2) {
        out.push(w[1] - w[0]);
    }
    out
}

/// `index2num(flag)`: 0-based indices where `flag` is non-zero
/// (R's 1-based `which` equivalent).
pub fn index2num(flag: &[bool]) -> Vec<usize> {
    flag.iter()
        .enumerate()
        .filter(|&(_, &f)| f)
        .map(|(i, _)| i)
        .collect()
}

/// Minimum coefficient of a vector (R's `minCoeff`).
pub fn min_coeff(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, |a, b| a.min(b))
}
