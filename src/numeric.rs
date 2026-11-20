//! Quadrature, linear solves and root finding shared by the geometry and measurement modules.

use std::sync::OnceLock;

use crate::{Error, Result};

/// Gauss–Legendre nodes and weights of order `N` on [-1, 1].
fn legendre_rule(n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        out.push((x, 2.0 / ((1.0 - x * x) * dp * dp)));
    }
    out
}

pub(crate) fn gauss(n: usize) -> &'static [(f64, f64)] {
    static RULES: [OnceLock<Vec<(f64, f64)>>; 4] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    let slot = match n {
        8 => 0,
        12 => 1,
        16 => 2,
        _ => 3,
    };
    let order = [8, 12, 16, 24][slot];
    RULES[slot].get_or_init(|| legendre_rule(order))
}

/// The 15-point Kronrod rule on [−1, 1] as `(x, weight, 7-point Gauss weight)` for `x ≥ 0`, each
/// node but the centre standing for `±x`: the Gauss rule is the nodes with a weight of their own
/// (Kronrod, Nodes and Weights of Quadrature Formulas, 1965).
pub(crate) const KRONROD_15: [(f64, f64, f64); 8] = [
    (0.991_455_371_120_812_6, 0.022_935_322_010_529_225, 0.0),
    (
        0.949_107_912_342_758_5,
        0.063_092_092_629_978_55,
        0.129_484_966_168_869_7,
    ),
    (0.864_864_423_359_769_1, 0.104_790_010_322_250_18, 0.0),
    (
        0.741_531_185_599_394_4,
        0.140_653_259_715_525_92,
        0.279_705_391_489_276_7,
    ),
    (0.586_087_235_467_691_1, 0.169_004_726_639_267_9, 0.0),
    (
        0.405_845_151_377_397_2,
        0.190_350_578_064_785_4,
        0.381_830_050_505_118_9,
    ),
    (0.207_784_955_007_898_47, 0.204_432_940_075_298_9, 0.0),
    (0.0, 0.209_482_141_084_727_83, 0.417_959_183_673_469_4),
];

/// The 7-point Kronrod rule on [−1, 1] with the 3-point Gauss rule inside it, as [`KRONROD_15`]
/// (Kronrod, Nodes and Weights of Quadrature Formulas, 1965).
pub(crate) const KRONROD_7: [(f64, f64, f64); 4] = [
    (0.960_491_268_708_020_3, 0.104_656_226_026_467_27, 0.0),
    (
        0.774_596_669_241_483_4,
        0.268_488_089_868_333_44,
        0.555_555_555_555_555_6,
    ),
    (0.434_243_749_346_802_56, 0.401_397_414_775_962_2, 0.0),
    (0.0, 0.450_916_538_658_474_14, 0.888_888_888_888_889),
];

/// ∫ f over [a, b] with an `n`-point Gauss rule.
pub(crate) fn gauss_fixed(f: &mut impl FnMut(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let (m, h) = ((a + b) * 0.5, (b - a) * 0.5);
    gauss(n).iter().map(|&(x, w)| w * f(m + h * x)).sum::<f64>() * h
}

/// ∫ f over [a, b], halving until two levels agree to `abs_tol` (spread over the interval).
pub(crate) fn integrate(f: &mut impl FnMut(f64) -> f64, a: f64, b: f64, abs_tol: f64) -> f64 {
    let whole = gauss_fixed(f, a, b, 16);
    adapt(f, a, b, whole, abs_tol, 0)
}

fn adapt(f: &mut impl FnMut(f64) -> f64, a: f64, b: f64, whole: f64, tol: f64, depth: u32) -> f64 {
    let m = (a + b) * 0.5;
    let left = gauss_fixed(f, a, m, 16);
    let right = gauss_fixed(f, m, b, 16);
    if (left + right - whole).abs() <= tol || depth >= 24 {
        return left + right;
    }
    adapt(f, a, m, left, tol * 0.5, depth + 1) + adapt(f, m, b, right, tol * 0.5, depth + 1)
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting. `a` is row-major `n × n`, and
/// `b` holds `k` right-hand sides as `n` rows of `k`.
pub(crate) fn solve_dense(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    n: usize,
    k: usize,
) -> Result<Vec<f64>> {
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .unwrap_or(col);
        if a[piv * n + col].abs() < 1e-300 {
            return Err(Error::Numerical("singular linear system".into()));
        }
        if piv != col {
            for j in 0..n {
                a.swap(piv * n + j, col * n + j);
            }
            for j in 0..k {
                b.swap(piv * k + j, col * k + j);
            }
        }
        let d = a[col * n + col];
        for row in col + 1..n {
            let f = a[row * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                a[row * n + j] -= f * a[col * n + j];
            }
            for j in 0..k {
                b[row * k + j] -= f * b[col * k + j];
            }
        }
    }
    for col in (0..n).rev() {
        let d = a[col * n + col];
        for j in 0..k {
            let mut s = b[col * k + j];
            for i in col + 1..n {
                s -= a[col * n + i] * b[i * k + j];
            }
            b[col * k + j] = s / d;
        }
    }
    Ok(b)
}

/// One row of a sparse system: its first column and the values of the columns that follow.
pub(crate) type SparseRow = (usize, Vec<f64>);

/// Solves `a x = b` where each row of `a` is a short run of columns, by Gaussian elimination with
/// partial pivoting among the rows that reach each column. Banded rows (B-spline collocation) cost
/// `O(n w²)` time and `O(n w)` memory for bandwidth `w`; `b` is as in [`solve_dense`].
pub(crate) fn solve_banded(rows: Vec<SparseRow>, mut b: Vec<f64>, k: usize) -> Result<Vec<f64>> {
    let n = rows.len();
    let singular = || Error::Numerical("singular linear system".into());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&r| rows[r].0);
    let mut rows: Vec<Option<SparseRow>> = rows.into_iter().map(Some).collect();
    let mut next = 0;
    let mut active: Vec<usize> = Vec::new();
    // The pivot row of each column, with its right-hand sides, for back substitution.
    let mut upper: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(n);
    for col in 0..n {
        while next < n && rows[order[next]].as_ref().is_some_and(|r| r.0 <= col) {
            active.push(order[next]);
            next += 1;
        }
        let piv = active
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| {
                let row = rows[r].as_ref()?;
                (row.0 == col).then(|| (i, row.1.first().copied().unwrap_or(0.0).abs()))
            })
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .filter(|&(_, v)| v >= 1e-300)
            .ok_or_else(singular)?
            .0;
        let pr = active.swap_remove(piv);
        let (_, pv) = rows[pr].take().ok_or_else(singular)?;
        let pb: Vec<f64> = b[pr * k..pr * k + k].to_vec();
        for &r in &active {
            let row = rows[r].as_mut().ok_or_else(singular)?;
            if row.0 != col {
                return Err(singular());
            }
            let f = row.1.first().copied().unwrap_or(0.0) / pv[0];
            let len = row.1.len().max(pv.len());
            let vals: Vec<f64> = (1..len)
                .map(|j| {
                    row.1.get(j).copied().unwrap_or(0.0) - f * pv.get(j).copied().unwrap_or(0.0)
                })
                .collect();
            *row = (col + 1, vals);
            for j in 0..k {
                b[r * k + j] -= f * pb[j];
            }
        }
        upper.push((pv, pb));
    }
    let mut x = vec![0.0; n * k];
    for col in (0..n).rev() {
        let (pv, pb) = &upper[col];
        for j in 0..k {
            let mut s = pb[j];
            for (i, &a) in pv.iter().enumerate().skip(1) {
                s -= a * x[(col + i) * k + j];
            }
            x[col * k + j] = s / pv[0];
        }
    }
    Ok(x)
}

/// A banded matrix factored once by Gaussian elimination with partial pivoting (Golub and Van Loan,
/// *Matrix Computations*, §4.3), so each solve costs `O(n w)` for bandwidth `w`. Rows are as
/// [`solve_banded`] takes them.
#[derive(Clone, Debug)]
pub(crate) struct BandLu {
    n: usize,
    kl: usize,
    ku: usize,
    a: Vec<f64>,
    piv: Vec<usize>,
}

impl BandLu {
    pub(crate) fn factor(rows: &[SparseRow]) -> Result<BandLu> {
        let n = rows.len();
        let singular = || Error::Numerical("singular linear system".into());
        let (mut kl, mut ku) = (0, 0);
        for (r, (first, vals)) in rows.iter().enumerate() {
            if vals.is_empty() {
                continue;
            }
            if first + vals.len() > n {
                return Err(singular());
            }
            kl = kl.max(r.saturating_sub(*first));
            ku = ku.max((first + vals.len() - 1).saturating_sub(r));
        }
        // Pivoting fills each row up to `kl` columns past its own upper band.
        let s = 2 * kl + ku + 1;
        let mut lu = BandLu {
            n,
            kl,
            ku,
            a: vec![0.0; n * s],
            piv: Vec::with_capacity(n),
        };
        for (r, (first, vals)) in rows.iter().enumerate() {
            for (t, &v) in vals.iter().enumerate() {
                let i = lu.at(r, first + t);
                lu.a[i] = v;
            }
        }
        for col in 0..n {
            let last = (col + kl).min(n - 1);
            let hi = (col + kl + ku).min(n - 1);
            let piv = (col..=last)
                .max_by(|&x, &y| {
                    lu.a[lu.at(x, col)]
                        .abs()
                        .total_cmp(&lu.a[lu.at(y, col)].abs())
                })
                .unwrap_or(col);
            if lu.a[lu.at(piv, col)].abs() < 1e-300 {
                return Err(singular());
            }
            lu.piv.push(piv);
            if piv != col {
                for j in col..=hi {
                    let (x, y) = (lu.at(piv, j), lu.at(col, j));
                    lu.a.swap(x, y);
                }
            }
            let d = lu.a[lu.at(col, col)];
            for r in col + 1..=last {
                let rc = lu.at(r, col);
                let f = lu.a[rc] / d;
                lu.a[rc] = f;
                if f != 0.0 {
                    for j in col + 1..=hi {
                        let (x, y) = (lu.at(r, j), lu.at(col, j));
                        lu.a[x] -= f * lu.a[y];
                    }
                }
            }
        }
        Ok(lu)
    }

    fn at(&self, i: usize, j: usize) -> usize {
        i * (2 * self.kl + self.ku + 1) + (j + self.kl - i)
    }

    /// Overwrites `b`, one row of `K` right-hand sides per unknown, with the solution.
    pub(crate) fn solve<const K: usize>(&self, b: &mut [[f64; K]]) {
        let n = self.n;
        for col in 0..n {
            b.swap(col, self.piv[col]);
            let x = b[col];
            let last = (col + self.kl).min(n - 1);
            for (r, row) in b.iter_mut().enumerate().take(last + 1).skip(col + 1) {
                let f = self.a[self.at(r, col)];
                for (v, xv) in row.iter_mut().zip(x) {
                    *v -= f * xv;
                }
            }
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            let last = (i + self.kl + self.ku).min(n - 1);
            for (j, xs) in b.iter().enumerate().take(last + 1).skip(i + 1) {
                let f = self.a[self.at(i, j)];
                for (v, xv) in s.iter_mut().zip(xs) {
                    *v -= f * xv;
                }
            }
            let d = self.a[self.at(i, i)];
            b[i] = s.map(|v| v / d);
        }
    }
}

/// Solves `a x = b` where each row of `a` is a short run of columns that may wrap past the last
/// column to the first, as a periodic spline's collocation does. Each row's largest value should
/// lie on the diagonal: the last `w` rows and columns (`w` the longest run) are solved as a border
/// round the banded rest, so the cost stays linear in `n`.
pub(crate) fn solve_cyclic(rows: Vec<SparseRow>, b: Vec<f64>, k: usize) -> Result<Vec<f64>> {
    let n = rows.len();
    let w = rows.iter().map(|r| r.1.len()).max().unwrap_or(0);
    if n <= 4 * w + 4 {
        let mut a = vec![0.0; n * n];
        for (r, (lo, vals)) in rows.iter().enumerate() {
            for (j, v) in vals.iter().enumerate() {
                a[r * n + (lo + j) % n] += v;
            }
        }
        return solve_dense(a, b, n, k);
    }
    let m = n - w;
    // The inner rows, with each one's border columns as extra right-hand sides.
    let kk = k + w;
    let mut inner = Vec::with_capacity(m);
    let mut rhs = vec![0.0; m * kk];
    let mut lower = vec![0.0; w * m];
    let mut corner = vec![0.0; w * w];
    for (r, (lo, vals)) in rows.into_iter().enumerate() {
        let mut band: Vec<(usize, f64)> = Vec::new();
        for (j, v) in vals.into_iter().enumerate() {
            let c = (lo + j) % n;
            match (r < m, c < m) {
                (true, true) => band.push((c, v)),
                (true, false) => rhs[r * kk + k + c - m] += v,
                (false, true) => lower[(r - m) * m + c] += v,
                (false, false) => corner[(r - m) * w + c - m] += v,
            }
        }
        if r < m {
            let lo = band.iter().map(|e| e.0).min().unwrap_or(r);
            let hi = band.iter().map(|e| e.0).max().unwrap_or(r);
            let mut run = vec![0.0; hi + 1 - lo];
            for (c, v) in band {
                run[c - lo] += v;
            }
            inner.push((lo, run));
            rhs[r * kk..r * kk + k].copy_from_slice(&b[r * k..r * k + k]);
        }
    }
    let y = solve_banded(inner, rhs, kk)?;
    // The border's Schur complement, corner − lower · y_border, against b_border − lower · y_b.
    let mut s = corner;
    let mut sb: Vec<f64> = b[m * k..].to_vec();
    for r in 0..w {
        for c in 0..m {
            let l = lower[r * m + c];
            if l == 0.0 {
                continue;
            }
            for j in 0..w {
                s[r * w + j] -= l * y[c * kk + k + j];
            }
            for j in 0..k {
                sb[r * k + j] -= l * y[c * kk + j];
            }
        }
    }
    let border = solve_dense(s, sb, w, k)?;
    let mut x = vec![0.0; n * k];
    for c in 0..m {
        for j in 0..k {
            let mut v = y[c * kk + j];
            for q in 0..w {
                v -= y[c * kk + k + q] * border[q * k + j];
            }
            x[c * k + j] = v;
        }
    }
    x[m * k..].copy_from_slice(&border);
    Ok(x)
}

/// Solves the 2 × 2 system `[a b; c d] x = r`, or `None` when it is singular.
pub(crate) fn solve2(a: f64, b: f64, c: f64, d: f64, r0: f64, r1: f64) -> Option<(f64, f64)> {
    let det = a * d - b * c;
    let scale = (a.abs() + b.abs()) * (c.abs() + d.abs());
    if det.abs() <= 1e-300 || det.abs() <= 1e-14 * scale {
        return None;
    }
    Some(((r0 * d - b * r1) / det, (a * r1 - c * r0) / det))
}

/// A root of `f` in [a, b] where `f(a)` and `f(b)` differ in sign, by bisection to full precision.
pub(crate) fn bisect(f: &mut impl FnMut(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let mut fa = f(a);
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if m <= a || m >= b {
            break;
        }
        let fm = f(m);
        if fm == 0.0 {
            return m;
        }
        if (fm < 0.0) == (fa < 0.0) {
            a = m;
            fa = fm;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// Every root of `f` on [a, b] found by sampling `n` intervals and bisecting each sign change, plus
/// sampled points where |f| touches zero without changing sign.
pub(crate) fn roots(f: &mut impl FnMut(f64) -> f64, a: f64, b: f64, n: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut prev_t = a;
    let mut prev = f(a);
    if prev == 0.0 {
        out.push(a);
    }
    for i in 1..=n {
        let t = a + (b - a) * i as f64 / n as f64;
        let v = f(t);
        if v == 0.0 {
            out.push(t);
        } else if prev != 0.0 && (v < 0.0) != (prev < 0.0) {
            out.push(bisect(f, prev_t, t));
        }
        prev_t = t;
        prev = v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A banded system that needs row swaps (zero diagonal, larger values below) solves to what
    /// the dense solver finds.
    #[test]
    fn a_band_factored_once_solves_as_the_dense_system_does() {
        let n = 9;
        let rows: Vec<SparseRow> = (0..n)
            .map(|r: usize| {
                let first = r.saturating_sub(1);
                let vals = (first..(r + 3).min(n))
                    .map(|j| {
                        if j == r {
                            0.0
                        } else {
                            1.0 + ((r * 7 + j * 3) % 5) as f64
                        }
                    })
                    .collect();
                (first, vals)
            })
            .collect();
        let mut dense = vec![0.0; n * n];
        for (r, (first, vals)) in rows.iter().enumerate() {
            for (t, &v) in vals.iter().enumerate() {
                dense[r * n + first + t] = v;
            }
        }
        let b: Vec<[f64; 2]> = (0..n).map(|r| [r as f64 - 3.0, (r * r) as f64]).collect();
        let want = solve_dense(dense, b.iter().flatten().copied().collect(), n, 2).unwrap();
        let mut got = b.clone();
        BandLu::factor(&rows).unwrap().solve(&mut got);
        for (r, g) in got.iter().enumerate() {
            for k in 0..2 {
                assert!((g[k] - want[r * 2 + k]).abs() < 1e-9, "row {r}: {g:?}");
            }
        }
    }

    /// The Kronrod rule is exact to degree 22 and the Gauss rule inside it to degree 13:
    /// `∫ xᵏ` over [−1, 1] is `2 / (k + 1)` for even `k`.
    #[test]
    fn the_kronrod_pair_integrates_polynomials_exactly() {
        let sum = |k: i32, gauss: bool| -> f64 {
            KRONROD_15
                .iter()
                .map(|&(x, wk, wg)| {
                    let w = if gauss { wg } else { wk };
                    w * x.powi(k) * if x == 0.0 { 1.0 } else { 2.0 }
                })
                .sum()
        };
        for k in (0..=22).step_by(2) {
            assert!(
                (sum(k, false) - 2.0 / (k + 1) as f64).abs() < 1e-15,
                "Kronrod x^{k}"
            );
        }
        for k in (0..=12).step_by(2) {
            assert!(
                (sum(k, true) - 2.0 / (k + 1) as f64).abs() < 1e-15,
                "Gauss x^{k}"
            );
        }
        assert!((sum(14, true) - 2.0 / 15.0).abs() > 1e-6);
    }

    /// The 7-point Kronrod rule is exact to degree 10 and the Gauss rule inside it to degree 5.
    #[test]
    fn the_short_kronrod_pair_integrates_polynomials_exactly() {
        let sum = |k: i32, gauss: bool| -> f64 {
            KRONROD_7
                .iter()
                .map(|&(x, wk, wg)| {
                    let w = if gauss { wg } else { wk };
                    w * x.powi(k) * if x == 0.0 { 1.0 } else { 2.0 }
                })
                .sum()
        };
        for k in (0..=10).step_by(2) {
            assert!(
                (sum(k, false) - 2.0 / (k + 1) as f64).abs() < 1e-15,
                "Kronrod x^{k}"
            );
        }
        for k in (0..=4).step_by(2) {
            assert!(
                (sum(k, true) - 2.0 / (k + 1) as f64).abs() < 1e-15,
                "Gauss x^{k}"
            );
        }
        assert!((sum(12, false) - 2.0 / 13.0).abs() > 1e-6);
    }

    #[test]
    fn gauss_rules_integrate_polynomials_exactly() {
        for n in [8, 12, 16, 24] {
            let s: f64 = gauss(n).iter().map(|&(_, w)| w).sum();
            assert!((s - 2.0).abs() < 1e-14, "order {n}");
            let mut f = |x: f64| x.powi(2 * n as i32 - 2);
            let v = gauss_fixed(&mut f, 0.0, 1.0, n);
            assert!(
                (v - 1.0 / (2 * n - 1) as f64).abs() < 1e-14,
                "order {n}: {v}"
            );
        }
    }

    #[test]
    fn adaptive_quadrature_handles_a_kink() {
        let v = integrate(&mut |x: f64| (x - 0.3).abs(), 0.0, 1.0, 1e-14);
        assert!((v - (0.045 + 0.245)).abs() < 1e-12);
    }

    #[test]
    fn banded_and_cyclic_solves_match_dense() {
        let n = 40;
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for cyclic in [false, true] {
            let mut rows = Vec::new();
            let mut a = vec![0.0; n * n];
            for r in 0..n {
                let lo = if cyclic {
                    (r + n - 1) % n
                } else {
                    r.saturating_sub(1)
                };
                let len = if cyclic { 3 } else { 3.min(n - lo) };
                let vals: Vec<f64> = (0..len)
                    .map(|j| {
                        if (lo + j) % n == r {
                            2.0 + rnd()
                        } else {
                            rnd() - 0.5
                        }
                    })
                    .collect();
                for (j, v) in vals.iter().enumerate() {
                    a[r * n + (lo + j) % n] += v;
                }
                rows.push((lo, vals));
            }
            let b: Vec<f64> = (0..2 * n).map(|_| rnd()).collect();
            let want = solve_dense(a, b.clone(), n, 2).unwrap();
            let got = if cyclic {
                solve_cyclic(rows, b, 2).unwrap()
            } else {
                solve_banded(rows, b, 2).unwrap()
            };
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-12, "cyclic {cyclic}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn dense_solve_pivots() {
        let x = solve_dense(vec![0.0, 1.0, 1.0, 1.0], vec![2.0, 3.0], 2, 1).unwrap();
        assert_eq!(x, vec![1.0, 2.0]);
    }
}
