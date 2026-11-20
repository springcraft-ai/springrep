//! A closed plane curve's parallel at a distance on one side: the curve moved along its normal,
//! with the loops it makes where it moves past the curve's radius of curvature cut away where it
//! crosses itself (Farouki and Neff 1990). What is left is the set of points exactly that far from
//! the curve on that side, as ranges of the curve's own parameter.

use crate::{
    error::{Error, Result},
    math::Vec2,
};

/// A closed plane curve: its point, first and second derivative at a parameter, over one period
/// from `start`.
pub(crate) struct Closed<'a> {
    pub eval: &'a dyn Fn(f64) -> (Vec2, Vec2, Vec2),
    pub start: f64,
    pub period: f64,
}

/// Samples along the curve where the parallel is searched for crossings.
const SAMPLES: usize = 4096;

impl Closed<'_> {
    fn at(&self, u: f64) -> (Vec2, Vec2, Vec2) {
        (self.eval)(self.start + (u - self.start).rem_euclid(self.period))
    }

    /// The parallel's point at `u` and its derivative, `q′·(1 − d·k)` with `k` the curvature
    /// toward `side` (+1 the left of the curve's direction, −1 its right): it runs backwards
    /// where `d·k > 1`.
    pub(crate) fn parallel(&self, u: f64, side: f64, d: f64) -> (Vec2, Vec2) {
        let (p, dp, _) = self.at(u);
        let n = dp.perp() * (side / dp.norm());
        (p + n * d, dp * (1.0 - d * self.bend(u, side)))
    }

    /// The signed curvature toward `side` at `u`.
    fn bend(&self, u: f64, side: f64) -> f64 {
        let (_, dp, ddp) = self.at(u);
        let n = dp.perp() * (side / dp.norm());
        n.dot(ddp) / dp.dot(dp)
    }

    /// The least distance from `p` to the curve.
    fn distance(&self, p: Vec2) -> f64 {
        let step = self.period / SAMPLES as f64;
        let mut best = (f64::INFINITY, self.start);
        for i in 0..SAMPLES {
            let u = self.start + step * i as f64;
            let dist = self.at(u).0.dist(p);
            if dist < best.0 {
                best = (dist, u);
            }
        }
        // Newton steps on (q − p)·q′ = 0 from the nearest sample, kept within its neighbours.
        let mut u = best.1;
        for _ in 0..30 {
            let (q, dq, ddq) = self.at(u);
            let g = (q - p).dot(dq);
            let dg = dq.dot(dq) + (q - p).dot(ddq);
            if dg <= 0.0 {
                break;
            }
            let next = (u - g / dg).clamp(best.1 - step, best.1 + step);
            if (next - u).abs() <= 1e-15 * self.period {
                u = next;
                break;
            }
            u = next;
        }
        best.0.min(self.at(u).0.dist(p))
    }
}

/// The parallel of `curve` at distance `d` on `side`, trimmed: loops of parameter ranges, each
/// range a piece of the parallel run the way the curve runs, each loop closed where a piece ends
/// at the crossing the next starts from. A loop of one range a whole period long is the whole
/// parallel. No loop is left where the curve is everywhere nearer than `d` to the other side.
pub(crate) fn trimmed(curve: &Closed, side: f64, d: f64) -> Result<Vec<Vec<(f64, f64)>>> {
    let step = curve.period / SAMPLES as f64;
    let us: Vec<f64> = (0..SAMPLES)
        .map(|i| curve.start + step * i as f64)
        .collect();
    let pts: Vec<Vec2> = us.iter().map(|&u| curve.parallel(u, side, d).0).collect();
    let mut cuts: Vec<f64> = Vec::new();
    for (i, j, a, b) in crossings(&pts) {
        let (ua, ub) = refined(curve, side, d, us[i] + step * a, us[j] + step * b)?;
        for u in [ua, ub] {
            let u = curve.start + (u - curve.start).rem_euclid(curve.period);
            if !cuts
                .iter()
                .any(|&c| circular(c - u, curve.period) <= 1e-9 * curve.period)
            {
                cuts.push(u);
            }
        }
    }
    let folds = |u0: f64, u1: f64| {
        (0..=64).any(|k| d * curve.bend(u0 + (u1 - u0) * k as f64 / 64.0, side) >= 1.0)
    };
    if cuts.is_empty() {
        if folds(curve.start, curve.start + curve.period) {
            return Err(Error::Unsupported(
                "a parallel curve that folds without crossing itself".into(),
            ));
        }
        return Ok(vec![vec![(curve.start, curve.start + curve.period)]]);
    }
    cuts.sort_by(f64::total_cmp);
    let mut kept: Vec<(f64, f64)> = Vec::new();
    for k in 0..cuts.len() {
        let u0 = cuts[k];
        let u1 = if k + 1 < cuts.len() {
            cuts[k + 1]
        } else {
            cuts[0] + curve.period
        };
        let mid = curve.parallel(0.5 * (u0 + u1), side, d).0;
        if curve.distance(mid) < d * (1.0 - 1e-6) {
            continue;
        }
        if folds(u0, u1) {
            return Err(Error::Unsupported(
                "a parallel curve whose trimmed piece folds".into(),
            ));
        }
        kept.push((u0, u1));
    }
    let end = |r: (f64, f64)| curve.parallel(r.1, side, d).0;
    let start = |r: (f64, f64)| curve.parallel(r.0, side, d).0;
    let mut loops = Vec::new();
    let mut left = kept;
    while let Some(first) = left.pop() {
        let mut lp = vec![first];
        loop {
            let at = end(*lp.last().expect("a piece"));
            if at.dist(start(lp[0])) <= 1e-6 * (1.0 + d) {
                break;
            }
            let Some(next) = left
                .iter()
                .position(|&r| start(r).dist(at) <= 1e-6 * (1.0 + d))
            else {
                return Err(Error::Unsupported(
                    "a trimmed parallel curve whose pieces do not close".into(),
                ));
            };
            lp.push(left.swap_remove(next));
        }
        loops.push(lp);
    }
    Ok(loops)
}

/// `x` folded into the circle of length `period`: how far it is from a whole number of periods.
fn circular(x: f64, period: f64) -> f64 {
    let r = x.rem_euclid(period);
    r.min(period - r)
}

/// Where the closed polyline through `pts` crosses itself: the two segments' indices and the
/// fraction along each, segments that share an end left out.
fn crossings(pts: &[Vec2]) -> Vec<(usize, usize, f64, f64)> {
    let n = pts.len();
    let seg = |i: usize| (pts[i], pts[(i + 1) % n]);
    let mut order: Vec<usize> = (0..n).collect();
    let lo = |i: usize| seg(i).0.x.min(seg(i).1.x);
    let hi = |i: usize| seg(i).0.x.max(seg(i).1.x);
    order.sort_by(|&a, &b| lo(a).total_cmp(&lo(b)));
    let mut out = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for &i in &order {
        active.retain(|&j| hi(j) >= lo(i));
        let (p, q) = seg(i);
        for &j in &active {
            if (i + 1) % n == j || (j + 1) % n == i {
                continue;
            }
            let (r, s) = seg(j);
            let (dp, dr) = (q - p, s - r);
            let den = dp.cross(dr);
            if den == 0.0 {
                continue;
            }
            let a = (r - p).cross(dr) / den;
            let b = (r - p).cross(dp) / den;
            if (0.0..1.0).contains(&a) && (0.0..1.0).contains(&b) {
                let (i, j, a, b) = if i < j { (i, j, a, b) } else { (j, i, b, a) };
                out.push((i, j, a, b));
            }
        }
        active.push(i);
    }
    out
}

/// The crossing of the parallel with itself near parameters `a` and `b`, by Newton steps on
/// `o(a) − o(b) = 0`.
fn refined(curve: &Closed, side: f64, d: f64, mut a: f64, mut b: f64) -> Result<(f64, f64)> {
    for _ in 0..50 {
        let (pa, da) = curve.parallel(a, side, d);
        let (pb, db) = curve.parallel(b, side, d);
        let f = pa - pb;
        if f.norm() <= 1e-13 * (1.0 + d) {
            return Ok((a, b));
        }
        // da·Δa − db·Δb = −f, by Cramer's rule.
        let det = db.cross(da);
        if det.abs() <= 1e-300 {
            break;
        }
        a += f.cross(db) / det;
        b += f.cross(da) / det;
    }
    let (pa, _) = curve.parallel(a, side, d);
    let (pb, _) = curve.parallel(b, side, d);
    if pa.dist(pb) <= 1e-9 * (1.0 + d) {
        Ok((a, b))
    } else {
        Err(Error::Unsupported(
            "a parallel curve whose crossing with itself is not found".into(),
        ))
    }
}
