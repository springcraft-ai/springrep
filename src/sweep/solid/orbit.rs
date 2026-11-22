//! Whether the faces that curves sweep under a steady motion pass through themselves or each
//! other. A steady motion carries each point along its orbit: a line, a circle or a helix. Two
//! points of the curves on one orbit, less than the motion's span apart along it, are one point of
//! two sheets, or of one sheet twice. An orbit is named by two lengths the motion keeps, so the
//! curves are drawn in the plane of those two, and the sheets cross where the drawings do.

use std::f64::consts::TAU;

use super::sheet::Steady;
use crate::{
    geom::Curve,
    math::{Vec2, Vec3},
};

/// Samples along each curve.
const SAMPLES: usize = 128;

/// A curve and the range of it that is swept.
type Swept = (Curve, (f64, f64));

/// A curve drawn in the plane of the two lengths the motion keeps, and how far along its orbit
/// each point is, in the motion's own parameter.
struct Drawing {
    pts: Vec<Vec2>,
    phase: Vec<f64>,
}

/// What a whole turn about the axis adds to a point's drawing and to its place along the orbit.
fn turn_of(motion: &Steady) -> Option<(Vec2, f64)> {
    match motion {
        Steady::Slide { .. } => None,
        Steady::Turn { .. } => Some((Vec2::ZERO, TAU)),
        Steady::Screw { screw, .. } => Some((Vec2::new(0.0, -screw.rise * TAU), TAU)),
    }
}

/// Where the point `p` is drawn and how far along its orbit it is, an angle about the axis read
/// within half a turn of `near`.
fn drawn(motion: &Steady, p: Vec3, near: f64) -> (Vec2, f64) {
    let (origin, axis, sign, rise) = match motion {
        Steady::Slide { by } => (Vec3::ZERO, by.unit_or_zero(), 0.0, 0.0),
        Steady::Turn { origin, axis, .. } => (*origin, *axis, 1.0, 0.0),
        Steady::Screw { screw, .. } => (screw.origin, screw.axis, screw.sign, screw.rise),
    };
    let x = axis.any_perpendicular().unit_or_zero();
    let y = axis.cross(x);
    let q = p - origin;
    let (a, b, z) = (q.dot(x), q.dot(y), q.dot(axis));
    if sign == 0.0 {
        return (Vec2::new(a, b), z);
    }
    let mut phase = sign * b.atan2(a);
    phase += TAU * ((near - phase) / TAU).round();
    (Vec2::new(a.hypot(b), z - rise * phase), phase)
}

fn draw(motion: &Steady, (curve, range): &Swept) -> Drawing {
    let mut out = Drawing {
        pts: Vec::with_capacity(SAMPLES + 1),
        phase: Vec::with_capacity(SAMPLES + 1),
    };
    for k in 0..=SAMPLES {
        let p = curve.eval(range.0 + (range.1 - range.0) * k as f64 / SAMPLES as f64);
        // An angle is carried on from the last sample's without a jump.
        let (at, phase) = drawn(motion, p, out.phase.last().copied().unwrap_or(0.0));
        out.pts.push(at);
        out.phase.push(phase);
    }
    out
}

/// Where two curves' drawings cross, the second drawn `shift` on, and how far along their orbit
/// the first curve's point is past the second's: Newton's method on the two parameters from
/// `seed`, where their chords cross, the points there `phases` along their orbits. Chords cross
/// where the curves only touch, as two that leave a shared end along one line do; there the steps
/// run to the end, or nowhere.
fn crossing(
    motion: &Steady,
    curves: [&Swept; 2],
    seed: [f64; 2],
    phases: [f64; 2],
    shift: Vec2,
    size: f64,
) -> Option<(Vec2, f64)> {
    let at = |k: usize, t: f64| drawn(motion, curves[k].0.eval(t), phases[k]);
    let inside = |k: usize, t: f64| {
        let (a, b) = curves[k].1;
        t >= a.min(b) && t <= a.max(b)
    };
    let mut t = seed;
    for _ in 0..50 {
        let (a, b) = (at(0, t[0]), at(1, t[1]));
        let gap = a.0 - b.0 - shift;
        if gap.norm() <= 1e-12 * size {
            return (inside(0, t[0]) && inside(1, t[1])).then_some((a.0, a.1 - b.1));
        }
        let rate = |k: usize| {
            let h = 1e-6 * (curves[k].1.1 - curves[k].1.0);
            (at(k, t[k] + h).0 - at(k, t[k] - h).0) / (2.0 * h)
        };
        // gap + da·δ₀ − db·δ₁ = 0.
        let (da, db) = (rate(0), rate(1));
        let det = db.cross(da);
        if det.abs() <= 1e-12 * da.norm() * db.norm() {
            return None;
        }
        t[0] += gap.cross(db) / det;
        t[1] += gap.cross(da) / det;
        if !t[0].is_finite() || !t[1].is_finite() {
            return None;
        }
    }
    None
}

/// Whether two of the faces that `curves` sweep under `motion`, or one of them twice, hold one
/// point away from the curves' ends: the drawings cross, at points less than the motion's span
/// apart along their orbit. `size` is the body's.
pub(super) fn crossed(motion: &Steady, curves: &[Swept], size: f64) -> bool {
    let span = motion.span();
    let drawings: Vec<Drawing> = curves.iter().map(|c| draw(motion, c)).collect();
    let turn = turn_of(motion);
    let turns = match turn {
        Some(_) => (span / TAU).ceil() as i32 + 1,
        None => 0,
    };
    let near = 1e-6 * size;
    let slack = 1e-6 * span;
    let param = |c: usize, m: usize, s: f64| {
        let (t0, t1) = curves[c].1;
        t0 + (t1 - t0) * (m as f64 + s) / SAMPLES as f64
    };
    for (i, a) in drawings.iter().enumerate() {
        for (j, b) in drawings.iter().enumerate().skip(i) {
            for k in -turns..=turns {
                let (shift, lag) = turn.map_or((Vec2::ZERO, 0.0), |(s, p)| {
                    (s * f64::from(k), p * f64::from(k))
                });
                let ends = [
                    a.pts[0],
                    a.pts[SAMPLES],
                    b.pts[0] + shift,
                    b.pts[SAMPLES] + shift,
                ];
                for m in 0..SAMPLES {
                    // A curve's neighbouring stretches meet where they join.
                    let from = if i == j && k == 0 { m + 2 } else { 0 };
                    for n in from..SAMPLES {
                        let (p, q) = (a.pts[m], b.pts[n] + shift);
                        let (dp, dq) = (a.pts[m + 1] - p, b.pts[n + 1] - b.pts[n]);
                        let det = dp.cross(dq);
                        if det.abs() <= 1e-9 * dp.norm() * dq.norm() {
                            continue;
                        }
                        let (s, t) = ((q - p).cross(dq) / det, (q - p).cross(dp) / det);
                        if !(0.0..1.0).contains(&s) || !(0.0..1.0).contains(&t) {
                            continue;
                        }
                        let phases = [
                            a.phase[m] + (a.phase[m + 1] - a.phase[m]) * s,
                            b.phase[n] + (b.phase[n + 1] - b.phase[n]) * t,
                        ];
                        let seed = [param(i, m, s), param(j, n, t)];
                        let pair = [&curves[i], &curves[j]];
                        let Some((at, apart)) = crossing(motion, pair, seed, phases, shift, size)
                        else {
                            continue;
                        };
                        let apart = (apart - lag).abs();
                        if ends.iter().all(|e| e.dist(at) > near)
                            && apart > slack
                            && apart < span - slack
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::geom::Line;

    #[test]
    fn sheets_cross_where_two_points_share_an_orbit_within_the_span() {
        let turn = |angle: f64| Steady::Turn {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            from: 0.0,
            angle,
        };
        let line = |origin: Vec3, dir: Vec3| {
            Curve::Line(Line {
                origin,
                dir: dir.unit().unwrap(),
            })
        };
        // A line in a plane square to the axis is drawn running back along itself, and a leaning
        // one has its points at one distance from the axis at different heights: neither crosses.
        let level = line(Vec3::new(1.0, -2.0, 0.0), Vec3::Y);
        assert!(!crossed(&turn(PI), &[(level, (0.0, 4.0))], 4.0));
        let leaning = line(Vec3::new(1.0, -2.0, -1.0), Vec3::new(0.0, 2.0, 1.0));
        assert!(!crossed(&turn(PI), &[(leaning, (0.0, 4.4))], 4.0));
        // `aslant` passes (1, −0.5, 0.5), a quarter turn back round the axis from `upright` there.
        let aslant = line(Vec3::new(1.0, -2.0, -1.0), Vec3::new(0.0, 1.0, 1.0));
        let upright = line(Vec3::new(0.5, 1.0, 0.5), Vec3::Z);
        let pair = [(aslant, (0.0, 3.0)), (upright, (-1.0, 1.0))];
        assert!(crossed(&turn(2.0), &pair, 4.0));
        assert!(!crossed(&turn(1.5), &pair, 4.0));
    }
}
