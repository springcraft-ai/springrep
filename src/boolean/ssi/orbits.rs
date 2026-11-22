//! Where a spline surface swept by a steady motion meets another surface the same motion leaves
//! on itself: along orbits of the motion, lines under a slide, circles under a turn and helices
//! under a screw.
//!
//! A swept surface `A` is the points `M(u)·g(v)`, `g` its curve and `M(u)` the motion after `u`,
//! over a rectangle of `(u, v)`. Let `B` be `M(u′)·h(v′)` over its own rectangle, with the same
//! `M`. A common point `M(u)·g(v) = M(u′)·h(v′)` stays common as `u` and `u′` fall together, the
//! motions of one family undoing each other, until one of them reaches the start of its range. So
//! every piece of the intersection is an orbit that starts either on `A`'s first curve, where it
//! meets `B`, or on `B`'s first curve, where it meets `A`, and runs for as long as both
//! parameters stay in range. Those meetings are curve–surface intersections, and each orbit has a
//! closed form. Against a surface the motion carries onto itself whole, a cylinder about a screw's
//! axis or a surface of revolution about a turn's, only `A`'s first curve starts orbits, and each
//! runs the length of `A`.
//!
//! A point of a first curve may lie on the other surface at more than one pair of its parameters,
//! where that surface passes through itself, and each pair bounds its own run of the orbit. The
//! pairs are found on the other surface's curve: its points at the orbit's distance from the axis
//! (or, under a slide, in a plane along the slide through the point) that the motion carries to
//! the point.
//!
//! An orbit that starts where a first curve touches the other surface is one the surfaces touch
//! along, and is marked so.
//!
//! The pair is left to the trace wherever this is not certain to be the whole intersection: where
//! a first curve lies in the other surface, or touches it beside another meeting, where an orbit passes the axis or the
//! other surface's curve keeps one distance from the axis as it crosses the orbit, where a turn
//! comes round to its start, and where a point's parameters on the other surface are not among
//! those found on its curve.

use std::f64::consts::TAU;

use super::{Axis, Ssi, SsiCurve, parallel, rotation_invariant, tangent_at, translation_invariant};
use crate::{
    boolean::csi,
    geom::{Circle, Curve, Cylinder, Line, NurbsSurface, Plane, Steady, Surface, Swept},
    math::{Aabb, Frame, Point3, Vec3},
    sweep::{Helix, SCREW_FIT},
    tol,
};

/// A swept surface: what it keeps of its making, and the rectangle of parameters it covers.
struct Sheet<'a> {
    surface: &'a Surface,
    spline: &'a NurbsSurface,
    swept: &'a Swept,
    u: (f64, f64),
    v: (f64, f64),
}

impl<'a> Sheet<'a> {
    fn of(surface: &'a Surface) -> Option<Sheet<'a>> {
        let Surface::Nurbs(spline) = surface else {
            return None;
        };
        let swept = spline.swept()?;
        let (u, v) = spline.domain();
        let m = &swept.motion;
        // A turn that comes round to its start covers its orbits more than once.
        let round = m.turn != 0.0 && m.rise == 0.0 && (u.1 - u.0) * m.turn.abs() >= TAU - 1e-9;
        let still = m.turn == 0.0 && m.rise == 0.0;
        (!round && !still && spline.u_period().is_none()).then_some(Sheet {
            surface,
            spline,
            swept,
            u,
            v,
        })
    }

    /// The surface's curve where its `u` starts: a line where the curve is one, which meets
    /// surfaces in closed form.
    fn first(&self) -> (Curve, (f64, f64)) {
        let pose = self.swept.motion.pose(self.u.0);
        let g = self.swept.generator.transformed(&pose);
        let (a, b) = (g.eval(self.v.0), g.eval(self.v.1));
        if g.degree() == 1
            && g.control_points().len() == 2
            && g.weights().is_none()
            && let Some(dir) = (b - a).unit()
        {
            return (Curve::Line(Line { origin: a, dir }), (0.0, a.dist(b)));
        }
        (Curve::Nurbs(Box::new(g)), self.v)
    }
}

/// How far `p` is along the motion's axis, and its offset from the axis.
fn split(m: &Steady, p: Point3) -> (f64, Vec3) {
    let q = p - m.origin;
    let z = q.dot(m.axis);
    (z, q - m.axis * z)
}

/// Whether two motions carry every point alike over `span` of their parameter, within `reach` of
/// the first one's origin, to within the fit of a swept surface.
fn same_motion(a: &Steady, b: &Steady, span: f64, reach: f64) -> bool {
    let flip = if a.axis.dot(b.axis) < 0.0 { -1.0 } else { 1.0 };
    let tilt = a.axis.cross(b.axis).norm();
    let apart = if a.turn == 0.0 {
        0.0
    } else {
        split(a, b.origin).1.norm()
    };
    a.turn == b.turn * flip
        && tilt * (reach + a.rise.abs() * span) + apart + (a.rise - b.rise * flip).abs() * span
            <= SCREW_FIT
}

/// Whether the motion carries `s` onto itself.
fn invariant(s: &Surface, m: &Steady) -> bool {
    let axis = Axis {
        origin: m.origin,
        dir: m.axis,
    };
    if m.turn == 0.0 {
        translation_invariant(s, m.axis)
    } else if m.rise == 0.0 {
        rotation_invariant(s, axis)
    } else {
        matches!(s, Surface::Cylinder(c) if parallel(c.frame.z, m.axis)
            && split(m, c.frame.origin).1.norm() <= tol::LINEAR)
    }
}

/// How much of the motion carries `from` to `to`, two points of one orbit; for a turn, the least
/// that is not negative.
fn between(m: &Steady, from: Point3, to: Point3) -> f64 {
    let ((z0, r0), (z1, r1)) = (split(m, from), split(m, to));
    if m.rise != 0.0 {
        return (z1 - z0) / m.rise;
    }
    let angle = r0.cross(r1).dot(m.axis).atan2(r0.dot(r1)) / m.turn;
    angle.rem_euclid(TAU / m.turn.abs())
}

/// Every `u` at which the sheet passes through `p`, none where it does not: a curve met on a
/// spline surface is met on the surface run on past its edges, so `p` may lie off it. `None`
/// where that cannot be told for certain.
fn feet(sheet: &Sheet, p: Point3) -> Option<Vec<f64>> {
    let m = &sheet.swept.motion;
    let g = Curve::Nurbs(Box::new(sheet.swept.generator.clone()));
    let (_, out) = split(m, p);
    let level = if m.turn == 0.0 {
        let normal = m.axis.any_perpendicular();
        Surface::Plane(Plane {
            frame: Frame::from_axis(p, normal).ok()?,
        })
    } else {
        if out.norm() <= tol::LINEAR {
            return None;
        }
        Surface::Cylinder(Cylinder {
            frame: Frame::from_axis(m.origin, m.axis).ok()?,
            radius: out.norm(),
        })
    };
    let hits = csi::curve_surface(&g, sheet.v, &level);
    if hits.on_surface {
        return None;
    }
    let slack = tol::LINEAR / (m.turn.abs() * out.norm()).hypot(m.rise);
    let mut us = Vec::new();
    // An end of the curve on the level is a meeting that rounding may leave out of the range.
    let ends = [sheet.v.0, sheet.v.1]
        .into_iter()
        .filter(|&t| csi::signed_distance(&level, g.eval(t)).abs() <= tol::LINEAR);
    for t in hits.ts.into_iter().chain(ends) {
        let (q, along) = g.d1(t);
        // Where the curve runs along the level it is on, its meeting with the level is not sharp.
        let n = match &level {
            Surface::Plane(pl) => pl.frame.z,
            _ => split(m, q).1.unit()?,
        };
        if along.unit()?.dot(n).abs() <= tol::TANGENT {
            return None;
        }
        let mut u = between(m, q, p);
        if m.rise == 0.0 && u > sheet.u.1 + slack {
            u -= TAU / m.turn.abs();
        }
        if m.pose(u).point(q).dist(p) <= tol::LINEAR
            && u >= sheet.u.0 - slack
            && u <= sheet.u.1 + slack
        {
            us.push(u.clamp(sheet.u.0, sheet.u.1));
        }
    }
    // Where the surface's own nearest point is `p`, its parameters must be among those found.
    let uv = sheet.spline.project(p);
    let at = sheet.spline.eval(uv.x, uv.y);
    let missed = at.dist(p) <= tol::LINEAR
        && !us
            .iter()
            .any(|&u| m.pose(u - uv.x).point(at).dist(p) <= 2.0 * tol::LINEAR);
    (!missed).then_some(us)
}

/// A run of an orbit: from `start` through `length` of the motion, and whether the surfaces
/// touch along it.
struct Run {
    start: Point3,
    length: f64,
    tangent: bool,
}

/// `run` added to `runs`, joined with any run of the same orbit it overlaps or abuts.
fn add(m: &Steady, runs: &mut Vec<Run>, mut run: Run) {
    let speed = |p: Point3| (m.turn * split(m, p).1.norm()).hypot(m.rise);
    let mut k = 0;
    while k < runs.len() {
        let other = &runs[k];
        let ((z0, r0), (z1, r1)) = (split(m, other.start), split(m, run.start));
        // Along the other's orbit, where this one starts: before the other's start or after it.
        let ahead = between(m, other.start, run.start);
        let offset = if m.rise != 0.0 || ahead <= 0.5 * TAU / m.turn.abs() {
            ahead
        } else {
            ahead - TAU / m.turn.abs()
        };
        let same = (r0.norm() - r1.norm()).abs() <= tol::LINEAR
            && (m.turn != 0.0 || (r0 - r1).norm() <= tol::LINEAR)
            && (m.rise != 0.0 || (z0 - z1).abs() <= tol::LINEAR)
            && m.pose(offset).point(other.start).dist(run.start) <= tol::LINEAR;
        let slack = tol::LINEAR / speed(run.start);
        if same && offset <= other.length + slack && offset + run.length >= -slack {
            let end = (offset + run.length).max(other.length);
            let from = offset.min(0.0);
            run = Run {
                start: if offset < 0.0 { run.start } else { other.start },
                length: end - from,
                tangent: run.tangent || other.tangent,
            };
            runs.swap_remove(k);
            k = 0;
        } else {
            k += 1;
        }
    }
    runs.push(run);
}

/// The orbit of `run` as a curve: a line, an arc of a circle, or a helix within the fit of a
/// screw surface. `None` for a run shorter than the tolerance.
fn curve(m: &Steady, run: &Run) -> Option<Result<SsiCurve, ()>> {
    let tangent = run.tangent;
    let (z, out) = split(m, run.start);
    let radius = out.norm();
    if (m.turn * radius).hypot(m.rise) * run.length <= tol::LINEAR {
        return None;
    }
    let (curve, range) = if m.turn == 0.0 {
        let line = Line {
            origin: run.start,
            dir: m.axis * m.rise.signum(),
        };
        (Curve::Line(line), (0.0, m.rise.abs() * run.length))
    } else if m.rise == 0.0 {
        let centre = m.origin + m.axis * z;
        let Ok(frame) = Frame::new(centre, m.axis * m.turn.signum(), out) else {
            return Some(Err(()));
        };
        if m.turn.abs() * run.length >= TAU {
            return Some(Err(()));
        }
        let circle = Circle { frame, radius };
        (Curve::Circle(circle), (0.0, m.turn.abs() * run.length))
    } else {
        let screw = Helix {
            origin: m.origin,
            axis: m.axis,
            rise: m.rise / m.turn.abs(),
            sign: m.turn.signum(),
        };
        let angle = m.turn.abs() * run.length;
        match screw.spiral(run.start, angle, 0.0) {
            Ok(c) => (c, (0.0, angle)),
            Err(_) => return Some(Err(())),
        }
    };
    Some(Ok(SsiCurve {
        curve,
        range,
        closed: false,
        tangent,
    }))
}

/// How near a touch another meeting of the same curve is taken to be the same touch, found twice:
/// two crossings a tolerance deep into a surface that curves at the resolved radius lie
/// `2·√(2·LINEAR·RESOLVED)` apart.
const NEAR_TOUCH: f64 = 3e-4;

/// The runs of orbits that start on `a`'s first curve where it meets `b`: each as long as `a`,
/// or, where `b` is a sheet of the same motion, as long as both.
fn starts(a: &Sheet, b: &Surface, other: Option<&Sheet>, runs: &mut Vec<Run>) -> Option<()> {
    let m = &a.swept.motion;
    let (first, range) = a.first();
    let hits = csi::curve_surface(&first, range, b);
    if hits.on_surface {
        return None;
    }
    let whole = a.u.1 - a.u.0;
    let mut met: Vec<(Point3, bool)> = Vec::new();
    for t in hits.ts {
        let p = first.eval(t);
        let lengths = match other {
            Some(other) => feet(other, p)?
                .into_iter()
                .map(|u| whole.min(other.u.1 - u))
                .collect(),
            None => vec![whole],
        };
        if lengths.is_empty() {
            continue;
        }
        let tangent = tangent_at(a.surface, b, p);
        // A touch found as two meetings a hair apart is one the meetings do not settle.
        if met
            .iter()
            .any(|&(q, touch)| (touch || tangent) && p.dist(q) <= NEAR_TOUCH)
        {
            return None;
        }
        met.push((p, tangent));
        for length in lengths {
            let start = p;
            add(
                m,
                runs,
                Run {
                    start,
                    length,
                    tangent,
                },
            );
        }
    }
    Some(())
}

/// Held by a test for as long as it needs [`TRACE_ONLY`] to stay as it is.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Set by a test that holds [`TEST_LOCK`] to leave every pair to the trace, for a build to compare
/// against.
#[cfg(test)]
pub(crate) static TRACE_ONLY: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn overlaps(a: &Aabb, b: &Aabb) -> bool {
    (0..3).all(|k| a.min[k] <= b.max[k] && b.min[k] <= a.max[k])
}

/// Where the swept sheet `a` meets `b`, a sheet of the same motion or a surface the motion
/// carries onto itself, within `region`; `None` where the pair is neither, or is left to the
/// trace.
pub(super) fn intersect(a: &Surface, b: &Surface, region: &Aabb) -> Option<Ssi> {
    #[cfg(test)]
    if TRACE_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let sheet = Sheet::of(a)?;
    let m = &sheet.swept.motion;
    let mut runs = Vec::new();
    if let Some(other) = Sheet::of(b) {
        let span = (sheet.u.1 - sheet.u.0).max(other.u.1 - other.u.0);
        let reach = (0..8)
            .map(|k| {
                let pick = |i: usize, lo: f64, hi: f64| if k >> i & 1 == 0 { lo } else { hi };
                Point3::new(
                    pick(0, region.min.x, region.max.x),
                    pick(1, region.min.y, region.max.y),
                    pick(2, region.min.z, region.max.z),
                )
                .dist(m.origin)
            })
            .fold(0.0, f64::max);
        if !same_motion(m, &other.swept.motion, span, reach) {
            return None;
        }
        starts(&sheet, b, Some(&other), &mut runs)?;
        starts(&other, a, Some(&sheet), &mut runs)?;
    } else if invariant(b, m) {
        starts(&sheet, b, None, &mut runs)?;
    } else {
        return None;
    }
    let mut curves = Vec::new();
    for run in &runs {
        match curve(m, run) {
            Some(Ok(c)) => {
                if overlaps(&c.curve.bbox(c.range.0, c.range.1), region) {
                    curves.push(c);
                }
            }
            Some(Err(())) => return None,
            None => {}
        }
    }
    Some(Ssi::Curves {
        curves,
        points: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boolean::{march, ssi},
        geom::Cone,
        loft::rail::Rail,
        math::Transform,
        sweep::screw_sheet,
    };

    fn hold() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn screw(rise: f64) -> Helix {
        Helix {
            origin: Point3::new(0.0, 0.0, 0.0),
            axis: Vec3::Z,
            rise,
            sign: 1.0,
        }
    }

    fn pt(p: [f64; 3]) -> Point3 {
        Point3::new(p[0], p[1], p[2])
    }

    /// What the segment from `a` to `b` sweeps through `angle` of `screw`.
    fn swept(a: Point3, b: Point3, screw: &Helix, angle: f64) -> Surface {
        let line = Curve::Line(Line {
            origin: a,
            dir: (b - a).unit().unwrap(),
        });
        let rail = Rail::of_curve(&line, 0.0, a.dist(b)).unwrap();
        Surface::Nurbs(Box::new(screw_sheet(&rail, screw, angle).unwrap()))
    }

    /// Two hyperboloids turned through 1.5 about z: they share the circles of radius √(4 + 7/12)
    /// at heights 1 ± √(7/12)/3.
    fn turned() -> (Surface, Surface) {
        let still = screw(0.0);
        (
            swept(pt([2.0, -3.0, 0.0]), pt([2.0, 3.0, 2.0]), &still, 1.5),
            swept(pt([1.5, -3.0, 0.5]), pt([1.5, 3.0, 1.5]), &still, 1.5),
        )
    }

    /// Two sheets of the screw rising ½ a radian, each through 2, that cross once.
    fn screwed(s: &Helix) -> (Surface, Surface) {
        (
            swept(pt([1.0, 0.0, 0.0]), pt([4.0, 0.0, 1.0]), s, 2.0),
            swept(pt([1.0, 0.2, 1.2]), pt([4.0, -0.3, -0.4]), s, 2.0),
        )
    }

    fn region() -> Aabb {
        let mut r = Aabb::EMPTY;
        r.add(Point3::new(-20.0, -20.0, -20.0));
        r.add(Point3::new(20.0, 20.0, 20.0));
        r
    }

    fn curves(a: &Surface, b: &Surface) -> Vec<SsiCurve> {
        match intersect(a, b, &region()) {
            Some(Ssi::Curves { curves, points }) => {
                assert!(points.is_empty());
                curves
            }
            other => panic!("no closed form: {other:?}"),
        }
    }

    fn samples(c: &SsiCurve) -> impl Iterator<Item = Point3> + '_ {
        (0..=16).map(|k| {
            c.curve
                .eval(c.range.0 + (c.range.1 - c.range.0) * k as f64 / 16.0)
        })
    }

    fn off(s: &Surface, p: Point3) -> f64 {
        let uv = s.project(p);
        s.eval(uv.x, uv.y).dist(p)
    }

    /// Fails unless every curve lies on both surfaces within `fit`, and the trace of the pair from
    /// a point of each curve finds the same branches: as many, each along one of the curves from
    /// end to end and marked as it is.
    fn assert_traced_alike(a: &Surface, b: &Surface, cs: &[SsiCurve], fit: f64) {
        for c in cs {
            for p in samples(c) {
                assert!(off(a, p) <= fit && off(b, p) <= fit, "{p:?} off a surface");
            }
        }
        let seeds: Vec<Point3> = cs
            .iter()
            .map(|c| c.curve.eval(0.5 * (c.range.0 + c.range.1)))
            .collect();
        let (traced, _) = march::trace(a, b, &region(), &seeds).unwrap();
        assert_eq!(traced.len(), cs.len());
        let near = |c: &SsiCurve, p: Point3| {
            let t = c.curve.project(p, c.range.0, c.range.1);
            c.curve.eval(t).dist(p)
        };
        for t in &traced {
            let c = cs
                .iter()
                .find(|c| samples(t).all(|p| near(c, p) <= 1e-6))
                .expect("a traced branch along no orbit");
            for end in [c.range.0, c.range.1] {
                assert!(near(t, c.curve.eval(end)) <= 1e-4, "the trace stops short");
            }
            assert_eq!(t.tangent, c.tangent);
        }
    }

    // Each circle runs while both sheets cover it: the second sheet's line meets a circle
    // atan(6t − 3, 2) − atan(12t − 6, 1.5) round from the first's, t = ½ ± √(7/12)/6, and both
    // turn through 1.5, so each arc is 1.5 less that angle.
    #[test]
    fn sheets_of_one_turn_meet_in_arcs_both_cover() {
        let _held = hold();
        let (a, b) = turned();
        let cs = curves(&a, &b);
        assert_eq!(cs.len(), 2);
        let root = (7.0f64 / 12.0).sqrt();
        for c in &cs {
            let Curve::Circle(circle) = &c.curve else {
                panic!("an orbit of a turn is a circle");
            };
            // A line meets the other sheet where the sheet's fit puts it.
            let r = circle.radius;
            assert!((r - (4.0 + 7.0 / 12.0f64).sqrt()).abs() <= 1e-9, "{r}");
            let z = circle.frame.origin.z;
            let side = (z - 1.0).signum();
            assert!((z - (1.0 + side * root / 3.0)).abs() <= 1e-9, "{z}");
            let between = (side * root).atan2(2.0) - (side * 2.0 * root).atan2(1.5);
            let arc = c.range.1 - c.range.0;
            assert!((arc - (1.5 - between.abs())).abs() <= 1e-9, "{arc}");
            assert!(!c.closed && !c.tangent);
        }
        assert_traced_alike(&a, &b, &cs, 4.0 * SCREW_FIT);
        assert_eq!(curves(&b, &a).len(), 2);
    }

    #[test]
    fn sheets_of_one_screw_meet_in_helices() {
        let _held = hold();
        let (a, b) = screwed(&screw(0.5));
        let cs = curves(&a, &b);
        assert_eq!(cs.len(), 1);
        // A helix keeps its distance from the axis and rises with its turn.
        let c = &cs[0];
        let (p0, p1) = (c.curve.eval(c.range.0), c.curve.eval(c.range.1));
        assert!((p0.x.hypot(p0.y) - p1.x.hypot(p1.y)).abs() <= 1e-10);
        assert!((p1.z - p0.z - 0.5 * (c.range.1 - c.range.0)).abs() <= 1e-10);
        assert!(c.range.1 - c.range.0 > 1.5 && c.range.1 - c.range.0 < 2.0);
        assert_traced_alike(&a, &b, &cs, 4.0 * SCREW_FIT);
        // Moved together, the sheets meet in the moved helix.
        let t = Transform::rotation(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.6, 0.0, 0.8), 0.7)
            .then(&Transform::translation(Vec3::new(-2.0, 1.0, 0.5)));
        let (ta, tb) = (a.transformed(&t), b.transformed(&t));
        let moved = curves(&ta, &tb);
        assert_eq!(moved.len(), 1);
        let q = moved[0].curve.eval(moved[0].range.0);
        assert!(q.dist(t.point(p0)).min(q.dist(t.point(p1))) <= 1e-9);
        assert_traced_alike(&ta, &tb, &moved, 4.0 * SCREW_FIT);
    }

    /// The parabola through `ends` and bent towards `mid`, slid 3 along z.
    fn slid(ends: [[f64; 3]; 2], mid: [f64; 3]) -> Surface {
        let pts = [ends[0], mid, ends[1]].map(pt);
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let net = pts
            .iter()
            .copied()
            .chain(pts.iter().map(|&p| p + Vec3::Z * 3.0))
            .collect();
        let along = vec![0.0, 0.0, 3.0, 3.0];
        let spline = NurbsSurface::new((1, 2), (along, knots.clone()), (2, 3), net, None).unwrap();
        let swept = Swept {
            generator: crate::geom::NurbsCurve::new(2, knots, pts.to_vec(), None).unwrap(),
            motion: Steady {
                origin: Point3::new(5.0, 5.0, 5.0),
                axis: Vec3::Z,
                turn: 0.0,
                rise: 1.0,
            },
        };
        Surface::Nurbs(Box::new(spline.swept_by(swept)))
    }

    // y = x² from z = 0 to 3 and y = ½ − x² from z = 1 to 4 share the lines at x = ±½, y = ¼, from
    // z = 1 to 3.
    #[test]
    fn sheets_of_one_slide_meet_in_lines_both_cover() {
        let _held = hold();
        let a = slid([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], [0.0, -1.0, 0.0]);
        let b = slid([[-1.0, -0.5, 1.0], [1.0, -0.5, 1.0]], [0.0, 1.5, 1.0]);
        let cs = curves(&a, &b);
        assert_eq!(cs.len(), 2);
        for c in &cs {
            assert!(matches!(c.curve, Curve::Line(_)));
            let (p, q) = (c.curve.eval(c.range.0), c.curve.eval(c.range.1));
            for (end, z) in [(p, 1.0), (q, 3.0)] {
                let want = Point3::new(0.5 * end.x.signum(), 0.25, z);
                assert!(end.dist(want) <= 1e-12, "{end:?}");
            }
        }
        assert!(cs[0].curve.eval(0.0).x * cs[1].curve.eval(0.0).x < 0.0);
        assert_traced_alike(&a, &b, &cs, 1e-12);
    }

    // The hyperboloid of the line x = 2 is 2.5 from the axis where 4 + (6t − 3)² = 6.25, at heights
    // ½ and 1½, and nearest it, at 2, at height 1.
    #[test]
    fn a_turned_sheet_meets_surfaces_about_its_axis_in_arcs_of_its_own_length() {
        let _held = hold();
        let (a, _) = turned();
        let about_z = |radius: f64| {
            Surface::Cylinder(Cylinder {
                frame: Frame::WORLD,
                radius,
            })
        };
        let wide = about_z(2.5);
        let cs = curves(&a, &wide);
        assert_eq!(cs.len(), 2);
        for c in &cs {
            let Curve::Circle(circle) = &c.curve else {
                panic!("an orbit of a turn is a circle");
            };
            let z = circle.frame.origin.z;
            assert!((circle.radius - 2.5).abs() <= 1e-12);
            assert!(((z - 1.0).abs() - 0.5).abs() <= 1e-12, "{z}");
            assert!((c.range.1 - c.range.0 - 1.5).abs() <= 1e-12 && !c.tangent);
        }
        assert_traced_alike(&a, &wide, &cs, 4.0 * SCREW_FIT);
        // Through the line's nearest point the cylinder touches the sheet along one circle.
        let touching = curves(&a, &about_z(2.0));
        assert_eq!(touching.len(), 1);
        assert!(touching[0].tangent);
        assert!((touching[0].curve.eval(0.3).z - 1.0).abs() <= 1e-9);
        let cone = Surface::Cone(Cone {
            frame: Frame::WORLD,
            radius: 1.5,
            half_angle: 0.6,
        });
        let cs = curves(&a, &cone);
        assert!(!cs.is_empty());
        assert_traced_alike(&a, &cone, &cs, 4.0 * SCREW_FIT);
        // Through the whole pair, either way round.
        for (x, y) in [(&a, &cone), (&cone, &a)] {
            let found = ssi::intersect(x, y, &region(), &mut || panic!("traced")).unwrap();
            let Ssi::Curves { curves, .. } = found else {
                panic!("coincident");
            };
            assert_eq!(curves.len(), cs.len());
        }
    }

    #[test]
    fn a_screwed_sheet_meets_a_cylinder_about_its_axis_in_helices() {
        let _held = hold();
        let (a, _) = screwed(&screw(0.5));
        let rod = Surface::Cylinder(Cylinder {
            frame: Frame::WORLD,
            radius: 2.5,
        });
        let cs = curves(&a, &rod);
        assert_eq!(cs.len(), 1);
        assert!((cs[0].range.1 - cs[0].range.0 - 2.0).abs() <= 1e-12);
        for p in samples(&cs[0]) {
            assert!((p.x.hypot(p.y) - 2.5).abs() <= 2.0 * SCREW_FIT);
        }
        assert_traced_alike(&a, &rod, &cs, 4.0 * SCREW_FIT);
        // A cylinder about another line is not made of the screw's orbits.
        let beside = Surface::Cylinder(Cylinder {
            frame: Frame::WORLD.translated(Vec3::new(0.5, 0.0, 0.0)),
            radius: 2.5,
        });
        assert!(intersect(&a, &beside, &region()).is_none());
    }

    #[test]
    fn pairs_the_orbits_do_not_settle_are_left_to_the_trace() {
        let _held = hold();
        let s = screw(0.5);
        let (a, b) = screwed(&s);
        // Another pitch, another hand and another axis are other motions.
        let steeper = screw(0.5 + 1e-6);
        let left = Helix { sign: -1.0, ..s };
        let beside = Helix {
            origin: Point3::new(1e-6, 0.0, 0.0),
            ..s
        };
        for other in [steeper, left, beside] {
            let (_, b) = screwed(&other);
            assert!(intersect(&a, &b, &region()).is_none());
        }
        // The same sheet begun a little on lies in the first: they share no curve.
        let on = s.transform(0.5);
        let ends = [[1.0, 0.0, 0.0], [4.0, 0.0, 1.0]].map(|p| on.point(pt(p)));
        let later = swept(ends[0], ends[1], &s, 2.0);
        assert!(intersect(&a, &later, &region()).is_none());
        // A turn that comes round covers each circle more than once.
        let round = swept(pt([2.0, -3.0, 0.0]), pt([2.0, 3.0, 2.0]), &screw(0.0), TAU);
        let (_, other) = turned();
        assert!(intersect(&round, &other, &region()).is_none());
        assert!(intersect(&other, &round, &region()).is_none());
        // Such a pair is still intersected, by the trace.
        let seed = curves(&a, &b)[0].curve.eval(1.0);
        let (_, b) = screwed(&steeper);
        let traced = ssi::intersect(&a, &b, &region(), &mut || vec![seed]).unwrap();
        assert!(matches!(traced, Ssi::Curves { curves, .. } if curves.len() == 1));
    }
}
