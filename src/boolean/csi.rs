//! Curve–surface and curve–curve intersection, signed distance to a surface, and ray hits.
//!
//! Two crossings closer than the linear tolerance are one touching point: a curve whose excursion
//! past a surface is within [`tol::LINEAR`] touches it once, at the excursion's extreme.

use std::{collections::HashMap, f64::consts::TAU};

use super::hulls;
use crate::{
    geom::{
        Bezier, Circle, Curve, Cylinder, Extrusion, Line, NurbsSurface, Revolution, Ruled, Surface,
        Torus,
    },
    math::{Aabb, Point3, Vec2, Vec3},
    numeric, tol,
};

/// The signed distance from `p` to `s`, positive on the side of the natural normal: exact near the
/// analytic surfaces, and along the projection's normal for the others.
pub(crate) fn signed_distance(s: &Surface, p: Point3) -> f64 {
    match s {
        Surface::Plane(pl) => (p - pl.frame.origin).dot(pl.frame.z),
        Surface::Cylinder(c) => {
            let l = c.frame.to_local(p);
            l.x.hypot(l.y) - c.radius
        }
        Surface::Sphere(sp) => (p - sp.frame.origin).norm() - sp.radius,
        // In the meridian plane through `p` a cone is two lines through its apex and a torus two
        // circles, one each side of the axis, each with its own natural normal (a cone's turns
        // over past the apex); the nearer of the two is the distance. Past a cone's apex, and
        // inside a spindle torus, the far one can be nearer.
        Surface::Cone(c) => {
            let l = c.frame.to_local(p);
            let (sa, ca) = c.half_angle.sin_cos();
            let rho = l.x.hypot(l.y);
            let near = (rho - c.radius) * ca - l.z * sa;
            let far = (rho + c.radius) * ca + l.z * sa;
            if far.abs() < near.abs() { far } else { near }
        }
        Surface::Torus(t) => {
            let l = t.frame.to_local(p);
            let rho = l.x.hypot(l.y);
            let near = (rho - t.major).hypot(l.z) - t.minor;
            let far = (rho + t.major).hypot(l.z) - t.minor;
            if far.abs() < near.abs() { far } else { near }
        }
        _ => {
            let uv = s.project(p);
            (p - s.eval(uv.x, uv.y)).dot(s.normal(uv.x, uv.y))
        }
    }
}

/// The signed distance to `s` along a path of points: each projection onto a B-spline surface
/// continues from the last point's where the last point lay near enough the surface, and the
/// step from it is short enough, for that to find the nearest point. A surface of revolution's
/// meridian is sampled once for the whole path; each point is still projected from every sample,
/// as a meridian that turns back has two sheets no step is short enough to tell apart.
pub(crate) struct Tracker<'a> {
    s: &'a Surface,
    last: Option<(Point3, Vec2, f64)>,
    meridian: Option<crate::geom::Meridian>,
}

impl<'a> Tracker<'a> {
    pub(crate) fn new(s: &'a Surface) -> Tracker<'a> {
        Tracker {
            s,
            last: None,
            meridian: None,
        }
    }

    pub(crate) fn at(&mut self, p: Point3) -> f64 {
        if let Surface::Revolution(r) = self.s {
            let m = (self.meridian).get_or_insert_with(|| crate::geom::Meridian::of(r));
            let uv = self.s.project_on(m, p);
            return (p - self.s.eval(uv.x, uv.y)).dot(self.s.normal(uv.x, uv.y));
        }
        let Surface::Nurbs(n) = self.s else {
            return signed_distance(self.s, p);
        };
        let uv = match self.last {
            Some((q, uv, d)) if d.abs() + q.dist(p) <= n.near_radius() => n.project_near(p, uv),
            _ => n.project(p),
        };
        let d = (p - n.eval(uv.x, uv.y)).dot(self.s.normal(uv.x, uv.y));
        self.last = Some((p, uv, d));
        d
    }
}

/// Where a curve meets a surface: the parameters of its crossing and touching points in its range,
/// or `on_surface` when the whole curve lies in the surface.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Hits {
    pub ts: Vec<f64>,
    pub on_surface: bool,
}

fn speed(c: &Curve, t: f64) -> f64 {
    c.d1(t).1.norm()
}

fn in_range(c: &Curve, t: f64, range: (f64, f64)) -> Option<f64> {
    let pt = tol::param(speed(c, t));
    if t >= range.0 - pt && t <= range.1 + pt {
        Some(t.clamp(range.0, range.1))
    } else {
        None
    }
}

/// Moves an angle by whole turns into `range` when it fits there.
fn angle_in(t: f64, range: (f64, f64)) -> Option<f64> {
    let k = ((range.0 - t) / TAU).ceil();
    let a = t + k * TAU;
    let eps = 1e-14 * (1.0 + a.abs());
    if a <= range.1 + eps {
        Some(a.min(range.1))
    } else if (t + (k - 1.0) * TAU - range.0).abs() <= eps {
        Some(range.0)
    } else {
        None
    }
}

/// `c` and `s` meet where `f(t) = A + B cos t + C sin t` vanishes.
fn trig_roots(a: f64, b: f64, c: f64, range: (f64, f64)) -> Hits {
    let r = b.hypot(c);
    if r <= tol::LINEAR {
        return Hits {
            ts: Vec::new(),
            on_surface: a.abs() <= tol::LINEAR,
        };
    }
    let phi = c.atan2(b);
    let mut ts = Vec::new();
    if a.abs() >= r - tol::LINEAR {
        // At most one touching point, where the curve comes nearest.
        if a.abs() - r <= tol::LINEAR {
            ts.push(if a < 0.0 { phi } else { phi + TAU / 2.0 });
        }
    } else {
        let w = (-a / r).acos();
        ts.push(phi + w);
        ts.push(phi - w);
    }
    Hits {
        ts: ts.into_iter().filter_map(|t| angle_in(t, range)).collect(),
        on_surface: false,
    }
}

/// The crossings of a line `o + t·d` (`d` unit) with the cylinder.
fn line_cylinder(o: Point3, d: Vec3, cyl: &Cylinder) -> Hits {
    let lo = cyl.frame.to_local(o);
    let ld = cyl.frame.dir_to_local(d);
    let a = ld.x * ld.x + ld.y * ld.y;
    if a.sqrt() <= tol::ANGULAR {
        return Hits {
            ts: Vec::new(),
            on_surface: (lo.x.hypot(lo.y) - cyl.radius).abs() <= tol::LINEAR,
        };
    }
    let t_near = -(lo.x * ld.x + lo.y * ld.y) / a;
    let m = (lo.x + t_near * ld.x).hypot(lo.y + t_near * ld.y);
    let r = cyl.radius;
    let ts = if m > r + tol::LINEAR {
        Vec::new()
    } else if m >= r - tol::LINEAR {
        vec![t_near]
    } else {
        let h = ((r - m) * (r + m) / a).sqrt();
        vec![t_near - h, t_near + h]
    };
    Hits {
        ts,
        on_surface: false,
    }
}

/// The crossings of a line with a quadric whose implicit form along the line is `qa·t² + qb·t +
/// qc`; a pair the line runs between within the tolerance band is one touching point.
fn line_quadric(o: Point3, d: Vec3, s: &Surface, qa: f64, qb: f64, qc: f64) -> Hits {
    let at = |t: f64| signed_distance(s, o + d * t).abs();
    if qa.abs() < 1e-300 {
        if qb.abs() < 1e-300 {
            return Hits {
                ts: Vec::new(),
                on_surface: at(0.0) <= tol::LINEAR,
            };
        }
        return Hits {
            ts: vec![-qc / qb],
            on_surface: false,
        };
    }
    let t_near = -qb / (2.0 * qa);
    let disc = qb * qb - 4.0 * qa * qc;
    let ts = if disc <= 0.0 {
        if at(t_near) <= tol::LINEAR {
            vec![t_near]
        } else {
            Vec::new()
        }
    } else {
        let q = -0.5 * (qb + qb.signum() * disc.sqrt());
        let (mut t1, mut t2) = (q / qa, qc / q);
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }
        // One touching point where the line is within the band midway between the roots, which
        // is where it strays furthest. A pair split only by rounding, as a double root at a cone's
        // apex is, can lie more than the band apart. A pair further apart than a model has one
        // root out where a line nearly parallel to a cone's generator meets it, and the surface's
        // distance read that far out says nothing.
        if (t2 - t1) * d.norm() <= tol::MODEL_SIZE && at(t_near) <= tol::LINEAR {
            vec![t_near]
        } else {
            vec![t1, t2]
        }
    };
    Hits {
        ts,
        on_surface: false,
    }
}

/// Golden-section minimisation of `f` on [a, b].
fn minimise(f: &mut impl FnMut(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let g = 0.5 * (5f64.sqrt() - 1.0);
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if (b - a).abs() <= 1e-15 * (1.0 + a.abs()) {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// Where `f`'s slope changes sign in [a, b], if it does: an extremum pinned by the sign of the slope,
/// to rounding.
fn stationary(f: &mut impl FnMut(f64) -> f64, a: f64, b: f64) -> Option<f64> {
    let h = 1e-3 * (b - a);
    let mut slope = |t: f64| f(t + h) - f(t - h);
    let (sa, sb) = (slope(a + h), slope(b - h));
    ((sa < 0.0) != (sb < 0.0) && sa != 0.0 && sb != 0.0)
        .then(|| numeric::bisect(&mut slope, a + h, b - h))
}

/// The value at an extremum below which the curve touches the surface there rather than crossing
/// it twice: a curve that dips into the surface by no more than the tolerance lies on it there, and
/// its sign changes sit about √(dip / curvature) away, far more than the tolerance along the curve.
const TOUCH: f64 = tol::LINEAR;

/// A root bisected between samples, moved onto the extremum it lies beside when the curve only
/// touches the surface there, held to [lo, hi]. A touch leaves the surface on one side of it both
/// ways; a curve that runs along the surface has slope changes from rounding alone.
fn settle(f: &mut impl FnMut(f64) -> f64, t: f64, w: f64, (lo, hi): (f64, f64)) -> f64 {
    let (fa, fb) = (f(t - w), f(t + w));
    if (fa < 0.0) != (fb < 0.0) || fa.abs().min(fb.abs()) <= tol::LINEAR {
        return t;
    }
    match stationary(f, t - w, t + w) {
        Some(s) if f(s).abs() <= TOUCH => s.clamp(lo, hi),
        _ => t,
    }
}

/// Sign changes and near-zero minima of `f` on `cuts` spans, `n` samples per span.
fn sampled_roots(f: &mut impl FnMut(f64) -> f64, cuts: &[f64], n: usize) -> (Vec<f64>, bool) {
    let mut ts = Vec::new();
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for w in cuts.windows(2) {
        for i in 0..n {
            let t = w[0] + (w[1] - w[0]) * i as f64 / n as f64;
            samples.push((t, f(t)));
        }
    }
    let last = cuts[cuts.len() - 1];
    samples.push((last, f(last)));
    if samples.iter().all(|s| s.1.abs() <= tol::LINEAR) {
        return (Vec::new(), true);
    }
    for k in 0..samples.len() {
        let (t, v) = samples[k];
        if v == 0.0 {
            ts.push(t);
            continue;
        }
        if k + 1 < samples.len() {
            let (t2, v2) = samples[k + 1];
            if v2 != 0.0 && (v < 0.0) != (v2 < 0.0) {
                let r = numeric::bisect(f, t, t2);
                ts.push(settle(f, r, t2 - t, (cuts[0], last)));
            }
        }
        if k > 0 && k + 1 < samples.len() {
            let (tp, vp) = samples[k - 1];
            let (tn, vn) = samples[k + 1];
            let same = (vp < 0.0) == (v < 0.0) && (vn < 0.0) == (v < 0.0);
            if same && v.abs() <= vp.abs() && v.abs() <= vn.abs() {
                // The extremum between the samples, not where |f| first reaches zero: a curve that
                // touches the surface to within the tolerance dips through it by rounding, and its
                // zeros then lie √(dip / curvature) either side of the touching point.
                let s = v.signum();
                let tm = minimise(&mut |x: f64| s * f(x), tp, tn);
                let fm = f(tm);
                if fm.abs() <= tol::LINEAR {
                    ts.push(tm);
                } else if (fm < 0.0) != (v < 0.0) {
                    ts.push(numeric::bisect(f, tp, tm));
                    ts.push(numeric::bisect(f, tm, tn));
                }
            }
        }
    }
    // A touch is reached from the sign changes either side of it and from the extremum between.
    ts.sort_by(f64::total_cmp);
    ts.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs())));
    (ts, false)
}

/// A curve's boxes over stretches of its parameter, kept while it is tested against many surfaces:
/// a spline's box costs a pass over its pieces. One holds the boxes of one curve only.
#[derive(Default)]
pub(crate) struct CurveBoxes(HashMap<(u64, u64), Aabb>);

impl CurveBoxes {
    pub(crate) fn bbox(&mut self, c: &Curve, t0: f64, t1: f64) -> Aabb {
        if matches!(c, Curve::Line(_) | Curve::Circle(_) | Curve::Ellipse(_)) {
            return c.bbox(t0, t1);
        }
        *self
            .0
            .entry((t0.to_bits(), t1.to_bits()))
            .or_insert_with(|| c.bbox(t0, t1))
    }
}

fn numeric_hits(c: &Curve, range: (f64, f64), s: &Surface, boxes: &mut CurveBoxes) -> Hits {
    let mut cuts = vec![range.0];
    cuts.extend(c.breakpoints(range.0, range.1));
    cuts.push(range.1);
    let n = if cuts.len() > 2 { 12 } else { 96 };
    let mut track = Tracker::new(s);
    let mut f = |t: f64| track.at(c.eval(t));
    let Surface::Nurbs(ns) = s else {
        let (ts, on) = sampled_roots(&mut f, &cuts, n);
        return Hits { ts, on_surface: on };
    };
    // The curve can meet the surface only where a stretch of it between samples comes near a
    // patch's box: only runs of such stretches are sampled, at the same points.
    let samples: Vec<f64> = cuts
        .windows(2)
        .flat_map(|w| (0..n).map(move |i| w[0] + (w[1] - w[0]) * i as f64 / n as f64))
        .chain(std::iter::once(range.1))
        .collect();
    // A box that holds a patch's box meets the stretch's wherever the patch's does, so the tree
    // finds a patch exactly where a scan of them all would.
    let tree = ns.patch_tree();
    let meets = |a: &Aabb, b: &Aabb| {
        (0..3).all(|k| a.min[k] <= b.max[k] + tol::LINEAR && b.min[k] <= a.max[k] + tol::LINEAR)
    };
    let mut near = |t0: f64, t1: f64| {
        let bb = boxes.bbox(c, t0, t1);
        tree.any(|q| meets(&bb, q))
    };
    let (mut ts, mut on) = (Vec::new(), true);
    let mut run: Vec<f64> = Vec::new();
    for w in samples.windows(2) {
        if near(w[0], w[1]) {
            if run.is_empty() {
                run.push(w[0]);
            }
            run.push(w[1]);
            continue;
        }
        on = false;
        if !run.is_empty() {
            ts.extend(sampled_roots(&mut f, &run, 1).0);
            run.clear();
        }
    }
    if !run.is_empty() {
        let (r, all) = sampled_roots(&mut f, &run, 1);
        ts.extend(r);
        on &= all;
    }
    if on {
        return Hits {
            ts: Vec::new(),
            on_surface: true,
        };
    }
    Hits {
        ts,
        on_surface: false,
    }
}

/// Where the curve, over `range`, meets the surface.
pub(crate) fn curve_surface(c: &Curve, range: (f64, f64), s: &Surface) -> Hits {
    curve_surface_in(c, range, s, &mut CurveBoxes::default())
}

/// [`curve_surface`], with the curve's boxes kept in `boxes` for its tests against other surfaces.
pub(crate) fn curve_surface_in(
    c: &Curve,
    range: (f64, f64),
    s: &Surface,
    boxes: &mut CurveBoxes,
) -> Hits {
    let raw = match (c, s) {
        (Curve::Line(l), Surface::Plane(p)) => {
            let dn = l.dir.dot(p.frame.z);
            let h = (l.origin - p.frame.origin).dot(p.frame.z);
            if dn.abs() <= tol::ANGULAR {
                Hits {
                    ts: Vec::new(),
                    on_surface: h.abs() <= tol::LINEAR,
                }
            } else {
                Hits {
                    ts: vec![-h / dn],
                    on_surface: false,
                }
            }
        }
        (Curve::Line(l), Surface::Cylinder(cyl)) => line_cylinder(l.origin, l.dir, cyl),
        (Curve::Line(l), Surface::Torus(t)) => line_torus(l.origin, l.dir, t, range),
        (Curve::Circle(ci), Surface::Sphere(sp)) => return circle_sphere(ci, sp, range),
        // A line along an extrusion's generators keeps one distance from it.
        (Curve::Line(l), Surface::Extrusion(e)) if l.dir.cross(e.dir).norm() <= tol::ANGULAR => {
            Hits {
                ts: Vec::new(),
                on_surface: signed_distance(s, l.origin).abs() <= tol::LINEAR,
            }
        }
        (Curve::Line(l), Surface::Extrusion(e)) => line_extrusion(l.origin, l.dir, e),
        (Curve::Line(l), Surface::Revolution(r)) => match line_revolution(l.origin, l.dir, r) {
            Some(hits) => hits,
            None => return numeric_hits(c, range, s, boxes),
        },
        (Curve::Line(l), Surface::Ruled(r)) => match line_ruled(l.origin, l.dir, r) {
            Some(hits) => hits,
            None => return numeric_hits(c, range, s, boxes),
        },
        (Curve::Line(l), Surface::Nurbs(ns)) => match line_spline(l.origin, l.dir, ns, range) {
            Some(hits) => hits,
            None => return numeric_hits(c, range, s, boxes),
        },
        (Curve::Line(l), Surface::Sphere(sp)) => {
            let oc = l.origin - sp.frame.origin;
            let b = oc.dot(l.dir);
            let c0 = oc.norm2() - sp.radius * sp.radius;
            line_quadric(l.origin, l.dir, s, 1.0, 2.0 * b, c0)
        }
        (Curve::Line(l), Surface::Cone(cone)) if cone.half_angle.cos().abs() > 1e-9 => {
            let lo = cone.frame.to_local(l.origin);
            let ld = cone.frame.dir_to_local(l.dir);
            let k = cone.half_angle.tan();
            let r0 = cone.radius + k * lo.z;
            let qa = ld.x * ld.x + ld.y * ld.y - k * k * ld.z * ld.z;
            let qb = 2.0 * (lo.x * ld.x + lo.y * ld.y - k * r0 * ld.z);
            let qc = lo.x * lo.x + lo.y * lo.y - r0 * r0;
            line_quadric(l.origin, l.dir, s, qa, qb, qc)
        }
        (Curve::Circle(ci), Surface::Plane(p)) => {
            let n = p.frame.z;
            let a = (ci.frame.origin - p.frame.origin).dot(n);
            return trig_roots(
                a,
                ci.radius * ci.frame.x.dot(n),
                ci.radius * ci.frame.y.dot(n),
                range,
            );
        }
        (Curve::Ellipse(e), Surface::Plane(p)) => {
            let n = p.frame.z;
            let a = (e.frame.origin - p.frame.origin).dot(n);
            return trig_roots(
                a,
                e.major * e.frame.x.dot(n),
                e.minor * e.frame.y.dot(n),
                range,
            );
        }
        (Curve::Circle(ci), Surface::Cylinder(cyl))
            if ci.frame.z.cross(cyl.frame.z).norm() <= tol::ANGULAR =>
        {
            return circle_parallel_cylinder(ci, cyl, range);
        }
        _ => return numeric_hits(c, range, s, boxes),
    };
    if raw.on_surface {
        return raw;
    }
    Hits {
        ts: raw
            .ts
            .into_iter()
            .filter_map(|t| in_range(c, t, range))
            .collect(),
        on_surface: false,
    }
}

/// A circle square to a cylinder's axis meets it where it meets the cylinder's section in its own
/// plane; a fillet's tangent contact sampled numerically would split into two nearby crossings.
fn circle_parallel_cylinder(ci: &Circle, cyl: &Cylinder, range: (f64, f64)) -> Hits {
    let l = ci.frame.to_local(cyl.frame.origin);
    coplanar_circles(ci, Vec2::new(l.x, l.y), cyl.radius, range)
}

/// A circle against another of radius `big` centred at `centre` in its plane (local coordinates).
fn coplanar_circles(ci: &Circle, centre: Vec2, big: f64, range: (f64, f64)) -> Hits {
    let d = centre.x.hypot(centre.y);
    let r = ci.radius;
    if d <= tol::LINEAR {
        return Hits {
            ts: Vec::new(),
            on_surface: (r - big).abs() <= tol::LINEAR,
        };
    }
    if d > r + big + tol::LINEAR || d < (r - big).abs() - tol::LINEAR {
        return Hits::default();
    }
    let toward = centre.y.atan2(centre.x);
    let x = (d * d + r * r - big * big) / (2.0 * d);
    let touching =
        (d - (r + big)).abs() <= tol::LINEAR || (d - (r - big).abs()).abs() <= tol::LINEAR;
    let angles = if touching {
        vec![if x >= 0.0 { toward } else { toward + TAU / 2.0 }]
    } else {
        let a = (x / r).clamp(-1.0, 1.0).acos();
        vec![toward - a, toward + a]
    };
    Hits {
        ts: angles
            .into_iter()
            .filter_map(|a| angle_in(a, range))
            .collect(),
        on_surface: false,
    }
}

/// The values of `c[0] + c[1]·t + …` and its derivative's coefficients.
fn poly(c: &[f64], t: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &k| acc * t + k)
}

fn derivative(c: &[f64]) -> Vec<f64> {
    c.iter()
        .enumerate()
        .skip(1)
        .map(|(i, &k)| k * i as f64)
        .collect()
}

/// The polynomial's stationary points inside (lo, hi), in order: each is the one root of its
/// derivative on a stretch where the derivative is monotone, found from the next derivative's
/// stationary points in turn, so a double root is found as a stationary point, not missed between
/// samples.
fn stationary_points(c: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let d = derivative(c);
    if d.len() < 2 {
        return Vec::new();
    }
    let mut knots = vec![lo];
    knots.extend(stationary_points(&d, lo, hi));
    knots.push(hi);
    let mut out = Vec::new();
    for w in knots.windows(2) {
        let (a, b) = (poly(&d, w[0]), poly(&d, w[1]));
        if a == 0.0 && w[0] > lo {
            out.push(w[0]);
        } else if a != 0.0 && b != 0.0 && (a < 0.0) != (b < 0.0) {
            out.push(numeric::bisect(&mut |t| poly(&d, t), w[0], w[1]));
        }
    }
    out
}

/// Where `f`, a distance in mm, is zero over [lo, hi], given `knots` that split the range into
/// stretches on which it has at most one root. A knot within the tolerance of the surface is a
/// meeting there, a crossing or a touch alike; a stretch whose ends lie beyond it on opposite
/// sides has its crossing bisected; a stretch with one end on the surface adds nothing.
fn roots_between(f: &mut impl FnMut(f64) -> f64, knots: &[f64]) -> Vec<f64> {
    let side = |v: f64| {
        if v > tol::LINEAR {
            1
        } else if v < -tol::LINEAR {
            -1
        } else {
            0
        }
    };
    let signs: Vec<i8> = knots.iter().map(|&t| side(f(t))).collect();
    let mut ts = Vec::new();
    for (i, &t) in knots.iter().enumerate() {
        if signs[i] == 0 {
            ts.push(t);
        }
        if i + 1 < knots.len() && signs[i] * signs[i + 1] < 0 {
            ts.push(numeric::bisect(f, t, knots[i + 1]));
        }
    }
    ts
}

/// The coefficients `c[0] + c[1]·s + …` of the polynomial of degree `n` through `n + 1` values at
/// `s = 0, 1/n, …, 1`; `None` where the fit is singular.
fn fitted(values: &[f64]) -> Option<Vec<f64>> {
    let n = values.len();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let s = j as f64 / (n - 1).max(1) as f64;
            let mut row: Vec<f64> = (0..n).map(|k| s.powi(k as i32)).collect();
            row.push(values[j]);
            row
        })
        .collect();
    for col in 0..n {
        let pivot = (col..n).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, pivot);
        let by = m[col].clone();
        for (row, values) in m.iter_mut().enumerate() {
            if row != col {
                let f = values[col] / by[col];
                for (x, p) in values.iter_mut().zip(&by).skip(col) {
                    *x -= f * p;
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

/// A stretch of a curve's parameter over which the curve's point, times a positive scale, is a
/// polynomial: the parameter at `s` in [0, 1], the scale there, and the polynomial's degree in `s`.
/// A function of order `k` in the point (1 for a distance from a plane, 2 for a squared radius) is
/// then a polynomial of degree `k·degree` once multiplied by the scale `k` times over.
struct Piece<'a> {
    param: Box<dyn Fn(f64) -> f64 + 'a>,
    scale: Box<dyn Fn(f64) -> f64 + 'a>,
    degree: usize,
}

/// The pieces of `c` over `(lo, hi)`: a spline's own, each a polynomial, or a rational one's with
/// its denominator for scale; a line in one; a circle or an ellipse in two half turns, each a
/// quadratic over `1 + τ²` in the tangent `τ` of the half angle, whatever the range. `None` for a
/// line without ends and for an intersection curve, which is no polynomial in any parameter.
fn pieces<'a>(c: &'a Curve, (lo, hi): (f64, f64)) -> Option<Vec<Piece<'a>>> {
    let spans = |degree: usize, weight: Option<&'a crate::geom::NurbsCurve>| {
        let mut cuts = vec![lo];
        cuts.extend(c.breakpoints(lo, hi));
        cuts.push(hi);
        cuts.windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                let param = move |s: f64| a + (b - a) * s;
                Piece {
                    param: Box::new(param),
                    scale: match weight {
                        Some(nc) => Box::new(move |s| nc.weight(param(s))),
                        None => Box::new(|_| 1.0),
                    },
                    degree,
                }
            })
            .collect::<Vec<_>>()
    };
    match c {
        Curve::Line(_) | Curve::BSpline(_) | Curve::Nurbs(_)
            if !(lo.is_finite() && hi.is_finite()) =>
        {
            None
        }
        Curve::Line(_) => Some(spans(1, None)),
        Curve::BSpline(b) => Some(spans(b.degree(), None)),
        Curve::Nurbs(nc) => Some(spans(nc.degree(), Some(nc))),
        Curve::Circle(_) | Curve::Ellipse(_) => Some(
            [TAU / 4.0, 3.0 * TAU / 4.0]
                .into_iter()
                .map(|middle| Piece {
                    param: Box::new(move |s| middle + 2.0 * (2.0 * s - 1.0).atan()),
                    scale: Box::new(|s| 1.0 + (2.0 * s - 1.0).powi(2)),
                    degree: 2,
                })
                .collect(),
        ),
        Curve::Intersection(_) => None,
    }
}

/// Where a function of a curve's point vanishes, over the curve's `pieces`. `raw` is the function
/// in any unit, a polynomial over each piece once scaled, of order `order` in the point; `mm` has
/// its sign and is a distance in mm. The polynomial's stationary points split each piece into
/// stretches with one crossing at most and mark every place the function can touch zero, so no
/// crossing is stepped over however close two lie. Also whether the function vanishes all along.
fn piece_roots(
    pieces: &[Piece],
    order: usize,
    raw: &dyn Fn(f64) -> f64,
    mm: &mut impl FnMut(f64) -> f64,
) -> (Vec<f64>, bool) {
    let (mut roots, mut flat) = (Vec::new(), true);
    for piece in pieces {
        let n = piece.degree * order;
        let at = |j: usize| j as f64 / n.max(1) as f64;
        let values: Vec<f64> = (0..=n)
            .map(|j| {
                let s = at(j);
                raw((piece.param)(s)) * (piece.scale)(s).powi(order as i32)
            })
            .collect();
        flat &= (0..=n).all(|j| mm((piece.param)(at(j))).abs() <= tol::LINEAR);
        let mut knots = vec![(piece.param)(0.0)];
        if let Some(coefficients) = fitted(&values) {
            knots.extend(
                stationary_points(&coefficients, 0.0, 1.0)
                    .into_iter()
                    .map(|s| (piece.param)(s)),
            );
        }
        knots.push((piece.param)(1.0));
        roots.extend(roots_between(mm, &knots));
    }
    (roots, flat)
}

/// Line parameters in order, two within rounding of each other as one: a crossing at the join of
/// two pieces is found from both.
fn in_order(mut ts: Vec<f64>) -> Vec<f64> {
    ts.sort_by(f64::total_cmp);
    ts.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs())));
    ts
}

/// A line against an extrusion, whose profile is carried along `dir`: the line meets it where the
/// profile passes through the plane that holds the line and runs along `dir`. That is a function
/// of the profile's parameter alone, the profile's signed distance from that plane: a sine wave
/// on a circle or an ellipse, and on a spline a polynomial over each piece. Sampling along the
/// line instead steps over crossings closer than its step, as a gear's root has.
fn line_extrusion(o: Point3, d: Vec3, e: &Extrusion) -> Hits {
    let across = d - e.dir * d.dot(e.dir);
    let Some(n) = e.dir.cross(across).unit() else {
        return Hits::default();
    };
    let c = &e.curve;
    let height = |u: f64| n.dot(c.eval(u) - o);
    let (lo, hi) = c.domain();
    let on_profile = match c {
        Curve::Circle(ci) => trig_roots(
            n.dot(ci.frame.origin - o),
            ci.radius * n.dot(ci.frame.x),
            ci.radius * n.dot(ci.frame.y),
            (0.0, TAU),
        ),
        Curve::Ellipse(el) => trig_roots(
            n.dot(el.frame.origin - o),
            el.major * n.dot(el.frame.x),
            el.minor * n.dot(el.frame.y),
            (0.0, TAU),
        ),
        Curve::Line(l) => {
            let slope = n.dot(l.dir);
            if slope.abs() <= tol::ANGULAR {
                Hits {
                    ts: Vec::new(),
                    on_surface: n.dot(l.origin - o).abs() <= tol::LINEAR,
                }
            } else {
                Hits {
                    ts: vec![-n.dot(l.origin - o) / slope],
                    on_surface: false,
                }
            }
        }
        _ => match pieces(c, (lo, hi)) {
            Some(pieces) => {
                let (ts, flat) = piece_roots(&pieces, 1, &height, &mut |u| height(u));
                Hits {
                    ts: if flat { Vec::new() } else { ts },
                    on_surface: flat,
                }
            }
            // An intersection curve is no polynomial: sampled along the profile, piece by piece.
            None => {
                let mut cuts = vec![lo];
                cuts.extend(c.breakpoints(lo, hi));
                cuts.push(hi);
                let (ts, on) = sampled_roots(&mut |u| height(u), &cuts, 24);
                Hits { ts, on_surface: on }
            }
        },
    };
    if on_profile.on_surface {
        return on_profile;
    }
    let ts = on_profile
        .ts
        .into_iter()
        .map(|u| (c.eval(u) - o).dot(across) / across.norm2())
        .collect();
    Hits {
        ts: in_order(ts),
        on_surface: false,
    }
}

/// A line against a surface of revolution. Each point of the profile is a ring about the axis, at
/// a height and of a radius; a line that climbs passes that height once, at a known distance from
/// the axis, and meets the ring where the two radii agree. Their squares' difference is a
/// polynomial over each piece of the profile. A line square to the axis keeps one height: it meets
/// the rings the profile draws at that height, each in the two roots of a quadratic. `None` where
/// the profile has no pieces.
fn line_revolution(o: Point3, d: Vec3, r: &Revolution) -> Option<Hits> {
    let c = &r.curve;
    let pieces = pieces(c, c.domain())?;
    let rel = o - r.origin;
    let (z0, dz) = (rel.dot(r.axis), d.dot(r.axis));
    // The line's squared distance from the axis at `t`: a·t² + b·t + c0.
    let (a, b, c0) = (
        1.0 - dz * dz,
        2.0 * (rel.dot(d) - z0 * dz),
        rel.norm2() - z0 * z0,
    );
    // A point of the profile as a ring: its height and its squared radius.
    let ring = |v: f64| {
        let q = c.eval(v) - r.origin;
        let z = q.dot(r.axis);
        (z, q.norm2() - z * z)
    };
    if dz.abs() > tol::ANGULAR {
        let at = |v: f64| (ring(v).0 - z0) / dz;
        let line2 = |t: f64| (a * t + b) * t + c0;
        let raw = |v: f64| line2(at(v)) - ring(v).1;
        let mut mm = |v: f64| line2(at(v)).max(0.0).sqrt() - ring(v).1.max(0.0).sqrt();
        let (vs, flat) = piece_roots(&pieces, 2, &raw, &mut mm);
        return Some(Hits {
            ts: if flat {
                Vec::new()
            } else {
                in_order(vs.into_iter().map(at).collect())
            },
            on_surface: flat,
        });
    }
    let level = |v: f64| ring(v).0 - z0;
    let (vs, flat) = piece_roots(&pieces, 1, &level, &mut |v| level(v));
    if flat {
        // The profile runs square to the axis at the line's height: a flat ring the line lies in.
        return Some(Hits {
            ts: Vec::new(),
            on_surface: true,
        });
    }
    let mut ts = Vec::new();
    for v in vs {
        let radius = ring(v).1.max(0.0).sqrt();
        // The line's nearest approach to the axis, and where.
        let (near, at) = ((c0 - b * b / (4.0 * a)).max(0.0).sqrt(), -b / (2.0 * a));
        if radius < near - tol::LINEAR {
            continue;
        }
        let half = ((radius * radius - near * near).max(0.0) / a).sqrt();
        if half <= tol::LINEAR {
            ts.push(at);
        } else {
            ts.extend([at - half, at + half]);
        }
    }
    Some(Hits {
        ts: in_order(ts),
        on_surface: false,
    })
}

/// A line against a ruled surface, whose rulings run from one curve to the other at each
/// parameter: the line meets a ruling where the two lines lie in one plane, the triple product of
/// the ruling, the line's direction and the step between them vanishing. Over each piece of both
/// curves that is a polynomial, of the sum of their degrees. `None` where either curve has no
/// pieces, or is a circle or an ellipse, whose angle is no polynomial in the other's parameter.
fn line_ruled(o: Point3, d: Vec3, r: &Ruled) -> Option<Hits> {
    let poly = |c: &Curve| matches!(c, Curve::Line(_) | Curve::BSpline(_) | Curve::Nurbs(_));
    if !poly(&r.a) || !poly(&r.b) {
        return None;
    }
    let (pa, pb) = (pieces(&r.a, r.a_range)?, pieces(&r.b, r.b_range)?);
    let on_a = |u: f64| r.a_range.0 + u * (r.a_range.1 - r.a_range.0);
    let on_b = |u: f64| r.b_range.0 + u * (r.b_range.1 - r.b_range.0);
    // Both curves' joins, as the surface's own parameter.
    let back = |t: f64, (lo, hi): (f64, f64)| (t - lo) / (hi - lo);
    let mut cuts: Vec<f64> = vec![0.0, 1.0];
    cuts.extend(pa.iter().map(|p| back((p.param)(0.0), r.a_range)));
    cuts.extend(pb.iter().map(|p| back((p.param)(0.0), r.b_range)));
    cuts.retain(|u| (0.0..=1.0).contains(u));
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|x, y| (*x - *y).abs() <= 1e-12);
    let weight = |c: &Curve, t: f64| match c {
        Curve::Nurbs(nc) => nc.weight(t),
        _ => 1.0,
    };
    let degree = pa[0].degree + pb[0].degree;
    let both: Vec<Piece> = cuts
        .windows(2)
        .map(|w| {
            let (lo, hi) = (w[0], w[1]);
            let param = move |s: f64| lo + (hi - lo) * s;
            Piece {
                param: Box::new(param),
                scale: Box::new(move |s| {
                    let u = param(s);
                    weight(&r.a, on_a(u)) * weight(&r.b, on_b(u))
                }),
                degree,
            }
        })
        .collect();
    // The ruling at `u`: its foot, and the step to its other end.
    let ruling = |u: f64| {
        let foot = r.a.eval(on_a(u));
        (foot, r.b.eval(on_b(u)) - foot)
    };
    let raw = |u: f64| {
        let (foot, step) = ruling(u);
        (foot - o).dot(step.cross(d))
    };
    // The distance between the line and the ruling's line: the triple product over the area the
    // two directions span.
    let mut mm = |u: f64| {
        let (foot, step) = ruling(u);
        let span = step.cross(d);
        let area = span.norm();
        if area > 0.0 {
            (foot - o).dot(span) / area
        } else {
            0.0
        }
    };
    let (us, flat) = piece_roots(&both, 1, &raw, &mut mm);
    if flat {
        return Some(Hits {
            ts: Vec::new(),
            on_surface: true,
        });
    }
    let ts = us
        .into_iter()
        .filter_map(|u| {
            let (foot, step) = ruling(u);
            let span = d.cross(step);
            // A ruling along the line meets it nowhere in particular.
            (span.norm2() > tol::ANGULAR * tol::ANGULAR * step.norm2())
                .then(|| (foot - o).cross(step).dot(span) / span.norm2())
        })
        .collect();
    Some(Hits {
        ts: in_order(ts),
        on_surface: false,
    })
}

/// The most patches of the tolerance's size [`line_spline`] finds along a line before it takes the
/// line to run in the surface: a touch or a pair of close crossings comes to a few such patches,
/// and a line that stays within the tolerance of the surface comes to as many as its length holds.
const SPLINE_RUN: usize = 64;

/// The most patches [`line_spline`] opens in all, a bound on its work whatever the surface.
const SPLINE_PATCHES: usize = 20_000;

/// A line against a spline surface, where no one parameter carries the problem.
///
/// The surface is cut into Bézier patches, each read in the line's own coordinates: its distance
/// from two planes through the line, and its place along the line, each a polynomial once
/// multiplied by the patch's weight. A patch lies among its control points, and so does each half
/// of a patch cut in two. So a patch whose control points all stand to one side of either plane,
/// or all beyond one end of the range, holds no crossing, and is dropped. Of what is left:
///
/// - A patch whose two distances cannot both vanish twice is solved outright. That is so where
///   their slopes, bounded by the steps between its control points, make the map from the
///   patch's parameters to the two distances one to one.
/// - Any other is halved, down to patches the size of the tolerance, which a touch or a pair of
///   crossings a hair apart comes to.
///
/// Every crossing lies in a patch that is solved or runs down to that size, so none is stepped
/// over however close two lie. `None` where the patches of that size run on along the line: a
/// line that lies in the surface, which has no crossings to find and is left to the caller.
fn line_spline(o: Point3, d: Vec3, ns: &NurbsSurface, range: (f64, f64)) -> Option<Hits> {
    // Two directions square to the line and to each other.
    let side = if d.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = d.cross(side).unit()?;
    let e2 = d.cross(e1);
    // A control point as its distance from each plane and its place along the line, all times
    // its weight, and the weight.
    let own = |q: [f64; 4]| -> [f64; 4] {
        let rel = Vec3::new(q[0], q[1], q[2]) - o * q[3];
        [rel.dot(e1), rel.dot(e2), rel.dot(d), q[3]]
    };
    // Whether every control point stands clear of zero in coordinate `c` on one side, or beyond
    // an end of the range along the line: the patch then holds no crossing in the range.
    let clear = |patch: &Bezier| {
        let pts = patch.points();
        let beyond = |f: &dyn Fn(&[f64; 4]) -> f64| pts.iter().all(|q| f(q) > 0.0);
        (0..2).any(|c| {
            beyond(&|q| q[c] - tol::LINEAR * q[3]) || beyond(&|q| -q[c] - tol::LINEAR * q[3])
        }) || beyond(&|q| q[2] - (range.1 + tol::LINEAR) * q[3])
            || beyond(&|q| (range.0 - tol::LINEAR) * q[3] - q[2])
    };
    let mut open: Vec<Bezier> = ns
        .beziers()
        .iter()
        .filter(|b| line_passes(o, d, &b.bbox(), range))
        .map(|b| b.mapped(own))
        .filter(|b| !clear(b))
        .collect();
    let mut ts: Vec<f64> = Vec::new();
    // Each patch of the tolerance's size the line passes: its stretch of the line, and where on
    // the surface it lies.
    let mut small: Vec<(f64, f64, Vec2)> = Vec::new();
    let mut opened = 0;
    while let Some(patch) = open.pop() {
        opened += 1;
        if opened > SPLINE_PATCHES {
            return None;
        }
        if let Some(t) = only_crossing(&patch) {
            if t >= range.0 - tol::LINEAR && t <= range.1 + tol::LINEAR {
                ts.push(t);
            }
            continue;
        }
        let bb = patch.bbox();
        let span = patch.hi - patch.lo;
        let spent = span.x.abs().max(span.y.abs()) <= 1e-14;
        if bb.diagonal() <= tol::LINEAR || spent {
            small.push((bb.min.z, bb.max.z, (patch.lo + patch.hi) * 0.5));
            if small.len() > SPLINE_RUN {
                return None;
            }
            continue;
        }
        open.extend(
            patch
                .halves(patch.longer_in_u())
                .into_iter()
                .filter(|h| !clear(h)),
        );
    }
    small.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut stretches: Vec<(f64, f64, Vec2)> = Vec::new();
    for (a, b, uv) in small {
        match stretches.last_mut() {
            Some(last) if a <= last.1 + tol::LINEAR => last.1 = last.1.max(b),
            _ => stretches.push((a, b, uv)),
        }
    }
    for (a, b, hint) in stretches {
        // The line's height over the surface near the patch: positive on the side of its normal.
        let mut last = hint;
        let mut height = |t: f64| {
            let q = o + d * t;
            let uv = ns.project_near(q, last);
            last = uv;
            let (at, su, sv) = ns.d1(uv.x, uv.y);
            match su.cross(sv).unit() {
                Some(n) => (q - at).dot(n),
                // At a pole no normal is defined: the distance, which is zero on the surface.
                None => q.dist(at),
            }
        };
        let (a, b) = (
            (a - tol::LINEAR).max(range.0),
            (b + tol::LINEAR).min(range.1),
        );
        let steps = (((b - a) / tol::LINEAR).ceil() as usize).clamp(2, 32);
        ts.extend(sampled_roots(&mut height, &[a, b], steps).0);
    }
    // A crossing on the side two patches share is found from both, a rounding apart.
    ts.sort_by(f64::total_cmp);
    ts.dedup_by(|a, b| (*a - *b).abs() <= 1e-3 * tol::LINEAR);
    Some(Hits {
        ts,
        on_surface: false,
    })
}

/// Whether the line `o + t·d` passes within the tolerance of the box for some `t` in the range.
fn line_passes(o: Point3, d: Vec3, bb: &Aabb, range: (f64, f64)) -> bool {
    let (mut lo, mut hi) = range;
    for k in 0..3 {
        let (mn, mx) = (bb.min[k] - tol::LINEAR, bb.max[k] + tol::LINEAR);
        if d[k].abs() < 1e-300 {
            if o[k] < mn || o[k] > mx {
                return false;
            }
            continue;
        }
        let (a, b) = ((mn - o[k]) / d[k], (mx - o[k]) / d[k]);
        lo = lo.max(a.min(b));
        hi = hi.min(a.max(b));
    }
    hi >= lo
}

/// Where the line crosses a patch read in the line's coordinates (its distances from two planes
/// through the line, its place along the line, its weight), when the patch can hold one crossing
/// at most and holds one. `None` where it may hold more, or holds none.
///
/// The two distances, as functions of the patch's parameters, cannot both vanish twice where that
/// map is one to one, and it is where its matrix of slopes, turned by a fixed matrix, keeps a
/// positive definite symmetric part: then two points of the patch always map apart. The slopes
/// are bounded by the steps between control points, so the bound holds all over the patch.
fn only_crossing(patch: &Bezier) -> Option<f64> {
    let slopes = [patch.slopes(0), patch.slopes(1)];
    let mid = |r: (f64, f64)| 0.5 * (r.0 + r.1);
    // The slopes at their middle, and the turn that makes them the identity there.
    let m = [
        [mid(slopes[0][0]), mid(slopes[0][1])],
        [mid(slopes[1][0]), mid(slopes[1][1])],
    ];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let size = m[0][0].abs().max(m[0][1].abs()) * m[1][0].abs().max(m[1][1].abs());
    if det.abs() <= 1e-9 * size || size == 0.0 {
        return None;
    }
    let turn = [
        [m[1][1] / det, -m[0][1] / det],
        [-m[1][0] / det, m[0][0] / det],
    ];
    // The range of `x·a + y·b` for `a` and `b` each in a range.
    let mixed = |x: f64, a: (f64, f64), y: f64, b: (f64, f64)| {
        let part = |k: f64, r: (f64, f64)| (k * r.0).min(k * r.1)..=(k * r.0).max(k * r.1);
        let (p, q) = (part(x, a), part(y, b));
        (p.start() + q.start(), p.end() + q.end())
    };
    let turned =
        |row: usize, col: usize| mixed(turn[row][0], slopes[0][col], turn[row][1], slopes[1][col]);
    let (a, b, c, e) = (turned(0, 0), turned(0, 1), turned(1, 0), turned(1, 1));
    let off = b.0.abs().max(b.1.abs()) + c.0.abs().max(c.1.abs());
    if !(a.0 > 0.0 && e.0 > 0.0 && off * off < 4.0 * a.0 * e.0) {
        return None;
    }
    // One crossing at most: Newton's method from the middle finds it, or it lies outside.
    let (mut s, mut t) = (0.5, 0.5);
    for _ in 0..40 {
        let [q, qs, qt] = patch.at(s, t);
        let (ds, dt) = numeric::solve2(qs[0], qt[0], qs[1], qt[1], -q[0], -q[1])?;
        (s, t) = (s + ds, t + dt);
        if !(-0.5..=1.5).contains(&s) || !(-0.5..=1.5).contains(&t) {
            return None;
        }
        if ds.abs().max(dt.abs()) <= 1e-14 {
            break;
        }
    }
    let inside = |x: f64| (-1e-9..=1.0 + 1e-9).contains(&x);
    if !inside(s) || !inside(t) {
        return None;
    }
    let [q, ..] = patch.at(s.clamp(0.0, 1.0), t.clamp(0.0, 1.0));
    // On the line within rounding, or the search settled short of it.
    let off = q[0].hypot(q[1]) / q[3];
    (off <= 1e-3 * tol::LINEAR * (1.0 + (q[2] / q[3]).abs())).then_some(q[2] / q[3])
}

/// A line against a torus: the torus is the zero set of the quartic
/// `(|p|² + R² − r²)² − 4R²ρ²` along the line, so the quartic's stationary points split the line
/// into stretches with one crossing at most and mark every place it can touch. That holds for a
/// tube that reaches its axis as for a ring: the quartic is the product of the two sheets' own.
fn line_torus(o: Point3, d: Vec3, t: &Torus, range: (f64, f64)) -> Hits {
    let (lo, ld) = (t.frame.to_local(o), t.frame.dir_to_local(d));
    let (big, small) = (t.major, t.minor);
    // Only where the line is within the torus's bounding sphere.
    let (b, c) = (lo.dot(ld), lo.dot(lo) - (big + small).powi(2));
    let disc = b * b - c;
    if disc < 0.0 {
        return Hits::default();
    }
    let (t0, t1) = (
        (-b - disc.sqrt()).max(range.0),
        (-b + disc.sqrt()).min(range.1),
    );
    if t0 > t1 {
        return Hits::default();
    }
    let k = big * big - small * small;
    let sq = [lo.dot(lo) + k, 2.0 * b, 1.0];
    let rho = [
        lo.x * lo.x + lo.y * lo.y,
        2.0 * (lo.x * ld.x + lo.y * ld.y),
        ld.x * ld.x + ld.y * ld.y,
    ];
    let mut q = [0.0; 5];
    for i in 0..3 {
        for j in 0..3 {
            q[i + j] += sq[i] * sq[j];
        }
        q[i] -= 4.0 * big * big * rho[i];
    }
    let mut knots = vec![t0];
    knots.extend(stationary_points(&q, t0, t1));
    knots.push(t1);
    // The distance to the nearer of the surface's two sheets, with the quartic's sign: the tube
    // about the circle on the near side of the axis and, where the tube reaches past the axis, its
    // far side come round. A ring torus has the first alone; the second lies always outside it.
    let mut f = |s: f64| {
        let p = lo + ld * s;
        let rho = p.x.hypot(p.y);
        let (near, far) = (
            (rho - big).hypot(p.z) - small,
            (rho + big).hypot(p.z) - small,
        );
        if near.abs() <= far.abs() {
            near * far.signum()
        } else {
            far * near.signum()
        }
    };
    Hits {
        ts: roots_between(&mut f, &knots),
        on_surface: false,
    }
}

/// A circle against a sphere meets it where it meets the sphere's section in the circle's plane.
fn circle_sphere(ci: &Circle, sp: &crate::geom::Sphere, range: (f64, f64)) -> Hits {
    let l = ci.frame.to_local(sp.frame.origin);
    let rho2 = (sp.radius - l.z.abs()) * (sp.radius + l.z.abs());
    if rho2 < 0.0 {
        // The plane misses the sphere; within the tolerance it touches it at the foot.
        let gap = l.z.abs() - sp.radius;
        let d = l.x.hypot(l.y);
        return if gap <= tol::LINEAR && (d - ci.radius).abs() <= tol::LINEAR {
            Hits {
                ts: angle_in(l.y.atan2(l.x), range).into_iter().collect(),
                on_surface: false,
            }
        } else {
            Hits::default()
        };
    }
    coplanar_circles(ci, Vec2::new(l.x, l.y), rho2.sqrt(), range)
}

/// The curve's plane, when it is a planar conic.
fn conic_plane(c: &Curve) -> Option<crate::math::Frame> {
    match c {
        Curve::Circle(ci) => Some(ci.frame),
        Curve::Ellipse(e) => Some(e.frame),
        _ => None,
    }
}

/// The distance from `p` to the curve over `range`, and the parameter of the nearest point.
pub(crate) fn curve_distance(c: &Curve, range: (f64, f64), p: Point3) -> (f64, f64) {
    let t = c.project(p, range.0, range.1);
    (c.eval(t).dist(p), t)
}

/// Whether `p` lies within `reach` of the curve over `range`, as [`curve_distance`] measures it:
/// a point farther than that from every piece's hull box is farther from the curve, unprojected.
pub(crate) fn within(c: &Curve, range: (f64, f64), p: Point3, reach: f64) -> bool {
    if let Some(pieces) = hulls::pieces(c, range) {
        let mut bb = Aabb::EMPTY;
        bb.add(p);
        if pieces.iter().all(|q| hulls::gap(&q.2, &bb) > reach) {
            return false;
        }
    }
    curve_distance(c, range, p).0 <= reach
}

/// Two vertices closer than this, mm, along curves running nearly together are one meeting.
pub(crate) const GRAZE_GAP: f64 = 1e-5;

/// Where `c1` crosses `c2` while the two run within [`GRAZE_GAP`] of each other: a local least of
/// the gap over 64 samples, settled by golden-section search, where the offset from one to the
/// other turns round. Two splines a hair apart cross at angles of 1e-7 rad, where a general step
/// loses them; a touch without a crossing is left alone.
pub(crate) fn grazing(c1: &Curve, r1: (f64, f64), c2: &Curve, r2: (f64, f64)) -> Vec<Point3> {
    const SAMPLES: usize = 64;
    let gap = |t: f64| curve_distance(c2, r2, c1.eval(t)).0;
    let ts: Vec<f64> = (0..=SAMPLES)
        .map(|k| r1.0 + (r1.1 - r1.0) * k as f64 / SAMPLES as f64)
        .collect();
    // A sample on no piece of `c1` near `c2` is farther than the graze from it, and so never
    // starts a search, nor lets a neighbour start one.
    let reach = hulls::near(c1, r1, c2, r2, GRAZE_GAP + 2.0 * tol::LINEAR);
    if reach.as_ref().is_some_and(Vec::is_empty) {
        return Vec::new();
    }
    let ds: Vec<f64> = ts
        .iter()
        .map(|&t| match &reach {
            Some(r) if !r.iter().any(|&(lo, hi)| lo <= t && t <= hi) => f64::INFINITY,
            _ => gap(t),
        })
        .collect();
    let mut out = Vec::new();
    for k in 1..SAMPLES {
        // Both sides apart by more than rounding, so the offset's turn is a crossing: curves that
        // coincide (a symmetric solid turned onto itself) have offsets that are rounding alone.
        let (lo, hi) = (ds[k - 1].min(ds[k + 1]), ds[k - 1].max(ds[k + 1]));
        if ds[k] > ds[k - 1] || ds[k] > ds[k + 1] || hi > GRAZE_GAP || lo <= 1e-3 * tol::LINEAR {
            continue;
        }
        let (mut a, mut b) = (ts[k - 1], ts[k + 1]);
        let g = 0.5 * (5f64.sqrt() - 1.0);
        for _ in 0..100 {
            let (x1, x2) = (b - g * (b - a), a + g * (b - a));
            if gap(x1) <= gap(x2) {
                b = x2;
            } else {
                a = x1;
            }
        }
        let t = 0.5 * (a + b);
        let offset = |s: f64| {
            let p = c1.eval(s);
            c2.eval(curve_distance(c2, r2, p).1) - p
        };
        let crosses = offset(ts[k - 1]).dot(offset(ts[k + 1])) < 0.0;
        if gap(t) <= 0.5 * tol::LINEAR && crosses {
            out.push(c1.eval(t));
        }
    }
    out
}

/// The points where two curves meet, transversally or touching; nothing where they overlap, since
/// an overlap's ends are vertices of one curve or the other.
pub(crate) fn curve_curve(c1: &Curve, r1: (f64, f64), c2: &Curve, r2: (f64, f64)) -> Vec<Point3> {
    let on2 = |p: Point3| within(c2, r2, p, tol::LINEAR);
    let on1 = |p: Point3| within(c1, r1, p, tol::LINEAR);
    match (c1, c2) {
        (Curve::Line(a), Curve::Line(b)) => {
            let w = a.origin - b.origin;
            let c = a.dir.dot(b.dir);
            let den = 1.0 - c * c;
            if den.sqrt() <= tol::ANGULAR {
                return Vec::new();
            }
            let (d, e) = (a.dir.dot(w), b.dir.dot(w));
            let s = (c * e - d) / den;
            let t = (e - c * d) / den;
            let (p, q) = (a.origin + a.dir * s, b.origin + b.dir * t);
            if p.dist(q) <= tol::LINEAR
                && in_range(c1, s, r1).is_some()
                && in_range(c2, t, r2).is_some()
            {
                vec![p.lerp(q, 0.5)]
            } else {
                Vec::new()
            }
        }
        (Curve::Line(_), _) if conic_plane(c2).is_some() => line_conic(c1, r1, c2, r2),
        (_, Curve::Line(_)) if conic_plane(c1).is_some() => line_conic(c2, r2, c1, r1),
        (Curve::Circle(a), Curve::Circle(b))
            if a.frame.z.cross(b.frame.z).norm() <= tol::ANGULAR
                && (b.frame.origin - a.frame.origin).dot(a.frame.z).abs() <= tol::LINEAR =>
        {
            let l = a.frame.to_local(b.frame.origin);
            let d = l.x.hypot(l.y);
            let (ra, rb) = (a.radius, b.radius);
            if d <= tol::LINEAR {
                return Vec::new();
            }
            if d > ra + rb + tol::LINEAR || d < (ra - rb).abs() - tol::LINEAR {
                return Vec::new();
            }
            let ex = Vec3::new(l.x / d, l.y / d, 0.0);
            let ey = Vec3::new(-ex.y, ex.x, 0.0);
            let x = (d * d + ra * ra - rb * rb) / (2.0 * d);
            let touching =
                (d - (ra + rb)).abs() <= tol::LINEAR || (d - (ra - rb).abs()).abs() <= tol::LINEAR;
            let pts: Vec<Vec3> = if touching {
                vec![ex * x.clamp(-ra, ra)]
            } else {
                let y = (ra * ra - x * x).max(0.0).sqrt();
                vec![ex * x + ey * y, ex * x - ey * y]
            };
            pts.into_iter()
                .map(|q| a.frame.to_world(q))
                .filter(|&p| on1(p) && on2(p))
                .collect()
        }
        _ => {
            if let Some(f) = conic_plane(c2) {
                let plane = Surface::Plane(crate::geom::Plane { frame: f });
                let h = curve_surface(c1, r1, &plane);
                if !h.on_surface {
                    return h
                        .ts
                        .iter()
                        .map(|&t| c1.eval(t))
                        .filter(|&p| on2(p))
                        .collect();
                }
            }
            if let Some(f) = conic_plane(c1) {
                let plane = Surface::Plane(crate::geom::Plane { frame: f });
                let h = curve_surface(c2, r2, &plane);
                if !h.on_surface {
                    return h
                        .ts
                        .iter()
                        .map(|&t| c2.eval(t))
                        .filter(|&p| on1(p))
                        .collect();
                }
            }
            generic_curve_curve(c1, r1, c2, r2)
        }
    }
}

/// Where curve `c` crosses curve `e` near `e`'s point at `t` near `p`, by Gauss–Newton steps on
/// both parameters from there: the point between them, `e`'s parameter, and the sine of the angle
/// they cross at, where they pass within the tolerance of each other; the point is moved along `c`
/// onto `through` where `c` crosses it there. `None` where the curves run together.
pub(crate) fn curve_curve_near(
    c: &Curve,
    rc: (f64, f64),
    e: &Curve,
    re: (f64, f64),
    p: Point3,
    t: f64,
    through: Option<&Surface>,
) -> Option<(Point3, f64, f64)> {
    let (mut s, mut t) = (c.project(p, rc.0, rc.1), t);
    for _ in 0..40 {
        let ((a, da), (b, db)) = (c.d1(s), e.d1(t));
        let r = a - b;
        let (aa, ab, bb) = (da.dot(da), -da.dot(db), db.dot(db));
        let (ds, dt) = numeric::solve2(aa, ab, ab, bb, -da.dot(r), db.dot(r))?;
        let (ns, nt) = ((s + ds).clamp(rc.0, rc.1), (t + dt).clamp(re.0, re.1));
        let still =
            (ns - s).abs() <= 1e-15 * (1.0 + s.abs()) && (nt - t).abs() <= 1e-15 * (1.0 + t.abs());
        (s, t) = (ns, nt);
        if still {
            break;
        }
    }
    let (a, b) = (c.eval(s), e.eval(t));
    // Nearly parallel curves meet along a stretch, not at a point.
    let (da, db) = (c.d1(s).1.unit_or_zero(), e.d1(t).1.unit_or_zero());
    let sine = da.cross(db).norm();
    if a.dist(b) > tol::LINEAR || sine <= tol::TANGENT {
        return None;
    }
    // Moved along `c` onto `through` as well, the point lies on every surface of the meeting, so
    // within each fitted curve's own error of every curve through it.
    let on = through.and_then(|surf| {
        let s = onto_surface(c, rc, s, surf)?;
        let q = c.eval(s);
        let (d, tq) = curve_distance(e, re, q);
        (d <= tol::LINEAR).then_some((q, tq))
    });
    Some(match on {
        Some((q, tq)) => (q, tq, sine),
        None => (a.lerp(b, 0.5), t, sine),
    })
}

/// The point near `p` on all three surfaces, by Newton steps on their tangent planes; `None` where
/// two of them run together there, so the point is not fixed by them.
pub(crate) fn on_three(surfs: [&Surface; 3], p: Point3) -> Option<Point3> {
    let planes = |p: Point3| {
        surfs.map(|s| {
            let uv = s.project(p);
            let n = s.normal(uv.x, uv.y);
            (n, (p - s.eval(uv.x, uv.y)).dot(n))
        })
    };
    let mut p = p;
    for _ in 0..30 {
        let [(n0, d0), (n1, d1), (n2, d2)] = planes(p);
        let det = n0.dot(n1.cross(n2));
        if det.abs() <= tol::TANGENT {
            return None;
        }
        let step = (n1.cross(n2) * -d0 + n2.cross(n0) * -d1 + n0.cross(n1) * -d2) * (1.0 / det);
        p += step;
        if step.norm() <= 1e-15 * (1.0 + p.norm()) {
            break;
        }
    }
    planes(p)
        .iter()
        .all(|&(_, d)| d.abs() <= 1e-3 * tol::LINEAR)
        .then_some(p)
}

/// Where curve `c` crosses surface `surf` near parameter `s0`, by Newton steps along it; `None`
/// where it does not cross there steeply enough to converge.
fn onto_surface(c: &Curve, rc: (f64, f64), s0: f64, surf: &Surface) -> Option<f64> {
    let mut s = s0;
    for _ in 0..30 {
        let (p, dp) = c.d1(s);
        let uv = surf.project(p);
        let slope = surf.normal(uv.x, uv.y).dot(dp);
        if slope.abs() <= tol::TANGENT * dp.norm() {
            return None;
        }
        let next = (s - signed_distance(surf, p) / slope).clamp(rc.0, rc.1);
        let still = (next - s).abs() <= 1e-15 * (1.0 + s.abs());
        s = next;
        if still {
            break;
        }
    }
    (signed_distance(surf, c.eval(s)).abs() <= 1e-3 * tol::LINEAR).then_some(s)
}

fn line_conic(line: &Curve, rl: (f64, f64), conic: &Curve, rc: (f64, f64)) -> Vec<Point3> {
    let Curve::Line(l) = line else {
        return Vec::new();
    };
    let frame = conic_plane(conic).expect("conic has a plane");
    let in_plane = l.dir.dot(frame.z).abs() <= tol::ANGULAR
        && (l.origin - frame.origin).dot(frame.z).abs() <= tol::LINEAR;
    let on_conic = |p: Point3| curve_distance(conic, rc, p).0 <= tol::LINEAR;
    if !in_plane {
        let plane = Surface::Plane(crate::geom::Plane { frame });
        return curve_surface(line, rl, &plane)
            .ts
            .into_iter()
            .map(|t| line.eval(t))
            .filter(|&p| on_conic(p))
            .collect();
    }
    let ts = match conic {
        Curve::Circle(ci) => {
            line_cylinder(
                l.origin,
                l.dir,
                &Cylinder {
                    frame: ci.frame,
                    radius: ci.radius,
                },
            )
            .ts
        }
        _ => {
            let Curve::Ellipse(e) = conic else {
                return Vec::new();
            };
            // In the ellipse's frame scaled to a unit circle, the line stays a line.
            let lo = e.frame.to_local(l.origin);
            let ld = e.frame.dir_to_local(l.dir);
            let (ox, oy) = (lo.x / e.major, lo.y / e.minor);
            let (dx, dy) = (ld.x / e.major, ld.y / e.minor);
            let a = dx * dx + dy * dy;
            let t_near = -(ox * dx + oy * dy) / a;
            let disc = 1.0 - ((ox + dx * t_near).powi(2) + (oy + dy * t_near).powi(2));
            let near = line.eval(t_near);
            if on_conic(near) {
                vec![t_near]
            } else if disc > 0.0 {
                let h = (disc / a).sqrt();
                vec![t_near - h, t_near + h]
            } else {
                Vec::new()
            }
        }
    };
    ts.into_iter()
        .filter_map(|t| in_range(line, t, rl))
        .map(|t| line.eval(t))
        .filter(|&p| on_conic(p))
        .collect()
}

/// A curve's range cut into spans, each with a box that holds it, for nearest-point queries that
/// only project onto the spans that can hold the nearest point.
struct CurveIndex<'a> {
    curve: &'a Curve,
    spans: Vec<(f64, f64, crate::math::Aabb)>,
}

impl<'a> CurveIndex<'a> {
    fn new(curve: &'a Curve, range: (f64, f64)) -> CurveIndex<'a> {
        let mut cuts = vec![range.0];
        cuts.extend(curve.breakpoints(range.0, range.1));
        cuts.push(range.1);
        if cuts.len() < 9 {
            cuts = (0..=16)
                .map(|i| range.0 + (range.1 - range.0) * i as f64 / 16.0)
                .collect();
        }
        let spans = cuts
            .windows(2)
            .map(|w| {
                let n = 8;
                let pts: Vec<Point3> = (0..=n)
                    .map(|i| curve.eval(w[0] + (w[1] - w[0]) * i as f64 / n as f64))
                    .collect();
                let mut bb = crate::math::Aabb::EMPTY;
                let mut chord: f64 = 0.0;
                for (i, &p) in pts.iter().enumerate() {
                    bb.add(p);
                    if i > 0 {
                        chord = chord.max(p.dist(pts[i - 1]));
                    }
                }
                // Between samples the curve strays from them by less than a chord.
                let pad = Vec3::new(chord, chord, chord);
                bb.min -= pad;
                bb.max += pad;
                (w[0], w[1], bb)
            })
            .collect();
        CurveIndex { curve, spans }
    }

    /// As [`curve_distance`].
    fn distance(&self, p: Point3) -> (f64, f64) {
        let gap = |bb: &crate::math::Aabb| {
            let d = Vec3::new(
                (bb.min.x - p.x).max(p.x - bb.max.x).max(0.0),
                (bb.min.y - p.y).max(p.y - bb.max.y).max(0.0),
                (bb.min.z - p.z).max(p.z - bb.max.z).max(0.0),
            );
            d.norm()
        };
        let gaps: Vec<f64> = self.spans.iter().map(|s| gap(&s.2)).collect();
        // The search stops at the first span whose gap passes the best distance, and the span of
        // least gap comes first, so no span whose gap passes that span's distance is reached.
        let first = (0..gaps.len())
            .min_by(|&a, &b| gaps[a].total_cmp(&gaps[b]))
            .expect("an index has spans");
        let (a, b, _) = self.spans[first];
        let bound = curve_distance(self.curve, (a, b), p).0;
        let mut order: Vec<(f64, usize)> = gaps
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g <= bound)
            .map(|(i, &g)| (g, i))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best = (f64::INFINITY, self.spans[0].0);
        for (g, i) in order {
            if g > best.0 {
                break;
            }
            let (a, b, _) = self.spans[i];
            let r = curve_distance(self.curve, (a, b), p);
            if r.0 < best.0 {
                best = r;
            }
        }
        best
    }
}

fn generic_curve_curve(c1: &Curve, r1: (f64, f64), c2: &Curve, r2: (f64, f64)) -> Vec<Point3> {
    sampled_curve_curve(c1, r1, c2, r2, true)
}

/// The search behind [`generic_curve_curve`]: the distance to `c2` sampled along `c1`, each local
/// least settled by golden-section search. With `prune`, the distance is only sampled about the
/// pieces of `c1` whose hull boxes come within the tolerance of a hull box of `c2`'s: a least
/// anywhere else is farther than the tolerance and found nothing, so the answer is the same.
fn sampled_curve_curve(
    c1: &Curve,
    r1: (f64, f64),
    c2: &Curve,
    r2: (f64, f64),
    prune: bool,
) -> Vec<Point3> {
    const N: usize = 16;
    let mut cuts = vec![r1.0];
    cuts.extend(c1.breakpoints(r1.0, r1.1));
    cuts.push(r1.1);
    let mut ts: Vec<f64> = Vec::with_capacity(N * (cuts.len() - 1) + 1);
    for w in cuts.windows(2) {
        ts.extend((0..N).map(|i| w[0] + (w[1] - w[0]) * i as f64 / N as f64));
    }
    ts.push(cuts[cuts.len() - 1]);
    let n = ts.len();
    // Where the search may find a meeting: every sample whose neighbours' span reaches a piece of
    // `c1` near `c2`, or every sample where the curves have no hulls.
    let near = |reach: &[(f64, f64)]| {
        let mut k = 0;
        let mut want = vec![false; n];
        for &(lo, hi) in reach {
            while k < n && ts[(k + 1).min(n - 1)] < lo {
                k += 1;
            }
            let mut j = k;
            while j < n && ts[j.saturating_sub(1)] <= hi {
                want[j] = true;
                j += 1;
            }
        }
        want
    };
    // Twice the tolerance, so rounding in the boxes can never drop a meeting.
    let reach = prune
        .then(|| hulls::near(c1, r1, c2, r2, 2.0 * tol::LINEAR))
        .flatten();
    let wanted = match reach {
        Some(r) if r.is_empty() => return Vec::new(),
        Some(r) => near(&r),
        None => vec![true; n],
    };
    let index = CurveIndex::new(c2, r2);
    let mut samples: Vec<Option<f64>> = vec![None; n];
    for k in (0..n).filter(|&k| wanted[k]) {
        for j in k.saturating_sub(1)..(k + 2).min(n) {
            if samples[j].is_none() {
                samples[j] = Some(index.distance(c1.eval(ts[j])).0);
            }
        }
    }
    let value = |k: usize| {
        (
            ts[k],
            samples[k].expect("a wanted sample's neighbours are sampled"),
        )
    };
    let mut out: Vec<Point3> = Vec::new();
    for k in (0..n).filter(|&k| wanted[k]) {
        let at = value(k);
        let lo = if k > 0 { value(k - 1) } else { at };
        let hi = if k + 1 < n { value(k + 1) } else { at };
        if at.1 > lo.1 || at.1 > hi.1 {
            continue;
        }
        // Inside a stretch where the curves run together: the stretch's ends are vertices.
        if (k > 0 && lo.1 <= tol::LINEAR) || (k + 1 < n && hi.1 <= tol::LINEAR) {
            continue;
        }
        let t = minimise(&mut |t: f64| index.distance(c1.eval(t)).0, lo.0, hi.0);
        let p = c1.eval(t);
        let (d, s) = index.distance(p);
        if d <= tol::LINEAR {
            let q = p.lerp(c2.eval(s), 0.5);
            if !out.iter().any(|o| o.dist(q) <= tol::LINEAR) {
                out.push(q);
            }
        }
    }
    out
}

/// Whether [`ray_surface`] finds every crossing of a line with `s` from a closed form.
pub(crate) fn ray_surface_is_closed_form(s: &Surface) -> bool {
    match s {
        Surface::Plane(_) | Surface::Cylinder(_) | Surface::Sphere(_) => true,
        Surface::Cone(cone) => cone.half_angle.cos().abs() > 1e-9,
        _ => false,
    }
}

/// Whether [`line_crossings`] can give the crossings of a line with `s`.
pub(crate) fn line_crossings_known(s: &Surface) -> bool {
    ray_surface_is_closed_form(s) || matches!(s, Surface::Torus(_) | Surface::Nurbs(_))
}

/// Where the line `p + t·d` (`d` unit) crosses `s` for `t` in `range`, and for a surface with a
/// closed form anywhere along the line: from the closed form, a torus's quartic, or the search of
/// [`line_spline`], none of which steps over a crossing. `None` for any other surface, and where
/// the line lies in the surface or the search gives up.
pub(crate) fn line_crossings(
    s: &Surface,
    p: Point3,
    d: Vec3,
    range: (f64, f64),
) -> Option<Vec<f64>> {
    let hits = match s {
        _ if ray_surface_is_closed_form(s) => return Some(ray_surface(s, p, d, range.1)),
        Surface::Torus(t) => line_torus(p, d, t, range),
        Surface::Nurbs(ns) => line_spline(p, d, ns, range)?,
        _ => return None,
    };
    (!hits.on_surface).then_some(hits.ts)
}

/// The parameters `t > 0` where the ray `p + t·d` (`d` unit) crosses the surface, up to `t_max`.
pub(crate) fn ray_surface(s: &Surface, p: Point3, d: Vec3, t_max: f64) -> Vec<f64> {
    let quad = |a: f64, b: f64, c: f64| -> Vec<f64> {
        let disc = b * b - 4.0 * a * c;
        if a.abs() < 1e-300 || disc < 0.0 {
            return if a.abs() < 1e-300 && b.abs() > 1e-300 {
                vec![-c / b]
            } else {
                Vec::new()
            };
        }
        let q = -0.5 * (b + b.signum() * disc.sqrt());
        let mut v = vec![q / a];
        if q.abs() > 1e-300 {
            v.push(c / q);
        }
        v
    };
    match s {
        Surface::Plane(pl) => {
            let dn = d.dot(pl.frame.z);
            if dn.abs() < 1e-300 {
                Vec::new()
            } else {
                vec![(pl.frame.origin - p).dot(pl.frame.z) / dn]
            }
        }
        Surface::Cylinder(c) => {
            let lo = c.frame.to_local(p);
            let ld = c.frame.dir_to_local(d);
            quad(
                ld.x * ld.x + ld.y * ld.y,
                2.0 * (lo.x * ld.x + lo.y * ld.y),
                lo.x * lo.x + lo.y * lo.y - c.radius * c.radius,
            )
        }
        Surface::Sphere(sp) => {
            let oc = p - sp.frame.origin;
            quad(1.0, 2.0 * oc.dot(d), oc.norm2() - sp.radius * sp.radius)
        }
        Surface::Cone(cone) if cone.half_angle.cos().abs() > 1e-9 => {
            let lo = cone.frame.to_local(p);
            let ld = cone.frame.dir_to_local(d);
            let k = cone.half_angle.tan();
            let r0 = cone.radius + k * lo.z;
            quad(
                ld.x * ld.x + ld.y * ld.y - k * k * ld.z * ld.z,
                2.0 * (lo.x * ld.x + lo.y * ld.y - k * r0 * ld.z),
                lo.x * lo.x + lo.y * lo.y - r0 * r0,
            )
        }
        // Every other surface by the routine the booleans use: exact where a closed form or a
        // search that cannot step over a crossing exists, sampled only where none does.
        _ => {
            let line = Curve::Line(Line { origin: p, dir: d });
            curve_surface(&line, (0.0, t_max), s).ts
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geom::{Circle, Line, Plane},
        math::Frame,
    };

    fn line(o: [f64; 3], d: [f64; 3]) -> Curve {
        Curve::Line(Line {
            origin: Vec3::from_array(o),
            dir: Vec3::from_array(d).unit().unwrap(),
        })
    }

    /// A closed cubic whose right side ripples across the plane x = 5 every hundredth of a
    /// millimetre, extruded along z: the small features of a gear's root in outline.
    fn rippled_prism() -> (crate::geom::BSpline, Surface) {
        let mut ctrl: Vec<Vec3> = Vec::new();
        // Up the right side in ripples, then back round the left in a wide arc.
        for k in 0..16 {
            let x = 5.0 + if k % 2 == 0 { 0.004 } else { -0.004 };
            ctrl.push(Vec3::new(x, -0.075 + 0.01 * k as f64, 0.0));
        }
        for k in 1..8 {
            let a = TAU / 4.0 + TAU / 2.0 * k as f64 / 8.0;
            ctrl.push(Vec3::new(5.0 + 4.0 * a.cos(), 4.0 * a.sin(), 0.0));
        }
        let n = ctrl.len();
        ctrl.extend_from_within(..3);
        let knots = (0..n + 7).map(|i| i as f64 - 3.0).collect();
        let curve = crate::geom::BSpline::periodic_cubic(knots, ctrl).unwrap();
        let surface = Surface::Extrusion(Box::new(crate::geom::Extrusion {
            curve: Curve::BSpline(curve.clone()),
            dir: Vec3::Z,
        }));
        (curve, surface)
    }

    #[test]
    fn a_line_along_an_extrusions_ripples_meets_it_at_every_crossing() {
        let (curve, surface) = rippled_prism();
        let (u0, u1) = curve.domain();
        let steps = 400_000;
        let at = |i: usize| u0 + (u1 - u0) * i as f64 / steps as f64;
        for shift in [0.0, 0.001, -0.001] {
            // A line in the plane x = 5 + shift, climbing as it goes: it runs along the ripples
            // and crosses the profile at each.
            let o = Vec3::new(5.0 + shift, -20.0, -3.0);
            let d = Vec3::new(0.0, 1.0, 0.2).unit().unwrap();
            let l = Curve::Line(Line { origin: o, dir: d });
            // Every crossing, from the profile itself: where it passes through the line's plane.
            let side = |u: f64| curve.eval(u).x - (5.0 + shift);
            let mut want = Vec::new();
            for i in 0..steps {
                let (a, b) = (side(at(i)), side(at(i + 1)));
                if (a < 0.0) != (b < 0.0) {
                    let u = numeric::bisect(&mut |u| side(u), at(i), at(i + 1));
                    want.push((curve.eval(u) - o).y / d.y);
                }
            }
            want.sort_by(f64::total_cmp);
            assert!(want.len() >= 10, "shift {shift}: {} crossings", want.len());
            let got = curve_surface(&l, (-100.0, 100.0), &surface);
            assert!(!got.on_surface);
            let mut ts = got.ts.clone();
            ts.sort_by(f64::total_cmp);
            assert_eq!(
                ts.len(),
                want.len(),
                "shift {shift}: {ts:?} against {want:?}"
            );
            for (t, w) in ts.iter().zip(&want) {
                assert!(
                    (t - w).abs() < 1e-7,
                    "shift {shift}: {ts:?} against {want:?}"
                );
            }
        }
    }

    /// Every crossing of a line with an extrusion along z of `profile`, as the profile's own
    /// passings through the line's plane `x = at`, each as the line's parameter: the answer a
    /// search along the line must reach.
    fn passings(profile: &Curve, at: f64, o: Vec3, d: Vec3) -> Vec<f64> {
        let (u0, u1) = profile.domain();
        let steps = 400_000;
        let u = |i: usize| u0 + (u1 - u0) * i as f64 / steps as f64;
        let side = |t: f64| profile.eval(t).x - at;
        let mut out = Vec::new();
        for i in 0..steps {
            let (a, b) = (side(u(i)), side(u(i + 1)));
            if (a < 0.0) != (b < 0.0) {
                let t = numeric::bisect(&mut |t| side(t), u(i), u(i + 1));
                out.push((profile.eval(t) - o).y / d.y);
            }
        }
        out.sort_by(f64::total_cmp);
        out
    }

    #[test]
    fn a_line_along_a_rational_extrusions_ripples_meets_it_at_every_crossing() {
        // An open rational cubic rippling across x = 5, its weights uneven, extruded along z.
        let ctrl: Vec<Vec3> = (0..20)
            .map(|k| {
                let x = 5.0 + if k % 2 == 0 { 0.004 } else { -0.004 };
                Vec3::new(x, -0.1 + 0.01 * k as f64, 0.0)
            })
            .collect();
        let weights = (0..20).map(|k| 1.0 + 0.7 * (k % 3) as f64).collect();
        let mut knots = vec![0.0; 4];
        knots.extend((1..17).map(|i| i as f64));
        knots.extend([17.0; 4]);
        let nurbs = crate::geom::NurbsCurve::new(3, knots, ctrl, Some(weights)).unwrap();
        let profile = Curve::Nurbs(Box::new(nurbs));
        let surface = Surface::Extrusion(Box::new(crate::geom::Extrusion {
            curve: profile.clone(),
            dir: Vec3::Z,
        }));
        for shift in [0.0, 0.0007, -0.0005] {
            let o = Vec3::new(5.0 + shift, -20.0, -3.0);
            let d = Vec3::new(0.0, 1.0, 0.2).unit().unwrap();
            let want = passings(&profile, 5.0 + shift, o, d);
            assert!(want.len() >= 4, "shift {shift}: {} crossings", want.len());
            let got = curve_surface(
                &Curve::Line(Line { origin: o, dir: d }),
                (-100.0, 100.0),
                &surface,
            );
            let mut ts = got.ts.clone();
            ts.sort_by(f64::total_cmp);
            assert_eq!(
                ts.len(),
                want.len(),
                "shift {shift}: {ts:?} against {want:?}"
            );
            for (t, w) in ts.iter().zip(&want) {
                assert!(
                    (t - w).abs() < 1e-7,
                    "shift {shift}: {ts:?} against {want:?}"
                );
            }
        }
    }

    #[test]
    fn a_line_meets_an_extrusion_three_times_within_a_fraction_of_one_piece() {
        // One cubic piece, x = 5 + 300·(s − 0.505)(s − 0.515)(s − 0.525), y = 10·s: it passes
        // through the plane x = 5 three times within a fiftieth of its length. With weights in
        // geometric progression it is the same curve, its parameter stretched towards one end.
        let (a, b, c) = (0.505, 0.515, 0.525);
        let k = 300.0;
        let power = [
            -k * a * b * c,
            k * (a * b + b * c + c * a),
            -k * (a + b + c),
            k,
        ];
        let bezier = [
            power[0],
            power[0] + power[1] / 3.0,
            power[0] + 2.0 * power[1] / 3.0 + power[2] / 3.0,
            power[0] + power[1] + power[2] + power[3],
        ];
        let ctrl: Vec<Vec3> = (0..4)
            .map(|i| Vec3::new(5.0 + bezier[i], 10.0 * i as f64 / 3.0, 0.0))
            .collect();
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        for weights in [None, Some(vec![1.0, 2.0, 4.0, 8.0])] {
            let nurbs = crate::geom::NurbsCurve::new(3, knots.clone(), ctrl.clone(), weights);
            let profile = Curve::Nurbs(Box::new(nurbs.unwrap()));
            let surface = Surface::Extrusion(Box::new(crate::geom::Extrusion {
                curve: profile.clone(),
                dir: Vec3::Z,
            }));
            let o = Vec3::new(5.0, -20.0, -3.0);
            let d = Vec3::new(0.0, 1.0, 0.2).unit().unwrap();
            let want = passings(&profile, 5.0, o, d);
            assert_eq!(want.len(), 3);
            let line = Curve::Line(Line { origin: o, dir: d });
            let mut ts = curve_surface(&line, (-100.0, 100.0), &surface).ts;
            ts.sort_by(f64::total_cmp);
            assert_eq!(ts.len(), 3, "{ts:?} against {want:?}");
            for (t, w) in ts.iter().zip(&want) {
                assert!((t - w).abs() < 1e-7, "{ts:?} against {want:?}");
            }
        }
    }

    /// An open cubic in the plane y = 0 rippling across the radius 5 as it climbs z.
    fn rippled_meridian() -> Curve {
        let ctrl: Vec<Vec3> = (0..20)
            .map(|k| {
                let r = 5.0 + if k % 2 == 0 { 0.004 } else { -0.004 };
                Vec3::new(r, 0.0, -0.1 + 0.01 * k as f64)
            })
            .collect();
        let mut knots = vec![0.0; 4];
        knots.extend((1..17).map(|i| i as f64));
        knots.extend([17.0; 4]);
        Curve::Nurbs(Box::new(
            crate::geom::NurbsCurve::new(3, knots, ctrl, None).unwrap(),
        ))
    }

    /// Where the line meets the surface of revolution of `profile` about z, from the profile:
    /// each of its points is a ring, and the line passes that ring's height at a known distance
    /// from the axis. The answer a search along the line must reach, for a line that climbs.
    fn ring_passings(profile: &Curve, o: Vec3, d: Vec3) -> Vec<f64> {
        let (u0, u1) = profile.domain();
        let steps = 400_000;
        let u = |i: usize| u0 + (u1 - u0) * i as f64 / steps as f64;
        let line_at = |z: f64| (z - o.z) / d.z;
        let side = |t: f64| {
            let q = profile.eval(t);
            let x = o + d * line_at(q.z);
            x.x.hypot(x.y) - q.x.hypot(q.y)
        };
        let mut out = Vec::new();
        for i in 0..steps {
            let (a, b) = (side(u(i)), side(u(i + 1)));
            if (a < 0.0) != (b < 0.0) {
                let t = numeric::bisect(&mut |t| side(t), u(i), u(i + 1));
                out.push(line_at(profile.eval(t).z));
            }
        }
        out.sort_by(f64::total_cmp);
        out
    }

    #[test]
    fn a_line_along_a_revolved_surfaces_ripples_meets_it_at_every_crossing() {
        let profile = rippled_meridian();
        let surface = Surface::Revolution(Box::new(crate::geom::Revolution {
            curve: profile.clone(),
            origin: Vec3::ZERO,
            axis: Vec3::Z,
        }));
        // Up the side at radius 5, along the axis and then askew to it, from far below.
        for (o, d) in [
            (Vec3::new(5.0, 0.0, -30.0), Vec3::Z),
            (Vec3::new(3.0, 4.0005, -30.0), Vec3::Z),
            (
                Vec3::new(5.0 - 0.3, 0.4, -30.0),
                Vec3::new(0.01, -0.013, 1.0),
            ),
        ] {
            let d = d.unit().unwrap();
            let want = ring_passings(&profile, o, d);
            assert!(want.len() >= 6, "{o:?}: {} crossings", want.len());
            let line = Curve::Line(Line { origin: o, dir: d });
            let mut ts = curve_surface(&line, (0.0, 100.0), &surface).ts;
            ts.sort_by(f64::total_cmp);
            assert_eq!(ts.len(), want.len(), "{o:?}: {ts:?} against {want:?}");
            for (t, w) in ts.iter().zip(&want) {
                assert!((t - w).abs() < 1e-7, "{o:?}: {ts:?} against {want:?}");
            }
        }
    }

    #[test]
    fn a_line_square_to_a_revolved_surfaces_axis_meets_its_ring_twice() {
        let profile = rippled_meridian();
        let surface = Surface::Revolution(Box::new(crate::geom::Revolution {
            curve: profile.clone(),
            origin: Vec3::ZERO,
            axis: Vec3::Z,
        }));
        // At a height the profile passes once, the surface is one ring: its radius from the
        // profile, and the line's two crossings of a circle of that radius from a quadratic.
        let z = 0.0312;
        let (u0, u1) = profile.domain();
        let u = numeric::bisect(&mut |t| profile.eval(t).z - z, u0, u1);
        let ring = profile.eval(u).x;
        let (o, d) = (Vec3::new(-20.0, 1.5, z), Vec3::X);
        let half = (ring * ring - 1.5 * 1.5).sqrt();
        let line = Curve::Line(Line { origin: o, dir: d });
        let mut ts = curve_surface(&line, (0.0, 100.0), &surface).ts;
        ts.sort_by(f64::total_cmp);
        assert_eq!(ts.len(), 2, "{ts:?}");
        assert!((ts[0] - (20.0 - half)).abs() < 1e-7 && (ts[1] - (20.0 + half)).abs() < 1e-7);
    }

    #[test]
    fn a_line_along_a_ruled_surfaces_ripples_meets_it_at_every_crossing() {
        // Rulings from the rippled curve to the same curve 3 mm along y: the surface is the
        // curve's extrusion along y, drawn as a ruled one, so a line in a plane of constant y
        // meets it where it meets the curve.
        let near = rippled_meridian();
        let far = match &near {
            Curve::Nurbs(n) => Curve::Nurbs(Box::new(n.transformed(
                &crate::math::Transform::translation(Vec3::new(0.0, 3.0, 0.0)),
            ))),
            _ => unreachable!(),
        };
        let range = near.domain();
        let surface = Surface::Ruled(Box::new(crate::geom::Ruled {
            a: near.clone(),
            a_range: range,
            b: far,
            b_range: range,
        }));
        for shift in [0.0, 0.0007, -0.0005] {
            // Up through the ripples in the plane y = 1, leaning along y a little.
            let o = Vec3::new(5.0 + shift, 1.0, -30.0);
            let d = Vec3::new(0.0, 0.02, 1.0).unit().unwrap();
            // The curve passes the plane x = 5 + shift where the line, which stays in it, meets
            // the ruling through that point.
            let steps = 400_000;
            let u = |i: usize| range.0 + (range.1 - range.0) * i as f64 / steps as f64;
            let side = |t: f64| near.eval(t).x - (5.0 + shift);
            let mut want = Vec::new();
            for i in 0..steps {
                let (a, b) = (side(u(i)), side(u(i + 1)));
                if (a < 0.0) != (b < 0.0) {
                    let t = numeric::bisect(&mut |t| side(t), u(i), u(i + 1));
                    want.push((near.eval(t).z - o.z) / d.z);
                }
            }
            want.sort_by(f64::total_cmp);
            assert!(want.len() >= 6, "shift {shift}: {} crossings", want.len());
            let line = Curve::Line(Line { origin: o, dir: d });
            let mut ts = curve_surface(&line, (0.0, 100.0), &surface).ts;
            ts.sort_by(f64::total_cmp);
            assert_eq!(
                ts.len(),
                want.len(),
                "shift {shift}: {ts:?} against {want:?}"
            );
            for (t, w) in ts.iter().zip(&want) {
                assert!(
                    (t - w).abs() < 1e-7,
                    "shift {shift}: {ts:?} against {want:?}"
                );
            }
        }
    }

    #[test]
    fn a_line_through_a_torus_whose_tube_reaches_its_axis_meets_both_its_sheets() {
        // A tube of radius 2 about a circle of radius 1: it passes the axis, and the surface has
        // an outer sheet and an inner one. Along a diameter the line meets them at 3 and at 1
        // from the axis, either side.
        let torus = Surface::Torus(Torus {
            frame: Frame::WORLD,
            major: 1.0,
            minor: 2.0,
        });
        let line = |o: [f64; 3], d: [f64; 3]| line(o, d);
        let mut ts = curve_surface(
            &line([-10.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            (0.0, 100.0),
            &torus,
        )
        .ts;
        ts.sort_by(f64::total_cmp);
        assert_eq!(ts.len(), 4, "{ts:?}");
        for (t, w) in ts.iter().zip([7.0, 9.0, 11.0, 13.0]) {
            assert!((t - w).abs() < 1e-9, "{ts:?}");
        }
        // Askew: every crossing lies on one sheet or the other, and a fine walk along the line
        // finds as many.
        let (o, d) = (
            Vec3::new(-9.0, 0.3, -0.4),
            Vec3::new(1.0, 0.02, 0.05).unit().unwrap(),
        );
        let sheet = |t: f64, sign: f64| {
            let q = o + d * t;
            (q.x.hypot(q.y) + sign).hypot(q.z) - 2.0
        };
        let both = |t: f64| sheet(t, -1.0) * sheet(t, 1.0);
        let steps = 2_000_000;
        let changes = (0..steps)
            .filter(|&i| {
                let (a, b) = (
                    20.0 * i as f64 / steps as f64,
                    20.0 * (i + 1) as f64 / steps as f64,
                );
                (both(a) < 0.0) != (both(b) < 0.0)
            })
            .count();
        let ts = curve_surface(
            &Curve::Line(Line { origin: o, dir: d }),
            (0.0, 20.0),
            &torus,
        )
        .ts;
        assert_eq!(ts.len(), changes, "{ts:?}");
        for &t in &ts {
            assert!(sheet(t, -1.0).abs().min(sheet(t, 1.0).abs()) < 1e-9, "{t}");
        }
    }

    /// A cubic by cubic sheet over the square 0..0.24 by 0..0.24 whose height ripples along x
    /// every hundredth of a millimetre: its x and y run evenly with its parameters, so the point
    /// above (x, y) is the surface at (x/0.24, y/0.24), and a line's height over it is known at
    /// every point of the line.
    fn rippled_sheet() -> crate::geom::NurbsSurface {
        let n = 24;
        let mut knots = vec![0.0; 4];
        knots.extend((1..n - 3).map(|i| i as f64 / (n - 3) as f64));
        knots.extend([1.0; 4]);
        // Greville abscissae: control points placed there make a coordinate run evenly.
        let greville = |i: usize| (knots[i + 1] + knots[i + 2] + knots[i + 3]) / 3.0;
        let mut ctrl = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let z = if i % 2 == 0 { 0.004 } else { -0.004 };
                ctrl.push(Vec3::new(0.24 * greville(i), 0.24 * greville(j), z));
            }
        }
        crate::geom::NurbsSurface::new((3, 3), (knots.clone(), knots), (n, n), ctrl, None).unwrap()
    }

    #[test]
    fn the_patch_tree_finds_a_patch_near_each_stretch_where_a_scan_does() {
        let sheet = rippled_sheet();
        let tilted = sheet.transformed(&crate::math::Transform::rotation(
            Vec3::new(0.12, 0.12, 0.0),
            Vec3::new(1.0, 1.0, 0.0).unit().unwrap(),
            0.3,
        ));
        let wave = |z: f64, lift: f64| {
            let pts: Vec<Vec3> = (0..=40)
                .map(|i| {
                    let x = -0.05 + 0.34 * i as f64 / 40.0;
                    Vec3::new(
                        x,
                        0.12 + 0.08 * (20.0 * x).sin(),
                        z * (31.0 * x).cos() + lift,
                    )
                })
                .collect();
            Curve::BSpline(crate::geom::BSpline::interpolate(&pts).unwrap())
        };
        // Through the sheet many times, past it just outside the tolerance, and well clear.
        let curves = [wave(0.01, 0.0), wave(0.002, 0.008), wave(0.01, 0.05)];
        let meets = |a: &Aabb, b: &Aabb| {
            (0..3).all(|k| a.min[k] <= b.max[k] + tol::LINEAR && b.min[k] <= a.max[k] + tol::LINEAR)
        };
        let (mut near, mut far) = (0, 0);
        for ns in [&sheet, &tilted] {
            for c in &curves {
                let range = c.domain();
                let mut cuts = vec![range.0];
                cuts.extend(c.breakpoints(range.0, range.1));
                cuts.push(range.1);
                let n = if cuts.len() > 2 { 12 } else { 96 };
                let samples: Vec<f64> = cuts
                    .windows(2)
                    .flat_map(|w| (0..n).map(move |i| w[0] + (w[1] - w[0]) * i as f64 / n as f64))
                    .chain(std::iter::once(range.1))
                    .collect();
                for w in samples.windows(2) {
                    let bb = c.bbox(w[0], w[1]);
                    let scan = ns.patches().iter().any(|q| meets(&bb, &q.bbox));
                    assert_eq!(ns.patch_tree().any(|q| meets(&bb, q)), scan, "{w:?}");
                    if scan { near += 1 } else { far += 1 }
                }
            }
        }
        assert!(near > 100 && far > 100, "{near} {far}");
        // Boxes kept from one surface's test serve the next, and the same one again.
        let surfaces = [
            Surface::Nurbs(Box::new(sheet)),
            Surface::Nurbs(Box::new(tilted)),
        ];
        for c in &curves {
            let mut boxes = CurveBoxes::default();
            for _ in 0..2 {
                for s in &surfaces {
                    let got = curve_surface_in(c, c.domain(), s, &mut boxes);
                    assert_eq!(got, curve_surface(c, c.domain(), s));
                }
            }
        }
    }

    #[test]
    fn a_line_skimming_a_spline_sheets_ripples_meets_it_at_every_crossing() {
        let sheet = rippled_sheet();
        let surface = Surface::Nurbs(Box::new(sheet.clone()));
        for (o, d) in [
            // Level with the sheet's middle height, across the ripples and askew to them.
            (Vec3::new(-1.0, 0.1, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(-1.0, 0.05, 0.0003), Vec3::new(1.0, 0.05, 0.0)),
            // Falling through it at a shallow angle.
            (Vec3::new(-1.0, 0.2, 0.004), Vec3::new(1.0, -0.02, -0.0035)),
        ] {
            let d = d.unit().unwrap();
            let above = |t: f64| {
                let q = o + d * t;
                q.z - sheet.eval(q.x / 0.24, q.y / 0.24).z
            };
            let over = |t: f64| {
                let q = o + d * t;
                (0.0..=0.24).contains(&q.x) && (0.0..=0.24).contains(&q.y)
            };
            // A walk along the line in steps a hundredth of a ripple long.
            let steps = 1_000_000;
            let at = |i: usize| 8.0 * i as f64 / steps as f64;
            let mut want = Vec::new();
            for i in 0..steps {
                if over(at(i))
                    && over(at(i + 1))
                    && (above(at(i)) < 0.0) != (above(at(i + 1)) < 0.0)
                {
                    want.push(numeric::bisect(&mut |t| above(t), at(i), at(i + 1)));
                }
            }
            assert!(want.len() >= 6, "{o:?}: {} crossings", want.len());
            let line = Curve::Line(Line { origin: o, dir: d });
            let mut ts = curve_surface(&line, (0.0, 8.0), &surface).ts;
            ts.sort_by(f64::total_cmp);
            assert_eq!(
                ts.len(),
                want.len(),
                "{o:?}: {} against {}",
                ts.len(),
                want.len()
            );
            for (t, w) in ts.iter().zip(&want) {
                assert!((t - w).abs() < 1e-6, "{o:?}: {t} against {w}");
            }
        }
    }

    fn cylinder(r: f64) -> Surface {
        Surface::Cylinder(Cylinder {
            frame: Frame::WORLD,
            radius: r,
        })
    }

    #[test]
    fn a_line_touching_a_torus_within_rounding_meets_it_once_where_it_touches() {
        // The tube (major 5.5, minor 1.5 about z) reaches in to r = 4 at z = 0. A line along z at
        // r = 4 + 3e-11 dips 3e-11 into it, crossing at z = ±√(3·3e-11) ≈ ±9.5e-6: one touch.
        let torus = Surface::Torus(crate::geom::Torus {
            frame: Frame::WORLD,
            major: 5.5,
            minor: 1.5,
        });
        let hits = curve_surface(
            &line([4.0 + 3e-11, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (-5.0, 5.0),
            &torus,
        );
        assert_eq!(hits.ts.len(), 1, "{hits:?}");
        assert!(hits.ts[0].abs() < 1e-7, "{hits:?}");
    }

    #[test]
    fn a_line_cutting_a_tube_between_two_samples_meets_it_twice() {
        // The tube about z, centred at z = 0.05, cut by a line along z at r = 4 + d where
        // (1.5 − d)² = 1.5² − 0.04²: it crosses at z = 0.05 ± 0.04, both between the samples at
        // z = 0 and z ≈ 0.104 of the 96 over [−5, 5].
        let torus = Surface::Torus(crate::geom::Torus {
            frame: Frame::WORLD.translated(Vec3::new(0.0, 0.0, 0.05)),
            major: 5.5,
            minor: 1.5,
        });
        let d = 1.5 - (1.5f64 * 1.5 - 0.04 * 0.04).sqrt();
        let hits = curve_surface(
            &line([4.0 + d, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (-5.0, 5.0),
            &torus,
        );
        assert_eq!(hits.ts.len(), 2, "{hits:?}");
        assert!((hits.ts[0] - 0.01).abs() < 1e-9, "{hits:?}");
        assert!((hits.ts[1] - 0.09).abs() < 1e-9, "{hits:?}");
    }

    #[test]
    fn line_meets_cylinder_twice_once_or_never() {
        let s = cylinder(5.0);
        let two = curve_surface(
            &line([-10.0, 4.999999, 0.0], [1.0, 0.0, 0.0]),
            (0.0, 20.0),
            &s,
        );
        assert_eq!(two.ts.len(), 2);
        let gap = (two.ts[1] - two.ts[0]).abs();
        assert!((gap - 2.0 * (5.0f64.powi(2) - 4.999999f64.powi(2)).sqrt()).abs() < 1e-12);
        let touch = curve_surface(&line([-10.0, 5.0, 0.0], [1.0, 0.0, 0.0]), (0.0, 20.0), &s);
        assert_eq!(touch.ts, vec![10.0]);
        let miss = curve_surface(
            &line([-10.0, 5.000001, 0.0], [1.0, 0.0, 0.0]),
            (0.0, 20.0),
            &s,
        );
        assert!(miss.ts.is_empty() && !miss.on_surface);
        let along = curve_surface(&line([5.0, 0.0, 0.0], [0.0, 0.0, 1.0]), (0.0, 20.0), &s);
        assert!(along.on_surface);
    }

    #[test]
    fn circle_meets_plane_in_closed_form() {
        let c = Curve::Circle(Circle {
            frame: Frame::WORLD,
            radius: 5.0,
        });
        let at = |x: f64| {
            Surface::Plane(Plane {
                frame: Frame::new(Vec3::new(x, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap(),
            })
        };
        assert_eq!(curve_surface(&c, (0.0, TAU), &at(3.0)).ts.len(), 2);
        let touch = curve_surface(&c, (0.0, TAU), &at(-5.0)).ts;
        assert_eq!(touch.len(), 1);
        assert!((touch[0] - TAU / 2.0).abs() < 1e-12);
        assert!(
            curve_surface(
                &c,
                (0.0, TAU),
                &Surface::Plane(Plane {
                    frame: Frame::WORLD
                })
            )
            .on_surface
        );
    }

    #[test]
    fn a_round_touching_its_pillar_meets_it_at_the_touching_point() {
        // A quarter round's meridian edge, off the origin and turned, touching its pillar; rounding
        // there made sign changes 3e-8 mm from the touching point.
        for k in 0..64 {
            let a = 0.1 * k as f64;
            let axis = Vec3::new(20.0 * a.sin(), 0.0, 20.0 * (1.3 * a).cos());
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let c = Curve::Circle(Circle {
                frame: Frame::new(
                    axis + out * 4.5 - Vec3::Y * 16.0,
                    out.cross(Vec3::Y),
                    Vec3::Y,
                )
                .unwrap(),
                radius: 3.0,
            });
            let pillar = Surface::Cylinder(Cylinder {
                frame: Frame::new(axis, -Vec3::Y, out).unwrap(),
                radius: 1.5,
            });
            let touch = axis + out * 1.5 - Vec3::Y * 16.0;
            let end = (0..4)
                .map(|k| k as f64 * TAU / 4.0)
                .find(|&t| c.eval(t).dist(touch) <= 1e-12)
                .unwrap();
            // The range ends on the touching point, as the round's does, so the touch need not be
            // reported; a crossing reported beside it would be a second vertex.
            for t in curve_surface(&c, (end - TAU / 4.0, end), &pillar).ts {
                let d = c.eval(t).dist(touch);
                assert!(d <= 1e-9, "{k} {d}");
            }
        }
    }

    #[test]
    fn a_ray_clipping_a_lobe_of_an_extruded_outline_meets_it_twice() {
        // The wall r = 30 + 3·cos 9θ extruded along z, and rays along y at x = 33 − δ, a hair in
        // from the lobe tip at (33, 0): near the tip x ≈ 33 − 0.127·y², so each ray enters and
        // leaves 2·√(δ / 0.127) apart, under 0.06 mm for δ = 1e-4, a fraction of the step its
        // 140 mm are sampled at. Each must meet the wall as often as the outline crosses its line,
        // counted on a dense sampling of the outline.
        let r = |a: f64| 30.0 + 3.0 * (9.0 * a).cos();
        let pts: Vec<Vec3> = (0..360)
            .map(|i| {
                let a = TAU * i as f64 / 360.0;
                Vec3::new(r(a) * a.cos(), r(a) * a.sin(), 0.0)
            })
            .collect();
        let curve = Curve::BSpline(crate::geom::BSpline::interpolate_periodic(&pts).unwrap());
        let wall = Surface::Extrusion(Box::new(crate::geom::Extrusion {
            curve: curve.clone(),
            dir: Vec3::Z,
        }));
        let (t0, t1) = curve.domain();
        let dense: Vec<Vec3> = (0..=400_000)
            .map(|k| curve.eval(t0 + (t1 - t0) * k as f64 / 400_000.0))
            .collect();
        let tip = dense.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        for delta in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2] {
            let x = tip - delta;
            let o = Vec3::new(x, -70.0, 1.0);
            let want = dense
                .windows(2)
                .filter(|w| (w[0].x - x) * (w[1].x - x) < 0.0)
                .count();
            assert!(want >= 2, "δ = {delta}");
            let hits = ray_surface(&wall, o, Vec3::Y, 140.0);
            assert_eq!(hits.len(), want, "δ = {delta}: {hits:?}");
            for t in hits {
                assert!(signed_distance(&wall, o + Vec3::Y * t).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn numeric_hits_find_a_circle_crossing_a_cylinder() {
        let c = Curve::Circle(Circle {
            frame: Frame::new(Vec3::new(4.0, 0.0, 0.0), Vec3::Z, Vec3::X).unwrap(),
            radius: 3.0,
        });
        let h = curve_surface(&c, (0.0, TAU), &cylinder(5.0));
        assert_eq!(h.ts.len(), 2);
        for t in h.ts {
            assert!(signed_distance(&cylinder(5.0), c.eval(t)).abs() < 1e-12);
        }
    }

    #[test]
    fn coplanar_circles_and_lines_cross() {
        let a = Curve::Circle(Circle {
            frame: Frame::WORLD,
            radius: 5.0,
        });
        let b = Curve::Circle(Circle {
            frame: Frame::new(Vec3::new(6.0, 0.0, 0.0), Vec3::Z, Vec3::X).unwrap(),
            radius: 3.0,
        });
        let pts = curve_curve(&a, (0.0, TAU), &b, (0.0, TAU));
        assert_eq!(pts.len(), 2);
        let l = line([0.0, -10.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(curve_curve(&a, (0.0, TAU), &l, (0.0, 20.0)).len(), 2);
        let tangent = line([5.0, -10.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(curve_curve(&a, (0.0, TAU), &tangent, (0.0, 20.0)).len(), 1);
    }

    #[test]
    fn rays_hit_quadrics() {
        let hits = ray_surface(&cylinder(5.0), Vec3::ZERO, Vec3::X, 100.0);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|&t| (t - 5.0).abs() < 1e-12));
    }

    /// A cubic through `per_turn` points a turn of the helix of radius `r` rising 1 a turn about
    /// z, from turn `a` to turn `b`, every half turn among them; mirrored in y = 0 when `mirror`.
    fn helix(r: f64, per_turn: usize, (a, b): (f64, f64), mirror: bool) -> Curve {
        let n = ((b - a) * per_turn as f64).round() as usize;
        let pts: Vec<Vec3> = (0..=n)
            .map(|i| {
                let turn = a + (b - a) * i as f64 / n as f64;
                let th = TAU * turn;
                Vec3::new(r * th.cos(), r * th.sin(), turn)
            })
            .collect();
        let s = crate::geom::BSpline::interpolate(&pts).unwrap();
        if !mirror {
            return Curve::BSpline(s);
        }
        let flip: Vec<Vec3> = s
            .control_points()
            .iter()
            .map(|p| Vec3::new(p.x, -p.y, p.z))
            .collect();
        Curve::BSpline(crate::geom::BSpline::new(s.degree(), s.knots().to_vec(), flip).unwrap())
    }

    #[test]
    fn a_helix_crosses_its_mirror_image_every_half_turn() {
        // A right-hand helix of radius 5 rising 1 a turn, from a quarter turn before 0 to a
        // quarter past 3, and its mirror in y = 0, a left-hand one: a point of both lies in the
        // mirror plane, where the right-hand one passes at each half turn k/2, at (±5, 0, k/2).
        // Each is a node of the fit, so the splines pass there exactly: seven crossings.
        let a = helix(5.0, 360, (-0.25, 3.25), false);
        let b = helix(5.0, 360, (-0.25, 3.25), true);
        let (ra, rb) = (a.domain(), b.domain());
        let mut got = curve_curve(&a, ra, &b, rb);
        got.sort_by(|p, q| p.z.total_cmp(&q.z));
        assert_eq!(got.len(), 7, "{got:?}");
        for (k, p) in got.iter().enumerate() {
            let want = Vec3::new(if k % 2 == 0 { 5.0 } else { -5.0 }, 0.0, 0.5 * k as f64);
            assert!(p.dist(want) <= 1e-9, "{k}: {p:?}");
        }
    }

    #[test]
    fn pruning_by_hulls_keeps_every_meeting_of_spline_curves() {
        // The search sampled everywhere and the search sampled only near the other curve's hulls
        // find the same points, bit for bit: a helix crossing its mirror image (every half turn),
        // a helix and a coaxial one 0.15 wider (never), a helix and a spline along the line
        // x = 5, y = 0 (each whole turn, at (5, 0, k)), and a helix and its copy raised by half
        // the tolerance (running together, no point).
        let h = helix(5.0, 40, (-0.25, 2.25), false);
        let up = crate::math::Transform::translation(Vec3::Z * (0.5 * tol::LINEAR));
        let post = Curve::BSpline(
            crate::geom::BSpline::interpolate(&[
                Vec3::new(5.0, 0.0, -0.6),
                Vec3::new(5.0, 0.0, 0.7),
                Vec3::new(5.0, 0.0, 1.4),
                Vec3::new(5.0, 0.0, 2.6),
            ])
            .unwrap(),
        );
        let cases = [
            (h.clone(), helix(5.0, 40, (-0.25, 2.25), true), 5),
            (h.clone(), helix(5.15, 40, (-0.25, 2.25), false), 0),
            (h.clone(), post, 3),
            (h.clone(), h.transformed(&up), 0),
        ];
        for (i, (a, b, n)) in cases.iter().enumerate() {
            let (ra, rb) = (a.domain(), b.domain());
            let full = sampled_curve_curve(a, ra, b, rb, false);
            let pruned = sampled_curve_curve(a, ra, b, rb, true);
            assert_eq!(pruned, full, "case {i}");
            assert_eq!(pruned.len(), *n, "case {i}: {pruned:?}");
        }
    }

    #[test]
    fn within_agrees_with_the_distance() {
        // Points at the helix's node (5, 0, 1) and moved out from the axis by 0.5, 1.5 and 3
        // times the tolerance, and one far off on the axis: within the tolerance exactly when
        // the projection says so.
        let h = helix(5.0, 40, (-0.25, 2.25), false);
        let r = h.domain();
        for (out, want) in [(0.0, true), (0.5, true), (1.5, false), (3.0, false)] {
            let p = Vec3::new(5.0 + out * tol::LINEAR, 0.0, 1.0);
            assert_eq!(within(&h, r, p, tol::LINEAR), want, "{out}");
            assert_eq!(curve_distance(&h, r, p).0 <= tol::LINEAR, want, "{out}");
        }
        assert!(!within(&h, r, Vec3::new(0.0, 0.0, 1.0), tol::LINEAR));
    }

    // A circle in the plane x = 5, turned about the z axis, draws a surface whose meridian runs out
    // to radius √(25 + r²) and back: the circle lies on it all round, and every point of it is its
    // own nearest point there, also beside the top and bottom, where the meridian turns back.
    #[test]
    fn a_circle_lies_on_its_own_revolution_where_the_meridian_turns_back() {
        let c = Curve::Circle(Circle {
            frame: Frame::new(Vec3::new(5.0, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap(),
            radius: 0.5,
        });
        let s = Surface::Revolution(Box::new(Revolution {
            curve: c.clone(),
            origin: Vec3::ZERO,
            axis: Vec3::Z,
        }));
        for k in 0..64 {
            let p = c.eval(std::f64::consts::TAU * k as f64 / 64.0);
            let uv = s.project(p);
            assert!(s.eval(uv.x, uv.y).dist(p) < 1e-12, "{k}");
        }
        let hits = curve_surface(&c, (0.0, std::f64::consts::TAU), &s);
        assert!(hits.on_surface, "{hits:?}");
    }

    #[test]
    fn the_signed_distance_to_a_cone_or_a_spindle_torus_counts_its_far_sheet() {
        // As far from a point as the nearer of the two lines or circles in its meridian plane, and
        // nothing on the far sheet itself.
        let f = Frame::new(Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.2, 0.3, 1.0), Vec3::X).unwrap();
        let cone = crate::geom::Cone {
            frame: f,
            radius: 2.0,
            half_angle: 0.4,
        };
        let spindle = Torus {
            frame: f,
            major: 2.0,
            minor: 3.0,
        };
        let (sa, ca) = 0.4f64.sin_cos();
        for k in 0..40 {
            let (rho, z) = (0.25 * k as f64, -10.0 + 0.6 * k as f64);
            let p = f.to_world(Vec3::new(rho * 0.6, rho * 0.8, z));
            let near = ((rho - 2.0) * ca - z * sa).abs();
            let far = ((rho + 2.0) * ca + z * sa).abs();
            let d = signed_distance(&Surface::Cone(cone.clone()), p);
            assert!(
                (d.abs() - near.min(far)).abs() <= 1e-12,
                "cone at ({rho}, {z}): {d}"
            );
            let near = ((rho - 2.0).hypot(z / 3.0) - 3.0).abs();
            let far = ((rho + 2.0).hypot(z / 3.0) - 3.0).abs();
            let q = f.to_world(Vec3::new(rho * 0.6, rho * 0.8, z / 3.0));
            let d = signed_distance(&Surface::Torus(spindle.clone()), q);
            assert!(
                (d.abs() - near.min(far)).abs() <= 1e-12,
                "torus at ({rho}, {z}): {d}"
            );
        }
        // The far nappe's point at v below the apex, turned half a turn: on the surface.
        let s = Surface::Cone(cone);
        for v in [-8.0, -12.0, -20.0] {
            let p = s.eval(0.7, v);
            assert!(signed_distance(&s, p).abs() <= 1e-12, "v {v}");
        }
        let s = Surface::Torus(spindle);
        for v in [2.5, 3.0, 3.5] {
            let p = s.eval(1.1, v);
            assert!(signed_distance(&s, p).abs() <= 1e-12, "v {v}");
        }
    }
}
