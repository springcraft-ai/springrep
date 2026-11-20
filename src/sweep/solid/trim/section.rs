//! How deep a point lies in a body of revolution, read in the body's meridian section: the body is
//! the section turned about its axis, so in the half-plane through the axis and a point, the
//! point's distance from the body's surface is its distance from the section's outline, each face's
//! meridian there a segment or an arc of a circle.

use std::f64::consts::{FRAC_PI_2, TAU};

use crate::{
    boolean::face_domain,
    geom::Surface,
    math::{Point3, Vec2, Vec3},
    topo::Brep,
};

/// A stretch of the outline, run with the outside of the body on its right.
#[derive(Clone, Debug)]
enum Stretch {
    Segment {
        a: Vec2,
        b: Vec2,
    },
    /// From angle `from` to angle `to` about `centre`, counter-clockwise where `to > from`.
    Arc {
        centre: Vec2,
        radius: f64,
        from: f64,
        to: f64,
    },
}

/// The point of a stretch nearest a point: the distance, the point, and the outward normal there,
/// or `None` where it is an end of the stretch.
struct Foot {
    dist: f64,
    point: Vec2,
    normal: Option<Vec2>,
}

impl Stretch {
    fn foot(&self, p: Vec2) -> Foot {
        match *self {
            Stretch::Segment { a, b } => {
                let d = b - a;
                let t = ((p - a).dot(d) / d.dot(d)).clamp(0.0, 1.0);
                let q = a + d * t;
                let inner = t > 0.0 && t < 1.0;
                Foot {
                    dist: p.dist(q),
                    point: q,
                    normal: inner.then(|| Vec2::new(d.y, -d.x) / d.norm()),
                }
            }
            Stretch::Arc {
                centre,
                radius,
                from,
                to,
            } => {
                let r = p - centre;
                let theta = r.y.atan2(r.x);
                let ccw = to > from;
                let sweep = (to - from).abs();
                let start = if ccw { from } else { to };
                let into = (theta - start).rem_euclid(TAU);
                let radial = Vec2::new(theta.cos(), theta.sin());
                if into <= sweep {
                    let normal = if ccw { radial } else { -radial };
                    return Foot {
                        dist: (r.norm() - radius).abs(),
                        point: centre + radial * radius,
                        normal: Some(normal),
                    };
                }
                let end = |a: f64| centre + Vec2::new(a.cos(), a.sin()) * radius;
                let (ea, eb) = (end(from), end(to));
                let point = if p.dist(ea) <= p.dist(eb) { ea } else { eb };
                Foot {
                    dist: p.dist(point),
                    point,
                    normal: None,
                }
            }
        }
    }

    /// The angle the stretch turns through seen from `p`, the winding of the outline about it.
    fn angle(&self, p: Vec2) -> f64 {
        let chord = |a: Vec2, b: Vec2| {
            let (x, y) = (a - p, b - p);
            x.cross(y).atan2(x.dot(y))
        };
        match *self {
            Stretch::Segment { a, b } => chord(a, b),
            Stretch::Arc {
                centre,
                radius,
                from,
                to,
            } => {
                // A piece of the arc short of a quarter turn turns as its chord does, and a whole
                // turn more where `p` lies between the chord and the arc.
                let n = ((to - from).abs() / FRAC_PI_2).ceil().max(1.0) as usize;
                let at = |k: usize| {
                    let a = from + (to - from) * k as f64 / n as f64;
                    centre + Vec2::new(a.cos(), a.sin()) * radius
                };
                let inside_circle = p.dist(centre) < radius;
                (0..n)
                    .map(|k| {
                        let (a, b) = (at(k), at(k + 1));
                        let mut turn = chord(a, b);
                        let beyond = (b - a).cross(p - a) * (b - a).cross(centre - a) < 0.0;
                        if inside_circle && beyond {
                            turn += if to > from { TAU } else { -TAU };
                        }
                        turn
                    })
                    .sum()
            }
        }
    }

    /// The stretch seen in a mirror across the axis, run the other way so the outside stays on
    /// its right.
    fn mirrored(&self) -> Stretch {
        let m = |q: Vec2| Vec2::new(-q.x, q.y);
        match *self {
            Stretch::Segment { a, b } => Stretch::Segment { a: m(b), b: m(a) },
            Stretch::Arc {
                centre,
                radius,
                from,
                to,
            } => Stretch::Arc {
                centre: m(centre),
                radius,
                from: std::f64::consts::PI - to,
                to: std::f64::consts::PI - from,
            },
        }
    }
}

/// A body of revolution as its meridian section: points `(ρ, z)`, `ρ` the distance from the axis
/// through `origin` along the unit `axis` and `z` the height along it.
#[derive(Clone, Debug)]
pub(crate) struct Section {
    origin: Point3,
    axis: Vec3,
    stretches: Vec<Stretch>,
    /// The stretches and their mirror images: the whole closed outline of the section through the
    /// axis, whose winding tells inside from outside.
    whole: Vec<Stretch>,
}

/// Samples along a face's meridian where it is looked for inside the face.
const SAMPLES: usize = 512;

impl Section {
    /// The section of `body` where every face is a plane square to one axis, or a cylinder, cone,
    /// sphere or torus about it; else `None`.
    pub(crate) fn of(body: &Brep) -> Option<Section> {
        let frame_of = |s: &Surface| match s {
            Surface::Cylinder(c) => Some(c.frame),
            Surface::Cone(c) => Some(c.frame),
            Surface::Sphere(c) => Some(c.frame),
            Surface::Torus(c) => Some(c.frame),
            _ => None,
        };
        let first = body
            .face_ids()
            .find_map(|f| frame_of(&body.face(f).surface))?;
        let (origin, axis) = (first.origin, first.z);
        let size = crate::measure::bbox(body).diagonal().max(1e-300);
        let on_axis = |p: Point3| {
            let q = p - origin;
            (q - axis * q.dot(axis)).norm() <= 1e-9 * size
        };
        let along = |z: Vec3| z.cross(axis).norm() <= 1e-9;
        // The half-plane the section is read in, at an angle unlikely to be a face's seam.
        let e = first.radial(1.234_567);
        let flat = |p: Point3| {
            let q = p - origin;
            let z = q.dot(axis);
            Vec2::new((q - axis * z).norm(), z)
        };
        let mut stretches = Vec::new();
        for f in body.face_ids() {
            let face = body.face(f);
            let out = if face.reversed { -1.0 } else { 1.0 };
            let dom = face_domain(body, f);
            let s = &face.surface;
            match s {
                Surface::Plane(pl) => {
                    if !along(pl.frame.z) {
                        return None;
                    }
                    let foot = origin + axis * (pl.frame.origin - origin).dot(axis);
                    let reach = [
                        Vec2::new(dom.lo.x, dom.lo.y),
                        Vec2::new(dom.hi.x, dom.lo.y),
                        Vec2::new(dom.lo.x, dom.hi.y),
                        Vec2::new(dom.hi.x, dom.hi.y),
                    ]
                    .iter()
                    .map(|q| s.eval(q.x, q.y).dist(foot))
                    .fold(0.0, f64::max)
                        * 1.01;
                    let point = |r: f64| foot + e * r;
                    let inside = |r: f64| {
                        let q = s.project(point(r));
                        dom.winding(q) != 0
                    };
                    for (a, b) in runs(&inside, 0.0, reach) {
                        let n = s.normal(0.0, 0.0) * out;
                        stretches.push(oriented_segment(
                            flat(point(a)),
                            flat(point(b)),
                            Vec2::new(n.dot(e), n.dot(axis)),
                        ));
                    }
                }
                Surface::Cylinder(_)
                | Surface::Cone(_)
                | Surface::Sphere(_)
                | Surface::Torus(_) => {
                    let fr = frame_of(s)?;
                    if !along(fr.z) || !on_axis(fr.origin) {
                        return None;
                    }
                    let u0 = e.dot(fr.y).atan2(e.dot(fr.x));
                    let inside = |v: f64| dom.inside_image(Vec2::new(u0, v)).is_some();
                    let (v0, v1) = match s.v_period() {
                        Some(p) if dom.hi.y - dom.lo.y >= p - 1e-9 => (dom.lo.y, dom.lo.y + p),
                        _ => (dom.lo.y, dom.hi.y),
                    };
                    for (a, b) in runs(&inside, v0, v1) {
                        let mid = 0.5 * (a + b);
                        let n = s.normal(u0, mid) * out;
                        let normal = Vec2::new(n.dot(e), n.dot(axis));
                        let (pa, pb, pm) = (
                            flat(s.eval(u0, a)),
                            flat(s.eval(u0, b)),
                            flat(s.eval(u0, mid)),
                        );
                        let whole = s.v_period().is_some_and(|p| b - a >= p - 1e-9);
                        stretches.push(match s {
                            Surface::Sphere(sp) => oriented_arc(
                                flat(sp.frame.origin),
                                sp.radius,
                                pa,
                                pb,
                                pm,
                                normal,
                                whole,
                            ),
                            Surface::Torus(t) => oriented_arc(
                                flat(t.frame.origin + fr.radial(u0) * t.major),
                                t.minor,
                                pa,
                                pb,
                                pm,
                                normal,
                                whole,
                            ),
                            _ => oriented_segment(pa, pb, normal),
                        });
                    }
                }
                _ => return None,
            }
        }
        if stretches.is_empty() {
            return None;
        }
        let mut whole = stretches.clone();
        whole.extend(stretches.iter().map(Stretch::mirrored));
        Some(Section {
            origin,
            axis,
            stretches,
            whole,
        })
    }

    /// How deep `p` lies in the body: its distance from the surface, positive inside.
    pub(crate) fn depth(&self, p: Point3) -> f64 {
        let q = p - self.origin;
        let z = q.dot(self.axis);
        let x = Vec2::new((q - self.axis * z).norm(), z);
        let mut best: Option<Foot> = None;
        for s in &self.stretches {
            let f = s.foot(x);
            if best.as_ref().is_none_or(|b| f.dist < b.dist) {
                best = Some(f);
            }
        }
        let Some(best) = best else {
            return f64::NEG_INFINITY;
        };
        // Off the middle of a stretch the side follows from its normal; at a corner of the outline
        // from the outline's winding about the point.
        let inside = match best.normal {
            Some(n) => (x - best.point).dot(n) <= 0.0,
            None => self.winding(x) != 0,
        };
        if inside { best.dist } else { -best.dist }
    }

    fn winding(&self, x: Vec2) -> i64 {
        let total: f64 = self.whole.iter().map(|s| s.angle(x)).sum();
        (total / TAU).round() as i64
    }
}

/// The runs of `[lo, hi]` where `inside` holds, each end pinned by bisection.
fn runs(inside: &dyn Fn(f64) -> bool, lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let at = |k: usize| lo + (hi - lo) * k as f64 / SAMPLES as f64;
    let pin = |mut a: f64, mut b: f64| {
        let ia = inside(a);
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            if inside(m) == ia {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    };
    let mut out = Vec::new();
    let mut start: Option<f64> = None;
    let mut last = inside(at(0));
    if last {
        start = Some(lo);
    }
    for k in 1..=SAMPLES {
        let now = inside(at(k));
        if now != last {
            let x = pin(at(k - 1), at(k));
            if now {
                start = Some(x);
            } else if let Some(s) = start.take() {
                out.push((s, x));
            }
        }
        last = now;
    }
    if let Some(s) = start {
        out.push((s, hi));
    }
    out
}

/// The segment from `a` to `b`, run so the outward `normal` lies on its right.
fn oriented_segment(a: Vec2, b: Vec2, normal: Vec2) -> Stretch {
    let d = b - a;
    if Vec2::new(d.y, -d.x).dot(normal) >= 0.0 {
        Stretch::Segment { a, b }
    } else {
        Stretch::Segment { a: b, b: a }
    }
}

/// The arc about `centre` from `a` to `b` through `m` (a whole circle where `whole`), run so the
/// outward `normal` at `m` lies on its right.
fn oriented_arc(
    centre: Vec2,
    radius: f64,
    a: Vec2,
    b: Vec2,
    m: Vec2,
    normal: Vec2,
    whole: bool,
) -> Stretch {
    let angle = |q: Vec2| (q.y - centre.y).atan2(q.x - centre.x);
    let (ta, tb, tm) = (angle(a), angle(b), angle(m));
    // Counter-clockwise from `a`, how far to `m` and to `b`.
    let to_m = (tm - ta).rem_euclid(TAU);
    let to_b = if whole {
        TAU
    } else {
        (tb - ta).rem_euclid(TAU)
    };
    let (from, to) = if whole || to_m <= to_b {
        (ta, ta + to_b)
    } else {
        (ta, ta - (TAU - to_b))
    };
    // Counter-clockwise, the right of the run is away from the centre.
    let radial = (m - centre) / (m - centre).norm();
    let ccw_out = radial.dot(normal) >= 0.0;
    let ccw = to > from;
    if ccw == ccw_out {
        Stretch::Arc {
            centre,
            radius,
            from,
            to,
        }
    } else {
        Stretch::Arc {
            centre,
            radius,
            from: to,
            to: from,
        }
    }
}
