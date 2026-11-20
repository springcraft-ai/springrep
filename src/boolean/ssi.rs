//! Surface–surface intersection. Planes against the analytic surfaces, coaxial surfaces of
//! revolution, parallel cylinders and equal crossing cylinders have closed forms; every other pair
//! is traced by [`super::march`].
//!
//! Curves come back over a range that covers the region asked about; the caller cuts them at the
//! faces' boundaries. A curve along which the surfaces are tangent is marked, and isolated points
//! where they are tangent (where two branches cross) come back as `points`.

use std::f64::consts::TAU;

use super::{domain::Domain, march};

mod conic;
mod sweep;
use crate::{
    Result,
    geom::{Circle, Curve, Ellipse, Line, Surface},
    math::{Aabb, Frame, Point3, Vec2, Vec3},
    tol,
};

#[derive(Clone, Debug)]
pub(crate) struct SsiCurve {
    pub curve: Curve,
    pub range: (f64, f64),
    pub closed: bool,
    pub tangent: bool,
}

#[derive(Clone, Debug)]
pub(crate) enum Ssi {
    /// The surfaces are the same point set.
    Coincident,
    Curves {
        curves: Vec<SsiCurve>,
        points: Vec<Point3>,
    },
}

fn none() -> Ssi {
    Ssi::Curves {
        curves: Vec::new(),
        points: Vec::new(),
    }
}

fn parallel(a: Vec3, b: Vec3) -> bool {
    a.cross(b).norm() <= tol::ANGULAR
}

/// The distance from `p` to the line through `o` along the unit `d`.
fn line_distance(p: Point3, o: Point3, d: Vec3) -> f64 {
    let w = p - o;
    (w - d * w.dot(d)).norm()
}

/// An axis of symmetry: a surface of revolution's, or any line through a sphere's centre.
#[derive(Clone, Copy, Debug)]
struct Axis {
    origin: Point3,
    dir: Vec3,
}

fn axis_of(s: &Surface) -> Option<Axis> {
    let f = |fr: &Frame| Axis {
        origin: fr.origin,
        dir: fr.z,
    };
    match s {
        Surface::Cylinder(c) => Some(f(&c.frame)),
        Surface::Cone(c) => Some(f(&c.frame)),
        Surface::Torus(t) => Some(f(&t.frame)),
        Surface::Sphere(sp) => Some(f(&sp.frame)),
        _ => None,
    }
}

fn same_axis(a: Axis, b: Axis) -> bool {
    parallel(a.dir, b.dir) && line_distance(b.origin, a.origin, a.dir) <= tol::LINEAR
}

/// Whether two surfaces are the same point set.
pub fn same_domain(a: &Surface, b: &Surface) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= tol::LINEAR;
    match (a, b) {
        (Surface::Plane(p), Surface::Plane(q)) => {
            parallel(p.frame.z, q.frame.z)
                && (q.frame.origin - p.frame.origin).dot(p.frame.z).abs() <= tol::LINEAR
        }
        (Surface::Cylinder(p), Surface::Cylinder(q)) => {
            close(p.radius, q.radius) && same_axis(axis_of(a).unwrap(), axis_of(b).unwrap())
        }
        (Surface::Sphere(p), Surface::Sphere(q)) => {
            close(p.radius, q.radius) && p.frame.origin.dist(q.frame.origin) <= tol::LINEAR
        }
        (Surface::Torus(p), Surface::Torus(q)) => {
            close(p.major, q.major)
                && close(p.minor, q.minor)
                && p.frame.origin.dist(q.frame.origin) <= tol::LINEAR
                && parallel(p.frame.z, q.frame.z)
        }
        (Surface::Cone(p), Surface::Cone(q)) => {
            if !same_axis(axis_of(a).unwrap(), axis_of(b).unwrap()) {
                return false;
            }
            let apex = |c: &crate::geom::Cone| {
                let (sa, ca) = c.half_angle.sin_cos();
                (sa.abs() > 1e-12).then(|| c.frame.origin + c.frame.z * (-c.radius / sa * ca))
            };
            let slope = |c: &crate::geom::Cone, z: Vec3| {
                let t = c.half_angle.tan();
                if c.frame.z.dot(z) > 0.0 { t } else { -t }
            };
            match (apex(p), apex(q)) {
                (Some(x), Some(y)) => {
                    x.dist(y) <= tol::LINEAR
                        && (slope(p, p.frame.z).abs() - slope(q, p.frame.z).abs()).abs()
                            <= tol::ANGULAR
                }
                _ => false,
            }
        }
        _ => a == b || (within(a, b) && within(b, a)),
    }
}

/// A piece of a swept or spline surface's parameters that stands for the whole of it: a spline's
/// own domain, a turn of a revolution, a ruled surface's unit square, and of an extrusion its
/// profile carried a way along. `None` for a surface with a closed form, and for one whose profile
/// is a line without ends.
fn sample_window(s: &Surface) -> Option<(Vec2, Vec2)> {
    let ends = |c: &Curve| {
        let (lo, hi) = c.domain();
        (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
    };
    Some(match s {
        Surface::Extrusion(e) => {
            let (lo, hi) = ends(&e.curve)?;
            let reach = e.curve.bbox(lo, hi).diagonal().max(1.0);
            (Vec2::new(lo, 0.0), Vec2::new(hi, reach))
        }
        Surface::Revolution(r) => {
            let (lo, hi) = ends(&r.curve)?;
            (Vec2::new(0.0, lo), Vec2::new(TAU, hi))
        }
        Surface::Ruled(_) => (Vec2::ZERO, Vec2::new(1.0, 1.0)),
        Surface::Nurbs(ns) => {
            let ((u0, u1), (v0, v1)) = ns.domain();
            (Vec2::new(u0, v0), Vec2::new(u1, v1))
        }
        _ => return None,
    })
}

/// Whether surface `a`, over the piece of its parameters that stands for it, lies on `b` with the
/// same normal line: the test [`same_domain_faces`] makes over two faces' own windows. Two swept
/// surfaces on one point set are seldom equal part for part: the same profile slid along its
/// extrusion, or turned about its axis, draws the same surface from another start.
fn within(a: &Surface, b: &Surface) -> bool {
    let Some((lo, hi)) = sample_window(a) else {
        return false;
    };
    let n = 6;
    (0..=n).all(|i| {
        (0..=n).all(|j| {
            let u = lo.x + (hi.x - lo.x) * i as f64 / n as f64;
            let v = lo.y + (hi.y - lo.y) * j as f64 / n as f64;
            let p = a.eval(u, v);
            let q = b.project(p);
            b.eval(q.x, q.y).dist(p) <= tol::LINEAR
                && a.normal(u, v).cross(b.normal(q.x, q.y)).norm() <= tol::TANGENT
        })
    })
}

fn analytic(s: &Surface) -> bool {
    matches!(
        s,
        Surface::Plane(_)
            | Surface::Cylinder(_)
            | Surface::Cone(_)
            | Surface::Sphere(_)
            | Surface::Torus(_)
    )
}

/// Whether two faces' surfaces are the same point set where the faces are. Analytic pairs are
/// decided in closed form by [`same_domain`]; any other pair (swept, ruled or spline surfaces in
/// different frames or parameterisations, or one of them against an analytic surface) by checking
/// that one face's parameter window lies on the other's surface, with the same normal line, within
/// the tolerance.
pub(crate) fn same_domain_faces(a: &Domain, b: &Domain) -> bool {
    if same_domain(&a.surface, &b.surface) {
        return true;
    }
    // Two planes a hair off parallel are one plane over faces small enough that they stay within
    // the tolerance of each other: a part placed by a rounded pose sits on its seat that way, and
    // their meeting line, at the angle between them, would run anywhere along the band.
    let planes = matches!(
        (&a.surface, &b.surface),
        (Surface::Plane(_), Surface::Plane(_))
    );
    if analytic(&a.surface) && analytic(&b.surface) && !planes {
        return false;
    }
    // One face lying on the other's surface is enough: they can overlap only where it is. A
    // spline surface ends where its face does, so the other face, larger, does not lie on it.
    let n = 6;
    [(a, b), (b, a)].iter().any(|(from, to)| {
        (0..=n).all(|i| {
            (0..=n).all(|j| {
                let u = from.lo.x + (from.hi.x - from.lo.x) * i as f64 / n as f64;
                let v = from.lo.y + (from.hi.y - from.lo.y) * j as f64 / n as f64;
                let p = from.surface.eval(u, v);
                let q = to.surface.project(p);
                to.surface.eval(q.x, q.y).dist(p) <= tol::LINEAR
                    && from
                        .surface
                        .normal(u, v)
                        .cross(to.surface.normal(q.x, q.y))
                        .norm()
                        <= tol::TANGENT
            })
        })
    })
}

/// The part of the line `o + t·d` inside `region`, as a parameter range.
pub(crate) fn clip_line(o: Point3, d: Vec3, region: &Aabb) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        let (ok, dk) = (o[k], d[k]);
        let (mn, mx) = (region.min[k], region.max[k]);
        if dk.abs() < 1e-300 {
            if ok < mn || ok > mx {
                return None;
            }
            continue;
        }
        let (a, b) = ((mn - ok) / dk, (mx - ok) / dk);
        lo = lo.max(a.min(b));
        hi = hi.min(a.max(b));
    }
    (hi > lo).then_some((lo, hi))
}

fn line_curve(o: Point3, d: Vec3, region: &Aabb, tangent: bool) -> Option<SsiCurve> {
    let (t0, t1) = clip_line(o, d, region)?;
    Some(SsiCurve {
        curve: Curve::Line(Line { origin: o, dir: d }),
        range: (t0, t1),
        closed: false,
        tangent,
    })
}

fn circle_curve(frame: Frame, radius: f64, tangent: bool) -> SsiCurve {
    SsiCurve {
        curve: Curve::Circle(Circle { frame, radius }),
        range: (0.0, TAU),
        closed: true,
        tangent,
    }
}

fn frame_at(origin: Point3, z: Vec3, x_hint: Vec3) -> Frame {
    Frame::new(origin, z, x_hint)
        .or_else(|_| Frame::from_axis(origin, z))
        .expect("axis is a unit vector")
}

fn plane_plane(a: &Frame, b: &Frame, region: &Aabb) -> Ssi {
    let (n1, n2) = (a.z, b.z);
    let d = n1.cross(n2);
    if d.norm() <= tol::ANGULAR {
        return none();
    }
    let (h1, h2) = (n1.dot(a.origin), n2.dot(b.origin));
    // n1·(h1 − h2·c)/(1 − c²) + n2·(h2 − h1·c)/(1 − c²), arranged so planes a hair off parallel do
    // not take the difference of two large numbers: at an angle θ, 1 − c² is θ², and that form's
    // rounding put the point θ⁻² · 1e-16 off both planes, 1e-5 mm at θ = 1.5e-5.
    let p0 = (n2 * h1 - n1 * h2).cross(d) * (1.0 / d.dot(d));
    let dir = d.unit().expect("planes are not parallel");
    Ssi::Curves {
        curves: line_curve(p0, dir, region, false).into_iter().collect(),
        points: Vec::new(),
    }
}

fn plane_cylinder(p: &Frame, c: &crate::geom::Cylinder, region: &Aabb) -> Ssi {
    let (n, a, r) = (p.z, c.frame.z, c.radius);
    let an = a.dot(n);
    if parallel(a, n) {
        let h = (p.origin - c.frame.origin).dot(a);
        let fr = Frame {
            origin: c.frame.origin + a * h,
            ..c.frame
        };
        return Ssi::Curves {
            curves: vec![circle_curve(fr, r, false)],
            points: Vec::new(),
        };
    }
    if an.abs() <= tol::ANGULAR {
        let delta = (c.frame.origin - p.origin).dot(n);
        let foot = c.frame.origin - n * delta;
        if delta.abs() > r + tol::LINEAR {
            return none();
        }
        let curves = if delta.abs() >= r - tol::LINEAR {
            // Midway between the plane and the cylinder's nearest line, so within the tolerance
            // of both however the gap falls in the band.
            let mid = c.frame.origin - n * (0.5 * (delta + delta.signum() * r));
            line_curve(mid, a, region, true).into_iter().collect()
        } else {
            let w = a.cross(n).unit().expect("axis lies along the plane");
            let s = ((r - delta.abs()) * (r + delta.abs())).sqrt();
            [foot + w * s, foot - w * s]
                .into_iter()
                .filter_map(|q| line_curve(q, a, region, false))
                .collect()
        };
        return Ssi::Curves {
            curves,
            points: Vec::new(),
        };
    }
    if an.abs() <= NEARLY_ALONG
        && let Some(curves) = nearly_lines(p, c, region)
    {
        return Ssi::Curves {
            curves,
            points: Vec::new(),
        };
    }
    let t = (p.origin - c.frame.origin).dot(n) / an;
    let centre = c.frame.origin + a * t;
    let w = a.cross(n).unit().expect("axis is oblique to the plane");
    let m = n.cross(w);
    let frame = Frame {
        origin: centre,
        x: m,
        y: n.cross(m),
        z: n,
    };
    Ssi::Curves {
        curves: vec![SsiCurve {
            curve: Curve::Ellipse(Ellipse {
                frame,
                major: r / an.abs(),
                minor: r,
            }),
            range: (0.0, TAU),
            closed: true,
            tangent: false,
        }],
        points: Vec::new(),
    }
}

/// How far off lying along a plane, in radians, a cylinder's axis may be for [`nearly_lines`] to
/// try lines for their crossing.
const NEARLY_ALONG: f64 = 1e-6;

/// The plane `p` across the cylinder `c` whose axis lies a hair off along it, as two lines through
/// `region`, where those lines lie within a tenth of the tolerance of the crossing: an axis turned
/// 6e-10 off by a rounded pose makes it an ellipse 2.8e9 mm long for a radius of 1.6, whose centre,
/// 1e9 mm away, rounds its points by the tolerance. `None` where they do not, or where the plane
/// touches the cylinder or misses it somewhere across the region.
fn nearly_lines(p: &Frame, c: &crate::geom::Cylinder, region: &Aabb) -> Option<Vec<SsiCurve>> {
    let (n, a, r) = (p.z, c.frame.z, c.radius);
    let o = c.frame.origin;
    let (ax, ay) = (c.frame.x.dot(n), c.frame.y.dot(n));
    let rho = ax.hypot(ay);
    let phase = ay.atan2(ax);
    let lift = (o - p.origin).dot(n);
    let corners = (0..8).map(|k| {
        Point3::new(
            if k & 1 == 0 {
                region.min.x
            } else {
                region.max.x
            },
            if k & 2 == 0 {
                region.min.y
            } else {
                region.max.y
            },
            if k & 4 == 0 {
                region.min.z
            } else {
                region.max.z
            },
        )
    });
    let (lo, hi) = corners.fold((f64::MAX, f64::MIN), |(lo, hi), q| {
        let t = (q - o).dot(a);
        (lo.min(t), hi.max(t))
    });
    if lo >= hi {
        return None;
    }
    // Where along the axis at `t` the cylinder's two points on the plane are: the angle round it
    // whose cosine against the plane's normal makes up the rest of the plane's equation.
    let at = |t: f64, side: f64| -> Option<Point3> {
        let k = -(lift + a.dot(n) * t) / (r * rho);
        if k.abs() >= 1.0 - 1e-9 {
            return None;
        }
        let theta = phase + side * k.acos();
        Some(o + a * t + (c.frame.x * theta.cos() + c.frame.y * theta.sin()) * r)
    };
    let mut out = Vec::new();
    for side in [1.0, -1.0] {
        let (q0, q1, qm) = (at(lo, side)?, at(hi, side)?, at(0.5 * (lo + hi), side)?);
        let dir = (q1 - q0).unit()?;
        let off = (qm - q0) - dir * (qm - q0).dot(dir);
        if off.norm() > 0.1 * tol::LINEAR {
            return None;
        }
        out.extend(line_curve(q0, dir, region, false));
    }
    Some(out)
}

fn plane_sphere(p: &Frame, s: &crate::geom::Sphere) -> Ssi {
    let n = p.z;
    let delta = (s.frame.origin - p.origin).dot(n);
    let r = s.radius;
    let foot = s.frame.origin - n * delta;
    if delta.abs() > r + tol::LINEAR {
        none()
    } else if delta.abs() >= r - tol::LINEAR {
        Ssi::Curves {
            curves: Vec::new(),
            points: vec![s.frame.origin - n * (0.5 * (delta + delta.signum() * r))],
        }
    } else {
        let rho = ((r - delta.abs()) * (r + delta.abs())).sqrt();
        Ssi::Curves {
            curves: vec![circle_curve(frame_at(foot, n, s.frame.x), rho, false)],
            points: Vec::new(),
        }
    }
}

/// A meridian of a surface of revolution, in the half plane `(ρ, z)` about an axis.
#[derive(Clone, Copy, Debug)]
enum Meridian {
    /// Through `p` along the unit `d`.
    Line {
        p: [f64; 2],
        d: [f64; 2],
    },
    Circle {
        c: [f64; 2],
        r: f64,
    },
}

fn meridian(s: &Surface, axis: Axis) -> Option<Meridian> {
    let z_of = |p: Point3| (p - axis.origin).dot(axis.dir);
    match s {
        Surface::Cylinder(c) => Some(Meridian::Line {
            p: [c.radius, z_of(c.frame.origin)],
            d: [0.0, 1.0],
        }),
        Surface::Cone(c) => {
            let (sa, ca) = c.half_angle.sin_cos();
            let flip = if c.frame.z.dot(axis.dir) > 0.0 {
                1.0
            } else {
                -1.0
            };
            Some(Meridian::Line {
                p: [c.radius, z_of(c.frame.origin)],
                d: [sa, ca * flip],
            })
        }
        Surface::Sphere(sp) => Some(Meridian::Circle {
            c: [0.0, z_of(sp.frame.origin)],
            r: sp.radius,
        }),
        Surface::Torus(t) => Some(Meridian::Circle {
            c: [t.major, z_of(t.frame.origin)],
            r: t.minor,
        }),
        _ => None,
    }
}

/// The meridians' common points, each with whether they touch there; `None` when they coincide.
fn meridian_points(a: Meridian, b: Meridian) -> Option<Vec<([f64; 2], bool)>> {
    let cross = |u: [f64; 2], v: [f64; 2]| u[0] * v[1] - u[1] * v[0];
    let sub = |u: [f64; 2], v: [f64; 2]| [u[0] - v[0], u[1] - v[1]];
    let line_circle = |p: [f64; 2], d: [f64; 2], c: [f64; 2], r: f64| {
        let w = sub(p, c);
        let t_near = -(w[0] * d[0] + w[1] * d[1]);
        let q = [p[0] + d[0] * t_near, p[1] + d[1] * t_near];
        let m = sub(q, c);
        let m = m[0].hypot(m[1]);
        if m > r + tol::LINEAR {
            Vec::new()
        } else if m >= r - tol::LINEAR {
            vec![(q, true)]
        } else {
            let h = ((r - m) * (r + m)).sqrt();
            vec![
                ([q[0] + d[0] * h, q[1] + d[1] * h], false),
                ([q[0] - d[0] * h, q[1] - d[1] * h], false),
            ]
        }
    };
    match (a, b) {
        (Meridian::Line { p: p1, d: d1 }, Meridian::Line { p: p2, d: d2 }) => {
            let den = cross(d1, d2);
            let w = sub(p2, p1);
            if den.abs() <= tol::ANGULAR {
                return if cross(w, d1).abs() <= tol::LINEAR {
                    None
                } else {
                    Some(Vec::new())
                };
            }
            let t = cross(w, d2) / den;
            Some(vec![([p1[0] + d1[0] * t, p1[1] + d1[1] * t], false)])
        }
        (Meridian::Line { p, d }, Meridian::Circle { c, r })
        | (Meridian::Circle { c, r }, Meridian::Line { p, d }) => Some(line_circle(p, d, c, r)),
        (Meridian::Circle { c: c1, r: r1 }, Meridian::Circle { c: c2, r: r2 }) => {
            let w = sub(c2, c1);
            let d = w[0].hypot(w[1]);
            if d <= tol::LINEAR {
                return if (r1 - r2).abs() <= tol::LINEAR {
                    None
                } else {
                    Some(Vec::new())
                };
            }
            if d > r1 + r2 + tol::LINEAR || d < (r1 - r2).abs() - tol::LINEAR {
                return Some(Vec::new());
            }
            let ex = [w[0] / d, w[1] / d];
            let ey = [-ex[1], ex[0]];
            let x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d);
            let touching =
                (d - (r1 + r2)).abs() <= tol::LINEAR || (d - (r1 - r2).abs()).abs() <= tol::LINEAR;
            if touching {
                return Some(vec![([c1[0] + ex[0] * x, c1[1] + ex[1] * x], true)]);
            }
            let y = (r1 * r1 - x * x).max(0.0).sqrt();
            Some(vec![
                (
                    [c1[0] + ex[0] * x + ey[0] * y, c1[1] + ex[1] * x + ey[1] * y],
                    false,
                ),
                (
                    [c1[0] + ex[0] * x - ey[0] * y, c1[1] + ex[1] * x - ey[1] * y],
                    false,
                ),
            ])
        }
    }
}

/// Two surfaces of revolution about one axis meet in circles about it.
fn coaxial(a: &Surface, b: &Surface, axis: Axis) -> Option<Ssi> {
    let (ma, mb) = (meridian(a, axis)?, meridian(b, axis)?);
    let Some(pts) = meridian_points(ma, mb) else {
        return Some(Ssi::Coincident);
    };
    let x_hint = match a {
        Surface::Cylinder(c) => c.frame.x,
        Surface::Cone(c) => c.frame.x,
        Surface::Torus(t) => t.frame.x,
        Surface::Sphere(s) => s.frame.x,
        _ => axis.dir.any_perpendicular(),
    };
    Some(circles_about(axis, x_hint, pts))
}

/// The circles about `axis` through meridian points; a point on the axis is a single point, and a
/// point at negative `ρ` (a cone's other nappe) is the circle of its distance.
fn circles_about(axis: Axis, x_hint: Vec3, pts: Vec<([f64; 2], bool)>) -> Ssi {
    let mut curves: Vec<SsiCurve> = Vec::new();
    let mut points = Vec::new();
    let mut seen: Vec<(f64, f64)> = Vec::new();
    for ([rho, z], touching) in pts {
        let rho = rho.abs();
        if seen
            .iter()
            .any(|&(r, h)| (r - rho).abs() <= tol::LINEAR && (h - z).abs() <= tol::LINEAR)
        {
            continue;
        }
        seen.push((rho, z));
        let centre = axis.origin + axis.dir * z;
        if rho <= tol::LINEAR {
            points.push(centre);
        } else {
            curves.push(circle_curve(
                frame_at(centre, axis.dir, x_hint),
                rho,
                touching,
            ));
        }
    }
    Ssi::Curves { curves, points }
}

/// Two cylinders of one radius whose axes cross meet in two ellipses, in the planes bisecting
/// the axes, which touch where the common perpendicular meets the cylinders.
fn equal_crossing_cylinders(o: Point3, a1: Vec3, a2: Vec3, r: f64) -> Ssi {
    let nhat = a1.cross(a2).unit().expect("axes are not parallel");
    let mut curves = Vec::new();
    for m in [a1 - a2, a1 + a2] {
        let m = m.unit().expect("axes are not parallel");
        let e = m.cross(nhat);
        let frame = Frame {
            origin: o,
            x: e,
            y: m.cross(e),
            z: m,
        };
        curves.push(SsiCurve {
            curve: Curve::Ellipse(Ellipse {
                frame,
                major: r / e.cross(a1).norm(),
                minor: r,
            }),
            range: (0.0, TAU),
            closed: true,
            tangent: false,
        });
    }
    Ssi::Curves {
        curves,
        points: vec![o + nhat * r, o - nhat * r],
    }
}

/// The closest points of two lines, as parameters along each; `None` when they are parallel.
fn closest(o1: Point3, d1: Vec3, o2: Point3, d2: Vec3) -> Option<(f64, f64)> {
    let w = o1 - o2;
    let c = d1.dot(d2);
    let den = 1.0 - c * c;
    if den.sqrt() <= tol::ANGULAR {
        return None;
    }
    let (d, e) = (d1.dot(w), d2.dot(w));
    Some(((c * e - d) / den, (e - c * d) / den))
}

fn cylinder_cylinder(
    a: &crate::geom::Cylinder,
    b: &crate::geom::Cylinder,
    region: &Aabb,
) -> Option<Ssi> {
    let (a1, a2) = (a.frame.z, b.frame.z);
    if parallel(a1, a2) {
        // Parallel axes: the circles of a cross-section meet at points along the axis.
        let w = b.frame.origin - a.frame.origin;
        let off = w - a1 * w.dot(a1);
        let d = off.norm();
        let (ra, rb) = (a.radius, b.radius);
        if d > ra + rb + tol::LINEAR || d < (ra - rb).abs() - tol::LINEAR {
            return Some(none());
        }
        let ex = off / d;
        let ey = a1.cross(ex);
        let x = (d * d + ra * ra - rb * rb) / (2.0 * d);
        let touching =
            (d - (ra + rb)).abs() <= tol::LINEAR || (d - (ra - rb).abs()).abs() <= tol::LINEAR;
        let feet: Vec<Point3> = if touching {
            vec![a.frame.origin + ex * x]
        } else {
            let y = (ra * ra - x * x).max(0.0).sqrt();
            vec![
                a.frame.origin + ex * x + ey * y,
                a.frame.origin + ex * x - ey * y,
            ]
        };
        return Some(Ssi::Curves {
            curves: feet
                .into_iter()
                .filter_map(|q| line_curve(q, a1, region, touching))
                .collect(),
            points: Vec::new(),
        });
    }
    if (a.radius - b.radius).abs() <= tol::LINEAR {
        let (s, t) = closest(a.frame.origin, a1, b.frame.origin, a2)?;
        let (p, q) = (a.frame.origin + a1 * s, b.frame.origin + a2 * t);
        // About the axes' midpoint with the mean radius, the ellipses lie within half the radii's
        // and the axes' gaps of each surface: within half the tolerance, which their meetings with
        // other curves need on top of it. Past that the generators sweep (`sweep.rs`).
        if p.dist(q) + (a.radius - b.radius).abs() <= tol::LINEAR {
            let r = 0.5 * (a.radius + b.radius);
            return Some(equal_crossing_cylinders(p.lerp(q, 0.5), a1, a2, r));
        }
    }
    None
}

/// The intersection of `a` and `b` within `region`. `seeds` gives points on both surfaces where
/// tracing may start, for the pairs with no closed form.
pub(crate) fn intersect(
    a: &Surface,
    b: &Surface,
    region: &Aabb,
    seeds: &mut dyn FnMut() -> Vec<Point3>,
) -> Result<Ssi> {
    if same_domain(a, b) {
        return Ok(Ssi::Coincident);
    }
    if let Some(r) = closed_form(a, b, region) {
        return Ok(r);
    }
    if let Some(r) = closed_form(b, a, region) {
        return Ok(r);
    }
    let (curves, points) = march::trace(a, b, region, &seeds())?;
    Ok(Ssi::Curves { curves, points })
}

/// Whether moving along `dir` leaves `s` unchanged.
fn translation_invariant(s: &Surface, dir: Vec3) -> bool {
    match s {
        Surface::Plane(p) => p.frame.z.dot(dir).abs() <= tol::ANGULAR,
        Surface::Cylinder(c) => parallel(c.frame.z, dir),
        Surface::Extrusion(e) => parallel(e.dir, dir),
        _ => false,
    }
}

/// Whether turning about `axis` leaves `s` unchanged.
fn rotation_invariant(s: &Surface, axis: Axis) -> bool {
    match s {
        Surface::Plane(p) => parallel(p.frame.z, axis.dir),
        Surface::Sphere(sp) => line_distance(sp.frame.origin, axis.origin, axis.dir) <= tol::LINEAR,
        Surface::Cylinder(_) | Surface::Cone(_) | Surface::Torus(_) => {
            axis_of(s).is_some_and(|x| same_axis(x, axis))
        }
        Surface::Revolution(r) => same_axis(
            Axis {
                origin: r.origin,
                dir: r.axis,
            },
            axis,
        ),
        _ => false,
    }
}

fn tangent_at(a: &Surface, b: &Surface, p: Point3) -> bool {
    let (ua, ub) = (a.project(p), b.project(p));
    a.normal(ua.x, ua.y).cross(b.normal(ub.x, ub.y)).norm() <= tol::TANGENT
}

/// `c` carried along `dir` onto the plane through `o` square to `n`: a parallel projection, which
/// is affine, so a spline's image is the spline of its control points' images and a circle's or
/// ellipse's is the ellipse on its two images of conjugate radii. A curve lying level with the
/// plane is moved whole, and keeps its parameter.
fn projected_along(c: &Curve, dir: Vec3, o: Point3, n: Vec3) -> Option<Curve> {
    let rate = dir.dot(n);
    let onto = |q: Point3| q + dir * ((o - q).dot(n) / rate);
    let flat = |v: Vec3| v - dir * (v.dot(n) / rate);
    let (d0, d1) = c.domain();
    let start = if d0.is_finite() { d0 } else { 0.0 };
    let shift = onto(c.eval(start)) - c.eval(start);
    let level = match c {
        Curve::Line(l) => l.dir.dot(n).abs() <= tol::ANGULAR,
        Curve::Circle(k) => parallel(k.frame.z, n),
        Curve::Ellipse(k) => parallel(k.frame.z, n),
        _ => {
            let off = |t: f64| (c.eval(t) - c.eval(start)).dot(n).abs() <= tol::LINEAR;
            (0..=32).all(|i| off(d0 + (d1 - d0) * i as f64 / 32.0))
                && c.extremes(n, d0, d1).into_iter().all(off)
        }
    };
    if level {
        return Some(c.transformed(&crate::math::Transform::translation(shift)));
    }
    let conic = |centre: Point3, a: Vec3, b: Vec3| -> Option<Curve> {
        // The two radii of the image that are conjugate and square to each other.
        let t0 = 0.5 * (2.0 * a.dot(b)).atan2(a.dot(a) - b.dot(b));
        let (s, k) = t0.sin_cos();
        let (u, w) = (a * k + b * s, b * k - a * s);
        let (u, w) = if u.norm() >= w.norm() {
            (u, w)
        } else {
            (w, -u)
        };
        let frame = Frame::new(centre, u.cross(w), u).ok()?;
        Some(Curve::Ellipse(Ellipse {
            frame,
            major: u.norm(),
            minor: w.norm(),
        }))
    };
    match c {
        Curve::Line(l) => Some(Curve::Line(Line {
            origin: onto(l.origin),
            dir: flat(l.dir).unit()?,
        })),
        Curve::Circle(k) => conic(
            onto(k.frame.origin),
            flat(k.frame.x * k.radius),
            flat(k.frame.y * k.radius),
        ),
        Curve::Ellipse(k) => conic(
            onto(k.frame.origin),
            flat(k.frame.x * k.major),
            flat(k.frame.y * k.minor),
        ),
        Curve::BSpline(s) => {
            let ctrl = s.control_points().iter().map(|&q| onto(q)).collect();
            let image = if s.is_periodic() {
                crate::geom::BSpline::periodic(s.degree(), s.knots().to_vec(), ctrl).ok()?
            } else {
                crate::geom::BSpline::new(s.degree(), s.knots().to_vec(), ctrl).ok()?
            };
            Some(Curve::BSpline(image))
        }
        Curve::Nurbs(s) => {
            let ctrl = s.control_points().iter().map(|&q| onto(q)).collect();
            let w = s.weights().map(|w| w.to_vec());
            let image = crate::geom::NurbsCurve::new(s.degree(), s.knots().to_vec(), ctrl, w);
            Some(Curve::Nurbs(Box::new(image.ok()?)))
        }
        Curve::Intersection(_) => None,
    }
}

/// An extrusion or revolution against a surface that shares its symmetry: the profile curve's
/// meetings with the other surface, swept the same way, are lines or circles; a plane across an
/// extrusion cuts it in its curve carried along the extrusion onto the plane.
fn swept(a: &Surface, b: &Surface, region: &Aabb) -> Option<Ssi> {
    match a {
        Surface::Extrusion(e) => {
            if let Surface::Plane(p) = b
                && parallel(p.frame.z, e.dir)
            {
                let curve = projected_along(&e.curve, e.dir, p.frame.origin, p.frame.z)?;
                let range = curve.domain();
                let closed = curve.period().is_some();
                return Some(Ssi::Curves {
                    curves: vec![SsiCurve {
                        curve,
                        range,
                        closed,
                        tangent: false,
                    }],
                    points: Vec::new(),
                });
            }
            if !translation_invariant(b, e.dir) {
                return None;
            }
            let hits = super::csi::curve_surface(&e.curve, e.curve.domain(), b);
            if hits.on_surface {
                return Some(Ssi::Coincident);
            }
            let curves = hits
                .ts
                .into_iter()
                .map(|t| e.curve.eval(t))
                .filter_map(|q| line_curve(q, e.dir, region, tangent_at(a, b, q)))
                .collect();
            Some(Ssi::Curves {
                curves,
                points: Vec::new(),
            })
        }
        Surface::Revolution(r) => {
            let axis = Axis {
                origin: r.origin,
                dir: r.axis,
            };
            if !rotation_invariant(b, axis) {
                return None;
            }
            let hits = super::csi::curve_surface(&r.curve, r.curve.domain(), b);
            if hits.on_surface {
                return Some(Ssi::Coincident);
            }
            let pts = hits
                .ts
                .into_iter()
                .map(|t| {
                    let q = r.curve.eval(t);
                    let z = (q - axis.origin).dot(axis.dir);
                    (
                        [line_distance(q, axis.origin, axis.dir), z],
                        tangent_at(a, b, q),
                    )
                })
                .collect();
            Some(circles_about(axis, axis.dir.any_perpendicular(), pts))
        }
        _ => None,
    }
}

fn closed_form(a: &Surface, b: &Surface, region: &Aabb) -> Option<Ssi> {
    if let Some(r) = swept(a, b, region) {
        return Some(r);
    }
    match (a, b) {
        (Surface::Plane(p), Surface::Plane(q)) => {
            return Some(plane_plane(&p.frame, &q.frame, region));
        }
        (Surface::Plane(p), Surface::Cylinder(c)) => {
            return Some(plane_cylinder(&p.frame, c, region));
        }
        (Surface::Plane(p), Surface::Sphere(s)) => return Some(plane_sphere(&p.frame, s)),
        (Surface::Plane(p), Surface::Torus(t)) => {
            if let Some(r) = sweep::plane_through_axis(t, p) {
                return Some(r);
            }
            let axis = axis_of(b)?;
            if parallel(p.frame.z, axis.dir) {
                let h = (p.frame.origin - axis.origin).dot(axis.dir);
                let plane_meridian = Meridian::Line {
                    p: [0.0, h],
                    d: [1.0, 0.0],
                };
                let pts = meridian_points(plane_meridian, meridian(b, axis)?)?;
                return Some(circles_about(axis, axis.dir.any_perpendicular(), pts));
            }
            return sweep::intersect(a, b, region);
        }
        (Surface::Plane(p), Surface::Cone(c)) => {
            let axis = axis_of(b)?;
            if parallel(p.frame.z, axis.dir) {
                // A plane across the axis is a degenerate surface of revolution about it.
                let h = (p.frame.origin - axis.origin).dot(axis.dir);
                let plane_meridian = Meridian::Line {
                    p: [0.0, h],
                    d: [1.0, 0.0],
                };
                let pts = meridian_points(plane_meridian, meridian(b, axis)?)?;
                return Some(circles_about(axis, axis.dir.any_perpendicular(), pts));
            }
            return conic::plane_cone(p, c, region);
        }
        (Surface::Cylinder(x), Surface::Cylinder(y)) => {
            return cylinder_cylinder(x, y, region).or_else(|| sweep::intersect(a, b, region));
        }
        _ => {}
    }
    let (xa, xb) = (axis_of(a)?, axis_of(b)?);
    let symmetric = match (a, b) {
        (Surface::Sphere(sa), Surface::Sphere(sb)) => {
            let d = sb.frame.origin - sa.frame.origin;
            let dir = d.unit()?;
            coaxial(
                a,
                b,
                Axis {
                    origin: sa.frame.origin,
                    dir,
                },
            )
        }
        (Surface::Sphere(sp), _) => (line_distance(sp.frame.origin, xb.origin, xb.dir)
            <= tol::LINEAR)
            .then(|| coaxial(a, b, xb))
            .flatten(),
        (_, Surface::Sphere(sp)) => (line_distance(sp.frame.origin, xa.origin, xa.dir)
            <= tol::LINEAR)
            .then(|| coaxial(a, b, xa))
            .flatten(),
        _ => same_axis(xa, xb).then(|| coaxial(a, b, xa)).flatten(),
    };
    symmetric.or_else(|| sweep::intersect(a, b, region))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Cone, Cylinder, Plane, Sphere, Torus};

    /// A closed cubic in the plane z = 0, as a profile for swept surfaces.
    fn wavy() -> Curve {
        let n = 8;
        let mut ctrl: Vec<Vec3> = (0..n)
            .map(|k| {
                let a = TAU * k as f64 / n as f64;
                let r = if k % 2 == 0 { 4.0 } else { 6.0 };
                Vec3::new(10.0 + r * a.cos(), r * a.sin(), 0.0)
            })
            .collect();
        ctrl.extend_from_within(..3);
        let knots = (0..n + 7).map(|i| i as f64 - 3.0).collect();
        Curve::BSpline(crate::geom::BSpline::periodic_cubic(knots, ctrl).unwrap())
    }

    // A circle of radius 2 tilted 30° from level, extruded straight down, is cut by a level plane in
    // its shadow: the ellipse of semi-axes 2 and 2 cos 30°, on the plane and on the extrusion.
    #[test]
    fn a_level_plane_cuts_an_extrusion_of_a_tilted_circle_in_its_shadow() {
        let tilt = 30f64.to_radians();
        let frame = Frame::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, -tilt.sin(), tilt.cos()),
            Vec3::X,
        )
        .unwrap();
        let ext = Surface::Extrusion(Box::new(crate::geom::Extrusion {
            curve: Curve::Circle(Circle { frame, radius: 2.0 }),
            dir: -Vec3::Z,
        }));
        let plane = Surface::Plane(Plane {
            frame: Frame::WORLD.translated(Vec3::new(0.0, 0.0, -5.0)),
        });
        let Some(Ssi::Curves { curves, .. }) = swept(&ext, &plane, &Aabb::EMPTY) else {
            panic!("an extrusion square to a plane meets it in a curve");
        };
        let Curve::Ellipse(e) = &curves[0].curve else {
            panic!("a tilted circle's shadow is an ellipse");
        };
        assert!((e.major - 2.0).abs() < 1e-12 && (e.minor - 2.0 * tilt.cos()).abs() < 1e-12);
        for k in 0..16 {
            let p = curves[0].curve.eval(TAU * k as f64 / 16.0);
            let uv = ext.project(p);
            assert!((p.z + 5.0).abs() < 1e-12 && ext.eval(uv.x, uv.y).dist(p) < 1e-12);
        }
    }

    #[test]
    fn swept_surfaces_drawn_from_other_starts_are_the_same_surface() {
        use crate::geom::{Extrusion, Revolution};
        use crate::math::Transform;
        let profile = wavy();
        let extruded =
            |curve: Curve, dir: Vec3| Surface::Extrusion(Box::new(Extrusion { curve, dir }));
        let along = Transform::translation(Vec3::new(0.0, 0.0, 3.0));
        let aside = Transform::translation(Vec3::new(0.01, 0.0, 0.0));
        let a = extruded(profile.clone(), Vec3::Z);
        // The profile slid along the extrusion, and the extrusion run the other way: one surface.
        assert!(same_domain(
            &a,
            &extruded(profile.transformed(&along), Vec3::Z)
        ));
        assert!(same_domain(&a, &extruded(profile.clone(), -Vec3::Z)));
        assert!(!same_domain(
            &a,
            &extruded(profile.transformed(&aside), Vec3::Z)
        ));

        // The profile in the plane y = 0, turned about z: turned a way round first, one surface.
        let upright = profile.transformed(&Transform::rotation(Vec3::ZERO, Vec3::X, TAU / 4.0));
        let revolved = |curve: Curve| {
            Surface::Revolution(Box::new(Revolution {
                curve,
                origin: Vec3::ZERO,
                axis: Vec3::Z,
            }))
        };
        let turned = Transform::rotation(Vec3::ZERO, Vec3::Z, 1.0);
        let b = revolved(upright.clone());
        assert!(same_domain(&b, &revolved(upright.transformed(&turned))));
        assert!(!same_domain(&b, &revolved(upright.transformed(&aside))));
        assert!(!same_domain(&a, &b));
    }

    fn region() -> Aabb {
        Aabb {
            min: Vec3::new(-50.0, -50.0, -50.0),
            max: Vec3::new(50.0, 50.0, 50.0),
        }
    }

    fn plane(o: [f64; 3], n: [f64; 3]) -> Surface {
        let n = Vec3::from_array(n);
        Surface::Plane(Plane {
            frame: Frame::from_axis(Vec3::from_array(o), n).unwrap(),
        })
    }

    fn cyl(o: [f64; 3], z: [f64; 3], r: f64) -> Surface {
        Surface::Cylinder(Cylinder {
            frame: Frame::from_axis(Vec3::from_array(o), Vec3::from_array(z)).unwrap(),
            radius: r,
        })
    }

    fn curves(a: &Surface, b: &Surface) -> (Vec<SsiCurve>, Vec<Point3>) {
        match intersect(a, b, &region(), &mut Vec::new).unwrap() {
            Ssi::Curves { curves, points } => (curves, points),
            Ssi::Coincident => panic!("coincident"),
        }
    }

    /// Every curve lies on both surfaces.
    fn on_both(a: &Surface, b: &Surface, cs: &[SsiCurve]) {
        for c in cs {
            for i in 0..=16 {
                let t = c.range.0 + (c.range.1 - c.range.0) * i as f64 / 16.0;
                let p = c.curve.eval(t);
                for s in [a, b] {
                    let d = super::super::csi::signed_distance(s, p).abs();
                    assert!(d < 1e-9, "{} off {} by {d}", c.curve.kind(), s.kind());
                }
            }
        }
    }

    #[test]
    fn plane_pairs_have_closed_forms() {
        let p = plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]);
        let q = plane([0.0, 2.0, 0.0], [0.0, 1.0, 1.0]);
        let (cs, _) = curves(&p, &q);
        assert_eq!(cs.len(), 1);
        on_both(&p, &q, &cs);
        let c_perp = cyl([1.0, 1.0, -5.0], [0.0, 0.0, 1.0], 3.0);
        let (cs, _) = curves(&p, &c_perp);
        assert!(matches!(cs[0].curve, Curve::Circle(_)));
        on_both(&p, &c_perp, &cs);
        let c_oblique = cyl([0.0, 0.0, 0.0], [0.3, 0.2, 1.0], 3.0);
        let (cs, _) = curves(&p, &c_oblique);
        assert!(matches!(cs[0].curve, Curve::Ellipse(_)));
        on_both(&p, &c_oblique, &cs);
        let c_along = cyl([0.0, 0.0, 3.0], [1.0, 0.0, 0.0], 2.5);
        let (cs, _) = curves(&p, &c_along);
        assert_eq!(cs.len(), 2);
        on_both(&p, &c_along, &cs);
        let sphere = Surface::Sphere(Sphere {
            frame: Frame::new(Vec3::new(0.0, 0.0, 4.0), Vec3::Z, Vec3::X).unwrap(),
            radius: 5.0,
        });
        let (cs, _) = curves(&p, &sphere);
        assert!(matches!(&cs[0].curve, Curve::Circle(c) if (c.radius - 4.0).abs() < 1e-12));
        let cone = Surface::Cone(Cone {
            frame: Frame::WORLD,
            radius: 2.0,
            half_angle: 0.3,
        });
        let (cs, _) = curves(&p, &cone);
        assert_eq!(cs.len(), 1);
        on_both(&p, &cone, &cs);
        let torus = Surface::Torus(Torus {
            frame: Frame::WORLD,
            major: 6.0,
            minor: 2.0,
        });
        let (cs, _) = curves(&p, &torus);
        assert_eq!(cs.len(), 2);
        on_both(&p, &torus, &cs);
    }

    #[test]
    fn tangent_plane_and_cylinder_touch_along_a_line() {
        let p = plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let c = cyl([5.0, 20.0, 0.0], [0.0, 0.0, 1.0], 5.0);
        let (cs, _) = curves(&p, &c);
        assert_eq!(cs.len(), 1);
        assert!(cs[0].tangent);
        let past = cyl([4.999999, 20.0, 0.0], [0.0, 0.0, 1.0], 5.0);
        let (cs, _) = curves(&p, &past);
        assert_eq!(cs.len(), 2);
        assert!(cs.iter().all(|c| !c.tangent));
        let short = cyl([5.000001, 20.0, 0.0], [0.0, 0.0, 1.0], 5.0);
        assert!(curves(&p, &short).0.is_empty());
    }

    #[test]
    fn coaxial_and_parallel_pairs_meet_in_circles_and_lines() {
        let c = cyl([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 3.0);
        let sphere = Surface::Sphere(Sphere {
            frame: Frame::WORLD,
            radius: 5.0,
        });
        let (cs, _) = curves(&c, &sphere);
        assert_eq!(cs.len(), 2);
        on_both(&c, &sphere, &cs);
        let cone = Surface::Cone(Cone {
            frame: Frame::WORLD,
            radius: 1.0,
            half_angle: 0.4,
        });
        let (cs, _) = curves(&c, &cone);
        assert_eq!(cs.len(), 1);
        on_both(&c, &cone, &cs);
        let d = cyl([4.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0);
        let (cs, _) = curves(&c, &d);
        assert_eq!(cs.len(), 2);
        on_both(&c, &d, &cs);
        let same = cyl([0.0, 0.0, 7.0], [0.0, 0.0, -1.0], 3.0);
        assert!(matches!(
            intersect(&c, &same, &region(), &mut Vec::new).unwrap(),
            Ssi::Coincident
        ));
    }

    #[test]
    fn equal_crossing_cylinders_meet_in_two_ellipses_that_touch() {
        let a = cyl([-20.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0);
        let b = cyl([0.0, -20.0, 0.0], [0.0, 1.0, 0.0], 5.0);
        let (cs, pts) = curves(&a, &b);
        assert_eq!(cs.len(), 2);
        on_both(&a, &b, &cs);
        assert_eq!(pts.len(), 2);
        for p in pts {
            assert!((p.z.abs() - 5.0).abs() < 1e-12 && p.x.abs() < 1e-12);
        }
    }

    fn spline_extrusion() -> Surface {
        let pts: Vec<Vec3> = (0..12)
            .map(|i| {
                let t = TAU * i as f64 / 12.0;
                let r = if i % 2 == 0 { 10.0 } else { 9.0 };
                Vec3::new(r * t.cos(), r * t.sin(), 0.0)
            })
            .collect();
        Surface::Extrusion(Box::new(crate::geom::Extrusion {
            curve: Curve::BSpline(crate::geom::BSpline::interpolate_periodic(&pts).unwrap()),
            dir: Vec3::Z,
        }))
    }

    #[test]
    fn extrusions_meet_planes_and_parallel_cylinders_in_closed_form() {
        let e = spline_extrusion();
        let across = plane([0.0, 0.0, 3.0], [0.0, 0.0, 1.0]);
        let (cs, _) = curves(&e, &across);
        assert_eq!(cs.len(), 1);
        assert!(cs[0].closed && matches!(cs[0].curve, Curve::BSpline(_)));
        on_both(&e, &across, &cs);
        let along = plane([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let (cs, _) = curves(&e, &along);
        assert_eq!(cs.len(), 2);
        assert!(cs.iter().all(|c| matches!(c.curve, Curve::Line(_))));
        on_both(&e, &along, &cs);
        let rod = cyl([9.5, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0);
        let (cs, _) = curves(&rod, &e);
        assert_eq!(cs.len(), 2);
        on_both(&rod, &e, &cs);
    }

    #[test]
    fn revolutions_meet_coaxial_surfaces_in_circles() {
        let profile = Curve::BSpline(
            crate::geom::BSpline::interpolate(&[
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(6.0, 0.0, 2.0),
                Vec3::new(5.0, 0.0, 4.0),
                Vec3::new(7.0, 0.0, 6.0),
            ])
            .unwrap(),
        );
        let r = Surface::Revolution(Box::new(crate::geom::Revolution {
            curve: profile,
            origin: Vec3::ZERO,
            axis: Vec3::Z,
        }));
        let across = plane([0.0, 0.0, 3.0], [0.0, 0.0, 1.0]);
        let (cs, _) = curves(&r, &across);
        assert_eq!(cs.len(), 1);
        on_both(&r, &across, &cs);
        let c = cyl([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.5);
        let (cs, _) = curves(&c, &r);
        assert!(cs.len() >= 2);
        assert!(cs.iter().all(|c| matches!(c.curve, Curve::Circle(_))));
        on_both(&c, &r, &cs);
    }

    #[test]
    fn spheres_meet_in_a_circle_or_touch_at_a_point() {
        let a = Surface::Sphere(Sphere {
            frame: Frame::WORLD,
            radius: 5.0,
        });
        let b = Surface::Sphere(Sphere {
            frame: Frame::new(Vec3::new(4.0, 3.0, 0.0), Vec3::Z, Vec3::X).unwrap(),
            radius: 4.0,
        });
        let (cs, _) = curves(&a, &b);
        assert_eq!(cs.len(), 1);
        on_both(&a, &b, &cs);
        let touching = Surface::Sphere(Sphere {
            frame: Frame::new(Vec3::new(8.0, 0.0, 0.0), Vec3::Z, Vec3::X).unwrap(),
            radius: 3.0,
        });
        let (cs, pts) = curves(&a, &touching);
        assert!(cs.is_empty());
        assert!(pts.len() == 1 && pts[0].dist(Vec3::new(5.0, 0.0, 0.0)) < 1e-12);
        let tangent_plane = plane([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]);
        let (cs, pts) = curves(&a, &tangent_plane);
        assert!(cs.is_empty());
        assert_eq!(pts.len(), 1);
    }

    #[test]
    fn crossing_cylinders_of_unequal_radii_are_traced() {
        let a = cyl([-20.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0);
        let b = cyl([0.0, 0.0, -20.0], [0.0, 0.0, 1.0], 3.0);
        let seeds = vec![Vec3::new(0.0, 3.0, 4.0), Vec3::new(0.0, 3.0, -4.0)];
        let region = Aabb {
            min: Vec3::new(-4.0, -6.0, -6.0),
            max: Vec3::new(4.0, 6.0, 6.0),
        };
        let Ssi::Curves { curves, .. } = intersect(&a, &b, &region, &mut || seeds.clone()).unwrap()
        else {
            panic!("coincident");
        };
        assert_eq!(curves.len(), 2);
        assert!(curves.iter().all(|c| c.closed));
        for c in &curves {
            for i in 0..=32 {
                let p = c
                    .curve
                    .eval(c.range.0 + (c.range.1 - c.range.0) * i as f64 / 32.0);
                for s in [&a, &b] {
                    assert!(super::super::csi::signed_distance(s, p).abs() <= tol::LINEAR);
                }
            }
        }
    }
}
