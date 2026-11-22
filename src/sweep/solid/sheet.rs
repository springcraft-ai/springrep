//! What a curve sweeps under a steady motion, exactly where the motion has a closed form: an
//! extrusion under a slide, a revolution under a turn, and under a screw the rational B-spline whose
//! columns follow its control points' helices.

use std::f64::consts::{PI, TAU};

use super::motion::Motion;
use crate::{
    Error, Result,
    geom::{
        Circle, Cone, Curve, Cylinder, Extrusion, Line, NurbsCurve, Plane, Revolution, Sphere,
        Surface, Torus,
    },
    loft::rail::{Rail, point},
    math::{Frame, Point3, Transform, Vec2, Vec3},
    sweep::helix::{self, Helix},
    tol,
};

/// A slide, a turn or a screw, each run forward over a positive span.
pub(super) enum Steady {
    Slide {
        by: Vec3,
    },
    /// Turned by `from + τ·angle` about the unit `axis` through `origin`.
    Turn {
        origin: Point3,
        axis: Vec3,
        from: f64,
        angle: f64,
    },
    /// Carried to turn `from + τ·angle` of `screw`, whose axis is a unit vector.
    Screw {
        screw: Helix,
        from: f64,
        angle: f64,
    },
}

/// The face a curve sweeps: its surface, and where on it the curve lies at each step of the
/// motion. The point the motion carries the curve's point at `t` to after `s`, `s` from 0 to
/// `span`, is at `origin + s·along + t·across` in the surface's parameters, or where `exact` puts
/// it. `affine` says the curve's copies are parameter lines on which `t` runs affinely, and
/// `along_affine` the same of the paths and `s`.
pub(super) struct Sheet {
    pub surface: Surface,
    pub span: f64,
    pub origin: Vec2,
    pub along: Vec2,
    pub across: Vec2,
    pub affine: bool,
    pub along_affine: bool,
    pub exact: Option<Box<dyn Fn(f64, f64) -> Vec2>>,
}

impl Sheet {
    pub(super) fn uv(&self, s: f64, t: f64) -> Vec2 {
        match &self.exact {
            Some(f) => f(s, t),
            None => self.origin + self.along * s + self.across * t,
        }
    }

    fn affine(surface: Surface, span: f64, along: Vec2, across: Vec2, origin: Vec2) -> Sheet {
        Sheet {
            surface,
            span,
            origin,
            along,
            across,
            affine: true,
            along_affine: true,
            exact: None,
        }
    }
}

/// Whether `curve` over `range` lies in one plane square to the unit `dir`: every point at one
/// height along it.
fn lies_square_to(curve: &Curve, range: (f64, f64), dir: Vec3) -> bool {
    match curve {
        Curve::Line(l) => l.dir.dot(dir).abs() <= tol::ANGULAR,
        Curve::Circle(c) => c.frame.z.cross(dir).norm() <= tol::ANGULAR,
        Curve::Ellipse(c) => c.frame.z.cross(dir).norm() <= tol::ANGULAR,
        _ => {
            let h0 = curve.eval(range.0).dot(dir);
            let off = |t: f64| (curve.eval(t).dot(dir) - h0).abs() <= tol::LINEAR;
            (0..=32).all(|k| off(range.0 + (range.1 - range.0) * k as f64 / 32.0))
                && curve.extremes(dir, range.0, range.1).into_iter().all(off)
        }
    }
}

/// The unit normal of a plane holding the unit `dir` and all of `curve` over `range`, where there
/// is one and the curve is not a line along `dir`.
fn plane_along(curve: &Curve, range: (f64, f64), dir: Vec3) -> Option<Vec3> {
    let normal = match curve {
        Curve::Line(_) => return None,
        Curve::Circle(c) => c.frame.z,
        Curve::Ellipse(c) => c.frame.z,
        _ => {
            // Square to the slide and to the chord from the start to the point farthest across it.
            let p0 = curve.eval(range.0);
            let across = |p: Point3| (p - p0) - dir * (p - p0).dot(dir);
            let far = (1..=32)
                .map(|k| curve.eval(range.0 + (range.1 - range.0) * k as f64 / 32.0))
                .max_by(|a, b| across(*a).norm().total_cmp(&across(*b).norm()))?;
            across(far).cross(dir).unit()?
        }
    };
    if normal.dot(dir).abs() > tol::ANGULAR {
        return None;
    }
    let p0 = curve.eval(range.0);
    let off = |t: f64| (curve.eval(t) - p0).dot(normal).abs() <= tol::LINEAR;
    ((0..=32).all(|k| off(range.0 + (range.1 - range.0) * k as f64 / 32.0))
        && curve
            .extremes(normal, range.0, range.1)
            .into_iter()
            .all(off))
    .then_some(normal)
}

/// What `curve` sweeps turning through `span` about the unit `axis` through `o`: a cylinder,
/// cone or plane where the curve is a line in a plane through the axis, a torus or sphere where it
/// is a circle in one, else the revolution of the curve. On each, the angle turned is `u`. The
/// surface follows from the whole curve's `range`, so the pieces of one edge share it (those on one
/// side of the axis, for a line through it); `piece` is the part of it this sheet's face is swept
/// from.
fn turned(
    curve: &Curve,
    (range, piece): ((f64, f64), (f64, f64)),
    o: Point3,
    axis: Vec3,
    span: f64,
) -> Result<Sheet> {
    let revolution = || {
        Sheet::affine(
            Surface::Revolution(Box::new(Revolution {
                curve: curve.clone(),
                origin: o,
                axis,
            })),
            span,
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::ZERO,
        )
    };
    let height = |p: Point3| (p - o).dot(axis);
    let radial = |p: Point3| (p - o) - axis * height(p);
    // A curve level with the turn turns in its own plane.
    if lies_square_to(curve, range, axis) {
        let at = curve.eval(range.0);
        let plane = Frame::new(o + axis * height(at), axis, axis.any_perpendicular())?;
        let c = curve.clone();
        return Ok(Sheet {
            surface: Surface::Plane(Plane { frame: plane }),
            span,
            origin: Vec2::ZERO,
            along: Vec2::ZERO,
            across: Vec2::ZERO,
            affine: false,
            along_affine: false,
            exact: Some(Box::new(move |s, t| {
                let q = plane.to_local(Transform::rotation(o, axis, s).point(c.eval(t)));
                Vec2::new(q.x, q.y)
            })),
        });
    }
    let still = Helix {
        origin: o,
        axis,
        rise: 0.0,
        sign: 1.0,
    };
    if !matches!(curve, Curve::Line(_))
        && let Some(sheet) = coaxial(curve, range, &still, span)?
    {
        return Ok(sheet);
    }
    match curve {
        Curve::Line(l) => {
            if l.dir.cross(axis).dot(l.origin - o).abs() > tol::LINEAR {
                let still = Helix {
                    origin: o,
                    axis,
                    rise: 0.0,
                    sign: 1.0,
                };
                return screwed(curve, range, &still, span);
            }
            // The plane through the axis and the line, its x pointing out to the line.
            let far = [range.0, range.1, 0.5 * (range.0 + range.1)]
                .into_iter()
                .map(|t| radial(l.origin + l.dir * t))
                .max_by(|a, b| a.norm().total_cmp(&b.norm()))
                .expect("three points");
            let Some(x) = far.unit() else {
                return Ok(revolution());
            };
            let frame = Frame::new(o, axis, x)?;
            let (dr, dh) = (l.dir.dot(x), l.dir.dot(axis));
            let (rho, h) = (radial(l.origin).dot(x), height(l.origin));
            if dr.abs() <= tol::ANGULAR {
                let surface = Surface::Cylinder(Cylinder { frame, radius: rho });
                return Ok(Sheet::affine(
                    surface,
                    span,
                    Vec2::new(1.0, 0.0),
                    Vec2::new(0.0, dh),
                    Vec2::new(0.0, h),
                ));
            }
            // The cone through the end of the line farthest out on the piece's side of the axis,
            // `v` the distance along the line from it. A line through the axis sweeps two nappes;
            // the piece's lies where the cone's radius is positive, as its closed form takes it.
            let side = radial(l.origin + l.dir * (0.5 * (piece.0 + piece.1)));
            let at = if radial(l.origin + l.dir * range.0).dot(side)
                >= radial(l.origin + l.dir * range.1).dot(side)
            {
                range.0
            } else {
                range.1
            };
            let end = l.origin + l.dir * at;
            let Some(x) = radial(end).unit() else {
                return Ok(revolution());
            };
            let cone = Cone {
                frame: Frame::new(o + axis * height(end), axis, x)?,
                radius: radial(end).norm(),
                half_angle: l.dir.dot(x).atan2(dh),
            };
            Ok(Sheet::affine(
                Surface::Cone(cone),
                span,
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, 1.0),
                Vec2::new(0.0, -at),
            ))
        }
        Curve::Circle(c) => {
            let in_plane = c.frame.z.dot(axis).abs() <= tol::ANGULAR
                && (c.frame.origin - o).dot(c.frame.z).abs() <= tol::LINEAR;
            if !in_plane {
                return Ok(match spherical(c, (range, piece), o, axis, span)? {
                    Some(sheet) => sheet,
                    None => revolution(),
                });
            }
            let r = radial(c.frame.origin);
            let rho = r.norm();
            let x = if rho > tol::LINEAR {
                r / rho
            } else {
                c.frame.z.cross(axis)
            };
            let frame = Frame::new(o + axis * height(c.frame.origin), axis, x)?;
            // The circle's angle `t` is the tube's angle `phi0 + sign·t`.
            let (cx, cy) = (c.frame.x, c.frame.y);
            let phi0 = cx.dot(axis).atan2(cx.dot(x));
            let sign = (cx.dot(x) * cy.dot(axis) - cx.dot(axis) * cy.dot(x)).signum();
            if rho > tol::LINEAR {
                let surface = Surface::Torus(Torus {
                    frame,
                    major: rho,
                    minor: c.radius,
                });
                return Ok(Sheet::affine(
                    surface,
                    span,
                    Vec2::new(1.0, 0.0),
                    Vec2::new(0.0, sign),
                    Vec2::new(0.0, phi0),
                ));
            }
            // About its own centre a circle sweeps a sphere. Where the piece stays on the side of
            // the axis `x` points to, the latitude reads the circle's angle; on the other side the
            // longitude is half a turn on and the latitude reads the angle back from a half turn.
            let (lo, hi) = (phi0 + sign * piece.0, phi0 + sign * piece.1);
            let half = std::f64::consts::FRAC_PI_2;
            let shift = (0.5 * (lo + hi) / TAU).round() * TAU;
            let near = lo.min(hi) - shift >= -half - 1e-12 && lo.max(hi) - shift <= half + 1e-12;
            let back = (0.5 * (lo + hi) / TAU - 0.5).round() * TAU + PI;
            let far = lo.min(hi) - back >= -half - 1e-12 && lo.max(hi) - back <= half + 1e-12;
            let surface = Surface::Sphere(Sphere {
                frame,
                radius: c.radius,
            });
            let (u0, flip, at) = match (near, far) {
                (true, _) => (0.0, 1.0, shift),
                (_, true) => (PI, -1.0, back),
                _ => return Ok(revolution()),
            };
            Ok(Sheet::affine(
                surface,
                span,
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, flip * sign),
                Vec2::new(u0, flip * (phi0 - at)),
            ))
        }
        _ => Ok(revolution()),
    }
}

/// What the circle `c` sweeps turning through `span` about the unit `axis` through `o` where the
/// circle's own axis meets the turn's at a point `m`: every point of the circle lies at one distance
/// from `m`, so it sweeps the sphere about `m` through it. Its revolution would draw that sphere
/// twice, out to the circle's points farthest from the axis and back, and a point of it would have
/// two parameters. `None` where the axes do not meet, or where the circle over `piece` reaches the
/// axis or crosses the longitude half a turn from the middle of `range`, where its longitude would
/// not be one smooth reading.
fn spherical(
    c: &Circle,
    (range, piece): ((f64, f64), (f64, f64)),
    o: Point3,
    axis: Vec3,
    span: f64,
) -> Result<Option<Sheet>> {
    let n = c.frame.z;
    let b = n.dot(axis);
    let denom = 1.0 - b * b;
    if n.cross(axis).norm() <= tol::ANGULAR {
        return Ok(None);
    }
    let w = c.frame.origin - o;
    let (d, e) = (n.dot(w), axis.dot(w));
    let m = o + axis * ((e - b * d) / denom);
    if (c.frame.origin + n * ((b * e - d) / denom)).dist(m) > tol::LINEAR {
        return Ok(None);
    }
    let circle = Curve::Circle(c.clone());
    let mid = circle.eval(0.5 * (range.0 + range.1)) - m;
    let frame = Frame::new(m, axis, mid - axis * mid.dot(axis))?;
    let radius = circle.eval(range.0).dist(m);
    let reading = move |t: f64| {
        let q = frame.to_local(circle.eval(t));
        let rho = q.x.hypot(q.y);
        (q.y.atan2(q.x), q.z.atan2(rho), rho)
    };
    let clear = (0..=64).all(|k| {
        let (lon, _, rho) = reading(piece.0 + (piece.1 - piece.0) * k as f64 / 64.0);
        rho > tol::LINEAR && lon.abs() < PI - 1e-3
    });
    if !clear {
        return Ok(None);
    }
    Ok(Some(Sheet {
        surface: Surface::Sphere(Sphere { frame, radius }),
        span,
        origin: Vec2::ZERO,
        along: Vec2::new(1.0, 0.0),
        across: Vec2::ZERO,
        affine: false,
        along_affine: true,
        exact: Some(Box::new(move |s, t| {
            let (lon, lat, _) = reading(t);
            Vec2::new(lon + s, lat)
        })),
    }))
}

impl Steady {
    pub(super) fn of(m: &Motion) -> Option<Steady> {
        match *m {
            Motion::Slide(by) => Some(Steady::Slide { by }),
            Motion::Turn {
                origin,
                axis,
                from,
                to,
            } => {
                let k = axis.unit()?;
                Some(if to >= from {
                    Steady::Turn {
                        origin,
                        axis: k,
                        from,
                        angle: to - from,
                    }
                } else {
                    Steady::Turn {
                        origin,
                        axis: -k,
                        from: -from,
                        angle: from - to,
                    }
                })
            }
            Motion::Screw { screw, from, to } => {
                let k = screw.axis.unit()?;
                let s = Helix { axis: k, ..screw };
                Some(if to >= from {
                    Steady::Screw {
                        screw: s,
                        from,
                        angle: to - from,
                    }
                } else {
                    Steady::Screw {
                        screw: Helix {
                            sign: -s.sign,
                            rise: -s.rise,
                            ..s
                        },
                        from: -from,
                        angle: from - to,
                    }
                })
            }
            _ => None,
        }
    }

    pub(super) fn pose(&self, tau: f64) -> Transform {
        match self {
            Steady::Slide { by } => Transform::translation(*by * tau),
            Steady::Turn {
                origin,
                axis,
                from,
                angle,
            } => Transform::rotation(*origin, *axis, from + angle * tau),
            Steady::Screw { screw, from, angle } => screw.transform(from + angle * tau),
        }
    }

    pub(super) fn pose0(&self) -> Transform {
        self.pose(0.0)
    }

    pub(super) fn pose1(&self) -> Transform {
        self.pose(1.0)
    }

    /// How fast the point at `x` moves where the motion starts, per unit of `τ`.
    pub(super) fn velocity(&self, x: Point3) -> Vec3 {
        match self {
            Steady::Slide { by } => *by,
            Steady::Turn {
                origin,
                axis,
                angle,
                ..
            } => axis.cross(x - *origin) * *angle,
            Steady::Screw { screw, angle, .. } => {
                (screw.axis.cross(x - screw.origin) * screw.sign + screw.axis * screw.rise) * *angle
            }
        }
    }

    /// Where the paths are cut, in the motion's own parameter: their ends, and each whole turn of
    /// a screw between them.
    pub(super) fn stops(&self) -> Vec<f64> {
        let span = self.span();
        let mut out = vec![0.0];
        if let Steady::Screw { .. } = self {
            let mut s = TAU;
            while s < span - tol::ANGULAR {
                out.push(s);
                s += TAU;
            }
        }
        out.push(span);
        out
    }

    /// The line the motion holds still: a turn's axis.
    pub(super) fn held(&self) -> Option<(Point3, Vec3)> {
        match self {
            Steady::Turn { origin, axis, .. } => Some((*origin, *axis)),
            _ => None,
        }
    }

    /// Whether the motion holds the body's point `p` still: a point on a turn's axis.
    pub(super) fn holds(&self, p: Point3) -> bool {
        match self {
            Steady::Turn { origin, axis, .. } => (p - *origin).cross(*axis).norm() <= tol::LINEAR,
            _ => false,
        }
    }

    /// The span of the motion's own parameter on its paths and sheets: the slide's length, or the
    /// angle turned.
    fn span(&self) -> f64 {
        match self {
            Steady::Slide { by } => by.norm(),
            Steady::Turn { angle, .. } | Steady::Screw { angle, .. } => *angle,
        }
    }

    /// The path of the point at `p` where the motion starts, over `[0, span]`.
    pub(super) fn path(&self, p: Point3) -> Result<Curve> {
        match self {
            Steady::Slide { by } => Ok(Curve::Line(Line {
                origin: p,
                dir: by
                    .unit()
                    .ok_or_else(|| Error::InvalidInput("a slide of no length".into()))?,
            })),
            Steady::Turn { origin, axis, .. } => {
                let foot = *origin + *axis * (p - *origin).dot(*axis);
                Ok(Curve::Circle(Circle {
                    frame: Frame::new(foot, *axis, p - foot)?,
                    radius: p.dist(foot),
                }))
            }
            // Turns of one screw add, so the path from where the motion starts is the screw's own
            // from turn 0.
            Steady::Screw { screw, angle, .. } => {
                let q = p - screw.origin;
                let radius = (q - screw.axis * q.dot(screw.axis)).norm();
                let nodes = helix::nodes(*angle, radius);
                let column = helix::column(screw, p, &nodes);
                Ok(Curve::Nurbs(Box::new(NurbsCurve::new(
                    helix::DEGREE,
                    helix::knots(&nodes),
                    column,
                    None,
                )?)))
            }
        }
    }

    /// What `curve`, where the motion starts, sweeps over its `range`.
    /// `piece` is the part of the range whose face the sheet is for.
    pub(super) fn sheet(
        &self,
        curve: &Curve,
        range: (f64, f64),
        piece: (f64, f64),
    ) -> Result<Sheet> {
        let span = self.span();
        match self {
            Steady::Slide { by } => {
                let dir = by
                    .unit()
                    .ok_or_else(|| Error::InvalidInput("a slide of no length".into()))?;
                // A line slides in a plane, and a circle along its axis in a cylinder.
                match curve {
                    Curve::Line(l) => {
                        let frame = Frame::new(l.origin, l.dir.cross(dir), l.dir)?;
                        return Ok(Sheet {
                            surface: Surface::Plane(Plane { frame }),
                            span,
                            origin: Vec2::ZERO,
                            along: Vec2::new(dir.dot(frame.x), dir.dot(frame.y)),
                            across: Vec2::new(1.0, 0.0),
                            affine: true,
                            along_affine: true,
                            exact: None,
                        });
                    }
                    Curve::Circle(c) if c.frame.z.cross(dir).norm() <= tol::ANGULAR => {
                        let turn = c.frame.z.dot(dir).signum();
                        let frame = Frame::new(c.frame.origin, dir, c.frame.x)?;
                        return Ok(Sheet {
                            surface: Surface::Cylinder(Cylinder {
                                frame,
                                radius: c.radius,
                            }),
                            span,
                            origin: Vec2::ZERO,
                            along: Vec2::new(0.0, 1.0),
                            across: Vec2::new(turn, 0.0),
                            affine: true,
                            along_affine: true,
                            exact: None,
                        });
                    }
                    _ => {}
                }
                // A curve in a plane that holds the slide slides within that plane.
                if let Some(normal) = plane_along(curve, range, dir) {
                    let at = curve.eval(range.0);
                    let plane = Frame::new(at, normal, dir)?;
                    let c = curve.clone();
                    return Ok(Sheet {
                        surface: Surface::Plane(Plane { frame: plane }),
                        span,
                        origin: Vec2::ZERO,
                        along: Vec2::new(1.0, 0.0),
                        across: Vec2::ZERO,
                        affine: false,
                        along_affine: true,
                        exact: Some(Box::new(move |s, t| {
                            let q = plane.to_local(c.eval(t) + dir * s);
                            Vec2::new(q.x, q.y)
                        })),
                    });
                }
                Ok(Sheet {
                    surface: Surface::Extrusion(Box::new(Extrusion {
                        curve: curve.clone(),
                        dir,
                    })),
                    span,
                    origin: Vec2::ZERO,
                    along: Vec2::new(0.0, 1.0),
                    across: Vec2::new(1.0, 0.0),
                    affine: true,
                    along_affine: true,
                    exact: None,
                })
            }
            Steady::Turn { origin, axis, .. } => {
                turned(curve, (range, piece), *origin, *axis, span)
            }
            Steady::Screw { screw, angle, .. } => screwed(curve, range, screw, *angle),
        }
    }
}

/// What `curve` sweeps under `screw` through `angle` where it lies at one distance from the
/// screw's axis, as an edge cut by a cylinder about it does: a strip of that cylinder, its angle the
/// curve's own plus the turn and its height the curve's own plus the rise.
fn coaxial(curve: &Curve, range: (f64, f64), screw: &Helix, angle: f64) -> Result<Option<Sheet>> {
    let (o, axis) = (screw.origin, screw.axis);
    let radial = |p: Point3| {
        let q = p - o;
        q - axis * q.dot(axis)
    };
    let at = |k: usize| curve.eval(range.0 + (range.1 - range.0) * k as f64 / 32.0);
    let radius = radial(at(0)).norm();
    let level = |p: Point3| (radial(p).norm() - radius).abs() <= tol::LINEAR;
    if radius <= tol::LINEAR || !(0..=32).all(|k| level(at(k))) {
        return Ok(None);
    }
    // Angles from the curve's middle, which no curve within a half turn either side of it
    // carries across the reading's jump.
    let frame = Frame::new(o, axis, radial(at(16)))?;
    let c = curve.clone();
    let (sign, rise) = (screw.sign, screw.rise);
    Ok(Some(Sheet {
        surface: Surface::Cylinder(Cylinder { frame, radius }),
        span: angle,
        origin: Vec2::ZERO,
        along: Vec2::new(sign, rise),
        across: Vec2::ZERO,
        affine: false,
        along_affine: true,
        exact: Some(Box::new(move |s, t| {
            let q = frame.to_local(c.eval(t));
            Vec2::new(q.y.atan2(q.x) + sign * s, q.z + rise * s)
        })),
    }))
}

/// What `curve` sweeps over its `range` carried through `angle` of `screw` from where it is: the
/// rational B-spline whose columns follow the helices of its control points, within `SCREW_FIT`
/// of them, `u` the turn and `v` the curve's rational parameter on `[0, 1]`.
fn screwed(curve: &Curve, range: (f64, f64), screw: &Helix, angle: f64) -> Result<Sheet> {
    // A line along the axis, a thread's crest, sweeps the exact cylinder, so faces two such lines
    // sweep at one radius are on one surface.
    if let Curve::Line(l) = curve
        && l.dir.cross(screw.axis).norm() <= tol::ANGULAR
    {
        let q = l.origin - screw.origin;
        let height = q.dot(screw.axis);
        let out = q - screw.axis * height;
        if out.norm() > tol::LINEAR {
            let frame = Frame::new(screw.origin, screw.axis, out)?;
            return Ok(Sheet::affine(
                Surface::Cylinder(Cylinder {
                    frame,
                    radius: out.norm(),
                }),
                angle,
                Vec2::new(screw.sign, screw.rise),
                Vec2::new(0.0, l.dir.dot(screw.axis)),
                Vec2::new(0.0, height),
            ));
        }
    }
    if let Some(sheet) = coaxial(curve, range, screw, angle)? {
        return Ok(sheet);
    }
    let rail = Rail::of_curve(curve, range.0, range.1)?;
    let surface = Surface::Nurbs(Box::new(helix::sheet(&rail, screw, angle)?));
    let w: Vec<f64> = rail.pw.iter().map(|h| h[3]).collect();
    let width = range.1 - range.0;
    // A spline's rail runs with its own parameter; a conic's rational parameter does not run with
    // its angle, so a point's is found on the rail.
    let conic = matches!(curve, Curve::Circle(_) | Curve::Ellipse(_));
    let exact: Option<Box<dyn Fn(f64, f64) -> Vec2>> = if conic {
        let (ctrl, w) = (rail.pw.iter().map(|&h| point(h)).collect(), Some(w));
        let on = Curve::Nurbs(Box::new(NurbsCurve::new(
            rail.degree,
            rail.knots.clone(),
            ctrl,
            w,
        )?));
        let c = curve.clone();
        let closed = rail.start().dist(rail.end()) <= tol::LINEAR;
        // Round a whole conic its two ends are one point: the reading nearer the angle's share of
        // the range is the one along the rail.
        Some(Box::new(move |s, t| {
            let v = on.project(c.eval(t), 0.0, 1.0);
            let share = (t - range.0) / width;
            let v = if closed && (v - share).abs() > 0.5 {
                v + (share - v).signum()
            } else {
                v
            };
            Vec2::new(s, v)
        }))
    } else {
        None
    };
    Ok(Sheet {
        surface,
        span: angle,
        origin: Vec2::new(0.0, -range.0 / width),
        along: Vec2::new(1.0, 0.0),
        across: Vec2::new(0.0, 1.0 / width),
        affine: !conic,
        along_affine: true,
        exact,
    })
}
