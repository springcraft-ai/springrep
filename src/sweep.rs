//! Sweeps of planar profiles into solids: straight, turned about an axis, along a path of legs
//! and bends, along a helix, and along a space curve. Every side face takes the names of the
//! profile segment that sweeps it; caps are `start` (on the sketch plane) and `end`. A whole solid
//! moved rigidly sweeps the solid [`solid`] builds.

mod curve;
mod helix;
mod miter;
mod solid;
mod until;

pub use curve::along_curve;
pub(crate) use helix::SCREW_FIT;
#[cfg(test)]
pub(crate) use helix::sheet as screw_sheet;
pub use helix::{Helix, helix};
pub use solid::{Motion, solid};
pub use until::{Meet, extrude_to_next, revolve_to_next};

use std::f64::consts::TAU;

use crate::{
    Error, Result, boolean, check,
    error::invalid,
    geom::{Cone, Curve, Cylinder, Extrusion, Line, Plane, Revolution, Sphere, Surface, Torus},
    math::{Frame, Point3, Transform, Vec2, Vec3},
    profile::{
        PlacedSegment, Profile,
        region::{Op, Region},
    },
    tol,
    topo::{Brep, CoedgeSpec, EdgeId, FaceId, FaceName, PCurve, VertexId},
};

/// One leg of a sweep's path, from where the last ended, heading along the carried frame's z.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Leg {
    /// Straight ahead by `length`.
    Line { length: f64 },
    /// A bend of `radius` through `angle`, right-handed about `axis`, a direction square to the
    /// heading: the path turns toward `axis × heading`.
    Arc { radius: f64, angle: f64, axis: Vec3 },
    /// A turn through `angle` on the spot, right-handed about `axis`, with no bend: the legs
    /// either side run to the corner, and the part of the profile outside the turn is turned
    /// about the corner to fill it. The profile must be bounded by lines and arcs.
    Turn { angle: f64, axis: Vec3 },
    /// A sharp corner through `angle` on the spot, right-handed about `axis`, mitered: the
    /// straight legs either side run on past the corner and are cut by, and meet in, the plane
    /// through it that bisects the turn. Both neighbouring legs must be straight, and the turn
    /// less than half a turn. A path whose last leg is a miter that brings it back to its start,
    /// heading as it started, is closed: that miter joins the last straight leg to the first.
    Miter { angle: f64, axis: Vec3 },
}

/// A turn's axis as a unit vector, which must be square to the heading.
fn turn_axis(axis: Vec3, fr: &Frame) -> Result<Vec3> {
    match axis.unit().filter(|n| n.dot(fr.z).abs() <= tol::ANGULAR) {
        Some(n) => Ok(n),
        None => invalid("a bend's axis must be square to the path"),
    }
}

/// What fills a sharp turn about `n` at `fr`'s origin: the part of the profile on the outside of
/// the turn, turned through it. The part inside sweeps nothing the legs either side do not hold.
fn sharp_turn(
    profile: &Profile,
    fr: &Frame,
    n: Vec3,
    angle: f64,
    feature: &str,
) -> Result<Vec<Brep>> {
    let face = Region::of(profile)?;
    let local = |v: Vec3| Vec2::new(v.dot(fr.x), v.dot(fr.y));
    let along = local(n);
    let outward = local(fr.z.cross(n));
    let mut reach: f64 = 1.0;
    for s in face.segs() {
        for t in [0.0, 0.5, 1.0] {
            reach = reach.max(s.geom.at(t).norm());
        }
    }
    let l = 4.0 * reach;
    let half = Region::polygon(&[
        along * -l,
        along * l,
        along * l + outward * l,
        along * -l + outward * l,
    ]);
    let outside = face.combine(&half, Op::Intersect)?;
    if outside.loops.is_empty() {
        return Ok(Vec::new());
    }
    let parts = outside.profiles()?;
    parts
        .iter()
        .map(|p| revolve(p, fr, fr.origin, n, angle, feature))
        .collect()
}

/// `profile`, drawn on `frame`'s xy plane, carried along `legs` from the frame's origin, starting
/// along its z and staying square to the path: an extrusion for each straight leg, a revolution
/// for each bend and the fill of each sharp turn, joined into one solid. A straight leg beside a
/// miter is extruded on past the corner and cut back to the miter's plane.
pub fn along(profile: &Profile, frame: &Frame, legs: &[Leg], feature: &str) -> Result<Brep> {
    let path = miter::Path::of(profile, frame, legs)?;
    let mut whole: Option<Brep> = None;
    for (i, leg) in legs.iter().enumerate() {
        let fr = path.frames[i];
        let pieces = match *leg {
            Leg::Line { length } => vec![path.line(profile, i, length, feature)?],
            Leg::Arc {
                radius,
                angle,
                axis,
            } => {
                let n = turn_axis(axis, &fr)?;
                let centre = fr.origin + n.cross(fr.z) * radius;
                vec![revolve(profile, &fr, centre, n, angle, feature)?]
            }
            Leg::Turn { angle, axis } => {
                let n = turn_axis(axis, &fr)?;
                sharp_turn(profile, &fr, n, angle, feature)?
            }
            Leg::Miter { .. } => Vec::new(),
        };
        for piece in pieces {
            whole = Some(match whole {
                None => piece,
                Some(w) => boolean::union(&w, &piece)?,
            });
        }
    }
    match whole {
        Some(w) => Ok(w),
        None => invalid("a sweep's path needs a leg"),
    }
}

fn affine(origin: Vec2, dir: Vec2) -> PCurve {
    PCurve::Affine { origin, dir }
}

fn plane_uv(frame: &Frame, p: Point3) -> Vec2 {
    let l = frame.to_local(p);
    Vec2::new(l.x, l.y)
}

/// The pcurve of `curve` on the plane of `frame`, starting at `start`.
fn plane_pcurve(frame: &Frame, curve: &Curve, start: Point3) -> PCurve {
    match curve {
        Curve::Line(l) => affine(
            plane_uv(frame, l.origin),
            Vec2::new(l.dir.dot(frame.x), l.dir.dot(frame.y)),
        ),
        _ => PCurve::Projected {
            seed: plane_uv(frame, start),
        },
    }
}

fn coedge(edge: EdgeId, reversed: bool, pcurve: PCurve) -> CoedgeSpec {
    CoedgeSpec {
        edge,
        reversed,
        pcurve,
    }
}

/// The parameters a placed segment starts and ends at, in the loop's direction.
fn traversal(s: &PlacedSegment) -> (f64, f64) {
    if s.forward {
        s.range
    } else {
        (s.range.1, s.range.0)
    }
}

fn finish(b: Brep) -> Result<Brep> {
    let issues = check::check(&b);
    if issues.is_empty() {
        Ok(b)
    } else {
        Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.to_string()).collect(),
        ))
    }
}

struct PlacedLoop {
    segs: Vec<(PlacedSegment, crate::profile::Names)>,
}

fn place(profile: &Profile, frame: &Frame) -> Vec<PlacedLoop> {
    profile
        .loops
        .iter()
        .map(|lp| PlacedLoop {
            segs: lp
                .segments
                .iter()
                .map(|s| (s.segment.place(frame), s.names.clone()))
                .collect(),
        })
        .collect()
}

/// Sweeps `profile`, drawn on `frame`'s xy plane, `distance` along the frame's normal (against it
/// when negative). `start` is the cap on the sketch plane.
pub fn extrude(profile: &Profile, frame: &Frame, distance: f64, feature: &str) -> Result<Brep> {
    if !distance.is_finite() {
        return invalid("extrusion distance must be finite");
    }
    if distance.abs() <= tol::LINEAR {
        return invalid("extrusion distance must be nonzero");
    }
    let profile = profile.normalized()?;
    let n = frame.z;
    let (base, h, swap) = if distance > 0.0 {
        (*frame, distance, false)
    } else {
        (frame.translated(n * distance), -distance, true)
    };
    let up = n * h;
    let lift = Transform::translation(up);
    let top = base.translated(up);
    let (lo_name, hi_name) = if swap {
        (FaceName::END, FaceName::START)
    } else {
        (FaceName::START, FaceName::END)
    };

    let mut b = Brep::new();
    let lo = b.add_face(
        Surface::Plane(Plane { frame: base }),
        true,
        [FaceName::new(feature, lo_name)],
    );
    let hi = b.add_face(
        Surface::Plane(Plane { frame: top }),
        false,
        [FaceName::new(feature, hi_name)],
    );
    let mut faces = vec![lo, hi];
    let mut lo_loops = Vec::new();
    let mut hi_loops = Vec::new();

    for lp in place(&profile, &base) {
        let k = lp.segs.len();
        let start_pts: Vec<Point3> = lp
            .segs
            .iter()
            .map(|(s, _)| s.curve.eval(traversal(s).0))
            .collect();
        let bottom_v: Vec<VertexId> = start_pts.iter().map(|&p| b.add_vertex(p)).collect();
        let top_v: Vec<VertexId> = start_pts.iter().map(|&p| b.add_vertex(p + up)).collect();
        let mut bottom_e = Vec::with_capacity(k);
        let mut top_e = Vec::with_capacity(k);
        for (i, (s, _)) in lp.segs.iter().enumerate() {
            let (va, vb) = (i, (i + 1) % k);
            let (s0, s1) = if s.forward { (va, vb) } else { (vb, va) };
            bottom_e.push(b.add_edge(s.curve.clone(), s.range, bottom_v[s0], bottom_v[s1]));
            top_e.push(b.add_edge(s.curve.transformed(&lift), s.range, top_v[s0], top_v[s1]));
        }
        let vertical_e: Vec<EdgeId> = (0..k)
            .map(|i| {
                let line = Line {
                    origin: start_pts[i],
                    dir: n,
                };
                b.add_edge(Curve::Line(line), (0.0, h), bottom_v[i], top_v[i])
            })
            .collect();
        for (i, (s, name)) in lp.segs.iter().enumerate() {
            let surface = match &s.curve {
                Curve::Line(l) => Surface::Plane(Plane {
                    frame: Frame {
                        origin: l.origin,
                        x: l.dir,
                        y: n,
                        z: l.dir.cross(n),
                    },
                }),
                Curve::Circle(c) => Surface::Cylinder(Cylinder {
                    frame: c.frame,
                    radius: c.radius,
                }),
                other => Surface::Extrusion(Box::new(Extrusion {
                    curve: other.clone(),
                    dir: n,
                })),
            };
            let face = b.add_face(surface, !s.forward, FaceName::all(feature, name));
            let (ta, tb) = traversal(s);
            b.add_loop(
                face,
                vec![
                    coedge(
                        bottom_e[i],
                        !s.forward,
                        affine(Vec2::ZERO, Vec2::new(1.0, 0.0)),
                    ),
                    coedge(
                        vertical_e[(i + 1) % k],
                        false,
                        affine(Vec2::new(tb, 0.0), Vec2::new(0.0, 1.0)),
                    ),
                    coedge(
                        top_e[i],
                        s.forward,
                        affine(Vec2::new(0.0, h), Vec2::new(1.0, 0.0)),
                    ),
                    coedge(
                        vertical_e[i],
                        true,
                        affine(Vec2::new(ta, 0.0), Vec2::new(0.0, 1.0)),
                    ),
                ],
            );
            faces.push(face);
        }
        let lo_loop: Vec<CoedgeSpec> = (0..k)
            .rev()
            .map(|i| {
                let s = &lp.segs[i].0;
                let at = s.curve.eval(traversal(s).1);
                coedge(bottom_e[i], s.forward, plane_pcurve(&base, &s.curve, at))
            })
            .collect();
        let hi_loop: Vec<CoedgeSpec> = (0..k)
            .map(|i| {
                let s = &lp.segs[i].0;
                let c = s.curve.transformed(&lift);
                let at = c.eval(traversal(s).0);
                coedge(top_e[i], !s.forward, plane_pcurve(&top, &c, at))
            })
            .collect();
        lo_loops.push(lo_loop);
        hi_loops.push(hi_loop);
    }
    for l in lo_loops {
        b.add_loop(lo, l);
    }
    for l in hi_loops {
        b.add_loop(hi, l);
    }
    let shell = b.add_shell(faces);
    b.add_solid(vec![shell]);
    finish(b)
}

/// How a swept face's surface parameterises the profile curve at angle 0: `v = v0 + dv·t`. A
/// plane has cartesian parameters instead.
enum Sweep {
    Curved { surface: Surface, v0: f64, dv: f64 },
    Plane { surface: Surface, frame: Frame },
}

struct Axis {
    origin: Point3,
    frame: Frame,
}

impl Axis {
    fn height(&self, p: Point3) -> f64 {
        (p - self.origin).dot(self.frame.z)
    }

    fn radius(&self, p: Point3) -> f64 {
        let d = p - self.origin;
        (d - self.frame.z * d.dot(self.frame.z)).norm()
    }

    fn at_height(&self, h: f64) -> Frame {
        self.frame.translated(self.frame.z * h)
    }
}

fn classify(axis: &Axis, s: &PlacedSegment) -> Sweep {
    let (e_rho, a) = (axis.frame.x, axis.frame.z);
    match &s.curve {
        Curve::Line(l) => {
            let (rho, h) = (axis.radius(l.origin), axis.height(l.origin));
            let (dr, dh) = (l.dir.dot(e_rho), l.dir.dot(a));
            let frame = axis.at_height(h);
            if dr.abs() <= tol::ANGULAR {
                Sweep::Curved {
                    surface: Surface::Cylinder(Cylinder { frame, radius: rho }),
                    v0: 0.0,
                    dv: dh,
                }
            } else if dh.abs() <= tol::ANGULAR {
                Sweep::Plane {
                    surface: Surface::Plane(Plane { frame }),
                    frame,
                }
            } else {
                let cone = Cone {
                    frame,
                    radius: rho,
                    half_angle: dr.atan2(dh),
                };
                Sweep::Curved {
                    surface: Surface::Cone(cone),
                    v0: 0.0,
                    dv: 1.0,
                }
            }
        }
        Curve::Circle(c) => {
            let (rho, h) = (axis.radius(c.frame.origin), axis.height(c.frame.origin));
            let x = c.frame.x;
            let phi0 = x.dot(a).atan2(x.dot(e_rho));
            let sign = if c.frame.z.dot(axis.frame.y) > 0.0 {
                -1.0
            } else {
                1.0
            };
            let mid = phi0 + sign * 0.5 * (s.range.0 + s.range.1);
            let phi0 = phi0 - (mid / TAU).round() * TAU;
            let frame = axis.at_height(h);
            if rho <= tol::LINEAR {
                let surface = Surface::Sphere(Sphere {
                    frame,
                    radius: c.radius,
                });
                return Sweep::Curved {
                    surface,
                    v0: phi0,
                    dv: sign,
                };
            }
            if (c.frame.origin - axis.origin).dot(e_rho) < 0.0 {
                // The circle's centre lies across the axis from the arc: the arc turns into the
                // inner sheet of a spindle torus, where a torus's closed-form projection, which
                // takes the half-plane of the point, finds the outer sheet. Its exact surface of
                // revolution projects onto the arc itself.
                return revolution(axis, &s.curve);
            }
            let surface = Surface::Torus(Torus {
                frame,
                major: rho,
                minor: c.radius,
            });
            Sweep::Curved {
                surface,
                v0: phi0,
                dv: sign,
            }
        }
        other => revolution(axis, other),
    }
}

fn revolution(axis: &Axis, curve: &Curve) -> Sweep {
    Sweep::Curved {
        surface: Surface::Revolution(Box::new(Revolution {
            curve: curve.clone(),
            origin: axis.origin,
            axis: axis.frame.z,
        })),
        v0: 0.0,
        dv: 1.0,
    }
}

/// Turns `profile`, drawn on `frame`'s xy plane, by `angle` (radians, right-handed) about the line
/// through `origin` along `dir`, which must lie in the sketch plane and not cross the profile. A
/// partial turn has `start` and `end` caps; a full one has none.
pub fn revolve(
    profile: &Profile,
    frame: &Frame,
    origin: Point3,
    dir: Vec3,
    angle: f64,
    feature: &str,
) -> Result<Brep> {
    let Some(mut a) = dir.unit() else {
        return invalid("revolve axis has no direction");
    };
    if a.dot(frame.z).abs() > tol::ANGULAR
        || (origin - frame.origin).dot(frame.z).abs() > tol::LINEAR
    {
        return invalid("revolve axis must lie in the sketch plane");
    }
    let mut angle = angle;
    if angle < 0.0 {
        a = -a;
        angle = -angle;
    }
    if !(angle > tol::ANGULAR && angle <= TAU + tol::ANGULAR) {
        return invalid("revolve angle must be in (0, 360°]");
    }
    let full = angle >= TAU - tol::ANGULAR;
    if full {
        angle = TAU;
    }
    let mut profile = profile.normalized()?;

    let mut e_rho = frame.z.cross(a);
    let side = |e: Vec3| -> (f64, f64) {
        let mut range = (f64::INFINITY, f64::NEG_INFINITY);
        for lp in &profile.loops {
            for p in lp.polygon() {
                let s = (frame.at(p.x, p.y) - origin).dot(e);
                range = (range.0.min(s), range.1.max(s));
            }
        }
        range
    };
    let (lo, hi) = side(e_rho);
    if lo < -tol::LINEAR && hi > tol::LINEAR {
        return invalid("revolve axis crosses the profile");
    }
    if hi <= tol::LINEAR {
        e_rho = -e_rho;
    }
    let axis = Axis {
        origin,
        frame: Frame {
            origin,
            x: e_rho,
            y: a.cross(e_rho),
            z: a,
        },
    };
    let e_theta = axis.frame.y;
    if frame.z.dot(e_theta) < 0.0 {
        for lp in &mut profile.loops {
            lp.reverse();
        }
    }
    let turn = Transform::rotation(origin, a, angle);
    let on_axis = |p: Point3| axis.radius(p) <= tol::LINEAR;

    let mut b = Brep::new();
    let start_frame = Frame {
        origin,
        x: e_rho,
        y: a,
        z: e_rho.cross(a),
    };
    let end_frame = start_frame.transformed(&turn);
    let caps = (!full).then(|| {
        (
            b.add_face(
                Surface::Plane(Plane { frame: start_frame }),
                false,
                [FaceName::new(feature, FaceName::START)],
            ),
            b.add_face(
                Surface::Plane(Plane { frame: end_frame }),
                true,
                [FaceName::new(feature, FaceName::END)],
            ),
        )
    });
    let mut shells: Vec<Vec<FaceId>> = Vec::new();
    if let Some((s, e)) = caps {
        shells.push(vec![s, e]);
    }

    for lp in place(&profile, frame) {
        let k = lp.segs.len();
        let pts: Vec<Point3> = lp
            .segs
            .iter()
            .map(|(s, _)| s.curve.eval(traversal(s).0))
            .collect();
        let v0: Vec<VertexId> = pts.iter().map(|&p| b.add_vertex(p)).collect();
        let v1: Vec<VertexId> = (0..k)
            .map(|i| {
                if full || on_axis(pts[i]) {
                    v0[i]
                } else {
                    b.add_vertex(turn.point(pts[i]))
                }
            })
            .collect();
        let rings: Vec<Option<EdgeId>> = (0..k)
            .map(|i| {
                (!on_axis(pts[i])).then(|| {
                    let h = axis.height(pts[i]);
                    let circle = crate::geom::Circle {
                        frame: axis.at_height(h),
                        radius: axis.radius(pts[i]),
                    };
                    b.add_edge(Curve::Circle(circle), (0.0, angle), v0[i], v1[i])
                })
            })
            .collect();
        let ends = |i: usize, s: &PlacedSegment, v: &[VertexId]| {
            let (va, vb) = (v[i], v[(i + 1) % k]);
            if s.forward { (va, vb) } else { (vb, va) }
        };
        let on_axis_seg: Vec<bool> = lp
            .segs
            .iter()
            .map(|(s, _)| {
                let (t0, t1) = s.range;
                [t0, 0.5 * (t0 + t1), t1]
                    .iter()
                    .all(|&t| on_axis(s.curve.eval(t)))
            })
            .collect();
        let sweeps: Vec<Option<Sweep>> = lp
            .segs
            .iter()
            .zip(&on_axis_seg)
            .map(|((s, _), &ax)| (!ax).then(|| classify(&axis, s)))
            .collect();
        // Profile edges at angle 0 and at the end angle; a full turn needs them only as seams.
        let mut e0: Vec<Option<EdgeId>> = vec![None; k];
        let mut e1: Vec<Option<EdgeId>> = vec![None; k];
        for (i, (s, _)) in lp.segs.iter().enumerate() {
            let needed = !full || matches!(sweeps[i], Some(Sweep::Curved { .. }));
            if !needed {
                continue;
            }
            let (s0, s1) = ends(i, s, &v0);
            let first = b.add_edge(s.curve.clone(), s.range, s0, s1);
            e0[i] = Some(first);
            e1[i] = Some(if full || on_axis_seg[i] {
                first
            } else {
                let (s0, s1) = ends(i, s, &v1);
                b.add_edge(s.curve.transformed(&turn), s.range, s0, s1)
            });
        }
        let mut loop_faces = Vec::new();
        for (i, (s, name)) in lp.segs.iter().enumerate() {
            let Some(sweep) = &sweeps[i] else { continue };
            let j = (i + 1) % k;
            let (ta, tb) = traversal(s);
            let tm = 0.5 * (s.range.0 + s.range.1);
            let (_, tangent) = s.curve.d1(tm);
            let tangent = if s.forward { tangent } else { -tangent };
            let half = Transform::rotation(origin, a, 0.5 * angle);
            let outward = half.vector(tangent.cross(e_theta));
            match sweep {
                Sweep::Curved {
                    surface,
                    v0: vo,
                    dv,
                } => {
                    let v_at = |t: f64| vo + dv * t;
                    let natural = surface.normal(0.5 * angle, v_at(tm));
                    let face = b.add_face(
                        surface.clone(),
                        natural.dot(outward) < 0.0,
                        FaceName::all(feature, name),
                    );
                    let ring_or_pole = |b: &mut Brep, idx: usize| -> EdgeId {
                        match rings[idx] {
                            Some(r) => r,
                            None => b.add_degenerate_edge(v0[idx], (0.0, angle)),
                        }
                    };
                    let rb = ring_or_pole(&mut b, j);
                    let ra = ring_or_pole(&mut b, i);
                    let (e0i, e1i) = (
                        e0[i].expect("curved sweep has edges"),
                        e1[i].expect("curved sweep has edges"),
                    );
                    b.add_loop(
                        face,
                        vec![
                            coedge(
                                e0i,
                                !s.forward,
                                affine(Vec2::new(0.0, *vo), Vec2::new(0.0, *dv)),
                            ),
                            coedge(
                                rb,
                                false,
                                affine(Vec2::new(0.0, v_at(tb)), Vec2::new(1.0, 0.0)),
                            ),
                            coedge(
                                e1i,
                                s.forward,
                                affine(Vec2::new(angle, *vo), Vec2::new(0.0, *dv)),
                            ),
                            coedge(
                                ra,
                                true,
                                affine(Vec2::new(0.0, v_at(ta)), Vec2::new(1.0, 0.0)),
                            ),
                        ],
                    );
                    loop_faces.push(face);
                }
                Sweep::Plane { surface, frame: pf } => {
                    let natural = pf.z;
                    let face = b.add_face(
                        surface.clone(),
                        natural.dot(outward) < 0.0,
                        FaceName::all(feature, name),
                    );
                    let ring_coedge = |idx: usize, reversed: bool| {
                        rings[idx].map(|r| {
                            let p = if reversed {
                                turn.point(pts[idx])
                            } else {
                                pts[idx]
                            };
                            coedge(
                                r,
                                reversed,
                                PCurve::Projected {
                                    seed: plane_uv(pf, p),
                                },
                            )
                        })
                    };
                    if full {
                        let mut loops: Vec<(f64, CoedgeSpec)> = Vec::new();
                        for (idx, rev) in [(j, false), (i, true)] {
                            if let Some(c) = ring_coedge(idx, rev) {
                                loops.push((axis.radius(pts[idx]), c));
                            }
                        }
                        loops.sort_by(|x, y| y.0.total_cmp(&x.0));
                        for (_, c) in loops {
                            b.add_loop(face, vec![c]);
                        }
                    } else {
                        let start = s.curve.eval(s.range.0);
                        let sr = axis.radius(start);
                        let (_, d) = s.curve.d1(s.range.0);
                        let dr = d.dot(e_rho);
                        let (c, sn) = (angle.cos(), angle.sin());
                        let mut l = vec![coedge(
                            e0[i].expect("partial sweep has edges"),
                            !s.forward,
                            affine(Vec2::new(sr, 0.0), Vec2::new(dr, 0.0)),
                        )];
                        l.extend(ring_coedge(j, false));
                        l.push(coedge(
                            e1[i].expect("partial sweep has edges"),
                            s.forward,
                            affine(Vec2::new(sr * c, sr * sn), Vec2::new(dr * c, dr * sn)),
                        ));
                        l.extend(ring_coedge(i, true));
                        b.add_loop(face, l);
                    }
                    loop_faces.push(face);
                }
            }
        }
        if let Some((sc, ec)) = caps {
            let lo_loop: Vec<CoedgeSpec> = (0..k)
                .rev()
                .map(|i| {
                    let s = &lp.segs[i].0;
                    let e = e0[i].expect("partial sweep has edges");
                    coedge(
                        e,
                        s.forward,
                        plane_pcurve(&start_frame, &s.curve, s.curve.eval(traversal(s).1)),
                    )
                })
                .collect();
            b.add_loop(sc, lo_loop);
            let hi_loop: Vec<CoedgeSpec> = (0..k)
                .map(|i| {
                    let s = &lp.segs[i].0;
                    let e = e1[i].expect("partial sweep has edges");
                    let c = s.curve.transformed(&turn);
                    coedge(
                        e,
                        !s.forward,
                        plane_pcurve(&end_frame, &c, c.eval(traversal(s).0)),
                    )
                })
                .collect();
            b.add_loop(ec, hi_loop);
            shells[0].extend(loop_faces);
        } else {
            shells.push(loop_faces);
        }
    }
    let shell_ids: Vec<_> = shells.into_iter().map(|f| b.add_shell(f)).collect();
    b.add_solid(shell_ids);
    finish(b)
}
