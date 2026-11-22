//! A planar profile swept along a helix, its frame turning with the helix: the profile screwed
//! about the helix's axis. Each side face is a rational B-spline surface, `u` the turn and `v` the
//! profile segment's own rational parameter, so its edges are the surface's parameter lines.
//!
//! A screw motion has no rational form, so each column of the surface's control net follows its
//! control point's helix as degree-7 Hermite pieces through the helix's points and first three
//! derivatives at evenly spaced turns. The pieces are spaced so the column stays within
//! `SCREW_FIT` of the helix, and a rational surface is a convex mix of its columns, so the surface
//! stays as close to the exact screw surface.

use std::f64::consts::TAU;

use crate::{
    Error, Result,
    boolean::csi,
    check,
    error::invalid,
    geom::{Curve, Cylinder, NurbsCurve, NurbsSurface, Plane, Steady, Surface, Swept},
    loft::rail::{Hom, Rail, hom, point},
    math::{Frame, Point3, Transform, Vec2, Vec3},
    profile::{Names, Profile},
    tol,
    topo::{Brep, CoedgeSpec, EdgeId, FaceName, PCurve, VertexId},
};

/// How far a column may stray from its helix. A helix crosses a plane square to its axis, or a
/// cone about it, at its lead angle or less, a hundredth of a radian on a fine thread, and its
/// crossing slides along it by its error over that angle: a ten-thousandth of the linear
/// tolerance keeps the crossing within a hundredth of the tolerance of the true one.
pub(crate) const SCREW_FIT: f64 = tol::LINEAR * 1e-4;
pub(super) const DEGREE: usize = 7;

/// A screw motion: a turn of `sign·θ` about the axis through `origin` along `axis`, and a rise of
/// `rise·θ` along it.
#[derive(Clone, Copy, Debug)]
pub struct Helix {
    pub origin: Point3,
    pub axis: Vec3,
    /// Rise per radian of turn, mm.
    pub rise: f64,
    /// 1 for a right-hand screw about `axis`, −1 for a left-hand one.
    pub sign: f64,
}

impl Helix {
    /// Where `p` goes at turn `t`, and its first three derivatives along `t`.
    fn d3(&self, p: Point3, t: f64) -> [Vec3; 4] {
        let q = p - self.origin;
        let along = self.axis * q.dot(self.axis);
        let x = q - along;
        let y = self.axis.cross(x);
        let (s, c) = (self.sign * t).sin_cos();
        let at = self.origin + along + x * c + y * s + self.axis * (self.rise * t);
        let d1 = (y * c - x * s) * self.sign + self.axis * self.rise;
        let d2 = -(x * c + y * s);
        let d3 = (x * s - y * c) * self.sign;
        [at, d1, d2, d3]
    }

    fn at(&self, p: Point3, t: f64) -> Point3 {
        self.d3(p, t)[0]
    }

    /// The rigid move of turn `t`.
    pub fn transform(&self, t: f64) -> Transform {
        Transform::rotation(self.origin, self.axis, self.sign * t)
            .then(&Transform::translation(self.axis * (self.rise * t)))
    }

    /// The path of `p` through turns 0 to `angle`, its parameter the turn, while its distance from
    /// the axis changes by `growth` per radian: a helix when `growth` is 0, a conical spiral
    /// otherwise. Degree-7 Hermite pieces through the path's points and first three derivatives,
    /// spaced as a screw sweep's columns are, so they stay within `SCREW_FIT` of it.
    pub fn spiral(&self, p: Point3, angle: f64, growth: f64) -> Result<Curve> {
        let Some(axis) = self.axis.unit() else {
            return invalid("a spiral needs an axis");
        };
        if !(angle.is_finite() && angle > tol::ANGULAR && growth.is_finite()) {
            return invalid("a spiral runs through a positive angle");
        }
        let screw = Helix { axis, ..*self };
        let q = p - screw.origin;
        let r0 = (q - axis * q.dot(axis)).norm();
        let r1 = r0 + growth * angle;
        if r0 <= tol::LINEAR || r1 <= tol::LINEAR {
            return invalid("a spiral keeps off its axis");
        }
        // With f(t) = 1 + k·t the radius's scale and R(t) the point turned about the axis, the
        // path is f·R plus the rise: P′ = kR + fR′, P″ = 2kR′ + fR″, P‴ = 3kR″ + fR‴.
        let k = growth / r0;
        let d3 = |t: f64| -> [Vec3; 4] {
            let [at, d1, d2, d3] = screw.d3(p, t);
            let lift = screw.axis * (screw.rise * t);
            let r = at - lift - (screw.origin + axis * q.dot(axis));
            let r1 = d1 - screw.axis * screw.rise;
            let f = 1.0 + k * t;
            [
                at - r + r * f,
                r * k + r1 * f + screw.axis * screw.rise,
                r1 * (2.0 * k) + d2 * f,
                d2 * (3.0 * k) + d3 * f,
            ]
        };
        // An eighth derivative of f·R is at most r·(f + 8|k|): the nodes for that radius.
        let reach = r0.max(r1) + 8.0 * growth.abs();
        let nodes = nodes(angle, reach);
        let mut pts = vec![d3(nodes[0])[0]];
        for w in nodes.windows(2) {
            pts.extend(hermite(d3(w[0]), d3(w[1]), w[1] - w[0]));
        }
        let c = NurbsCurve::new(DEGREE, knots(&nodes), pts, None)?;
        Ok(Curve::Nurbs(Box::new(c)))
    }
}

/// Knots of a curve in Hermite pieces between `nodes`.
pub(super) fn knots(nodes: &[f64]) -> Vec<f64> {
    let mut k = vec![nodes[0]; DEGREE + 1];
    for &n in &nodes[1..nodes.len() - 1] {
        k.extend(std::iter::repeat_n(n, DEGREE));
    }
    k.extend(std::iter::repeat_n(nodes[nodes.len() - 1], DEGREE + 1));
    k
}

/// The Bézier points of a degree-7 piece of length `h` from its end values and derivatives:
/// `f⁽ᵏ⁾(0) = 7!/(7−k)!/hᵏ · Δᵏ P₀`, and the same from the far end.
fn hermite(a: [Vec3; 4], b: [Vec3; 4], h: f64) -> [Point3; 7] {
    let (k1, k2, k3) = (h / 7.0, h * h / 42.0, h * h * h / 210.0);
    let p1 = a[0] + a[1] * k1;
    let p2 = p1 * 2.0 - a[0] + a[2] * k2;
    let p3 = (p2 - p1) * 3.0 + a[0] + a[3] * k3;
    let q1 = b[0] - b[1] * k1;
    let q2 = q1 * 2.0 - b[0] + b[2] * k2;
    let q3 = (q2 - q1) * 3.0 + b[0] - b[3] * k3;
    [p1, p2, p3, q3, q2, q1, b[0]]
}

/// The control points of `p`'s helix in Hermite pieces between `nodes`.
pub(super) fn column(screw: &Helix, p: Point3, nodes: &[f64]) -> Vec<Point3> {
    let mut out = vec![screw.at(p, nodes[0])];
    for w in nodes.windows(2) {
        let (a, b) = (screw.d3(p, w[0]), screw.d3(p, w[1]));
        out.extend(hermite(a, b, w[1] - w[0]));
    }
    out
}

/// Evenly spaced turns from 0 to `angle`, close enough that a Hermite piece follows a helix of
/// radius up to `radius` within `SCREW_FIT`: its error is at most `radius · (h/2)⁸ / 8!`.
pub(super) fn nodes(angle: f64, radius: f64) -> Vec<f64> {
    let h = 2.0 * (40320.0 * SCREW_FIT / radius.max(tol::LINEAR)).powf(1.0 / 8.0);
    let n = (angle / h).ceil().max(1.0) as usize;
    (0..=n).map(|k| angle * k as f64 / n as f64).collect()
}

/// What `rail` sweeps carried through `angle` of `screw`, whose axis is a unit vector: the
/// rational B-spline whose columns follow its control points' helices, `u` the turn and `v` the
/// rail's own parameter. It keeps the rail and the screw as the curve and motion it was swept by.
pub(crate) fn sheet(rail: &Rail, screw: &Helix, angle: f64) -> Result<NurbsSurface> {
    let radius = rail
        .pw
        .iter()
        .map(|&h| {
            let q = point(h) - screw.origin;
            (q - screw.axis * q.dot(screw.axis)).norm()
        })
        .fold(0.0, f64::max);
    let nodes = nodes(angle, radius);
    let cols: Vec<Vec<Point3>> = rail
        .pw
        .iter()
        .map(|&h| column(screw, point(h), &nodes))
        .collect();
    let (nu, nv) = (cols[0].len(), cols.len());
    let mut net = Vec::with_capacity(nu * nv);
    for row in 0..nu {
        net.extend(cols.iter().map(|c| c[row]));
    }
    let w: Vec<f64> = rail.pw.iter().map(|h| h[3]).collect();
    let rational = w.iter().any(|&x| (x - 1.0).abs() > 1e-15);
    let weights = rational.then(|| (0..nu).flat_map(|_| w.iter().copied()).collect());
    let spline = NurbsSurface::new(
        (DEGREE, rail.degree),
        (knots(&nodes), rail.knots.clone()),
        (nu, nv),
        net,
        weights,
    )?;
    // The surface is a mean of its columns, and so as near the swept one as they are to their
    // helices, only where every weight is positive.
    if !w.iter().all(|&x| x > 0.0) {
        return Ok(spline);
    }
    let generator = NurbsCurve::new(
        rail.degree,
        rail.knots.clone(),
        rail.pw.iter().map(|&h| point(h)).collect(),
        rational.then_some(w),
    )?;
    let motion = Steady {
        origin: screw.origin,
        axis: screw.axis,
        turn: screw.sign,
        rise: screw.rise,
    };
    Ok(spline.swept_by(Swept { generator, motion }))
}

fn coedge(edge: EdgeId, reversed: bool, pcurve: PCurve) -> CoedgeSpec {
    CoedgeSpec {
        edge,
        reversed,
        pcurve,
    }
}

fn affine(origin: Vec2, dir: Vec2) -> PCurve {
    PCurve::Affine { origin, dir }
}

fn plane_seed(frame: &Frame, p: Point3) -> PCurve {
    let l = frame.to_local(p);
    PCurve::Projected {
        seed: Vec2::new(l.x, l.y),
    }
}

fn split(pw: &[Hom]) -> (Vec<Point3>, Option<Vec<f64>>) {
    let w: Vec<f64> = pw.iter().map(|h| h[3]).collect();
    let rational = w.iter().any(|&x| (x - 1.0).abs() > 1e-15);
    (
        pw.iter().map(|&h| point(h)).collect(),
        rational.then_some(w),
    )
}

/// One profile segment: its rail on `[0, 1]` in the loop's direction and its name.
pub(crate) struct Seg {
    pub rail: Rail,
    pub name: Names,
}

/// Where the swept profile would meet itself: the profile must keep to one side of the axis, and
/// where the sweep turns a whole turn or more, the profile must stay clear of its copy a pitch
/// along the axis, which the sweep lays in the same half-plane a turn later.
fn self_contact(profile: &Profile, frame: &Frame, screw: &Helix, angle: f64) -> Result<()> {
    let side = screw.axis.cross(frame.z);
    let curves = placed_curves(profile, frame);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut z_lo, mut z_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (c, (t0, t1)) in &curves {
        let mut ts = vec![*t0, *t1];
        ts.extend(c.extremes(side, *t0, *t1));
        ts.extend(c.extremes(screw.axis, *t0, *t1));
        for t in ts {
            let q = c.eval(t) - screw.origin;
            let (r, z) = (q.dot(side), q.dot(screw.axis));
            (lo, hi) = (lo.min(r), hi.max(r));
            (z_lo, z_hi) = (z_lo.min(z), z_hi.max(z));
        }
    }
    let near = lo.abs().min(hi.abs());
    if lo <= tol::LINEAR && hi >= -tol::LINEAR {
        return invalid("a screw sweep's profile reaches its axis");
    }
    // Half-planes a hair short of a whole turn apart hold the end cap within the tolerance of the
    // start cap where the profile is nearest the axis.
    let turns = ((angle + tol::LINEAR / near) / TAU).floor();
    let rise = screw.rise * TAU;
    let mut k = 1.0;
    while k <= turns && (k * rise).abs() <= z_hi - z_lo + tol::LINEAR {
        let shift = Transform::translation(screw.axis * (k * rise));
        let meets = curves.iter().any(|(c, r)| {
            let moved = c.transformed(&shift);
            curves
                .iter()
                .any(|(d, rd)| !csi::curve_curve(&moved, *r, d, *rd).is_empty())
        });
        if meets || (k * rise).abs() <= tol::LINEAR {
            return invalid("a screw sweep's coils meet: its pitch is too short for its profile");
        }
        k += 1.0;
    }
    Ok(())
}

/// The profile's boundary as curves placed on `frame`, each with its parameter range.
fn placed_curves(profile: &Profile, frame: &Frame) -> Vec<(Curve, (f64, f64))> {
    profile
        .loops
        .iter()
        .flat_map(|lp| &lp.segments)
        .map(|s| {
            let placed = s.segment.place(frame);
            (placed.curve, placed.range)
        })
        .collect()
}

/// The side of its plane the screw carries a profile on an oblique plane to, 1 along `frame.z`
/// and −1 against it. The rate a point crosses the plane,
/// `n·v(p) = sign·(n × axis)·(p − origin) + rise·n·axis`, is affine in `p`, so over the profile
/// it is least and greatest where the boundary is extreme along `n × axis`. Where it is 0 the motion runs along the plane and the
/// swept solid folds over itself.
fn crossing_side(curves: &[(Curve, (f64, f64))], frame: &Frame, screw: &Helix) -> Result<f64> {
    let across = frame.z.cross(screw.axis);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (c, (t0, t1)) in curves {
        let mut ts = vec![*t0, *t1];
        ts.extend(c.extremes(across, *t0, *t1));
        for t in ts {
            let v = screw.d3(c.eval(t), 0.0)[1];
            let rate = v.dot(frame.z) / v.norm().max(f64::MIN_POSITIVE);
            (lo, hi) = (lo.min(rate), hi.max(rate));
        }
    }
    if lo > tol::ANGULAR {
        Ok(1.0)
    } else if hi < -tol::ANGULAR {
        Ok(-1.0)
    } else {
        invalid("a screw sweep folds: its motion runs along its profile's plane over the profile")
    }
}

/// Boundary points tested along each segment for a turn that brings them back across the plane.
const RETURN_SAMPLES: usize = 64;

/// Where the sweep of a profile on an oblique plane might meet itself. The profile and its copy
/// turned by `t` meet only where a boundary point of one lies on the other, so a boundary point's
/// helix, run forward or back by `t`, crosses the plane in the direction `side` that the profile
/// crosses it. The copies are `rise·t` apart along the axis, so only turns within the profile's
/// height over the rise count. Along each point's helix the height over the plane is
/// `h(t) = C·(cos σt − 1) + B·sin σt + L·t`, monotone between the turns where `h′` is 0, which
/// are found in closed form, so each crossing is bracketed and found by bisection. A helix that
/// does cross that way within reach may still miss the profile, so this refuses as unsupported
/// rather than invalid.
fn returns(
    curves: &[(Curve, (f64, f64))],
    frame: &Frame,
    screw: &Helix,
    angle: f64,
    side: f64,
) -> Result<()> {
    let (n, a, sigma) = (frame.z, screw.axis, screw.sign);
    let (mut z_lo, mut z_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (c, (t0, t1)) in curves {
        let mut ts = vec![*t0, *t1];
        ts.extend(c.extremes(a, *t0, *t1));
        for t in ts {
            let z = (c.eval(t) - screw.origin).dot(a);
            (z_lo, z_hi) = (z_lo.min(z), z_hi.max(z));
        }
    }
    let reach = if screw.rise.abs() > 0.0 {
        angle.min((z_hi - z_lo + tol::LINEAR) / screw.rise.abs())
    } else {
        angle
    };
    let l = screw.rise * n.dot(a);
    for (c, (t0, t1)) in curves {
        for k in 0..=RETURN_SAMPLES {
            let b = c.eval(t0 + (t1 - t0) * k as f64 / RETURN_SAMPLES as f64);
            let q = b - screw.origin;
            let x = q - a * q.dot(a);
            let (cx, by) = (n.dot(x), n.dot(a.cross(x)));
            let k_amp = cx.hypot(by);
            if k_amp <= l.abs() {
                continue;
            }
            let h = |t: f64| cx * ((sigma * t).cos() - 1.0) + by * (sigma * t).sin() + l * t;
            let dh = |t: f64| sigma * (by * (sigma * t).cos() - cx * (sigma * t).sin()) + l;
            // h′ = 0 where cos(σt + δ) = −σL/K, δ = atan2(C, B).
            let (delta, w) = (cx.atan2(by), (-sigma * l / k_amp).acos());
            let mut nodes = vec![-reach, 0.0, reach];
            let m_max = (reach / TAU).ceil() + 1.0;
            let mut m = -m_max;
            while m <= m_max {
                for u in [-delta + w + TAU * m, -delta - w + TAU * m] {
                    let t = sigma * u;
                    if t.abs() < reach && t != 0.0 {
                        nodes.push(t);
                    }
                }
                m += 1.0;
            }
            nodes.sort_by(f64::total_cmp);
            for win in nodes.windows(2) {
                let (mut lo, mut hi) = (win[0], win[1]);
                // The piece beside 0 is monotone from h(0) = 0.
                if lo == 0.0 || hi == 0.0 || h(lo) * h(hi) > 0.0 {
                    continue;
                }
                let up = h(hi) > h(lo);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if mid <= lo || mid >= hi {
                        break;
                    }
                    if (h(mid) < 0.0) == up {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                if dh(0.5 * (lo + hi)) * side > 0.0 {
                    return Err(Error::Unsupported(
                        "a screw sweep whose turns bring its oblique profile back across its \
                         plane"
                            .into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Screws `profile`, drawn on `frame`'s xy plane, through `angle` radians of `screw`. The
/// profile's plane may hold the axis, and the profile must then keep to one side of it and clear
/// of itself a turn later; or its plane is square to the axis, a twisted extrusion; or it is
/// oblique, and the screw must carry every point of the profile across the plane the same way, and
/// no turn within reach may bring the profile's boundary back across it. Caps are `start` and
/// `end`; each side face is named after its profile segment.
pub fn helix(
    profile: &Profile,
    frame: &Frame,
    screw: &Helix,
    angle: f64,
    feature: &str,
) -> Result<Brep> {
    if !(angle.is_finite() && angle > tol::ANGULAR) {
        return invalid("a screw sweep turns through a positive angle");
    }
    let Some(axis) = screw.axis.unit() else {
        return invalid("a screw sweep needs an axis");
    };
    let screw = Helix { axis, ..*screw };
    // A profile square to the axis rises as it turns, a twisted extrusion: each level holds one
    // turned copy of it, so it can never meet itself, and its outline may go round the axis.
    let square = frame.z.cross(axis).norm() <= tol::ANGULAR;
    let through = frame.z.dot(axis).abs() <= tol::ANGULAR
        && (frame.origin - screw.origin).dot(frame.z).abs() <= tol::LINEAR;
    if square && screw.rise.abs() <= tol::LINEAR {
        return invalid("a twisted extrusion must rise");
    }
    let mut profile = profile.normalized()?;
    if through {
        self_contact(&profile, frame, &screw, angle)?;
    } else if !square {
        let curves = placed_curves(&profile, frame);
        let side = crossing_side(&curves, frame, &screw)?;
        returns(&curves, frame, &screw, angle, side)?;
    }
    let (mut radius, mut far) = (0.0f64, frame.origin);
    for lp in &profile.loops {
        for p in lp.polygon() {
            let q = frame.at(p.x, p.y) - screw.origin;
            let r = (q - axis * q.dot(axis)).norm();
            if r > radius {
                (radius, far) = (r, frame.at(p.x, p.y));
            }
        }
    }
    // The motion of a point of the profile: on the axis, where the frame's origin may lie, a screw
    // only rises, along the very plane it must cross.
    let motion = screw.d3(far, 0.0)[1];
    if motion.dot(frame.z).abs() <= tol::ANGULAR * motion.norm() {
        return invalid("a screw sweep's profile plane runs along its motion");
    }
    // Loops run counter-clockwise about the motion, and the start cap's frame faces along it.
    let sketch = *frame;
    let mut frame = *frame;
    if motion.dot(frame.z) < 0.0 {
        for lp in &mut profile.loops {
            lp.reverse();
        }
        frame = Frame::new(frame.origin, -frame.z, frame.x)?;
    }
    let loops: Vec<Vec<Seg>> = profile
        .loops
        .iter()
        .map(|lp| {
            lp.segments
                .iter()
                .map(|s| {
                    let placed = s.segment.place(&sketch);
                    let (t0, t1) = if placed.forward {
                        placed.range
                    } else {
                        (placed.range.1, placed.range.0)
                    };
                    Ok(Seg {
                        rail: Rail::of_curve(&placed.curve, t0, t1)?,
                        name: s.names.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<_>>()?;
    // A rational rail's inner control points lie off the profile, up to √2 its radius away.
    let far = loops
        .iter()
        .flatten()
        .flat_map(|s| s.rail.pw.iter())
        .map(|&h| {
            let q = point(h) - screw.origin;
            (q - axis * q.dot(axis)).norm()
        })
        .fold(radius, f64::max);
    let nodes = nodes(angle, far);
    let ku = knots(&nodes);
    let end_frame = frame.transformed(&screw.transform(angle));
    let motion = Motion {
        degree: DEGREE,
        knots: ku,
        range: (0.0, angle),
        column: Box::new(|p| column(&screw, p, &nodes)),
        at_end: Box::new(|p| screw.at(p, angle)),
        velocity: Box::new(|p| screw.d3(p, 0.0)[1]),
        end_frame,
        screw: Some(screw),
    };
    assemble(&loops, frame, &motion, feature)
}

/// How a profile moves along a sweep: each point's track as a curve of `degree` on `knots` over
/// `range` (its control points), where it ends, and its velocity at the start; and the frame the
/// end cap lies in.
pub(crate) struct Motion<'a> {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub range: (f64, f64),
    pub column: Box<dyn Fn(Point3) -> Vec<Point3> + 'a>,
    pub at_end: Box<dyn Fn(Point3) -> Point3 + 'a>,
    pub velocity: Box<dyn Fn(Point3) -> Vec3 + 'a>,
    pub end_frame: Frame,
    /// The screw, where the motion is one: then a segment along its axis sweeps a cylinder.
    pub screw: Option<Helix>,
}

/// The solid `loops` sweep moving by `m` from the start cap in `frame`: its caps, and a side face
/// for each segment whose `u` runs along the motion and `v` along the segment.
///
/// A screw's segment along its axis, a thread's crest, sweeps a strip of a cylinder, which it is
/// made exactly, so that a boolean meeting a shank of that radius finds the two faces on one
/// surface. A cylinder's angle runs once round, so the strip is cut at each whole turn, and with
/// it the helices along every corner of the profile, each other face bounded by their pieces.
pub(crate) fn assemble(
    loops: &[Vec<Seg>],
    frame: Frame,
    m: &Motion,
    feature: &str,
) -> Result<Brep> {
    let helix = |p: Point3| -> Result<Curve> {
        let c = NurbsCurve::new(m.degree, m.knots.clone(), (m.column)(p), None)?;
        Ok(Curve::Nurbs(Box::new(c)))
    };
    // The turns the sweep is cut at: its ends, and each whole turn of a screw between them.
    let mut cuts = vec![m.range.0];
    if m.screw.is_some() {
        let mut t = m.range.0 + TAU;
        while t < m.range.1 - tol::ANGULAR {
            cuts.push(t);
            t += TAU;
        }
    }
    cuts.push(m.range.1);
    let pieces = cuts.len() - 1;
    let at = |p: Point3, k: usize| -> Point3 {
        match (&m.screw, k) {
            (_, 0) => p,
            (_, k) if k == pieces => (m.at_end)(p),
            (Some(sc), k) => sc.at(p, cuts[k]),
            (None, _) => unreachable!("only a screw is cut between its ends"),
        }
    };
    let line = |a: Point3, b: Point3| -> Result<Curve> {
        Ok(Curve::Nurbs(Box::new(NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![a, b],
            None,
        )?)))
    };
    let end_frame = m.end_frame;
    let mut b = Brep::new();
    let start = b.add_face(
        Surface::Plane(Plane { frame }),
        true,
        [FaceName::new(feature, FaceName::START)],
    );
    let end = b.add_face(
        Surface::Plane(Plane { frame: end_frame }),
        false,
        [FaceName::new(feature, FaceName::END)],
    );
    let mut faces = vec![start, end];
    for segs in loops {
        let k = segs.len();
        let corners: Vec<Point3> = segs.iter().map(|s| s.rail.start()).collect();
        // Each corner's vertex at every cut, and its helix in a piece between each two.
        let verts: Vec<Vec<VertexId>> = corners
            .iter()
            .map(|&p| (0..=pieces).map(|c| b.add_vertex(at(p, c))).collect())
            .collect();
        let rings: Vec<Vec<EdgeId>> = (0..k)
            .map(|i| {
                let curve = helix(corners[i])?;
                Ok((0..pieces)
                    .map(|c| {
                        b.add_edge(
                            curve.clone(),
                            (cuts[c], cuts[c + 1]),
                            verts[i][c],
                            verts[i][c + 1],
                        )
                    })
                    .collect())
            })
            .collect::<Result<_>>()?;
        let mut e0 = Vec::with_capacity(k);
        let mut e1 = Vec::with_capacity(k);
        for (i, s) in segs.iter().enumerate() {
            let r = &s.rail;
            let (pts, w) = split(&r.pw);
            let at_end: Vec<Hom> =
                r.pw.iter()
                    .map(|&h| hom((m.at_end)(point(h)), h[3]))
                    .collect();
            let (end_pts, _) = split(&at_end);
            let j = (i + 1) % k;
            let c0 = NurbsCurve::new(r.degree, r.knots.clone(), pts.clone(), w.clone())?;
            let c1 = NurbsCurve::new(r.degree, r.knots.clone(), end_pts, w.clone())?;
            e0.push(b.add_edge(
                Curve::Nurbs(Box::new(c0)),
                (0.0, 1.0),
                verts[i][0],
                verts[j][0],
            ));
            e1.push(b.add_edge(
                Curve::Nurbs(Box::new(c1)),
                (0.0, 1.0),
                verts[i][pieces],
                verts[j][pieces],
            ));
            let name = FaceName::all(feature, &s.name);
            let widest = cuts.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
            let strip = m.screw.as_ref().and_then(|sc| {
                along_axis(sc, &pts, &w)
                    .map(|(h0, dh, radius)| (sc, h0, dh, radius, 0.0))
                    // An arc's strip must fit one turn of its cylinder's angle.
                    .or_else(|| {
                        about_axis(sc, r)
                            .filter(|&(_, _, sweep)| sweep.abs() + widest <= TAU - 1e-9)
                            .map(|(h0, radius, sweep)| (sc, h0, 0.0, radius, sweep * sc.sign))
                    })
            });
            if let Some((sc, h0, dh, radius, round)) = strip {
                // The cylinder of the strip, angle 0 where the turn `c` starts and z along the
                // turning, so the angle runs with the screw's turn: at turn t and segment
                // parameter s the angle is `round·s + t − t0` (for an arc, at the angle it has
                // turned by s) and the height along z `sign·(h0 + rise·t + s·dh)`.
                let z = sc.axis * sc.sign;
                let (sign, rise) = (sc.sign, sc.rise);
                let seps: Vec<EdgeId> = (0..=pieces)
                    .map(|c| match c {
                        0 => Ok(e0[i]),
                        c if c == pieces => Ok(e1[i]),
                        c if round == 0.0 => Ok(b.add_edge(
                            line(at(corners[i], c), at(corners[j], c))?,
                            (0.0, 1.0),
                            verts[i][c],
                            verts[j][c],
                        )),
                        c => {
                            let turned: Vec<Hom> =
                                r.pw.iter()
                                    .map(|&h| hom(sc.at(point(h), cuts[c]), h[3]))
                                    .collect();
                            let (ctrl, wts) = split(&turned);
                            let arc = NurbsCurve::new(r.degree, r.knots.clone(), ctrl, wts)?;
                            Ok(b.add_edge(
                                Curve::Nurbs(Box::new(arc)),
                                (0.0, 1.0),
                                verts[i][c],
                                verts[j][c],
                            ))
                        }
                    })
                    .collect::<Result<_>>()?;
                for c in 0..pieces {
                    let p = at(corners[i], c) - sc.origin;
                    let x = (p - sc.axis * p.dot(sc.axis)).unit().ok_or_else(|| {
                        Error::InvalidInput("a screw's crest lies on its axis".into())
                    })?;
                    let surface = Surface::Cylinder(Cylinder {
                        frame: Frame::new(sc.origin, z, x)?,
                        radius,
                    });
                    let (t0, t1) = (cuts[c], cuts[c + 1]);
                    let v = |t: f64, s: f64| sign * (h0 + rise * t + s * dh);
                    // A line's end is affine in its parameter; an arc's angle is not in its
                    // rational one, so its end is placed by projection.
                    let across = |t: f64| {
                        if round == 0.0 {
                            affine(Vec2::new(t - t0, v(t, 0.0)), Vec2::new(0.0, sign * dh))
                        } else {
                            PCurve::Projected {
                                seed: Vec2::new(t - t0, v(t, 0.0)),
                            }
                        }
                    };
                    // Outward where the motion and the segment's direction say, as below.
                    let (_, su, sv) = surface.d1(0.5 * round, v(t0, 0.5));
                    let (pm, d) = r_mid(r);
                    let outward = d.cross((m.velocity)(pm));
                    let face = b.add_face(surface, su.cross(sv).dot(outward) < 0.0, name.clone());
                    b.add_loop(
                        face,
                        vec![
                            coedge(seps[c], false, across(t0)),
                            coedge(
                                rings[j][c],
                                false,
                                affine(
                                    Vec2::new(round - t0, v(0.0, 1.0)),
                                    Vec2::new(1.0, sign * rise),
                                ),
                            ),
                            coedge(seps[c + 1], true, across(t1)),
                            coedge(
                                rings[i][c],
                                true,
                                affine(Vec2::new(-t0, v(0.0, 0.0)), Vec2::new(1.0, sign * rise)),
                            ),
                        ],
                    );
                    faces.push(face);
                }
                continue;
            }
            let nv = pts.len();
            let mut net = Vec::with_capacity(nv * (m.knots.len() - m.degree - 1));
            let cols: Vec<Vec<Point3>> = pts.iter().map(|&p| (m.column)(p)).collect();
            for row in 0..cols[0].len() {
                net.extend(cols.iter().map(|c| c[row]));
            }
            let nu = cols[0].len();
            let weights = w.map(|w| (0..nu).flat_map(|_| w.iter().copied()).collect());
            let surface = Surface::Nurbs(Box::new(NurbsSurface::new(
                (m.degree, r.degree),
                (m.knots.clone(), r.knots.clone()),
                (nu, nv),
                net,
                weights,
            )?));
            // At the start turn, where the surface's column points are the profile's own.
            let (_, su, sv) = surface.d1(m.range.0, 0.5);
            let (p, d) = r_mid(r);
            let outward = d.cross((m.velocity)(p));
            let face = b.add_face(surface, su.cross(sv).dot(outward) < 0.0, name);
            let mut lp = vec![coedge(
                e0[i],
                false,
                affine(Vec2::new(m.range.0, 0.0), Vec2::new(0.0, 1.0)),
            )];
            lp.extend(
                rings[j]
                    .iter()
                    .map(|&e| coedge(e, false, affine(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)))),
            );
            lp.push(coedge(
                e1[i],
                true,
                affine(Vec2::new(m.range.1, 0.0), Vec2::new(0.0, 1.0)),
            ));
            lp.extend(
                rings[i]
                    .iter()
                    .rev()
                    .map(|&e| coedge(e, true, affine(Vec2::ZERO, Vec2::new(1.0, 0.0)))),
            );
            b.add_loop(face, lp);
            faces.push(face);
        }
        let lo: Vec<CoedgeSpec> = (0..k)
            .rev()
            .map(|i| coedge(e0[i], true, plane_seed(&frame, corners[(i + 1) % k])))
            .collect();
        b.add_loop(start, lo);
        let hi: Vec<CoedgeSpec> = (0..k)
            .map(|i| coedge(e1[i], false, plane_seed(&end_frame, (m.at_end)(corners[i]))))
            .collect();
        b.add_loop(end, hi);
    }
    let shell = b.add_shell(faces);
    b.add_solid(vec![shell]);
    let issues = check::check(&b);
    if issues.is_empty() {
        Ok(b)
    } else {
        Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.to_string()).collect(),
        ))
    }
}

/// Where a rail is a straight run along the screw's axis at a radius off it: its height along
/// the axis at its start, how far it rises, and the radius.
fn along_axis(sc: &Helix, pts: &[Point3], w: &Option<Vec<f64>>) -> Option<(f64, f64, f64)> {
    if pts.len() != 2
        || w.as_ref()
            .is_some_and(|w| w.iter().any(|&x| (x - w[0]).abs() > 1e-12))
    {
        return None;
    }
    let (a, b) = (pts[0] - sc.origin, pts[1] - sc.origin);
    let d = b - a;
    if d.cross(sc.axis).norm() > tol::LINEAR * 1e-3 {
        return None;
    }
    let radius = (a - sc.axis * a.dot(sc.axis)).norm();
    (radius > tol::LINEAR).then(|| (a.dot(sc.axis), d.dot(sc.axis), radius))
}

/// Where a rail is an arc about the screw's axis in a plane square to it: its height along the
/// axis, its radius, and the angle it turns through about the axis, signed.
fn about_axis(sc: &Helix, r: &Rail) -> Option<(f64, f64, f64)> {
    const N: usize = 16;
    let (ctrl, w) = split(&r.pw);
    let c = NurbsCurve::new(r.degree, r.knots.clone(), ctrl, w).ok()?;
    let at = |k: usize| {
        let q = c.eval(k as f64 / N as f64) - sc.origin;
        let h = q.dot(sc.axis);
        (q - sc.axis * h, h)
    };
    let (first, h0) = at(0);
    let radius = first.norm();
    if radius <= tol::LINEAR {
        return None;
    }
    let (mut last, mut sweep) = (first, 0.0);
    for k in 1..=N {
        let (q, h) = at(k);
        let fit = tol::LINEAR * 1e-3;
        if (h - h0).abs() > fit || (q.norm() - radius).abs() > fit {
            return None;
        }
        sweep += last.cross(q).dot(sc.axis).atan2(last.dot(q));
        last = q;
    }
    (sweep.abs() > tol::ANGULAR).then_some((h0, radius, sweep))
}

/// The rail's middle point and its direction there.
fn r_mid(r: &Rail) -> (Point3, Vec3) {
    let c = NurbsCurve::new(r.degree, r.knots.clone(), split(&r.pw).0, split(&r.pw).1)
        .expect("a rail is a valid curve");
    let (p, d, _) = c.d2(0.5);
    (p, d)
}

#[cfg(test)]
mod tests;
