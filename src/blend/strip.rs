//! A piece's blend face built alone, as the strip its section sweeps, trimmed where it ends: for
//! the sewing, with no solid tool and no boolean. A strip ends square on its own section; on a
//! plane, along the plane's cut of its surface; or at a corner it shares with another strip.
//!
//! At such a corner the two blend surfaces cross on one curve, the crease, whose ends are known
//! outright. Its far end is where the two strips' contact curves on the face they share meet:
//! there both surfaces touch that face, and so each other. Its near end lies where the two
//! contact curves on the walls come to the edge between the walls. Where a plane mirrors one
//! strip onto the other, both come to that edge at one point, and the crease, which lies in the
//! plane, starts there. Otherwise one strip, the lower, comes to the edge nearer the corner. Its
//! contact curve runs on past the edge, over the plane of its own wall, until it meets the other
//! strip's surface, and the crease starts there; the other strip ends on that plane from the
//! edge to that point. Between its ends the crease is followed section by section along the
//! lower strip: each section crosses the other surface once, between its two contacts, and that
//! crossing is solved on the section itself, so no step along the crease is taken and its
//! touching end is given, never searched for.

use std::cell::Cell;
use std::f64::consts::{PI, TAU};

use super::{
    Outline, Path, Picked, SIDES, Section, angle, end_reach, meridian_past, nearest_axis,
    ring_frame, sew::Use,
};
use crate::{
    boolean::{
        csi::{curve_distance, curve_surface, signed_distance},
        ssi::{Ssi, intersect_closed},
    },
    geom::{Curve, IntersectionCurve, Plane, Surface},
    math::{Aabb, Frame, Point3, Vec2, Vec3},
    profile::Segment,
    sweep, tol,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, PCurve, VertexId},
};

/// How a strip ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Close {
    /// On its own section, where its edge ends.
    Square,
    /// On the plane through `origin` square to `normal`.
    Plane { origin: Point3, normal: Vec3 },
    /// At a corner with end `end` of piece `other`: across the edge between their two walls,
    /// this piece's wall being the plane through `wall.0` square to `wall.1`. A `mirror` plane
    /// takes each of the two strips onto the other. The blends run on `past` the edge between
    /// the walls where it bends the other way from them.
    Cross {
        other: usize,
        end: usize,
        wall: (Point3, Vec3),
        mirror: Option<(Point3, Vec3)>,
        past: bool,
    },
}

/// A piece's blend face alone in an arena, and what each of its edges is to the sewing.
pub(super) struct Strip {
    pub face: Brep,
    pub uses: Vec<(EdgeId, Use)>,
}

/// A curve from `from` to `to`, at its parameters `t.0` and `t.1`.
#[derive(Clone, Debug)]
struct Run {
    curve: Curve,
    t: (f64, f64),
    from: Point3,
    to: Point3,
}

impl Run {
    fn reversed(&self) -> Run {
        Run {
            curve: self.curve.clone(),
            t: (self.t.1, self.t.0),
            from: self.to,
            to: self.from,
        }
    }
}

/// A piece's section swept past both ends of its edge: the strip before it is trimmed.
struct Lane {
    section: Section,
    blend: Segment,
    swept: Brep,
    face: FaceId,
    /// The blend face's coedges in the order of its loop.
    order: Vec<CoedgeId>,
    /// The coedge along the contact with the picked edge's face `k`.
    contacts: [CoedgeId; 2],
    /// The coedge across each end.
    caps: [CoedgeId; 2],
}

fn plane(origin: Point3, normal: Vec3) -> Result<Surface, String> {
    let frame = Frame::from_axis(origin, normal).map_err(|e| e.to_string())?;
    Ok(Surface::Plane(Plane { frame }))
}

impl Lane {
    fn swept(
        s: &Section,
        o: &Outline,
        closes: [Close; 2],
        reach: [f64; 2],
        feature: &str,
    ) -> Result<Lane, String> {
        let blend = o
            .profile
            .loops
            .iter()
            .flat_map(|l| &l.segments)
            .find(|sg| sg.names.contains("blend"))
            .ok_or("a section with no blend side")?
            .segment
            .clone();
        let swept = match s.path {
            Path::Prism { frame, length } => sweep::extrude(
                &o.profile,
                &frame.translated(frame.z * -reach[0]),
                length + reach[0] + reach[1],
                feature,
            ),
            Path::Ring {
                circle,
                t0,
                span,
                full,
            } => {
                if full {
                    return Err("a blend all round a circle has no ends".into());
                }
                let inner = nearest_axis(&o.profile.loops[0].segments);
                if inner <= tol::LINEAR {
                    return Err("a blend reaching its circular edge's axis".into());
                }
                let mut turn = reach.map(|e| e / inner);
                for k in 0..2 {
                    if let Close::Plane { origin, normal } = closes[k]
                        && let Some(past) = meridian_past(&circle, t0, span, k, origin, normal)
                    {
                        turn[k] = past;
                    }
                }
                let total = span + turn[0] + turn[1];
                if total >= TAU - 1e-3 {
                    return Err("a blend on most of a circle that ends on other faces".into());
                }
                let f = ring_frame(&circle, t0 - turn[0]);
                sweep::revolve(&o.profile, &f, circle.origin, circle.z, total, feature)
            }
        }
        .map_err(|e| e.to_string())?;
        let named =
            |b: &Brep, f: FaceId, name: &str| b.face(f).names.iter().any(|n| n.face == name);
        let mut faces = swept.face_ids().filter(|&f| named(&swept, f, "blend"));
        let (Some(face), None) = (faces.next(), faces.next()) else {
            return Err("a blend swept into other than one face".into());
        };
        let [lp] = swept.face(face).loops.as_slice() else {
            return Err("a blend face with a hole".into());
        };
        let order = swept.lp(*lp).coedges.clone();
        let mut lane = Lane {
            section: *s,
            blend,
            face,
            contacts: [order[0]; 2],
            caps: [order[0]; 2],
            order,
            swept,
        };
        let mut found = [[false; 2]; 2];
        for &c in &lane.order.clone() {
            let e = lane.swept.coedge(c).edge;
            let edge = lane.swept.edge(e);
            let Some(curve) = &edge.curve else {
                return Err("a blend face that reaches a pole".into());
            };
            let beside = lane
                .swept
                .edge_uses(e)
                .iter()
                .map(|&x| lane.swept.coedge_face(x))
                .find(|&f| f != face)
                .ok_or("a blend face with a seam")?;
            if let Some(k) = (0..2).find(|&k| named(&lane.swept, beside, SIDES[k])) {
                lane.contacts[k] = c;
                found[0][k] = true;
            } else {
                let mid = curve.eval(0.5 * (edge.range.0 + edge.range.1));
                let k = usize::from(lane.locate(mid).1 > 0.5);
                lane.caps[k] = c;
                found[1][k] = true;
            }
        }
        if lane.order.len() != 4 || found != [[true; 2]; 2] {
            return Err("a blend face bounded other than by two contacts and two ends".into());
        }
        Ok(lane)
    }

    fn surface(&self) -> &Surface {
        &self.swept.face(self.face).surface
    }

    /// The point of the strip `tau` of the way across its section from the contact with face 0 to
    /// the contact with face 1, on the section `f` of the way along the edge.
    fn point(&self, tau: f64, f: f64) -> Point3 {
        self.section.path.place_along(self.blend.at(tau), f)
    }

    /// Where `p`, on the strip, lies across its section and along its edge.
    fn locate(&self, p: Point3) -> (f64, f64) {
        let (q, f) = match self.section.path {
            Path::Prism { frame, length } => {
                let l = frame.to_local(p);
                (Vec2::new(l.x, l.y), l.z / length)
            }
            Path::Ring {
                circle, t0, span, ..
            } => {
                let l = circle.to_local(p);
                let mid = t0 + 0.5 * span;
                let off = (l.y.atan2(l.x) - mid + PI).rem_euclid(TAU) - PI;
                (Vec2::new(l.x.hypot(l.y), l.z), 0.5 + off / span)
            }
        };
        let tau = match self.blend {
            Segment::Line { a, b } => (q - a).dot(b - a) / (b - a).dot(b - a),
            Segment::Arc {
                center,
                start,
                sweep,
                ..
            } => {
                let m = sweep.abs();
                let mut d = ((angle(q - center) - start) * sweep.signum()).rem_euclid(TAU);
                if d > m + 0.5 * (TAU - m) {
                    d -= TAU;
                }
                d / m
            }
            _ => f64::NAN,
        };
        (tau, f)
    }

    /// The length of the blend's section, and of the edge's path where the blend runs.
    fn lengths(&self) -> (f64, f64) {
        let across = match self.blend {
            Segment::Arc { radius, sweep, .. } => radius * sweep.abs(),
            _ => self.blend.at(0.0).dist(self.blend.at(1.0)),
        };
        let along = match self.section.path {
            Path::Prism { length, .. } => length,
            Path::Ring { span, .. } => span * self.blend.at(0.5).x,
        };
        (across, along)
    }

    fn contact(&self, k: usize) -> (&Curve, (f64, f64)) {
        let edge = self.swept.edge(self.swept.coedge(self.contacts[k]).edge);
        (edge.curve.as_ref().expect("checked when swept"), edge.range)
    }

    /// Where the contact with face `k` meets `s`, nearest `near`.
    fn contact_on(&self, k: usize, s: &Surface, near: Point3) -> Option<Point3> {
        let (curve, range) = self.contact(k);
        let hits = curve_surface(curve, range, s);
        if hits.on_surface {
            return None;
        }
        hits.ts
            .into_iter()
            .map(|t| curve.eval(t))
            .min_by(|a, b| a.dist(near).total_cmp(&b.dist(near)))
    }

    /// The plane a contact curve lies in: a circle's own, or a line's with the direction `up`.
    fn contact_plane(&self, k: usize, up: Vec3) -> Result<Surface, String> {
        match self.contact(k).0 {
            Curve::Circle(c) => Ok(Surface::Plane(Plane { frame: c.frame })),
            Curve::Line(l) => plane(l.origin, l.dir.cross(up)),
            other => Err(format!("a blend whose contact is a {}", other.kind())),
        }
    }
}

/// The root of `h` between `lo` and `hi`, where it changes sign: false position, halving the
/// kept end's weight when it is kept twice (the Illinois rule; G. Dahlquist and Å. Björck,
/// "Numerical Methods", 1974, section 6.4).
fn root(h: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> Option<f64> {
    let (mut a, mut b) = (h(lo), h(hi));
    if a == 0.0 {
        return Some(lo);
    }
    if b == 0.0 {
        return Some(hi);
    }
    if a * b > 0.0 || a.is_nan() || b.is_nan() {
        return None;
    }
    let mut side = 0;
    for _ in 0..200 {
        let x = (lo * b - hi * a) / (b - a);
        let x = if x > lo.min(hi) && x < lo.max(hi) {
            x
        } else {
            0.5 * (lo + hi)
        };
        let y = h(x);
        if y == 0.0 || (hi - lo).abs() <= 4.0 * f64::EPSILON * (1.0 + x.abs()) {
            return Some(x);
        }
        if y * b < 0.0 {
            (lo, a) = (hi, b);
            side = 0;
        } else if side == -1 {
            a *= 0.5;
        } else {
            side = -1;
        }
        (hi, b) = (x, y);
    }
    Some(0.5 * (lo + hi))
}

/// The curve where the lane's strip crosses `other`, from `a` to `b`, both on the two surfaces.
/// A conic where the pair has one; otherwise the curve held on both surfaces, through points
/// each solved on a line across the strip.
fn crossing(lane: &Lane, other: &Surface, a: Point3, b: Point3) -> Result<Run, String> {
    // The strip laid flat, each point by its length across the section and along the edge. The
    // curve is met once by each line square to the chord from `a` to `b`, between them.
    let (across, along) = lane.lengths();
    let flat = |p: Point3| {
        let (tau, f) = lane.locate(p);
        Vec2::new(tau * across, f * along)
    };
    let (fa, fb) = (flat(a), flat(b));
    let chord = fb - fa;
    let length = chord.norm();
    if length <= tol::LINEAR {
        return Err("a blend whose end has no length".into());
    }
    let square = Vec2::new(-chord.y, chord.x) * (1.0 / length);
    // From one contact to the other the curve mostly stays in the box the two points are corners
    // of, which then parts the strip's two sides of it. Where two blend surfaces touch at `b`
    // they cross on a second curve through it, which lies outside that box.
    let on_contact = |q: Vec2| q.x.abs() <= tol::LINEAR || (q.x - across).abs() <= tol::LINEAR;
    let cornered = on_contact(fa) && on_contact(fb) && (fa.x - fb.x).abs() > 0.5 * across;
    let lost = Cell::new(false);
    let sample = |s: f64| -> Point3 {
        if s <= 0.0 {
            return a;
        }
        if s >= 1.0 {
            return b;
        }
        let base = fa + chord * s;
        let at = |u: f64| {
            let q = base + square * u;
            lane.point(q.x / across, q.y / along)
        };
        let gap = |u: f64| signed_distance(other, at(u));
        let boxed = cornered.then(|| {
            // The line leaves the box by one side of each pair that meets at `a` or at `b`.
            let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
            for (x, d, (min, max)) in [
                (base.x, square.x, (0.0, across)),
                (base.y, square.y, (fa.y.min(fb.y), fa.y.max(fb.y))),
            ] {
                if d.abs() > 1e-300 {
                    let (t0, t1) = ((min - x) / d, (max - x) / d);
                    lo = lo.max(t0.min(t1));
                    hi = hi.min(t0.max(t1));
                }
            }
            root(gap, lo, hi)
        });
        // A curve that bows out of the box, or stops inside the strip, is taken at the nearest
        // change of side.
        let found = boxed.flatten().or_else(|| {
            let mut pad = 0.125 * length;
            (0..8).find_map(|_| {
                let u = root(gap, -pad, pad);
                pad *= 2.0;
                u
            })
        });
        found.map(at).unwrap_or_else(|| {
            lost.set(true);
            a.lerp(b, s)
        })
    };
    let middle = sample(0.5);
    if lost.get() {
        return Err("a blend whose end no section of it crosses".into());
    }
    let reach = a.dist(b) + a.dist(middle) + tol::LINEAR;
    let pad = Vec3::new(reach, reach, reach);
    let region = Aabb {
        min: a - pad,
        max: a + pad,
    };
    if let Some(Ssi::Curves { curves, .. }) = intersect_closed(lane.surface(), other, &region) {
        for c in curves {
            let on = |p: Point3| {
                let (d, t) = curve_distance(&c.curve, c.range, p);
                (d <= tol::LINEAR).then_some(t)
            };
            let (Some(t0), Some(t1)) = (on(a), on(b)) else {
                continue;
            };
            let period = match &c.curve {
                Curve::Circle(_) | Curve::Ellipse(_) => Some(TAU),
                _ => None,
            };
            // A closed conic joins the two points by two arcs: the one through the middle.
            let mut arcs = vec![(t0, t1)];
            if let Some(p) = period {
                arcs.push(if t0 < t1 { (t0 + p, t1) } else { (t0, t1 + p) });
            }
            let off =
                |t: (f64, f64)| curve_distance(&c.curve, (t.0.min(t.1), t.0.max(t.1)), middle).0;
            let best = arcs
                .into_iter()
                .map(|t| (off(t), t))
                .min_by(|x, y| x.0.total_cmp(&y.0))
                .expect("one arc or two");
            if best.0 <= tol::LINEAR {
                return Ok(Run {
                    curve: c.curve,
                    t: best.1,
                    from: a,
                    to: b,
                });
            }
        }
    }
    let fitted = IntersectionCurve::fit_sampled(
        &sample,
        (0.0, 1.0),
        lane.surface().clone(),
        other.clone(),
        false,
        8,
    );
    if lost.get() {
        return Err("a blend whose end no section of it crosses".into());
    }
    let curve = fitted.map_err(|e| e.to_string())?;
    let t = curve.approx.domain();
    Ok(Run {
        curve: Curve::Intersection(Box::new(curve)),
        t,
        from: a,
        to: b,
    })
}

/// One end of a strip, trimmed: where each contact stops, and the curves across the end from the
/// contact with face `first` to the other.
struct Trim {
    stops: [Point3; 2],
    first: usize,
    runs: Vec<Run>,
}

/// The two strips' ends at the corner they share, each piece with its own.
#[allow(clippy::too_many_arguments)]
fn corner(
    body: &Brep,
    picked: &[Picked],
    lanes: &[Lane],
    (i, wall_i, on_i): (usize, (Point3, Vec3), Vec3),
    (j, wall_j, on_j): (usize, (Point3, Vec3), Vec3),
    mirror: Option<(Point3, Vec3)>,
    past: bool,
    at: Point3,
) -> Result<[Trim; 2], String> {
    let shared = |a: usize, b: usize| {
        (0..2)
            .find(|&k| picked[b].faces.contains(&picked[a].faces[k]))
            .ok_or("blends at a corner that share no face")
    };
    // The side of each strip on the face both share, and the side on its own wall.
    let (ci, cj) = (shared(i, j)?, shared(j, i)?);
    let (wi, wj) = (1 - ci, 1 - cj);
    let common = picked[i].faces[ci];
    let up = super::outward(body, common, at);
    let far = lanes[i]
        .contact_on(ci, &lanes[j].contact_plane(cj, up)?, at)
        .ok_or("blends whose contacts on the face they share do not meet")?;
    let (other, _) = lanes[j].contact(cj);
    if curve_distance(other, lanes[j].contact(cj).1, far).0 > tol::LINEAR {
        return Err("blends whose contacts on the face they share do not meet".into());
    }
    let (plane_i, plane_j) = (plane(wall_i.0, wall_i.1)?, plane(wall_j.0, wall_j.1)?);
    let missed = "a blend whose contact does not reach the edge between the walls";
    let edge_i = lanes[i].contact_on(wi, &plane_j, at).ok_or(missed)?;
    let edge_j = lanes[j].contact_on(wj, &plane_i, at).ok_or(missed)?;
    if edge_i.dist(edge_j) <= tol::LINEAR || mirror.is_some() {
        // Both come to the edge at one point, and the crease runs from it.
        let other = match mirror {
            Some((origin, normal)) => plane(origin, normal)?,
            None => lanes[j].surface().clone(),
        };
        let crease = crossing(&lanes[i], &other, edge_i, far)?;
        let trim = |w: usize| Trim {
            stops: if w == 0 { [edge_i, far] } else { [far, edge_i] },
            first: w,
            runs: vec![crease.clone()],
        };
        return Ok([trim(wi), trim(wj)]);
    }
    // The lower strip comes to the edge nearer the corner. Where the edge between the walls
    // bends the other way from the blends, air or material lies past both walls, and the lower
    // strip's contact runs on past the edge to the higher's surface; where it bends the same
    // way, the higher strip has taken the edge that far, and the lower's contact stops short.
    let i_low = edge_i.dist(at) < edge_j.dist(at);
    let (low, high) = if i_low { (i, j) } else { (j, i) };
    let (w_low, w_high) = if i_low { (wi, wj) } else { (wj, wi) };
    let (edge_low, edge_high) = if i_low {
        (edge_i, edge_j)
    } else {
        (edge_j, edge_i)
    };
    let (plane_low, on_low) = if i_low {
        (&plane_i, on_i)
    } else {
        (&plane_j, on_j)
    };
    let (curve, range) = lanes[low].contact(w_low);
    let hits = curve_surface(curve, range, lanes[high].surface());
    let near = hits
        .ts
        .into_iter()
        .map(|t| curve.eval(t))
        .filter(|p| ((*p - edge_low).dot(on_low) > 0.0) == past)
        .min_by(|a, b| a.dist(edge_low).total_cmp(&b.dist(edge_low)))
        .ok_or("a blend whose contact runs past a corner without meeting the other blend")?;
    let crease = crossing(&lanes[low], lanes[high].surface(), near, far)?;
    let step = crossing(&lanes[high], plane_low, edge_high, near)?;
    let stops = |w: usize, wall: Point3| if w == 0 { [wall, far] } else { [far, wall] };
    let lower = Trim {
        stops: stops(w_low, near),
        first: w_low,
        runs: vec![crease.clone()],
    };
    let higher = Trim {
        stops: stops(w_high, edge_high),
        first: w_high,
        runs: vec![step, crease],
    };
    Ok(if i_low {
        [lower, higher]
    } else {
        [higher, lower]
    })
}

/// The direction piece `i`'s edge runs on in past its end `k`.
fn onward(body: &Brep, picked: &[Picked], i: usize, k: usize) -> Result<Vec3, String> {
    let edge = body.edge(picked[i].edge);
    let v = if k == 0 { edge.start } else { edge.end };
    super::leaving(body, picked[i].edge, v)
        .map(|d| -d)
        .ok_or_else(|| "an edge with no tangent".into())
}

/// How far past each end a piece's section is swept so its strip can be trimmed there.
fn reaches(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    outlines: &[Outline],
    closes: &[[Close; 2]],
    i: usize,
) -> Result<[f64; 2], String> {
    let mut reach = [0.0; 2];
    for k in 0..2 {
        reach[k] = match closes[i][k] {
            Close::Square => 0.0,
            Close::Plane { origin, normal } => {
                let clip = super::Clip::Plane {
                    origin,
                    keep: normal,
                };
                end_reach(&sections[i], &outlines[i], super::End::Trim(clip), k)
                    .map_err(|e| e.to_string())?
                    .1
            }
            Close::Cross { other, end, .. } => {
                // Past the other piece's wall by all of both sections, at the angle the edge
                // meets that wall.
                let along = onward(body, picked, i, k)?;
                let Close::Cross { wall, .. } = closes[other][end] else {
                    return Err("a corner one of its two blends does not know".into());
                };
                let c = wall.1.dot(along).abs();
                if c < super::GLANCING {
                    return Err("blends that meet at a corner at a glancing angle".into());
                }
                let size = outlines[i].radius.max(outlines[other].radius);
                1.25 * size * (1.0 + (1.0 - c * c).max(0.0).sqrt()) / c + 0.25 * size
            }
        };
    }
    Ok(reach)
}

/// Every piece's strip, each end closed as `closes` gives; or why they cannot be built.
pub(super) fn strips(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    outlines: &[Outline],
    closes: &[[Close; 2]],
    feature: &str,
) -> Result<Vec<Strip>, String> {
    let mut lanes = Vec::with_capacity(picked.len());
    for i in 0..picked.len() {
        let reach = reaches(body, picked, sections, outlines, closes, i)?;
        lanes.push(Lane::swept(
            &sections[i],
            &outlines[i],
            closes[i],
            reach,
            feature,
        )?);
    }
    let mut trims: Vec<[Option<Trim>; 2]> = (0..picked.len()).map(|_| [None, None]).collect();
    for i in 0..picked.len() {
        for k in 0..2 {
            if trims[i][k].is_some() {
                continue;
            }
            let edge = body.edge(picked[i].edge);
            let at = body
                .vertex(if k == 0 { edge.start } else { edge.end })
                .point;
            match closes[i][k] {
                Close::Square => {}
                Close::Plane { origin, normal } => {
                    // A ring swept to the half plane through its axis ends there square.
                    if let Path::Ring {
                        circle, t0, span, ..
                    } = sections[i].path
                        && meridian_past(&circle, t0, span, k, origin, normal).is_some()
                    {
                        continue;
                    }
                    let cut = plane(origin, normal)?;
                    let missed = "a blend whose contact does not reach the face it ends on";
                    let a = lanes[i].contact_on(0, &cut, at).ok_or(missed)?;
                    let b = lanes[i].contact_on(1, &cut, at).ok_or(missed)?;
                    trims[i][k] = Some(Trim {
                        stops: [a, b],
                        first: 0,
                        runs: vec![crossing(&lanes[i], &cut, a, b)?],
                    });
                }
                Close::Cross {
                    other,
                    end,
                    wall,
                    mirror,
                    past,
                } => {
                    let Close::Cross { wall: theirs, .. } = closes[other][end] else {
                        return Err("a corner one of its two blends does not know".into());
                    };
                    let [mine, yours] = corner(
                        body,
                        picked,
                        &lanes,
                        (i, wall, onward(body, picked, i, k)?),
                        (other, theirs, onward(body, picked, other, end)?),
                        mirror,
                        past,
                        at,
                    )?;
                    trims[i][k] = Some(mine);
                    trims[other][end] = Some(yours);
                }
            }
        }
    }
    lanes
        .iter()
        .zip(&trims)
        .map(|(lane, trims)| trimmed(lane, trims))
        .collect()
}

/// The lane's blend face with each end cut back to its trim.
fn trimmed(lane: &Lane, trims: &[Option<Trim>; 2]) -> Result<Strip, String> {
    let swept = &lane.swept;
    let surface = lane.surface();
    let mut face = Brep::new();
    let mut made: Vec<VertexId> = Vec::new();
    let mut vertex = |face: &mut Brep, p: Point3| -> VertexId {
        if let Some(&v) = made
            .iter()
            .find(|&&v| face.vertex(v).point.dist(p) <= tol::LINEAR)
        {
            return v;
        }
        let v = face.add_vertex(p);
        made.push(v);
        v
    };
    let mut uses: Vec<(EdgeId, Use)> = Vec::new();
    let mut specs: Vec<CoedgeSpec> = Vec::new();
    // The parameters the last coedge ended at, which the next one's projection starts from.
    let mut seed: Option<Vec2> = None;
    let n = lane.order.len();
    // Start on a contact, so every end's curves follow the contact they leave.
    let first = lane
        .order
        .iter()
        .position(|c| lane.contacts.contains(c))
        .expect("two contacts");
    for step in 0..n {
        let c = lane.order[(first + step) % n];
        let co = swept.coedge(c);
        let edge = swept.edge(co.edge);
        let curve = edge.curve.as_ref().expect("checked when swept");
        if let Some(k) = lane.contacts.iter().position(|&x| x == c) {
            let mut range = edge.range;
            for (end, trim) in trims.iter().enumerate() {
                let Some(trim) = trim else { continue };
                // The contact's end at this end of the piece is its nearer one to the stop.
                let t = curve.project(trim.stops[k], edge.range.0, edge.range.1);
                let at_end = lane.locate(curve.eval(edge.range.1)).1 > 0.5;
                if (end == 1) == at_end {
                    range.1 = t;
                } else {
                    range.0 = t;
                }
            }
            let speed = curve.d1(0.5 * (range.0 + range.1)).1.norm();
            if (range.1 - range.0) * speed <= tol::LINEAR {
                return Err("a blend whose ends take its whole contact".into());
            }
            let (s, t) = (
                vertex(&mut face, curve.eval(range.0)),
                vertex(&mut face, curve.eval(range.1)),
            );
            let e = face.add_edge(curve.clone(), range, s, t);
            uses.push((e, Use::Contact(k)));
            let (from, to) = if co.reversed {
                (range.1, range.0)
            } else {
                range
            };
            let uvs = swept.coedge_uvs(c, &[from, to]);
            let pcurve = match &co.pcurve {
                PCurve::Projected { .. } => PCurve::Projected { seed: uvs[0] },
                other => other.clone(),
            };
            seed = Some(uvs[1]);
            specs.push(CoedgeSpec {
                edge: e,
                reversed: co.reversed,
                pcurve,
            });
            continue;
        }
        let end = usize::from(lane.caps[1] == c);
        let Some(trim) = &trims[end] else {
            let (s, t) = (
                vertex(&mut face, swept.vertex(edge.start).point),
                vertex(&mut face, swept.vertex(edge.end).point),
            );
            let e = face.add_edge(curve.clone(), edge.range, s, t);
            uses.push((e, Use::Cross));
            let (from, to) = if co.reversed {
                (edge.range.1, edge.range.0)
            } else {
                edge.range
            };
            let uvs = swept.coedge_uvs(c, &[from, to]);
            let pcurve = match &co.pcurve {
                PCurve::Projected { .. } => PCurve::Projected { seed: uvs[0] },
                other => other.clone(),
            };
            seed = Some(uvs[1]);
            specs.push(CoedgeSpec {
                edge: e,
                reversed: co.reversed,
                pcurve,
            });
            continue;
        };
        // The contact the loop has just left is the one these curves start from, or end at.
        let before = lane.order[(first + step + n - 1) % n];
        let leaving = lane
            .contacts
            .iter()
            .position(|&x| x == before)
            .ok_or("a blend face whose two ends meet")?;
        let runs: Vec<Run> = if leaving == trim.first {
            trim.runs.clone()
        } else {
            trim.runs.iter().rev().map(Run::reversed).collect()
        };
        for run in runs {
            let reversed = run.t.0 > run.t.1;
            let range = if reversed { (run.t.1, run.t.0) } else { run.t };
            let (from, to) = (vertex(&mut face, run.from), vertex(&mut face, run.to));
            let (s, t) = if reversed { (to, from) } else { (from, to) };
            let e = face.add_edge(run.curve, range, s, t);
            uses.push((e, Use::Cross));
            let hint = seed.ok_or("a blend face that starts on an end")?;
            let here = surface.project_near(run.from, hint);
            seed = Some(surface.project_near(run.to, here));
            specs.push(CoedgeSpec {
                edge: e,
                reversed,
                pcurve: PCurve::Projected { seed: here },
            });
        }
    }
    let old = swept.face(lane.face);
    let f = face.add_face(surface.clone(), old.reversed, old.names.clone());
    face.add_loop(f, specs);
    Ok(Strip { face, uses })
}
