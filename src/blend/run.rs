//! An end whose contact curve runs on past a short edge at its vertex: the strip a blend takes off
//! a face reaches beyond the next edge of that face's loop, onto the faces past it. Walking the
//! face's loop from the vertex finds the edge the contact curve finally crosses; every edge before
//! it lies wholly inside the strip and goes with the blend, and so does each vertex where two of
//! them meet. The faces across the walked edges bound the blend's end, so the piece is trimmed by
//! each of their surfaces as it is by the face at its vertex. The walk stops at an edge that rises
//! off the end face, away from the piece, which ends beneath it; and, for a concave piece, at a
//! convex edge its strip runs over.

use std::collections::BTreeSet;

use super::{
    Clip, Outline, Picked, Section, Trace, about, faces_at, in_section, inward_point, outward,
};
use crate::{
    geom::Surface,
    math::Point3,
    tol,
    topo::{Brep, CoedgeId, EdgeId, FaceId, VertexId},
};

/// Why a blend too big for its faces is refused: it would take a whole face, leaving nothing for
/// it to be tangent to. That is the request's fault, where every other reason here is a blend this
/// kernel does not build yet.
pub(crate) const WHOLE_FACE: &str = "a blend that takes a whole face";

/// What a piece's end at a vertex takes off the body past its own edge.
#[derive(Clone, Debug, Default)]
pub(super) struct Beyond {
    /// Vertices past the edge's end that the blend takes away.
    pub gone: Vec<VertexId>,
    /// Edges that lie wholly inside the blend.
    pub dropped: Vec<EdgeId>,
    /// Surfaces of the faces past the end, other than the one at its vertex, that trim the piece.
    pub extra: Vec<Clip>,
    /// Whether the face past the short edge rises off the end face, away from the piece: the
    /// piece stops on the end face beneath it, and the vertex where they meet stays, so `gone`
    /// does not say what sewing would need.
    pub overhung: bool,
    /// Whether a contact curve ends on a vertex past the edge, which sewing cannot place.
    pub on_vertex: bool,
}

impl Beyond {
    /// Whether the end takes nothing past its own edge.
    pub(super) fn takes_nothing(&self) -> bool {
        self.gone.is_empty()
            && self.dropped.is_empty()
            && self.extra.is_empty()
            && !self.overhung
            && !self.on_vertex
    }
}

/// Whether the edge from `near` to `far` of `curve` leaves `v` onto the outer side of the end face
/// `end_face` there: the piece, which stops on the end face, never reaches it.
fn rises_off(
    body: &Brep,
    end_face: FaceId,
    v: VertexId,
    curve: &crate::geom::Curve,
    (near, far): (f64, f64),
    samples: usize,
) -> bool {
    let at = body.vertex(v).point;
    let n = outward(body, end_face, at);
    let off: Vec<f64> = (1..=samples)
        .map(|i| (curve.eval(near + (far - near) * i as f64 / samples as f64) - at).dot(n))
        .collect();
    off.iter().all(|&x| x >= -tol::LINEAR) && off.iter().any(|&x| x > tol::LINEAR)
}

/// The one face at `v` other than `pk`'s own two.
fn end_face(body: &Brep, pk: &Picked, v: VertexId) -> Option<FaceId> {
    let others: Vec<FaceId> = faces_at(body, v)
        .into_iter()
        .filter(|f| !pk.faces.contains(f))
        .collect();
    match others.as_slice() {
        [f] => Some(*f),
        _ => None,
    }
}

/// How far the point `q` of face `k` lies past the contact curve on it, across the strip the blend
/// takes off that face: negative between the edge and the contact.
fn past_contact(s: &Section, o: &Outline, k: usize, q: Point3) -> f64 {
    let (q2, _) = in_section(s, q);
    let (dir, width) = o.reach[k];
    let t = s.p + dir * width;
    match s.trace[k] {
        Trace::Line => (q2 - t).dot(s.m[k]),
        Trace::Circle { centre, radius } => {
            let (a, b) = (t - centre, q2 - centre);
            let turn = a.cross(b).atan2(a.dot(b));
            let sense = if a.perp().dot(t - s.p) >= 0.0 {
                1.0
            } else {
                -1.0
            };
            turn * radius * sense
        }
    }
}

/// The other coedge of `c`'s edge.
fn partner(body: &Brep, c: CoedgeId) -> Option<CoedgeId> {
    let e = body.coedge(c).edge;
    body.coedge_ids()
        .find(|&o| o != c && body.coedge(o).edge == e)
}

/// The face across the edge of coedge `c` from `c`'s own.
pub(super) fn across(body: &Brep, c: CoedgeId) -> Option<FaceId> {
    partner(body, c).map(|o| body.coedge_face(o))
}

/// Whether the edge of coedge `c` is convex: its faces' material meets at less than a half turn.
pub(super) fn convex(body: &Brep, c: CoedgeId) -> Option<bool> {
    let o = partner(body, c)?;
    let edge = body.edge(body.coedge(c).edge);
    let curve = edge.curve.as_ref()?;
    let t = 0.5 * (edge.range.0 + edge.range.1);
    let (p, d) = curve.d1(t);
    let along = if body.coedge(c).reversed { -d } else { d };
    let (fa, fb) = (body.coedge_face(c), body.coedge_face(o));
    let m = outward(body, fa, p).cross(along);
    let bend = m.dot(outward(body, fb, p));
    (bend.abs() > tol::TANGENT * m.norm()).then_some(bend < 0.0)
}

/// The edges of face `k`'s loop past `v` that the end at `v` answers for: each one the strip covers
/// wholly, then the first its contact curve crosses. Empty when the face turns away from the strip
/// at `v`.
pub(super) fn chain(
    body: &Brep,
    pk: &Picked,
    s: &Section,
    o: &Outline,
    k: usize,
    v: VertexId,
) -> Vec<EdgeId> {
    const SAMPLES: usize = 32;
    let c0 = pk.coedges[k];
    let lp = body.lp(body.coedge(c0).lp);
    let n = lp.coedges.len();
    let Some(at) = lp.coedges.iter().position(|&c| c == c0) else {
        return Vec::new();
    };
    let forward = body.coedge_end(c0) == v;
    let end = end_face(body, pk, v);
    let mut out = Vec::new();
    for step in 1..n {
        let c = lp.coedges[if forward {
            (at + step) % n
        } else {
            (at + n - step) % n
        }];
        let e = body.coedge(c).edge;
        let edge = body.edge(e);
        let Some(curve) = edge.curve.as_ref() else {
            break;
        };
        let (near, far) = if forward == body.coedge(c).reversed {
            (edge.range.1, edge.range.0)
        } else {
            (edge.range.0, edge.range.1)
        };
        let f = |t: f64| past_contact(s, o, k, curve.eval(t));
        if step == 1 && turns_away(f, near, far, SAMPLES) {
            break;
        }
        if step > 1 && end.is_some_and(|g| rises_off(body, g, v, curve, (near, far), SAMPLES)) {
            break;
        }
        out.push(e);
        let crosses = (0..=SAMPLES)
            .map(|i| f(near + (far - near) * i as f64 / SAMPLES as f64))
            .any(|x| x > tol::LINEAR);
        if crosses || f(far) >= -tol::LINEAR || spills_over(body, s, c, step) {
            break;
        }
    }
    out
}

/// Whether the edge after the vertex, `f` its points' distance past the contact from `near` to
/// `far`, keeps out of the strip: it ends no further into it than it starts, or never gets further
/// in at all, as an edge that bows out behind the picked one's line and comes back to it does.
fn turns_away(f: impl Fn(f64) -> f64, near: f64, far: f64, samples: usize) -> bool {
    let start = f(near);
    f(far) <= start
        || (1..=samples)
            .map(|i| f(near + (far - near) * i as f64 / samples as f64))
            .all(|x| x <= start + tol::LINEAR)
}

/// Whether a concave strip runs over the edge of `c`, `step` edges past the end, onto the face
/// across it: a convex edge past the end's own short edge bounds the body under the strip there,
/// and the face across it trims the blend ([`super::Spill`]) rather than the walk going on round.
fn spills_over(body: &Brep, s: &Section, c: CoedgeId, step: usize) -> bool {
    !s.convex && step > 1 && convex(body, c) == Some(true)
}

/// What the convex piece of `pk` takes off past its end at `v`, where the face at `v` other than
/// its own two is `end_face`; empty when each contact curve meets the first edge of its face past
/// `v`. An error says why the end cannot be sewn.
pub(super) fn beyond(
    body: &Brep,
    picked: &[Picked],
    pk: &Picked,
    s: &Section,
    o: &Outline,
    v: VertexId,
) -> Result<Beyond, String> {
    const SAMPLES: usize = 32;
    let Some(end_face) = &end_face(body, pk, v) else {
        return Ok(Beyond::default());
    };
    let is_picked = |e: EdgeId| picked.iter().any(|p| p.edge == e);
    let mut out = Beyond::default();
    let mut walls: BTreeSet<FaceId> = BTreeSet::new();
    let mut joints: Vec<CoedgeId> = Vec::new();
    for k in 0..2 {
        let c0 = pk.coedges[k];
        let lp = body.lp(body.coedge(c0).lp);
        let n = lp.coedges.len();
        let at = lp
            .coedges
            .iter()
            .position(|&c| c == c0)
            .ok_or("a coedge missing from its loop")?;
        let forward = body.coedge_end(c0) == v;
        let mut walked: Vec<CoedgeId> = Vec::new();
        for step in 1..n {
            let c = lp.coedges[if forward {
                (at + step) % n
            } else {
                (at + n - step) % n
            }];
            let e = body.coedge(c).edge;
            if is_picked(e) {
                return Err("a blend that runs on past a short edge onto another blend".into());
            }
            let edge = body.edge(e);
            let Some(curve) = edge.curve.as_ref() else {
                return Err("a blend that runs on past a pole".into());
            };
            let (near, far) = if forward == body.coedge(c).reversed {
                (edge.range.1, edge.range.0)
            } else {
                (edge.range.0, edge.range.1)
            };
            let f = |t: f64| past_contact(s, o, k, curve.eval(t));
            let f_far = f(far);
            if step == 1 && turns_away(f, near, far, SAMPLES) {
                // The face turns away from the strip at `v`: its edge runs on past the vertex.
                break;
            }
            if step > 1 && rises_off(body, *end_face, v, curve, (near, far), SAMPLES) {
                out.overhung = true;
                break;
            }
            let crosses = (0..=SAMPLES)
                .map(|i| f(near + (far - near) * i as f64 / SAMPLES as f64))
                .any(|x| x > tol::LINEAR);
            walked.push(c);
            // A contact that ends on the edge's far vertex leaves that vertex where it is, on the
            // contact; the trims are the same, but sewing has no point to cut the edge back to.
            if f_far.abs() <= tol::LINEAR {
                out.on_vertex = true;
                break;
            }
            if crosses {
                break;
            }
            if spills_over(body, s, c, step) {
                break;
            }
            let w = if forward {
                body.coedge_end(c)
            } else {
                body.coedge_start(c)
            };
            out.gone.push(w);
            out.dropped.push(e);
            if step == n - 1 {
                return Err(WHOLE_FACE.into());
            }
        }
        if walked.len() <= 1 {
            continue;
        }
        for &c in &walked {
            let across = partner(body, c).ok_or("an edge used once")?;
            walls.insert(body.coedge_face(across));
        }
        // Where two walked edges meet, the faces across them meet too, at the edges into the
        // vertex they share.
        for pair in walked.windows(2) {
            let w = if forward {
                body.coedge_end(pair[0])
            } else {
                body.coedge_start(pair[0])
            };
            for c in body.coedge_ids() {
                let x = body.edge(body.coedge(c).edge);
                if (x.start == w || x.end == w) && !walked.contains(&c) {
                    joints.push(c);
                }
            }
        }
    }
    if out.gone.is_empty() && !out.on_vertex {
        return Ok(Beyond::default());
    }
    for &c in &joints {
        let Some(o) = partner(body, c) else {
            continue;
        };
        let (fa, fb) = (body.coedge_face(c), body.coedge_face(o));
        let bounding = |f: FaceId| walls.contains(&f) || f == *end_face;
        if bounding(fa) && bounding(fb) && fa != fb && convex(body, c) != Some(true) {
            return Err("a blend that runs on past a short edge into a concave corner".into());
        }
    }
    let at = body.vertex(v).point;
    let q = inward_point(body, pk.edge, v, o.radius);
    for &f in &walls {
        if f == *end_face || pk.faces.contains(&f) {
            continue;
        }
        let clip = match &body.face(f).surface {
            Surface::Plane(p) => {
                let n = outward(body, f, at);
                let keep = if (q - p.frame.origin).dot(n) >= 0.0 {
                    n
                } else {
                    -n
                };
                Clip::Plane {
                    origin: p.frame.origin,
                    keep,
                }
            }
            Surface::Cylinder(c) => about(c.frame, c.radius, 0.0, q),
            Surface::Cone(c) => about(c.frame, c.radius, c.half_angle.tan(), q),
            other => {
                return Err(format!(
                    "a blend that runs on past a short edge onto a {}",
                    other.kind()
                ));
            }
        };
        out.extra.push(clip);
    }
    Ok(out)
}
