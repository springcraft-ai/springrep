//! The sweep of a solid along a motion whose velocity on the body changes as it runs (doc/design.md,
//! "Sweeps of solids", "Moving W"): the region `W(τ)` the motion carries outward slides over the
//! body. What bounds the sweep is found in planes of two parameters, `τ` and a place along a curve
//! of the body: an edge's plane holds where the edge bounds `W(τ)`, a face's plane where its
//! grazing curve lies on it. Both are cut out by the curves where a face grazes the motion along an
//! edge, which are found on a grid; each region is a face of the result on the surface fitted to
//! where the plane's points are carried.

mod charmap;
mod edges;
mod faces;
mod fit;
pub(super) mod graph;
mod region;
#[cfg(test)]
mod tests;
mod zeros;

use std::collections::HashMap;
use std::rc::Rc;

use super::cut::{self, Cut};
use super::motion::{Motion, Twist};
use crate::{
    Error, Result, boolean,
    boolean::{
        domain::Domain,
        face_domain,
        sew::{self, OutFace, PieceGeom},
    },
    geom::{Curve, Surface},
    math::{Point3, Transform, Vec3},
    measure, tol,
    topo::{Brep, CoedgeId, EdgeId, FaceId},
};

/// Most halvings of the motion where its sweep runs into itself.
const DEPTH: usize = 2;

/// The solid `body` passes through as `motion` carries it.
pub(super) fn sweep(body: &Brep, motion: &Motion, feature: &str) -> Result<Brep> {
    // Where a face moves within its own surface for an instant, the side of it the motion carries
    // outward changes over at once; the sweep is the union of the sweeps either side.
    let mut stops = vec![0.0];
    stops.extend(stills(body, motion));
    stops.push(1.0);
    let mut out: Option<Brep> = None;
    let last = stops.len() - 2;
    // A face may move within its own surface where the motion starts or stops, too.
    let held_at = |tau: f64| body.face_ids().any(|f| held(body, motion, f, tau));
    let ends = (held_at(0.0), held_at(1.0));
    for (i, w) in stops.windows(2).enumerate() {
        let span = Span {
            at: (w[0], w[1]),
            held: (i > 0 || ends.0, i < last || ends.1),
        };
        let part = within(body, motion, feature, span, 0)?;
        out = Some(match out {
            None => part,
            Some(o) => union(&o, &part)?,
        });
    }
    out.ok_or_else(|| Error::Numerical("a sweep over no motion".into()))
}

fn union(a: &Brep, b: &Brep) -> Result<Brep> {
    let facts = crate::blend::shared_edges(a, b);
    boolean::boolean_with(a, b, boolean::Op::Union, &facts)
}

/// The sweep over `span` of the motion. Where its boundary runs into itself, it is the body at the
/// span's start united with what each face sweeps where the motion carries it outward (a point the
/// body reaches is in it at the start or is crossed by such a face on the way), and where one of
/// those runs into itself too, the union of the sweeps over the span's two halves.
fn within(body: &Brep, motion: &Motion, feature: &str, whole: Span, depth: usize) -> Result<Brep> {
    let span = whole.at;
    let mut ctx = Ctx::new(body, motion, whole)?;
    ctx.laterals(feature)?;
    // A face that folds back on itself cannot bound the whole: what the faces moving outward
    // sweep is united instead.
    if !ctx.sides.iter().any(|s| s.whole && s.folds) {
        match ctx.assemble(None) {
            Err(Error::InvalidResult(_)) => {}
            other => return other,
        }
    }
    let mut parts = Vec::new();
    for f in 0..ctx.faces.len() {
        if !ctx.moves_out(f) {
            continue;
        }
        match ctx.assemble(Some(f)) {
            Ok(b) => parts.push(b),
            Err(Error::InvalidResult(_)) if depth < DEPTH => {
                let mid = 0.5 * (span.0 + span.1);
                let first = Span {
                    at: (span.0, mid),
                    held: (whole.held.0, false),
                };
                let second = Span {
                    at: (mid, span.1),
                    held: (false, whole.held.1),
                };
                let a = within(body, motion, feature, first, depth + 1)?;
                let b = within(body, motion, feature, second, depth + 1)?;
                return union(&a, &b);
            }
            Err(e) => return Err(e),
        }
    }
    let mut out = body.clone();
    out.transform(&motion.rate(span.0).0);
    for p in &parts {
        out = union(&out, p)?;
    }
    Ok(out)
}

/// A face of the body: its surface, which way is out, its domain, whether the motion keeps it in
/// its own surface throughout.
struct Face {
    id: FaceId,
    surface: Surface,
    out: f64,
    dom: Domain,
    still: bool,
}

/// An edge of the body: its curve and ends, and each of its coedges with its face's index.
struct Edge {
    id: EdgeId,
    curve: Curve,
    range: (f64, f64),
    v0: usize,
    v1: usize,
    sides: Vec<(CoedgeId, usize)>,
}

/// A piece of curve where a face grazes the motion along an edge: its points in the edge's plane
/// of (τ, edge parameter), the face, and the piece of the result it is.
struct Graze {
    edge: usize,
    face: usize,
    pts: Vec<crate::math::Vec2>,
    piece: usize,
    ends: (usize, usize),
}

/// What the sweep over one span is built from.
struct Ctx<'a> {
    body: &'a Brep,
    motion: &'a Motion,
    span: (f64, f64),
    bounds: Span,
    size: f64,
    faces: Vec<Face>,
    edges: Vec<Edge>,
    cut0: Cut,
    cut1: Cut,
    pool: Vec<Point3>,
    pieces: Vec<PieceGeom>,
    /// Each body vertex's places along its path where a face's grazing curve passes it.
    events: HashMap<usize, Vec<(f64, usize)>>,
    /// The piece of a vertex's path between two of its places.
    paths: HashMap<(usize, usize), usize>,
    grazes: Vec<Graze>,
    sides: Vec<Side>,
}

/// A swept face: the body face whose outward-moving part's sweep it bounds, faced out of that
/// sweep, and whether it bounds the whole sweep too (an edge's face where just one face beside the
/// edge moves outward, or a grazing curve's).
struct Side {
    face: OutFace,
    owner: usize,
    whole: bool,
    /// Whether it folds back on itself, past a cusp of the envelope.
    folds: bool,
}

/// A stretch of the motion's parameter, and whether a face moves within its own surface at each
/// end.
#[derive(Clone, Copy)]
struct Span {
    at: (f64, f64),
    held: (bool, bool),
}

/// The velocity of the body's points, in its own frame, at `tau`, read a hair inside the span at an
/// end where a face moves within its own surface.
fn twist_at(motion: &Motion, span: Span, tau: f64) -> Twist {
    let eps = 1e-9 * (span.at.1 - span.at.0);
    let lo = if span.held.0 {
        span.at.0 + eps
    } else {
        f64::NEG_INFINITY
    };
    let hi = if span.held.1 {
        span.at.1 - eps
    } else {
        f64::INFINITY
    };
    motion.body_twist(tau.clamp(lo, hi))
}

/// The points of face `f` of `body` the rate is sampled at, with the outward normal there.
fn face_samples(body: &Brep, f: FaceId) -> Vec<(Point3, Vec3)> {
    let face = body.face(f);
    let out = if face.reversed { -1.0 } else { 1.0 };
    let dom = face_domain(body, f);
    let mut uvs = dom.interior_points();
    uvs.extend(dom.samples().iter().flatten().flatten().copied());
    uvs.iter()
        .map(|uv| {
            (
                face.surface.eval(uv.x, uv.y),
                face.surface.normal(uv.x, uv.y) * out,
            )
        })
        .collect()
}

/// The places inside the motion where a face that moves outward somewhere moves within its own
/// surface for an instant.
fn stills(body: &Brep, motion: &Motion) -> Vec<f64> {
    let size = measure::bbox(body).diagonal();
    let n = 256;
    let speed = (0..=n)
        .map(|k| {
            let tw = motion.body_twist(k as f64 / n as f64);
            tw.omega.norm() * size + tw.v.norm()
        })
        .fold(0.0, f64::max);
    let mut out: Vec<f64> = Vec::new();
    for f in body.face_ids() {
        let samples = face_samples(body, f);
        let rate = |tau: f64| {
            let tw = motion.body_twist(tau);
            samples
                .iter()
                .map(|(p, n)| n.dot(tw.at(*p)).abs())
                .fold(0.0, f64::max)
        };
        let vals: Vec<f64> = (0..=n).map(|k| rate(k as f64 / n as f64)).collect();
        if vals.iter().all(|&v| v <= 1e-9 * speed) {
            continue;
        }
        for k in 1..n {
            if vals[k] > vals[k - 1] || vals[k] > vals[k + 1] || vals[k] > 1e-3 * speed {
                continue;
            }
            // Golden-section search for the least rate between the neighbours.
            let (mut a, mut b) = ((k - 1) as f64 / n as f64, (k + 1) as f64 / n as f64);
            let g = 0.5 * (5f64.sqrt() - 1.0);
            for _ in 0..80 {
                let (x1, x2) = (b - g * (b - a), a + g * (b - a));
                if rate(x1) < rate(x2) {
                    b = x2;
                } else {
                    a = x1;
                }
            }
            let t = 0.5 * (a + b);
            if rate(t) <= 1e-10 * speed && !out.iter().any(|&o| (o - t).abs() <= 1e-9) {
                out.push(t);
            }
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

/// The rate of change of the body's velocity at `tau`, read toward the inside of a span that
/// runs `len` from there (negative where the span ends at `tau`), times that span's sign: the sign
/// the velocity takes on just inside. A one-sided difference of fourth order, so where a face moves
/// within its own surface at `tau`, its grazing curve there is placed as exactly as one inside.
pub(super) fn turning(motion: &Motion, tau: f64, len: f64) -> Twist {
    let h = 1e-3 * len;
    let weights = [-25.0, 48.0, -36.0, 16.0, -3.0];
    let mut out = Twist {
        omega: Vec3::ZERO,
        v: Vec3::ZERO,
    };
    for (k, w) in weights.iter().enumerate() {
        let tw = motion.body_twist(tau + h * k as f64);
        out.omega += tw.omega * *w;
        out.v += tw.v * *w;
    }
    out
}

/// Whether the motion keeps face `f` of `body` in its own surface at `tau`.
pub(super) fn held(body: &Brep, motion: &Motion, f: FaceId, tau: f64) -> bool {
    let tw = motion.body_twist(tau);
    let size = measure::bbox(body).diagonal();
    let speed = tw.omega.norm() * size + tw.v.norm();
    face_samples(body, f)
        .iter()
        .all(|(p, n)| n.dot(tw.at(*p)).abs() <= 1e-9 * speed.max(1e-300))
}

/// The body cut along its grazing curves at the start of `span` (`first`) or its end. A face the
/// motion keeps in its own surface there grazes, as the motion runs on, where the rate of change
/// of its velocity lies in it, read by one-sided differences into the span.
fn end_cut(
    body: &Brep,
    motion: &Motion,
    span: (f64, f64),
    faces: &[Face],
    first: bool,
) -> Result<Cut> {
    let (tau, dir) = if first { (span.0, 1.0) } else { (span.1, -1.0) };
    let tw = motion.body_twist(tau);
    let change = turning(motion, tau, dir * (span.1 - span.0));
    let plain = move |p: Point3| tw.at(p);
    let turned = move |p: Point3| change.at(p);
    let fields: Vec<&dyn Fn(Point3) -> Vec3> = faces
        .iter()
        .map(|f| {
            if !f.still && held(body, motion, f.id, tau) {
                &turned as &dyn Fn(Point3) -> Vec3
            } else {
                &plain as &dyn Fn(Point3) -> Vec3
            }
        })
        .collect();
    cut::cut_by(body, &fields, None)
}

impl<'a> Ctx<'a> {
    fn new(body: &'a Brep, motion: &'a Motion, whole: Span) -> Result<Ctx<'a>> {
        let span = whole.at;
        let size = measure::bbox(body).diagonal();
        let mut faces = Vec::new();
        let n = 64;
        for f in body.face_ids() {
            let face = body.face(f);
            let samples = face_samples(body, f);
            let still = (0..=n).all(|k| {
                let tw = motion.body_twist(span.0 + (span.1 - span.0) * k as f64 / n as f64);
                let speed = tw.omega.norm() * size + tw.v.norm();
                samples
                    .iter()
                    .all(|(p, nn)| nn.dot(tw.at(*p)).abs() <= 1e-9 * speed.max(1e-300))
            });
            let closed = matches!(
                face.surface,
                Surface::Plane(_)
                    | Surface::Cylinder(_)
                    | Surface::Cone(_)
                    | Surface::Extrusion(_)
                    | Surface::Torus(_)
            );
            if !still && !closed {
                return Err(Error::Unsupported(format!(
                    "a sweep along a motion whose velocity changes on the body, of a {} face",
                    face.surface.kind()
                )));
            }
            faces.push(Face {
                id: f,
                surface: face.surface.clone(),
                out: if face.reversed { -1.0 } else { 1.0 },
                dom: face_domain(body, f),
                still,
            });
        }
        let index: HashMap<FaceId, usize> =
            faces.iter().enumerate().map(|(i, f)| (f.id, i)).collect();
        let mut sides: HashMap<EdgeId, Vec<(CoedgeId, usize)>> = HashMap::new();
        for f in &faces {
            for &lp in &body.face(f.id).loops {
                for &c in &body.lp(lp).coedges {
                    sides
                        .entry(body.coedge(c).edge)
                        .or_default()
                        .push((c, index[&f.id]));
                }
            }
        }
        let mut edges = Vec::new();
        for e in body.edge_ids() {
            let edge = body.edge(e);
            let Some(curve) = edge.curve.clone() else {
                continue;
            };
            edges.push(Edge {
                id: e,
                curve,
                range: edge.range,
                v0: edge.start.index(),
                v1: edge.end.index(),
                sides: sides.remove(&e).unwrap_or_default(),
            });
        }
        let cut0 = end_cut(body, motion, span, &faces, true)?;
        let cut1 = end_cut(body, motion, span, &faces, false)?;
        let (pose0, pose1) = (motion.rate(span.0).0, motion.rate(span.1).0);
        let n0 = cut0.pool.len();
        let np0 = cut0.pieces.len();
        let mut pool: Vec<Point3> = cut0.pool.iter().map(|&p| pose0.point(p)).collect();
        pool.extend(cut1.pool.iter().map(|&p| pose1.point(p)));
        let mut pieces = Vec::new();
        for (c, t, shift) in [(&cut0, &pose0, 0), (&cut1, &pose1, n0)] {
            for pg in &c.pieces {
                pieces.push(PieceGeom {
                    curve: pg.curve.as_ref().map(|k| Rc::new(k.transformed(t))),
                    range: pg.range,
                    v0: pg.v0 + shift,
                    v1: pg.v1 + shift,
                    tol: pg.tol,
                });
            }
        }
        debug_assert_eq!(pieces.len(), np0 + cut1.pieces.len());
        Ok(Ctx {
            body,
            motion,
            span,
            bounds: whole,
            size,
            faces,
            edges,
            cut0,
            cut1,
            pool,
            pieces,
            events: HashMap::new(),
            paths: HashMap::new(),
            grazes: Vec::new(),
            sides: Vec::new(),
        })
    }

    fn pose(&self, tau: f64) -> Transform {
        self.motion.rate(tau).0
    }

    fn twist(&self, tau: f64) -> Twist {
        twist_at(self.motion, self.bounds, tau)
    }

    /// The outward normal of face `f` where coedge `c` passes at its edge's parameter `t`.
    fn normal_on(&self, c: CoedgeId, f: usize, t: f64) -> Vec3 {
        let uv = self.body.coedge_uvs(c, &[t])[0];
        self.faces[f].surface.normal(uv.x, uv.y) * self.faces[f].out
    }

    /// A new vertex of the result.
    fn vertex(&mut self, p: Point3) -> usize {
        self.pool.push(p);
        self.pool.len() - 1
    }

    /// The vertex where the path of body vertex `v` is at `tau`, made once for every face whose
    /// grazing curve passes `v` there.
    fn event(&mut self, v: usize, tau: f64) -> usize {
        let reach = 1e-7 * (self.span.1 - self.span.0);
        if let Some(&(_, id)) = self
            .events
            .get(&v)
            .and_then(|l| l.iter().find(|e| (e.0 - tau).abs() <= reach))
        {
            return id;
        }
        let p = self.pose(tau).point(self.cut0.pool[v]);
        let id = self.vertex(p);
        self.events.entry(v).or_default().push((tau, id));
        id
    }

    /// The places along body vertex `v`'s path, from the start of the span to its end.
    fn stops(&self, v: usize) -> Vec<(f64, usize)> {
        let mut list = vec![(self.span.0, v), (self.span.1, v + self.cut0.pool.len())];
        list.extend(self.events.get(&v).cloned().unwrap_or_default());
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        list
    }

    /// The piece of body vertex `v`'s path from `a` to `b`, made once.
    fn path(&mut self, v: usize, a: (f64, usize), b: (f64, usize)) -> Result<usize> {
        if let Some(&p) = self.paths.get(&(a.1, b.1)) {
            return Ok(p);
        }
        // Fitted at τ itself, so its parameter is the plane's along the path.
        let q = self.cut0.pool[v];
        let at = |tau: f64| self.motion.rate(tau).0.point(q);
        let curve = fit::fit_curve(&at, (a.0, b.0))?;
        let range = (a.0, b.0);
        self.pieces.push(PieceGeom {
            curve: Some(Rc::new(curve)),
            range,
            v0: a.1,
            v1: b.1,
            tol: tol::LINEAR,
        });
        let id = self.pieces.len() - 1;
        self.paths.insert((a.1, b.1), id);
        Ok(id)
    }

    /// The faces the edges and the grazing curves sweep.
    fn laterals(&mut self, feature: &str) -> Result<()> {
        edges::grazes(self)?;
        edges::faces(self, feature)?;
        faces::faces(self, feature)
    }

    /// Whether the motion carries any of face `f` outward over the span.
    fn moves_out(&self, f: usize) -> bool {
        self.sides.iter().any(|s| s.owner == f)
            || self.cut0.parts.iter().any(|p| p.face == f && p.front)
            || self.cut1.parts.iter().any(|p| p.face == f && p.front)
    }

    /// The swept faces, closed by the body where the span starts less the faces the motion
    /// carries outward there, and those faces where it ends. With `only`, what that face sweeps
    /// where the motion carries it outward: its swept faces, closed by the parts of it carried
    /// outward at the start, turned inside out, and at the end.
    fn assemble(&self, only: Option<usize>) -> Result<Brep> {
        let mut out: Vec<OutFace> = self
            .sides
            .iter()
            .filter(|s| match only {
                None => s.whole,
                Some(f) => s.owner == f,
            })
            .map(|s| s.face.clone())
            .collect();
        let n0 = self.cut0.pool.len();
        let np0 = self.cut0.pieces.len();
        let curves: Vec<Option<Rc<Curve>>> = self.pieces.iter().map(|p| p.curve.clone()).collect();
        for (cut, t, start, shift, pshift) in [
            (&self.cut0, self.pose(self.span.0), true, 0, 0),
            (&self.cut1, self.pose(self.span.1), false, n0, np0),
        ] {
            for part in &cut.parts {
                let keep = match only {
                    None => part.front != start,
                    Some(f) => part.face == f && part.front,
                };
                if !keep {
                    continue;
                }
                let loops = part
                    .loops
                    .iter()
                    .map(|l| {
                        l.iter()
                            .map(|tr| {
                                let p = tr.piece + pshift;
                                crate::boolean::domain::Trim {
                                    piece: p,
                                    curve: curves[p].clone(),
                                    pole: t.point(tr.pole),
                                    v0: tr.v0 + shift,
                                    v1: tr.v1 + shift,
                                    ..tr.clone()
                                }
                            })
                            .collect()
                    })
                    .collect();
                let mut face = OutFace {
                    surface: part.surface.transformed(&t),
                    reversed: part.reversed,
                    names: part.names.clone(),
                    loops,
                    memo: None,
                    side: None,
                };
                if start && only.is_some() {
                    face.flip();
                }
                out.push(face);
            }
        }
        let vtol = vec![tol::LINEAR; self.pool.len()];
        if only.is_none() {
            // Where the whole is refused the sweep is built face by face, whatever the refusal
            // lists.
            return sew::assemble_or_refuse(out, &self.pool, &vtol, &self.pieces);
        }
        sew::assemble(out, &self.pool, &vtol, &self.pieces)
    }
}
