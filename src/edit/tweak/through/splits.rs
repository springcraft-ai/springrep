//! The two events of a followed move that add topology rather than take it away: a vertex that
//! lands inside an edge splits the edge there, and a vertex of four faces that the move pulls
//! apart becomes two vertices of three faces each, joined by a new edge on the two faces they
//! share (the straight skeleton's split and vertex events, Aichholzer et al. 1995).

use crate::{
    geom::{Curve, Line},
    math::{Point3, Vec2},
    topo::{Brep, CoedgeSpec, EdgeId, FaceId, PCurve, VertexId},
};

use super::super::near_uv;

/// An edge of a rebuilt solid: its curve (none for a pole), its range and its two vertices.
type EdgeSpec = (Option<Curve>, (f64, f64), usize, usize);

/// A use of an edge in a rebuilt loop: the edge, whether the loop runs against it, and the pcurve
/// a pole keeps.
type Use = (usize, bool, Option<PCurve>);

/// `b`'s faces, shells and solids over the vertices `points` and the edges `edges`, each face's
/// loops as `loops` gives them; curved edges take pcurves projected along them.
fn rebuild(b: &Brep, points: &[Point3], edges: &[EdgeSpec], loops: &[Vec<Vec<Use>>]) -> Brep {
    let mut out = Brep::new();
    let vs: Vec<VertexId> = points.iter().map(|&p| out.add_vertex(p)).collect();
    let es: Vec<EdgeId> = edges
        .iter()
        .map(|(c, r, s, t)| match c {
            Some(c) => out.add_edge(c.clone(), *r, vs[*s], vs[*t]),
            None => out.add_degenerate_edge(vs[*s], *r),
        })
        .collect();
    let mut faces = Vec::new();
    for (f, face_loops) in b.face_ids().zip(loops) {
        let face = b.face(f);
        let id = out.add_face(face.surface.clone(), face.reversed, face.names.clone());
        for l in face_loops {
            let mut last: Option<Vec2> = None;
            let mut specs = Vec::new();
            for (e, reversed, pc) in l {
                let edge = es[*e];
                let x = out.edge(edge);
                let pcurve = match &x.curve {
                    None => pc.clone().expect("a pole keeps its pcurve"),
                    Some(cv) => {
                        let from = if *reversed { x.end } else { x.start };
                        let seed = near_uv(&face.surface, out.vertex(from).point, last);
                        let (t0, t1) = if *reversed {
                            (x.range.1, x.range.0)
                        } else {
                            x.range
                        };
                        let mut uv = seed;
                        for k in 1..=32 {
                            let p = cv.eval(t0 + (t1 - t0) * k as f64 / 32.0);
                            uv = near_uv(&face.surface, p, Some(uv));
                        }
                        last = Some(uv);
                        PCurve::Projected { seed }
                    }
                };
                specs.push(CoedgeSpec {
                    edge,
                    reversed: *reversed,
                    pcurve,
                });
            }
            out.add_loop(id, specs);
        }
        faces.push(id);
    }
    for sh in b.shell_ids() {
        let ids = b.shell(sh).faces.iter().map(|f| faces[f.index()]).collect();
        let _ = out.add_shell(ids);
    }
    for so in b.solid_ids() {
        out.add_solid_of(b.solid(so).shells.clone(), b.solid(so).outer);
    }
    out.pair_radial();
    out
}

/// `b`'s vertices and edges as rebuild specs, and each face's loops as uses of them.
fn specs(b: &Brep) -> (Vec<Point3>, Vec<EdgeSpec>, Vec<Vec<Vec<Use>>>) {
    let points = b.vertex_ids().map(|v| b.vertex(v).point).collect();
    let edges = b
        .edge_ids()
        .map(|e| {
            let x = b.edge(e);
            (x.curve.clone(), x.range, x.start.index(), x.end.index())
        })
        .collect();
    let loops = b
        .face_ids()
        .map(|f| {
            b.face(f)
                .loops
                .iter()
                .map(|&lp| {
                    b.lp(lp)
                        .coedges
                        .iter()
                        .map(|&c| {
                            let co = b.coedge(c);
                            let pole = b.edge(co.edge).curve.is_none();
                            (
                                co.edge.index(),
                                co.reversed,
                                pole.then(|| co.pcurve.clone()),
                            )
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    (points, edges, loops)
}

/// `b` with each edge split at every vertex lying within `eps` of its inside, more than `eps` from
/// its ends: where the move has carried a corner onto an edge it did not touch. `None` where no
/// vertex lies so.
pub(super) fn split_at_vertices(b: &Brep, eps: f64) -> Option<Brep> {
    let (points, edges, loops) = specs(b);
    let mut new_edges: Vec<EdgeSpec> = Vec::new();
    // Each old edge's pieces, in order along it.
    let mut pieces: Vec<Vec<usize>> = Vec::new();
    let mut any = false;
    for (curve, range, s, t) in &edges {
        let mut at: Vec<(f64, usize)> = Vec::new();
        if let Some(c) = curve {
            for (v, &p) in points.iter().enumerate() {
                if v == *s || v == *t {
                    continue;
                }
                let u = c.project(p, range.0, range.1);
                let q = c.eval(u);
                if q.dist(p) <= eps
                    && q.dist(points[*s]) > eps
                    && q.dist(points[*t]) > eps
                    && u > range.0
                    && u < range.1
                {
                    at.push((u, v));
                }
            }
        }
        at.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut ids = Vec::new();
        let (mut from_t, mut from_v) = (range.0, *s);
        for &(u, v) in &at {
            ids.push(new_edges.len());
            new_edges.push((curve.clone(), (from_t, u), from_v, v));
            (from_t, from_v) = (u, v);
        }
        ids.push(new_edges.len());
        new_edges.push((curve.clone(), (from_t, range.1), from_v, *t));
        any |= !at.is_empty();
        pieces.push(ids);
    }
    if !any {
        return None;
    }
    let loops: Vec<Vec<Vec<Use>>> = loops
        .into_iter()
        .map(|fl| {
            fl.into_iter()
                .map(|l| {
                    l.into_iter()
                        .flat_map(|(e, rev, pc)| {
                            let mut ids = pieces[e].clone();
                            if rev {
                                ids.reverse();
                            }
                            ids.into_iter().map(move |i| (i, rev, pc.clone()))
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    Some(rebuild(b, &points, &new_edges, &loops))
}

/// The ways to pull vertex `v` of `b` apart, where four faces meet at it about four edges: for
/// each pair of opposite faces, `v` made two vertices, each with the two edges between the faces
/// on one side of the pair, joined by a new straight edge on the pair, laid either way. Empty
/// where `v` is not such a corner.
pub(super) fn corner_splits(b: &Brep, v: VertexId) -> Vec<Brep> {
    // Each face's corner at v: its loop, and the place of the coedge arriving there.
    let mut corners: Vec<(FaceId, usize, usize)> = Vec::new();
    for f in b.face_ids() {
        for (li, &lp) in b.face(f).loops.iter().enumerate() {
            let cs = &b.lp(lp).coedges;
            for k in 0..cs.len() {
                if b.coedge_end(cs[k]) == v && b.coedge_start(cs[(k + 1) % cs.len()]) == v {
                    corners.push((f, li, k));
                }
            }
        }
    }
    if corners.len() != 4 {
        return Vec::new();
    }
    let coedge = |(f, li, k): (FaceId, usize, usize), next: bool| {
        let cs = &b.lp(b.face(f).loops[li]).coedges;
        cs[(k + usize::from(next)) % cs.len()]
    };
    // Round v: the face whose arriving edge is this face's leaving one comes next.
    let mut ring = vec![corners[0]];
    while ring.len() < 4 {
        let out = b.coedge(coedge(ring[ring.len() - 1], true)).edge;
        let Some(&n) = corners
            .iter()
            .find(|&&c| b.coedge(coedge(c, false)).edge == out)
        else {
            return Vec::new();
        };
        if ring.contains(&n) {
            return Vec::new();
        }
        ring.push(n);
    }
    let mut faces: Vec<FaceId> = ring.iter().map(|c| c.0).collect();
    faces.sort();
    faces.dedup();
    if faces.len() != 4 {
        return Vec::new();
    }
    let (points, edges, loops) = specs(b);
    let mut out = Vec::new();
    for d in 0..2 {
        // The new edge lies on ring[d] and ring[d + 2]; the edges about ring[d + 1] keep `v`,
        // those about ring[d + 3] go to the new vertex `w`.
        let (pa, pb) = (ring[d], ring[d + 2]);
        let far = ring[(d + 3) % 4];
        let moving = [
            b.coedge(coedge(far, false)).edge.index(),
            b.coedge(coedge(far, true)).edge.index(),
        ];
        for dir in [1.0, -1.0] {
            let mut points = points.clone();
            let w = points.len();
            points.push(b.vertex(v).point);
            let mut edges = edges.clone();
            for &e in &moving {
                let (_, _, s, t) = &mut edges[e];
                if *s == v.index() {
                    *s = w;
                }
                if *t == v.index() {
                    *t = w;
                }
            }
            let n = b.face(pa.0).surface.normal(0.0, 0.0);
            let m = b.face(pb.0).surface.normal(0.0, 0.0);
            let Some(along) = n.cross(m).unit() else {
                continue;
            };
            let line = Line {
                origin: b.vertex(v).point,
                dir: along * dir,
            };
            let e = edges.len();
            edges.push((Some(Curve::Line(line)), (0.0, 0.0), v.index(), w));
            let mut loops = loops.clone();
            // On `pa` the loop arrives at v from the far side's edge, so it runs w → v; on `pb`
            // it leaves towards the far side, v → w.
            for (c, reversed) in [(pa, true), (pb, false)] {
                let l = &mut loops[c.0.index()][c.1];
                l.insert(c.2 + 1, (e, reversed, None));
            }
            out.push(rebuild(b, &points, &edges, &loops));
        }
    }
    out
}
