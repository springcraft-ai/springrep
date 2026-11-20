//! Finding a solid's faces and edges: by the names its faces carry, and by shape.

use std::collections::BTreeSet;

use crate::{
    blend::{self, EdgeName, has_name},
    boolean,
    error::Result,
    math::Point3,
    topo::{Brep, EdgeId, FaceId},
};

/// The faces `name` stands for when an operation picks faces by name: those carrying it, and every
/// face on the surface of one. A name picks by surface because an edge is defined by the surfaces
/// its two faces lie on: where a later feature's face runs on along a named face's surface (a pad
/// flush with a block's side), the edge between them runs on too. Empty when no face carries it.
pub fn faces_named(body: &Brep, name: &str) -> BTreeSet<FaceId> {
    let carriers: Vec<FaceId> = body
        .face_ids()
        .filter(|&f| has_name(body, f, name))
        .collect();
    body.face_ids()
        .filter(|&f| {
            carriers.iter().any(|&c| {
                c == f || boolean::same_domain(&body.face(c).surface, &body.face(f).surface)
            })
        })
        .collect()
}

/// The edges of `b` where its faces meet convexly, the solid's inside a wedge narrower than a half
/// turn. A face's material lies left of its coedge seen from outside, along `n × t`; the edge is
/// convex where the other face's outward normal leans away from that side.
pub fn convex_edges(b: &Brep) -> Result<Vec<EdgeId>> {
    turned_edges(b, true)
}

/// The edges of `b` where its faces meet concavely, the solid's inside a wedge wider than a half
/// turn.
pub(crate) fn concave_edges(b: &Brep) -> Result<Vec<EdgeId>> {
    turned_edges(b, false)
}

/// The edges of `b` whose faces meet at an angle: the convex ones, or the concave ones.
fn turned_edges(b: &Brep, convex: bool) -> Result<Vec<EdgeId>> {
    let outward = |f: FaceId, p: Point3| {
        let s = &b.face(f).surface;
        let uv = s.project(p);
        let n = s.normal(uv.x, uv.y);
        if b.face(f).reversed { -n } else { n }
    };
    let mut out = Vec::new();
    for e in b.edge_ids() {
        let edge = b.edge(e);
        let Some(c) = &edge.curve else { continue };
        let uses: Vec<_> = b.coedge_ids().filter(|&o| b.coedge(o).edge == e).collect();
        let [cf, cg] = uses[..] else { continue };
        let (f, g) = (b.coedge_face(cf), b.coedge_face(cg));
        if f == g {
            continue;
        }
        let (p, dp) = c.d1(0.5 * (edge.range.0 + edge.range.1));
        let (n, m) = (outward(f, p), outward(g, p));
        if n.cross(m).norm() <= 1e-7 {
            continue;
        }
        let t = if b.coedge(cf).reversed { -dp } else { dp };
        if (m.dot(n.cross(t)) < 0.0) == convex {
            out.push(e);
        }
    }
    Ok(out)
}

/// The edges `pair` names, and every edge those run on into tangentially, as a blend of the pair
/// takes them. An error when the pair names no edge.
pub fn edges_named(body: &Brep, pair: &EdgeName) -> Result<Vec<EdgeId>> {
    blend::picked_edges(body, pair)
}
