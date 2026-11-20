//! Dissolving redundant vertices: a vertex only two edges meet, both between the same two faces and
//! on one curve (one line, one circle, or the same curve running on), divides nothing. A boolean
//! leaves such vertices where an imprint split an edge that the result then keeps whole on both
//! sides; here each such pair of edges becomes one edge.

use std::collections::{BTreeSet, HashMap};

use crate::{
    geom::Curve,
    math::Point3,
    tol,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, VertexId},
};

/// `b` with every redundant vertex dissolved, merging the two edges at it, until none is left.
/// Faces, names, shells and local tolerances are kept; a vertex or edge with a local tolerance is
/// never merged.
pub(crate) fn merge_edges(b: Brep) -> Brep {
    let mut b = b;
    for _ in 0..64 {
        match merge_once(&b) {
            Some(next) => b = next,
            None => break,
        }
    }
    b
}

/// The parameter on `c` of `p`, for the curves merging continues: a line's or a circle's.
fn param(c: &Curve, p: Point3, near: f64) -> Option<f64> {
    match c {
        Curve::Line(l) => Some((p - l.origin).dot(l.dir)),
        Curve::Circle(ci) => {
            let q = ci.frame.to_local(p);
            let a = q.y.atan2(q.x);
            Some(a + ((near - a) / std::f64::consts::TAU).round() * std::f64::consts::TAU)
        }
        _ => None,
    }
}

/// Whether curve `b` lies on curve `a`: one line, one circle, or the same curve.
fn continues(a: &Curve, b: &Curve) -> bool {
    match (a, b) {
        (Curve::Line(x), Curve::Line(y)) => {
            let off = (y.origin - x.origin) - x.dir * (y.origin - x.origin).dot(x.dir);
            x.dir.cross(y.dir).norm() <= tol::ANGULAR && off.norm() <= tol::LINEAR
        }
        (Curve::Circle(x), Curve::Circle(y)) => {
            (x.radius - y.radius).abs() <= tol::LINEAR
                && x.frame.origin.dist(y.frame.origin) <= tol::LINEAR
                && x.frame.z.cross(y.frame.z).norm() <= tol::ANGULAR
                && x.frame.z.dot(y.frame.z) > 0.0
        }
        _ => a == b,
    }
}

fn merge_once(b: &Brep) -> Option<Brep> {
    let (tol_vs, tol_es) = b.tolerant();
    let tolerant_v: BTreeSet<VertexId> = tol_vs.iter().map(|x| x.0).collect();
    let tolerant_e: BTreeSet<EdgeId> = tol_es.iter().map(|x| x.0).collect();
    let mut at: HashMap<VertexId, Vec<EdgeId>> = HashMap::new();
    for e in b.edge_ids() {
        let x = b.edge(e);
        at.entry(x.start).or_default().push(e);
        if x.end != x.start {
            at.entry(x.end).or_default().push(e);
        }
    }
    let mut uses: HashMap<EdgeId, Vec<CoedgeId>> = HashMap::new();
    for c in b.coedge_ids() {
        uses.entry(b.coedge(c).edge).or_default().push(c);
    }
    let faces = |e: EdgeId| -> BTreeSet<FaceId> {
        uses.get(&e)
            .map(|cs| cs.iter().map(|&c| b.coedge_face(c)).collect())
            .unwrap_or_default()
    };
    // The first redundant vertex: its two edges, `first` running into it and `second` out of it.
    let mut vertices: Vec<&VertexId> = at.keys().collect();
    vertices.sort();
    let (v, first, second) = vertices.into_iter().find_map(|&v| {
        let es = &at[&v];
        let [e1, e2] = es[..] else {
            return None;
        };
        if e1 == e2
            || tolerant_v.contains(&v)
            || tolerant_e.contains(&e1)
            || tolerant_e.contains(&e2)
        {
            return None;
        }
        let (x1, x2) = (b.edge(e1), b.edge(e2));
        let (c1, c2) = (x1.curve.as_ref()?, x2.curve.as_ref()?);
        let shared = faces(e1);
        if shared.len() != 2 || shared != faces(e2) || uses[&e1].len() != 2 || uses[&e2].len() != 2
        {
            return None;
        }
        // The merged edge would close on itself, or the two run back over each other.
        let other = |x: &crate::topo::Edge| if x.start == v { x.end } else { x.start };
        if other(x1) == other(x2) || !continues(c1, c2) {
            return None;
        }
        // Orient: e1 must end at v and e2 start there, along one parameter.
        let (first, second) = if x1.end == v && x2.start == v {
            (e1, e2)
        } else if x2.end == v && x1.start == v {
            (e2, e1)
        } else {
            return None;
        };
        Some((v, first, second))
    })?;

    // A spline pcurve holds parameters over its own edge's range alone.
    if b.coedge_ids().any(|c| {
        let co = b.coedge(c);
        (co.edge == first || co.edge == second)
            && matches!(co.pcurve, crate::topo::PCurve::Spline { .. })
    }) {
        return None;
    }
    let (x1, x2) = (b.edge(first), b.edge(second));
    let c1 = x1.curve.clone()?;
    let c2 = x2.curve.as_ref()?;
    // The merged edge on the first edge's curve, through the second's far end.
    let end = c2.eval(x2.range.1);
    let range = if c1 == *c2 {
        (x1.range.0, x2.range.1)
    } else {
        (
            x1.range.0,
            param(&c1, end, x1.range.1 + (x2.range.1 - x2.range.0))?,
        )
    };
    if range.1 <= range.0 || c1.eval(range.1).dist(end) > tol::LINEAR {
        return None;
    }

    let mut out = Brep::new();
    let mut vmap: Vec<Option<VertexId>> = vec![None; b.vertex_ids().count()];
    for w in b.vertex_ids() {
        if w != v {
            vmap[w.index()] = Some(out.add_vertex(b.vertex(w).point));
        }
    }
    let mut emap: Vec<Option<EdgeId>> = vec![None; b.edge_ids().count()];
    for e in b.edge_ids() {
        if e == second {
            continue;
        }
        let x = b.edge(e);
        let (curve, rng, s, t) = if e == first {
            (Some(c1.clone()), range, x1.start, x2.end)
        } else {
            (x.curve.clone(), x.range, x.start, x.end)
        };
        let ne = match curve {
            Some(c) => out.add_edge(c, rng, vmap[s.index()]?, vmap[t.index()]?),
            None => out.add_degenerate_edge(vmap[s.index()]?, rng),
        };
        emap[e.index()] = Some(ne);
    }
    let mut cmap: HashMap<CoedgeId, CoedgeId> = HashMap::new();
    for f in b.face_ids() {
        let face = b.face(f);
        let nf = out.add_face(
            face.surface.clone(),
            face.reversed,
            face.names.iter().cloned(),
        );
        for &lp in &face.loops {
            let mut specs = Vec::new();
            let mut kept = Vec::new();
            for &c in &b.lp(lp).coedges {
                let co = b.coedge(c);
                if co.edge == second {
                    continue;
                }
                kept.push(c);
                let edge = emap[co.edge.index()]?;
                let mut pcurve = co.pcurve.clone();
                // Running against the merged edge, the coedge now starts at the second edge's far
                // end, so a projected pcurve's seed moves there.
                if co.edge == first
                    && co.reversed
                    && let crate::topo::PCurve::Projected { seed } = &co.pcurve
                {
                    let other = uses[&second].iter().find(|&&o| b.coedge_face(o) == f)?;
                    let uv = b.coedge_uvs(*other, &[x2.range.1])[0];
                    let _ = seed;
                    pcurve = crate::topo::PCurve::Projected { seed: uv };
                }
                specs.push(CoedgeSpec {
                    edge,
                    reversed: co.reversed,
                    pcurve,
                });
            }
            let nl = out.add_loop(nf, specs);
            for (c, &n) in kept.into_iter().zip(&out.lp(nl).coedges) {
                cmap.insert(c, n);
            }
        }
    }
    // Where pieces touch along an edge, which of its uses pair across material, and which face
    // each other across space, carries over.
    for c in b.coedge_ids() {
        if let (Some(&n), Some(&m)) = (cmap.get(&c), b.partner(c).and_then(|p| cmap.get(&p))) {
            out.pair(n, m);
        }
        if let (Some(&n), Some(&m)) = (cmap.get(&c), b.beside(c).and_then(|p| cmap.get(&p))) {
            out.set_beside(n, m);
        }
    }
    for sh in b.shell_ids() {
        out.add_shell(b.shell(sh).faces.clone());
    }
    for so in b.solid_ids() {
        out.add_solid_of(b.solid(so).shells.clone(), b.solid(so).outer);
    }
    for (w, t) in tol_vs {
        out.set_vertex_tolerance(vmap[w.index()]?, t);
    }
    for (e, t) in tol_es {
        out.set_edge_tolerance(emap[e.index()]?, t);
    }
    Some(out)
}
