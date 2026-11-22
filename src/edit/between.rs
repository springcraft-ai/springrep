//! The solid between a solid and the same solid with some faces moved, the smaller taken from the
//! larger: a shell's wall, and the slab thickened faces sweep.
//!
//! The two solids share every face that did not move, but for the strips those faces gained or
//! lost beside the moved ones. What lies between them is bounded by the larger's moved faces, the
//! smaller's turned to face the other way, and those strips, each bounded by the larger's edges,
//! the smaller's run backwards, and the stretches of the edges that lead up to both, between the
//! corner of one and the corner of the other. Nothing is intersected: every surface and curve is
//! one of the two solids' own.

use std::collections::HashMap;

use super::local;
use crate::{
    check,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, LoopId, PCurve, VertexId},
};

/// Two solids with one outer boundary but for the faces that moved.
pub(crate) struct Pairing<'a> {
    /// The solid that holds the other.
    pub big: &'a Brep,
    pub small: &'a Brep,
    /// For each face of `big`, whether it is its own: moved, or a face `small` has nothing of.
    pub own_big: Vec<bool>,
    /// For each face of `small`, whether it is its own.
    pub own_small: Vec<bool>,
    /// The other faces, each as its face of `big` and its face of `small` on the same surface.
    pub same: Vec<(FaceId, FaceId)>,
    /// Whether the faces of `same` have their loops and coedges in the same order in both, as one
    /// solid moved with its topology kept has. Otherwise only faces whose every edge is beside a
    /// face that moved are paired, each of one loop.
    pub kept: bool,
}

/// One of the two solids, and what of it the result holds so far.
struct Side<'a> {
    b: &'a Brep,
    verts: HashMap<VertexId, VertexId>,
    edges: HashMap<EdgeId, EdgeId>,
}

impl<'a> Side<'a> {
    fn new(b: &'a Brep) -> Side<'a> {
        Side {
            b,
            verts: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    fn vertex(&mut self, out: &mut Brep, v: VertexId) -> VertexId {
        let b = self.b;
        *self
            .verts
            .entry(v)
            .or_insert_with(|| out.add_vertex(b.vertex(v).point))
    }

    fn edge(&mut self, out: &mut Brep, e: EdgeId) -> EdgeId {
        if let Some(&id) = self.edges.get(&e) {
            return id;
        }
        let x = self.b.edge(e);
        let (s, t) = (self.vertex(out, x.start), self.vertex(out, x.end));
        let id = match &x.curve {
            Some(c) => out.add_edge(c.clone(), x.range, s, t),
            None => out.add_degenerate_edge(s, x.range),
        };
        self.edges.insert(e, id);
        id
    }

    /// Coedge `c` as the result uses it, run backwards if `flip`.
    fn spec(&mut self, out: &mut Brep, c: CoedgeId, flip: bool) -> CoedgeSpec {
        let b = self.b;
        let co = b.coedge(c);
        let pcurve = match &co.pcurve {
            // A projected pcurve continues from where its coedge starts: backwards, its old end.
            PCurve::Projected { .. } if flip => {
                let ts = check::coedge_params(b, c, 32);
                let uvs = b.coedge_uvs(c, &ts);
                PCurve::Projected {
                    seed: uvs[uvs.len() - 1],
                }
            }
            p => p.clone(),
        };
        CoedgeSpec {
            edge: self.edge(out, co.edge),
            reversed: co.reversed != flip,
            pcurve,
        }
    }

    fn lp(&mut self, out: &mut Brep, lp: LoopId, flip: bool) -> Vec<CoedgeSpec> {
        let cs = &self.b.lp(lp).coedges;
        if flip {
            cs.iter().rev().map(|&c| self.spec(out, c, true)).collect()
        } else {
            cs.iter().map(|&c| self.spec(out, c, false)).collect()
        }
    }

    /// Face `f` whole, turned to face the other way if `flip`.
    fn face(&mut self, out: &mut Brep, f: FaceId, flip: bool) -> FaceId {
        let b = self.b;
        let face = b.face(f);
        let id = out.add_face(
            face.surface.clone(),
            face.reversed != flip,
            face.names.clone(),
        );
        for &lp in &face.loops {
            let specs = self.lp(out, lp, flip);
            out.add_loop(id, specs);
        }
        if !flip {
            out.share_memo(id, b.shared_memo(f));
        }
        id
    }

    /// Whether coedge `c` runs along a face of `own`.
    fn beside(&self, own: &[bool], c: CoedgeId) -> bool {
        self.b
            .partner(c)
            .is_some_and(|o| own[self.b.coedge_face(o).index()])
    }
}

fn root(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

/// `big` less `small`, built from their own faces; `None` where the two are not paired as the
/// strips need, or the result does not pass the checker.
///
/// That is the difference where `small` lies in `big`. The faces built are the two boundaries
/// with what they share left out, `small`'s turned over, so a solid they bound is the points in
/// one of the two and not the other. Each strip is laid as a piece `big` has and `small` has not:
/// a corner of `small` must lie on `big`'s edge short of `big`'s corner, and a loop of `small`
/// taken whole lies in `big`'s as a hole, or round `big`'s hole. Where `small` reached out of
/// `big` instead, a strip or a shell would be turned inside out, which the checker refuses, as it
/// does faces that meet away from their edges.
pub(crate) fn between(p: &Pairing) -> Option<Brep> {
    if !local::direct_allowed() {
        return None;
    }
    let (big, small) = (p.big, p.small);
    if [big, small]
        .iter()
        .any(|b| b.solid_ids().count() != 1 || b.shell_ids().count() != 1)
    {
        return None;
    }
    let shared = |own: &[bool]| own.iter().filter(|&&o| !o).count();
    if shared(&p.own_big) != p.same.len() || shared(&p.own_small) != p.same.len() {
        return None;
    }
    let mut out = Brep::new();
    let (mut bs, mut ss) = (Side::new(big), Side::new(small));
    let mut outer = Vec::new();
    for f in big.face_ids().filter(|f| p.own_big[f.index()]) {
        outer.push(bs.face(&mut out, f, false));
    }
    let mut inner = Vec::new();
    for f in small.face_ids().filter(|f| p.own_small[f.index()]) {
        inner.push(ss.face(&mut out, f, true));
    }
    // The stretch of each of `big`'s edges past `small`'s corner on it, by the edge and its end.
    let mut pieces: HashMap<(EdgeId, bool), EdgeId> = HashMap::new();
    let mut strips = 0;
    let mut seen = (vec![false; p.own_big.len()], vec![false; p.own_small.len()]);
    for &(n, m) in &p.same {
        if p.own_big[n.index()] || p.own_small[m.index()] {
            return None;
        }
        if std::mem::replace(&mut seen.0[n.index()], true)
            || std::mem::replace(&mut seen.1[m.index()], true)
        {
            return None;
        }
        let (fb, fs) = (big.face(n), small.face(m));
        if fb.surface != fs.surface
            || fb.reversed != fs.reversed
            || fb.loops.len() != fs.loops.len()
            || (!p.kept && fb.loops.len() != 1)
        {
            return None;
        }
        for (k, (&lb, &ls)) in fb.loops.iter().zip(&fs.loops).enumerate() {
            let (cb, cs) = (&big.lp(lb).coedges, &small.lp(ls).coedges);
            let cut_b: Vec<bool> = cb.iter().map(|&c| bs.beside(&p.own_big, c)).collect();
            let cut_s: Vec<bool> = cs.iter().map(|&c| ss.beside(&p.own_small, c)).collect();
            if cut_b.iter().all(|&c| c) && cut_s.iter().all(|&c| c) {
                // The whole loop moved: the strip lies between it and where it was.
                let id = out.add_face(fb.surface.clone(), fb.reversed, fb.names.clone());
                let (ours, theirs) = (bs.lp(&mut out, lb, false), ss.lp(&mut out, ls, true));
                let loops = if k == 0 {
                    [ours, theirs]
                } else {
                    [theirs, ours]
                };
                for specs in loops {
                    out.add_loop(id, specs);
                }
                outer.push(id);
                strips += 1;
                continue;
            }
            if !p.kept || cut_b != cut_s {
                return None;
            }
            let n = cb.len();
            // A corner no moved face is at stays where it was.
            for j in 0..n {
                let moves = cut_b[j] || cut_b[(j + n - 1) % n];
                let (x, y) = (big.coedge_start(cb[j]), small.coedge_start(cs[j]));
                if !moves && big.vertex(x).point != small.vertex(y).point {
                    return None;
                }
            }
            for j0 in (0..n).filter(|&j| cut_b[j] && !cut_b[(j + n - 1) % n]) {
                let mut j1 = j0;
                while cut_b[(j1 + 1) % n] {
                    j1 = (j1 + 1) % n;
                }
                let (before, after) = ((j0 + n - 1) % n, (j1 + 1) % n);
                let mut specs = Vec::new();
                let from = small.coedge_start(cs[j0]);
                specs.push(piece(
                    &mut out,
                    (&mut bs, &mut ss),
                    &mut pieces,
                    cb[before],
                    true,
                    from,
                )?);
                let mut j = j0;
                loop {
                    specs.push(bs.spec(&mut out, cb[j], false));
                    if j == j1 {
                        break;
                    }
                    j = (j + 1) % n;
                }
                let to = small.coedge_end(cs[j1]);
                specs.push(piece(
                    &mut out,
                    (&mut bs, &mut ss),
                    &mut pieces,
                    cb[after],
                    false,
                    to,
                )?);
                let mut j = j1;
                loop {
                    specs.push(ss.spec(&mut out, cs[j], true));
                    if j == j0 {
                        break;
                    }
                    j = (j + n - 1) % n;
                }
                let id = out.add_face(fb.surface.clone(), fb.reversed, fb.names.clone());
                out.add_loop(id, specs);
                outer.push(id);
                strips += 1;
            }
        }
    }
    // One shell where the strips join the two solids' own faces; with no strip, the smaller's
    // faces are a void in the larger's.
    let count = out.face_count();
    let mut sheet: Vec<usize> = (0..count).collect();
    for e in out.edge_ids() {
        let us = out.edge_uses(e);
        for w in us.windows(2) {
            let (x, y) = (
                root(&mut sheet, out.coedge_face(w[0]).index()),
                root(&mut sheet, out.coedge_face(w[1]).index()),
            );
            sheet[x.max(y)] = x.min(y);
        }
    }
    let mut together = |ids: &[FaceId]| {
        let first = root(&mut sheet, ids.first()?.index());
        ids.iter()
            .all(|g| root(&mut sheet, g.index()) == first)
            .then_some(first)
    };
    let (a, c) = (together(&outer)?, together(&inner)?);
    if a == c {
        outer.extend(inner);
        let sh = out.add_shell(outer);
        out.add_solid(vec![sh]);
    } else if strips == 0 {
        let shells = vec![out.add_shell(outer), out.add_shell(inner)];
        out.add_solid(shells);
    } else {
        return None;
    }
    out.pair_radial();
    if !check::check(&out).is_empty() {
        return None;
    }
    #[cfg(test)]
    local::count_direct();
    Some(out)
}

/// The stretch of `big`'s coedge `c` between `small`'s corner `corner` on it and the end of it a
/// moved face is at (its end if `at_end`), as a coedge that runs the way `c` does. `None` where
/// the corner does not lie on the edge short of that end.
fn piece(
    out: &mut Brep,
    (bs, ss): (&mut Side, &mut Side),
    pieces: &mut HashMap<(EdgeId, bool), EdgeId>,
    c: CoedgeId,
    at_end: bool,
    corner: VertexId,
) -> Option<CoedgeSpec> {
    let big = bs.b;
    let co = big.coedge(c);
    let x = big.edge(co.edge);
    let curve = x.curve.as_ref()?;
    let (t0, t1) = x.range;
    let far = at_end != co.reversed;
    let q = ss.b.vertex(corner).point;
    let t = curve.project(q, t0, t1);
    let end = big.vertex(if far { x.end } else { x.start }).point;
    let reach = 10.0 * crate::tol::LINEAR;
    if curve.eval(t).dist(q) > crate::tol::LINEAR
        || t <= t0
        || t >= t1
        || q.dist(end) <= reach
        || q.dist(big.vertex(if far { x.start } else { x.end }).point) <= reach
    {
        return None;
    }
    let range = if far { (t, t1) } else { (t0, t) };
    let edge = match pieces.get(&(co.edge, far)) {
        Some(&e) => e,
        None => {
            let v = ss.vertex(out, corner);
            let e = if far {
                let w = bs.vertex(out, x.end);
                out.add_edge(curve.clone(), range, v, w)
            } else {
                let w = bs.vertex(out, x.start);
                out.add_edge(curve.clone(), range, w, v)
            };
            pieces.insert((co.edge, far), e);
            e
        }
    };
    let pcurve = match &co.pcurve {
        // The stretch at the coedge's end starts inside it, where its parameters have run on to.
        PCurve::Projected { .. } if at_end => {
            let (a, start) = if co.reversed {
                (t1, range.1)
            } else {
                (t0, range.0)
            };
            let ts: Vec<f64> = (0..=16)
                .map(|k| a + (start - a) * k as f64 / 16.0)
                .collect();
            let uvs = big.coedge_uvs(c, &ts);
            PCurve::Projected {
                seed: uvs[uvs.len() - 1],
            }
        }
        p => p.clone(),
    };
    Some(CoedgeSpec {
        edge,
        reversed: co.reversed,
        pcurve,
    })
}

#[cfg(test)]
mod tests;
