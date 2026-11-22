//! A move of faces followed from where they are to where they go, its topology changing where a
//! face closes on the way: walls drafted until they meet close over the top between them, a step
//! raised level with the top closes the riser, and a round offset past its radius closes into the
//! corner it rounded.
//!
//! The faces move along paths from their old surfaces to their new ones. The furthest the solid can
//! go with its topology kept is found by bisection; there each edge that has shrunk to nothing is
//! collapsed to a vertex, and each face left a strip between two edges that now run together is
//! taken out, the two edges one; a corner carried onto an edge splits it first, and a corner of
//! four faces that the move cannot carry on pulls apart into two joined by a new edge. The move
//! then goes on from there. An event that needs more ends it with the error the move gave.

use std::collections::{HashMap, HashSet};

mod splits;

use super::{move_faces_pinned, near_uv, pole_v};
use crate::{
    Error, Result,
    boolean::{csi::signed_distance, unify},
    geom::{Cone, Surface},
    math::{Frame, Point3, Vec2},
    tol,
    topo::{Brep, CoedgeSpec, EdgeId, FaceId, PCurve, VertexId, radial},
};

/// A face's surface a share `s` of the way along its move, `s` from 0 to 1; an error where it has
/// no surface there, as a cylinder offset past its radius.
pub(crate) type Path<'a> = Box<dyn Fn(f64) -> Result<Surface> + 'a>;

/// The plane a share `s` of the way from `from` to `to` through the pencil of planes they span, so
/// a plane tilted about a hinge keeps the hinge and one moved along its normal stays parallel.
pub(crate) fn pencil(from: &Frame, to: &Frame, s: f64) -> Result<Frame> {
    let (c0, c1) = (from.z.dot(from.origin), to.z.dot(to.origin));
    let n = from.z * (1.0 - s) + to.z * s;
    let k = n.norm();
    let (n, c) = (n / k, (c0 * (1.0 - s) + c1 * s) / k);
    let origin = from.origin + n * (c - n.dot(from.origin));
    Frame::new(origin, n, from.x)
}

/// The cylinder `from` tilted into the cone `to` about the circle where they meet, a share `s` of
/// the way: the cone through that circle at `s` of the half angle.
pub(crate) fn opening(to: &Cone, s: f64) -> Surface {
    Surface::Cone(Cone {
        half_angle: to.half_angle * s,
        ..to.clone()
    })
}

/// How short an edge is, in mm, before the move takes it to have closed.
const CLOSED: f64 = 10.0 * tol::LINEAR;

/// The most guesses a search for a closing takes before it only halves its bracket.
const GUESSES: usize = 40;

/// Every edge's length, a pole's infinite.
fn lengths(b: &Brep) -> Vec<f64> {
    b.edge_ids()
        .map(|e| {
            let x = b.edge(e);
            x.curve
                .as_ref()
                .map_or(f64::INFINITY, |c| radial::length(c, x.range))
        })
        .collect()
}

/// `b` with the faces of `paths` moved to the ends of their paths, its topology changed where a
/// face closes on the way, and the faces of `b` that closed.
pub(crate) fn through(b: &Brep, paths: &HashMap<FaceId, Path>) -> Result<(Brep, Vec<FaceId>)> {
    let mut cur = b.clone();
    // The face of `b` each face of `cur` comes from.
    let mut from: HashMap<FaceId, FaceId> = b.face_ids().map(|f| (f, f)).collect();
    let mut s0 = 0.0;
    let mut changed = false;
    // `on` with its faces a share `s` of the way along their paths.
    let moved = |on: &Brep, from: &HashMap<FaceId, FaceId>, changed: bool, s: f64| {
        let mut moved = HashMap::new();
        for f in on.face_ids() {
            if let Some(p) = paths.get(&from[&f]) {
                moved.insert(f, p(s)?);
            } else if changed {
                // After a collapse every vertex settles again where its faces meet, those the
                // closed faces carried as well.
                moved.insert(f, on.face(f).surface.clone());
            }
        }
        move_faces_pinned(on, &moved, &HashMap::new())
    };
    loop {
        let at = |s: f64| moved(&cur, &from, changed, s);
        let last = match at(1.0) {
            Ok(out) if changed => {
                let kept: HashSet<FaceId> = from.values().copied().collect();
                let closed = b.face_ids().filter(|f| !kept.contains(f)).collect();
                return Ok((unify(&out)?, closed));
            }
            Ok(out) => return Ok((out, Vec::new())),
            Err(e) => e,
        };
        let (mut lo, mut hi) = (s0, 1.0);
        // The last two shares the move reached, with every edge's length there: edges shrinking
        // between them are run on to the shortest length the checker passes, and the first to
        // get there is a guess at where the move stops. A guess, and the probe that follows it
        // just to its other side, only choose where the bracket is cut: it still closes on a
        // share the move reaches beside one it does not, as halving it does.
        let mut reached: Vec<(f64, Vec<f64>)> = Vec::new();
        if let Ok(start) = at(s0) {
            reached.push((s0, lengths(&start)));
        }
        let (mut guesses, mut guessed, mut probe) = (0, None::<(f64, f64)>, None::<f64>);
        while hi - lo > 1e-15 * (1.0 + hi) {
            let mut mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            let inside = |s: &f64| *s > lo && *s < hi;
            let mut guess = None;
            if let Some(s) = probe.take().filter(inside) {
                mid = s;
            } else if guesses < GUESSES
                && let [.., (sa, la), (sb, lb)] = &reached[..]
                && *sb == lo
            {
                let first = la
                    .iter()
                    .zip(lb)
                    .filter(|(a, b)| b < a && a.is_finite())
                    .map(|(a, b)| sb + (b - tol::LINEAR) * (sb - sa) / (a - b))
                    .fold(f64::INFINITY, f64::min);
                if inside(&first) {
                    mid = first;
                    guess = Some(first);
                    guesses += 1;
                }
            }
            let ok = match at(mid) {
                Ok(out) => {
                    lo = mid;
                    reached.push((mid, lengths(&out)));
                    if reached.len() > 2 {
                        reached.remove(0);
                    }
                    true
                }
                Err(_) => {
                    hi = mid;
                    false
                }
            };
            // A guess moved since the last one by about what the last was off by, and is itself
            // off by less in the ratio the moves shrink in: the probe goes a few times that to
            // its other side, and no nearer than the bracket closes to.
            if let Some(s) = guess {
                let least = 0.5e-15 * (1.0 + hi);
                let step = match guessed {
                    Some((q, before)) => {
                        let moved = (s - q).abs();
                        guessed = Some((s, moved));
                        4.0 * moved * (moved / before).min(1.0)
                    }
                    None => {
                        guessed = Some((s, f64::INFINITY));
                        f64::INFINITY
                    }
                };
                let step = step.max(least);
                probe = Some(if ok { s + step } else { s - step });
            }
        }
        let closing = at(lo).ok().and_then(|near| {
            // A corner carried onto an edge it did not touch splits the edge there.
            if let Some(split) = splits::split_at_vertices(&near, CLOSED) {
                return collapse(&split, CLOSED, &HashSet::new());
            }
            let shrunk = if collapse(&near, CLOSED, &HashSet::new()).is_none() {
                at(s0)
                    .map(|start| shrinking(&start, &near))
                    .unwrap_or_default()
            } else {
                HashSet::new()
            };
            collapse(&near, CLOSED, &shrunk)
        });
        let Some((next, kept)) = closing else {
            // Nothing closed, and the move cannot go on from where it is: a corner of four faces
            // that the move pulls apart into two.
            let Some((split, s)) = pulled_apart(&cur, |on, s| moved(on, &from, changed, s), s0)
            else {
                return Err(last);
            };
            cur = split;
            s0 = s;
            changed = true;
            continue;
        };
        // Every face closed: the solid has shrunk to nothing.
        if next.face_count() == 0 {
            let path_of = |f: FaceId| from.get(&f).and_then(|g| paths.get(g));
            if at(lo).is_ok_and(|near| passing(&near, path_of)) {
                return Err(Error::InvalidInput(
                    "faces moved through each other: nothing of the solid is left".into(),
                ));
            }
            return Ok((Brep::new(), b.face_ids().collect()));
        }
        // Faces of `near` are `cur`'s in order; each kept one maps on to its face of `next`.
        let next_from = cur
            .face_ids()
            .zip(&kept)
            .filter_map(|(f, nf)| nf.map(|nf| (nf, from[&f])))
            .collect();
        cur = next;
        from = next_from;
        s0 = lo;
        changed = true;
    }
}

/// Whether `near`, a solid closed to nothing, has a face with more than [`CLOSED`] of its move
/// left that faces another face head on, their outward normals opposite: the two would pass
/// through each other. A face that closes onto its own axis or apex, as a cone's side shrinking
/// to its apex, passes no other face. `path_of` gives the path of each face that moves.
fn passing<'a>(near: &Brep, path_of: impl Fn(FaceId) -> Option<&'a Path<'a>>) -> bool {
    let outward = |f: FaceId, p: Point3| {
        let face = near.face(f);
        let uv = face.surface.project(p);
        let n = face.surface.normal(uv.x, uv.y);
        if face.reversed { -n } else { n }
    };
    let point = |f: FaceId| {
        let dom = crate::boolean::face_domain(near, f);
        match dom.interior_point() {
            Some(uv) => Some(dom.surface.eval(uv.x, uv.y)),
            None => near.face(f).loops.first().and_then(|&lp| {
                let c = *near.lp(lp).coedges.first()?;
                Some(near.vertex(near.edge(near.coedge(c).edge).start).point)
            }),
        }
    };
    // A ball or a tube shrunk to nothing has closed on its own centre: every point of it is a
    // pole within the tolerance, and tracing its boundary for a point inside finds none.
    let closed_on_itself = |f: FaceId| match &near.face(f).surface {
        Surface::Sphere(s) => s.radius <= CLOSED,
        Surface::Torus(t) => t.minor <= CLOSED,
        _ => false,
    };
    near.face_ids().any(|f| {
        let Some(path) = path_of(f).filter(|_| !closed_on_itself(f)) else {
            return false;
        };
        let Some(p) = point(f) else {
            return false;
        };
        let left = path(1.0).map_or(0.0, |end| signed_distance(&end, p).abs());
        left > CLOSED
            && near
                .face_ids()
                .any(|g| g != f && outward(f, p).dot(outward(g, p)) < -1.0 + tol::TANGENT)
    })
}

/// The most corners of four faces [`pulled_apart`] tries every way to pull apart together.
const MOST_CORNERS: usize = 4;

/// `cur` with corners of four faces pulled apart, each into two vertices joined by a new edge, and
/// the first share of the move past `s0` at which `at` moves it: tried at steps growing four times
/// from a billionth of what is left of the move, so each new edge is longer than the tolerance.
/// Fewer corners pulled apart are tried first. `None` where no way moves.
fn pulled_apart(
    cur: &Brep,
    at: impl Fn(&Brep, f64) -> Result<Brep>,
    s0: f64,
) -> Option<(Brep, f64)> {
    let corners: Vec<VertexId> = cur
        .vertex_ids()
        .filter(|&v| !splits::corner_splits(cur, v).is_empty())
        .collect();
    if corners.is_empty() || corners.len() > MOST_CORNERS {
        return None;
    }
    // Each corner kept (0) or pulled apart one of its four ways (1 to 4).
    let mut ways: Vec<Vec<usize>> = (0..5usize.pow(corners.len() as u32))
        .map(|mut k| {
            (0..corners.len())
                .map(|_| {
                    let w = k % 5;
                    k /= 5;
                    w
                })
                .collect()
        })
        .filter(|w: &Vec<usize>| w.iter().any(|&x| x > 0))
        .collect();
    ways.sort_by_key(|w| w.iter().filter(|&&x| x > 0).count());
    for way in ways {
        let mut split = Some(cur.clone());
        for (&v, &w) in corners.iter().zip(&way) {
            if w > 0 {
                split = split.and_then(|b| splits::corner_splits(&b, v).into_iter().nth(w - 1));
            }
        }
        let Some(split) = split else { continue };
        let mut step = 1e-9 * (1.0 - s0);
        while step < 1e-2 * (1.0 - s0) {
            if at(&split, s0 + step).is_ok() {
                return Some((split, s0 + step));
            }
            step *= 4.0;
        }
    }
    None
}

/// The closed edges of `near` that have shrunk to under a thousandth of their length in `start`,
/// the same solid earlier on the move: a ball's cap shrinking through a plane reads as touching it
/// while its rim is still far longer than [`CLOSED`], and the move cannot be followed closer.
fn shrinking(start: &Brep, near: &Brep) -> HashSet<EdgeId> {
    near.edge_ids()
        .filter(|&e| {
            let (x, y) = (near.edge(e), start.edge(e));
            let (Some(c), Some(d)) = (&x.curve, &y.curve) else {
                return false;
            };
            x.start == x.end && radial::length(c, x.range) <= 1e-3 * radial::length(d, y.range)
        })
        .collect()
}

fn root(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

/// `b` with what has closed taken out: vertices within `eps` of each other made one, edges then
/// running together made one, and edges shorter than `eps` gone, or a pole where a closed one
/// shrank round its face's; faces left a strip along one edge, poles alone, or lying on each other
/// face to face, taken out; and a face left running along an edge and back that nothing else uses
/// rid of it. The edges of `shrunk` are closed whatever their length. With, for each face of `b`,
/// its face in the result. `None` where nothing has closed.
fn collapse(b: &Brep, eps: f64, shrunk: &HashSet<EdgeId>) -> Option<(Brep, Vec<Option<FaceId>>)> {
    let eids: Vec<EdgeId> = b.edge_ids().collect();
    let vids: Vec<VertexId> = b.vertex_ids().collect();
    let nv = vids.len();
    let mut parent: Vec<usize> = (0..nv).collect();
    let mut changed = false;
    let mut order: Vec<VertexId> = b.vertex_ids().collect();
    order.sort_by(|x, y| b.vertex(*x).point.x.total_cmp(&b.vertex(*y).point.x));
    for (k, &v) in order.iter().enumerate() {
        let p = b.vertex(v).point;
        for &w in &order[k + 1..] {
            let q = b.vertex(w).point;
            if q.x - p.x > eps {
                break;
            }
            if p.dist(q) <= eps {
                let (a, c) = (root(&mut parent, v.index()), root(&mut parent, w.index()));
                if a != c {
                    parent[a.max(c)] = a.min(c);
                    changed = true;
                }
            }
        }
    }
    let at = |p: &mut Vec<usize>, v: VertexId| root(p, v.index());
    // Each edge's replacement and whether it runs the other way; `None` for one closed to a point.
    let ne = b.edge_ids().count();
    let mut into: Vec<Option<(usize, bool)>> = (0..ne).map(|i| Some((i, false))).collect();
    let mut mids = Vec::with_capacity(ne);
    for e in b.edge_ids() {
        let x = b.edge(e);
        mids.push(
            x.curve
                .as_ref()
                .map(|c| c.d1(0.5 * (x.range.0 + x.range.1))),
        );
        if let Some(c) = &x.curve
            && (radial::length(c, x.range) <= eps || shrunk.contains(&e))
        {
            into[e.index()] = None;
            changed = true;
        }
    }
    let mut by_ends: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for e in b.edge_ids() {
        let x = b.edge(e);
        if into[e.index()].is_none() || x.curve.is_none() {
            continue;
        }
        let (s, t) = (at(&mut parent, x.start), at(&mut parent, x.end));
        by_ends
            .entry((s.min(t), s.max(t)))
            .or_default()
            .push(e.index());
    }
    for group in by_ends.values().filter(|g| g.len() > 1) {
        for (k, &i) in group.iter().enumerate() {
            if into[i] != Some((i, false)) {
                continue;
            }
            let (m, d) = mids[i].expect("curved");
            for &j in &group[k + 1..] {
                if into[j] != Some((j, false)) {
                    continue;
                }
                let y = b.edge(eids[j]);
                let c = y.curve.as_ref().expect("curved");
                let t = c.project(m, y.range.0, y.range.1);
                let (q, dq) = c.d1(t);
                if q.dist(m) <= eps {
                    into[j] = Some((i, d.dot(dq) < 0.0));
                    changed = true;
                }
            }
        }
    }
    if !changed {
        return None;
    }
    // A closed edge shrunk round a pole of a face's surface (a cone's apex) becomes that pole there,
    // a degenerate edge numbered from `ne` on.
    let mut poles: Vec<(VertexId, PCurve)> = Vec::new();
    let mut pole_of = |c: crate::topo::CoedgeId| -> Option<usize> {
        let x = b.edge(b.coedge(c).edge);
        if x.start != x.end {
            return None;
        }
        let surface = &b.face(b.coedge_face(c)).surface;
        let period = surface.u_period()?;
        let uvs = b.coedge_uvs(c, &crate::check::coedge_params(b, c, 32));
        let (a, z) = (uvs[0], uvs[uvs.len() - 1]);
        let v = pole_v(surface, a.y)?;
        let reach = match &x.curve {
            Some(cv) if shrunk.contains(&b.coedge(c).edge) => radial::length(cv, x.range),
            _ => eps,
        };
        if ((z.x - a.x).abs() - period).abs() > 1e-6 * period
            || surface.eval(a.x, v).dist(b.vertex(x.start).point) > reach
        {
            return None;
        }
        poles.push((
            x.start,
            PCurve::Affine {
                origin: Vec2::new(a.x, v),
                dir: Vec2::new(z.x - a.x, 0.0),
            },
        ));
        Some(ne + poles.len() - 1)
    };
    // Each face's loops as (edge, reversed) after the merges, closed edges left out, and whether
    // the face lost an edge so.
    let mut lost = vec![false; b.face_ids().count()];
    let mut loops: Vec<Vec<Vec<(usize, bool, crate::topo::CoedgeId)>>> = b
        .face_ids()
        .map(|f| {
            b.face(f)
                .loops
                .iter()
                .map(|&lp| {
                    b.lp(lp)
                        .coedges
                        .iter()
                        .filter_map(|&c| {
                            let co = b.coedge(c);
                            match into[co.edge.index()] {
                                Some((e, flip)) => Some((e, co.reversed != flip, c)),
                                None => {
                                    let pole = pole_of(c);
                                    lost[f.index()] |= pole.is_none();
                                    pole.map(|e| (e, false, c))
                                }
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .filter(|l: &Vec<_>| !l.is_empty())
                .collect()
        })
        .collect();
    let nf = loops.len();
    let mut gone = vec![false; nf];
    let edge_set = |l: &Vec<Vec<(usize, bool, crate::topo::CoedgeId)>>| {
        let mut s: Vec<usize> = l.iter().flatten().map(|x| x.0).collect();
        s.sort_unstable();
        s
    };
    let curved = |e: usize| e < ne && b.edge(eids[e]).curve.is_some();
    for f in 0..nf {
        let es = edge_set(&loops[f]);
        // No boundary, or a strip along one edge (in one loop, or an annulus in two), or poles
        // alone left where an edge closed: a ball's cap shrunk onto its pole.
        if es.is_empty()
            || (es.len() == 2 && es[0] == es[1])
            || (lost[f] && !es.iter().any(|&e| curved(e)))
        {
            gone[f] = true;
        }
    }
    // Faces face to face: the same edges, one lying on the other.
    let faces: Vec<FaceId> = b.face_ids().collect();
    for f in 0..nf {
        for g in f + 1..nf {
            if gone[f] || gone[g] || edge_set(&loops[f]) != edge_set(&loops[g]) {
                continue;
            }
            let dom = crate::boolean::face_domain(b, faces[f]);
            let Some(uv) = dom.interior_point() else {
                continue;
            };
            let p = dom.surface.eval(uv.x, uv.y);
            let on = &b.face(faces[g]).surface;
            if crate::boolean::csi::signed_distance(on, p).abs() <= eps {
                gone[f] = true;
                gone[g] = true;
            }
        }
    }
    // A face running along an edge and back: a slit, which is no edge of the face. The edge goes
    // where no other face uses it, and stays a ridge where two others do (a top closed between two
    // walls drafted until they meet).
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for f in (0..nf).filter(|&f| !gone[f]) {
        for &(e, _, _) in loops[f].iter().flatten() {
            *uses.entry(e).or_default() += 1;
        }
    }
    let alive: Vec<usize> = (0..nf).filter(|&f| !gone[f]).collect();
    for f in alive {
        for l in &mut loops[f] {
            loop {
                let n = l.len();
                let found = (0..n).find(|&k| {
                    let (a, c) = (l[k], l[(k + 1) % n]);
                    n >= 2 && a.0 == c.0 && a.1 != c.1 && matches!(uses[&a.0], 2 | 4)
                });
                let Some(k) = found else { break };
                *uses.get_mut(&l[k].0).expect("counted") -= 2;
                let j = (k + 1) % n;
                let (hi, lo) = (k.max(j), k.min(j));
                l.remove(hi);
                l.remove(lo);
            }
        }
        loops[f].retain(|l| !l.is_empty());
        if loops[f].is_empty() {
            gone[f] = true;
        }
    }
    let used: std::collections::BTreeSet<usize> = (0..nf)
        .filter(|&f| !gone[f])
        .flat_map(|f| loops[f].iter().flatten().map(|x| x.0).collect::<Vec<_>>())
        .collect();
    let mut out = Brep::new();
    let mut verts: HashMap<usize, VertexId> = HashMap::new();
    let mut vertex = |out: &mut Brep, p: &mut Vec<usize>, v: VertexId| -> VertexId {
        let r = root(p, v.index());
        *verts
            .entry(r)
            .or_insert_with(|| out.add_vertex(b.vertex(vids[r]).point))
    };
    let mut edges: HashMap<usize, EdgeId> = HashMap::new();
    for &i in &used {
        if i >= ne {
            let s = vertex(&mut out, &mut parent, poles[i - ne].0);
            edges.insert(i, out.add_degenerate_edge(s, (0.0, 1.0)));
            continue;
        }
        let x = b.edge(eids[i]);
        let s = vertex(&mut out, &mut parent, x.start);
        let t = vertex(&mut out, &mut parent, x.end);
        edges.insert(
            i,
            match &x.curve {
                Some(c) => out.add_edge(c.clone(), x.range, s, t),
                None => out.add_degenerate_edge(s, x.range),
            },
        );
    }
    let mut kept = vec![None; nf];
    for f in 0..nf {
        if gone[f] {
            continue;
        }
        let face = b.face(faces[f]);
        let id = out.add_face(face.surface.clone(), face.reversed, face.names.clone());
        for l in &loops[f] {
            let mut last: Option<Vec2> = None;
            let mut specs = Vec::new();
            for &(e, reversed, c) in l {
                let edge = edges[&e];
                let x = out.edge(edge);
                let pcurve = match &x.curve {
                    None if e >= ne => poles[e - ne].1.clone(),
                    None => b.coedge(c).pcurve.clone(),
                    Some(cv) => {
                        let from = if reversed { x.end } else { x.start };
                        let seed = near_uv(&face.surface, out.vertex(from).point, last);
                        let (t0, t1) = if reversed {
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
                    reversed,
                    pcurve,
                });
            }
            out.add_loop(id, specs);
        }
        kept[f] = Some(id);
    }
    for sh in b.shell_ids() {
        let ids: Vec<FaceId> = b
            .shell(sh)
            .faces
            .iter()
            .filter_map(|f| kept[f.index()])
            .collect();
        let _ = out.add_shell(ids);
    }
    for so in b.solid_ids() {
        out.add_solid_of(b.solid(so).shells.clone(), b.solid(so).outer);
    }
    out.pair_radial();
    Some((out, kept))
}
