//! Joining the kept pieces of faces into solids: faces on one surface that share an edge merge
//! (carrying every name either had), each edge must be used twice in opposite directions, shells
//! are the connected sets of faces, and a shell enclosing negative volume is a void of the shell
//! around it. The result passes the checker or is an error.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use super::{
    classify,
    domain::{Domain, Trim, polygon_area},
    ssi::{same_domain, same_domain_faces},
};
use crate::{
    Error, Result, check,
    geom::{Curve, Surface},
    math::{Aabb, Point3, Vec2},
    measure, tol,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceMemo, FaceName, VertexId, radial},
};

/// An edge piece's geometry, by piece id.
#[derive(Clone, Debug)]
pub(crate) struct PieceGeom {
    pub curve: Option<Rc<Curve>>,
    pub range: (f64, f64),
    pub v0: usize,
    pub v1: usize,
    /// The local tolerance of the input edge it was cut from, or [`crate::tol::LINEAR`].
    pub tol: f64,
}

/// A face of the result, before it is placed in an arena.
#[derive(Clone, Debug)]
pub(crate) struct OutFace {
    pub surface: Surface,
    pub reversed: bool,
    pub names: BTreeSet<FaceName>,
    pub loops: Vec<Vec<Trim>>,
    /// The memo of an input face this one is an uncut copy of.
    pub memo: Option<Arc<FaceMemo>>,
    /// The input every piece of it was cut from, when they all come from one.
    pub side: Option<usize>,
}

impl OutFace {
    /// Turns the face inside out: the material moves to the other side.
    pub(crate) fn flip(&mut self) {
        self.memo = None;
        self.reversed = !self.reversed;
        for l in &mut self.loops {
            l.reverse();
            for t in l.iter_mut() {
                *t = t.reversed();
            }
        }
    }

    fn domain(&self) -> Domain {
        Domain::new(self.surface.clone(), self.reversed, self.loops.clone())
    }
}

fn find(p: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while p[r] != r {
        r = p[r];
    }
    let mut k = i;
    while p[k] != r {
        let nx = p[k];
        p[k] = r;
        k = nx;
    }
    r
}

/// Each piece's uses: `(face, loop, trim)`.
fn uses(faces: &[OutFace]) -> HashMap<usize, Vec<(usize, usize, usize)>> {
    let mut out: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for (li, l) in f.loops.iter().enumerate() {
            for (ti, t) in l.iter().enumerate() {
                out.entry(t.piece).or_default().push((fi, li, ti));
            }
        }
    }
    out
}

/// A face's use of a piece: the face, the loop and the trim.
pub(super) type Use = (usize, usize, usize);

/// Takes out each slit no other face uses: a loop that runs along a piece and straight back, as a
/// face does along a contact, where the result keeps nothing on the contact's other side. A loop
/// left with nothing is dropped.
fn drop_lone_slits(faces: &mut [OutFace]) {
    loop {
        let all = uses(faces);
        let mut changed = false;
        for f in faces.iter_mut() {
            for l in &mut f.loops {
                let n = l.len();
                if n < 2 {
                    continue;
                }
                let back = (0..n).find(|&i| {
                    let (a, b) = (&l[i], &l[(i + 1) % n]);
                    a.piece == b.piece
                        && a.forward != b.forward
                        && all.get(&a.piece).is_some_and(|us| us.len() == 2)
                });
                if let Some(i) = back {
                    let j = (i + 1) % n;
                    let (lo, hi) = (i.min(j), i.max(j));
                    l.remove(hi);
                    l.remove(lo);
                    // An uncut copy's memo describes the loops it came with.
                    f.memo = None;
                    changed = true;
                }
            }
            f.loops.retain(|l| !l.is_empty());
        }
        if !changed {
            break;
        }
    }
}

/// Where pieces of a result touch along an edge, more than two faces use one piece of it. Ordered
/// round the edge, each use pairs with its neighbour across the wedge of material both bound: its
/// partner. Each use's place round the edge is read a short step into its face, where faces that
/// leave the edge tangent to each other, as a cylinder lying on a plane does, have parted. Only
/// pieces with more than two uses are paired here; a piece's two uses are each other's partners.
pub(super) fn pair_uses(faces: &[OutFace], pieces: &[PieceGeom]) -> Result<HashMap<Use, Use>> {
    Ok(around_uses(faces, pieces)?.0)
}

/// [`pair_uses`], and each use's neighbour across the empty wedge beside it.
fn around_uses(
    faces: &[OutFace],
    pieces: &[PieceGeom],
) -> Result<(HashMap<Use, Use>, HashMap<Use, Use>)> {
    let mut out = HashMap::new();
    let mut beside = HashMap::new();
    let mut shared: Vec<(usize, Vec<Use>)> = uses(faces)
        .into_iter()
        .filter(|(p, us)| us.len() > 2 && pieces[*p].curve.is_some())
        .collect();
    shared.sort_by_key(|x| x.0);
    for (piece, us) in shared {
        let pg = &pieces[piece];
        let c = pg.curve.as_ref().expect("filtered");
        let (m, d) = c.d1(0.5 * (pg.range.0 + pg.range.1));
        let refused = || {
            Error::InvalidResult(vec![format!(
                "the {} faces round the edge through ({:.6}, {:.6}, {:.6}) do not pair off",
                us.len(),
                m.x,
                m.y,
                m.z
            )])
        };
        let t = d.unit().ok_or_else(refused)?;
        let len = radial::length(c, pg.range);
        let sides: Vec<radial::Side> = us
            .iter()
            .map(|&(fi, li, ti)| {
                let tr = &faces[fi].loops[li][ti];
                radial::Side {
                    surface: &faces[fi].surface,
                    reversed: faces[fi].reversed,
                    travel: if tr.t1 >= tr.t0 { t } else { -t },
                }
            })
            .collect();
        let (pairs, besides) = radial::round(m, t, len, &sides).ok_or_else(refused)?;
        for (i, j) in pairs {
            out.insert(us[i], us[j]);
            out.insert(us[j], us[i]);
        }
        for (i, j) in besides {
            beside.insert(us[i], us[j]);
            beside.insert(us[j], us[i]);
        }
    }
    Ok((out, beside))
}

fn outward_at(f: &OutFace, p: Point3) -> crate::math::Vec3 {
    let uv = f.surface.project(p);
    let n = f.surface.normal(uv.x, uv.y);
    if f.reversed { -n } else { n }
}

/// Moves `uv` by whole periods of `s` to lie nearest `to`.
fn period_shift(s: &Surface, uv: Vec2, to: Vec2) -> Vec2 {
    let f = |x: f64, t: f64, p: Option<f64>| match p {
        Some(p) => ((t - x) / p).round() * p,
        None => 0.0,
    };
    Vec2::new(f(uv.x, to.x, s.u_period()), f(uv.y, to.y, s.v_period()))
}

/// How far apart two parameter-plane points are, in mm along the surface where it has extent and
/// in parameter units where it does not, so two places on a pole stay apart.
fn uv_gap(s: &Surface, a: Vec2, b: Vec2) -> f64 {
    let (_, su, sv) = s.d1(a.x, a.y);
    (a.x - b.x).abs() * su.norm().max(1.0) + (a.y - b.y).abs() * sv.norm().max(1.0)
}

/// `uv = cu·u' + cv·v' + c`: one surface's parameters in terms of another's with the same points.
#[derive(Clone, Copy, Debug)]
pub(super) struct UvAffine {
    cu: Vec2,
    cv: Vec2,
    c: Vec2,
}

impl UvAffine {
    const IDENTITY: UvAffine = UvAffine {
        cu: Vec2::new(1.0, 0.0),
        cv: Vec2::new(0.0, 1.0),
        c: Vec2::ZERO,
    };

    pub(super) fn point(&self, p: Vec2) -> Vec2 {
        self.cu * p.x + self.cv * p.y + self.c
    }

    fn trim(&self, t: &Trim) -> Trim {
        let mut out = t.clone();
        out.uv0 = self.point(t.uv0);
        out.uv1 = self.point(t.uv1);
        out.map = t.map.mapped(&|p| self.point(p));
        out
    }
}

/// The affine map from `m`'s parameters to `base`'s, when one exists: planes in any frames, and
/// surfaces of revolution about one axis, whose angles and heights differ by sign and offset.
pub(super) fn uv_affine(base: &Surface, m: &Surface) -> Option<UvAffine> {
    if base == m {
        return Some(UvAffine::IDENTITY);
    }
    let at = |uv: Vec2| base.project_near(m.eval(uv.x, uv.y), Vec2::ZERO);
    let (o, pu, pv) = (at(Vec2::ZERO), Vec2::new(0.25, 0.0), Vec2::new(0.0, 0.25));
    let (au, av) = (at(pu), at(pv));
    let step = |a: Vec2| a + period_shift(base, a, o) - o;
    let map = UvAffine {
        cu: step(au) * 4.0,
        cv: step(av) * 4.0,
        c: o,
    };
    let checks = [
        Vec2::new(0.6, -0.3),
        Vec2::new(-1.1, 0.45),
        Vec2::new(2.0, 0.2),
    ];
    checks
        .iter()
        .all(|&q| {
            let want = m.eval(q.x, q.y);
            let got = map.point(q);
            base.eval(got.x, got.y).dist(want) <= 10.0 * crate::tol::LINEAR
        })
        .then_some(map)
}

/// The faces of `group` as one face on the first's surface, or `None` when they do not make one
/// face with closed parameter-plane loops and a single outer loop. Each member is mapped onto the
/// first's parameters and moved by whole periods to meet its neighbours; an edge two members share
/// at one place in the parameter plane is inside the merged face, and one they share a period
/// apart is its seam.
fn merge_group(
    faces: &[OutFace],
    group: &[usize],
    vtol: &[f64],
    contact: &BTreeSet<usize>,
) -> Option<OutFace> {
    // An imported vertex's edges end anywhere in its sphere, so their images part by up to twice
    // its tolerance.
    let gap = |v: usize| vtol.get(v).map_or(1e-5, |&t| (2.0 * t).max(1e-5));
    let base = &faces[group[0]];
    let s = &base.surface;
    let mut members: Vec<Vec<Trim>> = Vec::new();
    for &m in group {
        let map = uv_affine(s, &faces[m].surface)?;
        members.push(
            faces[m]
                .loops
                .iter()
                .flatten()
                .map(|t| map.trim(t))
                .collect(),
        );
    }
    let mut uses: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (k, ts) in members.iter().enumerate() {
        for (i, t) in ts.iter().enumerate() {
            uses.entry(t.piece).or_default().push((k, i));
        }
    }
    let mut shift: Vec<Option<Vec2>> = vec![None; members.len()];
    shift[0] = Some(Vec2::ZERO);
    let mut queue = vec![0];
    while let Some(k) = queue.pop() {
        let sk = shift[k].expect("queued members are placed");
        for (i, t) in members[k].iter().enumerate() {
            for &(j, ti) in &uses[&t.piece] {
                if shift[j].is_some() || (j, ti) == (k, i) {
                    continue;
                }
                let here = t.mid_uv(s) + sk;
                let there = members[j][ti].mid_uv(s);
                shift[j] = Some(period_shift(s, there, here));
                queue.push(j);
            }
        }
    }
    let mut trims: Vec<Trim> = Vec::new();
    for (k, ts) in members.iter().enumerate() {
        let d = shift[k]?;
        for t in ts {
            let mut t = t.clone();
            t.uv0 = t.uv0 + d;
            t.uv1 = t.uv1 + d;
            t.map = t.map.mapped(&|p| p + d);
            trims.push(t);
        }
    }
    let mids: Vec<Vec2> = trims.iter().map(|t| t.mid_uv(s)).collect();
    let mut inside = vec![false; trims.len()];
    for i in 0..trims.len() {
        for j in i + 1..trims.len() {
            // A contact stays: the merged face runs along it both ways, as along a seam.
            if trims[i].piece == trims[j].piece
                && !contact.contains(&trims[i].piece)
                && uv_gap(s, mids[i], mids[j]) <= 1e-5
            {
                inside[i] = true;
                inside[j] = true;
            }
        }
    }
    let trims: Vec<Trim> = trims
        .into_iter()
        .zip(inside)
        .filter_map(|(t, i)| (!i).then_some(t))
        .collect();
    let mut used = vec![false; trims.len()];
    let mut loops: Vec<Vec<Trim>> = Vec::new();
    while let Some(first) = used.iter().position(|u| !u) {
        used[first] = true;
        let mut lp = vec![trims[first].clone()];
        loop {
            let last = lp.last().expect("loop has a trim");
            let (v, at) = (last.v1, last.uv1);
            if v == lp[0].v0 && uv_gap(s, at, lp[0].uv0) <= gap(v) {
                break;
            }
            let next = (0..trims.len())
                .find(|&k| !used[k] && trims[k].v0 == v && uv_gap(s, trims[k].uv0, at) <= gap(v))?;
            used[next] = true;
            lp.push(trims[next].clone());
        }
        loops.push(lp);
    }
    let sigma = if base.reversed { -1.0 } else { 1.0 };
    let areas: Vec<f64> = loops
        .iter()
        .map(|l| {
            let mut pts = Vec::new();
            for t in l {
                let pl = t.polyline(s, 16);
                pts.extend(
                    pl[..pl.len() - 1]
                        .iter()
                        .map(|p| Vec2::new(sigma * p.x, p.y)),
                );
            }
            polygon_area(&pts)
        })
        .collect();
    let outer: Vec<usize> = (0..loops.len()).filter(|&i| areas[i] > 0.0).collect();
    if outer.len() != 1 {
        return None;
    }
    let o = outer[0];
    let mut ordered = vec![loops[o].clone()];
    for (i, l) in loops.into_iter().enumerate() {
        if i != o {
            ordered.push(l);
        }
    }
    let names = group
        .iter()
        .flat_map(|&m| faces[m].names.iter().cloned())
        .collect();
    let side = base
        .side
        .filter(|&k| group.iter().all(|&m| faces[m].side == Some(k)));
    Some(OutFace {
        surface: s.clone(),
        reversed: base.reversed,
        names,
        loops: ordered,
        memo: None,
        side,
    })
}

/// Faces on one surface, facing one way, that share an edge become one face.
pub(super) fn merge_same_domain(
    faces: Vec<OutFace>,
    pieces: &[PieceGeom],
    vtol: &[f64],
    partners: &HashMap<Use, Use>,
) -> Vec<OutFace> {
    let n = faces.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let domains: Vec<std::cell::OnceCell<Domain>> = (0..n).map(|_| Default::default()).collect();
    let domain = |i: usize| domains[i].get_or_init(|| faces[i].domain());
    let same = |f: usize, g: usize| {
        same_domain(&faces[f].surface, &faces[g].surface) || same_domain_faces(domain(f), domain(g))
    };
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    // Two faces merge across a piece they are partners on: its only two uses, or a pair of its
    // uses that bound one wedge of material where every face using it lies on that one surface, as
    // coplanar faces of both inputs do. Elsewhere a piece more faces use is a contact, where pieces
    // of the result touch, and the faces either side of it stay apart.
    let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
    let mut contact: BTreeSet<usize> = BTreeSet::new();
    for (piece, us) in uses(&faces) {
        let one_surface = us.iter().all(|u| same(us[0].0, u.0));
        if us.len() > 2 && !one_surface {
            contact.insert(piece);
        }
        for (i, &x) in us.iter().enumerate() {
            for &y in &us[i + 1..] {
                let partners = if us.len() == 2 {
                    faces[x.0].loops[x.1][x.2].forward != faces[y.0].loops[y.1][y.2].forward
                } else {
                    one_surface && partners.get(&x) == Some(&y)
                };
                if x.0 != y.0 && partners {
                    pairs.push((piece, x.0, y.0));
                }
            }
        }
    }
    pairs.sort_unstable();
    for (piece, f, g) in pairs {
        if !same(f, g) {
            continue;
        }
        let pg = &pieces[piece];
        let Some(c) = &pg.curve else { continue };
        let m = c.eval(0.5 * (pg.range.0 + pg.range.1));
        if outward_at(&faces[f], m).dot(outward_at(&faces[g], m)) <= 0.0 {
            continue;
        }
        let (rf, rg) = (find(&mut parent, f), find(&mut parent, g));
        if rf == rg {
            continue;
        }
        // The merged face lies on its root's surface, so every face joining it must lie on that
        // surface too: surfaces each within the tolerance of the next can drift apart along a run.
        let (root, other) = (rf.min(rg), rf.max(rg));
        if !members[other].iter().all(|&x| same(root, x)) {
            continue;
        }
        parent[other] = root;
        let moved = std::mem::take(&mut members[other]);
        members[root].extend(moved);
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut roots: Vec<usize> = groups.keys().copied().collect();
    roots.sort_unstable();
    let mut out = Vec::new();
    for r in roots {
        let g = &groups[&r];
        if g.len() == 1 {
            out.push(faces[g[0]].clone());
            continue;
        }
        match merge_group(&faces, g, vtol, &contact) {
            Some(f) => out.push(f),
            None => out.extend(g.iter().map(|&i| faces[i].clone())),
        }
    }
    out
}

/// Every edge must be used by two coedges running opposite ways, or by pairs of them where pieces
/// touch along it, and a pole by one.
fn check_uses(faces: &[OutFace], pieces: &[PieceGeom], partners: &HashMap<Use, Use>) -> Result<()> {
    let mut issues = Vec::new();
    for (piece, us) in uses(faces) {
        let pg = &pieces[piece];
        if pg.curve.is_none() {
            if us.len() != 1 {
                issues.push(format!("a pole is used {} times", us.len()));
            }
            continue;
        }
        let dirs: Vec<bool> = us
            .iter()
            .map(|&(f, l, t)| faces[f].loops[l][t].forward)
            .collect();
        let paired = dirs.len() > 2
            && dirs.len().is_multiple_of(2)
            && us.iter().all(|u| {
                partners.get(u).is_some_and(|p| {
                    faces[u.0].loops[u.1][u.2].forward != faces[p.0].loops[p.1][p.2].forward
                })
            });
        if !paired && (dirs.len() != 2 || dirs[0] == dirs[1]) {
            let p = pg
                .curve
                .as_ref()
                .expect("checked")
                .eval(0.5 * (pg.range.0 + pg.range.1));
            issues.push(format!(
                "the edge through ({:.6}, {:.6}, {:.6}) would be used by {} faces{}",
                p.x,
                p.y,
                p.z,
                dirs.len(),
                if dirs.len() == 2 {
                    " in the same direction"
                } else {
                    ""
                }
            ));
        }
    }
    if issues.is_empty() {
        Ok(())
    } else {
        issues.sort();
        Err(Error::InvalidResult(issues))
    }
}

/// A solid to build: its sheets, each a list of faces, the outer ones first, and how many are
/// outer.
type SolidPlan = (Vec<Vec<usize>>, usize);

/// The solid one closed sheet of `faces` bounds, each face kept as it is: neighbours on one
/// surface stay apart, as a face cut along a curve another operation needs as an edge.
pub(crate) fn assemble_as_cut(
    faces: Vec<OutFace>,
    pool: &[Point3],
    vtol: &[f64],
    pieces: &[PieceGeom],
) -> Result<Brep> {
    let plan: Vec<SolidPlan> = vec![(vec![(0..faces.len()).collect()], 1)];
    let (b, _, _) = build(&faces, &plan, pool, vtol, pieces, true);
    let issues = check::check(&b);
    if issues.is_empty() {
        Ok(b)
    } else {
        Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.to_string()).collect(),
        ))
    }
}

/// An arena holding `faces`, grouped into solids of sheets, with each face use's coedge. With
/// `share`, a solid's sheets share the vertices and edges where they touch; without, each sheet
/// has its own, as a trial of each sheet alone needs.
fn build(
    faces: &[OutFace],
    solids: &[SolidPlan],
    pool: &[Point3],
    vtol: &[f64],
    pieces: &[PieceGeom],
    share: bool,
) -> (Brep, Vec<Option<usize>>, HashMap<Use, CoedgeId>) {
    let mut b = Brep::new();
    let mut sides = Vec::new();
    let mut coedge_of: HashMap<Use, CoedgeId> = HashMap::new();
    // One vertex per solid at each point: pieces of a solid that touch at a point share it, while
    // separate solids that touch there close round their own.
    let mut verts: HashMap<(usize, usize), VertexId> = HashMap::new();
    let mut edges: HashMap<(usize, usize), EdgeId> = HashMap::new();
    let mut sheet = 0usize;
    for (si, (sheets, outer)) in solids.iter().enumerate() {
        let mut shell_ids = Vec::new();
        for faces_of in sheets {
            sheet += 1;
            let key = if share { si } else { sheet };
            let mut face_ids = Vec::new();
            for &fi in faces_of {
                let f = &faces[fi];
                let id = b.add_face(f.surface.clone(), f.reversed, f.names.iter().cloned());
                for (li, l) in f.loops.iter().enumerate() {
                    let mut specs = Vec::new();
                    for t in l {
                        let e = match edges.get(&(t.piece, key)) {
                            Some(&e) => e,
                            None => {
                                let pg = &pieces[t.piece];
                                let mut vertex = |b: &mut Brep, i: usize| {
                                    *verts.entry((i, key)).or_insert_with(|| {
                                        let v = b.add_vertex(pool[i]);
                                        b.set_vertex_tolerance(v, vtol[i]);
                                        v
                                    })
                                };
                                let (v0, v1) = (vertex(&mut b, pg.v0), vertex(&mut b, pg.v1));
                                let e = match &pg.curve {
                                    Some(c) => b.add_edge((**c).clone(), pg.range, v0, v1),
                                    None => b.add_degenerate_edge(v0, pg.range),
                                };
                                b.set_edge_tolerance(e, pg.tol);
                                edges.insert((t.piece, key), e);
                                e
                            }
                        };
                        let pcurve = t.map.pcurve(t.uv0);
                        specs.push(CoedgeSpec {
                            edge: e,
                            reversed: !t.forward,
                            pcurve,
                        });
                    }
                    let lp = b.add_loop(id, specs);
                    for (ti, &c) in b.lp(lp).coedges.iter().enumerate() {
                        coedge_of.insert((fi, li, ti), c);
                    }
                }
                if let Some(m) = &f.memo {
                    b.share_memo(id, m.clone());
                }
                sides.push(f.side);
                face_ids.push(id);
            }
            shell_ids.push(b.add_shell(face_ids));
        }
        b.add_solid_of(shell_ids, *outer);
    }
    (b, sides, coedge_of)
}

/// The average thickness below which a separate sheet of a result holds no material.
const THIN: f64 = 10.0 * tol::LINEAR;

/// The result solid from its faces. `vtol` is each pool vertex's local tolerance.
pub(crate) fn assemble(
    faces: Vec<OutFace>,
    pool: &[Point3],
    vtol: &[f64],
    pieces: &[PieceGeom],
) -> Result<Brep> {
    if faces.is_empty() {
        return Ok(Brep::new());
    }
    let mut faces = faces;
    drop_lone_slits(&mut faces);
    // Pair first: faces on one surface that meet only where pieces of the result touch must not
    // merge across the contact.
    let partners = pair_uses(&faces, pieces)?;
    let faces = merge_same_domain(faces, pieces, vtol, &partners);
    let (partners, beside) = around_uses(&faces, pieces)?;
    check_uses(&faces, pieces, &partners)?;
    let n = faces.len();
    let all = uses(&faces);
    let join = |parent: &mut [usize], a: usize, c: usize| {
        let (a, c) = (find(parent, a), find(parent, c));
        if a != c {
            parent[a.max(c)] = a.min(c);
        }
    };
    // Sheets: faces joined across each piece to their partners.
    let mut parent: Vec<usize> = (0..n).collect();
    for us in all.values() {
        if us.len() == 2 {
            join(&mut parent, us[0].0, us[1].0);
        } else {
            for u in us {
                if let Some(p) = partners.get(u) {
                    join(&mut parent, u.0, p.0);
                }
            }
        }
    }
    let mut sheets: Vec<Vec<usize>> = Vec::new();
    let mut sheet_of = vec![0usize; n];
    let mut index: HashMap<usize, usize> = HashMap::new();
    for (i, sheet) in sheet_of.iter_mut().enumerate() {
        let r = find(&mut parent, i);
        let k = *index.entry(r).or_insert_with(|| {
            sheets.push(Vec::new());
            sheets.len() - 1
        });
        sheets[k].push(i);
        *sheet = k;
    }
    // Each sheet's own volume: positive where it bounds material from outside, negative a void.
    let volumes: Vec<f64> = if sheets.len() == 1 {
        vec![1.0]
    } else {
        let alone: Vec<SolidPlan> = sheets.iter().map(|s| (vec![s.clone()], 1)).collect();
        let trial = build(&faces, &alone, pool, vtol, pieces, false).0;
        let volumes: Vec<f64> = trial
            .solid_ids()
            .map(|s| measure::solid_volume_sign(&trial, s))
            .collect();
        // A sheet thinner on average than a few tolerances, as between two surfaces that cross at
        // a glancing angle, holds no material the tolerance resolves: its volume's sign is
        // rounding. Its thickness is twice its volume over its area.
        let flat: Vec<bool> = trial
            .solid_ids()
            .zip(&volumes)
            .map(|(s, v)| {
                // No sheet of a model's size this thin holds more; most are ruled out unmeasured.
                v.abs() <= THIN * tol::MODEL_SIZE * tol::MODEL_SIZE
                    && 2.0 * measure::solid_volume(&trial, s).abs()
                        <= THIN
                            * trial
                                .solid_faces(s)
                                .into_iter()
                                .map(|f| measure::face_area(&trial, f))
                                .sum::<f64>()
            })
            .collect();
        if flat.iter().any(|&x| x) && !flat.iter().all(|&x| x) {
            let kept = faces
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !flat[sheet_of[*i]])
                .map(|(_, f)| f)
                .collect();
            return assemble(kept, pool, vtol, pieces);
        }
        volumes
    };
    // Pieces of one solid: sheets that touch along a piece more than two faces use, or at a
    // vertex both hold.
    let m = sheets.len();
    let mut group: Vec<usize> = (0..m).collect();
    for us in all.values() {
        for w in us.windows(2) {
            join(&mut group, sheet_of[w[0].0], sheet_of[w[1].0]);
        }
    }
    let mut at_vertex: HashMap<usize, usize> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for t in f.loops.iter().flatten() {
            let pg = &pieces[t.piece];
            for v in [pg.v0, pg.v1] {
                match at_vertex.get(&v) {
                    Some(&other) => join(&mut group, other, sheet_of[fi]),
                    None => {
                        at_vertex.insert(v, sheet_of[fi]);
                    }
                }
            }
        }
    }
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for k in 0..m {
        members.entry(find(&mut group, k)).or_default().push(k);
    }
    let mut roots: Vec<usize> = members.keys().copied().collect();
    roots.sort_unstable();
    let mut solids: Vec<SolidPlan> = Vec::new();
    let mut stray: Vec<usize> = Vec::new();
    for r in roots {
        let ks = &members[&r];
        let (outer, inner): (Vec<usize>, Vec<usize>) = ks.iter().partition(|&&k| volumes[k] > 0.0);
        if outer.is_empty() {
            stray.extend(inner);
            continue;
        }
        let mut plan: Vec<Vec<usize>> = outer.iter().map(|&k| sheets[k].clone()).collect();
        plan.extend(inner.iter().map(|&k| sheets[k].clone()));
        solids.push((plan, outer.len()));
    }
    // A void touching nothing lies in the smallest solid whose outer sheets hold it.
    let doms: Vec<std::cell::OnceCell<Domain>> = (0..n).map(|_| Default::default()).collect();
    let dom = |i: usize| doms[i].get_or_init(|| faces[i].domain());
    let size = |plan: &SolidPlan| -> f64 {
        plan.0[..plan.1]
            .iter()
            .map(|sh| volumes[sheet_of[sh[0]]])
            .sum()
    };
    for k in stray {
        let d = dom(sheets[k][0]);
        let probe = d
            .interior_point()
            .map(|uv| d.surface.eval(uv.x, uv.y))
            .ok_or_else(|| Error::Numerical("a void's face has no interior point".into()))?;
        let host = (0..solids.len())
            .filter(|&s| {
                let shell: Vec<(&Domain, Aabb)> = solids[s].0[..solids[s].1]
                    .iter()
                    .flatten()
                    .map(|&f| (dom(f), super::face_bbox(dom(f))))
                    .collect();
                classify::contains(&shell, probe) == Some(true)
            })
            .min_by(|&a, &b| size(&solids[a]).total_cmp(&size(&solids[b])));
        match host {
            Some(s) => solids[s].0.push(sheets[k].clone()),
            None => {
                return Err(Error::InvalidResult(vec![
                    "a shell encloses negative volume and lies in no solid".into(),
                ]));
            }
        }
    }
    let (mut b, sides, coedge_of) = build(&faces, &solids, pool, vtol, pieces, true);
    for (u, p) in &partners {
        if let (Some(&c), Some(&d)) = (coedge_of.get(u), coedge_of.get(p)) {
            b.pair(c, d);
        }
    }
    for (u, p) in &beside {
        if let (Some(&c), Some(&d)) = (coedge_of.get(u), coedge_of.get(p)) {
            b.set_beside(c, d);
        }
    }
    let issues = check::check_cut(&b, &sides);
    if issues.is_empty() {
        Ok(b)
    } else {
        Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.to_string()).collect(),
        ))
    }
}
