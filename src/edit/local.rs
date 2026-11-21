//! What local edits share: a solid rebuilt with some faces taken out and the loops round them
//! closed again, every other face copied with its memo; and the test that the region an edit
//! adds or removes lies wholly on one side of each surface that bounds it.
//!
//! A [`Splice`] takes the faces that go and marks every coedge on their edges as cut. The edit
//! then says how each loop a cut opened closes: the edges that led up to the cut run on along
//! their own curves to new vertices, two that face each other along one line become one edge, new
//! edges fill what is left, and a loop cut away whole is dropped or replaced. Loops are traced
//! from those steps; faces a traced loop passes through become one face.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;

use super::tweak::{affine_on, near_uv};
use crate::{
    boolean::{csi::signed_distance, same_domain},
    geom::{Curve, Surface},
    math::{Point3, Vec2, Vec3},
    tol,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, LoopId, PCurve, VertexId},
};

/// An edge of a [`Splice`] as a loop uses it. Edges and vertices of the solid keep their indices;
/// new ones follow them.
#[derive(Clone, Copy)]
pub(crate) struct Use {
    pub edge: usize,
    pub reversed: bool,
}

#[derive(Clone)]
struct Rec {
    curve: Option<Curve>,
    range: (f64, f64),
    start: usize,
    end: usize,
}

enum Step {
    /// New uses, then the loop goes on at this coedge.
    Fill(Vec<Use>, CoedgeId),
    /// This coedge's edge is one with the last's: the loop goes on after it.
    Merge(CoedgeId),
}

pub(crate) struct Splice<'a> {
    b: &'a Brep,
    gone: Vec<bool>,
    cut: Vec<bool>,
    points: Vec<Point3>,
    edges: Vec<Rec>,
    alias: Vec<Option<(usize, bool)>>,
    touched: Vec<bool>,
    steps: HashMap<CoedgeId, Step>,
    absorbed: HashSet<CoedgeId>,
    loops: Vec<(FaceId, Vec<Use>, bool)>,
    joins: Vec<(FaceId, FaceId)>,
}

/// A spliced solid, and its faces whose loops changed.
pub(crate) struct Spliced {
    pub out: Brep,
    pub changed: Vec<FaceId>,
}

struct Item {
    src: Option<CoedgeId>,
    edge: usize,
    reversed: bool,
}

/// What to add to parameters on `from` to get the same point's on `to`, where the two are one
/// surface with one parameterisation up to that: the same surface, or cylinders whose frames
/// differ by a shift along the axis.
fn shifted(from: &Surface, to: &Surface) -> Option<Vec2> {
    if from == to {
        return Some(Vec2::ZERO);
    }
    let (Surface::Cylinder(a), Surface::Cylinder(c)) = (from, to) else {
        return None;
    };
    let d = a.frame.origin - c.frame.origin;
    let along = d.dot(c.frame.z);
    ((a.radius - c.radius).abs() <= tol::LINEAR
        && a.frame.z.dot(c.frame.z) >= 1.0 - 1e-12
        && a.frame.x.dot(c.frame.x) >= 1.0 - 1e-12
        && (d - c.frame.z * along).norm() <= tol::LINEAR)
        .then_some(Vec2::new(0.0, along))
}

fn find(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

impl<'a> Splice<'a> {
    pub(crate) fn new(b: &'a Brep, faces: &[FaceId]) -> Splice<'a> {
        let mut gone = vec![false; b.face_ids().count()];
        for &f in faces {
            gone[f.index()] = true;
        }
        let mut cut = vec![false; b.coedge_ids().count()];
        for e in b.edge_ids() {
            let uses = b.edge_uses(e);
            if uses.iter().any(|&c| gone[b.coedge_face(c).index()]) {
                for &c in uses {
                    cut[c.index()] = true;
                }
            }
        }
        let edges: Vec<Rec> = b
            .edge_ids()
            .map(|e| {
                let x = b.edge(e);
                Rec {
                    curve: x.curve.clone(),
                    range: x.range,
                    start: x.start.index(),
                    end: x.end.index(),
                }
            })
            .collect();
        Splice {
            b,
            gone,
            cut,
            points: b.vertex_ids().map(|v| b.vertex(v).point).collect(),
            alias: vec![None; edges.len()],
            touched: vec![false; edges.len()],
            edges,
            steps: HashMap::new(),
            absorbed: HashSet::new(),
            loops: Vec::new(),
            joins: Vec::new(),
        }
    }

    pub(crate) fn point(&self, v: usize) -> Point3 {
        self.points[v]
    }

    pub(crate) fn add_vertex(&mut self, p: Point3) -> usize {
        self.points.push(p);
        self.points.len() - 1
    }

    pub(crate) fn add_edge(
        &mut self,
        curve: Curve,
        range: (f64, f64),
        start: usize,
        end: usize,
    ) -> usize {
        self.edges.push(Rec {
            curve: Some(curve),
            range,
            start,
            end,
        });
        self.touched.push(true);
        self.edges.len() - 1
    }

    /// Runs edge `e` on from its vertex `at` along its own curve to the new vertex `to`, which
    /// must lie on the curve past `at`. Gives the stretch of the curve added.
    pub(crate) fn extend(&mut self, e: EdgeId, at: VertexId, to: usize) -> Option<(f64, f64)> {
        let i = e.index();
        if self.alias[i].is_some() {
            return None;
        }
        let x = self.point(to);
        let was = self.point(at.index());
        let rec = &mut self.edges[i];
        let curve = rec.curve.as_ref()?;
        let (t0, t1) = rec.range;
        let (at_start, at_end) = (rec.start == at.index(), rec.end == at.index());
        if at_start == at_end {
            return None;
        }
        let t = match curve {
            Curve::Line(l) => (x - l.origin).dot(l.dir),
            Curve::Circle(_) if at_end => curve.project(x, t1, t1 + PI),
            Curve::Circle(_) => curve.project(x, t0 - PI, t0),
            _ => return None,
        };
        let past = if at_end { t > t1 } else { t < t0 };
        if !past || curve.eval(t).dist(x) > tol::LINEAR || x.dist(was) <= 10.0 * tol::LINEAR {
            return None;
        }
        self.touched[i] = true;
        if at_end {
            rec.range.1 = t;
            rec.end = to;
            Some((t1, t))
        } else {
            rec.range.0 = t;
            rec.start = to;
            Some((t, t0))
        }
    }

    /// Makes one edge of the straight edges `s`, which ends at `p`, and `s2`, which ends at `q`,
    /// where they lie on one line and face each other across the gap from `p` to `q`. Gives the
    /// new edge and the gap as a stretch of `s`'s line.
    pub(crate) fn merge(
        &mut self,
        (s, p): (EdgeId, VertexId),
        (s2, q): (EdgeId, VertexId),
    ) -> Option<(usize, (f64, f64))> {
        let (i, j) = (s.index(), s2.index());
        if i == j
            || [i, j]
                .iter()
                .any(|&k| self.alias[k].is_some() || self.touched[k])
        {
            return None;
        }
        let (a, c) = (self.edges[i].clone(), self.edges[j].clone());
        let (Some(Curve::Line(l)), Some(Curve::Line(l2))) = (&a.curve, &c.curve) else {
            return None;
        };
        let off = l2.origin - l.origin;
        if l.dir.cross(l2.dir).norm() > 1e-9 || (off - l.dir * off.dot(l.dir)).norm() > tol::LINEAR
        {
            return None;
        }
        let far = match (c.start == q.index(), c.end == q.index()) {
            (true, false) => c.end,
            (false, true) => c.start,
            _ => return None,
        };
        let at = |v: usize| (self.point(v) - l.origin).dot(l.dir);
        let (tq, tw) = (at(q.index()), at(far));
        let (t0, t1) = a.range;
        let (rec, gap) = match (a.start == p.index(), a.end == p.index()) {
            (false, true) if tq > t1 + tol::LINEAR && tw > tq => (
                Rec {
                    range: (t0, tw),
                    end: far,
                    ..a.clone()
                },
                (t1, tq),
            ),
            (true, false) if tq < t0 - tol::LINEAR && tw < tq => (
                Rec {
                    range: (tw, t1),
                    start: far,
                    ..a.clone()
                },
                (tq, t0),
            ),
            _ => return None,
        };
        let flip = l.dir.dot(l2.dir) < 0.0;
        self.edges.push(rec);
        self.touched.push(true);
        let m = self.edges.len() - 1;
        self.alias[i] = Some((m, false));
        self.alias[j] = Some((m, flip));
        Some((m, gap))
    }

    /// After coedge `before`, where its loop is cut, come `uses` and then coedge `after`.
    pub(crate) fn fill(&mut self, before: CoedgeId, uses: Vec<Use>, after: CoedgeId) {
        self.steps.insert(before, Step::Fill(uses, after));
    }

    /// Coedge `before`'s edge has taken in coedge `after`'s: the loop goes on past `after`.
    pub(crate) fn merged(&mut self, before: CoedgeId, after: CoedgeId) {
        self.steps.insert(before, Step::Merge(after));
        self.absorbed.insert(after);
    }

    /// A new loop of `face`, its outer one if `first`.
    pub(crate) fn new_loop(&mut self, face: FaceId, uses: Vec<Use>, first: bool) {
        self.loops.push((face, uses, first));
    }

    /// Faces `f` and `g` become one face.
    pub(crate) fn join(&mut self, f: FaceId, g: FaceId) {
        self.joins.push((f, g));
    }

    fn resolve(&self, e: usize, reversed: bool) -> (usize, bool) {
        match self.alias.get(e).copied().flatten() {
            Some((m, flip)) => (m, reversed != flip),
            None => (e, reversed),
        }
    }

    fn ends(&self, it: &Item) -> (usize, usize) {
        let r = &self.edges[it.edge];
        if it.reversed {
            (r.end, r.start)
        } else {
            (r.start, r.end)
        }
    }

    /// The loop through coedge `c0` as the steps close it, the faces it passes through, and
    /// whether any of its coedges came from a face's outer loop.
    fn trace(
        &self,
        c0: CoedgeId,
        visited: &mut HashSet<CoedgeId>,
    ) -> Option<(Vec<Item>, Vec<FaceId>, bool)> {
        let b = self.b;
        let (mut items, mut members, mut first) = (Vec::new(), Vec::new(), false);
        let next = |c: CoedgeId| {
            let cs = &b.lp(b.coedge(c).lp).coedges;
            let k = cs
                .iter()
                .position(|&x| x == c)
                .expect("a coedge is in its loop");
            cs[(k + 1) % cs.len()]
        };
        let note = |c: CoedgeId, members: &mut Vec<FaceId>, first: &mut bool| {
            let f = b.coedge_face(c);
            members.push(f);
            *first |= b.face(f).loops[0] == b.coedge(c).lp;
        };
        let mut cur = c0;
        loop {
            if !visited.insert(cur) || self.cut[cur.index()] {
                return None;
            }
            note(cur, &mut members, &mut first);
            let co = b.coedge(cur);
            let (edge, reversed) = self.resolve(co.edge.index(), co.reversed);
            items.push(Item {
                src: Some(cur),
                edge,
                reversed,
            });
            let nxt = next(cur);
            cur = if self.cut[nxt.index()] {
                match self.steps.get(&cur)? {
                    Step::Fill(uses, after) => {
                        items.extend(uses.iter().map(|u| Item {
                            src: None,
                            edge: u.edge,
                            reversed: u.reversed,
                        }));
                        *after
                    }
                    Step::Merge(after) => {
                        if !visited.insert(*after) && *after != c0 {
                            return None;
                        }
                        note(*after, &mut members, &mut first);
                        next(*after)
                    }
                }
            } else {
                nxt
            };
            if cur == c0 {
                break;
            }
        }
        Some((items, members, first))
    }

    /// The pcurve of a use of `rec` on `surface`: the old coedge's where it still holds, a line's
    /// in a plane in closed form, and otherwise the projection continued from `last`, which is
    /// left at the use's end.
    fn pcurve(
        &self,
        surface: &Surface,
        it: &Item,
        shift: Option<Vec2>,
        last: &mut Option<Vec2>,
    ) -> Option<PCurve> {
        let b = self.b;
        let rec = &self.edges[it.edge];
        let (ta, tz) = if it.reversed {
            (rec.range.1, rec.range.0)
        } else {
            rec.range
        };
        let src = it.src.filter(|_| shift.is_some()).map(|c| b.coedge(c));
        let moved = |p: PCurve| match shift {
            Some(by) if by != Vec2::ZERO => p.mapped(&|uv| uv + by),
            _ => p,
        };
        let Some(curve) = &rec.curve else {
            let pcurve = moved(src?.pcurve.clone());
            if let PCurve::Affine { origin, dir } = &pcurve {
                *last = Some(*origin + *dir * tz);
            }
            return Some(pcurve);
        };
        let mut found = None;
        if let Some(co) = src {
            let old = b.edge(co.edge);
            let start = if co.reversed {
                old.range.1
            } else {
                old.range.0
            };
            match (&old.curve, &co.pcurve) {
                (Some(c), PCurve::Affine { .. }) if c == curve => found = Some(co.pcurve.clone()),
                (Some(c), PCurve::Projected { .. })
                    if c == curve && start == ta && co.reversed == it.reversed =>
                {
                    found = Some(co.pcurve.clone());
                }
                (Some(c), PCurve::Spline { .. }) if c == curve && old.range == rec.range => {
                    found = Some(co.pcurve.clone());
                }
                (Some(Curve::Line(lo)), PCurve::Affine { origin, dir }) => {
                    if let Curve::Line(ln) = curve {
                        let shift = (ln.origin - lo.origin).dot(lo.dir);
                        found = Some(PCurve::Affine {
                            origin: *origin + *dir * shift,
                            dir: *dir * lo.dir.dot(ln.dir),
                        });
                    }
                }
                _ => {}
            }
        }
        let mut found = found.map(moved);
        if found.is_none()
            && let (Surface::Plane(p), Curve::Line(l)) = (surface, curve)
        {
            found = Some(affine_on(&p.frame, l));
        }
        let pcurve = match found {
            Some(p) => p,
            None => {
                let hint = near_uv(surface, curve.eval(ta), *last);
                let seed = crate::topo::pcurve::project_on(surface, curve, ta, tz, hint);
                PCurve::Projected { seed }
            }
        };
        *last = match &pcurve {
            PCurve::Affine { origin, dir } => Some(*origin + *dir * tz),
            PCurve::Projected { seed } => {
                let mut uv = *seed;
                for k in 1..=32 {
                    let p = curve.eval(ta + (tz - ta) * k as f64 / 32.0);
                    uv = near_uv(surface, p, Some(uv));
                }
                Some(uv)
            }
            PCurve::Spline { .. } => None,
        };
        Some(pcurve)
    }

    /// The solid with the cuts closed as the steps say; `None` where they do not close every
    /// loop, or faces that would become one do not lie on one surface the same way round.
    pub(crate) fn finish(self) -> Option<Spliced> {
        let b = self.b;
        let nf = self.gone.len();
        let mut kept: Vec<Vec<LoopId>> = vec![Vec::new(); nf];
        let mut changed = vec![false; nf];
        let mut live: Vec<CoedgeId> = Vec::new();
        for f in b.face_ids().filter(|f| !self.gone[f.index()]) {
            for &lp in &b.face(f).loops {
                let cs = &b.lp(lp).coedges;
                let ncut = cs.iter().filter(|c| self.cut[c.index()]).count();
                let moved = cs.iter().any(|&c| {
                    let e = b.coedge(c).edge.index();
                    self.touched[e] || self.alias[e].is_some()
                });
                if ncut == 0 && !moved {
                    kept[f.index()].push(lp);
                    continue;
                }
                changed[f.index()] = true;
                live.extend(cs.iter().copied().filter(|c| !self.cut[c.index()]));
            }
        }
        let mut group: Vec<usize> = (0..nf).collect();
        let join = |group: &mut Vec<usize>, f: FaceId, g: FaceId| {
            let (x, y) = (find(group, f.index()), find(group, g.index()));
            group[x.max(y)] = x.min(y);
        };
        let mut visited = HashSet::new();
        let mut traced: Vec<(FaceId, Vec<Item>, bool)> = Vec::new();
        for &c0 in &live {
            if visited.contains(&c0) || self.absorbed.contains(&c0) {
                continue;
            }
            let (items, members, first) = self.trace(c0, &mut visited)?;
            for i in 0..items.len() {
                if self.ends(&items[i]).1 != self.ends(&items[(i + 1) % items.len()]).0 {
                    return None;
                }
            }
            for &m in &members[1..] {
                join(&mut group, members[0], m);
            }
            traced.push((members[0], items, first));
        }
        if live.iter().any(|c| !visited.contains(c)) {
            return None;
        }
        for (f, uses, first) in &self.loops {
            changed[f.index()] = true;
            let items = uses
                .iter()
                .map(|u| Item {
                    src: None,
                    edge: u.edge,
                    reversed: u.reversed,
                })
                .collect();
            traced.push((*f, items, *first));
        }
        for &(f, g) in &self.joins {
            if self.gone[f.index()] || self.gone[g.index()] {
                return None;
            }
            join(&mut group, f, g);
        }
        let rep: Vec<usize> = (0..nf).map(|f| find(&mut group, f)).collect();
        let ids: Vec<FaceId> = b.face_ids().collect();
        for f in 0..nf {
            let (x, y) = (b.face(ids[f]), b.face(ids[rep[f]]));
            if rep[f] == f {
                continue;
            }
            changed[rep[f]] = true;
            let planes = matches!(
                (&x.surface, &y.surface),
                (Surface::Plane(_), Surface::Plane(_))
            );
            if x.reversed != y.reversed
                || !(shifted(&x.surface, &y.surface).is_some()
                    || planes && same_domain(&x.surface, &y.surface))
            {
                return None;
            }
            if let (Surface::Plane(p), Surface::Plane(q)) = (&x.surface, &y.surface)
                && p.frame.z.dot(q.frame.z) <= 0.0
            {
                return None;
            }
        }

        let mut out = Brep::new();
        let mut verts: HashMap<usize, VertexId> = HashMap::new();
        let mut edges: HashMap<usize, EdgeId> = HashMap::new();
        let mut edge = |out: &mut Brep, e: usize| -> EdgeId {
            if let Some(&id) = edges.get(&e) {
                return id;
            }
            let rec = &self.edges[e];
            let mut vertex = |out: &mut Brep, v: usize| {
                *verts
                    .entry(v)
                    .or_insert_with(|| out.add_vertex(self.point(v)))
            };
            let (s, t) = (vertex(out, rec.start), vertex(out, rec.end));
            let id = match &rec.curve {
                Some(c) => out.add_edge(c.clone(), rec.range, s, t),
                None => out.add_degenerate_edge(s, rec.range),
            };
            edges.insert(e, id);
            id
        };
        let mut faces: Vec<Option<FaceId>> = vec![None; nf];
        for f in (0..nf).filter(|&f| !self.gone[f] && rep[f] == f) {
            let face = b.face(ids[f]);
            let members: Vec<usize> = (0..nf).filter(|&m| !self.gone[m] && rep[m] == f).collect();
            let names: BTreeSet<_> = members
                .iter()
                .flat_map(|&m| b.face(ids[m]).names.iter().cloned())
                .collect();
            // Each loop as its items, whether it was an outer loop, and whether it is unchanged
            // on the surface it was on.
            let mut loops: Vec<(Vec<Item>, bool, bool)> = Vec::new();
            for &m in &members {
                for &lp in &kept[m] {
                    let items = b
                        .lp(lp)
                        .coedges
                        .iter()
                        .map(|&c| Item {
                            src: Some(c),
                            edge: b.coedge(c).edge.index(),
                            reversed: b.coedge(c).reversed,
                        })
                        .collect();
                    let same = shifted(&b.face(ids[m]).surface, &face.surface).is_some();
                    loops.push((items, b.face(ids[m]).loops[0] == lp, same));
                }
            }
            let mut rest = Vec::new();
            for (g, items, first) in traced.drain(..) {
                if rep[g.index()] == f {
                    loops.push((items, first, false));
                } else {
                    rest.push((g, items, first));
                }
            }
            traced = rest;
            let outer = loops.iter().filter(|l| l.1).count();
            if loops.is_empty() || (outer != 1 && matches!(face.surface, Surface::Plane(_))) {
                return None;
            }
            loops.sort_by_key(|l| !l.1);
            let whole = !changed[f] && members.len() == 1;
            let id = out.add_face(face.surface.clone(), face.reversed, names);
            for (mut items, _, untouched) in loops {
                // A loop is laid from a coedge that keeps its pcurve, so new ones continue it.
                if !untouched
                    && let Some(at) = items.iter().position(|it| {
                        it.src.is_some_and(|c| {
                            shifted(&b.face(b.coedge_face(c)).surface, &face.surface).is_some()
                                && !self.touched[b.coedge(c).edge.index()]
                                && self.alias[b.coedge(c).edge.index()].is_none()
                        })
                    })
                {
                    items.rotate_left(at);
                }
                let mut last = None;
                let mut specs = Vec::with_capacity(items.len());
                for it in &items {
                    let shift = it
                        .src
                        .and_then(|c| shifted(&b.face(b.coedge_face(c)).surface, &face.surface));
                    let pcurve = match (it.src, shift) {
                        (Some(c), Some(by)) if untouched => {
                            let p = b.coedge(c).pcurve.clone();
                            if by == Vec2::ZERO {
                                p
                            } else {
                                p.mapped(&|uv| uv + by)
                            }
                        }
                        _ => self.pcurve(&face.surface, it, shift, &mut last)?,
                    };
                    specs.push(CoedgeSpec {
                        edge: edge(&mut out, it.edge),
                        reversed: it.reversed,
                        pcurve,
                    });
                }
                out.add_loop(id, specs);
            }
            if whole {
                out.share_memo(id, b.shared_memo(ids[f]));
            }
            faces[f] = Some(id);
        }
        if !traced.is_empty() {
            return None;
        }
        for f in 0..nf {
            if !self.gone[f] {
                faces[f] = faces[rep[f]];
            }
        }
        // Each shell's faces must still hang together edge to edge.
        let n = out.face_count();
        let mut sheet: Vec<usize> = (0..n).collect();
        for e in out.edge_ids() {
            let us = out.edge_uses(e);
            for w in us.windows(2) {
                let (x, y) = (
                    find(&mut sheet, out.coedge_face(w[0]).index()),
                    find(&mut sheet, out.coedge_face(w[1]).index()),
                );
                sheet[x.max(y)] = x.min(y);
            }
        }
        for sh in b.shell_ids() {
            let mut ids: Vec<FaceId> = Vec::new();
            for f in &b.shell(sh).faces {
                if let Some(g) = faces[f.index()]
                    && !ids.contains(&g)
                {
                    ids.push(g);
                }
            }
            let first = ids.first()?;
            let root = find(&mut sheet, first.index());
            if ids.iter().any(|g| find(&mut sheet, g.index()) != root) {
                return None;
            }
            let _ = out.add_shell(ids);
        }
        for so in b.solid_ids() {
            out.add_solid_of(b.solid(so).shells.clone(), b.solid(so).outer);
        }
        out.pair_radial();
        let changed = (0..nf)
            .filter(|&f| changed[f] && rep[f] == f)
            .filter_map(|f| faces[f])
            .collect();
        Some(Spliced { out, changed })
    }
}

/// A surface that bounds an edit's region, with the way its face's material lies.
pub(crate) struct Wall {
    pub surface: Surface,
    pub reversed: bool,
}

impl Wall {
    /// The distance from the wall's surface to `p`, positive out of its face's material.
    pub(crate) fn out(&self, p: Point3) -> f64 {
        let d = signed_distance(&self.surface, p);
        if self.reversed { -d } else { d }
    }
}

/// A stretch of curve on the boundary of an edit's region, and the walls it lies on by the way
/// it was made.
pub(crate) struct Rim {
    pub curve: Curve,
    pub range: (f64, f64),
    pub on: Vec<usize>,
}

/// Which side of every wall an edit's region lies on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    /// In each wall's material.
    Within,
    /// Out of each wall's material.
    Beyond,
}

fn off_axis(p: Point3, o: Point3, k: Vec3) -> Vec3 {
    let d = p - o;
    d - k * d.dot(k)
}

/// The axis a wall's surface is turned about, for the walls that are not planes.
fn axis(s: &Surface) -> Option<(Point3, Vec3)> {
    match s {
        Surface::Cylinder(c) => Some((c.frame.origin, c.frame.z)),
        Surface::Cone(c) => Some((c.frame.origin, c.frame.z)),
        Surface::Sphere(c) => Some((c.frame.origin, c.frame.z)),
        Surface::Torus(c) => Some((c.frame.origin, c.frame.z)),
        _ => None,
    }
}

/// Whether `s` is a surface of revolution about the axis through `o` along `k`.
fn turned_about(s: &Surface, o: Point3, k: Vec3) -> bool {
    let along = |z: Vec3| z.cross(k).norm() <= 1e-9;
    let on = |p: Point3| off_axis(p, o, k).norm() <= tol::LINEAR;
    match s {
        Surface::Plane(p) => along(p.frame.z),
        Surface::Cylinder(c) => along(c.frame.z) && on(c.frame.origin),
        Surface::Cone(c) => along(c.frame.z) && on(c.frame.origin),
        Surface::Torus(c) => along(c.frame.z) && on(c.frame.origin),
        Surface::Sphere(c) => on(c.frame.origin),
        _ => false,
    }
}

/// The parameters of `rim` at which `wall`'s distance can be greatest or least along it: its ends
/// first, then its middle and every turning point between. `None` where they are not known in
/// closed form: against a plane the rim must be a line or a circle; against a wall turned about
/// an axis it must be a circle about that axis, or a line or circle in a plane through the axis
/// and to one side of it, since the distance then depends only on the height along the axis and
/// the distance from it.
fn turning_points(wall: &Wall, rim: &Rim) -> Option<Vec<f64>> {
    let (t0, t1) = rim.range;
    let mut ts = vec![t0, t1, 0.5 * (t0 + t1)];
    if !matches!(rim.curve, Curve::Line(_) | Curve::Circle(_)) {
        return None;
    }
    if let Surface::Plane(p) = &wall.surface {
        ts.extend(rim.curve.extremes(p.frame.z, t0, t1));
        return Some(ts);
    }
    let (o, k) = axis(&wall.surface)?;
    let mid = rim.curve.eval(ts[2]);
    let side = off_axis(mid, o, k).unit();
    // The point the distance is measured from in the half plane, for a sphere and a torus.
    let focus = |side: Vec3| match &wall.surface {
        Surface::Sphere(s) => Some(s.frame.origin),
        Surface::Torus(t) => Some(t.frame.origin + side * t.major),
        _ => None,
    };
    let one_side = |side: Option<Vec3>| {
        side.is_none_or(|s| {
            ts.iter()
                .all(|&t| off_axis(rim.curve.eval(t), o, k).dot(s) >= -tol::LINEAR)
        })
    };
    match &rim.curve {
        Curve::Circle(c)
            if c.frame.z.cross(k).norm() <= 1e-9
                && off_axis(c.frame.origin, o, k).norm() <= tol::LINEAR => {}
        Curve::Line(l) => {
            if (l.origin - o).dot(l.dir.cross(k)).abs() > tol::LINEAR || !one_side(side) {
                return None;
            }
            if let Some(f) = focus(side.unwrap_or(k.any_perpendicular())) {
                let t = (f - l.origin).dot(l.dir);
                if t > t0 && t < t1 {
                    ts.push(t);
                }
            }
        }
        Curve::Circle(c) => {
            let side = side?;
            if c.frame.z.dot(k).abs() > 1e-9
                || (c.frame.origin - o).dot(c.frame.z).abs() > tol::LINEAR
                || !one_side(Some(side))
            {
                return None;
            }
            let towards = match &wall.surface {
                Surface::Cylinder(_) => side,
                Surface::Cone(cone) => {
                    let (sa, ca) = cone.half_angle.sin_cos();
                    side * ca - cone.frame.z * sa
                }
                _ => c.frame.origin - focus(side)?,
            };
            if towards.norm() > tol::LINEAR {
                ts.extend(rim.curve.extremes(towards, t0, t1));
            }
        }
        _ => return None,
    }
    Some(ts)
}

/// The side of each wall on which every rim lies, strictly between its ends, unless it lies on
/// that wall by the way it was made: the side of that wall the region the rims bound lies on.
/// `faces` are the curved faces of the region's boundary, each with whether one of its rims is a
/// meridian of it. `None` where the rims lie on both sides of a wall, touch a wall they do not lie
/// on, or the test has no closed form.
///
/// The rims decide it for the whole region: against a plane, the distance is linear, so over a
/// planar, cylindrical or conical face it is greatest and least on the face's edges, and over a
/// sphere or torus turned about the plane's normal on its meridian; against a wall turned about
/// an axis, every face must be turned about that axis too, and the distance is then a function on
/// the half plane through the axis, where each face is a curve the rims hold.
pub(crate) fn sides(walls: &[Wall], rims: &[Rim], faces: &[(&Surface, bool)]) -> Option<Vec<Side>> {
    (0..walls.len())
        .map(|k| {
            side(
                &walls[k],
                rims.iter().filter(|r| !r.on.contains(&k)),
                faces,
                false,
            )
        })
        .collect()
}

/// Whether the region `rims` bound lies strictly to one side of `surface`, touching it nowhere:
/// no face on that surface then reaches the region, unless it lies wholly inside it.
pub(crate) fn clear_of(surface: &Surface, rims: &[Rim], faces: &[(&Surface, bool)]) -> bool {
    let wall = Wall {
        surface: surface.clone(),
        reversed: false,
    };
    side(&wall, rims.iter(), faces, true).is_some()
}

/// [`sides`] for one wall and the rims that do not lie on it; with `apart`, a rim may not touch
/// the wall even at its ends.
fn side<'a>(
    wall: &Wall,
    rims: impl Iterator<Item = &'a Rim>,
    faces: &[(&Surface, bool)],
    apart: bool,
) -> Option<Side> {
    for &(s, meridian) in faces {
        let ok = match (&wall.surface, s) {
            (Surface::Plane(_), Surface::Plane(_) | Surface::Cylinder(_) | Surface::Cone(_)) => {
                true
            }
            (Surface::Plane(p), Surface::Sphere(_) | Surface::Torus(_)) => {
                let (_, z) = axis(s)?;
                meridian && z.cross(p.frame.z).norm() <= 1e-9
            }
            (w, _) => {
                let (o, z) = axis(w)?;
                let round = matches!(s, Surface::Sphere(_) | Surface::Torus(_));
                turned_about(s, o, z) && (meridian || !round)
            }
        };
        if !ok {
            return None;
        }
    }
    let mut found: Option<Side> = None;
    for rim in rims {
        let ts = turning_points(wall, rim)?;
        let ends = [rim.curve.eval(ts[0]), rim.curve.eval(ts[1])];
        for (i, &t) in ts.iter().enumerate() {
            let p = rim.curve.eval(t);
            let d = wall.out(p);
            // A rim tangent to the wall at an end turns there: that is the end, not a touch.
            let at_end = i < 2 || (i > 2 && ends.iter().any(|e| e.dist(p) <= tol::LINEAR));
            let here = if d < -tol::LINEAR {
                Side::Within
            } else if d > tol::LINEAR {
                Side::Beyond
            } else if at_end && !apart {
                continue;
            } else {
                return None;
            };
            if *found.get_or_insert(here) != here {
                return None;
            }
        }
    }
    found
}
