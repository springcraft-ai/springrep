//! The topology arena: solids of shells of faces bounded by loops of coedges, each the use of an
//! edge by one face. Every entity lives in one [`Brep`] and is referred to by a typed id.

mod codec;
mod facts;
mod memo;
mod names;
pub(crate) mod pcurve;
pub(crate) mod radial;
mod similar;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

pub use facts::{Facts, OnFace};
pub(crate) use memo::{ArenaMemo, FaceMemo, FaceVolume, IntegralKey};
pub use names::FaceName;
pub use pcurve::PCurve;
pub(crate) use pcurve::project_on;

use crate::{
    geom::{Curve, Surface},
    math::{Point3, Transform, Vec2},
    tol,
};

macro_rules! id {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }

            fn at(i: usize) -> $name {
                $name(u32::try_from(i).expect("arena index fits in u32"))
            }
        }
    };
}

id!(VertexId);
id!(EdgeId);
id!(CoedgeId);
id!(LoopId);
id!(FaceId);
id!(ShellId);
id!(SolidId);

#[derive(Clone, Debug, PartialEq)]
pub struct Vertex {
    pub point: Point3,
}

/// A bounded piece of a curve from `curve(range.0)` at `start` to `curve(range.1)` at `end`,
/// `range.0 < range.1`. A degenerate edge has no curve: it is a pole, a single vertex that its face
/// spans a stretch of parameter plane at, and `range` is the parameter its pcurve runs over.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub curve: Option<Curve>,
    pub range: (f64, f64),
    pub start: VertexId,
    pub end: VertexId,
}

impl Edge {
    pub fn is_degenerate(&self) -> bool {
        self.curve.is_none()
    }
}

/// One face's use of an edge, along the edge's parameter or against it when `reversed`.
#[derive(Clone, Debug, PartialEq)]
pub struct Coedge {
    pub edge: EdgeId,
    pub reversed: bool,
    pub pcurve: PCurve,
    pub lp: LoopId,
}

/// A closed chain of coedges with the face's material on its left, seen from outside the solid.
#[derive(Clone, Debug, PartialEq)]
pub struct Loop {
    pub coedges: Vec<CoedgeId>,
    pub face: FaceId,
}

/// A bounded piece of a surface. Its outward normal is the surface's natural normal, or its
/// opposite when `reversed`. The first loop is the outer one.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    pub surface: Surface,
    pub reversed: bool,
    pub loops: Vec<LoopId>,
    pub names: BTreeSet<FaceName>,
    pub shell: Option<ShellId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shell {
    pub faces: Vec<FaceId>,
    pub solid: Option<SolidId>,
}

/// A solid: one connected piece of material. Its first `outer` shells bound it from outside, any
/// others bound voids. Each shell is a sheet: faces joined edge to edge through each coedge's
/// partner. A solid has several outer sheets where pieces touch along an edge or at a vertex they
/// share, which is the solid's own: an edge used by four faces, a vertex two fans meet at.
#[derive(Clone, Debug, PartialEq)]
pub struct Solid {
    pub shells: Vec<ShellId>,
    pub outer: usize,
}

/// What `Brep::add_loop` needs for each coedge.
#[derive(Clone, Debug, PartialEq)]
pub struct CoedgeSpec {
    pub edge: EdgeId,
    pub reversed: bool,
    pub pcurve: PCurve,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Brep {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    coedges: Vec<Coedge>,
    loops: Vec<Loop>,
    faces: Vec<Face>,
    shells: Vec<Shell>,
    solids: Vec<Solid>,
    memos: memo::Memos,
    arena: memo::Arena,
    /// Local tolerances above [`tol::LINEAR`], which only an import's edges and vertices carry.
    tolerances: Box<Tolerances>,
    /// Each coedge of an edge more than two faces use, with the coedge across the wedge of
    /// material it bounds. An edge two faces use pairs its two coedges without an entry.
    partners: BTreeMap<CoedgeId, CoedgeId>,
    /// Each coedge of such an edge with the coedge across the empty wedge next to its face, round
    /// the edge the other way from its partner.
    beside: BTreeMap<CoedgeId, CoedgeId>,
    /// Each edge's coedges, in the order they were added: worked out from the coedges, never
    /// encoded.
    uses: Vec<Vec<CoedgeId>>,
}

/// Each vertex and each edge with a local tolerance, with it.
pub type Tolerant = (Vec<(VertexId, f64)>, Vec<(EdgeId, f64)>);

/// The vertices and edges of imported geometry that meet only to their source's tolerance
/// (doc/design.md, "Tolerance policy"). A vertex's sphere holds the tube of every edge ending at
/// it, and two entities coincide within the sum of their tolerances.
#[derive(Clone, Debug, Default, PartialEq)]
struct Tolerances {
    vertices: BTreeMap<VertexId, f64>,
    edges: BTreeMap<EdgeId, f64>,
}

impl Brep {
    pub fn new() -> Brep {
        Brep::default()
    }

    /// What queries have worked out about the whole arena as it stands.
    pub(crate) fn arena(&self) -> &ArenaMemo {
        self.arena.0.get_or_init(Arc::default)
    }

    /// Drops what was worked out about the whole arena: every method that adds an entity, or
    /// changes one's geometry, tolerance or place in the topology, calls it first.
    fn changed(&mut self) {
        self.arena = memo::Arena::default();
    }

    pub(crate) fn add_vertex(&mut self, point: Point3) -> VertexId {
        self.changed();
        self.vertices.push(Vertex { point });
        VertexId::at(self.vertices.len() - 1)
    }

    pub(crate) fn add_edge(
        &mut self,
        curve: Curve,
        range: (f64, f64),
        start: VertexId,
        end: VertexId,
    ) -> EdgeId {
        self.changed();
        self.edges.push(Edge {
            curve: Some(curve),
            range,
            start,
            end,
        });
        self.uses.push(Vec::new());
        EdgeId::at(self.edges.len() - 1)
    }

    pub(crate) fn add_degenerate_edge(&mut self, vertex: VertexId, range: (f64, f64)) -> EdgeId {
        self.changed();
        self.edges.push(Edge {
            curve: None,
            range,
            start: vertex,
            end: vertex,
        });
        self.uses.push(Vec::new());
        EdgeId::at(self.edges.len() - 1)
    }

    pub(crate) fn add_face(
        &mut self,
        surface: Surface,
        reversed: bool,
        names: impl IntoIterator<Item = FaceName>,
    ) -> FaceId {
        self.changed();
        self.faces.push(Face {
            surface,
            reversed,
            loops: Vec::new(),
            names: names.into_iter().collect(),
            shell: None,
        });
        self.memos.0.push(Arc::default());
        FaceId::at(self.faces.len() - 1)
    }

    pub(crate) fn add_loop(&mut self, face: FaceId, coedges: Vec<CoedgeSpec>) -> LoopId {
        self.changed();
        let lp = LoopId::at(self.loops.len());
        let mut ids = Vec::with_capacity(coedges.len());
        for c in coedges {
            let edge = c.edge;
            self.coedges.push(Coedge {
                edge,
                reversed: c.reversed,
                pcurve: c.pcurve,
                lp,
            });
            let id = CoedgeId::at(self.coedges.len() - 1);
            self.uses[edge.index()].push(id);
            ids.push(id);
        }
        self.loops.push(Loop { coedges: ids, face });
        self.faces[face.index()].loops.push(lp);
        self.memos.0[face.index()] = Arc::default();
        lp
    }

    pub(crate) fn add_shell(&mut self, faces: Vec<FaceId>) -> ShellId {
        self.changed();
        let id = ShellId::at(self.shells.len());
        for &f in &faces {
            self.faces[f.index()].shell = Some(id);
        }
        self.shells.push(Shell { faces, solid: None });
        id
    }

    /// A solid of one outer shell, the first, and voids.
    pub(crate) fn add_solid(&mut self, shells: Vec<ShellId>) -> SolidId {
        self.add_solid_of(shells, 1)
    }

    /// A solid whose first `outer` shells bound it from outside and the rest voids.
    pub(crate) fn add_solid_of(&mut self, shells: Vec<ShellId>, outer: usize) -> SolidId {
        self.changed();
        let id = SolidId::at(self.solids.len());
        for &s in &shells {
            self.shells[s.index()].solid = Some(id);
        }
        self.solids.push(Solid { shells, outer });
        id
    }

    /// Pairs coedges `a` and `b` of one edge across the wedge of material between their faces: for
    /// an edge more than two faces use.
    pub fn pair(&mut self, a: CoedgeId, b: CoedgeId) {
        self.changed();
        self.partners.insert(a, b);
        self.partners.insert(b, a);
    }

    /// Sets coedges `a` and `b` of one edge either side of an empty wedge between their faces: for
    /// an edge more than two faces use.
    pub(crate) fn set_beside(&mut self, a: CoedgeId, b: CoedgeId) {
        self.changed();
        self.beside.insert(a, b);
        self.beside.insert(b, a);
    }

    /// The coedge across the empty wedge next to coedge `c`'s face at its edge: on an edge two
    /// faces use, its partner, the two faces bounding the empty space outside them both; on one
    /// more use, the neighbour the other way round from its partner.
    pub fn beside(&self, c: CoedgeId) -> Option<CoedgeId> {
        let us = &self.uses[self.coedges[c.index()].edge.index()];
        match us.len() {
            2 => us.iter().copied().find(|&o| o != c),
            n if n > 2 => self.beside.get(&c).copied(),
            _ => None,
        }
    }

    /// The coedges using edge `e`.
    pub fn edge_uses(&self, e: EdgeId) -> &[CoedgeId] {
        &self.uses[e.index()]
    }

    /// The coedge across the wedge of material coedge `c`'s face bounds at its edge: the edge's
    /// other use where two faces use it, the paired one where more do. `None` on a pole, and on an
    /// edge of more uses left unpaired.
    pub fn partner(&self, c: CoedgeId) -> Option<CoedgeId> {
        let us = &self.uses[self.coedges[c.index()].edge.index()];
        match us.len() {
            2 => us.iter().copied().find(|&o| o != c),
            n if n > 2 => self.partners.get(&c).copied(),
            _ => None,
        }
    }

    /// The edges more than two faces use: where pieces of a solid touch.
    pub fn radial_edges(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edge_ids().filter(|&e| self.uses[e.index()].len() > 2)
    }

    /// Pairs the coedges of every edge more than two faces use that are not paired yet, from the
    /// faces' geometry round it: for an arena built from another one's faces.
    pub(crate) fn pair_radial(&mut self) {
        self.changed();
        let edges: Vec<EdgeId> = self.radial_edges().collect();
        for e in edges {
            let us = self.uses[e.index()].clone();
            if us.iter().all(|c| self.partners.contains_key(c)) {
                continue;
            }
            let edge = &self.edges[e.index()];
            let Some(curve) = &edge.curve else { continue };
            let (m, d) = curve.d1(0.5 * (edge.range.0 + edge.range.1));
            let Some(t) = d.unit() else { continue };
            let len = crate::topo::radial::length(curve, edge.range);
            let sides: Vec<radial::Side> = us
                .iter()
                .map(|&c| {
                    let face = &self.faces[self.coedge_face(c).index()];
                    radial::Side {
                        surface: &face.surface,
                        reversed: face.reversed,
                        travel: if self.coedges[c.index()].reversed {
                            -t
                        } else {
                            t
                        },
                    }
                })
                .collect();
            let Some((pairs, beside)) = radial::round(m, t, len, &sides) else {
                continue;
            };
            for (i, j) in pairs {
                self.pair(us[i], us[j]);
            }
            for (i, j) in beside {
                self.set_beside(us[i], us[j]);
            }
        }
    }

    /// Every edge's coedges, worked out again from the coedges.
    pub(crate) fn index_uses(&mut self) {
        self.uses = vec![Vec::new(); self.edges.len()];
        for (i, c) in self.coedges.iter().enumerate() {
            self.uses[c.edge.index()].push(CoedgeId::at(i));
        }
    }

    pub fn vertex(&self, id: VertexId) -> &Vertex {
        &self.vertices[id.index()]
    }

    pub fn edge(&self, id: EdgeId) -> &Edge {
        &self.edges[id.index()]
    }

    pub fn coedge(&self, id: CoedgeId) -> &Coedge {
        &self.coedges[id.index()]
    }

    pub fn lp(&self, id: LoopId) -> &Loop {
        &self.loops[id.index()]
    }

    /// Whether loop `id` is a single point inside its face: one coedge of a degenerate edge, where
    /// the solid touches itself there or another piece of it touches the face.
    pub fn is_vertex_loop(&self, id: LoopId) -> bool {
        let cs = &self.lp(id).coedges;
        cs.len() == 1 && self.edge(self.coedge(cs[0]).edge).is_degenerate()
    }

    pub fn face(&self, id: FaceId) -> &Face {
        &self.faces[id.index()]
    }

    pub(crate) fn face_mut(&mut self, id: FaceId) -> &mut Face {
        self.changed();
        self.memos.0[id.index()] = Arc::default();
        &mut self.faces[id.index()]
    }

    /// Replaces every name of every face by what `names_for` gives for it: no names to drop it,
    /// itself to keep it. Nothing else about the solid changes.
    pub fn rename_faces<I>(&mut self, mut names_for: impl FnMut(&FaceName) -> I)
    where
        I: IntoIterator<Item = FaceName>,
    {
        for face in &mut self.faces {
            face.names = face.names.iter().flat_map(&mut names_for).collect();
        }
    }

    /// Renames a face, keeping what has been worked out about its geometry.
    pub(crate) fn set_face_names(&mut self, id: FaceId, names: BTreeSet<FaceName>) {
        self.faces[id.index()].names = names;
    }

    pub(crate) fn memo(&self, id: FaceId) -> &FaceMemo {
        &self.memos.0[id.index()]
    }

    pub(crate) fn shared_memo(&self, id: FaceId) -> Arc<FaceMemo> {
        #[cfg(debug_assertions)]
        self.memo(id).layout.get_or_init(|| self.layout(id));
        self.memos.0[id.index()].clone()
    }

    /// Gives face `id` the memo of a face with exactly its geometry, its loops and coedges in the
    /// same order on edges of the same ranges: the memo keeps some of what it holds by a
    /// coedge's place, and an edge's parameters with it.
    pub(crate) fn share_memo(&mut self, id: FaceId, memo: Arc<FaceMemo>) {
        self.memos.0[id.index()] = memo;
        #[cfg(debug_assertions)]
        if let Some(was) = self.memo(id).layout.get() {
            debug_assert!(
                *was == self.layout(id),
                "face {} takes the memo of a face laid out another way",
                id.index()
            );
        }
    }

    #[cfg(debug_assertions)]
    fn layout(&self, id: FaceId) -> Vec<Vec<(u64, u64, bool)>> {
        let of = |&c: &CoedgeId| {
            let co = self.coedge(c);
            let (t0, t1) = self.edge(co.edge).range;
            (t0.to_bits(), t1.to_bits(), co.reversed)
        };
        let coedges = |&lp: &LoopId| self.lp(lp).coedges.iter().map(of).collect();
        self.face(id).loops.iter().map(coedges).collect()
    }

    pub fn shell(&self, id: ShellId) -> &Shell {
        &self.shells[id.index()]
    }

    pub fn solid(&self, id: SolidId) -> &Solid {
        &self.solids[id.index()]
    }

    pub fn vertex_ids(&self) -> impl Iterator<Item = VertexId> + use<> {
        (0..self.vertices.len()).map(VertexId::at)
    }

    pub fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + use<> {
        (0..self.edges.len()).map(EdgeId::at)
    }

    pub fn coedge_ids(&self) -> impl Iterator<Item = CoedgeId> + use<> {
        (0..self.coedges.len()).map(CoedgeId::at)
    }

    pub fn loop_ids(&self) -> impl Iterator<Item = LoopId> + use<> {
        (0..self.loops.len()).map(LoopId::at)
    }

    pub fn face_ids(&self) -> impl Iterator<Item = FaceId> + use<> {
        (0..self.faces.len()).map(FaceId::at)
    }

    pub fn shell_ids(&self) -> impl Iterator<Item = ShellId> + use<> {
        (0..self.shells.len()).map(ShellId::at)
    }

    pub fn solid_ids(&self) -> impl Iterator<Item = SolidId> + use<> {
        (0..self.solids.len()).map(SolidId::at)
    }

    /// The radius within which vertex `v` is its point: [`tol::LINEAR`] unless it was imported
    /// with a local tolerance.
    pub fn vertex_tolerance(&self, v: VertexId) -> f64 {
        self.tolerances
            .vertices
            .get(&v)
            .copied()
            .unwrap_or(tol::LINEAR)
    }

    /// The radius of the tube round edge `e`'s curve within which its faces' surfaces pass.
    pub fn edge_tolerance(&self, e: EdgeId) -> f64 {
        self.tolerances
            .edges
            .get(&e)
            .copied()
            .unwrap_or(tol::LINEAR)
    }

    /// Gives an imported vertex a tolerance; a value at or below [`tol::LINEAR`] clears it.
    pub(crate) fn set_vertex_tolerance(&mut self, v: VertexId, t: f64) {
        self.changed();
        let old = self.vertex_tolerance(v);
        if t > tol::LINEAR {
            self.tolerances.vertices.insert(v, t);
        } else {
            self.tolerances.vertices.remove(&v);
        }
        if self.vertex_tolerance(v) != old {
            self.forget_faces_of(|b, e| b.edge(e).start == v || b.edge(e).end == v);
        }
    }

    /// Gives an imported edge a tolerance; a value at or below [`tol::LINEAR`] clears it.
    pub(crate) fn set_edge_tolerance(&mut self, e: EdgeId, t: f64) {
        self.changed();
        let old = self.edge_tolerance(e);
        if t > tol::LINEAR {
            self.tolerances.edges.insert(e, t);
        } else {
            self.tolerances.edges.remove(&e);
        }
        if self.edge_tolerance(e) != old {
            self.forget_faces_of(|_, x| x == e);
        }
    }

    /// Drops what has been worked out about the faces with an edge `uses` picks: a tolerance
    /// changes whether their geometry is sound.
    fn forget_faces_of(&mut self, uses: impl Fn(&Brep, EdgeId) -> bool) {
        let faces: Vec<FaceId> = self
            .coedges
            .iter()
            .filter(|c| uses(self, c.edge))
            .map(|c| self.loops[c.lp.index()].face)
            .collect();
        for f in faces {
            self.memos.0[f.index()] = Arc::default();
        }
    }

    /// The vertices and edges with a tolerance above [`tol::LINEAR`], each with it.
    pub fn tolerant(&self) -> Tolerant {
        (
            self.tolerances
                .vertices
                .iter()
                .map(|(&v, &t)| (v, t))
                .collect(),
            self.tolerances
                .edges
                .iter()
                .map(|(&e, &t)| (e, t))
                .collect(),
        )
    }

    /// Whether any vertex or edge carries a tolerance above [`tol::LINEAR`].
    pub fn has_tolerances(&self) -> bool {
        !self.tolerances.vertices.is_empty() || !self.tolerances.edges.is_empty()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// The faces of a solid, over all its shells.
    pub fn solid_faces(&self, s: SolidId) -> Vec<FaceId> {
        self.solid(s)
            .shells
            .iter()
            .flat_map(|&sh| self.shell(sh).faces.iter().copied())
            .collect()
    }

    /// The vertex a coedge starts at, in its own direction.
    pub fn coedge_start(&self, c: CoedgeId) -> VertexId {
        let co = self.coedge(c);
        let e = self.edge(co.edge);
        if co.reversed { e.end } else { e.start }
    }

    pub fn coedge_end(&self, c: CoedgeId) -> VertexId {
        let co = self.coedge(c);
        let e = self.edge(co.edge);
        if co.reversed { e.start } else { e.end }
    }

    pub fn coedge_face(&self, c: CoedgeId) -> FaceId {
        self.lp(self.coedge(c).lp).face
    }

    /// The coedge's parameter-plane points at edge parameters `ts`, given in the coedge's own
    /// direction (descending when it is reversed).
    pub fn coedge_uvs(&self, c: CoedgeId, ts: &[f64]) -> Vec<Vec2> {
        let co = self.coedge(c);
        let surface = &self.face(self.coedge_face(c)).surface;
        let periodic = surface.u_period().is_some() || surface.v_period().is_some();
        if !periodic || ts.len() < 2 || !matches!(co.pcurve, PCurve::Projected { .. }) {
            return self.coedge_uvs_at(c, ts);
        }
        // An edge winding round its surface several times (a thread's helix) can turn more than
        // a period between the points asked for, and continuation then reads a turn as none. A
        // turn is at least the surface's circumference, so steps an eighth of that along the
        // curve can never skip one; finer samples cannot tell, as their remainders alias alike.
        let (Some(curve), PCurve::Projected { seed }) = (&self.edge(co.edge).curve, &co.pcurve)
        else {
            return self.coedge_uvs_at(c, ts);
        };
        let (t0, t1) = (ts[0], ts[ts.len() - 1]);
        let mut length = 0.0;
        let mut prev = curve.eval(t0);
        for k in 1..=256 {
            let p = curve.eval(t0 + (t1 - t0) * k as f64 / 256.0);
            length += p.dist(prev);
            prev = p;
        }
        let (_, su, sv) = surface.d1(seed.x, seed.y);
        let around = [(surface.u_period(), su), (surface.v_period(), sv)]
            .iter()
            .filter_map(|&(p, d)| p.map(|p| p * d.norm()))
            .filter(|&c| c > 1e-9)
            .fold(f64::INFINITY, f64::min);
        let steps = if around.is_finite() {
            (8.0 * length / around).ceil() as usize
        } else {
            0
        };
        // That counts the turns a curve of this length could make. Where it asks for steps, how
        // fast the curve does turn says how many it needs: an edge along a cylinder's axis turns
        // not at all, however thin the cylinder, and one a tolerance thick would otherwise take
        // millions.
        let steps = if steps >= ts.len() - 1 {
            let n = 32;
            let mut turns = 0.0;
            for k in 0..n {
                let (p, dp) = curve.d1(t0 + (t1 - t0) * (k as f64 + 0.5) / n as f64);
                let uv = surface.project(p);
                let (_, su, sv) = surface.d1(uv.x, uv.y);
                let rate = [(surface.u_period(), su), (surface.v_period(), sv)]
                    .iter()
                    .filter_map(|&(per, d)| per.map(|per| (per * d.norm(), d)))
                    .filter(|&(c, _)| c > 1e-9)
                    .map(|(c, d)| dp.dot(d).abs() / (d.norm() * c))
                    .fold(0.0, f64::max);
                turns += rate * (t1 - t0).abs() / n as f64;
            }
            // Sixteen steps a turn, twice what the bound above takes, for the rate between samples.
            let needed = (16.0 * turns).ceil();
            if needed.is_finite() {
                steps.min(needed as usize)
            } else {
                steps
            }
        } else {
            steps
        };
        let per = (steps / (ts.len() - 1) + 1).min(4096);
        if per <= 1 {
            return self.coedge_uvs_at(c, ts);
        }
        let mut dense = Vec::with_capacity((ts.len() - 1) * per + 1);
        for w in ts.windows(2) {
            for k in 0..per {
                dense.push(w[0] + (w[1] - w[0]) * k as f64 / per as f64);
            }
        }
        dense.push(t1);
        let uvs = self.coedge_uvs_at(c, &dense);
        (0..ts.len()).map(|i| uvs[i * per]).collect()
    }

    /// Loop `lp`'s projected coedges each moved by whole periods to start where the coedge before
    /// it ends, going round from a coedge whose pcurve is exact where there is one. A seed placed
    /// by nearness to a hint comes out a turn away where the hint lies half a turn off, as a
    /// point just past a pole does from one just short of it.
    pub(crate) fn continue_loop(&mut self, lp: LoopId) {
        let face = self.lp(lp).face;
        let surface = &self.face(face).surface;
        let periods = [surface.u_period(), surface.v_period()];
        if periods == [None, None] {
            return;
        }
        let cs = self.lp(lp).coedges.clone();
        let n = cs.len();
        let projected = |c: CoedgeId| matches!(self.coedge(c).pcurve, PCurve::Projected { .. });
        let first = cs.iter().position(|&c| !projected(c)).unwrap_or(0);
        let ends = |c: CoedgeId| {
            let co = self.coedge(c);
            let (t0, t1) = self.edge(co.edge).range;
            let uvs = self.coedge_uvs(c, &if co.reversed { [t1, t0] } else { [t0, t1] });
            (uvs[0], uvs[1])
        };
        let mut moves: Vec<(CoedgeId, Vec2)> = Vec::new();
        let mut at = ends(cs[first]).1;
        for k in 1..n {
            let c = cs[(first + k) % n];
            let (start, end) = ends(c);
            if !projected(c) {
                at = end;
                continue;
            }
            let turn = |from: f64, to: f64, period: Option<f64>| match period {
                Some(p) => ((to - from) / p).round() * p,
                None => 0.0,
            };
            let by = Vec2::new(
                turn(start.x, at.x, periods[0]),
                turn(start.y, at.y, periods[1]),
            );
            if by != Vec2::ZERO {
                moves.push((c, by));
            }
            at = end + by;
        }
        if moves.is_empty() {
            return;
        }
        self.changed();
        self.memos.0[face.index()] = Arc::default();
        for (c, by) in moves {
            if let PCurve::Projected { seed } = &mut self.coedges[c.index()].pcurve {
                *seed = *seed + by;
            }
        }
    }

    /// [`Brep::coedge_uvs`] continued from each point asked for to the next.
    fn coedge_uvs_at(&self, c: CoedgeId, ts: &[f64]) -> Vec<Vec2> {
        let co = self.coedge(c);
        let edge = self.edge(co.edge);
        match &co.pcurve {
            PCurve::Affine { origin, dir } => ts.iter().map(|&t| *origin + *dir * t).collect(),
            PCurve::Spline { uv } => ts.iter().map(|&t| pcurve::spline_uv(uv, t)).collect(),
            PCurve::Projected { seed } => {
                let f = self.coedge_face(c);
                let surface = &self.face(f).surface;
                let past_pole = |at: Vec2| self.past_pole(f, at);
                let mut hint = *seed;
                let mut out = Vec::with_capacity(ts.len());
                for (i, &t) in ts.iter().enumerate() {
                    let beside = if i == 0 { ts.get(1) } else { ts.get(i - 1) };
                    let inside = t > edge.range.0 && t < edge.range.1;
                    let uv = match (&edge.curve, beside) {
                        (Some(curve), Some(&b)) if inside && i > 0 => {
                            pcurve::follow(surface, curve, (b, t), hint, &past_pole)
                        }
                        (Some(curve), Some(&b)) => {
                            let uv = pcurve::project_on(surface, curve, t, b, hint);
                            let p = curve.eval(t);
                            self.exact_beside(c, i == 0, p, uv).unwrap_or(uv)
                        }
                        (Some(curve), None) => surface.project_near(curve.eval(t), hint),
                        (None, _) => surface.project_near(self.vertex(edge.start).point, hint),
                    };
                    hint = uv;
                    out.push(uv);
                }
                out
            }
        }
    }

    /// Which way `u` turns, by half a period, where a coedge of face `f` passes through the pole of
    /// its sphere that the parameters `at` lie beside; 0 on other surfaces. The face's loops,
    /// running with it on their left, cover the pole's line in the parameter plane once over the
    /// longitudes the face takes there. Where no loop has an edge at the pole, the coedge alone
    /// runs across those, as the face's outline does along the top of its rectangle; where one
    /// runs the pole's whole line, the coedge runs back across the longitudes the face lacks.
    fn past_pole(&self, f: FaceId, at: Vec2) -> f64 {
        let face = self.face(f);
        let Surface::Sphere(s) = &face.surface else {
            return 0.0;
        };
        let up = at.y > 0.0;
        let pole = s.frame.origin + s.frame.z * if up { s.radius } else { -s.radius };
        let lined = face.loops.iter().any(|&lp| {
            self.lp(lp).coedges.iter().any(|&c| {
                let e = self.edge(self.coedge(c).edge);
                e.curve.is_none() && self.vertex(e.start).point.dist(pole) <= tol::LINEAR
            })
        });
        if (up != face.reversed) != lined {
            -1.0
        } else {
            1.0
        }
    }

    /// The parameters at the start (or end) of coedge `c` that the coedge beside it there gives the
    /// shared vertex, where that coedge's pcurve is exact and reaches `p`, the periodic parameters
    /// moved nearest `near`. Where the surface's map is nearly singular, as along a fold of a swept
    /// sheet, a projected end lands anywhere along the direction the map barely moves in, and the
    /// two ends must still meet in the parameter plane.
    fn exact_beside(&self, c: CoedgeId, at_start: bool, p: Point3, near: Vec2) -> Option<Vec2> {
        let co = self.coedge(c);
        let cs = &self.lp(co.lp).coedges;
        let k = cs.iter().position(|&x| x == c)?;
        let n = cs.len();
        let other = if at_start {
            cs[(k + n - 1) % n]
        } else {
            cs[(k + 1) % n]
        };
        if other == c {
            return None;
        }
        let oc = self.coedge(other);
        let range = self.edge(oc.edge).range;
        let t = if at_start != oc.reversed {
            range.1
        } else {
            range.0
        };
        let uv = match &oc.pcurve {
            PCurve::Affine { origin, dir } => *origin + *dir * t,
            PCurve::Spline { uv } => pcurve::spline_uv(uv, t),
            PCurve::Projected { .. } => return None,
        };
        let surface = &self.face(self.coedge_face(c)).surface;
        let shift = |x: f64, h: f64, period: Option<f64>| match period {
            Some(t) => x + ((h - x) / t).round() * t,
            None => x,
        };
        let uv = Vec2::new(
            shift(uv.x, near.x, surface.u_period()),
            shift(uv.y, near.y, surface.v_period()),
        );
        (surface.eval(uv.x, uv.y).dist(p) <= tol::LINEAR).then_some(uv)
    }

    /// How many faces carry each name, over the whole arena.
    pub fn name_counts(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for f in &self.faces {
            for n in &f.names {
                *out.entry(n.to_string()).or_insert(0) += 1;
            }
        }
        out
    }

    /// Moves every entity by `t`. Parameters are unchanged: frames and curves move with the
    /// geometry, so pcurves stay valid.
    pub fn transform(&mut self, t: &Transform) {
        self.changed();
        for v in &mut self.vertices {
            v.point = t.point(v.point);
        }
        for e in &mut self.edges {
            if let Some(c) = &e.curve {
                e.curve = Some(c.transformed(t));
            }
        }
        for f in &mut self.faces {
            f.surface = f.surface.transformed(t);
        }
        // A face's volume share about its own anchor, and whether its loops are clean, are the
        // same wherever a rigid motion puts it: an assembly sweep moves a part before each
        // boolean, and its spline faces would otherwise be integrated again at every step.
        // Slid, a face's box slides with it; turned, the turned box's own would be looser.
        let slid = t.r == Transform::IDENTITY.r;
        for m in &mut self.memos.0 {
            let fresh = FaceMemo::default();
            if let (true, Some(b)) = (slid, m.loose_box.get()) {
                let _ = fresh.loose_box.set(crate::math::Aabb {
                    min: b.min + t.t,
                    max: b.max + t.t,
                });
            }
            if let Some(v) = m.volume.get() {
                let _ = fresh.volume.set(FaceVolume {
                    anchor: t.point(v.anchor),
                    volume: v.volume,
                    vector_area: t.vector(v.vector_area),
                });
            }
            if m.clean.get().is_some() {
                let _ = fresh.clean.set(());
            }
            *m = Arc::new(fresh);
        }
    }

    /// Copies every entity of `other` into `self`, and returns the new ids of its solids.
    pub fn append(&mut self, other: &Brep) -> Vec<SolidId> {
        self.changed();
        let (v0, e0, c0, l0, f0, sh0, so0) = (
            self.vertices.len(),
            self.edges.len(),
            self.coedges.len(),
            self.loops.len(),
            self.faces.len(),
            self.shells.len(),
            self.solids.len(),
        );
        self.vertices.extend(other.vertices.iter().cloned());
        self.edges.extend(other.edges.iter().map(|e| Edge {
            curve: e.curve.clone(),
            range: e.range,
            start: VertexId::at(e.start.index() + v0),
            end: VertexId::at(e.end.index() + v0),
        }));
        self.coedges.extend(other.coedges.iter().map(|c| Coedge {
            edge: EdgeId::at(c.edge.index() + e0),
            reversed: c.reversed,
            pcurve: c.pcurve.clone(),
            lp: LoopId::at(c.lp.index() + l0),
        }));
        self.loops.extend(other.loops.iter().map(|l| {
            Loop {
                coedges: l
                    .coedges
                    .iter()
                    .map(|c| CoedgeId::at(c.index() + c0))
                    .collect(),
                face: FaceId::at(l.face.index() + f0),
            }
        }));
        self.faces.extend(other.faces.iter().map(|f| Face {
            surface: f.surface.clone(),
            reversed: f.reversed,
            loops: f.loops.iter().map(|l| LoopId::at(l.index() + l0)).collect(),
            names: f.names.clone(),
            shell: f.shell.map(|s| ShellId::at(s.index() + sh0)),
        }));
        self.memos.0.extend(other.memos.0.iter().cloned());
        self.shells.extend(other.shells.iter().map(|s| Shell {
            faces: s.faces.iter().map(|f| FaceId::at(f.index() + f0)).collect(),
            solid: s.solid.map(|s| SolidId::at(s.index() + so0)),
        }));
        self.solids.extend(other.solids.iter().map(|s| {
            Solid {
                shells: s
                    .shells
                    .iter()
                    .map(|sh| ShellId::at(sh.index() + sh0))
                    .collect(),
                outer: s.outer,
            }
        }));
        for (&a, &b) in &other.partners {
            self.partners
                .insert(CoedgeId::at(a.index() + c0), CoedgeId::at(b.index() + c0));
        }
        for (&a, &b) in &other.beside {
            self.beside
                .insert(CoedgeId::at(a.index() + c0), CoedgeId::at(b.index() + c0));
        }
        self.index_uses();
        for (&v, &t) in &other.tolerances.vertices {
            self.tolerances
                .vertices
                .insert(VertexId::at(v.index() + v0), t);
        }
        for (&e, &t) in &other.tolerances.edges {
            self.tolerances.edges.insert(EdgeId::at(e.index() + e0), t);
        }
        (so0..self.solids.len()).map(SolidId::at).collect()
    }
}
