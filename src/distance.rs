//! The closest distance between two sets of faces, edges and vertices, from the exact geometry.
//!
//! Both sides are meshed, and the closest pair of mesh pieces gives the distance to within the two
//! meshes' deflections. Every pair of pieces that close to it seeds a Newton search for a
//! stationary point of the exact distance between the surfaces, curves and points they come from,
//! and a result counts only where both its points lie on their face or edge. The true minimum lies
//! in the basin of one of those seeds, so the least result is exact. Where a surface's nearest
//! point lies outside its face, the face's minimum is on its boundary, which its edges' pieces
//! cover. Whether a point lies on its face, and whether one solid holds another, are asked of the
//! exact faces, as the booleans ask them: the meshes only seed the search. Solids that share
//! material are at distance zero: a vertex of one, or the middle of an edge, clearly inside the
//! other says so before either is meshed, and a corner of one's mesh clearly inside the other
//! before the search.

use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::{
    boolean::{
        classify,
        csi::signed_distance,
        domain::{Domain, Where},
        face_domain,
    },
    error::Result,
    geom::{Curve, Surface},
    math::{Aabb, Point3, Vec3},
    mesh::{
        self,
        bvh::{Bvh, on_segment, on_triangle, segment_triangle, segments},
    },
    tol,
    topo::{Brep, EdgeId, FaceId, SolidId, VertexId},
};

/// What one side of a distance is: faces, edges, vertices and solids of one arena. A face brings
/// its edges and vertices, an edge its vertices, and a solid its faces, and counts its inside too.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Selection {
    pub faces: Vec<FaceId>,
    pub edges: Vec<EdgeId>,
    pub vertices: Vec<VertexId>,
    pub solids: Vec<SolidId>,
}

impl Selection {
    /// Every solid of the arena, inside and boundary.
    pub fn body(b: &Brep) -> Selection {
        Selection {
            solids: b.solid_ids().collect(),
            ..Selection::default()
        }
    }

    pub fn face(f: FaceId) -> Selection {
        Selection {
            faces: vec![f],
            ..Selection::default()
        }
    }

    pub fn faces(faces: impl IntoIterator<Item = FaceId>) -> Selection {
        Selection {
            faces: faces.into_iter().collect(),
            ..Selection::default()
        }
    }

    pub fn edge(e: EdgeId) -> Selection {
        Selection {
            edges: vec![e],
            ..Selection::default()
        }
    }

    pub fn edges(edges: impl IntoIterator<Item = EdgeId>) -> Selection {
        Selection {
            edges: edges.into_iter().collect(),
            ..Selection::default()
        }
    }

    pub fn vertex(v: VertexId) -> Selection {
        Selection {
            vertices: vec![v],
            ..Selection::default()
        }
    }
}

/// The closest distance between two selections, and a point of each at that distance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Closest {
    pub distance: f64,
    pub on_a: Point3,
    pub on_b: Point3,
}

/// Which entity a mesh piece or a candidate point belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Entity {
    Face(FaceId),
    Edge(EdgeId),
    Vertex(VertexId),
}

/// A piece of one side's mesh: a face's triangle, an edge's segment between two curve parameters,
/// or a vertex.
#[derive(Clone, Copy, Debug)]
enum Piece {
    Tri(FaceId, [Point3; 3]),
    Seg(EdgeId, [Point3; 2]),
    Pt(VertexId, Point3),
}

impl Piece {
    fn entity(&self) -> Entity {
        match *self {
            Piece::Tri(f, _) => Entity::Face(f),
            Piece::Seg(e, _) => Entity::Edge(e),
            Piece::Pt(v, _) => Entity::Vertex(v),
        }
    }

    fn bbox(&self) -> Aabb {
        let mut bb = Aabb::EMPTY;
        match self {
            Piece::Tri(_, p) => p.iter().for_each(|&q| bb.add(q)),
            Piece::Seg(_, p) => p.iter().for_each(|&q| bb.add(q)),
            Piece::Pt(_, p) => bb.add(*p),
        }
        bb
    }
}

/// What a selection names, expanded: a face with its edges, an edge with its vertices.
struct Extent<'a> {
    b: &'a Brep,
    faces: Vec<FaceId>,
    edges: Vec<EdgeId>,
    vertices: Vec<VertexId>,
    /// The faces of the selected solids, for whether a point lies inside one.
    solid_faces: Vec<FaceId>,
    /// The arena's own memo of the selection's mesh pieces, where the selection is every solid of
    /// the arena or every face.
    kept: Option<&'a OnceLock<Result<Arc<Meshed>>>>,
}

/// A side's solids' exact faces, each with its box widened by the tolerance: worked out when a
/// point is first asked about, and kept for one distance.
type Domains = OnceCell<Vec<(Domain, Aabb)>>;

/// A selection's mesh pieces, and the hierarchy of their boxes.
#[derive(Debug)]
pub(crate) struct Meshed {
    pieces: Vec<Piece>,
    /// Over each piece's box, in the pieces' order.
    tree: Bvh,
    bbox: Aabb,
    deflection: f64,
}

/// One side: what it names, and its mesh.
struct Side<'a> {
    on: Extent<'a>,
    mesh: Arc<Meshed>,
}

impl<'a> Extent<'a> {
    fn new(b: &'a Brep, sel: &Selection) -> Extent<'a> {
        let mut solid_faces: Vec<FaceId> =
            sel.solids.iter().flat_map(|&s| b.solid_faces(s)).collect();
        solid_faces.sort();
        solid_faces.dedup();
        let mut faces = sel.faces.clone();
        faces.extend(&solid_faces);
        faces.sort();
        faces.dedup();
        let mut edges = sel.edges.clone();
        for &f in &faces {
            for &lp in &b.face(f).loops {
                edges.extend(b.lp(lp).coedges.iter().map(|&co| b.coedge(co).edge));
            }
        }
        edges.sort();
        edges.dedup();
        let mut vertices = sel.vertices.clone();
        for &e in &edges {
            vertices.extend([b.edge(e).start, b.edge(e).end]);
        }
        vertices.sort();
        vertices.dedup();

        let mut solids = sel.solids.clone();
        solids.sort();
        solids.dedup();
        let bare = sel.edges.is_empty() && sel.vertices.is_empty();
        let kept = if bare && sel.faces.is_empty() && b.solid_ids().eq(solids.iter().copied()) {
            Some(&b.arena().solids)
        } else if bare && sel.solids.is_empty() && b.face_ids().eq(faces.iter().copied()) {
            Some(&b.arena().faces)
        } else {
            None
        };
        Extent {
            b,
            faces,
            edges,
            vertices,
            solid_faces,
            kept,
        }
    }

    /// The exact faces of this side's solids.
    fn domains<'d>(&self, kept: &'d Domains) -> Vec<(&'d Domain, Aabb)> {
        let pad = Vec3::new(tol::LINEAR, tol::LINEAR, tol::LINEAR);
        let build = || {
            let padded = |f: FaceId| {
                let bb = crate::measure::face_box(self.b, f);
                Aabb {
                    min: bb.min - pad,
                    max: bb.max + pad,
                }
            };
            self.solid_faces
                .iter()
                .map(|&f| (face_domain(self.b, f), padded(f)))
                .collect()
        };
        kept.get_or_init(build)
            .iter()
            .map(|(d, bb)| (d, *bb))
            .collect()
    }

    /// Face `f`'s exact domain, where it is a face of this side's solids and theirs are worked out.
    fn domain<'d>(&self, kept: &'d Domains, f: FaceId) -> Option<&'d Domain> {
        let i = self.solid_faces.binary_search(&f).ok()?;
        Some(&kept.get()?[i].0)
    }

    /// A vertex of `other`, or the middle of one of its edges, that one of this side's solids
    /// holds, farther than the tolerance from the surface of every face of the solids whose box
    /// reaches it: a point of `other` inside this side, so the two share it. A few of those in
    /// the solids' box are asked; where none is held, `None`. Neither side is meshed for it.
    fn holds_point(&self, other: &Extent, kept: &Domains) -> Option<Point3> {
        if self.solid_faces.is_empty() {
            return None;
        }
        let mut bb = Aabb::EMPTY;
        for &f in &self.solid_faces {
            bb.union(&crate::measure::face_box(self.b, f));
        }
        let within = |bb: &Aabb, p: Point3| (0..3).all(|k| p[k] >= bb.min[k] && p[k] <= bb.max[k]);
        let ends = other.vertices.iter().map(|&v| other.b.vertex(v).point);
        let mids = other.edges.iter().filter_map(|&e| {
            let e = other.b.edge(e);
            let c = e.curve.as_ref()?;
            Some(c.eval(0.5 * (e.range.0 + e.range.1)))
        });
        let inside: Vec<Point3> = ends.chain(mids).filter(|&p| within(&bb, p)).collect();
        if inside.is_empty() {
            return None;
        }
        let faces = self.domains(kept);
        let n = inside.len();
        let tries = n.min(HELD_TRIES);
        (0..tries).map(|k| inside[k * n / tries]).find(|&p| {
            let clear = |(d, bb): &(&Domain, Aabb)| {
                !within(bb, p) || signed_distance(&d.surface, p).abs() > tol::LINEAR
            };
            faces.iter().all(clear) && classify::contains(&faces, p) == Some(true)
        })
    }
}

impl Meshed {
    fn new(on: &Extent) -> Result<Meshed> {
        let b = on.b;
        let seed = mesh::seed_mesh(b)?;
        let (m, opt) = (&seed.mesh, &seed.options);
        let tri = |t: &[u32; 3]| t.map(|i| m.positions[i as usize]);
        let mut pieces = Vec::new();
        for (f, range) in &m.faces {
            if on.faces.binary_search(f).is_ok() {
                pieces.extend(
                    m.triangles[range.clone()]
                        .iter()
                        .map(|t| Piece::Tri(*f, tri(t))),
                );
            }
        }
        for &e in &on.edges {
            let edge = b.edge(e);
            let Some(c) = &edge.curve else { continue };
            let (t0, t1) = edge.range;
            let ts = mesh::edge_params(c, t0, t1, opt);
            for w in ts.windows(2) {
                pieces.push(Piece::Seg(e, [c.eval(w[0]), c.eval(w[1])]));
            }
        }
        for &v in &on.vertices {
            pieces.push(Piece::Pt(v, b.vertex(v).point));
        }
        let boxes: Vec<Aabb> = pieces.iter().map(Piece::bbox).collect();
        let mut bbox = Aabb::EMPTY;
        boxes.iter().for_each(|bb| bbox.union(bb));
        Ok(Meshed {
            pieces,
            tree: Bvh::of_boxes(boxes),
            bbox,
            deflection: opt.linear,
        })
    }
}

impl<'a> Side<'a> {
    fn new(b: &'a Brep, sel: &Selection) -> Result<Side<'a>> {
        Side::meshed(Extent::new(b, sel))
    }

    fn meshed(on: Extent<'a>) -> Result<Side<'a>> {
        let build = || Meshed::new(&on).map(Arc::new);
        let mesh = match on.kept {
            Some(kept) => kept.get_or_init(build).clone()?,
            None => build()?,
        };
        Ok(Side { on, mesh })
    }

    /// The exact geometry an entity lies on.
    fn geo(&self, e: Entity) -> Geo<'a> {
        match e {
            Entity::Face(f) => Geo::Surface(&self.on.b.face(f).surface),
            Entity::Edge(id) => {
                let edge = self.on.b.edge(id);
                let (t0, t1) = edge.range;
                match &edge.curve {
                    Some(c) => Geo::Curve(c, t0.min(t1), t0.max(t1)),
                    None => Geo::Point(self.on.b.vertex(edge.start).point),
                }
            }
            Entity::Vertex(v) => Geo::Point(self.on.b.vertex(v).point),
        }
    }
}

impl Side<'_> {
    /// Whether one of this side's solids holds `inner`'s, given that their boundaries do not meet:
    /// then every point of `inner`'s boundary is in or every one is out, and one of its mesh's
    /// corners, which lie on its exact boundary, says which. The solid's own classifier decides,
    /// on the exact faces: a corner within the mesh's deflection of this side's boundary falls
    /// either side of the mesh, as a dowel a hair smaller than its hole does all round. The point,
    /// in both, where it holds.
    fn holds(&self, inner: &Side, kept: &Domains) -> Option<Point3> {
        if self.on.solid_faces.is_empty() {
            return None;
        }
        // Only a solid whose box holds the other's can hold it.
        let (outer, within) = (self.mesh.bbox, inner.mesh.bbox);
        let pad = self.mesh.deflection + inner.mesh.deflection;
        if (0..3).any(|k| within.min[k] < outer.min[k] - pad || within.max[k] > outer.max[k] + pad)
        {
            return None;
        }
        let faces = self.on.domains(kept);
        // A corner no ray decides lies on this side's boundary within rounding: the next is asked.
        corners(&inner.mesh.pieces)
            .take(16)
            .find_map(|p| classify::contains(&faces, p).map(|inside| inside.then_some(p)))
            .flatten()
    }

    /// A corner of `other`'s mesh that one of this side's solids holds, farther from this side's
    /// mesh than the mesh can lie from the faces: inside by more than
    /// the tolerance, where the classifier is exact, so the two share material there. A few
    /// corners spread over those in this side's box are asked; where none is held, `None`.
    fn holds_corner(&self, other: &Side, kept: &Domains) -> Option<Point3> {
        if self.on.solid_faces.is_empty() {
            return None;
        }
        let (bb, pad) = (self.mesh.bbox, self.mesh.deflection);
        let boxed = |p: Point3| (0..3).all(|k| p[k] >= bb.min[k] - pad && p[k] <= bb.max[k] + pad);
        let inside: Vec<Point3> = corners(&other.mesh.pieces).filter(|&p| boxed(p)).collect();
        if inside.is_empty() {
            return None;
        }
        let clear = 2.0 * self.mesh.deflection + tol::LINEAR;
        let mut hits = Vec::new();
        let mut is_clear = |p: Point3| {
            self.mesh
                .tree
                .within(&Aabb { min: p, max: p }, clear, &mut hits);
            hits.iter().all(|&i| gap(p, &self.mesh.pieces[i]) > clear)
        };
        let n = inside.len();
        let tries = n.min(HELD_TRIES);
        (0..tries)
            .map(|k| inside[k * n / tries])
            .find(|&p| is_clear(p) && classify::contains(&self.on.domains(kept), p) == Some(true))
    }
}

/// Corners of a side's mesh asked whether the other side's solid holds them: enough, spread over
/// the side, that solids sharing a fair part of their volume show it.
const HELD_TRIES: usize = 16;

/// The corners of mesh pieces, every one on its entity's exact geometry.
fn corners(pieces: &[Piece]) -> impl Iterator<Item = Point3> + '_ {
    pieces.iter().flat_map(|p| match *p {
        Piece::Tri(_, t) => t.to_vec(),
        Piece::Seg(_, s) => s.to_vec(),
        Piece::Pt(_, q) => vec![q],
    })
}

/// How far point `p` lies from mesh piece `x`.
fn gap(p: Point3, x: &Piece) -> f64 {
    match x {
        Piece::Tri(_, t) => p.dist(on_triangle(p, t)),
        Piece::Seg(_, s) => p.dist(on_segment(p, s[0], s[1])),
        Piece::Pt(_, q) => p.dist(*q),
    }
}

/// The exact geometry under one side of a candidate, with its free parameters.
#[derive(Clone, Copy)]
enum Geo<'a> {
    Point(Point3),
    /// A curve over its edge's range.
    Curve(&'a Curve, f64, f64),
    Surface(&'a Surface),
}

/// A point, its first derivatives by each free parameter, and its second.
type Jet = (Point3, [Vec3; 2], [[Vec3; 2]; 2]);

impl Geo<'_> {
    fn dim(&self) -> usize {
        match self {
            Geo::Point(_) => 0,
            Geo::Curve(..) => 1,
            Geo::Surface(_) => 2,
        }
    }

    fn eval(&self, x: &[f64]) -> Jet {
        let z = Vec3::ZERO;
        match *self {
            Geo::Point(p) => (p, [z; 2], [[z; 2]; 2]),
            Geo::Curve(c, ..) => {
                let (p, d, dd) = c.d2(x[0]);
                (p, [d, z], [[dd, z], [z, z]])
            }
            // The surface gives first derivatives; the second come from their central differences,
            // which only steer the search: the gradient it stops on is exact.
            Geo::Surface(s) => {
                let (u, v) = (x[0], x[1]);
                let (p, su, sv) = s.d1(u, v);
                let (hu, hv) = (1e-6 * (1.0 + u.abs()), 1e-6 * (1.0 + v.abs()));
                let (_, su1, sv1) = s.d1(u + hu, v);
                let (_, su0, sv0) = s.d1(u - hu, v);
                let (_, su2, sv2) = s.d1(u, v + hv);
                let (_, su3, sv3) = s.d1(u, v - hv);
                let suu = (su1 - su0) * (0.5 / hu);
                let svv = (sv2 - sv3) * (0.5 / hv);
                let suv = ((sv1 - sv0) * (0.5 / hu) + (su2 - su3) * (0.5 / hv)) * 0.5;
                (p, [su, sv], [[suu, suv], [suv, svv]])
            }
        }
    }

    /// Keeps a curve's parameter in its range.
    /// A curve's parameter kept in its edge's range: held at an end, or, on an edge that closes
    /// on itself, carried round to the other.
    fn clamp(&self, x: &mut [f64]) {
        if let Geo::Curve(_, lo, hi) = *self {
            x[0] = match self.closes() {
                Some(period) => lo + (x[0] - lo).rem_euclid(period),
                None => x[0].clamp(lo, hi),
            };
        }
    }

    /// The period of a curve whose range runs once round it, an edge whose two ends are one
    /// point: a foot just past that point is past neither end.
    fn closes(&self) -> Option<f64> {
        let Geo::Curve(c, lo, hi) = *self else {
            return None;
        };
        c.period().filter(|&w| hi - lo >= w * (1.0 - 1e-12))
    }

    /// The parameters of the point of the geometry nearest `p`, near it.
    fn seed(&self, p: Point3) -> [f64; 2] {
        match *self {
            Geo::Point(_) => [0.0; 2],
            Geo::Curve(c, lo, hi) => [c.project(p, lo, hi), 0.0],
            Geo::Surface(s) => {
                let uv = s.project(p);
                [uv.x, uv.y]
            }
        }
    }
}

/// A stationary point of the distance between `a` and `b` near the seeds, by Newton's method with
/// a damping that keeps each step downhill; the points and their distance.
fn descend(a: Geo, b: Geo, sa: [f64; 2], sb: [f64; 2]) -> (Point3, Point3, [f64; 4]) {
    let (da, db) = (a.dim(), b.dim());
    let n = da + db;
    let mut z = [sa[0], sa[1], sb[0], sb[1]];
    let split = |z: &[f64; 4]| ([z[0], z[1]], [z[2], z[3]]);
    let value = |z: &[f64; 4]| {
        let (xa, xb) = split(z);
        let (p, ..) = a.eval(&xa);
        let (q, ..) = b.eval(&xb);
        let r = p - q;
        0.5 * r.dot(r)
    };
    // Variables in order: a's first `da`, then b's first `db`, at their slots in `z`.
    let slot = |i: usize| if i < da { i } else { 2 + i - da };
    let mut damp = 0.0;
    for _ in 0..200 {
        let (xa, xb) = split(&z);
        let (p, pd, pdd) = a.eval(&xa);
        let (q, qd, qdd) = b.eval(&xb);
        let r = p - q;
        let f = 0.5 * r.dot(r);
        let col = |i: usize| if i < da { pd[i] } else { -qd[i - da] };
        let mut g = [0.0; 4];
        let mut h = [[0.0; 4]; 4];
        for i in 0..n {
            g[i] = r.dot(col(i));
            for j in 0..n {
                let second = match (i < da, j < da) {
                    (true, true) => r.dot(pdd[i][j]),
                    (false, false) => -r.dot(qdd[i - da][j - da]),
                    _ => 0.0,
                };
                h[i][j] = col(i).dot(col(j)) + second;
            }
        }
        // A curve parameter held at an end of its range by a gradient pushing past it stays.
        let mut free = [true; 4];
        for (i, f) in free.iter_mut().enumerate().take(n) {
            let geo = if i < da { a } else { b };
            if let Geo::Curve(_, lo, hi) = geo
                && geo.closes().is_none()
            {
                let x = z[slot(i)];
                if (x <= lo && g[i] > 0.0) || (x >= hi && g[i] < 0.0) {
                    *f = false;
                }
            }
        }
        let scale = (0..n).map(|i| h[i][i].abs()).fold(1e-300, f64::max);
        let mut improved = false;
        for _ in 0..30 {
            let Some(step) = solve(&h, &g, &free, n, damp * scale) else {
                damp = (damp * 10.0).max(1e-12);
                continue;
            };
            let mut next = z;
            for i in 0..n {
                next[slot(i)] -= step[i];
            }
            let (mut na, mut nb) = split(&next);
            a.clamp(&mut na);
            b.clamp(&mut nb);
            let next = [na[0], na[1], nb[0], nb[1]];
            if value(&next) < f {
                let moved = (0..4).map(|k| (next[k] - z[k]).abs()).fold(0.0, f64::max);
                z = next;
                damp *= 0.1;
                improved = moved > 1e-15 * (1.0 + z.iter().fold(0.0f64, |m, x| m.max(x.abs())));
                break;
            }
            damp = (damp * 10.0).max(1e-12);
        }
        if !improved {
            break;
        }
    }
    let (xa, xb) = split(&z);
    (a.eval(&xa).0, b.eval(&xb).0, z)
}

/// `(H + λ·I)⁻¹ g` over the free variables, by Cholesky; `None` where that is not positive
/// definite, which more damping cures.
#[allow(clippy::needless_range_loop)]
fn solve(
    h: &[[f64; 4]; 4],
    g: &[f64; 4],
    free: &[bool; 4],
    n: usize,
    lambda: f64,
) -> Option<[f64; 4]> {
    let idx: Vec<usize> = (0..n).filter(|&i| free[i]).collect();
    let m = idx.len();
    let mut out = [0.0; 4];
    if m == 0 {
        return Some(out);
    }
    let mut l = [[0.0; 4]; 4];
    for i in 0..m {
        for j in 0..=i {
            let mut s = h[idx[i]][idx[j]] + if i == j { lambda } else { 0.0 };
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = [0.0; 4];
    for i in 0..m {
        let mut s = g[idx[i]];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0.0; 4];
    for i in (0..m).rev() {
        let mut s = y[i];
        for k in i + 1..m {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    for i in 0..m {
        out[idx[i]] = x[i];
    }
    Some(out)
}

/// The closest points of two mesh pieces, and their distance.
fn piece_distance(a: &Piece, b: &Piece) -> (f64, Point3, Point3) {
    match (a, b) {
        (Piece::Pt(_, p), Piece::Pt(_, q)) => (p.dist(*q), *p, *q),
        (Piece::Pt(_, p), Piece::Seg(_, s)) => {
            let q = on_segment(*p, s[0], s[1]);
            (p.dist(q), *p, q)
        }
        (Piece::Seg(..), Piece::Pt(..)) => flip(piece_distance(b, a)),
        (Piece::Pt(_, p), Piece::Tri(_, t)) => {
            let q = on_triangle(*p, t);
            (p.dist(q), *p, q)
        }
        (Piece::Tri(..), Piece::Pt(..)) => flip(piece_distance(b, a)),
        (Piece::Seg(_, s), Piece::Seg(_, u)) => segments(s[0], s[1], u[0], u[1]),
        (Piece::Seg(_, s), Piece::Tri(_, t)) => segment_triangle(s[0], s[1], t),
        (Piece::Tri(..), Piece::Seg(..)) => flip(piece_distance(b, a)),
        (Piece::Tri(_, s), Piece::Tri(_, t)) => {
            let mut best = (f64::INFINITY, s[0], t[0]);
            for k in 0..3 {
                let x = segment_triangle(s[k], s[(k + 1) % 3], t);
                if x.0 < best.0 {
                    best = x;
                }
                let y = flip(segment_triangle(t[k], t[(k + 1) % 3], s));
                if y.0 < best.0 {
                    best = y;
                }
            }
            best
        }
    }
}

fn flip((d, p, q): (f64, Point3, Point3)) -> (f64, Point3, Point3) {
    (d, q, p)
}

fn offer(answer: &mut Option<Closest>, c: Closest) {
    if answer.is_none_or(|a| c.distance < a.distance) {
        *answer = Some(c);
    }
}

/// Seeds per pair of entities kept for the exact search, spread apart: a pair can have several
/// minima (a point on a torus's axis has a ring of them), and two large faces held parallel bring
/// thousands of pieces all equally close.
const SEEDS_PER_PAIR: usize = 32;

/// The closest distance between selection `sa` of `a` and `sb` of `b`, and a point of each at it.
pub fn closest(a: &Brep, sa: &Selection, b: &Brep, sb: &Selection) -> Result<Closest> {
    let (on_a, on_b) = (Extent::new(a, sa), Extent::new(b, sb));
    let (kept_a, kept_b) = (Domains::new(), Domains::new());
    // Solids that share a point are no distance apart, whatever else of them lies where.
    let held = on_b
        .holds_point(&on_a, &kept_b)
        .or_else(|| on_a.holds_point(&on_b, &kept_a));
    if let Some(p) = held {
        return Ok(Closest {
            distance: 0.0,
            on_a: p,
            on_b: p,
        });
    }
    between(
        &Side::meshed(on_a)?,
        &Side::meshed(on_b)?,
        [&kept_a, &kept_b],
    )
}

/// An arena prepared for asking which of its points is nearest a point: its faces, edges and
/// vertices, meshed once.
pub struct Nearest<'a> {
    side: Side<'a>,
}

impl<'a> Nearest<'a> {
    /// The arena's boundary alone: a point inside a solid is as far as its nearest wall.
    pub fn boundary(b: &'a Brep) -> Result<Nearest<'a>> {
        let faces = Selection {
            faces: b.face_ids().collect(),
            ..Selection::default()
        };
        Ok(Nearest {
            side: Side::new(b, &faces)?,
        })
    }

    /// The faces, edges, vertices and solids `selection` names: the nearest point of one face, say,
    /// wherever the rest of the arena lies.
    pub fn of(b: &'a Brep, selection: &Selection) -> Result<Nearest<'a>> {
        Ok(Nearest {
            side: Side::new(b, selection)?,
        })
    }

    /// The arena's solids, insides counted in: a point a solid holds is its own nearest point.
    pub fn solid(b: &'a Brep) -> Result<Nearest<'a>> {
        Ok(Nearest {
            side: Side::new(b, &Selection::body(b))?,
        })
    }

    /// The nearest point to `p`, as `on_b`, and its distance; `on_a` is `p`.
    pub fn to(&self, p: Point3) -> Result<Closest> {
        let mut point = Brep::new();
        let v = point.add_vertex(p);
        let side = Side::new(&point, &Selection::vertex(v))?;
        between(&side, &self.side, [&Domains::new(), &Domains::new()])
    }
}

/// [`closest`] between two sides already expanded and meshed, each with its solids' faces.
fn between(side_a: &Side, side_b: &Side, [kept_a, kept_b]: [&Domains; 2]) -> Result<Closest> {
    let (a, b) = (side_a.on.b, side_b.on.b);
    let (pa, pb) = (&side_a.mesh.pieces, &side_b.mesh.pieces);
    if pa.is_empty() || pb.is_empty() {
        return Err(crate::Error::InvalidInput(
            "a distance needs something on each side".into(),
        ));
    }
    let (tree_a, tree) = (&side_a.mesh.tree, &side_b.mesh.tree);
    let boxes = tree_a.boxes();

    // The closest pair of pieces.
    let best = tree_a.least(tree, |i, j| piece_distance(&pa[i], &pb[j]).0);
    let (defl_a, defl_b) = (side_a.mesh.deflection, side_b.mesh.deflection);
    let margin = 2.0 * (defl_a + defl_b) + tol::LINEAR;
    // Solids that share material are no distance apart. Boundaries farther apart than the meshes
    // can err meet nowhere, so only a solid holding the other can; nearer, a corner of one inside
    // the other shows the overlap before the search below, which would find it on the crossing.
    let apart = best > margin;
    let held = if apart {
        side_b
            .holds(side_a, kept_b)
            .or_else(|| side_a.holds(side_b, kept_a))
    } else {
        side_b
            .holds_corner(side_a, kept_b)
            .or_else(|| side_a.holds_corner(side_b, kept_a))
    };
    if let Some(p) = held {
        return Ok(Closest {
            distance: 0.0,
            on_a: p,
            on_b: p,
        });
    }
    // Every pair the exact minimum could lie between.
    let mut hits = Vec::new();
    let reach = best + margin;
    let spread = 4.0 * (defl_a + defl_b);
    // Nearest first, so the closest pair of pieces always seeds its entities: on parts smaller
    // than the meshes' deflection every pair lies within reach and spread of every other.
    let mut near_pairs = Vec::new();
    for (i, x) in pa.iter().enumerate() {
        tree.within(&boxes[i], reach, &mut hits);
        for &j in &hits {
            let (d, p, q) = piece_distance(x, &pb[j]);
            if d <= reach {
                near_pairs.push((d, (x.entity(), pb[j].entity()), p, q));
            }
        }
    }
    near_pairs.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
    let mut seeds: BTreeMap<(Entity, Entity), Vec<(Point3, Point3)>> = BTreeMap::new();
    for (_, pair, p, q) in near_pairs {
        let kept = seeds.entry(pair).or_default();
        let near = kept
            .iter()
            .any(|&(p0, q0)| p0.dist(p) <= spread && q0.dist(q) <= spread);
        if !near && kept.len() < SEEDS_PER_PAIR {
            kept.push((p, q));
        }
    }

    let mut domains: BTreeMap<(bool, FaceId), Domain> = BTreeMap::new();
    // Whether parameters `x` of entity `e` of a side lie on it: a curve's are kept in its range, a
    // surface's must fall inside its face, by the face's exact domain. A polygon of the face's
    // outline cuts inside a curved edge, and takes a point of a hole's opening for the face.
    let mut lies_on = |side: &Side, which: bool, e: Entity, x: [f64; 2]| match e {
        Entity::Face(f) => {
            let kept = if which { kept_b } else { kept_a };
            let dom = match side.on.domain(kept, f) {
                Some(dom) => dom,
                None => domains
                    .entry((which, f))
                    .or_insert_with(|| face_domain(side.on.b, f)),
            };
            let p = dom.surface.eval(x[0], x[1]);
            !matches!(dom.locate(p), Where::Outside)
        }
        _ => true,
    };
    let mut answer: Option<Closest> = None;
    'seeds: for ((ea, eb), list) in &seeds {
        let (ga, gb) = (side_a.geo(*ea), side_b.geo(*eb));
        for &(p, q) in list {
            if answer.is_some_and(|a| a.distance == 0.0) {
                break 'seeds;
            }
            let (x, y, z) = descend(ga, gb, ga.seed(p), gb.seed(q));
            if lies_on(side_a, false, *ea, [z[0], z[1]]) && lies_on(side_b, true, *eb, [z[2], z[3]])
            {
                offer(
                    &mut answer,
                    Closest {
                        distance: x.dist(y),
                        on_a: x,
                        on_b: y,
                    },
                );
            }
        }
    }
    // Every vertex pair is a real pair of points, so there is always an answer.
    if answer.is_none() {
        for &va in &side_a.on.vertices {
            for &vb in &side_b.on.vertices {
                let (x, y) = (a.vertex(va).point, b.vertex(vb).point);
                offer(
                    &mut answer,
                    Closest {
                        distance: x.dist(y),
                        on_a: x,
                        on_b: y,
                    },
                );
            }
        }
    }
    let mut answer = match answer {
        Some(c) => c,
        None => {
            let (_, x, y) = piece_distance(&pa[0], &pb[0]);
            Closest {
                distance: x.dist(y),
                on_a: x,
                on_b: y,
            }
        }
    };
    // Boundaries apart, however little, one solid may still hold the other.
    if !apart
        && answer.distance > tol::LINEAR
        && let Some(p) = side_b
            .holds(side_a, kept_b)
            .or_else(|| side_a.holds(side_b, kept_a))
    {
        answer = Closest {
            distance: 0.0,
            on_a: p,
            on_b: p,
        };
    }
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boolean, math::Frame, primitive};

    fn block(lo: [f64; 3], size: [f64; 3]) -> Brep {
        let frame = Frame::new(Vec3::from_array(lo), Vec3::Z, Vec3::X).unwrap();
        primitive::block(&frame, Vec3::from_array(size), "b").unwrap()
    }

    fn bodies(a: &Brep, b: &Brep) -> Closest {
        closest(a, &Selection::body(a), b, &Selection::body(b)).unwrap()
    }

    /// The distance by the search alone, with no point asked about before the sides are meshed.
    fn searched(a: &Brep, sa: &Selection, b: &Brep, sb: &Selection) -> Closest {
        let (side_a, side_b) = (Side::new(a, sa).unwrap(), Side::new(b, sb).unwrap());
        between(&side_a, &side_b, [&Domains::new(), &Domains::new()]).unwrap()
    }

    fn is_meshed(b: &Brep) -> bool {
        b.arena().seed.get().is_some()
    }

    #[test]
    fn a_corner_deep_in_the_other_solid_answers_before_either_is_meshed() {
        // Each block holds one corner of the other, 2 from its nearest faces.
        let a = block([0.0; 3], [4.0; 3]);
        let b = block([2.0; 3], [4.0; 3]);
        for (x, y) in [(&a, &b), (&b, &a)] {
            let c = bodies(x, y);
            assert_eq!(c.distance, 0.0);
            assert_eq!(c.on_a, c.on_b);
            let p = c.on_a;
            assert!((0..3).all(|k| p[k] >= 2.0 && p[k] <= 4.0), "{p:?}");
        }
        assert!(!is_meshed(&a) && !is_meshed(&b));
        let (sa, sb) = (Selection::body(&a), Selection::body(&b));
        assert_eq!(searched(&a, &sa, &b, &sb).distance, 0.0);
    }

    #[test]
    fn a_corner_within_the_tolerance_of_a_face_is_left_to_the_search() {
        // The upper block's bottom corners stand over the middle of the lower's top face: on it,
        // above it, or below it and no farther inside than the tolerance.
        let a = block([0.0; 3], [4.0; 3]);
        let hair = 10.0 * tol::LINEAR;
        for lift in [
            0.0,
            0.5 * tol::LINEAR,
            -0.5 * tol::LINEAR,
            -tol::LINEAR,
            hair,
        ] {
            let b = block([1.0, 1.0, 4.0 + lift], [2.0; 3]);
            let (sa, sb) = (Selection::body(&a), Selection::body(&b));
            assert_eq!(bodies(&a, &b), searched(&a, &sa, &b, &sb), "lift {lift}");
            assert_eq!(bodies(&b, &a), searched(&b, &sb, &a, &sa), "lift {lift}");
            assert!(is_meshed(&b), "lift {lift}");
        }
        let b = block([1.0, 1.0, 4.0 + hair], [2.0; 3]);
        let d = bodies(&a, &b).distance;
        assert!((d - hair).abs() <= 1e-12, "{d}");
    }

    #[test]
    fn a_block_in_a_void_is_as_far_as_the_voids_wall() {
        // A cube with a closed cavity 4 across, and a cube 2 across in the cavity's middle: the
        // inner cube's corners lie in the outer's box and in none of its material.
        let outer = boolean::difference(&block([0.0; 3], [10.0; 3]), &block([3.0; 3], [4.0; 3]));
        let (outer, inner) = (outer.unwrap(), block([4.0; 3], [2.0; 3]));
        for (x, y) in [(&outer, &inner), (&inner, &outer)] {
            let d = bodies(x, y).distance;
            assert!((d - 1.0).abs() <= 1e-9, "{d}");
        }
    }

    #[test]
    fn a_face_or_an_edge_reaching_into_a_solid_is_no_distance_from_it() {
        let a = block([0.0; 3], [4.0; 3]);
        let b = block([2.0; 3], [4.0; 3]);
        let whole = Selection::body(&b);
        let holds = |sel: &Selection| {
            let x = Extent::new(&a, sel);
            x.vertices
                .iter()
                .any(|&v| (0..3).all(|k| a.vertex(v).point[k] > 2.0))
        };
        let mut reached = 0;
        for f in a.face_ids() {
            let sel = Selection::face(f);
            let c = closest(&a, &sel, &b, &whole).unwrap();
            assert_eq!(c.distance, searched(&a, &sel, &b, &whole).distance, "{f:?}");
            assert_eq!(c.distance == 0.0, holds(&sel), "{f:?}");
            reached += usize::from(c.distance == 0.0);
        }
        // The three faces at the corner (4, 4, 4), which the other block holds.
        assert_eq!(reached, 3);
        for e in a.edge_ids() {
            let sel = Selection::edge(e);
            let c = closest(&a, &sel, &b, &whole).unwrap();
            assert_eq!(c.distance, searched(&a, &sel, &b, &whole).distance, "{e:?}");
            assert_eq!(c.distance == 0.0, holds(&sel), "{e:?}");
        }
    }

    #[test]
    fn the_pieces_of_every_solid_or_every_face_are_meshed_once() {
        let a = block([0.0; 3], [4.0; 3]);
        let mesh = |sel: &Selection| Side::new(&a, sel).unwrap().mesh;
        let body = Selection::body(&a);
        assert!(Arc::ptr_eq(&mesh(&body), &mesh(&body)));
        let faces = Selection::faces(a.face_ids());
        assert!(Arc::ptr_eq(&mesh(&faces), &mesh(&faces)));
        assert!(!Arc::ptr_eq(&mesh(&body), &mesh(&faces)));
        // A copy is the same solid, and a moved copy is not.
        let mut copy = a.clone();
        let of = |b: &Brep| Side::new(b, &Selection::body(b)).unwrap().mesh;
        assert!(Arc::ptr_eq(&of(&copy), &mesh(&body)));
        copy.transform(&crate::math::Transform::translation(Vec3::new(
            10.0, 0.0, 0.0,
        )));
        assert!(!Arc::ptr_eq(&of(&copy), &mesh(&body)));
        assert_eq!(of(&copy).bbox.min.x, 10.0);
        assert_eq!(mesh(&body).bbox.min.x, 0.0);
        // One face of six is its own selection, with its own pieces.
        let one = Selection::face(a.face_ids().next().unwrap());
        assert!(mesh(&one).pieces.len() < mesh(&faces).pieces.len());
        assert!(!Arc::ptr_eq(&mesh(&one), &mesh(&one)));
    }
}
