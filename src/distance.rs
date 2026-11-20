//! The closest distance between two sets of faces, edges and vertices, from the exact geometry.
//!
//! Both sides are meshed, and the closest pair of mesh pieces gives the distance to within the two
//! meshes' deflections. Every pair of pieces that close to it seeds a Newton search for a
//! stationary point of the exact distance between the surfaces, curves and points they come from,
//! and a result counts only where both its points lie on their face or edge. The true minimum lies
//! in the basin of one of those seeds, so the least result is exact. Where a surface's nearest
//! point lies outside its face, the face's minimum is on its boundary, which its edges' pieces
//! cover. Whether a point lies on its face, and whether one solid holds another, are asked of the
//! exact faces, as the booleans ask them: the meshes only seed the search. Solids one inside the
//! other are at distance zero.

use std::collections::BTreeMap;

use crate::{
    boolean::{
        classify,
        domain::{Domain, Where},
        face_domain,
    },
    error::Result,
    geom::{Curve, Surface},
    math::{Aabb, Point3, Vec3},
    mesh::{
        self, MeshOptions,
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

    pub fn edge(e: EdgeId) -> Selection {
        Selection {
            edges: vec![e],
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

/// One side, expanded: its vertices, its mesh pieces, and its solids' meshes.
struct Side<'a> {
    b: &'a Brep,
    vertices: Vec<VertexId>,
    pieces: Vec<Piece>,
    /// The faces of the selected solids, for whether a point lies inside one.
    solid_faces: Vec<FaceId>,
    deflection: f64,
}

/// The meshes tried, finest first: the mesher gives up on a very fine chord over a large part.
const MESHES: [MeshOptions; 3] = [
    MeshOptions {
        linear: 0.01,
        angular: 0.2,
    },
    MeshOptions {
        linear: 0.05,
        angular: 0.5,
    },
    MeshOptions {
        linear: 0.2,
        angular: 0.8,
    },
];

impl<'a> Side<'a> {
    fn new(b: &'a Brep, sel: &Selection) -> Result<Side<'a>> {
        let mut faces = sel.faces.clone();
        for &s in &sel.solids {
            faces.extend(b.solid_faces(s));
        }
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

        let mut err = None;
        let mut found = None;
        for opt in MESHES.iter().map(|o| mesh::seeding(o, b)) {
            match mesh::tessellate(b, &opt) {
                Ok(m) => {
                    found = Some((m, opt));
                    break;
                }
                Err(e) => err = Some(e),
            }
        }
        let Some((m, opt)) = found else {
            return Err(err.expect("a mesh was tried"));
        };
        let tri = |t: &[u32; 3]| t.map(|i| m.positions[i as usize]);
        let wanted: BTreeMap<FaceId, ()> = faces.iter().map(|&f| (f, ())).collect();
        let solid_faces: BTreeMap<FaceId, ()> = sel
            .solids
            .iter()
            .flat_map(|&s| b.solid_faces(s))
            .map(|f| (f, ()))
            .collect();
        let mut pieces = Vec::new();
        for (f, range) in &m.faces {
            for t in &m.triangles[range.clone()] {
                if wanted.contains_key(f) {
                    pieces.push(Piece::Tri(*f, tri(t)));
                }
            }
        }
        for &e in &edges {
            let edge = b.edge(e);
            let Some(c) = &edge.curve else { continue };
            let (t0, t1) = edge.range;
            let ts = mesh::edge_params(c, t0, t1, &opt);
            for w in ts.windows(2) {
                pieces.push(Piece::Seg(e, [c.eval(w[0]), c.eval(w[1])]));
            }
        }
        for &v in &vertices {
            pieces.push(Piece::Pt(v, b.vertex(v).point));
        }
        Ok(Side {
            b,
            vertices,
            pieces,
            solid_faces: solid_faces.into_keys().collect(),
            deflection: opt.linear,
        })
    }

    /// The exact geometry an entity lies on.
    fn geo(&self, e: Entity) -> Geo<'a> {
        match e {
            Entity::Face(f) => Geo::Surface(&self.b.face(f).surface),
            Entity::Edge(id) => {
                let edge = self.b.edge(id);
                let (t0, t1) = edge.range;
                match &edge.curve {
                    Some(c) => Geo::Curve(c, t0.min(t1), t0.max(t1)),
                    None => Geo::Point(self.b.vertex(edge.start).point),
                }
            }
            Entity::Vertex(v) => Geo::Point(self.b.vertex(v).point),
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
    fn holds(&self, inner: &Side) -> Option<Point3> {
        if self.solid_faces.is_empty() {
            return None;
        }
        // Only a solid whose box holds the other's can hold it.
        let boxed = |ps: &[Piece]| {
            let mut bb = Aabb::EMPTY;
            ps.iter().for_each(|p| bb.union(&p.bbox()));
            bb
        };
        let (outer, within) = (boxed(&self.pieces), boxed(&inner.pieces));
        let pad = self.deflection + inner.deflection;
        if (0..3).any(|k| within.min[k] < outer.min[k] - pad || within.max[k] > outer.max[k] + pad)
        {
            return None;
        }
        let pad = Vec3::new(tol::LINEAR, tol::LINEAR, tol::LINEAR);
        let faces: Vec<(Domain, Aabb)> = self
            .solid_faces
            .iter()
            .map(|&f| {
                let bb = crate::measure::face_box(self.b, f);
                let padded = Aabb {
                    min: bb.min - pad,
                    max: bb.max + pad,
                };
                (face_domain(self.b, f), padded)
            })
            .collect();
        let faces: Vec<(&Domain, Aabb)> = faces.iter().map(|(d, bb)| (d, *bb)).collect();
        let corners = inner.pieces.iter().flat_map(|p| match *p {
            Piece::Tri(_, t) => t.to_vec(),
            Piece::Seg(_, s) => s.to_vec(),
            Piece::Pt(_, q) => vec![q],
        });
        // A corner no ray decides lies on this side's boundary within rounding: the next is asked.
        corners
            .take(16)
            .find_map(|p| classify::contains(&faces, p).map(|inside| inside.then_some(p)))
            .flatten()
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
    between(&Side::new(a, sa)?, &Side::new(b, sb)?)
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
        between(&Side::new(&point, &Selection::vertex(v))?, &self.side)
    }
}

/// [`closest`] between two sides already expanded.
fn between(side_a: &Side, side_b: &Side) -> Result<Closest> {
    let (a, b) = (side_a.b, side_b.b);
    let (pa, pb) = (&side_a.pieces, &side_b.pieces);
    if pa.is_empty() || pb.is_empty() {
        return Err(crate::Error::InvalidInput(
            "a distance needs something on each side".into(),
        ));
    }
    let tree = Bvh::of_boxes(pb.iter().map(Piece::bbox).collect());
    let boxes: Vec<Aabb> = pa.iter().map(Piece::bbox).collect();

    // The closest pair of pieces, each query widened by the best so far.
    let mut best = piece_distance(&pa[0], &pb[0]).0;
    let mut hits = Vec::new();
    for (i, x) in pa.iter().enumerate() {
        tree.within(&boxes[i], best, &mut hits);
        for &j in &hits {
            best = best.min(piece_distance(x, &pb[j]).0);
        }
    }
    // Every pair the exact minimum could lie between.
    let margin = 2.0 * (side_a.deflection + side_b.deflection) + tol::LINEAR;
    let reach = best + margin;
    let spread = 4.0 * (side_a.deflection + side_b.deflection);
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
            let dom = domains
                .entry((which, f))
                .or_insert_with(|| face_domain(side.b, f));
            let p = dom.surface.eval(x[0], x[1]);
            !matches!(dom.locate(p), Where::Outside)
        }
        _ => true,
    };
    let mut answer: Option<Closest> = None;
    for ((ea, eb), list) in &seeds {
        let (ga, gb) = (side_a.geo(*ea), side_b.geo(*eb));
        for &(p, q) in list {
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
        for &va in &side_a.vertices {
            for &vb in &side_b.vertices {
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
    if answer.distance > tol::LINEAR
        && let Some(p) = side_b.holds(side_a).or_else(|| side_a.holds(side_b))
    {
        answer = Closest {
            distance: 0.0,
            on_a: p,
            on_b: p,
        };
    }
    Ok(answer)
}
