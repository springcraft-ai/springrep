//! Boolean union, difference and intersection of solids.
//!
//! The steps, each in its own module:
//!
//! 1. Every edge of one solid is intersected with every face of the other (`csi`), coplanar edges
//!    with each other, and every face pair's surfaces with each other (`ssi`, `march`). Each
//!    meeting point becomes a vertex, shared by every curve through it.
//! 2. Every edge and intersection curve is split at every vertex on it, and an edge lying on a face
//!    of the other solid cuts it. Pieces that run along each other are one edge, so both faces of
//!    every edge use the same one.
//! 3. Each face is split by the intersection pieces inside it, and by the boundary of any face of
//!    the other solid that lies on its surface (`arrange`).
//! 4. Each piece of face is classified inside, outside, or on the other solid (`classify`); the
//!    operation selects pieces, and `sew` joins them, merges faces on one surface, and checks the
//!    result.

pub(crate) mod arrange;
pub(crate) mod classify;
mod contact;
pub(crate) mod csi;
pub(crate) mod domain;
mod hulls;
mod march;
mod near;
mod plane;
mod same;
pub(crate) mod sew;
mod simplify;
mod snap;
pub(crate) mod ssi;
mod unify;

pub use plane::{half_space, split};
pub use ssi::same_domain;
pub use unify::unify;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;
use std::rc::Rc;

use self::{
    domain::{Domain, Trim, UvMap, Where},
    sew::{OutFace, PieceGeom},
    ssi::Ssi,
};
use crate::{
    Error, Result,
    check::coedge_params,
    geom::{Curve, Surface},
    math::{Aabb, Frame, Point3, Vec2, Vec3},
    measure,
    mesh::bvh::Bvh,
    tol,
    topo::{
        Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, FaceName, Facts, OnFace, PCurve, SolidId,
        VertexId,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Union,
    Difference,
    Intersection,
}

pub fn union(a: &Brep, b: &Brep) -> Result<Brep> {
    boolean(a, b, Op::Union)
}

/// `a` with `b` taken away.
pub fn difference(a: &Brep, b: &Brep) -> Result<Brep> {
    boolean(a, b, Op::Difference)
}

pub fn intersection(a: &Brep, b: &Brep) -> Result<Brep> {
    boolean(a, b, Op::Intersection)
}

/// The operands sorted into groups whose members' boxes are apart from each other, each operand
/// by its place in `parts`. Solids whose boxes are apart share no point, so a group in one arena
/// is its own union, and one boolean combines the whole group with another solid.
pub fn apart_groups(parts: &[&Brep]) -> Vec<Vec<usize>> {
    let boxes: Vec<Aabb> = parts
        .iter()
        .map(|p| padded(&measure::bbox(p), tol::LINEAR))
        .collect();
    let meet = |a: &Aabb, b: &Aabb| {
        (0..3).all(|k| a.min.to_array()[k] <= b.max.to_array()[k])
            && (0..3).all(|k| b.min.to_array()[k] <= a.max.to_array()[k])
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, bx) in boxes.iter().enumerate() {
        if bx.is_empty() {
            continue;
        }
        match groups
            .iter_mut()
            .find(|g| g.iter().all(|&j| !meet(bx, &boxes[j])))
        {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    groups
}

/// One arena holding every solid of the operands named.
fn gathered(parts: &[&Brep], group: &[usize]) -> Brep {
    let mut out = Brep::new();
    for &i in group {
        out.append(parts[i]);
    }
    out
}

/// The union of every operand. Operands apart from each other are gathered and combined in one
/// boolean, so a pattern costs as many booleans as its overlaps need, not one for each solid.
pub fn union_all(parts: &[&Brep]) -> Result<Brep> {
    let mut groups = apart_groups(parts).into_iter();
    let Some(first) = groups.next() else {
        return Ok(Brep::new());
    };
    let mut out = gathered(parts, &first);
    for g in groups {
        out = union(&out, &gathered(parts, &g))?;
    }
    Ok(out)
}

/// `target` with every tool taken away. Tools apart from each other are gathered and cut in one
/// boolean, so a drilled pattern costs as many booleans as its overlaps need, not one for each
/// hole.
pub fn difference_all(target: &Brep, tools: &[&Brep]) -> Result<Brep> {
    let mut out = target.clone();
    for g in apart_groups(tools) {
        out = difference(&out, &gathered(tools, &g))?;
    }
    Ok(out)
}

/// Combines every solid of `a` with every solid of `b`. The result passes [`crate::check`], or the
/// operation returns [`Error::InvalidResult`]. Pieces of the result that touch, along a curve or at
/// a point, are one solid: the curve an edge four faces use, the point a vertex two fans meet at.
/// A result with no material, such as a solid minus itself or the common
/// part of disjoint solids, is an empty arena: no solids, which callers must check for.
pub fn boolean(a: &Brep, b: &Brep, op: Op) -> Result<Brep> {
    boolean_with(a, b, op, &Facts::default())
}

/// [`boolean`], taking `facts` the caller's construction makes true instead of computing them. A
/// fact that does not hold is [`Error::InvalidInput`]. Facts name the inputs' own ids; where some
/// solids pass through untouched, the facts about the rest follow them into the copies combined.
pub(crate) fn boolean_with(a: &Brep, b: &Brep, op: Op, facts: &Facts) -> Result<Brep> {
    Ok(simplify::merge_edges(combined(a, b, op, facts)?))
}

/// [`boolean_with`] before redundant vertices are dissolved.
fn combined(a: &Brep, b: &Brep, op: Op, facts: &Facts) -> Result<Brep> {
    // Facts name `b`'s own faces and edges, which a snapped copy would renumber.
    if !facts.on_face.is_empty() {
        return combined_as(a, b, op, facts).and_then(|out| not_lost(a, b, op, out));
    }
    let snapped = snap::snapped(a, b, snap::Reach::Exact);
    let used = snapped.as_ref().unwrap_or(b);
    let first = combined_as(a, used, op, facts).and_then(|out| not_lost(a, used, op, out));
    if first.is_ok() {
        return first;
    }
    // Shapes a hair off each other that the boolean could not place, retried with them on one
    // surface: the answer then differs from the exact one by at most the gap times their area.
    match snap::snapped(a, b, snap::Reach::Grazing) {
        Some(wide) => combined_as(a, &wide, op, facts)
            .and_then(|out| not_lost(a, &wide, op, out))
            .or(first),
        None => first,
    }
}

/// `out`, the result of `op` on `a` and `b`, unless it is empty where the two share material.
fn not_lost(a: &Brep, b: &Brep, op: Op, out: Brep) -> Result<Brep> {
    if out.solid_ids().next().is_some() || !classify::material_lost(a, b, op) {
        return Ok(out);
    }
    Err(Error::Numerical(
        "the result is empty, but a point of one solid lies well inside the other: an \
         intersection was missed"
            .into(),
    ))
}

fn combined_as(a: &Brep, b: &Brep, op: Op, facts: &Facts) -> Result<Brep> {
    let (ba, bb) = (SolidBoxes::new(a), SolidBoxes::new(b));
    if let Some(out) = without_coincident(a, &ba, b, &bb, op, facts)? {
        return Ok(out);
    }
    let near = |own: &SolidBoxes, others: &SolidBoxes| -> Vec<bool> {
        (0..own.ids.len())
            .map(|i| (0..others.ids.len()).any(|j| own.meet(i, others, j)))
            .collect()
    };
    let (na, nb) = (near(&ba, &bb), near(&bb, &ba));
    if na.iter().chain(&nb).all(|&n| n) {
        return combine(a, b, op, facts);
    }
    let pick = |boxes: &SolidBoxes, n: &[bool], want: bool| -> Vec<SolidId> {
        boxes
            .ids
            .iter()
            .zip(n)
            .filter(|&(_, &k)| k == want)
            .map(|(&s, _)| s)
            .collect()
    };
    // Solids whose boxes meet nothing of the other input pass through untouched.
    let (near_a, near_b) = (pick(&ba, &na, true), pick(&bb, &nb, true));
    let mut out = if near_a.is_empty() {
        Brep::new()
    } else {
        let (sa, sb) = (copy_solids(a, &near_a), copy_solids(b, &near_b));
        combine(&sa.0, &sb.0, op, &follow(facts, Some(&sa), Some(&sb)))?
    };
    match op {
        Op::Union => {
            out.append(&solids_of(a, &pick(&ba, &na, false)));
            out.append(&solids_of(b, &pick(&bb, &nb, false)));
        }
        Op::Difference => {
            out.append(&solids_of(a, &pick(&ba, &na, false)));
        }
        Op::Intersection => {}
    }
    Ok(out)
}

/// The boolean where some solid of `a` and some solid of `b` are one solid within their tolerances
/// ([`same::same_solid`]), from the rest: such a pair is its own union and intersection, and takes
/// all of itself from a difference. `None` when no solid of `a` is one of `b`'s.
fn without_coincident(
    a: &Brep,
    ba: &SolidBoxes,
    b: &Brep,
    bb: &SolidBoxes,
    op: Op,
    facts: &Facts,
) -> Result<Option<Brep>> {
    let widest = |x: &Brep| {
        let (vs, es) = x.tolerant();
        vs.iter()
            .map(|t| t.1)
            .chain(es.iter().map(|t| t.1))
            .fold(tol::LINEAR, f64::max)
    };
    // A copy's box moves with it; this only keeps the topology comparisons to likely pairs.
    let reach = 2.0 * (widest(a) + widest(b)) + 1e-6;
    let close = |x: &Aabb, y: &Aabb| {
        (0..3).all(|k| (x.min[k] - y.min[k]).abs() <= reach && (x.max[k] - y.max[k]).abs() <= reach)
    };
    let mut taken = vec![false; bb.ids.len()];
    let mut matched = vec![false; ba.ids.len()];
    for (i, sa) in ba.ids.iter().enumerate() {
        let mut one: Option<Brep> = None;
        for (j, sb) in bb.ids.iter().enumerate() {
            // Boxes that close overlap, so solids whose cheap boxes miss cannot match.
            if taken[j]
                || !overlap(&ba.hulls[i], &bb.hulls[j])
                || !close(&ba.loose(i), &bb.loose(j))
            {
                continue;
            }
            let x = one.get_or_insert_with(|| solids_of(a, &[*sa]));
            if same::same_solid(x, &solids_of(b, &[*sb])) {
                taken[j] = true;
                matched[i] = true;
                break;
            }
        }
    }
    if !matched.contains(&true) {
        return Ok(None);
    }
    let pick = |boxes: &SolidBoxes, flags: &[bool], want: bool| -> Vec<SolidId> {
        boxes
            .ids
            .iter()
            .zip(flags)
            .filter(|&(_, &f)| f == want)
            .map(|(&s, _)| s)
            .collect()
    };
    let rest_a = pick(ba, &matched, false);
    let rest_b = pick(bb, &taken, false);
    Ok(Some(match op {
        Op::Union if rest_b.is_empty() => a.clone(),
        Op::Union => {
            let cb = copy_solids(b, &rest_b);
            boolean_with(a, &cb.0, op, &follow(facts, None, Some(&cb)))?
        }
        Op::Intersection => {
            let mut out = if rest_a.is_empty() || rest_b.is_empty() {
                Brep::new()
            } else {
                let (ca, cb) = (copy_solids(a, &rest_a), copy_solids(b, &rest_b));
                boolean_with(&ca.0, &cb.0, op, &follow(facts, Some(&ca), Some(&cb)))?
            };
            out.append(&solids_of(a, &pick(ba, &matched, true)));
            out
        }
        // The matched solids of `b` stay: they may overlap the rest of `a` where `a`'s own
        // solids overlap.
        Op::Difference if rest_a.is_empty() => Brep::new(),
        Op::Difference => {
            let ca = copy_solids(a, &rest_a);
            boolean_with(&ca.0, b, op, &follow(facts, Some(&ca), None))?
        }
    }))
}

/// `facts` about the inputs restated for copies of some of their solids, `None` for an input
/// passed whole. The copies number their edges and faces afresh, and a fact about a solid left out
/// has nothing left to say.
fn follow(facts: &Facts, a: Option<&Copied>, b: Option<&Copied>) -> Facts {
    let edge = |c: Option<&Copied>, e: EdgeId| c.map_or(Some(e), |c| c.1.get(&e).copied());
    let face = |c: Option<&Copied>, f: FaceId| c.map_or(Some(f), |c| c.2.get(&f).copied());
    let on_face = facts
        .on_face
        .iter()
        .filter_map(|f| {
            let (own, other) = if f.side == 0 { (a, b) } else { (b, a) };
            Some(OnFace {
                side: f.side,
                edge: edge(own, f.edge)?,
                face: face(other, f.face)?,
            })
        })
        .collect();
    Facts { on_face }
}

/// How far from `q`, a point where surfaces `a` and `b` meet along `dir`, a curve lying within `t`
/// of one and [`tol::LINEAR`] of the other may stray and still be their meeting: where they cross
/// at an angle whose sine is s, either one moved by its tolerance moves the crossing by that over
/// s; where they are tangent, the band either side where they stay within the tolerance.
fn crossing_spread(a: &Surface, b: &Surface, q: Point3, dir: Vec3, t: f64) -> f64 {
    let (ua, ub) = (a.project(q), b.project(q));
    let sine = a.normal(ua.x, ua.y).cross(b.normal(ub.x, ub.y)).norm();
    if sine > tol::TANGENT {
        (t + tol::LINEAR) / sine
    } else {
        contact::band(a, b, q, dir)
    }
}

/// Whether `c` stays within `t` of surface `s` over `range`, at 33 points: a tolerant edge lying
/// on a face of the other solid within its own tolerance.
fn lies_within(c: &Curve, range: (f64, f64), s: &Surface, t: f64) -> bool {
    (0..=32).all(|k| {
        let p = c.eval(range.0 + (range.1 - range.0) * k as f64 / 32.0);
        csi::signed_distance(s, p).abs() <= t
    })
}

/// Each solid's box, widened by the tolerance so solids within it of each other overlap.
/// Each solid of an arena with a box that holds it, from [`measure::face_hull_box`], and its loose
/// box, worked out only where asked: the decisions read loose boxes, and solids whose hull boxes
/// miss each other have loose boxes that miss too.
struct SolidBoxes<'a> {
    x: &'a Brep,
    ids: Vec<SolidId>,
    hulls: Vec<Aabb>,
    loose: Vec<std::cell::OnceCell<Aabb>>,
}

impl<'a> SolidBoxes<'a> {
    fn new(x: &'a Brep) -> SolidBoxes<'a> {
        let ids: Vec<SolidId> = x.solid_ids().collect();
        let hulls = ids
            .iter()
            .map(|&s| {
                let mut bb = Aabb::EMPTY;
                for f in x.solid_faces(s) {
                    bb.union(&measure::face_hull_box(x, f));
                }
                padded(&bb, tol::LINEAR)
            })
            .collect();
        SolidBoxes {
            x,
            loose: vec![std::cell::OnceCell::new(); ids.len()],
            ids,
            hulls,
        }
    }

    fn loose(&self, i: usize) -> Aabb {
        *self.loose[i].get_or_init(|| {
            let bb = if self.ids.len() == 1 {
                measure::loose_bbox(self.x)
            } else {
                measure::loose_bbox(&solids_of(self.x, &[self.ids[i]]))
            };
            padded(&bb, tol::LINEAR)
        })
    }

    /// Whether solid `i` of these and solid `j` of `other` have loose boxes that meet.
    fn meet(&self, i: usize, other: &SolidBoxes, j: usize) -> bool {
        overlap(&self.hulls[i], &other.hulls[j]) && overlap(&self.loose(i), &other.loose(j))
    }
}

/// A new arena holding copies of `solids` of `x`: one piece of a result that came apart, say, to
/// measure or cut on its own.
pub fn solids_of(x: &Brep, solids: &[SolidId]) -> Brep {
    copy_solids(x, solids).0
}

type Copied = (Brep, HashMap<EdgeId, EdgeId>, HashMap<FaceId, FaceId>);

/// [`solids_of`], with the copy's id of each edge and face copied.
fn copy_solids(x: &Brep, solids: &[SolidId]) -> Copied {
    let mut out = Brep::new();
    let mut verts: HashMap<VertexId, VertexId> = HashMap::new();
    let mut edges: HashMap<EdgeId, EdgeId> = HashMap::new();
    let mut face_ids: HashMap<FaceId, FaceId> = HashMap::new();
    for &s in solids {
        let mut shells = Vec::new();
        for &sh in &x.solid(s).shells {
            let mut faces = Vec::new();
            for &f in &x.shell(sh).faces {
                let face = x.face(f);
                let nf = out.add_face(face.surface.clone(), face.reversed, face.names.clone());
                for &lp in &face.loops {
                    let mut specs = Vec::new();
                    for &c in &x.lp(lp).coedges {
                        let co = x.coedge(c);
                        let e = match edges.get(&co.edge) {
                            Some(&e) => e,
                            None => {
                                let edge = x.edge(co.edge);
                                let mut v = |id: VertexId| {
                                    *verts.entry(id).or_insert_with(|| {
                                        let nv = out.add_vertex(x.vertex(id).point);
                                        out.set_vertex_tolerance(nv, x.vertex_tolerance(id));
                                        nv
                                    })
                                };
                                let (v0, v1) = (v(edge.start), v(edge.end));
                                let e = match &edge.curve {
                                    Some(cv) => out.add_edge(cv.clone(), edge.range, v0, v1),
                                    None => out.add_degenerate_edge(v0, edge.range),
                                };
                                out.set_edge_tolerance(e, x.edge_tolerance(co.edge));
                                edges.insert(co.edge, e);
                                e
                            }
                        };
                        specs.push(CoedgeSpec {
                            edge: e,
                            reversed: co.reversed,
                            pcurve: co.pcurve.clone(),
                        });
                    }
                    out.add_loop(nf, specs);
                }
                out.share_memo(nf, x.shared_memo(f));
                face_ids.insert(f, nf);
                faces.push(nf);
            }
            shells.push(out.add_shell(faces));
        }
        out.add_solid_of(shells, x.solid(s).outer);
    }
    out.pair_radial();
    (out, edges, face_ids)
}

fn combine(a: &Brep, b: &Brep, op: Op, facts: &Facts) -> Result<Brep> {
    let mut w = Work::new([a, b]);
    if w.many_pairs() {
        w.find_near();
    }
    w.finish(op, facts)
}

/// What a boolean reads from an input face's geometry, kept in the face's memo: each coedge's
/// whole trim (parameter range and parameter-plane ends) and its ends as one piece of the
/// boundary, the domain's samples and boxes, and the face's box.
#[derive(Debug)]
pub(crate) struct FaceImages {
    trims: Vec<Vec<TrimImage>>,
    samples: Vec<Vec<Vec<Vec2>>>,
    boxes: Vec<Vec<Aabb>>,
    bbox: Aabb,
}

#[derive(Debug)]
struct TrimImage {
    t: (f64, f64),
    uv: (Vec2, Vec2),
    whole: (Vec2, Vec2),
}

/// A face of either input.
struct Face {
    side: usize,
    id: FaceId,
    dom: Domain,
    bbox: Aabb,
}

/// A curve the result's edges are cut from: an input edge, or a face pair's intersection.
struct Base {
    curve: Option<Rc<Curve>>,
    range: (f64, f64),
    v0: Option<usize>,
    v1: Option<usize>,
    pole: Point3,
    closed: bool,
    bbox: Aabb,
    kind: BaseKind,
    /// The input edge's local tolerance, or [`tol::LINEAR`] for an intersection curve.
    tol: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BaseKind {
    Edge,
    /// The intersection of faces `fa` and `fb`.
    Cut {
        fa: usize,
        fb: usize,
        tangent: bool,
    },
}

/// A base's stretch between two adjacent vertices on it.
#[derive(Clone, Debug)]
struct Piece {
    base: usize,
    range: (f64, f64),
    v0: usize,
    v1: usize,
    /// Which of coincident pieces stands for them all: input edges before intersection curves.
    rank: u8,
}

struct Work<'a> {
    breps: [&'a Brep; 2],
    pool: Vec<Point3>,
    /// Each pool vertex's tolerance: an input vertex's own, [`tol::LINEAR`] for a meeting, and a
    /// tolerant edge's for a meeting on it, grown where it takes in another.
    vtol: Vec<f64>,
    /// The widest of `vtol`, which bounds how far apart two pool vertices can coincide.
    widest: f64,
    /// The pool vertices whose tolerance is above [`tol::LINEAR`]: few, and searched one by one.
    tolerant: Vec<usize>,
    /// The widest tolerance of the inputs, which no tolerance grows past.
    cap: f64,
    grid: HashMap<[i64; 3], Vec<usize>>,
    vmap: [Vec<usize>; 2],
    faces: Vec<Face>,
    face_of: [Vec<usize>; 2],
    bases: Vec<Base>,
    edge_base: [Vec<usize>; 2],
    coincident: Vec<(usize, usize)>,
    pieces: Vec<Piece>,
    base_pieces: Vec<Vec<usize>>,
    /// Faces each intersection piece crosses the inside of.
    cut_faces: Vec<Vec<usize>>,
    canon: Vec<usize>,
    /// Where each edge base meets a face of the other solid, from step 1a, before they are placed.
    hits: Vec<Hit>,
    /// Each edge base that lies in the surface of a face of the other solid, with that face.
    on_face: BTreeSet<(usize, usize)>,
    /// How many vertices of the pool are the inputs' own.
    inputs: usize,
    /// Each placed meeting where an edge runs tangent to a face: the vertex, the edge base and the
    /// face.
    met: Vec<(usize, usize, usize)>,
    /// The vertices where two faces' surfaces meet at an isolated point, as two balls touch.
    touch_points: Vec<usize>,
    /// Vertices a hit made where it was found, on no intersection curve: anywhere in the band
    /// where its edge grazes the face, as far from where they stand as the meeting fixes them.
    unplaced: HashMap<usize, f64>,
    /// Each vertex an edge of one face met another face at: the two faces, and how far along the
    /// graze the meeting may lie from where their intersection curve runs.
    meetings: HashMap<usize, Vec<(usize, usize, f64)>>,
    /// Which faces and edges of the two inputs come near each other; every pair where unknown.
    near: Option<near::Near>,
}

/// How many face pairs, or edge pairs, with overlapping boxes make [`near::Near`] worth building.
const MANY_FACE_PAIRS: usize = 256;
const MANY_EDGE_PAIRS: usize = 4096;

/// Where edge base `base` meets face `face` of the other solid: at `p`, its parameter `t`, crossing
/// at an angle whose sine is `sine`.
#[derive(Clone, Copy, Debug)]
struct Hit {
    p: Point3,
    t: f64,
    base: usize,
    face: usize,
    sine: f64,
}

const CELL: f64 = 1e-3;

/// The furthest a grazing meeting is taken to lie from the intersection curve it belongs to.
const GRAZE_REACH: f64 = 100.0 * tol::LINEAR;

/// The sine between a piece's way in from a cut and the other face's normal below which the cut
/// does not settle the piece's side. It is a guard against rounding, not a tolerance: where the
/// surfaces cross more shallowly a ray decides.
const LOCAL_SINE: f64 = 1e-3;

fn cell(p: Point3) -> [i64; 3] {
    [
        (p.x / CELL).floor() as i64,
        (p.y / CELL).floor() as i64,
        (p.z / CELL).floor() as i64,
    ]
}

fn overlap(a: &Aabb, b: &Aabb) -> bool {
    !a.is_empty() && !b.is_empty() && (0..3).all(|k| a.min[k] <= b.max[k] && b.min[k] <= a.max[k])
}

fn padded(b: &Aabb, pad: f64) -> Aabb {
    Aabb {
        min: b.min - Vec3::new(pad, pad, pad),
        max: b.max + Vec3::new(pad, pad, pad),
    }
}

fn curve_bbox(c: &Curve, range: (f64, f64)) -> Aabb {
    c.bbox(range.0, range.1)
}

/// Adds to `bb` every point of a sphere or torus about `frame` inside the face's parameter window
/// where a world coordinate is stationary: a face's extremes lie on its boundary or at those. The
/// window is widened by a tenth, since it is the box of the boundary's sampled images.
fn stationary_points(s: &Surface, frame: &Frame, dom: &Domain, bb: &mut Aabb) {
    let pad = (dom.hi - dom.lo) * 0.1;
    let (lo, hi) = (dom.lo - pad, dom.hi + pad);
    let images = |x: f64, lo: f64, hi: f64, period: Option<f64>| -> Vec<f64> {
        let Some(p) = period else {
            return if (lo..=hi).contains(&x) {
                vec![x]
            } else {
                Vec::new()
            };
        };
        let mut c = x + ((lo - x) / p).ceil() * p;
        let mut out = Vec::new();
        while c <= hi {
            out.push(c);
            c += p;
        }
        out
    };
    for d in [Vec3::X, Vec3::Y, Vec3::Z] {
        let (dx, dy, dz) = (d.dot(frame.x), d.dot(frame.y), d.dot(frame.z));
        // Along the axis every angle is stationary, and any one stands for the rest.
        let us = if dx.hypot(dy) <= f64::EPSILON {
            vec![0.5 * (lo.x + hi.x)]
        } else {
            let a = dy.atan2(dx);
            vec![a, a + PI]
        };
        for u in us
            .into_iter()
            .flat_map(|u| images(u, lo.x, hi.x, s.u_period()))
        {
            let b = dz.atan2(dx * u.cos() + dy * u.sin());
            for v in [b - PI, b, b + PI] {
                for v in images(v, lo.y, hi.y, s.v_period()) {
                    bb.add(s.eval(u, v));
                }
            }
        }
    }
}

/// A box that holds the face: its boundary's, with whatever the surface can reach past it. The
/// one box of a face for every cull against it, so a ray that leaves through a face's bulge is
/// not dropped by a box that stops short of it.
///
/// A plane, and any surface drawn by straight lines (a cylinder, a cone, an extrusion along its
/// direction, a ruled surface), reaches no further than its boundary: its extremes lie along its
/// lines, which run to the boundary. A sphere and a torus have their stationary points in closed
/// form. A surface of revolution lies on the rings of its profile, inside the cylinder that holds
/// them. A spline surface lies in its patches' boxes. The parameter window is widened by a tenth,
/// since it is the box of the boundary's sampled images.
pub(crate) fn face_bbox(dom: &Domain) -> Aabb {
    let mut bb = Aabb::EMPTY;
    for tr in dom.trims() {
        bb.union(&tr.bbox());
    }
    let s = &dom.surface;
    let pad = (dom.hi - dom.lo) * 0.1;
    let (lo, hi) = (dom.lo - pad, dom.hi + pad);
    match s {
        Surface::Sphere(sp) => stationary_points(s, &sp.frame, dom, &mut bb),
        Surface::Torus(t) => stationary_points(s, &t.frame, dom, &mut bb),
        Surface::Revolution(r) => {
            // The profile's piece lies in its own box, so its rings lie between the box's least
            // and greatest heights along the axis, no further from the axis than its furthest
            // corner: a distance from a line is greatest over a box at a corner.
            let profile = r.curve.bbox(lo.y, hi.y);
            let (mut low, mut high, mut reach) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
            for k in 0..8 {
                let pick = |bit: usize, a: f64, b: f64| if k >> bit & 1 == 0 { a } else { b };
                let corner = Vec3::new(
                    pick(0, profile.min.x, profile.max.x),
                    pick(1, profile.min.y, profile.max.y),
                    pick(2, profile.min.z, profile.max.z),
                ) - r.origin;
                let z = corner.dot(r.axis);
                (low, high) = (low.min(z), high.max(z));
                reach = reach.max((corner - r.axis * z).norm());
            }
            // A disc of that reach square to the axis at each end.
            let spread = Vec3::new(
                reach * (1.0 - r.axis.x * r.axis.x).max(0.0).sqrt(),
                reach * (1.0 - r.axis.y * r.axis.y).max(0.0).sqrt(),
                reach * (1.0 - r.axis.z * r.axis.z).max(0.0).sqrt(),
            );
            for z in [low, high] {
                let centre = r.origin + r.axis * z;
                bb.add(centre - spread);
                bb.add(centre + spread);
            }
        }
        Surface::Nurbs(ns) => {
            // A periodic parameter's window may lie a period off the patches' own.
            let periodic = ns.u_period().is_some() || ns.v_period().is_some();
            for q in ns.patches() {
                let apart = q.hi.x < lo.x || q.lo.x > hi.x || q.hi.y < lo.y || q.lo.y > hi.y;
                if periodic || !apart {
                    bb.union(&q.bbox);
                }
            }
        }
        _ => {}
    }
    padded(&bb, 1e-6 * (1.0 + bb.diagonal()))
}

/// A coedge's parameter-plane points at edge parameters `ts`, given in its direction; points in
/// between are sampled so periodic parameters continue across wide gaps.
fn coedge_uv_at(b: &Brep, c: CoedgeId, ts: &[f64]) -> Vec<Vec2> {
    match &b.coedge(c).pcurve {
        PCurve::Affine { origin, dir } => return ts.iter().map(|&t| *origin + *dir * t).collect(),
        PCurve::Spline { .. } => return b.coedge_uvs(c, ts),
        PCurve::Projected { .. } => {}
    }
    let s = &b.face(b.coedge_face(c)).surface;
    // Only a periodic parameter needs the dense samples, to continue across a wide gap.
    if s.u_period().is_none() && s.v_period().is_none() {
        return b.coedge_uvs(c, ts);
    }
    let per = 16;
    let mut dense = Vec::new();
    for w in ts.windows(2) {
        for k in 0..per {
            dense.push(w[0] + (w[1] - w[0]) * k as f64 / per as f64);
        }
    }
    dense.push(ts[ts.len() - 1]);
    let uvs = b.coedge_uvs(c, &dense);
    (0..ts.len()).map(|i| uvs[i * per]).collect()
}

/// A trim for coedge `c` over curve parameters `t0 → t1`, at `uv0 → uv1`, between vertices `v`.
pub(crate) fn coedge_trim_with(
    b: &Brep,
    c: CoedgeId,
    piece: usize,
    curve: Option<Rc<Curve>>,
    (t0, t1): (f64, f64),
    (uv0, uv1): (Vec2, Vec2),
    (v0, v1): (usize, usize),
) -> Trim {
    let co = b.coedge(c);
    let edge = b.edge(co.edge);
    let map = UvMap::of(&co.pcurve);
    Trim {
        piece,
        forward: !co.reversed,
        curve,
        pole: b.vertex(edge.start).point,
        t0,
        t1,
        map: if edge.curve.is_none() {
            match map {
                UvMap::Projected => UvMap::Affine {
                    origin: uv0,
                    dir: Vec2::ZERO,
                },
                m => m,
            }
        } else {
            map
        },
        uv0,
        uv1,
        v0,
        v1,
    }
}

/// Where a piece of curve through `[start, middle, end]` lies in a face: as its middle does,
/// except that a piece not running along one trim is inside where the exact side puts its middle.
/// Shorter than twice the tolerance, a piece leaving the boundary, or cutting across a corner, has
/// its middle within the tolerance of the boundary, though only one lying along a trim is on it.
fn locate_piece(dom: &Domain, pts: [Point3; 3]) -> Where {
    match dom.locate(pts[1]) {
        Where::Boundary if !dom.along_one_trim(&pts) => dom
            .exact_side(pts[1])
            .map_or(Where::Boundary, Where::Inside),
        w => w,
    }
}

/// Face `f`'s trimmed domain outside any boolean: its trims numbered by the arena's vertices.
pub(crate) fn face_domain(b: &Brep, f: FaceId) -> Domain {
    let face = b.face(f);
    let loops = face
        .loops
        .iter()
        .map(|&lp| {
            b.lp(lp)
                .coedges
                .iter()
                .map(|&c| {
                    let ts = coedge_params(b, c, 16);
                    let uvs = coedge_uv_at(b, c, &ts);
                    let curve = b.edge(b.coedge(c).edge).curve.clone().map(Rc::new);
                    let v = (b.coedge_start(c).index(), b.coedge_end(c).index());
                    let (t, uv) = ((ts[0], ts[16]), (uvs[0], uvs[16]));
                    coedge_trim_with(b, c, usize::MAX, curve, t, uv, v)
                })
                .collect()
        })
        .collect();
    Domain::new(face.surface.clone(), face.reversed, loops)
}

impl<'a> Work<'a> {
    fn new(breps: [&'a Brep; 2]) -> Work<'a> {
        let mut w = Work {
            breps,
            pool: Vec::new(),
            vtol: Vec::new(),
            widest: tol::LINEAR,
            tolerant: Vec::new(),
            cap: tol::LINEAR,
            grid: HashMap::new(),
            vmap: [Vec::new(), Vec::new()],
            faces: Vec::new(),
            face_of: [Vec::new(), Vec::new()],
            bases: Vec::new(),
            edge_base: [Vec::new(), Vec::new()],
            coincident: Vec::new(),
            pieces: Vec::new(),
            base_pieces: Vec::new(),
            cut_faces: Vec::new(),
            canon: Vec::new(),
            hits: Vec::new(),
            on_face: BTreeSet::new(),
            inputs: 0,
            met: Vec::new(),
            touch_points: Vec::new(),
            unplaced: HashMap::new(),
            meetings: HashMap::new(),
            near: None,
        };
        for b in breps {
            let (vs, es) = b.tolerant();
            for t in vs.iter().map(|x| x.1).chain(es.iter().map(|x| x.1)) {
                w.cap = w.cap.max(t);
            }
        }
        for (side, &b) in breps.iter().enumerate() {
            w.vmap[side] = b
                .vertex_ids()
                .map(|v| w.vertex_with(b.vertex(v).point, b.vertex_tolerance(v)))
                .collect();
            for e in b.edge_ids() {
                let edge = b.edge(e);
                let curve = edge.curve.clone().map(Rc::new);
                let bbox = match &curve {
                    Some(c) => curve_bbox(c, edge.range),
                    None => {
                        let mut bb = Aabb::EMPTY;
                        bb.add(b.vertex(edge.start).point);
                        bb
                    }
                };
                w.edge_base[side].push(w.bases.len());
                w.bases.push(Base {
                    curve,
                    range: edge.range,
                    v0: Some(w.vmap[side][edge.start.index()]),
                    v1: Some(w.vmap[side][edge.end.index()]),
                    pole: b.vertex(edge.start).point,
                    closed: edge.start == edge.end,
                    bbox,
                    kind: BaseKind::Edge,
                    tol: b.edge_tolerance(e),
                });
            }
            for f in b.face_ids() {
                let (mut dom, mut bbox) = w.original_domain(side, f);
                if b.has_tolerances() {
                    let tols: Vec<Vec<f64>> = b
                        .face(f)
                        .loops
                        .iter()
                        .map(|&lp| {
                            let cs = &b.lp(lp).coedges;
                            cs.iter()
                                .map(|&c| b.edge_tolerance(b.coedge(c).edge))
                                .collect()
                        })
                        .collect();
                    dom.set_tolerances(tols);
                    bbox = padded(&bbox, dom.widest());
                }
                w.face_of[side].push(w.faces.len());
                w.faces.push(Face {
                    side,
                    id: f,
                    dom,
                    bbox,
                });
            }
        }
        w.inputs = w.pool.len();
        w
    }

    /// The pool vertex that is the meeting of faces `fa` and `fb` found at `p` again, if one lies
    /// on both surfaces within [`GRAZE_REACH`] of it with both surfaces within the tolerance all
    /// the way between: surfaces crossing at a shallow angle meet within the tolerance along a
    /// band, and two points with no excursion past it between them are one meeting.
    fn same_meeting(&self, p: Point3, fa: usize, fb: usize) -> Option<usize> {
        let (sa, sb) = (&self.faces[fa].dom.surface, &self.faces[fb].dom.surface);
        let on = |x: Point3| {
            csi::signed_distance(sa, x).abs() <= tol::LINEAR
                && csi::signed_distance(sb, x).abs() <= tol::LINEAR
        };
        let c = cell(p);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(ids) = self.grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                        continue;
                    };
                    for &i in ids {
                        let q = self.pool[i];
                        let d = q.dist(p);
                        if d > GRAZE_REACH || best.is_some_and(|b| b.0 <= d) {
                            continue;
                        }
                        if (0..=8).all(|k| on(p + (q - p) * (k as f64 / 8.0))) {
                            best = Some((d, i));
                        }
                    }
                }
            }
        }
        best.map(|b| b.1)
    }

    fn vertex(&mut self, p: Point3) -> usize {
        self.vertex_with(p, tol::LINEAR)
    }

    /// The pool vertex that `p`, with tolerance `t`, coincides with, or a new one.
    fn vertex_with(&mut self, p: Point3, t: f64) -> usize {
        if let Some(i) = self.coincident_vertex(p, t) {
            return i;
        }
        self.pool.push(p);
        self.vtol.push(t);
        self.widest = self.widest.max(t);
        let i = self.pool.len() - 1;
        if t > tol::LINEAR {
            self.tolerant.push(i);
        }
        self.grid.entry(cell(p)).or_default().push(i);
        i
    }

    /// The nearest pool vertex within the sum of its tolerance and `t` of `p` (D. J. Jackson,
    /// "Boundary representation modelling with local tolerances", Solid Modeling '95; "Jackson
    /// 1995" below).
    /// Where either is tolerant, the vertex's tolerance grows to hold `p`'s sphere, as every curve
    /// that ends at `p` must end within it, but never past the inputs' widest.
    fn coincident_vertex(&mut self, p: Point3, t: f64) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        let mut consider = |i: usize| {
            let d = self.pool[i].dist(p);
            if d <= tol::between(t, self.vtol[i]) && best.is_none_or(|b| d < b.0) {
                best = Some((d, i));
            }
        };
        // Exact pool vertices lie within `t` of `p`; tolerant ones are searched one by one.
        let reach = (t / CELL).ceil().max(1.0) as i64;
        if (2 * reach + 1).pow(3) as usize > self.pool.len() {
            (0..self.pool.len()).for_each(&mut consider);
        } else {
            let c = cell(p);
            for dx in -reach..=reach {
                for dy in -reach..=reach {
                    for dz in -reach..=reach {
                        if let Some(ids) = self.grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            ids.iter().for_each(|&i| consider(i));
                        }
                    }
                }
            }
            self.tolerant.iter().for_each(|&i| consider(i));
        }
        let (d, i) = best?;
        if t > tol::LINEAR || self.vtol[i] > tol::LINEAR {
            self.vtol[i] = self.vtol[i].max(d + t).min(self.cap);
            if !self.tolerant.contains(&i) {
                self.tolerant.push(i);
            }
            self.widest = self.widest.max(self.vtol[i]);
        }
        Some(i)
    }

    /// A trim for an input coedge, whole.
    fn coedge_trim(&self, side: usize, c: CoedgeId, piece: usize) -> Trim {
        let b = self.breps[side];
        let ts = coedge_params(b, c, 16);
        let uvs = coedge_uv_at(b, c, &ts);
        self.trim_of(
            side,
            c,
            piece,
            (ts[0], ts[ts.len() - 1]),
            (uvs[0], uvs[uvs.len() - 1]),
        )
    }

    /// A trim for an input coedge from its parameter range and parameter-plane ends.
    fn trim_of(
        &self,
        side: usize,
        c: CoedgeId,
        piece: usize,
        t: (f64, f64),
        uv: (Vec2, Vec2),
    ) -> Trim {
        let b = self.breps[side];
        let base = &self.bases[self.edge_base[side][b.coedge(c).edge.index()]];
        let v = (
            self.vmap[side][b.coedge_start(c).index()],
            self.vmap[side][b.coedge_end(c).index()],
        );
        coedge_trim_with(b, c, piece, base.curve.clone(), t, uv, v)
    }

    /// An input face's domain and box, from its memo when an earlier boolean worked them out.
    fn original_domain(&self, side: usize, f: FaceId) -> (Domain, Aabb) {
        let b = self.breps[side];
        let face = b.face(f);
        let coedges = |lp: &crate::topo::LoopId| b.lp(*lp).coedges.iter();
        if let Some(img) = b.memo(f).images.get() {
            let loops = face
                .loops
                .iter()
                .zip(&img.trims)
                .map(|(lp, ims)| {
                    coedges(lp)
                        .zip(ims)
                        .map(|(&c, im)| self.trim_of(side, c, usize::MAX, im.t, im.uv))
                        .collect()
                })
                .collect();
            let dom = Domain::from_samples(
                face.surface.clone(),
                face.reversed,
                loops,
                img.samples.clone(),
                img.boxes.clone(),
            );
            return (dom, img.bbox);
        }
        let loops: Vec<Vec<Trim>> = face
            .loops
            .iter()
            .map(|lp| {
                coedges(lp)
                    .map(|&c| self.coedge_trim(side, c, usize::MAX))
                    .collect()
            })
            .collect();
        let ends = face
            .loops
            .iter()
            .map(|lp| {
                coedges(lp)
                    .map(|&c| {
                        let (t0, t1) = b.edge(b.coedge(c).edge).range;
                        let ts = if b.coedge(c).reversed {
                            [t1, t0]
                        } else {
                            [t0, t1]
                        };
                        let uvs = coedge_uv_at(b, c, &ts);
                        (uvs[0], uvs[1])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let trims = loops
            .iter()
            .zip(&ends)
            .map(|(l, e)| {
                l.iter()
                    .zip(e)
                    .map(|(tr, &whole)| TrimImage {
                        t: (tr.t0, tr.t1),
                        uv: (tr.uv0, tr.uv1),
                        whole,
                    })
                    .collect()
            })
            .collect();
        let dom = Domain::new(face.surface.clone(), face.reversed, loops);
        let bbox = face_bbox(&dom);
        let _ = b.memo(f).images.set(FaceImages {
            trims,
            samples: dom.samples().to_vec(),
            boxes: dom.boxes().to_vec(),
            bbox,
        });
        (dom, bbox)
    }

    /// `e`, saying which faces it arose on: one boolean of many solids otherwise fails without a
    /// trace of where.
    fn at_faces(&self, e: Error, faces: &[usize]) -> Error {
        let named: Vec<String> = faces
            .iter()
            .map(|&f| {
                let face = &self.faces[f];
                let c = (face.bbox.min + face.bbox.max) * 0.5;
                let b = self.breps[face.side];
                let solid = b
                    .face(face.id)
                    .shell
                    .and_then(|sh| b.shell(sh).solid)
                    .map_or(String::new(), |s| format!(" of solid {}", s.index()));
                format!(
                    "{} face {}{solid} of input {} about ({:.4}, {:.4}, {:.4})",
                    face.dom.surface.kind(),
                    face.id.index(),
                    face.side,
                    c.x,
                    c.y,
                    c.z
                )
            })
            .collect();
        let at = named.join(" and ");
        match e {
            Error::Numerical(m) => Error::Numerical(format!("{m}, on the {at}")),
            other => other,
        }
    }

    fn other(&self, side: usize) -> &[usize] {
        &self.face_of[1 - side]
    }

    /// Step 1a: where each edge of one solid crosses a face of the other, kept to be placed once
    /// every other meeting point is known.
    fn find_near(&mut self) {
        let faces = [0, 1].map(|s| -> Vec<Aabb> {
            self.face_of[s]
                .iter()
                .map(|&f| self.faces[f].bbox)
                .collect()
        });
        let edges = [0, 1].map(|s| -> Vec<Aabb> {
            self.edge_base[s]
                .iter()
                .map(|&e| self.bases[e].bbox)
                .collect()
        });
        self.near = near::Near::new(
            self.breps,
            [&faces[0], &faces[1]],
            [&edges[0], &edges[1]],
            2.0 * self.cap + csi::GRAZE_GAP,
        );
    }

    /// Every step of the boolean after setting up.
    fn finish(mut self, op: Op, facts: &Facts) -> Result<Brep> {
        self.take_facts(facts)?;
        self.meet()?;
        self.split_curves();
        self.imprint_on_faces();
        self.dedup();
        let (frags, poles, intact) = self.split_faces()?;
        let mut kept = self.select(frags, &intact, op)?;
        let mut geoms = self.piece_geoms();
        geoms.extend(poles);
        self.touch_loops(&mut kept, &mut geoms);
        sew::assemble(kept, &self.pool, &self.vtol, &geoms)
    }

    /// Whether the face pairs or edge pairs whose boxes overlap are many enough that meshing both
    /// inputs to rule most of them out costs less than the exact work on all of them.
    fn many_pairs(&self) -> bool {
        let count = |a: Vec<&Aabb>, b: Vec<&Aabb>, enough: usize| {
            let mut n = 0;
            for x in &a {
                n += b.iter().filter(|y| overlap(x, y)).count();
                if n >= enough {
                    return true;
                }
            }
            false
        };
        let faces = |side: usize| {
            self.face_of[side]
                .iter()
                .map(|&f| &self.faces[f].bbox)
                .collect()
        };
        let edges = |side: usize| {
            self.edge_base[side]
                .iter()
                .map(|&e| &self.bases[e].bbox)
                .collect()
        };
        count(faces(0), faces(1), MANY_FACE_PAIRS) || count(edges(0), edges(1), MANY_EDGE_PAIRS)
    }

    /// For each face that stays clear of the other solid, a representative of the faces joined to
    /// it through edges that stay clear too: crossing such an edge crosses nothing of the other
    /// solid, so every face of one group lies on the same side of it. `None` for the rest, and for
    /// every face where nothing is known of what is near.
    fn clear_groups(&self) -> Vec<Option<usize>> {
        let n = self.faces.len();
        let Some(near) = &self.near else {
            return vec![None; n];
        };
        let clear: Vec<bool> = (0..n)
            .map(|fi| {
                let side = self.faces[fi].side;
                near.face_clear(side, fi - self.face_of[side][0])
            })
            .collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let mut parent: Vec<usize> = (0..n).collect();
        for side in 0..2 {
            let b = self.breps[side];
            let mut users: Vec<Vec<usize>> = vec![Vec::new(); self.edge_base[side].len()];
            for (k, f) in b.face_ids().enumerate() {
                let fi = self.face_of[side][k];
                for &lp in &b.face(f).loops {
                    for &c in &b.lp(lp).coedges {
                        users[b.coedge(c).edge.index()].push(fi);
                    }
                }
            }
            for (e, faces) in users.iter().enumerate() {
                if !near.edge_clear(side, e) {
                    continue;
                }
                for w in faces.windows(2) {
                    if clear[w[0]] && clear[w[1]] {
                        let (x, y) = (root(&mut parent, w[0]), root(&mut parent, w[1]));
                        parent[x] = y;
                    }
                }
            }
        }
        (0..n)
            .map(|fi| clear[fi].then(|| root(&mut parent, fi)))
            .collect()
    }

    /// Whether faces `fa` and `fb`, of the first solid and the second, may meet.
    fn faces_near(&self, fa: usize, fb: usize) -> bool {
        self.near
            .as_ref()
            .is_none_or(|n| n.faces(fa - self.face_of[0][0], fb - self.face_of[1][0]))
    }

    /// Whether edge bases `ba` and `bb`, of the first solid and the second, may meet.
    fn edges_near(&self, ba: usize, bb: usize) -> bool {
        self.near
            .as_ref()
            .is_none_or(|n| n.edges(ba - self.edge_base[0][0], bb - self.edge_base[1][0]))
    }

    /// Whether edge base `bi` of solid `side` may meet face `fi` of the other.
    fn edge_face_near(&self, side: usize, bi: usize, fi: usize) -> bool {
        self.near.as_ref().is_none_or(|n| {
            n.edge_face(
                side,
                bi - self.edge_base[side][0],
                fi - self.face_of[1 - side][0],
            )
        })
    }

    fn pierce(&mut self) {
        let mut found: Vec<Hit> = Vec::new();
        let mut on = Vec::new();
        // An edge one face uses on both sides (a seam, or a slit a contact left) bounds no
        // material: the face's surface goes on across it. Lying on the other solid's face it marks
        // no change of side there, and where the two surfaces cross at a shallow angle a seam runs
        // within the tolerance of the other face along its length without being a cut.
        let one_face: HashSet<usize> = self
            .edge_faces()
            .into_iter()
            .filter(|(_, fs)| fs.len() == 1)
            .map(|(bi, _)| bi)
            .collect();
        for side in 0..2 {
            for &bi in &self.edge_base[side] {
                let base = &self.bases[bi];
                let Some(curve) = &base.curve else { continue };
                let bounds_nothing = one_face.contains(&bi);
                for &fi in self.other(side) {
                    let face = &self.faces[fi];
                    if !overlap(&base.bbox, &face.bbox)
                        || self.on_face.contains(&(bi, fi))
                        || !self.edge_face_near(side, bi, fi)
                    {
                        continue;
                    }
                    let range = match &**curve {
                        Curve::Line(l) => {
                            match ssi::clip_line(l.origin, l.dir, &padded(&face.bbox, 1e-6)) {
                                Some((a, b)) => (a.max(base.range.0), b.min(base.range.1)),
                                None => continue,
                            }
                        }
                        _ => base.range,
                    };
                    if range.1 < range.0 {
                        continue;
                    }
                    if base.tol > tol::LINEAR
                        && lies_within(curve, base.range, &face.dom.surface, base.tol)
                    {
                        if !bounds_nothing {
                            on.push((bi, fi));
                        }
                        continue;
                    }
                    let hits = csi::curve_surface(curve, range, &face.dom.surface);
                    if hits.on_surface {
                        if !bounds_nothing {
                            on.push((bi, fi));
                        }
                        continue;
                    }
                    for t in hits.ts {
                        let p = curve.eval(t);
                        let at = face.dom.locate(p);
                        if at != Where::Outside {
                            let s = &face.dom.surface;
                            let uv = s.project(p);
                            let tangent = curve.d1(t).1.unit_or_zero();
                            let sine = tangent.dot(s.normal(uv.x, uv.y)).abs();
                            // Grazing the face along its boundary, the edge runs along a boundary
                            // edge: the vertices of that edge-edge contact are each one's ends
                            // lying on the other, and roots along the band are only rounding.
                            // Beside a pole the parameter plane puts a point a hair from the pole
                            // on the boundary though no edge is near it in space, and a seam bounds
                            // no material: a root near neither is where the edge crosses the face.
                            let seam = |c: &Rc<Curve>| {
                                face.dom
                                    .trims()
                                    .filter(|o| o.curve.as_ref().is_some_and(|x| Rc::ptr_eq(x, c)))
                                    .count()
                                    > 1
                            };
                            if at == Where::Boundary
                                && sine <= tol::TANGENT
                                && face.dom.trims().any(|tr| {
                                    tr.curve.as_ref().is_some_and(|c| {
                                        let r = (tr.t0.min(tr.t1), tr.t0.max(tr.t1));
                                        !seam(c) && csi::within(c, r, p, tol::LINEAR)
                                    })
                                })
                            {
                                continue;
                            }
                            found.push(Hit {
                                p,
                                t,
                                base: bi,
                                face: fi,
                                sine,
                            });
                        }
                    }
                }
            }
        }
        self.hits = found;
        self.on_face.extend(on);
    }

    /// Records each fact that an edge lies on a face, once it is checked at the edge's ends and
    /// quarter points.
    fn take_facts(&mut self, facts: &Facts) -> Result<()> {
        for f in &facts.on_face {
            let (side, other) = match f.side {
                0 => (0, 1),
                1 => (1, 0),
                _ => {
                    return Err(Error::InvalidInput(
                        "a fact names an input other than 0 or 1".into(),
                    ));
                }
            };
            let (Some(&bi), Some(&fi)) = (
                self.edge_base[side].get(f.edge.index()),
                self.face_of[other].get(f.face.index()),
            ) else {
                return Err(Error::InvalidInput(
                    "a fact names an edge or face not in its input".into(),
                ));
            };
            let base = &self.bases[bi];
            let surface = &self.faces[fi].dom.surface;
            let holds = (0..=4).all(|k| {
                let p = match &base.curve {
                    Some(c) => {
                        c.eval(base.range.0 + (base.range.1 - base.range.0) * k as f64 / 4.0)
                    }
                    None => base.pole,
                };
                csi::signed_distance(surface, p).abs() <= base.tol
            });
            if !holds {
                return Err(Error::InvalidInput(format!(
                    "the fact that edge {} of input {side} lies on face {} of the other does not \
                     hold",
                    f.edge.index(),
                    f.face.index()
                )));
            }
            self.on_face.insert((bi, fi));
        }
        Ok(())
    }

    /// Each edge base's faces, by face index.
    fn edge_faces(&self) -> HashMap<usize, Vec<usize>> {
        let mut out: HashMap<usize, Vec<usize>> = HashMap::new();
        for (f, face) in self.faces.iter().enumerate() {
            let b = self.breps[face.side];
            for &lp in &b.face(face.id).loops {
                for &c in &b.lp(lp).coedges {
                    let eb = self.edge_base[face.side][b.coedge(c).edge.index()];
                    let list = out.entry(eb).or_default();
                    if !list.contains(&f) {
                        list.push(f);
                    }
                }
            }
        }
        out
    }

    /// Step 2c: an edge lying on a face of the other solid cuts that face, as the face pair's
    /// intersection curve along it would: where the edge's own faces cross the face, and where they
    /// only touch it, a contact the result may keep both sides of.
    fn imprint_on_faces(&mut self) {
        for &(bi, fi) in &self.on_face {
            for &pi in &self.base_pieces[bi] {
                if self.bases[bi].curve.is_none() {
                    continue;
                }
                let ends = [0.0, 0.5, 1.0].map(|f| self.piece_point(pi, f));
                if !matches!(locate_piece(&self.faces[fi].dom, ends), Where::Inside(_)) {
                    continue;
                }
                if !self.cut_faces[pi].contains(&fi) {
                    self.cut_faces[pi].push(fi);
                }
            }
        }
    }

    /// Step 1: every meeting point. Edges' meetings with faces are placed last, once the faces'
    /// intersection curves are known.
    fn meet(&mut self) -> Result<()> {
        self.pierce();
        self.cross_edges();
        self.intersect_faces()?;
        self.place_hits();
        Ok(())
    }

    /// Places step 1a's meetings, steepest first, so where several computations find one grazing
    /// meeting the vertex is the one the best-conditioned of them found, whatever order they ran in.
    fn place_hits(&mut self) {
        let mut hits = std::mem::take(&mut self.hits);
        hits.sort_by(|a, b| {
            b.sine
                .total_cmp(&a.sine)
                .then_with(|| a.p.x.total_cmp(&b.p.x))
                .then_with(|| a.p.y.total_cmp(&b.p.y))
                .then_with(|| a.p.z.total_cmp(&b.p.z))
                .then_with(|| (a.base, a.face).cmp(&(b.base, b.face)))
        });
        let edge_faces = self.edge_faces();
        let mut cuts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (bi, base) in self.bases.iter().enumerate() {
            if let BaseKind::Cut {
                fa,
                fb,
                tangent: false,
            } = base.kind
            {
                cuts.entry((fa.min(fb), fa.max(fb))).or_default().push(bi);
            }
        }
        for h in hits {
            let (mut p, mut t, mut on) = (h.p, h.t, None);
            // The steepest crossing, then the nearest.
            let mut best = (0.0, f64::INFINITY);
            let own = edge_faces.get(&h.base).map_or(&[][..], Vec::as_slice);
            for &f in own {
                let key = (f.min(h.face), f.max(h.face));
                for &ci in cuts.get(&key).map_or(&[][..], Vec::as_slice) {
                    if let Some((q, tq, sine)) = self.crossing(ci, h, own) {
                        let d = q.dist(h.p);
                        if sine > best.0 || (sine == best.0 && d < best.1) {
                            (p, t, on, best) = (q, tq, Some(ci), (sine, d));
                        }
                    }
                }
            }
            let v = self.touch_vertex(p, t, h.base, h.face, on);
            // Two surfaces each known to the tolerance fix where they cross to the tolerance over
            // the sine between them: a grazing meeting lies that far along the graze.
            let reach = (tol::LINEAR / h.sine.max(1e-12)).min(GRAZE_REACH);
            if reach > tol::LINEAR {
                let pairs = own.iter().map(|&f| (f.min(h.face), f.max(h.face), reach));
                self.meetings.entry(v).or_default().extend(pairs);
            }
            if h.sine <= tol::TANGENT {
                self.met.push((v, h.base, h.face));
            }
        }
    }

    /// Where intersection curve `ci`, of one of the faces `own` of the hit's edge and the hit's face,
    /// crosses the edge within the band where the edge stays on the face from the hit. Where the
    /// faces nearly coincide, the edge grazes the face and the hit can lie anywhere in that band;
    /// the crossing lies on the curve the faces are cut along, and is no worse placed, since inside
    /// the edge's own face the edge crosses the curve at least as steeply as it crosses the face.
    fn crossing(&self, ci: usize, h: Hit, own: &[usize]) -> Option<(Point3, f64, f64)> {
        let cut = &self.bases[ci];
        let edge = &self.bases[h.base];
        let (c, e) = (cut.curve.as_ref()?, edge.curve.as_ref()?);
        let BaseKind::Cut { fa, fb, .. } = cut.kind else {
            return None;
        };
        let mine = if own.contains(&fa) { fa } else { fb };
        // The edge's other face: the curve crosses it where it crosses the edge.
        let across = match own {
            [f, g] => Some(&self.faces[if *f == mine { *g } else { *f }].dom.surface),
            _ => None,
        };
        let (mut q, mut tq, sine) =
            csi::curve_curve_near(c, cut.range, e, edge.range, h.p, h.t, across)?;
        let face = &self.faces[h.face];
        // A fitted cut curve puts the crossing off the exact surfaces by its fit error; where the
        // three surfaces fix a point within the tolerance of it, that point is the meeting.
        if let Some(b) = across {
            let exact = csi::on_three([&self.faces[mine].dom.surface, b, &face.dom.surface], q)
                .filter(|x| x.dist(q) <= tol::LINEAR)
                .map(|x| (x, csi::curve_distance(e, edge.range, x)))
                .filter(|(_, (d, _))| *d <= tol::LINEAR);
            if let Some((x, (_, tx))) = exact {
                (q, tq) = (x, tx);
            }
        }
        let grazes = |k: usize| {
            let m = e.eval(h.t + (tq - h.t) * k as f64 / 16.0);
            csi::signed_distance(&face.dom.surface, m).abs() <= tol::LINEAR
        };
        // On the edge, the point is on its own faces; where the hit was, it is on the face.
        ((0..=16).all(grazes)
            && (q.dist(h.p) <= tol::LINEAR || face.dom.locate(q) != Where::Outside))
            .then_some((q, tq, sine))
    }

    /// The vertex where base `bi` meets face `fi` at `p`, its parameter `t`. Where the edge only
    /// touches the face, rounding hides the gap along a band, and the meeting found can lie
    /// anywhere in it: a vertex already on both, with the edge within the tolerance of the face
    /// all the way to it, is the same meeting, as two crossings with no excursion past the
    /// tolerance between them are one touching point. A meeting placed on intersection curve `on`
    /// is only that curve's: a vertex off it is another meeting in the same band, unless an earlier
    /// hit left it where it was found, on no curve. That one is this meeting, found from the other
    /// solid's edge, and anywhere in the band: it moves onto the curve, so the two solids' edges
    /// and the curve all pass one vertex where they cross.
    fn touch_vertex(
        &mut self,
        p: Point3,
        t: f64,
        bi: usize,
        fi: usize,
        on: Option<usize>,
    ) -> usize {
        let bt = self.bases[bi].tol;
        if bt > tol::LINEAR || self.widest > tol::LINEAR {
            // A meeting on a tolerant edge takes the edge's tolerance (Jackson 1995).
            if let Some(i) = self.coincident_vertex(p, bt) {
                return i;
            }
            if bt > tol::LINEAR {
                return self.vertex_with(p, bt);
            }
        }
        let base = &self.bases[bi];
        let curve = base.curve.clone().expect("pierced edges have curves");
        let range = base.range;
        let surface = &self.faces[fi].dom.surface;
        let near = |d: f64| d.abs() <= tol::LINEAR;
        let on_cut = |q: Point3| {
            on.is_none_or(|ci| {
                let cut = &self.bases[ci];
                let c = cut.curve.as_ref().expect("cuts have curves");
                csi::within(c, cut.range, q, tol::LINEAR)
            })
        };
        let c = cell(p);
        let mut best: Option<(f64, usize)> = None;
        let mut placed_here = false;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(ids) = self.grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                        continue;
                    };
                    for &i in ids {
                        let q = self.pool[i];
                        let d = q.dist(p);
                        if d <= tol::LINEAR {
                            return i;
                        }
                        if d > CELL || best.is_some_and(|b| b.0 <= d) {
                            continue;
                        }
                        let (off, tq) = csi::curve_distance(&curve, range, q);
                        let along = (1..8).all(|k| {
                            let m = curve.eval(t + (tq - t) * k as f64 / 8.0);
                            near(csi::signed_distance(surface, m))
                        });
                        let unplaced =
                            on.is_some() && self.unplaced.get(&i).is_some_and(|&r| d <= r);
                        if near(off)
                            && near(csi::signed_distance(surface, q))
                            && along
                            && (unplaced || on_cut(q))
                        {
                            best = Some((d, i));
                            placed_here = unplaced && !on_cut(q);
                        }
                    }
                }
            }
        }
        match best {
            Some((_, i)) => {
                if placed_here {
                    self.pool[i] = p;
                    self.grid.entry(cell(p)).or_default().push(i);
                    self.unplaced.remove(&i);
                }
                i
            }
            None => {
                let v = self.vertex(p);
                if on.is_none() {
                    self.unplaced.insert(v, GRAZE_REACH);
                }
                v
            }
        }
    }

    /// Step 1b: where edges of the two solids cross; this is what places vertices where a face's
    /// boundary crosses a coincident face's.
    fn cross_edges(&mut self) {
        let mut found = Vec::new();
        for &ba in &self.edge_base[0] {
            let a = &self.bases[ba];
            let Some(ca) = &a.curve else { continue };
            for &bb in &self.edge_base[1] {
                let b = &self.bases[bb];
                let Some(cb) = &b.curve else { continue };
                if !overlap(&padded(&a.bbox, tol::between(a.tol, b.tol)), &b.bbox)
                    || !self.edges_near(ba, bb)
                {
                    continue;
                }
                let t = a.tol.max(b.tol);
                found.extend(
                    csi::curve_curve(ca, a.range, cb, b.range)
                        .into_iter()
                        .map(|p| (p, t, ba, bb)),
                );
            }
        }
        for (p, t, ba, bb) in found {
            if t <= tol::LINEAR && self.touching_vertex(p, ba, bb).is_some() {
                continue;
            }
            // Two edges crossing at a shallow angle fix their crossing only loosely along them:
            // to the tolerance over the sine between them. Edges crossing square fix it to the
            // tolerance, and a meeting of other edges further off is another one.
            let v = self.vertex_with(p, t);
            if t <= tol::LINEAR && v >= self.inputs {
                let tangent = |b: usize| -> Option<Vec3> {
                    let base = &self.bases[b];
                    let c = base.curve.as_ref()?;
                    let (_, at) = csi::curve_distance(c, base.range, p);
                    c.d1(at).1.unit()
                };
                let sine = match (tangent(ba), tangent(bb)) {
                    (Some(x), Some(y)) => x.cross(y).norm(),
                    _ => 0.0,
                };
                let reach = (tol::LINEAR / sine.max(1e-12)).min(GRAZE_REACH);
                let r = self.unplaced.entry(v).or_insert(0.0);
                *r = r.max(reach);
            }
        }
        self.graze_edges();
    }

    /// Step 1b, after the meetings: where edges of the two solids run within [`csi::GRAZE_GAP`] of
    /// each other and cross ([`csi::grazing`]). A seam bounds no material and is left out.
    fn graze_edges(&mut self) {
        let owners = self.edge_faces();
        let seam = |b: usize| owners.get(&b).is_some_and(|f| f.len() == 1);
        let mut parts = Vec::new();
        // Meetings the closed forms and steps missed: two splines a hair apart cross at angles of
        // 1e-7 rad, where only the gap's least finds them.
        for &ba in &self.edge_base[0] {
            let a = &self.bases[ba];
            let Some(ca) = &a.curve else { continue };
            if a.tol > tol::LINEAR || seam(ba) {
                continue;
            }
            for &bb in &self.edge_base[1] {
                let b = &self.bases[bb];
                let Some(cb) = &b.curve else { continue };
                if b.tol > tol::LINEAR
                    || seam(bb)
                    || !overlap(&padded(&a.bbox, tol::LINEAR), &b.bbox)
                    || !self.edges_near(ba, bb)
                {
                    continue;
                }
                parts.extend(csi::grazing(ca, a.range, cb, b.range));
            }
        }
        for p in parts {
            if self.pool.iter().all(|q| q.dist(p) > csi::GRAZE_GAP) {
                self.vertex(p);
            }
        }
    }

    /// A pool vertex on both edges `ba` and `bb` near their meeting `p`, the two staying within the
    /// tolerance of each other all the way from it to `p`. Where edges touch rather than cross,
    /// rounding hides the gap along a band and the meeting found can lie anywhere in it; a vertex
    /// already on both in that band is the same meeting.
    fn touching_vertex(&self, p: Point3, ba: usize, bb: usize) -> Option<usize> {
        let near = |d: f64| d <= tol::LINEAR;
        let (a, b) = (&self.bases[ba], &self.bases[bb]);
        let (ca, cb) = (a.curve.as_ref()?, b.curve.as_ref()?);
        let (_, tp) = csi::curve_distance(ca, a.range, p);
        let c = cell(p);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(ids) = self.grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                        continue;
                    };
                    for &i in ids {
                        let q = self.pool[i];
                        let d = q.dist(p);
                        if d > CELL || best.is_some_and(|x| x.0 <= d) {
                            continue;
                        }
                        let (off_a, tq) = csi::curve_distance(ca, a.range, q);
                        if !near(off_a) || !csi::within(cb, b.range, q, tol::LINEAR) {
                            continue;
                        }
                        let along = (1..8).all(|k| {
                            let m = ca.eval(tp + (tq - tp) * k as f64 / 8.0);
                            csi::within(cb, b.range, m, tol::LINEAR)
                        });
                        if along {
                            best = Some((d, i));
                        }
                    }
                }
            }
        }
        best.map(|x| x.1)
    }

    /// Step 1c: every face pair's intersection curves, and coincident pairs.
    fn intersect_faces(&mut self) -> Result<()> {
        let pairs: Vec<(usize, usize)> = self.face_of[0]
            .iter()
            .flat_map(|&fa| self.face_of[1].iter().map(move |&fb| (fa, fb)))
            .filter(|&(fa, fb)| {
                overlap(&self.faces[fa].bbox, &self.faces[fb].bbox) && self.faces_near(fa, fb)
            })
            .collect();
        let owners = self.edge_faces();
        for (fa, fb) in pairs {
            let (sa, sb) = (&self.faces[fa].dom.surface, &self.faces[fb].dom.surface);
            if ssi::same_domain_faces(&self.faces[fa].dom, &self.faces[fb].dom) {
                self.coincident.push((fa, fb));
                continue;
            }
            let (ba, bb) = (&self.faces[fa].bbox, &self.faces[fb].bbox);
            let mut region = Aabb {
                min: Vec3::new(
                    ba.min.x.max(bb.min.x),
                    ba.min.y.max(bb.min.y),
                    ba.min.z.max(bb.min.z),
                ),
                max: Vec3::new(
                    ba.max.x.min(bb.max.x),
                    ba.max.y.min(bb.max.y),
                    ba.max.z.min(bb.max.z),
                ),
            };
            region = padded(&region, 1e-3 * (ba.diagonal() + bb.diagonal()) + 1e-4);
            // Edges of one face lying on the other's surface: the faces' contact along each is that
            // edge.
            let on_edges: Vec<usize> = self
                .on_face
                .iter()
                .filter(|&&(bi, f)| {
                    let own = owners.get(&bi).map_or(&[][..], Vec::as_slice);
                    (f == fb && own.contains(&fa)) || (f == fa && own.contains(&fb))
                })
                .map(|&(bi, _)| bi)
                .collect();
            // On the edge, or where the faces are tangent along it, within the band either side
            // of it where they stay within the tolerance of each other: a trace there is noise.
            // `past` widens the edge's range by that many times its length at either end.
            let along_past = |bi: usize, p: Point3, past: f64| {
                let base = &self.bases[bi];
                let (t0, t1) = base.range;
                let range = (t0 - past * (t1 - t0), t1 + past * (t1 - t0));
                base.curve.as_ref().is_some_and(|c| {
                    let (d, t) = csi::curve_distance(c, range, p);
                    let (q, dir) = c.d1(t);
                    d <= base.tol || d <= crossing_spread(sa, sb, q, dir, base.tol)
                })
            };
            let along = |bi: usize, p: Point3| along_past(bi, p, 0.0);
            // A tolerant edge lying on the other face is where the two faces meet; within its tube
            // they lie within its tolerance of each other, so a trace there finds only noise.
            // So is an edge lying on the other face where the two are tangent along it, within the
            // band either side where they stay within the tolerance of each other.
            let tangent_band = |b: &Base| {
                let c = b.curve.as_ref()?;
                let band = (1..8)
                    .map(|k| {
                        let t = b.range.0 + (b.range.1 - b.range.0) * k as f64 / 8.0;
                        contact::band(sa, sb, c.eval(t), c.d1(t).1)
                    })
                    .fold(f64::INFINITY, f64::min);
                (band > 0.0).then_some(band)
            };
            let tubes: Vec<(Rc<Curve>, (f64, f64), f64)> = on_edges
                .iter()
                .map(|&bi| &self.bases[bi])
                .filter_map(|b| {
                    let width = if b.tol > tol::LINEAR {
                        2.0 * b.tol
                    } else {
                        tangent_band(b)?
                    };
                    Some((b.curve.clone()?, b.range, width))
                })
                .collect();
            let mut touches: Vec<contact::Touch> = Vec::new();
            let result = {
                // The inputs' vertices, then the edges' meetings with faces, then the rest: a
                // trace starts from the first seed on its branch.
                let (inputs, found) = self.pool.split_at(self.inputs);
                let known = inputs
                    .iter()
                    .chain(self.hits.iter().map(|h| &h.p))
                    .chain(found);
                let (da, db) = (&self.faces[fa].dom, &self.faces[fb].dom);
                let pool = &self.pool;
                let touches = &mut touches;
                let mut seeds = || {
                    *touches = known_touches(pool, sa, sb, &region);
                    let mut out = seeds_for(known.clone(), da, db, &region);
                    out.retain(|&p| !touches.iter().any(|t| p.dist(t.p) <= t.reach));
                    out.retain(|&p| {
                        !tubes
                            .iter()
                            .any(|(c, r, w)| csi::curve_distance(c, *r, p).0 <= *w)
                    });
                    out
                };
                match ssi::intersect(sa, sb, &region, &mut seeds) {
                    Ok(r) => r,
                    // Faces that meet along a tolerant edge lying on both are that contact, as
                    // Jackson's face-face step leaves to edge-face coincidence; the checker still
                    // reads the result for faces that pass through each other elsewhere.
                    Err(_) if !tubes.is_empty() => Ssi::Curves {
                        curves: Vec::new(),
                        points: Vec::new(),
                    },
                    Err(e) => return Err(self.at_faces(e, &[fa, fb])),
                }
            };
            let Ssi::Curves { mut curves, points } = result else {
                self.coincident.push((fa, fb));
                continue;
            };
            // Where the surfaces touch at a known vertex and nowhere else near it, a trace that
            // stays within the touch's reach found only that vertex.
            curves.retain(|c| {
                !touches.iter().any(|t| {
                    (0..=8).all(|k| {
                        let x = c.range.0 + (c.range.1 - c.range.0) * k as f64 / 8.0;
                        c.curve.eval(x).dist(t.p) <= t.reach
                    })
                })
            });
            // So is a trace along an edge the faces are tangent along that runs on past both faces,
            // where the surfaces stay tangent along the edge's curve beyond its ends: the region it
            // traces in reaches beyond the faces.
            let off = |p: Point3| {
                self.faces[fa].dom.locate(p) == Where::Outside
                    && self.faces[fb].dom.locate(p) == Where::Outside
            };
            curves.retain(|c| {
                let at = |k: usize| {
                    c.curve
                        .eval(c.range.0 + (c.range.1 - c.range.0) * k as f64 / 8.0)
                };
                !on_edges.iter().any(|&bi| {
                    let tangent = tangent_band(&self.bases[bi]).is_some();
                    (0..=8).all(|k| {
                        along(bi, at(k)) || tangent && off(at(k)) && along_past(bi, at(k), 1.0)
                    })
                })
            });
            for p in points {
                if self.faces[fa].dom.locate(p) != Where::Outside
                    && self.faces[fb].dom.locate(p) != Where::Outside
                {
                    let v = self
                        .same_meeting(p, fa, fb)
                        .unwrap_or_else(|| self.vertex(p));
                    self.touch_points.push(v);
                }
            }
            for c in curves {
                let bbox = curve_bbox(&c.curve, c.range);
                self.bases.push(Base {
                    pole: c.curve.eval(c.range.0),
                    curve: Some(Rc::new(c.curve)),
                    range: c.range,
                    v0: None,
                    v1: None,
                    closed: c.closed,
                    bbox,
                    kind: BaseKind::Cut {
                        fa,
                        fb,
                        tangent: c.tangent,
                    },
                    tol: tol::LINEAR,
                });
            }
        }
        Ok(())
    }

    /// The vertices that lie inside a base's range, as `(parameter, vertex, distance)` in order.
    /// The pool vertices within reach of base `bi`'s box: those `points` (a hierarchy over the
    /// first `indexed` of the pool) finds there, and every vertex added since.
    fn splits(&self, bi: usize, points: &Bvh, indexed: usize) -> Vec<(f64, usize, f64)> {
        let base = &self.bases[bi];
        let Some(curve) = &base.curve else {
            return Vec::new();
        };
        let cut = match base.kind {
            BaseKind::Cut { fa, fb, .. } => Some((fa.min(fb), fa.max(fb))),
            _ => None,
        };
        // A meeting of the cut's own two faces is on the cut, however loosely a graze placed it.
        let grazing = |vi: usize| -> f64 {
            let met = self.meetings.get(&vi).into_iter().flatten();
            met.filter(|m| Some((m.0, m.1)) == cut)
                .map(|m| m.2)
                .fold(0.0, f64::max)
        };
        let reach = if cut.is_some() { GRAZE_REACH } else { 0.0 };
        let bb = padded(&base.bbox, base.tol + self.widest + reach);
        let mut near = Vec::new();
        points.query(&bb, &mut near);
        near.extend(indexed..self.pool.len());
        near.sort_unstable();
        let mut out: Vec<(f64, usize, f64)> = Vec::new();
        for vi in near {
            let p = self.pool[vi];
            if Some(vi) == base.v0 || Some(vi) == base.v1 {
                continue;
            }
            if !(0..3).all(|k| p[k] >= bb.min[k] && p[k] <= bb.max[k]) {
                continue;
            }
            let (d, t) = csi::curve_distance(curve, base.range, p);
            let within = tol::between(base.tol, self.vtol[vi]).max(grazing(vi));
            if d <= within || (d <= GRAZE_REACH && self.branch_point(cut, vi, d)) {
                out.push((t, vi, d));
                // A curve that crosses itself, as where a plane cuts a swept sheet trimmed where
                // it runs into itself, passes such a vertex twice.
                let gap = 1e-3 * (base.range.1 - base.range.0);
                for (lo, hi) in [(base.range.0, t - gap), (t + gap, base.range.1)] {
                    if hi - lo <= gap {
                        continue;
                    }
                    let (d2, t2) = csi::curve_distance(curve, (lo, hi), p);
                    if d2 <= within && t2 - lo > 0.5 * gap && hi - t2 > 0.5 * gap {
                        out.push((t2, vi, d2));
                    }
                }
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        let span = base.range.1 - base.range.0;
        out.dedup_by(|x, y| x.1 == y.1 && (x.0 - y.0).abs() <= 1e-6 * span);
        out
    }

    /// Whether vertex `vi`, `d` off the cut between faces `cut`, lies on both their surfaces where
    /// they are tangent, within the reach there that a trace through the point keeps to: two
    /// branches of the intersection cross at such a point (a swept sheet's edge and the curve where
    /// the sheet comes back into the face it left), and the trace passes it only that near.
    fn branch_point(&self, cut: Option<(usize, usize)>, vi: usize, d: f64) -> bool {
        let Some((fa, fb)) = cut else { return false };
        let p = self.pool[vi];
        let (sa, sb) = (&self.faces[fa].dom.surface, &self.faces[fb].dom.surface);
        let on = |s: &Surface| csi::signed_distance(s, p).abs() <= self.vtol[vi];
        on(sa) && on(sb) && contact::crossing_reach(sa, sb, p).is_some_and(|r| d <= r)
    }

    /// Step 2a: every base split at its vertices into pieces; intersection pieces outside either
    /// face are dropped.
    fn split_curves(&mut self) {
        self.base_pieces = vec![Vec::new(); self.bases.len()];
        let mut plans: Vec<(usize, Vec<f64>, Vec<usize>)> = Vec::new();
        let indexed = self.pool.len();
        let points = Bvh::of_boxes(self.pool.iter().map(|&p| Aabb { min: p, max: p }).collect());
        for bi in 0..self.bases.len() {
            let sp: Vec<(f64, usize)> = self
                .splits(bi, &points, indexed)
                .into_iter()
                .map(|(t, v, d)| {
                    // The piece ends where the curve passes the vertex, which must hold it.
                    if d > self.vtol[v] {
                        // A grazing meeting is known only to its reach, and a branch point only
                        // to the trace's: the vertex says so.
                        let met = self.meetings.get(&v).into_iter().flatten();
                        let mut reach = met.map(|m| m.2).fold(0.0, f64::max);
                        if let BaseKind::Cut { fa, fb, .. } = self.bases[bi].kind
                            && self.branch_point(Some((fa, fb)), v, d)
                        {
                            reach = reach.max(d);
                        }
                        self.vtol[v] = d.min(self.cap.max(reach));
                        self.widest = self.widest.max(self.vtol[v]);
                        if !self.tolerant.contains(&v) {
                            self.tolerant.push(v);
                        }
                    }
                    (t, v)
                })
                .collect();
            let base = &self.bases[bi];
            let mut ts: Vec<f64> = Vec::new();
            let mut vs: Vec<usize> = Vec::new();
            match (base.v0, base.v1) {
                (Some(v0), Some(v1)) => {
                    ts.push(base.range.0);
                    vs.push(v0);
                    for &(t, v) in &sp {
                        ts.push(t);
                        vs.push(v);
                    }
                    ts.push(base.range.1);
                    vs.push(v1);
                }
                _ if base.closed => {
                    let period = base.range.1 - base.range.0;
                    if sp.is_empty() {
                        let t = base.range.0;
                        let mid = base
                            .curve
                            .as_ref()
                            .expect("cut has a curve")
                            .eval(t + 0.5 * period);
                        if !self.cut_inside_both(bi, mid) {
                            continue;
                        }
                        // A closed curve that meets no vertex gets one of its own.
                        let p = base.pole;
                        let v = self.vertex(p);
                        ts = vec![t, t + period];
                        vs = vec![v, v];
                    } else {
                        for &(t, v) in &sp {
                            ts.push(t);
                            vs.push(v);
                        }
                        ts.push(sp[0].0 + period);
                        vs.push(sp[0].1);
                    }
                }
                _ => {
                    for &(t, v) in &sp {
                        ts.push(t);
                        vs.push(v);
                    }
                }
            }
            plans.push((bi, ts, vs));
        }
        // Vertices that split one curve less than the tolerance apart along it are one place on
        // it: faces lying within the tolerance of each other can put two corners there whose
        // points are just over it apart, and the piece between them would have no direction.
        let mut parent: Vec<usize> = (0..self.pool.len()).collect();
        fn root(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        let shorts: Vec<Vec<bool>> = plans
            .iter()
            .map(|(bi, ts, _)| {
                (0..ts.len().saturating_sub(1))
                    .map(|k| {
                        self.bases[*bi].curve.as_ref().is_some_and(|c| {
                            let (t0, t1) = (ts[k], ts[k + 1]);
                            let a = c.eval(t0);
                            a.dist(c.eval(t1)) <= tol::LINEAR
                                && a.dist(c.eval(0.5 * (t0 + t1))) <= tol::LINEAR
                        })
                    })
                    .collect()
            })
            .collect();
        for ((_, ts, vs), short) in plans.iter().zip(&shorts) {
            for k in 0..ts.len().saturating_sub(1) {
                if vs[k] != vs[k + 1] && short[k] {
                    let (a, b) = (root(&mut parent, vs[k]), root(&mut parent, vs[k + 1]));
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
        // Each merged vertex at its members' centroid where that holds them all: two corners just
        // over the tolerance apart are each within it of their middle.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for v in 0..parent.len() {
            let r = root(&mut parent, v);
            if r != v {
                groups.entry(r).or_insert_with(|| vec![r]).push(v);
            }
        }
        for (r, members) in &groups {
            let sum = members
                .iter()
                .fold(Vec3::ZERO, |acc, &m| acc + self.pool[m]);
            let c = sum * (1.0 / members.len() as f64);
            if members.iter().all(|&m| self.pool[m].dist(c) <= tol::LINEAR) {
                self.pool[*r] = c;
            }
        }
        for v in 0..parent.len() {
            let r = root(&mut parent, v);
            if r != v {
                let gap = self.pool[r].dist(self.pool[v]) + self.vtol[v];
                if gap > self.vtol[r] {
                    self.vtol[r] = gap.min(self.cap);
                    self.widest = self.widest.max(self.vtol[r]);
                    if self.vtol[r] > tol::LINEAR && !self.tolerant.contains(&r) {
                        self.tolerant.push(r);
                    }
                }
            }
        }
        for ((bi, ts, vs), short) in plans.into_iter().zip(shorts) {
            let vs: Vec<usize> = vs.into_iter().map(|v| root(&mut parent, v)).collect();
            for k in 0..ts.len().saturating_sub(1) {
                if vs[k] == vs[k + 1] && short[k] {
                    continue;
                }
                self.add_piece(bi, (ts[k], ts[k + 1]), vs[k], vs[k + 1]);
            }
        }
    }

    fn cut_inside_both(&self, bi: usize, p: Point3) -> bool {
        let BaseKind::Cut { fa, fb, .. } = self.bases[bi].kind else {
            return true;
        };
        self.faces[fa].dom.locate(p) != Where::Outside
            && self.faces[fb].dom.locate(p) != Where::Outside
    }

    fn add_piece(&mut self, bi: usize, range: (f64, f64), v0: usize, v1: usize) {
        let base = &self.bases[bi];
        let mut inside = Vec::new();
        let rank = match base.kind {
            BaseKind::Edge => self.edge_side(bi) as u8,
            // A tangent curve cuts both faces as any other: where the result keeps both, the
            // pieces touch along it, an edge four faces use; where it keeps one, the two halves of
            // that face merge again across it.
            BaseKind::Cut { fa, fb, .. } => {
                let c = base.curve.as_ref().expect("cut has a curve");
                let ends = [0.0, 0.5, 1.0].map(|f| c.eval(range.0 + (range.1 - range.0) * f));
                let (la, lb) = (
                    locate_piece(&self.faces[fa].dom, ends),
                    locate_piece(&self.faces[fb].dom, ends),
                );
                if la == Where::Outside || lb == Where::Outside {
                    return;
                }
                for (f, l) in [(fa, la), (fb, lb)] {
                    if matches!(l, Where::Inside(_)) {
                        inside.push(f);
                    }
                }
                2
            }
        };
        let id = self.pieces.len();
        self.pieces.push(Piece {
            base: bi,
            range,
            v0,
            v1,
            rank,
        });
        self.cut_faces.push(inside);
        self.base_pieces[bi].push(id);
    }

    /// Which input an edge base came from: the first input's edges come first.
    fn edge_side(&self, bi: usize) -> usize {
        usize::from(self.edge_base[1].first().is_some_and(|&b| bi >= b))
    }

    fn piece_point(&self, pi: usize, f: f64) -> Point3 {
        let p = &self.pieces[pi];
        let b = &self.bases[p.base];
        match &b.curve {
            Some(c) => c.eval(p.range.0 + (p.range.1 - p.range.0) * f),
            None => b.pole,
        }
    }

    /// How far from `q` on intersection base `bi`, running along `dir`, another curve within `t`
    /// of both its faces may lie and still be that intersection; zero for an edge base.
    fn crossing_spread(&self, bi: usize, q: Point3, dir: Vec3, t: f64) -> f64 {
        match self.bases[bi].kind {
            BaseKind::Cut { fa, fb, .. } => {
                let (sa, sb) = (&self.faces[fa].dom.surface, &self.faces[fb].dom.surface);
                crossing_spread(sa, sb, q, dir, t)
            }
            BaseKind::Edge => 0.0,
        }
    }

    /// Step 2b: pieces that run along each other become one, the lowest-ranked.
    fn dedup(&mut self) {
        let n = self.pieces.len();
        let mut parent: Vec<usize> = (0..n).collect();
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
        let mut by_ends: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, p) in self.pieces.iter().enumerate() {
            if self.bases[p.base].curve.is_none() {
                continue;
            }
            by_ends
                .entry((p.v0.min(p.v1), p.v0.max(p.v1)))
                .or_default()
                .push(i);
        }
        for group in by_ends.values() {
            for (k, &i) in group.iter().enumerate() {
                for &j in &group[k + 1..] {
                    if self.pieces[i].base == self.pieces[j].base {
                        continue;
                    }
                    let pj = &self.pieces[j];
                    let cj = self.bases[pj.base].curve.as_ref().expect("has a curve");
                    let (ti, tj) = (self.bases[self.pieces[i].base].tol, self.bases[pj.base].tol);
                    let within = tol::between(tj, ti);
                    let same = [0.25, 0.5, 0.75].iter().all(|&f| {
                        let (d, t) = csi::curve_distance(cj, pj.range, self.piece_point(i, f));
                        let (q, dir) = cj.d1(t);
                        d <= within
                            || d <= self.crossing_spread(self.pieces[i].base, q, dir, tj)
                            || d <= self.crossing_spread(pj.base, q, dir, ti)
                    });
                    if same {
                        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                        if ri != rj {
                            let key = |x: usize| (self.pieces[x].rank, x);
                            if key(ri) < key(rj) {
                                parent[rj] = ri;
                            } else {
                                parent[ri] = rj;
                            }
                        }
                    }
                }
            }
        }
        self.canon = (0..n).map(|i| find(&mut parent, i)).collect();
    }

    fn piece_geoms(&self) -> Vec<PieceGeom> {
        self.pieces
            .iter()
            .map(|p| {
                let b = &self.bases[p.base];
                PieceGeom {
                    curve: b.curve.clone(),
                    range: p.range,
                    v0: p.v0,
                    v1: p.v1,
                    tol: b.tol,
                }
            })
            .collect()
    }

    /// A trim along canonical piece `ci` from vertex `start`; `along` breaks the tie for a closed
    /// piece, by the direction the trim must run at the piece's middle.
    fn canon_trim(&self, ci: usize, start: usize, along: Option<Vec3>) -> Trim {
        let p = &self.pieces[ci];
        let b = &self.bases[p.base];
        let forward = if p.v0 != p.v1 {
            p.v0 == start
        } else {
            match (along, &b.curve) {
                (Some(d), Some(c)) => c.d1(0.5 * (p.range.0 + p.range.1)).1.dot(d) >= 0.0,
                _ => true,
            }
        };
        let (t0, t1, v0, v1) = if forward {
            (p.range.0, p.range.1, p.v0, p.v1)
        } else {
            (p.range.1, p.range.0, p.v1, p.v0)
        };
        Trim {
            piece: ci,
            forward,
            curve: b.curve.clone(),
            pole: self.pool[p.v0],
            t0,
            t1,
            map: UvMap::Projected,
            uv0: Vec2::ZERO,
            uv1: Vec2::ZERO,
            v0,
            v1,
        }
    }

    /// The face's boundary, its input coedges cut into pieces.
    fn boundary_trims(&self, fi: usize) -> Vec<Vec<Trim>> {
        let face = &self.faces[fi];
        let b = self.breps[face.side];
        let img = b.memo(face.id).images.get();
        let mut loops = Vec::new();
        for (li, &lp) in b.face(face.id).loops.iter().enumerate() {
            let mut trims = Vec::new();
            for (k, &c) in b.lp(lp).coedges.iter().enumerate() {
                let co = b.coedge(c);
                let bi = self.edge_base[face.side][co.edge.index()];
                let mut ps = self.base_pieces[bi].clone();
                if co.reversed {
                    ps.reverse();
                }
                let mut ts: Vec<f64> = Vec::new();
                for &pi in &ps {
                    let r = self.pieces[pi].range;
                    let (a, z) = if co.reversed { (r.1, r.0) } else { (r.0, r.1) };
                    if ts.is_empty() {
                        ts.push(a);
                    }
                    ts.push(z);
                }
                let uvs = match img {
                    Some(img) if ps.len() == 1 => {
                        let w = img.trims[li][k].whole;
                        vec![w.0, w.1]
                    }
                    _ => coedge_uv_at(b, c, &ts),
                };
                let whole = &face.dom.loops[li][k];
                for (k, &pi) in ps.iter().enumerate() {
                    let piece = &self.pieces[pi];
                    let ci = self.canon[pi];
                    let start = if co.reversed { piece.v1 } else { piece.v0 };
                    let mut tr = if self.bases[piece.base].curve.is_none() {
                        Trim {
                            piece: pi,
                            ..whole.clone()
                        }
                    } else {
                        let along = self.bases[piece.base].curve.as_ref().map(|c| {
                            let d = c.d1(0.5 * (piece.range.0 + piece.range.1)).1;
                            if co.reversed { -d } else { d }
                        });
                        let mut tr = self.canon_trim(ci, start, along);
                        if ci == pi && !matches!(whole.map, UvMap::Projected) {
                            tr.map = whole.map.clone();
                        }
                        tr
                    };
                    tr.uv0 = uvs[k];
                    tr.uv1 = uvs[k + 1];
                    trims.push(tr);
                }
            }
            loops.push(trims);
        }
        loops
    }

    /// A cut across face `fi` along canonical piece `ci`, its parameter-plane image continued
    /// from its middle, placed in the face's window.
    fn cut_trim(&self, fi: usize, ci: usize) -> Trim {
        let dom = &self.faces[fi].dom;
        let s = &dom.surface;
        let p = &self.pieces[ci];
        let mut tr = self.canon_trim(ci, p.v0, None);
        let tm = 0.5 * (tr.t0 + tr.t1);
        // A face that runs on past its seam's line (the seam cut away there) spans more than a
        // period, so only the image its loops enclose meets the boundary's nodes.
        let near = s.project(tr.point(tm));
        let mid = dom
            .inside_image(near)
            .unwrap_or_else(|| dom.shift_into(near));
        let march = |to: f64| {
            let mut uv = mid;
            let n = 16;
            for k in 1..=n {
                let t = tm + (to - tm) * k as f64 / n as f64;
                uv = tr.uv(s, t, uv);
            }
            uv
        };
        let (uv0, uv1) = (march(tr.t0), march(tr.t1));
        tr.uv0 = uv0;
        tr.uv1 = uv1;
        tr
    }

    /// Step 3: every face split into pieces.
    /// Also returns the pieces of poles that cuts end on, numbered after the other pieces, and
    /// whether each face came through whole: no cut crosses it and none of its edges is split or
    /// stands in for another, so its one piece has exactly the input face's geometry.
    #[allow(clippy::type_complexity)]
    fn split_faces(&self) -> Result<(Vec<Vec<Vec<Vec<Trim>>>>, Vec<PieceGeom>, Vec<bool>)> {
        let mut cuts: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); self.faces.len()];
        for (pi, fs) in self.cut_faces.iter().enumerate() {
            for &f in fs {
                cuts[f].insert(self.canon[pi]);
            }
        }
        for &(fa, fb) in &self.coincident {
            for (into, from) in [(fa, fb), (fb, fa)] {
                let from_face = &self.faces[from];
                let b = self.breps[from_face.side];
                let edges: BTreeSet<EdgeId> = b
                    .face(from_face.id)
                    .loops
                    .iter()
                    .flat_map(|&lp| b.lp(lp).coedges.iter().map(|&c| b.coedge(c).edge))
                    .collect();
                for e in edges {
                    let bi = self.edge_base[from_face.side][e.index()];
                    for &pi in &self.base_pieces[bi] {
                        if self.bases[bi].curve.is_none() {
                            continue;
                        }
                        let mid = self.piece_point(pi, 0.5);
                        if let Where::Inside(_) = self.faces[into].dom.locate(mid) {
                            cuts[into].insert(self.canon[pi]);
                        }
                    }
                }
            }
        }
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for (pi, &c) in self.canon.iter().enumerate() {
            members.entry(c).or_default().push(pi);
        }
        let mut out = Vec::with_capacity(self.faces.len());
        let mut poles: Vec<PieceGeom> = Vec::new();
        let mut intact = Vec::with_capacity(self.faces.len());
        for (fi, face_cuts) in cuts.iter().enumerate() {
            let mut boundary = self.boundary_trims(fi);
            let on_boundary: BTreeSet<usize> = boundary.iter().flatten().map(|t| t.piece).collect();
            let trims: Vec<Trim> = face_cuts
                .iter()
                .filter(|ci| !on_boundary.contains(ci))
                .map(|&ci| self.cut_trim(fi, ci))
                .collect();
            let next_id = self.pieces.len() + poles.len();
            let made = split_poles(&mut boundary, &trims, next_id);
            // Each of its edges whole and its own: where an edge lies along one of the other
            // input, the piece both take is one of the two, and on the other's the face runs
            // along another curve over another range, which its memo knows nothing of. A trim
            // names the piece taken, so the edge's own is looked up.
            let own = {
                let face = &self.faces[fi];
                let b = self.breps[face.side];
                b.face(face.id).loops.iter().all(|&lp| {
                    b.lp(lp).coedges.iter().all(|&c| {
                        let bi = self.edge_base[face.side][b.coedge(c).edge.index()];
                        let ps = &self.base_pieces[bi];
                        ps.len() == 1 && self.canon[ps[0]] == ps[0]
                    })
                })
            };
            let whole = trims.is_empty() && made.is_empty() && own;
            let sequence = |loops: &[Vec<Trim>]| -> Vec<Vec<usize>> {
                loops
                    .iter()
                    .map(|l| l.iter().map(|t| t.piece).collect())
                    .collect()
            };
            let before = whole.then(|| sequence(&boundary));
            poles.extend(made);
            let face = &self.faces[fi];
            let vertex_tol = |v: usize| self.vtol.get(v).copied().unwrap_or(tol::LINEAR);
            // A contact may end inside the face: a tangent curve, or an edge of the other solid
            // lying on it.
            let slit = |ci: usize| {
                members.get(&ci).is_some_and(|ps| {
                    ps.iter().any(|&pi| {
                        let bi = self.pieces[pi].base;
                        matches!(self.bases[bi].kind, BaseKind::Cut { tangent: true, .. })
                            || self.on_face.contains(&(bi, fi))
                    })
                })
            };
            let regions = arrange::split(
                &face.dom.surface,
                face.dom.reversed,
                boundary,
                trims,
                &vertex_tol,
                &slit,
            )
            .map_err(|e| self.at_faces(e, &[fi]))?;
            intact.push(match before {
                Some(b) => regions.len() == 1 && sequence(&regions[0]) == b,
                None => false,
            });
            out.push(regions);
        }
        Ok((out, poles, intact))
    }

    /// Step 4: each piece of face classified and kept or dropped.
    fn select(
        &self,
        frags: Vec<Vec<Vec<Vec<Trim>>>>,
        intact: &[bool],
        op: Op,
    ) -> Result<Vec<OutFace>> {
        let solids: [Vec<(&Domain, Aabb)>; 2] = [0, 1].map(|side| {
            self.face_of[side]
                .iter()
                .map(|&f| (&self.faces[f].dom, self.faces[f].bbox))
                .collect()
        });
        let reach = solids.each_ref().map(|faces| {
            let mut bb = Aabb::EMPTY;
            for (_, fb) in faces {
                bb.union(fb);
            }
            bb
        });
        let mut kept: Vec<OutFace> = Vec::new();
        let mut kept_doms: Vec<(usize, Domain)> = Vec::new();
        let groups = self.clear_groups();
        let mut group_class: HashMap<usize, Class> = HashMap::new();
        for (fi, pieces) in frags.into_iter().enumerate() {
            let face = &self.faces[fi];
            let b = self.breps[face.side];
            let names = &b.face(face.id).names;
            let apart = !overlap(&face.bbox, &reach[1 - face.side]);
            // A whole face clear of the other solid takes its group's class once a face of the
            // group has been classified.
            let group = groups[fi].filter(|_| pieces.len() == 1);
            let known = if apart {
                Some(Class::Out)
            } else {
                group.and_then(|g| group_class.get(&g).copied())
            };
            for loops in pieces {
                // A face apart from the other solid, or clear of it, is on one side of it whole,
                // and no contact, slit or tangent meeting the checks below look for can involve
                // it: kept or dropped without a domain.
                if let Some(class) = known {
                    let Some((flip, _)) = choose(op, face.side, class) else {
                        continue;
                    };
                    let mut out = OutFace {
                        surface: face.dom.surface.clone(),
                        reversed: face.dom.reversed,
                        names: names.clone(),
                        loops,
                        memo: intact[fi].then(|| b.shared_memo(face.id)),
                        side: Some(face.side),
                    };
                    if flip {
                        out.flip();
                    }
                    kept.push(out);
                    continue;
                }
                let dom = Domain::new(face.dom.surface.clone(), face.dom.reversed, loops);
                let class = if apart {
                    Class::Out
                } else {
                    let other = &solids[1 - face.side];
                    let uv = dom.interior_point().ok_or_else(|| {
                        Error::Numerical("a piece of a face has no interior point".into())
                    })?;
                    // Where a piece touches the other solid without crossing it (a tangent line
                    // is not a cut), a point on the contact is on both sides at once; any other
                    // point of the piece stands for it.
                    let touching = |c: Vec2| {
                        let q = dom.surface.eval(c.x, c.y);
                        other.iter().any(|(d, bb)| {
                            (0..3).all(|k| q[k] >= bb.min[k] && q[k] <= bb.max[k])
                                && d.locate(q) != Where::Outside
                        })
                    };
                    let at = if touching(uv) {
                        dom.interior_points()
                            .into_iter()
                            .find(|&c| !touching(c))
                            .unwrap_or(uv)
                    } else {
                        uv
                    };
                    let p = dom.surface.eval(at.x, at.y);
                    self.classify(fi, &dom, at, p, other)?
                };
                if let Some(g) = group
                    && !matches!(class, Class::On { .. })
                {
                    group_class.insert(g, class);
                }
                let Some((flip, extra)) = choose(op, face.side, class) else {
                    continue;
                };
                let mut all_names: BTreeSet<FaceName> = names.clone();
                if let Some(g) = extra {
                    let gf = &self.faces[g];
                    all_names.extend(self.breps[gf.side].face(gf.id).names.iter().cloned());
                }
                let mut out = OutFace {
                    surface: dom.surface.clone(),
                    reversed: dom.reversed,
                    names: all_names,
                    loops: dom.loops.clone(),
                    memo: intact[fi].then(|| b.shared_memo(face.id)),
                    side: Some(face.side),
                };
                if flip {
                    out.flip();
                }
                kept.push(out);
                kept_doms.push((fi, dom));
            }
        }
        Ok(kept)
    }

    /// Gives each kept face a vertex loop where the result touches it at a point inside it: where
    /// another kept face meets it at an isolated point, as two balls touch, or an edge of the
    /// result runs tangent to it, as a rim touching a plane does. The point is then a vertex both
    /// sides share, which joins them into one solid. The loops' pieces are added to `geoms`.
    fn touch_loops(&self, kept: &mut [OutFace], geoms: &mut Vec<PieceGeom>) {
        let touch: BTreeSet<usize> = self
            .touch_points
            .iter()
            .copied()
            .chain(self.met.iter().map(|m| m.0))
            .collect();
        for v in touch {
            let p = self.pool[v];
            let bounds = |f: &OutFace| f.loops.iter().flatten().any(|t| t.v0 == v || t.v1 == v);
            let on_boundary = kept.iter().any(bounds);
            let inside: Vec<(usize, Vec2)> = kept
                .iter()
                .enumerate()
                .filter(|(_, f)| {
                    !bounds(f) && csi::signed_distance(&f.surface, p).abs() <= tol::LINEAR
                })
                .filter_map(|(i, f)| {
                    let dom = Domain::new(f.surface.clone(), f.reversed, f.loops.clone());
                    match dom.locate(p) {
                        Where::Inside(uv) => Some((i, uv)),
                        _ => None,
                    }
                })
                .collect();
            if inside.len() + usize::from(on_boundary) < 2 {
                continue;
            }
            for (i, uv) in inside {
                // An uncut copy's memo describes the input face's loops, which this one outgrows.
                kept[i].memo = None;
                geoms.push(PieceGeom {
                    curve: None,
                    range: (0.0, 1.0),
                    v0: v,
                    v1: v,
                    tol: tol::LINEAR,
                });
                kept[i].loops.push(vec![Trim {
                    piece: geoms.len() - 1,
                    forward: true,
                    curve: None,
                    pole: p,
                    t0: 0.0,
                    t1: 1.0,
                    map: UvMap::Affine {
                        origin: uv,
                        dir: Vec2::ZERO,
                    },
                    uv0: uv,
                    uv1: uv,
                    v0: v,
                    v1: v,
                }]);
            }
        }
    }

    /// Whether a piece of face `fi` lies inside the other solid, read where one of its cuts
    /// crosses the inside of a face of that solid: the piece runs into the solid when the
    /// direction into it, square to the cut along its surface, opposes that face's outward
    /// normal. `None` when no cut crosses a face cleanly, so a ray must decide.
    fn local_class(&self, fi: usize, dom: &Domain) -> Option<bool> {
        for (li, l) in dom.loops.iter().enumerate() {
            for (k, tr) in l.iter().enumerate() {
                let Some(piece) = self.pieces.get(tr.piece) else {
                    continue;
                };
                let BaseKind::Cut {
                    fa,
                    fb,
                    tangent: false,
                } = self.bases[piece.base].kind
                else {
                    continue;
                };
                let g = if fa == fi {
                    fb
                } else if fb == fi {
                    fa
                } else {
                    continue;
                };
                let Some(c) = &tr.curve else { continue };
                let (q, dq) = c.d1(0.5 * (tr.t0 + tr.t1));
                let Where::Inside(uvg) = self.faces[g].dom.locate(q) else {
                    continue;
                };
                let pl = &dom.samples()[li][k];
                let uv = pl[pl.len() / 2];
                let travel = if tr.t1 >= tr.t0 { dq } else { -dq };
                // The loops keep the material on their left, seen from outside.
                let into = dom.outward(uv).cross(travel).unit_or_zero();
                let s = into.dot(self.faces[g].dom.outward(uvg));
                if s.abs() >= LOCAL_SINE {
                    return Some(s < 0.0);
                }
            }
        }
        None
    }

    fn classify(
        &self,
        fi: usize,
        dom: &Domain,
        uv: Vec2,
        p: Point3,
        other: &[(&Domain, Aabb)],
    ) -> Result<Class> {
        for &(fa, fb) in &self.coincident {
            let g = if fa == fi {
                fb
            } else if fb == fi {
                fa
            } else {
                continue;
            };
            let gd = &self.faces[g].dom;
            let at = match gd.locate(p) {
                // The piece is thinner than twice the tolerance here. It was cut along every
                // stretch of g's boundary that crosses it, so its interior lies wholly on one side.
                Where::Boundary => gd.exact_side(p).map_or(Where::Outside, Where::Inside),
                w => w,
            };
            if let Where::Inside(uvg) = at {
                let same = dom.outward(uv).dot(gd.outward(uvg)) > 0.0;
                return Ok(Class::On { same, partner: g });
            }
        }
        let inside = self
            .local_class(fi, dom)
            .or_else(|| classify::touching(&dom.surface, other, p))
            .or_else(|| classify::contains(other, p));
        match inside {
            Some(true) => Ok(Class::In),
            Some(false) => Ok(Class::Out),
            None => Err(Error::Numerical(
                "could not tell whether a face piece is inside the other solid".into(),
            )),
        }
    }
}

/// Splits each pole of a face's boundary where a cut arrives at it, so the cut meets the pole's
/// parameter-plane image at a node. The new pieces are numbered from `next_id`.
pub(crate) fn split_poles(
    boundary: &mut [Vec<Trim>],
    cuts: &[Trim],
    next_id: usize,
) -> Vec<PieceGeom> {
    let mut made = Vec::new();
    for lp in boundary.iter_mut() {
        let mut k = 0;
        while k < lp.len() {
            let tr = &lp[k];
            let UvMap::Affine { origin, dir } = tr.map else {
                k += 1;
                continue;
            };
            if tr.curve.is_some() || dir.x == 0.0 {
                k += 1;
                continue;
            }
            let v = tr.v0;
            let (lo, hi) = (tr.t0.min(tr.t1), tr.t0.max(tr.t1));
            let margin = 1e-9 * (hi - lo);
            let mut at: Vec<f64> = cuts
                .iter()
                .flat_map(|c| [(c.v0, c.uv0), (c.v1, c.uv1)])
                .filter(|&(w, _)| w == v)
                .map(|(_, uv)| (uv.x - origin.x) / dir.x)
                .filter(|&t| t > lo + margin && t < hi - margin)
                .collect();
            if at.is_empty() {
                k += 1;
                continue;
            }
            at.sort_by(f64::total_cmp);
            at.dedup_by(|a, b| (*a - *b).abs() <= margin);
            if tr.t1 < tr.t0 {
                at.reverse();
            }
            let mut ts = vec![tr.t0];
            ts.extend(at);
            ts.push(tr.t1);
            let parts: Vec<Trim> = ts
                .windows(2)
                .map(|w| {
                    let id = next_id + made.len();
                    made.push(PieceGeom {
                        curve: None,
                        range: (w[0].min(w[1]), w[0].max(w[1])),
                        v0: v,
                        v1: v,
                        tol: tol::LINEAR,
                    });
                    Trim {
                        piece: id,
                        forward: w[1] > w[0],
                        t0: w[0],
                        t1: w[1],
                        uv0: origin + dir * w[0],
                        uv1: origin + dir * w[1],
                        ..tr.clone()
                    }
                })
                .collect();
            let n = parts.len();
            lp.splice(k..k + 1, parts);
            k += n;
        }
    }
    made
}

/// The known vertices in `region` where `a` and `b` touch and share nothing else nearby.
fn known_touches(pool: &[Point3], a: &Surface, b: &Surface, region: &Aabb) -> Vec<contact::Touch> {
    pool.iter()
        .filter(|&&p| {
            (0..3).all(|k| p[k] >= region.min[k] && p[k] <= region.max[k])
                && csi::signed_distance(a, p).abs() <= tol::LINEAR
                && csi::signed_distance(b, p).abs() <= tol::LINEAR
        })
        .filter_map(|&p| contact::touch_at(a, b, p))
        .collect()
}

/// Points on both surfaces where tracing their intersection may start: the vertices already found
/// on both, and points reached from a grid over each face.
fn seeds_for<'p>(
    known: impl Iterator<Item = &'p Point3>,
    a: &Domain,
    b: &Domain,
    region: &Aabb,
) -> Vec<Point3> {
    let mut out: Vec<Point3> = known
        .copied()
        .filter(|&p| {
            (0..3).all(|k| p[k] >= region.min[k] && p[k] <= region.max[k])
                && csi::signed_distance(&a.surface, p).abs() <= tol::LINEAR
                && csi::signed_distance(&b.surface, p).abs() <= tol::LINEAR
        })
        .collect();
    let reach = 0.25 * region.diagonal();
    for (from, to) in [(a, b), (b, a)] {
        // A spline surface lies in its control points' box: a point farther than `reach` from
        // that box has no projection near enough to seed, so it is not projected at all.
        let hull = match &to.surface {
            Surface::Nurbs(n) => Some(padded(&n.hull_box(), reach)),
            _ => None,
        };
        let n = 8;
        for i in 0..=n {
            for j in 0..=n {
                let u = from.lo.x + (from.hi.x - from.lo.x) * (i as f64 + 0.37) / (n as f64 + 1.0);
                let v = from.lo.y + (from.hi.y - from.lo.y) * (j as f64 + 0.61) / (n as f64 + 1.0);
                let p = from.surface.eval(u, v);
                if hull
                    .as_ref()
                    .is_some_and(|h| !(0..3).all(|k| p[k] >= h.min[k] && p[k] <= h.max[k]))
                {
                    continue;
                }
                let uv = to.surface.project(p);
                let q = to.surface.eval(uv.x, uv.y);
                if p.dist(q) <= reach {
                    out.push(p.lerp(q, 0.5));
                }
            }
        }
    }
    out
}

/// Where a piece of face lies relative to the other solid.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    In,
    Out,
    /// On face `partner` of the other solid, with the same outward normal or the opposite one.
    On {
        same: bool,
        partner: usize,
    },
}

/// Whether the operation keeps a piece of a face of solid `side`, whether it is turned inside out,
/// and the coincident face whose names it also takes.
fn choose(op: Op, side: usize, class: Class) -> Option<(bool, Option<usize>)> {
    let a = side == 0;
    match (op, class) {
        (Op::Union, Class::Out) | (Op::Intersection, Class::In) => Some((false, None)),
        (
            Op::Union | Op::Intersection,
            Class::On {
                same: true,
                partner,
            },
        ) if a => Some((false, Some(partner))),
        (Op::Difference, Class::Out) if a => Some((false, None)),
        (Op::Difference, Class::On { same: false, .. }) if a => Some((false, None)),
        (Op::Difference, Class::In) if !a => Some((true, None)),
        _ => None,
    }
}

crate::codec::fields!(FaceImages {
    trims,
    samples,
    boxes,
    bbox
});
crate::codec::fields!(TrimImage { t, uv, whole });

#[cfg(test)]
mod tests;
