//! The validity checker: a solid is valid when its shells are closed oriented 2-manifolds, every
//! edge lies on both its faces within tolerance, and each loop closes in its face's parameters.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::{
    boolean::{
        classify,
        csi::{
            CurveBoxes, curve_surface_in, ray_surface, ray_surface_is_closed_form, signed_distance,
        },
        domain::{Domain, Where},
        face_domain,
    },
    geom::{Curve, NurbsSurface, Surface, distance_to},
    math::{Aabb, Point3, Vec2, Vec3},
    measure, tol,
    topo::{Brep, CoedgeId, EdgeId, FaceId, FaceMemo, ShellId, SolidId, VertexId},
};

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Issue {
    pub message: String,
}

/// Each face's trimmed domain, built the first time a check asks for it: a face of many loops is
/// asked for once per edge round it.
struct Domains<'a> {
    b: &'a Brep,
    built: std::cell::RefCell<HashMap<FaceId, std::rc::Rc<Domain>>>,
}

impl<'a> Domains<'a> {
    fn new(b: &'a Brep) -> Self {
        Domains {
            b,
            built: Default::default(),
        }
    }

    fn of(&self, f: FaceId) -> std::rc::Rc<Domain> {
        if let Some(d) = self.built.borrow().get(&f) {
            return d.clone();
        }
        let d = std::rc::Rc::new(face_domain(self.b, f));
        self.built.borrow_mut().insert(f, d.clone());
        d
    }
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

const SAMPLES: usize = 16;
/// A loop whose sampled outline encloses less than this times its perimeter squared is a sliver: a
/// circle encloses 1/4π of it, a strip a hundred times longer than it is wide 1/400.
const SLIVER_AREA: f64 = 1e-3;

/// Edge parameters along a coedge, in its own direction.
pub(crate) fn coedge_params(b: &Brep, c: CoedgeId, n: usize) -> Vec<f64> {
    let co = b.coedge(c);
    let (t0, t1) = b.edge(co.edge).range;
    let (a, z) = if co.reversed { (t1, t0) } else { (t0, t1) };
    spaced(a, z, n)
}

/// `n + 1` parameters from `a` to `z`, evenly spaced, the last exactly `z`: an end an ulp inside
/// the edge would be taken for an interior point.
fn spaced(a: f64, z: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| match i {
            _ if i == n => z,
            _ => a + (z - a) * i as f64 / n as f64,
        })
        .collect()
}

/// Where an intersection curve turns back on itself over `range`: its direction at two
/// neighbouring samples, a quarter of a knot span apart, more than a right angle apart. A fitted
/// branch turns little across one span, so that is a fold, and a face bounded by it touches itself.
/// Only fitted curves are tested; an imported spline may have a cusp its designer drew.
pub(crate) fn fold(c: &Curve, range: (f64, f64)) -> Option<crate::math::Point3> {
    let Curve::Intersection(ic) = c else {
        return None;
    };
    let sp = &ic.approx;
    let (t0, t1) = range;
    let mut cuts: Vec<f64> = vec![t0];
    cuts.extend(sp.breakpoints(t0, t1));
    cuts.push(t1);
    cuts.dedup();
    let mut prev: Option<crate::math::Vec3> = None;
    for w in cuts.windows(2) {
        for j in 0..4 {
            let t = w[0] + (w[1] - w[0]) * j as f64 / 4.0;
            let [p, d] = {
                let [p, d, _] = sp.eval_ders(t, 1);
                [p, d]
            };
            if d.norm() <= 1e-12 {
                continue;
            }
            if prev.is_some_and(|q| q.dot(d) < 0.0) {
                return Some(p);
            }
            prev = Some(d);
        }
    }
    None
}

fn uv_gap(b: &Brep, face: FaceId, p: Vec2, q: Vec2) -> f64 {
    let (_, su, sv) = b.face(face).surface.d1(p.x, p.y);
    // Each parameter at its own speed, so a gap is measured alike at any scale; where one stands
    // still beside the other, at a pole, a gap in it still opens the loop in the parameter plane
    // and is measured at the other's.
    let (a, c) = (su.norm(), sv.norm());
    let wu = if a <= 1e-9 * c { c } else { a };
    let wv = if c <= 1e-9 * a { a } else { c };
    (p.x - q.x).abs() * wu + (p.y - q.y).abs() * wv
}

/// Whether the two images `p` and `q` of the tolerant vertex that coedge `c` ends at are apart by
/// no more than its diameter, as the surface carries the step between them. Each parameter's own
/// speed overstates that step where the parameters run askew, as on a blend's spline: there the
/// two speeds partly cancel.
fn within_vertex(b: &Brep, face: FaceId, c: CoedgeId, p: Vec2, q: Vec2) -> bool {
    let tol = b.vertex_tolerance(b.coedge_end(c));
    if tol <= tol::LINEAR {
        return false;
    }
    carried(&b.face(face).surface, p, q) <= 2.0 * (tol - tol::LINEAR) + 10.0 * tol::LINEAR
}

/// The length of the step from `p` to `q` in parameters as the surface carries it at `p`.
fn carried(s: &Surface, p: Vec2, q: Vec2) -> f64 {
    let (_, su, sv) = s.d1(p.x, p.y);
    (su * (q.x - p.x) + sv * (q.y - p.y)).norm()
}

/// Checks every solid of `b` as one shape: solids of one result may touch, at vertices or along
/// faces within the tolerance, as two pieces of a cut that meet at a point do; one passing into
/// another or lying inside it makes it invalid.
pub fn check(b: &Brep) -> Vec<Issue> {
    check_with(b, false, None, true)
}

/// [`check`] without the search for faces passing through each other: the structure, the geometry
/// each entity lies on, and every loop's closure and direction. An imported solid is refused for
/// these; one passing through itself is kept and reported by [`self_crossings`].
pub(crate) fn check_structure(b: &Brep) -> Vec<Issue> {
    check_with(b, false, None, false)
}

/// Where two faces of one solid of `b` that share no edge meet, and how far they pass through each
/// other there: an imported mesh stitched with small folds passes through itself by microns.
pub fn self_crossings(b: &Brep) -> Vec<SelfCrossing> {
    crossings(b, None, true, &mut PairCache::default())
        .into_iter()
        .map(|x| SelfCrossing {
            a: x.a,
            b: x.b,
            at: x.at,
            depth: depth(b, x.a, x.b).max(depth(b, x.b, x.a)),
        })
        .collect()
}

/// Two faces of one solid that share no edge but meet at `at`, and how deep one passes through
/// the other in mm: zero where they only touch.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SelfCrossing {
    pub a: FaceId,
    pub b: FaceId,
    pub at: Point3,
    pub depth: f64,
}

/// How far face `a`'s boundary reaches through face `b`'s surface: the lesser of how far it lies on
/// each side, from its vertices and 8 points along each edge. A boundary on one side only touches.
fn depth(b: &Brep, fa: FaceId, fb: FaceId) -> f64 {
    let s = &b.face(fb).surface;
    let (mut above, mut below) = (0.0f64, 0.0f64);
    for &lp in &b.face(fa).loops {
        for &c in &b.lp(lp).coedges {
            let e = b.edge(b.coedge(c).edge);
            let pts: Vec<Point3> = match &e.curve {
                Some(cv) => (0..=8)
                    .map(|k| cv.eval(e.range.0 + (e.range.1 - e.range.0) * k as f64 / 8.0))
                    .collect(),
                None => vec![b.vertex(e.start).point],
            };
            for p in pts {
                let d = signed_distance(s, p);
                above = above.max(d);
                below = below.max(-d);
            }
        }
    }
    above.min(below)
}

/// Checks each solid of `b` on its own, as separate parts that may touch one another: a STEP
/// assembly's placed solids, where a part seated on a board shares the board's corners within
/// the tolerance. Within any one solid it finds everything [`check`] does.
pub fn check_each(b: &Brep) -> Vec<Issue> {
    check_with(b, true, None, true)
}

/// Checks a boolean's result, whose faces were cut from the inputs `sides` gives, by face. Each
/// input is a valid solid, so two faces cut from one meet only where the input faces they were cut
/// from did, along edges and at vertices the result keeps; only pairs from different inputs are
/// searched for faces passing through each other.
pub(crate) fn check_cut(b: &Brep, sides: &[Option<usize>]) -> Vec<Issue> {
    check_with(b, false, Some(sides), true)
}

/// The shell each vertex belongs to, by the faces whose edges end there.
fn vertex_shells(b: &Brep) -> HashMap<VertexId, ShellId> {
    let mut out = HashMap::new();
    for f in b.face_ids() {
        let Some(sh) = b.face(f).shell else { continue };
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let e = b.edge(b.coedge(c).edge);
                out.insert(e.start, sh);
                out.insert(e.end, sh);
            }
        }
    }
    out
}

/// Checks a sheet: open shells of faces in no solid (doc/design.md, "Sheets"). Every edge is used
/// once, on the sheet's boundary, or twice, by coedges that run opposite ways in faces of one shell;
/// everything else [`check`] finds of faces, loops, edges and vertices holds as for a solid.
pub fn check_sheet(b: &Brep) -> Vec<Issue> {
    checked(b, false, None, true, true)
}

fn check_with(b: &Brep, each: bool, sides: Option<&[Option<usize>]>, crossing: bool) -> Vec<Issue> {
    checked(b, each, sides, crossing, false)
}

fn checked(
    b: &Brep,
    each: bool,
    sides: Option<&[Option<usize>]>,
    crossing: bool,
    sheet: bool,
) -> Vec<Issue> {
    let mut out = Vec::new();
    let domains = Domains::new(b);
    let found = std::cell::Cell::new(0usize);
    let mut issue = |m: String| {
        found.set(found.get() + 1);
        out.push(Issue { message: m });
    };

    for e in b.edge_ids() {
        let edge = b.edge(e);
        if edge.range.0.partial_cmp(&edge.range.1) != Some(std::cmp::Ordering::Less) {
            issue(format!("edge {} has an empty range", e.index()));
            continue;
        }
        if let Some(c) = &edge.curve {
            if let Some(p) = fold(c, edge.range) {
                issue(format!(
                    "edge {} turns back on itself near ({:.6}, {:.6}, {:.6})",
                    e.index(),
                    p.x,
                    p.y,
                    p.z
                ));
            }
            for (t, v) in [(edge.range.0, edge.start), (edge.range.1, edge.end)] {
                let d = c.eval(t).dist(b.vertex(v).point);
                if d > b.vertex_tolerance(v) {
                    issue(format!(
                        "edge {} ends {d:e} mm from vertex {}",
                        e.index(),
                        v.index()
                    ));
                }
            }
        }
    }

    for v in b.vertex_ids() {
        let p = b.vertex(v).point;
        if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
            issue(format!("vertex {} is not a finite point", v.index()));
        }
    }
    // Vertices of different shells may coincide: the shells touch there.
    let solid_of = Some(vertex_shells(b));
    let apart = |v: VertexId, w: VertexId| {
        solid_of
            .as_ref()
            .is_some_and(|m| m.get(&v).is_some() && m.get(&v) != m.get(&w))
    };
    for (v, w) in coincident_vertices(b) {
        if apart(v, w) {
            continue;
        }
        issue(format!(
            "vertices {} and {} are within the tolerance, so the same point",
            v.index(),
            w.index()
        ));
    }
    for e in b.edge_ids() {
        let edge = b.edge(e);
        let Some(c) = &edge.curve else { continue };
        let ts: Vec<f64> = (0..=SAMPLES)
            .map(|i| edge.range.0 + (edge.range.1 - edge.range.0) * i as f64 / SAMPLES as f64)
            .collect();
        let len: f64 = ts.windows(2).map(|w| c.eval(w[0]).dist(c.eval(w[1]))).sum();
        if len.is_nan() || len <= tol::LINEAR {
            issue(format!(
                "edge {} is {len:e} mm long, within the tolerance",
                e.index()
            ));
        }
    }

    // Several fans of faces may meet at a vertex: pieces of the solid touch there.
    for e in b.edge_ids() {
        let u = b.edge_uses(e);
        if b.edge(e).is_degenerate() {
            if u.len() != 1 {
                issue(format!(
                    "degenerate edge {} is used {} times",
                    e.index(),
                    u.len()
                ));
            }
            continue;
        }
        // A sheet's boundary edge has one use, and its other edges two.
        if sheet && u.len() == 1 {
            continue;
        }
        if sheet && u.len() > 2 {
            issue(format!(
                "edge {} is used by {} coedges, not 1 or 2",
                e.index(),
                u.len()
            ));
            continue;
        }
        // Two uses, or pairs of them where pieces touch along the edge: each coedge and its
        // partner run opposite ways and bound one sheet.
        if u.len() < 2 || !u.len().is_multiple_of(2) {
            issue(format!(
                "edge {} is used by {} coedges, not 2 or pairs of them",
                e.index(),
                u.len()
            ));
            continue;
        }
        for &c in u {
            let Some(p) = b.partner(c).filter(|&p| b.partner(p) == Some(c)) else {
                issue(format!(
                    "edge {} is used by {} coedges without pairing them",
                    e.index(),
                    u.len()
                ));
                break;
            };
            if c > p {
                continue;
            }
            if b.coedge(c).reversed == b.coedge(p).reversed {
                issue(format!(
                    "edge {} is used twice in the same direction",
                    e.index()
                ));
            }
            if b.face(b.coedge_face(c)).shell != b.face(b.coedge_face(p)).shell {
                issue(format!(
                    "edge {} joins faces of different shells",
                    e.index()
                ));
            }
        }
        if u.len() > 2
            && let Some((f, g)) = lying_on_each_other(b, &domains, u)
        {
            issue(format!(
                "faces {} and {} lie on each other from edge {}",
                f.index(),
                g.index(),
                e.index()
            ));
        }
    }

    for f in b.face_ids() {
        let face = b.face(f);
        if face.loops.is_empty() {
            issue(format!("face {} has no loops", f.index()));
        }
        match face.shell {
            None => issue(format!("face {} is in no shell", f.index())),
            Some(s) if !sheet && b.shell(s).solid.is_none() => {
                issue(format!("shell {} is in no solid", s.index()))
            }
            Some(s) if sheet && b.shell(s).solid.is_some() => {
                issue(format!("shell {} of a sheet is in a solid", s.index()))
            }
            _ => {}
        }
        // A face's own geometry, once found sound, stays sound while the memo lives. A pole's
        // vertex is not part of that geometry, so a face with one is sampled every time.
        let memo = b.memo(f);
        let has_pole = face.loops.iter().any(|&lp| {
            b.lp(lp)
                .coedges
                .iter()
                .any(|&c| b.edge(b.coedge(c).edge).is_degenerate())
        });
        let sampled = has_pole || memo.clean.get().is_none();
        let before = found.get();
        for (li, &lp) in face.loops.iter().enumerate() {
            let cs = &b.lp(lp).coedges;
            if cs.is_empty() {
                issue(format!("face {} loop {li} is empty", f.index()));
                continue;
            }
            for (k, &c) in cs.iter().enumerate() {
                let next = cs[(k + 1) % cs.len()];
                if b.coedge_end(c) != b.coedge_start(next) {
                    issue(format!(
                        "face {} loop {li} breaks after coedge {}",
                        f.index(),
                        c.index()
                    ));
                }
            }
            if !sampled {
                continue;
            }
            let mut ends: Vec<(Vec2, Vec2)> = Vec::new();
            let mut ring: Vec<Vec2> = Vec::new();
            for &c in cs {
                let ts = coedge_params(b, c, SAMPLES);
                let uvs = b.coedge_uvs(c, &ts);
                ring.extend_from_slice(&uvs[..uvs.len() - 1]);
                let e = b.coedge(c).edge;
                let edge = b.edge(e);
                let within = match edge.curve {
                    Some(_) => b.edge_tolerance(e),
                    None => b.vertex_tolerance(edge.start),
                };
                for (t, uv) in ts.iter().zip(&uvs) {
                    let want = match &edge.curve {
                        Some(curve) => curve.eval(*t),
                        None => b.vertex(edge.start).point,
                    };
                    let d = face.surface.eval(uv.x, uv.y).dist(want);
                    if !tol::within(d, within, want) {
                        issue(format!(
                            "coedge {} of face {} is {d:e} mm off its surface",
                            c.index(),
                            f.index()
                        ));
                        break;
                    }
                }
                ends.push((uvs[0], uvs[uvs.len() - 1]));
            }
            for k in 0..ends.len() {
                let (end, start) = (ends[k].1, ends[(k + 1) % ends.len()].0);
                // Two coedges' images of a tolerant vertex are each within its sphere, which
                // spans more parameter where the surface moves slower than one mm per unit.
                let mut spread = 2.0 * (b.vertex_tolerance(b.coedge_end(cs[k])) - tol::LINEAR);
                if spread > 0.0 {
                    let (_, su, sv) = face.surface.d1(end.x, end.y);
                    let slowest = su.norm().min(sv.norm()).clamp(1e-12, 1.0);
                    spread *= std::f64::consts::SQRT_2 / slowest;
                }
                if uv_gap(b, f, end, start) > 10.0 * tol::LINEAR + spread
                    && !within_vertex(b, f, cs[k], end, start)
                {
                    issue(format!(
                        "face {} ({}) loop {li} does not close in parameters after coedge {}: \
                         ({:.6}, {:.6}) to ({:.6}, {:.6})",
                        f.index(),
                        face.surface.kind(),
                        cs[k].index(),
                        end.x,
                        end.y,
                        start.x,
                        start.y
                    ));
                }
            }
            let area: f64 = (0..ring.len())
                .map(|i| ring[i].cross(ring[(i + 1) % ring.len()]))
                .sum();
            // A sliver's chords sag further than it is wide, so its sampled outline can enclose
            // either sign: there the exact parameter curves decide.
            let perimeter: f64 = (0..ring.len())
                .map(|i| ring[i].dist(ring[(i + 1) % ring.len()]))
                .sum();
            let area = if area.abs() < SLIVER_AREA * perimeter * perimeter {
                crate::measure::loop_area(b, lp)
            } else {
                area
            };
            let want_positive = (li == 0) != face.reversed;
            if !b.is_vertex_loop(lp) && (area > 0.0) != want_positive {
                issue(format!("face {} loop {li} runs the wrong way", f.index()));
            }
        }
        if sampled
            && !has_pole
            && found.get() == before
            && face.loops.len() > 1
            && let Some(li) = stray_loop(&domains, f)
        {
            issue(format!(
                "face {} loop {li} lies outside the region its other loops bound",
                f.index()
            ));
        }
        if sampled && !has_pole && found.get() == before {
            let _ = memo.clean.set(());
        }
    }

    for s in b.solid_ids() {
        if b.solid(s).shells.is_empty() {
            issue(format!("solid {} has no shells", s.index()));
        }
    }
    let mut cache = PairCache::default();
    // A boolean's result of one sheet is settled by a ray where one settles it; the volume
    // integral settles every other solid, and any the ray leaves open.
    let mut outward: HashSet<SolidId> = HashSet::new();
    if out.is_empty() {
        for s in b.solid_ids() {
            if sides.is_some() && faces_outward(b, &domains, &mut cache, s) {
                outward.insert(s);
                continue;
            }
            let v = measure::solid_volume_sign(b, s);
            if v.is_nan() || v <= 0.0 {
                out.push(Issue {
                    message: format!("solid {} has volume {v}", s.index()),
                });
            }
        }
    }
    if out.is_empty() {
        for (s, sh, outer) in turned_shells_but(b, &outward) {
            let what = if outer { "an outer sheet" } else { "a void" };
            out.push(Issue {
                message: format!(
                    "shell {} of solid {} is {what} turned inside out",
                    sh.index(),
                    s.index()
                ),
            });
        }
    }
    if out.is_empty() && crossing {
        // Faces of different shells that meet without passing into each other only touch, at
        // points or along curves. Faces on one surface meeting inside either share an area, which
        // makes the two shells one: the regularized result has that area inside it.
        let inside = |f: FaceId, p: Point3| {
            matches!(
                domains.of(f).locate(p),
                crate::boolean::domain::Where::Inside(_)
            )
        };
        // Whether some point inside face `f` lies inside face `g` too.
        let overlaps = |f: FaceId, g: FaceId| {
            let d = domains.of(f);
            d.interior_point()
                .is_some_and(|uv| inside(g, d.surface.eval(uv.x, uv.y)))
        };
        let touching = |x: &Crossing| {
            b.face(x.a).shell != b.face(x.b).shell
                && depth(b, x.a, x.b).max(depth(b, x.b, x.a)) <= tol::LINEAR
                && !(crate::boolean::ssi::same_domain(&b.face(x.a).surface, &b.face(x.b).surface)
                    && (inside(x.a, x.at)
                        || inside(x.b, x.at)
                        || overlaps(x.a, x.b)
                        || overlaps(x.b, x.a)))
        };
        for x in crossings(b, sides, each, &mut cache)
            .into_iter()
            .filter(|x| !touching(x))
        {
            out.push(Issue {
                message: format!(
                    "faces {} and {} share no edge but meet at ({:.6}, {:.6}, {:.6}): the solid \
                     passes through itself",
                    x.a.index(),
                    x.b.index(),
                    x.at.x,
                    x.at.y,
                    x.at.z
                ),
            });
        }
    }
    if out.is_empty() {
        for (s, sh) in stray_voids(b) {
            out.push(Issue {
                message: format!(
                    "void shell {} of solid {} lies outside its outer shell",
                    sh.index(),
                    s.index()
                ),
            });
        }
    }
    if out.is_empty() && !each {
        for (s, t) in nested(b) {
            out.push(Issue {
                message: format!("solid {} lies inside solid {}", s.index(), t.index()),
            });
        }
    }
    out
}

/// Two faces beside each other across an empty wedge round an edge more than two faces use, where
/// every interior point of one lies on the other's surface inside it: the wedge has no width, and
/// the faces lie inside the solid, as a union that kept both of two coincident faces leaves them.
/// Faces that touch along the edge part away from it; partners lying on each other bound a sliver.
fn lying_on_each_other(b: &Brep, domains: &Domains, uses: &[CoedgeId]) -> Option<(FaceId, FaceId)> {
    // Faces lying on each other leave the edge the same way, and two analytic surfaces other than
    // planes do only where they are one surface: both are decided without the faces' domains.
    let apart = |c: CoedgeId, n: CoedgeId| {
        let (s, t) = (
            &b.face(b.coedge_face(c)).surface,
            &b.face(b.coedge_face(n)).surface,
        );
        let planes = matches!((s, t), (Surface::Plane(_), Surface::Plane(_)));
        let other =
            analytic(s) && analytic(t) && !planes && !crate::boolean::ssi::same_domain(s, t);
        other
            || into_face(b, c)
                .zip(into_face(b, n))
                .is_some_and(|(x, y)| x.dot(y) <= 0.0)
    };
    let on = |f: FaceId, g: FaceId| {
        let (d, o) = (domains.of(f), domains.of(g));
        let pts = d.interior_points();
        !pts.is_empty()
            && pts.iter().all(|uv| {
                let p = d.surface.eval(uv.x, uv.y);
                signed_distance(&o.surface, p).abs() <= tol::LINEAR
                    && matches!(o.locate(p), Where::Inside(_))
            })
    };
    uses.iter().find_map(|&c| {
        let n = b.beside(c)?;
        let (f, g) = (b.coedge_face(c), b.coedge_face(n));
        (c < n && f != g && !apart(c, n) && (on(f, g) || on(g, f))).then_some((f, g))
    })
}

/// The unit direction in which coedge `c`'s face leaves the middle of its edge, tangent to the
/// surface: left of the coedge in the parameter plane, or right on a reversed face.
fn into_face(b: &Brep, c: CoedgeId) -> Option<Vec3> {
    let co = b.coedge(c);
    let (t0, t1) = b.edge(co.edge).range;
    let (m, h) = (0.5 * (t0 + t1), 1e-4 * (t1 - t0));
    let ts = if co.reversed {
        [m + h, m, m - h]
    } else {
        [m - h, m, m + h]
    };
    let uvs = b.coedge_uvs(c, &ts);
    let face = b.face(b.coedge_face(c));
    let d = uvs[2] - uvs[0];
    let left = if face.reversed {
        Vec2::new(d.y, -d.x)
    } else {
        Vec2::new(-d.y, d.x)
    };
    let (_, su, sv) = face.surface.d1(uvs[1].x, uvs[1].y);
    (su * left.x + sv * left.y).unit()
}

/// The first inner loop of face `f` that the region bounded by its other loops does not hold,
/// tested at a point halfway along the loop's first coedge. A loop that winds once round a period,
/// as each rim of a band round a cylinder, bounds nothing in the parameters, so such a face has no
/// region to test.
fn stray_loop(domains: &Domains, f: FaceId) -> Option<usize> {
    let d = domains.of(f);
    let periods = [d.surface.u_period(), d.surface.v_period()];
    let wraps = d.samples().iter().any(|l| {
        let mut run = [0.0f64; 2];
        for pl in l {
            let (a, z) = (pl[0], pl[pl.len() - 1]);
            run[0] += z.x - a.x;
            run[1] += z.y - a.y;
        }
        (0..2).any(|k| periods[k].is_some_and(|p| run[k].abs() > 0.5 * p))
    });
    if wraps {
        return None;
    }
    (1..d.loops.len()).find(|&li| {
        let pl = &d.samples()[li][0];
        !d.inside_without(pl[pl.len() / 2], li)
    })
}

/// Void shells, with their solids, that a point inside one of their faces finds outside the
/// solid's outer shell.
fn stray_voids(b: &Brep) -> Vec<(SolidId, ShellId)> {
    let mut out = Vec::new();
    for s in b.solid_ids() {
        let shells = &b.solid(s).shells;
        let n_outer = b.solid(s).outer;
        if shells.len() <= n_outer {
            continue;
        }
        let outer: Vec<(Domain, Aabb)> = shells[..n_outer]
            .iter()
            .flat_map(|&sh| b.shell(sh).faces.iter())
            .map(|&f| (face_domain(b, f), padded_box(measure::face_box(b, f))))
            .collect();
        let faces: Vec<(&Domain, Aabb)> = outer.iter().map(|(d, bb)| (d, *bb)).collect();
        for &sh in &shells[n_outer..] {
            let probe = b.shell(sh).faces.iter().find_map(|&f| {
                let d = face_domain(b, f);
                d.interior_point().map(|uv| d.surface.eval(uv.x, uv.y))
            });
            if probe.is_some_and(|p| classify::contains(&faces, p) == Some(false)) {
                out.push((s, sh));
            }
        }
    }
    out
}

/// The shells, with their solids and whether each is an outer sheet, whose faces enclose volume of
/// the wrong sign: an outer sheet's is positive, a void's negative.
pub(crate) fn turned_shells(b: &Brep) -> Vec<(SolidId, ShellId, bool)> {
    turned_shells_but(b, &HashSet::new())
}

/// [`turned_shells`] of every solid but those of `settled`.
fn turned_shells_but(b: &Brep, settled: &HashSet<SolidId>) -> Vec<(SolidId, ShellId, bool)> {
    let mut out = Vec::new();
    for s in b.solid_ids().filter(|s| !settled.contains(s)) {
        let solid = b.solid(s);
        for (k, &sh) in solid.shells.iter().enumerate() {
            let outer = k < solid.outer;
            let v = measure::enclosed_volume_sign(b, &b.shell(sh).faces);
            if (v > 0.0) != outer {
                out.push((s, sh, outer));
            }
        }
    }
    out
}

/// How many faces of a sheet a ray is started from before the volume integral is left to settle
/// which way the sheet faces.
const RAY_STARTS: usize = 4;

/// Whether solid `s` is one sheet that a ray shows facing outward. The sheet is closed and its
/// faces are joined through partners that run opposite ways, so where its faces meet only along
/// the edges and at the vertices they share, it has an inside and an outside and every face's
/// normal points to the same one: the volume it encloses is positive when one normal points out.
/// `false` wherever that is not certain, and for a solid of several shells.
fn faces_outward(b: &Brep, domains: &Domains, cache: &mut PairCache, s: SolidId) -> bool {
    let solid = b.solid(s);
    let &[sh] = solid.shells.as_slice() else {
        return false;
    };
    let faces = &b.shell(sh).faces;
    if solid.outer != 1 || !one_sheet(b, faces) {
        return false;
    }
    let boxes: Vec<Aabb> = faces.iter().map(|&f| cache.face_box(b, f)).collect();
    let finite = |bb: &Aabb| (0..3).all(|k| bb.min[k].is_finite() && bb.max[k].is_finite());
    if !boxes.iter().all(finite) {
        return false;
    }
    // A face that reaches the sheet's own box is likeliest to see nothing along its normal.
    let mut starts: Vec<usize> = Vec::new();
    for k in 0..3 {
        let least = |i: &usize, j: &usize| boxes[*i].min[k].total_cmp(&boxes[*j].min[k]);
        let most = |i: &usize, j: &usize| boxes[*i].max[k].total_cmp(&boxes[*j].max[k]);
        let reach = [
            (0..faces.len()).min_by(least),
            (0..faces.len()).max_by(most),
        ];
        for i in reach.into_iter().flatten() {
            if !starts.contains(&i) && ray_surface_is_closed_form(&b.face(faces[i]).surface) {
                starts.push(i);
            }
        }
    }
    starts
        .into_iter()
        .take(RAY_STARTS)
        .find_map(|i| side_seen_from(b, domains, faces, &boxes, i))
        == Some(true)
}

/// Whether every face of `faces` is reached from the first through coedges' partners.
fn one_sheet(b: &Brep, faces: &[FaceId]) -> bool {
    let Some(&first) = faces.first() else {
        return false;
    };
    let mut seen: HashSet<FaceId> = HashSet::from([first]);
    let mut stack = vec![first];
    while let Some(f) = stack.pop() {
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                if let Some(p) = b.partner(c) {
                    let g = b.coedge_face(p);
                    if seen.insert(g) {
                        stack.push(g);
                    }
                }
            }
        }
    }
    seen.len() == faces.len() && faces.iter().all(|f| seen.contains(f))
}

/// Whether the outward normal of face `faces[start]` points out of the closed sheet `faces`, by
/// the ray along it from a point inside the face. Just off that point the sheet winds round the
/// ray's foot once for each face the ray leaves the material through, less once for each it
/// enters through, and round a point outside it winds no times. `Some(false)` where it winds once
/// the other way, so the normal points in, and `None` where the ray decides nothing: it passes
/// the box of a face whose crossings no closed form gives, meets a face on its boundary or at a
/// tangent, or another face lies on the start.
fn side_seen_from(
    b: &Brep,
    domains: &Domains,
    faces: &[FaceId],
    boxes: &[Aabb],
    start: usize,
) -> Option<bool> {
    let from = domains.of(faces[start]);
    let uv = point_inside(&from)?;
    let (p, d) = (from.surface.eval(uv.x, uv.y), from.outward(uv));
    let (mut winds, mut at_start) = (0i64, 0usize);
    for (i, (&f, bb)) in faces.iter().zip(boxes).enumerate() {
        let Some((lo, hi)) = classify::ray_box(p, d, bb) else {
            continue;
        };
        let surface = &b.face(f).surface;
        if !ray_surface_is_closed_form(surface) {
            return None;
        }
        for t in ray_surface(surface, p, d, hi.max(0.0)) {
            if !t.is_finite() {
                return None;
            }
            if t.abs() <= tol::LINEAR {
                if i != start {
                    return None;
                }
                at_start += 1;
                continue;
            }
            if t < 0.0 || t < lo - tol::LINEAR || t > hi + tol::LINEAR {
                continue;
            }
            let dom = domains.of(f);
            match dom.locate(p + d * t) {
                Where::Outside => {}
                Where::Boundary => return None,
                Where::Inside(at) => {
                    let across = dom.outward(at).dot(d);
                    if across.is_nan() || across.abs() < tol::TANGENT {
                        return None;
                    }
                    winds += if across > 0.0 { 1 } else { -1 };
                }
            }
        }
    }
    match (at_start, winds) {
        (1, 0) => Some(true),
        (1, -1) => Some(false),
        _ => None,
    }
}

/// The parameters of a point inside the face of `dom`: the middle of its first loop's sampled
/// outline or a point between that and the outline, where the exact winding number finds one of
/// them inside, and otherwise [`Domain::interior_point`], which searches for one.
fn point_inside(dom: &Domain) -> Option<Vec2> {
    let outline: Vec<Vec2> = dom.samples().first()?.iter().flatten().copied().collect();
    let n = outline.len();
    let middle = outline.iter().fold(Vec2::ZERO, |a, &p| a + p) * (1.0 / n.max(1) as f64);
    let towards = |k: usize, by: f64| middle + (outline[k * n / 3] - middle) * by;
    let tries = [
        middle,
        towards(0, 0.5),
        towards(1, 0.5),
        towards(2, 0.5),
        towards(0, 0.9),
        towards(1, 0.9),
        towards(2, 0.9),
    ];
    let inside = |uv: &Vec2| match dom.locate(dom.surface.eval(uv.x, uv.y)) {
        Where::Inside(at) => Some(at),
        _ => None,
    };
    tries
        .iter()
        .find_map(inside)
        .or_else(|| dom.interior_point())
}

/// For each void shell of solid `s`, the outer sheet it lies inside, as an index into its outer
/// shells: found by a point inside one of its faces, the first sheet where that is not found.
pub(crate) fn void_hosts(b: &Brep, s: SolidId) -> Vec<usize> {
    let shells = &b.solid(s).shells;
    let n_outer = b.solid(s).outer;
    let sheets: Vec<Vec<(Domain, Aabb)>> = shells[..n_outer]
        .iter()
        .map(|&sh| {
            b.shell(sh)
                .faces
                .iter()
                .map(|&f| (face_domain(b, f), padded_box(measure::face_box(b, f))))
                .collect()
        })
        .collect();
    shells[n_outer..]
        .iter()
        .map(|&sh| {
            let probe = b.shell(sh).faces.iter().find_map(|&f| {
                let d = face_domain(b, f);
                d.interior_point().map(|uv| d.surface.eval(uv.x, uv.y))
            });
            let inside = |k: &usize| {
                let faces: Vec<(&Domain, Aabb)> =
                    sheets[*k].iter().map(|(d, bb)| (d, *bb)).collect();
                probe.is_some_and(|p| classify::contains(&faces, p) == Some(true))
            };
            (0..n_outer).find(inside).unwrap_or(0)
        })
        .collect()
}

/// Pairs of solids whose faces do not meet but one of which lies inside the other, found by a
/// point inside a face of the first.
fn nested(b: &Brep) -> Vec<(SolidId, SolidId)> {
    let solids: Vec<SolidId> = b.solid_ids().collect();
    if solids.len() < 2 {
        return Vec::new();
    }
    let parts: Vec<Vec<(Domain, Aabb)>> = solids
        .iter()
        .map(|&s| {
            b.solid_faces(s)
                .into_iter()
                .map(|f| (face_domain(b, f), padded_box(measure::face_box(b, f))))
                .collect()
        })
        .collect();
    let boxes: Vec<Aabb> = parts
        .iter()
        .map(|fs| {
            fs.iter().fold(Aabb::EMPTY, |mut acc, (_, bb)| {
                acc.union(bb);
                acc
            })
        })
        .collect();
    let mut out = Vec::new();
    for (i, inner) in parts.iter().enumerate() {
        let Some(probe) = inner
            .iter()
            .find_map(|(d, _)| d.interior_point().map(|uv| d.surface.eval(uv.x, uv.y)))
        else {
            continue;
        };
        for (j, outer) in parts.iter().enumerate() {
            if i == j || !in_box(probe, &boxes[j]) {
                continue;
            }
            let faces: Vec<(&Domain, Aabb)> = outer.iter().map(|(d, bb)| (d, *bb)).collect();
            if classify::contains(&faces, probe) == Some(true) {
                out.push((solids[i], solids[j]));
            }
        }
    }
    out
}

/// A point where two faces that share no edge meet, away from any vertex they share: the solid
/// passes through itself there, or touches itself.
struct Crossing {
    a: FaceId,
    b: FaceId,
    at: Point3,
}

fn padded_box(bb: Aabb) -> Aabb {
    let pad = Vec3::new(tol::LINEAR, tol::LINEAR, tol::LINEAR);
    Aabb {
        min: bb.min - pad,
        max: bb.max + pad,
    }
}

fn boxes_meet(a: &Aabb, b: &Aabb) -> bool {
    (0..3).all(|k| a.min[k] <= b.max[k] && b.min[k] <= a.max[k])
}

fn in_box(p: Point3, bb: &Aabb) -> bool {
    (0..3).all(|k| p[k] >= bb.min[k] && p[k] <= bb.max[k])
}

/// What the pair tests of one check share: each face's domain, built when a point is first
/// located in it, each edge's box and its boxes over stretches, and each edge's meeting with each
/// face.
#[derive(Default)]
struct PairCache {
    domains: HashMap<FaceId, Domain>,
    edge_boxes: HashMap<EdgeId, Aabb>,
    curve_boxes: HashMap<EdgeId, CurveBoxes>,
    meetings: HashMap<(EdgeId, FaceId), Option<Point3>>,
}

impl PairCache {
    fn edge_box(&mut self, b: &Brep, e: EdgeId) -> Option<Aabb> {
        let edge = b.edge(e);
        let curve = edge.curve.as_ref()?;
        Some(
            *self
                .edge_boxes
                .entry(e)
                .or_insert_with(|| padded_box(curve.bbox(edge.range.0, edge.range.1))),
        )
    }

    /// A box that holds face `f`: its loose box where that is already worked out, otherwise one
    /// cheaper to find, its edges' boxes and, where the surface can bulge past them, the box of a
    /// B-spline's control points or of a whole sphere or torus.
    fn face_box(&mut self, b: &Brep, f: FaceId) -> Aabb {
        if let Some(bb) = b.memo(f).loose_box.get() {
            return padded_box(*bb);
        }
        let s = &b.face(f).surface;
        let mut bb = match s {
            Surface::Plane(_) | Surface::Cylinder(_) | Surface::Cone(_) | Surface::Extrusion(_) => {
                Aabb::EMPTY
            }
            Surface::Nurbs(n) => n.hull_box(),
            Surface::Sphere(sp) => {
                let r = Vec3::new(sp.radius, sp.radius, sp.radius);
                Aabb {
                    min: sp.frame.origin - r,
                    max: sp.frame.origin + r,
                }
            }
            Surface::Torus(t) => {
                let r = t.major + t.minor;
                let r = Vec3::new(r, r, r);
                Aabb {
                    min: t.frame.origin - r,
                    max: t.frame.origin + r,
                }
            }
            _ => return padded_box(measure::face_box(b, f)),
        };
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let e = b.coedge(c).edge;
                match self.edge_box(b, e) {
                    Some(eb) => bb.union(&eb),
                    None => bb.add(b.vertex(b.edge(e).start).point),
                }
            }
        }
        padded_box(bb)
    }
}

/// Faces that share no edge but meet: an edge of one crosses or touches the other away from the
/// vertices the two share. Face pairs are found by their boxes, sorted along `x`. A pair whose
/// memos were found apart before is not tested again, and neither is a pair `sides` gives one
/// input of a boolean. With `each`, only faces of one solid are paired, as an import's parts may
/// rest on one another. Two curved faces that meet only along a closed curve inside both are not
/// found: no edge reaches it.
fn crossings(
    b: &Brep,
    sides: Option<&[Option<usize>]>,
    each: bool,
    cache: &mut PairCache,
) -> Vec<Crossing> {
    let faces: Vec<FaceId> = b.face_ids().collect();
    let solid_of = |f: FaceId| b.face(f).shell.and_then(|s| b.shell(s).solid);
    let boxes: Vec<Aabb> = faces.iter().map(|&f| cache.face_box(b, f)).collect();
    let mut order: Vec<usize> = (0..faces.len()).collect();
    order.sort_by(|&i, &j| boxes[i].min.x.total_cmp(&boxes[j].min.x));

    let mut by_edge: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    let mut verts: HashMap<FaceId, Vec<VertexId>> = HashMap::new();
    for &f in &faces {
        let vs = verts.entry(f).or_default();
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let e = b.coedge(c).edge;
                by_edge.entry(e).or_default().push(f);
                vs.push(b.edge(e).start);
                vs.push(b.edge(e).end);
            }
        }
    }
    let adjacent: HashSet<(FaceId, FaceId)> = by_edge
        .values()
        .flat_map(|fs| fs.iter().flat_map(|&x| fs.iter().map(move |&y| (x, y))))
        .collect();
    let apart = |ma: &FaceMemo, mb: &FaceMemo| {
        for (m, other) in [(ma, mb.id), (mb, ma.id)] {
            if let Ok(mut s) = m.apart.lock() {
                s.insert(other);
            }
        }
    };

    let mut out = Vec::new();
    for (k, &i) in order.iter().enumerate() {
        for &j in &order[k + 1..] {
            if boxes[j].min.x > boxes[i].max.x {
                break;
            }
            let (fa, fb) = (faces[i], faces[j]);
            if !boxes_meet(&boxes[i], &boxes[j])
                || adjacent.contains(&(fa, fb))
                || (each && solid_of(fa) != solid_of(fb))
            {
                continue;
            }
            let (ma, mb) = (b.memo(fa), b.memo(fb));
            if sides.is_some_and(|s| s[fa.index()].is_some() && s[fa.index()] == s[fb.index()]) {
                apart(ma, mb);
                continue;
            }
            if ma.apart.lock().is_ok_and(|s| s.contains(&mb.id)) {
                continue;
            }
            // Where either face's box is clear of the other's surface, they cannot meet.
            let (sa, sb) = (&b.face(fa).surface, &b.face(fb).surface);
            if (analytic(sb) && clear_of(&boxes[i], sb))
                || (analytic(sa) && clear_of(&boxes[j], sa))
            {
                apart(ma, mb);
                continue;
            }
            let mut hit = None;
            'pair: for (x, y, ybox) in [(fa, fb, &boxes[j]), (fb, fa, &boxes[i])] {
                for &lp in &b.face(x).loops {
                    for &c in &b.lp(lp).coedges {
                        let e = b.coedge(c).edge;
                        let Some(eb) = cache.edge_box(b, e) else {
                            continue;
                        };
                        if !boxes_meet(&eb, ybox) {
                            continue;
                        }
                        hit = match cache.meetings.get(&(e, y)) {
                            Some(&m) => m,
                            None => {
                                let m = edge_meets_face(b, e, y, ybox, &verts[&y], cache);
                                cache.meetings.insert((e, y), m);
                                m
                            }
                        };
                        if hit.is_some() {
                            break 'pair;
                        }
                    }
                }
            }
            if hit.is_none() {
                hit = crossing_from_shared_vertex(b, (fa, fb), (&verts[&fa], &verts[&fb]), cache);
            }
            match hit {
                Some(at) => out.push(Crossing { a: fa, b: fb, at }),
                None => apart(ma, mb),
            }
        }
    }
    out
}

/// The distance from `p` to `s`, or a lower bound of it: the analytic surfaces' signed distances
/// change no faster than the point moves, so they bound the distance to the surface from below.
fn gap(s: &Surface, p: Point3) -> f64 {
    match s {
        Surface::Plane(_)
        | Surface::Cylinder(_)
        | Surface::Cone(_)
        | Surface::Sphere(_)
        | Surface::Torus(_) => signed_distance(s, p).abs(),
        _ => distance_to(s, p),
    }
}

fn analytic(s: &Surface) -> bool {
    matches!(
        s,
        Surface::Plane(_)
            | Surface::Cylinder(_)
            | Surface::Cone(_)
            | Surface::Sphere(_)
            | Surface::Torus(_)
    )
}

/// Whether no point of `bb` lies within the tolerance of `s`: the box lies wholly on one side of
/// a plane, or the surface is farther from its centre than its corners are.
fn clear_of(bb: &Aabb, s: &Surface) -> bool {
    let reach = match s {
        Surface::Plane(p) => {
            let h = (bb.max - bb.min) * 0.5;
            let n = p.frame.z;
            n.x.abs() * h.x + n.y.abs() * h.y + n.z.abs() * h.z
        }
        _ => 0.5 * bb.diagonal(),
    };
    gap(s, (bb.min + bb.max) * 0.5) > reach + tol::LINEAR
}

/// The stretches of `range` along which `curve`, whose box over all of it is `whole`, may come
/// within the tolerance of `s`, inside the box `bound`. A stretch is dropped where its box misses
/// `bound` or is [`clear_of`] the surface, or where it lies [`apart_along`] the line from the
/// surface to an enclosing stretch's middle. Kept stretches are halved, so a curve that only passes
/// near the surface costs a few distances, not the sampling a crossing is found by.
fn near_ranges(
    curve: &Curve,
    range: (f64, f64),
    whole: Aabb,
    s: &Surface,
    bound: &Aabb,
    boxes: &mut CurveBoxes,
) -> Vec<(f64, f64)> {
    near_ranges_by(curve, range, whole, s, bound, true, boxes)
}

/// [`near_ranges`], with the test [`apart_along`] left out unless `along`.
fn near_ranges_by(
    curve: &Curve,
    range: (f64, f64),
    whole: Aabb,
    s: &Surface,
    bound: &Aabb,
    along: bool,
    boxes: &mut CurveBoxes,
) -> Vec<(f64, f64)> {
    #[allow(clippy::too_many_arguments)]
    fn halve(
        c: &Curve,
        (t0, t1): (f64, f64),
        bb: Aabb,
        depth: u32,
        s: &Surface,
        bound: &Aabb,
        along: bool,
        away: Option<Vec3>,
        boxes: &mut CurveBoxes,
        out: &mut Vec<(f64, f64)>,
    ) {
        if !boxes_meet(&bb, bound) || (analytic(s) && clear_of(&bb, s)) {
            return;
        }
        if let (Surface::Nurbs(n), Some(d)) = (s, away)
            && apart_along(c, (t0, t1), &bb, n, d)
        {
            return;
        }
        let m = 0.5 * (t0 + t1);
        let p = c.eval(m);
        let (g, away) = match s {
            Surface::Nurbs(_) => {
                let uv = s.project(p);
                let foot = s.eval(uv.x, uv.y);
                (foot.dist(p), (p - foot).unit().filter(|_| along))
            }
            _ => (gap(s, p), None),
        };
        // Off the analytic surfaces a distance costs a projection: the one from the curve's middle
        // also bounds the stretch, which lies within the box's farthest corner of it.
        if !analytic(s) {
            let mut reach: f64 = 0.0;
            for k in 0..8 {
                let corner = Vec3::new(
                    if k & 1 == 0 { bb.min.x } else { bb.max.x },
                    if k & 2 == 0 { bb.min.y } else { bb.max.y },
                    if k & 4 == 0 { bb.min.z } else { bb.max.z },
                );
                reach = reach.max(corner.dist(p));
            }
            if g > reach + tol::LINEAR {
                return;
            }
        }
        // Where the curve's own middle is on the surface, halving drops nothing that matters: the
        // stretch is sampled whole.
        if depth == 0 || g <= tol::LINEAR {
            match out.last_mut() {
                Some(last) if last.1 == t0 => last.1 = t1,
                _ => out.push((t0, t1)),
            }
            return;
        }
        for r in [(t0, m), (m, t1)] {
            let bb = padded_box(boxes.bbox(c, r.0, r.1));
            halve(c, r, bb, depth - 1, s, bound, along, away, boxes, out);
        }
    }
    let mut out = Vec::new();
    halve(
        curve, range, whole, 8, s, bound, along, None, boxes, &mut out,
    );
    out
}

/// Whether the stretch `t0`–`t1` of `c`, whose padded box is `bb`, lies farther than the tolerance
/// from `n` along the unit `d`: each Bézier patch of the surface whose box meets `bb` has every
/// control point, and so its convex hull and the patch, short of the stretch's least reach along
/// `d` by more than the tolerance. A curve running beside a narrow face, as a gear's flanks do, is
/// told apart by one such test where the ball about its middle needs its length halved to the gap.
fn apart_along(c: &Curve, (t0, t1): (f64, f64), bb: &Aabb, n: &NurbsSurface, d: Vec3) -> bool {
    let mut high = f64::NEG_INFINITY;
    for z in n.beziers() {
        if boxes_meet(bb, &z.bbox()) {
            for q in z.points() {
                high = high.max(d.dot(Vec3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3])));
            }
        }
    }
    let ends = d.dot(c.eval(t0)).min(d.dot(c.eval(t1)));
    if high + tol::LINEAR >= ends {
        return false;
    }
    let h = (bb.max - bb.min) * 0.5;
    let boxed =
        d.dot((bb.min + bb.max) * 0.5) - (d.x.abs() * h.x + d.y.abs() * h.y + d.z.abs() * h.z);
    boxed - tol::LINEAR > high
        || c.extremes(d, t0, t1)
            .into_iter()
            .all(|t| d.dot(c.eval(t)) - tol::LINEAR > high)
}

/// Where edge `e` meets face `y`, whose box is `bound`, other than at the face's own vertices
/// `own`: inside it or on its boundary. Near a shared vertex, a meeting the edge reaches without
/// leaving the surface by more than the tolerance is that vertex, however far along it rounding put
/// the root. A stretch of edge that lies in the face's surface meets the face where it crosses the
/// face's boundary, or else lies wholly inside or outside it, as its middle does.
fn edge_meets_face(
    b: &Brep,
    e: EdgeId,
    y: FaceId,
    bound: &Aabb,
    own: &[VertexId],
    cache: &mut PairCache,
) -> Option<Point3> {
    let edge = b.edge(e);
    let curve = edge.curve.as_ref()?;
    let whole = cache.edge_box(b, e)?;
    let s = &b.face(y).surface;
    let ends: Vec<(f64, Point3, f64)> = [(edge.range.0, edge.start), (edge.range.1, edge.end)]
        .into_iter()
        .filter(|(_, v)| own.contains(v))
        .map(|(t, v)| (t, b.vertex(v).point, b.vertex_tolerance(v)))
        .collect();
    let along = b.edge_tolerance(e);
    let at_shared = |t: f64, p: Point3| {
        ends.iter().any(|&(tv, q, within)| {
            q.dist(p) <= within
                || (0..=8).all(|k| {
                    let x = curve.eval(t + (tv - t) * k as f64 / 8.0);
                    signed_distance(s, x).abs() <= along
                })
        })
    };
    let near_end = |p: Point3| ends.iter().any(|&(_, q, within)| q.dist(p) <= within);
    let mut found: Vec<Point3> = Vec::new();
    let boxes = cache.curve_boxes.entry(e).or_default();
    for range in near_ranges(curve, edge.range, whole, s, bound, boxes) {
        let hits = curve_surface_in(curve, range, s, boxes);
        if !hits.on_surface {
            found.extend(
                hits.ts
                    .iter()
                    .map(|&t| (t, curve.eval(t)))
                    .filter(|&(t, p)| !at_shared(t, p))
                    .map(|(_, p)| p),
            );
            continue;
        }
        found.extend((1..4).map(|k| curve.eval(range.0 + (range.1 - range.0) * k as f64 / 4.0)));
    }
    // A root is checked against the surface itself: beside a spline face's boundary a point
    // projects onto the boundary, and its signed distance along the normal there reads nil
    // however far off it lies in the tangent plane.
    let reach = 10.0 * along.max(tol::LINEAR);
    found.retain(|&p| in_box(p, bound) && !near_end(p) && gap(s, p) <= reach);
    if found.is_empty() {
        return None;
    }
    let dom = cache.domains.entry(y).or_insert_with(|| face_domain(b, y));
    found
        .into_iter()
        .find(|&p| dom.on_boundary(p) || dom.exact_side(p).is_some())
}

/// Where two faces that share a vertex but no edge cross each other along a curve out of it: their
/// surfaces cut across each other there, and a step along the line where their tangent planes
/// meet, either way, lands inside both faces. No edge of either need meet the other's inside, as
/// where two balls turned apart overlap between the two points where the turn's axis pierces both.
fn crossing_from_shared_vertex(
    b: &Brep,
    (fa, fb): (FaceId, FaceId),
    (va, vb): (&[VertexId], &[VertexId]),
    cache: &mut PairCache,
) -> Option<Point3> {
    let (sa, sb) = (&b.face(fa).surface, &b.face(fb).surface);
    let size = cache
        .face_box(b, fa)
        .diagonal()
        .min(cache.face_box(b, fb).diagonal());
    let mut seen: Vec<VertexId> = Vec::new();
    for &v in va.iter().filter(|v| vb.contains(v)) {
        if seen.contains(&v) {
            continue;
        }
        seen.push(v);
        let p = b.vertex(v).point;
        let normal = |s: &Surface| {
            let uv = s.project(p);
            s.normal(uv.x, uv.y)
        };
        let Some(line) = normal(sa).cross(normal(sb)).unit() else {
            continue;
        };
        if normal(sa).cross(normal(sb)).norm() <= tol::TANGENT {
            continue;
        }
        let step = 1e-3 * size;
        for sign in [1.0, -1.0] {
            // A point of both surfaces a step out along the line, by alternating projections.
            let mut q = p + line * (sign * step);
            for _ in 0..8 {
                let on_a = sa.project(q);
                q = sa.eval(on_a.x, on_a.y);
                let on_b = sb.project(q);
                q = sb.eval(on_b.x, on_b.y);
            }
            if signed_distance(sa, q).abs() > tol::LINEAR || q.dist(p) < 0.5 * step {
                continue;
            }
            let inside = |f: FaceId, cache: &mut PairCache| {
                let dom = cache.domains.entry(f).or_insert_with(|| face_domain(b, f));
                matches!(dom.locate(q), Where::Inside(_))
            };
            if inside(fa, cache) && inside(fb, cache) {
                return Some(q);
            }
        }
    }
    None
}

/// Pairs of distinct vertices closer than the linear tolerance.
fn coincident_vertices(b: &Brep) -> Vec<(VertexId, VertexId)> {
    let cell = |x: f64| (x / (4.0 * tol::LINEAR)).floor() as i64;
    let mut grid: HashMap<[i64; 3], Vec<VertexId>> = HashMap::new();
    let mut out = Vec::new();
    for v in b.vertex_ids() {
        let p = b.vertex(v).point;
        if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
            continue;
        }
        let c = [cell(p.x), cell(p.y), cell(p.z)];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &w in grid
                        .get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                        .into_iter()
                        .flatten()
                    {
                        if b.vertex(w).point.dist(p) <= tol::LINEAR {
                            out.push((w, v));
                        }
                    }
                }
            }
        }
        grid.entry(c).or_default().push(v);
    }
    // A tolerant vertex coincides with any within the sum of the two tolerances, which the grid's
    // cells are too small to find: a sweep along x finds those pairs.
    let (tolerant, _) = b.tolerant();
    if tolerant.is_empty() {
        return out;
    }
    let widest = tolerant.iter().map(|&(_, t)| t).fold(tol::LINEAR, f64::max);
    let mut by_x: Vec<(f64, VertexId)> = b
        .vertex_ids()
        .map(|v| (b.vertex(v).point.x, v))
        .filter(|(x, _)| x.is_finite())
        .collect();
    by_x.sort_by(|a, b| a.0.total_cmp(&b.0));
    for &(v, tv) in &tolerant {
        let p = b.vertex(v).point;
        let reach = tol::between(tv, widest);
        let from = by_x.partition_point(|&(x, _)| x < p.x - reach);
        for &(x, w) in &by_x[from..] {
            if x > p.x + reach {
                break;
            }
            if w == v || (b.vertex_tolerance(w) > tol::LINEAR && w < v) {
                continue;
            }
            let d = b.vertex(w).point.dist(p);
            if d > tol::LINEAR && d <= tol::between(tv, b.vertex_tolerance(w)) {
                out.push((v.min(w), v.max(w)));
            }
        }
    }
    out
}

/// How many fans of faces meet at each vertex: following each corner there across the edge it
/// leaves by, to its partner, reaches the corners of one fan. More than one is a point where
/// pieces of the solid touch.
pub(crate) fn vertex_fans(b: &Brep) -> HashMap<VertexId, usize> {
    fans_at_vertices(b)
        .into_iter()
        .map(|(v, fs)| (v, fs.len()))
        .collect()
}

/// Each vertex's fans of faces, each as the coedges arriving at the vertex round it. A degenerate
/// coedge, a pole or a vertex loop, leaves the vertex where it arrived, so its face's corners on
/// either side of it are one corner.
pub(crate) fn fans_at_vertices(b: &Brep) -> HashMap<VertexId, Vec<Vec<CoedgeId>>> {
    // A corner at a vertex: the coedge arriving there and the next one in its loop, leaving.
    let mut corners: HashMap<VertexId, Vec<(CoedgeId, CoedgeId)>> = HashMap::new();
    for lp in b.loop_ids() {
        let cs = &b.lp(lp).coedges;
        for k in 0..cs.len() {
            let (a, n) = (cs[k], cs[(k + 1) % cs.len()]);
            let v = b.coedge_end(a);
            if b.coedge_start(n) == v {
                corners.entry(v).or_default().push((a, n));
            }
        }
    }
    let mut out = HashMap::new();
    for (&v, cs) in &corners {
        let by_arrival: HashMap<CoedgeId, usize> =
            cs.iter().enumerate().map(|(i, &(a, _))| (a, i)).collect();
        let mut seen = vec![false; cs.len()];
        let mut fans = Vec::new();
        for start in 0..cs.len() {
            if seen[start] {
                continue;
            }
            let mut fan = Vec::new();
            let mut i = start;
            while !seen[i] {
                seen[i] = true;
                fan.push(cs[i].0);
                let leaving = cs[i].1;
                let mate = if b.edge(b.coedge(leaving).edge).is_degenerate() {
                    Some(leaving)
                } else {
                    b.partner(leaving).filter(|&m| b.coedge_end(m) == v)
                };
                match mate.and_then(|m| by_arrival.get(&m)) {
                    Some(&j) => i = j,
                    None => break,
                }
            }
            fans.push(fan);
        }
        out.insert(v, fans);
    }
    out
}

/// Where the solid touches itself: the edges more than two faces use, and the vertices those end
/// at or that more than one fan of faces meets at.
pub(crate) fn contacts(b: &Brep) -> (HashSet<EdgeId>, HashSet<VertexId>) {
    let edges: HashSet<EdgeId> = b.radial_edges().collect();
    let mut vertices: HashSet<VertexId> = vertex_fans(b)
        .into_iter()
        .filter(|&(_, n)| n > 1)
        .map(|(v, _)| v)
        .collect();
    for &e in &edges {
        vertices.extend([b.edge(e).start, b.edge(e).end]);
    }
    (edges, vertices)
}

/// The faces of `faces` that run up to where the solid touches itself.
pub(crate) fn faces_at_contacts(b: &Brep, faces: impl IntoIterator<Item = FaceId>) -> Vec<FaceId> {
    let (edges, vertices) = contacts(b);
    if edges.is_empty() && vertices.is_empty() {
        return Vec::new();
    }
    faces
        .into_iter()
        .filter(|&f| {
            b.face(f).loops.iter().any(|&lp| {
                b.lp(lp).coedges.iter().any(|&c| {
                    edges.contains(&b.coedge(c).edge) || vertices.contains(&b.coedge_start(c))
                })
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{BSpline, IntersectionCurve, Plane};
    use crate::math::{Frame, Vec3};

    #[test]
    fn a_step_on_a_surface_whose_parameters_run_askew_is_measured_as_it_lies_in_space() {
        // A plane whose parameters run nearly opposite ways, as a blend's spline can near a
        // corner: a step along both moves the point little, though each alone moves it far.
        let (su, sv) = (Vec3::new(10.0, 0.0, 0.0), Vec3::new(-9.9, 0.5, 0.0));
        let net = vec![Vec3::ZERO, sv, su, su + sv];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let s = Surface::Nurbs(Box::new(
            crate::geom::NurbsSurface::new((1, 1), (knots.clone(), knots), (2, 2), net, None)
                .unwrap(),
        ));
        let (p, q) = (Vec2::new(0.2, 0.3), Vec2::new(0.21, 0.31));
        let apart = s.eval(p.x, p.y).dist(s.eval(q.x, q.y));
        assert!((carried(&s, p, q) - apart).abs() < 1e-12, "{apart}");
        // Each speed alone, as the strict test of a vertex weighs a gap, overstates it twentyfold.
        let each = (q.x - p.x).abs() * su.norm() + (q.y - p.y).abs() * sv.norm();
        assert!(each > 20.0 * apart);
    }

    #[test]
    fn a_coedge_s_samples_end_exactly_at_its_edge_s_ends() {
        // Run backwards, `a + (z - a) * 1` falls an ulp short of `z` for this range, and a last
        // sample inside the edge is taken for an interior point: at a cone's apex its angle is
        // then any angle, and the loop seems not to close.
        let (t0, t1) = (-2.9474992780948828, 2.9474992780505347);
        for (a, z) in [(t1, t0), (t0, t1)] {
            for n in [1, 8, 64] {
                let ts = spaced(a, z, n);
                assert_eq!((ts[0], ts[n]), (a, z), "{n} steps from {a} to {z}");
            }
        }
    }

    fn fitted(points: &[Vec3]) -> Curve {
        let plane = |z: Vec3| {
            Surface::Plane(Plane {
                frame: Frame::from_axis(Vec3::ZERO, z).unwrap(),
            })
        };
        Curve::Intersection(Box::new(IntersectionCurve {
            approx: BSpline::interpolate(points).unwrap(),
            surfaces: [plane(Vec3::Z), plane(Vec3::Y)],
        }))
    }

    #[test]
    fn a_fitted_curve_that_doubles_back_folds_and_one_that_bends_does_not() {
        // Points along x that step back once, as scatter across a near-tangent band does, and
        // points round a quarter circle.
        let zigzag: Vec<Vec3> = [0.0, 1.0, 2.0, 1.9, 3.0, 4.0]
            .iter()
            .enumerate()
            .map(|(i, &x)| Vec3::new(x, 0.01 * i as f64, 0.0))
            .collect();
        let bend: Vec<Vec3> = (0..=8)
            .map(|k| {
                let a = std::f64::consts::FRAC_PI_2 * k as f64 / 8.0;
                Vec3::new(a.cos(), a.sin(), 0.0)
            })
            .collect();
        for (pts, folds) in [(zigzag, true), (bend, false)] {
            let c = fitted(&pts);
            let knots = c.spline().unwrap().knots();
            let range = (knots[0], knots[knots.len() - 1]);
            assert_eq!(fold(&c, range).is_some(), folds);
        }
    }

    // Faces two steps apart along a tooth's bent side, screwed as a helical gear's flank is, run
    // 0.04 apart all the way up: where the line to the surface tells a stretch apart, halving down
    // to the gap keeps nothing more of it, and both find the same stretches.
    #[test]
    fn a_stretch_told_apart_along_the_line_to_the_surface_is_one_halving_drops_too() {
        let mut pts = vec![Vec2::new(16.0, -1.0)];
        pts.extend((0..=6).map(|k| {
            let a = 0.04 / 3.0 * k as f64;
            Vec2::new(19.0 + 3.0 * a.sin(), 2.0 - 3.0 * a.cos())
        }));
        pts.extend([Vec2::new(19.5, 1.5), Vec2::new(16.0, 1.5)]);
        let twist = 10.0 * 15f64.to_radians().tan() / 18.0;
        let screw = crate::sweep::Helix {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            rise: 10.0 / twist,
            sign: 1.0,
        };
        let p = crate::profile::Profile::polygon(&pts, &["side"]);
        let b = crate::sweep::helix(&p, &Frame::WORLD, &screw, twist, "tooth").unwrap();
        let mut cache = PairCache::default();
        let mut pairs = 0;
        for e in b.edge_ids() {
            let edge = b.edge(e);
            let Some(curve) = edge.curve.as_ref() else {
                continue;
            };
            let whole = cache.edge_box(&b, e).unwrap();
            for y in b.face_ids() {
                let s = &b.face(y).surface;
                let bound = cache.face_box(&b, y);
                if !matches!(s, Surface::Nurbs(_)) || !boxes_meet(&whole, &bound) {
                    continue;
                }
                let at = |along| {
                    let mut boxes = CurveBoxes::default();
                    near_ranges_by(curve, edge.range, whole, s, &bound, along, &mut boxes)
                };
                assert_eq!(at(true), at(false));
                pairs += 1;
            }
        }
        assert!(pairs > 50, "{pairs}");
    }

    #[test]
    fn faces_round_the_line_where_two_holes_touch_leave_it_into_themselves_and_lie_apart() {
        // A plate less two holes of radius 1 whose axes are 2 apart: they touch along a line,
        // which four coedges use. A face that uses the line once leaves its middle toward any
        // point inside the face, a chord at most once round a cylinder from the edge.
        let (z, x) = (Vec3::Z, Vec3::X);
        let plate = crate::primitive::block(
            &Frame::new(Vec3::ZERO, z, x).unwrap(),
            Vec3::new(5.0, 3.0, 1.0),
            "p",
        )
        .unwrap();
        let hole = |cx: f64| {
            let f = Frame::new(Vec3::new(cx, 1.5, -1.0), z, x).unwrap();
            crate::primitive::cylinder(&f, 1.0, 3.0, "h").unwrap()
        };
        let one = crate::boolean::difference(&plate, &hole(1.5)).unwrap();
        let b = crate::boolean::difference(&one, &hole(3.5)).unwrap();
        assert!(check(&b).is_empty());
        let shared: Vec<EdgeId> = b
            .edge_ids()
            .filter(|&e| b.edge_uses(e).len() == 4)
            .collect();
        assert!(!shared.is_empty());
        for e in shared {
            let uses = b.edge_uses(e);
            let once = |c: &&CoedgeId| {
                let f = b.coedge_face(**c);
                uses.iter().filter(|&&o| b.coedge_face(o) == f).count() == 1
            };
            for &c in uses.iter().filter(once) {
                let (t0, t1) = b.edge(e).range;
                let mid = b.edge(e).curve.as_ref().unwrap().eval(0.5 * (t0 + t1));
                let d = face_domain(&b, b.coedge_face(c));
                let uv = d.interior_point().unwrap();
                let inside = d.surface.eval(uv.x, uv.y);
                assert!(into_face(&b, c).unwrap().dot(inside - mid) > 0.0);
            }
            assert_eq!(lying_on_each_other(&b, &Domains::new(&b), uses), None);
        }
    }

    fn at(x: f64, y: f64, z: f64) -> Frame {
        Frame::new(Vec3::new(x, y, z), Vec3::Z, Vec3::X).unwrap()
    }

    fn block(o: [f64; 3], size: [f64; 3]) -> Brep {
        crate::primitive::block(&at(o[0], o[1], o[2]), Vec3::from_array(size), "b").unwrap()
    }

    /// A copy of `b` with the faces of each shell `flip` picks turned inside out, as one solid of
    /// one shell when `merged`.
    fn rebuilt(b: &Brep, flip: impl Fn(ShellId) -> bool, merged: bool) -> Brep {
        use crate::topo::{CoedgeSpec, PCurve};
        let mut out = Brep::new();
        let vs: Vec<VertexId> = b
            .vertex_ids()
            .map(|v| out.add_vertex(b.vertex(v).point))
            .collect();
        let es: Vec<EdgeId> = b
            .edge_ids()
            .map(|e| {
                let x = b.edge(e);
                let (v0, v1) = (vs[x.start.index()], vs[x.end.index()]);
                match &x.curve {
                    Some(c) => out.add_edge(c.clone(), x.range, v0, v1),
                    None => out.add_degenerate_edge(v0, x.range),
                }
            })
            .collect();
        let mut all = Vec::new();
        for s in b.solid_ids() {
            let mut shells = Vec::new();
            for &sh in &b.solid(s).shells {
                let flip = flip(sh);
                let mut faces = Vec::new();
                for &f in &b.shell(sh).faces {
                    let face = b.face(f);
                    let names = face.names.iter().cloned();
                    let g = out.add_face(face.surface.clone(), face.reversed != flip, names);
                    for &lp in &face.loops {
                        let mut specs: Vec<CoedgeSpec> = b
                            .lp(lp)
                            .coedges
                            .iter()
                            .map(|&c| {
                                let co = b.coedge(c);
                                // A projected pcurve is seeded where its coedge starts.
                                let pcurve = match &co.pcurve {
                                    PCurve::Projected { .. } if flip => PCurve::Projected {
                                        seed: b.coedge_uvs(c, &coedge_params(b, c, 1))[1],
                                    },
                                    other => other.clone(),
                                };
                                CoedgeSpec {
                                    edge: es[co.edge.index()],
                                    reversed: co.reversed != flip,
                                    pcurve,
                                }
                            })
                            .collect();
                        if flip {
                            specs.reverse();
                        }
                        out.add_loop(g, specs);
                    }
                    faces.push(g);
                }
                if merged {
                    all.extend(faces);
                } else {
                    shells.push(out.add_shell(faces));
                }
            }
            if !merged {
                out.add_solid_of(shells, b.solid(s).outer);
            }
        }
        if merged {
            let sh = out.add_shell(all);
            out.add_solid(vec![sh]);
        }
        out.pair_radial();
        out
    }

    fn seen_outward(b: &Brep) -> Vec<bool> {
        let (domains, mut cache) = (Domains::new(b), PairCache::default());
        b.solid_ids()
            .map(|s| faces_outward(b, &domains, &mut cache, s))
            .collect()
    }

    /// The issues of `b` as a boolean's result with no two faces known to come from one input.
    fn cut_issues(b: &Brep) -> Vec<String> {
        check_cut(b, &vec![None; b.face_count()])
            .into_iter()
            .map(|i| i.message)
            .collect()
    }

    #[test]
    fn a_ray_shows_a_boolean_s_one_sheet_facing_outward_and_one_turned_inside_out_is_refused() {
        use crate::boolean::{difference, intersection, union};
        use crate::primitive::{cone, cylinder, sphere};
        let cube = || block([0.0; 3], [10.0; 3]);
        let bore = cylinder(&at(5.0, 5.0, -1.0), 3.0, 12.0, "h").unwrap();
        let ball = |x: f64| sphere(&at(x, 0.0, 0.0), 5.0, "s").unwrap();
        let spike = cone(&at(5.0, 5.0, 5.0), 4.0, 1.0, 9.0, "k").unwrap();
        let results = [
            union(&cube(), &block([5.0; 3], [10.0; 3])).unwrap(),
            intersection(&cube(), &bore).unwrap(),
            difference(&cube(), &bore).unwrap(),
            difference(&ball(0.0), &ball(4.0)).unwrap(),
            union(&cube(), &spike).unwrap(),
        ];
        for r in &results {
            assert_eq!(seen_outward(r), [true]);
            assert_eq!(cut_issues(r), Vec::<String>::new());
            // Inside out, every edge is still used twice in opposite ways and every loop runs its
            // face's way: only which side the sheet faces is wrong.
            let t = rebuilt(r, |_| true, false);
            assert_eq!(seen_outward(&t), [false]);
            assert!(measure::solid_volume_sign(&t, t.solid_ids().next().unwrap()) < 0.0);
            let issues = cut_issues(&t);
            assert_eq!(issues.len(), 1, "{issues:?}");
            assert!(issues[0].starts_with("solid 0 has volume -"), "{issues:?}");
            assert_eq!(
                issues,
                check(&t).into_iter().map(|i| i.message).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn a_ray_from_a_face_turned_inside_out_shows_it_facing_in() {
        let r = crate::boolean::union(&block([0.0; 3], [10.0; 3]), &block([5.0; 3], [10.0; 3]));
        let t = rebuilt(&r.unwrap(), |_| true, false);
        let faces: Vec<FaceId> = t.face_ids().collect();
        let (domains, mut cache) = (Domains::new(&t), PairCache::default());
        let boxes: Vec<Aabb> = faces.iter().map(|&f| cache.face_box(&t, f)).collect();
        // A ray that leaves through an edge decides nothing; none shows the sheet facing out.
        let seen: Vec<Option<bool>> = (0..faces.len())
            .map(|start| side_seen_from(&t, &domains, &faces, &boxes, start))
            .collect();
        assert!(seen.contains(&Some(false)), "{seen:?}");
        assert!(!seen.contains(&Some(true)), "{seen:?}");
    }

    #[test]
    fn a_solid_with_a_void_is_settled_by_its_volume_and_a_turned_shell_of_it_is_refused() {
        let hollow =
            crate::boolean::difference(&block([0.0; 3], [10.0; 3]), &block([3.0; 3], [4.0; 3]))
                .unwrap();
        let s = hollow.solid_ids().next().unwrap();
        let shells = hollow.solid(s).shells.clone();
        assert_eq!(shells.len(), 2);
        assert_eq!(seen_outward(&hollow), [false]);
        assert_eq!(cut_issues(&hollow), Vec::<String>::new());
        // The void turned to face outward, then the outer shell turned to face in.
        let void_out = rebuilt(&hollow, |sh| sh == shells[1], false);
        let issues = cut_issues(&void_out);
        assert!(
            issues
                .iter()
                .any(|m| m.contains("is a void turned inside out")),
            "{issues:?}"
        );
        let outer_in = rebuilt(&hollow, |sh| sh == shells[0], false);
        let issues = cut_issues(&outer_in);
        assert!(
            issues.iter().any(|m| m.starts_with("solid 0 has volume -")),
            "{issues:?}"
        );
    }

    #[test]
    fn a_shell_of_two_sheets_is_settled_by_its_volume() {
        // Two blocks apart in one shell: a ray from a face of one says nothing of the other.
        let mut two = block([0.0; 3], [10.0; 3]);
        two.append(&block([20.0, 0.0, 0.0], [2.0; 3]));
        let small = two.shell_ids().nth(1).unwrap();
        for flip in [false, true] {
            let b = rebuilt(&two, |sh| flip && sh == small, true);
            let faces: Vec<FaceId> = b.face_ids().collect();
            assert!(!one_sheet(&b, &faces));
            assert_eq!(seen_outward(&b), [false]);
            assert_eq!(
                cut_issues(&b),
                check(&b).into_iter().map(|i| i.message).collect::<Vec<_>>()
            );
        }
        // The large block alone turned in leaves the shell's volume negative.
        let large = two.shell_ids().next().unwrap();
        let b = rebuilt(&two, |sh| sh == large, true);
        let issues = cut_issues(&b);
        assert!(
            issues.iter().any(|m| m.starts_with("solid 0 has volume -")),
            "{issues:?}"
        );
    }

    #[test]
    fn a_ray_past_a_face_no_closed_form_crosses_decides_nothing() {
        // A ring half sunk in a plate's top: the disc of the top inside the ring looks up through
        // the ring's box, the plate's other faces look away from it.
        let plate = block([0.0; 3], [10.0, 10.0, 2.0]);
        let ring = crate::primitive::torus(&at(5.0, 5.0, 2.0), 3.0, 1.0, "r").unwrap();
        let b = crate::boolean::union(&plate, &ring).unwrap();
        let faces: Vec<FaceId> = b.face_ids().collect();
        let (domains, mut cache) = (Domains::new(&b), PairCache::default());
        let boxes: Vec<Aabb> = faces.iter().map(|&f| cache.face_box(&b, f)).collect();
        let seen: Vec<Option<bool>> = (0..faces.len())
            .filter(|&i| ray_surface_is_closed_form(&b.face(faces[i]).surface))
            .map(|i| side_seen_from(&b, &domains, &faces, &boxes, i))
            .collect();
        assert!(seen.contains(&None), "{seen:?}");
        assert!(seen.contains(&Some(true)), "{seen:?}");
        assert!(!seen.contains(&Some(false)), "{seen:?}");
        assert_eq!(seen_outward(&b), [true]);
        let t = rebuilt(&b, |_| true, false);
        assert_eq!(seen_outward(&t), [false]);
        let issues = cut_issues(&t);
        assert!(
            issues.iter().any(|m| m.starts_with("solid 0 has volume -")),
            "{issues:?}"
        );
    }
}
