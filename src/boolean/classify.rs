//! Whether a point lies inside a solid, by the parity of a ray's crossings of its exact faces. A
//! ray that passes through an edge, grazes a surface or starts on a face is ambiguous and the next
//! direction is tried. A piece of face whose point lies on the solid's boundary, touching it, is
//! decided by the contact instead, since no ray from there has a parity.

use super::{
    Op, contact,
    csi::{ray_surface, signed_distance},
    domain::{Domain, OnBoundary, Where},
    face_bbox, face_domain,
};
use crate::{
    geom::Surface,
    math::{Aabb, Point3, Vec3},
    tol,
    topo::Brep,
};

/// A gap between a point of one surface and another surface that is no rounding of a touch: a
/// thousand times what evaluating both leaves at [`tol::MODEL_SIZE`], and a thousandth of the
/// tolerance.
const CLEAR: f64 = tol::LINEAR * 1e-3;

/// Directions with no simple relation to the axes or each other; each is normalised before use.
pub(crate) const DIRECTIONS: [[f64; 3]; 9] = [
    [0.5773, 0.6129, 0.5395],
    [-0.4219, 0.7064, 0.5675],
    [0.8123, -0.3311, 0.4801],
    [-0.2672, -0.5345, 0.8018],
    [0.6247, 0.2179, -0.7499],
    [-0.7064, -0.3963, -0.5857],
    [0.1302, -0.9129, 0.3869],
    [0.9411, 0.1903, -0.2793],
    [-0.3517, 0.1491, -0.9241],
];

/// The stretch of the line `p + t·d` inside `bb`, a little widened, where the ray from `p`
/// reaches it.
pub(crate) fn ray_box(p: Point3, d: Vec3, bb: &Aabb) -> Option<(f64, f64)> {
    let pad = 1e-6 * (1.0 + bb.diagonal());
    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        let (mn, mx) = (bb.min[k] - pad, bb.max[k] + pad);
        if d[k].abs() < 1e-300 {
            if p[k] < mn || p[k] > mx {
                return None;
            }
            continue;
        }
        let (a, b) = ((mn - p[k]) / d[k], (mx - p[k]) / d[k]);
        lo = lo.max(a.min(b));
        hi = hi.min(a.max(b));
    }
    (hi >= lo && hi >= 0.0).then_some((lo, hi))
}

/// The crossings of one ray, or `None` when the ray is ambiguous.
fn crossings(faces: &[(&Domain, Aabb)], p: Point3, d: Vec3) -> Option<usize> {
    let mut count = 0;
    for (dom, bb) in faces {
        let Some((lo, hi)) = ray_box(p, d, bb) else {
            continue;
        };
        for t in ray_surface(&dom.surface, p, d, hi.max(0.0)) {
            if t < lo - tol::LINEAR || t > hi + tol::LINEAR {
                continue;
            }
            let q = p + d * t;
            let near = t.abs() <= tol::LINEAR;
            let uv = match dom.locate(q) {
                Where::Outside => continue,
                Where::Inside(uv) => uv,
                // A piece within the tolerance of another solid's face is a sliver cut along that
                // face's edges, so the exact side of the hit stands for the whole piece.
                Where::Boundary if near => match dom.exact_side(q) {
                    Some(uv) => uv,
                    None => continue,
                },
                Where::Boundary => return None,
            };
            if t == 0.0 {
                return None;
            }
            if t < 0.0 {
                continue;
            }
            if dom.outward(uv).dot(d).abs() < tol::TANGENT {
                return None;
            }
            count += 1;
        }
    }
    Some(count)
}

/// Whether a piece of `surface` that touches the solid bounded by `faces` at `p`, without crossing
/// it, lies inside. A point on the solid's boundary has no parity, but every other point of the
/// piece lies on one side of it, which the contact decides: on a face, the side of it the piece's
/// surface bends to (the gap between the two surfaces); on an edge, whether the piece's tangent
/// plane, which holds the edge, runs into the wedge of material the edge's faces bound. `None` where
/// `p` touches nothing, or the contact does not decide (a corner, surfaces that osculate or cross).
pub(crate) fn touching(surface: &Surface, faces: &[(&Domain, Aabb)], p: Point3) -> Option<bool> {
    let mut on_face = Vec::new();
    let mut on_edge = Vec::new();
    for (dom, bb) in faces {
        if !(0..3).all(|k| p[k] >= bb.min[k] - tol::LINEAR && p[k] <= bb.max[k] + tol::LINEAR) {
            continue;
        }
        match dom.locate(p) {
            Where::Outside => {}
            Where::Boundary => match dom.boundary_at(p) {
                Some(OnBoundary::Seam) => on_face.push(*dom),
                Some(OnBoundary::Edge(along)) => on_edge.push((*dom, along)),
                _ => return None,
            },
            Where::Inside(_) => on_face.push(*dom),
        }
    }
    match (on_face.as_slice(), on_edge.as_slice()) {
        ([], []) => None,
        ([dom], []) => {
            // A piece cut where it crosses the face at a small angle lies within the tolerance
            // of it for some way, a sliver's wall between two crossings all over, and still
            // stands clear of it by more than rounding: on the side `p` stands on. Only a piece
            // that touches the face without a gap is left to how the two bend.
            let uv = dom.surface.project(p);
            let gap = (p - dom.surface.eval(uv.x, uv.y)).dot(dom.outward(uv));
            if gap.abs() > CLEAR {
                return Some(gap < 0.0);
            }
            let above = contact::side_of(surface, &dom.surface, p)?;
            Some(above == dom.reversed)
        }
        ([], &[a, b]) => in_wedge(surface, a, b, p),
        _ => None,
    }
}

/// Whether the plane tangent to `surface` at `p`, which holds the edge between faces `a` and `b`
/// of a solid, runs into the solid on both sides of the edge; `None` where it runs in on one side
/// only (it crosses the solid). Along a face it is tangent to, the gap decides that face's side.
fn in_wedge(
    surface: &Surface,
    (a, along): (&Domain, Vec3),
    (b, _): (&Domain, Vec3),
    p: Point3,
) -> Option<bool> {
    let outward = |dom: &Domain| dom.outward(dom.surface.project(p));
    let (na, nb) = (outward(a), outward(b));
    // The loops keep the material on their left, seen from outside.
    let ta = na.cross(along).unit_or_zero();
    let w = surface.project(p);
    let normal = surface.normal(w.x, w.y);
    let clear = |x: f64| x.abs() > tol::TANGENT;
    if clear(normal.dot(along)) {
        return None;
    }
    let across = normal.cross(along).unit()?;
    if !clear(ta.dot(nb)) {
        return None;
    }
    let convex = ta.dot(nb) < 0.0;
    // Whether a step along `d` enters the material behind face `dom`; along the face, the side
    // its surface bends the piece's to.
    let behind = |dom: &Domain, n: Vec3, d: Vec3| -> Option<bool> {
        let s = d.dot(n);
        if clear(s) {
            return Some(s < 0.0);
        }
        let above = contact::side_of(surface, &dom.surface, p)?;
        Some(above == dom.reversed)
    };
    let into = |d: Vec3| -> Option<bool> {
        let (ia, ib) = (behind(a, na, d)?, behind(b, nb, d)?);
        Some(if convex { ia && ib } else { ia || ib })
    };
    let (fwd, back) = (into(across)?, into(-across)?);
    if fwd == back {
        return Some(fwd);
    }
    // The plane runs into the solid on one side of the edge only. The edge was cut into the piece,
    // so the piece lies on one side, the side of `b`'s surface `p` is on: exact however near the
    // edge, where a piece thinner than twice the tolerance always is.
    let off = signed_distance(&b.surface, p) * if b.reversed { -1.0 } else { 1.0 };
    let tilt = across.dot(nb);
    if off.abs() <= 1e-3 * tol::LINEAR || !clear(tilt) {
        return None;
    }
    Some(if (tilt < 0.0) == (off < 0.0) {
        fwd
    } else {
        back
    })
}

/// Whether `p` is inside the solid bounded by `faces`, or `None` when every ray was ambiguous.
/// Every unambiguous ray gives the same parity, so the directions are tried cheapest first: by
/// how far each runs through the boxes of faces whose crossings are found by sampling.
pub(crate) fn contains(faces: &[(&Domain, Aabb)], p: Point3) -> Option<bool> {
    let sampled = |s: &Surface| {
        !matches!(
            s,
            Surface::Plane(_) | Surface::Cylinder(_) | Surface::Sphere(_) | Surface::Cone(_)
        )
    };
    let cost = |d: Vec3| -> f64 {
        faces
            .iter()
            .filter(|(dom, _)| sampled(&dom.surface))
            .filter_map(|(_, bb)| ray_box(p, d, bb))
            .map(|(lo, hi)| hi - lo.max(0.0))
            .sum()
    };
    let mut order: Vec<(f64, Vec3)> = DIRECTIONS
        .iter()
        .map(|d| {
            let d = Vec3::from_array(*d).unit_or_zero();
            (cost(d), d)
        })
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    order
        .into_iter()
        .find_map(|(_, d)| crossings(faces, p, d).map(|n| n % 2 == 1))
}

/// Whether solids `a` and `b` share material that an empty result of `op` leaves out: a vertex or
/// the middle of an edge of one lies farther than the tolerance from every surface of the other's
/// faces, and inside it for an intersection, or a point of `a` outside `b` for a difference. A
/// point of one solid's boundary that far inside the other has the first solid's material beside
/// it inside the other. Only such points are read, so overlaps they miss are not found.
pub(crate) fn material_lost(a: &Brep, b: &Brep, op: Op) -> bool {
    let points = |x: &Brep| -> Vec<Point3> {
        let mids = x.edge_ids().filter_map(|e| {
            let e = x.edge(e);
            let c = e.curve.as_ref()?;
            Some(c.eval(0.5 * (e.range.0 + e.range.1)))
        });
        x.vertex_ids()
            .map(|v| x.vertex(v).point)
            .chain(mids)
            .collect()
    };
    let deep = |from: &Brep, of: &Brep, inside: bool| {
        let doms: Vec<Domain> = of.face_ids().map(|f| face_domain(of, f)).collect();
        let faces: Vec<(&Domain, Aabb)> = doms.iter().map(|d| (d, face_bbox(d))).collect();
        points(from).into_iter().any(|p| {
            doms.iter()
                .all(|d| signed_distance(&d.surface, p).abs() > tol::LINEAR)
                && contains(&faces, p) == Some(inside)
        })
    };
    match op {
        Op::Union => false,
        Op::Intersection => deep(a, b, true) || deep(b, a, true),
        Op::Difference => deep(a, b, false),
    }
}
