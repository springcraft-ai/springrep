//! Three or more chamfers meeting at a convex corner of planes, every edge there picked: on each
//! face at the corner the contact lines of its two chamfers cross, and the corner is cut flat by
//! the plane through those points. The chamfers mitre at the corner as they do one after another
//! ([`super::corner_trims`]), and the corner plane then takes off what lies past it.

use std::collections::BTreeSet;

use super::{Outline, Picked, Section, end_index, faces_at, leaving, run};
use crate::{
    Error, Result, boolean,
    geom::Surface,
    math::{Frame, Point3, Vec2, Vec3},
    profile::Profile,
    sweep, tol,
    topo::{Brep, FaceName, VertexId},
};

/// The plane that closes a chamfered corner: a point on it, and its normal pointing to the corner.
#[derive(Clone, Copy, Debug)]
pub(super) struct Flat {
    centre: Point3,
    normal: Vec3,
    /// The furthest from `centre` that the plane meets an edge at the corner.
    reach: f64,
    /// How far the corner stands off the plane.
    height: f64,
}

/// The flat corner at `v`, where the picked edges `at` meet; an error where the corner is not one
/// of convex edges between planes, all picked, or their contact lines do not cross in one plane.
pub(super) fn find(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    outlines: &[Outline],
    at: &[usize],
    v: VertexId,
) -> Result<Flat> {
    let unsupported = |why: &str| Err(Error::Unsupported(format!("chamfers meeting at {why}")));
    let corner = body.vertex(v).point;
    let faces = faces_at(body, v);
    if faces
        .iter()
        .any(|&f| !matches!(body.face(f).surface, Surface::Plane(_)))
    {
        return unsupported("a corner of faces other than planes");
    }
    let mut edges_at = BTreeSet::new();
    for c in body.coedge_ids().filter(|&c| body.coedge_start(c) == v) {
        if run::convex(body, c) != Some(true) {
            return unsupported("a corner whose edges are not all convex");
        }
        edges_at.insert(body.coedge(c).edge);
    }
    let picked_at: BTreeSet<_> = at.iter().map(|&i| picked[i].edge).collect();
    if picked_at != edges_at {
        return unsupported("a corner with an edge that is not chamfered");
    }
    // Where each chamfer's contact line on a face leaves the corner's section, and its direction.
    let contact = |i: usize, k: usize| -> Result<(Point3, Vec3)> {
        let (s, o, pk) = (&sections[i], &outlines[i], &picked[i]);
        let (dir, w) = o.reach[k];
        let end = end_index(body, pk.edge, v) == 1;
        let d = leaving(body, pk.edge, v)
            .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
        Ok((s.path.place(s.p + dir * w, end), d))
    };
    let mut points = Vec::new();
    for &f in &faces {
        let lines: Vec<(Point3, Vec3)> = at
            .iter()
            .filter_map(|&i| {
                let k = picked[i].faces.iter().position(|&g| g == f)?;
                Some(contact(i, k))
            })
            .collect::<Result<_>>()?;
        let [(a, da), (b, db)] = lines[..] else {
            return unsupported("a corner whose faces are not each between two chamfers");
        };
        // a + s·da = b + t·db, by least squares in the face's plane.
        let (aa, ab, bb) = (da.dot(da), da.dot(db), db.dot(db));
        let det = aa * bb - ab * ab;
        if det.abs() <= tol::ANGULAR {
            return unsupported("a corner whose contact lines on a face run side by side");
        }
        let r = b - a;
        let s = (bb * r.dot(da) - ab * r.dot(db)) / det;
        points.push(a + da * s);
    }
    let centre = points.iter().fold(Point3::ZERO, |acc, &p| acc + p) * (1.0 / points.len() as f64);
    let inward = edges_at
        .iter()
        .filter_map(|&e| leaving(body, e, v))
        .fold(Vec3::ZERO, |acc, d| acc + d)
        .unit()
        .ok_or_else(|| Error::Numerical("a corner whose edges cancel".into()))?;
    let frame = Frame::new(corner, inward, inward.any_perpendicular())?;
    let angle = |p: Point3| {
        let l = frame.to_local(p);
        l.y.atan2(l.x)
    };
    points.sort_by(|a, b| angle(*a).total_cmp(&angle(*b)));
    // Newell's normal of the polygon the points make round the corner.
    let mut normal = Vec3::ZERO;
    for (k, &p) in points.iter().enumerate() {
        let q = points[(k + 1) % points.len()];
        normal += (p - centre).cross(q - centre);
    }
    let mut normal = normal
        .unit()
        .ok_or_else(|| Error::Numerical("a corner whose contact points fall on a line".into()))?;
    if points
        .iter()
        .any(|&p| (p - centre).dot(normal).abs() > tol::LINEAR)
    {
        return unsupported("a corner whose contact lines do not cross in one plane");
    }
    if normal.dot(corner - centre) < 0.0 {
        normal = -normal;
    }
    let height = normal.dot(corner - centre);
    if height <= tol::LINEAR {
        return unsupported("a corner the plane of its contact points does not stand clear of");
    }
    let mut reach = 0.0f64;
    for &e in &edges_at {
        let d = leaving(body, e, v).expect("an edge with a tangent");
        let along = -height / normal.dot(d);
        let edge = body.edge(e);
        let length = body
            .vertex(edge.start)
            .point
            .dist(body.vertex(edge.end).point);
        if !(along > 0.0 && along < length - tol::LINEAR) {
            return unsupported("a corner whose plane runs past the end of one of its edges");
        }
        reach = reach.max((corner + d * along).dist(centre));
    }
    Ok(Flat {
        centre,
        normal,
        reach,
        height,
    })
}

/// `out` less the material past each flat corner's plane, its face there named
/// `<feature>.corner`.
pub(super) fn cut(out: Brep, flats: &[Flat], feature: &str) -> Result<Brep> {
    let mut out = out;
    for flat in flats {
        let frame = Frame::new(flat.centre, flat.normal, flat.normal.any_perpendicular())?;
        let side = 2.0 * flat.reach * 1.25;
        let profile = Profile::rect(Vec2::ZERO, side, side);
        let mut tool = sweep::extrude(&profile, &frame, 2.0 * flat.height, feature)?;
        for f in tool.face_ids().collect::<Vec<_>>() {
            let on_plane = match &tool.face(f).surface {
                Surface::Plane(p) => {
                    (p.frame.origin - flat.centre).dot(flat.normal).abs() <= tol::LINEAR
                        && p.frame.z.cross(flat.normal).norm() <= tol::ANGULAR
                }
                _ => false,
            };
            let names = if on_plane {
                BTreeSet::from([FaceName::new(feature, "corner")])
            } else {
                BTreeSet::new()
            };
            tool.set_face_names(f, names);
        }
        out = boolean::difference(&out, &tool)?;
    }
    Ok(out)
}
