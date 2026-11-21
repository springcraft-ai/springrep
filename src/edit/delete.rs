//! Faces taken out of a solid, and the gap they leave closed by the faces round it, each run on
//! along its own surface until they meet.

use std::collections::BTreeSet;

mod direct;

use crate::{
    blend::shaped::plane_of,
    boolean::{self, half_space, same_domain, solids_of},
    error::{Error, Result},
    geom::{Curve, Surface},
    math::{Aabb, Frame, Point3, Vec3},
    measure, primitive,
    topo::{Brep, FaceId, FaceName},
};

/// Marks a face of the solid that is to go.
fn gone() -> FaceName {
    FaceName::new("\u{1}delete_faces", "gone")
}

/// Marks a face of a tool that lies on a neighbour's surface.
fn run_on() -> FaceName {
    FaceName::new("\u{1}delete_faces", "run_on")
}

/// Marks a neighbour whose material holds the deleted faces.
fn holder() -> FaceName {
    FaceName::new("\u{1}delete_faces", "holder")
}

/// Which side of a neighbour's surface the deleted faces beside it lie on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    /// In its material: the gap was cut into the solid, and the surface runs on to fill it.
    Within,
    /// Out of it: the faces stood out of the solid, and the surface runs on to cut them off.
    Beyond,
}

/// `s` with `faces` taken out and the gap they leave closed by the faces across their edges, each
/// run on along its own surface until they meet. Where the deleted faces lie in a neighbour's
/// material, that neighbour bounds what the gap gains: the material of all such neighbours at
/// once. Where they lie out of it, the neighbour bounds what the gap loses: material of any such
/// neighbour is kept. A neighbour is a plane, a cylinder, a cone, a sphere or a torus; a cylinder
/// whose straight edges are not on the gap runs on only round its own angle. Every piece the
/// closing adds or removes must be bounded by the deleted faces and the neighbours' surfaces
/// alone, or the call is refused.
pub fn delete_faces(s: &Brep, faces: &[FaceId]) -> Result<Brep> {
    if faces.is_empty() {
        return Err(Error::InvalidInput("no faces to delete".into()));
    }
    match direct::direct(s, faces) {
        Some(out) => Ok(out),
        None => general(s, faces),
    }
}

/// [`delete_faces`] for any gap: the neighbours' material built as solids and combined, and the
/// pieces of it at the deleted faces added to the solid or cut from it.
fn general(s: &Brep, faces: &[FaceId]) -> Result<Brep> {
    let near = neighbours(s, faces);
    if near.is_empty() {
        return Err(Error::InvalidInput(
            "the faces to delete border no other face".into(),
        ));
    }
    let bb = measure::bbox(s);
    let pad = 0.5 * bb.diagonal() + 1.0;
    let bound = Aabb {
        min: bb.min - Vec3::new(pad, pad, pad),
        max: bb.max + Vec3::new(pad, pad, pad),
    };
    let mut within: Vec<Brep> = Vec::new();
    let mut within_outside: Vec<Brep> = Vec::new();
    let mut beyond: Vec<Brep> = Vec::new();
    let mut cuts: Vec<Brep> = Vec::new();
    let mut seen: Vec<(Surface, bool, Side)> = Vec::new();
    let mut holders: Vec<FaceId> = Vec::new();
    for (n, beside) in &near {
        let face = s.face(*n);
        if faces
            .iter()
            .any(|&d| same_domain(&s.face(d).surface, &face.surface))
        {
            return Err(Error::Unsupported(
                "deleting a face that lies on the surface of a face beside it".into(),
            ));
        }
        let side = side_of(s, *n, beside)?;
        if side == Side::Within {
            holders.push(*n);
            if let Some(cut) = sector_cut(s, *n, faces, &bound)? {
                cuts.push(cut);
                continue;
            }
        }
        if seen
            .iter()
            .any(|(g, r, k)| *r == face.reversed && *k == side && same_domain(g, &face.surface))
        {
            continue;
        }
        seen.push((face.surface.clone(), face.reversed, side));
        let (tool, out) = material(s, *n, &bound)?;
        match (side, out) {
            (Side::Within, false) => within.push(tool),
            (Side::Within, true) => within_outside.push(tool),
            (Side::Beyond, false) => beyond.push(tool),
            (Side::Beyond, true) => beyond.push(boolean::difference(&block(&bound)?, &tool)?),
        }
    }
    let mut q = None;
    if !within.is_empty() || !within_outside.is_empty() {
        let mut w = match within.split_first() {
            Some((first, rest)) => rest
                .iter()
                .try_fold(first.clone(), |a, b| boolean::intersection(&a, b))?,
            None => block(&bound)?,
        };
        for t in within_outside.iter().chain(&cuts) {
            w = boolean::difference(&w, t)?;
        }
        q = Some(w);
    }
    if !beyond.is_empty() {
        let refs: Vec<&Brep> = beyond.iter().collect();
        let u = boolean::union_all(&refs)?;
        q = Some(match q {
            Some(w) => boolean::intersection(&w, &u)?,
            None => u,
        });
    }
    let q = q.expect("a neighbour gives a tool");

    let mut marked = s.clone();
    for &f in faces {
        let mut names = marked.face(f).names.clone();
        names.insert(gone());
        marked.set_face_names(f, names);
    }
    // A piece cut out of the solid lies in the material of the neighbours that hold the gap, and
    // where a tool's face lies on one of them it may take that neighbour's own face.
    for &n in &holders {
        let mut names = marked.face(n).names.clone();
        names.insert(holder());
        marked.set_face_names(n, names);
    }
    let lost = gap_pieces(&boolean::difference(&marked, &q)?, true)?;
    let gained = gap_pieces(&boolean::difference(&q, &marked)?, false)?;
    if lost.is_none() && gained.is_none() {
        return Err(Error::Unsupported(
            "the faces round the gap run on along the deleted faces themselves".into(),
        ));
    }
    let mut out = marked;
    if let Some(lost) = lost {
        out = boolean::difference(&out, &lost)?;
    }
    if let Some(gained) = gained {
        out = boolean::union(&out, &gained)?;
    }
    if out.face_ids().any(|f| out.face(f).names.contains(&gone())) {
        return Err(Error::Unsupported(
            "the faces round the gap do not close it: part of a deleted face is left".into(),
        ));
    }
    if measure::volume(&out) <= 0.0 {
        return Err(Error::Unsupported("closing the gap leaves no solid".into()));
    }
    out.rename_faces(|n| (*n != gone() && *n != run_on() && *n != holder()).then(|| n.clone()));
    Ok(out)
}

/// Each face across an edge of `faces` and not among them, with the deleted faces it borders.
fn neighbours(s: &Brep, faces: &[FaceId]) -> Vec<(FaceId, Vec<FaceId>)> {
    let mut near: Vec<(FaceId, Vec<FaceId>)> = Vec::new();
    for &f in faces {
        for &lp in &s.face(f).loops {
            for &c in &s.lp(lp).coedges {
                for &o in s.edge_uses(s.coedge(c).edge) {
                    let g = s.coedge_face(o);
                    if faces.contains(&g) {
                        continue;
                    }
                    match near.iter_mut().find(|(n, _)| *n == g) {
                        Some((_, beside)) if !beside.contains(&f) => beside.push(f),
                        Some(_) => {}
                        None => near.push((g, vec![f])),
                    }
                }
            }
        }
    }
    near
}

/// Which side of neighbour `n`'s surface the deleted faces `beside` it lie on, from points spread
/// over them. Points on the surface, as along the edge they share, say nothing.
fn side_of(s: &Brep, n: FaceId, beside: &[FaceId]) -> Result<Side> {
    let face = s.face(n);
    let eps = 1e-7 * measure::bbox(s).diagonal().max(1.0);
    let (mut within, mut beyond) = (0, 0);
    for &d in beside {
        for (p, _) in measure::face_samples(s, d, 9) {
            let uv = face.surface.project(p);
            let mut out = face.surface.normal(uv.x, uv.y);
            if face.reversed {
                out = -out;
            }
            let h = (p - face.surface.eval(uv.x, uv.y)).dot(out);
            if h < -eps {
                within += 1;
            } else if h > eps {
                beyond += 1;
            }
        }
    }
    match (within, beyond) {
        (_, 0) if within > 0 => Ok(Side::Within),
        (0, _) if beyond > 0 => Ok(Side::Beyond),
        (0, 0) => Err(Error::Unsupported(
            "deleting faces that lie on the surface of a face beside them".into(),
        )),
        _ => Err(Error::Unsupported(
            "deleting faces that cross the surface of a face beside them".into(),
        )),
    }
}

/// The solid whose boundary is `n`'s surface within `bound` and beyond, its faces on that surface
/// named as `n` and marked, and whether `n`'s material is outside it rather than in it.
fn material(s: &Brep, n: FaceId, bound: &Aabb) -> Result<(Brep, bool)> {
    let face = s.face(n);
    let reach = bound.diagonal() + 1.0;
    let mid = bound.center();
    let tool = match &face.surface {
        Surface::Plane(_) => {
            let (o, out) = plane_of(s, n)?;
            return Ok((on_surface(s, n, half_space(s, o, -out)?), false));
        }
        Surface::Cylinder(c) => {
            let len = reach + mid.dist(c.frame.origin);
            let base = c.frame.origin - c.frame.z * len;
            let fr = Frame::new(base, c.frame.z, c.frame.x)?;
            primitive::cylinder(&fr, c.radius, 2.0 * len, "cyl")?
        }
        Surface::Sphere(sp) => primitive::sphere(&sp.frame, sp.radius, "ball")?,
        Surface::Torus(t) => {
            if t.major <= t.minor {
                return Err(Error::Unsupported(
                    "closing a gap against a torus whose tube reaches its axis".into(),
                ));
            }
            primitive::torus(&t.frame, t.major, t.minor, "ring")?
        }
        Surface::Cone(c) => {
            let slope = c.half_angle.tan();
            if slope.abs() < 1e-9 {
                return Err(Error::Unsupported(
                    "closing a gap against a cone with no taper".into(),
                ));
            }
            // The apex, and the axis pointing the way the cone widens.
            let apex = c.frame.origin - c.frame.z * (c.radius / slope);
            let up = if slope > 0.0 { c.frame.z } else { -c.frame.z };
            let len = reach + mid.dist(apex);
            let fr = Frame::new(apex, up, c.frame.x)?;
            primitive::cone(&fr, 0.0, len * slope.abs(), len, "cone")?
        }
        other => {
            return Err(Error::Unsupported(format!(
                "closing a gap against a {} face",
                other.kind()
            )));
        }
    };
    Ok((on_surface(s, n, tool), !material_inside(s, n)))
}

/// Whether `n`'s material lies on the side of its surface toward the axis, the centre or the
/// tube's centre circle: whether its outward normal points away from them. Judged where the face
/// lies farthest from them, since at a cone's apex, on its axis, the normal says nothing.
fn material_inside(s: &Brep, n: FaceId) -> bool {
    let face = s.face(n);
    let away_at = |p: Point3, out: Vec3| {
        let off_axis = |f: &Frame| {
            let d = p - f.origin;
            d - f.z * d.dot(f.z)
        };
        match &face.surface {
            Surface::Cylinder(c) => off_axis(&c.frame),
            Surface::Cone(c) => off_axis(&c.frame),
            Surface::Sphere(sp) => p - sp.frame.origin,
            Surface::Torus(t) => {
                let ring = t.frame.origin + off_axis(&t.frame).unit_or_zero() * t.major;
                p - ring
            }
            _ => out,
        }
    };
    // The points spread over the face, and its vertices for a face too narrow to catch any.
    let mut at = measure::face_samples(s, n, 9);
    for &lp in &face.loops {
        for &c in &s.lp(lp).coedges {
            let uv = face.surface.project(s.vertex(s.coedge_start(c)).point);
            let mut out = face.surface.normal(uv.x, uv.y);
            if face.reversed {
                out = -out;
            }
            at.push((face.surface.eval(uv.x, uv.y), out));
        }
    }
    at.into_iter()
        .map(|(p, out)| (away_at(p, out), out))
        .max_by(|a, b| a.0.norm().total_cmp(&b.0.norm()))
        .is_some_and(|(away, out)| away.dot(out) > 0.0)
}

/// `tool` with its faces on `n`'s surface named as `n` and marked, and its others unnamed.
fn on_surface(s: &Brep, n: FaceId, mut tool: Brep) -> Brep {
    let face = s.face(n);
    for f in tool.face_ids().collect::<Vec<_>>() {
        let names = if same_domain(&tool.face(f).surface, &face.surface) {
            let mut names = face.names.clone();
            names.insert(run_on());
            names
        } else {
            BTreeSet::new()
        };
        tool.set_face_names(f, names);
    }
    tool
}

/// For a cylinder `n` whose material holds the gap and which has a straight edge off the gap,
/// the part of the sector between those edges that lies outside the cylinder: there the
/// cylinder does not run on, and its neighbours across those edges bound the solid instead.
fn sector_cut(s: &Brep, n: FaceId, faces: &[FaceId], bound: &Aabb) -> Result<Option<Brep>> {
    let face = s.face(n);
    let Surface::Cylinder(c) = &face.surface else {
        return Ok(None);
    };
    if face.reversed {
        return Ok(None);
    }
    let samples = measure::face_samples(s, n, 9);
    let mut sector: Option<Brep> = None;
    for &lp in &face.loops {
        for &co in &s.lp(lp).coedges {
            let e = s.coedge(co).edge;
            let Some(Curve::Line(l)) = &s.edge(e).curve else {
                continue;
            };
            if l.dir.cross(c.frame.z).norm() > 1e-9 {
                continue;
            }
            let across: Vec<FaceId> = s.edge_uses(e).iter().map(|&o| s.coedge_face(o)).collect();
            if across.iter().all(|&g| g == n) || across.iter().any(|g| faces.contains(g)) {
                continue;
            }
            let radial = l.origin - c.frame.origin;
            let Some(m) = c.frame.z.cross(radial).unit() else {
                continue;
            };
            let at = |p: Point3| (p - c.frame.origin).dot(m);
            let reach = 1e-9 * bound.diagonal();
            let up = samples.iter().filter(|(p, _)| at(*p) > reach).count();
            let down = samples.iter().filter(|(p, _)| at(*p) < -reach).count();
            let towards = match (up, down) {
                (_, 0) if up > 0 => m,
                (0, _) if down > 0 => -m,
                _ => {
                    return Err(Error::Unsupported(
                        "closing a gap against a cylinder that turns more than half way round \
                         between its straight edges"
                            .into(),
                    ));
                }
            };
            let half = on_surface(s, n, half_space(s, c.frame.origin, towards)?);
            sector = Some(match sector {
                Some(w) => boolean::intersection(&w, &half)?,
                None => half,
            });
        }
    }
    let Some(sector) = sector else {
        return Ok(None);
    };
    let (cyl, _) = material(s, n, bound)?;
    Ok(Some(boolean::difference(&sector, &cyl)?))
}

/// The box `b` as a solid with unnamed faces.
fn block(b: &Aabb) -> Result<Brep> {
    let fr = Frame::new(b.min, Vec3::Z, Vec3::X)?;
    let mut out = primitive::block(&fr, b.max - b.min, "bound")?;
    out.rename_faces(|_| None);
    Ok(out)
}

/// The solids of `x` that hold part of a deleted face, which must be bounded by deleted faces and
/// the neighbours' surfaces alone, and with `holders`, the faces of neighbours that hold the gap;
/// `None` if there are none.
fn gap_pieces(x: &Brep, holders: bool) -> Result<Option<Brep>> {
    let mut keep = Vec::new();
    for sid in x.solid_ids() {
        let fs = x.solid_faces(sid);
        if !fs.iter().any(|&f| x.face(f).names.contains(&gone())) {
            continue;
        }
        let bad = fs.iter().any(|&f| {
            let names = &x.face(f).names;
            !(names.contains(&gone())
                || names.contains(&run_on())
                || holders && names.contains(&holder()))
        });
        if bad {
            return Err(Error::Unsupported(
                "the faces round the gap do not close it alone: their surfaces run on into other \
                 faces of the solid"
                    .into(),
            ));
        }
        keep.push(sid);
    }
    Ok((!keep.is_empty()).then(|| solids_of(x, &keep)))
}

#[cfg(test)]
mod tests;
