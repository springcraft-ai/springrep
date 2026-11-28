//! The points on a sheet's front within a distance of it, as the union of a piece for each face,
//! crease and corner (Rossignac and Requicha 1986): the slab each face sweeps moving along its
//! normal, the sector of the tube about each crease the front turns convex at that lies between
//! its two faces' normals, and the part of the ball about each corner where such creases meet that
//! lies within the cone of its faces' normals. Where the front turns concave the slabs overlap, cut
//! at the plane bisecting the crease where it closes to an acute wedge, and where two faces meet
//! tangentially their slabs share a side.

use std::collections::{BTreeSet, HashMap};
use std::f64::consts::{PI, TAU};

use crate::{
    boolean,
    edit::near::{ball, slab_of},
    error::{Error, Result},
    geom::{Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    pick::{concave_edges, convex_edges},
    profile::{NamedSegment, Profile},
    sweep::{extrude, revolve},
    tol,
    topo::{Brep, EdgeId, FaceId, FaceName, VertexId},
};

/// The feature of the names that mark, while a thicken runs, the faces lying on sheet face `i`.
pub(super) const ON_SHEET: &str = "#sheet";

fn unsupported<T>(why: &str) -> Result<T> {
    Err(Error::Unsupported(why.to_string()))
}

/// The name marking a face that lies on face `f` of the sheet.
pub(super) fn on_sheet(f: FaceId) -> FaceName {
    FaceName::new(ON_SHEET, f.index().to_string())
}

/// Face `f`'s outward normal at the point of its surface nearest `p`.
fn outward(s: &Brep, f: FaceId, p: Point3) -> Vec3 {
    let face = s.face(f);
    let uv = face.surface.project(p);
    let n = face.surface.normal(uv.x, uv.y);
    if face.reversed { -n } else { n }
}

/// The points within `d` of `sheet` on its front, as the union of its faces' slabs, its convex
/// creases' sectors and its corners' ball pieces. Each face lying on sheet face `f` is marked
/// [`on_sheet`]`(f)`, each face of a slab on the far side of `f` carries `f`'s names, its sides are
/// `<feature>.side` and the rest `<feature>.round`.
pub(super) fn front(sheet: &Brep, d: f64, feature: &str) -> Result<Brep> {
    let concave = concave_edges(sheet)?;
    let mut parts = Vec::new();
    for f in sheet.face_ids() {
        parts.push(mitred(
            sheet,
            f,
            face_slab(sheet, f, d, feature)?,
            &concave,
            d,
            feature,
        )?);
    }
    let convex = convex_edges(sheet)?;
    for &e in &convex {
        parts.push(sector(sheet, e, d, feature)?);
    }
    for v in corners(sheet, &convex)? {
        parts.push(corner(sheet, v, d, feature)?);
    }
    let refs: Vec<&Brep> = parts.iter().collect();
    boolean::union_all(&refs)
}

/// The slab sheet face `f` sweeps moving `d` along its normal, its faces named as [`front`] says.
fn face_slab(sheet: &Brep, f: FaceId, d: f64, feature: &str) -> Result<Brep> {
    let mut out = slab_of(sheet, f, d, feature)?;
    let surface = &sheet.face(f).surface;
    let side: BTreeSet<FaceName> = std::iter::once(FaceName::new(feature, "side")).collect();
    for g in out.face_ids().collect::<Vec<_>>() {
        let names = if boolean::same_domain(&out.face(g).surface, surface) {
            std::iter::once(on_sheet(f)).collect()
        } else if far_side(&out, g, sheet, f, d) {
            sheet.face(f).names.clone()
        } else {
            side.clone()
        };
        out.set_face_names(g, names);
    }
    Ok(out)
}

/// Whether face `g` of a slab lies `d` from sheet face `f`'s surface and parallel to it: the face
/// the slab's far side moved to.
fn far_side(slab: &Brep, g: FaceId, sheet: &Brep, f: FaceId, d: f64) -> bool {
    let Some(uv) = boolean::face_domain(slab, g).interior_point() else {
        return false;
    };
    let surface = &slab.face(g).surface;
    let p = surface.eval(uv.x, uv.y);
    let on = &sheet.face(f).surface;
    let st = on.project(p);
    let q = on.eval(st.x, st.y);
    let n = surface.normal(uv.x, uv.y);
    (p.dist(q) - d).abs() <= tol::LINEAR.max(1e-9 * d)
        && n.cross(on.normal(st.x, st.y)).norm() <= tol::ANGULAR
}

/// Slab `slab` of sheet face `f` less what it sweeps past the plane bisecting each crease of
/// `concave` beside `f` whose faces' normals are more than a right angle apart. There the front
/// closes to an acute wedge, and the slab's side at the crease, along `f`'s normal, would leave the
/// wedge across the other face; past the bisecting plane every point is nearer that face, whose
/// slab holds it. The slabs so cut meet in the plane, and their union is the wedge's points within
/// `d` of its faces, its faces on the sheet running all the way to the crease.
fn mitred(
    sheet: &Brep,
    f: FaceId,
    mut slab: Brep,
    concave: &[EdgeId],
    d: f64,
    feature: &str,
) -> Result<Brep> {
    for &e in concave {
        let Some((a, b)) = faces_of(sheet, e) else {
            continue;
        };
        let g = match (a == f, b == f) {
            (true, false) => b,
            (false, true) => a,
            _ => continue,
        };
        let x = sheet.edge(e);
        let acute = |p: Point3| outward(sheet, f, p).dot(outward(sheet, g, p)) < -tol::ANGULAR;
        let Some(Curve::Line(l)) = &x.curve else {
            if x.curve.as_ref().is_some_and(|c| acute(c.eval(x.range.0))) {
                return unsupported("a sheet creased to an acute wedge along a curve");
            }
            continue;
        };
        let (t0, t1) = x.range;
        let p0 = l.origin + l.dir * t0;
        if !acute(p0) {
            continue;
        }
        let Some((nf, ng)) = line_normals(sheet, e, f, g) else {
            return unsupported("a sheet creased to an acute wedge whose faces' normals turn");
        };
        let Some(bis) = (nf + ng).unit() else {
            return unsupported("a sheet folded flat on itself");
        };
        // The slab past the bisecting plane lies within d / cos α of the crease, α the angle
        // from either normal to the bisector: the moved face crosses the plane that far out.
        let reach = 2.0 * d / nf.dot(bis);
        let fr = Frame::new(p0, l.dir, bis)?;
        let away = if nf.dot(l.dir.cross(bis)) > 0.0 {
            reach
        } else {
            -reach
        };
        let p = Vec2::new;
        let mut pts = [
            p(-reach, 0.0),
            p(reach, 0.0),
            p(reach, away),
            p(-reach, away),
        ];
        if away < 0.0 {
            pts.reverse();
        }
        let side = format!("{feature}.side");
        let past = extrude(
            &Profile::polygon(&pts, &[side.as_str(); 4]),
            &fr,
            t1 - t0,
            feature,
        )?;
        slab = boolean::difference(&slab, &past)?;
    }
    Ok(slab)
}

/// The outward normals of faces `f` and `g` along straight crease `e`, if each keeps its
/// direction along it: where its surface is a plane, or a cylinder, a cone or an extrusion, whose
/// every straight line is a ruling (or lies in a flat strip) along which the tangent plane is
/// fixed. On other surfaces a line's normals may turn, and none are given.
fn line_normals(sheet: &Brep, e: EdgeId, f: FaceId, g: FaceId) -> Option<(Vec3, Vec3)> {
    let x = sheet.edge(e);
    let Some(Curve::Line(l)) = &x.curve else {
        return None;
    };
    let fixed = |h: FaceId| {
        matches!(
            sheet.face(h).surface,
            Surface::Plane(_) | Surface::Cylinder(_) | Surface::Cone(_) | Surface::Extrusion(_)
        )
    };
    let p0 = l.origin + l.dir * x.range.0;
    (fixed(f) && fixed(g)).then(|| (outward(sheet, f, p0), outward(sheet, g, p0)))
}

/// The two faces using edge `e`.
fn faces_of(sheet: &Brep, e: EdgeId) -> Option<(FaceId, FaceId)> {
    match sheet.edge_uses(e) {
        [a, b] => Some((sheet.coedge_face(*a), sheet.coedge_face(*b))),
        _ => None,
    }
}

/// The profile of the sector of radius `d` about `center` from direction `a` to direction `b`, the
/// lesser way round, its sides named `<feature>.round`.
fn sector_profile(center: Vec2, a: Vec2, b: Vec2, d: f64, feature: &str) -> Profile {
    let (from, to) = (a.y.atan2(a.x), b.y.atan2(b.x));
    let sweep = (to - from + PI).rem_euclid(TAU) - PI;
    let (start, sweep) = if sweep >= 0.0 {
        (from, sweep)
    } else {
        (to, -sweep)
    };
    let polar = |t: f64| center + Vec2::new(t.cos(), t.sin()) * d;
    let name = format!("{feature}.round");
    Profile::outline(vec![
        NamedSegment::line(center, polar(start), &name),
        NamedSegment::arc(center, d, start, sweep, &name),
        NamedSegment::line(polar(start + sweep), center, &name),
    ])
}

/// Every face of `b` named `<feature>.round`.
fn rounded(mut b: Brep, feature: &str) -> Brep {
    let name: BTreeSet<FaceName> = std::iter::once(FaceName::new(feature, "round")).collect();
    for g in b.face_ids().collect::<Vec<_>>() {
        b.set_face_names(g, name.clone());
    }
    b
}

/// The sector of the tube of radius `d` about crease `e` between its two faces' normals: a straight
/// crease's whose normals keep their directions along it, extruded, and a circle's whose normals
/// turn with it about its axis, turned.
fn sector(sheet: &Brep, e: EdgeId, d: f64, feature: &str) -> Result<Brep> {
    let Some((f, g)) = faces_of(sheet, e) else {
        return unsupported("a crease of a sheet used by other than two faces");
    };
    let x = sheet.edge(e);
    let (t0, t1) = x.range;
    let at = |k: usize| t0 + (t1 - t0) * k as f64 / 4.0;
    let fixed = || unsupported("a rounded crease whose faces' normals turn along it");
    match &x.curve {
        Some(Curve::Line(l)) => {
            let p0 = l.origin + l.dir * t0;
            let Some((nf, ng)) = line_normals(sheet, e, f, g) else {
                return fixed();
            };
            let fr = Frame::new(p0, l.dir, nf)?;
            let flat = |v: Vec3| {
                let l = fr.dir_to_local(v);
                Vec2::new(l.x, l.y)
            };
            let profile = sector_profile(Vec2::ZERO, flat(nf), flat(ng), d, feature);
            Ok(rounded(extrude(&profile, &fr, t1 - t0, feature)?, feature))
        }
        Some(Curve::Circle(c)) => {
            let axis = c.frame.z;
            // A normal as (out from the axis, along it) at parameter t, if it has no part along
            // the circle.
            let meridian = |n: Vec3, t: f64| {
                let r = c.frame.radial(t);
                (n.dot(axis.cross(r)).abs() <= tol::ANGULAR)
                    .then(|| Vec2::new(n.dot(r), n.dot(axis)))
            };
            let curve = x.curve.as_ref().expect("a circle");
            let normals = |t: f64| {
                let p = curve.eval(t);
                meridian(outward(sheet, f, p), t).zip(meridian(outward(sheet, g, p), t))
            };
            let Some((nf, ng)) = normals(t0) else {
                return fixed();
            };
            for k in 1..=4 {
                match normals(at(k)) {
                    Some((a, b)) if a.dist(nf) <= tol::ANGULAR && b.dist(ng) <= tol::ANGULAR => {}
                    _ => return fixed(),
                }
            }
            let center = Vec2::new(c.radius, 0.0);
            let profile = sector_profile(center, nf, ng, d, feature);
            let reach = (0..=64).any(|k| {
                let a = nf.y.atan2(nf.x);
                let b = ng.y.atan2(ng.x);
                let sweep = (b - a + PI).rem_euclid(TAU) - PI;
                let t = a + sweep * k as f64 / 64.0;
                center.x + d * t.cos() < tol::LINEAR
            });
            if reach {
                return unsupported("a rounded crease about a circle whose round reaches its axis");
            }
            let x0 = c.frame.radial(t0);
            let plane = Frame::new(c.frame.origin, -axis.cross(x0), x0)?;
            let turn = (t1 - t0).min(TAU);
            Ok(rounded(
                revolve(&profile, &plane, c.frame.origin, axis, turn, feature)?,
                feature,
            ))
        }
        _ => unsupported("a rounded crease other than a line or a circle"),
    }
}

/// The vertices of `sheet` where two or more creases in `convex` meet other than running on into
/// each other tangentially: the corners whose ball pieces the sectors leave uncovered. An error
/// where such a vertex is on the sheet's boundary or on a crease turned the other way.
fn corners(sheet: &Brep, convex: &[EdgeId]) -> Result<Vec<VertexId>> {
    let mut leaving: HashMap<VertexId, Vec<Vec3>> = HashMap::new();
    for &e in convex {
        let x = sheet.edge(e);
        let Some(c) = &x.curve else { continue };
        let (_, d0) = c.d1(x.range.0);
        let (_, d1) = c.d1(x.range.1);
        leaving.entry(x.start).or_default().push(d0);
        leaving.entry(x.end).or_default().push(-d1);
    }
    let concave = concave_edges(sheet)?;
    let mut out = Vec::new();
    for (v, dirs) in leaving {
        let smooth = dirs.len() == 2
            && dirs[0]
                .unit()
                .zip(dirs[1].unit())
                .is_some_and(|(a, b)| a.dot(b) <= -1.0 + tol::ANGULAR);
        if dirs.len() < 2 || smooth {
            continue;
        }
        let touches = |e: EdgeId| {
            let x = sheet.edge(e);
            x.start == v || x.end == v
        };
        if sheet
            .edge_ids()
            .any(|e| touches(e) && (sheet.edge_uses(e).len() == 1 || concave.contains(&e)))
        {
            return unsupported(
                "a rounded corner of a sheet on its boundary or beside a crease turned the other way",
            );
        }
        out.push(v);
    }
    out.sort();
    Ok(out)
}

/// The part of the ball of radius `d` about corner `v` within the cone its faces' normals span:
/// the ball cut by each plane through `v` that two of the normals span and the rest lie on one side
/// of.
fn corner(sheet: &Brep, v: VertexId, d: f64, feature: &str) -> Result<Brep> {
    let p = sheet.vertex(v).point;
    let mut normals: Vec<Vec3> = Vec::new();
    for c in sheet.coedge_ids() {
        let x = sheet.edge(sheet.coedge(c).edge);
        if x.start != v && x.end != v {
            continue;
        }
        let n = outward(sheet, sheet.coedge_face(c), p);
        if !normals
            .iter()
            .any(|m| m.cross(n).norm() <= tol::ANGULAR && m.dot(n) > 0.0)
        {
            normals.push(n);
        }
    }
    let mut piece = ball(p, d, feature)?;
    for i in 0..normals.len() {
        for j in i + 1..normals.len() {
            let Some(m) = normals[i].cross(normals[j]).unit() else {
                continue;
            };
            let sides: Vec<f64> = normals.iter().map(|n| n.dot(m)).collect();
            let m = if sides.iter().all(|&s| s >= -tol::ANGULAR) {
                m
            } else if sides.iter().all(|&s| s <= tol::ANGULAR) {
                -m
            } else {
                continue;
            };
            let outside = boolean::half_space(&piece, p, -m)?;
            piece = boolean::difference(&piece, &outside)?;
        }
    }
    Ok(piece)
}
