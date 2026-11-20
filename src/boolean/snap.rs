//! Faces of the second input lying a hair off a face of the first, moved onto that face's surface
//! before the boolean: within [`SNAP`] all over, or within [`SNAP_GRAZING`] where the two cross at
//! an angle below [`tol::TANGENT`], which the tolerance cannot place. A part placed by a rounded
//! pose sits in its seat that way: a cylinder 1 µm off its seat's axis of 24.5 mm radius crosses it
//! at 4e-8 rad, and the crescent between them has tips that run within the tolerance of both for
//! millimetres. Moved onto one surface, the faces are one face where they overlap, as a part sitting
//! in its seat is meant to be, and the volume moved is at most the gap times the face's area. Faces
//! that cross at a steeper angle are left for the boolean, which places their crossing.

use std::collections::HashMap;

use crate::{
    check::{self, coedge_params},
    edit::tweak,
    geom::Surface,
    math::{Aabb, Frame, Point3},
    measure, tol,
    topo::{Brep, FaceId, VertexId},
};

use super::{
    csi::{Tracker, signed_distance},
    ssi::same_domain,
};

/// How far [`snapped`] moves shapes onto each other: [`Reach::Grazing`] also moves cylinders of one
/// radius within [`SNAP_GRAZING`] of one axis onto it, for a boolean that failed without.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Reach {
    Exact,
    Grazing,
}

/// How far apart two faces may lie all over and still be moved onto one surface, in mm.
pub(crate) const SNAP: f64 = 1e-6;

/// How far apart two faces that cross below [`tol::TANGENT`] may lie and still be moved onto one
/// surface, in mm: a cylinder of 100 mm radius off its seat's axis by this crosses it at that angle.
pub(crate) const SNAP_GRAZING: f64 = 1e-4;

/// How far moving faces may move a vertex, in multiples of how far they were moved: further means
/// a neighbour meets the face at an angle so shallow that its corner slides along it.
const MOST_MOVED: f64 = 10.0;

/// How far apart two axes may turn, in radians, and still be turned onto one: a pose's rounding
/// leaves a part's axis 2e-10 off its seat's, which moves a point 100 mm along it by 2e-8 mm.
const COAXIAL_TILT: f64 = 1e-8;

/// Points along every edge of face `f`, where two surfaces of one kind a hair apart are furthest
/// apart over the face: their gap is linear across a plane and varies round a cylinder's turn,
/// which the boundary's circles cover.
fn boundary_points(b: &Brep, f: FaceId) -> Vec<Point3> {
    let mut out = Vec::new();
    for &lp in &b.face(f).loops {
        for &c in &b.lp(lp).coedges {
            let e = b.edge(b.coedge(c).edge);
            match &e.curve {
                Some(curve) => out.extend(coedge_params(b, c, 16).iter().map(|&t| curve.eval(t))),
                None => out.push(b.vertex(e.start).point),
            }
        }
    }
    out
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

/// Whether `s` and `t` are surfaces of one kind that might lie a hair apart: it rules out most pairs
/// before any sampling.
fn alike(s: &Surface, t: &Surface) -> bool {
    let near = |x: f64, y: f64| (x - y).abs() <= SNAP_GRAZING;
    let parallel = |a: crate::math::Vec3, b: crate::math::Vec3| a.cross(b).norm() <= 1e-3;
    match (s, t) {
        (Surface::Plane(p), Surface::Plane(q)) => parallel(p.frame.z, q.frame.z),
        (Surface::Cylinder(p), Surface::Cylinder(q)) => {
            near(p.radius, q.radius) && parallel(p.frame.z, q.frame.z)
        }
        (Surface::Sphere(p), Surface::Sphere(q)) => near(p.radius, q.radius),
        (Surface::Cone(p), Surface::Cone(q)) => {
            (p.half_angle - q.half_angle).abs() <= 1e-3 && parallel(p.frame.z, q.frame.z)
        }
        (Surface::Torus(p), Surface::Torus(q)) => {
            near(p.major, q.major) && near(p.minor, q.minor) && parallel(p.frame.z, q.frame.z)
        }
        // Spline surfaces have no closed form to compare; sampling decides.
        (Surface::Extrusion(_), Surface::Extrusion(_))
        | (Surface::Revolution(_), Surface::Revolution(_))
        | (Surface::Ruled(_), Surface::Ruled(_))
        | (Surface::Nurbs(_), Surface::Nurbs(_)) => s != t,
        _ => false,
    }
}

/// Surface `s` with its natural normal turned the way `like`'s points: a plane given the face's
/// normal flipped would turn the face inside out.
fn facing(s: &Surface, like: &Surface) -> Surface {
    match (s, like) {
        (Surface::Plane(p), Surface::Plane(q)) if p.frame.z.dot(q.frame.z) < 0.0 => {
            match crate::math::Frame::new(p.frame.origin, -p.frame.z, p.frame.x) {
                Ok(frame) => Surface::Plane(crate::geom::Plane { frame }),
                Err(_) => s.clone(),
            }
        }
        _ => s.clone(),
    }
}

/// Each vertex of a face of `b` that `moved` moves, with the vertex of `a` within `reach` of it.
/// A vertex its new surfaces leave free to slide along a curve, as a seam's end on a circle is,
/// would otherwise keep how far along the curve the move left it: a cylinder moved across its axis
/// by 1.1e-7 mm onto its seat's surface has its seam 1.1e-7 mm round the circle from the seat's,
/// and the boolean would cut a piece of edge just over the tolerance long between the two.
fn pins(
    a: &Brep,
    b: &Brep,
    moved: &HashMap<FaceId, Surface>,
    reach: f64,
) -> HashMap<VertexId, Point3> {
    let mut on_moved: Vec<VertexId> = Vec::new();
    for &f in moved.keys() {
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                on_moved.push(b.coedge_start(c));
            }
        }
    }
    on_moved.sort();
    on_moved.dedup();
    let mut zone = Aabb::EMPTY;
    for &v in &on_moved {
        zone.add(b.vertex(v).point);
    }
    let zone = padded(&zone, reach);
    let theirs: Vec<Point3> = a
        .vertex_ids()
        .map(|w| a.vertex(w).point)
        .filter(|p| (0..3).all(|k| p[k] >= zone.min[k] && p[k] <= zone.max[k]))
        .collect();
    on_moved
        .into_iter()
        .filter_map(|v| {
            let p = b.vertex(v).point;
            let (d, q) = theirs
                .iter()
                .map(|&q| (q.dist(p), q))
                .min_by(|x, y| x.0.total_cmp(&y.0))?;
            (d <= reach).then_some((v, q))
        })
        .collect()
}

/// How far a face of `sb` through `pts` lies from `sa` at most, where it is to be moved onto it:
/// within [`SNAP`] all over, or within [`SNAP_GRAZING`] where it crosses `sa` (lies on both sides)
/// with the two surfaces' normals within [`tol::TANGENT`] of each other everywhere.
fn snaps(sa: &Surface, sb: &Surface, pts: &[Point3]) -> Option<f64> {
    // Most faces tried lie far off at their first point: stop there, and follow the boundary
    // with each projection from the last rather than afresh.
    let mut track = Tracker::new(sa);
    let mut d = Vec::with_capacity(pts.len());
    for &p in pts {
        let x = track.at(p);
        if x.abs() > SNAP_GRAZING {
            return None;
        }
        d.push(x);
    }
    let worst = d.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if worst <= SNAP {
        return Some(worst);
    }
    // Two planes cross along a line found in closed form however small the angle, and a wedge
    // between them may be all the material there is: only a curved pair is moved this far.
    if worst > SNAP_GRAZING || matches!((sa, sb), (Surface::Plane(_), Surface::Plane(_))) {
        return None;
    }
    let crosses = d.iter().any(|&x| x > 0.5 * worst) && d.iter().any(|&x| x < -0.5 * worst);
    let grazing = pts.iter().all(|&p| {
        let (ua, ub) = (sa.project(p), sb.project(p));
        sa.normal(ua.x, ua.y).cross(sb.normal(ub.x, ub.y)).norm() <= tol::TANGENT
    });
    (crosses && grazing).then_some(worst)
}

/// `sb` moved across its axis to touch the parallel cylinder `sa` exactly, and how far, where it
/// touches it to within [`SNAP`] but not exactly, inside or out. Cylinders whose curvatures nearly
/// match, one inside the other, cross along two lines far apart for an overlap far under the
/// tolerance: a rib of 1.95 mm radius 3e-9 into a notch of 2 mm crosses it 1.4 um either side of
/// where they touch. The end faces' circles cross there too, and the sliver between, thinner
/// than the tolerance, has no side to tell. Touching exactly, they meet along one line.
fn tangent(sa: &Surface, sb: &Surface) -> Option<(Surface, f64)> {
    let (Surface::Cylinder(a), Surface::Cylinder(b)) = (sa, sb) else {
        return None;
    };
    let axis = a.frame.z;
    if axis.cross(b.frame.z).norm() > tol::ANGULAR {
        return None;
    }
    let w = b.frame.origin - a.frame.origin;
    let off = w - axis * w.dot(axis);
    let d = off.norm();
    if d <= tol::LINEAR {
        return None;
    }
    [a.radius + b.radius, (a.radius - b.radius).abs()]
        .into_iter()
        .filter(|&touch| touch > tol::LINEAR)
        .map(|touch| touch - d)
        .find(|&by| by != 0.0 && by.abs() <= SNAP)
        .map(|by| {
            let mut moved = b.clone();
            moved.frame.origin = b.frame.origin + off * (by / d);
            (Surface::Cylinder(moved), by.abs())
        })
}

/// `sb`, a cylinder or a cone, turned parallel to the axis of the cylinder or cone `sa` where it
/// lies within [`COAXIAL_TILT`] of it, and moved across onto that axis where the two lie within
/// [`SNAP`], with how far it moved across. Two surfaces of revolution about one axis meet in
/// circles; a hair off it, they meet in curves that are those circles to the tolerance, which only
/// a trace can follow and which a press fit's chamfer, crossing its bore, makes at every point
/// round it. Parallel cylinders a hair apart cross in lines; turned 2e-10 off, in curves that run
/// nearly along each other for the whole length, which a trace loses where they meet a seam.
fn coaxial(sa: &Surface, sb: &Surface, reach: Reach) -> Option<(Surface, Option<f64>)> {
    let frame = |s: &Surface| match s {
        Surface::Cylinder(c) => Some(c.frame),
        Surface::Cone(c) => Some(c.frame),
        _ => None,
    };
    let (fa, fb) = (frame(sa)?, frame(sb)?);
    let tilt = fa.z.cross(fb.z).norm();
    if tilt > COAXIAL_TILT {
        return None;
    }
    let w = fb.origin - fa.origin;
    let along = fa.z * w.dot(fa.z);
    let off = w - along;
    let d = off.norm();
    // Cylinders of one radius a hair off one axis cross at an angle of about the gap over the
    // radius, where ends and edges meeting them are placed no better than the tolerance over it.
    let across = match (reach, sa, sb) {
        (Reach::Grazing, Surface::Cylinder(a), Surface::Cylinder(b)) => {
            if (a.radius - b.radius).abs() <= d {
                SNAP_GRAZING
            } else {
                SNAP
            }
        }
        _ => SNAP,
    };
    let onto_axis = d <= across;
    // Axes within the angular tolerance are parallel to the boolean already; further off, only
    // cylinders that come within the grazing reach of each other cross where it cannot follow.
    let near = match (sa, sb) {
        (Surface::Cylinder(a), Surface::Cylinder(b)) => {
            [a.radius + b.radius, (a.radius - b.radius).abs()]
                .iter()
                .any(|&touch| (touch - d).abs() <= SNAP_GRAZING)
        }
        _ => false,
    };
    if !onto_axis && (tilt <= tol::ANGULAR || !near) {
        return None;
    }
    let origin = if onto_axis {
        fa.origin + along
    } else {
        fb.origin
    };
    let z = if fa.z.dot(fb.z) < 0.0 { -fa.z } else { fa.z };
    let onto = Frame::new(origin, z, fb.x).ok()?;
    if onto == fb {
        return None;
    }
    let moved = match sb {
        Surface::Cylinder(c) => {
            let mut c = c.clone();
            c.frame = onto;
            Surface::Cylinder(c)
        }
        Surface::Cone(c) => {
            let mut c = c.clone();
            c.frame = onto;
            Surface::Cone(c)
        }
        _ => return None,
    };
    Some((moved, onto_axis.then_some(d)))
}

fn padded(b: &Aabb, by: f64) -> Aabb {
    let d = crate::math::Vec3::new(by, by, by);
    Aabb {
        min: b.min - d,
        max: b.max + d,
    }
}

fn overlap(a: &Aabb, b: &Aabb) -> bool {
    (0..3).all(|k| a.min[k] <= b.max[k] && b.min[k] <= a.max[k])
}

/// `b` with each of its faces that lies within [`SNAP`] of a face of `a` all over moved onto that
/// face's surface; `None` where none does, or where moving them would not keep `b` a valid solid
/// with every vertex within [`MOST_MOVED`] of where it was.
pub(crate) fn snapped(a: &Brep, b: &Brep, how: Reach) -> Option<Brep> {
    // A face of `a` whose hull box, which holds its box, misses every face of `b` by the reach
    // cannot pair with one, and its box is not worked out.
    let mut reach = Aabb::EMPTY;
    for f in b.face_ids() {
        reach.union(&measure::face_hull_box(b, f));
    }
    let reach = padded(&reach, SNAP_GRAZING);
    let a_faces: Vec<(FaceId, Aabb)> = a
        .face_ids()
        .filter(|&f| overlap(&measure::face_hull_box(a, f), &reach))
        .map(|f| (f, padded(&measure::face_box(a, f), SNAP_GRAZING)))
        .collect();
    let mut moved: HashMap<FaceId, Surface> = HashMap::new();
    let mut gap: f64 = SNAP;
    // Faces already within the tolerance of a face of `a`, moved onto it only when other faces
    // move: two nearly parallel faces meet at a vertex that slides along their line by the gap
    // over the angle between them, so one left a hair off would carry it past the tolerance.
    let mut within: HashMap<FaceId, Surface> = HashMap::new();
    for fb in b.face_ids() {
        let sb = &b.face(fb).surface;
        let bb = measure::face_box(b, fb);
        let mut points: Option<Vec<Point3>> = None;
        let mut turn: Option<Surface> = None;
        for (fa, ab) in &a_faces {
            let sa = &a.face(*fa).surface;
            if !overlap(ab, &bb) {
                continue;
            }
            // Turned only, it may still touch `sa` or lie a hair off it, which the tests below
            // read from the turned surface.
            let turned = match coaxial(sa, sb, how) {
                Some((onto, Some(by))) => {
                    moved.insert(fb, onto);
                    gap = gap.max(by);
                    break;
                }
                Some((onto, None)) => Some(onto),
                None => None,
            };
            let sb = turned.as_ref().unwrap_or(sb);
            if turn.is_none() {
                turn = turned.clone();
            }
            if let Some((touching, by)) = tangent(sa, sb) {
                moved.insert(fb, touching);
                gap = gap.max(by);
                break;
            }
            if !alike(sa, sb) {
                continue;
            }
            if same_domain(sa, sb) {
                if sa != sb {
                    within.entry(fb).or_insert_with(|| facing(sa, sb));
                    // Half the tolerance off or more all over is where reading a distance with or
                    // without the ulps to spare decides differently: a face there moves on its own.
                    let pts = points.get_or_insert_with(|| boundary_points(b, fb));
                    if pts
                        .iter()
                        .all(|&p| signed_distance(sa, p).abs() >= 0.5 * tol::LINEAR)
                    {
                        moved.insert(fb, facing(sa, sb));
                    }
                }
                continue;
            }
            let pts = points.get_or_insert_with(|| boundary_points(b, fb));
            if let Some(worst) = snaps(sa, sb, pts) {
                // A spline face has no closed form to call it the same surface by; one within the
                // tolerance all over already is (a symmetric part turned onto itself), and moves
                // only with others, as a plane within it does.
                if worst <= tol::LINEAR && !analytic(sb) {
                    within.entry(fb).or_insert_with(|| facing(sa, sb));
                    break;
                }
                moved.insert(fb, facing(sa, sb));
                gap = gap.max(worst);
                break;
            }
        }
        if let Some(t) = turn
            && !moved.contains_key(&fb)
            && !within.contains_key(&fb)
        {
            moved.insert(fb, t);
        }
    }
    if moved.is_empty() {
        return None;
    }
    for (f, s) in within {
        moved.entry(f).or_insert(s);
    }
    // All at once where the moved faces' corners meet; otherwise one at a time, keeping each that
    // leaves the rest meeting: two neighbours moved onto surfaces of different faces may not.
    if let Some(out) = moved_whole(a, b, &moved, gap) {
        return Some(out);
    }
    let mut order: Vec<FaceId> = moved.keys().copied().collect();
    order.sort();
    let mut kept: HashMap<FaceId, Surface> = HashMap::new();
    let mut out = None;
    for f in order {
        let mut trial = kept.clone();
        trial.insert(f, moved[&f].clone());
        if let Some(o) = moved_whole(a, b, &trial, gap) {
            kept = trial;
            out = Some(o);
        }
    }
    out
}

/// `b` with the faces of `moved` on their new surfaces, where that keeps it a valid solid with no
/// vertex moved further than [`MOST_MOVED`] times `gap`.
fn moved_whole(a: &Brep, b: &Brep, moved: &HashMap<FaceId, Surface>, gap: f64) -> Option<Brep> {
    let out = tweak::move_faces_pinned(b, moved, &pins(a, b, moved, MOST_MOVED * gap)).ok()?;
    let far = b
        .vertex_ids()
        .zip(out.vertex_ids())
        .map(|(v, w)| b.vertex(v).point.dist(out.vertex(w).point))
        .fold(0.0, f64::max);
    (far <= MOST_MOVED * gap && check::check(&out).is_empty()).then_some(out)
}
