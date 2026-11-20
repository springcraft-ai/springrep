//! Changes to a solid's faces: faces moved, tilted or taken out, a solid hollowed, grown by a
//! distance or stretched, and the slab a face sweeps. Each builds exact tools and combines them
//! with the booleans, blends and face moves.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use crate::{
    blend,
    blend::shaped::plane_of,
    boolean,
    error::{Error, Result},
    geom::{Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    pick::{concave_edges, convex_edges},
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::revolve,
    tol,
    topo::{Brep, EdgeId, FaceId, FaceName},
};

mod affine;
mod delete;
pub(crate) mod near;
mod parallel;
mod parted;
mod rounded;
pub(crate) mod tweak;

pub use affine::{Affine, affine};
pub use delete::delete_faces;
pub use parted::draft_parted;
pub use rounded::shell_rounded;
pub use tweak::{draft, move_faces, move_faces_by, offset_faces};

fn named(segment: Segment, name: &str) -> NamedSegment {
    NamedSegment::new(segment, name)
}

/// Which side of a solid's faces a shell's wall lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wall {
    /// Inside the solid: its outside stays where it is.
    Inward,
    /// Outside the solid: its faces become the wall's inside.
    Outward,
}

/// `s` hollowed to a wall `thickness` thick, open through the faces of `open`. The cavity's faces
/// are `<feature>.inner`. An inward wall leaves every face of `s` in place with its names. An
/// outward wall's outside is new and carries no name, and each face of `s` is the wall's inside,
/// keeping its names beside `inner`.
pub fn shell(s: &Brep, open: &[FaceId], thickness: f64, wall: Wall, feature: &str) -> Result<Brep> {
    if thickness.is_nan() || thickness <= tol::LINEAR {
        return Err(Error::InvalidInput(
            "a shell's wall must be thicker than the tolerance".into(),
        ));
    }
    let inner = FaceName::new(feature, FaceName::INNER);
    // Every face moved `by` along its outward normal, the open ones `open_by`.
    // A wall that closes beside an edge turned the other way leaves no corner to meet sharp: the
    // points within `thickness` of that edge would round it, which the moved faces cannot.
    let moved = |by: f64, open_by: f64, turned: Vec<EdgeId>| {
        let by: HashMap<FaceId, f64> = s
            .face_ids()
            .map(|f| (f, if open.contains(&f) { open_by } else { by }))
            .filter(|&(_, d)| d != 0.0)
            .collect();
        let (out, closed) = tweak::offset_faces_closing(s, &by)?;
        closed_beside(
            s,
            &turned,
            &closed,
            "a shell whose walls close past an edge",
        )?;
        Ok::<_, Error>(out)
    };
    // The open faces stay where they are: moved out, they would carry the cut past their plane
    // into a wall that stands beside the opening. The cut lies on them there and leaves no skin.
    match wall {
        Wall::Inward => {
            let sharp = || {
                let mut cavity = moved(-thickness, 0.0, concave_edges(s)?)?;
                for f in cavity.face_ids().collect::<Vec<_>>() {
                    cavity.set_face_names(f, std::iter::once(inner.clone()).collect());
                }
                boolean::difference(s, &cavity)
            };
            // Walls moved through each other leave no cavity: the shell is as thick as the part.
            sharp().or_else(|e| match e {
                Error::InvalidInput(_) => Err(e),
                _ => hollowed(s, open, thickness, feature).map_err(|_| e),
            })
        }
        Wall::Outward => {
            let mut outer = moved(thickness, 0.0, convex_edges(s)?)?;
            for f in outer.face_ids().collect::<Vec<_>>() {
                if !open.contains(&f) {
                    outer.set_face_names(f, Default::default());
                }
            }
            let mut cavity = s.clone();
            for f in cavity.face_ids().collect::<Vec<_>>() {
                let mut names = cavity.face(f).names.clone();
                names.insert(inner.clone());
                cavity.set_face_names(f, names);
            }
            boolean::difference(&outer, &cavity)
        }
    }
}

/// The two faces of edge `e`, where two use it.
fn edge_faces(s: &Brep, e: EdgeId) -> Option<(FaceId, FaceId)> {
    let uses: Vec<FaceId> = s
        .coedge_ids()
        .filter(|&c| s.coedge(c).edge == e)
        .map(|c| s.coedge_face(c))
        .collect();
    match uses[..] {
        [f, g] => Some((f, g)),
        _ => None,
    }
}

/// An inward shell of `s` built as `s` less its points farther than `t` from its faces other than
/// `open` ([`rounded::near_inside`]), where the moved faces cannot be followed (walls that close
/// beside a concave edge, or a bore's moved wall running into another). That is the sharp shell
/// only where it rounds every concave edge, which it may only where a face beside the edge closes,
/// and where each open face meets the walls at a right angle or less, so the walls' slabs reach it
/// as the moved walls would; otherwise an error.
fn hollowed(s: &Brep, open: &[FaceId], t: f64, feature: &str) -> Result<Brep> {
    let refuse = || {
        Err(Error::Unsupported(
            "a shell whose moved walls cannot be followed, and whose wall is not the points near them"
                .into(),
        ))
    };
    let by: HashMap<FaceId, f64> = s
        .face_ids()
        .filter(|f| !open.contains(f))
        .map(|f| (f, -t))
        .collect();
    let closed = tweak::offset_faces_closing(s, &by)
        .map(|(_, closed)| closed)
        .unwrap_or_default();
    for &e in &concave_edges(s)? {
        let Some((f, g)) = edge_faces(s, e) else {
            return refuse();
        };
        if open.contains(&f) || open.contains(&g) || !(closed.contains(&f) || closed.contains(&g)) {
            return refuse();
        }
    }
    if rounded::square_openings(s, open).is_err() {
        return refuse();
    }
    let mut cavity = rounded::near_inside(s, open, t, feature)?;
    let inner = FaceName::new(feature, FaceName::INNER);
    for f in cavity.face_ids().collect::<Vec<_>>() {
        cavity.set_face_names(f, std::iter::once(inner.clone()).collect());
    }
    boolean::difference(s, &cavity)
}

/// The points within `distance` of `s`: every face moved out that far, a hole that closes on the
/// way followed there, each convex edge rounded at `distance` (which leaves a ball's patch at each
/// convex corner), and over each end of a hole that closed, the dimple its rim leaves. Where that
/// cannot be built (a face beside a convex edge closes on the way, as the walls of a narrow slot,
/// or a cone's apex is convex), the union of `s` with a slab for each face, a tube about each
/// convex edge and a ball about each corner; an error where a piece of that is not built.
///
/// A negative `distance` shrinks `s`: the points of `s` at least `-distance` from its outside,
/// each face moved in and each concave edge rounded, or where a face beside a concave edge closes
/// on the way, `s` less a slab for each face, a tube about each concave edge and a ball about each
/// concave corner. A solid the shrink splits keeps every piece.
pub fn offset(s: &Brep, distance: f64, feature: &str) -> Result<Brep> {
    if distance == 0.0 {
        return Ok(s.clone());
    }
    if distance < 0.0 {
        return rounded::shrunk(s, -distance, feature);
    }
    moved_and_rounded(s, distance, feature).or_else(|e| grown(s, distance, feature).map_err(|_| e))
}

/// Whether face `f` is a cone whose apex is a convex point: its outward normal leans away from
/// the axis, so the ball of a distance about the apex would cap the moved cone.
fn convex_apex(s: &Brep, f: FaceId) -> bool {
    let face = s.face(f);
    let Surface::Cone(cone) = &face.surface else {
        return false;
    };
    let dom = boolean::face_domain(s, f);
    dom.interior_point().is_some_and(|uv| {
        let p = face.surface.eval(uv.x, uv.y);
        let n = face.surface.normal(uv.x, uv.y);
        let off = p - cone.frame.origin;
        let out = off - cone.frame.z * off.dot(cone.frame.z);
        (n.dot(out) > 0.0) != face.reversed
    })
}

/// `s` grown by `distance` as the union of `s` with the points near each face, each convex edge
/// and each corner (Rossignac and Requicha 1986).
fn grown(s: &Brep, distance: f64, feature: &str) -> Result<Brep> {
    let edges = convex_edges(s)?;
    let mut parts = vec![s.clone()];
    for f in s.face_ids() {
        parts.push(near::slab(s, f, distance, feature)?);
    }
    for &e in &edges {
        parts.push(near::tube(s, e, distance, feature)?);
    }
    for v in near::corners(s, &edges, |f| convex_apex(s, f)) {
        parts.push(near::ball(s.vertex(v).point, distance, feature)?);
    }
    let refs: Vec<&Brep> = parts.iter().collect();
    boolean::union_all(&refs)
}

/// [`offset`] by moving the faces and rounding the convex edges.
fn moved_and_rounded(s: &Brep, distance: f64, feature: &str) -> Result<Brep> {
    let apex = s
        .coedge_ids()
        .any(|c| s.edge(s.coedge(c).edge).curve.is_none() && convex_apex(s, s.coedge_face(c)));
    if apex {
        return Err(Error::Unsupported(
            "an offset of a cone's apex: the ball about it is not built".into(),
        ));
    }
    let by: HashMap<FaceId, f64> = s.face_ids().map(|f| (f, distance)).collect();
    let (moved, closed) = tweak::offset_faces_closing(s, &by)?;
    // A hole's rim is rounded by the dimple below; any other convex edge beside a face that
    // closed would leave its round missing.
    let closed: Vec<FaceId> = closed
        .into_iter()
        .filter(|&f| hole_radius(s, f, distance).is_none())
        .collect();
    closed_beside(
        s,
        &convex_edges(s)?,
        &closed,
        "an offset whose faces close past a convex edge",
    )?;
    let edges = convex_edges(&moved)?;
    let mut out = if edges.is_empty() {
        moved
    } else {
        blend::fillet_edges_mitred(&moved, &edges, distance, feature)?
    };
    for f in s.face_ids() {
        let Some(radius) = hole_radius(s, f, distance) else {
            continue;
        };
        for &lp in &s.face(f).loops {
            for &co in &s.lp(lp).coedges {
                let e = s.coedge(co).edge;
                let Some(Curve::Circle(rim)) = &s.edge(e).curve else {
                    continue;
                };
                let Some(g) = s
                    .coedge_ids()
                    .filter(|&o| s.coedge(o).edge == e)
                    .map(|o| s.coedge_face(o))
                    .find(|&g| g != f)
                else {
                    continue;
                };
                let (_, up) = plane_of(s, g)?;
                out = boolean::difference(
                    &out,
                    &dimple(rim.frame.origin, up, radius, distance, feature)?,
                )?;
            }
        }
    }
    Ok(out)
}

/// The radius of face `f` where it is a hole's cylindrical wall that an offset by `distance`
/// closes: no wider than `distance`, or wider by under the tolerance, which closes onto its axis
/// just as the move ends, its dimples coming down to a cusp there.
fn hole_radius(s: &Brep, f: FaceId, distance: f64) -> Option<f64> {
    let face = s.face(f);
    match &face.surface {
        Surface::Cylinder(c) if face.reversed && c.radius < distance + tol::LINEAR => {
            Some(c.radius.min(distance))
        }
        _ => None,
    }
}

/// An error naming `what` when a face of `closed` lies beside one of `edges`: the move followed
/// past it took the edge away, with the round about it the distance from the edge would leave.
fn closed_beside(s: &Brep, edges: &[EdgeId], closed: &[FaceId], what: &str) -> Result<()> {
    if closed.is_empty() {
        return Ok(());
    }
    let beside = s
        .coedge_ids()
        .filter(|&co| edges.contains(&s.coedge(co).edge))
        .any(|co| closed.contains(&s.coedge_face(co)));
    if beside {
        return Err(Error::Unsupported(format!(
            "{what}: a face beside it closes, and the round about the edge is not built"
        )));
    }
    Ok(())
}

/// What the offset by `d` leaves open over a hole's rim of radius `r` about `o`, opening along
/// `up`: the hole's cylinder beyond the rim, farther than `d` from it. In the meridian (ρ out from
/// the axis, z up from the rim's plane), the region above the arc of radius `d` about the rim.
fn dimple(o: Point3, up: Vec3, r: f64, d: f64, feature: &str) -> Result<Brep> {
    let q = (d * d - r * r).sqrt();
    let from = q.atan2(-r);
    let line = |a: [f64; 2], b: [f64; 2]| Segment::Line {
        a: Vec2::new(a[0], a[1]),
        b: Vec2::new(b[0], b[1]),
    };
    let meridian = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                named(
                    Segment::Arc {
                        center: Vec2::new(r, 0.0),
                        radius: d,
                        start: from,
                        sweep: PI / 2.0 - from,
                    },
                    "ring",
                ),
                named(line([r, d], [r, d + 1.0]), "wall"),
                named(line([r, d + 1.0], [0.0, d + 1.0]), "top"),
                named(line([0.0, d + 1.0], [0.0, q]), "axis"),
            ],
        }],
    };
    let f = Frame::from_axis(o, up)?;
    revolve(&meridian, &Frame::new(o, -f.y, f.x)?, o, f.z, TAU, feature)
}

/// The solid `faces` sweep moving from 0 to `thickness` along their outward normals, inward where
/// it is negative: `s` with them moved, less `s`, or `s` less `s` with them moved in.
pub fn thicken(s: &Brep, faces: &[FaceId], thickness: f64) -> Result<Brep> {
    let by: HashMap<FaceId, f64> = faces.iter().map(|&f| (f, thickness)).collect();
    let moved = tweak::offset_faces(s, &by)?;
    if thickness > 0.0 {
        boolean::difference(&moved, s)
    } else {
        boolean::difference(s, &moved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{boolean::half_space, measure, sweep::extrude};

    /// A prism on the polygon `pts`, `h` tall, its sides named `e0`, `e1`, … from the first edge.
    fn prism(pts: &[[f64; 2]], h: f64) -> Brep {
        let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
        let names: Vec<String> = (0..pts.len()).map(|i| format!("e{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        extrude(&Profile::polygon(&pts, &names), &Frame::WORLD, h, "p").unwrap()
    }

    fn face(b: &Brep, name: &str) -> FaceId {
        let n = FaceName::new("p", name);
        b.face_ids()
            .find(|&f| b.face(f).names.contains(&n))
            .unwrap()
    }

    fn close(got: f64, want: f64) {
        assert!(
            (got - want).abs() < 1e-7 * want.max(1.0),
            "{got} against {want}"
        );
    }

    #[test]
    fn a_block_offset_is_the_block_rounded_by_the_distance() {
        // 4 × 3 × 2 offset by 0.5: V + S·d + π(a + b + c)·d² + (4/3)π·d³, Steiner's formula.
        let b = prism(&[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], 2.0);
        let d = 0.5;
        let want = 24.0 + 52.0 * d + PI * 9.0 * d * d + 4.0 / 3.0 * PI * d.powi(3);
        close(measure::volume(&offset(&b, d, "o").unwrap()), want);
    }

    #[test]
    fn an_l_s_rounded_edges_swing_round_its_concave_edge() {
        // An L of arms 8 × 3, 5 tall, every convex edge rounded 1: the union of the balls of
        // radius 1 inside it. The two arm boxes rounded, less their overlap (the corner column
        // rounded, with (1 − π/4)·r² short of a square along the concave edge, and 4r³/3 of that
        // at its ends), and at each end of the concave edge what the ball keeps swinging round
        // it, (1 − π/4)·r³·(5/3 − π/2). A box rounded r all over: the inner box, its faces moved
        // out r, its edges' quarter cylinders and its corners' eighth balls.
        // Its area likewise, the swing's torus, (π/2)·r²(π/2 − 1), standing for two cylinder
        // pieces of r²(π/2 − 1), and the flat face keeping the corner's r × r square less its
        // quarter disk. The arms' widths, depths, heights and radii include draws of the
        // conformance suite's L-prisms the swing once failed on.
        let cases = [
            (8.0, 3.0, 8.0, 5.0, 1.0),
            (
                6.199379767061393,
                3.2446824245098926,
                5.944930609453334,
                2.787400445599377,
                0.983805266315506,
            ),
            (
                8.410642646708611,
                2.4606909192521025,
                9.679346455567444,
                7.230068216963447,
                0.2821417462255193,
            ),
            (
                9.564764789551234,
                3.3401708771362455,
                6.900089620242228,
                3.7297726032339753,
                0.7396202116839262,
            ),
            (
                7.342452085970791,
                2.923584026607261,
                6.1264770305629295,
                2.2167560977768446,
                0.8678845173029267,
            ),
            (
                5.52965122853034,
                2.429579012791682,
                9.404723647837528,
                5.938985170662157,
                0.6834351663091055,
            ),
            (
                6.3990454944784565,
                2.3841696916601958,
                8.181436493454777,
                4.839043994693185,
                0.7102451360532922,
            ),
            (
                6.538663752237994,
                3.1564303512067906,
                9.224051200212097,
                5.94676849583945,
                1.2821630460001838,
            ),
        ];
        for (w, t, d, h, r) in cases {
            let b = prism(&[[0.0, 0.0], [w, 0.0], [w, t], [t, t], [t, d], [0.0, d]], h);
            let out = blend::fillet_edges(&b, &convex_edges(&b).unwrap(), r, "fil").unwrap();
            assert!(
                crate::check::check(&out).is_empty(),
                "{:?}",
                crate::check::check(&out)
            );
            let rounded = |x: f64, y: f64, z: f64| {
                let (a, b, c) = (x - 2.0 * r, y - 2.0 * r, z - 2.0 * r);
                (
                    a * b * c
                        + 2.0 * r * (a * b + b * c + c * a)
                        + PI * r * r * (a + b + c)
                        + 4.0 / 3.0 * PI * r.powi(3),
                    2.0 * (a * b + b * c + c * a) + 2.0 * PI * r * (a + b + c) + 4.0 * PI * r * r,
                )
            };
            let k = 1.0 - PI / 4.0;
            let ((va, sa), (vb, sb), (vc, sc)) =
                (rounded(w, t, h), rounded(t, d, h), rounded(t, t, h));
            let core_h = h - 2.0 * r;
            let volume = va + vb - vc - k * (core_h * r * r + 4.0 / 3.0 * r.powi(3))
                + 2.0 * k * (5.0 / 3.0 - PI / 2.0) * r.powi(3);
            let area = sa + sb
                - sc
                - k * (2.0 * core_h * r + 4.0 * r * r)
                - 2.0 * ((2.0 - PI / 2.0) * (PI / 2.0 - 1.0) - k) * r * r;
            close(measure::volume(&out), volume);
            close(measure::area(&out), area);
            // Written and read back, the swing's torus, whose pole lies on its boundary, is
            // the same solid.
            let text = crate::step::write(&out, "l").unwrap();
            let back = crate::step::read(text.as_bytes(), "l").unwrap();
            assert!(back.rejected.is_empty(), "{:?}", back.rejected);
            close(measure::volume(&back.brep), volume);
        }
    }

    #[test]
    fn an_l_prism_offset_turned_away_is_the_same_size() {
        let (w, t, h) = (9.536367478693311, 2.7154798898513985, 6.344052519590756);
        let b = prism(
            &[[0.0, 0.0], [w, 0.0], [w, t], [t, t], [t, h], [0.0, h]],
            5.222922150702244,
        );
        let d = 0.5756566176826041;
        let here = measure::volume(&offset(&b, d, "o").unwrap());
        let mut moved = b.clone();
        moved.transform(&crate::math::Transform {
            r: [
                [
                    -0.2124025946267396,
                    -0.9641070956655728,
                    -0.15931932049542274,
                ],
                [
                    0.9494953687642673,
                    -0.1650876859267214,
                    -0.26684190197674823,
                ],
                [0.2309625131726732, -0.20795086930006934, 0.9504802751589895],
            ],
            t: Vec3::new(-45.51695804202156, 41.045298753592036, -21.51869167292315),
        });
        close(measure::volume(&offset(&moved, d, "o").unwrap()), here);
    }

    #[test]
    fn a_shell_wall_closing_a_neck_rounds_the_mouths_about_its_corners() {
        // A neck 2 wide between two squares, walled 1.5 thick from inside: the walls cross in the
        // neck, which stays solid, and each pocket's mouth to it is bounded by arcs of radius 1.5
        // about the neck's corners. A pocket is the square 7 × 7 and the bulge between x = 8.5 and
        // the arcs about (10, 4) and (10, 6), which meet at (10 − √5/2, 5): β = 3 − √5/2 −
        // (9/4)·asin(2/3). Both pockets run 4.5 deep in the plate of 208 × 6.
        let dumbbell = prism(
            &[
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 4.0],
                [14.0, 4.0],
                [14.0, 0.0],
                [24.0, 0.0],
                [24.0, 10.0],
                [14.0, 10.0],
                [14.0, 6.0],
                [10.0, 6.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ],
            6.0,
        );
        let top = FaceName::new("p", "end");
        let top: Vec<FaceId> = dumbbell
            .face_ids()
            .filter(|&f| dumbbell.face(f).names.contains(&top))
            .collect();
        assert_eq!(top.len(), 1);
        let shelled = shell(&dumbbell, &top, 1.5, Wall::Inward, "w").unwrap();
        let beta = 3.0 - 5f64.sqrt() / 2.0 - 2.25 * (2.0f64 / 3.0).asin();
        close(
            measure::volume(&shelled),
            208.0 * 6.0 - 4.5 * 2.0 * (49.0 + beta),
        );
        // A wall 0.5 thick leaves the neck open: the cavity is two squares 9 on a side and a
        // channel 5 × 1 between them, 5.5 deep, in the plate of 208 × 6.
        let shelled = shell(&dumbbell, &top, 0.5, Wall::Inward, "w").unwrap();
        close(
            measure::volume(&shelled),
            208.0 * 6.0 - 5.5 * (2.0 * 81.0 + 5.0),
        );
    }

    #[test]
    fn a_bore_s_moved_wall_running_into_a_shell_s_wall_merges_with_it() {
        // The block 20 × 20 × 10 bored through by a hole of radius 3 about (15.5, 10), walled 1
        // thick from inside, open at its top. The bore's wall, moved to radius 4, reaches past the
        // inside wall x = 19 by the segment of the disk of radius 4 at 3.5 from its centre,
        // 16·acos(7/8) − 7√15/4; the cavity is the square 18 × 18 less the rest of that disk, 9
        // deep.
        let bore = Profile::named_circle(Vec2::new(15.5, 10.0), 6.0, "bore").loops;
        let section = Profile::rect(Vec2::new(10.0, 10.0), 20.0, 20.0).with_holes(bore);
        let block = extrude(&section, &Frame::WORLD, 10.0, "p").unwrap();
        let top = face(&block, "end");
        let shelled = shell(&block, &[top], 1.0, Wall::Inward, "w").unwrap();
        let segment = 16.0 * (7.0f64 / 8.0).acos() - 7.0 * 15f64.sqrt() / 4.0;
        close(
            measure::volume(&shelled),
            4000.0 - 90.0 * PI - 9.0 * (324.0 - 16.0 * PI + segment),
        );
    }

    #[test]
    fn a_notch_s_faces_taken_out_leave_the_block_whole() {
        let b = prism(
            &[
                [0.0, 0.0],
                [6.0, 0.0],
                [6.0, 4.0],
                [4.0, 4.0],
                [4.0, 3.0],
                [2.0, 3.0],
                [2.0, 4.0],
                [0.0, 4.0],
            ],
            1.0,
        );
        let notch = [face(&b, "e3"), face(&b, "e4"), face(&b, "e5")];
        close(measure::volume(&delete_faces(&b, &notch).unwrap()), 24.0);
    }

    fn feature_faces(b: &Brep, feature: &str) -> Vec<FaceId> {
        b.face_ids()
            .filter(|&f| b.face(f).names.iter().any(|n| n.feature == feature))
            .collect()
    }

    /// `b` with the faces of `feature` deleted: its volume and area.
    fn deleted(b: &Brep, feature: &str) -> (f64, f64) {
        let out = delete_faces(b, &feature_faces(b, feature)).unwrap();
        assert!(
            crate::check::check(&out).is_empty(),
            "{:?}",
            crate::check::check(&out)
        );
        (measure::volume(&out), measure::area(&out))
    }

    #[test]
    fn a_boss_and_a_concave_round_deleted_take_material_away() {
        use crate::primitive::cylinder;
        let at = |z: f64| Frame::new(Point3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap();
        // A plate of radius 8 and 4 thick with a boss of radius 3 and 8 tall standing on it.
        let plate = cylinder(&at(-4.0), 8.0, 4.0, "plate").unwrap();
        let boss = cylinder(&at(0.0), 3.0, 8.0, "boss").unwrap();
        let solid = boolean::union(&plate, &boss).unwrap();
        let (v, a) = deleted(&solid, "boss");
        close(v, 256.0 * PI);
        close(a, 192.0 * PI);
        // An L of outline area 150 and perimeter 70, 8 tall, its inside corner rounded 2.
        let pts = [
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 5.0],
            [5.0, 5.0],
            [5.0, 15.0],
            [0.0, 15.0],
        ];
        let l = prism(&pts, 8.0);
        let rounded = blend::fillet(&l, &[("p.e2", "p.e3")], 2.0, "round").unwrap();
        let (v, a) = deleted(&rounded, "round");
        close(v, 150.0 * 8.0);
        close(a, 2.0 * 150.0 + 70.0 * 8.0);
    }

    #[test]
    fn deleted_flats_close_where_the_surfaces_round_them_bulge_past_their_edges() {
        use crate::primitive::{cone, cylinder, sphere};
        // A shaft of radius 5 and 20 long with |x| > 4 milled away: whole again, 500π.
        let shaft = cylinder(&Frame::WORLD, 5.0, 20.0, "shaft").unwrap();
        let right = half_space(&shaft, Point3::new(4.0, 0.0, 0.0), Vec3::X).unwrap();
        let left = half_space(&shaft, Point3::new(-4.0, 0.0, 0.0), -Vec3::X).unwrap();
        let milled = boolean::difference_all(&shaft, &[&right, &left]).unwrap();
        let (v, a) = deleted(&milled, "half");
        close(v, 500.0 * PI);
        close(a, 250.0 * PI);
        // A ball of radius 5 cut flat at z = 3: its flat's edge lies in its own plane.
        let ball = sphere(&Frame::WORLD, 5.0, "ball").unwrap();
        let above = half_space(&ball, Point3::new(0.0, 0.0, 3.0), Vec3::Z).unwrap();
        let (v, a) = deleted(&boolean::difference(&ball, &above).unwrap(), "half");
        close(v, 500.0 * PI / 3.0);
        close(a, 100.0 * PI);
        // A frustum of radii 6 and 3, 6 tall, its top rim rounded 1: (π·6/3)(6² + 6·3 + 3²).
        let frustum = cone(&Frame::WORLD, 6.0, 3.0, 6.0, "f").unwrap();
        let rounded = blend::fillet(&frustum, &[("f.top", "f.side")], 1.0, "round").unwrap();
        let (v, a) = deleted(&rounded, "round");
        close(v, 126.0 * PI);
        close(a, 45.0 * PI + 9.0 * PI * 45f64.sqrt());
        // A cone of radius 6 up to its apex 8 above, its foot rounded 1: the cone again, its first
        // vertex the apex, where the side's normal says nothing of which side its material is.
        // Volume π·6²·8/3; area the base 36π and the side π·6·10.
        let point = cone(&Frame::WORLD, 6.0, 0.0, 8.0, "c").unwrap();
        let rounded = blend::fillet(&point, &[("c.bottom", "c.side")], 1.0, "round").unwrap();
        let (v, a) = deleted(&rounded, "round");
        close(v, 96.0 * PI);
        close(a, 96.0 * PI);
    }

    #[test]
    fn a_rounded_box_s_top_rim_deleted_runs_its_corner_cylinders_up_only() {
        // The box 20 × 12 × 8, its upright edges rounded 3, then its top rim rounded 1. Deleted,
        // the rim leaves the box with its outline a rectangle with quarter circles of radius 3 at
        // its corners: area 20·12 − (4 − π)·3², perimeter 2(14 + 6) + 2π·3.
        let body = crate::primitive::block(&Frame::WORLD, Vec3::new(20.0, 12.0, 8.0), "b");
        let uprights = [
            ("b.u_min", "b.v_min"),
            ("b.v_min", "b.u_max"),
            ("b.u_max", "b.v_max"),
            ("b.v_max", "b.u_min"),
        ];
        let walls = blend::fillet(&body.unwrap(), &uprights, 3.0, "corner").unwrap();
        let rounded = blend::fillet(&walls, &[("b.end", "b.v_min")], 1.0, "rim").unwrap();
        let outline = 240.0 - (4.0 - PI) * 9.0;
        let (v, a) = deleted(&rounded, "rim");
        close(v, 8.0 * outline);
        close(a, 2.0 * outline + 8.0 * (40.0 + 6.0 * PI));
    }

    #[test]
    fn a_face_thickened_is_a_slab_on_it() {
        // The 4 × 2 side of a 4 × 3 × 2 block thickened 0.5 outward, then 1 inward.
        let b = prism(&[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], 2.0);
        let side = face(&b, "e0");
        close(measure::volume(&thicken(&b, &[side], 0.5).unwrap()), 4.0);
        close(measure::volume(&thicken(&b, &[side], -1.0).unwrap()), 8.0);
    }

    #[test]
    fn a_dome_thickened_in_by_its_radius_is_the_half_ball() {
        // The half ball of radius 5: its dome moved 5 in closes on the centre, and the solid
        // between them is the half ball itself, 250π/3.
        let meridian = Profile::outline(vec![
            NamedSegment::line(Vec2::new(0.0, 0.0), Vec2::new(5.0, 0.0), "base"),
            NamedSegment::arc(Vec2::ZERO, 5.0, 0.0, PI / 2.0, "sheet"),
            NamedSegment::line(Vec2::new(0.0, 5.0), Vec2::new(0.0, 0.0), "axis"),
        ]);
        let plane = Frame::new(Point3::ZERO, -Vec3::Y, Vec3::X).unwrap();
        let h = revolve(&meridian, &plane, Point3::ZERO, Vec3::Z, TAU, "p").unwrap();
        let out = thicken(&h, &[face(&h, "sheet")], -5.0).unwrap();
        close(measure::volume(&out), 250.0 * PI / 3.0);
    }

    #[test]
    fn a_cone_grown_caps_its_apex_with_a_ball() {
        // The cone of base radius R = 3 and height H = 4 grown by d = 1, by Steiner's formula:
        // its integrated mean curvature is the side's πH and the base rim's torus turning through
        // π − β, β = atan(H/R) the angle inside the cone at the rim; the apex is a ball's cap.
        let c = crate::primitive::cone(&Frame::WORLD, 3.0, 0.0, 4.0, "c").unwrap();
        let out = offset(&c, 1.0, "o").unwrap();
        let (r, h, l, d) = (3.0f64, 4.0f64, 5.0, 1.0f64);
        let m = PI * (h + r * (PI - (h / r).atan()));
        let s = PI * (r * r + r * l);
        let v = PI * r * r * h / 3.0;
        close(
            measure::volume(&out),
            v + s * d + m * d * d + 4.0 / 3.0 * PI * d.powi(3),
        );
        close(measure::area(&out), s + 2.0 * m * d + 4.0 * PI * d * d);
    }

    /// The solid turned a whole turn about z from the polygon `pts` in (ρ, z).
    fn turned_polygon(pts: &[[f64; 2]]) -> Brep {
        let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
        let names: Vec<String> = (0..pts.len()).map(|i| format!("m{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let plane = Frame::new(Point3::ZERO, -Vec3::Y, Vec3::X).unwrap();
        let profile = Profile::polygon(&pts, &names);
        revolve(&profile, &plane, Point3::ZERO, Vec3::Z, TAU, "p").unwrap()
    }

    #[test]
    fn a_groove_grown_shut_leaves_the_notch_its_rims_round() {
        // A drum of radius 10 and height 10 with a groove 1.6 wide and 3 deep about z = 5, grown
        // by 1: the groove's walls cross, and its rims' tori meet in a crease at ρ = 10.6. By
        // Pappus, 2π times the section's moment about the axis: the drum with no groove grown
        // (the rectangle 10 × 12, the strip 1 × 10 beside it and the rims' quarter disks), less the notch
        // between the rims' disks over z = 5 ± a, a = 0.8, of moment 20a − 10(a√(1 − a²) +
        // asin a) + a³/3 a side; and 2π times the moment of the outline for the area.
        let drum = turned_polygon(&[
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 4.2],
            [7.0, 4.2],
            [7.0, 5.8],
            [10.0, 5.8],
            [10.0, 10.0],
            [0.0, 10.0],
        ]);
        let out = offset(&drum, 1.0, "o").unwrap();
        let a: f64 = 0.8;
        let whole = 705.0 + 5.0 * PI + 2.0 / 3.0;
        let notch = 20.0 * a - 10.0 * (a * (1.0 - a * a).sqrt() + a.asin()) + a.powi(3) / 3.0;
        close(measure::volume(&out), TAU * (whole - notch));
        let outline = 100.0 + 2.0 * (5.0 * PI + 1.0) + 11.0 * (10.0 - 2.0 * a);
        close(
            measure::area(&out),
            TAU * (outline + 2.0 * (10.0 * a.asin() + a)),
        );
    }

    #[test]
    fn a_torus_grown_past_its_hole_is_an_apple() {
        // The tube of radius 2 about the circle of radius R = 4 grown by 3: every point within
        // s = 5 of the circle. Its section is the disk of radius s about (R, 0) less the part past
        // the axis, which it crosses at heights ±c, c = √(s² − R²), at the half angle
        // φ = acos(R/s) from the disk's centre.
        let t = crate::primitive::torus(&Frame::WORLD, 4.0, 2.0, "t").unwrap();
        let out = offset(&t, 3.0, "o").unwrap();
        let (r, s) = (4.0f64, 5.0f64);
        let c = (s * s - r * r).sqrt();
        let phi = (r / s).acos();
        close(
            measure::volume(&out),
            TAU * (PI * r * s * s + 2.0 * c.powi(3) / 3.0 - r * (s * s * phi - r * c)),
        );
        close(
            measure::area(&out),
            TAU * (2.0 * r * s * (PI - phi) + 2.0 * s * c),
        );
    }

    #[test]
    fn a_cylinder_cut_aslant_grown_past_its_ellipse_s_bend_holds_steiner_s_formula() {
        // The cylinder of radius 3 from z = −10 up to the plane z = x·tan 45°, grown by 4: more
        // than the cut ellipse's least radius of curvature, 9/(3√2). The body is convex, so by
        // Steiner's formula its volume is V + S·d + M·d² + 4πd³/3: V = 90π, S = 69π + 9π√2, and
        // M the side's 10π, the bottom rim's 3π²/2 and half of J, the ellipse's turning angle
        // between its faces' normals, acos(−sin α cos φ), integrated along its length element
        // 3√(1 + tan²α sin²φ) dφ.
        let a = PI / 4.0;
        let c =
            crate::primitive::cylinder(&Frame::WORLD.translated(-Vec3::Z * 10.0), 3.0, 20.0, "c")
                .unwrap();
        let above = half_space(&c, Point3::ZERO, Vec3::new(-a.sin(), 0.0, a.cos())).unwrap();
        let cut = boolean::difference(&c, &above).unwrap();
        let d = 4.0;
        let out = offset(&cut, d, "o").unwrap();
        let mut turn =
            |p: f64| (-a.sin() * p.cos()).acos() * 3.0 * (1.0 + (a.tan() * p.sin()).powi(2)).sqrt();
        let j = crate::numeric::integrate(&mut turn, 0.0, TAU, 1e-13);
        let m = 10.0 * PI + 1.5 * PI * PI + 0.5 * j;
        let s = 69.0 * PI + 9.0 * PI / a.cos();
        close(
            measure::volume(&out),
            90.0 * PI + s * d + m * d * d + 4.0 / 3.0 * PI * d.powi(3),
        );
        close(measure::area(&out), s + 2.0 * m * d + 4.0 * PI * d * d);
    }

    #[test]
    fn an_elliptic_cylinder_of_two_halves_grown_rounds_both_rims_whole() {
        // The ellipse of semi-axes 5 and 3 drawn as two half ellipses, extruded 10 and made one
        // face round, grown by 1: each rim's chain is the two halves, its contact on the side
        // starting where the side's loop meets it. By Steiner's formula, with p the ellipse's
        // perimeter: V + S·d + M·d² + 4πd³/3, V = πabh, S = 2πab + ph, M = πh + πp/2.
        let half = |start: f64| {
            let arc = Segment::Ellipse {
                center: Vec2::ZERO,
                rotation: 0.0,
                major: 5.0,
                minor: 3.0,
                start,
                sweep: PI,
            };
            NamedSegment::new(arc, "side")
        };
        let base = Profile::outline(vec![half(0.0), half(PI)]);
        let halves = extrude(&base, &Frame::WORLD, 10.0, "p").unwrap();
        let body = boolean::unify(&halves).unwrap();
        let out = offset(&body, 1.0, "o").unwrap();
        let (a, b, h, d) = (5.0f64, 3.0f64, 10.0, 1.0);
        let mut speed = |t: f64| (a * a * t.sin().powi(2) + b * b * t.cos().powi(2)).sqrt();
        let p = crate::numeric::integrate(&mut speed, 0.0, TAU, 1e-13);
        let (s, m) = (2.0 * PI * a * b + p * h, PI * h + PI * p / 2.0);
        close(
            measure::volume(&out),
            PI * a * b * h + s * d + m * d * d + 4.0 / 3.0 * PI * d.powi(3),
        );
        close(measure::area(&out), s + 2.0 * m * d + 4.0 * PI * d * d);
    }

    #[test]
    fn an_elliptic_cylinder_shrunk_past_its_curvature_keeps_its_parallel_trimmed() {
        // The ellipse of semi-axes 5 and 3 extruded 10, shrunk by 2.5 as the cylinder less its
        // complement grown by 2.5. The ellipse's least radius of curvature is 9/5, so its parallel
        // 2.5 in loops past both ends of the long axis; the section left is bounded by the
        // parallel's arcs between where they cross that axis, and its area is ½∮(x dy − y dx)
        // along them, with o′ = e′·(1 − d·κ) and κ = ab/|e′|³.
        let (a, b, d) = (5.0f64, 3.0f64, 2.5f64);
        let half = |start: f64| {
            let arc = Segment::Ellipse {
                center: Vec2::ZERO,
                rotation: 0.0,
                major: a,
                minor: b,
                start,
                sweep: PI,
            };
            NamedSegment::new(arc, "side")
        };
        let base = Profile::outline(vec![half(0.0), half(PI)]);
        let body = boolean::unify(&extrude(&base, &Frame::WORLD, 10.0, "p").unwrap()).unwrap();
        let room = crate::primitive::block(
            &Frame::WORLD.translated(Vec3::new(-10.0, -10.0, -5.0)),
            Vec3::new(20.0, 20.0, 20.0),
            "room",
        )
        .unwrap();
        let outside = boolean::difference(&room, &body).unwrap();
        let near = offset(&outside, d, "o").unwrap();
        let out = boolean::difference(&body, &near).unwrap();
        let o = |t: f64| {
            let n = Vec2::new(b * t.cos(), a * t.sin());
            let e1 = Vec2::new(-a * t.sin(), b * t.cos());
            let k = a * b / e1.norm().powi(3);
            let p = Vec2::new(a * t.cos(), b * t.sin()) - n * (d / n.norm());
            (p, e1 * (1.0 - d * k))
        };
        let (mut lo, mut hi) = (1e-9, 0.5 * PI);
        for _ in 0..200 {
            let m = 0.5 * (lo + hi);
            if o(m).0.y < 0.0 {
                lo = m;
            } else {
                hi = m;
            }
        }
        let mut green = |t: f64| {
            let (p, dp) = o(t);
            p.cross(dp)
        };
        let area = crate::numeric::integrate(&mut green, lo, PI - lo, 1e-13);
        close(measure::volume(&out), area * 5.0);
    }

    #[test]
    fn a_plate_grown_by_its_hole_s_radius_fills_the_hole_and_dimples_to_a_cusp() {
        // A plate a × b × c with a hole of radius ρ = d through it, grown by d: the box grown by
        // Steiner's formula, V₀ = abc + 2(ab + bc + ca)d + π(a + b + c)d² + 4πd³/3, with the hole
        // filled, less over each end of it the dimple inside the horn torus of radius d about the
        // rim: in the meridian, under z = d and above the quarter circle of radius d about (d, 0),
        // of volume 2π∫₀^d ρ(d − √(d² − (d − ρ)²)) dρ = 2πd³(5/6 − π/4). Its area: the box's grown,
        // A₀ = 2(ab + bc + ca) + 2π(a + b + c)d + 4πd², less the disk πd² over each end and plus
        // the torus inside the rim, 2π∫ρ ds = 2πd²(π/2 − 1).
        for (a, b, c, d) in [(20.0, 20.0, 4.0, 3.0), (30.0, 24.0, 2.0, 10.0)] {
            let plate =
                crate::primitive::block(&Frame::WORLD, Vec3::new(a, b, c), "plate").unwrap();
            let at = Frame::WORLD.translated(Vec3::new(a / 2.0, b / 2.0, -1.0));
            let drill = crate::primitive::cylinder(&at, d, c + 2.0, "hole").unwrap();
            let s = boolean::difference(&plate, &drill).unwrap();
            let out = offset(&s, d, "o").unwrap();
            assert!(crate::check::check(&out).is_empty());
            let v0 = a * b * c
                + 2.0 * (a * b + b * c + c * a) * d
                + PI * (a + b + c) * d * d
                + 4.0 / 3.0 * PI * d.powi(3);
            let a0 = 2.0 * (a * b + b * c + c * a) + 2.0 * PI * (a + b + c) * d + 4.0 * PI * d * d;
            let dimple = TAU * d.powi(3) * (5.0 / 6.0 - PI / 4.0);
            close(measure::volume(&out), v0 - 2.0 * dimple);
            let rim = TAU * d * d * (PI / 2.0 - 1.0) - PI * d * d;
            close(measure::area(&out), a0 + 2.0 * rim);
        }
    }
}
