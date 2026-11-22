use std::f64::consts::PI;

use crate::{
    blend,
    edit::{
        Wall,
        local::{assert_same, counted, general},
        shell, thicken,
    },
    error::Result,
    math::{Frame, Point3, Vec2, Vec3},
    measure,
    pick::faces_named,
    primitive,
    profile::Profile,
    sweep::extrude,
    topo::{Brep, FaceId},
};

fn block() -> Brep {
    primitive::block(&Frame::WORLD, Vec3::new(10.0, 8.0, 6.0), "cube").unwrap()
}

/// A prism on the polygon `pts`, `h` tall, its sides named `e0`, `e1`, … from the first edge.
fn prism(pts: &[[f64; 2]], h: f64) -> Brep {
    let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
    let names: Vec<String> = (0..pts.len()).map(|i| format!("e{i}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    extrude(&Profile::polygon(&pts, &names), &Frame::WORLD, h, "p").unwrap()
}

fn l_prism() -> Brep {
    prism(
        &[
            [0.0, 0.0],
            [9.0, 0.0],
            [9.0, 3.0],
            [3.0, 3.0],
            [3.0, 8.0],
            [0.0, 8.0],
        ],
        5.0,
    )
}

/// A 12 × 9 × 6 block with its upright edges rounded to `rho`.
fn rounded_block(rho: f64) -> Brep {
    let b = primitive::block(&Frame::WORLD, Vec3::new(12.0, 9.0, 6.0), "cube").unwrap();
    let pairs = [
        ("cube.u_min", "cube.v_min"),
        ("cube.u_min", "cube.v_max"),
        ("cube.u_max", "cube.v_min"),
        ("cube.u_max", "cube.v_max"),
    ];
    blend::fillet(&b, &pairs, rho, "round").unwrap()
}

/// A 10 × 4 × 2 plate with a bore of radius 1 through it.
fn bored_plate() -> Brep {
    let hole = Profile::named_circle(Vec2::new(3.0, 2.0), 2.0, "bore").loops;
    let section = Profile::rect(Vec2::new(5.0, 2.0), 10.0, 4.0).with_holes(hole);
    extrude(&section, &Frame::WORLD, 2.0, "p").unwrap()
}

fn faces(b: &Brep, names: &[&str]) -> Vec<FaceId> {
    let mut out = Vec::new();
    for n in names {
        let found = faces_named(b, n);
        assert!(!found.is_empty(), "no face {n}");
        out.extend(found);
    }
    out
}

fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-9 * want.abs().max(1.0),
        "{got} against {want}"
    );
}

/// What `f` builds from the two solids' own faces, which must be the solid its general way
/// builds.
fn both(f: impl Fn() -> Result<Brep>) -> Brep {
    let (fast, direct) = counted(|| f().unwrap());
    assert!(direct > 0, "built the general way");
    let (slow, direct) = counted(|| general(&f).unwrap());
    assert_eq!(direct, 0);
    assert_same(&fast, &slow);
    fast
}

/// What `f` builds, which must be its general way's, no step of it skipped.
fn general_only(f: impl Fn() -> Result<Brep>) -> Brep {
    let (out, direct) = counted(|| f().unwrap());
    assert_eq!(direct, 0, "skipped the general way");
    assert_same(&out, &general(&f).unwrap());
    out
}

#[test]
fn a_block_shelled_open_at_its_top_is_its_wall() {
    let b = block();
    let open = faces(&b, &["cube.end"]);
    let out = both(|| shell(&b, &open, 1.0, Wall::Inward, "s"));
    close(measure::volume(&out), 480.0 - 8.0 * 6.0 * 5.0);
    assert_eq!(out.face_count(), 11);
    assert_eq!(out.shell_ids().count(), 1);
    assert_eq!(faces_named(&out, "s.inner").len(), 5);
    assert_eq!(faces_named(&out, "cube.end").len(), 1);
}

#[test]
fn a_block_shelled_with_no_opening_holds_a_void() {
    let b = block();
    let out = both(|| shell(&b, &[], 1.0, Wall::Inward, "s"));
    close(measure::volume(&out), 480.0 - 8.0 * 6.0 * 4.0);
    assert_eq!(out.shell_ids().count(), 2);
}

#[test]
fn a_block_shelled_outward_keeps_its_faces_as_the_wall_s_inside() {
    let b = block();
    let open = faces(&b, &["cube.end"]);
    let out = both(|| shell(&b, &open, 1.0, Wall::Outward, "s"));
    close(measure::volume(&out), 12.0 * 10.0 * 7.0 - 480.0);
    assert_eq!(faces_named(&out, "s.inner").len(), 5);
}

#[test]
fn a_block_shelled_open_at_two_ends_is_a_tube() {
    let b = block();
    let open = faces(&b, &["cube.end", "cube.start"]);
    let out = both(|| shell(&b, &open, 1.0, Wall::Inward, "s"));
    close(measure::volume(&out), 480.0 - 8.0 * 6.0 * 6.0);
    assert_eq!(out.face_count(), 10);
}

#[test]
fn an_l_prism_shelled_keeps_its_concave_edge_sharp() {
    let b = l_prism();
    let open = faces(&b, &["p.end"]);
    let out = both(|| shell(&b, &open, 0.7, Wall::Inward, "s"));
    assert_eq!(out.face_count(), 15);
}

#[test]
fn turned_solids_shelled_keep_their_walls_exact() {
    let cyl = primitive::cylinder(&Frame::WORLD, 4.0, 6.0, "cyl").unwrap();
    let open = faces(&cyl, &["cyl.end"]);
    let out = both(|| shell(&cyl, &open, 1.0, Wall::Inward, "s"));
    close(measure::volume(&out), PI * (16.0 * 6.0 - 9.0 * 5.0));
    let cone = primitive::cone(&Frame::WORLD, 6.0, 3.0, 4.0, "frustum").unwrap();
    let open = faces(&cone, &["frustum.top"]);
    both(|| shell(&cone, &open, 1.0, Wall::Inward, "s"));
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let out = both(|| shell(&ball, &[], 1.0, Wall::Inward, "s"));
    close(measure::volume(&out), 4.0 / 3.0 * PI * (125.0 - 64.0));
}

#[test]
fn a_rounded_block_shelled_thinner_than_its_rounds_keeps_them() {
    let b = rounded_block(2.0);
    let open = faces(&b, &["cube.end"]);
    let out = both(|| shell(&b, &open, 0.8, Wall::Inward, "s"));
    assert_eq!(out.face_count(), 19);
}

#[test]
fn a_rounded_block_shelled_thicker_than_its_rounds_has_a_sharp_pocket() {
    // The rounds close on the way in: the cavity's top is paired with the block's by its plane.
    let b = rounded_block(2.0);
    let open = faces(&b, &["cube.end"]);
    let out = both(|| shell(&b, &open, 2.5, Wall::Inward, "s"));
    assert_eq!(out.face_count(), 15);
    let outside = 12.0 * 9.0 - (4.0 - PI) * 4.0;
    close(measure::volume(&out), outside * 6.0 - 7.0 * 4.0 * 3.5);
}

#[test]
fn a_bored_plate_shelled_open_at_its_top_has_a_rim_round_its_bore() {
    // The top's outer loop and its bore's each bound a strip of their own.
    let b = bored_plate();
    let open = faces(&b, &["p.end"]);
    let out = both(|| shell(&b, &open, 0.4, Wall::Inward, "s"));
    assert_eq!(faces_named(&out, "p.end").len(), 2);
}

#[test]
fn openings_that_share_an_edge_keep_its_ends_as_the_wall_s_corners() {
    // The edge between the openings is neither's: only the stretches at its ends bound a strip.
    let b = block();
    let open = faces(&b, &["cube.end", "cube.u_max"]);
    let out = both(|| shell(&b, &open, 1.0, Wall::Inward, "s"));
    close(measure::volume(&out), 480.0 - 9.0 * 6.0 * 5.0);
    assert_eq!(out.face_count(), 10);
}

#[test]
fn a_curved_opening_round_its_axis_goes_the_general_way() {
    let cyl = primitive::cylinder(&Frame::WORLD, 4.0, 6.0, "cyl").unwrap();
    let open = faces(&cyl, &["cyl.side"]);
    general_only(|| shell(&cyl, &open, 1.0, Wall::Inward, "s"));
}

#[test]
fn an_opening_beside_a_concave_edge_goes_the_general_way() {
    // The wall beside the opening moves in under itself: the cavity's face on the opening's plane
    // reaches past the opening there, where it is the end of the wall and no strip of the opening.
    let b = l_prism();
    let open = faces(&b, &["p.e2"]);
    let out = general_only(|| shell(&b, &open, 0.7, Wall::Inward, "s"));
    assert!(measure::volume(&out) < measure::volume(&b));
}

#[test]
fn a_shell_too_thick_is_refused_either_way() {
    let b = block();
    let open = faces(&b, &["cube.end"]);
    let fast = shell(&b, &open, 4.5, Wall::Inward, "s").unwrap_err();
    let slow = general(|| shell(&b, &open, 4.5, Wall::Inward, "s")).unwrap_err();
    assert_eq!(fast.to_string(), slow.to_string());
}

#[test]
fn a_block_s_face_thickened_is_the_slab_it_sweeps() {
    let b = block();
    let top = faces(&b, &["cube.end"]);
    let out = both(|| thicken(&b, &top, 1.2));
    close(measure::volume(&out), 80.0 * 1.2);
    assert_eq!(out.face_count(), 6);
    let out = both(|| thicken(&b, &top, -2.0));
    close(measure::volume(&out), 160.0);
    // The slab's ends keep the face's names, and its sides those of the faces round it.
    assert_eq!(faces_named(&out, "cube.end").len(), 2);
    assert_eq!(faces_named(&out, "cube.u_min").len(), 1);
}

#[test]
fn faces_thickened_across_the_edge_they_share_make_one_slab() {
    let b = l_prism();
    let walls = faces(&b, &["p.e2", "p.e3"]);
    let out = both(|| thicken(&b, &walls, 1.0));
    close(measure::volume(&out), 50.0);
    assert_eq!(out.face_count(), 8);
    let out = both(|| thicken(&b, &walls, -1.0));
    close(measure::volume(&out), 60.0);
}

#[test]
fn a_cylinder_s_side_thickened_is_a_tube() {
    let cyl = primitive::cylinder(&Frame::WORLD, 4.0, 6.0, "cyl").unwrap();
    let side = faces(&cyl, &["cyl.side"]);
    let out = both(|| thicken(&cyl, &side, 1.2));
    close(measure::volume(&out), PI * (5.2 * 5.2 - 16.0) * 6.0);
    let out = both(|| thicken(&cyl, &side, -2.0));
    close(measure::volume(&out), PI * (16.0 - 4.0) * 6.0);
    // Its end thickened is a disc, the side's strip closed on its seam.
    let end = faces(&cyl, &["cyl.end"]);
    let out = both(|| thicken(&cyl, &end, 1.5));
    close(measure::volume(&out), PI * 16.0 * 1.5);
}

#[test]
fn whole_closed_faces_thickened_hold_a_void() {
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let skin: Vec<FaceId> = ball.face_ids().collect();
    let out = both(|| thicken(&ball, &skin, 1.2));
    close(
        measure::volume(&out),
        4.0 / 3.0 * PI * (6.2f64.powi(3) - 125.0),
    );
    assert_eq!(out.shell_ids().count(), 2);
    let out = both(|| thicken(&ball, &skin, -2.0));
    close(measure::volume(&out), 4.0 / 3.0 * PI * (125.0 - 27.0));
    let ring = primitive::torus(&Frame::WORLD, 6.0, 2.0, "ring").unwrap();
    let skin: Vec<FaceId> = ring.face_ids().collect();
    let out = both(|| thicken(&ring, &skin, -1.0));
    close(measure::volume(&out), 2.0 * PI * PI * 6.0 * (4.0 - 1.0));
}

#[test]
fn a_frustum_s_side_thickened_inward_ends_on_its_flats() {
    let cone = primitive::cone(&Frame::WORLD, 6.0, 3.0, 4.0, "frustum").unwrap();
    let side = faces(&cone, &["frustum.side"]);
    let out = both(|| thicken(&cone, &side, -1.0));
    assert_eq!(out.face_count(), 4);
}

#[test]
fn a_plate_s_bore_thickened_is_a_sleeve() {
    let b = bored_plate();
    let bore = faces(&b, &["p.bore"]);
    // The bore's wall faces its axis: thickened outward it fills toward it, inward it widens.
    let out = both(|| thicken(&b, &bore, 0.4));
    close(measure::volume(&out), PI * (1.0 - 0.36) * 2.0);
    let out = both(|| thicken(&b, &bore, -0.5));
    close(measure::volume(&out), PI * (2.25 - 1.0) * 2.0);
    // The plate's top thickened carries the bore up through the slab.
    let top = faces(&b, &["p.end"]);
    let out = both(|| thicken(&b, &top, 1.0));
    close(measure::volume(&out), 40.0 - PI);
}

#[test]
fn a_face_thickened_until_its_neighbour_closes_goes_the_general_way() {
    // The step's lower top raised level with the upper closes the riser between them.
    let b = prism(
        &[
            [0.0, 0.0],
            [9.0, 0.0],
            [9.0, 2.0],
            [4.0, 2.0],
            [4.0, 6.0],
            [0.0, 6.0],
        ],
        5.0,
    );
    let low = faces(&b, &["p.e2"]);
    let (out, _) = counted(|| thicken(&b, &low, 4.0).unwrap());
    assert_same(&out, &general(|| thicken(&b, &low, 4.0)).unwrap());
    close(measure::volume(&out), 5.0 * 4.0 * 5.0);
}

#[test]
fn a_solid_turned_and_moved_is_shelled_and_thickened_the_same() {
    let mut b = l_prism();
    b.transform(&crate::math::Transform::rotation(
        Point3::new(1.0, 2.0, 3.0),
        Vec3::new(0.3, -0.5, 0.8).unit().unwrap(),
        0.7,
    ));
    let open = faces(&b, &["p.end"]);
    both(|| shell(&b, &open, 0.7, Wall::Inward, "s"));
    let walls = faces(&b, &["p.e2", "p.e3"]);
    both(|| thicken(&b, &walls, 1.0));
}

#[test]
fn a_smaller_solid_that_reaches_out_of_the_larger_is_not_laid() {
    use super::{Pairing, between};
    use std::collections::HashMap;
    // Each solid with a face moved out, paired the right way round and then the wrong way.
    let refused = |b: &Brep, moved: &[FaceId]| {
        let by: HashMap<FaceId, f64> = moved.iter().map(|&f| (f, 1.0)).collect();
        let grown = crate::edit::offset_faces(b, &by).unwrap();
        let own: Vec<bool> = b.face_ids().map(|f| moved.contains(&f)).collect();
        let same: Vec<(FaceId, FaceId)> = b
            .face_ids()
            .filter(|f| !moved.contains(f))
            .map(|f| (f, f))
            .collect();
        let pairing = |big, small| Pairing {
            big,
            small,
            own_big: own.clone(),
            own_small: own.clone(),
            same: same.clone(),
            kept: true,
        };
        assert!(between(&pairing(&grown, b)).is_some());
        between(&pairing(b, &grown)).is_none()
    };
    // A corner of the smaller past the larger's, on the edge that leads up to both.
    let b = block();
    assert!(refused(&b, &faces(&b, &["cube.end"])));
    // A whole loop of the smaller round the larger's.
    let cyl = primitive::cylinder(&Frame::WORLD, 4.0, 6.0, "cyl").unwrap();
    assert!(refused(&cyl, &faces(&cyl, &["cyl.side"])));
    // A void that holds its solid.
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    assert!(refused(&ball, &ball.face_ids().collect::<Vec<_>>()));
}
