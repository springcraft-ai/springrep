use std::f64::consts::PI;

use super::*;
use crate::{
    check,
    error::Error,
    geom::{Curve, Surface},
    math::{Frame, Vec2, Vec3},
    measure,
    pick::faces_named,
    primitive::{block, cylinder, sphere},
    profile::{NamedSegment, Profile, Segment},
    sweep::extrude,
    topo::{Brep, FaceId, FaceName},
};

fn at(x: f64, y: f64, z: f64) -> Frame {
    Frame::WORLD.translated(Vec3::new(x, y, z))
}

fn named(b: &Brep, names: &[&str]) -> Vec<FaceId> {
    names
        .iter()
        .flat_map(|n| faces_named(b, n))
        .collect::<Vec<_>>()
}

fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() < 1e-7 * want.max(1.0),
        "{got} against {want}"
    );
}

/// The floor, z = 0 from (0, 0) to (10, 8), and the wall, x = 0 up to z = 10, of the block from
/// the origin to (10, 8, 10): an angle whose normals, -z and -x, point out of its crease.
fn angle() -> Brep {
    let b = block(&Frame::WORLD, Vec3::new(10.0, 8.0, 10.0), "b").unwrap();
    from_faces(&b, &named(&b, &["b.start", "b.u_min"])).unwrap()
}

#[test]
fn faces_of_a_solid_are_a_sheet_sharing_their_edges() {
    let s = angle();
    assert!(check::check_sheet(&s).is_empty());
    assert!(!check::check(&s).is_empty(), "a sheet is no solid");
    assert_eq!(s.face_count(), 2);
    // Two rectangles share one edge: 4 + 4 - 1 edges, and they are one shell.
    assert_eq!(s.edge_ids().count(), 7);
    assert_eq!(s.shell_ids().count(), 1);
    let boundary = s.edge_ids().filter(|&e| s.edge_uses(e).len() == 1).count();
    assert_eq!(boundary, 6);
}

#[test]
fn faces_apart_are_shells_apart_and_a_face_is_taken_once() {
    let b = block(&Frame::WORLD, Vec3::new(1.0, 2.0, 3.0), "b").unwrap();
    let ends = named(&b, &["b.start", "b.end"]);
    assert_eq!(from_faces(&b, &ends).unwrap().shell_ids().count(), 2);
    assert!(matches!(
        from_faces(&b, &[ends[0], ends[0]]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(from_faces(&b, &[]), Err(Error::InvalidInput(_))));
}

#[test]
fn a_sheet_turned_over_faces_the_other_way() {
    let s = angle();
    let t = flipped(&s).unwrap();
    assert!(check::check_sheet(&t).is_empty());
    for (f, g) in s.face_ids().zip(t.face_ids()) {
        assert_ne!(s.face(f).reversed, t.face(g).reversed);
        assert_eq!(s.face(f).names, t.face(g).names);
    }
}

#[test]
fn a_rectangle_thickened_is_a_box() {
    let b = block(&at(0.0, 0.0, -1.0), Vec3::new(10.0, 8.0, 1.0), "p").unwrap();
    let sheet = from_faces(&b, &named(&b, &["p.end"])).unwrap();
    let out = thicken(&sheet, 2.0, 0.0, Corners::Sharp, "t").unwrap();
    close(measure::volume(&out), 160.0);
    close(measure::area(&out), 2.0 * 80.0 + 2.0 * 18.0 * 2.0);
    // The sheet and its moved copy keep its name; the four walls are the thicken's sides.
    let side = FaceName::new("t", "side");
    let walls = out
        .face_ids()
        .filter(|&f| out.face(f).names.contains(&side))
        .count();
    assert_eq!(walls, 4);
    assert_eq!(faces_named(&out, "p.end").len(), 2);
}

#[test]
fn an_angle_thickened_outside_inside_and_rounded() {
    // Each arm 10 × 8 × 1; outside, the corner square 1 × 1 × 8 joins them; inside they overlap
    // in it; rounded, the quarter cylinder of radius 1 and length 8 joins them.
    let s = angle();
    let out = thicken(&s, 1.0, 0.0, Corners::Sharp, "t").unwrap();
    close(measure::volume(&out), 168.0);
    let inside = thicken(&s, 0.0, 1.0, Corners::Sharp, "t").unwrap();
    close(measure::volume(&inside), 152.0);
    let round = thicken(&s, 1.0, 0.0, Corners::Rounded, "t").unwrap();
    close(measure::volume(&round), 160.0 + 8.0 * PI / 4.0);
}

#[test]
fn an_open_box_thickened_out_rounded_is_the_cube_grown_below_its_top() {
    // Steiner's formula over five faces, eight edges and four corners: 5a²t + 8·(π/4)t²a +
    // 4·(π/6)t³.
    let (a, t) = (4.0, 1.0);
    let cube = block(&Frame::WORLD, Vec3::new(a, a, a), "c").unwrap();
    let names = ["c.start", "c.u_min", "c.u_max", "c.v_min", "c.v_max"];
    let tray = from_faces(&cube, &named(&cube, &names)).unwrap();
    let out = thicken(&tray, t, 0.0, Corners::Rounded, "t").unwrap();
    let want = 5.0 * a * a * t + 2.0 * PI * t * t * a + 2.0 * PI * t.powi(3) / 3.0;
    close(measure::volume(&out), want);
    // Into the cube, sharp: the cube less the box (a − 2t)² × (a − t).
    let inward = thicken(&tray, 0.0, t, Corners::Sharp, "t").unwrap();
    close(
        measure::volume(&inward),
        a.powi(3) - (a - 2.0 * t).powi(2) * (a - t),
    );
}

#[test]
fn a_cylinder_and_a_half_one_thickened_are_tubes() {
    let can = cylinder(&Frame::WORLD, 5.0, 10.0, "can").unwrap();
    let side = from_faces(&can, &named(&can, &["can.side"])).unwrap();
    let tube = thicken(&side, 1.0, 0.0, Corners::Sharp, "t").unwrap();
    close(measure::volume(&tube), PI * 10.0 * (36.0 - 25.0));
    // The half y >= 0, thickened toward the axis: a part of a turn's slab.
    let half_disk = Profile::outline(vec![
        NamedSegment::arc(Vec2::ZERO, 5.0, 0.0, PI, "skin"),
        NamedSegment::line(Vec2::new(-5.0, 0.0), Vec2::new(5.0, 0.0), "flat"),
    ]);
    let half = extrude(&half_disk, &Frame::WORLD, 10.0, "pipe").unwrap();
    let skin = from_faces(&half, &named(&half, &["pipe.skin"])).unwrap();
    let pipe = thicken(&skin, 0.0, 2.0, Corners::Sharp, "t").unwrap();
    close(measure::volume(&pipe), PI * 10.0 * (25.0 - 9.0) / 2.0);
}

#[test]
fn a_sphere_thickened_both_ways_is_a_shell() {
    let ball = sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let skin = from_faces(&ball, &named(&ball, &["ball.skin"])).unwrap();
    let out = thicken(&skin, 1.0, 1.0, Corners::Sharp, "t").unwrap();
    close(measure::volume(&out), 4.0 * PI / 3.0 * (216.0 - 64.0));
}

#[test]
fn a_parabolic_sheet_thickened_toward_its_focus() {
    // z = x²/8 over -4 <= x <= 4, 10 long. The strip between a curve and its parallel at t on its
    // concave side has area t·L − t²·θ/2, θ its turning, here a right angle; L is the parabola's
    // length, 4√2 + 4 asinh 1.
    let parabola = Segment::Bezier {
        ctrl: vec![
            Vec2::new(-4.0, 2.0),
            Vec2::new(0.0, -2.0),
            Vec2::new(4.0, 2.0),
        ],
        weights: vec![1.0; 3],
    };
    let plane = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let trough = extrude(
        &Profile::outline(vec![
            NamedSegment::new(parabola, "sheet"),
            NamedSegment::line(Vec2::new(4.0, 2.0), Vec2::new(-4.0, 2.0), "top"),
        ]),
        &plane,
        -10.0,
        "trough",
    )
    .unwrap();
    let sheet = from_faces(&trough, &named(&trough, &["trough.sheet"])).unwrap();
    assert!(matches!(
        &sheet.face(sheet.face_ids().next().unwrap()).surface,
        Surface::Extrusion(e) if !matches!(e.curve, Curve::Line(_))
    ));
    let out = thicken(&sheet, 0.0, 1.0, Corners::Sharp, "t").unwrap();
    let length = 4.0 * 2f64.sqrt() + 4.0 * 1f64.asinh();
    close(measure::volume(&out), 10.0 * (length - PI / 4.0));
}

#[test]
fn only_a_valid_sheet_is_thickened() {
    let b = block(&Frame::WORLD, Vec3::new(1.0, 1.0, 1.0), "b").unwrap();
    assert!(matches!(
        thicken(&b, 1.0, 0.0, Corners::Sharp, "t"),
        Err(Error::InvalidInput(_))
    ));
    let s = angle();
    for (front, back) in [(0.0, 0.0), (-1.0, 2.0), (f64::NAN, 1.0)] {
        assert!(matches!(
            thicken(&s, front, back, Corners::Sharp, "t"),
            Err(Error::InvalidInput(_))
        ));
    }
}
