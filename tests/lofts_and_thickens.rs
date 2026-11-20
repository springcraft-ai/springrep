//! Ruled lofts through square sections against the frustum and Cavalieri volumes, and faces
//! thickened by moving them along their outward normals, flat walls and a bore's cylinder.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use springrep::{
    boolean::{difference, unify},
    check,
    edit::offset_faces,
    loft::{Section, loft},
    math::{Frame, Vec2, Vec3},
    measure,
    pick::faces_named,
    profile::Profile,
    sweep::extrude,
    topo::{Brep, FaceId},
};

fn up(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
}

/// A `w` × `w` square centred on (`cx`, 0) on the plane at `z`.
fn square_section(z: f64, cx: f64, w: f64) -> Section {
    Section::new(Profile::rect(Vec2::new(cx, 0.0), w, w), up(z))
}

/// The 20 × 20 × 10 block on the xy plane, centred on the origin.
fn block() -> Brep {
    extrude(
        &Profile::rect(Vec2::ZERO, 20.0, 20.0),
        &up(0.0),
        10.0,
        "block",
    )
    .unwrap()
}

/// `solid` with each face named in `faces`, and every face on the surface of one, moved `d` along
/// its outward normal.
fn thickened(solid: &Brep, faces: &[&str], d: f64) -> Brep {
    let mut by: HashMap<FaceId, f64> = HashMap::new();
    for &name in faces {
        let found = faces_named(solid, name);
        assert!(!found.is_empty(), "no face is named {name}");
        by.extend(found.into_iter().map(|f| (f, d)));
    }
    unify(&offset_faces(solid, &by).unwrap()).unwrap()
}

/// The area of the one face carrying `name`.
fn named_area(b: &Brep, name: &str) -> f64 {
    let faces: Vec<_> = b
        .face_ids()
        .filter(|&f| b.face(f).names.iter().any(|n| n.to_string() == name))
        .collect();
    assert_eq!(faces.len(), 1, "{name}");
    measure::face_area(b, faces[0])
}

#[test]
fn a_ruled_loft_through_three_sections_is_two_frustums() {
    // Squares 20, 10 and 20 at z = 0, 10 and 20, ruled: two square frustums 10 high, each
    // h/3 · (A₁ + A₂ + √(A₁A₂)) = 10/3 · 700.
    let sections = [
        square_section(0.0, 0.0, 20.0),
        square_section(10.0, 0.0, 10.0),
        square_section(20.0, 0.0, 20.0),
    ];
    let b = unify(&loft(&sections, true, "waist").unwrap()).unwrap();
    assert!(check::check(&b).is_empty());
    let want = 2.0 * 10.0 / 3.0 * 700.0;
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
}

#[test]
fn a_ruled_loft_between_shifted_equal_sections_holds_their_area_all_the_way() {
    // Two 10 mm squares 10 apart, the top one 5 along x: a leaning prism, every level the same
    // square, so by Cavalieri 100 · 10.
    let sections = [
        square_section(0.0, 0.0, 10.0),
        square_section(10.0, 5.0, 10.0),
    ];
    let b = unify(&loft(&sections, true, "lean").unwrap()).unwrap();
    assert!(check::check(&b).is_empty());
    assert!(
        (measure::volume(&b) - 1000.0).abs() < 1e-9 * 1000.0,
        "{}",
        measure::volume(&b)
    );
}

#[test]
fn thickening_a_wall_moves_it_out_and_stretches_the_faces_beside_it() {
    // The 20 × 20 × 10 block's +x wall moved out 2: a 22 × 20 × 10 block whose top is 22 × 20.
    let b = thickened(&block(), &["block.u_max"], 2.0);
    assert!((measure::volume(&b) - 4400.0).abs() < 1e-9 * 4400.0);
    assert_eq!(b.face_count(), 6);
    assert!((named_area(&b, "block.end") - 440.0).abs() < 1e-9);
    assert!((named_area(&b, "block.u_max") - 200.0).abs() < 1e-9);
    // Negative takes material away: both side walls in 1 leaves 18 × 20 × 10.
    let b = thickened(&block(), &["block.u_min", "block.u_max"], -1.0);
    assert!((measure::volume(&b) - 3600.0).abs() < 1e-9 * 3600.0);
    assert!((named_area(&b, "block.start") - 360.0).abs() < 1e-9);
}

#[test]
fn thickening_a_bore_closes_it_and_thinning_it_opens_it() {
    // A Ø6 hole through the 10 mm block: its wall's outward normal points into the hole, so +1
    // leaves a Ø4 bore and −1 a Ø8 one.
    let hole = Profile::named_circle(Vec2::ZERO, 6.0, "wall");
    let hole = extrude(&hole, &up(0.0), 10.0, "hole").unwrap();
    let drilled = difference(&block(), &hole).unwrap();
    for (d, r) in [(1.0, 2.0), (-1.0, 4.0)] {
        let b = thickened(&drilled, &["hole.wall"], d);
        let want = 4000.0 - PI * r * r * 10.0;
        assert!((measure::volume(&b) - want).abs() < 1e-9 * want, "{d}");
        assert!(
            (named_area(&b, "hole.wall") - TAU * r * 10.0).abs() < 1e-8,
            "{d}"
        );
    }
}
