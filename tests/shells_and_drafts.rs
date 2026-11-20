//! Shells and drafts made by moving faces: a shell is the solid less its own faces offset inward
//! (or, grown outward, the offset solid less the original), open where a face is left in place,
//! and a draft tilts faces about where they meet a neutral plane. Flat, cylindrical and spherical
//! walls are each measured against closed forms.

use std::collections::BTreeSet;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

use springrep::{
    Error,
    blend::fillet,
    boolean::unify,
    check,
    edit::{self, Wall},
    geom::Surface,
    math::{Frame, Vec2, Vec3},
    measure,
    pick::faces_named,
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
    topo::{Brep, FaceId},
};

fn up(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
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

/// A cylinder R`r` × `h` standing on the xy plane.
fn can(r: f64, h: f64) -> Brep {
    extrude(&Profile::circle(Vec2::ZERO, 2.0 * r), &up(0.0), h, "can").unwrap()
}

/// The faces `names` stand for: those carrying one, and every face on the surface of one.
fn picked(b: &Brep, names: &[&str]) -> BTreeSet<FaceId> {
    let mut out = BTreeSet::new();
    for &name in names {
        let found = faces_named(b, name);
        assert!(!found.is_empty(), "no face is named {name}");
        out.extend(found);
    }
    out
}

/// `solid` hollowed to a wall `t` thick, open at the faces named `open`.
fn shell(solid: &Brep, open: &[&str], t: f64, id: &str) -> Brep {
    let open: Vec<FaceId> = picked(solid, open).into_iter().collect();
    unify(&edit::shell(solid, &open, t, Wall::Inward, id).unwrap()).unwrap()
}

/// `solid` with a wall `t` thick grown round it, open at the faces named `open`.
fn shell_outward(solid: &Brep, open: &[&str], t: f64, id: &str) -> Brep {
    let open: Vec<FaceId> = picked(solid, open).into_iter().collect();
    unify(&edit::shell(solid, &open, t, Wall::Outward, id).unwrap()).unwrap()
}

/// `solid` with the faces named `faces` tilted by `angle` about where they meet the xy plane.
fn drafted(solid: &Brep, faces: &[&str], angle: f64) -> Brep {
    let faces: Vec<FaceId> = picked(solid, faces).into_iter().collect();
    unify(&edit::draft(solid, &faces, &up(0.0), angle).unwrap()).unwrap()
}

fn faces_of(b: &Brep, kind: &str) -> usize {
    b.face_ids()
        .filter(|&f| b.face(f).surface.kind() == kind)
        .count()
}

#[test]
fn a_shell_open_at_the_top_leaves_a_cup() {
    // A 20 × 20 × 10 mm block less a 16 × 16 × 8 cavity open at the top: 4000 − 2048.
    let b = shell(&block(), &["block.end"], 2.0, "hollow");
    assert!((measure::volume(&b) - 1952.0).abs() < 1e-9);
    let names = b.name_counts();
    assert_eq!(names.get("hollow.inner"), Some(&5), "{names:?}");
    assert_eq!(names.get("block.end"), Some(&1), "{names:?}");
}

// The 20 × 20 × 10 block open at its top, shelled 10: the walls meet in the planes x = 0 and
// y = 0 and the floor reaches the top, so nothing is hollowed. Shelled 11 the walls pass through
// each other, and a 40 × 40 × 10 block shelled 11 has its floor pass the top: both are refused.
#[test]
fn a_shell_as_thick_as_the_part_is_refused() {
    let top = |b: &Brep| picked(b, &["block.end"]).into_iter().collect::<Vec<_>>();
    let b = block();
    let met = edit::shell(&b, &top(&b), 10.0, Wall::Inward, "hollow").unwrap();
    assert!((measure::volume(&met) - 4000.0).abs() < 1e-9);
    let through = edit::shell(&b, &top(&b), 11.0, Wall::Inward, "hollow");
    assert!(
        matches!(through, Err(Error::InvalidInput(_))),
        "{through:?}"
    );
    let wide = extrude(
        &Profile::rect(Vec2::ZERO, 40.0, 40.0),
        &up(0.0),
        10.0,
        "block",
    )
    .unwrap();
    let floor = edit::shell(&wide, &top(&wide), 11.0, Wall::Inward, "hollow");
    assert!(matches!(floor, Err(Error::InvalidInput(_))), "{floor:?}");
}

// A frustum of radius 6 at z = 0 and 3 at z = 4, open at its top, shelled 3.5: its side moved in
// closes onto its apex below the moved base, with no two faces moved through each other, and
// nothing is hollowed. Left is the frustum, πH(R² + Rr + r²)/3 = 84π.
#[test]
fn a_frustum_shelled_past_its_apex_is_left_whole() {
    let f = springrep::primitive::cone(&up(0.0), 6.0, 3.0, 4.0, "frustum").unwrap();
    let open: Vec<FaceId> = picked(&f, &["frustum.top"]).into_iter().collect();
    let out = edit::shell(&f, &open, 3.5, Wall::Inward, "hollow").unwrap();
    assert!((measure::volume(&out) - 84.0 * PI).abs() < 1e-9 * 84.0 * PI);
}

#[test]
fn a_shell_grown_outward_keeps_the_inside() {
    // The 20 × 20 × 10 mm block grown 2 mm out on every side but the open top: 24 · 24 · 12 − 4000.
    let b = shell_outward(&block(), &["block.end"], 2.0, "skin");
    assert!((measure::volume(&b) - (24.0 * 24.0 * 12.0 - 4000.0)).abs() < 1e-9);
    // The inside is the block's own faces, where they were: each keeps its name and is the wall's
    // inside too. The grown outside is new and names nothing; the rim keeps the open top's name.
    let names = b.name_counts();
    assert_eq!(names.get("block.u_min"), Some(&1), "{names:?}");
    assert_eq!(names.get("block.start"), Some(&1), "{names:?}");
    assert_eq!(names.get("skin.inner"), Some(&5), "{names:?}");
    assert_eq!(names.get("block.end"), Some(&1), "{names:?}");
    assert_eq!(b.face_count(), 11);
}

#[test]
fn a_shell_open_at_a_step_keeps_the_wall_beside_the_opening() {
    // An L, (0, 0) to (20, 20) less x < 15, y > 5, 20 tall: 3500. Open at its step y = 5, the
    // face x = 15 stands beside the opening, and its wall runs down to the opening's plane.
    let pts = [
        [0.0, 0.0],
        [20.0, 0.0],
        [20.0, 20.0],
        [15.0, 20.0],
        [15.0, 5.0],
        [0.0, 5.0],
    ];
    let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
    let names = ["s0", "s1", "s2", "s3", "step", "s5"];
    let l = extrude(&Profile::polygon(&pts, &names), &up(0.0), 20.0, "l").unwrap();
    // Inward 1: the cavity is (1, 1) to (19, 5) and (16, 5) to (19, 19), 18 tall.
    let b = shell(&l, &["l.step"], 1.0, "in");
    assert!(check::check(&b).is_empty());
    let want = 3500.0 - 18.0 * (18.0 * 4.0 + 3.0 * 14.0);
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
    // Outward 1: (−1, −1) to (21, 5) and (14, 5) to (21, 21), 22 tall, less the L.
    let b = shell_outward(&l, &["l.step"], 1.0, "out");
    assert!(check::check(&b).is_empty());
    let want = 22.0 * (22.0 * 6.0 + 7.0 * 16.0) - 3500.0;
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
}

#[test]
fn a_draft_tapers_the_walls_from_the_neutral_plane() {
    // Two opposite walls of the 20 × 20 × 10 mm block tilted so they come 2 mm in at the top: a
    // trapezoid 20 at the foot and 16 at the top, 10 high, 20 deep.
    let angle = (2.0f64 / 10.0).atan();
    let b = drafted(&block(), &["block.u_min", "block.u_max"], angle);
    let want = 20.0 * 10.0 * 36.0 / 2.0;
    assert!((measure::volume(&b) - want).abs() < 1e-9 * want);
    assert_eq!(b.name_counts().get("block.u_min"), Some(&1));
}

#[test]
fn a_cylinder_shelled_open_at_the_top_is_a_cup_with_a_round_wall() {
    // R10 × 20 hollowed 2 mm, the top open: the cavity is the cylinder R8 from z = 2 to 20, its
    // wall the cylinder's offset 2 mm in, exactly.
    let b = shell(&can(10.0, 20.0), &["can.end"], 2.0, "cup");
    assert!(check::check(&b).is_empty());
    let want = PI * (100.0 * 20.0 - 64.0 * 18.0);
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
    let walls: Vec<f64> = b
        .face_ids()
        .filter_map(|f| match &b.face(f).surface {
            Surface::Cylinder(c) => Some(c.radius),
            _ => None,
        })
        .collect();
    assert_eq!(walls.len(), 2);
    assert!(walls.iter().any(|r| (r - 8.0).abs() < 1e-12), "{walls:?}");
}

#[test]
fn a_cylinder_shelled_outward_grows_its_wall_round_it() {
    // R10 × 20 shelled 2 mm outward, the top open: the outside is R12 from z = −2 to 20 and the
    // inside the original cylinder.
    let b = shell_outward(&can(10.0, 20.0), &["can.end"], 2.0, "cup");
    assert!(check::check(&b).is_empty());
    let want = PI * (144.0 * 22.0 - 100.0 * 20.0);
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
}

#[test]
fn a_dome_shelled_open_at_its_base_is_a_spherical_cap() {
    // An R10 hemisphere standing on its flat, hollowed 1 mm with the flat open: the shell between
    // the spheres R10 and R9, 2π(10³ − 9³)/3.
    let segment = |segment: Segment, name: &str| NamedSegment::new(segment, name);
    let (foot, pole) = (Vec2::new(10.0, 0.0), Vec2::new(0.0, 10.0));
    let quarter = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                segment(
                    Segment::Arc {
                        center: Vec2::ZERO,
                        radius: 10.0,
                        start: 0.0,
                        sweep: FRAC_PI_2,
                    },
                    "ball_side",
                ),
                segment(
                    Segment::Line {
                        a: pole,
                        b: Vec2::ZERO,
                    },
                    "half_u_min",
                ),
                segment(
                    Segment::Line {
                        a: Vec2::ZERO,
                        b: foot,
                    },
                    "half_v_min",
                ),
            ],
        }],
    };
    let rz = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let dome = revolve(&quarter, &rz, Vec3::ZERO, rz.y, TAU, "dome").unwrap();
    let b = shell(&dome, &["dome.half_v_min"], 1.0, "cap");
    assert!(check::check(&b).is_empty());
    let want = 2.0 * PI * (1000.0 - 729.0) / 3.0;
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
}

#[test]
fn a_rounded_block_shelled_keeps_its_rounds_as_offset_cylinders() {
    // The 20 × 20 × 10 block with its four vertical edges rounded R4, hollowed 1 mm with the top
    // open. Each section is the rounded square, and the cavity's the square 2 smaller with rounds
    // R3: (20² − (4 − π)·4²)·10 − (18² − (4 − π)·3²)·9.
    let rounded = fillet(
        &block(),
        &[
            ("block.u_min", "block.v_min"),
            ("block.v_min", "block.u_max"),
            ("block.u_max", "block.v_max"),
            ("block.v_max", "block.u_min"),
        ],
        4.0,
        "round",
    )
    .unwrap();
    let b = shell(&rounded, &["block.end"], 1.0, "tray");
    assert!(check::check(&b).is_empty());
    let section = |w: f64, r: f64| w * w - (4.0 - PI) * r * r;
    let want = section(20.0, 4.0) * 10.0 - section(18.0, 3.0) * 9.0;
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
}

#[test]
fn a_drafted_cylinder_boss_is_a_frustum() {
    // R10 × 10, its side drafted 5° from the base: a frustum from R10 to R10 − 10·tan 5°.
    let b = drafted(&can(10.0, 10.0), &["can.side"], 5f64.to_radians());
    assert!(check::check(&b).is_empty());
    let r = 10.0 - 10.0 * 5f64.to_radians().tan();
    let want = PI * 10.0 / 3.0 * (100.0 + 10.0 * r + r * r);
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
    assert_eq!(faces_of(&b, "cone"), 1);
}

#[test]
fn a_cylinder_parted_at_its_middle_is_drafted_both_ways() {
    // R10 × 20 parted at z = 10, each half drafted 3° away from the parting line: two frustums from
    // R10 at z = 10 to R10 − 10·tan 3° at either end, with a crease between them.
    let post = can(10.0, 20.0);
    let side: Vec<FaceId> = faces_named(&post, "can.side").into_iter().collect();
    let angle = 3f64.to_radians();
    let b = edit::draft_parted(&post, &side, &up(10.0), angle).unwrap();
    assert!(check::check(&b).is_empty());
    let r = 10.0 - 10.0 * angle.tan();
    let want = 2.0 * PI * 10.0 / 3.0 * (100.0 + 10.0 * r + r * r);
    let area = 2.0 * PI * r * r + 2.0 * PI * (10.0 + r) * 10.0 / angle.cos();
    assert!((measure::volume(&b) - want).abs() < 1e-9 * want);
    assert!((measure::area(&b) - area).abs() < 1e-9 * area);
    assert_eq!(faces_of(&b, "cone"), 2);
    // both halves keep the side's name
    assert_eq!(faces_named(&b, "can.side").len(), 2);
    // Parted below the part, the whole side is drafted with the pull along the normal, narrowing
    // from R10 at z = −5.
    let b = edit::draft_parted(&post, &side, &up(-5.0), angle).unwrap();
    let r = 10.0 - 5.0 * angle.tan();
    let top = 10.0 - 25.0 * angle.tan();
    let want = PI * 20.0 / 3.0 * (r * r + r * top + top * top);
    assert!((measure::volume(&b) - want).abs() < 1e-9 * want);
}

#[test]
fn a_drafted_bore_widens_along_the_pull() {
    // The 20 × 20 × 10 block with a Ø8 through hole, the hole's wall drafted 3° from the base: the
    // hole is a frustum from R4 at z = 0 to R4 + 10·tan 3° at the top.
    let bore = Profile::named_circle(Vec2::ZERO, 8.0, "bore_side").loops;
    let drilled = Profile::rect(Vec2::ZERO, 20.0, 20.0).with_holes(bore);
    let drilled = extrude(&drilled, &up(0.0), 10.0, "block").unwrap();
    let b = drafted(&drilled, &["block.bore_side"], 3f64.to_radians());
    assert!(check::check(&b).is_empty());
    let r = 4.0 + 10.0 * 3f64.to_radians().tan();
    let want = 4000.0 - PI * 10.0 / 3.0 * (16.0 + 4.0 * r + r * r);
    assert!(
        (measure::volume(&b) - want).abs() < 1e-9 * want,
        "{}",
        measure::volume(&b)
    );
}
