//! Operations a caller once assembled from others, each as one call, against closed forms.

use std::f64::consts::{FRAC_PI_2, PI};

use springrep::{
    Error, check, edit,
    geom::{BSpline, Curve, Line},
    math::{Frame, Transform, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    primitive,
    profile::{Profile, Segment},
    step,
    sweep::{self, Leg},
    topo::{Brep, FaceId},
};

fn valid(b: &Brep) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
}

fn face_through(b: &Brep, p: Vec3) -> FaceId {
    let (f, d) = measure::face_at(b, p).unwrap();
    assert!(d < 1e-9, "no face through {p:?}: {d}");
    f
}

#[test]
fn a_bend_turns_about_the_axis_it_is_given() {
    // A 2 × 2 square 5 mm up z, then a quarter turn of radius 4 about x: the path turns toward
    // x × z = −y and ends at y = −4, z = 9, heading along −y, so the solid reaches y = −4 and
    // z = 5 + 4 + 1.
    let p = Profile::rect(Vec2::ZERO, 2.0, 2.0);
    let legs = [
        Leg::Line { length: 5.0 },
        Leg::Arc {
            radius: 4.0,
            angle: FRAC_PI_2,
            axis: Vec3::X,
        },
    ];
    let b = sweep::along(&p, &Frame::WORLD, &legs, "swp").unwrap();
    valid(&b);
    let want = 4.0 * (5.0 + 4.0 * FRAC_PI_2);
    assert!((measure::volume(&b) - want).abs() < 1e-9 * want);
    let bb = measure::bbox(&b);
    assert!((bb.min.y + 4.0).abs() < 1e-9 && (bb.max.y - 1.0).abs() < 1e-9);
    assert!((bb.max.z - 10.0).abs() < 1e-9 && (bb.max.x - 1.0).abs() < 1e-9);
    let askew = [Leg::Arc {
        radius: 4.0,
        angle: FRAC_PI_2,
        axis: Vec3::Z,
    }];
    assert!(sweep::along(&p, &Frame::WORLD, &askew, "swp").is_err());
}

#[test]
fn a_sharp_turn_is_filled_by_the_outside_of_the_profile_turned_about_the_corner() {
    // A 2 × 2 square 5 mm up z, a quarter turn on the spot about y, then 5 mm along x. Each leg
    // is 20 mm³ and they share the 1 × 1 × 2 where they cross inside the turn. Outside it the
    // half of the square beyond the corner, 1 × 2, turns a quarter about the corner's axis: a
    // quarter cylinder of radius 1, 2 long.
    let p = Profile::rect(Vec2::ZERO, 2.0, 2.0);
    let legs = [
        Leg::Line { length: 5.0 },
        Leg::Turn {
            angle: FRAC_PI_2,
            axis: Vec3::Y,
        },
        Leg::Line { length: 5.0 },
    ];
    let b = sweep::along(&p, &Frame::WORLD, &legs, "swp").unwrap();
    valid(&b);
    let want = 20.0 + 20.0 - 2.0 + FRAC_PI_2;
    assert!((measure::volume(&b) - want).abs() < 1e-9 * want);
    let bb = measure::bbox(&b);
    assert!((bb.min.x + 1.0).abs() < 1e-9 && (bb.max.x - 5.0).abs() < 1e-9);
    assert!((bb.max.z - 6.0).abs() < 1e-9);
}

#[test]
fn mitered_legs_meet_in_the_plane_that_bisects_their_corner() {
    // A 2 × 2 square along z for 5, a turn of 120° about y, then 5 on: each leg cut by the miter's
    // plane holds the section times its length on the path, and the faces either side of the
    // miter are each one plane.
    let p = Profile::rect(Vec2::ZERO, 2.0, 2.0);
    let turn = 2.0 * PI / 3.0;
    let legs = [
        Leg::Line { length: 5.0 },
        Leg::Miter {
            angle: turn,
            axis: Vec3::Y,
        },
        Leg::Line { length: 5.0 },
    ];
    let b = sweep::along(&p, &Frame::WORLD, &legs, "swp").unwrap();
    valid(&b);
    assert!((measure::volume(&b) - 40.0).abs() < 1e-9 * 40.0);
    assert!((measure::area(&b) - (4.0 * 2.0 * 10.0 + 2.0 * 4.0)).abs() < 1e-9 * 88.0);
    assert_eq!(b.face_count(), 2 + 2 + 2 * 2);
    // The outer corner of the miter lies 1·tan 60° past the corner along the first leg.
    let reach = (turn / 2.0).tan();
    let hi = measure::bbox(&b).max;
    assert!((hi.z - (5.0 + reach)).abs() < 1e-9, "{hi:?}");
    // A miter beside a bend, or ending the path, is refused.
    let bent = [
        Leg::Line { length: 5.0 },
        Leg::Miter {
            angle: turn,
            axis: Vec3::Y,
        },
        Leg::Arc {
            radius: 4.0,
            angle: FRAC_PI_2,
            axis: Vec3::Y,
        },
    ];
    assert!(sweep::along(&p, &Frame::WORLD, &bent, "swp").is_err());
    assert!(sweep::along(&p, &Frame::WORLD, &legs[..2], "swp").is_err());
}

#[test]
fn a_closed_path_of_mitered_legs_has_no_ends() {
    // A 2 × 1 section standing outside a 10 × 10 square path, round all four mitered corners back
    // to the start: the ring between the squares 10 and 14 across, 1 tall.
    let start = Frame::new(Vec3::ZERO, Vec3::X, Vec3::Z).unwrap();
    // The sketch's x is up and its y outward, away from the square's inside, −y on the first side.
    let p = Profile::rect(Vec2::new(0.5, 1.0), 1.0, 2.0);
    let mut legs = Vec::new();
    for _ in 0..4 {
        legs.push(Leg::Line { length: 10.0 });
        legs.push(Leg::Miter {
            angle: FRAC_PI_2,
            axis: Vec3::Z,
        });
    }
    let b = sweep::along(&p, &start, &legs, "frame").unwrap();
    valid(&b);
    let (v, a) = (measure::volume(&b), measure::area(&b));
    assert!((v - 96.0).abs() < 1e-9 * 96.0, "{v}");
    assert!(
        (a - (2.0 * 96.0 + 4.0 * 14.0 + 4.0 * 10.0)).abs() < 1e-9 * a,
        "{a}"
    );
    let bb = measure::bbox(&b);
    assert!(
        (bb.min - Vec3::new(-2.0, -2.0, 0.0)).max_abs() < 1e-9,
        "{bb:?}"
    );
    assert!(
        (bb.max - Vec3::new(12.0, 12.0, 1.0)).max_abs() < 1e-9,
        "{bb:?}"
    );
}

#[test]
fn a_round_pipe_mitered_round_a_corner_out_of_its_plane() {
    // A disk of radius 1 along three edges of a cube of side 10: each run between its miters'
    // planes holds πr² times its length on the path.
    let start = Frame::new(Vec3::ZERO, Vec3::X, Vec3::Z).unwrap();
    let legs = [
        Leg::Line { length: 10.0 },
        Leg::Miter {
            angle: FRAC_PI_2,
            axis: Vec3::Z,
        },
        Leg::Line { length: 10.0 },
        Leg::Miter {
            angle: FRAC_PI_2,
            axis: Vec3::X,
        },
        Leg::Line { length: 10.0 },
    ];
    let b = sweep::along(&Profile::circle(Vec2::ZERO, 2.0), &start, &legs, "pipe").unwrap();
    valid(&b);
    let want = PI * 30.0;
    assert!((measure::volume(&b) - want).abs() < 1e-9 * want);
    let area = 2.0 * PI * 30.0 + 2.0 * PI;
    assert!((measure::area(&b) - area).abs() < 1e-9 * area);
    let hi = measure::bbox(&b).max;
    assert!(
        (hi - Vec3::new(11.0, 11.0, 10.0)).max_abs() < 1e-9,
        "{hi:?}"
    );
}

#[test]
fn a_sweep_along_lines_and_an_arc_has_its_profile_s_area_times_the_path() {
    // A 2 × 1 rectangle centred on the path: 5 mm up z, a quarter turn of radius 4 toward x, 3 mm
    // on along x. By Pappus the volume is the area times the path's length.
    let p = Profile::rect(Vec2::ZERO, 2.0, 1.0);
    let legs = [
        Leg::Line { length: 5.0 },
        Leg::Arc {
            radius: 4.0,
            angle: FRAC_PI_2,
            axis: Vec3::Y,
        },
        Leg::Line { length: 3.0 },
    ];
    let b = sweep::along(&p, &Frame::WORLD, &legs, "swp").unwrap();
    valid(&b);
    let length = 5.0 + 4.0 * FRAC_PI_2 + 3.0;
    let v = measure::volume(&b);
    assert!((v - 2.0 * length).abs() < 1e-9, "{v}");
    // The bend ends 4 out and 4 up from its start, the profile 1 to either side; 3 more along x.
    let hi = measure::bbox(&b).max;
    assert!(
        (hi.x - 7.0).abs() < 1e-9 && (hi.z - 10.0).abs() < 1e-9,
        "{hi:?}"
    );
}

#[test]
fn a_sweep_along_no_legs_is_invalid() {
    let p = Profile::rect(Vec2::ZERO, 2.0, 1.0);
    assert!(matches!(
        sweep::along(&p, &Frame::WORLD, &[], "swp"),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn a_path_sweep_along_a_straight_path_is_a_prism_or_a_twisted_one() {
    // A 2 × 1 rectangle 6 mm along z: a line, and a cubic whose control points lie on that line.
    let p = Profile::rect(Vec2::ZERO, 2.0, 1.0);
    let line = Curve::Line(Line {
        origin: Vec3::ZERO,
        dir: Vec3::Z,
    });
    let ctrl = [0.0, 1.0, 4.0, 6.0]
        .map(|z| Vec3::new(0.0, 0.0, z))
        .to_vec();
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let cubic = Curve::BSpline(BSpline::new(3, knots, ctrl).unwrap());
    for (path, range) in [(&line, (0.0, 6.0)), (&cubic, (0.0, 1.0))] {
        let flat = sweep::along_curve(&p, &Frame::WORLD, path, range, 0.0, "swp").unwrap();
        valid(&flat);
        assert_eq!(flat.face_ids().count(), 6);
        let (v, a) = (measure::volume(&flat), measure::area(&flat));
        assert!(
            (v - 12.0).abs() < 1e-9 && (a - 40.0).abs() < 1e-9,
            "{v} {a}"
        );
        // Twisted a quarter turn, each section is still the rectangle: the volume is unchanged.
        let twisted = sweep::along_curve(&p, &Frame::WORLD, path, range, FRAC_PI_2, "swp").unwrap();
        valid(&twisted);
        let v = measure::volume(&twisted);
        assert!((v - 12.0).abs() < 1e-7, "{v}");
        // Its top is the rectangle turned: 1 wide along x and 2 along y.
        let top = face_through(&twisted, Vec3::new(0.0, 0.9, 6.0));
        assert!(
            measure::face_at(&twisted, Vec3::new(0.9, 0.0, 6.0))
                .is_none_or(|(f, d)| f != top || d > 1e-6)
        );
    }
}

#[test]
fn a_shell_open_at_the_top_keeps_walls_of_its_thickness() {
    // A 10 × 8 × 6 block hollowed to 1 mm walls, open at the top: the block less a 8 × 6 × 5
    // pocket.
    let block = primitive::block(&Frame::WORLD, Vec3::new(10.0, 8.0, 6.0), "blk").unwrap();
    let top = face_through(&block, Vec3::new(5.0, 4.0, 6.0));
    let b = edit::shell(&block, &[top], 1.0, edit::Wall::Inward, "sh").unwrap();
    valid(&b);
    let v = measure::volume(&b);
    assert!((v - (480.0 - 240.0)).abs() < 1e-9, "{v}");
    // A wall as thick as half the part leaves no hollow: the cavity closes, and the block stays
    // whole.
    let b = edit::shell(&block, &[top], 4.0, edit::Wall::Inward, "sh").unwrap();
    valid(&b);
    let v = measure::volume(&b);
    assert!((v - 480.0).abs() < 1e-9, "{v}");
}

#[test]
fn faces_moved_by_a_motion_go_where_it_takes_them() {
    // The top of a 10 × 8 × 6 block lifted 2 mm: a 10 × 8 × 8 block.
    let block = primitive::block(&Frame::WORLD, Vec3::new(10.0, 8.0, 6.0), "blk").unwrap();
    let top = face_through(&block, Vec3::new(5.0, 4.0, 6.0));
    let lift = Transform::translation(Vec3::new(0.0, 0.0, 2.0));
    let b = edit::move_faces_by(&block, &[top], &lift).unwrap();
    valid(&b);
    let v = measure::volume(&b);
    assert!((v - 640.0).abs() < 1e-9, "{v}");
    // A map that stretches is no motion.
    let stretch = Transform {
        r: [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t: Vec3::ZERO,
    };
    assert!(matches!(
        edit::move_faces_by(&block, &[top], &stretch),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn a_conic_segment_is_the_ellipse_arc_it_lies_on() {
    // The quarter of the unit circle from (1, 0) to (0, 1): apex (1, 1), weight cos 45°.
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let s = Segment::conic(
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
        w,
    )
    .unwrap();
    for k in 0..=8 {
        let p = s.at(f64::from(k) / 8.0);
        assert!((p.norm() - 1.0).abs() < 1e-12, "{p:?}");
    }
    assert!((s.start() - Vec2::new(1.0, 0.0)).norm() < 1e-12);
    assert!((s.end() - Vec2::new(0.0, 1.0)).norm() < 1e-12);
    // A slanted one: the rational quadratic's own points satisfy the arc's ellipse, and the arc
    // starts and ends where it does.
    let (a, apex, b, w) = (
        Vec2::new(3.0, -1.0),
        Vec2::new(4.5, 2.0),
        Vec2::new(1.0, 3.5),
        0.83,
    );
    let s = Segment::conic(a, apex, b, w).unwrap();
    assert!((s.start() - a).norm() < 1e-12 && (s.end() - b).norm() < 1e-12);
    let Segment::Ellipse {
        center,
        rotation,
        major,
        minor,
        ..
    } = s
    else {
        panic!("{s:?}")
    };
    for i in 0..=20 {
        let t = f64::from(i) / 20.0;
        let (b0, b1, b2) = ((1.0 - t).powi(2), 2.0 * w * t * (1.0 - t), t * t);
        let d = (a * b0 + apex * b1 + b * b2) * (1.0 / (b0 + b1 + b2)) - center;
        let (c, sn) = (rotation.cos(), rotation.sin());
        let (x, y) = (d.x * c + d.y * sn, -d.x * sn + d.y * c);
        let e = (x / major).powi(2) + (y / minor).powi(2) - 1.0;
        assert!(e.abs() < 1e-12, "{e}");
    }
    // A weight of 0 is no conic.
    let e = Segment::conic(
        Vec2::new(1.0, 0.0),
        Vec2::new(1.0, 1.0),
        Vec2::new(0.0, 1.0),
        0.0,
    );
    assert!(e.is_err());
}

#[test]
fn a_conic_segment_of_weight_one_or_more_is_the_parabola_or_hyperbola_arc() {
    // y = x² from (−1, 1) to (1, 1): its tangents there, of slope ∓2, meet at (0, −1).
    let s = Segment::conic(
        Vec2::new(-1.0, 1.0),
        Vec2::new(0.0, -1.0),
        Vec2::new(1.0, 1.0),
        1.0,
    )
    .unwrap();
    for k in 0..=8 {
        let p = s.at(f64::from(k) / 8.0);
        assert!((p.y - p.x * p.x).abs() < 1e-12, "{p:?}");
    }
    // x² − 2y² = 2 from (r, −k) to (r, k), k = 1/√3 and r = √(8/3): its tangents there meet at
    // (2/r, 0), and its vertex (√2, 0) lies at the weight (r − √2)/(√2 − 2/r) = 2/√3.
    let (k, r) = (1.0 / 3f64.sqrt(), (8.0f64 / 3.0).sqrt());
    let s = Segment::conic(
        Vec2::new(r, -k),
        Vec2::new(2.0 / r, 0.0),
        Vec2::new(r, k),
        2.0 / 3f64.sqrt(),
    )
    .unwrap();
    assert!((s.start() - Vec2::new(r, -k)).norm() < 1e-12);
    assert!((s.end() - Vec2::new(r, k)).norm() < 1e-12);
    assert!((s.at(0.5) - Vec2::new(2f64.sqrt(), 0.0)).norm() < 1e-12);
    for i in 0..=8 {
        let p = s.at(f64::from(i) / 8.0);
        assert!((p.x * p.x - 2.0 * p.y * p.y - 2.0).abs() < 1e-12, "{p:?}");
    }
}

#[test]
fn a_file_of_two_touching_solids_reads_as_the_one_part() {
    // Two blocks face to face, written as a file of two solids, read back as one part of their
    // joint volume in one piece.
    let a = primitive::block(&Frame::WORLD, Vec3::new(4.0, 4.0, 4.0), "a").unwrap();
    let at = Frame::new(Vec3::new(4.0, 0.0, 0.0), Vec3::Z, Vec3::X).unwrap();
    let b = primitive::block(&at, Vec3::new(2.0, 4.0, 4.0), "b").unwrap();
    let mut both = a.clone();
    both.append(&b);
    assert_eq!(both.solid_ids().count(), 2);
    let text = step::write(&both, "two").unwrap();
    let part = step::read_part(text.as_bytes(), "step").unwrap();
    valid(&part);
    assert_eq!(part.solid_ids().count(), 1);
    let v = measure::volume(&part);
    assert!((v - 96.0).abs() < 1e-9, "{v}");
}

#[test]
fn a_mesh_within_a_tolerance_lies_that_close() {
    // A ball of radius 5 meshed within 0.01: the mesh's volume falls short of the ball's by no
    // more than the tolerance times its area.
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "sph").unwrap();
    let m = mesh::tessellate(&ball, &MeshOptions::within(0.01)).unwrap();
    let solid = ball.solid_ids().next().unwrap();
    let r = mesh::report(&ball, &m, solid);
    assert!(r.watertight(), "{r:?}");
    let exact = 4.0 / 3.0 * PI * 125.0;
    assert!(
        (r.volume - exact).abs() <= 0.01 * 4.0 * PI * 25.0,
        "{}",
        r.volume
    );
}

#[test]
fn the_step_writer_says_which_order_it_wrote_the_faces_in() {
    let b = sweep::extrude(
        &Profile::rect(Vec2::ZERO, 4.0, 2.0),
        &Frame::WORLD,
        3.0,
        "blk",
    )
    .unwrap();
    let (text, faces) = step::write_faces(&b, "blk").unwrap();
    let written: Vec<&str> = text
        .split("ADVANCED_FACE('")
        .skip(1)
        .map(|rest| rest.split('\'').next().unwrap())
        .collect();
    let named: Vec<String> = faces
        .iter()
        .map(|&f| b.face(f).names.iter().next().unwrap().to_string())
        .collect();
    assert_eq!(written.len(), 6);
    assert_eq!(written, named);
    assert_eq!(text, step::write(&b, "blk").unwrap());
}

#[test]
fn a_section_is_given_as_the_curves_it_runs_along() {
    // A Ø10 rod cut square to its axis: one outline, every piece of it a circle of radius 5
    // about the axis at the cut's height.
    use springrep::geom::Curve;
    let rod = primitive::cylinder(&Frame::WORLD, 5.0, 8.0, "rod").unwrap();
    let origin = springrep::math::Point3::new(0.0, 0.0, 3.0);
    let outlines = measure::section_curves(&rod, origin, Vec3::Z).unwrap();
    assert_eq!(outlines.len(), 1);
    assert!(!outlines[0].is_empty());
    let mut turned = 0.0;
    for piece in &outlines[0] {
        let Curve::Circle(c) = &piece.curve else {
            panic!("a rod's section is circles, not {}", piece.curve.kind());
        };
        assert!((c.radius - 5.0).abs() < 1e-9 && (c.frame.origin.z - 3.0).abs() < 1e-9);
        turned += (piece.range.1 - piece.range.0).abs();
    }
    assert!((turned - std::f64::consts::TAU).abs() < 1e-9, "{turned}");
}
