use std::f64::consts::PI;

use super::*;
use crate::{
    check,
    math::{Frame, Point3, Vec3},
    measure,
    primitive::{block, cylinder, sphere},
};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Fails unless `a` and `b` are one solid built twice: as many faces under each name, each face
/// of one matched by a face of the other on the same kind of surface with the same area, and the
/// same volume, all within `tol` of their size.
pub(super) fn assert_same_solid(a: &Brep, b: &Brep, tol: f64) {
    assert_eq!(a.name_counts(), b.name_counts());
    let faces = |s: &Brep| {
        let mut out: Vec<(&'static str, f64)> = s
            .face_ids()
            .map(|f| (s.face(f).surface.kind(), measure::face_area(s, f)))
            .collect();
        out.sort_by(|x, y| x.0.cmp(y.0).then(x.1.total_cmp(&y.1)));
        out
    };
    let (fa, fb) = (faces(a), faces(b));
    assert_eq!(fa.len(), fb.len());
    let size = measure::area(a);
    for (x, y) in fa.iter().zip(&fb) {
        assert!(x.0 == y.0 && (x.1 - y.1).abs() <= tol * size, "{x:?} {y:?}");
    }
    assert!(rel(measure::area(a), measure::area(b)) <= tol);
    assert!(rel(measure::volume(a), measure::volume(b)) <= tol);
}

// A ball slid along a line sweeps a capsule: a cylinder of its radius along the slide, with half a
// ball at each end.
#[test]
fn a_ball_slid_along_a_line_is_a_capsule() {
    let (r, d) = (1.5, Vec3::new(3.0, -2.0, 4.0));
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(1.0, 2.0, 3.0)),
        r,
        "ball",
    )
    .unwrap();
    let b = solid(&ball, &Motion::Slide(d), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let l = d.norm();
    let (area, volume) = measure::area_and_volume(&b);
    assert!(
        rel(volume, 4.0 / 3.0 * PI * r.powi(3) + PI * r * r * l) < 1e-9,
        "{volume}"
    );
    assert!(
        rel(area, 4.0 * PI * r * r + 2.0 * PI * r * l) < 1e-9,
        "{area}"
    );
}

// A convex body slid by `d` sweeps its own volume and the prism of its shadow along `d`: for the
// unit cube the shadow along a unit `u` has area |u.x| + |u.y| + |u.z|, so the volume is
// 1 + |d.x| + |d.y| + |d.z|.
#[test]
fn a_cube_slid_aslant_adds_the_prism_of_its_shadow() {
    let d = Vec3::new(2.0, 1.0, -3.0);
    let cube = block(&Frame::WORLD, Vec3::new(1.0, 1.0, 1.0), "cube").unwrap();
    let b = solid(&cube, &Motion::Slide(d), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    assert!(
        rel(measure::volume(&b), 7.0) < 1e-12,
        "{}",
        measure::volume(&b)
    );
}

// A torus slid along its own axis sweeps itself and the prism of its shadow, the annulus between
// radii R − r and R + r: 2π²Rr² + 4πRr·L. It grazes the slide along its top and bottom circles.
#[test]
fn a_torus_slid_along_its_axis_adds_the_prism_of_its_annulus() {
    let (big, small, l) = (3.0, 1.0, 5.0);
    let frame = Frame::new(Vec3::new(1.0, 0.0, 2.0), Vec3::new(1.0, 1.0, 1.0), Vec3::X).unwrap();
    let ring = crate::primitive::torus(&frame, big, small, "ring").unwrap();
    let b = solid(&ring, &Motion::Slide(frame.z * l), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = 2.0 * PI * PI * big * small * small + 4.0 * PI * big * small * l;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// A leaning drum pushed straight down through a level plane: the parts of its sweep below and
// above the plane, each cut off by the plane across its swept rims, make up the whole.
#[test]
fn a_slid_drum_cut_by_a_level_plane_keeps_its_volume() {
    let lean = Frame::new(Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.5, 0.0, 0.8), Vec3::X).unwrap();
    let drum = cylinder(&lean, 1.5, 2.0, "drum").unwrap();
    let b = solid(&drum, &Motion::Slide(Vec3::new(0.0, 0.0, -8.0)), "sw").unwrap();
    let below = block(
        &Frame::WORLD.translated(Vec3::new(-8.0, -8.0, -8.0)),
        Vec3::new(16.0, 16.0, 8.0),
        "below",
    )
    .unwrap();
    let above = block(
        &Frame::WORLD.translated(Vec3::new(-8.0, -8.0, 0.0)),
        Vec3::new(16.0, 16.0, 8.0),
        "above",
    )
    .unwrap();
    let low = crate::boolean::intersection(&b, &below).unwrap();
    let high = crate::boolean::intersection(&b, &above).unwrap();
    let whole = measure::volume(&b);
    assert!(rel(measure::volume(&low) + measure::volume(&high), whole) < 1e-9);
}

// A cylinder of radius r and height h, its axis along z, casts a shadow along a unit u of area
// π r² |u.z| + 2 r h |u × z|: the ellipse of each end and the rectangle of its side between.
#[test]
fn a_cylinder_slid_aslant_adds_the_prism_of_its_shadow() {
    let (r, h, d) = (1.0, 2.0, Vec3::new(3.0, 0.0, 4.0));
    let drum = cylinder(&Frame::WORLD, r, h, "drum").unwrap();
    let b = solid(&drum, &Motion::Slide(d), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let u = d.unit().unwrap();
    let shadow = PI * r * r * u.z.abs() + 2.0 * r * h * u.cross(Vec3::Z).norm();
    let want = PI * r * r * h + d.norm() * shadow;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// A drum of radius 2 along x from 6 to 14 at z = −5 pushed up by 16 through the plate (0, −5, 0)–
// (20, 5, 4) cuts the slot 6 ≤ x ≤ 14, |y| ≤ 2 through it: 800 − 128. Its end discs and the rims
// they slide past sweep within the planes x = 6 and x = 14, so every face of the slot is a plane.
#[test]
fn a_drum_pushed_across_its_axis_slides_its_ends_within_their_planes() {
    let end = Frame::new(Vec3::new(6.0, 0.0, -5.0), Vec3::X, Vec3::Y).unwrap();
    let drum = cylinder(&end, 2.0, 8.0, "drum").unwrap();
    let b = solid(&drum, &Motion::Slide(Vec3::new(0.0, 0.0, 16.0)), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let plate = block(
        &Frame::WORLD.translated(Vec3::new(0.0, -5.0, 0.0)),
        Vec3::new(20.0, 10.0, 4.0),
        "plate",
    )
    .unwrap();
    let cut = crate::boolean::difference(&plate, &b).unwrap();
    let planes = cut
        .face_ids()
        .all(|f| matches!(cut.face(f).surface, crate::geom::Surface::Plane(_)));
    assert!(planes);
    let got = measure::volume(&cut);
    assert!(rel(got, 672.0) < 1e-12, "{got}");
}

/// The turn about the z axis through `angle`, from where the body is drawn.
fn about_z(angle: f64) -> Motion {
    Motion::Turn {
        origin: Vec3::ZERO,
        axis: Vec3::Z,
        from: 0.0,
        to: angle,
    }
}

// Where each circle about the axis meets the body in one arc, a turn through θ adds to the body θ
// times the moment about the axis of its shadow on a half-plane through the axis, ∫ρ dA (Pappus).
// A ball of radius r whose centre is R from the axis casts a disc: πr²R.
#[test]
fn a_ball_turned_about_a_line_adds_a_piece_of_torus() {
    let (r, big, angle) = (1.0, 4.0, 1.2);
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(big, 0.0, 0.5)),
        r,
        "ball",
    )
    .unwrap();
    let b = solid(&ball, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = 4.0 / 3.0 * PI * r.powi(3) + angle * PI * r * r * big;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// The volume of the ball of radius `r` about (`big`, 0, 0) turned by `angle` about the z axis,
// `big` less than `r`. At height z the ball is the disc of radius s = √(r² − z²) about (big, 0),
// and the circle of radius ρ about the axis meets it whole for ρ ≤ s − big, else in an arc
// 2 acos((ρ² + big² − s²)/(2ρ·big)) wide, which the turn widens by θ up to the whole circle: up to
// ρ_c, where the arc is 2π − θ, the circle is whole. So the solid is ∫∫ ρ min(arc + θ, 2π) dρ dz.
fn turned_ball(r: f64, big: f64, angle: f64) -> f64 {
    let slice = |z: f64| {
        let s = (r * r - z * z).sqrt();
        let arc = |rho: f64| {
            let c = (rho * rho + big * big - s * s) / (2.0 * rho * big);
            2.0 * c.clamp(-1.0, 1.0).acos()
        };
        let c = -(angle / 2.0).cos() * big;
        let whole = (s - big)
            .max(0.0)
            .max(c + (c * c - big * big + s * s).max(0.0).sqrt());
        let mut widened = |rho: f64| rho * (arc(rho) + angle);
        // a circle nearer the axis than big − s misses a disc that does not reach it
        let from = whole.max(big - s);
        let partial = crate::numeric::integrate(&mut widened, from, s + big, 1e-13);
        PI * whole * whole + partial
    };
    let mut column = |a: f64| slice(r * a.sin()) * r * a.cos();
    crate::numeric::integrate(&mut column, -PI / 2.0, PI / 2.0, 1e-12)
}

// The axis passes through the ball, so the circles about it that the ball holds whole stay whole,
// and the rest each meet it in one arc the turn widens by θ, the circles just beyond those to the
// whole: a little less than πr²Rθ is added. Beyond the axis the ball at the start and the ball at
// the end pass through each other between the points the axis pierces both, along an arc no edge
// of either reaches: the two hemispheres sewn to the swept torus as they stand cross there and
// count that lens twice.
#[test]
fn a_ball_turned_about_a_line_through_it_adds_a_piece_of_torus() {
    let (r, big, angle) = (1.0, 0.5, 1.0);
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(big, 0.0, 0.0)),
        r,
        "ball",
    )
    .unwrap();
    let b = solid(&ball, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = turned_ball(r, big, angle);
    let twice = 4.0 / 3.0 * PI * r.powi(3) + angle * PI * r * r * big;
    assert!(want < twice - 5e-3, "{want} {twice}");
    let got = measure::volume(&b);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// The ball of radius 2 about (1, 0, 0) turned 120° about the z axis. Near the axis the balls at
// the two ends overlap between the points the axis pierces both, with no edge of either crossing
// the other.
#[test]
fn a_ball_turned_far_about_a_line_through_it_closes_round_the_axis() {
    let angle = 2.0 * PI / 3.0;
    let ball = sphere(&Frame::WORLD.translated(Vec3::X), 2.0, "ball").unwrap();
    let b = solid(&ball, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = turned_ball(2.0, 1.0, angle);
    let got = measure::volume(&b);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// The unit cube from (1, 0, 0) turned about the z axis: its shadow spans radii 1 to √5 at every
// height, a moment of (5 − 1)/2.
#[test]
fn a_cube_turned_about_a_line_beside_it() {
    // Past half a turn the faces the turn carries round lie beyond the axis from where they start.
    for angle in [0.9, 4.0] {
        let cube = block(
            &Frame::WORLD.translated(Vec3::X),
            Vec3::new(1.0, 1.0, 1.0),
            "cube",
        )
        .unwrap();
        let b = solid(&cube, &about_z(angle), "sw").unwrap();
        assert!(check::check(&b).is_empty());
        assert!(
            rel(measure::volume(&b), 1.0 + 2.0 * angle) < 1e-9,
            "{angle} {}",
            measure::volume(&b)
        );
    }
}

// The same cube turned a whole turn fills the ring between radii 1 and √5: 4π.
#[test]
fn a_cube_turned_a_whole_turn_is_a_ring() {
    let cube = block(
        &Frame::WORLD.translated(Vec3::X),
        Vec3::new(1.0, 1.0, 1.0),
        "cube",
    )
    .unwrap();
    let b = solid(&cube, &about_z(2.0 * PI), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    assert!(
        rel(measure::volume(&b), 4.0 * PI) < 1e-9,
        "{}",
        measure::volume(&b)
    );
}

// A whole turn's second half is the first turned half a turn: the same solid as when the second
// half is swept as the first is, for a turn either way, started at any angle.
#[test]
fn a_whole_turn_built_from_one_half_is_the_one_built_from_both() {
    let cube = block(
        &Frame::WORLD.translated(Vec3::X),
        Vec3::new(1.0, 1.0, 1.0),
        "cube",
    )
    .unwrap();
    let ball = sphere(&Frame::WORLD.translated(Vec3::new(3.0, 0.0, 1.0)), 1.0, "b").unwrap();
    let drum = cylinder(
        &Frame::new(Vec3::new(0.0, 5.0, 0.0), Vec3::Y, Vec3::X).unwrap(),
        1.0,
        0.5,
        "drum",
    )
    .unwrap();
    let turn = |from: f64, to: f64| Motion::Turn {
        origin: Point3::new(0.0, 0.0, 0.5),
        axis: Vec3::Z,
        from,
        to,
    };
    for (body, motion) in [
        (&cube, turn(0.0, 2.0 * PI)),
        (&cube, turn(0.7, 0.7 - 2.0 * PI)),
        (&ball, turn(-1.0, 2.0 * PI - 1.0)),
        (&drum, turn(0.3, 0.3 + 2.0 * PI)),
    ] {
        let copied = solid(body, &motion, "sw").unwrap();
        SWEEP_BOTH_HALVES.set(true);
        let swept = solid(body, &motion, "sw");
        SWEEP_BOTH_HALVES.set(false);
        assert!(check::check(&copied).is_empty());
        assert_same_solid(&copied, &swept.unwrap(), 1e-10);
    }
}

// A drum of radius 1 and length ½ along y, its axis 5 from the z axis, turned about it: at height z
// its shadow runs from 5 − s to √((5 + s)² + ¼), s = √(1 − z²), a moment of ∫(20s + ¼)/2 dz =
// 5π + ¼. Its side grazes the turn along curves that are neither lines nor circles.
#[test]
fn a_drum_turned_about_a_line_square_to_its_axis() {
    let angle = 0.7;
    let frame = Frame::new(Vec3::new(5.0, 0.0, 0.0), Vec3::Y, Vec3::X).unwrap();
    let drum = cylinder(&frame, 1.0, 0.5, "drum").unwrap();
    let b = solid(&drum, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = PI * 0.5 + angle * (5.0 * PI + 0.25);
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// A disc of radius 2 and thickness ½ along y, whose rim passes through the z axis, turned a
// quarter turn about it: the turn holds that point of the rim still. At height z its shadow runs
// from 2 − s to √((2 + s)² + ¼), s = √(4 − z²), a moment of ∫(8s + ¼)/2 dz = 8π + ½.
#[test]
fn a_disc_turned_about_a_line_through_its_rim() {
    let angle = PI / 2.0;
    let frame = Frame::new(Vec3::new(2.0, 0.0, 0.0), Vec3::Y, Vec3::X).unwrap();
    let disc = cylinder(&frame, 2.0, 0.5, "disc").unwrap();
    let b = solid(&disc, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = 2.0 * PI + angle * (8.0 * PI + 0.5);
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// A rod of radius r along x from 0 to L, turned through θ about the z axis, which passes through
// the centre of its near end. Its far rim's axis meets the turn's, so the rim sweeps a sphere about
// that point. At height z it is the box of the next test with a = 0, b = L and c = w = √(r² − z²),
// whose swept area 2wL + θ(L² + w²)/2 + atan(w/L)(L² + w²) − Lw is integrated over z.
#[test]
fn a_rod_turned_about_a_line_through_its_end_sweeps_its_rim_on_a_sphere() {
    let (r, l, angle) = (0.5, 5.0, 2.0 * PI / 3.0);
    let rod = cylinder(
        &Frame::new(Vec3::ZERO, Vec3::X, Vec3::Y).unwrap(),
        r,
        l,
        "rod",
    )
    .unwrap();
    let swept = solid(&rod, &about_z(angle), "sw").unwrap();
    assert!(check::check(&swept).is_empty());
    let area = |w: f64| {
        let r2 = l * l + w * w;
        2.0 * w * l + angle * r2 / 2.0 + (w / l).atan() * r2 - l * w
    };
    let mut slice = |a: f64| area(r * a.cos()) * r * a.cos();
    let want = crate::numeric::integrate(&mut slice, -PI / 2.0, PI / 2.0, 1e-13);
    let got = measure::volume(&swept);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// The box a ≤ x ≤ b, |y| ≤ c, 0 ≤ z ≤ h turned through θ about the z axis. Its faces square to x
// each graze the turn along y = 0 and cross themselves turned, so it is built in parts. At radius ρ
// it spans one arc out to √(b² + c²), which the turn widens by θ, except past b, where it spans two
// arcs ±(acos(b/ρ), asin(c/ρ)) that the turn joins into one, widening it by θ + 2 acos(b/ρ). With
// ∫ρ acos(b/ρ) dρ from b to √(b² + c²) = (φ(b² + c²) − bc)/2, φ = atan(c/b), the area swept is
// 2c(b − a) + θ(b² + c² − a²)/2 + φ(b² + c²) − bc.
#[test]
fn a_box_turned_about_a_line_beside_it_is_built_in_parts() {
    let (a, b, c, h, angle) = (10.0, 13.0, 2.0, 3.0, PI / 3.0);
    let corner = Frame::WORLD.translated(Vec3::new(a, -c, 0.0));
    let cube = block(&corner, Vec3::new(b - a, 2.0 * c, h), "box").unwrap();
    let swept = solid(&cube, &about_z(angle), "sw").unwrap();
    assert!(check::check(&swept).is_empty());
    let phi = (c / b).atan();
    let r2 = b * b + c * c;
    let want = h * (2.0 * c * (b - a) + angle * (r2 - a * a) / 2.0 + phi * r2 - b * c);
    let got = measure::volume(&swept);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// The same box's whole product passes through itself at more than one pair of faces. It is
// refused at the first pair found, which is all the sweep in parts needs of it.
#[test]
fn a_product_that_passes_through_itself_is_refused_at_the_first_pair_found() {
    let (a, b, c, h, angle) = (10.0, 13.0, 2.0, 3.0, PI / 3.0);
    let corner = Frame::WORLD.translated(Vec3::new(a, -c, 0.0));
    let cube = block(&corner, Vec3::new(b - a, 2.0 * c, h), "box").unwrap();
    let motion = about_z(angle);
    let (steady, twist) = (
        sheet::Steady::of(&motion).unwrap(),
        motion.steady().unwrap(),
    );
    let cut = cut::cut(&cube, &move |p| twist.at(p), steady.held()).unwrap();
    match product::product(&cut, &steady, "sw", None, None) {
        Err(Error::InvalidResult(issues)) => assert_eq!(issues.len(), 1, "{issues:?}"),
        other => panic!("{:?}", other.map(|b| b.face_count())),
    }
}

/// The right-hand screw about the z axis that rises `lead` a turn.
fn screw_z(lead: f64) -> crate::sweep::Helix {
    crate::sweep::Helix {
        origin: Vec3::ZERO,
        axis: Vec3::Z,
        rise: lead / (2.0 * PI),
        sign: 1.0,
    }
}

// A ball screwed along a helix no tighter than itself sweeps the tube about its centre's helix,
// π r² times the helix's length θ√(R² + c²), and a ball at each end.
#[test]
fn a_ball_screwed_along_a_helix_is_a_tube() {
    let (r, big, lead, angle) = (1.0, 5.0, 6.0, 4.0);
    let ball = sphere(
        &Frame::WORLD.translated(Vec3::new(big, 0.0, 0.0)),
        r,
        "ball",
    )
    .unwrap();
    let screw = screw_z(lead);
    let motion = Motion::Screw {
        screw,
        from: 0.0,
        to: angle,
    };
    let b = solid(&ball, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let length = angle * (big * big + screw.rise * screw.rise).sqrt();
    let want = 4.0 / 3.0 * PI * r.powi(3) + PI * r * r * length;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

// The bar 4 ≤ x ≤ 12, |y| ≤ 8, |z| ≤ 1 screwed one turn of lead 6 about the z axis, within the
// cylinder of radius 5. A point at radius ρ is in the bar at turn t while its angle from the bar's
// is within a = acos(4/ρ), so at each angle the turn covers heights 2 + 2ca long, c the rise a
// radian, or two such runs 2 + c(a ± φ) long within a of where the bar starts. Over the turn that is
// 4π + 4πca + 4a a unit of ρ dρ: 18π from radius 4 to 5, and with ∫ρ acos(4/ρ) dρ = 8(25φ₁/16 −
// 3/4), φ₁ = acos(4/5), (4πc + 4)·8(25φ₁/16 − 3/4) = 200φ₁ − 96 more.
#[test]
fn a_bar_screwed_a_turn_cuts_a_square_thread() {
    let bar = block(
        &Frame::WORLD.translated(Vec3::new(4.0, -8.0, -1.0)),
        Vec3::new(8.0, 16.0, 2.0),
        "bar",
    )
    .unwrap();
    let motion = Motion::Screw {
        screw: screw_z(6.0),
        from: 0.0,
        to: 2.0 * PI,
    };
    let b = solid(&bar, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let rod = cylinder(
        &Frame::WORLD.translated(Vec3::new(0.0, 0.0, -2.0)),
        5.0,
        10.0,
        "rod",
    )
    .unwrap();
    let thread = crate::boolean::intersection(&b, &rod).unwrap();
    let want = 18.0 * PI + 200.0 * 0.8f64.acos() - 96.0;
    let got = measure::volume(&thread);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// Two slides at once are the slide by their sum: the unit cube slid by (2, 1, 0) adds the prism
// of its shadow along the slide, 2 + 1. Built as a motion whose velocity may change.
#[test]
fn a_cube_slid_two_ways_at_once_adds_the_prism_of_its_shadow() {
    let cube = block(&Frame::WORLD, Vec3::new(1.0, 1.0, 1.0), "cube").unwrap();
    let motion = Motion::Then(
        Box::new(Motion::Slide(Vec3::new(2.0, 0.0, 0.0))),
        Box::new(Motion::Slide(Vec3::new(0.0, 1.0, 0.0))),
    );
    let b = solid(&cube, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    assert!(
        rel(measure::volume(&b), 4.0) < 1e-9,
        "{}",
        measure::volume(&b)
    );
}

/// The volume a cube of side 2 about the origin fills as it is slid without turning along the arc
/// `(−r sin a, r (1 − cos a), 0)` for `a` from `a0` to `a1` (within a quarter turn of 0): a section
/// at `x` meets the squares whose `sin a` lies within `(−x ∓ 1) / r`, and its height runs from the
/// lowest of their bottoms to the highest of their tops, as the height of the arc grows with `|a|`.
fn arc_sweep_volume(r: f64, a0: f64, a1: f64) -> f64 {
    let height = |a: f64| r * (1.0 - a.cos());
    let (x0, x1) = (-r * a1.sin() - 1.0, -r * a0.sin() + 1.0);
    let n = 200_000;
    let dx = (x1 - x0) / n as f64;
    let section = |x: f64| {
        let lo = ((-x - 1.0) / r).max(a0.sin()).asin();
        let hi = ((1.0 - x) / r).min(a1.sin()).asin();
        let near = if lo <= 0.0 && hi >= 0.0 {
            0.0
        } else {
            lo.abs().min(hi.abs())
        };
        let far = lo.abs().max(hi.abs());
        (height(far) + 1.0) - (height(near) - 1.0)
    };
    // Midpoints, the section's kinks at the ends of the arc's own run of x aside, which a rule
    // this fine settles far inside the test's tolerance.
    (0..n)
        .map(|i| section(x0 + (i as f64 + 0.5) * dx))
        .sum::<f64>()
        * dx
        * 2.0
}

/// Turned by `a` about the z axis and then by `−a` about the line through `(0, r, 0)` along it, a
/// body is slid without turning along the arc of [`arc_sweep_volume`].
fn along_arc(r: f64, a0: f64, a1: f64) -> Motion {
    Motion::Then(
        Box::new(Motion::Turn {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            from: a0,
            to: a1,
        }),
        Box::new(Motion::Turn {
            origin: Vec3::new(0.0, r, 0.0),
            axis: Vec3::Z,
            from: -a0,
            to: -a1,
        }),
    )
}

// Slid along an arc from where it heads along −x, a cube's faces square to y move within their own
// planes where the motion starts. For the half arc the section's height is the square's 2 plus the
// arc's rise, and the volume is 2 (4 + 2r (sin a + 1 − cos a)).
#[test]
fn a_cube_slid_along_an_arc_from_where_two_faces_stand_still() {
    let cube = block(
        &Frame::WORLD.translated(Vec3::new(-1.0, -1.0, -1.0)),
        Vec3::new(2.0, 2.0, 2.0),
        "cube",
    )
    .unwrap();
    let (r, a) = (10.0, PI / 3.0);
    let b = solid(&cube, &along_arc(r, 0.0, a), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let closed = 2.0 * (4.0 + 2.0 * r * (a.sin() + 1.0 - a.cos()));
    assert!(rel(arc_sweep_volume(r, 0.0, a), closed) < 1e-9);
    assert!(
        rel(measure::volume(&b), closed) < 1e-9,
        "{}",
        measure::volume(&b)
    );
    // Across that instant the faces square to y turn from moving in to moving out.
    let b = solid(&cube, &along_arc(r, -a, a), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let want = arc_sweep_volume(r, -a, a);
    assert!(
        rel(measure::volume(&b), want) < 1e-7,
        "{} {want}",
        measure::volume(&b)
    );
}

// A frame riding a circle about the z axis turns about the axis as it goes, so a cube it carries
// sweeps what the cube turned through the same angle does. The unit cube from (1, ½, 0) casts a
// square on a half-plane through the axis from radius √(5/4) to radius 5/2: a moment of 5/2.
#[test]
fn a_cube_carried_round_a_circle_is_a_cube_turned() {
    let angle = 0.9;
    let cube = block(
        &Frame::WORLD.translated(Vec3::new(1.0, 0.5, 0.0)),
        Vec3::new(1.0, 1.0, 1.0),
        "cube",
    )
    .unwrap();
    let circle = crate::geom::Curve::Circle(crate::geom::Circle {
        frame: Frame::WORLD,
        radius: 3.0,
    });
    let motion = Motion::Along {
        path: circle,
        range: (0.0, angle),
        up: Vec3::Z,
    };
    let b = solid(&cube, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let got = measure::volume(&b);
    assert!(rel(got, 1.0 + 2.5 * angle) < 1e-7, "{got}");
}

// An upright drum of radius 1 whose axis is 3 from the z axis, carried round a circle about it:
// the drum turned, its disc's shadow on a half-plane through the axis spanning radii 2 to 4, a
// moment of 6. It grazes the turn along whole rulings, found at each height by its rate's closed
// form round the drum.
#[test]
fn a_drum_carried_round_a_circle_is_a_drum_turned() {
    let angle = 0.8;
    // Its seam away from the rulings it grazes along, which face the axis and face away.
    let frame = Frame::new(Vec3::new(3.0, 0.0, 0.0), Vec3::Z, Vec3::Y).unwrap();
    let drum = cylinder(&frame, 1.0, 1.0, "drum").unwrap();
    let circle = crate::geom::Curve::Circle(crate::geom::Circle {
        frame: Frame::WORLD,
        radius: 3.0,
    });
    let motion = Motion::Along {
        path: circle,
        range: (0.0, angle),
        up: Vec3::Z,
    };
    let b = solid(&drum, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let got = measure::volume(&b);
    assert!(rel(got, PI + 6.0 * angle) < 1e-7, "{got}");
}

// Slid along x in its own plane, a torus about z grazes along its whole meridians at u = ±90°,
// where the rate `cos u cos v` vanishes for every v. Under a motion whose velocity may change that
// is refused, not built wrong.
#[test]
fn a_torus_grazing_along_a_whole_meridian_is_refused() {
    let ring = crate::primitive::torus(&Frame::WORLD, 2.0, 0.5, "ring").unwrap();
    let motion = Motion::Then(
        Box::new(Motion::Slide(Vec3::new(1.0, 0.0, 0.0))),
        Box::new(Motion::Slide(Vec3::new(2.0, 0.0, 0.0))),
    );
    assert!(matches!(
        solid(&ring, &motion, "sw"),
        Err(crate::Error::Unsupported(_))
    ));
}

// A torus turned a quarter as fast about a vertical line as the frame it is seen from turns about
// a level one. The grazing curve where the span starts runs round the tube across a sheet whose map
// nearly drops rank, where projecting its points lands up to 2e-2 mm off the sheet; its pcurve is
// the spline through its points in the sheet's own parameters, and the solid is valid.
#[test]
fn a_tumbling_torus_keeps_its_pcurves_on_its_sheets() {
    let ring = crate::primitive::torus(
        &Frame::WORLD.translated(Vec3::new(3.0, 0.0, 2.0)),
        1.0,
        0.5,
        "ring",
    )
    .unwrap();
    let (from, to) = (-75f64.to_radians(), 75f64.to_radians());
    let h = 0.5f64.sqrt() * 45f64.to_radians().cos();
    let motion = Motion::Then(
        Box::new(Motion::Turn {
            origin: Vec3::new(2.0, 0.0, 0.0),
            axis: Vec3::Z,
            from: 0.25 * from,
            to: 0.25 * to,
        }),
        Box::new(Motion::Turn {
            origin: Vec3::new(0.0, h, h),
            axis: Vec3::X,
            from: -from,
            to: -to,
        }),
    );
    let b = solid(&ring, &motion, "sw").unwrap();
    let issues = check::check(&b);
    assert!(issues.is_empty(), "{issues:?}");
}

// A rod of radius r along x from −m to l crosses the z axis, which pierces its side where its
// grazing curves pass. At height z its two arms either side of the axis are rods turned about a
// line through an end, as above, with w = √(r² − z²), each sweeping 2wL + θ(L² + w²)/2 +
// atan(w/L)(L² + w²) − Lw for its length L. Their
// sweeps overlap in two sectors θ wide that start square to the rod, where each arm reaches w/sin
// of its angle from the rod: they cross halfway, at (π − θ)/2 from the rod, and the nearer one
// bounds the overlap, w²∫csc² = w² cot((π − θ)/2) in each sector.
#[test]
fn a_rod_across_the_axis_turned_partway() {
    let (r, m, l, angle) = (0.5, 3.0, 5.0, 2.0 * PI / 3.0);
    let end = Frame::new(Vec3::new(-m, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap();
    let rod = cylinder(&end, r, m + l, "rod").unwrap();
    let b = solid(&rod, &about_z(angle), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let arm = |l: f64, w: f64| {
        let r2 = l * l + w * w;
        2.0 * w * l + angle * r2 / 2.0 + (w / l).atan() * r2 - l * w
    };
    let overlap = |w: f64| 2.0 * w * w / ((PI - angle) / 2.0).tan();
    let mut slice = |a: f64| {
        let w = r * a.cos();
        (arm(l, w) + arm(m, w) - overlap(w)) * w
    };
    let want = crate::numeric::integrate(&mut slice, -PI / 2.0, PI / 2.0, 1e-13);
    let got = measure::volume(&b);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// A V-wheel leaning 20°, its rim the triangle (7.5, ±2.5 tan 30°), (10, 0) about its axis, turned a
// whole turn about the z axis, which passes outside it. Its inner cylinder grazes the turn along a
// curve that crosses the cylinder's seam where the turn runs along the seam, so the seam's swept
// face leaves the cylinder there at no angle and comes back into it along a second curve through
// that point; the sweep is built in parts, whose union must find that curve. A turn keeps each
// point's distance from the axis, so the rod of radius 5 about it loses to the whole wheel's sweep
// what it loses to the sweep of the wheel's part within 5.5 of the axis, which grazes nowhere near.
#[test]
fn a_leaning_v_wheel_turned_a_whole_turn_cuts_a_rod_as_its_near_part_does() {
    let flank = 2.5 * 30f64.to_radians().tan();
    let points =
        [(7.5, -flank), (10.0, 0.0), (7.5, flank)].map(|(x, y)| crate::math::Vec2::new(x, y));
    let rim = crate::profile::Profile::polygon(&points, &["rim"]);
    let (s, c) = 20f64.to_radians().sin_cos();
    let (center, axis, across) = (Vec3::new(14.0, 0.0, 10.0), Vec3::new(0.0, s, c), -Vec3::X);
    let sketch = Frame::new(center, across.cross(axis), across).unwrap();
    let wheel = crate::sweep::revolve(&rim, &sketch, center, axis, 2.0 * PI, "wheel").unwrap();
    let rod = cylinder(&Frame::WORLD, 5.0, 20.0, "rod").unwrap();
    let low = Frame::WORLD.translated(Vec3::new(0.0, 0.0, -10.0));
    let reach = cylinder(&low, 5.5, 40.0, "reach").unwrap();
    let near = crate::boolean::intersection(&wheel, &reach).unwrap();
    let swept = solid(&wheel, &about_z(2.0 * PI), "sw").unwrap();
    assert!(check::check(&swept).is_empty());
    let got = measure::volume(&crate::boolean::difference(&rod, &swept).unwrap());
    let part = solid(&near, &about_z(2.0 * PI), "sw").unwrap();
    let want = measure::volume(&crate::boolean::difference(&rod, &part).unwrap());
    assert!(rel(got, want) < 1e-9, "{got} {want}");
}

// A drum of radius 1 and height 1 about the upright line through (10, 0), screwed about the z axis
// rising c = ½/(2π) a radian from −Θ to Θ, Θ = 140°. Along each helix of the screw at radius ρ the
// drum spans angles within α(ρ) of its own, ∫ρα dρ = π/2 (half its area), at heights a band 1 +
// 2cα wide; the screw widens each run by 2Θ, so it adds 2Θ(20 + cπ). Every run covers the angles
// 0 to 60°, so the ring sector 8 ≤ ρ ≤ 12, |z| ≤ 1 over those angles, 80π/3, loses (π/3)(20 + cπ).
// The sector's face at angle 0 cuts the face the half of the drum's rim that stands out of the
// band sweeps along its own parameter line, ending on the row where the screw runs along the rim's
// end.
#[test]
fn a_drum_screwed_across_a_ring_sector_cuts_a_groove() {
    let drum = cylinder(
        &Frame::WORLD.translated(Vec3::new(10.0, 0.0, -0.5)),
        1.0,
        1.0,
        "drum",
    )
    .unwrap();
    let theta = 140f64.to_radians();
    let motion = Motion::Screw {
        screw: screw_z(0.5),
        from: -theta,
        to: theta,
    };
    let b = solid(&drum, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let c = 0.5 / (2.0 * PI);
    let got = measure::volume(&b);
    let want = PI + 2.0 * theta * (20.0 + c * PI);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
    let rect = crate::profile::Profile::rect(crate::math::Vec2::new(10.0, 0.0), 4.0, 2.0);
    let axial = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let sector =
        crate::sweep::revolve(&rect, &axial, Vec3::ZERO, Vec3::Z, PI / 3.0, "block").unwrap();
    let cut = crate::boolean::difference(&sector, &b).unwrap();
    let got = measure::volume(&cut);
    let want = PI / 3.0 * (60.0 - c * PI);
    assert!(rel(got, want) < 1e-9, "{got} {want}");
    // The groove's corners on the sector's two ends, where the drum's edges at radii 9 and 11
    // cross them, its bottom and top at z = ∓0.5 risen by c per radian. Each lies on an exact
    // line of the drum's side and the end's plane, so it is placed far closer than the tolerance;
    // placed from the curve fitted to the drum's swept end and the plane, it can be off by nearly
    // the tolerance, and which of the two is taken then turns on rounding.
    for phi in [0.0, PI / 3.0] {
        for r in [9.0, 11.0] {
            for z in [-0.5, 0.5] {
                let corner = Vec3::new(r * phi.cos(), r * phi.sin(), z + c * phi);
                let near = cut
                    .vertex_ids()
                    .map(|v| cut.vertex(v).point.dist(corner))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    near < 1e-9,
                    "{corner:?} is {near:e} from the nearest vertex"
                );
            }
        }
    }
}

// A line through the turn's axis sweeps a cone with its apex on the axis, a nappe each side. The
// sheet of a piece either side of the apex holds that piece's points turned through s, R(s)·L(t),
// at its parameters, and they lie at distance 0 from its surface by the cone's closed form.
#[test]
fn a_line_through_the_axis_sweeps_each_piece_on_its_own_nappe() {
    let dir = Vec3::new(2.0, 0.0, 3.0).unit().unwrap();
    let line = crate::geom::Curve::Line(crate::geom::Line {
        origin: Vec3::new(-2.0, 0.0, -1.0),
        dir,
    });
    // It meets the axis at t = √13, at (0, 0, 2).
    let len = 2.0 * 13f64.sqrt();
    let turn = sheet::Steady::of(&about_z(1.0)).unwrap();
    for piece in [(0.1 * len, 0.4 * len), (0.6 * len, 0.9 * len)] {
        let sh = turn.sheet(&line, (0.0, len), piece).unwrap();
        for s in [0.0, 0.4, 1.0] {
            for f in [0.0, 0.5, 1.0] {
                let t = piece.0 + (piece.1 - piece.0) * f;
                let want =
                    crate::math::Transform::rotation(Vec3::ZERO, Vec3::Z, s).point(line.eval(t));
                let uv = sh.uv(s, t);
                let got = sh.surface.eval(uv.x, uv.y);
                assert!(
                    got.dist(want) < 1e-12,
                    "{piece:?} {s} {t}: {got:?} {want:?}"
                );
                let off = crate::boolean::csi::signed_distance(&sh.surface, want);
                assert!(off.abs() < 1e-12, "{piece:?} {s} {t}: {off}");
            }
        }
    }
}

/// The volume a block sweeps turned through `angle` about the z axis, integrated over height and
/// radius: at each, the arcs of the circle inside the block, each widened by the turn. Each
/// integral is cut where the arcs change (a corner, a side the circle touches), and its points
/// crowd toward the cuts, where the arcs grow as square roots.
fn turned_block_volume(frame: &Frame, size: Vec3, angle: f64) -> f64 {
    let sides: Vec<(Vec3, f64)> = [(frame.x, size.x), (frame.y, size.y), (frame.z, size.z)]
        .into_iter()
        .flat_map(|(axis, s)| {
            let d = axis.dot(frame.origin);
            [(axis, d), (axis, d + s)]
        })
        .collect();
    let inside = |p: Vec3| {
        let l = frame.to_local(p);
        (0.0..=size.x).contains(&l.x)
            && (0.0..=size.y).contains(&l.y)
            && (0.0..=size.z).contains(&l.z)
    };
    // ∫f between the cuts by five-point Gauss panels in u, where x = mid + half·(3u − u³)/2.
    let integral = |cuts: &mut Vec<f64>, panels: usize, f: &dyn Fn(f64) -> f64| {
        const GAUSS: [(f64, f64); 5] = [
            (0.0, 0.568_888_888_888_888_9),
            (-0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
            (0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
            (-0.906_179_845_938_664, 0.236_926_885_056_189_1),
            (0.906_179_845_938_664, 0.236_926_885_056_189_1),
        ];
        cuts.sort_by(f64::total_cmp);
        let h = 1.0 / panels as f64;
        let mut sum = 0.0;
        for w in cuts.windows(2) {
            let (mid, half) = ((w[0] + w[1]) / 2.0, (w[1] - w[0]) / 2.0);
            for k in 0..panels {
                for (node, weight) in GAUSS {
                    let u = -1.0 + (2.0 * k as f64 + 1.0 + node) * h;
                    let x = mid + half * (3.0 * u - u * u * u) / 2.0;
                    sum += f(x) * half * 1.5 * (1.0 - u * u) * weight * h;
                }
            }
        }
        sum
    };
    let covered = |r: f64, z: f64| {
        // The circle crosses a side's plane where a cos φ + b sin φ = c.
        let mut cuts = vec![-PI, PI];
        for &(axis, d) in &sides {
            let c = d - axis.z * z;
            let (a, b) = (axis.x * r, axis.y * r);
            let m = a.hypot(b);
            if c.abs() < m {
                let (mid, half) = (b.atan2(a), (c / m).acos());
                for phi in [mid - half, mid + half] {
                    cuts.push((phi + PI).rem_euclid(2.0 * PI) - PI);
                }
            }
        }
        cuts.sort_by(f64::total_cmp);
        let (mut covered, mut reach) = (0.0, f64::MIN);
        for w in cuts.windows(2) {
            let mid = (w[0] + w[1]) / 2.0;
            if inside(Vec3::new(r * mid.cos(), r * mid.sin(), z)) {
                covered += w[1] + angle - w[0].max(reach);
                reach = w[1] + angle;
            }
        }
        covered.min(2.0 * PI)
    };
    let area = |z: f64| {
        let mut radii = vec![0.0];
        for (i, &(p, dp)) in sides.iter().enumerate() {
            let (cp, lp) = (dp - p.z * z, p.x.hypot(p.y));
            if lp > 1e-12 {
                radii.push(cp.abs() / lp);
            }
            for &(q, dq) in &sides[..i] {
                let (cq, det) = (dq - q.z * z, p.x * q.y - p.y * q.x);
                if det.abs() > 1e-12 {
                    radii.push(((cp * q.y - cq * p.y) / det).hypot((p.x * cq - q.x * cp) / det));
                }
            }
        }
        integral(&mut radii, 6, &|r| r * covered(r, z))
    };
    let mut heights: Vec<f64> = (0..8)
        .map(|i| {
            let on = |bit: usize, s: f64| if i >> bit & 1 == 1 { s } else { 0.0 };
            let corner =
                frame.x * on(0, size.x) + frame.y * on(1, size.y) + frame.z * on(2, size.z);
            frame.origin.z + corner.z
        })
        .collect();
    integral(&mut heights, 40, &area)
}

// A block that leans, no side of it square to a line through the axis, turned about the z axis:
// it is built in parts, and a part's sheet meets a corner the parts before it left on an end.
#[test]
fn a_leaning_block_turned_about_a_line_beside_it() {
    let size = Vec3::new(2.0, 1.0, 3.0);
    for lean in [Vec3::new(0.2, 0.4, 1.0), Vec3::new(0.4, 0.2, 1.0)] {
        let frame = Frame::new(Vec3::new(10.0, -2.0, 0.0), lean, Vec3::X).unwrap();
        let leaning = block(&frame, size, "block").unwrap();
        for angle in [0.5, 2.2] {
            let b = solid(&leaning, &about_z(angle), "sw").unwrap();
            assert!(check::check(&b).is_empty(), "{lean:?} {angle}");
            let (got, want) = (
                measure::volume(&b),
                turned_block_volume(&frame, size, angle),
            );
            assert!(rel(got, want) < 1e-7, "{lean:?} {angle} {got} {want}");
        }
    }
}

// The same block slid adds the prism of its shadow, the sum over its three pairs of sides.
#[test]
fn a_leaning_block_slid_adds_the_prism_of_its_shadow() {
    let (size, d) = (Vec3::new(2.0, 1.0, 3.0), Vec3::new(3.0, 4.0, 1.0));
    let lean = Vec3::new(0.2, 0.4, 1.0);
    let frame = Frame::new(Vec3::new(10.0, -2.0, 0.0), lean, Vec3::X).unwrap();
    let b = solid(
        &block(&frame, size, "block").unwrap(),
        &Motion::Slide(d),
        "sw",
    )
    .unwrap();
    assert!(check::check(&b).is_empty());
    let (x, y, z) = (frame.x * size.x, frame.y * size.y, frame.z * size.z);
    let shadow = d.dot(x.cross(y)).abs() + d.dot(y.cross(z)).abs() + d.dot(z.cross(x)).abs();
    let got = measure::volume(&b);
    assert!(rel(got, size.x * size.y * size.z + shadow) < 1e-12, "{got}");
}

// A bar leaning across a turn, or lying across a screw's axis, sweeps its edges into spline
// sheets, which meet each other and the sweep's cones and cylinders along the motion's orbits.
#[test]
fn a_sweep_built_on_orbits_is_the_one_built_on_traces() {
    use crate::boolean::ssi::{TEST_LOCK, TRACE_ONLY};
    use std::sync::atomic::Ordering;
    let _held = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (s, c) = 20f64.to_radians().sin_cos();
    let (v, n) = (Vec3::new(0.0, c, -s), Vec3::new(0.0, s, c));
    let leaning = Frame::new(Vec3::X * 10.0 - v * 2.0 - n * 4.0, n, Vec3::X).unwrap();
    let turned = (
        block(&leaning, Vec3::new(3.0, 4.0, 8.0), "bar").unwrap(),
        Motion::Turn {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            from: -1.0,
            to: 1.0,
        },
    );
    let across = Frame::WORLD.translated(Vec3::new(4.0, -1.5, -1.0));
    let screwed = (
        block(&across, Vec3::new(8.0, 3.0, 2.0), "bar").unwrap(),
        Motion::Screw {
            screw: screw_z(6.0),
            from: -1.0,
            to: 3.0,
        },
    );
    for (bar, motion) in [turned, screwed] {
        let built = solid(&bar, &motion, "sw").unwrap();
        assert!(check::check(&built).is_empty());
        let swept = |f| match &built.face(f).surface {
            crate::geom::Surface::Nurbs(n) => n.swept().is_some(),
            _ => false,
        };
        assert!(built.face_ids().any(swept));
        TRACE_ONLY.store(true, Ordering::Relaxed);
        let traced = solid(&bar, &motion, "sw");
        TRACE_ONLY.store(false, Ordering::Relaxed);
        assert_same_solid(&built, &traced.unwrap(), 1e-7);
    }
}

/// The volume a second that the frustum about `frame`'s z axis, of radius `bottom` at its origin
/// and `top` a `height` up, carries outward under the velocity `field`: the flow out through its
/// surface where that is outward, by the midpoint rule. A steady motion that never brings the body
/// back over where it was sweeps the body and this for as long as it runs.
fn frustum_outflow(
    frame: &Frame,
    (bottom, top, height): (f64, f64, f64),
    field: &dyn Fn(Vec3) -> Vec3,
) -> f64 {
    let n = 1200;
    let mid = |k: usize| (k as f64 + 0.5) / n as f64;
    let at = |rho: f64, u: f64, z: f64| {
        frame.origin + (frame.x * u.cos() + frame.y * u.sin()) * rho + frame.z * z
    };
    let slope = (bottom - top) / height;
    let cell = 2.0 * PI / (n * n) as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let u = 2.0 * PI * mid(i);
        let out = frame.x * u.cos() + frame.y * u.sin() + frame.z * slope;
        for j in 0..n {
            let z = height * mid(j);
            let rho = bottom - slope * z;
            sum += field(at(rho, u, z)).dot(out).max(0.0) * rho * height * cell;
            for (radius, z, out) in [(bottom, 0.0, -frame.z), (top, height, frame.z)] {
                let rho = radius * mid(j);
                sum += field(at(rho, u, z)).dot(out).max(0.0) * rho * radius * cell;
            }
        }
    }
    sum
}

// Far from a screw's axis a leaning drum or cone grazes nearly along two of its rulings: each
// curve crosses the side within a thousandth of a radian of `u`, from rim to rim or to the apex.
#[test]
fn a_leaning_drum_or_cone_screwed_far_from_the_axis_is_cut_along_its_steep_curves() {
    let frame = Frame::new(
        Vec3::new(10.0, -2.0, 0.0),
        Vec3::new(0.3, 0.0, 1.0),
        Vec3::X,
    )
    .unwrap();
    for (bottom, top, height, lead) in [(1.0, 1.0, 3.0, 40.0), (1.0, 0.0, 2.0, 10.0)] {
        let body = crate::primitive::cone(&frame, bottom, top, height, "body").unwrap();
        let motion = Motion::Screw {
            screw: screw_z(lead),
            from: 0.0,
            to: 1.0,
        };
        let b = solid(&body, &motion, "sw").unwrap();
        assert!(check::check(&b).is_empty());
        let field = |p: Vec3| Vec3::Z.cross(p) + Vec3::Z * (lead / (2.0 * PI));
        let want = measure::volume(&body) + frustum_outflow(&frame, (bottom, top, height), &field);
        let got = measure::volume(&b);
        assert!(rel(got, want) < 1e-5, "{got} {want}");
    }
}

// A cone screwed about a line that grazes it turns back through what it swept: the sheets its rim
// and its grazing curves sweep pass through themselves and each other, though they share edges.
// Sewn as they stand they bound the overlap twice, 6% more than the solid.
#[test]
fn a_cone_screwed_back_through_its_own_sweep_is_no_product() {
    let frame = Frame::new(
        Vec3::new(-1.0, -0.5, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::X,
    )
    .unwrap();
    let body = crate::primitive::cone(&frame, 1.0, 0.0, 2.0, "cone").unwrap();
    let motion = Motion::Screw {
        screw: screw_z(2.0),
        from: 0.0,
        to: 2.2,
    };
    let (steady, twist) = (
        sheet::Steady::of(&motion).unwrap(),
        motion.steady().unwrap(),
    );
    let cut = cut::cut(&body, &move |p| twist.at(p), steady.held()).unwrap();
    assert!(matches!(
        product::product(&cut, &steady, "sw", None, None),
        Err(Error::InvalidResult(_))
    ));
    assert!(solid(&body, &motion, "sw").is_err());
}

// A leaning block screwed far from the axis is built in parts. What its top sweeps stands on the
// block, and the sheet one top edge sweeps leaves the block's side at the top corner, a third of a
// degree off the corner's own path: their intersection runs out of the sheet there, so the two
// meet at the corner alone, and the parts share no volume.
#[test]
fn a_swept_top_joins_its_block_where_a_sheet_leaves_the_side_at_a_corner() {
    let frame = Frame::new(
        Vec3::new(10.0, -2.0, 0.0),
        Vec3::new(0.3, 0.0, 1.0),
        Vec3::X,
    )
    .unwrap();
    let body = block(&frame, Vec3::new(2.0, 1.0, 3.0), "block").unwrap();
    let motion = Motion::Screw {
        screw: screw_z(40.0),
        from: 0.0,
        to: 1.0,
    };
    let (steady, twist) = (
        sheet::Steady::of(&motion).unwrap(),
        motion.steady().unwrap(),
    );
    let cut = cut::cut(&body, &move |p| twist.at(p), steady.held()).unwrap();
    let start = product::start(&cut, &steady).unwrap();
    let top = cut
        .parts
        .iter()
        .position(|p| p.front && p.names.iter().any(|n| n.face == "end"))
        .unwrap();
    let piece = product::product(&cut, &steady, "sw", Some(&[top]), None).unwrap();
    let facts = crate::blend::shared_edges(&start, &piece);
    let both = boolean::boolean_with(&start, &piece, boolean::Op::Union, &facts).unwrap();
    assert!(check::check(&both).is_empty());
    let want = measure::volume(&start) + measure::volume(&piece);
    assert!(rel(measure::volume(&both), want) < 1e-12);
}

// A cone lying nearly across a far screw's axis grazes along a curve that runs into its apex a
// few thousandths of a radian from the seam: within the tolerance of the seam well short of the
// apex, and it ends at the apex all the same.
#[test]
fn a_grazing_curve_beside_a_cones_seam_runs_on_to_the_apex() {
    let frame = Frame::new(
        Vec3::new(10.0, -2.0, 0.0),
        Vec3::new(1.0, 0.0, 0.2),
        Vec3::X,
    )
    .unwrap();
    let body = crate::primitive::cone(&frame, 1.0, 0.0, 2.0, "cone").unwrap();
    let motion = Motion::Screw {
        screw: screw_z(10.0),
        from: 0.0,
        to: 1.0,
    };
    let b = solid(&body, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let field = |p: Vec3| Vec3::Z.cross(p) + Vec3::Z * (10.0 / (2.0 * PI));
    let want = measure::volume(&body) + frustum_outflow(&frame, (1.0, 0.0, 2.0), &field);
    let got = measure::volume(&b);
    assert!(rel(got, want) < 1e-5, "{got} {want}");
}

// A drum lying nearly across a screw's axis carries a sliver of its far end outward, and the rest
// of the drum then turns through where the sliver went: the faces the sliver's edges sweep pass
// through an end of the sweep, though they cross no swept face. The product is refused, and
// built in parts no face of the solid lies inside the drum anywhere along the way.
#[test]
fn a_drums_sliver_swept_under_the_drum_is_no_product() {
    let frame = Frame::new(Vec3::new(3.0, 1.0, -1.0), Vec3::new(1.0, 0.0, 0.2), Vec3::X).unwrap();
    let (r, h, rise) = (1.0, 3.0, 2.0 / (2.0 * PI));
    let body = cylinder(&frame, r, h, "drum").unwrap();
    let motion = Motion::Screw {
        screw: screw_z(2.0),
        from: 0.0,
        to: 1.0,
    };
    let (steady, twist) = (
        sheet::Steady::of(&motion).unwrap(),
        motion.steady().unwrap(),
    );
    let cut = cut::cut(&body, &move |p| twist.at(p), steady.held()).unwrap();
    assert!(matches!(
        product::product(&cut, &steady, "sw", None, None),
        Err(Error::InvalidResult(_))
    ));
    let b = solid(&body, &motion, "sw").unwrap();
    assert!(check::check(&b).is_empty());
    // Where the drum at turn `t` was when it started, and whether it holds a point by a margin.
    let held = |p: Vec3| {
        (0..=400).map(|k| k as f64 / 400.0).any(|t| {
            let q = p - Vec3::Z * (rise * t);
            let (s, c) = t.sin_cos();
            let l = frame.to_local(Vec3::new(c * q.x + s * q.y, c * q.y - s * q.x, q.z));
            (0.02..=h - 0.02).contains(&l.z) && l.x.hypot(l.y) <= r - 0.02
        })
    };
    for f in b.face_ids() {
        let d = crate::boolean::face_domain(&b, f);
        for i in 0..12 {
            for j in 0..12 {
                let at = |lo: f64, hi: f64, k: usize| lo + (hi - lo) * (k as f64 + 0.5) / 12.0;
                let uv = crate::math::Vec2::new(at(d.lo.x, d.hi.x, i), at(d.lo.y, d.hi.y, j));
                let p = d.surface.eval(uv.x, uv.y);
                assert!(d.winding(uv) == 0 || !held(p), "{p:?}");
            }
        }
    }
}

// A turn's axis pierces a cone's side. A point of the grazing curve a hundredth from the axis
// turns on a circle that leaves the side, and comes back to it where the sweep ends, at a shallow
// angle: its meeting with the side at the end is the path's own end, found a millionth short of
// it, and not a place the swept face passes through the side.
#[test]
fn a_path_that_ends_shallow_on_its_own_cap_does_not_pass_through_it() {
    let frame = Frame::new(Vec3::new(0.5, 0.5, 0.0), Vec3::new(0.4, 0.2, 1.0), Vec3::X).unwrap();
    let body = crate::primitive::cone(&frame, 1.0, 0.0, 2.0, "cone").unwrap();
    let motion = about_z(0.5);
    let (steady, twist) = (
        sheet::Steady::of(&motion).unwrap(),
        motion.steady().unwrap(),
    );
    let cut = cut::cut(&body, &move |p| twist.at(p), steady.held()).unwrap();
    let side = cut
        .parts
        .iter()
        .position(|p| p.front && p.names.iter().any(|n| n.face == "side"))
        .unwrap();
    let piece = product::product(&cut, &steady, "sw", Some(&[side]), None).unwrap();
    assert!(check::check(&piece).is_empty());
}

// An upright ring turned half a turn about a far line beside it: over each orbit it is the ring's
// outline filled in, from where the orbit first goes into the ring to where it last leaves, and
// half a turn on. That is the outline's section, a stadium, turned half a turn (Pappus), and the
// filled outline itself, the stadium turned about the ring's own axis.
#[test]
fn a_ring_turned_half_a_turn_beside_a_far_axis() {
    let (big, small, at) = (2.0, 0.5, Vec3::new(10.0, -2.0, 0.0));
    let ring = crate::primitive::torus(&Frame::WORLD.translated(at), big, small, "ring").unwrap();
    let b = solid(&ring, &about_z(PI), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let section = 4.0 * big * small + PI * small * small;
    let filled =
        PI * (2.0 * big * big * small + big * PI * small * small + 4.0 / 3.0 * small.powi(3));
    let want = PI * at.x.hypot(at.y) * section + filled;
    assert!(
        rel(measure::volume(&b), want) < 1e-9,
        "{} {want}",
        measure::volume(&b)
    );
}

/// Whether no face of `b` lies inside a body as it turns about z through `angle`: `held` says
/// whether the body, where it started, holds a point by a margin, and each face is sampled over
/// its window of parameters. The boundary of what a body sweeps touches the body and never lies
/// in it.
fn no_face_inside(b: &Brep, angle: f64, held: &dyn Fn(Vec3) -> bool) -> bool {
    let reached = |p: Vec3| {
        (0..=400).map(|k| angle * f64::from(k) / 400.0).any(|t| {
            let (s, c) = t.sin_cos();
            held(Vec3::new(c * p.x + s * p.y, c * p.y - s * p.x, p.z))
        })
    };
    b.face_ids().all(|f| {
        let d = crate::boolean::face_domain(b, f);
        (0..12).all(|i| {
            (0..12).all(|j| {
                let at = |lo: f64, hi: f64, k: usize| lo + (hi - lo) * (k as f64 + 0.5) / 12.0;
                let uv = crate::math::Vec2::new(at(d.lo.x, d.hi.x, i), at(d.lo.y, d.hi.y, j));
                d.winding(uv) == 0 || !reached(d.surface.eval(uv.x, uv.y))
            })
        })
    })
}

// A leaning ring turned a radian beside a far axis hides part of itself along the orbits: its
// near rim's outline casts a shadow on the far side of the hole, the faces at each end of the
// sweep are cut along that shadow, and a wall of orbit arcs stands between outline and shadow,
// closing to nothing at each fold of the outline.
#[test]
fn a_leaning_ring_turned_beside_a_far_axis_is_cut_along_its_own_shadow() {
    let frame = Frame::new(
        Vec3::new(10.0, -2.0, 0.0),
        Vec3::new(0.3, 0.0, 1.0),
        Vec3::X,
    )
    .unwrap();
    let (big, small) = (2.0, 0.5);
    let ring = crate::primitive::torus(&frame, big, small, "ring").unwrap();
    let b = solid(&ring, &about_z(1.0), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let held = |p: Vec3| {
        let l = frame.to_local(p);
        (l.x.hypot(l.y) - big).hypot(l.z) <= small - 0.02
    };
    assert!(no_face_inside(&b, 1.0, &held));
    let ring_volume = 2.0 * PI * PI * big * small * small;
    assert!(measure::volume(&b) > ring_volume);
}

// A leaning drum turned a radian about a line close beside it turns back through what it swept.
// Read off its orbits, each orbit's intervals merged, the overlap is bounded once: less than the
// count of all the drum's faces carry outward, which bounds it twice.
#[test]
fn a_leaning_drum_turned_beside_a_near_axis_bounds_its_overlap_once() {
    let frame = Frame::new(Vec3::new(3.0, 1.0, -1.0), Vec3::new(0.4, 0.2, 1.0), Vec3::X).unwrap();
    let (r, h) = (1.0, 3.0);
    let drum = cylinder(&frame, r, h, "drum").unwrap();
    let b = solid(&drum, &about_z(1.0), "sw").unwrap();
    assert!(check::check(&b).is_empty());
    let held = |p: Vec3| {
        let l = frame.to_local(p);
        (0.02..=h - 0.02).contains(&l.z) && l.x.hypot(l.y) <= r - 0.02
    };
    assert!(no_face_inside(&b, 1.0, &held));
    let field = |p: Vec3| Vec3::Z.cross(p);
    let twice = measure::volume(&drum) + frustum_outflow(&frame, (r, r, h), &field);
    let got = measure::volume(&b);
    assert!(
        got > measure::volume(&drum) && got < twice - 1e-3,
        "{got} {twice}"
    );
}

// A slab turned a radian about a line through it has orbits that leave it and come back after
// more than a radian: the gap is not closed by the turn, and the sweep has a hollow there. Read
// off its orbits with every orbit's intervals merged into one it came out valid and 1.1% too
// large, so such a turn is not read that way: the slab spans more of a turn than it is turned.
#[test]
fn a_turn_shorter_than_the_body_spans_is_not_merged_along_its_orbits() {
    let frame = Frame::new(
        Vec3::new(-1.0, -0.5, -1.0),
        Vec3::new(0.2, 0.4, 1.0),
        Vec3::new(1.0, 1.0, 0.0),
    )
    .unwrap();
    let slab = block(&frame, Vec3::new(3.0, 0.5, 2.0), "slab").unwrap();
    let motion = about_z(1.0);
    let (steady, twist) = (
        sheet::Steady::of(&motion).unwrap(),
        motion.steady().unwrap(),
    );
    let cut = cut::cut(&slab, &move |p| twist.at(p), steady.held()).unwrap();
    assert!(matches!(
        stack::sweep(&cut, &steady, "sw"),
        Err(Error::Unsupported(_))
    ));
}

// A whole turn is two half turns united. A cube's half turn about a line beside it is read off
// its orbits, and the union of two such halves is refused; the whole turn is then built as it
// was before.
#[test]
fn a_whole_turn_whose_halves_read_off_their_orbits_do_not_unite_is_built_the_older_way() {
    let frame = Frame::new(Vec3::new(0.5, 0.5, 0.0), Vec3::new(1.0, 1.0, 1.0), Vec3::X).unwrap();
    let cube = block(&frame, Vec3::new(1.0, 1.0, 1.0), "cube").unwrap();
    let b = solid(&cube, &about_z(2.0 * PI), "sw").unwrap();
    assert!(check::check(&b).is_empty());
}
