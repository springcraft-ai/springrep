//! Offsets that grow a solid past the radius of a concave cylindrical face, which closes on the
//! way: a blind bore, a shallow recess, a rounded inside corner and the rounded corners of a
//! pocket and of a window. Each is checked against its volume and area in closed form.

use std::f64::consts::{FRAC_PI_2, PI};

use springrep::{
    blend::fillet,
    boolean::difference,
    check, edit,
    math::{Frame, Vec2, Vec3},
    measure,
    primitive::{block, cylinder},
    profile::{NamedSegment, Profile},
    sweep::extrude,
    topo::Brep,
};

fn at(x: f64, y: f64, z: f64) -> Frame {
    Frame::new(Vec3::new(x, y, z), Vec3::Z, Vec3::X).unwrap()
}

/// `b` is a valid solid of `volume` and `area`, to a part in 10⁹.
fn assert_solid(b: &Brep, volume: f64, area: f64) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
    let v = measure::volume(b);
    let a = measure::area(b);
    assert!(
        (v - volume).abs() <= 1e-9 * volume,
        "volume {v} vs {volume}"
    );
    assert!((a - area).abs() <= 1e-9 * area, "area {a} vs {area}");
}

/// The volume and the area of the box `a` × `b` × `c` grown by `d`: its faces moved out, its
/// edges quarter cylinders and its corners eighths of a ball (Steiner's formula).
fn grown_box(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let v = a * b * c
        + 2.0 * (a * b + b * c + c * a) * d
        + PI * (a + b + c) * d * d
        + 4.0 * PI * d * d * d / 3.0;
    let s = 2.0 * (a * b + b * c + c * a) + 2.0 * PI * (a + b + c) * d + 4.0 * PI * d * d;
    (v, s)
}

/// Over a quarter of a round's rim of radius `r`, grown by `d` past it: the volume between the
/// grown rim's tube and the square corner the faces beside the round would leave, integrated over
/// the height above the rim's plane, per unit of the corner's angle weight `1 − π/4`.
fn rim_gap(r: f64, d: f64) -> f64 {
    let q = (d * d - r * r).sqrt();
    r * r * (d - q) - 2.0 * r * (PI * d * d / 4.0 - q * r / 2.0 - d * d / 2.0 * (q / d).asin())
        + d * d * (d - q)
        - (d * d * d - q * q * q) / 3.0
}

/// The rectangle from (−10, −7) to (10, 7) with its corners rounded to radius 2.
fn rounded_rectangle(name: &str) -> Profile {
    let p = Vec2::new;
    Profile::outline(vec![
        NamedSegment::line(p(-8.0, -7.0), p(8.0, -7.0), name),
        NamedSegment::arc(p(8.0, -5.0), 2.0, -FRAC_PI_2, FRAC_PI_2, "corner"),
        NamedSegment::line(p(10.0, -5.0), p(10.0, 5.0), name),
        NamedSegment::arc(p(8.0, 5.0), 2.0, 0.0, FRAC_PI_2, "corner"),
        NamedSegment::line(p(8.0, 7.0), p(-8.0, 7.0), name),
        NamedSegment::arc(p(-8.0, 5.0), 2.0, FRAC_PI_2, FRAC_PI_2, "corner"),
        NamedSegment::line(p(-10.0, 5.0), p(-10.0, -5.0), name),
        NamedSegment::arc(p(-8.0, -5.0), 2.0, PI, FRAC_PI_2, "corner"),
    ])
}

#[test]
fn a_blind_bore_grown_past_its_radius_fills_and_leaves_a_dimple_over_its_mouth_only() {
    let b = block(&at(-10.0, -10.0, 0.0), Vec3::new(20.0, 20.0, 10.0), "block").unwrap();
    let bore = cylinder(&at(0.0, 0.0, 2.0), 3.0, 9.0, "bore").unwrap();
    let grown = edit::offset(&difference(&b, &bore).unwrap(), 4.0, "grown").unwrap();

    // the dimple over the mouth is the inner side of the tube of radius d about the rim of radius
    // ρ; the floor, 8 down, is too far to matter
    let (rho, d) = (3.0f64, 4.0f64);
    let q = (d * d - rho * rho).sqrt();
    let phi = (rho / d).asin();
    let (v0, a0) = grown_box(20.0, 20.0, 10.0, d);
    let volume = v0
        - 2.0
            * PI
            * (d * rho * rho / 2.0 - rho / 2.0 * (rho * q + d * d * phi)
                + (d.powi(3) - q.powi(3)) / 3.0);
    let area = a0 - PI * rho * rho + 2.0 * PI * d * (rho * phi - d * (1.0 - phi.cos()));
    assert_solid(&grown, volume, area);
}

#[test]
fn a_shallow_recess_grown_past_its_radius_leaves_a_dimple_floored_by_its_moved_floor() {
    let plate = block(&at(-10.0, -10.0, 0.0), Vec3::new(20.0, 20.0, 4.0), "plate").unwrap();
    let recess = cylinder(&at(0.0, 0.0, 3.0), 3.0, 2.0, "recess").unwrap();
    let grown = edit::offset(&difference(&plate, &recess).unwrap(), 4.0, "grown").unwrap();

    let s7 = 7f64.sqrt();
    let w = (s7 / 4.0).asin();
    let (v0, a0) = grown_box(20.0, 20.0, 4.0, 4.0);
    let volume =
        v0 - 2.0 * PI * ((3.0 - s7).powi(2) / 2.0 + 15.0 * s7 / 2.0 - 24.0 * w - 5.0 / 3.0);
    let area = a0 - 9.0 * PI + PI * (3.0 - s7).powi(2) + 8.0 * PI * (3.0 * w - 1.0);
    assert_solid(&grown, volume, area);
}

#[test]
fn an_l_plate_with_its_inside_corner_rounded_grown_past_the_round_keeps_the_rounds_rim() {
    let outline = [
        (0., 0.),
        (20., 0.),
        (20., 5.),
        (5., 5.),
        (5., 15.),
        (0., 15.),
    ]
    .map(|(x, y)| Vec2::new(x, y));
    let section = Profile::polygon(&outline, &["front", "end", "ledge", "riser", "top", "back"]);
    let plate = extrude(&section, &at(0.0, 0.0, 0.0), 8.0, "plate").unwrap();
    let rounded = fillet(&plate, &[("plate.ledge", "plate.riser")], 4.0, "round").unwrap();
    let grown = edit::offset(&rounded, 5.0, "grown").unwrap();

    // the sharp-cornered L's area and perimeter, its thickness, its corners' turning weight
    let (a, p, t, k, r, d) = (150.0, 70.0, 8.0, 5.0 * PI / 4.0 - 1.0, 4.0f64, 5.0f64);
    let q = (d * d - r * r).sqrt();
    let corner = 1.0 - PI / 4.0;
    let volume = t * (a + p * d + k * d * d)
        + 2.0 * (a * d + PI * p * d * d / 4.0 + 2.0 * k * d.powi(3) / 3.0)
        + 2.0 * corner * rim_gap(r, d);
    let area = t * (p + 2.0 * k * d)
        + 2.0 * (a + PI * p * d / 2.0 + 2.0 * k * d * d)
        + 2.0 * corner * (r * r - 2.0 * d * r * (r / d).asin() + 2.0 * d * (d - q));
    assert_solid(&grown, volume, area);
}

/// The block (−20, −15, 0)–(20, 15, `height`) less the rounded rectangle's prism from `floor` up
/// past its top, grown by 3.
fn pocketed_grown(height: f64, floor: f64) -> Brep {
    let body = block(
        &at(-20.0, -15.0, 0.0),
        Vec3::new(40.0, 30.0, height),
        "block",
    )
    .unwrap();
    let cutter = extrude(
        &rounded_rectangle("wall"),
        &at(0.0, 0.0, floor),
        height + 1.0 - floor,
        "pocket",
    )
    .unwrap();
    edit::offset(&difference(&body, &cutter).unwrap(), 3.0, "grown").unwrap()
}

#[test]
fn a_pocket_grown_past_its_corners_radius_keeps_its_corners_rims() {
    let grown = pocketed_grown(10.0, 4.0);
    let (w, h, r, depth, d) = (20.0, 14.0, 2.0f64, 6.0, 3.0f64);
    let q = (d * d - r * r).sqrt();
    let (v0, a0) = grown_box(40.0, 30.0, 10.0, d);
    let volume = v0
        - ((w - 2.0 * d) * (h - 2.0 * d) * (depth - d) + w * h * d - PI * (w + h) * d * d / 2.0
            + 8.0 * d.powi(3) / 3.0
            - (4.0 - PI) * rim_gap(r, d));
    let area = a0 - (w * h - (4.0 - PI) * r * r)
        + (w - 2.0 * d) * (h - 2.0 * d)
        + (2.0 * (w + h) - 8.0 * d) * (depth - d)
        + PI * (w + h) * d
        - 8.0 * d * d
        + (2.0 * PI - 8.0) * d * (r * (r / d).asin() - (d - q));
    assert_solid(&grown, volume, area);
}

#[test]
fn a_window_grown_past_its_corners_radius_keeps_its_corners_rims_on_both_faces() {
    let grown = pocketed_grown(4.0, -1.0);
    let (c, w, h, r, d) = (4.0, 20.0, 14.0, 2.0f64, 3.0f64);
    let q = (d * d - r * r).sqrt();
    let (v0, a0) = grown_box(40.0, 30.0, c, d);
    let volume = v0
        - (c * (w - 2.0 * d) * (h - 2.0 * d)
            + 2.0
                * (w * h * d - PI * (w + h) * d * d / 2.0 + 8.0 * d.powi(3) / 3.0
                    - (4.0 - PI) * rim_gap(r, d)));
    let area = a0 - 2.0 * (w * h - (4.0 - PI) * r * r)
        + c * (2.0 * (w + h) - 8.0 * d)
        + 2.0
            * (PI * (w + h) * d - 8.0 * d * d
                + (2.0 * PI - 8.0) * d * (r * (r / d).asin() - (d - q)));
    assert_solid(&grown, volume, area);
}
