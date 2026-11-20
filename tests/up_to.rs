//! Sweeps that run until they cross another solid's boundary, into it or out of its material,
//! against closed forms.

use std::f64::consts::PI;

use springrep::{
    boolean::{difference, union},
    check,
    math::{Frame, Vec2, Vec3},
    measure,
    primitive::cylinder,
    profile::Profile,
    sweep::{Meet, extrude, extrude_to_next, revolve, revolve_to_next},
    topo::Brep,
};

fn at(x: f64, y: f64, z: f64) -> Frame {
    Frame::new(Vec3::new(x, y, z), Vec3::Z, Vec3::X).unwrap()
}

fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-9 * want, "{got} against {want}");
}

fn valid(b: &Brep) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
}

#[test]
fn a_pad_extruded_down_stops_on_the_cylinder_below() {
    // The square |x|, |y| ≤ a at z = h, down onto the side of the cylinder of radius R along x.
    let (r, a, h) = (10.0f64, 3.0f64, 15.0);
    let along_x = Frame::new(Vec3::new(-20.0, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap();
    let body = cylinder(&along_x, r, 40.0, "cyl").unwrap();
    let square = Profile::rect(Vec2::ZERO, 2.0 * a, 2.0 * a);
    let pad = extrude_to_next(&square, &at(0.0, 0.0, h), true, &body, Meet::Into, "pad").unwrap();
    valid(&pad);
    // Its section across x is the rectangle under z = h less the circle's segment above z = 0:
    // 2ah − S, S = a√(R² − a²) + R²·asin(a/R), the part of the disk with |y| ≤ a and z ≥ 0.
    let s = a * (r * r - a * a).sqrt() + r * r * (a / r).asin();
    close(measure::volume(&pad), 2.0 * a * (2.0 * a * h - s));
    let lo = measure::bbox(&pad).min.z;
    assert!((lo - (r * r - a * a).sqrt()).abs() < 1e-9, "{lo}");
    let joined = union(&body, &pad).unwrap();
    close(
        measure::volume(&joined),
        PI * r * r * 40.0 + 2.0 * a * (2.0 * a * h - s),
    );
    // Aimed past the cylinder, it never meets it.
    let past = extrude_to_next(&square, &at(0.0, 30.0, h), true, &body, Meet::Into, "pad");
    assert!(past.is_err());
}

#[test]
fn a_hole_drilled_out_of_the_material_stops_at_the_gap() {
    // Two plates 4 thick, 0 ≤ z ≤ 4 and 10 ≤ z ≤ 14, joined by a back wall; a hole of radius 2
    // from the top runs out of the upper plate into the gap and no further.
    let section = [
        Vec2::new(0.0, 0.0),
        Vec2::new(20.0, 0.0),
        Vec2::new(20.0, 4.0),
        Vec2::new(4.0, 4.0),
        Vec2::new(4.0, 10.0),
        Vec2::new(20.0, 10.0),
        Vec2::new(20.0, 14.0),
        Vec2::new(0.0, 14.0),
    ];
    let xz = Frame::new(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, Vec3::X).unwrap();
    let bracket = extrude(&Profile::polygon(&section, &["s"]), &xz, 10.0, "b").unwrap();
    let disk = Profile::circle(Vec2::ZERO, 4.0);
    let top = at(12.0, 0.0, 14.0);
    let drill = extrude_to_next(&disk, &top, true, &bracket, Meet::OutOf, "hole").unwrap();
    valid(&drill);
    close(measure::volume(&drill), PI * 4.0 * 4.0);
    let drilled = difference(&bracket, &drill).unwrap();
    let area = 2.0 * 20.0 * 4.0 + 6.0 * 4.0;
    close(measure::volume(&drilled), area * 10.0 - PI * 4.0 * 4.0);
    // Started above the part, it keeps nothing out of the material.
    let above = at(12.0, 0.0, 20.0);
    assert!(extrude_to_next(&disk, &above, true, &bracket, Meet::OutOf, "hole").is_err());
}

#[test]
fn a_rib_turned_about_a_hub_stops_at_the_stop() {
    // A hub of radius 10 and a stop between 120° and 135°; the rib from 5 to 20 out and 4 tall,
    // part of it inside the hub, turned from 0° until it meets the stop: the annular sector from
    // 10 to 20 through 120°.
    let deg = PI / 180.0;
    let plane = |from: f64| {
        let x = Vec3::new(from.cos(), from.sin(), 0.0);
        // its normal x × z, so its y is the z axis
        Frame::new(Vec3::ZERO, x.cross(Vec3::Z), x).unwrap()
    };
    let rect =
        |r0: f64, r1: f64, h: f64| Profile::rect(Vec2::new((r0 + r1) / 2.0, h / 2.0), r1 - r0, h);
    let hub = cylinder(&at(0.0, 0.0, 0.0), 10.0, 10.0, "hub").unwrap();
    let stop = revolve(
        &rect(0.0, 25.0, 10.0),
        &plane(120.0 * deg),
        Vec3::ZERO,
        Vec3::Z,
        15.0 * deg,
        "stop",
    )
    .unwrap();
    let body = union(&hub, &stop).unwrap();
    let rib = revolve_to_next(
        &rect(5.0, 20.0, 4.0),
        &plane(0.0),
        Vec3::ZERO,
        Vec3::Z,
        &body,
        Meet::Into,
        "rib",
    )
    .unwrap();
    valid(&rib);
    close(measure::volume(&rib), PI / 3.0 * (400.0 - 100.0) * 4.0);
}
