//! The five primitive solids against closed-form volume and area, each checked valid, on a frame
//! away from the origin and tilted off every axis.

use std::f64::consts::PI;

use springrep::{
    Error, check,
    math::{Frame, Vec3},
    measure,
    primitive::{block, cone, cylinder, ellipsoid, sphere, torus},
    topo::Brep,
};

fn tilted() -> Frame {
    Frame::new(
        Vec3::new(3.0, -2.0, 1.5),
        Vec3::new(0.3, -0.5, 0.8),
        Vec3::new(1.0, 0.2, 0.0),
    )
    .unwrap()
}

fn assert_solid(b: &Brep, volume: f64, area: f64) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
    let (v, a) = (measure::volume(b), measure::area(b));
    assert!(
        (v - volume).abs() <= 1e-10 * volume,
        "volume {v} vs {volume}"
    );
    assert!((a - area).abs() <= 1e-10 * area, "area {a} vs {area}");
}

/// The centroid in the frame's own coordinates: where along its axes the solid sits.
fn local_centroid(b: &Brep, f: &Frame) -> Vec3 {
    f.to_local(measure::mass_properties(b).centre)
}

fn assert_at(got: Vec3, want: [f64; 3]) {
    assert!((got - Vec3::from_array(want)).max_abs() < 1e-9, "{got:?}");
}

#[test]
fn a_block_runs_from_its_frames_origin_along_each_axis() {
    let f = tilted();
    let b = block(&f, Vec3::new(3.0, 2.0, 4.0), "blk").unwrap();
    assert_solid(&b, 24.0, 2.0 * (6.0 + 12.0 + 8.0));
    assert_at(local_centroid(&b, &f), [1.5, 1.0, 2.0]);
}

#[test]
fn a_cylinder_stands_on_its_frames_origin_about_z() {
    let f = tilted();
    let b = cylinder(&f, 1.5, 4.0, "cyl").unwrap();
    assert_solid(&b, PI * 2.25 * 4.0, 2.0 * PI * 2.25 + 2.0 * PI * 1.5 * 4.0);
    assert_at(local_centroid(&b, &f), [0.0, 0.0, 2.0]);
}

#[test]
fn a_sphere_is_centred_on_its_frames_origin() {
    let f = tilted();
    let b = sphere(&f, 2.0, "sph").unwrap();
    assert_solid(&b, 4.0 / 3.0 * PI * 8.0, 4.0 * PI * 4.0);
    assert_at(local_centroid(&b, &f), [0.0; 3]);
}

#[test]
fn a_torus_rings_its_frames_z_axis() {
    let f = tilted();
    let b = torus(&f, 3.0, 0.75, "tor").unwrap();
    let tube = PI * 0.75 * 0.75;
    assert_solid(&b, tube * 2.0 * PI * 3.0, 2.0 * PI * 0.75 * 2.0 * PI * 3.0);
    assert_at(local_centroid(&b, &f), [0.0; 3]);
}

#[test]
fn an_ellipsoid_stretches_the_ball_along_its_frames_axes() {
    let f = tilted();
    let (a, b, c) = (3.0, 2.0, 1.5);
    let e = ellipsoid(&f, Vec3::new(a, b, c), "ell").unwrap();
    assert!(check::check(&e).is_empty());
    let v = measure::volume(&e);
    let want = 4.0 / 3.0 * PI * a * b * c;
    assert!((v - want).abs() <= 1e-9 * want, "volume {v} vs {want}");
    assert_at(local_centroid(&e, &f), [0.0; 3]);
    assert_eq!(e.face_count(), 1);
    // The box's half-extent along a world axis w is √Σ (rₖ eₖ·w)², eₖ the frame's axes.
    let bb = measure::bbox(&e);
    for (k, w) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
        let half = [(a, f.x), (b, f.y), (c, f.z)]
            .iter()
            .map(|&(r, e)| (r * e.dot(w)).powi(2))
            .sum::<f64>()
            .sqrt();
        let mid = f.origin.to_array()[k];
        assert!((bb.min.to_array()[k] - (mid - half)).abs() < 1e-9, "{bb:?}");
        assert!((bb.max.to_array()[k] - (mid + half)).abs() < 1e-9, "{bb:?}");
    }
    // A prolate spheroid's area: 2πa²(1 + c·asin(ε)/(a·ε)), ε² = 1 − a²/c².
    let s = ellipsoid(&f, Vec3::new(2.0, 2.0, 3.0), "sph").unwrap();
    let eps = (1.0 - 4.0 / 9.0f64).sqrt();
    let area = 2.0 * PI * 4.0 * (1.0 + 3.0 * eps.asin() / (2.0 * eps));
    let got = measure::area(&s);
    assert!((got - area).abs() <= 1e-9 * area, "area {got} vs {area}");
    // Equal radii are the sphere.
    let ball = ellipsoid(&f, Vec3::new(2.0, 2.0, 2.0), "sph").unwrap();
    assert_solid(&ball, 4.0 / 3.0 * PI * 8.0, 4.0 * PI * 4.0);
    assert!(ellipsoid(&f, Vec3::new(2.0, 0.0, 2.0), "sph").is_err());
}

#[test]
fn a_horn_torus_pinches_to_a_point_at_its_frames_origin() {
    let f = tilted();
    let b = torus(&f, 1.5, 1.5, "tor").unwrap();
    // Pappus: the disk's area and perimeter times the path of its centre, 2π·1.5
    assert_solid(&b, PI * 2.25 * 3.0 * PI, 3.0 * PI * 3.0 * PI);
    assert_at(local_centroid(&b, &f), [0.0; 3]);
    assert_eq!(b.face_count(), 1);
}

#[test]
fn a_spindle_torus_is_its_tubes_outer_arc_turned() {
    let f = tilted();
    let (big, small) = (1.5, 2.5);
    let b = torus(&f, big, small, "tor").unwrap();
    // The axis cuts a segment off the disk, its chord subtending 2θ at the tube's centre,
    // cos θ = R/r. The volume is 2π times the first moment about the axis of the rest of the disk,
    // and the area is the outer arc turned, 4πr(Rφ + r sin φ), φ = π − θ.
    let theta = (big / small).acos();
    let chord = (small * small - big * big).sqrt();
    let cut = small * small * theta - big * chord;
    let moment = PI * small * small * big - big * cut + 2.0 / 3.0 * chord.powi(3);
    let phi = PI - theta;
    let area = 4.0 * PI * small * (big * phi + small * phi.sin());
    assert_solid(&b, 2.0 * PI * moment, area);
    assert_at(local_centroid(&b, &f), [0.0; 3]);
    assert_eq!(b.face_count(), 1);
}

#[test]
fn a_cone_is_a_frustum_or_comes_to_an_apex_at_either_end() {
    let f = tilted();
    let frustum = |r0: f64, r1: f64, h: f64| {
        let slant = ((r0 - r1).powi(2) + h * h).sqrt();
        (
            PI * h / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1),
            PI * (r0 * r0 + r1 * r1 + (r0 + r1) * slant),
        )
    };
    for (r0, r1) in [(2.0, 0.5), (2.0, 0.0), (0.0, 1.5)] {
        let b = cone(&f, r0, r1, 3.0, "con").unwrap();
        let (v, a) = frustum(r0, r1, 3.0);
        assert_solid(&b, v, a);
    }
    // Equal radii are the frustum that is a cylinder.
    let b = cone(&f, 1.5, 1.5, 4.0, "con").unwrap();
    assert_solid(&b, PI * 2.25 * 4.0, 2.0 * PI * 2.25 + 2.0 * PI * 1.5 * 4.0);
}

#[test]
fn a_size_at_or_below_zero_is_invalid_input() {
    let f = tilted();
    let refused = |r: Result<Brep, Error>| matches!(r, Err(Error::InvalidInput(_)));
    assert!(refused(block(&f, Vec3::new(3.0, 0.0, 4.0), "blk")));
    assert!(refused(cylinder(&f, -1.0, 4.0, "cyl")));
    assert!(refused(cylinder(&f, 1.0, 0.0, "cyl")));
    assert!(refused(sphere(&f, 0.0, "sph")));
    assert!(refused(cone(&f, 0.0, 0.0, 3.0, "con")));
    assert!(refused(cone(&f, 1.0, -0.5, 3.0, "con")));
    assert!(refused(torus(&f, 0.0, 1.0, "tor")));
    assert!(refused(torus(&f, 1.0, 0.0, "tor")));
}
