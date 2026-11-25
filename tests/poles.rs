//! Plain solids cut by a plane through a pole, an apex or a seam of a face on a periodic surface:
//! there the longitude is free, or the parameter plane is cut, and the pieces' outlines in it
//! must still close and run their faces' way.

use std::f64::consts::PI;

use springrep::{
    boolean::{difference, intersection},
    check,
    math::{Frame, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    primitive::{cone, cylinder, sphere, torus},
    profile::Profile,
    sweep::extrude,
    topo::Brep,
};

fn assert_solid(b: &Brep, volume: f64, area: f64, what: &str) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{what}: {issues:?}");
    let (v, a) = (measure::volume(b), measure::area(b));
    assert!(
        (v - volume).abs() <= 1e-9 * volume,
        "{what}: volume {v} vs {volume}"
    );
    assert!(
        (a - area).abs() <= 1e-9 * area,
        "{what}: area {a} vs {area}"
    );
    let m = mesh::tessellate(b, &MeshOptions::new(0.05, 0.3)).unwrap();
    for s in b.solid_ids() {
        assert!(mesh::report(b, &m, s).watertight(), "{what}");
    }
}

/// The half-space past the plane through `at` with normal `n`, as a slab far larger than `size`.
fn past(at: Vec3, n: Vec3, size: f64) -> Brep {
    let frame = Frame::new(at, n, n.any_perpendicular()).unwrap();
    extrude(
        &Profile::rect(Vec2::ZERO, 8.0 * size, 8.0 * size),
        &frame,
        4.0 * size,
        "past",
    )
    .unwrap()
}

/// The part of `solid` past the plane and the rest, against their hand-worked volumes and areas.
fn assert_cut(solid: &Brep, plane: &Brep, kept: (f64, f64), rest: (f64, f64), what: &str) {
    let k = intersection(solid, plane).unwrap();
    assert_solid(&k, kept.0, kept.1, &format!("{what}, kept"));
    let r = difference(solid, plane).unwrap();
    assert_solid(&r, rest.0, rest.1, &format!("{what}, rest"));
}

#[test]
fn a_ball_cut_by_a_plane_through_its_pole_keeps_its_cap_and_its_rest() {
    // A plane through the pole (0, 0, ±r) of a ball of radius r, leaning by `lean` from the
    // plane that touches the ball there, cuts off a cap of height h = r·(1 − cos lean): the cut
    // circle passes through the pole. The cap holds π·h²·(3r − h)/3 under 2π·r·h of the ball
    // and a disc of radius r·sin lean. Turned by `round` about the ball's axis, the circle
    // leaves the pole along the seam (round π/2 at the top, 3π/2 at the bottom) or across it.
    let r = 10.0;
    let ball = sphere(&Frame::WORLD, r, "ball").unwrap();
    for pole in [1.0, -1.0] {
        for lean in [PI / 4.0, 1.1, 2.0] {
            for round in [0.0, 1.0, PI / 2.0, PI, 1.5 * PI, 4.0] {
                let what = format!("pole {pole} lean {lean} round {round}");
                let flat = Vec3::new(round.cos(), round.sin(), 0.0);
                let n = flat * lean.sin() + Vec3::Z * (pole * lean.cos());
                let h = r * (1.0 - lean.cos());
                let cap = PI * h * h * (3.0 * r - h) / 3.0;
                let disc = PI * (r * lean.sin()).powi(2);
                let whole = 4.0 / 3.0 * PI * r.powi(3);
                let skin = 4.0 * PI * r * r;
                let cut = past(Vec3::new(0.0, 0.0, pole * r), n, r);
                let kept = (cap, 2.0 * PI * r * h + disc);
                let rest = (whole - cap, skin - 2.0 * PI * r * h + disc);
                assert_cut(&ball, &cut, kept, rest, &what);
            }
        }
    }
}

#[test]
fn a_cone_cut_by_a_plane_through_its_apex_keeps_a_cone_over_a_segment_of_its_base() {
    // A cone of base radius a and height H, cut by the plane through its apex and the chord of
    // its base at d from the axis, square to `round`. Past the plane lies the cone over the
    // base's segment beyond the chord, of half angle φ = acos(d/a) and area a²·(φ − sin φ·cos φ):
    // it holds H/3 of that, under the cone's side over the arc 2aφ, which unrolls to a sector of
    // the slant L = √(a² + H²) of area L·a·φ, and the triangle from the apex to the chord, of
    // base 2a·sin φ and height √(d² + H²). Its generators run along the seam at round ±φ.
    let (a, height, d) = (6.0, 8.0, 3.0);
    let body = cone(&Frame::WORLD, a, 0.0, height, "cone").unwrap();
    let phi = (d / a).acos();
    let slant = a.hypot(height);
    let segment = a * a * (phi - phi.sin() * phi.cos());
    let side = slant * a * phi;
    let triangle = a * phi.sin() * d.hypot(height);
    let whole = (PI * a * a * height / 3.0, PI * a * slant + PI * a * a);
    for round in [0.0, phi, -phi, 1.0, PI, 4.0] {
        let flat = Vec3::new(round.cos(), round.sin(), 0.0);
        let apex = Vec3::new(0.0, 0.0, height);
        let along = Vec3::Z.cross(flat);
        let n = along.cross(flat * d - apex).unit().unwrap();
        let n = if n.dot(flat) > 0.0 { n } else { -n };
        let cut = past(apex, n, a + height);
        let kept = (segment * height / 3.0, side + segment + triangle);
        let rest = (whole.0 - kept.0, whole.1 - side - segment + triangle);
        assert_cut(&body, &cut, kept, rest, &format!("round {round}"));
    }
}

#[test]
fn a_cylinder_and_a_torus_cut_through_their_axes_along_their_seams_keep_their_halves() {
    // A plane through the axis halves each. Turned by `round`, it runs along the seam (round 0)
    // or across the parameter plane elsewhere. Half a cylinder of radius r and height h holds
    // π·r²·h/2 under π·r·h, the halves of its ends and the 2r by h rectangle; half a torus of
    // radii R and r holds π²·R·r² under 2π²·R·r and two discs of radius r.
    let (r, h) = (3.0, 7.0);
    let can = cylinder(&Frame::WORLD, r, h, "can").unwrap();
    let (big, small) = (5.0, 2.0);
    let ring = torus(&Frame::WORLD, big, small, "ring").unwrap();
    for round in [0.0, 0.5 * PI, PI, 1.5 * PI, 2.5] {
        let n = Vec3::new(round.cos(), round.sin(), 0.0);
        let half_can = (PI * r * r * h / 2.0, PI * r * h + PI * r * r + 2.0 * r * h);
        let cut = past(Vec3::ZERO, n, r + h);
        assert_cut(
            &can,
            &cut,
            half_can,
            half_can,
            &format!("cylinder, round {round}"),
        );
        let half_ring = (
            PI * PI * big * small * small,
            2.0 * PI * PI * big * small + 2.0 * PI * small * small,
        );
        let cut = past(Vec3::ZERO, n, big + small);
        assert_cut(
            &ring,
            &cut,
            half_ring,
            half_ring,
            &format!("torus, round {round}"),
        );
    }
}
