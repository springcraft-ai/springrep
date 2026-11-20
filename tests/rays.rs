//! Rays against faces, against closed forms: where they cross, how often, and the normal there.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, PI, SQRT_2, TAU};

use springrep::{
    boolean::{intersection, union},
    edit,
    geom::{Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    primitive,
    profile::{NamedSegment, PointsInterp, Profile, ProfileLoop, Segment},
    rays::{Hit, Rays},
    sweep::{extrude, revolve},
    topo::Brep,
};

fn seg(segment: Segment, name: &str) -> NamedSegment {
    NamedSegment::new(segment, name)
}

fn ball(c: Vec3, r: f64) -> Brep {
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                seg(
                    Segment::Arc {
                        center: Vec2::ZERO,
                        radius: r,
                        start: -PI / 2.0,
                        sweep: PI,
                    },
                    "surface",
                ),
                seg(
                    Segment::Line {
                        a: Vec2::new(0.0, r),
                        b: Vec2::new(0.0, -r),
                    },
                    "axis",
                ),
            ],
        }],
    };
    let frame = Frame::new(c, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap();
    revolve(&p, &frame, c, Vec3::Z, TAU, "ball").unwrap()
}

fn block(lo: [f64; 3], hi: [f64; 3]) -> Brep {
    let frame = Frame::new(Vec3::new(0.0, 0.0, lo[2]), Vec3::Z, Vec3::X).unwrap();
    let centre = Vec2::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
    let p = Profile::rect(centre, hi[0] - lo[0], hi[1] - lo[1]);
    extrude(&p, &frame, hi[2] - lo[2], "b").unwrap()
}

fn cast(b: &Brep, o: Point3, d: Vec3, range: (f64, f64)) -> Vec<Hit> {
    Rays::new(b).unwrap().cast(o, d, range).unwrap()
}

fn assert_ts(hits: &[Hit], want: &[f64]) {
    let got: Vec<f64> = hits.iter().map(|h| h.t).collect();
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
            "{got:?} vs {want:?}"
        );
    }
}

const FAR: (f64, f64) = (0.0, 1e3);

#[test]
fn a_ray_beside_a_balls_seam_crosses_it_on_either_side() {
    // The seam is the meridian in y = 0, x > 0. A ray along z just either side of it is inside
    // the face on both sides: the seam bounds it both ways. A second ball, of radius 2 about
    // (-0.5, -1, 1.5), cuts the face up towards its pole, where a chord in the parameters strays
    // far across the surface, past the seam's distance from where the rays leave. Each ray passes
    // through the second ball inside the first, so it crosses the first at ±√(9 − x² − y²) only.
    let b = union(
        &ball(Vec3::ZERO, 3.0),
        &ball(Vec3::new(-0.5, -1.0, 1.5), 2.0),
    )
    .unwrap();
    let x: f64 = 1.0;
    for y in [-0.05, -0.02, 0.02, 0.05] {
        let z = (9.0 - x * x - y * y).sqrt();
        let hits = cast(&b, Vec3::new(x, y, -10.0), Vec3::Z, FAR);
        assert_ts(&hits, &[10.0 - z, 10.0 + z]);
    }
}

#[test]
fn a_ray_through_a_ball_crosses_twice_touching_once_and_missing_never() {
    // A ray along x passing h from the centre crosses at ±√(r² − h²) about its nearest approach.
    let (c, r) = (Vec3::new(1.0, 2.0, 3.0), 3.0);
    let b = ball(c, r);
    let o = c + Vec3::new(-10.0, 1.0, 0.0);
    let hits = cast(&b, o, Vec3::X, FAR);
    let half = (r * r - 1.0f64).sqrt();
    assert_ts(&hits, &[10.0 - half, 10.0 + half]);
    // The normal points out of the ball: from the centre to the point.
    for h in &hits {
        assert!(
            ((h.point - c) * (1.0 / r) - h.normal).norm() < 1e-9,
            "{h:?}"
        );
    }
    // Grazing the ball at h = r touches it once; past it, nothing.
    assert_ts(
        &cast(&b, c + Vec3::new(-10.0, r, 0.0), Vec3::X, FAR),
        &[10.0],
    );
    assert_ts(&cast(&b, c + Vec3::new(-10.0, 3.5, 0.0), Vec3::X, FAR), &[]);
    // A range that stops inside keeps the entry only.
    assert_ts(&cast(&b, o, Vec3::X, (0.0, 10.0)), &[10.0 - half]);
}

/// A cylinder of radius 50 km and height 40 km: rays from points inside it leave through its side
/// where the circle says, the far one along the ray's line in its plane. Each direction is
/// solved for exactly: the ray `p + t·d` meets `x² + y² = r²` at the positive root of a quadratic.
#[test]
fn rays_from_inside_a_cylinder_kilometres_across_leave_through_its_side() {
    let (r, h) = (5e4, 4e4);
    let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
    let b = extrude(&Profile::circle(Vec2::ZERO, 2.0 * r), &frame, h, "c").unwrap();
    let rays = Rays::new(&b).unwrap();
    // Directions that leave through the side before the caps from these points.
    let d = Vec3::new(0.6213, -0.2841, 0.07302).unit().unwrap();
    for p in [
        Vec3::new(-24471.3, -24.0, 9520.7),
        Vec3::new(21381.1, 5364.3, 3236.0),
        Vec3::new(-40844.9, 3285.6, 18088.7),
    ] {
        let (a, bq, c) = (
            d.x * d.x + d.y * d.y,
            2.0 * (p.x * d.x + p.y * d.y),
            p.x * p.x + p.y * p.y - r * r,
        );
        let t = (-bq + (bq * bq - 4.0 * a * c).sqrt()) / (2.0 * a);
        assert!((p + d * t).z < h, "the ray leaves through the side");
        let hits = rays.cast(p, d, (0.0, 1e6)).unwrap();
        assert_ts(&hits, &[t]);
    }
}

/// A ray that clips a cylinder's side in a chord shallower than the caster's mesh resolves: it
/// crosses twice, in and out, at the two roots of the quadratic in the cylinder's plane. Chords of
/// 3 µm down to 3 nm deep, on a cylinder of 2 mm radius.
#[test]
fn a_ray_clipping_a_cylinder_shallower_than_the_mesh_crosses_it_twice() {
    let (r, h) = (2.0, 10.0);
    let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
    let b = extrude(&Profile::circle(Vec2::ZERO, 2.0 * r), &frame, h, "c").unwrap();
    let rays = Rays::new(&b).unwrap();
    let d = Vec3::new(0.3713, 0.5347, 0.07591).unit().unwrap();
    let across = Vec3::new(d.y, -d.x, 0.0).unit().unwrap();
    let steep = Vec3::new(0.3713, 0.5347, 0.7591).unit().unwrap();
    let found = (
        Vec3::new(0.6263603775040814, -2.600153099460448, -1.2639721248298266),
        steep,
    );
    let shallow = [3e-3, 3e-4, 3e-5, 3e-6].map(|depth| {
        // A start whose line in the plane passes `depth` inside the circle.
        (across * (r - depth) - d * 4.0 + Vec3::new(0.0, 0.0, 5.0), d)
    });
    for (p, d) in shallow.into_iter().chain([found]) {
        let (a, bq, c) = (
            d.x * d.x + d.y * d.y,
            p.x * d.x + p.y * d.y,
            p.x * p.x + p.y * p.y - r * r,
        );
        let root = (bq * bq - a * c).sqrt();
        let hits = rays.cast(p, d, (0.0, 100.0)).unwrap();
        assert_ts(&hits, &[(-bq - root) / a, (-bq + root) / a]);
        assert!(hits[0].normal.dot(d) < 0.0 && hits[1].normal.dot(d) > 0.0);
    }
}

/// The lens two balls a few micrometres across share: rays from a point inside it leave where the
/// nearer ball's sphere says, `−b + √(b² − c)` along the unit direction for each sphere, the nearer
/// of the two. A part far smaller than the caster's own mesh deflection is still found.
#[test]
fn rays_from_inside_a_lens_micrometres_across_leave_where_its_spheres_say() {
    let (ca, ra) = (Vec3::ZERO, 0.004054250852155359);
    let (cb, rb) = (
        Vec3::new(0.00338806114604008, 0.0, 0.0),
        0.0026689525243786373,
    );
    let lens = intersection(&ball(ca, ra), &ball(cb, rb)).unwrap();
    let rays = Rays::new(&lens).unwrap();
    let p = Vec3::new(0.001017, -0.000028, -0.000248);
    for d in [
        Vec3::new(0.3713, 0.5347, 0.7591),
        Vec3::new(-0.6213, 0.2841, 0.7302),
        Vec3::new(0.4472, -0.8018, 0.3961),
    ] {
        let d = d.unit().unwrap();
        let exit = |c: Vec3, r: f64| {
            let (b, cc) = ((p - c).dot(d), (p - c).dot(p - c) - r * r);
            -b + (b * b - cc).sqrt()
        };
        assert_ts(
            &rays.cast(p, d, FAR).unwrap(),
            &[exit(ca, ra).min(exit(cb, rb))],
        );
    }
}

#[test]
fn a_ray_through_a_box_enters_and_leaves_where_the_slabs_say() {
    // x 0..10, y 0..6, z 0..3, crossed on a skew line: each axis's slab gives the t interval it
    // lies in, and the entry and exit are the latest start and the earliest end.
    let b = block([0.0; 3], [10.0, 6.0, 3.0]);
    let o = Vec3::new(-5.0, 2.0, 1.0);
    let d = Vec3::new(1.0, 0.1, 0.05).unit().unwrap();
    let slab = |o: f64, d: f64, lo: f64, hi: f64| {
        let (a, z) = ((lo - o) / d, (hi - o) / d);
        (a.min(z), a.max(z))
    };
    let s = [
        slab(o.x, d.x, 0.0, 10.0),
        slab(o.y, d.y, 0.0, 6.0),
        slab(o.z, d.z, 0.0, 3.0),
    ];
    let enter = s.iter().map(|x| x.0).fold(f64::MIN, f64::max);
    let leave = s.iter().map(|x| x.1).fold(f64::MAX, f64::min);
    assert_ts(&cast(&b, o, d, FAR), &[enter, leave]);
}

#[test]
fn a_ray_through_an_edge_crosses_there_once() {
    // From (−1, −1) along the diagonal at mid-height: in at the edge x = y = 0 after √2, out
    // through the face y = 6 at (6, 6) after 7√2.
    let b = block([0.0; 3], [10.0, 6.0, 3.0]);
    let d = Vec3::new(1.0, 1.0, 0.0);
    let hits = cast(&b, Vec3::new(-1.0, -1.0, 1.5), d, FAR);
    assert_ts(&hits, &[2.0f64.sqrt(), 7.0 * 2.0f64.sqrt()]);
}

#[test]
fn a_ray_along_a_torus_diameter_crosses_four_times_and_down_its_axis_none() {
    // Tube radius 2 at R = 10 round z: along x through the centre at x = ±8 and ±12.
    let xz = Frame::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap();
    let t = revolve(
        &Profile::circle(Vec2::new(10.0, 0.0), 4.0),
        &xz,
        Vec3::ZERO,
        Vec3::Z,
        TAU,
        "t",
    )
    .unwrap();
    let o = Vec3::new(-20.0, 0.0, 0.0);
    assert_ts(&cast(&t, o, Vec3::X, FAR), &[8.0, 12.0, 28.0, 32.0]);
    assert_ts(&cast(&t, Vec3::new(0.0, 0.0, -20.0), Vec3::Z, FAR), &[]);
}

#[test]
fn a_ray_across_a_spline_extrusion_crosses_where_the_outline_does() {
    // A disc whose outline is a periodic spline, 10 high, crossed at mid-height along y = 9: the
    // crossings are where the outline passes y = 9, found by sampling its curve and bisecting
    // each change of side, which shares nothing with the ray's search.
    let pts: Vec<Vec2> = (0..24)
        .map(|k| {
            let a = TAU * k as f64 / 24.0;
            Vec2::new(30.0 * a.cos(), 20.0 * a.sin() + 3.0 * (3.0 * a).sin())
        })
        .collect();
    let p = Profile::points(&pts, PointsInterp::PeriodicSpline, "side").unwrap();
    let disc = extrude(&p, &Frame::WORLD, 10.0, "disc").unwrap();
    let outline = disc
        .edge_ids()
        .map(|e| disc.edge(e))
        .find(|e| {
            let c = e.curve.as_ref().unwrap();
            let mid = c.eval(0.5 * (e.range.0 + e.range.1));
            mid.z.abs() < 1e-9 && !matches!(c, Curve::Line(_))
        })
        .unwrap();
    let c = outline.curve.as_ref().unwrap();
    let (t0, t1) = outline.range;
    let side = |t: f64| c.eval(t).y - 9.0;
    let n = 20_000;
    let at = |i: usize| t0 + (t1 - t0) * i as f64 / n as f64;
    let o = Vec3::new(-40.0, 9.0, 5.0);
    let mut want = Vec::new();
    for i in 0..n {
        let (mut lo, mut hi) = (at(i), at(i + 1));
        if side(lo).signum() == side(hi).signum() {
            continue;
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if side(mid).signum() == side(lo).signum() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        want.push(c.eval(0.5 * (lo + hi)).x - o.x);
    }
    want.sort_by(f64::total_cmp);
    assert!(want.len() >= 2, "{want:?}");
    assert_ts(&cast(&disc, o, Vec3::X, FAR), &want);
}

#[test]
fn a_ray_just_inside_a_curved_rim_crosses_both_ends() {
    // 0.2 µm inside a 3 mm disc's rim, between the outline's samples, where its chords cut 0.9 µm
    // inside the arc: the ends' polygons would leave the point out of both faces.
    let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
    let disc = extrude(&Profile::circle(Vec2::ZERO, 6.0), &frame, 4.0, "d").unwrap();
    let a = PI / 128.0;
    let r = 3.0 - 2e-4;
    let o = Vec3::new(r * a.cos(), r * a.sin(), -5.0);
    let hits = cast(&disc, o, Vec3::Z, FAR);
    assert_ts(&hits, &[5.0, 9.0]);
}

#[test]
fn rays_parallel_to_a_45_degree_cones_generators_cross_it_where_they_meet_it() {
    // A cylinder of radius 20, 10 tall, its side drafted 45° about its base: a frustum narrowing
    // to radius 10 at its top, its side a cone whose half-angle is stored as exactly π/4. At 45°
    // a cone's normal runs parallel to the generator across the axis, so a ray from 1 mm outside
    // along the inward normal meets the side once, at t = 1, and then leaves through the base,
    // as far again as the point is high, times √2.
    let cylinder = primitive::cylinder(&Frame::WORLD, 20.0, 10.0, "c").unwrap();
    let side = cylinder
        .face_ids()
        .find(|&f| matches!(cylinder.face(f).surface, Surface::Cylinder(_)))
        .unwrap();
    let frustum = edit::draft(&cylinder, &[side], &Frame::WORLD, FRAC_PI_4).unwrap();
    for z in [2.0, 4.0, 6.0, 8.0] {
        for k in 0..64 {
            let a = TAU * f64::from(k) / 64.0;
            let r = 20.0 - z;
            let (radial, n) = (
                Vec3::new(a.cos(), a.sin(), 0.0),
                Vec3::new(a.cos(), a.sin(), 1.0),
            );
            let n = n * FRAC_1_SQRT_2;
            let p = radial * r + Vec3::Z * z;
            let hits = cast(&frustum, p + n, -n, FAR);
            assert_ts(&hits, &[1.0, 1.0 + z * SQRT_2]);
        }
    }
}
