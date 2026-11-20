use std::f64::consts::PI;

use super::*;
use crate::{geom::NurbsCurve, measure, profile::PointsInterp};

fn at_z(z: f64) -> Frame {
    Frame::WORLD.translated(Vec3::Z * z)
}

fn sec(profile: Profile, frame: Frame) -> Section {
    Section { profile, frame }
}

fn volume(b: &Brep) -> f64 {
    measure::volume(b)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn ruled_rectangles_make_a_planar_frustum() {
    let b = loft(
        &[
            sec(Profile::rect(Vec2::ZERO, 20.0, 10.0), at_z(0.0)),
            sec(Profile::rect(Vec2::ZERO, 10.0, 6.0), at_z(4.0)),
        ],
        true,
        "lf",
    )
    .unwrap();
    assert_eq!(b.face_count(), 6);
    assert!(b.face_ids().all(|f| b.face(f).surface.kind() == "plane"));
    // Prismoid: h/6 (A0 + 4 Am + A1).
    let want = 4.0 / 6.0 * (200.0 + 4.0 * 15.0 * 8.0 + 60.0);
    assert!(rel(volume(&b), want) < 1e-12, "{}", volume(&b));
    let names = b.name_counts();
    assert_eq!(names["lf.side"], 4);
    assert_eq!(names["lf.start"], 1);
    assert_eq!(names["lf.end"], 1);
}

fn named_area(b: &Brep, name: &str) -> Vec<f64> {
    b.face_ids()
        .filter(|&f| b.face(f).names.iter().any(|n| n.to_string() == name))
        .map(|f| measure::face_area(b, f))
        .collect()
}

#[test]
fn a_side_carries_the_name_of_the_first_sections_edge() {
    let b = loft(
        &[
            sec(Profile::rect(Vec2::ZERO, 20.0, 10.0), at_z(0.0)),
            sec(Profile::rect(Vec2::ZERO, 10.0, 6.0), at_z(4.0)),
        ],
        true,
        "lf",
    )
    .unwrap();
    // Each side is a trapezoid: the two edges it joins, and a slant of the 4 mm rise and half the
    // change in the other dimension. The long sides lean 2 mm, the short ones 5 mm.
    let long = 0.5 * (20.0 + 10.0) * 20f64.sqrt();
    let short = 0.5 * (10.0 + 6.0) * 41f64.sqrt();
    for (name, want) in [
        ("lf.v_min", long),
        ("lf.v_max", long),
        ("lf.u_min", short),
        ("lf.u_max", short),
    ] {
        let got = named_area(&b, name);
        assert_eq!(got.len(), 1, "{name}");
        assert!(rel(got[0], want) < 1e-12, "{name}: {}", got[0]);
    }
}

#[test]
fn a_split_edge_gives_its_name_to_each_part() {
    let b = loft(
        &[
            sec(Profile::rect(Vec2::ZERO, 20.0, 20.0), at_z(0.0)),
            sec(Profile::circle(Vec2::ZERO, 20.0), at_z(30.0)),
        ],
        false,
        "lf",
    )
    .unwrap();
    // The circle starts on +x, which splits the square's +x edge at its middle.
    let names = b.name_counts();
    assert_eq!(names["lf.side"], 5);
    assert_eq!(names["lf.u_max"], 2);
    for whole in ["lf.u_min", "lf.v_min", "lf.v_max"] {
        assert_eq!(names[whole], 1, "{whole}");
    }
    let halves = named_area(&b, "lf.u_max");
    assert!(rel(halves[0], halves[1]) < 1e-9, "{halves:?}");
}

#[test]
fn parallel_sections_out_of_order_are_refused() {
    let circle = |r| Profile::circle(Vec2::ZERO, r);
    for zs in [[0.0, 6.0, 3.0], [0.0, 3.0, 3.0]] {
        let sections = zs.map(|z| sec(circle(10.0), at_z(z)));
        assert!(loft(&sections, false, "lf").is_err(), "{zs:?}");
    }
    let falling = [6.0, 3.0, 0.0].map(|z| sec(circle(10.0), at_z(z)));
    assert!(loft(&falling, false, "lf").is_ok());
}

#[test]
fn circles_loft_to_a_frustum() {
    let b = loft(
        &[
            sec(Profile::circle(Vec2::ZERO, 20.0), at_z(0.0)),
            sec(Profile::circle(Vec2::ZERO, 10.0), at_z(6.0)),
        ],
        false,
        "lf",
    )
    .unwrap();
    assert_eq!(b.face_count(), 3);
    let want = PI * 6.0 / 3.0 * (100.0 + 50.0 + 25.0);
    assert!(rel(volume(&b), want) < 1e-9, "{}", volume(&b));
}

#[test]
fn circle_to_square_splits_at_every_vertex_angle() {
    let b = loft(
        &[
            sec(Profile::circle(Vec2::ZERO, 20.0), at_z(0.0)),
            sec(Profile::rect(Vec2::ZERO, 20.0, 20.0), at_z(30.0)),
        ],
        false,
        "lf",
    )
    .unwrap();
    // The circle's start and the square's four corners: five sides. The circle's one edge is
    // `side`, so the sides take no second name.
    assert_eq!(b.face_count(), 7);
    let names: Vec<_> = b.name_counts().into_iter().collect();
    let want = [("lf.end", 1), ("lf.side", 5), ("lf.start", 1)].map(|(n, c)| (n.to_string(), c));
    assert_eq!(names, want);
    assert!(rel(volume(&b), 10667.102709516572) < 0.02, "{}", volume(&b));
    let bb = measure::bbox(&b);
    assert!((bb.min.x + 10.0).abs() < 1e-6 && (bb.max.z - 30.0).abs() < 1e-6);
}

/// An ellipse turned steadily along z: every z section is the same outline, so the volume is the
/// straight prism's to within the skinning's fit.
#[test]
fn twisted_sections_skin_smoothly() {
    let pts: Vec<Vec2> = (0..12)
        .map(|i| {
            let a = TAU * i as f64 / 12.0;
            Vec2::new(5.0 * a.cos() + 2.0, 3.0 * a.sin())
        })
        .collect();
    let p = Profile::points(&pts, PointsInterp::PeriodicSpline, "side").unwrap();
    let prism = volume(&crate::sweep::extrude(&p, &at_z(0.0), 10.0, "x").unwrap());
    let sections: Vec<Section> = (0..=4)
        .map(|i| {
            let z = 10.0 * i as f64 / 4.0;
            let th = 0.06 * z;
            let x = Vec3::new(th.cos(), th.sin(), 0.0);
            sec(p.clone(), Frame::new(Vec3::Z * z, Vec3::Z, x).unwrap())
        })
        .collect();
    let b = loft(&sections, false, "tw").unwrap();
    assert_eq!(b.face_count(), 3);
    assert!(rel(volume(&b), prism) < 1e-6, "{} {prism}", volume(&b));
}

/// A helical gear's tooth space, 1.4 mm across at its root, lofted through 9 sections from
/// z = -0.5 over `length`, turned `twist` rad/mm.
fn tooth_space(twist: f64, length: f64) -> Brep {
    let flank = |i: usize, s: f64| {
        let t = i as f64 / 12.0;
        let (r, a) = (21.5 - 4.5 * t, s * (0.09 - 0.05 * t * t));
        Vec2::new(r * a.cos(), r * a.sin())
    };
    let mut pts: Vec<Vec2> = (0..=12).map(|i| flank(i, -1.0)).collect();
    pts.extend((1..12).map(|i| {
        let a = -0.04 + 0.08 * i as f64 / 12.0;
        let r = 17.0 - 0.3 * (PI * i as f64 / 12.0).sin();
        Vec2::new(r * a.cos(), r * a.sin())
    }));
    pts.extend((0..=12).rev().map(|i| flank(i, 1.0)));
    let corner = 30f64.to_radians();
    let p = Profile::points(&pts, PointsInterp::Smooth { corner }, "space").unwrap();
    let sections: Vec<Section> = (0..=8)
        .map(|i| {
            let z = -0.5 + length * i as f64 / 8.0;
            let x = Vec3::new((twist * z).cos(), (twist * z).sin(), 0.0);
            sec(p.clone(), Frame::new(Vec3::Z * z, Vec3::Z, x).unwrap())
        })
        .collect();
    loft(&sections, false, "g").unwrap()
}

/// Points `off` from each face, on an `n`×`m` grid of its parameters, project back `off` away.
fn assert_projects_nearest(b: &Brep, off: f64, n: usize, m: usize) {
    for f in b.face_ids() {
        let s = &b.face(f).surface;
        for i in 0..=n {
            for j in 0..=m {
                let (u, v) = (i as f64 / n as f64, j as f64 / m as f64);
                let q = s.eval(u, v) + s.normal(u, v) * off;
                let uv = s.project(q);
                let d = s.eval(uv.x, uv.y).dist(q);
                assert!(d <= off + 1e-9, "face {f:?} at ({u}, {v}): {d}");
            }
        }
    }
}

#[test]
fn projection_finds_the_nearest_point_on_a_narrow_twisted_side() {
    // The root winds 1.4 rad round the axis, so seeds spaced along its length alone sit in other
    // points' basins.
    assert_projects_nearest(&tooth_space(0.2, 7.0), 0.01, 12, 12);
}

// A tooth space's wedge, 8.5 to 11.5 mm from the axis and 0.1 to 0.3 rad across, lofted through 9
// sections over 15.5 mm and turned 5 rad about the axis, as a helical gear's is: each section moves
// 6 mm, twice its width, from the one before. Matched by where their vertices lie in space, the
// sections paired each vertex with the next one round, and the skin came out inside out. Its
// volume is the wedge's area times the length, to the Bézier skin's departure from the helix
// between sections.
#[test]
fn sections_turned_further_than_their_width_keep_their_edges_matched() {
    let polar = [
        (8.5, 0.05),
        (8.5, -0.05),
        (9.5, -0.09),
        (10.5, -0.12),
        (11.5, -0.15),
        (11.5, 0.15),
        (10.5, 0.12),
        (9.5, 0.09),
    ];
    let pts: Vec<Vec2> = polar
        .iter()
        .map(|&(r, a): &(f64, f64)| Vec2::new(r * a.cos(), r * a.sin()))
        .collect();
    let area = 0.5 * (0..8).map(|i| pts[i].cross(pts[(i + 1) % 8])).sum::<f64>();
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let p = Profile::polygon(&pts, &names);
    let sections: Vec<Section> = (0..=8)
        .map(|i| {
            let z = 15.5 * i as f64 / 8.0;
            let th = 5.0 * z / 15.5;
            let x = Vec3::new(th.cos(), th.sin(), 0.0);
            sec(p.clone(), Frame::new(Vec3::Z * z, Vec3::Z, x).unwrap())
        })
        .collect();
    let b = loft(&sections, false, "g").unwrap();
    assert_eq!(b.face_count(), 10);
    let want = area * 15.5;
    assert!(rel(volume(&b), want) < 1e-4, "{}", rel(volume(&b), want));
}

// The same 20 x 10 rectangle in space, the second drawn 10 x 20 on a plane whose x is world y, or
// on one facing down the loft: matched in each section's own frame, each corner paired with the
// next one round and the 2000 mm³ prism came out a twisted 1333.33.
#[test]
fn sections_on_differently_turned_planes_match_where_they_lie() {
    for z in [Vec3::Z, -Vec3::Z] {
        for ruled in [true, false] {
            let b = loft(
                &[
                    sec(Profile::rect(Vec2::ZERO, 20.0, 10.0), at_z(0.0)),
                    sec(
                        Profile::rect(Vec2::ZERO, 10.0, 20.0),
                        Frame::new(Vec3::Z * 10.0, z, Vec3::Y).unwrap(),
                    ),
                ],
                ruled,
                "lf",
            )
            .unwrap();
            assert_eq!(b.face_count(), 6);
            assert!(
                rel(volume(&b), 2000.0) < 1e-12,
                "{z:?} {ruled} {}",
                volume(&b)
            );
        }
    }
}

#[test]
fn projection_finds_the_nearest_point_on_a_side_wound_past_a_turn() {
    // One span along the sections wound 6.6 rad: the root is about 110 mm long, so seeds a few
    // millimetres apart along it miss its basin.
    assert_projects_nearest(&tooth_space(0.6, 11.0), 1e-3, 4, 24);
}

// The quarter circle of radius 1 as a rational quadratic: a rail cut from it runs on the circle.
#[test]
fn a_rail_cut_from_a_rational_spline_stays_on_it() {
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let quarter = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
        Some(vec![1.0, w, 1.0]),
    )
    .unwrap();
    let curve = Curve::Nurbs(Box::new(quarter));
    let r = Rail::of_curve(&curve, 0.75, 0.25).unwrap();
    assert!(r.start().dist(curve.eval(0.75)) < 1e-12);
    assert!(r.end().dist(curve.eval(0.25)) < 1e-12);
    for k in 0..=10 {
        let s = f64::from(k) / 10.0;
        let mut h = [0.0; 4];
        for (b, p) in rail::basis(r.degree, &r.knots, s).iter().zip(&r.pw) {
            for i in 0..4 {
                h[i] += b * p[i];
            }
        }
        assert!((point(h).norm() - 1.0).abs() < 1e-12, "{s}");
    }
}

fn hollow_square(outside: f64, hole: f64) -> Profile {
    let hole = Profile::rect(Vec2::ZERO, hole, hole).loops.remove(0);
    Profile::rect(Vec2::ZERO, outside, outside).with_holes([hole])
}

// A tube tapering from 24 square with a 20 square hole to 12 with an 8 hole over 40: the outside
// frustum less the hole's, each h/3 · (A₀ + A₁ + √(A₀A₁)).
#[test]
fn a_hole_is_lofted_with_its_match() {
    let b = loft(
        &[
            sec(hollow_square(24.0, 20.0), at_z(0.0)),
            sec(hollow_square(12.0, 8.0), at_z(40.0)),
        ],
        true,
        "tube",
    )
    .unwrap();
    assert_eq!(b.face_count(), 10);
    let frustum = |a: f64, c: f64| 40.0 / 3.0 * (a * a + c * c + a * c);
    let want = frustum(24.0, 12.0) - frustum(20.0, 8.0);
    assert!(rel(volume(&b), want) < 1e-12, "{}", volume(&b));
    let caps = 24.0 * 24.0 - 400.0 + 144.0 - 64.0;
    let walls = 4.0 * (18.0 + 14.0) * (40f64 * 40.0 + 36.0).sqrt();
    assert!(rel(measure::area(&b), caps + walls) < 1e-12);
}

#[test]
fn sections_with_unequal_holes_are_refused() {
    let sections = [
        sec(hollow_square(24.0, 20.0), at_z(0.0)),
        sec(Profile::rect(Vec2::ZERO, 12.0, 12.0), at_z(40.0)),
    ];
    assert!(loft(&sections, true, "tube").is_err());
}

// A square to a point straight above its centre is a pyramid of four flat triangles, A·h/3; a
// circle to a point off its axis an oblique cone, its volume the same by Cavalieri.
#[test]
fn a_loft_to_a_point_closes_in_a_pyramid_or_a_cone() {
    let square = [sec(Profile::rect(Vec2::ZERO, 10.0, 10.0), at_z(0.0))];
    let b = loft_to_point(&square, Vec3::new(0.0, 0.0, 6.0), true, "py").unwrap();
    assert_eq!(b.face_count(), 5);
    assert!(b.face_ids().all(|f| b.face(f).surface.kind() == "plane"));
    assert!(rel(volume(&b), 200.0) < 1e-12, "{}", volume(&b));
    let slant = (25.0f64 + 36.0).sqrt();
    assert!(rel(measure::area(&b), 100.0 + 2.0 * 10.0 * slant) < 1e-12);

    let circle = [sec(Profile::circle(Vec2::ZERO, 10.0), at_z(0.0))];
    let b = loft_to_point(&circle, Vec3::new(3.0, 1.0, 6.0), true, "cone").unwrap();
    assert_eq!(b.face_count(), 2);
    let want = PI * 25.0 * 6.0 / 3.0;
    assert!(rel(volume(&b), want) < 1e-9, "{}", volume(&b));
}

// Two squares and an apex, smooth: the sides pass through every section and close in a point.
#[test]
fn a_smooth_loft_runs_through_its_sections_to_the_apex() {
    let sections = [
        sec(Profile::rect(Vec2::ZERO, 10.0, 10.0), at_z(0.0)),
        sec(Profile::rect(Vec2::ZERO, 12.0, 12.0), at_z(4.0)),
    ];
    let b = loft_to_point(&sections, Vec3::new(0.0, 0.0, 10.0), false, "sp").unwrap();
    assert_eq!(b.face_count(), 5);
    let bb = measure::bbox(&b);
    assert!(
        bb.max.x >= 6.0 - 1e-9 && (bb.max.z - 10.0).abs() < 1e-9,
        "{bb:?}"
    );
}

// A square 40 at z = 0 to the 28 square with its corners cut to 8√2 sides at z = 30: each side of
// the square to the side parallel to it (a trapezoid), each corner to the cut side facing it (a
// triangle). The level section at t = z/30 is the Minkowski mix (1 − t)S + tO, of area
// (1 − t)²|S| + 2t(1 − t)M + t²|O|, M = ½Σ h_S(n)·ℓ_O(n) the mixed area, so the volume is
// 30(|S| + M + |O|)/3.
#[test]
fn a_developable_loft_pairs_a_corner_with_a_side() {
    let o: Vec<Vec2> = [
        (6.0, -14.0),
        (14.0, -6.0),
        (14.0, 6.0),
        (6.0, 14.0),
        (-6.0, 14.0),
        (-14.0, 6.0),
        (-14.0, -6.0),
        (-6.0, -14.0),
    ]
    .iter()
    .map(|&(x, y)| Vec2::new(x, y))
    .collect();
    let b = loft_developable(
        &[
            sec(Profile::rect(Vec2::ZERO, 40.0, 40.0), at_z(0.0)),
            sec(Profile::polygon(&o, &["o"]), at_z(30.0)),
        ],
        "hood",
    )
    .unwrap();
    assert_eq!(b.face_count(), 10);
    assert!(b.face_ids().all(|f| b.face(f).surface.kind() == "plane"));
    let (s, oct) = (1600.0, 28.0 * 28.0 - 4.0 * 32.0);
    let mixed = 0.5 * (4.0 * 20.0 * 12.0 + 4.0 * 20.0 * 2f64.sqrt() * 8.0 * 2f64.sqrt());
    let want = 30.0 * (s + mixed + oct) / 3.0;
    assert!(rel(volume(&b), want) < 1e-12, "{}", volume(&b));
    let trapezoids = 4.0 * 0.5 * (40.0 + 12.0) * (36.0f64 + 900.0).sqrt();
    let triangles = 4.0 * 0.5 * 8.0 * 2f64.sqrt() * 1100f64.sqrt();
    let area = s + oct + trapezoids + triangles;
    assert!(
        rel(measure::area(&b), area) < 1e-12,
        "{}",
        measure::area(&b)
    );
}

// A square 40 at z = 0 to a circle of radius 15 at z = 30: each side to the circle's nearest
// point (a triangle), each corner to the quarter circle between (an oblique cone). The section at
// t is (1 − t)S + tD, of area (1 − t)²·1600 + t(1 − t)·160·15 + t²·225π.
#[test]
fn a_developable_loft_pairs_a_corner_with_an_arc() {
    let b = loft_developable(
        &[
            sec(Profile::rect(Vec2::ZERO, 40.0, 40.0), at_z(0.0)),
            sec(Profile::circle(Vec2::ZERO, 30.0), at_z(30.0)),
        ],
        "sq",
    )
    .unwrap();
    assert_eq!(b.face_count(), 10);
    let planes = b
        .face_ids()
        .filter(|&f| b.face(f).surface.kind() == "plane")
        .count();
    assert_eq!(planes, 6);
    let want = 30.0 * (1600.0 / 3.0 + 160.0 * 15.0 / 6.0 + 225.0 * PI / 3.0);
    assert!(rel(volume(&b), want) < 1e-9, "{}", volume(&b));
}

#[test]
fn a_developable_loft_refuses_a_section_that_is_not_convex() {
    let l: Vec<Vec2> = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 5.0),
        (5.0, 5.0),
        (5.0, 10.0),
        (0.0, 10.0),
    ]
    .iter()
    .map(|&(x, y)| Vec2::new(x, y))
    .collect();
    let sections = [
        sec(Profile::polygon(&l, &["l"]), at_z(0.0)),
        sec(Profile::rect(Vec2::new(5.0, 5.0), 10.0, 10.0), at_z(10.0)),
    ];
    assert!(loft_developable(&sections, "l").is_err());
}

/// The cubic Bézier through `ctrl` and its mirror images in the xz plane, the yz plane and both,
/// as guides over `[0, 1]`.
fn mirrored_guides(ctrl: [[f64; 3]; 4]) -> Vec<Guide> {
    [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
        .iter()
        .map(|&(sx, sy)| {
            let pts = ctrl
                .iter()
                .map(|c| Vec3::new(sx * c[0], sy * c[1], c[2]))
                .collect();
            let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
            Guide {
                curve: Curve::Nurbs(Box::new(NurbsCurve::new(3, knots, pts, None).unwrap())),
                range: (0.0, 1.0),
            }
        })
        .collect()
}

// Rectangles centred on z, their corners on four mirrored cubic guides (p, q, z)(t): the volume is
// ∫ 4pq dz over t, a polynomial, 426312/175; each wall is the guide's plane curve carried across,
// its area ∫ 2q·|(p′, z′)| dt, and these guides are Pythagorean-hodograph cubics,
// |(p′, z′)| = |(q′, z′)| = 3(9 + (1 − 3t)²), so the walls are ∫ 4(p + q)·3(9 + (1 − 3t)²) dt with
// p + q = 15.2: 4·15.2·30.
#[test]
fn a_loft_along_guides_follows_them_between_its_sections() {
    let b = loft_guided(
        &[
            sec(Profile::rect(Vec2::ZERO, 24.0, 6.4), at_z(0.0)),
            sec(Profile::rect(Vec2::ZERO, 6.0, 24.4), at_z(24.0)),
        ],
        &mirrored_guides([
            [12.0, 3.2, 0.0],
            [18.0, -2.8, 8.0],
            [15.0, 0.2, 19.0],
            [3.0, 12.2, 24.0],
        ]),
        None,
        "slab",
    )
    .unwrap();
    assert_eq!(b.face_count(), 6);
    assert!(rel(volume(&b), 426312.0 / 175.0) < 1e-9, "{}", volume(&b));
    let area = 24.0 * 6.4 + 6.0 * 24.4 + 4.0 * 15.2 * 30.0;
    assert!(rel(measure::area(&b), area) < 1e-9, "{}", measure::area(&b));
}

// A square at z = 0 closing to a point at (0, 0, 24) along four ridges (g, g, z)(t): the volume is
// ∫ 4g² dz = 329832/35, and the walls' area ∫ 8g·3(9 + (1 − 3t)²) dt = 10368/5.
#[test]
fn a_loft_along_guides_closes_at_its_apex() {
    let b = loft_guided(
        &[sec(Profile::rect(Vec2::ZERO, 18.0, 18.0), at_z(0.0))],
        &mirrored_guides([
            [9.0, 9.0, 0.0],
            [15.0, 15.0, 8.0],
            [12.0, 12.0, 19.0],
            [0.0, 0.0, 24.0],
        ]),
        Some(Vec3::new(0.0, 0.0, 24.0)),
        "spire",
    )
    .unwrap();
    assert_eq!(b.face_count(), 5);
    assert!(rel(volume(&b), 329832.0 / 35.0) < 1e-9, "{}", volume(&b));
    let area = 324.0 + 10368.0 / 5.0;
    assert!(rel(measure::area(&b), area) < 1e-9, "{}", measure::area(&b));
}

// A 10 × 2 rectangle at z = 0, 10 and 20, its corners on four helices turning once over the 20:
// every level section is the same rectangle turned, so the volume is 20·20 to within the helices'
// fit, whatever the middle section adds.
#[test]
fn a_loft_along_helical_guides_twists_its_section() {
    let screw = crate::sweep::Helix {
        origin: Vec3::ZERO,
        axis: Vec3::Z,
        rise: 20.0 / TAU,
        sign: 1.0,
    };
    let corners = [(5.0, -1.0), (5.0, 1.0), (-5.0, 1.0), (-5.0, -1.0)];
    let guides: Vec<Guide> = corners
        .iter()
        .map(|&(x, y)| Guide {
            curve: screw.spiral(Vec3::new(x, y, 0.0), TAU, 0.0).unwrap(),
            range: (0.0, TAU),
        })
        .collect();
    let turned = |z: f64| {
        let a = TAU * z / 20.0;
        sec(
            Profile::rect(Vec2::ZERO, 10.0, 2.0),
            Frame::new(Vec3::Z * z, Vec3::Z, Vec3::new(a.cos(), a.sin(), 0.0)).unwrap(),
        )
    };
    for sections in [
        vec![turned(0.0), turned(20.0)],
        vec![turned(0.0), turned(10.0), turned(20.0)],
    ] {
        let b = loft_guided(&sections, &guides, None, "blade").unwrap();
        assert_eq!(b.face_count(), 6);
        assert!(rel(volume(&b), 400.0) < 1e-9, "{}", volume(&b));
    }
}

#[test]
fn a_guide_that_misses_a_vertex_is_refused() {
    let mut guides = mirrored_guides([
        [12.0, 3.2, 0.0],
        [18.0, -2.8, 8.0],
        [15.0, 0.2, 19.0],
        [3.0, 12.2, 24.0],
    ]);
    guides[0].range = (0.0, 0.9);
    let sections = [
        sec(Profile::rect(Vec2::ZERO, 24.0, 6.4), at_z(0.0)),
        sec(Profile::rect(Vec2::ZERO, 6.0, 24.4), at_z(24.0)),
    ];
    assert!(loft_guided(&sections, &guides, None, "slab").is_err());
}

// A square of side 30 − z/2 from z = 0 to 40, turned half a turn on the way, its corners on four
// conical spirals: every level is a square, so the volume is ∫ s² dz = 52000/3.
#[test]
fn a_loft_along_spiral_guides_twists_and_tapers() {
    let screw = crate::sweep::Helix {
        origin: Vec3::ZERO,
        axis: Vec3::Z,
        rise: 40.0 / PI,
        sign: 1.0,
    };
    let growth = -10.0 * 2f64.sqrt() / PI;
    let guides: Vec<Guide> = [(15.0, 15.0), (-15.0, 15.0), (-15.0, -15.0), (15.0, -15.0)]
        .iter()
        .map(|&(x, y)| Guide {
            curve: screw.spiral(Vec3::new(x, y, 0.0), PI, growth).unwrap(),
            range: (0.0, PI),
        })
        .collect();
    let sections = [
        sec(Profile::rect(Vec2::ZERO, 30.0, 30.0), at_z(0.0)),
        sec(Profile::rect(Vec2::ZERO, 10.0, 10.0), at_z(40.0)),
    ];
    let b = loft_guided(&sections, &guides, None, "tw").unwrap();
    assert_eq!(b.face_count(), 6);
    assert!(rel(volume(&b), 52000.0 / 3.0) < 1e-9, "{}", volume(&b));
}

// A hollow square duct round a quarter turn of radius 30 about z, its half-width h running 9 → 6
// and its wall 2: each corner, at (30 ± h) from the axis and z = ±h, runs on a spiral. Pappus
// over the turn: ∫ 30·16(h − 1) dθ = 1560π.
#[test]
fn a_loft_along_guides_carries_a_hole_on_its_own_guides() {
    let dh = -6.0 / PI;
    let mut guides = Vec::new();
    for wall in [0.0, 2.0] {
        for (sr, sz) in [(1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let h = 9.0 - wall;
            let screw = crate::sweep::Helix {
                origin: Vec3::ZERO,
                axis: Vec3::Z,
                rise: sz * dh,
                sign: 1.0,
            };
            let p = Vec3::new(30.0 + sr * h, 0.0, sz * h);
            guides.push(Guide {
                curve: screw.spiral(p, PI / 2.0, sr * dh).unwrap(),
                range: (0.0, PI / 2.0),
            });
        }
    }
    let sections = [
        sec(
            hollow_square(18.0, 14.0),
            Frame::new(Vec3::new(30.0, 0.0, 0.0), Vec3::Y, Vec3::X).unwrap(),
        ),
        sec(
            hollow_square(12.0, 8.0),
            Frame::new(Vec3::new(0.0, 30.0, 0.0), -Vec3::X, Vec3::Y).unwrap(),
        ),
    ];
    let b = loft_guided(&sections, &guides, None, "duct").unwrap();
    assert_eq!(b.face_count(), 10);
    assert!(rel(volume(&b), 1560.0 * PI) < 1e-9, "{}", volume(&b));
}

// Data whose second derivative jumps at the middle, as a frame carried past a cubic path's
// breakpoint does: cubics joined there follow it at fourth order, halving the spacing dividing the
// error by about 16, and stay C¹ across the join.
#[test]
fn pieces_joined_where_the_data_bends_converge_at_fourth_order() {
    use super::rail::{Across, interpolation_pieces};
    let g = |u: f64| (u - 0.5) * (u - 0.5).abs();
    let f = |u: f64| [g(u), (2.0 * u).sin(), 0.0, 1.0];
    let fit = |per: usize| {
        let n = 2 * per;
        let us: Vec<f64> = (0..=n).map(|k| k as f64 / n as f64).collect();
        let data: Vec<_> = us.iter().map(|&u| f(u)).collect();
        let a: Across = interpolation_pieces(&us, per).unwrap();
        let c = a.apply(&data);
        let err = (0..n)
            .map(|k| {
                let u = (k as f64 + 0.5) / n as f64;
                let p = a.eval(&c, u);
                (0..2).map(|i| (p[i] - f(u)[i]).abs()).fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        // The slopes either side of the join, from the control points beside it.
        let j = per;
        let slope = |i: usize, h: f64| (c[i + 1][0] - c[i][0]) / h;
        let (left, right) = (a.knots[j + 1] - a.knots[j], a.knots[j + 4] - a.knots[j + 1]);
        (err, slope(j - 1, left), slope(j, right))
    };
    let errs: Vec<f64> = [8, 16, 32]
        .into_iter()
        .map(|per| {
            let (err, l, r) = fit(per);
            assert!((l - r).abs() < 1e-12, "slopes {l} and {r} at the join");
            err
        })
        .collect();
    for w in errs.windows(2) {
        assert!(w[0] / w[1] > 12.0, "errors {errs:?}");
    }
}
