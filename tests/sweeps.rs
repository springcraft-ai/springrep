//! Swept solids against closed-form volume, area and box, each checked valid and watertight.

use std::f64::consts::PI;

use springrep::{
    Error, check,
    math::{Frame, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    profile::{NamedSegment, PointsInterp, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
    topo::Brep,
};

const COARSE: MeshOptions = MeshOptions {
    linear: 0.05,
    angular: 0.3,
};

fn assert_solid(b: &Brep, volume: f64, area: f64, lo: [f64; 3], hi: [f64; 3], rel: f64) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
    let v = measure::volume(b);
    let a = measure::area(b);
    assert!((v - volume).abs() <= rel * volume, "volume {v} vs {volume}");
    assert!((a - area).abs() <= rel * area, "area {a} vs {area}");
    let bb = measure::bbox(b);
    let d = (bb.min - Vec3::from_array(lo))
        .max_abs()
        .max((bb.max - Vec3::from_array(hi)).max_abs());
    assert!(d < 1e-6, "bbox {bb:?}");
    let m = mesh::tessellate(b, &COARSE).unwrap();
    for s in b.solid_ids() {
        let r = mesh::report(b, &m, s);
        assert!(r.watertight(), "{r:?}");
        assert!(
            (r.volume - volume).abs() < 0.02 * volume,
            "mesh volume {}",
            r.volume
        );
    }
}

fn xz_frame() -> Frame {
    Frame::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap()
}

#[test]
fn box_extrusion_is_exact_and_named() {
    let b = extrude(
        &Profile::rect(Vec2::new(5.0, 2.0), 10.0, 4.0),
        &Frame::WORLD,
        3.0,
        "blk",
    )
    .unwrap();
    assert_solid(
        &b,
        120.0,
        2.0 * (40.0 + 30.0 + 12.0),
        [0.0, 0.0, 0.0],
        [10.0, 4.0, 3.0],
        1e-12,
    );
    let names = b.name_counts();
    for n in [
        "blk.start",
        "blk.end",
        "blk.v_min",
        "blk.u_max",
        "blk.v_max",
        "blk.u_min",
    ] {
        assert_eq!(names.get(n), Some(&1), "{n}");
    }
}

#[test]
fn plate_with_a_hole_either_way_along_its_normal() {
    let hole = Profile::circle(Vec2::new(2.0, 0.0), 2.0).loops.remove(0);
    let p = Profile::rect(Vec2::ZERO, 10.0, 6.0).with_holes([hole]);
    let frame = Frame::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::X).unwrap();
    let b = extrude(&p, &frame, 2.0, "p").unwrap();
    let v = 2.0 * (60.0 - PI);
    let a = 2.0 * (60.0 - PI) + 2.0 * 32.0 + 2.0 * PI * 2.0;
    assert_solid(&b, v, a, [-5.0, -3.0, -2.0], [5.0, 3.0, 0.0], 1e-12);
    let negative = extrude(&p, &Frame::WORLD, -2.0, "p").unwrap();
    assert_solid(&negative, v, a, [-5.0, -3.0, -2.0], [5.0, 3.0, 0.0], 1e-12);
    assert_eq!(negative.name_counts().get("p.start"), Some(&1));
}

#[test]
fn arcs_in_a_profile_sweep_cylinders_both_ways() {
    // A 10 × 10 square with a half-disc bump on the right and a half-disc bite on the left.
    let seg = |segment: Segment, name: &str| NamedSegment::new(segment, name);
    let lp = ProfileLoop {
        segments: vec![
            seg(
                Segment::Line {
                    a: Vec2::new(0.0, 0.0),
                    b: Vec2::new(10.0, 0.0),
                },
                "bottom",
            ),
            seg(
                Segment::Arc {
                    center: Vec2::new(10.0, 5.0),
                    radius: 5.0,
                    start: -PI / 2.0,
                    sweep: PI,
                },
                "bump",
            ),
            seg(
                Segment::Line {
                    a: Vec2::new(10.0, 10.0),
                    b: Vec2::new(0.0, 10.0),
                },
                "top",
            ),
            seg(
                Segment::Arc {
                    center: Vec2::new(0.0, 5.0),
                    radius: 3.0,
                    start: PI / 2.0,
                    sweep: -PI,
                },
                "bite_a",
            ),
        ],
    };
    // The bite's ends must meet the square's: the left side runs (0,10)→(0,8), arc, (0,2)→(0,0).
    let mut segments = lp.segments;
    segments.insert(
        3,
        seg(
            Segment::Line {
                a: Vec2::new(0.0, 10.0),
                b: Vec2::new(0.0, 8.0),
            },
            "l1",
        ),
    );
    segments[4] = seg(
        Segment::Arc {
            center: Vec2::new(0.0, 5.0),
            radius: 3.0,
            start: PI / 2.0,
            sweep: -PI,
        },
        "bite",
    );
    segments.push(seg(
        Segment::Line {
            a: Vec2::new(0.0, 2.0),
            b: Vec2::new(0.0, 0.0),
        },
        "l2",
    ));
    let p = Profile {
        loops: vec![ProfileLoop { segments }],
    };
    let b = extrude(&p, &Frame::WORLD, 4.0, "f").unwrap();
    let area2d = 100.0 + PI * 12.5 - PI * 4.5;
    let perim = 10.0 + PI * 5.0 + 10.0 + 2.0 + PI * 3.0 + 2.0;
    assert_solid(
        &b,
        4.0 * area2d,
        2.0 * area2d + 4.0 * perim,
        [0.0, 0.0, 0.0],
        [15.0, 10.0, 4.0],
        1e-12,
    );
}

#[test]
fn periodic_spline_disc_matches_its_ellipse() {
    let pts: Vec<Vec2> = (0..72)
        .map(|i| {
            let a = 2.0 * PI * i as f64 / 72.0;
            Vec2::new(30.0 * a.cos(), 20.0 * a.sin())
        })
        .collect();
    let p = Profile::points(&pts, PointsInterp::PeriodicSpline, "side").unwrap();
    let b = extrude(&p, &Frame::WORLD, 10.0, "disc").unwrap();
    assert!(check::check(&b).is_empty());
    let v = measure::volume(&b);
    assert!((v / (PI * 6000.0) - 1.0).abs() < 2e-5, "{v}");
    assert_eq!(b.face_count(), 3);
    let bb = measure::bbox(&b);
    assert!((bb.max.x - 30.0).abs() < 1e-3 && (bb.min.y + 20.0).abs() < 1e-3);
}

#[test]
fn partial_revolve_of_a_square() {
    let p = Profile::polygon(
        &[
            Vec2::new(10.0, 0.0),
            Vec2::new(20.0, 0.0),
            Vec2::new(20.0, 10.0),
            Vec2::new(10.0, 10.0),
        ],
        &["bottom", "outer", "top", "inner"],
    );
    let b = revolve(&p, &xz_frame(), Vec3::ZERO, Vec3::Z, PI / 2.0, "rv").unwrap();
    let v = PI * 300.0 * 10.0 / 4.0;
    let a = 200.0 + 2.0 * PI * 300.0 / 4.0 + 2.0 * PI * 200.0 / 4.0 + 2.0 * PI * 100.0 / 4.0;
    assert_solid(&b, v, a, [0.0, 0.0, 0.0], [20.0, 20.0, 10.0], 1e-12);
    assert_eq!(b.name_counts().len(), 6);
}

#[test]
fn partial_revolve_touching_the_axis_has_a_pole() {
    // A right triangle on the axis turned 120°: a cone sector closed by two caps.
    let p = Profile::polygon(
        &[
            Vec2::new(0.0, 0.0),
            Vec2::new(6.0, 0.0),
            Vec2::new(0.0, 8.0),
        ],
        &["base", "slant", "axis"],
    );
    let b = revolve(&p, &xz_frame(), Vec3::ZERO, Vec3::Z, 2.0 * PI / 3.0, "c").unwrap();
    let v = PI * 36.0 * 8.0 / 3.0 / 3.0;
    let a = (PI * 36.0 + PI * 6.0 * 10.0) / 3.0 + 2.0 * 24.0;
    assert_solid(&b, v, a, [-3.0, 0.0, 0.0], [6.0, 6.0, 8.0], 1e-12);
}

#[test]
fn full_revolve_of_a_half_disc_is_a_sphere() {
    let seg = |segment: Segment, name: &str| NamedSegment::new(segment, name);
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                seg(
                    Segment::Arc {
                        center: Vec2::ZERO,
                        radius: 5.0,
                        start: -PI / 2.0,
                        sweep: PI,
                    },
                    "ball",
                ),
                seg(
                    Segment::Line {
                        a: Vec2::new(0.0, 5.0),
                        b: Vec2::new(0.0, -5.0),
                    },
                    "axis",
                ),
            ],
        }],
    };
    let b = revolve(&p, &xz_frame(), Vec3::ZERO, Vec3::Z, 2.0 * PI, "s").unwrap();
    assert_solid(
        &b,
        4.0 / 3.0 * PI * 125.0,
        4.0 * PI * 25.0,
        [-5.0; 3],
        [5.0; 3],
        1e-11,
    );
}

#[test]
fn full_revolve_of_an_off_axis_circle_is_a_torus() {
    let p = Profile::circle(Vec2::new(10.0, 0.0), 4.0);
    let b = revolve(&p, &xz_frame(), Vec3::ZERO, Vec3::Z, 2.0 * PI, "tor").unwrap();
    let v = 2.0 * PI * PI * 10.0 * 4.0;
    let a = 4.0 * PI * PI * 10.0 * 2.0;
    assert_solid(&b, v, a, [-12.0, -12.0, -2.0], [12.0, 12.0, 2.0], 1e-11);
}

/// The lemon: the segment of the disk of radius `r` about (−d, 0) cut off by the axis x = 0, its
/// arc's circle across the axis from the arc, turned through `angle`.
fn lemon(r: f64, d: f64, angle: f64) -> Brep {
    let reach = (d / r).acos();
    let top = (r * r - d * d).sqrt();
    let p = Profile::outline(vec![
        NamedSegment::arc(Vec2::new(-d, 0.0), r, -reach, 2.0 * reach, "skin"),
        NamedSegment::line(Vec2::new(0.0, top), Vec2::new(0.0, -top), "axis"),
    ]);
    revolve(&p, &xz_frame(), Vec3::ZERO, Vec3::Z, angle, "lemon").unwrap()
}

#[test]
fn an_arc_whose_circle_lies_across_the_axis_turns_into_a_lemon() {
    let (r, d): (f64, f64) = (50.0, 30.0);
    let theta = (d / r).acos();
    let top = (r * r - d * d).sqrt();
    // The segment's area, and its first moment about its chord: 2π times that is the volume.
    let segment = r * r * theta - d * top;
    let moment = 2.0 / 3.0 * top.powi(3) - d * segment;
    let skin = 4.0 * PI * r * (r * theta.sin() - d * theta);
    let b = lemon(r, d, 2.0 * PI);
    assert_eq!(b.face_count(), 1);
    let w = r - d;
    assert_solid(
        &b,
        2.0 * PI * moment,
        skin,
        [-w, -w, -top],
        [w, w, top],
        1e-11,
    );
    // half a turn has the segment as its two flat ends
    let b = lemon(r, d, PI);
    assert_eq!(b.face_count(), 3);
    let (lo, hi) = ([-w, 0.0, -top], [w, w, top]);
    assert_solid(&b, PI * moment, skin / 2.0 + 2.0 * segment, lo, hi, 1e-11);
}

/// The closed uniform quadratic on the square of half-side `h` about `(cx, cy)`: four parabolic
/// arcs, each from the middle of one side to the middle of the next.
fn rounded_square(cx: f64, cy: f64, h: f64) -> Profile {
    let corners = [(h, h), (-h, h), (-h, -h), (h, -h)];
    let polygon: Vec<Vec3> = corners
        .iter()
        .map(|&(x, y)| Vec3::new(cx + x, cy + y, 0.0))
        .collect();
    let curve = springrep::geom::BSpline::periodic_uniform(2, &polygon).unwrap();
    let side = Segment::Spline {
        curve,
        reversed: false,
    };
    Profile::outline(vec![NamedSegment::new(side, "side")])
}

/// The area of [`rounded_square`] (the square less a third of each corner's triangle the arc cuts
/// off) and the length of its outline, four arcs h(1 + ln(1 + √2)/√2) long.
fn rounded_square_measures(h: f64) -> (f64, f64) {
    let area = 4.0 * h * h - 4.0 * (h * h / 2.0) / 3.0;
    let arc = h * (1.0 + (1.0 + 2f64.sqrt()).ln() / 2f64.sqrt());
    (area, 4.0 * arc)
}

#[test]
fn a_periodic_quadratic_profile_extrudes_and_turns_into_closed_sides() {
    let h = 10.0;
    let (section, outline) = rounded_square_measures(h);
    let b = extrude(&rounded_square(0.0, 0.0, h), &Frame::WORLD, 10.0, "prism").unwrap();
    let (lo, hi) = ([-h, -h, 0.0], [h, h, 10.0]);
    assert_solid(
        &b,
        10.0 * section,
        2.0 * section + 10.0 * outline,
        lo,
        hi,
        1e-9,
    );
    assert_eq!(b.face_count(), 3);
    // The section of half-side 3 about (10, 0), turned whole: Pappus on both.
    let h = 3.0;
    let (section, outline) = rounded_square_measures(h);
    let ring = rounded_square(10.0, 0.0, h);
    let b = revolve(&ring, &xz_frame(), Vec3::ZERO, Vec3::Z, 2.0 * PI, "ring").unwrap();
    let (lo, hi) = ([-13.0, -13.0, -h], [13.0, 13.0, h]);
    let path = 2.0 * PI * 10.0;
    assert_solid(&b, path * section, path * outline, lo, hi, 1e-9);
    assert_eq!(b.face_count(), 1);
}

#[test]
fn turned_stack_with_a_cone() {
    let pts = [
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (10.0, 10.0),
        (10.0, 28.0),
        (8.0, 30.0),
        (0.0, 30.0),
    ];
    let pts: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    let p = Profile::polygon(&pts, &["a", "b", "c", "d", "e", "f", "axis"]);
    let b = revolve(&p, &xz_frame(), Vec3::ZERO, Vec3::Z, 2.0 * PI, "t").unwrap();
    let frustum = PI * 2.0 / 3.0 * (100.0 + 80.0 + 64.0);
    let v = PI * 400.0 * 10.0 + PI * 100.0 * 18.0 + frustum;
    let a = 400.0 * PI
        + 400.0 * PI
        + 300.0 * PI
        + 360.0 * PI
        + 64.0 * PI
        + PI * 18.0 * 2.0 * 2f64.sqrt();
    assert_solid(&b, v, a, [-20.0, -20.0, 0.0], [20.0, 20.0, 30.0], 1e-12);
    assert_eq!(b.face_count(), 6);
}

#[test]
fn revolving_a_spline_makes_a_surface_of_revolution() {
    // A spline bulge on the outside of a tube section: the volume by Pappus would need the
    // spline's centroid, so check against the tessellation instead, and validity exactly.
    let seg = |segment: Segment, name: &str| NamedSegment::new(segment, name);
    let spline = springrep::geom::BSpline::interpolate(&[
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::new(12.0, 3.0, 0.0),
        Vec3::new(11.0, 7.0, 0.0),
        Vec3::new(10.0, 10.0, 0.0),
    ])
    .unwrap();
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                seg(
                    Segment::Line {
                        a: Vec2::new(5.0, 0.0),
                        b: Vec2::new(10.0, 0.0),
                    },
                    "bottom",
                ),
                seg(
                    Segment::Spline {
                        curve: spline,
                        reversed: false,
                    },
                    "bulge",
                ),
                seg(
                    Segment::Line {
                        a: Vec2::new(10.0, 10.0),
                        b: Vec2::new(5.0, 10.0),
                    },
                    "top",
                ),
                seg(
                    Segment::Line {
                        a: Vec2::new(5.0, 10.0),
                        b: Vec2::new(5.0, 0.0),
                    },
                    "bore",
                ),
            ],
        }],
    };
    let b = revolve(&p, &xz_frame(), Vec3::ZERO, Vec3::Z, 2.0 * PI, "r").unwrap();
    assert!(check::check(&b).is_empty());
    let v = measure::volume(&b);
    let m = mesh::tessellate(&b, &COARSE).unwrap();
    let r = mesh::report(&b, &m, b.solid_ids().next().unwrap());
    assert!(r.watertight(), "{r:?}");
    assert!((r.volume - v).abs() < 5e-3 * v, "{} vs {v}", r.volume);
    let bb = measure::bbox(&b);
    assert!(bb.max.x > 12.0 && bb.max.x < 12.5);
}

#[test]
fn bad_dimensions_are_invalid_input() {
    let rect = |w: f64| Profile::rect(Vec2::ZERO, w, 1.0);
    let bad = [
        extrude(&rect(f64::NAN), &Frame::WORLD, 1.0, "e"),
        extrude(&rect(1e-9), &Frame::WORLD, 1.0, "e"),
        extrude(&rect(1.0), &Frame::WORLD, f64::INFINITY, "e"),
        extrude(&rect(1.0), &Frame::WORLD, f64::NAN, "e"),
    ];
    for r in bad {
        assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
    }
}
