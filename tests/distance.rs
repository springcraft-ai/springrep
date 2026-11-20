//! Closest distances against closed forms: between bodies, faces, edges and vertices.

use std::f64::consts::{PI, TAU};

use springrep::{
    distance::{self, Closest, Selection},
    geom::Curve,
    math::{Frame, Point3, Vec2, Vec3},
    measure,
    profile::{NamedSegment, PointsInterp, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
    topo::{Brep, FaceId, VertexId},
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

fn bodies(a: &Brep, b: &Brep) -> Closest {
    distance::closest(a, &Selection::body(a), b, &Selection::body(b)).unwrap()
}

/// The distance, to rounding, and that both points lie that far apart.
fn assert_distance(c: Closest, want: f64) {
    assert!(
        (c.distance - want).abs() <= 1e-9 * (1.0 + want),
        "{c:?} vs {want}"
    );
    assert!(
        (c.on_a.dist(c.on_b) - c.distance).abs() <= 1e-12 * (1.0 + want),
        "{c:?}"
    );
}

fn face_at(b: &Brep, p: Point3) -> FaceId {
    b.face_ids()
        .min_by(|&x, &y| {
            let d = |f| measure::face_centroid(b, f).dist(p);
            d(x).total_cmp(&d(y))
        })
        .unwrap()
}

fn vertex_at(b: &Brep, p: Point3) -> VertexId {
    b.vertex_ids()
        .min_by(|&x, &y| {
            b.vertex(x)
                .point
                .dist(p)
                .total_cmp(&b.vertex(y).point.dist(p))
        })
        .unwrap()
}

/// Balls a few micrometres across that overlap, centres 3.4 µm apart and radii 4.1 and 2.7 µm: they
/// share points, so they are no distance apart, to rounding, however small the parts.
#[test]
fn overlapping_balls_micrometres_across_are_no_distance_apart() {
    let a = ball(Vec3::ZERO, 0.004054250852155359);
    let b = ball(
        Vec3::new(0.00338806114604008, 0.0, 0.0),
        0.0026689525243786373,
    );
    let d = bodies(&a, &b).distance;
    assert!(d <= 1e-15, "{d}");
}

#[test]
fn two_balls_are_their_centres_apart_less_both_radii() {
    let (c1, c2) = (Vec3::new(1.0, -2.0, 0.5), Vec3::new(9.0, 3.0, -1.5));
    let c = bodies(&ball(c1, 3.0), &ball(c2, 2.0));
    assert_distance(c, c1.dist(c2) - 5.0);
    // Each point lies on its ball, on the line through the centres.
    assert!((c.on_a.dist(c1) - 3.0).abs() < 1e-9 && (c.on_b.dist(c2) - 2.0).abs() < 1e-9);
}

#[test]
fn a_ball_by_a_box_meets_its_face_edge_or_corner() {
    // The box is x 0..10, y 0..6, z 0..3; a unit ball off each kind of place.
    let b = block([0.0; 3], [10.0, 6.0, 3.0]);
    // Off the face x = 10.
    assert_distance(bodies(&ball(Vec3::new(13.0, 3.0, 1.5), 1.0), &b), 2.0);
    // Past the edge x = 10, y = 6, where the face planes' nearest points lie off their faces.
    let edge = (2.0f64 * 2.0 + 2.0 * 2.0).sqrt() - 1.0;
    assert_distance(bodies(&ball(Vec3::new(12.0, 8.0, 1.5), 1.0), &b), edge);
    // Past the corner (10, 6, 3).
    let corner = 12.0f64.sqrt() - 1.0;
    assert_distance(bodies(&ball(Vec3::new(12.0, 8.0, 5.0), 1.0), &b), corner);
}

#[test]
fn boxes_set_corner_to_corner_meet_edge_to_edge() {
    // Unit cubes at (0..1)² and (2..3)² in plan, the same height: their vertical edges at (1, 1)
    // and (2, 2) are √2 apart.
    let c = bodies(
        &block([0.0; 3], [1.0; 3]),
        &block([2.0, 2.0, 0.0], [3.0, 3.0, 1.0]),
    );
    assert_distance(c, 2.0f64.sqrt());
}

#[test]
fn crossed_rods_are_their_axes_apart_less_both_radii() {
    // A rod of radius 1 along z, and one of radius 1.5 along x whose axis passes 4 from it.
    let a = extrude(
        &Profile::circle(Vec2::ZERO, 2.0),
        &Frame::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, Vec3::X).unwrap(),
        10.0,
        "a",
    )
    .unwrap();
    let b = extrude(
        &Profile::circle(Vec2::ZERO, 3.0),
        &Frame::new(Vec3::new(-5.0, 4.0, 0.0), Vec3::X, Vec3::Y).unwrap(),
        10.0,
        "b",
    )
    .unwrap();
    assert_distance(bodies(&a, &b), 4.0 - 1.0 - 1.5);
}

#[test]
fn a_ball_on_a_torus_axis_is_nearest_along_a_whole_ring() {
    // Tube radius 2 at R = 10 round z; a unit ball at height 6 on the axis is nearest every
    // point of the ring where the tube faces it: √(R² + h²) − a − r.
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
    let c = bodies(&ball(Vec3::new(0.0, 0.0, 6.0), 1.0), &t);
    assert_distance(c, (100.0f64 + 36.0).sqrt() - 3.0);
}

#[test]
fn a_ball_inside_a_box_or_through_it_is_no_distance_away() {
    let b = block([0.0; 3], [10.0, 6.0, 3.0]);
    // Inside, touching nothing: its boundary is a whole unit from the box's.
    assert_eq!(
        bodies(&ball(Vec3::new(5.0, 3.0, 1.5), 1.0), &b).distance,
        0.0
    );
    // Through the face x = 10.
    assert_distance(bodies(&ball(Vec3::new(10.5, 3.0, 1.5), 1.0), &b), 0.0);
}

#[test]
fn balls_sharing_volume_meet_at_a_point_inside_both() {
    let (c1, c2) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.5, 1.0, -0.5));
    for (a, b) in [
        (ball(c1, 2.0), ball(c2, 1.5)),
        (ball(c2, 1.5), ball(c1, 2.0)),
    ] {
        let c = bodies(&a, &b);
        assert_eq!(c.distance, 0.0);
        assert_eq!(c.on_a, c.on_b);
        let p = c.on_a;
        assert!(
            p.dist(c1) <= 2.0 + 1e-9 && p.dist(c2) <= 1.5 + 1e-9,
            "{c:?}"
        );
    }
}

#[test]
fn bars_crossed_with_no_corner_inside_the_other_are_no_distance_apart() {
    // A plus sign: every vertex of each bar lies outside the other, yet they share a cube.
    let a = block([-5.0, -1.0, 0.0], [5.0, 1.0, 2.0]);
    let b = block([-1.0, -5.0, -1.0], [1.0, 5.0, 3.0]);
    assert_distance(bodies(&a, &b), 0.0);
    assert_distance(bodies(&b, &a), 0.0);
}

#[test]
fn blocks_face_to_face_or_a_hair_apart_measure_the_gap() {
    let a = block([0.0; 3], [4.0, 3.0, 2.0]);
    for gap in [0.0, 1e-4, 3e-3] {
        let b = block([1.0, 0.5, 2.0 + gap], [3.0, 2.5, 5.0]);
        assert_distance(bodies(&a, &b), gap);
        assert_distance(bodies(&b, &a), gap);
    }
}

#[test]
fn a_rod_inside_one_a_hair_wider_is_no_distance_away() {
    // The inner rod's side is 3.3 µm inside the outer's, far closer than the meshes' chords, so
    // most of its boundary's points sit within a chord of the outer mesh; its ends do not.
    let frame = |z: f64| Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap();
    let outer = extrude(
        &Profile::circle(Vec2::ZERO, 6.6391),
        &frame(1.28),
        9.17,
        "o",
    )
    .unwrap();
    let inner = extrude(
        &Profile::circle(Vec2::ZERO, 6.6390934),
        &frame(1.38),
        5.64,
        "i",
    )
    .unwrap();
    assert_distance(bodies(&outer, &inner), 0.0);
    assert_distance(bodies(&inner, &outer), 0.0);
}

#[test]
fn faces_edges_and_vertices_measure_as_themselves() {
    let b = block([0.0; 3], [10.0, 6.0, 3.0]);
    let (bottom, top) = (
        face_at(&b, Vec3::new(5.0, 3.0, 0.0)),
        face_at(&b, Vec3::new(5.0, 3.0, 3.0)),
    );
    // The box's floor and roof are its height apart, though they bound one solid.
    let c = distance::closest(&b, &Selection::face(bottom), &b, &Selection::face(top)).unwrap();
    assert_distance(c, 3.0);
    // A corner of a second box set 1 above and 2 beyond, to the first's roof: its edge (10, 6) is
    // the nearest part of the roof.
    let other = block([12.0, 7.0, 4.0], [14.0, 9.0, 5.0]);
    let v = vertex_at(&other, Vec3::new(12.0, 7.0, 4.0));
    let c = distance::closest(&other, &Selection::vertex(v), &b, &Selection::face(top)).unwrap();
    assert_distance(c, (4.0f64 + 1.0 + 1.0).sqrt());
    // Floor and roof together, or every edge of the box, are as near as that same corner.
    let both = Selection::faces([bottom, top]);
    let c = distance::closest(&other, &Selection::vertex(v), &b, &both).unwrap();
    assert_distance(c, (4.0f64 + 1.0 + 1.0).sqrt());
    let edges = Selection::edges(b.edge_ids());
    let c = distance::closest(&other, &Selection::vertex(v), &b, &edges).unwrap();
    assert_distance(c, (4.0f64 + 1.0 + 1.0).sqrt());
}

#[test]
fn a_ball_beside_a_spline_extrusion_is_the_profile_curves_distance_away() {
    // A disc whose outline is a periodic spline, 10 high, and a unit ball at mid-height beside it:
    // the nearest point lies on the side at that height, so the distance is the plan distance
    // from the ball's centre to the outline less the radius. The outline's own distance comes
    // from sampling its curve densely and refining the nearest sample by golden sections, which
    // shares nothing with the search.
    let pts: Vec<Vec2> = (0..24)
        .map(|k| {
            let a = TAU * k as f64 / 24.0;
            Vec2::new(30.0 * a.cos(), 20.0 * a.sin() + 3.0 * (3.0 * a).sin())
        })
        .collect();
    let p = Profile::points(&pts, PointsInterp::PeriodicSpline, "side").unwrap();
    let disc = extrude(&p, &Frame::WORLD, 10.0, "disc").unwrap();
    let centre = Vec3::new(33.0, 9.0, 5.0);
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
    let plan = |t: f64| {
        let q = c.eval(t);
        (q.x - centre.x).hypot(q.y - centre.y)
    };
    let n = 200_000;
    let at = |i: usize| t0 + (t1 - t0) * i as f64 / n as f64;
    let k = (0..=n)
        .min_by(|&i, &j| plan(at(i)).total_cmp(&plan(at(j))))
        .unwrap();
    let (mut lo, mut hi) = (at(k.saturating_sub(1)), at((k + 1).min(n)));
    let g = 0.5 * (5.0f64.sqrt() - 1.0);
    for _ in 0..200 {
        let (x1, x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
        if plan(x1) < plan(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    let want = plan(0.5 * (lo + hi)) - 1.0;
    assert_distance(bodies(&ball(centre, 1.0), &disc), want);
}

/// A plate 30 square and 2 thick with a hole of radius 10 through its middle.
fn drilled_plate() -> Brep {
    use springrep::{boolean::difference, primitive};
    let plate = primitive::block(&Frame::WORLD, Vec3::new(30.0, 30.0, 2.0), "plate").unwrap();
    let at = Frame::new(Vec3::new(15.0, 15.0, -1.0), Vec3::Z, Vec3::X).unwrap();
    difference(
        &plate,
        &primitive::cylinder(&at, 10.0, 4.0, "hole").unwrap(),
    )
    .unwrap()
}

#[test]
fn a_point_just_inside_a_holes_rim_is_as_far_as_the_rim_not_on_the_face() {
    // In the plane of the plate's top, a few micrometres off the rim either way: inside the hole
    // the top face is not there and the rim is the nearest thing, and on the face the distance is
    // nothing. A chord of the rim's polygon cuts 12 µm inside the circle at this radius.
    let plate = drilled_plate();
    let boundary = distance::Nearest::boundary(&plate).unwrap();
    for (off, want) in [(-0.005, 0.005), (-0.001, 0.001), (0.003, 0.0)] {
        for turn in [0.1f64, 1.3, 2.9, 4.4] {
            let r = 10.0 + off;
            let p = Vec3::new(15.0 + r * turn.cos(), 15.0 + r * turn.sin(), 2.0);
            let got = boundary.to(p).unwrap().distance;
            assert!(
                (got - want).abs() < 1e-7,
                "{off} at {turn}: {got} against {want}"
            );
        }
    }
}

#[test]
fn a_dowel_in_a_hole_is_as_far_from_the_plate_as_its_clearance() {
    // A dowel through the hole with micrometres to spare all round, less than the chords of the
    // hole's mesh stand off its wall: the two solids share no point, and the gap is the clearance.
    use springrep::primitive;
    let plate = drilled_plate();
    let at = Frame::new(Vec3::new(15.0, 15.0, -3.0), Vec3::Z, Vec3::X).unwrap();
    for clearance in [0.002, 0.02, 0.2] {
        let dowel = primitive::cylinder(&at, 10.0 - clearance, 8.0, "dowel").unwrap();
        let c = distance::closest(
            &plate,
            &Selection::body(&plate),
            &dowel,
            &Selection::body(&dowel),
        )
        .unwrap();
        assert!(
            (c.distance - clearance).abs() < 1e-7,
            "clearance {clearance}: {}",
            c.distance
        );
    }
}

#[test]
fn the_face_at_a_point_just_inside_a_holes_rim_is_the_holes_wall() {
    // In the plane of the plate's top, 5 µm inside the hole: the top face is not there, and the
    // face whose own region holds its nearest point is the hole's wall, 5 µm away.
    let plate = drilled_plate();
    for turn in [0.1f64, 1.3, 2.9, 4.4] {
        let r = 10.0 - 0.005;
        let p = Vec3::new(15.0 + r * turn.cos(), 15.0 + r * turn.sin(), 2.0);
        let (face, d) = measure::face_at(&plate, p).unwrap();
        assert_eq!(plate.face(face).surface.kind(), "cylinder", "at {turn}");
        assert!((d - 0.005).abs() < 1e-9, "at {turn}: {d}");
        // And 5 µm outside it, on the plate: the top, at no distance.
        let r = 10.0 + 0.005;
        let p = Vec3::new(15.0 + r * turn.cos(), 15.0 + r * turn.sin(), 2.0);
        let (face, d) = measure::face_at(&plate, p).unwrap();
        assert_eq!(plate.face(face).surface.kind(), "plane", "at {turn}");
        assert!(d < 1e-9, "at {turn}: {d}");
    }
}
