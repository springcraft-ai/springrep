use std::f64::consts::{PI, TAU};

use super::*;
use crate::{
    geom::Cylinder,
    math::{Frame, Transform},
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
};

fn block(lo: [f64; 3], hi: [f64; 3]) -> Brep {
    let c = Vec2::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
    let frame = Frame::new(Vec3::new(0.0, 0.0, lo[2]), Vec3::Z, Vec3::X).unwrap();
    let p = Profile::rect(c, hi[0] - lo[0], hi[1] - lo[1]);
    extrude(&p, &frame, hi[2] - lo[2], "k").unwrap()
}

fn rod(base: [f64; 3], axis: [f64; 3], r: f64, len: f64) -> Brep {
    let frame = Frame::from_axis(Vec3::from_array(base), Vec3::from_array(axis)).unwrap();
    extrude(&Profile::circle(Vec2::ZERO, 2.0 * r), &frame, len, "h").unwrap()
}

/// The work of a boolean of `a` and `b` up to splitting faces.
fn staged<'a>(a: &'a Brep, b: &'a Brep) -> (Work<'a>, Vec<Vec<Vec<Vec<Trim>>>>) {
    let mut w = Work::new([a, b]);
    w.meet().unwrap();
    w.split_curves();
    w.dedup();
    let (frags, _, _) = w.split_faces().unwrap();
    (w, frags)
}

fn face_named(w: &Work, side: usize, name: &str) -> usize {
    (0..w.faces.len())
        .find(|&f| {
            let face = &w.faces[f];
            face.side == side
                && w.breps[side]
                    .face(face.id)
                    .names
                    .iter()
                    .any(|n| n.face == name)
        })
        .unwrap()
}

#[test]
fn points_are_located_inside_on_or_off_a_face() {
    let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let far = block([20.0, 0.0, 0.0], [30.0, 1.0, 1.0]);
    let (w, _) = staged(&a, &far);
    let top = &w.faces[face_named(&w, 0, "end")].dom;
    assert!(matches!(
        top.locate(Vec3::new(5.0, 5.0, 10.0)),
        Where::Inside(_)
    ));
    let near = 0.5 * tol::LINEAR;
    assert_eq!(
        top.locate(Vec3::new(10.0 - near, 5.0, 10.0)),
        Where::Boundary
    );
    let out = 2.0 * tol::LINEAR;
    assert_eq!(top.locate(Vec3::new(10.0 + out, 5.0, 10.0)), Where::Outside);
    assert_eq!(top.locate(Vec3::new(5.0, 5.0, 10.0 + out)), Where::Outside);
}

#[test]
fn a_seamed_face_finds_points_a_period_away() {
    let r = rod([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    let far = block([20.0, 0.0, 0.0], [30.0, 1.0, 1.0]);
    let (w, _) = staged(&r, &far);
    let side = &w.faces[face_named(&w, 0, "side")].dom;
    for u in [0.1, 3.0, TAU - 0.1] {
        let p = side.surface.eval(u, 4.0);
        assert!(matches!(side.locate(p), Where::Inside(_)), "{u}");
    }
    assert_eq!(side.locate(side.surface.eval(0.0, 4.0)), Where::Boundary);
}

#[test]
fn a_hole_tangent_to_a_wall_splits_the_cap_and_the_wall() {
    let a = block([0.0, 0.0, 0.0], [20.0, 20.0, 10.0]);
    let h = rod([5.0, 10.0, -1.0], [0.0, 0.0, 1.0], 5.0, 20.0);
    let (w, frags) = staged(&a, &h);
    let top = face_named(&w, 0, "end");
    // The disc, and the rest of the cap, whose boundary touches itself where the circle meets
    // the wall.
    assert_eq!(frags[top].len(), 2);
    assert!(frags[top].iter().all(|loops| loops.len() == 1));
    // The wall, cut along the line the hole touches it on, which runs its whole height.
    let wall = face_named(&w, 0, "u_min");
    assert_eq!(frags[wall].len(), 2);
}

#[test]
fn rays_count_crossings_of_a_box() {
    let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let far = block([20.0, 0.0, 0.0], [30.0, 1.0, 1.0]);
    let (w, _) = staged(&a, &far);
    let faces: Vec<(&Domain, Aabb)> = w.face_of[0]
        .iter()
        .map(|&f| (&w.faces[f].dom, w.faces[f].bbox))
        .collect();
    assert_eq!(
        classify::contains(&faces, Vec3::new(5.0, 5.0, 5.0)),
        Some(true)
    );
    assert_eq!(
        classify::contains(&faces, Vec3::new(15.0, 5.0, 5.0)),
        Some(false)
    );
    // On a face's plane but off the face: no crossing there.
    assert_eq!(
        classify::contains(&faces, Vec3::new(15.0, 5.0, 10.0)),
        Some(false)
    );
    assert_eq!(
        classify::contains(&faces, Vec3::new(5.0, 5.0, 10.0 - 1e-6)),
        Some(true)
    );
}

#[test]
fn the_operations_keep_the_pieces_they_should() {
    let on = |same| Class::On { same, partner: 7 };
    assert_eq!(choose(Op::Union, 0, Class::Out), Some((false, None)));
    assert_eq!(choose(Op::Union, 1, Class::In), None);
    assert_eq!(choose(Op::Union, 0, on(true)), Some((false, Some(7))));
    assert_eq!(choose(Op::Union, 1, on(true)), None);
    assert_eq!(choose(Op::Union, 0, on(false)), None);
    assert_eq!(choose(Op::Intersection, 1, Class::In), Some((false, None)));
    assert_eq!(choose(Op::Intersection, 0, on(false)), None);
    assert_eq!(choose(Op::Difference, 0, on(false)), Some((false, None)));
    assert_eq!(choose(Op::Difference, 0, on(true)), None);
    assert_eq!(choose(Op::Difference, 1, Class::In), Some((true, None)));
    assert_eq!(choose(Op::Difference, 1, Class::Out), None);
}

#[test]
fn coaxial_surfaces_in_opposite_frames_share_parameters_affinely() {
    let a = Surface::Cylinder(Cylinder {
        frame: Frame::WORLD,
        radius: 3.0,
    });
    let flipped = Frame::new(Vec3::new(0.0, 0.0, 7.0), -Vec3::Z, Vec3::new(1.0, 1.0, 0.0)).unwrap();
    let b = Surface::Cylinder(Cylinder {
        frame: flipped,
        radius: 3.0,
    });
    let m = sew::uv_affine(&a, &b).unwrap();
    for (u, v) in [(0.3, 1.0), (2.0, -4.0), (5.5, 0.0)] {
        let uv = m.point(Vec2::new(u, v));
        assert!(a.eval(uv.x, uv.y).dist(b.eval(u, v)) < 1e-12);
    }
}

#[test]
fn a_cut_through_a_pole_splits_the_pole() {
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
                    "s",
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
    let frame = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let ball = revolve(&p, &frame, Vec3::ZERO, Vec3::Z, TAU, "b").unwrap();
    let half = block([0.0, -9.0, -9.0], [9.0, 9.0, 9.0]);
    let mut w = Work::new([&ball, &half]);
    w.meet().unwrap();
    w.split_curves();
    w.dedup();
    let (frags, poles, _) = w.split_faces().unwrap();
    // The plane x = 0 meets the sphere in a meridian circle through both poles, which arrives at
    // each pole from two sides: each pole's stretch of parameter plane becomes three pieces.
    assert_eq!(poles.len(), 6);
    // Cut at u = π/2 and 3π/2, the face is three strips; the two either side of the seam are one
    // half of the ball, which sewing joins across the seam.
    let sphere = face_named(&w, 0, "s");
    assert_eq!(frags[sphere].len(), 3);
}

#[test]
fn a_branch_is_traced_both_ways_from_a_seed_in_its_middle() {
    let cyl = |axis: Vec3, radius: f64| {
        Surface::Cylinder(Cylinder {
            frame: Frame::from_axis(Vec3::ZERO, axis).unwrap(),
            radius,
        })
    };
    let (a, b) = (cyl(Vec3::X, 5.0), cyl(Vec3::Z, 6.0));
    // The half of the loop x = √(11 + z²) where y > 1: an open arc, seeded at its middle.
    let region = Aabb {
        min: Vec3::new(0.0, 1.0, -6.0),
        max: Vec3::new(7.0, 6.0, 6.0),
    };
    let seed = Vec3::new(11f64.sqrt(), 5.0, 0.0);
    let curves = march::trace(&a, &b, &region, &[seed]).unwrap().0;
    assert_eq!(curves.len(), 1);
    let c = &curves[0];
    let (p, q) = (c.curve.eval(c.range.0), c.curve.eval(c.range.1));
    assert!(p.y < 1.0 + 0.5 && q.y < 1.0 + 0.5, "{p:?} {q:?}");
    assert!(p.z * q.z < 0.0);
}

#[test]
fn solids_apart_from_the_other_input_pass_through() {
    let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let far = block([20.0, 0.0, 0.0], [30.0, 10.0, 10.0]);
    let u = union(&a, &far).unwrap();
    assert_eq!(u.solid_ids().count(), 2);
    assert_eq!(u.face_count(), 12);
    assert!((measure::volume(&u) - 2000.0).abs() < 1e-9);
    let d = difference(&a, &far).unwrap();
    assert_eq!((d.solid_ids().count(), d.face_count()), (1, 6));
    assert!((measure::volume(&d) - 1000.0).abs() < 1e-9);
    assert_eq!(intersection(&a, &far).unwrap().solid_ids().count(), 0);
    // One of two pins meets the block: only it is combined, the other is copied.
    let pins = union(
        &rod([5.0, 5.0, 5.0], [0.0, 0.0, 1.0], 1.0, 10.0),
        &rod([25.0, 5.0, 0.0], [0.0, 0.0, 1.0], 1.0, 10.0),
    )
    .unwrap();
    let joined = union(&a, &pins).unwrap();
    assert_eq!(joined.solid_ids().count(), 2);
    assert!(crate::check::check(&joined).is_empty());
    let want = 1000.0 + PI * 5.0 + PI * 10.0;
    assert!((measure::volume(&joined) - want).abs() < 1e-9 * want);
}

#[test]
fn a_torus_face_is_boxed_by_its_own_extremes_not_the_whole_torus() {
    let r = rod([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, 20.0);
    let rounded = crate::blend::fillet(&r, &[("h.end", "h.side")], 3.0, "f").unwrap();
    let far = block([50.0, 0.0, 0.0], [60.0, 1.0, 1.0]);
    let (w, _) = staged(&rounded, &far);
    let bb = w.faces[face_named(&w, 0, "round_0")].bbox;
    // The round is the upper outer quarter of a tube about the circle r 17 at z 17.
    let slack = 1e-4;
    assert!(
        (bb.min.z - 17.0).abs() < slack && (bb.max.z - 20.0).abs() < slack,
        "{bb:?}"
    );
    assert!(
        (bb.max.x - 20.0).abs() < slack && (bb.min.y + 20.0).abs() < slack,
        "{bb:?}"
    );
}

#[test]
fn unify_merges_no_run_of_planes_that_drifts_off_its_first() {
    // Fifty 20 mm edges, each turned 2e-11 rad from the last: every neighbour is within the
    // angular tolerance, but the run as a whole bends ten times past it.
    let mut pts = vec![Vec2::ZERO];
    let mut dir = 0.0f64;
    for _ in 0..50 {
        let p = pts[pts.len() - 1] + Vec2::new(dir.cos(), dir.sin()) * 20.0;
        pts.push(p);
        dir += 2e-11;
    }
    let end = pts[pts.len() - 1];
    pts.push(Vec2::new(end.x, 100.0));
    pts.push(Vec2::new(0.0, 100.0));
    let solid = extrude(&Profile::polygon(&pts, &["s"]), &Frame::WORLD, 5.0, "p").unwrap();
    let u = unify(&solid).unwrap();
    assert!(crate::check::check(&u).is_empty());
    assert!(u.face_count() < solid.face_count());
    let want = measure::volume(&solid);
    assert!((measure::volume(&u) - want).abs() < 1e-9 * want);
}

#[test]
fn an_edge_touching_a_face_meets_it_at_the_vertex_already_there() {
    use crate::loft::{Section, loft};
    let at_z = |z: f64| Frame::WORLD.translated(Vec3::Z * z);
    let section = |profile, z| Section {
        profile,
        frame: at_z(z),
    };
    let lofted = loft(
        &[
            section(Profile::circle(Vec2::ZERO, 20.0), 0.0),
            section(Profile::rect(Vec2::ZERO, 20.0, 20.0), 30.0),
        ],
        false,
        "lf",
    )
    .unwrap();
    // The box's edge along y at x 10, z 20 touches the loft's sides at their ruling through
    // (10, 0, 20), where the ruling pierces the box's floor. Rounding hides the gap for 3e-7 mm
    // either side, and the touch used to be found 1.3e-7 mm from the ruling's vertex.
    let b = block([0.0, -10.0, 20.0], [10.0, 10.0, 30.0]);
    let mut w = Work::new([&lofted, &b]);
    w.meet().unwrap();
    let at = Vec3::new(10.0, 0.0, 20.0);
    let close: Vec<Point3> = w
        .pool
        .iter()
        .copied()
        .filter(|p| p.dist(at) < 1e-5)
        .collect();
    assert_eq!(close.len(), 1, "{close:?}");
}

/// A cylinder with a radial hole, and a copy moved off every axis by a hair: the sides meet along
/// two lines at a sine of 3.5e-5, and the hole's fitted edges, which cross them, graze the copy's
/// side.
fn hair_moved() -> (Brep, Brep) {
    let body = rod([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, 20.0);
    let a = difference(&body, &rod([-30.0, 0.0, 10.0], [1.0, 0.0, 0.0], 2.0, 60.0)).unwrap();
    let mut b = a.clone();
    let shift = Vec3::new(0.0, 1.0, 1.0).unit().unwrap() * 1e-3;
    b.transform(&crate::math::Transform::translation(shift));
    (a, b)
}

#[test]
fn a_copy_moved_obliquely_by_a_hair_combines_with_its_original() {
    // The edges' meetings with the copy's side are placed on the lines the sides meet along.
    let (a, b) = hair_moved();
    let u = union(&a, &b).unwrap();
    assert!(crate::check::check(&u).is_empty());
    // The copy adds a sliver no thicker than the move over the body's boundary.
    let (va, extra) = (
        measure::volume(&a),
        measure::volume(&u) - measure::volume(&a),
    );
    assert!(
        extra > 0.0 && extra < 1e-3 * measure::area(&a),
        "{va} {extra}"
    );
}

#[test]
fn edges_meetings_are_placed_whatever_order_they_were_found_in() {
    let (a, b) = hair_moved();
    let placed = |reverse: bool| {
        let mut w = Work::new([&a, &b]);
        w.pierce();
        w.cross_edges();
        w.intersect_faces().unwrap();
        if reverse {
            w.hits.reverse();
        }
        w.place_hits();
        let mut pts: Vec<[f64; 3]> = w.pool.iter().map(|p| [p.x, p.y, p.z]).collect();
        pts.sort_by(|p, q| p.partial_cmp(q).unwrap());
        pts
    };
    assert_eq!(placed(false), placed(true));
}

#[test]
fn a_face_no_cut_reaches_keeps_its_memo_and_a_cut_one_does_not() {
    // A 10 mm cube with a hole of radius 1 down into its top: the floor keeps the input's memo,
    // the top is cut and starts afresh.
    let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    assert!(crate::check::check(&a).is_empty());
    let hole = rod([5.0, 5.0, 4.0], [0.0, 0.0, 1.0], 1.0, 10.0);
    let r = difference(&a, &hole).unwrap();
    let named = |b: &Brep, name: &str| {
        b.face_ids()
            .find(|&f| {
                b.face(f)
                    .names
                    .iter()
                    .any(|n| n.feature == "k" && n.face == name)
            })
            .unwrap()
    };
    let shared = |name: &str| {
        std::sync::Arc::ptr_eq(
            &a.shared_memo(named(&a, name)),
            &r.shared_memo(named(&r, name)),
        )
    };
    assert!(shared("start"));
    assert!(!shared("end"));
    assert!(r.memo(named(&r, "start")).clean.get().is_some());
    let exact = measure::volume(&r);
    let rough = measure::solid_volume_sign(&r, r.solid_ids().next().unwrap());
    assert!((exact - (1000.0 - 6.0 * PI)).abs() < 1e-9 * exact);
    assert!((rough - exact).abs() < 1e-6 * exact, "{rough} vs {exact}");
}

#[test]
fn a_piece_read_at_its_cut_lies_where_a_ray_puts_it() {
    // A slanted rod through a cube: every piece with a clean cut is classified both ways.
    let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = rod([-2.0, 3.0, 2.0], [1.0, 0.3, 0.4], 2.0, 20.0);
    let (w, frags) = staged(&a, &b);
    let mut local = 0;
    for (fi, pieces) in frags.into_iter().enumerate() {
        let face = &w.faces[fi];
        let other: Vec<(&Domain, Aabb)> = w.face_of[1 - face.side]
            .iter()
            .map(|&f| (&w.faces[f].dom, w.faces[f].bbox))
            .collect();
        for loops in pieces {
            let dom = Domain::new(face.dom.surface.clone(), face.dom.reversed, loops);
            let Some(inside) = w.local_class(fi, &dom) else {
                continue;
            };
            let uv = dom.interior_point().unwrap();
            let p = dom.surface.eval(uv.x, uv.y);
            assert_eq!(Some(inside), classify::contains(&other, p));
            local += 1;
        }
    }
    assert!(local >= 6, "{local}");
}

/// The plane `z = 0` and the surface `z = y·(x² − a²)` over `[-1, 1]²`: both hold the x axis, and
/// they touch only at `x = ±a`, crossing along the axis in between.
fn saddle_pair(a: f64) -> (Surface, Surface) {
    let plane = Surface::Plane(crate::geom::Plane {
        frame: Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap(),
    });
    // x² − a² on [-1, 1] as a quadratic Bézier in s = (x + 1)/2: values 1 − a², −1 − a², 1 − a².
    let c = [1.0 - a * a, -1.0 - a * a, 1.0 - a * a];
    let mut ctrl = Vec::new();
    for (i, ci) in c.iter().enumerate() {
        for y in [-1.0, 1.0] {
            ctrl.push(Vec3::new(i as f64 - 1.0, y, y * ci));
        }
    }
    let knots = (vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]);
    let saddle = crate::geom::NurbsSurface::new((2, 1), knots, (3, 2), ctrl, None).unwrap();
    (plane, Surface::Nurbs(Box::new(saddle)))
}

#[test]
fn two_touching_points_on_a_shared_line_are_two_crossings() {
    let (plane, saddle) = saddle_pair(0.05);
    let (p, q) = (Vec3::new(-0.05, 0.0, 0.0), Vec3::new(0.05, 0.0, 0.0));
    // Every point between them lies on both surfaces, but across the axis at x = 0 the saddle is
    // ±0.1·0.0025 mm from the plane, on opposite sides.
    assert!(!march::same_crossing(&plane, &saddle, p, q, 1.0));
    // A cylinder resting on the plane along the axis touches it all the way: one band.
    let resting = Surface::Cylinder(Cylinder {
        frame: Frame::new(Vec3::new(0.0, 0.0, 5.0), Vec3::X, Vec3::Y).unwrap(),
        radius: 5.0,
    });
    assert!(march::same_crossing(&plane, &resting, p, q, 1.0));
}

#[test]
fn branches_through_two_touching_points_a_step_apart_reach_both() {
    // The crossings at x = ±0.02 are 0.04 mm apart, within a step (0.078 mm here): merged into one,
    // the branches along x = ±0.02 fitted through a point off them and were lost.
    let (plane, saddle) = saddle_pair(0.02);
    let region = Aabb {
        min: Vec3::new(-0.9, -0.9, -1.0),
        max: Vec3::new(0.9, 0.9, 1.0),
    };
    let seeds = [
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(0.05, 0.5, 0.0),
        Vec3::new(-0.05, -0.5, 0.0),
    ];
    let (curves, crossings) = march::trace(&plane, &saddle, &region, &seeds).unwrap();
    assert_eq!(crossings.len(), 2);
    for x in [-0.02, 0.02] {
        let c = Vec3::new(x, 0.0, 0.0);
        assert!(crossings.iter().any(|p| p.dist(c) <= tol::LINEAR));
        // The line x = ±0.02 runs both ways out of its crossing.
        let ends: Vec<f64> = curves
            .iter()
            .flat_map(|k| [k.curve.eval(k.range.0), k.curve.eval(k.range.1)])
            .filter(|p| (p.x - x).abs() <= tol::LINEAR && p.y.abs() > 0.5)
            .map(|p| p.y)
            .collect();
        assert!(ends.iter().any(|&y| y > 0.0) && ends.iter().any(|&y| y < 0.0));
    }
}

// Solids of one arena are one shape to `check`: one passing through another, or lying wholly inside
// it, makes it invalid, though no vertex of one lies on the other.
#[test]
fn check_finds_solids_through_or_inside_one_another() {
    let shape = |other: Brep| {
        let mut b = block([0.0; 3], [10.0; 3]);
        b.append(&other);
        b
    };
    let crossed = shape(block([5.0; 3], [15.0; 3]));
    assert!(!crate::check::check(&crossed).is_empty());
    let inside = shape(block([4.0; 3], [6.0; 3]));
    let issues = crate::check::check(&inside);
    assert!(
        issues.iter().any(|i| i.message.contains("inside")),
        "{issues:?}"
    );
    assert!(crate::check::check(&shape(block([20.0; 3], [30.0; 3]))).is_empty());
}

/// A rod of radius 1 along x from the origin, its top line on z = 0, and one of radius 4 along y
/// from the origin, its bottom line on z = 0: the two sides bulge apart and touch only at the
/// origin, where both seams start.
fn rods_touching_at_a_vertex() -> (Brep, Brep) {
    let along_x = Frame::new(Point3::ZERO, Vec3::X, Vec3::Z).unwrap();
    let a = extrude(
        &Profile::circle(Vec2::new(-1.0, 0.0), 2.0),
        &along_x,
        2.0,
        "a",
    )
    .unwrap();
    let along_y = Frame::new(Point3::ZERO, Vec3::Y, -Vec3::Z).unwrap();
    let b = extrude(
        &Profile::circle(Vec2::new(-4.0, 0.0), 8.0),
        &along_y,
        3.0,
        "b",
    )
    .unwrap();
    (a, b)
}

#[test]
fn sides_that_touch_at_a_known_vertex_meet_only_there() {
    let (a, b) = rods_touching_at_a_vertex();
    assert!(
        b.vertex_ids()
            .any(|v| b.vertex(v).point.dist(Point3::ZERO) <= tol::LINEAR)
    );
    assert!(
        a.vertex_ids()
            .any(|v| a.vertex(v).point.dist(Point3::ZERO) <= tol::LINEAR)
    );
    let mut w = Work::new([&a, &b]);
    w.meet().unwrap();
    let (sa, sb) = (face_named(&w, 0, "side"), face_named(&w, 1, "side"));
    let cuts: Vec<f64> = w
        .bases
        .iter()
        .filter(|base| matches!(base.kind, BaseKind::Cut { fa, fb, .. } if (fa, fb) == (sa, sb)))
        .map(|base| base.bbox.diagonal())
        .collect();
    assert!(cuts.is_empty(), "{cuts:?}");
}

/// A box 10 on a side and a triangular prism along x through it whose edge at (y, z) = (5, 10) lies
/// on the box's top; its faces from that edge run down into the box and up out of it.
fn box_and_grooving_prism() -> (Brep, Brep) {
    let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let frame = Frame::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap();
    let tri = [
        Vec2::new(5.0, 10.0),
        Vec2::new(8.0, 7.0),
        Vec2::new(8.0, 13.0),
    ];
    let b = extrude(
        &Profile::polygon(&tri, &["low", "wall", "high"]),
        &frame,
        12.0,
        "g",
    )
    .unwrap();
    (a, b)
}

fn edge_at(b: &Brep, y: f64, z: f64) -> EdgeId {
    b.edge_ids()
        .find(|&e| {
            let x = b.edge(e);
            [x.start, x.end].iter().all(|&v| {
                let p = b.vertex(v).point;
                (p.y - y).abs() <= tol::LINEAR && (p.z - z).abs() <= tol::LINEAR
            })
        })
        .unwrap()
}

fn top_face(b: &Brep) -> FaceId {
    b.face_ids()
        .find(|&f| b.face(f).names.iter().any(|n| n.face == "end"))
        .unwrap()
}

#[test]
fn an_edge_lying_on_a_face_imprints_it() {
    // The groove takes the triangle (5, 10), (8, 7), (8, 10), 4.5 in area, along the box's 10: 45.
    let (a, b) = box_and_grooving_prism();
    let mut w = Work::new([&a, &b]);
    w.meet().unwrap();
    let edge = w.edge_base[1][edge_at(&b, 5.0, 10.0).index()];
    let top = w.face_of[0][top_face(&a).index()];
    assert!(w.on_face.contains(&(edge, top)));
    let out = difference(&a, &b).unwrap();
    assert!(
        (measure::volume(&out) - 955.0).abs() <= 1e-9,
        "{}",
        measure::volume(&out)
    );
    // Bottom, two notched ends, two sides, the top either side of the groove, and its two walls.
    assert_eq!(out.face_ids().count(), 9);
}

#[test]
fn a_fact_is_used_when_it_holds_and_refused_when_it_does_not() {
    let (a, b) = box_and_grooving_prism();
    let top = top_face(&a);
    let fact = |edge| Facts {
        on_face: vec![crate::topo::OnFace {
            side: 1,
            edge,
            face: top,
        }],
    };
    let out = boolean_with(&a, &b, Op::Difference, &fact(edge_at(&b, 5.0, 10.0))).unwrap();
    assert!((measure::volume(&out) - 955.0).abs() <= 1e-9);
    assert_eq!(out.face_ids().count(), 9);
    let wrong = boolean_with(&a, &b, Op::Difference, &fact(edge_at(&b, 8.0, 13.0)));
    assert!(matches!(wrong, Err(Error::InvalidInput(_))), "{wrong:?}");
}

#[test]
fn a_copy_moved_by_the_tolerance_is_the_same_solid() {
    // A holed block and its copy moved by exactly LINEAR, off every axis's rounding: every vertex,
    // edge and face coincides, so the pair is taken whole and never traced against itself.
    let a = difference(
        &block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        &rod([5.0, 5.0, -1.0], [0.0, 0.0, 1.0], 2.0, 12.0),
    )
    .unwrap();
    let mut b = a.clone();
    let d = tol::LINEAR / 3f64.sqrt();
    b.transform(&crate::math::Transform::translation(Vec3::new(d, d, d)));
    let u = union(&a, &b).unwrap();
    assert_eq!(u.face_count(), a.face_count());
    assert_eq!(measure::volume(&u), measure::volume(&a));
    assert_eq!(difference(&a, &b).unwrap().solid_ids().count(), 0);
    // Moved twice as far, it is another solid.
    let mut c = a.clone();
    c.transform(&crate::math::Transform::translation(Vec3::new(
        2.0 * d,
        2.0 * d,
        2.0 * d,
    )));
    assert!(!same::same_solid(&a, &c));
}

/// The box's top lies after a far block's faces in the arena, so the boolean, which passes the far
/// block through and combines copies of the rest, must renumber the fact to use it.
#[test]
fn a_fact_survives_solids_that_pass_through_being_split_off() {
    let (a, b) = box_and_grooving_prism();
    let mut body = block([50.0, 0.0, 0.0], [60.0, 10.0, 10.0]);
    body.append(&a);
    let top = body
        .face_ids()
        .filter(|&f| body.face(f).names.iter().any(|n| n.face == "end"))
        .last()
        .unwrap();
    let fact = |edge| Facts {
        on_face: vec![crate::topo::OnFace {
            side: 1,
            edge,
            face: top,
        }],
    };
    let out = boolean_with(&body, &b, Op::Difference, &fact(edge_at(&b, 5.0, 10.0))).unwrap();
    assert!((measure::volume(&out) - 1955.0).abs() <= 1e-9);
    let wrong = boolean_with(&body, &b, Op::Difference, &fact(edge_at(&b, 8.0, 13.0)));
    assert!(matches!(wrong, Err(Error::InvalidInput(_))), "{wrong:?}");
}

/// The far block is in both inputs, so it is taken from topology and the box is combined with the
/// tool on its own: the fact must follow the box's copy there too, and a wrong one is still caught.
#[test]
fn a_fact_survives_a_coincident_solid_being_taken_from_topology() {
    let (a, b) = box_and_grooving_prism();
    let far = block([50.0, 0.0, 0.0], [60.0, 10.0, 10.0]);
    let mut body = far.clone();
    body.append(&a);
    let mut tool = far;
    tool.append(&b);
    let top = body
        .face_ids()
        .filter(|&f| body.face(f).names.iter().any(|n| n.face == "end"))
        .last()
        .unwrap();
    let fact = |edge| Facts {
        on_face: vec![crate::topo::OnFace {
            side: 1,
            edge,
            face: top,
        }],
    };
    let out = boolean_with(
        &body,
        &tool,
        Op::Difference,
        &fact(edge_at(&tool, 5.0, 10.0)),
    )
    .unwrap();
    assert!((measure::volume(&out) - 955.0).abs() <= 1e-9);
    let wrong = boolean_with(
        &body,
        &tool,
        Op::Difference,
        &fact(edge_at(&tool, 8.0, 13.0)),
    );
    assert!(matches!(wrong, Err(Error::InvalidInput(_))), "{wrong:?}");
}

/// Rods of radius 2 and 2 + 1e-7 whose axes cross are one radius within the tolerance: their
/// common part is the Steinmetz solid, its four faces meeting at the two touching points. Moved
/// 1e-7 apart as well, their gaps add past the tolerance and the generators sweep instead: the
/// branches meet at the touching point below and pass each other above.
#[test]
fn rods_whose_radii_differ_by_the_tolerance_cross_as_equal_ones() {
    let a = rod([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0, 10.0);
    let r = 2.0 + tol::LINEAR;
    let b = rod([0.0, -5.0, 0.0], [0.0, 1.0, 0.0], r, 10.0);
    let out = intersection(&a, &b).unwrap();
    assert_eq!(out.face_ids().count(), 4);
    // 16/3 r³ for r = 2, its derivative 16 r² = 64 times half the radii's gap.
    assert!((measure::volume(&out) - (128.0 / 3.0 + 32.0 * tol::LINEAR)).abs() <= 1e-6);
    let b = rod([0.0, -5.0, tol::LINEAR], [0.0, 1.0, 0.0], r, 10.0);
    let out = intersection(&a, &b).unwrap();
    assert!(crate::check::check(&out).is_empty());
}

fn triangle(pts: [[f64; 2]; 3]) -> Profile {
    let pts = pts.map(|p| Vec2::new(p[0], p[1]));
    Profile::polygon(&pts, &["a", "b", "c"])
}

/// A rod along y whose cap meets a wedge's edge, so the rod's side keeps its seam only above the
/// wedge and runs on past the seam's line below it, its window wider than a period; a second
/// wedge then cuts the side where both images of a point lie in that window.
#[test]
fn a_cut_meets_a_face_that_runs_past_its_seam_at_the_image_inside_it() {
    let along_y = Frame::new(Vec3::new(0.0, 5.0, 0.0), Vec3::Y, Vec3::X).unwrap();
    let r = extrude(
        &Profile::circle(Vec2::new(-7.5, 0.0), 7.5),
        &along_y,
        10.0,
        "r",
    )
    .unwrap();
    let down = Frame::new(Vec3::new(0.0, 0.0, 10.0), -Vec3::Z, Vec3::X).unwrap();
    let wedge = triangle([[12.5, -7.5], [-7.5, -5.0], [-2.5, -10.0]]);
    let wedge = extrude(&wedge, &down, 12.5, "w").unwrap();
    let across = Frame::new(Vec3::new(2.5, 0.0, 0.0), -Vec3::X, Vec3::Y).unwrap();
    let cutter = triangle([[12.5, -5.0], [7.5, 2.5], [0.0, -2.5]]);
    let cutter = extrude(&cutter, &across, 17.5, "x").unwrap();
    let out = union(&union(&r, &wedge).unwrap(), &cutter).unwrap();
    assert!(crate::check::check(&out).is_empty());
}

/// A groove whose apex edge lies in the block's wall leaves the wall with no thickness along that
/// edge: the wall's material lies on both sides of the edge, and the groove's two sides meet there.
/// The block touches itself along it, the edge four faces use, and the groove is a void that
/// touches the outside only along that line, which is the solid's.
#[test]
fn a_groove_whose_edge_lies_in_a_wall_touches_the_wall_along_it() {
    let a = block([0.0; 3], [10.0; 3]);
    let along_y = Frame::new(Vec3::new(0.0, 2.0, 0.0), Vec3::Y, Vec3::X).unwrap();
    let groove = triangle([[0.0, -5.0], [3.0, -3.0], [3.0, -7.0]]);
    let groove = extrude(&groove, &along_y, 6.0, "g").unwrap();
    let out = difference(&a, &groove).unwrap();
    assert!(
        crate::check::check(&out).is_empty(),
        "{:?}",
        crate::check::check(&out)
    );
    assert_eq!(out.solid_ids().count(), 1);
    assert_eq!(out.radial_edges().count(), 1);
    assert!((crate::measure::volume(&out) - 964.0).abs() < 1e-9);
    assert_eq!(crate::measure::betti(&out), [1, 0, 1]);
}

/// A rod of radius 5 along x, its axis 3 below z = 0, meets z = 0 along y = ±4. A drill of radius
/// 4 down from z = 0 about (0, 0) cuts it, its rim tangent to y = ±4 at (0, ±4, 0) from inside: the
/// hole is a cavity reaching the rod's side at those two points. Each is a vertex loop in the
/// rod's side and a vertex of the rim, so the cavity is still a void.
#[test]
fn a_rim_touching_a_face_from_inside_touches_it_at_points() {
    let rod = rod([-10.0, 0.0, -3.0], [1.0, 0.0, 0.0], 5.0, 20.0);
    let drill = rod_down(4.0);
    let out = difference(&rod, &drill).unwrap();
    assert!(crate::check::check(&out).is_empty());
    assert_eq!(out.solid_ids().count(), 1);
    let points: Vec<Point3> = out
        .loop_ids()
        .filter(|&lp| out.is_vertex_loop(lp))
        .map(|lp| out.vertex(out.coedge_start(out.lp(lp).coedges[0])).point)
        .collect();
    assert_eq!(points.len(), 2);
    assert!(
        points
            .iter()
            .all(|p| p.x.abs() < 1e-9 && (p.y.abs() - 4.0).abs() < 1e-9)
    );
    assert!((crate::measure::volume(&out) - 468.0 * PI).abs() < 1e-9);
    assert_eq!(crate::measure::betti(&out), [1, 0, 1]);
}

fn rod_down(r: f64) -> Brep {
    let frame = Frame::new(Point3::ZERO, -Vec3::Z, Vec3::X).unwrap();
    extrude(&Profile::circle(Vec2::ZERO, 2.0 * r), &frame, 2.0, "d").unwrap()
}

fn faces_of<'w>(w: &'w Work, side: usize) -> Vec<(&'w Domain, Aabb)> {
    w.face_of[side]
        .iter()
        .map(|&f| (&w.faces[f].dom, w.faces[f].bbox))
        .collect()
}

fn plane_through(p: [f64; 3], normal: [f64; 3]) -> Surface {
    let n = Vec3::from_array(normal).unit().unwrap();
    Surface::Plane(crate::geom::Plane {
        frame: Frame::from_axis(Vec3::from_array(p), n).unwrap(),
    })
}

/// A rod of radius 1 along x touches the top of a box 10 on a side along y = 5: from above where
/// its axis is at z = 11, from below where it is at z = 9. A point on the line is on both solids'
/// boundaries; the side each surface bends to decides.
#[test]
fn a_piece_touching_a_face_along_a_line_lies_on_the_side_its_surface_bends_to() {
    let a = block([0.0; 3], [10.0; 3]);
    let p = Vec3::new(5.0, 5.0, 10.0);
    for (z, inside) in [(11.0, false), (9.0, true)] {
        let r = rod([-5.0, 5.0, z], [1.0, 0.0, 0.0], 1.0, 20.0);
        let (w, _) = staged(&a, &r);
        let side = &w.faces[face_named(&w, 1, "side")].dom.surface;
        assert_eq!(classify::touching(side, &faces_of(&w, 0), p), Some(inside));
        // The box's top lies outside the rod either way: the rod bends away from it.
        let top = &w.faces[face_named(&w, 0, "end")].dom.surface;
        assert_eq!(classify::touching(top, &faces_of(&w, 1), p), Some(false));
    }
}

/// A box 10 on a side; above it, a prism along x whose edge rests on the top along y = 5. The top
/// holds that edge, and both of the prism's faces rise away from it: the top lies outside.
#[test]
fn a_piece_holding_another_solids_edge_lies_where_its_plane_runs_past_the_wedge() {
    let a = block([0.0; 3], [10.0; 3]);
    let frame = Frame::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap();
    let tri = [
        Vec2::new(5.0, 10.0),
        Vec2::new(8.0, 13.0),
        Vec2::new(2.0, 13.0),
    ];
    let b = extrude(&Profile::polygon(&tri, &["a", "b", "c"]), &frame, 20.0, "p").unwrap();
    let (w, _) = staged(&a, &b);
    let p = Vec3::new(5.0, 5.0, 10.0);
    let top = &w.faces[face_named(&w, 0, "end")].dom.surface;
    assert_eq!(classify::touching(top, &faces_of(&w, 1), p), Some(false));
}

/// An L of two boxes has a concave edge along x at (y, z) = (10, 10). A plane through it at 45°
/// that both legs' material lies across is inside on both sides; one square to that runs in on one
/// side and out on the other, which is a crossing, not a touch.
#[test]
fn a_plane_through_a_concave_edge_is_inside_only_where_both_sides_run_in() {
    let upright = block([0.0; 3], [10.0, 10.0, 20.0]);
    let foot = block([0.0; 3], [10.0, 20.0, 10.0]);
    let l = union(&upright, &foot).unwrap();
    let far = block([40.0, 0.0, 0.0], [50.0, 1.0, 1.0]);
    let (w, _) = staged(&l, &far);
    let faces = faces_of(&w, 0);
    let p = [5.0, 10.0, 10.0];
    let across = plane_through(p, [0.0, 1.0, 1.0]);
    assert_eq!(
        classify::touching(&across, &faces, Vec3::from_array(p)),
        Some(true)
    );
    let along = plane_through(p, [0.0, 1.0, -1.0]);
    assert_eq!(
        classify::touching(&along, &faces, Vec3::from_array(p)),
        None
    );
}

/// The box's floor holds a rod along its whole length, tangent from inside: a face with no cut
/// whose middle lies on the contact. The union keeps the floor and the rod's 2.5 past the box:
/// 17.5 · 7.5 · 17.5 + π · 1.25² · 2.5.
#[test]
fn a_union_whose_floor_touches_a_rod_along_its_middle_is_decided_by_the_contact() {
    let a = block([5.0, 1.25, -3.75], [22.5, 8.75, 13.75]);
    let r = rod([5.0, 5.0, -2.5], [1.0, 0.0, 0.0], 1.25, 20.0);
    let out = union(&a, &r).unwrap();
    assert!(crate::check::check(&out).is_empty());
    let want = 17.5 * 7.5 * 17.5 + PI * 1.25 * 1.25 * 2.5;
    let v = measure::volume(&out);
    assert!((v - want).abs() <= 1e-9 * want, "{v} vs {want}");
}

/// A half rod of radius 5 along x, flat side down on z = 0, has an edge along (y, z) = (5, 0)
/// between its flat and its side. The plane y = 5 holds that edge and is tangent to the side
/// there: off the edge it runs up past the side, which bends away from it, and down below the
/// flat, so it lies outside.
#[test]
fn a_plane_tangent_to_one_face_of_an_edge_takes_that_faces_side_from_the_gap() {
    let full = rod([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0, 20.0);
    let half = difference(&full, &block([-1.0, -6.0, -6.0], [21.0, 6.0, 0.0])).unwrap();
    let far = block([40.0, 0.0, 0.0], [50.0, 1.0, 1.0]);
    let (w, _) = staged(&half, &far);
    let wall = plane_through([10.0, 5.0, 0.0], [0.0, 1.0, 0.0]);
    let p = Vec3::new(10.0, 5.0, 0.0);
    assert_eq!(classify::touching(&wall, &faces_of(&w, 0), p), Some(false));
}

#[test]
fn a_union_leaves_no_vertex_where_only_one_line_runs_through() {
    // Two unit cubes face to face are one 2 × 1 × 1 box: its long edges run straight through
    // x = 1, where the cubes met, so it has a box's 8 vertices and 12 edges.
    let u = union(
        &block([0.0; 3], [1.0; 3]),
        &block([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    )
    .unwrap();
    assert!(crate::check::check(&u).is_empty());
    assert_eq!(u.face_count(), 6);
    assert_eq!(u.vertex_ids().count(), 8);
    assert_eq!(u.edge_ids().count(), 12);
}

#[test]
fn a_union_keeps_a_vertex_where_a_third_edge_meets_it() {
    // A step: an L-shaped prism. The short cube's top edge ends where the tall one's side rises,
    // so those vertices stay, while the bottom's long edges run through x = 1: the L's six corners
    // front and back.
    let u = union(
        &block([0.0; 3], [1.0; 3]),
        &block([1.0, 0.0, 0.0], [2.0, 1.0, 2.0]),
    )
    .unwrap();
    assert!(crate::check::check(&u).is_empty());
    assert_eq!(u.vertex_ids().count(), 12);
    assert_eq!(u.edge_ids().count(), 18);
}

#[test]
fn tori_nearly_tangent_along_a_branch_meet_in_a_curve_that_runs_one_way() {
    // A torus section R200 r50, its tube from 0° to 150°, turned whole, united with its copy moved
    // 1e-3 along x. Near the tubes' tops the two cross at about 1e-7 rad, so points on both are
    // placed across the branch only to within a band far wider than the tolerance. The branch's
    // curve must still run one way along it: one that folds back gives the tube a face whose
    // boundary touches itself, which no mesher can triangulate.
    let ta = 150f64.to_radians();
    let end = Vec2::new(200.0 + 50.0 * ta.cos(), 50.0 * ta.sin());
    let seg = |segment, name: &str| NamedSegment::new(segment, name);
    let tube = Segment::Arc {
        center: Vec2::new(200.0, 0.0),
        radius: 50.0,
        start: 0.0,
        sweep: ta,
    };
    let profile = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                seg(
                    Segment::Line {
                        a: Vec2::new(200.0, 0.0),
                        b: Vec2::new(250.0, 0.0),
                    },
                    "bottom",
                ),
                seg(tube, "tube"),
                seg(
                    Segment::Line {
                        a: end,
                        b: Vec2::new(200.0, 0.0),
                    },
                    "cone",
                ),
            ],
        }],
    };
    let frame = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let a = revolve(&profile, &frame, Vec3::ZERO, Vec3::Z, TAU, "rv").unwrap();
    let mut b = a.clone();
    b.transform(&crate::math::Transform::translation(Vec3::new(
        1e-3, 0.0, 0.0,
    )));
    let u = union(&a, &b).unwrap();
    assert!(crate::check::check(&u).is_empty());
    let opt = crate::mesh::MeshOptions {
        linear: 0.01,
        angular: 0.1,
    };
    let mesh = crate::mesh::tessellate(&u, &opt).unwrap();
    let solid = u.solid_ids().next().unwrap();
    assert!(crate::mesh::report(&u, &mesh, solid).watertight());
}

fn ball(r: f64) -> Brep {
    let seg = |segment: Segment, name: &str| NamedSegment::new(segment, name);
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
                    "s",
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
    let frame = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    revolve(&p, &frame, Vec3::ZERO, Vec3::Z, TAU, "b").unwrap()
}

/// A solid every line along x crosses in one stretch, `area` its shadow on the yz plane, against
/// its copy moved `d` along x: the union gains `d·area` and the intersection loses it, to `rel` of
/// that (corners within the tolerance of each other are merged, which moves volume by about the
/// tolerance times the corner's stretch).
fn shifted_copy_combines(b: &Brep, area: f64, d: f64, rel: f64) {
    let v = measure::volume(b);
    let mut m = b.clone();
    m.transform(&crate::math::Transform::translation(Vec3::new(d, 0.0, 0.0)));
    let u = union(b, &m).unwrap();
    let c = intersection(b, &m).unwrap();
    for (got, want) in [
        (measure::volume(&u), v + d * area),
        (measure::volume(&c), v - d * area),
    ] {
        assert!((got - want).abs() <= rel * d * area, "{got} vs {want}");
    }
    assert!(crate::check::check(&u).is_empty() && crate::check::check(&c).is_empty());
}

#[test]
fn a_hair_moved_copy_of_a_rim_tangent_to_an_edge_combines_with_it() {
    // A unit ball with a unit box at its centre: on the box's bottom face the ball's rim touches
    // the box's edges at (1, 0, 0) and (0, 1, 0), where the copy 1e-5 along leaves a vertex on
    // the rim and the edge alike. Its shadow on yz is the disc and the square less a quarter disc.
    let b = union(&ball(1.0), &block([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])).unwrap();
    let v = 4.0 * PI / 3.0 + 1.0 - PI / 6.0;
    assert!((measure::volume(&b) - v).abs() < 1e-9);
    shifted_copy_combines(&b, PI + 1.0 - PI / 4.0, 1e-5, 1e-5);
}

#[test]
fn a_hair_moved_copy_of_a_boss_whose_arcs_end_side_by_side_combines_with_it() {
    // A 200 block with the 50 × 100 rectangle on its top turned up 90° about x = 200: a quarter
    // annulus from R150 to R200, 100 deep. The copy's arcs end beside the block's, running within
    // the tolerance of them for about a millimetre. Its shadow on yz: 200 × 200 and 100 × 200.
    let frame = Frame::new(Vec3::new(0.0, 0.0, 200.0), Vec3::Z, Vec3::X).unwrap();
    let rect = Profile::rect(Vec2::new(25.0, 100.0), 50.0, 100.0);
    let boss = revolve(
        &rect,
        &frame,
        Vec3::new(200.0, 0.0, 200.0),
        Vec3::Y,
        PI / 2.0,
        "r",
    );
    let b = union(
        &block([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        &boss.unwrap(),
    )
    .unwrap();
    let v = 8e6 + PI / 4.0 * (200.0f64.powi(2) - 150.0f64.powi(2)) * 100.0;
    assert!((measure::volume(&b) - v).abs() < 1e-9 * v);
    shifted_copy_combines(&b, 60000.0, 1e-5, 1e-5);
}

#[test]
fn splines_crossing_at_a_tiny_angle_meet_where_the_gap_turns_round() {
    // A gently bent spline and its copy turned 1e-7 rad about its middle cross there; a copy
    // that coincides, or one moved 5e-8 square to it, does not cross it.
    let pts: Vec<Vec3> = (0..9)
        .map(|k| {
            let x = f64::from(k);
            Vec3::new(x, 0.02 * x * x, 0.0)
        })
        .collect();
    let a = crate::geom::Curve::BSpline(crate::geom::BSpline::interpolate(&pts).unwrap());
    let range = a.domain();
    let mid = pts[4];
    let turned = a.transformed(&crate::math::Transform::rotation(mid, Vec3::Z, 1e-7));
    let met = csi::grazing(&a, range, &turned, range);
    assert_eq!(met.len(), 1, "{met:?}");
    assert!(met[0].dist(mid) < 1e-6, "{:?}", met[0]);
    assert!(csi::grazing(&a, range, &a.clone(), range).is_empty());
    let apart = a.transformed(&crate::math::Transform::translation(Vec3::new(
        0.0, 5e-8, 0.0,
    )));
    assert!(csi::grazing(&a, range, &apart, range).is_empty());
}

#[test]
fn a_hair_moved_copy_whose_meeting_passes_a_hair_from_its_poles_combines_with_it() {
    // A unit ball less the unit box in the −x octant. Moved 1e-6 along x, the two spheres meet in
    // a circle 5e-7 from the copy's poles, where the copy's boundary arc crosses the original
    // sphere at about 1e-6 rad; near the poles the original's seam runs along that arc. The
    // shadow on yz is the whole disc.
    // The ball turned from a profile in y = 0 whose x is along z, so its seam and its poles' parameter
    // lines lie where the arc's crossing needs them.
    let seg = |segment: Segment, name: &str| NamedSegment::new(segment, name);
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                seg(
                    Segment::Line {
                        a: Vec2::new(-1.0, 0.0),
                        b: Vec2::new(1.0, 0.0),
                    },
                    "axis",
                ),
                seg(
                    Segment::Arc {
                        center: Vec2::ZERO,
                        radius: 1.0,
                        start: 0.0,
                        sweep: PI,
                    },
                    "side",
                ),
            ],
        }],
    };
    let frame = Frame::new(Vec3::ZERO, Vec3::Y, Vec3::Z).unwrap();
    let sphere = revolve(&p, &frame, Vec3::ZERO, Vec3::Z, TAU, "s").unwrap();
    let b = difference(&sphere, &block([-1.0, 0.0, 0.0], [0.0, 1.0, 1.0])).unwrap();
    assert!((measure::volume(&b) - 7.0 * PI / 6.0).abs() < 1e-9);
    shifted_copy_combines(&b, PI, 1e-6, 1e-4);
}

fn combined_unsnapped(a: &Brep, b: &Brep, op: Op) -> Result<Brep> {
    combine(a, b, op, &Facts::default())
}

#[test]
fn faces_overlapping_by_less_than_twice_the_tolerance_leave_the_sliver_between() {
    // Unsnapped, as where a face lies within the band only partly: A is the cube 0..4; B starts `o`
    // inside A's face x = 4, its faces y = 0 and y = 4 on A's, from below A's floor to z = 3. Where
    // o lies between the tolerance and twice it, the sliver's sides on the shared faces are
    // narrower than the band a point cannot tell inside from on: o × 4 × 3 in common, A's 64 and B's
    // (4 + o) × 16 less that together, A's 64 less it alone.
    let a = block([0.0; 3], [4.0; 3]);
    for o in [1.2e-7, 1.6e-7, 1.99e-7, 3e-7] {
        let b = block([4.0 - o, 0.0, -1.0], [8.0, 4.0, 3.0]);
        let common = o * 4.0 * 3.0;
        for (name, op, want) in [
            ("common", Op::Intersection, common),
            ("union", Op::Union, 64.0 + (4.0 + o) * 16.0 - common),
            ("cut", Op::Difference, 64.0 - common),
        ] {
            let x = combined_unsnapped(&a, &b, op).unwrap_or_else(|e| panic!("{o} {name}: {e}"));
            assert!(crate::check::check(&x).is_empty(), "{o} {name}");
            let got = crate::measure::volume(&x);
            assert!((got - want).abs() <= 1e-11, "{o} {name}: {got} vs {want}");
        }
    }
}

/// A 4 × 4 square prism along z from `z0` to `z1`, turned `turn` about z, its top's edges
/// chamfered by `chamfer` where that is positive.
fn square_key(id: &str, turn: f64, (z0, z1): (f64, f64), chamfer: f64) -> Brep {
    let x = Vec3::new(turn.cos(), turn.sin(), 0.0);
    let f = Frame::new(Vec3::new(0.0, 0.0, z0), Vec3::Z, x).unwrap();
    let key = extrude(&Profile::rect(Vec2::ZERO, 4.0, 4.0), &f, z1 - z0, id).unwrap();
    if chamfer <= 0.0 {
        return key;
    }
    let end = format!("{id}.end");
    let sides = ["v_min", "u_max", "v_max", "u_min"].map(|s| format!("{id}.{s}"));
    let pairs: Vec<(&str, &str)> = sides.iter().map(|s| (end.as_str(), s.as_str())).collect();
    crate::blend::chamfer(&key, &pairs, chamfer, "ch").unwrap()
}

#[test]
fn a_key_in_a_pocket_of_its_own_size_a_hair_off_meets_the_wall_in_slivers() {
    // Unsnapped: a 4 × 4 key turned 30° in a pocket cut to the same square, through the wall's floor
    // at z = 0 to z = 3. Tilted θ about x and moved s across one side, its side runs
    // s − θ·cos 30°·z into the wall over the 3 mm of it, so they share 4 × (3s − 4.5·θ·cos 30°).
    // Chamfered 0.5 at its top, the sliver stops at z = 2.5. Unmoved, its sides stay within 5e-9 of
    // the walls, the tolerance's one plane, and the two only touch.
    let turn = 30f64.to_radians();
    let wall = combined_unsnapped(
        &block([-6.0, -6.0, 0.0], [6.0, 6.0, 6.0]),
        &square_key("pocket", turn, (-1.0, 3.2), 0.0),
        Op::Difference,
    )
    .unwrap();
    let side = Vec3::new(-turn.sin(), turn.cos(), 0.0);
    for (tilt, s, chamfer) in [
        (0.0, 1.6e-7, 0.0),
        (1.6e-9, 1.6e-7, 0.0),
        (1e-8, 5e-7, 0.0),
        (1.6e-9, 0.0, 0.0),
        (0.0, 1.6e-7, 0.5),
        (1.6e-9, 0.0, 0.5),
    ] {
        let mut key = square_key("key", turn, (-3.0, 3.0), chamfer);
        key.transform(&Transform::rotation(Vec3::ZERO, Vec3::X, tilt));
        key.transform(&Transform::translation(side * s));
        let x = combined_unsnapped(&wall, &key, Op::Intersection)
            .unwrap_or_else(|e| panic!("{tilt} {s} {chamfer}: {e}"));
        assert!(crate::check::check(&x).is_empty(), "{tilt} {s} {chamfer}");
        let height = if chamfer > 0.0 { 2.5 } else { 3.0 };
        let want = if s == 0.0 {
            0.0
        } else {
            4.0 * (height * s - 0.5 * height * height * tilt * turn.cos())
        };
        let got = crate::measure::volume(&x);
        assert!(
            (got - want).abs() <= 1e-11,
            "{tilt} {s} {chamfer}: {got} vs {want}"
        );
    }
}

/// Each face's and edge's index in its own solid's order, found by `pick` on its box.
fn local(w: &Work, side: usize, pick: impl Fn(&Aabb) -> bool) -> Vec<usize> {
    w.face_of[side]
        .iter()
        .enumerate()
        .filter(|&(_, &f)| pick(&w.faces[f].bbox))
        .map(|(k, _)| k)
        .collect()
}

#[test]
fn near_pairs_are_the_faces_and_edges_that_come_within_reach() {
    // A rod of radius 2 through the middle of a 10 mm block's top face, from 5 mm deep to 5 mm
    // above: its side crosses the top face, and everything else of it stays 3 mm or more from the
    // block's other faces and edges.
    let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = rod([5.0, 5.0, 5.0], [0.0, 0.0, 1.0], 2.0, 10.0);
    let mut w = Work::new([&a, &b]);
    w.find_near();
    let near = w.near.as_ref().unwrap();
    let at = |x: f64, y: f64| (x - y).abs() < 1e-3;
    let flat_at = |z: f64| move |bb: &Aabb| at(bb.min.z, z) && at(bb.max.z, z);
    let (top, bottom) = (
        local(&w, 0, flat_at(10.0))[0],
        local(&w, 0, flat_at(0.0))[0],
    );
    let wall = local(&w, 0, |bb| at(bb.max.x, 0.0))[0];
    let side = local(&w, 1, |bb| bb.max.z - bb.min.z > 9.0)[0];
    let (low_cap, high_cap) = (
        local(&w, 1, flat_at(5.0))[0],
        local(&w, 1, flat_at(15.0))[0],
    );
    assert!(near.faces(top, side));
    assert!(!near.faces(top, low_cap) && !near.faces(top, high_cap));
    for f in [side, low_cap, high_cap] {
        assert!(!near.faces(bottom, f) && !near.faces(wall, f));
    }
    let (ea, eb) = (a.edge_ids().count(), b.edge_ids().count());
    assert!((0..ea).all(|i| (0..eb).all(|j| !near.edges(i, j))));
    // Of the rod's edges, only the seam line from its low cap to its high one crosses the top.
    let crossing: Vec<usize> = (0..eb).filter(|&e| near.edge_face(1, e, top)).collect();
    assert_eq!(crossing.len(), 1);
    let seam = b.edge(b.edge_ids().nth(crossing[0]).unwrap());
    assert!(matches!(seam.curve, Some(Curve::Line(_))));
    assert!((0..eb).all(|e| !near.edge_face(1, e, bottom)));
}

#[test]
fn ruling_out_far_pairs_leaves_every_result_as_it_was() {
    let rodded = (
        block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        rod([5.0, 5.0, 5.0], [0.0, 0.0, 1.0], 2.0, 10.0),
    );
    for (a, b) in [hair_moved(), rodded, box_and_grooving_prism()] {
        for op in [Op::Union, Op::Intersection, Op::Difference] {
            let run = |find: bool| {
                let mut w = Work::new([&a, &b]);
                if find {
                    w.find_near();
                    assert!(w.near.is_some());
                }
                w.finish(op, &Facts::default()).unwrap()
            };
            let (with, without) = (run(true), run(false));
            assert_eq!(with.face_count(), without.face_count(), "{op:?}");
            assert_eq!(measure::volume(&with), measure::volume(&without), "{op:?}");
        }
    }
}

#[test]
fn a_face_hull_box_holds_its_loose_box() {
    use crate::loft::{Section, loft};
    let half_ball = {
        let line = Segment::Line {
            a: Vec2::new(-1.0, 0.0),
            b: Vec2::new(1.0, 0.0),
        };
        let arc = Segment::Arc {
            center: Vec2::ZERO,
            radius: 1.0,
            start: 0.0,
            sweep: PI,
        };
        let p = Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    NamedSegment::new(line, "axis"),
                    NamedSegment::new(arc, "side"),
                ],
            }],
        };
        let frame = Frame::new(Vec3::ZERO, Vec3::Y, Vec3::Z).unwrap();
        let ball = revolve(&p, &frame, Vec3::ZERO, Vec3::Z, TAU, "s").unwrap();
        difference(&ball, &block([-1.0, 0.0, 0.0], [0.0, 1.0, 1.0])).unwrap()
    };
    let rounded = {
        let r = rod([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0, 20.0);
        crate::blend::fillet(&r, &[("h.end", "h.side")], 3.0, "f").unwrap()
    };
    let lofted = {
        let section = |profile, z: f64| Section {
            profile,
            frame: Frame::WORLD.translated(Vec3::Z * z),
        };
        loft(
            &[
                section(Profile::circle(Vec2::ZERO, 20.0), 0.0),
                section(Profile::rect(Vec2::ZERO, 20.0, 20.0), 30.0),
            ],
            false,
            "lf",
        )
        .unwrap()
    };
    for mut x in [half_ball, rounded, lofted, hair_moved().0] {
        // Moving the solid clears its memos, so the hull box is worked out, not the memo's.
        x.transform(&Transform::translation(Vec3::new(1.0, 2.0, 3.0)));
        for f in x.face_ids() {
            let hull = measure::face_hull_box(&x, f);
            let loose = measure::face_box(&x, f);
            assert!(
                (0..3).all(|k| hull.min[k] <= loose.min[k] && loose.max[k] <= hull.max[k]),
                "{} face {f:?}: {hull:?} does not hold {loose:?}",
                x.face(f).surface.kind()
            );
        }
    }
}

#[test]
fn a_copy_moved_across_its_holes_axis_by_just_over_the_tolerance_is_the_same_solid() {
    // A 40 mm block with a Ø10 hole along y, and a copy moved across the hole's axis. Snapped onto
    // the block, the copy's hole wall kept its seam as far round the circle as the copy was moved,
    // and the boolean cut a piece of edge just over the tolerance long between the two seams.
    let a = difference(
        &block([-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]),
        &rod([0.0, -30.0, 0.0], [0.0, 1.0, 0.0], 5.0, 60.0),
    )
    .unwrap();
    let v = measure::volume(&a);
    for d in [1.1e-7, 2e-7, 5e-7] {
        let mut b = a.clone();
        b.transform(&Transform::translation(Vec3::new(d, 0.0, 0.0)));
        // The copy sits in the block's place: what it adds or takes is at most the move times the
        // area it moves across.
        let slack = d * 2.0 * 40.0 * 40.0;
        let u = union(&a, &b).unwrap();
        let i = intersection(&a, &b).unwrap();
        let c = difference(&a, &b).unwrap();
        for x in [&u, &i, &c] {
            assert!(crate::check::check(x).is_empty(), "{d}");
        }
        assert!((measure::volume(&u) - v).abs() <= slack, "{d}");
        assert!((measure::volume(&i) - v).abs() <= slack, "{d}");
        assert!(measure::volume(&c).abs() <= slack, "{d}");
    }
}

#[test]
fn a_key_moved_by_exactly_the_tolerance_in_its_pocket_only_touches_it() {
    // A 4 mm square key in a pocket of its own size, both turned 30° about z so their coordinates
    // round, the key moved across one side by exactly the tolerance: its side lies on the pocket's
    // wall within the tolerance with a few ulps to spare, where some tests read it within and
    // others just past.
    let turn = 30f64.to_radians();
    let square = |id: &str, z0: f64, h: f64| {
        let x = Vec3::new(turn.cos(), turn.sin(), 0.0);
        let f = Frame::new(Vec3::new(0.0, 0.0, z0), Vec3::Z, x).unwrap();
        extrude(&Profile::rect(Vec2::ZERO, 4.0, 4.0), &f, h, id).unwrap()
    };
    let wall = difference(
        &block([-6.0, -6.0, 0.0], [6.0, 6.0, 6.0]),
        &square("pocket", -1.0, 4.2),
    )
    .unwrap();
    let mut key = square("key", -3.0, 6.0);
    let side = Vec3::new(-turn.sin(), turn.cos(), 0.0);
    key.transform(&Transform::translation(side * tol::LINEAR));
    let x = intersection(&wall, &key).unwrap();
    assert!(crate::check::check(&x).is_empty());
    assert!(measure::volume(&x).abs() < 1e-5, "{}", measure::volume(&x));
}

#[test]
fn a_face_flush_inside_another_turned_a_hair_combines() {
    // A box with a face flush inside a 4 mm cube's, turned 5e-6 rad about an oblique axis: the two
    // faces cross along a diagonal, and an edge on it lies at the tolerance plus a rounding error
    // from its face, which the checker read as past it.
    let bx = |lo: [f64; 3], hi: [f64; 3]| block(lo, hi);
    let a = bx([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
    let mut b = bx([2.0, 1.0, 1.0], [4.0, 3.0, 3.0]);
    let axis = Vec3::new(0.0, 0.6, 0.8);
    b.transform(&Transform::rotation(Vec3::new(4.0, 2.2, 1.9), axis, 5e-6));
    let (u, i, c) = (
        union(&a, &b).unwrap(),
        intersection(&a, &b).unwrap(),
        difference(&a, &b).unwrap(),
    );
    for x in [&u, &i, &c] {
        assert!(crate::check::check(x).is_empty());
    }
    // The turn swings the box's far corners out of the cube by at most 5e-6 times 2 mm.
    let (vu, vi, vc) = (
        measure::volume(&u),
        measure::volume(&i),
        measure::volume(&c),
    );
    assert!(
        (vu - 64.0).abs() < 1e-4 && (vi - 8.0).abs() < 1e-4,
        "{vu} {vi}"
    );
    assert!(
        (vi + vc - 64.0).abs() < 1e-9 && (vu - vc - 8.0).abs() < 1e-9,
        "{vu} {vi} {vc}"
    );
}

/// Two balls touching at a point: the point is a vertex loop in a face of each, and so a vertex
/// two fans of faces meet at, one round each loop.
#[test]
fn balls_touching_at_a_point_share_a_vertex_of_two_fans() {
    let mut b = ball(5.0);
    b.transform(&Transform::translation(Vec3::new(10.0, 0.0, 0.0)));
    let u = union(&ball(5.0), &b).unwrap();
    let at = u
        .vertex_ids()
        .find(|&v| u.vertex(v).point.dist(Point3::new(5.0, 0.0, 0.0)) < 1e-9)
        .expect("a vertex at the contact");
    assert_eq!(crate::check::vertex_fans(&u).get(&at), Some(&2));
}

/// A vertex loop survives the codec: the point is still a vertex both balls' faces share.
#[test]
fn a_point_contact_round_trips_through_the_codec() {
    use crate::codec::{Codec, Reader, Writer};
    let mut b = ball(5.0);
    b.transform(&Transform::translation(Vec3::new(10.0, 0.0, 0.0)));
    let u = union(&ball(5.0), &b).unwrap();
    let mut w = Writer::default();
    u.put(&mut w);
    let back = Brep::get(&mut Reader::new(&w.0)).unwrap();
    assert!(crate::check::check(&back).is_empty());
    let points = |b: &Brep| b.loop_ids().filter(|&lp| b.is_vertex_loop(lp)).count();
    assert!(points(&u) > 0);
    assert_eq!(points(&back), points(&u));
    assert_eq!(crate::measure::betti(&back), [1, 0, 0]);
}

#[test]
fn a_rod_inside_a_wider_one_far_from_the_origin_meshes_to_its_deflection() {
    // Radius 4, 8 long, within one of radius 4.5 on its axis, 10⁶ out: π·16·8, and its mesh
    // within twice the deflection times its area of that.
    let o = Vec3::new(-441_397.9, 668_434.6, 510_315.9);
    let z = Vec3::new(
        0.4344780773957151,
        -0.46111673482862936,
        -0.7736925468967026,
    );
    let frame = Frame::new(o, z, Vec3::X).unwrap();
    let a = extrude(&Profile::circle(Vec2::ZERO, 8.0), &frame, 8.0, "a").unwrap();
    let b = extrude(&Profile::circle(Vec2::ZERO, 9.0), &frame, 12.0, "b").unwrap();
    let i = intersection(&a, &b).unwrap();
    let want = PI * 16.0 * 8.0;
    assert!((crate::measure::volume(&i) - want).abs() < 1e-9 * want);
    let opt = crate::mesh::MeshOptions {
        linear: 0.01,
        angular: 0.5,
    };
    let m = crate::mesh::tessellate(&i, &opt).unwrap();
    let v: f64 = m
        .triangles
        .iter()
        .map(|t| {
            let [p, q, r] = t.map(|k| m.positions[k as usize] - o);
            p.dot(q.cross(r)) / 6.0
        })
        .sum();
    let area = 2.0 * PI * 16.0 + 2.0 * PI * 4.0 * 8.0;
    assert!((v - want).abs() <= 2.0 * 0.01 * area, "{v} against {want}");
}

#[test]
fn spline_prisms_a_millionth_of_a_turn_apart_cut_into_whole_solids() {
    // One profile and the same turned 10⁻⁶ rad, the second risen: their walls lie 5·10⁻⁶ apart and
    // cross, leaving lenses thinner than the tolerance resolves between the crossings.
    let a_pts = [
        [4.5406183229289, 0.0],
        [3.773893703987082, 3.166672815117819],
        [0.7805185817605468, 4.426540842612385],
        [-2.3149733311022787, 4.0096514276361175],
        [-4.5208933619942595, 1.6454706160575705],
        [-5.6193960877739215, -2.045292910501655],
        [-2.5993084871928485, -4.5021343643630045],
        [0.9136240546157692, -5.1814194909078415],
        [2.897914277897692, -2.431638801767329],
    ];
    let b_pts = [
        [4.54061832292663, 4.540618322928143e-06],
        [3.77389053731238, 3.1666765890099398],
        [0.7805141552193139, 4.426541623128753],
        [-2.3149773407525487, 4.009649112660782],
        [-4.520895007462615, 1.6454660951633857],
        [-5.619394042478201, -2.0452985298967197],
        [-2.599303985057184, -4.5021369636692405],
        [0.9136292360348033, -5.181418577281196],
        [2.897916709535045, -2.431635903851835],
    ];
    let c_pts = [
        [4.7037838893619695, 0.0],
        [3.7605097497779063, 2.732170260192447],
        [1.4863056140978108, 4.574378319719987],
        [-1.5104511147993, 4.648690529725815],
        [-3.360795939317256, 2.441761177861711],
        [-4.188774078295128, 5.129768767335534e-16],
        [-3.1449374113512034, -2.284930777261738],
        [-1.4230590175636781, -4.3797253107845195],
        [1.4858251499627229, -4.572899603161227],
        [4.663852854933873, -3.388487443468678],
    ];
    let d_pts = [
        [4.7037838893596176, 4.703783889361185e-06],
        [3.760507017605766, 2.732174020700831],
        [1.486301039718748, 4.574379806023313],
        [-1.5104557634890743, 4.648689019272377],
        [-3.360798381076753, 2.441757817064551],
        [-4.188774078293034, -4.1887740777814534e-06],
        [-3.1449351264188534, -2.284933922198007],
        [-1.4230546378376556, -4.379726733841347],
        [1.485829722861583, -4.57289811733379],
        [4.6638562434189845, -3.3884827796141286],
    ];
    let prism_at = |pts: &[[f64; 2]],
                    z0: f64,
                    h: f64,
                    place: &dyn Fn(Vec3) -> Vec3,
                    turn: &dyn Fn(Vec3) -> Vec3| {
        let mut ctrl: Vec<Vec3> = pts.iter().map(|p| Vec3::new(p[0], p[1], 0.0)).collect();
        ctrl.extend(ctrl[..3].to_vec());
        let knots: Vec<f64> = (0..pts.len() as u32 + 7).map(f64::from).collect();
        let curve = crate::geom::BSpline::periodic_cubic(knots, ctrl).unwrap();
        let p = Profile {
            loops: vec![ProfileLoop {
                segments: vec![NamedSegment::new(
                    Segment::Spline {
                        curve,
                        reversed: false,
                    },
                    "s",
                )],
            }],
        };
        let frame = Frame::new(place(Vec3::new(0.0, 0.0, z0)), turn(Vec3::Z), turn(Vec3::X));
        extrude(&p, &frame.unwrap(), h, "p").unwrap()
    };
    let here = (|p: Vec3| p, |d: Vec3| d);
    let prism = |pts: &[[f64; 2]], z0: f64, h: f64| prism_at(pts, z0, h, &here.0, &here.1);
    let v = |x: &Brep| crate::measure::volume(x);
    let cases = [
        (
            &a_pts[..],
            &b_pts[..],
            1.2360966542470537,
            9.413739227421743,
        ),
        (&c_pts[..], &d_pts[..], 1.95054161052103, 8.067024648085315),
    ];
    for (pa, pb, z0, h) in cases {
        let (a, b) = (prism(pa, 0.0, h), prism(pb, z0, h));
        let within = v(&intersection(&a, &b).unwrap());
        for (cut, whole) in [
            (difference(&a, &b).unwrap(), &a),
            (difference(&b, &a).unwrap(), &b),
        ] {
            assert!(
                crate::check::check(&cut).is_empty(),
                "{:?}",
                crate::check::check(&cut)
            );
            assert!(
                (v(&cut) + within - v(whole)).abs() < 1e-6,
                "{} + {within}",
                v(&cut)
            );
        }
    }
    // Moved together 60 mm off and turned, a draw of the conformance suite's, they share what
    // they shared: their lenses lie no differently, only further from the origin.
    let (a, b) = (
        prism(&c_pts, 0.0, 8.067024648085315),
        prism(&d_pts, 1.95054161052103, 8.067024648085315),
    );
    let within = v(&intersection(&a, &b).unwrap());
    let motion = crate::math::Transform {
        r: [
            [0.26448997795581874, 0.9218406253280732, -0.283292980950248],
            [0.2133933115433927, -0.3424149072589213, -0.9149947135778485],
            [
                -0.9404830387356675,
                0.18155410428192326,
                -0.2872799350266453,
            ],
        ],
        t: Vec3::new(-33.721312168632736, -23.662168049862508, 37.20038819655298),
    };
    // Built in the moved frame, as the suite's adapter builds them.
    let turn = |d: Vec3| {
        let r = motion.r;
        Vec3::new(
            r[0][0] * d.x + r[0][1] * d.y + r[0][2] * d.z,
            r[1][0] * d.x + r[1][1] * d.y + r[1][2] * d.z,
            r[2][0] * d.x + r[2][1] * d.y + r[2][2] * d.z,
        )
    };
    let place = |p: Vec3| turn(p) + motion.t;
    let (ma, mb) = (
        prism_at(&c_pts, 0.0, 8.067024648085315, &place, &turn),
        prism_at(&d_pts, 1.95054161052103, 8.067024648085315, &place, &turn),
    );
    let moved = intersection(&ma, &mb).unwrap();
    assert!(
        (v(&moved) - within).abs() < 1e-6,
        "{} against {within}",
        v(&moved)
    );
}

#[test]
fn a_spline_sweep_cut_by_a_turned_box_is_the_same_either_way_round() {
    // The box's face crosses each side of the sweep where one of the sweep's edges meets it; the
    // curve traced there must end on that vertex whichever solid comes first, or the cut is lost.
    let (c, k, w) = (0.3169968843290124, 0.07546476426294255, 0.3924616485919549);
    let line = |a: [f64; 2], b: [f64; 2]| Segment::Line {
        a: Vec2::new(a[0], a[1]),
        b: Vec2::new(b[0], b[1]),
    };
    let arc = |x: f64, y: f64, start: f64| Segment::Arc {
        center: Vec2::new(x, y),
        radius: k,
        start,
        sweep: PI / 2.0,
    };
    let segs = vec![
        arc(-c, -c, PI),
        line([-c, -w], [c, -w]),
        arc(c, -c, -PI / 2.0),
        line([w, -c], [w, c]),
        arc(c, c, 0.0),
        line([c, w], [-c, w]),
        arc(-c, c, PI / 2.0),
        line([-w, c], [-w, -c]),
    ];
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: segs
                .into_iter()
                .enumerate()
                .map(|(i, segment)| NamedSegment::new(segment, format!("s{i}")))
                .collect(),
        }],
    };
    let ctrl = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.581937871174511),
        Vec3::new(1.3638310768803406, 0.6842338976783515, 3.7717731444167306),
        Vec3::new(-0.03585380959401152, 0.7411088162065418, 5.911327251059337),
        Vec3::new(-0.6095483120635864, 1.2120495889554561, 8.495257278772922),
        Vec3::new(-1.7173964878538315, 0.6609240460258358, 10.637325548739064),
        Vec3::new(-0.9004385042848417, 1.4996318290234067, 12.533797124031615),
    ];
    let knots = vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
    let path = crate::geom::Curve::BSpline(crate::geom::BSpline::new(3, knots, ctrl).unwrap());
    let a = crate::sweep::along_curve(&p, &Frame::WORLD, &path, (0.0, 1.0), 0.0, "swp").unwrap();
    let s = 13.64384629633171;
    let frame = Frame::new(
        Vec3::new(-6.775804291484357, 3.1364587822075767, 12.976144438924713),
        Vec3::new(-0.1417724780332835, 0.7983112641892967, -0.585320160203618),
        Vec3::new(
            0.9898992698614856,
            0.11433341715853237,
            -0.08382902390312598,
        ),
    )
    .unwrap();
    let b = extrude(
        &Profile::rect(Vec2::new(s / 2.0, s / 2.0), s, s),
        &frame,
        s,
        "box",
    )
    .unwrap();
    let ab = intersection(&a, &b).unwrap();
    let ba = intersection(&b, &a).unwrap();
    assert!(
        crate::check::check(&ba).is_empty(),
        "{:?}",
        crate::check::check(&ba)
    );
    let (vab, vba) = (crate::measure::volume(&ab), crate::measure::volume(&ba));
    assert!((vab - vba).abs() < 1e-9, "{vab} against {vba}");
}

#[test]
fn balls_crossing_far_from_the_origin_measure_their_lens() {
    // Radii 5 and 3 with centres 4 apart, 2·10⁵ out: the lens is two caps, of heights
    // h₁ = (r₂² − (d − r₁)²)/(2d) and h₂ = (r₁² − (d − r₂)²)/(2d), each πh²(3r − h)/3.
    let at = Vec3::new(154_814.75, -54_120.82, 42_744.44);
    let mut a = ball(5.0);
    a.transform(&crate::math::Transform::translation(at));
    let mut b = ball(3.0);
    b.transform(&crate::math::Transform::translation(
        at + Vec3::new(0.0, 4.0, 0.0),
    ));
    let (r1, r2, d) = (5.0f64, 3.0f64, 4.0f64);
    let h1 = (r2 * r2 - (d - r1).powi(2)) / (2.0 * d);
    let h2 = (r1 * r1 - (d - r2).powi(2)) / (2.0 * d);
    let cap = |h: f64, r: f64| PI * h * h * (3.0 * r - h) / 3.0;
    let want = cap(h1, r1) + cap(h2, r2);
    let got = crate::measure::volume(&intersection(&a, &b).unwrap());
    assert!((got - want).abs() < 1e-8 * want, "{got} against {want}");
}

/// Solids whose faces bulge past their own edges: a ball and a ring turned off the axes, a spline
/// profile turned part way round an axis, and a ball stretched into a spline surface.
fn bulging() -> Vec<Brep> {
    use crate::{
        edit::{Affine, affine},
        primitive,
        profile::PointsInterp,
    };
    let tilted = Frame::new(
        Vec3::new(1.0, -2.0, 0.5),
        Vec3::new(0.3, -0.5, 0.8),
        Vec3::new(1.0, 0.2, 0.0),
    )
    .unwrap();
    let outline = [
        Vec2::new(2.0, 0.0),
        Vec2::new(4.5, 1.0),
        Vec2::new(3.0, 2.5),
        Vec2::new(5.0, 4.0),
        Vec2::new(2.5, 5.0),
        Vec2::new(1.5, 2.5),
    ];
    let wavy = Profile::points(&outline, PointsInterp::PeriodicSpline, "w").unwrap();
    let plane = Frame::new(tilted.origin, -tilted.y, tilted.x).unwrap();
    let ball = primitive::sphere(&tilted, 2.0, "s").unwrap();
    let map = Affine {
        m: [[1.1, -0.9, 0.5], [-1.0, 0.6, 0.5], [1.3, 0.6, 0.4]],
        t: Vec3::new(2.0, -3.0, 1.0),
    };
    vec![
        primitive::torus(&tilted, 3.0, 1.0, "t").unwrap(),
        revolve(&wavy, &plane, tilted.origin, tilted.z, 2.2, "r").unwrap(),
        affine(&ball, &map).unwrap(),
        ball,
    ]
}

#[test]
fn a_faces_box_holds_every_point_of_the_face() {
    let fine = crate::mesh::MeshOptions {
        linear: 0.002,
        angular: 0.05,
    };
    for b in bulging() {
        let m = crate::mesh::tessellate(&b, &fine).unwrap();
        for (f, range) in &m.faces {
            let dom = face_domain(&b, *f);
            let bb = face_bbox(&dom);
            for t in &m.triangles[range.clone()] {
                for &i in t {
                    let q = m.positions[i as usize];
                    let out = (0..3)
                        .map(|k| (bb.min[k] - q[k]).max(q[k] - bb.max[k]))
                        .fold(0.0, f64::max);
                    assert!(
                        out <= 1e-9,
                        "{} face: {q:?} is {out} outside {bb:?}",
                        dom.surface.kind()
                    );
                }
            }
        }
    }
}

#[test]
fn an_empty_result_is_refused_only_where_the_solids_share_material() {
    use classify::material_lost;
    let a = block([0.0; 3], [2.0; 3]);
    // Overlapping by 1 in x: each has corners inside the other, and outside it.
    let crossing = block([1.0, 0.5, 0.5], [3.0, 1.5, 1.5]);
    assert!(material_lost(&a, &crossing, Op::Intersection));
    assert!(material_lost(&a, &crossing, Op::Difference));
    // Touching along the face x = 2 only: no point lies clear of the other's surfaces inside it.
    let touching = block([2.0, 0.0, 0.0], [4.0, 2.0, 2.0]);
    assert!(!material_lost(&a, &touching, Op::Intersection));
    // Wholly inside the other: nothing of the inner one is left by taking the outer away.
    let inner = block([0.5; 3], [1.5; 3]);
    assert!(!material_lost(&inner, &a, Op::Difference));
    assert!(material_lost(&inner, &a, Op::Intersection));
    assert!(!material_lost(&a, &crossing, Op::Union));
}
