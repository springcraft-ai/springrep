//! Results whose pieces touch, or that touch themselves, at points or along curves: the
//! regularized set operation, which need not be a manifold. Touching pieces are one solid: its
//! shells join through the edges and vertices the contact makes, an edge there used by four faces
//! and a vertex by two fans. Each result is checked valid and measured against closed forms, and
//! its Betti numbers are the point set's.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use springrep::{
    Error,
    blend::{chamfer, fillet},
    boolean::{difference, intersection, union},
    check,
    edit::offset_faces,
    math::{Frame, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    primitive::cylinder,
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
    topo::{Brep, EdgeId},
};

fn block(id: &str, lo: [f64; 3], hi: [f64; 3]) -> Brep {
    let c = Vec2::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
    let frame = Frame::new(Vec3::new(0.0, 0.0, lo[2]), Vec3::Z, Vec3::X).unwrap();
    extrude(
        &Profile::rect(c, hi[0] - lo[0], hi[1] - lo[1]),
        &frame,
        hi[2] - lo[2],
        id,
    )
    .unwrap()
}

/// A cylinder of radius `r` along `axis` from `base`, `len` long.
fn rod(id: &str, base: [f64; 3], axis: [f64; 3], r: f64, len: f64) -> Brep {
    let frame = Frame::from_axis(Vec3::from_array(base), Vec3::from_array(axis)).unwrap();
    extrude(&Profile::circle(Vec2::ZERO, 2.0 * r), &frame, len, id).unwrap()
}

fn ball(id: &str, centre: [f64; 3], r: f64) -> Brep {
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
    let c = Vec3::from_array(centre);
    let frame = Frame::new(c, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap();
    revolve(&p, &frame, c, Vec3::Z, TAU, id).unwrap()
}

/// A cone of base radius `r` and height `h` standing on its apex at `apex`, opening along `up`.
fn cone(id: &str, apex: [f64; 3], up: f64, r: f64, h: f64) -> Brep {
    let seg = |a: [f64; 2], b: [f64; 2], name: &str| {
        NamedSegment::new(
            Segment::Line {
                a: Vec2::new(a[0], a[1]),
                b: Vec2::new(b[0], b[1]),
            },
            name,
        )
    };
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                seg([0.0, 0.0], [r, up * h], "side"),
                seg([r, up * h], [0.0, up * h], "base"),
                seg([0.0, up * h], [0.0, 0.0], "axis"),
            ],
        }],
    };
    let c = Vec3::from_array(apex);
    let frame = Frame::new(c, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap();
    let mut b = revolve(&p, &frame, c, Vec3::Z, TAU, id).unwrap();
    if measure::volume(&b) < 0.0 {
        b = revolve(&p, &frame, c, -Vec3::Z, TAU, id).unwrap();
    }
    b
}

fn cone_measures(r: f64, h: f64) -> (f64, f64) {
    (
        PI * r * r * h / 3.0,
        PI * r * r + PI * r * (r * r + h * h).sqrt(),
    )
}

/// One valid solid of the volume, area and Betti numbers given.
fn assert_touching(b: &Brep, volume: f64, area: f64, betti: [u64; 3]) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
    assert_eq!(b.solid_ids().count(), 1, "touching pieces are one solid");
    let (v, a) = (measure::volume(b), measure::area(b));
    assert!(
        (v - volume).abs() <= 1e-9 * volume,
        "volume {v} vs {volume}"
    );
    assert!((a - area).abs() <= 1e-9 * area, "area {a} vs {area}");
    assert_eq!(measure::betti(b), betti);
    let m = mesh::tessellate(b, &COARSE).unwrap();
    for s in b.solid_ids() {
        let r = mesh::report(b, &m, s);
        assert!(r.watertight(), "{r:?}");
    }
}

const COARSE: MeshOptions = MeshOptions::new(0.2, 0.5);

/// The edges four faces use: where pieces meet along a curve.
fn four_face_edges(b: &Brep) -> Vec<EdgeId> {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for c in b.coedge_ids() {
        *uses.entry(b.coedge(c).edge).or_default() += 1;
    }
    let mut out: Vec<EdgeId> = uses
        .into_iter()
        .filter(|&(_, n)| n == 4)
        .map(|(e, _)| e)
        .collect();
    out.sort();
    out
}

#[test]
fn blocks_touching_along_an_edge_unite_round_it() {
    // Two 10 mm cubes sharing one vertical edge: one piece, both cubes, the edge used by all four
    // of their faces that meet there.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = union(&a, &b).unwrap();
    assert_touching(&u, 2000.0, 1200.0, [1, 0, 0]);
    assert_eq!(four_face_edges(&u).len(), 1);
}

#[test]
fn blocks_touching_at_a_corner_unite_there() {
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]);
    let u = union(&a, &b).unwrap();
    assert_touching(&u, 2000.0, 1200.0, [1, 0, 0]);
    assert_eq!(
        u.vertex_ids()
            .filter(|&v| u.vertex(v).point == Vec3::new(10.0, 10.0, 10.0))
            .count(),
        1
    );
}

#[test]
fn a_rod_lying_on_a_box_face_unites_along_the_line() {
    // The rod's side touches the face z = 10 along x = 5..15 at y = 5: both stay whole, and the
    // line is an edge.
    let a = block("a", [0.0, 0.0, 0.0], [20.0, 10.0, 10.0]);
    let c = rod("c", [5.0, 5.0, 12.0], [1.0, 0.0, 0.0], 2.0, 10.0);
    let u = union(&a, &c).unwrap();
    let rod_area = 2.0 * PI * 4.0 + 2.0 * PI * 2.0 * 10.0;
    assert_touching(&u, 2000.0 + PI * 4.0 * 10.0, 1000.0 + rod_area, [1, 0, 0]);
    assert_eq!(four_face_edges(&u).len(), 1);
}

#[test]
fn bored_tubes_fanned_from_a_shells_axis_touch_along_it() {
    // Tubes of radius 120 from the axis of a shell of radius 1000, each 87.5 above and 11.25°
    // round from the last, are united and bored to radius 100 in turn. The first tube's end ring
    // less the second bore leaves an 11.25° wedge whose edge runs along the axis from z = 700 to
    // 720, inside the third tube's end disc, which it touches only there until the third bore
    // takes the disc's middle.
    let at = |origin: Vec3, dir: Vec3| {
        let hint = if dir.z.abs() > 0.9 { Vec3::X } else { Vec3::Z };
        Frame::new(origin, dir, hint).unwrap()
    };
    let outer = cylinder(&at(Vec3::ZERO, Vec3::Z), 1000.0, 4000.0, "outer").unwrap();
    let inner = cylinder(&at(Vec3::ZERO, Vec3::Z), 980.0, 4000.0, "inner").unwrap();
    let mut body = difference(&outer, &inner).unwrap();
    // Each step's volume by inclusion–exclusion from the common part.
    let unite = |a: &Brep, b: &Brep| {
        let (u, i) = (union(a, b).unwrap(), intersection(a, b).unwrap());
        let want = measure::volume(a) + measure::volume(b) - measure::volume(&i);
        let v = measure::volume(&u);
        assert!((v - want).abs() <= 1e-9 * want, "union {v} vs {want}");
        u
    };
    let cut = |a: &Brep, b: &Brep| {
        let (d, i) = (difference(a, b).unwrap(), intersection(a, b).unwrap());
        let want = measure::volume(a) - measure::volume(&i);
        let v = measure::volume(&d);
        assert!((v - want).abs() <= 1e-9 * want, "difference {v} vs {want}");
        d
    };
    for i in 0..3 {
        let a = TAU * i as f64 / 32.0;
        let (origin, dir) = (
            Vec3::new(0.0, 0.0, 600.0 + 87.5 * i as f64),
            Vec3::new(a.cos(), a.sin(), 0.0),
        );
        let tube = cylinder(&at(origin, dir), 120.0, 1300.0, &format!("tube{i}")).unwrap();
        body = unite(&body, &tube);
        if i == 2 {
            assert_eq!(body.solid_ids().count(), 1);
            let ends = |e: EdgeId| {
                let edge = body.edge(e);
                [edge.start, edge.end].map(|v| body.vertex(v).point)
            };
            let contact = [Vec3::new(0.0, 0.0, 700.0), Vec3::new(0.0, 0.0, 720.0)];
            let on_axis = |e: EdgeId| {
                let mut ps = ends(e);
                ps.sort_by(|p, q| p.z.total_cmp(&q.z));
                ps.iter().zip(&contact).all(|(p, q)| p.dist(*q) < 1e-9)
            };
            assert_eq!(
                four_face_edges(&body)
                    .into_iter()
                    .filter(|&e| on_axis(e))
                    .count(),
                1
            );
        }
        let bore = cylinder(&at(origin, dir), 100.0, 1400.0, &format!("bore{i}")).unwrap();
        body = cut(&body, &bore);
    }
    let issues = check::check(&body);
    assert!(issues.is_empty(), "{issues:?}");
    assert_eq!(body.solid_ids().count(), 1);
}

#[test]
fn a_rod_lying_under_a_box_face_unites_along_the_line() {
    // The same contact on the face z = 0, whose outward normal runs against its plane's: the line
    // is a slit of its own inside the face, which encloses nothing and so runs neither way.
    let a = block("a", [0.0, 0.0, 0.0], [20.0, 10.0, 10.0]);
    let c = rod("c", [5.0, 5.0, -2.0], [1.0, 0.0, 0.0], 2.0, 10.0);
    let u = union(&a, &c).unwrap();
    let rod_area = 2.0 * PI * 4.0 + 2.0 * PI * 2.0 * 10.0;
    assert_touching(&u, 2000.0 + PI * 4.0 * 10.0, 1000.0 + rod_area, [1, 0, 0]);
    assert_eq!(four_face_edges(&u).len(), 1);
}

#[test]
fn a_box_less_an_inscribed_cylinder_is_four_corners_in_a_ring() {
    // A 10 mm square prism less the cylinder touching all four sides: four corner pieces, each
    // meeting the next along a line, round the bore — one tunnel.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 8.0]);
    let c = rod("c", [5.0, 5.0, -1.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    let d = difference(&a, &c).unwrap();
    let area = 2.0 * (100.0 - PI * 25.0) + 4.0 * 80.0 + 2.0 * PI * 5.0 * 8.0;
    assert_touching(&d, 800.0 - PI * 25.0 * 8.0, area, [1, 1, 0]);
    assert_eq!(four_face_edges(&d).len(), 4);
}

#[test]
fn a_block_less_a_bore_flush_to_its_top_touches_itself_along_a_line() {
    // A bore of radius 3 along x whose top line lies on the top face z = 8: one piece round the
    // bore, its material thinning to nothing along that line, still a tunnel.
    let a = block("a", [0.0, 0.0, 0.0], [12.0, 10.0, 8.0]);
    let c = rod("c", [-1.0, 5.0, 5.0], [1.0, 0.0, 0.0], 3.0, 14.0);
    let d = difference(&a, &c).unwrap();
    let area = 2.0 * (120.0 + 96.0 + 80.0) - 2.0 * PI * 9.0 + 2.0 * PI * 3.0 * 12.0;
    assert_touching(&d, 960.0 - PI * 9.0 * 12.0, area, [1, 1, 0]);
    assert_eq!(four_face_edges(&d).len(), 1);
}

#[test]
fn a_rod_inside_a_box_against_a_wall_leaves_a_void_touching_it() {
    // The rod lies inside, its side on the wall x = 0 along a line: the cavity is still closed off,
    // as the line is the solid's.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let c = rod("c", [2.0, 5.0, 2.0], [0.0, 0.0, 1.0], 2.0, 6.0);
    let d = difference(&a, &c).unwrap();
    let area = 600.0 + 2.0 * PI * 4.0 + 2.0 * PI * 2.0 * 6.0;
    assert_touching(&d, 1000.0 - PI * 4.0 * 6.0, area, [1, 0, 1]);
    assert_eq!(four_face_edges(&d).len(), 1);
}

#[test]
fn a_cube_less_a_ball_touching_its_six_faces_keeps_the_void() {
    // The ball touches each face at its centre: one piece whose cavity meets the outside at six
    // points, which are the solid's, so the cavity is still a void.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let s = ball("s", [5.0, 5.0, 5.0], 5.0);
    let d = difference(&a, &s).unwrap();
    assert_touching(
        &d,
        1000.0 - 4.0 / 3.0 * PI * 125.0,
        600.0 + 4.0 * PI * 25.0,
        [1, 0, 1],
    );
}

#[test]
fn a_touching_result_combined_with_itself_keeps_its_contact() {
    // Combining a solid with itself copies it: the copy's edge four faces use is paired again
    // from the faces round it.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = union(&a, &b).unwrap();
    for x in [intersection(&u, &u).unwrap(), union(&u, &u).unwrap()] {
        assert_touching(&x, 2000.0, 1200.0, [1, 0, 0]);
        assert_eq!(four_face_edges(&x).len(), 1);
    }
}

#[test]
fn balls_touching_at_a_point_unite_there() {
    // Each ball's surface meets the other only at (10, 0, 0), inside a face of each: that point is
    // a vertex both fans share.
    let a = ball("a", [5.0, 0.0, 0.0], 5.0);
    let b = ball("b", [15.0, 0.0, 0.0], 5.0);
    let u = union(&a, &b).unwrap();
    let (v, s) = (4.0 / 3.0 * PI * 125.0, 4.0 * PI * 25.0);
    assert_touching(&u, 2.0 * v, 2.0 * s, [1, 0, 0]);
    let at = Vec3::new(10.0, 0.0, 0.0);
    assert_eq!(
        u.vertex_ids()
            .filter(|&v| u.vertex(v).point.dist(at) < 1e-9)
            .count(),
        1
    );
}

#[test]
fn a_ball_on_a_box_face_unites_at_a_point() {
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let s = ball("s", [5.0, 5.0, 13.0], 3.0);
    let u = union(&a, &s).unwrap();
    let (v, area) = (4.0 / 3.0 * PI * 27.0, 4.0 * PI * 9.0);
    assert_touching(&u, 1000.0 + v, 600.0 + area, [1, 0, 0]);
}

#[test]
fn balls_touching_at_a_point_combine_as_inputs() {
    let a = ball("a", [5.0, 0.0, 0.0], 5.0);
    let b = ball("b", [15.0, 0.0, 0.0], 5.0);
    let u = union(&a, &b).unwrap();
    let (v, s) = (4.0 / 3.0 * PI * 125.0, 4.0 * PI * 25.0);
    // Combined with itself, the contact is kept.
    for x in [intersection(&u, &u).unwrap(), union(&u, &u).unwrap()] {
        assert_touching(&x, 2.0 * v, 2.0 * s, [1, 0, 0]);
    }
    // A third ball touching the second at (20, 0, 0) joins the chain at a second point.
    let c = ball("c", [25.0, 0.0, 0.0], 5.0);
    assert_touching(&union(&u, &c).unwrap(), 3.0 * v, 3.0 * s, [1, 0, 0]);
    // A slab through the contact takes a cap of height 1 off each ball, parting them; a cap's
    // volume is π h² (3r − h) / 3.
    let slab = block("s", [9.0, -6.0, -6.0], [11.0, 6.0, 6.0]);
    let d = difference(&u, &slab).unwrap();
    assert!(check::check(&d).is_empty());
    assert_eq!(d.solid_ids().count(), 2);
    assert_eq!(measure::betti(&d), [2, 0, 0]);
    let cap = PI * (3.0 * 5.0 - 1.0) / 3.0;
    assert!((measure::volume(&d) - 2.0 * (v - cap)).abs() < 1e-9 * v);
}

#[test]
fn a_cone_standing_on_its_apex_unites_with_the_face_under_it() {
    let a = block("a", [-10.0, -10.0, -10.0], [10.0, 10.0, 0.0]);
    let c = cone("c", [0.0, 0.0, 0.0], 1.0, 3.0, 6.0);
    let u = union(&a, &c).unwrap();
    let (v, s) = cone_measures(3.0, 6.0);
    assert_touching(&u, 4000.0 + v, 1600.0 + s, [1, 0, 0]);
}

#[test]
fn cones_touching_apex_to_apex_unite_there() {
    let a = cone("a", [0.0, 0.0, 0.0], 1.0, 3.0, 6.0);
    let b = cone("b", [0.0, 0.0, 0.0], -1.0, 2.0, 5.0);
    let u = union(&a, &b).unwrap();
    let (va, sa) = cone_measures(3.0, 6.0);
    let (vb, sb) = cone_measures(2.0, 5.0);
    assert_touching(&u, va + vb, sa + sb, [1, 0, 0]);
}

#[test]
fn a_cone_on_a_ball_unites_at_its_apex() {
    let a = ball("a", [0.0, 0.0, -4.0], 4.0);
    let c = cone("c", [0.0, 0.0, 0.0], 1.0, 3.0, 6.0);
    let u = union(&a, &c).unwrap();
    let (v, s) = cone_measures(3.0, 6.0);
    assert_touching(
        &u,
        4.0 / 3.0 * PI * 64.0 + v,
        4.0 * PI * 16.0 + s,
        [1, 0, 0],
    );
}

#[test]
fn filling_the_corner_between_touching_blocks_makes_one_manifold_piece() {
    // The cubes share the edge x = y = 10; a block over x 5..15, y 5..15 fills round it, so the
    // contact is inside the union and no edge is used by four faces.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = union(&a, &b).unwrap();
    let c = block("c", [5.0, 5.0, 0.0], [15.0, 15.0, 10.0]);
    let filled = union(&u, &c).unwrap();
    // Two 5 x 5 x 10 quarters of c are new; the plan is a stepped outline of 32 unit sides.
    let area = 2.0 * 250.0 + 10.0 * 80.0;
    assert_touching(&filled, 2500.0, area, [1, 0, 0]);
    assert!(four_face_edges(&filled).is_empty());
}

#[test]
fn cutting_the_contact_away_parts_the_pieces() {
    // A square bar along z round the shared edge removes it: two solids, each a cube less a
    // 2 x 2 corner.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = union(&a, &b).unwrap();
    let c = block("c", [8.0, 8.0, -1.0], [12.0, 12.0, 11.0]);
    let d = difference(&u, &c).unwrap();
    assert!(check::check(&d).is_empty());
    assert_eq!(d.solid_ids().count(), 2);
    assert!((measure::volume(&d) - 2.0 * 960.0).abs() < 1e-9);
    assert!(four_face_edges(&d).is_empty());
}

#[test]
fn cutting_across_the_contact_keeps_it_where_it_is_left() {
    // Cutting the top half off both cubes keeps the shared edge below z = 5.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = union(&a, &b).unwrap();
    let c = block("c", [-1.0, -1.0, 5.0], [21.0, 21.0, 11.0]);
    let d = difference(&u, &c).unwrap();
    assert_touching(&d, 1000.0, 2.0 * (200.0 + 200.0), [1, 0, 0]);
    assert_eq!(four_face_edges(&d).len(), 1);
}

#[test]
fn a_touching_result_with_its_contact_cut_into() {
    // A ring of four corners touching along four lines, less a slot through one contact: the ring
    // is broken there, so one piece with no tunnel, three contacts left.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 8.0]);
    let c = rod("c", [5.0, 5.0, -1.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    let ring = difference(&a, &c).unwrap();
    let slot = block("s", [9.0, 4.0, -1.0], [11.0, 6.0, 9.0]);
    let d = difference(&ring, &slot).unwrap();
    assert!(check::check(&d).is_empty());
    assert_eq!(d.solid_ids().count(), 1);
    assert_eq!(measure::betti(&d), [1, 0, 0]);
    assert_eq!(four_face_edges(&d).len(), 3);
}

fn refused_as_touching(r: Result<Brep, Error>) {
    match r {
        Err(Error::Unsupported(why)) => assert!(why.contains("touch"), "{why}"),
        r => panic!(
            "expected a refusal, got {:?}",
            r.map(|b| measure::volume(&b))
        ),
    }
}

#[test]
fn a_blend_along_a_contact_is_refused() {
    // The edge the cubes share is the contact: a round there would cut into the other cube.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = union(&a, &b).unwrap();
    refused_as_touching(fillet(&u, &[("a.u_max", "a.v_max")], 1.0, "f"));
    refused_as_touching(chamfer(&u, &[("a.u_max", "a.v_max")], 1.0, "f"));
    // An edge away from the contact still blends.
    let f = fillet(&u, &[("a.u_min", "a.v_min")], 1.0, "f").unwrap();
    assert!(check::check(&f).is_empty());
}

#[test]
fn moving_a_face_at_a_contact_is_refused() {
    // a.u_max is the face x = 10, which ends at the contact: moving it opens or overlaps it.
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = union(&a, &b).unwrap();
    let face = |name: &str| {
        u.face_ids()
            .find(|&f| {
                u.face(f)
                    .names
                    .iter()
                    .any(|n| n.feature == "a" && n.face == name)
            })
            .unwrap()
    };
    refused_as_touching(offset_faces(&u, &HashMap::from([(face("u_max"), 1.0)])));
    let moved = offset_faces(&u, &HashMap::from([(face("u_min"), 1.0)])).unwrap();
    assert!(check::check(&moved).is_empty());
    assert!((measure::volume(&moved) - 2100.0).abs() < 1e-9);
}

#[test]
fn a_face_a_rod_touches_through_a_hole_splits_round_the_hole() {
    // A 40 × 20 × 20 block with a hole r 2 through it along y at x = 20, z = 10, and a rod r 3
    // standing on its front face from z = 5 up: the line of contact runs from the top edge down
    // to the hole, and on below it to where the rod ends. The front face is one face, whose loop
    // runs down the contact, round the hole, down to the rod's end and back.
    let holed = difference(
        &block("block", [0.0, 0.0, 0.0], [40.0, 20.0, 20.0]),
        &rod("hole", [20.0, -1.0, 10.0], [0.0, 1.0, 0.0], 2.0, 22.0),
    )
    .unwrap();
    let standing = rod("rod", [20.0, -3.0, 5.0], [0.0, 0.0, 1.0], 3.0, 15.0);
    let touched = union(&holed, &standing).unwrap();
    assert!(check::check(&touched).is_empty());
    // Cut at x = 30, so the face is split away from the contact: where the loop reaches the
    // hole down the contact it goes on round the hole, not straight back up.
    let left = block("left", [-1.0, -10.0, -1.0], [30.0, 30.0, 21.0]);
    let kept = intersection(&touched, &left).unwrap();
    assert!(check::check(&kept).is_empty());
    assert_eq!(kept.solid_ids().count(), 1);
    // The contact is still held where the rod ends and where it meets the hole: the front face
    // has a vertex at each, as the faces of the rod there do.
    for z in [5.0, 8.0, 12.0, 20.0] {
        let at = Vec3::new(20.0, 0.0, z);
        let holds = |f: &springrep::topo::FaceId| {
            kept.face(*f).loops.iter().any(|&lp| {
                let ends = |c: &springrep::topo::CoedgeId| {
                    kept.vertex(kept.coedge_start(*c)).point.dist(at) < 1e-9
                };
                kept.lp(lp).coedges.iter().any(ends)
            })
        };
        let front = |f: &springrep::topo::FaceId| {
            kept.face(*f)
                .names
                .iter()
                .any(|n| n.feature == "block" && n.face == "v_min")
        };
        assert!(kept.face_ids().filter(front).any(|f| holds(&f)), "{z}");
        assert!(kept.face_ids().filter(holds).count() >= 2, "{z}");
    }
    // 30 · 20 · 20 of block, less the hole π · 2² · 20, and the rod π · 3² · 15.
    let want = 12_000.0 - 80.0 * PI + 135.0 * PI;
    let got = measure::volume(&kept);
    assert!((got - want).abs() < 1e-6 * want, "{got}");
}
