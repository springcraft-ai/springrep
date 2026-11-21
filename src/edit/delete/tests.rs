use std::f64::consts::{PI, TAU};

use super::{delete_faces, direct::direct, general};
use crate::{
    blend,
    boolean::{self, half_space, same_domain},
    check,
    math::{Frame, Point3, Transform, Vec2, Vec3},
    measure, primitive,
    profile::Profile,
    sweep::{extrude, revolve},
    topo::{Brep, FaceId, FaceName, radial},
};

/// A prism on the polygon `pts`, `h` tall, its sides named `e0`, `e1`, … from the first edge.
fn prism(pts: &[[f64; 2]], h: f64) -> Brep {
    let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
    let names: Vec<String> = (0..pts.len()).map(|i| format!("e{i}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    extrude(&Profile::polygon(&pts, &names), &Frame::WORLD, h, "p").unwrap()
}

/// The solid turned a whole turn about z from the polygon `pts` in (ρ, z), its faces `m0`, `m1`, ….
fn turned(pts: &[[f64; 2]]) -> Brep {
    let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
    let names: Vec<String> = (0..pts.len()).map(|i| format!("m{i}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let plane = Frame::new(Point3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let profile = Profile::polygon(&pts, &names);
    revolve(&profile, &plane, Point3::ZERO, Vec3::Z, TAU, "p").unwrap()
}

fn named(b: &Brep, names: &[&str]) -> Vec<FaceId> {
    let mut out = Vec::new();
    for n in names {
        let n = FaceName::new("p", *n);
        out.extend(b.face_ids().filter(|&f| b.face(f).names.contains(&n)));
    }
    assert_eq!(out.len(), names.len());
    out
}

fn of_feature(b: &Brep, feature: &str) -> Vec<FaceId> {
    let out: Vec<FaceId> = b
        .face_ids()
        .filter(|&f| b.face(f).names.iter().any(|n| n.feature == feature))
        .collect();
    assert!(!out.is_empty());
    out
}

fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-9 * want.abs().max(1.0),
        "{got} against {want}"
    );
}

fn edge_lengths(b: &Brep) -> f64 {
    b.edge_ids()
        .filter_map(|e| {
            let x = b.edge(e);
            x.curve.as_ref().map(|c| radial::length(c, x.range))
        })
        .sum()
}

/// Asserts `a` and `b` are one solid: face for face the same surface facing the same way with the
/// same names and area, and the same volume, box and length of edges. `b` may carry an edge in
/// more pieces than `a`.
fn assert_same(a: &Brep, b: &Brep) {
    assert!(check::check(a).is_empty(), "{:?}", check::check(a));
    assert_eq!(a.face_count(), b.face_count());
    close(measure::volume(a), measure::volume(b));
    close(measure::area(a), measure::area(b));
    close(edge_lengths(a), edge_lengths(b));
    assert!(a.edge_ids().count() <= b.edge_ids().count());
    let (ba, bb) = (measure::bbox(a), measure::bbox(b));
    assert!(ba.min.dist(bb.min) <= 1e-9 && ba.max.dist(bb.max) <= 1e-9);
    let mut left: Vec<FaceId> = b.face_ids().collect();
    for f in a.face_ids() {
        let face = a.face(f);
        let p = measure::face_centroid(a, f);
        let out = |s: &Brep, g: FaceId| {
            let x = s.face(g);
            let uv = x.surface.project(p);
            let n = x.surface.normal(uv.x, uv.y);
            if x.reversed { -n } else { n }
        };
        let at = left.iter().position(|&g| {
            let other = b.face(g);
            other.names == face.names
                && same_domain(&other.surface, &face.surface)
                && out(a, f).dot(out(b, g)) > 0.0
                && (measure::face_area(a, f) - measure::face_area(b, g)).abs() <= 1e-9
        });
        let at = at.unwrap_or_else(|| panic!("no face of the other like {:?}", face.names));
        left.swap_remove(at);
    }
}

/// Deletes `faces` both ways, asserts the results are one solid, and gives it.
fn both(b: &Brep, faces: &[FaceId]) -> Brep {
    let fast = direct(b, faces).expect("the gap closes in place");
    assert_same(&fast, &general(b, faces).unwrap());
    fast
}

fn chamfered() -> Brep {
    prism(
        &[[0.0, 0.0], [8.0, 0.0], [8.0, 4.0], [6.0, 6.0], [0.0, 6.0]],
        2.0,
    )
}

#[test]
fn a_chamfer_closes_at_the_edge_its_two_walls_meet_in() {
    let b = chamfered();
    let out = both(&b, &named(&b, &["e2"]));
    close(measure::volume(&out), 96.0);
    assert_eq!(out.edge_ids().count(), 12);
}

#[test]
fn a_chamfer_under_a_sloping_top_closes_the_one_way_that_holds() {
    // The top slopes, so top and bottom meet in a line too, and the chamfer's four neighbours
    // could close in two corners on either pair's line: only the walls' line is reached.
    let b = chamfered();
    let n = Vec3::new(0.2, 0.0, 1.0);
    let cut = half_space(&b, Point3::new(0.0, 0.0, 1.8), n).unwrap();
    let b = boolean::difference(&b, &cut).unwrap();
    let out = both(&b, &named(&b, &["e2"]));
    // The box 8 × 6 under z = 1.8 − 0.2·x.
    close(measure::volume(&out), 48.0 * 1.8 - 0.2 * 6.0 * 32.0);
}

#[test]
fn a_notch_closes_by_its_wall_s_two_halves_becoming_one_face() {
    let b = prism(
        &[
            [0.0, 0.0],
            [6.0, 0.0],
            [6.0, 4.0],
            [4.0, 4.0],
            [4.0, 3.0],
            [2.0, 3.0],
            [2.0, 4.0],
            [0.0, 4.0],
        ],
        1.0,
    );
    let out = both(&b, &named(&b, &["e3", "e4", "e5"]));
    close(measure::volume(&out), 24.0);
    assert_eq!(out.edge_ids().count(), 12);
}

#[test]
fn a_bore_closes_by_its_two_ends_dropping_out_of_the_faces_they_pierce() {
    let hole = |x: f64, name: &str| Profile::named_circle(Vec2::new(x, 2.0), 1.5, name).loops;
    let section = Profile::rect(Vec2::new(5.0, 2.0), 10.0, 4.0)
        .with_holes([hole(3.0, "bore"), hole(7.0, "other")].concat());
    let plate = extrude(&section, &Frame::WORLD, 2.0, "p").unwrap();
    let out = both(&plate, &named(&plate, &["bore"]));
    close(measure::volume(&out), 80.0 - 0.5625 * PI * 2.0);
    // Both at once, and every face the first left alone keeps what was worked out of it.
    let out = both(&plate, &named(&plate, &["bore", "other"]));
    close(measure::volume(&out), 80.0);
    let side = named(&plate, &["u_min"])[0];
    let kept = out
        .face_ids()
        .find(|&f| out.face(f).names == plate.face(side).names)
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &plate.shared_memo(side),
        &out.shared_memo(kept)
    ));
}

#[test]
fn a_groove_closes_by_the_shaft_s_two_lengths_becoming_one_face() {
    let b = turned(&[
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 3.0],
        [2.0, 3.0],
        [2.0, 6.0],
        [3.0, 6.0],
        [3.0, 10.0],
        [0.0, 10.0],
    ]);
    let out = both(&b, &named(&b, &["m2", "m3", "m4"]));
    close(measure::volume(&out), 90.0 * PI);
    assert_eq!(out.edge_ids().count(), 3);
}

#[test]
fn a_boss_closes_by_its_foot_dropping_out_of_the_face_it_stands_on() {
    let b = turned(&[
        [0.0, 0.0],
        [6.0, 0.0],
        [6.0, 2.0],
        [2.0, 2.0],
        [2.0, 5.0],
        [0.0, 5.0],
    ]);
    let out = both(&b, &named(&b, &["m3", "m4"]));
    close(measure::volume(&out), 72.0 * PI);
}

#[test]
fn a_bevelled_rim_closes_at_the_circle_the_wall_and_the_end_meet_in() {
    let b = turned(&[[0.0, 0.0], [4.0, 0.0], [4.0, 5.0], [3.0, 6.0], [0.0, 6.0]]);
    let out = both(&b, &named(&b, &["m2"]));
    close(measure::volume(&out), 96.0 * PI);
    // A countersink at a tube's bore closes the same way, against a hollow wall.
    let b = turned(&[[2.0, 0.0], [5.0, 0.0], [5.0, 6.0], [3.0, 6.0], [2.0, 5.0]]);
    let out = both(&b, &named(&b, &["m3"]));
    close(measure::volume(&out), 126.0 * PI);
}

#[test]
fn rounds_on_turned_parts_close_at_the_circle_their_faces_meet_in() {
    use primitive::{cone, cylinder};
    // A frustum's top rim, a cylinder's, and a cone's foot, each rounded 1.
    let frustum = cone(&Frame::WORLD, 6.0, 3.0, 6.0, "f").unwrap();
    let rounded = blend::fillet(&frustum, &[("f.top", "f.side")], 1.0, "round").unwrap();
    close(
        measure::volume(&both(&rounded, &of_feature(&rounded, "round"))),
        126.0 * PI,
    );
    let drum = cylinder(&Frame::WORLD, 4.0, 5.0, "c").unwrap();
    let rounded = blend::fillet(&drum, &[("c.end", "c.side")], 1.0, "round").unwrap();
    close(
        measure::volume(&both(&rounded, &of_feature(&rounded, "round"))),
        80.0 * PI,
    );
    let point = cone(&Frame::WORLD, 6.0, 0.0, 8.0, "c").unwrap();
    let rounded = blend::fillet(&point, &[("c.bottom", "c.side")], 1.0, "round").unwrap();
    close(
        measure::volume(&both(&rounded, &of_feature(&rounded, "round"))),
        96.0 * PI,
    );
}

#[test]
fn a_cut_corner_closes_where_its_three_walls_meet() {
    let block = primitive::block(&Frame::WORLD, Vec3::new(4.0, 3.0, 2.0), "b").unwrap();
    let n = Vec3::new(1.0, 1.0, 1.0);
    let cut = half_space(&block, Point3::new(3.0, 3.0, 2.0), n).unwrap();
    let b = boolean::difference(&block, &cut).unwrap();
    let out = both(&b, &of_feature(&b, "half"));
    close(measure::volume(&out), 24.0);
    assert_eq!(out.vertex_ids().count(), 8);
}

#[test]
fn a_nick_in_an_edge_closes_by_the_edge_s_two_lengths_becoming_one() {
    let block = primitive::block(&Frame::WORLD, Vec3::new(4.0, 3.0, 2.0), "b").unwrap();
    // A bar 1 × 1 along y from 1 to 2 about the edge x = 4, z = 2.
    let along = Frame::new(Point3::new(0.0, 1.0, 0.0), Vec3::Y, Vec3::Z).unwrap();
    let bar = extrude(
        &Profile::rect(Vec2::new(2.0, 4.0), 1.0, 1.0),
        &along,
        1.0,
        "bar",
    )
    .unwrap();
    let b = boolean::difference(&block, &bar).unwrap();
    let out = both(&b, &of_feature(&b, "bar"));
    close(measure::volume(&out), 24.0);
    assert_eq!(out.edge_ids().count(), 12);
}

#[test]
fn a_round_in_an_inside_corner_closes_by_taking_material_away() {
    let pts = [
        [0.0, 0.0],
        [20.0, 0.0],
        [20.0, 5.0],
        [5.0, 5.0],
        [5.0, 15.0],
        [0.0, 15.0],
    ];
    let l = prism(&pts, 8.0);
    let rounded = blend::fillet(&l, &[("p.e2", "p.e3")], 2.0, "round").unwrap();
    let out = both(&rounded, &of_feature(&rounded, "round"));
    close(measure::volume(&out), 1200.0);
}

/// An L whose inside corner is filled by a flat `e3` from (8, 5) to (5, 8), 4 tall.
fn filled_corner() -> Brep {
    prism(
        &[
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 5.0],
            [8.0, 5.0],
            [5.0, 8.0],
            [5.0, 15.0],
            [0.0, 15.0],
        ],
        4.0,
    )
}

#[test]
fn walls_that_would_run_on_through_a_bore_are_left_to_the_general_way() {
    // The flat alone closes in place, taking the corner's fill away.
    let b = filled_corner();
    close(
        measure::volume(&both(&b, &named(&b, &["e3"]))),
        4.0 * (150.0),
    );
    // A bore through the fill: the walls run on would cut through it.
    let below = Frame::new(Point3::new(6.0, 6.0, -1.0), Vec3::Z, Vec3::X).unwrap();
    let bore = primitive::cylinder(&below, 0.4, 6.0, "bore").unwrap();
    let bored = boolean::difference(&b, &bore).unwrap();
    let flat = named(&bored, &["e3"]);
    assert!(direct(&bored, &flat).is_none());
    assert_eq!(
        delete_faces(&bored, &flat).is_ok(),
        general(&bored, &flat).is_ok()
    );
}

#[test]
fn a_gap_the_faces_round_it_bulge_over_is_left_to_the_general_way() {
    // A ball cut flat: the flat's one neighbour is the ball, which closes over it past the
    // flat's own edge.
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let above = half_space(&ball, Point3::new(0.0, 0.0, 3.0), Vec3::Z).unwrap();
    let cut = boolean::difference(&ball, &above).unwrap();
    let flat = of_feature(&cut, "half");
    assert!(direct(&cut, &flat).is_none());
    close(
        measure::volume(&delete_faces(&cut, &flat).unwrap()),
        500.0 * PI / 3.0,
    );
}

#[test]
fn a_solid_standing_in_a_bore_leaves_it_to_the_general_way() {
    let hole = Profile::named_circle(Vec2::new(5.0, 2.0), 1.5, "bore").loops;
    let section = Profile::rect(Vec2::new(5.0, 2.0), 10.0, 4.0).with_holes(hole);
    let mut plate = extrude(&section, &Frame::WORLD, 2.0, "p").unwrap();
    let bore = named(&plate, &["bore"]);
    assert!(direct(&plate, &bore).is_some());
    let at = Frame::new(Point3::new(5.0, 2.0, 1.0), Vec3::Z, Vec3::X).unwrap();
    plate.append(&primitive::sphere(&at, 0.5, "ball").unwrap());
    assert!(direct(&plate, &bore).is_none());
}

#[test]
fn a_plane_that_cuts_the_plug_leaves_a_bore_to_the_general_way() {
    // A bore through a block whose top is cut aslant across the bore's mouth: the mouth lies in
    // two faces, and neither closes it alone.
    let hole = Profile::named_circle(Vec2::new(5.0, 2.0), 1.0, "bore").loops;
    let section = Profile::rect(Vec2::new(5.0, 2.0), 10.0, 4.0).with_holes(hole);
    let plate = extrude(&section, &Frame::WORLD, 4.0, "p").unwrap();
    let n = Vec3::new(1.0, 0.0, 1.0);
    let cut = half_space(&plate, Point3::new(5.0, 2.0, 4.0), n).unwrap();
    let b = boolean::difference(&plate, &cut).unwrap();
    let bore = named(&b, &["bore"]);
    assert!(direct(&b, &bore).is_none());
    assert_eq!(delete_faces(&b, &bore).is_ok(), general(&b, &bore).is_ok());
}

#[test]
fn a_part_turned_askew_closes_the_same_way() {
    // Askew, the boxes of the groove and of the shaft's ends meet; the ends' planes still clear
    // the groove.
    let mut b = turned(&[
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 3.0],
        [2.0, 3.0],
        [2.0, 6.0],
        [3.0, 6.0],
        [3.0, 10.0],
        [0.0, 10.0],
    ]);
    b.transform(&Transform::rotation(
        Point3::new(1.0, 2.0, 3.0),
        Vec3::new(1.0, 2.0, 0.5),
        0.7,
    ));
    let out = both(&b, &named(&b, &["m2", "m3", "m4"]));
    close(measure::volume(&out), 90.0 * PI);
    let mut b = chamfered();
    b.transform(&Transform::rotation(
        Point3::ZERO,
        Vec3::new(1.0, 1.0, 1.0),
        1.1,
    ));
    close(measure::volume(&both(&b, &named(&b, &["e2"]))), 96.0);
}
