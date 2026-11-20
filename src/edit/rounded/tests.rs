use std::f64::consts::PI;

use super::*;
use crate::{
    edit::offset,
    math::{Frame, Vec2, Vec3},
    measure, primitive,
    profile::Profile,
    sweep::extrude,
};

/// A prism on the polygon `pts`, `h` tall, its sides named `e0`, `e1`, … from the first edge.
fn prism(pts: &[[f64; 2]], h: f64) -> Brep {
    let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
    let names: Vec<String> = (0..pts.len()).map(|i| format!("e{i}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    extrude(&Profile::polygon(&pts, &names), &Frame::WORLD, h, "p").unwrap()
}

fn top(b: &Brep) -> Vec<FaceId> {
    let end = FaceName::new("p", "end");
    b.face_ids()
        .filter(|&f| b.face(f).names.contains(&end))
        .collect()
}

fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() < 1e-7 * want.max(1.0),
        "{got} against {want}"
    );
}

const L: [[f64; 2]; 6] = [
    [0.0, 0.0],
    [20.0, 0.0],
    [20.0, 5.0],
    [5.0, 5.0],
    [5.0, 15.0],
    [0.0, 15.0],
];

const DUMBBELL: [[f64; 2]; 12] = [
    [0.0, 0.0],
    [10.0, 0.0],
    [10.0, 4.0],
    [14.0, 4.0],
    [14.0, 0.0],
    [24.0, 0.0],
    [24.0, 10.0],
    [14.0, 10.0],
    [14.0, 6.0],
    [10.0, 6.0],
    [10.0, 10.0],
    [0.0, 10.0],
];

#[test]
fn a_block_shrunk_keeps_its_edges_sharp() {
    let b = primitive::block(&Frame::WORLD, Vec3::new(10.0, 8.0, 6.0), "b").unwrap();
    close(
        measure::volume(&offset(&b, -1.0, "o").unwrap()),
        8.0 * 6.0 * 4.0,
    );
}

#[test]
fn an_l_prism_shrunk_rounds_its_concave_edge_both_ways() {
    // The section 150 in area and 70 round, shrunk by d: A − P·d + k·d², where each of the five
    // convex right corners gives back d² and the concave one, rounded, takes π·d²/4; 8 − 2d tall.
    let b = prism(&L, 8.0);
    let d = 1.0;
    let want = (8.0 - 2.0 * d) * (150.0 - 70.0 * d + (5.0 - PI / 4.0) * d * d);
    close(
        measure::volume(&moved_and_rounded(&b, &[], -d, "o").unwrap()),
        want,
    );
    close(
        measure::volume(&near_inside(&b, &[], d, "o").unwrap()),
        want,
    );
}

#[test]
fn a_torus_shrunk_is_the_torus_of_the_thinner_tube() {
    let t = primitive::torus(&Frame::WORLD, 10.0, 3.0, "t").unwrap();
    let out = offset(&t, -1.5, "o").unwrap();
    close(measure::volume(&out), 2.0 * PI * PI * 10.0 * 1.5 * 1.5);
}

#[test]
fn a_dumbbell_shrunk_past_its_neck_keeps_both_halves() {
    // A neck 2 wide shrunk by 1.5 closes; each half's section is the square 7 × 7 and the bulge
    // between x = 8.5 and the arcs of radius 1.5 about the neck's corners, which meet at
    // (10 − √5/2, 5): β = 3 − √5/2 − (9/4)·asin(2/3). Each half is 6 − 3 tall.
    let b = prism(&DUMBBELL, 6.0);
    let out = offset(&b, -1.5, "o").unwrap();
    let beta = 3.0 - 5f64.sqrt() / 2.0 - 2.25 * (2.0f64 / 3.0).asin();
    close(measure::volume(&out), 2.0 * 3.0 * (49.0 + beta));
    assert_eq!(out.solid_ids().count(), 2);
}

#[test]
fn an_l_prism_shelled_round_has_the_quarter_cylinder_inside_its_corner() {
    // The cavity is the section shrunk by 1, 85 − π/4, from z = 1 up to the open top at z = 8.
    let b = prism(&L, 8.0);
    let out = shell_rounded(&b, &top(&b), 1.0, Wall::Inward, "w").unwrap();
    close(measure::volume(&out), 8.0 * 150.0 - 7.0 * (85.0 - PI / 4.0));
}

#[test]
fn a_shell_rounded_wall_closing_a_neck_takes_the_points_near_the_walls() {
    // As the sharp shell: the walls cross in the neck, and each pocket is the square 7 × 7 and the
    // bulge β, 4.5 deep, in the plate of 208 × 6.
    let b = prism(&DUMBBELL, 6.0);
    let out = shell_rounded(&b, &top(&b), 1.5, Wall::Inward, "w").unwrap();
    let beta = 3.0 - 5f64.sqrt() / 2.0 - 2.25 * (2.0f64 / 3.0).asin();
    close(
        measure::volume(&out),
        208.0 * 6.0 - 4.5 * 2.0 * (49.0 + beta),
    );
}

#[test]
fn a_box_shelled_outward_round_is_the_box_grown_up_to_its_top() {
    // The walls 2t·c·(a + b), the quarter cylinders up the four sides πt²c, the floor ab·t, the
    // quarter cylinders round its rim πt²(a + b)/2 and the four eighths of a ball 2πt³/3.
    let b = prism(&[[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]], 6.0);
    let out = shell_rounded(&b, &top(&b), 1.0, Wall::Outward, "w").unwrap();
    close(measure::volume(&out), 296.0 + 47.0 * PI / 3.0);
    let inner = FaceName::new("w", FaceName::INNER);
    let inside = out
        .face_ids()
        .filter(|&f| out.face(f).names.contains(&inner))
        .count();
    assert_eq!(inside, 5);
}
