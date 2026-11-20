//! Chamfers and fillets of edges picked by id: the result's faces keep the names they had.

use std::f64::consts::PI;

use springrep::{
    blend::{EdgeName, chamfer_distances, chamfer_edges, fillet_conic, fillet_edges, fillet_radii},
    boolean::union,
    edit,
    math::{Frame, Point3, Vec2, Vec3},
    measure,
    pick::edges_named,
    profile::Profile,
    sweep::extrude,
    topo::{Brep, EdgeId},
};

/// A 20 × 20 block 10 tall, and its one edge between the `u_max` and `v_max` sides.
fn block_and_edge() -> (Brep, Vec<EdgeId>) {
    let square = Profile::rect(Vec2::ZERO, 20.0, 20.0);
    let b = extrude(&square, &Frame::WORLD, 10.0, "block").unwrap();
    let pair = EdgeName {
        a: "block.u_max",
        b: "block.v_max",
        exact: false,
    };
    let edges = edges_named(&b, &pair).unwrap();
    assert_eq!(edges.len(), 1);
    (b, edges)
}

const OWN: [&str; 6] = [
    "block.start",
    "block.end",
    "block.u_min",
    "block.u_max",
    "block.v_min",
    "block.v_max",
];

#[test]
fn a_fillet_of_an_edge_by_id_keeps_every_face_name() {
    let (b, edges) = block_and_edge();
    let out = fillet_edges(&b, &edges, 2.0, "f").unwrap();
    // The corner's 2 × 2 square less a quarter disc of radius 2 goes, 10 tall.
    let want = 4000.0 - 10.0 * (4.0 - PI);
    assert!((measure::volume(&out) - want).abs() < 1e-9 * want);
    let names = out.name_counts();
    for n in OWN {
        assert_eq!(names.get(n), Some(&1), "{n} in {names:?}");
    }
    assert_eq!(names.get("f.round_0"), Some(&1), "{names:?}");
    assert_eq!(names.len(), 7, "{names:?}");
}

#[test]
fn a_chamfer_of_an_edge_by_id_keeps_every_face_name() {
    let (b, edges) = block_and_edge();
    let out = chamfer_edges(&b, &edges, 2.0, "c").unwrap();
    // The corner's 2 × 2 triangle goes, 10 tall.
    let want = 4000.0 - 10.0 * 2.0;
    assert!((measure::volume(&out) - want).abs() < 1e-9 * want);
    let names = out.name_counts();
    for n in OWN {
        assert_eq!(names.get(n), Some(&1), "{n} in {names:?}");
    }
    assert_eq!(names.get("c.bevel_0"), Some(&1), "{names:?}");
    assert_eq!(names.len(), 7, "{names:?}");
}

#[test]
fn a_shaped_blend_names_what_it_makes_under_its_feature() {
    let (b, edges) = block_and_edge();
    let own = |names: &std::collections::BTreeMap<String, usize>| {
        OWN.iter().all(|n| names.get(*n) == Some(&1))
    };
    let radii = fillet_radii(&b, &[(edges[0], 2.0)], "r")
        .unwrap()
        .name_counts();
    assert!(
        own(&radii) && radii.get("r.round") == Some(&1) && radii.len() == 7,
        "{radii:?}"
    );
    let conic = fillet_conic(&b, &edges, 2.0, 0.5, "k")
        .unwrap()
        .name_counts();
    assert!(
        own(&conic) && conic.get("k.round") == Some(&1) && conic.len() == 7,
        "{conic:?}"
    );
    let bevels = chamfer_distances(&b, &[(edges[0], 2.0)], "c")
        .unwrap()
        .name_counts();
    assert!(
        own(&bevels) && bevels.get("c.bevel") == Some(&1) && bevels.len() == 7,
        "{bevels:?}"
    );
    // A block grown by 1 mm: its six faces, a round on each of twelve edges, a ball's patch at
    // each of eight corners.
    let grown = edit::offset(&b, 1.0, "g").unwrap().name_counts();
    assert!(
        own(&grown) && grown.get("g.corner") == Some(&8),
        "{grown:?}"
    );
    assert_eq!(
        grown.keys().filter(|n| n.starts_with("g.round_")).count(),
        12,
        "{grown:?}"
    );
    assert_eq!(grown.len(), 6 + 12 + 1, "{grown:?}");
}

/// The edges of `b` along the line through `on` in direction `dir`.
fn edges_along(b: &Brep, on: Point3, dir: Vec3) -> Vec<EdgeId> {
    b.edge_ids()
        .filter(|&e| {
            let edge = b.edge(e);
            let Some(c) = edge.curve.as_ref() else {
                return false;
            };
            [edge.range.0, edge.range.1]
                .iter()
                .all(|&t| (c.eval(t) - on).cross(dir).norm() < 1e-9)
        })
        .collect()
}

#[test]
fn a_fillet_of_one_half_of_a_split_edge_rounds_the_whole_of_it() {
    // A 20 × 20 block 10 tall joined to its mirror image across its x = 10 side: the union's top
    // and back faces each span both, and the edge between them stays two edges, split at the join.
    let (left, _) = block_and_edge();
    let mut right = left.clone();
    let mirror = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    right
        .transform_similar(mirror, Vec3::new(20.0, 0.0, 0.0))
        .unwrap();
    let b = union(&left, &right).unwrap();
    let ridge = edges_along(&b, Point3::new(0.0, 10.0, 10.0), Vec3::X);
    assert_eq!(ridge.len(), 2, "the model's edge is split in two");
    let out = fillet_edges(&b, &ridge[..1], 2.0, "f").unwrap();
    // The 2 × 2 square less a quarter disc of radius 2 goes along the whole 40 of the edge.
    let want = 8000.0 - 40.0 * (4.0 - PI);
    assert!((measure::volume(&out) - want).abs() < 1e-9 * want);
}
