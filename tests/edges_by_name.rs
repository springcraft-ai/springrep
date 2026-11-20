//! Chamfers and fillets whose edges are picked by the names of the two faces that meet in them:
//! names a sweep gave from its profile, names on a revolved solid, an edge a keyway interrupts,
//! and a pair of surfaces that meet in more than one edge.

use std::f64::consts::{PI, TAU};

use springrep::{
    blend::{EdgeName, chamfer, fillet_named},
    boolean::{difference, unify, union},
    check,
    math::{Frame, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    profile::Profile,
    sweep::{extrude, revolve},
    topo::Brep,
};

/// How far a through hole runs, so the cut leaves no skin.
const THROUGH: f64 = 1000.0;

fn up(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
}

/// A `width` × `width` block 10 tall on the xy plane, centred on the origin.
fn block(width: f64) -> Brep {
    extrude(
        &Profile::rect(Vec2::ZERO, width, width),
        &up(0.0),
        10.0,
        "block",
    )
    .unwrap()
}

fn watertight(b: &Brep) -> bool {
    let opt = MeshOptions::new(0.01, 0.2);
    let mesh = mesh::tessellate(b, &opt).expect("the result meshes");
    let solid = b.solid_ids().next().expect("one solid");
    mesh::report(b, &mesh, solid).watertight()
}

fn faces_of(b: &Brep, kind: &str) -> usize {
    b.face_ids()
        .filter(|&f| b.face(f).surface.kind() == kind)
        .count()
}

#[test]
fn a_chamfer_bevels_the_named_edge() {
    let cube = extrude(&Profile::rect(Vec2::ZERO, 4.0, 4.0), &up(0.0), 4.0, "block").unwrap();
    let b = chamfer(&cube, &[("block.end", "block.u_max")], 1.0, "ch").unwrap();
    let b = unify(&b).unwrap();
    let got = measure::volume(&b);
    assert!((got - 62.0).abs() < 1e-9 * 62.0, "{got}");
    assert_eq!(b.name_counts().get("ch.bevel_0"), Some(&1));
}

#[test]
fn a_chamfer_picks_a_revolved_shafts_edge_by_its_profile_names() {
    // A 10 mm radius, 10 mm long shaft turned from a rectangle, its end ring bevelled 1 mm: the
    // bevel takes a 1 × 1 / 2 triangle whose centroid turns at 10 − 1/3 (Pappus).
    let rz = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let outline = Profile::polygon(
        &[
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ],
        &["face_a", "journal", "face_b", "axis"],
    );
    let shaft = revolve(&outline, &rz, Vec3::ZERO, rz.y, TAU, "shaft").unwrap();
    let b = chamfer(&shaft, &[("shaft.face_a", "shaft.journal")], 1.0, "break").unwrap();
    let b = unify(&b).unwrap();
    let want = 1000.0 * PI - 2.0 * PI * (10.0 - 1.0 / 3.0) * 0.5;
    assert!((measure::volume(&b) - want).abs() < 1e-9 * want);
    let names = b.name_counts();
    for n in ["shaft.face_a", "shaft.journal", "break.bevel_0"] {
        assert_eq!(names.get(n), Some(&1), "{n} in {names:?}");
    }
}

/// A 20 × 20 × 10 plate with a ⌀8 bore through it and a keyway 2 wide cut 1 past the bore's wall
/// on +x, the bore's edge on the plate's underside bevelled 0.5 when `chamfered`.
fn keyed_bore(chamfered: bool) -> Brep {
    let plate = extrude(
        &Profile::rect(Vec2::ZERO, 20.0, 20.0),
        &up(0.0),
        10.0,
        "plate",
    )
    .unwrap();
    let bore = extrude(
        &Profile::named_circle(Vec2::ZERO, 8.0, "wall"),
        &up(0.0),
        THROUGH,
        "bore",
    )
    .unwrap();
    let key = extrude(
        &Profile::rect(Vec2::new(4.25, 0.0), 1.5, 2.0),
        &up(0.0),
        10.0,
        "key",
    )
    .unwrap();
    let mut b = difference(&difference(&plate, &bore).unwrap(), &key).unwrap();
    if chamfered {
        b = chamfer(&b, &[("bore.wall", "plate.start")], 0.5, "break").unwrap();
    }
    unify(&b).unwrap()
}

#[test]
fn a_chamfer_round_most_of_a_bore_stops_at_the_keyway_walls() {
    // The bore's edge runs from one keyway wall (y = 1) round to the other (y = −1), 331°. The
    // chamfer's triangle, 0.5 mm each way, turns everywhere the keyway leaves material: at radius
    // r it is 4.5 − r tall and misses the 2·asin(1/r) the keyway takes. Each end is trimmed by
    // its own wall, which the far end lies across.
    let removed = measure::volume(&keyed_bore(false)) - measure::volume(&keyed_bore(true));
    let n = 2000;
    let f = |r: f64| (4.5 - r) * r * (TAU - 2.0 * (1.0 / r).asin());
    let h = 0.5 / n as f64;
    let mut want = f(4.0) + f(4.5);
    for i in 1..n {
        want += f(4.0 + h * i as f64) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    want *= h / 3.0;
    assert!((removed - want).abs() < 1e-9 * want, "{removed} vs {want}");
}

/// The block with a 10 × 10 × 5 boss on its top from x = 5 to 15, overhanging its +x side: at
/// (10, ±5, 10) the block's top, the boss's side, its underside and the block's side meet, the top
/// and the underside on one plane.
fn overhang() -> Brep {
    let boss = extrude(
        &Profile::rect(Vec2::new(10.0, 0.0), 10.0, 10.0),
        &up(10.0),
        5.0,
        "o",
    )
    .unwrap();
    union(&block(20.0), &boss).unwrap()
}

#[test]
fn an_exact_edge_picks_one_of_the_edges_two_surfaces_meet_in() {
    // The boss's side meets the z = 10 plane twice along y = 5: over the overhang, convex, where
    // the plane is the boss's underside, and over the block, concave, where it is the block's top.
    // Named by surfaces the pair picks both; named exactly, only the one its faces make. Filleted
    // R1 on its convex edges alone, each 5 long and ending flush at the saddle, the part loses
    // 20 × (1 − π/4); on its concave ones alone (two of 5 and the underside's 10) it gains as much.
    let cut = 20.0 * (1.0 - PI / 4.0);
    for (edges, want) in [
        (
            [
                ("o.v_max", "o.start"),
                ("o.v_min", "o.start"),
                ("block.u_max", "block.end"),
            ],
            4500.0 - cut,
        ),
        (
            [
                ("block.end", "o.v_max"),
                ("block.end", "o.v_min"),
                ("o.start", "block.u_max"),
            ],
            4500.0 + cut,
        ),
    ] {
        let exact = edges.map(|(a, b)| EdgeName { a, b, exact: true });
        let b = unify(&fillet_named(&overhang(), &exact, 1.0, "f").unwrap()).unwrap();
        assert!(check::check(&b).is_empty());
        assert!(watertight(&b));
        let got = measure::volume(&b);
        assert!(
            (got - want).abs() < 1e-12 * want,
            "{edges:?}: volume {got} for {want}"
        );
        assert_eq!(faces_of(&b, "cylinder"), if want < 4500.0 { 4 } else { 3 });
    }
}
