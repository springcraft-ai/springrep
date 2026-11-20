//! Rounds that end on a face cut at a slant: where a round on a leaning edge meets its neighbour's
//! they cross in a crease, and a concave round that runs past a short edge is trimmed by the ramp
//! beyond it.

use springrep::{
    blend::fillet,
    boolean::{difference, unify, union},
    check,
    geom::Surface,
    loft::{Section, loft},
    math::{Frame, Vec2, Vec3},
    measure,
    profile::Profile,
    sweep::extrude,
    topo::Brep,
};

fn up(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
}

/// A wedge ruled between two rectangles 30 deep, each from `x0` to `x1` on the plane at its `z`.
fn wedge(id: &str, sections: [(f64, f64, f64); 2]) -> Brep {
    let sections = sections.map(|(z, x0, x1)| {
        Section::new(
            Profile::rect(Vec2::new(0.5 * (x0 + x1), 0.0), x1 - x0, 30.0),
            up(z),
        )
    });
    loft(&sections, true, id).unwrap()
}

/// The 20 × 20 × 10 block with its +x top edge cut back along x + z = 15, a 135° edge on the top.
fn leaning_block() -> Brep {
    let block = extrude(
        &Profile::rect(Vec2::ZERO, 20.0, 20.0),
        &up(0.0),
        10.0,
        "block",
    )
    .unwrap();
    let lean = wedge("lean", [(5.0, 10.0, 11.0), (11.0, 4.0, 11.0)]);
    difference(&block, &lean).unwrap()
}

/// A triangular prism 10 tall on the xy plane whose first side is `wall`.
fn corner_cut(id: &str, points: [(f64, f64); 3]) -> Brep {
    let points = points.map(|(x, y)| Vec2::new(x, y));
    extrude(
        &Profile::polygon(&points, &["wall", "top", "back"]),
        &up(0.0),
        10.0,
        id,
    )
    .unwrap()
}

#[test]
fn rounds_on_a_square_and_a_leaning_edge_meet_in_a_crease() {
    let b = fillet(
        &leaning_block(),
        &[("block.end", "block.v_max"), ("block.end", "lean.side")],
        1.5,
        "rim",
    )
    .unwrap();
    assert!(check::check(&unify(&b).unwrap()).is_empty());
}

#[test]
fn rounds_on_a_diagonal_and_a_leaning_edge_meet_in_a_crease() {
    // A vertical wall along y = x + 2 cuts the block's +y corner, meeting the leaning edge at 45°.
    let diag = corner_cut("diag", [(-15.0, -13.0), (15.0, 17.0), (-15.0, 17.0)]);
    let b = fillet(
        &difference(&leaning_block(), &diag).unwrap(),
        &[("block.end", "diag.wall"), ("block.end", "lean.side")],
        1.5,
        "rim",
    )
    .unwrap();
    assert!(check::check(&unify(&b).unwrap()).is_empty());
}

#[test]
fn a_leaning_edges_round_creases_into_diagonal_rounds_at_both_ends() {
    // Walls along y = ±(x + 2) cut both of the block's +x corners, so both ends of the
    // leaning edge's round crease into a diagonal round.
    let diag = corner_cut("diag", [(-15.0, -13.0), (15.0, 17.0), (-15.0, 17.0)]);
    let diag2 = corner_cut("diag2", [(15.0, -17.0), (-15.0, 13.0), (-15.0, -17.0)]);
    let b = difference(&difference(&leaning_block(), &diag).unwrap(), &diag2).unwrap();
    let b = fillet(&b, &[("diag.wall", "diag2.wall")], 3.0, "corner").unwrap();
    let b = fillet(
        &b,
        &[("block.end", "diag.wall"), ("block.end", "lean.side")],
        1.5,
        "rim",
    )
    .unwrap();
    assert!(check::check(&unify(&b).unwrap()).is_empty());
}

#[test]
fn a_concave_round_past_a_short_edge_is_trimmed_by_the_ramp_beyond_it() {
    // A strut 10 × 10 × 10 (x 0–10, y ±5) against a block (x 10–15, y ±15, z 0–15), its top cut
    // back above z = x + 1, which leaves a 1 mm edge of its top at the block. The R2 rounds where
    // its walls meet the block run up to the top; past that 1 mm edge the ramp crosses their strip.
    let rect = |id: &str, cx: f64, w: f64, h: f64, d: f64| {
        extrude(&Profile::rect(Vec2::new(cx, 0.0), w, h), &up(0.0), d, id).unwrap()
    };
    let r = 2.0;
    let block = rect("block", 12.5, 5.0, 30.0, 15.0);
    let strut = rect("strut", 5.0, 10.0, 10.0, 10.0);
    let ramp = wedge("ramp", [(1.0, -1.0, 0.0), (10.5, -1.0, 9.5)]);
    let b = difference(&union(&block, &strut).unwrap(), &ramp).unwrap();
    let b = fillet(
        &b,
        &[
            ("strut.v_min", "block.u_min"),
            ("strut.v_max", "block.u_min"),
        ],
        r,
        "root",
    )
    .unwrap();
    let b = unify(&b).unwrap();
    assert!(check::check(&b).is_empty());
    // Each round's section is the R2 spandrel: at distance u from the block it is
    // L(u) = r − √(r² − (r − u)²) wide, and stands 10 high for u ≤ 1 and 11 − u (the ramp) beyond.
    // With t = r − u and S = √(r² − t²): ∫S dt = (t·S + r²·asin(t/r))/2 and
    // ∫t·S dt = −(r² − t²)^1.5/3.
    let s = |t: f64| (r * r - t * t).max(0.0).sqrt();
    let i0 = |t: f64| 0.5 * (t * s(t) + r * r * (t / r).asin());
    let i1 = |t: f64| -(r * r - t * t).powf(1.5) / 3.0;
    let flat = 10.0 * (r - (i0(r) - i0(r - 1.0)));
    let (t0, t1) = (0.0, r - 1.0);
    let sloped = (11.0 - r) * r * (t1 - t0) + r * (t1 * t1 - t0 * t0) / 2.0
        - ((11.0 - r) * (i0(t1) - i0(t0)) + (i1(t1) - i1(t0)));
    // Block 5 × 30 × 15, the strut less the ramp's 9 × 9 / 2 triangle over its 10 width, and two
    // rounds.
    let want = 2250.0 + (1000.0 - 405.0) + 2.0 * (flat + sloped);
    let got = measure::volume(&b);
    assert!((got - want).abs() < 1e-9 * want, "volume {got} for {want}");
    // Nothing of the rounds is left past the ramp: no face lies on a wall's plane beyond it.
    for f in b.face_ids() {
        if let Surface::Plane(p) = &b.face(f).surface
            && p.frame.z.y.abs() > 1.0 - 1e-12
            && (p.frame.origin.y.abs() - 5.0).abs() < 1e-9
        {
            for lp in &b.face(f).loops {
                for &c in &b.lp(*lp).coedges {
                    let at = b.vertex(b.coedge_start(c)).point;
                    assert!(at.z <= at.x + 1.0 + 1e-9, "a wall face reaches {at:?}");
                }
            }
        }
    }
}
