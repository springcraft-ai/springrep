//! Fillets along edges no exact surface rounds, built round the rolling ball: the rim a rod, a
//! slot, a D and a bitten boss make on a sloping plane. Every point of each blend is held to its
//! radius from the curve the ball's centre runs on, worked by hand.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use springrep::{
    blend::fillet,
    boolean::{difference, unify, union},
    check,
    geom::{Curve, Surface},
    loft::{Section, loft},
    math::{Frame, Vec2, Vec3},
    measure,
    profile::{NamedSegment, Profile},
    sweep::extrude,
    topo::{Brep, EdgeId},
};

fn up(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
}

fn line(a: (f64, f64), b: (f64, f64), name: &str) -> NamedSegment {
    NamedSegment::line(Vec2::new(a.0, a.1), Vec2::new(b.0, b.1), name)
}

fn arc(center: (f64, f64), radius: f64, start: f64, sweep: f64, name: &str) -> NamedSegment {
    NamedSegment::arc(Vec2::new(center.0, center.1), radius, start, sweep, name)
}

/// A 20 × 20 block, its top cut to the plane z = 7.5 − x/4.
fn sloped_block() -> Brep {
    let block = extrude(
        &Profile::rect(Vec2::ZERO, 20.0, 20.0),
        &up(0.0),
        10.0,
        "block",
    )
    .unwrap();
    let sections = [(4.0, 14.0, 16.0), (11.0, -14.0, 16.0)].map(|(z, x0, x1): (f64, f64, f64)| {
        Section::new(
            Profile::rect(Vec2::new(0.5 * (x0 + x1), 0.0), x1 - x0, 30.0),
            up(z),
        )
    });
    difference(&block, &loft(&sections, true, "slope").unwrap()).unwrap()
}

/// `profile` standing 14 high from the xy plane through the sloped block's top.
fn standing(profile: &Profile, id: &str) -> Brep {
    union(
        &sloped_block(),
        &extrude(profile, &up(0.0), 14.0, id).unwrap(),
    )
    .unwrap()
}

/// The sloped block with a ⌀6 hole up z through the slope, or a ⌀6 boss 14 high standing through
/// it: the rim on the slope is an ellipse.
fn sloped_rod(boss: bool) -> Brep {
    let circle = Profile::circle(Vec2::ZERO, 6.0);
    if boss {
        standing(&circle, "rod")
    } else {
        let rod = extrude(&circle, &up(0.0), 12.0, "rod").unwrap();
        difference(&sloped_block(), &rod).unwrap()
    }
}

/// How far `p` is from the circle ρ = 4 about the z axis lifted onto the plane z = 7.5 − x/4 +
/// `offset`, searched either side of `p`'s own angle.
fn from_lifted_circle(p: Vec3, offset: f64) -> f64 {
    let centre = |phi: f64| {
        let (x, y) = (4.0 * phi.cos(), 4.0 * phi.sin());
        Vec3::new(x, y, 7.5 - x / 4.0 + offset)
    };
    let (mut lo, mut hi) = (p.y.atan2(p.x) - 0.6, p.y.atan2(p.x) + 0.6);
    for _ in 0..200 {
        let (a, c) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if centre(a).dist(p) < centre(c).dist(p) {
            hi = c;
        } else {
            lo = a;
        }
    }
    centre(0.5 * (lo + hi)).dist(p)
}

/// The points of edge `e` at 17 even steps of its parameter.
fn along(b: &Brep, e: EdgeId) -> Vec<Vec3> {
    let x = b.edge(e);
    let curve = x.curve.as_ref().unwrap();
    (0..=16)
        .map(|k| curve.eval(x.range.0 + (x.range.1 - x.range.0) * k as f64 / 16.0))
        .collect()
}

#[test]
fn a_rolling_ball_rounds_a_rod_through_a_slope() {
    // The ball rolls round the ellipse where the rod meets the plane z + x/4 = 7.5, so its centre
    // stays on the cylinder ρ = 3 + r, offset r from the plane: into the block round the hole,
    // above the slope round the boss. Every point of the blend lies r from that centre curve, and
    // each edge of the blend lies on its face.
    let r = 1.0;
    let lean = (1.0f64 + 1.0 / 16.0).sqrt();
    for boss in [false, true] {
        let b = fillet(&sloped_rod(boss), &[("rod.side", "slope.side")], r, "rim").unwrap();
        let b = unify(&b).unwrap();
        assert!(check::check(&b).is_empty());
        let plain = measure::volume(&unify(&sloped_rod(boss)).unwrap());
        let filled = measure::volume(&b);
        assert!(if boss { filled > plain } else { filled < plain });

        let offset = if boss { r * lean } else { -r * lean };
        let f = b
            .face_ids()
            .find(|&f| b.face(f).names.iter().any(|n| n.feature == "rim"))
            .expect("the blend face");
        let Surface::Nurbs(s) = &b.face(f).surface else {
            panic!("the blend is a NURBS surface");
        };
        let ((u0, u1), (v0, v1)) = s.domain();
        let on_plane = |p: Vec3| ((p.z + p.x / 4.0 - 7.5) / lean).abs() < 1e-8;
        let on_rod = |p: Vec3| (p.x.hypot(p.y) - 3.0).abs() < 1e-8;
        for i in 0..200 {
            let u = u0 + (u1 - u0) * i as f64 / 200.0;
            for j in 0..=8 {
                let p = s.eval(u, v0 + (v1 - v0) * j as f64 / 8.0);
                assert!(
                    (from_lifted_circle(p, offset) - r).abs() < 1e-8,
                    "boss {boss}: {p:?}"
                );
            }
            let (a, c) = (s.eval(u, v0), s.eval(u, v1));
            assert!(
                on_plane(a) && on_rod(c) || on_rod(a) && on_plane(c),
                "boss {boss}: {a:?} {c:?}"
            );
        }
    }
}

/// The sloped block with a slot 14 × 6 standing 14 high through it: flat sides y = ±3 for |x| ≤ 4
/// and round ends of radius 3 about (±4, 0).
fn sloped_slot() -> Brep {
    let slot = Profile::outline(vec![
        line((-4.0, -3.0), (4.0, -3.0), "mid_v_min"),
        arc((4.0, 0.0), 3.0, -FRAC_PI_2, PI, "e2_side"),
        line((4.0, 3.0), (-4.0, 3.0), "mid_v_max"),
        arc((-4.0, 0.0), 3.0, FRAC_PI_2, PI, "e1_side"),
    ]);
    standing(&slot, "slot")
}

#[test]
fn a_rolling_ball_rounds_a_slot_through_a_slope_across_its_tangent_faces() {
    // Round the slot's foot the ball touches the slope and, by turns, each flat side and each
    // round end, sliding from one onto the next where they meet tangentially. Its centre runs on
    // the slot's outline offset by r (half-width 4, ends of radius 4), lifted onto the plane r
    // above the slope; every point of the blend lies r from that curve.
    let r = 1.0;
    let lean = (1.0f64 + 1.0 / 16.0).sqrt();
    let b = fillet(
        &sloped_slot(),
        &[("slot.mid_v_min", "slope.side")],
        r,
        "foot",
    )
    .unwrap();
    let b = unify(&b).unwrap();
    assert!(check::check(&b).is_empty());
    let plain = measure::volume(&unify(&sloped_slot()).unwrap());
    assert!(measure::volume(&b) > plain);

    let lift = |x: f64, y: f64| Vec3::new(x, y, 7.5 - x / 4.0 + r * lean);
    let to_segment = |p: Vec3, a: Vec3, c: Vec3| {
        let d = c - a;
        let t = ((p - a).dot(d) / d.dot(d)).clamp(0.0, 1.0);
        (a + d * t).dist(p)
    };
    let to_arc = |p: Vec3, cx: f64, from: f64| {
        let at = |phi: f64| lift(cx + 4.0 * phi.cos(), 4.0 * phi.sin());
        let (mut lo, mut hi) = (from, from + PI);
        for _ in 0..200 {
            let (a, c) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if at(a).dist(p) < at(c).dist(p) {
                hi = c;
            } else {
                lo = a;
            }
        }
        at(0.5 * (lo + hi)).dist(p)
    };
    let from_centres = |p: Vec3| {
        to_segment(p, lift(-4.0, 4.0), lift(4.0, 4.0))
            .min(to_segment(p, lift(-4.0, -4.0), lift(4.0, -4.0)))
            .min(to_arc(p, 4.0, -FRAC_PI_2))
            .min(to_arc(p, -4.0, FRAC_PI_2))
    };
    let on_slope = |p: Vec3| ((p.z + p.x / 4.0 - 7.5) / lean).abs() < 1e-8;
    let on_slot = |p: Vec3| {
        let wall = if p.x.abs() <= 4.0 {
            p.y.abs() - 3.0
        } else {
            (p.x.abs() - 4.0).hypot(p.y) - 3.0
        };
        wall.abs() < 1e-8
    };
    let blends: Vec<_> = b
        .face_ids()
        .filter(|&f| b.face(f).names.iter().any(|n| n.feature == "foot"))
        .collect();
    assert!(blends.len() >= 4, "{} blend faces", blends.len());
    for f in blends {
        let Surface::Nurbs(s) = &b.face(f).surface else {
            panic!("the blend is a NURBS surface");
        };
        let ((u0, u1), (v0, v1)) = s.domain();
        for i in 0..=40 {
            let u = u0 + (u1 - u0) * i as f64 / 40.0;
            for j in 0..=8 {
                let p = s.eval(u, v0 + (v1 - v0) * j as f64 / 8.0);
                assert!((from_centres(p) - r).abs() < 1e-8, "{p:?}");
            }
            let (a, c) = (s.eval(u, v0), s.eval(u, v1));
            assert!(
                on_slot(a) && on_slope(c) || on_slope(a) && on_slot(c),
                "{a:?} {c:?}"
            );
        }
    }
}

/// The sloped block with a D 14 high standing through it: the ⌀6 circle less its half x > 0, so
/// its round wall meets a flat back on the plane x = 0.
fn sloped_d() -> Brep {
    let d = Profile::outline(vec![
        arc((0.0, 0.0), 3.0, FRAC_PI_2, PI, "round_side"),
        line((0.0, -3.0), (0.0, 3.0), "back_u_min"),
    ]);
    standing(&d, "d")
}

#[test]
fn a_rolling_ball_round_an_open_curved_edge_is_trimmed_by_the_face_at_each_end() {
    // The D's round wall meets the slope in half an ellipse, ending at y = ±3 on its flat back,
    // which it leaves at about 14°. Past each end the ball rolls on between the same two surfaces,
    // so every point of the blend still lies r from the circle ρ = 4 lifted r above the slope;
    // the flat back trims it, so its trim curves lie on x = 0.
    let r = 1.0;
    let lean = (1.0f64 + 1.0 / 16.0).sqrt();
    let b = fillet(&sloped_d(), &[("d.round_side", "slope.side")], r, "foot").unwrap();
    let b = unify(&b).unwrap();
    assert!(check::check(&b).is_empty());
    let plain = measure::volume(&unify(&sloped_d()).unwrap());
    assert!(measure::volume(&b) > plain);

    let blends: Vec<_> = b
        .face_ids()
        .filter(|&f| b.face(f).names.iter().any(|n| n.feature == "foot"))
        .collect();
    assert!(!blends.is_empty());
    let mut on_back = 0;
    for f in blends {
        let Surface::Nurbs(s) = &b.face(f).surface else {
            panic!("the blend is a NURBS surface");
        };
        // Sample the face itself: points of its surface inside its loops lie on the kept side.
        let ((u0, u1), (v0, v1)) = s.domain();
        for i in 0..=40 {
            let u = u0 + (u1 - u0) * i as f64 / 40.0;
            for j in 0..=8 {
                let p = s.eval(u, v0 + (v1 - v0) * j as f64 / 8.0);
                if p.x <= 0.0 {
                    assert!((from_lifted_circle(p, r * lean) - r).abs() < 1e-8, "{p:?}");
                }
            }
        }
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let e = b.coedge(c).edge;
                let points = along(&b, e);
                let on_plane = points.iter().all(|p| p.x.abs() < 1e-8);
                let at_back = points.iter().any(|p| p.x.abs() < 1e-3);
                let spline = matches!(b.edge(e).curve, Some(Curve::BSpline(_)));
                if at_back && !spline {
                    continue;
                }
                if on_plane {
                    on_back += 1;
                }
            }
        }
    }
    assert!(on_back >= 2, "{on_back} trim edges on the back");
}

/// The sloped block with a boss 14 high standing through it: the ⌀6 circle less the ⌀8 circle
/// about (5, 0), so its round wall meets a hollow back, the cylinder of radius 4 about x = 5, at
/// (1.8, ±2.4).
fn sloped_bitten() -> Brep {
    let (round, bite) = (2.4f64.atan2(1.8), 2.4f64.atan2(3.2));
    let boss = Profile::outline(vec![
        arc((0.0, 0.0), 3.0, round, TAU - 2.0 * round, "round_side"),
        arc((5.0, 0.0), 4.0, bite - PI, -2.0 * bite, "bite_side"),
    ]);
    standing(&boss, "boss")
}

#[test]
fn a_rolling_ball_round_an_open_curved_edge_ends_on_a_cylinder() {
    // The boss's round wall meets the slope in an arc of ellipse ending on its hollow back. The
    // ball rolls round it as round a full rod, its centre on the circle ρ = 4 lifted r above the
    // slope, and the back's cylinder trims it at each end: the trim curves lie on that cylinder.
    let r = 1.0;
    let lean = (1.0f64 + 1.0 / 16.0).sqrt();
    let b = fillet(
        &sloped_bitten(),
        &[("boss.round_side", "slope.side")],
        r,
        "foot",
    )
    .unwrap();
    let b = unify(&b).unwrap();
    assert!(check::check(&b).is_empty());
    let plain = measure::volume(&unify(&sloped_bitten()).unwrap());
    assert!(measure::volume(&b) > plain);

    let off_back = |p: Vec3| (p.x - 5.0).hypot(p.y) - 4.0;
    let mut on_back = 0;
    for f in b.face_ids() {
        if !b.face(f).names.iter().any(|n| n.feature == "foot") {
            continue;
        }
        let Surface::Nurbs(s) = &b.face(f).surface else {
            panic!("the blend is a NURBS surface");
        };
        let ((u0, u1), (v0, v1)) = s.domain();
        for i in 0..=40 {
            let u = u0 + (u1 - u0) * i as f64 / 40.0;
            for j in 0..=8 {
                let p = s.eval(u, v0 + (v1 - v0) * j as f64 / 8.0);
                if off_back(p) >= 0.0 {
                    assert!((from_lifted_circle(p, r * lean) - r).abs() < 1e-8, "{p:?}");
                }
            }
        }
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let e = b.coedge(c).edge;
                if !matches!(b.edge(e).curve, Some(Curve::BSpline(_))) {
                    continue;
                }
                if along(&b, e).iter().all(|&p| off_back(p).abs() < 1e-8) {
                    on_back += 1;
                }
            }
        }
    }
    assert!(on_back >= 2, "{on_back} trim edges on the back");
}

#[test]
fn a_rolling_ball_round_a_closed_junction_across_a_seam_matches_one_clear_of_it() {
    use springrep::primitive::cylinder;
    // A tee: the bar of radius 6 along x joined by the stem of radius 3 up z, the saddle junction
    // rounded 1. With the bar's seam along its top the junction crosses it twice and the bar's
    // loop meets the chain in two runs; with the seam along its side the junction stays clear of
    // it. The round is the same solid both ways.
    let rounded = |bar_frame: Frame| {
        let bar = cylinder(&bar_frame, 6.0, 40.0, "bar").unwrap();
        let stem = cylinder(&Frame::WORLD, 3.0, 15.0, "stem").unwrap();
        let tee = union(&bar, &stem).unwrap();
        let b = fillet(&tee, &[("bar.side", "stem.side")], 1.0, "r").unwrap();
        assert!(check::check(&b).is_empty());
        (measure::volume(&b), measure::area(&b))
    };
    let start = Vec3::new(-20.0, 0.0, 0.0);
    let across = rounded(Frame::new(start, Vec3::X, Vec3::Z).unwrap());
    let clear = rounded(Frame::new(start, Vec3::X, Vec3::Y).unwrap());
    assert!(
        (across.0 - clear.0).abs() < 1e-9 * clear.0,
        "{across:?} {clear:?}"
    );
    assert!(
        (across.1 - clear.1).abs() < 1e-9 * clear.1,
        "{across:?} {clear:?}"
    );
}

#[test]
fn a_rolling_ball_past_the_edge_of_a_spline_face_rolls_on_its_continuation() {
    use springrep::primitive::slab;
    // A slab on the square 20 × 20 whose spline top is the plane z = 5 + x/10 + y/20, its edge on
    // the front y = 0 rounded 1. The round is the cylinder along d = (1, 0, 1/10)/√1.01 whose
    // section turns through the angle φ between the faces' normals, of area tan(φ/2) − φ/2 past
    // the arc; the sides x = 0 and x = 20 trim it on a slant, each losing that section stretched
    // by √1.01. Past each side the ball rolls on the top's continuation beyond its parameters.
    let heights: Vec<Vec<f64>> = (0..4)
        .map(|i| (0..4).map(|j| 5.0 + (2 * i + j) as f64 / 3.0).collect())
        .collect();
    let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let body = slab(&Frame::WORLD, [20.0, 20.0], (&knots, &knots), &heights, "s").unwrap();
    let b = fillet(&body, &[("s.top", "s.front")], 1.0, "r").unwrap();
    assert!(check::check(&b).is_empty());
    let phi = (0.05 / 1.0125f64.sqrt()).acos();
    let section = (phi / 2.0).tan() - phi / 2.0;
    let length = 20.0 * 1.01f64.sqrt();
    let volume = 2600.0 - section * length;
    let area = 400.0 * 1.0125f64.sqrt() + 120.0 + 140.0 + 110.0 + 150.0 + 400.0
        - 2.0 * (phi / 2.0).tan() * length
        + phi * length
        - 2.0 * section * 1.01f64.sqrt();
    let (v, a) = (measure::volume(&b), measure::area(&b));
    assert!(
        (v - volume).abs() < 1e-9 * volume,
        "volume {v} against {volume}"
    );
    assert!((a - area).abs() < 1e-9 * area, "area {a} against {area}");
}
