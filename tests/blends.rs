//! Chamfers and fillets on small swept solids against closed-form volume and area, each result
//! checked valid and watertight, with the blend faces named by the operation.

use std::f64::consts::PI;

use springrep::{
    Error,
    blend::{chamfer, fillet},
    boolean::{difference, union},
    check,
    math::{Frame, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::extrude,
    topo::Brep,
};

const COARSE: MeshOptions = MeshOptions::new(0.2, 0.5);

fn at_z(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
}

fn block(id: &str, w: f64, d: f64, h: f64) -> Brep {
    extrude(
        &Profile::rect(Vec2::new(w / 2.0, d / 2.0), w, d),
        &at_z(0.0),
        h,
        id,
    )
    .unwrap()
}

fn disc(id: &str, z: f64, diameter: f64, h: f64) -> Brep {
    extrude(&Profile::circle(Vec2::ZERO, diameter), &at_z(z), h, id).unwrap()
}

fn assert_solid(b: &Brep, volume: f64, area: f64) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
    let v = measure::volume(b);
    let a = measure::area(b);
    assert!(
        (v - volume).abs() <= 1e-9 * volume,
        "volume {v} vs {volume}"
    );
    assert!((a - area).abs() <= 1e-9 * area, "area {a} vs {area}");
    let m = mesh::tessellate(b, &COARSE).unwrap();
    for s in b.solid_ids() {
        assert!(mesh::report(b, &m, s).watertight());
    }
}

fn names(b: &Brep, name: &str) -> usize {
    b.name_counts().get(name).copied().unwrap_or(0)
}

fn kinds(b: &Brep, kind: &str) -> usize {
    b.face_ids()
        .filter(|&f| b.face(f).surface.kind() == kind)
        .count()
}

#[test]
fn a_straight_convex_edge_takes_a_cylinder_fillet() {
    let b = fillet(
        &block("box", 40.0, 30.0, 20.0),
        &[("box.end", "box.v_min")],
        5.0,
        "f",
    )
    .unwrap();
    let spandrel = 25.0 * (1.0 - PI / 4.0);
    let area = 5200.0 - 2.0 * 5.0 * 40.0 - 2.0 * spandrel + PI / 2.0 * 5.0 * 40.0;
    assert_solid(&b, 24000.0 - spandrel * 40.0, area);
    assert_eq!(b.face_count(), 7);
    assert_eq!(kinds(&b, "cylinder"), 1);
    assert_eq!(names(&b, "f.round_0"), 1);
    assert_eq!(names(&b, "box.end"), 1);
}

#[test]
fn a_straight_concave_edge_takes_a_fillet_that_adds_material() {
    let pts = [
        Vec2::new(0.0, 0.0),
        Vec2::new(40.0, 0.0),
        Vec2::new(40.0, 10.0),
        Vec2::new(10.0, 10.0),
        Vec2::new(10.0, 30.0),
        Vec2::new(0.0, 30.0),
    ];
    let names_in = ["e0", "e1", "e2", "e3", "e4", "e5"];
    let ell = extrude(&Profile::polygon(&pts, &names_in), &at_z(0.0), 20.0, "l").unwrap();
    let b = fillet(&ell, &[("l.e2", "l.e3")], 4.0, "f").unwrap();
    let spandrel = 16.0 * (1.0 - PI / 4.0);
    let area = 2.0 * (600.0 + spandrel) + 20.0 * (140.0 - 8.0 + 2.0 * PI);
    assert_solid(&b, 600.0 * 20.0 + spandrel * 20.0, area);
    assert_eq!(names(&b, "f.round_0"), 1);
    assert_eq!(names(&b, "l.e2"), 1);
}

#[test]
fn a_circular_edge_takes_a_cone_chamfer() {
    let b = chamfer(
        &disc("c", 0.0, 40.0, 20.0),
        &[("c.end", "c.side")],
        2.0,
        "ch",
    )
    .unwrap();
    let volume = PI * 400.0 * 18.0 + PI * 2.0 / 3.0 * (400.0 + 360.0 + 324.0);
    let area = 400.0 * PI + 2.0 * PI * 20.0 * 18.0 + 324.0 * PI + PI * 38.0 * 2.0 * 2f64.sqrt();
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "cone"), 1);
    assert_eq!(names(&b, "ch.bevel_0"), 1);
}

#[test]
fn a_concave_circular_edge_takes_a_torus_fillet() {
    let plate = extrude(
        &Profile::rect(Vec2::ZERO, 60.0, 60.0),
        &at_z(0.0),
        10.0,
        "plate",
    )
    .unwrap();
    let body = union(&plate, &disc("boss", 10.0, 20.0, 10.0)).unwrap();
    let b = fillet(&body, &[("plate.end", "boss.side")], 3.0, "f").unwrap();
    // Pappus: the spandrel's centroid sits 3·(10 − 3π)/(12 − 3π) from the corner, in ρ and z.
    let spandrel = 9.0 * (1.0 - PI / 4.0);
    let offset = 3.0 * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
    let volume = 36000.0 + PI * 1000.0 + 2.0 * PI * (10.0 + offset) * spandrel;
    let area = (3600.0 - PI * 169.0)
        + 3600.0
        + 2400.0
        + 2.0 * PI * 70.0
        + PI * 100.0
        + 2.0 * PI * (13.0 - 6.0 / PI) * (1.5 * PI);
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "torus"), 1);
    assert_eq!(names(&b, "f.round_0"), 1);
}

#[test]
fn three_fillets_at_a_corner_close_with_a_sphere() {
    let pairs = [
        ("box.end", "box.u_max"),
        ("box.end", "box.v_max"),
        ("box.u_max", "box.v_max"),
    ];
    let b = fillet(&block("box", 20.0, 20.0, 20.0), &pairs, 4.0, "f").unwrap();
    let spandrel = 16.0 * (1.0 - PI / 4.0);
    let volume = 8000.0 - 3.0 * 16.0 * spandrel - (64.0 - PI * 64.0 / 6.0);
    let area = 2400.0 - 3.0 * 4.0 * 16.0 * 2.0 - 3.0 * 16.0 - 3.0 * spandrel
        + 3.0 * PI / 2.0 * 4.0 * 16.0
        + PI * 16.0 / 2.0;
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "sphere"), 1);
    assert_eq!(names(&b, "f.corner"), 1);
    for i in 0..3 {
        assert_eq!(names(&b, &format!("f.round_{i}")), 1);
    }
}

#[test]
fn a_blend_wider_than_its_face_is_refused() {
    let r = chamfer(
        &block("box", 40.0, 30.0, 4.0),
        &[("box.end", "box.v_min")],
        5.0,
        "ch",
    );
    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
}

/// Fillets of `r` on the top edge of `body`'s side y = `side`, from x = `from` to 60, whose end at
/// `from` dies into a curved face: a point w in from the side at x lies past that face where
/// w > `w0(x)`, which reaches r at `x1`. The round takes the part of its spandrel short of w₀,
/// S(u) = r·u − ∫_{r−u}^{r} √(r² − v²) dv, and the whole S(r) = r²(1 − π/4) from x₁ to 60.
fn assert_dies_into(body: &Brep, side: f64, r: f64, from: f64, x1: f64, w0: impl Fn(f64) -> f64) {
    use springrep::{blend::fillet_edges, measure::edge_at};
    let (edge, _) = edge_at(body, Vec3::new(30.0, side, 4.0)).unwrap();
    let b = fillet_edges(body, &[edge], r, "f").unwrap();
    let g = |v: f64| v / 2.0 * (r * r - v * v).max(0.0).sqrt() + r * r / 2.0 * (v / r).asin();
    let s = |u: f64| r * u - (g(r) - g(r - u));
    let n = 4000;
    let h = (x1 - from) / n as f64;
    let simpson: f64 = (0..=n)
        .map(|i| {
            let weight = if i == 0 || i == n {
                1.0
            } else {
                [2.0, 4.0][i % 2]
            };
            weight * s(w0(from + h * i as f64).clamp(0.0, r))
        })
        .sum::<f64>()
        * h
        / 3.0;
    let removed = simpson + (60.0 - x1) * s(r);
    let issues = check::check(&b);
    assert!(issues.is_empty(), "{issues:?}");
    let taken = measure::volume(body) - measure::volume(&b);
    assert!(
        (taken - removed).abs() <= 1e-6 * removed,
        "took {taken} vs {removed}"
    );
    assert_eq!(b.face_count(), body.face_count() + 1);
    assert_eq!(names(&b, "f.round_0"), 1);
}

#[test]
fn a_fillet_ending_on_a_hub_it_meets_at_a_glancing_angle_dies_into_it() {
    // A hub of radius 25 and a bar out to x = 60, both 4 thick, the bar's side on the line y = −d
    // with d = √624: the side leaves the hub at x = 1, at 1/25 of a right angle from its tangent.
    // A fillet of 2 on the side's top edge ends on the hub, and leaves the hub's own edge sharp.
    let d = 624f64.sqrt();
    let bar = extrude(
        &Profile::rect(Vec2::new(30.0, 0.0), 60.0, 2.0 * d),
        &at_z(0.0),
        4.0,
        "bar",
    )
    .unwrap();
    let body = union(&disc("hub", 0.0, 50.0, 4.0), &bar).unwrap();
    let x1 = (625.0 - (d - 2.0) * (d - 2.0)).sqrt();
    assert_dies_into(&body, -d, 2.0, 1.0, x1, |x| d - (625.0 - x * x).sqrt());
}

#[test]
fn a_fillet_end_whose_rim_bows_out_behind_its_edge_runs_no_further() {
    // A plate 4 thick whose side y = 0 runs from x = 0 to 60. At x = 0 its rim turns onto an arc
    // about (−1, 50) through the origin, which bows out below y = 0 and comes back to it short of
    // x = −2, ending 5e-8 above it, within the tolerance of the line, in a reflex corner. The
    // fillet of 2 on the side's top edge meets the arc at a glancing angle and dies into it; the
    // rim past the arc never reaches into the fillet's strip.
    let (c, radius, lift) = (Vec2::new(-1.0, 50.0), 2501f64.sqrt(), 5e-8);
    let end = Vec2::new(c.x - (radius * radius - (c.y - lift).powi(2)).sqrt(), lift);
    let at = |p: Vec2| (p.y - c.y).atan2(p.x - c.x);
    let line = |a: Vec2, b: Vec2, name: &str| seg(Segment::Line { a, b }, name);
    let corners = [
        Vec2::new(0.0, 0.0),
        Vec2::new(60.0, 0.0),
        Vec2::new(60.0, 20.0),
        Vec2::new(-10.0, 20.0),
        Vec2::new(-10.0, -0.5),
    ];
    let mut segments = vec![seg(
        Segment::Arc {
            center: c,
            radius,
            start: at(end),
            sweep: at(corners[0]) - at(end),
        },
        "rim",
    )];
    for (i, name) in ["side", "end", "back", "heel"].iter().enumerate() {
        segments.push(line(corners[i], corners[i + 1], name));
    }
    segments.push(line(corners[4], end, "toe"));
    let profile = Profile {
        loops: vec![ProfileLoop { segments }],
    };
    let body = extrude(&profile, &at_z(0.0), 4.0, "p").unwrap();
    let x1 = (2501.0f64 - 48.0 * 48.0).sqrt() - 1.0;
    assert_dies_into(&body, 0.0, 2.0, 0.0, x1, |x| {
        50.0 - (2501.0 - (x + 1.0) * (x + 1.0)).sqrt()
    });
}

#[test]
fn a_boss_fillet_wider_than_its_ledge_ends_on_the_ledge_side() {
    // A disc of radius 10 and height 4 with a boss of radius 8 and height 6 on it: the ledge is 2
    // wide and a fillet of 3 at the boss's foot runs over its rim, ending on the disc's side.
    let body = union(&disc("base", 0.0, 20.0, 4.0), &disc("boss", 4.0, 16.0, 6.0)).unwrap();
    let b = fillet(&body, &[("base.end", "boss.side")], 3.0, "f").unwrap();
    // In the section, s out from the boss and h up from the ledge, the arc's centre is (3, 3):
    // h = 3 − √(9 − w²), w = 3 − s, kept for s ≤ 2 (w ≥ 1). With J = ∫₁³ √(9 − w²) dw
    // = 9π/4 − √2 − 4.5·asin(1/3), the spandrel kept is 6 − J and its moment about the boss's
    // side ∫₁³ (3 − w)(3 − √(9 − w²)) dw = 6 − 3J + 16√2/3. The arc meets the disc's side at
    // h = 3 − √8 after turning acos(1/3), its moment about the boss 9·acos(1/3) − 6√2.
    let j = 9.0 * PI / 4.0 - 2f64.sqrt() - 4.5 * (1.0f64 / 3.0).asin();
    let (kept, moment) = (6.0 - j, 6.0 - 3.0 * j + 16.0 * 2f64.sqrt() / 3.0);
    let turn = (1.0f64 / 3.0).acos();
    let volume = 784.0 * PI + 2.0 * PI * (8.0 * kept + moment);
    let area = (376.0 - 36.0 - 48.0) * PI
        + 20.0 * PI * (3.0 - 8f64.sqrt())
        + 2.0 * PI * (33.0 * turn - 6.0 * 2f64.sqrt());
    assert_solid(&b, volume, area);
    assert_eq!(b.face_count(), 5);
    assert_eq!(kinds(&b, "torus"), 1);
    assert_eq!(names(&b, "f.round_0"), 1);
    assert_eq!(names(&b, "base.end"), 0);
}

#[test]
fn a_concave_fillet_wider_than_its_ledge_ends_on_the_ledge_face() {
    // A ledge 10 deep (u), 4 high (v) and 30 long (z), and on it a wall 4 thick and 6 high standing
    // 2 back from the ledge's front (u = 0), running 20 of its length. A fillet of 3 where the
    // ledge's top meets the wall needs 3 of ledge and finds 2: it runs over the ledge's front edge
    // and ends on the front face, which is carried up to meet it.
    let ledge = extrude(
        &Profile::rect(Vec2::new(5.0, 2.0), 10.0, 4.0),
        &at_z(0.0),
        30.0,
        "ledge",
    )
    .unwrap();
    let wall = extrude(
        &Profile::rect(Vec2::new(4.0, 7.0), 4.0, 6.0),
        &at_z(0.0),
        20.0,
        "wall",
    )
    .unwrap();
    let body = union(&ledge, &wall).unwrap();
    let b = fillet(&body, &[("ledge.v_max", "wall.u_min")], 3.0, "f").unwrap();
    // The arc's centre is at (−1, 7). The full spandrel is 9 − 9π/4; past u = 0 it would hold
    // ∫₀¹ (3 − √(9 − s²)) ds = 3 − √2 − 4.5·asin(1/3). The arc meets u = 0 at v = 7 − √8, having
    // turned acos(1/3) from the wall. The body's area is 1208.
    let kept = 6.0 - 9.0 * PI / 4.0 + 2f64.sqrt() + 4.5 * (1.0f64 / 3.0).asin();
    let rise = 3.0 - 8f64.sqrt();
    let area = 1208.0 + 2.0 * kept + 20.0 * rise - 40.0 - 60.0 + 60.0 * (1.0f64 / 3.0).acos();
    assert_solid(&b, 1680.0 + 20.0 * kept, area);
    assert_eq!(b.face_count(), 11);
    assert_eq!(names(&b, "f.round_0"), 1);
    assert_eq!(names(&b, "ledge.u_min"), 1);
}

#[test]
fn a_pair_that_names_no_edge_is_invalid() {
    let r = fillet(
        &block("box", 4.0, 4.0, 4.0),
        &[("box.end", "box.start")],
        1.0,
        "f",
    );
    assert!(matches!(r, Err(Error::InvalidInput(_))), "{r:?}");
}

fn seg(segment: Segment, name: &str) -> NamedSegment {
    NamedSegment::new(segment, name)
}

/// A `w × d` rectangle centred on the origin with corners of radius `r`: sides `s0..s3`
/// counter-clockwise from the bottom, each followed by its corner arc `c0..c3`.
fn rounded(w: f64, d: f64, r: f64) -> Profile {
    let (x, y) = (w / 2.0 - r, d / 2.0 - r);
    let (hw, hd) = (w / 2.0, d / 2.0);
    let arc = |cx: f64, cy: f64, start: f64, name: &str| {
        let center = Vec2::new(cx, cy);
        let sweep = PI / 2.0;
        seg(
            Segment::Arc {
                center,
                radius: r,
                start,
                sweep,
            },
            name,
        )
    };
    let line = |a: (f64, f64), b: (f64, f64), name: &str| {
        let (a, b) = (Vec2::new(a.0, a.1), Vec2::new(b.0, b.1));
        seg(Segment::Line { a, b }, name)
    };
    let segments = vec![
        line((-x, -hd), (x, -hd), "s0"),
        arc(x, -y, -PI / 2.0, "c0"),
        line((hw, -y), (hw, y), "s1"),
        arc(x, y, 0.0, "c1"),
        line((x, hd), (-x, hd), "s2"),
        arc(-x, y, PI / 2.0, "c2"),
        line((-hw, y), (-hw, -y), "s3"),
        arc(-x, -y, PI, "c3"),
    ];
    Profile {
        loops: vec![ProfileLoop { segments }],
    }
}

#[test]
fn a_chamfer_follows_a_tangent_chain_round_a_pocket() {
    // A rounded-rectangle pocket through a plate: picking two opposite sides takes the corner arcs
    // and the other sides with them, so the whole rim is bevelled, the arcs as cones.
    let plate = extrude(
        &Profile::rect(Vec2::ZERO, 60.0, 40.0),
        &at_z(0.0),
        10.0,
        "plate",
    );
    let slot = extrude(&rounded(30.0, 12.0, 5.0), &at_z(0.0), 10.0, "slot");
    let body = difference(&plate.unwrap(), &slot.unwrap()).unwrap();
    let d = 1.0;
    let pairs = [("plate.end", "slot.s0"), ("plate.end", "slot.s2")];
    let b = chamfer(&body, &pairs, d, "ch").unwrap();
    // The band a hole's chamfer takes grows as the outline's outward offset: perimeter L + 2πw.
    let l = 2.0 * 20.0 + 2.0 * 2.0 + 2.0 * PI * 5.0;
    let hole = 30.0 * 12.0 - (4.0 - PI) * 25.0;
    let volume = (2400.0 - hole) * 10.0 - (l * d * d / 2.0 + PI * d * d * d / 3.0);
    let area = 2.0 * (2400.0 - hole) + 10.0 * (200.0 + l) - (l * d + PI * d * d) - l * d
        + 2f64.sqrt() * (l * d + PI * d * d);
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "cone"), 4);
    assert_eq!(names(&b, "ch.bevel_0") + names(&b, "ch.bevel_1"), 8);
}

#[test]
fn an_edge_two_pairs_pick_is_blended_once() {
    let pairs = [("box.end", "box.v_min"), ("box.v_min", "box.end")];
    let b = fillet(&block("box", 40.0, 30.0, 20.0), &pairs, 5.0, "f").unwrap();
    let spandrel = 25.0 * (1.0 - PI / 4.0);
    assert!((measure::volume(&b) - (24000.0 - spandrel * 40.0)).abs() <= 1e-9 * 24000.0);
    assert_eq!(names(&b, "f.round_0"), 1);
    assert_eq!(names(&b, "f.round_1"), 1);
    assert_eq!(kinds(&b, "cylinder"), 1);
}

#[test]
fn chamfers_meeting_at_corners_are_mitred() {
    let (w, d, h, c) = (20.0, 12.0, 8.0, 1.5);
    let pairs = [
        ("box.end", "box.u_min"),
        ("box.end", "box.u_max"),
        ("box.end", "box.v_min"),
        ("box.end", "box.v_max"),
    ];
    let b = chamfer(&block("box", w, d, h), &pairs, c, "ch").unwrap();
    // A hipped top: at depth z the section is the rectangle inset by c − z on every side.
    let volume = w * d * h - (c * c * (w + d) - 4.0 * c * c * c / 3.0);
    let area = (w - 2.0 * c) * (d - 2.0 * c)
        + w * d
        + 2.0 * (w + d) * (h - c)
        + 2.0 * 2f64.sqrt() * c * (w - c + d - c);
    assert_solid(&b, volume, area);
    assert_eq!(b.face_count(), 10);
}

#[test]
fn a_chamfer_ending_under_an_overhang_stops_at_its_end_face() {
    // A bar 14 × 20 × 40 with a web 14 × 19 × 10 on top, flush with it at x = 0, x = 14 and
    // y = 0, which leaves a strip of the bar's top 1 wide along y = 20. A chamfer of 4 on the bar's
    // edge at x = 14, y = 20 runs past that strip's 1 mm edge, under the web, and into the concave
    // corner where the strip meets the web's side; it ends square on the strip's plane, z = 40.
    let rect = |id: &str, w: f64, d: f64, z: f64, h: f64| {
        extrude(
            &Profile::rect(Vec2::new(w / 2.0, d / 2.0), w, d),
            &at_z(z),
            h,
            id,
        )
        .unwrap()
    };
    let body = union(
        &rect("bar", 14.0, 20.0, 0.0, 40.0),
        &rect("web", 14.0, 19.0, 40.0, 10.0),
    )
    .unwrap();
    let b = chamfer(&body, &[("bar.u_max", "bar.v_max")], 4.0, "ch").unwrap();
    // The bevel takes the triangle x + y ≥ 30 off the bar's 40 height: 8 × 40 of volume, 4 of each
    // wall and 4√2 of bevel across, and 8 off the bottom. At z = 40 it takes 3.5 off the strip,
    // and leaves the web's underside over (14, 16), (14, 19), (11, 19) bare: 4.5. The body's area
    // is 3940.
    let area = 3940.0 - 320.0 + 160.0 * 2f64.sqrt() - 8.0 - 3.5 + 4.5;
    assert_solid(&b, 13860.0 - 320.0, area);
    assert_eq!(b.face_count(), 10);
    assert_eq!(names(&b, "ch.bevel_0"), 1);
}

#[test]
fn a_chamfer_ending_on_a_slanted_face_is_trimmed_by_it() {
    // A trapezoid prism whose right end leans back; the chamfered edge between its top and its
    // end cap runs from the square left end into that slant.
    let (l, s, hh, t, d) = (30.0, 6.0, 10.0, 8.0, 1.0);
    let pts = [
        Vec2::new(0.0, 0.0),
        Vec2::new(l, 0.0),
        Vec2::new(l - s, hh),
        Vec2::new(0.0, hh),
    ];
    let outline = Profile::polygon(&pts, &["bottom", "slant", "top", "left"]);
    let body = extrude(&outline, &at_z(0.0), t, "p").unwrap();
    let b = chamfer(&body, &[("p.top", "p.end")], d, "ch").unwrap();
    // Each line along x through the bevel's section ends on the slant x = l − s·y/hh.
    let tri = d * d / 2.0;
    let reach = |y: f64| l - s * y / hh;
    let volume = (l + l - s) / 2.0 * hh * t - tri * reach(hh - d / 3.0);
    let slant = (s * s + hh * hh).sqrt();
    let a0 = (l + l - s) * hh + t * (l + (l - s) + hh + slant);
    let cap_strip = d * l - s * (hh * hh - (hh - d) * (hh - d)) / (2.0 * hh);
    let area = a0 - d * (l - s) - cap_strip - tri * slant / hh - tri
        + d * 2f64.sqrt() * reach(hh - d / 2.0);
    assert_solid(&b, volume, area);
    assert_eq!(names(&b, "ch.bevel_0"), 1);
}

/// The area `½∮(x dy − y dx)` contributes along the shorter arc about `o` from `a` to `b`, and
/// that arc's length.
fn arc_green(o: Vec2, r: f64, a: Vec2, b: Vec2) -> (f64, f64) {
    let (sa, sb) = ((a - o).y.atan2((a - o).x), (b - o).y.atan2((b - o).x));
    let sw = (sb - sa + PI).rem_euclid(2.0 * PI) - PI;
    let chord = o.x * (sb.sin() - sa.sin()) - o.y * (sb.cos() - sa.cos());
    (0.5 * (r * r * sw + r * chord), r * sw.abs())
}

#[test]
fn a_straight_edge_between_a_plane_and_a_bore_takes_a_fillet() {
    // A key standing in a bore: the root of each side is filleted against the bore's wall.
    let plate = extrude(
        &Profile::rect(Vec2::ZERO, 60.0, 60.0),
        &at_z(0.0),
        10.0,
        "plate",
    );
    let bore = extrude(&Profile::circle(Vec2::ZERO, 40.0), &at_z(0.0), 10.0, "bore");
    let key = extrude(
        &Profile::rect(Vec2::new(18.0, 0.0), 6.0, 6.0),
        &at_z(0.0),
        10.0,
        "key",
    );
    let body = union(
        &difference(&plate.unwrap(), &bore.unwrap()).unwrap(),
        &key.unwrap(),
    )
    .unwrap();
    let r = 1.0;
    let pairs = [("key.v_max", "bore.side"), ("key.v_min", "bore.side")];
    let b = fillet(&body, &pairs, r, "f").unwrap();
    // The round's centre is 1 off the side y = 3 and 1 inside the bore of radius 20.
    let p = Vec2::new(391f64.sqrt(), 3.0);
    let c = Vec2::new(345f64.sqrt(), 4.0);
    let (t0, t1) = (Vec2::new(c.x, 3.0), c * (20.0 / 19.0));
    let (g_round, round_len) = arc_green(c, r, t0, t1);
    let (g_bore, bore_len) = arc_green(Vec2::ZERO, 20.0, t1, p);
    let spandrel = (0.5 * p.cross(t0) + g_round + g_bore).abs();
    let in_bore = |y: f64| y / 2.0 * (400.0 - y * y).sqrt() + 200.0 * (y / 20.0).asin();
    let key_in_bore = in_bore(3.0) - in_bore(-3.0) - 15.0 * 6.0;
    let volume = 10.0 * (3600.0 - 400.0 * PI + key_in_bore + 2.0 * spandrel);
    let area =
        measure::area(&body) + 4.0 * spandrel + 20.0 * (round_len - (t0 - p).norm() - bore_len);
    assert_solid(&b, volume, area);
    assert_eq!(names(&b, "f.round_0"), 1);
    assert_eq!(names(&b, "f.round_1"), 1);
    assert_eq!(kinds(&b, "cylinder"), 3);
}

#[test]
fn a_fillet_as_round_as_its_corners_closes_them_with_spheres() {
    // Every corner of the outline has the fillet's own radius, so each corner's round is an eighth
    // of a sphere; picking one side takes the whole tangent rim.
    let (w, d, h, r) = (20.0, 12.0, 6.0, 2.0);
    let body = extrude(&rounded(w, d, r), &at_z(0.0), h, "b").unwrap();
    let b = fillet(&body, &[("b.end", "b.s1")], r, "f").unwrap();
    let straight = 2.0 * (w - 2.0 * r) + 2.0 * (d - 2.0 * r);
    let spandrel = r * r * (1.0 - PI / 4.0);
    // Pappus: the spandrel's centroid sits r·(10 − 3π)/(12 − 3π) in from the rim.
    let off = r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
    let top = w * d - (4.0 - PI) * r * r;
    let volume = top * h - spandrel * straight - 2.0 * PI * (r - off) * spandrel;
    let perimeter = straight + 2.0 * PI * r;
    let area = 2.0 * top + perimeter * h - (straight * r + PI * r * r) - perimeter * r
        + PI / 2.0 * r * straight
        + 2.0 * PI * r * r;
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "sphere"), 4);
    assert_eq!(names(&b, "f.round_0"), 8);
}

#[test]
fn a_chamfer_as_wide_as_its_corner_arcs_narrows_them_to_points() {
    // Each corner arc's bevel is a cone whose apex lies on the arc's axis, where the top face keeps
    // only a sharp rectangle.
    let (w, d, h, r) = (40.0, 20.0, 10.0, 5.0);
    let body = extrude(&rounded(w, d, r), &at_z(0.0), h, "rr").unwrap();
    let b = chamfer(&body, &[("rr.end", "rr.s0")], r, "c").unwrap();
    let straight = 2.0 * (w - 2.0 * r) + 2.0 * (d - 2.0 * r);
    let section = w * d - (4.0 - PI) * r * r;
    // Pappus: the bevel's triangle has its centroid r/3 in from the rim.
    let volume = section * h - r * r / 2.0 * (straight + 2.0 * PI * (r - r / 3.0));
    let area = section
        + (w - 2.0 * r) * (d - 2.0 * r)
        + (straight + 2.0 * PI * r) * (h - r)
        + 2f64.sqrt() * r * (straight + PI * r);
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "cone"), 4);
}

#[test]
fn a_chamfer_of_two_distances_sets_the_first_back_on_the_face_named() {
    use springrep::{
        blend::chamfer_two_edges,
        measure::{edge_at, face_at},
        primitive,
    };
    // A block 10 by 8 by 6; its edge along y where the top meets the side x = 10. Set back 2 on
    // the top and 1 on the side, the top keeps 8 of its 10 across and the side 5 of its 6.
    let b = primitive::block(&Frame::WORLD, Vec3::new(10.0, 8.0, 6.0), "blk").unwrap();
    let (edge, _) = edge_at(&b, Vec3::new(10.0, 4.0, 6.0)).unwrap();
    let (top, _) = face_at(&b, Vec3::new(5.0, 4.0, 6.0)).unwrap();
    let (side, _) = face_at(&b, Vec3::new(10.0, 4.0, 3.0)).unwrap();
    for (first, on_top, on_side) in [(top, 2.0, 1.0), (side, 1.0, 2.0)] {
        let cut = chamfer_two_edges(&b, edge, first, 2.0, 1.0, "cha").unwrap();
        assert!(check::check(&cut).is_empty());
        let volume = 480.0 - 0.5 * 2.0 * 1.0 * 8.0;
        assert!((measure::volume(&cut) - volume).abs() < 1e-9);
        // Where each face now ends: a point just inside the kept part is on it, one just past
        // the chamfer's edge is not.
        let on = |p: Vec3| face_at(&cut, p).is_some_and(|(_, d)| d < 1e-9);
        assert!(on(Vec3::new(10.0 - on_top - 0.01, 4.0, 6.0)));
        assert!(!on(Vec3::new(10.0 - on_top + 0.01, 4.0, 6.0)));
        assert!(on(Vec3::new(10.0, 4.0, 6.0 - on_side - 0.01)));
        assert!(!on(Vec3::new(10.0, 4.0, 6.0 - on_side + 0.01)));
    }
    // A face that is not one of the edge's is refused.
    let (bottom, _) = face_at(&b, Vec3::new(5.0, 4.0, 0.0)).unwrap();
    assert!(chamfer_two_edges(&b, edge, bottom, 2.0, 1.0, "cha").is_err());
}

#[test]
fn a_chamfer_sets_back_along_a_ball_by_the_chord() {
    use springrep::{
        blend::chamfer_two_edges,
        measure::{edge_at, face_at},
        primitive,
    };
    // The upper half of the ball of radius 5, its rim set back 2 on the flat and 1 on the ball. The
    // chord 1 from the rim turns 2·asin(1/10) about the centre, to the circle of radius 4.9 at
    // height h = 3√11/10; the cone runs from there to the circle of radius 3 on the flat.
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let corner = Frame::new(Vec3::new(-6.0, -6.0, -6.0), Vec3::Z, Vec3::X).unwrap();
    let below = primitive::block(&corner, Vec3::new(12.0, 12.0, 6.0), "cut").unwrap();
    let half = difference(&ball, &below).unwrap();
    let (edge, _) = edge_at(&half, Vec3::new(5.0, 0.0, 0.0)).unwrap();
    let (flat, _) = face_at(&half, Vec3::new(1.0, 1.0, 0.0)).unwrap();
    let b = chamfer_two_edges(&half, edge, flat, 2.0, 1.0, "cha").unwrap();
    let h = 3.0 * 11f64.sqrt() / 10.0;
    let (r0, r1) = (3.0, 4.9);
    let volume = 250.0 * PI / 3.0 - PI * (25.0 * h - h.powi(3) / 3.0)
        + PI * h * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0;
    let slant = ((r1 - r0).powi(2) + h * h).sqrt();
    let area = PI * r0 * r0 + 2.0 * PI * 5.0 * (5.0 - h) + PI * (r0 + r1) * slant;
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "cone"), 1);
}

#[test]
fn a_chamfer_by_a_distance_and_an_angle_runs_to_where_its_bevel_meets_the_other_face() {
    use springrep::{
        blend::chamfer_distance_angle,
        measure::{edge_at, face_at},
        primitive,
    };
    // The upper half of the ball of radius 5, its rim set back 1 on the flat at 45° to it: the
    // bevel z = ρ − 4 meets the ball where (4 + u)² + u² = 25, u = (√34 − 4)/2. The ring it takes
    // has, at height z below u, the radii from z + 4 out to √(25 − z²).
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let corner = Frame::new(Vec3::new(-6.0, -6.0, -6.0), Vec3::Z, Vec3::X).unwrap();
    let below = primitive::block(&corner, Vec3::new(12.0, 12.0, 6.0), "cut").unwrap();
    let half = difference(&ball, &below).unwrap();
    let (edge, _) = edge_at(&half, Vec3::new(5.0, 0.0, 0.0)).unwrap();
    let (flat, _) = face_at(&half, Vec3::new(1.0, 1.0, 0.0)).unwrap();
    let b = chamfer_distance_angle(&half, edge, flat, 1.0, PI / 4.0, "cha").unwrap();
    let u = (34f64.sqrt() - 4.0) / 2.0;
    let ring = PI * (9.0 * u - 4.0 * u * u - 2.0 * u.powi(3) / 3.0);
    let area = 16.0 * PI + 50.0 * PI - 10.0 * PI * u + PI * (8.0 + u) * 2f64.sqrt() * u;
    assert_solid(&b, 250.0 * PI / 3.0 - ring, area);
    // An angle of a half turn or more leaves no bevel.
    assert!(chamfer_distance_angle(&half, edge, flat, 1.0, PI, "cha").is_err());
}

#[test]
fn a_chamfer_of_one_rim_of_a_bore_leaves_the_other_sharp() {
    use springrep::{
        blend::chamfer_two_edges,
        measure::{edge_at, face_at},
        primitive,
    };
    // A ball of radius 5 bored through by radius 2; its rims are circles of radius 2 at
    // z = ±√21, both between the same two faces. Only the upper one is set back, 2 down the bore
    // and 1 on the ball, to the circle of radius ρ₁ at z₁ that the chord 1 reaches.
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let bore = primitive::cylinder(&at_z(-6.0), 2.0, 12.0, "bore").unwrap();
    let body = difference(&ball, &bore).unwrap();
    let top = 21f64.sqrt();
    let (edge, _) = edge_at(&body, Vec3::new(2.0, 0.0, top)).unwrap();
    let (wall, _) = face_at(&body, Vec3::new(2.0, 0.0, 0.0)).unwrap();
    let b = chamfer_two_edges(&body, edge, wall, 2.0, 1.0, "cha").unwrap();
    // The rim (2, √21) turned by 2·asin(1/10) away from the bore, about the ball's centre.
    let turn = 2.0 * (0.1f64).asin();
    let (rho1, z1) = (
        2.0 * turn.cos() + top * turn.sin(),
        top * turn.cos() - 2.0 * turn.sin(),
    );
    // The bored ball, πH³/6 for its bore's length H = 2√21, less the ring between the bore, the
    // ball and the cone: the ball's zone over the frustum's and the bore's cylinder.
    let zone = |a: f64, b: f64| PI * (25.0 * (b - a) - (b.powi(3) - a.powi(3)) / 3.0);
    let (za, zb) = (top - 2.0, z1);
    let frustum = PI * (zb - za) * (4.0 + 2.0 * rho1 + rho1 * rho1) / 3.0;
    let ring = zone(za, top) - PI * 4.0 * (top - za) - (zone(za, zb) - frustum);
    let volume = PI * (2.0 * top).powi(3) / 6.0 - ring;
    let slant = ((rho1 - 2.0).powi(2) + (zb - za).powi(2)).sqrt();
    let area = 2.0 * PI * 5.0 * (2.0 * top) + 2.0 * PI * 2.0 * (2.0 * top)
        - 2.0 * PI * 5.0 * (top - z1)
        - 2.0 * PI * 2.0 * 2.0
        + PI * (2.0 + rho1) * slant;
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "cone"), 1);
}

#[test]
fn a_blend_that_takes_a_whole_end_closes_it_to_a_point_on_the_axis() {
    use springrep::primitive;
    // A cylinder of radius 5 and height 10. A chamfer of 5 takes the whole top: a cone of height 5
    // stands on the cylinder's lower half, its apex the top's centre. A fillet of 5 rolls a ball
    // centred on the axis: a hemisphere stands there instead.
    let c = primitive::cylinder(&Frame::WORLD, 5.0, 10.0, "cyl").unwrap();
    let base = (125.0 * PI, 25.0 * PI + 50.0 * PI);
    let b = chamfer(&c, &[("cyl.end", "cyl.side")], 5.0, "c").unwrap();
    let cone = (125.0 * PI / 3.0, 25.0 * 2f64.sqrt() * PI);
    assert_solid(&b, base.0 + cone.0, base.1 + cone.1);
    assert_eq!((b.face_count(), kinds(&b, "cone")), (3, 1));
    let b = fillet(&c, &[("cyl.end", "cyl.side")], 5.0, "f").unwrap();
    let dome = (2.0 * PI * 125.0 / 3.0, 50.0 * PI);
    assert_solid(&b, base.0 + dome.0, base.1 + dome.1);
    assert_eq!((b.face_count(), kinds(&b, "sphere")), (3, 1));
}

/// The octahedron |x| + |y| + |z| ≤ `a`, its eight faces planes.
fn octahedron(a: f64) -> Brep {
    let p = [
        Vec3::new(a, 0.0, 0.0),
        Vec3::new(-a, 0.0, 0.0),
        Vec3::new(0.0, a, 0.0),
        Vec3::new(0.0, -a, 0.0),
        Vec3::new(0.0, 0.0, a),
        Vec3::new(0.0, 0.0, -a),
    ];
    let mut tris = Vec::new();
    for (x, y, z) in [(0, 2, 4), (1, 3, 4), (0, 3, 5), (1, 2, 5)] {
        tris.push([x, y, z]);
    }
    for (x, y, z) in [(1, 2, 4), (0, 3, 4), (0, 2, 5), (1, 3, 5)] {
        tris.push([y, x, z]);
    }
    mesh::solid(&p, &tris).unwrap()
}

#[test]
fn edges_chamfered_one_after_another_meet_at_each_corner_in_a_point() {
    use springrep::{blend::chamfer_edges, measure::edge_at, primitive};
    // A cube of side 10, its edges chamfered 2 one at a time: each later chamfer ends on the
    // earlier ones' faces. Twelve prisms of section 2 along the edges, and in each corner's cube
    // of side 2 all but the points whose distances from the faces pair up to at least 2, which
    // fill a quarter of it: 1000 − 12·2·(10 − 4) − 8·3·8/4. Each face keeps its 6 × 6 square and
    // each chamfer is a rectangle 6 by 2√2 with a triangle of base 2√2 and height 1 at each end.
    let cube = primitive::block(&Frame::WORLD, Vec3::new(10.0, 10.0, 10.0), "cube").unwrap();
    let sides = ["cube.v_min", "cube.u_max", "cube.v_max", "cube.u_min"];
    let mut b = cube;
    for (i, side) in sides.iter().enumerate() {
        for pair in [
            (*side, sides[(i + 1) % 4]),
            ("cube.start", *side),
            ("cube.end", *side),
        ] {
            b = chamfer(&b, &[pair], 2.0, "c").unwrap();
        }
    }
    let sqrt2 = 2f64.sqrt();
    assert_solid(&b, 1000.0 - 144.0 - 48.0, 216.0 + 12.0 * 14.0 * sqrt2);
    assert_eq!(b.face_count(), 18);
    // The octahedron |x| + |y| + |z| ≤ 10, its edges chamfered √6 across each face one at a time:
    // four faces meet at each corner. In each octant the three wedges past the chamfers
    // x + y ≤ 8, ..., each ∫ s(10 − s) ds over 8 ≤ s ≤ 10 = 52/3, less the three regions two of
    // them share, each 8/3. Each face keeps the triangle of side 4√2 and each chamfer is a
    // rectangle 4√2 by 4 with a triangle of base 4 and height 2√2 at each end.
    let mut b = octahedron(10.0);
    let mids = [
        (5.0, 5.0, 0.0),
        (5.0, -5.0, 0.0),
        (-5.0, 5.0, 0.0),
        (-5.0, -5.0, 0.0),
        (5.0, 0.0, 5.0),
        (5.0, 0.0, -5.0),
        (-5.0, 0.0, 5.0),
        (-5.0, 0.0, -5.0),
        (0.0, 5.0, 5.0),
        (0.0, 5.0, -5.0),
        (0.0, -5.0, 5.0),
        (0.0, -5.0, -5.0),
    ];
    for (x, y, z) in mids {
        let (e, _) = edge_at(&b, Vec3::new(x, y, z)).unwrap();
        b = chamfer_edges(&b, &[e], 6f64.sqrt(), "c").unwrap();
    }
    let volume = 4000.0 / 3.0 - 8.0 * (3.0 * 52.0 / 3.0 - 8.0);
    let area = 8.0 * 3f64.sqrt() / 4.0 * 32.0 + 12.0 * (4.0 * sqrt2 * 4.0 + 4.0 * 2.0 * sqrt2);
    assert_solid(&b, volume, area);
    assert_eq!(b.face_count(), 20);
}

#[test]
fn chamfers_meeting_at_a_corner_close_it_with_the_plane_of_their_contacts() {
    use springrep::{blend::chamfer_edges, primitive, topo::EdgeId};
    // A cube of side 10, all edges chamfered 2 at once. The contact lines on each face cross 2 in
    // from the corner, and the plane through the three crossings cuts it: in each corner's cube of
    // side 2 the tetrahedron (2, 2, 2), (2, 2, 0), (2, 0, 2), (0, 2, 2) stays, 1/6 of it. Each
    // corner face is the equilateral triangle of side 2√2.
    let cube = primitive::block(&Frame::WORLD, Vec3::new(10.0, 10.0, 10.0), "cube").unwrap();
    let edges: Vec<EdgeId> = cube.edge_ids().collect();
    let b = chamfer_edges(&cube, &edges, 2.0, "c").unwrap();
    let (sqrt2, sqrt3) = (2f64.sqrt(), 3f64.sqrt());
    assert_solid(
        &b,
        1000.0 - 144.0 - 8.0 * (8.0 - 8.0 / 6.0),
        216.0 + 144.0 * sqrt2 + 8.0 * sqrt3 / 4.0 * 8.0,
    );
    assert_eq!((b.face_count(), names(&b, "c.corner")), (26, 8));
    // The octahedron |x| + |y| + |z| ≤ 10, all edges chamfered √6 at once: four chamfers at each
    // corner, whose contact crossings lie on the plane 4 in from it, |x| = 6. Past the mitred
    // corners (as chamfered one at a time) each corner loses the square pyramid of height 2 on
    // the square of side 4.
    let o = octahedron(10.0);
    let edges: Vec<EdgeId> = o.edge_ids().collect();
    let b = chamfer_edges(&o, &edges, 6f64.sqrt(), "c").unwrap();
    let mitred = 4000.0 / 3.0 - 8.0 * (3.0 * 52.0 / 3.0 - 8.0);
    let area = 8.0 * sqrt3 / 4.0 * 32.0 + 12.0 * 4.0 * sqrt2 * 4.0 + 6.0 * 16.0;
    assert_solid(&b, mitred - 6.0 * 16.0 * 2.0 / 3.0, area);
    assert_eq!((b.face_count(), names(&b, "c.corner")), (26, 6));
}

#[test]
fn fillets_at_a_corner_of_more_planes_or_concave_close_with_their_ball() {
    use springrep::{blend::fillet_edges, primitive, topo::EdgeId};
    // The octahedron |x| + |y| + |z| ≤ 10, every edge rounded 2: four planes meet at each corner,
    // and the planes 2 in from them meet in one point, so one ball touches all four. The result is
    // the octahedron shrunk by 2, |x| + |y| + |z| ≤ s = 10 − 2√3, grown by 2 (Steiner): its edges
    // s√2 long each turn through arccos(1/3).
    let o = octahedron(10.0);
    let edges: Vec<EdgeId> = o.edge_ids().collect();
    let b = fillet_edges(&o, &edges, 2.0, "f").unwrap();
    let s = 10.0 - 2.0 * 3f64.sqrt();
    let (v_in, s_in) = (4.0 * s.powi(3) / 3.0, 4.0 * 3f64.sqrt() * s * s);
    let m_in = 12.0 * 2f64.sqrt() * s * (1.0f64 / 3.0).acos() / 2.0;
    assert_solid(
        &b,
        v_in + 2.0 * s_in + 4.0 * m_in + 32.0 * PI / 3.0,
        s_in + 4.0 * m_in + 16.0 * PI,
    );
    assert_eq!(kinds(&b, "sphere"), 6);
    // A cube of side 20 with the cube of side 10 cut from one corner; the notch's three concave
    // edges rounded 3. Along each edge a strip of section (1 − π/4)·9, and at the inside corner
    // the cube of side 3 less an eighth of the ball of radius 3.
    let cube = primitive::block(&Frame::WORLD, Vec3::new(20.0, 20.0, 20.0), "cube").unwrap();
    let at = Frame::new(Vec3::new(10.0, 10.0, 10.0), Vec3::Z, Vec3::X).unwrap();
    let notch = primitive::block(&at, Vec3::new(10.0, 10.0, 10.0), "notch").unwrap();
    let body = difference(&cube, &notch).unwrap();
    let pairs = [
        ("notch.u_min", "notch.v_min"),
        ("notch.v_min", "notch.start"),
        ("notch.start", "notch.u_min"),
    ];
    let b = fillet(&body, &pairs, 3.0, "f").unwrap();
    let r: f64 = 3.0;
    let spandrel = (1.0 - PI / 4.0) * r * r;
    let volume = 7000.0 + r.powi(3) * (1.0 - PI / 6.0) + 3.0 * spandrel * (10.0 - r);
    let area = 2400.0 - 300.0
        + 3.0 * (10.0 - r).powi(2)
        + 1.5 * PI * r * (10.0 - r)
        + PI * r * r / 2.0
        + 3.0 * spandrel;
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "sphere"), 1);
}

#[test]
fn fillets_at_an_apex_no_ball_fits_split_it_along_a_ridge() {
    use springrep::{blend::fillet_edges, boolean::unify, topo::EdgeId};
    // The pyramid on the rectangle (±10, ±6, 0) with its apex at (0, 0, 12), every edge rounded
    // 3/2. No ball touches all four planes at the apex: the long sides' planes 3/2 in meet lower
    // than the short ends'. The result is the pyramid shrunk by 3/2, a hip roof on the rectangle
    // 2a × 2b at z = 3/2 with its ridge along x from −x' to x' at H above that, grown by 3/2
    // (Steiner): V' + S'r + M'r² + 4πr³/3, M' each edge's length times half the angle between its
    // faces' normals.
    let p = [
        Vec3::new(-10.0, -6.0, 0.0),
        Vec3::new(10.0, -6.0, 0.0),
        Vec3::new(10.0, 6.0, 0.0),
        Vec3::new(-10.0, 6.0, 0.0),
        Vec3::new(0.0, 0.0, 12.0),
    ];
    let tris = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ];
    let pyramid = unify(&mesh::solid(&p, &tris).unwrap()).unwrap();
    assert_eq!(pyramid.face_count(), 5);
    let edges: Vec<EdgeId> = pyramid.edge_ids().collect();
    let r = 1.5;
    let b = fillet_edges(&pyramid, &edges, r, "f").unwrap();
    let (s5, s61) = (5f64.sqrt(), 61f64.sqrt());
    let (a, bb, x) = (
        (35.0 - s61) / 4.0,
        (21.0 - 3.0 * s5) / 4.0,
        (5.0 * s5 - s61) / 4.0,
    );
    let h = (21.0 - 3.0 * s5) / 2.0;
    let v_in = 2.0 / 3.0 * h * bb * (2.0 * a + x);
    let s_in = 4.0 * a * bb
        + 2.0 * bb * ((a - x).powi(2) + h * h).sqrt()
        + 2.0 * (a + x) * (bb * bb + h * h).sqrt();
    let hip = ((a - x).powi(2) + bb * bb + h * h).sqrt();
    let m_in = 0.5
        * (2.0 * 2.0 * a * (-1.0 / s5).acos()
            + 2.0 * 2.0 * bb * (-5.0 / s61).acos()
            + 4.0 * hip * (5.0 / 305f64.sqrt()).acos()
            + 2.0 * x * (-0.6f64).acos());
    let volume = v_in + s_in * r + m_in * r * r + 4.0 * PI * r.powi(3) / 3.0;
    let area = s_in + 2.0 * m_in * r + 4.0 * PI * r * r;
    assert_solid(&b, volume, area);
    assert_eq!(kinds(&b, "sphere"), 6);
}

#[test]
fn a_straight_edge_between_a_plane_and_a_bore_takes_a_chamfer_by_the_chord() {
    // The key standing in a bore, as above, the root of each side chamfered 1 on both faces: 1
    // along the side, and a chord 1 long on the bore's wall, which turns 2·asin(1/40) about its
    // axis.
    let plate = extrude(
        &Profile::rect(Vec2::ZERO, 60.0, 60.0),
        &at_z(0.0),
        10.0,
        "plate",
    );
    let bore = extrude(&Profile::circle(Vec2::ZERO, 40.0), &at_z(0.0), 10.0, "bore");
    let key = extrude(
        &Profile::rect(Vec2::new(18.0, 0.0), 6.0, 6.0),
        &at_z(0.0),
        10.0,
        "key",
    );
    let body = union(
        &difference(&plate.unwrap(), &bore.unwrap()).unwrap(),
        &key.unwrap(),
    )
    .unwrap();
    let pairs = [("key.v_max", "bore.side"), ("key.v_min", "bore.side")];
    let b = chamfer(&body, &pairs, 1.0, "c").unwrap();
    let p = Vec2::new(391f64.sqrt(), 3.0);
    let t0 = Vec2::new(p.x - 1.0, 3.0);
    let turn = 2.0 * (1.0f64 / 40.0).asin();
    let t1 = Vec2::new(
        p.x * turn.cos() - p.y * turn.sin(),
        p.x * turn.sin() + p.y * turn.cos(),
    );
    let (g_bore, bore_len) = arc_green(Vec2::ZERO, 20.0, t1, p);
    let spandrel = (0.5 * (p.cross(t0) + t0.cross(t1)) + g_bore).abs();
    let in_bore = |y: f64| y / 2.0 * (400.0 - y * y).sqrt() + 200.0 * (y / 20.0).asin();
    let key_in_bore = in_bore(3.0) - in_bore(-3.0) - 15.0 * 6.0;
    let volume = 10.0 * (3600.0 - 400.0 * PI + key_in_bore + 2.0 * spandrel);
    let area = measure::area(&body) + 4.0 * spandrel + 20.0 * ((t1 - t0).norm() - 1.0 - bore_len);
    assert_solid(&b, volume, area);
    assert_eq!(names(&b, "c.bevel_0"), 1);
    assert_eq!(names(&b, "c.bevel_1"), 1);
}
