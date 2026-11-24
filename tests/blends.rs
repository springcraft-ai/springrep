//! Chamfers and fillets on small swept solids against closed-form volume and area, each result
//! checked valid and watertight, with the blend faces named by the operation.

use std::f64::consts::{FRAC_PI_2, PI};

use springrep::{
    Error,
    blend::{chamfer, fillet},
    boolean::{difference, union},
    check,
    geom::Surface,
    math::{Frame, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    primitive::sphere,
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::extrude,
    tol,
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
fn a_plate_rounded_at_half_its_thickness_is_a_full_round() {
    // Every edge of a plate filleted at half its thickness: the side faces shrink to nothing, and
    // each corner's two balls are one, so its upright edge keeps no round. The solid is every
    // point within r of the rectangle r in from the sides at mid-height.
    let (w, d, h) = (40.0, 30.0, 6.0);
    let r = h / 2.0;
    let plate = springrep::primitive::block(&Frame::WORLD, Vec3::new(w, d, h), "plate").unwrap();
    let sides = ["plate.v_min", "plate.u_max", "plate.v_max", "plate.u_min"];
    let mut pairs = Vec::new();
    for (i, side) in sides.iter().enumerate() {
        pairs.push(("plate.start", *side));
        pairs.push(("plate.end", *side));
        pairs.push((*side, sides[(i + 1) % 4]));
    }
    let b = fillet(&plate, &pairs, r, "f").unwrap();
    let (core, rim) = ((w - 2.0 * r) * (d - 2.0 * r), 2.0 * (w + d - 4.0 * r));
    let volume = 2.0 * r * core + PI * r * r / 2.0 * rim + 4.0 / 3.0 * PI * r.powi(3);
    let area = 2.0 * core + PI * r * rim + 4.0 * PI * r * r;
    assert_solid(&b, volume, area);
    assert_eq!(b.face_count(), 10);
    assert_eq!(kinds(&b, "cylinder"), 4);
    assert_eq!(kinds(&b, "sphere"), 4);
    for side in sides {
        assert_eq!(names(&b, side), 0, "{side}");
    }
    for i in 0..12 {
        let upright = i % 3 == 2;
        assert_eq!(names(&b, &format!("f.round_{i}")), usize::from(!upright));
    }
    // A ball larger than half the thickness does not fit.
    assert!(fillet(&plate, &pairs, r + 1e-3, "f").is_err());
}

#[test]
fn a_rib_rounded_at_half_its_width_is_a_full_round() {
    // The top and upright edges of a rib filleted at half its width: the top face and the end
    // faces shrink to nothing, and the top edge across each end keeps no round. In plan the rib
    // is a stadium; above its rounds' axis it is the half cylinder and the two quarter balls.
    let (l, h, r) = (40.0, 10.0, 3.0);
    let rib = springrep::primitive::block(&Frame::WORLD, Vec3::new(l, 2.0 * r, h), "rib").unwrap();
    let sides = ["rib.v_min", "rib.u_max", "rib.v_max", "rib.u_min"];
    let mut pairs = Vec::new();
    for (i, side) in sides.iter().enumerate() {
        pairs.push(("rib.end", *side));
        pairs.push((*side, sides[(i + 1) % 4]));
    }
    let b = fillet(&rib, &pairs, r, "f").unwrap();
    let c = l - 2.0 * r;
    let (plan, rim) = (2.0 * r * c + PI * r * r, 2.0 * c + 2.0 * PI * r);
    let volume = plan * (h - r) + PI * r * r / 2.0 * c + 2.0 / 3.0 * PI * r.powi(3);
    let area = plan + rim * (h - r) + PI * r * c + 2.0 * PI * r * r;
    assert_solid(&b, volume, area);
    for gone in ["rib.end", "rib.u_min", "rib.u_max"] {
        assert_eq!(names(&b, gone), 0, "{gone}");
    }
    for i in 0..8 {
        let across = i == 2 || i == 6;
        assert_eq!(names(&b, &format!("f.round_{i}")), usize::from(!across));
    }
}

/// The pairs of faces every edge of a prism lies between: its `sides` in order round it, and its
/// two `caps`.
fn prism_edges<'a>(sides: &[&'a str; 4], caps: [&'a str; 2]) -> Vec<(&'a str, &'a str)> {
    let mut pairs = Vec::new();
    for (i, side) in sides.iter().enumerate() {
        pairs.push((caps[0], *side));
        pairs.push((caps[1], *side));
        pairs.push((*side, sides[(i + 1) % 4]));
    }
    pairs
}

/// The volume and area of every point within `r` of an a × b × c box: the box, a slab `r` thick on
/// each face, a quarter cylinder along each edge and an eighth of a ball at each corner.
fn grown_box(a: f64, b: f64, c: f64, r: f64) -> (f64, f64) {
    let (faces, edges) = (2.0 * (a * b + b * c + c * a), a + b + c);
    (
        a * b * c + r * faces + PI * r * r * edges + 4.0 / 3.0 * PI * r.powi(3),
        faces + 2.0 * PI * r * edges + 4.0 * PI * r * r,
    )
}

/// The volume and area of every point within `r` of a flat convex polygon of area `core` and
/// perimeter `rim`: a slab 2r thick, half a cylinder along each side, and a lune of the ball at
/// each corner, whose angles add up to the whole ball.
fn grown_flat(core: f64, rim: f64, r: f64) -> (f64, f64) {
    (
        2.0 * r * core + PI * r * r / 2.0 * rim + 4.0 / 3.0 * PI * r.powi(3),
        2.0 * core + PI * r * rim + 4.0 * PI * r * r,
    )
}

fn assert_full_round(b: &Brep, volume: f64, area: f64) {
    assert_solid(b, volume, area);
    assert_eq!(b.face_count(), 10);
    assert_eq!(kinds(b, "plane"), 2);
    assert_eq!(kinds(b, "cylinder"), 4);
    assert_eq!(kinds(b, "sphere"), 4);
}

#[test]
fn a_plate_within_the_tolerance_of_half_its_thickness_is_the_full_round() {
    // A radius within the tolerance of half the thickness leaves a side strip, or an overlap, no
    // wider than the tolerance: the strip is no face, and the solid is the full round at half the
    // thickness itself, not one within the tolerance of it.
    let (w, d, h) = (40.0, 30.0, 6.0);
    let plate = springrep::primitive::block(&Frame::WORLD, Vec3::new(w, d, h), "plate").unwrap();
    let sides = ["plate.v_min", "plate.u_max", "plate.v_max", "plate.u_min"];
    let pairs = prism_edges(&sides, ["plate.start", "plate.end"]);
    let (volume, area) = grown_box(w - h, d - h, 0.0, h / 2.0);
    for dr in [-0.5 * tol::LINEAR, -1e-9, 1e-9, 0.5 * tol::LINEAR] {
        let b = fillet(&plate, &pairs, h / 2.0 + dr, "f").unwrap_or_else(|e| panic!("{dr}: {e}"));
        assert_full_round(&b, volume, area);
        let v = measure::volume(&b);
        assert!(
            (v - volume).abs() <= 1e-12 * volume,
            "{dr}: {v} vs {volume}"
        );
    }
}

#[test]
fn a_plate_just_outside_the_tolerance_of_a_full_round_keeps_its_strip_or_is_refused() {
    // Under half the thickness by more than the tolerance, each side keeps a strip h − 2r high
    // and each upright edge a round as long: every point within r of the box r in from each face.
    let (w, d, h) = (40.0, 30.0, 6.0);
    let plate = springrep::primitive::block(&Frame::WORLD, Vec3::new(w, d, h), "plate").unwrap();
    let sides = ["plate.v_min", "plate.u_max", "plate.v_max", "plate.u_min"];
    let pairs = prism_edges(&sides, ["plate.start", "plate.end"]);
    for under in [2e-7, 1e-6, 1e-5] {
        let r = h / 2.0 - under;
        let b = fillet(&plate, &pairs, r, "f").unwrap_or_else(|e| panic!("{under}: {e}"));
        let (volume, area) = grown_box(w - 2.0 * r, d - 2.0 * r, h - 2.0 * r, r);
        assert_solid(&b, volume, area);
        assert_eq!(b.face_count(), 26, "{under}");
        assert_eq!(kinds(&b, "cylinder"), 12, "{under}");
        assert_eq!(kinds(&b, "sphere"), 8, "{under}");
    }
    // Over it by more, the ball does not fit between the top and the bottom.
    for over in [2e-7, 1e-6] {
        let out = fillet(&plate, &pairs, h / 2.0 + over, "f");
        assert!(
            matches!(out, Err(Error::InvalidInput(_) | Error::Unsupported(_))),
            "{over}: {:?}",
            out.map(|b| b.face_count())
        );
    }
}

#[test]
fn a_plate_of_slanted_sides_rounded_at_half_its_thickness_is_a_full_round() {
    // A plate 6 thick whose long sides lean in at 60° and 45°, every edge filleted at 3. The
    // edges up its ends are not square to the top and bottom: the ball at the foot of the 60°
    // side stands 3·cot 30° back along the edge and the ball at its head 3·tan 30°, which together
    // are the edge's whole length 6/sin 60°, and likewise at the 45° side. So all four sides
    // shrink to nothing, and the solid is every point within 3 of the rectangle at mid-thickness
    // that lies 3 inside each side: 3·cot(α/2) in from where a side leaning at α meets the bottom.
    let (base, l, h) = (30.0, 40.0, 6.0);
    let r = h / 2.0;
    let (alpha, beta) = (PI / 3.0, PI / 4.0);
    let pts = [
        Vec2::new(0.0, 0.0),
        Vec2::new(base, 0.0),
        Vec2::new(base - h / beta.tan(), h),
        Vec2::new(h / alpha.tan(), h),
    ];
    let sides = ["p.bottom", "p.right", "p.top", "p.left"];
    let plate = extrude(
        &Profile::polygon(&pts, &["bottom", "right", "top", "left"]),
        &at_z(0.0),
        l,
        "p",
    )
    .unwrap();
    let b = fillet(&plate, &prism_edges(&sides, ["p.start", "p.end"]), r, "f").unwrap();
    let across = base - r / (alpha / 2.0).tan() - r / (beta / 2.0).tan();
    let (volume, area) = grown_flat(across * (l - h), 2.0 * (across + l - h), r);
    assert_full_round(&b, volume, area);
    for gone in ["p.left", "p.right", "p.start", "p.end"] {
        assert_eq!(names(&b, gone), 0, "{gone}");
    }
}

#[test]
fn a_plate_on_a_parallelogram_rounded_at_half_its_thickness_is_a_full_round() {
    // A plate 6 thick on a parallelogram of sides 40 and 30 at 60°, every edge filleted at 3. The
    // balls at a corner of plan angle θ stand 3·cot(θ/2) back along the top and bottom edges, a
    // different length at each end of every one of them, and 3 back along the upright edge, which
    // they take whole. The solid is every point within 3 of the parallelogram 3 inside each side
    // at mid-thickness, whose sides are 2·3/sin 60° shorter.
    let (p, q, h, theta) = (40.0, 30.0, 6.0, PI / 3.0);
    let r = h / 2.0;
    let lean = Vec2::new(q * theta.cos(), q * theta.sin());
    let pts = [
        Vec2::new(0.0, 0.0),
        Vec2::new(p, 0.0),
        Vec2::new(p, 0.0) + lean,
        lean,
    ];
    let sides = ["p.a", "p.b", "p.c", "p.d"];
    let plate = extrude(
        &Profile::polygon(&pts, &["a", "b", "c", "d"]),
        &at_z(0.0),
        h,
        "p",
    )
    .unwrap();
    let b = fillet(&plate, &prism_edges(&sides, ["p.start", "p.end"]), r, "f").unwrap();
    let cut = 2.0 * r / theta.sin();
    let (volume, area) = grown_flat(
        (p - cut) * (q - cut) * theta.sin(),
        2.0 * (p + q - 2.0 * cut),
        r,
    );
    assert_full_round(&b, volume, area);
    for gone in sides {
        assert_eq!(names(&b, gone), 0, "{gone}");
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

/// A ball of radius `BALL` about the origin with a post standing on it up to `TOP`, `POST` square
/// unless a test says otherwise.
const BALL: f64 = 41.23;
const POST: Vec2 = Vec2::new(28.5, 28.5);
const TOP: f64 = 60.0;

/// The post's sides in the order of [`Profile::rect`]'s edges, each with its outward normal.
const WALLS: [(&str, Vec3); 4] = [
    ("post.v_min", Vec3::new(0.0, -1.0, 0.0)),
    ("post.u_max", Vec3::new(1.0, 0.0, 0.0)),
    ("post.v_max", Vec3::new(0.0, 1.0, 0.0)),
    ("post.u_min", Vec3::new(-1.0, 0.0, 0.0)),
];

fn post_on_ball(centre: Vec2, size: Vec2) -> Brep {
    let ball = sphere(&at_z(0.0), BALL, "ball").unwrap();
    let foot = Profile::rect(centre, size.x, size.y);
    let post = extrude(&foot, &at_z(0.0), TOP, "post").unwrap();
    union(&ball, &post).unwrap()
}

/// Wall `w`'s distance from the ball's centre, and where the post's middle and half its width
/// lie along the wall.
fn wall_place(centre: Vec2, size: Vec2, w: usize) -> (f64, f64, f64) {
    let n = WALLS[w].1;
    let along = Vec3::Z.cross(n);
    let c = Vec3::new(centre.x, centre.y, 0.0);
    let half = |v: Vec3| 0.5 * (v.x.abs() * size.x + v.y.abs() * size.y);
    (c.dot(n) + half(n), c.dot(along), half(along))
}

/// Gauss-Legendre quadrature of `f` over `[a, b]`, five points on each of 64 panels: exact for
/// degree 9 on each panel, so far below 1e-12 for the smooth integrands here.
fn gauss(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    let inner = (5.0 - 2.0 * (10.0f64 / 7.0).sqrt()).sqrt() / 3.0;
    let outer = (5.0 + 2.0 * (10.0f64 / 7.0).sqrt()).sqrt() / 3.0;
    let (wi, wo) = (
        (322.0 + 13.0 * 70f64.sqrt()) / 900.0,
        (322.0 - 13.0 * 70f64.sqrt()) / 900.0,
    );
    let rule = [
        (0.0, 128.0 / 225.0),
        (inner, wi),
        (-inner, wi),
        (outer, wo),
        (-outer, wo),
    ];
    let panels = 64;
    let h = (b - a) / panels as f64;
    (0..panels)
        .map(|k| {
            let mid = a + (k as f64 + 0.5) * h;
            let sum: f64 = rule.iter().map(|(x, w)| w * f(mid + 0.5 * h * x)).sum();
            sum * 0.5 * h
        })
        .sum()
}

/// Where a foot blend stops along its wall, as the coordinate `y` along the wall of its end at
/// the distance `x` from the ball's centre square to the wall: on the plane of the wall beside
/// it, with a flat end there; on that plane under the blend of that wall, which it crosses past
/// the corner, with no end of its own; or on the plane through the ball's centre between the two
/// walls, `y = ±x`.
#[derive(Clone, Copy)]
enum Stop {
    Wall(f64),
    Under(f64),
    Mitre(f64),
}

impl Stop {
    fn at(self, x: f64) -> f64 {
        match self {
            Stop::Wall(y) | Stop::Under(y) => y,
            Stop::Mitre(sign) => sign * x,
        }
    }
}

/// A foot blend's size: a fillet's radius, or a chamfer's setback on the wall and the ball.
#[derive(Clone, Copy)]
enum Size {
    Round(f64),
    Bevel(f64),
}

/// One foot blend, worked by hand in the half plane through the line from the ball's centre
/// square to the wall: `x` along that line and `ρ` the distance from it. The wall is the line
/// `x = a` and the ball the circle `x² + ρ² = R²`, meeting at the edge. A fillet's rolling ball of
/// radius `r` touches both with its centre at `(ρc, a + r)`, `ρc² = (R + r)² − (a + r)²`; a
/// chamfer runs straight from the wall `d` up from the edge to the ball a chord `d` from it. The
/// blend's material is the spandrel between the three, turned about the line: a point of it at
/// `(ρ, x)` lies between the two stops over the angle `asin(hi/ρ) − asin(lo/ρ)`.
struct Foot {
    /// The wall's distance from the ball's centre.
    a: f64,
    size: Size,
    stops: [Stop; 2],
}

impl Foot {
    fn edge(&self) -> f64 {
        (BALL * BALL - self.a * self.a).sqrt()
    }

    /// A fillet's spine: its ball's distance from the line.
    fn spine(&self, r: f64) -> f64 {
        ((BALL + r).powi(2) - (self.a + r).powi(2)).sqrt()
    }

    /// The elevation above the `ρ` axis, seen from the ball's centre, of the blend's contact
    /// with the wall, of the edge, and of its contact with the ball.
    fn elevations(&self) -> [f64; 3] {
        let edge = (self.a / BALL).asin();
        match self.size {
            Size::Round(r) => [
                self.a.atan2(self.spine(r)),
                edge,
                ((self.a + r) / (BALL + r)).asin(),
            ],
            Size::Bevel(d) => [
                self.a.atan2(self.edge() + d),
                edge,
                edge + 2.0 * (d / (2.0 * BALL)).asin(),
            ],
        }
    }

    /// The blend's section from the wall to the ball, as `(ρ, x)` and its speed `(ρ′, x′)`, over
    /// `t` from 0 to [`Foot::last`]: a fillet's arc by the angle its ball has turned from the
    /// wall's contact, a chamfer's line by the fraction of it.
    fn section(&self, t: f64) -> [f64; 4] {
        match self.size {
            Size::Round(r) => [
                self.spine(r) - r * t.sin(),
                self.a + r - r * t.cos(),
                -r * t.cos(),
                r * t.sin(),
            ],
            Size::Bevel(d) => {
                let ball = self.elevations()[2];
                let (w, b) = (self.edge() + d, Vec2::new(ball.cos(), ball.sin()) * BALL);
                let run = Vec2::new(b.x - w, b.y - self.a);
                [w + t * run.x, self.a + t * run.y, run.x, run.y]
            }
        }
    }

    fn last(&self) -> f64 {
        match self.size {
            Size::Round(r) => ((self.a + r) / (BALL + r)).acos(),
            Size::Bevel(_) => 1.0,
        }
    }

    fn turn(&self, rho: f64, x: f64) -> f64 {
        (self.stops[1].at(x) / rho).asin() - (self.stops[0].at(x) / rho).asin()
    }

    /// The integral of `f(ρ, x)` over the spandrel, by lines square to the wall: at the height
    /// `x` of each point of the blend's section, from the ball out to that point. This holds
    /// for a wall on either side of the ball's centre.
    fn over_spandrel(&self, f: impl Fn(f64, f64) -> f64) -> f64 {
        gauss(0.0, self.last(), |t| {
            let [rho, x, _, d_x] = self.section(t);
            gauss((BALL * BALL - x * x).sqrt(), rho, |q| f(q, x)) * d_x
        })
    }

    fn volume(&self) -> f64 {
        self.over_spandrel(|rho, x| rho * self.turn(rho, x))
    }

    /// The area the blend adds: its own face, less the strips of the wall and the ball it
    /// covers, and the flat end it leaves on each wall it stops on.
    fn area(&self) -> f64 {
        let [_, edge, ball] = self.elevations();
        let swept = |rho: f64, x: f64| rho * self.turn(rho, x);
        let face = gauss(0.0, self.last(), |t| {
            let [rho, x, d_rho, d_x] = self.section(t);
            d_rho.hypot(d_x) * swept(rho, x)
        });
        let on_wall = gauss(self.edge(), self.section(0.0)[0], |rho| swept(rho, self.a));
        let on_ball = gauss(edge, ball, |psi| {
            BALL * swept(BALL * psi.cos(), BALL * psi.sin())
        });
        let ends: f64 = self
            .stops
            .iter()
            .map(|stop| match *stop {
                Stop::Wall(y) => self.over_spandrel(|rho, _| rho / (rho * rho - y * y).sqrt()),
                Stop::Under(_) | Stop::Mitre(_) => 0.0,
            })
            .sum();
        face - on_wall - on_ball + ends
    }
}

/// The volume and area of the post on the ball before any blend: the ball, and the post above
/// it, whose foot is the ball's surface over the post's rectangle.
fn post_on_ball_measures(centre: Vec2, size: Vec2) -> (f64, f64) {
    let (lo, hi) = (centre - size * 0.5, centre + size * 0.5);
    let over_foot = |f: &dyn Fn(f64) -> f64| {
        gauss(lo.x, hi.x, |x| {
            gauss(lo.y, hi.y, |y| f(BALL * BALL - x * x - y * y))
        })
    };
    let volume =
        4.0 / 3.0 * PI * BALL.powi(3) + size.x * size.y * TOP - over_foot(&|zz: f64| zz.sqrt());
    // A wall at the distance `a` meets the ball in a circle of radius `c`, c² = R² − a².
    let under = |a: f64, lo: f64, hi: f64| {
        let c = (BALL * BALL - a * a).sqrt();
        let to = |y: f64| 0.5 * (y * (c * c - y * y).sqrt() + c * c * (y / c).asin());
        to(hi) - to(lo)
    };
    let walls = 2.0 * (size.x + size.y) * TOP
        - under(hi.x, lo.y, hi.y)
        - under(lo.x, lo.y, hi.y)
        - under(hi.y, lo.x, hi.x)
        - under(lo.y, lo.x, hi.x);
    let area =
        4.0 * PI * BALL * BALL - over_foot(&|zz: f64| BALL / zz.sqrt()) + size.x * size.y + walls;
    (volume, area)
}

/// The corner two foot blends cross past, worked by hand: their walls stand square to each
/// other, `far` the one farther from the ball's centre, and `x` and `y` run square to them. Over
/// the point `(x, y)` the far wall's blend, about the `x` axis at the distance `ρ(x)` from it,
/// stands `√(ρ(x)² − y²)` high, the near wall's the same with the two exchanged, and the ball
/// `√(R² − x² − y²)`; the lower blend bounds the corner. The far wall's blend is the lower along
/// its own wall up to where the near wall's surface comes down to its contact there, the near
/// end of the crease; from there the two cross once on every line square to either wall, out to
/// where both contacts on the ball meet.
struct Crossing<'a> {
    far: &'a Foot,
    near: &'a Foot,
}

impl Crossing<'_> {
    /// How much higher the far blend stands than the near one over the sections `tx` and `ty`.
    fn over(&self, tx: f64, ty: f64) -> f64 {
        let ([rho_x, x, ..], [rho_y, y, ..]) = (self.far.section(tx), self.near.section(ty));
        (rho_x * rho_x - y * y).sqrt() - (rho_y * rho_y - x * x).sqrt()
    }

    /// The heights of the far blend, the near blend and the ball over the sections `tx`, `ty`.
    fn heights(&self, tx: f64, ty: f64) -> [f64; 3] {
        let ([rho_x, x, ..], [rho_y, y, ..]) = (self.far.section(tx), self.near.section(ty));
        [
            (rho_x * rho_x - y * y).sqrt(),
            (rho_y * rho_y - x * x).sqrt(),
            (BALL * BALL - x * x - y * y).sqrt(),
        ]
    }

    fn bisect(f: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
        assert!(f(lo) < 0.0 && f(hi) > 0.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if f(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// The near blend's section the crease lies on over the far blend's section `tx`.
    fn across_near(&self, tx: f64) -> f64 {
        Self::bisect(|ty| self.over(tx, ty), 0.0, self.near.last())
    }

    /// The far blend's section the crease lies on over the near blend's section `ty`, past the
    /// crease's near end.
    fn across_far(&self, ty: f64) -> f64 {
        Self::bisect(|tx| -self.over(tx, ty), 0.0, self.far.last())
    }

    /// The volume by columns: the lower blend's height above the ball over every `(x, y)`.
    fn volume_by_columns(&self) -> f64 {
        gauss(0.0, self.far.last(), |tx| {
            let column = |ty: f64| {
                let [over_far, over_near, ball] = self.heights(tx, ty);
                (over_far.min(over_near) - ball) * self.near.section(ty)[3]
            };
            let crease = self.across_near(tx);
            let low = gauss(0.0, crease, column) + gauss(crease, self.near.last(), column);
            low * self.far.section(tx)[3]
        })
    }

    /// The patch of the far blend's surface up to the crease, `f(ρ, x, ρ′, x′)` summed over it:
    /// at the section `tx` it turns about the `x` axis from the near wall's plane to the crease.
    fn on_far(&self, f: impl Fn([f64; 4]) -> f64) -> f64 {
        gauss(0.0, self.far.last(), |tx| {
            let s = self.far.section(tx);
            let crease = self.near.section(self.across_near(tx))[1];
            f(s) * s[0] * ((crease / s[0]).asin() - (self.near.a / s[0]).asin())
        })
    }

    /// The same for the near blend's surface, which starts at the crease's near end.
    fn on_near(&self, f: impl Fn([f64; 4]) -> f64) -> f64 {
        gauss(self.across_near(0.0), self.near.last(), |ty| {
            let s = self.near.section(ty);
            let crease = self.far.section(self.across_far(ty))[1];
            f(s) * s[0] * ((crease / s[0]).asin() - (self.far.a / s[0]).asin())
        })
    }

    /// The ball's surface under the corner, `f(z)` summed over it.
    fn on_ball(&self) -> f64 {
        let (x1, y1) = (
            self.far.section(self.far.last())[1],
            self.near.section(self.near.last())[1],
        );
        gauss(self.far.a, x1, |x| {
            gauss(self.near.a, y1, |y| {
                BALL / (BALL * BALL - x * x - y * y).sqrt()
            })
        })
    }

    /// The corner's flat sides on the far wall's plane and the near wall's, and on the far
    /// wall's the step left above the far blend's contact, under the near blend's end.
    fn flats(&self) -> [f64; 3] {
        let start = self.across_near(0.0);
        let on_far_wall = |pick: fn([f64; 3]) -> f64| {
            move |ty: f64| pick(self.heights(0.0, ty)) * self.near.section(ty)[3]
        };
        let far_wall = gauss(0.0, start, on_far_wall(|h| h[0] - h[2]))
            + gauss(start, self.near.last(), on_far_wall(|h| h[1] - h[2]));
        let step = gauss(0.0, start, on_far_wall(|h| h[1] - h[0]));
        let near_wall = gauss(0.0, self.far.last(), |tx| {
            let [over_far, over_near, ball] = self.heights(tx, 0.0);
            assert!(over_far <= over_near);
            (over_far - ball) * self.far.section(tx)[3]
        });
        [far_wall, near_wall, step]
    }

    /// The volume by the divergence theorem: a third of the flux of position out through the
    /// corner's five sides. Position dotted with the outward normal is `−R` on the ball, minus
    /// the wall's distance on each flat side, and `(ρx′ − xρ′)` over the section's speed on a
    /// blend's surface.
    fn volume_by_flux(&self) -> f64 {
        let flux = |[rho, x, d_rho, d_x]: [f64; 4]| rho * d_x - x * d_rho;
        let [far_wall, near_wall, _] = self.flats();
        (self.on_far(flux) + self.on_near(flux)
            - BALL * self.on_ball()
            - self.far.a * far_wall
            - self.near.a * near_wall)
            / 3.0
    }

    /// The area the corner adds: the two patches and the step, less the ball it covers.
    fn area(&self) -> f64 {
        let speed = |[_, _, d_rho, d_x]: [f64; 4]| d_rho.hypot(d_x);
        self.on_far(speed) + self.on_near(speed) + self.flats()[2] - self.on_ball()
    }

    /// The crease's two ends as `(x, y, z)`: where the near blend's surface comes down to the
    /// far blend's contact on its wall, and where the two contacts on the ball meet.
    fn ends(&self) -> [[f64; 3]; 2] {
        let at = |tx: f64, ty: f64| {
            let (x, y) = (self.far.section(tx)[1], self.near.section(ty)[1]);
            [x, y, self.heights(tx, ty)[0]]
        };
        [
            at(0.0, self.across_near(0.0)),
            at(self.far.last(), self.near.last()),
        ]
    }
}

/// The fillet face named `name` rolls along wall `n`: it is the torus a ball of the fillet's
/// radius sweeps touching the wall and the ball, every point of its boundary lies on that torus,
/// on the wall's plane where it is tangent to the wall, on the ball where it is tangent to the
/// ball, on one of the two planes it stops on, or on the torus of a fillet it crosses, one of
/// `crossed`; and it has a boundary of each of the first two kinds.
fn assert_rolls(b: &Brep, name: &str, n: Vec3, foot: &Foot, crossed: &[(Vec3, Foot)]) {
    let Size::Round(r) = foot.size else {
        panic!("a fillet's face");
    };
    let faces: Vec<_> = b
        .face_ids()
        .filter(|&f| b.face(f).names.iter().any(|x| x.to_string() == name))
        .collect();
    assert_eq!(faces.len(), 1, "{name}");
    let face = b.face(faces[0]);
    let Surface::Torus(torus) = &face.surface else {
        panic!("{name} is a {}", face.surface.kind());
    };
    let exact = 1e-12 * BALL;
    assert!(
        (torus.frame.origin - n * (foot.a + r)).norm() <= exact,
        "{name}"
    );
    assert!(torus.frame.z.cross(n).norm() <= 1e-12, "{name}");
    assert!((torus.major - foot.spine(r)).abs() <= exact, "{name}");
    assert!((torus.minor - r).abs() <= exact, "{name}");

    let along = Vec3::Z.cross(n);
    let near = |x: f64| x.abs() <= 1e-9 * BALL;
    let on_torus = |q: Vec3, n: Vec3, foot: &Foot| {
        let x = q.dot(n);
        let rho = (q - n * x).norm();
        near((rho - foot.spine(r)).hypot(x - foot.a - r) - r)
    };
    let normal = |q: Vec3| {
        let uv = face.surface.project(q);
        face.surface.normal(uv.x, uv.y)
    };
    let mut seen = [0usize; 3];
    for &lp in &face.loops {
        for &c in &b.lp(lp).coedges {
            let edge = b.edge(b.coedge(c).edge);
            let curve = edge.curve.as_ref().expect("a fillet's edges are curves");
            let points: Vec<Vec3> = (0..=8)
                .map(|k| curve.eval(edge.range.0 + (edge.range.1 - edge.range.0) * k as f64 / 8.0))
                .collect();
            assert!(points.iter().all(|&q| on_torus(q, n, foot)), "{name}");
            // An end on the plane of the wall beside it may lie along the post's edge, on its own
            // wall's plane too: it is an end first.
            let stops = |stop: Stop| {
                let on = |q: &Vec3| near(q.dot(along) - stop.at(q.dot(n)));
                points.iter().all(on)
            };
            if stops(foot.stops[0]) || stops(foot.stops[1]) {
                seen[2] += 1;
            } else if points.iter().all(|q| near(q.dot(n) - foot.a)) {
                assert!(points.iter().all(|&q| normal(q).cross(n).norm() <= 1e-9));
                seen[0] += 1;
            } else if points.iter().all(|q| near(q.norm() - BALL)) {
                let radial = |q: Vec3| normal(q).cross(q).norm() <= 1e-9 * BALL;
                assert!(points.iter().all(|&q| radial(q)), "{name}");
                seen[1] += 1;
            } else {
                let crosses =
                    |(m, other): &(Vec3, Foot)| points.iter().all(|&q| on_torus(q, *m, other));
                assert!(crossed.iter().any(crosses), "{name}");
                seen[2] += 1;
            }
        }
    }
    assert!(
        seen[0] >= 1 && seen[1] >= 1 && seen[2] >= 2,
        "{name}: {seen:?}"
    );
}

/// The crease between the blend faces of walls `i` and `j`, the far and the near one, has the
/// hand-worked ends, and two rounds touch the ball there where their contacts on it meet.
fn assert_crease(b: &Brep, (i, far): (usize, Vec3), (j, near): (usize, Vec3), hand: &Crossing) {
    let face = |k: usize| {
        let name = match hand.far.size {
            Size::Round(_) => format!("foot.round_{k}"),
            Size::Bevel(_) => format!("foot.bevel_{k}"),
        };
        b.face_ids()
            .find(|&f| b.face(f).names.iter().any(|x| x.to_string() == name))
            .expect("a blend face")
    };
    let (fi, fj) = (face(i), face(j));
    let edges_of = |f| -> Vec<_> {
        let loops = b.face(f).loops.iter();
        loops
            .flat_map(|&lp| b.lp(lp).coedges.iter().map(|&c| b.coedge(c).edge))
            .collect()
    };
    let shared: Vec<_> = edges_of(fi)
        .into_iter()
        .filter(|e| edges_of(fj).contains(e))
        .collect();
    assert_eq!(shared.len(), 1, "one crease between {i} and {j}");
    let edge = b.edge(shared[0]);
    let ends = [edge.start, edge.end].map(|v| b.vertex(v).point);
    let round = matches!(hand.far.size, Size::Round(_));
    // Two blends that come to the post's edge within the tolerance of one height share one
    // corner there, within the tolerance of either's own.
    let level = (hand.far.a - hand.near.a).abs() < 1e-5;
    let within = if level { 1.5e-7 } else { 1e-9 * BALL };
    let hand = hand
        .ends()
        .map(|[x, y, z]| far * x + near * y + Vec3::Z * z);
    for (k, h) in hand.iter().enumerate() {
        let off = ends
            .iter()
            .map(|p| p.dist(*h))
            .fold(f64::INFINITY, f64::min);
        assert!(off <= within, "the crease's end {k} is {off} off");
    }
    if round {
        for f in [fi, fj] {
            let s = &b.face(f).surface;
            let uv = s.project(hand[1]);
            assert!(s.normal(uv.x, uv.y).cross(hand[1]).norm() <= 1e-9 * BALL);
        }
    }
}

/// The post of `size` on the ball at `centre` with the foot of each wall of `walls` blended at
/// `blend`, each blend stopping as `stops` gives for it, and each two of `crossing`, by their
/// places in `walls`, crossing past the corner between them: every fillet face held to the
/// hand-worked torus, every crease to its hand-worked ends, and the solid to the hand-worked
/// volume and area. A crossing corner's volume is worked two ways, which must agree.
fn assert_foot(
    (centre, size): (Vec2, Vec2),
    blend: Size,
    walls: &[usize],
    stops: &[[Stop; 2]],
    crossing: &[[usize; 2]],
) -> Brep {
    let body = post_on_ball(centre, size);
    let pairs: Vec<(&str, &str)> = walls.iter().map(|&w| (WALLS[w].0, "ball.skin")).collect();
    let b = match blend {
        Size::Round(r) => fillet(&body, &pairs, r, "foot"),
        Size::Bevel(d) => chamfer(&body, &pairs, d, "foot"),
    }
    .unwrap();
    let foot = |k: usize| {
        let a = wall_place(centre, size, walls[k]).0;
        let stops = stops[k];
        (
            WALLS[walls[k]].1,
            Foot {
                a,
                size: blend,
                stops,
            },
        )
    };
    let (mut volume, mut area) = post_on_ball_measures(centre, size);
    for k in 0..walls.len() {
        let (n, f) = foot(k);
        let crossed: Vec<(Vec3, Foot)> = crossing
            .iter()
            .filter(|pair| pair.contains(&k))
            .map(|pair| foot(pair[0] + pair[1] - k))
            .collect();
        if let Size::Round(_) = blend {
            assert_rolls(&b, &format!("foot.round_{k}"), n, &f, &crossed);
        }
        volume += f.volume();
        area += f.area();
    }
    for &[i, j] in crossing {
        let [(i, (ni, fi)), (j, (nj, fj))] = if foot(i).1.a > foot(j).1.a {
            [(i, foot(i)), (j, foot(j))]
        } else {
            [(j, foot(j)), (i, foot(i))]
        };
        let hand = Crossing {
            far: &fi,
            near: &fj,
        };
        let (columns, flux) = (hand.volume_by_columns(), hand.volume_by_flux());
        assert!(
            (columns - flux).abs() <= 1e-9 * columns,
            "the corner by columns {columns} and by flux {flux}"
        );
        volume += columns;
        area += hand.area();
        assert_crease(&b, (i, ni), (j, nj), &hand);
    }
    assert_solid(&b, volume, area);
    assert_eq!(b.face_count(), 6 + walls.len());
    let surface = match blend {
        Size::Round(_) => "torus",
        Size::Bevel(_) => "cone",
    };
    assert_eq!(kinds(&b, surface), walls.len());
    assert_eq!(kinds(&b, "sphere"), 1);
    b
}

const FILLET: Size = Size::Round(5.92);
const MITRED: [Stop; 2] = [Stop::Mitre(-1.0), Stop::Mitre(1.0)];

const SQUARE: (Vec2, Vec2) = (Vec2::ZERO, POST);

/// The stops of a blend on wall `w` of the post that ends on the planes of both walls beside it.
fn free(centre: Vec2, size: Vec2, w: usize) -> [Stop; 2] {
    let (_, middle, half) = wall_place(centre, size, w);
    [Stop::Wall(middle - half), Stop::Wall(middle + half)]
}

/// The same where the walls beside it are blended too, and it crosses their blends.
fn under(centre: Vec2, size: Vec2, w: usize) -> [Stop; 2] {
    let (_, middle, half) = wall_place(centre, size, w);
    [Stop::Under(middle - half), Stop::Under(middle + half)]
}

#[test]
fn a_square_post_on_a_ball_takes_fillets_mitred_at_its_corners() {
    // Each corner's two walls are mirror images in a plane through the ball's centre, so the two
    // tori cross on that plane and nothing else closes the corner.
    let b = assert_foot(SQUARE, FILLET, &[0, 1, 2, 3], &[MITRED; 4], &[]);
    assert_eq!(kinds(&b, "plane"), 5);
    for k in 0..4 {
        assert_eq!(names(&b, &format!("foot.round_{k}")), 1);
    }
}

#[test]
fn a_foot_round_the_seam_of_the_ball_parts_is_built_as_one_face() {
    // The ball's seam meets the foot of the wall `u_max` at its middle, y = 0, and parts that
    // edge in two. The round along both is one face of the solid as it is sewn: one edge along
    // the wall, where nothing else ends, and two along the ball, which meet where the seam now
    // ends, on the round's section through the seam.
    for (c, s) in [SQUARE, (Vec2::new(5.0, 3.0), POST)] {
        let pairs: Vec<(&str, &str)> = WALLS.iter().map(|w| (w.0, "ball.skin")).collect();
        let b = fillet(&post_on_ball(c, s), &pairs, 5.92, "foot").unwrap();
        let round = b
            .face_ids()
            .find(|&f| {
                b.face(f)
                    .names
                    .iter()
                    .any(|x| x.to_string() == "foot.round_1")
            })
            .unwrap();
        let wall = wall_place(c, s, 1).0;
        let (mut on_wall, mut on_ball, mut seam_ends) = (0, 0, 0);
        for &lp in &b.face(round).loops {
            for &co in &b.lp(lp).coedges {
                let edge = b.edge(b.coedge(co).edge);
                let curve = edge.curve.as_ref().unwrap();
                let points: Vec<Vec3> = (0..=4)
                    .map(|k| {
                        curve.eval(edge.range.0 + (edge.range.1 - edge.range.0) * k as f64 / 4.0)
                    })
                    .collect();
                if points.iter().all(|q| (q.x - wall).abs() <= 1e-9) {
                    on_wall += 1;
                } else if points.iter().all(|q| (q.norm() - BALL).abs() <= 1e-9) {
                    on_ball += 1;
                    let ends = [edge.start, edge.end].map(|v| b.vertex(v).point);
                    seam_ends += ends.iter().filter(|q| q.y.abs() <= 1e-9).count();
                }
            }
        }
        assert_eq!((on_wall, on_ball, seam_ends), (1, 2, 2));
    }
}

#[test]
fn a_square_post_on_a_ball_takes_chamfers_mitred_at_its_corners() {
    let b = assert_foot(SQUARE, Size::Bevel(3.0), &[0, 1, 2, 3], &[MITRED; 4], &[]);
    for k in 0..4 {
        assert_eq!(names(&b, &format!("foot.bevel_{k}")), 1);
    }
}

#[test]
fn a_foot_fillet_on_a_ball_ends_on_the_walls_beside_it() {
    let (c, s) = SQUARE;
    assert_foot(SQUARE, FILLET, &[1], &[free(c, s, 1)], &[]);
    assert_foot(
        SQUARE,
        FILLET,
        &[1, 3],
        &[free(c, s, 1), free(c, s, 3)],
        &[],
    );
    // Off the ball's centre the two ends differ.
    let c = Vec2::new(5.0, 3.0);
    assert_foot((c, s), FILLET, &[1], &[free(c, s, 1)], &[]);
    assert_foot((c, s), FILLET, &[0], &[free(c, s, 0)], &[]);
}

#[test]
fn foot_fillets_on_a_ball_mitre_where_they_meet_and_end_on_a_wall_where_they_do_not() {
    let (c, s) = SQUARE;
    let ([lo, _], [_, hi]) = (free(c, s, 0), free(c, s, 2));
    let two = [
        [lo, Stop::Mitre(1.0)],
        [Stop::Mitre(-1.0), free(c, s, 1)[1]],
    ];
    assert_foot(SQUARE, FILLET, &[0, 1], &two, &[]);
    let three = [[lo, Stop::Mitre(1.0)], MITRED, [Stop::Mitre(-1.0), hi]];
    assert_foot(SQUARE, FILLET, &[0, 1, 2], &three, &[]);
}

#[test]
fn foot_fillets_mitre_at_the_corners_whose_mirror_plane_holds_the_centre_of_the_ball() {
    // Moved along its diagonal, the post keeps the plane x = y through the ball's centre as the
    // mirror of the corners between `u_max` and `v_max` and between `u_min` and `v_min`; no plane
    // mirrors the other two, where the blends cross.
    let (c, s) = (Vec2::new(4.0, 4.0), POST);
    let at = |w: usize, k: usize| under(c, s, w)[k];
    let stops = [
        [Stop::Mitre(-1.0), at(0, 1)],
        [at(1, 0), Stop::Mitre(1.0)],
        [Stop::Mitre(-1.0), at(2, 1)],
        [at(3, 0), Stop::Mitre(1.0)],
    ];
    assert_foot((c, s), FILLET, &[0, 1, 2, 3], &stops, &[[0, 1], [2, 3]]);
    let ([lo, _], [_, hi]) = (free(c, s, 1), free(c, s, 2));
    let two = [[lo, Stop::Mitre(1.0)], [Stop::Mitre(-1.0), hi]];
    assert_foot((c, s), FILLET, &[1, 2], &two, &[]);
}

#[test]
fn foot_blends_of_a_post_off_the_centre_of_a_ball_cross_past_its_corners() {
    // Off the ball's centre no plane mirrors a corner's two blends: their surfaces differ, and
    // reach the post's edge at different heights. Each blend ends on the plane of the wall beside
    // it, and past the corner the lower of the two bounds the solid, up to the curve they cross
    // on.
    let (c, s) = (Vec2::new(5.0, 3.0), POST);
    let all = [0, 1, 2, 3].map(|w| under(c, s, w));
    let round = [[0, 1], [1, 2], [2, 3], [3, 0]];
    let two = [
        [free(c, s, 0)[0], under(c, s, 0)[1]],
        [under(c, s, 1)[0], free(c, s, 1)[1]],
    ];
    assert_foot((c, s), FILLET, &[0, 1], &two, &round[..1]);
    for size in [
        Size::Round(0.5),
        FILLET,
        Size::Round(12.0),
        Size::Bevel(3.0),
    ] {
        assert_foot((c, s), size, &[0, 1, 2, 3], &all, &round);
    }
}

#[test]
fn foot_fillets_a_hair_off_the_centre_of_a_ball_build_on_both_sides_of_one_level() {
    // A post ε off the ball's centre brings two foot rounds of 5.92 to its edge 0.147·ε apart.
    // Within 1.5e-7 of each other they share one corner on the edge; farther apart the higher
    // round has an end of its own between the two. Either way the solid is the one worked by
    // hand, the corner of two rounds that cross.
    let round = [[0, 1], [1, 2], [2, 3], [3, 0]];
    for offset in [5e-7, 6.9e-7, 9e-7, 1.1e-6, 2e-6, 1e-5] {
        let (c, s) = (Vec2::new(offset, 0.0), POST);
        let all = [0, 1, 2, 3].map(|w| under(c, s, w));
        assert_foot((c, s), FILLET, &[0, 1, 2, 3], &all, &round);
    }
}

#[test]
fn chamfers_so_wide_that_their_feet_on_a_ball_pass_each_other_are_refused_by_name() {
    // Far off the ball's centre and 12 wide, the chamfers of two walls reach the ball on the
    // planes x = 34.3 and y = −27.3, which meet outside it: their contacts on the ball never
    // cross, and the corner has no far end.
    let (c, s, wide) = (Vec2::new(12.0, -3.0), POST, Size::Bevel(12.0));
    let reach = |w: usize| {
        let foot = Foot {
            a: wall_place(c, s, w).0,
            size: wide,
            stops: free(c, s, w),
        };
        foot.section(1.0)[1]
    };
    assert!(reach(0).hypot(reach(1)) > BALL);
    let pairs: Vec<(&str, &str)> = WALLS.iter().map(|w| (w.0, "ball.skin")).collect();
    let Err(Error::Unsupported(why)) = chamfer(&post_on_ball(c, s), &pairs, 12.0, "foot") else {
        panic!("chamfers whose feet pass each other");
    };
    assert!(
        why.starts_with("chamfers that cross past the corner at"),
        "{why}"
    );
    assert!(why.ends_with("pass without meeting"), "{why}");
}

#[test]
fn foot_fillets_of_a_post_longer_than_it_is_wide_cross_past_its_corners() {
    // In the middle of the ball, but no square: the long walls stand nearer the centre than the
    // short ones, and no plane through a corner mirrors one onto the other.
    let (c, s) = (Vec2::ZERO, Vec2::new(20.0, 34.0));
    let all = [0, 1, 2, 3].map(|w| under(c, s, w));
    let round = [[0, 1], [1, 2], [2, 3], [3, 0]];
    assert_foot((c, s), Size::Round(4.0), &[0, 1, 2, 3], &all, &round);
    assert_foot((c, s), Size::Bevel(2.5), &[0, 1, 2, 3], &all, &round);
}

#[test]
fn foot_blends_that_reach_over_the_pole_of_a_ball_take_it_and_its_seam_with_them() {
    // A post 8 by 30 at (5, 3): its wall `u_min` stands at x = 1, past the ball's centre from
    // its own side, and the ball's seam runs from that wall's foot 1 mm on to the pole. A round
    // of 0.5 stops short of the pole, at x = 0.49; one of 2 touches the ball at x = −0.95, and
    // the larger ones farther out, so the pole and that stretch of the seam lie under the blend.
    let (c, s) = (Vec2::new(5.0, 3.0), Vec2::new(8.0, 30.0));
    let reach = |r: f64| (1.0 - r) * BALL / (BALL + r);
    assert!(reach(0.5) > 0.0 && reach(2.0) < 0.0);
    let all = [0, 1, 2, 3].map(|w| under(c, s, w));
    let round = [[0, 1], [1, 2], [2, 3], [3, 0]];
    for size in [
        Size::Round(0.5),
        Size::Round(2.0),
        FILLET,
        Size::Round(12.0),
        Size::Bevel(2.0),
        Size::Bevel(5.92),
    ] {
        assert_foot((c, s), size, &[0, 1, 2, 3], &all, &round);
    }
    // The wall alone, ending on the walls beside it, and with one neighbour.
    assert_foot((c, s), FILLET, &[3], &[free(c, s, 3)], &[]);
    let two = [
        [free(c, s, 3)[0], under(c, s, 3)[1]],
        [under(c, s, 0)[0], free(c, s, 0)[1]],
    ];
    assert_foot((c, s), FILLET, &[3, 0], &two, &[[0, 1]]);
    // The same post turned a quarter turn, its short wall beside the pole.
    let (c, s) = (Vec2::new(16.0, 1.0), Vec2::new(30.0, 8.0));
    let all = [0, 1, 2, 3].map(|w| under(c, s, w));
    assert_foot((c, s), FILLET, &[0, 1, 2, 3], &all, &round);
}

#[test]
fn a_foot_blend_that_reaches_over_a_seam_its_edge_does_not_meet_is_refused_by_name() {
    // The post 8 by 30 at (5, 16) stands 1 mm beside the ball's seam, which runs along its wall
    // `v_min` and meets none of its edges. A round of 2 at that wall's foot touches the ball at
    // y = −0.95, past the seam: its end on the plane of `u_min` crosses the seam at (1, 0), and
    // the round of `u_max` beside it crosses it again. Sewn as they are, the ball's face would
    // have the seam running through the hole the rounds leave in it.
    let (c, s) = (Vec2::new(5.0, 16.0), Vec2::new(8.0, 30.0));
    for walls in [&[0usize, 1][..], &[0, 1, 2, 3]] {
        let pairs: Vec<(&str, &str)> = walls.iter().map(|&w| (WALLS[w].0, "ball.skin")).collect();
        let Err(Error::Unsupported(why)) = fillet(&post_on_ball(c, s), &pairs, 2.0, "foot") else {
            panic!("rounds over a seam");
        };
        assert!(
            why.contains("crosses another edge of that face at ("),
            "{why}"
        );
        assert!(why.contains(", 0.000, "), "{why}");
    }
}

#[test]
fn a_foot_round_whose_contact_passes_through_the_pole_of_its_ball_is_built() {
    // The post 8 by 30 at (−6, −2) has its wall `u_max` at x = −2, and a round of 2 there
    // touches the ball on the plane x = 0, a meridian through the pole: the contact meets the
    // seam's end at the pole, where a face may touch itself, and crosses nothing.
    let (c, s) = (Vec2::new(-6.0, -2.0), Vec2::new(8.0, 30.0));
    let all = [0, 1, 2, 3].map(|w| under(c, s, w));
    let round = [[0, 1], [1, 2], [2, 3], [3, 0]];
    assert_foot((c, s), Size::Round(2.0), &[0, 1, 2, 3], &all, &round);
}

#[test]
fn rim_rounds_whose_crease_leaves_no_room_between_its_ends_are_built_or_refused() {
    // A hole 28.5 square at (−6, −2) through the ball, its rim rounded at 5.92: at one corner a
    // line across the lower round's strip leaves the box between the crease's two ends at the
    // very point it enters it, within rounding, and there is no stretch of it to look along.
    let ball = sphere(&at_z(0.0), BALL, "ball").unwrap();
    let foot = Profile::rect(Vec2::new(-6.0, -2.0), POST.x, POST.y);
    let post = extrude(&foot, &at_z(0.0), TOP, "post").unwrap();
    let body = difference(&ball, &post).unwrap();
    let pairs: Vec<(&str, &str)> = WALLS.iter().map(|w| (w.0, "ball.skin")).collect();
    match fillet(&body, &pairs, 5.92, "rim") {
        Ok(b) => assert!(check::check(&b).is_empty()),
        Err(Error::Unsupported(why)) => {
            assert!(
                why.starts_with("fillets that cross past the corner at"),
                "{why}"
            )
        }
        Err(other) => panic!("{other:?}"),
    }
}

#[test]
fn fillets_at_the_foot_of_a_leaning_wall_and_an_upright_one_cross_past_their_corner() {
    // A post on a plate, 20 square at its foot, whose wall `lean` leans in by 0.1 for each 1 up
    // while its other walls stand upright, with the feet of `lean` and of the upright wall
    // `start` beside it rounded at 3. The upright wall's round is a quarter circle; the leaning
    // wall's ball touches the plate `t` from the foot, where t = r·(1 − sin β)/cos β for the lean
    // β, and the wall r·(1 − sin β) up. No plane mirrors one round onto the other.
    let (r, slope, side, height) = (3.0, 0.1f64, 20.0, 30.0);
    let plate = extrude(
        &Profile::rect(Vec2::ZERO, 60.0, 60.0),
        &at_z(0.0),
        10.0,
        "plate",
    )
    .unwrap();
    let outline = [
        Vec2::new(-10.0, 10.0),
        Vec2::new(10.0, 10.0),
        Vec2::new(10.0 - slope * height, 10.0 + height),
        Vec2::new(-10.0, 10.0 + height),
    ];
    let upright = Frame::new(Vec3::new(0.0, 10.0, 0.0), -Vec3::Y, Vec3::X).unwrap();
    let post = extrude(
        &Profile::polygon(&outline, &["foot", "lean", "top", "left"]),
        &upright,
        side,
        "post",
    )
    .unwrap();
    let body = union(&plate, &post).unwrap();
    let pairs = [("post.lean", "plate.end"), ("post.start", "plate.end")];
    let b = fillet(&body, &pairs, r, "f").unwrap();

    let lean = slope.atan();
    let slant = (1.0 + slope * slope).sqrt();
    let t = r * (1.0 - lean.sin()) / lean.cos();
    let top = FRAC_PI_2 - lean;
    // The leaning round's section: the kite of the two tangents less the ball's sector. It runs
    // the whole wall, ending square on the upright wall `end`.
    let kite = r * t - 0.5 * r * r * top;
    // The upright round's section, a square less a quarter disc, and its first moment in z: it
    // ends on the leaning wall's plane, which takes z·tan β off its length at the height z.
    let quarter = r * r * (1.0 - PI / 4.0);
    let moment = r.powi(3) * (5.0 / 6.0 - PI / 4.0);
    // Past the corner the two rounds cross. The leaning round's section is run by the angle φ its
    // ball has turned from the plate's contact: r·(1 − cos φ) high and as wide as from the wall's
    // plane to the arc. The upright round stands that high r·(1 − sin φ) from its wall.
    let width = |phi: f64| t - r * phi.sin() + r * (1.0 - phi.cos()) * slope;
    let reach = |phi: f64| r * (1.0 - phi.sin());
    // The corner's volume by slices square to the upright wall: the leaning section below the
    // height the upright round stands at there.
    let below = |turned: f64| gauss(0.0, turned.min(top), |phi| width(phi) * r * phi.sin());
    let across = |a: f64, b: f64| gauss(a, b, |psi| below(psi) * r * psi.cos());
    let by_walls = across(0.0, top) + across(top, FRAC_PI_2);
    // And by level slices: at each height the leaning section's width by the upright's reach.
    let by_levels = gauss(0.0, top, |phi| width(phi) * reach(phi) * r * phi.sin());
    assert!((by_walls - by_levels).abs() <= 1e-12 * by_walls);
    let post_volume = 0.5 * (side + side - slope * height) * height * side;
    let volume = 36000.0 + post_volume + kite * side + quarter * side - slope * moment + by_walls;

    let plain = 2.0 * 3600.0 + 4.0 * 600.0 - side * side
        + (side - slope * height) * side
        + side * height
        + (side + side - slope * height) * height
        + side * height * slant;
    // Each round's face and flat end, less the strips of the plate and the wall it covers.
    let leaning = top * r * side - 2.0 * t * side + kite;
    let standing = r * (side * FRAC_PI_2 - slope * r * (FRAC_PI_2 - 1.0))
        - r * side
        - (side * r - 0.5 * slope * r * r)
        + quarter;
    // The corner: each round's patch up to the crease, less the plate beneath, and the step of
    // the upright round's end left on the leaning wall's plane above the leaning round.
    let step = gauss(top, FRAC_PI_2, |psi| {
        r * (top.cos() - psi.cos()) * r * psi.cos()
    }) * slant;
    let corner = gauss(0.0, top, |phi| width(phi) * r) + gauss(0.0, top, |phi| reach(phi) * r)
        - r * t
        + step;
    assert_solid(&b, volume, plain + leaning + standing + corner);
    assert_eq!(kinds(&b, "cylinder"), 2);
    assert_eq!(names(&b, "f.round_0"), 1);
    assert_eq!(names(&b, "f.round_1"), 1);
}
