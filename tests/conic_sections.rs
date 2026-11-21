//! Booleans whose new edges are sections of a cone, a cylinder or a sphere by a plane, against
//! closed-form volume and area: each edge is the conic itself, and the result is valid, watertight
//! and reads back from STEP as it was written.

use std::f64::consts::PI;

use springrep::{
    boolean::{difference, intersection, union},
    check,
    geom::Curve,
    math::{Frame, Vec2, Vec3},
    measure, primitive,
    profile::Profile,
    step,
    sweep::{extrude, revolve},
    topo::Brep,
};

const R: f64 = 3.0;
const H: f64 = 4.0;
const SLANT: f64 = 5.0;

/// The cone of radius 3 at z = 0 with its apex at (0, 0, 4).
fn cone() -> Brep {
    primitive::cone(&Frame::WORLD, R, 0.0, H, "cone").unwrap()
}

/// A block 40 square and 20 deep on the side of the plane through `o` that `n` points to.
fn half_space(o: [f64; 3], n: [f64; 3]) -> Brep {
    let frame = Frame::from_axis(Vec3::from_array(o), Vec3::from_array(n)).unwrap();
    extrude(&Profile::rect(Vec2::ZERO, 40.0, 40.0), &frame, 20.0, "half").unwrap()
}

fn kinds(b: &Brep) -> Vec<&'static str> {
    let mut out: Vec<_> = b
        .edge_ids()
        .filter_map(|e| b.edge(e).curve.as_ref().map(Curve::kind))
        .collect();
    out.sort_unstable();
    out
}

fn count(b: &Brep, kind: &str) -> usize {
    kinds(b).iter().filter(|k| **k == kind).count()
}

/// Every edge's curve lies on the surfaces of the faces it bounds, all along.
fn edges_lie_on_their_faces(b: &Brep, within: f64) {
    for c in b.coedge_ids() {
        let s = &b.face(b.coedge_face(c)).surface;
        {
            let edge = b.edge(b.coedge(c).edge);
            let Some(curve) = &edge.curve else {
                continue;
            };
            let (t0, t1) = edge.range;
            for i in 0..=64 {
                let p = curve.eval(t0 + (t1 - t0) * i as f64 / 64.0);
                let uv = s.project(p);
                let d = s.eval(uv.x, uv.y).dist(p);
                assert!(d <= within, "a {} off a {} by {d}", curve.kind(), s.kind());
            }
        }
    }
}

fn assert_solid(b: &Brep, volume: f64, area: f64, rel: f64) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{issues:?}");
    let (a, v) = measure::area_and_volume(b);
    assert!((v - volume).abs() <= rel * volume, "volume {v} vs {volume}");
    assert!((a - area).abs() <= rel * area, "area {a} vs {area}");
    edges_lie_on_their_faces(b, 1e-11);
    let text = step::write(b, "part").unwrap();
    let back = step::read_part(text.as_bytes(), "in").unwrap();
    assert!(check::check(&back).is_empty());
    assert_eq!(back.face_count(), b.face_count());
    let (a2, v2) = measure::area_and_volume(&back);
    assert!((v2 - v).abs() <= 1e-9 * v && (a2 - a).abs() <= 1e-9 * a);
}

/// The area of the disc of radius `r` about the origin where `x ≥ d`.
fn segment(r: f64, d: f64) -> f64 {
    if d >= r {
        0.0
    } else if d <= -r {
        PI * r * r
    } else {
        r * r * (d / r).acos() - d * ((r - d) * (r + d)).sqrt()
    }
}

/// The volume of the cone where `x ≥ c + m·z`, by Simpson's rule over its level discs' segments
/// between the heights where the plane enters and leaves the disc, each stretch run in a parameter
/// that stands still at its ends, where the segment's area has no slope.
fn cone_beyond(c: f64, m: f64) -> f64 {
    let (r, d) = (|z: f64| R * (1.0 - z / H), |z: f64| c + m * z);
    let mut cuts = vec![0.0, H];
    for sign in [1.0, -1.0] {
        // c + m·z = ±R·(1 − z/H).
        let z = (sign * R - c) / (m + sign * R / H);
        if z > 0.0 && z < H {
            cuts.push(z);
        }
    }
    cuts.sort_by(f64::total_cmp);
    let n = 4000;
    cuts.windows(2)
        .map(|w| {
            let f = |u: f64| {
                let z = w[0] + (w[1] - w[0]) * u * u * (3.0 - 2.0 * u);
                segment(r(z), d(z)) * (w[1] - w[0]) * 6.0 * u * (1.0 - u)
            };
            let sum: f64 = (0..=n)
                .map(|i| {
                    let k = if i == 0 || i == n {
                        1.0
                    } else if i % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    k * f(i as f64 / n as f64)
                })
                .sum();
            sum / (3.0 * n as f64)
        })
        .sum()
}

const CONE_VOLUME: f64 = PI * R * R * H / 3.0;
const CONE_AREA: f64 = PI * R * R + PI * R * SLANT;

#[test]
fn a_plane_along_a_cones_axis_cuts_it_in_a_hyperbola() {
    // The cone beyond x = 1: over the base's segment of half chord s the height is H·(1 − ρ/R),
    // which integrates to H·(R²·α/3 − 2·d·s/3 + d³·ln((R + s)/d)/(3·R)), α the segment's half
    // angle. Its flat side is H·(s − d²·ln((R + s)/d)/R), and its side of the cone lies over the
    // segment at the slope SLANT / R.
    let d = 1.0f64;
    let s = (R * R - d * d).sqrt();
    let log = ((R + s) / d).ln();
    let base = segment(R, d);
    let volume =
        H * (R * R * (d / R).acos() / 3.0 - 2.0 * d * s / 3.0 + d.powi(3) * log / (3.0 * R));
    let flat = H * (s - d * d * log / R);
    let piece_area = base + base * SLANT / R + flat;
    assert!((cone_beyond(d, 0.0) - volume).abs() < 1e-9);

    let tool = half_space([d, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let piece = intersection(&cone(), &tool).unwrap();
    assert_solid(&piece, volume, piece_area, 1e-9);
    assert_eq!(piece.face_count(), 3);
    assert!(count(&piece, "nurbs") > 0 && count(&piece, "intersection") == 0);

    let rest = difference(&cone(), &tool).unwrap();
    assert_solid(
        &rest,
        CONE_VOLUME - volume,
        CONE_AREA - piece_area + 2.0 * flat,
        1e-9,
    );
    assert!(count(&rest, "nurbs") > 0 && count(&rest, "intersection") == 0);

    // A box whose side x = 1 cuts the cone, standing through its base: the union loses the flat
    // section from the box, and the piece's base and side from the cone.
    let frame = Frame::new(Vec3::new(d, -10.0, -1.0), Vec3::Z, Vec3::X).unwrap();
    let block = primitive::block(&frame, Vec3::new(10.0, 20.0, 10.0), "box").unwrap();
    let both = union(&block, &cone()).unwrap();
    assert_solid(
        &both,
        2000.0 + CONE_VOLUME - volume,
        2.0 * (200.0 + 200.0 + 100.0) + CONE_AREA - piece_area,
        1e-9,
    );
    assert!(count(&both, "nurbs") > 0 && count(&both, "intersection") == 0);
}

#[test]
fn a_plane_along_a_cones_generator_cuts_it_in_a_parabola() {
    // 4·x + 3·z = 4 runs along the generator from (3, 0, 0) to the apex, through (1, 0, 0). The
    // section's vertex is where the far generator meets it, at (-1, 0, 8/3), and by Archimedes
    // its area is two thirds of its chord across the base times its height.
    let n = [4.0, 0.0, 3.0];
    let tool = half_space([1.0, 0.0, 0.0], n);
    let volume = cone_beyond(1.0, -R / H);
    let chord = 2.0 * (R * R - 1.0f64).sqrt();
    let flat = 2.0 / 3.0 * chord * Vec3::new(-2.0, 0.0, 8.0 / 3.0).norm();

    let piece = intersection(&cone(), &tool).unwrap();
    let rest = difference(&cone(), &tool).unwrap();
    for b in [&piece, &rest] {
        assert_eq!(b.face_count(), 3);
        assert!(count(b, "nurbs") > 0 && count(b, "intersection") == 0);
    }
    let (a0, _) = measure::area_and_volume(&piece);
    assert_solid(&piece, volume, a0, 1e-8);
    assert_solid(
        &rest,
        CONE_VOLUME - volume,
        CONE_AREA + 2.0 * flat - a0,
        1e-8,
    );
}

#[test]
fn a_plane_steeper_than_a_cone_cuts_it_in_a_hyperbola() {
    // x = 0.5 + 0.25·z leans less from the axis than the cone's side does.
    let tool = half_space([0.5, 0.0, 0.0], [1.0, 0.0, -0.25]);
    let volume = cone_beyond(0.5, 0.25);
    let piece = intersection(&cone(), &tool).unwrap();
    let rest = difference(&cone(), &tool).unwrap();
    for b in [&piece, &rest] {
        assert_eq!(b.face_count(), 3);
        assert!(count(b, "nurbs") > 0 && count(b, "intersection") == 0);
    }
    // The two pieces share the section, so their areas exceed the cone's by twice it.
    let section = piece
        .face_ids()
        .find(|&f| {
            let face = piece.face(f);
            face.surface.kind() == "plane" && face.names.iter().all(|n| n.face != "bottom")
        })
        .unwrap();
    let flat = measure::face_area(&piece, section);
    let (a0, _) = measure::area_and_volume(&piece);
    assert_solid(&piece, volume, a0, 1e-8);
    assert_solid(
        &rest,
        CONE_VOLUME - volume,
        CONE_AREA + 2.0 * flat - a0,
        1e-8,
    );
}

#[test]
fn a_tilted_rod_through_a_plate_is_cut_in_ellipses() {
    // A rod of radius 1 leaning 30° from the plate's normal: between the plate's faces it is
    // 2 / cos 30° long, and each end is an ellipse of area π / cos 30°.
    let tilt = 30f64.to_radians();
    let axis = Vec3::new(tilt.sin(), 0.0, tilt.cos());
    let frame = Frame::from_axis(Vec3::ZERO - axis * 10.0, axis).unwrap();
    let rod = primitive::cylinder(&frame, 1.0, 20.0, "rod").unwrap();
    let at = Frame::new(Vec3::new(-10.0, -10.0, -1.0), Vec3::Z, Vec3::X).unwrap();
    let plate = primitive::block(&at, Vec3::new(20.0, 20.0, 2.0), "plate").unwrap();
    let plug = intersection(&plate, &rod).unwrap();
    let len = 2.0 / tilt.cos();
    assert_solid(
        &plug,
        PI * len,
        2.0 * PI * len + 2.0 * PI / tilt.cos(),
        1e-9,
    );
    assert_eq!(plug.face_count(), 3);
    assert!(count(&plug, "ellipse") >= 2 && count(&plug, "intersection") == 0);
}

#[test]
fn a_ball_cut_by_a_tilted_plane_leaves_a_cap_on_a_circle() {
    let ball = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let n = Vec3::new(1.0, 2.0, 2.0) * (1.0 / 3.0);
    let tool = half_space((n * 3.0).to_array(), n.to_array());
    let cap = intersection(&ball, &tool).unwrap();
    let h = 2.0;
    assert_solid(
        &cap,
        PI * h * h * (15.0 - h) / 3.0,
        2.0 * PI * 5.0 * h + PI * 16.0,
        1e-9,
    );
    assert!(count(&cap, "circle") > 0);
    assert_eq!(count(&cap, "intersection") + count(&cap, "nurbs"), 0);
}

/// The triangle through `pts` in the plane y = 0, each a radius and a height, turned about z.
fn turned(pts: [[f64; 2]; 3], id: &str) -> Brep {
    let pts = pts.map(|p| Vec2::new(p[0], p[1]));
    let frame = Frame::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap();
    let p = Profile::polygon(&pts, &["a", "b", "c"]);
    revolve(&p, &frame, Vec3::ZERO, Vec3::Z, 2.0 * PI, id).unwrap()
}

#[test]
fn a_hexagon_chamfered_at_both_ends_is_cut_in_hyperbolas_on_every_flat() {
    // A bored hexagon 5.5 across its flats and 2.4 high, its corners cut at 30° to each end by a
    // cone that leaves a circle of radius 2.475 there: every flat meets each cone in a hyperbola.
    let (corner, h, land) = (5.5 / 3f64.sqrt(), 2.4, 2.475);
    let hexagon: Vec<Vec2> = (0..6)
        .map(|k| {
            let a = PI / 3.0 * k as f64;
            Vec2::new(corner * a.cos(), corner * a.sin())
        })
        .collect();
    let names = ["f0", "f1", "f2", "f3", "f4", "f5"];
    let mut outline = Profile::polygon(&hexagon, &names);
    outline.loops.extend(Profile::circle(Vec2::ZERO, 3.0).loops);
    let prism = extrude(&outline, &Frame::WORLD, h, "hex").unwrap();
    let slope = (PI / 6.0).tan();
    let (inner, out) = (land - 0.2 / slope, corner + 1.0);
    let rise = (out - land) * slope;
    let top = turned([[out, h - rise], [out, h + 0.2], [inner, h + 0.2]], "top");
    let bottom = turned([[inner, -0.2], [out, -0.2], [out, rise]], "bottom");
    let once = difference(&prism, &top).unwrap();
    let twice = difference(&once, &bottom).unwrap();
    for b in [&once, &twice] {
        assert!(check::check(b).is_empty(), "{:?}", check::check(b));
        assert!(count(b, "nurbs") >= 6 && count(b, "intersection") == 0);
        edges_lie_on_their_faces(b, 1e-11);
    }
    let (v0, v1, v2) = (
        measure::volume(&prism),
        measure::volume(&once),
        measure::volume(&twice),
    );
    assert!(v0 - v1 > 0.1 && ((v0 - v1) - (v1 - v2)).abs() <= 1e-9 * v0);
    assert_eq!(twice.face_count(), once.face_count() + 1);
}
