use std::f64::consts::{FRAC_1_SQRT_2, PI};

use super::*;
use crate::{geom::NurbsCurve, measure};

const H: f64 = FRAC_1_SQRT_2;
const CIRCLE: [(f64, f64); 9] = [
    (1., 0.),
    (1., 1.),
    (0., 1.),
    (-1., 1.),
    (-1., 0.),
    (-1., -1.),
    (0., -1.),
    (1., -1.),
    (1., 0.),
];
const CIRCLE_W: [f64; 9] = [1., H, 1., H, 1., H, 1., H, 1.];
const CIRCLE_KNOTS: [f64; 12] = [0., 0., 0., 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1., 1., 1.];

fn v3(x: f64, y: f64, z: f64) -> Point3 {
    Vec3::new(x, y, z)
}

/// The biquadratic 9 × 5 rational sphere of radius `r`, its rows `j = 0` and `4` its poles.
fn nurbs_sphere(r: f64) -> NurbsSurface {
    let meridian = [(0., -r), (r, -r), (r, 0.), (r, r), (0., r)];
    let mw = [1., H, 1., H, 1.];
    let mut ctrl = Vec::new();
    let mut weights = Vec::new();
    for (&(cx, cy), &wi) in CIRCLE.iter().zip(&CIRCLE_W) {
        for (&(rho, z), &wj) in meridian.iter().zip(&mw) {
            ctrl.push(v3(rho * cx, rho * cy, z));
            weights.push(wi * wj);
        }
    }
    NurbsSurface::new(
        (2, 2),
        (
            CIRCLE_KNOTS.to_vec(),
            vec![0., 0., 0., 0.5, 0.5, 1., 1., 1.],
        ),
        (9, 5),
        ctrl,
        Some(weights),
    )
    .unwrap()
}

fn close(a: f64, b: f64, rel: f64) {
    assert!((a - b).abs() <= rel * b.abs(), "{a} is not {b}");
}

#[test]
fn a_nurbs_sphere_closes_on_its_seam_and_poles() {
    let ball = sew(
        &[Piece::whole(nurbs_sphere(5.0), "skin").unwrap()],
        0.0,
        "ball",
    )
    .unwrap();
    assert_eq!(ball.face_count(), 1);
    close(measure::volume(&ball), 4.0 / 3.0 * PI * 125.0, 1e-9);
    close(measure::area(&ball), 4.0 * PI * 25.0, 1e-9);
}

#[test]
fn a_nurbs_cone_closes_at_its_apex_on_a_disk() {
    let mut ctrl = Vec::new();
    let mut weights = Vec::new();
    for (&(x, y), &w) in CIRCLE.iter().zip(&CIRCLE_W) {
        ctrl.extend([v3(5. * x, 5. * y, 0.), v3(0., 0., 10.)]);
        weights.extend([w, w]);
    }
    let side = NurbsSurface::new(
        (2, 1),
        (CIRCLE_KNOTS.to_vec(), vec![0., 0., 1., 1.]),
        (9, 2),
        ctrl,
        Some(weights),
    )
    .unwrap();
    let rim = side.iso_curve(false, 0.0);
    let base = Piece::planar(
        vec![vec![Span::new(Curve::Nurbs(Box::new(rim)), 0.0, 1.0)]],
        "base",
    )
    .unwrap();
    let cone = sew(&[Piece::whole(side, "side").unwrap(), base], 0.0, "cone").unwrap();
    // πr²h/3 and πr² + πr√(r² + h²), r = 5, h = 10
    close(measure::volume(&cone), 250.0 * PI / 3.0, 1e-9);
    close(
        measure::area(&cone),
        25.0 * PI + 25.0 * 5f64.sqrt() * PI,
        1e-9,
    );
}

/// The quadratic Bézier from `a` through the control point `m` to `b`.
fn arc(a: Point3, m: Point3, b: Point3) -> Span {
    let c = NurbsCurve::new(2, vec![0., 0., 0., 1., 1., 1.], vec![a, m, b], None).unwrap();
    Span::new(Curve::Nurbs(Box::new(c)), 0.0, 1.0)
}

#[test]
fn a_coons_patch_of_four_parabolas_is_their_bilinear_blend() {
    let p = |x, y, z| v3(x, y, z);
    let rim = [
        arc(p(0., 0., 10.), p(5., 0., 14.), p(10., 0., 10.)),
        arc(p(10., 0., 10.), p(10., 5., 6.), p(10., 10., 10.)),
        arc(p(10., 10., 10.), p(5., 10., 6.), p(0., 10., 10.)),
        arc(p(0., 10., 10.), p(0., 5., 14.), p(0., 0., 10.)),
    ];
    let s = coons(&rim).unwrap();
    // 10 + 8u(1 − u)(1 − 2v) + 8v(1 − v)(1 − 2u) over (10u, 10v)
    for (u, v) in [(0.3, 0.7), (0.5, 0.5), (0.9, 0.2), (0.0, 0.4)] {
        let want =
            10.0 + 8.0 * u * (1.0 - u) * (1.0 - 2.0 * v) + 8.0 * v * (1.0 - v) * (1.0 - 2.0 * u);
        let q = s.eval(u, v);
        assert!(
            q.dist(v3(10.0 * u, 10.0 * v, want)) < 1e-12,
            "{q:?} at {u}, {v}"
        );
    }
    // The block under it: its rise and fall over the square cancel, so its volume is 1000.
    let floor = |x: f64, y: f64| v3(x, y, 0.0);
    let corners = [(0., 0.), (10., 0.), (10., 10.), (0., 10.)];
    let mut pieces = vec![Piece::whole(s, "patch").unwrap()];
    for (k, top) in rim.iter().enumerate() {
        let (a, b) = (corners[k], corners[(k + 1) % 4]);
        let wall = vec![
            Span::line(floor(a.0, a.1), floor(b.0, b.1)).unwrap(),
            Span::line(floor(b.0, b.1), top.end()).unwrap(),
            Span::new(top.curve.clone(), top.t1, top.t0),
            Span::line(top.start(), floor(a.0, a.1)).unwrap(),
        ];
        pieces.push(Piece::planar(vec![wall], &format!("wall_{k}")).unwrap());
    }
    let bottom = (0..4)
        .map(|k| {
            let (a, b) = (corners[k], corners[(k + 1) % 4]);
            Span::line(floor(a.0, a.1), floor(b.0, b.1)).unwrap()
        })
        .collect();
    pieces.push(Piece::planar(vec![bottom], "floor").unwrap());
    let block = sew(&pieces, 0.0, "block").unwrap();
    assert_eq!(block.face_count(), 6);
    close(measure::volume(&block), 1000.0, 1e-9);
}

#[test]
fn two_arcs_between_one_pair_of_points_stay_two_edges() {
    // Each arc bulges 1.5 from its chord, further than the chord's half length: the point of
    // either arc nearest the chord's middle is an end, which both arcs share.
    let side = |y: f64| {
        let ctrl = vec![
            v3(-1., 0., 0.),
            v3(-1., 0., 1.),
            v3(0., y, 0.),
            v3(0., y, 1.),
            v3(1., 0., 0.),
            v3(1., 0., 1.),
        ];
        let knots = (vec![0., 0., 0., 1., 1., 1.], vec![0., 0., 1., 1.]);
        NurbsSurface::new((2, 1), knots, (3, 2), ctrl, None).unwrap()
    };
    let lens = |z: f64| {
        let p = |x: f64, y: f64| v3(x, y, z);
        vec![
            arc(p(-1., 0.), p(0., -3.), p(1., 0.)),
            arc(p(1., 0.), p(0., 3.), p(-1., 0.)),
        ]
    };
    let pieces = [
        Piece::whole(side(3.), "left").unwrap(),
        Piece::whole(side(-3.), "right").unwrap(),
        Piece::planar(vec![lens(0.)], "bottom").unwrap(),
        Piece::planar(vec![lens(1.)], "top").unwrap(),
    ];
    let prism = sew(&pieces, 0.0, "lens").unwrap();
    // Each parabolic segment holds two thirds of its chord times its height, 2 · 1.5.
    close(measure::volume(&prism), 2.0 * (2.0 / 3.0 * 2.0 * 1.5), 1e-9);
}

#[test]
fn a_nurbs_cylinders_boundary_lines_are_circles_and_a_line() {
    let mut ctrl = Vec::new();
    let mut weights = Vec::new();
    for (&(x, y), &w) in CIRCLE.iter().zip(&CIRCLE_W) {
        ctrl.extend([v3(5. * x, 5. * y, 0.), v3(5. * x, 5. * y, 10.)]);
        weights.extend([w, w]);
    }
    let knots = (CIRCLE_KNOTS.to_vec(), vec![0., 0., 1., 1.]);
    let side = NurbsSurface::new((2, 1), knots, (9, 2), ctrl, Some(weights)).unwrap();
    let piece = Piece::whole(side, "side").unwrap();
    let kinds: Vec<&str> = piece.loops()[0].iter().map(|s| s.curve.kind()).collect();
    assert_eq!(kinds, ["circle", "line", "circle", "line"]);
    for s in &piece.loops()[0] {
        if let Curve::Circle(c) = &s.curve {
            close(c.radius, 5.0, 1e-12);
            close((s.t1 - s.t0).abs(), 2.0 * PI, 1e-12);
        }
    }
}

#[test]
fn pieces_that_leave_a_gap_are_refused() {
    let half = |z: f64| {
        let ring = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)];
        let lp = (0..4)
            .map(|k| {
                let (a, b) = (ring[k], ring[(k + 1) % 4]);
                Span::line(v3(a.0, a.1, z), v3(b.0, b.1, z)).unwrap()
            })
            .collect();
        Piece::planar(vec![lp], "lid").unwrap()
    };
    assert!(sew(&[half(0.0), half(1.0)], 0.0, "open").is_err());
}
