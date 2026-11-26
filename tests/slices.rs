//! Planes cut face by face (`measure::slice`), against closed forms: circles from spheres, tori and
//! lofts through circles, an ellipse from a tilted rod, a drilled block, and sheets whose cuts end
//! on their boundary. On solids the cut agrees with the boolean one (`measure::section`).

use std::f64::consts::{PI, TAU};

use springrep::{
    boolean::difference_all,
    loft::{Section, loft},
    math::{Frame, Point3, Vec2, Vec3},
    measure::{self, SectionEdge, SliceCurve},
    pick::faces_named,
    primitive,
    profile::Profile,
    sheet,
    sweep::extrude,
    topo::Brep,
};

fn at(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
}

/// Gauss–Legendre nodes and weights on [-1, 1], five points.
const GL: [(f64, f64); 5] = [
    (0.0, 0.568_888_888_888_888_9),
    (-0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
    (0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
    (-0.906_179_845_938_664, 0.236_926_885_056_189_1),
    (0.906_179_845_938_664, 0.236_926_885_056_189_1),
];

/// The length of a piece, and its share `½∮ (p − o) × dp · n` of the signed area about `n`.
fn measures(e: &SectionEdge, o: Point3, n: Vec3) -> (f64, f64) {
    let (t0, t1) = e.range;
    let k = 512;
    let (mut len, mut area) = (0.0, 0.0);
    for i in 0..k {
        let (a, b) = (
            t0 + (t1 - t0) * i as f64 / k as f64,
            t0 + (t1 - t0) * (i + 1) as f64 / k as f64,
        );
        for (x, w) in GL {
            let t = 0.5 * (a + b) + 0.5 * (b - a) * x;
            let (p, d) = e.curve.d1(t);
            let h = 0.5 * (b - a) * w;
            len += d.norm() * h;
            area += 0.5 * (p - o).cross(d).dot(n) * h;
        }
    }
    (len, if e.reversed { -area } else { area })
}

/// The total length of the curves and their summed signed area about `n`.
fn totals<'a>(
    curves: impl IntoIterator<Item = &'a [SectionEdge]>,
    o: Point3,
    n: Vec3,
) -> (f64, f64) {
    let (mut len, mut area) = (0.0, 0.0);
    for c in curves {
        for e in c {
            let (l, a) = measures(e, o, n);
            len += l;
            area += a;
        }
    }
    (len, area)
}

fn ends(e: &SectionEdge) -> (Point3, Point3) {
    let (a, b) = (e.curve.eval(e.range.0), e.curve.eval(e.range.1));
    if e.reversed { (b, a) } else { (a, b) }
}

/// Each curve's pieces follow on, a closed curve's last back to its first, and every piece lies on
/// the plane.
fn chained(curves: &[SliceCurve], o: Point3, n: Vec3) {
    for c in curves {
        assert!(!c.edges.is_empty());
        for w in c.edges.windows(2) {
            let (a, b) = (ends(&w[0]).1, ends(&w[1]).0);
            assert!(a.dist(b) < 1e-6, "{a:?} {b:?}");
        }
        if c.closed {
            let (a, b) = (ends(&c.edges[c.edges.len() - 1]).1, ends(&c.edges[0]).0);
            assert!(a.dist(b) < 1e-6, "{a:?} {b:?}");
        }
        for e in &c.edges {
            for k in 0..=16 {
                let t = e.range.0 + (e.range.1 - e.range.0) * k as f64 / 16.0;
                let h = (e.curve.eval(t) - o).dot(n);
                assert!(h.abs() < 1e-6, "{h}");
            }
        }
    }
}

fn close(got: f64, want: f64, rel: f64) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1.0),
        "{got} against {want}"
    );
}

/// The slice of a solid, its curves closed and chained, agreeing with the boolean section in
/// length and area; its total length and signed area.
fn solid_slice(b: &Brep, o: Point3, n: Vec3) -> (Vec<SliceCurve>, f64, f64) {
    let nu = n.unit().unwrap();
    let got = measure::slice(b, o, n).unwrap();
    chained(&got, o, nu);
    assert!(got.iter().all(|c| c.closed), "{got:#?}");
    let (len, area) = totals(got.iter().map(|c| c.edges.as_slice()), o, nu);
    let want = measure::section_curves(b, o, n).unwrap();
    let (wlen, _) = totals(want.iter().map(Vec::as_slice), o, nu);
    let warea = measure::section_area(b, o, n).unwrap();
    close(len, wlen, 1e-6);
    close(area, warea, 1e-6);
    (got, len, area)
}

#[test]
fn a_sphere_is_cut_in_a_circle() {
    // R = 5 at z = 3: a circle of radius 4.
    let s = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let o = Point3::new(0.0, 0.0, 3.0);
    let (got, len, area) = solid_slice(&s, o, Vec3::Z);
    assert_eq!(got.len(), 1);
    close(len, TAU * 4.0, 1e-9);
    close(area, PI * 16.0, 1e-9);
    // From below the same circle runs the other way round: the area about the normal is the same.
    let (_, _, area) = solid_slice(&s, o, -Vec3::Z);
    close(area, PI * 16.0, 1e-9);
}

#[test]
fn a_rod_cut_aslant_is_an_ellipse() {
    // A Ø4 rod 10 long cut at 40° to its cross-section through its middle: an ellipse of half axes
    // 2 and 2 / cos 40°.
    let rod = primitive::cylinder(&Frame::WORLD, 2.0, 10.0, "rod").unwrap();
    let th = 40f64.to_radians();
    let n = Vec3::new(0.0, th.sin(), th.cos());
    let o = Point3::new(0.0, 0.0, 5.0);
    let (got, _, area) = solid_slice(&rod, o, n);
    assert_eq!(got.len(), 1);
    close(area, PI * 2.0 * 2.0 / th.cos(), 1e-9);
    for e in &got[0].edges {
        for k in 0..=8 {
            let p = e
                .curve
                .eval(e.range.0 + (e.range.1 - e.range.0) * k as f64 / 8.0);
            close(p.x.hypot(p.y), 2.0, 1e-9);
        }
    }
}

#[test]
fn a_torus_is_cut_in_two_circles_across_and_along_its_axis() {
    let t = primitive::torus(&Frame::WORLD, 5.0, 1.0, "ring").unwrap();
    // Square to the axis through the centre: circles of radius 6 and 4, the hole's against it.
    let (got, len, area) = solid_slice(&t, Point3::ZERO, Vec3::Z);
    assert_eq!(got.len(), 2);
    close(len, TAU * 10.0, 1e-9);
    close(area, PI * (36.0 - 16.0), 1e-9);
    // Through the axis: two circles of radius 1 about (±5, 0, 0).
    let (got, len, area) = solid_slice(&t, Point3::ZERO, Vec3::Y);
    assert_eq!(got.len(), 2);
    close(len, 2.0 * TAU, 1e-9);
    close(area, 2.0 * PI, 1e-9);
}

/// A 40 × 30 × 10 block with eighteen Ø2 holes through it, in six columns of three.
fn drilled() -> Brep {
    let block = extrude(
        &Profile::rect(Vec2::ZERO, 40.0, 30.0),
        &at(0.0),
        10.0,
        "blk",
    )
    .unwrap();
    let holes: Vec<Brep> = (0..6)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| {
            let c = Vec2::new(-15.0 + 6.0 * i as f64, -8.0 + 8.0 * j as f64);
            extrude(&Profile::circle(c, 2.0), &at(-1.0), 12.0, "hole").unwrap()
        })
        .collect();
    let refs: Vec<&Brep> = holes.iter().collect();
    difference_all(&block, &refs).unwrap()
}

#[test]
fn a_drilled_block_is_cut_in_its_outline_and_its_holes() {
    let b = drilled();
    let (got, len, area) = solid_slice(&b, Point3::new(0.0, 0.0, 4.0), Vec3::Z);
    assert_eq!(got.len(), 19);
    close(len, 140.0 + 18.0 * TAU, 1e-9);
    close(area, 1200.0 - 18.0 * PI, 1e-9);
    // Upright and aslant across the rows, through some holes: less than the block's cut there,
    // and as the boolean cuts it.
    let o = Point3::new(0.0, 0.0, 5.0);
    let (_, _, area) = solid_slice(&b, o, Vec3::new(0.3, 1.0, 0.0));
    let w = 40.0 * (1.0f64 + 0.09).sqrt();
    assert!(area > 0.0 && area < w * 10.0, "{area}");
    // The block's top lies in the plane: it is its own cut, the outline and the eighteen holes.
    let top = measure::slice(&b, Point3::new(0.0, 0.0, 10.0), Vec3::Z).unwrap();
    assert_eq!(top.len(), 19);
    let (_, area) = totals(
        top.iter().map(|c| c.edges.as_slice()),
        Point3::ZERO,
        Vec3::Z,
    );
    close(area, 1200.0 - 18.0 * PI, 1e-9);
}

#[test]
fn a_smooth_loft_through_circles_is_cut_in_them() {
    // Circles of radius 2, 3 and 2 at heights 0, 2 and 4: the side passes through each.
    let sections: Vec<Section> = [(0.0, 2.0), (2.0, 3.0), (4.0, 2.0)]
        .iter()
        .map(|&(z, r)| Section::new(Profile::circle(Vec2::ZERO, 2.0 * r), at(z)))
        .collect();
    let b = loft(&sections, false, "lf").unwrap();
    // At the middle section the boolean section is refused (#94), so only the closed form is
    // checked.
    let o = Point3::new(0.0, 0.0, 2.0);
    let got = measure::slice(&b, o, Vec3::Z).unwrap();
    chained(&got, o, Vec3::Z);
    assert_eq!(got.len(), 1);
    assert!(got[0].closed);
    let (len, area) = totals([got[0].edges.as_slice()], o, Vec3::Z);
    close(len, TAU * 3.0, 1e-6);
    close(area, PI * 9.0, 1e-6);
    // Between sections, and aslant, as the boolean cut.
    solid_slice(&b, Point3::new(0.0, 0.0, 1.3), Vec3::Z);
    solid_slice(&b, Point3::new(0.0, 0.0, 2.0), Vec3::new(0.2, 0.1, 1.0));
}

#[test]
fn a_sheet_is_cut_in_curves_that_end_on_its_boundary() {
    // A rod's side alone: square to its axis a whole circle; along the axis two lines from the
    // bottom rim to the top, each the rod's length.
    let rod = primitive::cylinder(&Frame::WORLD, 2.0, 10.0, "rod").unwrap();
    let side: Vec<_> = faces_named(&rod, "rod.side").into_iter().collect();
    assert!(!side.is_empty());
    let s = sheet::from_faces(&rod, &side).unwrap();
    assert!(s.solid_ids().next().is_none());
    let o = Point3::new(0.0, 0.0, 3.0);
    let got = measure::slice(&s, o, Vec3::Z).unwrap();
    chained(&got, o, Vec3::Z);
    assert_eq!(got.len(), 1);
    assert!(got[0].closed);
    let (len, area) = totals(got.iter().map(|c| c.edges.as_slice()), o, Vec3::Z);
    close(len, TAU * 2.0, 1e-9);
    // The sheet faces out, so its cut runs as the solid's would.
    close(area, PI * 4.0, 1e-9);
    let got = measure::slice(&s, Point3::new(0.5, 0.0, 0.0), Vec3::X).unwrap();
    chained(&got, Point3::new(0.5, 0.0, 0.0), Vec3::X);
    assert_eq!(got.len(), 2);
    for c in &got {
        assert!(!c.closed);
        let (len, _) = totals([c.edges.as_slice()], Point3::ZERO, Vec3::X);
        close(len, 10.0, 1e-9);
    }
    // A block's top and one side, joined along an edge: the cut is one open chain bent there,
    // 4 across the top and 3 down the side.
    let blk = primitive::block(&Frame::WORLD, Vec3::new(4.0, 5.0, 3.0), "b").unwrap();
    let top = measure::face_at(&blk, Vec3::new(2.0, 2.5, 3.0)).unwrap().0;
    let front = measure::face_at(&blk, Vec3::new(2.0, 0.0, 1.5)).unwrap().0;
    let s = sheet::from_faces(&blk, &[top, front]).unwrap();
    let o = Point3::new(1.0, 0.0, 0.0);
    let got = measure::slice(&s, o, Vec3::X).unwrap();
    chained(&got, o, Vec3::X);
    assert_eq!(got.len(), 1, "{got:#?}");
    assert!(!got[0].closed);
    assert_eq!(got[0].edges.len(), 2);
    let (len, _) = totals([got[0].edges.as_slice()], o, Vec3::X);
    close(len, 8.0, 1e-9);
}

#[test]
fn an_ellipsoid_is_cut_in_ellipses() {
    // Half axes 3, 4, 5 along x, y, z. At z = 0 the cut is the ellipse of half axes 3 and 4, area
    // 12π; at z = 2 the same ellipse scaled by √(1 − 4/25) = √21/5. Every point lies on it. The
    // boolean section is refused at z = 0 (#94), so only the closed form is checked.
    let e = primitive::ellipsoid(&Frame::WORLD, Vec3::new(3.0, 4.0, 5.0), "ell").unwrap();
    for (z, k) in [(0.0f64, 1.0f64), (2.0, (21.0f64).sqrt() / 5.0)] {
        let o = Point3::new(0.0, 0.0, z);
        let got = measure::slice(&e, o, Vec3::Z).unwrap();
        chained(&got, o, Vec3::Z);
        assert_eq!(got.len(), 1, "{got:#?}");
        assert!(got[0].closed);
        let (_, area) = totals([got[0].edges.as_slice()], o, Vec3::Z);
        close(area, 12.0 * PI * k * k, 1e-6);
        for s in &got[0].edges {
            for i in 0..=16 {
                let t = s.range.0 + (s.range.1 - s.range.0) * i as f64 / 16.0;
                let p = s.curve.eval(t);
                let q = (p.x / (3.0 * k)).powi(2) + (p.y / (4.0 * k)).powi(2);
                assert!((q - 1.0).abs() < 1e-6, "{p:?} off the ellipse: {q}");
            }
        }
    }
}

#[test]
fn a_loft_through_equal_circles_is_cut_in_a_circle() {
    // Circles of radius 4 at heights 0, 4 and 10 skin a cylinder: at z = 7 a circle, area 16π. The
    // boolean section is refused there (#94), so only the closed form is checked.
    let sections: Vec<Section> = [0.0, 4.0, 10.0]
        .iter()
        .map(|&z| Section::new(Profile::circle(Vec2::ZERO, 8.0), at(z)))
        .collect();
    let b = loft(&sections, false, "lf").unwrap();
    let o = Point3::new(0.0, 0.0, 7.0);
    let got = measure::slice(&b, o, Vec3::Z).unwrap();
    chained(&got, o, Vec3::Z);
    assert_eq!(got.len(), 1);
    let (len, area) = totals([got[0].edges.as_slice()], o, Vec3::Z);
    close(len, TAU * 4.0, 1e-6);
    close(area, PI * 16.0, 1e-6);
}

/// The points of a curve at 16 steps along each of its pieces.
fn points(c: &SliceCurve) -> Vec<Point3> {
    c.edges
        .iter()
        .flat_map(|e| {
            (0..=16).map(move |k| {
                e.curve
                    .eval(e.range.0 + (e.range.1 - e.range.0) * k as f64 / 16.0)
            })
        })
        .collect()
}

/// The solid under `z = 3 + (x − 1)² + (y − 1)²` over the square [0, 2]²: its top is the
/// quadratic Bézier patch with heights 5 3 5 / 3 1 3 / 5 3 5 at the Greville points 0, ½, 1 of
/// each parameter, lowest, 3, at the square's middle, and at least 4 on its boundary.
fn dimpled() -> Brep {
    let k = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let h = vec![
        vec![5.0, 3.0, 5.0],
        vec![3.0, 1.0, 3.0],
        vec![5.0, 3.0, 5.0],
    ];
    primitive::slab(&Frame::WORLD, [2.0, 2.0], (&k, &k), &h, "dimple").unwrap()
}

#[test]
fn a_dimple_in_a_spline_top_is_cut_in_a_circle_that_touches_no_edge() {
    // Between heights 3 and 4 the plane cuts the top in the circle of radius √c about (1, 1), a
    // loop inside the face touching none of its edges, round the point where the top's height is
    // least: a hole in the 2 × 2 square, area 4 − πc. With c = 10⁻⁶ the loop is 2 µm across: no
    // point of a grid over the face's parameters need lie on or near it, but the iso-parametric
    // lines through the critical point cross it.
    let b = dimpled();
    let top: Vec<_> = faces_named(&b, "dimple.top").into_iter().collect();
    assert_eq!(top.len(), 1);
    let sheet = sheet::from_faces(&b, &top).unwrap();
    for c in [0.25f64, 1e-6] {
        let o = Point3::new(0.0, 0.0, 3.0 + c);
        let got = measure::slice(&b, o, Vec3::Z).unwrap();
        chained(&got, o, Vec3::Z);
        assert_eq!(got.len(), 2, "c = {c}: {got:#?}");
        assert!(got.iter().all(|cv| cv.closed));
        let (len, area) = totals(got.iter().map(|cv| cv.edges.as_slice()), o, Vec3::Z);
        close(len, 8.0 + TAU * c.sqrt(), 1e-6);
        close(area, 4.0 - PI * c, 1e-6);
        let hole = got
            .iter()
            .find(|cv| points(cv).iter().all(|p| p.x > 0.25 && p.x < 1.75))
            .expect("the hole");
        for p in points(hole) {
            let r = (p.x - 1.0).hypot(p.y - 1.0);
            assert!((r - c.sqrt()).abs() < 1e-7, "c = {c}: radius {r}");
        }
        // Run clockwise about the normal, as a hole does.
        let (_, a) = totals([hole.edges.as_slice()], o, Vec3::Z);
        assert!((a + PI * c).abs() <= 1e-6 * PI * c, "c = {c}: {a}");
        // The top alone, as a sheet, is cut in the same circle.
        let got = measure::slice(&sheet, o, Vec3::Z).unwrap();
        chained(&got, o, Vec3::Z);
        assert_eq!(got.len(), 1, "c = {c}: {got:#?}");
        assert!(got[0].closed);
        let (len, _) = totals([got[0].edges.as_slice()], o, Vec3::Z);
        close(len, TAU * c.sqrt(), 1e-6);
    }
}

#[test]
fn a_plane_near_a_crest_cuts_a_cap_that_touches_no_edge() {
    // Each surface's one face has its seam at the frame's x axis and its poles on z; the planes
    // stand square to a normal toward (−1, −1, 1), so each cap lies clear of both.
    let n = Vec3::new(-1.0, -1.0, 1.0).unit().unwrap();
    // A sphere of radius 5 at distance d from its centre: a circle of radius √(25 − d²).
    let s = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    for d in [4.5f64, 4.999] {
        let o = n * d;
        let (got, _, area) = solid_slice(&s, o, n);
        assert_eq!(got.len(), 1);
        close(area, PI * (25.0 - d * d), 1e-9);
    }
    // The ellipsoid x²/9 + y²/16 + z²/25 = 1, a rational spline face: the plane n·x = d cuts it
    // in an ellipse of area πabc(1 − d²/m²)/m, m = √(a²n₁² + b²n₂² + c²n₃²) the greatest n·x on it.
    let e = primitive::ellipsoid(&Frame::WORLD, Vec3::new(3.0, 4.0, 5.0), "ell").unwrap();
    let m = (9.0 * n.x * n.x + 16.0 * n.y * n.y + 25.0 * n.z * n.z).sqrt();
    for k in [0.9f64, 0.9999] {
        let d = k * m;
        let o = n * d;
        let got = measure::slice(&e, o, n).unwrap();
        chained(&got, o, n);
        assert_eq!(got.len(), 1, "{got:#?}");
        assert!(got[0].closed);
        let (_, area) = totals([got[0].edges.as_slice()], o, n);
        close(area, PI * 60.0 * (1.0 - k * k) / m, 1e-6);
        for p in points(&got[0]) {
            let q = (p.x / 3.0).powi(2) + (p.y / 4.0).powi(2) + (p.z / 5.0).powi(2);
            assert!((q - 1.0).abs() < 1e-7, "{p:?} off the ellipsoid: {q}");
        }
    }
}

#[test]
fn a_torus_near_its_crest_is_cut_in_one_loop_that_touches_no_edge() {
    // A torus (5, 1), its seams the tube's circle at u = 0 and the outer equator. A normal at 45°
    // about the axis and 40° above the equator has its crest at that angle and latitude, at height
    // 5 cos 40° + 1; a plane 0.1 below it cuts one small loop about it on the outer side, clear of
    // both seams. Every point of it lies on the torus, and its area is the boolean section's, and
    // near 2π·0.1/√K, the cut of the osculating paraboloid there, K = cos v / (r(R + r cos v)) the
    // Gaussian curvature.
    let t = primitive::torus(&Frame::WORLD, 5.0, 1.0, "ring").unwrap();
    let (a, b) = (45f64.to_radians(), 40f64.to_radians());
    let n = Vec3::new(a.cos() * b.cos(), a.sin() * b.cos(), b.sin());
    let o = n * (5.0 * b.cos() + 1.0 - 0.1);
    let (got, _, area) = solid_slice(&t, o, n);
    assert_eq!(got.len(), 1);
    let k = b.cos() / (5.0 + b.cos());
    let near = TAU * 0.1 / k.sqrt();
    assert!((area - near).abs() < 0.05 * near, "{area} against {near}");
    for p in points(&got[0]) {
        let f = (p.x.hypot(p.y) - 5.0).powi(2) + p.z * p.z - 1.0;
        assert!(f.abs() < 1e-7, "{p:?} off the torus: {f}");
    }
}

#[test]
fn faces_that_pinch_at_a_point_are_cut_in_an_outline_round_each() {
    // Two unit cubes meeting only along the edge x = y = 0, in one arena: a plane across the edge
    // cuts two unit squares that touch at one corner. Four pieces meet there; each square is its
    // own closed outline of area 1 and length 4, not one figure of eight through the corner.
    let mut b = primitive::block(&Frame::WORLD, Vec3::new(1.0, 1.0, 1.0), "a").unwrap();
    let other = primitive::block(
        &Frame::new(Vec3::new(-1.0, -1.0, 0.0), Vec3::Z, Vec3::X).unwrap(),
        Vec3::new(1.0, 1.0, 1.0),
        "b",
    )
    .unwrap();
    b.append(&other);
    for z in [0.5, 0.25] {
        let o = Point3::new(0.0, 0.0, z);
        let got = measure::slice(&b, o, Vec3::Z).unwrap();
        chained(&got, o, Vec3::Z);
        assert_eq!(got.len(), 2, "{got:#?}");
        for c in &got {
            assert!(c.closed);
            let (len, area) = totals([c.edges.as_slice()], o, Vec3::Z);
            close(len, 4.0, 1e-9);
            close(area, 1.0, 1e-9);
            let ps = points(c);
            let side = ps.iter().map(|p| p.x + p.y).sum::<f64>().signum();
            assert!(ps.iter().all(|p| (p.x + p.y) * side >= -1e-9), "{ps:?}");
        }
    }
    // A horn torus pinches to a point at its centre: a plane through its axis cuts two circles of
    // radius 3 that touch there, each its own outline of area 9π.
    let h = primitive::horn_torus(&Frame::WORLD, 3.0, "horn").unwrap();
    let a = 30f64.to_radians();
    let n = Vec3::new(-a.sin(), a.cos(), 0.0);
    let got = measure::slice(&h, Point3::ZERO, n).unwrap();
    chained(&got, Point3::ZERO, n);
    assert_eq!(got.len(), 2, "{got:#?}");
    let along = Vec3::new(a.cos(), a.sin(), 0.0);
    for c in &got {
        assert!(c.closed);
        let (len, area) = totals([c.edges.as_slice()], Point3::ZERO, n);
        close(len, TAU * 3.0, 1e-9);
        close(area, 9.0 * PI, 1e-9);
        let ps = points(c);
        let side = ps.iter().map(|p| p.dot(along)).sum::<f64>().signum();
        assert!(ps.iter().all(|p| p.dot(along) * side >= -1e-9));
    }
}

/// A 10 × 6 × 3 block with one upright edge rounded to radius 1, written as another system's STEP
/// with the round's radius grown by 0.2 and read back: the round's edges stay where they were, on
/// the sides, so they miss the round by 0.2 and carry a local tolerance of 0.3.
fn tolerant_block() -> Brep {
    use springrep::{
        profile::{NamedSegment, ProfileLoop, Segment},
        step,
    };
    let (w, h, r) = (10.0, 6.0, 1.0);
    let seg = |segment: Segment, name: &str| NamedSegment {
        segment,
        names: [name.to_string()].into(),
    };
    let line = |a: (f64, f64), b: (f64, f64), name: &str| {
        seg(
            Segment::Line {
                a: Vec2::new(a.0, a.1),
                b: Vec2::new(b.0, b.1),
            },
            name,
        )
    };
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                line((0.0, 0.0), (w, 0.0), "s"),
                line((w, 0.0), (w, h - r), "e"),
                seg(
                    Segment::Arc {
                        center: Vec2::new(w - r, h - r),
                        radius: r,
                        start: 0.0,
                        sweep: PI / 2.0,
                    },
                    "round",
                ),
                line((w - r, h), (0.0, h), "n"),
                line((0.0, h), (0.0, 0.0), "w"),
            ],
        }],
    };
    let b = extrude(&p, &Frame::WORLD, 3.0, "rb").unwrap();
    let text = step::write(&b, "rb")
        .unwrap()
        .replace("'springrep'", "'other'");
    let grown: Vec<String> = text
        .lines()
        .map(|l| match l.find("=CYLINDRICAL_SURFACE(") {
            Some(_) => {
                let (head, r) = l.rsplit_once(',').unwrap();
                let r: f64 = r.trim_end_matches(");").parse().unwrap();
                format!("{head},{:?});", r + 0.2)
            }
            None => l.to_string(),
        })
        .collect();
    let text = grown.join("\n");
    let head = "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(";
    let at = text.find(head).unwrap() + head.len();
    let end = at + text[at..].find(')').unwrap();
    let text = format!("{}0.4{}", &text[..at], &text[end..]);
    let imp = step::read(text.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    imp.brep
}

#[test]
fn pieces_join_within_the_tolerance_of_the_edges_they_end_at() {
    // Across the block, the side faces' cuts end on the round's edges, at (10, 5) and (9, 6), and
    // the round's own cut, the circle of radius 1.2 about (9, 5), ends 0.2 from them, at (10.2, 5)
    // and (9, 6.2): within the edges' 0.3, so the outline closes, its five pieces each beginning
    // where the last ends or 0.2 from it. With those two gaps bridged by straight lines it bounds
    // the 10 × 6 rectangle less the unit corner square, with the quarter disk of radius 1.2: area
    // 59 + 0.36π.
    let b = tolerant_block();
    let (_, es) = b.tolerant();
    assert_eq!(es.len(), 4);
    for z in [1.5, 0.7] {
        let o = Point3::new(0.0, 0.0, z);
        let got = measure::slice(&b, o, Vec3::Z).unwrap();
        assert_eq!(got.len(), 1, "{got:#?}");
        let c = &got[0];
        assert!(c.closed);
        assert_eq!(c.edges.len(), 5);
        let n = c.edges.len();
        let mut gaps = Vec::new();
        let (_, mut area) = totals([c.edges.as_slice()], o, Vec3::Z);
        for k in 0..n {
            let (a, b) = (ends(&c.edges[k]).1, ends(&c.edges[(k + 1) % n]).0);
            gaps.push(a.dist(b));
            area += 0.5 * (a - o).cross(b - a).dot(Vec3::Z);
        }
        gaps.sort_by(f64::total_cmp);
        for (got, want) in gaps.iter().zip([0.0, 0.0, 0.0, 0.2, 0.2]) {
            assert!((got - want).abs() < 1e-9, "{gaps:?}");
        }
        close(area, 59.0 + 0.36 * PI, 1e-9);
    }
}

#[test]
fn a_plane_that_only_touches_a_face_cuts_nothing() {
    // The ellipsoid x²/9 + y²/16 + z²/25 = 1, a rational spline face, touched by the planes square
    // to each axis at its extremes there, the poles and the equator's four ends among them, and by
    // the plane n·x = m along a normal toward (−1, −1, 1), m = √(a²n₁² + b²n₂² + c²n₃²): each meets
    // it at one point and cuts nothing. A hair inside each, the cut is the ellipse of area
    // πabc(1 − k²)/m at n·x = km.
    let e = primitive::ellipsoid(&Frame::WORLD, Vec3::new(3.0, 4.0, 5.0), "ell").unwrap();
    let s = primitive::sphere(&Frame::WORLD, 5.0, "ball").unwrap();
    let r = [3.0, 4.0, 5.0];
    let mut normals: Vec<Vec3> = [Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .flat_map(|a| [a, -a])
        .collect();
    normals.push(Vec3::new(-1.0, -1.0, 1.0).unit().unwrap());
    for n in normals {
        let m =
            (r[0] * r[0] * n.x * n.x + r[1] * r[1] * n.y * n.y + r[2] * r[2] * n.z * n.z).sqrt();
        let got = measure::slice(&e, n * m, n).unwrap();
        assert!(got.is_empty(), "ellipsoid touched along {n:?}: {got:#?}");
        let got = measure::slice(&s, n * 5.0, n).unwrap();
        assert!(got.is_empty(), "sphere touched along {n:?}: {got:#?}");
        let k = 1.0 - 1e-4;
        let o = n * (k * m);
        let got = measure::slice(&e, o, n).unwrap();
        chained(&got, o, n);
        assert_eq!(got.len(), 1, "{n:?}: {got:#?}");
        assert!(got[0].closed);
        let (_, area) = totals([got[0].edges.as_slice()], o, n);
        close(area, PI * 60.0 * (1.0 - k * k) / m, 1e-5);
    }
    // The dimpled slab touched at the bottom of its dimple, z = 3: only the square outline.
    let b = dimpled();
    let o = Point3::new(0.0, 0.0, 3.0);
    let got = measure::slice(&b, o, Vec3::Z).unwrap();
    chained(&got, o, Vec3::Z);
    assert_eq!(got.len(), 1, "{got:#?}");
    let (len, area) = totals([got[0].edges.as_slice()], o, Vec3::Z);
    close(len, 8.0, 1e-9);
    close(area, 4.0, 1e-9);
}
