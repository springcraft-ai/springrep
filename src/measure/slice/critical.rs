//! Where to start tracing a plane's cut of a face so that no closed loop of it is missed.
//!
//! Let `h = (S(u, v) − o) · n` be the height of a face's surface above the plane. A closed loop of
//! the cut that touches no edge of the face is a level curve `h = 0` inside the face, and in the
//! parameter plane it is either a Jordan curve or it winds round a periodic parameter.
//!
//! - A Jordan curve bounds a disk in the face's parameter box, over which the surface is defined
//!   even where the face has holes. `h` is zero on its boundary and not zero throughout, so it has
//!   an interior extremum, which is a point where both partial derivatives of `h` vanish: a
//!   critical point of the height function (Morse theory; J. Milnor, *Morse Theory*, 1963, §2).
//!   An iso-parametric line through that point runs out of the disk to the box's side, so by the
//!   Jordan curve theorem it crosses the loop.
//! - A loop that winds round a periodic parameter crosses every iso-parametric line across that
//!   parameter, the seam among them.
//! - Where the surface may kink, along a spline's knot line of full multiplicity or a profile's
//!   breakpoint, an extremum can sit on the kink without a zero of the gradient; the line of the
//!   kink is itself an iso-parametric line through it.
//!
//! So the plane's meetings with the iso-parametric lines through every critical point, with the
//! seams and with the kinks lie on every such loop: a trace seeded from them finds it. On a ruled surface (a
//! plane, a cylinder, a cone, an extrusion, a ruled face) `h` is affine along each ruling, and a
//! ruling through an interior extremum would carry it to the loop, where `h` is zero: there is no
//! such loop, and only the seams are needed.
//!
//! The critical points are found in closed form on the sphere, the torus and a surface of
//! revolution. On a spline surface they are the common zeros of two polynomials in Bernstein form,
//! found by subdividing each Bézier patch (de Casteljau) and keeping only the pieces where both
//! polynomials' control values change sign, since a polynomial in Bernstein form lies within the
//! convex hull of its control values (L. Piegl and W. Tiller, *The NURBS Book*, 2nd ed., 1997,
//! §1.3 and §5.2). A piece over which the Jacobian's interval determinant keeps one sign holds at
//! most one zero, which Newton's method then finds (interval root isolation; R. E. Moore,
//! *Interval Analysis*, 1966). A spline line's meetings with the plane are isolated the same way,
//! by subdivision until a piece's control values change sign once, so that it holds exactly one
//! root (Descartes' rule of signs for the Bernstein basis; J. M. Lane and R. F. Riesenfeld, "Bounds
//! on a polynomial", BIT 21, 1981).

use std::f64::consts::{PI, TAU};

use crate::{
    boolean::{csi, domain::Domain},
    geom::{Circle, Curve, Line, NurbsSurface, Surface},
    math::{Frame, Point3, Transform, Vec2, Vec3},
    tol,
};

/// A curve on a face's surface that every closed loop of a cut it must be seeded for crosses.
pub(super) enum Crosser {
    /// A circle or a line, met with the plane in closed form, or a curve of the surface's.
    Curve(Curve, (f64, f64)),
    /// A spline surface's iso-parametric line: in each Bézier patch it crosses, its control points
    /// in homogeneous coordinates.
    Spline(Vec<Vec<[f64; 4]>>),
}

/// The crossers of the face with domain `dom` for planes square to the unit `n`: the iso-parametric
/// lines through each critical point of the height along `n`, the seams and the kinks. They depend
/// on the direction only, so parallel planes share them.
pub(super) fn crossers(dom: &Domain, n: Vec3) -> Vec<Crosser> {
    let s = &dom.surface;
    // The domain is the box of the boundary's sampled images: widened a tenth, as the face's box is.
    let pad = (dom.hi - dom.lo) * 0.1;
    let (lo, hi) = (dom.lo - pad, dom.hi + pad);
    let mut out = Vec::new();
    let circle = |c: Point3, z: Vec3, r: f64| {
        let frame = Frame::new(c, z, z.any_perpendicular()).ok()?;
        (r > tol::LINEAR)
            .then(|| Crosser::Curve(Curve::Circle(Circle { frame, radius: r }), (0.0, TAU)))
    };
    let line = |p: Point3, d: Vec3, (t0, t1): (f64, f64)| {
        Some(Crosser::Curve(
            Curve::Line(Line {
                origin: p,
                dir: d.unit()?,
            }),
            (t0, t1),
        ))
    };
    match s {
        Surface::Plane(_) | Surface::Ruled(_) => {}
        Surface::Cylinder(c) => {
            let f = &c.frame;
            out.extend(line(
                f.origin + f.radial(lo.x) * c.radius,
                f.z,
                (lo.y, hi.y),
            ));
        }
        Surface::Cone(_) => {
            let (p, _, sv) = s.d1(lo.x, 0.0);
            out.extend(line(p, sv, (lo.y, hi.y)));
        }
        Surface::Extrusion(e) => {
            if e.curve.period().is_some() {
                out.extend(line(e.curve.eval(lo.x), e.dir, (lo.y, hi.y)));
            }
        }
        Surface::Sphere(sp) => {
            let f = &sp.frame;
            // The two points where the normal is ±n, each on its circle of latitude.
            for d in [n, -n] {
                let p = f.origin + d * sp.radius;
                let z = (p - f.origin).dot(f.z);
                let rho = (sp.radius * sp.radius - z * z).max(0.0).sqrt();
                out.extend(circle(f.origin + f.z * z, f.z, rho));
            }
            // The seam: the great circle through the poles at u = lo.
            let e = f.radial(lo.x);
            if let Ok(frame) = Frame::new(f.origin, e.cross(f.z), e) {
                out.push(Crosser::Curve(
                    Curve::Circle(Circle {
                        frame,
                        radius: sp.radius,
                    }),
                    (0.0, TAU),
                ));
            }
        }
        Surface::Torus(t) => {
            let f = &t.frame;
            let l = f.dir_to_local(n);
            // Where the normal is ±n: in the meridians at the angle of n's part across the axis
            // and opposite it, at the latitudes where the tube circle's tangent is square to n.
            if l.x.hypot(l.y) > tol::ANGULAR {
                let u0 = l.y.atan2(l.x);
                for u in [u0, u0 + PI] {
                    let e = f.radial(u);
                    let v0 = n.dot(f.z).atan2(n.dot(e));
                    for v in [v0, v0 + PI] {
                        let (sv, cv) = v.sin_cos();
                        out.extend(circle(
                            f.origin + f.z * (t.minor * sv),
                            f.z,
                            t.major + t.minor * cv,
                        ));
                    }
                }
            }
            // The seams: the tube's circle at u = lo, and the circle about the axis at v = lo.
            let e = f.radial(lo.x);
            out.extend(circle(f.origin + e * t.major, e.cross(f.z), t.minor));
            let (sv, cv) = lo.y.sin_cos();
            out.extend(circle(
                f.origin + f.z * (t.minor * sv),
                f.z,
                t.major + t.minor * cv,
            ));
        }
        Surface::Revolution(r) => {
            let a = r.axis;
            let (d0, d1) = r.curve.domain();
            let (v0, v1) = if r.curve.period().is_some() {
                (lo.y, hi.y)
            } else {
                (lo.y.max(d0), hi.y.min(d1))
            };
            let ring = |v: f64| {
                let q = r.curve.eval(v) - r.origin;
                let z = q.dot(a);
                circle(r.origin + a * z, a, (q - a * z).norm())
            };
            if v1 > v0 {
                // And where the profile may kink, at its breakpoints, the ring through the kink.
                for v in revolution_critical(r, n, v0, v1)
                    .into_iter()
                    .chain(r.curve.breakpoints(v0, v1))
                {
                    out.extend(ring(v));
                }
                // The seams: the profile at u = lo, and the circle about the axis at v = lo.
                let rot = Transform::rotation(r.origin, a, lo.x);
                out.push(Crosser::Curve(r.curve.transformed(&rot), (v0, v1)));
                if r.curve.period().is_some() {
                    out.extend(ring(lo.y));
                }
            }
        }
        Surface::Nurbs(ns) => {
            let mut held: Vec<(bool, f64)> = Vec::new();
            for uv in spline_critical(ns, n, lo, hi) {
                held.push((true, uv.x));
                held.push((false, uv.y));
            }
            // Where a patch meets the next with a kink, an extremum can sit on the knot line
            // without a zero of the gradient: the knot line itself runs out across the loop.
            let (pu, pv) = ns.degrees();
            let (ku, kv) = ns.knots();
            held.extend(creases(ku, pu).into_iter().map(|u| (true, u)));
            held.extend(creases(kv, pv).into_iter().map(|v| (false, v)));
            if ns.u_period().is_some() {
                held.push((true, lo.x));
            }
            if ns.v_period().is_some() {
                held.push((false, lo.y));
            }
            held.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
            held.dedup_by(|a, b| a.0 == b.0 && (a.1 - b.1).abs() <= 1e-12);
            for (along_v, at) in held {
                out.push(Crosser::Spline(iso_nets(ns, along_v, at)));
            }
        }
    }
    out
}

/// The points where the plane through `o` square to unit `n` crosses the crossers.
pub(super) fn crossings(cs: &[Crosser], o: Point3, n: Vec3, plane: &Surface) -> Vec<Point3> {
    let mut out = Vec::new();
    for c in cs {
        match c {
            Crosser::Curve(curve, range) => {
                let hits = csi::curve_surface(curve, *range, plane);
                if !hits.on_surface {
                    out.extend(hits.ts.iter().map(|&t| curve.eval(t)));
                }
            }
            Crosser::Spline(nets) => {
                let off = o.dot(n);
                for net in nets {
                    let heights: Vec<f64> = net
                        .iter()
                        .map(|q| n.x * q[0] + n.y * q[1] + n.z * q[2] - off * q[3])
                        .collect();
                    for t in bernstein_roots(&heights) {
                        let q = casteljau(net, t);
                        out.push(Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3]));
                    }
                }
            }
        }
    }
    out
}

/// The profile parameters in (v0, v1) at which a surface of revolution has a critical point of
/// the height along unit `n`. Turning by `u` moves only a point's angle about the axis, so `∂h/∂u`
/// vanishes where the point lies in the half-planes through the axis toward and away from `n`'s
/// part across it, `n⊥`. There, with `ρ` and `z` the profile's distance from the axis and height
/// along it, `∂h/∂v = ±ρ′|n⊥| + z′(n·a)`, which for a profile in a half-plane through the axis
/// along `e` is its rate along `±|n⊥|e + (n·a)a`: its extremes along that direction.
fn revolution_critical(r: &crate::geom::Revolution, n: Vec3, v0: f64, v1: f64) -> Vec<f64> {
    let a = r.axis;
    let na = n.dot(a);
    let across = (n - a * na).norm();
    // The half-plane the profile lies in, from its point furthest from the axis among a few.
    let q = (0..=8)
        .map(|k| r.curve.eval(v0 + (v1 - v0) * k as f64 / 8.0) - r.origin)
        .map(|q| q - a * q.dot(a))
        .max_by(|x, y| x.norm().total_cmp(&y.norm()))
        .unwrap_or(Vec3::ZERO);
    let Some(e) = q.unit() else {
        return Vec::new();
    };
    let side = a.cross(e);
    let planar = match &r.curve {
        Curve::Line(l) => {
            (l.origin - r.origin).dot(side).abs() <= tol::LINEAR
                && l.dir.dot(side).abs() <= tol::ANGULAR
        }
        Curve::Circle(c) => {
            (c.frame.origin - r.origin).dot(side).abs() <= tol::LINEAR
                && c.frame.z.cross(side).norm() <= tol::ANGULAR
        }
        Curve::Ellipse(c) => {
            (c.frame.origin - r.origin).dot(side).abs() <= tol::LINEAR
                && c.frame.z.cross(side).norm() <= tol::ANGULAR
        }
        Curve::BSpline(b) => b
            .control_points()
            .iter()
            .all(|&p| (p - r.origin).dot(side).abs() <= tol::LINEAR),
        Curve::Nurbs(b) => b
            .control_points()
            .iter()
            .all(|&p| (p - r.origin).dot(side).abs() <= tol::LINEAR),
        Curve::Intersection(_) => false,
    };
    let mut out = Vec::new();
    if planar {
        for sign in [1.0, -1.0] {
            let d = e * (sign * across) + a * na;
            if d.norm() > 0.0 {
                out.extend(r.curve.extremes(d, v0, v1));
            }
        }
    } else {
        // A profile off any one half-plane: ρ′ = q⊥·q⊥′/|q⊥|, solved span by span.
        let mut cuts = vec![v0];
        cuts.extend(r.curve.breakpoints(v0, v1));
        cuts.push(v1);
        for sign in [1.0, -1.0] {
            let mut g = |v: f64| {
                let (p, d) = r.curve.d1(v);
                let q = p - r.origin;
                let qp = q - a * q.dot(a);
                let dp = d - a * d.dot(a);
                let rho = qp.norm().max(1e-300);
                sign * across * qp.dot(dp) / rho + na * d.dot(a)
            };
            for w in cuts.windows(2) {
                out.extend(crate::numeric::roots(&mut g, w[0], w[1], 32));
            }
        }
    }
    out
}

/// A polynomial in tensor-product Bernstein form over its own unit square: `rows` coefficients
/// along u by `cols` along v.
#[derive(Clone, Debug)]
struct Net {
    rows: usize,
    cols: usize,
    c: Vec<f64>,
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |b, i| b * (n - i) as f64 / (i + 1) as f64)
}

impl Net {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.c[i * self.cols + j]
    }

    /// The least and greatest control value, between which the polynomial lies.
    fn range(&self) -> (f64, f64) {
        self.c
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            })
    }

    fn clear_of_zero(&self) -> bool {
        let (lo, hi) = self.range();
        lo > 0.0 || hi < 0.0
    }

    /// The derivative across u, or across v: the steps between neighbouring control values times
    /// the degree (Piegl and Tiller, eq. 1.9).
    fn derivative(&self, across_u: bool) -> Net {
        let (rows, cols) = if across_u {
            (self.rows.saturating_sub(1).max(1), self.cols)
        } else {
            (self.rows, self.cols.saturating_sub(1).max(1))
        };
        let degree = if across_u { self.rows } else { self.cols } as f64 - 1.0;
        let c = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .map(|(i, j)| {
                if degree < 1.0 {
                    0.0
                } else if across_u {
                    degree * (self.at(i + 1, j) - self.at(i, j))
                } else {
                    degree * (self.at(i, j + 1) - self.at(i, j))
                }
            })
            .collect();
        Net { rows, cols, c }
    }

    /// The product of two polynomials, in Bernstein form of the summed degrees.
    fn times(&self, o: &Net) -> Net {
        let (m1, n1, m2, n2) = (self.rows - 1, self.cols - 1, o.rows - 1, o.cols - 1);
        let (rows, cols) = (m1 + m2 + 1, n1 + n2 + 1);
        let mut c = vec![0.0; rows * cols];
        for i1 in 0..=m1 {
            for j1 in 0..=n1 {
                let a = self.at(i1, j1) * binomial(m1, i1) * binomial(n1, j1);
                for i2 in 0..=m2 {
                    for j2 in 0..=n2 {
                        c[(i1 + i2) * cols + j1 + j2] +=
                            a * o.at(i2, j2) * binomial(m2, i2) * binomial(n2, j2);
                    }
                }
            }
        }
        for i in 0..rows {
            for j in 0..cols {
                c[i * cols + j] /= binomial(m1 + m2, i) * binomial(n1 + n2, j);
            }
        }
        Net { rows, cols, c }
    }

    fn minus(&self, o: &Net) -> Net {
        Net {
            rows: self.rows,
            cols: self.cols,
            c: self.c.iter().zip(&o.c).map(|(a, b)| a - b).collect(),
        }
    }

    /// The two halves, cut across u or across v at the middle (de Casteljau).
    fn halves(&self, across_u: bool) -> [Net; 2] {
        let (lines, len) = if across_u {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        let idx = |line: usize, k: usize| {
            if across_u {
                k * self.cols + line
            } else {
                line * self.cols + k
            }
        };
        let (mut first, mut second) = (self.c.clone(), self.c.clone());
        for line in 0..lines {
            let mut level: Vec<f64> = (0..len).map(|k| self.c[idx(line, k)]).collect();
            for step in 0..len {
                first[idx(line, step)] = level[0];
                second[idx(line, len - 1 - step)] = level[len - 1 - step];
                for k in 0..len - 1 - step {
                    level[k] = 0.5 * (level[k] + level[k + 1]);
                }
            }
        }
        let net = |c| Net {
            rows: self.rows,
            cols: self.cols,
            c,
        };
        [net(first), net(second)]
    }

    fn eval(&self, s: f64, t: f64) -> f64 {
        let along: Vec<f64> = (0..self.rows)
            .map(|i| scalar_casteljau(&self.c[i * self.cols..(i + 1) * self.cols], t))
            .collect();
        scalar_casteljau(&along, s)
    }
}

fn scalar_casteljau(c: &[f64], t: f64) -> f64 {
    let mut level = c.to_vec();
    for k in (1..level.len()).rev() {
        for i in 0..k {
            level[i] = (1.0 - t) * level[i] + t * level[i + 1];
        }
    }
    level.first().copied().unwrap_or(0.0)
}

fn casteljau(c: &[[f64; 4]], t: f64) -> [f64; 4] {
    let mut level = c.to_vec();
    for k in (1..level.len()).rev() {
        for i in 0..k {
            level[i] = std::array::from_fn(|q| (1.0 - t) * level[i][q] + t * level[i + 1][q]);
        }
    }
    level[0]
}

/// Interval product.
fn mul((a, b): (f64, f64), (c, d): (f64, f64)) -> (f64, f64) {
    let p = [a * c, a * d, b * c, b * d];
    (
        p.iter().copied().fold(f64::INFINITY, f64::min),
        p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Halvings of a Bézier patch, alternately across u and v, before a piece that still may hold a
/// critical point is kept whole: a piece 2⁻¹² of the patch on a side.
const DEPTH: usize = 24;

/// Pieces looked at in one patch before the rest are kept whole.
const BUDGET: usize = 1 << 16;

/// The search of one Bézier patch for the common zeros of `gu` and `gv`.
struct Search {
    found: Vec<Vec2>,
    /// Pieces of the deepest level that may hold a zero no test could isolate, by their place on
    /// that level's grid: where the critical points are not isolated, as along a ridge at one
    /// height.
    cells: Vec<(i64, i64)>,
    /// Pieces kept whole when the budget ran out, by their middle.
    left: Vec<Vec2>,
    looked: usize,
}

impl Search {
    /// The piece of the patch's unit square at `(iu, iv)` on the grid of its level `depth`.
    fn piece(&mut self, gu: &Net, gv: &Net, depth: usize, iu: i64, iv: i64) {
        if gu.clear_of_zero() || gv.clear_of_zero() {
            return;
        }
        self.looked += 1;
        let (ku, kv) = (depth.div_ceil(2), depth / 2);
        let (wu, wv) = (0.5f64.powi(ku as i32), 0.5f64.powi(kv as i32));
        let corner = Vec2::new(iu as f64 * wu, iv as f64 * wv);
        if self.looked > BUDGET {
            self.left.push(corner + Vec2::new(0.5 * wu, 0.5 * wv));
            return;
        }
        let d = [
            gu.derivative(true),
            gu.derivative(false),
            gv.derivative(true),
            gv.derivative(false),
        ];
        // The mean value theorem puts the difference of (gu, gv) between two points of the piece
        // through a matrix whose entries lie in these ranges: if none of those matrices is
        // singular, two points cannot both be zeros.
        let r: Vec<(f64, f64)> = d.iter().map(Net::range).collect();
        let (p, q) = (mul(r[0], r[3]), mul(r[1], r[2]));
        let det = (p.0 - q.1, p.1 - q.0);
        if (det.0 > 0.0 || det.1 < 0.0)
            && let Some(st) = newton(gu, gv, &d)
        {
            self.found.push(corner + Vec2::new(st.x * wu, st.y * wv));
            return;
        }
        if depth >= DEPTH {
            self.cells.push((iu, iv));
            return;
        }
        let across_u = depth.is_multiple_of(2);
        let [gu0, gu1] = gu.halves(across_u);
        let [gv0, gv1] = gv.halves(across_u);
        let (a, b) = if across_u {
            ((2 * iu, iv), (2 * iu + 1, iv))
        } else {
            ((iu, 2 * iv), (iu, 2 * iv + 1))
        };
        self.piece(&gu0, &gv0, depth + 1, a.0, a.1);
        self.piece(&gu1, &gv1, depth + 1, b.0, b.1);
    }

    /// One point of each run of neighbouring cells, the cell nearest the run's middle: a critical
    /// set that is a curve has one height along it, so a loop of the cut holds all of it or none.
    fn cell_points(&self) -> Vec<Vec2> {
        let w = 0.5f64.powi((DEPTH / 2) as i32);
        let index: std::collections::HashMap<(i64, i64), usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(k, &c)| (c, k))
            .collect();
        let mut root: Vec<usize> = (0..self.cells.len()).collect();
        fn find(root: &mut [usize], mut k: usize) -> usize {
            while root[k] != k {
                root[k] = root[root[k]];
                k = root[k];
            }
            k
        }
        for (k, &(i, j)) in self.cells.iter().enumerate() {
            for di in -1..=1 {
                for dj in -1..=1 {
                    if let Some(&m) = index.get(&(i + di, j + dj)) {
                        let (a, b) = (find(&mut root, k), find(&mut root, m));
                        root[a] = b;
                    }
                }
            }
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for k in 0..self.cells.len() {
            let r = find(&mut root, k);
            groups.entry(r).or_default().push(k);
        }
        let mid = |c: (i64, i64)| Vec2::new((c.0 as f64 + 0.5) * w, (c.1 as f64 + 0.5) * w);
        groups
            .values()
            .map(|g| {
                let centre = g.iter().fold(Vec2::ZERO, |s, &k| s + mid(self.cells[k]))
                    * (1.0 / g.len() as f64);
                g.iter()
                    .map(|&k| mid(self.cells[k]))
                    .min_by(|a, b| (*a - centre).norm().total_cmp(&(*b - centre).norm()))
                    .unwrap_or(centre)
            })
            .collect()
    }
}

/// The zero of `(gu, gv)` in their piece's unit square by Newton's method from its middle, `d`
/// their derivatives across u and v: `None` unless it settles inside the piece.
fn newton(gu: &Net, gv: &Net, d: &[Net; 4]) -> Option<Vec2> {
    let (mut s, mut t) = (0.5, 0.5);
    for _ in 0..40 {
        let (f, g) = (gu.eval(s, t), gv.eval(s, t));
        let (a, b, c, e) = (
            d[0].eval(s, t),
            d[1].eval(s, t),
            d[2].eval(s, t),
            d[3].eval(s, t),
        );
        let (ds, dt) = crate::numeric::solve2(a, b, c, e, -f, -g)?;
        s += ds;
        t += dt;
        if !(-0.5..=1.5).contains(&s) || !(-0.5..=1.5).contains(&t) {
            return None;
        }
        if ds.abs() + dt.abs() <= 1e-14 {
            let m = 1e-9;
            return ((-m..=1.0 + m).contains(&s) && (-m..=1.0 + m).contains(&t))
                .then(|| Vec2::new(s.clamp(0.0, 1.0), t.clamp(0.0, 1.0)));
        }
    }
    None
}

/// The parameters in the box `lo`–`hi` where the height along unit `n` of the spline surface is
/// stationary. On each Bézier patch, with the surface `X/W` in homogeneous coordinates and
/// `N = X·n`, the height `N/W` is stationary where `W·N_u − N·W_u` and `W·N_v − N·W_v` both
/// vanish, as `W > 0`: two polynomials, each a product of Bernstein forms put back in that form
/// (R. T. Farouki and V. T. Rajan, "Algorithms for polynomials in Bernstein form", CAGD 5, 1988).
fn spline_critical(ns: &NurbsSurface, n: Vec3, lo: Vec2, hi: Vec2) -> Vec<Vec2> {
    let (pu, pv) = ns.degrees();
    let (rows, cols) = (pu + 1, pv + 1);
    let (per_u, per_v) = (ns.u_period().is_some(), ns.v_period().is_some());
    let mut out: Vec<Vec2> = Vec::new();
    for bz in ns.beziers() {
        if (!per_u && (bz.hi.x < lo.x || bz.lo.x > hi.x))
            || (!per_v && (bz.hi.y < lo.y || bz.lo.y > hi.y))
        {
            continue;
        }
        let pts = bz.points();
        if pts.len() != rows * cols {
            continue;
        }
        let net = |f: &dyn Fn(&[f64; 4]) -> f64| Net {
            rows,
            cols,
            c: pts.iter().map(f).collect(),
        };
        let nn = net(&|q| n.x * q[0] + n.y * q[1] + n.z * q[2]);
        let w = net(&|q| q[3]);
        let rational =
            w.c.iter()
                .any(|&x| (x - w.c[0]).abs() > 1e-14 * w.c[0].abs());
        let (gu, gv) = if rational {
            (
                w.times(&nn.derivative(true))
                    .minus(&nn.times(&w.derivative(true))),
                w.times(&nn.derivative(false))
                    .minus(&nn.times(&w.derivative(false))),
            )
        } else {
            (nn.derivative(true), nn.derivative(false))
        };
        // A patch level with the plane has no point where the height turns.
        let scale = nn.c.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0)
            * w.c.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let flat = |g: &Net| g.c.iter().all(|x| x.abs() <= 1e-13 * scale);
        if flat(&gu) && flat(&gv) {
            continue;
        }
        let mut search = Search {
            found: Vec::new(),
            cells: Vec::new(),
            left: Vec::new(),
            looked: 0,
        };
        search.piece(&gu, &gv, 0, 0, 0);
        let mut local = std::mem::take(&mut search.found);
        local.extend(search.cell_points());
        local.extend(search.left);
        let size = bz.hi - bz.lo;
        for st in local {
            let uv = Vec2::new(bz.lo.x + st.x * size.x, bz.lo.y + st.y * size.y);
            if !out
                .iter()
                .any(|q| (*q - uv).norm() <= 1e-12 * (1.0 + size.norm()))
            {
                out.push(uv);
            }
        }
    }
    out
}

/// The knots strictly inside the domain of a knot vector of degree `p` repeated at least `p`
/// times, where the spline need not have a continuous first derivative.
fn creases(knots: &[f64], p: usize) -> Vec<f64> {
    if knots.len() < 2 * p + 2 {
        return Vec::new();
    }
    let (a, b) = (knots[p], knots[knots.len() - 1 - p]);
    let mut out = Vec::new();
    let mut k = 0;
    while k < knots.len() {
        let m = knots[k..].iter().take_while(|&&x| x == knots[k]).count();
        if knots[k] > a && knots[k] < b && m >= p.max(1) {
            out.push(knots[k]);
        }
        k += m;
    }
    out
}

/// The spline surface's iso-parametric line with u held at `at` (along v), or v held there (along
/// u), as one Bézier curve per patch it crosses, in homogeneous coordinates.
fn iso_nets(ns: &NurbsSurface, along_v: bool, at: f64) -> Vec<Vec<[f64; 4]>> {
    let (pu, pv) = ns.degrees();
    let (rows, cols) = (pu + 1, pv + 1);
    let ((u0, _), (v0, _)) = ns.domain();
    let wrap = |x: f64, period: Option<f64>, lo: f64| match period {
        Some(p) => lo + (x - lo).rem_euclid(p),
        None => x,
    };
    let at = if along_v {
        wrap(at, ns.u_period(), u0)
    } else {
        wrap(at, ns.v_period(), v0)
    };
    let mut out = Vec::new();
    for bz in ns.beziers() {
        let pts = bz.points();
        if pts.len() != rows * cols {
            continue;
        }
        let (a, b) = if along_v {
            (bz.lo.x, bz.hi.x)
        } else {
            (bz.lo.y, bz.hi.y)
        };
        if at < a || at > b || b <= a {
            continue;
        }
        let s = (at - a) / (b - a);
        let line: Vec<[f64; 4]> = if along_v {
            (0..cols)
                .map(|j| {
                    let column: Vec<[f64; 4]> = (0..rows).map(|i| pts[i * cols + j]).collect();
                    casteljau(&column, s)
                })
                .collect()
        } else {
            (0..rows)
                .map(|i| casteljau(&pts[i * cols..(i + 1) * cols], s))
                .collect()
        };
        out.push(line);
    }
    out
}

/// The roots in [0, 1] of the polynomial with Bernstein coefficients `c`, each to full precision
/// where it is a crossing. A piece whose coefficients keep one sign holds no root, and one whose
/// coefficients change sign once, with ends of opposite signs, holds exactly one (the variation
/// diminishing property); any other piece is halved.
fn bernstein_roots(c: &[f64]) -> Vec<f64> {
    fn go(c: &[f64], a: f64, b: f64, depth: usize, out: &mut Vec<f64>) {
        if c.iter().all(|&x| x > 0.0) || c.iter().all(|&x| x < 0.0) {
            return;
        }
        let m = c.len() - 1;
        let mut changes = 0;
        let mut last = 0.0;
        for &x in c {
            if x != 0.0 {
                if last != 0.0 && (x < 0.0) != (last < 0.0) {
                    changes += 1;
                }
                last = x;
            }
        }
        if changes == 0 {
            // No crossing: at most the polynomial touches zero, at an end whose value is zero.
            if c[0] == 0.0 {
                out.push(a);
            }
            if c[m] == 0.0 {
                out.push(b);
            }
            return;
        }
        if changes == 1 && c[0] * c[m] < 0.0 {
            let mut f = |t: f64| scalar_casteljau(c, t);
            let t = crate::numeric::bisect(&mut f, 0.0, 1.0);
            out.push(a + (b - a) * t);
            return;
        }
        if depth >= 60 {
            out.push(0.5 * (a + b));
            return;
        }
        let net = Net {
            rows: 1,
            cols: c.len(),
            c: c.to_vec(),
        };
        let [l, r] = net.halves(false);
        let mid = 0.5 * (a + b);
        go(&l.c, a, mid, depth + 1, out);
        go(&r.c, mid, b, depth + 1, out);
    }
    let mut out = Vec::new();
    if !c.is_empty() {
        go(c, 0.0, 1.0, 0, &mut out);
    }
    out.dedup_by(|x, y| (*x - *y).abs() <= 1e-15);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernstein_roots_are_every_crossing() {
        // (t − 0.2)(t − 0.5)(t − 0.9) has Bernstein coefficients from its values: check by value.
        let f = |t: f64| (t - 0.2) * (t - 0.5) * (t - 0.9);
        // Coefficients of a cubic from its power form: b_k = Σ_{i≤k} C(k,i)/C(3,i) a_i.
        let a = [-0.09, 0.73, -1.6, 1.0];
        let b: Vec<f64> = (0..4)
            .map(|k| {
                (0..=k)
                    .map(|i| binomial(k, i) / binomial(3, i) * a[i])
                    .sum()
            })
            .collect();
        for t in [0.0, 0.3, 0.7, 1.0] {
            assert!((scalar_casteljau(&b, t) - f(t)).abs() < 1e-14);
        }
        let r = bernstein_roots(&b);
        assert_eq!(r.len(), 3, "{r:?}");
        for (got, want) in r.iter().zip([0.2, 0.5, 0.9]) {
            assert!((got - want).abs() < 1e-14, "{got}");
        }
    }

    #[test]
    fn a_product_in_bernstein_form_is_the_product() {
        let a = Net {
            rows: 2,
            cols: 3,
            c: vec![1.0, -2.0, 0.5, 3.0, 0.25, -1.0],
        };
        let b = Net {
            rows: 3,
            cols: 2,
            c: vec![0.5, 2.0, -1.0, 1.5, 4.0, -3.0],
        };
        let p = a.times(&b);
        for (s, t) in [(0.1, 0.7), (0.5, 0.5), (0.9, 0.2)] {
            assert!((p.eval(s, t) - a.eval(s, t) * b.eval(s, t)).abs() < 1e-13);
        }
    }

    #[test]
    fn a_spline_bowl_has_its_lowest_point_found() {
        // (2u, 2v, 3 + (2u − 1)² + (2v − 1)²) as a quadratic Bézier patch: along n = (p, q, 1) the
        // height is stationary where 2p + 4(2u − 1) = 0 and 2q + 4(2v − 1) = 0.
        let mut ctrl = Vec::new();
        let e = [1.0, -1.0, 1.0];
        for (i, a) in e.iter().enumerate() {
            for (j, b) in e.iter().enumerate() {
                ctrl.push(Point3::new(i as f64, j as f64, 3.0 + a + b));
            }
        }
        let k = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let s = NurbsSurface::new((2, 2), (k.clone(), k), (3, 3), ctrl, None).unwrap();
        for (p, q) in [(0.0, 0.0), (0.3, -0.2), (-1.5, 1.0)] {
            let n = Vec3::new(p, q, 1.0).unit().unwrap();
            let got = spline_critical(&s, n, Vec2::new(-0.1, -0.1), Vec2::new(1.1, 1.1));
            let want = Vec2::new(0.5 - p / 4.0, 0.5 - q / 4.0);
            assert_eq!(got.len(), 1, "{got:?}");
            assert!((got[0] - want).norm() < 1e-12, "{got:?} against {want:?}");
        }
        // Tilted past the bowl's steepest side, the height has no stationary point on the patch.
        let n = Vec3::new(5.0, 0.0, 1.0).unit().unwrap();
        assert!(spline_critical(&s, n, Vec2::new(-0.1, -0.1), Vec2::new(1.1, 1.1)).is_empty());
    }

    #[test]
    fn a_circle_turned_about_an_axis_has_the_tori_critical_latitudes() {
        // The circle of radius 1 about (5, 0, 0) in the xz plane, turned about z: a torus, whose
        // height along n is stationary where the profile's normal is (±|n⊥|, n·z) in (ρ, z).
        let r = crate::geom::Revolution {
            curve: Curve::Circle(Circle {
                frame: Frame::new(Point3::new(5.0, 0.0, 0.0), -Vec3::Y, Vec3::X).unwrap(),
                radius: 1.0,
            }),
            origin: Point3::ZERO,
            axis: Vec3::Z,
        };
        let n = Vec3::new(0.6, 0.0, 0.8);
        let got = revolution_critical(&r, n, 0.0, TAU);
        assert_eq!(got.len(), 4, "{got:?}");
        for v in got {
            let w = r.curve.eval(v) - Point3::new(5.0, 0.0, 0.0);
            let along = |d: Vec3| w.cross(d).norm() < 1e-12;
            assert!(
                along(Vec3::new(0.6, 0.0, 0.8)) || along(Vec3::new(-0.6, 0.0, 0.8)),
                "{w:?}"
            );
        }
    }
}
