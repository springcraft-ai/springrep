//! Non-rational B-spline curves, and cubic interpolation through point lists.

use std::sync::{Arc, OnceLock};

use crate::{
    Result,
    error::invalid,
    math::{Aabb, Transform, Vec3},
    numeric::{solve_banded, solve_cyclic},
    tol,
};

/// The highest degree a B-spline may have; the kernel interpolates with cubics.
pub(super) const MAX_DEGREE: usize = 3;
const W: usize = MAX_DEGREE + 1;

/// A non-rational B-spline. A periodic one stores its first `degree` control points again at the
/// end and its knot vector unwrapped past both ends, so evaluation is the open algorithm on the
/// wrapped parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct BSpline {
    degree: usize,
    knots: Vec<f64>,
    ctrl: Vec<Vec3>,
    periodic: bool,
    spans: SpanCache,
    polys: PolyCache,
}

/// Each knot span's parameter range and a ball holding its control points, so the span (inside
/// their convex hull) lies in the ball.
#[derive(Debug)]
struct Span {
    t0: f64,
    t1: f64,
    centre: Vec3,
    radius: f64,
}

/// Built on first use; it depends only on the spline, so it never takes part in equality.
#[derive(Clone, Debug, Default)]
struct SpanCache(OnceLock<Arc<Vec<Span>>>);

impl PartialEq for SpanCache {
    fn eq(&self, _: &SpanCache) -> bool {
        true
    }
}

/// Each nonempty span's curve as polynomials in the span's local parameter, about its start and
/// about its end, by span number: evaluating one is a Horner pass with no basis recurrence. Built
/// on first use; it depends only on the spline, so it never takes part in equality.
#[derive(Clone, Debug, Default)]
struct PolyCache(OnceLock<Arc<Vec<Option<SpanSeries>>>>);

/// One span's coefficients about its start and about its end.
type SpanSeries = [Vec<Vec3>; 2];

impl PartialEq for PolyCache {
    fn eq(&self, _: &PolyCache) -> bool {
        true
    }
}

impl BSpline {
    /// An open spline from its parts. The knots must be nondecreasing and number
    /// `ctrl.len() + degree + 1`.
    pub fn new(degree: usize, knots: Vec<f64>, ctrl: Vec<Vec3>) -> Result<BSpline> {
        if degree == 0 || degree > MAX_DEGREE {
            return invalid("B-spline degree must be 1 to 3");
        }
        if ctrl.len() <= degree {
            return invalid("a B-spline needs more control points than its degree");
        }
        if knots.len() != ctrl.len() + degree + 1 {
            return invalid("B-spline knot count must be control points + degree + 1");
        }
        if knots.windows(2).any(|w| w[1] < w[0]) {
            return invalid("B-spline knots must not decrease");
        }
        let s = BSpline {
            degree,
            knots,
            ctrl,
            periodic: false,
            spans: SpanCache::default(),
            polys: PolyCache::default(),
        };
        let (a, b) = s.domain();
        if b - a <= 0.0 {
            return invalid("B-spline has an empty domain");
        }
        Ok(s)
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    pub fn control_points(&self) -> &[Vec3] {
        &self.ctrl
    }

    pub fn is_periodic(&self) -> bool {
        self.periodic
    }

    pub fn domain(&self) -> (f64, f64) {
        (self.knots[self.degree], self.knots[self.ctrl.len()])
    }

    pub fn period(&self) -> Option<f64> {
        self.periodic.then(|| {
            let (a, b) = self.domain();
            b - a
        })
    }

    /// The distinct knots strictly inside (t0, t1), where the curve may lose smoothness.
    pub fn breakpoints(&self, t0: f64, t1: f64) -> Vec<f64> {
        let (a, b) = self.domain();
        let mut out = Vec::new();
        let mut push = |k: f64| {
            if k > t0 && k < t1 && out.last().is_none_or(|&l: &f64| k > l) {
                out.push(k);
            }
        };
        match self.period() {
            Some(p) => {
                let start = ((t0 - a) / p).floor() as i64;
                let end = ((t1 - a) / p).ceil() as i64;
                for w in start..=end {
                    for &k in &self.knots[self.degree..=self.ctrl.len()] {
                        push(k + w as f64 * p);
                    }
                }
            }
            None => {
                for &k in &self.knots {
                    if k > a && k < b {
                        push(k);
                    }
                }
            }
        }
        out
    }

    fn wrap(&self, t: f64) -> f64 {
        let (a, b) = self.domain();
        match self.period() {
            Some(p) => {
                let w = a + (t - a).rem_euclid(p);
                if w >= b { a } else { w }
            }
            None => t,
        }
    }

    fn span(&self, t: f64) -> usize {
        let (p, m) = (self.degree, self.ctrl.len());
        if t >= self.knots[m] {
            let mut i = m - 1;
            while i > p && self.knots[i] >= self.knots[m] {
                i -= 1;
            }
            return i;
        }
        if t <= self.knots[p] {
            let mut i = p;
            while i + 1 < m && self.knots[i + 1] <= self.knots[p] {
                i += 1;
            }
            return i;
        }
        let (mut lo, mut hi) = (p, m);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t < self.knots[mid] {
                hi = mid
            } else {
                lo = mid
            }
        }
        lo
    }

    /// Basis function values and derivatives up to `nd` at `t` in span `i`.
    fn basis_ders(&self, i: usize, t: f64, nd: usize) -> [[f64; W]; W] {
        super::nurbs::basis_derivatives(self.degree, &self.knots, i, t, nd)
    }

    /// The point and its first `nd` derivatives at `t`, from the span's polynomial about the
    /// nearer end of the span: exact at both ends, and its local parameter at most ½.
    pub fn eval_ders(&self, t: f64, nd: usize) -> [Vec3; 3] {
        let t = self.wrap(t);
        let i = self.span(t);
        let (t0, t1) = (self.knots[i], self.knots[i + 1]);
        let h = t1 - t0;
        let x = (t - t0) / h;
        let (s, end) = if x > 0.5 { (x - 1.0, 1) } else { (x, 0) };
        let c = &self.polys()[i]
            .as_ref()
            .expect("a span holds its parameter")[end];
        let p = self.degree;
        let (mut p0, mut p1, mut p2) = (c[p], Vec3::ZERO, Vec3::ZERO);
        for k in (0..p).rev() {
            if nd >= 2 {
                p2 = p2 * s + p1;
            }
            if nd >= 1 {
                p1 = p1 * s + p0;
            }
            p0 = p0 * s + c[k];
        }
        [p0, p1 / h, p2 * (2.0 / (h * h))]
    }

    fn polys(&self) -> &[Option<SpanSeries>] {
        self.polys.0.get_or_init(|| {
            let p = self.degree;
            let mut out = vec![None; self.ctrl.len()];
            for (i, o) in out.iter_mut().enumerate().skip(p) {
                if self.knots[i + 1] <= self.knots[i] {
                    continue;
                }
                *o = Some([false, true].map(|end| {
                    let basis = super::nurbs::span_poly(p, &self.knots, i, end);
                    (0..=p)
                        .map(|k| {
                            (0..=p).fold(Vec3::ZERO, |a, j| a + self.ctrl[i - p + j] * basis[j][k])
                        })
                        .collect()
                }));
            }
            Arc::new(out)
        })
    }

    /// [`BSpline::eval_ders`] by the basis recurrence, for checking the polynomials.
    #[cfg(test)]
    fn eval_ders_by_basis(&self, t: f64, nd: usize) -> [Vec3; 3] {
        let t = self.wrap(t);
        let i = self.span(t);
        let ders = self.basis_ders(i, t, nd);
        let mut out = [Vec3::ZERO; 3];
        for (k, row) in ders.iter().enumerate().take(nd.min(2) + 1) {
            let mut s = Vec3::ZERO;
            for (j, &n) in row.iter().enumerate().take(self.degree + 1) {
                s += self.ctrl[i - self.degree + j] * n;
            }
            out[k] = s;
        }
        out
    }

    pub fn eval(&self, t: f64) -> Vec3 {
        self.eval_ders(t, 0)[0]
    }

    pub fn transformed(&self, t: &Transform) -> BSpline {
        BSpline {
            ctrl: self.ctrl.iter().map(|&p| t.point(p)).collect(),
            spans: SpanCache::default(),
            polys: PolyCache::default(),
            ..self.clone()
        }
    }

    fn spans(&self) -> &[Span] {
        self.spans.0.get_or_init(|| {
            let p = self.degree;
            let spans = (p..self.ctrl.len())
                .filter(|&i| self.knots[i + 1] > self.knots[i])
                .map(|i| {
                    let pts = &self.ctrl[i - p..=i];
                    let centre = pts.iter().fold(Vec3::ZERO, |a, &q| a + q) / pts.len() as f64;
                    let radius = pts.iter().map(|q| q.dist(centre)).fold(0.0, f64::max);
                    Span {
                        t0: self.knots[i],
                        t1: self.knots[i + 1],
                        centre,
                        radius: radius * (1.0 + 1e-12) + 1e-12,
                    }
                })
                .collect();
            Arc::new(spans)
        })
    }

    /// [t0, t1] cut where the polynomial pieces meet, each piece with its knot span, or `None`
    /// where an open spline's range runs past its ends and the end spans extend.
    fn pieces(&self, t0: f64, t1: f64) -> Vec<(f64, f64, Option<&Span>)> {
        let (a, b) = self.domain();
        let mut out = Vec::new();
        let shifts = match self.period() {
            Some(period) => {
                let k0 = ((t0 - b) / period).floor() as i64;
                let k1 = ((t1 - a) / period).ceil() as i64;
                (k0..=k1).map(|k| k as f64 * period).collect()
            }
            None => {
                if t0 < a {
                    out.push((t0, t1.min(a), None));
                }
                if t1 > b {
                    out.push((t0.max(b), t1, None));
                }
                vec![0.0]
            }
        };
        let spans = self.spans();
        for shift in shifts {
            let first = spans.partition_point(|s| s.t1 + shift < t0);
            for s in &spans[first..] {
                if s.t0 + shift > t1 {
                    break;
                }
                let (lo, hi) = ((s.t0 + shift).max(t0), (s.t1 + shift).min(t1));
                if lo <= hi {
                    out.push((lo, hi, Some(s)));
                }
            }
        }
        out
    }

    /// Parameters strictly inside (t0, t1) where the coordinate along `dir` is stationary: on each
    /// piece that coordinate's derivative is a polynomial of degree below three, solved exactly.
    pub fn extremes(&self, dir: Vec3, t0: f64, t1: f64) -> Vec<f64> {
        self.extremes_along(&[dir], t0, t1, |_| false)
            .swap_remove(0)
    }

    /// The tight box of the curve over [t0, t1]: its ends and every coordinate's extremes.
    pub fn bbox(&self, t0: f64, t1: f64) -> Aabb {
        let mut bb = Aabb::EMPTY;
        bb.add(self.eval(t0));
        bb.add(self.eval(t1));
        // A piece whose control points' ball lies in the ends' box cannot widen it.
        let ends = bb;
        let inside = |s: &Span| {
            (0..3).all(|k| {
                s.centre[k] - s.radius >= ends.min[k] && s.centre[k] + s.radius <= ends.max[k]
            })
        };
        for ts in self.extremes_along(&[Vec3::X, Vec3::Y, Vec3::Z], t0, t1, inside) {
            for t in ts {
                bb.add(self.eval(t));
            }
        }
        bb
    }

    /// [`BSpline::extremes`] along each of `dirs`, sharing the derivative evaluations, on the
    /// pieces whose control points are not all `inside`.
    fn extremes_along(
        &self,
        dirs: &[Vec3],
        t0: f64,
        t1: f64,
        inside: impl Fn(&Span) -> bool,
    ) -> Vec<Vec<f64>> {
        let mut outs = vec![Vec::new(); dirs.len()];
        let der = |t: f64| self.eval_ders(t, 1)[1];
        let mut last: Option<(f64, Vec3)> = None;
        for (lo, hi, span) in self.pieces(t0, t1) {
            if hi <= lo || span.is_some_and(&inside) {
                continue;
            }
            let d_lo = match last {
                Some((t, d)) if t == lo => d,
                _ => der(lo),
            };
            let (d_mid, d_hi) = (der(0.5 * (lo + hi)), der(hi));
            last = Some((hi, d_hi));
            for (dir, out) in dirs.iter().zip(&mut outs) {
                let (g0, g1, g2) = (d_lo.dot(*dir), d_mid.dot(*dir), d_hi.dot(*dir));
                extremes_on_piece(out, (g0, g1, g2), lo, hi, t0, t1);
            }
        }
        for out in &mut outs {
            out.sort_by(f64::total_cmp);
        }
        outs
    }

    /// The parameter in [t0, t1] of the point nearest `p`, measuring distance across `along`
    /// (a unit vector) when it is given, as an extrusion along it sees the curve. Only the spans
    /// whose ball could hold a nearer point than the best so far are searched.
    pub fn nearest(&self, p: Vec3, along: Option<Vec3>, t0: f64, t1: f64) -> f64 {
        let flat = |v: Vec3| match along {
            Some(d) => v - d * v.dot(d),
            None => v,
        };
        let pieces: Vec<(f64, f64, f64)> = self
            .pieces(t0, t1)
            .into_iter()
            .map(|(lo, hi, span)| {
                let gap = span.map_or(0.0, |s| (flat(s.centre - p).norm() - s.radius).max(0.0));
                (gap, lo, hi)
            })
            .collect();
        let dist = |t: f64| flat(self.eval(t) - p).norm();
        let search = |lo: f64, hi: f64| {
            let n = 8;
            let mut seed = (f64::INFINITY, lo);
            for i in 0..=n {
                let t = lo + (hi - lo) * i as f64 / n as f64;
                let d = dist(t);
                if d < seed.0 {
                    seed = (d, t);
                }
            }
            let mut t = seed.1;
            for _ in 0..50 {
                let [c, d1, d2] = self.eval_ders(t, 2);
                let (r, d1, d2) = (flat(c - p), flat(d1), flat(d2));
                let f = r.dot(d1);
                let df = d1.norm2() + r.dot(d2);
                if df.abs() < 1e-300 {
                    break;
                }
                let next = (t - f / df.max(d1.norm2() * 0.1)).clamp(lo, hi);
                let done = (next - t).abs() <= 1e-15 * (1.0 + t.abs());
                t = next;
                if done {
                    break;
                }
            }
            let d = dist(t);
            if d <= seed.0 { (d, t) } else { seed }
        };
        // The search stops at the first piece whose gap passes the best distance, and the piece of
        // least gap comes first, so no piece whose gap passes that piece's distance is reached.
        let Some(first) = (0..pieces.len()).min_by(|&a, &b| pieces[a].0.total_cmp(&pieces[b].0))
        else {
            return t0;
        };
        let bound = search(pieces[first].1, pieces[first].2).0;
        let mut order: Vec<(f64, f64, f64)> = pieces.into_iter().filter(|q| q.0 <= bound).collect();
        order.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut best = (f64::INFINITY, t0);
        for (gap, lo, hi) in order {
            if gap > best.0 {
                break;
            }
            let found = search(lo, hi);
            if found.0 < best.0 {
                best = found;
            }
        }
        best.1
    }

    /// The basis function values of every control point at `t`, for collocation.
    fn basis_row(&self, t: f64) -> (usize, Vec<f64>) {
        let i = self.span(t);
        (
            i - self.degree,
            self.basis_ders(i, t, 0)[0][..=self.degree].to_vec(),
        )
    }

    /// The open spline through `points` in order, at chord-length parameters. Four or more points
    /// give a cubic with not-a-knot ends; three a quadratic; two a line.
    pub fn interpolate(points: &[Vec3]) -> Result<BSpline> {
        let n = points.len();
        if n < 2 {
            return invalid("interpolation needs at least two points");
        }
        let params = chord_params(points, false)?;
        BSpline::interpolate_at(points, &params)
    }

    /// The open spline through `points` at the increasing parameters `params`: a cubic with
    /// not-a-knot ends from four points, a quadratic through three, a line through two.
    pub fn interpolate_at(points: &[Vec3], params: &[f64]) -> Result<BSpline> {
        let n = points.len();
        if n < 2 || params.len() != n {
            return invalid("interpolation needs a parameter for each of at least two points");
        }
        let p = (n - 1).min(3);
        let mut knots = vec![params[0]; p + 1];
        if p == 3 {
            knots.extend_from_slice(&params[2..n - 2]);
        }
        knots.extend(std::iter::repeat_n(params[n - 1], p + 1));
        let shape = BSpline {
            degree: p,
            knots,
            ctrl: vec![Vec3::ZERO; n],
            periodic: false,
            spans: SpanCache::default(),
            polys: PolyCache::default(),
        };
        let rows = params.iter().map(|&t| shape.basis_row(t)).collect();
        let b: Vec<f64> = points.iter().flat_map(|p| p.to_array()).collect();
        let x = solve_banded(rows, b, 3)?;
        let ctrl = x.chunks(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
        BSpline::new(p, shape.knots, ctrl)
    }

    /// The open cubic through `points` at chord-length parameters, with a knot at every inner
    /// point and each end's derivative that of the cubic through its four nearest points (the
    /// parabola through three when there are only three), the ends other modellers give a
    /// spline through points. Two points give a line.
    pub fn interpolate_bessel(points: &[Vec3]) -> Result<BSpline> {
        let n = points.len();
        if n < 3 {
            return BSpline::interpolate(points);
        }
        let t = chord_params(points, false)?;
        let mut knots = vec![t[0]; 4];
        knots.extend_from_slice(&t[1..n - 1]);
        knots.extend([t[n - 1]; 4]);
        let m = n + 2;
        let shape = BSpline {
            degree: 3,
            knots,
            ctrl: vec![Vec3::ZERO; m],
            periodic: false,
            spans: SpanCache::default(),
            polys: PolyCache::default(),
        };
        // The derivative at `t[at]` of the polynomial through the points `idx` (Lagrange form).
        let slope = |idx: &[usize], at: usize| {
            let mut d = Vec3::ZERO;
            for &j in idx {
                let l = if j == at {
                    idx.iter()
                        .filter(|&&m| m != j)
                        .map(|&m| 1.0 / (t[j] - t[m]))
                        .sum::<f64>()
                } else {
                    idx.iter()
                        .filter(|&&m| m != j && m != at)
                        .map(|&m| (t[at] - t[m]) / (t[j] - t[m]))
                        .product::<f64>()
                        / (t[j] - t[at])
                };
                d += points[j] * l;
            }
            d
        };
        let k = n.min(4);
        let head: Vec<usize> = (0..k).collect();
        let tail: Vec<usize> = (n - k..n).collect();
        let (start, end) = (slope(&head, 0), slope(&tail, n - 1));
        let mut a = Vec::with_capacity(m);
        let mut b: Vec<f64> = Vec::with_capacity(3 * m);
        let put = |a: &mut Vec<_>, at: f64, der: usize| {
            let i = shape.span(at);
            let ders = shape.basis_ders(i, at, der);
            a.push((i - 3, ders[der][..=3].to_vec()));
        };
        put(&mut a, t[0], 0);
        b.extend(points[0].to_array());
        put(&mut a, t[0], 1);
        b.extend(start.to_array());
        for k in 1..n - 1 {
            put(&mut a, t[k], 0);
            b.extend(points[k].to_array());
        }
        put(&mut a, t[n - 1], 1);
        b.extend(end.to_array());
        put(&mut a, t[n - 1], 0);
        b.extend(points[n - 1].to_array());
        let x = solve_banded(a, b, 3)?;
        let ctrl = x.chunks(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
        BSpline::new(3, shape.knots, ctrl)
    }

    /// The closed C² cubic through `points` and back to the first, at chord-length parameters.
    pub fn interpolate_periodic(points: &[Vec3]) -> Result<BSpline> {
        if points.len() < 3 {
            return invalid("a periodic spline needs at least three points");
        }
        let params = chord_params(points, true)?;
        BSpline::interpolate_periodic_at(points, &params)
    }

    /// The closed C² cubic through `points` at `params[..n]`, back to the first at `params[n]`,
    /// one period on. Splines through different points at the same parameters share their knots.
    pub fn interpolate_periodic_at(points: &[Vec3], params: &[f64]) -> Result<BSpline> {
        let n = points.len();
        if n < 3 || params.len() != n + 1 || params.windows(2).any(|w| w[1] <= w[0]) {
            return invalid(
                "a periodic spline needs three or more points at increasing parameters",
            );
        }
        let period = params[n] - params[0];
        let p = 3;
        // Knots u_{-3} .. u_{n+3}, with u_{i+n} = u_i + period.
        let knot = |i: isize| -> f64 {
            let k = i.rem_euclid(n as isize) as usize;
            params[k] + period * (i.div_euclid(n as isize)) as f64
        };
        let knots: Vec<f64> = (-3..=(n as isize + 3)).map(knot).collect();
        let shape = BSpline {
            degree: p,
            knots,
            ctrl: vec![Vec3::ZERO; n + p],
            periodic: false,
            spans: SpanCache::default(),
            polys: PolyCache::default(),
        };
        // Unknown `q` is control point `q + 1`, which puts each row's largest basis value, at the
        // point's own knot, on the diagonal as the cyclic solve wants.
        let rows = params[..n]
            .iter()
            .map(|&t| {
                let (first, vals) = shape.basis_row(t);
                ((first + n - 1) % n, vals)
            })
            .collect();
        let b: Vec<f64> = points.iter().flat_map(|p| p.to_array()).collect();
        let x = solve_cyclic(rows, b, 3)?;
        let mut ctrl: Vec<Vec3> = (0..n)
            .map(|i| {
                let q = (i + n - 1) % n;
                Vec3::new(x[3 * q], x[3 * q + 1], x[3 * q + 2])
            })
            .collect();
        for i in 0..p {
            ctrl.push(ctrl[i]);
        }
        Ok(BSpline {
            degree: p,
            knots: shape.knots,
            ctrl,
            periodic: true,
            spans: SpanCache::default(),
            polys: PolyCache::default(),
        })
    }
}

impl BSpline {
    /// The closed cubic of period `knots[n + 3] − knots[3]` over `n` control points, its knots
    /// `u₋₃ … u_{n+3}` (`u_{i+n} = u_i + period`) and its control points the `n` with the first
    /// three repeated after them, as [`BSpline::interpolate_periodic_at`] makes them.
    pub fn periodic_cubic(knots: Vec<f64>, ctrl: Vec<Vec3>) -> Result<BSpline> {
        let n = ctrl.len().saturating_sub(3);
        if n < 3 || knots.len() != n + 7 || ctrl[n..] != ctrl[..3] {
            return invalid("a periodic cubic needs n + 7 knots and its first three points again");
        }
        BSpline::periodic(3, knots, ctrl)
    }

    /// The closed spline of `degree` (1 to 3) over `n` control points, its knots
    /// `u₋ₚ … u_{n+p}` with `u_{i+n} = u_i + period`, `p` the degree, and its control points the
    /// `n` with the first `p` repeated after them. It has no end: the curve is as smooth across
    /// `u₀`, where its parameter wraps, as across any other knot of the same multiplicity.
    pub fn periodic(degree: usize, knots: Vec<f64>, ctrl: Vec<Vec3>) -> Result<BSpline> {
        let p = degree;
        if p == 0 || p > MAX_DEGREE {
            return invalid("B-spline degree must be 1 to 3");
        }
        let n = ctrl.len().saturating_sub(p);
        if n <= p || knots.len() != n + 2 * p + 1 || ctrl[n..] != ctrl[..p] {
            return invalid(
                "a periodic spline of degree p needs more than p points, n + 2p + 1 knots and its \
                 first p points again",
            );
        }
        if knots.windows(2).any(|w| w[1] < w[0]) {
            return invalid("B-spline knots must not decrease");
        }
        let period = knots[n + p] - knots[p];
        let wraps = (0..=2 * p).all(|i| {
            let step = knots[i + n] - knots[i];
            (step - period).abs() <= 1e-9 * period
        });
        if !(period > 0.0 && wraps) {
            return invalid("a periodic spline's knots must repeat a period apart");
        }
        Ok(BSpline {
            degree: p,
            knots,
            ctrl,
            periodic: true,
            spans: SpanCache::default(),
            polys: PolyCache::default(),
        })
    }

    /// The closed uniform spline of `degree` (1 to 3) on the control polygon `points`, taken in
    /// order and closing back on the first: one knot span to each point, `u_i = i`, so the
    /// period is the number of points. A quadratic's span from `u = i` to `i + 1` runs from the
    /// middle of the side from point `i` to point `i + 1` to the middle of the next side, tangent
    /// to both.
    pub fn periodic_uniform(degree: usize, points: &[Vec3]) -> Result<BSpline> {
        let n = points.len();
        if n <= degree {
            return invalid("a periodic spline needs more points than its degree");
        }
        let knots = (0..=n + 2 * degree)
            .map(|i| i as f64 - degree as f64)
            .collect();
        let ctrl = (0..n + degree).map(|i| points[i % n]).collect();
        BSpline::periodic(degree, knots, ctrl)
    }
}

pub(crate) fn chord_params(points: &[Vec3], closed: bool) -> Result<Vec<f64>> {
    let mut params = vec![0.0];
    let n = points.len();
    let count = if closed { n } else { n - 1 };
    for i in 0..count {
        let d = points[i].dist(points[(i + 1) % n]);
        if d <= tol::LINEAR {
            return invalid("interpolation points must be distinct");
        }
        params.push(params[i] + d);
    }
    Ok(params)
}

/// The stationary points in one piece of a coordinate whose derivative is `g0`, `g1`, `g2` at the
/// piece's start, middle and end: `g(lo + s·(hi − lo)) = g0 + b·s + a·s²` on s ∈ [0, 1].
fn extremes_on_piece(
    out: &mut Vec<f64>,
    (g0, g1, g2): (f64, f64, f64),
    lo: f64,
    hi: f64,
    t0: f64,
    t1: f64,
) {
    let qa = 2.0 * g0 - 4.0 * g1 + 2.0 * g2;
    let qb = -3.0 * g0 + 4.0 * g1 - g2;
    let scale = g0.abs().max(g1.abs()).max(g2.abs());
    if scale == 0.0 {
        return;
    }
    let mut ss = Vec::new();
    if qa.abs() <= 1e-14 * scale {
        if qb.abs() > 1e-14 * scale {
            ss.push(-g0 / qb);
        }
    } else {
        let disc = qb * qb - 4.0 * qa * g0;
        if disc >= 0.0 {
            let q = -0.5 * (qb + qb.signum() * disc.sqrt());
            ss.push(q / qa);
            if q != 0.0 {
                ss.push(g0 / q);
            }
        }
    }
    for s in ss {
        let t = lo + s * (hi - lo);
        if (0.0..=1.0).contains(&s) && t > t0 && t < t1 && !out.contains(&t) {
            out.push(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_periodic_uniform_quadratic_rounds_its_polygon_from_side_middle_to_side_middle() {
        let square = [
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(-10.0, 10.0, 0.0),
            Vec3::new(-10.0, -10.0, 0.0),
            Vec3::new(10.0, -10.0, 0.0),
        ];
        let c = BSpline::periodic_uniform(2, &square).unwrap();
        assert_eq!(c.domain(), (0.0, 4.0));
        assert_eq!(c.period(), Some(4.0));
        for i in 0..4 {
            let (a, b, z) = (square[i], square[(i + 1) % 4], square[(i + 2) % 4]);
            let t = i as f64;
            // The span's ends are the sides' middles; its middle is (a + 6b + z)/8, the uniform
            // quadratic's basis at its span's middle being 1/8, 6/8 and 1/8.
            assert!(c.eval(t).dist((a + b) * 0.5) < 1e-12);
            assert!(c.eval(t + 0.5).dist((a + b * 6.0 + z) / 8.0) < 1e-12);
            // Tangent to the side at its middle, both sides of the knot, and across the wrap.
            let [_, d, _] = c.eval_ders(t, 1);
            assert!(d.cross(b - a).norm() < 1e-9 && d.dot(b - a) > 0.0);
            let [_, before, _] = c.eval_ders(t - 1e-9 + if i == 0 { 4.0 } else { 0.0 }, 1);
            assert!(before.dist(d) < 1e-6 * d.norm());
        }
        assert!(BSpline::periodic_uniform(2, &square[..2]).is_err());
        let skewed = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0];
        let ctrl: Vec<Vec3> = (0..6).map(|i| square[i % 4]).collect();
        assert!(BSpline::periodic(2, skewed, ctrl).is_err());
    }

    fn ellipse_points(n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                Vec3::new(30.0 * a.cos(), 20.0 * a.sin(), 0.0)
            })
            .collect()
    }

    #[test]
    fn a_box_holds_every_point_of_its_range_and_touches_its_extremes() {
        let pts: Vec<Vec3> = (0..40)
            .map(|i| {
                let t = i as f64 * 0.37;
                Vec3::new(t.cos() * (2.0 + t * 0.1), t.sin() * 1.5, (2.0 * t).sin())
            })
            .collect();
        let c = BSpline::interpolate(&pts).unwrap();
        let (a, b) = c.domain();
        for (t0, t1) in [
            (a, b),
            (a + 0.3 * (b - a), a + 0.35 * (b - a)),
            (0.5, 0.5 + 1e-3),
        ] {
            let bb = c.bbox(t0, t1);
            let mut dense = Aabb::EMPTY;
            for i in 0..=4000 {
                dense.add(c.eval(t0 + (t1 - t0) * i as f64 / 4000.0));
            }
            for k in 0..3 {
                assert!(bb.min[k] <= dense.min[k] + 1e-12 && bb.max[k] >= dense.max[k] - 1e-12);
                assert!(dense.min[k] - bb.min[k] < 1e-4 && bb.max[k] - dense.max[k] < 1e-4);
            }
        }
    }

    #[test]
    fn open_cubic_passes_through_its_points_and_its_derivative_matches_differences() {
        let pts: Vec<Vec3> = (0..7)
            .map(|i| Vec3::new(i as f64, (i as f64 * 0.7).sin() * 3.0, 0.0))
            .collect();
        let s = BSpline::interpolate(&pts).unwrap();
        let params = chord_params(&pts, false).unwrap();
        for (p, &t) in pts.iter().zip(&params) {
            assert!(s.eval(t).dist(*p) < 1e-12);
        }
        let t = 2.3;
        let h = 1e-6;
        let [_, d1, d2] = s.eval_ders(t, 2);
        let fd = (s.eval(t + h) - s.eval(t - h)) / (2.0 * h);
        let fd2 = (s.eval(t + h) - s.eval(t) * 2.0 + s.eval(t - h)) / (h * h);
        assert!(d1.dist(fd) < 1e-7, "{d1:?} {fd:?}");
        assert!(d2.dist(fd2) < 1e-3, "{d2:?} {fd2:?}");
    }

    #[test]
    fn periodic_cubic_closes_smoothly_through_ellipse_points() {
        let pts = ellipse_points(72);
        let s = BSpline::interpolate_periodic(&pts).unwrap();
        let (a, b) = s.domain();
        let params = chord_params(&pts, true).unwrap();
        for (p, &t) in pts.iter().zip(&params) {
            assert!(s.eval(t).dist(*p) < 1e-12);
        }
        let [p0, d0, dd0] = s.eval_ders(a, 2);
        let [p1, d1, dd1] = s.eval_ders(b - 1e-12, 2);
        assert!(p0.dist(p1) < 1e-9);
        assert!(d0.dist(d1) < 1e-9);
        assert!(dd0.dist(dd1) < 1e-6);
        // Between samples the interpolant stays within 1e-5 of the ellipse.
        for i in 0..720 {
            let q = s.eval(a + (b - a) * (i as f64 + 0.5) / 720.0);
            let r = (q.x / 30.0).powi(2) + (q.y / 20.0).powi(2);
            assert!((r - 1.0).abs() < 2e-5, "{r}");
        }
    }

    #[test]
    fn ends_take_the_tangent_of_the_cubic_through_their_four_nearest_points() {
        let pts: Vec<Vec3> = (0..6)
            .map(|i| Vec3::new(i as f64, (i as f64 * 0.9).sin() * 2.0, 0.1 * i as f64))
            .collect();
        let s = BSpline::interpolate_bessel(&pts).unwrap();
        assert_eq!(s.control_points().len(), pts.len() + 2);
        let t = chord_params(&pts, false).unwrap();
        for (p, &u) in pts.iter().zip(&t) {
            assert!(s.eval(u).dist(*p) < 1e-12);
        }
        // Four points determine the cubic, so their own spline is it and its end tangents match.
        let four = BSpline::interpolate_bessel(&pts[..4]).unwrap();
        let tf = chord_params(&pts[..4], false).unwrap();
        assert!(s.eval_ders(t[0], 1)[1].dist(four.eval_ders(tf[0], 1)[1]) < 1e-12);
        let mid = 0.5 * (tf[1] + tf[2]);
        let lagrange: Vec3 = (0..4)
            .map(|j| {
                let l: f64 = (0..4)
                    .filter(|&m| m != j)
                    .map(|m| (mid - tf[m]) / (tf[j] - tf[m]))
                    .product();
                pts[j] * l
            })
            .fold(Vec3::ZERO, |a, b| a + b);
        assert!(four.eval(mid).dist(lagrange) < 1e-12);
        let tail = BSpline::interpolate_bessel(&pts[2..]).unwrap();
        let tt = chord_params(&pts[2..], false).unwrap();
        let end = s.eval_ders(t[5], 1)[1];
        assert!(end.dist(tail.eval_ders(tt[3], 1)[1]) < 1e-12);
        let three = BSpline::interpolate_bessel(&pts[..3]).unwrap();
        assert!(three.eval(t[1]).dist(pts[1]) < 1e-12);
    }

    fn wobbly_ring() -> BSpline {
        let pts: Vec<Vec3> = (0..90)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 90.0;
                let r = 30.0 + 1.5 * (9.0 * a).cos();
                Vec3::new(r * a.cos(), r * a.sin(), 0.0)
            })
            .collect();
        BSpline::interpolate_periodic(&pts).unwrap()
    }

    #[test]
    fn span_polynomials_agree_with_the_basis_recurrence() {
        // The point and both derivatives, at every knot, inside each span and past a periodic
        // spline's ends: an interpolated ring, an open cubic, a quadratic with a double knot (C⁰
        // there) and a polyline.
        let pts = |n: usize| -> Vec<Vec3> {
            (0..n)
                .map(|i| {
                    let x = i as f64;
                    Vec3::new(x * 3.1, (x * 0.7).sin() * 4.0, x * x * 0.05)
                })
                .collect()
        };
        let mut splines = vec![wobbly_ring(), BSpline::interpolate(&pts(11)).unwrap()];
        let ctrl = pts(6);
        splines.push(
            BSpline::new(
                2,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.5, 4.0, 4.0, 4.0],
                ctrl.clone(),
            )
            .unwrap(),
        );
        splines.push(BSpline::new(1, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.5, 5.0, 5.0], ctrl).unwrap());
        for c in &splines {
            let (a, b) = c.domain();
            let mut ts: Vec<f64> = c.knots().to_vec();
            ts.extend((0..=97).map(|k| a + (b - a) * (k as f64 / 97.0)));
            if c.is_periodic() {
                ts.extend([a - 0.3 * (b - a), b + 0.7 * (b - a)]);
            }
            for &t in &ts {
                if !c.is_periodic() && (t < a || t > b) {
                    continue;
                }
                let got = c.eval_ders(t, 2);
                let want = c.eval_ders_by_basis(t, 2);
                for (k, tol) in [1e-12, 1e-11, 1e-10].into_iter().enumerate() {
                    let size = 1.0 + want[k].norm();
                    assert!(
                        got[k].dist(want[k]) <= tol * size,
                        "order {k} at {t}: {:?} against {:?}",
                        got[k],
                        want[k]
                    );
                }
            }
        }
    }

    #[test]
    fn nearest_points_match_dense_sampling() {
        let s = wobbly_ring();
        let (a, b) = s.domain();
        let ranges = [(a, b), (a - 7.0, a + 5.0), (a + 40.0, a + 41.0)];
        let points = [
            Vec3::new(0.3, -0.2, 0.0),
            Vec3::new(31.0, 2.0, 4.0),
            Vec3::new(-50.0, 10.0, -3.0),
            Vec3::new(5.0, 28.0, 1.0),
        ];
        for (t0, t1) in ranges {
            for p in points {
                for along in [None, Some(Vec3::Z)] {
                    let flat = |v: Vec3| along.map_or(v, |d: Vec3| v - d * v.dot(d));
                    let dist = |t: f64| flat(s.eval(t) - p).norm();
                    let t = s.nearest(p, along, t0, t1);
                    assert!(t >= t0 && t <= t1);
                    let dense = (0..=4000)
                        .map(|i| dist(t0 + (t1 - t0) * i as f64 / 4000.0))
                        .fold(f64::INFINITY, f64::min);
                    assert!(dist(t) <= dense + 1e-12, "{p:?} {t0} {t1}");
                }
            }
        }
    }

    #[test]
    fn extremes_are_every_stationary_point() {
        let s = wobbly_ring();
        let (a, b) = s.domain();
        for dir in [Vec3::X, Vec3::new(0.6, 0.8, 0.0)] {
            for (t0, t1) in [(a, b), (a - 7.0, a + 5.0)] {
                let ts = s.extremes(dir, t0, t1);
                let g = |t: f64| s.eval_ders(t, 1)[1].dot(dir);
                for &t in &ts {
                    assert!(t > t0 && t < t1);
                    assert!(g(t).abs() < 1e-9, "{t} {}", g(t));
                }
                // A derivative of exactly zero at an end of the window is no change inside it.
                let n = 5000;
                let changes = (0..n)
                    .filter(|&i| {
                        let x = |i: usize| t0 + (t1 - t0) * i as f64 / n as f64;
                        let (g0, g1) = (g(x(i)), g(x(i + 1)));
                        let at_an_end = (i == 0 && g0 == 0.0) || (i + 1 == n && g1 == 0.0);
                        !at_an_end && (g0 < 0.0) != (g1 < 0.0)
                    })
                    .count();
                assert_eq!(ts.len(), changes);
            }
        }
    }

    #[test]
    fn three_points_give_a_quadratic() {
        let pts = [
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let s = BSpline::interpolate(&pts).unwrap();
        assert_eq!(s.degree(), 2);
        assert!(s.eval(s.domain().1).dist(pts[2]) < 1e-14);
    }
}
