//! 3-D curves: lines, circles and ellipses in closed form, B-splines, and intersection curves.

use std::f64::consts::{PI, TAU};

use super::{BSpline, IntersectionCurve, NurbsCurve};
use crate::math::{Aabb, Frame, Point3, Transform, Vec3};

/// `origin + t·dir`, with `dir` a unit vector, so `t` is arc length.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub origin: Point3,
    pub dir: Vec3,
}

/// `frame.origin + radius·(cos t·x + sin t·y)`: counter-clockwise about `frame.z`.
#[derive(Clone, Debug, PartialEq)]
pub struct Circle {
    pub frame: Frame,
    pub radius: f64,
}

/// `frame.origin + major·cos t·x + minor·sin t·y`.
#[derive(Clone, Debug, PartialEq)]
pub struct Ellipse {
    pub frame: Frame,
    pub major: f64,
    pub minor: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Curve {
    Line(Line),
    Circle(Circle),
    Ellipse(Ellipse),
    BSpline(BSpline),
    Intersection(Box<IntersectionCurve>),
    /// A rational B-spline of any degree, as STEP files carry them.
    Nurbs(Box<NurbsCurve>),
}

impl Line {
    /// The line from `a` to `b`, parameterised by arc length from `a`; `None` if they coincide.
    pub fn through(a: Point3, b: Point3) -> Option<(Line, f64)> {
        let d = b - a;
        let len = d.norm();
        let dir = d.unit()?;
        Some((Line { origin: a, dir }, len))
    }
}

impl Curve {
    pub fn kind(&self) -> &'static str {
        match self {
            Curve::Line(_) => "line",
            Curve::Circle(_) => "circle",
            Curve::Ellipse(_) => "ellipse",
            Curve::BSpline(_) => "bspline",
            Curve::Intersection(_) => "intersection",
            Curve::Nurbs(_) => "nurbs",
        }
    }

    pub fn eval(&self, t: f64) -> Point3 {
        match self {
            Curve::BSpline(b) => b.eval(t),
            Curve::Intersection(c) => c.approx.eval(t),
            Curve::Nurbs(c) => c.eval(t),
            _ => self.d2(t).0,
        }
    }

    pub fn d1(&self, t: f64) -> (Point3, Vec3) {
        let spline = |b: &BSpline| {
            let [p, d, _] = b.eval_ders(t, 1);
            (p, d)
        };
        match self {
            Curve::BSpline(b) => spline(b),
            Curve::Intersection(c) => spline(&c.approx),
            Curve::Nurbs(c) => c.d1(t),
            _ => {
                let (p, d, _) = self.d2(t);
                (p, d)
            }
        }
    }

    /// The spline the curve evaluates, for the B-spline and intersection curves.
    pub(crate) fn spline(&self) -> Option<&BSpline> {
        match self {
            Curve::BSpline(b) => Some(b),
            Curve::Intersection(c) => Some(&c.approx),
            _ => None,
        }
    }

    pub fn d2(&self, t: f64) -> (Point3, Vec3, Vec3) {
        match self {
            Curve::Line(l) => (l.origin + l.dir * t, l.dir, Vec3::ZERO),
            Curve::Circle(c) => {
                let (s, co) = t.sin_cos();
                let f = &c.frame;
                let r = c.radius;
                (
                    f.origin + (f.x * co + f.y * s) * r,
                    (f.y * co - f.x * s) * r,
                    (f.x * co + f.y * s) * -r,
                )
            }
            Curve::Ellipse(e) => {
                let (s, co) = t.sin_cos();
                let f = &e.frame;
                let (a, b) = (e.major, e.minor);
                (
                    f.origin + f.x * (a * co) + f.y * (b * s),
                    f.y * (b * co) - f.x * (a * s),
                    -(f.x * (a * co) + f.y * (b * s)),
                )
            }
            Curve::BSpline(b) => {
                let [p, d, dd] = b.eval_ders(t, 2);
                (p, d, dd)
            }
            Curve::Intersection(c) => {
                let [p, d, dd] = c.approx.eval_ders(t, 2);
                (p, d, dd)
            }
            Curve::Nurbs(c) => c.d2(t),
        }
    }

    pub fn period(&self) -> Option<f64> {
        match self {
            Curve::Circle(_) | Curve::Ellipse(_) => Some(TAU),
            Curve::BSpline(b) => b.period(),
            Curve::Intersection(c) => c.approx.period(),
            Curve::Nurbs(c) => c.period(),
            Curve::Line(_) => None,
        }
    }

    /// The parameter range the curve is defined on; one period for a periodic curve.
    pub fn domain(&self) -> (f64, f64) {
        match self {
            Curve::Line(_) => (f64::NEG_INFINITY, f64::INFINITY),
            Curve::Circle(_) | Curve::Ellipse(_) => (0.0, TAU),
            Curve::BSpline(b) => b.domain(),
            Curve::Intersection(c) => c.approx.domain(),
            Curve::Nurbs(c) => c.domain(),
        }
    }

    /// Parameters strictly inside (t0, t1) where the curve may lose smoothness; quadrature and
    /// sampling split there.
    pub fn breakpoints(&self, t0: f64, t1: f64) -> Vec<f64> {
        match self {
            Curve::BSpline(b) => b.breakpoints(t0, t1),
            Curve::Intersection(c) => c.approx.breakpoints(t0, t1),
            Curve::Nurbs(c) => c.breakpoints(t0, t1),
            _ => Vec::new(),
        }
    }

    pub fn transformed(&self, t: &Transform) -> Curve {
        match self {
            Curve::Line(l) => Curve::Line(Line {
                origin: t.point(l.origin),
                dir: t.vector(l.dir),
            }),
            Curve::Circle(c) => Curve::Circle(Circle {
                frame: c.frame.transformed(t),
                radius: c.radius,
            }),
            Curve::Ellipse(e) => Curve::Ellipse(Ellipse {
                frame: e.frame.transformed(t),
                ..*e
            }),
            Curve::BSpline(b) => Curve::BSpline(b.transformed(t)),
            Curve::Intersection(c) => Curve::Intersection(Box::new(c.transformed(t))),
            Curve::Nurbs(c) => Curve::Nurbs(Box::new(c.transformed(t))),
        }
    }

    /// The parameter in [t0, t1] of the point of the curve nearest `p`.
    pub fn project(&self, p: Point3, t0: f64, t1: f64) -> f64 {
        match self {
            Curve::Line(l) => (p - l.origin).dot(l.dir).clamp(t0, t1),
            Curve::Circle(c) => {
                let q = c.frame.to_local(p);
                let a = q.y.atan2(q.x);
                self.angle_in_range(a, p, t0, t1)
            }
            Curve::Ellipse(e) => {
                let q = e.frame.to_local(p);
                let t = ellipse_nearest(e.major, e.minor, q.x, q.y);
                let t = t0 + (t - t0).rem_euclid(TAU);
                if t <= t1 {
                    return t;
                }
                // The nearest point lies outside the range: the nearest within it is an end or a
                // second local least, which the sampled search finds.
                let s = self.project_numeric(p, t0, t1);
                [t0, t1, s]
                    .into_iter()
                    .min_by(|&x, &y| self.eval(x).dist(p).total_cmp(&self.eval(y).dist(p)))
                    .unwrap_or(s)
            }
            Curve::Nurbs(n) => {
                let (a, b) = n.domain();
                if n.period().is_none() && (t0 < a || t1 > b) {
                    self.project_numeric(p, t0, t1)
                } else {
                    self.project_pieces(p, t0, t1, |t| n.span_box(t))
                }
            }
            _ => match self.spline() {
                Some(b) => b.nearest(p, None, t0, t1),
                None => self.project_numeric(p, t0, t1),
            },
        }
    }

    fn angle_in_range(&self, a: f64, p: Point3, t0: f64, t1: f64) -> f64 {
        let a = t0 + (a - t0).rem_euclid(TAU);
        if a <= t1 {
            return a;
        }
        let near0 = self.eval(t0).dist(p);
        let near1 = self.eval(t1).dist(p);
        if near0 <= near1 { t0 } else { t1 }
    }

    /// The nearest point by samples, each local least of them refined by Newton's method. Over a
    /// whole period of a closed curve the parameter wraps round instead of stopping at the range's
    /// ends, which are one point: the nearest point can lie just across them from the best sample.
    fn project_numeric(&self, p: Point3, t0: f64, t1: f64) -> f64 {
        let wrap = self.period().filter(|&w| t1 - t0 >= w * (1.0 - 1e-12));
        let mut cuts = vec![t0];
        cuts.extend(self.breakpoints(t0, t1));
        cuts.push(t1);
        let mut samples: Vec<(f64, f64)> = Vec::new();
        for w in cuts.windows(2) {
            let n = 16;
            for i in 0..=n {
                let t = w[0] + (w[1] - w[0]) * i as f64 / n as f64;
                samples.push((self.eval(t).dist(p), t));
            }
        }
        let refine = |t: f64| self.refine_nearest(p, t, t0, t1, wrap);
        let n = samples.len();
        let mut best = samples
            .iter()
            .copied()
            .fold((f64::INFINITY, t0), |a, b| if b.0 < a.0 { b } else { a });
        for k in 0..n {
            let d = samples[k].0;
            let before = if k > 0 {
                samples[k - 1].0
            } else {
                f64::INFINITY
            };
            let after = if k + 1 < n {
                samples[k + 1].0
            } else {
                f64::INFINITY
            };
            if d > before || d > after {
                continue;
            }
            let t = refine(samples[k].1);
            let dt = self.eval(t).dist(p);
            if dt < best.0 {
                best = (dt, t);
            }
        }
        best.1
    }

    /// [`Curve::project_numeric`] searching only the pieces between breakpoints whose box, from
    /// `hull`, could hold a point nearer than the best found, nearest box first. Each piece is
    /// sampled as there, and a sample is a local least as there, its neighbour across a piece's end
    /// being the same point.
    fn project_pieces(&self, p: Point3, t0: f64, t1: f64, hull: impl Fn(f64) -> Aabb) -> f64 {
        let wrap = self.period().filter(|&w| t1 - t0 >= w * (1.0 - 1e-12));
        let mut cuts = vec![t0];
        cuts.extend(self.breakpoints(t0, t1));
        cuts.push(t1);
        let gap = |bb: &Aabb| {
            let d = Vec3::new(
                (bb.min.x - p.x).max(p.x - bb.max.x).max(0.0),
                (bb.min.y - p.y).max(p.y - bb.max.y).max(0.0),
                (bb.min.z - p.z).max(p.z - bb.max.z).max(0.0),
            );
            d.norm()
        };
        let mut order: Vec<(f64, usize)> = cuts
            .windows(2)
            .enumerate()
            .map(|(i, w)| (gap(&hull(0.5 * (w[0] + w[1]))), i))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let last = cuts.len() - 2;
        let n = 16;
        let mut best = (f64::INFINITY, t0);
        let mut ds = [0.0; 17];
        for (g, i) in order {
            if g > best.0 {
                break;
            }
            let (a, b) = (cuts[i], cuts[i + 1]);
            for (k, d) in ds.iter_mut().enumerate() {
                let t = a + (b - a) * k as f64 / n as f64;
                *d = self.eval(t).dist(p);
                if *d < best.0 {
                    best = (*d, t);
                }
            }
            for k in 0..=n {
                let before = if k > 0 {
                    ds[k - 1]
                } else if i == 0 {
                    f64::INFINITY
                } else {
                    ds[0]
                };
                let after = if k < n {
                    ds[k + 1]
                } else if i == last {
                    f64::INFINITY
                } else {
                    ds[n]
                };
                if ds[k] > before || ds[k] > after {
                    continue;
                }
                let t = self.refine_nearest(p, a + (b - a) * k as f64 / n as f64, t0, t1, wrap);
                let dt = self.eval(t).dist(p);
                if dt < best.0 {
                    best = (dt, t);
                }
            }
        }
        best.1
    }

    /// Newton's method on the squared distance to `p` from `t`, held to [t0, t1] or wrapped
    /// round `wrap`.
    fn refine_nearest(&self, p: Point3, mut t: f64, t0: f64, t1: f64, wrap: Option<f64>) -> f64 {
        for _ in 0..50 {
            let (c, d1, d2) = self.d2(t);
            let r = c - p;
            let f = r.dot(d1);
            let df = d1.norm2() + r.dot(d2);
            if df.abs() < 1e-300 {
                break;
            }
            let step = f / df.max(d1.norm2() * 0.1);
            let next = match wrap {
                Some(w) => t0 + (t - step - t0).rem_euclid(w),
                None => (t - step).clamp(t0, t1),
            };
            let done = step.abs() <= 1e-15 * (1.0 + t.abs()) || next == t;
            t = next;
            if done {
                break;
            }
        }
        t
    }

    /// Parameters strictly inside (t0, t1) where the curve's coordinate along `dir` is stationary.
    /// The tight box of the curve over [t0, t1].
    pub fn bbox(&self, t0: f64, t1: f64) -> Aabb {
        if let Some(s) = self.spline() {
            return s.bbox(t0, t1);
        }
        if let Curve::Nurbs(c) = self {
            return c.bbox(t0, t1);
        }
        let mut bb = Aabb::EMPTY;
        bb.add(self.eval(t0));
        bb.add(self.eval(t1));
        for ax in [Vec3::X, Vec3::Y, Vec3::Z] {
            for t in self.extremes(ax, t0, t1) {
                bb.add(self.eval(t));
            }
        }
        bb
    }

    pub fn extremes(&self, dir: Vec3, t0: f64, t1: f64) -> Vec<f64> {
        let in_range = |base: f64| -> Vec<f64> {
            let mut out = Vec::new();
            let k0 = ((t0 - base) / PI).floor() as i64;
            let k1 = ((t1 - base) / PI).ceil() as i64;
            for k in k0..=k1 {
                let t = base + k as f64 * PI;
                if t > t0 && t < t1 {
                    out.push(t);
                }
            }
            out
        };
        match self {
            Curve::Line(_) => Vec::new(),
            Curve::Circle(c) => in_range(dir.dot(c.frame.y).atan2(dir.dot(c.frame.x))),
            Curve::Ellipse(e) => {
                in_range((e.minor * dir.dot(e.frame.y)).atan2(e.major * dir.dot(e.frame.x)))
            }
            Curve::BSpline(b) => b.extremes(dir, t0, t1),
            Curve::Intersection(c) => c.approx.extremes(dir, t0, t1),
            Curve::Nurbs(c) => c.extremes(dir, t0, t1),
        }
    }
}

/// The parameter `t` of the point `(a cos t, b sin t)` nearest `(x, y)`, solved rather than
/// searched for.
///
/// The nearest point `(u, v)` has `(x, y)` on its normal: `x − u = λ·u/a²` and `y − v = λ·v/b²`
/// for one `λ`, so `u = a²x/(λ + a²)` and `v = b²y/(λ + b²)`. Putting those on the ellipse leaves
/// one equation, `F(λ) = (a·x/(λ + a²))² + (b·y/(λ + b²))² − 1 = 0`. With both of `x`, `y`
/// positive `F` falls steadily for `λ` above `−min(a, b)²`, so it has one root there and
/// bisection pins it to the last bit. The other quadrants follow by symmetry. On an axis the
/// equation loses a term, and the answer is the vertex on that axis, or, on the longer axis
/// inside the centres of curvature of its vertices, the point whose normal passes through.
pub(crate) fn ellipse_nearest(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let (px, py) = (x.abs(), y.abs());
    let (a2, b2) = (a * a, b * b);
    let (u, v) = if px == 0.0 && py == 0.0 {
        // The centre is nearest the ends of the shorter axis.
        if a <= b { (a, 0.0) } else { (0.0, b) }
    } else if py == 0.0 {
        // On the x axis: the normal from `(u, v)` meets it at `u·(a² − b²)/a²`.
        let foot = a * px / (a2 - b2);
        if a > b && foot < 1.0 {
            (a * foot, b * (1.0 - foot * foot).max(0.0).sqrt())
        } else {
            (a, 0.0)
        }
    } else if px == 0.0 {
        let foot = b * py / (b2 - a2);
        if b > a && foot < 1.0 {
            (a * (1.0 - foot * foot).max(0.0).sqrt(), b * foot)
        } else {
            (0.0, b)
        }
    } else {
        let f = |l: f64| (a * px / (l + a2)).powi(2) + (b * py / (l + b2)).powi(2) - 1.0;
        // F is positive just above −min(a, b)², and negative once λ passes the larger of the
        // two terms' own roots: each term is then below a half.
        let mut lo = -a2.min(b2);
        let mut hi = (std::f64::consts::SQRT_2 * a * px - a2)
            .max(std::f64::consts::SQRT_2 * b * py - b2)
            .max(lo);
        let mut l = hi;
        for _ in 0..1100 {
            l = 0.5 * (lo + hi);
            if l == lo || l == hi {
                break;
            }
            let at = f(l);
            if at > 0.0 {
                lo = l;
            } else if at < 0.0 {
                hi = l;
            } else {
                break;
            }
        }
        (a2 * px / (l + a2), b2 * py / (l + b2))
    };
    (v.copysign(y) / b).atan2(u.copysign(x) / a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle() -> Curve {
        let frame = Frame::new(Vec3::new(1.0, 2.0, 3.0), Vec3::Z, Vec3::X).unwrap();
        Curve::Circle(Circle { frame, radius: 5.0 })
    }

    #[test]
    fn circle_derivatives_match_differences() {
        let c = circle();
        let (_, d1, d2) = c.d2(0.7);
        let h = 1e-6;
        let fd = (c.eval(0.7 + h) - c.eval(0.7 - h)) / (2.0 * h);
        let fd2 = (c.eval(0.7 + h) - c.eval(0.7) * 2.0 + c.eval(0.7 - h)) / (h * h);
        assert!(d1.dist(fd) < 1e-8);
        assert!(d2.dist(fd2) < 1e-3);
    }

    #[test]
    fn circle_projection_respects_its_range() {
        let c = circle();
        let p = Vec3::new(1.0, 12.0, 3.0);
        assert!((c.project(p, 0.0, TAU) - PI / 2.0).abs() < 1e-15);
        assert!((c.project(p, 4.0, 7.0) - 7.0).abs() < 1e-15);
        assert!((c.project(p, 6.0, 8.0) - (PI / 2.0 + TAU)).abs() < 1e-14);
    }

    #[test]
    fn ellipse_projection_finds_a_nearest_point_just_across_its_start() {
        // The nearest point is at t = -0.08, a period from the range's end: the best of the
        // samples is the start, and a step clamped there stops short of it.
        let e = Curve::Ellipse(Ellipse {
            frame: Frame::WORLD,
            major: 5.0,
            minor: 3.0,
        });
        let want = TAU - 0.08;
        let p = e.eval(want);
        let t = e.project(p, 0.0, TAU);
        assert!(e.eval(t).dist(p) < 1e-12, "{t}");
        assert!((t - want).abs() < 1e-9, "{t}");
    }

    #[test]
    fn ellipse_projection_is_exact_beside_a_sharp_tip() {
        // A long thin ellipse, its two sides a hair apart near the tip: every point on it
        // projects to itself, wherever the samples fall.
        let e = Curve::Ellipse(Ellipse {
            frame: Frame::WORLD,
            major: 20.0,
            minor: 0.3,
        });
        for k in 0..400 {
            let t = TAU * k as f64 / 400.0 + 0.003;
            let p = e.eval(t);
            let s = e.project(p, 0.0, TAU);
            assert!(e.eval(s).dist(p) < 1e-12, "{t} → {s}");
            // Off the plane and off the curve the nearest point is unchanged by the height.
            let q = p + Vec3::Z * 3.0;
            assert!(e.eval(e.project(q, 0.0, TAU)).dist(p) < 1e-9, "{t}");
        }
        // Inside, near the major axis, the nearest points are a symmetric pair either side.
        let inside = Vec3::new(10.0, 0.0, 0.0);
        let s = e.project(inside, 0.0, TAU);
        let best = (0..100_000)
            .map(|k| e.eval(TAU * k as f64 / 100_000.0).dist(inside))
            .fold(f64::INFINITY, f64::min);
        assert!(e.eval(s).dist(inside) <= best + 1e-12);
    }

    #[test]
    fn ellipse_projection_and_extremes() {
        let frame = Frame::WORLD;
        let e = Curve::Ellipse(Ellipse {
            frame,
            major: 30.0,
            minor: 20.0,
        });
        let t = e.project(Vec3::new(40.0, 1.0, 0.0), 0.0, TAU);
        assert!(t.abs() < 0.1);
        let ext = e.extremes(Vec3::X, 0.0, TAU);
        assert_eq!(ext.len(), 1);
        assert!((ext[0] - PI).abs() < 1e-15);
    }

    /// A rational cubic winding three turns round z on radius 5, its weights uneven: rising 1 a
    /// turn and open, or bobbing and closed, its ends meeting.
    fn coil(closed: bool) -> Curve {
        let n = 60;
        let mut ctrl: Vec<Point3> = (0..n)
            .map(|i| {
                let a = TAU * 3.0 * i as f64 / (n - 1) as f64;
                let z = if closed { (a / 3.0).sin() } else { a / TAU };
                Vec3::new(5.0 * a.cos(), 5.0 * a.sin(), z)
            })
            .collect();
        if closed {
            ctrl[n - 1] = ctrl[0];
        }
        let knots: Vec<f64> = (0..n + 4)
            .map(|i| (i as f64 - 3.0).clamp(0.0, (n - 3) as f64))
            .collect();
        let weights = (0..n).map(|i| 1.0 + 0.3 * (i % 3) as f64).collect();
        Curve::Nurbs(Box::new(
            NurbsCurve::new(3, knots, ctrl, Some(weights)).unwrap(),
        ))
    }

    /// A curve's extremes along `dir` with nothing kept: the roots of its rate along `dir` over
    /// each stretch between breakpoints.
    fn extremes_afresh(c: &Curve, dir: Vec3, t0: f64, t1: f64) -> Vec<f64> {
        let mut cuts = vec![t0];
        cuts.extend(c.breakpoints(t0, t1));
        cuts.push(t1);
        let mut out = Vec::new();
        for w in cuts.windows(2) {
            let mut f = |t: f64| c.d1(t).1.dot(dir);
            out.extend(crate::numeric::roots(&mut f, w[0], w[1], 12));
        }
        out.retain(|&r| r > t0 && r < t1);
        out
    }

    fn bbox_afresh(c: &Curve, t0: f64, t1: f64) -> Aabb {
        let mut bb = Aabb::EMPTY;
        bb.add(c.eval(t0));
        bb.add(c.eval(t1));
        for ax in [Vec3::X, Vec3::Y, Vec3::Z] {
            for t in extremes_afresh(c, ax, t0, t1) {
                bb.add(c.eval(t));
            }
        }
        bb
    }

    fn same_box(a: &Aabb, b: &Aabb) -> bool {
        (0..3).all(|k| {
            a.min[k].to_bits() == b.min[k].to_bits() && a.max[k].to_bits() == b.max[k].to_bits()
        })
    }

    #[test]
    fn a_rational_splines_kept_extremes_give_the_box_worked_out_afresh() {
        for closed in [false, true] {
            let c = coil(closed);
            let (a, b) = c.domain();
            // Whole spans, stretches that end inside a span or lie within one, and, on the closed
            // coil, stretches a period on and across the join.
            let mut ranges = vec![
                (a, b),
                (a + 3.0, a + 11.0),
                (a + 3.3, b - 10.1),
                (a + 20.2, a + 20.7),
                (a + 5.0, a + 5.5),
            ];
            if closed {
                ranges.extend([(b - 4.0, b + 3.0), (b + 2.0, b + 9.0), (a - 6.5, a + 1.0)]);
            }
            // Twice over: the second pass reads every whole span from what the first kept.
            for pass in 0..2 {
                for &(t0, t1) in &ranges {
                    let bb = c.bbox(t0, t1);
                    assert!(same_box(&bb, &bbox_afresh(&c, t0, t1)), "{pass} {t0} {t1}");
                    for dir in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.6, 0.0, 0.8)] {
                        let got = c.extremes(dir, t0, t1);
                        let want = extremes_afresh(&c, dir, t0, t1);
                        assert_eq!(got.len(), want.len());
                        assert!(
                            got.iter()
                                .zip(&want)
                                .all(|(x, y)| x.to_bits() == y.to_bits())
                        );
                    }
                }
            }
            // The coil turns three times, so its box over all of it is the circle's.
            let bb = c.bbox(a, b);
            assert!(bb.min.x < -4.9 && bb.max.x > 4.9 && bb.min.y < -4.9 && bb.max.y > 4.9);
        }
    }

    #[test]
    fn a_moved_or_copied_spline_has_its_own_box() {
        let c = coil(false);
        let (a, b) = c.domain();
        let here = c.bbox(a, b);
        let parts = |c: &Curve| match c {
            Curve::Nurbs(n) => (
                n.knots().to_vec(),
                n.control_points().to_vec(),
                n.weights().map(<[f64]>::to_vec),
            ),
            _ => unreachable!(),
        };
        let built = |(knots, ctrl, weights)| {
            Curve::Nurbs(Box::new(NurbsCurve::new(3, knots, ctrl, weights).unwrap()))
        };
        // Moved after its extremes were kept: the box is the moved curve's, as built from its
        // parts with nothing kept.
        let t = Transform::rotation(Vec3::new(7.0, -2.0, 1.0), Vec3::new(1.0, 2.0, 2.0), 0.8);
        let moved = c.transformed(&t);
        let rebuilt = built(parts(&moved));
        let there = moved.bbox(a, b);
        assert!(same_box(&there, &rebuilt.bbox(a, b)));
        assert!(same_box(&there, &bbox_afresh(&rebuilt, a, b)));
        assert!(!same_box(&there, &here));
        for i in 0..=600 {
            let p = moved.eval(a + (b - a) * i as f64 / 600.0);
            assert!((0..3).all(|k| there.min[k] <= p[k] && p[k] <= there.max[k]));
        }
        // A copy reads the same box, and a curve with one control point elsewhere does not.
        assert!(same_box(&c.clone().bbox(a, b), &here));
        let (knots, mut ctrl, weights) = parts(&c);
        ctrl[20] = ctrl[20] * 3.0;
        let bulged = built((knots, ctrl, weights));
        assert!(same_box(&bulged.bbox(a, b), &bbox_afresh(&bulged, a, b)));
        assert!(bulged.bbox(a, b).diagonal() > here.diagonal() + 1.0);
    }

    #[test]
    fn projecting_by_pieces_finds_the_points_sampling_everywhere_finds() {
        // Points on a grid through and round the coil, over its whole range and over parts of it:
        // searching only the pieces whose boxes could hold a nearer point reaches the same least
        // distance as sampling every piece.
        for closed in [false, true] {
            let c = coil(closed);
            assert_eq!(c.period().is_some(), closed);
            let (a, b) = c.domain();
            let mut ranges = vec![(a, b), (a + 3.3, b - 10.1), (a + 20.0, a + 21.5)];
            if closed {
                // Across the closed coil's ends, where its parameter wraps round.
                ranges.push((b - 4.0, b + 3.0));
            }
            for (t0, t1) in ranges {
                for i in 0..9 {
                    for j in 0..9 {
                        for k in 0..5 {
                            let p = Vec3::new(
                                -7.0 + 1.75 * i as f64,
                                -7.0 + 1.75 * j as f64,
                                -0.5 + k as f64,
                            );
                            let full = c.eval(c.project_numeric(p, t0, t1)).dist(p);
                            let s = c.project(p, t0, t1);
                            assert!((t0..=t1).contains(&s));
                            let got = c.eval(s).dist(p);
                            assert!((got - full).abs() <= 1e-12, "{p:?}: {got} {full}");
                        }
                    }
                }
            }
        }
    }
}
