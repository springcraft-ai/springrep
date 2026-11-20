//! Where a moving body's faces graze their motion: the characteristic curves along which the
//! velocity lies in the tangent plane, `V·n = 0` (Wang and Wang 1986). Each is found from where it
//! meets its face's boundary, traced across the face by predictor steps along it and Newton steps
//! back onto it, and fitted with a cubic B-spline, or made the exact line or circle its points lie
//! on.

use crate::{
    Error, Result,
    boolean::domain::Domain,
    geom::{BSpline, Circle, Curve, Line, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    numeric::solve2,
    tol,
};

/// How far a fitted curve may stray from the characteristic curve.
pub(super) const FIT: f64 = tol::LINEAR / 10.0;

/// The rate a face's surface moves outward: its outward normal dotted with the velocity.
pub(super) struct Rate<'a> {
    pub surface: &'a Surface,
    /// 1 where the outward normal is the surface's natural normal, −1 where it is the opposite.
    pub out: f64,
    pub field: &'a dyn Fn(Point3) -> Vec3,
    /// The body's size, which sets the steps derivatives are taken over.
    pub size: f64,
}

impl Rate<'_> {
    pub(super) fn at(&self, uv: Vec2) -> f64 {
        let p = self.surface.eval(uv.x, uv.y);
        self.out * self.surface.normal(uv.x, uv.y).dot((self.field)(p))
    }

    /// The rate's gradient in the parameters, by central differences a millionth of the body's
    /// size apart on the surface.
    fn gradient(&self, uv: Vec2) -> Vec2 {
        let (_, su, sv) = self.surface.d1(uv.x, uv.y);
        let step = |d: Vec3| 1e-6 * self.size / d.norm().max(1e-300);
        let (hu, hv) = (step(su), step(sv));
        let du = (self.at(uv + Vec2::new(hu, 0.0)) - self.at(uv - Vec2::new(hu, 0.0))) / (2.0 * hu);
        let dv = (self.at(uv + Vec2::new(0.0, hv)) - self.at(uv - Vec2::new(0.0, hv))) / (2.0 * hv);
        Vec2::new(du, dv)
    }

    /// The parameter step that moves a point on the surface the least distance onto the zero
    /// contour, to first order: along the gradient taken with the surface's metric.
    fn newton(&self, uv: Vec2) -> Option<Vec2> {
        let g = self.at(uv);
        let grad = self.gradient(uv);
        let (_, su, sv) = self.surface.d1(uv.x, uv.y);
        let (e, f, h) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let (a, b) = solve2(e, f, f, h, grad.x, grad.y)?;
        let denom = grad.x * a + grad.y * b;
        if denom.abs() <= f64::MIN_POSITIVE {
            return None;
        }
        Some(Vec2::new(-g * a / denom, -g * b / denom))
    }

    /// The point of the zero contour nearest `uv`, by Newton steps.
    pub(super) fn settle(&self, uv: Vec2) -> Option<Vec2> {
        let mut uv = uv;
        let tiny = 1e-15 * self.size;
        for _ in 0..60 {
            let step = self.newton(uv)?;
            uv = uv + step;
            let (_, su, sv) = self.surface.d1(uv.x, uv.y);
            if (su * step.x + sv * step.y).norm() <= tiny {
                return Some(uv);
            }
        }
        let (_, su, sv) = self.surface.d1(uv.x, uv.y);
        let grad = self.gradient(uv);
        let slope = (su * grad.x + sv * grad.y).norm().max(f64::MIN_POSITIVE);
        (self.at(uv).abs() / slope * (su.norm() + sv.norm()) <= 1e-3 * tol::LINEAR).then_some(uv)
    }

    /// The contour's direction at `uv` in the parameter plane, scaled to unit length on the
    /// surface, turned to agree with `prev` where given.
    fn tangent(&self, uv: Vec2, prev: Option<Vec3>) -> Option<(Vec2, Vec3)> {
        let grad = self.gradient(uv);
        let t = Vec2::new(-grad.y, grad.x);
        let (_, su, sv) = self.surface.d1(uv.x, uv.y);
        let d = su * t.x + sv * t.y;
        let len = d.norm();
        if len <= f64::MIN_POSITIVE {
            return None;
        }
        let (t, d) = (t / len, d / len);
        match prev {
            Some(p) if p.dot(d) < 0.0 => Some((-t, -d)),
            _ => Some((t, d)),
        }
    }
}

/// A traced characteristic curve: its points on the contour in order, each with its parameters.
pub(super) struct Traced {
    pub points: Vec<Point3>,
    pub uvs: Vec<Vec2>,
}

/// Where a trace ended.
pub(super) enum Arrival {
    /// At the root of this index among those the trace was given.
    Root(usize),
    /// Across the boundary somewhere no root was given, at this point: where the contour runs
    /// into an edge the rate vanishes all along.
    Boundary(Point3),
}

/// Whether two parameter points of `s` are within a quarter period of each other in each periodic
/// parameter: one image of a point, and not another a period away.
pub(super) fn near_image(s: &Surface, a: Vec2, b: Vec2) -> bool {
    let d = a - b;
    s.u_period().is_none_or(|p| d.x.abs() < 0.25 * p)
        && s.v_period().is_none_or(|p| d.y.abs() < 0.25 * p)
}

/// Traces the contour from `start` into the face, until it reaches the face's boundary. `roots`
/// are the places it may end, with their parameters on this face; `from` is the one it starts at,
/// which it may not end at.
pub(super) fn trace(
    rate: &Rate,
    dom: &Domain,
    roots: &[(Point3, Vec2)],
    from: usize,
) -> Result<(Traced, Arrival)> {
    let size = rate.size;
    let failed = |why: &str| Error::Numerical(format!("a characteristic curve {why}"));
    // The trace keeps the parameters it starts with, continued, so a curve that runs round a
    // period leaves the face's window at the seam it ends on instead of reading as inside again.
    let inside = |uv: Vec2| dom.winding(uv) != 0;
    let (p0, uv0) = roots[from];
    // The first point: where the contour crosses a small circle about the start inside the face,
    // found by bisection between samples of the rate round it.
    let first = {
        let (_, su, sv) = rate.surface.d1(uv0.x, uv0.y);
        let r = 1e-3 * size;
        let n = 96;
        let at = |a: f64| -> Vec2 {
            let (c, s) = (a.cos(), a.sin());
            // A unit step on the surface along the angle a in the tangent plane.
            let dir = (su.unit_or_zero() * c) + (su.cross(sv).cross(su).unit_or_zero() * s);
            let (du, dv) = solve2(
                su.dot(su),
                su.dot(sv),
                su.dot(sv),
                sv.dot(sv),
                su.dot(dir),
                sv.dot(dir),
            )
            .unwrap_or((0.0, 0.0));
            uv0 + Vec2::new(du, dv) * r
        };
        let angle = |k: usize| std::f64::consts::TAU * k as f64 / n as f64;
        let gs: Vec<f64> = (0..=n).map(|k| rate.at(at(angle(k)))).collect();
        let mut found: Option<Vec2> = None;
        for k in 0..n {
            let (g0, g1) = (gs[k], gs[k + 1]);
            let uv = if g0 == 0.0 {
                at(angle(k))
            } else if g1 != 0.0 && g0.signum() != g1.signum() {
                let (mut lo, mut hi) = (angle(k), angle(k + 1));
                for _ in 0..60 {
                    let m = 0.5 * (lo + hi);
                    if rate.at(at(m)).signum() == g0.signum() {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                at(0.5 * (lo + hi))
            } else {
                continue;
            };
            if inside(uv) {
                found = Some(uv);
                break;
            }
        }
        found.ok_or_else(|| failed("leaves no root into its face"))?
    };
    let first = rate.settle(first).unwrap_or(first);
    let mut uvs = vec![uv0, first];
    let mut points = vec![p0, rate.surface.eval(first.x, first.y)];
    let mut prev = (points[1] - points[0]).unit();
    let mut h = 2e-3 * size;
    let h_max = size / 24.0;
    let h_min = 1e-9 * size;
    let mut length = 0.0;
    loop {
        let uv = *uvs.last().expect("a trace has points");
        let p = *points.last().expect("a trace has points");
        // A root within reach ahead ends the trace there. A root on a seam is two places, a period
        // apart in the parameters, and the trace ends at the one its own parameters have come to.
        let ahead = roots
            .iter()
            .enumerate()
            .filter(|&(i, &(q, quv))| {
                i != from
                    && q.dist(p) <= 1.5 * h
                    && prev.is_none_or(|d| (q - p).dot(d) > 0.5 * q.dist(p))
                    && near_image(rate.surface, quv, uv)
            })
            .min_by(|a, b| (a.1.1 - uv).norm().total_cmp(&(b.1.1 - uv).norm()));
        if let Some((i, &(q, quv))) = ahead {
            uvs.push(quv);
            points.push(q);
            return Ok((Traced { points, uvs }, Arrival::Root(i)));
        }
        let Some((t, d)) = rate.tangent(uv, prev) else {
            return Err(failed("reaches a point where it crosses another"));
        };
        let mut step = h;
        let next = loop {
            if step < h_min {
                return Err(failed("does not settle onto its contour"));
            }
            let guess = uv + t * step;
            let Some(got) = rate.settle(guess) else {
                step *= 0.5;
                continue;
            };
            let q = rate.surface.eval(got.x, got.y);
            let moved = q.dist(p);
            let Some((_, dq)) = rate.tangent(got, Some(d)) else {
                step *= 0.5;
                continue;
            };
            // Small turns between points, and points about the step apart.
            if dq.dot(d) < 0.995 || moved > 1.5 * step || moved < 0.5 * step {
                step *= 0.5;
                continue;
            }
            break (got, q, dq);
        };
        let (got, q, dq) = next;
        if !inside(got) {
            // Out of the face: a root near the step ends it, or else the step is shortened until
            // the crossing is pinned and the contour is carried on to the boundary.
            let reach = |r: Point3| segment_distance(r, p, q);
            if let Some((i, &(r, ruv))) = roots
                .iter()
                .enumerate()
                .filter(|&(i, &(r, ruv))| {
                    i != from
                        && reach(r) <= step.max(1e-6 * size)
                        && near_image(rate.surface, ruv, uv)
                })
                .min_by(|a, b| reach(a.1.0).total_cmp(&reach(b.1.0)))
            {
                uvs.push(ruv);
                points.push(r);
                return Ok((Traced { points, uvs }, Arrival::Root(i)));
            }
            if step <= 1e-7 * size {
                return Ok((Traced { points, uvs }, Arrival::Boundary(q)));
            }
            h = 0.25 * step;
            continue;
        }
        length += q.dist(p);
        if length > 200.0 * size || points.len() > 200_000 {
            return Err(failed("runs on without reaching its face's boundary"));
        }
        uvs.push(got);
        points.push(q);
        prev = Some(dq);
        h = (step * 1.5).min(h_max);
    }
}

/// The distance from `r` to the segment `pq`.
fn segment_distance(r: Point3, p: Point3, q: Point3) -> f64 {
    let d = q - p;
    let l2 = d.dot(d);
    if l2 <= f64::MIN_POSITIVE {
        return r.dist(p);
    }
    let s = ((r - p).dot(d) / l2).clamp(0.0, 1.0);
    r.dist(p + d * s)
}

/// The curve through a traced contour, and its parameter range: the line or circle its points lie
/// on, or a cubic B-spline through them with points added until it stays within [`FIT`] of the
/// contour.
pub(super) fn fit(rate: &Rate, traced: &Traced) -> Result<(Curve, (f64, f64))> {
    let pts = &traced.points;
    let (a, z) = (pts[0], pts[pts.len() - 1]);
    let exact = 1e-12 * rate.size;
    if let Some((line, len)) = Line::through(a, z)
        && pts.iter().all(|&p| (p - a).cross(line.dir).norm() <= exact)
    {
        return Ok((Curve::Line(line), (0.0, len)));
    }
    if let Some(arc) = on_circle(pts, exact) {
        return Ok(arc);
    }
    // A curve that comes round to where it started, through a seam, closes by its own period.
    let closed = a.dist(z) <= exact;
    let mut pts: Vec<Point3> = pts.clone();
    let mut uvs: Vec<Vec2> = traced.uvs.clone();
    for _ in 0..12 {
        let params = chords(&pts);
        let spline = if closed {
            BSpline::interpolate_periodic_at(&pts[..pts.len() - 1], &params)?
        } else {
            BSpline::interpolate_at(&pts, &params)?
        };
        let mut added = false;
        let mut next_pts = vec![pts[0]];
        let mut next_uvs = vec![uvs[0]];
        for i in 0..pts.len() - 1 {
            let mid = 0.5 * (params[i] + params[i + 1]);
            let m = spline.eval(mid);
            let guess = rate.surface.project_near(m, uvs[i]);
            if let Some(on) = rate.settle(guess) {
                let q = rate.surface.eval(on.x, on.y);
                if q.dist(m) > FIT {
                    next_pts.push(q);
                    next_uvs.push(on);
                    added = true;
                }
            }
            next_pts.push(pts[i + 1]);
            next_uvs.push(uvs[i + 1]);
        }
        if !added {
            let range = (params[0], params[params.len() - 1]);
            return Ok((Curve::BSpline(spline), range));
        }
        pts = next_pts;
        uvs = next_uvs;
    }
    Err(Error::Numerical(
        "a characteristic curve's fit did not reach its tolerance".into(),
    ))
}

/// The curve through the exact points `at(s)` for `s` over `range`, which starts and ends at
/// `ends`: the line or circle they lie on, or a cubic B-spline through them at chord-length
/// parameters, a point added in each span whose middle strays from the spline by more than [`FIT`].
pub(super) fn fit_exact(
    at: &dyn Fn(f64) -> Point3,
    range: (f64, f64),
    ends: (Point3, Point3),
    size: f64,
) -> Result<(Curve, (f64, f64))> {
    // Each round's middles are the next round's points: each point is found once.
    let seen: std::cell::RefCell<std::collections::HashMap<u64, Point3>> = Default::default();
    let at = |s: f64| -> Point3 {
        if let Some(&p) = seen.borrow().get(&s.to_bits()) {
            return p;
        }
        let p = at(s);
        seen.borrow_mut().insert(s.to_bits(), p);
        p
    };
    let n0 = 32;
    let mut ss: Vec<f64> = (0..=n0)
        .map(|k| range.0 + (range.1 - range.0) * k as f64 / n0 as f64)
        .collect();
    let exact = 1e-12 * size;
    let closed = ends.0.dist(ends.1) <= exact;
    for round in 0..20 {
        let last = ss.len() - 1;
        let pts: Vec<Point3> = ss
            .iter()
            .enumerate()
            .map(|(i, &s)| match i {
                0 => ends.0,
                i if i == last => ends.1,
                _ => at(s),
            })
            .collect();
        if round == 0 {
            if let Some((line, len)) = Line::through(ends.0, ends.1)
                && pts
                    .iter()
                    .all(|&p| (p - ends.0).cross(line.dir).norm() <= exact)
            {
                return Ok((Curve::Line(line), (0.0, len)));
            }
            if let Some(arc) = on_circle(&pts, exact) {
                return Ok(arc);
            }
        }
        let params = chords(&pts);
        let spline = if closed {
            BSpline::interpolate_periodic_at(&pts[..last], &params)?
        } else {
            BSpline::interpolate_at(&pts, &params)?
        };
        // Each new point moves the whole interpolant a little, so spans near the limit are split
        // too, or the refinement creeps on one span a round.
        let errs: Vec<(f64, f64)> = (0..last)
            .map(|i| {
                let mid = 0.5 * (ss[i] + ss[i + 1]);
                (
                    mid,
                    span_distance(&spline, at(mid), (params[i], params[i + 1])),
                )
            })
            .collect();
        let added = errs.iter().any(|e| e.1 > FIT);
        let mut next = vec![ss[0]];
        for i in 0..last {
            if added && errs[i].1 > 0.25 * FIT {
                next.push(errs[i].0);
            }
            next.push(ss[i + 1]);
        }
        if !added {
            return Ok((Curve::BSpline(spline), (params[0], params[last])));
        }
        ss = next;
    }
    Err(Error::Numerical(
        "a characteristic curve's fit did not reach its tolerance".into(),
    ))
}

/// The distance from `p` to the span `[lo, hi]` of `c`, by Newton steps from its middle.
fn span_distance(c: &BSpline, p: Point3, (lo, hi): (f64, f64)) -> f64 {
    let mut t = 0.5 * (lo + hi);
    for _ in 0..12 {
        let [q, d1, d2] = c.eval_ders(t, 2);
        let r = q - p;
        let df = d1.norm2() + r.dot(d2);
        if df <= 0.0 {
            break;
        }
        t = (t - r.dot(d1) / df).clamp(lo, hi);
    }
    c.eval(t).dist(p)
}

/// Chord-length parameters.
fn chords(pts: &[Point3]) -> Vec<f64> {
    let mut out = Vec::with_capacity(pts.len());
    let mut s = 0.0;
    out.push(0.0);
    for w in pts.windows(2) {
        s += w[0].dist(w[1]);
        out.push(s);
    }
    out
}

/// The arc of the circle through the points, where they all lie on one within `exact`.
fn on_circle(pts: &[Point3], exact: f64) -> Option<(Curve, (f64, f64))> {
    if pts.len() < 3 {
        return None;
    }
    let (a, m, z) = (pts[0], pts[pts.len() / 2], pts[pts.len() - 1]);
    // A closed curve's ends meet: the circle is then through its first, third and two-thirds points.
    let (a, m, z) = if a.dist(z) <= exact {
        (pts[0], pts[pts.len() / 3], pts[2 * pts.len() / 3])
    } else {
        (a, m, z)
    };
    let (ab, ac) = (m - a, z - a);
    let n = ab.cross(ac);
    let nn = n.dot(n);
    if nn <= f64::MIN_POSITIVE {
        return None;
    }
    let centre = a + (n.cross(ab) * ac.dot(ac) + ac.cross(n) * ab.dot(ab)) / (2.0 * nn);
    let radius = centre.dist(a);
    let normal = n.unit()?;
    if pts.iter().any(|&p| {
        ((p - centre).norm() - radius).abs() > exact || (p - centre).dot(normal).abs() > exact
    }) {
        return None;
    }
    let frame = Frame::new(centre, normal, pts[0] - centre).ok()?;
    let circle = Circle { frame, radius };
    // The angle each point is at, unwrapped along the points.
    let mut angle = 0.0;
    let mut last = pts[0] - centre;
    for &p in &pts[1..] {
        let q = p - centre;
        angle += last.cross(q).dot(normal).atan2(last.dot(q));
        last = q;
    }
    if angle <= 0.0 {
        return None;
    }
    Some((Curve::Circle(circle), (0.0, angle)))
}
