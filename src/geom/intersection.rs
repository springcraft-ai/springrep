//! Intersection curves of surface pairs with no closed form: a cubic B-spline through points
//! refined onto both surfaces, with enough points that the spline stays within tolerance of both.

use super::{BSpline, Surface, bspline::chord_params};
use crate::{
    Error, Result,
    math::{Point3, Transform, Vec2, Vec3},
    numeric::solve2,
    tol,
};

#[derive(Clone, Debug, PartialEq)]
pub struct IntersectionCurve {
    pub approx: BSpline,
    pub surfaces: [Surface; 2],
}

/// The distance from `p` to the nearest point of `s`.
pub(crate) fn distance_to(s: &Surface, p: Point3) -> f64 {
    let uv = s.project(p);
    s.eval(uv.x, uv.y).dist(p)
}

/// A point's parameters on each of two surfaces.
pub type UvPair = (Vec2, Vec2);

/// The point on both surfaces nearest `p`, by alternating tangent-plane steps. `None` where the
/// surfaces are tangent, so the step has no unique solution, or it does not converge.
pub(crate) fn refine_onto_both(a: &Surface, b: &Surface, p: Point3) -> Option<Point3> {
    refine(a, b, p, None, None).map(|r| r.0)
}

/// [`refine_onto_both`] from `hints`, the parameters of a point near `p` on each surface, and the
/// refined point's own parameters with it.
pub(crate) fn refine_onto_both_near(
    a: &Surface,
    b: &Surface,
    p: Point3,
    hints: Option<UvPair>,
) -> Option<(Point3, UvPair)> {
    refine_near(a, b, p, hints, None)
}

/// The point on both surfaces in the plane through `p` square to the unit `across`. Where a curve
/// turns sharply, the nearest point of the intersection can lie on another stretch of it; held to
/// the plane square to the curve's direction, the point stays between its neighbours.
pub(crate) fn refine_across(a: &Surface, b: &Surface, p: Point3, across: Vec3) -> Option<Point3> {
    refine(a, b, p, None, Some(across)).map(|r| r.0)
}

/// The points on both surfaces along the line through `x` across the curve: square to `dir` in
/// `a`'s tangent plane, out to `reach` either way, nearest first. Where the surfaces cross at a
/// tiny angle a Newton step's error is the gap's over that angle, and the steps wander; the gap
/// from `a` to `b` along the line changes sign at each crossing, and halving the bracket finds it
/// however small the angle.
fn cross_roots(
    a: &Surface,
    b: &Surface,
    x: Point3,
    dir: Vec3,
    hints: UvPair,
    reach: f64,
) -> Vec<(Point3, UvPair)> {
    let ua = a.project_near(x, hints.0);
    let w = a.normal(ua.x, ua.y).cross(dir).unit_or_zero();
    if w.norm() == 0.0 || reach.is_nan() || reach <= 0.0 {
        return Vec::new();
    }
    let at = |s: f64| {
        let ua = a.project_near(x + w * s, hints.0);
        let q = a.eval(ua.x, ua.y);
        let ub = b.project_near(q, hints.1);
        (
            (q - b.eval(ub.x, ub.y)).dot(b.normal(ub.x, ub.y)),
            q,
            (ua, ub),
        )
    };
    // A grid finer near `x`, a fifth wider each step, out to `reach` either way.
    let mut grid = vec![0.0];
    let mut s = 1e-6_f64.min(reach);
    while s <= reach {
        grid.push(s);
        grid.insert(0, -s);
        s *= 1.2;
    }
    let g: Vec<f64> = grid.iter().map(|&s| at(s).0).collect();
    let root = |mut lo: f64, mut hi: f64| {
        let below = at(lo).0 < 0.0;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if (at(mid).0 < 0.0) == below {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let mut found = Vec::new();
    for k in 0..grid.len() - 1 {
        if (g[k] < 0.0) != (g[k + 1] < 0.0) {
            found.push(root(grid[k], grid[k + 1]));
        } else if k > 0 && g[k].abs() < g[k - 1].abs() && g[k].abs() <= g[k + 1].abs() {
            // Two crossings close together (the two sides of a thin lens) can fall between
            // samples; the gap's least magnitude between the neighbours shows them.
            let (mut lo, mut hi) = (grid[k - 1], grid[k + 1]);
            let gr = 0.5 * (5f64.sqrt() - 1.0);
            for _ in 0..80 {
                let (x1, x2) = (hi - gr * (hi - lo), lo + gr * (hi - lo));
                if at(x1).0.abs() <= at(x2).0.abs() {
                    hi = x2;
                } else {
                    lo = x1;
                }
            }
            let sm = 0.5 * (lo + hi);
            if (at(sm).0 < 0.0) != (g[k] < 0.0) {
                found.push(root(grid[k - 1], sm));
                found.push(root(sm, grid[k + 1]));
            }
        }
    }
    let mut out: Vec<(f64, Point3, UvPair)> = found
        .into_iter()
        .map(|s| {
            let (_, q, uv) = at(s);
            (s.abs(), q, uv)
        })
        .collect();
    out.sort_by(|p, q| p.0.total_cmp(&q.0));
    out.into_iter().map(|(_, q, uv)| (q, uv)).collect()
}

/// The point of both surfaces about `step` on from the one at `uv`, found on `a`: the zero contour
/// of the gap from `a` to `b` over `a`'s parameters, reached by Newton steps along the gap's
/// gradient from the point `step` ahead. Where the surfaces are nearly tangent the nearest point
/// of both to a point ahead can lie across a thin lens; held to `a` and moved only square to the
/// contour, the step stays on its own side.
pub(crate) fn refine_on_contour(
    a: &Surface,
    b: &Surface,
    (ua, ub): UvPair,
    step: Vec3,
) -> Option<(Point3, UvPair)> {
    let (_, su, sv) = a.d1(ua.x, ua.y);
    let (du, dv) = solve2(
        su.dot(su),
        su.dot(sv),
        sv.dot(su),
        sv.dot(sv),
        su.dot(step),
        sv.dot(step),
    )?;
    // Past a bounded surface's parameter range its evaluation extrapolates, which no projection
    // onto it would return.
    let inside = |uv: Vec2| match a {
        Surface::Nurbs(s) => {
            let ((u0, u1), (v0, v1)) = s.domain();
            (u0..=u1).contains(&uv.x) && (v0..=v1).contains(&uv.y)
        }
        _ => true,
    };
    let mut uv = Vec2::new(ua.x + du, ua.y + dv);
    let mut hb = ub;
    // The gap is settled to rounding, not to the tolerance: across the contour a point is placed
    // only to the gap's error over its slope, the sine between the surfaces, which here is tiny.
    let mut best: Option<(f64, Point3, UvPair)> = None;
    for _ in 0..40 {
        if !inside(uv) {
            break;
        }
        let (q, su, sv) = a.d1(uv.x, uv.y);
        hb = b.project_near(q, hb);
        let qb = b.eval(hb.x, hb.y);
        let nb = b.normal(hb.x, hb.y);
        let g = (q - qb).dot(nb);
        // Off `b`'s edge its nearest point is on the boundary and the gap is not a distance.
        let on_b = q.dist(qb) <= g.abs() + 1e-3 * tol::LINEAR;
        if best.as_ref().is_some_and(|b| g.abs() >= b.0) {
            break;
        }
        best = on_b.then_some((g.abs(), q, (uv, hb)));
        if g.abs() <= 1e-14 * (1.0 + q.norm()) {
            break;
        }
        let grad = Vec2::new(su.dot(nb), sv.dot(nb));
        let g2 = grad.dot(grad);
        if g2 <= f64::MIN_POSITIVE {
            break;
        }
        uv = Vec2::new(uv.x - grad.x * g / g2, uv.y - grad.y * g / g2);
    }
    best.filter(|b| b.0 <= 1e-3 * tol::LINEAR)
        .map(|b| (b.1, b.2))
}

fn refine_near(
    a: &Surface,
    b: &Surface,
    p: Point3,
    hints: Option<UvPair>,
    across: Option<Vec3>,
) -> Option<(Point3, UvPair)> {
    // Where surfaces nearly coincide the hints can lead the steps astray; the global start is
    // the fallback.
    refine(a, b, p, hints, across).or_else(|| hints.and_then(|_| refine(a, b, p, None, across)))
}

fn refine(
    a: &Surface,
    b: &Surface,
    p: Point3,
    mut hints: Option<UvPair>,
    across: Option<Vec3>,
) -> Option<(Point3, UvPair)> {
    let mut x = p;
    for _ in 0..100 {
        let at = |s: &Surface, hint: Option<Vec2>| match hint {
            Some(h) => s.project_near_at(x, h),
            None => {
                let uv = s.project(x);
                (uv, s.eval(uv.x, uv.y), s.normal(uv.x, uv.y))
            }
        };
        let (ua, qa, na) = at(a, hints.map(|h| h.0));
        let (ub, qb, nb) = at(b, hints.map(|h| h.1));
        hints = Some((ua, ub));
        if qa.dist(x) <= 1e-3 * tol::LINEAR && qb.dist(x) <= 1e-3 * tol::LINEAR {
            return Some((x, (ua, ub)));
        }
        let (ra, rb) = (na.dot(qa - x), nb.dot(qb - x));
        let step = match across {
            None => {
                let (l1, l2) = solve2(na.dot(na), na.dot(nb), nb.dot(na), nb.dot(nb), ra, rb)?;
                na * l1 + nb * l2
            }
            Some(t) => {
                let det = na.dot(nb.cross(t));
                if det.abs() <= 1e-14 {
                    return None;
                }
                let rt = t.dot(p - x);
                (nb.cross(t) * ra + t.cross(na) * rb + na.cross(nb) * rt) * (1.0 / det)
            }
        };
        x += step;
        // A step that stalls off a surface means that surface's nearest point is on its boundary,
        // where its normal no longer points at `x`: there is no meeting there.
        if step.norm() <= 1e-15 * (1.0 + x.norm()) {
            let near = 1e-2 * tol::LINEAR;
            return (qa.dist(x) <= near && qb.dist(x) <= near).then_some((x, (ua, ub)));
        }
    }
    None
}

/// A point of a branch with its parameters on both surfaces, from which nearby points project.
#[derive(Clone, Copy)]
struct Node {
    p: Point3,
    uv: UvPair,
}

/// What a branch too short to fit a curve through is refused with.
pub(crate) const SHORT_BRANCH: &str = "an intersection branch is shorter than the tolerance";

/// Points that refine onto one place are one point; a branch shorter than the tolerance has no
/// curve to fit.
fn distinct(nodes: Vec<Node>, closed: bool) -> Result<Vec<Node>> {
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    let count = nodes.len();
    for (i, n) in nodes.into_iter().enumerate() {
        if out.last().is_none_or(|q| q.p.dist(n.p) > tol::LINEAR) {
            out.push(n);
        } else if !closed && i + 1 == count && out.len() > 1 {
            // An open curve ends at its last node, where it is cut off: the point beside it goes.
            *out.last_mut().expect("checked") = n;
        }
    }
    while closed && out.len() > 1 && out[0].p.dist(out[out.len() - 1].p) <= tol::LINEAR {
        out.pop();
    }
    if out.len() < if closed { 3 } else { 2 } {
        return Err(Error::Numerical(SHORT_BRANCH.into()));
    }
    Ok(out)
}

/// The nodes with each that would turn the polyline back on itself dropped. Where two surfaces
/// cross at a tiny angle θ, a point on both to within ε is placed across the curve only to within
/// ε/θ, so neighbouring points can zigzag across that band; an interpolant through them folds back
/// on itself. A traced branch never reverses between neighbouring points, so a node whose chords
/// to the nodes either side of it point against each other is scatter, and every point of the band
/// is on both surfaces.
fn forward(nodes: Vec<Node>, closed: bool) -> Result<Vec<Node>> {
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    for n in nodes {
        while out.len() >= 2 {
            let (p0, p1) = (out[out.len() - 2].p, out[out.len() - 1].p);
            if (p1 - p0).dot(n.p - p1) >= 0.0 {
                break;
            }
            out.pop();
        }
        out.push(n);
    }
    distinct(out, closed)
}

/// The distance from `p` to the span `[lo, hi]` of `c`, by Newton steps from the fraction `f` of it.
fn span_distance(c: &BSpline, p: Point3, (lo, hi): (f64, f64), f: f64) -> f64 {
    let mut t = lo + (hi - lo) * f;
    for _ in 0..8 {
        let [q, d1, d2] = c.eval_ders(t, 2);
        let r = q - p;
        let df = d1.norm2() + r.dot(d2);
        if df <= 0.0 {
            break;
        }
        let next = (t - r.dot(d1) / df).clamp(lo, hi);
        let done = (next - t).abs() <= 1e-14 * (hi - lo);
        t = next;
        if done {
            break;
        }
    }
    c.eval(t).dist(p)
}

/// Fitting is internal: a failure in it is numerical, not the caller's input.
fn numerical(e: Error) -> Error {
    match e {
        Error::InvalidInput(m) => Error::Numerical(format!("intersection curve: {m}")),
        other => other,
    }
}

impl IntersectionCurve {
    /// The curve through `points`, which lie near both surfaces in order along their
    /// intersection; `closed` joins the last back to the first.
    pub fn fit(
        points: &[Point3],
        a: Surface,
        b: Surface,
        closed: bool,
    ) -> Result<IntersectionCurve> {
        IntersectionCurve::fit_near(points, None, a, b, closed)
    }

    /// [`IntersectionCurve::fit`] with each point's parameters on the two surfaces, when known.
    pub fn fit_near(
        points: &[Point3],
        uvs: Option<&[UvPair]>,
        a: Surface,
        b: Surface,
        closed: bool,
    ) -> Result<IntersectionCurve> {
        let failed = || Error::Numerical("intersection point did not converge".into());
        let node = |p: Point3, near: Option<(Vec2, Vec2)>| {
            let uv = match near {
                Some((ha, hb)) => (a.project_near(p, ha), b.project_near(p, hb)),
                None => (a.project(p), b.project(p)),
            };
            Node { p, uv }
        };
        let gap = |m: Point3, near: (Vec2, Vec2)| {
            let n = node(m, Some(near));
            let (ua, ub) = n.uv;
            a.eval(ua.x, ua.y).dist(m).max(b.eval(ub.x, ub.y).dist(m))
        };
        // The direction the branch runs at a point.
        let heading = |n: &Node| {
            let (ua, ub) = n.uv;
            a.normal(ua.x, ua.y).cross(b.normal(ub.x, ub.y))
        };
        // Where the surfaces touch, refinement has no unique step; a point there already within
        // half the tolerance of both is as near as the fit needs.
        let on_both = |p: Point3| {
            distance_to(&a, p) <= 0.5 * tol::LINEAR && distance_to(&b, p) <= 0.5 * tol::LINEAR
        };
        // Where the surfaces lie within the tolerance of each other across a strip, the
        // intersection is anywhere in it; the point halfway between `x`'s feet on the two, settled
        // a few times, is on both when the strip reaches it.
        let between = |x: Point3, hint: Option<UvPair>| {
            let mut q = x;
            for _ in 0..4 {
                let (ua, ub) = node(q, hint).uv;
                q = a.eval(ua.x, ua.y).lerp(b.eval(ub.x, ub.y), 0.5);
            }
            on_both(q).then(|| node(q, hint))
        };
        // An open curve's ends are where it is cut off, at vertices it must reach: refinement slides
        // a point along the curve onto both surfaces, so an end already on both stays put.
        let last = points.len().saturating_sub(1);
        let mut nodes: Vec<Node> = points
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let hint = uvs.map(|u| u[i]);
                let exact = |p: Point3| {
                    distance_to(&a, p) <= 1e-2 * tol::LINEAR
                        && distance_to(&b, p) <= 1e-2 * tol::LINEAR
                };
                if !closed && (i == 0 || i == last) && exact(p) {
                    return Ok(node(p, hint));
                }
                refine_near(&a, &b, p, hint, None)
                    .map(|(p, uv)| Node { p, uv })
                    .or_else(|| on_both(p).then(|| node(p, hint)))
                    .or_else(|| between(p, hint))
                    .ok_or_else(failed)
            })
            .collect::<Result<_>>()?;
        for _ in 0..16 {
            nodes = forward(distinct(nodes, closed)?, closed)?;
            let pts: Vec<Point3> = nodes.iter().map(|n| n.p).collect();
            let approx = if closed {
                BSpline::interpolate_periodic(&pts)
            } else {
                BSpline::interpolate(&pts)
            }
            .map_err(numerical)?;
            let params = chord_params(&pts, closed).map_err(numerical)?;
            let n_seg = if closed { pts.len() } else { pts.len() - 1 };
            let next_of = |i: usize| (i + 1) % pts.len();
            let err: Vec<f64> = (0..n_seg)
                .map(|i| {
                    let (lo, hi) = (params[i], params[i + 1]);
                    [0.25, 0.5, 0.75]
                        .iter()
                        .map(|&f| {
                            let near = if f < 0.5 {
                                &nodes[i]
                            } else {
                                &nodes[next_of(i)]
                            };
                            gap(approx.eval(lo + (hi - lo) * f), near.uv)
                        })
                        .fold(0.0, f64::max)
                })
                .collect();
            if err.iter().all(|&e| e <= 0.5 * tol::LINEAR) {
                return Ok(IntersectionCurve {
                    approx,
                    surfaces: [a, b],
                });
            }
            // Spans near failing are split with the failing ones, and so are their neighbours:
            // splitting only the worst leaves uneven spacing, which moves the interpolant's error
            // into the spans beside it, one round at a time.
            let at = |i: usize, d: isize| {
                let k = i as isize + d;
                if closed {
                    Some(k.rem_euclid(n_seg as isize) as usize)
                } else {
                    (0..n_seg as isize).contains(&k).then_some(k as usize)
                }
            };
            let mut split: Vec<bool> = (0..n_seg)
                .map(|i| {
                    [-1, 0, 1]
                        .iter()
                        .any(|&d| at(i, d).is_some_and(|k| err[k] > 0.125 * tol::LINEAR))
                })
                .collect();
            // A span far longer than a split one beside it makes the interpolant swing across it,
            // so it is split too, until no span is over three times a split neighbour's halves.
            let span = |i: usize| pts[i].dist(pts[next_of(i)]);
            loop {
                let grown: Vec<usize> = (0..n_seg)
                    .filter(|&i| {
                        !split[i]
                            && [-1, 1].iter().any(|&d| {
                                at(i, d).is_some_and(|k| split[k] && span(i) > 1.5 * span(k))
                            })
                    })
                    .collect();
                if grown.is_empty() {
                    break;
                }
                for i in grown {
                    split[i] = true;
                }
            }
            let mut next = Vec::with_capacity(nodes.len() * 2);
            for i in 0..n_seg {
                next.push(nodes[i]);
                if !split[i] {
                    continue;
                }
                let (n0, n1) = (&nodes[i], &nodes[next_of(i)]);
                let [m, d, _] = approx.eval_ders(0.5 * (params[i] + params[i + 1]), 1);
                let (h0, h1) = (heading(n0), heading(n1));
                // Where the surfaces are nearly tangent, two branches bound a thin lens between
                // them, and a point off the curve by little is near both; the other branch runs
                // the other way, so a point heading against its neighbours is on it.
                let along = |n: &Node| {
                    let h = heading(n);
                    [h0, h1].iter().all(|e| {
                        e.norm() <= tol::TANGENT || h.norm() <= tol::TANGENT || h.dot(*e) >= 0.0
                    })
                };
                let tries = [
                    (m, Some(d)),
                    (n0.p.lerp(n1.p, 0.5), Some(n1.p - n0.p)),
                    (m, None),
                ];
                // From the neighbour's parameters first; from scratch where those settle on the
                // other branch of a lens.
                let q = [Some(n0.uv), None]
                    .into_iter()
                    .flat_map(|hint| tries.iter().map(move |&(x, dir)| (x, dir, hint)))
                    .filter_map(|(x, dir, hint)| {
                        refine(&a, &b, x, hint, dir.map(|d| d.unit_or_zero()))
                    })
                    .map(|(p, uv)| Node { p, uv })
                    .find(along)
                    .or_else(|| {
                        cross_roots(&a, &b, m, d.unit_or_zero(), n0.uv, n0.p.dist(n1.p))
                            .into_iter()
                            .map(|(p, uv)| Node { p, uv })
                            .filter(|n| on_both(n.p))
                            .find(along)
                    })
                    .or_else(|| between(m, Some(n0.uv)))
                    .ok_or_else(failed)?;
                next.push(q);
            }
            if !closed {
                next.push(nodes[nodes.len() - 1]);
            }
            nodes = next;
        }
        Err(Error::Numerical(
            "intersection curve did not reach tolerance".into(),
        ))
    }

    /// The curve through `sample(s)` for `s` over `range`, where `sample` gives points exactly on
    /// both surfaces (a closed form of the intersection): points are added at the middle parameter
    /// of every span that strays from either surface by more than half the tolerance, so nothing
    /// is refined numerically and a touching point inside the range is never lost to a step.
    pub fn fit_sampled(
        sample: &dyn Fn(f64) -> Point3,
        range: (f64, f64),
        a: Surface,
        b: Surface,
        closed: bool,
        n0: usize,
    ) -> Result<IntersectionCurve> {
        let n0 = n0.max(if closed { 3 } else { 1 });
        let at = |k: usize, n: usize| range.0 + (range.1 - range.0) * k as f64 / n as f64;
        let mut ss: Vec<f64> = (0..n0).map(|k| at(k, n0)).collect();
        if !closed {
            ss.push(range.1);
        }
        let gap = |p: Point3| distance_to(&a, p).max(distance_to(&b, p));
        for _ in 0..24 {
            let mut pts: Vec<Point3> = Vec::with_capacity(ss.len());
            let mut keep: Vec<f64> = Vec::with_capacity(ss.len());
            for &s in &ss {
                let p = sample(s);
                if pts.last().is_none_or(|q: &Point3| q.dist(p) > tol::LINEAR) {
                    pts.push(p);
                    keep.push(s);
                }
            }
            if closed {
                while pts.len() > 1 && pts[0].dist(pts[pts.len() - 1]) <= tol::LINEAR {
                    pts.pop();
                    keep.pop();
                }
            } else if keep.len() > 1 && keep[keep.len() - 1] != range.1 {
                // The end point merged into its neighbour: the end is the one kept.
                let n = keep.len();
                keep[n - 1] = range.1;
                pts[n - 1] = sample(range.1);
            }
            if pts.len() < if closed { 3 } else { 2 } {
                return Err(Error::Numerical(SHORT_BRANCH.into()));
            }
            ss = keep;
            if ss.len() > 1 << 14 {
                break;
            }
            let approx = if closed {
                BSpline::interpolate_periodic(&pts)
            } else {
                BSpline::interpolate(&pts)
            }
            .map_err(numerical)?;
            let params = chord_params(&pts, closed).map_err(numerical)?;
            let n_seg = if closed { pts.len() } else { pts.len() - 1 };
            let err: Vec<f64> = (0..n_seg)
                .map(|i| {
                    let (lo, hi) = (params[i], params[i + 1]);
                    let (s0, s1) = (ss[i], ss.get(i + 1).copied().unwrap_or(range.1));
                    [0.25, 0.5, 0.75]
                        .iter()
                        .map(|&f| {
                            let off = gap(approx.eval(lo + (hi - lo) * f));
                            // Where the surfaces nearly touch, a point within the tolerance of
                            // both can still lie far along the band from the curve: the closed
                            // form is the curve, so the spline must pass within it of the form.
                            let exact = sample(s0 + (s1 - s0) * f);
                            off.max(span_distance(&approx, exact, (lo, hi), f))
                        })
                        .fold(0.0, f64::max)
                })
                .collect();
            if err.iter().all(|&e| e <= 0.5 * tol::LINEAR) {
                return Ok(IntersectionCurve {
                    approx,
                    surfaces: [a, b],
                });
            }
            // As in `fit_near`: neighbours of a failing span split with it, so spacing stays even.
            let near = |i: usize| {
                [-1isize, 0, 1].iter().any(|&d| {
                    let k = i as isize + d;
                    let k = if closed {
                        Some(k.rem_euclid(n_seg as isize) as usize)
                    } else {
                        (0..n_seg as isize).contains(&k).then_some(k as usize)
                    };
                    k.is_some_and(|k| err[k] > 0.125 * tol::LINEAR)
                })
            };
            let mut next = Vec::with_capacity(2 * ss.len());
            for i in 0..n_seg {
                next.push(ss[i]);
                if near(i) {
                    let hi = if i + 1 < ss.len() { ss[i + 1] } else { range.1 };
                    next.push(0.5 * (ss[i] + hi));
                }
            }
            if !closed {
                next.push(range.1);
            }
            ss = next;
        }
        Err(Error::Numerical(
            "intersection curve did not reach tolerance".into(),
        ))
    }

    pub fn transformed(&self, t: &Transform) -> IntersectionCurve {
        IntersectionCurve {
            approx: self.approx.transformed(t),
            surfaces: [
                self.surfaces[0].transformed(t),
                self.surfaces[1].transformed(t),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geom::{Cylinder, Surface},
        math::{Frame, Vec3},
    };

    #[test]
    fn crossing_cylinders_fit_within_tolerance() {
        let a = Surface::Cylinder(Cylinder {
            frame: Frame::WORLD,
            radius: 5.0,
        });
        let fx = Frame::new(Vec3::ZERO, Vec3::X, Vec3::Y).unwrap();
        let b = Surface::Cylinder(Cylinder {
            frame: fx,
            radius: 3.0,
        });
        // The upper loop where the Ø6 bore along x pierces the Ø10 bore along z.
        let pts: Vec<Vec3> = (0..12)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 12.0;
                let (y, z) = (3.0 * t.cos(), 3.0 * t.sin());
                Vec3::new((25.0 - y * y).sqrt(), y, z)
            })
            .collect();
        // Points that refine onto one place, as marching near an end leaves, are one point.
        let mut doubled = pts.clone();
        doubled.insert(3, pts[3] + Vec3::new(0.0, 0.0, 1e-9));
        doubled.push(pts[0]);
        let c = IntersectionCurve::fit(&doubled, a.clone(), b.clone(), true).unwrap();
        let (t0, t1) = c.approx.domain();
        for i in 0..200 {
            let p = c.approx.eval(t0 + (t1 - t0) * i as f64 / 200.0);
            assert!(distance_to(&a, p) < tol::LINEAR, "{}", distance_to(&a, p));
            assert!(distance_to(&b, p) < tol::LINEAR);
        }
    }
}
