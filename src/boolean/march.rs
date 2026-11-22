//! The general surface–surface intersector: from each seed point on both surfaces, march along
//! `n_a × n_b` with Newton refinement onto both, until the branch closes, leaves the region, or
//! reaches a point where the surfaces are tangent; then fit an intersection curve to the points.

use super::ssi::SsiCurve;
use crate::{
    Error, Result,
    geom::{Curve, IntersectionCurve, Surface, UvPair, distance_to, refine_across},
    geom::{refine_on_contour, refine_onto_both_near},
    math::{Aabb, Point3, Vec2, Vec3},
    numeric::solve2,
    tol,
};

/// The largest turn of the curve's tangent across one step, in radians.
const MAX_TURN: f64 = 0.1;

/// The sine of the angle between the surfaces below which a step is refined from scratch rather
/// than from the last point's parameters: where they nearly coincide, refinement from nearby
/// parameters settles anywhere along them.
const HINTED_SIN: f64 = 0.05;

/// The parameters to refine the next step from, where the surfaces cross clearly enough there.
fn hint(a: &Surface, b: &Surface, (ua, ub): UvPair) -> Option<UvPair> {
    let sin = a.normal(ua.x, ua.y).cross(b.normal(ub.x, ub.y)).norm();
    (sin >= HINTED_SIN).then_some((ua, ub))
}

fn params(a: &Surface, b: &Surface, p: Point3) -> UvPair {
    (a.project(p), b.project(p))
}

/// Where the intersection through `p`, at parameters `uv`, meets the edge of a bounded surface's
/// parameter domain (a spline's knot range) within `reach` of `p`: on that edge's iso-curve, at the
/// point the other surface passes through, found by Newton's method along it.
fn onto_domain_edge(
    a: &Surface,
    b: &Surface,
    p: Point3,
    uv: UvPair,
    reach: f64,
) -> Option<(Point3, UvPair)> {
    for (s, other, at) in [(a, b, uv.0), (b, a, uv.1)] {
        let Surface::Nurbs(n) = s else { continue };
        let ((u0, u1), (v0, v1)) = n.domain();
        let (_, su, sv) = s.d1(at.x, at.y);
        for (along_u, bound) in [(true, u0), (true, u1), (false, v0), (false, v1)] {
            let (cur, speed) = if along_u {
                (at.x, su.norm())
            } else {
                (at.y, sv.norm())
            };
            if (cur - bound).abs() * speed > reach {
                continue;
            }
            let iso = |t: f64| {
                if along_u {
                    s.eval(bound, t)
                } else {
                    s.eval(t, bound)
                }
            };
            let (lo, hi) = if along_u { (v0, v1) } else { (u0, u1) };
            let gap = |t: f64| super::csi::signed_distance(other, iso(t));
            let mut t = if along_u { at.y } else { at.x };
            let h = 1e-7 * (hi - lo);
            for _ in 0..50 {
                let f = gap(t);
                if f.abs() <= 1e-3 * tol::LINEAR {
                    break;
                }
                let df = (gap((t + h).min(hi)) - gap((t - h).max(lo))) / (2.0 * h);
                if df == 0.0 || !df.is_finite() {
                    break;
                }
                t = (t - f / df).clamp(lo, hi);
            }
            let q = iso(t);
            if gap(t).abs() <= 1e-2 * tol::LINEAR && q.dist(p) <= reach {
                return Some((q, params(a, b, q)));
            }
        }
    }
    None
}

/// The direction of the intersection at a point of it, from its parameters on both surfaces.
fn tangent(a: &Surface, b: &Surface, (ua, ub): UvPair) -> Option<Vec3> {
    let n = a.normal(ua.x, ua.y).cross(b.normal(ub.x, ub.y));
    // Where rounding leaves the cross product of unit normals no direction: an arithmetic limit,
    // not the tangency test, which `ssi` makes with `tol::TANGENT` on the traced curve.
    (n.norm() > f64::EPSILON.sqrt()).then(|| n.unit_or_zero())
}

fn inside(region: &Aabb, p: Point3) -> bool {
    (0..3).all(|k| p[k] >= region.min[k] && p[k] <= region.max[k])
}

fn segment_distance(p: Point3, a: Point3, b: Point3) -> f64 {
    let d = b - a;
    let l2 = d.norm2();
    if l2 <= 0.0 {
        return p.dist(a);
    }
    let t = ((p - a).dot(d) / l2).clamp(0.0, 1.0);
    p.dist(a + d * t)
}

fn near_polyline(p: Point3, line: &[Point3], within: f64) -> bool {
    line.windows(2)
        .any(|w| segment_distance(p, w[0], w[1]) <= within)
}

/// Whether `p`, on both surfaces, lies on the branch traced through `line`: within `within` of a
/// chord, and on the curve where it passes that chord's nearest point. Two branches can run
/// closer than a step apart, where the surfaces cross along both sides of a sliver.
fn on_branch(
    a: &Surface,
    b: &Surface,
    p: Point3,
    uv: UvPair,
    line: &[Point3],
    within: f64,
) -> bool {
    line.windows(2).any(|w| {
        let d = w[1] - w[0];
        let l2 = d.norm2();
        let t = if l2 > 0.0 {
            ((p - w[0]).dot(d) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let foot = w[0] + d * t;
        if p.dist(foot) > within {
            return false;
        }
        if l2 <= 0.0 {
            return true;
        }
        let Some(q) = refine_across(a, b, foot, d * (1.0 / l2.sqrt())) else {
            return true;
        };
        p.dist(q) <= 10.0 * tol::LINEAR || joined(a, b, p, q, Some(uv))
    })
}

/// Whether the curve runs from `p` to `r` near their chord: a step whose ends are both on the
/// curve, with matching tangents, can still have jumped to another stretch of it.
fn joined(a: &Surface, b: &Surface, p: Point3, r: Point3, hints: Option<UvPair>) -> bool {
    let mid = (p + r) * 0.5;
    refine_onto_both_near(a, b, mid, hints).is_some_and(|m| m.0.dist(mid) < 0.1 * p.dist(r))
}

struct Branch {
    points: Vec<Point3>,
    uvs: Vec<UvPair>,
    closed: bool,
    /// The crossings among its two ends.
    ends: Vec<Point3>,
    /// The crossing it was traced from, when it was not found from a seed.
    from: Option<Point3>,
}

impl Branch {
    fn new(all: Vec<(Point3, UvPair)>, closed: bool, ends: Vec<Point3>) -> Branch {
        let (points, uvs) = all.into_iter().unzip();
        Branch {
            points,
            uvs,
            closed,
            ends,
            from: None,
        }
    }
}

/// Where `a` and `b` touch near `p`: the point of `a` where its normal is parallel to `b`'s, found
/// by Newton steps in `a`'s parameters from the point of `a` nearest `p`. It is a crossing of two
/// branches of the intersection only where it lies on `b` too; elsewhere it is `None`.
fn touching_point(a: &Surface, b: &Surface, p: Point3) -> Option<Point3> {
    let mut uv = a.project(p);
    let mut hint_b = b.project(p);
    for _ in 0..40 {
        let (g, j) = tilt(a, b, uv, &mut hint_b)?;
        if g.x.abs().max(g.y.abs()) <= 1e-14 {
            let q = a.eval(uv.x, uv.y);
            // Halfway to `b`: within half the tolerance of each when the gap is within it.
            let qb = b.project_near(q, hint_b);
            let q = q.lerp(b.eval(qb.x, qb.y), 0.5);
            // Measured afresh: near a surface's edge the steps above can follow a projection
            // held to that edge, whose normal is not the surface's at `q`.
            let (ua, ub) = (a.project(q), b.project(q));
            let touch = a.normal(ua.x, ua.y).cross(b.normal(ub.x, ub.y)).norm() <= tol::TANGENT;
            return (touch && distance_to(b, q) <= 0.5 * tol::LINEAR).then_some(q);
        }
        let (su, sv) = solve2(j[0].x, j[1].x, j[0].y, j[1].y, -g.x, -g.y)?;
        uv = Vec2::new(uv.x + su, uv.y + sv);
    }
    None
}

/// How far `b`'s normal tilts out of `a`'s at `a(uv)`: the sines along `a`'s two parameter
/// directions, and their derivatives along `u` and `v`. Zero where the surfaces touch.
fn tilt(a: &Surface, b: &Surface, uv: Vec2, hint_b: &mut Vec2) -> Option<(Vec2, [Vec2; 2])> {
    let at = |uv: Vec2, hint_b: &mut Vec2| {
        let (q, su, sv) = a.d1(uv.x, uv.y);
        let (lu, lv) = (su.norm(), sv.norm());
        if lu <= f64::EPSILON || lv <= f64::EPSILON {
            return None;
        }
        *hint_b = b.project_near(q, *hint_b);
        let nb = b.normal(hint_b.x, hint_b.y);
        Some((Vec2::new(su.dot(nb) / lu, sv.dot(nb) / lv), lu, lv))
    };
    let (g, lu, lv) = at(uv, hint_b)?;
    // Steps of about a micrometre: the normals turn measurably, rounding does not.
    let (du, dv) = (1e-6 / lu, 1e-6 / lv);
    let mut hb = *hint_b;
    let (gu, ..) = at(Vec2::new(uv.x + du, uv.y), &mut hb)?;
    let mut hb = *hint_b;
    let (gv, ..) = at(Vec2::new(uv.x, uv.y + dv), &mut hb)?;
    Some((g, [(gu - g) * (1.0 / du), (gv - g) * (1.0 / dv)]))
}

/// The directions the branches through a crossing `c` leave it in, both ways along each: where
/// the surfaces touch, the gap between them is a quadratic form in the step, zero along two lines.
fn crossing_directions(a: &Surface, b: &Surface, c: Point3) -> Vec<Vec3> {
    let uv = a.project(c);
    let mut hint_b = b.project(c);
    let Some((_, j)) = tilt(a, b, uv, &mut hint_b) else {
        return Vec::new();
    };
    let (_, su, sv) = a.d1(uv.x, uv.y);
    let (lu, lv) = (su.norm(), sv.norm());
    // The tilt along each direction is that row of the gap's second derivative over its length.
    let (h11, h22) = (lu * j[0].x, lv * j[1].y);
    let h12 = 0.5 * (lu * j[1].x + lv * j[0].y);
    let disc = h12 * h12 - h11 * h22;
    if disc <= 0.0 {
        return Vec::new();
    }
    let roots = |p: f64| [(-h12 + disc.sqrt()) / p, (-h12 - disc.sqrt()) / p];
    let steps: Vec<Vec2> = if h22.abs() >= h11.abs() {
        roots(h22).map(|m| Vec2::new(1.0, m)).to_vec()
    } else {
        roots(h11).map(|m| Vec2::new(m, 1.0)).to_vec()
    };
    steps
        .into_iter()
        .filter_map(|s| (su * s.x + sv * s.y).unit())
        .flat_map(|d| [d, -d])
        .collect()
}

/// One direction of a branch from `p0`, whose parameters on the two surfaces are `uv0`, with
/// every point's parameters, and whether it closed back to `p0`. A walk that stops short of the
/// region's edge because the surfaces become tangent ends on the point where they touch, when
/// there is one: two branches cross there.
fn walk(
    a: &Surface,
    b: &Surface,
    region: &Aabb,
    (p0, uv0): (Point3, UvPair),
    sign: f64,
    h_max: f64,
    known: &[Point3],
) -> (Vec<(Point3, UvPair)>, bool, Option<Point3>) {
    let h_min = h_max * 1e-7;
    let mut pts = vec![(p0, uv0)];
    let (mut p, mut uv) = (p0, uv0);
    let mut h = h_max;
    let mut travelled = 0.0;
    let mut heading = None;
    let mut start = None;
    for _ in 0..20_000 {
        let Some(t) = tangent(a, b, uv) else {
            break;
        };
        let t = t * sign;
        heading = Some(t);
        let t0 = *start.get_or_insert(t);
        let from = hint(a, b, uv);
        // Where the surfaces nearly coincide the step follows the gap's zero contour on `a`
        // instead, so it never settles across a thin lens.
        let next = |h: f64| match from {
            Some(_) => refine_onto_both_near(a, b, p + t * h, from),
            None => refine_on_contour(a, b, uv, t * h),
        };
        let linked = |r: Point3| match from {
            Some(_) => joined(a, b, p, r, from),
            None => refine_on_contour(a, b, uv, (r - p) * 0.5)
                .is_some_and(|m| m.0.dist(p.lerp(r, 0.5)) < 0.1 * p.dist(r)),
        };
        let mut accepted = None;
        while h >= h_min {
            if let Some((r, ruv)) = next(h) {
                let step = r.dist(p);
                let turn_ok =
                    tangent(a, b, ruv).is_some_and(|tr| (tr * sign).dot(t) > MAX_TURN.cos());
                // Refinement can land behind `p` on the same curve, or on another branch; a step
                // must go the way the tangent points, or the walk can shuttle between two points.
                let ahead = (r - p).dot(t) > 0.5 * step;
                if step > 0.3 * h && step < 1.7 * h && turn_ok && ahead && linked(r) {
                    accepted = Some((r, ruv));
                    break;
                }
            }
            h *= 0.5;
        }
        let Some((r, ruv)) = accepted else {
            break;
        };
        travelled += r.dist(p);
        let gap = r.dist(p0);
        let mut reach = h_max;
        // Across a thin lens the other branch passes within a step of `p0` running the other way;
        // the walk closes only arriving the way it left, or it skips the tip between them.
        let arriving = tangent(a, b, ruv).is_some_and(|tr| (tr * sign).dot(t0) > 0.0);
        if travelled > 3.0 * h && gap < 1.5 * h && (p0 - p).dot(t) > 0.0 && arriving {
            // The first step shows how tightly the curve turns at `p0`: closing across a gap much
            // wider than it can skip a fold there, so the walk closes in on `p0` in steps first.
            let first = pts.get(1).map_or(f64::INFINITY, |q| q.0.dist(p0));
            if gap <= 2.0 * first && joined(a, b, r, p0, hint(a, b, ruv)) {
                if gap > tol::LINEAR {
                    pts.push((r, ruv));
                }
                return (pts, true, None);
            }
            reach = 0.5 * gap;
        }
        pts.push((r, ruv));
        (p, uv) = (r, ruv);
        if !inside(region, r) {
            return (pts, false, None);
        }
        h = (h * 1.5).min(reach);
    }
    // A walk the edge of a bounded surface's parameters stops ends on that edge exactly, where the
    // vertex its face's boundary meets the other surface at lies: its last point, as close as the
    // shortest step got, can miss that vertex by more than the tolerance.
    if pts.len() > 1
        && let Some(q) = onto_domain_edge(a, b, p, uv, (10.0 * h_min).max(10.0 * tol::LINEAR))
    {
        if q.0.dist(p) <= tol::LINEAR {
            pts.pop();
        }
        pts.push(q);
        return (pts, false, None);
    }
    let end = touching_point(a, b, p)
        .filter(|&q| q.dist(p) <= h_max && heading.is_none_or(|t| (q - p).dot(t) >= -tol::LINEAR));
    // A known point of both surfaces in the band where they were found to touch is that
    // touching, exactly: the search finds a touching of high order only to a few millionths.
    let onto = heading.and_then(|t| stalled_onto(a, b, p, t, h_max, known));
    let end = match (end, onto) {
        (Some(q), Some(k)) if same_crossing(a, b, q, k, h_max) => Some(k),
        (end, _) => end,
    };
    // The branch ends on the crossing itself, which becomes a vertex: a last point a hair from it
    // could miss the vertex by more than the tolerance once fitted.
    if let Some(q) = end {
        if q.dist(p) <= tol::LINEAR && pts.len() > 1 {
            pts.pop();
        }
        pts.push((q, params(a, b, q)));
    } else if let Some(q) = onto {
        pts.push((q, params(a, b, q)));
    }
    (pts, false, end)
}

/// The known point of both surfaces a walk stalled at `p`, heading along `t`, was on its way to:
/// one where they are tangent, which is why it stalled, ahead of `p` within a step, with the
/// chord to it on both surfaces. Where the surfaces touch to a high order no walk reaches the
/// point and no tangency search finds it to the tolerance, but the faces' edges that cross
/// there do; of several such points the farthest, since the curve runs through the nearer.
fn stalled_onto(
    a: &Surface,
    b: &Surface,
    p: Point3,
    t: Vec3,
    h_max: f64,
    known: &[Point3],
) -> Option<Point3> {
    let on_both = |x: Point3| distance_to(a, x).max(distance_to(b, x));
    known
        .iter()
        .copied()
        .filter(|&q| {
            let along = (q - p).dot(t);
            along > tol::LINEAR && q.dist(p) <= h_max && on_both(q) <= tol::LINEAR
        })
        .filter(|&q| {
            let (ua, ub) = params(a, b, q);
            a.normal(ua.x, ua.y).cross(b.normal(ub.x, ub.y)).norm() <= tol::TANGENT
        })
        .filter(|&q| (1..4).all(|k| on_both(p.lerp(q, k as f64 / 4.0)) <= 0.5 * tol::LINEAR))
        .max_by(|x, y| (*x - p).dot(t).total_cmp(&(*y - p).dot(t)))
}

fn branch(
    a: &Surface,
    b: &Surface,
    region: &Aabb,
    p0: (Point3, UvPair),
    h_max: f64,
    known: &[Point3],
    crossings: &mut Vec<Point3>,
) -> Branch {
    let (fwd, closed, end) = walk(a, b, region, p0, 1.0, h_max, known);
    crossings.extend(end);
    if closed {
        return Branch::new(fwd, closed, Vec::new());
    }
    let (back, _, back_end) = walk(a, b, region, p0, -1.0, h_max, known);
    crossings.extend(back_end);
    let mut all: Vec<(Point3, UvPair)> = back.into_iter().skip(1).rev().collect();
    all.extend(fwd);
    Branch::new(all, false, end.into_iter().chain(back_end).collect())
}

/// Every branch of `a ∩ b` through the seeds, within `region`, and the points where branches
/// cross because the surfaces touch there.
pub(crate) fn trace(
    a: &Surface,
    b: &Surface,
    region: &Aabb,
    seeds: &[Point3],
) -> Result<(Vec<SsiCurve>, Vec<Point3>)> {
    let h_max = (region.diagonal() / 40.0).max(1e-6);
    let mut branches: Vec<Branch> = Vec::new();
    let mut crossings = Vec::new();
    for &s in seeds {
        let Some((p, uv)) = refine_onto_both_near(a, b, s, None) else {
            continue;
        };
        if !inside(region, p) || tangent(a, b, uv).is_none() {
            continue;
        }
        if branches
            .iter()
            .any(|br| on_branch(a, b, p, uv, &br.points, 0.25 * h_max))
        {
            continue;
        }
        let br = branch(a, b, region, (p, uv), h_max, seeds, &mut crossings);
        // A seed on a crossing can walk straight into it both ways: that is a point, not a
        // branch, and the branches through the crossing are traced from it below.
        if br.points.iter().any(|q| q.dist(br.points[0]) > tol::LINEAR) {
            branches.push(br);
        }
    }
    // Every branch through a crossing is traced from it, since seeds may reach only one of them.
    let mut k = 0;
    while k < crossings.len() {
        let c = crossings[k];
        k += 1;
        if crossings[..k - 1]
            .iter()
            .any(|&q| same_crossing(a, b, q, c, h_max))
        {
            continue;
        }
        for d in crossing_directions(a, b, c) {
            // Far enough out that the surfaces cross at an angle, near enough to stay on the branch.
            let Some(q) = refine_across(a, b, c + d * (1e-3 * h_max), d) else {
                continue;
            };
            let quv = params(a, b, q);
            let Some(t) = tangent(a, b, quv) else {
                continue;
            };
            let sign = if t.dot(d) >= 0.0 { 1.0 } else { -1.0 };
            let (pts, _, end) = walk(a, b, region, (q, quv), sign, h_max, seeds);
            // A branch already traced through `c` passes near where this one is half a step out.
            let probe = pts
                .iter()
                .map(|x| x.0)
                .find(|p| p.dist(c) >= 0.5 * h_max)
                .unwrap_or(pts[pts.len() - 1].0);
            if pts.len() < 2
                || branches
                    .iter()
                    .any(|br| near_polyline(probe, &br.points, 0.1 * h_max))
            {
                continue;
            }
            crossings.extend(end);
            let mut all = vec![(c, params(a, b, c))];
            all.extend(&pts[1..]);
            let mut br = Branch::new(all, false, std::iter::once(c).chain(end).collect());
            br.from = Some(c);
            branches.push(br);
        }
    }
    // A loop closed across a sharp tip leaves its seeds there off its chords, and one of them
    // traces the loop again: two closed branches each within reach of the other are one loop.
    let mut k = 0;
    while k < branches.len() {
        let twin = |o: &Branch, br: &Branch| {
            let along = |x: &Branch, y: &Branch| {
                let mut ring = y.points.clone();
                ring.push(y.points[0]);
                x.points
                    .iter()
                    .all(|&p| near_polyline(p, &ring, 0.25 * h_max))
            };
            o.closed && along(br, o) && along(o, br)
        };
        if branches[k].closed && branches[..k].iter().any(|o| twin(o, &branches[k])) {
            branches.remove(k);
        } else {
            k += 1;
        }
    }
    // A loop inside which the surfaces never part by more than the tolerance is where they touch:
    // it bounds no region of either.
    branches.retain(|br| !br.closed || lens_depth(a, b, br) > tol::LINEAR);
    let crossings = merge_crossings(a, b, crossings, &mut branches, h_max);
    // A crossing found where the walk stalls in a band the surfaces share within the tolerance can
    // lie off the curve the walk was on, so no curve fits through both; the branch then ends where
    // the walk stopped, as it would have without the crossing.
    let mut crossings = crossings;
    loop {
        match fit_branches(a, b, &branches) {
            Ok(out) => return Ok((out, crossings)),
            Err((i, e)) => {
                let bad = std::mem::take(&mut branches[i].ends);
                // A walk that leaves a known point of both surfaces and comes back to it within
                // the tolerance found no curve: where a curve leaves a bounded surface at its
                // corner, the points beside the corner are on both only to the tolerance.
                let br = &branches[i];
                let (first, last) = (br.points[0], br.points[br.points.len() - 1]);
                let known = seeds.iter().any(|s| s.dist(first) <= tol::LINEAR);
                let short = matches!(&e, Error::Numerical(m) if m == crate::geom::SHORT_BRANCH);
                let back = !br.closed && br.from.is_none() && first.dist(last) <= tol::LINEAR;
                if bad.is_empty() && short && known && back {
                    branches.remove(i);
                    continue;
                }
                if bad.is_empty() {
                    return Err(e);
                }
                let is_bad = |p: Point3| bad.iter().any(|q| q.dist(p) <= tol::LINEAR);
                crossings.retain(|&c| !is_bad(c));
                branches.retain(|br| br.from.is_none_or(|c| !is_bad(c)));
                for br in &mut branches {
                    br.ends.retain(|&c| !is_bad(c));
                    for end in [br.points.len() - 1, 0] {
                        if br.points.len() > 2 && is_bad(br.points[end]) {
                            br.points.remove(end);
                            br.uvs.remove(end);
                        }
                    }
                }
            }
        }
    }
}

/// The deepest the surfaces part inside a closed branch: the gap from `a` to `b` midway between
/// each point and the nearest point of the far half of the loop.
fn lens_depth(a: &Surface, b: &Surface, br: &Branch) -> f64 {
    let n = br.points.len();
    let mut deepest: f64 = 0.0;
    for i in 0..n {
        let p = br.points[i];
        let across = (0..n)
            .filter(|&j| {
                let d = (i as isize - j as isize).unsigned_abs();
                d.min(n - d) >= n / 4
            })
            .min_by(|&j, &k| p.dist(br.points[j]).total_cmp(&p.dist(br.points[k])));
        let Some(j) = across else {
            continue;
        };
        for f in [0.25, 0.5, 0.75] {
            let x = p.lerp(br.points[j], f);
            let ua = a.project_near(x, br.uvs[i].0);
            let q = a.eval(ua.x, ua.y);
            let ub = b.project_near(q, br.uvs[i].1);
            deepest = deepest.max((q - b.eval(ub.x, ub.y)).dot(b.normal(ub.x, ub.y)).abs());
        }
    }
    deepest
}

/// Whether crossings `p` and `q` are one touching point: within a step of each other with every
/// point between them on both surfaces, and the surfaces not passing through each other across the
/// segment. Walks that stall in a band the surfaces share within the tolerance each end anywhere
/// along it, so the branches through one crossing would miss each other; but two surfaces can hold
/// one straight line and touch at two points of it, crossing along it in between.
pub(super) fn same_crossing(a: &Surface, b: &Surface, p: Point3, q: Point3, h_max: f64) -> bool {
    let d = p.dist(q);
    if d <= tol::LINEAR {
        return true;
    }
    let n = (d / tol::LINEAR).ceil().min(16.0) as usize;
    let along = (q - p) * (1.0 / d);
    d <= h_max
        && (1..n).all(|k| {
            let x = p.lerp(q, k as f64 / n as f64);
            distance_to(a, x) <= tol::LINEAR
                && distance_to(b, x) <= tol::LINEAR
                && !crosses_across(a, b, x, along, d)
        })
}

/// Whether `b` passes through `a` across a line through `x` along `along`: the points of `a` as far
/// out either side of it as `reach` lie on opposite sides of `b`, both beyond the tolerance. Within
/// one touching band the gap there stays within the tolerance, or keeps its sign.
fn crosses_across(a: &Surface, b: &Surface, x: Point3, along: Vec3, reach: f64) -> bool {
    let uv = a.project(x);
    let side = a.normal(uv.x, uv.y).cross(along).unit_or_zero();
    let gap = |s: f64| {
        let uv = a.project(x + side * (s * reach));
        super::csi::signed_distance(b, a.eval(uv.x, uv.y))
    };
    let (g0, g1) = (gap(1.0), gap(-1.0));
    g0.abs() > tol::LINEAR && g1.abs() > tol::LINEAR && (g0 > 0.0) != (g1 > 0.0)
}

/// The crossings with each group of one touching point reduced to its first, and every branch
/// that ends on a later one of a group moved onto the first.
fn merge_crossings(
    a: &Surface,
    b: &Surface,
    crossings: Vec<Point3>,
    branches: &mut [Branch],
    h_max: f64,
) -> Vec<Point3> {
    let mut kept: Vec<Point3> = Vec::new();
    let mut moved: Vec<(Point3, Point3)> = Vec::new();
    for c in crossings {
        match kept.iter().find(|&&k| same_crossing(a, b, k, c, h_max)) {
            Some(&k) if k.dist(c) > tol::LINEAR => moved.push((c, k)),
            Some(_) => {}
            None => kept.push(c),
        }
    }
    if moved.is_empty() {
        return kept;
    }
    let onto = |p: Point3| {
        moved
            .iter()
            .find(|(c, _)| c.dist(p) <= tol::LINEAR)
            .map(|&(_, k)| k)
    };
    for br in branches.iter_mut().filter(|br| !br.closed) {
        for end in [0, br.points.len() - 1] {
            if let Some(k) = onto(br.points[end]) {
                br.points[end] = k;
                br.uvs[end] = params(a, b, k);
            }
        }
        for e in &mut br.ends {
            if let Some(k) = onto(*e) {
                *e = k;
            }
        }
        if let Some(k) = br.from.and_then(onto) {
            br.from = Some(k);
        }
    }
    kept
}

/// Each branch's fitted curve, or the index of the first branch that does not fit and why.
fn fit_branches(
    a: &Surface,
    b: &Surface,
    branches: &[Branch],
) -> std::result::Result<Vec<SsiCurve>, (usize, Error)> {
    let mut out = Vec::new();
    for (i, br) in branches.iter().enumerate() {
        if br.closed && br.points.len() < 3 {
            return Err((
                i,
                Error::Numerical("closed intersection branch has too few points".into()),
            ));
        }
        let fit =
            IntersectionCurve::fit_near(&br.points, Some(&br.uvs), a.clone(), b.clone(), br.closed)
                .map_err(|e| (i, e))?;
        let range = fit.approx.domain();
        out.push(SsiCurve {
            curve: Curve::Intersection(Box::new(fit)),
            range,
            closed: br.closed,
            tangent: false,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boolean::csi::signed_distance,
        geom::{Cylinder, Plane, Torus},
        loft::{Section, loft},
        math::{Frame, Vec2},
        profile::{PointsInterp, Profile},
    };

    /// A gear's tooth space: two flanks from r 21.5 down to a round root at r 17.
    fn tooth_space() -> Vec<Vec2> {
        let flank = |i: usize, s: f64| {
            let t = i as f64 / 12.0;
            let (r, a) = (21.5 - 4.5 * t, s * (0.09 - 0.05 * t * t));
            Vec2::new(r * a.cos(), r * a.sin())
        };
        let mut pts: Vec<Vec2> = (0..=12).map(|i| flank(i, -1.0)).collect();
        pts.extend((1..12).map(|i| {
            let a = -0.04 + 0.08 * i as f64 / 12.0;
            let r = 17.0 - 0.3 * (std::f64::consts::PI * i as f64 / 12.0).sin();
            Vec2::new(r * a.cos(), r * a.sin())
        }));
        pts.extend((0..=12).rev().map(|i| flank(i, 1.0)));
        pts
    }

    // The side of a circle-to-square loft and its copy moved 1e-5 mm along x meet down the side's
    // middle and along its straight top edge. Those cross where the sides coincide within the
    // tolerance for 2e-7 mm, and each walk used to end anywhere in that band: the line down the
    // side missed the crossing the top edge was cut at, and never split its face.
    #[test]
    fn branches_that_stall_in_one_shared_band_end_on_one_crossing() {
        let at_z = |z: f64| Frame::WORLD.translated(Vec3::Z * z);
        let lofted = loft(
            &[
                Section {
                    profile: Profile::circle(Vec2::ZERO, 20.0),
                    frame: at_z(0.0),
                },
                Section {
                    profile: Profile::rect(Vec2::ZERO, 20.0, 20.0),
                    frame: at_z(30.0),
                },
            ],
            false,
            "lf",
        )
        .unwrap();
        let side = lofted
            .face_ids()
            .map(|f| &lofted.face(f).surface)
            .filter(|s| !matches!(s, Surface::Plane(_)))
            .min_by(|s, t| {
                let p = Vec3::new(0.0, -10.0, 15.0);
                distance_to(s, p).total_cmp(&distance_to(t, p))
            })
            .unwrap();
        let moved = side.transformed(&crate::math::Transform::translation(Vec3::X * 1e-5));
        let region = Aabb {
            min: Vec3::new(-12.0, -11.0, -3.0),
            max: Vec3::new(12.0, -5.0, 33.0),
        };
        let seeds = [
            Vec3::new(0.0, -10.0, 15.0),
            Vec3::new(-5.0, -10.0, 30.0),
            Vec3::new(5.0, -10.0, 30.0),
        ];
        let (curves, crossings) = trace(side, &moved, &region, &seeds).unwrap();
        assert_eq!(crossings.len(), 1, "{crossings:?}");
        let ends: Vec<Point3> = curves
            .iter()
            .flat_map(|c| [c.curve.eval(c.range.0), c.curve.eval(c.range.1)])
            .filter(|p| p.z > 29.0 && p.x.abs() < 1e-3)
            .collect();
        assert_eq!(ends.len(), 3, "{ends:?}");
        for p in ends {
            assert!(p.dist(crossings[0]) <= tol::LINEAR, "{p:?} {crossings:?}");
        }
    }

    // Two lofted cylinders r 5 whose axes lie 10 − 3e-4 apart overlap in a sliver: their sides
    // cross along two lines at y = ±√(r² − (d/2)²) ≈ ±0.0387, nearer each other than a quarter
    // step. The seed on the second line is still on a branch of its own.
    #[test]
    fn two_crossings_nearer_than_a_step_are_both_traced() {
        let (r, d) = (5.0, 10.0 - 3e-4);
        let sections: Vec<Section> = [0.0, 5.0, 10.0, 15.0]
            .iter()
            .map(|&z| Section {
                profile: Profile::circle(Vec2::ZERO, 2.0 * r),
                frame: Frame::WORLD.translated(Vec3::Z * z),
            })
            .collect();
        let can = loft(&sections, false, "can").unwrap();
        let side = can
            .face_ids()
            .map(|f| &can.face(f).surface)
            .find(|s| matches!(s, Surface::Nurbs(_)))
            .unwrap();
        let moved = side.transformed(&crate::math::Transform::translation(Vec3::X * d));
        let y = (r * r - 0.25 * d * d).sqrt();
        let region = Aabb {
            min: Vec3::new(3.0, -2.0, 1.0),
            max: Vec3::new(7.0, 2.0, 14.0),
        };
        let seeds = [Vec3::new(0.5 * d, y, 7.5), Vec3::new(0.5 * d, -y, 7.5)];
        let (curves, _) = trace(side, &moved, &region, &seeds).unwrap();
        assert_eq!(curves.len(), 2);
        for c in &curves {
            for k in 0..=4 {
                let p = c
                    .curve
                    .eval(c.range.0 + (c.range.1 - c.range.0) * k as f64 / 4.0);
                assert!((p.x - 0.5 * d).abs() <= 1e-6, "{p:?}");
                assert!((p.y.abs() - y).abs() <= 1e-6, "{p:?}");
            }
        }
        let ys: Vec<f64> = curves.iter().map(|c| c.curve.eval(c.range.0).y).collect();
        assert!(ys[0] * ys[1] < 0.0, "{ys:?}");
    }

    #[test]
    fn a_plane_across_a_steeply_twisted_loft_never_shuttles_between_points() {
        let twist = 0.3;
        let outline = Profile::points(
            &tooth_space(),
            PointsInterp::Smooth {
                corner: 30f64.to_radians(),
            },
            "space",
        )
        .unwrap();
        let sections: Vec<Section> = (0..=8)
            .map(|i| {
                let z = -0.5 + 11.0 * i as f64 / 8.0;
                let x = Vec3::X * (twist * z).cos() + Vec3::Y * (twist * z).sin();
                Section {
                    profile: outline.clone(),
                    frame: Frame::new(Vec3::Z * z, Vec3::Z, x).unwrap(),
                }
            })
            .collect();
        let tool = loft(&sections, false, "g").unwrap();
        let plane = Surface::Plane(Plane {
            frame: Frame::WORLD,
        });
        let region = Aabb {
            min: Vec3::new(-19.16, -5.37, -0.107),
            max: Vec3::new(19.16, 19.16, 0.107),
        };
        let seeds = [Vec3::new(16.98647297164612, -0.6798377134004823, 0.0)];
        // From this seed on the second side face, refinement used to land each step behind the
        // last, and the walk went back and forth between two points to its 20 000-step cap.
        let side = &tool.face(tool.face_ids().nth(1).unwrap()).surface;
        for c in trace(&plane, side, &region, &seeds).unwrap().0 {
            let Curve::Intersection(fit) = &c.curve else {
                continue;
            };
            let n = fit.approx.control_points().len();
            assert!(n < 2000, "a branch of {n} points");
        }
    }

    #[test]
    fn a_crossing_no_curve_reaches_is_dropped_rather_than_failing_the_fit() {
        let at_z = |z: f64| Frame::WORLD.translated(Vec3::Z * z);
        let section = |profile, z| Section {
            profile,
            frame: at_z(z),
        };
        let a = loft(
            &[
                section(Profile::circle(Vec2::ZERO, 20.0), 0.0),
                section(Profile::rect(Vec2::ZERO, 20.0, 20.0), 30.0),
            ],
            false,
            "lf",
        )
        .unwrap();
        let mut b = a.clone();
        b.transform(&crate::math::Transform::translation(Vec3::new(
            3e-9, 0.0, 0.0,
        )));
        // The sides either side of the ruling through (10, 0, z) touch along it. Traced from the
        // loft's corners, the walk stalls 0.4 mm up and 0.045 mm across from a touching point.
        let side =
            |x: &crate::topo::Brep, i: usize| x.face(x.face_ids().nth(i).unwrap()).surface.clone();
        let (sa, sb) = (side(&a, 0), side(&b, 4));
        let region = Aabb {
            min: Vec3::new(-11.0, -11.0, -1.0),
            max: Vec3::new(11.0, 11.0, 31.0),
        };
        let seeds: Vec<Point3> = a.vertex_ids().map(|v| a.vertex(v).point).collect();
        let (curves, _) = trace(&sa, &sb, &region, &seeds).unwrap();
        assert!(!curves.is_empty());
    }

    #[test]
    fn crossing_cylinders_trace_two_closed_loops() {
        let a = Surface::Cylinder(Cylinder {
            frame: Frame::new(Vec3::new(-20.0, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap(),
            radius: 5.0,
        });
        let b = Surface::Cylinder(Cylinder {
            frame: Frame::new(Vec3::new(0.0, -20.0, 0.0), Vec3::Y, Vec3::Z).unwrap(),
            radius: 3.0,
        });
        let region = Aabb {
            min: Vec3::new(-4.0, -6.0, -6.0),
            max: Vec3::new(4.0, 6.0, 6.0),
        };
        let seeds = [Vec3::new(3.0, 4.0, 0.0), Vec3::new(-3.0, -4.0, 0.0)];
        let curves = trace(&a, &b, &region, &seeds).unwrap().0;
        assert_eq!(curves.len(), 2);
        for c in &curves {
            assert!(c.closed);
            for i in 0..50 {
                let t = c.range.0 + (c.range.1 - c.range.0) * i as f64 / 50.0;
                let p = c.curve.eval(t);
                assert!(signed_distance(&a, p).abs() < 1e-7);
                assert!(signed_distance(&b, p).abs() < 1e-7);
            }
        }
    }

    fn torus(centre: [f64; 3], major: f64, minor: f64) -> Surface {
        Surface::Torus(Torus {
            frame: Frame::new(Vec3::from_array(centre), Vec3::Z, Vec3::X).unwrap(),
            major,
            minor,
        })
    }

    fn assert_on_both(a: &Surface, b: &Surface, curves: &[SsiCurve]) {
        for c in curves {
            for i in 0..=64 {
                let p = c
                    .curve
                    .eval(c.range.0 + (c.range.1 - c.range.0) * i as f64 / 64.0);
                assert!(signed_distance(a, p).abs() <= tol::LINEAR, "{p:?}");
                assert!(signed_distance(b, p).abs() <= tol::LINEAR, "{p:?}");
            }
        }
    }

    #[test]
    fn branches_through_a_point_where_two_tori_touch_end_on_it() {
        // Two root blends, tubes of radius 3 about circles of radius 5 at z 3, touch the plane
        // z = 0 along circles that cross at (3, ±4, 0); there the tori touch each other too.
        let (a, b) = (
            torus([0.0, 0.0, 3.0], 5.0, 3.0),
            torus([6.0, 0.0, 3.0], 5.0, 3.0),
        );
        let region = Aabb {
            min: Vec3::new(1.5, 2.5, -0.5),
            max: Vec3::new(4.5, 5.5, 1.5),
        };
        // On the mirror plane x = 3, which holds one branch through the touching point.
        let seeds = [Vec3::new(3.0, 4.5, 0.03)];
        let (curves, crossings) = trace(&a, &b, &region, &seeds).unwrap();
        let touch = Vec3::new(3.0, 4.0, 0.0);
        assert!(
            crossings.iter().any(|c| c.dist(touch) <= 1e-9),
            "{crossings:?}"
        );
        assert_on_both(&a, &b, &curves);
        // Both branches through it reach it from each side.
        let ends = curves
            .iter()
            .flat_map(|c| [c.curve.eval(c.range.0), c.curve.eval(c.range.1)])
            .filter(|p| p.dist(touch) <= tol::LINEAR)
            .count();
        assert_eq!(ends, 4);
    }

    #[test]
    fn a_seed_a_hair_from_where_two_tori_touch_traces_the_branches_through_it() {
        // Tubes of radius 4.5 and 3 about circles of radius 8 and 6 touch the plane y = 0 along
        // circles that cross at (6.4, 0, 4.8). A seed 8e-8 mm away on that plane, as an edge's
        // meeting with a face can land, walks into the crossing both ways.
        let tube = |x: f64, y: f64, major: f64, minor: f64| {
            Surface::Torus(Torus {
                frame: Frame::new(Vec3::new(x, y, 0.0), Vec3::Y, Vec3::X).unwrap(),
                major,
                minor,
            })
        };
        let (a, b) = (tube(0.0, 4.5, 8.0, 4.5), tube(10.0, 3.0, 6.0, 3.0));
        let touch = Vec3::new(6.4, 0.0, 4.8);
        let region = Aabb {
            min: Vec3::new(5.0, -0.5, 3.0),
            max: Vec3::new(8.0, 2.0, 6.5),
        };
        let (s, c) = 22.5f64.to_radians().sin_cos();
        let seed = touch + Vec3::new(c, 0.0, s) * 8e-8;
        let (curves, crossings) = trace(&a, &b, &region, &[seed]).unwrap();
        assert!(crossings.iter().any(|c| c.dist(touch) <= tol::LINEAR));
        assert_on_both(&a, &b, &curves);
        let ends = curves
            .iter()
            .flat_map(|c| [c.curve.eval(c.range.0), c.curve.eval(c.range.1)])
            .filter(|p| p.dist(touch) <= tol::LINEAR)
            .count();
        assert_eq!(ends, 4);
    }

    #[test]
    fn a_curve_that_folds_back_within_micrometres_fits_through_the_fold() {
        // A cylinder's top fillet against a copy moved 1e-5 mm along x: the tube meets the
        // cylinder near z = 17 ± √(6e-5 cos θ), two arcs that turn back into each other at
        // θ = ±90° with a radius of about 1.5e-6 mm.
        let a = Surface::Cylinder(Cylinder {
            frame: Frame::WORLD,
            radius: 20.0,
        });
        let b = torus([1e-5, 0.0, 17.0], 17.0, 3.0);
        let region = Aabb {
            min: Vec3::new(-21.0, -21.0, 16.9),
            max: Vec3::new(21.0, 21.0, 17.1),
        };
        let seeds = [Vec3::new(20.0, 0.0, 17.0077), Vec3::new(20.0, 0.0, 16.9923)];
        let (curves, _) = trace(&a, &b, &region, &seeds).unwrap();
        assert!(!curves.is_empty());
        assert_on_both(&a, &b, &curves);
    }
}
