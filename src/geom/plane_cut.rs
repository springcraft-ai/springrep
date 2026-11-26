//! Where a curve meets a plane, exactly: lines and conics in closed form, splines, rational or
//! not, of any degree, by the hull of their control values and Bernstein subdivision.

use std::f64::consts::{PI, TAU};

use crate::geom::{Curve, Revolution};
use crate::math::{Aabb, Point3, Vec3};
use crate::tol;

fn meets(a: &Aabb, b: &Aabb) -> bool {
    (0..3).all(|k| a.min[k] <= b.max[k] && b.min[k] <= a.max[k])
}

/// The parameters in the open range `(t0, t1)` where curve `c` crosses the plane through `o`
/// with unit normal `n`, from beyond `tau` on one side to beyond it on the other: a curve that
/// only touches the plane, or reaches past it by no more than `tau`, crosses nothing. A line and
/// a conic are solved in closed form; a spline's signed distances to the plane are a spline
/// too, which lies within the hull of its control values, and is split until each piece either
/// stays on one side, proven so, or holds one crossing, which halving then locates. A knot span
/// whose control points' box misses `near` is passed over: the curve there lies in that box.
pub(crate) fn plane_crossings(
    c: &Curve,
    (t0, t1): (f64, f64),
    (o, n): (Point3, Vec3),
    tau: f64,
    near: &Aabb,
) -> Vec<f64> {
    let inside = |t: f64| t > t0 && t < t1;
    match c {
        Curve::Line(l) => {
            let (g0, gd) = (n.dot(l.origin - o), n.dot(l.dir));
            let (ga, gb) = (g0 + gd * t0, g0 + gd * t1);
            if ga.min(gb) < -tau && ga.max(gb) > tau {
                vec![-g0 / gd]
            } else {
                Vec::new()
            }
        }
        Curve::Circle(_) | Curve::Ellipse(_) => {
            // g(t) = a + b cos t + s sin t, which swings a ± r.
            let (a, bc, bs) = conic_terms(c, o, n);
            let r = bc.hypot(bs);
            if a - r >= -tau || a + r <= tau {
                return Vec::new();
            }
            let (phase, half) = (bs.atan2(bc), (-a / r).acos());
            let mut out = Vec::new();
            for root in [phase - half, phase + half] {
                let mut k = ((t0 - root) / TAU).ceil();
                while root + k * TAU < t1 {
                    if inside(root + k * TAU) {
                        out.push(root + k * TAU);
                    }
                    k += 1.0;
                }
            }
            out.sort_by(f64::total_cmp);
            out
        }
        _ => {
            let parts = spline_parts(c).expect("a spline");
            spline_crossings(parts, (t0, t1), (o, n), tau, near)
        }
    }
}

/// A spline's degree, knots, control points, weights (`None` where all are equal) and period.
type SplineParts<'a> = (
    usize,
    &'a [f64],
    &'a [Point3],
    Option<&'a [f64]>,
    Option<f64>,
);

/// The parts of a B-spline, an intersection curve's spline or a rational spline; `None` for a
/// line or a conic.
fn spline_parts(c: &Curve) -> Option<SplineParts<'_>> {
    match c {
        Curve::BSpline(s) => Some((s.degree(), s.knots(), s.control_points(), None, s.period())),
        Curve::Intersection(x) => {
            let s = &x.approx;
            Some((s.degree(), s.knots(), s.control_points(), None, s.period()))
        }
        Curve::Nurbs(s) => Some((
            s.degree(),
            s.knots(),
            s.control_points(),
            s.weights(),
            s.period(),
        )),
        _ => None,
    }
}

/// The signed distances `wᵢ·n·(Pᵢ − o)` of a spline's weighted control points from the plane
/// through `o` with unit normal `n`, and the least weight. With positive weights the spline's
/// distance has the sign of the spline of these, and is at least as far from zero divided by the
/// least weight, so a band of `tau` about the plane is one of `tau` times that about these.
fn spline_values((_, _, ctrl, weights, _): SplineParts<'_>, o: Point3, n: Vec3) -> (Vec<f64>, f64) {
    let weight = |j: usize| weights.map_or(1.0, |w| w[j]);
    let g = (0..ctrl.len())
        .map(|j| weight(j) * n.dot(ctrl[j] - o))
        .collect();
    let least = weights.map_or(1.0, |w| w.iter().copied().fold(f64::INFINITY, f64::min));
    (g, least)
}

/// The knot spans a spline's parameters `(t0, t1)` run through, in order: each span's index, the
/// multiple of the period it is shifted by, and the stretch of `(t0, t1)` on it.
fn spline_spans(
    (p, knots, ctrl, _, period): SplineParts<'_>,
    (t0, t1): (f64, f64),
) -> impl Iterator<Item = (usize, f64, f64, f64)> + '_ {
    let count = ctrl.len();
    let lo = knots[p];
    let copies = match period {
        Some(per) => (
            ((t0 - lo) / per).floor() as i64,
            ((t1 - lo) / per).floor() as i64,
        ),
        None => (0, 0),
    };
    (copies.0..=copies.1).flat_map(move |copy| {
        let shift = period.map_or(0.0, |per| copy as f64 * per);
        let first = knots.partition_point(|&k| k <= t0 - shift).max(1) - 1;
        (first.max(p)..count)
            .take_while(move |&i| knots[i] + shift < t1)
            .filter_map(move |i| {
                let (a, z) = ((knots[i] + shift).max(t0), (knots[i + 1] + shift).min(t1));
                (z > a).then_some((i, shift, a, z))
            })
    })
}

/// The Bernstein coefficients over `(a, z)`, within span `i`, of the spline with control values
/// `g`: its blossom at `a` and `z` (de Boor's algorithm, a different parameter at each level).
fn span_bernstein(
    (p, knots, ..): SplineParts<'_>,
    g: &[f64],
    i: usize,
    (a, z): (f64, f64),
) -> Vec<f64> {
    let mut d = vec![0.0; p + 1];
    (0..=p)
        .map(|ends| {
            d.copy_from_slice(&g[i - p..=i]);
            for r in 0..p {
                let t = if r < ends { z } else { a };
                for j in (r + 1..=p).rev() {
                    let k = i - p + j;
                    let span = knots[k + p - r] - knots[k];
                    let alpha = if span > 0.0 {
                        (t - knots[k]) / span
                    } else {
                        0.0
                    };
                    d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
                }
            }
            d[p]
        })
        .collect()
}

/// The polynomial with Bernstein coefficients `b` at `x` (de Casteljau).
fn bernstein_at(b: &[f64], x: f64) -> f64 {
    let mut small = [0.0; 16];
    let mut large = Vec::new();
    let row: &mut [f64] = if b.len() <= small.len() {
        &mut small[..b.len()]
    } else {
        large.extend_from_slice(b);
        &mut large
    };
    row.copy_from_slice(b);
    for k in (1..row.len()).rev() {
        for j in 0..k {
            row[j] = row[j] * (1.0 - x) + row[j + 1] * x;
        }
    }
    row[0]
}

/// The Bernstein coefficients of the polynomial `b` over `[0, s]` and `[s, 1]` (de Casteljau).
fn bernstein_split(b: &[f64], s: f64) -> (Vec<f64>, Vec<f64>) {
    let last = b.len() - 1;
    let mut row = b.to_vec();
    let (mut left, mut right) = (vec![row[0]], vec![row[last]]);
    for _ in 0..last {
        for j in 0..row.len() - 1 {
            row[j] = row[j] * (1.0 - s) + row[j + 1] * s;
        }
        row.pop();
        left.push(row[0]);
        right.push(row[row.len() - 1]);
    }
    right.reverse();
    (left, right)
}

/// The one root in `(0, 1)` of the polynomial `b`, whose ends have opposite signs and whose
/// coefficients change sign once, by regula falsi with the Illinois halving of the end that
/// stays (Dowell and Jarratt 1971).
fn bernstein_root(b: &[f64]) -> f64 {
    let last = b.len() - 1;
    let ((mut x0, mut f0), (mut x1, mut f1)) = ((0.0, b[0]), (1.0, b[last]));
    let mut kept = 0;
    for _ in 0..100 {
        let x = (x0 * f1 - x1 * f0) / (f1 - f0);
        if !(x > x0 && x < x1) || x1 - x0 <= 4.0 * f64::EPSILON {
            break;
        }
        let fx = bernstein_at(b, x);
        if fx == 0.0 {
            return x;
        }
        if (fx > 0.0) == (f0 > 0.0) {
            (x0, f0) = (x, fx);
            f1 = if kept == 1 { 0.5 * f1 } else { f1 };
            kept = 1;
        } else {
            (x1, f1) = (x, fx);
            f0 = if kept == -1 { 0.5 * f0 } else { f0 };
            kept = -1;
        }
    }
    // A step that lands on an end of the bracket found that end at the root, to rounding.
    let (g0, g1) = (bernstein_at(b, x0).abs(), bernstein_at(b, x1).abs());
    if g0 <= g1 { x0 } else { x1 }
}

/// The least and the greatest of `b`.
fn hull(b: &[f64]) -> (f64, f64) {
    b.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, m), &x| {
            (l.min(x), m.max(x))
        })
}

/// How many times the nonzero coefficients of `b` change sign: no more roots lie in the interval
/// than that (Descartes' rule of signs for the Bernstein basis).
fn sign_changes(b: &[f64]) -> usize {
    let mut last = 0.0f64;
    let mut changes = 0;
    for &x in b.iter().filter(|&&x| x != 0.0) {
        if last * x < 0.0 {
            changes += 1;
        }
        last = x;
    }
    changes
}

/// The middle of `b`'s interval, or a little short of it where the polynomial is zero there, so
/// a root is never left at the end of both halves.
fn bernstein_middle(b: &[f64]) -> f64 {
    if bernstein_at(b, 0.5) == 0.0 {
        0.4375
    } else {
        0.5
    }
}

/// [`plane_crossings`] of a spline, rational or not, of any degree, knot span by knot span. The
/// spline lies in its control values' hull and each span in its own `p + 1`, so most of it is
/// passed over on those alone.
fn spline_crossings(
    parts: SplineParts<'_>,
    range: (f64, f64),
    (o, n): (Point3, Vec3),
    tau: f64,
    near: &Aabb,
) -> Vec<f64> {
    let (p, _, ctrl, weights, _) = parts;
    // Most splines keep to one side: their control values say so before any are kept.
    let least = weights.map_or(1.0, |w| w.iter().copied().fold(f64::INFINITY, f64::min));
    let band = tau * least;
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for (j, &q) in ctrl.iter().enumerate() {
        let x = weights.map_or(1.0, |w| w[j]) * n.dot(q - o);
        (low, high) = (low.min(x), high.max(x));
    }
    if low >= -band || high <= band {
        return Vec::new();
    }
    let (g, _) = spline_values(parts, o, n);
    let one_side = |d: &[f64]| {
        let (l, m) = hull(d);
        l >= -band || m <= band
    };
    let mut out = Vec::new();
    for (i, shift, a, z) in spline_spans(parts, range) {
        if one_side(&g[i - p..=i]) {
            continue;
        }
        let mut span_box = Aabb::EMPTY;
        for &q in &ctrl[i - p..=i] {
            span_box.add(q);
        }
        if !meets(&span_box, near) {
            continue;
        }
        let bern = span_bernstein(parts, &g, i, (a - shift, z - shift));
        let mut roots = Vec::new();
        bernstein_crossings(&bern, (0.0, 1.0), band, 0, &mut roots);
        out.extend(roots.into_iter().map(|s| a + (z - a) * s));
    }
    out
}

/// The parameters in `(lo, hi)` where the polynomial with Bernstein coefficients `b` over that
/// interval changes sign, from beyond `band` on one side to beyond it on the other. The
/// polynomial lies within its coefficients' hull, so a piece whose coefficients keep within
/// `band` of one side holds no crossing; one whose coefficients change sign once, its ends on
/// either side, holds exactly one root, which [`bernstein_root`] finds; any other is split in two
/// and each half searched.
fn bernstein_crossings(b: &[f64], (lo, hi): (f64, f64), band: f64, depth: u32, out: &mut Vec<f64>) {
    let (least, most) = hull(b);
    if least >= -band || most <= band {
        return;
    }
    let last = b.len() - 1;
    if b[0] * b[last] < 0.0 && sign_changes(b) == 1 {
        out.push(lo + (hi - lo) * bernstein_root(b));
        return;
    }
    if depth >= 64 {
        if b[0] * b[last] < 0.0 {
            out.push(0.5 * (lo + hi));
        }
        return;
    }
    let s = bernstein_middle(b);
    let (left, right) = bernstein_split(b, s);
    let m = lo + (hi - lo) * s;
    bernstein_crossings(&left, (lo, m), band, depth + 1, out);
    bernstein_crossings(&right, (m, hi), band, depth + 1, out);
}

/// The parameters in `[0, 1]` where the polynomial `b` meets zero or comes within `band` of it
/// and turns back, each with its rank as a stand-in for the others along a stretch that keeps
/// within the band, and its distance from zero: its ends where they lie within the band, rank 0;
/// each sign change, found by [`bernstein_crossings`], rank 1; and each turn, a sign change of its
/// derivative, whose Bernstein coefficients are `n·(bⱼ₊₁ − bⱼ)`, where it lies within the band,
/// rank 2.
fn bernstein_zeros(b: &[f64], band: f64) -> Vec<(f64, u8, f64)> {
    let last = b.len() - 1;
    let mut out = Vec::new();
    if b[0].abs() <= band {
        out.push((0.0, 0, b[0].abs()));
    }
    if b[last].abs() <= band {
        out.push((1.0, 0, b[last].abs()));
    }
    let mut crossings = Vec::new();
    bernstein_crossings(b, (0.0, 1.0), 0.0, 0, &mut crossings);
    out.extend(
        crossings
            .into_iter()
            .map(|x| (x, 1, bernstein_at(b, x).abs())),
    );
    if last >= 2 {
        let d: Vec<f64> = b.windows(2).map(|w| last as f64 * (w[1] - w[0])).collect();
        let mut turns = Vec::new();
        bernstein_crossings(&d, (0.0, 1.0), 0.0, 0, &mut turns);
        for x in turns {
            let g = bernstein_at(b, x).abs();
            if g <= band {
                out.push((x, 2, g));
            }
        }
    }
    out
}

/// Whether the polynomial `b` keeps within `band` of zero over `[0, 1]`: its coefficients over a
/// piece bound it there, and a piece's end coefficients are its values, so halving settles it
/// unless it grazes the band's edge, which counts as leaving.
fn keeps_within(b: Vec<f64>, band: f64) -> bool {
    let mut stack = vec![(b, 0)];
    while let Some((q, depth)) = stack.pop() {
        let (l, m) = hull(&q);
        if l >= -band && m <= band {
            continue;
        }
        if depth >= 40 || q[0].abs() > band || q[q.len() - 1].abs() > band {
            return false;
        }
        let (left, right) = bernstein_split(&q, 0.5);
        stack.push((left, depth + 1));
        stack.push((right, depth + 1));
    }
    true
}

/// Where curve `c` comes within `tau` of the plane through `o` with unit normal `n` over
/// `[t0, t1]`, crossing it or touching it: closed forms for a line and a conic, and for a spline
/// [`bernstein_zeros`] on each knot span its control values do not keep off the band. Of a
/// spline's zeros joined by a stretch that keeps within the band, one stands for them all: an end
/// of the range, else a sign change, else the one nearest the plane. Without `ends`, an end of the
/// range within the band but off the plane stands only for a stretch that holds another zero.
/// Parameters nearer each other than a billionth of the range are one.
fn plane_zeros(
    c: &Curve,
    (t0, t1): (f64, f64),
    (o, n): (Point3, Vec3),
    tau: f64,
    ends: bool,
) -> Vec<f64> {
    let within = |t: f64| t >= t0 && t <= t1;
    let gap = 1e-9 * (t1 - t0).abs().min(1e9);
    let mut out = Vec::new();
    match c {
        Curve::Line(l) => {
            let (g0, gd) = (n.dot(l.origin - o), n.dot(l.dir));
            if gd != 0.0 && within(-g0 / gd) {
                out.push(-g0 / gd);
            }
        }
        Curve::Circle(_) | Curve::Ellipse(_) => {
            let (a, bc, bs) = conic_terms(c, o, n);
            let r = bc.hypot(bs);
            let phase = bs.atan2(bc);
            let roots: Vec<f64> = if (a + r).abs() <= tau {
                vec![phase + PI]
            } else if (a - r).abs() <= tau {
                vec![phase]
            } else if a.abs() < r {
                let half = (-a / r).acos();
                vec![phase - half, phase + half]
            } else {
                Vec::new()
            };
            for root in roots {
                let mut k = ((t0 - root) / TAU).ceil();
                while root + k * TAU <= t1 {
                    out.push(root + k * TAU);
                    k += 1.0;
                }
            }
        }
        _ => {
            let parts = spline_parts(c).expect("a spline");
            let p = parts.0;
            let (g, least) = spline_values(parts, o, n);
            let band = tau * least;
            let on_plane = |t: f64| n.dot(c.eval(t) - o) == 0.0;
            // Each zero with its rank, its distance from the plane, and whether it counts.
            let mut found: Vec<(f64, u8, f64, bool)> = Vec::new();
            for (i, shift, a, z) in spline_spans(parts, (t0, t1)) {
                let (l, m) = hull(&g[i - p..=i]);
                if l > band || m < -band {
                    continue;
                }
                let bern = span_bernstein(parts, &g, i, (a - shift, z - shift));
                for (x, rank, d) in bernstein_zeros(&bern, band) {
                    let t = a + (z - a) * x;
                    let end = [t0, t1].into_iter().find(|e| (t - e).abs() <= gap);
                    found.push(match (rank, end) {
                        (0, Some(e)) => (e, 0, d, ends || on_plane(e)),
                        (0, None) => (t, 2, d, true),
                        _ => (t, rank, d, true),
                    });
                }
            }
            found.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut kept: Vec<(f64, u8, f64, bool)> = Vec::new();
            for z in found {
                let joined = kept.last().is_some_and(|y| {
                    z.0 - y.0 <= gap
                        || spline_spans(parts, (y.0, z.0)).all(|(i, shift, a, b)| {
                            keeps_within(span_bernstein(parts, &g, i, (a - shift, b - shift)), band)
                        })
                });
                match kept.last_mut() {
                    Some(y) if joined => {
                        let counts = y.3 || z.3;
                        if (z.1, z.2) < (y.1, y.2) {
                            *y = z;
                        }
                        y.3 = counts;
                    }
                    _ => kept.push(z),
                }
            }
            return kept.into_iter().filter(|z| z.3).map(|z| z.0).collect();
        }
    }
    out.sort_by(f64::total_cmp);
    out.dedup_by(|x, y| (*x - *y).abs() <= gap);
    out
}

/// A conic's distance from the plane through `o` with unit normal `n` as `a + b cos t + s sin t`:
/// `(a, b, s)`.
fn conic_terms(c: &Curve, o: Point3, n: Vec3) -> (f64, f64, f64) {
    let (frame, rx, ry) = match c {
        Curve::Circle(k) => (&k.frame, k.radius, k.radius),
        Curve::Ellipse(k) => (&k.frame, k.major, k.minor),
        _ => unreachable!("a conic"),
    };
    (
        n.dot(frame.origin - o),
        rx * n.dot(frame.x),
        ry * n.dot(frame.y),
    )
}

/// Which side of the plane through `o` with unit normal `n` curve `c` lies on beside one end of
/// `(t0, t1)`, the far end when `from_end`: the sign of its distance where, going into the range
/// from that end, it first gets more than `tau` from the plane, or 0 where it never does. Its
/// tangent decides where it leaves the plane across it, and the first derivative across it that
/// is not zero where it leaves along it; the first exit past the band reads all of these at
/// once, and a curve that touches the plane and turns back leaves on the side it came from.
pub(crate) fn side_leaving(
    c: &Curve,
    (t0, t1): (f64, f64),
    from_end: bool,
    (o, n): (Point3, Vec3),
    tau: f64,
) -> i8 {
    let sign = |x: f64| {
        if x > tau {
            1
        } else if x < -tau {
            -1
        } else {
            0
        }
    };
    match c {
        Curve::Line(l) => {
            let far = if from_end { t0 } else { t1 };
            sign(n.dot(l.origin + l.dir * far - o))
        }
        Curve::Circle(_) | Curve::Ellipse(_) => {
            // The first parameter from the end, going in, where a + b cos t + s sin t = ±tau.
            let (a, bc, bs) = conic_terms(c, o, n);
            let r = bc.hypot(bs);
            let phase = bs.atan2(bc);
            let (start, dir) = if from_end { (t1, -1.0) } else { (t0, 1.0) };
            let mut best: Option<(f64, i8)> = None;
            for (level, side) in [(tau, 1), (-tau, -1)] {
                if r == 0.0 || ((level - a) / r).abs() > 1.0 {
                    continue;
                }
                let half = ((level - a) / r).acos();
                for root in [phase - half, phase + half] {
                    // How far from the start, going in, the next time it is reached.
                    let ahead = (dir * (root - start)).rem_euclid(TAU);
                    // Reaching the level from inside the band, going outward.
                    let t = start + dir * ahead;
                    let slope = -bc * t.sin() + bs * t.cos();
                    let outward = slope * dir * f64::from(side) > 0.0;
                    if ahead > 0.0
                        && ahead <= (t1 - t0)
                        && outward
                        && best.is_none_or(|b| ahead < b.0)
                    {
                        best = Some((ahead, side));
                    }
                }
            }
            best.map_or(0, |b| b.1)
        }
        _ => {
            let parts = spline_parts(c).expect("a spline");
            let (g, least) = spline_values(parts, o, n);
            let band = tau * least;
            let mut spans: Vec<_> = spline_spans(parts, (t0, t1)).collect();
            if from_end {
                spans.reverse();
            }
            for (i, shift, a, z) in spans {
                let mut bern = span_bernstein(parts, &g, i, (a - shift, z - shift));
                if from_end {
                    bern.reverse();
                }
                if let Some(side) = bernstein_exit(&bern, band, 0) {
                    return side;
                }
            }
            0
        }
    }
}

/// The sign of the polynomial `b` where, from the start of its interval, it first gets more than
/// `band` from zero; `None` where it stays within the band throughout. A piece whose coefficients
/// keep within the band never leaves it; otherwise the first half is searched before the second,
/// down to a piece no wider than rounding, where its larger end decides.
fn bernstein_exit(b: &[f64], band: f64, depth: u32) -> Option<i8> {
    let (least, most) = hull(b);
    if least >= -band && most <= band {
        return None;
    }
    let last = b.len() - 1;
    if depth >= 52 {
        let at = if b[0].abs() > band { b[0] } else { b[last] };
        return (at.abs() > band).then_some(if at > 0.0 { 1 } else { -1 });
    }
    let (left, right) = bernstein_split(b, 0.5);
    bernstein_exit(&left, band, depth + 1).or_else(|| bernstein_exit(&right, band, depth + 1))
}

/// The parameters of a surface of revolution's curve where it meets the axis: its poles. The
/// curve lies in a plane through the axis, and its distance from the axis there is its distance
/// from the plane through the axis square to that one, which [`plane_zeros`] solves exactly:
/// a line and a conic in closed form, a spline by its control values' bounds, where it crosses
/// the axis or touches it. None for a curve off every plane through the axis.
pub(crate) fn revolution_poles(r: &Revolution) -> Vec<f64> {
    let Some(z) = r.axis.unit() else {
        return Vec::new();
    };
    let o = r.origin;
    let points: Vec<Point3> = match &r.curve {
        Curve::Line(l) => vec![l.origin, l.origin + l.dir],
        Curve::Circle(k) => {
            let f = &k.frame;
            vec![
                f.origin + f.x * k.radius,
                f.origin + f.y * k.radius,
                f.origin - f.x * k.radius,
            ]
        }
        Curve::Ellipse(k) => {
            let f = &k.frame;
            vec![
                f.origin + f.x * k.major,
                f.origin + f.y * k.minor,
                f.origin - f.x * k.major,
            ]
        }
        Curve::Nurbs(n) => n.control_points().to_vec(),
        other => other
            .spline()
            .map_or(Vec::new(), |s| s.control_points().to_vec()),
    };
    let off = |p: Point3| (p - o) - z * (p - o).dot(z);
    let Some(far) = points
        .iter()
        .map(|&p| off(p))
        .max_by(|a, b| a.norm().total_cmp(&b.norm()))
    else {
        return Vec::new();
    };
    let Some(out) = far.unit() else {
        return Vec::new();
    };
    // Three points of a conic, or a spline's control points, fix its plane, or bound it.
    let normal = z.cross(out);
    if points
        .iter()
        .any(|&p| normal.dot(p - o).abs() > tol::LINEAR)
    {
        return Vec::new();
    }
    plane_zeros(&r.curve, r.curve.domain(), (o, out), tol::LINEAR, true)
}

/// Where a spline, rational or not, meets the plane through `o` with unit normal `n` over
/// `range`, as [`plane_zeros`] finds it, an end counting only where it lies on the plane, or that
/// it lies in the plane throughout, where its control values over the range all lie within `tau`
/// of it.
pub(crate) fn spline_plane_hits(
    c: &Curve,
    range: (f64, f64),
    (o, n): (Point3, Vec3),
    tau: f64,
) -> (Vec<f64>, bool) {
    let parts = spline_parts(c).expect("a spline");
    let p = parts.0;
    let (g, least) = spline_values(parts, o, n);
    let band = tau * least;
    let on = spline_spans(parts, range).all(|(i, ..)| g[i - p..=i].iter().all(|x| x.abs() <= band));
    if on {
        return (Vec::new(), true);
    }
    (plane_zeros(c, range, (o, n), tau, false), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::BSpline;

    /// A cubic Bézier over `[0, 1]` whose control points stand `heights` over the plane z = 0.
    fn bezier(heights: [f64; 4]) -> Curve {
        let ctrl = (0..4)
            .map(|i| Vec3::new(i as f64, 0.0, heights[i]))
            .collect();
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        Curve::BSpline(BSpline::new(3, knots, ctrl).unwrap())
    }

    #[test]
    fn a_spline_that_crosses_a_plane_a_hair_before_ending_on_it_meets_it_once() {
        // Its height is 3s²·10⁻³ − 3s·10⁻⁸ + 10⁻¹⁰ to first order in s = 1 − t: it crosses zero
        // about 10⁻⁵ before its end and keeps within 10⁻¹² of the plane from there on.
        let c = bezier([1.0, 1e-3, -1e-8, 1e-10]);
        let plane = (Point3::ZERO, Vec3::Z);
        let zeros = plane_zeros(&c, (0.0, 1.0), plane, tol::LINEAR, true);
        assert_eq!(zeros, vec![1.0]);
        let hits = spline_plane_hits(&c, (0.0, 1.0), plane, tol::LINEAR);
        assert_eq!(hits, (vec![1.0], false));
    }

    #[test]
    fn a_spline_ending_within_the_tolerance_of_a_plane_meets_it_only_where_ends_count() {
        let c = bezier([1.0, 1.0, 1.0, 1e-8]);
        let plane = (Point3::ZERO, Vec3::Z);
        let zeros = plane_zeros(&c, (0.0, 1.0), plane, tol::LINEAR, true);
        assert_eq!(zeros, vec![1.0]);
        let hits = spline_plane_hits(&c, (0.0, 1.0), plane, tol::LINEAR);
        assert_eq!(hits, (vec![], false));
    }
}
