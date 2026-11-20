//! Splitting a face by cuts in its parameter plane: the boundary trims and both directions of every
//! cut form a planar graph, whose regions, traced with the face's material on the left, are the
//! pieces of the face.

use std::f64::consts::TAU;

use super::domain::{Domain, Trim, UvMap, direction_angle, polygon_area, wrap_angle};
use crate::{Error, Result, geom::Surface, math::Vec2};

/// How far apart, in mm along the surface, two parameter-plane images of one vertex may be and
/// still be one node of the graph, or twice the vertex's local tolerance where that is wider: the
/// edges of an imported vertex end anywhere in its sphere. Images of a vertex at different nodes
/// (across a seam) are a period apart.
const NODE_GAP: f64 = 1e-5;

/// The direction a trim leaves its start in, as an exact tangent angle and, to order trims that
/// leave tangentially, the angle of a chord a little way along; and the oriented vector to its far
/// end, which orders trims whose tangents and chords agree to the last bit (two edges of solids
/// a hair apart leaving a shared corner a hair off parallel).
#[derive(Clone, Copy, Debug)]
struct Departure {
    tangent: f64,
    chord: f64,
    far: Vec2,
}

struct Graph<'a> {
    surface: &'a Surface,
    sigma: f64,
    scale: Vec2,
}

impl Graph<'_> {
    fn oriented(&self, d: Vec2) -> Vec2 {
        Vec2::new(self.sigma * d.x * self.scale.x, d.y * self.scale.y)
    }

    /// The departure, its chord reaching `reach` mm along the trim, or the whole trim if shorter;
    /// `node` is where the trim's node is, which its far end is measured from.
    fn departure(&self, tr: &Trim, reach: f64, node: Vec2) -> Departure {
        let chord_uv = match tr.map {
            UvMap::Affine { .. } => tr.uv1,
            UvMap::Projected | UvMap::Spline(_) => {
                tr.uv(self.surface, tr.param(fraction_at(tr, reach)), tr.uv0)
            }
        };
        let chord = direction_angle(self.oriented(chord_uv - tr.uv0)).unwrap_or(0.0);
        let tangent =
            direction_angle(self.oriented(tr.start_tangent(self.surface))).unwrap_or(chord);
        Departure {
            tangent,
            chord,
            far: self.oriented(tr.uv1 - node),
        }
    }

    /// What the trim adds to its loop's signed area, ½∫(u dv − v du) in the oriented plane: by
    /// Gauss–Legendre along the curve with the exact parameter-plane rate. Chords would sag from a
    /// curved trim by more than a sliver between two such trims is wide, and turn its sign. The
    /// rule is applied on halves of the range until halving no longer changes the sum: beside a
    /// pole the angle about it turns half round over a short stretch of the trim, which one rule
    /// over the whole trim does not resolve.
    fn swept_area(&self, tr: &Trim) -> f64 {
        let chord = |a: Vec2, b: Vec2| {
            let (p, q) = (self.oriented(a), self.oriented(b));
            0.5 * (p.x * q.y - p.y * q.x)
        };
        if matches!(tr.map, UvMap::Affine { .. }) || tr.curve.is_none() {
            return chord(tr.uv0, tr.uv1);
        }
        let mut nodes: Vec<(f64, f64)> = crate::numeric::gauss(24).to_vec();
        nodes.sort_by(|a, b| a.0.total_cmp(&b.0));
        // The rule over [a, b], each point's parameters found from the one before, from `hint` at
        // `a`; and the parameters at `b`, which the next piece starts from.
        let rule = |a: f64, b: f64, mut hint: Vec2| -> (f64, Vec2) {
            let (m, h) = (0.5 * (a + b), 0.5 * (b - a));
            let mut sum = 0.0;
            for &(x, w) in &nodes {
                let t = m + h * x;
                let uv = tr.uv(self.surface, t, hint);
                hint = uv;
                let rate = tr.uv_rate_at(self.surface, t, uv);
                let (p, d) = (self.oriented(uv), self.oriented(rate));
                sum += w * (p.x * d.y - p.y * d.x);
            }
            (0.5 * sum * h, tr.uv(self.surface, b, hint))
        };
        fn adapt(
            rule: &dyn Fn(f64, f64, Vec2) -> (f64, Vec2),
            (a, b): (f64, f64),
            hint: Vec2,
            whole: f64,
            depth: usize,
        ) -> (f64, Vec2) {
            let m = 0.5 * (a + b);
            let (left, at_m) = rule(a, m, hint);
            let (right, at_b) = rule(m, b, at_m);
            if depth == 0 || (left + right - whole).abs() <= 1e-13 * (1.0 + whole.abs()) {
                return (left + right, at_b);
            }
            let (left, at_m) = adapt(rule, (a, m), hint, left, depth - 1);
            let (right, at_b) = adapt(rule, (m, b), at_m, right, depth - 1);
            (left + right, at_b)
        }
        let (whole, _) = rule(tr.t0, tr.t1, tr.uv0);
        adapt(&rule, (tr.t0, tr.t1), tr.uv0, whole, 20).0
    }

    fn metric(&self, a: Vec2, b: Vec2) -> f64 {
        (a.x - b.x).abs() * self.scale.x + (a.y - b.y).abs() * self.scale.y
    }
}

/// A trim's length along its curve, from a few chords.
fn trim_length(tr: &Trim) -> f64 {
    let n = 16;
    (0..n)
        .map(|k| {
            let (a, b) = (k as f64 / n as f64, (k + 1) as f64 / n as f64);
            tr.point(tr.param(a)).dist(tr.point(tr.param(b)))
        })
        .sum()
}

/// The fraction of the trim at which it is first `reach` mm from its start, or 1 if it never is.
/// Chord angles grow with chord length, so two trims bending alike are compared at exactly equal
/// distances: a fraction from the trim's length alone is off by its parameter's unevenness.
fn fraction_at(tr: &Trim, reach: f64) -> f64 {
    let start = tr.point(tr.t0);
    let away = |f: f64| tr.point(tr.param(f)).dist(start);
    let guess = (reach / trim_length(tr).max(1e-300)).min(1.0);
    let mut hi = guess;
    while away(hi) < reach {
        if hi >= 1.0 {
            return 1.0;
        }
        hi = (2.0 * hi).min(1.0);
    }
    let mut lo = 0.0;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if away(mid) < reach {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn signed_angle(a: f64) -> f64 {
    let w = wrap_angle(a);
    if w > TAU / 2.0 { w - TAU } else { w }
}

/// Second-order weight of a departure's bend: it orders departures whose tangents agree, and is
/// too small to reorder any whose tangents differ by more than a few 1e-7 rad.
const BEND: f64 = 1e-7;

/// The clockwise turn from `back` to `c`, in (0, 2π]. Of departures along one tangent, the one
/// bending further counter-clockwise is met first.
fn clockwise_turn(back: Departure, c: Departure) -> f64 {
    let d = wrap_angle(back.tangent - c.tangent);
    // A fitted curve's tangent is good only to its fit, so tangents closer than the trims bend
    // over their chords are a tie that the chords decide.
    let eps = 1e-9_f64.max(bend(back) + bend(c));
    if signed_angle(d).abs() > eps {
        return d - BEND * signed_angle(c.chord - c.tangent);
    }
    let d2 = signed_angle(back.chord - c.chord);
    if d2.abs() <= 1e-13 && signed_angle(d).abs() <= 1e-13 {
        // `c` a hair counter-clockwise of `back` is the last turn, a hair clockwise the first.
        let cross = back.far.x * c.far.y - back.far.y * c.far.x;
        return if cross > 0.0 {
            TAU - BEND * 1e-9
        } else {
            BEND * 1e-9
        };
    }
    if d2 > 0.0 { d2 * BEND } else { TAU + d2 * BEND }
}

/// How far a departure's chord turns from its tangent.
fn bend(x: Departure) -> f64 {
    signed_angle(x.chord - x.tangent).abs()
}

/// Which of two departures is met first turning clockwise, their turns from the arrival being
/// `ka` and `kb`. Two whose turns are closer than they bend over their chords are ordered by the
/// chords of `ab`, both taken at one reach.
fn turn_order(ka: f64, kb: f64, ab: impl Fn() -> (Departure, Departure)) -> std::cmp::Ordering {
    let (a, b) = ab();
    if (ka - kb).abs() > 1e-6_f64.max(bend(a) + bend(b)) {
        return ka.total_cmp(&kb);
    }
    let same = |x: f64, y: f64| signed_angle(x - y).abs() <= 1e-13;
    if same(ka, kb) && same(a.tangent, b.tangent) && same(a.chord, b.chord) {
        // Of two leaving along one line to the last bit, the further counter-clockwise is met
        // first turning clockwise.
        let cross = a.far.x * b.far.y - a.far.y * b.far.x;
        return cross.total_cmp(&0.0);
    }
    let eps = bend(a) + bend(b);
    let clear = |k: f64| k > eps && k < TAU - eps;
    if (ka - kb).abs() <= eps && clear(ka) && clear(kb) {
        let by_chord = |k: f64, x: Departure| k + signed_angle(x.tangent - x.chord);
        return by_chord(ka, a).total_cmp(&by_chord(kb, b));
    }
    ka.total_cmp(&kb)
}

/// Whether `b` runs back along `a`: the same piece the other way, between the same images.
fn retraces(a: &Trim, b: &Trim, g: &Graph) -> bool {
    a.piece == b.piece
        && a.forward != b.forward
        && a.v0 == b.v1
        && a.v1 == b.v0
        && g.metric(a.uv0, b.uv1) <= NODE_GAP
        && g.metric(a.uv1, b.uv0) <= NODE_GAP
}

/// `lp` without the spikes in it: trims that run out along a piece and straight back. A boundary
/// tangent to itself at a corner (a circle meeting a line it touches) bounds a sliver no wider than
/// the tolerance between them; once a vertex of the other solid lands in that sliver, the line and
/// the circle up to it are one edge, and the loop runs out along it and back.
fn without_spikes(lp: Vec<Trim>, g: &Graph) -> Vec<Trim> {
    let mut out: Vec<Trim> = Vec::with_capacity(lp.len());
    for t in lp {
        if out.last().is_some_and(|l| retraces(l, &t, g)) {
            out.pop();
        } else {
            out.push(t);
        }
    }
    while out.len() >= 2 && retraces(&out[out.len() - 1], &out[0], g) {
        out.pop();
        out.remove(0);
    }
    out
}

/// The pieces of a face with `boundary` loops cut by `cuts` (each given in one direction), each as
/// its loops, outer first. A cut `slit` names may end where nothing else does: a contact, where
/// another solid touches the face along a curve. The face's loop then runs along it and back, or
/// it is a loop of its own inside the face, both ways along it.
pub(crate) fn split(
    surface: &Surface,
    reversed: bool,
    boundary: Vec<Vec<Trim>>,
    cuts: Vec<Trim>,
    vertex_tol: &dyn Fn(usize) -> f64,
    slit: &dyn Fn(usize) -> bool,
) -> Result<Vec<Vec<Vec<Trim>>>> {
    if cuts.is_empty() {
        return Ok(vec![boundary]);
    }
    let whole = Domain::new(surface.clone(), reversed, boundary.clone());
    let g = Graph {
        surface,
        sigma: if reversed { -1.0 } else { 1.0 },
        scale: whole.scale(),
    };
    let boundary: Vec<Vec<Trim>> = boundary
        .into_iter()
        .map(|lp| without_spikes(lp, &g))
        .filter(|lp| !lp.is_empty())
        .collect();
    let mut hs: Vec<Trim> = Vec::new();
    let mut twin: Vec<Option<usize>> = Vec::new();
    for tr in boundary.into_iter().flatten() {
        hs.push(tr);
        twin.push(None);
    }
    for c in cuts {
        let i = hs.len();
        hs.push(c.reversed());
        hs.push(c);
        twin.push(Some(i + 1));
        twin.push(Some(i));
    }
    let mut alive = vec![true; hs.len()];

    // Nodes: a vertex at one place in the parameter plane.
    let mut nodes: Vec<(usize, Vec2)> = Vec::new();
    let node_of = |v: usize, uv: Vec2, nodes: &mut Vec<(usize, Vec2)>| -> usize {
        if let Some(i) = nodes
            .iter()
            .position(|&(w, p)| w == v && g.metric(p, uv) <= NODE_GAP.max(2.0 * vertex_tol(v)))
        {
            return i;
        }
        nodes.push((v, uv));
        nodes.len() - 1
    };
    let mut start = Vec::with_capacity(hs.len());
    let mut end = Vec::with_capacity(hs.len());
    for h in &hs {
        start.push(node_of(h.v0, h.uv0, &mut nodes));
        end.push(node_of(h.v1, h.uv1, &mut nodes));
    }

    // A cut that ends where nothing else does bounds no region, unless it is a slit.
    loop {
        let mut degree = vec![0usize; nodes.len()];
        for h in 0..hs.len() {
            if alive[h] {
                degree[start[h]] += 1;
                degree[end[h]] += 1;
            }
        }
        let mut changed = false;
        for h in 0..hs.len() {
            if let Some(t) = twin[h]
                && alive[h]
                && !slit(hs[h].piece)
                && start[h] != end[h]
                && (degree[start[h]] == 2 || degree[end[h]] == 2)
            {
                alive[h] = false;
                alive[t] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut out_of: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for h in 0..hs.len() {
        if alive[h] {
            out_of[start[h]].push(h);
        }
    }
    // Chords compared reach equally far, so departures along one tangent are ordered by how they
    // bend, not by how long they are; a straight trim's chord is the whole of it.
    let lengths: Vec<f64> = hs
        .iter()
        .map(|h| match h.map {
            UvMap::Projected | UvMap::Spline(_) => trim_length(h),
            UvMap::Affine { .. } => f64::INFINITY,
        })
        .collect();
    let next = |h: usize| -> Option<usize> {
        let outs = &out_of[end[h]];
        // A boundary only turns straight back along the edge it came by at a dead end: a cut's
        // other side, or the face's own loop running back down a contact it already holds,
        // which leaves at exactly the arrival's tangent and so would read as the sharpest turn.
        let back = |o: usize| {
            Some(o) == twin[h]
                || (hs[o].piece == hs[h].piece
                    && hs[o].forward != hs[h].forward
                    && start[o] == end[h]
                    && end[o] == start[h])
        };
        let others: Vec<usize> = outs.iter().copied().filter(|&o| !back(o)).collect();
        let pool = if others.is_empty() {
            outs.clone()
        } else {
            others
        };
        // Each turn from the arrival is read with chords a twentieth of the shorter of the two
        // trims, so a sliver trim elsewhere at the node cannot shorten them until rounding
        // outweighs how two trims leaving along nearly one tangent part. Turns that come out
        // close are read again with one reach, a twentieth of the shortest of the three.
        let at = nodes[end[h]].1;
        let dep = |t: &Trim, r: f64| g.departure(t, r.min(1e300), at);
        let turn = |c: usize, r: f64| clockwise_turn(dep(&hs[h].reversed(), r), dep(&hs[c], r));
        let own: Vec<(usize, f64, f64)> = pool
            .into_iter()
            .map(|c| {
                let r = 0.05 * lengths[h].min(lengths[c]);
                (c, turn(c, r), bend(dep(&hs[c], r)))
            })
            .collect();
        own.into_iter()
            .min_by(|&(a, ka, ba), &(b, kb, bb)| {
                // A fitted cut's tangent is good only to its fit, so turns closer than the two
                // departures bend over their chords are read again by the chords.
                if (ka - kb).abs() > 1e-6_f64.max(ba + bb) {
                    return ka.total_cmp(&kb);
                }
                // Two turning well away from the arrival are ordered between themselves; the
                // arrival's length has no say in their reach.
                let clear = |k: f64| k > 1e-3 && k < TAU - 1e-3;
                let pair = 0.05 * lengths[a].min(lengths[b]);
                let r = if clear(ka) && clear(kb) {
                    pair
                } else {
                    pair.min(0.05 * lengths[h])
                };
                let back_h = dep(&hs[h].reversed(), r);
                let (la, lb) = (dep(&hs[a], r), dep(&hs[b], r));
                turn_order(
                    clockwise_turn(back_h, la),
                    clockwise_turn(back_h, lb),
                    || (la, lb),
                )
            })
            .map(|x| x.0)
    };

    let mut seen = vec![false; hs.len()];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for h0 in 0..hs.len() {
        if !alive[h0] || seen[h0] {
            continue;
        }
        let mut cyc = Vec::new();
        let mut h = h0;
        loop {
            if seen[h] {
                if h == h0 {
                    break;
                }
                return Err(Error::Numerical(
                    "face split: a region's boundary does not close".into(),
                ));
            }
            seen[h] = true;
            cyc.push(h);
            h = next(h).ok_or_else(|| {
                Error::Numerical("face split: a boundary ends at a node with no way on".into())
            })?;
        }
        cycles.push(cyc);
    }

    // A loop's area from 16 chords a trim, with its error judged against 8: where that error is
    // a fair share of the area (a sliver between two curved trims, which chords sag across), the
    // exact integral decides the sign instead. Chords follow a curve a hair from a pole, where a
    // fixed rule's points would not.
    let area = |cyc: &[usize]| -> f64 {
        let (mut fine, mut coarse) = (Vec::new(), Vec::new());
        for &h in cyc {
            let n = match hs[h].map {
                UvMap::Affine { .. } => 1,
                UvMap::Projected | UvMap::Spline(_) => 16,
            };
            let pl = hs[h].polyline(surface, n);
            for (k, &p) in pl[..pl.len() - 1].iter().enumerate() {
                fine.push(g.oriented(p));
                if k % 2 == 0 {
                    coarse.push(g.oriented(p));
                }
            }
        }
        let a = polygon_area(&fine);
        if a.abs() > 4.0 * (a - polygon_area(&coarse)).abs() / 3.0 {
            return a;
        }
        cyc.iter().map(|&h| g.swept_area(&hs[h])).sum()
    };
    // A cycle running only along slits and back, each trim's twin in it too, encloses nothing: a
    // slit inside the face, a loop of its own.
    let only_slits = |cyc: &[usize]| {
        cyc.iter()
            .all(|&h| twin[h].is_some_and(|t| cyc.contains(&t)))
    };
    let areas: Vec<f64> = cycles
        .iter()
        .map(|c| if only_slits(c) { 0.0 } else { area(c) })
        .collect();
    let loops_of = |cyc: &[usize]| -> Vec<Trim> { cyc.iter().map(|&h| hs[h].clone()).collect() };
    let outers: Vec<usize> = (0..cycles.len()).filter(|&i| areas[i] > 0.0).collect();
    let outer_domains: Vec<Domain> = outers
        .iter()
        .map(|&i| Domain::new(surface.clone(), reversed, vec![loops_of(&cycles[i])]))
        .collect();
    let mut pieces: Vec<Vec<Vec<Trim>>> =
        outers.iter().map(|&i| vec![loops_of(&cycles[i])]).collect();
    for (i, cyc) in cycles.iter().enumerate() {
        if areas[i] > 0.0 {
            continue;
        }
        let first = &hs[cyc[0]];
        let probe = first.mid_uv(surface);
        // A region on the other side of any of the hole's pieces cannot hold it.
        let pieces_of = |c: &[usize]| -> Vec<usize> { c.iter().map(|&h| hs[h].piece).collect() };
        let mine = pieces_of(cyc);
        let host = (0..outers.len())
            .filter(|&k| {
                !pieces_of(&cycles[outers[k]])
                    .iter()
                    .any(|p| mine.contains(p))
            })
            // A region winding past a period holds the probe at one of its images.
            .filter(|&k| outer_domains[k].inside_image(probe).is_some())
            .min_by(|&a, &b| areas[outers[a]].total_cmp(&areas[outers[b]]));
        match host {
            Some(k) => pieces[k].push(loops_of(cyc)),
            None => {
                return Err(Error::Numerical(
                    "face split: a hole lies in no region".into(),
                ));
            }
        }
    }
    Ok(pieces)
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::f64::consts::PI;

    use super::*;

    // A great circle of the unit sphere from (0, −1, 0) to (0, 1, 0) that passes 0.01 rad from the
    // south pole: its longitude turns half round within a few hundredths of its length there. What
    // it adds to a loop's area, ½∫(u dv − v du), is the sum of the chords of 2·10⁵ points read in
    // closed form, u = atan2(y, x) followed round and v = asin z.
    #[test]
    fn a_trim_passing_beside_a_pole_adds_its_area_exactly() {
        use crate::geom::{Circle, Curve, Sphere};
        use crate::math::{Frame, Vec3};
        let eps: f64 = 0.01;
        let surface = Surface::Sphere(Sphere {
            frame: Frame::WORLD,
            radius: 1.0,
        });
        let frame = Frame::new(
            Vec3::ZERO,
            Vec3::new(eps.cos(), 0.0, eps.sin()),
            Vec3::new(eps.sin(), 0.0, -eps.cos()),
        )
        .unwrap();
        let curve = Curve::Circle(Circle { frame, radius: 1.0 });
        let (t0, t1) = (-PI / 2.0, PI / 2.0);
        let lon_lat = |t: f64| {
            let p = curve.eval(t);
            Vec2::new(p.y.atan2(p.x), p.z.clamp(-1.0, 1.0).asin())
        };
        let n = 200_000;
        let mut want = 0.0;
        let mut last = lon_lat(t0);
        for k in 1..=n {
            let mut uv = lon_lat(t0 + (t1 - t0) * k as f64 / n as f64);
            uv.x += ((last.x - uv.x) / TAU).round() * TAU;
            want += 0.5 * (last.x * uv.y - last.y * uv.x);
            last = uv;
        }
        let tr = Trim {
            piece: 0,
            forward: true,
            curve: Some(std::rc::Rc::new(curve.clone())),
            pole: Vec3::ZERO,
            t0,
            t1,
            map: UvMap::Projected,
            uv0: lon_lat(t0),
            uv1: last,
            v0: 0,
            v1: 1,
        };
        let g = Graph {
            surface: &surface,
            sigma: 1.0,
            scale: Vec2::new(1.0, 1.0),
        };
        let got = g.swept_area(&tr);
        assert!((got - want).abs() < 1e-6, "{got} {want}");
    }

    fn order(back: Departure, a: Departure, b: Departure) -> Ordering {
        turn_order(clockwise_turn(back, a), clockwise_turn(back, b), || (a, b))
    }

    fn dep(tangent: f64, chord: f64) -> Departure {
        Departure {
            tangent,
            chord,
            far: Vec2::ZERO,
        }
    }

    #[test]
    fn near_tangent_departures_are_ordered_by_their_chords() {
        // Arriving from below: a straight cut leaving along -x, and a fitted curve whose tangent
        // is a hair above -x but whose chord bends below it, so it is the first clockwise.
        let back = dep(1.5 * PI, 1.5 * PI);
        let line = dep(PI, PI);
        let curve = dep(PI - 3e-8, PI + 2e-3);
        assert_eq!(order(back, curve, line), Ordering::Less);
        assert_eq!(order(back, line, curve), Ordering::Greater);
        // Tangents apart by more than the bends still decide.
        let apart = dep(PI - 0.1, PI - 0.1 + 2e-3);
        assert_eq!(order(back, apart, line), Ordering::Greater);
        // A curve leaving along the arrival's own line, bending away below it, is met just after
        // it rather than a full turn later.
        let back = dep(0.0, 0.0);
        let along = dep(7e-7, -0.04);
        assert!(clockwise_turn(back, along) < 1e-6);
    }

    #[test]
    fn a_fitted_tangent_off_by_more_than_a_micro_radian_yields_to_its_chord() {
        // A cut leaving a corner tangent to the boundary edge it bends away from (a handle's end
        // touching a body at one point): its fitted tangent lies 1.6e-6 rad on the wrong side of
        // the edge, but its chord bends 9e-3 rad inside the face, so it is met first.
        let back = dep(PI, PI);
        let edge = dep(0.5 * PI, 0.5 * PI);
        let cut = dep(0.5 * PI - 1.6e-6, 0.5 * PI + 9e-3);
        assert_eq!(order(back, cut, edge), Ordering::Less);
        assert_eq!(order(back, edge, cut), Ordering::Greater);
    }
}
