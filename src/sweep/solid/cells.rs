//! A body drawn in the plane of the two lengths a turn keeps, its distance from the axis and its
//! height along it. Each point of the plane is an orbit, a circle about the axis. The body's edges
//! and characteristic curves, drawn there, cut the plane into cells; over a cell the orbits meet
//! the same pieces of the body's faces in the same order, so the body is a stack of intervals of
//! angle over each cell, and what it sweeps is each interval lengthened by the turn and merged
//! with the ones it reaches.

use std::f64::consts::PI;
use std::rc::Rc;

use super::cut::Cut;
use super::moving::graph::{self, Link, Region};
use crate::{
    Result,
    boolean::{
        csi,
        domain::{Domain, Where},
        sew::PieceGeom,
    },
    geom::{Circle, Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
};

/// Samples along each piece's drawing.
const SAMPLES: usize = 96;

/// The turn's axis as a frame: orbits are circles about its z, angles read from its x.
#[derive(Clone, Copy)]
pub(super) struct Axis {
    pub frame: Frame,
}

impl Axis {
    pub(super) fn new(origin: Point3, axis: Vec3) -> Result<Axis> {
        Ok(Axis {
            frame: Frame::from_axis(origin, axis)?,
        })
    }

    /// Where `p` is drawn, and its angle about the axis within half a turn of `near`.
    pub(super) fn drawn(&self, p: Point3, near: f64) -> (Vec2, f64) {
        let l = self.frame.to_local(p);
        let phase = l.y.atan2(l.x);
        let phase = phase + 2.0 * PI * ((near - phase) / (2.0 * PI)).round();
        (Vec2::new(l.x.hypot(l.y), l.z), phase)
    }

    /// The orbit drawn at `o`, its parameter the angle about the axis.
    pub(super) fn orbit(&self, o: Vec2) -> Curve {
        Curve::Circle(Circle {
            frame: self.frame.translated(self.frame.z * o.y),
            radius: o.x,
        })
    }

    /// The velocity of the point at `p` a radian.
    pub(super) fn velocity(&self, p: Point3) -> Vec3 {
        self.frame.z.cross(p - self.frame.origin)
    }
}

/// A curve where the body's surface at the end of the turn meets its surface at the start: where
/// an orbit leaves the body by a face, by its place among the body's, and comes back into it by
/// another exactly a turn on. Across its drawing a gap along the orbits closes or opens. The
/// curve is where the orbits come back in.
#[derive(Clone, Debug)]
pub(super) struct Meet {
    pub curve: Rc<Curve>,
    pub range: (f64, f64),
    pub enter: usize,
    pub exit: usize,
}

/// A stretch of a piece of the cut body whose drawing is a link: the piece, and its parameters
/// at the link's two ends.
#[derive(Clone, Copy, Debug)]
pub(super) struct Gen {
    pub piece: usize,
    pub t: (f64, f64),
}

/// Where an orbit passes through a part of the body's surface.
#[derive(Clone, Copy, Debug)]
pub(super) struct Hit {
    pub phase: f64,
    pub part: usize,
    pub uv: Vec2,
    /// Whether the orbit goes into the body there.
    pub enters: bool,
}

/// The drawing: nodes, links between them, the stretches of piece each link draws, and the cells.
pub(super) struct Cells {
    pub axis: Axis,
    pub nodes: Vec<Vec2>,
    pub links: Vec<Link>,
    pub gens: Vec<Vec<Gen>>,
    /// Where along its first stretch each point of each link's polyline is, from 0 to 1.
    pub fracs: Vec<Vec<f64>>,
    /// The face each characteristic curve of the cut body lies on, by piece.
    grazes: Vec<Option<Surface>>,
    /// The cut body's pieces and, from `first_meet` on, the curves where the end of the turn
    /// meets its start; the range of the whole curve each is of; and those curves' faces.
    pub pieces: Vec<PieceGeom>,
    pub whole: Vec<(f64, f64)>,
    pub first_meet: usize,
    pub meets: Vec<Meet>,
    pub regions: Vec<Region>,
    pub domains: Vec<Domain>,
    pub size: f64,
}

/// One piece's drawing.
struct Drawing {
    piece: usize,
    ts: Vec<f64>,
    pts: Vec<Vec2>,
}

impl Drawing {
    fn last(&self) -> &Vec2 {
        &self.pts[self.pts.len() - 1]
    }

    fn length(&self) -> f64 {
        self.pts.windows(2).map(|w| w[0].dist(w[1])).sum()
    }
}

/// The distance from `p` to the polyline `pts`.
fn off(pts: &[Vec2], p: Vec2) -> f64 {
    pts.windows(2)
        .map(|w| {
            let d = w[1] - w[0];
            let t = ((p - w[0]).dot(d) / d.dot(d).max(1e-300)).clamp(0.0, 1.0);
            p.dist(w[0] + d * t)
        })
        .fold(f64::INFINITY, f64::min)
}

impl Cells {
    pub(super) fn new(cut: &Cut, axis: Axis, meets: Vec<Meet>) -> Result<Cells> {
        let mut bounds = crate::math::Aabb::EMPTY;
        cut.pool.iter().for_each(|&p| bounds.add(p));
        let size = bounds.diagonal();
        // The cut body's pieces, and after them the curves where the end of the turn meets its
        // start, each between two points of its own.
        let mut pieces = cut.pieces.clone();
        let mut whole = cut.whole.clone();
        let mut pool = cut.pool.clone();
        for m in &meets {
            pool.push(m.curve.eval(m.range.0));
            pool.push(m.curve.eval(m.range.1));
            pieces.push(PieceGeom {
                curve: Some(m.curve.clone()),
                range: m.range,
                v0: pool.len() - 2,
                v1: pool.len() - 1,
                tol: crate::tol::LINEAR,
            });
            whole.push(m.range);
        }
        let pieces = &pieces;
        let near = 1e-9 * size;
        // The face each characteristic curve lies on: its points are read on the face itself.
        let mut grazes: Vec<Option<Surface>> = vec![None; pieces.len()];
        for part in &cut.parts {
            for &k in &cut.curves[part.face] {
                grazes[k] = Some(part.surface.clone());
            }
        }
        let at = |piece: usize, t: f64| -> Vec2 {
            axis.drawn(exact(pieces, &axis, &grazes, piece, t), 0.0).0
        };
        let mut nodes: Vec<Vec2> = Vec::new();
        let node = |p: Vec2, nodes: &mut Vec<Vec2>| -> usize {
            match nodes.iter().position(|q| q.dist(p) <= near) {
                Some(k) => k,
                None => {
                    nodes.push(p);
                    nodes.len() - 1
                }
            }
        };
        let of_vertex: Vec<usize> = pool
            .iter()
            .map(|&p| node(axis.drawn(p, 0.0).0, &mut nodes))
            .collect();
        // A piece that lies along an orbit is drawn as a point, and is no link.
        let drawings: Vec<Drawing> = pieces
            .iter()
            .enumerate()
            .filter(|(_, pg)| pg.curve.is_some())
            .map(|(piece, pg)| {
                // Even steps, then each chord halved while the drawing strays from it: a
                // drawing turns sharply where its curve runs along an orbit.
                let mut ts: Vec<f64> = (0..=SAMPLES)
                    .map(|k| pg.range.0 + (pg.range.1 - pg.range.0) * k as f64 / SAMPLES as f64)
                    .collect();
                let mut pts: Vec<Vec2> = ts.iter().map(|&t| at(piece, t)).collect();
                for _ in 0..12 {
                    let (mut nt, mut np) = (vec![ts[0]], vec![pts[0]]);
                    let mut grew = false;
                    for i in 1..ts.len() {
                        let tm = 0.5 * (ts[i - 1] + ts[i]);
                        let pm = at(piece, tm);
                        if pm.dist(pts[i - 1].lerp(pts[i], 0.5)) > 1e-5 * size {
                            nt.push(tm);
                            np.push(pm);
                            grew = true;
                        }
                        nt.push(ts[i]);
                        np.push(pts[i]);
                    }
                    (ts, pts) = (nt, np);
                    if !grew {
                        break;
                    }
                }
                Drawing { piece, ts, pts }
            })
            .filter(|d| d.length() > 1e-7 * size)
            .collect();
        let same: Vec<usize> = (0..drawings.len()).collect();
        // Where two drawings cross away from their ends, each is cut.
        let mut cuts: Vec<Vec<(f64, usize)>> = vec![Vec::new(); drawings.len()];
        for i in 0..drawings.len() {
            if same[i] != i {
                continue;
            }
            for j in i..drawings.len() {
                if same[j] != j {
                    continue;
                }
                let (a, b) = (&drawings[i], &drawings[j]);
                for m in 0..a.pts.len() - 1 {
                    let from = if i == j { m + 2 } else { 0 };
                    for n in from..b.pts.len() - 1 {
                        let (p, q) = (a.pts[m], b.pts[n]);
                        let (dp, dq) = (a.pts[m + 1] - p, b.pts[n + 1] - q);
                        let det = dp.cross(dq);
                        if det.abs() <= 1e-9 * dp.norm() * dq.norm() {
                            continue;
                        }
                        let (s, t) = ((q - p).cross(dq) / det, (q - p).cross(dp) / det);
                        if !(0.0..1.0).contains(&s) || !(0.0..1.0).contains(&t) {
                            continue;
                        }
                        let seed = [
                            a.ts[m] + (a.ts[m + 1] - a.ts[m]) * s,
                            b.ts[n] + (b.ts[n + 1] - b.ts[n]) * t,
                        ];
                        let Some((x, ts)) = crossing(&at, [a, b], seed, size) else {
                            continue;
                        };
                        let ends = [a.pts[0], *a.last(), b.pts[0], *b.last()];
                        if ends.iter().any(|e| e.dist(x) <= 1e-6 * size) {
                            continue;
                        }
                        let k = node(x, &mut nodes);
                        for (d, t) in [(i, ts[0]), (j, ts[1])] {
                            if !cuts[d].iter().any(|c| c.1 == k) {
                                cuts[d].push((t, k));
                            }
                        }
                    }
                }
            }
        }
        // A drawing turns back on itself where its curve runs along an orbit: the surface the
        // curve sweeps folds there, and which side of it the body lies on changes, so the place
        // is a node. The velocity crossed with the curve's direction turns over there.
        for (i, d) in drawings.iter().enumerate() {
            let pg = &pieces[d.piece];
            // The curve's direction by differences a thousandth of its length apart, the step's
            // own error taken off (Richardson): a finer step loses the direction in the rounding
            // of the points, and the fold would be placed a millionth of the curve off.
            let h = 1e-3 * (pg.range.1 - pg.range.0);
            let slope = |t: f64, h: f64| {
                (exact(pieces, &axis, &grazes, d.piece, t + h)
                    - exact(pieces, &axis, &grazes, d.piece, t - h))
                    / (2.0 * h)
            };
            let cross = |t: f64| {
                let p = exact(pieces, &axis, &grazes, d.piece, t);
                let dp = (slope(t, h) * 4.0 - slope(t, 2.0 * h)) / 3.0;
                let v = axis.velocity(p);
                (v.cross(dp), v.norm() * dp.norm())
            };
            let mut last: Option<(f64, Vec3)> = None;
            for &t in &d.ts[1..d.ts.len() - 1] {
                let (w, scale) = cross(t);
                if w.norm() <= 1e-9 * scale {
                    continue;
                }
                if let Some((a, l)) = last
                    && l.dot(w) < 0.0
                {
                    let (mut lo, mut hi) = (a, t);
                    for _ in 0..80 {
                        let mid = 0.5 * (lo + hi);
                        if cross(mid).0.dot(l) > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let mut at_fold = 0.5 * (lo + hi);
                    // An outline folds where its orbit touches the face to the second order:
                    // the orbit's gap from the face, which has no first term anywhere on the
                    // outline, has no second term either. A shadow closes on the outline by that
                    // same term, so the fold is placed by it, between the places either side.
                    if let Some(s) = &grazes[d.piece] {
                        let bend = |t: f64| {
                            let (o, touch) =
                                axis.drawn(exact(pieces, &axis, &grazes, d.piece, t), 0.0);
                            let orbit = axis.orbit(o);
                            let even = |h: f64| {
                                (csi::signed_distance(s, orbit.eval(touch + h))
                                    + csi::signed_distance(s, orbit.eval(touch - h)))
                                    / (h * h)
                            };
                            (4.0 * even(1e-3) - even(2e-3)) / 3.0
                        };
                        let w = 1e-4 * (pg.range.1 - pg.range.0);
                        let (mut lo, mut hi) = (at_fold - w, at_fold + w);
                        let sign = bend(lo) > 0.0;
                        if (bend(hi) > 0.0) != sign {
                            for _ in 0..60 {
                                let mid = 0.5 * (lo + hi);
                                if (bend(mid) > 0.0) == sign {
                                    lo = mid;
                                } else {
                                    hi = mid;
                                }
                            }
                            at_fold = 0.5 * (lo + hi);
                        }
                    }
                    let x = at(d.piece, at_fold);
                    if d.pts[0].dist(x) > 1e-6 * size && d.last().dist(x) > 1e-6 * size {
                        let k = node(x, &mut nodes);
                        if !cuts[i].iter().any(|c| c.1 == k) {
                            cuts[i].push((at_fold, k));
                        }
                    }
                }
                last = Some((t, w));
            }
        }
        // A node on a drawing away from its ends cuts it too: a vertex on another curve's orbit,
        // as where two circles about one line are drawn along each other.
        for (i, d) in drawings.iter().enumerate() {
            for (k, &p) in nodes.iter().enumerate() {
                if d.pts[0].dist(p) <= near || d.last().dist(p) <= near {
                    continue;
                }
                if off(&d.pts, p) > 1e-3 * size || cuts[i].iter().any(|c| c.1 == k) {
                    continue;
                }
                // The nearest sample, then the nearest point of the drawing by golden section.
                let m = (0..d.pts.len())
                    .min_by(|&x, &y| d.pts[x].dist(p).total_cmp(&d.pts[y].dist(p)))
                    .expect("a drawing has samples");
                let (mut lo, mut hi) = (
                    d.ts[m.saturating_sub(1)],
                    d.ts[(m + 1).min(d.pts.len() - 1)],
                );
                for _ in 0..100 {
                    let (x, y) = (lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo));
                    if at(d.piece, x).dist(p) < at(d.piece, y).dist(p) {
                        hi = y;
                    } else {
                        lo = x;
                    }
                }
                let t = 0.5 * (lo + hi);
                if at(d.piece, t).dist(p) <= near {
                    cuts[i].push((t, k));
                }
            }
        }
        let mut links: Vec<Link> = Vec::new();
        let mut gens: Vec<Vec<Gen>> = Vec::new();
        let mut fracs: Vec<Vec<f64>> = Vec::new();
        for (i, d) in drawings.iter().enumerate() {
            let pg = &pieces[d.piece];
            let forward = pg.range.1 >= pg.range.0;
            let mut stops = cuts[i].clone();
            stops.sort_by(|x, y| {
                if forward {
                    x.0.total_cmp(&y.0)
                } else {
                    y.0.total_cmp(&x.0)
                }
            });
            let mut all = vec![(pg.range.0, of_vertex[pg.v0])];
            all.extend(stops);
            all.push((pg.range.1, of_vertex[pg.v1]));
            for w in all.windows(2) {
                let ((t0, a), (t1, b)) = (w[0], w[1]);
                // The drawing's own samples between the two ends, and a few even ones.
                let mut at_t: Vec<f64> = (1..8).map(|k| t0 + (t1 - t0) * k as f64 / 8.0).collect();
                let within = |t: f64| (t - t0) * (t - t1) < 0.0;
                at_t.extend(d.ts.iter().copied().filter(|&t| within(t)));
                at_t.sort_by(|x, y| ((x - t0).abs()).total_cmp(&(y - t0).abs()));
                at_t.dedup();
                let mut pts = vec![nodes[a]];
                pts.extend(at_t.iter().map(|&t| at(d.piece, t)));
                pts.push(nodes[b]);
                let mut frac = vec![0.0];
                frac.extend(at_t.iter().map(|&t| (t - t0) / (t1 - t0)));
                frac.push(1.0);
                // A stretch drawn along a link already made, end to end, is that link again.
                // The polylines agree to their sag; then the stretch's own points at a few places
                // lie on the link's curve as it is drawn, to rounding.
                let on_drawing = |g: &Gen, q: Vec2| {
                    let n = 32;
                    let t_at = |i: usize| g.t.0 + (g.t.1 - g.t.0) * i as f64 / n as f64;
                    let m = (0..=n)
                        .min_by(|&x, &y| {
                            at(g.piece, t_at(x))
                                .dist(q)
                                .total_cmp(&at(g.piece, t_at(y)).dist(q))
                        })
                        .expect("samples");
                    let (mut lo, mut hi) = (t_at(m.saturating_sub(1)), t_at((m + 1).min(n)));
                    for _ in 0..80 {
                        let (x, y) = (lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo));
                        if at(g.piece, x).dist(q) < at(g.piece, y).dist(q) {
                            hi = y;
                        } else {
                            lo = x;
                        }
                    }
                    at(g.piece, 0.5 * (lo + hi)).dist(q) <= 1e-8 * size
                };
                let twin = links.iter().zip(&gens).position(|(l, list)| {
                    ((l.a == a && l.b == b) || (l.a == b && l.b == a))
                        && pts.iter().all(|&q| off(&l.pts, q) <= 1e-4 * size)
                        && l.pts.iter().all(|&q| off(&pts, q) <= 1e-4 * size)
                        && [0.2, 0.5, 0.8]
                            .iter()
                            .all(|&f| on_drawing(&list[0], at(d.piece, t0 + (t1 - t0) * f)))
                });
                match twin {
                    Some(k) => {
                        let along = links[k].a == a
                            && (a != b || {
                                let m = pts.len() / 4;
                                off(&links[k].pts[..links[k].pts.len() / 2], pts[m]) <= 1e-4 * size
                            });
                        gens[k].push(Gen {
                            piece: d.piece,
                            t: if along { (t0, t1) } else { (t1, t0) },
                        });
                    }
                    None => {
                        fracs.push(frac);
                        links.push(Link { a, b, pts });
                        gens.push(vec![Gen {
                            piece: d.piece,
                            t: (t0, t1),
                        }]);
                    }
                }
            }
        }
        let regions = graph::regions(&nodes, &links, 1e-3 * size)?;
        let domains = cut
            .parts
            .iter()
            .map(|p| Domain::new(p.surface.clone(), p.reversed, p.loops.clone()))
            .collect();
        Ok(Cells {
            axis,
            nodes,
            links,
            gens,
            fracs,
            grazes,
            first_meet: cut.pieces.len(),
            whole,
            pieces: pieces.clone(),
            meets,
            regions,
            domains,
            size,
        })
    }

    /// The point of piece `piece` of the cut body at `t`: a characteristic curve's on its face
    /// exactly, where the fitted curve is a hair off.
    pub(super) fn point(&self, cut: &Cut, piece: usize, t: f64) -> Point3 {
        let _ = cut;
        exact(&self.pieces, &self.axis, &self.grazes, piece, t)
    }

    /// A point well inside cell `r`: of the points a step to the left of its links' chords, the
    /// one farthest from every link.
    pub(super) fn middle(&self, r: usize) -> Option<Vec2> {
        let region = &self.regions[r];
        let mut best: Option<(f64, Vec2)> = None;
        for lp in &region.loops {
            for &(k, fwd) in lp {
                for w in self.links[k].pts.windows(2) {
                    let (a, b) = if fwd { (w[0], w[1]) } else { (w[1], w[0]) };
                    let d = b - a;
                    for f in [3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.001] {
                        let p = a.lerp(b, 0.5) + d.perp() * f;
                        if !graph::holds(&self.links, region, p) {
                            continue;
                        }
                        let clear = region
                            .loops
                            .iter()
                            .flatten()
                            .map(|&(k, _)| off(&self.links[k].pts, p))
                            .fold(f64::INFINITY, f64::min);
                        if best.is_none_or(|x| clear > x.0) {
                            best = Some((clear, p));
                        }
                    }
                }
            }
        }
        best.map(|x| x.1)
    }

    /// Where the orbit drawn at `o` passes through the body's surface, in order of angle from
    /// −π, with `parts` the cut body's.
    pub(super) fn hits(&self, cut: &Cut, o: Vec2) -> Vec<Hit> {
        let orbit = self.axis.orbit(o);
        let mut out = Vec::new();
        let mut roots: Vec<Option<Vec<f64>>> = vec![None; self.domains.len()];
        for (part, dom) in self.domains.iter().enumerate() {
            // One face's parts lie on one surface, crossed once.
            let face = cut.parts[part].face;
            let found = roots[face]
                .get_or_insert_with(|| crossings(&orbit, o.x, &dom.surface))
                .clone();
            for phase in found {
                let p = orbit.eval(phase);
                if let Where::Inside(uv) = dom.locate(p) {
                    let n = dom.surface.normal(uv.x, uv.y);
                    let out_n = if cut.parts[part].reversed { -n } else { n };
                    out.push(Hit {
                        phase,
                        part,
                        uv,
                        enters: self.axis.velocity(p).dot(out_n) < 0.0,
                    });
                }
            }
        }
        out.sort_by(|a, b| a.phase.total_cmp(&b.phase));
        out
    }
}

/// The angles at which `orbit` passes through `surface`, in order. A touching, where the orbit
/// comes to the surface without passing through, is none.
///
/// On a plane, a cylinder, a cone, a sphere or a torus the surface is where a polynomial in the
/// point vanishes which, round a circle, is a constant, a sine and cosine of the angle and a sine
/// and cosine of twice it, nothing more: eight values round the orbit give those five numbers
/// exactly, and its changes of sign are the real roots of a quartic in the tangent of the half
/// angle. On any other surface the distance from the orbit's point to the surface changes no
/// faster than the point moves, so a stretch of orbit whose middle is farther from the surface
/// than half the stretch is long holds no crossing, and the rest is halved until it does or is
/// a hair long. Neither way is a pair of crossings missed for lying between two samples.
pub(super) fn crossings(orbit: &Curve, radius: f64, surface: &Surface) -> Vec<f64> {
    let Some(gap) = polynomial(surface) else {
        return bounded(orbit, radius, surface);
    };
    let at = |t: f64| gap(orbit.eval(t));
    let n = 8;
    let (mut a0, mut a1, mut b1, mut a2, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for k in 0..n {
        let t = 2.0 * PI * f64::from(k) / f64::from(n);
        let v = at(t);
        a0 += v / f64::from(n);
        a1 += 2.0 * v * t.cos() / f64::from(n);
        b1 += 2.0 * v * t.sin() / f64::from(n);
        a2 += 2.0 * v * (2.0 * t).cos() / f64::from(n);
        b2 += 2.0 * v * (2.0 * t).sin() / f64::from(n);
    }
    let series =
        |t: f64| a0 + a1 * t.cos() + b1 * t.sin() + a2 * (2.0 * t).cos() + b2 * (2.0 * t).sin();
    // The half angle's tangent runs off to infinity half a turn from where it is read from: that
    // place is put where the series is farthest from nothing.
    let from = (0..8)
        .map(|k| PI * f64::from(k) / 4.0)
        .max_by(|x, y| series(x + PI).abs().total_cmp(&series(y + PI).abs()))
        .expect("eight places");
    let (s1, c1) = from.sin_cos();
    let (s2, c2) = (2.0 * from).sin_cos();
    let (ra1, rb1) = (a1 * c1 + b1 * s1, b1 * c1 - a1 * s1);
    let (ra2, rb2) = (a2 * c2 + b2 * s2, b2 * c2 - a2 * s2);
    let quartic = [
        a0 + ra1 + ra2,
        2.0 * rb1 + 4.0 * rb2,
        2.0 * a0 - 6.0 * ra2,
        2.0 * rb1 - 4.0 * rb2,
        a0 - ra1 + ra2,
    ];
    let mut out: Vec<f64> = real_roots(&quartic)
        .into_iter()
        .map(|w| {
            let t = from + 2.0 * w.atan();
            settled(
                orbit,
                surface,
                t - 2.0 * PI * ((t + PI) / (2.0 * PI)).floor(),
            )
        })
        .collect();
    out.sort_by(f64::total_cmp);
    out
}

/// The crossing at `t` settled on the surface itself: the quartic places it to its own rounding,
/// which the polynomial's size and a shallow crossing both magnify, and the distance to the
/// surface, bisected between the nearest angles either side that it differs in sign at, to the
/// distance's.
fn settled(orbit: &Curve, surface: &Surface, t: f64) -> f64 {
    let gap = |t: f64| csi::signed_distance(surface, orbit.eval(t));
    let mut reach = 1e-13;
    while reach < 1e-5 {
        let (mut lo, mut hi) = (t - reach, t + reach);
        let below = gap(lo) > 0.0;
        if below != (gap(hi) > 0.0) {
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if (gap(mid) > 0.0) == below {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return 0.5 * (lo + hi);
        }
        reach *= 4.0;
    }
    t
}

/// A polynomial in a point that vanishes on `surface`, of degree two, or four on a torus as the
/// square of one of degree two less a multiple of another.
fn polynomial(surface: &Surface) -> Option<Box<dyn Fn(Point3) -> f64 + '_>> {
    Some(match surface {
        Surface::Plane(s) => Box::new(|p| s.frame.to_local(p).z),
        Surface::Cylinder(s) => Box::new(|p| {
            let l = s.frame.to_local(p);
            l.x * l.x + l.y * l.y - s.radius * s.radius
        }),
        Surface::Sphere(s) => Box::new(|p| {
            let l = s.frame.to_local(p);
            l.dot(l) - s.radius * s.radius
        }),
        Surface::Cone(s) => Box::new(|p| {
            let l = s.frame.to_local(p);
            let (sa, ca) = s.half_angle.sin_cos();
            (l.x * l.x + l.y * l.y) * ca * ca - (s.radius * ca + l.z * sa).powi(2)
        }),
        Surface::Torus(s) => Box::new(|p| {
            let l = s.frame.to_local(p);
            let k = l.dot(l) + s.major * s.major - s.minor * s.minor;
            k * k - 4.0 * s.major * s.major * (l.x * l.x + l.y * l.y)
        }),
        _ => return None,
    })
}

/// The real roots of the polynomial with coefficients `c`, constant first, at which it changes
/// sign: between each two neighbouring roots of its derivative it runs one way, so it changes
/// sign there at most once, found by bisection.
fn real_roots(c: &[f64]) -> Vec<f64> {
    let scale = c.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut c = c.to_vec();
    while c.last().is_some_and(|x| x.abs() <= 1e-14 * scale) {
        c.pop();
    }
    if c.len() < 2 {
        return Vec::new();
    }
    let value = |x: f64| c.iter().rev().fold(0.0, |acc, k| acc * x + k);
    let lead = c[c.len() - 1];
    if c.len() == 2 {
        return vec![-c[0] / lead];
    }
    let slope: Vec<f64> = c
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, x)| k as f64 * x)
        .collect();
    // Every root lies within Cauchy's bound.
    let bound = 1.0
        + c[..c.len() - 1]
            .iter()
            .fold(0.0f64, |m, x| m.max((x / lead).abs()));
    let mut stops = vec![-bound];
    stops.extend(real_roots(&slope).into_iter().filter(|x| x.abs() < bound));
    stops.push(bound);
    stops.sort_by(f64::total_cmp);
    let mut out = Vec::new();
    for w in stops.windows(2) {
        let (mut lo, mut hi) = (w[0], w[1]);
        let (flo, fhi) = (value(lo), value(hi));
        if flo == 0.0 || fhi == 0.0 || (flo > 0.0) == (fhi > 0.0) {
            continue;
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            if (value(mid) > 0.0) == (flo > 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        out.push(0.5 * (lo + hi));
    }
    out
}

/// [`crossings`] by the bound on how fast the distance to the surface can change.
fn bounded(orbit: &Curve, radius: f64, surface: &Surface) -> Vec<f64> {
    let gap = |t: f64| csi::signed_distance(surface, orbit.eval(t));
    let mut out = Vec::new();
    let mut stack: Vec<(f64, f64)> = (0..8)
        .rev()
        .map(|k| {
            (
                -PI + PI * f64::from(k) / 4.0,
                -PI + PI * f64::from(k + 1) / 4.0,
            )
        })
        .collect();
    while let Some((a, b)) = stack.pop() {
        let mid = 0.5 * (a + b);
        if gap(mid).abs() > radius * 0.5 * (b - a) {
            continue;
        }
        if b - a <= 1e-9 {
            if (gap(a) > 0.0) != (gap(b) > 0.0) {
                out.push(mid);
            }
            continue;
        }
        stack.push((mid, b));
        stack.push((a, mid));
    }
    out
}

/// The point of piece `piece` at `t`. A characteristic curve is fitted to within a hundredth of
/// the tolerance, and an orbit through a point that far off the face's true outline no longer
/// touches the face: beside a fold of the outline, where three of the orbit's meetings with the
/// face close up, they are a thousandth apart or missing. So a characteristic curve's point is
/// moved onto the outline itself, where the velocity lies in the face, by Newton's method across
/// the outline in the face's parameters.
fn exact(
    pieces: &[PieceGeom],
    axis: &Axis,
    grazes: &[Option<Surface>],
    piece: usize,
    t: f64,
) -> Point3 {
    let c = pieces[piece]
        .curve
        .as_ref()
        .expect("a drawn piece has a curve");
    let p = c.eval(t);
    let Some(s) = &grazes[piece] else {
        return p;
    };
    let rate = |uv: Vec2| s.normal(uv.x, uv.y).dot(axis.velocity(s.eval(uv.x, uv.y)));
    let mut uv = s.project(p);
    for _ in 0..4 {
        let h = 1e-6;
        let r = rate(uv);
        let grad = Vec2::new(
            (rate(uv + Vec2::new(h, 0.0)) - rate(uv - Vec2::new(h, 0.0))) / (2.0 * h),
            (rate(uv + Vec2::new(0.0, h)) - rate(uv - Vec2::new(0.0, h))) / (2.0 * h),
        );
        let g2 = grad.dot(grad);
        if g2 <= 0.0 {
            return p;
        }
        uv = uv - grad * (r / g2);
    }
    let q = s.eval(uv.x, uv.y);
    // The curve is on the outline already to the fit; a step farther is no correction.
    if q.dist(p) <= 1e-6 * (1.0 + p.norm()) {
        q
    } else {
        p
    }
}

/// Where two drawings cross near where their chords do, and the two pieces' parameters there:
/// Newton's method on the two parameters. Chords cross where the drawings only touch; there the
/// steps run to an end, or nowhere.
fn crossing(
    at: &dyn Fn(usize, f64) -> Vec2,
    d: [&Drawing; 2],
    seed: [f64; 2],
    size: f64,
) -> Option<(Vec2, [f64; 2])> {
    let range = |k: usize| (d[k].ts[0], d[k].ts[d[k].ts.len() - 1]);
    let inside = |k: usize, t: f64| {
        let (a, b) = range(k);
        t >= a.min(b) && t <= a.max(b)
    };
    let mut t = seed;
    for _ in 0..50 {
        let (a, b) = (at(d[0].piece, t[0]), at(d[1].piece, t[1]));
        let gap = a - b;
        if gap.norm() <= 1e-12 * size {
            return (inside(0, t[0]) && inside(1, t[1])).then_some((a, t));
        }
        let rate = |k: usize| {
            let h = 1e-6 * (range(k).1 - range(k).0);
            (at(d[k].piece, t[k] + h) - at(d[k].piece, t[k] - h)) / (2.0 * h)
        };
        let (da, db) = (rate(0), rate(1));
        let det = db.cross(da);
        if det.abs() <= 1e-12 * da.norm() * db.norm() {
            return None;
        }
        t[0] += gap.cross(db) / det;
        t[1] += gap.cross(da) / det;
        if !t[0].is_finite() || !t[1].is_finite() {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Torus;

    // An orbit that skims the top of a ring a ten-thousandth below its crest goes in and out of
    // it twice, each pair a hundredth of a radian apart. Both ways of finding the crossings give
    // all four.
    #[test]
    fn an_orbit_skimming_a_ring_crosses_it_four_times() {
        let ring = Surface::Torus(Torus {
            frame: Frame::WORLD.translated(Vec3::new(10.0, -2.0, 0.0)),
            major: 2.0,
            minor: 0.5,
        });
        let axis = Axis::new(Vec3::ZERO, Vec3::Z).unwrap();
        let o = Vec2::new(12.1736, 0.49993);
        let orbit = axis.orbit(o);
        let exact = crossings(&orbit, o.x, &ring);
        let slow = bounded(&orbit, o.x, &ring);
        assert_eq!(exact.len(), 4, "{exact:?}");
        assert_eq!(slow.len(), 4, "{slow:?}");
        for (a, b) in exact.iter().zip(&slow) {
            assert!((a - b).abs() < 1e-8, "{a} {b}");
            assert!(csi::signed_distance(&ring, orbit.eval(*a)).abs() < 1e-12);
        }
    }
}
