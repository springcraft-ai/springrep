//! A body drawn in the plane of the two lengths a turn keeps, its distance from the axis and its
//! height along it. Each point of the plane is an orbit, a circle about the axis. The body's edges
//! and characteristic curves, drawn there, cut the plane into cells; over a cell the orbits meet
//! the same pieces of the body's faces in the same order, so the body is a stack of intervals of
//! angle over each cell, and what it sweeps is each interval lengthened by the turn and merged
//! with the ones it reaches.

use std::f64::consts::PI;

use super::cut::Cut;
use super::moving::graph::{self, Link, Region};
use crate::{
    Result,
    boolean::{
        csi,
        domain::{Domain, Where},
    },
    geom::{Circle, Curve},
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
    pub(super) fn new(cut: &Cut, axis: Axis) -> Result<Cells> {
        let mut bounds = crate::math::Aabb::EMPTY;
        cut.pool.iter().for_each(|&p| bounds.add(p));
        let size = bounds.diagonal();
        let near = 1e-9 * size;
        let at = |piece: usize, t: f64| -> Vec2 {
            let c = cut.pieces[piece]
                .curve
                .as_ref()
                .expect("a drawn piece has a curve");
            axis.drawn(c.eval(t), 0.0).0
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
        let of_vertex: Vec<usize> = cut
            .pool
            .iter()
            .map(|&p| node(axis.drawn(p, 0.0).0, &mut nodes))
            .collect();
        // A piece that lies along an orbit is drawn as a point, and is no link.
        let drawings: Vec<Drawing> = cut
            .pieces
            .iter()
            .enumerate()
            .filter(|(_, pg)| pg.curve.is_some())
            .map(|(piece, pg)| {
                let ts: Vec<f64> = (0..=SAMPLES)
                    .map(|k| pg.range.0 + (pg.range.1 - pg.range.0) * k as f64 / SAMPLES as f64)
                    .collect();
                let pts = ts.iter().map(|&t| at(piece, t)).collect();
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
                for m in 0..SAMPLES {
                    let from = if i == j { m + 2 } else { 0 };
                    for n in from..SAMPLES {
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
                        let ends = [a.pts[0], a.pts[SAMPLES], b.pts[0], b.pts[SAMPLES]];
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
        // A node on a drawing away from its ends cuts it too: a vertex on another curve's orbit,
        // as where two circles about one line are drawn along each other.
        for (i, d) in drawings.iter().enumerate() {
            for (k, &p) in nodes.iter().enumerate() {
                if d.pts[0].dist(p) <= near || d.pts[SAMPLES].dist(p) <= near {
                    continue;
                }
                if off(&d.pts, p) > 1e-3 * size || cuts[i].iter().any(|c| c.1 == k) {
                    continue;
                }
                // The nearest sample, then the nearest point of the drawing by golden section.
                let m = (0..=SAMPLES)
                    .min_by(|&x, &y| d.pts[x].dist(p).total_cmp(&d.pts[y].dist(p)))
                    .expect("a drawing has samples");
                let (mut lo, mut hi) = (d.ts[m.saturating_sub(1)], d.ts[(m + 1).min(SAMPLES)]);
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
        for (i, d) in drawings.iter().enumerate() {
            let pg = &cut.pieces[d.piece];
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
                let n = SAMPLES / 4;
                let mut pts: Vec<Vec2> = (0..=n)
                    .map(|k| at(d.piece, t0 + (t1 - t0) * k as f64 / n as f64))
                    .collect();
                pts[0] = nodes[a];
                pts[n] = nodes[b];
                // A stretch drawn along a link already made, end to end, is that link again.
                let twin = links.iter().position(|l| {
                    ((l.a == a && l.b == b) || (l.a == b && l.b == a))
                        && pts.iter().all(|&q| off(&l.pts, q) <= 1e-6 * size)
                        && l.pts.iter().all(|&q| off(&pts, q) <= 1e-6 * size)
                });
                match twin {
                    Some(k) => {
                        let along = links[k].a == a
                            && (a != b || {
                                let m = pts.len() / 4;
                                pts[m].dist(links[k].pts[m]) <= 1e-6 * size
                            });
                        gens[k].push(Gen {
                            piece: d.piece,
                            t: if along { (t0, t1) } else { (t1, t0) },
                        });
                    }
                    None => {
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
            regions,
            domains,
            size,
        })
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
        for (part, dom) in self.domains.iter().enumerate() {
            let found = csi::curve_surface(&orbit, (-PI, PI), &dom.surface);
            for &phase in &found.ts {
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

/// Where two drawings cross near where their chords do, and the two pieces' parameters there:
/// Newton's method on the two parameters. Chords cross where the drawings only touch; there the
/// steps run to an end, or nowhere.
fn crossing(
    at: &dyn Fn(usize, f64) -> Vec2,
    d: [&Drawing; 2],
    seed: [f64; 2],
    size: f64,
) -> Option<(Vec2, [f64; 2])> {
    let range = |k: usize| (d[k].ts[0], d[k].ts[SAMPLES]);
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
