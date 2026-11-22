//! The body cut along its characteristic curves into the pieces of face the motion carries outward
//! and the rest.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use super::contour::{self, Branch, Closed, Run};
use super::grazing::{self, Arrival, Rate};
use crate::{
    Error, Result,
    boolean::{
        arrange, coedge_trim_with, csi,
        domain::{Domain, Trim, Where},
        face_domain,
        sew::PieceGeom,
        split_poles,
    },
    check::coedge_params,
    geom::{Curve, Line, Surface},
    math::{Point3, Vec2, Vec3},
    measure, tol,
    topo::{Brep, CoedgeId, EdgeId, FaceId, FaceName},
};

/// A piece of one of the body's faces, on one side of its characteristic curves.
pub(super) struct Part {
    pub surface: Surface,
    pub reversed: bool,
    pub names: BTreeSet<FaceName>,
    pub loops: Vec<Vec<Trim>>,
    /// Whether the motion carries it outward.
    pub front: bool,
    /// The place among the body's faces of the face it is cut from.
    pub face: usize,
}

/// The body cut along its characteristic curves: its vertices (the body's own first), its edge
/// pieces, and its face pieces, whose trims name the pieces.
pub(super) struct Cut {
    pub pool: Vec<Point3>,
    pub pieces: Vec<PieceGeom>,
    /// The range of the whole curve each piece was cut from: its edge's, or its own. The faces
    /// that pieces of one curve sweep lie on the one surface that curve sweeps.
    pub whole: Vec<(f64, f64)>,
    pub parts: Vec<Part>,
    /// The pieces of each edge, in order along it.
    pub edges: HashMap<EdgeId, Vec<usize>>,
    /// The pieces of characteristic curve across each face, by the face's place among the body's.
    pub curves: Vec<Vec<usize>>,
}

/// Samples along each coedge where the rate is looked for roots.
const SAMPLES: usize = 64;

/// Where a characteristic curve meets the boundary: a point on a coedge, at its edge's parameter. A
/// seam's two coedges are two places on the face's boundary, a period apart.
#[derive(Clone, Copy, Debug)]
struct Meet {
    coedge: CoedgeId,
    t: f64,
}

/// The rate's sign at a sample, or 0 where it is within `eps` of zero.
fn sign(g: f64, eps: f64) -> i8 {
    if g > eps {
        1
    } else if g < -eps {
        -1
    } else {
        0
    }
}

/// The roots of the rate along the loops of face `f`: where its sign changes along the boundary,
/// pinned by bisection, or a single sample where it is zero between opposite signs.
fn boundary_roots(b: &Brep, f: FaceId, rate: &Rate, eps: f64) -> Vec<Meet> {
    let mut out = Vec::new();
    for &lp in &b.face(f).loops {
        // The loop's samples in order: each coedge's, without its last, which is the next one's
        // first.
        // A pole is one point with one rate, whatever its row of parameters reads, and a normal
        // there taken from a parameter a hair along can come from past the pole, turned over: the
        // rate is not read at poles.
        let poles: Vec<_> = b
            .lp(lp)
            .coedges
            .iter()
            .map(|&c| b.edge(b.coedge(c).edge))
            .filter(|e| e.curve.is_none())
            .map(|e| e.start)
            .collect();
        let mut samples: Vec<(CoedgeId, f64, Vec2, f64)> = Vec::new();
        for &c in &b.lp(lp).coedges {
            let e = b.edge(b.coedge(c).edge);
            if e.curve.is_none() {
                continue;
            }
            let pieces = e
                .curve
                .as_ref()
                .map_or(0, |cv| cv.breakpoints(e.range.0, e.range.1).len());
            let n = SAMPLES * (pieces + 1);
            let ts = coedge_params(b, c, n);
            let uvs = b.coedge_uvs(c, &ts);
            let from_pole = poles.contains(&b.coedge_start(c));
            for k in usize::from(from_pole)..n {
                samples.push((c, ts[k], uvs[k], rate.at(uvs[k])));
            }
        }
        let n = samples.len();
        let signs: Vec<i8> = samples.iter().map(|s| sign(s.3, eps)).collect();
        if signs.iter().all(|&s| s == 0) {
            continue;
        }
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, z) = (&samples[i], &samples[j]);
            // The stretch from one sample to the next lies on the first one's coedge, which ends
            // where the next coedge starts.
            if signs[i] != 0 && signs[j] != 0 && signs[i] != signs[j] {
                let to = if a.0 == z.0 {
                    (z.1, z.2)
                } else {
                    let end = end_param(b, a.0);
                    (end, b.coedge_uvs(a.0, &[a.1, end])[1])
                };
                out.push(Meet {
                    coedge: a.0,
                    t: bisect_on(b, a.0, rate, (a.1, a.2, a.3), to),
                });
            }
            // A lone zero between opposite signs.
            if signs[i] == 0 {
                let before = signs[(i + n - 1) % n];
                let after = signs[j];
                if before != 0 && after != 0 && before != after {
                    out.push(Meet {
                        coedge: samples[i].0,
                        t: samples[i].1,
                    });
                }
            }
        }
    }
    out
}

/// The parameter a coedge starts at.
fn start_param(b: &Brep, c: CoedgeId) -> f64 {
    let e = b.edge(b.coedge(c).edge);
    if b.coedge(c).reversed {
        e.range.1
    } else {
        e.range.0
    }
}

/// The parameter a coedge ends at.
fn end_param(b: &Brep, c: CoedgeId) -> f64 {
    let e = b.edge(b.coedge(c).edge);
    if b.coedge(c).reversed {
        e.range.0
    } else {
        e.range.1
    }
}

/// The root of the rate along coedge `c` between `a` (parameter, uv, rate) and `z` (parameter, uv),
/// by bisection on the edge parameter.
fn bisect_on(b: &Brep, c: CoedgeId, rate: &Rate, a: (f64, Vec2, f64), z: (f64, Vec2)) -> f64 {
    let e = b.edge(b.coedge(c).edge);
    let Some(curve) = &e.curve else {
        return a.0;
    };
    let (mut lo, mut hi) = (a.0, z.0);
    let mut hint = a.1;
    let s0 = a.2.signum();
    for _ in 0..80 {
        let m = 0.5 * (lo + hi);
        if m == lo || m == hi {
            break;
        }
        let uv = rate.surface.project_near(curve.eval(m), hint);
        hint = uv;
        if rate.at(uv).signum() == s0 {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// The parameters where `edge` touches the line through `o` along the unit `axis`: the least
/// distances to it, bracketed by samples and pinned by golden-section search, that are within the
/// tolerance, its ends left out.
fn on_line(b: &Brep, edge: EdgeId, o: Point3, axis: Vec3) -> Vec<f64> {
    let e = b.edge(edge);
    let Some(c) = &e.curve else {
        return Vec::new();
    };
    let off = |t: f64| {
        let q = c.eval(t) - o;
        (q - axis * q.dot(axis)).norm()
    };
    let n = 256;
    let (t0, t1) = e.range;
    let at = |k: usize| t0 + (t1 - t0) * k as f64 / n as f64;
    let mut out = Vec::new();
    for k in 1..n {
        let (a, m, z) = (off(at(k - 1)), off(at(k)), off(at(k + 1)));
        if !(m <= a && m <= z) {
            continue;
        }
        let (mut lo, mut hi) = (at(k - 1), at(k + 1));
        for _ in 0..100 {
            let (x, y) = (lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo));
            if off(x) < off(y) {
                hi = y;
            } else {
                lo = x;
            }
        }
        let t = 0.5 * (lo + hi);
        if off(t) <= tol::LINEAR && out.last().is_none_or(|&s: &f64| (t - s).abs() > 1e-9) {
            out.push(t);
        }
    }
    out
}

/// Cuts `b` along the curves where `field`, the velocity of its points, lies in its faces. Where
/// the motion holds the line `held` still, the edges are cut where they touch it, so each point
/// the motion holds still on an edge is a vertex.
pub(super) fn cut(
    b: &Brep,
    field: &dyn Fn(Point3) -> Vec3,
    held: Option<(Point3, Vec3)>,
) -> Result<Cut> {
    let fields = vec![field; b.face_ids().count()];
    cut_by(b, &fields, held)
}

/// [`cut`] with each face's own field, in the order of the body's faces.
pub(super) fn cut_by(
    b: &Brep,
    fields: &[&dyn Fn(Point3) -> Vec3],
    held: Option<(Point3, Vec3)>,
) -> Result<Cut> {
    let size = measure::bbox(b).diagonal();
    let faces: Vec<FaceId> = b.face_ids().collect();
    let epss: Vec<f64> = fields
        .iter()
        .map(|field| {
            let speed = b
                .vertex_ids()
                .map(|v| field(b.vertex(v).point).norm())
                .fold(0.0, f64::max)
                .max(field(measure::bbox(b).center()).norm());
            tol::ANGULAR * speed.max(f64::MIN_POSITIVE)
        })
        .collect();
    let doms: Vec<Domain> = faces.iter().map(|&f| face_domain(b, f)).collect();
    let rates: Vec<Rate> = faces
        .iter()
        .zip(fields)
        .map(|(&f, &field)| Rate {
            surface: &b.face(f).surface,
            out: if b.face(f).reversed { -1.0 } else { 1.0 },
            field,
            size,
        })
        .collect();
    let still: Vec<bool> = doms
        .iter()
        .zip(&rates)
        .zip(&epss)
        .map(|((d, r), &eps)| {
            let mut pts = d.interior_points();
            pts.extend(d.samples().iter().flatten().flatten().copied());
            pts.iter().all(|&uv| r.at(uv).abs() <= eps)
        })
        .collect();
    // Every face's roots on its boundary, as places on edges.
    let mut meets: Vec<Vec<Meet>> = Vec::with_capacity(faces.len());
    for (i, &f) in faces.iter().enumerate() {
        meets.push(if still[i] {
            Vec::new()
        } else {
            boundary_roots(b, f, &rates[i], epss[i])
        });
    }
    let mut places = Places {
        pool: b.vertex_ids().map(|v| b.vertex(v).point).collect(),
        splits: HashMap::new(),
    };
    if let Some((o, axis)) = held {
        for e in b.edge_ids() {
            for t in on_line(b, e, o, axis) {
                places.on_edge(b, e, t);
            }
        }
    }
    // Where the held line pierces a face, the velocity vanishes and turns over: a curve through
    // there grazes on both sides of it, and its two halves sweep apart from a pole.
    let pierces: Vec<Vec<Point3>> = match held {
        Some((o, axis)) => {
            let c = (measure::bbox(b).center() - o).dot(axis);
            let line = Curve::Line(Line {
                origin: o,
                dir: axis,
            });
            doms.iter()
                .enumerate()
                .map(|(i, d)| {
                    if still[i] {
                        return Vec::new();
                    }
                    let hits = csi::curve_surface(&line, (c - size, c + size), &d.surface);
                    hits.ts
                        .iter()
                        .map(|&t| line.eval(t))
                        .filter(|&p| matches!(d.locate(p), Where::Inside(_)))
                        .collect()
                })
                .collect()
        }
        None => vec![Vec::new(); faces.len()],
    };
    let mut traced: Vec<Vec<CutCurve>> = Vec::with_capacity(faces.len());
    for (i, &f) in faces.iter().enumerate() {
        // Each place a curve meets the boundary: a vertex where the boundary passes it, so one
        // on a seam is two, a period apart.
        let mut roots: Vec<Root> = Vec::new();
        for m in &meets[i] {
            let (vertex, point) = places.on_edge(b, b.coedge(m.coedge).edge, m.t);
            let uv = b.coedge_uvs(m.coedge, &[start_param(b, m.coedge), m.t])[1];
            let seen = roots
                .iter()
                .any(|r| r.vertex == vertex && (r.uv - uv).norm() <= 1e-9 * (1.0 + uv.norm()));
            if !seen {
                roots.push(Root { vertex, point, uv });
            }
        }
        let curves = if still[i] {
            Vec::new()
        } else if let Some(form) = contour::form(&b.face(f).surface) {
            let closed = Closed {
                rate: &rates[i],
                form,
                lo: doms[i].lo,
                hi: doms[i].hi,
            };
            let marks = (&roots[..], &pierces[i][..]);
            closed_cuts(b, f, &closed, &doms[i], marks, &mut places, epss[i])?
        } else {
            traced_cuts(b, f, &rates[i], &doms[i], &roots, &mut places)?
        };
        traced.push(curves);
    }
    let Places { pool, mut splits } = places;
    // The edge pieces between splits, each edge's in order along it.
    let mut pieces: Vec<PieceGeom> = Vec::new();
    let mut whole: Vec<(f64, f64)> = Vec::new();
    let mut of_edge: HashMap<EdgeId, Vec<(usize, f64, f64)>> = HashMap::new();
    for e in b.edge_ids() {
        let edge = b.edge(e);
        let mut cuts = splits.remove(&e).unwrap_or_default();
        cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
        let mut stops = vec![(edge.range.0, edge.start.index())];
        stops.extend(cuts);
        stops.push((edge.range.1, edge.end.index()));
        let curve = edge.curve.clone().map(Rc::new);
        let list = of_edge.entry(e).or_default();
        for w in stops.windows(2) {
            list.push((pieces.len(), w[0].0, w[1].0));
            whole.push(edge.range);
            pieces.push(PieceGeom {
                curve: curve.clone(),
                range: (w[0].0, w[1].0),
                v0: w[0].1,
                v1: w[1].1,
                tol: tol::LINEAR,
            });
        }
    }
    let mut parts = Vec::new();
    let mut curves = Vec::with_capacity(faces.len());
    for (i, &f) in faces.iter().enumerate() {
        let face = b.face(f);
        let mut boundary: Vec<Vec<Trim>> = face
            .loops
            .iter()
            .map(|&lp| {
                let mut trims = Vec::new();
                for &c in &b.lp(lp).coedges {
                    let co = b.coedge(c);
                    let mut list = of_edge[&co.edge].clone();
                    if co.reversed {
                        list.reverse();
                    }
                    let mut ts: Vec<f64> = Vec::with_capacity(list.len() + 1);
                    for &(_, t0, t1) in &list {
                        ts.push(if co.reversed { t1 } else { t0 });
                    }
                    let last = list.last().expect("an edge has a piece");
                    ts.push(if co.reversed { last.1 } else { last.2 });
                    let uvs = b.coedge_uvs(c, &ts);
                    for (k, &(pi, _, _)) in list.iter().enumerate() {
                        let pg = &pieces[pi];
                        let (v0, v1) = if co.reversed {
                            (pg.v1, pg.v0)
                        } else {
                            (pg.v0, pg.v1)
                        };
                        trims.push(coedge_trim_with(
                            b,
                            c,
                            pi,
                            pg.curve.clone(),
                            (ts[k], ts[k + 1]),
                            (uvs[k], uvs[k + 1]),
                            (v0, v1),
                        ));
                    }
                }
                trims
            })
            .collect();
        let mut cuts = Vec::new();
        curves.push((pieces.len()..pieces.len() + traced[i].len()).collect());
        for tc in &traced[i] {
            let curve = Rc::new(tc.curve.clone());
            whole.push(tc.range);
            pieces.push(PieceGeom {
                curve: Some(curve.clone()),
                range: tc.range,
                v0: tc.ends.0,
                v1: tc.ends.1,
                tol: tol::LINEAR,
            });
            cuts.push(Trim {
                piece: pieces.len() - 1,
                forward: true,
                curve: Some(curve),
                pole: pool[tc.ends.0],
                t0: tc.range.0,
                t1: tc.range.1,
                map: crate::boolean::domain::UvMap::Projected,
                uv0: tc.uvs.0,
                uv1: tc.uvs.1,
                v0: tc.ends.0,
                v1: tc.ends.1,
            });
        }
        // A curve that ends at a pole meets the pole's row of parameters at its own `u`.
        let made = split_poles(&mut boundary, &cuts, pieces.len());
        whole.extend(made.iter().map(|pg| pg.range));
        pieces.extend(made);
        let regions = arrange::split(
            &face.surface,
            face.reversed,
            boundary,
            cuts,
            &|_| tol::LINEAR,
            &|_| false,
            &|_| false,
        )?;
        for loops in regions {
            let front = if still[i] {
                false
            } else {
                let dom = Domain::new(face.surface.clone(), face.reversed, loops.clone());
                let uv = dom.interior_point().ok_or_else(|| {
                    Error::Numerical(
                        "a piece of a face cut along its grazing curve is empty".into(),
                    )
                })?;
                rates[i].at(uv) > epss[i]
            };
            parts.push(Part {
                surface: face.surface.clone(),
                reversed: face.reversed,
                names: face.names.clone(),
                loops,
                front,
                face: i,
            });
        }
    }
    let edges = of_edge
        .into_iter()
        .map(|(e, list)| (e, list.into_iter().map(|x| x.0).collect()))
        .collect();
    Ok(Cut {
        pool,
        pieces,
        whole,
        parts,
        edges,
        curves,
    })
}

/// The cut's vertices so far, the body's own first, and where they split the body's edges.
struct Places {
    pool: Vec<Point3>,
    splits: HashMap<EdgeId, Vec<(f64, usize)>>,
}

impl Places {
    /// The point at `t` on `edge` as a vertex: an end of the edge where it lies on one, an
    /// earlier split where it lies on one, else a new vertex splitting the edge there. On a pole
    /// it is the pole's vertex.
    fn on_edge(&mut self, b: &Brep, edge: EdgeId, t: f64) -> (usize, Point3) {
        let e = b.edge(edge);
        let Some(curve) = e.curve.as_ref() else {
            return (e.start.index(), b.vertex(e.start).point);
        };
        let p = curve.eval(t);
        for v in [e.start, e.end] {
            let q = b.vertex(v).point;
            if q.dist(p) <= 10.0 * tol::LINEAR {
                return (v.index(), q);
            }
        }
        let list = self.splits.entry(edge).or_default();
        for &(_, vi) in list.iter() {
            if self.pool[vi].dist(p) <= 10.0 * tol::LINEAR {
                return (vi, self.pool[vi]);
            }
        }
        self.pool.push(p);
        list.push((t, self.pool.len() - 1));
        (self.pool.len() - 1, p)
    }

    /// The vertex where a characteristic curve leaves face `f` at `p` with no root found there:
    /// a pole, or the point of the nearest edge, which the rate vanishes all along.
    fn on_boundary(&mut self, b: &Brep, f: FaceId, p: Point3) -> Result<(usize, Point3)> {
        let reach = 1e-6 * (1.0 + p.norm());
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let e = b.edge(b.coedge(c).edge);
                if e.curve.is_none() && b.vertex(e.start).point.dist(p) <= reach {
                    return Ok((e.start.index(), b.vertex(e.start).point));
                }
            }
        }
        let (edge, t) = nearest_edge(b, f, p)?;
        Ok(self.on_edge(b, edge, t))
    }

    /// A new vertex inside a face.
    fn inside(&mut self, p: Point3) -> usize {
        self.pool.push(p);
        self.pool.len() - 1
    }
}

/// A place a characteristic curve meets a face's boundary: its vertex, and its parameters on the
/// face there, of which a vertex on a seam has two.
struct Root {
    vertex: usize,
    point: Point3,
    uv: Vec2,
}

/// A characteristic curve across a face, between two of the cut's vertices.
struct CutCurve {
    curve: Curve,
    range: (f64, f64),
    ends: (usize, usize),
    uvs: (Vec2, Vec2),
}

/// A face's characteristic curves where they have a closed form: each run of a branch inside the
/// face, cut where a line of the form crosses another branch or the held line pierces the face
/// (`marks`, with the roots on its boundary), fitted through its exact points.
fn closed_cuts(
    b: &Brep,
    f: FaceId,
    closed: &Closed,
    dom: &Domain,
    marks: (&[Root], &[Point3]),
    places: &mut Places,
    eps: f64,
) -> Result<Vec<CutCurve>> {
    let (roots, pierces) = marks;
    let size = closed.rate.size;
    let mut runs: Vec<Run> = Vec::new();
    for (branch, range) in closed.branches(eps) {
        let marks: Vec<f64> = roots
            .iter()
            .filter_map(|r| {
                let at = closed.param_of(branch, r.uv);
                let on = closed.point(branch, at)?.0;
                (on.dist(r.point) <= 1e-7 * size).then_some(at)
            })
            .collect();
        runs.extend(closed.runs(dom, branch, range, &marks));
    }
    // Each run's stops: its ends, and any crossing inside it, with the vertex there where known.
    let mut stops: Vec<Vec<(f64, Option<usize>)>> = runs
        .iter()
        .map(|r| vec![(r.s.0, None), (r.s.1, None)])
        .collect();
    // Where both coefficients vanish on a whole line, a branch crossing it may have no value on
    // it (a torus's two angles trade places there), so the crossing is its limit from the side it
    // runs on; a run that stops at the line ends at the crossing.
    let width = closed.hi.x - closed.lo.x;
    let near = 1e-7 * width;
    let mut crossings: Vec<(Point3, usize)> = Vec::new();
    for i in 0..runs.len() {
        let Branch::AtU { at } = runs[i].branch else {
            continue;
        };
        for j in 0..runs.len() {
            let (s0, s1) = runs[j].s;
            if matches!(runs[j].branch, Branch::AtU { .. } | Branch::Line { .. })
                || at < s0 - near
                || at > s1 + near
            {
                continue;
            }
            let side = (0.5 * (s0 + s1) - at).signum();
            let Some((p, uv)) = closed.crossing(runs[j].branch, at, side) else {
                continue;
            };
            let (v0, v1) = runs[i].s;
            let reach = 1e-6 * (v1 - v0);
            if uv.y < v0 - reach || uv.y > v1 + reach {
                continue;
            }
            let v = match crossings.iter().find(|c| c.0.dist(p) <= 1e-6 * size) {
                Some(c) => c.1,
                None => {
                    let v = places.inside(p);
                    crossings.push((p, v));
                    v
                }
            };
            for (k, s) in [(i, uv.y.clamp(v0, v1)), (j, at.clamp(s0, s1))] {
                match stops[k].iter_mut().find(|x| (x.0 - s).abs() <= near) {
                    Some(stop) => *stop = (stop.0, Some(v)),
                    None => stops[k].push((s, Some(v))),
                }
            }
        }
    }
    let surface = closed.rate.surface;
    for &q in pierces {
        let uv = surface.project(q);
        let mut vertex = None;
        for (run, stop) in runs.iter().zip(stops.iter_mut()) {
            let at = closed.param_of(run.branch, uv);
            let on = closed
                .point(run.branch, at)
                .map(|x| x.0.dist(q) <= 1e-7 * size);
            if at > run.s.0 + near && at < run.s.1 - near && on == Some(true) {
                let v = *vertex.get_or_insert_with(|| places.inside(q));
                stop.push((at, Some(v)));
            }
        }
    }
    let mut out = Vec::new();
    for (run, mut stop) in runs.into_iter().zip(stops) {
        stop.sort_by(|a, b| a.0.total_cmp(&b.0));
        // A pierce at a run's end or at a crossing is one stop with the vertex known there.
        stop.dedup_by(|later, kept| {
            let same = later.0 - kept.0 <= near;
            if same && kept.1.is_none() {
                kept.1 = later.1;
            }
            same
        });
        for w in stop.windows(2) {
            let mut ends = [(0usize, Point3::ZERO, Vec2::ZERO); 2];
            let mut span = [w[0].0, w[1].0];
            let reach = 0.05 * (w[1].0 - w[0].0);
            for (k, &(s, known)) in w.iter().enumerate() {
                // Read a hair inside the run: at a crossing the closed form may have no value.
                let inner = s.clamp(w[0].0 + 1e-3 * near, w[1].0 - 1e-3 * near);
                let Some((p, uv)) = closed.point(run.branch, inner) else {
                    return Err(Error::Numerical(
                        "a characteristic curve ends where its closed form has no point".into(),
                    ));
                };
                // A root this branch passes through near the end, which the face's winding
                // number places only to within the tolerance, is the end exactly.
                let snapped = roots
                    .iter()
                    .filter(|r| grazing::near_image(surface, r.uv, uv))
                    .filter_map(|r| {
                        let at = closed.param_of(run.branch, r.uv);
                        let on = closed.point(run.branch, at)?.0;
                        (on.dist(r.point) <= 1e-7 * size && (at - s).abs() <= reach)
                            .then_some((r, at))
                    })
                    .min_by(|a, b| (a.1 - s).abs().total_cmp(&(b.1 - s).abs()));
                ends[k] = match (known, snapped) {
                    (Some(v), _) => (v, places.pool[v], uv),
                    (None, Some((r, at))) => {
                        span[k] = at;
                        (r.vertex, r.point, r.uv)
                    }
                    (None, None) => {
                        let (v, q) = places.on_boundary(b, f, p)?;
                        (v, q, uv)
                    }
                };
            }
            let at = |s: f64| closed.point(run.branch, s).map_or(Point3::ZERO, |x| x.0);
            let (curve, range) =
                grazing::fit_exact(&at, (span[0], span[1]), (ends[0].1, ends[1].1), size)?;
            out.push(CutCurve {
                curve,
                range,
                ends: (ends[0].0, ends[1].0),
                uvs: (ends[0].2, ends[1].2),
            });
        }
    }
    // The rate changes sign along the boundary at each root, so a curve leaves the face there;
    // without it a piece of the face would be taken whole for one side.
    let ended = |v: usize| out.iter().any(|c| c.ends.0 == v || c.ends.1 == v);
    if !roots.iter().all(|r| ended(r.vertex)) {
        return Err(Error::Numerical(
            "no characteristic curve was found from a place the boundary crosses one".into(),
        ));
    }
    Ok(out)
}

/// A face's characteristic curves traced from root to root.
fn traced_cuts(
    b: &Brep,
    f: FaceId,
    rate: &Rate,
    dom: &Domain,
    roots: &[Root],
    places: &mut Places,
) -> Result<Vec<CutCurve>> {
    let ends: Vec<(Point3, Vec2)> = roots.iter().map(|r| (r.point, r.uv)).collect();
    let mut used = vec![false; roots.len()];
    let mut out = Vec::new();
    for s in 0..roots.len() {
        if used[s] {
            continue;
        }
        used[s] = true;
        let (path, arrival) = grazing::trace(rate, dom, &ends, s)?;
        let end = match arrival {
            Arrival::Root(k) => {
                if used[k] && k != s {
                    return Err(Error::Numerical(
                        "two characteristic curves end at one point".into(),
                    ));
                }
                used[k] = true;
                roots[k].vertex
            }
            Arrival::Boundary(p) => places.on_boundary(b, f, p)?.0,
        };
        let (curve, range) = grazing::fit(rate, &path)?;
        let uvs = (path.uvs[0], *path.uvs.last().expect("a trace has points"));
        out.push(CutCurve {
            curve,
            range,
            ends: (roots[s].vertex, end),
            uvs,
        });
    }
    Ok(out)
}

/// The edge of face `f` nearest `p`, and the parameter on it of the point nearest `p`.
fn nearest_edge(b: &Brep, f: FaceId, p: Point3) -> Result<(EdgeId, f64)> {
    let mut best: Option<(f64, EdgeId, f64)> = None;
    for &lp in &b.face(f).loops {
        for &c in &b.lp(lp).coedges {
            let e = b.coedge(c).edge;
            let edge = b.edge(e);
            let Some(curve) = &edge.curve else { continue };
            let t = curve.project(p, edge.range.0, edge.range.1);
            let d = curve.eval(t).dist(p);
            if best.is_none_or(|x| d < x.0) {
                best = Some((d, e, t));
            }
        }
    }
    match best {
        Some((d, e, t)) if d <= 1e-6 * (1.0 + p.norm()) => Ok((e, t)),
        _ => Err(Error::Numerical(
            "a characteristic curve leaves its face away from every edge".into(),
        )),
    }
}
