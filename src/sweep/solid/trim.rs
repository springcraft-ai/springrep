//! A sweep's faces trimmed to the swept solid (doc/design.md, "Sweeps of solids", "Folds"): where
//! the envelope folds, or the sweep runs into itself, parts of the faces built from the body's
//! curves lie inside the solid. Each face keeps what the moving body never reaches into, found in
//! the face's own parameter plane (Peternell, Pottmann, Steiner and Zhao 2005; Seong, Elber and Kim
//! 2006): the curves where that begins are found on a grid as a contour, each one where two faces
//! cross, shared by the two, and the faces are sewn once along them.

mod brent;
mod contour;
mod reach;
mod restrict;
mod section;
#[cfg(test)]
mod tests;

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

pub(crate) use reach::Reach;

#[cfg(test)]
thread_local! {
    /// Set by a test to search as the plain methods do: every sample of a path's depth, a peak
    /// by golden section, a sign change by regula falsi.
    static PLAIN_SEARCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn plain_search() -> bool {
    #[cfg(test)]
    return PLAIN_SEARCH.get();
    #[cfg(not(test))]
    false
}

use super::grazing::fit_exact;
use super::moving::graph::{self, Link};
use crate::{
    Error, Result,
    boolean::{
        domain::{Domain, Trim, UvMap},
        sew::{self, OutFace, PieceGeom},
    },
    geom::Surface,
    math::{Point3, Vec2},
    tol,
    topo::{Brep, pcurve::fit_spline},
};
use contour::{Chain, Grid};

/// A stretch of an input piece after it is cut where the body begins to reach into it.
#[derive(Clone, Debug)]
struct Sub {
    range: (f64, f64),
    piece: usize,
    v0: usize,
    v1: usize,
    held: bool,
}

/// A curve across a face where the reached part of it begins: its points in the face's plane,
/// and the vertices it runs between.
#[derive(Clone, Debug)]
struct Run {
    uvs: Vec<Vec2>,
    ends: (usize, usize),
}

/// A face being trimmed.
struct Work {
    face: OutFace,
    dom: Domain,
    /// The face's loops, each trim cut where its piece was: the trim and whether it is reached.
    bounds: Vec<Vec<(Trim, bool)>>,
    /// The vertices on its boundary where the reached part begins, and their parameters.
    marks: Vec<(usize, Vec2)>,
    runs: Vec<Run>,
    /// A cell of the grid the face was searched on.
    cell: Vec2,
    /// Whether the body reaches all of it, none of it, or some.
    state: State,
}

/// For each run, by face and run, the piece it is, whether it runs along the piece, and the map of
/// its trim on its face.
type Made = HashMap<(usize, usize), (usize, bool, UvMap)>;

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    None,
    All,
    Some,
}

/// The solid the faces bound once each is trimmed to what `reach` does not reach into: `faces`
/// on `pieces` between the vertices of `pool`, which bound the whole swept solid where nothing
/// folds or overlaps.
pub(super) fn assemble(
    faces: Vec<OutFace>,
    mut pool: Vec<Point3>,
    mut pieces: Vec<PieceGeom>,
    reach: &Reach,
    folds: &HashMap<usize, Vec<f64>>,
) -> Result<Brep> {
    let size = reach.size;
    let used: BTreeSet<usize> = faces
        .iter()
        .flat_map(|f| f.loops.iter().flatten().map(|t| t.piece))
        .collect();
    let mut subs: HashMap<usize, Vec<Sub>> = HashMap::new();
    for &k in &used {
        let at = folds.get(&k).map_or(&[][..], |v| &v[..]);
        subs.insert(k, split(&mut pool, &mut pieces, k, reach, at));
    }
    let mut works: Vec<Work> = Vec::with_capacity(faces.len());
    for face in faces {
        works.push(search(face, &subs, &mut pool, reach)?);
    }
    // Every vertex where a run may end, cut into each run it lies on.
    let marks: Vec<usize> = {
        let mut m: BTreeSet<usize> = BTreeSet::new();
        for w in &works {
            for r in &w.runs {
                m.insert(r.ends.0);
                m.insert(r.ends.1);
            }
            for (v, _) in &w.marks {
                m.insert(*v);
            }
        }
        m.into_iter().collect()
    };
    for w in &mut works {
        for k in 0..w.runs.len() {
            let uvs = densify(w, &w.runs[k], reach);
            w.runs[k].uvs = uvs;
        }
        cut_runs(w, &marks, &pool, reach, size);
    }
    let partner = partners(&works)?;
    let made = shared(&works, &partner, &pool, &mut pieces, reach, folds)?;
    let mut out: Vec<OutFace> = Vec::new();
    for (w, work) in works.iter().enumerate() {
        match work.state {
            State::All => continue,
            State::None => {
                let mut face = work.face.clone();
                face.loops = work
                    .bounds
                    .iter()
                    .map(|l| l.iter().map(|(t, _)| t.clone()).collect())
                    .collect();
                out.push(face);
            }
            State::Some => out.extend(rebuild(work, w, &made, &pieces, &pool, reach)?),
        }
    }
    let vtol = vec![tol::LINEAR; pool.len()];
    sew::assemble(out, &pool, &vtol, &pieces)
}

/// Each run (face, run) and the run it is one curve with: of another face, or of the same face
/// elsewhere, between the same two vertices and nearest it there.
fn partners(works: &[Work]) -> Result<HashMap<(usize, usize), (usize, usize)>> {
    let found: Vec<(usize, usize)> = works
        .iter()
        .enumerate()
        .flat_map(|(w, work)| (0..work.runs.len()).map(move |r| (w, r)))
        .collect();
    let mut partner: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for i in 0..found.len() {
        if partner.contains_key(&found[i]) {
            continue;
        }
        let (wa, ra) = found[i];
        let a = &works[wa].runs[ra];
        let mid = middle(&works[wa], a);
        let mut best: Option<((usize, usize), f64)> = None;
        for &(wb, rb) in &found[i + 1..] {
            if partner.contains_key(&(wb, rb)) {
                continue;
            }
            let b = &works[wb].runs[rb];
            if a.ends != b.ends && a.ends != (b.ends.1, b.ends.0) {
                continue;
            }
            let d = near(&works[wb], b, mid);
            if best.is_none_or(|x| d < x.1) {
                best = Some(((wb, rb), d));
            }
        }
        match best {
            Some((other, d)) if d <= 0.25 => {
                partner.insert((wa, ra), other);
                partner.insert(other, (wa, ra));
            }
            _ => {
                return Err(Error::Numerical(
                    "a curve where a swept face meets the swept solid's inside has no face \
                     across it"
                        .into(),
                ));
            }
        }
    }
    Ok(partner)
}

/// The piece of the result each pair of partner runs is, fitted once through the exact points of
/// one of them, and the map of each run's trim on its face: the spline through the run's points.
fn shared(
    works: &[Work],
    partner: &HashMap<(usize, usize), (usize, usize)>,
    pool: &[Point3],
    pieces: &mut Vec<PieceGeom>,
    reach: &Reach,
    folds: &HashMap<usize, Vec<f64>>,
) -> Result<Made> {
    let mut keys: Vec<&(usize, usize)> = partner.keys().collect();
    keys.sort_unstable();
    let mut made: Made = HashMap::new();
    for &(wa, ra) in keys {
        if made.contains_key(&(wa, ra)) {
            continue;
        }
        let (wb, rb) = partner[&(wa, ra)];
        // The curve is found on the run whose face crosses the body where the motion starts or
        // stops, if either does: there it is where the body's own depth changes sign, a smooth
        // function, and the reach, whose depth rises only with the square of the distance where
        // the two faces nearly touch, is not needed.
        let (span0, span1) = reach.span();
        let near_end = |w: usize, r: usize| -> Option<bool> {
            let t = partner_place(&works[w], &works[w].runs[r], reach)?;
            let eps = 1e-9 * (span1 - span0);
            if t <= span0 + eps {
                Some(false)
            } else if t >= span1 - eps {
                Some(true)
            } else {
                None
            }
        };
        let exact = !folds.is_empty();
        let near_end = |w: usize, r: usize| near_end(w, r).filter(|_| exact);
        let (fw, fr, end) = match (near_end(wa, ra), near_end(wb, rb)) {
            (Some(e), _) => (wa, ra, Some(e)),
            (None, Some(e)) => (wb, rb, Some(e)),
            _ => (wa, ra, None),
        };
        let g = move |p: Point3| match end {
            Some(e) => reach.depth_at_end(p, e),
            None => reach.deepest(p) - reach.hair,
        };
        let a = &works[fw].runs[fr];
        let ends = (pool[a.ends.0], pool[a.ends.1]);
        // The run's ends on the face lie on the fitted curves of the boundary, which may stray from
        // the exact ones by their fit: the difference is spread along the run.
        // Each end's point found on the curve, unless it strays from the vertex further than the
        // vertex may lie from it: at an end on a fold the function the curve is found by vanishes
        // to second order, and the point found right at the end is the hair's, not the curve's.
        let end_gap = |f: f64, v: Point3| {
            let p = along(&works[fw], a, f, &g);
            if p.dist(v) <= 1e-6 * reach.size.max(1.0) {
                v - p
            } else {
                v - along(&works[fw], a, f, &|_| 0.0)
            }
        };
        let gaps = (end_gap(0.0, ends.0), end_gap(1.0, ends.1));
        let at = |s: f64| along(&works[fw], a, s, &g) + gaps.0 * (1.0 - s) + gaps.1 * s;
        let (curve, range) = fit_exact(&at, (0.0, 1.0), ends, reach.size)?;
        let curve = Rc::new(curve);
        pieces.push(PieceGeom {
            curve: Some(curve.clone()),
            range,
            v0: a.ends.0,
            v1: a.ends.1,
            tol: tol::LINEAR,
        });
        let id = pieces.len() - 1;
        for (w, r) in [(wa, ra), (wb, rb)] {
            let run = &works[w].runs[r];
            let forward = run.ends == a.ends;
            let face = &works[w].face.surface;
            let mut pts = if (w, r) == (fw, fr) {
                // The fitted run's own exact points, moved onto the curve as it was found.
                let n = (2 * run.uvs.len()).max(64);
                let mut pts: Vec<Vec2> = (0..=n)
                    .map(|k| along_uv(&works[w], run, k as f64 / n as f64, &g))
                    .collect();
                pts[0] = run.uvs[0];
                pts[n] = run.uvs[run.uvs.len() - 1];
                pts
            } else {
                run.uvs.clone()
            };
            if !forward {
                pts.reverse();
            }
            let fitted = if (w, r) == (fw, fr) {
                fit_spline(face, &curve, range, &pts)
            } else {
                projected(face, &curve, range, &pts)
                    .or_else(|| fit_spline(face, &curve, range, &pts))
            };
            let map = match fitted {
                Some(s) => UvMap::Spline(Rc::new(s)),
                None => {
                    let (sp, off) = own_pcurve(&works[w].face.surface, &pts, &curve, range)?;
                    let pg = &mut pieces[id];
                    pg.tol = pg.tol.max(2.0 * off);
                    UvMap::Spline(Rc::new(sp))
                }
            };
            made.insert((w, r), (id, forward, map));
        }
    }
    Ok(made)
}

/// Cuts piece `k` where the body begins to reach into it, adding the vertices there to `pool`
/// and the stretches between to `pieces`: its stretches in order, each reached or not.
fn split(
    pool: &mut Vec<Point3>,
    pieces: &mut Vec<PieceGeom>,
    k: usize,
    reach: &Reach,
    folds: &[f64],
) -> Vec<Sub> {
    let pg = pieces[k].clone();
    let whole = |held| {
        vec![Sub {
            range: pg.range,
            piece: k,
            v0: pg.v0,
            v1: pg.v1,
            held,
        }]
    };
    let Some(curve) = pg.curve.clone() else {
        return whole(reach.holds(pool[pg.v0]));
    };
    let n = 64;
    let (a, b) = pg.range;
    let ts: Vec<f64> = (0..=n).map(|i| a + (b - a) * i as f64 / n as f64).collect();
    let hs: Vec<bool> = ts.iter().map(|&t| reach.holds(curve.eval(t))).collect();
    let mut cuts: Vec<f64> = Vec::new();
    for i in 0..n {
        if hs[i] != hs[i + 1] {
            let (mut lo, mut hi) = (ts[i], ts[i + 1]);
            for _ in 0..60 {
                let m = 0.5 * (lo + hi);
                if reach.holds(curve.eval(m)) == hs[i] {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            // Where the piece's swept face folds, the reached part begins exactly at the fold, and
            // the reach, whose depth rises only with a power of the distance there, finds it a
            // little off.
            let t = 0.5 * (lo + hi);
            let near = folds
                .iter()
                .copied()
                .filter(|&f| (f - t).abs() <= 1e-3 * (b - a))
                .min_by(|x, y| (x - t).abs().total_cmp(&(y - t).abs()));
            cuts.push(near.unwrap_or(t));
        }
    }
    if cuts.is_empty() {
        return whole(hs[0]);
    }
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut from = (a, pg.v0);
    let mut held = hs[0];
    for (i, &t) in cuts.iter().chain(std::iter::once(&b)).enumerate() {
        let to = if i == cuts.len() {
            pg.v1
        } else {
            pool.push(curve.eval(t));
            pool.len() - 1
        };
        pieces.push(PieceGeom {
            curve: Some(curve.clone()),
            range: (from.0, t),
            v0: from.1,
            v1: to,
            tol: pg.tol,
        });
        out.push(Sub {
            range: (from.0, t),
            piece: pieces.len() - 1,
            v0: from.1,
            v1: to,
            held,
        });
        from = (t, to);
        held = !held;
    }
    out
}

/// The trim `tr` of `s` cut where its piece was: each stretch and whether it is reached.
fn cut_trim(s: &Surface, tr: &Trim, subs: &[Sub]) -> Vec<(Trim, bool)> {
    if subs.len() == 1 {
        return vec![(
            Trim {
                piece: subs[0].piece,
                ..tr.clone()
            },
            subs[0].held,
        )];
    }
    let n = 64;
    let poly = tr.polyline(s, n);
    let uv_at = |t: f64| {
        // The polyline's point nearest in parameter is the hint a periodic parameter continues
        // from.
        let f = ((t - tr.t0) / (tr.t1 - tr.t0)).clamp(0.0, 1.0);
        let j = (f * n as f64).round() as usize;
        tr.uv(s, t, poly[j.min(n)])
    };
    let forward = tr.t1 >= tr.t0;
    let mut out: Vec<(Trim, bool)> = subs
        .iter()
        .map(|sub| {
            let (t0, t1, v0, v1) = if forward {
                (sub.range.0, sub.range.1, sub.v0, sub.v1)
            } else {
                (sub.range.1, sub.range.0, sub.v1, sub.v0)
            };
            let uv0 = if t0 == tr.t0 { tr.uv0 } else { uv_at(t0) };
            let uv1 = if t1 == tr.t1 { tr.uv1 } else { uv_at(t1) };
            (
                Trim {
                    piece: sub.piece,
                    t0,
                    t1,
                    uv0,
                    uv1,
                    v0,
                    v1,
                    ..tr.clone()
                },
                sub.held,
            )
        })
        .collect();
    if !forward {
        out.reverse();
    }
    out
}

/// Cells of a face's grid along its longer side.
const CELLS: f64 = 96.0;

/// Face `face` cut where its pieces were, and the runs across it where the reached part begins.
fn search(
    face: OutFace,
    subs: &HashMap<usize, Vec<Sub>>,
    pool: &mut Vec<Point3>,
    reach: &Reach,
) -> Result<Work> {
    let s = face.surface.clone();
    let bounds: Vec<Vec<(Trim, bool)>> = face
        .loops
        .iter()
        .map(|l| {
            l.iter()
                .flat_map(|tr| cut_trim(&s, tr, &subs[&tr.piece]))
                .collect()
        })
        .collect();
    let dom = Domain::new(s.clone(), face.reversed, face.loops.clone());
    let mut marks: Vec<(usize, Vec2)> = Vec::new();
    for l in &bounds {
        for (k, (_, held)) in l.iter().enumerate() {
            let (next, nh) = &l[(k + 1) % l.len()];
            if held != nh {
                marks.push((next.v0, next.uv0));
            }
        }
    }
    // The grid: a cell beyond the face's box each way, so every curve that crosses the face
    // leaves the grid outside it.
    let (lo, hi) = (dom.lo, dom.hi);
    let mid = (lo + hi) * 0.5;
    let length = |a: Vec2, b: Vec2| {
        (0..16)
            .map(|k| {
                let (p, q) = (a.lerp(b, k as f64 / 16.0), a.lerp(b, (k + 1) as f64 / 16.0));
                s.eval(p.x, p.y).dist(s.eval(q.x, q.y))
            })
            .sum::<f64>()
    };
    let lu = length(Vec2::new(lo.x, mid.y), Vec2::new(hi.x, mid.y)).max(1e-300);
    let lv = length(Vec2::new(mid.x, lo.y), Vec2::new(mid.x, hi.y)).max(1e-300);
    let longest = lu.max(lv);
    let nu = ((CELLS * lu / longest).ceil() as usize).max(8);
    let nv = ((CELLS * lv / longest).ceil() as usize).max(8);
    let cell = Vec2::new((hi.x - lo.x) / nu as f64, (hi.y - lo.y) / nv as f64);
    let grid = Grid {
        lo: lo - cell,
        hi: hi + cell,
        n: (nu + 2, nv + 2),
    };
    let (chains, vals) = contour::chains(&s, reach, grid);
    let boundary_held = bounds.iter().flatten().map(|(_, h)| *h);
    let all_bound = boundary_held.clone().all(|h| h);
    let none_bound = !bounds.iter().flatten().any(|(_, h)| *h);
    let inside_nodes: Vec<bool> = {
        let (nx, ny) = grid.n;
        (0..=nx)
            .flat_map(|i| (0..=ny).map(move |j| (i, j)))
            .map(|(i, j)| {
                let uv = Vec2::new(
                    grid.lo.x + (grid.hi.x - grid.lo.x) * i as f64 / nx as f64,
                    grid.lo.y + (grid.hi.y - grid.lo.y) * j as f64 / ny as f64,
                );
                dom.winding(uv) != 0
            })
            .collect()
    };
    let held_inside = vals.iter().zip(&inside_nodes).any(|(&v, &i)| v && i);
    let free_inside = vals.iter().zip(&inside_nodes).any(|(&v, &i)| !v && i);
    let mut work = Work {
        face,
        dom,
        bounds,
        marks,
        runs: Vec::new(),
        cell,
        state: State::Some,
    };
    if work.marks.is_empty() && none_bound && !held_inside {
        work.state = State::None;
        return Ok(work);
    }
    if work.marks.is_empty() && all_bound && !free_inside {
        work.state = State::All;
        return Ok(work);
    }
    for c in chains {
        clip(&mut work, c, pool)?;
    }
    Ok(work)
}

/// The parts of `chain` inside the face, each between the vertices where it crosses the face's
/// boundary; a part that closes inside it gets two vertices of its own.
fn clip(work: &mut Work, mut chain: Chain, pool: &mut Vec<Point3>) -> Result<()> {
    // Two cuts a sliver of a cell apart, either side of a grid node, make a chord too short to
    // say which way the curve runs: the second is left out.
    let unit = |q: Vec2| Vec2::new(q.x / work.cell.x, q.y / work.cell.y);
    let last = chain.uvs[chain.uvs.len() - 1];
    let mut thin: Vec<Vec2> = Vec::with_capacity(chain.uvs.len());
    for &q in &chain.uvs[..chain.uvs.len() - 1] {
        if thin.last().is_none_or(|&p| unit(q - p).norm() >= 0.25) {
            thin.push(q);
        }
    }
    if thin.len() > 1 && unit(last - thin[thin.len() - 1]).norm() < 0.25 {
        thin.pop();
    }
    thin.push(last);
    chain.uvs = thin;
    // A point on the face's boundary is outside: the reached part of the surface beyond the face
    // often begins right at its boundary, and a chain can run along it there.
    let edge: Vec<(Vec2, Vec2)> = work
        .dom
        .samples()
        .iter()
        .flatten()
        .flat_map(|pl| pl.windows(2).map(|w| (unit(w[0]), unit(w[1]))))
        .collect();
    let off_edge = |q: Vec2| {
        let q = unit(q);
        edge.iter().all(|&(a, b)| {
            let d = b - a;
            let t = ((q - a).dot(d) / d.dot(d).max(1e-300)).clamp(0.0, 1.0);
            q.dist(a + d * t) > 0.05
        })
    };
    let inside: Vec<bool> = chain
        .uvs
        .iter()
        .map(|&q| work.dom.winding(q) != 0 && off_edge(q))
        .collect();
    let s = &work.face.surface;
    if chain.closed && inside.iter().all(|&i| i) {
        let mut uvs = chain.uvs;
        uvs.pop();
        let half = uvs.len() / 2;
        let a = pool.len();
        pool.push(s.eval(uvs[0].x, uvs[0].y));
        pool.push(s.eval(uvs[half].x, uvs[half].y));
        let first: Vec<Vec2> = uvs[..=half].to_vec();
        let mut second: Vec<Vec2> = uvs[half..].to_vec();
        second.push(uvs[0]);
        work.runs.push(Run {
            uvs: first,
            ends: (a, a + 1),
        });
        work.runs.push(Run {
            uvs: second,
            ends: (a + 1, a),
        });
        return Ok(());
    }
    let n = chain.uvs.len();
    // A closed chain that leaves the face is read from a point outside it.
    let start = if chain.closed {
        inside.iter().position(|&i| !i).unwrap_or(0)
    } else {
        0
    };
    let order: Vec<usize> = if chain.closed {
        (0..n - 1)
            .map(|k| (start + k) % (n - 1))
            .chain([start])
            .collect()
    } else {
        (0..n).collect()
    };
    let mut current: Option<(usize, Vec<Vec2>)> = None;
    for w in order.windows(2) {
        let (i, j) = (w[0], w[1]);
        let (a, b) = (chain.uvs[i], chain.uvs[j]);
        match (inside[i], inside[j]) {
            (false, true) => {
                let (v, uv) = crossing(work, a, b)?;
                current = Some((v, vec![uv, b]));
            }
            (true, true) => {
                if let Some((_, pts)) = current.as_mut() {
                    pts.push(b);
                }
            }
            (true, false) => {
                let (v, uv) = crossing(work, a, b)?;
                if let Some((from, mut pts)) = current.take() {
                    pts.push(uv);
                    work.runs.push(Run {
                        uvs: pts,
                        ends: (from, v),
                    });
                }
            }
            (false, false) => {}
        }
    }
    if current.is_some() {
        return Err(Error::Numerical(
            "a curve where a swept face meets the swept solid's inside ends inside the face".into(),
        ));
    }
    Ok(())
}

/// The vertex where the boundary of the face crosses the segment from `a` to `b`: the vertex on
/// its boundary where the reached part begins nearest the segment.
fn crossing(work: &Work, a: Vec2, b: Vec2) -> Result<(usize, Vec2)> {
    let unit = |q: Vec2| Vec2::new(q.x / work.cell.x, q.y / work.cell.y);
    let to_segment = |p: Vec2| {
        let (p, a, b) = (unit(p), unit(a), unit(b));
        let d = b - a;
        let t = ((p - a).dot(d) / d.dot(d).max(1e-300)).clamp(0.0, 1.0);
        p.dist(a + d * t)
    };
    let best = work
        .marks
        .iter()
        .map(|&(v, uv)| (v, uv, to_segment(uv)))
        .min_by(|x, y| x.2.total_cmp(&y.2));
    match best {
        Some((v, uv, d)) if d <= 4.0 => Ok((v, uv)),
        _ => Err(Error::Numerical(
            "a curve where a swept face meets the swept solid's inside crosses the face's \
             boundary where nothing was found"
                .into(),
        )),
    }
}

/// Cuts each run of `work` at each vertex of `marks` that lies on it, between its ends.
fn cut_runs(work: &mut Work, marks: &[usize], pool: &[Point3], reach: &Reach, size: f64) {
    let s = work.face.surface.clone();
    let mut k = 0;
    while k < work.runs.len() {
        let run = work.runs[k].clone();
        let pts: Vec<Point3> = run.uvs.iter().map(|q| s.eval(q.x, q.y)).collect();
        let mut hit: Option<(usize, usize, Vec2)> = None;
        'marks: for &v in marks {
            if v == run.ends.0 || v == run.ends.1 {
                continue;
            }
            let p = pool[v];
            for i in 0..pts.len() - 1 {
                let (a, b) = (pts[i], pts[i + 1]);
                let d = b - a;
                let len = d.norm();
                let t = ((p - a).dot(d) / (len * len).max(1e-300)).clamp(0.0, 1.0);
                if p.dist(a + d * t) > 0.25 * len + 1e-7 * size {
                    continue;
                }
                let hint = run.uvs[i].lerp(run.uvs[i + 1], t);
                let uv = s.project_near(p, hint);
                if s.eval(uv.x, uv.y).dist(p) > 1e-7 * size.max(1.0) {
                    continue;
                }
                // On the curve itself, not just beside it: the curve's exact point there is the
                // vertex.
                let across = across(work.cell, run.uvs[i + 1] - run.uvs[i]);
                if let Some(q) = contour::settle(&s, reach, uv, across, 0.25)
                    && s.eval(q.x, q.y).dist(p) <= 1e-6 * size.max(1.0)
                {
                    hit = Some((i, v, uv));
                    break 'marks;
                }
            }
        }
        let Some((i, v, uv)) = hit else {
            k += 1;
            continue;
        };
        let mut first: Vec<Vec2> = run.uvs[..=i].to_vec();
        first.push(uv);
        let mut second = vec![uv];
        second.extend_from_slice(&run.uvs[i + 1..]);
        work.runs[k] = Run {
            uvs: first,
            ends: (run.ends.0, v),
        };
        work.runs.push(Run {
            uvs: second,
            ends: (v, run.ends.1),
        });
    }
}

/// The points of `run` with exact points added between any two more than half a cell apart, so
/// its polyline follows the curve closely: where a run leaves the face nearly along its boundary,
/// the grid finds no point between the last cell it crossed and the vertex where it ends.
fn densify(work: &Work, run: &Run, reach: &Reach) -> Vec<Vec2> {
    let s = &work.face.surface;
    let unit = |q: Vec2| Vec2::new(q.x / work.cell.x, q.y / work.cell.y);
    let mut out = vec![run.uvs[0]];
    for w in run.uvs.windows(2) {
        let mut stack = vec![(w[0], w[1], 0)];
        let mut seg = Vec::new();
        while let Some((a, b, depth)) = stack.pop() {
            let len = unit(b - a).norm();
            if len <= 0.5 || depth >= 6 {
                seg.push(b);
                continue;
            }
            let mid = a.lerp(b, 0.5);
            match contour::settle(s, reach, mid, across(work.cell, b - a), 0.5 * len) {
                Some(m) => {
                    stack.push((m, b, depth + 1));
                    stack.push((a, m, depth + 1));
                }
                None => seg.push(b),
            }
        }
        out.extend(seg);
    }
    out
}

/// The point halfway along `run` in space.
fn middle(work: &Work, run: &Run) -> Point3 {
    let q = run.uvs[run.uvs.len() / 2];
    work.face.surface.eval(q.x, q.y)
}

/// How near `run`'s polyline passes `p`, in lengths of the polyline's segment there.
fn near(work: &Work, run: &Run, p: Point3) -> f64 {
    let s = &work.face.surface;
    let pts: Vec<Point3> = run.uvs.iter().map(|q| s.eval(q.x, q.y)).collect();
    let mut best = f64::INFINITY;
    for w in pts.windows(2) {
        let d = w[1] - w[0];
        let t = ((p - w[0]).dot(d) / d.dot(d).max(1e-300)).clamp(0.0, 1.0);
        best = best.min(p.dist(w[0] + d * t) / d.norm().max(1e-300));
    }
    best
}

/// The exact point a share `f` of the way along `run`, by length in the grid's cells: the point
/// there on its polyline moved across it onto the curve.
fn along(work: &Work, run: &Run, f: f64, g: &dyn Fn(Point3) -> f64) -> Point3 {
    let q = along_uv(work, run, f, g);
    work.face.surface.eval(q.x, q.y)
}

/// [`along`]'s point in the face's parameters.
fn along_uv(work: &Work, run: &Run, f: f64, g: &dyn Fn(Point3) -> f64) -> Vec2 {
    let s = &work.face.surface;
    let unit = |q: Vec2| Vec2::new(q.x / work.cell.x, q.y / work.cell.y);
    let mut lens = vec![0.0];
    for w in run.uvs.windows(2) {
        lens.push(lens[lens.len() - 1] + unit(w[1] - w[0]).norm());
    }
    let total = lens[lens.len() - 1];
    let x = f * total;
    let i = lens
        .partition_point(|&l| l <= x)
        .clamp(1, run.uvs.len() - 1);
    let h = ((x - lens[i - 1]) / (lens[i] - lens[i - 1]).max(1e-300)).clamp(0.0, 1.0);
    let q = run.uvs[i - 1].lerp(run.uvs[i], h);
    let span = 0.5 * (lens[i] - lens[i - 1]).max(0.05);
    let dir = across(work.cell, run.uvs[i] - run.uvs[i - 1]);
    contour::settle_on(s, g, q, dir, span).unwrap_or(q)
}

/// The spline in the curve's parameter through the curve's points projected onto `s`, each from
/// the nearest of `hints` (the face's own points of the run), where the projection holds every
/// point and the face over the spline stays within [`PCURVE_FIT`] of the curve; else `None`. On
/// a face whose map is regular all along the curve, as a cap of the sweep, this is exact however
/// poorly the run's own points were found near a fold.
fn projected(
    s: &Surface,
    curve: &crate::geom::Curve,
    range: (f64, f64),
    hints: &[Vec2],
) -> Option<crate::geom::BSpline> {
    use crate::math::Vec3;
    let at: Vec<Point3> = hints.iter().map(|q| s.eval(q.x, q.y)).collect();
    let mut n = 32;
    while n <= 1024 {
        let ts: Vec<f64> = (0..=n)
            .map(|k| range.0 + (range.1 - range.0) * k as f64 / n as f64)
            .collect();
        let mut uvs = Vec::with_capacity(n + 1);
        let mut last: Option<Vec2> = None;
        for &t in &ts {
            let p = curve.eval(t);
            let hint = match last {
                Some(l) => l,
                None => {
                    let k =
                        (0..at.len()).min_by(|&a, &b| at[a].dist(p).total_cmp(&at[b].dist(p)))?;
                    hints[k]
                }
            };
            let uv = s.project_near(p, hint);
            if s.eval(uv.x, uv.y).dist(p) > PCURVE_FIT {
                return None;
            }
            uvs.push(uv);
            last = Some(uv);
        }
        let pts: Vec<Vec3> = uvs.iter().map(|q| Vec3::new(q.x, q.y, 0.0)).collect();
        let spline = crate::geom::BSpline::interpolate_at(&pts, &ts).ok()?;
        let fits = ts.windows(2).all(|w| {
            [0.25, 0.5, 0.75].iter().all(|f| {
                let t = w[0] + (w[1] - w[0]) * f;
                let q = spline.eval(t);
                s.eval(q.x, q.y).dist(curve.eval(t)) <= PCURVE_FIT
            })
        });
        if fits {
            return Some(spline);
        }
        n *= 2;
    }
    None
}

/// How far a run's pcurve may lead its face from the run's curve.
const PCURVE_FIT: f64 = tol::LINEAR / 4.0;

/// The spline in the curve's parameter through a run's points `pts` on face `s`, in order from
/// the curve's start, each at the curve's parameter nearest its point, and how far the face over it
/// strays from the curve. Where the face's map nearly drops rank, as where a run ends on a fold of
/// a swept sheet, a projection onto the face lands on the wrong foot; the run's own points need
/// none.
fn own_pcurve(
    s: &Surface,
    pts: &[Vec2],
    curve: &crate::geom::Curve,
    range: (f64, f64),
) -> Result<(crate::geom::BSpline, f64)> {
    use crate::math::Vec3;
    let (lo, hi) = range;
    let spacing = 1e-9 * (hi - lo);
    let mut ts = vec![lo];
    let mut uvs = vec![pts[0]];
    for q in &pts[1..pts.len() - 1] {
        let prev = ts[ts.len() - 1];
        let t = curve.project(s.eval(q.x, q.y), prev, hi);
        if t > prev + spacing && t < hi - spacing {
            ts.push(t);
            uvs.push(*q);
        }
    }
    ts.push(hi);
    uvs.push(pts[pts.len() - 1]);
    let at: Vec<Vec3> = uvs.iter().map(|q| Vec3::new(q.x, q.y, 0.0)).collect();
    let spline = crate::geom::BSpline::interpolate_at(&at, &ts)?;
    let mut off = 0.0f64;
    for w in ts.windows(2) {
        for f in [0.25, 0.5, 0.75] {
            let t = w[0] + (w[1] - w[0]) * f;
            let q = spline.eval(t);
            off = off.max(s.eval(q.x, q.y).dist(curve.eval(t)));
        }
    }
    Ok((spline, off))
}

/// Where along the motion the body reaches deepest just beside the middle of `run`, on the side
/// it reaches: the face the run's face crosses there is swept by the body at that place.
fn partner_place(work: &Work, run: &Run, reach: &Reach) -> Option<f64> {
    let s = &work.face.surface;
    let k = run.uvs.len() / 2;
    let (p, q) = (
        run.uvs[k.saturating_sub(1)],
        run.uvs[(k + 1).min(run.uvs.len() - 1)],
    );
    let m = run.uvs[k];
    let dir = across(work.cell, q - p) * 0.01;
    [m + dir, m - dir]
        .into_iter()
        .map(|x| s.eval(x.x, x.y))
        .find(|&x| reach.holds(x))
        .map(|x| reach.deepest_place(x))
}

/// The direction square to `dir` in a plane measured in grid cells of size `cell`, one cell
/// long there.
fn across(cell: Vec2, dir: Vec2) -> Vec2 {
    let d = Vec2::new(dir.x / cell.x, dir.y / cell.y);
    Vec2::new(-d.y * cell.x, d.x * cell.y) / d.norm().max(1e-300)
}

/// The faces `work` is trimmed to: each region of its plane between its boundary and its runs
/// that the body does not reach into.
fn rebuild(
    work: &Work,
    w: usize,
    made: &Made,
    pieces: &[PieceGeom],
    pool: &[Point3],
    reach: &Reach,
) -> Result<Vec<OutFace>> {
    let s = &work.face.surface;
    let mut nodes: Vec<Vec2> = Vec::new();
    let mut ids: Vec<usize> = Vec::new();
    let unit = |q: Vec2| Vec2::new(q.x / work.cell.x, q.y / work.cell.y);
    let node = |p: Vec2, id: usize, nodes: &mut Vec<Vec2>, ids: &mut Vec<usize>| -> usize {
        if let Some(k) =
            (0..nodes.len()).find(|&k| ids[k] == id && unit(nodes[k] - p).norm() <= 1e-3)
        {
            return k;
        }
        nodes.push(p);
        ids.push(id);
        nodes.len() - 1
    };
    // What each link stands for: a stretch of the boundary, or a run.
    enum Stands {
        Bound(Trim),
        Run(usize),
    }
    let mut links: Vec<Link> = Vec::new();
    let mut stands: Vec<Stands> = Vec::new();
    for l in &work.bounds {
        for (tr, _) in l {
            let a = node(tr.uv0, tr.v0, &mut nodes, &mut ids);
            let b = node(tr.uv1, tr.v1, &mut nodes, &mut ids);
            let mut pts = tr.polyline(s, tr.steps());
            let last = pts.len() - 1;
            pts[0] = nodes[a];
            pts[last] = nodes[b];
            links.push(Link { a, b, pts });
            stands.push(Stands::Bound(tr.clone()));
        }
    }
    for (r, run) in work.runs.iter().enumerate() {
        let a = node(run.uvs[0], run.ends.0, &mut nodes, &mut ids);
        let b = node(run.uvs[run.uvs.len() - 1], run.ends.1, &mut nodes, &mut ids);
        let mut pts = run.uvs.clone();
        let last = pts.len() - 1;
        pts[0] = nodes[a];
        pts[last] = nodes[b];
        links.push(Link { a, b, pts });
        stands.push(Stands::Run(r));
    }
    let reach_len = 1e-3 * work.cell.x.min(work.cell.y);
    let regions = graph::regions(&nodes, &links, reach_len)?;
    let mut out = Vec::new();
    for region in &regions {
        let Some(at) = graph::inside(&links, region, reach_len) else {
            return Err(Error::Numerical(
                "a trimmed piece of a swept face has no inside".into(),
            ));
        };
        if contour::held(s, reach, at) {
            continue;
        }
        let mut loops = Vec::with_capacity(region.loops.len());
        for lp in &region.loops {
            let mut trims = Vec::with_capacity(lp.len());
            for &(k, dir) in lp {
                let tr = match &stands[k] {
                    Stands::Bound(tr) => tr.clone(),
                    Stands::Run(r) => {
                        let (piece, forward, map) = &made[&(w, *r)];
                        let pg = &pieces[*piece];
                        let run = &work.runs[*r];
                        let (t0, t1) = if *forward {
                            (pg.range.0, pg.range.1)
                        } else {
                            (pg.range.1, pg.range.0)
                        };
                        Trim {
                            piece: *piece,
                            forward: *forward,
                            curve: pg.curve.clone(),
                            pole: pool[run.ends.0],
                            t0,
                            t1,
                            map: map.clone(),
                            uv0: links[k].pts[0],
                            uv1: links[k].pts[links[k].pts.len() - 1],
                            v0: run.ends.0,
                            v1: run.ends.1,
                        }
                    }
                };
                trims.push(if dir { tr } else { tr.reversed() });
            }
            // Loops run counter-clockwise in the plane on a face that is not reversed.
            if work.face.reversed {
                trims.reverse();
                for t in &mut trims {
                    *t = t.reversed();
                }
            }
            loops.push(trims);
        }
        // The face lies on one part of a sheet that folds or runs into itself, which is all its
        // surface is: past the fold or across the self-crossing a point of it has another foot.
        let (mut lo, mut hi) = (
            Vec2::new(f64::INFINITY, f64::INFINITY),
            Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for &(k, _) in region.loops.iter().flatten() {
            for p in &links[k].pts {
                lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
                hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
            }
        }
        let pad = work.cell * 0.5;
        let surface = restrict::restricted(s, lo - pad, hi + pad).unwrap_or_else(|| s.clone());
        out.push(OutFace {
            surface,
            reversed: work.face.reversed,
            names: work.face.names.clone(),
            loops,
            memo: None,
            side: work.face.side,
        });
    }
    Ok(out)
}
