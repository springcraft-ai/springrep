//! A turned body's sweep read off its drawing in the plane of orbits ([`cells`](super::cells)).
//! Over each cell the body is a stack of intervals of angle, and the sweep is each interval
//! lengthened by the turn and merged with those it reaches. A point is on the swept solid's
//! boundary where it ends a merged interval: the low ends are pieces of the body's faces where
//! the turn starts, the high ends pieces where it ends, and over a link between two cells whose
//! intervals differ stands a wall of orbit arcs, a piece of the surface the link's curve sweeps.
//! Nothing is kept or dropped by a test on the faces made: each is made because the intervals
//! say it is there.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use super::cells::{Axis, Cells, Gen, Hit};
use super::cut::Cut;
use super::moving::graph::{self, Link};
use super::sheet::Steady;
use crate::{
    Error, Result,
    boolean::{
        domain::{Trim, UvMap},
        sew::{self, OutFace, PieceGeom},
    },
    geom::Curve,
    math::{Point3, Transform, Vec2},
    tol,
    topo::{Brep, FaceName},
};

/// What the body sweeps over a cell: from where an orbit first goes into it to where the orbit
/// last leaves it, a turn on.
#[derive(Clone, Copy, Debug)]
struct Swept {
    lo: Hit,
    hi: Hit,
}

/// A curve of the result over a link: a stretch of a piece of the cut body, where the turn
/// starts or where it ends, from the link's `a` to its `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Lift {
    piece: usize,
    t: (f64, f64),
    end: bool,
}

/// The vertices, edge pieces and faces made so far.
struct Out {
    pool: Vec<Point3>,
    pieces: Vec<PieceGeom>,
    faces: Vec<OutFace>,
    made: HashMap<(usize, u64, u64, bool), usize>,
    near: f64,
}

impl Out {
    fn vertex(&mut self, p: Point3) -> usize {
        match self.pool.iter().position(|q| q.dist(p) <= self.near) {
            Some(k) => k,
            None => {
                self.pool.push(p);
                self.pool.len() - 1
            }
        }
    }

    fn piece(&mut self, curve: Rc<Curve>, t: (f64, f64)) -> usize {
        let (a, b) = (self.vertex(curve.eval(t.0)), self.vertex(curve.eval(t.1)));
        self.pieces.push(PieceGeom {
            curve: Some(curve),
            range: t,
            v0: a,
            v1: b,
            tol: tol::LINEAR,
        });
        self.pieces.len() - 1
    }
}

/// The sweep of the cut body `cut` under the turn `motion`, or `Unsupported` where the drawing
/// asks for what is not built yet.
pub(super) fn sweep(cut: &Cut, motion: &Steady, feature: &str) -> Result<Brep> {
    let Steady::Turn {
        origin,
        axis,
        angle: span,
        ..
    } = *motion
    else {
        return unsupported("only a turn is read off its orbits");
    };
    let frame = Axis::new(origin, axis)?;
    let cells = Cells::new(cut, frame)?;
    let turn = Transform::rotation(origin, axis, span);
    let swept: Vec<Option<Swept>> = (0..cells.regions.len())
        .map(|r| {
            let o = cells
                .middle(r)
                .ok_or_else(|| Error::Numerical("a cell of a turn's drawing is empty".into()))?;
            merged(&cells.hits(cut, o), span)
        })
        .collect::<Result<_>>()?;
    // The cell on each side of each link: on its left as it runs from `a` to `b`, and its right.
    let mut sides: Vec<[Option<usize>; 2]> = vec![[None, None]; cells.links.len()];
    for (r, region) in cells.regions.iter().enumerate() {
        for &(k, fwd) in region.loops.iter().flatten() {
            sides[k][usize::from(!fwd)] = Some(r);
        }
    }
    let on_part = |part: usize, k: usize| {
        let pieces: BTreeSet<usize> = cut.parts[part]
            .loops
            .iter()
            .flatten()
            .map(|t| t.piece)
            .collect();
        cells.gens[k].iter().any(|g| pieces.contains(&g.piece))
    };
    let mut out = Out {
        pool: Vec::new(),
        pieces: Vec::new(),
        faces: Vec::new(),
        made: HashMap::new(),
        near: 1e-7 * cells.size,
    };
    // The lift each side of each link ends on, low and high, where the cells there are of
    // different faces.
    let mut lifts: HashMap<(usize, usize, bool), Lift> = HashMap::new();
    for high in [false, true] {
        let role = |s: &Swept| if high { s.hi } else { s.lo };
        // Cells whose ends lie on one part, with no curve of its boundary between them, are one
        // face.
        let mut group: Vec<usize> = (0..cells.regions.len()).collect();
        for (k, side) in sides.iter().enumerate() {
            let [Some(l), Some(r)] = *side else { continue };
            let (Some(a), Some(b)) = (swept[l], swept[r]) else {
                continue;
            };
            if role(&a).part == role(&b).part && !on_part(role(&a).part, k) {
                let (x, y) = (find(&mut group, l), find(&mut group, r));
                group[x] = y;
            }
        }
        let roots: BTreeSet<usize> = (0..cells.regions.len())
            .filter(|&r| swept[r].is_some())
            .map(|r| find(&mut group, r))
            .collect();
        for root in roots {
            let member = |r: usize, group: &mut Vec<usize>| find(group, r) == root;
            let part = role(&swept[root].expect("a face's cells are swept")).part;
            // The links with the face on one side only, as a graph of their own: its regions that
            // hold a cell of the face are the face.
            let mut edge: Vec<usize> = Vec::new();
            for (k, side) in sides.iter().enumerate() {
                let within: Vec<bool> = side
                    .iter()
                    .map(|s| s.is_some_and(|r| member(r, &mut group)))
                    .collect();
                if within[0] != within[1] || (within[0] && on_part(part, k)) {
                    edge.push(k);
                }
            }
            let sub: Vec<Link> = edge
                .iter()
                .map(|&k| Link {
                    a: cells.links[k].a,
                    b: cells.links[k].b,
                    pts: cells.links[k].pts.clone(),
                })
                .collect();
            let regions = graph::regions(&cells.nodes, &sub, 1e-3 * cells.size)?;
            let mids: Vec<Vec2> = (0..cells.regions.len())
                .filter(|&r| member(r, &mut group))
                .filter_map(|r| cells.middle(r))
                .collect();
            for region in &regions {
                if !mids.iter().any(|&m| graph::holds(&sub, region, m)) {
                    continue;
                }
                let mut loops = Vec::new();
                for lp in &region.loops {
                    let mut trims = Vec::new();
                    for &(i, fwd) in lp {
                        let k = edge[i];
                        // The face's cell beside the link here.
                        let side = usize::from(!fwd);
                        let lift = lifted(cut, &cells, k, side, part, high, span)?;
                        lifts.insert((k, side, high), lift);
                        trims.push(trim_on_part(cut, &mut out, &turn, part, lift, fwd)?);
                    }
                    loops.push(trims);
                }
                let p = &cut.parts[part];
                let mut face = OutFace {
                    surface: if high {
                        p.surface.transformed(&turn)
                    } else {
                        p.surface.clone()
                    },
                    reversed: p.reversed,
                    names: p.names.clone(),
                    loops,
                    memo: None,
                    side: Some(usize::from(high)),
                };
                // Seen along the orbits, a face the body enters by keeps the drawing's turn sense
                // and one it leaves by has it mirrored.
                if high {
                    for l in &mut face.loops {
                        l.reverse();
                        for t in l.iter_mut() {
                            *t = t.reversed();
                        }
                    }
                }
                out.faces.push(face);
            }
        }
    }
    walls(
        cut, &cells, &sides, &swept, &lifts, span, &turn, feature, &mut out,
    )?;
    let vtol = vec![tol::LINEAR; out.pool.len()];
    let mut b = sew::assemble(out.faces, &out.pool, &vtol, &out.pieces)?;
    b.transform(&motion.pose0());
    Ok(b)
}

fn unsupported<T>(what: &str) -> Result<T> {
    Err(Error::Unsupported(format!(
        "a turn read off its orbits: {what}"
    )))
}

fn find(p: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while p[r] != r {
        r = p[r];
    }
    p[i] = r;
    r
}

/// The one interval the body sweeps along an orbit that meets it at `hits`, or `None` where the
/// orbit misses it.
fn merged(hits: &[Hit], span: f64) -> Result<Option<Swept>> {
    if hits.is_empty() {
        return Ok(None);
    }
    let alternate = hits.len().is_multiple_of(2)
        && hits
            .iter()
            .enumerate()
            .all(|(i, h)| h.enters == (i % 2 == 0));
    if !alternate {
        return unsupported("an orbit that does not go in and out of the body in turn");
    }
    for w in hits.chunks(2).collect::<Vec<_>>().windows(2) {
        let gap = w[1][0].phase - w[0][1].phase;
        if gap >= span - 1e-6 {
            return unsupported("a gap in the body along an orbit as long as the turn");
        }
    }
    let (lo, hi) = (hits[0], hits[hits.len() - 1]);
    if hi.phase + span - lo.phase >= std::f64::consts::TAU - 1e-6 {
        return unsupported("a sweep that closes round the axis");
    }
    Ok(Some(Swept { lo, hi }))
}

/// The curve over link `k` that the face on `part` of the cell on `side` of it ends on: the
/// stretch of the part's own boundary drawn there, the one nearest where orbits beside the link
/// meet the part.
fn lifted(
    cut: &Cut,
    cells: &Cells,
    k: usize,
    side: usize,
    part: usize,
    high: bool,
    span: f64,
) -> Result<Lift> {
    let own: Vec<(Gen, &Trim)> = cells.gens[k]
        .iter()
        .flat_map(|&g| {
            cut.parts[part]
                .loops
                .iter()
                .flatten()
                .filter(move |t| t.piece == g.piece)
                .map(move |t| (g, t))
        })
        .collect();
    if own.is_empty() {
        return unsupported("a face cut along the shadow of another's outline");
    }
    let pick = if own.len() == 1 {
        own[0].0
    } else {
        // An orbit a little into the cell from the link's middle meets the part beside the
        // right stretch.
        let pts = &cells.links[k].pts;
        let m = pts.len() / 2;
        let (a, b) = (pts[m - 1], pts[m]);
        let into = (b - a).perp() * if side == 0 { 1.0 } else { -1.0 };
        let mut found = None;
        for f in [1e-3, 1e-4, 1e-5] {
            let o = a.lerp(b, 0.5) + into * (f * cells.size / into.norm());
            if let Ok(Some(s)) = merged(&cells.hits(cut, o), span) {
                let hit = if high { s.hi } else { s.lo };
                if hit.part == part {
                    found = Some(hit);
                    break;
                }
            }
        }
        let Some(hit) = found else {
            return Err(Error::Numerical(
                "no orbit beside a link of a turn's drawing meets the face it bounds".into(),
            ));
        };
        let s = &cut.parts[part].surface;
        let at = s.eval(hit.uv.x, hit.uv.y);
        own.iter()
            .min_by(|x, y| {
                let d = |g: &Gen| {
                    let c = cut.pieces[g.piece].curve.as_ref().expect("a drawn piece");
                    c.eval(0.5 * (g.t.0 + g.t.1)).dist(at)
                };
                d(&x.0).total_cmp(&d(&y.0))
            })
            .expect("checked")
            .0
    };
    Ok(Lift {
        piece: pick.piece,
        t: pick.t,
        end: high,
    })
}

/// The piece of the result `lift` is, made once.
fn piece_of(cut: &Cut, out: &mut Out, turn: &Transform, lift: Lift) -> (usize, Rc<Curve>) {
    let (lo, hi) = (lift.t.0.min(lift.t.1), lift.t.0.max(lift.t.1));
    let key = (lift.piece, lo.to_bits(), hi.to_bits(), lift.end);
    let curve = cut.pieces[lift.piece]
        .curve
        .as_ref()
        .expect("a drawn piece");
    if let Some(&k) = out.made.get(&key) {
        return (k, out.pieces[k].curve.clone().expect("made with a curve"));
    }
    let curve = Rc::new(if lift.end {
        curve.transformed(turn)
    } else {
        (**curve).clone()
    });
    let k = out.piece(curve.clone(), (lo, hi));
    out.made.insert(key, k);
    (k, curve)
}

/// The trim of `part`'s own boundary along `lift`, run from the link's `a` to its `b` if `fwd`.
fn trim_on_part(
    cut: &Cut,
    out: &mut Out,
    turn: &Transform,
    part: usize,
    lift: Lift,
    fwd: bool,
) -> Result<Trim> {
    let p = &cut.parts[part];
    let (piece, curve) = piece_of(cut, out, turn, lift);
    let (t0, t1) = if fwd { lift.t } else { (lift.t.1, lift.t.0) };
    // The part's own trim of the piece; of a seam's two, the one that runs the same way.
    let mine = p
        .loops
        .iter()
        .flatten()
        .filter(|t| t.piece == lift.piece)
        .max_by_key(|t| (t.t1 >= t.t0) == (t1 >= t0))
        .ok_or_else(|| Error::Numerical("a lifted curve is not on its face".into()))?;
    let uv = |t: f64| {
        let f = ((t - mine.t0) / (mine.t1 - mine.t0)).clamp(0.0, 1.0);
        mine.uv(&p.surface, t, mine.uv0.lerp(mine.uv1, f))
    };
    let pg = &out.pieces[piece];
    let forward = t1 >= t0;
    Ok(Trim {
        piece,
        forward,
        curve: Some(curve),
        pole: out.pool[pg.v0],
        t0,
        t1,
        map: mine.map.clone(),
        uv0: uv(t0),
        uv1: uv(t1),
        v0: if forward { pg.v0 } else { pg.v1 },
        v1: if forward { pg.v1 } else { pg.v0 },
    })
}

/// A wall to build: over link `k`, from the lift `low` up to the lift `high`, with the solid on
/// the link's left or its right.
struct Wall {
    k: usize,
    low: Lift,
    high: Lift,
    solid_left: bool,
}

/// The walls over the links: where the two cells' swept intervals differ, the orbit arcs between
/// the ends that differ.
#[allow(clippy::too_many_arguments)]
fn walls(
    cut: &Cut,
    cells: &Cells,
    sides: &[[Option<usize>; 2]],
    swept: &[Option<Swept>],
    lifts: &HashMap<(usize, usize, bool), Lift>,
    span: f64,
    turn: &Transform,
    feature: &str,
    out: &mut Out,
) -> Result<()> {
    let axis = &cells.axis;
    let point = |l: &Lift, t: f64| {
        let p = cut.pieces[l.piece]
            .curve
            .as_ref()
            .expect("a drawn piece")
            .eval(t);
        if l.end { turn.point(p) } else { p }
    };
    let mut list: Vec<Wall> = Vec::new();
    for k in 0..cells.links.len() {
        // Each side's ends that are curves of the result here, with the angle of each at the
        // link's middle; an end both sides share is no wall's.
        let mut events: Vec<(f64, Lift, usize)> = Vec::new();
        for side in 0..2 {
            for high in [false, true] {
                if let Some(&l) = lifts.get(&(k, side, high)) {
                    let mid = point(&l, 0.5 * (l.t.0 + l.t.1));
                    events.push((0.0, l, side));
                    let near = swept[sides[k][side].expect("a lift has a cell")]
                        .map_or(0.0, |s| if high { s.hi.phase + span } else { s.lo.phase });
                    events.last_mut().expect("pushed").0 = axis.drawn(mid, near).1;
                }
            }
        }
        let mut kept: Vec<(f64, Lift, usize)> = Vec::new();
        for e in &events {
            let twin = events.iter().any(|o| o.2 != e.2 && o.1 == e.1);
            if !twin {
                kept.push(*e);
            }
        }
        kept.sort_by(|a, b| a.0.total_cmp(&b.0));
        if !kept.len().is_multiple_of(2) {
            return Err(Error::Numerical(
                "the swept intervals either side of a link differ at an odd number of ends".into(),
            ));
        }
        for pair in kept.chunks(2) {
            // The wall stands between the two ends; the solid is on the side whose interval
            // holds the angles between them.
            let between = 0.5 * (pair[0].0 + pair[1].0);
            let holds = |side: usize| {
                sides[k][side].and_then(|r| swept[r]).is_some_and(|s| {
                    let lo = events
                        .iter()
                        .filter(|e| e.2 == side)
                        .map(|e| e.0)
                        .fold(f64::INFINITY, f64::min);
                    let hi = events
                        .iter()
                        .filter(|e| e.2 == side)
                        .map(|e| e.0)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let _ = s;
                    between > lo && between < hi
                })
            };
            let (left, right) = (holds(0), holds(1));
            if left == right {
                return Err(Error::Numerical(
                    "a wall of a turn's sweep has the solid on both sides, or neither".into(),
                ));
            }
            list.push(Wall {
                k,
                low: pair[0].1,
                high: pair[1].1,
                solid_left: left,
            });
        }
    }
    // Each wall's corners, and at each node every corner there in order of angle: the orbit arcs
    // between neighbours are the walls' sides.
    let mut at_node: HashMap<usize, Vec<(f64, usize)>> = HashMap::new();
    let mut corners: Vec<[[(f64, usize); 2]; 2]> = Vec::new();
    for w in &list {
        let link = &cells.links[w.k];
        let mut quad = [[(0.0, 0usize); 2]; 2];
        for (e, node) in [link.a, link.b].into_iter().enumerate() {
            let mut near = 0.0;
            for (h, l) in [w.low, w.high].into_iter().enumerate() {
                let p = point(&l, if e == 0 { l.t.0 } else { l.t.1 });
                let phase = axis.drawn(p, near).1;
                // The high corner is the same turn round or the next, never behind the low.
                let phase = if h == 1 && phase < near - 1e-9 {
                    phase + std::f64::consts::TAU
                } else {
                    phase
                };
                near = phase;
                let v = out.vertex(p);
                quad[e][h] = (phase, v);
                let seen = at_node.entry(node).or_default();
                if !seen.iter().any(|x| x.1 == v) {
                    seen.push((phase, v));
                }
            }
        }
        corners.push(quad);
    }
    for seen in at_node.values_mut() {
        seen.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let mut arcs: HashMap<(usize, usize), usize> = HashMap::new();
    let names: BTreeSet<FaceName> = [FaceName::new(feature, FaceName::SIDE)].into();
    for (w, quad) in list.iter().zip(&corners) {
        let link = &cells.links[w.k];
        let g = cells.gens[w.k][0];
        let base = cut.pieces[g.piece].curve.as_ref().expect("a drawn piece");
        // The surface the link's curve sweeps, from a little before the wall's lowest angle to a
        // little past its highest, each read from the curve's own.
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for n in 0..=8 {
            let f = f64::from(n) / 8.0;
            let on = |t: (f64, f64)| t.0 + (t.1 - t.0) * f;
            let own = axis.drawn(base.eval(on(g.t)), 0.0).1;
            let low = axis.drawn(point(&w.low, on(w.low.t)), own).1;
            let mut high = axis.drawn(point(&w.high, on(w.high.t)), low).1;
            if high < low - 1e-9 {
                high += std::f64::consts::TAU;
            }
            lo = lo.min(low - own);
            hi = hi.max(high - own);
        }
        let pad = 0.05 * (hi - lo).max(1e-3);
        let from = Transform::rotation(axis.frame.origin, axis.frame.z, lo - pad);
        let sweep = Steady::Turn {
            origin: axis.frame.origin,
            axis: axis.frame.z,
            from: 0.0,
            angle: hi - lo + 2.0 * pad,
        };
        let range = (g.t.0.min(g.t.1), g.t.0.max(g.t.1));
        let surface = sweep
            .sheet(&base.transformed(&from), cut.whole[g.piece], range)?
            .surface;
        // Round the wall: along the low lift, up the orbit at `b`, back along the high lift, down
        // the orbit at `a`.
        let mut run: Vec<(usize, Rc<Curve>, f64, f64)> = Vec::new();
        let (kl, cl) = piece_of(cut, out, turn, w.low);
        run.push((kl, cl, w.low.t.0, w.low.t.1));
        let mut side = |node: usize,
                        lo: (f64, usize),
                        hi: (f64, usize),
                        up: bool,
                        out: &mut Out,
                        run: &mut Vec<(usize, Rc<Curve>, f64, f64)>| {
            let seen = &at_node[&node];
            let (i, j) = (
                seen.iter()
                    .position(|x| x.1 == lo.1)
                    .expect("a corner is at its node"),
                seen.iter()
                    .position(|x| x.1 == hi.1)
                    .expect("a corner is at its node"),
            );
            let orbit = Rc::new(axis.orbit(cells.nodes[node]));
            let mut steps: Vec<(usize, Rc<Curve>, f64, f64)> = Vec::new();
            for n in i..j {
                let (a, b) = (seen[n], seen[n + 1]);
                let piece = *arcs.entry((a.1, b.1)).or_insert_with(|| {
                    out.pieces.push(PieceGeom {
                        curve: Some(orbit.clone()),
                        range: (a.0, b.0),
                        v0: a.1,
                        v1: b.1,
                        tol: tol::LINEAR,
                    });
                    out.pieces.len() - 1
                });
                let c = out.pieces[piece].curve.clone().expect("an arc");
                steps.push((piece, c, a.0, b.0));
            }
            if !up {
                steps.reverse();
                for s in &mut steps {
                    (s.2, s.3) = (s.3, s.2);
                }
            }
            run.extend(steps);
        };
        side(link.b, quad[1][0], quad[1][1], true, out, &mut run);
        let (kh, ch) = piece_of(cut, out, turn, w.high);
        run.push((kh, ch, w.high.t.1, w.high.t.0));
        side(link.a, quad[0][0], quad[0][1], false, out, &mut run);
        // Each trim's parameters carried on from the last one's end, round the loop.
        let start = run[0].1.eval(run[0].2);
        let mut uv = surface.project(start);
        let mut loop_: Vec<Trim> = Vec::new();
        for (piece, curve, t0, t1) in run {
            let uv0 = uv;
            for n in 1..=16 {
                uv = surface.project_near(curve.eval(t0 + (t1 - t0) * f64::from(n) / 16.0), uv);
            }
            let pg = &out.pieces[piece];
            let forward = (t1 >= t0) == (pg.range.1 >= pg.range.0);
            loop_.push(Trim {
                piece,
                forward,
                curve: Some(curve),
                pole: out.pool[pg.v0],
                t0,
                t1,
                map: UvMap::Projected,
                uv0,
                uv1: uv,
                v0: if forward { pg.v0 } else { pg.v1 },
                v1: if forward { pg.v1 } else { pg.v0 },
            });
        }
        // As built the loop turns about the direction to the link's left; the solid is on one
        // side, and the face looks away from it.
        let mid = point(&w.low, 0.5 * (w.low.t.0 + w.low.t.1));
        let at = surface.project(mid);
        let on = surface.eval(at.x, at.y);
        let along = {
            let l = &w.low;
            point(l, l.t.1) - point(l, l.t.0)
        };
        let leftward = along.cross(axis.velocity(on));
        let outward = if w.solid_left { -leftward } else { leftward };
        let mut face = OutFace {
            surface: surface.clone(),
            reversed: false,
            names: names.clone(),
            loops: vec![loop_],
            memo: None,
            side: None,
        };
        if w.solid_left {
            for l in &mut face.loops {
                l.reverse();
                for t in l.iter_mut() {
                    *t = t.reversed();
                }
            }
        }
        face.reversed = surface.normal(at.x, at.y).dot(outward) < 0.0;
        out.faces.push(face);
    }
    Ok(())
}
