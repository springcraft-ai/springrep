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

use super::cells::{Axis, Cells, Gen, Hit, Meet, crossings};
use super::cut::Cut;
use super::grazing::fit_exact;
use super::moving::graph::{self, Link};
use super::sheet::Steady;
use crate::{
    Error, Result,
    boolean::{
        csi,
        domain::{Trim, UvMap, Where},
        sew::{self, OutFace, PieceGeom},
        ssi::{self, Ssi},
    },
    geom::{Curve, Line},
    math::{Point3, Transform, Vec2},
    tol,
    topo::{Brep, FaceName},
};

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
    /// The curves cut across faces along the shadows of other faces' outlines, numbered on from
    /// the cut body's pieces, each with its parameters on its face at its two ends.
    shadows: Vec<(Rc<Curve>, (Vec2, Vec2))>,
    near: f64,
}

/// The curve of a lift's piece where the turn starts: a piece of the cut body, or a shadow.
fn source(cells: &Cells, out: &Out, piece: usize) -> Rc<Curve> {
    match piece.checked_sub(cells.pieces.len()) {
        Some(k) => out.shadows[k].0.clone(),
        None => cells.pieces[piece].curve.clone().expect("a drawn piece"),
    }
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
    let turn = Transform::rotation(origin, axis, span);
    let back = Transform::rotation(origin, axis, -span);
    let stacked = |cells: &Cells| -> Result<Vec<Vec<Hit>>> {
        (0..cells.regions.len())
            .map(|r| {
                let o = cells.middle(r).ok_or_else(|| {
                    Error::Numerical("a cell of a turn's drawing is empty".into())
                })?;
                Ok(cells.hits(cut, o))
            })
            .collect()
    };
    let mut cells = Cells::new(cut, frame, Vec::new())?;
    // The drawing has a cell for every orbit that meets the body only if the axis passes beside
    // the body, and a line passes through a bounded body only by crossing its surface.
    let line = Curve::Line(Line { origin, dir: axis });
    let reach = cut.pool.iter().fold(0.0, |r: f64, p| r.max(p.dist(origin))) + cells.size;
    let through = cells.domains.iter().any(|d| {
        let hits = csi::curve_surface(&line, (-reach, reach), &d.surface);
        hits.on_surface
            || hits
                .ts
                .iter()
                .any(|&t| !matches!(d.locate(line.eval(t)), Where::Outside))
    });
    if through {
        return unsupported("a turn about a line through the body");
    }
    let mut stacks = stacked(&cells)?;
    // A gap along an orbit is closed by the turn or it is not, and which can change inside a
    // cell: across the curve where the end of the turn meets its start. A gap is shorter than the
    // whole angle the body takes up about the axis, the spread of its edges and characteristic
    // curves, so a turn of at least that closes every gap; a shorter one has those curves drawn
    // too, and then each cell's gaps are closed or open all over it.
    let mut spread = (f64::INFINITY, f64::NEG_INFINITY);
    for (k, pg) in cut.pieces.iter().enumerate() {
        if pg.curve.is_none() {
            continue;
        }
        for n in 0..=64 {
            let t = pg.range.0 + (pg.range.1 - pg.range.0) * f64::from(n) / 64.0;
            let angle = cells.axis.drawn(cells.point(cut, k, t), 0.0).1;
            spread = (spread.0.min(angle), spread.1.max(angle));
        }
    }
    let gapped = stacks.iter().any(|hits| hits.len() > 2);
    if gapped && span < spread.1 - spread.0 + 1e-2 {
        let meets = meetings(cut, &cells, &turn)?;
        cells = Cells::new(cut, frame, meets)?;
        stacks = stacked(&cells)?;
    }
    let cells = cells;
    let groups: Vec<Vec<(usize, usize)>> = stacks
        .iter()
        .map(|hits| grouped(hits, span))
        .collect::<Result<_>>()?;
    let overs: Vec<Over> = (0..cells.links.len())
        .map(|k| over(cut, &cells, k, &back))
        .collect();
    // The cell on each side of each link: on its left as it runs from `a` to `b`, and its right.
    let mut sides: Vec<[Option<usize>; 2]> = vec![[None, None]; cells.links.len()];
    for (r, region) in cells.regions.iter().enumerate() {
        for &(k, fwd) in region.loops.iter().flatten() {
            sides[k][usize::from(!fwd)] = Some(r);
        }
    }
    // What each face of each cell's order comes to at each of the cell's links.
    let mut at_link: HashMap<(usize, usize), Vec<(Beside, f64)>> = HashMap::new();
    for (k, side) in sides.iter().enumerate() {
        for (i, r) in side.iter().enumerate() {
            if let Some(r) = *r
                && !stacks[r].is_empty()
            {
                at_link.insert((k, i), align(cut, &overs[k], &stacks[r])?);
            }
        }
    }
    // Each end of each cell's swept intervals: the place in the cell's order, and whether it is
    // a high end.
    let ends_of = |r: usize| -> Vec<(usize, bool)> {
        groups[r]
            .iter()
            .flat_map(|&(lo, hi)| [(lo, false), (hi, true)])
            .collect()
    };
    let mut out = Out {
        pool: Vec::new(),
        pieces: Vec::new(),
        faces: Vec::new(),
        made: HashMap::new(),
        shadows: Vec::new(),
        near: 1e-7 * cells.size,
    };
    // Ends of cells either side of a link that are one meeting of the link's own orbit with a
    // face are ends on one face of the result.
    let first: Vec<usize> = stacks
        .iter()
        .scan(0, |sum, hits| {
            let at = *sum;
            *sum += hits.len();
            Some(at)
        })
        .collect();
    let mut face_of: Vec<usize> = (0..stacks.iter().map(Vec::len).sum()).collect();
    for (k, side) in sides.iter().enumerate() {
        let [Some(l), Some(r)] = *side else { continue };
        for (jl, high) in ends_of(l) {
            let Beside::Through(i) = at_link[&(k, 0)][jl].0 else {
                continue;
            };
            for (jr, other) in ends_of(r) {
                if other == high && at_link[&(k, 1)][jr].0 == Beside::Through(i) {
                    let (x, y) = (
                        find(&mut face_of, first[l] + jl),
                        find(&mut face_of, first[r] + jr),
                    );
                    face_of[x] = y;
                }
            }
        }
    }
    let mut members: HashMap<usize, Vec<(usize, usize, bool)>> = HashMap::new();
    for (r, &at) in first.iter().enumerate().take(cells.regions.len()) {
        for (j, high) in ends_of(r) {
            let root = find(&mut face_of, at + j);
            members.entry(root).or_default().push((r, j, high));
        }
    }
    let mut roots: Vec<usize> = members.keys().copied().collect();
    roots.sort_unstable();
    // The lift each end ends on at each link where its face stops, with its angle there, and
    // the angles between which each side of each link is swept.
    let mut events: HashMap<usize, Vec<(f64, Lift, usize)>> = HashMap::new();
    let mut spans: HashMap<(usize, usize), Vec<(f64, f64)>> = HashMap::new();
    for (&(k, side), list) in &at_link {
        let r = sides[k][side].expect("an aligned side has a cell");
        let swept = groups[r]
            .iter()
            .map(|&(lo, hi)| (list[lo].1, list[hi].1 + span))
            .collect();
        spans.insert((k, side), swept);
    }
    for root in roots {
        let list = &members[&root];
        let (r0, j0, high) = list[0];
        let part = stacks[r0][j0].part;
        let place: HashMap<usize, usize> = list.iter().map(|&(r, j, _)| (r, j)).collect();
        // The links with the face on one side only, or on both but not carried across, as a
        // graph of their own: its regions that hold a cell of the face are the face.
        let mut edge: Vec<usize> = Vec::new();
        for (k, side) in sides.iter().enumerate() {
            let on: Vec<Option<usize>> = side
                .iter()
                .map(|s| s.and_then(|r| place.get(&r).copied()))
                .collect();
            let carried = match (on[0], on[1]) {
                (Some(a), Some(b)) => {
                    let (x, y) = (at_link[&(k, 0)][a].0, at_link[&(k, 1)][b].0);
                    x == y && matches!(x, Beside::Through(_))
                }
                _ => false,
            };
            if (on[0].is_some() || on[1].is_some()) && !carried {
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
        let mids: Vec<Vec2> = place.keys().filter_map(|&r| cells.middle(r)).collect();
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
                    let j = sides[k][side]
                        .and_then(|r| place.get(&r).copied())
                        .ok_or_else(|| {
                            Error::Numerical(
                                "a face of a turn's sweep has no cell beside its edge".into(),
                            )
                        })?;
                    let (end, angle) = at_link[&(k, side)][j];
                    let lift = match end {
                        Beside::Own(i) => {
                            let g = cells.gens[k][i];
                            Lift {
                                piece: g.piece,
                                t: g.t,
                                // A curve where the turn's end meets its start is drawn where it
                                // lies, for the face at the end as for the face at the start.
                                end: high && g.piece < cells.first_meet,
                            }
                        }
                        Beside::Through(i) => {
                            let h = &overs[k].through[i];
                            let seed = cut.parts[h.part].surface.eval(h.uv.x, h.uv.y);
                            shadow(cut, &cells, &mut out, k, part, high, (overs[k].frac, seed))?
                        }
                    };
                    events.entry(k).or_default().push((
                        angle + if high { span } else { 0.0 },
                        lift,
                        side,
                    ));
                    let turns = (&turn, &back);
                    trims.push(trim_on_part(
                        cut, &cells, &mut out, turns, part, lift, fwd, high, &overs[k],
                    )?);
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
    walls(&cells, &events, &spans, &turn, feature, &mut out)?;
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

/// The intervals the body sweeps along an orbit that meets it at `hits`, each as the places in
/// `hits` of its low and high ends: each of the body's own intervals a turn longer, and joined
/// to the next where the gap between them is shorter than the turn.
fn grouped(hits: &[Hit], span: f64) -> Result<Vec<(usize, usize)>> {
    if hits.is_empty() {
        return Ok(Vec::new());
    }
    let alternate = hits.len().is_multiple_of(2)
        && hits
            .iter()
            .enumerate()
            .all(|(i, h)| h.enters == (i % 2 == 0));
    if !alternate {
        return unsupported("an orbit that does not go in and out of the body in turn");
    }
    let mut out = vec![(0, 1)];
    for i in (2..hits.len()).step_by(2) {
        let gap = hits[i].phase - hits[i - 1].phase;
        if (gap - span).abs() <= 1e-9 {
            return Err(Error::Numerical(
                "a cell's middle lies where the end of a turn meets its start".into(),
            ));
        }
        if gap < span {
            out.last_mut().expect("one interval").1 = i + 1;
        } else {
            out.push((i, i + 1));
        }
    }
    if hits[hits.len() - 1].phase + span - hits[0].phase >= std::f64::consts::TAU - 1e-6 {
        return unsupported("a sweep that closes round the axis");
    }
    Ok(out)
}

/// What the end of a cell's swept interval comes to at a link of the cell.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Beside {
    /// The curve of the link's that is drawn there, by its place in the link's list: the face the
    /// end lies on folds or stops at the link.
    Own(usize),
    /// A place where the link's own orbits pass through the face, by its place among those: the
    /// face runs on under the link.
    Through(usize),
}

/// The parameter of `g`'s stretch whose point is drawn nearest `o`, by golden-section search.
fn drawn_at(cut: &Cut, cells: &Cells, g: &Gen, o: Vec2) -> f64 {
    let off = |t: f64| {
        cells
            .axis
            .drawn(cells.point(cut, g.piece, t), 0.0)
            .0
            .dist(o)
    };
    // The nearest of a few samples brackets it: a drawing may turn back on itself.
    let n = 32;
    let at = |i: usize| g.t.0 + (g.t.1 - g.t.0) * i as f64 / n as f64;
    let m = (0..=n)
        .min_by(|&x, &y| off(at(x)).total_cmp(&off(at(y))))
        .expect("samples");
    let (mut lo, mut hi) = (at(m.saturating_sub(1)), at((m + 1).min(n)));
    for _ in 0..80 {
        let (x, y) = (lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo));
        if off(x) < off(y) {
            hi = y;
        } else {
            lo = x;
        }
    }
    0.5 * (lo + hi)
}

/// A face beside one of a link's curves: on the link's own orbit it is met at the curve, touched
/// or run through at its edge, and a cell's orbits may meet it or not.
#[derive(Clone, Copy, Debug)]
struct Item {
    /// The curve's place in the link's list.
    curve: usize,
    part: usize,
    enters: bool,
    at: Point3,
    angle: f64,
    /// The face's parameters there, where the orbit passes through it at the curve.
    uv: Option<Vec2>,
}

/// A link's own orbit, at the link's middle: how far along the link's first stretch that is, the
/// faces beside the link's curves where those lie on it, and its meetings with the body away from
/// those, in order of angle. Everywhere but at the link's curves it passes through the faces as
/// the orbits of the cells either side do.
struct Over {
    frac: f64,
    items: Vec<Item>,
    through: Vec<Hit>,
}

fn over(cut: &Cut, cells: &Cells, k: usize, back: &Transform) -> Over {
    let m = cells.links[k].pts.len() / 2;
    let frac = cells.fracs[k][m];
    let g = cells.gens[k][0];
    let first = cells.point(cut, g.piece, g.t.0 + (g.t.1 - g.t.0) * frac);
    let (o, near) = cells.axis.drawn(first, 0.0);
    let hits = cells.hits(cut, o);
    let mut items: Vec<Item> = Vec::new();
    for (i, g) in cells.gens[k].iter().enumerate() {
        let on = match i {
            0 => first,
            _ => cells.point(cut, g.piece, drawn_at(cut, cells, g, o)),
        };
        // A piece of the cut body lies on the parts it bounds; a curve where the turn's end
        // meets its start lies on the face the orbits come back in by, and a turn back from it
        // on the face they left by.
        let mut sides: Vec<(usize, Point3)> = Vec::new();
        match g.piece.checked_sub(cells.first_meet) {
            None => {
                for (part, p) in cut.parts.iter().enumerate() {
                    if p.loops.iter().flatten().any(|t| t.piece == g.piece) {
                        sides.push((part, on));
                    }
                }
            }
            Some(n) => {
                let meet = &cells.meets[n];
                for (part, p) in cut.parts.iter().enumerate() {
                    if p.face == meet.enter && !p.front {
                        sides.push((part, on));
                    }
                    if p.face == meet.exit && p.front {
                        sides.push((part, back.point(on)));
                    }
                }
            }
        }
        for (part, at) in sides {
            let uv = hits
                .iter()
                .filter(|h| h.part == part)
                .find(|h| {
                    let q = cut.parts[part].surface.eval(h.uv.x, h.uv.y);
                    q.dist(at) <= 1e-5 * cells.size
                })
                .map(|h| h.uv);
            items.push(Item {
                curve: i,
                part,
                enters: !cut.parts[part].front,
                at,
                angle: cells.axis.drawn(at, near).1,
                uv,
            });
        }
    }
    let through = hits
        .into_iter()
        .filter(|h| {
            let q = cut.parts[h.part].surface.eval(h.uv.x, h.uv.y);
            !items
                .iter()
                .any(|it| it.part == h.part && it.at.dist(q) <= 1e-5 * cells.size)
        })
        .collect();
    Over {
        frac,
        items,
        through,
    }
}

/// What each face of a cell's order `stack` comes to at a link of the cell, and its angle
/// there. The link's own orbit meets the faces in the order `over.through`, and the cell's order
/// is that one with some of the faces beside the link's curves put in, each where its curve
/// lies on the orbit. The cell's order is matched to that, face for face in order of angle; a
/// face put in is the link's curve there, and any other is the orbit's own meeting.
fn align(cut: &Cut, over: &Over, stack: &[Hit]) -> Result<Vec<(Beside, f64)>> {
    let _ = cut;
    let slack = 1e-6;
    fn fit(
        over: &Over,
        stack: &[Hit],
        (s, t): (usize, usize),
        used: &mut Vec<bool>,
        angle: f64,
        got: &mut Vec<(Beside, f64)>,
        slack: f64,
    ) -> bool {
        if s == stack.len() {
            return t == over.through.len();
        }
        let want = &stack[s];
        if let Some(h) = over.through.get(t)
            && h.part == want.part
            && h.enters == want.enters
        {
            got.push((Beside::Through(t), h.phase));
            if fit(over, stack, (s + 1, t + 1), used, h.phase, got, slack) {
                return true;
            }
            got.pop();
        }
        for i in 0..over.items.len() {
            let it = over.items[i];
            let before_next = over
                .through
                .get(t)
                .is_none_or(|h| it.angle <= h.phase + slack);
            if used[i]
                || it.part != want.part
                || it.enters != want.enters
                || it.angle < angle - slack
                || !before_next
            {
                continue;
            }
            used[i] = true;
            got.push((Beside::Own(it.curve), it.angle));
            if fit(over, stack, (s + 1, t), used, it.angle, got, slack) {
                return true;
            }
            got.pop();
            used[i] = false;
        }
        false
    }
    let mut got = Vec::with_capacity(stack.len());
    let mut used = vec![false; over.items.len()];
    if fit(
        over,
        stack,
        (0, 0),
        &mut used,
        f64::NEG_INFINITY,
        &mut got,
        slack,
    ) {
        Ok(got)
    } else {
        Err(Error::Numerical(
            "a cell's faces along its orbits are not its link's".into(),
        ))
    }
}

/// The curves where the body's surface at the end of the turn meets its surface at the start:
/// each face orbits leave the body by, turned, against each face they come into it by. Only a
/// meeting with a closed form is taken, which gives all of it.
fn meetings(cut: &Cut, cells: &Cells, turn: &Transform) -> Result<Vec<Meet>> {
    let mut region = crate::math::Aabb::EMPTY;
    for &p in &cut.pool {
        region.add(p);
        region.add(turn.point(p));
    }
    let pad = crate::math::Vec3::new(cells.size, cells.size, cells.size);
    region.min -= pad;
    region.max += pad;
    // A face the orbits run within, as a plane square to the axis, is met by none.
    let still = |part: usize| {
        let d = &cells.domains[part];
        // At places not on one line of the face's parameters: a part cut off along its face's
        // grazing line can have that line for the diagonal of its window.
        [
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
            (0.5, 0.4),
        ]
        .into_iter()
        .all(|(fu, fv)| {
            let uv = Vec2::new(
                d.lo.x + (d.hi.x - d.lo.x) * fu,
                d.lo.y + (d.hi.y - d.lo.y) * fv,
            );
            let v = cells.axis.velocity(d.surface.eval(uv.x, uv.y));
            d.surface.normal(uv.x, uv.y).dot(v).abs() <= 1e-9 * v.norm().max(1e-300)
        })
    };
    let mut seen: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut out = Vec::new();
    for (q, enter) in cut.parts.iter().enumerate() {
        if enter.front || still(q) {
            continue;
        }
        for exit in cut.parts.iter().filter(|p| p.front) {
            if !seen.insert((enter.face, exit.face)) {
                continue;
            }
            let turned = exit.surface.transformed(turn);
            // A pair that asks for places to trace from has no closed form: traced, a loop no
            // such place lies on is missed, and a gap would be read closed where it is open.
            let mut traced = false;
            let found = ssi::intersect(&enter.surface, &turned, &region, &mut || {
                traced = true;
                Vec::new()
            })?;
            if traced {
                return unsupported(
                    "the end of a turn meeting its start along a curve with no closed form",
                );
            }
            match found {
                Ssi::Coincident => {}
                Ssi::Curves { curves, .. } => {
                    for c in curves {
                        out.push(Meet {
                            curve: Rc::new(c.curve),
                            range: c.range,
                            enter: enter.face,
                            exit: exit.face,
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// The piece of the result `lift` is, made once.
fn piece_of(cells: &Cells, out: &mut Out, turn: &Transform, lift: Lift) -> (usize, Rc<Curve>) {
    let (lo, hi) = (lift.t.0.min(lift.t.1), lift.t.0.max(lift.t.1));
    let key = (lift.piece, lo.to_bits(), hi.to_bits(), lift.end);
    let curve = source(cells, out, lift.piece);
    if let Some(&k) = out.made.get(&key) {
        return (k, out.pieces[k].curve.clone().expect("made with a curve"));
    }
    let curve = Rc::new(if lift.end {
        curve.transformed(turn)
    } else {
        (*curve).clone()
    });
    let k = out.piece(curve.clone(), (lo, hi));
    out.made.insert(key, k);
    (k, curve)
}

/// The trim of `part`'s own boundary along `lift`, run from the link's `a` to its `b` if `fwd`.
#[allow(clippy::too_many_arguments)]
fn trim_on_part(
    cut: &Cut,
    cells: &Cells,
    out: &mut Out,
    (turn, back): (&Transform, &Transform),
    part: usize,
    lift: Lift,
    fwd: bool,
    high: bool,
    over: &Over,
) -> Result<Trim> {
    let p = &cut.parts[part];
    let (piece, curve) = piece_of(cells, out, turn, lift);
    let (t0, t1) = if fwd { lift.t } else { (lift.t.1, lift.t.0) };
    if lift.piece >= cells.first_meet && lift.piece < cells.pieces.len() {
        // A curve where the turn's end meets its start: its parameters on the face are carried
        // out from where the link's own orbit passes through the face at it.
        let item = over
            .items
            .iter()
            .find(|it| it.part == part && it.uv.is_some())
            .ok_or_else(|| {
                Error::Numerical("a face is not met where the turn's end meets its start".into())
            })?;
        let on_face = |t: f64| {
            let q = curve.eval(t);
            if high { back.point(q) } else { q }
        };
        let middle = lift.t.0 + (lift.t.1 - lift.t.0) * over.frac;
        let walk = |to: f64| {
            let mut uv = item.uv.expect("checked");
            for n in 1..=32 {
                uv = p
                    .surface
                    .project_near(on_face(middle + (to - middle) * f64::from(n) / 32.0), uv);
            }
            uv
        };
        let pg = &out.pieces[piece];
        let forward = t1 >= t0;
        let (uv0, uv1) = (walk(t0), walk(t1));
        return Ok(Trim {
            piece,
            forward,
            curve: Some(curve),
            pole: out.pool[pg.v0],
            t0,
            t1,
            map: UvMap::Projected,
            uv0,
            uv1,
            v0: if forward { pg.v0 } else { pg.v1 },
            v1: if forward { pg.v1 } else { pg.v0 },
        });
    }
    if let Some(k) = lift.piece.checked_sub(cells.pieces.len()) {
        let (ua, ub) = out.shadows[k].1;
        let pg = &out.pieces[piece];
        let forward = t1 >= t0;
        return Ok(Trim {
            piece,
            forward,
            curve: Some(curve),
            pole: out.pool[pg.v0],
            t0,
            t1,
            map: UvMap::Projected,
            uv0: if fwd { ua } else { ub },
            uv1: if fwd { ub } else { ua },
            v0: if forward { pg.v0 } else { pg.v1 },
            v1: if forward { pg.v1 } else { pg.v0 },
        });
    }
    // The part's own trim of the piece; of a seam's two, the one that runs the way this one
    // will once its loop is turned as its face needs: a face lies on the same side of its
    // boundary as the part it is cut from.
    let mine = p
        .loops
        .iter()
        .flatten()
        .filter(|t| t.piece == lift.piece)
        .max_by_key(|t| (t.t1 >= t.t0) == ((t1 >= t0) != high))
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
    cells: &Cells,
    ends: &HashMap<usize, Vec<(f64, Lift, usize)>>,
    spans: &HashMap<(usize, usize), Vec<(f64, f64)>>,
    turn: &Transform,
    feature: &str,
    out: &mut Out,
) -> Result<()> {
    let axis = &cells.axis;
    let sources: HashMap<usize, Rc<Curve>> = ends
        .values()
        .flatten()
        .map(|e| (e.1.piece, source(cells, out, e.1.piece)))
        .collect();
    let point = |l: &Lift, t: f64| {
        let p = sources[&l.piece].eval(t);
        if l.end { turn.point(p) } else { p }
    };
    let mut list: Vec<Wall> = Vec::new();
    for k in 0..cells.links.len() {
        // The ends that are curves of the result here, with the angle of each at the link's
        // middle. Two ends on one curve stand no wall between them: an end both sides share, or
        // a high end and a low end met where the turn's end meets its start.
        let mut kept: Vec<(f64, Lift, usize)> = Vec::new();
        for e in ends.get(&k).map_or(&[][..], |v| &v[..]) {
            match kept.iter().position(|o| o.1 == e.1) {
                Some(i) => {
                    kept.remove(i);
                }
                None => kept.push(*e),
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
                spans
                    .get(&(k, side))
                    .is_some_and(|list| list.iter().any(|&(lo, hi)| between > lo && between < hi))
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
        let base = cells.pieces[g.piece].curve.as_ref().expect("a drawn piece");
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
            .sheet(&base.transformed(&from), cells.whole[g.piece], range)?
            .surface;
        // Round the wall: along the low lift, up the orbit at `b`, back along the high lift, down
        // the orbit at `a`.
        let mut run: Vec<(usize, Rc<Curve>, f64, f64)> = Vec::new();
        let (kl, cl) = piece_of(cells, out, turn, w.low);
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
        let (kh, ch) = piece_of(cells, out, turn, w.high);
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
        // Read where the wall's own trim puts the point: a swept surface may pass the same place
        // twice.
        let at = {
            let t = &loop_[0];
            t.uv(&surface, 0.5 * (t.t0 + t.t1), t.uv0.lerp(t.uv1, 0.5))
        };
        let on = surface.eval(at.x, at.y);
        let _ = mid;
        // The low curve's own direction at its middle: a long arc's chord can point against it.
        let along = {
            let l = &w.low;
            let (m, h) = (0.5 * (l.t.0 + l.t.1), 1e-3 * (l.t.1 - l.t.0));
            point(l, m + h) - point(l, m - h)
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

/// The curve across `part` over link `k` where orbits through the link's curve meet the part: the
/// shadow of another face's outline, which the part's face on `side` of the link ends on. Each of
/// its points is where one orbit meets the part's surface, and it is fitted through them.
#[allow(clippy::too_many_arguments)]
fn shadow(
    cut: &Cut,
    cells: &Cells,
    out: &mut Out,
    k: usize,
    part: usize,
    high: bool,
    (middle, seed): (f64, Point3),
) -> Result<Lift> {
    let g = cells.gens[k][0];
    let dom = &cells.domains[part];
    let link = &cells.links[k];
    let on_link = |f: f64| cells.point(cut, g.piece, g.t.0 + (g.t.1 - g.t.0) * f);
    let roots = |f: f64| -> Vec<Point3> {
        let o = cells.axis.drawn(on_link(f), 0.0).0;
        let orbit = cells.axis.orbit(o);
        crossings(&orbit, o.x, &dom.surface)
            .into_iter()
            .map(|t| orbit.eval(t))
            .collect()
    };
    // The meeting nearest `near`. Where a shadow runs out at a fold of the part its orbit's two
    // meetings there close up and, within rounding of the fold, are no crossing at all: the
    // meeting is then where the orbit comes nearest the part, beside the last one found.
    let nearest = |f: f64, near: Point3| -> Point3 {
        if let Some(p) = roots(f)
            .into_iter()
            .min_by(|a, b| a.dist(near).total_cmp(&b.dist(near)))
        {
            return p;
        }
        let o = cells.axis.drawn(on_link(f), 0.0).0;
        let orbit = cells.axis.orbit(o);
        let about = cells.axis.drawn(near, 0.0).1;
        let off = |t: f64| csi::signed_distance(&dom.surface, orbit.eval(t)).abs();
        let (mut lo, mut hi) = (about - 0.05, about + 0.05);
        for _ in 0..80 {
            let (x, y) = (lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo));
            if off(x) < off(y) {
                hi = y;
            } else {
                lo = x;
            }
        }
        orbit.eval(0.5 * (lo + hi))
    };
    // The shadow is followed from the link's middle to each end, each point the meeting nearest
    // the one before: in even steps, then closing in on the end, where the part's other sheets
    // come close.
    let mut track: Vec<(f64, Point3)> = vec![(middle, seed)];
    for toward in [1.0, 0.0] {
        let mut last = seed;
        let mut steps: Vec<f64> = (1..=48).map(|i| f64::from(i) / 49.0).collect();
        steps.extend((1..40).map(|j| 1.0 - 0.5f64.powi(j) / 49.0));
        for x in steps {
            let f = middle + (toward - middle) * x;
            let next = nearest(f, last);
            track.push((f, next));
            last = next;
        }
    }
    track.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Where the link's curve is the part's own outline, its orbit touches the part at the curve
    // and passes through it at the shadow, and toward a fold of the outline the two close up:
    // three meetings in one place, which no search of the orbit tells apart. The touching is
    // known, so the gap from the orbit to the part is divided by the square of the angle from it,
    // and what is left changes sign once, at the shadow: behind the outline for the low end of
    // an interval, ahead of it for the high.
    let own = cut.parts[part]
        .loops
        .iter()
        .flatten()
        .any(|t| t.piece == g.piece);
    let ahead = if high { -1.0 } else { 1.0 };
    let apart = |f: f64, p: Point3| -> Point3 {
        let (o, touch) = cells.axis.drawn(on_link(f), 0.0);
        let at = cells.axis.drawn(p, touch).1;
        if !own || (at - touch).abs() > 0.02 {
            return p;
        }
        let orbit = cells.axis.orbit(o);
        // Close to the touching the gap is its own series there: no first or constant term, so
        // g₂d²/2 + g₃d³/6 + g₄d⁴/24, and the shadow is where g₂/2 + g₃d/6 + g₄d²/24 comes to
        // nothing. The terms are read from the gap about a thousandth of a radian either side,
        // far enough out to be well above rounding, with the step's own error taken off
        // (Richardson). A search for the change of sign this close is lost in rounding.
        let gap = |d: f64| csi::signed_distance(&dom.surface, orbit.eval(touch + ahead * d));
        let h = 5e-4;
        let even = |h: f64| (gap(h) + gap(-h)) / (h * h);
        let odd = |h: f64| (gap(h) - gap(-h)) / (2.0 * h * h * h);
        let g2 = (4.0 * even(h) - even(2.0 * h)) / 3.0;
        let g4 = 4.0 * (even(2.0 * h) - even(h)) / (h * h);
        let g3 = 6.0 * (4.0 * odd(h) - odd(2.0 * h)) / 3.0;
        if g3 != 0.0 {
            let mut d = -3.0 * g2 / g3;
            for _ in 0..4 {
                d = -(g2 / 2.0 + g4 * d * d / 24.0) / (g3 / 6.0);
            }
            if d.abs() <= 1e-3 {
                return orbit.eval(touch + ahead * d.max(0.0));
            }
        }
        let left = |d: f64| gap(d) / (d * d);
        let far = 0.03;
        let sign = left(far) > 0.0;
        let (mut lo, mut hi) = (1e-9, far);
        if (left(lo) > 0.0) == sign {
            // Too near the touching to tell: the first step out where the sign is the near
            // side's, if any.
            let mut d = lo;
            while d < far && (left(d) > 0.0) == sign {
                d *= 1.5;
            }
            if d >= far {
                return orbit.eval(touch + ahead * lo);
            }
            lo = d;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if (left(mid) > 0.0) == sign {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        orbit.eval(touch + ahead * 0.5 * (lo + hi))
    };
    let meet = |f: f64| -> Option<(Point3, Vec2)> {
        let i = track.partition_point(|x| x.0 < f).clamp(1, track.len() - 1);
        let (a, b) = (track[i - 1], track[i]);
        let near =
            a.1.lerp(b.1, ((f - a.0) / (b.0 - a.0).max(1e-300)).clamp(0.0, 1.0));
        Some(nearest(f, near))
            .map(|p| apart(f, p))
            .map(|p| (p, Vec2::ZERO))
    };
    // Where the shadow runs out onto the part's own outline, the orbit there only touches the
    // part: the end is the outline's point at the node, which is known exactly.
    let pieces: BTreeSet<usize> = cut.parts[part]
        .loops
        .iter()
        .flatten()
        .map(|t| t.piece)
        .collect();
    let end = |node: usize, f: f64, inner: f64| -> Result<Point3> {
        let close = meet(inner).expect("checked").0;
        let mut best: Option<Point3> = None;
        for (l, list) in cells.links.iter().zip(&cells.gens) {
            for own in list.iter().filter(|x| pieces.contains(&x.piece)) {
                for (at, t) in [(l.a, own.t.0), (l.b, own.t.1)] {
                    if at != node {
                        continue;
                    }
                    let p = cells.pieces[own.piece]
                        .curve
                        .as_ref()
                        .expect("a drawn piece")
                        .eval(t);
                    if best.is_none_or(|b| p.dist(close) < b.dist(close)) {
                        best = Some(p);
                    }
                }
            }
        }
        match best {
            Some(p) if p.dist(close) <= 1e-2 * cells.size => Ok(p),
            _ => meet(f).map(|x| x.0).ok_or_else(|| {
                Error::Numerical("a shadow across a face ends where nothing was found".into())
            }),
        }
    };
    let (pa, pb) = (end(link.a, 0.0, 1e-6)?, end(link.b, 1.0, 1.0 - 1e-6)?);
    let at = |f: f64| meet(f).map_or(if f < 0.5 { pa } else { pb }, |x| x.0);
    let (curve, range) = fit_exact(&at, (0.0, 1.0), (pa, pb), cells.size)?;
    // Each end's parameters on the face, carried out from the middle, which is inside the face
    // and so in its own window of a periodic parameter.
    let walk = |to: f64, p: Point3| {
        let mut uv = match dom.locate(seed) {
            Where::Inside(uv) => uv,
            _ => dom.surface.project(seed),
        };
        for n in 1..32 {
            let f = middle + (to - middle) * f64::from(n) / 32.0;
            if let Some((q, _)) = meet(f) {
                uv = dom.surface.project_near(q, uv);
            }
        }
        dom.surface.project_near(p, uv)
    };
    out.shadows
        .push((Rc::new(curve), (walk(0.0, pa), walk(1.0, pb))));
    Ok(Lift {
        piece: cells.pieces.len() + out.shadows.len() - 1,
        t: range,
        end: high,
    })
}
