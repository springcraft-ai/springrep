//! The edges' planes of (τ, edge parameter): the curves along which each face beside an edge grazes
//! the motion at the edge, and the regions where the edge bounds what the motion carries outward,
//! each swept into a face of the result.

use std::collections::HashMap;
use std::rc::Rc;

use super::region::{self, Along, Plane};
use super::zeros::{self, Chain, Grid};
use super::{Ctx, Graze, Side};
use crate::{
    Error, Result,
    boolean::sew::PieceGeom,
    math::{Point3, Vec2},
    tol,
};

/// Cells of the grid each way that grazing curves are looked for on.
const GRID: (usize, usize) = (256, 16);

/// The sides of edge `e` whose faces move off their own surfaces: none where both are one face.
fn moving_sides(ctx: &Ctx, e: usize) -> Vec<usize> {
    let sides = &ctx.edges[e].sides;
    if sides.len() != 2 || sides[0].1 == sides[1].1 {
        return Vec::new();
    }
    (0..2).filter(|&k| !ctx.faces[sides[k].1].still).collect()
}

/// The sides of edge `e` whose grazing curves are found: a seam's one face once, and of an edge
/// the two faces meet smoothly along, the one face's curve, which is the other's too.
fn grazing_sides(ctx: &Ctx, e: usize) -> Vec<usize> {
    let sides = &ctx.edges[e].sides;
    if sides.len() == 2 && sides[0].1 == sides[1].1 {
        return if ctx.faces[sides[0].1].still {
            Vec::new()
        } else {
            vec![0]
        };
    }
    let moving = moving_sides(ctx, e);
    if moving.len() == 2 && smooth(ctx, e) {
        return vec![0];
    }
    moving
}

/// Whether the two faces beside edge `e` have one normal all along it.
fn smooth(ctx: &Ctx, e: usize) -> bool {
    let edge = &ctx.edges[e];
    (0..=16).all(|k| {
        let t = edge.range.0 + (edge.range.1 - edge.range.0) * k as f64 / 16.0;
        let a = ctx.normal_on(edge.sides[0].0, edge.sides[0].1, t);
        let b = ctx.normal_on(edge.sides[1].0, edge.sides[1].1, t);
        a.dot(b) >= 1.0 - 1e-9
    })
}

/// The rate face `side` of edge `e` moves outward at the edge's point `t` at `tau`.
fn rate(ctx: &Ctx, e: usize, side: usize, tau: f64, t: f64) -> f64 {
    let (c, f) = ctx.edges[e].sides[side];
    let p = ctx.edges[e].curve.eval(t);
    ctx.normal_on(c, f, t).dot(ctx.twist(tau).at(p))
}

fn grid(ctx: &Ctx, e: usize) -> Grid {
    let r = ctx.edges[e].range;
    Grid {
        lo: Vec2::new(ctx.span.0, r.0),
        hi: Vec2::new(ctx.span.1, r.1),
        n: GRID,
    }
}

/// Where two polylines cross: the index of each one's segment and the point.
fn crossings(a: &[Vec2], b: &[Vec2]) -> Vec<(usize, usize, Vec2)> {
    let mut out = Vec::new();
    for i in 0..a.len().saturating_sub(1) {
        let (p, r) = (a[i], a[i + 1] - a[i]);
        for j in 0..b.len().saturating_sub(1) {
            let (q, s) = (b[j], b[j + 1] - b[j]);
            let d = r.cross(s);
            if d.abs() <= 1e-300 {
                continue;
            }
            let t = (q - p).cross(s) / d;
            let u = (q - p).cross(r) / d;
            if (0.0..1.0).contains(&t) && (0.0..1.0).contains(&u) {
                out.push((i, j, p + r * t));
            }
        }
    }
    out
}

/// The point near `p` where both `f` and `g` vanish, by Newton steps on differences `h` apart.
fn both(f: &dyn Fn(Vec2) -> f64, g: &dyn Fn(Vec2) -> f64, p: Vec2, h: Vec2) -> Vec2 {
    let mut x = p;
    for _ in 0..30 {
        let d = |k: &dyn Fn(Vec2) -> f64| {
            (
                (k(x + Vec2::new(h.x, 0.0)) - k(x - Vec2::new(h.x, 0.0))) / (2.0 * h.x),
                (k(x + Vec2::new(0.0, h.y)) - k(x - Vec2::new(0.0, h.y))) / (2.0 * h.y),
            )
        };
        let (fx, fy) = d(f);
        let (gx, gy) = d(g);
        let det = fx * gy - fy * gx;
        if det.abs() <= 1e-300 {
            break;
        }
        let (fv, gv) = (f(x), g(x));
        let step = Vec2::new((fv * gy - gv * fy) / det, (fx * gv - gx * fv) / det);
        x = x - step;
        if step.x.abs() <= 1e-6 * h.x && step.y.abs() <= 1e-6 * h.y {
            break;
        }
    }
    x
}

/// The vertex of the result where a grazing curve on edge `e` ends at `q`: on the start or the end
/// of the span, a place where the edge was cut there; at an end of the edge, a place along that
/// vertex's path; else a new vertex, one for each place of the plane `inner` holds.
fn end(ctx: &mut Ctx, e: usize, q: Vec2, inner: &mut Vec<(Vec2, usize)>) -> Result<usize> {
    let (span, range) = (ctx.span, ctx.edges[e].range);
    let near_tau = 1e-9 * (span.1 - span.0);
    let near_t = 1e-9 * (range.1 - range.0);
    for (at, first) in [(span.0, true), (span.1, false)] {
        if (q.x - at).abs() > near_tau {
            continue;
        }
        let (cut, shift) = if first {
            (&ctx.cut0, 0)
        } else {
            (&ctx.cut1, ctx.cut0.pool.len())
        };
        let list = &cut.edges[&ctx.edges[e].id];
        let mut best: Option<(f64, usize)> = None;
        for &pid in list {
            let pg = &cut.pieces[pid];
            for (t, v) in [(pg.range.0, pg.v0), (pg.range.1, pg.v1)] {
                if best.is_none_or(|b| (t - q.y).abs() < b.0) {
                    best = Some(((t - q.y).abs(), v));
                }
            }
        }
        let (_, v) = best.ok_or_else(|| Error::Numerical("an edge with no pieces".into()))?;
        let p = cut.pool[v];
        if p.dist(ctx.edges[e].curve.eval(q.y)) > 1e-6 * ctx.size.max(1.0) {
            return Err(Error::Numerical(
                "a grazing curve meets the start or end of a sweep where no root was found".into(),
            ));
        }
        return Ok(v + shift);
    }
    if (q.y - range.0).abs() <= near_t {
        return Ok(ctx.event(ctx.edges[e].v0, q.x));
    }
    if (q.y - range.1).abs() <= near_t {
        return Ok(ctx.event(ctx.edges[e].v1, q.x));
    }
    if let Some(&(_, v)) = inner.iter().find(|x| x.0.dist(q) <= 1e-9) {
        return Ok(v);
    }
    let p = ctx.pose(q.x).point(ctx.edges[e].curve.eval(q.y));
    let v = ctx.vertex(p);
    inner.push((q, v));
    Ok(v)
}

/// Whether face `side` of edge `e` grazes the motion all along the edge over the whole span, as a
/// face of a box turned about a line through one of its edges does there: no curve of its plane
/// stands apart, and the edge is no boundary of what the face carries outward.
fn level(ctx: &Ctx, e: usize, side: usize) -> bool {
    let g = grid(ctx, e);
    let n = (32, 8);
    let mut most = 0.0f64;
    let mut speed = 0.0f64;
    for i in 0..=n.0 {
        for j in 0..=n.1 {
            let tau = g.lo.x + (g.hi.x - g.lo.x) * i as f64 / n.0 as f64;
            let t = g.lo.y + (g.hi.y - g.lo.y) * j as f64 / n.1 as f64;
            most = most.max(rate(ctx, e, side, tau, t).abs());
            speed = speed.max(ctx.twist(tau).at(ctx.edges[e].curve.eval(t)).norm());
        }
    }
    most <= 1e-12 * speed
}

/// The grazing curves of every edge, each cut where the other face's crosses it, made pieces of
/// the result.
pub(super) fn grazes(ctx: &mut Ctx) -> Result<()> {
    for e in 0..ctx.edges.len() {
        let sides = grazing_sides(ctx, e);
        let g = grid(ctx, e);
        let h = (g.hi - g.lo) * 1e-7;
        let mut chains: Vec<(usize, Chain)> = Vec::new();
        for &s in &sides {
            if level(ctx, e, s) {
                continue;
            }
            let f = |q: Vec2| rate(ctx, e, s, q.x, q.y);
            chains.extend(zeros::chains(&f, g).into_iter().map(|c| (s, c)));
        }
        // Each chain's cuts: where along it (segment and point) and the vertex there.
        let mut cuts: Vec<Vec<(usize, Vec2, usize)>> = vec![Vec::new(); chains.len()];
        let mut inner: Vec<(Vec2, usize)> = Vec::new();
        for a in 0..chains.len() {
            for b in a + 1..chains.len() {
                if chains[a].0 == chains[b].0 {
                    continue;
                }
                for (i, j, p) in crossings(&chains[a].1.pts, &chains[b].1.pts) {
                    let (sa, sb) = (chains[a].0, chains[b].0);
                    let fa = |q: Vec2| rate(ctx, e, sa, q.x, q.y);
                    let fb = |q: Vec2| rate(ctx, e, sb, q.x, q.y);
                    let q = both(&fa, &fb, p, h);
                    let v = end(ctx, e, q, &mut inner)?;
                    cuts[a].push((i, q, v));
                    cuts[b].push((j, q, v));
                }
            }
        }
        for (k, (side, chain)) in chains.iter().enumerate() {
            let mut pts = chain.pts.clone();
            let mut marks: Vec<(usize, Vec2, usize)> = cuts[k].clone();
            if chain.closed {
                // A closed curve is cut at its start and halfway round, so no piece ends where
                // it starts.
                pts.pop();
                let half = pts.len() / 2;
                let v0 = end(ctx, e, pts[0], &mut inner)?;
                let v1 = end(ctx, e, pts[half], &mut inner)?;
                pts.push(pts[0]);
                marks.push((0, pts[0], v0));
                marks.push((half, pts[half], v1));
                marks.push((pts.len() - 1, pts[0], v0));
            } else {
                let v0 = end(ctx, e, pts[0], &mut inner)?;
                let v1 = end(ctx, e, *pts.last().expect("a chain has points"), &mut inner)?;
                marks.push((0, pts[0], v0));
                marks.push((pts.len() - 1, *pts.last().expect("a chain has points"), v1));
            }
            // Order the cuts along the chain: by segment, then by distance from its start.
            marks.sort_by(|x, y| {
                x.0.cmp(&y.0)
                    .then((x.1 - pts[x.0]).norm().total_cmp(&(y.1 - pts[y.0]).norm()))
            });
            for w in marks.windows(2) {
                let (ia, pa, va) = w[0];
                let (ib, pb, vb) = w[1];
                if va == vb && ia == ib {
                    continue;
                }
                let mut seg = vec![pa];
                seg.extend(
                    pts[(ia + 1).min(pts.len())..=ib.min(pts.len() - 1)]
                        .iter()
                        .copied(),
                );
                if seg.last().is_some_and(|l| l.dist(pb) > 0.0) {
                    seg.push(pb);
                }
                seg.dedup_by(|x, y| x.dist(*y) <= 0.0);
                if seg.len() < 2 {
                    continue;
                }
                piece(ctx, e, *side, seg, (va, vb))?;
            }
        }
    }
    Ok(())
}

/// The piece of the result along the grazing curve of side `side` of edge `e` through `pts`.
fn piece(ctx: &mut Ctx, e: usize, side: usize, pts: Vec<Vec2>, ends: (usize, usize)) -> Result<()> {
    let g = grid(ctx, e);
    let scale = g.hi - g.lo;
    let norm = |p: Vec2| Vec2::new(p.x / scale.x, p.y / scale.y);
    let mut lens = vec![0.0];
    for w in pts.windows(2) {
        lens.push(lens.last().copied().unwrap_or(0.0) + norm(w[1] - w[0]).norm());
    }
    let total = *lens.last().expect("a piece has points");
    let h = scale * 1e-7;
    let along = |s: f64| -> Point3 {
        let x = s * total;
        let i = lens.partition_point(|&l| l <= x).clamp(1, pts.len() - 1);
        let f = ((x - lens[i - 1]) / (lens[i] - lens[i - 1]).max(1e-300)).clamp(0.0, 1.0);
        let q = pts[i - 1].lerp(pts[i], f);
        let rate_of = |q: Vec2| rate(ctx, e, side, q.x, q.y);
        let q = zeros::settle(&rate_of, q, h);
        ctx.pose(q.x).point(ctx.edges[e].curve.eval(q.y))
    };
    let (curve, range) = super::super::grazing::fit_exact(
        &along,
        (0.0, 1.0),
        (ctx.pool[ends.0], ctx.pool[ends.1]),
        ctx.size,
    )?;
    ctx.pieces.push(PieceGeom {
        curve: Some(Rc::new(curve)),
        range,
        v0: ends.0,
        v1: ends.1,
        tol: tol::LINEAR,
    });
    let piece = ctx.pieces.len() - 1;
    let mut faces = vec![ctx.edges[e].sides[side].1];
    // Where the faces meet smoothly the one curve is where both graze.
    let sides = &ctx.edges[e].sides;
    if sides.len() == 2
        && sides[0].1 != sides[1].1
        && moving_sides(ctx, e).len() == 2
        && smooth(ctx, e)
    {
        faces.push(sides[1 - side].1);
    }
    for face in faces {
        ctx.grazes.push(Graze {
            edge: e,
            face,
            pts: pts.clone(),
            piece,
            ends,
        });
    }
    Ok(())
}

/// The faces the edges sweep where they bound what the motion carries outward.
pub(super) fn faces(ctx: &mut Ctx, feature: &str) -> Result<()> {
    for e in 0..ctx.edges.len() {
        let sides = moving_sides(ctx, e);
        if sides.is_empty() {
            continue;
        }
        let plane = plane_of(ctx, e)?;
        let g = grid(ctx, e);
        let reach = 1e-3 * (g.hi.x - g.lo.x).min(g.hi.y - g.lo.y);
        let regions = plane.regions(reach)?;
        let map = |q: Vec2| ctx.pose(q.x).point(ctx.edges[e].curve.eval(q.y));
        let sheet = region::sheet(ctx, &plane, &map, &|_| true)?;
        let mut made = Vec::new();
        for region in &regions {
            let at = plane.inside(region, reach).ok_or_else(|| {
                Error::Numerical("a region of an edge's sweep has no inside".into())
            })?;
            let speed = ctx.twist(at.x).omega.norm() * ctx.size + ctx.twist(at.x).v.norm();
            let fronts: Vec<usize> = sides
                .iter()
                .copied()
                .filter(|&s| rate(ctx, e, s, at.x, at.y) > 1e-12 * speed)
                .collect();
            let pose = ctx.pose(at.x);
            let h = (g.hi - g.lo) * 1e-6;
            let along_tau = map(at + Vec2::new(h.x, 0.0)) - map(at - Vec2::new(h.x, 0.0));
            let along_t = map(at + Vec2::new(0.0, h.y)) - map(at - Vec2::new(0.0, h.y));
            let normal = along_tau.cross(along_t);
            for &s in &fronts {
                // The face beside the edge lies inside what it sweeps: the swept face faces away
                // from it.
                let (c, f) = ctx.edges[e].sides[s];
                let (_, d) = ctx.edges[e].curve.d1(at.y);
                let d = if ctx.body.coedge(c).reversed { -d } else { d };
                let into = pose.vector(ctx.normal_on(c, f, at.y).cross(d));
                let outward = if normal.dot(into) < 0.0 {
                    normal
                } else {
                    -normal
                };
                let (face, folds) =
                    region::face(ctx, &plane, region, &sheet, at, outward, feature)?;
                made.push(Side {
                    face,
                    owner: f,
                    whole: fronts.len() == 1,
                    folds,
                });
            }
        }
        ctx.sides.extend(made);
    }
    Ok(())
}

/// Edge `e`'s plane: its sides the edge cut at the span's start and end and the paths of its two
/// vertices, and the grazing curves on it.
fn plane_of(ctx: &mut Ctx, e: usize) -> Result<Plane> {
    let range = ctx.edges[e].range;
    let mut plane = Plane::new(Vec2::new(ctx.span.1 - ctx.span.0, range.1 - range.0));
    let (span, id) = (ctx.span, ctx.edges[e].id);
    let (n0, np0) = (ctx.cut0.pool.len(), ctx.cut0.pieces.len());
    for (at, first) in [(span.0, true), (span.1, false)] {
        let (cut, shift, pshift) = if first {
            (&ctx.cut0, 0, 0)
        } else {
            (&ctx.cut1, n0, np0)
        };
        for &pid in &cut.edges[&id] {
            let pg = &cut.pieces[pid];
            let a = plane.node(Vec2::new(at, pg.range.0), Some(pg.v0 + shift));
            let b = plane.node(Vec2::new(at, pg.range.1), Some(pg.v1 + shift));
            let pts = vec![Vec2::new(at, pg.range.0), Vec2::new(at, pg.range.1)];
            let line = Along::Line {
                piece: pid + pshift,
                origin: Vec2::new(at, 0.0),
                dir: Vec2::new(0.0, 1.0),
            };
            plane.link(a, b, pts, line);
        }
    }
    for (v, t) in [
        (ctx.edges[e].v0, ctx.edges[e].range.0),
        (ctx.edges[e].v1, ctx.edges[e].range.1),
    ] {
        let stops = ctx.stops(v);
        for w in stops.windows(2) {
            let pid = ctx.path(v, w[0], w[1])?;
            let a = plane.node(Vec2::new(w[0].0, t), Some(w[0].1));
            let b = plane.node(Vec2::new(w[1].0, t), Some(w[1].1));
            let pts = vec![Vec2::new(w[0].0, t), Vec2::new(w[1].0, t)];
            let line = Along::Line {
                piece: pid,
                origin: Vec2::new(0.0, t),
                dir: Vec2::new(1.0, 0.0),
            };
            plane.link(a, b, pts, line);
        }
    }
    let mine: Vec<usize> = (0..ctx.grazes.len())
        .filter(|&k| ctx.grazes[k].edge == e)
        .collect();
    let mut seen: HashMap<usize, ()> = HashMap::new();
    for k in mine {
        let gz = &ctx.grazes[k];
        if seen.insert(gz.piece, ()).is_some() {
            continue;
        }
        let a = plane.node(gz.pts[0], Some(gz.ends.0));
        let b = plane.node(*gz.pts.last().expect("a piece has points"), Some(gz.ends.1));
        plane.link(a, b, gz.pts.clone(), Along::Piece(gz.piece, true));
    }
    Ok(plane)
}
