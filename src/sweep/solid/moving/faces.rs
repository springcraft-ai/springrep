//! The faces' planes of (τ, place along a grazing curve): each region where the curve lies on its
//! face is swept into a face of the result. Its sides are the curves where the face grazes the
//! motion along its edges, carried over from the edges' planes, and the face's grazing curves where
//! the span starts and ends.

use super::charmap::CharMap;
use super::region::{self, Along, Plane};
use super::{Ctx, Side};
use crate::{
    Error, Result,
    boolean::domain::Where,
    math::{Point3, Vec2},
};

/// The faces the grazing curves sweep where they lie on their faces.
pub(super) fn faces(ctx: &mut Ctx, feature: &str) -> Result<()> {
    let mut made = Vec::new();
    for f in 0..ctx.faces.len() {
        if ctx.faces[f].still {
            continue;
        }
        for map in CharMap::of(ctx, f)? {
            made.extend(branch(ctx, &map, f, feature)?);
        }
    }
    ctx.sides.extend(made);
    Ok(())
}

/// The coordinate along `map` of the body point `p`, a point of the face, at `tau`, taken on the
/// side of a seam nearest `hint`.
fn coord_of(
    ctx: &Ctx,
    map: &CharMap,
    f: usize,
    tau: f64,
    p: Point3,
    hint: Option<f64>,
) -> Option<f64> {
    let s = &ctx.faces[f].surface;
    let mut uv = s.project(p);
    if let (Some(h), Some(per)) = (hint, s.u_period()) {
        uv.x += ((h - uv.x) / per).round() * per;
    }
    map.coord(tau, uv)
}

/// The faces the grazing curve of face `f` sweeps.
fn branch(ctx: &Ctx, map: &CharMap, f: usize, feature: &str) -> Result<Vec<Side>> {
    let (n0, np0) = (ctx.cut0.pool.len(), ctx.cut0.pieces.len());
    let range = map.range();
    let mut plane = Plane::new(Vec2::new(ctx.span.1 - ctx.span.0, range.1 - range.0));
    let mut any = false;
    for (tau, first) in [(ctx.span.0, true), (ctx.span.1, false)] {
        let (cut, shift, pshift) = if first {
            (&ctx.cut0, 0, 0)
        } else {
            (&ctx.cut1, n0, np0)
        };
        for &pid in &cut.curves[f] {
            let pg = &cut.pieces[pid];
            let curve = pg
                .curve
                .as_ref()
                .ok_or_else(|| Error::Numerical("a grazing curve with no curve".into()))?;
            let mid = curve.eval(0.5 * (pg.range.0 + pg.range.1));
            if !map.holds(tau, ctx.faces[f].surface.project(mid)) {
                continue;
            }
            // Each end read on the side of a seam its piece reaches it from.
            let near = |t: f64| curve.eval(pg.range.0 + (pg.range.1 - pg.range.0) * t);
            let hint_a = coord_of(ctx, map, f, tau, near(0.05), None);
            let hint_b = coord_of(ctx, map, f, tau, near(0.95), None);
            let (Some(xa), Some(xb)) = (
                coord_of(ctx, map, f, tau, cut.pool[pg.v0], hint_a),
                coord_of(ctx, map, f, tau, cut.pool[pg.v1], hint_b),
            ) else {
                return Err(Error::Numerical(
                    "a face's grazing curve where the span starts or ends has no place".into(),
                ));
            };
            let a = plane.node(Vec2::new(tau, xa), Some(pg.v0 + shift));
            let b = plane.node(Vec2::new(tau, xb), Some(pg.v1 + shift));
            plane.link(
                a,
                b,
                vec![Vec2::new(tau, xa), Vec2::new(tau, xb)],
                Along::Piece(pid + pshift, true),
            );
            any = true;
        }
    }
    let mut seen: Vec<(usize, Vec2)> = Vec::new();
    for gz in ctx.grazes.iter().filter(|g| g.face == f) {
        let edge = &ctx.edges[gz.edge];
        for &(c, side_face) in &edge.sides {
            if side_face != f {
                continue;
            }
            let uvs = ctx
                .body
                .coedge_uvs(c, &gz.pts.iter().map(|q| q.y).collect::<Vec<f64>>());
            let k = gz.pts.len() / 2;
            if !map.holds(gz.pts[k].x, uvs[k]) {
                continue;
            }
            let pts = gz
                .pts
                .iter()
                .zip(&uvs)
                .map(|(q, &uv)| map.coord(q.x, uv).map(|x| Vec2::new(q.x, x)))
                .collect::<Option<Vec<Vec2>>>()
                .ok_or_else(|| Error::Numerical("a grazing curve runs off its face".into()))?;
            // A seam whose two sides differ only in the parameter the plane does not hold (a
            // torus's seam along `u`) is one link.
            let first = pts[0];
            if seen
                .iter()
                .any(|s: &(usize, Vec2)| s.0 == gz.piece && s.1.dist(first) <= 0.0)
            {
                continue;
            }
            seen.push((gz.piece, first));
            let a = plane.node(pts[0], Some(gz.ends.0));
            let b = plane.node(*pts.last().expect("a piece has points"), Some(gz.ends.1));
            plane.link(a, b, pts, Along::Piece(gz.piece, true));
            any = true;
        }
    }
    if !any {
        return Ok(Vec::new());
    }
    let reach = 1e-3 * (ctx.span.1 - ctx.span.0).min(range.1 - range.0);
    let regions = plane.regions(reach)?;
    let place = |q: Vec2| -> Point3 {
        let p = map.point(q.x, q.y).unwrap_or(Point3::ZERO);
        ctx.pose(q.x).point(p)
    };
    // The surface need follow the map only where the curve's point is on or near the face: beside
    // it the point may run off far, as where a cylinder's curve leaves past its end.
    let (dlo, dhi) = (ctx.faces[f].dom.lo, ctx.faces[f].dom.hi);
    let margin = (dhi - dlo) * 0.1;
    let periods = (
        ctx.faces[f].surface.u_period(),
        ctx.faces[f].surface.v_period(),
    );
    let within = |x: f64, lo: f64, hi: f64, m: f64, period: Option<f64>| match period {
        Some(p) => lo - m + (x - (lo - m)).rem_euclid(p) <= hi + m,
        None => x >= lo - m && x <= hi + m,
    };
    let care = |q: Vec2| {
        map.uv(q.x, q.y).is_some_and(|uv| {
            within(uv.x, dlo.x, dhi.x, margin.x, periods.0)
                && within(uv.y, dlo.y, dhi.y, margin.y, periods.1)
        })
    };
    // One surface for the whole plane where the curve's map holds over all of it; else one for
    // each region.
    let mut sheet: Option<Option<region::Sheet>> = None;
    let mut made = Vec::new();
    for region in &regions {
        let at = plane
            .inside(region, reach)
            .ok_or_else(|| Error::Numerical("a region of a face's sweep has no inside".into()))?;
        let Some(uv) = map.uv(at.x, at.y) else {
            continue;
        };
        let p = ctx.faces[f].surface.eval(uv.x, uv.y);
        if !matches!(ctx.faces[f].dom.locate(p), Where::Inside(_)) {
            continue;
        }
        // The face reaches back from the grazing curve against the velocity. Where that is the
        // side the face moves outward on, it folds back over that side and faces the other way:
        // on the outer side of a turn the curve bounds what that side sweeps from within.
        let side = if map.rise(at.x, uv) >= 0.0 { 1.0 } else { -1.0 };
        let outward = ctx.pose(at.x).vector(map.normal(uv) * side);
        if sheet.is_none() {
            sheet = Some(region::sheet(ctx, &plane, &place, &care).ok());
        }
        let own;
        let sh = match &sheet {
            Some(Some(s)) => s,
            _ => {
                own = region::sheet_of(ctx, &plane, region, &place, &care)?;
                &own
            }
        };
        let (face, folds) = region::face(ctx, &plane, region, sh, at, outward, feature)?;
        made.push(Side {
            face,
            owner: f,
            whole: true,
            folds,
        });
    }
    Ok(made)
}
