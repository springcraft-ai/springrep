//! A parameter plane of the sweep as a graph, and each of its regions made a face of the result:
//! the surface fitted to where the plane's points are carried, its loops trims along the pieces the
//! region's links stand for, and its side the one the body's outward normal there faces.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use super::graph::{self, Link, Region};
use super::{Ctx, fit};
use crate::{
    Error, Result,
    boolean::{
        domain::{Trim, UvMap},
        sew::OutFace,
    },
    geom::Surface,
    math::{Point3, Vec2, Vec3},
    topo::{FaceName, pcurve::fit_spline},
};

/// What a link stands for: a piece of the result, run along its parameter from the link's first
/// node or against it; or a piece run along its parameter `p` from the first node, whose point in
/// the plane is `origin + p·dir`.
#[derive(Clone, Copy, Debug)]
pub(super) enum Along {
    Piece(usize, bool),
    Line {
        piece: usize,
        origin: Vec2,
        dir: Vec2,
    },
}

/// A graph in a plane of parameters: nodes, each a vertex of the result or a corner of the plane,
/// and links between them. `scale` is the plane's extent each way.
pub(super) struct Plane {
    pub nodes: Vec<Vec2>,
    pub ids: Vec<Option<usize>>,
    pub links: Vec<Link>,
    pub along: Vec<Along>,
    scale: Vec2,
    by_id: HashMap<usize, Vec<usize>>,
}

impl Plane {
    pub(super) fn scale(&self) -> Vec2 {
        self.scale
    }

    pub(super) fn new(scale: Vec2) -> Plane {
        Plane {
            nodes: Vec::new(),
            ids: Vec::new(),
            links: Vec::new(),
            along: Vec::new(),
            scale,
            by_id: HashMap::new(),
        }
    }

    /// The node at `p` for vertex `id` of the result: one for each place in the plane the vertex
    /// is at (a vertex on a seam is at two).
    pub(super) fn node(&mut self, p: Vec2, id: Option<usize>) -> usize {
        let near =
            |q: Vec2| (q.x - p.x).abs() / self.scale.x + (q.y - p.y).abs() / self.scale.y <= 1e-6;
        if let Some(i) = id
            && let Some(&k) = self
                .by_id
                .get(&i)
                .and_then(|l| l.iter().find(|&&k| near(self.nodes[k])))
        {
            return k;
        }
        self.nodes.push(p);
        self.ids.push(id);
        let k = self.nodes.len() - 1;
        if let Some(i) = id {
            self.by_id.entry(i).or_default().push(k);
        }
        k
    }

    pub(super) fn link(&mut self, a: usize, b: usize, pts: Vec<Vec2>, along: Along) {
        if a == b && pts.len() <= 2 {
            return;
        }
        self.links.push(Link { a, b, pts });
        self.along.push(along);
    }

    pub(super) fn regions(&self, reach: f64) -> Result<Vec<Region>> {
        graph::regions(&self.nodes, &self.links, reach)
    }

    pub(super) fn inside(&self, region: &Region, reach: f64) -> Option<Vec2> {
        graph::inside(&self.links, region, reach)
    }
}

/// Where a plane's points are carried, and the surface through them.
pub(super) struct Sheet<'a> {
    pub surface: Surface,
    pub map: &'a dyn Fn(Vec2) -> Point3,
}

/// The surface through `map` over the links of `plane` that stand for pieces, one for every face of
/// the plane so the faces of one curve's sweep lie on one surface.
/// `care` says where in the plane the surface must follow the map: off it, the map may run away.
pub(super) fn sheet<'a>(
    ctx: &Ctx,
    plane: &Plane,
    map: &'a dyn Fn(Vec2) -> Point3,
    care: &dyn Fn(Vec2) -> bool,
) -> Result<Sheet<'a>> {
    let all: Vec<usize> = (0..plane.links.len()).collect();
    sheet_over(ctx, plane, &all, map, care)
}

/// The surface through `map` over `region` of `plane` alone, where the map runs off over the
/// rest of the plane.
pub(super) fn sheet_of<'a>(
    ctx: &Ctx,
    plane: &Plane,
    region: &Region,
    map: &'a dyn Fn(Vec2) -> Point3,
    care: &dyn Fn(Vec2) -> bool,
) -> Result<Sheet<'a>> {
    let links: Vec<usize> = region.loops.iter().flatten().map(|x| x.0).collect();
    sheet_over(ctx, plane, &links, map, care)
}

fn sheet_over<'a>(
    ctx: &Ctx,
    plane: &Plane,
    links: &[usize],
    map: &'a dyn Fn(Vec2) -> Point3,
    care: &dyn Fn(Vec2) -> bool,
) -> Result<Sheet<'a>> {
    let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
    let mut hi = Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &k in links {
        for p in &plane.links[k].pts {
            lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
    }
    if !(hi.x > lo.x && hi.y > lo.y) {
        return Err(Error::Numerical("a face of a sweep spans nothing".into()));
    }
    let sites: Vec<Vec2> = plane
        .nodes
        .iter()
        .zip(&plane.ids)
        .filter(|(_, id)| id.is_some())
        .map(|(q, _)| *q)
        .collect();
    Ok(Sheet {
        surface: fit::surface(map, lo, hi, ctx.size, care, &sites)?,
        map,
    })
}

/// The face `region` of `plane` makes on `sheet`, its outside the way `outward` points at the
/// region's point `at`, and whether it folds back on itself.
pub(super) fn face(
    ctx: &Ctx,
    plane: &Plane,
    region: &Region,
    sheet: &Sheet,
    at: Vec2,
    outward: Vec3,
    feature: &str,
) -> Result<(OutFace, bool)> {
    let (surface, map) = (sheet.surface.clone(), sheet.map);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(k, _) in region.loops.iter().flatten() {
        for p in &plane.links[k].pts {
            lo = lo.min(p.y);
            hi = hi.max(p.y);
        }
    }
    let h = Vec2::new(
        1e-6 * (ctx.span.1 - ctx.span.0),
        1e-6 * (hi - lo).max(1e-300),
    );
    let normal = |q: Vec2| {
        let dx = map(q + Vec2::new(h.x, 0.0)) - map(q - Vec2::new(h.x, 0.0));
        let dy = map(q + Vec2::new(0.0, h.y)) - map(q - Vec2::new(0.0, h.y));
        dx.cross(dy)
    };
    let here = normal(at);
    let flip = here.dot(outward) < 0.0;
    // Whether the map turns over inside the face, which then folds back on itself: the envelope
    // has a cusp there, past which it lies inside the solid. Neighbouring samples are compared,
    // as the normal of a face winding round may itself turn right round.
    let (xlo, xhi) = (ctx.span.0, ctx.span.1);
    let n = 64;
    let sample = |i: usize, j: usize| {
        let q = Vec2::new(
            xlo + (xhi - xlo) * i as f64 / n as f64,
            lo + (hi - lo) * j as f64 / n as f64,
        );
        graph::holds(&plane.links, region, q).then(|| normal(q))
    };
    let grid: Vec<Vec<Option<Vec3>>> = (1..n)
        .map(|i| (1..n).map(|j| sample(i, j)).collect())
        .collect();
    let mut folds = false;
    for i in 0..grid.len() {
        for j in 0..grid[i].len() {
            let Some(a) = grid[i][j] else { continue };
            let next = [
                grid.get(i + 1).and_then(|r| r[j]),
                grid[i].get(j + 1).copied().flatten(),
            ];
            if next.iter().flatten().any(|b| a.dot(*b) < 0.0) {
                folds = true;
            }
        }
    }
    // A region far thinner than it is long, as where a cylinder's grazing curve runs nearly along
    // its rulings and spans almost no `u`, is a sliver of the plane on a surface stretched across
    // it; such a face is refused.
    let ring: Vec<Vec2> = graph::outline(&plane.links, region)
        .iter()
        .map(|p| Vec2::new(p.x / plane.scale().x, p.y / plane.scale().y))
        .collect();
    let area: f64 = (0..ring.len())
        .map(|i| ring[i].cross(ring[(i + 1) % ring.len()]))
        .sum::<f64>()
        .abs()
        * 0.5;
    let perimeter: f64 = (0..ring.len())
        .map(|i| ring[i].dist(ring[(i + 1) % ring.len()]))
        .sum();
    if area < 1e-4 * perimeter * perimeter {
        return Err(Error::Unsupported(
            "a sweep along a motion whose velocity changes on the body, where a face's grazing \
             curve runs nearly along the face's second parameter"
                .into(),
        ));
    }
    let fitted = matches!(surface, Surface::Nurbs(_));
    let uv_of = |q: Vec2, p: Point3| if fitted { q } else { surface.project(p) };
    let uv = uv_of(at, map(at));
    let reversed = surface.normal(uv.x, uv.y).dot(outward) < 0.0;
    let mut loops = Vec::with_capacity(region.loops.len());
    for lp in &region.loops {
        let mut order: Vec<(usize, bool)> = lp.clone();
        if flip {
            order.reverse();
            for x in &mut order {
                x.1 = !x.1;
            }
        }
        let mut trims = Vec::with_capacity(order.len());
        for (k, dir) in order {
            let link = &plane.links[k];
            let (a, b) = if dir {
                (link.a, link.b)
            } else {
                (link.b, link.a)
            };
            // On a surface fitted at the plane's own parameters a piece along a line of the plane
            // has exact parameters there, and any other piece the spline through its points there:
            // where the map nearly drops rank, projection lands on the wrong foot.
            let (pid, along, uvmap) = match plane.along[k] {
                Along::Piece(pid, along) if fitted => {
                    (pid, along, exact_map(ctx, plane, k, pid, along, &surface))
                }
                Along::Piece(pid, along) => (pid, along, UvMap::Projected),
                Along::Line { piece, origin, dir } if fitted => {
                    (piece, true, UvMap::Affine { origin, dir })
                }
                Along::Line { piece, .. } => (piece, true, UvMap::Projected),
            };
            let (Some(va), Some(vb)) = (plane.ids[a], plane.ids[b]) else {
                return Err(Error::Numerical(
                    "a face of a sweep has a corner that is no vertex".into(),
                ));
            };
            let forward = dir == along;
            let pg = &ctx.pieces[pid];
            let (t0, t1) = if forward {
                (pg.range.0, pg.range.1)
            } else {
                (pg.range.1, pg.range.0)
            };
            trims.push(Trim {
                piece: pid,
                forward,
                curve: pg.curve.clone(),
                pole: ctx.pool[va],
                t0,
                t1,
                map: uvmap,
                uv0: uv_of(plane.nodes[a], ctx.pool[va]),
                uv1: uv_of(plane.nodes[b], ctx.pool[vb]),
                v0: va,
                v1: vb,
            });
        }
        loops.push(trims);
    }
    let names: BTreeSet<FaceName> = [FaceName::new(feature, FaceName::SIDE)].into();
    let face = OutFace {
        surface,
        reversed,
        names,
        loops,
        memo: None,
        side: None,
    };
    Ok((face, folds))
}

/// The map of link `k` of `plane`, along piece `pid` (`along` its parameter or against it), on
/// `surface` fitted at the plane's own parameters: the spline through the link's points.
fn exact_map(
    ctx: &Ctx,
    plane: &Plane,
    k: usize,
    pid: usize,
    along: bool,
    surface: &Surface,
) -> UvMap {
    let link = &plane.links[k];
    let pg = &ctx.pieces[pid];
    let Some(curve) = &pg.curve else {
        return UvMap::Projected;
    };
    let mut pts = link.pts.clone();
    let last = pts.len() - 1;
    pts[0] = plane.nodes[link.a];
    pts[last] = plane.nodes[link.b];
    if !along {
        pts.reverse();
    }
    match fit_spline(surface, curve, pg.range, &pts) {
        Some(s) => UvMap::Spline(Rc::new(s)),
        None => UvMap::Projected,
    }
}
