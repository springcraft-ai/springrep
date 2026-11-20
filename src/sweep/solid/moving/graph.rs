//! The faces of a graph drawn in a rectangle of parameters: each link a polyline between two nodes,
//! the faces traced with their inside on the left of their links, and each hole placed in the
//! smallest face around it.

use crate::{Error, Result, math::Vec2};

/// A polyline from node `a` to node `b`, its first and last points theirs.
pub(in crate::sweep::solid) struct Link {
    pub a: usize,
    pub b: usize,
    pub pts: Vec<Vec2>,
}

/// A face: its outer loop, then its holes, each as its links and whether each runs from `a` to `b`.
pub(in crate::sweep::solid) struct Region {
    pub loops: Vec<Vec<(usize, bool)>>,
}

fn points(links: &[Link], lp: &[(usize, bool)]) -> Vec<Vec2> {
    let mut out = Vec::new();
    for &(k, fwd) in lp {
        let pts = &links[k].pts;
        if fwd {
            out.extend(&pts[..pts.len() - 1]);
        } else {
            out.extend(pts[1..].iter().rev());
        }
    }
    out
}

fn area(poly: &[Vec2]) -> f64 {
    (0..poly.len())
        .map(|i| poly[i].cross(poly[(i + 1) % poly.len()]))
        .sum::<f64>()
        * 0.5
}

fn contains(poly: &[Vec2], p: Vec2) -> bool {
    let mut inside = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            inside = !inside;
        }
    }
    inside
}

/// The direction a link leaves its node along, read some way along it so links that leave
/// almost together are told apart.
fn heading(pts: &[Vec2], reach: f64) -> f64 {
    let start = pts[0];
    let total: f64 = pts.windows(2).map(|w| w[0].dist(w[1])).sum();
    let want = (total / 3.0).min(reach);
    let mut q = pts[1];
    for &p in &pts[1..] {
        q = p;
        if p.dist(start) >= want {
            break;
        }
    }
    let d = q - start;
    d.y.atan2(d.x)
}

/// The faces of the graph of `links` between `nodes`, bounded ones only. `reach` is a length in
/// the plane small against its features.
pub(in crate::sweep::solid) fn regions(
    nodes: &[Vec2],
    links: &[Link],
    reach: f64,
) -> Result<Vec<Region>> {
    let half = links.len() * 2;
    let origin = |h: usize| {
        if h.is_multiple_of(2) {
            links[h / 2].a
        } else {
            links[h / 2].b
        }
    };
    let pts_of = |h: usize| -> Vec<Vec2> {
        let p = &links[h / 2].pts;
        if h.is_multiple_of(2) {
            p.clone()
        } else {
            p.iter().rev().copied().collect()
        }
    };
    let mut around: Vec<Vec<(f64, usize)>> = vec![Vec::new(); nodes.len()];
    for h in 0..half {
        around[origin(h)].push((heading(&pts_of(h), reach), h));
    }
    for list in &mut around {
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let next = |h: usize| -> usize {
        let twin = h ^ 1;
        let list = &around[origin(twin)];
        let i = list
            .iter()
            .position(|x| x.1 == twin)
            .expect("each half-link leaves its node");
        list[(i + list.len() - 1) % list.len()].1
    };
    let mut seen = vec![false; half];
    let mut cycles: Vec<Vec<(usize, bool)>> = Vec::new();
    for h0 in 0..half {
        if seen[h0] {
            continue;
        }
        let mut cyc = Vec::new();
        let mut h = h0;
        while !seen[h] {
            seen[h] = true;
            cyc.push((h / 2, h % 2 == 0));
            h = next(h);
            if cyc.len() > half {
                return Err(Error::Numerical(
                    "a face of a sweep's parameter graph does not close".into(),
                ));
            }
        }
        if h != h0 {
            return Err(Error::Numerical(
                "a face of a sweep's parameter graph does not close".into(),
            ));
        }
        cycles.push(cyc);
    }
    let polys: Vec<Vec<Vec2>> = cycles.iter().map(|c| points(links, c)).collect();
    let areas: Vec<f64> = polys.iter().map(|p| area(p)).collect();
    let mut regions: Vec<Region> = Vec::new();
    let mut index = vec![usize::MAX; cycles.len()];
    for (i, c) in cycles.iter().enumerate() {
        if areas[i] > 0.0 {
            index[i] = regions.len();
            regions.push(Region {
                loops: vec![c.clone()],
            });
        }
    }
    for (i, c) in cycles.iter().enumerate() {
        if areas[i] > 0.0 {
            continue;
        }
        // A point just to the left of the cycle's longest segment, on the side of the face it
        // bounds: inside a face around it for a hole, outside everything for the plane's border.
        let poly = &polys[i];
        let Some(k) = (0..poly.len()).max_by(|&a, &b| {
            let la = poly[a].dist(poly[(a + 1) % poly.len()]);
            let lb = poly[b].dist(poly[(b + 1) % poly.len()]);
            la.total_cmp(&lb)
        }) else {
            continue;
        };
        let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
        let p = a.lerp(b, 0.5) + (b - a).perp() * 1e-6;
        let host = (0..cycles.len())
            .filter(|&k| areas[k] > 0.0 && contains(&polys[k], p))
            .min_by(|&a, &b| areas[a].total_cmp(&areas[b]));
        if let Some(k) = host {
            regions[index[k]].loops.push(c.clone());
        }
    }
    Ok(regions)
}

/// The points of `region`'s outer loop in order.
pub(in crate::sweep::solid) fn outline(links: &[Link], region: &Region) -> Vec<Vec2> {
    points(links, &region.loops[0])
}

/// Whether `p` lies inside `region`: inside its outer loop and outside its holes.
pub(in crate::sweep::solid) fn holds(links: &[Link], region: &Region, p: Vec2) -> bool {
    let polys: Vec<Vec<Vec2>> = region.loops.iter().map(|l| points(links, l)).collect();
    contains(&polys[0], p) && !polys[1..].iter().any(|h| contains(h, p))
}

/// A point inside `region`, a hair to the left of one of its links.
pub(in crate::sweep::solid) fn inside(links: &[Link], region: &Region, reach: f64) -> Option<Vec2> {
    let polys: Vec<Vec<Vec2>> = region.loops.iter().map(|l| points(links, l)).collect();
    let within = |p: Vec2| contains(&polys[0], p) && !polys[1..].iter().any(|h| contains(h, p));
    for lp in &region.loops {
        for &(k, fwd) in lp {
            let pts = &links[k].pts;
            for w in pts.windows(2) {
                let (a, b) = if fwd { (w[0], w[1]) } else { (w[1], w[0]) };
                let d = b - a;
                let len = d.norm();
                if len <= 0.0 {
                    continue;
                }
                for f in [1e-3, 1e-2, 1e-1] {
                    let p = a.lerp(b, 0.5) + d.perp() * (f * reach.min(len) / len);
                    if within(p) {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}
