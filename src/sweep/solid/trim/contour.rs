//! Where the swept solid's boundary crosses a face, found on a grid of the face's parameters as
//! marching squares finds a contour (W. E. Lorensen and H. E. Cline, "Marching cubes: a high
//! resolution 3D surface construction algorithm", SIGGRAPH 1987, here in two dimensions): each
//! grid line whose ends the moving body reaches differently is cut where it changes, by bisection,
//! and the cuts in each cell are joined.

use std::collections::HashMap;

use super::reach::Reach;
use crate::{geom::Surface, math::Vec2};

/// Points of the parameter plane in order along one curve where the reached part of the face
/// begins: open between two ends on the grid's border, or closed, its last point its first.
pub(super) struct Chain {
    pub uvs: Vec<Vec2>,
    pub closed: bool,
}

/// A rectangle of the parameter plane as a grid of `n` cells each way.
#[derive(Clone, Copy, Debug)]
pub(super) struct Grid {
    pub lo: Vec2,
    pub hi: Vec2,
    pub n: (usize, usize),
}

impl Grid {
    fn node(&self, i: usize, j: usize) -> Vec2 {
        Vec2::new(
            self.lo.x + (self.hi.x - self.lo.x) * i as f64 / self.n.0 as f64,
            self.lo.y + (self.hi.y - self.lo.y) * j as f64 / self.n.1 as f64,
        )
    }

    /// A cell's width each way.
    pub(super) fn cell(&self) -> Vec2 {
        Vec2::new(
            (self.hi.x - self.lo.x) / self.n.0 as f64,
            (self.hi.y - self.lo.y) / self.n.1 as f64,
        )
    }
}

/// Whether the body reaches the point of `s` at `uv`.
pub(super) fn held(s: &Surface, reach: &Reach, uv: Vec2) -> bool {
    reach.holds(s.eval(uv.x, uv.y))
}

/// The point between `a` and `b`, which the body reaches differently, where that changes.
pub(super) fn cut(s: &Surface, reach: &Reach, a: Vec2, b: Vec2, at_a: bool) -> Vec2 {
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..52 {
        let mid = 0.5 * (lo + hi);
        if held(s, reach, a.lerp(b, mid)) == at_a {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    a.lerp(b, 0.5 * (lo + hi))
}

/// The chains over `grid` of `s`, and whether the body reaches each node.
pub(super) fn chains(s: &Surface, reach: &Reach, grid: Grid) -> (Vec<Chain>, Vec<bool>) {
    let (nx, ny) = grid.n;
    let at = |i: usize, j: usize| i * (ny + 1) + j;
    let vals: Vec<bool> = (0..=nx)
        .flat_map(|i| (0..=ny).map(move |j| (i, j)))
        .map(|(i, j)| held(s, reach, grid.node(i, j)))
        .collect();
    let mut pts: Vec<Vec2> = Vec::new();
    let mut along_x: HashMap<(usize, usize), usize> = HashMap::new();
    let mut along_y: HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..=nx {
        for j in 0..=ny {
            let v = vals[at(i, j)];
            if i < nx && v != vals[at(i + 1, j)] {
                pts.push(cut(s, reach, grid.node(i, j), grid.node(i + 1, j), v));
                along_x.insert((i, j), pts.len() - 1);
            }
            if j < ny && v != vals[at(i, j + 1)] {
                pts.push(cut(s, reach, grid.node(i, j), grid.node(i, j + 1), v));
                along_y.insert((i, j), pts.len() - 1);
            }
        }
    }
    let mut links: Vec<(usize, usize)> = Vec::new();
    let cell = grid.cell();
    for i in 0..nx {
        for j in 0..ny {
            let sides = [
                along_x.get(&(i, j)),
                along_y.get(&(i + 1, j)),
                along_x.get(&(i, j + 1)),
                along_y.get(&(i, j)),
            ];
            let found: Vec<usize> = sides.iter().filter_map(|s| s.copied()).collect();
            match found.len() {
                2 => links.push((found[0], found[1])),
                4 => {
                    // Two corners across the cell reached, the other two not: the middle of the
                    // cell says which pair it joins.
                    let (b, r, t, l) = (found[0], found[1], found[2], found[3]);
                    let centre = grid.node(i, j) + cell * 0.5;
                    if held(s, reach, centre) == vals[at(i, j)] {
                        links.push((b, r));
                        links.push((t, l));
                    } else {
                        links.push((b, l));
                        links.push((r, t));
                    }
                }
                _ => {}
            }
        }
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
    for (k, &(a, b)) in links.iter().enumerate() {
        adj[a].push(k);
        adj[b].push(k);
    }
    let mut used = vec![false; links.len()];
    let mut out = Vec::new();
    let walk = |start: usize, first: usize, used: &mut Vec<bool>| -> (Vec<Vec2>, bool) {
        let mut path = vec![pts[start]];
        let (mut node, mut link) = (start, first);
        loop {
            used[link] = true;
            let (a, b) = links[link];
            node = if a == node { b } else { a };
            path.push(pts[node]);
            if node == start {
                return (path, true);
            }
            match adj[node].iter().find(|&&k| !used[k]) {
                Some(&k) => link = k,
                None => return (path, false),
            }
        }
    };
    for (p, at) in adj.iter().enumerate() {
        if at.len() != 1 {
            continue;
        }
        if !used[at[0]] {
            let (uvs, closed) = walk(p, at[0], &mut used);
            out.push(Chain { uvs, closed });
        }
    }
    for k in 0..links.len() {
        if !used[k] {
            let (uvs, closed) = walk(links[k].0, k, &mut used);
            out.push(Chain { uvs, closed });
        }
    }
    (out, vals)
}

/// The point where the reached part of the face begins nearest `uv` across the direction `dir`,
/// looked for within `span` each way, if it is there.
pub(super) fn settle(s: &Surface, reach: &Reach, uv: Vec2, dir: Vec2, span: f64) -> Option<Vec2> {
    // How deep the body reaches, less the hair: it rises through zero where the reach begins.
    let g = |p: crate::math::Point3| reach.deepest(p) - reach.hair;
    settle_on(s, &g, uv, dir, span)
}

/// The point nearest `uv` across the direction `dir`, within `span` each way, where `g` of the
/// face's point changes sign, if there is one: the sign change is bracketed within `1e-14` of the
/// reach it was looked for in, by Brent's root finder (Brent 1973, ch. 4), which a corner of the
/// function, where two places along the path reach equally deep, cannot stall.
pub(super) fn settle_on(
    s: &Surface,
    g: &dyn Fn(crate::math::Point3) -> f64,
    uv: Vec2,
    dir: Vec2,
    span: f64,
) -> Option<Vec2> {
    let mut at = |x: f64| {
        let q = uv + dir * x;
        g(s.eval(q.x, q.y))
    };
    let mut r = span;
    for _ in 0..4 {
        let (g0, g1) = (at(-r), at(r));
        if (g0 > 0.0) != (g1 > 0.0) {
            let x = if super::plain_search() {
                false_position(&mut at, (-r, g0), (r, g1), 1e-14 * r)
            } else {
                super::brent::root(&mut at, (-r, g0), (r, g1), 1e-14 * r)
            };
            return Some(uv + dir * x);
        }
        r *= 2.0;
    }
    None
}

/// The sign change of `f` between two places with their values, by regula falsi (the Illinois
/// variant) with a bisection after each step that does not halve the bracket.
fn false_position(
    f: &mut dyn FnMut(f64) -> f64,
    (mut x0, mut g0): (f64, f64),
    (mut x1, mut g1): (f64, f64),
    tol: f64,
) -> f64 {
    let mut width = (x1 - x0).abs();
    for k in 0..120 {
        let bisect = k % 2 == 1 && (x1 - x0).abs() > 0.5 * width;
        if k % 2 == 1 {
            width = (x1 - x0).abs();
        }
        let x = if bisect || g1 == g0 {
            0.5 * (x0 + x1)
        } else {
            (x0 * g1 - x1 * g0) / (g1 - g0)
        };
        let gx = f(x);
        if (gx > 0.0) != (g1 > 0.0) {
            x0 = x1;
            g0 = g1;
        } else {
            g0 *= 0.5;
        }
        x1 = x;
        g1 = gx;
        if (x1 - x0).abs() <= tol || gx == 0.0 {
            break;
        }
    }
    x1
}
