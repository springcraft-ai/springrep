//! The curves where a smooth function of two parameters vanishes over a rectangle, found on a grid
//! as marching squares finds a contour (W. E. Lorensen and H. E. Cline, "Marching cubes: a high
//! resolution 3D surface construction algorithm", SIGGRAPH 1987, here in two dimensions): each
//! sign change along a grid line pinned by bisection and the crossings in each cell joined. Where a
//! cell holds a saddle of the function on its zero set, the four branches meet there.

use std::collections::HashMap;

use crate::math::Vec2;

/// Zeros of the function in order along one curve: open between two ends, each on the rectangle's
/// side or at a saddle, or closed, its last point its first.
pub(super) struct Chain {
    pub pts: Vec<Vec2>,
    pub closed: bool,
}

/// The rectangle from `lo` to `hi` as a grid of `n` cells each way.
#[derive(Clone, Copy)]
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

    fn cell(&self) -> Vec2 {
        Vec2::new(
            (self.hi.x - self.lo.x) / self.n.0 as f64,
            (self.hi.y - self.lo.y) / self.n.1 as f64,
        )
    }
}

/// The zero between `a` and `b`, where `f` has opposite signs, by bisection.
fn root(f: &dyn Fn(Vec2) -> f64, a: Vec2, b: Vec2, fa: f64) -> Vec2 {
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if (f(a.lerp(b, mid)) >= 0.0) == (fa >= 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    a.lerp(b, 0.5 * (lo + hi))
}

/// The point near `p` where the gradient of `f` vanishes, by Newton steps on differences `h` apart,
/// if the steps settle within `reach` of `p`.
fn critical(f: &dyn Fn(Vec2) -> f64, p: Vec2, h: Vec2, reach: Vec2) -> Option<Vec2> {
    let mut x = p;
    for _ in 0..30 {
        let (dx, dy) = (Vec2::new(h.x, 0.0), Vec2::new(0.0, h.y));
        let fxp = f(x + dx);
        let fxm = f(x - dx);
        let fyp = f(x + dy);
        let fym = f(x - dy);
        let f0 = f(x);
        let gx = (fxp - fxm) / (2.0 * h.x);
        let gy = (fyp - fym) / (2.0 * h.y);
        let hxx = (fxp - 2.0 * f0 + fxm) / (h.x * h.x);
        let hyy = (fyp - 2.0 * f0 + fym) / (h.y * h.y);
        let hxy =
            (f(x + dx + dy) - f(x + dx - dy) - f(x - dx + dy) + f(x - dx - dy)) / (4.0 * h.x * h.y);
        let det = hxx * hyy - hxy * hxy;
        if det.abs() < 1e-300 {
            return None;
        }
        let step = Vec2::new((hyy * gx - hxy * gy) / det, (hxx * gy - hxy * gx) / det);
        x = x - step;
        if (x.x - p.x).abs() > reach.x || (x.y - p.y).abs() > reach.y {
            return None;
        }
        if step.x.abs() <= 1e-12 * reach.x && step.y.abs() <= 1e-12 * reach.y {
            return Some(x);
        }
    }
    Some(x)
}

/// The curves where `f` vanishes over `grid`. A zero on a grid node counts as positive.
pub(super) fn chains(f: &dyn Fn(Vec2) -> f64, grid: Grid) -> Vec<Chain> {
    let (nx, ny) = grid.n;
    let at = |i: usize, j: usize| i * (ny + 1) + j;
    let vals: Vec<f64> = (0..=nx)
        .flat_map(|i| (0..=ny).map(move |j| (i, j)))
        .map(|(i, j)| f(grid.node(i, j)))
        .collect();
    let pos = |v: f64| v >= 0.0;
    let mut pts: Vec<Vec2> = Vec::new();
    // The crossing on each grid line between neighbouring nodes, along x and along y.
    let mut along_x: HashMap<(usize, usize), usize> = HashMap::new();
    let mut along_y: HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..=nx {
        for j in 0..=ny {
            let v = vals[at(i, j)];
            if i < nx && pos(v) != pos(vals[at(i + 1, j)]) {
                pts.push(root(f, grid.node(i, j), grid.node(i + 1, j), v));
                along_x.insert((i, j), pts.len() - 1);
            }
            if j < ny && pos(v) != pos(vals[at(i, j + 1)]) {
                pts.push(root(f, grid.node(i, j), grid.node(i, j + 1), v));
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
                    let (b, r, t, l) = (found[0], found[1], found[2], found[3]);
                    let centre = grid.node(i, j) + cell * 0.5;
                    let scale = [at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)]
                        .iter()
                        .map(|&k| vals[k].abs())
                        .fold(0.0, f64::max);
                    let mid = critical(f, centre, cell * 1e-3, cell * 0.75);
                    let value = f(mid.unwrap_or(centre));
                    if let Some(x) = mid
                        && value.abs() <= 1e-9 * scale
                    {
                        pts.push(x);
                        let s = pts.len() - 1;
                        links.extend([(b, s), (r, s), (t, s), (l, s)]);
                    } else if pos(value) == pos(vals[at(i, j)]) {
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
            if adj[node].len() != 2 {
                return (path, false);
            }
            match adj[node].iter().find(|&&k| !used[k]) {
                Some(&k) => link = k,
                None => return (path, false),
            }
        }
    };
    for (p, at) in adj.iter().enumerate() {
        if at.len() == 2 {
            continue;
        }
        for &k in at {
            if !used[k] {
                let (pts, closed) = walk(p, k, &mut used);
                out.push(Chain { pts, closed });
            }
        }
    }
    for k in 0..links.len() {
        if !used[k] {
            let (pts, closed) = walk(links[k].0, k, &mut used);
            out.push(Chain { pts, closed });
        }
    }
    out
}

/// The zero of `f` nearest `p` along its gradient, by Newton steps on differences `h` apart.
pub(super) fn settle(f: &dyn Fn(Vec2) -> f64, p: Vec2, h: Vec2) -> Vec2 {
    let mut x = p;
    for _ in 0..30 {
        let v = f(x);
        let gx = (f(x + Vec2::new(h.x, 0.0)) - f(x - Vec2::new(h.x, 0.0))) / (2.0 * h.x);
        let gy = (f(x + Vec2::new(0.0, h.y)) - f(x - Vec2::new(0.0, h.y))) / (2.0 * h.y);
        let g2 = gx * gx + gy * gy;
        if g2 <= 1e-300 {
            break;
        }
        let step = Vec2::new(gx, gy) * (v / g2);
        x = x - step;
        if step.x.abs() <= 1e-6 * h.x && step.y.abs() <= 1e-6 * h.y {
            break;
        }
    }
    x
}
