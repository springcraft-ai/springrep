//! A bounding volume hierarchy over boxes, and the closest points and crossings of points,
//! segments and triangles: the search structures the thin-place probe, distances and rays share.

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

use crate::math::{Aabb, Point3, Transform, Vec3};

use super::Mesh;

/// A pair of nodes keyed by their boxes' gap, for a queue that opens the nearest first.
struct Gap(f64, usize, usize);

impl PartialEq for Gap {
    fn eq(&self, other: &Gap) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Gap {}

impl PartialOrd for Gap {
    fn partial_cmp(&self, other: &Gap) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Gap {
    fn cmp(&self, other: &Gap) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then((self.1, self.2).cmp(&(other.1, other.2)))
    }
}

/// A bounding volume hierarchy over boxes: a mesh's triangles, or any primitives' boxes.
#[derive(Clone, Debug)]
pub(crate) struct Bvh {
    boxes: Vec<Aabb>,
    /// Node: its box, and either two children or a run of `order`.
    nodes: Vec<(Aabb, Node)>,
    order: Vec<usize>,
}

/// The order a run of boxes stands in after a stable sort by centre along each of its ancestors'
/// axes, the root's first: by the last axis sorted on, then the one before, and at last by index,
/// the order the boxes came in. An axis sorted on again counts where it was sorted on last.
#[derive(Clone, Copy, Default)]
struct Keys {
    axes: [usize; 3],
    len: usize,
}

impl Keys {
    /// The order after one more stable sort, along `axis`.
    fn then(self, axis: usize) -> Keys {
        let mut next = Keys {
            axes: [axis; 3],
            len: 1,
        };
        for &a in &self.axes[..self.len] {
            if a != axis {
                next.axes[next.len] = a;
                next.len += 1;
            }
        }
        next
    }

    fn cmp(&self, centres: &[[f64; 3]], i: usize, j: usize) -> Ordering {
        self.axes[..self.len]
            .iter()
            .map(|&a| centres[i][a].total_cmp(&centres[j][a]))
            .find(|o| o.is_ne())
            .unwrap_or(i.cmp(&j))
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Inner(usize, usize),
    Leaf(usize, usize),
}

impl Bvh {
    pub(crate) fn new(mesh: &Mesh) -> Bvh {
        let boxes: Vec<Aabb> = mesh
            .triangles
            .iter()
            .map(|t| {
                let mut b = Aabb::EMPTY;
                for &i in t {
                    b.add(mesh.positions[i as usize]);
                }
                b
            })
            .collect();
        Bvh::of_boxes(boxes)
    }

    /// The hierarchy over `boxes`, whose indices a query returns.
    pub(crate) fn of_boxes(boxes: Vec<Aabb>) -> Bvh {
        let centres: Vec<[f64; 3]> = boxes
            .iter()
            .map(|b| ((b.min + b.max) * 0.5).to_array())
            .collect();
        let mut bvh = Bvh {
            order: (0..boxes.len()).collect(),
            boxes,
            nodes: Vec::new(),
        };
        if !bvh.order.is_empty() {
            bvh.split(0, bvh.order.len(), &centres, Keys::default());
        }
        bvh
    }

    /// The boxes the hierarchy is over, in the order they were given.
    pub(crate) fn boxes(&self) -> &[Aabb] {
        &self.boxes
    }

    /// The node over `order[lo..hi]`, halved at the median centre along its box's longest axis.
    ///
    /// `keys` orders the run as a stable sort along each ancestor's axis in turn would have left
    /// it. Halving needs only which boxes fall in each half of that order, so a selection stands
    /// for the sort, and a leaf alone is put in order.
    fn split(&mut self, lo: usize, hi: usize, centres: &[[f64; 3]], keys: Keys) -> usize {
        let mut bb = Aabb::EMPTY;
        for &i in &self.order[lo..hi] {
            bb.union(&self.boxes[i]);
        }
        let at = self.nodes.len();
        self.nodes.push((bb, Node::Leaf(lo, hi)));
        if hi - lo <= 4 {
            self.order[lo..hi].sort_unstable_by(|&i, &j| keys.cmp(centres, i, j));
            return at;
        }
        let ext = bb.max - bb.min;
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };
        let keys = keys.then(axis);
        let mid = (lo + hi) / 2;
        self.order[lo..hi].select_nth_unstable_by(mid - lo, |&i, &j| keys.cmp(centres, i, j));
        let l = self.split(lo, mid, centres, keys);
        let r = self.split(mid, hi, centres, keys);
        self.nodes[at].1 = Node::Inner(l, r);
        at
    }

    /// Every box that, slid any distance either way along `d`, meets `q`: every triangle a line
    /// along `d` through `q` can cross, so a ray's crossings keep their parity.
    pub(crate) fn along_line(&self, q: &Aabb, d: Vec3, out: &mut Vec<usize>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0];
        while let Some(n) = stack.pop() {
            let (bb, node) = &self.nodes[n];
            if !line_meets(bb, q, d) {
                continue;
            }
            match *node {
                Node::Inner(l, r) => {
                    stack.push(l);
                    stack.push(r);
                }
                Node::Leaf(lo, hi) => out.extend(
                    self.order[lo..hi]
                        .iter()
                        .copied()
                        .filter(|&i| line_meets(&self.boxes[i], q, d)),
                ),
            }
        }
    }

    /// Every box within straight-line distance `d` of `q`.
    pub(crate) fn within(&self, q: &Aabb, d: f64, out: &mut Vec<usize>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0];
        while let Some(n) = stack.pop() {
            let (bb, node) = &self.nodes[n];
            if box_gap(bb, q) > d {
                continue;
            }
            match *node {
                Node::Inner(l, r) => {
                    stack.push(l);
                    stack.push(r);
                }
                Node::Leaf(lo, hi) => out.extend(
                    self.order[lo..hi]
                        .iter()
                        .copied()
                        .filter(|&i| box_gap(&self.boxes[i], q) <= d),
                ),
            }
        }
    }

    /// Every box, widened by `pad` on each side, that the segment `o + t·d` for `t` in `[t0, t1]`
    /// passes through.
    pub(crate) fn along(
        &self,
        o: Point3,
        d: Vec3,
        (t0, t1): (f64, f64),
        pad: f64,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let meets = |b: &Aabb| {
            let (mut lo, mut hi) = (t0, t1);
            for (o, d, a, z) in [
                (o.x, d.x, b.min.x - pad, b.max.x + pad),
                (o.y, d.y, b.min.y - pad, b.max.y + pad),
                (o.z, d.z, b.min.z - pad, b.max.z + pad),
            ] {
                if d.abs() < 1e-300 {
                    if o < a || o > z {
                        return false;
                    }
                    continue;
                }
                let (s0, s1) = ((a - o) / d, (z - o) / d);
                lo = lo.max(s0.min(s1));
                hi = hi.min(s0.max(s1));
                if lo > hi {
                    return false;
                }
            }
            true
        };
        let mut stack = vec![0];
        while let Some(n) = stack.pop() {
            let (bb, node) = &self.nodes[n];
            if !meets(bb) {
                continue;
            }
            match *node {
                Node::Inner(l, r) => {
                    stack.push(l);
                    stack.push(r);
                }
                Node::Leaf(lo, hi) => out.extend(
                    self.order[lo..hi]
                        .iter()
                        .copied()
                        .filter(|&i| meets(&self.boxes[i])),
                ),
            }
        }
    }

    /// Whether `near` holds for some box of `self` and some box of `other` moved by `t` that lie
    /// within `d` of each other, trying pairs nearest first. Neither hierarchy is rebuilt: `other`'s
    /// boxes are moved whole, which keeps them around what they held.
    pub(crate) fn any_within(
        &self,
        other: &Bvh,
        t: &Transform,
        d: f64,
        mut near: impl FnMut(usize, usize) -> bool,
    ) -> bool {
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return false;
        }
        let moved = |b: &Aabb| moved_box(b, t);
        let mut stack = vec![(0usize, 0usize)];
        while let Some((a, b)) = stack.pop() {
            let (ba, na) = &self.nodes[a];
            let (bb, nb) = &other.nodes[b];
            let bb = moved(bb);
            if box_gap(ba, &bb) > d {
                continue;
            }
            match (na, nb) {
                (Node::Leaf(alo, ahi), Node::Leaf(blo, bhi)) => {
                    for &j in &other.order[*blo..*bhi] {
                        let bj = moved(&other.boxes[j]);
                        for &i in &self.order[*alo..*ahi] {
                            if box_gap(&self.boxes[i], &bj) <= d && near(i, j) {
                                return true;
                            }
                        }
                    }
                }
                (Node::Inner(l, r), Node::Leaf(..)) => {
                    stack.push((*l, b));
                    stack.push((*r, b));
                }
                (Node::Leaf(..), Node::Inner(l, r)) => {
                    stack.push((a, *l));
                    stack.push((a, *r));
                }
                (Node::Inner(al, ar), Node::Inner(bl, br)) => {
                    let ea = ba.max - ba.min;
                    let eb = bb.max - bb.min;
                    if ea.dot(ea) >= eb.dot(eb) {
                        stack.push((*al, b));
                        stack.push((*ar, b));
                    } else {
                        stack.push((a, *bl));
                        stack.push((a, *br));
                    }
                }
            }
        }
        false
    }

    /// The least `dist(i, j)` over box `i` of `self` and box `j` of `other`, where `dist` is never
    /// less than the boxes' gap. Pairs of nodes are opened nearest first, so a near pair soon
    /// bounds the rest; a pair farther apart than the least so far is skipped, as [`Bvh::within`]
    /// skips it, and nothing is less than zero.
    pub(crate) fn least(&self, other: &Bvh, mut dist: impl FnMut(usize, usize) -> f64) -> f64 {
        let mut best = f64::INFINITY;
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return best;
        }
        let gap = |a: usize, b: usize| box_gap(&self.nodes[a].0, &other.nodes[b].0);
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(Gap(gap(0, 0), 0, 0)));
        while let Some(Reverse(Gap(g, a, b))) = heap.pop() {
            if g > best || best == 0.0 {
                break;
            }
            let (ba, na) = &self.nodes[a];
            let (bb, nb) = &other.nodes[b];
            let pairs = match (na, nb) {
                (Node::Leaf(alo, ahi), Node::Leaf(blo, bhi)) => {
                    for &i in &self.order[*alo..*ahi] {
                        for &j in &other.order[*blo..*bhi] {
                            if box_gap(&self.boxes[i], &other.boxes[j]) <= best {
                                best = best.min(dist(i, j));
                            }
                        }
                    }
                    continue;
                }
                (Node::Inner(l, r), Node::Leaf(..)) => [(*l, b), (*r, b)],
                (Node::Leaf(..), Node::Inner(l, r)) => [(a, *l), (a, *r)],
                (Node::Inner(al, ar), Node::Inner(bl, br)) => {
                    let (ea, eb) = (ba.max - ba.min, bb.max - bb.min);
                    if ea.dot(ea) >= eb.dot(eb) {
                        [(*al, b), (*ar, b)]
                    } else {
                        [(a, *bl), (a, *br)]
                    }
                }
            };
            for (x, y) in pairs {
                let g = gap(x, y);
                if g <= best {
                    heap.push(Reverse(Gap(g, x, y)));
                }
            }
        }
        best
    }

    /// Whether `meets` holds for some box. A node's box holds its boxes, so `meets` must hold for
    /// a box that holds one it holds for: then a node it fails for holds no box it would pass.
    pub(crate) fn any(&self, meets: impl Fn(&Aabb) -> bool) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = vec![0];
        while let Some(n) = stack.pop() {
            let (bb, node) = &self.nodes[n];
            if !meets(bb) {
                continue;
            }
            match *node {
                Node::Inner(l, r) => {
                    stack.push(r);
                    stack.push(l);
                }
                Node::Leaf(lo, hi) => {
                    if self.order[lo..hi].iter().any(|&i| meets(&self.boxes[i])) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Every box that meets `q`.
    pub(crate) fn query(&self, q: &Aabb, out: &mut Vec<usize>) {
        out.clear();
        if self.nodes.is_empty() {
            return;
        }
        let meets = |b: &Aabb| {
            b.min.x <= q.max.x
                && q.min.x <= b.max.x
                && b.min.y <= q.max.y
                && q.min.y <= b.max.y
                && b.min.z <= q.max.z
                && q.min.z <= b.max.z
        };
        let mut stack = vec![0];
        while let Some(n) = stack.pop() {
            let (bb, node) = &self.nodes[n];
            if !meets(bb) {
                continue;
            }
            match *node {
                Node::Inner(l, r) => {
                    stack.push(l);
                    stack.push(r);
                }
                Node::Leaf(lo, hi) => out.extend(
                    self.order[lo..hi]
                        .iter()
                        .copied()
                        .filter(|&i| meets(&self.boxes[i])),
                ),
            }
        }
    }
}

/// The straight-line distance between two boxes; zero where they meet.
/// Whether `b` slid some distance either way along `d` meets `q`: on each axis the slide carries
/// it over `q` for a span of distances, and those spans share one.
fn line_meets(b: &Aabb, q: &Aabb, d: Vec3) -> bool {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        let (lo, hi, step) = (q.min[k] - b.max[k], q.max[k] - b.min[k], d[k]);
        if step.abs() < 1e-12 {
            if lo > 0.0 || hi < 0.0 {
                return false;
            }
            continue;
        }
        let (x, y) = (lo / step, hi / step);
        t0 = t0.max(x.min(y));
        t1 = t1.min(x.max(y));
    }
    t0 <= t1
}

pub(crate) fn box_gap(a: &Aabb, b: &Aabb) -> f64 {
    let gap = |lo1: f64, hi1: f64, lo2: f64, hi2: f64| (lo2 - hi1).max(lo1 - hi2).max(0.0);
    let (x, y, z) = (
        gap(a.min.x, a.max.x, b.min.x, b.max.x),
        gap(a.min.y, a.max.y, b.min.y, b.max.y),
        gap(a.min.z, a.max.z, b.min.z, b.max.z),
    );
    (x * x + y * y + z * z).sqrt()
}

/// The box round `b`'s corners moved by `t`.
pub(crate) fn moved_box(b: &Aabb, t: &Transform) -> Aabb {
    let mut out = Aabb::EMPTY;
    for &x in &[b.min.x, b.max.x] {
        for &y in &[b.min.y, b.max.y] {
            for &z in &[b.min.z, b.max.z] {
                out.add(t.point(Point3::new(x, y, z)));
            }
        }
    }
    out
}

/// The distance between two triangles: zero where they cross, else the nearest of each one's
/// sides to the other.
pub(crate) fn triangles(a: &[Point3; 3], b: &[Point3; 3]) -> f64 {
    let mut best = f64::INFINITY;
    for (s, t) in [(a, b), (b, a)] {
        for k in 0..3 {
            best = best.min(segment_triangle(s[k], s[(k + 1) % 3], t).0);
            if best == 0.0 {
                return 0.0;
            }
        }
    }
    best
}

/// Where the line `p + t·d` crosses the triangle, as `t`: where it meets the triangle's plane,
/// if that point lies within the three sides.
pub(crate) fn ray_triangle(p: Point3, d: Vec3, a: Point3, b: Point3, c: Point3) -> Option<f64> {
    let (side_b, side_c) = (b - a, c - a);
    let normal = side_b.cross(side_c);
    let approach = d.dot(normal);
    // A ray in the triangle's plane, or within rounding of it, crosses nowhere it can place: its
    // approach is rounding, and the crossing it gives anywhere along the ray.
    let flat = 1e-12 * d.norm() * side_b.norm() * side_c.norm();
    if approach.abs() <= flat || approach.abs() < 1e-300 {
        return None;
    }
    let t = (a - p).dot(normal) / approach;
    // The meeting point as `a + u·side_b + v·side_c`: each weight is the share of the
    // triangle's area that the point spans with the other side.
    let from_a = p + d * t - a;
    let area = normal.dot(normal);
    let u = from_a.cross(side_c).dot(normal) / area;
    let v = side_b.cross(from_a).dot(normal) / area;
    (u >= 0.0 && v >= 0.0 && u + v <= 1.0).then_some(t)
}

pub(crate) fn on_segment(p: Point3, a: Point3, b: Point3) -> Point3 {
    let d = b - a;
    let l = d.dot(d);
    if l <= 0.0 {
        return a;
    }
    a + d * ((p - a).dot(d) / l).clamp(0.0, 1.0)
}

/// The closest points of two segments, and their distance. The squared distance is a convex
/// quadratic over the square of the two parameters, so its least value is where its gradient
/// vanishes, if that lies in the square, and otherwise on the square's rim: an end of one segment
/// against the other.
pub(crate) fn segments(p1: Point3, q1: Point3, p2: Point3, q2: Point3) -> (f64, Point3, Point3) {
    let (d1, d2, apart) = (q1 - p1, q2 - p2, p1 - p2);
    let (l1, l2, lean) = (d1.dot(d1), d2.dot(d2), d1.dot(d2));
    let mut best = (f64::INFINITY, p1, p2);
    let mut offer = |c1: Point3, c2: Point3| {
        let d = c1.dist(c2);
        if d < best.0 {
            best = (d, c1, c2);
        }
    };
    offer(p1, on_segment(p1, p2, q2));
    offer(q1, on_segment(q1, p2, q2));
    offer(on_segment(p2, p1, q1), p2);
    offer(on_segment(q2, p1, q1), q2);
    // Where both lines are nearest, when they are not parallel.
    let det = l1 * l2 - lean * lean;
    if det > 0.0 {
        let (pull1, pull2) = (d1.dot(apart), d2.dot(apart));
        let s = (lean * pull2 - pull1 * l2) / det;
        let t = (l1 * pull2 - lean * pull1) / det;
        if (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t) {
            offer(p1 + d1 * s, p2 + d2 * t);
        }
    }
    best
}

/// The point of triangle `t` nearest `p`: the foot of `p` on the triangle's plane where that
/// lies within the triangle, and otherwise the nearest point of its three sides.
pub(crate) fn on_triangle(p: Point3, t: &[Point3; 3]) -> Point3 {
    let [a, b, c] = *t;
    let (side_b, side_c, from_a) = (b - a, c - a, p - a);
    // The foot as `a + u·side_b + v·side_c`, from the two conditions that `p − foot` is square
    // to both sides.
    let (bb, bc, cc) = (side_b.dot(side_b), side_b.dot(side_c), side_c.dot(side_c));
    let det = bb * cc - bc * bc;
    if det > 0.0 {
        let (pb, pc) = (from_a.dot(side_b), from_a.dot(side_c));
        let u = (cc * pb - bc * pc) / det;
        let v = (bb * pc - bc * pb) / det;
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            return a + side_b * u + side_c * v;
        }
    }
    let mut best = on_segment(p, a, b);
    for (from, to) in [(b, c), (c, a)] {
        let q = on_segment(p, from, to);
        if q.dist(p) < best.dist(p) {
            best = q;
        }
    }
    best
}

/// The closest points of a segment and a triangle: where the segment crosses the triangle, or the
/// nearest of its ends to the triangle and its sides to the segment.
pub(crate) fn segment_triangle(p: Point3, q: Point3, t: &[Point3; 3]) -> (f64, Point3, Point3) {
    if let Some(x) = crossing(p, q, t) {
        return (0.0, x, x);
    }
    let mut best = (f64::INFINITY, p, t[0]);
    for e in [p, q] {
        let c = on_triangle(e, t);
        let d = e.dist(c);
        if d < best.0 {
            best = (d, e, c);
        }
    }
    for k in 0..3 {
        let x = segments(p, q, t[k], t[(k + 1) % 3]);
        if x.0 < best.0 {
            best = x;
        }
    }
    best
}

/// Where segment `pq` crosses triangle `t`, if it does.
pub(crate) fn crossing(p: Point3, q: Point3, t: &[Point3; 3]) -> Option<Point3> {
    let d = q - p;
    let s = ray_triangle(p, d, t[0], t[1], t[2])?;
    (0.0..=1.0).contains(&s).then(|| p + d * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hierarchy as a stable sort of each node's run along its longest axis builds it.
    fn sorted(boxes: Vec<Aabb>) -> Bvh {
        fn split(bvh: &mut Bvh, lo: usize, hi: usize) -> usize {
            let mut bb = Aabb::EMPTY;
            for &i in &bvh.order[lo..hi] {
                bb.union(&bvh.boxes[i]);
            }
            let at = bvh.nodes.len();
            bvh.nodes.push((bb, Node::Leaf(lo, hi)));
            if hi - lo <= 4 {
                return at;
            }
            let ext = bb.max - bb.min;
            let axis = if ext.x >= ext.y && ext.x >= ext.z {
                0
            } else if ext.y >= ext.z {
                1
            } else {
                2
            };
            let key = |b: &Aabb| ((b.min + b.max) * 0.5).to_array()[axis];
            let boxes = &bvh.boxes;
            bvh.order[lo..hi].sort_by(|&i, &j| key(&boxes[i]).total_cmp(&key(&boxes[j])));
            let mid = (lo + hi) / 2;
            let l = split(bvh, lo, mid);
            let r = split(bvh, mid, hi);
            bvh.nodes[at].1 = Node::Inner(l, r);
            at
        }
        let mut bvh = Bvh {
            order: (0..boxes.len()).collect(),
            boxes,
            nodes: Vec::new(),
        };
        let n = bvh.order.len();
        if n > 0 {
            split(&mut bvh, 0, n);
        }
        bvh
    }

    #[test]
    fn selecting_each_half_builds_the_tree_a_stable_sort_builds() {
        // Boxes on a coarse grid, so many centres tie along every axis and the order the ties
        // are left in decides the halves below.
        let mut x: u64 = 987654321;
        let mut next = |n: u64| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) % n) as f64
        };
        for (count, cells) in [(0, 1), (1, 1), (4, 2), (5, 2), (37, 3), (500, 4), (3000, 7)] {
            for stretch in [[1.0, 1.0, 1.0], [1.0, 5.0, 0.25], [0.0, 1.0, 1.0]] {
                let boxes: Vec<Aabb> = (0..count)
                    .map(|_| {
                        let p = Point3::new(
                            next(cells) * stretch[0],
                            next(cells) * stretch[1],
                            next(cells) * stretch[2],
                        );
                        let mut bb = Aabb::EMPTY;
                        bb.add(p);
                        bb.add(p + Vec3::new(next(3), next(3), next(3)) * 0.5);
                        bb
                    })
                    .collect();
                let (want, got) = (sorted(boxes.clone()), Bvh::of_boxes(boxes));
                assert_eq!(got.order, want.order, "{count} boxes on {cells} cells");
                assert_eq!(got.nodes, want.nodes, "{count} boxes on {cells} cells");
            }
        }
    }

    #[test]
    fn the_least_pair_is_the_least_of_every_pair() {
        // Segments scattered by a fixed linear congruential sequence, against every pair.
        let mut x: u64 = 12345;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64 * 10.0
        };
        let mut segs = |n: usize, off: f64| -> Vec<[Point3; 2]> {
            (0..n)
                .map(|_| {
                    let p = Vec3::new(next() + off, next(), next());
                    [p, p + Vec3::new(next(), next(), next()) * 0.2]
                })
                .collect()
        };
        for off in [0.0, 4.0, 30.0] {
            let (a, b) = (segs(300, 0.0), segs(200, off));
            let boxes = |s: &[[Point3; 2]]| -> Vec<Aabb> {
                s.iter()
                    .map(|s| {
                        let mut bb = Aabb::EMPTY;
                        s.iter().for_each(|&p| bb.add(p));
                        bb
                    })
                    .collect()
            };
            let (ta, tb) = (Bvh::of_boxes(boxes(&a)), Bvh::of_boxes(boxes(&b)));
            let dist = |i: usize, j: usize| segments(a[i][0], a[i][1], b[j][0], b[j][1]).0;
            let every = (0..a.len())
                .flat_map(|i| (0..b.len()).map(move |j| (i, j)))
                .map(|(i, j)| dist(i, j))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(ta.least(&tb, dist), every, "offset {off}");
        }
    }

    #[test]
    fn a_segment_in_a_tilted_triangle_s_plane_does_not_cross_it_from_afar() {
        // Two gears' bottom faces, coplanar, tilted half a radian: a side of one's mesh triangle
        // lies in the other's plane, which rounding puts a hair off, about 4.9 from it.
        let tri = [
            Vec3::new(14.508628775221682, -16.768516875320916, -3.068077540287854),
            Vec3::new(23.328340064120123, -3.3624006213025357, 4.255717148417697),
            Vec3::new(15.411143728305284, -16.19447442838339, -2.754476722249861),
        ];
        let other = [
            Vec3::new(55.55756088708297, -0.7777271174955624, 5.667730718980721),
            Vec3::new(55.55756088708297, -3.2222728825044435, 4.3322692810192756),
            Vec3::new(28.176531602590885, -3.004352420640489, 4.451319771923463),
        ];
        for (s, t) in [(&tri, &other), (&other, &tri)] {
            for k in 0..3 {
                let (p, q) = (s[k], s[(k + 1) % 3]);
                assert!(crossing(p, q, t).is_none(), "side {k} crosses");
                assert!(segment_triangle(p, q, t).0 > 4.0, "side {k} is near");
            }
        }
    }
}
