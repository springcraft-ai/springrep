//! Boxes that hold a spline curve piece by piece, from each knot span's control points, and a tree
//! of them for finding which pieces of one curve come near another.
//!
//! A B-spline's piece over a knot span lies in the convex hull of the span's `degree + 1` control
//! points, and so does a rational one's, its weights being positive: the box of those points
//! holds the piece, with no sampling.

use crate::{
    geom::Curve,
    math::{Aabb, Point3},
};

/// The pieces of `c` over `range`, each a parameter interval and a box holding the curve over it;
/// `None` for a curve with no control points, or an open one whose range runs past its ends, where
/// it extends beyond its hull.
pub(crate) fn pieces(c: &Curve, range: (f64, f64)) -> Option<Vec<(f64, f64, Aabb)>> {
    let (degree, knots, ctrl, period): (usize, &[f64], &[Point3], Option<f64>) = match c {
        Curve::Nurbs(n) => (n.degree(), n.knots(), n.control_points(), n.period()),
        _ => {
            let b = c.spline()?;
            (b.degree(), b.knots(), b.control_points(), b.period())
        }
    };
    let (a, b) = (knots[degree], knots[ctrl.len()]);
    let shifts: Vec<f64> = match period {
        Some(p) if p > 0.0 => {
            let k0 = ((range.0 - b) / p).floor() as i64;
            let k1 = ((range.1 - a) / p).ceil() as i64;
            (k0..=k1).map(|k| k as f64 * p).collect()
        }
        _ => {
            // A range that ends a rounding past the knots', as a path's whole number of turns
            // does, is the curve's own.
            let slack = 1e-12 * (b - a).abs().max(1.0);
            if range.0 < a - slack || range.1 > b + slack {
                return None;
            }
            vec![0.0]
        }
    };
    let mut out = Vec::new();
    for shift in shifts {
        for i in degree..ctrl.len() {
            let (t0, t1) = (knots[i] + shift, knots[i + 1] + shift);
            if t1 <= t0 || t1 < range.0 || t0 > range.1 {
                continue;
            }
            let mut bb = Aabb::EMPTY;
            for &q in &ctrl[i - degree..=i] {
                bb.add(q);
            }
            out.push((t0.max(range.0), t1.min(range.1), bb));
        }
    }
    (!out.is_empty()).then_some(out)
}

/// The parameter intervals of `c1`'s pieces over `r1`, in order, whose boxes come within `reach`
/// of a box of `c2`'s over `r2`: away from them `c1` is farther than `reach` from `c2`. `None`
/// where either curve has no hulls.
pub(crate) fn near(
    c1: &Curve,
    r1: (f64, f64),
    c2: &Curve,
    r2: (f64, f64),
    reach: f64,
) -> Option<Vec<(f64, f64)>> {
    let (p1, p2) = (pieces(c1, r1)?, pieces(c2, r2)?);
    let t1 = Tree::new(p1.iter().map(|p| p.2).collect());
    let t2 = Tree::new(p2.iter().map(|p| p.2).collect());
    let marks = t1.near(&t2, reach);
    Some(
        p1.iter()
            .zip(marks)
            .filter(|(_, m)| *m)
            .map(|(p, _)| (p.0, p.1))
            .collect(),
    )
}

/// The gap between two boxes, as the largest gap along an axis: no more than the true distance.
pub(crate) fn gap(a: &Aabb, b: &Aabb) -> f64 {
    (0..3)
        .map(|k| (a.min[k] - b.max[k]).max(b.min[k] - a.max[k]))
        .fold(0.0, f64::max)
}

/// A binary tree over a curve's consecutive pieces, each node boxing the pieces under it.
pub(crate) struct Tree {
    boxes: Vec<Aabb>,
    nodes: Vec<Node>,
}

struct Node {
    bb: Aabb,
    lo: usize,
    hi: usize,
    kids: Option<(usize, usize)>,
}

const LEAF: usize = 4;

impl Tree {
    pub(crate) fn new(boxes: Vec<Aabb>) -> Tree {
        let mut t = Tree {
            boxes,
            nodes: Vec::new(),
        };
        if !t.boxes.is_empty() {
            t.build(0, t.boxes.len());
        }
        t
    }

    fn build(&mut self, lo: usize, hi: usize) -> usize {
        let at = self.nodes.len();
        self.nodes.push(Node {
            bb: Aabb::EMPTY,
            lo,
            hi,
            kids: None,
        });
        let mut bb = Aabb::EMPTY;
        if hi - lo <= LEAF {
            for b in &self.boxes[lo..hi] {
                bb.union(b);
            }
        } else {
            let mid = (lo + hi) / 2;
            let (l, r) = (self.build(lo, mid), self.build(mid, hi));
            bb = self.nodes[l].bb;
            bb.union(&self.nodes[r].bb);
            self.nodes[at].kids = Some((l, r));
        }
        self.nodes[at].bb = bb;
        at
    }

    /// For each of this tree's boxes, whether some box of `other` comes within `reach` of it.
    pub(crate) fn near(&self, other: &Tree, reach: f64) -> Vec<bool> {
        let mut out = vec![false; self.boxes.len()];
        if !self.nodes.is_empty() && !other.nodes.is_empty() {
            self.near_from(0, other, 0, reach, &mut out);
        }
        out
    }

    fn near_from(&self, a: usize, other: &Tree, b: usize, reach: f64, out: &mut [bool]) {
        let (na, nb) = (&self.nodes[a], &other.nodes[b]);
        if gap(&na.bb, &nb.bb) > reach || out[na.lo..na.hi].iter().all(|&x| x) {
            return;
        }
        match (na.kids, nb.kids) {
            (None, None) => {
                let mine = &self.boxes[na.lo..na.hi];
                for (o, bb) in out[na.lo..na.hi].iter_mut().zip(mine) {
                    if !*o {
                        *o = other.boxes[nb.lo..nb.hi]
                            .iter()
                            .any(|q| gap(bb, q) <= reach);
                    }
                }
            }
            (Some((l, r)), None) => {
                self.near_from(l, other, b, reach, out);
                self.near_from(r, other, b, reach, out);
            }
            (None, Some((l, r))) => {
                self.near_from(a, other, l, reach, out);
                self.near_from(a, other, r, reach, out);
            }
            (Some((l, r)), Some((ol, or))) => {
                if na.hi - na.lo >= nb.hi - nb.lo {
                    self.near_from(l, other, b, reach, out);
                    self.near_from(r, other, b, reach, out);
                } else {
                    self.near_from(a, other, ol, reach, out);
                    self.near_from(a, other, or, reach, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geom::{BSpline, Line},
        math::Vec3,
    };

    #[test]
    fn each_piece_lies_in_its_box() {
        // A cubic through points on a wave: every sample of each piece falls in that piece's box.
        let pts: Vec<Vec3> = (0..12)
            .map(|i| Vec3::new(i as f64, (i as f64).sin(), 0.5 * (i % 3) as f64))
            .collect();
        let c = Curve::BSpline(BSpline::interpolate(&pts).unwrap());
        let (a, b) = c.domain();
        let ps = pieces(&c, (a + 0.3, b - 0.2)).unwrap();
        assert_eq!(ps.first().unwrap().0, a + 0.3);
        assert_eq!(ps.last().unwrap().1, b - 0.2);
        for (t0, t1, bb) in ps {
            for k in 0..=50 {
                let p = c.eval(t0 + (t1 - t0) * k as f64 / 50.0);
                assert!((0..3).all(|j| p[j] >= bb.min[j] - 1e-12 && p[j] <= bb.max[j] + 1e-12));
            }
        }
        // Past an open curve's end it extends beyond its hull; a line has no control points. A
        // range a rounding past the end, as a path's whole number of turns gives, is the curve's.
        assert!(pieces(&c, (a - 1.0, b)).is_none());
        let rounded = b + 4.0 * f64::EPSILON * b.abs().max(1.0);
        assert!(rounded > b);
        assert_eq!(pieces(&c, (a, rounded)).unwrap().last().unwrap().1, b);
        let line = Curve::Line(Line {
            origin: Vec3::ZERO,
            dir: Vec3::X,
        });
        assert!(pieces(&line, (0.0, 1.0)).is_none());
    }

    #[test]
    fn the_tree_marks_the_boxes_within_reach() {
        // Unit boxes along x at 0, 2, 4, …, 18 against one box at x 7.5 to 8.5: the boxes at 6
        // and 8 reach it, 6 only from 0.5 away.
        let unit = |x: f64| Aabb {
            min: Vec3::new(x, 0.0, 0.0),
            max: Vec3::new(x + 1.0, 1.0, 1.0),
        };
        let a = Tree::new((0..10).map(|i| unit(2.0 * i as f64)).collect());
        let b = Tree::new(vec![unit(7.5)]);
        let want = |reach: f64, hits: &[usize]| {
            let got = a.near(&b, reach);
            let marked: Vec<usize> = (0..10).filter(|&i| got[i]).collect();
            assert_eq!(marked, hits, "reach {reach}");
        };
        want(0.0, &[4]);
        want(0.5, &[3, 4]);
        want(1.5, &[3, 4, 5]);
    }
}
