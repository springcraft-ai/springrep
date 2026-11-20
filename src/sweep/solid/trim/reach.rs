//! Which points the moving body reaches into: a point lies inside the swept solid where, at some
//! `τ`, the body moved there holds it, which is read along the point's path through the body's own
//! frame, `M(τ)⁻¹·p` (the inverted trajectory of Ilies and Shapiro 1999 and Erdim and Ilies 2008),
//! as the deepest the body holds it there.

use super::super::motion::Motion;
use super::section::Section;
use crate::{
    math::{Point3, Transform},
    measure,
    topo::Brep,
};

/// The moving body, ready to say how deep it reaches each point.
pub(crate) struct Reach<'a> {
    section: Section,
    motion: &'a Motion,
    span: (f64, f64),
    taus: Vec<f64>,
    backs: Vec<Transform>,
    /// The body's size: the diagonal of its box.
    pub size: f64,
    /// How deep a point must lie to count as reached: a hair, so a point on the swept solid's
    /// boundary, which the body touches but never holds, does not.
    pub hair: f64,
}

impl<'a> Reach<'a> {
    /// The reach of `body` as `motion` carries it over `span`, where the body's depth has a closed
    /// form; else `None`.
    pub(crate) fn new(body: &Brep, motion: &'a Motion, span: (f64, f64)) -> Option<Reach<'a>> {
        let section = Section::of(body)?;
        let size = measure::bbox(body).diagonal();
        // Samples close enough that the depth along a path changes little between them: a step
        // moves no point of the body more than a fiftieth of its size.
        let corners = {
            let b = measure::bbox(body);
            let (lo, hi) = (b.min, b.max);
            (0..8)
                .map(|k| {
                    Point3::new(
                        if k & 1 == 0 { lo.x } else { hi.x },
                        if k & 2 == 0 { lo.y } else { hi.y },
                        if k & 4 == 0 { lo.z } else { hi.z },
                    )
                })
                .collect::<Vec<_>>()
        };
        let probe = 1024;
        let poses: Vec<Transform> = (0..=probe)
            .map(|k| {
                motion
                    .rate(span.0 + (span.1 - span.0) * k as f64 / probe as f64)
                    .0
            })
            .collect();
        let travel: f64 = poses
            .windows(2)
            .map(|w| {
                corners
                    .iter()
                    .map(|&c| w[0].point(c).dist(w[1].point(c)))
                    .fold(0.0, f64::max)
            })
            .sum();
        let n = ((50.0 * travel / size.max(1e-300)).ceil() as usize).clamp(64, 4096);
        let taus: Vec<f64> = (0..=n)
            .map(|k| span.0 + (span.1 - span.0) * k as f64 / n as f64)
            .collect();
        let backs = taus.iter().map(|&t| motion.rate(t).0.inverse()).collect();
        Some(Reach {
            section,
            motion,
            span,
            taus,
            backs,
            size,
            hair: 1e-10 * size.max(1.0),
        })
    }

    fn depth_at(&self, p: Point3, tau: f64) -> f64 {
        self.section
            .depth(self.motion.rate(tau).0.inverse().point(p))
    }

    /// The most of the depth of `p` over `[a, b]`, and where: the best of a few samples, then
    /// golden-section search about it. Where two peaks share the bracket (the body's own touch at
    /// an end of the motion and another just inside it), the samples pick the higher.
    fn most(&self, p: Point3, a: f64, b: f64) -> (f64, f64) {
        let m = 8;
        let xs: Vec<f64> = (0..=m).map(|k| a + (b - a) * k as f64 / m as f64).collect();
        let (j, d) = (0..=m)
            .map(|k| (k, self.depth_at(p, xs[k])))
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap_or((0, f64::NEG_INFINITY));
        let mut best = (d, xs[j]);
        let keep = |f: f64, x: f64, best: &mut (f64, f64)| {
            if f > best.0 {
                *best = (f, x);
            }
        };
        let g = 0.5 * (5f64.sqrt() - 1.0);
        let (mut a, mut b) = (xs[j.saturating_sub(1)], xs[(j + 1).min(m)]);
        let mut x1 = b - g * (b - a);
        let mut x2 = a + g * (b - a);
        let mut f1 = self.depth_at(p, x1);
        let mut f2 = self.depth_at(p, x2);
        keep(f1, x1, &mut best);
        keep(f2, x2, &mut best);
        let tol = 1e-9 * (self.span.1 - self.span.0);
        while b - a > tol {
            if f1 >= f2 {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - g * (b - a);
                f1 = self.depth_at(p, x1);
                keep(f1, x1, &mut best);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + g * (b - a);
                f2 = self.depth_at(p, x2);
                keep(f2, x2, &mut best);
            }
        }
        best
    }

    /// The deepest the body reaches `p` over the whole motion, and where. Each sampled peak of
    /// the depth along the point's path that may reach `above` is searched for its top: a smooth
    /// peak rises above its highest sample by less than that sample's larger step to a neighbour.
    /// With `above`, the search stops at the first depth past it.
    fn deepest_past(&self, p: Point3, above: Option<f64>) -> (f64, f64) {
        let vals: Vec<f64> = self
            .backs
            .iter()
            .map(|b| self.section.depth(b.point(p)))
            .collect();
        let n = vals.len() - 1;
        let mut best =
            (0..=n)
                .map(|k| (vals[k], self.taus[k]))
                .fold(
                    (f64::NEG_INFINITY, 0.0),
                    |x, y| if y.0 > x.0 { y } else { x },
                );
        if above.is_some_and(|a| best.0 > a) {
            return best;
        }
        let floor = above.unwrap_or(best.0);
        for k in 0..=n {
            let left = if k == 0 { 0.0 } else { vals[k] - vals[k - 1] };
            let right = if k == n { 0.0 } else { vals[k] - vals[k + 1] };
            if left < 0.0 || right < 0.0 || vals[k] + 2.0 * left.max(right) < floor.min(best.0) {
                continue;
            }
            let found = self.most(p, self.taus[k.saturating_sub(1)], self.taus[(k + 1).min(n)]);
            if found.0 > best.0 {
                best = found;
            }
            if above.is_some_and(|a| best.0 > a) {
                return best;
            }
        }
        best
    }

    /// The deepest the body reaches `p` over the whole motion.
    pub(crate) fn deepest(&self, p: Point3) -> f64 {
        self.deepest_past(p, None).0
    }

    /// Where along the motion the body reaches `p` deepest.
    pub(crate) fn deepest_place(&self, p: Point3) -> f64 {
        self.deepest_past(p, None).1
    }

    /// How deep the body lies at `p` where the motion starts (`end` false) or stops.
    pub(crate) fn depth_at_end(&self, p: Point3, end: bool) -> f64 {
        self.depth_at(p, if end { self.span.1 } else { self.span.0 })
    }

    /// The motion's span of `τ`.
    pub(crate) fn span(&self) -> (f64, f64) {
        self.span
    }

    /// Whether the body reaches into `p`: whether `p` lies inside the swept solid.
    pub(crate) fn holds(&self, p: Point3) -> bool {
        self.deepest_past(p, Some(self.hair)).0 > self.hair
    }
}
