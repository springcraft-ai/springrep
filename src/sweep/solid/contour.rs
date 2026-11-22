//! Characteristic curves in closed form. On the surfaces a rigid velocity `ω × p + v` meets in a
//! simple way, the rate `g = n·V` is simple in one parameter: affine in both on a plane; affine in
//! `v` on a cylinder, cone or extrusion, whose normal is the same all along `v`; and
//! `a(u)·cos v + b(u)·sin v` on a sphere or torus. Each solution for the other parameter is a branch
//! of the characteristic set over the whole surface, whose points are exact; a branch is clipped
//! to a face by the face's winding number along it, and its runs inside the face are the face's
//! characteristic curves.

use super::grazing::Rate;
use crate::{
    boolean::domain::Domain,
    geom::Surface,
    math::{Point3, Vec2},
};

/// How the rate varies over a surface whose characteristic set has a closed form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Form {
    /// Affine in `(u, v)`: a plane.
    Affine,
    /// `n(u) + v·d(u)`: a cylinder, a cone or an extrusion.
    Linear,
    /// `a(u)·cos v + b(u)·sin v`: a sphere or a torus.
    Harmonic,
}

pub(super) fn form(s: &Surface) -> Option<Form> {
    match s {
        Surface::Plane(_) => Some(Form::Affine),
        Surface::Cylinder(_) | Surface::Cone(_) | Surface::Extrusion(_) => Some(Form::Linear),
        Surface::Sphere(_) | Surface::Torus(_) => Some(Form::Harmonic),
        _ => None,
    }
}

/// A branch of the characteristic set as a map from its own parameter to surface parameters.
#[derive(Clone, Copy, Debug)]
pub(super) enum Branch {
    /// `v = −n(u)/d(u)` on a [`Form::Linear`] surface.
    Ratio,
    /// The solution of `a cos v + b sin v = 0` at the angle `atan2(b, a) + shift`, on a
    /// [`Form::Harmonic`] surface, taken into the face's window of `v`.
    Angle { shift: f64 },
    /// The line `u = at`, all of whose points the rate vanishes on.
    AtU { at: f64 },
    /// The line `origin + s·dir` in the parameter plane, on a plane.
    Line { origin: Vec2, dir: Vec2 },
}

/// A run of a branch inside a face, from parameter `s.0` to `s.1`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Run {
    pub branch: Branch,
    pub s: (f64, f64),
}

/// The closed form of one face's rate: its window and coefficients.
pub(super) struct Closed<'a> {
    pub rate: &'a Rate<'a>,
    pub form: Form,
    pub lo: Vec2,
    pub hi: Vec2,
}

impl Closed<'_> {
    /// `(n, d)` of a linear form at `u`, from the rate at the window's two ends in `v`.
    fn linear(&self, u: f64) -> (f64, f64) {
        let (v0, v1) = (self.lo.y, self.hi.y.max(self.lo.y + 1.0));
        let (g0, g1) = (
            self.rate.at(Vec2::new(u, v0)),
            self.rate.at(Vec2::new(u, v1)),
        );
        let d = (g1 - g0) / (v1 - v0);
        (g0 - v0 * d, d)
    }

    /// `(a, b)` of a harmonic form at `u`.
    fn harmonic(&self, u: f64) -> (f64, f64) {
        (
            self.rate.at(Vec2::new(u, 0.0)),
            self.rate.at(Vec2::new(u, std::f64::consts::FRAC_PI_2)),
        )
    }

    /// Where `b` puts its point at `s`, in surface parameters: `None` where it has none.
    pub(super) fn uv(&self, b: Branch, s: f64) -> Option<Vec2> {
        match b {
            Branch::Ratio => {
                let (n, d) = self.linear(s);
                let v = -n / d;
                v.is_finite().then(|| Vec2::new(s, v))
            }
            Branch::Angle { shift } => {
                let (a, bb) = self.harmonic(s);
                if a == 0.0 && bb == 0.0 {
                    return None;
                }
                let v = bb.atan2(a) + shift;
                Some(Vec2::new(s, self.in_window_v(v)))
            }
            Branch::AtU { at } => Some(Vec2::new(at, s)),
            Branch::Line { origin, dir } => Some(origin + dir * s),
        }
    }

    /// `v` moved by whole periods into the face's window, where `v` is periodic.
    fn in_window_v(&self, v: f64) -> f64 {
        match self.rate.surface.v_period() {
            Some(p) => {
                let k = ((v - self.lo.y) / p).floor();
                v - k * p
            }
            None => {
                // A sphere's latitude: the solution a half turn away is the same point's other
                // reading, outside the range.
                let mut v = v;
                while v > std::f64::consts::FRAC_PI_2 + 1e-12 {
                    v -= std::f64::consts::PI;
                }
                while v < -std::f64::consts::FRAC_PI_2 - 1e-12 {
                    v += std::f64::consts::PI;
                }
                v
            }
        }
    }

    /// The branches over the face's window, each with the range of its own parameter.
    pub(super) fn branches(&self, eps: f64) -> Vec<(Branch, (f64, f64))> {
        let (lo, hi) = (self.lo, self.hi);
        let mut out = Vec::new();
        match self.form {
            Form::Affine => {
                let c = self.rate.at(lo);
                let gu = self.rate.at(Vec2::new(hi.x, lo.y)) - c;
                let gv = self.rate.at(Vec2::new(lo.x, hi.y)) - c;
                let (du, dv) = (hi.x - lo.x, hi.y - lo.y);
                // g = c + gu·(u − lo.u)/du + gv·(v − lo.v)/dv.
                let grad = Vec2::new(gu / du, gv / dv);
                let g2 = grad.dot(grad);
                if g2.sqrt() * (du + dv) <= eps {
                    return out;
                }
                let origin = lo - grad * (c / g2);
                let dir = Vec2::new(-grad.y, grad.x) / g2.sqrt();
                // The line's stretch across the window, padded.
                let reach = (hi - lo).norm();
                let mid = (lo + hi) * 0.5;
                let s_mid = (mid - origin).dot(dir);
                out.push((Branch::Line { origin, dir }, (s_mid - reach, s_mid + reach)));
            }
            Form::Linear => {
                // Where the rate does not change along `v` it vanishes on whole lines.
                let d_max = self.largest(|u| self.linear(u).1);
                if d_max * (hi.y - lo.y) > eps {
                    out.push((Branch::Ratio, (lo.x, hi.x)));
                }
                for u in self.whole_lines(eps, d_max <= eps, |u| self.linear(u)) {
                    out.push((Branch::AtU { at: u }, (lo.y, hi.y)));
                }
            }
            Form::Harmonic => {
                let periodic = self.rate.surface.v_period().is_some();
                let b_max = self.largest(|u| self.harmonic(u).1);
                // On a sphere the two solutions are one point's two readings, and with no `b` the
                // rate vanishes on whole meridians; on a torus the two are the outer and inner
                // curves.
                let shifts: &[f64] = if periodic {
                    &[std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2]
                } else if b_max > eps {
                    &[std::f64::consts::FRAC_PI_2]
                } else {
                    &[]
                };
                for &shift in shifts {
                    out.push((Branch::Angle { shift }, (lo.x, hi.x)));
                }
                for u in self.whole_lines(eps, b_max <= eps, |u| self.harmonic(u)) {
                    out.push((Branch::AtU { at: u }, (lo.y, hi.y)));
                }
            }
        }
        out
    }

    /// The largest magnitude of `f` over the window's `u`, sampled.
    fn largest(&self, f: impl Fn(f64) -> f64) -> f64 {
        let n = 360;
        (0..=n)
            .map(|k| f(self.lo.x + (self.hi.x - self.lo.x) * k as f64 / n as f64).abs())
            .fold(0.0, f64::max)
    }

    /// The values of `u` in the window where both coefficients vanish, so the rate does all along
    /// the line `u = const`: the roots of either coefficient at which the other is zero too, since
    /// one may only touch zero there, or, where the second vanishes everywhere (`flat`), the roots
    /// of the first.
    fn whole_lines(&self, eps: f64, flat: bool, coef: impl Fn(f64) -> (f64, f64)) -> Vec<f64> {
        let n = 720;
        let (lo, hi) = (self.lo.x, self.hi.x);
        let at = |k: usize| lo + (hi - lo) * k as f64 / n as f64;
        let roots = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
            let mut out = Vec::new();
            let mut prev = f(at(0));
            if prev == 0.0 {
                out.push(lo);
            }
            for k in 1..=n {
                let cur = f(at(k));
                if cur == 0.0 {
                    out.push(at(k));
                } else if prev != 0.0 && cur.signum() != prev.signum() {
                    let (mut a, mut b) = (at(k - 1), at(k));
                    for _ in 0..80 {
                        let m = 0.5 * (a + b);
                        if f(m).signum() == prev.signum() {
                            a = m;
                        } else {
                            b = m;
                        }
                    }
                    out.push(0.5 * (a + b));
                }
                prev = cur;
            }
            out
        };
        let mut found = roots(&|u| coef(u).0);
        if !flat {
            found.extend(roots(&|u| coef(u).1));
        }
        found.sort_by(f64::total_cmp);
        let mut out: Vec<f64> = Vec::new();
        for u in found {
            let (a, b) = coef(u);
            let zero = a.abs() <= eps && (flat || b.abs() <= eps);
            if zero && out.last().is_none_or(|&w| (u - w).abs() > 1e-9) {
                out.push(u);
            }
        }
        out
    }

    /// The runs of `branch` over `range` inside the face: each stretch between where it crosses
    /// the face's boundary, the window's edge in a periodic parameter, or an end of the branch.
    /// `marks` are the branch's parameters at the roots it passes through on the boundary: a run
    /// from one is found however short it is, as where the branch runs nearly along `v`.
    pub(super) fn runs(
        &self,
        dom: &Domain,
        branch: Branch,
        range: (f64, f64),
        marks: &[f64],
    ) -> Vec<Run> {
        let n = 720;
        let mut stops: Vec<f64> = (0..=n)
            .map(|k| range.0 + (range.1 - range.0) * k as f64 / n as f64)
            .collect();
        stops.extend(marks.iter().filter(|&&s| s > range.0 && s < range.1));
        stops.sort_by(f64::total_cmp);
        // The branch is inside the face on one side of a mark: samples close in on it from both.
        let mut samples = stops.clone();
        for w in stops.windows(2) {
            for (mark, other) in [(w[0], w[1]), (w[1], w[0])] {
                if marks.contains(&mark) {
                    samples.extend((1..=40).map(|k| mark + (other - mark) * 0.5f64.powi(k)));
                }
            }
        }
        samples.sort_by(f64::total_cmp);
        samples.dedup();
        // A branch running along the boundary, as one on an edge the rate vanishes all along, is
        // no curve across the face.
        let inside = |s: f64| -> bool {
            self.uv(branch, s).is_some_and(|uv| {
                self.in_window(uv)
                    && dom.winding(uv) != 0
                    && !dom.on_boundary(self.rate.surface.eval(uv.x, uv.y))
            })
        };
        // Two neighbouring samples belong to one run only where the branch has no jump between
        // them: a periodic parameter carried round to the window's other edge.
        let joined = |s0: f64, s1: f64| -> bool {
            match (self.uv(branch, s0), self.uv(branch, s1)) {
                (Some(a), Some(b)) => self
                    .rate
                    .surface
                    .v_period()
                    .is_none_or(|p| (a.y - b.y).abs() < 0.25 * p),
                _ => false,
            }
        };
        let edge = |a: f64, b: f64, a_in: bool| -> f64 {
            // The last parameter from `a` toward `b` that keeps `a`'s side.
            let (mut x, mut y) = (a, b);
            for _ in 0..80 {
                let m = 0.5 * (x + y);
                if inside(m) == a_in && joined(x, m) {
                    x = m;
                } else {
                    y = m;
                }
            }
            if a_in { x } else { y }
        };
        let mut out = Vec::new();
        let mut start: Option<f64> = inside(samples[0]).then_some(samples[0]);
        for w in samples.windows(2) {
            let (s0, s1) = (w[0], w[1]);
            let (i0, i1) = (inside(s0), inside(s1));
            let joins = joined(s0, s1);
            if i0 && (!i1 || !joins) {
                let end = edge(s0, s1, true);
                if let Some(s) = start.take() {
                    out.push(Run {
                        branch,
                        s: (s, end),
                    });
                }
            }
            if i1 && (!i0 || !joins) {
                start = Some(edge(s1, s0, true));
            }
        }
        if let Some(s) = start {
            out.push(Run {
                branch,
                s: (s, range.1),
            });
        }
        out.retain(|r| r.s.1 > r.s.0);
        out
    }

    fn in_window(&self, uv: Vec2) -> bool {
        let pad = 1e-9 * (1.0 + (self.hi - self.lo).norm());
        uv.x >= self.lo.x - pad
            && uv.x <= self.hi.x + pad
            && uv.y >= self.lo.y - pad
            && uv.y <= self.hi.y + pad
    }

    pub(super) fn point(&self, b: Branch, s: f64) -> Option<(Point3, Vec2)> {
        let uv = self.uv(b, s)?;
        Some((self.rate.surface.eval(uv.x, uv.y), uv))
    }

    /// Where branch `b`, coming from the side of `u = at` that `side` gives the sign of, meets the
    /// whole line `u = at`, on which both coefficients vanish: there the branch's own reading is
    /// `0/0`, and it is the limit of the coefficients' ratio, their derivatives' ratio.
    pub(super) fn crossing(&self, b: Branch, at: f64, side: f64) -> Option<(Point3, Vec2)> {
        let h = 1e-6 * (self.hi.x - self.lo.x).max(1e-9);
        let rate = |f: &dyn Fn(f64) -> (f64, f64)| {
            let (p, m) = (f(at + h), f(at - h));
            ((p.0 - m.0) / (2.0 * h), (p.1 - m.1) / (2.0 * h))
        };
        let v = match b {
            Branch::Ratio => {
                let (dn, dd) = rate(&|u| self.linear(u));
                -dn / dd
            }
            Branch::Angle { shift } => {
                let (da, db) = rate(&|u| self.harmonic(u));
                self.in_window_v((side * db).atan2(side * da) + shift)
            }
            _ => return None,
        };
        v.is_finite().then(|| {
            let uv = Vec2::new(at, v);
            (self.rate.surface.eval(uv.x, uv.y), uv)
        })
    }

    /// The parameter of `b` at surface parameters `uv`, which lie on it.
    pub(super) fn param_of(&self, b: Branch, uv: Vec2) -> f64 {
        match b {
            Branch::Ratio | Branch::Angle { .. } => uv.x,
            Branch::AtU { .. } => uv.y,
            Branch::Line { origin, dir } => (uv - origin).dot(dir),
        }
    }
}
