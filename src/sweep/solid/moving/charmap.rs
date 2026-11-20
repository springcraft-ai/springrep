//! Where a face of the body grazes the motion at each τ, as a curve of one coordinate `x`: on a
//! plane, the line `A x + B y + C = 0` by the place along it from its point nearest the face's
//! middle; on a cylinder, cone or extrusion, whose rate is affine along `v`, and on a torus, whose
//! rate is `a cos v + b sin v`, the surface's first parameter `u`, the curve's `v` at each `u` read
//! from that closed form (doc/design.md, "Finding the curves"). A torus grazes along two curves, half
//! a turn of `v` apart.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use super::Ctx;
use crate::sweep::solid::motion::Twist;
use crate::{
    Error, Result,
    geom::Surface,
    math::{Point3, Vec2, Vec3},
};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Line,
    Linear,
    /// The branch `atan2(−a, b)`, or the one half a turn of `v` on from it.
    Harmonic {
        half: bool,
    },
    /// On a cylinder or cone whose curves run nearly along its rulings, the curve by its `v`: at
    /// each `v` the rate is `a cos u + b sin u + c`, zero at `u = atan2(b, a) ± acos(−c / √(a² +
    /// b²))`, one branch each sign.
    Across {
        minus: bool,
    },
}

/// Whether the grazing curves of face `f`, a cylinder or cone, run nearly along its rulings: over
/// the span its curve's `v` lies on the face over less than a tenth of the face's `u`.
fn steep(ctx: &Ctx, f: usize) -> bool {
    let face = &ctx.faces[f];
    let (lo, hi) = (face.dom.lo, face.dom.hi);
    let (n, m) = (16, 64);
    let mut on = 0usize;
    let (mut change, mut size) = (0.0, 0.0);
    for i in 0..=n {
        let tau = ctx.span.0 + (ctx.span.1 - ctx.span.0) * i as f64 / n as f64;
        let tw = ctx.twist(tau);
        let rate = |u: f64, v: f64| {
            let p = face.surface.eval(u, v);
            face.surface.normal(u, v).dot(tw.at(p))
        };
        for j in 0..m {
            let u = lo.x + (hi.x - lo.x) * (j as f64 + 0.5) / m as f64;
            let (g0, g1) = (rate(u, lo.y), rate(u, hi.y));
            if g0 * g1 <= 0.0 {
                on += 1;
            }
            change += (g1 - g0).abs();
            size += g0.abs() + g1.abs();
        }
    }
    // A rate the same all along each ruling grazes along whole rulings.
    change <= 1e-9 * size || (on > 0 && (on as f64) < 0.1 * ((n + 1) * m) as f64)
}

/// The grazing curve of face `face` over the span.
pub(super) struct CharMap<'c, 'a> {
    ctx: &'c Ctx<'a>,
    face: usize,
    kind: Kind,
    centre: Vec2,
    /// At each end of the span where the motion keeps the face in its own surface, the rate of
    /// change of the velocity, whose grazing curve is the limit of the face's there.
    ends: [Option<Twist>; 2],
}

impl<'c, 'a> CharMap<'c, 'a> {
    /// The grazing curves of face `f`. A torus whose rate's two coefficients vanish together at
    /// some `u` and `τ`, where it grazes along a whole meridian and its curves' `v` turns once
    /// round about that place, is refused.
    pub(super) fn of(ctx: &'c Ctx<'a>, f: usize) -> Result<Vec<CharMap<'c, 'a>>> {
        let face = &ctx.faces[f];
        let (span, len) = (ctx.span, ctx.span.1 - ctx.span.0);
        let end = |held: bool, tau: f64, len: f64| {
            (held && super::held(ctx.body, ctx.motion, face.id, tau))
                .then(|| super::turning(ctx.motion, tau, len))
        };
        let ends = [
            end(ctx.bounds.held.0, span.0, len),
            end(ctx.bounds.held.1, span.1, -len),
        ];
        let kinds = match face.surface {
            Surface::Plane(_) => vec![Kind::Line],
            Surface::Cylinder(_) | Surface::Cone(_) if steep(ctx, f) => {
                vec![Kind::Across { minus: false }, Kind::Across { minus: true }]
            }
            Surface::Cylinder(_) | Surface::Cone(_) | Surface::Extrusion(_) => vec![Kind::Linear],
            Surface::Torus(_) => vec![
                Kind::Harmonic { half: false },
                Kind::Harmonic { half: true },
            ],
            _ => Vec::new(),
        };
        let maps: Vec<CharMap> = kinds
            .into_iter()
            .map(|kind| CharMap {
                ctx,
                face: f,
                kind,
                centre: (face.dom.lo + face.dom.hi) * 0.5,
                ends,
            })
            .collect();
        if let Some(m) = maps.first()
            && matches!(m.kind, Kind::Harmonic { .. })
            && m.meridian()
        {
            return Err(Error::Unsupported(
                "a sweep along a motion whose velocity changes on the body, of a torus that grazes \
                 along a whole meridian"
                    .into(),
            ));
        }
        Ok(maps)
    }

    /// The torus rate's coefficients `(a, b)` at `u` at `tau`.
    fn coefficients(&self, tau: f64, u: f64) -> Vec2 {
        Vec2::new(
            self.rate(tau, Vec2::new(u, 0.0)),
            self.rate(tau, Vec2::new(u, FRAC_PI_2)),
        )
    }

    /// Whether the coefficients wind round zero in some cell of a grid over the span and the face's
    /// `u`: the place where they vanish together.
    fn meridian(&self) -> bool {
        let (n, m) = (128, 64);
        let dom = &self.ctx.faces[self.face].dom;
        let (lo, hi) = (dom.lo.x, dom.hi.x);
        let span = self.ctx.span;
        let angle = |i: usize, j: usize| {
            let tau = span.0 + (span.1 - span.0) * i as f64 / n as f64;
            let u = lo + (hi - lo) * j as f64 / m as f64;
            let c = self.coefficients(tau, u);
            c.y.atan2(c.x)
        };
        let turn = |a: f64, b: f64| {
            let d = (b - a).rem_euclid(TAU);
            if d > PI { d - TAU } else { d }
        };
        let grid: Vec<Vec<f64>> = (0..=n)
            .map(|i| (0..=m).map(|j| angle(i, j)).collect())
            .collect();
        (0..n).any(|i| {
            (0..m).any(|j| {
                let ring = [
                    grid[i][j],
                    grid[i + 1][j],
                    grid[i + 1][j + 1],
                    grid[i][j + 1],
                ];
                let total: f64 = (0..4).map(|k| turn(ring[k], ring[(k + 1) % 4])).sum();
                total.abs() > PI
            })
        })
    }

    fn surface(&self) -> &Surface {
        &self.ctx.faces[self.face].surface
    }

    /// The rate the face moves outward at `uv` at `tau`.
    fn rate(&self, tau: f64, uv: Vec2) -> f64 {
        let s = self.surface();
        let p = s.eval(uv.x, uv.y);
        s.normal(uv.x, uv.y).dot(self.twist(tau).at(p)) * self.ctx.faces[self.face].out
    }

    /// The velocity at `tau`, or at an end where the face moves within its own surface, its rate
    /// of change: there the velocity is the rounding of terms that cancel, and its grazing curve
    /// would jump from the one just inside.
    fn twist(&self, tau: f64) -> Twist {
        let (span, len) = (self.ctx.span, self.ctx.span.1 - self.ctx.span.0);
        let hair = 1e-9 * len;
        match self.ends {
            [Some(tw), _] if tau <= span.0 + hair => tw,
            [_, Some(tw)] if tau >= span.1 - hair => tw,
            _ => self.ctx.motion.body_twist(tau),
        }
    }

    /// A plane's line at `tau`: its point nearest the face's middle and its direction.
    fn line(&self, tau: f64) -> Option<(Vec2, Vec2)> {
        let c = self.rate(tau, Vec2::ZERO);
        let a = self.rate(tau, Vec2::new(1.0, 0.0)) - c;
        let b = self.rate(tau, Vec2::new(0.0, 1.0)) - c;
        let l = a.hypot(b);
        let g = a * self.centre.x + b * self.centre.y + c;
        if l <= 1e-12 * (g.abs() + c.abs()) || l <= 1e-300 {
            return None;
        }
        let m = Vec2::new(a / l, b / l);
        Some((self.centre - m * (g / l), m.perp()))
    }

    /// The curve's point at `x` at `tau`, in the face's parameters.
    pub(super) fn uv(&self, tau: f64, x: f64) -> Option<Vec2> {
        match self.kind {
            Kind::Line => {
                let (foot, dir) = self.line(tau)?;
                Some(foot + dir * x)
            }
            Kind::Linear => {
                let dom = &self.ctx.faces[self.face].dom;
                let (v0, v1) = (dom.lo.y, dom.hi.y);
                let g0 = self.rate(tau, Vec2::new(x, v0));
                let g1 = self.rate(tau, Vec2::new(x, v1));
                if (g1 - g0).abs() <= 1e-300 {
                    return None;
                }
                Some(Vec2::new(x, v0 - g0 * (v1 - v0) / (g1 - g0)))
            }
            Kind::Harmonic { half } => {
                let c = self.coefficients(tau, x);
                if c.norm() <= 1e-300 {
                    return None;
                }
                Some(Vec2::new(
                    x,
                    (-c.x).atan2(c.y) + if half { PI } else { 0.0 },
                ))
            }
            Kind::Across { minus } => {
                let u = self.across(tau, x)?;
                Some(Vec2::new(if minus { u.1 } else { u.0 }, x))
            }
        }
    }

    /// At `v`, the two `u` the rate `a cos u + b sin u + c` of a cylinder or cone vanishes at.
    fn across(&self, tau: f64, v: f64) -> Option<(f64, f64)> {
        let g0 = self.rate(tau, Vec2::new(0.0, v));
        let g1 = self.rate(tau, Vec2::new(FRAC_PI_2, v));
        let g2 = self.rate(tau, Vec2::new(PI, v));
        let c = 0.5 * (g0 + g2);
        let (a, b) = (0.5 * (g0 - g2), g1 - c);
        let r = a.hypot(b);
        if r <= 1e-300 || c.abs() > r {
            return None;
        }
        let (mid, off) = (b.atan2(a), (-c / r).acos());
        Some((mid + off, mid - off))
    }

    /// Whether the face's point `uv` lies on this curve rather than the face's other one.
    pub(super) fn holds(&self, tau: f64, uv: Vec2) -> bool {
        let gap = |a: f64, b: f64| {
            let d = (a - b).rem_euclid(TAU);
            d.min(TAU - d)
        };
        match self.kind {
            Kind::Harmonic { .. } => self
                .uv(tau, uv.x)
                .is_some_and(|on| gap(uv.y, on.y) <= FRAC_PI_2),
            Kind::Across { minus } => self.across(tau, uv.y).is_some_and(|(p, m)| {
                let (mine, other) = if minus { (m, p) } else { (p, m) };
                gap(uv.x, mine) <= gap(uv.x, other)
            }),
            _ => true,
        }
    }

    /// The curve's point at `x` at `tau`, on the body.
    pub(super) fn point(&self, tau: f64, x: f64) -> Option<Point3> {
        let uv = self.uv(tau, x)?;
        Some(self.surface().eval(uv.x, uv.y))
    }

    /// The coordinate of the face's point `uv`, on the curve at `tau`.
    pub(super) fn coord(&self, tau: f64, uv: Vec2) -> Option<f64> {
        match self.kind {
            Kind::Line => {
                let (foot, dir) = self.line(tau)?;
                Some((uv - foot).dot(dir))
            }
            Kind::Linear | Kind::Harmonic { .. } => Some(uv.x),
            Kind::Across { .. } => Some(uv.y),
        }
    }

    /// The range of the coordinate the face spans.
    pub(super) fn range(&self) -> (f64, f64) {
        match self.kind {
            Kind::Line => {
                let s = 2.0 * self.ctx.size.max(1.0);
                (-s, s)
            }
            Kind::Linear | Kind::Harmonic { .. } => {
                let dom = &self.ctx.faces[self.face].dom;
                (dom.lo.x, dom.hi.x)
            }
            Kind::Across { .. } => {
                let dom = &self.ctx.faces[self.face].dom;
                (dom.lo.y, dom.hi.y)
            }
        }
    }

    /// The rate's change along the velocity at the curve's point `uv` at `tau`, less its change
    /// with `tau` there: the gradient of the rate along the way the curve's point on the body moves
    /// relative to the body, `V` plus the curve's own drift over the face (which takes up `−∂g/∂τ`
    /// as the rate stays zero on it). Positive where the swept face leaves the curve toward the
    /// side of the face the motion carries inward.
    pub(super) fn rise(&self, tau: f64, uv: Vec2) -> f64 {
        let dt = 1e-6 * (self.ctx.span.1 - self.ctx.span.0);
        let drift = (self.rate(tau + dt, uv) - self.rate(tau - dt, uv)) / (2.0 * dt);
        self.along_velocity(tau, uv) - drift
    }

    fn along_velocity(&self, tau: f64, uv: Vec2) -> f64 {
        let s = self.surface();
        let (p, su, sv) = s.d1(uv.x, uv.y);
        let vel = self.twist(tau).at(p);
        // The velocity in the face's parameters, by least squares on its two tangents.
        let (a, b, c) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let (r, t) = (vel.dot(su), vel.dot(sv));
        let det = a * c - b * b;
        if det.abs() <= 1e-300 {
            return 0.0;
        }
        let du = (c * r - b * t) / det;
        let dv = (a * t - b * r) / det;
        let h = 1e-6;
        let gu = (self.rate(tau, uv + Vec2::new(h, 0.0)) - self.rate(tau, uv - Vec2::new(h, 0.0)))
            / (2.0 * h);
        let gv = (self.rate(tau, uv + Vec2::new(0.0, h)) - self.rate(tau, uv - Vec2::new(0.0, h)))
            / (2.0 * h);
        du * gu + dv * gv
    }

    /// The face's outward normal at `uv`.
    pub(super) fn normal(&self, uv: Vec2) -> Vec3 {
        self.surface().normal(uv.x, uv.y) * self.ctx.faces[self.face].out
    }
}
