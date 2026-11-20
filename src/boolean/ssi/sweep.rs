//! Analytic pairs where one surface is swept by a family of lines or circles, each meeting the
//! other surface in at most two points: a cylinder's generators against another cylinder or a
//! sphere, and a torus's meridian circles against a plane or a sphere.
//!
//! Each family is laid out so that its member at angle `θ` depends on `c = cos θ` alone for how
//! far it crosses the other surface: the member's *margin*, in mm, is the least of two functions of
//! `c` that are each monotone on [-1, 1] (how far the member's nearest point lies inside the other
//! surface, and how far its farthest point lies outside it). So the margin has one maximum over `c`
//! and the contact is decided from three numbers, each a true distance: the margin at that maximum
//! and at `c = ±1`. A margin within [`tol::LINEAR`] of zero is zero, as a plane and a cylinder
//! within it touch; the branches are then evaluated in closed form along the family, never traced,
//! so a touching point comes back as a point and two branches through a node meet at it.

use std::f64::consts::{PI, TAU};

use super::{Ssi, SsiCurve};
use crate::{
    Result,
    geom::{Curve, Cylinder, IntersectionCurve, Plane, Surface, Torus},
    math::{Aabb, Point3, Vec3},
    numeric, tol,
};

/// A one-parameter family of lines or circles sweeping one surface.
trait Family {
    /// The member's two margins at `c = cos θ`, each monotone in `c`; it meets the other surface
    /// where both are at least zero. An unused margin is `+∞`.
    fn margins(&self, c: f64) -> [f64; 2];
    /// The member's meetings with the other surface at `θ`: `mid ± dir·√h2`, `h2` negative where it
    /// misses.
    fn chord(&self, theta: f64) -> (Point3, Vec3, f64);
    /// The member at `θ` as a curve, for a member that lies on the other surface.
    fn member(&self, theta: f64) -> Option<Curve>;
}

fn margin(f: &dyn Family, c: f64) -> f64 {
    let [m1, m2] = f.margins(c);
    m1.min(m2)
}

/// The `c` in [-1, 1] where the margin is largest: where its two monotone parts cross when they
/// run opposite ways, otherwise at an end.
fn peak(f: &dyn Family) -> f64 {
    let ([a0, b0], [a1, b1]) = (f.margins(-1.0), f.margins(1.0));
    let (da, db) = (a1 - a0, b1 - b0);
    let finite = a0.is_finite() && b0.is_finite();
    if finite && (da > 0.0) != (db > 0.0) && ((a0 - b0) > 0.0) != ((a1 - b1) > 0.0) {
        return numeric::bisect(
            &mut |c| {
                let [a, b] = f.margins(c);
                a - b
            },
            -1.0,
            1.0,
        );
    }
    if margin(f, 1.0) >= margin(f, -1.0) {
        1.0
    } else {
        -1.0
    }
}

/// How an arc of the family's angle ends.
#[derive(Clone, Copy, Debug)]
enum End {
    /// The two meetings join smoothly: the member turns tangent to the other surface there.
    Turn,
    /// The two branches cross where the surfaces touch; `h2` is the chord there, taken out along the
    /// arc so that both branches end exactly on the touching point.
    Node(f64),
    /// A cut in the middle of an arc whose ends are one node, so neither branch starts and ends on
    /// one point; the chord correction there is `h2`.
    Split(f64),
}

fn snap(m: f64) -> i8 {
    if m > tol::LINEAR {
        1
    } else if m < -tol::LINEAR {
        -1
    } else {
        0
    }
}

/// [`snap`] for a node at θ = 0 or π, where two branches are bent to end on one point: the bend
/// takes the node's margin out of their chords, which moves their points off the other surface by
/// up to the margin, and their fit holds half the tolerance on top of that. So a node's margin is
/// within half the tolerance; past it the members miss or cross there, by the margin's sign, and
/// the branches pass the point apart, as far apart as the chord the margin leaves.
fn node_snap(m: f64) -> i8 {
    match snap(m) {
        0 if m > 0.5 * tol::LINEAR => 1,
        0 if m < -0.5 * tol::LINEAR => -1,
        s => s,
    }
}

/// The meeting point of `f` at `θ` on branch `sheet` (±1, or 0 for the middle), with `corr`
/// subtracted from the squared half chord.
fn at(f: &dyn Family, theta: f64, sheet: f64, corr: f64) -> Point3 {
    let (mid, dir, h2) = f.chord(theta);
    if sheet == 0.0 {
        return mid;
    }
    mid + dir * (sheet * (h2 - corr).max(0.0).sqrt())
}

/// The meeting point at `θ` with the squared half chords `corrs` of the arc's nodes taken out
/// where the chord is as short as theirs, and less in proportion as it grows: a node's margin
/// within the tolerance but not zero leaves a chord there that the branches must close, and taking
/// the same amount out all along moves a point near the arc's middle along its member by about
/// `corr / 2√h2`, off the other surface by as much as the margin itself.
fn at_closed(f: &dyn Family, theta: f64, sheet: f64, corrs: [f64; 2]) -> Point3 {
    let h2 = f.chord(theta).2;
    let corr = corrs
        .iter()
        .map(|&c| if h2 > c { c * (c / h2) } else { c })
        .fold(0.0, f64::max);
    at(f, theta, sheet, corr)
}

struct Out {
    curves: Vec<SsiCurve>,
    points: Vec<Point3>,
    region: Aabb,
}

/// How many points a branch is read at to find where it passes through the region.
const READINGS: usize = 1024;

impl Out {
    fn fit(
        &mut self,
        sample: &dyn Fn(f64) -> Point3,
        range: (f64, f64),
        closed: bool,
        a: &Surface,
        b: &Surface,
    ) -> Result<()> {
        // Two cylinders whose axes are a hair off parallel meet in loops metres long, of which
        // only what passes through the region can lie on both faces: only that is fitted, each
        // stretch reaching one reading past it at each end. A branch between two readings lies
        // within their spacing of either, so the region is grown by the widest spacing.
        let s_at = |k: usize| range.0 + (range.1 - range.0) * k as f64 / READINGS as f64;
        let pts: Vec<Point3> = (0..=READINGS).map(|k| sample(s_at(k))).collect();
        let step = pts.windows(2).map(|w| w[0].dist(w[1])).fold(0.0, f64::max);
        let pad = Vec3::new(1.0, 1.0, 1.0) * (step + tol::LINEAR);
        let (lo, hi) = (self.region.min - pad, self.region.max + pad);
        // Round a loop the last reading is the first again.
        let n = if closed { READINGS } else { READINGS + 1 };
        let inside: Vec<bool> = pts[..n]
            .iter()
            .map(|p| (0..3).all(|i| p[i] >= lo[i] && p[i] <= hi[i]))
            .collect();
        if self.region.is_empty() || inside.iter().all(|&x| x) {
            return self.fit_whole(sample, range, closed, a, b);
        }
        // Stretches of readings inside, each fitted from the reading before it to the one after:
        // round a loop they are read from a reading outside, and may run past the range's end.
        let start = if closed {
            inside.iter().position(|&x| !x).unwrap_or(0)
        } else {
            0
        };
        let at = |k: usize| {
            let k = if closed { k } else { k.min(READINGS) };
            range.0 + (range.1 - range.0) * k as f64 / READINGS as f64
        };
        let mut k = start;
        while k < start + n {
            if !inside[k % n] {
                k += 1;
                continue;
            }
            let first = k;
            while k < start + n && inside[k % n] {
                k += 1;
            }
            let from = if first == 0 { range.0 } else { at(first - 1) };
            self.fit_whole(sample, (from, at(k)), false, a, b)?;
        }
        Ok(())
    }

    fn fit_whole(
        &mut self,
        sample: &dyn Fn(f64) -> Point3,
        range: (f64, f64),
        closed: bool,
        a: &Surface,
        b: &Surface,
    ) -> Result<()> {
        let c = IntersectionCurve::fit_sampled(sample, range, a.clone(), b.clone(), closed, 24)?;
        let range = c.approx.domain();
        self.curves.push(SsiCurve {
            curve: Curve::Intersection(Box::new(c)),
            range,
            closed,
            tangent: false,
        });
        Ok(())
    }

    /// The branches over the arc `θ` in [t0, t1].
    fn arc(
        &mut self,
        f: &dyn Family,
        (t0, t1): (f64, f64),
        (k0, k1): (End, End),
        (a, b): (&Surface, &Surface),
    ) -> Result<()> {
        if let (End::Turn, End::Turn) = (k0, k1) {
            let (m, h) = (0.5 * (t0 + t1), 0.5 * (t1 - t0));
            let sample = |phi: f64| {
                let sheet = if phi.sin() >= 0.0 { 1.0 } else { -1.0 };
                at(f, m - h * phi.cos(), sheet, 0.0)
            };
            return self.fit(&sample, (0.0, TAU), true, a, b);
        }
        if let (End::Node(h0), End::Node(_)) = (k0, k1)
            && t1 - t0 > 1.5 * PI
        {
            let m = 0.5 * (t0 + t1);
            for sheet in [1.0, -1.0] {
                self.points.push(at_closed(f, m, sheet, [h0, h0]));
            }
            self.arc(f, (t0, m), (k0, End::Split(h0)), (a, b))?;
            return self.arc(f, (m, t1), (End::Split(h0), k1), (a, b));
        }
        let corr = |k: End| match k {
            End::Node(h2) | End::Split(h2) => h2,
            End::Turn => 0.0,
        };
        let (c0, c1) = (corr(k0), corr(k1));
        let theta = |s: f64| match (k0, k1) {
            (End::Turn, _) => t0 + (t1 - t0) * (1.0 - (0.5 * PI * s).cos()),
            (_, End::Turn) => t0 + (t1 - t0) * (0.5 * PI * s).sin(),
            _ => t0 + (t1 - t0) * s,
        };
        for (k, t) in [(k0, t0), (k1, t1)] {
            if let End::Turn = k {
                self.points.push(at(f, t, 0.0, 0.0));
            }
        }
        for sheet in [1.0, -1.0] {
            let sample = |s: f64| at_closed(f, theta(s), sheet, [c0, c1]);
            self.fit(&sample, (0.0, 1.0), false, a, b)?;
        }
        Ok(())
    }
}

/// The intersection of `a` and `b` where `f` sweeps one of them and `other` is the other.
fn solve(f: &dyn Family, other: &Surface, a: &Surface, b: &Surface, region: &Aabb) -> Result<Ssi> {
    let c_peak = peak(f);
    let m_peak = margin(f, c_peak);
    let mut out = Out {
        curves: Vec::new(),
        points: Vec::new(),
        region: *region,
    };
    let thetas = |c: f64| {
        let t = c.clamp(-1.0, 1.0).acos();
        if t == 0.0 || t == PI {
            vec![t]
        } else {
            vec![t, -t]
        }
    };
    match snap(m_peak) {
        -1 => {}
        0 => {
            let [m1, m2] = f.margins(c_peak);
            for t in thetas(c_peak) {
                // A member within the tolerance of the other surface all along lies on it.
                if m1.abs() <= tol::LINEAR
                    && m2.abs() <= tol::LINEAR
                    && let Some(curve) = f.member(t)
                {
                    let range = curve.domain();
                    let p = curve.eval(range.0);
                    out.curves.push(SsiCurve {
                        curve,
                        range,
                        closed: true,
                        tangent: super::tangent_at(a, b, p),
                    });
                    continue;
                }
                let p = at(f, t, 0.0, 0.0);
                let uv = other.project(p);
                out.points.push(p.lerp(other.eval(uv.x, uv.y), 0.5));
            }
        }
        _ => {
            // Each end of the arc in `c`: past ±1 the arcs either side of θ = 0 (or π) are one.
            let end = |e: f64| -> Option<(f64, End)> {
                let m = margin(f, e);
                match node_snap(m) {
                    1 => None,
                    0 => Some((e, End::Node(f.chord(if e > 0.0 { 0.0 } else { PI }).2))),
                    _ => {
                        let root =
                            numeric::bisect(&mut |c| margin(f, c), e.min(c_peak), e.max(c_peak));
                        Some((root, End::Turn))
                    }
                }
            };
            match (end(-1.0), end(1.0)) {
                (None, None) => {
                    for sheet in [1.0, -1.0] {
                        let sample = |t: f64| at(f, t, sheet, 0.0);
                        out.fit(&sample, (0.0, TAU), true, a, b)?;
                    }
                }
                (Some((lo, kl)), None) => {
                    let t = lo.acos();
                    out.arc(f, (-t, t), (kl, kl), (a, b))?;
                }
                (None, Some((hi, kh))) => {
                    let t = hi.acos();
                    out.arc(f, (t, TAU - t), (kh, kh), (a, b))?;
                }
                (Some((lo, kl)), Some((hi, kh))) => {
                    let (tl, th) = (lo.acos(), hi.acos());
                    out.arc(f, (th, tl), (kh, kl), (a, b))?;
                    out.arc(f, (-tl, -th), (kl, kh), (a, b))?;
                }
            }
            for (c, t) in [(-1.0, PI), (1.0, 0.0)] {
                if node_snap(margin(f, c)) == 0 {
                    out.points.push(at(f, t, 0.0, 0.0));
                }
            }
        }
    }
    Ok(Ssi::Curves {
        curves: out.curves,
        points: out.points,
    })
}

/// Two unit vectors square to each other and to `z`, the first along `x` where it has a part
/// square to `z`.
fn basis(z: Vec3, x: Vec3) -> (Vec3, Vec3) {
    let e1 = (x - z * x.dot(z))
        .unit()
        .unwrap_or_else(|| z.any_perpendicular());
    (e1, z.cross(e1))
}

/// A cylinder's generators against a cylinder whose axis is not parallel to them.
struct GeneratorsCylinder {
    origin: Point3,
    axis: Vec3,
    radius: f64,
    e1: Vec3,
    e2: Vec3,
    other: Cylinder,
    /// The axes' signed distance along `e1`, their common normal.
    delta: f64,
}

impl Family for GeneratorsCylinder {
    fn margins(&self, c: f64) -> [f64; 2] {
        let s = self.delta + self.radius * c;
        [self.other.radius - s, self.other.radius + s]
    }

    fn chord(&self, theta: f64) -> (Point3, Vec3, f64) {
        let a1 = self.other.frame.z;
        let base = self.origin + (self.e1 * theta.cos() + self.e2 * theta.sin()) * self.radius;
        let w = base - self.other.frame.origin;
        let g = w - a1 * w.dot(a1);
        let h = self.axis - a1 * self.axis.dot(a1);
        let hh = h.norm2();
        let t0 = -g.dot(h) / hh;
        let d = (self.delta + self.radius * theta.cos()).abs();
        let r = self.other.radius;
        (base + self.axis * t0, self.axis, (r - d) * (r + d) / hh)
    }

    fn member(&self, _: f64) -> Option<Curve> {
        None
    }
}

/// A cylinder's generators against a sphere whose centre is off the axis.
struct GeneratorsSphere {
    origin: Point3,
    axis: Vec3,
    radius: f64,
    e1: Vec3,
    e2: Vec3,
    centre: Point3,
    big: f64,
    /// The centre's distance from the axis, along `e1`.
    sigma: f64,
}

impl Family for GeneratorsSphere {
    fn margins(&self, c: f64) -> [f64; 2] {
        let (s, r) = (self.sigma, self.radius);
        let d2 = s * s + r * r + 2.0 * s * r * c;
        [self.big - d2.max(0.0).sqrt(), f64::INFINITY]
    }

    fn chord(&self, theta: f64) -> (Point3, Vec3, f64) {
        let base = self.origin + (self.e1 * theta.cos() + self.e2 * theta.sin()) * self.radius;
        let w = base - self.centre;
        let t0 = -w.dot(self.axis);
        let foot = base + self.axis * t0;
        let d = foot.dist(self.centre);
        (foot, self.axis, (self.big - d) * (self.big + d))
    }

    fn member(&self, _: f64) -> Option<Curve> {
        None
    }
}

/// A torus's meridian circles, `centre(θ) = origin + major·ρ(θ)`, `ρ(θ) = cos θ·e1 + sin θ·e2`.
struct Meridians {
    torus: Torus,
    e1: Vec3,
    e2: Vec3,
}

impl Meridians {
    fn new(t: &Torus, toward: Vec3) -> Meridians {
        let (e1, e2) = basis(t.frame.z, toward);
        Meridians {
            torus: t.clone(),
            e1,
            e2,
        }
    }

    fn radial(&self, theta: f64) -> Vec3 {
        self.e1 * theta.cos() + self.e2 * theta.sin()
    }

    /// A point `(s, w)` of the meridian half plane at `θ`.
    fn world(&self, theta: f64, s: f64, w: f64) -> Point3 {
        self.torus.frame.origin + self.radial(theta) * s + self.torus.frame.z * w
    }

    fn circle(&self, theta: f64) -> Curve {
        let rho = self.radial(theta);
        let frame = crate::math::Frame {
            origin: self.torus.frame.origin + rho * self.torus.major,
            x: rho,
            y: self.torus.frame.z,
            z: rho.cross(self.torus.frame.z),
        };
        Curve::Circle(crate::geom::Circle {
            frame,
            radius: self.torus.minor,
        })
    }
}

/// A torus's meridians against a plane neither square to the axis nor holding it.
struct MeridiansPlane {
    m: Meridians,
    /// The normal's parts along `e1` and along the axis.
    na: f64,
    nz: f64,
    /// The plane's offset from the torus's centre along its normal.
    e: f64,
}

impl Family for MeridiansPlane {
    fn margins(&self, c: f64) -> [f64; 2] {
        let (big, r) = (self.m.torus.major, self.m.torus.minor);
        let reach = r * (self.na * self.na * c * c + self.nz * self.nz).sqrt();
        let off = big * self.na * c - self.e;
        [reach - off, reach + off]
    }

    fn chord(&self, theta: f64) -> (Point3, Vec3, f64) {
        let (big, r) = (self.m.torus.major, self.m.torus.minor);
        let (mx, mz) = (self.na * theta.cos(), self.nz);
        let mm = mx * mx + mz * mz;
        let off = big * mx - self.e;
        let (fs, fw) = (big - mx * off / mm, -mz * off / mm);
        let inv = 1.0 / mm.sqrt();
        let foot = self.m.world(theta, fs, fw);
        let dir = self.m.radial(theta) * (-mz * inv) + self.m.torus.frame.z * (mx * inv);
        (foot, dir, r * r - off * off / mm)
    }

    fn member(&self, _: f64) -> Option<Curve> {
        None
    }
}

/// A torus's meridians against a sphere whose centre is off the axis.
struct MeridiansSphere {
    m: Meridians,
    /// The centre's distance from the axis (along `e1`) and height along it.
    sigma: f64,
    zeta: f64,
    big: f64,
}

impl MeridiansSphere {
    /// In the meridian plane at `c`: the distance between the meridian's centre and the foot of
    /// the sphere's centre, and the centre's distance off the plane, squared.
    fn offsets(&self, c: f64) -> (f64, f64) {
        let r = self.m.torus.major;
        let d = (r - self.sigma * c).hypot(self.zeta);
        (d, (self.sigma * self.sigma * (1.0 - c * c)).max(0.0))
    }
}

impl Family for MeridiansSphere {
    fn margins(&self, c: f64) -> [f64; 2] {
        let r = self.m.torus.minor;
        let (d, h2) = self.offsets(c);
        let near = ((d - r) * (d - r) + h2).sqrt();
        let far = ((d + r) * (d + r) + h2).sqrt();
        [self.big - near, far - self.big]
    }

    fn chord(&self, theta: f64) -> (Point3, Vec3, f64) {
        let (big, r) = (self.m.torus.major, self.m.torus.minor);
        let c = theta.cos();
        let (d, h2) = self.offsets(c);
        let rho2 = (self.big - h2.sqrt()) * (self.big + h2.sqrt());
        // From the meridian's centre toward the foot of the sphere's centre, in (s, w).
        let (ux, uz) = if d > 0.0 {
            ((self.sigma * c - big) / d, self.zeta / d)
        } else {
            (1.0, 0.0)
        };
        let x = if d > 0.0 {
            (d * d + r * r - rho2) / (2.0 * d)
        } else {
            0.0
        };
        let foot = self.m.world(theta, big + ux * x, uz * x);
        let dir = self.m.radial(theta) * (-uz) + self.m.torus.frame.z * ux;
        (foot, dir, (r - x) * (r + x))
    }

    fn member(&self, theta: f64) -> Option<Curve> {
        Some(self.m.circle(theta))
    }
}

fn generators_cylinder(b: &Cylinder, other: &Cylinder) -> Option<GeneratorsCylinder> {
    let n = other.frame.z.cross(b.frame.z).unit()?;
    let mut delta = (b.frame.origin - other.frame.origin).dot(n);
    let n = if delta < 0.0 {
        delta = -delta;
        -n
    } else {
        n
    };
    let (e1, e2) = basis(b.frame.z, n);
    Some(GeneratorsCylinder {
        origin: b.frame.origin,
        axis: b.frame.z,
        radius: b.radius,
        e1,
        e2,
        other: other.clone(),
        delta,
    })
}

/// The intersection of `a` and `b` by a family sweeping one of them, for the pairs that have one:
/// cylinders whose axes are not parallel, a cylinder and a sphere off its axis, and a torus against
/// a plane or a sphere off its axis. `None` for any other pair, or one these do not decide.
/// The intersection of `a` and `b`, of whose branches only the stretches through `region` are
/// fitted.
pub(super) fn intersect(a: &Surface, b: &Surface, region: &Aabb) -> Option<Ssi> {
    let run = |f: &dyn Family, other: &Surface| solve(f, other, a, b, region).ok();
    match (a, b) {
        (Surface::Cylinder(x), Surface::Cylinder(y)) => {
            // The smaller one sweeps: its generators cross the other's axis at the steeper angle.
            let (sweep, other, os) = if x.radius <= y.radius {
                (x, y, b)
            } else {
                (y, x, a)
            };
            run(&generators_cylinder(sweep, other)?, os)
        }
        (Surface::Cylinder(c), Surface::Sphere(s)) | (Surface::Sphere(s), Surface::Cylinder(c)) => {
            let other = if matches!(a, Surface::Sphere(_)) {
                a
            } else {
                b
            };
            let w = s.frame.origin - c.frame.origin;
            let q = w - c.frame.z * w.dot(c.frame.z);
            let sigma = q.norm();
            if sigma <= tol::LINEAR {
                return None;
            }
            let (e1, e2) = basis(c.frame.z, -q);
            run(
                &GeneratorsSphere {
                    origin: c.frame.origin,
                    axis: c.frame.z,
                    radius: c.radius,
                    e1,
                    e2,
                    centre: s.frame.origin,
                    big: s.radius,
                    sigma,
                },
                other,
            )
        }
        // A torus whose tube passes its axis is not swept by meridians that each keep to one side;
        // one whose tube reaches the axis (a corner blend's) still is, its margins monotone.
        (Surface::Torus(t), _) | (_, Surface::Torus(t)) if t.major < t.minor => None,
        (Surface::Torus(t), Surface::Plane(p)) | (Surface::Plane(p), Surface::Torus(t)) => {
            let other = if matches!(a, Surface::Plane(_)) { a } else { b };
            meridians_plane(t, p).and_then(|f| run(&f, other))
        }
        (Surface::Torus(t), Surface::Sphere(s)) | (Surface::Sphere(s), Surface::Torus(t)) => {
            let other = if matches!(a, Surface::Sphere(_)) {
                a
            } else {
                b
            };
            let w = s.frame.origin - t.frame.origin;
            let zeta = w.dot(t.frame.z);
            let q = w - t.frame.z * zeta;
            let sigma = q.norm();
            if sigma <= tol::LINEAR {
                return None;
            }
            run(
                &MeridiansSphere {
                    m: Meridians::new(t, q),
                    sigma,
                    zeta,
                    big: s.radius,
                },
                other,
            )
        }
        _ => None,
    }
}

fn meridians_plane(t: &Torus, p: &Plane) -> Option<MeridiansPlane> {
    let n = p.frame.z;
    let nz = n.dot(t.frame.z);
    let perp = n - t.frame.z * nz;
    let na = perp.norm();
    if na <= tol::ANGULAR {
        return None;
    }
    Some(MeridiansPlane {
        m: Meridians::new(t, perp),
        na,
        nz,
        e: n.dot(p.frame.origin - t.frame.origin),
    })
}

/// A plane holding a torus's axis cuts it in the two meridian circles in that plane.
pub(super) fn plane_through_axis(t: &Torus, p: &Plane) -> Option<Ssi> {
    let n = p.frame.z;
    if n.dot(t.frame.z).abs() > tol::ANGULAR
        || n.dot(p.frame.origin - t.frame.origin).abs() > tol::LINEAR
    {
        return None;
    }
    let m = Meridians::new(t, n.cross(t.frame.z));
    let curves = [0.0, PI]
        .into_iter()
        .map(|theta| {
            let curve = m.circle(theta);
            let range = curve.domain();
            SsiCurve {
                curve,
                range,
                closed: true,
                tangent: false,
            }
        })
        .collect();
    Some(Ssi::Curves {
        curves,
        points: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boolean::csi::{curve_distance, signed_distance},
        geom::{Plane, Sphere},
        math::Frame,
    };

    fn cyl(o: [f64; 3], z: [f64; 3], r: f64) -> Surface {
        Surface::Cylinder(Cylinder {
            frame: Frame::from_axis(Vec3::from_array(o), Vec3::from_array(z)).unwrap(),
            radius: r,
        })
    }

    fn sphere(o: [f64; 3], r: f64) -> Surface {
        Surface::Sphere(Sphere {
            frame: Frame::from_axis(Vec3::from_array(o), Vec3::Z).unwrap(),
            radius: r,
        })
    }

    fn plane(o: [f64; 3], n: [f64; 3]) -> Surface {
        Surface::Plane(Plane {
            frame: Frame::from_axis(Vec3::from_array(o), Vec3::from_array(n).unit().unwrap())
                .unwrap(),
        })
    }

    /// A torus about z through the origin, major 6, minor 2.
    fn torus() -> Surface {
        Surface::Torus(Torus {
            frame: Frame::WORLD,
            major: 6.0,
            minor: 2.0,
        })
    }

    /// The curves and points, every one checked on both surfaces, and both orders alike.
    fn meet(a: &Surface, b: &Surface) -> (Vec<SsiCurve>, Vec<Point3>) {
        let all = Aabb {
            min: Vec3::new(-1e3, -1e3, -1e3),
            max: Vec3::new(1e3, 1e3, 1e3),
        };
        let Some(Ssi::Curves { curves, points }) = intersect(a, b, &all) else {
            panic!("no closed form");
        };
        let Some(Ssi::Curves {
            curves: c2,
            points: p2,
        }) = intersect(b, a, &all)
        else {
            panic!("no closed form");
        };
        assert_eq!((c2.len(), p2.len()), (curves.len(), points.len()));
        for c in &curves {
            for i in 0..=64 {
                let p = c
                    .curve
                    .eval(c.range.0 + (c.range.1 - c.range.0) * i as f64 / 64.0);
                for s in [a, b] {
                    let d = signed_distance(s, p).abs();
                    assert!(
                        d <= tol::LINEAR,
                        "{} off {} by {d}",
                        c.curve.kind(),
                        s.kind()
                    );
                }
            }
        }
        for p in &points {
            for s in [a, b] {
                assert!(signed_distance(s, *p).abs() <= tol::LINEAR);
            }
        }
        (curves, points)
    }

    fn ends(c: &SsiCurve) -> [Point3; 2] {
        [c.curve.eval(c.range.0), c.curve.eval(c.range.1)]
    }

    #[test]
    fn perpendicular_cylinders_touching_outside_meet_at_one_point() {
        // r 1.5 about y at z 5.5 against r 4 about x: the axes are 5.5 = 4 + 1.5 apart, so the
        // cylinders touch at (0, 0, 4) and nowhere else, also when moved within the tolerance.
        let big = cyl([0.0; 3], [1.0, 0.0, 0.0], 4.0);
        for dz in [0.0, 5e-8, -5e-8] {
            let small = cyl([0.0, 0.0, 5.5 + dz], [0.0, 1.0, 0.0], 1.5);
            let (cs, pts) = meet(&big, &small);
            assert!(cs.is_empty());
            assert_eq!(pts.len(), 1);
            let want = Vec3::new(0.0, 0.0, 4.0 + 0.5 * dz);
            assert!(pts[0].dist(want) <= 1e-12, "{:?}", pts[0]);
        }
        // Moved 1e-6 apart they miss; moved 1e-6 together they cross in one small loop.
        let apart = cyl([0.0, 0.0, 5.5 + 1e-6], [0.0, 1.0, 0.0], 1.5);
        let (cs, pts) = meet(&big, &apart);
        assert!(cs.is_empty() && pts.is_empty());
        let into = cyl([0.0, 0.0, 5.5 - 1e-6], [0.0, 1.0, 0.0], 1.5);
        let (cs, pts) = meet(&big, &into);
        assert_eq!(cs.len(), 1);
        assert!(cs[0].closed && pts.is_empty());
    }

    #[test]
    fn nearly_parallel_cylinders_are_fitted_only_through_the_region() {
        // r 1 about z, and r 1.2 about the line x = 0.5 leaning 1e-3 rad toward y: the second's
        // axis runs 2.2 off the first's only 2200 mm up and down, so they meet in loops some 9 m
        // long. Inside the box |x|, |y| ≤ 2, |z| ≤ 1 they meet in stretches a few mm long, and
        // only those, a reading's spacing past the box at each end, are fitted.
        let lean: f64 = 1e-3;
        let a = cyl([0.0; 3], [0.0, 0.0, 1.0], 1.0);
        let b = cyl([0.5, 0.0, 0.0], [0.0, lean.sin(), lean.cos()], 1.2);
        let region = Aabb {
            min: Vec3::new(-2.0, -2.0, -1.0),
            max: Vec3::new(2.0, 2.0, 1.0),
        };
        let Some(Ssi::Curves { curves, .. }) = intersect(&a, &b, &region) else {
            panic!("no closed form");
        };
        assert!(!curves.is_empty());
        let mut length = 0.0;
        for c in &curves {
            let n = 256;
            let at = |i: usize| {
                c.curve
                    .eval(c.range.0 + (c.range.1 - c.range.0) * i as f64 / n as f64)
            };
            for i in 0..=n {
                for s in [&a, &b] {
                    assert!(signed_distance(s, at(i)).abs() <= tol::LINEAR, "{i}");
                }
            }
            length += (1..=n).map(|i| at(i).dist(at(i - 1))).sum::<f64>();
        }
        assert!(length < 100.0, "{length}");
    }

    #[test]
    fn a_thin_cylinder_meets_a_thick_one_in_two_loops_or_one() {
        // r 2 about y through the axis of r 5 about x: every generator of the thin one crosses
        // the thick one twice, so it goes in and out in two loops; so too 1.5 off, as 1.5 + 2 < 5.
        let big = cyl([0.0; 3], [1.0, 0.0, 0.0], 5.0);
        for z in [0.0, 1.5] {
            let small = cyl([0.0, -10.0, z], [0.0, 1.0, 0.0], 2.0);
            let (cs, pts) = meet(&big, &small);
            assert_eq!(cs.len(), 2);
            assert!(cs.iter().all(|c| c.closed) && pts.is_empty());
        }
        // 4 off: 4 - 2 < 5 < 4 + 2, so only the generators nearer the axis cross, in one loop.
        let half = cyl([0.0, 0.0, 4.0], [0.0, 1.0, 0.0], 2.0);
        let (cs, _) = meet(&big, &half);
        assert!(cs.len() == 1 && cs[0].closed);
    }

    #[test]
    fn a_cylinder_touching_another_inside_crosses_it_through_the_touching_point() {
        // r 2 about y at z 3 inside r 5 about x touches it at (0, 0, 5), since 3 + 2 = 5. Its two
        // loops through that node come back as branches ending on it, each cut at its far side.
        let big = cyl([0.0; 3], [1.0, 0.0, 0.0], 5.0);
        let small = cyl([0.0, 0.0, 3.0], [0.0, 1.0, 0.0], 2.0);
        let (cs, pts) = meet(&big, &small);
        let node = Vec3::new(0.0, 0.0, 5.0);
        assert!(pts.iter().any(|p| p.dist(node) <= 1e-12));
        assert_eq!(cs.len(), 4);
        for c in &cs {
            assert!(!c.closed);
            let [p, q] = ends(c);
            assert!(p.dist(node).min(q.dist(node)) <= 1e-12);
            for e in [p, q] {
                assert!(pts.iter().any(|x| x.dist(e) <= 1e-12));
            }
        }
    }

    #[test]
    fn spheres_against_cylinders_touch_at_a_point_or_cross_in_loops() {
        // A sphere r 3 at (5, 0, 0) touches the cylinder r 2 about z at (2, 0, 0); r 4 crosses it.
        let c = cyl([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let (cs, pts) = meet(&c, &sphere([5.0, 0.0, 0.0], 3.0));
        assert!(cs.is_empty());
        assert!(pts.len() == 1 && pts[0].dist(Vec3::new(2.0, 0.0, 0.0)) <= 1e-12);
        let (cs, pts) = meet(&c, &sphere([5.0, 0.0, 0.0], 4.0));
        assert!(cs.len() == 1 && cs[0].closed && pts.is_empty());
        // r 1 at (1, 0, 0) touches it from inside at (2, 0, 0).
        let (cs, pts) = meet(&c, &sphere([1.0, 0.0, 0.0], 1.0));
        assert!(cs.is_empty());
        assert!(pts.len() == 1 && pts[0].dist(Vec3::new(2.0, 0.0, 0.0)) <= 1e-12);
        // r 3.5 there reaches past the far side too: every generator crosses it twice.
        let (cs, _) = meet(&c, &sphere([1.0, 0.0, 0.0], 3.5));
        assert!(cs.len() == 2 && cs.iter().all(|c| c.closed));
    }

    #[test]
    fn a_sphere_nearly_on_a_cylinder_of_its_radius_is_cut_where_the_closed_form_puts_it() {
        // r 5 about z against r 5 centred 1e-3 off the axis: z² = 10σ·cos θ − σ² on the cylinder,
        // a long thin loop whose ends turn where the surfaces cross at 1e-4 rad. Within the
        // tolerance of both surfaces there, a point can be 1e-4 mm along the band from the loop.
        let sigma = 1e-3;
        let (c, s) = (
            cyl([0.0; 3], [0.0, 0.0, 1.0], 5.0),
            sphere([sigma, 0.0, 0.0], 5.0),
        );
        let (cs, pts) = meet(&c, &s);
        assert!(cs.len() == 1 && cs[0].closed && pts.is_empty());
        let turn = (sigma / 10.0).acos();
        for back in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.0] {
            let th = turn - back;
            let z = (10.0 * sigma * th.cos() - sigma * sigma).sqrt();
            for (y, z) in [(1.0, z), (1.0, -z), (-1.0, z), (-1.0, -z)] {
                let p = Vec3::new(5.0 * th.cos(), y * 5.0 * th.sin(), z);
                let (d, _) = curve_distance(&cs[0].curve, cs[0].range, p);
                assert!(d <= tol::LINEAR, "{back}: {d:e} from the loop");
            }
        }
    }

    #[test]
    fn planes_along_a_torus_touch_it_or_cut_it_in_spiric_sections() {
        let t = torus();
        // x = 8 touches the outer equator at (8, 0, 0); x = 7 cuts one loop off the tube.
        let (cs, pts) = meet(&t, &plane([8.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        assert!(cs.is_empty() && pts.len() == 1);
        assert!(pts[0].dist(Vec3::new(8.0, 0.0, 0.0)) < 1e-12);
        let (cs, pts) = meet(&t, &plane([7.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        assert!(cs.len() == 1 && cs[0].closed && pts.is_empty());
        // x = 4 touches the inner equator at (4, 0, 0) and crosses the tube either side of it: a
        // lemniscate whose lobes cross there. Each lobe is two branches from the node to where a
        // meridian turns tangent to the plane, on the outer equator at y = ±√(8² - 4²).
        let (cs, pts) = meet(&t, &plane([4.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        assert_eq!(cs.len(), 4);
        assert!(pts.iter().any(|p| p.dist(Vec3::new(4.0, 0.0, 0.0)) < 1e-12));
        for y in [48f64.sqrt(), -48f64.sqrt()] {
            assert!(
                pts.iter().any(|p| p.dist(Vec3::new(4.0, y, 0.0)) < 1e-9),
                "{pts:?}"
            );
        }
        // A plane tilted 0.1 across the top of the tube cuts a loop off each side of it.
        let (cs, _) = meet(&t, &plane([0.0, 0.0, 1.0], [-0.1, 0.0, 1.0]));
        assert!(cs.len() == 2 && cs.iter().all(|c| c.closed));
    }

    #[test]
    fn a_plane_through_a_torus_axis_cuts_two_meridian_circles() {
        let Surface::Torus(t) = torus() else {
            unreachable!()
        };
        let p = Plane {
            frame: Frame::from_axis(Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0).unit().unwrap()).unwrap(),
        };
        let Some(Ssi::Curves { curves, .. }) = plane_through_axis(&t, &p) else {
            panic!("not through the axis");
        };
        assert_eq!(curves.len(), 2);
        for c in &curves {
            let Curve::Circle(ci) = &c.curve else {
                panic!("not a circle");
            };
            assert!(
                (ci.radius - 2.0).abs() < 1e-12 && (ci.frame.origin.norm() - 6.0).abs() < 1e-12
            );
            assert!(ci.frame.origin.dot(p.frame.z).abs() < 1e-12);
        }
    }

    #[test]
    fn spheres_against_a_torus_touch_at_a_point_or_along_a_meridian() {
        let t = torus();
        // r 1 centred 3 above the spine at (6, 0, 3) touches the top of the tube at (6, 0, 2).
        let (cs, pts) = meet(&t, &sphere([6.0, 0.0, 3.0], 1.0));
        assert!(cs.is_empty() && pts.len() == 1);
        assert!(pts[0].dist(Vec3::new(6.0, 0.0, 2.0)) < 1e-12);
        // A ball of the tube's radius on the spine touches it along the meridian circle there.
        let (cs, pts) = meet(&t, &sphere([6.0, 0.0, 0.0], 2.0));
        assert!(pts.is_empty() && cs.len() == 1 && cs[0].tangent);
        assert!(matches!(&cs[0].curve, Curve::Circle(c) if (c.radius - 2.0).abs() < 1e-12));
        // r 2.5 there crosses the tube in two loops, one either side of that meridian.
        let (cs, _) = meet(&t, &sphere([6.0, 0.0, 0.0], 2.5));
        assert!(cs.len() == 2 && cs.iter().all(|c| c.closed));
    }

    #[test]
    fn planes_along_a_corner_blends_horn_torus_touch_or_cut_it() {
        // Major and minor 2 about z: the tube reaches the axis. x = 4 touches its outer equator at
        // (4, 0, 0); x = 1 cuts one loop from the tube about (2, 0, 0).
        let t = Surface::Torus(Torus {
            frame: Frame::WORLD,
            major: 2.0,
            minor: 2.0,
        });
        let (cs, pts) = meet(&t, &plane([4.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        assert!(cs.is_empty() && pts.len() == 1);
        assert!(pts[0].dist(Vec3::new(4.0, 0.0, 0.0)) < 1e-12);
        let (cs, pts) = meet(&t, &plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        assert!(cs.len() == 1 && cs[0].closed && pts.is_empty());
    }
}
