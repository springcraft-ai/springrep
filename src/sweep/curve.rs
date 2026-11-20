//! A planar profile swept along a space curve, kept square to it: its axes turn with the curve's
//! rotation-minimizing frame (`N′ = −(T′·N) T`, no turn about `T`) and a twist that grows evenly
//! with arc length.
//!
//! The frame is carried along by double reflection (W. Wang, B. Jüttler, D. Zheng and Y. Liu,
//! "Computation of rotation minimizing frames", ACM Transactions on Graphics 27(1), 2008) in
//! steps fine enough to be exact to rounding. Each side face is a rational B-spline whose columns are
//! cubics through each profile control point's track at stations along the path; the stations
//! double until every column stays within a tenth of the tolerance of its track between them.

use super::helix::{Helix, Motion, Seg, assemble, helix};
use crate::{
    Error, Result,
    error::invalid,
    geom::Curve,
    loft::rail::{Rail, hom, interpolation_pieces, point},
    math::{Frame, Point3, Vec3},
    profile::Profile,
    sweep::extrude,
    tol,
};

/// How far a column may stray from its track, a tenth of the linear tolerance.
const FIT: f64 = tol::LINEAR / 10.0;

/// Reflection steps between neighbouring stations.
const STEPS: usize = 32;

/// The path's point and the profile's turned axes at one parameter.
#[derive(Clone, Copy, Debug)]
struct Station {
    at: Point3,
    t: Vec3,
    e1: Vec3,
    e2: Vec3,
}

impl Station {
    fn place(&self, xy: (f64, f64)) -> Point3 {
        self.at + self.e1 * xy.0 + self.e2 * xy.1
    }
}

/// `v` mirrored in the plane square to `n`; `v` itself where `n` is zero.
fn mirrored(v: Vec3, n: Vec3) -> Vec3 {
    let nn = n.dot(n);
    if nn == 0.0 {
        return v;
    }
    v - n * (2.0 / nn * n.dot(v))
}

/// The rotation-minimizing reference carried from `(x0, t0, r0)` to `(x1, t1)` by two mirrors,
/// fourth order in the step: the first, across the step, takes the frame to the new point; the
/// second takes the mirrored tangent onto the tangent there.
fn reflect(x0: Point3, t0: Vec3, r0: Vec3, x1: Point3, t1: Vec3) -> Vec3 {
    let step = x1 - x0;
    let (r_across, t_across) = (mirrored(r0, step), mirrored(t0, step));
    mirrored(r_across, t1 - t_across)
}

/// Arc length of `c` from `a` to `b`, by five-point Gauss–Legendre.
fn arc(c: &Curve, a: f64, b: f64) -> f64 {
    const X: [f64; 5] = [
        -0.906_179_845_938_664,
        -0.538_469_310_105_683,
        0.0,
        0.538_469_310_105_683,
        0.906_179_845_938_664,
    ];
    const W: [f64; 5] = [
        0.236_926_885_056_189,
        0.478_628_670_499_366,
        0.568_888_888_888_889,
        0.478_628_670_499_366,
        0.236_926_885_056_189,
    ];
    let (m, h) = (0.5 * (a + b), 0.5 * (b - a));
    X.iter()
        .zip(W)
        .map(|(&x, w)| w * h * c.d1(m + h * x).1.norm())
        .sum()
}

/// The frames at `us`, carried from `r0` at the path's start, the twist reaching `twist` at the
/// path's `length`.
fn stations(path: &Curve, us: &[f64], r0: Vec3, twist: f64, length: f64) -> Vec<Station> {
    let tangent = |u: f64| path.d1(u).1.unit().unwrap_or(Vec3::Z);
    let (mut x, mut t) = (path.eval(us[0]), tangent(us[0]));
    let mut r = (r0 - t * r0.dot(t)).unit().unwrap_or(r0);
    let mut s = 0.0;
    let mut out = Vec::with_capacity(us.len());
    let mut last = us[0];
    for &u in us {
        for k in 1..=STEPS {
            let a = last + (u - last) * (k - 1) as f64 / STEPS as f64;
            let b = last + (u - last) * k as f64 / STEPS as f64;
            if b == a {
                continue;
            }
            let (x1, t1) = (path.eval(b), tangent(b));
            r = reflect(x, t, r, x1, t1).unit().unwrap_or(r);
            s += arc(path, a, b);
            (x, t) = (x1, t1);
        }
        last = u;
        let n2 = t.cross(r);
        let phi = if length > 0.0 {
            twist * s / length
        } else {
            0.0
        };
        let (c, sn) = (phi.cos(), phi.sin());
        out.push(Station {
            at: x,
            t,
            e1: r * c + n2 * sn,
            e2: r * -sn + n2 * c,
        });
    }
    out
}

/// The length of `path` over `[u0, u1]` where it runs straight: a line, or a spline whose control
/// points all lie on the line through its ends.
fn straight(path: &Curve, u0: f64, u1: f64) -> Option<f64> {
    let (a, b) = (path.d1(u0).0, path.d1(u1).0);
    let along = b - a;
    let on_line = |p: Point3| (p - a).cross(along).norm() <= tol::ANGULAR * along.norm2();
    match path {
        Curve::Line(_) => Some(along.norm()),
        Curve::BSpline(s)
            if s.domain() == (u0, u1) && s.control_points().iter().all(|&p| on_line(p)) =>
        {
            Some(along.norm())
        }
        _ => None,
    }
}

/// Sweeps `profile`, drawn on `sketch`'s xy plane, along `path` over `range`, turning it through
/// `twist` radians by the end. The sketch lies at the path's start square to it.
pub fn along_curve(
    profile: &Profile,
    sketch: &Frame,
    path: &Curve,
    range: (f64, f64),
    twist: f64,
    feature: &str,
) -> Result<Brep> {
    let (u0, u1) = range;
    if u1.partial_cmp(&u0) != Some(std::cmp::Ordering::Greater) {
        return invalid("a path sweep runs over a range of the path");
    }
    let (start, d0) = path.d1(u0);
    let Some(t0) = d0.unit() else {
        return invalid("a path sweep's path must start moving");
    };
    if start.dist(sketch.origin) > tol::LINEAR || sketch.z.cross(t0).norm() > tol::ANGULAR {
        return Err(Error::InvalidInput(
            "a path sweep's profile must lie at the path's start, square to it".into(),
        ));
    }
    // A straight path has no frame to carry: the sweep is a prism, or with a twist the profile
    // screwed along the path, both exact.
    if sketch.z.dot(t0) > 0.0
        && let Some(length) = straight(path, u0, u1)
    {
        let placed = Frame::new(start, t0, sketch.x)?;
        if twist.abs() <= tol::ANGULAR {
            return extrude(profile, &placed, length, feature);
        }
        let motion = Helix {
            origin: start,
            axis: t0,
            rise: length / twist.abs(),
            sign: twist.signum(),
        };
        return helix(profile, &placed, &motion, twist.abs(), feature);
    }
    let length = arc_length(path, u0, u1);
    let mut profile = profile.normalized()?;
    // Loops run counter-clockwise about the motion, and the start cap's frame faces along it.
    let frame = if sketch.z.dot(t0) < 0.0 {
        for lp in &mut profile.loops {
            lp.reverse();
        }
        Frame::new(sketch.origin, -sketch.z, sketch.x)?
    } else {
        *sketch
    };
    let loops: Vec<Vec<Seg>> = profile
        .loops
        .iter()
        .map(|lp| {
            lp.segments
                .iter()
                .map(|s| {
                    let placed = s.segment.place(sketch);
                    let (a, b) = if placed.forward {
                        placed.range
                    } else {
                        (placed.range.1, placed.range.0)
                    };
                    Ok(Seg {
                        rail: Rail::of_curve(&placed.curve, a, b)?,
                        name: s.names.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<_>>()?;
    // Each world point of the profile as its place in the start frame's axes, which the frame at
    // any station carries along.
    let first = stations(path, &[u0], sketch.x, twist, length)[0];
    let local = |p: Point3| ((p - first.at).dot(first.e1), (p - first.at).dot(first.e2));
    let controls: Vec<(f64, f64)> = loops
        .iter()
        .flatten()
        .flat_map(|s| s.rail.pw.iter().map(|&h| local(point(h))))
        .collect();
    // Stations double until every control point's column follows its track between them. Each
    // path span has its own run of them: the frame's second derivative jumps where the path's
    // third does, which one cubic across a breakpoint follows only at second order.
    let mut breaks = vec![u0];
    breaks.extend(path.breakpoints(u0, u1));
    breaks.push(u1);
    let spans = breaks.len() - 1;
    let mut per = 8;
    let (us, frames, across) = loop {
        let n = per * spans;
        let mut fine: Vec<f64> = breaks
            .windows(2)
            .flat_map(|b| {
                (0..2 * per).map(move |k| b[0] + (b[1] - b[0]) * k as f64 / (2 * per) as f64)
            })
            .collect();
        fine.push(u1);
        let all = stations(path, &fine, sketch.x, twist, length);
        let us: Vec<f64> = fine.iter().step_by(2).copied().collect();
        let frames: Vec<Station> = all.iter().step_by(2).copied().collect();
        let across = interpolation_pieces(&us, per)?;
        let worst = controls
            .iter()
            .map(|&xy| {
                let data: Vec<_> = frames.iter().map(|f| hom(f.place(xy), 1.0)).collect();
                let ctrl = across.apply(&data);
                (0..n)
                    .map(|k| {
                        let got = point(across.eval(&ctrl, fine[2 * k + 1]));
                        got.dist(all[2 * k + 1].place(xy))
                    })
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        if worst <= FIT || n >= 512 {
            break (us, frames, across);
        }
        per *= 2;
    };
    let _ = us;
    let last = *frames.last().expect("a sweep has stations");
    let column = |p: Point3| -> Vec<Point3> {
        let xy = local(p);
        let data: Vec<_> = frames.iter().map(|f| hom(f.place(xy), 1.0)).collect();
        across.apply(&data).into_iter().map(point).collect()
    };
    let motion = Motion {
        degree: across.degree,
        knots: across.knots.clone(),
        range,
        column: Box::new(column),
        at_end: Box::new(|p| last.place(local(p))),
        velocity: Box::new(move |_| t0),
        end_frame: Frame::new(last.at, last.t, last.e1)?,
        screw: None,
    };
    assemble(&loops, frame, &motion, feature)
}

/// The path's arc length from `a` to `b`.
fn arc_length(path: &Curve, a: f64, b: f64) -> f64 {
    let n = 256 * (path.breakpoints(a, b).len() + 1);
    (0..n)
        .map(|k| {
            let (x, y) = (
                a + (b - a) * k as f64 / n as f64,
                a + (b - a) * (k + 1) as f64 / n as f64,
            );
            arc(path, x, y)
        })
        .sum()
}

use crate::topo::Brep;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geom::BSpline, math::Vec2, measure};
    use std::f64::consts::PI;

    fn path(ctrl: &[[f64; 3]]) -> Curve {
        let n = ctrl.len();
        let mut knots = vec![0.0; 4];
        knots.extend((1..n - 3).map(|i| i as f64 / (n - 3) as f64));
        knots.extend([1.0; 4]);
        let pts = ctrl.iter().map(|c| Vec3::new(c[0], c[1], c[2])).collect();
        Curve::BSpline(BSpline::new(3, knots, pts).unwrap())
    }

    #[test]
    fn a_disc_along_a_straight_path_is_a_cylinder_whatever_its_twist() {
        let p = path(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 6.0],
        ]);
        let disc = Profile::circle(Vec2::ZERO, 2.0);
        let b = along_curve(&disc, &Frame::WORLD, &p, (0.0, 1.0), 1.3, "sw").unwrap();
        assert!((measure::volume(&b) - PI * 6.0).abs() < 1e-7);
    }

    #[test]
    fn a_disc_along_a_bent_path_keeps_its_volume_by_pappus() {
        // A disc centred on the path sweeps π r² per unit of arc length, however the path bends,
        // while it stays within the least radius of curvature.
        let p = path(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.5, 4.0],
            [2.0, -0.5, 6.0],
            [2.5, 0.0, 8.0],
        ]);
        let disc = Profile::circle(Vec2::ZERO, 0.6);
        let b = along_curve(&disc, &Frame::WORLD, &p, (0.0, 1.0), 0.7, "sw").unwrap();
        let want = PI * 0.09 * arc_length(&p, 0.0, 1.0);
        let got = measure::volume(&b);
        assert!((got - want).abs() < 1e-6 * want, "{got} against {want}");
    }

    #[test]
    fn a_sweep_along_a_spline_stays_smooth_across_the_paths_breakpoints() {
        // The swept surface is C¹ where the path's third derivative jumps; its pieces join there
        // with no crease.
        let p = path(&[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.7],
            [1.0, 0.7, 4.5],
            [-0.4, 1.1, 7.5],
            [-0.8, 0.3, 9.9],
            [-1.9, 1.4, 12.8],
        ]);
        let disc = Profile::circle(Vec2::new(0.3, 0.0), 0.6);
        let b = along_curve(&disc, &Frame::WORLD, &p, (0.0, 1.0), 0.0, "sw").unwrap();
        let mut joins = 0;
        for f in b.face_ids() {
            let crate::geom::Surface::Nurbs(s) = &b.face(f).surface else {
                continue;
            };
            let (ku, kv) = s.knots();
            let (u0, u1, v0, v1) = (ku[0], ku[ku.len() - 1], kv[0], kv[kv.len() - 1]);
            let mut at: Vec<f64> = ku
                .windows(3)
                .filter(|w| w[0] == w[2] && w[0] > u0 && w[0] < u1)
                .map(|w| w[0])
                .collect();
            at.dedup();
            for u in at {
                joins += 1;
                for k in 0..=8 {
                    let v = v0 + (v1 - v0) * k as f64 / 8.0;
                    let normal = |u: f64| {
                        let (_, du, dv) = s.d1(u, v);
                        du.cross(dv).unit().unwrap()
                    };
                    let turn = normal(u - 1e-12).cross(normal(u + 1e-12)).norm();
                    assert!(turn <= tol::ANGULAR, "a crease of {turn} rad at u = {u}");
                }
            }
        }
        assert!(
            joins > 0,
            "the path's breakpoints join pieces of the surface"
        );
    }
}
