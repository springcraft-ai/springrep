//! A plane against a cone whose axis is not square to it: the lines through the apex each meet the
//! plane once, at `s = h / (n·g(θ))` along the generator `g(θ)`, `h` the plane's distance from the
//! apex. Where `n·g` keeps its sign the section is an ellipse, built exactly from its vertices;
//! where it changes sign the branches run out to infinity, and the part of each within reach of the
//! region is the rational quadratic curve it is. A plane through the apex holds the generators it
//! contains, touches the cone along one, or meets it at the apex alone.

use std::f64::consts::{PI, TAU};

use super::{Ssi, SsiCurve, line_curve};
use crate::{
    geom::{Cone, Curve, Ellipse, IntersectionCurve, NurbsCurve, Plane, Surface},
    math::{Aabb, Frame, Point3, Vec3},
    tol,
};

pub(super) fn plane_cone(p: &Plane, c: &Cone, region: &Aabb) -> Option<Ssi> {
    let (sa, ca) = c.half_angle.sin_cos();
    if sa.abs() <= tol::ANGULAR {
        return None;
    }
    let z = c.frame.z;
    let apex = c.frame.origin - z * (c.radius * ca / sa);
    let n = p.frame.z;
    let nz = n.dot(z);
    let perp = n - z * nz;
    let na = perp.norm();
    if na <= tol::ANGULAR {
        return None;
    }
    // e1 such that n·g(θ) = a·cos θ + b with a ≥ 0.
    let e1 = perp * (1.0 / na);
    let e2 = z.cross(e1);
    let (a, b) = (sa.abs() * na, ca * nz);
    let g = |t: f64| (e1 * t.cos() + e2 * t.sin()) * sa.abs() + z * ca;
    let h = (p.frame.origin - apex).dot(n);
    if h.abs() <= tol::LINEAR {
        // Through the apex. The generator nearest the plane leaves it at the sine |b| - a: past
        // the angular tolerance the plane meets the cone at the apex alone, within it it lies
        // along that generator, and short of it it holds the two generators where n·g = 0.
        let on = apex + n * (0.5 * h);
        let w = b.abs() - a;
        // Each generator from the apex onto the cone's own nappe only.
        let along = |t: f64| {
            let d = g(t) * sa.signum();
            (d - n * d.dot(n)).unit()
        };
        let ray = |d: Vec3, tangent: bool| {
            let mut c = line_curve(on, d, region, tangent)?;
            c.range.0 = c.range.0.max(0.0);
            (c.range.1 > c.range.0).then_some(c)
        };
        if w > tol::ANGULAR {
            return Some(Ssi::Curves {
                curves: Vec::new(),
                points: vec![on],
            });
        }
        if w >= -tol::ANGULAR {
            let d = along(if b < 0.0 { 0.0 } else { PI })?;
            return Some(Ssi::Curves {
                curves: ray(d, true).into_iter().collect(),
                points: Vec::new(),
            });
        }
        let t = (-b / a).acos();
        let curves = [t, -t]
            .into_iter()
            .filter_map(|t| ray(along(t)?, false))
            .collect();
        return Some(Ssi::Curves {
            curves,
            points: vec![on],
        });
    }
    let at = |t: f64| apex + g(t) * (h / (a * t.cos() + b));
    // The kernel's cone is the nappe its own parameters run on, where `s·sign(sin a) > 0`; the
    // other nappe is no part of any face.
    let own = |s: f64| s * sa > 0.0;
    // The farthest the region reaches from the apex.
    let reach = (0..8)
        .map(|i| {
            let corner = Vec3::new(
                if i & 1 == 0 {
                    region.min.x
                } else {
                    region.max.x
                },
                if i & 2 == 0 {
                    region.min.y
                } else {
                    region.max.y
                },
                if i & 4 == 0 {
                    region.min.z
                } else {
                    region.max.z
                },
            );
            corner.dist(apex)
        })
        .fold(0.0, f64::max);
    // An ellipse wholly within reach is built exactly; one that is not (a plane all but parallel to
    // a generator) is followed, as a parabola or hyperbola is, only as far as the region reaches.
    if b.abs() > a && h.abs() <= reach * (b.abs() - a) {
        if !own(h / (a + b)) {
            return Some(Ssi::Curves {
                curves: Vec::new(),
                points: Vec::new(),
            });
        }
        // Every generator meets the plane on one side of the apex: an ellipse, whose long axis
        // lies in the plane through the axis and the normal, from θ = 0 to θ = π.
        let (p0, p1) = (at(0.0), at(PI));
        let centre = p0.lerp(p1, 0.5);
        let half = (p0 - p1) * 0.5;
        let major = half.norm();
        let x = half * (1.0 / major);
        let y = n.cross(x);
        let q = at(0.5 * PI) - centre;
        let (qx, qy) = (q.dot(x) / major, q.dot(y));
        let minor = qy.abs() / (1.0 - qx * qx).max(0.0).sqrt();
        if !minor.is_finite() || minor <= 0.0 {
            return None;
        }
        let frame = Frame {
            origin: centre,
            x,
            y,
            z: x.cross(y),
        };
        return Some(Ssi::Curves {
            curves: vec![SsiCurve {
                curve: Curve::Ellipse(Ellipse {
                    frame,
                    major,
                    minor,
                }),
                range: (0.0, TAU),
                closed: true,
                tangent: false,
            }],
            points: Vec::new(),
        });
    }
    // Each branch runs over the angles where n·g keeps one sign, and the part of it within reach
    // over those where |n·g| ≥ |h| / reach.
    let k = h.abs() / reach;
    let mut arcs: Vec<(f64, f64)> = Vec::new();
    // a·cos θ + b ≥ k about θ = 0, and ≤ -k about θ = π.
    let c_hi = (k - b) / a;
    if c_hi < 1.0 && own(h) {
        let t = c_hi.max(-1.0).acos();
        arcs.push((-t, t));
    }
    let c_lo = (-k - b) / a;
    if c_lo > -1.0 && own(-h) {
        let t = c_lo.min(1.0).acos();
        arcs.push((t, TAU - t));
    }
    let mut curves = Vec::new();
    for (t0, t1) in arcs {
        // Over the arc about θ = π both parts change sign, so the weights are positive on either.
        let sign = (a * (0.5 * (t0 + t1)).cos() + b).signum();
        let hom = |cos: f64, sin: f64, one: f64| {
            let along = (e1 * cos + e2 * sin) * sa.abs() + z * (ca * one);
            (along * (h * sign), (a * cos + b * one) * sign)
        };
        let (curve, range) = match conic_arc(apex, &hom, (t0, t1)) {
            Some(arc) => {
                let range = arc.domain();
                (Curve::Nurbs(Box::new(arc)), range)
            }
            None => {
                let (sp, sc) = (Surface::Plane(p.clone()), Surface::Cone(c.clone()));
                let sample = |s: f64| at(t0 + (t1 - t0) * s);
                let fit = IntersectionCurve::fit_sampled(
                    &|s, _| sample(s),
                    (0.0, 1.0),
                    sp,
                    sc,
                    false,
                    32,
                )
                .ok()?;
                let range = fit.approx.domain();
                (Curve::Intersection(Box::new(fit)), range)
            }
        };
        curves.push(SsiCurve {
            curve,
            range,
            closed: false,
            tangent: false,
        });
    }
    Some(Ssi::Curves {
        curves,
        points: Vec::new(),
    })
}

/// The most pieces [`conic_arc`] cuts an arc into.
const MAX_PIECES: usize = 64;

/// The conic `apex + v / w` for `(v, w) = hom(cos θ, sin θ, 1)`, `hom` linear, over the angles
/// `t0..t1` where `w > 0`: the image of a circle's arc, so each piece of it is a rational quadratic
/// Bézier curve on the images of the arc's own three points, the middle one of weight the cosine
/// of half its angle (Piegl and Tiller 1997, ch. 7). The pieces are cut until none bends through
/// more than a quarter turn or runs far out toward an asymptote, and each takes the parameter in
/// which it starts and ends at unit speed, so the curve's parameter is close to its length and
/// its first derivative has no step. `None` where a middle weight is not positive, or the arc is
/// shorter than the tolerance.
fn conic_arc(
    apex: Point3,
    hom: &dyn Fn(f64, f64, f64) -> (Vec3, f64),
    (t0, t1): (f64, f64),
) -> Option<NurbsCurve> {
    let mut todo = vec![(t0, t1)];
    let mut knots = vec![0.0; 3];
    let mut ctrl: Vec<Point3> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut pieces = 0;
    while let Some((lo, hi)) = todo.pop() {
        let (mid, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        let (v, w): (Vec<Vec3>, Vec<f64>) = [
            hom(lo.cos(), lo.sin(), 1.0),
            hom(mid.cos(), mid.sin(), half.cos()),
            hom(hi.cos(), hi.sin(), 1.0),
        ]
        .into_iter()
        .unzip();
        if !(w[0] > 0.0 && w[2] > 0.0) {
            return None;
        }
        // 1 for a parabola's piece, and further from it the more unevenly the piece is run.
        let shape = w[1] * w[1] / (w[0] * w[2]);
        let even = w[1] > 0.0 && half <= 0.26 * PI && (0.4..=2.5).contains(&shape);
        if !even && pieces + todo.len() + 2 <= MAX_PIECES && half > 1e-9 {
            todo.push((mid, hi));
            todo.push((lo, mid));
            continue;
        }
        if w[1].is_nan() || w[1] <= 0.0 {
            return None;
        }
        let q = [0, 1, 2].map(|i| apex + v[i] * (1.0 / w[i]));
        let (l0, l1) = (q[1].dist(q[0]), q[2].dist(q[1]));
        if !(l0 > 0.0 && l1 > 0.0) {
            return None;
        }
        // Weights w0, k·w1, k²·w2 draw the same piece at another rate: this k makes its speed at
        // the two ends equal, and the knot span that speed makes it 1.
        let k = (w[0] * l1 / (w[2] * l0)).sqrt();
        let span = 2.0 * k * w[1] / w[0] * l0;
        let scale = match weights.last() {
            Some(&last) => last / w[0],
            None => {
                ctrl.push(q[0]);
                weights.push(1.0);
                1.0 / w[0]
            }
        };
        ctrl.extend([q[1], q[2]]);
        weights.extend([scale * k * w[1], scale * k * k * w[2]]);
        let end = knots[knots.len() - 1] + span;
        knots.extend([end, end]);
        pieces += 1;
    }
    knots.push(knots[knots.len() - 1]);
    let arc = NurbsCurve::new(2, knots, ctrl, Some(weights)).ok()?;
    arc.period().is_none().then_some(arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean::csi::signed_distance;

    /// A cone about z with its apex at the origin, opening upward at 45°.
    fn cone() -> Cone {
        Cone {
            frame: Frame::new(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, Vec3::X).unwrap(),
            radius: 1.0,
            half_angle: PI / 4.0,
        }
    }

    fn plane(o: [f64; 3], n: [f64; 3]) -> Plane {
        Plane {
            frame: Frame::from_axis(Vec3::from_array(o), Vec3::from_array(n).unit().unwrap())
                .unwrap(),
        }
    }

    fn region() -> Aabb {
        Aabb {
            min: Vec3::new(-10.0, -10.0, -10.0),
            max: Vec3::new(10.0, 10.0, 10.0),
        }
    }

    fn on_both(p: &Plane, c: &Cone, cs: &[SsiCurve]) {
        let (sp, sc) = (Surface::Plane(p.clone()), Surface::Cone(c.clone()));
        for cv in cs {
            for i in 0..=64 {
                let x = cv
                    .curve
                    .eval(cv.range.0 + (cv.range.1 - cv.range.0) * i as f64 / 64.0);
                if (0..3).any(|k| x[k].abs() > 10.0) {
                    continue;
                }
                for s in [&sp, &sc] {
                    let d = signed_distance(s, x).abs();
                    assert!(
                        d <= tol::LINEAR,
                        "{} off {} by {d}",
                        cv.curve.kind(),
                        s.kind()
                    );
                }
            }
        }
    }

    fn meet(p: &Plane, c: &Cone) -> (Vec<SsiCurve>, Vec<Point3>) {
        let Some(Ssi::Curves { curves, points }) = plane_cone(p, c, &region()) else {
            panic!("no closed form");
        };
        on_both(p, c, &curves);
        (curves, points)
    }

    #[test]
    fn a_narrowing_cone_keeps_the_branch_below_its_apex() {
        // r 4 at z 0 narrowing at 45° to its apex at (0, 0, 4): y = 2 cuts x² + 4 = (4 - z)², whose
        // branch on the cone's own nappe passes (0, 2, 2).
        let c = Cone {
            frame: Frame::WORLD,
            radius: 4.0,
            half_angle: -PI / 4.0,
        };
        let (cs, pts) = meet(&plane([0.0, 2.0, 0.0], [0.0, 1.0, 0.0]), &c);
        assert!(pts.is_empty() && cs.len() == 1);
        let at = Vec3::new(0.0, 2.0, 2.0);
        let (d, _) = crate::boolean::csi::curve_distance(&cs[0].curve, cs[0].range, at);
        assert!(d < 1e-9, "{d}");
    }

    #[test]
    fn a_plane_tilted_less_than_the_cone_cuts_an_ellipse() {
        // z = 2 + 0.5·x cuts the 45° cone in an ellipse whose vertices lie in y = 0, where
        // |x| = z: x = 2 + 0.5x gives (4, 0, 4), and -x = 2 + 0.5x gives (-4/3, 0, 4/3).
        let c = cone();
        let p = plane([0.0, 0.0, 2.0], [-0.5, 0.0, 1.0]);
        let (cs, pts) = meet(&p, &c);
        assert!(pts.is_empty() && cs.len() == 1);
        let Curve::Ellipse(e) = &cs[0].curve else {
            panic!("not an ellipse");
        };
        let half = 0.5 * Vec3::new(4.0, 0.0, 4.0).dist(Vec3::new(-4.0 / 3.0, 0.0, 4.0 / 3.0));
        assert!((e.major - half).abs() < 1e-12, "{}", e.major);
    }

    #[test]
    fn a_plane_steeper_than_the_cone_cuts_a_hyperbola_or_parabola() {
        // x = 1 is parallel to the axis: a hyperbola z² - y² = 1, whose branch on the cone's own
        // nappe (z > 0) is kept.
        let c = cone();
        let (cs, pts) = meet(&plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), &c);
        assert!(pts.is_empty() && cs.len() == 1 && !cs[0].closed);
        assert!(cs[0].curve.eval(cs[0].range.0).z > 0.0);
        // z = 1 + x is parallel to a generator: one parabola. Tilted 1e-12 further it is an
        // ellipse some 1e12 long, followed as far as the region reaches as the parabola is.
        let (cs, _) = meet(&plane([0.0, 0.0, 1.0], [-1.0, 0.0, 1.0]), &c);
        assert_eq!(cs.len(), 1);
        let (cs, _) = meet(&plane([0.0, 0.0, 1.0], [-1.0, 0.0, 1.0 + 1e-12]), &c);
        assert!(cs.len() == 1 && !cs[0].closed);
        let ends = [
            cs[0].curve.eval(cs[0].range.0),
            cs[0].curve.eval(cs[0].range.1),
        ];
        assert!(ends.iter().all(|p| p.norm() < 100.0), "{ends:?}");
    }

    /// Each curve is a rational quadratic lying on both surfaces to rounding all along, and runs
    /// at one speed across every join.
    fn exact_conic(p: &Plane, c: &Cone, cs: &[SsiCurve]) {
        let (sp, sc) = (Surface::Plane(p.clone()), Surface::Cone(c.clone()));
        for cv in cs {
            let Curve::Nurbs(nc) = &cv.curve else {
                panic!("a {} for a conic", cv.curve.kind());
            };
            assert_eq!(nc.degree(), 2);
            assert_eq!(cv.range, nc.domain());
            let (t0, t1) = cv.range;
            for i in 0..=512 {
                let x = nc.eval(t0 + (t1 - t0) * i as f64 / 512.0);
                for s in [&sp, &sc] {
                    let d = signed_distance(s, x).abs();
                    assert!(d <= 1e-12 * (1.0 + x.norm()), "off {} by {d}", s.kind());
                }
            }
            for t in nc.breakpoints(t0, t1) {
                let e = 1e-9 * (t1 - t0);
                let ((_, before, b2), (_, after, a2)) = (nc.d2(t - e), nc.d2(t + e));
                let turned = 2.0 * e * (b2.norm() + a2.norm());
                assert!(
                    before.dist(after) <= 1e-6 * before.norm() + turned,
                    "a kink at {t}"
                );
                assert!((before.norm() - 1.0).abs() <= 1e-6 + turned, "speed at {t}");
            }
        }
    }

    #[test]
    fn a_hyperbola_and_a_parabola_are_exact_rational_quadratics() {
        let c = cone();
        // x = 1 cuts z² − y² = 1: the branch through (1, 3, √10), out to the region's reach.
        let p = plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let (cs, _) = meet(&p, &c);
        exact_conic(&p, &c, &cs);
        let at = Vec3::new(1.0, 3.0, 10f64.sqrt());
        let (d, _) = crate::boolean::csi::curve_distance(&cs[0].curve, cs[0].range, at);
        assert!(d < 1e-12, "{d}");
        let ends = [cs[0].range.0, cs[0].range.1].map(|t| cs[0].curve.eval(t));
        assert!(ends.iter().all(|e| e.norm() > 17.0), "{ends:?}");
        // z = 1 + x cuts the parabola y² = 1 + 2x, through (4, 3, 5).
        let p = plane([0.0, 0.0, 1.0], [-1.0, 0.0, 1.0]);
        let (cs, _) = meet(&p, &c);
        exact_conic(&p, &c, &cs);
        let at = Vec3::new(4.0, 3.0, 5.0);
        let (d, _) = crate::boolean::csi::curve_distance(&cs[0].curve, cs[0].range, at);
        assert!(d < 1e-12, "{d}");
        // An ellipse too long for the region, an oblique hyperbola, and a narrowing cone's.
        let p = plane([0.0, 0.0, 1.0], [-1.0, 0.0, 1.0 + 1e-12]);
        exact_conic(&p, &c, &meet(&p, &c).0);
        let p = plane([0.3, -0.2, 0.5], [1.0, 0.4, 0.3]);
        let (cs, _) = meet(&p, &c);
        assert_eq!(cs.len(), 1);
        exact_conic(&p, &c, &cs);
        let narrowing = Cone {
            frame: Frame::WORLD,
            radius: 4.0,
            half_angle: -PI / 4.0,
        };
        let p = plane([0.0, 2.0, 0.0], [0.1, 1.0, 0.0]);
        exact_conic(&p, &narrowing, &meet(&p, &narrowing).0);
    }

    #[test]
    fn sections_of_wide_and_narrow_cones_near_and_far_from_the_apex_are_exact() {
        let mut open = 0;
        for half_angle in [0.05, 0.6, -0.6, 1.4] {
            let c = Cone {
                frame: Frame::new(Vec3::new(0.2, -0.1, 0.4), Vec3::new(0.1, 0.2, 1.0), Vec3::X)
                    .unwrap(),
                radius: 1.5,
                half_angle,
            };
            let apex = c.frame.origin - c.frame.z * (c.radius / half_angle.tan());
            for tilt in 0..24 {
                let turn = PI * (tilt as f64 + 0.5) / 24.0;
                let n = c.frame.x * turn.sin() + c.frame.z * turn.cos();
                for h in [-5.0, -1e-3, 1e-4, 0.3, 5.0] {
                    let p = Plane {
                        frame: Frame::from_axis(apex + n * h, n).unwrap(),
                    };
                    let Some(Ssi::Curves { curves, .. }) = plane_cone(&p, &c, &region()) else {
                        panic!("no closed form");
                    };
                    let arcs: Vec<SsiCurve> = curves.into_iter().filter(|cv| !cv.closed).collect();
                    exact_conic(&p, &c, &arcs);
                    open += arcs.len();
                }
            }
        }
        assert!(open > 200, "{open}");
    }

    #[test]
    fn a_plane_through_the_apex_holds_two_generators_touches_along_one_or_meets_the_apex() {
        let c = cone();
        let (cs, pts) = meet(&plane([0.0; 3], [1.0, 0.0, 0.0]), &c);
        assert!(cs.len() == 2 && pts.len() == 1 && cs.iter().all(|cv| !cv.tangent));
        let (cs, pts) = meet(&plane([0.0; 3], [-1.0, 0.0, 1.0]), &c);
        assert!(cs.len() == 1 && cs[0].tangent && pts.is_empty());
        let (cs, pts) = meet(&plane([0.0; 3], [-0.5, 0.0, 1.0]), &c);
        assert!(cs.is_empty() && pts.len() == 1 && pts[0].norm() < 1e-12);
    }
}
