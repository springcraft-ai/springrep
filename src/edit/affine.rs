//! A solid mapped by an affine map that need not keep its shape: stretched along axes, sheared or
//! mirrored. A plane maps to a plane and a line to a line; a circle maps to an ellipse, and each
//! surface of revolution to the image of its exact rational form, so nothing is approximated.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::{
    Error, Result,
    geom::{BSpline, Curve, Ellipse, Line, NurbsCurve, NurbsSurface, Plane, Surface},
    math::{Frame, Point3, Vec3},
    step,
    topo::{Brep, FaceId},
};

/// `x ↦ m·x + t`, `m` given by rows.
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    pub m: [[f64; 3]; 3],
    pub t: Vec3,
}

impl Affine {
    pub fn vector(&self, v: Vec3) -> Vec3 {
        let r = self.m.map(|row| row[0] * v.x + row[1] * v.y + row[2] * v.z);
        Vec3::new(r[0], r[1], r[2])
    }

    pub fn point(&self, p: Point3) -> Point3 {
        self.vector(p) + self.t
    }

    pub fn det(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

/// `b` mapped by `a`, which must be invertible.
pub fn affine(b: &Brep, a: &Affine) -> Result<Brep> {
    let det = a.det();
    if !det.is_finite() || det.abs() < 1e-12 {
        return Err(Error::InvalidInput(
            "an affine map that flattens the solid".into(),
        ));
    }
    let points = b.vertex_ids().map(|v| a.point(b.vertex(v).point)).collect();
    let curves = b
        .edge_ids()
        .map(|e| b.edge(e).curve.as_ref().map(|c| curve(c, a)).transpose())
        .collect::<Result<_>>()?;
    let surfaces = b
        .face_ids()
        .filter(|&f| b.face(f).shell.is_some())
        .map(|f| surface(b, f, a))
        .collect::<Result<_>>()?;
    step::rebuilt(b, points, curves, surfaces, det < 0.0)
}

fn curve(c: &Curve, a: &Affine) -> Result<Curve> {
    Ok(match c {
        Curve::Line(l) => Curve::Line(Line {
            origin: a.point(l.origin),
            dir: a
                .vector(l.dir)
                .unit()
                .ok_or_else(|| Error::InvalidInput("a line mapped to a point".into()))?,
        }),
        Curve::Circle(ci) => ellipse(&ci.frame, ci.radius, ci.radius, a)?,
        Curve::Ellipse(e) => ellipse(&e.frame, e.major, e.minor, a)?,
        Curve::BSpline(s) => {
            let ctrl = s.control_points().iter().map(|&p| a.point(p)).collect();
            Curve::BSpline(if s.is_periodic() {
                BSpline::periodic(s.degree(), s.knots().to_vec(), ctrl)?
            } else {
                BSpline::new(s.degree(), s.knots().to_vec(), ctrl)?
            })
        }
        Curve::Nurbs(n) => Curve::Nurbs(Box::new(NurbsCurve::new(
            n.degree(),
            n.knots().to_vec(),
            n.control_points().iter().map(|&p| a.point(p)).collect(),
            n.weights().map(<[f64]>::to_vec),
        )?)),
        Curve::Intersection(_) => {
            return Err(Error::Unsupported(
                "an affine map of an edge on an intersection curve".into(),
            ));
        }
    })
}

/// The image of `o + p·cos t·x + q·sin t·y` as an ellipse on its principal axes, its parameter
/// shifted but running the same way.
fn ellipse(f: &Frame, p: f64, q: f64, a: &Affine) -> Result<Curve> {
    let (u, w) = (a.vector(f.x * p), a.vector(f.y * q));
    let t0 = 0.5 * (2.0 * u.dot(w)).atan2(u.norm2() - w.norm2());
    let major = u * t0.cos() + w * t0.sin();
    let minor = w * t0.cos() - u * t0.sin();
    let frame = Frame::new(a.point(f.origin), major.cross(minor), major)?;
    Ok(Curve::Ellipse(Ellipse {
        frame,
        major: major.norm(),
        minor: minor.norm(),
    }))
}

fn surface(b: &Brep, f: FaceId, a: &Affine) -> Result<Surface> {
    let s = &b.face(f).surface;
    let revolved = |fr: &Frame, profile: (Vec<[f64; 2]>, Vec<f64>, Vec<f64>)| {
        let (ps, ws, knots) = profile;
        let h = FRAC_1_SQRT_2;
        let ring = [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, h],
            [0.0, 1.0, 1.0],
            [-1.0, 1.0, h],
            [-1.0, 0.0, 1.0],
            [-1.0, -1.0, h],
            [0.0, -1.0, 1.0],
            [1.0, -1.0, h],
            [1.0, 0.0, 1.0],
        ];
        let mut ctrl = Vec::new();
        let mut weights = Vec::new();
        for c in ring {
            for (p, w) in ps.iter().zip(&ws) {
                let local = fr.x * (p[0] * c[0]) + fr.y * (p[0] * c[1]) + fr.z * p[1];
                ctrl.push(a.point(fr.origin + local));
                weights.push(c[2] * w);
            }
        }
        let u_knots = vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ];
        let degree_v = knots.len() - ps.len() - 1;
        Ok::<_, Error>(Surface::Nurbs(Box::new(NurbsSurface::new(
            (2, degree_v),
            (u_knots, knots),
            (9, ps.len()),
            ctrl,
            Some(weights),
        )?)))
    };
    let half_circle = |c: f64, r: f64| {
        let h = FRAC_1_SQRT_2;
        (
            vec![[0.0, -r], [r, -r], [r, 0.0], [r, r], [0.0, r]]
                .into_iter()
                .map(|p: [f64; 2]| [c + p[0], p[1]])
                .collect(),
            vec![1.0, h, 1.0, h, 1.0],
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
        )
    };
    match s {
        Surface::Plane(p) => {
            let (x, y) = (a.vector(p.frame.x), a.vector(p.frame.y));
            Ok(Surface::Plane(Plane {
                frame: Frame::new(a.point(p.frame.origin), x.cross(y), x)?,
            }))
        }
        Surface::Cylinder(c) => {
            let (v0, v1) = heights(b, f, &c.frame, 1.0);
            revolved(
                &c.frame,
                (
                    vec![[c.radius, v0], [c.radius, v1]],
                    vec![1.0; 2],
                    vec![0.0, 0.0, 1.0, 1.0],
                ),
            )
        }
        Surface::Cone(c) => {
            let (sin, cos) = c.half_angle.sin_cos();
            let (v0, v1) = heights(b, f, &c.frame, cos);
            // Not past the apex, where the radius turns negative.
            let apex = -c.radius / sin;
            let (v0, v1) = if sin > 0.0 {
                (v0.max(apex), v1)
            } else {
                (v0, v1.min(apex))
            };
            let at = |v: f64| [(c.radius + v * sin).max(0.0), v * cos];
            revolved(
                &c.frame,
                (vec![at(v0), at(v1)], vec![1.0; 2], vec![0.0, 0.0, 1.0, 1.0]),
            )
        }
        Surface::Sphere(sp) => revolved(&sp.frame, half_circle(0.0, sp.radius)),
        Surface::Torus(t) => {
            let h = FRAC_1_SQRT_2;
            let ring = [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, h],
                [0.0, 1.0, 1.0],
                [-1.0, 1.0, h],
                [-1.0, 0.0, 1.0],
                [-1.0, -1.0, h],
                [0.0, -1.0, 1.0],
                [1.0, -1.0, h],
                [1.0, 0.0, 1.0],
            ];
            let ps = ring
                .iter()
                .map(|c| [t.major + t.minor * c[0], t.minor * c[1]])
                .collect();
            let ws = ring.iter().map(|c| c[2]).collect();
            let knots = vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ];
            revolved(&t.frame, (ps, ws, knots))
        }
        Surface::Nurbs(n) => Ok(Surface::Nurbs(Box::new(NurbsSurface::new(
            n.degrees(),
            (n.knots().0.to_vec(), n.knots().1.to_vec()),
            n.counts(),
            n.control_points().iter().map(|&p| a.point(p)).collect(),
            n.weights().map(<[f64]>::to_vec),
        )?))),
        _ => Err(Error::Unsupported(format!(
            "an affine map of a {} face",
            match s {
                Surface::Extrusion(_) => "swept",
                Surface::Revolution(_) => "revolved",
                _ => "ruled",
            }
        ))),
    }
}

/// The span of `f` along `frame.z`, divided by `per` (a cone's `cos` of its half angle, so the
/// span is in its generator's arc length), with a margin either side.
fn heights(b: &Brep, f: FaceId, frame: &Frame, per: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &l in &b.face(f).loops {
        for &c in &b.lp(l).coedges {
            let e = b.edge(b.coedge(c).edge);
            let n = 16;
            for k in 0..=n {
                let p = match &e.curve {
                    Some(c) => c.eval(e.range.0 + (e.range.1 - e.range.0) * k as f64 / n as f64),
                    None => b.vertex(e.start).point,
                };
                let h = (p - frame.origin).dot(frame.z) / per;
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
    }
    let pad = 0.05 * (hi - lo).max(1.0);
    (lo - pad, hi + pad)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{PI, TAU};

    use super::*;
    use crate::{
        check,
        math::Vec2,
        measure,
        profile::{NamedSegment, Profile, ProfileLoop, Segment},
        sweep::{extrude, revolve},
    };

    fn named(segment: Segment) -> NamedSegment {
        NamedSegment::new(segment, "s")
    }

    /// `segments` turned once about z, sketched in the plane through z and x.
    fn turned(segments: Vec<Segment>) -> Brep {
        let p = Profile {
            loops: vec![ProfileLoop {
                segments: segments.into_iter().map(named).collect(),
            }],
        };
        let frame = Frame::new(Point3::ZERO, -Vec3::Y, Vec3::X).unwrap();
        revolve(&p, &frame, Point3::ZERO, Vec3::Z, TAU, "r").unwrap()
    }

    fn maps() -> [Affine; 3] {
        let t = Vec3::new(1.0, -2.0, 0.5);
        [
            Affine {
                m: [[1.5, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 1.2]],
                t,
            },
            Affine {
                m: [[1.5, 0.6, 0.0], [0.0, 0.7, 0.0], [0.3, 0.0, 1.2]],
                t,
            },
            Affine {
                m: [[-1.5, 0.0, 0.4], [0.0, 0.7, 0.0], [0.0, 0.0, 1.2]],
                t,
            },
        ]
    }

    /// Each map scales every volume by its determinant, and keeps the solid valid.
    fn holds(b: &Brep, want: f64) {
        for a in maps() {
            let out = affine(b, &a).unwrap();
            assert!(check::check(&out).is_empty(), "{:?}", check::check(&out));
            let v = measure::volume(&out);
            let w = want * a.det().abs();
            assert!((v - w).abs() < 1e-6 * w, "{v} against {w} under {a:?}");
        }
    }

    #[test]
    fn a_ring_sheared_far_from_the_origin_keeps_its_volume_times_the_determinant() {
        // One vertex gives the ring no size to measure it to; its box does.
        let b = turned(vec![Segment::Circle {
            center: Vec2::new(2.0, 0.0),
            radius: 0.75,
            ccw: true,
        }]);
        let a = Affine {
            m: [[1.0, 0.7, -0.7], [-0.4, -1.7, -0.8], [-0.25, 0.7, -1.3]],
            t: Vec3::new(36.0, 20.0, 14.0),
        };
        let out = affine(&b, &a).unwrap();
        let want = 2.0 * PI * PI * 2.0 * 0.75 * 0.75 * a.det().abs();
        let v = measure::volume(&out);
        assert!((v - want).abs() < 1e-9 * want, "{v} against {want}");
    }

    #[test]
    fn a_stretched_cylinder_keeps_its_volume_times_the_determinant() {
        let b = extrude(&Profile::circle(Vec2::ZERO, 4.0), &Frame::WORLD, 3.0, "c").unwrap();
        holds(&b, PI * 4.0 * 3.0);
    }

    #[test]
    fn a_stretched_ball_keeps_its_volume_times_the_determinant() {
        let b = turned(vec![
            Segment::Arc {
                center: Vec2::ZERO,
                radius: 2.0,
                start: -PI / 2.0,
                sweep: PI,
            },
            Segment::Line {
                a: Vec2::new(0.0, 2.0),
                b: Vec2::new(0.0, -2.0),
            },
        ]);
        holds(&b, 4.0 / 3.0 * PI * 8.0);
    }

    #[test]
    fn a_stretched_ring_keeps_its_volume_times_the_determinant() {
        let b = turned(vec![Segment::Circle {
            center: Vec2::new(3.0, 0.0),
            radius: 1.0,
            ccw: true,
        }]);
        holds(&b, 2.0 * PI * PI * 3.0);
    }

    #[test]
    fn a_stretched_cone_to_a_point_keeps_its_volume_times_the_determinant() {
        let b = turned(vec![
            Segment::Line {
                a: Vec2::new(0.0, 0.0),
                b: Vec2::new(2.0, 0.0),
            },
            Segment::Line {
                a: Vec2::new(2.0, 0.0),
                b: Vec2::new(0.0, 3.0),
            },
            Segment::Line {
                a: Vec2::new(0.0, 3.0),
                b: Vec2::new(0.0, 0.0),
            },
        ]);
        holds(&b, PI * 4.0 * 3.0 / 3.0);
    }
}
