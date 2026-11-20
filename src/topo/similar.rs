//! A solid moved by a similarity: a uniform scale times a rotation or a reflection, then a
//! translation. Frames stay right-handed, which the analytic surfaces' normals rely on, so where a
//! reflection or a scale changes a surface's or curve's parameters, the parameter curves on it
//! change with them.

use super::{Brep, PCurve, pcurve};
use crate::{
    Error, Result,
    geom::{Circle, Cone, Curve, Cylinder, Ellipse, Line, Plane, Sphere, Surface, Torus},
    geom::{Extrusion, Revolution, Ruled},
    math::{Frame, Transform, Vec2, Vec3},
};

fn apply(r: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
        r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
        r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
    )
}

fn det(r: &[[f64; 3]; 3]) -> f64 {
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
}

/// `frame` reflected by `r`, kept right-handed: its x and y reflected and its z their cross
/// product, the reflection of z turned over. A curve or surface in its xy plane keeps its
/// parameters.
fn mirrored_xy(f: &Frame, r: &[[f64; 3]; 3]) -> Frame {
    Frame {
        origin: apply(r, f.origin),
        x: apply(r, f.x),
        y: apply(r, f.y),
        z: -apply(r, f.z),
    }
}

/// `frame` reflected by `r`, kept right-handed by turning its y over: a surface of revolution about
/// its z then runs its angle the other way, `u` becoming `−u`.
fn mirrored_z(f: &Frame, r: &[[f64; 3]; 3]) -> Frame {
    Frame {
        origin: apply(r, f.origin),
        x: apply(r, f.x),
        y: -apply(r, f.y),
        z: apply(r, f.z),
    }
}

/// `c` reflected by `r`, its parameters kept.
fn mirror_curve(c: &Curve, r: &[[f64; 3]; 3]) -> Curve {
    let t = Transform {
        r: *r,
        t: Vec3::ZERO,
    };
    match c {
        Curve::Line(l) => Curve::Line(Line {
            origin: apply(r, l.origin),
            dir: apply(r, l.dir),
        }),
        Curve::Circle(c) => Curve::Circle(Circle {
            frame: mirrored_xy(&c.frame, r),
            radius: c.radius,
        }),
        Curve::Ellipse(e) => Curve::Ellipse(Ellipse {
            frame: mirrored_xy(&e.frame, r),
            ..*e
        }),
        Curve::BSpline(b) => Curve::BSpline(b.transformed(&t)),
        Curve::Nurbs(n) => Curve::Nurbs(Box::new(n.transformed(&t))),
        Curve::Intersection(x) => {
            let mut x = x.as_ref().clone();
            x.approx = x.approx.transformed(&t);
            x.surfaces = x.surfaces.map(|s| mirror_surface(&s, r).0);
            Curve::Intersection(Box::new(x))
        }
    }
}

/// `s` reflected by `r`, and whether its `u` now runs the other way.
fn mirror_surface(s: &Surface, r: &[[f64; 3]; 3]) -> (Surface, bool) {
    let t = Transform {
        r: *r,
        t: Vec3::ZERO,
    };
    match s {
        Surface::Plane(p) => (
            Surface::Plane(Plane {
                frame: mirrored_xy(&p.frame, r),
            }),
            false,
        ),
        Surface::Cylinder(c) => (
            Surface::Cylinder(Cylinder {
                frame: mirrored_z(&c.frame, r),
                ..*c
            }),
            true,
        ),
        Surface::Cone(c) => (
            Surface::Cone(Cone {
                frame: mirrored_z(&c.frame, r),
                ..*c
            }),
            true,
        ),
        Surface::Sphere(c) => (
            Surface::Sphere(Sphere {
                frame: mirrored_z(&c.frame, r),
                ..*c
            }),
            true,
        ),
        Surface::Torus(c) => (
            Surface::Torus(Torus {
                frame: mirrored_z(&c.frame, r),
                ..*c
            }),
            true,
        ),
        Surface::Extrusion(e) => (
            Surface::Extrusion(Box::new(Extrusion {
                curve: mirror_curve(&e.curve, r),
                dir: apply(r, e.dir),
            })),
            false,
        ),
        // A turn about the reflected axis runs the other way.
        Surface::Revolution(v) => (
            Surface::Revolution(Box::new(Revolution {
                curve: mirror_curve(&v.curve, r),
                origin: apply(r, v.origin),
                axis: apply(r, v.axis),
            })),
            true,
        ),
        Surface::Ruled(v) => (
            Surface::Ruled(Box::new(Ruled {
                a: mirror_curve(&v.a, r),
                b: mirror_curve(&v.b, r),
                ..v.as_ref().clone()
            })),
            false,
        ),
        Surface::Nurbs(n) => (Surface::Nurbs(Box::new(n.transformed(&t))), false),
    }
}

fn scaled_frame(f: &Frame, s: f64) -> Frame {
    Frame {
        origin: f.origin * s,
        ..*f
    }
}

/// `c` scaled by `s` about the origin, and the factor its parameter grows by.
fn scale_curve(c: &Curve, s: f64) -> (Curve, f64) {
    let t = Transform {
        r: [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]],
        t: Vec3::ZERO,
    };
    match c {
        // Measured by length along a unit direction.
        Curve::Line(l) => (
            Curve::Line(Line {
                origin: l.origin * s,
                dir: l.dir,
            }),
            s,
        ),
        Curve::Circle(c) => (
            Curve::Circle(Circle {
                frame: scaled_frame(&c.frame, s),
                radius: c.radius * s,
            }),
            1.0,
        ),
        Curve::Ellipse(e) => (
            Curve::Ellipse(Ellipse {
                frame: scaled_frame(&e.frame, s),
                major: e.major * s,
                minor: e.minor * s,
            }),
            1.0,
        ),
        Curve::BSpline(b) => (Curve::BSpline(b.transformed(&t)), 1.0),
        Curve::Nurbs(n) => (Curve::Nurbs(Box::new(n.transformed(&t))), 1.0),
        Curve::Intersection(x) => {
            let mut x = x.as_ref().clone();
            x.approx = x.approx.transformed(&t);
            x.surfaces = x.surfaces.map(|f| scale_surface(&f, s).0);
            (Curve::Intersection(Box::new(x)), 1.0)
        }
    }
}

/// `f` scaled by `s` about the origin, and the factors its two parameters grow by.
fn scale_surface(f: &Surface, s: f64) -> (Surface, Vec2) {
    let t = Transform {
        r: [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]],
        t: Vec3::ZERO,
    };
    match f {
        Surface::Plane(p) => (
            Surface::Plane(Plane {
                frame: scaled_frame(&p.frame, s),
            }),
            Vec2::new(s, s),
        ),
        Surface::Cylinder(c) => (
            Surface::Cylinder(Cylinder {
                frame: scaled_frame(&c.frame, s),
                radius: c.radius * s,
            }),
            Vec2::new(1.0, s),
        ),
        // `v` runs along the generator by length.
        Surface::Cone(c) => (
            Surface::Cone(Cone {
                frame: scaled_frame(&c.frame, s),
                radius: c.radius * s,
                ..*c
            }),
            Vec2::new(1.0, s),
        ),
        Surface::Sphere(c) => (
            Surface::Sphere(Sphere {
                frame: scaled_frame(&c.frame, s),
                radius: c.radius * s,
            }),
            Vec2::new(1.0, 1.0),
        ),
        Surface::Torus(c) => (
            Surface::Torus(Torus {
                frame: scaled_frame(&c.frame, s),
                major: c.major * s,
                minor: c.minor * s,
            }),
            Vec2::new(1.0, 1.0),
        ),
        Surface::Extrusion(e) => {
            let (curve, k) = scale_curve(&e.curve, s);
            (
                Surface::Extrusion(Box::new(Extrusion { curve, dir: e.dir })),
                Vec2::new(k, s),
            )
        }
        Surface::Revolution(v) => {
            let (curve, k) = scale_curve(&v.curve, s);
            (
                Surface::Revolution(Box::new(Revolution {
                    curve,
                    origin: v.origin * s,
                    axis: v.axis,
                })),
                Vec2::new(1.0, k),
            )
        }
        // Each rail's range is mapped onto u in [0, 1], so u and v keep their values.
        Surface::Ruled(v) => {
            let (a, ka) = scale_curve(&v.a, s);
            let (b, kb) = scale_curve(&v.b, s);
            (
                Surface::Ruled(Box::new(Ruled {
                    a,
                    a_range: (v.a_range.0 * ka, v.a_range.1 * ka),
                    b,
                    b_range: (v.b_range.0 * kb, v.b_range.1 * kb),
                })),
                Vec2::new(1.0, 1.0),
            )
        }
        Surface::Nurbs(n) => (
            Surface::Nurbs(Box::new(n.transformed(&t))),
            Vec2::new(1.0, 1.0),
        ),
    }
}

impl Brep {
    /// Moves the arena by the similarity `p ↦ m·p + t`, `m` a uniform scale times a rotation or a
    /// reflection. A stretch or a shear, which would turn circles into ellipses, is unsupported.
    pub fn transform_similar(&mut self, m: [[f64; 3]; 3], t: Vec3) -> Result<()> {
        let d = det(&m);
        let s = d.abs().cbrt();
        if !(s.is_finite() && s > 0.0) {
            return Err(Error::InvalidInput(
                "a transform that flattens the solid".into(),
            ));
        }
        let Some(q) = crate::math::nearest_orthogonal(&m.map(|row| row.map(|x| x / s))) else {
            return Err(Error::InvalidInput(
                "only similarities: a uniform scale with a rotation or a mirror".into(),
            ));
        };
        if d < 0.0 {
            self.mirror(&q);
        } else {
            self.transform(&Transform {
                r: q,
                t: Vec3::ZERO,
            });
        }
        if (s - 1.0).abs() > 0.0 {
            self.scale(s);
        }
        self.transform(&Transform::translation(t));
        Ok(())
    }

    /// Reflects the arena by the orthogonal `r` of determinant −1. Every loop turns round, as a
    /// reflection reverses its sense about the face; a face's natural normal follows its new
    /// surface, so it is turned over where that surface keeps its parameters.
    fn mirror(&mut self, r: &[[f64; 3]; 3]) {
        // Where a loop turns round, each coedge starts at its old end.
        let ends: Vec<Vec2> = (0..self.coedges.len())
            .map(|i| {
                let c = super::CoedgeId::at(i);
                let e = &self.edges[self.coedges[i].edge.index()];
                // Followed along from its start: a parameter curve continues from its seed, and
                // its end read alone could land a period away.
                let (t0, t1) = if self.coedges[i].reversed {
                    (e.range.1, e.range.0)
                } else {
                    e.range
                };
                let ts: Vec<f64> = (0..=32).map(|k| t0 + (t1 - t0) * k as f64 / 32.0).collect();
                *self.coedge_uvs(c, &ts).last().expect("samples")
            })
            .collect();
        for v in &mut self.vertices {
            v.point = apply(r, v.point);
        }
        for e in &mut self.edges {
            if let Some(c) = &e.curve {
                e.curve = Some(mirror_curve(c, r));
            }
        }
        // Where u runs the other way, the period it does so over: u becomes the period less u,
        // the same reflection of the angle kept in the period's range.
        let mut turned: Vec<Option<f64>> = vec![None; self.faces.len()];
        for (i, f) in self.faces.iter_mut().enumerate() {
            let (surface, flips_u) = mirror_surface(&f.surface, r);
            turned[i] = flips_u.then(|| surface.u_period().unwrap_or(0.0));
            f.surface = surface;
            // A surface that keeps its parameters has its natural normal reflected and turned
            // over; one whose u runs the other way has it reflected only.
            if !flips_u {
                f.reversed = !f.reversed;
            }
        }
        let at = |uv: Vec2, on: Option<f64>| on.map_or(uv, |p| Vec2::new(p - uv.x, uv.y));
        let along = |d: Vec2, on: Option<f64>| on.map_or(d, |_| Vec2::new(-d.x, d.y));
        for (i, c) in self.coedges.iter_mut().enumerate() {
            let on = turned[self.loops[c.lp.index()].face.index()];
            c.pcurve = match &c.pcurve {
                PCurve::Affine { origin, dir } => PCurve::Affine {
                    origin: at(*origin, on),
                    dir: along(*dir, on),
                },
                PCurve::Projected { .. } => PCurve::Projected {
                    seed: at(ends[i], on),
                },
                spline @ PCurve::Spline { .. } => spline.mapped(&|uv| at(uv, on)),
            };
            c.reversed = !c.reversed;
        }
        for l in &mut self.loops {
            l.coedges.reverse();
        }
        for m in &mut self.memos.0 {
            *m = std::sync::Arc::default();
        }
    }

    /// Scales the arena by `s` about the origin.
    fn scale(&mut self, s: f64) {
        for v in &mut self.vertices {
            v.point = v.point * s;
        }
        let mut grows = vec![1.0; self.edges.len()];
        for (i, e) in self.edges.iter_mut().enumerate() {
            if let Some(c) = &e.curve {
                let (curve, k) = scale_curve(c, s);
                e.curve = Some(curve);
                e.range = (e.range.0 * k, e.range.1 * k);
                grows[i] = k;
            }
        }
        let mut params = vec![Vec2::new(1.0, 1.0); self.faces.len()];
        for (i, f) in self.faces.iter_mut().enumerate() {
            let (surface, k) = scale_surface(&f.surface, s);
            f.surface = surface;
            params[i] = k;
        }
        for c in &mut self.coedges {
            let k = params[self.loops[c.lp.index()].face.index()];
            let by = |uv: Vec2| Vec2::new(uv.x * k.x, uv.y * k.y);
            c.pcurve = match &c.pcurve {
                PCurve::Affine { origin, dir } => PCurve::Affine {
                    origin: by(*origin),
                    dir: by(*dir) / grows[c.edge.index()],
                },
                PCurve::Projected { seed } => PCurve::Projected { seed: by(*seed) },
                PCurve::Spline { uv } => PCurve::Spline {
                    uv: pcurve::stretch_spline(&pcurve::map_spline(uv, &by), grows[c.edge.index()]),
                },
            };
        }
        for t in self.tolerances.vertices.values_mut() {
            *t *= s;
        }
        for t in self.tolerances.edges.values_mut() {
            *t *= s;
        }
        for m in &mut self.memos.0 {
            *m = std::sync::Arc::default();
        }
    }
}
