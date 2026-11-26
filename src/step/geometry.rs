//! STEP geometry entities into kernel curves, surfaces and placements, in millimetres.

use std::collections::{HashMap, HashSet};

use super::part21::{Entity, File, Param};
use crate::{
    Error, Result,
    geom::{
        Circle, Cone, Curve, Cylinder, Ellipse, Extrusion, Line, NurbsCurve, NurbsSurface, Plane,
        Revolution, Sphere, Surface, Torus,
    },
    math::{Frame, Point3, Transform, Vec3},
    tol,
    topo::Brep,
};

/// The largest shape factor w₁²/(w₀w₂) of one piece of a hyperbola's arc read from a file.
const CONIC_SHAPE: f64 = 2.5;

fn bad<T>(id: u64, what: &str) -> Result<T> {
    Err(Error::InvalidInput(format!("STEP #{id}: {what}")))
}

fn unsupported<T>(id: u64, what: &str) -> Result<T> {
    Err(Error::Unsupported(format!("STEP #{id}: {what}")))
}

/// The scale factors of a representation context: millimetres per length unit and radians per
/// angle unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Units {
    pub length: f64,
    pub angle: f64,
}

impl Units {
    pub(super) const MM_RAD: Units = Units {
        length: 1.0,
        angle: 1.0,
    };
}

fn si_prefix(p: &Param) -> f64 {
    match p.as_enum() {
        Some("EXA") => 1e18,
        Some("PETA") => 1e15,
        Some("TERA") => 1e12,
        Some("GIGA") => 1e9,
        Some("MEGA") => 1e6,
        Some("KILO") => 1e3,
        Some("HECTO") => 1e2,
        Some("DECA") => 1e1,
        Some("DECI") => 1e-1,
        Some("CENTI") => 1e-2,
        Some("MILLI") => 1e-3,
        Some("MICRO") => 1e-6,
        Some("NANO") => 1e-9,
        Some("PICO") => 1e-12,
        _ => 1.0,
    }
}

/// The basis is orthonormal and can be left-handed. Scale is built into the solid's units before
/// the basis and origin are applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Placing {
    basis: [Vec3; 3],
    origin: Point3,
    pub scale: f64,
}

impl Placing {
    pub(super) const IDENTITY: Placing = Placing {
        basis: [Vec3::X, Vec3::Y, Vec3::Z],
        origin: Point3::ZERO,
        scale: 1.0,
    };

    fn from_frame(frame: Frame, scale: f64) -> Placing {
        Placing {
            basis: [frame.x, frame.y, frame.z],
            origin: frame.origin,
            scale,
        }
    }

    fn vector(&self, v: Vec3) -> Vec3 {
        let [x, y, z] = self.basis;
        x * v.x + y * v.y + z * v.z
    }

    /// Applies the basis and origin to a point already in scaled units.
    pub(super) fn place_scaled_point(&self, p: Point3) -> Point3 {
        self.origin + self.vector(p)
    }

    fn matrix(&self) -> [[f64; 3]; 3] {
        let [x, y, z] = self.basis;
        [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]]
    }

    /// Applies the basis and origin to a body already built in scaled units.
    pub(super) fn place_scaled_body(&self, body: &mut Brep) -> Result<()> {
        let [x, y, z] = self.basis;
        let r = self.matrix();
        if x.cross(y).dot(z) < 0.0 {
            body.transform_similar(r, self.origin)
        } else {
            body.transform(&Transform { r, t: self.origin });
            Ok(())
        }
    }

    /// `self` followed by `next`.
    pub(super) fn then(&self, next: &Placing) -> Placing {
        Placing {
            basis: self.basis.map(|axis| next.vector(axis)),
            origin: next.place_scaled_point(self.origin * next.scale),
            scale: self.scale * next.scale,
        }
    }

    pub(super) fn inverse(&self) -> Placing {
        let mut inv = Placing {
            basis: self.matrix().map(Vec3::from_array),
            origin: Point3::ZERO,
            scale: 1.0 / self.scale,
        };
        inv.origin = -inv.vector(self.origin) / self.scale;
        inv
    }
}

/// Converts the geometry of one file, caching each entity's result.
pub(super) struct Geometry<'a> {
    pub file: &'a File,
    units: Units,
    points: HashMap<u64, Point3>,
    curves: HashMap<u64, Curve>,
    surfaces: HashMap<u64, Surface>,
    /// The distance within which the current context's geometry meets, in mm, where it says.
    uncertainty: Option<f64>,
    /// Instances being converted, so a reference cycle is an error rather than endless recursion.
    open: HashSet<u64>,
}

impl<'a> Geometry<'a> {
    pub(super) fn new(file: &'a File) -> Geometry<'a> {
        Geometry {
            file,
            units: Units::MM_RAD,
            points: HashMap::new(),
            curves: HashMap::new(),
            surfaces: HashMap::new(),
            uncertainty: None,
            open: HashSet::new(),
        }
    }

    /// The distance within which the current context says its geometry meets, in mm: its
    /// `distance_accuracy_value`, or the largest length uncertainty it gives.
    pub(super) fn uncertainty(&self) -> Option<f64> {
        self.uncertainty
    }

    /// Switches to the units of representation context `ctx`, clearing the caches if they change.
    pub(super) fn use_context(&mut self, ctx: Option<u64>) -> Result<()> {
        self.use_scaled(ctx, 1.0)
    }

    /// Switches to the units of `ctx` with every length times `scale`.
    pub(super) fn use_scaled(&mut self, ctx: Option<u64>, scale: f64) -> Result<()> {
        let mut units = match ctx {
            Some(c) => self.context_units(c)?,
            None => Units::MM_RAD,
        };
        units.length *= scale;
        self.uncertainty = match ctx {
            Some(c) => self.context_uncertainty(c).map(|u| u * scale),
            None => None,
        };
        if units != self.units {
            self.units = units;
            self.points.clear();
            self.curves.clear();
            self.surfaces.clear();
        }
        Ok(())
    }

    fn entity(&self, id: u64) -> Result<&'a Entity> {
        match self.file.get(id) {
            Some(e) => Ok(e),
            None => bad(id, "missing instance"),
        }
    }

    fn context_units(&self, ctx: u64) -> Result<Units> {
        let e = self.entity(ctx)?;
        let mut units = Units::MM_RAD;
        let Some(p) = e.part("GLOBAL_UNIT_ASSIGNED_CONTEXT") else {
            return Ok(units);
        };
        for u in p.first().and_then(Param::as_list).unwrap_or(&[]) {
            let Some(uid) = u.as_ref() else { continue };
            let ue = self.entity(uid)?;
            if ue.has("LENGTH_UNIT") {
                units.length = self.unit_factor(uid, 1000.0)?;
            } else if ue.has("PLANE_ANGLE_UNIT") {
                units.angle = self.unit_factor(uid, 1.0)?;
            }
        }
        Ok(units)
    }

    /// A context's length uncertainty in mm: the one named `distance_accuracy_value`, otherwise the
    /// largest given in a length unit.
    fn context_uncertainty(&self, ctx: u64) -> Option<f64> {
        let p = self
            .entity(ctx)
            .ok()?
            .part("GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT")?;
        let mut named = None;
        let mut largest: Option<f64> = None;
        for u in p.first().and_then(Param::as_list).unwrap_or(&[]) {
            let Some(q) = u.as_ref().and_then(|id| self.file.get(id)) else {
                continue;
            };
            let Some(a) = q.part("UNCERTAINTY_MEASURE_WITH_UNIT") else {
                continue;
            };
            let (Some(v), Some(unit)) = (
                a.first().and_then(Param::as_f64),
                a.get(1).and_then(Param::as_ref),
            ) else {
                continue;
            };
            if !self.entity(unit).is_ok_and(|e| e.has("LENGTH_UNIT")) {
                continue;
            }
            let Ok(mm) = self.unit_factor(unit, 1000.0) else {
                continue;
            };
            let d = v.abs() * mm;
            if !(d.is_finite() && d > 0.0) {
                continue;
            }
            if a.get(2)
                .and_then(Param::as_str)
                .is_some_and(|n| n.eq_ignore_ascii_case("distance_accuracy_value"))
            {
                named = Some(d);
            }
            largest = Some(largest.map_or(d, |l| l.max(d)));
        }
        named.or(largest)
    }

    /// The size of unit `id` in millimetres (lengths, `si_base` 1000) or radians (angles, 1).
    fn unit_factor(&self, id: u64, si_base: f64) -> Result<f64> {
        self.unit_factor_within(id, si_base, 8)
    }

    fn unit_factor_within(&self, id: u64, si_base: f64, depth: usize) -> Result<f64> {
        if depth == 0 {
            return bad(id, "conversion-based units nest too deeply");
        }
        let e = self.entity(id)?;
        if let Some(si) = e.part("SI_UNIT") {
            return Ok(si_base * si.first().map_or(1.0, si_prefix));
        }
        if let Some(cb) = e.part("CONVERSION_BASED_UNIT") {
            let Some(m) = cb.get(1).and_then(Param::as_ref) else {
                return bad(id, "conversion-based unit without a measure");
            };
            let me = self.entity(m)?;
            let p = &me.parts[0].1;
            let (Some(v), Some(base)) = (
                p.first().and_then(Param::as_f64),
                p.get(1).and_then(Param::as_ref),
            ) else {
                return bad(m, "malformed unit measure");
            };
            return Ok(v * self.unit_factor_within(base, si_base, depth - 1)?);
        }
        bad(id, "unknown unit")
    }

    fn params(&self, id: u64, kind: &str) -> Result<&'a [Param]> {
        match self.entity(id)?.part(kind) {
            Some(p) => Ok(p),
            None => bad(id, &format!("expected {kind}")),
        }
    }

    fn reals(&self, id: u64, p: Option<&Param>) -> Result<Vec<f64>> {
        let Some(l) = p.and_then(Param::as_list) else {
            return bad(id, "expected a list of numbers");
        };
        l.iter()
            .map(|x| x.as_f64().ok_or(()))
            .collect::<std::result::Result<_, _>>()
            .or_else(|_| bad(id, "expected numbers"))
    }

    /// Millimetres per length unit of the current context.
    pub(super) fn length_unit(&self) -> f64 {
        self.units.length
    }

    /// The numbers of a list parameter, as `reals` reads them.
    pub(super) fn numbers(&self, id: u64, p: Option<&Param>) -> Result<Vec<f64>> {
        self.reals(id, p)
    }

    pub(super) fn real(&self, id: u64, p: Option<&Param>) -> Result<f64> {
        match p.and_then(Param::as_f64) {
            Some(x) => Ok(x),
            None => bad(id, "expected a number"),
        }
    }

    pub(super) fn reference(&self, id: u64, p: Option<&Param>) -> Result<u64> {
        match p.and_then(Param::as_ref) {
            Some(r) => Ok(r),
            None => bad(id, "expected an instance reference"),
        }
    }

    fn length(&self, id: u64, p: Option<&Param>) -> Result<f64> {
        Ok(self.real(id, p)? * self.units.length)
    }

    /// Runs `f` on instance `id` unless `id` is already being converted further up the stack.
    fn enter<T>(&mut self, id: u64, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if !self.open.insert(id) {
            return bad(id, "refers to itself");
        }
        let out = f(self);
        self.open.remove(&id);
        out
    }

    pub(super) fn point(&mut self, id: u64) -> Result<Point3> {
        if let Some(&p) = self.points.get(&id) {
            return Ok(p);
        }
        let p = self.enter(id, |s| s.make_point(id))?;
        self.points.insert(id, p);
        Ok(p)
    }

    fn make_point(&mut self, id: u64) -> Result<Point3> {
        let e = self.entity(id)?;
        match e.part("CARTESIAN_POINT") {
            Some(a) => {
                let c = self.coords(id, a.get(1))?;
                Ok(c * self.units.length)
            }
            None => match e.part("VERTEX_POINT") {
                Some(a) => self.point(self.reference(id, a.get(1))?),
                None => bad(id, "expected a point"),
            },
        }
    }

    /// Three finite coordinates; a 2-D point or direction is an error here, never padded.
    fn coords(&self, id: u64, p: Option<&Param>) -> Result<Vec3> {
        let c = self.reals(id, p)?;
        if c.len() != 3 || c.iter().any(|x| !x.is_finite()) {
            return bad(id, "expected three finite coordinates");
        }
        Ok(Vec3::new(c[0], c[1], c[2]))
    }

    pub(super) fn direction(&self, id: u64) -> Result<Vec3> {
        let a = self.params(id, "DIRECTION")?;
        let v = self.coords(id, a.get(1))?;
        match v.unit() {
            Some(u) => Ok(u),
            None => bad(id, "zero direction"),
        }
    }

    /// A `VECTOR`'s direction and length.
    fn vector(&self, id: u64) -> Result<(Vec3, f64)> {
        let a = self.params(id, "VECTOR")?;
        let d = self.direction(self.reference(id, a.get(1))?)?;
        Ok((d, self.length(id, a.get(2))?))
    }

    pub(super) fn placement(&mut self, id: u64) -> Result<Frame> {
        let e = self.entity(id)?;
        let Some(a) = e.part("AXIS2_PLACEMENT_3D") else {
            return unsupported(id, &format!("placement {}", e.kind()));
        };
        let origin = self.point(self.reference(id, a.get(1))?)?;
        let z = match a.get(2).and_then(Param::as_ref) {
            Some(r) => self.direction(r)?,
            None => Vec3::Z,
        };
        let x = match a.get(3).and_then(Param::as_ref) {
            Some(r) => self.direction(r)?,
            None if z.x.abs() < 0.9 => Vec3::X,
            None => Vec3::Y,
        };
        Frame::new(origin, z, x)
    }

    /// The motion of an `AXIS2_PLACEMENT_3D` or a `CARTESIAN_TRANSFORMATION_OPERATOR_3D`, which
    /// may also reflect or scale uniformly.
    pub(super) fn transform_of(&mut self, id: u64) -> Result<Placing> {
        let e = self.entity(id)?;
        if e.has("AXIS2_PLACEMENT_3D") {
            return Ok(Placing::from_frame(self.placement(id)?, 1.0));
        }
        if let Some(a) = e.part("CARTESIAN_TRANSFORMATION_OPERATOR_3D") {
            let dir = |s: &Self, i: usize, d: Vec3| -> Result<Vec3> {
                match a.get(i).and_then(Param::as_ref) {
                    Some(r) => s.direction(r),
                    None => Ok(d),
                }
            };
            let origin = self.point(self.reference(id, a.get(4))?)?;
            let scale = a.get(5).and_then(Param::as_f64).unwrap_or(1.0);
            if !(scale > 0.0 && scale.is_finite()) {
                return bad(id, "a transformation's scale must be positive");
            }
            // ISO 10303-42:2021 §§4.5.21, 4.6.6, 4.6.23 and 4.6.34 fix Z first.
            // Project X perpendicular to Z, then use Y's projection to set the handedness.
            let z = dir(self, 6, Vec3::Z)?;
            let default_x = if z.cross(Vec3::X).norm() <= tol::ANGULAR {
                Vec3::Y
            } else {
                Vec3::X
            };
            let x_hint = dir(self, 2, default_x)?;
            let Ok(frame) = Frame::new(origin, z, x_hint) else {
                return bad(id, "a transformation's first axis is parallel to its third");
            };
            let y_projection = dir(self, 3, Vec3::Y)?.dot(frame.y);
            if y_projection.abs() <= tol::ANGULAR {
                return bad(id, "a transformation's axes are coplanar");
            }
            let mut placing = Placing::from_frame(frame, scale);
            if y_projection < 0.0 {
                placing.basis[1] = -placing.basis[1];
            }
            return Ok(placing);
        }
        unsupported(id, &format!("transformation {}", e.kind()))
    }

    pub(super) fn curve(&mut self, id: u64) -> Result<Curve> {
        if let Some(c) = self.curves.get(&id) {
            return Ok(c.clone());
        }
        let c = self.enter(id, |s| s.make_curve(id))?;
        self.curves.insert(id, c.clone());
        Ok(c)
    }

    /// The arc of a `HYPERBOLA` or a `PARABOLA` from `start` to `end`, as the rational curve of
    /// degree 2 that is exactly that arc; `None` for any other curve. Such a conic has no end of
    /// its own, so the edge that names it gives the two points.
    ///
    /// A conic's arc is a rational quadratic Bézier piece whose middle control point is where the
    /// end tangents meet (Piegl and Tiller, *The NURBS Book*, 1997, ch. 7). For the hyperbola
    /// a·cosh u, b·sinh u over a half range d the middle weight is cosh d, and for the parabola it
    /// is 1. A hyperbola's arc is cut into equal ranges of u so that no piece's shape factor cosh² d
    /// passes [`CONIC_SHAPE`]; the pieces meet at double knots, tangent to each other.
    pub(super) fn conic_arc(
        &mut self,
        id: u64,
        start: Point3,
        end: Point3,
    ) -> Result<Option<Curve>> {
        let mut at = id;
        let (kind, a) = loop {
            let (kind, a) = &self.entity(at)?.parts[0];
            match kind.as_str() {
                "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" | "TRIMMED_CURVE" => {
                    at = self.reference(at, a.get(1))?;
                }
                _ => break (kind.as_str(), a),
            }
        };
        if kind != "HYPERBOLA" && kind != "PARABOLA" {
            return Ok(None);
        }
        if start.dist(end) <= tol::LINEAR {
            return bad(at, "an open conic cannot close on itself");
        }
        let f = self.placement(self.reference(at, a.get(1))?)?;
        let (ps, pe) = (f.to_local(start), f.to_local(end));
        let place = |x: f64, y: f64| f.origin + f.x * x + f.y * y;
        let (mut ctrl, mut weights, mut knots) = (vec![start], vec![1.0], vec![0.0; 3]);
        if kind == "HYPERBOLA" {
            let (sa, sb) = (self.length(at, a.get(2))?, self.length(at, a.get(3))?);
            let (u0, u1) = ((ps.y / sb).asinh(), (pe.y / sb).asinh());
            let pieces = ((u1 - u0).abs() / (2.0 * CONIC_SHAPE.sqrt().acosh())).ceil() as usize;
            let pieces = pieces.max(1);
            let d = 0.5 * (u1 - u0) / pieces as f64;
            for k in 0..pieces {
                let m = u0 + (2 * k + 1) as f64 * d;
                ctrl.push(place(sa * m.cosh() / d.cosh(), sb * m.sinh() / d.cosh()));
                weights.push(d.cosh());
                let u = m + d;
                ctrl.push(if k + 1 == pieces {
                    end
                } else {
                    place(sa * u.cosh(), sb * u.sinh())
                });
                weights.push(1.0);
                knots.extend([(k + 1) as f64; 2]);
            }
            knots.push(pieces as f64);
        } else {
            let focal = self.length(at, a.get(2))?;
            let (u0, u1) = (ps.y / (2.0 * focal), pe.y / (2.0 * focal));
            ctrl.extend([place(focal * u0 * u1, focal * (u0 + u1)), end]);
            weights.extend([1.0, 1.0]);
            knots.extend([1.0; 3]);
        }
        let c = NurbsCurve::new(2, knots, ctrl, Some(weights))?;
        Ok(Some(Curve::Nurbs(Box::new(c))))
    }

    fn make_curve(&mut self, id: u64) -> Result<Curve> {
        let e = self.entity(id)?;
        if SPLINE_CURVES.iter().any(|k| e.has(k)) {
            return self.bspline_curve(id, e).map(|c| Curve::Nurbs(Box::new(c)));
        }
        let (kind, a) = &e.parts[0];
        match kind.as_str() {
            "LINE" => {
                let origin = self.point(self.reference(id, a.get(1))?)?;
                let (dir, _) = self.vector(self.reference(id, a.get(2))?)?;
                Ok(Curve::Line(Line { origin, dir }))
            }
            "CIRCLE" => Ok(Curve::Circle(Circle {
                frame: self.placement(self.reference(id, a.get(1))?)?,
                radius: self.length(id, a.get(2))?,
            })),
            "ELLIPSE" => Ok(Curve::Ellipse(Ellipse {
                frame: self.placement(self.reference(id, a.get(1))?)?,
                major: self.length(id, a.get(2))?,
                minor: self.length(id, a.get(3))?,
            })),
            "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" | "TRIMMED_CURVE" => {
                self.curve(self.reference(id, a.get(1))?)
            }
            other => unsupported(id, &format!("curve {other}")),
        }
    }

    fn knot_vector(
        &self,
        id: u64,
        mults: Option<&Param>,
        knots: Option<&Param>,
    ) -> Result<Vec<f64>> {
        let m = self.reals(id, mults)?;
        let k = self.reals(id, knots)?;
        if m.len() != k.len() {
            return bad(id, "knot and multiplicity counts differ");
        }
        let mut out = Vec::new();
        for (&mi, &ki) in m.iter().zip(&k) {
            if mi < 0.0 || mi.fract() != 0.0 {
                return bad(id, "bad knot multiplicity");
            }
            out.extend(std::iter::repeat_n(ki, mi as usize));
        }
        Ok(out)
    }

    fn degree(&self, id: u64, p: Option<&Param>) -> Result<usize> {
        match p.and_then(Param::as_i64) {
            Some(d) if d > 0 => Ok(d as usize),
            _ => bad(id, "bad B-spline degree"),
        }
    }

    fn bspline_curve(&mut self, id: u64, e: &Entity) -> Result<NurbsCurve> {
        let kind = *SPLINE_CURVES
            .iter()
            .find(|k| e.has(k))
            .expect("the caller found one");
        let k = self.params(id, kind)?;
        // A simple instance carries name, degree, points, form, closed, self-intersect, then the
        // knot attributes where it writes them out; a complex one splits them between
        // B_SPLINE_CURVE and its kind.
        let written = kind == SPLINE_CURVES[0];
        let (head, knots): (&[Param], &[Param]) = match e.part("B_SPLINE_CURVE") {
            Some(h) => (h, k),
            None if k.len() >= if written { 9 } else { 6 } => (&k[1..6], &k[6..]),
            None => return bad(id, &format!("short {kind}")),
        };
        let degree = self.degree(id, head.first())?;
        let mut ctrl = Vec::new();
        for p in head.get(1).and_then(Param::as_list).unwrap_or(&[]) {
            ctrl.push(self.point(self.reference(id, Some(p))?)?);
        }
        let knots = if written {
            self.knot_vector(id, knots.first(), knots.get(1))?
        } else {
            implied_knots(id, kind, degree, ctrl.len())?
        };
        let weights = match e.part("RATIONAL_B_SPLINE_CURVE") {
            Some(w) => Some(self.reals(id, w.first())?),
            None => None,
        };
        NurbsCurve::new(degree, knots, ctrl, weights).map_err(|err| at(id, err))
    }

    pub(super) fn surface(&mut self, id: u64) -> Result<Surface> {
        if let Some(s) = self.surfaces.get(&id) {
            return Ok(s.clone());
        }
        let s = self.enter(id, |g| g.make_surface(id))?;
        self.surfaces.insert(id, s.clone());
        Ok(s)
    }

    fn make_surface(&mut self, id: u64) -> Result<Surface> {
        let e = self.entity(id)?;
        if SPLINE_SURFACES.iter().any(|k| e.has(k)) {
            return self
                .bspline_surface(id, e)
                .map(|s| Surface::Nurbs(Box::new(s)));
        }
        let (kind, a) = &e.parts[0];
        let frame = |s: &mut Self| -> Result<Frame> { s.placement(s.reference(id, a.get(1))?) };
        match kind.as_str() {
            "PLANE" => Ok(Surface::Plane(Plane {
                frame: frame(self)?,
            })),
            "CYLINDRICAL_SURFACE" => Ok(Surface::Cylinder(Cylinder {
                frame: frame(self)?,
                radius: self.length(id, a.get(2))?,
            })),
            "CONICAL_SURFACE" => Ok(Surface::Cone(Cone {
                frame: frame(self)?,
                radius: self.length(id, a.get(2))?,
                half_angle: self.real(id, a.get(3))? * self.units.angle,
            })),
            "SPHERICAL_SURFACE" => Ok(Surface::Sphere(Sphere {
                frame: frame(self)?,
                radius: self.length(id, a.get(2))?,
            })),
            "TOROIDAL_SURFACE" => Ok(Surface::Torus(Torus {
                frame: frame(self)?,
                major: self.length(id, a.get(2))?,
                minor: self.length(id, a.get(3))?,
            })),
            "SURFACE_OF_LINEAR_EXTRUSION" => {
                let curve = self.curve(self.reference(id, a.get(1))?)?;
                let (dir, _) = self.vector(self.reference(id, a.get(2))?)?;
                Ok(Surface::Extrusion(Box::new(Extrusion { curve, dir })))
            }
            "SURFACE_OF_REVOLUTION" => {
                let curve = self.curve(self.reference(id, a.get(1))?)?;
                let ax = self.reference(id, a.get(2))?;
                let p = self.params(ax, "AXIS1_PLACEMENT")?;
                let origin = self.point(self.reference(ax, p.get(1))?)?;
                let axis = match p.get(2).and_then(Param::as_ref) {
                    Some(r) => self.direction(r)?,
                    None => Vec3::Z,
                };
                Ok(Surface::Revolution(Box::new(Revolution {
                    curve,
                    origin,
                    axis,
                })))
            }
            "RECTANGULAR_TRIMMED_SURFACE" => self.surface(self.reference(id, a.get(1))?),
            "OFFSET_SURFACE" => {
                let basis = self.surface(self.reference(id, a.get(1))?)?;
                let d = self.length(id, a.get(2))?;
                offset(id, basis, d)
            }
            other => unsupported(id, &format!("surface {other}")),
        }
    }

    fn bspline_surface(&mut self, id: u64, e: &Entity) -> Result<NurbsSurface> {
        let kind = *SPLINE_SURFACES
            .iter()
            .find(|k| e.has(k))
            .expect("the caller found one");
        let k = self.params(id, kind)?;
        let written = kind == SPLINE_SURFACES[0];
        let (head, knots): (&[Param], &[Param]) = match e.part("B_SPLINE_SURFACE") {
            Some(h) => (h, k),
            None if k.len() >= if written { 13 } else { 8 } => (&k[1..8], &k[8..]),
            None => return bad(id, &format!("short {kind}")),
        };
        let du = self.degree(id, head.first())?;
        let dv = self.degree(id, head.get(1))?;
        let rows = head.get(2).and_then(Param::as_list).unwrap_or(&[]);
        let nu = rows.len();
        let nv = rows
            .first()
            .and_then(Param::as_list)
            .map_or(0, <[Param]>::len);
        let mut ctrl = Vec::with_capacity(nu * nv);
        for r in rows {
            let r = r.as_list().unwrap_or(&[]);
            if r.len() != nv {
                return bad(id, "ragged control net");
            }
            for p in r {
                ctrl.push(self.point(self.reference(id, Some(p))?)?);
            }
        }
        let (ku, kv) = if written {
            (
                self.knot_vector(id, knots.first(), knots.get(2))?,
                self.knot_vector(id, knots.get(1), knots.get(3))?,
            )
        } else {
            (
                implied_knots(id, kind, du, nu)?,
                implied_knots(id, kind, dv, nv)?,
            )
        };
        let weights = match e.part("RATIONAL_B_SPLINE_SURFACE") {
            Some(w) => {
                let mut out = Vec::with_capacity(nu * nv);
                for r in w.first().and_then(Param::as_list).unwrap_or(&[]) {
                    out.extend(self.reals(id, Some(r))?);
                }
                Some(out)
            }
            None => None,
        };
        NurbsSurface::new((du, dv), (ku, kv), (nu, nv), ctrl, weights).map_err(|err| at(id, err))
    }
}

/// The B-spline curves the reader takes: one whose knots are written out, then those whose kind
/// implies them.
pub(super) const SPLINE_CURVES: [&str; 4] = [
    "B_SPLINE_CURVE_WITH_KNOTS",
    "UNIFORM_CURVE",
    "QUASI_UNIFORM_CURVE",
    "BEZIER_CURVE",
];

/// The B-spline surfaces the reader takes, as [`SPLINE_CURVES`].
pub(super) const SPLINE_SURFACES: [&str; 4] = [
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "UNIFORM_SURFACE",
    "QUASI_UNIFORM_SURFACE",
    "BEZIER_SURFACE",
];

/// The knots a B-spline of `degree` over `n` control points has where its kind implies them,
/// one knot apart. A uniform spline's are single, from `−degree`. A quasi-uniform spline's ends
/// are each `degree + 1` alike, so it starts and ends on its end points, with single knots
/// between. A Bézier spline is a chain of pieces of `degree`, each `degree` points on from the
/// last: its ends `degree + 1` alike and each join `degree` alike.
pub(super) fn implied_knots(id: u64, kind: &str, degree: usize, n: usize) -> Result<Vec<f64>> {
    if n <= degree {
        return bad(id, "fewer control points than the degree needs");
    }
    let repeat = |x: usize, times: usize| std::iter::repeat_n(x as f64, times);
    let mut knots = Vec::with_capacity(n + degree + 1);
    match kind {
        "UNIFORM_CURVE" | "UNIFORM_SURFACE" => {
            knots.extend((0..n + degree + 1).map(|i| i as f64 - degree as f64));
        }
        "QUASI_UNIFORM_CURVE" | "QUASI_UNIFORM_SURFACE" => {
            let spans = n - degree;
            knots.extend(repeat(0, degree + 1));
            knots.extend((1..spans).map(|i| i as f64));
            knots.extend(repeat(spans, degree + 1));
        }
        "BEZIER_CURVE" | "BEZIER_SURFACE" => {
            if !(n - 1).is_multiple_of(degree) {
                return bad(id, "a Bézier spline's control points are not whole pieces");
            }
            let pieces = (n - 1) / degree;
            knots.extend(repeat(0, degree + 1));
            for j in 1..pieces {
                knots.extend(repeat(j, degree));
            }
            knots.extend(repeat(pieces, degree + 1));
        }
        other => return bad(id, &format!("{other} does not imply its knots")),
    }
    Ok(knots)
}

/// `basis` moved by `d` along its normal, where that is a surface of the same kind; a positive
/// radius must stay positive.
fn offset(id: u64, basis: Surface, d: f64) -> Result<Surface> {
    let grown = |r: f64| -> Result<f64> {
        if r + d > 0.0 {
            Ok(r + d)
        } else {
            bad(id, "an offset surface turns inside out")
        }
    };
    Ok(match basis {
        Surface::Plane(p) => Surface::Plane(Plane {
            frame: p.frame.translated(p.frame.z * d),
        }),
        Surface::Cylinder(c) => Surface::Cylinder(Cylinder {
            radius: grown(c.radius)?,
            ..c
        }),
        Surface::Sphere(s) => Surface::Sphere(Sphere {
            radius: grown(s.radius)?,
            ..s
        }),
        Surface::Torus(t) => Surface::Torus(Torus {
            minor: grown(t.minor)?,
            ..t
        }),
        Surface::Cone(c) => {
            let (sa, ca) = c.half_angle.sin_cos();
            Surface::Cone(Cone {
                frame: c.frame.translated(c.frame.z * (-d * sa)),
                radius: c.radius + d * ca,
                half_angle: c.half_angle,
            })
        }
        other => {
            return unsupported(id, &format!("OFFSET_SURFACE of a {} surface", other.kind()));
        }
    })
}

fn at(id: u64, e: Error) -> Error {
    match e {
        Error::InvalidInput(m) => Error::InvalidInput(format!("STEP #{id}: {m}")),
        other => other,
    }
}
