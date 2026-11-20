//! Solids sewn from pieces of given surfaces: each [`Piece`] a surface and the loops of curves that
//! bound it, and [`sew`] joins the pieces along the edges they share into one closed solid, as a
//! STEP file's open shells are sewn (doc/design.md, "Sewing").

use std::f64::consts::TAU;

use crate::{
    Error, Result, check,
    error::invalid,
    geom::{Circle, Curve, NurbsCurve, NurbsSurface, Plane, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    step::{PartFace, Parts, sewn},
    tol,
    topo::{Brep, FaceName},
};

mod coons;
pub use coons::coons;

/// A stretch of a curve, from parameter `t0` to `t1` in either order: a loop runs along the curve
/// where `t0 < t1` and against it otherwise.
#[derive(Clone, Debug)]
pub struct Span {
    pub curve: Curve,
    pub t0: f64,
    pub t1: f64,
}

impl Span {
    pub fn new(curve: Curve, t0: f64, t1: f64) -> Span {
        Span { curve, t0, t1 }
    }

    /// The straight stretch from `a` to `b`.
    pub fn line(a: Point3, b: Point3) -> Result<Span> {
        let Some((line, len)) = crate::geom::Line::through(a, b) else {
            return invalid("a straight span between two points that coincide");
        };
        Ok(Span::new(Curve::Line(line), 0.0, len))
    }

    fn start(&self) -> Point3 {
        self.curve.eval(self.t0)
    }

    fn end(&self) -> Point3 {
        self.curve.eval(self.t1)
    }

    /// Points along the span in its own direction, at least `per` to each piece between
    /// breakpoints.
    fn samples(&self, per: usize) -> Vec<Point3> {
        let (a, z) = (self.t0.min(self.t1), self.t0.max(self.t1));
        let mut cuts = vec![a];
        cuts.extend(self.curve.breakpoints(a, z));
        cuts.push(z);
        let mut ts: Vec<f64> = cuts
            .windows(2)
            .flat_map(|w| (0..per).map(move |k| w[0] + (w[1] - w[0]) * k as f64 / per as f64))
            .collect();
        ts.push(z);
        if self.t0 > self.t1 {
            ts.reverse();
        }
        ts.into_iter().map(|t| self.curve.eval(t)).collect()
    }
}

/// A bounded piece of one surface, to be a face of a sewn solid: its loops of spans, each closed
/// head to tail, the outer loop first.
#[derive(Clone, Debug)]
pub struct Piece {
    surface: Surface,
    loops: Vec<Vec<Span>>,
    name: String,
    /// Whether the outer loop runs clockwise in the parameter plane, where that is known.
    reversed: Option<bool>,
}

impl Piece {
    /// The piece of `surface` inside `loops`, the outer loop first. Which way the outer loop
    /// runs in the surface's parameters sets which side of the surface is the piece's front; the
    /// sewing turns pieces to agree with each other.
    pub fn bounded(surface: Surface, loops: Vec<Vec<Span>>, name: &str) -> Piece {
        Piece {
            surface,
            loops,
            name: name.into(),
            reversed: None,
        }
    }

    /// The whole of a B-spline surface, bounded by its four boundary parameter lines. A boundary
    /// line whose control points are all one point is a pole and bounds nothing; two boundary
    /// lines that lie on each other (a surface closed round a parameter) become one seam.
    pub fn whole(surface: NurbsSurface, name: &str) -> Result<Piece> {
        let ((u0, u1), (v0, v1)) = surface.domain();
        let sides = [
            (false, v0, u0, u1),
            (true, u1, v0, v1),
            (false, v1, u1, u0),
            (true, u0, v1, v0),
        ];
        let mut outer = Vec::new();
        for (along_v, at, t0, t1) in sides {
            let line = surface.iso_curve(along_v, at);
            let ctrl = line.control_points();
            if ctrl.iter().all(|p| p.dist(ctrl[0]) <= tol::LINEAR) {
                continue;
            }
            outer.push(simplest(Span::new(Curve::Nurbs(Box::new(line)), t0, t1)));
        }
        if outer.is_empty() {
            return invalid("a B-spline surface whose every boundary line is one point");
        }
        Ok(Piece {
            surface: Surface::Nurbs(Box::new(surface)),
            loops: vec![outer],
            name: name.into(),
            reversed: Some(false),
        })
    }

    /// The flat piece inside `loops`, the outer loop first, which must all lie in one plane. Its
    /// front faces the side the outer loop runs counter-clockwise about.
    pub fn planar(loops: Vec<Vec<Span>>, name: &str) -> Result<Piece> {
        let Some(outer) = loops.first() else {
            return invalid("a flat piece with no loop");
        };
        let ring: Vec<Point3> = outer.iter().flat_map(|s| s.samples(8)).collect();
        let centre = ring.iter().fold(Vec3::ZERO, |a, &p| a + p) / ring.len() as f64;
        // Newell's normal: the polygon's area vector, exact for points in one plane.
        let mut n = Vec3::ZERO;
        for (i, &p) in ring.iter().enumerate() {
            n += (p - centre).cross(ring[(i + 1) % ring.len()] - centre);
        }
        let Some(z) = n.unit() else {
            return invalid("a flat piece whose outer loop encloses no area");
        };
        let frame = Frame::from_axis(centre, z)?;
        let off = loops
            .iter()
            .flatten()
            .flat_map(|s| s.samples(8))
            .map(|p| (p - centre).dot(z).abs())
            .fold(0.0, f64::max);
        if off > tol::LINEAR {
            return invalid(format!(
                "a flat piece whose loops leave their plane by {off:.1e} mm"
            ));
        }
        Ok(Piece {
            surface: Surface::Plane(Plane { frame }),
            loops,
            name: name.into(),
            reversed: Some(false),
        })
    }

    /// The loops that bound the piece, the outer loop first.
    pub fn loops(&self) -> &[Vec<Span>] {
        &self.loops
    }

    /// Whether the outer loop runs clockwise in the surface's parameters, each sample projected
    /// near the one before.
    fn runs_clockwise(&self) -> bool {
        self.reversed
            .unwrap_or_else(|| area(&self.surface, &self.loops[0]) < 0.0)
    }
}

/// `s` as a line or a circle where its curve is a whole B-spline that traces one, as sweeps choose
/// the analytic form wherever one exists; otherwise `s`.
fn simplest(s: Span) -> Span {
    let Curve::Nurbs(n) = &s.curve else {
        return s;
    };
    let back = s.t0 > s.t1;
    if (s.t0.min(s.t1), s.t0.max(s.t1)) != n.domain() {
        return s;
    }
    straight(n.control_points(), back)
        .or_else(|| circular(n, back))
        .unwrap_or(s)
}

/// The straight span a B-spline whose control points lie in order along one line traces, run
/// backward when `back`: with positive weights it never leaves their hull, and a coordinate whose
/// coefficients only rise only rises (de Boor 1978).
fn straight(ctrl: &[Point3], back: bool) -> Option<Span> {
    let (a, z) = (ctrl[0], ctrl[ctrl.len() - 1]);
    let (line, len) = crate::geom::Line::through(a, z)?;
    let mut last = 0.0;
    for &p in ctrl {
        let t = (p - a).dot(line.dir);
        if t < last || (a + line.dir * t).dist(p) > tol::LINEAR {
            return None;
        }
        last = t;
    }
    let (t0, t1) = if back { (len, 0.0) } else { (0.0, len) };
    Some(Span::new(Curve::Line(line), t0, t1))
}

/// The circular span a quadratic B-spline traces, run backward when `back`, where it is one: each
/// knot span is a conic, which five of its points on a circle make that circle.
fn circular(c: &NurbsCurve, back: bool) -> Option<Span> {
    if c.degree() != 2 {
        return None;
    }
    let (a, z) = c.domain();
    let mut ts = vec![a];
    ts.extend(c.breakpoints(a, z));
    ts.push(z);
    let samples: Vec<Point3> = ts
        .windows(2)
        .flat_map(|w| (0..6).map(move |k| w[0] + (w[1] - w[0]) * k as f64 / 6.0))
        .chain([z])
        .map(|t| c.eval(t))
        .collect();
    let at = |f: f64| c.eval(a + (z - a) * f);
    let (p, q, r) = (at(0.0), at(1.0 / 3.0), at(2.0 / 3.0));
    let n = (q - p).cross(r - p);
    let axis = n.unit()?;
    // The circumcentre of p, q, r.
    let (u, v) = (q - p, r - p);
    let centre = p + (v.cross(n) * u.dot(u) + n.cross(u) * v.dot(v)) / (2.0 * n.dot(n));
    let radius = p.dist(centre);
    let frame = Frame::new(centre, axis, p - centre).ok()?;
    let mut sweep = 0.0;
    let mut last = frame.to_local(p);
    for &s in &samples {
        if ((s - centre).dot(axis)).abs() > tol::LINEAR
            || (s.dist(centre) - radius).abs() > tol::LINEAR
        {
            return None;
        }
        let l = frame.to_local(s);
        let turn = (last.x * l.y - last.y * l.x).atan2(last.x * l.x + last.y * l.y);
        if turn < 0.0 {
            return None;
        }
        sweep += turn;
        last = l;
    }
    if (sweep - TAU).abs() * radius <= tol::LINEAR {
        sweep = TAU;
    }
    let (t0, t1) = if back { (sweep, 0.0) } else { (0.0, sweep) };
    Some(Span::new(Curve::Circle(Circle { frame, radius }), t0, t1))
}

/// The signed area a loop encloses in the parameter plane of `s`.
fn area(s: &Surface, lp: &[Span]) -> f64 {
    let mut uvs: Vec<Vec2> = Vec::new();
    for p in lp.iter().flat_map(|sp| sp.samples(8)) {
        let uv = match uvs.last() {
            Some(&prev) => s.project_near(p, prev),
            None => s.project(p),
        };
        uvs.push(uv);
    }
    (0..uvs.len())
        .map(|i| uvs[i].cross(uvs[(i + 1) % uvs.len()]))
        .sum::<f64>()
}

/// The solid bounded by `pieces`, sewn along the edges they share: the ends of spans within
/// `tolerance` of each other become one vertex, spans that then share their ends and lie together
/// become one edge, and the pieces are turned to agree across their edges and to face out of the
/// volume they close. Each face is named `feature.<piece's name>`. Pieces that do not close a
/// volume, or meet anywhere but along their shared edges, are an error.
pub fn sew(pieces: &[Piece], tolerance: f64, feature: &str) -> Result<Brep> {
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return invalid("a sewing tolerance that is not a finite length");
    }
    let tolerance = tolerance.max(tol::LINEAR);
    if pieces.is_empty() {
        return invalid("nothing to sew");
    }
    let mut parts = Parts::default();
    for (k, piece) in pieces.iter().enumerate() {
        let reversed = piece.runs_clockwise();
        let mut loops = Vec::new();
        for (l, lp) in piece.loops.iter().enumerate() {
            if lp.is_empty() {
                return invalid(format!("piece {k} ({}) has an empty loop", piece.name));
            }
            let mut lp: Vec<Span> = lp.iter().cloned().map(simplest).collect();
            // A hole runs the other way round from the outer loop, whichever way it was given.
            if l > 0 && (area(&piece.surface, &lp) < 0.0) == reversed {
                lp = lp
                    .into_iter()
                    .rev()
                    .map(|s| Span::new(s.curve, s.t1, s.t0))
                    .collect();
            }
            for (i, s) in lp.iter().enumerate() {
                let next = &lp[(i + 1) % lp.len()];
                let gap = s.end().dist(next.start());
                if gap > tolerance {
                    return invalid(format!(
                        "a loop of piece {k} ({}) is open by {gap:.1e} mm after its span {i}",
                        piece.name
                    ));
                }
            }
            let mut uses = Vec::new();
            for s in &lp {
                let (a, z) = (s.t0.min(s.t1), s.t0.max(s.t1));
                let start = parts.points.len();
                parts.points.push(s.curve.eval(a));
                parts.points.push(s.curve.eval(z));
                uses.push((parts.edges.len(), s.t0 > s.t1));
                parts.edges.push((s.curve.clone(), start, start + 1));
            }
            loops.push(uses);
        }
        parts.faces.push(PartFace {
            surface: piece.surface.clone(),
            reversed,
            loops,
            names: [FaceName::new(feature, piece.name.clone())].into(),
            shell: 0,
        });
    }
    let b = sewn(&parts, tolerance)?;
    let issues = check::check(&b);
    if issues.is_empty() {
        Ok(b)
    } else {
        Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.to_string()).collect(),
        ))
    }
}

#[cfg(test)]
mod tests;
