//! Writing solids as a STEP AP214 file: one part whose advanced B-rep holds every solid of the
//! arena, each face named by the names it carries.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt::Write as _;

use super::{WRITER, part21::encode_string};
use crate::{
    Error, Result,
    geom::{BSpline, Cone, Curve, Ruled, Surface},
    math::{Frame, Point3, Transform, Vec2, Vec3},
    numeric::solve_dense,
    tol,
    topo::{Brep, CoedgeId, EdgeId, FaceId, PCurve, ShellId, SolidId, VertexId},
};

/// A STEP real: always with a decimal point, exponent in upper case.
fn real(x: f64) -> String {
    let s = format!("{x:?}");
    match s.split_once('e') {
        Some((m, e)) if m.contains('.') => format!("{m}E{e}"),
        Some((m, e)) => format!("{m}.E{e}"),
        None if s.contains('.') => s,
        None => format!("{s}."),
    }
}

fn reals(xs: impl IntoIterator<Item = f64>) -> String {
    let v: Vec<String> = xs.into_iter().map(real).collect();
    format!("({})", v.join(","))
}

fn refs(ids: &[usize]) -> String {
    let v: Vec<String> = ids.iter().map(|i| format!("#{i}")).collect();
    format!("({})", v.join(","))
}

/// Distinct knots and their multiplicities.
fn knot_mults(knots: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut k, mut m): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for &x in knots {
        if k.last() == Some(&x) {
            *m.last_mut().unwrap_or(&mut 0.0) += 1.0;
        } else {
            k.push(x);
            m.push(1.0);
        }
    }
    (k, m)
}

fn ints(xs: &[f64]) -> String {
    let v: Vec<String> = xs.iter().map(|x| format!("{}", *x as i64)).collect();
    format!("({})", v.join(","))
}

/// A curve as a B-spline: degree, knots, control points and weights (`None` when polynomial).
struct Spline {
    degree: usize,
    knots: Vec<f64>,
    ctrl: Vec<Point3>,
    weights: Option<Vec<f64>>,
}

pub(super) struct Writer {
    pub(super) out: String,
    next: usize,
    points: HashMap<[u64; 3], usize>,
    directions: HashMap<[u64; 3], usize>,
    /// Each vertex written, once for each fan of faces round it: where pieces touch at a vertex,
    /// each piece has its own.
    vertices: HashMap<(VertexId, usize), usize>,
    /// Each edge written, once for each pair of partner coedges, named by the first: where
    /// pieces touch along an edge, each piece has its own.
    edges: HashMap<(EdgeId, CoedgeId), usize>,
    /// The fan each coedge arriving at a vertex belongs to there.
    fans: HashMap<CoedgeId, usize>,
    /// Each face's surface, written before any of its edges so their pcurves can name it.
    surfaces: HashMap<FaceId, usize>,
    context_2d: Option<usize>,
    /// The faces in the order their `ADVANCED_FACE`s were written.
    faces: Vec<FaceId>,
}

fn key(p: Vec3) -> [u64; 3] {
    [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
}

impl Writer {
    pub(super) fn new() -> Writer {
        Writer {
            out: String::new(),
            next: 1,
            points: HashMap::new(),
            directions: HashMap::new(),
            vertices: HashMap::new(),
            edges: HashMap::new(),
            fans: HashMap::new(),
            surfaces: HashMap::new(),
            context_2d: None,
            faces: Vec::new(),
        }
    }

    fn add(&mut self, body: impl AsRef<str>) -> usize {
        let id = self.next;
        self.next += 1;
        let _ = writeln!(self.out, "#{id}={};", body.as_ref());
        id
    }

    fn point(&mut self, p: Point3) -> Result<usize> {
        if !p.is_finite() {
            return Err(Error::InvalidInput("STEP: a point is not finite".into()));
        }
        if let Some(&id) = self.points.get(&key(p)) {
            return Ok(id);
        }
        let id = self.add(format!("CARTESIAN_POINT('',{})", reals([p.x, p.y, p.z])));
        self.points.insert(key(p), id);
        Ok(id)
    }

    fn direction(&mut self, d: Vec3) -> usize {
        if let Some(&id) = self.directions.get(&key(d)) {
            return id;
        }
        let id = self.add(format!("DIRECTION('',{})", reals([d.x, d.y, d.z])));
        self.directions.insert(key(d), id);
        id
    }

    fn placement(&mut self, f: &Frame) -> Result<usize> {
        let o = self.point(f.origin)?;
        let z = self.direction(f.z);
        let x = self.direction(f.x);
        Ok(self.add(format!("AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})")))
    }

    fn spline_curve(&mut self, s: &Spline) -> Result<usize> {
        let mut pts = Vec::with_capacity(s.ctrl.len());
        for &p in &s.ctrl {
            pts.push(self.point(p)?);
        }
        let (k, m) = knot_mults(&s.knots);
        let (deg, pts, m, k) = (s.degree, refs(&pts), ints(&m), reals(k));
        Ok(match &s.weights {
            None => self.add(format!(
                "B_SPLINE_CURVE_WITH_KNOTS('',{deg},{pts},.UNSPECIFIED.,.F.,.F.,{m},{k},\
                 .UNSPECIFIED.)"
            )),
            Some(w) => self.add(format!(
                "(BOUNDED_CURVE() B_SPLINE_CURVE({deg},{pts},.UNSPECIFIED.,.F.,.F.) \
                 B_SPLINE_CURVE_WITH_KNOTS({m},{k},.UNSPECIFIED.) CURVE() \
                 GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE({}) \
                 REPRESENTATION_ITEM(''))",
                reals(w.iter().copied())
            )),
        })
    }

    pub(super) fn curve(&mut self, c: &Curve) -> Result<usize> {
        match c {
            Curve::Line(l) => {
                let p = self.point(l.origin)?;
                let d = self.direction(l.dir);
                let v = self.add(format!("VECTOR('',#{d},1.)"));
                Ok(self.add(format!("LINE('',#{p},#{v})")))
            }
            Curve::Circle(ci) => {
                let a = self.placement(&ci.frame)?;
                Ok(self.add(format!("CIRCLE('',#{a},{})", real(ci.radius))))
            }
            Curve::Ellipse(e) => {
                let a = self.placement(&e.frame)?;
                Ok(self.add(format!(
                    "ELLIPSE('',#{a},{},{})",
                    real(e.major),
                    real(e.minor)
                )))
            }
            Curve::BSpline(b) => self.spline_curve(&Spline {
                degree: b.degree(),
                knots: b.knots().to_vec(),
                ctrl: b.control_points().to_vec(),
                weights: None,
            }),
            Curve::Intersection(ic) => self.spline_curve(&Spline {
                degree: ic.approx.degree(),
                knots: ic.approx.knots().to_vec(),
                ctrl: ic.approx.control_points().to_vec(),
                weights: None,
            }),
            Curve::Nurbs(n) => self.spline_curve(&Spline {
                degree: n.degree(),
                knots: n.knots().to_vec(),
                ctrl: n.control_points().to_vec(),
                weights: n.weights().map(<[f64]>::to_vec),
            }),
        }
    }

    fn spline_surface(
        &mut self,
        degrees: (usize, usize),
        knots: (&[f64], &[f64]),
        counts: (usize, usize),
        ctrl: &[Point3],
        weights: Option<&[f64]>,
    ) -> Result<usize> {
        let mut rows = Vec::with_capacity(counts.0);
        for i in 0..counts.0 {
            let mut row = Vec::with_capacity(counts.1);
            for j in 0..counts.1 {
                row.push(self.point(ctrl[i * counts.1 + j])?);
            }
            rows.push(refs(&row));
        }
        let net = format!("({})", rows.join(","));
        let (ku, mu) = knot_mults(knots.0);
        let (kv, mv) = knot_mults(knots.1);
        let (du, dv) = degrees;
        let tail = format!(
            "{},{},{},{},.UNSPECIFIED.",
            ints(&mu),
            ints(&mv),
            reals(ku),
            reals(kv)
        );
        Ok(match weights {
            None => self.add(format!(
                "B_SPLINE_SURFACE_WITH_KNOTS('',{du},{dv},{net},.UNSPECIFIED.,.F.,.F.,.F.,{tail})"
            )),
            Some(w) => {
                let wrows: Vec<String> = w.chunks(counts.1).map(|r| reals(r.to_vec())).collect();
                self.add(format!(
                    "(BOUNDED_SURFACE() B_SPLINE_SURFACE({du},{dv},{net},.UNSPECIFIED.,.F.,.F.,\
                     .F.) B_SPLINE_SURFACE_WITH_KNOTS({tail}) GEOMETRIC_REPRESENTATION_ITEM() \
                     RATIONAL_B_SPLINE_SURFACE(({})) REPRESENTATION_ITEM('') SURFACE())",
                    wrows.join(",")
                ))
            }
        })
    }

    /// Writes `s`. A cone's STEP form may have the opposite natural normal (see [`step_cone`]),
    /// which the face that uses it accounts for.
    pub(super) fn surface(&mut self, s: &Surface) -> Result<usize> {
        match s {
            Surface::Plane(p) => {
                let a = self.placement(&p.frame)?;
                Ok(self.add(format!("PLANE('',#{a})")))
            }
            Surface::Cylinder(c) => {
                let a = self.placement(&c.frame)?;
                Ok(self.add(format!("CYLINDRICAL_SURFACE('',#{a},{})", real(c.radius))))
            }
            Surface::Cone(c) => {
                let (frame, angle, _) = step_cone(c)?;
                let a = self.placement(&frame)?;
                Ok(self.add(format!(
                    "CONICAL_SURFACE('',#{a},{},{})",
                    real(c.radius),
                    real(angle)
                )))
            }
            Surface::Sphere(sp) => {
                let a = self.placement(&sp.frame)?;
                Ok(self.add(format!("SPHERICAL_SURFACE('',#{a},{})", real(sp.radius))))
            }
            Surface::Torus(t) => {
                let a = self.placement(&t.frame)?;
                Ok(self.add(format!(
                    "TOROIDAL_SURFACE('',#{a},{},{})",
                    real(t.major),
                    real(t.minor)
                )))
            }
            Surface::Extrusion(e) => {
                let c = self.curve(&e.curve)?;
                let d = self.direction(e.dir);
                let v = self.add(format!("VECTOR('',#{d},1.)"));
                Ok(self.add(format!("SURFACE_OF_LINEAR_EXTRUSION('',#{c},#{v})")))
            }
            Surface::Revolution(r) => {
                let c = self.curve(&r.curve)?;
                let o = self.point(r.origin)?;
                let d = self.direction(r.axis);
                let ax = self.add(format!("AXIS1_PLACEMENT('',#{o},#{d})"));
                Ok(self.add(format!("SURFACE_OF_REVOLUTION('',#{c},#{ax})")))
            }
            Surface::Ruled(r) => {
                let (a, b) = ruled_rails(r)?;
                let n = a.ctrl.len();
                let mut ctrl = Vec::with_capacity(2 * n);
                for i in 0..n {
                    ctrl.push(a.ctrl[i]);
                    ctrl.push(b.ctrl[i]);
                }
                let weights = a
                    .weights
                    .as_ref()
                    .map(|w| w.iter().flat_map(|&x| [x, x]).collect::<Vec<f64>>());
                self.spline_surface(
                    (a.degree, 1),
                    (&a.knots, &[0.0, 0.0, 1.0, 1.0]),
                    (n, 2),
                    &ctrl,
                    weights.as_deref(),
                )
            }
            Surface::Nurbs(n) => self.spline_surface(
                n.degrees(),
                n.knots(),
                n.counts(),
                n.control_points(),
                n.weights(),
            ),
        }
    }

    fn point_2d(&mut self, p: Vec2) -> Result<usize> {
        if !(p.x.is_finite() && p.y.is_finite()) {
            return Err(Error::InvalidInput(
                "STEP: a pcurve point is not finite".into(),
            ));
        }
        Ok(self.add(format!("CARTESIAN_POINT('',{})", reals([p.x, p.y]))))
    }

    fn curve_2d(&mut self, c: &Curve2) -> Result<usize> {
        match c {
            Curve2::Line { origin, dir } => {
                let p = self.point_2d(*origin)?;
                let len = dir.norm();
                let u = *dir * (1.0 / len);
                let d = self.add(format!("DIRECTION('',{})", reals([u.x, u.y])));
                let v = self.add(format!("VECTOR('',#{d},{})", real(len)));
                Ok(self.add(format!("LINE('',#{p},#{v})")))
            }
            Curve2::Spline {
                degree,
                knots,
                ctrl,
                weights,
            } => {
                let mut pts = Vec::with_capacity(ctrl.len());
                for &p in ctrl {
                    pts.push(self.point_2d(p)?);
                }
                let (k, m) = knot_mults(knots);
                let (pts, m, k) = (refs(&pts), ints(&m), reals(k));
                Ok(match weights {
                    None => self.add(format!(
                        "B_SPLINE_CURVE_WITH_KNOTS('',{degree},{pts},.UNSPECIFIED.,.F.,.F.,{m},\
                         {k},.UNSPECIFIED.)"
                    )),
                    Some(w) => self.add(format!(
                        "(BOUNDED_CURVE() B_SPLINE_CURVE({degree},{pts},.UNSPECIFIED.,.F.,.F.) \
                         B_SPLINE_CURVE_WITH_KNOTS({m},{k},.UNSPECIFIED.) CURVE() \
                         GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE({}) \
                         REPRESENTATION_ITEM(''))",
                        reals(w.iter().copied())
                    )),
                })
            }
        }
    }

    /// Coedge `c`'s pcurve on its face's surface, when [`pcurve`] finds one.
    fn pcurve(&mut self, b: &Brep, c: CoedgeId) -> Result<Option<usize>> {
        let Some(c2) = pcurve(b, c) else {
            return Ok(None);
        };
        if let Curve2::Line { dir, .. } = &c2
            && (dir.norm() == 0.0 || dir.norm().is_nan())
        {
            return Ok(None);
        }
        let surface = self.face_surface(b, b.coedge_face(c))?;
        let curve = self.curve_2d(&c2)?;
        let ctx = match self.context_2d {
            Some(id) => id,
            None => {
                let id = self.add(
                    "(GEOMETRIC_REPRESENTATION_CONTEXT(2) PARAMETRIC_REPRESENTATION_CONTEXT() \
                     REPRESENTATION_CONTEXT('2D SPACE',''))",
                );
                self.context_2d = Some(id);
                id
            }
        };
        let rep = self.add(format!("DEFINITIONAL_REPRESENTATION('',(#{curve}),#{ctx})"));
        Ok(Some(self.add(format!("PCURVE('',#{surface},#{rep})"))))
    }

    fn face_surface(&mut self, b: &Brep, f: FaceId) -> Result<usize> {
        if let Some(&id) = self.surfaces.get(&f) {
            return Ok(id);
        }
        let id = self.surface(&b.face(f).surface)?;
        self.surfaces.insert(f, id);
        Ok(id)
    }

    /// Vertex `v` for the fan of faces that coedge `arriving` reaches it in.
    fn vertex(&mut self, b: &Brep, v: VertexId, arriving: CoedgeId) -> Result<usize> {
        let key = (v, self.fans.get(&arriving).copied().unwrap_or(0));
        if let Some(&id) = self.vertices.get(&key) {
            return Ok(id);
        }
        let p = self.point(b.vertex(v).point)?;
        let id = self.add(format!("VERTEX_POINT('',#{p})"));
        self.vertices.insert(key, id);
        Ok(id)
    }

    /// The edge coedge `co` uses, as the one it and its partner share.
    fn edge(&mut self, b: &Brep, co: CoedgeId) -> Result<usize> {
        let e = b.coedge(co).edge;
        let mate = b.partner(co).unwrap_or(co);
        let first = co.min(mate);
        if let Some(&id) = self.edges.get(&(e, first)) {
            return Ok(id);
        }
        let edge = b.edge(e);
        let curve = edge
            .curve
            .as_ref()
            .expect("degenerate edges are not written");
        // `first` arrives at its end, and the coedge before it in its loop at its start.
        let cs = &b.lp(b.coedge(first).lp).coedges;
        let k = cs.iter().position(|&c| c == first).expect("in its loop");
        let before = cs[(k + cs.len() - 1) % cs.len()];
        let (at_start, at_end) = if b.coedge(first).reversed {
            (first, before)
        } else {
            (before, first)
        };
        let s = self.vertex(b, edge.start, at_start)?;
        let t = self.vertex(b, edge.end, at_end)?;
        let c = self.curve(curve)?;
        // The curve with its pcurve on each face that uses it, which other kernels integrate
        // and check along instead of projecting the curve themselves. A seam needs both.
        let uses: Vec<CoedgeId> = if mate == co {
            vec![co]
        } else {
            vec![first, co.max(mate)]
        };
        let mut pcurves = Vec::new();
        for &co in &uses {
            if let Some(pc) = self.pcurve(b, co)? {
                pcurves.push(pc);
            }
        }
        let seam = uses.len() == 2 && b.coedge_face(uses[0]) == b.coedge_face(uses[1]);
        let geometry = match (pcurves.len(), seam) {
            (0, _) | (1, true) => c,
            (2, true) => self.add(format!("SEAM_CURVE('',#{c},{},.CURVE_3D.)", refs(&pcurves))),
            _ => self.add(format!(
                "SURFACE_CURVE('',#{c},{},.CURVE_3D.)",
                refs(&pcurves)
            )),
        };
        let id = self.add(format!("EDGE_CURVE('',#{s},#{t},#{geometry},.T.)"));
        self.edges.insert((e, first), id);
        Ok(id)
    }

    /// A face; `void` writes it as part of a void's shell, which STEP stores turned inside out.
    fn face(&mut self, b: &Brep, f: FaceId, void: bool) -> Result<usize> {
        let face = b.face(f);
        let mut bounds = Vec::new();
        for (i, &lp) in face.loops.iter().enumerate() {
            // A point where something touches the face is in neither piece's manifold boundary.
            if b.is_vertex_loop(lp) {
                continue;
            }
            // A slit, which the loop runs along and back with the face on both sides, is no edge
            // of the face's piece. The loop between its two uses is a bound of its own.
            let mut stack: Vec<Vec<usize>> = vec![Vec::new()];
            let mut parts: Vec<(bool, Vec<usize>)> = Vec::new();
            let mut opened = std::collections::HashSet::new();
            for &c in &b.lp(lp).coedges {
                let co = b.coedge(c);
                if b.edge(co.edge).is_degenerate() {
                    continue;
                }
                if let Some(p) = slit_mate(b, c) {
                    if !opened.contains(&p) {
                        opened.insert(c);
                        stack.push(Vec::new());
                    } else if stack.len() > 1 {
                        let inner = stack.pop().expect("more than one");
                        if !inner.is_empty() {
                            parts.push((false, inner));
                        }
                    }
                    continue;
                }
                let e = self.edge(b, c)?;
                let o = if co.reversed { ".F." } else { ".T." };
                let oe = self.add(format!("ORIENTED_EDGE('',*,*,#{e},{o})"));
                stack.last_mut().expect("never empty").push(oe);
            }
            for rest in stack.into_iter().rev() {
                if !rest.is_empty() {
                    parts.push((i == 0, rest));
                }
            }
            for (outer, oes) in parts {
                let el = self.add(format!("EDGE_LOOP('',{})", refs(&oes)));
                let kind = if outer {
                    "FACE_OUTER_BOUND"
                } else {
                    "FACE_BOUND"
                };
                let o = if void { ".F." } else { ".T." };
                bounds.push(self.add(format!("{kind}('',#{el},{o})")));
            }
        }
        let s = self.face_surface(b, f)?;
        let names: Vec<String> = face.names.iter().map(ToString::to_string).collect();
        let flips = match &face.surface {
            Surface::Cone(c) => step_cone(c)?.2,
            _ => false,
        };
        let same = if (face.reversed == void) != flips {
            ".T."
        } else {
            ".F."
        };
        Ok(self.add(format!(
            "ADVANCED_FACE({},{},#{s},{same})",
            encode_string(&names.join(";")),
            refs(&bounds)
        )))
    }

    fn shell(&mut self, b: &Brep, sh: ShellId, void: bool) -> Result<usize> {
        let mut faces = Vec::new();
        for &f in &b.shell(sh).faces {
            faces.push(self.face(b, f, void)?);
            self.faces.push(f);
        }
        Ok(self.add(format!("CLOSED_SHELL('',{})", refs(&faces))))
    }

    /// Solid `s` as the manifold boundary of its interior, as STEP's solids are: each outer sheet a
    /// solid of its own with the voids inside it, and where pieces touch, each with its own copy of
    /// the vertices and edges there.
    fn solid(&mut self, b: &Brep, s: SolidId) -> Result<Vec<usize>> {
        for f in b.solid_faces(s) {
            self.face_surface(b, f)?;
        }
        let shells = b.solid(s).shells.clone();
        let n_outer = b.solid(s).outer;
        let hosts = if n_outer > 1 {
            crate::check::void_hosts(b, s)
        } else {
            vec![0; shells.len() - n_outer]
        };
        let mut out = Vec::new();
        for (k, &sh) in shells[..n_outer].iter().enumerate() {
            let outer = self.shell(b, sh, false)?;
            let mut voids = Vec::new();
            for (&v, _) in shells[n_outer..]
                .iter()
                .zip(&hosts)
                .filter(|(_, h)| **h == k)
            {
                let inner = self.shell(b, v, true)?;
                voids.push(self.add(format!("ORIENTED_CLOSED_SHELL('',*,#{inner},.F.)")));
            }
            out.push(if voids.is_empty() {
                self.add(format!("MANIFOLD_SOLID_BREP('',#{outer})"))
            } else {
                self.add(format!("BREP_WITH_VOIDS('',#{outer},{})", refs(&voids)))
            });
        }
        Ok(out)
    }
}

/// The partner of coedge `c` where the two run along the same curve of one face and back, the face
/// on both sides of them: a slit, not a seam, whose two coedges lie a period apart.
fn slit_mate(b: &Brep, c: CoedgeId) -> Option<CoedgeId> {
    let p = b.partner(c).filter(|&p| p != c)?;
    if b.coedge_face(p) != b.coedge_face(c) {
        return None;
    }
    let (t0, t1) = b.edge(b.coedge(c).edge).range;
    let t = [0.5 * (t0 + t1)];
    let gap = b.coedge_uvs(c, &t)[0].dist(b.coedge_uvs(p, &t)[0]);
    (gap <= 1e-9 * (1.0 + t0.abs().max(t1.abs()))).then_some(p)
}

/// A curve in a surface's STEP parameters, as a function of its edge's own parameter.
#[derive(Clone, Debug)]
pub(super) enum Curve2 {
    /// `origin + t·dir`.
    Line { origin: Vec2, dir: Vec2 },
    Spline {
        degree: usize,
        knots: Vec<f64>,
        ctrl: Vec<Vec2>,
        weights: Option<Vec<f64>>,
    },
}

/// How a surface's STEP parameters follow from the kernel's: `(s·u, k·v)`. A cone's STEP form
/// measures `v` along its axis rather than its generator, and turns `u` the other way when it is
/// written about the reversed axis; every other surface is written in the kernel's parameters but a
/// ruled surface between conics, whose rational rails run at another rate than their angle.
fn step_scale(s: &Surface) -> Option<Vec2> {
    match s {
        Surface::Cone(c) => {
            let (frame, ..) = step_cone(c).ok()?;
            let sign = if frame.z.dot(c.frame.z) > 0.0 {
                1.0
            } else {
                -1.0
            };
            Some(Vec2::new(sign, sign * c.half_angle.cos()))
        }
        Surface::Ruled(r) => {
            let conic = |c: &Curve| matches!(c, Curve::Circle(_) | Curve::Ellipse(_));
            (!conic(&r.a) && !conic(&r.b)).then_some(Vec2::new(1.0, 1.0))
        }
        _ => Some(Vec2::new(1.0, 1.0)),
    }
}

/// The most samples a fitted pcurve takes before the writer leaves the coedge without one.
const MAX_PCURVE_SAMPLES: usize = 4096;

/// Coedge `c`'s pcurve in its face's STEP surface parameters. An affine pcurve is a line and a
/// spline one its spline, and on a plane a line or a spline is mapped exactly onto it; any other is
/// a cubic through the coedge's own parameter-plane points, at enough of them that the surface over it stays within half the
/// tolerance of the edge's curve. `None` where no such pcurve is found, or the surface's STEP form
/// has other parameters than the kernel's that no scale maps.
pub(super) fn pcurve(b: &Brep, c: CoedgeId) -> Option<Curve2> {
    let co = b.coedge(c);
    let edge = b.edge(co.edge);
    let curve = edge.curve.as_ref()?;
    let s = &b.face(b.coedge_face(c)).surface;
    let k = step_scale(s)?;
    let scaled = |p: Vec2| Vec2::new(p.x * k.x, p.y * k.y);
    if let PCurve::Affine { origin, dir } = &co.pcurve {
        return Some(Curve2::Line {
            origin: scaled(*origin),
            dir: scaled(*dir),
        });
    }
    if let PCurve::Spline { uv } = &co.pcurve {
        return Some(Curve2::Spline {
            degree: uv.degree(),
            knots: uv.knots().to_vec(),
            ctrl: uv
                .control_points()
                .iter()
                .map(|p| scaled(Vec2::new(p.x, p.y)))
                .collect(),
            weights: None,
        });
    }
    if let Surface::Plane(pl) = s {
        let local = |p: Point3| {
            let d = p - pl.frame.origin;
            Vec2::new(d.dot(pl.frame.x), d.dot(pl.frame.y))
        };
        let spline =
            |degree, knots: &[f64], ctrl: &[Point3], weights: Option<&[f64]>| Curve2::Spline {
                degree,
                knots: knots.to_vec(),
                ctrl: ctrl.iter().map(|&p| local(p)).collect(),
                weights: weights.map(<[f64]>::to_vec),
            };
        match curve {
            Curve::Line(l) => {
                return Some(Curve2::Line {
                    origin: local(l.origin),
                    dir: Vec2::new(l.dir.dot(pl.frame.x), l.dir.dot(pl.frame.y)),
                });
            }
            Curve::BSpline(sp) => {
                return Some(spline(sp.degree(), sp.knots(), sp.control_points(), None));
            }
            Curve::Intersection(ic) => {
                let sp = &ic.approx;
                return Some(spline(sp.degree(), sp.knots(), sp.control_points(), None));
            }
            Curve::Nurbs(n) => {
                return Some(spline(
                    n.degree(),
                    n.knots(),
                    n.control_points(),
                    n.weights(),
                ));
            }
            _ => {}
        }
    }
    let (t0, t1) = edge.range;
    let mut ts: Vec<f64> = (0..=8).map(|i| t0 + (t1 - t0) * i as f64 / 8.0).collect();
    while ts.len() <= MAX_PCURVE_SAMPLES {
        // Sampled the way the coedge runs, so periodic parameters continue from its own start.
        let mut run = ts.clone();
        if co.reversed {
            run.reverse();
        }
        let mut uvs = b.coedge_uvs(c, &run);
        if co.reversed {
            uvs.reverse();
        }
        let pts: Vec<Vec3> = uvs.iter().map(|uv| Vec3::new(uv.x, uv.y, 0.0)).collect();
        let fit = BSpline::interpolate_at(&pts, &ts).ok()?;
        let off = |t: f64| {
            let uv = fit.eval(t);
            s.eval(uv.x, uv.y).dist(curve.eval(t))
        };
        // Points go in only where the fit strays, so the pcurve has about as many poles as the
        // curve needs there. Each span is read at seven points against 0.45 of the tolerance, so a
        // stray peaking between them still stays within half: read at three against half, one
        // reached 5.2e-8 on a lofted helical gear.
        let mut next = Vec::with_capacity(2 * ts.len());
        for w in ts.windows(2) {
            next.push(w[0]);
            if (1..8).any(|i| off(w[0] + (w[1] - w[0]) * i as f64 / 8.0) > 0.45 * tol::LINEAR) {
                next.push(0.5 * (w[0] + w[1]));
            }
        }
        next.push(t1);
        if next.len() == ts.len() {
            return Some(Curve2::Spline {
                degree: fit.degree(),
                knots: fit.knots().to_vec(),
                ctrl: fit
                    .control_points()
                    .iter()
                    .map(|p| scaled(Vec2::new(p.x, p.y)))
                    .collect(),
                weights: None,
            });
        }
        ts = next;
    }
    None
}
/// The two rails of a ruled surface as B-splines with one knot vector and one set of weights, on
/// `u ∈ [0, 1]`, so the B-spline surface between them is the ruled surface itself. That holds for
/// two polynomial curves, and for two conics over equal angles; other pairs have no such form.
fn ruled_rails(r: &Ruled) -> Result<(Spline, Spline)> {
    let conic = |c: &Curve| matches!(c, Curve::Circle(_) | Curve::Ellipse(_));
    let span = |x: (f64, f64)| x.1 - x.0;
    if conic(&r.a) && conic(&r.b) && (span(r.a_range) - span(r.b_range)).abs() <= 1e-12 {
        return Ok((conic_spline(&r.a, r.a_range), conic_spline(&r.b, r.b_range)));
    }
    let (Some(da), Some(db)) = (poly_degree(&r.a), poly_degree(&r.b)) else {
        return Err(Error::Unsupported(format!(
            "STEP: a ruled surface between a {} and a {} has no exact B-spline form",
            r.a.kind(),
            r.b.kind()
        )));
    };
    let q = da.max(db);
    let to_s = |c: &Curve, rg: (f64, f64)| -> Vec<f64> {
        c.breakpoints(rg.0.min(rg.1), rg.0.max(rg.1))
            .into_iter()
            .map(|t| (t - rg.0) / (rg.1 - rg.0))
            .collect()
    };
    let mut cuts = vec![0.0, 1.0];
    cuts.extend(to_s(&r.a, r.a_range));
    cuts.extend(to_s(&r.b, r.b_range));
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|x, y| (*x - *y).abs() <= 1e-14);
    Ok((
        bezier_chain(&r.a, r.a_range, &cuts, q)?,
        bezier_chain(&r.b, r.b_range, &cuts, q)?,
    ))
}

fn poly_degree(c: &Curve) -> Option<usize> {
    match c {
        Curve::Line(_) => Some(1),
        Curve::BSpline(b) => Some(b.degree()),
        Curve::Intersection(i) => Some(i.approx.degree()),
        Curve::Nurbs(n) if n.weights().is_none() => Some(n.degree()),
        _ => None,
    }
}

fn bernstein(n: usize, i: usize, s: f64) -> f64 {
    let mut c = 1.0;
    for k in 0..i {
        c = c * (n - k) as f64 / (k + 1) as f64;
    }
    c * s.powi(i as i32) * (1.0 - s).powi((n - i) as i32)
}

/// The curve over `rg`, mapped to [0, 1] and cut at `cuts`, as one Bézier of degree `q` per piece:
/// each piece is a polynomial of degree at most `q`, so interpolating it at `q + 1` points is
/// exact.
fn bezier_chain(c: &Curve, rg: (f64, f64), cuts: &[f64], q: usize) -> Result<Spline> {
    let mut knots = vec![0.0; q + 1];
    let mut ctrl: Vec<Point3> = Vec::new();
    for (k, w) in cuts.windows(2).enumerate() {
        let n = q + 1;
        let mut a = vec![0.0; n * n];
        let mut rhs = Vec::with_capacity(3 * n);
        for i in 0..n {
            let s = i as f64 / q as f64;
            for j in 0..n {
                a[i * n + j] = bernstein(q, j, s);
            }
            let t = rg.0 + (w[0] + (w[1] - w[0]) * s) * (rg.1 - rg.0);
            rhs.extend(c.eval(t).to_array());
        }
        let x = solve_dense(a, rhs, n, 3)?;
        let pts: Vec<Point3> = x.chunks(3).map(|p| Vec3::new(p[0], p[1], p[2])).collect();
        ctrl.extend_from_slice(if k == 0 { &pts } else { &pts[1..] });
        if k + 2 < cuts.len() {
            knots.extend(std::iter::repeat_n(w[1], q));
        }
    }
    knots.extend(std::iter::repeat_n(1.0, q + 1));
    Ok(Spline {
        degree: q,
        knots,
        ctrl,
        weights: None,
    })
}

/// A circle or ellipse over `rg` as rational quadratic arcs of at most a quarter turn each, on
/// [0, 1]. Two conics over equal angles share the knots and the weights.
fn conic_spline(c: &Curve, rg: (f64, f64)) -> Spline {
    let (frame, a, b) = match c {
        Curve::Circle(ci) => (ci.frame, ci.radius, ci.radius),
        Curve::Ellipse(e) => (e.frame, e.major, e.minor),
        _ => unreachable!("conic_spline takes conics"),
    };
    let at = |t: f64, scale: f64| {
        frame.origin + frame.x * (a * t.cos() * scale) + frame.y * (b * t.sin() * scale)
    };
    let span = rg.1 - rg.0;
    let n = (span.abs() / FRAC_PI_2 - 1e-9).ceil().max(1.0) as usize;
    let d = span / n as f64;
    let wm = (d * 0.5).cos();
    let mut knots = vec![0.0; 3];
    let mut ctrl = vec![at(rg.0, 1.0)];
    let mut weights = vec![1.0];
    for k in 0..n {
        let t0 = rg.0 + d * k as f64;
        ctrl.push(at(t0 + 0.5 * d, 1.0 / wm));
        ctrl.push(at(t0 + d, 1.0));
        weights.extend([wm, 1.0]);
        if k + 1 < n {
            knots.extend([(k + 1) as f64 / n as f64; 2]);
        }
    }
    knots.extend([1.0; 3]);
    Spline {
        degree: 2,
        knots,
        ctrl,
        weights: Some(weights),
    }
}

/// What every product of one file shares.
struct Shared {
    app: usize,
    pc: usize,
}

impl Writer {
    fn shared(&mut self) -> Shared {
        let app =
            self.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')");
        self.add(format!(
            "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,\
             #{app})"
        ));
        let pc = self.add(format!("PRODUCT_CONTEXT('',#{app},'mechanical')"));
        Shared { app, pc }
    }

    /// A product named `n` (already encoded): its definition and that definition's shape.
    fn product(&mut self, sh: &Shared, n: &str) -> (usize, usize) {
        let (app, pc) = (sh.app, sh.pc);
        let prod = self.add(format!("PRODUCT({n},{n},'',(#{pc}))"));
        self.add(format!(
            "PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod}))"
        ));
        let pdf = self.add(format!("PRODUCT_DEFINITION_FORMATION('','',#{prod})"));
        let pdc = self.add(format!(
            "PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')"
        ));
        let pd = self.add(format!("PRODUCT_DEFINITION('design','',#{pdf},#{pdc})"));
        let pds = self.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
        (pd, pds)
    }

    /// A context in millimetres and radians whose geometry meets to `widest`.
    fn context(&mut self, widest: f64) -> usize {
        let mm = self.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))");
        let rad = self.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))");
        let sr = self.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())");
        let why = if widest > crate::tol::LINEAR {
            "the widest local tolerance of the imported geometry"
        } else {
            "the kernel linear tolerance"
        };
        let unc = self.add(format!(
            "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({}),#{mm},'distance_accuracy_value',\
             '{why}')",
            real(widest)
        ));
        self.add(format!(
            "(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) \
             GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm},#{rad},#{sr})) REPRESENTATION_CONTEXT('','3D'))"
        ))
    }

    /// Every solid of `b` as the shape of the product whose definition shape is `pds`, drawn in
    /// `colour` where one is given. Returns the representation and the faces in written order.
    fn body(
        &mut self,
        b: &Brep,
        n: &str,
        pds: usize,
        colour: Option<[f64; 3]>,
    ) -> Result<(usize, Vec<FaceId>)> {
        // Ids are one arena's: another part's vertices, edges and faces are other things.
        self.vertices.clear();
        self.edges.clear();
        self.fans.clear();
        self.surfaces.clear();
        self.faces.clear();
        // Other systems read one uncertainty for the whole part: where imported geometry meets
        // only to local tolerances, the widest of them, so that they accept its edges as the
        // kernel does.
        let (vs, es) = b.tolerant();
        let widest = vs
            .iter()
            .map(|x| x.1)
            .chain(es.iter().map(|x| x.1))
            .fold(crate::tol::LINEAR, f64::max);
        let ctx = self.context(widest);
        for fans in crate::check::fans_at_vertices(b).into_values() {
            for (k, fan) in fans.into_iter().enumerate() {
                self.fans.extend(fan.into_iter().map(|c| (c, k)));
            }
        }
        let origin = self.placement(&Frame::WORLD)?;
        let mut items = vec![origin];
        for s in b.solid_ids() {
            items.extend(self.solid(b, s)?);
        }
        let rep = self.add(format!(
            "ADVANCED_BREP_SHAPE_REPRESENTATION({n},{},#{ctx})",
            refs(&items)
        ));
        self.add(format!("SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})"));
        if let Some(c) = colour {
            let rgb = self.add(format!(
                "COLOUR_RGB('',{},{},{})",
                real(c[0]),
                real(c[1]),
                real(c[2])
            ));
            let fill = self.add(format!("FILL_AREA_STYLE_COLOUR('',#{rgb})"));
            let area = self.add(format!("FILL_AREA_STYLE('',(#{fill}))"));
            let fills = self.add(format!("SURFACE_STYLE_FILL_AREA(#{area})"));
            let side = self.add(format!("SURFACE_SIDE_STYLE('',(#{fills}))"));
            let usage = self.add(format!("SURFACE_STYLE_USAGE(.BOTH.,#{side})"));
            let style = self.add(format!("PRESENTATION_STYLE_ASSIGNMENT((#{usage}))"));
            let styled: Vec<usize> = items[1..]
                .iter()
                .map(|solid| self.add(format!("STYLED_ITEM('',(#{style}),#{solid})")))
                .collect();
            self.add(format!(
                "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',{},#{ctx})",
                refs(&styled)
            ));
        }
        Ok((rep, std::mem::take(&mut self.faces)))
    }

    fn finish(self, n: &str) -> String {
        let mut text = String::from("ISO-10303-21;\nHEADER;\n");
        text += "FILE_DESCRIPTION(('springrep solid'),'2;1');\n";
        let _ = writeln!(
            text,
            "FILE_NAME({n},'',(''),(''),'',{},'');",
            encode_string(WRITER)
        );
        text += "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n";
        text += &self.out;
        text += "ENDSEC;\nEND-ISO-10303-21;\n";
        text
    }
}

/// The whole arena as one part's advanced B-rep, in millimetres and radians. `name` names the
/// product.
pub fn write(b: &Brep, name: &str) -> Result<String> {
    write_faces(b, name).map(|(text, _)| text)
}

/// [`write()`], and the faces in the order the file holds them: the n-th `ADVANCED_FACE` of the
/// file is the n-th face here. A reader that keeps the file's order can tell its faces by it.
pub fn write_faces(b: &Brep, name: &str) -> Result<(String, Vec<FaceId>)> {
    let mut w = Writer::new();
    let n = encode_string(name);
    let shared = w.shared();
    let (_, pds) = w.product(&shared, &n);
    let (_, faces) = w.body(b, &n, pds, None)?;
    Ok((w.finish(&n), faces))
}

/// One part of an assembly file: every solid of `brep`, written once however often it is placed.
#[derive(Clone, Copy, Debug)]
pub struct Part<'a> {
    pub name: &'a str,
    pub brep: &'a Brep,
    /// The colour its surfaces are drawn in, red, green and blue in 0 to 1.
    pub colour: Option<[f64; 3]>,
}

/// One use of a part in an assembly file: `parts[part]` moved by `at`.
#[derive(Clone, Copy, Debug)]
pub struct Instance<'a> {
    pub name: &'a str,
    pub part: usize,
    pub at: Transform,
}

/// An assembly named `name` as one file, in millimetres and radians: each part a product with its
/// own advanced B-rep, and each instance a use of its part's product, placed in the assembly.
/// [`read`](super::read) gives every instance's solids where it places them, with the part's and
/// the instance's names.
pub fn write_assembly(name: &str, parts: &[Part], instances: &[Instance]) -> Result<String> {
    if let Some(i) = instances.iter().find(|i| i.part >= parts.len()) {
        return Err(Error::InvalidInput(format!(
            "STEP: instance {:?} uses part {} of {}",
            i.name,
            i.part,
            parts.len()
        )));
    }
    let mut w = Writer::new();
    let n = encode_string(name);
    let shared = w.shared();
    let (root, root_shape) = w.product(&shared, &n);
    let mut made = Vec::with_capacity(parts.len());
    for part in parts {
        let pn = encode_string(part.name);
        let (pd, pds) = w.product(&shared, &pn);
        let (rep, _) = w.body(part.brep, &pn, pds, part.colour)?;
        made.push((pd, rep));
    }
    let ctx = w.context(crate::tol::LINEAR);
    let origin = w.placement(&Frame::WORLD)?;
    let mut places = Vec::with_capacity(instances.len());
    for i in instances {
        let r = &i.at.r;
        let column = |j: usize| Vec3::new(r[0][j], r[1][j], r[2][j]);
        places.push(w.placement(&Frame {
            origin: i.at.t,
            x: column(0),
            y: column(1),
            z: column(2),
        })?);
    }
    let mut items = vec![origin];
    items.extend(&places);
    let rep = w.add(format!("SHAPE_REPRESENTATION({n},{},#{ctx})", refs(&items)));
    w.add(format!(
        "SHAPE_DEFINITION_REPRESENTATION(#{root_shape},#{rep})"
    ));
    for (k, (i, place)) in instances.iter().zip(places).enumerate() {
        let (pd, part_rep) = made[i.part];
        let used = w.add(format!(
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE('{}',{},'',#{root},#{pd},$)",
            k + 1,
            encode_string(i.name)
        ));
        let shape = w.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{used})"));
        let moved = w.add(format!(
            "ITEM_DEFINED_TRANSFORMATION('','',#{origin},#{place})"
        ));
        let relation = w.add(format!(
            "(REPRESENTATION_RELATIONSHIP('','',#{part_rep},#{rep}) \
             REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{moved}) \
             SHAPE_REPRESENTATION_RELATIONSHIP())"
        ));
        w.add(format!(
            "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relation},#{shape})"
        ));
    }
    Ok(w.finish(&n))
}

/// The STEP form of a cone, whose semi-angle must lie in `[0, π/2)`: its placement, its semi-angle,
/// and whether that form's natural normal is opposite to `c`'s. A cone narrowing along its axis is
/// written about the reversed axis, which keeps its normal; one whose half angle is past a right
/// angle (opening against its axis) is the same point set as its supplement about the reversed
/// axis, whose normal is the opposite.
pub(super) fn step_cone(c: &Cone) -> Result<(Frame, f64, bool)> {
    let a = c.half_angle;
    let reversed = Frame {
        origin: c.frame.origin,
        x: c.frame.x,
        y: -c.frame.y,
        z: -c.frame.z,
    };
    let quarter = FRAC_PI_2;
    let out = if (0.0..quarter).contains(&a) {
        (c.frame, a, false)
    } else if (-quarter..0.0).contains(&a) {
        (reversed, -a, false)
    } else if a > quarter && a <= PI {
        (reversed, PI - a, true)
    } else if (-PI..-quarter).contains(&a) {
        (c.frame, a + PI, true)
    } else {
        return Err(Error::InvalidInput(format!(
            "STEP: a cone's half angle {a} has no STEP form"
        )));
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reals_are_step_reals() {
        assert_eq!(real(25.0), "25.0");
        assert_eq!(real(1e-7), "1.E-7");
        assert_eq!(real(-1.5e20), "-1.5E20");
        assert_eq!(real(0.1), "0.1");
        for x in [1e-7, -3.25, 1.0 / 3.0, 6.02e23] {
            let s = real(x);
            assert_eq!(s.parse::<f64>().unwrap(), x);
        }
    }
}
