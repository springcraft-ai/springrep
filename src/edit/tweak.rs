//! Planar faces of a solid moved onto new planes, its topology kept: every vertex goes where the
//! planes of its faces meet and every edge between two planes becomes the line between its new
//! vertices. Drafts and shells of solids bounded by planes are made this way.
//!
//! Only a vertex whose faces are all planes, three or more of them independent, can move, and only
//! straight edges move with it. A move that would need anything else, or that turns a face inside
//! out, is an error, never a solid with faces off their surfaces.

use std::collections::{BTreeSet, HashMap};
use std::f64::consts::TAU;

use crate::{
    Error, Result, check,
    error::invalid,
    geom::{Cone, Curve, Cylinder, Line, Plane, Sphere, Surface, Torus},
    math::{Frame, Point3, Transform, Vec2, Vec3},
    numeric::solve_dense,
    tol,
    topo::{Brep, CoedgeSpec, EdgeId, FaceId, PCurve, VertexId},
};

/// The faces round each vertex.
fn vertex_faces(b: &Brep) -> HashMap<VertexId, BTreeSet<FaceId>> {
    let mut out: HashMap<VertexId, BTreeSet<FaceId>> = HashMap::new();
    for f in b.face_ids() {
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let e = b.edge(b.coedge(c).edge);
                out.entry(e.start).or_default().insert(f);
                out.entry(e.end).or_default().insert(f);
            }
        }
    }
    out
}

/// The point on every plane `(n, d)`, `n·p = d`, by least squares, when they pin one point.
pub(super) fn meet(planes: &[(Vec3, f64)]) -> Option<Point3> {
    let mut a = vec![0.0; 9];
    let mut r = vec![0.0; 3];
    for &(n, d) in planes {
        let n = n.to_array();
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] += n[i] * n[j];
            }
            r[i] += n[i] * d;
        }
    }
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    // Unit normals: a determinant this small leaves the point free along a line, to rounding.
    if det.abs() < 1e-12 {
        return None;
    }
    let x = solve_dense(a, r, 3, 1).ok()?;
    let p = Vec3::new(x[0], x[1], x[2]);
    planes
        .iter()
        .all(|&(n, d)| (n.dot(p) - d).abs() <= tol::LINEAR)
        .then_some(p)
}

fn plane_of(s: &Surface) -> Option<Frame> {
    match s {
        Surface::Plane(p) => Some(p.frame),
        _ => None,
    }
}

pub(super) fn affine_on(frame: &Frame, line: &Line) -> PCurve {
    let l = frame.to_local(line.origin);
    PCurve::Affine {
        origin: Vec2::new(l.x, l.y),
        dir: Vec2::new(line.dir.dot(frame.x), line.dir.dot(frame.y)),
    }
}

fn unsupported<T>(why: &str) -> Result<T> {
    Err(Error::Unsupported(why.to_string()))
}

/// Refuses to move a face that runs up to where the solid touches itself: moving it would open
/// the contact or push one piece into the other.
fn clear_of_contacts<'a>(b: &Brep, faces: impl IntoIterator<Item = &'a FaceId>) -> Result<()> {
    if crate::check::faces_at_contacts(b, faces.into_iter().copied()).is_empty() {
        return Ok(());
    }
    unsupported("moving a face up to where the solid touches itself")
}

/// `b` with each of `faces` tilted by `angle` about the line where it meets the `neutral` plane, its
/// outward normal turned towards the plane's normal: positive angles narrow the part along it.
pub fn draft(b: &Brep, faces: &[FaceId], neutral: &Frame, angle: f64) -> Result<Brep> {
    let k = neutral.z;
    let mut moved = HashMap::new();
    let mut cones = HashMap::new();
    for &g in faces {
        let face = b.face(g);
        // A cylinder along the pull becomes the cone through its circle on the neutral plane:
        // narrowing along the pull where the material lies inside it, widening a bore.
        if let Surface::Cylinder(c) = &face.surface
            && c.frame.z.cross(k).norm() <= tol::ANGULAR
        {
            let level = (neutral.origin - c.frame.origin).dot(k) / c.frame.z.dot(k);
            let frame = Frame::new(c.frame.origin + c.frame.z * level, k, c.frame.x)?;
            let half_angle = if face.reversed { angle } else { -angle };
            cones.insert(
                g,
                Surface::Cone(Cone {
                    frame,
                    radius: c.radius,
                    half_angle,
                }),
            );
            continue;
        }
        let Some(fr) = plane_of(&face.surface) else {
            return unsupported(&format!("drafting a {} face", face.surface.kind()));
        };
        let n = if face.reversed { -fr.z } else { fr.z };
        let (Some(level), Some(up)) = ((n - k * n.dot(k)).unit(), (k - n * n.dot(k)).unit()) else {
            return Err(Error::InvalidInput(
                "a face parallel to the neutral plane has no draft".into(),
            ));
        };
        let hinge = fr.origin + up * ((neutral.origin - fr.origin).dot(k) / up.dot(k));
        let tilted = level * angle.cos() + k * angle.sin();
        let z = if face.reversed { -tilted } else { tilted };
        moved.insert(g, Frame::new(hinge, z, fr.x)?);
    }
    // Where a face closes on the way, walls leaning until they meet, the draft is followed there.
    let mut paths: HashMap<FaceId, through::Path> = HashMap::new();
    for (&g, &to) in &moved {
        let from = plane_of(&b.face(g).surface).expect("drafted planes are planes");
        paths.insert(
            g,
            Box::new(move |s| {
                Ok(Surface::Plane(Plane {
                    frame: through::pencil(&from, &to, s)?,
                }))
            }),
        );
    }
    for (&g, cone) in &cones {
        let Surface::Cone(cone) = cone.clone() else {
            unreachable!("a drafted cylinder is a cone")
        };
        paths.insert(g, Box::new(move |s| Ok(through::opening(&cone, s))));
    }
    let direct = if cones.is_empty() {
        planes_or_faces(b, &moved)
    } else {
        let mut surfaces = cones.clone();
        for (&g, &frame) in &moved {
            surfaces.insert(g, Surface::Plane(Plane { frame }));
        }
        move_faces(b, &surfaces)
    };
    direct.or_else(|e| {
        through::through(b, &paths)
            .map(|(out, _)| out)
            .map_err(|_| e)
    })
}

/// [`move_planes`], or [`move_faces`] where a curved face meets the moved planes.
pub(crate) fn planes_or_faces(b: &Brep, moved: &HashMap<FaceId, Frame>) -> Result<Brep> {
    match move_planes(b, moved) {
        Err(Error::Unsupported(_)) => {
            let surfaces = moved
                .iter()
                .map(|(&f, &frame)| (f, Surface::Plane(Plane { frame })))
                .collect();
            move_faces(b, &surfaces)
        }
        r => r,
    }
}

/// `b` with each face of `by` moved that far along its outward normal, onto its offset surface;
/// empty where every face closes on the way.
pub fn offset_faces(b: &Brep, by: &HashMap<FaceId, f64>) -> Result<Brep> {
    offset_faces_closing(b, by).map(|(out, _)| out)
}

/// [`offset_faces`], and the faces of `b` that closed on the way.
pub(crate) fn offset_faces_closing(
    b: &Brep,
    by: &HashMap<FaceId, f64>,
) -> Result<(Brep, Vec<FaceId>)> {
    offset_faces_traced(b, by).map(|(out, closed, _)| (out, closed))
}

/// [`offset_faces_closing`], and whether the solid kept its topology: its vertices, edges, loops
/// and faces are then `b`'s, one for one in `b`'s order.
pub(crate) fn offset_faces_traced(
    b: &Brep,
    by: &HashMap<FaceId, f64>,
) -> Result<(Brep, Vec<FaceId>, bool)> {
    clear_of_contacts(b, by.keys())?;
    // Where a face closes on the way, a round offset past its radius or a riser between two faces
    // brought level, the offset is followed there.
    offset_kept(b, by)
        .map(|out| (out, Vec::new(), true))
        .or_else(|e| {
            let mut paths: HashMap<FaceId, through::Path> = HashMap::new();
            for (&f, &d) in by {
                let face = b.face(f);
                let natural = if face.reversed { -d } else { d };
                let (surface, bounds) = (face.surface.clone(), face_box(b, f));
                paths.insert(
                    f,
                    Box::new(move |s| match offset_surface(&surface, s * natural) {
                        Err(Error::Unsupported(_)) => offset_fitted(&surface, s * natural, bounds),
                        r => r,
                    }),
                );
            }
            // A face moved through another or past the face it meets is refused as it is: the
            // other ways of building the move would give a solid the move does not describe.
            let refused = |r: Error| match r {
                Error::InvalidInput(_) => r,
                _ => e.clone(),
            };
            let followed = through::through(b, &paths).or_else(|r| match r {
                Error::InvalidInput(_) => Err(r),
                _ => slabbed(b, by).map(|out| (out, Vec::new())).map_err(refused),
            });
            followed.map(|(out, closed)| (out, closed, false))
        })
}

/// [`offset_faces`] as `b` with the slabs of the faces moved out added and those of the faces
/// moved in taken away (Rossignac and Requicha 1986), where moved faces run into faces they did not
/// meet (holes widened until they merge) and the move cannot be followed. That is the moved solid
/// only where every edge a moved face is on keeps the corner the slabs make: a face beside one
/// that stays meets it square, so the slab's side lies on the other face; two faces moved by one
/// distance meet where the slabs leave a sharp corner (moved in at a convex edge, out at a concave
/// one) or tangentially. Otherwise an error.
fn slabbed(b: &Brep, by: &HashMap<FaceId, f64>) -> Result<Brep> {
    let refuse = || unsupported("moving faces whose slabs do not make the moved solid");
    let convex = crate::pick::convex_edges(b)?;
    let concave = crate::pick::concave_edges(b)?;
    let outward = |f: FaceId, p: Point3| {
        let s = &b.face(f).surface;
        let uv = s.project(p);
        let n = s.normal(uv.x, uv.y);
        if b.face(f).reversed { -n } else { n }
    };
    for e in b.edge_ids() {
        let uses: Vec<FaceId> = b
            .coedge_ids()
            .filter(|&c| b.coedge(c).edge == e)
            .map(|c| b.coedge_face(c))
            .collect();
        let [f, g] = uses[..] else {
            return refuse();
        };
        let (df, dg) = (
            by.get(&f).copied().unwrap_or(0.0),
            by.get(&g).copied().unwrap_or(0.0),
        );
        if f == g || (df == 0.0 && dg == 0.0) {
            continue;
        }
        let x = b.edge(e);
        let Some(c) = &x.curve else { continue };
        let sharp = if df < 0.0 { &convex } else { &concave };
        let kept = if df != 0.0 && dg != 0.0 {
            df == dg && (sharp.contains(&e) || !convex.contains(&e) && !concave.contains(&e))
        } else {
            (0..=4).all(|k| {
                let p = c.eval(x.range.0 + (x.range.1 - x.range.0) * k as f64 / 4.0);
                outward(f, p).dot(outward(g, p)).abs() <= tol::ANGULAR
            })
        };
        if !kept {
            return refuse();
        }
        // The slab's side lies on the face that stays only as far as that face reaches: a face
        // moved further has passed the face it meets.
        if df == 0.0 || dg == 0.0 {
            let (moved, d, stays) = if df != 0.0 { (f, df, g) } else { (g, dg, f) };
            let dom = crate::boolean::face_domain(b, stays);
            let past = (0..=4).any(|k| {
                let p = c.eval(x.range.0 + (x.range.1 - x.range.0) * k as f64 / 4.0);
                matches!(
                    dom.locate(p + outward(moved, p) * d),
                    crate::boolean::domain::Where::Outside
                )
            });
            if past {
                return Err(Error::InvalidInput(
                    "a face moved past the face it meets".into(),
                ));
            }
        }
    }
    let feature = "moved";
    let mut grow = vec![b.clone()];
    let mut cut = Vec::new();
    for f in b.face_ids() {
        match by.get(&f).copied().unwrap_or(0.0) {
            d if d > 0.0 => grow.push(super::near::slab(b, f, d, feature)?),
            d if d < 0.0 => cut.push(super::near::slab(b, f, d, feature)?),
            _ => {}
        }
    }
    let grown = crate::boolean::union_all(&grow.iter().collect::<Vec<_>>())?;
    if cut.is_empty() {
        return Ok(grown);
    }
    let cut = crate::boolean::union_all(&cut.iter().collect::<Vec<_>>())?;
    crate::boolean::difference(&grown, &cut)
}

/// [`offset_faces`] with the solid's topology kept.
fn offset_kept(b: &Brep, by: &HashMap<FaceId, f64>) -> Result<Brep> {
    let mut planes = HashMap::new();
    let mut surfaces = HashMap::new();
    for (&f, &d) in by {
        let face = b.face(f);
        let natural = if face.reversed { -d } else { d };
        if let Some(fr) = plane_of(&face.surface) {
            planes.insert(f, fr.translated(fr.z * natural));
        }
        let moved = match offset_surface(&face.surface, natural) {
            Err(Error::Unsupported(_)) => offset_fitted(&face.surface, natural, face_box(b, f))?,
            r => r?,
        };
        surfaces.insert(f, moved);
    }
    if planes.len() == surfaces.len() {
        return planes_or_faces(b, &planes);
    }
    move_faces(b, &surfaces)
}

/// `b` with each face of `moved` on its new plane, given by a frame whose `z` is the face's new
/// natural normal (on the side of its old one).
pub(crate) fn move_planes(b: &Brep, moved: &HashMap<FaceId, Frame>) -> Result<Brep> {
    clear_of_contacts(b, moved.keys())?;
    for (&f, fr) in moved {
        let Some(old) = plane_of(&b.face(f).surface) else {
            return unsupported("moving a face that is not a plane");
        };
        if old.z.dot(fr.z) <= 0.0 {
            return Err(Error::InvalidInput(
                "a moved plane must face the way the old one did".into(),
            ));
        }
    }
    let frame_of = |f: FaceId| moved.get(&f).copied().or(plane_of(&b.face(f).surface));
    let around = vertex_faces(b);
    let mut out = Brep::new();
    let mut shifted = vec![false; b.vertex_ids().count()];
    let mut verts = Vec::new();
    for v in b.vertex_ids() {
        let faces = around.get(&v).cloned().unwrap_or_default();
        let p = b.vertex(v).point;
        if !faces.iter().any(|f| moved.contains_key(f)) {
            verts.push(out.add_vertex(p));
            continue;
        }
        let mut planes = Vec::new();
        for &f in &faces {
            let Some(fr) = frame_of(f) else {
                return unsupported("moving a corner where a curved face meets the planes");
            };
            planes.push((fr.z, fr.z.dot(fr.origin)));
        }
        let Some(q) = meet(&planes) else {
            return unsupported("moving a corner whose planes do not meet at one point");
        };
        shifted[v.index()] = q.dist(p) > 0.0;
        verts.push(out.add_vertex(q));
    }
    let mut edge_faces: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for f in b.face_ids() {
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                edge_faces.entry(b.coedge(c).edge).or_default().push(f);
            }
        }
    }
    let mut rebuilt = vec![false; b.edge_ids().count()];
    let mut edges = Vec::new();
    for e in b.edge_ids() {
        let x = b.edge(e);
        let (s, t) = (verts[x.start.index()], verts[x.end.index()]);
        let touched = edge_faces
            .get(&e)
            .is_some_and(|fs| fs.iter().any(|f| moved.contains_key(f)))
            || shifted[x.start.index()]
            || shifted[x.end.index()];
        let id = match &x.curve {
            Some(Curve::Line(old)) if touched => {
                let (a, z) = (out.vertex(s).point, out.vertex(t).point);
                let Some((line, len)) = Line::through(a, z) else {
                    return Err(Error::InvalidResult(vec![
                        "a moved edge shrinks to nothing".into(),
                    ]));
                };
                // A face turned inside out both ways keeps its loops' sense, so the checker
                // passes it; its edges are what run backwards.
                if line.dir.dot(old.dir) <= 0.0 {
                    return Err(Error::InvalidResult(vec![
                        "a moved edge runs backwards: a face turns inside out".into(),
                    ]));
                }
                rebuilt[e.index()] = true;
                out.add_edge(Curve::Line(line), (0.0, len), s, t)
            }
            _ if touched => {
                return unsupported("moving a face bounded by a curved edge");
            }
            Some(c) => out.add_edge(c.clone(), x.range, s, t),
            None => out.add_degenerate_edge(s, x.range),
        };
        edges.push(id);
    }
    let mut faces = Vec::new();
    for f in b.face_ids() {
        let face = b.face(f);
        let surface = match moved.get(&f) {
            Some(&frame) => Surface::Plane(Plane { frame }),
            None => face.surface.clone(),
        };
        let plane = plane_of(&surface);
        let nf = out.add_face(surface, face.reversed, face.names.clone());
        for &lp in &face.loops {
            let specs = b
                .lp(lp)
                .coedges
                .iter()
                .map(|&c| {
                    let co = b.coedge(c);
                    let edge = edges[co.edge.index()];
                    let pcurve = match (&out.edge(edge).curve, plane) {
                        (Some(Curve::Line(l)), Some(fr))
                            if rebuilt[co.edge.index()] || moved.contains_key(&f) =>
                        {
                            affine_on(&fr, l)
                        }
                        _ => co.pcurve.clone(),
                    };
                    CoedgeSpec {
                        edge,
                        reversed: co.reversed,
                        pcurve,
                    }
                })
                .collect();
            out.add_loop(nf, specs);
        }
        faces.push((f, nf));
    }
    for sh in b.shell_ids() {
        let ids = b
            .shell(sh)
            .faces
            .iter()
            .map(|f| faces[f.index()].1)
            .collect();
        let _ = out.add_shell(ids);
    }
    for so in b.solid_ids() {
        out.add_solid_of(b.solid(so).shells.clone(), b.solid(so).outer);
    }
    out.pair_radial();
    let issues = check::check(&out);
    if issues.is_empty() {
        Ok(out)
    } else {
        Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.to_string()).collect(),
        ))
    }
}

/// How far a fitted offset may stray from the true one at any point it is checked: a tenth of the
/// linear tolerance, so the edges traced on it keep their own budget.
const OFFSET_FIT: f64 = 0.1 * tol::LINEAR;

/// The most samples along one parameter a fitted offset may take before it is refused.
const OFFSET_SAMPLES: usize = 1 << 13;

/// The derivative of `f` at `t`, by fourth-order differences of step `h`, one-sided where `t` lies
/// within two steps of an end of `(t0, t1)`.
fn derivative(f: &dyn Fn(f64) -> Point3, t: f64, h: f64, (t0, t1): (f64, f64)) -> Vec3 {
    if t - 2.0 * h >= t0 && t + 2.0 * h <= t1 {
        return ((f(t + h) - f(t - h)) * 8.0 - (f(t + 2.0 * h) - f(t - 2.0 * h))) / (12.0 * h);
    }
    let s = if t - 2.0 * h < t0 { h } else { -h };
    let at = |k: f64| f(t + s * k);
    (at(0.0) * -25.0 + at(1.0) * 48.0 - at(2.0) * 36.0 + at(3.0) * 16.0 - at(4.0) * 3.0)
        / (12.0 * s)
}

/// The C¹ cubic through `true_at` with its derivative at every node: the nodes are `breaks` (where
/// the curve itself may lose smoothness, so the offset may too) with each piece between them cut
/// evenly, and the pieces doubled until the cubic lies within [`OFFSET_FIT`] of `true_at` at every
/// piece's middle. Each node's knot is double, so the cubic keeps the offset's own continuity there.
pub(crate) fn fit_offset_curve(
    true_at: &dyn Fn(f64) -> Point3,
    (t0, t1): (f64, f64),
    period: Option<f64>,
    breaks: &[f64],
) -> Result<crate::geom::BSpline> {
    // A closed curve's offset closes too, and is fitted round its whole period.
    let closed = period.filter(|&p| t1 - t0 >= p - 1e-12);
    let (t0, t1) = match closed {
        Some(p) => (t0, t0 + p),
        None => (t0, t1),
    };
    let mut cuts: Vec<f64> = vec![t0];
    cuts.extend(
        breaks
            .iter()
            .copied()
            .filter(|&b| b > t0 + 1e-12 && b < t1 - 1e-12),
    );
    cuts.push(t1);
    let h = 1e-4 * (t1 - t0) / cuts.len() as f64;
    let mut per = 2;
    while (cuts.len() - 1) * per <= OFFSET_SAMPLES {
        let mut nodes: Vec<f64> = Vec::new();
        for w in cuts.windows(2) {
            for k in 0..per {
                nodes.push(w[0] + (w[1] - w[0]) * k as f64 / per as f64);
            }
        }
        nodes.push(t1);
        let pts: Vec<Point3> = nodes.iter().map(|&t| true_at(t)).collect();
        let dirs: Vec<Vec3> = nodes
            .iter()
            .map(|&t| {
                derivative(
                    true_at,
                    t,
                    h,
                    if closed.is_some() {
                        (f64::MIN, f64::MAX)
                    } else {
                        (t0, t1)
                    },
                )
            })
            .collect();
        let m = nodes.len() - 1;
        let step = |j: usize| nodes[j + 1] - nodes[j];
        let fit = if closed.is_some() {
            // Nodes 0 … m − 1, node m being node 0 a period on.
            let hs: Vec<f64> = (0..m).map(step).collect();
            let minus = |j: usize| pts[j] - dirs[j] * (hs[(j + m - 1) % m] / 3.0);
            let plus = |j: usize| pts[j] + dirs[j] * (hs[j] / 3.0);
            let mut ctrl = vec![plus(m - 1)];
            for j in 0..m {
                ctrl.push(minus(j));
                if j + 1 < m {
                    ctrl.push(plus(j));
                }
            }
            for k in 0..3 {
                ctrl.push(ctrl[k]);
            }
            let n = 2 * m;
            let period = t1 - t0;
            let s = |k: isize| -> f64 {
                let q = k.rem_euclid(n as isize) as usize;
                nodes[q / 2] + period * k.div_euclid(n as isize) as f64
            };
            let knots: Vec<f64> = (-3..=(n as isize + 3)).map(s).collect();
            crate::geom::BSpline::periodic_cubic(knots, ctrl)?
        } else {
            // Bézier pieces joined at double knots: an inner node is no control point, the two
            // either side of it carry it.
            let mut full = vec![pts[0]];
            for j in 0..m {
                full.push(pts[j] + dirs[j] * (step(j) / 3.0));
                full.push(pts[j + 1] - dirs[j + 1] * (step(j) / 3.0));
            }
            full.push(pts[m]);
            let mut knots = vec![t0; 4];
            for &t in &nodes[1..m] {
                knots.push(t);
                knots.push(t);
            }
            knots.extend([t1; 4]);
            crate::geom::BSpline::new(3, knots, full)?
        };
        let worst = nodes
            .windows(2)
            .map(|w| {
                let t = 0.5 * (w[0] + w[1]);
                fit.eval(t).dist(true_at(t))
            })
            .fold(0.0f64, f64::max);
        if worst <= OFFSET_FIT {
            return Ok(fit);
        }
        per *= 2;
    }
    unsupported("offsetting a face whose offset no spline holds to the tolerance")
}

/// Samples along each parameter where [`folds`] looks for the offset running backwards.
const FOLD_SAMPLES: usize = 256;

/// Whether `s` offset by `d` along its natural normal runs backwards somewhere over `(u, v)`:
/// where `d` passes a radius of curvature, the offset's derivative along that parameter turns
/// against the surface's own, `(1 − d·k)` times it, and the offset loops over itself (Farouki and
/// Neff 1990). An extrusion is checked along its curve and a revolution along its meridian, the
/// only directions in which they bend; any other surface along both parameters on a grid.
fn folds(s: &Surface, d: f64, (u, v): ((f64, f64), (f64, f64))) -> bool {
    let at = |a: f64, b: f64| s.eval(a, b) + s.normal(a, b) * d;
    // Against the surface's own derivative, the offset's by central differences over a step small
    // beside the span sampled. Where the parameters degenerate, as at a pole, the surface's own
    // derivative vanishes and says nothing.
    let backwards = |a: f64, b: f64, along_u: bool| {
        let (_, su, sv) = s.d1(a, b);
        let (own, h) = if along_u {
            (su, 1e-6 * (u.1 - u.0))
        } else {
            (sv, 1e-6 * (v.1 - v.0))
        };
        if own.norm() <= 1e-9 * (1.0 + d.abs()) {
            return false;
        }
        let moved = if along_u {
            at(a + h, b) - at(a - h, b)
        } else {
            at(a, b + h) - at(a, b - h)
        };
        moved.dot(own) < 0.0
    };
    let steps =
        |(lo, hi): (f64, f64), n: usize| (0..=n).map(move |k| lo + (hi - lo) * k as f64 / n as f64);
    match s {
        Surface::Extrusion(_) => steps(u, FOLD_SAMPLES).any(|a| backwards(a, v.0, true)),
        Surface::Revolution(_) => steps(v, FOLD_SAMPLES).any(|b| backwards(u.0, b, false)),
        _ => {
            let n = FOLD_SAMPLES / 8;
            steps(u, n)
                .any(|a| steps(v, n).any(|b| backwards(a, b, true) || backwards(a, b, false)))
        }
    }
}

/// The offset of a surface with no exact offset of its own kind, `d` along its natural normal,
/// over `(u, v)` (a face's parameter box, with room): an extrusion's is the extrusion of its
/// curve's offset and a revolution's the revolution of its meridian's, so only that curve is
/// fitted; any other surface is fitted as a bicubic through a grid. Every fit is checked against
/// the true offset `p + d·n` at the middle of each span and refined until within [`OFFSET_FIT`].
fn offset_fitted(s: &Surface, d: f64, (u, v): ((f64, f64), (f64, f64))) -> Result<Surface> {
    let at = |a: f64, b: f64| s.eval(a, b) + s.normal(a, b) * d;
    if folds(s, d, (u, v)) {
        return unsupported(
            "offsetting a face past its radius of curvature, where its offset loops over itself",
        );
    }
    match s {
        Surface::Extrusion(e) => {
            let breaks = e.curve.breakpoints(u.0, u.1);
            let fit = fit_offset_curve(&|t| at(t, 0.0), u, e.curve.period(), &breaks)?;
            Ok(Surface::Extrusion(Box::new(crate::geom::Extrusion {
                curve: Curve::BSpline(fit),
                dir: e.dir,
            })))
        }
        Surface::Revolution(r) => {
            let breaks = r.curve.breakpoints(v.0, v.1);
            let fit = fit_offset_curve(&|t| at(0.0, t), v, r.curve.period(), &breaks)?;
            Ok(Surface::Revolution(Box::new(crate::geom::Revolution {
                curve: Curve::BSpline(fit),
                origin: r.origin,
                axis: r.axis,
            })))
        }
        _ => {
            // The surface's own knots cut the box into blocks: across one the offset's curvature
            // jumps, so an interpolant through it would converge only as h². Each block is fitted
            // on its own and the blocks are joined at triple knots, along the curves both share.
            let cuts = |(lo, hi): (f64, f64), breaks: Vec<f64>| {
                let mut c = vec![lo];
                c.extend(
                    breaks
                        .into_iter()
                        .filter(|&k| k > lo + 1e-12 && k < hi - 1e-12),
                );
                c.push(hi);
                c
            };
            let (ucuts, vcuts) = (
                cuts(u, s.u_breakpoints(u.0, u.1)),
                cuts(v, s.v_breakpoints(v.0, v.1)),
            );
            // Each column of blocks has its own steps across u and each row its own across v, so
            // the rows of a column, and the columns of a row, still share their nodes along each
            // join. A block off the tolerance halves the steps of its column where the fit strays
            // between its u nodes, and of its row where it strays between its v nodes.
            let (mut uper, mut vper) =
                (vec![8usize; ucuts.len() - 1], vec![8usize; vcuts.len() - 1]);
            while uper.iter().sum::<usize>() <= 512 && vper.iter().sum::<usize>() <= 512 {
                // Even steps, each block's end steps halved: a not-a-knot interpolant's error is
                // largest there, and at a block's corner both directions' ends meet.
                let nodes = |c: &[f64], per: &[usize]| -> Vec<Vec<f64>> {
                    c.windows(2)
                        .zip(per)
                        .map(|(w, &per)| {
                            let h = (w[1] - w[0]) / per as f64;
                            let mut n: Vec<f64> = (0..=per).map(|k| w[0] + h * k as f64).collect();
                            n.insert(1, w[0] + 0.5 * h);
                            n.insert(n.len() - 1, w[1] - 0.5 * h);
                            n
                        })
                        .collect()
                };
                let (ubl, vbl) = (nodes(&ucuts, &uper), nodes(&vcuts, &vper));
                let fit = fit_blocks(&at, &ubl, &vbl)?;
                let miss = |a: f64, b: f64| fit.eval(a, b).dist(at(a, b));
                let (mut finer_u, mut finer_v) = (vec![false; uper.len()], vec![false; vper.len()]);
                for (i, us) in ubl.iter().enumerate() {
                    for (j, vs) in vbl.iter().enumerate() {
                        for w in us.windows(2) {
                            for z in vs.windows(2) {
                                let (a, b) = (0.5 * (w[0] + w[1]), 0.5 * (z[0] + z[1]));
                                let (across_u, across_v) = (miss(a, z[0]), miss(w[0], b));
                                // Off only in the middle: the direction that strays more.
                                let middle =
                                    across_u.max(across_v) <= OFFSET_FIT && miss(a, b) > OFFSET_FIT;
                                finer_u[i] |=
                                    across_u > OFFSET_FIT || middle && across_u >= across_v;
                                finer_v[j] |=
                                    across_v > OFFSET_FIT || middle && across_v > across_u;
                            }
                        }
                    }
                }
                if !finer_u.contains(&true) && !finer_v.contains(&true) {
                    return Ok(Surface::Nurbs(Box::new(fit)));
                }
                for (per, finer) in uper
                    .iter_mut()
                    .zip(finer_u)
                    .chain(vper.iter_mut().zip(finer_v))
                {
                    if finer {
                        *per *= 2;
                    }
                }
            }
            unsupported("offsetting a face whose offset no spline holds to the tolerance")
        }
    }
}

/// The bicubic through `at` on the grid of every block of `ubl` × `vbl`, the parameter nodes of
/// each block in each direction: each block's tensor-product interpolant (rows through the grid,
/// then columns through the rows' control points), joined to the next at a triple knot. Blocks
/// that meet share their boundary nodes, so their nets agree along the join.
fn fit_blocks(
    at: &dyn Fn(f64, f64) -> Point3,
    ubl: &[Vec<f64>],
    vbl: &[Vec<f64>],
) -> Result<crate::geom::NurbsSurface> {
    use crate::geom::BSpline;
    // One block's knots in each direction and its net, u-major.
    let block = |us: &[f64], vs: &[f64]| -> Result<(Vec<f64>, Vec<f64>, Vec<Point3>)> {
        let mut rows = Vec::with_capacity(vs.len());
        for &b in vs {
            let pts: Vec<Point3> = us.iter().map(|&a| at(a, b)).collect();
            rows.push(BSpline::interpolate_at(&pts, us)?);
        }
        let n = rows[0].control_points().len();
        let mut cols = Vec::with_capacity(n);
        for i in 0..n {
            let pts: Vec<Point3> = rows.iter().map(|r| r.control_points()[i]).collect();
            cols.push(BSpline::interpolate_at(&pts, vs)?);
        }
        let ctrl = cols
            .iter()
            .flat_map(|c| c.control_points().iter().copied())
            .collect();
        Ok((rows[0].knots().to_vec(), cols[0].knots().to_vec(), ctrl))
    };
    // Clamped cubic knot vectors joined end to start, the join's knot kept three times.
    let join = |parts: &[Vec<f64>]| -> Vec<f64> {
        let mut out = Vec::new();
        for (k, p) in parts.iter().enumerate() {
            let from = if k == 0 { 0 } else { 4 };
            let to = if k + 1 == parts.len() {
                p.len()
            } else {
                p.len() - 1
            };
            out.extend_from_slice(&p[from..to]);
        }
        out
    };
    let mut blocks = Vec::new();
    for us in ubl {
        let mut row = Vec::new();
        for vs in vbl {
            row.push(block(us, vs)?);
        }
        blocks.push(row);
    }
    let ku = join(&blocks.iter().map(|r| r[0].0.clone()).collect::<Vec<_>>());
    let kv = join(&blocks[0].iter().map(|b| b.1.clone()).collect::<Vec<_>>());
    let counts: Vec<(usize, usize)> = blocks[0]
        .iter()
        .map(|b| (b.0.len() - 4, b.1.len() - 4))
        .collect();
    let nus: Vec<usize> = blocks.iter().map(|r| r[0].0.len() - 4).collect();
    let nvs: Vec<usize> = counts.iter().map(|c| c.1).collect();
    let nu = nus.iter().sum::<usize>() + 1 - nus.len();
    let nv = nvs.iter().sum::<usize>() + 1 - nvs.len();
    let mut ctrl = vec![Point3::ZERO; nu * nv];
    let mut i0 = 0;
    for (bi, row) in blocks.iter().enumerate() {
        let mut j0 = 0;
        for (bj, (_, _, net)) in row.iter().enumerate() {
            for i in 0..nus[bi] {
                for j in 0..nvs[bj] {
                    ctrl[(i0 + i) * nv + j0 + j] = net[i * nvs[bj] + j];
                }
            }
            j0 += nvs[bj] - 1;
        }
        i0 += nus[bi] - 1;
    }
    crate::geom::NurbsSurface::new((3, 3), (ku, kv), (nu, nv), ctrl, None)
}

/// The box of `f`'s boundary in its surface's parameters, read from its pcurves (a pole's among
/// them, so a face closing on one spans all of it), each side grown by a tenth of its span so the
/// moved edges stay inside. Along a period the face wraps whole, it is that one period; along one
/// it does not, it may run past the period's ends; otherwise it is held to the surface's domain.
fn face_box(b: &Brep, f: FaceId) -> ((f64, f64), (f64, f64)) {
    let face = b.face(f);
    let s = &face.surface;
    let (mut lo, mut hi) = (
        Vec2::new(f64::INFINITY, f64::INFINITY),
        Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for &lp in &face.loops {
        for &c in &b.lp(lp).coedges {
            for uv in b.coedge_uvs(c, &check::coedge_params(b, c, 16)) {
                lo = Vec2::new(lo.x.min(uv.x), lo.y.min(uv.y));
                hi = Vec2::new(hi.x.max(uv.x), hi.y.max(uv.y));
            }
        }
    }
    let (du, dv) = match s {
        Surface::Extrusion(e) => (Some(e.curve.domain()), None),
        Surface::Revolution(r) => (None, Some(r.curve.domain())),
        Surface::Nurbs(n) => {
            let (du, dv) = n.domain();
            (Some(du), Some(dv))
        }
        _ => (None, None),
    };
    let side = |a: f64, z: f64, period: Option<f64>, dom: Option<(f64, f64)>| {
        if let Some(p) = period.filter(|&p| z - a >= p - 1e-9 * p) {
            return (a, a + p);
        }
        let m = 0.1 * (z - a).max(1e-6);
        match (period, dom) {
            (None, Some(d)) => ((a - m).max(d.0), (z + m).min(d.1)),
            _ => (a - m, z + m),
        }
    };
    (
        side(lo.x, hi.x, s.u_period(), du),
        side(lo.y, hi.y, s.v_period(), dv),
    )
}

/// `s` moved `d` along its natural normal: for planes and the analytic surfaces of revolution the
/// same kind of surface, exactly. A plane moves along its normal; a cylinder's, sphere's or
/// torus's (minor) radius grows by `d`; a cone keeps its angle, its radius at the frame's origin
/// growing by `d·cos a` as the origin moves `−d·sin a` along the axis.
pub(crate) fn offset_surface(s: &Surface, d: f64) -> Result<Surface> {
    let positive = |r: f64, what: &str| {
        if r > tol::LINEAR {
            Ok(r)
        } else {
            Err(Error::InvalidInput(format!(
                "an offset of {d} mm closes a {what} to nothing"
            )))
        }
    };
    Ok(match s {
        Surface::Plane(p) => Surface::Plane(Plane {
            frame: p.frame.translated(p.frame.z * d),
        }),
        Surface::Cylinder(c) => Surface::Cylinder(Cylinder {
            frame: c.frame,
            radius: positive(c.radius + d, "cylinder")?,
        }),
        Surface::Cone(c) => {
            let (sa, ca) = c.half_angle.sin_cos();
            Surface::Cone(Cone {
                frame: c.frame.translated(c.frame.z * (-d * sa)),
                radius: c.radius + d * ca,
                half_angle: c.half_angle,
            })
        }
        Surface::Sphere(sp) => Surface::Sphere(Sphere {
            frame: sp.frame,
            radius: positive(sp.radius + d, "sphere")?,
        }),
        Surface::Torus(t) => Surface::Torus(Torus {
            frame: t.frame,
            major: t.major,
            minor: positive(t.minor + d, "torus")?,
        }),
        other => return unsupported(&format!("offsetting a {} face", other.kind())),
    })
}

/// The point nearest `p` on every surface of `on`, by Gauss–Newton steps along their normals:
/// the least move that puts it on all of them where they leave it free (a seam's end on two),
/// their one meeting where three or more pin it.
fn settle(p: Point3, on: &[&Surface]) -> Option<Point3> {
    let mut x = p;
    for _ in 0..60 {
        let rows: Vec<(Vec3, f64)> = on
            .iter()
            .map(|s| {
                let uv = s.project(x);
                let q = s.eval(uv.x, uv.y);
                let n = s.normal(uv.x, uv.y);
                (n, n.dot(q - x))
            })
            .collect();
        if rows.iter().all(|r| r.1.abs() <= 1e-3 * tol::LINEAR) {
            return Some(x);
        }
        // Faces tangent at the point (a round and the wall it runs into) stay tangent when both
        // move by one distance: their rows say the same, and one of them is kept.
        let mut kept: Vec<(Vec3, f64)> = Vec::new();
        for r in rows {
            if !kept.iter().any(|q| q.0.cross(r.0).norm() <= 1e-9) {
                kept.push(r);
            }
        }
        let rows = kept;
        let k = rows.len();
        let step = if k <= 3 {
            // Least norm: x moves by Jᵀy with (J Jᵀ) y = r.
            let mut a = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    a[i * k + j] = rows[i].0.dot(rows[j].0);
                }
            }
            let y = solve_dense(a, rows.iter().map(|r| r.1).collect(), k, 1).ok()?;
            rows.iter()
                .zip(&y)
                .fold(Vec3::ZERO, |acc, (r, &yi)| acc + r.0 * yi)
        } else {
            let mut a = vec![0.0; 9];
            let mut rhs = vec![0.0; 3];
            for (n, r) in &rows {
                let n = n.to_array();
                for i in 0..3 {
                    for j in 0..3 {
                        a[i * 3 + j] += n[i] * n[j];
                    }
                    rhs[i] += n[i] * r;
                }
            }
            let s = solve_dense(a, rhs, 3, 1).ok()?;
            Vec3::new(s[0], s[1], s[2])
        };
        if !(step.x.is_finite() && step.y.is_finite() && step.z.is_finite()) {
            return None;
        }
        x += step;
    }
    None
}

/// The point nearest `p` where the two surfaces of `on` meet, for a point [`settle`] cannot move:
/// where they are tangent at `p` (a wall moved off the round it ran into), their normals give its
/// steps one direction and they turn back and forth along it. Points on the faces of `fixed`, which
/// stay where they are, come before the rest.
fn meet_near(b: &Brep, p: Point3, on: &[&Surface], fixed: &[FaceId]) -> Option<Point3> {
    let [s0, s1] = on else { return None };
    let bx = crate::measure::bbox(b);
    let reach = bx.min.dist(bx.max) + 1.0;
    let pad = Vec3::new(reach, reach, reach);
    let region = crate::math::Aabb {
        min: bx.min - pad,
        max: bx.max + pad,
    };
    let crate::boolean::ssi::Ssi::Curves { curves, points } =
        crate::boolean::ssi::intersect(s0, s1, &region, &mut || vec![p]).ok()?
    else {
        return None;
    };
    let domains: Vec<_> = fixed
        .iter()
        .map(|&f| crate::boolean::face_domain(b, f))
        .collect();
    let on_fixed = |q: Point3| {
        domains
            .iter()
            .all(|d| !matches!(d.locate(q), crate::boolean::domain::Where::Outside))
    };
    curves
        .iter()
        .map(|sc| sc.curve.eval(sc.curve.project(p, sc.range.0, sc.range.1)))
        .chain(points)
        .min_by(|q, r| {
            (!on_fixed(*q), q.dist(p))
                .partial_cmp(&(!on_fixed(*r), r.dist(p)))
                .expect("finite")
        })
}

/// The circle through three points, if they are not on a line.
fn circle_through(a: Point3, b: Point3, c: Point3) -> Option<crate::geom::Circle> {
    let (ab, ac) = (b - a, c - a);
    let n = ab.cross(ac);
    let n2 = n.norm2();
    // Relative to the chords, so a circle closing to nothing (a tube's) is still found while small.
    if n2 <= 1e-24 * ab.norm2() * ac.norm2() {
        return None;
    }
    let centre = a + (n.cross(ab) * ac.norm2() + ac.cross(n) * ab.norm2()) * (0.5 / n2);
    let radius = centre.dist(a);
    let frame = Frame::new(centre, n, a - centre).ok()?;
    Some(crate::geom::Circle { frame, radius })
}

/// The surface parameters of `p` on `s` near `last`, moved by whole periods to continue it.
pub(super) fn near_uv(s: &Surface, p: Point3, last: Option<Vec2>) -> Vec2 {
    let uv = s.project(p);
    let Some(to) = last else { return uv };
    let shift = |x: f64, t: f64, per: Option<f64>| per.map_or(x, |q| x + ((t - x) / q).round() * q);
    Vec2::new(
        shift(uv.x, to.x, s.u_period()),
        shift(uv.y, to.y, s.v_period()),
    )
}

/// The `v` of the pole of `s` nearest `near`, where `u` runs round the axis and the surface closes
/// on it: a cone's apex, a sphere's pole. `None` for a surface with no such pole in closed form.
fn pole_v(s: &Surface, near: f64) -> Option<f64> {
    match s {
        Surface::Cone(c) if c.half_angle.sin().abs() > tol::ANGULAR => {
            Some(-c.radius / c.half_angle.sin())
        }
        Surface::Sphere(_) => Some(std::f64::consts::FRAC_PI_2.copysign(near)),
        _ => None,
    }
}

/// A pole's pcurve on `from` carried onto the pole of `to`, the surface it moved to: a moved cone's
/// apex lies at another `v` than the old one's.
fn on_pole(from: &Surface, to: &Surface, pcurve: &PCurve) -> PCurve {
    if let PCurve::Affine { origin, dir } = pcurve
        && dir.y == 0.0
        && let (Some(old), Some(new)) = (pole_v(from, origin.y), pole_v(to, origin.y))
        && (origin.y - old).abs() <= 1e-9 * (1.0 + old.abs())
    {
        return PCurve::Affine {
            origin: Vec2::new(origin.x, new),
            dir: *dir,
        };
    }
    pcurve.clone()
}

/// `b` with each face of `moved` on its new surface (natural normal on the side of its old one),
/// its topology kept: every vertex goes where its faces' surfaces meet, and every edge becomes the
/// meeting of its two faces' surfaces between its new vertices (a straight one stays a line where
/// that line lies on both), found in closed form where one exists and traced otherwise.
pub fn move_faces(b: &Brep, moved: &HashMap<FaceId, Surface>) -> Result<Brep> {
    move_faces_pinned(b, moved, &HashMap::new())
}

/// `b` with each face of `faces` carried by the rigid motion `motion`, as [`move_faces`] puts
/// each on its moved surface.
pub fn move_faces_by(b: &Brep, faces: &[FaceId], motion: &Transform) -> Result<Brep> {
    let r = motion.r;
    let dot = |i: usize, j: usize| (0..3).map(|k| r[k][i] * r[k][j]).sum::<f64>();
    let square = (0..3)
        .all(|i| (0..3).all(|j| (dot(i, j) - f64::from(u8::from(i == j))).abs() <= tol::ANGULAR));
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if !square || (det - 1.0).abs() > tol::ANGULAR {
        return invalid("faces move by a rigid motion: a rotation and a shift");
    }
    let moved = faces
        .iter()
        .map(|&f| (f, b.face(f).surface.transformed(motion)))
        .collect();
    move_faces_carried(b, &moved, &HashMap::new(), Some(motion))
}

/// [`move_faces`], with each vertex of `pins` put at its point where that lies on every new surface
/// at the vertex. A vertex its surfaces leave free to slide (a seam's end, on a circle) otherwise
/// moves the least distance onto them, which keeps how far along the curve it was.
pub(crate) fn move_faces_pinned(
    b: &Brep,
    moved: &HashMap<FaceId, Surface>,
    pins: &HashMap<VertexId, Point3>,
) -> Result<Brep> {
    move_faces_carried(b, moved, pins, None)
}

/// [`move_faces_pinned`], each edge of a face carried by `motion` found through its old middle
/// carried by it too. Settled from where it was, the middle of an arc moved further than its
/// radius lands on the far side of the arc's circle, and the edge would take the rest of it.
fn move_faces_carried(
    b: &Brep,
    moved: &HashMap<FaceId, Surface>,
    pins: &HashMap<VertexId, Point3>,
    motion: Option<&Transform>,
) -> Result<Brep> {
    clear_of_contacts(b, moved.keys())?;
    let surface_of = |f: FaceId| moved.get(&f).unwrap_or(&b.face(f).surface);
    let around = vertex_faces(b);
    let mut edge_faces: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for f in b.face_ids() {
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                edge_faces.entry(b.coedge(c).edge).or_default().push(f);
            }
        }
    }
    // A pole (a degenerate edge's vertex) goes to its surface's new pole exactly: its parameters
    // there are the old ones, while a point settled onto the surface sits off the pole by about √ε
    // in angle, where u is no longer free and projection gives any u at all.
    let mut poles: HashMap<VertexId, Point3> = HashMap::new();
    for c in b.coedge_ids() {
        let co = b.coedge(c);
        let edge = b.edge(co.edge);
        let f = b.coedge_face(c);
        if edge.curve.is_some() || !moved.contains_key(&f) {
            continue;
        }
        if let PCurve::Affine { origin, dir } =
            on_pole(&b.face(f).surface, surface_of(f), &co.pcurve)
        {
            let uv = origin + dir * edge.range.0;
            poles.insert(edge.start, surface_of(f).eval(uv.x, uv.y));
        }
    }
    let mut out = Brep::new();
    let mut shifted = vec![false; b.vertex_ids().count()];
    let mut verts = Vec::new();
    for v in b.vertex_ids() {
        let faces = around.get(&v).cloned().unwrap_or_default();
        let p = b.vertex(v).point;
        if !faces.iter().any(|f| moved.contains_key(f)) {
            verts.push(out.add_vertex(p));
            continue;
        }
        let on: Vec<&Surface> = faces.iter().map(|&f| surface_of(f)).collect();
        let pinned = pins.get(&v).or(poles.get(&v)).copied().filter(|&q| {
            on.iter()
                .all(|s| crate::boolean::csi::signed_distance(s, q).abs() <= 1e-2 * tol::LINEAR)
        });
        let fixed: Vec<FaceId> = faces
            .iter()
            .copied()
            .filter(|f| !moved.contains_key(f))
            .collect();
        let Some(q) = pinned
            .or_else(|| settle(p, &on))
            .or_else(|| meet_near(b, p, &on, &fixed))
        else {
            return unsupported("moving a corner whose faces' new surfaces do not meet near it");
        };
        shifted[v.index()] = q.dist(p) > 0.0;
        verts.push(out.add_vertex(q));
    }
    let mut flipped = vec![false; b.edge_ids().count()];
    let mut rebuilt = vec![false; b.edge_ids().count()];
    let mut edges = Vec::new();
    for e in b.edge_ids() {
        let x = b.edge(e);
        let (s, t) = (verts[x.start.index()], verts[x.end.index()]);
        let mut faces = edge_faces.get(&e).cloned().unwrap_or_default();
        faces.sort();
        faces.dedup();
        let touched = faces.iter().any(|f| moved.contains_key(f))
            || shifted[x.start.index()]
            || shifted[x.end.index()];
        let Some(old) = &x.curve else {
            edges.push(out.add_degenerate_edge(s, x.range));
            continue;
        };
        if !touched {
            edges.push(out.add_edge(old.clone(), x.range, s, t));
            continue;
        }
        rebuilt[e.index()] = true;
        let on: Vec<&Surface> = faces.iter().map(|&f| surface_of(f)).collect();
        let lies_on = |c: &Curve, r: (f64, f64)| {
            (1..8).all(|k| {
                let p = c.eval(r.0 + (r.1 - r.0) * k as f64 / 8.0);
                on.iter()
                    .all(|sf| crate::boolean::csi::signed_distance(sf, p).abs() <= tol::LINEAR)
            })
        };
        let (a, z) = (out.vertex(s).point, out.vertex(t).point);
        if let (Curve::Line(old_line), Some((line, len))) = (old, Line::through(a, z))
            && lies_on(&Curve::Line(line.clone()), (0.0, len))
        {
            if line.dir.dot(old_line.dir) <= 0.0 {
                return Err(Error::InvalidResult(vec![
                    "a moved edge runs backwards: a face turns inside out".into(),
                ]));
            }
            edges.push(out.add_edge(Curve::Line(line), (0.0, len), s, t));
            continue;
        }
        if on.len() != 2 {
            // A seam: the new surface keeps its parameters (a frame kept, a radius grown), so the
            // seam is the same parameter line on it, here a circle through three of its points.
            let (Some(&f), true) = (faces.first(), faces.len() == 1) else {
                return unsupported("moving a curved edge of three faces");
            };
            let (from, to) = (&b.face(f).surface, surface_of(f));
            let fractions: &[f64] = if x.start == x.end {
                &[0.0, 1.0 / 3.0, 2.0 / 3.0]
            } else {
                &[0.0, 0.5, 1.0]
            };
            let pts: Vec<Point3> = fractions
                .iter()
                .map(|&k| {
                    let q = old.eval(x.range.0 + (x.range.1 - x.range.0) * k);
                    let uv = from.project(q);
                    to.eval(uv.x, uv.y)
                })
                .collect();
            // Its ends are the moved vertices, which a face beside it may have carried along it.
            let ends = if x.start == x.end {
                [a, pts[1], pts[2]]
            } else {
                [a, pts[1], z]
            };
            let pts = ends;
            let Some(circle) = circle_through(pts[0], pts[1], pts[2]) else {
                return unsupported("moving a seam that is not a circle");
            };
            let c = Curve::Circle(circle);
            let lo = c.project(pts[0], 0.0, TAU);
            let hi = if x.start == x.end {
                lo + TAU
            } else {
                let mut h = c.project(pts[2], 0.0, TAU);
                while h <= lo {
                    h += TAU;
                }
                h
            };
            let (old_mid, old_dir) = old.d1(0.5 * (x.range.0 + x.range.1));
            let _ = old_mid;
            let mid_t = 0.5 * (lo + hi);
            if c.d1(mid_t).1.dot(old_dir) <= 0.0 && x.start != x.end {
                return unsupported("moving a seam that turns the other way");
            }
            edges.push(out.add_edge(c, (lo, hi), s, t));
            continue;
        }
        // The old edge's middle and direction, carried onto both new surfaces.
        let mid_t = 0.5 * (x.range.0 + x.range.1);
        let (old_mid, old_dir) = old.d1(mid_t);
        let fixed: Vec<FaceId> = faces
            .iter()
            .copied()
            .filter(|f| !moved.contains_key(f))
            .collect();
        let (from, old_dir) = match motion {
            Some(m) if fixed.len() < faces.len() => (m.point(old_mid), m.vector(old_dir)),
            _ => (old_mid, old_dir),
        };
        let Some(mid) = settle(from, &on).or_else(|| meet_near(b, from, &on, &fixed)) else {
            return unsupported("moving an edge whose faces' new surfaces do not meet near it");
        };
        let mut region = old.bbox(x.range.0, x.range.1);
        region.add(a);
        region.add(z);
        region.add(mid);
        let reach = old_mid.dist(mid) + a.dist(b.vertex(x.start).point) + 1.0;
        let pad = Vec3::new(reach, reach, reach);
        let region = crate::math::Aabb {
            min: region.min - pad,
            max: region.max + pad,
        };
        let ssi = crate::boolean::ssi::intersect(on[0], on[1], &region, &mut || vec![mid])?;
        let crate::boolean::ssi::Ssi::Curves { curves, .. } = ssi else {
            return unsupported("moving an edge between faces that come to coincide");
        };
        let Some(best) = curves
            .iter()
            .map(|sc| {
                let tm = sc.curve.project(mid, sc.range.0, sc.range.1);
                (sc.curve.eval(tm).dist(mid), sc)
            })
            .filter(|(d, _)| *d <= 10.0 * tol::LINEAR)
            .min_by(|p, q| p.0.total_cmp(&q.0))
            .map(|(_, sc)| sc)
        else {
            return unsupported("moving an edge whose new faces meet nowhere near it");
        };
        let c = &best.curve;
        let (r0, r1) = best.range;
        let ts = c.project(a, r0, r1);
        let te = c.project(z, r0, r1);
        let tm = c.project(mid, r0, r1);
        // A closed edge's direction is its turn: moved further than its size, its old middle
        // settles anywhere round the new one, where the tangent can run either way.
        let along = if x.start == x.end {
            let (_, d1, d2) = old.d2(mid_t);
            let (_, e1, e2) = c.d2(tm);
            d1.cross(d2).dot(e1.cross(e2)) > 0.0
        } else {
            c.d1(tm).1.dot(old_dir) > 0.0
        };
        let period = c.period();
        // The stretch from the start vertex to the end one through the middle, in the curve's own
        // direction; where that runs from the end, the edge is laid the other way.
        let span = |from: f64, to: f64| -> (f64, f64) {
            match period {
                Some(p) => {
                    let mut hi = to;
                    while hi <= from + 1e-12 {
                        hi += p;
                    }
                    while hi - from > p + 1e-12 {
                        hi -= p;
                    }
                    (from, hi)
                }
                None => (from, to),
            }
        };
        let closed = x.start == x.end;
        let (lo, hi, start, end) = if along {
            let (lo, hi) = if closed {
                (ts, ts + period.unwrap_or(r1 - r0))
            } else {
                span(ts, te)
            };
            (lo, hi, s, t)
        } else {
            let (lo, hi) = if closed {
                (ts, ts + period.unwrap_or(r1 - r0))
            } else {
                span(te, ts)
            };
            flipped[e.index()] = true;
            (lo, hi, t, s)
        };
        if hi - lo <= 0.0
            || c.eval(lo).dist(out.vertex(start).point) > tol::LINEAR
            || c.eval(hi).dist(out.vertex(end).point) > tol::LINEAR
        {
            return unsupported("moving an edge whose new curve does not run between its vertices");
        }
        edges.push(out.add_edge(c.clone(), (lo, hi), start, end));
    }
    let mut faces = Vec::new();
    for f in b.face_ids() {
        let face = b.face(f);
        let surface = surface_of(f).clone();
        // A loop with anything rebuilt is re-seeded whole, so its parameters run on unbroken.
        let changed = moved.contains_key(&f)
            || face.loops.iter().any(|&lp| {
                b.lp(lp)
                    .coedges
                    .iter()
                    .any(|&c| rebuilt[b.coedge(c).edge.index()])
            });
        let nf = out.add_face(surface.clone(), face.reversed, face.names.clone());
        for &lp in &face.loops {
            // The loop starts where it did: a moved surface keeps its parameters, so the old
            // start places a periodic u on the right side of the period.
            let mut last: Option<Vec2> = b.lp(lp).coedges.first().map(|&c| {
                let ts = check::coedge_params(b, c, 1);
                b.coedge_uvs(c, &ts[..1])[0]
            });
            let specs = b
                .lp(lp)
                .coedges
                .iter()
                .map(|&c| {
                    let co = b.coedge(c);
                    let edge = edges[co.edge.index()];
                    let reversed = co.reversed != flipped[co.edge.index()];
                    let x = out.edge(edge);
                    // A degenerate edge (a pole) keeps its own line across the parameter plane,
                    // and the next coedge carries on from its end.
                    if x.curve.is_none() {
                        let pcurve = on_pole(&face.surface, &surface, &co.pcurve);
                        if let PCurve::Affine { origin, dir } = &pcurve {
                            let t = if reversed { x.range.0 } else { x.range.1 };
                            last = Some(*origin + *dir * t);
                        }
                        return CoedgeSpec {
                            edge,
                            reversed,
                            pcurve,
                        };
                    }
                    let pcurve = if changed || rebuilt[co.edge.index()] {
                        let from = if reversed { x.end } else { x.start };
                        let mut seed = near_uv(&surface, out.vertex(from).point, last);
                        if let Some(cv) = &x.curve {
                            let (t0, t1) = if reversed {
                                (x.range.1, x.range.0)
                            } else {
                                x.range
                            };
                            // As the pcurve itself reads its start: at a pole, the u the edge
                            // leaves it by, not whatever projecting the pole gives.
                            let hint = last.unwrap_or(seed);
                            seed = crate::topo::pcurve::project_on(&surface, cv, t0, t1, hint);
                        }
                        let mut uv = seed;
                        if let Some(cv) = &x.curve {
                            let (t0, t1) = if reversed {
                                (x.range.1, x.range.0)
                            } else {
                                x.range
                            };
                            for k in 1..=32 {
                                let p = cv.eval(t0 + (t1 - t0) * k as f64 / 32.0);
                                uv = near_uv(&surface, p, Some(uv));
                            }
                        }
                        last = Some(uv);
                        PCurve::Projected { seed }
                    } else {
                        co.pcurve.clone()
                    };
                    CoedgeSpec {
                        edge,
                        reversed,
                        pcurve,
                    }
                })
                .collect();
            out.add_loop(nf, specs);
        }
        faces.push((f, nf));
    }
    for sh in b.shell_ids() {
        let ids = b
            .shell(sh)
            .faces
            .iter()
            .map(|f| faces[f.index()].1)
            .collect();
        let _ = out.add_shell(ids);
    }
    for so in b.solid_ids() {
        out.add_solid_of(b.solid(so).shells.clone(), b.solid(so).outer);
    }
    out.pair_radial();
    let issues = check::check(&out);
    if issues.is_empty() {
        Ok(out)
    } else {
        Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.to_string()).collect(),
        ))
    }
}

#[cfg(test)]
mod tests;
mod through;
