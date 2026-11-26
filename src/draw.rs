//! Hidden-line drawings: a solid's edges and the outlines of its curved faces seen along a
//! direction, each cut into the stretches the viewer sees and those the solid hides.
//!
//! Visibility along a curve changes only where, seen from the viewer, it passes behind another
//! edge or outline, or meets one. So every stroke (an edge between faces that meet at an angle,
//! or an outline where the line of sight grazes a curved face, trimmed to the face) is cut where
//! its projection crosses another's, found by Newton's method from the crossing chords, and where
//! another stroke ends on it. Each piece between cuts is then seen or hidden throughout, and the
//! exact ray caster decides which at its middle, far from where the line of sight grazes anything.

use std::f64::consts::TAU;

use crate::{
    error::{Error, Result},
    geom::{Circle, Curve, Line, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    measure::{bbox, in_face, uv_polygons},
    rays::Rays,
    tol,
    topo::{Brep, FaceId},
};

/// A drawing's polylines, in space: the stretches seen and the stretches hidden.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Drawing {
    pub visible: Vec<Vec<Point3>>,
    pub hidden: Vec<Vec<Point3>>,
}

/// One curve of the drawing over `[t0, t1]`, and the faces it lies on: an edge's two, or the one
/// an outline grazes.
struct Stroke {
    curve: Curve,
    t0: f64,
    t1: f64,
    faces: Vec<FaceId>,
}

/// The drawing of `b` looking along `direction`, every chord within `tolerance` of its curve.
pub fn hidden_lines(b: &Brep, direction: Vec3, tolerance: f64) -> Result<Drawing> {
    let d = direction
        .unit()
        .ok_or_else(|| Error::InvalidInput("a view needs a direction".into()))?;
    let mut out = Drawing::default();
    if b.solid_ids().next().is_none() {
        return Ok(out);
    }
    let scale = bbox(b).diagonal().max(1.0);
    let view = Frame::new(Point3::ZERO, d, d.any_perpendicular())?;
    let flat = |p: Point3| {
        let l = view.to_local(p);
        Vec2::new(l.x, l.y)
    };
    let strokes = strokes(b, d, tolerance);

    // Each stroke's samples, close enough for the tolerance, and their projections.
    let samples: Vec<Vec<f64>> = strokes
        .iter()
        .map(|s| {
            let mut ts = Vec::new();
            sample(&|t| s.curve.eval(t), s.t0, s.t1, tolerance, 0, &mut ts);
            ts.push(s.t1);
            ts
        })
        .collect();
    let flats: Vec<Vec<Vec2>> = strokes
        .iter()
        .zip(&samples)
        .map(|(s, ts)| ts.iter().map(|&t| flat(s.curve.eval(t))).collect())
        .collect();

    // Where each stroke is cut: crossings with the others' projections, and their ends on it.
    let mut cuts: Vec<Vec<f64>> = vec![Vec::new(); strokes.len()];
    let near = 1e-9 * scale;
    for i in 0..strokes.len() {
        for j in 0..strokes.len() {
            if i == j {
                continue;
            }
            for (k, w) in flats[i].windows(2).enumerate() {
                for (l, x) in flats[j].windows(2).enumerate() {
                    if let Some((sa, _)) = chords_cross(w[0], w[1], x[0], x[1]) {
                        let ta = samples[i][k] + (samples[i][k + 1] - samples[i][k]) * sa;
                        let tb = samples[j][l] + (samples[j][l + 1] - samples[j][l]) * 0.5;
                        if let Some(t) = pin_crossing(&strokes[i], &strokes[j], ta, tb, &flat) {
                            cuts[i].push(t);
                        }
                    }
                }
            }
            let sj = &strokes[j];
            for end in [sj.curve.eval(sj.t0), sj.curve.eval(sj.t1)] {
                if let Some(t) = foot(&strokes[i], &samples[i], end, &flat)
                    && flat(strokes[i].curve.eval(t)).dist(flat(end)) <= near
                {
                    cuts[i].push(t);
                }
            }
        }
    }

    let rays = Rays::new(b)?;
    let eps = 1e-9 * scale;
    // Hidden where the ray toward the viewer leaves the solid. It starts a hair off the faces the
    // point lies on, out of the solid, so it runs beside them rather than along them.
    let hidden = |s: &Stroke, p: Point3| {
        let off = s
            .faces
            .iter()
            .map(|&f| outward(b, f, p))
            .fold(Vec3::ZERO, |a, n| a + n)
            .unit()
            .unwrap_or(Vec3::ZERO);
        rays.cast(p + (off - d) * (100.0 * eps), -d, (0.0, 1e3 * scale))
            .is_ok_and(|hits| hits.iter().any(|h| h.normal.dot(-d) > 1e-9))
    };

    for (i, s) in strokes.iter().enumerate() {
        let span = s.t1 - s.t0;
        let mut ts: Vec<f64> = cuts[i]
            .iter()
            .copied()
            .filter(|&t| (t - s.t0) * span.signum() > 1e-9 * span.abs())
            .filter(|&t| (s.t1 - t) * span.signum() > 1e-9 * span.abs())
            .collect();
        ts.push(s.t0);
        ts.push(s.t1);
        ts.sort_by(|x, y| ((x - s.t0) / span).total_cmp(&((y - s.t0) / span)));
        ts.dedup_by(|x, y| (*x - *y).abs() <= 1e-12 * span.abs().max(1.0));
        // Consecutive pieces of one visibility join.
        let mut runs: Vec<(bool, f64, f64)> = Vec::new();
        for w in ts.windows(2) {
            let mid = s.curve.eval(0.5 * (w[0] + w[1]));
            let h = hidden(s, mid);
            match runs.last_mut() {
                Some(last) if last.0 == h => last.2 = w[1],
                _ => runs.push((h, w[0], w[1])),
            }
        }
        for (h, a, z) in runs {
            let mut pts: Vec<f64> = samples[i]
                .iter()
                .copied()
                .filter(|&t| ((t - a) / span) > 1e-12 && ((z - t) / span) > 1e-12)
                .collect();
            pts.insert(0, a);
            pts.push(z);
            let line: Vec<Point3> = pts.iter().map(|&t| s.curve.eval(t)).collect();
            if h {
                out.hidden.push(line);
            } else {
                out.visible.push(line);
            }
        }
    }
    Ok(out)
}

/// The drawing's strokes: every edge but a seam or one between faces that meet smoothly, and each
/// outline trimmed to the stretches of it that lie on its face.
fn strokes(b: &Brep, d: Vec3, tolerance: f64) -> Vec<Stroke> {
    let mut out = Vec::new();
    for e in b.edge_ids() {
        let edge = b.edge(e);
        let Some(c) = &edge.curve else { continue };
        let faces: Vec<FaceId> = b
            .coedge_ids()
            .filter(|&co| b.coedge(co).edge == e)
            .map(|co| b.coedge_face(co))
            .collect();
        if faces.len() == 2 && faces[0] == faces[1] {
            continue;
        }
        let (t0, t1) = edge.range;
        let mid = c.eval(0.5 * (t0 + t1));
        if faces.len() == 2 && smooth(b, faces[0], faces[1], mid) {
            continue;
        }
        out.push(Stroke {
            curve: c.clone(),
            t0,
            t1,
            faces,
        });
    }
    for f in b.face_ids() {
        let s = &b.face(f).surface;
        let polys = uv_polygons(b, f);
        let on = |p: Point3| in_face(s, &polys, s.project(p)).is_some();
        for (curve, t0, t1) in outlines(b, f, d) {
            let mut ts = Vec::new();
            sample(&|t| curve.eval(t), t0, t1, tolerance, 0, &mut ts);
            ts.push(t1);
            let inside: Vec<bool> = ts.iter().map(|&t| on(curve.eval(t))).collect();
            // Each change of membership pinned by bisection.
            let pin = |mut a: f64, mut z: f64, la: bool| {
                for _ in 0..80 {
                    let m = 0.5 * (a + z);
                    if on(curve.eval(m)) == la {
                        a = m;
                    } else {
                        z = m;
                    }
                }
                0.5 * (a + z)
            };
            let mut start = inside[0].then_some(ts[0]);
            for k in 0..ts.len() - 1 {
                if inside[k] != inside[k + 1] {
                    let t = pin(ts[k], ts[k + 1], inside[k]);
                    match start.take() {
                        Some(a) => out.push(Stroke {
                            curve: curve.clone(),
                            t0: a,
                            t1: t,
                            faces: vec![f],
                        }),
                        None => start = Some(t),
                    }
                }
            }
            if let Some(a) = start {
                out.push(Stroke {
                    curve: curve.clone(),
                    t0: a,
                    t1: ts[ts.len() - 1],
                    faces: vec![f],
                });
            }
        }
    }
    out.retain(|s| (s.t1 - s.t0).abs() > 0.0);
    out
}

/// Face `f`'s outward unit normal at its point nearest `p`.
fn outward(b: &Brep, f: FaceId, p: Point3) -> Vec3 {
    let s = &b.face(f).surface;
    let uv = s.project(p);
    let n = s.normal(uv.x, uv.y);
    if b.face(f).reversed { -n } else { n }
}

/// Whether faces `f` and `g` meet smoothly at `p`: their outward normals agree there.
fn smooth(b: &Brep, f: FaceId, g: FaceId, p: Point3) -> bool {
    let (n, m) = (outward(b, f, p), outward(b, g, p));
    n.cross(m).norm() <= 1e-7 && n.dot(m) > 0.0
}

fn grazing_angles(view: Vec3, half_angle: f64) -> Option<[f64; 2]> {
    // In the surface frame, n(u)·view vanishes where x cos(u) + y sin(u) = axial.
    let (sin, cos) = half_angle.sin_cos();
    let (x, y, axial) = (view.x * cos, view.y * cos, view.z * sin);
    let amplitude = x.hypot(y);
    if amplitude <= axial.abs() + tol::ANGULAR {
        return None;
    }
    let phase = y.atan2(x);
    let spread = (axial / amplitude).acos();
    Some([phase - spread, phase + spread])
}

/// The curves on face `f`'s surface where the line of sight along `d` grazes it, each over a
/// parameter range that covers the face: generators of a cylinder or cone, the great circle of a
/// ball. Other surfaces have none drawn.
fn outlines(b: &Brep, f: FaceId, d: Vec3) -> Vec<(Curve, f64, f64)> {
    let s = &b.face(f).surface;
    let angles = match s {
        Surface::Cylinder(c) => grazing_angles(c.frame.dir_to_local(d), 0.0),
        Surface::Cone(c) => grazing_angles(c.frame.dir_to_local(d), c.half_angle),
        Surface::Sphere(sp) => {
            let Ok(frame) = Frame::new(sp.frame.origin, d, d.any_perpendicular()) else {
                return Vec::new();
            };
            let circle = Circle {
                frame,
                radius: sp.radius,
            };
            return vec![(Curve::Circle(circle), 0.0, TAU)];
        }
        _ => return Vec::new(),
    };
    let Some(angles) = angles else {
        return Vec::new();
    };
    let polys = uv_polygons(b, f);
    let (vlo, vhi) = polys
        .iter()
        .flatten()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), q| {
            (lo.min(q.y), hi.max(q.y))
        });
    angles
        .into_iter()
        .filter_map(|u| {
            let (line, len) = Line::through(s.eval(u, vlo), s.eval(u, vhi))?;
            Some((Curve::Line(line), 0.0, len))
        })
        .collect()
}

/// Where chords `ab` and `cd` cross, as fractions along each, if they do.
fn chords_cross(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Option<(f64, f64)> {
    let (r, s) = (b - a, d - c);
    let den = r.cross(s);
    if den.abs() <= 1e-300 {
        return None;
    }
    let q = c - a;
    let (t, u) = (q.cross(s) / den, q.cross(r) / den);
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then_some((t, u))
}

/// The parameter on `a` where its projection crosses `b`'s, by Newton's method on the two
/// parameters from `(ta, tb)`; `None` where it does not settle within both strokes.
fn pin_crossing(
    a: &Stroke,
    b: &Stroke,
    mut ta: f64,
    mut tb: f64,
    flat: &dyn Fn(Point3) -> Vec2,
) -> Option<f64> {
    let flat_d = |v: Vec3| flat(v + Point3::ZERO) - flat(Point3::ZERO);
    for _ in 0..50 {
        let (pa, da) = a.curve.d1(ta);
        let (pb, db) = b.curve.d1(tb);
        let r = flat(pa) - flat(pb);
        let (ja, jb) = (flat_d(da), -flat_d(db));
        let det = ja.cross(jb);
        if det.abs() <= 1e-300 {
            return None;
        }
        // Solve ja·x + jb·y = −r.
        let (x, y) = ((-r).cross(jb) / det, ja.cross(-r) / det);
        ta += x;
        tb += y;
        if x.abs() <= 1e-15 * (1.0 + ta.abs()) && y.abs() <= 1e-15 * (1.0 + tb.abs()) {
            break;
        }
    }
    let within = |s: &Stroke, t: f64| {
        let (lo, hi) = (s.t0.min(s.t1), s.t0.max(s.t1));
        t >= lo - 1e-12 && t <= hi + 1e-12
    };
    let gap = flat(a.curve.eval(ta)).dist(flat(b.curve.eval(tb)));
    (within(a, ta) && within(b, tb) && gap <= 1e-9 * (1.0 + flat(a.curve.eval(ta)).norm()))
        .then_some(ta)
}

/// The parameter of the point of stroke `s` nearest `p` seen along the view: the nearest sample,
/// refined by Newton's method on the projected distance.
fn foot(s: &Stroke, ts: &[f64], p: Point3, flat: &dyn Fn(Point3) -> Vec2) -> Option<f64> {
    let q = flat(p);
    let mut t = *ts.iter().min_by(|&&x, &&y| {
        flat(s.curve.eval(x))
            .dist(q)
            .total_cmp(&flat(s.curve.eval(y)).dist(q))
    })?;
    let flat_d = |v: Vec3| flat(v + Point3::ZERO) - flat(Point3::ZERO);
    let (lo, hi) = (s.t0.min(s.t1), s.t0.max(s.t1));
    for _ in 0..30 {
        let (c, d1) = s.curve.d1(t);
        let h = 1e-6 * (1.0 + t.abs());
        let d2 = (flat_d(s.curve.d1(t + h).1) - flat_d(s.curve.d1(t - h).1)) * (0.5 / h);
        let (r, g) = (flat(c) - q, flat_d(d1));
        let df = g.dot(g) + r.dot(d2);
        if df <= 0.0 {
            break;
        }
        let step = r.dot(g) / df;
        t = (t - step).clamp(lo, hi);
        if step.abs() <= 1e-15 * (1.0 + t.abs()) {
            break;
        }
    }
    Some(t)
}

/// Parameters from `t0` toward `t1`, `t1` left out, close enough that each chord's middle and
/// quarter points lie within half the tolerance of it; at least sixteen.
fn sample(
    at: &dyn Fn(f64) -> Point3,
    t0: f64,
    t1: f64,
    tolerance: f64,
    depth: u32,
    out: &mut Vec<f64>,
) {
    let (a, b) = (at(t0), at(t1));
    let off = |t: f64| {
        let p = at(t);
        match (b - a).unit() {
            Some(u) => (p - a).cross(u).norm(),
            None => p.dist(a),
        }
    };
    let worst = [0.25, 0.5, 0.75]
        .iter()
        .map(|&f| off(t0 + (t1 - t0) * f))
        .fold(0.0, f64::max);
    if depth >= 4 && (worst <= 0.5 * tolerance || depth >= 24) {
        out.push(t0);
        return;
    }
    let m = 0.5 * (t0 + t1);
    sample(at, t0, m, tolerance, depth + 1, out);
    sample(at, m, t1, tolerance, depth + 1, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive;

    fn lateral_face(body: &Brep) -> FaceId {
        body.face_ids()
            .find(|&f| {
                matches!(
                    body.face(f).surface,
                    Surface::Cylinder(_) | Surface::Cone(_)
                )
            })
            .unwrap()
    }

    fn lateral_outlines(body: &Brep, direction: Vec3) -> Vec<(Curve, f64, f64)> {
        outlines(body, lateral_face(body), direction)
    }

    fn assert_generators(outlines: &[(Curve, f64, f64)], angles: [f64; 2], bottom: f64, top: f64) {
        assert_eq!(outlines.len(), angles.len(), "{outlines:?}");
        for angle in angles {
            let radial = Frame::WORLD.radial(angle);
            let foot = radial * bottom;
            let tip = radial * top + Vec3::Z * 2.0;
            assert!(
                outlines.iter().any(|(curve, lo, hi)| {
                    let (a, b) = (curve.eval(*lo), curve.eval(*hi));
                    (a.dist(foot) <= tol::LINEAR && b.dist(tip) <= tol::LINEAR)
                        || (b.dist(foot) <= tol::LINEAR && a.dist(tip) <= tol::LINEAR)
                }),
                "missing generator at {angle}: {outlines:?}"
            );
        }
    }

    #[test]
    fn a_cylinder_draws_a_generator_at_an_interior_sample() {
        let body = primitive::cylinder(&Frame::WORLD, 5.0, 2.0, "post").unwrap();
        let angle = TAU * 120.0 / 720.0;
        let direction = Frame::WORLD.radial(angle).cross(Vec3::Z);
        assert_generators(
            &lateral_outlines(&body, direction),
            [angle, angle + TAU / 2.0],
            5.0,
            5.0,
        );
    }

    #[test]
    fn a_cone_draws_both_generators_at_adjacent_samples() {
        let body = primitive::cone(&Frame::WORLD, 2.0, 4.0, 2.0, "cone").unwrap();
        let step = TAU / 720.0;
        // Along this view the slope-one cone grazes at 0 and at step.
        let direction = Vec3::new(1.0, (step / 2.0).tan(), 1.0).unit().unwrap();
        assert_generators(&lateral_outlines(&body, direction), [0.0, step], 2.0, 4.0);
    }

    #[test]
    fn an_oblique_view_of_a_cone_draws_its_two_grazing_generators() {
        let body = primitive::cone(&Frame::WORLD, 2.0, 4.0, 2.0, "cone").unwrap();
        let angle = TAU / 6.0;
        assert_generators(
            &lateral_outlines(&body, Vec3::new(1.0, 0.0, 0.5).unit().unwrap()),
            [-angle, angle],
            2.0,
            4.0,
        );
    }

    #[test]
    fn a_view_along_a_cone_generator_draws_no_lateral_outline() {
        let body = primitive::cone(&Frame::WORLD, 2.0, 4.0, 2.0, "cone").unwrap();
        let direction = body.face(lateral_face(&body)).surface.d1(0.0, 0.0).2;
        assert!(lateral_outlines(&body, direction).is_empty());
    }

    #[test]
    fn an_axial_view_of_a_tilted_cylinder_draws_no_generators() {
        let frame = Frame::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0), Vec3::X).unwrap();
        let body = primitive::cylinder(&frame, 5.0, 2.0, "post").unwrap();
        assert!(lateral_outlines(&body, frame.z).is_empty());
    }
}
