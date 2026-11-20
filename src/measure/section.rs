//! Where a plane cuts a solid: the cut's outline and its area.

use crate::{
    boolean,
    boolean::half_space,
    error::{Error, Result},
    geom::{Curve, Surface},
    math::{Point3, Vec3},
    measure, tol,
    topo::{Brep, FaceId},
};

/// The part of `s` on the normal's side of the plane, and its faces lying on the plane: the cut.
fn caps(s: &Brep, origin: Point3, normal: Vec3) -> Result<(Brep, Vec<FaceId>)> {
    if s.solid_ids().next().is_none() {
        return Ok((s.clone(), Vec::new()));
    }
    let n = normal
        .unit()
        .ok_or_else(|| Error::InvalidInput("a plane needs a normal".into()))?;
    let cut = boolean::intersection(s, &half_space(s, origin, n)?)?;
    let caps = cut
        .face_ids()
        .filter(|&f| match &cut.face(f).surface {
            Surface::Plane(pl) => {
                pl.frame.z.cross(n).norm() <= 1e-9
                    && (pl.frame.origin - origin).dot(n).abs() <= 10.0 * tol::LINEAR
            }
            _ => false,
        })
        .collect();
    Ok((cut, caps))
}

/// Where the plane through `origin` square to `normal` cuts the boundary of `s`: each closed curve
/// as points along it, every chord within `tolerance` of the curve.
pub fn section(s: &Brep, origin: Point3, normal: Vec3, tolerance: f64) -> Result<Vec<Vec<Point3>>> {
    let (cut, caps) = caps(s, origin, normal)?;
    let mut out = Vec::new();
    for f in caps {
        for &l in &cut.face(f).loops {
            let mut pts: Vec<Point3> = Vec::new();
            for &c in &cut.lp(l).coedges {
                let co = cut.coedge(c);
                let edge = cut.edge(co.edge);
                let Some(curve) = &edge.curve else { continue };
                let (t0, t1) = edge.range;
                let mut run = Vec::new();
                chords(curve, t0, t1, tolerance, 0, &mut run);
                run.push(curve.eval(t1));
                if co.reversed {
                    run.reverse();
                }
                // Each run's last point is the next one's first.
                run.pop();
                pts.extend(run);
            }
            if !pts.is_empty() {
                out.push(pts);
            }
        }
    }
    Ok(out)
}

/// The area of the cut the plane through `origin` square to `normal` makes in `s`.
pub fn section_area(s: &Brep, origin: Point3, normal: Vec3) -> Result<f64> {
    let (cut, caps) = caps(s, origin, normal)?;
    Ok(caps.iter().map(|&f| measure::face_area(&cut, f)).sum())
}

/// Points along `c` from `t0` toward `t1`, `t1` left out, every chord within `tolerance` of the
/// curve: halved until each chord's middle and quarter points lie that close to it.
fn chords(c: &Curve, t0: f64, t1: f64, tolerance: f64, depth: u32, out: &mut Vec<Point3>) {
    let (a, b) = (c.eval(t0), c.eval(t1));
    let off = |t: f64| {
        let p = c.eval(t);
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
        out.push(a);
        return;
    }
    let m = 0.5 * (t0 + t1);
    chords(c, t0, m, tolerance, depth + 1, out);
    chords(c, m, t1, tolerance, depth + 1, out);
}

/// One piece of a section's outline: the curve it runs along, the stretch of it, and whether the
/// outline runs it backwards.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SectionEdge {
    pub curve: Curve,
    pub range: (f64, f64),
    pub reversed: bool,
}

/// Where the plane through `origin` square to `normal` cuts the boundary of `s`, as curves: each
/// closed outline as the pieces it is made of, in order. A circle the plane cuts from a cylinder
/// is a circle here, with its centre and radius.
pub fn section_curves(s: &Brep, origin: Point3, normal: Vec3) -> Result<Vec<Vec<SectionEdge>>> {
    let (cut, caps) = caps(s, origin, normal)?;
    let mut out = Vec::new();
    for f in caps {
        for &l in &cut.face(f).loops {
            let mut outline = Vec::new();
            for &c in &cut.lp(l).coedges {
                let co = cut.coedge(c);
                let edge = cut.edge(co.edge);
                if let Some(curve) = &edge.curve {
                    outline.push(SectionEdge {
                        curve: curve.clone(),
                        range: edge.range,
                        reversed: co.reversed,
                    });
                }
            }
            if !outline.is_empty() {
                out.push(outline);
            }
        }
    }
    Ok(out)
}
