//! Same-domain face merging on a finished solid: faces on
//! one surface, facing one way, that share an edge become one face carrying every name the merged
//! faces carried. Booleans already merge the faces they keep; this is for solids no boolean made,
//! such as an imported file's faces.

use std::rc::Rc;

use super::{
    coedge_uv_at,
    domain::{Trim, UvMap},
    sew::{self, OutFace, PieceGeom},
    solids_of,
};
use crate::{
    Result,
    check::coedge_params,
    math::{Point3, Vec2},
    topo::{Brep, CoedgeId, PCurve, SolidId},
};

/// `b` with its same-domain neighbours merged, or a copy of `b` when no two faces merge. The
/// result passes the checker or is an error. Faces merge only within a solid, so the solids of
/// several are unified one by one: they may touch, as an import's parts seated on a board do.
pub fn unify(b: &Brep) -> Result<Brep> {
    let solids: Vec<SolidId> = b.solid_ids().collect();
    if solids.len() < 2 {
        return Ok(unify_solid(b)?.unwrap_or_else(|| b.clone()));
    }
    let mut out = Brep::new();
    let mut merged = false;
    for s in solids {
        let one = solids_of(b, &[s]);
        match unify_solid(&one)? {
            Some(u) => {
                merged = true;
                out.append(&u);
            }
            None => {
                out.append(&one);
            }
        }
    }
    Ok(if merged { out } else { b.clone() })
}

/// One solid with its same-domain neighbours merged; `None` where no two faces merge. Faces merge
/// on exact surfaces, so an imported solid's local tolerances only follow the edges and vertices
/// that remain.
fn unify_solid(b: &Brep) -> Result<Option<Brep>> {
    let pool: Vec<Point3> = b.vertex_ids().map(|v| b.vertex(v).point).collect();
    let vtol: Vec<f64> = b.vertex_ids().map(|v| b.vertex_tolerance(v)).collect();
    let curves: Vec<_> = b
        .edge_ids()
        .map(|e| b.edge(e).curve.clone().map(Rc::new))
        .collect();
    let pieces: Vec<PieceGeom> = b
        .edge_ids()
        .map(|e| {
            let edge = b.edge(e);
            PieceGeom {
                curve: curves[e.index()].clone(),
                range: edge.range,
                v0: edge.start.index(),
                v1: edge.end.index(),
                tol: b.edge_tolerance(e),
            }
        })
        .collect();
    let mut faces: Vec<OutFace> = Vec::with_capacity(b.face_count());
    for s in b.solid_ids() {
        for f in b.solid_faces(s) {
            let face = b.face(f);
            let loops = face
                .loops
                .iter()
                .map(|&lp| {
                    b.lp(lp)
                        .coedges
                        .iter()
                        .map(|&c| trim(b, c, &curves, &pool))
                        .collect()
                })
                .collect();
            faces.push(OutFace {
                surface: face.surface.clone(),
                reversed: face.reversed,
                names: face.names.clone(),
                loops,
                memo: Some(b.shared_memo(f)),
                side: Some(0),
            });
        }
    }
    let n = faces.len();
    let partners = sew::pair_uses(&faces, &pieces)?;
    let merged = sew::merge_same_domain(faces, &pieces, &vtol, &partners);
    if merged.len() == n {
        return Ok(None);
    }
    sew::assemble(merged, &pool, &vtol, &pieces).map(Some)
}

/// A coedge as a whole trim of its own edge, the edge's index its piece.
fn trim(b: &Brep, c: CoedgeId, curves: &[Option<Rc<crate::geom::Curve>>], pool: &[Point3]) -> Trim {
    let co = b.coedge(c);
    let edge = b.edge(co.edge);
    let ts = coedge_params(b, c, 16);
    let uvs = coedge_uv_at(b, c, &ts);
    let map = match (&co.pcurve, &edge.curve) {
        (PCurve::Projected { .. }, None) => UvMap::Affine {
            origin: uvs[0],
            dir: Vec2::ZERO,
        },
        (pcurve, _) => UvMap::of(pcurve),
    };
    Trim {
        piece: co.edge.index(),
        forward: !co.reversed,
        curve: curves[co.edge.index()].clone(),
        pole: pool[edge.start.index()],
        t0: ts[0],
        t1: ts[ts.len() - 1],
        map,
        uv0: uvs[0],
        uv1: uvs[uvs.len() - 1],
        v0: b.coedge_start(c).index(),
        v1: b.coedge_end(c).index(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check,
        math::Frame,
        measure,
        profile::{NamedSegment, Profile, ProfileLoop, Segment},
        sweep,
    };

    fn line(a: (f64, f64), b: (f64, f64), name: &str) -> NamedSegment {
        NamedSegment::new(
            Segment::Line {
                a: Vec2::new(a.0, a.1),
                b: Vec2::new(b.0, b.1),
            },
            name,
        )
    }

    #[test]
    fn coplanar_sides_merge_and_keep_both_names() {
        // The bottom edge in two collinear pieces sweeps two faces on one plane.
        let p = Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    line((0.0, 0.0), (4.0, 0.0), "left"),
                    line((4.0, 0.0), (10.0, 0.0), "right"),
                    line((10.0, 0.0), (10.0, 5.0), "u_max"),
                    line((10.0, 5.0), (0.0, 5.0), "v_max"),
                    line((0.0, 5.0), (0.0, 0.0), "u_min"),
                ],
            }],
        };
        let b = sweep::extrude(&p, &Frame::WORLD, 3.0, "f-x").unwrap();
        assert_eq!(b.face_count(), 7);
        let u = unify(&b).unwrap();
        assert!(check::check(&u).is_empty());
        assert_eq!(u.face_count(), 6);
        assert!((measure::volume(&u) - 150.0).abs() < 1e-9);
        let both = u.face_ids().filter(|&f| u.face(f).names.len() == 2).count();
        assert_eq!(both, 1);
        // Nothing is left to merge, so the solid comes back as it was.
        assert_eq!(unify(&u).unwrap(), u);
    }
}
