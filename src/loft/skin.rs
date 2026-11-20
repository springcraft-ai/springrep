//! The faces of one lofted loop, built from its matched rows: each row a section (or the apex)
//! cut into as many pieces as the loop has columns, a piece either an edge of the section or one
//! of its vertices standing for a whole edge of another row. A column's face between two rows is
//! a plane or the surface its band gives; where its piece on a row is a vertex, the face meets that
//! row at a point: a triangle when flat, a pole of its surface otherwise.

use std::collections::HashMap;

use super::{Piece, rail::Rail};
use crate::{
    Error, Result,
    geom::{Curve, Line, NurbsCurve, Plane, Surface},
    math::{Frame, Point3, Vec2},
    profile::Names,
    tol,
    topo::{Brep, CoedgeSpec, EdgeId, FaceId, FaceName, PCurve, VertexId},
};

/// The side faces between rows `lo` and `hi`.
pub(super) struct Band {
    pub lo: usize,
    pub hi: usize,
    /// The edge at each column's first vertex, from row `lo` to row `hi`, on `[0, 1]`.
    pub sides: Vec<Rail>,
    /// Each column's surface, `u` along its pieces and `v` from `lo` to `hi`; `None` where the
    /// face is flat and its plane is found from its corners.
    pub surfaces: Vec<Option<Surface>>,
}

/// How an edge's parameter maps onto a side face's `u` or `v`: `scale · t`.
#[derive(Clone, Copy)]
struct EdgeParam {
    id: EdgeId,
    scale: f64,
}

/// One edge of a cap's loop, from `start` to `end` along the loop.
pub(super) struct CapEdge {
    pub id: EdgeId,
    pub start: Point3,
    pub end: Point3,
}

/// The faces of a loop and the edges of its first and last rows, in loop order.
pub(super) struct Sides {
    pub faces: Vec<FaceId>,
    pub first: Vec<CapEdge>,
    pub last: Vec<CapEdge>,
}

fn coedge(edge: EdgeId, reversed: bool, pcurve: PCurve) -> CoedgeSpec {
    CoedgeSpec {
        edge,
        reversed,
        pcurve,
    }
}

fn affine(origin: Vec2, dir: Vec2) -> PCurve {
    PCurve::Affine { origin, dir }
}

pub(super) fn projected(frame: &Frame, p: Point3) -> PCurve {
    let l = frame.to_local(p);
    PCurve::Projected {
        seed: Vec2::new(l.x, l.y),
    }
}

pub(super) fn nurbs_curve(r: &Rail) -> Result<NurbsCurve> {
    if r.pw.iter().any(|h| h[3].is_nan() || h[3] <= 0.0) {
        return Err(Error::Numerical(
            "a loft edge came out with a weight that is not positive".into(),
        ));
    }
    let (pts, w) = super::split_hom(&r.pw);
    NurbsCurve::new(r.degree, r.knots.clone(), pts, w)
}

/// An edge along `r` from `a` to `b`: a line when `r` is one straight span.
fn edge_of(b: &mut Brep, r: &Rail, straight: bool, a: VertexId, z: VertexId) -> Result<EdgeParam> {
    if straight || (r.degree == 1 && r.pw.len() == 2) {
        let (line, len) = Line::through(r.start(), r.end())
            .ok_or_else(|| Error::InvalidInput("a loft has an edge of zero length".into()))?;
        Ok(EdgeParam {
            id: b.add_edge(Curve::Line(line), (0.0, len), a, z),
            scale: 1.0 / len,
        })
    } else {
        Ok(EdgeParam {
            id: b.add_edge(Curve::Nurbs(Box::new(nurbs_curve(r)?)), (0.0, 1.0), a, z),
            scale: 1.0,
        })
    }
}

/// The plane through a side face's corners, `x` along the loop, `z` outward, when they are
/// coplanar: `p` the start and end of its piece on row `lo`, `q` on row `hi`.
pub(super) fn flat_side(p: [Point3; 2], q: [Point3; 2]) -> Option<Frame> {
    let x = (p[1] - p[0]).unit().or_else(|| (q[1] - q[0]).unit())?;
    let w = (q[0] + q[1] - p[0] - p[1]) * 0.5;
    let y = (w - x * w.dot(x)).unit()?;
    let z = x.cross(y);
    let off = [p[1], q[0], q[1]]
        .iter()
        .map(|&c| (c - p[0]).dot(z).abs())
        .fold(0.0, f64::max);
    (off <= tol::LINEAR).then_some(Frame {
        origin: p[0],
        x,
        y,
        z,
    })
}

/// A side face's names: `side`, and the names of the first section's edge it grows from.
fn side_names(feature: &str, edge: &Names) -> Vec<FaceName> {
    let mut names = vec![FaceName::new(feature, FaceName::SIDE)];
    names.extend(FaceName::all(
        feature,
        edge.iter().filter(|n| !n.is_empty()),
    ));
    names
}

/// Each column's first vertex on a row: a piece that is a point shares its one vertex with the
/// pieces either side of it.
fn row_vertices(b: &mut Brep, pieces: &[Piece]) -> Vec<VertexId> {
    let k = pieces.len();
    let Some(j0) = (0..k).find(|&j| !pieces[(j + k - 1) % k].point) else {
        let v = b.add_vertex(pieces[0].at(0.0));
        return vec![v; k];
    };
    let mut out = vec![None; k];
    let mut prev: Option<VertexId> = None;
    for i in 0..k {
        let j = (j0 + i) % k;
        let v = match prev {
            Some(v) if pieces[(j + k - 1) % k].point => v,
            _ => b.add_vertex(pieces[j].at(0.0)),
        };
        out[j] = Some(v);
        prev = Some(v);
    }
    out.into_iter()
        .map(|v| v.expect("every column has a vertex"))
        .collect()
}

/// Adds the side faces of one loop: `pieces[r][j]` is column `j`'s piece on row `r`, and
/// `columns[j][r]` its rail, compatible down the column.
pub(super) fn add_sides(
    b: &mut Brep,
    pieces: &[Vec<Piece>],
    columns: &[Vec<Rail>],
    bands: &[Band],
    first_names: &[Names],
    feature: &str,
) -> Result<Sides> {
    let k = columns.len();
    let mut rows: Vec<usize> = bands.iter().flat_map(|bd| [bd.lo, bd.hi]).collect();
    rows.sort_unstable();
    rows.dedup();
    let mut verts: HashMap<usize, Vec<VertexId>> = HashMap::new();
    let mut sec_edges: HashMap<usize, Vec<Option<EdgeParam>>> = HashMap::new();
    for &r in &rows {
        let v = row_vertices(b, &pieces[r]);
        let mut edges = Vec::with_capacity(k);
        for j in 0..k {
            let p = &pieces[r][j];
            edges.push(if p.point {
                None
            } else {
                Some(edge_of(
                    b,
                    &columns[j][r],
                    p.straight,
                    v[j],
                    v[(j + 1) % k],
                )?)
            });
        }
        verts.insert(r, v);
        sec_edges.insert(r, edges);
    }

    let mut faces = Vec::new();
    for band in bands {
        let (lo, hi) = (band.lo, band.hi);
        let (vl, vh) = (&verts[&lo], &verts[&hi]);
        let mut by_ends: HashMap<(VertexId, VertexId), EdgeParam> = HashMap::new();
        let mut sides: Vec<EdgeParam> = Vec::with_capacity(k);
        for j in 0..k {
            let key = (vl[j], vh[j]);
            if vl[j] == vh[j] {
                return Err(Error::Unsupported(
                    "a loft whose sections share a vertex".into(),
                ));
            }
            let ep = match by_ends.get(&key) {
                Some(&ep) => ep,
                None => {
                    let ep = edge_of(b, &band.sides[j], false, vl[j], vh[j])?;
                    by_ends.insert(key, ep);
                    ep
                }
            };
            sides.push(ep);
        }
        for j in 0..k {
            let (pl, ph) = (&pieces[lo][j], &pieces[hi][j]);
            if pl.point && ph.point {
                continue;
            }
            let (bottom, top) = (sec_edges[&lo][j], sec_edges[&hi][j]);
            let (left, right) = (sides[j], sides[(j + 1) % k]);
            let (rl, rh) = (&columns[j][lo], &columns[j][hi]);
            let names = side_names(feature, &first_names[j]);
            let face = match &band.surfaces[j] {
                None => {
                    let fr = flat_side([rl.start(), rl.end()], [rh.start(), rh.end()])
                        .ok_or_else(|| Error::Numerical("a flat loft side is not flat".into()))?;
                    let face = b.add_face(Surface::Plane(Plane { frame: fr }), false, names);
                    let mut lp = Vec::with_capacity(4);
                    if let Some(e) = bottom {
                        lp.push(coedge(e.id, false, projected(&fr, rl.start())));
                    }
                    lp.push(coedge(right.id, false, projected(&fr, rl.end())));
                    if let Some(e) = top {
                        lp.push(coedge(e.id, true, projected(&fr, rh.end())));
                    }
                    lp.push(coedge(left.id, true, projected(&fr, rh.start())));
                    b.add_loop(face, lp);
                    face
                }
                Some(surface) => {
                    let face = b.add_face(surface.clone(), false, names);
                    let bottom = match bottom {
                        Some(e) => coedge(e.id, false, affine(Vec2::ZERO, Vec2::new(e.scale, 0.0))),
                        None => {
                            let pole = b.add_degenerate_edge(vl[j], (0.0, 1.0));
                            coedge(pole, false, affine(Vec2::ZERO, Vec2::new(1.0, 0.0)))
                        }
                    };
                    let top = match top {
                        Some(e) => coedge(
                            e.id,
                            true,
                            affine(Vec2::new(0.0, 1.0), Vec2::new(e.scale, 0.0)),
                        ),
                        None => {
                            let pole = b.add_degenerate_edge(vh[j], (0.0, 1.0));
                            coedge(pole, true, affine(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)))
                        }
                    };
                    b.add_loop(
                        face,
                        vec![
                            bottom,
                            coedge(
                                right.id,
                                false,
                                affine(Vec2::new(1.0, 0.0), Vec2::new(0.0, right.scale)),
                            ),
                            top,
                            coedge(
                                left.id,
                                true,
                                affine(Vec2::ZERO, Vec2::new(0.0, left.scale)),
                            ),
                        ],
                    );
                    face
                }
            };
            faces.push(face);
        }
    }
    let cap = |r: usize| -> Vec<CapEdge> {
        (0..k)
            .filter_map(|j| {
                sec_edges[&r][j].map(|e| CapEdge {
                    id: e.id,
                    start: columns[j][r].start(),
                    end: columns[j][r].end(),
                })
            })
            .collect()
    };
    let (first, last) = (rows[0], rows[rows.len() - 1]);
    Ok(Sides {
        faces,
        first: cap(first),
        last: cap(last),
    })
}
