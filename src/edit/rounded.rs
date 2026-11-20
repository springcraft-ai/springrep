//! A solid shrunk by a distance, and a shell whose wall is every point within its thickness of the
//! faces it walls. Both move the faces and round the edges the distance turns about, where the
//! moved faces can be followed; otherwise they take the points near the faces, edges and corners
//! away as the union of exact pieces (Rossignac and Requicha 1986).

use std::collections::{BTreeSet, HashMap};

use super::{Wall, closed_beside, convex_apex, edge_faces, near, tweak};
use crate::{
    blend, boolean,
    error::{Error, Result},
    geom::Surface,
    math::Point3,
    pick::{concave_edges, convex_edges},
    tol,
    topo::{Brep, EdgeId, FaceId, FaceName},
};

/// Marks a face that a shell leaves open while its other faces move.
fn opening() -> FaceName {
    FaceName::new("\u{1}shell", "open")
}

/// Whether face `f` is a cone whose apex is a concave point: its outward normal leans in to the
/// axis, as at the bottom of a conical pit.
fn concave_apex(s: &Brep, f: FaceId) -> bool {
    matches!(s.face(f).surface, Surface::Cone(_)) && !convex_apex(s, f)
}

/// The points of `s` at least `d` from its outside: every face moved in by `d` and each concave
/// edge rounded at `d`, or, where that cannot be followed, `s` less the points within `d` of its
/// faces, concave edges and concave corners. A solid the distance splits keeps every piece.
pub(super) fn shrunk(s: &Brep, d: f64, feature: &str) -> Result<Brep> {
    moved_and_rounded(s, &[], -d, feature)
        .or_else(|e| near_inside(s, &[], d, feature).map_err(|_| e))
}

/// `s` with every face but those of `open` moved `by` along its outward normal, the faces of
/// `open` left where they are, and each edge between two moved faces that the move turns about
/// rounded at `|by|`: the convex edges when `by` is positive, the concave ones when negative. The
/// round about such an edge is the cylinder of radius `|by|` about the edge itself. The faces of
/// `open` carry [`opening`] besides their names. An error where a face beside an edge rounded
/// closes on the way, where a concave edge to round has a face of `open` beside it (the moved
/// face does not reach the round), or where a cone's apex is a point the move turns about.
fn moved_and_rounded(s: &Brep, open: &[FaceId], by: f64, feature: &str) -> Result<Brep> {
    let turned_about = |f: FaceId| {
        if by > 0.0 {
            convex_apex(s, f)
        } else {
            concave_apex(s, f)
        }
    };
    let apex = s.coedge_ids().any(|c| {
        let f = s.coedge_face(c);
        s.edge(s.coedge(c).edge).curve.is_none() && !open.contains(&f) && turned_about(f)
    });
    if apex {
        return Err(Error::Unsupported(
            "an offset of a cone's apex: the ball about it is not built".into(),
        ));
    }
    let turned = if by > 0.0 {
        convex_edges(s)?
    } else {
        concave_edges(s)?
    };
    let mut rounded = Vec::new();
    for &e in &turned {
        let Some((f, g)) = edge_faces(s, e) else {
            continue;
        };
        match (open.contains(&f), open.contains(&g)) {
            (false, false) => rounded.push(e),
            (true, true) => {}
            _ if by < 0.0 => {
                return Err(Error::Unsupported(
                    "a round about a concave edge beside an opening: the moved face does not \
                     reach it"
                        .into(),
                ));
            }
            // A convex edge on an opening is the rim, where the moved face meets the opening.
            _ => {}
        }
    }
    let mut marked = s.clone();
    for &f in open {
        let mut names = marked.face(f).names.clone();
        names.insert(opening());
        marked.set_face_names(f, names);
    }
    let moves: HashMap<FaceId, f64> = s
        .face_ids()
        .filter(|f| !open.contains(f))
        .map(|f| (f, by))
        .collect();
    let (moved, closed) = tweak::offset_faces_closing(&marked, &moves)?;
    closed_beside(
        s,
        &rounded,
        &closed,
        "an offset whose faces close past an edge it rounds",
    )?;
    if rounded.is_empty() {
        return Ok(moved);
    }
    let on_opening = |b: &Brep, e: EdgeId| {
        b.edge_uses(e)
            .iter()
            .any(|&c| b.face(b.coedge_face(c)).names.contains(&opening()))
    };
    let edges: Vec<EdgeId> = if by > 0.0 {
        convex_edges(&moved)?
    } else {
        concave_edges(&moved)?
    }
    .into_iter()
    .filter(|&e| !on_opening(&moved, e))
    .collect();
    if edges.is_empty() {
        return Ok(moved);
    }
    blend::fillet_edges_mitred(&moved, &edges, by.abs(), feature)
}

/// `s` less the points within `t` of its faces other than `open`: the union of those faces' slabs
/// `t` in, a tube about each concave edge beside one of them and a ball at each corner of those
/// edges. Each piece holds only points within `t` of a face, edge or vertex of those faces, and a
/// point inside `s` within `t` of them has its foot inside a face, on a concave edge or at a
/// concave corner, so the union takes away exactly those points. A solid the distance splits
/// keeps every piece.
pub(super) fn near_inside(s: &Brep, open: &[FaceId], t: f64, feature: &str) -> Result<Brep> {
    let kept = |f: FaceId| !open.contains(&f);
    let concave: Vec<EdgeId> = concave_edges(s)?
        .into_iter()
        .filter(|&e| edge_faces(s, e).is_some_and(|(f, g)| kept(f) || kept(g)))
        .collect();
    let mut parts = Vec::new();
    for f in s.face_ids().filter(|&f| kept(f)) {
        parts.push(near::slab(s, f, -t, feature)?);
    }
    for &e in &concave {
        parts.push(near::tube(s, e, t, feature)?);
    }
    for v in near::corners(s, &concave, |f| kept(f) && concave_apex(s, f)) {
        parts.push(near::ball(s.vertex(v).point, t, feature)?);
    }
    let refs: Vec<&Brep> = parts.iter().collect();
    boolean::difference(s, &boolean::union_all(&refs)?)
}

/// An error unless each face of `open` meets the faces beside it at a right angle or less, where
/// the slabs of those faces reach the opening as the moved faces would.
pub(super) fn square_openings(s: &Brep, open: &[FaceId]) -> Result<()> {
    let outward = |f: FaceId, p: Point3| {
        let sf = &s.face(f).surface;
        let uv = sf.project(p);
        let n = sf.normal(uv.x, uv.y);
        if s.face(f).reversed { -n } else { n }
    };
    let convex = convex_edges(s)?;
    for e in s.edge_ids() {
        let Some((f, g)) = edge_faces(s, e) else {
            continue;
        };
        if open.contains(&f) == open.contains(&g) {
            continue;
        }
        let x = s.edge(e);
        let Some(c) = &x.curve else { continue };
        let p = c.eval(0.5 * (x.range.0 + x.range.1));
        if !convex.contains(&e) || outward(f, p).dot(outward(g, p)) > tol::ANGULAR {
            return Err(Error::Unsupported(
                "a shell open at a face that meets a wall at more than a right angle: the points \
                 near the wall reach past the moved wall there"
                    .into(),
            ));
        }
    }
    Ok(())
}

/// `s` hollowed to a wall `thickness` thick, open through the faces of `open`, the wall every
/// point within `thickness` of the faces of `s` other than `open` on the `wall` side. Where that
/// differs from [`shell`](super::shell), it is round: an inward wall's inside is the cylinder of
/// radius `thickness` about each concave edge, and an outward wall's outside the cylinder about
/// each convex edge and the ball about each corner. Where a wall meets an opening, the moved face
/// runs on to the opening's surface, which is the rim. The cavity's faces are `<feature>.inner`;
/// an inward wall leaves every face of `s` in place with its names, and an outward wall's outside
/// carries no name while each face of `s` keeps its names beside `inner`.
pub fn shell_rounded(
    s: &Brep,
    open: &[FaceId],
    thickness: f64,
    wall: Wall,
    feature: &str,
) -> Result<Brep> {
    if thickness.is_nan() || thickness <= tol::LINEAR {
        return Err(Error::InvalidInput(
            "a shell's wall must be thicker than the tolerance".into(),
        ));
    }
    let inner: BTreeSet<FaceName> =
        std::iter::once(FaceName::new(feature, FaceName::INNER)).collect();
    match wall {
        Wall::Inward => {
            let mut cavity = moved_and_rounded(s, open, -thickness, feature).or_else(|e| {
                square_openings(s, open)
                    .and_then(|()| near_inside(s, open, thickness, feature))
                    .map_err(|_| e)
            })?;
            for f in cavity.face_ids().collect::<Vec<_>>() {
                cavity.set_face_names(f, inner.clone());
            }
            boolean::difference(s, &cavity)
        }
        Wall::Outward => {
            let mut outer = moved_and_rounded(s, open, thickness, feature)?;
            for f in outer.face_ids().collect::<Vec<_>>() {
                let mut names = outer.face(f).names.clone();
                if !names.remove(&opening()) {
                    names.clear();
                }
                outer.set_face_names(f, names);
            }
            let mut cavity = s.clone();
            for f in cavity.face_ids().collect::<Vec<_>>() {
                let mut names = cavity.face(f).names.clone();
                names.extend(inner.iter().cloned());
                cavity.set_face_names(f, names);
            }
            boolean::difference(&outer, &cavity)
        }
    }
}

#[cfg(test)]
mod tests;
