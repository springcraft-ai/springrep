//! Four convex fillets at a corner of four planes that no one ball touches (a pyramid's apex on a
//! rectangle): the union of the balls inside the body (Rossignac and Requicha 1984) splits the
//! corner. Two opposite planes' lines `r` in meet the other two's in two points, each where three
//! of the planes `r` in meet; a ball there touches those three, and between the two balls one
//! rolls along the line where the two opposite planes `r` in meet, touching both. So the corner is
//! two sphere corners joined by a round between the two opposite planes, along an edge the body
//! does not have: the line where those planes meet, through the corner.

use std::collections::BTreeSet;

use super::{
    Ball, End, Kind, Path, Picked, Section, Trace, end_index, faces_at, leaving, outline, outward,
    piece, renamed, solve3, unit2,
};
use crate::{
    Error, Result,
    geom::Surface,
    math::{Frame, Point3, Vec2, Vec3},
    tol,
    topo::{Brep, EdgeId, FaceId, FaceName, VertexId},
};

/// The two sphere corners a split corner closes with, each with the picked edges that end on it,
/// and the round between them, cut as a tool of its own.
pub(super) struct Split {
    pub balls: [(Vec<usize>, Ball); 2],
    pub ridge: Brep,
}

/// The split corner at `v`, where the picked edges `at` meet, or None where `v` is not a convex
/// corner of four planes, all four edges filleted, that splits.
pub(super) fn find(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    at: &[usize],
    v: VertexId,
    radius: f64,
    feature: &str,
) -> Result<Option<Split>> {
    let faces = faces_at(body, v);
    let edges_at: BTreeSet<EdgeId> = body
        .coedge_ids()
        .filter(|&c| body.coedge_start(c) == v)
        .map(|c| body.coedge(c).edge)
        .collect();
    let picked_at: BTreeSet<EdgeId> = at.iter().map(|&i| picked[i].edge).collect();
    if at.len() != 4 || faces.len() != 4 || edges_at != picked_at {
        return Ok(None);
    }
    if faces
        .iter()
        .any(|&f| !matches!(body.face(f).surface, Surface::Plane(_)))
        || at
            .iter()
            .any(|&i| !sections[i].convex || !matches!(sections[i].path, Path::Prism { .. }))
    {
        return Ok(None);
    }
    // The faces in order round the corner, each picked edge between one and the next.
    let mut order: Vec<FaceId> = vec![picked[at[0]].faces[0]];
    while order.len() < 4 {
        let last = *order.last().expect("one face to start");
        let next = at.iter().find_map(|&i| {
            let [a, b] = picked[i].faces;
            let other = if a == last {
                b
            } else if b == last {
                a
            } else {
                return None;
            };
            (!order.contains(&other)).then_some(other)
        });
        let Some(next) = next else {
            return Ok(None);
        };
        order.push(next);
    }
    let corner = body.vertex(v).point;
    let n: Vec<Vec3> = order.iter().map(|&f| outward(body, f, corner)).collect();
    // The point `radius` inside each of three planes.
    let centre = |k: [usize; 3]| -> Option<Point3> {
        let x = solve3(
            [
                [n[k[0]].x, n[k[0]].y, n[k[0]].z],
                [n[k[1]].x, n[k[1]].y, n[k[1]].z],
                [n[k[2]].x, n[k[2]].y, n[k[2]].z],
            ],
            [-radius; 3],
        )?;
        Some(corner + Vec3::new(x[0], x[1], x[2]))
    };
    let inside = |c: Point3, k: usize| n[k].dot(c - corner) < -radius - tol::LINEAR;
    // The two opposite faces `a` and `a + 2` the round runs between, the one way that leaves each
    // ball clear of the fourth plane.
    let mut found = None;
    for a in 0..2 {
        let (b, c, d) = (a + 1, a + 2, (a + 3) % 4);
        let (Some(c1), Some(c2)) = (centre([a, b, c]), centre([a, c, d])) else {
            continue;
        };
        if inside(c1, d) && inside(c2, b) && c1.dist(c2) > tol::LINEAR {
            found = Some((a, c1, c2));
        }
    }
    let Some((a, c1, c2)) = found else {
        return Ok(None);
    };
    let c = a + 2;
    let along = (c2 - c1)
        .unit()
        .ok_or_else(|| Error::Numerical("a ridge of no length".into()))?;
    // The picked edge between faces `p` and `q`, and the direction it leaves the corner in.
    let edge_between = |p: usize, q: usize| -> Result<(usize, Vec3)> {
        let i = *at
            .iter()
            .find(|&&i| {
                let fs = picked[i].faces;
                fs.contains(&order[p]) && fs.contains(&order[q])
            })
            .ok_or_else(|| Error::Numerical("a corner face pair with no edge".into()))?;
        let d = leaving(body, picked[i].edge, v)
            .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
        Ok((i, d))
    };
    let mut balls = Vec::with_capacity(2);
    for (centre, rest, toward) in [(c1, a + 1, along), (c2, (a + 3) % 4, -along)] {
        let (i, di) = edge_between(a, rest)?;
        let (j, dj) = edge_between(rest, c)?;
        let ball = Ball {
            centre,
            radius,
            feet: vec![n[a], n[rest], n[c]],
            convex: true,
            corner,
            edges: vec![di, dj, toward],
            whole: None,
        };
        let m = ball.middle();
        if ball.feet.iter().any(|f| f.dot(m) <= tol::TANGENT) {
            return Ok(None);
        }
        balls.push((vec![i, j], ball));
    }
    let ridge = ridge_tool(n[a], n[c], corner, c1, c2, radius, feature)?;
    let [b0, b1]: [(Vec<usize>, Ball); 2] = balls
        .try_into()
        .map_err(|_| Error::Numerical("a split corner without two balls".to_string()))?;
    Ok(Some(Split {
        balls: [b0, b1],
        ridge,
    }))
}

/// Where each picked edge at a split corner stops: on the section through its own ball's centre.
pub(super) fn ends(
    body: &Brep,
    picked: &[Picked],
    split: &Split,
    v: VertexId,
) -> Result<Vec<(usize, usize, End)>> {
    let corner = body.vertex(v).point;
    let mut out = Vec::new();
    for (ids, ball) in &split.balls {
        for &i in ids {
            let along = leaving(body, picked[i].edge, v)
                .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
            let short = (ball.centre - corner).dot(along);
            if short <= tol::LINEAR {
                return Err(Error::Unsupported(
                    "fillets meeting at a corner their ball does not stand back from".into(),
                ));
            }
            out.push((i, end_index(body, picked[i].edge, v), End::Short(short)));
        }
    }
    Ok(out)
}

/// The round between the planes of outward normals `na` and `nc`, which meet along the line
/// through `corner`, swept from the section through `c1` to the one through `c2`: the region the
/// ball rolling from one corner ball to the other takes off, its round face a corner face.
fn ridge_tool(
    na: Vec3,
    nc: Vec3,
    corner: Point3,
    c1: Point3,
    c2: Point3,
    radius: f64,
    feature: &str,
) -> Result<Brep> {
    let along = (c2 - c1)
        .unit()
        .ok_or_else(|| Error::Numerical("a ridge of no length".into()))?;
    let start = corner + along * (c1 - corner).dot(along);
    let length = (c2 - c1).dot(along);
    let frame = Frame::new(start, along, along.any_perpendicular())?;
    let to2 = |w: Vec3| Vec2::new(w.dot(frame.x), w.dot(frame.y));
    let mut m = [Vec2::ZERO; 2];
    let mut n = [Vec2::ZERO; 2];
    for (k, (own, other)) in [(na, nc), (nc, na)].into_iter().enumerate() {
        let dir = own.cross(along);
        let dir = if dir.dot(other) < 0.0 { dir } else { -dir };
        m[k] = unit2(to2(dir))
            .ok_or_else(|| Error::Numerical("a ridge plane along its own line".into()))?;
        n[k] = unit2(to2(own)).ok_or_else(|| Error::Numerical("a ridge plane".into()))?;
    }
    let section = Section {
        path: Path::Prism { frame, length },
        p: Vec2::ZERO,
        m,
        n,
        trace: [Trace::Line; 2],
        convex: true,
    };
    let o = outline(&section, Kind::Fillet { radius })?;
    let tool = piece(&section, &o, [End::Exact; 2], feature)?;
    let corner_name = BTreeSet::from([FaceName::new(feature, "corner")]);
    Ok(renamed(tool, "blend", &corner_name))
}
