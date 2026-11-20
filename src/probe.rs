//! Thin places in a valid solid, for design hygiene: faces narrower than a size, and material or
//! empty space thinner than one between faces that face each other. A valid solid can still hold
//! a 0.1 µm sliver or a hairline gap with real wall area when the design states two values a hair
//! apart; the checker passes them, since they are well above `LINEAR`, and this finds them.
//!
//! Widths are exact (area and perimeter integrated on the exact geometry). Thicknesses are found
//! by casting rays along the exact normal at each mesh triangle's centroid against the mesh, then
//! measured between the exact surfaces at the two ends.

use std::collections::BTreeMap;

use crate::{
    Result,
    math::{Aabb, Point3, Vec3},
    measure,
    mesh::{
        MeshOptions,
        bvh::{Bvh, ray_triangle},
        tessellate,
    },
    numeric,
    topo::{Brep, FaceId},
};

/// A face whose mean width is under the size asked for.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Narrow {
    pub face: FaceId,
    /// Twice the area over the perimeter, seams left out: a strip's width, a disc's radius.
    pub width: f64,
    pub area: f64,
    pub perimeter: f64,
    /// A point on the face's boundary, to find it by.
    pub at: Point3,
}

/// Faces that face each other closer than the size asked for: across empty space (a gap) or
/// across material (a sliver).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Thin {
    /// The face sampled, and the face its rays met.
    pub faces: (FaceId, FaceId),
    /// Material between them rather than empty space.
    pub material: bool,
    /// The least distance found, between the exact surfaces.
    pub min: f64,
    /// The area of the first face sampled as that thin.
    pub area: f64,
    /// Where the least distance was found, on the first face.
    pub at: Point3,
}

/// Faces closer to antiparallel than this cosine face each other: walls within about 11° of
/// parallel. Two faces meeting at a wider angle (an inside corner) are not a sliver however close
/// their points come by the edge.
const FACING: f64 = -0.98;

fn face_perimeter(b: &Brep, f: FaceId) -> (f64, Option<Point3>) {
    let mut total = 0.0;
    let mut at = None;
    let face = b.face(f);
    let mut uses: BTreeMap<usize, usize> = BTreeMap::new();
    for &l in &face.loops {
        for &c in &b.lp(l).coedges {
            *uses.entry(b.coedge(c).edge.index()).or_default() += 1;
        }
    }
    for &l in &face.loops {
        for &c in &b.lp(l).coedges {
            let e = b.edge(b.coedge(c).edge);
            // A seam is not a boundary: the face runs on across it.
            if uses[&b.coedge(c).edge.index()] > 1 {
                continue;
            }
            let Some(curve) = &e.curve else { continue };
            let (t0, t1) = e.range;
            let len = numeric::integrate(&mut |t| curve.d1(t).1.norm(), t0, t1, 1e-9);
            total += len;
            at.get_or_insert_with(|| curve.eval(0.5 * (t0 + t1)));
        }
    }
    (total, at)
}

/// Every face whose mean width (twice its area over its perimeter) is under `below` mm. For a
/// strip that is its width; no face is narrower everywhere than it, since a region all within
/// w / 2 of its boundary has area at most perimeter · w / 2.
pub fn narrow_faces(b: &Brep, below: f64) -> Vec<Narrow> {
    let mut out = Vec::new();
    for f in b.face_ids() {
        let (perimeter, at) = face_perimeter(b, f);
        let Some(at) = at else { continue };
        if perimeter <= 0.0 {
            continue;
        }
        let area = measure::face_area(b, f);
        let width = 2.0 * area / perimeter;
        if width < below {
            out.push(Narrow {
                face: f,
                width,
                area,
                perimeter,
                at,
            });
        }
    }
    out
}

/// The meshes tried, finest first: the mesher gives up on a very fine chord over a large part.
const MESHES: [MeshOptions; 3] = [
    MeshOptions {
        linear: 0.002,
        angular: 0.1,
    },
    MeshOptions {
        linear: 0.01,
        angular: 0.2,
    },
    MeshOptions {
        linear: 0.05,
        angular: 0.5,
    },
];

/// The point of face `f`'s surface nearest `p`, and the face's outward normal there.
fn on_face(b: &Brep, f: FaceId, p: Point3) -> (Point3, Vec3) {
    let face = b.face(f);
    let uv = face.surface.project(p);
    let n = face.surface.normal(uv.x, uv.y);
    let n = if face.reversed { -n } else { n };
    (face.surface.eval(uv.x, uv.y), n)
}

/// Pairs of faces that face each other across less than `below` mm, of material or of empty
/// space. Each mesh triangle's centroid is moved onto its face's exact surface and a ray cast both
/// ways along the exact normal; a crossing of another face's triangle that faces it is measured
/// again between the exact surfaces. The mesh's chord error widens the search, never the verdict.
pub fn thin_places(b: &Brep, below: f64) -> Result<Vec<Thin>> {
    let mut last = None;
    let mut mesh = None;
    for opt in MESHES {
        match tessellate(b, &opt) {
            Ok(m) => {
                mesh = Some((m, opt.linear));
                break;
            }
            Err(e) => last = Some(e),
        }
    }
    let Some((mesh, chord)) = mesh else {
        return Err(last.expect("a mesh was tried"));
    };
    let mut face_of = Vec::with_capacity(mesh.triangles.len());
    for (f, r) in &mesh.faces {
        face_of.resize(r.end.max(face_of.len()), *f);
    }
    let bvh = Bvh::new(&mesh);
    let reach = below + 2.0 * chord;
    let mut found: BTreeMap<(FaceId, FaceId, bool), Thin> = BTreeMap::new();
    let mut near = Vec::new();
    for (f, range) in &mesh.faces {
        for ti in range.clone() {
            let [i, j, k] = mesh.triangles[ti].map(|i| mesh.positions[i as usize]);
            let area = 0.5 * (j - i).cross(k - i).norm();
            if area <= 0.0 {
                continue;
            }
            let (p, n) = on_face(b, *f, (i + j + k) * (1.0 / 3.0));
            let mut q = Aabb::EMPTY;
            q.add(p + n * reach);
            q.add(p - n * reach);
            bvh.query(&q, &mut near);
            // The nearest facing triangle on either side: outside is empty space, inside material.
            let mut best: [Option<(f64, usize)>; 2] = [None, None];
            for &u in &near {
                if u == ti {
                    continue;
                }
                let [a, bb, c] = mesh.triangles[u].map(|i| mesh.positions[i as usize]);
                let Some(t) = ray_triangle(p, n, a, bb, c) else {
                    continue;
                };
                let tn = (bb - a).cross(c - a);
                let other = face_of[u];
                let (_, on) = on_face(b, other, p + n * t);
                if on.dot(n) > FACING || tn.norm() == 0.0 {
                    continue;
                }
                let side = usize::from(t < 0.0);
                if t.abs() <= reach && best[side].is_none_or(|(s, _)| t.abs() < s) {
                    best[side] = Some((t.abs(), u));
                }
            }
            for (side, hit) in best.into_iter().enumerate() {
                let Some((t, u)) = hit else { continue };
                let other = face_of[u];
                let dir = if side == 0 { n } else { -n };
                let (q, _) = on_face(b, other, p + dir * t);
                let gap = (q - p).norm();
                if gap >= below || gap <= crate::tol::LINEAR {
                    continue;
                }
                let e = found.entry((*f, other, side == 1)).or_insert(Thin {
                    faces: (*f, other),
                    material: side == 1,
                    min: gap,
                    area: 0.0,
                    at: p,
                });
                e.area += area;
                if gap < e.min {
                    e.min = gap;
                    e.at = p;
                }
            }
        }
    }
    // Each pair is found from both faces; keep the one sampled over more area.
    let mut out: Vec<Thin> = Vec::new();
    for t in found.into_values() {
        let (a, c) = t.faces;
        match out
            .iter_mut()
            .find(|o| o.faces == (c, a) && o.material == t.material)
        {
            Some(o) => {
                if t.area > o.area {
                    let min = o.min.min(t.min);
                    *o = t;
                    o.min = min;
                } else {
                    o.min = o.min.min(t.min);
                }
            }
            None => out.push(t),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boolean::{self, Op},
        math::{Frame, Transform, Vec2},
        profile::Profile,
        sweep,
    };

    fn block(x0: f64, x1: f64, y0: f64, y1: f64, z0: f64, h: f64) -> Brep {
        let p = Profile::polygon(
            &[
                Vec2::new(x0, y0),
                Vec2::new(x1, y0),
                Vec2::new(x1, y1),
                Vec2::new(x0, y1),
            ],
            &["a", "b", "c", "d"],
        );
        let mut b = sweep::extrude(&p, &Frame::WORLD, h, "f-block").unwrap();
        b.transform(&Transform::translation(Vec3::new(0.0, 0.0, z0)));
        b
    }

    /// A 10 × 10 × 5 block with a slot 1 µm wide cut 2 mm deep right across it, 3 mm in from
    /// x = 0: the slot's two walls face each other across 0.001 mm of air over 10 × 2 mm each,
    /// and its floor is a face 0.001 mm wide and 10 mm long, 2·A/P = 2 · 0.01 / 20.002 mm.
    #[test]
    fn a_hairline_slot_is_a_gap_and_its_floor_a_narrow_face() {
        let body = block(0.0, 10.0, 0.0, 10.0, 0.0, 5.0);
        let slot = block(3.0, 3.001, -1.0, 11.0, 3.0, 3.0);
        let cut = boolean::boolean(&body, &slot, Op::Difference).unwrap();
        let narrow = narrow_faces(&cut, 0.05);
        assert_eq!(narrow.len(), 1, "{narrow:?}");
        assert!(
            (narrow[0].width - 2.0 * 0.01 / 20.002).abs() < 1e-12,
            "{narrow:?}"
        );
        let thin = thin_places(&cut, 0.01).unwrap();
        assert_eq!(thin.len(), 1, "{thin:?}");
        assert!(!thin[0].material);
        assert!((thin[0].min - 0.001).abs() < 1e-9, "{thin:?}");
        assert!((thin[0].area - 20.0).abs() < 1e-6, "{thin:?}");
    }

    /// Two pockets 3.999 mm wide, 8 mm long and 2 mm deep leave a wall 2 µm thick standing
    /// between them: its sides face each other through 0.002 mm of material over 8 × 2 mm. Its top
    /// is a neck of the block's top face, which is wide elsewhere, so no face is narrow.
    #[test]
    fn a_wall_two_microns_thick_is_a_sliver() {
        let body = block(0.0, 10.0, 0.0, 10.0, 0.0, 5.0);
        let left = block(1.0, 4.999, 1.0, 9.0, 3.0, 3.0);
        let right = block(5.001, 9.0, 1.0, 9.0, 3.0, 3.0);
        let cut = boolean::boolean(&body, &left, Op::Difference).unwrap();
        let cut = boolean::boolean(&cut, &right, Op::Difference).unwrap();
        let thin = thin_places(&cut, 0.01).unwrap();
        assert_eq!(thin.len(), 1, "{thin:?}");
        assert!(thin[0].material);
        assert!((thin[0].min - 0.002).abs() < 1e-9, "{thin:?}");
        assert!((thin[0].area - 16.0).abs() < 1e-6, "{thin:?}");
        assert!(narrow_faces(&cut, 0.05).is_empty());
    }

    #[test]
    fn a_block_has_no_thin_places() {
        let b = block(0.0, 10.0, 0.0, 10.0, 0.0, 5.0);
        assert!(narrow_faces(&b, 0.05).is_empty());
        assert!(thin_places(&b, 0.01).unwrap().is_empty());
    }
}
