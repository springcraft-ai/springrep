//! Where a ray crosses a solid's faces, from the exact geometry.
//!
//! [`Rays`] meshes the arena once. A ray's candidate faces are those with a mesh triangle within
//! twice the mesh deflection of it: a face's exact surface lies within the deflection of its mesh,
//! so no crossing is missed. On each, over the stretches of the ray that pass those triangles, the
//! line is intersected with the face's surface exactly, in closed form on a plane, cylinder, cone,
//! sphere and torus. A crossing counts where it lies inside the face and its `t` in the ray's
//! range. One crossing on an edge between two faces counts once, as does a ray that touches a
//! surface without crossing it.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{
    boolean::{
        classify::DIRECTIONS,
        csi::curve_surface,
        domain::{Domain, Where},
        face_domain,
    },
    error::{Error, Result},
    geom::{Curve, Line},
    math::{Aabb, Point3, Vec3},
    mesh::{
        self,
        bvh::{Bvh, segment_triangle},
    },
    tol,
    topo::{Brep, FaceId},
};

/// Where a ray crosses a face: `t` along its unit direction, the point, the face and the face's
/// outward normal there. `on_edge` where the point lies on the face's boundary: the ray passes
/// through an edge there, and the face next to it has a crossing at the same point.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Hit {
    pub t: f64,
    pub point: Point3,
    pub face: FaceId,
    pub normal: Vec3,
    pub on_edge: bool,
}

/// How far past the ends of its range a crossing may lie and still count, and how near the
/// surface a ray stays over a stretch it only grazes.
const ON: f64 = 0.01 * tol::LINEAR;

/// Where a point lies against an arena's solids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Inside,
    /// On the boundary, within the linear tolerance.
    On,
    Outside,
}

/// An arena prepared for casting rays at its faces: its mesh, a box tree over it, and each face's
/// trimmed domain on its surface.
pub struct Rays<'a> {
    b: &'a Brep,
    seeds: Arc<Seeds>,
    domains: Vec<Domain>,
}

/// Where a ray's search starts: the arena's seed mesh, triangle by triangle.
#[derive(Debug)]
pub(crate) struct Seeds {
    tris: Vec<(FaceId, [Point3; 3])>,
    tree: Bvh,
    deflection: f64,
    /// The mesh's box: the arena lies within the deflection of it.
    reach: Aabb,
}

impl Seeds {
    fn new(b: &Brep) -> Result<Arc<Seeds>> {
        let seed = mesh::seed_mesh(b)?;
        let m = &seed.mesh;
        let tree = Bvh::new(m);
        let mut face_of = vec![None; m.triangles.len()];
        for (f, range) in &m.faces {
            for k in range.clone() {
                face_of[k] = Some(*f);
            }
        }
        let tris = m
            .triangles
            .iter()
            .zip(face_of)
            // The tree names triangles by index, so each keeps its place.
            .map(|(t, f)| {
                let f = f.expect("every mesh triangle lies in its face's range");
                (f, t.map(|i| m.positions[i as usize]))
            })
            .collect();
        let mut reach = Aabb::EMPTY;
        for &q in &m.positions {
            reach.add(q);
        }
        Ok(Arc::new(Seeds {
            tris,
            tree,
            deflection: seed.options.linear,
            reach,
        }))
    }
}

impl<'a> Rays<'a> {
    pub fn new(b: &'a Brep) -> Result<Rays<'a>> {
        Ok(Rays {
            b,
            seeds: b.arena().rays.get_or_init(|| Seeds::new(b)).clone()?,
            domains: b.face_ids().map(|f| face_domain(b, f)).collect(),
        })
    }

    /// Whether a solid of the arena holds `p`, by the parity of a ray's crossings of its faces: a
    /// point in a void, or between solids, is held by none. A ray that passes through an edge,
    /// grazes a face or starts on one does not say, and the next direction is tried. `None` where
    /// none says: the point is on the boundary.
    pub fn holds(&self, p: Point3) -> Option<bool> {
        if self.seeds.tris.is_empty() {
            return Some(false);
        }
        let reach = &self.seeds.reach;
        let far = reach.min.dist(p).max(reach.max.dist(p)) + reach.diagonal() + 1.0;
        DIRECTIONS.iter().find_map(|d| {
            let d = Vec3::from_array(*d).unit()?;
            let hits = self.crossings(p, d, (0.0, far));
            let unclear =
                |h: &Hit| h.on_edge || h.t <= tol::LINEAR || h.normal.dot(d).abs() < tol::TANGENT;
            (!hits.iter().any(unclear)).then_some(hits.len() % 2 == 1)
        })
    }

    /// Where `p` lies against the arena's solids. A point no ray decides is one every ray leaves
    /// through a face, an edge or a touch at its start: a point of the boundary.
    pub fn side(&self, p: Point3) -> Side {
        match self.holds(p) {
            Some(true) => Side::Inside,
            Some(false) => Side::Outside,
            None => Side::On,
        }
    }

    /// Every crossing of the ray `o + t·d` with the arena's faces for `t` in `[t0, t1]`, nearest
    /// first, `t` measured along `d` made a unit.
    pub fn cast(&self, o: Point3, d: Vec3, range: (f64, f64)) -> Result<Vec<Hit>> {
        let Some(d) = d.unit() else {
            return Err(Error::InvalidInput("a ray needs a direction".into()));
        };
        // Seeds from neighbouring triangles land on one crossing, and a crossing on an edge is on
        // both its faces.
        let mut hits = self.crossings(o, d, range);
        hits.dedup_by(|later, kept| later.t - kept.t <= tol::LINEAR);
        Ok(hits)
    }

    /// Every crossing of the ray, nearest first, a crossing on an edge once for each face.
    ///
    /// The mesh says where to look, and no more: each face is searched over the stretches of the
    /// ray that pass one of its triangles, and what is found there is every meeting of the line
    /// with the face's exact surface, by the intersection the booleans use. A chord across a
    /// curved face shallower than the mesh resolves is two crossings, as any other chord is.
    fn crossings(&self, o: Point3, d: Vec3, (t0, t1): (f64, f64)) -> Vec<Hit> {
        let pad = 2.0 * self.seeds.deflection;
        let (s0, s1) = (t0 - pad, t1 + pad);
        let mut near = Vec::new();
        self.seeds.tree.along(o, d, (s0, s1), pad, &mut near);
        let (a, z) = (o + d * s0, o + d * s1);
        let mut stretches: BTreeMap<FaceId, Vec<(f64, f64)>> = BTreeMap::new();
        for &k in &near {
            let (f, tri) = &self.seeds.tris[k];
            if segment_triangle(a, z, tri).0 > pad {
                continue;
            }
            // The surface over this triangle lies within the deflection of it, so the ray can
            // meet it only this far past the triangle's own reach along the ray.
            let along = tri.map(|v| (v - o).dot(d));
            let lo = along.iter().copied().fold(f64::INFINITY, f64::min) - pad;
            let hi = along.iter().copied().fold(f64::NEG_INFINITY, f64::max) + pad;
            stretches
                .entry(*f)
                .or_default()
                .push((lo.max(s0), hi.min(s1)));
        }
        let line = Curve::Line(Line { origin: o, dir: d });
        let mut hits: Vec<Hit> = Vec::new();
        for (f, mut spans) in stretches {
            spans.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut merged: Vec<(f64, f64)> = Vec::new();
            for (lo, hi) in spans {
                match merged.last_mut() {
                    Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                    _ => merged.push((lo, hi)),
                }
            }
            let face = self.b.face(f);
            let s = &face.surface;
            for span in merged {
                // A ray that runs in the surface crosses it nowhere.
                let found = curve_surface(&line, span, s);
                for t in found.ts {
                    if t < t0 - ON || t > t1 + ON {
                        continue;
                    }
                    let q = o + d * t;
                    let dom = &self.domains[f.index()];
                    let (uv, on_edge) = match dom.locate(q) {
                        Where::Outside => continue,
                        Where::Inside(uv) => (uv, false),
                        Where::Boundary => (s.project(q), true),
                    };
                    hits.push(Hit {
                        t: t.clamp(t0, t1),
                        point: q,
                        face: f,
                        normal: dom.outward(uv),
                        on_edge,
                    });
                }
            }
        }
        hits.sort_by(|x, y| x.t.total_cmp(&y.t).then(x.face.cmp(&y.face)));
        self.touches(hits, o, d)
    }

    /// `hits` with each run on one face that the ray stays on between, a ray grazing the surface,
    /// as one touch. A grazing ray stays within [`ON`] of the surface over a stretch as long as
    /// the square root of that over the curvature, so its seeds settle anywhere along it; the touch
    /// is where the surface's normal turns square to the ray, which bisection finds exactly.
    fn touches(&self, hits: Vec<Hit>, o: Point3, d: Vec3) -> Vec<Hit> {
        let mut out: Vec<Hit> = Vec::with_capacity(hits.len());
        let mut i = 0;
        while i < hits.len() {
            let h = hits[i];
            let s = &self.b.face(h.face).surface;
            let foot = |t: f64| {
                let p = o + d * t;
                let uv = s.project(p);
                (uv, s.eval(uv.x, uv.y).dist(p))
            };
            let mut j = i;
            while j + 1 < hits.len()
                && hits[j + 1].face == h.face
                && foot(0.5 * (hits[j].t + hits[j + 1].t)).1 <= ON
            {
                j += 1;
            }
            if j == i {
                out.push(h);
                i += 1;
                continue;
            }
            let square = |t: f64| {
                let (uv, _) = foot(t);
                s.normal(uv.x, uv.y).dot(d)
            };
            let (mut lo, mut hi) = (hits[i].t, hits[j].t);
            if square(lo).signum() != square(hi).signum() {
                for _ in 0..100 {
                    let mid = 0.5 * (lo + hi);
                    if square(mid).signum() == square(lo).signum() {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
            }
            let t = 0.5 * (lo + hi);
            let (uv, _) = foot(t);
            let n = s.normal(uv.x, uv.y);
            out.push(Hit {
                t,
                point: o + d * t,
                face: h.face,
                normal: if self.b.face(h.face).reversed { -n } else { n },
                on_edge: hits[i..=j].iter().any(|x| x.on_edge),
            });
            i = j + 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boolean,
        math::{Frame, Vec2},
        profile::{NamedSegment, Profile, ProfileLoop, Segment},
        sweep::extrude,
        sweep::{Helix, helix},
    };
    use std::f64::consts::{PI, TAU};

    #[test]
    fn a_ray_up_a_threaded_bore_meets_its_wall() {
        // A hex nut 7.6 tall, bored 4 in radius, a 60° thread ridge of pitch 1.2 joined inside it
        // for three turns from z = 2: the bore's wall left between the turns is a strip that winds
        // three times round, and a ray from inside the bore must cross it or a flank to leave.
        let seg = |segment, name: &str| NamedSegment::new(segment, name);
        let d: f64 = 6.8;
        let rc = 2.0 * d / 3f64.sqrt();
        let hex: Vec<Vec2> = (0..6)
            .map(|i| {
                let a = PI / 3.0 * i as f64;
                Vec2::new(rc * a.cos(), rc * a.sin())
            })
            .collect();
        let mut nut_profile = Profile::polygon(&hex, &["a", "b", "c", "d", "e", "f"]);
        nut_profile.loops.push(ProfileLoop {
            segments: vec![seg(
                Segment::Circle {
                    center: Vec2::ZERO,
                    radius: 4.0,
                    ccw: false,
                },
                "bore",
            )],
        });
        let nut = extrude(&nut_profile, &Frame::WORLD, 7.6, "nut").unwrap();
        let (pitch, major) = (1.2, 4.0);
        let minor = major - 0.6134 * pitch;
        let root = major + 0.1 * pitch;
        let half = |x: f64| (x - minor) * (PI / 6.0).tan();
        let tri = [
            Vec2::new(root, 2.0 - half(root)),
            Vec2::new(root, 2.0 + half(root)),
            Vec2::new(minor, 2.0),
        ];
        let ridge_profile = Profile {
            loops: vec![ProfileLoop {
                segments: (0..3)
                    .map(|i| {
                        seg(
                            Segment::Line {
                                a: tri[i],
                                b: tri[(i + 1) % 3],
                            },
                            "s",
                        )
                    })
                    .collect(),
            }],
        };
        let motion = Helix {
            origin: Point3::ZERO,
            axis: Vec3::Z,
            rise: pitch / TAU,
            sign: 1.0,
        };
        let meridian = Frame::new(Point3::ZERO, -Vec3::Y, Vec3::X).unwrap();
        let ridge = helix(&ridge_profile, &meridian, &motion, 3.0 * TAU, "hel").unwrap();
        let threaded = boolean::union(&nut, &ridge).unwrap();
        let rays = Rays::new(&threaded).unwrap();
        let a = 72.7f64.to_radians();
        let p = Point3::new(2.031 * a.cos(), 2.031 * a.sin(), 4.655);
        let dir = Vec3::new(0.3713, 0.5347, 0.7591);
        let hits = rays.cast(p, dir, (0.0, 1e6)).unwrap();
        let first = hits
            .first()
            .expect("the ray leaves the bore through something");
        assert!(
            first.normal.dot(dir) < 0.0,
            "the bore's first crossing enters the nut"
        );
    }
}
