use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::{
    error::Result,
    math::{Aabb, Point3, Vec2, Vec3},
    mesh::SeedMesh,
};

/// What the checker and booleans derive from one face's geometry, kept while that geometry is
/// unchanged. A boolean that passes a face through uncut gives the result's face the same memo, so
/// a part built by many features measures and samples each face once rather than once a feature.
#[derive(Debug)]
pub(crate) struct FaceMemo {
    /// Names this memo, and so one face's geometry, for as long as the process runs.
    pub id: u64,
    pub volume: OnceLock<FaceVolume>,
    /// Each coedge's edge range and sense, loop by loop, of the first face the memo was handed
    /// on from: what is kept here by a coedge's place holds only for a face laid out the same.
    #[cfg(debug_assertions)]
    pub layout: OnceLock<Vec<Vec<(u64, u64, bool)>>>,
    /// The face's last area and volume share, and its last moments with its area, each with the
    /// centre, size, tolerance and sense they were integrated for: a solid measured for its
    /// volume, then its area, then its centre, asks for each face's again with the same ones.
    pub integrals: Mutex<Option<(IntegralKey, (f64, f64))>>,
    pub moments: Mutex<Option<(IntegralKey, [f64; 11])>>,
    /// Each projected coedge's parameters along its edge, loop by loop, that seed the integrals'
    /// projections (see `measure::pcurve_tables`).
    pub pcurves: OnceLock<Vec<Vec<Vec<Vec2>>>>,
    /// The face's own loops lie on its surface, close in its parameters and run its way.
    pub clean: OnceLock<()>,
    pub images: OnceLock<crate::boolean::FaceImages>,
    /// The face's share of [`crate::measure::loose_bbox`].
    pub loose_box: OnceLock<Aabb>,
    /// The memos of faces the checker found this one does not meet away from their shared edges
    /// and vertices.
    pub apart: Mutex<HashSet<u64>>,
    /// Those of [`FaceMemo::apart`] found so with nothing left unsearched: the two faces' surfaces
    /// stand clear of each other there, or every curve they meet in is known and none runs
    /// inside both faces.
    pub clear: Mutex<HashSet<u64>>,
    /// The memos of faces sharing edges with this one that the checker found meeting it nowhere
    /// else, each with a hash of which coedges of the two the shared edges are.
    pub beside: Mutex<HashSet<(u64, u64)>>,
}

impl Default for FaceMemo {
    fn default() -> FaceMemo {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        FaceMemo {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            volume: OnceLock::new(),
            #[cfg(debug_assertions)]
            layout: OnceLock::new(),
            integrals: Mutex::new(None),
            moments: Mutex::new(None),
            pcurves: OnceLock::new(),
            clean: OnceLock::new(),
            images: OnceLock::new(),
            loose_box: OnceLock::new(),
            apart: Mutex::new(HashSet::new()),
            clear: Mutex::new(HashSet::new()),
            beside: Mutex::new(HashSet::new()),
        }
    }
}

/// What queries derive from a whole arena, kept for as long as the arena stays as it is.
#[derive(Default)]
pub(crate) struct ArenaMemo {
    /// The mesh that seeds exact searches (see [`crate::mesh::seed_mesh`]).
    pub seed: OnceLock<Result<Arc<SeedMesh>>>,
    /// That mesh's triangles and the tree over them, for rays.
    pub rays: OnceLock<Result<Arc<crate::rays::Seeds>>>,
    /// The mesh pieces of every solid, and of every face, for distances.
    pub solids: OnceLock<Result<Arc<crate::distance::Meshed>>>,
    pub faces: OnceLock<Result<Arc<crate::distance::Meshed>>>,
}

/// One arena's memo. A copy of the arena shares it, its geometry being the same, until either
/// changes and starts a fresh one. Never part of equality.
#[derive(Clone, Default)]
pub(crate) struct Arena(pub OnceLock<Arc<ArenaMemo>>);

impl PartialEq for Arena {
    fn eq(&self, _: &Arena) -> bool {
        true
    }
}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Arena")
    }
}

/// What a face's exact integrals depend on besides its geometry, compared bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IntegralKey([u64; 5], bool);

impl IntegralKey {
    pub(crate) fn new(c: Point3, scale: f64, rel: f64, reversed: bool) -> IntegralKey {
        IntegralKey([c.x, c.y, c.z, scale, rel].map(f64::to_bits), reversed)
    }
}

/// A face's volume contribution `⅓∬ (p − anchor)·n dA` at the rough tolerance, and `∬ n dA`, which
/// moves it to any other centre.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceVolume {
    pub anchor: Point3,
    pub volume: f64,
    pub vector_area: Vec3,
}

/// One memo per face of an arena. Never part of equality: two arenas with the same entities are
/// equal whatever either has worked out about them.
#[derive(Clone, Default)]
pub(crate) struct Memos(pub Vec<Arc<FaceMemo>>);

impl PartialEq for Memos {
    fn eq(&self, _: &Memos) -> bool {
        true
    }
}

impl fmt::Debug for Memos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Memos({})", self.0.len())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        geom::{Curve, Line, Plane, Surface},
        math::{Frame, Point3, Transform, Vec3},
        mesh, primitive,
        rays::{Rays, Side},
        topo::{Brep, CoedgeId, EdgeId, FaceId, VertexId},
    };
    use std::sync::Arc;

    fn block() -> Brep {
        primitive::block(&Frame::WORLD, Vec3::new(4.0, 3.0, 2.0), "b").unwrap()
    }

    type Change = (&'static str, fn(&mut Brep));

    #[test]
    fn every_change_to_an_arena_drops_what_was_kept_about_it() {
        let changes: Vec<Change> = vec![
            ("add_vertex", |b| {
                let _ = b.add_vertex(Point3::ZERO);
            }),
            ("add_edge", |b| {
                let (v, e) = (VertexId::at(0), VertexId::at(1));
                let line = Curve::Line(Line {
                    origin: Point3::ZERO,
                    dir: Vec3::X,
                });
                b.add_edge(line, (0.0, 1.0), v, e);
            }),
            ("add_degenerate_edge", |b| {
                b.add_degenerate_edge(VertexId::at(0), (0.0, 1.0));
            }),
            ("add_face", |b| {
                b.add_face(
                    Surface::Plane(Plane {
                        frame: Frame::WORLD,
                    }),
                    false,
                    [],
                );
            }),
            ("add_loop", |b| {
                let _ = b.add_loop(FaceId::at(0), Vec::new());
            }),
            ("add_shell", |b| {
                let _ = b.add_shell(Vec::new());
            }),
            ("add_solid", |b| {
                let _ = b.add_solid(Vec::new());
            }),
            ("pair", |b| b.pair(CoedgeId::at(0), CoedgeId::at(1))),
            ("set_beside", |b| {
                b.set_beside(CoedgeId::at(0), CoedgeId::at(1))
            }),
            ("pair_radial", |b| b.pair_radial()),
            ("face_mut", |b| {
                let _ = b.face_mut(FaceId::at(0));
            }),
            ("set_vertex_tolerance", |b| {
                b.set_vertex_tolerance(VertexId::at(0), 1e-3)
            }),
            ("set_edge_tolerance", |b| {
                b.set_edge_tolerance(EdgeId::at(0), 1e-3)
            }),
            ("transform", |b| {
                b.transform(&Transform::translation(Vec3::X))
            }),
            ("mirror", |b| {
                let m = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                b.transform_similar(m, Vec3::ZERO).unwrap();
            }),
            ("scale", |b| {
                let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
                b.transform_similar(m, Vec3::ZERO).unwrap();
            }),
            ("append", |b| {
                let _ = b.append(&block());
            }),
        ];
        let whole = block();
        let kept = mesh::seed_mesh(&whole).unwrap();
        for (name, change) in changes {
            let mut b = whole.clone();
            assert!(Arc::ptr_eq(&mesh::seed_mesh(&b).unwrap(), &kept), "{name}");
            change(&mut b);
            assert!(b.arena.0.get().is_none(), "{name}");
            // The solid it was copied from stays as it was, and so does what is kept about it.
            assert!(
                Arc::ptr_eq(&mesh::seed_mesh(&whole).unwrap(), &kept),
                "{name}"
            );
        }
    }

    #[test]
    fn an_arena_and_what_is_kept_about_it_cross_threads() {
        fn crosses<T: Send + Sync>() {}
        crosses::<Brep>();
        crosses::<crate::distance::Nearest>();
    }

    #[test]
    fn a_solid_asked_then_moved_answers_for_where_it_now_stands() {
        let mut b = block();
        let p = Point3::new(1.0, 1.0, 1.0);
        let side = |b: &Brep, p: Point3| Rays::new(b).unwrap().side(p);
        assert_eq!(side(&b, p), Side::Inside);
        let before = b.clone();
        // Slid 10 along x.
        b.transform(&Transform::translation(Vec3::new(10.0, 0.0, 0.0)));
        assert_eq!(side(&b, p), Side::Outside);
        assert_eq!(side(&b, Point3::new(11.0, 1.0, 1.0)), Side::Inside);
        // Mirrored in the plane x = 0: from x in [10, 14] to [-14, -10].
        let mirror = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        b.transform_similar(mirror, Vec3::ZERO).unwrap();
        assert_eq!(side(&b, Point3::new(11.0, 1.0, 1.0)), Side::Outside);
        assert_eq!(side(&b, Point3::new(-11.0, 1.0, 1.0)), Side::Inside);
        // Doubled about the origin: x in [-28, -20], y in [0, 6], z in [0, 4].
        let double = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        b.transform_similar(double, Vec3::ZERO).unwrap();
        assert_eq!(side(&b, Point3::new(-11.0, 1.0, 1.0)), Side::Outside);
        assert_eq!(side(&b, Point3::new(-24.0, 5.0, 3.0)), Side::Inside);
        assert_eq!(side(&b, Point3::new(-20.0, 5.0, 3.0)), Side::On);
        // The copy taken before any of it still stands where the solid did.
        assert_eq!(side(&before, p), Side::Inside);
        let d = crate::distance::closest(
            &before,
            &crate::distance::Selection::body(&before),
            &b,
            &crate::distance::Selection::body(&b),
        );
        assert!((d.unwrap().distance - 20.0).abs() <= 1e-9);
    }
}
