//! [`Codec`] for the topology arena. A decoded arena keeps every id, so each name and every
//! reference into it holds; one whose ids point outside its own arrays is refused.

use std::sync::Arc;

use super::memo::{FaceMemo, Memos};
use super::*;
use crate::codec::{Codec, Reader, Writer, fields};

macro_rules! ids {
    ($($t:ident),*) => {$(
        impl Codec for $t {
            fn put(&self, w: &mut Writer) {
                self.0.put(w);
            }
            fn get(r: &mut Reader) -> Option<$t> {
                Some($t(u32::get(r)?))
            }
        }
    )*};
}
ids!(VertexId, EdgeId, CoedgeId, LoopId, FaceId, ShellId, SolidId);

fields!(FaceName { feature, face });
fields!(Vertex { point });
fields!(Edge {
    curve,
    range,
    start,
    end
});
fields!(Coedge {
    edge,
    reversed,
    pcurve,
    lp
});
fields!(Loop { coedges, face });
fields!(Face {
    surface,
    reversed,
    loops,
    names,
    shell
});
fields!(Shell { faces, solid });
fields!(Solid { shells, outer });
fields!(Tolerances { vertices, edges });

impl Codec for PCurve {
    fn put(&self, w: &mut Writer) {
        match self {
            PCurve::Affine { origin, dir } => {
                0u8.put(w);
                origin.put(w);
                dir.put(w);
            }
            PCurve::Projected { seed } => {
                1u8.put(w);
                seed.put(w);
            }
            PCurve::Spline { uv } => {
                2u8.put(w);
                uv.put(w);
            }
        }
    }
    fn get(r: &mut Reader) -> Option<PCurve> {
        Some(match u8::get(r)? {
            0 => PCurve::Affine {
                origin: Codec::get(r)?,
                dir: Codec::get(r)?,
            },
            1 => PCurve::Projected {
                seed: Codec::get(r)?,
            },
            2 => PCurve::Spline { uv: Codec::get(r)? },
            _ => return None,
        })
    }
}

/// What a face's memo has worked out that holds in any process: its volume share, its box, that
/// its loops are clean, and what a boolean reads from its trims. A boolean takes those trims from
/// the memo rather than working them out again, so a state read back must carry them, and share
/// each memo among the faces that shared it, to build what the state in memory would have. What
/// names other memos by their process-local ids is left out: it only spares the checker work.
fn put_memo(m: &Arc<FaceMemo>, w: &mut Writer) {
    let at = Arc::as_ptr(m) as usize;
    if let Some(&i) = w.1.get(&at) {
        1u8.put(w);
        i.put(w);
        return;
    }
    let n = u32::try_from(w.1.len()).expect("fewer than 2³² memos");
    w.1.insert(at, n);
    0u8.put(w);
    m.volume
        .get()
        .map(|v| (v.anchor, (v.volume, v.vector_area)))
        .put(w);
    m.loose_box.get().copied().put(w);
    m.clean.get().is_some().put(w);
    match m.images.get() {
        None => 0u8.put(w),
        Some(i) => {
            1u8.put(w);
            i.put(w);
        }
    }
}

fn get_memo(r: &mut Reader) -> Option<Arc<FaceMemo>> {
    if bool::get(r)? {
        let i = u32::get(r)? as usize;
        return r.memos.get(i).cloned();
    }
    let volume: Option<(Point3, (f64, crate::math::Vec3))> = Codec::get(r)?;
    let loose_box: Option<crate::math::Aabb> = Codec::get(r)?;
    let clean = bool::get(r)?;
    let images: Option<crate::boolean::FaceImages> = Codec::get(r)?;
    let m = FaceMemo::default();
    if let Some((anchor, (volume, vector_area))) = volume {
        let _ = m.volume.set(super::FaceVolume {
            anchor,
            volume,
            vector_area,
        });
    }
    if let Some(b) = loose_box {
        let _ = m.loose_box.set(b);
    }
    if clean {
        let _ = m.clean.set(());
    }
    if let Some(i) = images {
        let _ = m.images.set(i);
    }
    let m = Arc::new(m);
    r.memos.push(m.clone());
    Some(m)
}

impl Codec for Brep {
    fn put(&self, w: &mut Writer) {
        let Brep {
            vertices,
            edges,
            coedges,
            loops,
            faces,
            shells,
            solids,
            memos,
            arena: _,
            tolerances,
            partners,
            beside,
            uses: _,
        } = self;
        vertices.put(w);
        edges.put(w);
        coedges.put(w);
        loops.put(w);
        faces.put(w);
        shells.put(w);
        solids.put(w);
        for m in &memos.0 {
            put_memo(m, w);
        }
        tolerances.put(w);
        partners.put(w);
        beside.put(w);
    }

    fn get(r: &mut Reader) -> Option<Brep> {
        let vertices: Vec<Vertex> = Codec::get(r)?;
        let edges: Vec<Edge> = Codec::get(r)?;
        let coedges: Vec<Coedge> = Codec::get(r)?;
        let loops: Vec<Loop> = Codec::get(r)?;
        let faces: Vec<Face> = Codec::get(r)?;
        let shells: Vec<Shell> = Codec::get(r)?;
        let solids: Vec<Solid> = Codec::get(r)?;
        let memos = (0..faces.len())
            .map(|_| get_memo(r))
            .collect::<Option<Vec<_>>>()?;
        let tolerances: Box<Tolerances> = Codec::get(r)?;
        let partners: BTreeMap<CoedgeId, CoedgeId> = Codec::get(r)?;
        let beside: BTreeMap<CoedgeId, CoedgeId> = Codec::get(r)?;
        let mut b = Brep {
            vertices,
            edges,
            coedges,
            loops,
            faces,
            shells,
            solids,
            memos: Memos(memos),
            arena: Default::default(),
            tolerances,
            partners,
            beside,
            uses: Vec::new(),
        };
        if !b.ids_in_range() {
            return None;
        }
        b.index_uses();
        Some(b)
    }
}

impl Brep {
    /// Every id the arena holds names an entity of it.
    fn ids_in_range(&self) -> bool {
        let v = |id: VertexId| id.index() < self.vertices.len();
        let e = |id: EdgeId| id.index() < self.edges.len();
        let c = |id: CoedgeId| id.index() < self.coedges.len();
        let l = |id: LoopId| id.index() < self.loops.len();
        let f = |id: FaceId| id.index() < self.faces.len();
        let sh = |id: ShellId| id.index() < self.shells.len();
        let so = |id: SolidId| id.index() < self.solids.len();
        self.partners
            .iter()
            .chain(&self.beside)
            .all(|(&a, &b)| c(a) && c(b))
            && self.edges.iter().all(|x| v(x.start) && v(x.end))
            && self.coedges.iter().all(|x| e(x.edge) && l(x.lp))
            && self
                .loops
                .iter()
                .all(|x| f(x.face) && x.coedges.iter().all(|&i| c(i)))
            && self
                .faces
                .iter()
                .all(|x| x.loops.iter().all(|&i| l(i)) && x.shell.is_none_or(sh))
            && self
                .shells
                .iter()
                .all(|x| x.faces.iter().all(|&i| f(i)) && x.solid.is_none_or(so))
            && self.solids.iter().all(|x| x.shells.iter().all(|&i| sh(i)))
            && self.tolerances.vertices.keys().all(|&i| v(i))
            && self.tolerances.edges.keys().all(|&i| e(i))
    }
}
