//! Whether two placed solids share material, and how much: interference for an assembly sweep.
//!
//! Each solid is meshed once, with a bounding volume hierarchy over its triangles. A mesh lies
//! within its deflection of the exact surface, so two meshes whose triangles stay farther apart
//! than both deflections, with room to spare, belong to solids whose surfaces do not meet: those
//! two either share nothing or one holds the other whole, which a point of one tells. Only a pair
//! whose meshes come that near is intersected exactly, and its volume measured on the result.

use std::collections::BinaryHeap;

use crate::{
    Result, boolean,
    distance::{self, Selection},
    math::{Aabb, Point3, Transform, Vec3},
    measure,
    mesh::{
        Mesh, MeshOptions,
        bvh::{Bvh, moved_box, on_triangle, ray_triangle, triangles},
        tessellate,
    },
    tol,
    topo::Brep,
};

/// The meshes tried, finest first: the mesher can give up on a fine chord over a large part.
const MESHES: [MeshOptions; 3] = [
    MeshOptions {
        linear: 0.01,
        angular: 0.2,
    },
    MeshOptions {
        linear: 0.05,
        angular: 0.5,
    },
    MeshOptions {
        linear: 0.2,
        angular: 0.8,
    },
];

/// Overlap at most this, mm³, is surfaces in contact rather than shared material: the exact
/// intersection of two touching solids measures zero up to rounding.
pub const TOUCHING_MM3: f64 = 1e-9;

/// The tolerance shared volumes are integrated to, relative to their size cubed.
const ROUGH_VOLUME: f64 = 1e-10;

/// Below this the shared volume is integrated again to every digit, mm³.
const SLIVER_MM3: f64 = 1e-6;

/// A solid made ready to be tried against others: meshed once in its own frame.
pub struct Body {
    brep: Brep,
    mesh: Mesh,
    /// Per triangle, which solid of the body it bounds.
    part: Vec<u32>,
    /// How many solids the body holds.
    solids: u32,
    bvh: Bvh,
    deflection: f64,
    bbox: Aabb,
    /// The solid's volume, measured the first time one body is found whole inside another: on a
    /// part with large spline faces it takes longer than meshing, and most pairs never need it.
    volume: std::sync::OnceLock<f64>,
}

impl Body {
    pub fn new(brep: Brep) -> Result<Body> {
        // Every placed copy of the body inherits the spline surfaces' seeds and polynomials.
        for f in brep.face_ids() {
            if let crate::geom::Surface::Nurbs(n) = &brep.face(f).surface {
                n.prepare();
            }
        }
        let mut last = None;
        for opt in MESHES {
            match tessellate(&brep, &opt) {
                Ok(mesh) => {
                    let bvh = Bvh::new(&mesh);
                    let mut bbox = Aabb::EMPTY;
                    for p in &mesh.positions {
                        bbox.add(*p);
                    }
                    return Ok(Body {
                        part: vec![0; mesh.triangles.len()],
                        solids: 1,
                        brep,
                        mesh,
                        bvh,
                        deflection: opt.linear,
                        bbox,
                        volume: std::sync::OnceLock::new(),
                    });
                }
                Err(e) => last = Some(e),
            }
        }
        Err(last.expect("MESHES is not empty"))
    }

    pub fn brep(&self) -> &Brep {
        &self.brep
    }

    /// Several bodies as one that moves whole, each kept a solid of its own: a host with the
    /// inserts pressed into it needs no union to be tried against others.
    pub fn joined(parts: &[&Body]) -> Body {
        let mut brep = Brep::default();
        let mut mesh = Mesh::default();
        let mut part = Vec::new();
        let mut solids = 0;
        let mut bbox = Aabb::EMPTY;
        let mut deflection: f64 = 0.0;
        for b in parts {
            brep.append(&b.brep);
            let base = mesh.positions.len() as u32;
            mesh.positions.extend_from_slice(&b.mesh.positions);
            mesh.triangles
                .extend(b.mesh.triangles.iter().map(|t| t.map(|i| i + base)));
            part.extend(b.part.iter().map(|k| solids + k));
            solids += b.solids;
            bbox.union(&b.bbox);
            deflection = deflection.max(b.deflection);
        }
        Body {
            bvh: Bvh::new(&mesh),
            brep,
            mesh,
            part,
            solids,
            deflection,
            bbox,
            volume: std::sync::OnceLock::new(),
        }
    }

    /// A copy placed by the rigid motion `t`: its mesh moved with it rather than meshed again,
    /// within the same deflection of the moved solid, as a rigid motion keeps every distance.
    pub fn moved(&self, t: &Transform) -> Body {
        let mut brep = self.brep.clone();
        brep.transform(t);
        let mut mesh = self.mesh.clone();
        for p in &mut mesh.positions {
            *p = t.point(*p);
        }
        let mut bbox = Aabb::EMPTY;
        for p in &mesh.positions {
            bbox.add(*p);
        }
        let volume = std::sync::OnceLock::new();
        if let Some(&v) = self.volume.get() {
            let _ = volume.set(v);
        }
        Body {
            bvh: Bvh::new(&mesh),
            brep,
            mesh,
            part: self.part.clone(),
            solids: self.solids,
            deflection: self.deflection,
            bbox,
            volume,
        }
    }

    /// The mesh's corners, in the body's frame.
    pub fn points(&self) -> &[Point3] {
        &self.mesh.positions
    }

    /// Whether `p` lies farther than `r` from every triangle of the mesh.
    fn clear_of(&self, p: Point3, r: f64) -> bool {
        let mut near = Vec::new();
        self.bvh.within(&Aabb { min: p, max: p }, r, &mut near);
        near.iter().all(|&i| {
            let t = self.triangle(i);
            on_triangle(p, &t).dist(p) > r
        })
    }

    fn volume(&self) -> f64 {
        *self.volume.get_or_init(|| measure::volume(&self.brep))
    }

    fn triangle(&self, i: usize) -> [Point3; 3] {
        let t = self.mesh.triangles[i];
        [0, 1, 2].map(|k| self.mesh.positions[t[k] as usize])
    }

    /// Whether `p`, in this body's frame, lies inside its mesh: the parity of a ray's crossings,
    /// taken along three directions and decided by two. `p` must lie well clear of the mesh.
    fn holds(&self, p: Point3) -> bool {
        const RAYS: [Vec3; 3] = [
            Vec3::new(0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
            Vec3::new(-0.2672612419124244, 0.5345224838248488, 0.8017837257372732),
            Vec3::new(0.8164965809277261, -0.4082482904638631, 0.4082482904638631),
        ];
        let reach = 2.0 * (self.bbox.max - self.bbox.min).norm() + 1.0;
        let mut out = Vec::new();
        let votes = RAYS
            .iter()
            .filter(|&&d| {
                self.bvh.along(p, d, (0.0, reach), 0.0, &mut out);
                let hits = out
                    .iter()
                    .filter(|&&i| {
                        let [a, b, c] = self.triangle(i);
                        ray_triangle(p, d, a, b, c).is_some_and(|s| s > 0.0)
                    })
                    .count();
                hits % 2 == 1
            })
            .count();
        votes >= 2
    }
}

/// How two placed solids meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Overlap {
    /// Their surfaces stay farther apart than the meshes can tell, and neither holds the other.
    Apart,
    /// They meet, and share at most [`TOUCHING_MM3`]: faces or edges in contact.
    Touching,
    /// They share this much material, mm³.
    Shared(f64),
}

impl Overlap {
    pub fn volume(&self) -> f64 {
        match self {
            Overlap::Shared(v) => *v,
            _ => 0.0,
        }
    }
}

/// How `a` and `b` meet with `b` placed by `rel` in `a`'s frame.
pub fn overlap(a: &Body, b: &Body, rel: &Transform) -> Result<Overlap> {
    // Twice both deflections: where the exact surfaces meet, their meshes lie within both.
    let margin = 2.0 * (a.deflection + b.deflection);
    let moved = moved_box(&b.bbox, rel);
    let boxes_meet = crate::mesh::bvh::box_gap(&a.bbox, &moved) <= margin;
    let near = boxes_meet
        && a.bvh.any_within(&b.bvh, rel, margin, |i, j| {
            let tb = b.triangle(j).map(|p| rel.point(p));
            triangles(&a.triangle(i), &tb) <= margin
        });
    if !near {
        if !boxes_meet {
            return Ok(Overlap::Apart);
        }
        // No surfaces near: one holds the other whole, or they share nothing.
        if let Some(&p) = b.mesh.positions.first()
            && a.holds(rel.point(p))
        {
            return Ok(Overlap::Shared(b.volume()));
        }
        if let Some(&p) = a.mesh.positions.first()
            && b.holds(rel.inverse().point(p))
        {
            return Ok(Overlap::Shared(a.volume()));
        }
        return Ok(Overlap::Apart);
    }
    // Each face of `b` that can reach `a` has its volume share worked out on `b` itself, once:
    // every copy moved for a later pose keeps it, where the copy's own would be lost.
    let back = rel.inverse();
    let reach = moved_box(&a.bbox, &back);
    for f in b.brep.face_ids() {
        if crate::mesh::bvh::box_gap(&measure::face_bbox(&b.brep, f), &reach) <= margin {
            measure::prime_face_volume(&b.brep, f);
        }
    }
    let mut placed = b.brep.clone();
    placed.transform(rel);
    let shared = boolean::intersection(&a.brep, &placed)?;
    // To 1e-10 of the size cubed the volume is good to well under a micron cubed; only a sliver
    // near the touching bound needs every digit to say which side of it it falls.
    let mut v = measure::volume_to(&shared, ROUGH_VOLUME);
    if v < SLIVER_MM3 {
        v = measure::volume(&shared);
    }
    Ok(if v > TOUCHING_MM3 {
        Overlap::Shared(v)
    } else {
        Overlap::Touching
    })
}

/// How two solids meet, with the gap between them where they do not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Meeting {
    /// They share this much material, mm³.
    Shared(f64),
    /// They share none, and their surfaces lie within the linear tolerance of each other.
    Touching,
    /// They share none, and their surfaces come no nearer than this, mm.
    Apart(f64),
}

/// How `a` and `b` meet with `b` placed by `rel` in `a`'s frame. Where [`overlap`] stops at
/// whether the two share material, calling a hair's gap touching, this measures the gap.
pub fn meeting(a: &Body, b: &Body, rel: &Transform) -> Result<Meeting> {
    if let Overlap::Shared(v) = overlap(a, b, rel)? {
        return Ok(Meeting::Shared(v));
    }
    let mut placed = b.brep.clone();
    placed.transform(rel);
    let gap = distance::closest(
        &a.brep,
        &Selection::body(&a.brep),
        &placed,
        &Selection::body(&placed),
    )?
    .distance;
    Ok(if gap <= tol::LINEAR {
        Meeting::Touching
    } else {
        Meeting::Apart(gap)
    })
}

/// How two solids meet as they stand, each in the one frame.
pub fn between(a: &Brep, b: &Brep) -> Result<Meeting> {
    meeting(
        &Body::new(a.clone())?,
        &Body::new(b.clone())?,
        &Transform::IDENTITY,
    )
}

/// How finely [`swept_contact`] searches, and what it counts as one solid running into another.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Sweep {
    /// How deep inside both a point must lie for the move to have run in, mm, never less than the
    /// meshes' deflection: a sliver of the two thinner than twice this only touches.
    pub inside: f64,
    /// The finest cell the plane across the move is split into, mm: one ray stands for each, so
    /// material narrower than this across the move can slip between them.
    pub cell: f64,
    /// How much farther than the first contact the answer may lie, mm: the search stops once no
    /// cell left could hold a contact sooner by more than this.
    pub slack: f64,
}

impl Sweep {
    pub const fn new(inside: f64, cell: f64, slack: f64) -> Sweep {
        Sweep {
            inside,
            cell,
            slack,
        }
    }
}

impl Default for Sweep {
    /// Any point certainly inside both, the first contact itself, on cells a tenth of a
    /// millimetre wide.
    fn default() -> Sweep {
        Sweep {
            inside: 0.0,
            cell: 0.1,
            slack: 0.0,
        }
    }
}

/// A triangle seen along a move: its corners across it and how far along it each lies.
struct Seen {
    uv: [[f64; 2]; 3],
    w: [f64; 3],
    lo: [f64; 2],
    hi: [f64; 2],
    /// Facing along the move, or seen edge on: only such a triangle can lead its body into
    /// another.
    front: bool,
    /// Which solid of its body it bounds.
    part: u32,
    /// How far along the move its corners reach, least and most.
    own: (f64, f64),
    /// Its plane as `w = w0 + slope · (u − u0, v − v0)` from its first corner, or none seen edge
    /// on.
    slope: Option<[f64; 2]>,
}

impl Seen {
    fn new(t: &[Point3; 3], part: u32, [u, v, d]: [Vec3; 3]) -> Seen {
        let at = |p: Point3| {
            let x = p - Point3::new(0.0, 0.0, 0.0);
            ([x.dot(u), x.dot(v)], x.dot(d))
        };
        let (c, w): (Vec<[f64; 2]>, Vec<f64>) = t.iter().map(|&p| at(p)).unzip();
        let uv = [c[0], c[1], c[2]];
        let lo = [0, 1].map(|k| uv[0][k].min(uv[1][k]).min(uv[2][k]));
        let hi = [0, 1].map(|k| uv[0][k].max(uv[1][k]).max(uv[2][k]));
        // The mesh winds each triangle about its outward normal, and (u, v, d) is right-handed,
        // so the signed area across the move is the normal's share along it.
        let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
        let facing = e1.cross(e2).dot(d);
        let span = e1.norm().max(e2.norm());
        let w = [w[0], w[1], w[2]];
        let own = (w[0].min(w[1]).min(w[2]), w[0].max(w[1]).max(w[2]));
        let [p, q, r] = uv;
        let (f1, f2) = ([q[0] - p[0], q[1] - p[1]], [r[0] - p[0], r[1] - p[1]]);
        let det = f1[0] * f2[1] - f1[1] * f2[0];
        let reach = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let slope = (det.abs() > 1e-12 * reach * reach.max(1.0)).then(|| {
            let (dw1, dw2) = (w[1] - w[0], w[2] - w[0]);
            [
                (dw1 * f2[1] - dw2 * f1[1]) / det,
                (dw2 * f1[0] - dw1 * f2[0]) / det,
            ]
        });
        Seen {
            uv,
            w,
            lo,
            hi,
            front: facing >= -1e-9 * span * span,
            part,
            own,
            slope,
        }
    }

    /// Whether the triangle, seen along the move, meets the cell: no edge's normal, nor either of
    /// the cell's sides, separates them, by the separating axis theorem (Gottschalk, Lin and
    /// Manocha, "OBBTree: A Hierarchical Structure for Rapid Interference Detection", SIGGRAPH
    /// 1996). A triangle seen edge on is a slanting segment, whose box would claim cells it never
    /// crosses.
    fn meets(&self, cell: &Cell) -> bool {
        if self.lo[0] > cell.hi[0]
            || self.hi[0] < cell.lo[0]
            || self.lo[1] > cell.hi[1]
            || self.hi[1] < cell.lo[1]
        {
            return false;
        }
        let corners = [
            [cell.lo[0], cell.lo[1]],
            [cell.hi[0], cell.lo[1]],
            [cell.lo[0], cell.hi[1]],
            [cell.hi[0], cell.hi[1]],
        ];
        (0..3).all(|k| {
            let (p, q) = (self.uv[k], self.uv[(k + 1) % 3]);
            let n = [q[1] - p[1], p[0] - q[0]];
            if n[0] == 0.0 && n[1] == 0.0 {
                return true;
            }
            let side = |c: [f64; 2]| n[0] * (c[0] - p[0]) + n[1] * (c[1] - p[1]);
            let tri = self.uv.map(side);
            let (t0, t1) = (
                tri[0].min(tri[1]).min(tri[2]),
                tri[0].max(tri[1]).max(tri[2]),
            );
            let box_side = corners.map(side);
            let (b0, b1) = (
                box_side.iter().copied().fold(f64::INFINITY, f64::min),
                box_side.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
            b0 <= t1 && b1 >= t0
        })
    }

    /// How far along the move the triangle reaches over `cell`: within its own corners', and,
    /// where it is not seen edge on, within its plane's at the cell's corners.
    fn depth_over(&self, cell: &Cell) -> (f64, f64) {
        let Some([a, b]) = self.slope else {
            return self.own;
        };
        // A plane over a rectangle is least and most at two of its corners.
        let p = self.uv[0];
        let base = self.w[0] - a * p[0] - b * p[1];
        let (au, bv) = (
            (a * cell.lo[0], a * cell.hi[0]),
            (b * cell.lo[1], b * cell.hi[1]),
        );
        let lo = base + au.0.min(au.1) + bv.0.min(bv.1);
        let hi = base + au.0.max(au.1) + bv.0.max(bv.1);
        (self.own.0.max(lo), self.own.1.min(hi))
    }

    /// Where the line across the plane at `(x, y)`, along the move, crosses the triangle.
    fn crossing(&self, x: f64, y: f64) -> Option<f64> {
        let [p, q, r] = self.uv;
        let (e1, e2) = ([q[0] - p[0], q[1] - p[1]], [r[0] - p[0], r[1] - p[1]]);
        let det = e1[0] * e2[1] - e1[1] * e2[0];
        if det.abs() <= 1e-300 {
            return None;
        }
        let (dx, dy) = (x - p[0], y - p[1]);
        let s = (dx * e2[1] - dy * e2[0]) / det;
        let t = (e1[0] * dy - e1[1] * dx) / det;
        (s >= 0.0 && t >= 0.0 && s + t <= 1.0)
            .then(|| self.w[0] + s * (self.w[1] - self.w[0]) + t * (self.w[2] - self.w[0]))
    }
}

/// A rectangle of the plane across a move, with the triangles of each body over it and the
/// least distance the move could go before running in there.
struct Cell {
    lo: [f64; 2],
    hi: [f64; 2],
    a: Vec<u32>,
    b: Vec<u32>,
    soonest: f64,
}

impl PartialEq for Cell {
    fn eq(&self, other: &Cell) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Cell {}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Cell) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    // Soonest first out of a max-heap; then by place, so the search is the same every run.
    fn cmp(&self, other: &Cell) -> std::cmp::Ordering {
        other
            .soonest
            .total_cmp(&self.soonest)
            .then(other.lo[0].total_cmp(&self.lo[0]))
            .then(other.lo[1].total_cmp(&self.lo[1]))
    }
}

/// How far `a` moves along `d` before it runs into `b`, both in one frame, or `None` when it passes
/// clear: [`swept_contacts`] for one body against one.
pub fn swept_contact(a: &Body, b: &Body, d: Vec3, how: &Sweep) -> Option<f64> {
    swept_contacts(&[a], &[b], d, how)[0]
}

/// How far the bodies `moving`, moved together along `d`, go before they run into each of
/// `fixed`, all in one frame: per fixed body, `None` where they pass it clear. They run into one
/// where they first carry a point at least [`Sweep::inside`] inside both, beyond what the two
/// share where they stand. A point inside both meshes farther than each mesh's deflection from it
/// is inside both exact solids, so the meshes' slivers where two surfaces coincide, a shaft along
/// its bore, never count. The moving bodies' triangles are seen along the move once for all the
/// fixed ones.
///
/// For each fixed body the plane across the move is split into cells, soonest possible contact
/// first. Over a cell each body's material lies within the reach of its triangles there, so a
/// cell where nothing of the fixed body lies ahead of the moving faces that face along the move is
/// clear, and the soonest they could reach it there is the gap from those faces' farthest to its
/// nearest. Only cells that could hold the first contact are split, down to [`Sweep::cell`],
/// where one ray decides each.
pub fn swept_contacts(moving: &[&Body], fixed: &[&Body], d: Vec3, how: &Sweep) -> Vec<Option<f64>> {
    let Some(frame) = across(d) else {
        return vec![None; fixed.len()];
    };
    let d = frame[2];
    let boxed = |bodies: &[&Body]| {
        let mut all = Aabb::EMPTY;
        for body in bodies {
            all.union(&body.bbox);
        }
        all
    };
    let (reach, from) = (boxed(fixed), boxed(moving));
    // The moving triangles a line along the move through any fixed body can cross, each solid of
    // each body numbered apart, so a ray's stretches in each keep their own parity.
    let mut sa = Vec::new();
    let mut first_part = 0;
    for body in moving {
        let mut near = Vec::new();
        body.bvh.along_line(&reach, d, &mut near);
        sa.extend(
            near.iter()
                .map(|&i| Seen::new(&body.triangle(i), first_part + body.part[i], frame)),
        );
        first_part += body.solids;
    }
    let ra = moving
        .iter()
        .map(|b| 1.01 * b.deflection)
        .fold(how.inside, f64::max);
    let clear_of_moving = |p: Point3| moving.iter().all(|b| b.clear_of(p, ra));
    fixed
        .iter()
        .map(|body| {
            let mut near = Vec::new();
            body.bvh.along_line(&from, d, &mut near);
            let sb: Vec<Seen> = near
                .iter()
                .map(|&i| Seen::new(&body.triangle(i), body.part[i], frame))
                .collect();
            let rb = (1.01 * body.deflection).max(how.inside);
            first_contact(&sa, &sb, frame, (ra, rb), how, &clear_of_moving, &|p| {
                body.clear_of(p, rb)
            })
        })
        .collect()
}

/// Two directions square to the unit `d` and `d` itself, a right-handed frame.
fn across(d: Vec3) -> Option<[Vec3; 3]> {
    let d = d.unit()?;
    let u = if d.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = (u - d * u.dot(d)).unit()?;
    Some([u, d.cross(u), d])
}

/// [`swept_contacts`] for the moving triangles `sa` against one fixed body's `sb`, seen along the
/// move in `frame`: `(ra, rb)` how far a point must lie from each side's meshes, and
/// `clear_a`/`clear_b` whether it does.
fn first_contact(
    sa: &[Seen],
    sb: &[Seen],
    [u, v, d]: [Vec3; 3],
    (ra, rb): (f64, f64),
    how: &Sweep,
    clear_a: &dyn Fn(Point3) -> bool,
    clear_b: &dyn Fn(Point3) -> bool,
) -> Option<f64> {
    if sb.is_empty() {
        return None;
    }
    let shadow = |s: &[Seen], of: &[u32]| {
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for &i in of {
            let t = &s[i as usize];
            for k in 0..2 {
                lo[k] = lo[k].min(t.lo[k]);
                hi[k] = hi[k].max(t.hi[k]);
            }
        }
        (lo, hi)
    };
    let all_b: Vec<u32> = (0..sb.len() as u32).collect();
    let (lb, hb) = shadow(sb, &all_b);
    // Only the moving triangles over the fixed body's shadow.
    let all_a: Vec<u32> = (0..sa.len() as u32)
        .filter(|&i| {
            let t = &sa[i as usize];
            t.lo[0] <= hb[0] && t.hi[0] >= lb[0] && t.lo[1] <= hb[1] && t.hi[1] >= lb[1]
        })
        .collect();
    if all_a.is_empty() {
        return None;
    }
    let (la, ha) = shadow(sa, &all_a);
    let lo = [la[0].max(lb[0]), la[1].max(lb[1])];
    let hi = [ha[0].min(hb[0]), ha[1].min(hb[1])];
    // Shadows meeting only along a line hold no material in common.
    if hi[0] <= lo[0] || hi[1] <= lo[1] {
        return None;
    }
    // Over a cell: the triangles of each over it, and the soonest `a` could run into `b` there.
    let make = |lo: [f64; 2], hi: [f64; 2], pa: &[u32], pb: &[u32]| -> Option<Cell> {
        let mut cell = Cell {
            lo,
            hi,
            a: Vec::new(),
            b: Vec::new(),
            soonest: 0.0,
        };
        // Only `a`'s faces facing along the move can lead it into `b`.
        let mut ka = Vec::new();
        let (mut a_near, mut a_far) = (f64::INFINITY, f64::NEG_INFINITY);
        for &i in pa {
            let t = &sa[i as usize];
            if t.meets(&cell) {
                ka.push(i);
                if t.front {
                    let (x, y) = t.depth_over(&cell);
                    a_near = a_near.min(x);
                    a_far = a_far.max(y);
                }
            }
        }
        if ka.is_empty() || a_near > a_far {
            return None;
        }
        // Only what of `b` reaches past `a`'s near end can be run into: a wall wholly behind it
        // says nothing of how soon.
        let mut kb = Vec::new();
        let mut ahead = f64::INFINITY;
        for &i in pb {
            let t = &sb[i as usize];
            if t.meets(&cell) {
                kb.push(i);
                let (near, far) = t.depth_over(&cell);
                if far > a_near {
                    ahead = ahead.min(near);
                }
            }
        }
        if !ahead.is_finite() {
            return None;
        }
        cell.a = ka;
        cell.b = kb;
        cell.soonest = (ahead - a_far).max(0.0);
        Some(cell)
    };
    let mut queue = BinaryHeap::new();
    queue.extend(make(lo, hi, &all_a, &all_b));
    let mut best = f64::INFINITY;
    while let Some(cell) = queue.pop() {
        if cell.soonest >= best - how.slack {
            break;
        }
        let size = (cell.hi[0] - cell.lo[0]).max(cell.hi[1] - cell.lo[1]);
        if size > how.cell {
            // Halved along each side longer than the finest cell, so no two children are one.
            let halves = |k: usize| {
                let (lo, hi) = (cell.lo[k], cell.hi[k]);
                if hi - lo > how.cell {
                    let mid = 0.5 * (lo + hi);
                    vec![(lo, mid), (mid, hi)]
                } else {
                    vec![(lo, hi)]
                }
            };
            for (x0, x1) in halves(0) {
                for (y0, y1) in halves(1) {
                    queue.extend(make([x0, y0], [x1, y1], &cell.a, &cell.b));
                }
            }
            continue;
        }
        // Off the centre by an irrational share, so the ray does not run along a mesh's edges.
        let x = cell.lo[0] + (0.5 + 0.0618) * (cell.hi[0] - cell.lo[0]);
        let y = cell.lo[1] + (0.5 - 0.0414) * (cell.hi[1] - cell.lo[1]);
        let (Some(on_a), Some(on_b)) = (stretches(sa, &cell.a, x, y), stretches(sb, &cell.b, x, y))
        else {
            continue;
        };
        let place = Point3::new(0.0, 0.0, 0.0) + u * x + v * y;
        if let Some(t) = ray_contact(
            &on_a,
            &on_b,
            |at, travel| {
                let p = place + d * at;
                clear_b(p) && clear_a(p - d * travel)
            },
            2.0 * (ra + rb),
        ) {
            best = best.min(t);
        }
    }
    best.is_finite().then_some(best)
}

/// Where the line across the plane at `(x, y)` runs inside the body the triangles `tris` of `s`
/// bound: inside each solid between its crossings in pairs, and inside the body where inside any.
/// A host's exit and its insert's entry fall at one place, which one list would cancel. `None`
/// where a solid is crossed an odd number of times, through an edge or a corner.
fn stretches(s: &[Seen], tris: &[u32], x: f64, y: f64) -> Option<Vec<(f64, f64)>> {
    let mut ws: Vec<(u32, f64)> = tris
        .iter()
        .filter_map(|&i| {
            let t = &s[i as usize];
            Some((t.part, t.crossing(x, y)?))
        })
        .collect();
    ws.sort_by(|p, q| p.0.cmp(&q.0).then(p.1.total_cmp(&q.1)));
    ws.dedup_by(|p, q| p.0 == q.0 && p.1 - q.1 <= 1e-9);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for run in ws.chunk_by(|p, q| p.0 == q.0) {
        if !run.len().is_multiple_of(2) {
            return None;
        }
        out.extend(run.chunks(2).map(|c| (c[0].1, c[1].1)));
    }
    out.sort_by(|p, q| p.0.total_cmp(&q.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (lo, hi) in out {
        match merged.last_mut() {
            Some(last) if lo <= last.1 + 1e-9 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    Some(merged)
}

/// Along one ray with `a`'s stretches `on_a` moving ahead into `b`'s `on_b`: the travel of a
/// leading end to `b` where it then reaches a point `certain` holds inside both, the point's place
/// and the travel to it given, first tried `first` past the meeting and twice as deep each time.
/// The move must add to what the two share along the ray by half that depth: a part sliding along
/// a fit modelled tight pushes its front into the other as far as its back comes out of it.
fn ray_contact(
    on_a: &[(f64, f64)],
    on_b: &[(f64, f64)],
    certain: impl Fn(f64, f64) -> bool,
    first: f64,
) -> Option<f64> {
    let mut tries: Vec<(f64, f64, f64)> = Vec::new();
    for &(p0, lead) in on_a {
        for &(q0, q1) in on_b {
            if q1 <= lead {
                continue;
            }
            let from = q0.max(lead);
            let mut depth = first;
            while from + depth < q1 && depth <= lead - p0 {
                tries.push((from - lead, from + depth - lead, from + 0.5 * depth));
                depth *= 2.0;
            }
        }
    }
    tries.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.total_cmp(&y.1)));
    let shared = |t: f64| -> f64 {
        let mut total = 0.0;
        for &(p0, p1) in on_a {
            for &(q0, q1) in on_b {
                total += ((p1 + t).min(q1) - (p0 + t).max(q0)).max(0.0);
            }
        }
        total
    };
    let start = shared(0.0);
    tries
        .into_iter()
        .find(|&(contact, travel, at)| {
            shared(travel) - start >= 0.5 * (travel - contact) && certain(at, travel)
        })
        .map(|t| t.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        math::{Frame, Vec2},
        profile::Profile,
        sweep::extrude,
    };

    fn solid(lo: [f64; 3], hi: [f64; 3]) -> Brep {
        let c = Vec2::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
        let frame = Frame::new(Vec3::new(0.0, 0.0, lo[2]), Vec3::Z, Vec3::X).unwrap();
        let p = Profile::rect(c, hi[0] - lo[0], hi[1] - lo[1]);
        extrude(&p, &frame, hi[2] - lo[2], "k").unwrap()
    }

    fn block(lo: [f64; 3], hi: [f64; 3]) -> Body {
        Body::new(solid(lo, hi)).unwrap()
    }

    fn moved(x: f64, y: f64, z: f64) -> Transform {
        Transform::translation(Vec3::new(x, y, z))
    }

    #[test]
    fn blocks_apart_touching_and_sharing() {
        // A 10 mm cube against a 4 mm one moved along x: 1 mm clear of it, on its face, and 1.5 mm
        // into it, sharing 1.5 × 4 × 4 = 24 mm³.
        let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = block([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
        assert_eq!(
            overlap(&a, &b, &moved(11.0, 3.0, 3.0)).unwrap(),
            Overlap::Apart
        );
        assert_eq!(
            overlap(&a, &b, &moved(10.0, 3.0, 3.0)).unwrap(),
            Overlap::Touching
        );
        let v = overlap(&a, &b, &moved(8.5, 3.0, 3.0)).unwrap().volume();
        assert!((v - 24.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn a_meeting_measures_a_gap_the_meshes_cannot_see() {
        // The 4 mm cube a hundred-thousandth of a millimetre clear of the 10 mm one's face: their
        // meshes meet, they share nothing, and the gap is the hundred-thousandth.
        let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = block([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
        let Meeting::Apart(gap) = meeting(&a, &b, &moved(10.0 + 1e-5, 3.0, 3.0)).unwrap() else {
            panic!("a clear pair read as meeting");
        };
        assert!((gap - 1e-5).abs() < 1e-9, "{gap}");
        assert_eq!(
            meeting(&a, &b, &moved(10.0, 3.0, 3.0)).unwrap(),
            Meeting::Touching
        );
        let Meeting::Apart(far) = meeting(&a, &b, &moved(13.0, 3.0, 3.0)).unwrap() else {
            panic!("a clear pair read as meeting");
        };
        assert!((far - 3.0).abs() < 1e-9, "{far}");
        let shared = between(
            &solid([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
            &solid([8.5, 3.0, 3.0], [12.5, 7.0, 7.0]),
        )
        .unwrap();
        let Meeting::Shared(v) = shared else {
            panic!("an overlapping pair read as {shared:?}");
        };
        assert!((v - 24.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn a_block_buried_whole_shares_its_volume() {
        // The 4 mm cube 3 mm inside the 10 mm one on every side: no faces meet and the meshes stay
        // apart, so the ray parity finds it held. It shares all 64 mm³, either way round.
        let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let b = block([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
        let inside = moved(3.0, 3.0, 3.0);
        for v in [
            overlap(&a, &b, &inside).unwrap().volume(),
            overlap(&b, &a, &inside.inverse()).unwrap().volume(),
        ] {
            assert!((v - 64.0).abs() < 1e-9, "{v}");
        }
    }

    #[test]
    fn a_block_in_a_notch_inside_its_box_but_clear_of_it_is_apart() {
        // An L of two blocks with its 5 × 5 notch open at a corner: a 2 mm cube 1.5 mm into the
        // notch from both open sides lies inside the L's box, clear of every face.
        let lower = solid([0.0, 0.0, 0.0], [10.0, 5.0, 4.0]);
        let upper = solid([0.0, 5.0, 0.0], [5.0, 10.0, 4.0]);
        let l = Body::new(boolean::union(&lower, &upper).unwrap()).unwrap();
        let cube = block([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert_eq!(
            overlap(&l, &cube, &moved(6.5, 6.5, 1.0)).unwrap(),
            Overlap::Apart
        );
    }

    #[test]
    fn a_swept_block_runs_into_a_wall_after_its_gap_and_never_backing_away() {
        let a = block([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let wall = block([5.0, -1.0, -1.0], [6.0, 3.0, 3.0]);
        let ahead = swept_contact(&a, &wall, Vec3::X, &Sweep::default()).unwrap();
        assert!((ahead - 3.0).abs() < 1e-9, "{ahead}");
        assert_eq!(swept_contact(&a, &wall, -Vec3::X, &Sweep::default()), None);
        // Across, the wall is never in the way.
        assert_eq!(swept_contact(&a, &wall, Vec3::Y, &Sweep::default()), None);
    }

    #[test]
    fn a_swept_shaft_slides_on_through_a_bore_its_own_size_but_not_sideways() {
        let plate = solid([-5.0, -5.0, 0.0], [5.0, 5.0, 4.0]);
        let axis = Frame::new(Vec3::new(0.0, 0.0, -3.0), Vec3::Z, Vec3::X).unwrap();
        let hole = crate::primitive::cylinder(&axis, 2.0, 10.0, "h").unwrap();
        let plate = Body::new(crate::boolean::difference(&plate, &hole).unwrap()).unwrap();
        let mut shaft = crate::primitive::cylinder(&axis, 2.0, 5.0, "s").unwrap();
        shaft.transform(&Transform::rotation(Vec3::ZERO, Vec3::Z, 0.137));
        let shaft = Body::new(shaft).unwrap();
        assert_eq!(
            swept_contact(&shaft, &plate, Vec3::Z, &Sweep::default()),
            None
        );
        assert_eq!(
            swept_contact(&shaft, &plate, -Vec3::Z, &Sweep::default()),
            None
        );
        // Across, and tilted off the axis, it is in the bore's wall at once.
        let hit = swept_contact(&shaft, &plate, Vec3::X, &Sweep::default()).unwrap();
        assert!(hit < 1e-6, "{hit}");
        let tilted = Vec3::new(0.05, 0.0, 1.0);
        assert!(swept_contact(&shaft, &plate, tilted, &Sweep::default()).is_some_and(|t| t < 1e-6));
    }

    #[test]
    fn a_swept_move_counts_only_what_it_adds_to_an_overlap_in_place() {
        let a = block([0.0, 0.0, 0.0], [3.0, 2.0, 2.0]);
        let b = block([2.0, 0.0, 0.0], [6.0, 2.0, 2.0]);
        assert_eq!(swept_contact(&a, &b, -Vec3::X, &Sweep::default()), None);
        let on = swept_contact(&a, &b, Vec3::X, &Sweep::default()).unwrap();
        assert!(on < 1e-9, "{on}");
    }

    #[test]
    fn a_swept_ball_runs_into_a_slanted_block_where_its_face_first_touches_it() {
        // A ball r 1.3 at (1, 1.2, 0.9) moving along d toward a block's face x = 4, turned 0.4 rad
        // about z round (4.5, 1, 1): the face's outward normal is n = (−cos 0.4, −sin 0.4, 0), and
        // the ball touches it once its centre has closed to r, after (n·(c − p) − r) / (−n·d),
        // with p any point of the face; the point it touches lies well inside the face.
        let c = Vec3::new(1.0, 1.2, 0.9);
        let f = Frame::new(c, Vec3::Z, Vec3::X).unwrap();
        let ball = Body::new(crate::primitive::sphere(&f, 1.3, "s").unwrap()).unwrap();
        let turn = 0.4_f64;
        let mut slab = solid([4.0, -2.0, -2.0], [5.0, 4.0, 4.0]);
        slab.transform(&Transform::rotation(
            Vec3::new(4.5, 1.0, 1.0),
            Vec3::Z,
            turn,
        ));
        let slab = Body::new(slab).unwrap();
        let d = Vec3::new(1.0, 0.2, 0.1).unit().unwrap();
        let n = Vec3::new(-turn.cos(), -turn.sin(), 0.0);
        let p = Vec3::new(4.5 - 0.5 * turn.cos(), 1.0 - 0.5 * turn.sin(), 1.0);
        let want = ((c - p).dot(n) - 1.3) / -n.dot(d);
        let t = swept_contact(&ball, &slab, d, &Sweep::default()).unwrap();
        // Within the ball's mesh's chord of its surface, and the rays' spacing across its curve.
        assert!((t - want).abs() < 0.02, "{t} against {want}");
    }

    #[test]
    fn a_host_and_its_insert_joined_move_as_one_and_meet_as_one() {
        // A 10 × 10 × 4 plate with a pocket r 2 down to z = 2, and an insert filling it flush:
        // up the middle a ray leaves the plate where it enters the insert, which one list of
        // crossings would cancel, losing every ray under a 2 mm block 3 mm above.
        let plate = solid([-5.0, -5.0, 0.0], [5.0, 5.0, 4.0]);
        let mouth = Frame::new(Vec3::new(0.0, 0.0, 2.0), Vec3::Z, Vec3::X).unwrap();
        let pocket = crate::primitive::cylinder(&mouth, 2.0, 3.0, "h").unwrap();
        let plate = Body::new(crate::boolean::difference(&plate, &pocket).unwrap()).unwrap();
        let insert = Body::new(crate::primitive::cylinder(&mouth, 2.0, 2.0, "i").unwrap()).unwrap();
        let both = Body::joined(&[&plate, &insert]);
        let above = block([-1.0, -1.0, 7.0], [1.0, 1.0, 8.0]);
        let t = swept_contact(&both, &above, Vec3::Z, &Sweep::default()).unwrap();
        assert!((t - 3.0).abs() < 1e-9, "{t}");
        let beside = block([6.0, -1.0, 0.0], [8.0, 1.0, 9.0]);
        assert_eq!(
            swept_contact(&both, &beside, Vec3::Z, &Sweep::default()),
            None
        );
    }

    #[test]
    fn a_pin_modelled_tight_in_its_hole_slides_along_it_but_not_across() {
        // A pin r 2.1 from z = 1 to 3 in a 4 mm plate's hole r 2: they share a ring 0.1 deep in
        // place. Along its axis the pin's front pushes into the plate as far as its back comes
        // out, so it adds nothing; across, it does.
        let plate = solid([-5.0, -5.0, 0.0], [5.0, 5.0, 4.0]);
        let axis = Frame::new(Vec3::new(0.0, 0.0, -1.0), Vec3::Z, Vec3::X).unwrap();
        let hole = crate::primitive::cylinder(&axis, 2.0, 6.0, "h").unwrap();
        let plate = Body::new(crate::boolean::difference(&plate, &hole).unwrap()).unwrap();
        let low = Frame::new(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, Vec3::X).unwrap();
        let pin = Body::new(crate::primitive::cylinder(&low, 2.1, 2.0, "p").unwrap()).unwrap();
        assert_eq!(
            swept_contact(&pin, &plate, Vec3::Z, &Sweep::default()),
            None
        );
        assert_eq!(
            swept_contact(&pin, &plate, -Vec3::Z, &Sweep::default()),
            None
        );
        assert!(swept_contact(&pin, &plate, Vec3::X, &Sweep::default()).is_some_and(|t| t < 1e-6));
    }

    #[test]
    fn a_hair_of_interference_only_touches_but_a_fifth_of_a_millimetre_runs_in() {
        // A 2 mm cube rising past a block 1 mm above that overlaps its side by 0.03 mm: thinner than
        // twice 0.025, so only touching it then. Overlapping by 0.2, it runs in after 1.
        let a = block([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let how = Sweep {
            inside: 0.025,
            ..Sweep::default()
        };
        let hair = block([1.97, -1.0, 3.0], [4.0, 3.0, 5.0]);
        assert_eq!(swept_contact(&a, &hair, Vec3::Z, &how), None);
        // Any point certainly inside both counts by default.
        assert!(swept_contact(&a, &hair, Vec3::Z, &Sweep::default()).is_some());
        let fifth = block([1.8, -1.0, 3.0], [4.0, 3.0, 5.0]);
        let t = swept_contact(&a, &fifth, Vec3::Z, &how).unwrap();
        assert!((t - 1.0).abs() < 1e-9, "{t}");
    }

    #[test]
    fn a_body_moved_meets_what_its_new_place_meets_without_meshing_again() {
        // A 2 mm cube 3 mm short of a wall along x: slid 1 mm on, it meets the wall after 2;
        // turned a quarter round z about its corner (0, 0, 0) to span x from -2 to 0 and slid
        // 2 mm on, after 3.
        let cube = block([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let wall = block([5.0, -3.0, -1.0], [6.0, 3.0, 3.0]);
        let slid = cube.moved(&Transform::translation(Vec3::new(1.0, 0.0, 0.0)));
        assert_eq!(slid.points().len(), cube.points().len());
        let t = swept_contact(&slid, &wall, Vec3::X, &Sweep::default()).unwrap();
        assert!((t - 2.0).abs() < 1e-9, "{t}");
        let turned = cube.moved(
            &Transform::rotation(Vec3::ZERO, Vec3::Z, std::f64::consts::FRAC_PI_2)
                .then(&Transform::translation(Vec3::new(2.0, 0.0, 0.0))),
        );
        let t = swept_contact(&turned, &wall, Vec3::X, &Sweep::default()).unwrap();
        assert!((t - 3.0).abs() < 1e-9, "{t}");
    }

    #[test]
    fn one_body_against_many_meets_each_after_its_own_gap() {
        // A 2 mm cube along x: a wall 1 mm ahead, one 3 mm ahead above it, and one behind.
        let cube = block([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let near = block([3.0, -1.0, -1.0], [4.0, 3.0, 1.0]);
        let far = block([5.0, -1.0, 1.5], [6.0, 3.0, 3.0]);
        let behind = block([-3.0, -1.0, -1.0], [-2.0, 3.0, 3.0]);
        let got = swept_contacts(
            &[&cube],
            &[&near, &far, &behind],
            Vec3::X,
            &Sweep::default(),
        );
        assert!((got[0].unwrap() - 1.0).abs() < 1e-9, "{got:?}");
        assert!((got[1].unwrap() - 3.0).abs() < 1e-9, "{got:?}");
        assert_eq!(got[2], None);
    }

    #[test]
    fn bodies_moved_together_keep_each_solid_apart() {
        // The pocketed plate and its flush insert as two moving bodies, not joined: up the middle
        // a ray leaves one where it enters the other, and they meet a 2 mm block 3 mm above.
        let plate = solid([-5.0, -5.0, 0.0], [5.0, 5.0, 4.0]);
        let mouth = Frame::new(Vec3::new(0.0, 0.0, 2.0), Vec3::Z, Vec3::X).unwrap();
        let pocket = crate::primitive::cylinder(&mouth, 2.0, 3.0, "h").unwrap();
        let plate = Body::new(crate::boolean::difference(&plate, &pocket).unwrap()).unwrap();
        let insert = Body::new(crate::primitive::cylinder(&mouth, 2.0, 2.0, "i").unwrap()).unwrap();
        let above = block([-1.0, -1.0, 7.0], [1.0, 1.0, 8.0]);
        let got = swept_contacts(&[&plate, &insert], &[&above], Vec3::Z, &Sweep::default());
        assert!((got[0].unwrap() - 3.0).abs() < 1e-9, "{got:?}");
    }
}
