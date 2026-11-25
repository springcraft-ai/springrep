//! Tessellation. Each edge is discretised once and both its faces use those points, so the mesh of
//! a valid solid is watertight by construction. Each face is a constrained Delaunay triangulation
//! of those points in its parameter plane, refined until every triangle is within the deflection
//! limits of the surface (`face`).

pub(crate) mod bvh;
mod cdt;
pub mod display;
mod face;
mod predicates;
mod solid;
#[cfg(test)]
mod tests;

pub(crate) use solid::from_indexed;
pub use solid::solid;

use std::collections::{HashMap, HashSet};
use std::f64::consts::FRAC_PI_2;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cdt::BOX;
use predicates::orient;

use crate::{
    Error, Result,
    geom::{Curve, Surface},
    math::{Point3, Vec2, Vec3},
    threads,
    topo::{Brep, CoedgeId, EdgeId, FaceId, SolidId},
};

#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct MeshOptions {
    /// The largest distance from a triangle or segment to the exact geometry, in mm.
    pub linear: f64,
    /// The largest turn of the normal or tangent across one triangle or segment, in radians.
    pub angular: f64,
}

impl MeshOptions {
    /// The turn allowed across one triangle where only a distance is asked for, in radians.
    pub const ANGULAR: f64 = 0.5;

    pub const fn new(linear: f64, angular: f64) -> MeshOptions {
        MeshOptions { linear, angular }
    }

    /// A mesh within `linear` of the exact geometry, at the kernel's own limit on the turn.
    pub fn within(linear: f64) -> MeshOptions {
        MeshOptions {
            linear,
            angular: MeshOptions::ANGULAR,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Mesh {
    pub positions: Vec<Point3>,
    pub triangles: Vec<[u32; 3]>,
    /// Each face's triangles, as a range of `triangles`.
    pub faces: Vec<(FaceId, Range<usize>)>,
}

/// `opt` with its deflection between a ten-thousandth and a thousandth of `b`'s size: for a mesh
/// that seeds exact searches over `b` (rays, distances), whose triangles must stand near the faces
/// however small the part, and need not be finer than the search's own margin however large.
fn seeding(opt: &MeshOptions, b: &Brep) -> MeshOptions {
    let size = crate::measure::loose_bbox(b).diagonal();
    MeshOptions {
        linear: if size > 0.0 {
            opt.linear.clamp(1e-4 * size, 1e-3 * size)
        } else {
            opt.linear
        },
        angular: opt.angular,
    }
}

/// The deflections a seed mesh is tried at, finest first: the mesher gives up on a very fine
/// chord over a large part.
const SEEDS: [MeshOptions; 3] = [
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

/// An arena's mesh for seeding exact searches, and the deflections it was built to.
#[derive(Debug)]
pub(crate) struct SeedMesh {
    pub mesh: Mesh,
    pub options: MeshOptions,
}

/// The mesh that seeds exact searches over `b`, built once for as long as `b` stays as it is.
pub(crate) fn seed_mesh(b: &Brep) -> Result<Arc<SeedMesh>> {
    let build = || {
        let mut err = None;
        for options in SEEDS.iter().map(|o| seeding(o, b)) {
            match tessellate(b, &options) {
                Ok(mesh) => return Ok(Arc::new(SeedMesh { mesh, options })),
                Err(e) => err = Some(e),
            }
        }
        Err(err.expect("a mesh was tried"))
    };
    b.arena().seed.get_or_init(build).clone()
}

/// The parameters at which an edge's curve is sampled.
pub fn edge_params(curve: &Curve, t0: f64, t1: f64, opt: &MeshOptions) -> Vec<f64> {
    match curve {
        Curve::Line(_) => vec![t0, t1],
        Curve::Circle(c) => {
            let chord = if opt.linear < c.radius {
                2.0 * (1.0 - opt.linear / c.radius).acos()
            } else {
                f64::INFINITY
            };
            // A chord no longer than the deflection cannot stray by more, so the turn alone never
            // splits one: a hair-wide circle at a fine angle would put its samples within the
            // tolerance of each other.
            let step = chord
                .min(opt.angular.max((opt.linear / c.radius).min(FRAC_PI_2)))
                .max(1e-4);
            let n = ((t1 - t0) / step).ceil().max(1.0) as usize;
            (0..=n)
                .map(|i| t0 + (t1 - t0) * i as f64 / n as f64)
                .collect()
        }
        _ => {
            let mut cuts = vec![t0];
            // A knot that rounding puts beside an end would crowd samples onto the end's vertex.
            let margin = 1e-9 * (t1 - t0);
            cuts.extend(
                curve
                    .breakpoints(t0, t1)
                    .into_iter()
                    .filter(|&k| k > t0 + margin && k < t1 - margin),
            );
            cuts.push(t1);
            let mut out = vec![t0];
            for w in cuts.windows(2) {
                for k in 0..4 {
                    let a = w[0] + (w[1] - w[0]) * k as f64 / 4.0;
                    let b = w[0] + (w[1] - w[0]) * (k + 1) as f64 / 4.0;
                    subdivide(curve, a, b, opt, 0, &mut out);
                }
            }
            coarsen(curve, &out, opt)
        }
    }
}

/// The fewest of the `dense` parameters whose chords stay within the deflection limits, tested
/// at every dense parameter and the middles between them. A spline has a break at every knot, and
/// an intersection curve has many knots, so sampling each knot span alone is far too fine.
fn coarsen(curve: &Curve, dense: &[f64], opt: &MeshOptions) -> Vec<f64> {
    let n = dense.len();
    if n <= 2 {
        return dense.to_vec();
    }
    let pts: Vec<Point3> = dense.iter().map(|&t| curve.eval(t)).collect();
    let mids: Vec<Point3> = dense
        .windows(2)
        .map(|w| curve.eval(0.5 * (w[0] + w[1])))
        .collect();
    let tangents: Vec<Vec3> = dense.iter().map(|&t| curve.d1(t).1).collect();
    let fits = |i: usize, j: usize| {
        if tangents[i].angle(tangents[j]) > opt.angular {
            return false;
        }
        let (a, b) = (pts[i], pts[j]);
        (i..j).all(|k| {
            (k == i || segment_dist(pts[k], a, b) <= opt.linear)
                && segment_dist(mids[k], a, b) <= opt.linear
        })
    };
    let mut out = vec![dense[0]];
    let mut i = 0;
    while i < n - 1 {
        let mut j = i + 1;
        while j + 1 < n && fits(i, j + 1) {
            j += 1;
        }
        out.push(dense[j]);
        i = j;
    }
    out
}

fn segment_dist(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    let s = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    p.dist(a + ab * s)
}

fn subdivide(c: &Curve, a: f64, b: f64, opt: &MeshOptions, depth: u32, out: &mut Vec<f64>) {
    let (pa, da) = c.d1(a);
    let (pb, db) = c.d1(b);
    let chord = pb - pa;
    let dev = [0.25, 0.5, 0.75]
        .iter()
        .map(|f| {
            let q = c.eval(a + (b - a) * f);
            let along = pa + chord * *f;
            q.dist(along)
        })
        .fold(0.0, f64::max);
    let turn = da.angle(db);
    let long = chord.norm() > opt.linear;
    if depth < 24 && (dev > opt.linear || (long && turn > opt.angular)) {
        let m = 0.5 * (a + b);
        subdivide(c, a, m, opt, depth + 1, out);
        subdivide(c, m, b, opt, depth + 1, out);
    } else {
        out.push(b);
    }
}

/// The most edges of a solid of flat faces and straight edges meshed on the calling thread alone.
const PLAIN_EDGES: usize = 128;

/// Whether `b` is flat faces and straight edges, and few of them: an edge is then its two ends,
/// a face the triangles of its outline, and all of it less work than starting a thread.
fn plain(b: &Brep) -> bool {
    b.edge_ids().count() <= PLAIN_EDGES
        && b.face_ids()
            .all(|f| matches!(b.face(f).surface, Surface::Plane(_)))
        && b.edge_ids()
            .all(|e| matches!(b.edge(e).curve, Some(Curve::Line(_))))
}

/// Tessellates every face of `b` within `opt`.
pub fn tessellate(b: &Brep, opt: &MeshOptions) -> Result<Mesh> {
    let threads = if plain(b) { 1 } else { usize::MAX };
    tessellate_on(threads, b, opt)
}

/// [`tessellate`] on at most `threads` threads: the mesh is the same on any number.
fn tessellate_on(threads: usize, b: &Brep, opt: &MeshOptions) -> Result<Mesh> {
    let positive = |x: f64| x > 0.0 && x.is_finite();
    if !positive(opt.linear) || !positive(opt.angular) {
        return Err(Error::InvalidInput(format!(
            "mesh deflections must be positive and finite: linear {}, angular {}",
            opt.linear, opt.angular
        )));
    }
    let mut mesh = Mesh::default();
    let vgid: Vec<u32> = b
        .vertex_ids()
        .map(|v| {
            mesh.positions.push(b.vertex(v).point);
            (mesh.positions.len() - 1) as u32
        })
        .collect();
    let mut uses: Vec<Vec<CoedgeId>> = vec![Vec::new(); b.edge_ids().count()];
    for c in b.coedge_ids() {
        uses[b.coedge(c).edge.index()].push(c);
    }
    let edges: Vec<EdgeId> = b.edge_ids().collect();
    let params = par_map(threads, edges.len(), |i| {
        let edge = b.edge(edges[i]);
        let (t0, t1) = edge.range;
        match &edge.curve {
            Some(c) => fit_faces(b, c, edge_params(c, t0, t1, opt), &uses[i], opt),
            None => {
                let n = ((t1 - t0) / opt.angular).ceil().max(1.0) as usize;
                (0..=n)
                    .map(|k| t0 + (t1 - t0) * k as f64 / n as f64)
                    .collect()
            }
        }
    });
    let mut params = params;
    // Two edges between one pair of vertices, each drawn as one chord, would draw one segment
    // that all four of their faces use.
    let mut by_ends: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, &e) in edges.iter().enumerate() {
        let edge = b.edge(e);
        if edge.curve.is_some() && params[i].len() == 2 && edge.start != edge.end {
            let (a, c) = (edge.start.index(), edge.end.index());
            by_ends.entry((a.min(c), a.max(c))).or_default().push(i);
        }
    }
    for group in by_ends.values().filter(|g| g.len() > 1) {
        for &i in group {
            let ts = &mut params[i];
            ts.insert(1, 0.5 * (ts[0] + ts[1]));
        }
    }
    let mut samples: Vec<Vec<(f64, u32)>> = Vec::with_capacity(edges.len());
    for (&e, ts) in edges.iter().zip(params) {
        let edge = b.edge(e);
        let last = ts.len() - 1;
        let s = ts
            .iter()
            .enumerate()
            .map(|(k, &t)| {
                let gid = match &edge.curve {
                    _ if k == 0 => vgid[edge.start.index()],
                    _ if k == last => vgid[edge.end.index()],
                    Some(c) => {
                        mesh.positions.push(c.eval(t));
                        (mesh.positions.len() - 1) as u32
                    }
                    None => vgid[edge.start.index()],
                };
                (t, gid)
            })
            .collect();
        samples.push(s);
    }
    uncross(threads, b, &edges, &mut samples, &mut mesh.positions);
    drop_unresolved(b, &edges, &mut samples);
    faces(threads, b, opt, &samples, &mut mesh)?;
    Ok(mesh)
}

/// Splits boundary segments that cross another of the same face's segments in its parameter
/// plane. Two edges that leave nearly one point at a small angle can have chords that cross where
/// the curves do not; halving the curved one's chords near there separates them, a few samples
/// for each such place. Segments that still cross after the rounds are left for the face to
/// refuse, and so are those of edges whose images overlap, which halving never parts: it would
/// double their samples every round. The rounds stop once they have added as many samples as
/// the edges had, so the work is bounded whatever the boundary.
fn uncross(
    threads: usize,
    b: &Brep,
    edges: &[EdgeId],
    samples: &mut [Vec<(f64, u32)>],
    positions: &mut Vec<Point3>,
) {
    let faces: Vec<FaceId> = b.face_ids().collect();
    let count = |samples: &[Vec<(f64, u32)>]| samples.iter().map(Vec::len).sum::<usize>();
    let budget = 2 * count(samples);
    for _ in 0..32 {
        let before = count(samples);
        if before >= budget {
            return;
        }
        let found = par_map(threads, faces.len(), |i| {
            crossing_segments(b, faces[i], samples)
        });
        let split: HashSet<(usize, usize)> = found.into_iter().flatten().collect();
        if split.is_empty() {
            return;
        }
        let mut by_edge: HashMap<usize, Vec<usize>> = HashMap::new();
        for (e, k) in split {
            by_edge.entry(e).or_default().push(k);
        }
        for (e, ks) in by_edge {
            let Some(curve) = &b.edge(edges[e]).curve else {
                continue;
            };
            // A chord no longer than the edge's tolerance (an imported edge's own, else the linear
            // one) cannot be told apart from one it crosses by halving it: its image in the face
            // is known no better, and below the linear tolerance a curve's parameter-plane points
            // are rounding, whose wobble halving would chase down to coincident samples.
            let within = b.edge_tolerance(edges[e]).max(crate::tol::LINEAR);
            let resolved = |a: f64, z: f64| curve.eval(a).dist(curve.eval(z)) > within;
            let old = std::mem::take(&mut samples[e]);
            let mut new = Vec::with_capacity(old.len() + ks.len());
            for k in 0..old.len() {
                new.push(old[k]);
                if ks.contains(&k) && k + 1 < old.len() && resolved(old[k].0, old[k + 1].0) {
                    let t = 0.5 * (old[k].0 + old[k + 1].0);
                    positions.push(curve.eval(t));
                    new.push((t, (positions.len() - 1) as u32));
                }
            }
            samples[e] = new;
        }
        // Every crossing left is between chords too short to halve: the next round would find
        // the same.
        if count(samples) == before {
            return;
        }
    }
}

/// Drops the samples of an imported edge with a local tolerance that one of its faces sees at the
/// same parameters as a neighbour: the edge misses that face by up to its tolerance, and where it
/// runs past a pole, a stretch of it projects to one point. Both faces drop the sample, so they
/// still share every chord.
fn drop_unresolved(b: &Brep, edges: &[EdgeId], samples: &mut [Vec<(f64, u32)>]) {
    for (i, &e) in edges.iter().enumerate() {
        if b.edge_tolerance(e) <= crate::tol::LINEAR || samples[i].len() < 3 {
            continue;
        }
        let ts: Vec<f64> = samples[i].iter().map(|x| x.0).collect();
        let images: Vec<Vec<Vec2>> = b
            .coedge_ids()
            .filter(|&c| b.coedge(c).edge == e)
            .map(|c| {
                let reversed = b.coedge(c).reversed;
                let order: Vec<f64> = if reversed {
                    ts.iter().rev().copied().collect()
                } else {
                    ts.clone()
                };
                let mut uvs = b.coedge_uvs(c, &order);
                if reversed {
                    uvs.reverse();
                }
                uvs
            })
            .collect();
        let last = ts.len() - 1;
        let same = |a: usize, z: usize| {
            images.iter().any(|uvs| {
                let near = 1e-12 * (1.0 + uvs[a].x.abs() + uvs[a].y.abs());
                (uvs[a] - uvs[z]).norm() <= near
            })
        };
        let mut kept = vec![0];
        for k in 1..last {
            if !same(*kept.last().unwrap_or(&0), k) && !same(k, last) {
                kept.push(k);
            }
        }
        kept.push(last);
        if kept.len() < ts.len() {
            samples[i] = kept.into_iter().map(|k| samples[i][k]).collect();
        }
    }
}

/// Where one coedge of a loop begins among a loop's points, and the parameters it ends at.
pub(super) type Run = (usize, Vec2);

/// Lays a face's boundary out as one figure in the parameter plane of a periodic surface. Each
/// coedge's parameters are seeded on their own, and may lie a whole period from the coedge before
/// it: a cylinder's seam at −π under a rim that runs from π to 3π. Where every loop closes once
/// its coedges are laid end to end, they are laid so, and each other loop is moved by whole
/// periods into the largest. A loop that winds round the surface does not close, and a face with
/// one is left as it is seeded. A loop already inside the largest stays where it is. `runs` gives
/// each loop's coedges in order.
pub(super) fn one_sheet<T>(
    periods: (Option<f64>, Option<f64>),
    loops: &mut [Vec<(Vec2, T)>],
    runs: &[Vec<Run>],
) {
    if periods.0.is_none() && periods.1.is_none() {
        return;
    }
    let whole = |d: f64, p: Option<f64>| p.map_or(0.0, |p| (d / p).round() * p);
    let turns = |d: Vec2| Vec2::new(whole(d.x, periods.0), whole(d.y, periods.1));
    let mut shifts: Vec<Vec<Vec2>> = Vec::with_capacity(loops.len());
    for (pts, run) in loops.iter().zip(runs) {
        let mut shift = Vec2::ZERO;
        let mut each = Vec::with_capacity(run.len());
        let mut end: Option<Vec2> = None;
        for &(at, last) in run {
            let Some(&(first, _)) = pts.get(at) else {
                return;
            };
            if let Some(e) = end {
                shift = turns(e - first);
            }
            each.push(shift);
            end = Some(last + shift);
        }
        // Back at its start, to well within a period: else the loop winds round the surface.
        let (Some(e), Some(&(first, _))) = (end, pts.first()) else {
            return;
        };
        if turns(e - first) != Vec2::ZERO {
            return;
        }
        shifts.push(each);
    }
    for ((pts, run), each) in loops.iter_mut().zip(runs).zip(&shifts) {
        for (k, &(at, _)) in run.iter().enumerate() {
            let to = run.get(k + 1).map_or(pts.len(), |next| next.0);
            for p in &mut pts[at..to] {
                p.0 = p.0 + each[k];
            }
        }
    }
    let area = |pts: &[(Vec2, T)]| -> f64 {
        let n = pts.len();
        (0..n)
            .map(|i| {
                let (a, b) = (pts[i].0, pts[(i + 1) % n].0);
                a.x * b.y - b.x * a.y
            })
            .sum::<f64>()
            .abs()
    };
    let Some(outer) = (0..loops.len()).max_by(|&i, &j| area(&loops[i]).total_cmp(&area(&loops[j])))
    else {
        return;
    };
    let ring: Vec<Vec2> = loops[outer].iter().map(|p| p.0).collect();
    let inside = |q: Vec2| -> bool {
        let n = ring.len();
        let mut odd = false;
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            if (a.y > q.y) != (b.y > q.y) && q.x < a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x) {
                odd = !odd;
            }
        }
        odd
    };
    for (i, pts) in loops.iter_mut().enumerate() {
        if i == outer || pts.is_empty() {
            continue;
        }
        // A point of the loop's own, not on the outer loop's boundary: the middle of its points.
        let n = pts.len() as f64;
        let mid = pts.iter().fold(Vec2::ZERO, |s, p| s + p.0) * (1.0 / n);
        let steps = |p: Option<f64>| -> Vec<f64> {
            match p {
                Some(p) => (-3..=3).map(|k| f64::from(k) * p).collect(),
                None => vec![0.0],
            }
        };
        let mut best = None;
        for du in steps(periods.0) {
            for dv in steps(periods.1) {
                let d = Vec2::new(du, dv);
                if inside(mid + d) && best.is_none_or(|b: Vec2| d.norm() < b.norm()) {
                    best = Some(d);
                }
            }
        }
        if let Some(d) = best.filter(|d| *d != Vec2::ZERO) {
            for p in pts.iter_mut() {
                p.0 = p.0 + d;
            }
        }
    }
}

/// Moves each parameter-plane image of a mesh point onto an earlier image of the same point when
/// rounding alone parts them, so the face is meshed through one point there and not two a hair
/// apart. Images a period or a pole's span apart are far more than this apart.
pub(super) fn snap_repeats(pts: &mut [(Vec2, u32)]) {
    let (mut lo, mut hi) = (
        Vec2::new(f64::INFINITY, f64::INFINITY),
        Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for (uv, _) in pts.iter() {
        lo = Vec2::new(lo.x.min(uv.x), lo.y.min(uv.y));
        hi = Vec2::new(hi.x.max(uv.x), hi.y.max(uv.y));
    }
    let near = 1e-9 * (hi - lo).norm();
    let mut seen: HashMap<u32, Vec<Vec2>> = HashMap::new();
    for (uv, g) in pts.iter_mut() {
        let images = seen.entry(*g).or_default();
        match images.iter().find(|q| (**q - *uv).norm() <= near) {
            Some(&q) => *uv = q,
            None => images.push(*uv),
        }
    }
}

/// The (edge, segment) pairs of a face's boundary segments that cross or touch another of its
/// segments other than at a shared end, every segment of a loop too short to bound an area, and
/// every curved segment with another loop point between it and its curve, which a coarse
/// deflection leaves on the wrong side of the chord.
fn crossing_segments(b: &Brep, f: FaceId, samples: &[Vec<(f64, u32)>]) -> Vec<(usize, usize)> {
    // Each loop's points as the face mesher takes them: every coedge's samples but its last.
    let mut segs: Vec<(Vec2, Vec2, usize, usize)> = Vec::new();
    let mut neighbours: Vec<(usize, usize)> = Vec::new();
    let mut out = Vec::new();
    let mut loops: Vec<Vec<(Vec2, usize, usize)>> = Vec::new();
    let mut points: Vec<(Vec2, u32)> = Vec::new();
    let mut mids: Vec<Option<Vec2>> = Vec::new();
    let mut sheet: Vec<Vec<(Vec2, usize)>> = Vec::new();
    let mut runs: Vec<Vec<Run>> = Vec::new();
    for &lp in &b.face(f).loops {
        let mut pts: Vec<(Vec2, usize, usize)> = Vec::new();
        let mut run: Vec<Run> = Vec::new();
        for &co in &b.lp(lp).coedges {
            let c = b.coedge(co);
            let e = c.edge.index();
            let n = samples[e].len();
            let mut ts: Vec<f64> = samples[e].iter().map(|x| x.0).collect();
            if c.reversed {
                ts.reverse();
            }
            let uvs = b.coedge_uvs(co, &ts);
            run.push((pts.len(), uvs[n - 1]));
            let curved = matches!(&b.edge(c.edge).curve, Some(cv) if !matches!(cv, Curve::Line(_)));
            // Interleaved with the samples, so a periodic parameter continues as it does there.
            let halves: Vec<Option<Vec2>> = if curved {
                let mut both = Vec::with_capacity(2 * n);
                for w in ts.windows(2) {
                    both.push(w[0]);
                    both.push(0.5 * (w[0] + w[1]));
                }
                both.push(ts[n - 1]);
                let uv2 = b.coedge_uvs(co, &both);
                (0..n - 1).map(|k| Some(uv2[2 * k + 1])).collect()
            } else {
                vec![None; n - 1]
            };
            for (k, &uv) in uvs.iter().enumerate().take(n - 1) {
                let (seg, at) = if c.reversed {
                    (n - 2 - k, n - 1 - k)
                } else {
                    (k, k)
                };
                pts.push((uv, e, seg));
                points.push((uv, samples[e][at].1));
                mids.push(halves[k]);
            }
        }
        sheet.push(
            (0..pts.len())
                .map(|k| (pts[k].0, points.len() - pts.len() + k))
                .collect(),
        );
        runs.push(run);
        loops.push(pts);
    }
    let surface = &b.face(f).surface;
    one_sheet((surface.u_period(), surface.v_period()), &mut sheet, &runs);
    for &(uv, k) in sheet.iter().flatten() {
        if let Some(mid) = &mut mids[k] {
            *mid = *mid + (uv - points[k].0);
        }
        points[k].0 = uv;
    }
    snap_repeats(&mut points);
    let mut snapped = points.into_iter().map(|p| p.0);
    for mut pts in loops {
        for p in &mut pts {
            p.0 = snapped.next().expect("one snapped point per loop point");
        }
        let first = segs.len();
        let m = pts.len();
        if m < 3 {
            out.extend(pts.iter().map(|&(_, e, k)| (e, k)));
        }
        for i in 0..m {
            let (a, e, k) = pts[i];
            segs.push((a, pts[(i + 1) % m].0, e, k));
            neighbours.push((first + (i + m - 1) % m, first + (i + 1) % m));
        }
    }
    let lo = |s: &(Vec2, Vec2, usize, usize)| s.0.x.min(s.1.x);
    let mut order: Vec<usize> = (0..segs.len()).collect();
    order.sort_by(|&i, &j| lo(&segs[i]).total_cmp(&lo(&segs[j])));
    for (oi, &i) in order.iter().enumerate() {
        let (a, c, ..) = segs[i];
        if a == c {
            continue;
        }
        let hi = a.x.max(c.x);
        for &j in &order[oi + 1..] {
            let (p, q, ..) = segs[j];
            if lo(&segs[j]) > hi {
                break;
            }
            if p == q || neighbours[i].0 == j || neighbours[i].1 == j {
                continue;
            }
            if a.y.max(c.y) < p.y.min(q.y) || p.y.max(q.y) < a.y.min(c.y) {
                continue;
            }
            if [p, q].contains(&a) || [p, q].contains(&c) {
                continue;
            }
            let (o1, o2) = (orient(a, c, p), orient(a, c, q));
            let (o3, o4) = (orient(p, q, a), orient(p, q, c));
            if o1 * o2 <= 0.0 && o3 * o4 <= 0.0 {
                out.push((segs[i].2, segs[i].3));
                out.push((segs[j].2, segs[j].3));
            }
        }
    }
    let mut by_x: Vec<Vec2> = segs.iter().map(|s| s.0).collect();
    by_x.sort_by(|p, q| p.x.total_cmp(&q.x));
    for (i, &(a, c, e, k)) in segs.iter().enumerate() {
        let Some(m) = mids[i] else { continue };
        let turn = orient(a, c, m);
        if turn == 0.0 {
            continue;
        }
        let (x0, x1) = (a.x.min(c.x).min(m.x), a.x.max(c.x).max(m.x));
        let from = by_x.partition_point(|p| p.x < x0);
        let inside = by_x[from..]
            .iter()
            .take_while(|p| p.x <= x1)
            .filter(|&&p| p != a && p != c)
            .any(|&p| {
                // On the chord counts: a loop whose points all lie on one line bounds nothing.
                orient(a, c, p) * turn >= 0.0
                    && orient(c, m, p) * turn > 0.0
                    && orient(m, a, p) * turn > 0.0
            });
        if inside {
            out.push((e, k));
        }
    }
    out
}

/// Adds samples to an edge until, on each face that uses it, the surface over the parameter-plane
/// chord between neighbouring samples stays within the linear deflection of the curve. A face's
/// triangles along its boundary follow those chords, and the face cannot split them, since the
/// next face shares them.
fn fit_faces(
    b: &Brep,
    c: &Curve,
    mut ts: Vec<f64>,
    uses: &[CoedgeId],
    opt: &MeshOptions,
) -> Vec<f64> {
    for _ in 0..24 {
        let mut split = vec![false; ts.len().saturating_sub(1)];
        for &co in uses {
            let reversed = b.coedge(co).reversed;
            let mut order = ts.clone();
            if reversed {
                order.reverse();
            }
            let uvs = b.coedge_uvs(co, &order);
            let surface = &b.face(b.coedge_face(co)).surface;
            let n = ts.len() - 1;
            for k in 0..n {
                let mid = (uvs[k] + uvs[k + 1]) * 0.5;
                let (ta, tb) = (order[k], order[k + 1]);
                let on_curve = c.eval(0.5 * (ta + tb));
                // A step shorter than the deflection cannot stray by more than it; one that
                // still seems to stray is a parameter-plane jump that splitting cannot mend.
                let short = c.eval(ta).dist(c.eval(tb)) <= opt.linear;
                // A straight edge along a twisted surface's ruling meets the surface everywhere,
                // but the triangles on it can only follow the surface's turn if it is sampled.
                // The middle counts too: a loft side's normal can turn away and back between
                // two corners that face the same way.
                // Each end's normal a hair in from the end: on a cone's apex the radius is zero
                // to rounding, and the normal there takes its sign from that radius.
                let inward = |uv: Vec2| {
                    let at = uv + (mid - uv) * 1e-6;
                    surface.normal(at.x, at.y)
                };
                let (na, nm) = (inward(uvs[k]), surface.normal(mid.x, mid.y));
                let nb = inward(uvs[k + 1]);
                let turn = na.angle(nb).max(na.angle(nm)).max(nm.angle(nb));
                if !short
                    && (surface.eval(mid.x, mid.y).dist(on_curve) > opt.linear
                        || turn > opt.angular)
                {
                    split[if reversed { n - 1 - k } else { k }] = true;
                }
            }
        }
        if !split.contains(&true) {
            break;
        }
        let mut next = Vec::with_capacity(2 * ts.len());
        for k in 0..ts.len() - 1 {
            next.push(ts[k]);
            if split[k] {
                next.push(0.5 * (ts[k] + ts[k + 1]));
            }
        }
        next.push(ts[ts.len() - 1]);
        ts = next;
    }
    ts
}

/// Meshes every face, on as many threads as the machine offers and at most `threads`, and appends
/// them to `mesh`.
fn faces(
    threads: usize,
    b: &Brep,
    opt: &MeshOptions,
    samples: &[Vec<(f64, u32)>],
    mesh: &mut Mesh,
) -> Result<()> {
    let ids: Vec<FaceId> = b.face_ids().collect();
    let positions = &mesh.positions;
    let results = par_map(threads, ids.len(), |i| {
        face::mesh_face(b, ids[i], opt, samples, positions)
    });
    let mut meshes = Vec::with_capacity(ids.len());
    for (r, &f) in results.into_iter().zip(&ids) {
        let named = |m: String| {
            let names: Vec<String> = b.face(f).names.iter().map(ToString::to_string).collect();
            format!("{m} (face {} {})", f.index(), names.join(";"))
        };
        meshes.push(r.map_err(|e| match e {
            Error::Numerical(m) => Error::Numerical(named(m)),
            Error::Unsupported(m) => Error::Unsupported(named(m)),
            other => other,
        })?);
    }
    split_shared_chords(b, &ids, &mut meshes);
    for (fm, &f) in meshes.iter().zip(&ids) {
        let start = mesh.triangles.len();
        let local: Vec<u32> = fm
            .gid
            .iter()
            .zip(&fm.pos)
            .enumerate()
            .map(|(i, (g, p))| match g {
                Some(g) => *g,
                None if i < BOX as usize => u32::MAX,
                None => {
                    mesh.positions.push(*p);
                    (mesh.positions.len() - 1) as u32
                }
            })
            .collect();
        let reversed = b.face(f).reversed;
        let tris: Vec<[u32; 3]> = fm
            .triangles()
            .map(|(_, t)| t.map(|i| local[i as usize]))
            .filter(|g| g[0] != g[1] && g[1] != g[2] && g[0] != g[2])
            .collect();
        for g in unfold(tris) {
            mesh.triangles
                .push(if reversed { [g[0], g[2], g[1]] } else { g });
        }
        mesh.faces.push((f, start..mesh.triangles.len()));
    }
    Ok(())
}

/// `tris` less every pair of triangles on the same three points facing opposite ways. Beside a pole,
/// whose copies along the parameter plane are one point, a point can be joined to the pole on both
/// sides of one chord: the two triangles cover nothing, and each of their edges would be used twice
/// too often.
fn unfold(tris: Vec<[u32; 3]>) -> Vec<[u32; 3]> {
    // Each triangle turned to start at its least point: a folded pair then reads a, b, c and a, c, b.
    let turned = |g: [u32; 3]| {
        let k = (0..3).min_by_key(|&k| g[k]).unwrap_or(0);
        [g[k], g[(k + 1) % 3], g[(k + 2) % 3]]
    };
    let mut open: HashMap<[u32; 3], Vec<usize>> = HashMap::new();
    let mut folded = vec![false; tris.len()];
    for (i, &g) in tris.iter().enumerate() {
        let [a, b, c] = turned(g);
        if let Some(j) = open.get_mut(&[a, c, b]).and_then(Vec::pop) {
            folded[i] = true;
            folded[j] = true;
        } else {
            open.entry([a, b, c]).or_default().push(i);
        }
    }
    tris.into_iter()
        .zip(folded)
        .filter_map(|(g, f)| (!f).then_some(g))
        .collect()
}

/// An interior edge between two boundary points could be the very segment another face draws
/// between them, as a chord or as a boundary segment, which would make that mesh edge
/// non-manifold; a point of the face's own in its middle keeps each face's interior edges its own.
/// Uses are counted over triangles with three distinct points, since the copies of a pole or seam
/// point in one face draw the same segment more than once without using it more than twice.
fn split_shared_chords(b: &Brep, ids: &[FaceId], meshes: &mut [face::FaceMesh]) {
    let key = |fm: &face::FaceMesh, a: u32, c: u32| {
        let (ga, gc) = (fm.gid[a as usize]?, fm.gid[c as usize]?);
        (ga != gc).then(|| (ga.min(gc), ga.max(gc)))
    };
    let distinct = |fm: &face::FaceMesh, tri: [u32; 3]| {
        let g = tri.map(|v| fm.gid[v as usize]);
        (0..3).all(|k| g[k].is_none() || g[k] != g[(k + 1) % 3])
    };
    let mut uses: HashMap<(u32, u32), u32> = HashMap::new();
    for fm in meshes.iter() {
        for (_, tri) in fm.triangles().filter(|&(_, tri)| distinct(fm, tri)) {
            for k in 0..3 {
                if let Some(g) = key(fm, tri[k], tri[(k + 1) % 3]) {
                    *uses.entry(g).or_default() += 1;
                }
            }
        }
    }
    for (fm, &f) in meshes.iter_mut().zip(ids) {
        let mut todo = Vec::new();
        for (t, tri) in fm.triangles() {
            for k in 0..3 {
                let (a, c) = (tri[k], tri[(k + 1) % 3]);
                if a > c || fm.cdt.fixed[t as usize][k] {
                    continue;
                }
                if key(fm, a, c).is_some_and(|g| uses.get(&g).is_some_and(|&n| n > 2)) {
                    todo.push((a, c));
                }
            }
        }
        let surf = &b.face(f).surface;
        for (a, c) in todo {
            if let Some((t, k)) = fm.cdt.find_edge(a, c) {
                fm.split_chord(surf, t, k);
            }
        }
    }
}

/// `f(0..n)` in order, spread over the machine's threads, at most `threads` of them.
fn par_map<T: Send>(threads: usize, n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let runs = threads::spread(n.min(threads), || {
        let mut mine = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break mine;
            }
            mine.push((i, f(i)));
        }
    });
    let mut parts: Vec<(usize, T)> = runs.into_iter().flatten().collect();
    parts.sort_by_key(|p| p.0);
    parts.into_iter().map(|p| p.1).collect()
}

/// How a solid's part of a mesh fails to close, if it does.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct MeshReport {
    pub triangles: usize,
    pub boundary_edges: usize,
    pub nonmanifold_edges: usize,
    /// Edges more than two triangles use, as many each way: where pieces of the solid touch.
    pub shared_edges: usize,
    pub inconsistent_orientation_edges: usize,
    pub faces_without_mesh: usize,
    /// Triangles with a corner within the linear tolerance of the line through the other two.
    pub degenerate_triangles: usize,
    pub volume: f64,
}

impl MeshReport {
    /// Every edge used by as many triangles one way as the other, and every face meshed.
    pub fn watertight(&self) -> bool {
        self.boundary_edges == 0
            && self.nonmanifold_edges == 0
            && self.inconsistent_orientation_edges == 0
            && self.faces_without_mesh == 0
    }
}

pub fn report(b: &Brep, mesh: &Mesh, solid: SolidId) -> MeshReport {
    let faces: HashSet<FaceId> = b.solid_faces(solid).into_iter().collect();
    let mut r = MeshReport::default();
    let mut directed: HashMap<(u32, u32), usize> = HashMap::new();
    for (f, range) in &mesh.faces {
        if !faces.contains(f) {
            continue;
        }
        if range.is_empty() {
            r.faces_without_mesh += 1;
        }
        for t in &mesh.triangles[range.clone()] {
            r.triangles += 1;
            let p = t.map(|i| mesh.positions[i as usize]);
            let n: Vec3 = (p[1] - p[0]).cross(p[2] - p[0]);
            let longest = p[0].dist(p[1]).max(p[1].dist(p[2])).max(p[2].dist(p[0]));
            if n.norm() <= crate::tol::LINEAR * longest {
                r.degenerate_triangles += 1;
            }
            r.volume += p[0].dot(p[1].cross(p[2])) / 6.0;
            for k in 0..3 {
                *directed.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
    }
    let mut seen = HashSet::new();
    for (&(a, c), &n_ab) in &directed {
        let k = if a < c { (a, c) } else { (c, a) };
        if !seen.insert(k) {
            continue;
        }
        let n_ba = directed.get(&(c, a)).copied().unwrap_or(0);
        match n_ab + n_ba {
            1 => r.boundary_edges += 1,
            2 if n_ab == 1 => {}
            2 => r.inconsistent_orientation_edges += 1,
            _ if n_ab == n_ba => r.shared_edges += 1,
            _ => r.nonmanifold_edges += 1,
        }
    }
    r
}
