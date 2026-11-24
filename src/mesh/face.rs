//! One face's mesh: a constrained Delaunay triangulation of its boundary samples in a scaled
//! parameter plane, refined by inserting points where a triangle strays from the surface.
//!
//! The plane is scaled per face so that one unit along `u` or `v` is about the longest step the
//! deflection limits allow in that direction, so a cylinder's triangles run long along its
//! straight lines and short round it. A triangle is refined at its circumcentre in that plane, or
//! at its worst sample when the circumcentre is outside the face.

use std::collections::{HashMap, HashSet};
use std::f64::consts::FRAC_PI_2;

use super::{
    MeshOptions,
    cdt::{BOX, Cdt, Located, NONE},
    predicates::{circumcenter, orient},
};
use crate::{
    Error, Result,
    geom::Surface,
    math::{Point3, Vec2, Vec3},
    tol,
    topo::{Brep, FaceId},
};

/// The most vertices one face's mesh may have before meshing gives up.
const MAX_FACE_POINTS: usize = 8_000_000;

/// In a curved face, a triangle whose smallest angle in the scaled plane is under about 20° is
/// refined while its circumcircle is wider than one unit.
const MIN_ANGLE_SINE: f64 = 0.34;

/// A refinement point closer than this fraction of its triangle's longest side to one of the
/// triangle's corners, or in a triangle no taller than it, is not inserted.
const CROWDED: f64 = 1e-3;

pub(super) struct FaceMesh {
    pub cdt: Cdt,
    pub uv: Vec<Vec2>,
    pub pos: Vec<Point3>,
    /// The global mesh point of each boundary sample; interior points have none.
    pub gid: Vec<Option<u32>>,
    pub inside: Vec<bool>,
    /// At a pole, the parameter-plane direction the pole spans.
    pole: Vec<Option<Vec2>>,
    normal: Vec<Option<Vec3>>,
    scale: Vec2,
    curved: bool,
}

/// A face's boundary samples, loop by loop, as (parameters, global point).
type Loops = Vec<Vec<(Vec2, u32)>>;

pub(super) fn mesh_face(
    b: &Brep,
    f: FaceId,
    opt: &MeshOptions,
    samples: &[Vec<(f64, u32)>],
    positions: &[Point3],
) -> Result<FaceMesh> {
    let face = b.face(f);
    let surf = &face.surface;
    let mut loops: Loops = Vec::new();
    let mut poles: HashMap<u32, Vec2> = HashMap::new();
    let mut runs: Vec<Vec<super::Run>> = Vec::new();
    // A point the face holds that is already a sample of its boundary (one on its seam) is
    // meshed there, and not again as a loop of its own.
    let on_boundary: HashSet<u32> = face
        .loops
        .iter()
        .filter(|&&lp| !b.is_vertex_loop(lp))
        .flat_map(|&lp| &b.lp(lp).coedges)
        .flat_map(|&co| samples[b.coedge(co).edge.index()].iter().map(|x| x.1))
        .collect();
    for &lp in &face.loops {
        if b.is_vertex_loop(lp) {
            let e = b.coedge(b.lp(lp).coedges[0]).edge;
            if samples[e.index()]
                .iter()
                .all(|x| on_boundary.contains(&x.1))
            {
                continue;
            }
        }
        let mut pts = Vec::new();
        let mut run = Vec::new();
        for &co in &b.lp(lp).coedges {
            let c = b.coedge(co);
            let mut s = samples[c.edge.index()].clone();
            if c.reversed {
                s.reverse();
            }
            let ts: Vec<f64> = s.iter().map(|x| x.0).collect();
            let uvs = b.coedge_uvs(co, &ts);
            run.push((pts.len(), uvs[uvs.len() - 1]));
            if b.edge(c.edge).is_degenerate() {
                let d = uvs[uvs.len() - 1] - uvs[0];
                if d.norm() > 0.0 {
                    poles.insert(s[0].1, d / d.norm());
                }
            }
            for k in 0..s.len() - 1 {
                pts.push((uvs[k], s[k].1));
            }
        }
        runs.push(run);
        loops.push(pts);
    }
    super::one_sheet((surf.u_period(), surf.v_period()), &mut loops, &runs);
    let mut flat: Vec<(Vec2, u32)> = loops.iter().flatten().copied().collect();
    super::snap_repeats(&mut flat);
    let mut snapped = flat.into_iter();
    for p in loops.iter_mut().flatten() {
        *p = snapped.next().expect("one snapped point per loop point");
    }
    let (scale, curved) = metric(surf, &loops, &poles, positions, opt);
    let mut fm = FaceMesh {
        cdt: Cdt::new(Vec2::ZERO, Vec2::ZERO),
        uv: vec![Vec2::ZERO; BOX as usize],
        pos: vec![Point3::ZERO; BOX as usize],
        gid: vec![None; BOX as usize],
        inside: Vec::new(),
        pole: vec![None; BOX as usize],
        normal: vec![None; BOX as usize],
        scale,
        curved,
    };
    let all = loops.iter().flatten();
    let (mut lo, mut hi) = (
        Vec2::new(f64::INFINITY, f64::INFINITY),
        Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for (uv, _) in all {
        let m = fm.m(*uv);
        lo = Vec2::new(lo.x.min(m.x), lo.y.min(m.y));
        hi = Vec2::new(hi.x.max(m.x), hi.y.max(m.y));
    }
    if !(lo.x.is_finite() && hi.x.is_finite() && lo.y.is_finite() && hi.y.is_finite()) {
        return Err(Error::Numerical(
            "face boundary has no finite points".into(),
        ));
    }
    fm.cdt = Cdt::new(lo, hi);
    let mut ids: Vec<Vec<u32>> = Vec::new();
    for lp in &loops {
        let mut row = Vec::with_capacity(lp.len());
        for &(uv, g) in lp {
            let h = fm.cdt.hint();
            let v = match fm.cdt.insert(fm.m(uv), h) {
                Ok(v) => {
                    fm.push(uv, positions[g as usize], Some(g), poles.get(&g).copied());
                    v
                }
                Err(x) if x != NONE && fm.gid[x as usize] == Some(g) => x,
                Err(_) => {
                    return Err(Error::Numerical("face boundary touches itself".into()));
                }
            };
            row.push(v);
        }
        ids.push(row);
    }
    let area: f64 = ids
        .iter()
        .map(|row| {
            (0..row.len())
                .map(|i| {
                    let a = fm.cdt.p[row[i] as usize];
                    let c = fm.cdt.p[row[(i + 1) % row.len()] as usize];
                    a.cross(c)
                })
                .sum::<f64>()
        })
        .sum();
    if area < 0.0 {
        for row in &mut ids {
            row.reverse();
        }
    }
    let mut segs: Vec<(u32, u32)> = Vec::new();
    for row in &ids {
        for i in 0..row.len() {
            let (a, c) = (row[i], row[(i + 1) % row.len()]);
            if a != c {
                segs.push((a, c));
            }
        }
    }
    for &(a, c) in &segs {
        fm.cdt
            .insert_segment(a, c)
            .map_err(|m| Error::Numerical(m.into()))?;
    }
    fm.cdt.legalize_all();
    fm.classify(&segs)?;
    fm.refine(b, f, opt)?;
    fm.unflatten();
    Ok(fm)
}

/// The parameter-plane scale for a face: one unit along each parameter is about the longest step
/// the deflection limits allow along it, at most the face's size. Whether the face is curved.
fn metric(
    surf: &Surface,
    loops: &Loops,
    poles: &HashMap<u32, Vec2>,
    positions: &[Point3],
    opt: &MeshOptions,
) -> (Vec2, bool) {
    let pts: Vec<(Vec2, u32)> = loops.iter().flatten().copied().collect();
    let (mut lo, mut hi) = (
        Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        -Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
    );
    for &(_, g) in &pts {
        let p = positions[g as usize];
        lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    let size = (hi - lo).norm();
    let size = if size.is_finite() && size > 0.0 {
        size
    } else {
        1.0
    };
    // A step no longer than the deflection cannot stray by more, so the turn alone never
    // shortens one below it, up to a quarter turn: a wire thinner than the deflection takes a
    // few chords round, as its rims do.
    let allowed = |k: f64| {
        if k > 1e-9 / size {
            let turn = (opt.angular / k).max(opt.linear.min(FRAC_PI_2 / k));
            (8.0 * opt.linear / k).sqrt().min(turn)
        } else {
            f64::INFINITY
        }
    };
    let stride = (pts.len() / 256).max(1);
    let (mut su_sum, mut sv_sum, mut n) = (0.0, 0.0, 0.0);
    let (mut du, mut dv) = (f64::INFINITY, f64::INFINITY);
    for &(uv, g) in pts.iter().step_by(stride) {
        if poles.contains_key(&g) {
            continue;
        }
        let (_, su, sv) = surf.d1(uv.x, uv.y);
        let (lu, lv) = (su.norm(), sv.norm());
        if !(lu > 1e-12 && lv > 1e-12) {
            continue;
        }
        su_sum += lu;
        sv_sum += lv;
        n += 1.0;
        let h = 1e-4 * size;
        let (hu, hv) = (h / lu, h / lv);
        // One-sided turns, the smaller kept: a boundary point can sit a hair from a singularity
        // just outside the face (a spindle torus's apex), where the normal flips.
        let n0 = surf.normal(uv.x, uv.y);
        let turn = |du: f64, dv: f64| {
            let a = surf.normal(uv.x + du, uv.y + dv).dist(n0);
            let b = surf.normal(uv.x - du, uv.y - dv).dist(n0);
            a.min(b) / h
        };
        let ku = turn(hu, 0.0);
        let kv = turn(0.0, hv);
        du = du.min(allowed(ku) / lu);
        dv = dv.min(allowed(kv) / lv);
    }
    if n == 0.0 {
        return (Vec2::new(1.0, 1.0), true);
    }
    let curved = du.is_finite() || dv.is_finite();
    let du = du.min(size / (su_sum / n));
    let dv = dv.min(size / (sv_sum / n));
    (Vec2::new(1.0 / du, 1.0 / dv), curved)
}

/// Whether a triangle needs refining, and the parameters of its worst sample.
enum Verdict {
    Fine,
    Refine(Vec2),
    /// Fine on the surface but badly shaped: refine at the circumcentre if it can go in.
    Shape,
}

impl FaceMesh {
    fn m(&self, uv: Vec2) -> Vec2 {
        Vec2::new(uv.x * self.scale.x, uv.y * self.scale.y)
    }

    fn push(&mut self, uv: Vec2, pos: Point3, gid: Option<u32>, pole: Option<Vec2>) {
        self.uv.push(uv);
        self.pos.push(pos);
        self.gid.push(gid);
        self.pole.push(pole);
        self.normal.push(None);
    }

    /// Marks the triangles inside the face: those on the left of each boundary segment and every
    /// triangle reached from them without crossing a segment.
    fn classify(&mut self, segs: &[(u32, u32)]) -> Result<()> {
        let n = self.cdt.tri.len();
        self.inside = vec![false; n];
        let mut stack = Vec::new();
        for &(a, c) in segs {
            let Some((t, _)) = self.cdt.find_edge(a, c) else {
                return Err(Error::Numerical("face boundary segment is missing".into()));
            };
            if !self.inside[t as usize] {
                self.inside[t as usize] = true;
                stack.push(t);
            }
        }
        while let Some(t) = stack.pop() {
            for k in 0..3 {
                let nb = self.cdt.nbr[t as usize][k];
                if nb != NONE && !self.cdt.fixed[t as usize][k] && !self.inside[nb as usize] {
                    self.inside[nb as usize] = true;
                    stack.push(nb);
                }
            }
        }
        let directed: HashSet<(u32, u32)> = segs.iter().copied().collect();
        for t in 0..n {
            if self.inside[t] && self.cdt.tri[t].iter().any(|&v| v < BOX) {
                return Err(Error::Numerical("face boundary does not close".into()));
            }
        }
        for &(a, c) in segs {
            if directed.contains(&(c, a)) {
                continue;
            }
            if let Some((t, _)) = self.cdt.find_edge(c, a)
                && self.inside[t as usize]
            {
                return Err(Error::Numerical(
                    "face boundary loops do not bound one region".into(),
                ));
            }
        }
        Ok(())
    }

    /// Parameters to sample a triangle at: a pole corner is moved along its pole to face the
    /// triangle's other corners, since every point of the pole is the same point.
    fn sample_uvs(&self, c: [u32; 3]) -> Option<[Vec2; 3]> {
        let mut sum = Vec2::ZERO;
        let mut n = 0.0;
        for &v in &c {
            if self.pole[v as usize].is_none() {
                sum = sum + self.uv[v as usize];
                n += 1.0;
            }
        }
        if n == 0.0 {
            return None;
        }
        let toward = sum / n;
        Some(c.map(|v| self.facing(v, toward)))
    }

    fn facing(&self, v: u32, toward: Vec2) -> Vec2 {
        let uv = self.uv[v as usize];
        match self.pole[v as usize] {
            Some(d) => uv + d * (toward - uv).dot(d),
            None => uv,
        }
    }

    fn same_point(&self, a: u32, c: u32) -> bool {
        self.gid[a as usize].is_some() && self.gid[a as usize] == self.gid[c as usize]
    }

    fn corner_normal(&mut self, surf: &Surface, v: u32, at: Vec2, toward: Vec2) -> Vec3 {
        if self.pole[v as usize].is_some() {
            // A hair in from the pole, towards the triangle: on the pole itself a cone's radius is
            // zero to rounding, and its normal takes its sign from that radius.
            let at = at + (toward - at) * 1e-6;
            return surf.normal(at.x, at.y);
        }
        *self.normal[v as usize].get_or_insert_with(|| surf.normal(at.x, at.y))
    }

    fn assess(&mut self, surf: &Surface, t: u32, opt: &MeshOptions) -> Verdict {
        let c = self.cdt.tri[t as usize];
        if c.iter().any(|&v| v < BOX) {
            return Verdict::Fine;
        }
        let p = c.map(|v| self.pos[v as usize]);
        // However small the face, a triangle across which the surface turns more than a right angle
        // stands for none of it: a ball narrower than the deflection, sampled at its poles and
        // seam alone, would be triangles from pole to pole that fold flat.
        if let Some(s) = self.sample_uvs(c) {
            for i in 0..3 {
                let j = (i + 1) % 3;
                if self.same_point(c[i], c[j]) {
                    continue;
                }
                let mid = (s[0] + s[1] + s[2]) / 3.0;
                let ni = self.corner_normal(surf, c[i], s[i], mid);
                let nj = self.corner_normal(surf, c[j], s[j], mid);
                if ni.dot(nj) < 0.0 {
                    return Verdict::Refine((s[0] + s[1] + s[2]) / 3.0);
                }
            }
        }
        let diam = p[0].dist(p[1]).max(p[1].dist(p[2])).max(p[2].dist(p[0]));
        // A triangle no wider than the deflection cannot stray from the surface by more.
        if diam <= opt.linear {
            return Verdict::Fine;
        }
        let Some(s) = self.sample_uvs(c) else {
            return Verdict::Fine;
        };
        let fixed = self.cdt.fixed[t as usize];
        // A boundary segment is shared with the next face and never split here; what the surface
        // strays from it is the edge's error, so the centroid may stray that much more.
        let mut allow: f64 = 0.0;
        let mut worst: Option<(f64, Vec2)> = None;
        let mut bad = false;
        for (i, &on_boundary) in fixed.iter().enumerate() {
            let j = (i + 1) % 3;
            if self.same_point(c[i], c[j]) {
                continue;
            }
            let ui = self.facing(c[i], self.uv[c[j] as usize]);
            let uj = self.facing(c[j], self.uv[c[i] as usize]);
            let mid = (ui + uj) * 0.5;
            let dev = surf.eval(mid.x, mid.y).dist((p[i] + p[j]) * 0.5);
            if on_boundary {
                allow = allow.max(dev);
                continue;
            }
            if dev > opt.linear {
                bad = true;
                if worst.is_none_or(|w| dev > w.0) {
                    worst = Some((dev, mid));
                }
            }
            let mid = (s[0] + s[1] + s[2]) / 3.0;
            let ni = self.corner_normal(surf, c[i], s[i], mid);
            let nj = self.corner_normal(surf, c[j], s[j], mid);
            // The turn between two corners is taken across the step between them the way the
            // normal turns. Where that is no longer than the deflection the surface there is
            // within it of the triangle whatever the turn, as on a wire thinner than it.
            let across = (nj - ni).unit().map_or(0.0, |t| (p[j] - p[i]).dot(t).abs());
            if across > opt.linear && ni.angle(nj) > opt.angular {
                bad = true;
            }
        }
        let cuv = (s[0] + s[1] + s[2]) / 3.0;
        let cdev = surf.eval(cuv.x, cuv.y).dist((p[0] + p[1] + p[2]) / 3.0);
        if cdev > opt.linear + allow {
            bad = true;
            if worst.is_none_or(|w| cdev - allow > w.0) {
                worst = Some((cdev - allow, cuv));
            }
        }
        if bad {
            let at = worst.map_or(cuv, |w| w.1);
            return Verdict::Refine(at);
        }
        if self.curved && self.skinny(c) {
            return Verdict::Shape;
        }
        Verdict::Fine
    }

    /// Badly shaped in the scaled plane and larger than one unit there.
    fn skinny(&self, c: [u32; 3]) -> bool {
        let m = c.map(|v| self.cdt.p[v as usize]);
        let l = [m[0].dist(m[1]), m[1].dist(m[2]), m[2].dist(m[0])];
        let area2 = orient(m[0], m[1], m[2]).abs();
        let lmax = l[0].max(l[1]).max(l[2]);
        // Circumradius = abc / (4·area); the smallest angle's sine = 2·area / (the two longer).
        let r = l[0] * l[1] * l[2] / (2.0 * area2).max(1e-300);
        let mut ls = l;
        ls.sort_by(f64::total_cmp);
        let sine = area2 / (ls[1] * ls[2]).max(1e-300);
        lmax > 1.0 && r > 1.0 && sine < MIN_ANGLE_SINE
    }

    /// Adds an interior point at `uv`, walking from `t`. Returns it, or `None` when the point is
    /// outside the face or on an existing point or boundary segment.
    fn add(&mut self, surf: &Surface, t: u32, uv: Vec2) -> Option<u32> {
        if !(uv.x.is_finite() && uv.y.is_finite()) {
            return None;
        }
        let m = self.m(uv);
        let loc = self.cdt.locate(t, m);
        if let Located::Inside(x) | Located::OnEdge(x, _) = loc {
            if !self.inside[x as usize] {
                return None;
            }
            // A triangle whose error lies along a boundary segment it cannot split would otherwise
            // be refined again and again a little nearer that segment, leaving slivers.
            let c = self.cdt.tri[x as usize].map(|v| self.cdt.p[v as usize]);
            let longest = c[0].dist(c[1]).max(c[1].dist(c[2])).max(c[2].dist(c[0]));
            let nearest = c.iter().map(|q| q.dist(m)).fold(f64::INFINITY, f64::min);
            let height = orient(c[0], c[1], c[2]).abs() / longest;
            if nearest.min(height) < CROWDED * longest {
                return None;
            }
            // Nor one within the tolerance of a side's line in space, where it would make a flat
            // triangle that no flip may mend (a pole's narrowing spline face, past the stretch the
            // scaled plane sees).
            let tri = self.cdt.tri[x as usize];
            let q = surf.eval(uv.x, uv.y);
            let flat = (0..3).any(|k| {
                let (a, b) = (
                    self.pos[tri[k] as usize],
                    self.pos[tri[(k + 1) % 3] as usize],
                );
                (b - a).cross(q - a).norm() <= tol::LINEAR * a.dist(b)
            });
            if flat {
                return None;
            }
        }
        let n_before = self.cdt.tri.len();
        let v = self.cdt.insert_at(m, loc).ok()?;
        self.push(uv, surf.eval(uv.x, uv.y), None, None);
        self.inside.resize(self.cdt.tri.len(), true);
        debug_assert!(self.cdt.tri.len() >= n_before);
        Some(v)
    }

    fn refine(&mut self, b: &Brep, f: FaceId, opt: &MeshOptions) -> Result<()> {
        let surf = &b.face(f).surface;
        let mut queued = self.inside.clone();
        let mut queue: Vec<u32> = (0..self.cdt.tri.len() as u32)
            .filter(|&t| self.inside[t as usize])
            .collect();
        let mut pops: usize = 0;
        while let Some(t) = queue.pop() {
            queued[t as usize] = false;
            pops += 1;
            if self.uv.len() > MAX_FACE_POINTS {
                return Err(Error::Unsupported(format!(
                    "a face that needs more than {MAX_FACE_POINTS} points at a deflection of {} mm",
                    opt.linear
                )));
            }
            if pops > 40 * MAX_FACE_POINTS {
                return Err(Error::Numerical("tessellation does not converge".into()));
            }
            let at = match self.assess(surf, t, opt) {
                Verdict::Fine => continue,
                Verdict::Refine(at) => Some(at),
                Verdict::Shape => None,
            };
            let c = self.cdt.tri[t as usize].map(|v| self.cdt.p[v as usize]);
            let center = circumcenter(c[0], c[1], c[2])
                .map(|m| Vec2::new(m.x / self.scale.x, m.y / self.scale.y));
            let mut v = center.and_then(|uv| self.add(surf, t, uv));
            if v.is_none()
                && let Some(at) = at
            {
                v = self.add(surf, t, at);
                if v.is_none() {
                    let cuv = self.cdt.tri[t as usize]
                        .iter()
                        .fold(Vec2::ZERO, |s, &x| s + self.uv[x as usize])
                        / 3.0;
                    v = self.add(surf, t, cuv);
                }
            }
            let Some(v) = v else { continue };
            queued.resize(self.cdt.tri.len(), false);
            for x in self
                .cdt
                .around(v)
                .into_iter()
                .map(|(x, _)| x)
                .chain(std::iter::once(t))
            {
                if !queued[x as usize] {
                    queued[x as usize] = true;
                    queue.push(x);
                }
            }
        }
        Ok(())
    }

    /// A triangle's height over its longest side in space, and that side.
    fn height(&self, c: [u32; 3]) -> (f64, usize) {
        let p = c.map(|v| self.pos[v as usize]);
        let side = |k: usize| p[k].dist(p[(k + 1) % 3]);
        let k = (0..3)
            .max_by(|&a, &b| side(a).total_cmp(&side(b)))
            .unwrap_or(0);
        let area2 = (p[1] - p[0]).cross(p[2] - p[0]).norm();
        (area2 / side(k).max(1e-300), k)
    }

    /// Flips the longest side of each triangle that is flat in space, within the linear tolerance,
    /// where that leaves both triangles taller. On a face far narrower one way than the other (a
    /// cylinder of radius 1e-3 mm, 4 mm long) the scaled plane stretches the narrow way, so a
    /// triangle well shaped there can have a corner within 1e-7 mm of its opposite side in space.
    fn unflatten(&mut self) {
        for _ in 0..4 {
            let mut flipped = false;
            for t in 0..self.cdt.tri.len() as u32 {
                let c = self.cdt.tri[t as usize];
                if !self.inside[t as usize] || c.iter().any(|&v| v < BOX) {
                    continue;
                }
                let (h, k) = self.height(c);
                if h > tol::LINEAR {
                    continue;
                }
                let u = self.cdt.nbr[t as usize][k];
                if u == NONE || self.cdt.fixed[t as usize][k] || !self.inside[u as usize] {
                    continue;
                }
                let d = self.cdt.tri[u as usize];
                if d.iter().any(|&v| v < BOX) {
                    continue;
                }
                let before = h.min(self.height(d).0);
                let (a, b, x) = (c[k], c[(k + 1) % 3], c[(k + 2) % 3]);
                let Some(&y) = d.iter().find(|&&v| v != a && v != b) else {
                    continue;
                };
                let after = self.height([x, a, y]).0.min(self.height([y, b, x]).0);
                if after > before && self.cdt.flip(t, k).is_some() {
                    flipped = true;
                }
            }
            if !flipped {
                break;
            }
        }
    }

    /// Splits the interior edge `k` of triangle `t` at its middle.
    pub(super) fn split_chord(&mut self, surf: &Surface, t: u32, k: usize) {
        let tri = self.cdt.tri[t as usize];
        let (a, c) = (tri[k], tri[(k + 1) % 3]);
        let ua = self.facing(a, self.uv[c as usize]);
        let uc = self.facing(c, self.uv[a as usize]);
        let uv = (ua + uc) * 0.5;
        let m = (self.cdt.p[a as usize] + self.cdt.p[c as usize]) * 0.5;
        let v = self.cdt.p.len() as u32;
        self.cdt.push_point(m, t);
        self.cdt.split_edge(t, k, v);
        self.push(uv, surf.eval(uv.x, uv.y), None, None);
        self.inside.resize(self.cdt.tri.len(), true);
    }

    /// The face's triangles, as corner indices.
    pub(super) fn triangles(&self) -> impl Iterator<Item = (u32, [u32; 3])> + '_ {
        (0..self.cdt.tri.len() as u32)
            .filter(|&t| self.inside[t as usize])
            .map(|t| (t, self.cdt.tri[t as usize]))
    }
}
