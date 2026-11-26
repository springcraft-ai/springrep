//! A face's trimmed domain: its loops as trims (directed curve pieces with their parameter-plane
//! images), point-in-face classification by an exact winding number, and interior points.

use std::f64::consts::TAU;
use std::rc::Rc;

use crate::{
    geom::{BSpline, Curve, Surface},
    math::{Aabb, Point3, Vec2, Vec3},
    numeric::solve2,
    tol,
    topo::{
        PCurve,
        pcurve::{map_spline, spline_rate, spline_uv},
        project_on,
    },
};

/// How a trim's parameter-plane point follows from its curve parameter.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum UvMap {
    /// `origin + t·dir`, exactly.
    Affine { origin: Vec2, dir: Vec2 },
    /// The curve point projected onto the surface, periodic parameters continued from a hint.
    Projected,
    /// A spline's point at `t`, as [`PCurve::Spline`] holds it.
    Spline(Rc<BSpline>),
}

impl UvMap {
    /// The map a coedge's pcurve gives its trims.
    pub(crate) fn of(pcurve: &PCurve) -> UvMap {
        match pcurve {
            PCurve::Affine { origin, dir } => UvMap::Affine {
                origin: *origin,
                dir: *dir,
            },
            PCurve::Projected { .. } => UvMap::Projected,
            PCurve::Spline { uv } => UvMap::Spline(Rc::new(uv.clone())),
        }
    }

    /// The pcurve a coedge made from a trim at `seed` carries.
    pub(crate) fn pcurve(&self, seed: Vec2) -> PCurve {
        match self {
            UvMap::Affine { origin, dir } => PCurve::Affine {
                origin: *origin,
                dir: *dir,
            },
            UvMap::Projected => PCurve::Projected { seed },
            UvMap::Spline(uv) => PCurve::Spline { uv: (**uv).clone() },
        }
    }

    /// The map with its parameter plane mapped by `f`, an affine map.
    pub(crate) fn mapped(&self, f: &dyn Fn(Vec2) -> Vec2) -> UvMap {
        match self {
            UvMap::Affine { origin, dir } => UvMap::Affine {
                origin: f(*origin),
                dir: f(*origin + *dir) - f(*origin),
            },
            UvMap::Projected => UvMap::Projected,
            UvMap::Spline(uv) => UvMap::Spline(Rc::new(map_spline(uv, f))),
        }
    }
}

/// One directed piece of a face's boundary or of a cut across it.
#[derive(Clone, Debug)]
pub(crate) struct Trim {
    /// The piece of edge it runs along, and whether it runs along that piece's parameter.
    pub piece: usize,
    pub forward: bool,
    /// `None` for a pole, which sits at `pole` while its parameter-plane image runs on.
    pub curve: Option<Rc<Curve>>,
    pub pole: Point3,
    /// The curve parameters where it starts and ends.
    pub t0: f64,
    pub t1: f64,
    pub map: UvMap,
    pub uv0: Vec2,
    pub uv1: Vec2,
    /// The vertices it starts and ends at.
    pub v0: usize,
    pub v1: usize,
}

impl Trim {
    pub(crate) fn point(&self, t: f64) -> Point3 {
        match &self.curve {
            Some(c) => c.eval(t),
            None => self.pole,
        }
    }

    pub(crate) fn param(&self, f: f64) -> f64 {
        self.t0 + (self.t1 - self.t0) * f
    }

    pub(crate) fn uv(&self, s: &Surface, t: f64, hint: Vec2) -> Vec2 {
        match &self.map {
            UvMap::Affine { origin, dir } => *origin + *dir * t,
            UvMap::Spline(uv) => spline_uv(uv, t),
            UvMap::Projected => match &self.curve {
                Some(c) => {
                    // Towards the trim's inside, the side its points near a pole come from.
                    let inward = if (t - self.t0).abs() <= (t - self.t1).abs() {
                        self.t1
                    } else {
                        self.t0
                    };
                    project_on(s, c, t, inward, hint)
                }
                None => s.project_near(self.pole, hint),
            },
        }
    }

    /// `n + 1` parameter-plane points from start to end, each continued from the last.
    pub(crate) fn polyline(&self, s: &Surface, n: usize) -> Vec<Vec2> {
        // A trim winding round its surface several times (a thread's helix) turns more than a
        // period between `n` points, and continuation reads a turn as none: each gap is stepped
        // through more finely until the walk arrives at the trim's own end.
        let periodic = s.u_period().is_some() || s.v_period().is_some();
        let mut sub = 1;
        loop {
            let mut out = Vec::with_capacity(n + 1);
            out.push(self.uv0);
            let mut hint = self.uv0;
            let steps = n * sub;
            for j in 1..=steps {
                let uv = self.uv(s, self.param(j as f64 / steps as f64), hint);
                hint = uv;
                if j % sub == 0 && j < steps {
                    out.push(uv);
                }
            }
            let arrived = (hint - self.uv1).norm() <= 1e-6 * (1.0 + self.uv1.norm());
            if arrived || !periodic || !matches!(self.map, UvMap::Projected) || sub >= 256 {
                out.push(self.uv1);
                return out;
            }
            sub *= 2;
        }
    }

    /// The steps its parameter-plane image is sampled at: [`SAMPLES`], or two to each polynomial
    /// piece of its curve where that is more. A chord across several pieces (a trim round a whole
    /// cycloid, lobe after lobe) can sit near the curve at its middle while the lobes either side
    /// swing past a point, and the winding about the point read from the chord is then wrong.
    pub(crate) fn steps(&self) -> usize {
        let pieces = match (&self.map, &self.curve) {
            (UvMap::Projected | UvMap::Spline(_), Some(c)) => {
                c.breakpoints(self.t0.min(self.t1), self.t0.max(self.t1))
                    .len()
                    + 1
            }
            _ => 1,
        };
        (2 * pieces).max(SAMPLES)
    }

    /// The parameter-plane point halfway along, reached in small steps so a periodic parameter
    /// cannot jump half a period.
    pub(crate) fn mid_uv(&self, s: &Surface) -> Vec2 {
        self.polyline(s, 16)[8]
    }

    pub(crate) fn reversed(&self) -> Trim {
        Trim {
            forward: !self.forward,
            t0: self.t1,
            t1: self.t0,
            uv0: self.uv1,
            uv1: self.uv0,
            v0: self.v1,
            v1: self.v0,
            ..self.clone()
        }
    }

    /// The parameter-plane direction it leaves its start in.
    pub(crate) fn start_tangent(&self, s: &Surface) -> Vec2 {
        let sign = if self.t1 >= self.t0 { 1.0 } else { -1.0 };
        match &self.map {
            UvMap::Affine { dir, .. } => *dir * sign,
            UvMap::Spline(uv) => spline_rate(uv, self.t0) * sign,
            UvMap::Projected => {
                let Some(c) = &self.curve else {
                    return Vec2::ZERO;
                };
                let (_, su, sv) = s.d1(self.uv0.x, self.uv0.y);
                pole_rate(s, (su, sv), c, self.t0, sign)
                    .unwrap_or_else(|| rate_of(su, sv, c.d1(self.t0).1 * sign))
            }
        }
    }

    pub(crate) fn bbox(&self) -> Aabb {
        let mut bb = Aabb::EMPTY;
        let (lo, hi) = (self.t0.min(self.t1), self.t0.max(self.t1));
        match &self.curve {
            Some(c) => bb = c.bbox(lo, hi),
            None => bb.add(self.pole),
        }
        bb
    }

    /// The distance from `p` to the trim's curve.
    /// The parameter-plane rate at `t`, where its point is `uv`.
    pub(crate) fn uv_rate_at(&self, s: &Surface, t: f64, uv: Vec2) -> Vec2 {
        match (&self.map, &self.curve) {
            (UvMap::Affine { dir, .. }, _) => *dir,
            (UvMap::Spline(sp), _) => spline_rate(sp, t),
            (UvMap::Projected, Some(c)) => uv_rate(s, uv, c.d1(t).1),
            (UvMap::Projected, None) => Vec2::ZERO,
        }
    }

    pub(crate) fn distance(&self, p: Point3) -> f64 {
        match &self.curve {
            Some(c) => {
                let (lo, hi) = (self.t0.min(self.t1), self.t0.max(self.t1));
                c.eval(c.project(p, lo, hi)).dist(p)
            }
            None => self.pole.dist(p),
        }
    }
}

/// The parameter-plane rate of a surface curve moving at `dp` through `uv`.
pub(crate) fn uv_rate(s: &Surface, uv: Vec2, dp: Vec3) -> Vec2 {
    let (_, su, sv) = s.d1(uv.x, uv.y);
    rate_of(su, sv, dp)
}

/// [`uv_rate`] where the surface's derivatives there are `su` and `sv`.
fn rate_of(su: Vec3, sv: Vec3, dp: Vec3) -> Vec2 {
    solve2(
        su.dot(su),
        su.dot(sv),
        su.dot(sv),
        sv.dot(sv),
        su.dot(dp),
        sv.dot(dp),
    )
    .map(|(a, b)| Vec2::new(a, b))
    .unwrap_or(Vec2::ZERO)
}

/// The parameter-plane rate of `curve` leaving a pole of `s` at `t`, along its parameter or
/// against it as `sign` says, where `u` is the angle about the surface's axis and its rate is not
/// the first-order map's to solve: `u` is the angle of the curve's step `h·t + k·t²/2` across the
/// axis, `h` and `k` those parts of its first and second derivatives, which turns at
/// `z·(h × k) / 2|h|²`. `None` away from such a pole (`su` and `sv` are the surface's derivatives
/// there), or for a curve leaving along the axis.
fn pole_rate(
    s: &Surface,
    (su, sv): (Vec3, Vec3),
    curve: &Curve,
    t: f64,
    sign: f64,
) -> Option<Vec2> {
    let z = match s {
        Surface::Sphere(x) => x.frame.z,
        Surface::Cone(x) => x.frame.z,
        Surface::Torus(x) => x.frame.z,
        Surface::Revolution(x) => x.axis.unit()?,
        _ => return None,
    };
    if su.norm() > tol::LINEAR || sv.norm() <= tol::LINEAR {
        return None;
    }
    let (_, dp, ddp) = curve.d2(t);
    let dp = dp * sign;
    let across = |w: Vec3| w - z * w.dot(z);
    let (h, k) = (across(dp), across(ddp));
    let hh = h.dot(h);
    if hh <= 1e-12 * dp.dot(dp) {
        return None;
    }
    Some(Vec2::new(
        z.dot(h.cross(k)) / (2.0 * hh),
        sv.dot(dp) / sv.dot(sv),
    ))
}

fn segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    let d = b - a;
    let l2 = d.dot(d);
    if l2 <= 0.0 {
        return p.dist(a);
    }
    let t = ((p - a).dot(d) / l2).clamp(0.0, 1.0);
    p.dist(a + d * t)
}

/// Segments hashed into the square cells their boxes touch, for the distance from a point to the
/// nearest of them: a plate with hundreds of holes has tens of thousands, and a point is near few.
struct SegmentGrid<'a> {
    segs: &'a [(Vec2, Vec2)],
    lo: Vec2,
    cell: f64,
    nx: usize,
    ny: usize,
    cells: Vec<Vec<u32>>,
}

impl SegmentGrid<'_> {
    const MAX_SIDE: usize = 512;

    fn new(segs: &[(Vec2, Vec2)]) -> SegmentGrid<'_> {
        let ends = || segs.iter().flat_map(|&(a, b)| [a, b]);
        let lo = ends().fold(Vec2::new(f64::INFINITY, f64::INFINITY), |m, p| {
            Vec2::new(m.x.min(p.x), m.y.min(p.y))
        });
        let hi = ends().fold(lo, |m, p| Vec2::new(m.x.max(p.x), m.y.max(p.y)));
        let (w, h) = ((hi.x - lo.x).max(0.0), (hi.y - lo.y).max(0.0));
        // About one segment a cell over the box, and no more cells than a side allows.
        let even = (w * h / segs.len().max(1) as f64).sqrt();
        let cell = even.max(w.max(h) / Self::MAX_SIDE as f64);
        let cell = if cell > 0.0 && cell.is_finite() {
            cell
        } else {
            1.0
        };
        let side = |len: f64| ((len / cell) as usize + 1).min(Self::MAX_SIDE);
        let (nx, ny) = (side(w), side(h));
        let mut grid = SegmentGrid {
            segs,
            lo,
            cell,
            nx,
            ny,
            cells: vec![Vec::new(); nx * ny],
        };
        for (i, &(a, b)) in segs.iter().enumerate() {
            let (x0, y0) = grid.cell_of(Vec2::new(a.x.min(b.x), a.y.min(b.y)));
            let (x1, y1) = grid.cell_of(Vec2::new(a.x.max(b.x), a.y.max(b.y)));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    grid.cells[y * nx + x].push(i as u32);
                }
            }
        }
        grid
    }

    fn cell_of(&self, p: Vec2) -> (usize, usize) {
        let at = |v: f64, n: usize| (v / self.cell).clamp(0.0, (n - 1) as f64) as usize;
        (at(p.x - self.lo.x, self.nx), at(p.y - self.lo.y, self.ny))
    }

    /// The distance from `p` to the nearest segment, infinite when there is none.
    fn nearest(&self, p: Vec2) -> f64 {
        let (cx, cy) = self.cell_of(p);
        let mut best = f64::INFINITY;
        for r in 0..self.nx.max(self.ny) {
            // A cell in ring `r` or beyond lies at least `r − 1` cells from `p`, wherever `p` is
            // in its own cell.
            if r > 0 && best <= (r - 1) as f64 * self.cell {
                break;
            }
            let (x0, x1) = (cx.saturating_sub(r), (cx + r).min(self.nx - 1));
            let (y0, y1) = (cy.saturating_sub(r), (cy + r).min(self.ny - 1));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if x.abs_diff(cx).max(y.abs_diff(cy)) != r {
                        continue;
                    }
                    for &i in &self.cells[y * self.nx + x] {
                        let (a, b) = self.segs[i as usize];
                        best = best.min(segment_distance(p, a, b));
                    }
                }
            }
        }
        best
    }
}

fn angle(a: Vec2, b: Vec2) -> f64 {
    a.cross(b).atan2(a.dot(b))
}

/// The chords a trim's winding is read from: one for a trim that is a line in the parameter
/// plane, the steps of its polyline otherwise.
fn chords(tr: &Trim, samples: &[Vec2]) -> Vec<Vec2> {
    match tr.map {
        UvMap::Affine { .. } => vec![tr.uv0, tr.uv1],
        UvMap::Projected | UvMap::Spline(_) => samples.to_vec(),
    }
}

/// Where the trim itself passes at the middle of each of its [`chords`]: how far each chord
/// stands off its arc, which does not turn on the point a winding is asked about.
fn chord_mids(s: &Surface, tr: &Trim, samples: &[Vec2]) -> Vec<Vec2> {
    let pts = chords(tr, samples);
    let n = pts.len() - 1;
    (0..n)
        .map(|i| {
            let ta = tr.param(i as f64 / n as f64);
            let tb = tr.param((i + 1) as f64 / n as f64);
            tr.uv(s, 0.5 * (ta + tb), (pts[i] + pts[i + 1]) * 0.5)
        })
        .collect()
}

/// The angle the trim sweeps as seen from `q`, exactly: each chord stands in for its arc only once
/// the arc lies well clear of `q`. `samples` is the trim's [`Trim::steps`]-step polyline, and
/// `mids` its [`chord_mids`].
fn trim_angle(s: &Surface, tr: &Trim, samples: &[Vec2], mids: &[Vec2], q: Vec2) -> f64 {
    /// The chord from `a` to `b`, whose arc passes through `m` at its middle.
    #[allow(clippy::too_many_arguments)]
    fn rec(
        s: &Surface,
        tr: &Trim,
        q: Vec2,
        (ta, a): (f64, Vec2),
        (tb, b): (f64, Vec2),
        m: Vec2,
        depth: u32,
    ) -> f64 {
        let tm = 0.5 * (ta + tb);
        let mid = (a + b) * 0.5;
        let dev = m.dist(mid);
        let d = segment_distance(q, a, b);
        if depth >= 48 || dev * 4.0 < d || dev <= 1e-15 * (1.0 + mid.norm()) {
            return angle(a - q, b - q);
        }
        let half = |(t0, p0): (f64, Vec2), (t1, p1): (f64, Vec2)| {
            let through = tr.uv(s, 0.5 * (t0 + t1), (p0 + p1) * 0.5);
            rec(s, tr, q, (t0, p0), (t1, p1), through, depth + 1)
        };
        half((ta, a), (tm, m)) + half((tm, m), (tb, b))
    }
    let pts = chords(tr, samples);
    let n = pts.len() - 1;
    let mut sum = 0.0;
    for i in 0..n {
        let ta = tr.param(i as f64 / n as f64);
        let tb = tr.param((i + 1) as f64 / n as f64);
        sum += rec(s, tr, q, (ta, pts[i]), (tb, pts[i + 1]), mids[i], 0);
    }
    sum
}

/// What part of a face's boundary a point lies on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum OnBoundary {
    /// An edge, running in this unit direction with the face's loops.
    Edge(Vec3),
    /// A seam: the face lies on both sides.
    Seam,
    /// Where pieces of the boundary meet at an angle, or a pole.
    Corner,
}

/// Where a point lies relative to a face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Where {
    Outside,
    Boundary,
    Inside(Vec2),
}

/// A face's trimmed domain on its surface.
#[derive(Clone, Debug)]
pub(crate) struct Domain {
    pub surface: Surface,
    pub reversed: bool,
    pub loops: Vec<Vec<Trim>>,
    pub lo: Vec2,
    pub hi: Vec2,
    boxes: Vec<Vec<Aabb>>,
    /// Each trim's [`Trim::steps`]-step polyline.
    samples: Vec<Vec<Vec<Vec2>>>,
    /// Each trim's [`chord_mids`], worked out at the first winding asked for.
    mids: std::cell::OnceCell<Vec<Vec<Vec<Vec2>>>>,
    /// Each trim's local tolerance where an imported edge carries one above [`tol::LINEAR`]: a
    /// point within it of the trim's curve is on the boundary, though off the surface by as much.
    tols: Option<Vec<Vec<f64>>>,
}

/// The fewest steps a domain samples each trim's parameter-plane image at.
pub(crate) const SAMPLES: usize = 8;

impl Domain {
    pub(crate) fn new(surface: Surface, reversed: bool, loops: Vec<Vec<Trim>>) -> Domain {
        let samples = loops
            .iter()
            .map(|l| {
                l.iter()
                    .map(|tr| tr.polyline(&surface, tr.steps()))
                    .collect()
            })
            .collect();
        let boxes = loops
            .iter()
            .map(|l| l.iter().map(Trim::bbox).collect())
            .collect();
        Domain::from_samples(surface, reversed, loops, samples, boxes)
    }

    /// A domain from its trims' polylines and boxes, worked out before.
    pub(crate) fn from_samples(
        surface: Surface,
        reversed: bool,
        loops: Vec<Vec<Trim>>,
        samples: Vec<Vec<Vec<Vec2>>>,
        boxes: Vec<Vec<Aabb>>,
    ) -> Domain {
        let (mut lo, mut hi) = (
            Vec2::new(f64::INFINITY, f64::INFINITY),
            Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for &p in samples.iter().flatten().flatten() {
            lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
        Domain {
            surface,
            reversed,
            loops,
            lo,
            hi,
            boxes,
            samples,
            mids: std::cell::OnceCell::new(),
            tols: None,
        }
    }

    /// Gives each trim its edge's local tolerance, loop by loop.
    pub(crate) fn set_tolerances(&mut self, tols: Vec<Vec<f64>>) {
        self.tols = tols
            .iter()
            .flatten()
            .any(|&t| t > tol::LINEAR)
            .then_some(tols);
    }

    /// The widest tolerance of any trim.
    pub(crate) fn widest(&self) -> f64 {
        self.tols
            .iter()
            .flatten()
            .flatten()
            .fold(tol::LINEAR, |a, &b| a.max(b))
    }

    fn trim_tol(&self, l: usize, k: usize) -> f64 {
        self.tols.as_ref().map_or(tol::LINEAR, |t| t[l][k])
    }

    pub(crate) fn samples(&self) -> &[Vec<Vec<Vec2>>] {
        &self.samples
    }

    pub(crate) fn boxes(&self) -> &[Vec<Aabb>] {
        &self.boxes
    }

    pub(crate) fn trims(&self) -> impl Iterator<Item = &Trim> {
        self.loops.iter().flatten()
    }

    /// `uv` moved by whole periods to lie in the domain's parameter window, or nearest its middle.
    pub(crate) fn shift_into(&self, uv: Vec2) -> Vec2 {
        let fit = |x: f64, lo: f64, hi: f64, period: Option<f64>| -> f64 {
            let Some(p) = period else { return x };
            let mid = 0.5 * (lo + hi);
            let base = x + ((mid - x) / p).round() * p;
            let eps = 1e-9 * p;
            for k in [0.0, -1.0, 1.0] {
                let c = base + k * p;
                if c >= lo - eps && c <= hi + eps {
                    return c;
                }
            }
            base
        };
        Vec2::new(
            fit(uv.x, self.lo.x, self.hi.x, self.surface.u_period()),
            fit(uv.y, self.lo.y, self.hi.y, self.surface.v_period()),
        )
    }

    /// Each loop's trims' chord middles.
    fn mids(&self) -> &Vec<Vec<Vec<Vec2>>> {
        self.mids.get_or_init(|| {
            self.loops
                .iter()
                .zip(&self.samples)
                .map(|(l, samples)| {
                    l.iter()
                        .zip(samples)
                        .map(|(tr, pl)| chord_mids(&self.surface, tr, pl))
                        .collect()
                })
                .collect()
        })
    }

    /// The winding number of the loops about `uv`.
    pub(crate) fn winding(&self, uv: Vec2) -> i64 {
        let total: f64 = self
            .trims()
            .zip(self.samples.iter().flatten())
            .zip(self.mids().iter().flatten())
            .map(|((tr, pl), mids)| trim_angle(&self.surface, tr, pl, mids, uv))
            .sum();
        (total / TAU).round() as i64
    }

    /// Whether `p` lies in the face: off its surface or outside its loops, on a boundary curve
    /// (within the linear tolerance), or inside at parameters `uv`.
    pub(crate) fn locate(&self, p: Point3) -> Where {
        let uv = self.surface.project(p);
        let off = self.surface.eval(uv.x, uv.y).dist(p);
        if off > self.widest() {
            return Where::Outside;
        }
        if self.on_boundary(p) {
            return Where::Boundary;
        }
        if off > tol::LINEAR {
            return Where::Outside;
        }
        self.inside_image(uv).map_or(Where::Outside, Where::Inside)
    }

    /// The image of `uv`, moved by whole periods, that the loops enclose. A face that winds along
    /// a helix spans more than a period, so a point can have two images in its window, and only one
    /// of them need be inside.
    pub(crate) fn inside_image(&self, uv: Vec2) -> Option<Vec2> {
        self.images(uv).into_iter().find(|&c| self.winding(c) != 0)
    }

    /// Whether the loops other than `skip` enclose `uv`, or an image of it a whole number of
    /// periods away: for a point on loop `skip`, whether that loop lies inside the face the others
    /// bound. A loop whose sampled box, widened by its own size, misses the point cannot wind
    /// round it.
    pub(crate) fn inside_without(&self, uv: Vec2, skip: usize) -> bool {
        let reach: Vec<(Vec2, Vec2)> = self
            .samples
            .iter()
            .map(|l| {
                let (mut lo, mut hi) = (
                    Vec2::new(f64::INFINITY, f64::INFINITY),
                    Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
                );
                for &p in l.iter().flatten() {
                    lo = Vec2::new(lo.x.min(p.x), lo.y.min(p.y));
                    hi = Vec2::new(hi.x.max(p.x), hi.y.max(p.y));
                }
                let pad = Vec2::new(hi.x - lo.x, hi.y - lo.y);
                (
                    Vec2::new(lo.x - pad.x, lo.y - pad.y),
                    Vec2::new(hi.x + pad.x, hi.y + pad.y),
                )
            })
            .collect();
        self.images(uv).into_iter().any(|q| {
            let total: f64 = (0..self.loops.len())
                .filter(|&l| l != skip)
                .filter(|&l| {
                    let (lo, hi) = reach[l];
                    q.x >= lo.x && q.x <= hi.x && q.y >= lo.y && q.y <= hi.y
                })
                .flat_map(|l| {
                    self.loops[l]
                        .iter()
                        .zip(&self.samples[l])
                        .zip(&self.mids()[l])
                })
                .map(|((tr, pl), mids)| trim_angle(&self.surface, tr, pl, mids, q))
                .sum();
            (total / TAU).round() as i64 != 0
        })
    }

    /// `uv` and its images a whole number of periods away within the domain's parameter window.
    fn images(&self, uv: Vec2) -> Vec<Vec2> {
        let images = |x: f64, lo: f64, hi: f64, period: Option<f64>| -> Vec<f64> {
            let Some(p) = period else { return vec![x] };
            let eps = 1e-9 * p;
            let first = x + ((lo - eps - x) / p).ceil() * p;
            let all: Vec<f64> = (0..)
                .map(|k| first + k as f64 * p)
                .take_while(|&c| c <= hi + eps)
                .collect();
            if all.is_empty() {
                let mid = 0.5 * (lo + hi);
                vec![x + ((mid - x) / p).round() * p]
            } else {
                all
            }
        };
        let us = images(uv.x, self.lo.x, self.hi.x, self.surface.u_period());
        let vs = images(uv.y, self.lo.y, self.hi.y, self.surface.v_period());
        us.iter()
            .flat_map(|&u| vs.iter().map(move |&v| Vec2::new(u, v)))
            .collect()
    }

    /// Whether `p`, which lies within the tolerance of a boundary curve, is inside the loops by the
    /// exact winding number, and at which parameters. Only a piece that was cut along the boundary
    /// may rely on this: it tells which side of the curve the point is on, however close.
    pub(crate) fn exact_side(&self, p: Point3) -> Option<Vec2> {
        self.inside_image(self.surface.project(p))
    }

    /// Whether the points all lie within the tolerance of one trim: a piece of curve through them
    /// runs along the boundary there, where one whose ends lie on two different trims cuts across a
    /// corner however short it is.
    pub(crate) fn along_one_trim(&self, pts: &[Point3]) -> bool {
        self.loops
            .iter()
            .zip(&self.boxes)
            .enumerate()
            .any(|(li, (l, boxes))| {
                l.iter().zip(boxes).enumerate().any(|(k, (tr, bb))| {
                    let t = self.trim_tol(li, k);
                    pts.iter().all(|p| {
                        (0..3).all(|j| p[j] >= bb.min[j] - t && p[j] <= bb.max[j] + t)
                            && tr.distance(*p) <= t
                    })
                })
            })
    }

    pub(crate) fn on_boundary(&self, p: Point3) -> bool {
        for (li, (l, boxes)) in self.loops.iter().zip(&self.boxes).enumerate() {
            for (k, (tr, bb)) in l.iter().zip(boxes).enumerate() {
                let t = self.trim_tol(li, k);
                let near = (0..3).all(|k| p[k] >= bb.min[k] - t && p[k] <= bb.max[k] + t);
                if near && tr.distance(p) <= t {
                    return true;
                }
            }
        }
        false
    }

    /// What of the boundary `p` lies on, within the tolerance; `None` where it lies on none.
    pub(crate) fn boundary_at(&self, p: Point3) -> Option<OnBoundary> {
        let mut found: Option<(Vec3, usize)> = None;
        let mut seam = false;
        for (li, (l, boxes)) in self.loops.iter().zip(&self.boxes).enumerate() {
            for (k, (tr, bb)) in l.iter().zip(boxes).enumerate() {
                let t = self.trim_tol(li, k);
                let near = (0..3).all(|k| p[k] >= bb.min[k] - t && p[k] <= bb.max[k] + t);
                if !near || tr.distance(p) > t {
                    continue;
                }
                let Some(c) = tr.curve.as_ref() else {
                    return Some(OnBoundary::Corner);
                };
                let (lo, hi) = (tr.t0.min(tr.t1), tr.t0.max(tr.t1));
                let (_, d) = c.d1(c.project(p, lo, hi));
                let Some(d) = if tr.t1 >= tr.t0 { d } else { -d }.unit() else {
                    return Some(OnBoundary::Corner);
                };
                match found {
                    None => found = Some((d, tr.piece)),
                    Some((f, _)) if f.cross(d).norm() > tol::TANGENT => {
                        return Some(OnBoundary::Corner);
                    }
                    // The same piece of edge both ways round: the face continues across it.
                    Some((f, piece)) if f.dot(d) < 0.0 && piece == tr.piece => seam = true,
                    Some((f, _)) if f.dot(d) < 0.0 => return Some(OnBoundary::Corner),
                    Some(_) => {}
                }
            }
        }
        found.map(|(d, _)| {
            if seam {
                OnBoundary::Seam
            } else {
                OnBoundary::Edge(d)
            }
        })
    }

    /// The surface's outward normal at `uv`.
    pub(crate) fn outward(&self, uv: Vec2) -> Vec3 {
        let n = self.surface.normal(uv.x, uv.y);
        if self.reversed { -n } else { n }
    }

    /// Parameter-plane scale: lengths along `u` and `v` per unit parameter near the middle.
    pub(crate) fn scale(&self) -> Vec2 {
        let mid = (self.lo + self.hi) * 0.5;
        let (_, su, sv) = self.surface.d1(mid.x, mid.y);
        Vec2::new(su.norm().max(1e-12), sv.norm().max(1e-12))
    }

    /// The loops as parameter-plane polygons.
    pub(crate) fn polygons(&self, per_trim: usize) -> Vec<Vec<Vec2>> {
        self.loops
            .iter()
            .map(|l| {
                let mut pts = Vec::new();
                for tr in l {
                    let n = match tr.map {
                        UvMap::Affine { .. } => 1,
                        UvMap::Projected | UvMap::Spline(_) => per_trim,
                    };
                    let pl = tr.polyline(&self.surface, n);
                    pts.extend_from_slice(&pl[..pl.len() - 1]);
                }
                pts
            })
            .collect()
    }

    /// A point well inside the domain: the candidate on a grid of scan lines farthest from the
    /// boundary that the exact winding number confirms is inside.
    pub(crate) fn interior_point(&self) -> Option<Vec2> {
        self.interior_points()
            .first()
            .copied()
            .or_else(|| self.interior_point_from_boundary())
    }

    /// Points inside the domain, farthest from the boundary first: the candidates of
    /// [`Domain::interior_point`] that the exact winding number confirms.
    pub(crate) fn interior_points(&self) -> Vec<Vec2> {
        let polys = self.polygons(48);
        let sc = self.scale();
        let scaled = |p: Vec2| Vec2::new(p.x * sc.x, p.y * sc.y);
        let segs: Vec<(Vec2, Vec2)> = polys
            .iter()
            .flat_map(|poly| (0..poly.len()).map(move |i| (poly[i], poly[(i + 1) % poly.len()])))
            .collect();
        let lengths: Vec<(Vec2, Vec2)> =
            segs.iter().map(|&(a, b)| (scaled(a), scaled(b))).collect();
        let grid = SegmentGrid::new(&lengths);
        let clearance = |p: Vec2| grid.nearest(scaled(p));
        let mut cands: Vec<Vec2> = Vec::new();
        let n = 17;
        for axis in 0..2 {
            let (lo, hi) = if axis == 0 {
                (self.lo.y, self.hi.y)
            } else {
                (self.lo.x, self.hi.x)
            };
            for k in 0..n {
                let f = (k as f64 + 0.5 + 0.1234 * (k % 3) as f64) / (n as f64 + 0.5);
                let level = lo + (hi - lo) * f;
                let mut xs: Vec<f64> = Vec::new();
                for &(a, b) in &segs {
                    let (ay, by, ax, bx) = if axis == 0 {
                        (a.y, b.y, a.x, b.x)
                    } else {
                        (a.x, b.x, a.y, b.y)
                    };
                    if (ay <= level) != (by <= level) {
                        xs.push(ax + (bx - ax) * (level - ay) / (by - ay));
                    }
                }
                xs.sort_by(f64::total_cmp);
                for w in xs.chunks(2) {
                    if w.len() == 2 {
                        let m = 0.5 * (w[0] + w[1]);
                        cands.push(if axis == 0 {
                            Vec2::new(m, level)
                        } else {
                            Vec2::new(level, m)
                        });
                    }
                }
            }
        }
        let mut scored: Vec<(f64, Vec2)> = cands.into_iter().map(|c| (clearance(c), c)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored
            .into_iter()
            .take(8)
            .filter(|&(_, c)| self.winding(c) != 0)
            .map(|(_, c)| c)
            .collect()
    }

    /// An interior point found from the exact boundary rather than its polygon, for a domain
    /// thinner than the polygon's sagitta: from the middle of each trim, the deepest point along
    /// its normal that the exact winding number places inside.
    fn interior_point_from_boundary(&self) -> Option<Vec2> {
        let sc = self.scale();
        let extent = ((self.hi.x - self.lo.x) * sc.x).hypot((self.hi.y - self.lo.y) * sc.y);
        let mut best: Option<(f64, Vec2)> = None;
        for tr in self.trims() {
            let pl = tr.polyline(&self.surface, 16);
            let (m, a, b) = (pl[8], pl[7], pl[9]);
            let t = Vec2::new((b.x - a.x) * sc.x, (b.y - a.y) * sc.y);
            let n = t.norm();
            if n <= 0.0 {
                continue;
            }
            let t = t * (1.0 / n);
            for side in [1.0, -1.0] {
                // Along the normal in true lengths, mapped back to parameters.
                let dir = Vec2::new(-t.y * side / sc.x, t.x * side / sc.y);
                let at = |s: f64| m + dir * s;
                let mut s = 0.5 * extent;
                while s > 1e-6 * tol::LINEAR && self.winding(at(s)) == 0 {
                    s *= 0.5;
                }
                if s <= 1e-6 * tol::LINEAR {
                    continue;
                }
                // Inside at s; find where the normal leaves the domain and take the middle.
                let (mut lo, mut hi) = (s, (2.0 * s).min(extent));
                if self.winding(at(hi)) != 0 {
                    lo = hi;
                } else {
                    for _ in 0..40 {
                        let mid = 0.5 * (lo + hi);
                        if self.winding(at(mid)) != 0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                }
                let depth = 0.5 * lo;
                let c = at(if self.winding(at(depth)) != 0 {
                    depth
                } else {
                    s
                });
                if best.is_none_or(|(d, _)| depth > d) {
                    best = Some((depth, c));
                }
            }
        }
        best.map(|(_, c)| c)
    }
}

/// The signed area of a closed polygon, positive counter-clockwise.
pub(crate) fn polygon_area(p: &[Vec2]) -> f64 {
    (0..p.len())
        .map(|i| p[i].cross(p[(i + 1) % p.len()]))
        .sum::<f64>()
        * 0.5
}

/// An angle wrapped into [0, 2π).
pub(crate) fn wrap_angle(a: f64) -> f64 {
    let w = a.rem_euclid(TAU);
    if w >= TAU { 0.0 } else { w }
}

/// The direction `d` as an angle in [0, 2π), or `None` for a zero vector.
pub(crate) fn direction_angle(d: Vec2) -> Option<f64> {
    (d.norm() > 0.0).then(|| wrap_angle(d.y.atan2(d.x)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geom::Cylinder, math::Frame};

    /// The grid's answer is the least distance over every segment, for points inside the
    /// segments' box, on it and well outside it.
    #[test]
    fn the_segment_grid_finds_the_nearest_segment() {
        let mut segs = Vec::new();
        for i in 0..40 {
            for j in 0..25 {
                let c = Vec2::new(3.0 * i as f64, 2.0 * j as f64);
                let d = Vec2::new(0.4 + 0.01 * j as f64, 0.3 - 0.02 * (i % 7) as f64);
                segs.push((c - d, c + d));
            }
        }
        let grid = SegmentGrid::new(&segs);
        for k in 0..400 {
            let p = Vec2::new(
                -30.0 + 0.47 * k as f64,
                -20.0 + 0.23 * ((k * 37) % 400) as f64,
            );
            let all = segs
                .iter()
                .map(|&(a, b)| segment_distance(p, a, b))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(grid.nearest(p), all, "at {p:?}");
        }
        let flat = [(Vec2::new(1.0, 2.0), Vec2::new(1.0, 2.0))];
        assert_eq!(SegmentGrid::new(&flat).nearest(Vec2::new(4.0, 6.0)), 5.0);
        assert_eq!(SegmentGrid::new(&[]).nearest(Vec2::ZERO), f64::INFINITY);
    }

    #[test]
    fn a_band_wound_further_than_a_period_finds_the_image_inside_it() {
        // On a cylinder of radius 21, the band between two helices a period less 0.4 apart: what a
        // disc's side keeps after one helical tooth is cut across its seam. Its window in u is
        // wider than a period, so a point has two images there and only one is inside.
        let surface = Surface::Cylinder(Cylinder {
            frame: Frame::WORLD,
            radius: 21.0,
        });
        let corners = [
            Vec2::new(0.2, 0.0),
            Vec2::new(0.2 + TAU - 0.4, 0.0),
            Vec2::new(1.4 + TAU - 0.4, 6.0),
            Vec2::new(1.4, 6.0),
        ];
        // Trims drawn only in the parameter plane: their pole, far off, is never near a test point.
        let trims = (0..4)
            .map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                Trim {
                    piece: i,
                    forward: true,
                    curve: None,
                    pole: Vec3::new(0.0, 0.0, -100.0),
                    t0: 0.0,
                    t1: 1.0,
                    map: UvMap::Affine {
                        origin: a,
                        dir: b - a,
                    },
                    uv0: a,
                    uv1: b,
                    v0: i,
                    v1: (i + 1) % 4,
                }
            })
            .collect();
        let dom = Domain::new(surface.clone(), false, vec![trims]);
        let at = |u: f64, v: f64| surface.eval(u, v);
        assert!(matches!(dom.locate(at(0.539, 1.46)), Where::Inside(_)));
        assert!(matches!(dom.locate(at(6.0, 1.46)), Where::Inside(_)));
        assert_eq!(dom.locate(at(0.3, 1.46)), Where::Outside);
    }

    #[test]
    fn a_lobed_outline_in_one_trim_holds_points_a_hair_inside_it() {
        // A cycloid disc's cap: its outline, r = 30 + 3·cos 9θ, one closed spline and one trim.
        // A few chords across all nine lobes can sit near the outline at their middles while the
        // lobes either side swing past a point. Points 0.02 mm either side of the outline, all the
        // way round, must be inside and outside.
        let r = |a: f64| 30.0 + 3.0 * (9.0 * a).cos();
        let pts: Vec<Vec3> = (0..360)
            .map(|i| {
                let a = TAU * i as f64 / 360.0;
                Vec3::new(r(a) * a.cos(), r(a) * a.sin(), 0.0)
            })
            .collect();
        let spline = crate::geom::BSpline::interpolate_periodic(&pts).unwrap();
        let (t0, t1) = spline.domain();
        let curve = Curve::BSpline(spline);
        let start = curve.eval(t0);
        let uv0 = Vec2::new(start.x, start.y);
        let trim = Trim {
            piece: 0,
            forward: true,
            curve: Some(Rc::new(curve)),
            pole: Vec3::ZERO,
            t0,
            t1,
            map: UvMap::Projected,
            uv0,
            uv1: uv0,
            v0: 0,
            v1: 0,
        };
        let plane = Surface::Plane(crate::geom::Plane {
            frame: Frame::WORLD,
        });
        let dom = Domain::new(plane, false, vec![vec![trim]]);
        for i in 0..360 {
            let a = TAU * (i as f64 + 0.37) / 360.0;
            // The outline's normal at `a`, from its tangent.
            let (x, y) = (r(a) * a.cos(), r(a) * a.sin());
            let dr = -27.0 * (9.0 * a).sin();
            let t = Vec2::new(dr * a.cos() - r(a) * a.sin(), dr * a.sin() + r(a) * a.cos());
            let n = Vec2::new(t.y, -t.x) * (1.0 / t.norm());
            let near = |off: f64| Vec3::new(x + n.x * off, y + n.y * off, 0.0);
            assert!(
                matches!(dom.locate(near(-0.02)), Where::Inside(_)),
                "inside at {a}"
            );
            assert_eq!(dom.locate(near(0.02)), Where::Outside, "outside at {a}");
        }
    }
}
