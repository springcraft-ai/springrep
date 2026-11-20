//! A fillet along a tangent chain of edges no constant section sweeps: the surface a ball of the
//! fillet's radius sweeps rolling along the chain in contact with the faces on both sides.
//!
//! The chain runs the way its side-0 face uses it, with one parameter `s` along it. At each `s` the
//! ball's centre lies in the plane square to the chain, at the radius from both sides, found by
//! Newton steps whose rows are the two surface normals and the chain's tangent. A ball larger than
//! the chain's bend has no centre in some of those planes (they cross before they reach the
//! centres' path), and the fillet is then made again with each section in the plane square to the
//! centres' path through the chain's point, where the ball's circle of contact lies. A side's contact is
//! the nearest point on whichever of that side's faces holds it: where a side changes face at a
//! joint, between faces that meet tangentially, the ball's contact slides from one onto the other,
//! and a foot counts only on its own face's side of the edge between them.
//!
//! An open chain ends where one other face, a plane, meets each end. Past an end the chain runs on
//! straight along its tangent there, so the ball keeps rolling between the same two surfaces, and
//! the blend is trimmed by that plane: for each `v` across it, `s*(v)` is where the true section
//! meets the plane, and the trim curve through those points is fitted like the blend.
//!
//! The blend is cut into segments at every chain vertex and wherever a contact curve crosses onto
//! the next face or over a face's seam, so each segment is smooth. Across, a segment is the exact
//! circular arc between the contacts, a rational quadratic; along, its control rows are interpolated
//! by a cubic in `s`, sampled more finely until every midway section and both contact curves lie
//! within a tenth of the linear tolerance of the true ones. Neighbouring segments share the section
//! arc between them, and a trim cuts each segment it reaches.
//!
//! The result is sewn by surgery: each face gives up the strip between the chain and its contact
//! curve, the edges and seams at the chain's vertices are cut back to where the contact curves cross
//! them, an end's plane gives up the corner the blend takes and gains its trim curve, and the
//! segments' faces fill the strip.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{Picked, outward, run};
use crate::{
    Error, Result,
    boolean::{
        domain::{Domain, Where},
        face_domain,
    },
    check,
    geom::{BSpline, Curve, NurbsCurve, NurbsSurface, Surface},
    math::{Point3, Vec2, Vec3},
    tol,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, FaceName, PCurve, VertexId},
};

/// How far a sampled blend may stray from the true one, as a fraction of the linear tolerance.
const FIT: f64 = 0.1;
/// Section counts tried in turn for a segment; the last that still misses [`FIT`] refuses.
const SAMPLES: [usize; 10] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

fn unsupported<T>(why: &str) -> Result<T> {
    Err(Error::Unsupported(format!("a rolling-ball fillet {why}")))
}

/// One cross-section: the ball's centre and its contact point and face on each side.
#[derive(Clone, Copy, Debug)]
struct Section {
    centre: Point3,
    contact: [Point3; 2],
    face: [FaceId; 2],
}

impl Section {
    /// The rational quadratic arc from contact 0 to contact 1: its middle control point and weight.
    fn arc(&self, radius: f64) -> Result<(Point3, f64)> {
        let (a, b) = (self.contact[0] - self.centre, self.contact[1] - self.centre);
        let half = 0.5 * a.cross(b).norm().atan2(a.dot(b));
        let w = half.cos();
        if w <= 1e-3 {
            return unsupported("whose faces turn by nearly a half turn");
        }
        let Some(mid) = (a + b).unit() else {
            return unsupported("whose faces turn by a half turn");
        };
        Ok((self.centre + mid * (radius / w), w))
    }

    /// The arc's point at `v` in [0, 1].
    fn at(&self, radius: f64, v: f64) -> Result<Point3> {
        let (m, w) = self.arc(radius)?;
        let (b0, b1, b2) = ((1.0 - v) * (1.0 - v), 2.0 * v * (1.0 - v) * w, v * v);
        Ok((self.contact[0] * b0 + m * b1 + self.contact[1] * b2) / (b0 + b1 + b2))
    }

    fn arc_curve(&self, radius: f64) -> Result<NurbsCurve> {
        let (mid, w) = self.arc(radius)?;
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![self.contact[0], mid, self.contact[1]],
            Some(vec![1.0, w, 1.0]),
        )
    }
}

/// One picked edge in the chain, `s` running from `s0` for `len` along it.
struct Link {
    pk: usize,
    edge: EdgeId,
    forward: bool,
    faces: [FaceId; 2],
    s0: f64,
    len: f64,
}

/// An edge at a chain vertex between the two faces a side has on either side of it, or a seam of
/// the one face it has on both, with which way is into each.
struct Junction {
    edge: EdgeId,
    at: Point3,
    /// The face before the vertex along the chain, and the face after it.
    before: FaceId,
    after: FaceId,
    seam: bool,
}

/// A free end of an open chain: the vertex, the one other face there, and that face's edges at the
/// vertex with each side's face.
struct End {
    vertex: VertexId,
    face: FaceId,
    surface: Surface,
    /// +1 or −1, so [`End::side`] is positive on the side the blend keeps.
    sign: f64,
    edges: [EdgeId; 2],
    /// `s*` at `v` = 0, 1 and the extreme values between.
    lo: f64,
    hi: f64,
}

impl End {
    /// How far `p` lies from the end face's surface, along its normal, positive on the kept side.
    fn side(&self, p: Point3) -> f64 {
        let uv = self.surface.project(p);
        let q = self.surface.eval(uv.x, uv.y);
        self.sign * (p - q).dot(self.surface.normal(uv.x, uv.y))
    }
}

/// The chain and the ball.
struct Roll<'a> {
    body: &'a Brep,
    links: Vec<Link>,
    closed: bool,
    total: f64,
    radius: f64,
    /// −1 where the ball lies inside the body (a convex chain), +1 outside it.
    side: f64,
    /// Whether each section lies square to the centres' path, the two contact normals' cross
    /// product, rather than to the chain.
    spine: bool,
    /// `joints[i][k]`: the junction on side `k` where link `i` ends and the next begins.
    joints: Vec<[Option<Junction>; 2]>,
    /// The trimmed domain of every face a side has, to tell whether a foot lies in the face.
    domains: HashMap<FaceId, Domain>,
}

impl Roll<'_> {
    fn wrap(&self, s: f64) -> f64 {
        if self.closed {
            s.rem_euclid(self.total)
        } else {
            s
        }
    }

    fn link_at(&self, s: f64) -> usize {
        let s = self.wrap(s);
        self.links
            .iter()
            .rposition(|l| l.s0 <= s + 1e-12)
            .unwrap_or(0)
    }

    /// The point and unit tangent of link `i`'s edge `d` along it from its start in `s`.
    fn on_link(&self, i: usize, d: f64) -> Result<(Point3, Vec3)> {
        let l = &self.links[i];
        let e = self.body.edge(l.edge);
        let t = if l.forward {
            e.range.0 + d
        } else {
            e.range.1 - d
        };
        let (p, dt) = e.curve.as_ref().expect("a picked edge has a curve").d1(t);
        let Some(along) = (if l.forward { dt } else { -dt }).unit() else {
            return unsupported("on an edge with no tangent");
        };
        Ok((p, along))
    }

    /// The chain's point and unit tangent at `s`, and the link it lies on. Past an open chain's
    /// end it runs on straight along its tangent there.
    fn chain_at(&self, s: f64) -> Result<(Point3, Vec3, usize)> {
        let s = self.wrap(s);
        let last = self.links.len() - 1;
        if !self.closed && s < 0.0 {
            let (p, d) = self.on_link(0, 0.0)?;
            return Ok((p + d * s, d, 0));
        }
        if !self.closed && s > self.total {
            let (p, d) = self.on_link(last, self.links[last].len)?;
            return Ok((p + d * (s - self.total), d, last));
        }
        let i = self.link_at(s);
        let l = &self.links[i];
        let (p, d) = self.on_link(i, (s - l.s0).clamp(0.0, l.len))?;
        Ok((p, d, i))
    }

    fn surface(&self, f: FaceId) -> &Surface {
        &self.body.face(f).surface
    }

    /// The nearest point to `q` on face `f`'s surface, and the outward normal there. A spline
    /// surface runs on past the edge of its parameters as its end spans' polynomials, so a ball
    /// rolling on past the chain's end keeps a smooth contact instead of one held at that edge.
    fn foot_on(&self, f: FaceId, q: Point3) -> (Point3, Vec3) {
        let s = self.surface(f);
        let uv = s.project(q);
        if let Surface::Nurbs(n) = s
            && let Some(uv) = extended_foot(n, q, uv)
        {
            let face = self.body.face(f);
            let nrm = s.normal(uv.x, uv.y);
            return (s.eval(uv.x, uv.y), if face.reversed { -nrm } else { nrm });
        }
        let p = s.eval(uv.x, uv.y);
        (p, outward(self.body, f, p))
    }

    /// The junctions on side `k` at either end of link `i`.
    fn near(&self, i: usize, k: usize) -> Vec<&Junction> {
        let n = self.links.len();
        let mut out = Vec::new();
        if self.closed || i > 0 {
            out.extend(self.joints[(i + n - 1) % n][k].as_ref());
        }
        if self.closed || i + 1 < n {
            out.extend(self.joints[i][k].as_ref());
        }
        out
    }

    /// Side `k`'s contact for a centre `q` near link `i`: the nearest foot on that side's faces
    /// about the link that lies in its face, by the face's own trimmed domain. Where none does (past
    /// an open chain's end, or where the strip runs off its face) the ball rolls on the link's own
    /// face's surface.
    fn foot(&self, i: usize, k: usize, q: Point3) -> (Point3, Vec3, FaceId) {
        let own = self.links[i].faces[k];
        let near = self.near(i, k);
        let (p, nrm) = self.foot_on(own, q);
        let in_face = |f: FaceId, p: Point3| {
            self.domains
                .get(&f)
                .is_some_and(|d| d.locate(p) != Where::Outside)
        };
        // A foot on the own face's surface can lie in another face only where the two surfaces meet,
        // along the curve between them: far from every such curve it is the contact.
        let far = near.iter().filter(|j| !j.seam).all(|j| {
            let x = self.body.edge(j.edge);
            match x.curve.as_ref() {
                Some(c @ (Curve::Line(_) | Curve::Circle(_))) => nearest_param(c, p, x.range)
                    .map(|t| c.eval(t).dist(p) > 4.0 * self.radius)
                    .unwrap_or(false),
                _ => false,
            }
        });
        // With no other face on this side about the link, or a foot in the link's own face, that
        // is the contact: a foot lies inside at most one face.
        if far || in_face(own, p) {
            return (p, nrm, own);
        }
        let mut faces = vec![own];
        for j in near {
            for f in [j.before, j.after] {
                if !faces.contains(&f) {
                    faces.push(f);
                }
            }
        }
        let mut best: Option<(f64, Point3, Vec3, FaceId)> = None;
        for &f in &faces {
            if f == own {
                continue;
            }
            let (p, nrm) = self.foot_on(f, q);
            let inside = in_face(f, p);
            let d = p.dist(q);
            if inside && best.is_none_or(|b| d < b.0) {
                best = Some((d, p, nrm, f));
            }
        }
        match best {
            Some((_, p, nrm, f)) => (p, nrm, f),
            None => (p, nrm, own),
        }
    }

    /// The section at chain parameter `s`, starting Newton from `guess`, and from the chain where
    /// there is none or Newton does not settle from it: a neighbouring section's centre lies far
    /// off where a large ball rolls round a tight bend.
    fn section(&self, s: f64, guess: Option<Point3>) -> Result<Section> {
        match guess {
            Some(g) => self.settled(s, g).or_else(|_| self.section(s, None)),
            None => {
                let (e, _, i) = self.chain_at(s)?;
                // The centre of the ball in the wedge of the two faces' tangent planes at the
                // edge, `r` from both: `(na + nb)·r/(1 + na·nb)` off it.
                let na = outward(self.body, self.links[i].faces[0], e);
                let nb = outward(self.body, self.links[i].faces[1], e);
                let k = self.side * self.radius / (1.0 + na.dot(nb)).max(1e-3);
                self.settled(s, e + (na + nb) * k)
            }
        }
    }

    /// The section at chain parameter `s` by Newton steps from the centre `c`.
    fn settled(&self, s: f64, mut c: Point3) -> Result<Section> {
        let (e, along, i) = self.chain_at(s)?;
        for _ in 0..60 {
            let (pa, na, _) = self.foot(i, 0, c);
            let (pb, nb, _) = self.foot(i, 1, c);
            // The centres' path runs along both contact normals' cross product: square to it, the
            // plane through the centre holds both contacts. That direction turns with the centre,
            // so its row is the gradient of the distance from the plane, by central differences.
            let off_plane = |q: Point3| {
                let (_, na, _) = self.foot(i, 0, q);
                let (_, nb, _) = self.foot(i, 1, q);
                (q - e).dot(na.cross(nb).unit_or_zero())
            };
            let (row, third) = if self.spine {
                let h = 1e-6 * (1.0 + self.radius);
                let d = |v: Vec3| (off_plane(c + v * h) - off_plane(c - v * h)) / (2.0 * h);
                (Vec3::new(d(Vec3::X), d(Vec3::Y), d(Vec3::Z)), off_plane(c))
            } else {
                (along, (c - e).dot(along))
            };
            let f = [
                (c - pa).dot(na) - self.side * self.radius,
                (c - pb).dot(nb) - self.side * self.radius,
                third,
            ];
            let step = solve3([na, nb, row], f).ok_or_else(|| {
                Error::Unsupported("a rolling-ball fillet whose faces are parallel".into())
            })?;
            c -= step;
            if step.norm() <= 1e-3 * tol::LINEAR {
                let (a, _, fa) = self.foot(i, 0, c);
                let (b, _, fb) = self.foot(i, 1, c);
                if a.dist(e) > 10.0 * self.radius || b.dist(e) > 10.0 * self.radius {
                    return unsupported("whose ball leaves the edge");
                }
                return Ok(Section {
                    centre: c,
                    contact: [a, b],
                    face: [fa, fb],
                });
            }
        }
        unsupported("whose ball's centre does not settle")
    }

    /// Where, near the joint at the end of link `i`, side `k`'s contact crosses its junction.
    fn crossing(&self, i: usize, k: usize) -> Result<Option<f64>> {
        let Some(j) = &self.joints[i][k] else {
            return Ok(None);
        };
        let n = self.links.len();
        let next = (i + 1) % n;
        let at = self.links[next].s0 + if next == 0 { self.total } else { 0.0 };
        // Up to four radii either side of the joint, in each link's own parameter.
        let speed = |li: usize, d: f64| -> f64 {
            let l = &self.links[li];
            let e = self.body.edge(l.edge);
            let t = if l.forward {
                e.range.0 + d
            } else {
                e.range.1 - d
            };
            e.curve
                .as_ref()
                .map_or(1.0, |c| c.d1(t).1.norm())
                .max(1e-12)
        };
        let back = self.links[i]
            .len
            .min(4.0 * self.radius / speed(i, self.links[i].len));
        let fwd = self.links[next]
            .len
            .min(4.0 * self.radius / speed(next, 0.0));
        let s = self.surface(j.before);
        let seam_u = s.project(j.at).x;
        // Zero where the contact reaches the junction: across a seam's parameter, or across the
        // curve between two faces.
        let g = |t: f64, guess: Option<Point3>| -> Result<(f64, Point3)> {
            let sec = self.section(t, guess)?;
            let d = if j.seam {
                let up = s.u_period().unwrap_or(f64::INFINITY);
                let u = s.project(sec.contact[k]).x;
                (u - seam_u + 0.5 * up).rem_euclid(up) - 0.5 * up
            } else {
                // The foot on the face before the junction, across the junction curve from it.
                let x = self.body.edge(j.edge);
                let jc = x.curve.as_ref().expect("a junction has a curve");
                let (p, _) = self.foot_on(j.before, sec.centre);
                let tq = nearest_param(jc, p, x.range)?;
                let (q, tj) = jc.d1(tq);
                let across = outward(self.body, j.before, q).cross(tj).unit_or_zero();
                (p - q).dot(across)
            };
            Ok((d, sec.centre))
        };
        let steps = 32;
        let mut guess = None;
        let mut prev: Option<(f64, f64)> = None;
        for m in 0..=steps {
            let t = at - back + (back + fwd) * m as f64 / steps as f64;
            let (d, c) = g(t, guess)?;
            guess = Some(c);
            if d.abs() <= 1e-12 {
                return Ok(Some(t));
            }
            if let Some((tp, dp)) = prev {
                let wraps = j.seam && (d - dp).abs() > 1.0;
                if dp.signum() != d.signum() && !wraps {
                    return Ok(Some(bisect(tp, t, dp, |x| Ok(g(x, guess)?.0))?));
                }
            }
            prev = Some((t, d));
        }
        unsupported("whose contact never crosses onto the next face")
    }

    /// `s*(v)`: where the section's arc point at `v` meets end `e`'s plane, near `from`. Secant
    /// steps from `from`, each section's Newton started from the last centre; a search for a sign
    /// change and bisection where the secant strays.
    fn trim_at(&self, e: &End, v: f64, from: f64) -> Result<f64> {
        let guess = std::cell::Cell::new(None);
        let g = |s: f64| -> Result<f64> {
            let sec = self.section(s, guess.get())?;
            guess.set(Some(sec.centre));
            Ok(e.side(sec.at(self.radius, v)?))
        };
        let g0 = g(from)?;
        if g0 == 0.0 {
            return Ok(from);
        }
        let (mut a, mut ga) = (from, g0);
        let (mut b, mut gb) = (from + 1e-3 * self.radius, 0.0);
        gb = if b != a { g(b)? } else { gb };
        for _ in 0..40 {
            if gb == ga {
                break;
            }
            let c = b - gb * (b - a) / (gb - ga);
            if !c.is_finite() || (c - from).abs() > 20.0 * self.radius {
                break;
            }
            (a, ga) = (b, gb);
            b = c;
            gb = g(b)?;
            if gb == 0.0 || (b - a).abs() <= 1e-14 * (1.0 + b.abs()) {
                return Ok(b);
            }
        }
        let step = 0.25 * self.radius;
        for k in 1..=80 {
            for dir in [1.0, -1.0] {
                let (p, b) = (
                    from + dir * step * (k - 1) as f64,
                    from + dir * step * k as f64,
                );
                let gb = g(b)?;
                if gb.signum() != g0.signum() {
                    let gp = g(p)?;
                    return if p < b {
                        bisect(p, b, gp, g)
                    } else {
                        bisect(b, p, gb, g)
                    };
                }
            }
        }
        unsupported("whose end face never crosses it")
    }
}

/// Where the foot of `q` on spline surface `n` lies past the edge of its parameters, the
/// projection `uv` having stopped on that edge: Gauss–Newton steps on the surface's polynomial
/// continuation. None where `uv` lies inside, or the steps do not settle.
fn extended_foot(n: &NurbsSurface, q: Point3, uv: Vec2) -> Option<Vec2> {
    let ((u0, u1), (v0, v1)) = n.domain();
    let edge = |x: f64, a: f64, b: f64| {
        (x - a).abs() <= 1e-12 * (b - a) || (b - x).abs() <= 1e-12 * (b - a)
    };
    let periodic = (n.u_period().is_some(), n.v_period().is_some());
    let on_edge = (!periodic.0 && edge(uv.x, u0, u1)) || (!periodic.1 && edge(uv.y, v0, v1));
    if !on_edge {
        return None;
    }
    let mut uv = uv;
    for _ in 0..40 {
        let (p, su, sv) = n.d1(uv.x, uv.y);
        let r = q - p;
        let (a, b, c) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let det = a * c - b * b;
        if det.abs() <= 1e-300 {
            return None;
        }
        let (gu, gv) = (su.dot(r), sv.dot(r));
        let step = Vec2::new((c * gu - b * gv) / det, (a * gv - b * gu) / det);
        uv = uv + step;
        if step.x.abs() * su.norm() + step.y.abs() * sv.norm() <= 1e-3 * tol::LINEAR {
            return Some(uv);
        }
    }
    None
}

/// The root of `g` between `lo` and `hi`, `g(lo)` having the sign of `glo`.
fn bisect(mut lo: f64, mut hi: f64, glo: f64, g: impl Fn(f64) -> Result<f64>) -> Result<f64> {
    let mut glo = glo;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let gm = g(mid)?;
        if gm == 0.0 {
            return Ok(mid);
        }
        if gm.signum() == glo.signum() {
            (lo, glo) = (mid, gm);
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// `x` with `[r0; r1; r2]·x = f`, or `None` when the rows are dependent.
fn solve3(r: [Vec3; 3], f: [f64; 3]) -> Option<Vec3> {
    let det = r[0].dot(r[1].cross(r[2]));
    if det.abs() <= 1e-12 {
        return None;
    }
    let (c0, c1, c2) = (r[1].cross(r[2]), r[2].cross(r[0]), r[0].cross(r[1]));
    Some((c0 * f[0] + c1 * f[1] + c2 * f[2]) / det)
}

/// One segment's fitted blend from `a` to `b`.
struct Segment {
    a: f64,
    b: f64,
    link: usize,
    face: [FaceId; 2],
    surface: NurbsSurface,
    contact: [BSpline; 2],
}

fn fit(roll: &Roll, a: f64, b: f64) -> Result<Segment> {
    let link = roll.link_at(0.5 * (a + b));
    let mut guess = None;
    for &n in &SAMPLES {
        let params: Vec<f64> = (0..=n).map(|i| a + (b - a) * i as f64 / n as f64).collect();
        let mut secs = Vec::with_capacity(n + 1);
        for &t in &params {
            let s = roll.section(t, guess)?;
            guess = Some(s.centre);
            secs.push(s);
        }
        guess = Some(secs[0].centre);
        let mut rows: [Vec<Vec3>; 4] = Default::default();
        for s in &secs {
            let (m, w) = s.arc(roll.radius)?;
            rows[0].push(s.contact[0]);
            rows[1].push(m * w);
            rows[2].push(s.contact[1]);
            rows[3].push(Vec3::new(w, 0.0, 0.0));
        }
        let splines: Vec<BSpline> = rows
            .iter()
            .map(|r| BSpline::interpolate_at(r, &params))
            .collect::<Result<_>>()?;
        let knots = splines[0].knots().to_vec();
        let count = splines[0].control_points().len();
        let mut ctrl = Vec::with_capacity(3 * count);
        let mut weights = Vec::with_capacity(3 * count);
        for i in 0..count {
            let w = splines[3].control_points()[i].x;
            if w <= 0.0 {
                return unsupported("whose sections turn too fast to fit");
            }
            ctrl.extend([
                splines[0].control_points()[i],
                splines[1].control_points()[i] / w,
                splines[2].control_points()[i],
            ]);
            weights.extend([1.0, w, 1.0]);
        }
        let surface = NurbsSurface::new(
            (3, 2),
            (knots, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            (count, 3),
            ctrl,
            Some(weights),
        )?;
        let mid = roll.section(0.5 * (a + b), Some(secs[n / 2].centre))?;
        let mut worst = 0.0f64;
        for (i, near) in secs[..n].iter().enumerate() {
            let t = a + (b - a) * (i as f64 + 0.5) / n as f64;
            let s = roll.section(t, Some(near.centre))?;
            if s.face != mid.face {
                return unsupported("whose contact changes face inside a segment");
            }
            for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                worst = worst.max(surface.eval(t, v).dist(s.at(roll.radius, v)?));
            }
            for (k, v) in [(0, 0.0), (1, 1.0)] {
                let p = surface.eval(t, v);
                worst = worst.max(roll.foot_on(s.face[k], p).0.dist(p));
            }
        }
        if worst <= FIT * tol::LINEAR {
            let mut it = splines.into_iter();
            let c0 = it.next().expect("four splines");
            let _ = it.next();
            let c1 = it.next().expect("four splines");
            return Ok(Segment {
                a,
                b,
                link,
                face: mid.face,
                surface,
                contact: [c0, c1],
            });
        }
    }
    unsupported("whose sections do not fit within the tolerance")
}

/// An end's trim: `s*` against `v`, and the fitted curve of the points where it meets the plane.
struct Trim {
    /// `(v, s*)` samples, `v` increasing from 0 to 1.
    samples: Vec<(f64, f64)>,
    curve: BSpline,
}

impl Trim {
    /// `s*` at `v`, read off the samples.
    fn s_at(&self, v: f64) -> f64 {
        let k = self
            .samples
            .windows(2)
            .position(|w| v <= w[1].0)
            .unwrap_or(self.samples.len() - 2);
        let ((v0, s0), (v1, s1)) = (self.samples[k], self.samples[k + 1]);
        s0 + (s1 - s0) * (v - v0) / (v1 - v0)
    }
}

fn fit_trim(roll: &Roll, e: &End, from: f64) -> Result<Trim> {
    for &n in &SAMPLES {
        let vs: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let mut samples = Vec::with_capacity(n + 1);
        let mut pts = Vec::with_capacity(n + 1);
        let mut last = from;
        for &v in &vs {
            let s = roll.trim_at(e, v, last)?;
            last = s;
            samples.push((v, s));
            pts.push(roll.section(s, None)?.at(roll.radius, v)?);
        }
        let curve = BSpline::interpolate_at(&pts, &vs)?;
        let mut worst = 0.0f64;
        for (i, &(_, near)) in samples[..n].iter().enumerate() {
            let v = (i as f64 + 0.5) / n as f64;
            let s = roll.trim_at(e, v, near)?;
            let p = roll.section(s, None)?.at(roll.radius, v)?;
            worst = worst.max(curve.eval(v).dist(p));
        }
        if worst <= FIT * tol::LINEAR {
            return Ok(Trim { samples, curve });
        }
    }
    unsupported("whose trim by its end face does not fit within the tolerance")
}

/// The picked edges as one chain run the way side 0's faces use them, closed or open, or `None`.
fn chain(body: &Brep, picked: &[Picked]) -> Option<(Vec<Link>, bool)> {
    let end_of = |i: usize| body.coedge_end(picked[i].coedges[0]);
    let start_of = |i: usize| body.coedge_start(picked[i].coedges[0]);
    let first = (0..picked.len())
        .find(|&i| !(0..picked.len()).any(|j| end_of(j) == start_of(i)))
        .unwrap_or(0);
    let mut order = vec![first];
    let mut used = vec![false; picked.len()];
    used[first] = true;
    while order.len() < picked.len() {
        let v = end_of(*order.last()?);
        let next = (0..picked.len()).find(|&j| !used[j] && start_of(j) == v)?;
        used[next] = true;
        order.push(next);
    }
    let closed = end_of(*order.last()?) == start_of(order[0]);
    let mut s0 = 0.0;
    let mut links = Vec::with_capacity(order.len());
    for i in order {
        let pk = &picked[i];
        let e = body.edge(pk.edge);
        let len = e.range.1 - e.range.0;
        links.push(Link {
            pk: i,
            edge: pk.edge,
            forward: !body.coedge(pk.coedges[0]).reversed,
            faces: pk.faces,
            s0,
            len,
        });
        s0 += len;
    }
    Some((links, closed))
}

/// The fillet of `radius` along the tangent chain `picked`, edge `i`'s faces named `names[i]`.
/// Each section lies in the plane square to the chain where it crosses it; where that plane misses
/// the ball's path (a ball larger than the chain's bend, whose square planes cross before they
/// reach the centres), in the plane square to the centres' path through the chain's point
/// instead, the plane the ball's circle of contact lies in.
pub(super) fn fillet(
    body: &Brep,
    picked: &[Picked],
    radius: f64,
    names: &[BTreeSet<FaceName>],
) -> Result<Brep> {
    rolled(body, picked, radius, names, false)
        .or_else(|e| rolled(body, picked, radius, names, true).map_err(|_| e))
}

/// [`fillet`], its sections square to the centres' path where `spine`, else to the chain.
fn rolled(
    body: &Brep,
    picked: &[Picked],
    radius: f64,
    names: &[BTreeSet<FaceName>],
    spine: bool,
) -> Result<Brep> {
    if body.has_tolerances() {
        return unsupported("on a body with local tolerances");
    }
    if picked.iter().any(|pk| body.edge(pk.edge).curve.is_none()) {
        return unsupported("on a degenerate edge");
    }
    let Some((links, closed)) = chain(body, picked) else {
        return unsupported("on edges that do not run on from one to the next");
    };
    let Some(convex) = run::convex(body, picked[0].coedges[0]) else {
        return unsupported("between faces that meet tangentially");
    };
    if picked
        .iter()
        .any(|pk| run::convex(body, pk.coedges[0]) != Some(convex))
    {
        return unsupported("on a chain that is convex in places and concave in others");
    }
    let n = links.len();
    let total: f64 = links.iter().map(|l| l.len).sum();
    let chain_edges: BTreeSet<EdgeId> = links.iter().map(|l| l.edge).collect();
    let mut uses: HashMap<EdgeId, Vec<CoedgeId>> = HashMap::new();
    for c in body.coedge_ids() {
        uses.entry(body.coedge(c).edge).or_default().push(c);
    }
    let faces_of =
        |e: EdgeId| -> BTreeSet<FaceId> { uses[&e].iter().map(|&c| body.coedge_face(c)).collect() };
    let at_vertex = |v: VertexId| -> Vec<EdgeId> {
        body.edge_ids()
            .filter(|&e| {
                let x = body.edge(e);
                x.start == v || x.end == v
            })
            .collect()
    };
    let tangent = |link: &Link, leaving: bool| -> Vec3 {
        let e = body.edge(link.edge);
        let c = e.curve.as_ref().expect("checked above");
        let at_end = leaving != link.forward;
        let t = if at_end { e.range.1 } else { e.range.0 };
        let d = c.d1(t).1.unit_or_zero();
        if link.forward == leaving { d } else { -d }
    };

    // The junctions at each inner chain vertex, and every other edge there refused.
    let inner = if closed { n } else { n - 1 };
    let mut joints: Vec<[Option<Junction>; 2]> = Vec::with_capacity(inner);
    let mut vertices: Vec<VertexId> = Vec::with_capacity(inner);
    for i in 0..inner {
        let (l, m) = (&links[i], &links[(i + 1) % n]);
        let v = body.coedge_end(picked[l.pk].coedges[0]);
        vertices.push(v);
        let at = body.vertex(v).point;
        let mut pair: [Option<Junction>; 2] = [None, None];
        let mut taken: BTreeSet<EdgeId> = BTreeSet::new();
        #[allow(clippy::needless_range_loop)]
        for k in 0..2 {
            let (fa, fb) = (l.faces[k], m.faces[k]);
            let found = at_vertex(v).into_iter().find(|&e| {
                !chain_edges.contains(&e)
                    && faces_of(e) == BTreeSet::from([fa, fb])
                    && (fa != fb || uses[&e].len() == 2)
            });
            let Some(e) = found else {
                if fa != fb {
                    return unsupported("where a side changes face without an edge between them");
                }
                continue;
            };
            taken.insert(e);
            let x = body.edge(e);
            let Some(c) = x.curve.as_ref() else {
                return unsupported("at a pole");
            };
            let _ = c;
            pair[k] = Some(Junction {
                edge: e,
                at,
                before: fa,
                after: fb,
                seam: fa == fb,
            });
        }
        if at_vertex(v)
            .iter()
            .any(|e| !chain_edges.contains(e) && !taken.contains(e))
        {
            return unsupported("through a vertex other faces meet");
        }
        joints.push(pair);
    }
    let roll = Roll {
        body,
        links,
        closed,
        total,
        radius,
        side: if convex { -1.0 } else { 1.0 },
        spine,
        domains: picked
            .iter()
            .flat_map(|pk| pk.faces)
            .chain(
                joints
                    .iter()
                    .flatten()
                    .flatten()
                    .flat_map(|j| [j.before, j.after]),
            )
            .collect::<BTreeSet<FaceId>>()
            .into_iter()
            .map(|f| (f, face_domain(body, f)))
            .collect(),
        joints,
    };

    // An open chain's ends, each on the plane of the one other face there.
    let mut ends: Vec<End> = Vec::new();
    if !closed {
        for (at_start, li) in [(true, 0), (false, n - 1)] {
            let l = &roll.links[li];
            let v = if at_start {
                body.coedge_start(picked[l.pk].coedges[0])
            } else {
                body.coedge_end(picked[l.pk].coedges[0])
            };
            let others: BTreeSet<FaceId> = at_vertex(v)
                .iter()
                .flat_map(|&e| faces_of(e))
                .filter(|f| !l.faces.contains(f))
                .collect();
            let [f] = others.iter().copied().collect::<Vec<_>>()[..] else {
                return unsupported("ending where more than one other face meets it");
            };
            let mut found = Vec::with_capacity(2);
            for k in 0..2 {
                let Some(e) = at_vertex(v)
                    .into_iter()
                    .find(|&e| faces_of(e) == BTreeSet::from([f, l.faces[k]]))
                else {
                    return unsupported("ending where its end face meets a side at no edge");
                };
                found.push(e);
            }
            let edges = [found[0], found[1]];
            if at_vertex(v).len() != 3 {
                return unsupported("ending at a vertex where other edges meet");
            }
            // The side the chain comes from: a point a little way back along it.
            let surface = body.face(f).surface.clone();
            let back = tangent(l, at_start);
            let at = body.vertex(v).point;
            let uv = surface.project(at);
            let nrm = surface.normal(uv.x, uv.y);
            let sign = if nrm.dot(back) >= 0.0 { 1.0 } else { -1.0 };
            let s_end = if at_start { 0.0 } else { total };
            ends.push(End {
                vertex: v,
                face: f,
                surface,
                sign,
                edges,
                lo: s_end,
                hi: s_end,
            });
        }
    }
    let mut trims: Vec<Trim> = Vec::with_capacity(ends.len());
    for (k, e) in ends.iter_mut().enumerate() {
        let from = if k == 0 { 0.0 } else { total };
        let t = fit_trim(&roll, e, from)?;
        e.lo = t.samples.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        e.hi = t
            .samples
            .iter()
            .map(|x| x.1)
            .fold(f64::NEG_INFINITY, f64::max);
        trims.push(t);
    }

    // From each end's vertex along its face's boundary, on each side, to the edge that side's
    // contact ends on: the edges passed over and the vertices between them go with the blend.
    let mut last: Vec<[(EdgeId, bool); 2]> = Vec::with_capacity(ends.len());
    let mut dropped_edges: BTreeSet<EdgeId> = BTreeSet::new();
    let mut dropped_vertices: BTreeSet<VertexId> = BTreeSet::new();
    for (ti, e) in ends.iter().enumerate() {
        let mut found = Vec::with_capacity(2);
        for k in 0..2 {
            let p = trims[ti].curve.eval(k as f64);
            let (mut edge, mut from) = (e.edges[k], e.vertex);
            let mut steps = 0;
            loop {
                let x = body.edge(edge);
                let c = x.curve.as_ref().expect("an end face's edge has a curve");
                let starts = x.start == from;
                // A line, circle or ellipse runs on past its ends, so the first edge's contact may
                // lie past the vertex, where the blend opens the end face.
                let span = x.range.1 - x.range.0;
                let reach = match c {
                    Curve::Line(_) | Curve::Circle(_) | Curve::Ellipse(_) if steps == 0 => {
                        (x.range.0 - span, x.range.1 + span)
                    }
                    _ => x.range,
                };
                if let Ok(t) = nearest_param(c, p, reach) {
                    let inside = t >= x.range.0 - tol::LINEAR && t <= x.range.1 + tol::LINEAR;
                    // Where the end face turns past a half turn at the vertex, the blend opens it
                    // beyond the corner: its first edge runs on past the vertex to the contact.
                    let past = steps == 0 && if starts { t < x.range.0 } else { t > x.range.1 };
                    if (inside || past) && c.eval(t).dist(p) <= 10.0 * tol::LINEAR {
                        found.push((edge, starts));
                        break;
                    }
                }
                steps += 1;
                let w = if starts { x.end } else { x.start };
                let onward: Vec<EdgeId> = at_vertex(w).into_iter().filter(|&o| o != edge).collect();
                let (&[next], true) = (onward.as_slice(), steps < 6) else {
                    return unsupported("whose contact ends past more of its end face than a run");
                };
                if !faces_of(next).contains(&e.face) {
                    return unsupported("whose contact ends off its end face");
                }
                dropped_edges.insert(edge);
                dropped_vertices.insert(w);
                (edge, from) = (next, w);
            }
        }
        last.push([found[0], found[1]]);
    }

    // Segment boundaries: every chain vertex, every crossing, and an open chain's two ends past
    // its trims.
    // An open chain's ends bound segments too: past them it runs on straight, a kink in the
    // sections' turning that no one cubic follows.
    let mut cuts: Vec<f64> = roll.links.iter().map(|l| l.s0).collect();
    if !closed {
        cuts.push(total);
    }
    let mut crossing: Vec<[Option<f64>; 2]> = Vec::with_capacity(inner);
    for i in 0..inner {
        let mut c = [None, None];
        for (k, ck) in c.iter_mut().enumerate() {
            *ck = roll.crossing(i, k)?.map(|t| roll.wrap(t));
            if let Some(t) = *ck {
                cuts.push(t);
            }
        }
        crossing.push(c);
    }
    let margin = 0.25 * radius;
    let (lo, hi) = if closed {
        (0.0, total)
    } else {
        (ends[0].lo - margin, ends[1].hi + margin)
    };
    if !closed {
        cuts.retain(|&t| t > lo && t < hi);
        cuts.push(lo);
        cuts.push(hi);
    }
    cuts.sort_by(f64::total_cmp);
    let close = |a: f64, b: f64| {
        let d = if closed {
            let d = (a - b).rem_euclid(total);
            d.min(total - d)
        } else {
            (a - b).abs()
        };
        d <= 1e-9 * total.max(1.0)
    };
    let mut bounds: Vec<f64> = Vec::new();
    for t in cuts {
        if !bounds.iter().any(|&b| close(t, b)) {
            bounds.push(t);
        }
    }
    let m = if closed {
        bounds.len()
    } else {
        bounds.len() - 1
    };
    let mut segments = Vec::with_capacity(m);
    for j in 0..m {
        let a = bounds[j];
        let b = if j + 1 < bounds.len() {
            bounds[j + 1]
        } else {
            bounds[0] + total
        };
        segments.push(fit(&roll, a, b)?);
    }
    let bound_index =
        |t: f64| -> Option<usize> { (0..bounds.len()).find(|&x| close(t, bounds[x])) };

    // Where each segment is kept: `s` from `lower(v)` to `upper(v)`.
    let start_trim = (!closed).then(|| &trims[0]);
    let end_trim = (!closed).then(|| &trims[1]);
    let lower = |j: usize, v: f64| match start_trim {
        Some(t) => segments[j].a.max(t.s_at(v)),
        None => segments[j].a,
    };
    let upper = |j: usize, v: f64| match end_trim {
        Some(t) => segments[j].b.min(t.s_at(v)),
        None => segments[j].b,
    };
    let alive: Vec<bool> = (0..m)
        .map(|j| (0..=64).any(|i| upper(j, i as f64 / 64.0) > lower(j, i as f64 / 64.0) + 1e-9))
        .collect();

    // The rebuilt solid.
    let mut out = Brep::new();
    let mut gone: BTreeSet<VertexId> = vertices.iter().copied().collect();
    gone.extend(ends.iter().map(|e| e.vertex));
    gone.extend(dropped_vertices.iter().copied());
    let mut vmap: Vec<Option<VertexId>> = vec![None; body.vertex_ids().count()];
    for w in body.vertex_ids() {
        if !gone.contains(&w) {
            vmap[w.index()] = Some(out.add_vertex(body.vertex(w).point));
        }
    }

    // Vertices on each segment's corners and on the trims, shared by position.
    let mut pool: Vec<VertexId> = Vec::new();
    let mut vertex_at = |out: &mut Brep, p: Point3| -> VertexId {
        if let Some(&w) = pool
            .iter()
            .find(|&&w| out.vertex(w).point.dist(p) <= tol::LINEAR)
        {
            return w;
        }
        let w = out.add_vertex(p);
        pool.push(w);
        w
    };
    // The trims' curves, split where a segment's arc meets them.
    let mut trim_splits: Vec<BTreeSet<OrderedF64>> = vec![BTreeSet::new(); trims.len()];
    // Where each trim meets each bound's arc: (bound index, v).
    let mut meets: Vec<(usize, f64)> = Vec::new();
    for (ti, t) in trims.iter().enumerate() {
        let mut vs = vec![0.0, 1.0];
        for (bi, &b) in bounds.iter().enumerate() {
            for w in t.samples.windows(2) {
                let ((v0, s0), (v1, s1)) = (w[0], w[1]);
                if (s0 - b) * (s1 - b) < 0.0 {
                    // Exactly where the true trim meets the bound's arc, so the two share a vertex.
                    let g = |v: f64| Ok(roll.trim_at(&ends[ti], v, s0)? - b);
                    let v = bisect(v0, v1, s0 - b, g)?;
                    vs.push(v);
                    meets.push((bi, v));
                }
            }
        }
        // The trim's own ends first, so a crossing a hair from either merges into it.
        vs[2..].sort_by(f64::total_cmp);
        for v in vs {
            let p = t.curve.eval(v);
            let near = |x: &OrderedF64| t.curve.eval(x.0).dist(p) <= 10.0 * tol::LINEAR;
            if !trim_splits[ti].iter().any(near) {
                trim_splits[ti].insert(OrderedF64(v));
            }
        }
    }
    let mut trim_edges: Vec<BTreeMap<OrderedF64, EdgeId>> = vec![BTreeMap::new(); trims.len()];
    for (ti, t) in trims.iter().enumerate() {
        let vs: Vec<f64> = trim_splits[ti].iter().map(|x| x.0).collect();
        for w in vs.windows(2) {
            let a = vertex_at(&mut out, t.curve.eval(w[0]));
            let b = vertex_at(&mut out, t.curve.eval(w[1]));
            let e = out.add_edge(Curve::BSpline(t.curve.clone()), (w[0], w[1]), a, b);
            trim_edges[ti].insert(OrderedF64(w[0]), e);
        }
    }
    // Each inner bound's arc, split where a trim crosses it, kept where some segment keeps it.
    let mut arcs: Vec<Vec<(f64, f64, EdgeId)>> = vec![Vec::new(); bounds.len()];
    for (bi, &b) in bounds.iter().enumerate() {
        let sec = roll.section(b, None)?;
        let curve = sec.arc_curve(radius)?;
        let mut vs = vec![0.0, 1.0];
        let _ = b;
        for &(at, v) in &meets {
            let snapped = trim_splits
                .iter()
                .flatten()
                .map(|x| x.0)
                .find(|&x| (x - v).abs() <= 1e-6)
                .unwrap_or(v);
            if at == bi && snapped > 0.0 && snapped < 1.0 {
                vs.push(snapped);
            }
        }
        vs.sort_by(f64::total_cmp);
        for w in vs.windows(2) {
            let mid = 0.5 * (w[0] + w[1]);
            let kept_by = |j: usize| -> bool {
                alive[j] && lower(j, mid) < upper(j, mid) - 1e-9 && {
                    let (a, c) = (lower(j, mid), upper(j, mid));
                    close(a, b) || close(c, b)
                }
            };
            if !(0..m).any(kept_by) {
                continue;
            }
            let a = vertex_at(&mut out, curve.eval(w[0]));
            let c = vertex_at(&mut out, curve.eval(w[1]));
            let e = out.add_edge(Curve::Nurbs(Box::new(curve.clone())), (w[0], w[1]), a, c);
            arcs[bi].push((w[0], w[1], e));
        }
    }
    // Each kept segment's two contact pieces.
    let mut pieces: Vec<[Option<EdgeId>; 2]> = vec![[None, None]; m];
    for j in 0..m {
        if !alive[j] {
            continue;
        }
        for (k, v) in [(0, 0.0), (1, 1.0)] {
            let (a, b) = (lower(j, v), upper(j, v));
            if b <= a + 1e-9 {
                continue;
            }
            let c = Curve::BSpline(segments[j].contact[k].clone());
            let va = vertex_at(&mut out, c.eval(a));
            let vb = vertex_at(&mut out, c.eval(b));
            pieces[j][k] = Some(out.add_edge(c, (a, b), va, vb));
        }
    }

    // The body's edges: each junction cut back to where its side's contact crosses it, each end
    // face's edges with the sides cut back to the contacts' ends.
    // Per edge, the point its start is cut back to and the point its end is.
    let mut cut_at: HashMap<EdgeId, [Option<Point3>; 2]> = HashMap::new();
    let mut cut = |e: EdgeId, starts: bool, p: Point3| -> Result<()> {
        let slot = &mut cut_at.entry(e).or_default()[if starts { 0 } else { 1 }];
        if slot.replace(p).is_some() {
            return unsupported("that cuts one end of an edge back twice");
        }
        Ok(())
    };
    for i in 0..inner {
        for (k, &crossed) in crossing[i].iter().enumerate() {
            if let (Some(j), Some(t)) = (&roll.joints[i][k], crossed) {
                if bound_index(t).is_none() {
                    return unsupported("whose crossing is not a segment's end");
                }
                let p = roll.section(t, None)?.contact[k];
                cut(j.edge, body.edge(j.edge).start == vertices[i], p)?;
            }
        }
    }
    for (ti, l) in last.iter().enumerate() {
        for (k, &(edge, starts)) in l.iter().enumerate() {
            cut(edge, starts, trims[ti].curve.eval(k as f64))?;
        }
    }
    let mut emap: Vec<Option<EdgeId>> = vec![None; body.edge_ids().count()];
    for e in body.edge_ids() {
        if chain_edges.contains(&e) || dropped_edges.contains(&e) {
            continue;
        }
        let x = body.edge(e);
        let Some(c) = x.curve.clone() else {
            let Some(w) = vmap[x.start.index()] else {
                return unsupported("through a pole at a chain vertex");
            };
            emap[e.index()] = Some(out.add_degenerate_edge(w, x.range));
            continue;
        };
        let mut range = x.range;
        let (mut na, mut nb) = (vmap[x.start.index()], vmap[x.end.index()]);
        if let Some(&[at_start, at_end]) = cut_at.get(&e) {
            // An edge may run on past its end to a contact, where the blend opens its face.
            let span = x.range.1 - x.range.0;
            let reach = match c {
                Curve::Line(_) | Curve::Circle(_) | Curve::Ellipse(_) => {
                    (x.range.0 - span, x.range.1 + span)
                }
                _ => x.range,
            };
            if let Some(p) = at_start {
                range.0 = nearest_param(&c, p, reach)?;
                na = Some(vertex_at(&mut out, p));
            }
            if let Some(p) = at_end {
                range.1 = nearest_param(&c, p, reach)?;
                nb = Some(vertex_at(&mut out, p));
            }
            let off =
                |t: f64, p: Option<Point3>| p.is_some_and(|p| c.eval(t).dist(p) > tol::LINEAR);
            if off(range.0, at_start) || off(range.1, at_end) {
                return unsupported("that cuts an edge to a point off its curve");
            }
            if range.1 - range.0 <= tol::LINEAR {
                return unsupported("that cuts an edge back past its other end");
            }
        }
        let (Some(na), Some(nb)) = (na, nb) else {
            return unsupported("along an edge whose vertex another edge ends on");
        };
        emap[e.index()] = Some(out.add_edge(c, range, na, nb));
    }

    // The body's faces: each run of chain edges swapped for the contact pieces on that face, and
    // each end face's corner swapped for its trim curve.
    let mut fmap: Vec<Option<FaceId>> = vec![None; body.face_ids().count()];
    for f in body.face_ids() {
        let face = body.face(f);
        let nf = out.add_face(
            face.surface.clone(),
            face.reversed,
            face.names.iter().cloned(),
        );
        fmap[f.index()] = Some(nf);
        let end_here: Vec<usize> = (0..ends.len()).filter(|&t| ends[t].face == f).collect();
        for &lp in &face.loops {
            let coedges = &body.lp(lp).coedges;
            let is_chain = |c: CoedgeId| chain_edges.contains(&body.coedge(c).edge);
            let first = coedges.iter().position(|&c| !is_chain(c)).unwrap_or(0);
            let order: Vec<CoedgeId> = (0..coedges.len())
                .map(|i| coedges[(first + i) % coedges.len()])
                .collect();
            let mut specs: Vec<CoedgeSpec> = Vec::new();
            let mut hint: Option<Vec2> = None;
            let runs = (0..order.len())
                .filter(|&x| is_chain(order[x]) && (x == 0 || !is_chain(order[x - 1])))
                .count();
            let push_new = |specs: &mut Vec<CoedgeSpec>,
                            hint: &mut Option<Vec2>,
                            out: &Brep,
                            e: EdgeId,
                            reversed: bool| {
                let x = out.edge(e);
                let curve = x.curve.as_ref().expect("a new edge has a curve");
                let (ta, tb) = if reversed {
                    (x.range.1, x.range.0)
                } else {
                    (x.range.0, x.range.1)
                };
                let start = curve.eval(ta);
                let seed = match *hint {
                    Some(h) => face.surface.project_near(start, h),
                    None => face.surface.project(start),
                };
                let mut uv = seed;
                for q in 1..=32 {
                    let t = ta + (tb - ta) * q as f64 / 32.0;
                    uv = face.surface.project_near(curve.eval(t), uv);
                }
                *hint = Some(uv);
                specs.push(CoedgeSpec {
                    edge: e,
                    reversed,
                    pcurve: PCurve::Projected { seed },
                });
            };
            let mut i = 0;
            while i < order.len() {
                let c = order[i];
                if dropped_edges.contains(&body.coedge(c).edge) {
                    i += 1;
                    continue;
                }
                if !is_chain(c) {
                    let co = body.coedge(c);
                    let ne = emap[co.edge.index()].expect("every other edge is copied");
                    let x = out.edge(ne);
                    let (from, to) = if co.reversed {
                        (x.end, x.start)
                    } else {
                        (x.start, x.end)
                    };
                    let orig = body.edge(co.edge);
                    let orig_end = if co.reversed {
                        orig.range.0
                    } else {
                        orig.range.1
                    };
                    let end_uv = body.coedge_uvs(c, &[orig_end])[0];
                    let pcurve = match &co.pcurve {
                        PCurve::Projected { seed } => PCurve::Projected {
                            seed: face.surface.project_near(out.vertex(from).point, *seed),
                        },
                        other => other.clone(),
                    };
                    specs.push(CoedgeSpec {
                        edge: ne,
                        reversed: co.reversed,
                        pcurve,
                    });
                    hint = Some(face.surface.project_near(out.vertex(to).point, end_uv));
                    // At an end face's corner, its trim curve runs from this edge to the next.
                    for &t in &end_here {
                        // The walked-to edge on side k, cut where this coedge now ends.
                        let k = last[t]
                            .iter()
                            .position(|&(x, starts)| x == co.edge && starts == co.reversed);
                        let Some(k) = k else {
                            continue;
                        };
                        // From side k's contact end (v = k) to the other's.
                        let mut run: Vec<(EdgeId, bool)> =
                            trim_edges[t].values().map(|&te| (te, false)).collect();
                        if k == 1 {
                            run.reverse();
                            for r in &mut run {
                                r.1 = true;
                            }
                        }
                        for (te, rev) in run {
                            push_new(&mut specs, &mut hint, &out, te, rev);
                        }
                    }
                    i += 1;
                    continue;
                }
                let co = body.coedge(c);
                let li = roll
                    .links
                    .iter()
                    .position(|l| l.edge == co.edge)
                    .expect("a chain edge");
                let k = (0..2)
                    .find(|&k| roll.links[li].faces[k] == f)
                    .expect("a chain edge's face is one of its sides");
                let along = k == 0;
                let mut run = 0;
                while i + run < order.len() && is_chain(order[i + run]) {
                    run += 1;
                }
                let mut mine: Vec<usize> = (0..m)
                    .filter(|&j| pieces[j][k].is_some() && segments[j].face[k] == f)
                    .collect();
                // A loop the chain passes through more than once (a closed chain across the
                // face's seam) takes for each run only the contact between the crossings at its
                // ends.
                if runs > 1 {
                    let links: BTreeSet<usize> = order[i..i + run]
                        .iter()
                        .map(|&c| {
                            let e = body.coedge(c).edge;
                            roll.links
                                .iter()
                                .position(|l| l.edge == e)
                                .expect("a chain edge")
                        })
                        .collect();
                    let n = roll.links.len();
                    let in_run = |x: usize| links.contains(&x);
                    let before = |x: usize| (closed || x > 0).then(|| (x + n - 1) % n);
                    let after = |x: usize| (closed || x + 1 < n).then_some(x);
                    let first = *links
                        .iter()
                        .find(|&&x| before(x).is_none_or(|b| !in_run(b)))
                        .expect("a run has a first link");
                    let last = *links
                        .iter()
                        .find(|&&x| after(x).is_none_or(|a| !in_run((a + 1) % n)))
                        .expect("a run has a last link");
                    let from = before(first)
                        .and_then(|vi| crossing[vi][k])
                        .unwrap_or(roll.links[first].s0);
                    let to = after(last)
                        .and_then(|vi| crossing[vi][k])
                        .unwrap_or(roll.links[last].s0 + roll.links[last].len);
                    let span = roll.wrap(to - from);
                    mine.retain(|&j| {
                        let past = roll.wrap(0.5 * (segments[j].a + segments[j].b) - from);
                        past >= 0.0 && past < span
                    });
                }
                if closed
                    && let Some(gap) = (0..mine.len()).find(|&x| {
                        let (p, q) = (mine[x], mine[(x + 1) % mine.len()]);
                        (p + 1) % m != q
                    })
                {
                    let len = mine.len();
                    mine.rotate_left((gap + 1) % len);
                }
                if mine.windows(2).any(|w| (w[0] + 1) % m != w[1]) {
                    return unsupported("whose contact leaves a face and comes back to it");
                }
                // A closed chain's contact all round the face starts where the loop's run does:
                // the chain's own start may lie on a later edge of the run.
                if closed && runs == 1 && mine.len() == m {
                    let l = &roll.links[li];
                    let start = roll.wrap(if along { l.s0 } else { l.s0 + l.len });
                    let close = |a: f64| {
                        let d = roll.wrap(a - start);
                        d.min(roll.total - d) <= 1e-9 * (1.0 + roll.total)
                    };
                    if let Some(p) = mine.iter().position(|&j| close(segments[j].a)) {
                        mine.rotate_left(p);
                    }
                }
                if !along {
                    mine.reverse();
                }
                for j in mine {
                    let e = pieces[j][k].expect("filtered above");
                    push_new(&mut specs, &mut hint, &out, e, !along);
                }
                i += run;
            }
            out.add_loop(nf, specs);
        }
    }

    // The segments' faces: side 0's contact against its use there, then round by the arcs and
    // trims at its start, side 1's contact, and the arcs and trims at its end.
    let row = |v: f64| PCurve::Affine {
        origin: Vec2::new(0.0, v),
        dir: Vec2::new(1.0, 0.0),
    };
    let at_u = |u: f64| PCurve::Affine {
        origin: Vec2::new(u, 0.0),
        dir: Vec2::new(0.0, 1.0),
    };
    let mut blends = Vec::with_capacity(m);
    for (j, sg) in segments.iter().enumerate() {
        if !alive[j] {
            continue;
        }
        let pk = roll.links[sg.link].pk;
        let surface = Surface::Nurbs(Box::new(sg.surface.clone()));
        let bf = out.add_face(surface.clone(), true, names[pk].iter().cloned());
        let mut specs = Vec::new();
        let on_face = |e: EdgeId, reversed: bool, hint: Vec2, out: &Brep| -> CoedgeSpec {
            let x = out.edge(e);
            let t = if reversed { x.range.1 } else { x.range.0 };
            let p = x.curve.as_ref().expect("a curve").eval(t);
            CoedgeSpec {
                edge: e,
                reversed,
                pcurve: PCurve::Projected {
                    seed: surface.project_near(p, hint),
                },
            }
        };
        if let Some(e) = pieces[j][0] {
            specs.push(CoedgeSpec {
                edge: e,
                reversed: true,
                pcurve: row(0.0),
            });
        }
        // The start side, v rising: the arc at `a` where kept, the start trim where it cuts.
        let side_run = |at_start: bool, specs: &mut Vec<CoedgeSpec>, out: &Brep| {
            let b_here = if at_start { sg.a } else { sg.b };
            let bi = bound_index(b_here).expect("a segment ends on a bound");
            let mut parts: Vec<(f64, CoedgeSpec)> = Vec::new();
            for &(v0, v1, e) in &arcs[bi] {
                let mid = 0.5 * (v0 + v1);
                let edge_here = if at_start {
                    close(lower(j, mid), b_here)
                } else {
                    close(upper(j, mid), b_here)
                };
                if edge_here {
                    parts.push((
                        v0,
                        CoedgeSpec {
                            edge: e,
                            reversed: !at_start,
                            pcurve: at_u(b_here),
                        },
                    ));
                }
            }
            let ti = if at_start { 0 } else { 1 };
            if !closed {
                for (&OrderedF64(v0), &e) in &trim_edges[ti] {
                    let x = out.edge(e);
                    let mid = 0.5 * (x.range.0 + x.range.1);
                    let s = trims[ti].s_at(mid);
                    if s > sg.a + 1e-9 && s < sg.b - 1e-9 {
                        parts.push((v0, on_face(e, !at_start, Vec2::new(s, mid), out)));
                    }
                }
            }
            parts.sort_by(|x, y| x.0.total_cmp(&y.0));
            if !at_start {
                parts.reverse();
            }
            specs.extend(parts.into_iter().map(|x| x.1));
        };
        side_run(true, &mut specs, &out);
        if let Some(e) = pieces[j][1] {
            specs.push(CoedgeSpec {
                edge: e,
                reversed: false,
                pcurve: row(1.0),
            });
        }
        side_run(false, &mut specs, &out);
        out.add_loop(bf, specs);
        // Its outward side faces away from a ball inside the body, towards one outside it.
        let s = &out.face(bf).surface;
        let u = 0.5 * (lower(j, 0.5) + upper(j, 0.5));
        let p = s.eval(u, 0.5);
        let sec = roll.section(u, None)?;
        let away = (p - sec.centre) * if convex { 1.0 } else { -1.0 };
        if (-s.normal(u, 0.5)).dot(away) <= 0.0 {
            return Err(Error::Numerical(
                "a rolling-ball fillet's face came out facing into the body".into(),
            ));
        }
        blends.push(bf);
    }

    for sh in body.shell_ids() {
        let mut faces: Vec<FaceId> = body
            .shell(sh)
            .faces
            .iter()
            .map(|f| fmap[f.index()].expect("every face is copied"))
            .collect();
        if body.shell(sh).faces.contains(&picked[0].faces[0]) {
            faces.extend(&blends);
        }
        out.add_shell(faces);
    }
    for so in body.solid_ids() {
        out.add_solid_of(body.solid(so).shells.clone(), body.solid(so).outer);
    }
    out.pair_radial();
    let problems = check::check(&out);
    if !problems.is_empty() {
        return unsupported(&format!("whose result does not check: {}", problems[0]));
    }
    Ok(out)
}

/// An `f64` ordered for use as a key.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// The parameter of curve `c` nearest `p`: a line's or circle's in closed form (a circle's within
/// a turn of `range`), any other's by sampling `range` and refining with Newton steps.
fn nearest_param(c: &Curve, p: Point3, range: (f64, f64)) -> Result<f64> {
    match c {
        Curve::Line(l) => Ok((p - l.origin).dot(l.dir)),
        Curve::Circle(ci) => {
            let q = ci.frame.to_local(p);
            let a = q.y.atan2(q.x);
            let mid = 0.5 * (range.0 + range.1);
            Ok(a + ((mid - a) / std::f64::consts::TAU).round() * std::f64::consts::TAU)
        }
        _ => {
            let n = 64;
            let mut t = (0..=n)
                .map(|i| range.0 + (range.1 - range.0) * i as f64 / n as f64)
                .min_by(|&a, &b| c.eval(a).dist(p).total_cmp(&c.eval(b).dist(p)))
                .expect("samples");
            for _ in 0..50 {
                let (q, d) = c.d1(t);
                let h = 1e-6 * (range.1 - range.0).max(1e-9);
                let d2 = (c.d1(t + h).1 - c.d1(t - h).1) / (2.0 * h);
                let f = (q - p).dot(d);
                let df = d.dot(d) + (q - p).dot(d2);
                if df.abs() <= 1e-300 {
                    break;
                }
                let step = f / df;
                t = (t - step).clamp(range.0, range.1);
                if step.abs() <= 1e-15 * (1.0 + t.abs()) {
                    break;
                }
            }
            Ok(t)
        }
    }
}
