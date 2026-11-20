//! Rounds and chamfers beyond one size on edges picked together: a round between faces that
//! share no edge, a full round, sections that are conics or curvature continuous, a radius that
//! grows along its edge, a size for each edge, and corners set back. Each builds an exact tool
//! and cuts it from the solid, and names the faces it makes `<feature>.round`, `<feature>.bevel`
//! for a chamfer's and `<feature>.corner` for a corner patch's.

use std::f64::consts::{PI, TAU};

use crate::{
    blend, boolean,
    boolean::half_space,
    error::{Error, Result},
    geom::{Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    measure,
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
    tol,
    topo::{Brep, EdgeId, FaceId, FaceName},
};

mod corner;
mod uneven;

/// `out` with the faces an operation here made named for its caller: `<feature>.bevel` for a
/// chamfer's, `<feature>.corner` for a corner patch's and `<feature>.round` for every other. Until
/// then they are named under features that start with a control character, which no caller's do.
fn named_made(mut out: Brep, feature: &str) -> Brep {
    out.rename_faces(|n| match n.feature.strip_prefix('\u{1}') {
        Some(made) => {
            let face = match (made, n.face.as_str()) {
                ("cha", _) => "bevel",
                ("corner" | "set", _) | (_, "corner") => "corner",
                _ => "round",
            };
            vec![FaceName::new(feature, face)]
        }
        None => vec![n.clone()],
    });
    out
}

pub use corner::fillet_setback_g2;

fn named(segment: Segment, name: &str) -> NamedSegment {
    NamedSegment::new(segment, name)
}

/// A plane face's point and outward unit normal.
pub(crate) fn plane_of(b: &Brep, f: FaceId) -> Result<(Point3, Vec3)> {
    match &b.face(f).surface {
        Surface::Plane(pl) => {
            let n = if b.face(f).reversed {
                -pl.frame.z
            } else {
                pl.frame.z
            };
            Ok((pl.frame.origin, n))
        }
        s => Err(Error::Unsupported(format!(
            "a blend between faces on a {} surface",
            s.kind()
        ))),
    }
}

/// The section square to the direction planes all run along: a frame drawing in it, placed so an
/// extrusion `len` long from it runs through all of `s`.
struct Section {
    frame: Frame,
    len: f64,
}

impl Section {
    fn new(s: &Brep, normals: &[Vec3]) -> Result<Section> {
        let d = normals[0]
            .cross(normals[1])
            .unit()
            .ok_or_else(|| Error::InvalidInput("the faces are parallel".into()))?;
        if normals.iter().any(|n| n.dot(d).abs() > 1e-9) {
            return Err(Error::Unsupported(
                "faces that do not all run along one direction".into(),
            ));
        }
        let b = measure::bbox(s);
        let reach = b.diagonal() + 1.0;
        let frame = Frame::new(b.center() - d * reach, d, normals[0])?;
        Ok(Section {
            frame,
            len: 2.0 * reach,
        })
    }

    fn point(&self, p: Point3) -> Vec2 {
        let l = self.frame.to_local(p);
        Vec2::new(l.x, l.y)
    }

    fn dir(&self, v: Vec3) -> Vec2 {
        let l = self.frame.dir_to_local(v);
        Vec2::new(l.x, l.y)
    }

    /// `s` less the region the loop bounds, run through all of it.
    fn cut(&self, s: &Brep, lp: ProfileLoop) -> Result<Brep> {
        let tool = extrude(
            &Profile { loops: vec![lp] },
            &self.frame,
            self.len,
            "\u{1}blend",
        )?;
        boolean::difference(s, &tool)
    }
}

/// The loop through `pts` by lines, closed from the last back to the first by the arc round
/// `centre` of `radius` through `via`, counter-clockwise.
fn loop_with_arc(pts: &[Vec2], centre: Vec2, radius: f64, via: Vec2) -> ProfileLoop {
    let ang = |p: Vec2| (p.y - centre.y).atan2(p.x - centre.x);
    let (from, to) = (pts[pts.len() - 1], pts[0]);
    let ccw = (ang(to) - ang(from)).rem_euclid(TAU);
    let sweep = if (ang(via) - ang(from)).rem_euclid(TAU) < ccw {
        ccw
    } else {
        ccw - TAU
    };
    let mut segments: Vec<Segment> = pts
        .windows(2)
        .map(|w| Segment::Line { a: w[0], b: w[1] })
        .collect();
    segments.push(Segment::Arc {
        center: centre,
        radius,
        start: ang(from),
        sweep,
    });
    // Counter-clockwise by the shoelace over the corners and the arc's middle.
    let mid = ang(from) + sweep / 2.0;
    let mut ring: Vec<Vec2> = pts.to_vec();
    ring.push(centre + Vec2::new(mid.cos(), mid.sin()) * radius);
    let twice: f64 = (0..ring.len())
        .map(|i| ring[i].cross(ring[(i + 1) % ring.len()]))
        .sum();
    if twice < 0.0 {
        for s in &mut segments {
            s.reverse();
        }
        segments.reverse();
    }
    ProfileLoop {
        segments: segments
            .into_iter()
            .enumerate()
            .map(|(i, s)| named(s, &format!("s{i}")))
            .collect(),
    }
}

/// Rounds between planes `a` and `b`, which meet nowhere on the solid (a chamfer may lie between
/// them), by the ball of `radius` against both: the material between their meeting line and the
/// ball's arc goes, whatever lay there already.
pub fn fillet_faces(s: &Brep, a: FaceId, b: FaceId, radius: f64, feature: &str) -> Result<Brep> {
    Ok(named_made(fillet_faces_raw(s, a, b, radius)?, feature))
}

fn fillet_faces_raw(s: &Brep, a: FaceId, b: FaceId, radius: f64) -> Result<Brep> {
    let ((pa, na), (pb, nb)) = (plane_of(s, a)?, plane_of(s, b)?);
    let sec = Section::new(s, &[na, nb])?;
    let (pa, na, pb, nb) = (sec.point(pa), sec.dir(na), sec.point(pb), sec.dir(nb));
    let solve = |ra: f64, rb: f64| {
        let det = na.cross(nb);
        let (ca, cb) = (na.dot(pa) - ra, nb.dot(pb) - rb);
        Vec2::new((ca * nb.y - cb * na.y) / det, (na.x * cb - nb.x * ca) / det)
    };
    let (centre, corner) = (solve(radius, radius), solve(0.0, 0.0));
    let (ta, tb) = (centre + na * radius, centre + nb * radius);
    let pts = [
        ta,
        ta + na * radius,
        corner + (na + nb) * radius,
        tb + nb * radius,
        tb,
    ];
    let via = centre + (corner - centre) * (radius / (corner - centre).norm());
    sec.cut(s, loop_with_arc(&pts, centre, radius, via))
}

/// Replaces the plane `middle` by a round tangent to it and to the planes `side_a` and `side_b`
/// either side: the circle touching all three, the material outside it beyond its touch points
/// gone.
pub fn full_round(
    s: &Brep,
    side_a: FaceId,
    middle: FaceId,
    side_b: FaceId,
    feature: &str,
) -> Result<Brep> {
    Ok(named_made(
        full_round_raw(s, side_a, middle, side_b)?,
        feature,
    ))
}

fn full_round_raw(s: &Brep, side_a: FaceId, middle: FaceId, side_b: FaceId) -> Result<Brep> {
    let faces = [
        plane_of(s, side_a)?,
        plane_of(s, middle)?,
        plane_of(s, side_b)?,
    ];
    let sec = Section::new(s, &[faces[0].1, faces[1].1, faces[2].1])?;
    let planes: Vec<(Vec2, Vec2)> = faces
        .iter()
        .map(|&(p, n)| (sec.point(p), sec.dir(n)))
        .collect();
    // The circle inside all three, touching each: n·c + ρ = n·p for each plane.
    let m: Vec<[f64; 4]> = planes
        .iter()
        .map(|(p, n)| [n.x, n.y, 1.0, n.dot(*p)])
        .collect();
    let det3 = |c: [usize; 3]| {
        let e = |i: usize, j: usize| m[i][c[j]];
        e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
            - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
            + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0))
    };
    let den = det3([0, 1, 2]);
    if den.abs() <= 1e-12 {
        return Err(Error::InvalidInput(
            "no circle touches all three faces".into(),
        ));
    }
    let (cx, cy, rho) = (
        det3([3, 1, 2]) / den,
        det3([0, 3, 2]) / den,
        det3([0, 1, 3]) / den,
    );
    if rho <= tol::LINEAR {
        return Err(Error::InvalidInput(
            "no circle inside the faces touches all three".into(),
        ));
    }
    let centre = Vec2::new(cx, cy);
    let touch = |k: usize| centre + planes[k].1 * rho;
    let (ta, tm, tb) = (touch(0), touch(1), touch(2));
    let (na, nm, nb) = (planes[0].1, planes[1].1, planes[2].1);
    let far = 2.0 * rho + 1.0;
    let pts = [
        ta,
        ta + na * rho,
        ta + na * rho + nm * far,
        tb + nb * rho + nm * far,
        tb + nb * rho,
        tb,
    ];
    sec.cut(s, loop_with_arc(&pts, centre, rho, tm))
}

/// The corner of a straight edge between two planes as the section square to it sees it: the
/// section, the corner, and each face's direction away from the edge along it.
fn corner_of(s: &Brep, e: EdgeId) -> Result<(Section, Vec2, [Vec2; 2], [Vec2; 2])> {
    let faces: Vec<FaceId> = s
        .coedge_ids()
        .filter(|&c| s.coedge(c).edge == e)
        .map(|c| s.coedge_face(c))
        .collect();
    let [f, g] = faces[..] else {
        return Err(Error::InvalidInput(
            "a blended edge lies between two faces".into(),
        ));
    };
    let Some(Curve::Line(_)) = &s.edge(e).curve else {
        return Err(Error::Unsupported(
            "a section blend on an edge that is not straight".into(),
        ));
    };
    let ((_, na), (_, nb)) = (plane_of(s, f)?, plane_of(s, g)?);
    let sec = Section::new(s, &[na, nb])?;
    let p = s.vertex(s.edge(e).start).point;
    let corner = sec.point(p);
    let (na, nb) = (sec.dir(na), sec.dir(nb));
    // Along each face away from the edge: square to its normal, toward the other face's inside.
    let along = |n: Vec2, other: Vec2| {
        let d = Vec2::new(-n.y, n.x);
        if d.dot(other) < 0.0 { d } else { -d }
    };
    Ok((sec, corner, [along(na, nb), along(nb, na)], [na, nb]))
}

/// `s` with each of `edges`, a straight convex edge between planes, rounded by the section
/// `blend` gives from the tangent points `setback` along each face, the corner between them.
fn section_blend(
    s: &Brep,
    edges: &[EdgeId],
    setback: &dyn Fn(f64) -> f64,
    blend: &dyn Fn(Vec2, Vec2, Vec2) -> Segment,
) -> Result<Brep> {
    // Points along each edge, found again after each cut, which renumbers the solid.
    let marks: Vec<Point3> = edges
        .iter()
        .map(|&e| {
            let x = s.edge(e);
            x.curve.as_ref().map_or(s.vertex(x.start).point, |c| {
                c.eval(0.5 * (x.range.0 + x.range.1))
            })
        })
        .collect();
    let mut out = s.clone();
    for p in marks {
        out = section_cut(&out, p, None, setback, blend)?;
    }
    Ok(out)
}

/// `s` with its straight convex edge through `mark`, between planes, rounded as [`section_blend`]
/// rounds one; with `from`, only from the plane through it square to the edge, on `mark`'s side.
fn section_cut(
    s: &Brep,
    mark: Point3,
    from: Option<Point3>,
    setback: &dyn Fn(f64) -> f64,
    blend: &dyn Fn(Vec2, Vec2, Vec2) -> Segment,
) -> Result<Brep> {
    let e = s
        .edge_ids()
        .filter_map(|e| {
            let x = s.edge(e);
            let c = x.curve.as_ref()?;
            let t = c.project(mark, x.range.0.min(x.range.1), x.range.0.max(x.range.1));
            Some((c.eval(t).dist(mark), e))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .filter(|x| x.0 <= 10.0 * tol::LINEAR)
        .map(|x| x.1)
        .ok_or_else(|| Error::InvalidInput("a blended edge is gone".into()))?;
    let (mut sec, corner, along, normals) = corner_of(s, e)?;
    if let Some(q) = from {
        let z = sec.frame.z;
        let ahead = (q - sec.frame.origin).dot(z);
        let back = if (mark - q).dot(z) > 0.0 {
            0.0
        } else {
            sec.len
        };
        sec.frame.origin += z * (ahead - back);
    }
    let setback = setback(along[0].dot(along[1]).clamp(-1.0, 1.0).acos());
    let (ta, tb) = (corner + along[0] * setback, corner + along[1] * setback);
    let reach = setback;
    let pts = [
        ta,
        ta + normals[0] * reach,
        corner + (normals[0] + normals[1]) * reach,
        tb + normals[1] * reach,
        tb,
    ];
    let mut segments: Vec<Segment> = pts
        .windows(2)
        .map(|w| Segment::Line { a: w[0], b: w[1] })
        .collect();
    segments.push(blend(tb, corner, ta));
    let mut ring: Vec<Vec2> = pts.to_vec();
    ring.push(segments.last().expect("a closing section").at(0.5));
    let twice: f64 = (0..ring.len())
        .map(|i| ring[i].cross(ring[(i + 1) % ring.len()]))
        .sum();
    if twice < 0.0 {
        for seg in &mut segments {
            seg.reverse();
        }
        segments.reverse();
    }
    let lp = ProfileLoop {
        segments: segments
            .into_iter()
            .enumerate()
            .map(|(i, seg)| named(seg, &format!("s{i}")))
            .collect(),
    };
    sec.cut(s, lp)
}

/// `s` with each of `edges`, a straight convex edge between planes, rounded by a conic: the
/// rational quadratic arc from the point `setback` along one face through the edge's corner as
/// apex to the point `setback` along the other, its shoulder `rho` of the way from the chord's
/// middle to the corner (middle weight `rho/(1 − rho)`).
pub fn fillet_conic(
    s: &Brep,
    edges: &[EdgeId],
    setback: f64,
    rho: f64,
    feature: &str,
) -> Result<Brep> {
    Ok(named_made(
        fillet_conic_raw(s, edges, setback, rho)?,
        feature,
    ))
}

fn fillet_conic_raw(s: &Brep, edges: &[EdgeId], setback: f64, rho: f64) -> Result<Brep> {
    if !(rho > 0.0 && rho < 1.0) {
        return Err(Error::InvalidInput(
            "a conic's rho lies between 0 and 1".into(),
        ));
    }
    let w = rho / (1.0 - rho);
    section_blend(s, edges, &|_| setback, &|a, apex, b| Segment::Bezier {
        ctrl: vec![a, apex, b],
        weights: vec![1.0, w, 1.0],
    })
}

/// The weights `(α, β)` of a G^n section's legs at `t`, and their derivatives: `α = f·(1 − p₁)`,
/// `β = (1 − f)·p₂`, `f = t̄ⁿ⁺¹/(t̄ⁿ⁺¹ + tⁿ⁺¹)`, `p₁ = t/(1 − w + wt)`, `p₂ = t(1 − w)/(1 − wt)`.
fn gn_weights(order: u32, w: f64, t: f64) -> ([f64; 2], [f64; 2]) {
    let m = order as i32 + 1;
    let (s, u) = (1.0 - t, t);
    let (big_a, big_b) = (s.powi(m), u.powi(m));
    let (da, db) = (-(m as f64) * s.powi(m - 1), m as f64 * u.powi(m - 1));
    let sum = big_a + big_b;
    let f = big_a / sum;
    let df = (da * big_b - big_a * db) / (sum * sum);
    let (d1, d2) = (1.0 - w + w * t, 1.0 - w * t);
    let (p1, p2) = (t / d1, t * (1.0 - w) / d2);
    let (dp1, dp2) = ((1.0 - w) / (d1 * d1), (1.0 - w) / (d2 * d2));
    (
        [f * (1.0 - p1), (1.0 - f) * p2],
        [df * (1.0 - p1) - f * dp1, -df * p2 + (1.0 - f) * dp2],
    )
}

/// The radius of curvature at the middle of a G^n section of thumbweight `w` whose legs are `d`
/// long and meet at `alpha`: `d·(α'² + β'² + 2α'β' cos alpha)^(3/2) / (|α'β'' − α''β'| sin alpha)`
/// at `t = ½`, the second derivatives by a five-point difference of the exact first ones.
///
/// "Thumbweight" is this shape parameter's name in E. Hartmann, "Parametric Gⁿ blending of curves
/// and surfaces", The Visual Computer 17, 2001, pp. 1–13.
fn gn_center_radius(order: u32, w: f64, alpha: f64, d: f64) -> f64 {
    let t = 0.5;
    let (_, d1) = gn_weights(order, w, t);
    let h = 1e-3;
    let second = |k: usize| {
        let at = |s: f64| gn_weights(order, w, t + s * h).1[k];
        (at(-2.0) - 8.0 * at(-1.0) + 8.0 * at(1.0) - at(2.0)) / (12.0 * h)
    };
    let (a1, b1, a2, b2) = (d1[0], d1[1], second(0), second(1));
    let speed = a1 * a1 + b1 * b1 + 2.0 * a1 * b1 * alpha.cos();
    d * speed.powf(1.5) / ((a1 * b2 - a2 * b1).abs() * alpha.sin())
}

/// The share `∫ αβ' − α'β dt` of the triangle on its two legs that a G^n section of thumbweight
/// `w` takes off, by five-point Gauss on 256 panels.
fn gn_area_share(order: u32, w: f64) -> f64 {
    const X: [f64; 5] = [
        0.0,
        -0.538_469_310_105_683_1,
        0.538_469_310_105_683_1,
        -0.906_179_845_938_664,
        0.906_179_845_938_664,
    ];
    const W: [f64; 5] = [
        0.568_888_888_888_888_9,
        0.478_628_670_499_366_5,
        0.478_628_670_499_366_5,
        0.236_926_885_056_189_1,
        0.236_926_885_056_189_1,
    ];
    let panels = 256;
    let h = 1.0 / panels as f64;
    let mut sum = 0.0;
    for k in 0..panels {
        let mid = (k as f64 + 0.5) * h;
        for (x, wt) in X.iter().zip(W) {
            let (v, d) = gn_weights(order, w, mid + 0.5 * h * x);
            sum += wt * 0.5 * h * (v[0] * d[1] - d[0] * v[1]);
        }
    }
    sum
}

/// The thumbweight at which a G^n section in a wedge of angle `alpha` takes off what the circle
/// tangent at the same contacts does, `(cot(α/2) − (π − α)/2) / (cot(α/2)·cos²(α/2))` of the
/// triangle: the share falls as `w` rises.
fn gn_equal_area_w(order: u32, alpha: f64) -> f64 {
    let c = 1.0 / (alpha / 2.0).tan();
    let target = (c - (PI - alpha) / 2.0) / (c * (alpha / 2.0).cos().powi(2));
    let (mut lo, mut hi) = (-1.0f64, 1.0 - 1e-9);
    while gn_area_share(order, lo) < target && lo > -1e12 {
        hi = lo;
        lo = 2.0 * lo - 1.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gn_area_share(order, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-15 * (1.0 - lo) {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// The thumbweight at which a G^n section with legs `d` long meeting at `alpha` bends with radius
/// `center` at its middle: that radius falls from unbounded to zero as `w` rises to 1.
fn gn_center_w(order: u32, alpha: f64, d: f64, center: f64) -> f64 {
    let (mut lo, mut hi) = (-1.0f64, 1.0 - 1e-9);
    while gn_center_radius(order, lo, alpha, d) < center && lo > -1e12 {
        hi = lo;
        lo = 2.0 * lo - 1.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gn_center_radius(order, mid, alpha, d) > center {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-15 * (1.0 - lo) {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// The order-2 section of thumbweight `w` from `a` to `b` on the legs from `apex`, exactly: with
/// `s = 1 − t`, `α = (1 − w)s⁴(1 − wt)/D` and `β = (1 − w)t⁴(1 − w + wt)/D` over
/// `D = (s³ + t³)(1 − w + wt)(1 − wt)`, a rational quintic whose weights are D's Bernstein
/// coefficients, positive for `w < 1`.
fn g2_section(a: Vec2, apex: Vec2, b: Vec2, w: f64) -> Segment {
    let den = |t: f64| {
        let s = 1.0 - t;
        (s.powi(3) + t.powi(3)) * (1.0 - w + w * t) * (1.0 - w * t)
    };
    let (wa, wb) = (a - apex, b - apex);
    let num = |t: f64| {
        let s = 1.0 - t;
        let al = (1.0 - w) * s.powi(4) * (1.0 - w * t);
        let be = (1.0 - w) * t.powi(4) * (1.0 - w + w * t);
        apex * den(t) + wa * al + wb * be
    };
    // Bernstein coefficients of the three quintics, from their values at six points.
    let ts: Vec<f64> = (0..6).map(|k| k as f64 / 5.0).collect();
    let binom = [1.0, 5.0, 10.0, 10.0, 5.0, 1.0];
    let mut m = vec![0.0; 36];
    for (r, &t) in ts.iter().enumerate() {
        for (i, c) in binom.iter().enumerate() {
            m[r * 6 + i] = c * t.powi(i as i32) * (1.0 - t).powi(5 - i as i32);
        }
    }
    let mut rhs = vec![0.0; 18];
    for (r, &t) in ts.iter().enumerate() {
        let (p, d) = (num(t), den(t));
        rhs[r * 3] = p.x;
        rhs[r * 3 + 1] = p.y;
        rhs[r * 3 + 2] = d;
    }
    let x = crate::numeric::solve_dense(m, rhs, 6, 3).expect("the Bernstein matrix is invertible");
    let (ctrl, weights) = (0..6)
        .map(|i| {
            let wgt = x[i * 3 + 2];
            (Vec2::new(x[i * 3] / wgt, x[i * 3 + 1] / wgt), wgt)
        })
        .unzip();
    Segment::Bezier { ctrl, weights }
}

/// `s` with each of `edges`, a straight convex edge between planes, rounded by the
/// curvature-continuous section of order 2 between the points a ball of radius `offset` touches,
/// its thumbweight such that it bends with radius `center` at its middle.
pub fn fillet_g2(
    s: &Brep,
    edges: &[EdgeId],
    offset: f64,
    center: f64,
    feature: &str,
) -> Result<Brep> {
    Ok(named_made(
        fillet_g2_raw(s, edges, offset, center)?,
        feature,
    ))
}

fn fillet_g2_raw(s: &Brep, edges: &[EdgeId], offset: f64, center: f64) -> Result<Brep> {
    if !(offset > tol::LINEAR && center > tol::LINEAR) {
        return Err(Error::InvalidInput(
            "a G2 fillet's sizes must be positive".into(),
        ));
    }
    section_blend(
        s,
        edges,
        &|alpha| offset / (alpha / 2.0).tan(),
        &|a, apex, b| {
            let (da, db) = (a - apex, b - apex);
            let alpha = (da.dot(db) / (da.norm() * db.norm()))
                .clamp(-1.0, 1.0)
                .acos();
            let w = gn_center_w(2, alpha, da.norm(), center);
            g2_section(a, apex, b, w)
        },
    )
}

/// `s` with the straight convex edge `e` between two planes rounded by a ball whose radius runs
/// evenly with length along it, from `r_from` at the end nearest `from` to `r_to` at the other.
/// The balls' centres run along a line, their radius growing evenly along it, so their envelope is
/// a right circular cone about that line, tangent to each plane along the line of the balls'
/// feet: the corner wedge less that cone is cut away. A ball whose feet leave a face is refused.
pub fn fillet_variable(
    s: &Brep,
    e: EdgeId,
    from: Point3,
    r_from: f64,
    r_to: f64,
    feature: &str,
) -> Result<Brep> {
    Ok(named_made(
        fillet_variable_raw(s, e, from, r_from, r_to)?,
        feature,
    ))
}

fn fillet_variable_raw(s: &Brep, e: EdgeId, from: Point3, r_from: f64, r_to: f64) -> Result<Brep> {
    if !(r_from > tol::LINEAR && r_to > tol::LINEAR) {
        return Err(Error::InvalidInput(
            "a fillet's radii must be positive".into(),
        ));
    }
    let x = s.edge(e);
    let (mut p0, mut p1) = (s.vertex(x.start).point, s.vertex(x.end).point);
    if p1.dist(from) < p0.dist(from) {
        std::mem::swap(&mut p0, &mut p1);
    }
    let (sec, _, along, normals) = corner_of(s, e)?;
    let len = p0.dist(p1);
    let dir = (p1 - p0) * (1.0 / len);
    let alpha = along[0].dot(along[1]).clamp(-1.0, 1.0).acos();
    let half = alpha / 2.0;
    // The section's 2-D directions back in space.
    let world = |v: Vec2| sec.frame.dir_to_world(Vec3::new(v.x, v.y, 0.0));
    let (ua, ub) = (world(along[0]), world(along[1]));
    let bis = (ua + ub)
        .unit()
        .ok_or_else(|| Error::InvalidInput("a flat edge".into()))?;
    // The faces must hold each ball's feet, `r / tan(α/2)` out along them, all along the edge.
    for (p, r) in [(p0, r_from), (p1, r_to)] {
        for (u, k) in [(ua, 0usize), (ub, 1usize)] {
            let at = p + dir * (if p == p0 { 1e-6 } else { -1e-6 } * len) + u * (r / half.tan());
            let holds = crate::measure::face_at(s, at).is_some_and(|(f, d)| {
                d <= 10.0 * tol::LINEAR && {
                    let n = s.face(f).surface.normal(0.0, 0.0);
                    let n = if s.face(f).reversed { -n } else { n };
                    (sec.dir(n) - normals[k]).norm() <= 1e-6
                }
            });
            if !holds {
                return Err(Error::InvalidInput(
                    "a variable fillet whose ball does not fit on its faces".into(),
                ));
            }
        }
    }
    // Centres `r / sin(α/2)` along the bisector from the edge; along that line the radius grows
    // by σ per unit, and the cone's radius at axial L is cos θ·(r₀ + σ(L + σ r₀)/(1 − σ²)).
    let c0 = p0 + bis * (r_from / half.sin());
    let c1 = p1 + bis * (r_to / half.sin());
    let span = c0.dist(c1);
    let axis = (c1 - c0) * (1.0 / span);
    let sigma = (r_to - r_from) / span;
    if sigma.abs() >= 1.0 {
        return Err(Error::InvalidInput(
            "a fillet's radius may not grow faster than its edge".into(),
        ));
    }
    let cos = (1.0 - sigma * sigma).sqrt();
    let rho = |l: f64| cos * (r_from + sigma * (l + sigma * r_from) / (1.0 - sigma * sigma));
    // Past each end by twice the larger ball, short of the apex where the radius reaches nought.
    let margin = 2.0 * r_from.max(r_to);
    let (mut la, mut lb) = (-margin, span + margin);
    if sigma != 0.0 {
        let apex = -r_from * (1.0 - sigma * sigma) / sigma - sigma * r_from;
        if sigma > 0.0 {
            la = la.max(0.5 * apex);
        } else {
            lb = lb.min(0.5 * (apex + span));
        }
    }
    let (ra, rb) = (rho(la), rho(lb));
    if ra <= 0.0 || rb <= 0.0 {
        return Err(Error::Unsupported(
            "a variable fillet's cone reaches its apex".into(),
        ));
    }
    let meridian = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                named(
                    Segment::Line {
                        a: Vec2::new(0.0, la),
                        b: Vec2::new(ra, la),
                    },
                    "bottom",
                ),
                named(
                    Segment::Line {
                        a: Vec2::new(ra, la),
                        b: Vec2::new(rb, lb),
                    },
                    "side",
                ),
                named(
                    Segment::Line {
                        a: Vec2::new(rb, lb),
                        b: Vec2::new(0.0, lb),
                    },
                    "top",
                ),
                named(
                    Segment::Line {
                        a: Vec2::new(0.0, lb),
                        b: Vec2::new(0.0, la),
                    },
                    "axis",
                ),
            ],
        }],
    };
    let f = Frame::from_axis(c0, axis)?;
    let cone = revolve(
        &meridian,
        &Frame::new(c0, -f.y, f.x)?,
        c0,
        f.z,
        TAU,
        "\u{1}ball",
    )?;
    // The two lines of feet are mirror images meeting where the radius would reach nought, so they
    // lie in one plane: the corner's side of it, less the cone, is what goes.
    let (d0, d1) = (r_from / half.tan(), r_to / half.tan());
    let (qa0, qa1, qb0) = (p0 + ua * d0, p1 + ua * d1, p0 + ub * d0);
    let mut n = (qa1 - qa0)
        .cross(qb0 - qa0)
        .unit()
        .ok_or_else(|| Error::InvalidInput("a fillet with no width".into()))?;
    if n.dot(p0 - qa0) < 0.0 {
        n = -n;
    }
    let corner_side = half_space(s, qa0, n)?;
    let tool = boolean::difference(&corner_side, &cone)?;
    boolean::difference(s, &tool)
}

/// `s` with each of `edges` rounded at its own radius, each edge with its tangent chain. Where
/// three rounds of different radii meet at a corner of three planes the corner is closed without
/// a crease (`uneven`); where rounds of different radii meet any other way they are mitred, the
/// solid less every round's region.
pub fn fillet_radii(s: &Brep, edges: &[(EdgeId, f64)], feature: &str) -> Result<Brep> {
    Ok(named_made(fillet_radii_raw(s, edges)?, feature))
}

fn fillet_radii_raw(s: &Brep, edges: &[(EdgeId, f64)]) -> Result<Brep> {
    // Where rounds of different radii meet, each runs its radius along the other's edge.
    let length = |e: EdgeId| {
        let x = s.edge(e);
        x.curve
            .as_ref()
            .map_or(0.0, |c| crate::topo::radial::length(c, x.range))
    };
    let ends = |e: EdgeId| [s.edge(e).start, s.edge(e).end];
    for &(a, ra) in edges {
        for &(b, rb) in edges {
            let meet = ends(a).iter().any(|v| ends(b).contains(v));
            if a != b && ra != rb && meet && ra > length(b) + tol::LINEAR {
                return Err(Error::InvalidInput(format!(
                    "a round of {ra} runs past the end of an edge {} long at a corner",
                    length(b)
                )));
            }
        }
    }
    if let Ok(Some(smooth)) = uneven::rounded(s, edges) {
        return Ok(smooth);
    }
    let mut radii: Vec<f64> = edges.iter().map(|e| e.1).collect();
    radii.sort_by(|a, b| b.total_cmp(a));
    radii.dedup();
    let mut out: Option<Brep> = None;
    for r in radii {
        let these: Vec<EdgeId> = edges.iter().filter(|e| e.1 == r).map(|e| e.0).collect();
        let round = blend::fillet_edges(s, &these, r, "\u{1}fil")?;
        out = Some(match out {
            None => round,
            Some(o) => boolean::intersection(&o, &round)?,
        });
    }
    out.ok_or_else(|| Error::InvalidInput("no edges to round".into()))
}

/// `s` with each of `edges` chamfered at its own distance, each edge with its tangent chain: each
/// distance's chamfer cut from `s` itself and the cuts taken together, so a larger chamfer never
/// takes away an edge a smaller one names.
pub fn chamfer_distances(s: &Brep, edges: &[(EdgeId, f64)], feature: &str) -> Result<Brep> {
    Ok(named_made(chamfer_distances_raw(s, edges)?, feature))
}

fn chamfer_distances_raw(s: &Brep, edges: &[(EdgeId, f64)]) -> Result<Brep> {
    let mut sizes: Vec<f64> = edges.iter().map(|e| e.1).collect();
    sizes.sort_by(|a, b| b.total_cmp(a));
    sizes.dedup();
    let mut out: Option<Brep> = None;
    for d in sizes {
        let these: Vec<EdgeId> = edges.iter().filter(|e| e.1 == d).map(|e| e.0).collect();
        let cut = blend::chamfer_edges(s, &these, d, "\u{1}cha")?;
        out = Some(match out {
            None => cut,
            Some(o) => boolean::intersection(&o, &cut)?,
        });
    }
    out.ok_or_else(|| Error::InvalidInput("no edges to chamfer".into()))
}

/// `s` with the three edges at a corner of three square planes rounded at `radius`, each round
/// held whole until `setback` from the corner and the corner between them a patch tangent to
/// them all.
///
/// In the corner's own axes, the distances `u, v, w` in from its faces, a round's ball has its
/// centre on the lines `r` in from two faces. Set back, the centres near the corner lie on the
/// sphere about `(s, s, s)` through each line's point `s` along it, `√2·(s − r)` in radius, which
/// meets each line there tangentially; where the sphere passes beneath a face the centres run
/// along its circle `r` above that face. The patch the ball sweeps is that sphere grown by `r`, the
/// solid `P` it cuts from the corner's wedge, with `P`'s rims rounded at `r`: tori, each tangent
/// to its face and to the sphere. Within the cube of side `s` at the corner that rounded `P` is the
/// solid; outside it the three rounds are.
pub fn fillet_setback(
    s: &Brep,
    edges: &[EdgeId],
    radius: f64,
    setback: f64,
    feature: &str,
) -> Result<Brep> {
    Ok(named_made(
        fillet_setback_raw(s, edges, radius, setback)?,
        feature,
    ))
}

fn fillet_setback_raw(s: &Brep, edges: &[EdgeId], radius: f64, setback: f64) -> Result<Brep> {
    let (r, d) = (radius, setback);
    let (at, a) = square_corner(s, edges, r, d)?;
    let cube = |size: f64| corner_cube(at, &a, size);
    let centre = at + (a[0] + a[1] + a[2]) * d;
    let big = std::f64::consts::SQRT_2 * (d - r) + r;
    let ball = {
        // Poles across the corner's diagonal and the seam on its far side keep both off the patch
        // and the rims: each rim lies `d` beneath a face, further than √½·big from the centre.
        let axis = Frame::new(centre, a[0] - a[1], a[0] + a[1] + a[2])?;
        let skin = Segment::Arc {
            center: Vec2::ZERO,
            radius: big,
            start: -PI / 2.0,
            sweep: PI,
        };
        let line = Segment::Line {
            a: Vec2::new(0.0, big),
            b: Vec2::new(0.0, -big),
        };
        let half = Profile {
            loops: vec![ProfileLoop {
                segments: vec![named(skin, "ball"), named(line, "axis")],
            }],
        };
        revolve(
            &half,
            &Frame::new(centre, -axis.y, axis.x)?,
            centre,
            axis.z,
            TAU,
            "set",
        )?
    };
    let wedge = boolean::intersection(&ball, &cube(d + big + 1.0)?)?;
    let faces_of = |e: EdgeId| {
        let mut f: Vec<FaceId> = wedge
            .coedge_ids()
            .filter(|&c| wedge.coedge(c).edge == e)
            .map(|c| wedge.coedge_face(c))
            .collect();
        f.dedup();
        f.len()
    };
    let rims: Vec<EdgeId> = wedge
        .edge_ids()
        .filter(|&e| !wedge.edge(e).is_degenerate() && faces_of(e) == 2)
        .collect();
    let patch = blend::fillet_edges(&wedge, &rims, r, "set")?;
    let rounds = blend::fillet_edges(s, edges, r, "\u{1}fil")?;
    let k = cube(d)?;
    boolean::union(
        &boolean::difference(&rounds, &k)?,
        &boolean::intersection(&patch, &k)?,
    )
}

/// The corner the three `edges` of `s` meet at, square between flat faces and each edge longer
/// than `setback`, itself more than `radius`: the corner, and each face's direction into the
/// solid.
fn square_corner(
    s: &Brep,
    edges: &[EdgeId],
    radius: f64,
    setback: f64,
) -> Result<(Point3, [Vec3; 3])> {
    if edges.len() != 3 {
        return Err(Error::Unsupported(
            "a set-back corner of other than three edges".into(),
        ));
    }
    if !(radius > 0.0 && setback > radius) {
        return Err(Error::InvalidInput(
            "a corner set back no further than its radius".into(),
        ));
    }
    let ends = |e: EdgeId| [s.edge(e).start, s.edge(e).end];
    let corner = ends(edges[0])
        .into_iter()
        .find(|v| edges[1..].iter().all(|&e| ends(e).contains(v)))
        .ok_or_else(|| Error::InvalidInput("the edges do not meet at one corner".into()))?;
    let mut normals: Vec<Vec3> = Vec::new();
    for co in s.coedge_ids() {
        let c = s.coedge(co);
        if !edges.contains(&c.edge) {
            continue;
        }
        let face = s.face(s.coedge_face(co));
        let Surface::Plane(p) = &face.surface else {
            return Err(Error::Unsupported(
                "a set-back corner of other than flat faces".into(),
            ));
        };
        let n = if face.reversed { -p.frame.z } else { p.frame.z };
        if !normals.iter().any(|m| m.dot(n) > 1.0 - 1e-9) {
            normals.push(n);
        }
    }
    let square =
        normals.len() == 3 && (0..3).all(|i| normals[i].dot(normals[(i + 1) % 3]).abs() < 1e-9);
    if !square {
        return Err(Error::Unsupported(
            "a set-back corner whose faces are not square".into(),
        ));
    }
    for &e in edges {
        let x = s.edge(e);
        if s.vertex(x.start).point.dist(s.vertex(x.end).point) <= setback + tol::LINEAR {
            return Err(Error::InvalidInput(
                "an edge shorter than the corner's setback".into(),
            ));
        }
    }
    Ok((
        s.vertex(corner).point,
        [-normals[0], -normals[1], -normals[2]],
    ))
}

/// A box from the corner `at` along its faces' directions `a` into the solid, `size` on a side.
fn corner_cube(at: Point3, a: &[Vec3; 3], size: f64) -> Result<Brep> {
    let frame = Frame::new(at, a[2], a[0])?;
    let side = if frame.y.dot(a[1]) > 0.0 { 1.0 } else { -1.0 };
    extrude(
        &Profile::rect(Vec2::new(size / 2.0, side * size / 2.0), size, size),
        &frame,
        size,
        "\u{1}corner",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_back_corner_whose_rounds_cross_shallowly_keeps_each_corner_once() {
        // A box 14.88 × 14.51 × 12.13, its three edges at the far corner rounded 1.419 and set back
        // 3.875. Two of the patch's rims' rounds cross at a shallow angle at (11.003, 14.097,
        // 11.716), within the tolerance of each other along a band 3e-7 long: their meeting found
        // from either end of it is one vertex, and the result is valid. On aarch64 Linux and
        // wasm32-wasip1 the two ends were kept apart.
        let size = Vec3::new(14.878279824052331, 14.512511110418185, 12.131310523342954);
        let b = crate::primitive::block(&Frame::WORLD, size, "box").unwrap();
        let at = |p: Point3| crate::measure::edge_at(&b, p).unwrap().0;
        let edges = [
            at(Point3::new(6.497951769482199, size.y, size.z)),
            at(Point3::new(size.x, 8.360032841382534, size.z)),
            at(Point3::new(size.x, size.y, 7.959480197539578)),
        ];
        let out =
            fillet_setback(&b, &edges, 1.419369652573463, 3.875221429567226, "blend").unwrap();
        assert!(crate::check::check(&out).is_empty());
        let corner = Point3::new(11.003058394485105, 14.096787364189812, 11.71558677711458);
        let near = out
            .vertex_ids()
            .filter(|&v| out.vertex(v).point.dist(corner) < 1e-5)
            .count();
        assert_eq!(near, 1);
    }
    use crate::{rays::Rays, topo::FaceName};

    /// A prism on the polygon `pts`, `h` tall, its sides named `e0`, `e1`, … from the first edge.
    fn prism(pts: &[[f64; 2]], h: f64) -> Brep {
        let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
        let names: Vec<String> = (0..pts.len()).map(|i| format!("e{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        extrude(&Profile::polygon(&pts, &names), &Frame::WORLD, h, "p").unwrap()
    }

    fn face(b: &Brep, name: &str) -> FaceId {
        let n = FaceName::new("p", name);
        b.face_ids()
            .find(|&f| b.face(f).names.contains(&n))
            .unwrap()
    }

    fn close(got: f64, want: f64) {
        assert!(
            (got - want).abs() < 1e-7 * want.max(1.0),
            "{got} against {want}"
        );
    }

    #[test]
    fn a_rib_s_full_round_is_a_half_cylinder_on_its_sides() {
        // A rib 2 wide, 3 tall, 5 long: its top becomes the half cylinder of radius 1, so it
        // loses (2·1 − π/2)·5 of its 30.
        let b = prism(&[[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]], 5.0);
        let out = full_round_raw(&b, face(&b, "e1"), face(&b, "e2"), face(&b, "e3")).unwrap();
        close(measure::volume(&out), 30.0 - (2.0 - PI / 2.0) * 5.0);
    }

    #[test]
    fn a_round_between_the_faces_beside_a_chamfer_replaces_it() {
        // A 4 × 4 square prism 1 tall, its corner at (4, 4) chamfered 1 back; the ball of
        // radius 2 between the two sides takes the corner to the round: 16 − (4 − π).
        let b = prism(
            &[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [3.0, 4.0], [0.0, 4.0]],
            1.0,
        );
        let out = fillet_faces_raw(&b, face(&b, "e1"), face(&b, "e3"), 2.0).unwrap();
        close(measure::volume(&out), 16.0 - (4.0 - PI));
    }

    #[test]
    fn rounds_of_two_radii_at_a_corner_meet_mitred() {
        // A 10 cube, the top edge along x rounded 2 and the one along y rounded 1. Each takes
        // (1 − π/4)·r²·10; where they meet at the corner the strips share ∫ L₂(w)·L₁(w) dw over
        // the depth w below the top, L_r(w) = r − √(r² − (r − w)²) being a strip's width there.
        let b = prism(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], 10.0);
        let at = |x, y, z| measure::edge_at(&b, Point3::new(x, y, z)).unwrap().0;
        let edges = [(at(5.0, 10.0, 10.0), 2.0), (at(10.0, 5.0, 10.0), 1.0)];
        let width = |r: f64, w: f64| r - (r * r - (r - w) * (r - w)).sqrt();
        let n = 100_000;
        let shared: f64 = (0..n)
            .map(|i| (i as f64 + 0.5) / n as f64)
            .map(|w| width(2.0, w) * width(1.0, w) / n as f64)
            .sum();
        let k = 1.0 - PI / 4.0;
        let want = 1000.0 - k * 4.0 * 10.0 - k * 10.0 + shared;
        close(
            measure::volume(&fillet_radii_raw(&b, &edges).unwrap()),
            want,
        );
    }

    #[test]
    fn a_round_wider_than_a_chamfer_swallows_it() {
        // A 10 square prism 4 tall, its corner at (10, 10) chamfered a hundredth back along each
        // side, both edges of the chamfer filleted at 2. No ball of 2 fits on a face 0.014 wide,
        // so it rolls on the sides beyond and the chamfer goes: the uncut corner rounded at 2,
        // 400 less (1 − π/4)·2²·4.
        let c = 0.01;
        let b = prism(
            &[
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0 - c],
                [10.0 - c, 10.0],
                [0.0, 10.0],
            ],
            4.0,
        );
        let at = |x, y, z| measure::edge_at(&b, Point3::new(x, y, z)).unwrap().0;
        let edges = [at(10.0, 10.0 - c, 2.0), at(10.0 - c, 10.0, 2.0)];
        let out = crate::blend::fillet_edges(&b, &edges, 2.0, "f").unwrap();
        assert!(
            crate::check::check(&out).is_empty(),
            "{:?}",
            crate::check::check(&out)
        );
        close(measure::volume(&out), 400.0 - (1.0 - PI / 4.0) * 16.0);
    }

    #[test]
    fn a_set_back_corner_keeps_its_rounds_whole_beyond_the_setback() {
        // A 10 cube's corner at (10, 10, 10), its edges rounded 1 and set back 3. Beyond the 3
        // cube at the corner each round takes (1 − π/4) from every unit of its 7 left; within
        // it the patch takes less than the whole cube and more than the rounds alone would.
        let b = prism(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], 10.0);
        let at = |x, y, z| measure::edge_at(&b, Point3::new(x, y, z)).unwrap().0;
        let edges = [
            at(5.0, 10.0, 10.0),
            at(10.0, 5.0, 10.0),
            at(10.0, 10.0, 5.0),
        ];
        let out = fillet_setback_raw(&b, &edges, 1.0, 3.0).unwrap();
        assert!(
            crate::check::check(&out).is_empty(),
            "{:?}",
            crate::check::check(&out)
        );
        let frame = Frame::new(Point3::new(7.0, 7.0, 7.0), Vec3::Z, Vec3::X).unwrap();
        let k = extrude(
            &Profile::rect(Vec2::new(1.5, 1.5), 3.0, 3.0),
            &frame,
            3.0,
            "k",
        )
        .unwrap();
        let k_ = 1.0 - PI / 4.0;
        close(
            measure::volume(&boolean::difference(&out, &k).unwrap()),
            1000.0 - 27.0 - 21.0 * k_,
        );
        let inside = measure::volume(&boolean::intersection(&out, &k).unwrap());
        let rounds = 27.0 - 9.0 * k_ - (1.0 - PI / 6.0);
        assert!(inside > 0.0 && inside < rounds, "{inside} against {rounds}");
        // The point 1 in from two faces on the third, where a ball's corner would touch, is cut.
        let r = Rays::new(&out).unwrap();
        let hits = r
            .cast(
                Point3::new(9.0, 9.0, 10.0 - 1e-4),
                Vec3::new(0.3, 0.5, 0.8),
                (0.0, 100.0),
            )
            .unwrap();
        assert!(hits.len().is_multiple_of(2), "{} crossings", hits.len());
    }

    #[test]
    fn chamfers_of_two_distances_at_a_corner_both_cut_the_block() {
        // A 10 cube, its top edge along x chamfered 4 and the one along y chamfered 1. In from the
        // faces by u, v, w, they take v + w < 4 and u + w < 1, a²/2 and b²/2 of each unit of their
        // 10 long edges, and share ∫₀¹ (4 − w)(1 − w) dw = ab² − (a + b)b²/2 + b³/3 at the corner.
        let b = prism(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], 10.0);
        let at = |x, y, z| measure::edge_at(&b, Point3::new(x, y, z)).unwrap().0;
        let edges = [(at(5.0, 10.0, 10.0), 4.0), (at(10.0, 5.0, 10.0), 1.0)];
        let out = chamfer_distances_raw(&b, &edges).unwrap();
        assert!(
            crate::check::check(&out).is_empty(),
            "{:?}",
            crate::check::check(&out)
        );
        let (a, c) = (4.0f64, 1.0f64);
        let shared = a * c * c - (a + c) * c * c / 2.0 + c.powi(3) / 3.0;
        close(
            measure::volume(&out),
            1000.0 - 5.0 * a * a - 5.0 * c * c + shared,
        );
    }

    #[test]
    fn a_round_wider_than_an_edge_it_meets_is_refused() {
        let b = prism(&[[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]], 10.0);
        let at = |x, y, z| measure::edge_at(&b, Point3::new(x, y, z)).unwrap().0;
        let edges = [(at(5.0, 1.0, 10.0), 2.0), (at(10.0, 0.5, 10.0), 0.2)];
        assert!(fillet_radii_raw(&b, &edges).is_err());
    }

    #[test]
    fn a_parabolic_conic_takes_a_third_of_its_corner_triangle() {
        // A 4 × 4 square prism 2 tall, its edge at (4, 4) blended by the conic of rho 0.5, a
        // parabola, set back 1 along each side: between a parabola and its two tangents lies a
        // third of their triangle, 1/6, so the prism loses 1/3.
        let b = prism(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], 2.0);
        let e = b
            .edge_ids()
            .find(|&e| {
                let x = b.edge(e);
                let (p, q) = (b.vertex(x.start).point, b.vertex(x.end).point);
                (p.x - 4.0).abs() < 1e-12
                    && (p.y - 4.0).abs() < 1e-12
                    && (q.x - 4.0).abs() < 1e-12
                    && (q.y - 4.0).abs() < 1e-12
            })
            .unwrap();
        let out = fillet_conic_raw(&b, &[e], 1.0, 0.5).unwrap();
        close(measure::volume(&out), 32.0 - 1.0 / 3.0);
    }

    #[test]
    fn curvature_continuous_fillets_on_edges_that_meet_run_into_each_other() {
        // A 10 cube's edges at the corner (10, 10, 10), blended with contacts 1 from each edge
        // at the equal-area thumbweight: two of them, then all three. Each takes (1 − π/4) from
        // every unit of its 10. With the section (u, v) = (β, α)(t) from one face to the other,
        // what is taken twice where two cross is ∬ h(max(x, y)) = 2∫αβ·β′ dt, h the section's
        // height over a face; what all three take is 2(∫αβ·β′ over t < ½ + ∫α²·β′ over t > ½).
        let b = prism(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], 10.0);
        let at = |x, y, z| measure::edge_at(&b, Point3::new(x, y, z)).unwrap().0;
        let edges = [
            at(5.0, 10.0, 10.0),
            at(10.0, 5.0, 10.0),
            at(10.0, 10.0, 5.0),
        ];
        let w = gn_equal_area_w(2, PI / 2.0);
        let center = gn_center_radius(2, w, PI / 2.0, 1.0);
        // Simpson's rule on 2000 panels of [from, to].
        let integral = |f: &dyn Fn(f64) -> f64, from: f64, to: f64| {
            let n = 2000;
            let h = (to - from) / n as f64;
            (0..=n)
                .map(|i| {
                    let k = if i == 0 || i == n {
                        1.0
                    } else if i % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    k * f(from + h * i as f64)
                })
                .sum::<f64>()
                * h
                / 3.0
        };
        let twice = |t: f64| {
            let (v, d) = gn_weights(2, w, t);
            v[0] * v[1] * d[1]
        };
        let thrice = |t: f64| {
            let (v, d) = gn_weights(2, w, t);
            v[0] * v[0] * d[1]
        };
        let pair = 2.0 * integral(&twice, 0.0, 1.0);
        let all = 2.0 * (integral(&twice, 0.0, 0.5) + integral(&thrice, 0.5, 1.0));
        let round = 10.0 * (1.0 - PI / 4.0);
        for (n, taken) in [(2, 2.0 * round - pair), (3, 3.0 * round - 3.0 * pair + all)] {
            let out = fillet_g2_raw(&b, &edges[..n], 1.0, center).unwrap();
            assert!(
                crate::check::check(&out).is_empty(),
                "{:?}",
                crate::check::check(&out)
            );
            close(measure::volume(&out), 1000.0 - taken);
        }
    }

    #[test]
    fn the_g2_section_as_a_quintic_is_the_section_it_defines() {
        let (a, apex, b) = (
            Vec2::new(3.0, 0.5),
            Vec2::new(0.2, 0.1),
            Vec2::new(-0.4, 2.7),
        );
        for w in [-1.7, -0.3, 0.0, 0.4, 0.8] {
            let seg = g2_section(a, apex, b, w);
            for k in 0..=10 {
                let t = k as f64 / 10.0;
                let (v, _) = gn_weights(2, w, t);
                let want = apex + (a - apex) * v[0] + (b - apex) * v[1];
                assert!(seg.at(t).dist(want) < 1e-12, "w {w} t {t}");
            }
        }
    }

    #[test]
    fn a_ball_growing_along_an_edge_rounds_it_between_its_ends_radii() {
        // A 10 × 8 × 6 block's upright edge at (10, 8) rounded by a ball growing from 1 at the
        // bottom to 2 at the top: the volume lies between the edge rounded at 2 and at 1 all
        // along, the ball's foot halfway up is on the boundary with the face's normal, and a ball
        // too big for the block's side is refused.
        let b = prism(&[[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]], 6.0);
        let e = b
            .edge_ids()
            .find(|&e| {
                let x = b.edge(e);
                let (p, q) = (b.vertex(x.start).point, b.vertex(x.end).point);
                [p, q]
                    .iter()
                    .all(|v| (v.x - 10.0).abs() < 1e-12 && (v.y - 8.0).abs() < 1e-12)
            })
            .unwrap();
        let from = Point3::new(10.0, 8.0, 0.0);
        let out = fillet_variable_raw(&b, e, from, 1.0, 2.0).unwrap();
        let k = 1.0 - PI / 4.0;
        let v = measure::volume(&out);
        assert!(
            v > 480.0 - k * 4.0 * 6.0 && v < 480.0 - k * 1.0 * 6.0,
            "{v}"
        );
        let foot = Point3::new(10.0, 8.0 - 1.5, 3.0);
        let n = measure::normal_at(&out, foot, 1e-6).unwrap();
        assert!((n - Vec3::X).norm() < 1e-6, "{n:?}");
        assert!(fillet_variable_raw(&b, e, from, 1.0, 9.0).is_err());
    }
}
