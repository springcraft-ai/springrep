//! How two surfaces meet at a point already known to lie on both: whether they touch there only.
//!
//! Where the surfaces are tangent at the point, the gap between them is, to second order, the
//! quadratic form of the difference of their shape operators on the common tangent plane. When it
//! is definite the surfaces share the point alone: no intersection curve passes through it, and
//! within `reach` of it they lie within the tolerance of each other, so anything a trace finds there
//! is that point.

use super::csi;
use crate::{
    geom::Surface,
    math::{Point3, Vec2, Vec3},
    tol,
};

/// A point where two surfaces touch and share nothing else nearby.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Touch {
    pub p: Point3,
    /// How far from `p` the surfaces stay within the tolerance of each other.
    pub reach: f64,
}

/// The change of `s`'s unit normal along the tangent directions `e`, by central differences of the
/// parameters; at a pole, where they do not move along the surface, by [`pole_turn`].
fn normal_turn(s: &Surface, p: Point3, e: [Vec3; 2], flip: bool) -> Option<[Vec3; 2]> {
    let uv = s.project(p);
    let (_, su, sv) = s.d1(uv.x, uv.y);
    let (g11, g12, g22) = (su.dot(su), su.dot(sv), sv.dot(sv));
    let det = g11 * g22 - g12 * g12;
    let sign = if flip { -1.0 } else { 1.0 };
    if det <= f64::EPSILON * g11 * g22 {
        return pole_turn(s, uv, e).map(|t| t.map(|x| x * sign));
    }
    let turn = |d: Vec3| {
        let (bu, bv) = (su.dot(d) * STEP, sv.dot(d) * STEP);
        let step = Vec2::new((g22 * bu - g12 * bv) / det, (g11 * bv - g12 * bu) / det);
        let (a, b) = (uv + step, uv - step);
        (s.normal(a.x, a.y) - s.normal(b.x, b.y)) * (sign / (2.0 * STEP))
    };
    Some([turn(e[0]), turn(e[1])])
}

/// At a pole of a spline surface at `p`, the parameter lines leaving it into the patch, each as its
/// fixed parameter's value and a function giving its point at a parameter step from the pole along
/// it; `None` away from a pole.
fn pole_lines(s: &Surface, p: Point3) -> Option<Vec<(Vec2, Vec2)>> {
    let Surface::Nurbs(n) = s else { return None };
    let uv = s.project(p);
    let (q, su, sv) = s.d1(uv.x, uv.y);
    if q.dist(p) > tol::LINEAR || su.norm().min(sv.norm()) > 1e-9 * su.norm().max(sv.norm()) {
        return None;
    }
    let ((u0, u1), (v0, v1)) = n.domain();
    let free_u = su.norm() <= sv.norm();
    let mut out = Vec::new();
    for k in 0..=16 {
        let f = k as f64 / 16.0;
        out.push(if free_u {
            let inward = if uv.y - v0 > v1 - uv.y { -1.0 } else { 1.0 };
            (Vec2::new(u0 + (u1 - u0) * f, uv.y), Vec2::new(0.0, inward))
        } else {
            let inward = if uv.x - u0 > u1 - uv.x { -1.0 } else { 1.0 };
            (Vec2::new(uv.x, v0 + (v1 - v0) * f), Vec2::new(inward, 0.0))
        });
    }
    Some(out)
}

/// The touch at `p`, a pole of `s` on `other`: a surface need not curve as a quadratic form at its
/// pole (a patch closing on one point is curved by its own section in each half plane), so the gap
/// is measured along each line leaving the pole into the patch, a step from it. They touch only
/// there when every gap lies on one side, and stay within the tolerance out to the reach of the
/// least curved line.
fn pole_touch(s: &Surface, other: &Surface, p: Point3, lines: &[(Vec2, Vec2)]) -> Option<Touch> {
    let uv = other.project(p);
    let n_other = other.normal(uv.x, uv.y);
    let su = s.project(p);
    if s.normal(su.x, su.y).cross(n_other).norm() > tol::TANGENT {
        return None;
    }
    // A step of a micrometre along the surface.
    let h = 1e-3;
    let mut least = f64::INFINITY;
    let mut sign = 0.0;
    for &(at, inward) in lines {
        let (_, au, av) = s.d1(at.x, at.y);
        let speed = (au * inward.x + av * inward.y).norm();
        if speed == 0.0 {
            return None;
        }
        let w = at + inward * (h / speed);
        let q = s.eval(w.x, w.y);
        let gap = csi::signed_distance(other, q);
        let d = (q - p).norm();
        if gap.abs() <= f64::EPSILON * d || sign * gap < 0.0 {
            return None;
        }
        sign = gap.signum();
        least = least.min(2.0 * gap.abs() / (d * d));
    }
    Some(Touch {
        p,
        reach: (2.0 * tol::LINEAR / least).sqrt(),
    })
}

/// Steps of a tenth of a micrometre: the normals turn measurably, rounding does not.
const STEP: f64 = 1e-4;

/// The normal's turn along `e` at a pole of a spline surface, where one parameter is free. A patch
/// may meet its pole in a wedge, so no step crosses it: the turn is taken one-sided along two of
/// the lines leaving the pole into the patch, and being linear in the direction, is solved for
/// along `e` from those two.
fn pole_turn(s: &Surface, uv: Vec2, e: [Vec3; 2]) -> Option<[Vec3; 2]> {
    let Surface::Nurbs(n) = s else { return None };
    let ((u0, u1), (v0, v1)) = n.domain();
    let (_, su, sv) = s.d1(uv.x, uv.y);
    let free_u = su.norm() <= sv.norm();
    let n0 = s.normal(uv.x, uv.y);
    // A line leaving the pole at the free parameter's value `f`: its direction, and its point at
    // arc length `h`, to first order, as parameters.
    let leave = |f: f64, h: f64| -> Option<(Vec3, Vec2)> {
        let (at, inward) = if free_u {
            let inward = if uv.y - v0 > v1 - uv.y { -1.0 } else { 1.0 };
            (Vec2::new(f, uv.y), Vec2::new(0.0, inward))
        } else {
            let inward = if uv.x - u0 > u1 - uv.x { -1.0 } else { 1.0 };
            (Vec2::new(uv.x, f), Vec2::new(inward, 0.0))
        };
        let (_, au, av) = s.d1(at.x, at.y);
        let along = au * inward.x + av * inward.y;
        let speed = along.norm();
        (speed > 0.0).then(|| (along / speed, at + inward * (h / speed)))
    };
    let (lo, hi) = if free_u { (u0, u1) } else { (v0, v1) };
    let mut dirs = Vec::new();
    for f in [lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)] {
        let (d, a) = leave(f, STEP)?;
        let (_, b) = leave(f, 2.0 * STEP)?;
        let turn = (s.normal(a.x, a.y) * 4.0 - s.normal(b.x, b.y) - n0 * 3.0) / (2.0 * STEP);
        dirs.push((d, turn));
    }
    let [(da, ta), (db, tb)] = [dirs[0], dirs[1]];
    let (aa, ab, bb) = (da.dot(da), da.dot(db), db.dot(db));
    let det = aa * bb - ab * ab;
    if det <= 1e-12 * aa * bb {
        return None;
    }
    let along = |w: Vec3| {
        let (ra, rb) = (da.dot(w), db.dot(w));
        let (x, y) = ((bb * ra - ab * rb) / det, (aa * rb - ab * ra) / det);
        ta * x + tb * y
    };
    Some([along(e[0]), along(e[1])])
}

/// The eigenvalues, least first, of the gap form at `p` between tangent surfaces: the difference of
/// their normals' turn along the tangent plane, `b`'s normal taken the way `a`'s points. `a`'s
/// height over `b` along `a`'s normal is `−½ wᵀMw` for a step `w`. Also `a`'s normal at `p`.
fn gap_form(a: &Surface, b: &Surface, p: Point3) -> Option<((f64, f64), Vec3, Vec3)> {
    let (ua, ub) = (a.project(p), b.project(p));
    let (na, nb) = (a.normal(ua.x, ua.y), b.normal(ub.x, ub.y));
    if na.cross(nb).norm() > tol::TANGENT {
        return None;
    }
    let e0 = na.any_perpendicular().unit()?;
    let e = [e0, na.cross(e0)];
    let ta = normal_turn(a, p, e, false)?;
    let tb = normal_turn(b, p, e, na.dot(nb) < 0.0)?;
    let m = |i: usize, j: usize| (ta[i] - tb[i]).dot(e[j]);
    let (m11, m22, m12) = (m(0, 0), m(1, 1), 0.5 * (m(0, 1) + m(1, 0)));
    let mean = 0.5 * (m11 + m22);
    let spread = (0.25 * (m11 - m22) * (m11 - m22) + m12 * m12).sqrt();
    Some(((mean - spread, mean + spread), na, nb))
}

/// How far either side of `q`, a point of both surfaces on a curve along `dir` where they are
/// tangent, they stay within the tolerance of each other: `√(2·tol/κ)` for the curvature `κ` of
/// the gap across the curve, measured a step off it each way; zero where they are not tangent.
pub(crate) fn band(a: &Surface, b: &Surface, q: Point3, dir: Vec3) -> f64 {
    let (ua, ub) = (a.project(q), b.project(q));
    let (na, nb) = (a.normal(ua.x, ua.y), b.normal(ub.x, ub.y));
    if na.cross(nb).norm() > tol::TANGENT {
        return 0.0;
    }
    let Some(across) = na.cross(dir).unit() else {
        return 0.0;
    };
    // A step of a micrometre: past the band of any surfaces that bend apart measurably.
    let h = 1e-3;
    let gap = [1.0, -1.0]
        .into_iter()
        .map(|side: f64| {
            let w = a.project(q + across * (side * h));
            csi::signed_distance(b, a.eval(w.x, w.y)).abs()
        })
        .fold(0.0, f64::max);
    if gap <= tol::LINEAR {
        return h;
    }
    (2.0 * tol::LINEAR / (2.0 * gap / (h * h))).sqrt()
}

/// How far from `p`, a point of both surfaces where they are tangent and cross, a point within the
/// tolerance of both can lie in every direction: `√(2·tol/|λ|)` for the gap form's eigenvalue `λ`
/// of largest size. Two branches of their intersection cross there, and a trace of either passes
/// the point only to within that reach. `None` where they are not tangent at `p`, or the gap form
/// is definite (they touch at `p` alone), nearly singular (they touch along a curve), or bends
/// apart by less than [`CROSSING_BEND`] (they lie along each other over a patch).
pub(crate) fn crossing_reach(a: &Surface, b: &Surface, p: Point3) -> Option<f64> {
    let ((l1, l2), _, _) = gap_form(a, b, p)?;
    let (least, most) = (l1.abs().min(l2.abs()), l1.abs().max(l2.abs()));
    // A form whose one eigenvalue is this small a share of the other is read off differences of
    // normals that only nearly agree along a curve of contact, not off two branches crossing.
    (l1 * l2 < 0.0 && least > 1e-3 * most && most >= CROSSING_BEND)
        .then(|| (2.0 * tol::LINEAR / most).sqrt())
}

/// The least bend of the gap, per mm, at which two tangent surfaces are taken to cross at a point:
/// flatter, they stay within the tolerance of each other for over a hundredth of a millimetre all
/// round, as two nearly flat faces of a milled floor do, and any vertex there would be taken as on
/// every curve through that patch.
const CROSSING_BEND: f64 = 1e-3;

/// A guard against rounding in the normals' differences, not a tolerance: an eigenvalue nearer
/// zero than this share of the other may be zero.
const ROUNDING: f64 = 1e-6;

/// Which side of `b` the surface `a` lies on near `p`, a point of both where they touch without
/// crossing: `Some(true)` where `a` bends away along `b`'s normal. `None` where the gap form
/// changes sign (they cross) or vanishes (they osculate, or coincide).
pub(crate) fn side_of(a: &Surface, b: &Surface, p: Point3) -> Option<bool> {
    let ((l1, l2), na, nb) = gap_form(a, b, p)?;
    let big = l1.abs().max(l2.abs());
    if big == 0.0 || !big.is_finite() {
        return None;
    }
    let guard = ROUNDING * big;
    // A semidefinite form is a touch along the null direction's curve; off it, the other eigenvalue
    // alone gives the side.
    let below = if l1 >= -guard && l2 > guard {
        true
    } else if l2 <= guard && l1 < -guard {
        false
    } else {
        return None;
    };
    Some(below != (na.dot(nb) > 0.0))
}

/// The touch at `p`, which lies on both surfaces: `Some` where their normals are parallel within
/// [`tol::TANGENT`] and their gap is a definite quadratic form.
pub(crate) fn touch_at(a: &Surface, b: &Surface, p: Point3) -> Option<Touch> {
    for (s, other) in [(a, b), (b, a)] {
        if let Some(lines) = pole_lines(s, p) {
            return pole_touch(s, other, p, &lines);
        }
    }
    let ((l1, l2), _, _) = gap_form(a, b, p)?;
    // Nearer zero than the guard the form may be only semidefinite, the surfaces touching along a
    // curve.
    let small = l1.abs().min(l2.abs());
    if l1 * l2 <= 0.0 || small <= ROUNDING * l1.abs().max(l2.abs()) {
        return None;
    }
    Some(Touch {
        p,
        reach: (2.0 * tol::LINEAR / small).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{Cylinder, Plane, Sphere};
    use crate::math::Frame;

    fn cyl(o: [f64; 3], z: [f64; 3], r: f64) -> Surface {
        let frame = Frame::from_axis(Vec3::from_array(o), Vec3::from_array(z)).unwrap();
        Surface::Cylinder(Cylinder { frame, radius: r })
    }

    #[test]
    fn a_plane_on_a_spline_sphere_s_pole_touches_it_there() {
        // The plane z = 2 meets the sphere of radius 2 at its pole, where the octant's parameters
        // do not move: the gap is (x² + y²)/4, so they stay within the tolerance for √(4·tol) mm.
        let sphere = crate::geom::octant(2.0);
        let frame = Frame::from_axis(Vec3::new(0.0, 0.0, 2.0), Vec3::Z).unwrap();
        let plane = Surface::Plane(Plane { frame });
        let t = touch_at(&plane, &sphere, Vec3::new(0.0, 0.0, 2.0)).expect("a point touch");
        let want = (4.0 * tol::LINEAR).sqrt();
        assert!((t.reach - want).abs() <= 1e-4 * want, "{}", t.reach);
    }

    #[test]
    fn crossed_cylinders_bulging_apart_touch_at_a_point() {
        // Radius 1 about x below z = 0 and radius 4 about y above it: the gap is x²/8 + y²/2, so
        // its smaller curvature is 1/4 and they stay within 1e-7 for √(2e-7 · 4) mm.
        let a = cyl([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], 1.0);
        let b = cyl([0.0, 0.0, 4.0], [0.0, 1.0, 0.0], 4.0);
        let t = touch_at(&a, &b, Point3::ZERO).expect("a point touch");
        assert!(
            (t.reach - (8e-7f64).sqrt()).abs() <= 1e-6 * t.reach,
            "{}",
            t.reach
        );
    }

    #[test]
    fn nested_cylinders_crossing_at_a_tangent_point_are_not_a_point_touch() {
        // Radius 1 about x and radius 4 about y, both below z = 0: the gap x²/8 − y²/2 changes
        // sign, so two curves cross at the point.
        let a = cyl([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], 1.0);
        let b = cyl([0.0, 0.0, -4.0], [0.0, 1.0, 0.0], 4.0);
        assert!(touch_at(&a, &b, Point3::ZERO).is_none());
    }

    #[test]
    fn a_plane_on_a_cylinder_touches_along_a_line_not_a_point() {
        let a = cyl([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], 1.0);
        let b = Surface::Plane(Plane {
            frame: Frame::from_axis(Point3::ZERO, Vec3::Z).unwrap(),
        });
        assert!(touch_at(&a, &b, Point3::ZERO).is_none());
    }

    #[test]
    fn a_ball_on_a_plane_touches_at_a_point_and_crossing_surfaces_do_not() {
        // Its axis along x, so the touching point lies on its equator, not at a pole.
        let ball = Surface::Sphere(Sphere {
            frame: Frame::from_axis(Vec3::new(0.0, 0.0, 2.0), Vec3::X).unwrap(),
            radius: 2.0,
        });
        let floor = Surface::Plane(Plane {
            frame: Frame::from_axis(Point3::ZERO, Vec3::Z).unwrap(),
        });
        let t = touch_at(&ball, &floor, Point3::ZERO).expect("a point touch");
        assert!((t.reach - (4e-7f64).sqrt()).abs() <= 1e-6 * t.reach);
        let wall = Surface::Plane(Plane {
            frame: Frame::from_axis(Point3::ZERO, Vec3::X).unwrap(),
        });
        assert!(touch_at(&ball, &wall, Point3::ZERO).is_none());
    }

    // The plane x = R − r is tangent to the torus about z at its inner equator, a saddle that
    // bends 1/r toward the plane along its tube and 1/(R − r) away along the equator: the gap form
    // is ±1/r there for R = 2r, and the two cross along z = ±y within √(2·tol·r). A torus a hundred
    // metres across bends apart from its plane by under a thousandth per millimetre: no crossing.
    #[test]
    fn a_plane_tangent_to_a_torus_inside_crosses_it_unless_the_torus_is_flat() {
        let reach = |r: f64| {
            let torus = Surface::Torus(crate::geom::Torus {
                frame: Frame::WORLD,
                major: 2.0 * r,
                minor: r,
            });
            let frame = Frame::from_axis(Point3::new(r, 0.0, 0.0), Vec3::X).unwrap();
            let plane = Surface::Plane(Plane { frame });
            crossing_reach(&torus, &plane, Point3::new(r, 0.0, 0.0))
        };
        let near = reach(1.0).expect("a crossing");
        let want = (2.0 * tol::LINEAR).sqrt();
        assert!((near - want).abs() <= 1e-6 * want, "{near} {want}");
        assert!(reach(5e4).is_none());
    }
}
