//! Surfaces through the points of a map of two parameters: a plane or an extrusion where the points
//! lie on one, else the B-spline that interpolates them on a grid in those parameters (C. de Boor,
//! "A Practical Guide to Splines", 1978, chapter XVII), its grid refined where it strays until it
//! lies within [`SHEET_FIT`] of the map halfway between the grid's points.

use super::super::grazing::FIT;
use crate::{
    Error, Result,
    geom::{BSpline, Curve, Extrusion, NurbsSurface, Plane, Surface},
    math::{Frame, Point3, Vec2},
};

/// How near a fitted sheet lies to its map. Tighter than a curve's: where a sheet's two parameter
/// directions run nearly together, a point off it by `d` is read off its parameters by far more.
const SHEET_FIT: f64 = FIT / 10.0;

/// The surface through `map` over the rectangle from `lo` to `hi`, whose points are `size` across,
/// held to the map where `care` says it matters, and a spline one through it exactly at each of
/// `sites`.
pub(super) fn surface(
    map: &dyn Fn(Vec2) -> Point3,
    lo: Vec2,
    hi: Vec2,
    size: f64,
    care: &dyn Fn(Vec2) -> bool,
    sites: &[Vec2],
) -> Result<Surface> {
    let at = |i: usize, j: usize, n: usize| {
        map(Vec2::new(
            lo.x + (hi.x - lo.x) * i as f64 / n as f64,
            lo.y + (hi.y - lo.y) * j as f64 / n as f64,
        ))
    };
    let k = 8;
    let pts: Vec<Point3> = (0..=k)
        .flat_map(|i| (0..=k).map(move |j| (i, j)))
        .map(|(i, j)| at(i, j, k))
        .collect();
    let exact = 1e-11 * size.max(1.0);
    if let Some(plane) = flat(&pts, exact) {
        return Ok(plane);
    }
    // Ruled along y with one direction everywhere: the extrusion of its edge at `lo.y`.
    let w = at(0, k, k) - at(0, 0, k);
    let straight = (0..=k).all(|i| {
        let (a, b) = (at(i, 0, k), at(i, k, k));
        (b - a - w).norm() <= exact
            && (1..k).all(|j| (at(i, j, k) - a.lerp(b, j as f64 / k as f64)).norm() <= exact)
    });
    // Points are found on an extrusion through its curve seen along it, so the curve must not
    // come round over itself so seen: a path winding about the extrusion's direction is fitted.
    if straight
        && let Some(dir) = w.unit()
        && winding(&|x| map(Vec2::new(x, lo.y)), lo.x, hi.x, dir) < std::f64::consts::PI
    {
        let curve = fit_curve(&|x| map(Vec2::new(x, lo.y)), (lo.x, hi.x))?;
        return Ok(Surface::Extrusion(Box::new(Extrusion { curve, dir })));
    }
    let ruled = (0..=k).all(|i| {
        let (a, b) = (at(i, 0, k), at(i, k, k));
        (1..k).all(|j| (at(i, j, k) - a.lerp(b, j as f64 / k as f64)).norm() <= exact)
    });
    let even = |a: f64, b: f64, n: usize| -> Vec<f64> {
        (0..=n).map(|i| a + (b - a) * i as f64 / n as f64).collect()
    };
    // The grid's lines pass through the sites, so the sheet holds them exactly: a vertex where the
    // map drops rank, at a cusp of the envelope, is then found at its own parameters, which no
    // projection onto the sheet there can recover.
    let (mut xs, mut ys) = (
        through(even(lo.x, hi.x, 16), sites.iter().map(|q| q.x)),
        even(lo.y, hi.y, if ruled { 1 } else { 4 }),
    );
    if !ruled {
        ys = through(ys, sites.iter().map(|q| q.y));
    }
    for _ in 0..16 {
        let s = grid(map, &xs, &ys)?;
        let err = |u: f64, v: f64| {
            let q = Vec2::new(u, v);
            if care(q) {
                s.eval(u, v).dist(map(q))
            } else {
                0.0
            }
        };
        let mid = |a: &[f64], i: usize| 0.5 * (a[i] + a[i + 1]);
        // The worst stray at the middle of each interval each way, along the other way's lines;
        // a cell's middle counts toward the way whose lines stray more there.
        let mut bad_x: Vec<f64> = (0..xs.len() - 1)
            .map(|i| ys.iter().map(|&v| err(mid(&xs, i), v)).fold(0.0, f64::max))
            .collect();
        let mut bad_y: Vec<f64> = (0..ys.len() - 1)
            .map(|j| xs.iter().map(|&u| err(u, mid(&ys, j))).fold(0.0, f64::max))
            .collect();
        for (i, bx) in bad_x.iter_mut().enumerate() {
            for (j, by) in bad_y.iter_mut().enumerate() {
                let e = err(mid(&xs, i), mid(&ys, j));
                if *bx >= *by || ruled {
                    *bx = bx.max(e);
                } else {
                    *by = by.max(e);
                }
            }
        }
        let worst = bad_x.iter().chain(&bad_y).fold(0.0f64, |a, &b| a.max(b));
        if worst <= SHEET_FIT {
            return Ok(Surface::Nurbs(Box::new(s)));
        }
        xs = refine(&xs, &bad_x);
        ys = refine(&ys, &bad_y);
        if xs.len() > 4097 || ys.len() > 1025 || xs.len() * ys.len() > 1 << 18 {
            break;
        }
    }
    Err(Error::Numerical(
        "a swept surface's fit did not reach its tolerance".into(),
    ))
}

/// The increasing `lines` with a line at each of `sites` inside them: the nearest inner line moved
/// there where it is within a quarter of a spacing, else one more line.
fn through(mut lines: Vec<f64>, sites: impl Iterator<Item = f64>) -> Vec<f64> {
    let (lo, hi) = (lines[0], lines[lines.len() - 1]);
    for x in sites {
        if x.is_nan() || x <= lo || x >= hi || lines.contains(&x) {
            continue;
        }
        let i = lines.partition_point(|&l| l < x);
        let width = lines[i] - lines[i - 1];
        let near = if x - lines[i - 1] < lines[i] - x {
            i - 1
        } else {
            i
        };
        if near > 0 && near + 1 < lines.len() && (lines[near] - x).abs() <= 0.25 * width {
            lines[near] = x;
        } else {
            lines.insert(i, x);
        }
    }
    lines
}

/// `sites` with each interval whose middle strays more than [`SHEET_FIT`] halved, and then each
/// interval more than twice as long as a neighbour halved, so the spacing changes gradually.
fn refine(sites: &[f64], bad: &[f64]) -> Vec<f64> {
    let mut out = vec![sites[0]];
    for i in 0..bad.len() {
        if bad[i] > SHEET_FIT {
            out.push(0.5 * (sites[i] + sites[i + 1]));
        }
        out.push(sites[i + 1]);
    }
    loop {
        let w: Vec<f64> = out.windows(2).map(|p| p[1] - p[0]).collect();
        let long: Vec<usize> = (0..w.len())
            .filter(|&i| {
                (i > 0 && w[i] > 2.0 * w[i - 1]) || (i + 1 < w.len() && w[i] > 2.0 * w[i + 1])
            })
            .collect();
        if long.is_empty() {
            return out;
        }
        for &i in long.iter().rev() {
            out.insert(i + 1, 0.5 * (out[i] + out[i + 1]));
        }
    }
}

/// How far the tangent of `c` over `[a, b]`, seen along `dir`, turns in all.
fn winding(c: &dyn Fn(f64) -> Point3, a: f64, b: f64, dir: crate::math::Vec3) -> f64 {
    let n = 256;
    let flat = |p: Point3| p - dir * p.dot(dir);
    let pts: Vec<Point3> = (0..=n)
        .map(|i| flat(c(a + (b - a) * i as f64 / n as f64)))
        .collect();
    let mut total = 0.0;
    for i in 1..n {
        let (d0, d1) = (pts[i] - pts[i - 1], pts[i + 1] - pts[i]);
        let s = d0.cross(d1).dot(dir);
        total += s.atan2(d0.dot(d1)).abs();
    }
    total
}

/// The plane all of `pts` lie on within `exact`, if there is one.
fn flat(pts: &[Point3], exact: f64) -> Option<Surface> {
    let p0 = pts[0];
    let p1 = *pts
        .iter()
        .max_by(|a, b| a.dist(p0).total_cmp(&b.dist(p0)))?;
    let d = (p1 - p0).unit()?;
    let off = |p: Point3| (p - p0) - d * (p - p0).dot(d);
    let p2 = *pts
        .iter()
        .max_by(|a, b| off(**a).norm().total_cmp(&off(**b).norm()))?;
    let n = d.cross(p2 - p0).unit()?;
    if pts.iter().all(|&p| (p - p0).dot(n).abs() <= exact) {
        let frame = Frame::new(p0, n, d).ok()?;
        return Some(Surface::Plane(Plane { frame }));
    }
    None
}

/// The B-spline through `map` on the grid of lines `xs` by `ys`, at the grid's own parameters.
fn grid(map: &dyn Fn(Vec2) -> Point3, xs: &[f64], ys: &[f64]) -> Result<NurbsSurface> {
    // Each row along x through its points, then each column of their control points along y.
    let mut rows: Vec<Spline> = Vec::with_capacity(ys.len());
    for &y in ys {
        let pts: Vec<Point3> = xs.iter().map(|&x| map(Vec2::new(x, y))).collect();
        rows.push(interpolate(&pts, xs)?);
    }
    let cu = rows[0].ctrl.len();
    let mut cols: Vec<Spline> = Vec::with_capacity(cu);
    for i in 0..cu {
        let pts: Vec<Point3> = rows.iter().map(|r| r.ctrl[i]).collect();
        cols.push(interpolate(&pts, ys)?);
    }
    let cv = cols[0].ctrl.len();
    let mut ctrl = Vec::with_capacity(cu * cv);
    for c in &cols {
        ctrl.extend_from_slice(&c.ctrl);
    }
    NurbsSurface::new(
        (rows[0].degree, cols[0].degree),
        (rows[0].knots.clone(), cols[0].knots.clone()),
        (cu, cv),
        ctrl,
        None,
    )
}

/// Degree of the splines the grid is interpolated with where it has points enough: a quintic's
/// error falls with the sixth power of the spacing, so a sheet that winds several turns needs far
/// fewer spans than a cubic's.
const DEGREE: usize = 5;

/// A B-spline's degree, knots and control points.
struct Spline {
    degree: usize,
    knots: Vec<f64>,
    ctrl: Vec<Point3>,
}

/// The spline of [`DEGREE`] through `pts` at the increasing `params`, its inner knots at the data
/// sites with the first and last few left out (the not-a-knot ends of de Boor, chapter XIII); a
/// cubic where there are too few points.
fn interpolate(pts: &[Point3], params: &[f64]) -> Result<Spline> {
    let n = pts.len();
    if n <= DEGREE + 1 {
        let c = BSpline::interpolate_at(pts, params)?;
        return Ok(Spline {
            degree: c.degree(),
            knots: c.knots().to_vec(),
            ctrl: c.control_points().to_vec(),
        });
    }
    let p = DEGREE;
    let skip = p.div_ceil(2);
    let mut knots = vec![params[0]; p + 1];
    knots.extend_from_slice(&params[skip..n - skip]);
    knots.extend(std::iter::repeat_n(params[n - 1], p + 1));
    let rows: Vec<(usize, Vec<f64>)> = params.iter().map(|&t| local_basis(p, &knots, t)).collect();
    let rhs: Vec<f64> = pts.iter().flat_map(|q| [q.x, q.y, q.z]).collect();
    let x = crate::numeric::solve_banded(rows, rhs, 3)?;
    let ctrl = x.chunks(3).map(|c| Point3::new(c[0], c[1], c[2])).collect();
    Ok(Spline {
        degree: p,
        knots,
        ctrl,
    })
}

/// The basis functions of `degree` on the clamped `knots` that do not vanish at `t`, and the index
/// of the first (de Boor, chapter X).
fn local_basis(degree: usize, knots: &[f64], t: f64) -> (usize, Vec<f64>) {
    let n = knots.len() - degree - 1;
    // The span holding `t`, the last one at the domain's end.
    let span = (degree..n)
        .rev()
        .find(|&i| knots[i] <= t && knots[i] < knots[i + 1])
        .unwrap_or(degree);
    let mut b = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    b[0] = 1.0;
    for j in 1..=degree {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        let mut saved = 0.0;
        for r in 0..j {
            let tmp = b[r] / (right[r + 1] + left[j - r]);
            b[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        b[j] = saved;
    }
    (span - degree, b)
}

/// The cubic B-spline through `at` over `range` at its own parameter, its points refined until it
/// lies within [`FIT`] of `at` halfway between them. A cubic, as the points nearest a cubic's
/// extrusion are found along it directly.
pub(super) fn fit_curve(at: &dyn Fn(f64) -> Point3, range: (f64, f64)) -> Result<Curve> {
    let mut n = 16usize;
    while n <= 1 << 14 {
        let xs: Vec<f64> = (0..=n)
            .map(|i| range.0 + (range.1 - range.0) * i as f64 / n as f64)
            .collect();
        let pts: Vec<Point3> = xs.iter().map(|&x| at(x)).collect();
        let c = BSpline::interpolate_at(&pts, &xs)?;
        let worst = (0..n)
            .map(|i| {
                let x = 0.5 * (xs[i] + xs[i + 1]);
                c.eval(x).dist(at(x))
            })
            .fold(0.0, f64::max);
        if worst <= FIT {
            return Ok(Curve::BSpline(c));
        }
        n *= 2;
    }
    Err(Error::Numerical(
        "a swept curve's fit did not reach its tolerance".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A sheet that rises by 1 across a band a hundredth wide, `z = tanh((y − ½) / 0.01)`, needs
    // grid lines a few thousandths apart in the band and few elsewhere: the grid is refined only
    // where it strays, so it fits within the tolerance everywhere.
    #[test]
    fn a_sheet_with_a_steep_band_is_refined_in_the_band() {
        let map = |q: Vec2| Point3::new(q.x, q.y, ((q.y - 0.5) / 0.01).tanh() + q.x * q.x);
        let (lo, hi) = (Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0));
        let Surface::Nurbs(s) = surface(&map, lo, hi, 2.0, &|_| true, &[]).unwrap() else {
            panic!("not a spline");
        };
        let n = 97;
        for i in 0..=n {
            for j in 0..=n {
                let q = Vec2::new(i as f64 / n as f64, j as f64 / n as f64);
                let d = s.eval(q.x, q.y).dist(map(q));
                assert!(d <= 10.0 * SHEET_FIT, "{d} at {q:?}");
            }
        }
        // An even grid fine enough for the band would hold hundreds of rows.
        assert!(
            s.control_points().len() < 64 * 200,
            "{}",
            s.control_points().len()
        );
    }

    // The sheet `(u + v, (u − v)², (u − v)³)` has both tangents along x where `u = v`: a cusp, past
    // which it folds back. A cubic, its quintic interpolant is exact. The point at `(½, ½)` is
    // found there from a start a little off, as both ends of two edges meeting at a cusp of a
    // swept face must be: the distance falls only with the fourth power of the step there.
    #[test]
    fn a_point_at_a_cusp_projects_to_its_own_parameters() {
        let map = |q: Vec2| {
            let w = q.x - q.y;
            Point3::new(q.x + q.y, w * w, w * w * w)
        };
        let lines: Vec<f64> = (0..=8).map(|i| i as f64 / 8.0).collect();
        let s = grid(&map, &lines, &lines).unwrap();
        let foot = Vec2::new(0.5, 0.5);
        for hint in [
            Vec2::new(0.503, 0.497),
            Vec2::new(0.49, 0.51),
            Vec2::new(0.52, 0.5),
        ] {
            let uv = s.project_near(map(foot), hint);
            assert!(uv.dist(foot) <= 2e-8, "{uv:?} from {hint:?}");
        }
    }
}
