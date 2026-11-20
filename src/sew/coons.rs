//! The bilinearly blended Coons patch of a loop of four polynomial curves (Coons 1967; Farin 2002,
//! "Coons patches"), as one B-spline surface.

use super::Span;
use crate::{
    Error, Result,
    error::invalid,
    geom::NurbsSurface,
    loft::rail::{Rail, make_compatible, point},
    math::Point3,
    tol,
};

/// The bilinearly blended Coons patch of a loop of four spans, head to tail: `S(u, 0)` runs along
/// the first, `S(1, v)` along the second, `S(u, 1)` back along the third and `S(0, v)` back along
/// the fourth, each over `[0, 1]` in proportion to its parameter. The patch is
/// `(1 − v)·S(u, 0) + v·S(u, 1) + (1 − u)·S(0, v) + u·S(1, v)` less the bilinear blend of the four
/// corners, so it passes through all four spans. Each span must be polynomial: lines, B-splines
/// and B-splines whose weights are all equal.
pub fn coons(boundary: &[Span; 4]) -> Result<NurbsSurface> {
    for (i, s) in boundary.iter().enumerate() {
        let gap = s.end().dist(boundary[(i + 1) % 4].start());
        if gap > tol::LINEAR {
            return invalid(format!(
                "a Coons patch's loop is open by {gap:.1e} mm after its span {i}"
            ));
        }
    }
    let rail = |s: &Span, back: bool| -> Result<Vec<Rail>> {
        let (t0, t1) = if back { (s.t1, s.t0) } else { (s.t0, s.t1) };
        let r = Rail::of_curve(&s.curve, t0, t1)?.normalized();
        let w = r.pw[0][3];
        if r.pw.iter().any(|h| (h[3] / w - 1.0).abs() > 1e-12) {
            return Err(Error::Unsupported(format!(
                "a Coons patch of a rational {} curve, which is no B-spline surface",
                s.curve.kind()
            )));
        }
        Ok(vec![r])
    };
    let mut along_u = [rail(&boundary[0], false)?, rail(&boundary[2], true)?].concat();
    let mut along_v = [rail(&boundary[3], true)?, rail(&boundary[1], false)?].concat();
    make_compatible(&mut along_u);
    make_compatible(&mut along_v);
    let ctrl = |r: &Rail| -> Vec<Point3> { r.pw.iter().map(|&h| point(h)).collect() };
    let (c0, c1) = (ctrl(&along_u[0]), ctrl(&along_u[1]));
    let (d0, d1) = (ctrl(&along_v[0]), ctrl(&along_v[1]));
    // Each corner is where two spans meet, within the tolerance checked above.
    let p00 = c0[0].lerp(d0[0], 0.5);
    let p10 = c0[c0.len() - 1].lerp(d1[0], 0.5);
    let p01 = c1[0].lerp(d0[d0.len() - 1], 0.5);
    let p11 = c1[c1.len() - 1].lerp(d1[d1.len() - 1], 0.5);
    // A B-spline whose coefficients are a linear function at the Greville abscissae is that
    // linear function (de Boor 1978), so each linear blend is exact on the other direction's
    // knots.
    let greville = |r: &Rail| -> Vec<f64> {
        let p = r.degree;
        (0..r.pw.len())
            .map(|i| r.knots[i + 1..=i + p].iter().sum::<f64>() / p as f64)
            .collect()
    };
    let (xi, eta) = (greville(&along_u[0]), greville(&along_v[0]));
    let mut net = Vec::with_capacity(xi.len() * eta.len());
    for (i, &u) in xi.iter().enumerate() {
        for (j, &v) in eta.iter().enumerate() {
            let ruled_v = c0[i] * (1.0 - v) + c1[i] * v;
            let ruled_u = d0[j] * (1.0 - u) + d1[j] * u;
            let corners = (p00 * (1.0 - u) + p10 * u) * (1.0 - v) + (p01 * (1.0 - u) + p11 * u) * v;
            net.push(ruled_v + ruled_u - corners);
        }
    }
    let (ru, rv) = (&along_u[0], &along_v[0]);
    NurbsSurface::new(
        (ru.degree, rv.degree),
        (ru.knots.clone(), rv.knots.clone()),
        (xi.len(), eta.len()),
        net,
        None,
    )
}
