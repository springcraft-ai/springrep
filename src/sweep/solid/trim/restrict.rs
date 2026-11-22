//! A spline surface cut down to a rectangle of its parameters, the parameters kept: each end of the
//! rectangle inserted as a knot until the spline is cut there (Boehm 1980; Piegl and Tiller 1997,
//! ch. 5), and the control points between kept. A trimmed face of a sheet that folds or runs into
//! itself lies on one fold-free part of the sheet; on that part alone a point of the face has one
//! foot, where on the whole sheet it has another past the fold or across the self-crossing.

use crate::{
    geom::{BSpline, Curve, Extrusion, NurbsCurve, NurbsSurface, Revolution, Surface},
    math::{Point3, Vec2},
};

/// A point of a rational spline in homogeneous form: the point times its weight, and the weight.
type H = [f64; 4];

/// Knot `t` inserted once into the spline of `degree` over `knots` with points `pts` (Boehm).
fn insert(knots: &[f64], degree: usize, pts: &[H], t: f64) -> (Vec<f64>, Vec<H>) {
    let k = (degree..pts.len())
        .rev()
        .find(|&i| knots[i] <= t)
        .unwrap_or(degree);
    let mut q = Vec::with_capacity(pts.len() + 1);
    for i in 0..=pts.len() {
        if i + degree <= k {
            q.push(pts[i]);
        } else if i > k {
            q.push(pts[i - 1]);
        } else {
            let den = knots[i + degree] - knots[i];
            let a = if den > 0.0 { (t - knots[i]) / den } else { 0.0 };
            let (p0, p1) = (pts[i - 1], pts[i]);
            q.push(std::array::from_fn(|c| (1.0 - a) * p0[c] + a * p1[c]));
        }
    }
    let mut kn = knots.to_vec();
    kn.insert(k + 1, t);
    (kn, q)
}

/// The knots and points of the spline over `knots` cut to `[a, b]`, a range inside its domain.
fn cut(knots: &[f64], degree: usize, pts: &[H], a: f64, b: f64) -> (Vec<f64>, Vec<H>) {
    let (mut kn, mut q) = (knots.to_vec(), pts.to_vec());
    for t in [a, b] {
        let have = kn.iter().filter(|&&k| k == t).count();
        for _ in have..degree {
            (kn, q) = insert(&kn, degree, &q, t);
        }
    }
    // The last copy of `a` ends the points before the cut, which `degree` copies hold at `a`.
    let last_a = kn.iter().rposition(|&k| k == a).unwrap_or(degree);
    let start = last_a - degree;
    let mut out_knots = vec![a; degree + 1];
    out_knots.extend(kn.iter().copied().filter(|&k| k > a && k < b));
    out_knots.extend(std::iter::repeat_n(b, degree + 1));
    let count = out_knots.len() - degree - 1;
    (out_knots, q[start..start + count].to_vec())
}

fn homogeneous(p: Point3, w: f64) -> H {
    [p.x * w, p.y * w, p.z * w, w]
}

fn point(h: &H) -> Point3 {
    Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
}

/// `range` cut to the part of `domain` it covers, or `None` where that is all of the domain.
fn within(domain: (f64, f64), range: (f64, f64)) -> Option<(f64, f64)> {
    let a = range.0.max(domain.0);
    let b = range.1.min(domain.1);
    (b > a && (a > domain.0 || b < domain.1)).then_some((a, b))
}

/// The surface `n` over `u` by `v`, each a range of its parameter.
fn nurbs(n: &NurbsSurface, u: (f64, f64), v: (f64, f64)) -> Option<NurbsSurface> {
    let (du, dv) = n.degrees();
    let (mut ku, mut kv) = (n.knots().0.to_vec(), n.knots().1.to_vec());
    let (mut nu, mut nv) = n.counts();
    let w = n.weights();
    let mut net: Vec<H> = n
        .control_points()
        .iter()
        .enumerate()
        .map(|(i, &p)| homogeneous(p, w.map_or(1.0, |w| w[i])))
        .collect();
    let (dom_u, dom_v) = n.domain();
    if let Some((a, b)) = within(dom_u, u) {
        let mut cols = Vec::with_capacity(nv);
        let mut knots = Vec::new();
        for j in 0..nv {
            let col: Vec<H> = (0..nu).map(|i| net[i * nv + j]).collect();
            let (k, q) = cut(&ku, du, &col, a, b);
            knots = k;
            cols.push(q);
        }
        nu = cols[0].len();
        net = (0..nu)
            .flat_map(|i| cols.iter().map(move |c| c[i]))
            .collect();
        ku = knots;
    }
    if let Some((a, b)) = within(dom_v, v) {
        let mut next = Vec::new();
        let mut knots = Vec::new();
        for i in 0..nu {
            let (k, q) = cut(&kv, dv, &net[i * nv..(i + 1) * nv], a, b);
            knots = k;
            next.extend(q);
        }
        nv = next.len() / nu;
        net = next;
        kv = knots;
    }
    let ctrl = net.iter().map(point).collect();
    let weights = w.map(|_| net.iter().map(|p| p[3]).collect());
    let cut = NurbsSurface::new((du, dv), (ku, kv), (nu, nv), ctrl, weights).ok()?;
    // The cut keeps the parameters, so it is the same sweep of the same curve over less of each.
    Some(match n.swept() {
        Some(swept) => cut.swept_by(swept.clone()),
        None => cut,
    })
}

/// The curve `c` over `range` of its parameter, where it is a spline that is not periodic.
fn curve(c: &Curve, range: (f64, f64)) -> Option<Curve> {
    match c {
        Curve::BSpline(s) if !s.is_periodic() => {
            let (a, b) = within(s.domain(), range)?;
            let pts: Vec<H> = s
                .control_points()
                .iter()
                .map(|&p| homogeneous(p, 1.0))
                .collect();
            let (k, q) = cut(s.knots(), s.degree(), &pts, a, b);
            BSpline::new(s.degree(), k, q.iter().map(point).collect())
                .ok()
                .map(Curve::BSpline)
        }
        Curve::Nurbs(n) if n.period().is_none() => {
            let (a, b) = within(n.domain(), range)?;
            let w = n.weights();
            let pts: Vec<H> = n
                .control_points()
                .iter()
                .enumerate()
                .map(|(i, &p)| homogeneous(p, w.map_or(1.0, |w| w[i])))
                .collect();
            let (k, q) = cut(n.knots(), n.degree(), &pts, a, b);
            let weights = w.map(|_| q.iter().map(|p| p[3]).collect());
            NurbsCurve::new(n.degree(), k, q.iter().map(point).collect(), weights)
                .ok()
                .map(|n| Curve::Nurbs(Box::new(n)))
        }
        _ => None,
    }
}

/// `s` cut down to the rectangle from `lo` to `hi` of its parameters where it is a spline surface,
/// or a swept one whose curve is a spline, with the same parameters; else `None`.
pub(super) fn restricted(s: &Surface, lo: Vec2, hi: Vec2) -> Option<Surface> {
    match s {
        Surface::Nurbs(n) => {
            nurbs(n, (lo.x, hi.x), (lo.y, hi.y)).map(|n| Surface::Nurbs(Box::new(n)))
        }
        Surface::Extrusion(e) => curve(&e.curve, (lo.x, hi.x))
            .map(|curve| Surface::Extrusion(Box::new(Extrusion { curve, dir: e.dir }))),
        Surface::Revolution(r) => curve(&r.curve, (lo.y, hi.y)).map(|curve| {
            Surface::Revolution(Box::new(Revolution {
                curve,
                origin: r.origin,
                axis: r.axis,
            }))
        }),
        _ => None,
    }
}
