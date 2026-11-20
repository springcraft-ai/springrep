//! Section curves as clamped rational B-splines in homogeneous coordinates, and the operations
//! that make a column of them share one degree and knot vector, so their control points can be
//! skinned into a surface.

use std::f64::consts::FRAC_PI_2;

use crate::{
    Error, Result,
    geom::Curve,
    math::{Frame, Point3, Vec3},
    numeric::{BandLu, SparseRow, solve_dense},
};

/// `(w·x, w·y, w·z, w)`.
pub(crate) type Hom = [f64; 4];

pub(crate) fn hom(p: Point3, w: f64) -> Hom {
    [p.x * w, p.y * w, p.z * w, w]
}

pub(crate) fn point(h: Hom) -> Point3 {
    Vec3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
}

fn mix(a: Hom, b: Hom, t: f64) -> Hom {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// A clamped B-spline on `[0, 1]` once [`Rail::normalized`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Rail {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub pw: Vec<Hom>,
}

impl Rail {
    /// The curve from `t0` to `t1` (either order) of a line, conic, spline or fitted intersection.
    pub(crate) fn of_curve(curve: &Curve, t0: f64, t1: f64) -> Result<Rail> {
        let r = match curve {
            Curve::Line(_) => Rail {
                degree: 1,
                knots: vec![0.0, 0.0, 1.0, 1.0],
                pw: vec![hom(curve.eval(t0), 1.0), hom(curve.eval(t1), 1.0)],
            },
            Curve::Circle(c) => conic(&c.frame, c.radius, c.radius, t0, t1),
            Curve::Ellipse(e) => conic(&e.frame, e.major, e.minor, t0, t1),
            Curve::BSpline(b) => {
                let whole = Rail {
                    degree: b.degree(),
                    knots: b.knots().to_vec(),
                    pw: b.control_points().iter().map(|&p| hom(p, 1.0)).collect(),
                };
                let sub = whole.sub(t0.min(t1), t0.max(t1))?;
                if t0 > t1 { sub.reversed() } else { sub }
            }
            // A fitted intersection curve is its spline.
            Curve::Intersection(c) => {
                return Rail::of_curve(&Curve::BSpline(c.approx.clone()), t0, t1);
            }
            Curve::Nurbs(n) => {
                let w = n.weights().map(|w| w.to_vec());
                let whole = Rail {
                    degree: n.degree(),
                    knots: n.knots().to_vec(),
                    pw: n
                        .control_points()
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| hom(p, w.as_ref().map_or(1.0, |w| w[i])))
                        .collect(),
                };
                let sub = whole.sub(t0.min(t1), t0.max(t1))?;
                if t0 > t1 { sub.reversed() } else { sub }
            }
        };
        Ok(r.normalized())
    }

    pub(crate) fn domain(&self) -> (f64, f64) {
        (self.knots[self.degree], self.knots[self.pw.len()])
    }

    pub(crate) fn start(&self) -> Point3 {
        point(self.pw[0])
    }

    pub(crate) fn end(&self) -> Point3 {
        point(self.pw[self.pw.len() - 1])
    }

    fn multiplicity(&self, t: f64) -> usize {
        self.knots.iter().filter(|&&k| k == t).count()
    }

    /// Inserts `t` once (Boehm's algorithm).
    fn insert(&mut self, t: f64) {
        let (p, n) = (self.degree, self.pw.len());
        let u = &self.knots;
        let mut k = p;
        while k + 1 < n && u[k + 1] <= t {
            k += 1;
        }
        let mut q = Vec::with_capacity(n + 1);
        for i in 0..=n {
            if i + p <= k {
                q.push(self.pw[i]);
            } else if i > k {
                q.push(self.pw[i - 1]);
            } else {
                let den = u[i + p] - u[i];
                let a = if den > 0.0 { (t - u[i]) / den } else { 0.0 };
                q.push(mix(self.pw[i - 1], self.pw[i], a));
            }
        }
        self.pw = q;
        self.knots.insert(k + 1, t);
    }

    fn insert_to(&mut self, t: f64, m: usize) {
        while self.multiplicity(t) < m {
            self.insert(t);
        }
    }

    /// The clamped piece over `[t0, t1]`, `t0 < t1` inside the domain.
    fn sub(&self, t0: f64, t1: f64) -> Result<Rail> {
        let p = self.degree;
        let mut r = self.clone();
        r.insert_to(t0, p);
        r.insert_to(t1, p);
        let first = r.knots.iter().position(|&k| k == t0).unwrap_or(0);
        let j0 = first + r.multiplicity(t0) - 1 - p;
        let last_start = r.knots.iter().position(|&k| k == t1).unwrap_or(0);
        let j1 = last_start - 1;
        let mut knots = vec![t0; p + 1];
        knots.extend(r.knots.iter().filter(|&&k| k > t0 && k < t1));
        knots.extend(std::iter::repeat_n(t1, p + 1));
        let pw = r.pw[j0..=j1].to_vec();
        if knots.len() != pw.len() + p + 1 {
            return Err(Error::Numerical(
                "B-spline piece has a malformed knot vector".into(),
            ));
        }
        Ok(Rail {
            degree: p,
            knots,
            pw,
        })
    }

    pub(crate) fn reversed(&self) -> Rail {
        let (a, b) = (self.knots[0], self.knots[self.knots.len() - 1]);
        Rail {
            degree: self.degree,
            knots: self.knots.iter().rev().map(|&k| a + b - k).collect(),
            pw: self.pw.iter().rev().copied().collect(),
        }
    }

    /// The same curve with its domain mapped onto `[0, 1]`.
    pub(crate) fn normalized(&self) -> Rail {
        let (a, b) = self.domain();
        let mut knots: Vec<f64> = self.knots.iter().map(|&k| (k - a) / (b - a)).collect();
        for k in &mut knots {
            *k = k.clamp(0.0, 1.0);
        }
        Rail {
            knots,
            ..self.clone()
        }
    }

    /// The distinct knots strictly inside the domain.
    pub(crate) fn breaks(&self) -> Vec<f64> {
        let (a, b) = self.domain();
        let mut out: Vec<f64> = Vec::new();
        for &k in &self.knots {
            if k > a && k < b && out.last().is_none_or(|&l| k > l) {
                out.push(k);
            }
        }
        out
    }

    /// The same curve at degree `q ≥ degree`, split into Bézier pieces to raise each.
    pub(crate) fn elevated(&self, q: usize) -> Rail {
        let p = self.degree;
        if q <= p {
            return self.clone();
        }
        let mut r = self.clone();
        let breaks = r.breaks();
        for &b in &breaks {
            r.insert_to(b, p);
        }
        let segs = (r.pw.len() - 1) / p;
        let mut pw: Vec<Hom> = Vec::new();
        for s in 0..segs {
            let mut bez: Vec<Hom> = r.pw[s * p..=s * p + p].to_vec();
            for d in p..q {
                let mut up = Vec::with_capacity(d + 2);
                up.push(bez[0]);
                for i in 1..=d {
                    let a = i as f64 / (d + 1) as f64;
                    up.push(mix(bez[i], bez[i - 1], a));
                }
                up.push(bez[d]);
                bez = up;
            }
            if s > 0 {
                bez.remove(0);
            }
            pw.extend(bez);
        }
        let (a, b) = r.domain();
        let mut knots = vec![a; q + 1];
        for &k in &breaks {
            knots.extend(std::iter::repeat_n(k, q));
        }
        knots.extend(std::iter::repeat_n(b, q + 1));
        Rail {
            degree: q,
            knots,
            pw,
        }
    }

    /// The point at `t`.
    pub(crate) fn at(&self, t: f64) -> Point3 {
        let mut h = [0.0; 4];
        for (b, p) in basis(self.degree, &self.knots, t).iter().zip(&self.pw) {
            for i in 0..4 {
                h[i] += b * p[i];
            }
        }
        point(h)
    }

    /// The same curve with its parameter mapped piecewise linearly, `from[i]` onto `to[i]`, both
    /// rising and spanning the domain: each `from[i]` becomes a knot of full multiplicity first,
    /// so every piece is a polynomial of its own and the map changes no point.
    pub(crate) fn reparameterized(&self, from: &[f64], to: &[f64]) -> Rail {
        let r = self.refined(&from[1..from.len() - 1]);
        let map = |t: f64| {
            let i = from
                .windows(2)
                .position(|w| t <= w[1])
                .unwrap_or(from.len() - 2);
            let s = (t - from[i]) / (from[i + 1] - from[i]);
            to[i] + (to[i + 1] - to[i]) * s.clamp(0.0, 1.0)
        };
        Rail {
            knots: r.knots.iter().map(|&k| map(k)).collect(),
            ..r
        }
    }

    /// The same curve with every one of `breaks` a knot of full multiplicity.
    pub(crate) fn refined(&self, breaks: &[f64]) -> Rail {
        let mut r = self.clone();
        for &b in breaks {
            r.insert_to(b, r.degree);
        }
        r
    }
}

/// The blossom of the Bézier `pts` at `ts`, one parameter per de Casteljau level.
fn blossom(pts: &[Hom], ts: &[f64]) -> Hom {
    let mut q = pts.to_vec();
    for (k, &t) in ts.iter().enumerate() {
        for i in 0..q.len() - 1 - k {
            q[i] = mix(q[i], q[i + 1], t);
        }
    }
    q[0]
}

/// The clamped B-spline `(knots, pw)` of `degree` carried on past both ends of its domain by
/// `delta`, each end span's polynomial continued; unchanged inside the domain.
pub(crate) fn extended(
    degree: usize,
    knots: &[f64],
    pw: &[Hom],
    delta: f64,
) -> (Vec<f64>, Vec<Hom>) {
    let p = degree;
    let mut r = Rail {
        degree,
        knots: knots.to_vec(),
        pw: pw.to_vec(),
    };
    let (a, b) = r.domain();
    let first = r.knots.iter().copied().find(|&k| k > a).unwrap_or(b);
    if first < b {
        r.insert_to(first, p);
    }
    let last = r.knots.iter().rev().copied().find(|&k| k < b).unwrap_or(a);
    if last > a {
        r.insert_to(last, p);
    }
    let lo = -delta / (first - a);
    let head: Vec<Hom> = (0..=p)
        .map(|i| {
            let ts: Vec<f64> = (0..p).map(|k| if k < p - i { lo } else { 1.0 }).collect();
            blossom(&r.pw[..=p], &ts)
        })
        .collect();
    let n = r.pw.len();
    let hi = (b + delta - last) / (b - last);
    let tail: Vec<Hom> = (0..=p)
        .map(|i| {
            let ts: Vec<f64> = (0..p).map(|k| if k < p - i { 0.0 } else { hi }).collect();
            blossom(&r.pw[n - 1 - p..], &ts)
        })
        .collect();
    if n == p + 1 {
        // One span: both ends continue the same polynomial.
        let up = (b + delta - a) / (b - a);
        let ts =
            |i: usize| -> Vec<f64> { (0..p).map(|k| if k < p - i { lo } else { up }).collect() };
        let pw = (0..=p).map(|i| blossom(&r.pw, &ts(i))).collect();
        let mut knots = vec![a - delta; p + 1];
        knots.extend(std::iter::repeat_n(b + delta, p + 1));
        return (knots, pw);
    }
    r.pw[..=p].copy_from_slice(&head);
    r.pw[n - 1 - p..].copy_from_slice(&tail);
    let m = r.knots.len();
    for k in &mut r.knots[..=p] {
        *k = a - delta;
    }
    for k in &mut r.knots[m - 1 - p..] {
        *k = b + delta;
    }
    (r.knots, r.pw)
}

/// An arc of `frame.origin + a·cos t·x + b·sin t·y` from `t0` to `t1` as rational quadratics, one
/// per quarter turn or less, each with end weights 1.
fn conic(frame: &Frame, a: f64, b: f64, t0: f64, t1: f64) -> Rail {
    let n = ((t1 - t0).abs() / FRAC_PI_2 - 1e-9).ceil().max(1.0) as usize;
    let step = (t1 - t0) / n as f64;
    let at =
        |t: f64, s: f64| frame.origin + frame.x * (a * t.cos() * s) + frame.y * (b * t.sin() * s);
    let w = (step / 2.0).cos();
    let mut pw = vec![hom(at(t0, 1.0), 1.0)];
    let mut knots = vec![0.0; 3];
    for k in 0..n {
        let ta = t0 + step * k as f64;
        pw.push(hom(at(ta + step / 2.0, 1.0 / w), w));
        pw.push(hom(at(ta + step, 1.0), 1.0));
        let e = (k + 1) as f64;
        knots.extend(if k + 1 == n { vec![e; 3] } else { vec![e; 2] });
    }
    Rail {
        degree: 2,
        knots,
        pw,
    }
}

/// Every rail of one column at one degree on one knot vector.
pub(crate) fn make_compatible(rails: &mut [Rail]) {
    let same = rails.windows(2).all(|w| {
        w[0].degree == w[1].degree
            && w[0].knots.len() == w[1].knots.len()
            && w[0]
                .knots
                .iter()
                .zip(&w[1].knots)
                .all(|(a, b)| (a - b).abs() <= 1e-12)
    });
    if same {
        let knots = rails[0].knots.clone();
        for r in rails.iter_mut() {
            r.knots = knots.clone();
        }
        return;
    }
    let q = rails.iter().map(|r| r.degree).max().unwrap_or(1);
    let mut all: Vec<f64> = rails.iter().flat_map(|r| r.breaks()).collect();
    all.sort_by(f64::total_cmp);
    let mut breaks: Vec<f64> = Vec::new();
    for k in all {
        if breaks.last().is_none_or(|&l| k - l > 1e-12) {
            breaks.push(k);
        }
    }
    for r in rails.iter_mut() {
        let mut e = r.elevated(q);
        for k in &mut e.knots {
            if let Some(&b) = breaks.iter().find(|&&b| (b - *k).abs() <= 1e-12) {
                *k = b;
            }
        }
        *r = e.refined(&breaks);
    }
}

/// A linear map from data at `n` sections to the control points of a curve across them, the
/// B-spline of `degree` on `knots`.
#[derive(Clone, Debug)]
pub(crate) struct Across {
    pub degree: usize,
    pub knots: Vec<f64>,
    map: Map,
}

#[derive(Clone, Debug)]
enum Map {
    Dense { coef: Vec<f64>, n: usize },
    Band(BandLu),
}

impl Across {
    pub(crate) fn controls(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    pub(crate) fn apply(&self, data: &[Hom]) -> Vec<Hom> {
        match &self.map {
            Map::Dense { coef, n } => (0..self.controls())
                .map(|k| {
                    let mut c = [0.0; 4];
                    for (s, d) in data.iter().enumerate() {
                        let f = coef[k * n + s];
                        for i in 0..4 {
                            c[i] += f * d[i];
                        }
                    }
                    c
                })
                .collect(),
            Map::Band(lu) => {
                let mut x = data.to_vec();
                lu.solve(&mut x);
                x
            }
        }
    }

    pub(crate) fn eval(&self, ctrl: &[Hom], v: f64) -> Hom {
        let vals = basis(self.degree, &self.knots, v);
        let mut out = [0.0; 4];
        for (c, b) in ctrl.iter().zip(vals) {
            for i in 0..4 {
                out[i] += b * c[i];
            }
        }
        out
    }
}

/// Every basis function of `degree` on `knots` at `t` (Cox–de Boor), right-continuous except at
/// the domain's end.
pub(crate) fn basis(degree: usize, knots: &[f64], t: f64) -> Vec<f64> {
    let n = knots.len() - degree - 1;
    let end = knots[n];
    let mut nb: Vec<f64> = (0..knots.len() - 1)
        .map(|i| {
            let inside = knots[i] <= t && t < knots[i + 1];
            let last = t >= end && knots[i] < end && knots[i + 1] >= end;
            if inside || last { 1.0 } else { 0.0 }
        })
        .collect();
    for d in 1..=degree {
        for i in 0..knots.len() - 1 - d {
            let l = knots[i + d] - knots[i];
            let r = knots[i + d + 1] - knots[i + 1];
            let a = if l > 0.0 {
                (t - knots[i]) / l * nb[i]
            } else {
                0.0
            };
            let b = if r > 0.0 {
                (knots[i + d + 1] - t) / r * nb[i + 1]
            } else {
                0.0
            };
            nb[i] = a + b;
        }
    }
    nb.truncate(n);
    nb
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The Bézier of `degree` through the first and last data points that fits the others at
/// `params` by least squares.
pub(crate) fn bezier_fit(params: &[f64], degree: usize) -> Result<Across> {
    let n = params.len();
    let p = degree;
    let bern =
        |k: usize, v: f64| binomial(p, k) * v.powi(k as i32) * (1.0 - v).powi((p - k) as i32);
    let mut coef = vec![0.0; (p + 1) * n];
    coef[0] = 1.0;
    coef[p * n + n - 1] = 1.0;
    let m = p - 1;
    if m > 0 {
        let rows = n - 2;
        let mut ata = vec![0.0; m * m];
        let mut atb = vec![0.0; m * rows];
        for (r, &v) in params[1..n - 1].iter().enumerate() {
            for a in 0..m {
                let ba = bern(a + 1, v);
                atb[a * rows + r] = ba;
                for b in 0..m {
                    ata[a * m + b] += ba * bern(b + 1, v);
                }
            }
        }
        let x = solve_dense(ata, atb, m, rows)?;
        for a in 0..m {
            let k = a + 1;
            for (r, &v) in params[1..n - 1].iter().enumerate() {
                let g = x[a * rows + r];
                coef[k * n + r + 1] += g;
                coef[k * n] -= g * bern(0, v);
                coef[k * n + n - 1] -= g * bern(p, v);
            }
        }
    }
    let mut knots = vec![0.0; p + 1];
    knots.extend(std::iter::repeat_n(1.0, p + 1));
    Ok(Across {
        degree: p,
        knots,
        map: Map::Dense { coef, n },
    })
}

/// The cubic (quadratic, linear for fewer sections) through every data point at `params`, with
/// not-a-knot ends.
pub(crate) fn interpolation(params: &[f64]) -> Result<Across> {
    let n = params.len();
    let p = (n - 1).min(3);
    let mut knots = vec![params[0]; p + 1];
    if p == 3 {
        knots.extend_from_slice(&params[2..n - 2]);
    }
    knots.extend(std::iter::repeat_n(params[n - 1], p + 1));
    let rows: Vec<SparseRow> = params
        .iter()
        .map(|&t| {
            let b = basis(p, &knots, t);
            let first = b.iter().position(|&v| v != 0.0).unwrap_or(0);
            let last = b.iter().rposition(|&v| v != 0.0).unwrap_or(first);
            (first, b[first..=last].to_vec())
        })
        .collect();
    Ok(Across {
        degree: p,
        knots,
        map: Map::Band(BandLu::factor(&rows)?),
    })
}
