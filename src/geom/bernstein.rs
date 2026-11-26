//! Polynomials over the unit square in tensor-product Bernstein form: their bounds, products,
//! derivatives and de Casteljau halves (L. Piegl and W. Tiller, *The NURBS Book*, 2nd ed., 1997,
//! ch. 1; R. T. Farouki and V. T. Rajan, "Algorithms for polynomials in Bernstein form", CAGD 5,
//! 1988), with Newton's method for a common zero of two of them.

use crate::math::Vec2;

/// A polynomial in tensor-product Bernstein form over its own unit square: `rows` coefficients
/// along u by `cols` along v.
#[derive(Clone, Debug)]
pub(crate) struct Net {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) c: Vec<f64>,
}

pub(crate) fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |b, i| b * (n - i) as f64 / (i + 1) as f64)
}

impl Net {
    pub(crate) fn at(&self, i: usize, j: usize) -> f64 {
        self.c[i * self.cols + j]
    }

    /// The least and greatest control value, between which the polynomial lies.
    pub(crate) fn range(&self) -> (f64, f64) {
        self.c
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                (lo.min(x), hi.max(x))
            })
    }

    pub(crate) fn clear_of_zero(&self) -> bool {
        let (lo, hi) = self.range();
        lo > 0.0 || hi < 0.0
    }

    /// The least and greatest control value of [`Net::derivative`], without building it.
    pub(crate) fn derivative_range(&self, across_u: bool) -> (f64, f64) {
        let len = if across_u { self.rows } else { self.cols };
        if len < 2 {
            return (0.0, 0.0);
        }
        let degree = (len - 1) as f64;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..self.rows - usize::from(across_u) {
            for j in 0..self.cols - usize::from(!across_u) {
                let next = if across_u {
                    self.at(i + 1, j)
                } else {
                    self.at(i, j + 1)
                };
                let d = degree * (next - self.at(i, j));
                lo = lo.min(d);
                hi = hi.max(d);
            }
        }
        (lo, hi)
    }

    /// The derivative across u, or across v: the steps between neighbouring control values times
    /// the degree (Piegl and Tiller, eq. 1.9).
    pub(crate) fn derivative(&self, across_u: bool) -> Net {
        let (rows, cols) = if across_u {
            (self.rows.saturating_sub(1).max(1), self.cols)
        } else {
            (self.rows, self.cols.saturating_sub(1).max(1))
        };
        let degree = if across_u { self.rows } else { self.cols } as f64 - 1.0;
        let c = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .map(|(i, j)| {
                if degree < 1.0 {
                    0.0
                } else if across_u {
                    degree * (self.at(i + 1, j) - self.at(i, j))
                } else {
                    degree * (self.at(i, j + 1) - self.at(i, j))
                }
            })
            .collect();
        Net { rows, cols, c }
    }

    /// The product of two polynomials, in Bernstein form of the summed degrees.
    pub(crate) fn times(&self, o: &Net) -> Net {
        let (m1, n1, m2, n2) = (self.rows - 1, self.cols - 1, o.rows - 1, o.cols - 1);
        let (rows, cols) = (m1 + m2 + 1, n1 + n2 + 1);
        let table = |n: usize| (0..=n).map(|k| binomial(n, k)).collect::<Vec<f64>>();
        let (bm1, bn1, bm2, bn2) = (table(m1), table(n1), table(m2), table(n2));
        let mut c = vec![0.0; rows * cols];
        for i1 in 0..=m1 {
            for j1 in 0..=n1 {
                let a = self.at(i1, j1) * bm1[i1] * bn1[j1];
                for i2 in 0..=m2 {
                    let line = &mut c[(i1 + i2) * cols + j1..(i1 + i2) * cols + j1 + n2 + 1];
                    for (j2, x) in line.iter_mut().enumerate() {
                        *x += a * o.at(i2, j2) * bm2[i2] * bn2[j2];
                    }
                }
            }
        }
        let (bm, bn) = (table(m1 + m2), table(n1 + n2));
        for i in 0..rows {
            for j in 0..cols {
                c[i * cols + j] /= bm[i] * bn[j];
            }
        }
        Net { rows, cols, c }
    }

    pub(crate) fn plus(&self, o: &Net) -> Net {
        Net {
            rows: self.rows,
            cols: self.cols,
            c: self.c.iter().zip(&o.c).map(|(a, b)| a + b).collect(),
        }
    }

    pub(crate) fn scaled(&self, k: f64) -> Net {
        Net {
            rows: self.rows,
            cols: self.cols,
            c: self.c.iter().map(|a| a * k).collect(),
        }
    }

    pub(crate) fn minus(&self, o: &Net) -> Net {
        Net {
            rows: self.rows,
            cols: self.cols,
            c: self.c.iter().zip(&o.c).map(|(a, b)| a - b).collect(),
        }
    }

    /// The two halves, cut across u or across v at the middle (de Casteljau).
    pub(crate) fn halves(&self, across_u: bool) -> [Net; 2] {
        let (lines, len) = if across_u {
            (self.cols, self.rows)
        } else {
            (self.rows, self.cols)
        };
        let idx = |line: usize, k: usize| {
            if across_u {
                k * self.cols + line
            } else {
                line * self.cols + k
            }
        };
        let (mut first, mut second) = (self.c.clone(), self.c.clone());
        for line in 0..lines {
            let mut level: Vec<f64> = (0..len).map(|k| self.c[idx(line, k)]).collect();
            for step in 0..len {
                first[idx(line, step)] = level[0];
                second[idx(line, len - 1 - step)] = level[len - 1 - step];
                for k in 0..len - 1 - step {
                    level[k] = 0.5 * (level[k] + level[k + 1]);
                }
            }
        }
        let net = |c| Net {
            rows: self.rows,
            cols: self.cols,
            c,
        };
        [net(first), net(second)]
    }

    pub(crate) fn eval(&self, s: f64, t: f64) -> f64 {
        if self.rows > SMALL {
            let along: Vec<f64> = (0..self.rows)
                .map(|i| scalar_casteljau(&self.c[i * self.cols..(i + 1) * self.cols], t))
                .collect();
            return scalar_casteljau(&along, s);
        }
        let mut along = [0.0; SMALL];
        for (i, a) in along.iter_mut().enumerate().take(self.rows) {
            *a = scalar_casteljau(&self.c[i * self.cols..(i + 1) * self.cols], t);
        }
        scalar_casteljau(&along[..self.rows], s)
    }
}

/// The longest run of control values worked on in place, without an allocation.
const SMALL: usize = 64;

pub(crate) fn scalar_casteljau(c: &[f64], t: f64) -> f64 {
    let mut small = [0.0; SMALL];
    let mut large = Vec::new();
    let level: &mut [f64] = if c.len() <= SMALL {
        small[..c.len()].copy_from_slice(c);
        &mut small[..c.len()]
    } else {
        large.extend_from_slice(c);
        &mut large
    };
    for k in (1..level.len()).rev() {
        for i in 0..k {
            level[i] = (1.0 - t) * level[i] + t * level[i + 1];
        }
    }
    level.first().copied().unwrap_or(0.0)
}

/// Interval product.
pub(crate) fn mul((a, b): (f64, f64), (c, d): (f64, f64)) -> (f64, f64) {
    let p = [a * c, a * d, b * c, b * d];
    (
        p.iter().copied().fold(f64::INFINITY, f64::min),
        p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// The zero of `(gu, gv)` in their piece's unit square by Newton's method from its middle, `d`
/// their derivatives across u and v: `None` unless it settles inside the piece.
pub(crate) fn newton(gu: &Net, gv: &Net, d: &[Net; 4]) -> Option<Vec2> {
    let (mut s, mut t) = (0.5, 0.5);
    for _ in 0..40 {
        let (f, g) = (gu.eval(s, t), gv.eval(s, t));
        let (a, b, c, e) = (
            d[0].eval(s, t),
            d[1].eval(s, t),
            d[2].eval(s, t),
            d[3].eval(s, t),
        );
        let (ds, dt) = crate::numeric::solve2(a, b, c, e, -f, -g)?;
        s += ds;
        t += dt;
        if !(-0.5..=1.5).contains(&s) || !(-0.5..=1.5).contains(&t) {
            return None;
        }
        if ds.abs() + dt.abs() <= 1e-14 {
            let m = 1e-9;
            return ((-m..=1.0 + m).contains(&s) && (-m..=1.0 + m).contains(&t))
                .then(|| Vec2::new(s.clamp(0.0, 1.0), t.clamp(0.0, 1.0)));
        }
    }
    None
}
