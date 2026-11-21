//! A root and a peak of a function of one variable that is only continuous: Brent's methods
//! (Brent 1973, Algorithms for Minimization without Derivatives, ch. 4 and 5). Each takes an
//! interpolating step where the function is smooth, and falls back to a bisection or a
//! golden-section step where it is not, so a corner or a flat stretch cannot stall it.

/// Where `f` changes sign between `a` and `b`, whose values `fa` and `fb` lie on opposite sides
/// of zero (a zero counts as negative): a point with a sign change within `4ε|x| + tol` of it,
/// or an exact zero. Inverse quadratic interpolation or a secant step where it stays inside the
/// bracket and shrinks it fast enough, else a bisection.
pub(super) fn root(
    f: &mut dyn FnMut(f64) -> f64,
    (mut a, mut fa): (f64, f64),
    (mut b, mut fb): (f64, f64),
    tol: f64,
) -> f64 {
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;
    // Bisection alone ends within 2100 steps from any bracket of finite numbers.
    for _ in 0..4200 {
        if (fb > 0.0) == (fc > 0.0) {
            (c, fc) = (a, fa);
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            (a, fa) = (b, fb);
            (b, fb) = (c, fc);
            (c, fc) = (a, fa);
        }
        let eps = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
        let m = 0.5 * (c - b);
        if m.abs() <= eps || fb == 0.0 {
            break;
        }
        if e.abs() >= eps && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let (q, r) = (fa / fc, fb / fc);
                (
                    s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (eps * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        (a, fa) = (b, fb);
        b += if d.abs() > eps { d } else { eps.copysign(m) };
        fb = f(b);
    }
    b
}

/// A peak of `f` over `[a, b]`, climbed from `x` in it, where `f` is `fx`: the highest value
/// found and where, the peak bracketed within `4·(2ε|x| + tol)`. A step to the top of the
/// parabola through the best three points where it lands inside the bracket and is under half the
/// step before last, else a golden-section step into the larger part. Where `f` has several peaks
/// in the bracket it finds one of them. The bracket is brought down to `tol` whatever the
/// function's rounding, as a peak at a corner is that sharp.
pub(super) fn peak(
    f: &mut dyn FnMut(f64) -> f64,
    (mut a, mut b): (f64, f64),
    (mut x, mut fx): (f64, f64),
    tol: f64,
) -> (f64, f64) {
    let golden = 0.5 * (3.0 - 5f64.sqrt());
    let (mut w, mut fw) = (x, fx);
    let (mut v, mut fv) = (x, fx);
    let (mut d, mut e) = (0.0f64, 0.0f64);
    // Golden section alone ends within 1500 steps from any bracket of finite numbers.
    for _ in 0..3000 {
        let m = 0.5 * (a + b);
        let eps = 2.0 * f64::EPSILON * x.abs() + tol;
        if (x - m).abs() <= 2.0 * eps - 0.5 * (b - a) {
            break;
        }
        let (mut p, mut q, mut r) = (0.0, 0.0, 0.0);
        if e.abs() > eps {
            // The parabola through the three: of −f, whose least value is the peak.
            r = (x - w) * (fv - fx);
            q = (x - v) * (fw - fx);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if p.abs() < (0.5 * q * r).abs() && p > q * (a - x) && p < q * (b - x) {
            d = p / q;
            let u = x + d;
            if u - a < 2.0 * eps || b - u < 2.0 * eps {
                d = if x < m { eps } else { -eps };
            }
        } else {
            e = if x < m { b - x } else { a - x };
            d = golden * e;
        }
        let u = x + if d.abs() >= eps { d } else { eps.copysign(d) };
        let fu = f(u);
        if fu >= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            (v, fv) = (w, fw);
            (w, fw) = (x, fx);
            (x, fx) = (u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu >= fw || w == x {
                (v, fv) = (w, fw);
                (w, fw) = (u, fu);
            } else if fu >= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }
    (fx, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counted_root(f: &dyn Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> (f64, usize) {
        let mut n = 0;
        let mut g = |x: f64| {
            n += 1;
            f(x)
        };
        let x = root(&mut g, (a, f(a)), (b, f(b)), tol);
        (x, n)
    }

    fn counted_peak(f: &dyn Fn(f64) -> f64, a: f64, b: f64, x: f64, tol: f64) -> (f64, f64, usize) {
        let mut n = 0;
        let mut g = |x: f64| {
            n += 1;
            f(x)
        };
        let (fx, at) = peak(&mut g, (a, b), (x, f(x)), tol);
        (fx, at, n)
    }

    #[test]
    fn a_root_of_a_smooth_function_is_found_in_a_few_steps() {
        let (x, n) = counted_root(&|x| x * x * x - 2.0, 0.0, 3.0, 1e-14);
        assert!((x - 2f64.cbrt()).abs() < 1e-13, "{x}");
        assert!(n < 20, "{n}");
    }

    // The larger of two lines has a corner at the root's side: the slope changes from 1 to 50 at
    // x = 0.3, a hair past the root at 0.25.
    #[test]
    fn a_root_beside_a_corner_is_found() {
        let f = |x: f64| (x - 0.25).max(50.0 * (x - 0.3) + 0.05);
        let (x, n) = counted_root(&f, -1.0, 1.0, 1e-14);
        assert!((x - 0.25).abs() < 1e-13, "{x}");
        assert!(n < 60, "{n}");
        // The corner itself as the root.
        let f = |x: f64| (x - 0.3).max(50.0 * (x - 0.3));
        let (x, n) = counted_root(&f, -1.0, 1.0, 1e-14);
        assert!((x - 0.3).abs() < 1e-13, "{x}");
        assert!(n < 60, "{n}");
    }

    // Flat at −1 up to 0.7, then a steep rise through zero at 0.7 + 1e-3: the secant of the ends
    // lands on the flat stretch again and again.
    #[test]
    fn a_root_past_a_flat_stretch_is_found() {
        let f = |x: f64| (1e3 * (x - 0.7) - 1.0).clamp(-1.0, 2.0);
        let (x, n) = counted_root(&f, -4.0, 4.0, 1e-14);
        assert!((x - 0.701).abs() < 1e-13, "{x}");
        assert!(n < 80, "{n}");
    }

    // A zero of high order, where the function is flat about its root.
    #[test]
    fn a_root_the_function_is_flat_about_is_bracketed() {
        let f = |x: f64| (x - 0.4).powi(9);
        let (x, _) = counted_root(&f, -1.0, 1.0, 1e-14);
        assert!((x - 0.4).abs() < 1e-12, "{x}");
    }

    // A jump through zero: the root is the place of the jump, to the tolerance.
    #[test]
    fn a_jump_through_zero_is_bracketed() {
        let f = |x: f64| if x > 0.123 { 1.0 } else { -3.0 };
        let (x, n) = counted_root(&f, -1.0, 1.0, 1e-14);
        assert!((x - 0.123).abs() < 2e-14, "{x}");
        assert!(n < 120, "{n}");
    }

    #[test]
    fn a_zero_at_an_end_is_returned() {
        let (x, _) = counted_root(&|x| x - 1.0, 1.0, 2.0, 1e-14);
        assert_eq!(x, 1.0);
        let (x, _) = counted_root(&|x| 2.0 - x, 1.0, 2.0, 1e-14);
        assert_eq!(x, 2.0);
    }

    #[test]
    fn a_smooth_peak_is_found_in_a_few_steps() {
        let f = |x: f64| -(x - 0.3) * (x - 0.3) + (x - 0.3).powi(3);
        let (fx, x, n) = counted_peak(&f, -1.0, 1.0, 0.0, 1e-9);
        assert!((x - 0.3).abs() < 1e-7 && fx.abs() < 1e-14, "{x} {fx}");
        assert!(n < 20, "{n}");
    }

    // The smaller of two lines peaks at a corner, as the depth does where two faces are as near.
    #[test]
    fn a_peak_at_a_corner_is_found() {
        let f = |x: f64| (x + 0.2).min(3.0 * (0.1 - x));
        let top = 0.025;
        for start in [-1.0, -0.5, 0.0, 0.9, 1.0] {
            let (fx, x, n) = counted_peak(&f, -1.0, 1.0, start, 1e-10);
            assert!((x - top).abs() < 1e-9, "{start}: {x}");
            assert!((fx - 0.225).abs() < 1e-9, "{start}: {fx}");
            assert!(n < 70, "{n}");
        }
    }

    // A plateau with a single higher point region: flat at 0 except a tent about 0.6.
    #[test]
    fn a_peak_between_flat_stretches_is_found_from_on_it() {
        let f = |x: f64| (0.05 - (x - 0.6).abs()).max(0.0);
        let (fx, x, _) = counted_peak(&f, -1.0, 1.0, 0.58, 1e-10);
        assert!(
            (x - 0.6).abs() < 1e-9 && (fx - 0.05).abs() < 1e-9,
            "{x} {fx}"
        );
    }

    // A function that only rises has its peak at the bracket's end.
    #[test]
    fn a_peak_at_an_end_is_reached() {
        let (fx, x, _) = counted_peak(&|x| x, 0.0, 2.0, 1.0, 1e-10);
        assert!((x - 2.0).abs() < 1e-9 && (fx - 2.0).abs() < 1e-9, "{x}");
        let (_, x, _) = counted_peak(&|x| -x, 0.0, 2.0, 0.0, 1e-10);
        assert!(x.abs() < 1e-9, "{x}");
    }

    // A flat function: every point is a peak, and the search ends.
    #[test]
    fn a_flat_function_ends() {
        let (fx, x, n) = counted_peak(&|_| 1.5, 0.0, 2.0, 0.7, 1e-10);
        assert!((0.0..=2.0).contains(&x) && fx == 1.5);
        assert!(n < 100, "{n}");
    }
}
