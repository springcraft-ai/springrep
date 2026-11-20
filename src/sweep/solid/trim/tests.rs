use super::section::Section;
use crate::{
    check,
    math::{Frame, Point3, Vec3},
    measure,
    primitive::torus,
    sweep::{Motion, solid},
};

// A torus's depth has a closed form: the tube's radius less the distance from its middle circle.
#[test]
fn a_torus_section_reads_the_depth_of_a_tube() {
    let frame = Frame::new(Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.3, 0.4, 1.0), Vec3::X).unwrap();
    let (big, small) = (3.0, 1.0);
    let ring = torus(&frame, big, small, "ring").unwrap();
    let section = Section::of(&ring).unwrap();
    for k in 0..500 {
        let a = k as f64 * 0.731;
        let p = Point3::new(
            4.0 * (a * 1.3).sin(),
            4.0 * (a * 0.7).cos(),
            2.0 * (a * 2.1).sin(),
        ) + frame.origin;
        let l = frame.to_local(p);
        let rho = (l.x * l.x + l.y * l.y).sqrt();
        let want = small - ((rho - big).powi(2) + l.z * l.z).sqrt();
        let got = section.depth(p);
        assert!((got - want).abs() <= 1e-12, "{got} {want} at {p:?}");
    }
}

/// The volume a torus (radii `big`, `small`, about `frame`) sweeps slid straight down by `drop`,
/// by the midpoint rule over upright lines `h` apart: along each, the stretches inside the torus,
/// each from its closed form `(|q|² + R² − r²)² = 4R²(|q|² − (q·a)²)`, run down by `drop` and
/// joined.
fn slid_torus_volume(frame: &Frame, big: f64, small: f64, drop: f64, h: f64) -> f64 {
    let inside = |p: Point3| {
        let l = frame.to_local(p);
        let rho = (l.x * l.x + l.y * l.y).sqrt();
        (rho - big).powi(2) + l.z * l.z < small * small
    };
    let reach = big + small;
    let (z0, z1) = (frame.origin.z - reach, frame.origin.z + reach);
    let n = ((2.0 * reach) / h).ceil() as i64;
    let steps = 400;
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            let x = frame.origin.x - reach + h * (i as f64 + 0.5);
            let y = frame.origin.y - reach + h * (j as f64 + 0.5);
            let at = |k: usize| z0 + (z1 - z0) * k as f64 / steps as f64;
            let mut ends = Vec::new();
            let mut last = inside(Point3::new(x, y, at(0)));
            for k in 1..=steps {
                let now = inside(Point3::new(x, y, at(k)));
                if now != last {
                    let (mut lo, mut hi) = (at(k - 1), at(k));
                    for _ in 0..50 {
                        let m = 0.5 * (lo + hi);
                        if inside(Point3::new(x, y, m)) == last {
                            lo = m;
                        } else {
                            hi = m;
                        }
                    }
                    ends.push(0.5 * (lo + hi));
                }
                last = now;
            }
            let mut length = 0.0;
            let mut cur: Option<(f64, f64)> = None;
            for w in ends.chunks(2) {
                let (lo, hi) = (w[0] - drop, w[1]);
                cur = match cur {
                    Some((a, b)) if lo <= b => Some((a, b.max(hi))),
                    Some((a, b)) => {
                        length += b - a;
                        Some((lo, hi))
                    }
                    None => Some((lo, hi)),
                };
            }
            if let Some((a, b)) = cur {
                length += b - a;
            }
            total += length;
        }
    }
    total * h * h
}

// A ring leaning 65° pushed straight down: past a lean of atan √((R − r)/r) its inner outline, seen
// along the push, folds into two swallowtails, and the faces it sweeps run into each other there.
// The trimmed sweep is a valid solid of the volume its upright lines add up to.
#[test]
fn a_ring_pushed_past_its_fold_is_trimmed_where_it_folds() {
    let lean = 65f64.to_radians();
    let frame = Frame::new(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::new(lean.sin(), 0.0, lean.cos()),
        Vec3::X,
    )
    .unwrap();
    let ring = torus(&frame, 3.0, 1.0, "ring").unwrap();
    let b = solid(&ring, &Motion::Slide(Vec3::new(0.0, 0.0, -8.0)), "push").unwrap();
    assert!(check::check(&b).is_empty());
    let volume = measure::volume(&b);
    let want = slid_torus_volume(&frame, 3.0, 1.0, 8.0, 0.02);
    assert!((volume - want).abs() <= 1e-4 * want, "{volume} {want}");
}

// The block under the plane z = 0 less the trimmed sweep of the leaning ring: the sweep is symmetric
// through the origin, where the ring's path is halved, so the block loses half of it. The plane
// cuts each face of the sheet's self-crossing, and the cut curve passes the vertex there twice.
#[test]
fn a_block_less_a_ring_pushed_past_its_fold_loses_half_of_it() {
    let lean = 65f64.to_radians();
    let frame = Frame::new(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::new(lean.sin(), 0.0, lean.cos()),
        Vec3::X,
    )
    .unwrap();
    let ring = torus(&frame, 3.0, 1.0, "ring").unwrap();
    let b = solid(&ring, &Motion::Slide(Vec3::new(0.0, 0.0, -8.0)), "push").unwrap();
    let block = crate::primitive::block(
        &Frame::WORLD.translated(Vec3::new(-8.0, -8.0, -8.0)),
        Vec3::new(16.0, 16.0, 8.0),
        "block",
    )
    .unwrap();
    let cut = crate::boolean::difference(&block, &b).unwrap();
    assert!(check::check(&cut).is_empty());
    let want = 2048.0 - 0.5 * measure::volume(&b);
    let got = measure::volume(&cut);
    assert!((got - want).abs() <= 1e-8 * want, "{got} {want}");
}

/// The volume within `r` of the helix of radius `a` about the z axis rising `rise` a turn, from
/// turn 0 to `turns`, by the midpoint rule over upright lines `h` apart: along each, the stretches
/// within `r` of the helix, each end pinned by bisection, the distance the least over the helix's
/// parameter (samples, then Newton steps on the squared distance).
fn helix_tube_volume(a: f64, rise: f64, turns: f64, r: f64, h: f64) -> f64 {
    let end = turns * std::f64::consts::TAU;
    let k = rise / std::f64::consts::TAU;
    let dist2 = |p: Point3, s: f64| {
        let (x, y, z) = (p.x - a * s.cos(), p.y - a * s.sin(), p.z - k * s);
        x * x + y * y + z * z
    };
    let near = |p: Point3| {
        let n = 400;
        let mut best = (f64::INFINITY, 0.0);
        for i in 0..=n {
            let s = end * i as f64 / n as f64;
            let d = dist2(p, s);
            if d < best.0 {
                best = (d, s);
            }
        }
        let mut s = best.1;
        for _ in 0..30 {
            let ds = 1e-6;
            let (f0, fp, fm) = (dist2(p, s), dist2(p, s + ds), dist2(p, s - ds));
            let g = (fp - fm) / (2.0 * ds);
            let hh = (fp - 2.0 * f0 + fm) / (ds * ds);
            if hh <= 0.0 {
                break;
            }
            s = (s - g / hh).clamp(0.0, end);
        }
        dist2(p, s).min(best.0) <= r * r
    };
    let reach = a + r;
    let (z0, z1) = (-r, k * end + r);
    let n = ((2.0 * reach) / h).ceil() as i64;
    let steps = 200;
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            let x = -reach + h * (i as f64 + 0.5);
            let y = -reach + h * (j as f64 + 0.5);
            let at = |q: usize| z0 + (z1 - z0) * q as f64 / steps as f64;
            let mut last = near(Point3::new(x, y, at(0)));
            let mut from = if last { Some(at(0)) } else { None };
            for q in 1..=steps {
                let now = near(Point3::new(x, y, at(q)));
                if now != last {
                    let (mut lo, mut hi) = (at(q - 1), at(q));
                    for _ in 0..40 {
                        let m = 0.5 * (lo + hi);
                        if near(Point3::new(x, y, m)) == last {
                            lo = m;
                        } else {
                            hi = m;
                        }
                    }
                    let z = 0.5 * (lo + hi);
                    match from.take() {
                        Some(f) => total += z - f,
                        None => from = Some(z),
                    }
                }
                last = now;
            }
        }
    }
    total * h * h
}

// A ball of radius 2 screwed two turns round a helix of radius 1.5 rising 3 a turn: the helix's
// radius of curvature, (1.5² + (3/2π)²)/1.5 ≈ 1.65, is less than the ball's, so the tube about it
// folds on its inner side, and the turns, 3 apart, run into each other. The trimmed sweep is a
// valid solid of the volume within 2 of the helix.
#[test]
fn a_ball_screwed_round_a_helix_tighter_than_itself_is_trimmed_where_it_folds() {
    let ball = crate::primitive::sphere(
        &Frame::WORLD.translated(Vec3::new(1.5, 0.0, 0.0)),
        2.0,
        "ball",
    )
    .unwrap();
    let screw = crate::sweep::Helix {
        origin: Point3::ZERO,
        axis: Vec3::Z,
        rise: 3.0 / std::f64::consts::TAU,
        sign: 1.0,
    };
    let m = Motion::Screw {
        screw,
        from: 0.0,
        to: 2.0 * std::f64::consts::TAU,
    };
    let b = solid(&ball, &m, "screw").unwrap();
    assert!(check::check(&b).is_empty());
    let volume = measure::volume(&b);
    let want = helix_tube_volume(1.5, 3.0, 2.0, 2.0, 0.1);
    assert!((volume - want).abs() <= 2e-3 * want, "{volume} {want}");
}
