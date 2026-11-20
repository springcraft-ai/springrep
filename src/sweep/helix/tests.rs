use std::f64::consts::{PI, TAU};

use super::*;
use crate::measure;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// The xz plane through the point 10 mm out along x, its normal along y: a meridian of the z axis.
fn meridian() -> Frame {
    Frame::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Y, -Vec3::X).unwrap()
}

fn about_z(pitch: f64, sign: f64) -> Helix {
    Helix {
        origin: Vec3::ZERO,
        axis: Vec3::Z,
        rise: pitch / TAU,
        sign,
    }
}

// A meridian section screwed about an axis sweeps the volume it would turned the same angle
// (Pappus): area · angle · centroid radius. A 2 mm circle 10 mm out, a quarter turn rising 2 mm:
// π · π/2 · 10, from 1 mm below the start plane to 1 mm over the rise.
#[test]
fn a_circle_screwed_a_quarter_turn_holds_pappus_volume() {
    let p = Profile::circle(Vec2::ZERO, 2.0);
    let b = helix(&p, &meridian(), &about_z(8.0, 1.0), PI / 2.0, "coil").unwrap();
    assert_eq!(b.face_count(), 3);
    assert!(rel(measure::volume(&b), 5.0 * PI * PI) < 1e-9);
    let bb = measure::bbox(&b);
    assert!((bb.min.z + 1.0).abs() < 1e-6 && (bb.max.z - 3.0).abs() < 1e-6);
    let names = b.name_counts();
    assert_eq!(names["coil.start"], 1);
    assert_eq!(names["coil.end"], 1);
}

// A 2 × 2 mm square, its centre 10 mm out, a quarter turn left-handed: 4 · π/2 · 10. Each side
// is its own face.
#[test]
fn a_profile_drawn_on_a_frame_at_the_axis_is_screwed() {
    // The circle above, its frame's origin on the axis and the circle drawn 10 mm out along the
    // frame's x (world -x, so at world +x 10).
    let at_axis = Frame::new(Vec3::ZERO, Vec3::Y, -Vec3::X).unwrap();
    let p = Profile::circle(Vec2::new(-10.0, 0.0), 2.0);
    let b = helix(&p, &at_axis, &about_z(8.0, 1.0), PI / 2.0, "coil").unwrap();
    assert!(
        rel(measure::volume(&b), 5.0 * PI * PI) < 1e-9,
        "{}",
        measure::volume(&b)
    );
}

#[test]
fn a_square_screwed_left_handed_keeps_its_sides() {
    let p = Profile::rect(Vec2::ZERO, 2.0, 2.0);
    let b = helix(&p, &meridian(), &about_z(8.0, -1.0), PI / 2.0, "thread").unwrap();
    assert_eq!(b.face_count(), 6);
    assert!(rel(measure::volume(&b), 20.0 * PI) < 1e-9);
}

#[test]
fn a_profile_on_the_axis_is_refused() {
    let p = Profile::circle(Vec2::new(10.0, 0.0), 2.0);
    let e = helix(&p, &meridian(), &about_z(5.0, 1.0), TAU, "coil");
    assert!(matches!(e, Err(Error::InvalidInput(_))));
}

// Each side face's orientation is read at the start turn, so it holds at any turn count: half a
// turn, a whole one and three used to be refused as a face running the wrong way.
#[test]
fn whole_and_half_turns_keep_their_faces_outward() {
    let p = Profile::circle(Vec2::ZERO, 2.0);
    for turns in [0.5, 1.0, 3.0] {
        let b = helix(&p, &meridian(), &about_z(5.0, 1.0), turns * TAU, "coil").unwrap();
        let want = PI * turns * TAU * 10.0;
        assert!(rel(measure::volume(&b), want) < 1e-9, "{turns}");
    }
}

// A 2 mm wire wound at a 1.5 mm pitch runs into its own last turn; at a 2 mm pitch it touches it
// along a line. Either would be a solid that meets itself.
#[test]
fn coils_that_meet_are_refused() {
    let p = Profile::circle(Vec2::ZERO, 2.0);
    for pitch in [1.5, 2.0, 2.0 + 1e-7] {
        let e = helix(&p, &meridian(), &about_z(pitch, 1.0), 2.0 * TAU, "coil");
        assert!(matches!(e, Err(Error::InvalidInput(_))), "{pitch}");
    }
    // Under a whole turn the copy a pitch along never comes round.
    assert!(helix(&p, &meridian(), &about_z(1.5, 1.0), 0.9 * TAU, "coil").is_ok());
    let b = helix(&p, &meridian(), &about_z(2.1, 1.0), 2.0 * TAU, "coil").unwrap();
    assert!(rel(measure::volume(&b), PI * 2.0 * TAU * 10.0) < 1e-9);
}

// A 2 mm wire 0.8 mm from the axis crosses it between the profile's sample points.
#[test]
fn a_profile_across_the_axis_is_refused() {
    let frame = Frame::new(Vec3::new(0.8, 0.0, 0.0), Vec3::Y, -Vec3::X).unwrap();
    let p = Profile::circle(Vec2::ZERO, 2.0);
    let e = helix(&p, &frame, &about_z(5.0, 1.0), PI, "coil");
    assert!(matches!(e, Err(Error::InvalidInput(_))));
}

// A 2 × 3 rectangle about the z axis and a disc off it, each risen 5 while turning 0.6: a twisted
// extrusion keeps its section's area at every level, so its volume is the area times the rise.
#[test]
fn a_twisted_extrusion_holds_its_section_s_area_all_the_way_up() {
    let rise = 5.0;
    let twist = 0.6;
    let motion = about_z(rise / twist * TAU, 1.0);
    let rect = Profile::rect(Vec2::ZERO, 2.0, 3.0);
    let b = helix(&rect, &Frame::WORLD, &motion, twist, "tw").unwrap();
    assert!(check::check(&b).is_empty());
    assert!(rel(measure::volume(&b), 6.0 * rise) < 1e-7);
    let disc = Profile::circle(Vec2::new(3.0, 1.0), 2.0);
    let b = helix(
        &disc,
        &Frame::WORLD,
        &about_z(rise / twist * TAU, -1.0),
        twist,
        "tw",
    )
    .unwrap();
    assert!(check::check(&b).is_empty());
    assert!(rel(measure::volume(&b), PI * rise) < 1e-7);
}

// A screw is rigid, so a region it carries across its own plane at the rate `v·n` sweeps
// `v·n dA dt`; `v·n` is affine over the plane, so the volume is the angle times the area times
// the rate at the centroid. A disc 10 out on a plane leaning toward the axis and a quarter turn:
// its rate at the centre, v = (0, ±10, ρ) on n = (1, 1, 1)/√3, is |ρ ± 10|/√3.
#[test]
fn an_oblique_disc_screwed_sweeps_its_area_times_its_crossing_rate() {
    let rho = 8.0 / TAU;
    let n = Vec3::new(1.0, 1.0, 1.0) * (1.0 / 3f64.sqrt());
    let frame = Frame::from_axis(Vec3::new(10.0, 0.0, 0.0), n).unwrap();
    let p = Profile::circle(Vec2::ZERO, 2.0);
    let b = helix(&p, &frame, &about_z(8.0, 1.0), PI / 2.0, "coil").unwrap();
    assert!(check::check(&b).is_empty());
    let want = PI / 2.0 * PI * (10.0 + rho) / 3f64.sqrt();
    assert!(
        rel(measure::volume(&b), want) < 1e-7,
        "{}",
        measure::volume(&b)
    );
    let b = helix(&p, &frame, &about_z(8.0, -1.0), PI / 2.0, "coil").unwrap();
    assert!(check::check(&b).is_empty());
    let want = PI / 2.0 * PI * (10.0 - rho) / 3f64.sqrt();
    assert!(
        rel(measure::volume(&b), want) < 1e-7,
        "{}",
        measure::volume(&b)
    );
}

// The triangle (5, 1, 1), (5, −1, −1), (3, 1, −1), area 2√3 on n = (1, 1, −1)/√3, its centroid
// (13/3, 1/3, −1/3) crossing at (4 − ρ)/√3, screwed one and a half turns rising 5 a turn. Each
// turn clears the last: the triangle is 2 high.
#[test]
fn an_oblique_triangle_screwed_one_and_a_half_turns_sweeps_its_area_times_its_crossing_rate() {
    let rho = 5.0 / TAU;
    let (a, b, c) = (
        Vec3::new(5.0, 1.0, 1.0),
        Vec3::new(5.0, -1.0, -1.0),
        Vec3::new(3.0, 1.0, -1.0),
    );
    let frame = Frame::new(a, (b - a).cross(c - a), b - a).unwrap();
    let local = |p: Vec3| {
        let l = frame.to_local(p);
        Vec2::new(l.x, l.y)
    };
    let p = Profile::polygon(&[local(a), local(b), local(c)], &["ab", "bc", "ca"]);
    let s = helix(&p, &frame, &about_z(5.0, 1.0), 3.0 * PI, "tri").unwrap();
    assert!(check::check(&s).is_empty());
    let want = 3.0 * PI * 2.0 * (4.0 - rho);
    assert!(
        rel(measure::volume(&s), want) < 1e-7,
        "{}",
        measure::volume(&s)
    );
}

// A disc of radius 2 square to the helix of its centre, 1.5 from the axis: the rate it crosses
// its plane, |v| at the centre, falls by 1.5/|v| a millimetre toward the axis, so it is 0 on the
// line |v|²/1.5 = (1.5² + ρ²)/1.5 ≈ 1.65 from the centre, inside the disc: the sweep folds.
#[test]
fn an_oblique_profile_the_motion_runs_along_is_refused() {
    let screw = about_z(3.0, 1.0);
    let c = Vec3::new(1.5, 0.0, 0.0);
    let frame = Frame::from_axis(c, screw.d3(c, 0.0)[1]).unwrap();
    let p = Profile::circle(Vec2::ZERO, 4.0);
    let e = helix(&p, &frame, &screw, TAU, "tube");
    assert!(matches!(e, Err(Error::InvalidInput(_))), "{e:?}");
}

// A disc 2 across, 10 out and nearly square to the helix of its centre, wound three turns: at a
// 3 mm pitch each turn clears the last, and at 1.5 mm the next turn comes back across it.
#[test]
fn an_oblique_profile_its_next_turn_comes_back_across_is_refused() {
    let p = Profile::circle(Vec2::ZERO, 2.0);
    let c = Vec3::new(10.0, 0.0, 0.0);
    let n = Vec3::new(0.0, 1.0, 0.05);
    let frame = Frame::from_axis(c, n).unwrap();
    let b = helix(&p, &frame, &about_z(3.0, 1.0), 3.0 * TAU, "coil").unwrap();
    assert!(check::check(&b).is_empty());
    let rate = (10.0 + 0.05 * 3.0 / TAU) / n.norm();
    assert!(rel(measure::volume(&b), 3.0 * TAU * PI * rate) < 1e-7);
    let e = helix(&p, &frame, &about_z(1.5, 1.0), 3.0 * TAU, "coil");
    assert!(matches!(e, Err(Error::Unsupported(_))), "{e:?}");
}

/// A metric thread's ridge on its shank where it runs out, as a screw's: the shank a cylinder at
/// the major radius, a cone down to the minor over the runout, the minor cylinder; the ridge the
/// ISO basic profile (crest flat P/8 on the major radius, 60° flanks) reaching 0.1P below the
/// minor, cut square at the cylinder's edge with the cone and 4 pitches past the cone. So the
/// crest lies on the plain cylinder and runs across that edge. Each height's slice of the overlap
/// covers the fraction w(ρ)/P of each circle of the ridge inside the shank, w the profile's width
/// at ρ, so the overlap is ∫ 2π/P ∫ ρ w(ρ) dρ dz.
fn runout() -> (Brep, Brep, f64) {
    use crate::profile::{NamedSegment, ProfileLoop, Segment};
    let (p, major) = (1.25, 4.0);
    let depth = 5.0 / 8.0 * 3f64.sqrt() / 2.0 * p;
    let minor = major - depth;
    let base = minor - 0.1 * p;
    let flank = 2.0 / 3f64.sqrt();
    let width = |rho: f64| p / 8.0 + (major - rho) * flank;
    let (top, runout) = (6.0 * p, 2.5 * p);
    let len = top + runout + 6.0 * p;
    let (lo, hi) = (top, top + runout + 4.0 * p);
    let sketch = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let poly = |pts: &[[f64; 2]], name: &str| {
        let n = pts.len();
        Profile {
            loops: vec![ProfileLoop {
                segments: (0..n)
                    .map(|i| {
                        NamedSegment::new(
                            Segment::Line {
                                a: Vec2::new(pts[i][0], pts[i][1]),
                                b: Vec2::new(pts[(i + 1) % n][0], pts[(i + 1) % n][1]),
                            },
                            format!("{name}{i}"),
                        )
                    })
                    .collect(),
            }],
        }
    };
    let shank_pts = [
        [0.0, 0.0],
        [major, 0.0],
        [major, top],
        [minor, top + runout],
        [minor, len],
        [0.0, len],
    ];
    let shank = crate::sweep::revolve(
        &poly(&shank_pts, "shank"),
        &sketch,
        Vec3::ZERO,
        Vec3::Z,
        TAU,
        "shank",
    )
    .unwrap();
    let from = lo - p;
    let (wb, wc) = (width(base), width(major));
    let ridge_profile = poly(
        &[
            [base, from - wb / 2.0],
            [major, from - wc / 2.0],
            [major, from + wc / 2.0],
            [base, from + wb / 2.0],
        ],
        "ridge",
    );
    let turns = (hi + p - from) / p;
    let ridge = helix(
        &ridge_profile,
        &sketch,
        &about_z(p, 1.0),
        TAU * turns,
        "ridge",
    )
    .unwrap();
    let band = crate::sweep::extrude(
        &Profile::circle(Vec2::ZERO, 2.0 * (major + 1.0)),
        &Frame::new(Vec3::new(0.0, 0.0, lo), Vec3::Z, Vec3::X).unwrap(),
        hi - lo,
        "band",
    )
    .unwrap();
    let ridge = crate::boolean::intersection(&ridge, &band).unwrap();
    // ∫ 2π/P ρ (α + βρ) dρ from the base to the shank's radius, by height; Gauss–Legendre on each
    // piece is exact for the cubic it is.
    let slice = |r: f64| {
        let (alpha, beta) = (p / 8.0 + major * flank, -flank);
        let f = |x: f64| alpha * x * x / 2.0 + beta * x.powi(3) / 3.0;
        TAU / p * (f(r) - f(base))
    };
    let core = |z: f64| major - (major - minor) * ((z - top) / runout).clamp(0.0, 1.0);
    let gauss = |a: f64, b: f64| {
        let (m, h) = ((a + b) / 2.0, (b - a) / 2.0);
        let (x, w) = ((3.0f64 / 5.0).sqrt(), [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]);
        h * [-x, 0.0, x]
            .iter()
            .zip(w)
            .map(|(t, w)| w * slice(core(m + h * t)))
            .sum::<f64>()
    };
    let overlap = gauss(lo, top + runout) + gauss(top + runout, hi);
    (shank, ridge, overlap)
}

#[test]
fn a_thread_running_out_onto_its_shank_meets_it_in_its_overlap() {
    let (shank, ridge, overlap) = runout();
    let both = crate::boolean::intersection(&shank, &ridge).unwrap();
    assert!(check::check(&both).is_empty());
    assert!(
        rel(measure::volume(&both), overlap) < 1e-7,
        "{}",
        measure::volume(&both)
    );
}

// Where the ridge is cut, at the circle the shank's cylinder meets its cone on, the crest runs on
// from the cylinder as one surface with it and the cut's plane lies inside the shank: of the four
// faces round that arc the join keeps the two cylinders, as one face.
#[test]
fn a_thread_running_out_onto_its_shank_joins_it() {
    let (shank, ridge, overlap) = runout();
    let joined = crate::boolean::union(&shank, &ridge).unwrap();
    assert!(check::check(&joined).is_empty());
    let want = measure::volume(&shank) + measure::volume(&ridge) - overlap;
    assert!(
        rel(measure::volume(&joined), want) < 1e-7,
        "{}",
        measure::volume(&joined)
    );
}

// A heat-set insert's lower knurl on its sleeve: the sleeve turned, 2.05 in radius between a
// pilot and a groove; the knurl 18 flat-topped teeth out to 2.3 on arcs of the sleeve's own
// circle, risen 1.55 from z = 1 while turning 30° at the sleeve's radius, reaching 0.15 into it.
// The arcs sweep the sleeve's cylinder itself, and the join adds the teeth: their area by 1.55.
#[test]
fn a_knurl_twisted_round_its_sleeve_joins_it() {
    use crate::profile::{NamedSegment, ProfileLoop, Segment};
    let (rv, peak, n) = (2.05, 2.3, 18usize);
    let named = |segment: Segment, name: String| NamedSegment::new(segment, name);
    let poly = |pts: &[[f64; 2]]| ProfileLoop {
        segments: (0..pts.len())
            .map(|i| {
                let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                named(
                    Segment::Line {
                        a: Vec2::new(a[0], a[1]),
                        b: Vec2::new(b[0], b[1]),
                    },
                    format!("s{i}"),
                )
            })
            .collect(),
    };
    let sketch = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
    let sleeve = crate::sweep::revolve(
        &Profile {
            loops: vec![poly(&[
                [1.5, 0.0],
                [1.8, 0.0],
                [1.95, 0.15],
                [1.95, 1.0],
                [rv, 1.0],
                [rv, 2.55],
                [1.9, 2.55],
                [1.9, 3.15],
                [rv, 3.15],
                [rv, 5.55],
                [1.9, 5.7],
                [1.5, 5.7],
            ])],
        },
        &sketch,
        Vec3::ZERO,
        Vec3::Z,
        TAU,
        "sleeve",
    )
    .unwrap();
    let step = TAU / n as f64;
    let at = |r: f64, a: f64| Vec2::new(r * a.cos(), r * a.sin());
    let mut outline = Vec::new();
    let mut teeth = 0.0;
    for k in 0..n {
        let a = step * k as f64;
        outline.push(named(
            Segment::Arc {
                center: Vec2::ZERO,
                radius: rv,
                start: a,
                sweep: step / 2.0,
            },
            format!("arc{k}"),
        ));
        let pts = [
            at(rv, a + 0.5 * step),
            at(peak, a + 0.625 * step),
            at(peak, a + 0.875 * step),
            at(rv, a + step),
        ];
        for (i, w) in pts.windows(2).enumerate() {
            outline.push(named(
                Segment::Line { a: w[0], b: w[1] },
                format!("t{k}{i}"),
            ));
        }
        // The tooth beyond the sleeve's circle: its fan from the axis less the sector under it.
        let fan: f64 = pts.windows(2).map(|w| 0.5 * w[0].cross(w[1])).sum();
        teeth += fan - 0.5 * rv * rv * (step / 2.0);
    }
    let hole = ProfileLoop {
        segments: vec![named(
            Segment::Circle {
                center: Vec2::ZERO,
                radius: rv - 0.15,
                ccw: false,
            },
            "hole".into(),
        )],
    };
    let (h, twist) = (1.55, 1.55 * (PI / 6.0).tan() / rv);
    let band = helix(
        &Profile {
            loops: vec![ProfileLoop { segments: outline }, hole],
        },
        &Frame::new(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, Vec3::X).unwrap(),
        &about_z(h / twist * TAU, 1.0),
        twist,
        "knurl",
    )
    .unwrap();
    let joined = crate::boolean::union(&sleeve, &band).unwrap();
    assert!(check::check(&joined).is_empty());
    let want = measure::volume(&sleeve) + teeth * h;
    assert!(
        rel(measure::volume(&joined), want) < 1e-7,
        "{} against {want}",
        measure::volume(&joined)
    );
}
