use std::f64::consts::{PI, TAU};

use super::{Mesh, MeshOptions, report, tessellate};
use crate::{
    boolean::difference,
    math::{Frame, Vec2, Vec3},
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
    topo::Brep,
};

/// The largest distance from a triangle's centroid or edge middles to its face's surface.
fn deviation(b: &Brep, m: &Mesh) -> f64 {
    let mut worst: f64 = 0.0;
    for (f, range) in &m.faces {
        let surf = &b.face(*f).surface;
        for t in &m.triangles[range.clone()] {
            let p = t.map(|i| m.positions[i as usize]);
            for q in [
                (p[0] + p[1] + p[2]) / 3.0,
                (p[0] + p[1]) * 0.5,
                (p[1] + p[2]) * 0.5,
                (p[2] + p[0]) * 0.5,
            ] {
                let uv = surf.project(q);
                worst = worst.max(surf.eval(uv.x, uv.y).dist(q));
            }
        }
    }
    worst
}

/// Meshes `b`, checks every solid is watertight and the mesh near its surfaces, and returns the
/// triangle count. A triangle along a boundary may stray by its boundary segment's own sag on top
/// of the deflection, so the bound is twice it.
fn mesh_ok(b: &Brep, opt: &MeshOptions) -> usize {
    let m = tessellate(b, opt).unwrap();
    for s in b.solid_ids() {
        let r = report(b, &m, s);
        assert!(r.watertight(), "{r:?}");
        assert_eq!(r.degenerate_triangles, 0);
    }
    let d = deviation(b, &m);
    assert!(d <= 2.0 * opt.linear, "deviation {d}");
    m.triangles.len()
}

fn ball(r: f64) -> Brep {
    ball_in(
        r,
        Frame::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap(),
    )
}

/// A ball about z, its half disc drawn on `frame`, whose `x` sets where `u` starts.
fn ball_in(r: f64, frame: Frame) -> Brep {
    let seg = |segment: Segment, name: &str| NamedSegment::new(segment, name);
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                seg(
                    Segment::Arc {
                        center: Vec2::ZERO,
                        radius: r,
                        start: -PI / 2.0,
                        sweep: PI,
                    },
                    "surface",
                ),
                seg(
                    Segment::Line {
                        a: Vec2::new(0.0, r),
                        b: Vec2::new(0.0, -r),
                    },
                    "axis",
                ),
            ],
        }],
    };
    revolve(&p, &frame, Vec3::ZERO, Vec3::Z, TAU, "ball").unwrap()
}

#[test]
fn a_plate_with_a_grid_of_holes_meshes_its_caps() {
    let holes = (0..7).flat_map(|i| {
        (0..7).flat_map(move |j| {
            let c = Vec2::new(-30.0 + 10.0 * i as f64, -30.0 + 10.0 * j as f64);
            Profile::circle(c, 6.0).loops
        })
    });
    let p = Profile::rect(Vec2::ZERO, 80.0, 80.0).with_holes(holes);
    let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
    let b = extrude(&p, &frame, 5.0, "plate").unwrap();
    let n = mesh_ok(
        &b,
        &MeshOptions {
            linear: 0.05,
            angular: 0.5,
        },
    );
    // About one triangle per rim sample on each cap and two on the wall: no points inside.
    assert!(n < 4000, "{n} triangles");
}

#[test]
fn a_torus_meshes_near_its_surface_without_excess() {
    let frame = Frame::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap();
    let p = Profile::circle(Vec2::new(10.0, 0.0), 4.0);
    let b = revolve(&p, &frame, Vec3::ZERO, Vec3::Z, TAU, "tor").unwrap();
    let n = mesh_ok(
        &b,
        &MeshOptions {
            linear: 0.05,
            angular: 0.5,
        },
    );
    assert!(n < 3000, "{n} triangles");
}

#[test]
fn a_ball_cut_beside_its_pole_meshes() {
    let s = ball(10.0);
    let frame = Frame::from_axis(Vec3::new(1.5, 0.0, 5.0), Vec3::Z).unwrap();
    let rod = extrude(&Profile::circle(Vec2::ZERO, 2.0), &frame, 10.0, "rod").unwrap();
    let d = difference(&s, &rod).unwrap();
    let n = mesh_ok(
        &d,
        &MeshOptions {
            linear: 0.1,
            angular: 0.5,
        },
    );
    assert!(n < 2000, "{n} triangles");
}

#[test]
fn a_cut_passing_a_hair_from_both_poles_goes_round_them_its_own_way() {
    // A ball r 2 centred 1e-6 inside the surface of a rod r 3: their meeting passes 1e-6 from both
    // poles, so the ball's u turns by half a period between samples 0.2 mm apart, and the nearest
    // image took it round each pole the wrong way.
    let frame = Frame::from_axis(Vec3::new(0.0, 0.0, -15.0), Vec3::Z).unwrap();
    let rod = extrude(&Profile::circle(Vec2::ZERO, 6.0), &frame, 30.0, "rod").unwrap();
    let mut s = ball_in(2.0, Frame::new(Vec3::ZERO, Vec3::Y, -Vec3::X).unwrap());
    s.transform(&crate::math::Transform::translation(Vec3::new(
        3.0 - 1e-6,
        0.0,
        1.0,
    )));
    let u = crate::boolean::union(&rod, &s).unwrap();
    let opt = MeshOptions {
        linear: 0.2,
        angular: 0.5,
    };
    mesh_ok(&u, &opt);
    let m = tessellate(&u, &opt).unwrap();
    let face = |kind: &str| {
        u.face_ids()
            .find(|&f| u.face(f).surface.kind() == kind)
            .unwrap()
    };
    let f = face("sphere");
    let (_, range) = m.faces.iter().find(|(g, _)| *g == f).unwrap();
    let area: f64 = m.triangles[range.clone()]
        .iter()
        .map(|t| {
            let p = t.map(|i| m.positions[i as usize]);
            0.5 * (p[1] - p[0]).cross(p[2] - p[0]).norm()
        })
        .sum();
    // The ball's face outside the rod is 29.38 mm²; the wrong way round it meshed as 20.31.
    let exact = crate::measure::face_area(&u, f);
    assert!((area - exact).abs() < 0.05 * exact, "{area} vs {exact}");
}

#[test]
fn an_edge_ending_beside_a_knot_keeps_its_samples_apart() {
    use super::edge_params;
    use crate::geom::{BSpline, Curve};
    let ctrl = [[0.0, 0.0], [1.0, 2.0], [2.0, 0.0], [3.0, 2.0], [4.0, 0.0]]
        .map(|[x, y]| Vec3::new(x, y, 0.0))
        .to_vec();
    let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let c = Curve::BSpline(BSpline::new(3, knots, ctrl).unwrap());
    let opt = MeshOptions {
        linear: 0.01,
        angular: 0.1,
    };
    let ts = edge_params(&c, 0.1, 0.5 + 1e-15, &opt);
    for w in ts.windows(2) {
        assert!(
            c.eval(w[0]).dist(c.eval(w[1])) > crate::tol::LINEAR,
            "{ts:?}"
        );
    }
}

/// A ball 8 µm across meshed to 0.01 mm: its whole surface lies within the deflection of a few
/// triangles, but it still gets them, every face meshed and closed.
#[test]
fn a_ball_narrower_than_the_deflection_still_meshes() {
    let opt = MeshOptions {
        linear: 0.01,
        angular: 0.2,
    };
    assert!(mesh_ok(&ball(0.004), &opt) > 0);
}

#[test]
fn a_circle_narrower_than_the_deflection_keeps_its_samples_apart() {
    use super::edge_params;
    use crate::geom::{Circle, Curve};
    let c = Curve::Circle(Circle {
        frame: Frame::WORLD,
        radius: 1e-5,
    });
    let opt = MeshOptions {
        linear: 0.001,
        angular: 0.01,
    };
    let ts = edge_params(&c, 0.0, TAU, &opt);
    assert!(ts.len() <= 5, "{ts:?}");
    let chord = c.eval(ts[0]).dist(c.eval(ts[1]));
    assert!(chord > 100.0 * crate::tol::LINEAR, "{chord}");
}

#[test]
fn images_of_one_point_a_rounding_apart_become_one() {
    // A crescent's loop passes its pinch point twice, once per circle, 4e-13 apart; a seam's two
    // images of one point lie a period apart.
    let mut pts = vec![
        (Vec2::new(4.075, 0.0), 7),
        (Vec2::new(0.0, 4.075), 8),
        (Vec2::new(4.075, -4e-13), 7),
        (Vec2::new(4.075 - TAU, 0.0), 7),
    ];
    super::snap_repeats(&mut pts);
    assert_eq!(pts[2].0, pts[0].0);
    assert_eq!(pts[3].0, Vec2::new(4.075 - TAU, 0.0));
    assert_eq!(pts[1].0, Vec2::new(0.0, 4.075));
}

#[test]
fn a_triangle_folded_onto_itself_beside_a_pole_is_dropped() {
    // An imported part's patch closes in a pole: point 2 is joined to the pole 0 on both sides of the
    // chord 1-2, so 0 1 2 and 1 0 2 cover one triangle both ways and 0-1 is used four times.
    let tris = vec![[1, 0, 3], [4, 0, 1], [0, 1, 2], [1, 0, 2], [0, 2, 5]];
    assert_eq!(super::unfold(tris), vec![[1, 0, 3], [4, 0, 1], [0, 2, 5]]);
    // A repeated triangle facing one way is not a fold.
    assert_eq!(super::unfold(vec![[0, 1, 2], [1, 2, 0]]).len(), 2);
}

/// A fillet as wide as the cylinder leaves a top cap 0.001 mm across: the round is a spindle
/// torus whose apex sits a hair above the face, where its normal flips.
#[test]
fn a_fillet_ending_a_hair_from_the_axis_meshes() {
    let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
    let c = extrude(&Profile::circle(Vec2::ZERO, 20.0), &frame, 20.0, "c").unwrap();
    let b = crate::blend::fillet(&c, &[("c.end", "c.side")], 9.999, "f").unwrap();
    let coarse = MeshOptions {
        linear: 0.2,
        angular: 0.5,
    };
    assert!(mesh_ok(&b, &coarse) < 1000);
    let fine = MeshOptions {
        linear: 0.01,
        angular: 0.1,
    };
    assert!(mesh_ok(&b, &fine) < 10_000);
}

/// Each side of a loft from a circle to a square has the same normal at both ends of its straight
/// edge but turns between them, so that edge must be sampled for its triangles to follow; before
/// it was, refinement crowded points onto the edge and left zero-area triangles.
#[test]
fn a_loft_from_a_circle_to_a_square_meshes_without_slivers() {
    let at = |z: f64| Frame::WORLD.translated(Vec3::Z * z);
    let b = crate::loft::loft(
        &[
            crate::loft::Section {
                profile: Profile::circle(Vec2::ZERO, 20.0),
                frame: at(0.0),
            },
            crate::loft::Section {
                profile: Profile::rect(Vec2::ZERO, 20.0, 20.0),
                frame: at(30.0),
            },
        ],
        false,
        "l",
    )
    .unwrap();
    let m = tessellate(
        &b,
        &MeshOptions {
            linear: 1.0,
            angular: 0.05,
        },
    )
    .unwrap();
    let s = b.solid_ids().next().unwrap();
    let r = report(&b, &m, s);
    assert!(r.watertight(), "{r:?}");
    assert_eq!(r.degenerate_triangles, 0);
    // Projecting every triangle onto the spline sides is too slow here; the volume stands in.
    let exact = crate::measure::volume(&b);
    assert!(
        (r.volume - exact).abs() < 1e-3 * exact,
        "{} {exact}",
        r.volume
    );
}

#[test]
fn a_coarse_chord_never_leaves_another_loop_on_its_wrong_side() {
    // At 20 mm and 3 rad the Ø100 rim is a square or a triangle whose sides pass inside the bolt
    // holes at r 40 or leave them outside the face.
    let mut disc = extrude(
        &Profile::circle(Vec2::ZERO, 100.0),
        &Frame::WORLD,
        10.0,
        "d",
    )
    .unwrap();
    for k in 0..8 {
        let a = k as f64 * TAU / 8.0;
        let at = Frame::from_axis(Vec3::new(40.0 * a.cos(), 40.0 * a.sin(), 0.0), Vec3::Z).unwrap();
        let bolt = extrude(&Profile::circle(Vec2::ZERO, 10.0), &at, 1000.0, "b").unwrap();
        disc = difference(&disc, &bolt).unwrap();
    }
    for (linear, angular) in [(20.0, 3.0), (20.0, 2.0), (50.0, 3.1)] {
        let m = tessellate(&disc, &MeshOptions { linear, angular }).unwrap();
        let s = disc.solid_ids().next().unwrap();
        assert!(report(&disc, &m, s).watertight(), "{linear} {angular}");
    }
    for (linear, angular) in [(0.0, 0.1), (0.1, -1.0), (f64::NAN, 0.1)] {
        let r = tessellate(&disc, &MeshOptions { linear, angular });
        assert!(matches!(r, Err(crate::Error::InvalidInput(_))), "{r:?}");
    }
}

#[test]
fn two_short_arcs_between_one_pair_of_vertices_draw_different_segments() {
    // A slab with a hole along y, unioned with itself turned 17°: on each end face the two holes'
    // circles cross 0.51 mm apart, and the arcs between those points belong to different faces.
    let frame = Frame::new(Vec3::new(0.0, 0.0, -4.0), Vec3::Z, Vec3::X).unwrap();
    let slab = extrude(&Profile::rect(Vec2::ZERO, 40.0, 40.0), &frame, 8.0, "k").unwrap();
    let axis = Frame::from_axis(Vec3::new(0.0, -20.0, 0.0), Vec3::Y).unwrap();
    let hole = extrude(&Profile::circle(Vec2::ZERO, 6.0), &axis, 1000.0, "h").unwrap();
    let a = difference(&slab, &hole).unwrap();
    let mut b = a.clone();
    b.transform(&crate::math::Transform::rotation(
        Vec3::ZERO,
        Vec3::Z,
        17f64.to_radians(),
    ));
    let u = crate::boolean::union(&a, &b).unwrap();
    mesh_ok(
        &u,
        &MeshOptions {
            linear: 0.2,
            angular: 0.5,
        },
    );
}

#[test]
fn a_round_a_micrometre_wide_gives_no_flat_triangles() {
    // A pocket whose corners are rounded at 1e-3 mm, as small imported parts often are: scaled
    // for the deflection, a round's triangles are well shaped, but four had a corner within 1e-7 mm
    // of the opposite side in space.
    let (wx, wy, r) = (1.5, 7.95, 1e-3);
    let seg = |segment: Segment| NamedSegment::new(segment, "s");
    let line = |a: (f64, f64), b: (f64, f64)| {
        seg(Segment::Line {
            a: Vec2::new(a.0, a.1),
            b: Vec2::new(b.0, b.1),
        })
    };
    let arc = |c: (f64, f64), start: f64| {
        seg(Segment::Arc {
            center: Vec2::new(c.0, c.1),
            radius: r,
            start,
            sweep: PI / 2.0,
        })
    };
    let (kx, ky) = (wx - r, wy - r);
    let hole = ProfileLoop {
        segments: vec![
            line((-kx, -wy), (kx, -wy)),
            arc((kx, -ky), -PI / 2.0),
            line((wx, -ky), (wx, ky)),
            arc((kx, ky), 0.0),
            line((kx, wy), (-kx, wy)),
            arc((-kx, ky), PI / 2.0),
            line((-wx, ky), (-wx, -ky)),
            arc((-kx, -ky), PI),
        ],
    };
    let p = Profile::rect(Vec2::ZERO, 2.0 * wx + 2.0, 2.0 * wy + 2.0).with_holes([hole]);
    let frame = Frame::new(Vec3::new(33.2875, -10.0, 1.716), Vec3::Z, Vec3::X).unwrap();
    let b = extrude(&p, &frame, 3.8, "round").unwrap();
    mesh_ok(
        &b,
        &MeshOptions {
            linear: 0.005,
            angular: 0.1,
        },
    );
}

fn cone(bottom: f64, top: f64, height: f64) -> Brep {
    let pts = [
        Vec2::new(0.0, 0.0),
        Vec2::new(bottom, 0.0),
        Vec2::new(top, height),
        Vec2::new(0.0, height),
    ];
    let names = ["bottom", "side", "top", "axis"];
    let segments = (0..4)
        .filter(|&i| pts[i] != pts[(i + 1) % 4])
        .map(|i| {
            NamedSegment::new(
                Segment::Line {
                    a: pts[i],
                    b: pts[(i + 1) % 4],
                },
                names[i],
            )
        })
        .collect();
    let p = Profile {
        loops: vec![ProfileLoop { segments }],
    };
    let frame = Frame::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap();
    revolve(&p, &frame, Vec3::ZERO, Vec3::Z, TAU, "cone").unwrap()
}

/// A cone's side is ruled, so one triangle from the apex to each chord of the rim follows it
/// exactly, whichever end of the axis the apex is at.
#[test]
fn a_pointed_cone_meshes_as_a_fan_from_either_end() {
    let opt = MeshOptions {
        linear: 0.01,
        angular: 0.2,
    };
    let up = mesh_ok(&cone(2.6, 0.0, 7.6), &opt);
    let down = mesh_ok(&cone(0.0, 2.6, 7.6), &opt);
    assert!(up < 100, "apex on top: {up} triangles");
    assert!(down < 100, "apex at the bottom: {down} triangles");
}

/// A wire thinner than the deflection is within it of its own axis, so a few chords round it
/// follow it as well as any number: its turn alone never splits a step no longer than the
/// deflection, on its side as on its rims.
#[test]
fn a_wire_thinner_than_the_deflection_takes_few_triangles() {
    let opt = MeshOptions {
        linear: 0.008,
        angular: 0.2,
    };
    for r in [0.001, 0.004] {
        let n = mesh_ok(&cone(r, r, 8.0), &opt);
        assert!(n < 100, "radius {r}: {n} triangles");
    }
    // A rod the deflection resolves still turns no more than the angle between its triangles.
    let rod = mesh_ok(&cone(1.0, 1.0, 8.0), &opt);
    assert!(rod >= 100, "{rod} triangles");
}

#[test]
fn a_cap_whose_corners_share_a_latitude_meshes() {
    // A box less three balls, drawn at 0.02 mm to 0.05 mm: one ball's cap is a latitude arc closed by a crossing
    // arc drawn as one chord, which in the ball's parameter plane runs through the arc's middle.
    use crate::{boolean::union, primitive};
    let ball = |c: [f64; 3], r: f64| {
        let at = Frame::new(Vec3::new(c[0], c[1], c[2]), Vec3::Z, Vec3::X).unwrap();
        primitive::sphere(&at, r, "s").unwrap()
    };
    let balls = union(
        &union(
            &ball([0.0, 0.0, 0.0], 4.582532090371927),
            &ball(
                [2.052251643929374, 0.47303714450016393, -4.798087593307755],
                4.861338367991554,
            ),
        )
        .unwrap(),
        &ball(
            [-0.4289764855758635, 2.0588351497787443, -4.299588486273758],
            2.3390042555267674,
        ),
    )
    .unwrap();
    let z = -0.5969227145952956;
    let slab = primitive::block(
        &Frame::new(Vec3::new(-20.0, -20.0, z), Vec3::Z, Vec3::X).unwrap(),
        Vec3::new(40.0, 40.0, 20.0 - z),
        "b",
    )
    .unwrap();
    let cut = difference(&slab, &balls).unwrap();
    for linear in [0.05, 0.03, 0.02] {
        mesh_ok(
            &cut,
            &MeshOptions {
                linear,
                angular: super::display::ANGLE,
            },
        );
    }
    mesh_ok(&cut, &MeshOptions::within(0.03));
}

#[test]
fn par_map_gives_each_result_in_its_place() {
    // However many threads take the items, and whichever takes each.
    let want: Vec<usize> = (0..1000).map(|i| 3 * i + 1).collect();
    for threads in [1, 3, usize::MAX] {
        assert_eq!(super::par_map(threads, 1000, |i| 3 * i + 1), want);
        assert_eq!(super::par_map(threads, 0, |i| i), Vec::<usize>::new());
    }
}

#[test]
fn a_small_solid_of_flat_faces_is_meshed_alone_as_it_is_on_every_thread() {
    let rect = |w: f64, h: f64| Profile::rect(Vec2::ZERO, w, h);
    let block = extrude(&rect(4.0, 3.0), &Frame::WORLD, 2.0, "b").unwrap();
    // A slot cut through the block: flat faces still, and a face with a hole in it.
    let lifted = Frame::new(Vec3::new(0.0, 0.0, -1.0), Vec3::Z, Vec3::X).unwrap();
    let slotted = difference(
        &block,
        &extrude(&rect(2.0, 1.0), &lifted, 4.0, "s").unwrap(),
    );
    let slotted = slotted.unwrap();
    // A prism of 50 sides has 150 edges, and a rod's side is not flat.
    let ring: Vec<Vec2> = (0..50)
        .map(|i| {
            let a = TAU * i as f64 / 50.0;
            Vec2::new(5.0 * a.cos(), 5.0 * a.sin())
        })
        .collect();
    let names: Vec<String> = (0..50).map(|i| format!("s{i}")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let prism = extrude(&Profile::polygon(&ring, &names), &Frame::WORLD, 2.0, "p").unwrap();
    let rod = extrude(&Profile::circle(Vec2::ZERO, 2.0), &Frame::WORLD, 5.0, "r").unwrap();
    let opt = MeshOptions::within(0.05);
    for (solid, alone) in [
        (&block, true),
        (&slotted, true),
        (&prism, false),
        (&rod, false),
    ] {
        assert_eq!(super::plain(solid), alone);
        let one = super::tessellate_on(1, solid, &opt).unwrap();
        assert_eq!(one, super::tessellate_on(usize::MAX, solid, &opt).unwrap());
        assert_eq!(one, tessellate(solid, &opt).unwrap());
    }
}
