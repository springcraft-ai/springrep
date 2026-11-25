//! Area, volume and centre of solids whose edges pass a pole or an apex of a face's surface a
//! hair's breadth away, or through it, against closed forms worked by hand: past a pole the
//! longitude turns half a turn along a stretch of edge as short as the gap.

use std::f64::consts::PI;

use springrep::{
    boolean::{difference, intersection, union},
    check,
    geom::Surface,
    math::{Frame, Vec2, Vec3},
    measure,
    primitive::{block, cone, sphere, spindle_torus},
    profile::Profile,
    sweep::extrude,
    topo::Brep,
};

const GAPS: [f64; 4] = [1e-2, 1e-4, 1e-6, 0.0];

fn assert_near(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-12 * want.abs(),
        "{what}: {got} vs {want}, {:e} off",
        (got - want) / want
    );
}

fn assert_measures(b: &Brep, volume: f64, area: f64, what: &str) {
    let issues = check::check(b);
    assert!(issues.is_empty(), "{what}: {issues:?}");
    assert_near(measure::volume(b), volume, &format!("{what}, volume"));
    assert_near(measure::area(b), area, &format!("{what}, area"));
    let (a, v) = measure::area_and_volume(b);
    assert_near(v, volume, &format!("{what}, volume with the area"));
    assert_near(a, area, &format!("{what}, area with the volume"));
    let mass = measure::mass_properties(b);
    assert_near(
        mass.volume,
        volume,
        &format!("{what}, volume of the mass properties"),
    );
}

/// The half-space x > `d`, as a slab far larger than `size`.
fn past_x(d: f64, size: f64) -> Brep {
    let frame = Frame::new(Vec3::new(d, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap();
    extrude(
        &Profile::rect(Vec2::ZERO, 8.0 * size, 8.0 * size),
        &frame,
        4.0 * size,
        "past",
    )
    .unwrap()
}

#[test]
fn a_ball_cut_a_hair_from_its_poles_keeps_its_cap_and_its_rest_exactly() {
    // The plane x = d cuts a ball of radius r about the origin, its poles (0, 0, ±r), in a circle
    // that passes about d from each pole, through both at d = 0. The cap past it has height
    // h = r − d and holds π·h²·(3r − h)/3 under 2π·r·h of the ball and a disc of radius
    // √(r² − d²); its centre lies 3·(2r − h)²/(4·(3r − h)) out along x.
    let r = 10.0;
    let ball = sphere(&Frame::WORLD, r, "ball").unwrap();
    let whole = (4.0 / 3.0 * PI * r.powi(3), 4.0 * PI * r * r);
    for d in GAPS {
        let h = r - d;
        let cap = PI * h * h * (3.0 * r - h) / 3.0;
        let skin = 2.0 * PI * r * h;
        let disc = PI * (r * r - d * d);
        let cut = past_x(d, r);
        let kept = intersection(&ball, &cut).unwrap();
        let what = format!("gap {d}, cap");
        assert_measures(&kept, cap, skin + disc, &what);
        let centre = measure::mass_properties(&kept).centre;
        let x = 3.0 * (2.0 * r - h).powi(2) / (4.0 * (3.0 * r - h));
        assert_near(centre.x, x, &format!("{what}, centre"));
        let rest = difference(&ball, &cut).unwrap();
        let what = format!("gap {d}, rest");
        assert_measures(&rest, whole.0 - cap, whole.1 - skin + disc, &what);
    }
}

/// `∬ f` over the rectangle `[x0, x1] × [y0, y1]` from `g`, with `∂²g/∂x∂y = f`.
fn over_rect(g: &dyn Fn(f64, f64) -> f64, [x0, x1, y0, y1]: [f64; 4]) -> f64 {
    g(x1, y1) - g(x0, y1) - g(x1, y0) + g(x0, y0)
}

#[test]
fn a_post_on_a_ball_with_a_wall_a_hair_from_its_pole_measures_exactly() {
    // A post over the rectangle [x0, x1] × [−2, 2], inside the ball's equator, from the ball's
    // centre up to z = 15, on a ball of radius r = 10. With w = √(r² − x² − y²), the post holds
    // its box less ∬ w over the rectangle inside the ball, and the ball's skin under it,
    // ∬ r/w, is gone. Each wall rises from the skin, where z = w, to the top. By hand,
    // ∂²/∂x∂y of xyw/3 + x(3r² − x²)/6·atan(y/w) + y(3r² − y²)/6·atan(x/w) − r³/3·atan(xy/(rw))
    // is w, and of x·atan(y/w) + y·atan(x/w) − r·atan(xy/(rw)) it is 1/w; along a wall at x, a
    // chord of half length a = √(r² − x²), ∫ w dy is (y·√(a² − y²) + a²·asin(y/a))/2. The pole
    // lies beside the wall x0 = d with the post past it, or inside the post beside x1 = −d.
    let (r, top) = (10.0, 15.0);
    let ball = sphere(&Frame::WORLD, r, "ball").unwrap();
    let w = |x: f64, y: f64| (r * r - x * x - y * y).sqrt();
    let under = |x: f64, y: f64| {
        let w = w(x, y);
        x * y * w / 3.0
            + x * (3.0 * r * r - x * x) / 6.0 * (y / w).atan()
            + y * (3.0 * r * r - y * y) / 6.0 * (x / w).atan()
            - r.powi(3) / 3.0 * (x * y / (r * w)).atan()
    };
    let skin = |x: f64, y: f64| {
        let w = w(x, y);
        r * (x * (y / w).atan() + y * (x / w).atan() - r * (x * y / (r * w)).atan())
    };
    let chord = |at: f64, lo: f64, hi: f64| {
        let a2 = r * r - at * at;
        let s = |y: f64| (y * (a2 - y * y).sqrt() + a2 * (y / a2.sqrt()).asin()) / 2.0;
        top * (hi - lo) - (s(hi) - s(lo))
    };
    for d in GAPS {
        for [x0, x1] in [[d, d + 4.0], [-d - 4.0, -d]] {
            let rect = [x0, x1, -2.0, 2.0];
            let foot = (x1 - x0) * 4.0;
            let frame = Frame::new(Vec3::new(x0, -2.0, 0.0), Vec3::Z, Vec3::X).unwrap();
            let post = block(&frame, Vec3::new(x1 - x0, 4.0, top), "post").unwrap();
            let both = union(&ball, &post).unwrap();
            let volume = 4.0 / 3.0 * PI * r.powi(3) + foot * top - over_rect(&under, rect);
            let walls = chord(x0, -2.0, 2.0) + chord(x1, -2.0, 2.0) + chord(2.0, x0, x1) * 2.0;
            let area = 4.0 * PI * r * r - over_rect(&skin, rect) + foot + walls;
            assert_measures(
                &both,
                volume,
                area,
                &format!("gap {d}, post over {x0}..{x1}"),
            );
        }
    }
}

#[test]
fn a_cone_cut_a_hair_from_its_apex_keeps_both_pieces_exactly() {
    // A cone of base radius a and height H, its apex up, cut by the plane x = d, a hair from
    // the apex. At the height where the cone's radius is R the piece past the plane is a
    // segment of R²·acos(d/R) − d·√(R² − d²), and R falls by a over H, so the piece holds
    // H/a·∫ that dR from d to a. By hand, with q = √(R² − d²) and l = ln(R + q),
    // ∫ R²·acos(d/R) dR = R³/3·acos(d/R) − d/3·(R·q/2 + d²·l/2) and ∫ q dR = R·q/2 − d²·l/2.
    // Its skin on the cone is the base's segment over the cosine a/L of the side's slope, L the
    // slant √(a² + H²), its cut face the hyperbola's H/a·∫ 2q dR, and its base the segment.
    let (a, height) = (6.0f64, 8.0f64);
    let body = cone(&Frame::WORLD, a, 0.0, height, "cone").unwrap();
    let slant = a.hypot(height);
    let whole = (PI * a * a * height / 3.0, PI * a * slant + PI * a * a);
    for d in GAPS.into_iter().filter(|&d| d > 0.0) {
        let q = |x: f64| (x * x - d * d).max(0.0).sqrt();
        let l = |x: f64| (x + q(x)).ln();
        let acos_part = |x: f64| {
            x.powi(3) / 3.0 * (d / x).acos() - d / 3.0 * (x * q(x) / 2.0 + d * d * l(x) / 2.0)
        };
        let q_part = |x: f64| x * q(x) / 2.0 - d * d * l(x) / 2.0;
        let segment = a * a * (d / a).acos() - d * q(a);
        let volume = height / a * ((acos_part(a) - d * q_part(a)) - (acos_part(d) - d * q_part(d)));
        let hyperbola = height / a * 2.0 * (q_part(a) - q_part(d));
        let side = slant / a * segment;
        let cut = past_x(d, a + height);
        let kept = intersection(&body, &cut).unwrap();
        let what = format!("gap {d}");
        assert_measures(
            &kept,
            volume,
            side + segment + hyperbola,
            &format!("{what}, kept"),
        );
        let rest = difference(&body, &cut).unwrap();
        let area = whole.1 - side - segment + hyperbola;
        assert_measures(&rest, whole.0 - volume, area, &format!("{what}, rest"));
    }
}

#[test]
fn a_spindle_torus_cut_a_hair_from_its_poles_keeps_its_whole_skin() {
    // A spindle torus, its tube of radius m about a centre circle of radius M < m, meets its
    // axis at two poles, where the outer arc ends at ±v_p from the tube's centre, v_p =
    // π − acos(M/m). With k = √(m² − M²), its skin is 4π·m·(M·v_p + k) and, turning the part of
    // the tube's disc out from the axis, it holds 2π·(π·M·m² + 2k³/3 + M²·k − M·m²·acos(M/m)).
    // The plane x = d passes a hair from both poles along an intersection curve that both pieces
    // share, so their skins on the torus make up its whole skin, and about one centre their
    // volumes add up to its whole volume.
    let (big, small) = (3.0f64, 5.0f64);
    let body = spindle_torus(&Frame::WORLD, big, small, "spindle").unwrap();
    let k = (small * small - big * big).sqrt();
    let reach = PI - (big / small).acos();
    let skin = 4.0 * PI * small * (big * reach + k);
    let volume = 2.0
        * PI
        * (PI * big * small * small + 2.0 * k.powi(3) / 3.0 + big * big * k
            - big * small * small * (big / small).acos());
    assert_measures(&body, volume, skin, "whole");
    for d in GAPS {
        let cut = past_x(d, 2.0 * (big + small));
        let mut both = intersection(&body, &cut).unwrap();
        both.append(&difference(&body, &cut).unwrap());
        let flat: f64 = both
            .face_ids()
            .filter(|&f| matches!(both.face(f).surface, Surface::Plane(_)))
            .map(|f| measure::face_area(&both, f))
            .sum();
        assert_near(measure::area(&both) - flat, skin, &format!("gap {d}, skin"));
        assert_near(measure::volume(&both), volume, &format!("gap {d}, volume"));
    }
}
