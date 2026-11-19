use std::f64::consts::{PI, TAU};

use super::{read, write};
use crate::{
    boolean, check,
    geom::{Circle, Cone, Curve, Line, Ruled, Surface},
    math::{Frame, Vec2, Vec3},
    measure,
    profile::{NamedSegment, PointsInterp, Profile, ProfileLoop, Segment},
    sweep::{extrude, revolve},
    topo::Brep,
};

fn seg(segment: Segment, name: &str) -> NamedSegment {
    NamedSegment::new(segment, name)
}

fn block(id: &str, lo: [f64; 3], hi: [f64; 3]) -> Brep {
    let c = Vec2::new(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
    let frame = Frame::new(Vec3::new(0.0, 0.0, lo[2]), Vec3::Z, Vec3::X).unwrap();
    let p = Profile::rect(c, hi[0] - lo[0], hi[1] - lo[1]);
    extrude(&p, &frame, hi[2] - lo[2], id).unwrap()
}

fn xz() -> Frame {
    Frame::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), Vec3::X).unwrap()
}

fn ball(r: f64) -> Brep {
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
    revolve(&p, &xz(), Vec3::ZERO, Vec3::Z, TAU, "ball").unwrap()
}

/// Writes `b`, reads it back, and checks the copy is valid with the same measures and names.
fn round_trip(b: &Brep) -> Brep {
    assert!(check::check(b).is_empty());
    let text = write(b, "part").unwrap();
    let back = read(text.as_bytes(), "imp").unwrap();
    assert!(back.rejected.is_empty(), "{:?}", back.rejected);
    let c = back.brep;
    assert!(check::check(&c).is_empty(), "{:?}", check::check(&c));
    assert_eq!(c.solid_ids().count(), b.solid_ids().count());
    assert_eq!(c.face_count(), b.face_count());
    for (got, want, what) in [
        (measure::volume(&c), measure::volume(b), "volume"),
        (measure::area(&c), measure::area(b), "area"),
    ] {
        assert!((got - want).abs() <= 1e-9 * want, "{what} {got} vs {want}");
    }
    let (bb, cb) = (measure::bbox(b), measure::bbox(&c));
    assert!((bb.min - cb.min).max_abs().max((bb.max - cb.max).max_abs()) < 1e-9);
    assert_eq!(c.name_counts(), b.name_counts());
    c
}

#[test]
fn planar_solids_round_trip() {
    round_trip(&block("blk", [0.0, 0.0, 0.0], [10.0, 4.0, 3.0]));
    let hole = Profile::circle(Vec2::new(2.0, 0.0), 2.0).loops.remove(0);
    let p = Profile::rect(Vec2::ZERO, 10.0, 6.0).with_holes([hole]);
    round_trip(&extrude(&p, &Frame::WORLD, 2.0, "plate").unwrap());
}

#[test]
fn spheres_and_tori_round_trip_through_their_poles_and_seams() {
    round_trip(&ball(5.0));
    let ring = Profile::circle(Vec2::new(8.0, 0.0), 4.0);
    round_trip(&revolve(&ring, &xz(), Vec3::ZERO, Vec3::Z, TAU, "ring").unwrap());
    let half = revolve(&ring, &xz(), Vec3::ZERO, Vec3::Z, PI, "arc").unwrap();
    round_trip(&half);
}

/// Cones of all four kinds of half angle, each written with a STEP semi-angle in `[0, π/2)`.
#[test]
fn cones_round_trip_both_ways() {
    for pts in [
        [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (3.0, 5.0), (0.0, 5.0)],
        [(0.0, 0.0), (3.0, 0.0), (6.0, 3.0), (6.0, 5.0), (0.0, 5.0)],
    ] {
        for (flip, axis) in [
            (1.0, Vec3::Z),
            (-1.0, Vec3::Z),
            (1.0, -Vec3::Z),
            (-1.0, -Vec3::Z),
        ] {
            let pts: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, flip * y)).collect();
            let p = Profile::polygon(&pts, &["a", "b", "c", "d", "e"]);
            let b = revolve(&p, &xz(), Vec3::ZERO, axis, TAU, "t").unwrap();
            let text = write(&b, "t").unwrap();
            for l in text.lines().filter(|l| l.contains("=CONICAL_SURFACE(")) {
                let a: f64 = l
                    .rsplit(',')
                    .next()
                    .unwrap()
                    .trim_end_matches(");")
                    .parse()
                    .unwrap();
                assert!((0.0..PI / 2.0).contains(&a), "{l}");
            }
            round_trip(&b);
        }
    }
    // Each quadrant of half angle reads back as the same points, with the normal flip the face's
    // sense undoes.
    for a in [0.4, -0.4, PI - 0.4, 0.4 - PI] {
        let cone = Surface::Cone(Cone {
            frame: Frame::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.6, 0.8), Vec3::X).unwrap(),
            radius: 4.0,
            half_angle: a,
        });
        let mut w = super::write::Writer::new();
        let id = w.surface(&cone).unwrap();
        let text = format!(
            "ISO-10303-21;\nDATA;\n{}ENDSEC;\nEND-ISO-10303-21;\n",
            w.out
        );
        let file = super::part21::parse(text.as_bytes()).unwrap();
        let back = super::geometry::Geometry::new(&file)
            .surface(id as u64)
            .unwrap();
        let flips = a.abs() > PI / 2.0;
        for (u, v) in [(0.3, 0.5), (2.0, -1.0), (4.0, 2.5)] {
            let p = cone.eval(u, v);
            let q = back.project(p);
            assert!(back.eval(q.x, q.y).dist(p) < 1e-9, "{a}");
            let dot = back.normal(q.x, q.y).dot(cone.normal(u, v));
            assert!(
                (dot - if flips { -1.0 } else { 1.0 }).abs() < 1e-9,
                "{a}: {dot}"
            );
        }
    }
}

#[test]
fn spline_extrusions_and_revolutions_round_trip() {
    let pts: Vec<Vec2> = (0..7)
        .map(|i| {
            let a = TAU * i as f64 / 7.0;
            let r = 10.0 + 2.0 * (3.0 * a).cos();
            Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect();
    let p = Profile::points(&pts, PointsInterp::PeriodicSpline, "side").unwrap();
    round_trip(&extrude(&p, &Frame::WORLD, 4.0, "cam").unwrap());
    let open = [
        (0.0, 0.0),
        (4.0, 0.0),
        (5.0, 2.0),
        (4.5, 4.0),
        (5.5, 6.0),
        (0.0, 6.0),
    ];
    let open: Vec<Vec2> = open.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    let prof = Profile::points(&open, PointsInterp::Smooth { corner: 0.5 }, "s").unwrap();
    round_trip(&revolve(&prof, &xz(), Vec3::ZERO, Vec3::Z, TAU, "vase").unwrap());
}

#[test]
fn boolean_results_with_voids_round_trip() {
    let outer = block("o", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let inner = block("i", [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]);
    let hollow = boolean::difference(&outer, &inner).unwrap();
    assert_eq!(
        hollow
            .solid(hollow.solid_ids().next().unwrap())
            .shells
            .len(),
        2
    );
    round_trip(&hollow);
}

/// Some systems write a void's faces already facing into it and turn its shell as well, so the
/// shell's own orientation and the solid's say opposite things: the void is read by the volume it
/// encloses, which is negative.
#[test]
fn a_void_turned_twice_is_read_as_a_void() {
    let outer = block("o", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let inner = block("i", [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]);
    let hollow = boolean::difference(&outer, &inner).unwrap();
    let text = write(&hollow, "part").unwrap();
    // The void's faces: those of the shell its ORIENTED_CLOSED_SHELL names.
    let entity = |id: &str| {
        let at = text.find(&format!("\n#{id}=")).unwrap() + 1;
        text[at..at + text[at..].find(";\n").unwrap()].to_string()
    };
    let refs_in = |s: &str| -> Vec<String> {
        s.split('#')
            .skip(1)
            .map(|r| r.chars().take_while(char::is_ascii_digit).collect())
            .collect()
    };
    let at = text.find("=ORIENTED_CLOSED_SHELL(").unwrap();
    let line = &text[text[..at].rfind('\n').unwrap() + 1..at + text[at..].find(";\n").unwrap()];
    let shell = refs_in(line.split_once('=').unwrap().1)[0].clone();
    // Each face's sense and each of its bounds' orientation turned: the same face, facing the
    // other way, its loops still keeping it on their left.
    let mut turned = text.clone();
    let mut turn = |id: &str| {
        let line = entity(id);
        let flipped = match line.strip_suffix(".T.)") {
            Some(head) => format!("{head}.F.)"),
            None => format!("{}.T.)", line.strip_suffix(".F.)").unwrap()),
        };
        turned = turned.replace(&format!("\n{line};"), &format!("\n{flipped};"));
    };
    for f in refs_in(entity(&shell).split_once('=').unwrap().1) {
        let face = entity(&f);
        let bounds = face
            .split_once("(")
            .unwrap()
            .1
            .split_once(")")
            .unwrap()
            .0
            .to_string();
        for bd in refs_in(&bounds) {
            turn(&bd);
        }
        turn(&f);
    }
    assert_ne!(turned, text);
    let back = read(turned.as_bytes(), "imp").unwrap();
    assert!(back.rejected.is_empty(), "{:?}", back.rejected);
    assert!(check::check(&back.brep).is_empty());
    assert!((measure::volume(&back.brep) - 973.0).abs() < 1e-9);
}

/// How many instances of each of `kinds` the file has.
fn instances(text: &str, kind: &str) -> usize {
    text.matches(&format!("={kind}(")).count()
}

/// Writes a solid whose pieces touch, checks the file holds them as manifold solids, as systems
/// that write the manifold boundary of a solid's interior do: each piece its own solid, the
/// contact's vertices and edges written once for each side, no `VERTEX_LOOP`. Returns the file and
/// what reading it gives.
fn written_as_pieces(b: &Brep, solids: usize) -> (String, Brep) {
    assert!(check::check(b).is_empty());
    let text = write(b, "part").unwrap();
    let written = instances(&text, "MANIFOLD_SOLID_BREP") + instances(&text, "BREP_WITH_VOIDS");
    assert_eq!(written, solids, "{text}");
    assert_eq!(instances(&text, "VERTEX_LOOP"), 0);
    let back = read(text.as_bytes(), "imp").unwrap();
    assert!(back.rejected.is_empty(), "{:?}", back.rejected);
    let c = back.brep;
    assert!(
        check::check_each(&c).is_empty(),
        "{:?}",
        check::check_each(&c)
    );
    let (vb, vc) = (measure::volume(b), measure::volume(&c));
    assert!((vb - vc).abs() <= 1e-9 * vb, "volume {vc} vs {vb}");
    (text, c)
}

/// The vertices with a point, and the edges with a curve, of `b`.
fn vertices_and_edges(b: &Brep) -> (usize, usize) {
    let edges = b.edge_ids().filter(|&e| !b.edge(e).is_degenerate()).count();
    (b.vertex_ids().count(), edges)
}

/// Two cubes sharing an edge are two solids in the file, each with all 8 corners and 12 edges.
#[test]
fn pieces_touching_along_an_edge_are_written_as_two_solids() {
    let a = block("a", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let b = block("b", [10.0, 10.0, 0.0], [20.0, 20.0, 10.0]);
    let u = boolean::union(&a, &b).unwrap();
    let (text, back) = written_as_pieces(&u, 2);
    assert_eq!(instances(&text, "VERTEX_POINT"), 16);
    assert_eq!(instances(&text, "EDGE_CURVE"), 24);
    assert_eq!(back.solid_ids().count(), 2);
}

/// A rod lying on a box's top face along a line: in the result the line is a slit in the top face,
/// which runs along it and back, and splits the rod's side in two. In the box's own boundary it is
/// no edge at all, so the file has a plain box; the rod keeps its two halves.
#[test]
fn a_slit_where_a_piece_touches_a_face_is_not_written() {
    let a = block("a", [0.0, 0.0, 0.0], [20.0, 10.0, 10.0]);
    let frame = Frame::new(Vec3::new(5.0, 5.0, 12.0), Vec3::X, Vec3::Y).unwrap();
    let rod = extrude(&Profile::circle(Vec2::ZERO, 4.0), &frame, 10.0, "c").unwrap();
    let u = boolean::union(&a, &rod).unwrap();
    assert_eq!(u.radial_edges().count(), 1);
    let (text, back) = written_as_pieces(&u, 2);
    // The box's 12 edges, and the rod's: the line between its halves, its seam and the two arcs
    // of each end.
    assert_eq!(instances(&text, "EDGE_CURVE"), 18);
    assert_eq!(back.solid_ids().count(), 2);
}

/// Two balls touching at a point are two balls in the file; the point is in neither.
#[test]
fn balls_touching_at_a_point_are_written_as_two_solids() {
    let mut b = ball(5.0);
    b.transform(&crate::math::Transform::translation(Vec3::new(
        0.0, 10.0, 0.0,
    )));
    let u = boolean::union(&ball(5.0), &b).unwrap();
    let (_, back) = written_as_pieces(&u, 2);
    assert_eq!(back.solid_ids().count(), 2);
}

/// A block less a bore whose top line lies on its top face is one solid touching itself along
/// that line: the file gives each side of the contact its own edge, and reading it joins them into
/// the edge four faces use again.
#[test]
fn a_solid_touching_itself_along_a_line_is_one_solid_read_back_joined() {
    let a = block("a", [0.0, 0.0, 0.0], [12.0, 10.0, 8.0]);
    let frame = Frame::new(Vec3::new(-1.0, 5.0, 5.0), Vec3::X, Vec3::Y).unwrap();
    let bore = extrude(&Profile::circle(Vec2::ZERO, 6.0), &frame, 14.0, "c").unwrap();
    let d = boolean::difference(&a, &bore).unwrap();
    assert_eq!(d.radial_edges().count(), 1);
    let (text, back) = written_as_pieces(&d, 1);
    let (vertices, edges) = vertices_and_edges(&d);
    // The contact line and its two ends, once more each.
    assert_eq!(instances(&text, "EDGE_CURVE"), edges + 1);
    assert_eq!(instances(&text, "VERTEX_POINT"), vertices + 2);
    assert!(check::check(&back).is_empty());
    assert_eq!(back.radial_edges().count(), 1);
    assert_eq!(vertices_and_edges(&back), (vertices, edges));
    assert_eq!(measure::betti(&back), [1, 1, 0]);
}

/// A cube less the ball touching its six faces: the void and the skin are written apart, the
/// six points where they touch in neither, so the file's solid has its void but not the contacts.
#[test]
fn a_void_touching_the_skin_at_points_is_written_apart_from_it() {
    let mut s = ball(5.0);
    s.transform(&crate::math::Transform::translation(Vec3::new(
        5.0, 5.0, 5.0,
    )));
    let d = boolean::difference(&block("c", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]), &s).unwrap();
    assert!(d.loop_ids().any(|lp| d.is_vertex_loop(lp)));
    let (text, back) = written_as_pieces(&d, 1);
    assert_eq!(instances(&text, "BREP_WITH_VOIDS"), 1);
    assert_eq!(back.solid_ids().count(), 1);
    assert_eq!(measure::betti(&back), [1, 0, 1]);
}

#[test]
fn intersection_curves_are_written_as_their_splines() {
    let a = Surface::Cylinder(crate::geom::Cylinder {
        frame: Frame::WORLD,
        radius: 5.0,
    });
    let b = Surface::Cylinder(crate::geom::Cylinder {
        frame: Frame::new(Vec3::ZERO, Vec3::X, Vec3::Y).unwrap(),
        radius: 3.0,
    });
    let pts: Vec<Vec3> = (0..12)
        .map(|i| {
            let t = TAU * i as f64 / 12.0;
            let (y, z) = (3.0 * t.cos(), 3.0 * t.sin());
            Vec3::new((25.0 - y * y).sqrt(), y, z)
        })
        .collect();
    let fit = crate::geom::IntersectionCurve::fit(&pts, a.clone(), b.clone(), true).unwrap();
    let curve = Curve::Intersection(Box::new(fit));
    let mut w = super::write::Writer::new();
    let id = w.curve(&curve).unwrap();
    let text = format!(
        "ISO-10303-21;\nDATA;\n{}ENDSEC;\nEND-ISO-10303-21;\n",
        w.out
    );
    let file = super::part21::parse(text.as_bytes()).unwrap();
    let back = super::geometry::Geometry::new(&file)
        .curve(id as u64)
        .unwrap();
    assert!(back.period().is_some());
    let (t0, t1) = curve.domain();
    for i in 0..=50 {
        let t = t0 + (t1 - t0) * i as f64 / 50.0;
        assert!(back.eval(t).dist(curve.eval(t)) < 1e-12);
        assert!(crate::geom::distance_to(&a, back.eval(t)) < crate::tol::LINEAR);
        assert!(crate::geom::distance_to(&b, back.eval(t)) < crate::tol::LINEAR);
    }
}

#[test]
fn ruled_surfaces_are_written_as_the_same_surface() {
    let circle = |z: f64, r: f64| {
        Curve::Circle(Circle {
            frame: Frame::WORLD.translated(Vec3::new(0.0, 0.0, z)),
            radius: r,
        })
    };
    let line = |a: Vec3, b: Vec3| Curve::Line(Line::through(a, b).unwrap().0);
    let cases = [
        Ruled {
            a: circle(0.0, 3.0),
            a_range: (0.0, TAU),
            b: circle(5.0, 2.0),
            b_range: (1.0, 1.0 + TAU),
        },
        Ruled {
            a: line(Vec3::ZERO, Vec3::X),
            a_range: (0.0, 1.0),
            b: Curve::BSpline(
                crate::geom::BSpline::interpolate(&[
                    Vec3::new(0.0, 2.0, 0.0),
                    Vec3::new(0.3, 2.5, 0.0),
                    Vec3::new(0.7, 2.2, 1.0),
                    Vec3::new(1.0, 2.0, 0.0),
                ])
                .unwrap(),
            ),
            b_range: (0.0, 1.0),
        },
    ];
    for r in cases {
        let ruled = Surface::Ruled(Box::new(r));
        let mut w = super::write::Writer::new();
        let id = w.surface(&ruled).unwrap();
        let text = format!(
            "ISO-10303-21;\nDATA;\n{}ENDSEC;\nEND-ISO-10303-21;\n",
            w.out
        );
        let file = super::part21::parse(text.as_bytes()).unwrap();
        let mut geo = super::geometry::Geometry::new(&file);
        let Surface::Nurbs(n) = geo.surface(id as u64).unwrap() else {
            panic!("expected a B-spline surface")
        };
        // The B-spline's parameter along the rails need not be the ruled surface's, but each
        // of its rulings must be one of the ruled surface's, run at the same rate.
        let ((u0, u1), _) = n.domain();
        for i in 0..=10 {
            let u = u0 + (u1 - u0) * i as f64 / 10.0;
            let at = ruled.project(n.eval(u, 0.0));
            assert!(at.y.abs() < 1e-9);
            for j in 0..=4 {
                let v = j as f64 / 4.0;
                let d = n.eval(u, v).dist(ruled.eval(at.x, v));
                assert!(d < 1e-9, "{d} at {u},{v}");
            }
        }
    }
}

/// A part of two blocks placed by an assembly, in inches, as other systems write them. With
/// `cycle`, the part also uses itself.
fn assembly_text(cycle: bool) -> String {
    let part = write(&block("b", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), "blk").unwrap();
    let data = part
        .split("DATA;\n")
        .nth(1)
        .unwrap()
        .split("ENDSEC;")
        .next()
        .unwrap()
        .replace(
            "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))",
            "(CONVERSION_BASED_UNIT('INCH',#9001) LENGTH_UNIT() NAMED_UNIT(*))",
        );
    let pd = data
        .lines()
        .find(|l| l.contains("=PRODUCT_DEFINITION('design'"))
        .and_then(|l| l[1..].split('=').next())
        .unwrap()
        .to_string();
    let rep = data
        .lines()
        .find(|l| l.contains("ADVANCED_BREP_SHAPE_REPRESENTATION"))
        .and_then(|l| l[1..].split('=').next())
        .unwrap()
        .to_string();
    let extra = format!(
        "#9000=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));
#9001=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#9000);
#9010=PRODUCT('asm','asm','',(#3));
#9011=PRODUCT_DEFINITION_FORMATION('','',#9010);
#9012=PRODUCT_DEFINITION('design','',#9011,#6);
#9013=PRODUCT_DEFINITION_SHAPE('','',#9012);
#9014=SHAPE_REPRESENTATION('asm',(#9020),#9099);
#9015=SHAPE_DEFINITION_REPRESENTATION(#9013,#9014);
#9020=AXIS2_PLACEMENT_3D('',#9021,#9022,#9023);
#9021=CARTESIAN_POINT('',(0.,0.,0.));
#9022=DIRECTION('',(0.,0.,1.));
#9023=DIRECTION('',(1.,0.,0.));
#9030=AXIS2_PLACEMENT_3D('',#9031,#9022,#9023);
#9031=CARTESIAN_POINT('',(10.,0.,0.));
#9040=AXIS2_PLACEMENT_3D('',#9041,#9042,#9023);
#9041=CARTESIAN_POINT('',(0.,0.,5.));
#9042=DIRECTION('',(0.,-1.,0.));
#9099=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#9000)) \
       REPRESENTATION_CONTEXT('','3D'));
{}",
        [("9030", "1", "9012"), ("9040", "2", "9012"), ("9030", "3", &pd)]
            .iter()
            .take(if cycle { 3 } else { 2 })
            .map(|(place, k, parent)| format!(
                "#91{k}0=NEXT_ASSEMBLY_USAGE_OCCURRENCE('{k}','','',#{parent},#{pd},$);
#91{k}1=ITEM_DEFINED_TRANSFORMATION('','',#9020,#{place});
#91{k}2=(REPRESENTATION_RELATIONSHIP('','',#{rep},#9014) \
       REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#91{k}1) SHAPE_REPRESENTATION_RELATIONSHIP());
#91{k}3=PRODUCT_DEFINITION_SHAPE('','',#91{k}0);
#91{k}4=CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#91{k}2,#91{k}3);
"
            ))
            .collect::<String>()
    );
    format!(
        "ISO-10303-21;\nHEADER;\nFILE_NAME('a','',(''),(''),'','other','');\nENDSEC;\nDATA;\n\
         {data}{extra}ENDSEC;\nEND-ISO-10303-21;\n"
    )
}

#[test]
fn assemblies_place_each_instance_in_file_units() {
    let text = assembly_text(false);
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    let b = imp.brep;
    assert_eq!(b.solid_ids().count(), 2);
    let inch: f64 = 25.4;
    assert!((measure::volume(&b) - 2.0 * inch.powi(3)).abs() < 1e-9 * inch.powi(3));
    let bb = measure::bbox(&b);
    // The first block moves 10 mm along x; the second turns y→z, z→−y and rises 5 mm.
    assert!(
        (bb.min - Vec3::new(0.0, -inch, 0.0)).max_abs() < 1e-9,
        "{bb:?}"
    );
    assert!(
        (bb.max - Vec3::new(10.0 + inch, inch, 5.0 + inch)).max_abs() < 1e-9,
        "{bb:?}"
    );
    assert_eq!(b.name_counts().get("imp.face_0"), Some(&1));
    assert_eq!(b.name_counts().len(), 12);
}

/// The text of `b` written as STEP with the point of its first vertex moved by `d`.
fn with_vertex_moved(b: &Brep, d: Vec3) -> String {
    let text = write(b, "blk").unwrap();
    let vp = text.lines().find(|l| l.contains("=VERTEX_POINT(")).unwrap();
    let pid = vp.split('#').nth(2).unwrap().trim_end_matches(");");
    let head = format!("#{pid}=CARTESIAN_POINT('',(");
    let line = text.lines().find(|l| l.starts_with(&head)).unwrap();
    let xyz: Vec<f64> = line[head.len()..line.len() - 3]
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    // A new point, so geometry that shared the vertex's point stays where it was.
    let moved = format!(
        "#99999=CARTESIAN_POINT('',({:?},{:?},{:?}));\n",
        xyz[0] + d.x,
        xyz[1] + d.y,
        xyz[2] + d.z
    );
    let vp_moved = vp.replace(&format!("#{pid})"), "#99999)");
    text.replace(vp, &vp_moved)
        .replace("ENDSEC;\nEND-ISO", &format!("{moved}ENDSEC;\nEND-ISO"))
        .replace("'springrep'", "'other'")
}

#[test]
fn a_vertex_off_its_faces_is_moved_back_onto_them() {
    let blk = block("b", [0.0, 0.0, 0.0], [10.0, 4.0, 3.0]);
    let text = with_vertex_moved(&blk, Vec3::new(2e-5, -1e-5, 3e-5));
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(check::check(&imp.brep).is_empty());
    assert!((measure::volume(&imp.brep) - 120.0).abs() < 1e-9);
    let bb = measure::bbox(&imp.brep);
    assert!(
        bb.min
            .max_abs()
            .max((bb.max - Vec3::new(10.0, 4.0, 3.0)).max_abs())
            < 1e-9
    );
}

#[test]
fn faces_that_do_not_meet_are_rejected_not_widened() {
    // A vertex a millimetre off its corner is far past anything healing may move.
    let blk = block("b", [0.0, 0.0, 0.0], [10.0, 4.0, 3.0]);
    let text = with_vertex_moved(&blk, Vec3::new(1.0, 0.0, 0.0));
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert_eq!(imp.brep.solid_ids().count(), 0);
    assert_eq!(imp.rejected.len(), 1);
}

#[test]
fn reference_cycles_and_short_points_are_errors() {
    assert!(read(assembly_text(true).as_bytes(), "imp").is_err());
    let blk = block("b", [0.0, 0.0, 0.0], [10.0, 4.0, 3.0]);
    let text = write(&blk, "blk").unwrap();
    let line = |kind: &str| {
        let l = text.lines().find(|l| l.contains(kind)).unwrap();
        (l.to_string(), l.split('=').next().unwrap().to_string())
    };
    let (vp, vid) = line("=VERTEX_POINT(");
    let (pl, pid) = line("=PLANE(");
    let (cp, _) = line("=CARTESIAN_POINT(");
    let mutants = [
        text.replace(&vp, &format!("{vid}=VERTEX_POINT('',{vid});")),
        text.replace(
            &pl,
            &format!("{pid}=RECTANGULAR_TRIMMED_SURFACE('',{pid},0.,1.,0.,1.,.T.,.T.);"),
        ),
        text.replace(
            &cp,
            &format!(
                "{}=CARTESIAN_POINT('',(1.0));",
                cp.split('=').next().unwrap()
            ),
        ),
    ];
    for m in mutants {
        let imp = read(m.as_bytes(), "imp").unwrap();
        assert_eq!(imp.brep.solid_ids().count(), 0);
        assert_eq!(imp.rejected.len(), 1, "{:?}", imp.rejected);
    }
}

/// A block with one vertical edge rounded: the round is a cylinder tangent to two planes.
fn rounded_block() -> Brep {
    let (w, h, r) = (10.0, 6.0, 1.0);
    let line = |a: (f64, f64), b: (f64, f64), name: &str| {
        seg(
            Segment::Line {
                a: Vec2::new(a.0, a.1),
                b: Vec2::new(b.0, b.1),
            },
            name,
        )
    };
    let round = seg(
        Segment::Arc {
            center: Vec2::new(w - r, h - r),
            radius: r,
            start: 0.0,
            sweep: PI / 2.0,
        },
        "round",
    );
    let p = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                line((0.0, 0.0), (w, 0.0), "s"),
                line((w, 0.0), (w, h - r), "e"),
                round,
                line((w - r, h), (0.0, h), "n"),
                line((0.0, h), (0.0, 0.0), "w"),
            ],
        }],
    };
    extrude(&p, &Frame::WORLD, 3.0, "rb").unwrap()
}

/// `b` written as another system's STEP with every cylinder's radius grown by `dr`, so the
/// rounds no longer touch the sides they are tangent to.
fn with_rounds_grown(b: &Brep, dr: f64) -> String {
    let text = write(b, "rb").unwrap().replace("'springrep'", "'other'");
    text.lines()
        .map(|l| match l.find("=CYLINDRICAL_SURFACE(") {
            Some(_) => {
                let (head, r) = l.rsplit_once(',').unwrap();
                let r: f64 = r.trim_end_matches(");").parse().unwrap();
                format!("{head},{:?});", r + dr)
            }
            None => l.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn faces_held_apart_are_closed_by_moving_their_surfaces() {
    let b = rounded_block();
    let dr = 3e-5;
    let imp = read(with_rounds_grown(&b, dr).as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(check::check(&imp.brep).is_empty());
    assert_eq!(imp.healed.len(), 1);
    let h = &imp.healed[0];
    assert!(h.moved >= 0.5 * dr && h.moved <= 0.05, "{h:?}");
    assert!(h.surfaces > 0.0 && h.surfaces <= h.moved, "{h:?}");
    // Moving surfaces by at most `surfaces` changes the volume by at most that times the area.
    let (v0, v1) = (measure::volume(&b), measure::volume(&imp.brep));
    assert!(
        (v1 - v0).abs() <= measure::area(&b) * h.surfaces,
        "{v1} vs {v0}"
    );
}

#[test]
fn geometry_healing_may_not_close_is_kept_with_the_tolerances_it_measures() {
    // The rounds grown 0.2 mm: healing would move a surface 0.15 mm, more than it may, so the
    // file's geometry stays and the round's four edges and four vertices hold the gap, whatever
    // the file states.
    let imp = read(with_rounds_grown(&rounded_block(), 0.2).as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(imp.healed.is_empty());
    assert!(check::check(&imp.brep).is_empty());
    assert_eq!((imp.tolerant[0].vertices, imp.tolerant[0].edges), (4, 4));
    assert!(
        (imp.tolerant[0].largest - 0.3).abs() < 1e-6,
        "{:?}",
        imp.tolerant
    );
}

#[test]
fn a_gap_too_wide_to_tell_an_edge_s_ends_apart_is_rejected() {
    // Grown 0.5 mm, the round's edges need tolerances of 0.75 mm, and vertices 1.4 mm apart are
    // then one point within them: the solid has no definite shape, whatever the file states.
    let text = with_uncertainty(&with_rounds_grown(&rounded_block(), 0.5), 5.0);
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert_eq!(imp.brep.solid_ids().count(), 0);
    let why = &imp.rejected[0];
    assert!(
        why.contains("within the tolerance, so the same point")
            || why.contains("too far for an edge"),
        "{why}"
    );
}

/// `text` with its file's stated uncertainty set to `mm`.
fn with_uncertainty(text: &str, mm: f64) -> String {
    let head = "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(";
    let at = text.find(head).unwrap() + head.len();
    let end = at + text[at..].find(')').unwrap();
    format!("{}{mm:?}{}", &text[..at], &text[end..])
}

/// The rounded block's rounds grown by 0.2 mm, past what healing may move (0.05 mm): the round
/// misses the two sides it was tangent to, and its two end arcs, by 0.2 mm.
fn rounds_held_apart(uncertainty: f64) -> super::Import {
    let text = with_uncertainty(&with_rounds_grown(&rounded_block(), 0.2), uncertainty);
    read(text.as_bytes(), "imp").unwrap()
}

#[test]
fn faces_held_apart_within_the_files_uncertainty_keep_local_tolerances() {
    let imp = rounds_held_apart(0.4);
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(imp.healed.is_empty());
    assert!(check::check(&imp.brep).is_empty());
    // The round's two lines and two arcs, and the four vertices they end at, each 0.2 mm off it,
    // get 1.5 times that; every other edge and vertex lies on its faces.
    let (vs, es) = imp.brep.tolerant();
    assert_eq!((vs.len(), es.len()), (4, 4));
    for t in vs.iter().map(|x| x.1).chain(es.iter().map(|x| x.1)) {
        assert!((t - 0.3).abs() < 1e-6, "{t}");
    }
    assert_eq!(imp.tolerant.len(), 1);
    assert_eq!((imp.tolerant[0].vertices, imp.tolerant[0].edges), (4, 4));
    // Faces bounded by edges that miss them by at most 0.3 enclose the block to within its
    // area times that.
    let b = rounded_block();
    let (v0, v1) = (measure::volume(&b), measure::volume(&imp.brep));
    assert!((v1 - v0).abs() <= 0.3 * measure::area(&b), "{v1} vs {v0}");
    // Clearing the tolerances leaves edges that miss their faces.
    let mut exact = imp.brep.clone();
    for (e, _) in es {
        exact.set_edge_tolerance(e, crate::tol::LINEAR);
    }
    for (v, _) in vs {
        exact.set_vertex_tolerance(v, crate::tol::LINEAR);
    }
    assert!(!check::check(&exact).is_empty());
}

#[test]
fn gaps_wider_than_the_file_states_are_kept_and_counted() {
    // The file states 0.15 mm and its round misses by 0.2: the eight tolerances of 0.3 mm are
    // all past what it states, and the report says so.
    let imp = rounds_held_apart(0.15);
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    let t = &imp.tolerant[0];
    assert_eq!((t.vertices, t.edges, t.past_stated), (4, 4, 8));
    assert_eq!(t.stated, Some(0.15));
    // Stated at 0.4 mm, none is past it.
    let within = rounds_held_apart(0.4);
    assert_eq!(within.tolerant[0].past_stated, 0);
}

/// `b` written as another system's STEP with its first straight edge's curve bent into a spline
/// through its two vertices whose middle lies `d` off the edge, along the unit vector (1, 1, 1)
/// leaves across it.
fn with_edge_bent(b: &Brep, d: f64) -> String {
    let text = write(b, "b").unwrap().replace("'springrep'", "'other'");
    let ents: std::collections::HashMap<&str, &str> =
        text.lines().filter_map(|l| l.split_once('=')).collect();
    let refs = |s: &str| -> Vec<String> {
        s.split('#')
            .skip(1)
            .map(|r| {
                format!(
                    "#{}",
                    r.split(|c: char| !c.is_ascii_digit()).next().unwrap()
                )
            })
            .collect()
    };
    let point = |id: &str| -> Vec3 {
        let body = ents[id];
        let inner = &body[body.rfind('(').unwrap() + 1..body.find(')').unwrap()];
        let c: Vec<f64> = inner
            .split(',')
            .map(|x| x.trim().parse().unwrap())
            .collect();
        Vec3::new(c[0], c[1], c[2])
    };
    let (curve, pa, pb) = ents
        .values()
        .filter(|body| body.starts_with("EDGE_CURVE("))
        .find_map(|body| {
            let r = refs(body);
            let geom = ents[r[2].as_str()];
            let curve = match geom.starts_with("SURFACE_CURVE(") {
                true => refs(geom)[0].clone(),
                false => r[2].clone(),
            };
            let vertex = |v: &str| refs(ents[v])[0].clone();
            ents[curve.as_str()]
                .starts_with("LINE(")
                .then(|| (curve, vertex(&r[0]), vertex(&r[1])))
        })
        .unwrap();
    let (a, b) = (point(&pa), point(&pb));
    let along = (b - a).unit_or_zero();
    let diag = Vec3::new(1.0, 1.0, 1.0);
    let off = (diag - along * diag.dot(along)).unit_or_zero();
    // A quadratic's middle lies halfway to its middle control point.
    let m = (a + b) * 0.5 + off * (2.0 * d);
    let old = format!("{curve}={}", ents[curve.as_str()]);
    text.replace(
        &old,
        &format!(
            "{curve}=B_SPLINE_CURVE_WITH_KNOTS('',2,({pa},#99990,{pb}),.UNSPECIFIED.,.F.,.F.,\
             (3,3),(0.,1.),.UNSPECIFIED.);"
        ),
    )
    .replace(
        "ENDSEC;\nEND-ISO",
        &format!(
            "#99990=CARTESIAN_POINT('',({:?},{:?},{:?}));\nENDSEC;\nEND-ISO",
            m.x, m.y, m.z
        ),
    )
}

#[test]
fn an_edge_off_its_faces_past_the_files_uncertainty_takes_a_curve_between_them() {
    // The bent edge's middle lies 0.1 mm off the block's edge along a unit vector across both
    // faces' normals, so off each plane by 0.1/√2, more than the file's 0.05 mm, and healing may
    // move it only 0.05 mm. Between two planes that cross, the curve midway is their line: the
    // block comes back exact, with no local tolerance.
    let b = block("b", [0.0; 3], [10.0, 6.0, 3.0]);
    let text = with_uncertainty(&with_edge_bent(&b, 0.1), 0.05);
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(check::check(&imp.brep).is_empty());
    assert_eq!(imp.brep.tolerant(), (vec![], vec![]));
    assert!((measure::volume(&imp.brep) - 180.0).abs() < 1e-9);
}

#[test]
fn local_tolerances_are_written_as_the_files_uncertainty_and_read_back() {
    let imp = rounds_held_apart(0.4);
    let text = write(&imp.brep, "rb").unwrap();
    assert!(text.contains("UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.3"));
    let back = read(text.as_bytes(), "imp").unwrap();
    assert!(back.rejected.is_empty(), "{:?}", back.rejected);
    assert_eq!(back.brep.tolerant(), imp.brep.tolerant());
    let (v0, v1) = (measure::volume(&imp.brep), measure::volume(&back.brep));
    assert!((v1 - v0).abs() <= 1e-9 * v0, "{v1} vs {v0}");
}

#[test]
fn booleans_pass_tolerant_solids_they_do_not_reach() {
    let imp = rounds_held_apart(0.4).brep;
    let far = block("far", [20.0, 0.0, 0.0], [21.0, 1.0, 1.0]);
    let both = boolean::union(&imp, &far).unwrap();
    assert!(check::check(&both).is_empty());
    assert_eq!(both.tolerant().1.len(), 4);
}

#[test]
fn a_cut_away_from_tolerant_entities_stays_exact() {
    // A pocket in the block's top, 4 mm from the round: its edges and vertices are kernel-made
    // and touch nothing tolerant, so only the round's 4 edges and 4 vertices keep 0.3 mm.
    let imp = rounds_held_apart(0.4).brep;
    let pocket = block("pocket", [2.0, 2.0, 2.0], [4.0, 3.0, 4.0]);
    let cut = boolean::difference(&imp, &pocket).unwrap();
    assert!(check::check(&cut).is_empty());
    let (vs, es) = cut.tolerant();
    assert_eq!((vs.len(), es.len()), (4, 4));
    // The pocket takes 2 x 1 x 1 mm³ of exact material.
    let lost = measure::volume(&imp) - measure::volume(&cut);
    assert!((lost - 2.0).abs() < 1e-9, "{lost}");
}

#[test]
fn a_cut_through_tolerant_edges_gives_its_vertices_their_tolerance() {
    // A slab z 1..2 across the whole block cuts the round's two tolerant lines at z = 1 and 2: the
    // four new vertices on them inherit the lines' 0.3 mm (D. J. Jackson, "Boundary
    // representation modelling with local tolerances", Solid Modeling '95), the lines split into
    // four tolerant pieces, and the two arcs keep theirs. What remains is two solids.
    let imp = rounds_held_apart(0.4).brep;
    let slab = block("slab", [-1.0, -1.0, 1.0], [12.0, 8.0, 2.0]);
    let cut = boolean::difference(&imp, &slab).unwrap();
    assert!(check::check(&cut).is_empty());
    assert_eq!(cut.solid_ids().count(), 2);
    let (vs, es) = cut.tolerant();
    assert_eq!((vs.len(), es.len()), (8, 6));
    for t in vs.iter().map(|x| x.1).chain(es.iter().map(|x| x.1)) {
        assert!((t - 0.3).abs() < 1e-6, "{t}");
    }
    // The slab's part and the rest make up the whole, to within the area its faces bound by
    // tolerant edges can differ by: the slab's two faces cross the 1.2 mm round's 0.2 mm gap.
    let common = boolean::intersection(&imp, &slab).unwrap();
    assert!(check::check(&common).is_empty());
    let whole = measure::volume(&imp);
    let parts = measure::volume(&cut) + measure::volume(&common);
    assert!(
        (parts - whole).abs() <= 2.0 * 0.3 * 1.2,
        "{parts} vs {whole}"
    );
}

#[test]
fn a_tolerant_solid_and_its_copy_within_tolerance_are_one_solid() {
    let imp = rounds_held_apart(0.4).brep;
    let mut copy = imp.clone();
    copy.transform(&crate::math::Transform::translation(Vec3::new(
        1e-9, 0.0, 0.0,
    )));
    let u = boolean::union(&imp, &copy).unwrap();
    assert_eq!(u.face_count(), imp.face_count());
    assert_eq!(u.tolerant(), imp.tolerant());
    let i = boolean::intersection(&imp, &copy).unwrap();
    assert_eq!(measure::volume(&i), measure::volume(&imp));
    let d = boolean::difference(&imp, &copy).unwrap();
    assert_eq!(d.solid_ids().count(), 0);
}

/// A file of `data` lines, from another system.
fn step_file(data: &str) -> String {
    format!(
        "ISO-10303-21;\nHEADER;\nFILE_NAME('a','',(''),(''),'','other','');\nENDSEC;\nDATA;\n\
         {data}\nENDSEC;\nEND-ISO-10303-21;\n"
    )
}

#[test]
fn faceted_solids_read_their_polygons() {
    let corners = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 3.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 4.0),
        (2.0, 0.0, 4.0),
        (2.0, 3.0, 4.0),
        (0.0, 3.0, 4.0),
    ];
    // Each side counter-clockwise seen from outside, with its outward normal.
    let sides: [([usize; 4], (f64, f64, f64)); 6] = [
        ([0, 3, 2, 1], (0.0, 0.0, -1.0)),
        ([4, 5, 6, 7], (0.0, 0.0, 1.0)),
        ([0, 1, 5, 4], (0.0, -1.0, 0.0)),
        ([2, 3, 7, 6], (0.0, 1.0, 0.0)),
        ([1, 2, 6, 5], (1.0, 0.0, 0.0)),
        ([0, 4, 7, 3], (-1.0, 0.0, 0.0)),
    ];
    let mut d = String::new();
    for (i, c) in corners.iter().enumerate() {
        d += &format!(
            "#{}=CARTESIAN_POINT('',({:?},{:?},{:?}));\n",
            1 + i,
            c.0,
            c.1,
            c.2
        );
    }
    let mut faces = Vec::new();
    for (k, (idx, n)) in sides.iter().enumerate() {
        let b = 100 + 10 * k;
        let o = corners[idx[0]];
        let pts: Vec<String> = idx.iter().map(|i| format!("#{}", 1 + i)).collect();
        let x = if n.0 == 0.0 {
            "(1.,0.,0.)"
        } else {
            "(0.,1.,0.)"
        };
        d += &format!("#{b}=POLY_LOOP('',({}));\n", pts.join(","));
        d += &format!("#{}=FACE_OUTER_BOUND('',#{b},.T.);\n", b + 1);
        d += &format!(
            "#{}=CARTESIAN_POINT('',({:?},{:?},{:?}));\n",
            b + 2,
            o.0,
            o.1,
            o.2
        );
        d += &format!(
            "#{}=DIRECTION('',({:?},{:?},{:?}));\n",
            b + 3,
            n.0,
            n.1,
            n.2
        );
        d += &format!("#{}=DIRECTION('',{x});\n", b + 4);
        d += &format!(
            "#{}=AXIS2_PLACEMENT_3D('',#{},#{},#{});\n",
            b + 5,
            b + 2,
            b + 3,
            b + 4
        );
        d += &format!("#{}=PLANE('',#{});\n", b + 6, b + 5);
        d += &format!("#{}=FACE_SURFACE('',(#{}),#{},.T.);\n", b + 7, b + 1, b + 6);
        faces.push(format!("#{}", b + 7));
    }
    d += &format!(
        "#200=CLOSED_SHELL('',({}));\n#201=FACETED_BREP('',#200);",
        faces.join(",")
    );
    let imp = read(step_file(&d).as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(check::check(&imp.brep).is_empty());
    assert_eq!(imp.brep.face_count(), 6);
    assert!((measure::volume(&imp.brep) - 24.0).abs() < 1e-12);
}

/// A cone of base radius 6 and height 5 whose sloped face closes on its apex by a vertex loop,
/// as many systems write it, on `surface`.
fn cone_with_vertex_loop(surface: &str) -> String {
    step_file(&format!(
        "#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=CIRCLE('',#4,6.);
#6=CARTESIAN_POINT('',(6.,0.,0.));
#7=VERTEX_POINT('',#6);
#8=EDGE_CURVE('',#7,#7,#5,.T.);
#9=CARTESIAN_POINT('',(0.,0.,5.));
#10=VERTEX_POINT('',#9);
#11=DIRECTION('',(0.,0.,-1.));
#12=AXIS2_PLACEMENT_3D('',#1,#11,#3);
#13={surface};
#14=PLANE('',#12);
#20=ORIENTED_EDGE('',*,*,#8,.F.);
#21=EDGE_LOOP('',(#20));
#22=FACE_OUTER_BOUND('',#21,.T.);
#23=ADVANCED_FACE('',(#22),#14,.T.);
#30=ORIENTED_EDGE('',*,*,#8,.T.);
#31=EDGE_LOOP('',(#30));
#32=FACE_OUTER_BOUND('',#31,.T.);
#33=VERTEX_LOOP('',#10);
#34=FACE_BOUND('',#33,.T.);
#35=ADVANCED_FACE('',(#32,#34),#13,.T.);
#40=CLOSED_SHELL('',(#23,#35));
#41=MANIFOLD_SOLID_BREP('',#40);"
    ))
}

/// A block whose top face has a further bound: a point (a `VERTEX_LOOP`), or a slit, a line inside
/// it the bound runs along and back. Each marks where something touched the face, and is kept.
#[test]
fn a_point_or_a_slit_bounding_a_face_is_kept() {
    let text = write(&block("b", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]), "part").unwrap();
    let top = text
        .lines()
        .find(|l| l.contains("ADVANCED_FACE('b.end'"))
        .unwrap()
        .to_string();
    let n = text.matches("\n#").count() + 100;
    let with = |extra: String, bound: usize| {
        let (head, tail) = top.split_once("),").unwrap();
        let face = format!("{head},#{bound}),{tail}");
        text.replace(&top, &face)
            .replace("ENDSEC;\nEND-ISO", &format!("{extra}ENDSEC;\nEND-ISO"))
    };
    let point = format!(
        "#{n}=CARTESIAN_POINT('',(5.,5.,10.));\n#{}=VERTEX_POINT('',#{n});\n\
         #{}=VERTEX_LOOP('',#{});\n#{}=FACE_BOUND('',#{},.T.);\n",
        n + 1,
        n + 2,
        n + 1,
        n + 3,
        n + 2
    );
    let slit = format!(
        "#{n}=CARTESIAN_POINT('',(3.,5.,10.));\n#{a}=VERTEX_POINT('',#{n});\n\
         #{b}=CARTESIAN_POINT('',(7.,5.,10.));\n#{c}=VERTEX_POINT('',#{b});\n\
         #{d}=DIRECTION('',(1.,0.,0.));\n#{e}=VECTOR('',#{d},1.);\n#{f}=LINE('',#{n},#{e});\n\
         #{g}=EDGE_CURVE('',#{a},#{c},#{f},.T.);\n#{h}=ORIENTED_EDGE('',*,*,#{g},.T.);\n\
         #{i}=ORIENTED_EDGE('',*,*,#{g},.F.);\n#{j}=EDGE_LOOP('',(#{h},#{i}));\n\
         #{k}=FACE_BOUND('',#{j},.T.);\n",
        a = n + 1,
        b = n + 2,
        c = n + 3,
        d = n + 4,
        e = n + 5,
        f = n + 6,
        g = n + 7,
        h = n + 8,
        i = n + 9,
        j = n + 10,
        k = n + 11
    );
    for (file, vertices, edges) in [(with(point, n + 3), 9, 13), (with(slit, n + 11), 10, 13)] {
        let back = read(file.as_bytes(), "imp").unwrap();
        assert!(back.rejected.is_empty(), "{:?}", back.rejected);
        let b = back.brep;
        assert!(check::check(&b).is_empty(), "{:?}", check::check(&b));
        assert_eq!(b.face_count(), 6);
        assert_eq!(
            (b.vertex_ids().count(), b.edge_ids().count()),
            (vertices, edges)
        );
        assert!((measure::volume(&b) - 1000.0).abs() < 1e-9);
        assert!((measure::area(&b) - 600.0).abs() < 1e-9);
    }
}

/// A ball of radius 5 written as one face bounded only by a `VERTEX_LOOP` at its south pole: a
/// closed surface needs no edge, and the loop names a point of it.
#[test]
fn a_ball_bounded_by_a_vertex_loop_alone_is_read_whole() {
    let text = step_file(
        "#1=CARTESIAN_POINT('',(1.,2.,3.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=SPHERICAL_SURFACE('',#4,5.);
#6=CARTESIAN_POINT('',(1.,2.,-2.));
#7=VERTEX_POINT('',#6);
#8=VERTEX_LOOP('',#7);
#9=FACE_BOUND('',#8,.T.);
#10=ADVANCED_FACE('',(#9),#5,.T.);
#11=CLOSED_SHELL('',(#10));
#12=MANIFOLD_SOLID_BREP('',#11);",
    );
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(check::check(&imp.brep).is_empty());
    let want = 4.0 / 3.0 * PI * 125.0;
    assert!((measure::volume(&imp.brep) - want).abs() < 1e-9 * want);
    assert!((measure::area(&imp.brep) - 100.0 * PI).abs() < 1e-9 * 100.0 * PI);
}

#[test]
fn vertex_loops_close_cones_on_their_apex() {
    let cone = cone_with_vertex_loop("CONICAL_SURFACE('',#12,6.,0.876058050598193)");
    let imp = read(cone.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(check::check(&imp.brep).is_empty());
    let want = 60.0 * PI;
    assert!((measure::volume(&imp.brep) - want).abs() < 1e-9 * want);
    // On a cylinder the vertex is no apex, so it would be a point the face holds, but it is off
    // the surface: refused by name.
    let bad = cone_with_vertex_loop("CYLINDRICAL_SURFACE('',#12,6.)");
    let imp = read(bad.as_bytes(), "imp").unwrap();
    assert_eq!(imp.rejected.len(), 1);
    assert!(
        imp.rejected[0].contains("VERTEX_LOOP"),
        "{:?}",
        imp.rejected
    );
}

/// A solid of revolution about z whose side is one face on `surface` bounded by two circles and no
/// seam, as some systems write a band: the circle of radius `rb` at height `zb` and of `rt` at
/// `zt`, each closed by a disc, with the top circle's vertex turned by `turn` from the bottom's.
fn seamless_band(surface: &str, (zb, rb): (f64, f64), (zt, rt): (f64, f64), turn: f64) -> String {
    let (c, s) = (turn.cos(), turn.sin());
    step_file(&format!(
        "#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=DIRECTION('',(0.,0.,-1.));
#10=CARTESIAN_POINT('',(0.,0.,{zb:?}));
#11=AXIS2_PLACEMENT_3D('',#10,#2,#3);
#12=CIRCLE('',#11,{rb:?});
#13=CARTESIAN_POINT('',({rb:?},0.,{zb:?}));
#14=VERTEX_POINT('',#13);
#15=EDGE_CURVE('',#14,#14,#12,.T.);
#16=AXIS2_PLACEMENT_3D('',#10,#5,#3);
#17=PLANE('',#16);
#20=CARTESIAN_POINT('',(0.,0.,{zt:?}));
#21=AXIS2_PLACEMENT_3D('',#20,#2,#3);
#22=CIRCLE('',#21,{rt:?});
#23=CARTESIAN_POINT('',({:?},{:?},{zt:?}));
#24=VERTEX_POINT('',#23);
#25=EDGE_CURVE('',#24,#24,#22,.T.);
#26=PLANE('',#21);
#30={surface};
#40=ORIENTED_EDGE('',*,*,#15,.F.);
#41=EDGE_LOOP('',(#40));
#42=FACE_OUTER_BOUND('',#41,.T.);
#43=ADVANCED_FACE('',(#42),#17,.T.);
#50=ORIENTED_EDGE('',*,*,#25,.T.);
#51=EDGE_LOOP('',(#50));
#52=FACE_OUTER_BOUND('',#51,.T.);
#53=ADVANCED_FACE('',(#52),#26,.T.);
#60=ORIENTED_EDGE('',*,*,#15,.T.);
#61=EDGE_LOOP('',(#60));
#62=FACE_BOUND('',#61,.T.);
#63=ORIENTED_EDGE('',*,*,#25,.F.);
#64=EDGE_LOOP('',(#63));
#65=FACE_BOUND('',#64,.T.);
#66=ADVANCED_FACE('',(#62,#65),#30,.T.);
#70=CLOSED_SHELL('',(#43,#53,#66));
#71=MANIFOLD_SOLID_BREP('',#70);",
        rt * c,
        rt * s,
    ))
}

#[test]
fn seamless_bands_of_revolution_are_given_a_seam() {
    // Cylinder r 4 from z 0 to 5: 80π. Sphere r 5 between z ±3: π∫(25 − z²)dz = 132π. The outer
    // half of a torus R 10 r 2 and the cylinder r 10 inside it: 400π, plus by Pappus the half
    // disc's 2π about its centroid at 10 + 8/3π.
    let torus = 400.0 * PI + TAU * (10.0 + 8.0 / (3.0 * PI)) * TAU;
    let cases = [
        (
            "CYLINDRICAL_SURFACE('',#4,4.)",
            (0.0, 4.0),
            (5.0, 4.0),
            80.0 * PI,
        ),
        (
            "SPHERICAL_SURFACE('',#4,5.)",
            (-3.0, 4.0),
            (3.0, 4.0),
            132.0 * PI,
        ),
        (
            "TOROIDAL_SURFACE('',#4,10.,2.)",
            (-2.0, 10.0),
            (2.0, 10.0),
            torus,
        ),
    ];
    for (surface, bottom, top, want) in cases {
        // Vertices on one line along the surface, and turned apart, which splits an edge.
        for turn in [0.0, 1.0] {
            let imp = read(seamless_band(surface, bottom, top, turn).as_bytes(), "imp").unwrap();
            assert!(
                imp.rejected.is_empty(),
                "{surface} {turn}: {:?}",
                imp.rejected
            );
            assert!(check::check(&imp.brep).is_empty(), "{surface} {turn}");
            let got = measure::volume(&imp.brep);
            assert!(
                (got - want).abs() < 1e-9 * want,
                "{surface} {turn}: {got} vs {want}"
            );
        }
    }
}

/// The part `above` or below the plane at height `z` of a solid on `surface`, as a disc and one
/// face bounded by the circle of radius √(25 − z²) there and no seam. `#4` frames along z at the
/// origin, `#6` along x, and `#11` along z at the circle.
fn cap(surface: &str, z: f64, above: bool) -> String {
    let r = (25.0 - z * z).sqrt();
    let (disc_axis, disc_use, cap_use) = if above {
        ("#16", ".F.", ".T.")
    } else {
        ("#11", ".T.", ".F.")
    };
    step_file(&format!(
        "#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=DIRECTION('',(0.,0.,-1.));
#6=AXIS2_PLACEMENT_3D('',#1,#3,#7);
#7=DIRECTION('',(0.,1.,0.));
#10=CARTESIAN_POINT('',(0.,0.,{z:?}));
#11=AXIS2_PLACEMENT_3D('',#10,#2,#3);
#12=CIRCLE('',#11,{r:?});
#13=CARTESIAN_POINT('',({r:?},0.,{z:?}));
#14=VERTEX_POINT('',#13);
#15=EDGE_CURVE('',#14,#14,#12,.T.);
#16=AXIS2_PLACEMENT_3D('',#10,#5,#3);
#17=PLANE('',{disc_axis});
#30={surface};
#40=ORIENTED_EDGE('',*,*,#15,{disc_use});
#41=EDGE_LOOP('',(#40));
#42=FACE_OUTER_BOUND('',#41,.T.);
#43=ADVANCED_FACE('',(#42),#17,.T.);
#60=ORIENTED_EDGE('',*,*,#15,{cap_use});
#61=EDGE_LOOP('',(#60));
#62=FACE_OUTER_BOUND('',#61,.T.);
#66=ADVANCED_FACE('',(#62),#30,.T.);
#70=CLOSED_SHELL('',(#43,#66));
#71=MANIFOLD_SOLID_BREP('',#70);"
    ))
}

#[test]
fn caps_without_a_seam_close_on_the_pole_or_apex_they_cover() {
    // A cap of height h on a sphere of radius 5: πh²(15 − h)/3. Cut at z 3, the small cap above
    // has h 2, and the rest of the ball, 500π/3, lies below. Cut at z 0, a hemisphere: 250π/3,
    // its circle through the poles of a sphere framed along x. A cone of radius 4 at z 3 and half
    // angle atan(4/3) has its apex at the origin, a drill's point: π·16·3/3 = 16π.
    let (along_z, along_x) = ("SPHERICAL_SURFACE('',#4,5.)", "SPHERICAL_SURFACE('',#6,5.)");
    let cone = format!("CONICAL_SURFACE('',#11,4.,{:?})", (4.0f64 / 3.0).atan());
    let cases = [
        (along_z, 3.0, true, 52.0 * PI / 3.0),
        (along_z, 3.0, false, 448.0 * PI / 3.0),
        (along_x, 0.0, true, 250.0 * PI / 3.0),
        (along_x, 0.0, false, 250.0 * PI / 3.0),
        (cone.as_str(), 3.0, false, 16.0 * PI),
    ];
    for (surface, z, above, want) in cases {
        let imp = read(cap(surface, z, above).as_bytes(), "imp").unwrap();
        assert!(
            imp.rejected.is_empty(),
            "{surface} {z} {above}: {:?}",
            imp.rejected
        );
        assert!(check::check(&imp.brep).is_empty(), "{surface} {z} {above}");
        let got = measure::volume(&imp.brep);
        assert!(
            (got - want).abs() < 1e-9 * want,
            "{surface} {z} {above}: {got} vs {want}"
        );
    }
}

/// The text of `b` with its first cylinder written as an offset by `d` of a cylinder `d`
/// thinner, or of `basis` where given.
fn with_offset_cylinder(b: &Brep, d: f64, basis: Option<&str>) -> String {
    let text = write(b, "c").unwrap().replace("'springrep'", "'other'");
    let l = text
        .lines()
        .find(|l| l.contains("=CYLINDRICAL_SURFACE("))
        .unwrap();
    let (id, rest) = l.split_once('=').unwrap();
    let (head, r) = rest.rsplit_once(',').unwrap();
    let r: f64 = r.trim_end_matches(");").parse().unwrap();
    let inner = match basis {
        Some(s) => s.to_string(),
        None => format!("{head},{:?})", r - d),
    };
    text.replace(l, &format!("{id}=OFFSET_SURFACE('',#99990,{d:?},.F.);"))
        .replace(
            "ENDSEC;\nEND-ISO",
            &format!("#99990={inner};\nENDSEC;\nEND-ISO"),
        )
}

#[test]
fn offset_surfaces_read_as_the_surface_they_are() {
    let rod = extrude(&Profile::circle(Vec2::ZERO, 4.0), &Frame::WORLD, 5.0, "c").unwrap();
    let imp = read(with_offset_cylinder(&rod, 0.5, None).as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(imp.healed.is_empty());
    let want = measure::volume(&rod);
    assert!((measure::volume(&imp.brep) - want).abs() < 1e-9 * want);
    // An offset with no closed form is refused by name, never approximated.
    let ext = "SURFACE_OF_LINEAR_EXTRUSION('',#99991,#99992);\n\
               #99991=LINE('',#99993,#99992);\n#99992=VECTOR('',#99994,1.);\n\
               #99993=CARTESIAN_POINT('',(0.,0.,0.));\n#99994=DIRECTION('',(0.,0.,1.))";
    let imp = read(with_offset_cylinder(&rod, 0.5, Some(ext)).as_bytes(), "imp").unwrap();
    assert_eq!(imp.rejected.len(), 1);
    assert!(
        imp.rejected[0].contains("OFFSET_SURFACE"),
        "{:?}",
        imp.rejected
    );
}

#[test]
fn scaled_transformations_place_scaled_copies() {
    let part = write(&block("b", [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]), "blk").unwrap();
    let data = part
        .split("DATA;\n")
        .nth(1)
        .unwrap()
        .split("ENDSEC;")
        .next()
        .unwrap();
    let rep = data
        .lines()
        .find(|l| l.contains("ADVANCED_BREP_SHAPE_REPRESENTATION"))
        .and_then(|l| l[1..].split('=').next())
        .unwrap();
    let extra = format!(
        "#9000=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));
#9010=PRODUCT('asm','asm','',(#3));
#9011=PRODUCT_DEFINITION_FORMATION('','',#9010);
#9012=PRODUCT_DEFINITION('design','',#9011,#6);
#9013=PRODUCT_DEFINITION_SHAPE('','',#9012);
#9014=SHAPE_REPRESENTATION('asm',(#9016),#9099);
#9015=SHAPE_DEFINITION_REPRESENTATION(#9013,#9014);
#9016=MAPPED_ITEM('',#9017,#9018);
#9017=REPRESENTATION_MAP(#9020,#{rep});
#9018=CARTESIAN_TRANSFORMATION_OPERATOR_3D('','',#9023,#9024,#9031,2.,#9022);
#9020=AXIS2_PLACEMENT_3D('',#9021,#9022,#9023);
#9021=CARTESIAN_POINT('',(0.,0.,0.));
#9022=DIRECTION('',(0.,0.,1.));
#9023=DIRECTION('',(1.,0.,0.));
#9024=DIRECTION('',(0.,1.,0.));
#9031=CARTESIAN_POINT('',(10.,0.,0.));
#9099=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#9000)) \
       REPRESENTATION_CONTEXT('','3D'));
"
    );
    let text = format!(
        "ISO-10303-21;\nHEADER;\nFILE_NAME('a','',(''),(''),'','other','');\nENDSEC;\nDATA;\n\
         {data}{extra}ENDSEC;\nEND-ISO-10303-21;\n"
    );
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    let b = imp.brep;
    // The part itself, and a copy twice its size whose corner sits at x = 10.
    assert_eq!(b.solid_ids().count(), 2);
    assert!((measure::volume(&b) - 9.0 * 6.0).abs() < 1e-9);
    let bb = measure::bbox(&b);
    assert!(
        (bb.max - Vec3::new(12.0, 4.0, 6.0)).max_abs() < 1e-9,
        "{bb:?}"
    );
}

#[test]
fn touching_placed_solids_check_valid_each_but_not_as_one() {
    let mut b = block("a", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    b.append(&block("b", [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]));
    let text = write(&b, "board").unwrap();
    let imp = read(text.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert_eq!(imp.brep.solid_ids().count(), 2);
    assert!(!check::check(&imp.brep).is_empty());
    assert!(check::check_each(&imp.brep).is_empty());
}

/// Where a pcurve puts the edge at its parameter `t`, on the face's surface as STEP parameterizes
/// it: a cone by height along its STEP axis, every other surface as the kernel does.
fn on_step_surface(s: &Surface, c2: &super::write::Curve2, t: f64) -> Vec3 {
    let uv = match c2 {
        super::write::Curve2::Line { origin, dir } => *origin + *dir * t,
        super::write::Curve2::Spline {
            degree,
            knots,
            ctrl,
            weights,
        } => {
            let pts: Vec<Vec3> = ctrl.iter().map(|p| Vec3::new(p.x, p.y, 0.0)).collect();
            let c = crate::geom::NurbsCurve::new(*degree, knots.clone(), pts, weights.clone());
            let p = c.unwrap().eval(t);
            Vec2::new(p.x, p.y)
        }
    };
    match s {
        Surface::Cone(c) => {
            let (fr, a, _) = super::write::step_cone(c).unwrap();
            let (u, h) = (uv.x, uv.y);
            fr.origin + (fr.x * u.cos() + fr.y * u.sin()) * (c.radius + h * a.tan()) + fr.z * h
        }
        _ => s.eval(uv.x, uv.y),
    }
}

/// Every coedge's pcurve, read on its face's surface as STEP parameterizes it, follows its edge
/// within the tolerance, and each edge is written with one per face, a seam with both.
fn assert_pcurves_follow(b: &Brep) {
    for c in b.coedge_ids() {
        let edge = b.edge(b.coedge(c).edge);
        let Some(curve) = &edge.curve else { continue };
        let s = &b.face(b.coedge_face(c)).surface;
        let c2 = super::write::pcurve(b, c).unwrap_or_else(|| panic!("{} has none", s.kind()));
        let (t0, t1) = edge.range;
        for i in 0..=200 {
            let t = t0 + (t1 - t0) * i as f64 / 200.0;
            let d = on_step_surface(s, &c2, t).dist(curve.eval(t));
            assert!(d <= crate::tol::LINEAR, "{} {d:e}", s.kind());
        }
    }
    let text = write(b, "part").unwrap();
    let edges = b.edge_ids().filter(|&e| !b.edge(e).is_degenerate()).count();
    let with = text.matches("=SURFACE_CURVE(").count() + text.matches("=SEAM_CURVE(").count();
    assert_eq!(with, edges);
    assert_eq!(text.matches("=PCURVE(").count(), 2 * edges);
}

#[test]
fn pcurves_follow_their_edges_on_every_surface() {
    // A frustum narrowing up its axis (written about the reversed axis) and one widening, each cut
    // by a tilted block, so their cones carry traced conic sections as well as their own circles.
    for pts in [
        [(0.0, 0.0), (6.0, 0.0), (3.0, 5.0), (0.0, 5.0)],
        [(0.0, 0.0), (3.0, 0.0), (6.0, 5.0), (0.0, 5.0)],
    ] {
        let pts: Vec<Vec2> = pts.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
        let p = Profile::polygon(&pts, &["a", "b", "c", "d"]);
        let cone = revolve(&p, &xz(), Vec3::ZERO, Vec3::Z, TAU, "t").unwrap();
        let mut cut = block("k", [0.0, -10.0, -10.0], [20.0, 10.0, 10.0]);
        cut.transform(&crate::math::Transform::rotation(
            Vec3::new(2.0, 0.0, 2.5),
            Vec3::new(0.3, 1.0, 0.2).unit().unwrap(),
            0.4,
        ));
        let d = boolean::difference(&cone, &cut).unwrap();
        assert_pcurves_follow(&d);
        round_trip(&d);
    }
    // A twisted loft's sides crossed by a rod: traced curves on rational B-spline sides, and the
    // loft's own edges on its plane caps.
    let sq = Profile::rect(Vec2::ZERO, 6.0, 6.0);
    let sections: Vec<crate::loft::Section> = (0..2)
        .map(|i| {
            let th = 0.6 * i as f64;
            crate::loft::Section {
                profile: sq.clone(),
                frame: Frame::new(
                    Vec3::new(0.0, 0.0, 8.0 * i as f64),
                    Vec3::Z,
                    Vec3::new(th.cos(), th.sin(), 0.0),
                )
                .unwrap(),
            }
        })
        .collect();
    let loft = crate::loft::loft(&sections, false, "l").unwrap();
    let rod = extrude(
        &Profile::circle(Vec2::new(4.0, 1.0), 3.0),
        &Frame::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::X, Vec3::Y).unwrap(),
        20.0,
        "r",
    )
    .unwrap();
    assert_pcurves_follow(&boolean::difference(&loft, &rod).unwrap());
}

// Fitted pcurves on a ball and on a rod, each crossed obliquely by another rod: read at three
// points a span, the surface over them strayed 5.33e-8 and 5.35e-8 mm from their edges between
// those points, past the half tolerance the writer promises.
#[test]
fn fitted_pcurves_stay_within_half_the_tolerance_between_their_checks() {
    let rod = |base: [f64; 3], axis: [f64; 3], r: f64| {
        let f = Frame::from_axis(Vec3::from_array(base), Vec3::from_array(axis)).unwrap();
        extrude(&Profile::circle(Vec2::ZERO, 2.0 * r), &f, 40.0, "r").unwrap()
    };
    let shapes = [
        boolean::difference(&ball(6.0), &rod([1.0, -20.0, -15.0], [0.1, 1.0, 0.8], 2.5)),
        boolean::union(
            &rod([0.0, 0.0, -20.0], [0.0, 0.0, 1.0], 4.0),
            &rod([-20.0, 0.5, 0.0], [1.0, 0.0, 0.3], 3.0),
        ),
    ];
    for b in shapes.iter().map(|b| b.as_ref().unwrap()) {
        for c in b.coedge_ids() {
            let edge = b.edge(b.coedge(c).edge);
            let Some(curve) = &edge.curve else { continue };
            let s = &b.face(b.coedge_face(c)).surface;
            let c2 = super::write::pcurve(b, c).unwrap();
            let (t0, t1) = edge.range;
            for i in 0..=1000 {
                let t = t0 + (t1 - t0) * i as f64 / 1000.0;
                let d = on_step_surface(s, &c2, t).dist(curve.eval(t));
                assert!(d <= 0.5 * crate::tol::LINEAR, "{} {d:e}", s.kind());
            }
        }
    }
}

#[test]
fn a_solid_passing_through_itself_is_kept_with_where_and_how_deep() {
    // Two 2 mm blocks overlapping by 1 mm each way, written as two solids, then made one solid whose
    // one shell holds both blocks' faces: it passes through itself where their faces cross, the
    // faces reaching 1 mm either side of each other.
    let mut b = block("a", [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    b.append(&block("b", [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]));
    let text = write(&b, "board").unwrap();
    let shells: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("=CLOSED_SHELL("))
        .collect();
    assert_eq!(shells.len(), 2);
    let id = |l: &str| l[..l.find('=').unwrap()].to_string();
    let faces = |l: &str| l[l.find("(#").unwrap() + 1..l.rfind("))").unwrap()].to_string();
    let merged = shells[0].replace(
        &faces(shells[0]),
        &format!("{},{}", faces(shells[0]), faces(shells[1])),
    );
    let second = text
        .lines()
        .find(|l| l.contains(&format!("MANIFOLD_SOLID_BREP('',{})", id(shells[1]))))
        .unwrap();
    let one: String = text
        .lines()
        .filter(|&l| l != shells[1] && l != second)
        .map(|l| {
            let l = if l == shells[0] {
                merged.clone()
            } else {
                l.to_string()
            };
            l.replace(&format!(",{}", id(second)), "") + "\n"
        })
        .collect();
    let imp = read(one.as_bytes(), "imp").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert_eq!(imp.brep.solid_ids().count(), 1);
    assert!(check::check_structure(&imp.brep).is_empty());
    assert!(!check::check(&imp.brep).is_empty());
    assert!(!imp.crossed.is_empty());
    let deepest = imp.crossed.iter().map(|x| x.depth).fold(0.0, f64::max);
    assert!((deepest - 1.0).abs() < 1e-9, "{deepest}");
    for x in &imp.crossed {
        assert_eq!((x.index, x.step_id), (0, imp.crossed[0].step_id));
        // Every crossing lies on the overlap's box.
        assert!(
            (0..3).all(|k| x.at[k] >= 1.0 - 1e-9 && x.at[k] <= 2.0 + 1e-9),
            "{:?}",
            x.at
        );
    }
}

#[test]
fn a_tessellated_solid_reads_as_the_polyhedron_of_its_triangles() {
    // The corner of the unit cube cut off by x + y + z = 1, its four triangles counter-clockwise
    // seen from outside: volume 1/6.
    let text = "ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('t','',(''),(''),'','','');
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));
ENDSEC;
DATA;
#1=COORDINATES_LIST('',4,((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)));
#2=TRIANGULATED_FACE('',#1,0,(),$,(),((1,3,2),(1,2,4),(1,4,3),(2,3,4)));
#3=TESSELLATED_SOLID('',(#2),$);
ENDSEC;
END-ISO-10303-21;
";
    let imp = read(text.as_bytes(), "t").unwrap();
    assert!(imp.rejected.is_empty(), "{:?}", imp.rejected);
    assert!(check::check(&imp.brep).is_empty());
    assert!((measure::volume(&imp.brep) - 1.0 / 6.0).abs() < 1e-12);
    assert_eq!(imp.brep.face_count(), 4);
}

/// The triangles as loose plane faces, each its own open shell with its own vertices and edges.
fn loose_faces(tris: &[[[f64; 3]; 3]]) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut add = |s: String| {
        lines.push(s);
        lines.len()
    };
    let num = |x: f64| format!("{x:.17E}");
    let pt = |p: [f64; 3]| format!("({},{},{})", num(p[0]), num(p[1]), num(p[2]));
    for t in tris {
        let vs: Vec<usize> = t
            .iter()
            .map(|&p| {
                let c = add(format!("CARTESIAN_POINT('',{})", pt(p)));
                add(format!("VERTEX_POINT('',#{c})"))
            })
            .collect();
        let mut oes = Vec::new();
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let c = add(format!("CARTESIAN_POINT('',{})", pt(a)));
            let dir = add(format!("DIRECTION('',{})", pt(d.map(|x| x / l))));
            let vec = add(format!("VECTOR('',#{dir},1.)"));
            let line = add(format!("LINE('',#{c},#{vec})"));
            let e = add(format!(
                "EDGE_CURVE('',#{},#{},#{line},.T.)",
                vs[k],
                vs[(k + 1) % 3]
            ));
            oes.push(add(format!("ORIENTED_EDGE('',*,*,#{e},.T.)")));
        }
        let lp = add(format!(
            "EDGE_LOOP('',({}))",
            oes.iter()
                .map(|o| format!("#{o}"))
                .collect::<Vec<_>>()
                .join(",")
        ));
        let bound = add(format!("FACE_OUTER_BOUND('',#{lp},.T.)"));
        let (a, b, c) = (t[0], t[1], t[2]);
        let (u, v) = (
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let ul = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let o = add(format!("CARTESIAN_POINT('',{})", pt(a)));
        let z = add(format!("DIRECTION('',{})", pt(n.map(|x| x / nl))));
        let x = add(format!("DIRECTION('',{})", pt(u.map(|x| x / ul))));
        let ax = add(format!("AXIS2_PLACEMENT_3D('',#{o},#{z},#{x})"));
        let plane = add(format!("PLANE('',#{ax})"));
        let face = add(format!("ADVANCED_FACE('',(#{bound}),#{plane},.T.)"));
        add(format!("OPEN_SHELL('',(#{face}))"));
    }
    let mut out = String::from(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
         FILE_NAME('t','',(''),(''),'','','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n",
    );
    for (i, l) in lines.iter().enumerate() {
        out.push_str(&format!("#{}={l};\n", i + 1));
    }
    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    out
}

#[test]
fn loose_faces_a_hair_apart_and_some_inside_out_sew_into_their_solid() {
    // The tetrahedron on the unit cube's corner, each face shifted up to 5e-5 along its normal
    // and two of them written inside out: sewn within 1e-3, it is the tetrahedron, volume 1/6 to
    // what the shifts move it.
    let (o, x, y, z) = ([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    let shift = |t: [[f64; 3]; 3], d: f64| {
        let (u, v) = (
            [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]],
            [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]],
        );
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        t.map(|p| [0, 1, 2].map(|k| p[k] + d * n[k] / l))
    };
    let tris = [
        shift([o, y, x], 3e-5),
        [x, o, z],
        shift([o, z, y], -5e-5),
        [x, z, y],
    ];
    let text = loose_faces(&tris);
    let b = super::sew(text.as_bytes(), 1e-3).unwrap();
    assert!(check::check(&b).is_empty());
    assert_eq!(b.face_count(), 4);
    assert!((measure::volume(&b) - 1.0 / 6.0).abs() < 1e-4);
}

/// A cylinder of radius 2 and height 3 whose side is a rational spline surface, straight along its
/// axis and an exact circle round it, so periodic round it: its side is bounded by the two circles
/// alone, a band with no seam, between two plane ends. `top_first` lists the top circle's bound
/// first, so that the seam's first end comes last along the surface's iso line.
fn spline_band_cylinder(top_first: bool) -> String {
    let (r, h, w) = (2.0, 3.0, std::f64::consts::FRAC_1_SQRT_2);
    let round = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
    ];
    let mut out = String::new();
    let mut ids = [Vec::new(), Vec::new()];
    for (row, z) in [0.0, h].iter().enumerate() {
        for (k, (x, y)) in round.iter().enumerate() {
            let id = 100 + 10 * row + k;
            out += &format!("#{id}=CARTESIAN_POINT('',({},{},{z:?}));\n", x * r, y * r);
            ids[row].push(format!("#{id}"));
        }
    }
    let weights: Vec<String> = (0..9)
        .map(|k| {
            if k % 2 == 0 {
                "1.".into()
            } else {
                format!("{w:?}")
            }
        })
        .collect();
    let row = |i: usize| format!("({})", ids[i].join(","));
    let wrow = format!("({})", weights.join(","));
    out += &format!(
        "#1=(BOUNDED_SURFACE() B_SPLINE_SURFACE(1,2,({},{}),.UNSPECIFIED.,.F.,.T.,.F.) \
         B_SPLINE_SURFACE_WITH_KNOTS((2,2),(3,2,2,2,3),(0.,1.),(0.,0.25,0.5,0.75,1.),.UNSPECIFIED.) \
         GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE(({wrow},{wrow})) \
         REPRESENTATION_ITEM('') SURFACE());\n",
        row(0),
        row(1)
    );
    let bounds = if top_first { "#25,#24" } else { "#24,#25" };
    out += &format!(
        "#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=CARTESIAN_POINT('',(0.,0.,0.));
#5=CARTESIAN_POINT('',(0.,0.,{h:?}));
#6=AXIS2_PLACEMENT_3D('',#4,#2,#3);
#7=AXIS2_PLACEMENT_3D('',#5,#2,#3);
#8=CIRCLE('',#6,{r:?});
#9=CIRCLE('',#7,{r:?});
#10=CARTESIAN_POINT('',({r:?},0.,0.));
#11=CARTESIAN_POINT('',({r:?},0.,{h:?}));
#12=VERTEX_POINT('',#10);
#13=VERTEX_POINT('',#11);
#14=EDGE_CURVE('',#12,#12,#8,.T.);
#15=EDGE_CURVE('',#13,#13,#9,.T.);
#20=ORIENTED_EDGE('',*,*,#14,.T.);
#21=EDGE_LOOP('',(#20));
#22=ORIENTED_EDGE('',*,*,#15,.F.);
#23=EDGE_LOOP('',(#22));
#24=FACE_BOUND('',#21,.T.);
#25=FACE_BOUND('',#23,.T.);
#26=ADVANCED_FACE('',({bounds}),#1,.F.);
#30=ORIENTED_EDGE('',*,*,#14,.F.);
#31=EDGE_LOOP('',(#30));
#32=FACE_OUTER_BOUND('',#31,.T.);
#33=PLANE('',#6);
#34=ADVANCED_FACE('',(#32),#33,.F.);
#40=ORIENTED_EDGE('',*,*,#15,.T.);
#41=EDGE_LOOP('',(#40));
#42=FACE_OUTER_BOUND('',#41,.T.);
#43=PLANE('',#7);
#44=ADVANCED_FACE('',(#42),#43,.T.);
#50=CLOSED_SHELL('',(#26,#34,#44));
#51=MANIFOLD_SOLID_BREP('',#50);
"
    );
    format!("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n{out}ENDSEC;\nEND-ISO-10303-21;\n")
}

#[test]
fn a_band_of_a_spline_surface_closed_round_one_parameter_is_read() {
    // The side's two circles each go once round the surface's period, opposite ways; the reader
    // joins them with a seam along the surface, as it does for a cylinder of its own.
    for top_first in [false, true] {
        let imp = read(spline_band_cylinder(top_first).as_bytes(), "band").unwrap();
        assert!(imp.rejected.is_empty(), "{top_first}: {:?}", imp.rejected);
        // The file is exact: it reads with nothing healed and no tolerance widened.
        assert!(
            imp.healed.is_empty() && imp.tolerant.is_empty(),
            "{top_first}: healed {:?}, tolerant {:?}",
            imp.healed,
            imp.tolerant
        );
        let b = imp.brep;
        assert_eq!(b.solid_ids().count(), 1);
        assert!(check::check(&b).is_empty(), "{:?}", check::check(&b));
        let want = PI * 4.0 * 3.0;
        assert!((measure::volume(&b) - want).abs() < 1e-6 * want);
    }
}
