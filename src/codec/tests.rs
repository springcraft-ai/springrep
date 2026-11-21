use std::collections::BTreeSet;
use std::f64::consts::FRAC_1_SQRT_2;

use super::*;
use crate::blend::fillet;
use crate::boolean::difference;
use crate::math::Point3;
use crate::measure;
use crate::profile::Profile;
use crate::sweep::extrude;
use crate::topo::Brep;

fn bytes<T: Codec>(x: &T) -> Vec<u8> {
    let mut w = Writer::default();
    x.put(&mut w);
    w.0
}

fn back<T: Codec>(b: &[u8]) -> Option<T> {
    let mut r = Reader::new(b);
    let x = T::get(&mut r)?;
    (r.remaining() == 0).then_some(x)
}

/// `x` read back from its bytes equals it, and writes the same bytes again: every number kept to
/// its last bit, signed zeros included.
fn round_trips<T: Codec + PartialEq + std::fmt::Debug>(x: &T) {
    let b = bytes(x);
    let y: T = back(&b).expect("reads back");
    assert_eq!(&y, x);
    assert_eq!(bytes(&y), b);
}

fn p(x: f64, y: f64, z: f64) -> Point3 {
    Vec3::new(x, y, z)
}

fn frame() -> Frame {
    Frame::new(p(1.0, -2.0, 0.5), p(0.0, 0.6, 0.8), p(1.0, 0.0, 0.0)).unwrap()
}

fn spline() -> BSpline {
    BSpline::interpolate(&[
        p(0.0, 0.0, 0.0),
        p(1.0, 2.0, -0.0),
        p(3.0, 1.0, 1.0),
        p(4.0, 0.0, 0.0),
    ])
    .unwrap()
}

fn curves() -> Vec<Curve> {
    let closed = BSpline::interpolate_periodic(&[
        p(1.0, 0.0, 0.0),
        p(0.0, 1.0, 0.0),
        p(-1.0, 0.0, 0.0),
        p(0.0, -1.0, 0.3),
    ])
    .unwrap();
    let quarter = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![p(1.0, 0.0, 0.0), p(1.0, 1.0, 0.0), p(0.0, 1.0, 0.0)],
        Some(vec![1.0, FRAC_1_SQRT_2, 1.0]),
    )
    .unwrap();
    vec![
        Curve::Line(Line {
            origin: p(0.1, 0.2, 0.3),
            dir: p(0.0, 0.0, -1.0),
        }),
        Curve::Circle(Circle {
            frame: frame(),
            radius: 2.5,
        }),
        Curve::Ellipse(Ellipse {
            frame: frame(),
            major: 3.0,
            minor: 1.0 / 3.0,
        }),
        Curve::BSpline(spline()),
        Curve::BSpline(closed),
        Curve::Intersection(Box::new(IntersectionCurve {
            approx: spline(),
            surfaces: [
                Surface::Plane(Plane { frame: frame() }),
                Surface::Sphere(Sphere {
                    frame: frame(),
                    radius: 4.0,
                }),
            ],
        })),
        Curve::Nurbs(Box::new(quarter)),
    ]
}

fn surfaces() -> Vec<Surface> {
    let patch = NurbsSurface::new(
        (1, 2),
        (vec![0.0, 0.0, 1.0, 1.0], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        (2, 3),
        (0..6)
            .map(|i| p(f64::from(i % 2), f64::from(i / 2), 0.1 * f64::from(i)))
            .collect(),
        Some(vec![1.0, 2.0, 1.0, 0.5, 1.0, 1.0]),
    )
    .unwrap();
    let line = |o: Point3, d: Point3| Curve::Line(Line { origin: o, dir: d });
    vec![
        Surface::Plane(Plane { frame: frame() }),
        Surface::Cylinder(Cylinder {
            frame: frame(),
            radius: 1.25,
        }),
        Surface::Cone(Cone {
            frame: frame(),
            radius: 2.0,
            half_angle: 0.3,
        }),
        Surface::Sphere(Sphere {
            frame: frame(),
            radius: 0.75,
        }),
        Surface::Torus(Torus {
            frame: frame(),
            major: 5.0,
            minor: 1.5,
        }),
        Surface::Extrusion(Box::new(Extrusion {
            curve: Curve::BSpline(spline()),
            dir: p(0.0, 0.0, 1.0),
        })),
        Surface::Revolution(Box::new(Revolution {
            curve: line(p(2.0, 0.0, 0.0), p(0.0, 0.0, 1.0)),
            origin: p(0.0, 0.0, 0.0),
            axis: p(0.0, 0.0, 1.0),
        })),
        Surface::Ruled(Box::new(Ruled {
            a: line(p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0)),
            a_range: (0.0, 2.0),
            b: Curve::BSpline(spline()),
            b_range: spline().domain(),
        })),
        Surface::Nurbs(Box::new(patch)),
    ]
}

/// Which variant `c` is; a curve kind added to the kernel fails to compile here until it is.
fn curve_kind(c: &Curve) -> usize {
    match c {
        Curve::Line(_) => 0,
        Curve::Circle(_) => 1,
        Curve::Ellipse(_) => 2,
        Curve::BSpline(_) => 3,
        Curve::Intersection(_) => 4,
        Curve::Nurbs(_) => 5,
    }
}

fn surface_kind(s: &Surface) -> usize {
    match s {
        Surface::Plane(_) => 0,
        Surface::Cylinder(_) => 1,
        Surface::Cone(_) => 2,
        Surface::Sphere(_) => 3,
        Surface::Torus(_) => 4,
        Surface::Extrusion(_) => 5,
        Surface::Revolution(_) => 6,
        Surface::Ruled(_) => 7,
        Surface::Nurbs(_) => 8,
    }
}

#[test]
fn every_curve_and_surface_kind_reads_back_exactly() {
    let cs = curves();
    let ss = surfaces();
    assert_eq!(cs.iter().map(curve_kind).collect::<BTreeSet<_>>().len(), 6);
    assert_eq!(
        ss.iter().map(surface_kind).collect::<BTreeSet<_>>().len(),
        9
    );
    for c in &cs {
        round_trips(c);
    }
    for s in &ss {
        round_trips(s);
    }
    // A periodic spline stays periodic, and a NURBS keeps the period it was found to have.
    let Curve::BSpline(closed) = back::<Curve>(&bytes(&cs[4])).unwrap() else {
        panic!("a spline")
    };
    assert!(closed.is_periodic());
}

/// A 20 × 20 × 10 block with a Ø6 hole through it and its top's two edges along y rounded R1.
fn drilled_and_rounded() -> Brep {
    let block = extrude(
        &Profile::rect(Vec2::ZERO, 20.0, 20.0),
        &at_z(0.0),
        10.0,
        "block",
    )
    .unwrap();
    let hole = extrude(&Profile::circle(Vec2::ZERO, 6.0), &at_z(0.0), 10.0, "hole").unwrap();
    let drilled = difference(&block, &hole).unwrap();
    let edges = [("block.end", "block.u_min"), ("block.end", "block.u_max")];
    fillet(&drilled, &edges, 1.0, "round").unwrap()
}

fn at_z(z: f64) -> Frame {
    Frame::new(p(0.0, 0.0, z), p(0.0, 0.0, 1.0), p(1.0, 0.0, 0.0)).unwrap()
}

#[test]
fn a_built_solid_reads_back_with_its_ids_names_and_memos() {
    let b = drilled_and_rounded();
    let volume = measure::volume(&b);
    // Asking for the volume's sign leaves each face's share in its memo.
    for s in b.solid_ids() {
        assert!(measure::solid_volume_sign(&b, s) > 0.0);
    }
    let back: Brep = back(&bytes(&b)).unwrap();
    assert_eq!(back, b);
    assert_eq!(bytes(&back), bytes(&b));
    // The volume each face's memo holds came across, and gives the same total to the last bit.
    assert!(back.face_ids().all(|f| back.memo(f).volume.get().is_some()));
    assert_eq!(measure::volume(&back).to_bits(), volume.to_bits());
}

#[test]
fn memos_come_back_shared_as_they_were_with_the_trims_a_boolean_read() {
    // A cut reads the trims of the faces it meets into their memos; a later boolean takes those
    // instead of working them out again. Here a 4 mm square post through the block's corner.
    let b = drilled_and_rounded();
    let square = Profile::rect(Vec2::new(10.0, 10.0), 4.0, 4.0);
    let post = extrude(&square, &at_z(-5.0), 20.0, "post").unwrap();
    difference(&b, &post).unwrap();
    let read: Vec<_> = b
        .face_ids()
        .filter(|&f| b.memo(f).images.get().is_some())
        .collect();
    assert!(!read.is_empty());
    let pair = vec![b.clone(), b];
    let back: Vec<Brep> = back(&bytes(&pair)).unwrap();
    for &f in &read {
        assert!(back[0].memo(f).images.get().is_some());
        assert!(Arc::ptr_eq(
            &back[0].shared_memo(f),
            &back[1].shared_memo(f)
        ));
    }
    assert_eq!(bytes(&back), bytes(&pair));
}

#[test]
fn local_tolerances_read_back() {
    let mut b = drilled_and_rounded();
    let v = b.vertex_ids().next().unwrap();
    let e = b.edge_ids().last().unwrap();
    b.set_vertex_tolerance(v, 2e-4);
    b.set_edge_tolerance(e, 1e-4);
    round_trips(&b);
}

#[test]
fn an_arena_whose_ids_point_past_its_own_entities_is_refused() {
    let mut other = Brep::new();
    other.add_vertex(p(0.0, 0.0, 0.0));
    let far = other.add_vertex(p(1.0, 0.0, 0.0));
    let mut b = Brep::new();
    let near = b.add_vertex(p(0.0, 0.0, 0.0));
    b.add_edge(
        Curve::Line(Line {
            origin: p(0.0, 0.0, 0.0),
            dir: p(1.0, 0.0, 0.0),
        }),
        (0.0, 1.0),
        near,
        far,
    );
    assert!(back::<Brep>(&bytes(&b)).is_none());
}

#[test]
fn damaged_bytes_read_back_as_nothing() {
    let b = bytes(&surfaces());
    for n in 0..b.len() {
        assert!(
            back::<Vec<Surface>>(&b[..n]).is_none(),
            "{n} bytes of {}",
            b.len()
        );
    }
    // A length no remaining bytes could hold is refused before anything is allocated for it.
    let mut huge = Writer::default();
    u64::MAX.put(&mut huge);
    assert!(back::<Vec<f64>>(&huge.0).is_none());
    // A variant tag past the last kind.
    assert!(back::<Curve>(&[9]).is_none());
}
