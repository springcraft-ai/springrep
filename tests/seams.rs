//! Faces whose seam does not run straight: two lengths of one cylinder joined, each with its seam
//! at its own angle, are one face round the cylinder whose seam jogs where they meet. Its outline
//! in the parameter plane spans more than one period, and a hole in it lies in one of them.

use std::f64::consts::PI;

use springrep::{
    boolean::{difference, union},
    check,
    math::{Frame, Vec3},
    measure,
    mesh::{self, MeshOptions},
    primitive, step,
    topo::Brep,
};

/// A rod of radius 5 from z = 0 to 30: 10 mm with its seam at +x under 20 mm with its seam at −x.
fn rod() -> Brep {
    let low = primitive::cylinder(&Frame::WORLD, 5.0, 10.0, "low").unwrap();
    let frame = Frame::new(Vec3::new(0.0, 0.0, 10.0), Vec3::Z, -Vec3::X).unwrap();
    let high = primitive::cylinder(&frame, 5.0, 20.0, "high").unwrap();
    union(&low, &high).unwrap()
}

/// A blind hole of radius 0.5, 2 mm deep, drilled radially at height 20 and `angle` round from +x.
fn drill(angle: f64) -> Brep {
    let out = Vec3::new(angle.cos(), angle.sin(), 0.0);
    let frame = Frame::new(Vec3::new(0.0, 0.0, 20.0) + out * 3.0, out, Vec3::Z).unwrap();
    primitive::cylinder(&frame, 0.5, 4.0, "drill").unwrap()
}

#[test]
fn a_hole_in_a_face_whose_seam_jogs_meshes_wherever_round_it_is() {
    let rod = rod();
    assert!(check::check(&rod).is_empty());
    // The two sides are one face: with the two ends, three.
    assert_eq!(rod.face_count(), 3);
    let opt = MeshOptions::new(0.01, 0.2);
    // Either side of each seam: the upper length's outline lies a period on from the lower's
    // for half the way round, so a hole there is not where its own angle alone would put it.
    for angle in [0.2, -0.2, PI - 0.2, PI + 0.2] {
        let cut = difference(&rod, &drill(angle)).unwrap();
        assert!(check::check(&cut).is_empty(), "{angle}");
        // As a file gives it, with each edge's place on the face found on its own.
        let text = step::write(&cut, "rod").unwrap();
        let drilled = step::read_part(text.as_bytes(), "in").unwrap();
        let m = mesh::tessellate(&drilled, &opt).unwrap_or_else(|e| panic!("{angle}: {e}"));
        let solid = drilled.solid_ids().next().unwrap();
        let report = mesh::report(&drilled, &m, solid);
        assert!(report.watertight(), "{angle}: {report:?}");
        // Chords 0.01 mm deep leave out under 2 · 0.01 / 5 of the rod's π · 5² · 30.
        let exact = measure::volume(&drilled);
        assert!((exact - PI * 750.0).abs() < 2.0, "{angle}: {exact}");
        assert!(
            (exact * (1.0 - 0.004)..exact).contains(&report.volume),
            "{angle}: {report:?}"
        );
    }
}
