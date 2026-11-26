//! A face may hold a point inside it, a `VERTEX_LOOP`, where another body touches it: a ball on
//! its race. On a face round an axis that point can lie on the face's seam, and its mesh must
//! then run through the point on both sides of the seam.
//!
//! The fixtures are written for springrep entity by entity from the ISO 10303-42 definitions,
//! with no CAD program. Each is a cylinder of radius 5 from z = 0 to 10, its side bounded by its
//! two circles, whose vertices are at +x, and no seam, holding a point at half height:
//! - `on`: the point at +x, on the line through the circles' vertices, where the side's seam goes.
//! - `off`: the point at −x.

use std::f64::consts::PI;

use springrep::{
    check, measure,
    mesh::{self, MeshOptions},
    step,
};

fn reads_and_meshes(text: &[u8], name: &str, x: f64) {
    let (r, h) = (5.0_f64, 10.0);
    let volume = PI * r * r * h;
    let area = 2.0 * PI * r * r + 2.0 * PI * r * h;
    let imp = step::read(text, name).unwrap();
    assert!(imp.rejected.is_empty(), "{name}: {:?}", imp.rejected);
    let b = imp.brep;
    assert_eq!(b.solid_ids().count(), 1, "{name}");
    assert!(
        check::check(&b).is_empty(),
        "{name}: {:?}",
        check::check(&b)
    );
    let (a, v) = measure::area_and_volume(&b);
    assert!((v - volume).abs() < 1e-9 * volume, "{name}: volume {v}");
    assert!((a - area).abs() < 1e-9 * area, "{name}: area {a}");
    let linear = 0.01;
    let m = mesh::tessellate(&b, &MeshOptions::within(linear))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let s = b.solid_ids().next().unwrap();
    let rep = mesh::report(&b, &m, s);
    assert!(rep.watertight(), "{name}: {rep:?}");
    assert_eq!(rep.degenerate_triangles, 0, "{name}");
    // Every chord of the circles and the side lies within the deflection inside the cylinder,
    // and every mesh point on it, so the mesh holds the cylinder of radius r − deflection.
    let least = PI * (r - linear).powi(2) * h;
    let got = rep.volume.abs();
    assert!(got <= volume && got >= least, "{name}: mesh volume {got}");
    let held = [x, 0.0, 0.5 * h];
    let corner = m.triangles.iter().flatten().any(|&i| {
        let p = m.positions[i as usize];
        (p.x - held[0]).abs() + (p.y - held[1]).abs() + (p.z - held[2]).abs() <= 1e-12
    });
    assert!(corner, "{name}: no triangle has the held point as a corner");
}

#[test]
fn a_point_held_on_the_seam_meshes_through_the_point() {
    let text = include_bytes!("fixtures/vertex-loop-on-seam.step");
    reads_and_meshes(text, "vertex-loop-on-seam", 5.0);
}

#[test]
fn a_point_held_off_the_seam_meshes_through_the_point() {
    let text = include_bytes!("fixtures/vertex-loop-off-seam.step");
    reads_and_meshes(text, "vertex-loop-off-seam", -5.0);
}
