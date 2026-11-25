//! A face round a cylinder that the file bounds by two loops round its axis and no seam: the
//! reader gives it a seam along the cylinder from one loop to the other. Where the line it would
//! take runs through a hole in the face, or along an edge of a hole or of the loops, the face's
//! boundary would overlap itself, and the mesher would halve the overlapping chords without end.
//! The seam runs clear of them.
//!
//! The fixtures are written for springrep entity by entity from the ISO 10303-42 definitions,
//! with no CAD program. Each is a rod of radius 2 from z = 0 to 10, its side bounded by its bottom
//! circle, whose vertex is at +x, and by a top loop:
//! - `notch-*`: the half of the rod at y < 0, turned `phi` about the axis, cut away between z = 4
//!   and 6. The side's top loop is its top circle, and the notch's window is a hole in it: two
//!   half circles and two straight edges along the axis, at `phi` and `phi + pi`.
//!   - `on`: phi = 0, an edge of the hole on the line through the circles' vertices.
//!   - `across`: phi = pi / 2, that line through the middle of the hole.
//!   - `off`: phi = −pi / 2, the hole on the far side.
//! - `slot-*`: the half at y < 0 cut away above z = 8. The side's top loop has two straight edges
//!   along the axis, and is written from the top of one of them.
//!   - `on`: phi = 0, that edge on the line through the bottom circle's vertex.
//!   - `near`: phi = 3e-8, that edge 6e-8 from the line, within the linear tolerance of 1e-7, so
//!     that a seam on the line would lie along it all the same.
//!   - `off`: phi = pi / 2.
//! - `spline-notch-*`: the notched rod with its side a B-spline surface that is the cylinder
//!   exactly: degree 1 along the axis, and round it the rational quadratic circle of nine control
//!   points, weights 1 and sqrt(2) / 2 on knots 0, 1/4, 1/2, 3/4, 1 (Piegl and Tiller, The NURBS
//!   Book, 2nd ed., section 7.5), whose quarters of the parameter are the circle's quadrants. A
//!   spline band's seam runs in the middle of the widest gap between the circles' vertices, both at
//!   parameter 0, so at the angle pi.
//!   - `on`: phi = 0, an edge of the hole on that line, the other on the vertices' line.
//!   - `across`: phi = −pi / 2, that line through the middle of the hole.
//!   - `off`: phi = pi / 2, the hole on the far side, across the vertices' line.

use std::f64::consts::PI;

use springrep::{
    check, measure,
    mesh::{self, MeshOptions},
    step,
};

/// Reads `text`, checks it whole, with the volume and area of its closed form, and meshes it
/// watertight within the deflection.
fn reads_and_meshes(text: &[u8], name: &str, area: f64) {
    let volume = 36.0 * PI;
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
    // Every mesh point lies within the deflection of the boundary, so the mesh's volume is
    // within the area times the deflection of the solid's.
    assert!(
        (rep.volume - volume).abs() <= area * linear,
        "{name}: mesh volume {}",
        rep.volume
    );
}

#[test]
fn a_seam_runs_clear_of_a_hole_on_its_line() {
    let area = 48.0 * PI + 8.0;
    reads_and_meshes(include_bytes!("fixtures/notch-on-seam.step"), "on", area);
    reads_and_meshes(
        include_bytes!("fixtures/notch-across-seam.step"),
        "across",
        area,
    );
    reads_and_meshes(include_bytes!("fixtures/notch-off-seam.step"), "off", area);
}

#[test]
fn a_seam_runs_clear_of_a_loop_edge_on_its_line() {
    let area = 44.0 * PI + 8.0;
    reads_and_meshes(include_bytes!("fixtures/slot-on-seam.step"), "on", area);
    reads_and_meshes(include_bytes!("fixtures/slot-near-seam.step"), "near", area);
    reads_and_meshes(include_bytes!("fixtures/slot-off-seam.step"), "off", area);
}

#[test]
fn a_spline_band_seam_runs_clear_of_a_hole_on_its_line() {
    // The surface is the cylinder, so the closed forms are the notched rod's.
    let area = 48.0 * PI + 8.0;
    reads_and_meshes(
        include_bytes!("fixtures/spline-notch-on-seam.step"),
        "spline on",
        area,
    );
    reads_and_meshes(
        include_bytes!("fixtures/spline-notch-across-seam.step"),
        "spline across",
        area,
    );
    reads_and_meshes(
        include_bytes!("fixtures/spline-notch-off-seam.step"),
        "spline off",
        area,
    );
}
