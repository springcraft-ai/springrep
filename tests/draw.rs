//! The outlines of curved faces in a hidden-line drawing.

use springrep::{
    draw::hidden_lines,
    math::{Frame, Point3, Vec3},
    primitive::cylinder,
    tol,
};

#[test]
fn a_side_view_shows_both_generators_of_a_cylinder() {
    let (radius, height) = (5.0, 2.0);
    let b = cylinder(&Frame::WORLD, radius, height, "post").unwrap();
    let drawing = hidden_lines(&b, Vec3::Y, 0.1).unwrap();
    let generators: Vec<_> = drawing
        .visible
        .iter()
        .filter(|line| {
            let Some((a, z)) = line.first().zip(line.last()) else {
                return false;
            };
            ((a.z - z.z).abs() - height).abs() <= tol::LINEAR
        })
        .collect();
    assert_eq!(generators.len(), 2);
    // Looking along y, the grazing points lie at (±r, 0) all the way up the cylinder.
    for x in [-radius, radius] {
        let foot = Point3::new(x, 0.0, 0.0);
        let tip = Point3::new(x, 0.0, height);
        assert!(generators.iter().any(|line| {
            let (a, z) = (line[0], line[line.len() - 1]);
            (a.dist(foot) <= tol::LINEAR && z.dist(tip) <= tol::LINEAR)
                || (z.dist(foot) <= tol::LINEAR && a.dist(tip) <= tol::LINEAR)
        }));
    }
}
