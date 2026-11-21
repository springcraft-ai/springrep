//! A solid cut in two by a plane.

use crate::{
    boolean,
    error::{Error, Result},
    math::{Aabb, Frame, Point3, Vec2, Vec3},
    measure,
    profile::Profile,
    sweep::extrude,
    topo::Brep,
};

/// A box holding everything of `s` on the normal's side of the plane through `origin`, its face
/// on the plane.
pub fn half_space(s: &Brep, origin: Point3, normal: Vec3) -> Result<Brep> {
    // Only how far the box must reach is read from `s`: any box that holds it will do.
    half_space_past(&measure::loose_bbox(s), origin, normal)
}

/// [`half_space`] reaching past everything in the box `b`.
fn half_space_past(b: &Aabb, origin: Point3, normal: Vec3) -> Result<Brep> {
    let n = normal
        .unit()
        .ok_or_else(|| Error::InvalidInput("a plane needs a normal".into()))?;
    let reach = 2.0 * (b.diagonal() + b.center().dist(origin)) + 1.0;
    let frame = Frame::new(origin, n, n.any_perpendicular())?;
    extrude(
        &Profile::rect(Vec2::ZERO, 2.0 * reach, 2.0 * reach),
        &frame,
        reach,
        "half",
    )
}

/// `s` cut by the plane through `origin` square to `normal`: the part on the normal's side, then
/// the part on the other; either may be empty.
pub fn split(s: &Brep, origin: Point3, normal: Vec3) -> Result<(Brep, Brep)> {
    if s.solid_ids().next().is_none() {
        return Ok((s.clone(), s.clone()));
    }
    let h = half_space(s, origin, normal)?;
    Ok((boolean::intersection(s, &h)?, boolean::difference(s, &h)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::section_area;

    /// A prism on the polygon `pts`, `h` tall, its sides named `e0`, `e1`, … from the first edge.
    fn prism(pts: &[[f64; 2]], h: f64) -> Brep {
        let pts: Vec<Vec2> = pts.iter().map(|p| Vec2::new(p[0], p[1])).collect();
        let names: Vec<String> = (0..pts.len()).map(|i| format!("e{i}")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        extrude(&Profile::polygon(&pts, &names), &Frame::WORLD, h, "p").unwrap()
    }

    fn close(got: f64, want: f64) {
        assert!(
            (got - want).abs() < 1e-7 * want.max(1.0),
            "{got} against {want}"
        );
    }

    #[test]
    fn a_half_space_sized_from_the_loose_box_cuts_the_same_parts() {
        // A slab whose top humps up inside its rim: its top's patch boxes, and so its loose
        // box, rise past the hump itself. Cut through the hump, level and aslant.
        let k = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let heights: Vec<Vec<f64>> = (0..4)
            .map(|i| {
                let inner = |j: usize| (1..3).contains(&i) && (1..3).contains(&j);
                (0..4).map(|j| if inner(j) { 3.0 } else { 1.0 }).collect()
            })
            .collect();
        let slab = crate::primitive::slab(&Frame::WORLD, [4.0, 3.0], (&k, &k), &heights, "s");
        let slab = slab.unwrap();
        let (tight, loose) = (measure::bbox(&slab), measure::loose_bbox(&slab));
        assert!(loose.max.z > tight.max.z + 1e-3, "{tight:?} {loose:?}");
        assert!((0..3).all(|k| loose.min[k] <= tight.min[k] && loose.max[k] >= tight.max[k]));
        let cuts = [
            (Point3::new(0.0, 0.0, 1.4), Vec3::Z),
            (Point3::new(2.0, 1.5, 1.2), Vec3::new(0.3, -0.2, 1.0)),
            (Point3::new(1.3, 0.0, 0.0), Vec3::new(-1.0, 0.1, 0.0)),
        ];
        for (o, n) in cuts {
            let parts = |b: &Aabb| {
                let h = half_space_past(b, o, n).unwrap();
                let kept = boolean::intersection(&slab, &h).unwrap();
                let rest = boolean::difference(&slab, &h).unwrap();
                (kept, rest)
            };
            let (from_tight, from_loose) = (parts(&tight), parts(&loose));
            let (kept, rest) = split(&slab, o, n).unwrap();
            let whole = measure::volume(&slab);
            for (got, want) in [(&kept, &from_tight.0), (&rest, &from_tight.1)] {
                assert_eq!(got.face_count(), want.face_count());
                assert_eq!(got.solid_ids().count(), want.solid_ids().count());
                close(measure::volume(got), measure::volume(want));
                close(measure::area(got), measure::area(want));
                let (a, b) = (measure::bbox(got), measure::bbox(want));
                assert!(a.min.dist(b.min) < 1e-7 && a.max.dist(b.max) < 1e-7);
            }
            // The box `half_space` reads is the loose one, and the two parts are the slab.
            close(measure::volume(&from_loose.0), measure::volume(&kept));
            close(measure::volume(&kept) + measure::volume(&rest), whole);
            assert!(crate::check::check(&kept).is_empty() && crate::check::check(&rest).is_empty());
        }
    }

    #[test]
    fn a_block_split_across_keeps_each_part_and_its_cut() {
        // 4 × 3 × 2 cut at x = 1: 6 on the far side, 18 on the near, a 3 × 2 cut.
        let b = prism(&[[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]], 2.0);
        let (o, n) = (Point3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let (near, far) = split(&b, o, n).unwrap();
        close(measure::volume(&near), 6.0);
        close(measure::volume(&far), 18.0);
        close(section_area(&b, o, n).unwrap(), 6.0);
    }
}
