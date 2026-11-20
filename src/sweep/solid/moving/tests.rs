use std::f64::consts::PI;

use super::charmap::CharMap;
use super::{Ctx, Span};
use crate::{
    math::{Frame, Vec3},
    primitive::block,
    sweep::solid::Motion,
};

// Two cones rolling on each other: turned by t about x and by t/2 about z, the body turns at t = 0
// about the line along (2, 0, 1), and a box whose ends are square to that line holds them in their
// own planes there. Each face's grazing curve at the end of the span stopping there is the limit
// of its curves just inside, read off cubic extrapolation from four places a thousandth apart.
#[test]
fn grazing_curves_run_on_to_a_span_end_where_faces_stand_still() {
    let g = Vec3::new(2.0, 0.0, 1.0).unit().unwrap();
    let d = Vec3::new(-1.0, 0.0, 2.0).unit().unwrap();
    let frame = Frame::new(g * 25.0 - d * 2.0 - Vec3::Y * 1.5, g, d).unwrap();
    let body = block(&frame, Vec3::new(4.0, 3.0, 10.0), "tooth").unwrap();
    let a = PI / 18.0;
    let motion = Motion::Then(
        Box::new(Motion::Turn {
            origin: Vec3::ZERO,
            axis: Vec3::X,
            from: -a,
            to: a,
        }),
        Box::new(Motion::Turn {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            from: -a / 2.0,
            to: a / 2.0,
        }),
    );
    let span = Span {
        at: (0.0, 0.5),
        held: (false, true),
    };
    let ctx = Ctx::new(&body, &motion, span).unwrap();
    let h = 1e-3;
    let mut checked = 0;
    for f in 0..ctx.faces.len() {
        if ctx.faces[f].still {
            continue;
        }
        for map in CharMap::of(&ctx, f).unwrap() {
            let (lo, hi) = map.range();
            let x = 0.5 * (lo + hi);
            let Some(end) = map.point(0.5, x) else {
                continue;
            };
            let at = |k: f64| map.point(0.5 - k * h, x).unwrap();
            let guess = at(1.0) * 4.0 - at(2.0) * 6.0 + at(3.0) * 4.0 - at(4.0);
            assert!(end.dist(guess) < 1e-9, "face {f}: {end:?} {guess:?}");
            checked += 1;
        }
    }
    assert!(checked >= 4, "{checked}");
}
