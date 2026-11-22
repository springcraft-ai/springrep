//! Everything a solid passes through as it moves rigidly (doc/design.md, "Sweeps of solids"): the
//! body at the start, and what the region of its surface that the motion carries outward sweeps.
//! That region's boundary is its characteristic curves, where the velocity lies in the surface,
//! and the sharp edges between it and the rest; under a slide, a turn or a screw it is fixed on the
//! body, and the solid's boundary is that boundary swept, capped by the rest of the body at the
//! start and the region at the end.

mod contour;
mod cut;
mod grazing;
mod motion;
mod moving;
mod orbit;
mod product;
mod sheet;
#[cfg(test)]
mod tests;
mod trim;

pub use motion::Motion;

use std::f64::consts::{PI, TAU};

use crate::{Error, Result, boolean, error::invalid, math::Transform, tol, topo::Brep};

#[cfg(test)]
thread_local! {
    /// Set by a test to build a whole turn's second half as the first is built.
    static SWEEP_BOTH_HALVES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The solid `body` passes through as `motion` carries it, `τ` from 0 to 1: the union of
/// `M(τ)·body`. The faces the body's faces are cut from keep their names; each face a curve sweeps
/// is `feature.side`.
pub fn solid(body: &Brep, motion: &Motion, feature: &str) -> Result<Brep> {
    motion.validate()?;
    let mut solids = body.solid_ids();
    let (Some(s), None) = (solids.next(), solids.next()) else {
        return invalid("a sweep moves one solid");
    };
    if body.solid(s).shells.len() != 1 {
        return Err(Error::Unsupported("a sweep of a solid with a void".into()));
    }
    // A whole turn ends where it starts, so its two ends would be one; it is two half turns. Turns
    // about one line compose, so the second half's sweep is the first's turned half a turn.
    if let Motion::Turn {
        origin,
        axis,
        from,
        to,
    } = *motion
        && (to - from).abs() >= TAU - tol::ANGULAR
    {
        let step = PI.copysign(to - from);
        let half = |a: f64| Motion::Turn {
            origin,
            axis,
            from: a,
            to: a + step,
        };
        let a = solid(body, &half(from), feature)?;
        #[cfg(test)]
        let again = SWEEP_BOTH_HALVES.get();
        #[cfg(not(test))]
        let again = false;
        let b = if again {
            solid(body, &half(from + step), feature)?
        } else {
            let mut b = a.clone();
            b.transform(&Transform::rotation(origin, axis, step));
            b
        };
        let facts = crate::blend::shared_edges(&a, &b);
        return boolean::boolean_with(&a, &b, boolean::Op::Union, &facts);
    }
    let (Some(steady), Some(twist)) = (sheet::Steady::of(motion), motion.steady()) else {
        return moving::sweep(body, motion, feature);
    };
    let field = move |p| twist.at(p);
    let cut = cut::cut(body, &field, steady.held())?;
    // Where the envelope folds, the faces are trimmed to what the moving body never reaches into.
    if product::folded(&cut, &steady)
        && let Some(reach) = trim::Reach::new(body, motion, (0.0, 1.0))
    {
        return product::product(&cut, &steady, feature, None, Some(&reach));
    }
    match product::product(&cut, &steady, feature, None, None) {
        Err(Error::InvalidResult(_)) => in_parts(&cut, &steady, feature),
        other => other,
    }
}

/// The sweep as the union of the body at the start and what each piece of face the motion carries
/// outward sweeps on its own, for where the body's other faces at the start and the pieces at the
/// end cross each other: a plane turned past the line it grazes along crosses where it started
/// beyond that line, and a sheet swept from a concave edge runs into the body.
fn in_parts(cut: &cut::Cut, steady: &sheet::Steady, feature: &str) -> Result<Brep> {
    let mut out = product::start(cut, steady)?;
    for (i, part) in cut.parts.iter().enumerate() {
        if part.front {
            let piece = product::product(cut, steady, feature, Some(&[i]), None)?;
            // The piece's sweep starts on the body's face, and grazes it along the curves that
            // bound the piece there.
            let facts = crate::blend::shared_edges(&out, &piece);
            out = boolean::boolean_with(&out, &piece, boolean::Op::Union, &facts)?;
        }
    }
    Ok(out)
}
