//! Sweeps that run until they meet a solid: a profile extruded or turned from its sketch plane up
//! to the first face of the solid its sweep crosses, into the solid or out of its material.

use std::f64::consts::TAU;

use super::{extrude, revolve};
use crate::{
    Result, boolean,
    error::invalid,
    math::{Frame, Point3, Vec3},
    measure,
    profile::Profile,
    topo::{Brep, FaceName},
};

/// Which crossing of a solid's boundary ends a sweep up to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Meet {
    /// From outside the solid up to the first face where the sweep enters it: a boss or a rib, to
    /// be joined to the solid. What of the sweep starts inside the solid is left out.
    Into,
    /// From inside the solid's material out through the first face where the sweep leaves it: a
    /// hole, to be cut from the solid. What of the sweep starts outside the solid is left out.
    OutOf,
}

/// The pieces of the long sweep `long` on `meet`'s side of `target` that its start cap reaches
/// before the sweep first crosses `target`'s boundary: those carrying a face of the start cap. A
/// piece that also carries the end cap never crossed it.
fn first_stretch(long: &Brep, target: &Brep, meet: Meet, feature: &str) -> Result<Brep> {
    let pieces = match meet {
        Meet::Into => boolean::difference(long, target)?,
        Meet::OutOf => boolean::intersection(long, target)?,
    };
    let carries = |s, cap: &str| {
        let name = FaceName::new(feature, cap);
        pieces
            .solid_faces(s)
            .iter()
            .any(|&f| pieces.face(f).names.contains(&name))
    };
    let touching: Vec<_> = pieces
        .solid_ids()
        .filter(|&s| carries(s, FaceName::START))
        .collect();
    if touching.is_empty() {
        return invalid("a sweep up to a solid keeps nothing of its start on that side of it");
    }
    if touching.iter().any(|&s| carries(s, FaceName::END)) {
        return invalid("a sweep up to a solid runs past it without crossing it");
    }
    Ok(boolean::solids_of(&pieces, &touching))
}

/// How far from `from` the farthest point of `target`'s box lies, and a millimetre more.
fn reach(target: &Brep, from: Point3) -> f64 {
    let b = measure::bbox(target);
    b.diagonal() + b.center().dist(from) + 1.0
}

/// Sweeps `profile`, drawn on `frame`'s xy plane, along the frame's normal (against it when
/// `backward`) until it first crosses the boundary of `target` the way `meet` says. `start` is the
/// cap on the sketch plane; the far end lies on `target`'s faces. The result is the swept solid
/// alone, to be joined to `target` or cut from it.
pub fn extrude_to_next(
    profile: &Profile,
    frame: &Frame,
    backward: bool,
    target: &Brep,
    meet: Meet,
    feature: &str,
) -> Result<Brep> {
    let length = reach(target, frame.origin);
    let distance = if backward { -length } else { length };
    let long = extrude(profile, frame, distance, feature)?;
    first_stretch(&long, target, meet, feature)
}

/// Turns `profile`, drawn on `frame`'s xy plane, right-handed about the line through `origin` along
/// `dir` (as [`revolve`]) until it first crosses the boundary of `target` the way `meet` says. The
/// turn is tried up to 359°, so the far end of the turn never lies on its own start; `start` is the
/// cap on the sketch plane.
pub fn revolve_to_next(
    profile: &Profile,
    frame: &Frame,
    origin: Point3,
    dir: Vec3,
    target: &Brep,
    meet: Meet,
    feature: &str,
) -> Result<Brep> {
    let long = revolve(profile, frame, origin, dir, TAU * 359.0 / 360.0, feature)?;
    first_stretch(&long, target, meet, feature)
}
