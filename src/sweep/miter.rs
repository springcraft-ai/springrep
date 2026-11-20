//! A path of legs laid out frame by frame, and its straight legs cut back to the mitered corners
//! beside them.

use std::f64::consts::PI;

use super::{Leg, extrude, turn_axis};
use crate::{
    Result,
    boolean::split,
    error::invalid,
    math::{Frame, Point3, Transform, Vec3},
    profile::Profile,
    tol,
    topo::Brep,
};

/// A mitered corner: its point, and the normal of its plane, toward the leg after it.
#[derive(Clone, Copy)]
struct Corner {
    at: Point3,
    normal: Vec3,
    /// How far a straight leg must run past the corner for the plane to cut its whole section.
    past: f64,
}

/// The frame each leg starts from, and the mitered corners beside each straight leg.
pub(super) struct Path {
    pub(super) frames: Vec<Frame>,
    before: Vec<Option<Corner>>,
    after: Vec<Option<Corner>>,
}

impl Path {
    pub(super) fn of(profile: &Profile, frame: &Frame, legs: &[Leg]) -> Result<Path> {
        if legs.is_empty() {
            return invalid("a sweep's path needs a leg");
        }
        let mut frames = Vec::with_capacity(legs.len() + 1);
        let mut fr = *frame;
        for leg in legs {
            frames.push(fr);
            fr = match *leg {
                Leg::Line { length } => fr.translated(fr.z * length),
                Leg::Arc {
                    radius,
                    angle,
                    axis,
                } => {
                    let n = turn_axis(axis, &fr)?;
                    let centre = fr.origin + n.cross(fr.z) * radius;
                    fr.transformed(&Transform::rotation(centre, n, angle))
                }
                Leg::Turn { angle, axis } | Leg::Miter { angle, axis } => {
                    let n = turn_axis(axis, &fr)?;
                    fr.transformed(&Transform::rotation(fr.origin, n, angle))
                }
            };
        }
        let last = legs.len() - 1;
        let closed = matches!(legs[last], Leg::Miter { .. })
            && fr.origin.dist(frame.origin) <= tol::LINEAR
            && (fr.z - frame.z).norm() <= tol::ANGULAR
            && (fr.x - frame.x).norm() <= tol::ANGULAR;
        frames.push(fr);
        let reach = profile
            .loops
            .iter()
            .flat_map(|lp| lp.polygon())
            .map(|p| p.norm())
            .fold(0.0, f64::max);
        let mut before = vec![None; legs.len()];
        let mut after = vec![None; legs.len()];
        for (m, leg) in legs.iter().enumerate() {
            let Leg::Miter { angle, .. } = *leg else {
                continue;
            };
            let next = if m < last {
                m + 1
            } else if closed {
                0
            } else {
                return invalid("a mitered corner needs a straight leg after it");
            };
            let Some(prev) = m.checked_sub(1) else {
                return invalid("a mitered corner needs a straight leg before it");
            };
            let straight = |i: usize| matches!(legs[i], Leg::Line { .. });
            if !(straight(prev) && straight(next)) {
                return invalid("a mitered corner joins two straight legs");
            }
            if angle.is_nan() || angle.abs() >= PI * (1.0 - 1e-6) {
                return invalid("a mitered corner turns less than half a turn");
            }
            let normal = (frames[m].z + frames[m + 1].z)
                .unit()
                .expect("a turn of less than half a turn");
            // The plane leans half the turn from the section, so a point of the section `reach`
            // from the path lies up to reach·tan(θ/2) along it; a leg runs on a reach more.
            let past = reach * (1.0 + (0.5 * angle.abs()).tan());
            let corner = Corner {
                at: frames[m].origin,
                normal,
                past,
            };
            after[prev] = Some(corner);
            before[next] = Some(corner);
        }
        Ok(Path {
            frames,
            before,
            after,
        })
    }

    /// Straight leg `i`, `length` long: extruded from its frame, or, beside a miter, on past the
    /// corner and cut back to the miter's plane.
    pub(super) fn line(
        &self,
        profile: &Profile,
        i: usize,
        length: f64,
        feature: &str,
    ) -> Result<Brep> {
        let fr = self.frames[i];
        let (before, after) = (self.before[i], self.after[i]);
        let back = before.map_or(0.0, |c| c.past);
        let on = after.map_or(0.0, |c| c.past);
        let mut b = extrude(
            profile,
            &fr.translated(fr.z * -back),
            length + back + on,
            feature,
        )?;
        if let Some(c) = before {
            b = split(&b, c.at, c.normal)?.0;
        }
        if let Some(c) = after {
            b = split(&b, c.at, c.normal)?.1;
        }
        Ok(b)
    }
}
