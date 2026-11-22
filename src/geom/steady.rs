//! What a spline surface swept by a steady motion keeps of how it was made: the curve swept and
//! the motion, a slide, a turn or a screw. The surface at `(u, v)` is the curve's point at `v`
//! after `u` of the motion, to within the fit the surface was built to, all over its own domain.
//! So the surface is a union of the motion's orbits, lines, circles or helices, one through each
//! point of the curve, and meets another surface made of the same orbits along whole orbits.

use super::NurbsCurve;
use crate::math::{Point3, Transform, Vec3};

/// A steady motion about the line through `origin` along the unit `axis`: after `u` of it a point
/// has turned by `turn·u` about the line and moved `rise·u` along it. A slide has no `turn`, a
/// turn no `rise`, and a screw both.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Steady {
    pub origin: Point3,
    pub axis: Vec3,
    pub turn: f64,
    pub rise: f64,
}

impl Steady {
    /// The rigid move after `u` of the motion.
    pub(crate) fn pose(&self, u: f64) -> Transform {
        Transform::rotation(self.origin, self.axis, self.turn * u)
            .then(&Transform::translation(self.axis * (self.rise * u)))
    }

    /// The motion that carries `t` of a point wherever this one carries the point.
    pub(crate) fn transformed(&self, t: &Transform) -> Steady {
        Steady {
            origin: t.point(self.origin),
            axis: t.vector(self.axis),
            ..*self
        }
    }
}

/// The curve a swept spline surface was swept from, its parameter the surface's `v`, and the
/// motion that swept it, its parameter the surface's `u`.
#[derive(Clone, Debug)]
pub(crate) struct Swept {
    pub generator: NurbsCurve,
    pub motion: Steady,
}

impl Swept {
    pub(crate) fn transformed(&self, t: &Transform) -> Swept {
        Swept {
            generator: self.generator.transformed(t),
            motion: self.motion.transformed(t),
        }
    }
}
