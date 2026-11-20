//! Rigid motions over one parameter `τ ∈ [0, 1]`: where each places a body, and how fast each point
//! of the body moves.

use crate::{
    Result,
    error::invalid,
    geom::Curve,
    math::{Point3, Transform, Vec3},
    sweep::Helix,
    tol,
};

/// How a body moves as `τ` runs from 0 to 1. The body is given where it is drawn, and the motion
/// places it at every `τ`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Motion {
    /// Carried straight by the vector, from where it is drawn.
    Slide(Vec3),
    /// Turned right-handed about the line through `origin` along `axis`, from angle `from` to
    /// angle `to` (radians).
    Turn {
        origin: Point3,
        axis: Vec3,
        from: f64,
        to: f64,
    },
    /// Carried along `screw` from its turn `from` to its turn `to` (radians).
    Screw { screw: Helix, from: f64, to: f64 },
    /// Carried by a frame riding `path` from `range.0` to `range.1`: its origin on the path, its x
    /// along the path's tangent, its z as near `up` as square to the tangent allows. The body is
    /// drawn in the frame at `range.0`.
    Along {
        path: Curve,
        range: (f64, f64),
        up: Vec3,
    },
    /// Two motions over one parameter: the first places the body, and the second carries the
    /// whole of it, so at `τ` the body is where the first puts it, moved by the second.
    Then(Box<Motion>, Box<Motion>),
}

/// The velocity of a rigid motion as a field: a point at `x` moves at `omega × x + v`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Twist {
    pub omega: Vec3,
    pub v: Vec3,
}

impl Twist {
    pub(crate) fn at(&self, x: Point3) -> Vec3 {
        self.omega.cross(x) + self.v
    }
}

/// The frame riding a path at `t`: its origin, its axes, and their rates along `t`.
struct Riding {
    origin: Point3,
    axes: [Vec3; 3],
    rates: [Vec3; 3],
    speed: Vec3,
}

fn riding(path: &Curve, up: Vec3, t: f64) -> Option<Riding> {
    let (c, d1, d2) = path.d2(t);
    let len = d1.norm();
    let x = d1.unit()?;
    let dx = (d2 - x * d2.dot(x)) / len;
    let w = up - x * up.dot(x);
    let wn = w.norm();
    let z = w.unit()?;
    let dw = -(dx * up.dot(x) + x * up.dot(dx));
    let dz = (dw - z * dw.dot(z)) / wn;
    let y = z.cross(x);
    let dy = dz.cross(x) + z.cross(dx);
    Some(Riding {
        origin: c,
        axes: [x, y, z],
        rates: [dx, dy, dz],
        speed: d1,
    })
}

fn frame_transform(r: &Riding) -> Transform {
    let [x, y, z] = r.axes;
    Transform {
        r: [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]],
        t: r.origin,
    }
}

impl Motion {
    /// Refuses a motion that cannot place a body: no direction, no length, or a path whose frame
    /// is undefined somewhere along it.
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Motion::Slide(d) => {
                if !(d.is_finite() && d.norm() > tol::LINEAR) {
                    return invalid("a slide moves by a vector longer than the tolerance");
                }
            }
            Motion::Turn { axis, from, to, .. } => {
                if axis.unit().is_none() {
                    return invalid("a turn needs an axis");
                }
                if !(from.is_finite() && to.is_finite() && (to - from).abs() > tol::ANGULAR) {
                    return invalid("a turn turns through an angle");
                }
            }
            Motion::Screw { screw, from, to } => {
                if screw.axis.unit().is_none() {
                    return invalid("a screw needs an axis");
                }
                if !(from.is_finite() && to.is_finite() && (to - from).abs() > tol::ANGULAR) {
                    return invalid("a screw motion turns through an angle");
                }
            }
            Motion::Along { path, range, up } => {
                if range.1.partial_cmp(&range.0) != Some(std::cmp::Ordering::Greater) {
                    return invalid("a frame along a path runs over a range of the path");
                }
                for k in 0..=64 {
                    let t = range.0 + (range.1 - range.0) * k as f64 / 64.0;
                    if riding(path, *up, t).is_none() {
                        return invalid("a frame along a path needs a tangent apart from up");
                    }
                }
            }
            Motion::Then(a, b) => {
                a.validate()?;
                b.validate()?;
            }
        }
        Ok(())
    }

    /// Where the body is at `τ`, and the velocity of the motion there as a field over space.
    pub(crate) fn rate(&self, tau: f64) -> (Transform, Twist) {
        match self {
            Motion::Slide(d) => (
                Transform::translation(*d * tau),
                Twist {
                    omega: Vec3::ZERO,
                    v: *d,
                },
            ),
            Motion::Turn {
                origin,
                axis,
                from,
                to,
            } => {
                let k = axis.unit_or_zero();
                let omega = k * (to - from);
                (
                    Transform::rotation(*origin, k, from + (to - from) * tau),
                    Twist {
                        omega,
                        v: -omega.cross(*origin),
                    },
                )
            }
            Motion::Screw { screw, from, to } => {
                let k = screw.axis.unit_or_zero();
                let s = Helix { axis: k, ..*screw };
                let omega = k * (screw.sign * (to - from));
                (
                    s.transform(from + (to - from) * tau),
                    Twist {
                        omega,
                        v: -omega.cross(screw.origin) + k * (screw.rise * (to - from)),
                    },
                )
            }
            Motion::Along { path, range, up } => {
                let dt = range.1 - range.0;
                let at = |t: f64| riding(path, *up, t).expect("a validated path has a frame");
                let (r0, r) = (at(range.0), at(range.0 + dt * tau));
                let pose = frame_transform(&r0).inverse().then(&frame_transform(&r));
                let half: Vec3 = (0..3).fold(Vec3::ZERO, |s, i| s + r.axes[i].cross(r.rates[i]));
                let omega = half * (0.5 * dt);
                (
                    pose,
                    Twist {
                        omega,
                        v: r.speed * dt - omega.cross(r.origin),
                    },
                )
            }
            Motion::Then(a, b) => {
                let (pa, ta) = a.rate(tau);
                let (pb, tb) = b.rate(tau);
                let turned = pb.vector(ta.omega);
                (
                    pa.then(&pb),
                    Twist {
                        omega: tb.omega + turned,
                        v: tb.v - turned.cross(pb.t) + pb.vector(ta.v),
                    },
                )
            }
        }
    }

    /// The velocity of each point of the body as drawn, in the body's own frame, where it is the
    /// same at every `τ`: a slide, a turn or a screw.
    pub(crate) fn steady(&self) -> Option<Twist> {
        match self {
            Motion::Slide(_) | Motion::Turn { .. } | Motion::Screw { .. } => {
                Some(self.body_twist(0.0))
            }
            _ => None,
        }
    }

    /// The velocity at `τ` of each point of the body as drawn, in the body's own frame: the
    /// world velocity of its image, turned back by the pose.
    pub(crate) fn body_twist(&self, tau: f64) -> Twist {
        let (pose, w) = self.rate(tau);
        let back = pose.inverse();
        Twist {
            omega: back.vector(w.omega),
            v: back.vector(w.omega.cross(pose.t) + w.v),
        }
    }
}
