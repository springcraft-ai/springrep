//! Curves and surfaces, each with evaluation, derivatives, projection and parameter ranges.

pub(crate) mod bernstein;
mod bspline;
mod curve;
mod intersection;
mod nurbs;
mod steady;
#[cfg(test)]
pub(crate) use nurbs::tests::octant;
mod surface;

pub use bspline::BSpline;
pub use curve::{Circle, Curve, Ellipse, Line};
pub(crate) use intersection::SHORT_BRANCH;
pub use intersection::{IntersectionCurve, UvPair};
pub(crate) use intersection::{
    distance_to, refine_across, refine_on_contour, refine_onto_both, refine_onto_both_near,
};
pub(crate) use nurbs::Bezier;
pub(crate) use nurbs::piece as spline_piece;
pub use nurbs::{IsoLine, NurbsCurve, NurbsSurface, Patch};
pub(crate) use steady::{Steady, Swept};
pub(crate) use surface::Meridian;
pub use surface::{Cone, Cylinder, Extrusion, Plane, Revolution, Ruled, Sphere, Surface, Torus};
