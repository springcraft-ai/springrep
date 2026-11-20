//! Planar profiles in sketch coordinates: closed loops of named segments, the first loop the
//! outer boundary and the rest holes. A sweep names each face it makes after its segment.

pub mod region;

use std::collections::BTreeSet;
use std::f64::consts::TAU;

use crate::{
    Result,
    error::invalid,
    geom::{BSpline, Circle, Curve, Ellipse, Line},
    math::{Frame, Vec2, Vec3},
    tol,
};

/// A turn between consecutive points smaller than this, in radians, leaves them on one straight
/// edge; generated profiles are read with it.
pub const STRAIGHT_TURN: f64 = 1e-7;

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Segment {
    Line {
        a: Vec2,
        b: Vec2,
    },
    /// Counter-clockwise from `start` (radians) when `sweep > 0`, clockwise when it is negative.
    Arc {
        center: Vec2,
        radius: f64,
        start: f64,
        sweep: f64,
    },
    /// A whole circle from angle 0, counter-clockwise when `ccw`.
    Circle {
        center: Vec2,
        radius: f64,
        ccw: bool,
    },
    /// `center + major·cos t·m + minor·sin t·(m rotated 90°)`, `m` at `rotation`, from parameter
    /// `start` through `sweep`.
    Ellipse {
        center: Vec2,
        rotation: f64,
        major: f64,
        minor: f64,
        start: f64,
        sweep: f64,
    },
    /// A spline in sketch coordinates (control points with z = 0), against its parameter when
    /// `reversed`.
    Spline {
        curve: BSpline,
        reversed: bool,
    },
    /// A rational Bézier from `ctrl[0]` to its last point, with a weight for each control point:
    /// a conic arc at degree 2, and the exact form of any rational polynomial section.
    Bezier {
        ctrl: Vec<Vec2>,
        weights: Vec<f64>,
    },
}

/// The names a profile edge carries. Each face swept from the edge takes them all: an edge two
/// outlines share, joined into one, has a name from each.
pub type Names = BTreeSet<String>;

#[derive(Clone, Debug, PartialEq)]
pub struct NamedSegment {
    pub segment: Segment,
    pub names: Names,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProfileLoop {
    pub segments: Vec<NamedSegment>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub loops: Vec<ProfileLoop>,
}

/// A segment as a 3-D curve on a sketch plane: the curve, the parameter range it covers, and
/// whether the loop runs along the parameter.
pub struct PlacedSegment {
    pub curve: Curve,
    pub range: (f64, f64),
    pub forward: bool,
}

fn v3(p: Vec2) -> Vec3 {
    Vec3::new(p.x, p.y, 0.0)
}

fn polar(c: Vec2, r: f64, a: f64) -> Vec2 {
    Vec2::new(c.x + r * a.cos(), c.y + r * a.sin())
}

impl Segment {
    /// The conic arc from `a` to `b` whose tangents there meet at `apex`, with weight `w` at the
    /// apex: the rational quadratic Bézier on those three points. Below a weight of 1 it is an
    /// arc of an ellipse, and is returned as that arc; a parabola (1) and a hyperbola (above) are
    /// returned as the Bézier itself, their exact form.
    ///
    /// With `m` the chord's middle, the centre lies on the line from `m` through the apex at
    /// `m − w²/(1 − w²)·(apex − m)`; the shoulder and the half chord are conjugate semi-diameters,
    /// from which the principal axes follow.
    pub fn conic(a: Vec2, apex: Vec2, b: Vec2, w: f64) -> Result<Segment> {
        if w.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return invalid(format!("a conic of weight {w}: a weight is above 0"));
        }
        if !w.is_finite() {
            return invalid(format!("a conic of weight {w}: a weight is finite"));
        }
        if w >= 1.0 {
            return Ok(Segment::Bezier {
                ctrl: vec![a, apex, b],
                weights: vec![1.0, w, 1.0],
            });
        }
        let m = (a + b) * 0.5;
        let c = m - (apex - m) * (w * w / (1.0 - w * w));
        let shoulder = m + (apex - m) * (w / (1.0 + w));
        let u = shoulder - c;
        let phi = ((m - c).norm() / u.norm()).clamp(-1.0, 1.0).acos();
        let v = (b - m) * (1.0 / phi.sin());
        // Principal axes from the conjugate pair: P(t) = c + u cos t + v sin t.
        let t = 0.5 * (2.0 * u.dot(v)).atan2(u.dot(u) - v.dot(v));
        let p = u * t.cos() + v * t.sin();
        let q = u * (-t.sin()) + v * t.cos();
        let (major, minor) = if p.norm() >= q.norm() { (p, q) } else { (q, p) };
        let (ra, rb) = (major.norm(), minor.norm());
        let (mx, my) = (
            major * (1.0 / ra),
            Vec2::new(-major.y, major.x) * (1.0 / ra),
        );
        let param = |x: Vec2| {
            let d = x - c;
            (d.dot(my) / rb).atan2(d.dot(mx) / ra)
        };
        let (t0, t1, ts) = (param(a), param(b), param(shoulder));
        let ccw = (t1 - t0).rem_euclid(TAU);
        let sweep = if (ts - t0).rem_euclid(TAU) < ccw {
            ccw
        } else {
            ccw - TAU
        };
        Ok(Segment::Ellipse {
            center: c,
            rotation: major.y.atan2(major.x),
            major: ra,
            minor: rb,
            start: t0,
            sweep,
        })
    }

    /// The point at fraction `s` ∈ [0, 1] along the segment, in its direction.
    pub fn at(&self, s: f64) -> Vec2 {
        match self {
            Segment::Line { a, b } => a.lerp(*b, s),
            Segment::Arc {
                center,
                radius,
                start,
                sweep,
            } => polar(*center, *radius, start + sweep * s),
            Segment::Circle {
                center,
                radius,
                ccw,
            } => polar(*center, *radius, if *ccw { TAU * s } else { -TAU * s }),
            Segment::Ellipse {
                center,
                rotation,
                major,
                minor,
                start,
                sweep,
            } => {
                let t = start + sweep * s;
                let (m, n) = (
                    Vec2::new(rotation.cos(), rotation.sin()),
                    Vec2::new(-rotation.sin(), rotation.cos()),
                );
                *center + m * (major * t.cos()) + n * (minor * t.sin())
            }
            Segment::Spline { curve, reversed } => {
                let (a, b) = curve.domain();
                let t = if *reversed {
                    b + (a - b) * s
                } else {
                    a + (b - a) * s
                };
                let p = curve.eval(t);
                Vec2::new(p.x, p.y)
            }
            Segment::Bezier { ctrl, weights } => {
                // De Casteljau on the homogeneous points.
                let mut h: Vec<(Vec2, f64)> = ctrl
                    .iter()
                    .zip(weights)
                    .map(|(&p, &w)| (p * w, w))
                    .collect();
                for k in (1..h.len()).rev() {
                    for i in 0..k {
                        h[i] = (
                            h[i].0 * (1.0 - s) + h[i + 1].0 * s,
                            h[i].1 * (1.0 - s) + h[i + 1].1 * s,
                        );
                    }
                }
                h[0].0 * (1.0 / h[0].1)
            }
        }
    }

    pub fn start(&self) -> Vec2 {
        self.at(0.0)
    }

    pub fn end(&self) -> Vec2 {
        self.at(1.0)
    }

    pub fn is_closed(&self) -> bool {
        match self {
            Segment::Circle { .. } => true,
            Segment::Spline { curve, .. } => curve.is_periodic(),
            _ => false,
        }
    }

    pub fn reverse(&mut self) {
        match self {
            Segment::Line { a, b } => std::mem::swap(a, b),
            Segment::Arc { start, sweep, .. } | Segment::Ellipse { start, sweep, .. } => {
                *start += *sweep;
                *sweep = -*sweep;
            }
            Segment::Circle { ccw, .. } => *ccw = !*ccw,
            Segment::Spline { reversed, .. } => *reversed = !*reversed,
            Segment::Bezier { ctrl, weights } => {
                ctrl.reverse();
                weights.reverse();
            }
        }
    }

    fn samples(&self) -> usize {
        match self {
            Segment::Line { .. } => 1,
            Segment::Arc { .. } | Segment::Ellipse { .. } => 32,
            Segment::Circle { .. } => 64,
            Segment::Spline { curve, .. } => 8 * (curve.control_points().len() + 4),
            Segment::Bezier { ctrl, .. } => 16 * ctrl.len(),
        }
    }

    /// The segment as a curve on `frame`'s plane, sketch `(x, y)` at `frame.at(x, y)`.
    pub fn place(&self, frame: &Frame) -> PlacedSegment {
        let at = |p: Vec2| frame.at(p.x, p.y);
        let dir = |a: f64| frame.radial(a);
        let circle_frame = |c: Vec2, a: f64| Frame {
            origin: at(c),
            x: dir(a),
            y: dir(a + TAU / 4.0),
            z: frame.z,
        };
        match self {
            Segment::Line { a, b } => {
                let (line, len) = Line::through(at(*a), at(*b)).unwrap_or((
                    Line {
                        origin: at(*a),
                        dir: frame.x,
                    },
                    0.0,
                ));
                PlacedSegment {
                    curve: Curve::Line(line),
                    range: (0.0, len),
                    forward: true,
                }
            }
            Segment::Arc {
                center,
                radius,
                start,
                sweep,
            } => {
                let x_at = if *sweep >= 0.0 { *start } else { start + sweep };
                PlacedSegment {
                    curve: Curve::Circle(Circle {
                        frame: circle_frame(*center, x_at),
                        radius: *radius,
                    }),
                    range: (0.0, sweep.abs()),
                    forward: *sweep >= 0.0,
                }
            }
            Segment::Circle {
                center,
                radius,
                ccw,
            } => PlacedSegment {
                curve: Curve::Circle(Circle {
                    frame: circle_frame(*center, 0.0),
                    radius: *radius,
                }),
                range: (0.0, TAU),
                forward: *ccw,
            },
            Segment::Ellipse {
                center,
                rotation,
                major,
                minor,
                start,
                sweep,
            } => PlacedSegment {
                curve: Curve::Ellipse(Ellipse {
                    frame: circle_frame(*center, *rotation),
                    major: *major,
                    minor: *minor,
                }),
                range: if *sweep >= 0.0 {
                    (*start, start + sweep)
                } else {
                    (start + sweep, *start)
                },
                forward: *sweep >= 0.0,
            },
            Segment::Spline { curve, reversed } => PlacedSegment {
                curve: Curve::BSpline(curve.transformed(&frame.to_transform())),
                range: curve.domain(),
                forward: !reversed,
            },
            Segment::Bezier { ctrl, weights } => {
                let d = ctrl.len() - 1;
                let mut knots = vec![0.0; d + 1];
                knots.extend(vec![1.0; d + 1]);
                let pts = ctrl.iter().map(|&p| at(p)).collect();
                let c = crate::geom::NurbsCurve::new(d, knots, pts, Some(weights.clone()))
                    .expect("a Bézier's knots and weights fit its points");
                PlacedSegment {
                    curve: Curve::Nurbs(Box::new(c)),
                    range: (0.0, 1.0),
                    forward: true,
                }
            }
        }
    }
}

impl ProfileLoop {
    /// The loop's points, densely enough that their polygon has the loop's orientation.
    pub fn polygon(&self) -> Vec<Vec2> {
        let mut out = Vec::new();
        for s in &self.segments {
            let n = s.segment.samples();
            out.extend((0..n).map(|i| s.segment.at(i as f64 / n as f64)));
        }
        out
    }

    /// Twice the signed area of [`ProfileLoop::polygon`]: positive when counter-clockwise.
    pub fn signed_area2(&self) -> f64 {
        let p = self.polygon();
        (0..p.len()).map(|i| p[i].cross(p[(i + 1) % p.len()])).sum()
    }

    pub fn reverse(&mut self) {
        self.segments.reverse();
        for s in &mut self.segments {
            s.segment.reverse();
        }
    }

    fn check_closed(&self) -> Result<()> {
        let n = self.segments.len();
        if n == 0 {
            return invalid("a profile loop has no segments");
        }
        for i in 0..n {
            let a = &self.segments[i].segment;
            let b = &self.segments[(i + 1) % n].segment;
            if n == 1 && !a.is_closed() {
                return invalid("a single-segment loop must be a closed curve");
            }
            if n > 1 && a.is_closed() {
                return invalid("a closed curve must be a loop on its own");
            }
            if a.end().dist(b.start()) > tol::LINEAR {
                return invalid(format!(
                    "profile segment {} ends at {:?} but the next starts at {:?}",
                    self.segments[i].label(),
                    a.end(),
                    b.start()
                ));
            }
        }
        Ok(())
    }
}

fn named(segment: Segment, name: &str) -> NamedSegment {
    NamedSegment::new(segment, name)
}

/// How a list of points becomes a profile loop.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PointsInterp {
    /// Straight segments point to point.
    Polyline,
    /// One closed C² interpolating cubic.
    PeriodicSpline,
    /// The rule for generated outlines: a corner where the outline turns by more than
    /// `corner` radians, a straight edge through each collinear run, and a spline through every
    /// other run between corners.
    Smooth { corner: f64 },
}

fn turn(a: Vec2, b: Vec2, c: Vec2) -> f64 {
    let (u, v) = (b - a, c - b);
    u.cross(v).atan2(u.dot(v)).abs()
}

fn spline_through(points: &[Vec2]) -> Result<Segment> {
    if points.len() == 2 {
        return Ok(Segment::Line {
            a: points[0],
            b: points[1],
        });
    }
    let p3: Vec<Vec3> = points.iter().map(|&p| v3(p)).collect();
    Ok(Segment::Spline {
        curve: BSpline::interpolate_bessel(&p3)?,
        reversed: false,
    })
}

/// The edges through one run of points between corners: one line through each stretch of
/// collinear points, and one spline through each stretch between them.
fn run_segments(run: &[Vec2]) -> Result<Vec<Segment>> {
    let n = run.len();
    let straight =
        |j: usize| j > 0 && j + 1 < n && turn(run[j - 1], run[j], run[j + 1]) < STRAIGHT_TURN;
    // Segment k joins run[k] and run[k + 1]; it is straight when an end is a collinear point.
    let seg_straight: Vec<bool> = (0..n - 1).map(|k| straight(k) || straight(k + 1)).collect();
    let mut out = Vec::new();
    let mut k = 0;
    while k < n - 1 {
        let kind = seg_straight[k];
        let mut e = k;
        while e + 1 < n - 1 && seg_straight[e + 1] == kind {
            e += 1;
        }
        let pts = &run[k..=e + 1];
        out.push(if kind {
            Segment::Line {
                a: pts[0],
                b: pts[pts.len() - 1],
            }
        } else {
            spline_through(pts)?
        });
        k = e + 1;
    }
    Ok(out)
}

fn ccw_points(points: &[Vec2]) -> Vec<Vec2> {
    let a: f64 = (0..points.len())
        .map(|i| points[i].cross(points[(i + 1) % points.len()]))
        .sum();
    let mut p = points.to_vec();
    if a < 0.0 {
        p.reverse();
    }
    p
}

impl NamedSegment {
    /// `segment` carrying the one name `name`.
    pub fn new(segment: Segment, name: impl Into<String>) -> NamedSegment {
        NamedSegment {
            segment,
            names: Names::from([name.into()]),
        }
    }

    /// The edge's names as one word, for a message: `a+b`.
    pub fn label(&self) -> String {
        self.names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("+")
    }

    /// The straight edge from `a` to `b`.
    pub fn line(a: Vec2, b: Vec2, name: &str) -> NamedSegment {
        named(Segment::Line { a, b }, name)
    }

    /// The arc about `center` from angle `start` through `sweep`, both in radians: counter-clockwise
    /// when `sweep > 0`.
    pub fn arc(center: Vec2, radius: f64, start: f64, sweep: f64, name: &str) -> NamedSegment {
        let arc = Segment::Arc {
            center,
            radius,
            start,
            sweep,
        };
        named(arc, name)
    }
}

impl Profile {
    /// The profile of one loop, `segments` end to end.
    pub fn outline(segments: Vec<NamedSegment>) -> Profile {
        Profile {
            loops: vec![ProfileLoop { segments }],
        }
    }

    /// A `width × height` rectangle; its edges are `v_min`, `u_max`, `v_max`, `u_min`.
    pub fn rect(center: Vec2, width: f64, height: f64) -> Profile {
        let (hw, hh) = (width / 2.0, height / 2.0);
        let p = [
            Vec2::new(center.x - hw, center.y - hh),
            Vec2::new(center.x + hw, center.y - hh),
            Vec2::new(center.x + hw, center.y + hh),
            Vec2::new(center.x - hw, center.y + hh),
        ];
        Profile::polygon(&p, &["v_min", "u_max", "v_max", "u_min"])
    }

    /// A circle; its one edge is `side`.
    pub fn circle(center: Vec2, diameter: f64) -> Profile {
        Profile::named_circle(center, diameter, crate::topo::FaceName::SIDE)
    }

    /// A circle whose one edge is `name`.
    pub fn named_circle(center: Vec2, diameter: f64, name: &str) -> Profile {
        let circle = Segment::Circle {
            center,
            radius: diameter / 2.0,
            ccw: true,
        };
        Profile::outline(vec![named(circle, name)])
    }

    /// A polygon; edge `i` runs from point `i` to point `i + 1` and is named `names[i]`.
    pub fn polygon(points: &[Vec2], names: &[&str]) -> Profile {
        let n = points.len();
        let segments = (0..n)
            .map(|i| {
                named(
                    Segment::Line {
                        a: points[i],
                        b: points[(i + 1) % n],
                    },
                    names[i % names.len()],
                )
            })
            .collect();
        Profile {
            loops: vec![ProfileLoop { segments }],
        }
    }

    /// A closed outline through `points`, every edge named `name`.
    pub fn points(points: &[Vec2], interp: PointsInterp, name: &str) -> Result<Profile> {
        if points.len() < 3 {
            return invalid("a point-list profile needs at least three points");
        }
        let pts = ccw_points(points);
        let n = pts.len();
        let segments: Vec<Segment> = match interp {
            PointsInterp::Polyline => (0..n)
                .map(|i| Segment::Line {
                    a: pts[i],
                    b: pts[(i + 1) % n],
                })
                .collect(),
            PointsInterp::PeriodicSpline => {
                let p3: Vec<Vec3> = pts.iter().map(|&p| v3(p)).collect();
                vec![Segment::Spline {
                    curve: BSpline::interpolate_periodic(&p3)?,
                    reversed: false,
                }]
            }
            PointsInterp::Smooth { corner } => {
                let corners: Vec<usize> = (0..n)
                    .filter(|&i| turn(pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]) > corner)
                    .collect();
                if corners.is_empty() {
                    let p3: Vec<Vec3> = pts.iter().map(|&p| v3(p)).collect();
                    vec![Segment::Spline {
                        curve: BSpline::interpolate_periodic(&p3)?,
                        reversed: false,
                    }]
                } else {
                    let mut out = Vec::new();
                    for (k, &c) in corners.iter().enumerate() {
                        let end = corners[(k + 1) % corners.len()];
                        let len = match (end + n - c) % n {
                            0 => n,
                            l => l,
                        };
                        let run: Vec<Vec2> = (0..=len).map(|j| pts[(c + j) % n]).collect();
                        out.extend(run_segments(&run)?);
                    }
                    out
                }
            }
        };
        Ok(Profile {
            loops: vec![ProfileLoop {
                segments: segments.into_iter().map(|s| named(s, name)).collect(),
            }],
        })
    }

    /// The profile with holes added; each hole's segments keep their names.
    pub fn with_holes(mut self, holes: impl IntoIterator<Item = ProfileLoop>) -> Profile {
        self.loops.extend(holes);
        self
    }

    /// Checks every loop is closed and orients the outer loop counter-clockwise and each hole
    /// clockwise.
    pub fn normalized(&self) -> Result<Profile> {
        if self.loops.is_empty() {
            return invalid("a profile needs a loop");
        }
        let mut out = self.clone();
        for (i, lp) in out.loops.iter_mut().enumerate() {
            for seg in &lp.segments {
                let pts: Vec<Vec2> = (0..=seg.segment.samples())
                    .map(|k| seg.segment.at(k as f64 / seg.segment.samples() as f64))
                    .collect();
                if pts.iter().any(|p| !(p.x.is_finite() && p.y.is_finite())) {
                    return invalid(format!("profile segment {} is not finite", seg.label()));
                }
                let len: f64 = pts.windows(2).map(|w| w[0].dist(w[1])).sum();
                if len <= tol::LINEAR {
                    return invalid(format!(
                        "profile segment {} is {len:e} mm long, within the tolerance",
                        seg.label()
                    ));
                }
            }
            lp.check_closed()?;
            let a = lp.signed_area2();
            if a.abs() <= tol::LINEAR * tol::LINEAR {
                return invalid("a profile loop encloses no area");
            }
            if (i == 0) != (a > 0.0) {
                lp.reverse();
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_is_counter_clockwise_and_named() {
        let p = Profile::rect(Vec2::new(1.0, 1.0), 4.0, 2.0)
            .normalized()
            .unwrap();
        assert!(p.loops[0].signed_area2() > 0.0);
        let names: Vec<String> = p.loops[0]
            .segments
            .iter()
            .map(NamedSegment::label)
            .collect();
        assert_eq!(names, ["v_min", "u_max", "v_max", "u_min"]);
    }

    #[test]
    fn holes_are_turned_clockwise() {
        let hole = Profile::circle(Vec2::ZERO, 2.0).loops.remove(0);
        let p = Profile::rect(Vec2::ZERO, 10.0, 10.0)
            .with_holes([hole])
            .normalized()
            .unwrap();
        assert!(p.loops[1].signed_area2() < 0.0);
        assert_eq!(
            p.loops[1].segments[0].segment,
            Segment::Circle {
                center: Vec2::ZERO,
                radius: 1.0,
                ccw: false
            }
        );
    }

    #[test]
    fn smooth_points_split_at_corners_and_straight_runs() {
        // Collinear points along the bottom and right sides, then a half circle back to the left
        // side: corners only at the two bottom corners.
        let mut pts: Vec<Vec2> = (0..4).map(|i| Vec2::new(i as f64, 0.0)).collect();
        pts.extend((1..3).map(|i| Vec2::new(3.0, i as f64)));
        for i in 0..=24 {
            let a = std::f64::consts::PI * i as f64 / 24.0;
            pts.push(Vec2::new(1.5 + 1.5 * a.cos(), 3.0 + 1.5 * a.sin()));
        }
        let p = Profile::points(
            &pts,
            PointsInterp::Smooth {
                corner: 15f64.to_radians(),
            },
            "side",
        )
        .unwrap();
        let kinds: Vec<&str> = p.loops[0]
            .segments
            .iter()
            .map(|s| match s.segment {
                Segment::Line { .. } => "line",
                Segment::Spline { .. } => "spline",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["line", "line", "spline"]);
        assert!(p.normalized().is_ok());
    }
}
