//! The developable loft between two convex sections on parallel planes: every ruling joins a point
//! of one section to the point of the other where the tangent runs the same way. The plane through
//! the ruling and those two parallel tangents touches the side along the whole ruling, so the side
//! is developable, and the solid is the convex hull of the two sections.
//!
//! Each section's outline, taken by the direction of its tangent, sweeps once round the circle: a
//! straight edge holds one direction, a curved edge a range, and a corner the range between the
//! edges it joins. The two outlines are cut at every direction where either has a corner or the
//! end of an edge; a straight edge is paired with what the other holds at its direction, and each
//! range between cuts with what the other holds over it. So an edge is paired with an edge (a
//! trapezoid, or a ruled patch between curves), and a corner with an edge (a triangle, or a cone
//! over a curve with its apex at the corner).

use std::f64::consts::TAU;

use super::{Piece, Ring, Section, close, columns_of, prepare, ring, ruled_bands, skin};
use crate::{
    Error, Result,
    error::invalid,
    math::{Point3, Vec3},
    profile::Names,
    tol,
    topo::Brep,
};

/// Tangent directions closer than this are one direction.
const SAME_DIRECTION: f64 = 1e-9;

/// Samples along an edge where its tangent's direction is read.
const SAMPLES: usize = 64;

/// What an outline holds over a stretch of tangent directions, `a0` to `a1` counted on from the
/// direction its first edge starts in.
enum Hold {
    /// Piece `i`, curved: its direction turns from `a0` to `a1`.
    Curve { i: usize, turn: Vec<f64> },
    /// Piece `i`, straight (or turning by no more than [`SAME_DIRECTION`]): `a0 = a1`.
    Straight { i: usize },
    /// The corner at `p`, its two edges' directions `a0` and `a1`.
    Corner { p: Point3 },
}

struct Stretch {
    a0: f64,
    a1: f64,
    hold: Hold,
}

/// An outline taken by its tangent's direction.
struct Gauss {
    stretches: Vec<Stretch>,
    /// The direction the first edge starts in, from `x` about `z`.
    base: f64,
    x: Vec3,
    y: Vec3,
}

/// The direction of `d` about `z` from `x`.
fn direction(d: Vec3, x: Vec3, y: Vec3) -> f64 {
    d.dot(y).atan2(d.dot(x))
}

/// The tangent along the loop of `piece` at `s`.
fn tangent(piece: &Piece, s: f64) -> Vec3 {
    let d = piece.curve.d1(piece.param(s)).1;
    if piece.t1 >= piece.t0 { d } else { -d }
}

impl Gauss {
    fn of(r: &Ring, x: Vec3, y: Vec3) -> Result<Gauss> {
        let not_convex = || {
            Err(Error::Unsupported(
                "a developable loft between sections that are not convex".into(),
            ))
        };
        let dir = |p: &Piece, s: f64| direction(tangent(p, s), x, y);
        let base = dir(&r.pieces[0], 0.0);
        let mut at = 0.0;
        let mut stretches = Vec::new();
        let k = r.pieces.len();
        for (i, p) in r.pieces.iter().enumerate() {
            let a = (dir(p, 0.0) - base - at).rem_euclid(TAU);
            // A corner turns forward by under a half turn; a direction just behind the last is
            // the same one read across the cut.
            let a = if a > TAU - SAME_DIRECTION { 0.0 } else { a };
            if a >= std::f64::consts::PI {
                return not_convex();
            }
            if a > SAME_DIRECTION {
                stretches.push(Stretch {
                    a0: at,
                    a1: at + a,
                    hold: Hold::Corner { p: p.at(0.0) },
                });
                at += a;
            }
            let start = at;
            let mut turn = Vec::with_capacity(SAMPLES + 1);
            turn.push(0.0);
            let mut prev = 0.0;
            let mut last = dir(p, 0.0);
            for m in 1..=SAMPLES {
                let d = dir(p, m as f64 / SAMPLES as f64);
                let step = wrapped(d - last);
                last = d;
                let t = prev + step;
                if t + SAME_DIRECTION < prev {
                    return not_convex();
                }
                prev = t.max(prev);
                turn.push(prev);
            }
            if prev <= SAME_DIRECTION {
                stretches.push(Stretch {
                    a0: start,
                    a1: start,
                    hold: Hold::Straight { i },
                });
            } else {
                at += prev;
                stretches.push(Stretch {
                    a0: start,
                    a1: at,
                    hold: Hold::Curve { i, turn },
                });
            }
            if i + 1 == k {
                let a = (base + TAU - (base + at)).rem_euclid(TAU);
                let close = if a > TAU - SAME_DIRECTION { 0.0 } else { a };
                if close >= std::f64::consts::PI || (at + close - TAU).abs() > 1e-6 {
                    return not_convex();
                }
                if close > SAME_DIRECTION {
                    stretches.push(Stretch {
                        a0: at,
                        a1: at + close,
                        hold: Hold::Corner {
                            p: r.pieces[0].at(0.0),
                        },
                    });
                }
            }
        }
        Ok(Gauss {
            stretches,
            base,
            x,
            y,
        })
    }

    /// `a`, a direction from `x`, counted on from the first edge's start.
    fn local(&self, a: f64) -> f64 {
        let t = (a - self.base).rem_euclid(TAU);
        if t > TAU - SAME_DIRECTION { 0.0 } else { t }
    }

    /// Every direction, from `x`, where a stretch starts or ends.
    fn cuts(&self) -> Vec<f64> {
        self.stretches
            .iter()
            .flat_map(|s| [s.a0, s.a1])
            .map(|a| a + self.base)
            .collect()
    }

    fn straight_at(&self, a: f64) -> Option<usize> {
        let t = self.local(a);
        self.stretches.iter().find_map(|s| match s.hold {
            Hold::Straight { i } if angle_close(s.a0, t) => Some(i),
            _ => None,
        })
    }

    /// The parameter along `p` where its direction has turned `dt` from its start, between the
    /// samples of `turn` that bracket it.
    fn curve_param(&self, p: &Piece, turn: &[f64], dt: f64) -> f64 {
        let m = turn.partition_point(|&v| v < dt).clamp(1, SAMPLES);
        let (mut lo, mut hi) = ((m - 1) as f64 / SAMPLES as f64, m as f64 / SAMPLES as f64);
        let from = direction(tangent(p, lo), self.x, self.y);
        let base = turn[m - 1];
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let got = base + wrapped(direction(tangent(p, mid), self.x, self.y) - from);
            if got < dt { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }

    /// What the outline holds over `a0` to `a1` (`a0 < a1`), or at `a0` when they are equal: a
    /// part of an edge, or a point.
    fn piece(&self, pieces: &[Piece], a0: f64, a1: f64) -> Piece {
        let width = a1 - a0;
        let (t0, mid) = (self.local(a0), self.local(a0 + 0.5 * width));
        let held = self
            .stretches
            .iter()
            .filter(|s| s.a1 - s.a0 > SAME_DIRECTION)
            .find(|s| {
                let t = if width > 0.0 { mid } else { t0 };
                s.a0 - SAME_DIRECTION <= t && t <= s.a1 + SAME_DIRECTION
            });
        match held.map(|s| (s, &s.hold)) {
            Some((s, Hold::Curve { i, turn })) => {
                let p = &pieces[*i];
                let from = if width > 0.0 {
                    mid - 0.5 * width - s.a0
                } else {
                    t0 - s.a0
                };
                let from = from.max(0.0);
                let s0 = self.curve_param(p, turn, from);
                if width <= 0.0 {
                    return Piece::point(p.at(s0));
                }
                let s1 = self.curve_param(p, turn, from + width);
                let (s0, s1) = (
                    if from <= SAME_DIRECTION { 0.0 } else { s0 },
                    if from + width >= s.a1 - s.a0 - SAME_DIRECTION {
                        1.0
                    } else {
                        s1
                    },
                );
                Piece {
                    t0: p.param(s0),
                    t1: p.param(s1),
                    ..p.clone()
                }
            }
            Some((_, Hold::Corner { p })) => Piece::point(*p),
            Some((_, Hold::Straight { .. })) | None => {
                // Between stretches that meet with no corner: the point they share.
                let s = self
                    .stretches
                    .iter()
                    .find(|s| angle_close(s.a1, t0) || angle_close(s.a0, t0))
                    .expect("a direction lies in some stretch");
                let p = match &s.hold {
                    Hold::Curve { i, .. } | Hold::Straight { i } => {
                        let at_end = angle_close(s.a1, t0) && s.a1 > s.a0;
                        pieces[*i].at(if at_end { 1.0 } else { 0.0 })
                    }
                    Hold::Corner { p } => *p,
                };
                Piece::point(p)
            }
        }
    }
}

/// `a` brought into `(−π, π]`.
fn wrapped(a: f64) -> f64 {
    std::f64::consts::PI - (std::f64::consts::PI - a).rem_euclid(TAU)
}

fn angle_close(a: f64, b: f64) -> bool {
    let d = (a - b).rem_euclid(TAU);
    d.min(TAU - d) <= SAME_DIRECTION
}

/// Lofts between two convex sections on parallel planes along the rulings that join points where
/// their tangents run the same way: the convex hull of the two outlines. A side between two
/// parallel straight edges is a trapezoid, between a corner and a straight edge a triangle, between
/// a corner and a curved edge a piece of the cone over that edge with its apex at the corner, and
/// between two curved edges a ruled surface. The sections have no holes; the first one's outline
/// is the `start` face and the second's the `end` face, and every face between is `side`, also
/// named after the first section's edge it grows from, if it grows from one.
pub fn loft_developable(sections: &[Section], feature: &str) -> Result<Brep> {
    if sections.len() != 2 {
        return invalid("a developable loft joins two sections");
    }
    let normal = sections[0].frame.z;
    if sections[1].frame.z.cross(normal).norm() > tol::ANGULAR {
        return invalid("a developable loft's sections lie on parallel planes");
    }
    let (profiles, along) = prepare(sections, None)?;
    if profiles[0].loops.len() > 1 {
        return invalid("a developable loft takes sections without holes");
    }
    let rings: Vec<Ring> = sections
        .iter()
        .zip(&profiles)
        .map(|(s, p)| ring(s, &p.loops[0], along))
        .collect::<Result<_>>()?;
    let (x, y) = (rings[0].frame.x, rings[0].frame.y);
    let gauss: Vec<Gauss> = rings
        .iter()
        .map(|r| Gauss::of(r, x, y))
        .collect::<Result<_>>()?;
    let base = gauss[0].base;
    let mut cuts: Vec<f64> = gauss
        .iter()
        .flat_map(Gauss::cuts)
        .map(|a| (a - base).rem_euclid(TAU))
        .map(|a| if a > TAU - SAME_DIRECTION { 0.0 } else { a })
        .collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| *a - *b <= SAME_DIRECTION);
    let m = cuts.len();
    let mut pieces: Vec<Vec<Piece>> = vec![Vec::new(), Vec::new()];
    let mut names: Vec<Names> = Vec::new();
    let mut push = |row: [Piece; 2], names: &mut Vec<Names>| {
        if row[0].point && row[1].point {
            return;
        }
        names.push(row[0].names.clone());
        let [a, b] = row;
        pieces[0].push(a);
        pieces[1].push(b);
    };
    for c in 0..m {
        let a0 = base + cuts[c];
        let a1 = base
            + if c + 1 < m {
                cuts[c + 1]
            } else {
                cuts[0] + TAU
            };
        let straight: Vec<Option<usize>> = gauss.iter().map(|g| g.straight_at(a0)).collect();
        if straight.iter().any(Option::is_some) {
            let row = [0, 1].map(|r| match straight[r] {
                Some(i) => rings[r].pieces[i].clone(),
                None => gauss[r].piece(&rings[r].pieces, a0, a0),
            });
            push(row, &mut names);
        }
        if a1 - a0 > SAME_DIRECTION {
            let row = [0, 1].map(|r| gauss[r].piece(&rings[r].pieces, a0, a1));
            push(row, &mut names);
        }
    }
    let columns = columns_of(&pieces)?;
    let bands = ruled_bands(&pieces, &columns)?;
    let mut b = Brep::new();
    let s = skin::add_sides(&mut b, &pieces, &columns, &bands, &names, feature)?;
    let frames = [rings[0].frame, rings[1].frame];
    close(b, s.faces, frames, &[s.first], &[s.last], feature)
}
