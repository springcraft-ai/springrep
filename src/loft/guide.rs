//! Lofts along guide curves: each section's vertices lie on the guides, and the side between two
//! neighbouring guides is the Gordon surface of its curve network (Gordon 1969): the two guides
//! one way, the section edges between them the other. With `P_u` the ruled surface between the
//! guides, `P_v` the section edges interpolated across the sections, and `P_uv` the same
//! interpolation of the corners ruled across, the side is `P_u + P_v − P_uv`, which runs along
//! every curve of the network (Piegl and Tiller 1997, §10.5).
//! When the section edges are straight, `P_v = P_uv` and the side is the ruled surface between
//! its guides.
//!
//! The sum is taken on the control points once the three surfaces share degrees and knots, so the
//! curves must be polynomial: a weight would make the sum of the points differ from the point of
//! the summed net. The guides are first reparameterized piecewise linearly, so each crosses every
//! section at one common parameter.

use super::{
    Piece, Ring, Section, close, columns_of,
    rail::{self, Across, Hom, Rail},
    ring, skin,
    skin::Band,
};
use crate::{
    Error, Result,
    error::invalid,
    geom::{Curve, Surface},
    math::Point3,
    profile::Names,
    tol,
    topo::Brep,
};

/// A guide curve over `range` of its parameter, from the first section to the last (or the apex).
#[derive(Clone, Debug)]
pub struct Guide {
    pub curve: Curve,
    pub range: (f64, f64),
}

/// Samples along a guide where it is tested for crossing a section's plane.
const SAMPLES: usize = 512;

fn rational(r: &Rail) -> bool {
    r.pw.iter().any(|h| (h[3] - 1.0).abs() > 1e-15)
}

/// The parameter where `g` crosses section `s`'s plane, strictly between its ends.
fn crossing(g: &Rail, s: &Section) -> Result<f64> {
    let f = |t: f64| (g.at(t) - s.frame.origin).dot(s.frame.z);
    let mut found = Vec::new();
    let mut prev = f(0.0);
    for i in 1..=SAMPLES {
        let t = i as f64 / SAMPLES as f64;
        let cur = f(t);
        if (prev < 0.0) != (cur < 0.0) {
            let (mut lo, mut hi) = ((i - 1) as f64 / SAMPLES as f64, t);
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if (f(mid) < 0.0) == (prev < 0.0) {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            found.push(0.5 * (lo + hi));
        }
        prev = cur;
    }
    match found[..] {
        [t] if t > 0.0 && t < 1.0 => Ok(t),
        _ => invalid("a loft's guide must cross each section between its ends once"),
    }
}

/// Which loop of section `s`, and which of its vertices, lies at `p`.
fn vertex_at(rings: &[Vec<Ring>], s: usize, p: Point3) -> Result<(usize, usize)> {
    rings
        .iter()
        .enumerate()
        .find_map(|(li, rl)| {
            rl[s]
                .starts()
                .iter()
                .position(|&v| v.dist(p) <= tol::LINEAR)
                .map(|vi| (li, vi))
        })
        .ok_or_else(|| {
            Error::InvalidInput("a loft's guide misses every vertex of a section".into())
        })
}

/// `data` at every row carried across the rows by `acr`: one rail in `v`.
fn across_rail(acr: &Across, data: &[Hom]) -> Rail {
    Rail {
        degree: acr.degree,
        knots: acr.knots.clone(),
        pw: acr.apply(data),
    }
}

/// A rail in `u` through `pw`, of `degree` on `knots`.
fn u_rail(degree: usize, knots: &[f64], pw: Vec<Hom>) -> Rail {
    Rail {
        degree,
        knots: knots.to_vec(),
        pw,
    }
}

/// The Gordon surface of one side: `col` the section rails at every row (compatible), `left` and
/// `right` the guides at its ends (reparameterized onto the rows' parameters).
fn gordon(col: &[Rail], left: &Rail, right: &Rail, acr: &Across) -> Result<Surface> {
    let nu = col[0].pw.len();
    let row_data = |i: usize| -> Vec<Hom> { col.iter().map(|r| r.pw[i]).collect() };
    // Every surface as rails in `v`, one per control point in `u`: `P_v`, then `P_uv`, then `P_u`.
    let mut vr: Vec<Rail> = (0..nu).map(|i| across_rail(acr, &row_data(i))).collect();
    vr.push(across_rail(acr, &row_data(0)));
    vr.push(across_rail(acr, &row_data(nu - 1)));
    vr.push(left.clone());
    vr.push(right.clone());
    rail::make_compatible(&mut vr);
    let (dv, kv) = (vr[0].degree, vr[0].knots.clone());
    let nv = vr[0].pw.len();
    let (du0, ku0) = (col[0].degree, col[0].knots.clone());
    let lin = [0.0, 0.0, 1.0, 1.0];
    let mut net: Vec<Vec<Hom>> = Vec::new();
    let mut ku = Vec::new();
    let mut du = 0;
    for m in 0..nv {
        let mut us = vec![
            u_rail(du0, &ku0, (0..nu).map(|i| vr[i].pw[m]).collect()),
            u_rail(1, &lin, vec![vr[nu].pw[m], vr[nu + 1].pw[m]]),
            u_rail(1, &lin, vec![vr[nu + 2].pw[m], vr[nu + 3].pw[m]]),
        ];
        rail::make_compatible(&mut us);
        let row: Vec<Hom> = (0..us[0].pw.len())
            .map(|i| {
                let (v, uv, u) = (us[0].pw[i], us[1].pw[i], us[2].pw[i]);
                [
                    v[0] + u[0] - uv[0],
                    v[1] + u[1] - uv[1],
                    v[2] + u[2] - uv[2],
                    1.0,
                ]
            })
            .collect();
        du = us[0].degree;
        ku = us[0].knots.clone();
        net.push(row);
    }
    super::extended_surface((du, dv), (&ku, &kv), &net)
}

/// Lofts through `sections` in order, and on to `apex` if given, along `guides`: every vertex of
/// every section lies on a guide and every guide runs through a vertex of each section, from the
/// first section (where its range starts) to the last section or the apex (where it ends). The
/// guides pass through the sections' vertices in the order the vertices run round each loop.
/// Each side between two neighbouring guides runs along both guides and through every section's
/// edge between them. Sections and guides must be polynomial (lines and splines, no arcs). A hole
/// is lofted along the guides through its own vertices. Faces are named as [`super::loft`] names
/// them; a loft to an apex has no `end` face.
pub fn loft_guided(
    sections: &[Section],
    guides: &[Guide],
    apex: Option<Point3>,
    feature: &str,
) -> Result<Brep> {
    let (profiles, along) = super::prepare(sections, apex)?;
    let loops = profiles[0].loops.len();
    // The rings of each loop, through every section.
    let mut rings: Vec<Vec<Ring>> = (0..loops)
        .map(|li| {
            sections
                .iter()
                .zip(&profiles)
                .map(|(s, p)| ring(s, &p.loops[li], along))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<_>>()?;
    let rails: Vec<Rail> = guides
        .iter()
        .map(|g| Rail::of_curve(&g.curve, g.range.0, g.range.1))
        .collect::<Result<_>>()?;
    let n = sections.len();
    let last = n - 1;
    let rows = n + usize::from(apex.is_some());
    // Each guide's parameter at every row, and the loop and vertex it passes through on each
    // section.
    let mut params = vec![vec![0.0; rows]; guides.len()];
    let mut at = vec![vec![(0, 0); n]; guides.len()];
    for (g, r) in rails.iter().enumerate() {
        for s in 0..n {
            let t = if s == 0 {
                0.0
            } else if s == last && apex.is_none() {
                1.0
            } else {
                crossing(r, &sections[s])?
            };
            params[g][s] = t;
            at[g][s] = vertex_at(&rings, s, r.at(t))?;
            if at[g][s].0 != at[g][0].0 {
                return invalid("a loft's guide must run through one loop of every section");
            }
        }
        if let Some(a) = apex {
            if r.end().dist(a) > tol::LINEAR {
                return invalid("a loft's guide must end at its apex");
            }
            params[g][n] = 1.0;
        }
    }
    let frames = [rings[0][0].frame, rings[0][last].frame];
    let mut b = Brep::new();
    let mut faces = Vec::new();
    let (mut first, mut end) = (Vec::new(), Vec::new());
    for (li, rl) in rings.iter_mut().enumerate() {
        // The loop's guides in the order of the first section's vertices; every other section
        // turned to start at the first guide's vertex, and holding the rest in the same order.
        let mut order: Vec<usize> = (0..guides.len()).filter(|&g| at[g][0].0 == li).collect();
        let k = order.len();
        if k < 2 || rl.iter().any(|r| r.pieces.len() != k) {
            return invalid("a loft has one guide through each vertex of its sections");
        }
        order.sort_by_key(|&g| at[g][0].1);
        for (s, r) in rl.iter_mut().enumerate() {
            let start = at[order[0]][s].1;
            r.pieces.rotate_left(start);
            for (j, &g) in order.iter().enumerate() {
                if (at[g][s].1 + k - start) % k != j {
                    return invalid(
                        "a loft's guides must run through the sections' vertices in order",
                    );
                }
            }
        }
        let common: Vec<f64> = (0..rows)
            .map(|s| order.iter().map(|&g| params[g][s]).sum::<f64>() / k as f64)
            .collect();
        if common.windows(2).any(|w| w[1] - w[0] <= 0.0) {
            return invalid("a loft's guides must reach its sections in order");
        }
        let sides: Vec<Rail> = order
            .iter()
            .map(|&g| rails[g].reparameterized(&params[g], &common))
            .collect();
        let mut pieces: Vec<Vec<Piece>> = rl
            .iter_mut()
            .map(|r| std::mem::take(&mut r.pieces))
            .collect();
        if let Some(a) = apex {
            pieces.push(vec![Piece::point(a); k]);
        }
        let columns = columns_of(&pieces)?;
        if sides.iter().chain(columns.iter().flatten()).any(rational) {
            return Err(Error::Unsupported(
                "a loft along guides through rational curves".into(),
            ));
        }
        let acr = if rows == 2 {
            rail::bezier_fit(&[0.0, 1.0], 1)?
        } else {
            rail::interpolation(&common)?
        };
        let surfaces = (0..k)
            .map(|j| gordon(&columns[j], &sides[j], &sides[(j + 1) % k], &acr).map(Some))
            .collect::<Result<_>>()?;
        let band = Band {
            lo: 0,
            hi: rows - 1,
            sides,
            surfaces,
        };
        let names: Vec<Names> = pieces[0].iter().map(|p| p.names.clone()).collect();
        let s = skin::add_sides(&mut b, &pieces, &columns, &[band], &names, feature)?;
        faces.extend(s.faces);
        first.push(s.first);
        end.push(s.last);
    }
    close(b, faces, frames, &first, &end, feature)
}
