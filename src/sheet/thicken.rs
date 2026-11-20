//! A sheet thickened into a solid, to its front, its back or both.

use std::collections::BTreeSet;

use super::{
    copy::flipped,
    near::{ON_SHEET, front},
};
use crate::{
    boolean, check, edit,
    error::{Error, Result, invalid},
    tol,
    topo::{Brep, FaceId},
};

/// How a thickened sheet closes the gap its moved faces leave where they part, at a crease its
/// thickened side turns convex at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corners {
    /// The moved faces extended until they meet.
    Sharp,
    /// The tube about the crease, and the ball about a corner where such creases meet: the points
    /// within the thickness of the sheet.
    Rounded,
}

/// The solid `sheet` sweeps moving from `back` behind it to `front` in front of it along its
/// faces' normals, each at least zero: the sheet's faces moved `front` along their normals and
/// `back` against them, and walls along the normals at its boundary. Where moved faces part at a
/// crease, `corners` closes the gap. The faces on the sheet and those moved from it keep its faces'
/// names, the walls are `<feature>.side` and the rounds `<feature>.round`.
pub fn thicken(
    sheet: &Brep,
    front: f64,
    back: f64,
    corners: Corners,
    feature: &str,
) -> Result<Brep> {
    if !(front >= 0.0 && back >= 0.0 && (front + back).is_finite()) {
        return invalid("a sheet's thickness on each side must be finite and at least zero");
    }
    if front + back <= tol::LINEAR {
        return invalid("a thickened sheet must be thicker than the tolerance");
    }
    if sheet.face_count() == 0 || sheet.solid_ids().next().is_some() {
        return invalid("only a sheet, faces in no solid, is thickened");
    }
    let issues = check::check_sheet(sheet);
    if !issues.is_empty() {
        let issues: Vec<String> = issues.into_iter().map(|i| i.message).collect();
        return invalid(format!("an invalid sheet: {}", issues.join("; ")));
    }
    let mut parts = Vec::new();
    if front > tol::LINEAR {
        parts.push(one_side(sheet, front, corners, feature)?);
    }
    if back > tol::LINEAR {
        parts.push(one_side(&flipped(sheet)?, back, corners, feature)?);
    }
    let mut out = match &parts[..] {
        [a, b] => boolean::union(a, b)?,
        _ => parts.pop().expect("one side is thicker than the tolerance"),
    };
    for g in out.face_ids().collect::<Vec<_>>() {
        let mut names = BTreeSet::new();
        for n in &out.face(g).names {
            match n.feature.as_str() {
                ON_SHEET => {
                    let i: usize = n.face.parse().expect("a sheet face's index");
                    let f = sheet.face_ids().nth(i).expect("a face of the sheet");
                    names.extend(sheet.face(f).names.iter().cloned());
                }
                _ => {
                    names.insert(n.clone());
                }
            }
        }
        out.set_face_names(g, names);
    }
    Ok(out)
}

/// `sheet` thickened `d` to its front. Rounded, the points within `d` of it on that side. Sharp,
/// the sheet's faces moved `d` out of the solid of the points within `d` of it on its back, whose
/// faces beside the sheet's boundary lie along the sheet's normals there: that solid with them
/// moved, less it.
fn one_side(sheet: &Brep, d: f64, corners: Corners, feature: &str) -> Result<Brep> {
    match corners {
        Corners::Rounded => front(sheet, d, feature),
        Corners::Sharp => {
            let behind = front(&flipped(sheet)?, d, feature)?;
            let faces: Vec<FaceId> = behind
                .face_ids()
                .filter(|&g| behind.face(g).names.iter().any(|n| n.feature == ON_SHEET))
                .collect();
            if faces.is_empty() {
                return Err(Error::Unsupported(
                    "a sheet whose faces the solid behind it lost".into(),
                ));
            }
            edit::thicken(&behind, &faces, d)
        }
    }
}
