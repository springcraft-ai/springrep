//! A draft both ways from a parting line: the solid split at the parting plane, each side's faces
//! drafted away from it, and the sides joined again.

use super::draft;
use crate::{
    Result, boolean,
    error::invalid,
    math::Frame,
    topo::{Brep, FaceId},
};

/// The faces of `piece` cut from one of `faces` of `b`: on its surface and carrying its names.
fn cut_from(b: &Brep, faces: &[FaceId], piece: &Brep) -> Vec<FaceId> {
    piece
        .face_ids()
        .filter(|&p| {
            let q = piece.face(p);
            q.shell.is_some()
                && faces.iter().any(|&f| {
                    let f = b.face(f);
                    f.names == q.names && f.surface == q.surface
                })
        })
        .collect()
}

/// `b` with each of `faces` split where it crosses the `parting` plane, as a part made in a
/// two-piece mold is, and each piece tilted by `angle` away from that plane about the line where
/// it meets it: on the side the plane's normal points to with the pull along the normal, on the
/// other with the pull against it, so positive angles narrow the part away from the parting line
/// both ways. A face on one side only is drafted on that side. Each piece keeps its face's names.
pub fn draft_parted(b: &Brep, faces: &[FaceId], parting: &Frame, angle: f64) -> Result<Brep> {
    if faces.is_empty() {
        return invalid("a parted draft needs a face");
    }
    let (above, below) = boolean::split(b, parting.origin, parting.z)?;
    let against = Frame::new(parting.origin, -parting.z, parting.x)?;
    let mut sides = Vec::new();
    for (side, pull) in [(above, parting), (below, &against)] {
        if side.solid_ids().next().is_none() {
            continue;
        }
        let tilted = cut_from(b, faces, &side);
        sides.push(if tilted.is_empty() {
            side
        } else {
            draft(&side, &tilted, pull, angle)?
        });
    }
    match sides.as_slice() {
        [one] => Ok(one.clone()),
        [up, down] => boolean::union(up, down),
        _ => invalid("a parted draft of an empty solid"),
    }
}
