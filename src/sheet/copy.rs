//! Sheets copied from faces: the faces of a solid as a sheet, and a sheet turned over.

use std::collections::HashMap;

use crate::{
    check,
    error::{Error, Result, invalid},
    topo::{Brep, CoedgeSpec, EdgeId, FaceId, VertexId},
};

/// A sheet of copies of `faces` of `s`, each keeping its surface, its outward normal and its names:
/// faces that share an edge in `s` share it in the sheet, and each edge only one of them uses is on
/// the sheet's boundary. Each piece of faces joined edge to edge is a shell. The sheet's face `i` is
/// the copy of `faces[i]`.
pub fn from_faces(s: &Brep, faces: &[FaceId]) -> Result<Brep> {
    if faces.is_empty() {
        return invalid("a sheet needs at least one face");
    }
    let mut seen = faces.to_vec();
    seen.sort();
    seen.dedup();
    if seen.len() != faces.len() {
        return invalid("a sheet takes each face once");
    }
    copied(s, faces, false)
}

/// `sheet` turned over: the same faces with their outward normals and their loops reversed, so
/// that its front is the side that was its back. Face `i` is the copy of face `i` of `sheet`.
pub fn flipped(sheet: &Brep) -> Result<Brep> {
    let faces: Vec<FaceId> = sheet.face_ids().collect();
    copied(sheet, &faces, true)
}

/// Copies of `faces` of `s` as a checked sheet, each turned over when `flip`.
fn copied(s: &Brep, faces: &[FaceId], flip: bool) -> Result<Brep> {
    let mut out = Brep::new();
    let mut vertices: HashMap<VertexId, VertexId> = HashMap::new();
    let mut edges: HashMap<EdgeId, EdgeId> = HashMap::new();
    let mut vertex = |out: &mut Brep, v: VertexId| {
        *vertices.entry(v).or_insert_with(|| {
            let w = out.add_vertex(s.vertex(v).point);
            out.set_vertex_tolerance(w, s.vertex_tolerance(v));
            w
        })
    };
    let mut copies = Vec::with_capacity(faces.len());
    for &f in faces {
        let face = s.face(f);
        let g = out.add_face(
            face.surface.clone(),
            face.reversed != flip,
            face.names.iter().cloned(),
        );
        copies.push(g);
        for &lp in &face.loops {
            let mut specs = Vec::new();
            for &c in &s.lp(lp).coedges {
                let co = s.coedge(c);
                let edge = match edges.get(&co.edge) {
                    Some(&e) => e,
                    None => {
                        let x = s.edge(co.edge);
                        let start = vertex(&mut out, x.start);
                        let end = vertex(&mut out, x.end);
                        let e = match &x.curve {
                            Some(curve) => out.add_edge(curve.clone(), x.range, start, end),
                            None => out.add_degenerate_edge(start, x.range),
                        };
                        out.set_edge_tolerance(e, s.edge_tolerance(co.edge));
                        edges.insert(co.edge, e);
                        e
                    }
                };
                specs.push(CoedgeSpec {
                    edge,
                    reversed: co.reversed != flip,
                    pcurve: co.pcurve.clone(),
                });
            }
            if flip {
                specs.reverse();
            }
            out.add_loop(g, specs);
        }
    }
    // Shells: faces joined through the edges two of them use.
    let mut group: Vec<usize> = (0..copies.len()).collect();
    fn root(group: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while group[r] != r {
            r = group[r];
        }
        group[i] = r;
        r
    }
    for e in out.edge_ids() {
        let users: Vec<usize> = out
            .edge_uses(e)
            .iter()
            .map(|&c| out.coedge_face(c).index())
            .collect();
        for w in users.windows(2) {
            let (a, b) = (root(&mut group, w[0]), root(&mut group, w[1]));
            group[a] = b;
        }
    }
    let mut shells: Vec<(usize, Vec<FaceId>)> = Vec::new();
    for (i, &g) in copies.iter().enumerate() {
        let r = root(&mut group, i);
        match shells.iter_mut().find(|(k, _)| *k == r) {
            Some((_, fs)) => fs.push(g),
            None => shells.push((r, vec![g])),
        }
    }
    for (_, fs) in shells {
        out.add_shell(fs);
    }
    let issues = check::check_sheet(&out);
    if !issues.is_empty() {
        return Err(Error::InvalidResult(
            issues.into_iter().map(|i| i.message).collect(),
        ));
    }
    Ok(out)
}
