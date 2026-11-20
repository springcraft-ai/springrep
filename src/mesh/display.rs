//! What a viewer draws: each face meshed as its own triangles, with the exact surface's normal at
//! every node, and the solid's edges as polylines. Faces share no nodes, so shading is smooth
//! within a face and creased where two faces meet. At a pole or an apex each triangle has its own
//! node, so a cone shades along each ruling up to its point. Faces come in the order of the
//! solid's shells, the order the STEP writer writes them. Each edge comes with the faces it joins,
//! the range of the angle between their normals along it, and whether they lie on one surface, so
//! a viewer chooses which edges are creases; seams are listed apart, and poles are left out.
//!
//! [`glb`] packs the faces as one binary glTF mesh in millimetres, z up: `POSITION`, `NORMAL`, a
//! `_FACE` attribute holding each node's face index, the triangles, and `COLOR_0` when each face
//! is given a colour. [`glb_with_edges`] adds the edges as a primitive of lines, each node with
//! its edge's index as `_EDGE` into a table of the edges' faces, turns and splits in the mesh's
//! `extras`, and the seams and split edges as lines of their own.

use std::collections::{BTreeMap, HashMap};

use super::{MeshOptions, edge_params, tessellate};
use crate::{
    Result,
    boolean::same_domain,
    math::Point3,
    tol,
    topo::{Brep, EdgeId, FaceId},
};

/// The face mesh's largest distance from the surface, in mm.
pub const DEFLECTION_MM: f64 = 0.03;
/// The face mesh's largest turn of the normal across a triangle, in radians.
pub const ANGLE: f64 = 0.2;
/// The edge polylines' largest distance from their curves, in mm.
pub const EDGE_DEFLECTION_MM: f64 = 0.005;

/// One face as drawn: its nodes, their normals, and its triangles over them.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Face {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// A solid as drawn.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Display {
    pub faces: Vec<Face>,
    /// The face of the solid each of `faces` draws.
    pub ids: Vec<FaceId>,
    /// Each edge between faces as a polyline.
    pub edges: Vec<Vec<Point3>>,
    /// For each edge, the indices in `faces` of the faces it joins, one for each use.
    pub edge_faces: Vec<Vec<usize>>,
    /// For each edge between two distinct faces, the least and greatest signed angle between
    /// their outward normals over its points, in radians, positive where the edge is convex.
    pub edge_turns: Vec<Option<[f64; 2]>>,
    /// For each edge, whether its two faces lie on one surface: an edge only a split made.
    pub edge_split: Vec<bool>,
    /// Where a face meets itself, as polylines.
    pub seams: Vec<Vec<Point3>>,
}

/// The faces in the order of the solid's shells.
fn shell_order(b: &Brep) -> Vec<FaceId> {
    let mut out = Vec::new();
    for s in b.solid_ids() {
        for &sh in &b.solid(s).shells {
            out.extend(b.shell(sh).faces.iter().copied());
        }
    }
    out
}

fn f32s(p: Point3) -> [f32; 3] {
    [p.x as f32, p.y as f32, p.z as f32]
}

/// `b` as a viewer draws it.
pub fn display(b: &Brep) -> Result<Display> {
    display_within(b, DEFLECTION_MM, EDGE_DEFLECTION_MM)
}

/// `b` as a viewer draws it, its faces within `linear` mm of the surface and its edges within
/// `edge_linear` mm of their curves, each turning by at most [`ANGLE`] across a piece.
pub fn display_within(b: &Brep, linear: f64, edge_linear: f64) -> Result<Display> {
    let mesh = tessellate(
        b,
        &MeshOptions {
            linear,
            angular: ANGLE,
        },
    )?;
    let ranges: HashMap<FaceId, std::ops::Range<usize>> = mesh.faces.iter().cloned().collect();
    let order = shell_order(b);
    let index: HashMap<FaceId, usize> = order.iter().enumerate().map(|(k, &f)| (f, k)).collect();
    let mut faces = Vec::with_capacity(order.len());
    for &f in &order {
        let face = b.face(f);
        let mut out = Face::default();
        let mut local: BTreeMap<u32, u32> = BTreeMap::new();
        for tri in ranges
            .get(&f)
            .map_or(&[][..], |r| &mesh.triangles[r.clone()])
        {
            let mut t = [0u32; 3];
            for (k, &g) in tri.iter().enumerate() {
                let p = mesh.positions[g as usize];
                let uv = face.surface.project(p);
                let (_, su, sv) = face.surface.d1(uv.x, uv.y);
                let outward = |u: f64, v: f64| {
                    let n = face.surface.normal(u, v);
                    f32s(if face.reversed { -n } else { n })
                };
                // At a pole or an apex one parameter moves the point nowhere, and the normal can
                // turn with the other: there the triangle takes its own node, with the normal
                // along the triangle's middle, at the far side's midpoint's parameter.
                if su.norm() <= tol::LINEAR || sv.norm() <= tol::LINEAR {
                    let far =
                        [tri[(k + 1) % 3], tri[(k + 2) % 3]].map(|j| mesh.positions[j as usize]);
                    let mid = face.surface.project(far[0].lerp(far[1], 0.5));
                    let n = if su.norm() <= sv.norm() {
                        outward(mid.x, uv.y)
                    } else {
                        outward(uv.x, mid.y)
                    };
                    out.positions.push(f32s(p));
                    out.normals.push(n);
                    t[k] = (out.positions.len() - 1) as u32;
                    continue;
                }
                t[k] = *local.entry(g).or_insert_with(|| {
                    out.positions.push(f32s(p));
                    out.normals.push(outward(uv.x, uv.y));
                    (out.positions.len() - 1) as u32
                });
            }
            out.triangles.push(t);
        }
        faces.push(out);
    }
    // Each use of an edge: the face, and whether its coedge runs against the edge.
    let mut uses: HashMap<EdgeId, Vec<(FaceId, bool)>> = HashMap::new();
    for f in b.face_ids() {
        for &lp in &b.face(f).loops {
            for &c in &b.lp(lp).coedges {
                let c = b.coedge(c);
                uses.entry(c.edge).or_default().push((f, c.reversed));
            }
        }
    }
    let edge_opt = MeshOptions {
        linear: edge_linear,
        angular: ANGLE,
    };
    let outward = |f: FaceId, p: Point3| {
        let face = b.face(f);
        let uv = face.surface.project(p);
        let (_, su, sv) = face.surface.d1(uv.x, uv.y);
        if su.norm() <= tol::LINEAR || sv.norm() <= tol::LINEAR {
            return None;
        }
        let n = face.surface.normal(uv.x, uv.y);
        Some(if face.reversed { -n } else { n })
    };
    let mut out = Display {
        faces,
        ids: order,
        ..Default::default()
    };
    for e in b.edge_ids() {
        let x = b.edge(e);
        let Some(curve) = &x.curve else { continue };
        let used = uses.get(&e).cloned().unwrap_or_default();
        let ts = edge_params(curve, x.range.0, x.range.1, &edge_opt);
        let points: Vec<Point3> = ts.iter().map(|&t| curve.eval(t)).collect();
        let pair = match used[..] {
            [(f, _), (g, _)] if f == g => {
                out.seams.push(points);
                continue;
            }
            [(f, against), (g, _)] => Some((f, g, against)),
            _ => None,
        };
        // A loop runs with its face on the left seen from outside, so along the first face's
        // coedge the normals of a convex edge turn about the tangent by the right-hand rule.
        let turn = pair.and_then(|(f, g, against)| {
            let mut range: Option<[f64; 2]> = None;
            for (&t, &p) in ts.iter().zip(&points) {
                let (Some(n), Some(m)) = (outward(f, p), outward(g, p)) else {
                    continue;
                };
                let Some(along) = curve.d1(t).1.unit() else {
                    continue;
                };
                let along = if against { -along } else { along };
                let a = n.cross(m).dot(along).atan2(n.dot(m));
                range = Some(range.map_or([a, a], |[lo, hi]| [lo.min(a), hi.max(a)]));
            }
            range
        });
        out.edges.push(points);
        out.edge_faces.push(
            used.iter()
                .filter_map(|(f, _)| index.get(f).copied())
                .collect(),
        );
        out.edge_turns.push(turn);
        out.edge_split.push(
            pair.is_some_and(|(f, g, _)| same_domain(&b.face(f).surface, &b.face(g).surface)),
        );
    }
    Ok(out)
}

/// The faces as one binary glTF mesh, `colors` (`#rrggbb`, one per face) giving `COLOR_0`.
pub fn glb(faces: &[Face], colors: Option<&[String]>) -> Vec<u8> {
    pack(faces, colors, None)
}

/// [`glb`] of the display's faces, with its edges as lines: one primitive of the edges between
/// faces and one of the seams and the edges a split made, told apart by `extras.lines` (`edges`
/// or `seams`). Each line node has its edge's index as `_EDGE`, a seam's being one past the last,
/// and the mesh's `extras.edges` gives each edge's `faces`, `turn` and `split`.
pub fn glb_with_edges(d: &Display, colors: Option<&[String]>) -> Vec<u8> {
    pack(&d.faces, colors, Some(d))
}

/// One kind of line: its nodes, the edge each belongs to, and its segments.
#[derive(Default)]
struct Lines {
    positions: Vec<[f32; 3]>,
    edge_of: Vec<f32>,
    segments: Vec<u32>,
}

impl Lines {
    fn add(&mut self, line: &[Point3], edge: usize) {
        if line.len() < 2 {
            return;
        }
        let base = self.positions.len() as u32;
        self.positions.extend(line.iter().map(|&p| f32s(p)));
        self.edge_of
            .extend(std::iter::repeat_n(edge as f32, line.len()));
        self.segments
            .extend((0..line.len() as u32 - 1).flat_map(|k| [base + k, base + k + 1]));
    }
}

fn pack(faces: &[Face], colors: Option<&[String]>, edges: Option<&Display>) -> Vec<u8> {
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut nor: Vec<[f32; 3]> = Vec::new();
    let mut ids: Vec<f32> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();
    let mut rgba: Vec<[u8; 4]> = Vec::new();
    for (k, f) in faces.iter().enumerate() {
        let base = pos.len() as u32;
        pos.extend(&f.positions);
        nor.extend(&f.normals);
        ids.extend(std::iter::repeat_n(k as f32, f.positions.len()));
        idx.extend(f.triangles.iter().flat_map(|t| t.map(|i| i + base)));
        if let Some(c) = colors.and_then(|c| c.get(k)) {
            let hex =
                |i: usize| u8::from_str_radix(c.get(i..i + 2).unwrap_or("b4"), 16).unwrap_or(180);
            rgba.extend(std::iter::repeat_n(
                [hex(1), hex(3), hex(5), 255],
                f.positions.len(),
            ));
        }
    }
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut add = |bytes: Vec<u8>, kind: &str, count: usize, comp: u32, target: u32| {
        let offset = bin.len();
        bin.extend(&bytes);
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{offset},"byteLength":{},"target":{target}}}"#,
            bytes.len()
        ));
        accessors.push((views.len() - 1, comp, count, kind.to_string()));
        accessors.len() - 1
    };
    let le3 = |v: &[[f32; 3]]| {
        v.iter()
            .flat_map(|p| p.iter().flat_map(|c| c.to_le_bytes()))
            .collect()
    };
    let a_pos = add(le3(&pos), "VEC3", pos.len(), 5126, 34962);
    let a_nor = add(le3(&nor), "VEC3", nor.len(), 5126, 34962);
    let a_face = add(
        ids.iter().flat_map(|c| c.to_le_bytes()).collect(),
        "SCALAR",
        ids.len(),
        5126,
        34962,
    );
    let a_idx = add(
        idx.iter().flat_map(|c| c.to_le_bytes()).collect(),
        "SCALAR",
        idx.len(),
        5125,
        34963,
    );
    let a_color = (!rgba.is_empty()).then(|| {
        add(
            rgba.iter().flatten().copied().collect(),
            "VEC4",
            rgba.len(),
            5121,
            34962,
        )
    });
    let (mut drawn, mut quiet) = (Lines::default(), Lines::default());
    if let Some(d) = edges {
        for (k, line) in d.edges.iter().enumerate() {
            let split = d.edge_split.get(k).copied().unwrap_or(false);
            if split { &mut quiet } else { &mut drawn }.add(line, k);
        }
        for seam in &d.seams {
            quiet.add(seam, d.edges.len());
        }
    }
    // Each kind's accessors: positions, edge indices and segments.
    let line_prims: Vec<(&str, [usize; 3], &Lines)> = [("edges", &drawn), ("seams", &quiet)]
        .into_iter()
        .filter(|(_, l)| !l.segments.is_empty())
        .map(|(kind, l)| {
            let p = add(le3(&l.positions), "VEC3", l.positions.len(), 5126, 34962);
            let e = add(
                l.edge_of.iter().flat_map(|c| c.to_le_bytes()).collect(),
                "SCALAR",
                l.edge_of.len(),
                5126,
                34962,
            );
            let i = add(
                l.segments.iter().flat_map(|c| c.to_le_bytes()).collect(),
                "SCALAR",
                l.segments.len(),
                5125,
                34963,
            );
            (kind, [p, e, i], l)
        })
        .collect();
    let bounds = |v: &[[f32; 3]]| {
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in v {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        (lo, hi)
    };
    let line_bounds: Vec<(usize, [f32; 3], [f32; 3])> = line_prims
        .iter()
        .map(|(_, [p, ..], l)| {
            let (lo, hi) = bounds(&l.positions);
            (*p, lo, hi)
        })
        .collect();
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &pos {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    let num = |v: f32| {
        if v.is_finite() {
            format!("{v}")
        } else {
            "0".to_string()
        }
    };
    let acc_json: Vec<String> = accessors
        .iter()
        .enumerate()
        .map(|(i, (view, comp, count, kind))| {
            let mut s = format!(
                r#"{{"bufferView":{view},"componentType":{comp},"count":{count},"type":"{kind}""#
            );
            let range = if i == a_pos {
                Some((lo, hi))
            } else {
                line_bounds
                    .iter()
                    .find(|b| b.0 == i)
                    .map(|&(_, lo, hi)| (lo, hi))
            };
            if let Some((lo, hi)) = range {
                s += &format!(
                    r#","min":[{},{},{}],"max":[{},{},{}]"#,
                    num(lo[0]),
                    num(lo[1]),
                    num(lo[2]),
                    num(hi[0]),
                    num(hi[1]),
                    num(hi[2])
                );
            }
            if Some(i) == a_color {
                s += r#","normalized":true"#;
            }
            s + "}"
        })
        .collect();
    let mut attrs = format!(r#""POSITION":{a_pos},"NORMAL":{a_nor},"_FACE":{a_face}"#);
    if let Some(c) = a_color {
        attrs += &format!(r#","COLOR_0":{c}"#);
    }
    let mut prims = format!(r#"{{"attributes":{{{attrs}}},"indices":{a_idx}}}"#);
    for (kind, [p, e, i], _) in &line_prims {
        prims += &format!(
            concat!(
                r#",{{"attributes":{{"POSITION":{},"_EDGE":{}}},"indices":{},"mode":1,"#,
                r#""extras":{{"lines":"{}"}}}}"#
            ),
            p, e, i, kind
        );
    }
    let table = edges.map_or(String::new(), |d| {
        let rows: Vec<String> = (0..d.edges.len())
            .map(|k| {
                // A list shorter than `edges` says nothing of the edges past its end.
                let faces: Vec<String> = d
                    .edge_faces
                    .get(k)
                    .map_or(Vec::new(), |f| f.iter().map(usize::to_string).collect());
                let turn = match d.edge_turns.get(k).copied().flatten() {
                    Some([a, b]) if a.is_finite() && b.is_finite() => format!("[{a},{b}]"),
                    _ => "null".to_string(),
                };
                format!(
                    r#"{{"faces":[{}],"turn":{turn},"split":{}}}"#,
                    faces.join(","),
                    d.edge_split.get(k).copied().unwrap_or(false)
                )
            })
            .collect();
        format!(r#","extras":{{"edges":[{}]}}"#, rows.join(","))
    });
    let json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"springrep"}},"scene":0,"#,
            r#""scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"#,
            r#""meshes":[{{"primitives":[{prims}]{table}}}],"#,
            r#""buffers":[{{"byteLength":{len}}}],"bufferViews":[{views}],"accessors":[{accs}]}}"#
        ),
        prims = prims,
        table = table,
        len = bin.len(),
        views = views.join(","),
        accs = acc_json.join(","),
    );
    let mut body = json.into_bytes();
    while !body.len().is_multiple_of(4) {
        body.push(b' ');
    }
    let total = 12 + 8 + body.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(0x4654_6C67u32.to_le_bytes());
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((body.len() as u32).to_le_bytes());
    out.extend(0x4E4F_534Au32.to_le_bytes());
    out.extend(&body);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(0x004E_4942u32.to_le_bytes());
    out.extend(&bin);
    out
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use super::*;
    use crate::{
        math::{Frame, Vec2, Vec3},
        profile::Profile,
        sweep::extrude,
    };

    #[test]
    fn a_cylinder_draws_three_faces_with_exact_normals_and_its_two_rims() {
        // A Ø20 × 10 cylinder: two caps and a side, the side's seam no edge, so the two circles
        // are drawn; every side node's normal points straight out from the axis.
        let b = extrude(
            &Profile::circle(Vec2::ZERO, 20.0),
            &Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap(),
            10.0,
            "can",
        )
        .unwrap();
        let d = display(&b).unwrap();
        assert_eq!(d.faces.len(), 3);
        assert_eq!(d.edges.len(), 2);
        assert!(d.edge_faces.iter().all(|f| f.len() == 2 && f[0] != f[1]));
        assert_eq!(d.seams.len(), 1);
        let ends = [d.seams[0][0], *d.seams[0].last().unwrap()];
        assert!((ends[0].z - ends[1].z).abs() == 10.0 && ends[0].x == ends[1].x);
        assert!(d.edge_split.iter().all(|s| !s));
        for t in &d.edge_turns {
            let [lo, hi] = t.expect("a rim joins two faces");
            assert!((lo - FRAC_PI_2).abs() < 1e-9 && (hi - FRAC_PI_2).abs() < 1e-9);
        }
        let side = d
            .faces
            .iter()
            .find(|f| f.normals.iter().all(|n| n[2].abs() < 1e-6))
            .expect("a side face");
        for (p, n) in side.positions.iter().zip(&side.normals) {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((n[0] - p[0] / r).abs() < 1e-5 && (n[1] - p[1] / r).abs() < 1e-5);
        }
        let bytes = glb(&d.faces, None);
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            bytes.len()
        );
    }

    fn block(size: [f64; 3]) -> Brep {
        let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
        crate::primitive::block(&frame, Vec3::new(size[0], size[1], size[2]), "block").unwrap()
    }

    /// How many edges turn by `angle` all along, within 1e-9.
    fn turning(d: &Display, angle: f64) -> usize {
        d.edge_turns
            .iter()
            .flatten()
            .filter(|t| t.iter().all(|a| (a - angle).abs() < 1e-9))
            .count()
    }

    #[test]
    fn a_block_s_twelve_edges_each_join_two_faces_at_a_convex_right_angle() {
        let d = display(&block([2.0, 3.0, 4.0])).unwrap();
        assert_eq!((d.edges.len(), d.seams.len()), (12, 0));
        assert_eq!(turning(&d, FRAC_PI_2), 12);
        assert!(d.edge_split.iter().all(|s| !s));
        // Each edge's points lie on both of the faces it names.
        for (line, faces) in d.edges.iter().zip(&d.edge_faces) {
            assert!(faces.len() == 2 && faces[0] != faces[1]);
            for &k in faces {
                let f = &d.faces[k];
                let (p, n) = (f.positions[0], f.normals[0]);
                for q in line {
                    let off = [q.x as f32 - p[0], q.y as f32 - p[1], q.z as f32 - p[2]];
                    assert!((0..3).map(|i| off[i] * n[i]).sum::<f32>().abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn a_notch_s_inner_edge_is_concave() {
        // A 4 × 4 × 4 block less a 2 × 2 corner through its height: an L whose one inner edge
        // has the solid on three sides of four.
        let frame = Frame::new(Vec3::new(2.0, 2.0, -1.0), Vec3::Z, Vec3::X).unwrap();
        let corner = crate::primitive::block(&frame, Vec3::new(3.0, 3.0, 6.0), "corner").unwrap();
        let l = crate::boolean::difference(&block([4.0; 3]), &corner).unwrap();
        let d = display(&l).unwrap();
        assert_eq!(d.edges.len(), 18);
        assert_eq!((turning(&d, FRAC_PI_2), turning(&d, -FRAC_PI_2)), (17, 1));
    }

    #[test]
    fn a_fillet_meets_the_faces_it_rounds_into_with_no_turn() {
        let b = block([4.0, 5.0, 6.0]);
        let edge = b
            .edge_ids()
            .find(|&e| {
                let x = b.edge(e);
                let c = x.curve.as_ref().unwrap();
                let (p, q) = (c.eval(x.range.0), c.eval(x.range.1));
                p.x == 0.0 && q.x == 0.0 && p.y == 0.0 && q.y == 0.0
            })
            .expect("the edge along z through the origin");
        let d = display(&crate::blend::fillet_edges(&b, &[edge], 1.0, "round").unwrap()).unwrap();
        // The round's two straight sides are smooth; its two arcs and the other eleven edges,
        // two of them cut short, are right angles.
        assert_eq!(d.edges.len(), 15);
        assert_eq!((turning(&d, 0.0), turning(&d, FRAC_PI_2)), (2, 13));
        assert!(d.edge_split.iter().all(|s| !s));
    }

    #[test]
    fn an_edge_between_two_faces_on_one_plane_is_flagged_as_a_split() {
        // A 4 × 3 × 2 prism whose outline has a fifth corner halfway along one side: that side is
        // two faces on one plane, and the upright edge between them is no change in the shape.
        let pts = [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]];
        let outline = Profile::polygon(
            &pts.map(|p| Vec2::new(p[0], p[1])),
            &["a", "b", "c", "d", "e"],
        );
        let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
        let d = display(&extrude(&outline, &frame, 2.0, "prism").unwrap()).unwrap();
        assert_eq!((d.edges.len(), d.seams.len()), (15, 0));
        let split: Vec<usize> = (0..15).filter(|&k| d.edge_split[k]).collect();
        assert_eq!(split.len(), 1);
        let k = split[0];
        assert!(d.edges[k].iter().all(|p| p.x == 2.0 && p.y == 0.0));
        assert_eq!(d.edge_turns[k], Some([0.0, 0.0]));
        assert!(d.edge_faces[k].len() == 2 && d.edge_faces[k][0] != d.edge_faces[k][1]);
        assert_eq!(turning(&d, FRAC_PI_2), 14);
    }

    /// The JSON chunk of a binary glTF.
    fn json_of(bytes: &[u8]) -> String {
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            bytes.len()
        );
        let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        String::from_utf8(bytes[20..20 + n].to_vec()).unwrap()
    }

    #[test]
    fn a_glb_with_edges_tables_each_edge_and_draws_seams_apart() {
        let b = extrude(
            &Profile::circle(Vec2::ZERO, 20.0),
            &Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap(),
            10.0,
            "can",
        )
        .unwrap();
        let d = display(&b).unwrap();
        assert!(!json_of(&glb(&d.faces, None)).contains(r#""mode":1"#));
        let json = json_of(&glb_with_edges(&d, None));
        assert_eq!(json.matches(r#""mode":1"#).count(), 2);
        assert_eq!(json.matches("_EDGE").count(), 2);
        assert!(json.contains(r#""extras":{"lines":"edges"}"#));
        assert!(json.contains(r#""extras":{"lines":"seams"}"#));
        // Both rims: two faces each, a right angle all round, no split.
        let turn = format!("[{FRAC_PI_2},{FRAC_PI_2}]");
        let row = |faces: &[usize]| {
            format!(
                r#"{{"faces":[{},{}],"turn":{turn},"split":false}}"#,
                faces[0], faces[1]
            )
        };
        let table = format!(
            r#""extras":{{"edges":[{},{}]}}"#,
            row(&d.edge_faces[0]),
            row(&d.edge_faces[1])
        );
        assert!(json.contains(&table), "{json}");
        // A display put together by hand, its lists short of its edges, is unknown past them.
        let mut bare = d.clone();
        bare.edge_faces.truncate(1);
        bare.edge_turns.clear();
        bare.edge_split.clear();
        let json = json_of(&glb_with_edges(&bare, None));
        assert!(
            json.contains(r#"{"faces":[],"turn":null,"split":false}]}"#),
            "{json}"
        );
    }

    #[test]
    fn each_triangle_at_a_cone_apex_has_its_own_node_with_the_normal_along_its_middle() {
        // A cone of radius 2 and height 4, its apex up: the apex has no one normal, so each
        // triangle there has its own apex node, whose normal is that of the ruling halfway across
        // the triangle.
        let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
        let b = crate::primitive::cone(&frame, 2.0, 0.0, 4.0, "cone").unwrap();
        let d = display_within(&b, 0.01, 0.01).unwrap();
        let (cos, sin) = (4.0 / 20f32.sqrt(), 2.0 / 20f32.sqrt());
        let mut fans = 0;
        for f in &d.faces {
            let apex = |i: u32| {
                let p = f.positions[i as usize];
                (p[0].hypot(p[1]) + (p[2] - 4.0).abs()) < 1e-5
            };
            for t in &f.triangles {
                let Some(k) = (0..3).find(|&k| apex(t[k])) else {
                    continue;
                };
                fans += 1;
                let a = t[k];
                assert_eq!(
                    f.triangles.iter().filter(|s| s.contains(&a)).count(),
                    1,
                    "apex node {a} is shared"
                );
                let [p, q] = [t[(k + 1) % 3], t[(k + 2) % 3]].map(|j| f.positions[j as usize]);
                let mid = ((p[1] + q[1]) / 2.0).atan2((p[0] + q[0]) / 2.0);
                let want = [cos * mid.cos(), cos * mid.sin(), sin];
                let n = f.normals[a as usize];
                assert!(
                    (0..3).all(|j| (n[j] - want[j]).abs() < 1e-5),
                    "apex normal {n:?} against {want:?}"
                );
            }
        }
        assert!(fans >= 8, "{fans} triangles at the apex");
    }

    #[test]
    fn a_ball_keeps_one_normal_at_each_pole() {
        let frame = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap();
        let b = crate::primitive::sphere(&frame, 3.0, "ball").unwrap();
        let d = display_within(&b, 0.01, 0.01).unwrap();
        for f in &d.faces {
            for (p, n) in f.positions.iter().zip(&f.normals) {
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                assert!(
                    (0..3).all(|j| (n[j] - p[j] / r).abs() < 1e-5),
                    "{p:?} has {n:?}"
                );
            }
        }
    }
}
