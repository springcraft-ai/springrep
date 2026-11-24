//! Blends built as faces and sewn into the body (D. J. Jackson, "Boundary representation
//! modelling with local tolerances", Solid Modeling '95, section 5): each piece's blend faces
//! are taken whole from its swept tool, whose sides lie on the two faces its edge joins, so the
//! contact curves where the blend leaves those faces are the tool's own edges, exact by
//! construction. The faces the edge joins give up the edge for those contact curves, every other
//! edge at the edge's ends is cut back to where a contact curve meets it, and a face the blend ends
//! on takes the tool's end section in place of its corner. Nothing is intersected: every curve and
//! point is one the sweep made or one already on the body. Where two rounds on one axis meet
//! along the face between them, as a full round's do, their contact curves are one edge and that
//! face is gone; a piece its corner balls take whole has no tool, and the two balls are one patch.
//! Where the pieces do not close up that way, the caller cuts the tools instead.

use std::collections::{BTreeSet, HashMap};

use super::{Ball, Picked, SIDES, run::Beyond};
use crate::{
    boolean::{
        csi::{curve_distance, signed_distance},
        same_domain, unify,
    },
    check,
    geom::{Curve, Surface},
    math::{Point3, Vec3},
    tol,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, FaceName, PCurve, VertexId},
};

/// What a face of a piece's tool is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Role {
    Blend,
    /// A blend face of another piece sharing the tool, across a crease.
    Other,
    /// The side swept along the picked edge's face `k`.
    Side(usize),
    /// An end of the piece: square to its edge, or the surface it was trimmed by.
    Cap,
}

/// The name that marks a tool's blend and side faces with their piece, so pieces whose tools are
/// joined into one keep them apart.
const PIECE: &str = "#piece";

/// Piece `i`'s tool with its blend faces marked `i` and its side `k` marked `i_k`.
pub(super) fn marked(mut tool: Brep, i: usize) -> Brep {
    for f in tool.face_ids().collect::<Vec<_>>() {
        let mut names = tool.face(f).names.clone();
        let mark = if names.iter().any(|n| n.face == "blend") {
            format!("{i}")
        } else if let Some(k) = SIDES
            .iter()
            .position(|s| names.iter().any(|n| n.face == *s))
        {
            format!("{i}_{k}")
        } else {
            continue;
        };
        names.insert(FaceName::new(PIECE, mark));
        tool.set_face_names(f, names);
    }
    tool
}

fn role(tool: &Brep, f: FaceId, i: usize) -> Role {
    let names = &tool.face(f).names;
    let marked = |m: String| names.iter().any(|n| n.feature == PIECE && n.face == m);
    if names.iter().any(|n| n.face == "blend") {
        if marked(format!("{i}")) {
            Role::Blend
        } else {
            Role::Other
        }
    } else if let Some(k) = (0..SIDES.len()).find(|k| marked(format!("{i}_{k}"))) {
        Role::Side(k)
    } else {
        Role::Cap
    }
}

/// What an edge of a piece's blend faces is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Use {
    /// Where the blend leaves the picked edge's face `k`.
    Contact(usize),
    /// Where the blend ends, on a face of the body or on the next piece's end.
    Cross,
    /// Between two blend faces of the piece, or a blend face's seam.
    Inner,
    /// A point the blend narrows to on its edge's axis.
    Pole,
}

/// A piece's blend faces and how each of their edges is used.
struct Parts {
    blends: Vec<FaceId>,
    uses: Vec<(EdgeId, Use)>,
}

/// Piece `i`'s blend faces in its tool, which other pieces' blends may share.
fn parts(tool: &Brep, i: usize) -> Option<Parts> {
    let mut faces_of: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for c in tool.coedge_ids() {
        faces_of
            .entry(tool.coedge(c).edge)
            .or_default()
            .push(tool.coedge_face(c));
    }
    let role = |f: FaceId| role(tool, f, i);
    let blends: Vec<FaceId> = tool
        .face_ids()
        .filter(|&f| role(f) == Role::Blend)
        .collect();
    let mut uses: Vec<(EdgeId, Use)> = Vec::new();
    for &f in &blends {
        for &lp in &tool.face(f).loops {
            for &c in &tool.lp(lp).coedges {
                let e = tool.coedge(c).edge;
                if uses.iter().any(|u| u.0 == e) {
                    continue;
                }
                if tool.edge(e).curve.is_none() {
                    uses.push((e, Use::Pole));
                    continue;
                }
                let roles: Vec<Role> = faces_of[&e].iter().map(|&g| role(g)).collect();
                let u = match roles.as_slice() {
                    [Role::Blend, Role::Blend | Role::Other] | [Role::Other, Role::Blend] => {
                        Use::Inner
                    }
                    [Role::Blend, Role::Side(k)] | [Role::Side(k), Role::Blend] => Use::Contact(*k),
                    [Role::Blend, Role::Cap] | [Role::Cap, Role::Blend] => Use::Cross,
                    _ => return None,
                };
                uses.push((e, u));
            }
        }
    }
    (!blends.is_empty()).then_some(Parts { blends, uses })
}

/// One of the result's edges standing for a tool's edge: whether it runs the other way, and
/// whether it is the tool's edge itself, so the tool's pcurves hold for it.
#[derive(Clone, Copy)]
struct Mapped {
    edge: EdgeId,
    flipped: bool,
    own: bool,
}

/// The blend sewn into `body`: every piece's blend faces, named `names[i]`, facing out of the body
/// when the piece is `convex[i]`, cut away from it, and into it otherwise; or why it could not be.
/// A piece with no tool has no faces.
#[allow(clippy::too_many_arguments)]
pub(super) fn sewn(
    body: &Brep,
    picked: &[Picked],
    tools: &[Option<Brep>],
    beyond: &[[Beyond; 2]],
    convex: &[bool],
    names: &[BTreeSet<FaceName>],
    balls: &[(Vec<usize>, Ball)],
    feature: &str,
) -> Result<Brep, String> {
    if body.has_tolerances() {
        return Err("a body with local tolerances".into());
    }
    let parts: Vec<Parts> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| match t {
            Some(t) => parts(t, i),
            None => Some(Parts {
                blends: Vec::new(),
                uses: Vec::new(),
            }),
        })
        .collect::<Option<_>>()
        .ok_or("a tool whose blend faces meet it other than at its sides and ends")?;
    let picked_of: HashMap<EdgeId, usize> = picked
        .iter()
        .enumerate()
        .map(|(i, pk)| (pk.edge, i))
        .collect();
    // The body's vertices the blend takes away, each with the pieces that take it.
    let mut gone: BTreeSet<VertexId> = BTreeSet::new();
    let mut near: HashMap<VertexId, BTreeSet<usize>> = HashMap::new();
    for (i, pk) in picked.iter().enumerate() {
        let e = body.edge(pk.edge);
        for v in [e.start, e.end] {
            gone.insert(v);
            near.entry(v).or_default().insert(i);
        }
    }
    let mut dropped: BTreeSet<EdgeId> = BTreeSet::new();
    for (i, b) in beyond.iter().enumerate() {
        for bk in b {
            for &w in &bk.gone {
                gone.insert(w);
                near.entry(w).or_default().insert(i);
            }
            dropped.extend(bk.dropped.iter().copied());
        }
    }

    let mut out = Brep::new();
    let mut vmap: Vec<Option<VertexId>> = vec![None; body.vertex_ids().count()];
    for v in body.vertex_ids() {
        if !gone.contains(&v) {
            vmap[v.index()] = Some(out.add_vertex(body.vertex(v).point));
        }
    }

    // The tools' vertices, one wherever two pieces' ends meet.
    let mut pool: Vec<VertexId> = Vec::new();
    let mut tvmap: Vec<HashMap<VertexId, VertexId>> = vec![HashMap::new(); tools.len()];
    for (i, tool) in tools.iter().enumerate() {
        let Some(tool) = tool else { continue };
        for &(e, _) in &parts[i].uses {
            let edge = tool.edge(e);
            for v in [edge.start, edge.end] {
                if tvmap[i].contains_key(&v) {
                    continue;
                }
                let p = tool.vertex(v).point;
                let id = match pool
                    .iter()
                    .find(|&&w| out.vertex(w).point.dist(p) <= tol::LINEAR)
                {
                    Some(&w) => w,
                    None => {
                        let w = out.add_vertex(p);
                        pool.push(w);
                        w
                    }
                };
                tvmap[i].insert(v, id);
            }
        }
    }
    let points: Vec<BTreeSet<VertexId>> = tvmap
        .iter()
        .map(|m| m.values().copied().collect())
        .collect();
    for &w in &pool {
        let p = out.vertex(w).point;
        if vmap
            .iter()
            .flatten()
            .any(|&v| out.vertex(v).point.dist(p) <= tol::LINEAR)
        {
            return Err("a blend that ends on a vertex of the body".into());
        }
    }

    // Every other edge at a blend's end is cut back to where a contact curve meets it, or runs on
    // to it past the end where the blend opens the face beyond.
    let mut ranges: Vec<(f64, f64)> = body.edge_ids().map(|e| body.edge(e).range).collect();
    let mut ends: Vec<[Option<VertexId>; 2]> = body
        .edge_ids()
        .map(|e| {
            let x = body.edge(e);
            [vmap[x.start.index()], vmap[x.end.index()]]
        })
        .collect();
    for e in body.edge_ids() {
        if picked_of.contains_key(&e) || dropped.contains(&e) {
            continue;
        }
        let edge = body.edge(e);
        let at = [gone.contains(&edge.start), gone.contains(&edge.end)];
        if !at[0] && !at[1] {
            continue;
        }
        let Some(curve) = &edge.curve else {
            return Err("a degenerate edge at a blend's end".into());
        };
        if edge.start == edge.end {
            return Err("a closed edge through a blend's end".into());
        }
        for (k, &cut) in at.iter().enumerate() {
            if !cut {
                continue;
            }
            let v = if k == 0 { edge.start } else { edge.end };
            let corner = body.vertex(v).point;
            let candidates: BTreeSet<VertexId> = near[&v]
                .iter()
                .flat_map(|&i| points[i].iter().copied())
                .collect();
            let on = |range: (f64, f64)| -> Vec<(VertexId, f64)> {
                candidates
                    .iter()
                    .filter_map(|&w| {
                        let p = out.vertex(w).point;
                        let (d, t) = curve_distance(curve, range, p);
                        (d <= tol::LINEAR && p.dist(corner) > tol::LINEAR).then_some((w, t))
                    })
                    .collect()
            };
            let mut hits = on(edge.range);
            // Where the face the blend ends on turns more than a half turn at the corner, the end
            // section opens it past the corner, and the edge between it and the blend's own face
            // runs on to the section.
            let span = edge.range.1 - edge.range.0;
            let more = match curve {
                Curve::Line(_) => Some(1e3 * span.max(1.0)),
                Curve::Circle(_) => Some(std::f64::consts::TAU - span),
                _ => None,
            };
            if let Some(more) = more.filter(|_| hits.is_empty()) {
                let past = if k == 0 {
                    (edge.range.0 - more, edge.range.0)
                } else {
                    (edge.range.1, edge.range.1 + more)
                };
                hits = on(past);
            }
            let Some(&(w, t)) = hits.first() else {
                return Err(format!(
                    "a {} edge at a blend's end at ({:.3}, {:.3}, {:.3}) that no contact curve \
                     meets",
                    curve.kind(),
                    corner.x,
                    corner.y,
                    corner.z
                ));
            };
            if hits.iter().any(|h| h.0 != w) {
                return Err("an edge at a blend's end that two contact curves meet".into());
            }
            ends[e.index()][k] = Some(w);
            if k == 0 {
                ranges[e.index()].0 = t;
            } else {
                ranges[e.index()].1 = t;
            }
        }
        let (t0, t1) = ranges[e.index()];
        let speed = curve.d1(0.5 * (t0 + t1)).1.norm().max(1e-300);
        // Cut back from both ends to the one point where the blends at its ends meet, it is gone.
        let [a, b] = ends[e.index()];
        if at[0] && at[1] && a == b && (t1 - t0).abs() * speed <= tol::LINEAR {
            dropped.insert(e);
            continue;
        }
        if (t1 - t0) * speed <= tol::LINEAR {
            return Err("an edge a blend cuts back past its other end".into());
        }
    }
    let mut emap: Vec<Option<EdgeId>> = vec![None; body.edge_ids().count()];
    for e in body.edge_ids() {
        if picked_of.contains_key(&e) || dropped.contains(&e) {
            continue;
        }
        let edge = body.edge(e);
        let [Some(s), Some(t)] = ends[e.index()] else {
            unreachable!("every end of a kept edge is kept or cut back");
        };
        emap[e.index()] = Some(match &edge.curve {
            Some(c) => out.add_edge(c.clone(), ranges[e.index()], s, t),
            None => out.add_degenerate_edge(s, edge.range),
        });
    }

    // The tools' edges; an end section two pieces share is one edge.
    let mut tmap: Vec<HashMap<EdgeId, Mapped>> = vec![HashMap::new(); tools.len()];
    let mut crosses: Vec<(EdgeId, Point3)> = Vec::new();
    // A crease between two pieces whose tools were joined is one edge of both.
    let mut creases: Vec<(EdgeId, Point3)> = Vec::new();
    // So is the contact curve of two rounds that meet along the face between them.
    let mut contacts: Vec<(EdgeId, Point3)> = Vec::new();
    let mut met: BTreeSet<EdgeId> = BTreeSet::new();
    for (i, tool) in tools.iter().enumerate() {
        let Some(tool) = tool else { continue };
        for &(e, u) in &parts[i].uses {
            let edge = tool.edge(e);
            let (s, t) = (tvmap[i][&edge.start], tvmap[i][&edge.end]);
            let Some(curve) = edge.curve.as_ref() else {
                let x = out.add_degenerate_edge(s, edge.range);
                tmap[i].insert(
                    e,
                    Mapped {
                        edge: x,
                        flipped: false,
                        own: true,
                    },
                );
                continue;
            };
            let mid = curve.eval(0.5 * (edge.range.0 + edge.range.1));
            let shared = match u {
                Use::Cross => Some(&crosses),
                Use::Inner => Some(&creases),
                Use::Contact(_) => Some(&contacts),
                Use::Pole => None,
            };
            if let Some(list) = shared.filter(|_| s != t) {
                let same = list.iter().find_map(|&(x, m)| {
                    let xe = out.edge(x);
                    let ends_match =
                        (xe.start == s && xe.end == t) || (xe.start == t && xe.end == s);
                    (ends_match && m.dist(mid) <= tol::LINEAR).then_some((x, xe.start != s))
                });
                if let Some((x, flipped)) = same {
                    if matches!(u, Use::Contact(_)) {
                        met.insert(x);
                    }
                    tmap[i].insert(
                        e,
                        Mapped {
                            edge: x,
                            flipped,
                            own: false,
                        },
                    );
                    continue;
                }
            }
            let x = out.add_edge(curve.clone(), edge.range, s, t);
            match u {
                Use::Cross => crosses.push((x, mid)),
                Use::Inner => creases.push((x, mid)),
                Use::Contact(_) => contacts.push((x, mid)),
                Use::Pole => {}
            }
            tmap[i].insert(
                e,
                Mapped {
                    edge: x,
                    flipped: false,
                    own: true,
                },
            );
        }
    }
    let cross_edges: Vec<EdgeId> = crosses.iter().map(|c| c.0).collect();

    // The body's faces, each a face of the input (`Some(0)` to the checker) or one the blend
    // changed.
    let mut fmap: Vec<Option<FaceId>> = Vec::with_capacity(body.face_count());
    let mut sides: Vec<Option<usize>> = Vec::new();
    for f in body.face_ids() {
        let face = body.face(f);
        let mut touched = false;
        let mut loops = Vec::with_capacity(face.loops.len());
        for &lp in &face.loops {
            let (specs, changed) = rebuild_loop(
                body,
                lp,
                &Rebuild {
                    picked,
                    picked_of: &picked_of,
                    parts: &parts,
                    tools,
                    tmap: &tmap,
                    emap: &emap,
                    ranges: &ranges,
                    cross_edges: &cross_edges,
                    dropped: &dropped,
                    met: &met,
                },
                &out,
            )?;
            touched |= changed;
            loops.push(specs);
        }
        // A face the rounds on either side of it meet along is gone.
        if !loops.is_empty() && loops.iter().all(|specs| specs.is_empty()) {
            fmap.push(None);
            continue;
        }
        if loops.iter().any(|specs| specs.is_empty()) {
            return Err("a face that rounds meet along about a loop it keeps".into());
        }
        let nf = out.add_face(face.surface.clone(), face.reversed, face.names.clone());
        fmap.push(Some(nf));
        for specs in loops {
            out.add_loop(nf, specs);
        }
        sides.push((!touched).then_some(0));
    }

    let mut blend_faces: Vec<(usize, FaceId)> = Vec::new();
    for (i, tool) in tools.iter().enumerate() {
        let Some(tool) = tool else { continue };
        for &f in &parts[i].blends {
            let face = tool.face(f);
            let nf = out.add_face(
                face.surface.clone(),
                face.reversed != convex[i],
                names[i].iter().cloned(),
            );
            for &lp in &face.loops {
                let mut specs = Vec::new();
                for &c in &tool.lp(lp).coedges {
                    let co = tool.coedge(c);
                    let m = tmap[i][&co.edge];
                    let params = check::coedge_params(tool, c, 1);
                    let pcurve = |at_end: bool| -> PCurve {
                        if m.own && !matches!(co.pcurve, PCurve::Projected { .. }) {
                            return co.pcurve.clone();
                        }
                        let uvs = tool.coedge_uvs(c, &params);
                        PCurve::Projected {
                            seed: if at_end { uvs[1] } else { uvs[0] },
                        }
                    };
                    specs.push(if convex[i] {
                        CoedgeSpec {
                            edge: m.edge,
                            reversed: co.reversed == m.flipped,
                            pcurve: pcurve(true),
                        }
                    } else {
                        CoedgeSpec {
                            edge: m.edge,
                            reversed: co.reversed != m.flipped,
                            pcurve: pcurve(false),
                        }
                    });
                }
                if convex[i] {
                    specs.reverse();
                }
                out.add_loop(nf, specs);
            }
            sides.push(None);
            blend_faces.push((i, nf));
        }
    }

    for (at, ball) in balls {
        let f = corner_face(&mut out, ball, at, &tmap, tools, &parts, feature)?;
        sides.push(None);
        blend_faces.push((at[0], f));
    }

    for s in body.solid_ids() {
        let mut shells = Vec::new();
        for &sh in &body.solid(s).shells {
            let mut faces: Vec<FaceId> = body
                .shell(sh)
                .faces
                .iter()
                .filter_map(|f| fmap[f.index()])
                .collect();
            let members: BTreeSet<FaceId> = body.shell(sh).faces.iter().copied().collect();
            faces.extend(
                blend_faces
                    .iter()
                    .filter(|(i, _)| members.contains(&picked[*i].faces[0]))
                    .map(|&(_, f)| f),
            );
            shells.push(out.add_shell(faces));
        }
        out.add_solid_of(shells, body.solid(s).outer);
    }
    out.pair_radial();

    let issues = check::check(&out);
    if !issues.is_empty() {
        let first: Vec<String> = issues.iter().take(3).map(|i| i.to_string()).collect();
        return Err(format!("the sewn solid is invalid: {}", first.join("; ")));
    }
    if mergeable(&out, &sides) {
        return unify(&out).map_err(|e| format!("merging the sewn faces: {e}"));
    }
    Ok(out)
}

/// The sphere patch closing a corner where the pieces `at` stop short: bounded by each one's end
/// section, an arc of a great circle of the sphere, and facing out of the body.
fn corner_face(
    out: &mut Brep,
    ball: &Ball,
    at: &[usize],
    tmap: &[HashMap<EdgeId, Mapped>],
    tools: &[Option<Brep>],
    parts: &[Parts],
    feature: &str,
) -> Result<FaceId, String> {
    let sphere = ball.surface().map_err(|e| e.to_string())?;
    let mut arcs: Vec<EdgeId> = Vec::new();
    for (&i, &along) in at.iter().zip(&ball.edges) {
        let Some(tool) = &tools[i] else {
            return Err("a corner piece its corners take whole".into());
        };
        // On the plane through the centre square to the piece's edge: the far end section of a
        // piece much shorter than the radius lies within the tolerance of the sphere too.
        let on_ball = |p: Point3| {
            (p.dist(ball.centre) - ball.radius).abs() <= tol::LINEAR
                && (p - ball.centre).dot(along).abs() <= tol::LINEAR
        };
        let mut ends = parts[i].uses.iter().filter(|u| {
            let e = tool.edge(u.0);
            u.1 == Use::Cross
                && on_ball(tool.vertex(e.start).point)
                && on_ball(tool.vertex(e.end).point)
        });
        let (Some(&(e, _)), None) = (ends.next(), ends.next()) else {
            return Err("a corner piece without one end section on its sphere".into());
        };
        arcs.push(tmap[i][&e].edge);
    }
    // Chained end to end, then turned to run counter-clockwise seen from outside.
    let mut lp: Vec<(EdgeId, bool)> = vec![(arcs[0], false)];
    let mut left: Vec<EdgeId> = arcs[1..].to_vec();
    while !left.is_empty() {
        let (e, rev) = *lp.last().expect("started with one");
        let x = out.edge(e);
        let tip = if rev { x.start } else { x.end };
        let k = left
            .iter()
            .position(|&a| out.edge(a).start == tip || out.edge(a).end == tip)
            .ok_or("a corner whose end sections do not close")?;
        let a = left.remove(k);
        lp.push((a, out.edge(a).end == tip));
    }
    let start = |&(e, rev): &(EdgeId, bool)| {
        let x = out.edge(e);
        out.vertex(if rev { x.end } else { x.start }).point
    };
    let corners: Vec<Point3> = lp.iter().map(start).collect();
    let n: Vec3 = ball.middle();
    let turn = (corners[1] - corners[0])
        .cross(corners[2] - corners[1])
        .dot(n);
    if turn < 0.0 {
        lp.reverse();
        for c in &mut lp {
            c.1 = !c.1;
        }
    }
    let surface = Surface::Sphere(sphere);
    let hint = surface.project(ball.centre + n.unit_or_zero() * ball.radius);
    let specs = lp
        .iter()
        .map(|c| CoedgeSpec {
            edge: c.0,
            reversed: c.1,
            pcurve: PCurve::Projected {
                seed: surface.project_near(start(c), hint),
            },
        })
        .collect();
    let f = out.add_face(surface, false, [FaceName::new(feature, "corner")]);
    out.add_loop(f, specs);
    Ok(f)
}

/// What rebuilding a body face's loop reads.
struct Rebuild<'a> {
    picked: &'a [Picked],
    picked_of: &'a HashMap<EdgeId, usize>,
    parts: &'a [Parts],
    tools: &'a [Option<Brep>],
    tmap: &'a [HashMap<EdgeId, Mapped>],
    emap: &'a [Option<EdgeId>],
    ranges: &'a [(f64, f64)],
    cross_edges: &'a [EdgeId],
    dropped: &'a BTreeSet<EdgeId>,
    /// The contact curves two rounds share.
    met: &'a BTreeSet<EdgeId>,
}

/// A coedge spec's ends in `out`.
fn spec_ends(out: &Brep, s: &CoedgeSpec) -> (VertexId, VertexId) {
    let e = out.edge(s.edge);
    if s.reversed {
        (e.end, e.start)
    } else {
        (e.start, e.end)
    }
}

/// Body loop `lp` with each picked edge replaced by its piece's contact curve on this face, each
/// edge cut back to its new ends, and an end section across each corner the blend took off; and
/// whether anything changed. Empty where the blends leave nothing of the loop.
fn rebuild_loop(
    body: &Brep,
    lp: crate::topo::LoopId,
    r: &Rebuild,
    out: &Brep,
) -> Result<(Vec<CoedgeSpec>, bool), String> {
    let face = body.face(body.lp(lp).face);
    let surface = &face.surface;
    let mut changed = false;
    // Each spec with the body coedge it stands for.
    let mut specs: Vec<(CoedgeSpec, CoedgeId)> = Vec::new();
    for &c in &body.lp(lp).coedges {
        let co = body.coedge(c);
        let edge = body.edge(co.edge);
        let (t_start, _) = if co.reversed {
            (edge.range.1, edge.range.0)
        } else {
            edge.range
        };
        if let Some(&i) = r.picked_of.get(&co.edge) {
            changed = true;
            // The corners at the two ends of a piece with no tool meet where it would lie.
            let Some(tool) = &r.tools[i] else { continue };
            let k = r.picked[i]
                .faces
                .iter()
                .position(|&g| g == body.lp(lp).face)
                .ok_or("a picked edge on a face it does not join")?;
            let contacts: Vec<EdgeId> = r.parts[i]
                .uses
                .iter()
                .filter(|u| u.1 == Use::Contact(k))
                .map(|u| u.0)
                .collect();
            let ce = match contacts.as_slice() {
                [ce] => ce,
                // The blend meets this face only at a pole, where its neighbours' contacts meet.
                [] if r.parts[i].uses.iter().any(|u| u.1 == Use::Pole) => continue,
                _ => {
                    return Err("a piece with its contact on a face in other than one edge".into());
                }
            };
            let tc = tool.edge(*ce);
            let tcurve = tc.curve.as_ref().expect("checked in parts");
            let curve = edge.curve.as_ref().expect("a picked edge has a curve");
            let tm = 0.5 * (edge.range.0 + edge.range.1);
            let (p, d) = curve.d1(tm);
            let d = if co.reversed { -d } else { d };
            let (_, s) = curve_distance(tcurve, tc.range, p);
            let m = r.tmap[i][ce];
            let reversed = (tcurve.d1(s).1.dot(d) < 0.0) != m.flipped;
            let spec_edge = out.edge(m.edge);
            let start = if reversed {
                spec_edge.end
            } else {
                spec_edge.start
            };
            let hint = body.coedge_uvs(c, &[t_start])[0];
            let seed = surface.project_near(out.vertex(start).point, hint);
            specs.push((
                CoedgeSpec {
                    edge: m.edge,
                    reversed,
                    pcurve: PCurve::Projected { seed },
                },
                c,
            ));
            continue;
        }
        if r.dropped.contains(&co.edge) {
            changed = true;
            continue;
        }
        let ne = r.emap[co.edge.index()].expect("every unpicked edge is kept");
        let range = r.ranges[co.edge.index()];
        let new_start = if co.reversed { range.1 } else { range.0 };
        let pcurve = match &co.pcurve {
            PCurve::Projected { .. } if new_start != t_start => {
                changed = true;
                PCurve::Projected {
                    seed: body.coedge_uvs(c, &[t_start, new_start])[1],
                }
            }
            other => {
                changed |= range != edge.range;
                other.clone()
            }
        };
        specs.push((
            CoedgeSpec {
                edge: ne,
                reversed: co.reversed,
                pcurve,
            },
            c,
        ));
    }
    if !changed {
        return Ok((specs.into_iter().map(|s| s.0).collect(), false));
    }
    let n = specs.len();
    let mut joined: Vec<CoedgeSpec> = Vec::with_capacity(n + 2);
    for i in 0..n {
        let (a, ca) = &specs[i];
        joined.push(a.clone());
        let (b, _) = &specs[(i + 1) % n];
        let (_, end) = spec_ends(out, a);
        let (start, _) = spec_ends(out, b);
        if end == start {
            continue;
        }
        let chain = across(out, surface, r.cross_edges, end, start)
            .ok_or("a corner of a face that no one end section closes")?;
        let old = body.coedge(*ca);
        let range = body.edge(old.edge).range;
        let t_end = if old.reversed { range.0 } else { range.1 };
        let hint = body.coedge_uvs(*ca, &[t_end])[0];
        for (x, reversed) in chain {
            let from = if reversed {
                out.edge(x).end
            } else {
                out.edge(x).start
            };
            let seed = surface.project_near(out.vertex(from).point, hint);
            joined.push(CoedgeSpec {
                edge: x,
                reversed,
                pcurve: PCurve::Projected { seed },
            });
        }
    }
    // A contact curve two rounds share, run out and back, has nothing of the face between.
    while let Some(i) = (0..joined.len()).find(|&i| {
        let (a, b) = (&joined[i], &joined[(i + 1) % joined.len()]);
        joined.len() > 1 && a.edge == b.edge && a.reversed != b.reversed && r.met.contains(&a.edge)
    }) {
        let j = (i + 1) % joined.len();
        joined.remove(i.max(j));
        joined.remove(i.min(j));
    }
    Ok((joined, true))
}

/// The end sections that close a face's corner from `from` to `to`: one, or a chain of them where
/// the blend's end crosses the face past a short edge, each lying on the face's `surface`.
fn across(
    out: &Brep,
    surface: &Surface,
    cross_edges: &[EdgeId],
    from: VertexId,
    to: VertexId,
) -> Option<Vec<(EdgeId, bool)>> {
    let joins = |x: EdgeId, at: VertexId| {
        let xe = out.edge(x);
        if xe.start == at {
            Some((x, false))
        } else if xe.end == at {
            Some((x, true))
        } else {
            None
        }
    };
    let direct: Vec<(EdgeId, bool)> = cross_edges
        .iter()
        .filter_map(|&x| joins(x, from))
        .filter(|&(x, rev)| {
            let xe = out.edge(x);
            (if rev { xe.start } else { xe.end }) == to
        })
        .collect();
    match direct.as_slice() {
        [one] => return Some(vec![*one]),
        [] => {}
        _ => return None,
    }
    let on_face = |x: EdgeId| {
        let xe = out.edge(x);
        let Some(c) = xe.curve.as_ref() else {
            return false;
        };
        (0..=4).all(|i| {
            let t = xe.range.0 + (xe.range.1 - xe.range.0) * i as f64 / 4.0;
            signed_distance(surface, c.eval(t)).abs() <= tol::LINEAR
        })
    };
    let mut chain: Vec<(EdgeId, bool)> = Vec::new();
    let mut at = from;
    for _ in 0..4 {
        let next: Vec<(EdgeId, bool)> = cross_edges
            .iter()
            .filter(|&&x| !chain.iter().any(|c| c.0 == x) && on_face(x))
            .filter_map(|&x| joins(x, at))
            .collect();
        let [(x, rev)] = next.as_slice() else {
            return None;
        };
        chain.push((*x, *rev));
        let xe = out.edge(*x);
        at = if *rev { xe.start } else { xe.end };
        if at == to {
            return Some(chain);
        }
    }
    None
}

/// Whether a face the blend made or changed shares an edge with another face on its surface, so
/// [`unify`] would merge them.
fn mergeable(b: &Brep, sides: &[Option<usize>]) -> bool {
    let mut faces_of: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for c in b.coedge_ids() {
        faces_of
            .entry(b.coedge(c).edge)
            .or_default()
            .push(b.coedge_face(c));
    }
    faces_of.values().any(|fs| {
        fs.len() == 2
            && fs[0] != fs[1]
            && (sides[fs[0].index()].is_none() || sides[fs[1].index()].is_none())
            && same_domain(&b.face(fs[0]).surface, &b.face(fs[1]).surface)
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

    use super::super::{FORCE, Method, chamfer, fillet};
    use crate::{
        Error, Result, check,
        math::{Frame, Vec2, Vec3},
        measure,
        profile::{NamedSegment, Profile, ProfileLoop, Segment},
        sweep::{extrude, revolve},
        topo::Brep,
    };

    fn up() -> Frame {
        Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap()
    }

    fn prism(profile: &Profile, h: f64, id: &str) -> Brep {
        extrude(profile, &up(), h, id).unwrap()
    }

    fn block(w: f64, d: f64, h: f64) -> Brep {
        prism(&Profile::rect(Vec2::new(w / 2.0, d / 2.0), w, d), h, "b")
    }

    fn with(method: Method, make: &dyn Fn() -> Result<Brep>) -> Result<Brep> {
        FORCE.set(Some(method));
        let out = make();
        FORCE.set(None);
        out
    }

    /// The blend sewn, checked valid, equal to the tools' result in area, volume, faces and names,
    /// and to the hand-worked `volume` and `area`.
    fn sewn_as_tools(make: &dyn Fn() -> Result<Brep>, volume: f64, area: f64) -> Brep {
        let sewn = with(Method::Sewn, make).unwrap();
        let tools = with(Method::Tools, make).unwrap();
        assert!(check::check(&sewn).is_empty(), "{:?}", check::check(&sewn));
        let (sa, sv) = measure::area_and_volume(&sewn);
        let (ta, tv) = measure::area_and_volume(&tools);
        assert!(
            (sv - tv).abs() <= 1e-9 * tv,
            "volume {sv} vs the tools' {tv}"
        );
        assert!((sa - ta).abs() <= 1e-9 * ta, "area {sa} vs the tools' {ta}");
        assert!(
            (sv - volume).abs() <= 1e-9 * volume,
            "volume {sv} vs {volume}"
        );
        assert!((sa - area).abs() <= 1e-9 * area, "area {sa} vs {area}");
        assert_eq!(sewn.name_counts(), tools.name_counts());
        assert_eq!(sewn.face_count(), tools.face_count());
        sewn
    }

    /// ∫ₐᵇ f by Simpson's rule on 2000 panels, for the smooth integrands of hand values.
    fn simpson_on(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        let n = 2000;
        let h = (b - a) / n as f64;
        let inner: f64 = (1..n)
            .map(|i| f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 })
            .sum();
        (f(a) + inner + f(b)) * h / 3.0
    }

    fn simpson(f: &dyn Fn(f64) -> f64, b: f64) -> f64 {
        simpson_on(f, 0.0, b)
    }

    fn named(b: &Brep, name: &str) -> usize {
        b.name_counts().get(name).copied().unwrap_or(0)
    }

    #[test]
    fn a_straight_round_between_square_ends_is_sewn() {
        // A 40 × 30 × 20 box rounded at radius 5 along its 40 long top edge at y = 0: a quarter
        // cylinder replaces two 5 × 40 strips, and each end face loses a spandrel 25·(1 − π/4).
        let spandrel = 25.0 * (1.0 - PI / 4.0);
        let b = sewn_as_tools(
            &|| fillet(&block(40.0, 30.0, 20.0), &[("b.end", "b.v_min")], 5.0, "f"),
            24000.0 - 40.0 * spandrel,
            5200.0 - 400.0 + FRAC_PI_2 * 5.0 * 40.0 - 2.0 * spandrel,
        );
        assert_eq!(b.face_count(), 7);
        assert_eq!(named(&b, "f.round_0"), 1);
    }

    #[test]
    fn a_concave_round_adds_material_between_square_ends() {
        // An L of 40 × 10 and 10 × 30 legs, 20 high: radius 2 in its inner corner adds a spandrel
        // 4·(1 − π/4) along 20, trades two 2 × 20 strips for a quarter cylinder, and widens the
        // top and bottom by a spandrel each.
        let pts = [
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, 10.0),
            (10.0, 10.0),
            (10.0, 30.0),
            (0.0, 30.0),
        ]
        .map(|(x, y)| Vec2::new(x, y));
        let l = prism(
            &Profile::polygon(&pts, &["a", "b", "c", "d", "e", "f"]),
            20.0,
            "l",
        );
        let spandrel = 4.0 * (1.0 - PI / 4.0);
        let b = sewn_as_tools(
            &|| fillet(&l, &[("l.c", "l.d")], 2.0, "f"),
            600.0 * 20.0 + 20.0 * spandrel,
            1200.0 + 140.0 * 20.0 - 80.0 + FRAC_PI_2 * 2.0 * 20.0 + 2.0 * spandrel,
        );
        assert_eq!(named(&b, "f.round_0"), 1);
    }

    #[test]
    fn a_round_on_a_whole_circle_is_a_torus_with_its_own_seam() {
        // A disc of radius 10, 10 high, rounded at radius 2 round its top. Pappus: the spandrel's
        // centroid sits 2·(10 − 3π)/(12 − 3π) in from the rim, the quarter arc's 2·(1 − 2/π).
        let (rr, r, h) = (10.0, 2.0, 10.0);
        let disc = prism(&Profile::circle(Vec2::ZERO, 2.0 * rr), h, "d");
        let spandrel = r * r * (1.0 - PI / 4.0);
        let off = r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
        let arc = FRAC_PI_2 * r * 2.0 * PI * (rr - r * (1.0 - 2.0 / PI));
        let b = sewn_as_tools(
            &|| fillet(&disc, &[("d.end", "d.side")], r, "f"),
            PI * rr * rr * h - spandrel * 2.0 * PI * (rr - off),
            PI * rr * rr + PI * (rr - r).powi(2) + 2.0 * PI * rr * (h - r) + arc,
        );
        assert_eq!(b.face_count(), 4);
    }

    #[test]
    fn mitred_chamfers_share_their_end_on_the_plane_between_them() {
        // Chamfers of 2 along the top edges at y = 0 (40 long) and x = 40 (30 long) of a
        // 40 × 30 × 20 box, mitred at their corner. Each prism of section 2 runs along its
        // centroid line, 2/3 in from each face, to the mitre: 40 − 2/3 and 30 − 2/3. The top keeps
        // 38 × 28, the two walls 18 of their height, the far ends lose a triangle each, and each
        // bevel is a trapezoid 2√2 wide between its 40 (30) long foot and 38 (28) long head.
        let d = 2.0;
        let b = sewn_as_tools(
            &|| {
                chamfer(
                    &block(40.0, 30.0, 20.0),
                    &[("b.end", "b.v_min"), ("b.end", "b.u_max")],
                    d,
                    "c",
                )
            },
            24000.0 - d * d / 2.0 * (70.0 - 2.0 * d / 3.0),
            1200.0
                + 38.0 * 28.0
                + 70.0 * 18.0
                + (600.0 + 800.0 - d * d)
                + d * SQRT_2 * ((80.0 - d) + (60.0 - d)) / 2.0,
        );
        assert_eq!(named(&b, "c.bevel_0"), 1);
        assert_eq!(named(&b, "c.bevel_1"), 1);
    }

    #[test]
    fn three_rounds_close_with_a_sewn_sphere_octant() {
        // A 20 cube rounded at radius 4 on the three edges at one corner: each round runs 16 to
        // the corner's cube of side 4, which keeps an eighth of the ball.
        let pairs = [
            ("b.end", "b.u_max"),
            ("b.end", "b.v_max"),
            ("b.u_max", "b.v_max"),
        ];
        let spandrel = 16.0 * (1.0 - PI / 4.0);
        let b = sewn_as_tools(
            &|| fillet(&block(20.0, 20.0, 20.0), &pairs, 4.0, "f"),
            8000.0 - 3.0 * 16.0 * spandrel - (64.0 - PI * 64.0 / 6.0),
            2400.0 - 3.0 * 4.0 * 16.0 * 2.0 - 3.0 * 16.0 - 3.0 * spandrel
                + 3.0 * FRAC_PI_2 * 4.0 * 16.0
                + PI * 16.0 / 2.0,
        );
        assert_eq!(named(&b, "f.corner"), 1);
    }

    /// The area of a polygon by the shoelace formula.
    fn shoelace(p: &[Vec2]) -> f64 {
        (0..p.len())
            .map(|i| p[i].cross(p[(i + 1) % p.len()]))
            .sum::<f64>()
            / 2.0
    }

    #[test]
    fn three_rounds_at_a_sharp_corner_close_with_a_sphere_triangle() {
        // A kite V (0, 0), A (20, 0), D, B = 20·(cos 60°, sin 60°), square at A and B, 10 high,
        // rounded at radius 2 on the vertical edge at V and both top edges there. The ball 2 from
        // all three planes sits T = 2·cot 30° along each side from V, so each round stops T from
        // the corner. The cell between the ball's centre and the corner is three pyramids of
        // height 2 on T × 2 rectangles, and the ball's part of it is the solid angle of the
        // triangle of the normals: two right angles and π − α between the sides, so Ω = π − α.
        let (r, h, side, alpha) = (2.0_f64, 10.0, 20.0, PI / 3.0);
        let t = r / (alpha / 2.0).tan();
        let b_pt = Vec2::new(side * alpha.cos(), side * alpha.sin());
        let d_pt = Vec2::new(side, side * (alpha / 2.0).tan());
        let kite = [Vec2::ZERO, Vec2::new(side, 0.0), d_pt, b_pt];
        let body = prism(&Profile::polygon(&kite, &["va", "ad", "db", "bv"]), h, "k");
        let pairs = [("k.va", "k.bv"), ("k.end", "k.va"), ("k.end", "k.bv")];
        let base = shoelace(&kite);
        let s_v = r * t - r * r * (PI - alpha) / 2.0;
        let s_q = r * r * (1.0 - PI / 4.0);
        let omega = PI - alpha;
        let cell = t * r * r - omega * r.powi(3) / 3.0;
        // The top keeps the kite less a strip r wide along each top edge: P is T along VA and r
        // in, and B' is r along BD from B.
        let along_bd = (d_pt - b_pt) * (1.0 / d_pt.dist(b_pt));
        let top = shoelace(&[
            Vec2::new(t, r),
            Vec2::new(side, r),
            d_pt,
            b_pt + along_bd * r,
        ]);
        sewn_as_tools(
            &|| fillet(&body, &pairs, r, "f"),
            base * h - s_v * (h - r) - 2.0 * s_q * (side - t) - cell,
            (base - s_v)
                + top
                + 2.0 * (side - t) * (h - r)
                + 2.0 * (d_pt.dist(b_pt) * h - s_q)
                + (PI - alpha) * r * (h - r)
                + 2.0 * FRAC_PI_2 * r * (side - t)
                + omega * r * r,
        );
    }

    #[test]
    fn a_round_where_a_hemisphere_meets_its_base_is_a_torus() {
        // A hemisphere of radius 10 rounded at radius 2 along its equator. In the meridian half
        // plane the round's centre is (c, 2), c = √(8² − 2²), touching the base at (c, 0) and the
        // sphere at θ = atan(2/c). The region it takes off is bounded by the base, the sphere's
        // arc and the round's arc; Pappus turns ∮x²/2·dy round the axis into its volume.
        let (rr, r) = (10.0_f64, 2.0_f64);
        let c = ((rr - r).powi(2) - r * r).sqrt();
        let th = (r / c).atan();
        let s = th.sin();
        let sphere_arc = rr.powi(3) / 2.0 * (s - s.powi(3) / 3.0);
        let round_arc = -r / 2.0
            * (c * c * (s + 1.0)
                + 2.0 * c * r * ((th + FRAC_PI_2) / 2.0 + (2.0 * th).sin() / 4.0)
                + r * r * ((s - s.powi(3) / 3.0) + 2.0 / 3.0));
        let quarter = Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    NamedSegment::new(
                        Segment::Line {
                            a: Vec2::ZERO,
                            b: Vec2::new(rr, 0.0),
                        },
                        "base",
                    ),
                    NamedSegment::new(
                        Segment::Arc {
                            center: Vec2::ZERO,
                            radius: rr,
                            start: 0.0,
                            sweep: FRAC_PI_2,
                        },
                        "side",
                    ),
                    NamedSegment::new(
                        Segment::Line {
                            a: Vec2::new(0.0, rr),
                            b: Vec2::ZERO,
                        },
                        "axis",
                    ),
                ],
            }],
        };
        let frame = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
        let dome = revolve(&quarter, &frame, Vec3::ZERO, Vec3::Z, 2.0 * PI, "h").unwrap();
        let torus = 2.0 * PI * r * (c * (th + FRAC_PI_2) + r * (s + 1.0));
        let b = sewn_as_tools(
            &|| fillet(&dome, &[("h.side", "h.base")], r, "f"),
            2.0 / 3.0 * PI * rr.powi(3) - 2.0 * PI * (sphere_arc + round_arc),
            3.0 * PI * rr * rr - 2.0 * PI * rr * rr * s - PI * (rr * rr - c * c) + torus,
        );
        assert_eq!(b.face_count(), 3);
    }

    #[test]
    fn a_round_runs_past_a_chamfers_short_edge_onto_the_top() {
        // A 40 × 30 × 20 box chamfered 1 along its top edge at y = 0, then rounded at radius 2 on
        // its vertical edge at x = 40, y = 0, which now stops at z = 19 on the bevel. The round's
        // contact on x = 40 lies at y = 2, past the bevel's edge there (y ≤ 1), so the round ends
        // on the bevel and on the top. In (u, w) = (40 − x, y) the spandrel is u ≤ g(w),
        // g = 2 − √(4w − w²), and the body over it stands min(19 + w, 20) high; with w = 2 − 2·cos φ
        // each integral is smooth on either side of w = 1 (φ = π/3).
        let body = chamfer(&block(40.0, 30.0, 20.0), &[("b.end", "b.v_min")], 1.0, "c").unwrap();
        let g = |phi: f64| 2.0 - 2.0 * phi.sin();
        let dw = |phi: f64| 2.0 * phi.sin();
        let w = |phi: f64| 2.0 - 2.0 * phi.cos();
        let (mid, end) = (PI / 3.0, FRAC_PI_2);
        let spandrel = 4.0 * (1.0 - PI / 4.0);
        // The spandrel's area under the bevel (w ≤ 1) and under the top, and its rise above 19.
        let under_bevel = simpson_on(&|p| g(p) * dw(p), 0.0, mid);
        let under_top = simpson_on(&|p| g(p) * dw(p), mid, end);
        let rise = simpson_on(&|p| w(p) * g(p) * dw(p), 0.0, mid) + under_top;
        // The round's generator at φ is 19 + min(w, 1) long.
        let round = 2.0 * (19.0 * end + simpson_on(&w, 0.0, mid) + (end - mid));
        sewn_as_tools(
            &|| fillet(&body, &[("b.u_max", "b.v_min")], 2.0, "f"),
            24000.0 - 20.0 - 19.0 * spandrel - rise,
            5200.0 - 80.0 + 40.0 * SQRT_2
                - 1.0
                - under_top
                - SQRT_2 * under_bevel
                - 39.5
                - 38.0
                - spandrel
                + round,
        );
    }

    #[test]
    fn a_chamfer_into_an_inner_corner_opens_the_wall_past_it() {
        // An L of 40 × 10 and 10 × 30 legs, 20 high, chamfered 1 along the top of its inner face
        // y = 10, which ends square on the inner face x = 10 at the inner corner. That face lies
        // only at y ≥ 10, so the chamfer's end triangle, at 9 ≤ y ≤ 10, opens it past the corner:
        // the face gains the half square the far end x = 40 loses.
        let pts = [
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, 10.0),
            (10.0, 10.0),
            (10.0, 30.0),
            (0.0, 30.0),
        ]
        .map(|(x, y)| Vec2::new(x, y));
        let l = prism(
            &Profile::polygon(&pts, &["a", "b", "c", "d", "e", "f"]),
            20.0,
            "l",
        );
        sewn_as_tools(
            &|| chamfer(&l, &[("l.end", "l.c")], 1.0, "c"),
            12000.0 - 15.0,
            4000.0 - 60.0 + 30.0 * SQRT_2,
        );
    }

    /// A teardrop: lines from the origin at ±45° to where they touch a circle of radius `r` about
    /// (r√2, 0), and the 270° arc of it between, so the outline is one tangent chain from one
    /// side of the corner at the origin round to the other.
    fn teardrop(r: f64) -> Profile {
        let c = Vec2::new(r * SQRT_2, 0.0);
        let t = r / SQRT_2;
        let (t0, t1) = (Vec2::new(t, -t), Vec2::new(t, t));
        let seg = |segment, name: &str| NamedSegment::new(segment, name);
        Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    seg(
                        Segment::Line {
                            a: Vec2::ZERO,
                            b: t0,
                        },
                        "s0",
                    ),
                    seg(
                        Segment::Arc {
                            center: c,
                            radius: r,
                            start: -0.75 * PI,
                            sweep: 1.5 * PI,
                        },
                        "arc",
                    ),
                    seg(
                        Segment::Line {
                            a: t1,
                            b: Vec2::ZERO,
                        },
                        "s1",
                    ),
                ],
            }],
        }
    }

    #[test]
    fn a_chain_that_closes_on_itself_at_a_mitre() {
        // Chamfer 2 round the top of a teardrop of radius 10, 5 high. Its outline is the kite of
        // the 90° corner and the circle, area r², with the 270° sector, 3π/4·r², and runs 2r
        // along the lines and 3π/2·r round the arc. The bevel's triangles, area 2, have their
        // centroids 2/3 in, where the lines stop 2/3 short of the mitre and the arc's radius is
        // r − 2/3. The top keeps the teardrop of radius r − 2; each bevel band is 2√2 wide about
        // its middle line, r − 1 along each straight and at radius r − 1 round the arc.
        let (r, h, d) = (10.0, 5.0, 2.0);
        let body = prism(&teardrop(r), h, "t");
        let shape = 1.0 + 0.75 * PI;
        let along = 2.0 + 1.5 * PI;
        let b = sewn_as_tools(
            &|| chamfer(&body, &[("t.end", "t.arc")], d, "c"),
            r * r * shape * h - d * d / 2.0 * (r - d / 3.0) * along,
            (r * r + (r - d).powi(2)) * shape
                + r * along * (h - d)
                + d * SQRT_2 * (r - d / 2.0) * along,
        );
        assert_eq!(named(&b, "c.bevel_0"), 3);
    }

    #[test]
    fn a_round_chain_that_closes_on_itself_at_a_mitre() {
        // The teardrop rounded at radius 2: the mitre's ends are cut by the plane between the
        // lines along ellipses, which the tools cannot glue, so the tools cut the last straight
        // piece on its own. The spandrel, area r²(1 − π/4), and the quarter arc, length πr/2,
        // have their centroids r(10 − 3π)/(12 − 3π) and r(1 − 2/π) in from each face (Pappus).
        let (rr, h, r) = (10.0, 5.0, 2.0);
        let body = prism(&teardrop(rr), h, "t");
        let shape = 1.0 + 0.75 * PI;
        let along = 2.0 + 1.5 * PI;
        let inset = r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
        let b = sewn_as_tools(
            &|| fillet(&body, &[("t.end", "t.arc")], r, "f"),
            rr * rr * shape * h - r * r * (1.0 - PI / 4.0) * (rr - inset) * along,
            (rr * rr + (rr - r).powi(2)) * shape
                + rr * along * (h - r)
                + FRAC_PI_2 * r * (rr - r * (1.0 - 2.0 / PI)) * along,
        );
        assert_eq!(named(&b, "f.round_0"), 3);
    }

    /// The blend refused sewn, for the checker's `why`, and built by the tools to `volume` with
    /// `faces` faces.
    fn sewn_refused(make: &dyn Fn() -> Result<Brep>, why: &str, volume: f64, faces: usize) {
        match with(Method::Sewn, make) {
            Err(Error::Unsupported(m)) => assert!(m.contains(why), "{m}"),
            other => panic!("sewn: {:?}", other.map(|b| b.face_count())),
        }
        let b = make().unwrap();
        assert!(check::check(&b).is_empty(), "{:?}", check::check(&b));
        let v = measure::volume(&b);
        assert!(
            (v - volume).abs() <= 1e-9 * volume,
            "volume {v} vs {volume}"
        );
        assert_eq!(b.face_count(), faces);
    }

    #[test]
    fn a_hole_wholly_inside_the_removed_wedge_goes_with_it() {
        // A Ø1.6 hole along x about (y, z) = (1.2, 18.8) of a 40 × 30 × 20 box lies wholly above
        // the plane z = y + 16 of a chamfer 4 on the top edge at y = 0, and 9.6 − 0.8 = 8.8 from
        // the axis (y, z) = (8, 12) of a round 8 there, so either takes the whole hole away. Sewn,
        // the end faces would keep the hole's rims outside their cut-back outlines.
        let frame = Frame::new(Vec3::new(-1.0, 1.2, 18.8), Vec3::X, Vec3::Y).unwrap();
        let hole = extrude(&Profile::circle(Vec2::ZERO, 1.6), &frame, 42.0, "h").unwrap();
        let body = crate::boolean::difference(&block(40.0, 30.0, 20.0), &hole).unwrap();
        let stray = "lies outside the region its other loops bound";
        sewn_refused(
            &|| chamfer(&body, &[("b.end", "b.v_min")], 4.0, "c"),
            stray,
            24000.0 - 40.0 * 8.0,
            7,
        );
        sewn_refused(
            &|| fillet(&body, &[("b.end", "b.v_min")], 8.0, "f"),
            stray,
            24000.0 - 40.0 * 64.0 * (1.0 - PI / 4.0),
            7,
        );
    }

    #[test]
    fn a_void_wholly_inside_the_removed_wedge_goes_with_it() {
        // A 1 mm cube void about (20, 1, 19) lies above the chamfer plane z = y + 16 and outside
        // the round of radius 8 about (y, z) = (8, 12): its nearest corner is 6.5√2 from the axis.
        let at = Frame::new(Vec3::new(0.0, 0.0, 18.5), Vec3::Z, Vec3::X).unwrap();
        let void = extrude(
            &Profile::rect(Vec2::new(20.0, 1.0), 1.0, 1.0),
            &at,
            1.0,
            "v",
        )
        .unwrap();
        let body = crate::boolean::difference(&block(40.0, 30.0, 20.0), &void).unwrap();
        let stray = "lies outside its outer shell";
        sewn_refused(
            &|| chamfer(&body, &[("b.end", "b.v_min")], 4.0, "c"),
            stray,
            24000.0 - 40.0 * 8.0,
            7,
        );
        sewn_refused(
            &|| fillet(&body, &[("b.end", "b.v_min")], 8.0, "f"),
            stray,
            24000.0 - 40.0 * 64.0 * (1.0 - PI / 4.0),
            7,
        );
    }

    #[test]
    fn rounds_that_cross_at_a_crease_are_sewn_along_it() {
        // A half disc of radius 10, 5 high, rounded at radius 1 along its whole top rim: the
        // straight round and the semicircular one meet square at each end of the diameter, each
        // ending on the other's wall, and their tools joined into one hold the crease where the
        // cylinder crosses the torus.
        let half = Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    NamedSegment::new(
                        Segment::Line {
                            a: Vec2::new(-10.0, 0.0),
                            b: Vec2::new(10.0, 0.0),
                        },
                        "flat",
                    ),
                    NamedSegment::new(
                        Segment::Arc {
                            center: Vec2::ZERO,
                            radius: 10.0,
                            start: 0.0,
                            sweep: PI,
                        },
                        "round",
                    ),
                ],
            }],
        };
        let body = prism(&half, 5.0, "h");
        let pairs = [("h.end", "h.flat"), ("h.end", "h.round")];
        let make = || fillet(&body, &pairs, 1.0, "f");
        // Both rounds meet square walls, so at the section angle φ of either arc both strips are
        // δ = 1 − sin φ wide, at depth 1 − cos φ. The face left at that depth is the segment of
        // the circle of radius ρ = 10 − δ beyond the chord δ from its centre, and the crease runs
        // where the chord meets the circle: the straight round is 2√(ρ² − δ²) long there, and the
        // ring turns through π − 2·asin(δ/ρ) at radius ρ.
        let seg = |rho: f64, d: f64| rho * rho * (d / rho).acos() - d * (rho * rho - d * d).sqrt();
        let delta = |phi: f64| 1.0 - phi.sin();
        let rho = |phi: f64| 10.0 - delta(phi);
        let cut = simpson(
            &|phi| (50.0 * PI - seg(rho(phi), delta(phi))) * phi.sin(),
            FRAC_PI_2,
        );
        let straight = simpson(
            &|phi| 2.0 * (rho(phi).powi(2) - delta(phi).powi(2)).sqrt(),
            FRAC_PI_2,
        );
        let ring = simpson(
            &|phi| (PI - 2.0 * (delta(phi) / rho(phi)).asin()) * rho(phi),
            FRAC_PI_2,
        );
        let b = sewn_as_tools(
            &make,
            250.0 * PI - cut,
            seg(9.0, 1.0) + 50.0 * PI + 80.0 + 40.0 * PI + straight + ring,
        );
        assert_eq!(b.face_count(), 6);
    }

    const SIDES: [&str; 4] = ["b.v_min", "b.u_max", "b.v_max", "b.u_min"];

    /// The pairs of faces along the edges of a block's top and, where `bottom`, its bottom, and
    /// along its upright edges.
    fn block_edges(bottom: bool) -> Vec<(&'static str, &'static str)> {
        let mut pairs = Vec::new();
        for (i, side) in SIDES.iter().enumerate() {
            pairs.push(("b.end", *side));
            if bottom {
                pairs.push(("b.start", *side));
            }
            pairs.push((*side, SIDES[(i + 1) % 4]));
        }
        pairs
    }

    #[test]
    fn a_plate_rounded_at_half_its_thickness_is_sewn() {
        // Every edge of a 40 × 30 × 6 plate rounded at 3: every point within 3 of the 34 × 24
        // rectangle at mid-height. The sides are gone, the top and bottom rounds along each are
        // one half cylinder, and the two balls at each upright edge one quarter ball.
        let (core, rim, r) = (34.0 * 24.0, 2.0 * (34.0 + 24.0), 3.0);
        let b = sewn_as_tools(
            &|| fillet(&block(40.0, 30.0, 6.0), &block_edges(true), r, "f"),
            2.0 * r * core + PI * r * r / 2.0 * rim + 4.0 / 3.0 * PI * r.powi(3),
            2.0 * core + PI * r * rim + 4.0 * PI * r * r,
        );
        assert_eq!(b.face_count(), 10);
        assert_eq!(named(&b, "f.corner"), 4);
        for side in SIDES {
            assert_eq!(named(&b, side), 0, "{side}");
        }
    }

    #[test]
    fn a_rib_rounded_at_half_its_width_is_sewn() {
        // The top and upright edges of a 40 × 6 × 10 rib rounded at 3. Its plan is a stadium on a
        // 34 × 6 rectangle; above the rounds' axis at height 7 it is half a cylinder 34 long and
        // two quarter balls. The top and both ends are gone, and the bottom edge of each end with
        // them: the bottom's outline runs from one upright round's foot straight into the next.
        let (c, r, h) = (34.0, 3.0, 10.0);
        let (plan, rim) = (2.0 * r * c + PI * r * r, 2.0 * c + 2.0 * PI * r);
        let b = sewn_as_tools(
            &|| fillet(&block(40.0, 6.0, h), &block_edges(false), r, "f"),
            plan * (h - r) + PI * r * r / 2.0 * c + 2.0 / 3.0 * PI * r.powi(3),
            plan + rim * (h - r) + PI * r * c + 2.0 * PI * r * r,
        );
        assert_eq!(b.face_count(), 8);
        assert_eq!(named(&b, "f.corner"), 2);
        for gone in ["b.end", "b.u_min", "b.u_max"] {
            assert_eq!(named(&b, gone), 0, "{gone}");
        }
    }

    #[test]
    fn a_plate_of_slanted_sides_rounded_at_half_its_thickness_is_sewn() {
        // A plate 6 thick, 40 long, on a section 30 wide at the bottom whose sides lean in at 60°
        // and 45°, rounded at 3. The balls at the foot and head of a side leaning at α stand
        // 3·cot(α/2) and 3·tan(α/2) back along the edge up it, its whole length 6/sin α. The solid
        // is every point within 3 of the rectangle at mid-thickness 3 inside each side, which
        // lies 3·cot(α/2) in from where the side meets the bottom.
        let (base, l, h, r) = (30.0, 40.0, 6.0, 3.0_f64);
        let (alpha, beta) = (PI / 3.0, PI / 4.0);
        let pts = [
            Vec2::new(0.0, 0.0),
            Vec2::new(base, 0.0),
            Vec2::new(base - h / beta.tan(), h),
            Vec2::new(h / alpha.tan(), h),
        ];
        let body = prism(
            &Profile::polygon(&pts, &["bottom", "right", "top", "left"]),
            l,
            "b",
        );
        let sides = ["b.bottom", "b.right", "b.top", "b.left"];
        let mut pairs = Vec::new();
        for (i, side) in sides.iter().enumerate() {
            pairs.push(("b.start", *side));
            pairs.push(("b.end", *side));
            pairs.push((*side, sides[(i + 1) % 4]));
        }
        let across = base - r / (alpha / 2.0).tan() - r / (beta / 2.0).tan();
        let (core, rim) = (across * (l - h), 2.0 * (across + l - h));
        let b = sewn_as_tools(
            &|| fillet(&body, &pairs, r, "f"),
            2.0 * r * core + PI * r * r / 2.0 * rim + 4.0 / 3.0 * PI * r.powi(3),
            2.0 * core + PI * r * rim + 4.0 * PI * r * r,
        );
        assert_eq!(b.face_count(), 10);
        assert_eq!(named(&b, "f.corner"), 4);
    }

    #[test]
    fn a_round_far_shorter_than_its_radius_between_two_corners_is_sewn() {
        // A 40 × 30 × 6 plate rounded at 3 − 1e-5: each upright edge keeps a round 2e-5 long,
        // both of whose end sections lie within the tolerance of either corner's sphere. The
        // solid is every point within r of the box r inside each face.
        let r = 3.0 - 1e-5;
        let (a, b, c) = (40.0 - 2.0 * r, 30.0 - 2.0 * r, 6.0 - 2.0 * r);
        let (faces, edges) = (2.0 * (a * b + b * c + c * a), a + b + c);
        let out = sewn_as_tools(
            &|| fillet(&block(40.0, 30.0, 6.0), &block_edges(true), r, "f"),
            a * b * c + r * faces + PI * r * r * edges + 4.0 / 3.0 * PI * r.powi(3),
            faces + 2.0 * PI * r * edges + 4.0 * PI * r * r,
        );
        assert_eq!(out.face_count(), 26);
        assert_eq!(named(&out, "f.corner"), 8);
    }
}
