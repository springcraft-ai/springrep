//! Chamfers and fillets on solid edges. Each comes three ways: `fillet` and `chamfer` pick every
//! edge where the surfaces of two named faces meet; the `_named` forms take [`EdgeName`]s, which
//! can also pick only where the faces carrying the names meet; the `_edges` forms take the edges
//! themselves. The shaped blends (`shaped`) take edges or faces.
//!
//! A blend's material is the region between the edge and the blend face: a prism along a straight
//! edge, or a ring (whole or a sector) about a circular one, of the cross-section that the two
//! faces and the chamfer line or fillet arc bound, swept into a tool solid whose sides lie on the
//! two faces, so the blend surface's tangent contact lies on the tool's own edges. The tools' blend
//! faces are sewn into the body (`sew`); where they do not close up that way each region is cut
//! away at a convex edge and added at a concave one with the kernel's booleans instead.
//!
//! Each piece ends where its edge does. Pieces whose edges continue each other tangentially share
//! their end section exactly; two straight pieces that meet at a corner of one face, mirror images
//! of each other, are trimmed on the plane between them; two convex pieces at a corner each leaves
//! through the other's wall into air run on through it, and their crease is where the two cuts
//! meet; a piece whose edge runs into one other face is extended past it and trimmed by that face's
//! surface, and by the faces past any short edge its contact runs beyond (`run`); on a curved
//! face met at a glancing angle it runs on only until that surface takes in its whole section. A
//! concave piece whose strip is wider than its face runs over the face's convex edge and is
//! trimmed by the face past it. Three fillets
//! meeting at a corner of three planes close with the sphere touching all three.

pub(crate) mod shaped;

pub use shaped::{
    chamfer_distances, fillet_conic, fillet_faces, fillet_g2, fillet_radii, fillet_setback,
    fillet_setback_g2, fillet_variable, full_round,
};

use std::collections::{BTreeSet, HashMap};
use std::f64::consts::{PI, TAU};

mod corner;
mod saddle;
mod saddle_patch;
#[cfg(test)]
#[cfg(test)]
pub(crate) use saddle::corner as saddle_corner;
mod rolling;
mod run;
mod sew;

use crate::pick::faces_named;
use crate::{
    Error, Result, boolean, check,
    error::invalid,
    geom::{Curve, Surface},
    math::{Frame, Point3, Vec2, Vec3},
    profile::{NamedSegment, Profile, ProfileLoop, Segment},
    sweep, tol,
    topo::{Brep, CoedgeId, CoedgeSpec, EdgeId, FaceId, FaceName, Facts, OnFace, PCurve, VertexId},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    /// Setback `setback[k]` along the edge's face `k`, face 0 the one carrying a pair's first name.
    Chamfer { setback: [f64; 2] },
    /// Constant `radius`, tangent to both faces.
    Fillet { radius: f64 },
}

impl Kind {
    fn face_name(self, feature: &str, pair: usize) -> FaceName {
        match self {
            Kind::Chamfer { .. } => FaceName::new(feature, format!("bevel_{pair}")),
            Kind::Fillet { .. } => FaceName::new(feature, format!("round_{pair}")),
        }
    }
}

/// One edge pair of a blend: the faces named `a` and `b` (as `feature.face`). It picks every edge
/// where a face on `a`'s surface meets one on `b`'s, or with `exact` only where the faces carrying
/// those names meet, so a pair on two surfaces that meet more than once picks one of the edges.
#[derive(Clone, Copy, Debug)]
pub struct EdgeName<'a> {
    pub a: &'a str,
    pub b: &'a str,
    pub exact: bool,
}

fn by_surface<'a>(pairs: &[(&'a str, &'a str)]) -> Vec<EdgeName<'a>> {
    pairs
        .iter()
        .map(|&(a, b)| EdgeName { a, b, exact: false })
        .collect()
}

/// Chamfers every edge where a face named `pairs[i].0` meets one named `pairs[i].1` (names as
/// `feature.face`), in one operation; the bevel faces are `<feature>.bevel_<i>`.
pub fn chamfer(body: &Brep, pairs: &[(&str, &str)], distance: f64, feature: &str) -> Result<Brep> {
    chamfer_named(body, &by_surface(pairs), distance, feature)
}

/// [`chamfer`] of the edges `pairs` name.
pub fn chamfer_named(
    body: &Brep,
    pairs: &[EdgeName],
    distance: f64,
    feature: &str,
) -> Result<Brep> {
    blend(
        body,
        pairs,
        Kind::Chamfer {
            setback: [distance; 2],
        },
        feature,
    )
}

/// A chamfer of the edges `pairs` name set back `along_a` on each pair's face `a` and `along_b`
/// on its face `b`.
pub fn chamfer_two_named(
    body: &Brep,
    pairs: &[EdgeName],
    along_a: f64,
    along_b: f64,
    feature: &str,
) -> Result<Brep> {
    blend(
        body,
        pairs,
        Kind::Chamfer {
            setback: [along_a, along_b],
        },
        feature,
    )
}

/// Fillets every edge where a face named `pairs[i].0` meets one named `pairs[i].1`, in one
/// operation; the round faces are `<feature>.round_<i>` and a vertex blend `<feature>.corner`.
pub fn fillet(body: &Brep, pairs: &[(&str, &str)], radius: f64, feature: &str) -> Result<Brep> {
    fillet_named(body, &by_surface(pairs), radius, feature)
}

/// [`fillet`] of `edges` themselves, each between two faces. Every face keeps its names.
pub fn fillet_edges(body: &Brep, edges: &[EdgeId], radius: f64, feature: &str) -> Result<Brep> {
    let (apart, names) = apart_names(body, edges)?;
    let pairs: Vec<EdgeName> = names
        .iter()
        .map(|(a, b)| EdgeName { a, b, exact: true })
        .collect();
    Ok(own_names(
        body,
        fillet_named(&apart, &pairs, radius, feature)?,
    ))
}

/// [`chamfer`] of `edges` themselves, each between two faces, set back `distance` on both.
pub fn chamfer_edges(body: &Brep, edges: &[EdgeId], distance: f64, feature: &str) -> Result<Brep> {
    let (apart, names) = apart_names(body, edges)?;
    let pairs: Vec<EdgeName> = names
        .iter()
        .map(|(a, b)| EdgeName { a, b, exact: true })
        .collect();
    Ok(own_names(
        body,
        chamfer_named(&apart, &pairs, distance, feature)?,
    ))
}

/// A chamfer of `edge` itself set back `along_first` on face `first`, one of the two it lies
/// between, and `along_other` on the other.
pub fn chamfer_two_edges(
    body: &Brep,
    edge: EdgeId,
    first: FaceId,
    along_first: f64,
    along_other: f64,
    feature: &str,
) -> Result<Brep> {
    let (apart, names) = apart_names(body, &[edge])?;
    let (a, b) = &names[0];
    let on = format!("{APART}.f{}", first.index());
    let (a, b) = if on == *a {
        (a, b)
    } else if on == *b {
        (b, a)
    } else {
        return invalid("the face a chamfer's first distance lies on must be one of the edge's");
    };
    let pair = EdgeName { a, b, exact: true };
    let out = chamfer_two_named(&apart, &[pair], along_first, along_other, feature)?;
    Ok(own_names(body, out))
}

/// [`fillet_edges`] whose rounds meet over a concave edge in the crease where they cross, the
/// ball kept from swinging round it: the solid `body` grown by `radius` rounds this way, whose
/// faces moved out meet over the concave edge where the growth leaves it sharp.
pub(crate) fn fillet_edges_mitred(
    body: &Brep,
    edges: &[EdgeId],
    radius: f64,
    feature: &str,
) -> Result<Brep> {
    let (apart, names) = apart_names(body, edges)?;
    let pairs: Vec<EdgeName> = names
        .iter()
        .map(|(a, b)| EdgeName { a, b, exact: true })
        .collect();
    let out = blend_with(&apart, &pairs, Kind::Fillet { radius }, feature, false)?;
    Ok(own_names(body, out))
}

/// The feature the faces of a solid are named under while its edges are picked by id.
const APART: &str = "edge";

/// `out`, blended from [`apart_names`]' copy of `body`, with each face's own names in place of
/// the one it was picked by. A design's names may be shared by faces a boolean brought together
/// and would pick more edges than the ones asked for, so the blend runs on names of its own.
fn own_names(body: &Brep, mut out: Brep) -> Brep {
    let own: HashMap<String, &BTreeSet<FaceName>> = body
        .face_ids()
        .map(|f| (format!("f{}", f.index()), &body.face(f).names))
        .collect();
    for f in out.face_ids().collect::<Vec<_>>() {
        let names = out
            .face(f)
            .names
            .iter()
            .flat_map(|n| match own.get(&n.face) {
                Some(own) if n.feature == APART => own.iter().cloned().collect(),
                _ => vec![n.clone()],
            })
            .collect();
        out.set_face_names(f, names);
    }
    out
}

/// `body` with its faces named apart, and the pair of names of each of `edges`.
fn apart_names(body: &Brep, edges: &[EdgeId]) -> Result<(Brep, Vec<(String, String)>)> {
    let mut apart = body.clone();
    let name = |f: FaceId| format!("{APART}.f{}", f.index());
    for f in body.face_ids() {
        apart.set_face_names(
            f,
            std::iter::once(FaceName::new(APART, format!("f{}", f.index()))).collect(),
        );
    }
    let mut names: Vec<(String, String)> = Vec::new();
    for &e in edges {
        let faces: Vec<FaceId> = body
            .coedge_ids()
            .filter(|&c| body.coedge(c).edge == e)
            .map(|c| body.coedge_face(c))
            .collect();
        let [f, g] = faces[..] else {
            return invalid("a filleted edge must lie between two faces");
        };
        names.push((name(f), name(g)));
    }
    Ok((apart, names))
}

/// [`fillet`] of the edges `pairs` name.
pub fn fillet_named(body: &Brep, pairs: &[EdgeName], radius: f64, feature: &str) -> Result<Brep> {
    blend(body, pairs, Kind::Fillet { radius }, feature)
}

/// A picked edge: `faces[k]` is the face `coedges[k]` belongs to, `faces[0]` the one carrying the
/// first name of the pairs in `pairs`. An edge reached only by following a tangent chain carries
/// the pairs of the edge it continues.
#[derive(Clone, Debug)]
struct Picked {
    edge: EdgeId,
    pairs: BTreeSet<usize>,
    faces: [FaceId; 2],
    coedges: [CoedgeId; 2],
}

/// How a cross-section sweeps along its edge.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// Along a straight edge: the section is on `frame`'s xy plane, `frame.z` along the edge from
    /// its start, and the edge is `length` long.
    Prism { frame: Frame, length: f64 },
    /// About a circular edge's axis, `axis` through `centre`, from the angle `t0` of `circle`'s
    /// frame through `span`; section `x` is the distance from the axis and `y` the height along it.
    Ring {
        circle: Frame,
        t0: f64,
        span: f64,
        full: bool,
    },
}

impl Path {
    /// The frame whose xy plane holds the section at the edge's start.
    fn frame(&self) -> Frame {
        match *self {
            Path::Prism { frame, .. } => frame,
            Path::Ring { circle, t0, .. } => ring_frame(&circle, t0),
        }
    }

    /// Where the section point `q` lies at the edge's start (`end` false) or end.
    fn place(&self, q: Vec2, end: bool) -> Point3 {
        match *self {
            Path::Prism { frame, length } => {
                frame.to_world(Vec3::new(q.x, q.y, if end { length } else { 0.0 }))
            }
            Path::Ring {
                circle, t0, span, ..
            } => {
                let a = if end { t0 + span } else { t0 };
                circle.origin + circle.radial(a) * q.x + circle.z * q.y
            }
        }
    }
}

/// The section plane of a ring at angle `a`: `x` radial, `y` along the axis.
fn ring_frame(circle: &Frame, a: f64) -> Frame {
    let x = circle.radial(a);
    Frame {
        origin: circle.origin,
        x,
        y: circle.z,
        z: x.cross(circle.z),
    }
}

/// How a face crosses the section plane: a straight line, or a circle about `centre`.
#[derive(Clone, Copy, Debug)]
enum Trace {
    Line,
    Circle { centre: Vec2, radius: f64 },
}

/// The blend's cross-section: the edge at `p`, face `k` leaving it along the unit `m[k]` with
/// outward normal `n[k]`, on the curve `trace[k]`.
#[derive(Clone, Copy, Debug)]
struct Section {
    path: Path,
    p: Vec2,
    m: [Vec2; 2],
    n: [Vec2; 2],
    trace: [Trace; 2],
    convex: bool,
}

/// The cross-section's outline, its blend side named `blend`; the chord from the edge to where it
/// leaves each face, and the largest distance of any of its points from the edge.
struct Outline {
    profile: Profile,
    reach: [(Vec2, f64); 2],
    radius: f64,
}

/// What one end of a piece does at its edge's vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
enum End {
    /// Stops at the vertex, its end section on a face square to it or on its neighbour's.
    Exact,
    /// A straight piece stops this far short of the vertex, where a sphere corner takes over.
    Short(f64),
    /// Runs past the vertex and is trimmed by a surface through it.
    Trim(Clip),
}

/// A surface through a vertex, and the side of it a piece keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Clip {
    /// Keeps the points `x` with `(x − origin)·keep ≥ 0`.
    Plane { origin: Point3, keep: Vec3 },
    /// The cylinder or cone of radius `r0 + slope·h` at height `h` along `axis` from `origin`,
    /// keeping its inside or its outside.
    About {
        origin: Point3,
        axis: Vec3,
        r0: f64,
        slope: f64,
        inside: bool,
    },
}

pub(crate) fn blend(body: &Brep, pairs: &[EdgeName], kind: Kind, feature: &str) -> Result<Brep> {
    blend_with(body, pairs, kind, feature, true)
}

/// [`blend`], the ball's swing round concave edges between its rounds put back where `swing`.
fn blend_with(
    body: &Brep,
    pairs: &[EdgeName],
    kind: Kind,
    feature: &str,
    swing: bool,
) -> Result<Brep> {
    let size = match kind {
        Kind::Chamfer { setback } => setback[0].min(setback[1]),
        Kind::Fillet { radius } => radius,
    };
    if !(size.is_finite() && size > tol::LINEAR) {
        return invalid("a blend's size must be positive");
    }
    let picked = pick(body, pairs, true)?;
    // Concave and convex edges meeting at a vertex: the concave ones are blended first, and the
    // convex ones then run on round the rounds those leave, as a tangent chain picked again.
    let convex: Vec<Option<bool>> = picked
        .iter()
        .map(|pk| run::convex(body, pk.coedges[0]))
        .collect();
    // A saddle's four edges are blended together, so their vertex does not split them.
    let saddles: BTreeSet<VertexId> = match kind {
        Kind::Fillet { radius } => picked_at_vertices(body, &picked)
            .into_iter()
            .filter(|(v, at)| saddle::find(body, &picked, at, *v, radius).is_some())
            .map(|(v, _)| v)
            .collect(),
        Kind::Chamfer { .. } => BTreeSet::new(),
    };
    let mut sense: HashMap<VertexId, [bool; 2]> = HashMap::new();
    for (pk, c) in picked.iter().zip(&convex) {
        let Some(c) = *c else { continue };
        let e = body.edge(pk.edge);
        for v in [e.start, e.end] {
            if !saddles.contains(&v) {
                sense.entry(v).or_default()[usize::from(c)] = true;
            }
        }
    }
    if sense.values().any(|s| s[0] && s[1]) {
        let concave: Vec<Picked> = picked
            .iter()
            .zip(&convex)
            .filter(|(_, c)| **c == Some(false))
            .map(|(pk, _)| pk.clone())
            .collect();
        let out = blend_picked(body, pairs, concave, kind, feature, swing)?;
        let again = pick(&out, pairs, false)?;
        return blend_picked(&out, pairs, again, kind, feature, swing);
    }
    blend_picked(body, pairs, picked, kind, feature, swing)
}

/// [`blend`] of the edges `picked` from `pairs`; where `swing`, a fillet's ball swings round each
/// sharp concave edge two of its pieces meet over ([`corner::sharp`]).
fn blend_picked(
    body: &Brep,
    pairs: &[EdgeName],
    picked: Vec<Picked>,
    kind: Kind,
    feature: &str,
    swing: bool,
) -> Result<Brep> {
    // A tangent chain with an edge no constant section sweeps (neither a line nor a circle about
    // both faces' axis) takes the rolling ball's surface all along. Each such chain is filleted
    // in turn and the rest picked again from the result, where that chain's edges are gone.
    if let Kind::Fillet { radius } = kind {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (i, pk) in picked.iter().enumerate() {
            let e = body.edge(pk.edge);
            let touching: Vec<usize> = (0..groups.len())
                .filter(|&g| {
                    groups[g].iter().any(|&j| {
                        let f = body.edge(picked[j].edge);
                        [f.start, f.end]
                            .iter()
                            .any(|v| *v == e.start || *v == e.end)
                    })
                })
                .collect();
            let mut merged = vec![i];
            for &g in touching.iter().rev() {
                merged.extend(groups.remove(g));
            }
            groups.push(merged);
        }
        // Only a pair the closed forms do not cover goes to the rolling ball. A ball the closed
        // form shows cannot fit is the request's fault, and stays with the closed form to say so.
        let swept = |pk: &Picked| !matches!(section(body, pk), Err(Error::Unsupported(_)));
        let rolled = groups
            .iter()
            .find(|g| rolling_forced() || !g.iter().all(|&i| swept(&picked[i])));
        if let Some(group) = rolled {
            let chain: Vec<Picked> = group.iter().map(|&i| picked[i].clone()).collect();
            let names: Vec<BTreeSet<FaceName>> = chain
                .iter()
                .map(|pk| {
                    pk.pairs
                        .iter()
                        .map(|&p| kind.face_name(feature, p))
                        .collect()
                })
                .collect();
            let out = rolling::fillet(body, &chain, radius, &names)?;
            if groups.len() == 1 {
                return Ok(out);
            }
            // The rest picked again under their own indices, which name their faces.
            let rest = pick(&out, pairs, false)?;
            return if rest.is_empty() {
                Ok(out)
            } else {
                blend_picked(&out, pairs, rest, kind, feature, swing)
            };
        }
    }
    // A fillet larger than a convex corner round it runs over: the arc leaves the chain, the
    // straight pieces beside it end on the corner's planes, and a corner tool cuts the rest.
    let swallowed = match kind {
        Kind::Fillet { radius } => {
            let mut sw = corner::swallowed(body, &picked, radius)?;
            if swing {
                sw.extend(corner::sharp(body, &picked, radius)?);
            }
            sw
        }
        Kind::Chamfer { .. } => Vec::new(),
    };
    let mut corner_ends: HashMap<(EdgeId, VertexId), Clip> = HashMap::new();
    for sw in &swallowed {
        for (e, v, clip) in sw.ends {
            corner_ends.insert((e, v), clip);
        }
    }
    let picked: Vec<Picked> = {
        let arcs: BTreeSet<usize> = swallowed.iter().filter_map(|sw| sw.arc).collect();
        picked
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !arcs.contains(i))
            .map(|(_, p)| p)
            .collect()
    };
    let by_vertex = picked_at_vertices(body, &picked);

    let mut sections = Vec::with_capacity(picked.len());
    let mut outlines = Vec::with_capacity(picked.len());
    let mut spills = Vec::with_capacity(picked.len());
    for pk in &picked {
        let s = section(body, pk)?;
        let o = outline(&s, kind)?;
        spills.push(check_width(body, pk, &s, &o)?);
        sections.push(s);
        outlines.push(o);
    }
    if let Kind::Fillet { radius } = kind
        && let Some(o) = overlap(body, &picked, &sections, &outlines)
        && let Some((a, b)) = swallowed_strip(body, &picked, &o)
    {
        let out = shaped::fillet_faces(body, a, b, radius, feature)?;
        let rest = pick(&out, pairs, false)?;
        return if rest.is_empty() {
            Ok(out)
        } else {
            blend_picked(&out, pairs, rest, kind, feature, swing)
        };
    }
    check_apart(body, &picked, &sections, &outlines)?;

    let mut ends = vec![[End::Exact; 2]; picked.len()];
    let mut corners: Vec<(VertexId, Ball)> = Vec::new();
    let mut saddles: Vec<saddle::Saddle> = Vec::new();
    let mut joints: Vec<(usize, usize)> = Vec::new();
    let mut creases: Vec<[(usize, usize); 2]> = Vec::new();
    let mut beyonds: Vec<[std::result::Result<run::Beyond, String>; 2]> =
        vec![[Ok(run::Beyond::default()), Ok(run::Beyond::default())]; picked.len()];
    let mut vertices: Vec<&VertexId> = by_vertex.keys().collect();
    vertices.sort();
    for &v in vertices {
        let at = &by_vertex[&v];
        match at.as_slice() {
            [i] => {
                let end = end_index(body, picked[*i].edge, v);
                if let Some(&clip) = corner_ends.get(&(picked[*i].edge, v)) {
                    ends[*i][end] = End::Trim(clip);
                    continue;
                }
                ends[*i][end] = free_end(body, &picked[*i], &outlines[*i], v)?;
                beyonds[*i][end] =
                    run::beyond(body, &picked, &picked[*i], &sections[*i], &outlines[*i], v);
            }
            [i, j] => {
                let (ei, ej) = (
                    end_index(body, picked[*i].edge, v),
                    end_index(body, picked[*j].edge, v),
                );
                // Pieces meeting over a sharp concave corner end on its planes; its tool takes
                // the rest.
                if let (Some(&ci), Some(&cj)) = (
                    corner_ends.get(&(picked[*i].edge, v)),
                    corner_ends.get(&(picked[*j].edge, v)),
                ) {
                    ends[*i][ei] = End::Trim(ci);
                    ends[*j][ej] = End::Trim(cj);
                    continue;
                }
                // Edges across a vertex from each other, sharing no face (a saddle's two convex
                // edges), blend apart: each piece ends as if it met no other.
                if !picked[*i]
                    .faces
                    .iter()
                    .any(|f| picked[*j].faces.contains(f))
                {
                    for (k, end) in [(*i, ei), (*j, ej)] {
                        ends[k][end] = free_end(body, &picked[k], &outlines[k], v)?;
                        beyonds[k][end] =
                            run::beyond(body, &picked, &picked[k], &sections[k], &outlines[k], v);
                    }
                    continue;
                }
                if continues(body, &picked, &sections, &outlines, (*i, ei), (*j, ej), v) {
                    joints.push((*i, *j));
                    continue;
                }
                // Pieces that meet at an angle stay separate tools, cut one after the other: the
                // second cut meets the first's blend surface across it and leaves their crease,
                // where a union of the two would have to resolve their shared faces at a seam.
                if let Some([ci, cj]) = mitre(body, &picked, &sections, (*i, *j), v)? {
                    ends[*i][ei] = End::Trim(ci);
                    ends[*j][ej] = End::Trim(cj);
                    continue;
                }
                // At a convex corner each edge runs on past the vertex into air, so each piece runs
                // on through the other's wall.
                if leaves_into_air(body, &picked, &sections, (*i, *j), v) {
                    ends[*i][ei] = through(body, &picked[*i], &outlines[*i], v)?;
                    ends[*j][ej] = through(body, &picked[*j], &outlines[*j], v)?;
                    creases.push([(*i, ei), (*j, ej)]);
                    continue;
                }
                {
                    let at = body.vertex(v).point;
                    let walls = |pk: &Picked| {
                        pk.faces
                            .map(|f| {
                                let fc = body.face(f);
                                let names: Vec<String> =
                                    fc.names.iter().map(|n| n.to_string()).collect();
                                format!("{}:{}", names.join("|"), fc.surface.kind())
                            })
                            .join(" / ")
                    };
                    return Err(Error::Unsupported(format!(
                        "{} that meet at a vertex other than as mirror images on one plane, at \
                         ({:.3}, {:.3}, {:.3}) between [{}] and [{}]",
                        plural(kind),
                        at.x,
                        at.y,
                        at.z,
                        walls(&picked[*i]),
                        walls(&picked[*j]),
                    )));
                }
            }
            _ => {
                // A saddle's fillets stop at the cube about it, the corner closed after.
                if let Kind::Fillet { radius } = kind
                    && let Some(s) = saddle::find(body, &picked, at, v, radius)
                {
                    for &i in at {
                        ends[i][end_index(body, picked[i].edge, v)] = End::Short(radius);
                    }
                    saddles.push(s);
                    continue;
                }
                let Kind::Fillet { radius } = kind else {
                    return Err(Error::Unsupported(
                        "three or more chamfers that meet at a vertex".into(),
                    ));
                };
                let ball = corner_ball(body, &picked, &sections, at, v, radius)?;
                for &i in at {
                    let end = end_index(body, picked[i].edge, v);
                    let along = leaving(body, picked[i].edge, v)
                        .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
                    // Each round stops where its section passes through the ball's centre.
                    let short = (ball.centre - body.vertex(v).point).dot(along);
                    if short <= tol::LINEAR {
                        return Err(Error::Unsupported(
                            "fillets meeting at a corner their ball does not stand back from"
                                .into(),
                        ));
                    }
                    ends[i][end] = End::Short(short);
                }
                corners.push((v, ball));
            }
        }
    }

    let mut tools = Vec::with_capacity(picked.len());
    for i in 0..picked.len() {
        let tool = piece(&sections[i], &outlines[i], ends[i], feature)?;
        let tool = trimmed(
            tool,
            &sections[i],
            &outlines[i],
            ends[i],
            &beyonds[i],
            feature,
        )?;
        tools.push(spilled(body, tool, &spills[i], feature)?);
    }
    let sewing = std::time::Instant::now();
    let sew_tools = if spills.iter().any(|s| !s.is_empty()) {
        Err("a concave blend that runs over the edge of its face".to_string())
    } else {
        sewing_tools(
            body,
            &picked,
            &tools,
            &Ends {
                sections: &sections,
                outlines: &outlines,
                ends: &ends,
                beyonds: &beyonds,
                creases: &creases,
            },
            feature,
        )
    };
    let names: Vec<BTreeSet<FaceName>> = picked
        .iter()
        .map(|pk| {
            pk.pairs
                .iter()
                .map(|&p| kind.face_name(feature, p))
                .collect()
        })
        .collect();
    let convex: Vec<bool> = sections.iter().map(|s| s.convex).collect();
    let balls: Vec<(Vec<usize>, Ball)> = corners
        .into_iter()
        .map(|(v, ball)| (by_vertex[&v].clone(), ball))
        .collect();

    // Each corner tool, the pieces it lies between, and whether it joins the first of them.
    let mut corner_tools: Vec<(usize, usize, Brep, bool)> = Vec::new();
    for sw in swallowed {
        let index = |e: EdgeId| picked.iter().position(|pk| pk.edge == e);
        let (Some(i), Some(j)) = (index(sw.ends[0].0), index(sw.ends[1].0)) else {
            return Err(Error::Numerical(
                "a fillet's corner lost its straight pieces".into(),
            ));
        };
        let k_plate = picked[i]
            .faces
            .iter()
            .position(|&f| f == sw.plate)
            .expect("a corner's piece lies on the plate");
        let mut tool = sw.tool;
        for f in tool.face_ids().collect::<Vec<_>>() {
            let which = tool
                .face(f)
                .names
                .iter()
                .find(|n| n.feature == corner::MARK)
                .cloned();
            let Some(which) = which else { continue };
            tool.face_mut(f).names = match which.face.as_str() {
                "patch" => names[i].union(&names[j]).cloned().collect(),
                "plate" => BTreeSet::from([FaceName::new(SIDE_MARK, format!("{i}_{k_plate}"))]),
                // A sharp corner's ends lie on the pieces' end sections, which they meet face to face.
                "start" | "end" | "axis" => BTreeSet::new(),
                other => {
                    let index = other
                        .strip_prefix("face_")
                        .expect("a corner tool face's name");
                    BTreeSet::from([FaceName::new(SIDE_MARK, format!("r_{index}"))])
                }
            };
        }
        corner_tools.push((i, j, tool, sw.arc.is_some()));
    }
    // The corner tools only join the straight pieces as tools: sewing has no corner patch.
    let sewn = if corner_tools.is_empty() && saddles.is_empty() {
        sew_tools.and_then(|(tools, beyond)| {
            sew::sewn(
                body, &picked, &tools, &beyond, &convex, &names, &balls, feature,
            )
        })
    } else {
        Err("a fillet with a corner patch".to_string())
    };
    let saddled = |out: Result<Brep>| -> Result<Brep> {
        let mut out = out?;
        for s in &saddles {
            out = saddle::close(&out, s, s.edges.map(|i| &names[i]))?;
        }
        Ok(out)
    };
    let sewing = sewing.elapsed();
    #[cfg(test)]
    match FORCE.get() {
        Some(Method::Sewn) => return sewn.map_err(Error::Unsupported),
        Some(Method::Tools) => {
            return saddled(by_tools(
                body,
                &picked,
                &sections,
                tools,
                &names,
                &joints,
                &balls,
                &corner_tools,
                feature,
            ));
        }
        None => {}
    }
    let verify = verifying();
    let tooled = |tools| {
        saddled(by_tools(
            body,
            &picked,
            &sections,
            tools,
            &names,
            &joints,
            &balls,
            &corner_tools,
            feature,
        ))
    };
    match sewn {
        Ok(out) if !verify => Ok(out),
        Ok(out) => {
            let cutting = std::time::Instant::now();
            let tooled = tooled(tools);
            compare(out, tooled, sewing, cutting.elapsed(), feature)
        }
        Err(why) => {
            if verify {
                eprintln!("springrep-blend: {feature} tool ({why})");
            }
            tooled(tools)
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Sends this thread's fillets on closed edges round the rolling ball even where a constant
    /// section sweeps them, so tests can hold it to the exact construction.
    static ROLLING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn force_rolling(on: bool) {
    ROLLING.set(on);
}

fn rolling_forced() -> bool {
    #[cfg(test)]
    {
        ROLLING.get()
    }
    #[cfg(not(test))]
    {
        false
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
enum Method {
    Sewn,
    Tools,
}

#[cfg(test)]
thread_local! {
    /// The one way this thread's blends are made, in tests, in place of sewing with the tools to
    /// fall back on.
    static FORCE: std::cell::Cell<Option<Method>> = const { std::cell::Cell::new(None) };
}

/// Whether blends run both ways and compare (`SPRINGREP_BLEND_VERIFY` set), for a run by hand.
fn verifying() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SPRINGREP_BLEND_VERIFY").is_some())
}

/// The sewn result, when it equals the tools' to 1e-9 of its area and volume with the same faces
/// named alike; otherwise an error naming the difference.
fn compare(
    sewn: Brep,
    tooled: Result<Brep>,
    sewing: std::time::Duration,
    cutting: std::time::Duration,
    feature: &str,
) -> Result<Brep> {
    let tooled = match tooled {
        Ok(t) => t,
        Err(e) => {
            eprintln!("springrep-blend: {feature} sewn (tools fail: {e})");
            return Ok(sewn);
        }
    };
    let (sa, sv) = crate::measure::area_and_volume(&sewn);
    let (ta, tv) = crate::measure::area_and_volume(&tooled);
    let (da, dv) = ((sa - ta).abs() / ta.abs(), (sv - tv).abs() / tv.abs());
    let same_names = sewn.name_counts() == tooled.name_counts();
    eprintln!(
        "springrep-blend: {feature} sewn (area {da:.1e}, volume {dv:.1e}, faces {} vs {}, names {}, \
         sewing {:.3} ms, cutting {:.3} ms)",
        sewn.face_count(),
        tooled.face_count(),
        if same_names { "same" } else { "differ" },
        sewing.as_secs_f64() * 1e3,
        cutting.as_secs_f64() * 1e3,
    );
    if da > 1e-9 || dv > 1e-9 || !same_names {
        return Err(Error::InvalidResult(vec![format!(
            "the sewn blend differs from the tools': area {sa} vs {ta}, volume {sv} vs {tv}, \
             names {:?} vs {:?}",
            sewn.name_counts(),
            tooled.name_counts()
        )]));
    }
    Ok(sewn)
}

/// The blend made by cutting each piece from the body, or adding it, with a boolean.
#[allow(clippy::too_many_arguments)]
fn by_tools(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    tools: Vec<Brep>,
    names: &[BTreeSet<FaceName>],
    joints: &[(usize, usize)],
    balls: &[(Vec<usize>, Ball)],
    corners: &[(usize, usize, Brep, bool)],
    feature: &str,
) -> Result<Brep> {
    let mut cut_pieces: Vec<(usize, Brep)> = Vec::new();
    let mut add_pieces: Vec<(usize, Brep)> = Vec::new();
    for (i, tool) in tools.into_iter().enumerate() {
        let tool = marked(tool, i, &names[i]);
        if sections[i].convex {
            cut_pieces.push((i, tool));
        } else {
            add_pieces.push((i, tool));
        }
    }
    // A corner tool joins the straight piece before it, which it meets face to face on that
    // corner's plane; the piece after it is cut on its own and meets the corner's other plane in
    // the body. Joining both would close a ring of pieces round a closed chain.
    // A sharp concave corner's tool is cut on its own once both pieces are: its patch comes to a
    // point on the concave edge, which a join would put on the piece's side.
    let mut apart: Vec<(bool, Brep)> = Vec::new();
    for (i, _j, corner, joins) in corners {
        if !joins {
            apart.push((sections[*i].convex, corner.clone()));
            continue;
        }
        let pieces = if sections[*i].convex {
            &mut cut_pieces
        } else {
            &mut add_pieces
        };
        let Some(pi) = pieces.iter().position(|p| p.0 == *i) else {
            return Err(Error::Numerical(
                "a fillet's corner lost the piece it joins".into(),
            ));
        };
        let facts = shared_edges(&pieces[pi].1, corner);
        pieces[pi].1 = boolean::boolean_with(&pieces[pi].1, corner, boolean::Op::Union, &facts)?;
    }
    let mut cuts = joined(joints, cut_pieces)?;
    let mut adds = joined(joints, add_pieces)?;
    // First: on the body as it is, the patch's point lies on the concave edge alone, where once
    // the pieces are cut it would land on the vertex their ends and the walls' rims meet at.
    for (convex, corner) in apart.into_iter().rev() {
        if convex {
            cuts.insert(0, corner);
        } else {
            adds.insert(0, corner);
        }
    }
    for (_, ball) in balls {
        cuts.push(corner_tool(ball, feature)?);
    }

    let mut out = body.clone();
    for tool in adds {
        let (tool, facts) = sides_on_body(tool, body, picked, &out);
        out = boolean::boolean_with(&out, &tool, boolean::Op::Union, &facts)?;
    }
    for tool in cuts {
        let (tool, facts) = sides_on_body(tool, body, picked, &out);
        out = boolean::boolean_with(&out, &tool, boolean::Op::Difference, &facts)?;
    }
    Ok(out)
}

/// Each edge of one solid that lies on a face of the other, as facts for a boolean of the two. A
/// corner patch is tangent to the pieces it joins along the curves they share, where a boolean left
/// to find the meeting finds it only to within the band the two stay within the tolerance over.
/// Each edge is checked along its length before it is given.
fn shared_edges(a: &Brep, b: &Brep) -> Facts {
    let mut facts = Facts::default();
    for (side, own, other) in [(0, a, b), (1, b, a)] {
        for f in other.face_ids() {
            let s = &other.face(f).surface;
            let dom = boolean::face_domain(other, f);
            for e in own.edge_ids() {
                let x = own.edge(e);
                let Some(c) = &x.curve else { continue };
                let at = |k: usize| c.eval(x.range.0 + (x.range.1 - x.range.0) * k as f64 / 16.0);
                let on = (0..=16)
                    .all(|k| boolean::csi::signed_distance(s, at(k)).abs() <= 0.1 * tol::LINEAR);
                if on && dom.locate(at(8)) != boolean::domain::Where::Outside {
                    facts.on_face.push(OnFace {
                        side,
                        edge: e,
                        face: f,
                    });
                }
            }
        }
    }
    facts
}

/// The names a piece's section sides sweep into faces with: side `k` lies on the picked edge's
/// face `k`.
const SIDES: [&str; 2] = ["side_0", "side_1"];

/// What a side face of a piece is marked with until the tool meets the body: the piece and side.
const SIDE_MARK: &str = "#side";

/// `tool` with its side marks dropped, and the fact that every edge of a side face lies on each
/// face of `out` on the surface of the body face that side was swept along.
fn sides_on_body(mut tool: Brep, body: &Brep, picked: &[Picked], out: &Brep) -> (Brep, Facts) {
    let mut facts = Facts::default();
    for f in tool.face_ids().collect::<Vec<_>>() {
        let marks: Vec<(usize, usize)> = tool
            .face(f)
            .names
            .iter()
            .filter(|n| n.feature == SIDE_MARK)
            .filter_map(|n| {
                let (i, k) = n.face.split_once('_')?;
                Some((i.parse().ok()?, k.parse().ok()?))
            })
            .collect();
        // A corner tool's face on a body face no piece lies on: `r_<index>` names it.
        let bodies: Vec<FaceId> = tool
            .face(f)
            .names
            .iter()
            .filter(|n| n.feature == SIDE_MARK)
            .filter_map(|n| n.face.strip_prefix("r_")?.parse::<usize>().ok())
            .filter_map(|ix| body.face_ids().nth(ix))
            .collect();
        if marks.is_empty() && bodies.is_empty() {
            continue;
        }
        let edges: BTreeSet<EdgeId> = tool
            .face(f)
            .loops
            .iter()
            .flat_map(|&lp| tool.lp(lp).coedges.iter().map(|&c| tool.coedge(c).edge))
            .filter(|&e| tool.edge(e).curve.is_some())
            .collect();
        let mut on: BTreeSet<FaceId> = BTreeSet::new();
        let walls = marks
            .iter()
            .map(|&(i, k)| picked[i].faces[k])
            .chain(bodies.iter().copied());
        for w in walls {
            let wall = &body.face(w).surface;
            on.extend(
                out.face_ids()
                    .filter(|&g| boolean::same_domain(&out.face(g).surface, wall)),
            );
        }
        // Faces apart on one surface (two plates' rounds on one cylinder) each hold only their
        // own stretch of it: an edge lies on a face only where its middle lies inside it.
        for &face in &on {
            let dom = boolean::face_domain(out, face);
            for &edge in &edges {
                let x = tool.edge(edge);
                let Some(c) = &x.curve else { continue };
                if dom.locate(c.eval(0.5 * (x.range.0 + x.range.1)))
                    == boolean::domain::Where::Outside
                {
                    continue;
                }
                facts.on_face.push(OnFace {
                    side: 1,
                    edge,
                    face,
                });
            }
        }
        tool.face_mut(f).names.clear();
    }
    (tool, facts)
}

fn plural(kind: Kind) -> &'static str {
    match kind {
        Kind::Chamfer { .. } => "chamfers",
        Kind::Fillet { .. } => "fillets",
    }
}

pub(crate) fn has_name(body: &Brep, f: FaceId, name: &str) -> bool {
    body.face(f).names.iter().any(|n| n.to_string() == name)
}

pub(crate) fn picked_edges(body: &Brep, pair: &EdgeName) -> Result<Vec<EdgeId>> {
    Ok(pick(body, &[*pair], true)?
        .into_iter()
        .map(|pk| pk.edge)
        .collect())
}

/// Every edge each pair names, in edge order, then every edge a picked one runs on into
/// tangentially: a blend runs round the whole tangent chain of each edge it names. An edge two pairs pick is blended once and carries both
/// pairs' names.
fn pick(body: &Brep, pairs: &[EdgeName], strict: bool) -> Result<Vec<Picked>> {
    let mut uses: HashMap<EdgeId, Vec<CoedgeId>> = HashMap::new();
    for c in body.coedge_ids() {
        uses.entry(body.coedge(c).edge).or_default().push(c);
    }
    let two_faced = |e: EdgeId| -> Option<([FaceId; 2], [CoedgeId; 2])> {
        let cs = uses.get(&e).filter(|cs| cs.len() == 2)?;
        let fs = [body.coedge_face(cs[0]), body.coedge_face(cs[1])];
        (fs[0] != fs[1]).then_some((fs, [cs[0], cs[1]]))
    };
    let mut out: Vec<Picked> = Vec::new();
    let mut index: HashMap<EdgeId, usize> = HashMap::new();
    let carriers = |name: &str| -> BTreeSet<FaceId> {
        body.face_ids()
            .filter(|&f| has_name(body, f, name))
            .collect()
    };
    for (i, pair) in pairs.iter().enumerate() {
        let (a, b) = (pair.a, pair.b);
        let (on_a, on_b) = if pair.exact {
            (carriers(a), carriers(b))
        } else {
            (faces_named(body, a), faces_named(body, b))
        };
        let mut found = false;
        for e in body.edge_ids() {
            let Some((fs, cs)) = two_faced(e) else {
                continue;
            };
            let order = if on_a.contains(&fs[0]) && on_b.contains(&fs[1]) {
                [0, 1]
            } else if on_a.contains(&fs[1]) && on_b.contains(&fs[0]) {
                [1, 0]
            } else {
                continue;
            };
            found = true;
            if let Some(&k) = index.get(&e) {
                out[k].pairs.insert(i);
                continue;
            }
            index.insert(e, out.len());
            out.push(Picked {
                edge: e,
                pairs: BTreeSet::from([i]),
                faces: order.map(|k| fs[k]),
                coedges: order.map(|k| cs[k]),
            });
        }
        let at_contact = || {
            body.radial_edges().any(|e| {
                let fs: Vec<FaceId> = body
                    .edge_uses(e)
                    .iter()
                    .map(|&c| body.coedge_face(c))
                    .collect();
                fs.iter().any(|f| on_a.contains(f)) && fs.iter().any(|f| on_b.contains(f))
            })
        };
        if !found && strict && at_contact() {
            return Err(Error::Unsupported(format!(
                "blending where faces {a} and {b} meet, along which the solid touches itself"
            )));
        }
        // Picked again after some of a feature's edges are blended, a pair may have none left.
        if !found && strict {
            return invalid(format!("no edge where a face {a} meets a face {b}"));
        }
    }
    // A blend's end there would cut the contact away from one of the pieces that touch.
    let (_, contact) = crate::check::contacts(body);
    if let Some(v) = out
        .iter()
        .flat_map(|pk| {
            let e = body.edge(pk.edge);
            [e.start, e.end]
        })
        .find(|v| contact.contains(v))
    {
        let p = body.vertex(v).point;
        return Err(Error::Unsupported(format!(
            "blending an edge up to ({:.6}, {:.6}, {:.6}), where the solid touches itself",
            p.x, p.y, p.z
        )));
    }

    let mut at_vertex: HashMap<VertexId, Vec<EdgeId>> = HashMap::new();
    for e in body.edge_ids() {
        let edge = body.edge(e);
        if edge.curve.is_none() || edge.start == edge.end {
            continue;
        }
        at_vertex.entry(edge.start).or_default().push(e);
        at_vertex.entry(edge.end).or_default().push(e);
    }
    let mut k = 0;
    while k < out.len() {
        let pk = out[k].clone();
        k += 1;
        let edge = body.edge(pk.edge);
        if edge.start == edge.end {
            continue;
        }
        for v in [edge.start, edge.end] {
            let Some(d) = leaving(body, pk.edge, v) else {
                continue;
            };
            let p = body.vertex(v).point;
            for &e in at_vertex.get(&v).into_iter().flatten() {
                if index.contains_key(&e) {
                    continue;
                }
                let Some((fs, cs)) = two_faced(e) else {
                    continue;
                };
                let Some(d2) = leaving(body, e, v) else {
                    continue;
                };
                if !tangent(d, -d2) {
                    continue;
                }
                // The face it shares keeps its place; the other must meet the picked edge's other
                // face tangentially, so the two sections are one.
                let Some((s, o)) = (0..2)
                    .flat_map(|s| (0..2).map(move |o| (s, o)))
                    .find(|&(s, o)| fs[o] == pk.faces[s] && fs[1 - o] != pk.faces[1 - s])
                else {
                    continue;
                };
                let (here, there) = (pk.faces[1 - s], fs[1 - o]);
                if !tangent(outward(body, here, p), outward(body, there, p)) {
                    continue;
                }
                let order = if s == 0 { [o, 1 - o] } else { [1 - o, o] };
                index.insert(e, out.len());
                out.push(Picked {
                    edge: e,
                    pairs: pk.pairs.clone(),
                    faces: order.map(|i| fs[i]),
                    coedges: order.map(|i| cs[i]),
                });
            }
        }
    }
    Ok(out)
}

/// Whether two unit directions agree within the tangency tolerance.
fn tangent(a: Vec3, b: Vec3) -> bool {
    a.dot(b) > 0.0 && a.cross(b).norm() <= tol::TANGENT
}

/// The unit direction in which edge `e` leaves its vertex `v`.
fn leaving(body: &Brep, e: EdgeId, v: VertexId) -> Option<Vec3> {
    let edge = body.edge(e);
    let curve = edge.curve.as_ref()?;
    if v == edge.start {
        curve.d1(edge.range.0).1.unit()
    } else {
        curve.d1(edge.range.1).1.unit().map(|d| -d)
    }
}

/// 0 when `v` is the edge's start, 1 when it is its end.
fn end_index(body: &Brep, e: EdgeId, v: VertexId) -> usize {
    usize::from(body.edge(e).start != v)
}

/// The picked edges at each vertex.
fn picked_at_vertices(body: &Brep, picked: &[Picked]) -> HashMap<VertexId, Vec<usize>> {
    let mut out: HashMap<VertexId, Vec<usize>> = HashMap::new();
    for (i, pk) in picked.iter().enumerate() {
        let e = body.edge(pk.edge);
        if e.start == e.end {
            continue;
        }
        out.entry(e.start).or_default().push(i);
        out.entry(e.end).or_default().push(i);
    }
    out
}

/// A face's outward normal at a point on it.
fn outward(body: &Brep, f: FaceId, p: Point3) -> Vec3 {
    let face = body.face(f);
    let uv = face.surface.project(p);
    let n = face.surface.normal(uv.x, uv.y);
    if face.reversed { -n } else { n }
}

fn parallel(a: Vec3, b: Vec3) -> bool {
    a.cross(b).norm() <= tol::ANGULAR * a.norm() * b.norm()
}

/// Whether `s` is a plane square to `axis`, or a cylinder or cone about the line through `origin`
/// along `axis`: the surfaces whose meridians about that axis are straight.
fn straight_meridian(s: &Surface, origin: Point3, axis: Vec3) -> bool {
    let about =
        |f: &Frame| parallel(f.z, axis) && (f.origin - origin).cross(axis).norm() <= tol::LINEAR;
    match s {
        Surface::Plane(p) => parallel(p.frame.z, axis),
        Surface::Cylinder(c) => about(&c.frame),
        Surface::Cone(c) => about(&c.frame),
        _ => false,
    }
}

/// Where `s`'s meridian about the line through `origin` along `axis` is a circle: a sphere
/// centred on the line, or a torus about it. Its centre's distance from the line, a point on the
/// line level with the centre, and its radius.
fn circle_meridian(s: &Surface, origin: Point3, axis: Vec3) -> Option<(f64, Point3, f64)> {
    let on = |p: Point3| (p - origin).cross(axis).norm() <= tol::LINEAR;
    match s {
        Surface::Sphere(sp) if on(sp.frame.origin) => Some((0.0, sp.frame.origin, sp.radius)),
        Surface::Torus(t) if on(t.frame.origin) && parallel(t.frame.z, axis) => {
            Some((t.major, t.frame.origin, t.minor))
        }
        _ => None,
    }
}

fn section(body: &Brep, pk: &Picked) -> Result<Section> {
    let e = body.edge(pk.edge);
    let curve = e
        .curve
        .as_ref()
        .expect("a picked edge is used by two faces");
    let (path, t) = match curve {
        Curve::Line(l) => {
            let start = curve.eval(e.range.0);
            let frame = Frame::new(start, l.dir, l.dir.any_perpendicular())?;
            let length = e.range.1 - e.range.0;
            (Path::Prism { frame, length }, e.range.0)
        }
        Curve::Circle(c) => {
            let span = e.range.1 - e.range.0;
            let full = e.start == e.end && (span - TAU).abs() <= tol::ANGULAR;
            let (centre, axis) = (c.frame.origin, c.frame.z);
            for f in pk.faces {
                let s = &body.face(f).surface;
                if !straight_meridian(s, centre, axis) && circle_meridian(s, centre, axis).is_none()
                {
                    return Err(Error::Unsupported(
                        "a blend on a circular edge between faces other than planes, \
                         cylinders, cones, spheres and tori about its axis"
                            .into(),
                    ));
                }
            }
            let path = Path::Ring {
                circle: c.frame,
                t0: e.range.0,
                span: if full { TAU } else { span },
                full,
            };
            (path, e.range.0)
        }
        _ => {
            return Err(Error::Unsupported(format!(
                "a blend on a {} edge",
                curve.kind()
            )));
        }
    };
    let (p3, d) = curve.d1(t);
    let d = d
        .unit()
        .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
    let frame = path.frame();
    let to2 = |v: Vec3| Vec2::new(v.dot(frame.x), v.dot(frame.y));
    let rel = p3 - frame.origin;
    let p = Vec2::new(rel.dot(frame.x), rel.dot(frame.y));
    let mut n = [Vec2::ZERO; 2];
    let mut m = [Vec2::ZERO; 2];
    let mut trace = [Trace::Line; 2];
    for k in 0..2 {
        let n3 = outward(body, pk.faces[k], p3);
        let along = if body.coedge(pk.coedges[k]).reversed {
            -d
        } else {
            d
        };
        m[k] = unit2(to2(n3.cross(along))).ok_or_else(|| {
            Error::Numerical("a face with no direction away from its edge".into())
        })?;
        n[k] = unit2(to2(n3)).ok_or_else(|| Error::Numerical("a face with no normal".into()))?;
        if let Path::Ring { circle, .. } = path {
            // A sphere or torus about the axis meets every section in one circle.
            let s = &body.face(pk.faces[k]).surface;
            if let Some((rho, level, radius)) = circle_meridian(s, circle.origin, circle.z) {
                trace[k] = Trace::Circle {
                    centre: Vec2::new(rho, (level - frame.origin).dot(frame.y)),
                    radius,
                };
            }
        }
        if let Path::Prism { .. } = path {
            trace[k] = match &body.face(pk.faces[k]).surface {
                Surface::Plane(_) => Trace::Line,
                Surface::Cylinder(c) if parallel(c.frame.z, d) => {
                    let o = c.frame.origin - frame.origin;
                    Trace::Circle {
                        centre: Vec2::new(o.dot(frame.x), o.dot(frame.y)),
                        radius: c.radius,
                    }
                }
                _ => {
                    return Err(Error::Unsupported(
                        "a blend on a straight edge between faces other than planes and \
                         cylinders along it"
                            .into(),
                    ));
                }
            };
        }
    }
    let bend = m[0].dot(n[1]);
    if bend.abs() <= tol::TANGENT {
        return invalid("a blend on an edge whose faces meet tangentially");
    }
    Ok(Section {
        path,
        p,
        m,
        n,
        trace,
        convex: bend < 0.0,
    })
}

fn line(a: Vec2, b: Vec2, name: &str) -> NamedSegment {
    NamedSegment::new(Segment::Line { a, b }, name)
}

fn unit2(v: Vec2) -> Option<Vec2> {
    let n = v.norm();
    (n > 1e-300).then(|| v * (1.0 / n))
}

fn angle(v: Vec2) -> f64 {
    v.y.atan2(v.x)
}

/// The shorter arc about `centre` from `a` to `b`.
fn arc(centre: Vec2, radius: f64, a: Vec2, b: Vec2, name: &str) -> NamedSegment {
    let start = angle(a - centre);
    let sweep = (angle(b - centre) - start + PI).rem_euclid(TAU) - PI;
    NamedSegment::new(
        Segment::Arc {
            center: centre,
            radius,
            start,
            sweep,
        },
        name,
    )
}

/// The trace of face `k` from `a` to `b`, both on it.
fn along_face(s: &Section, k: usize, a: Vec2, b: Vec2) -> NamedSegment {
    match s.trace[k] {
        Trace::Line => line(a, b, SIDES[k]),
        Trace::Circle { centre, radius } => arc(centre, radius, a, b, SIDES[k]),
    }
}

/// The centre of the circle of `radius` tangent to both faces' traces near the edge, on the side
/// `sign` (−1 inside the material, +1 outside) of each.
fn fillet_centre(s: &Section, radius: f64, sign: f64) -> Result<Vec2> {
    // Each trace offset by `sign·radius` along its outward normal: a line `n·x = c`, or a circle.
    enum Offset {
        Line { n: Vec2, c: f64 },
        Circle { o: Vec2, r: f64 },
    }
    let offset = |k: usize| -> Result<Offset> {
        Ok(match s.trace[k] {
            Trace::Line => Offset::Line {
                n: s.n[k],
                c: s.n[k].dot(s.p) + sign * radius,
            },
            Trace::Circle {
                centre,
                radius: rho,
            } => {
                let away = if s.n[k].dot(s.p - centre) > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let r = rho + sign * radius * away;
                if r <= tol::LINEAR {
                    return Err(Error::InvalidInput(format!(
                        "a fillet of {radius} on a face curved tighter than it"
                    )));
                }
                Offset::Circle { o: centre, r }
            }
        })
    };
    let line_circle = |n: Vec2, c: f64, o: Vec2, r: f64| -> Vec<Vec2> {
        let foot = o + n * (c - n.dot(o));
        let h2 = r * r - (c - n.dot(o)).powi(2);
        if h2 < 0.0 {
            return Vec::new();
        }
        let h = h2.sqrt();
        vec![foot + n.perp() * h, foot - n.perp() * h]
    };
    let cands = match (offset(0)?, offset(1)?) {
        (Offset::Line { n: n0, c: c0 }, Offset::Line { n: n1, c: c1 }) => {
            let det = n0.cross(n1);
            vec![Vec2::new(c0 * n1.y - c1 * n0.y, n0.x * c1 - n1.x * c0) * (1.0 / det)]
        }
        (Offset::Line { n, c }, Offset::Circle { o, r })
        | (Offset::Circle { o, r }, Offset::Line { n, c }) => line_circle(n, c, o, r),
        (Offset::Circle { o: o0, r: r0 }, Offset::Circle { o: o1, r: r1 }) => {
            let d = o1 - o0;
            let dd = d.norm();
            if dd <= tol::LINEAR {
                Vec::new()
            } else {
                let u = d * (1.0 / dd);
                let a = (r0 * r0 - r1 * r1 + dd * dd) / (2.0 * dd);
                line_circle(u, a + u.dot(o0), o0, r0)
            }
        }
    };
    cands
        .into_iter()
        .filter(|c| c.x.is_finite() && c.y.is_finite())
        .min_by(|a, b| a.dist(s.p).total_cmp(&b.dist(s.p)))
        .ok_or_else(|| Error::InvalidInput("a fillet that touches neither face".into()))
}

/// The cross-section of the blend's material.
fn outline(s: &Section, kind: Kind) -> Result<Outline> {
    let p = s.p;
    let (segments, t) = match kind {
        Kind::Chamfer { setback } => {
            if s.trace.iter().any(|t| matches!(t, Trace::Circle { .. })) {
                return Err(Error::Unsupported(
                    "a chamfer on a straight edge of a cylinder".into(),
                ));
            }
            let t = [p + s.m[0] * setback[0], p + s.m[1] * setback[1]];
            (
                vec![
                    line(p, t[0], SIDES[0]),
                    line(t[0], t[1], "blend"),
                    line(t[1], p, SIDES[1]),
                ],
                t,
            )
        }
        Kind::Fillet { radius } => {
            let sign = if s.convex { -1.0 } else { 1.0 };
            let c = fillet_centre(s, radius, sign)?;
            let mut t = [Vec2::ZERO; 2];
            for (k, tk) in t.iter_mut().enumerate() {
                *tk = match s.trace[k] {
                    Trace::Line => c - s.n[k] * (sign * radius),
                    Trace::Circle {
                        centre,
                        radius: rho,
                    } => {
                        centre
                            + unit2(c - centre).ok_or_else(|| {
                                Error::Numerical("a fillet centred on its face's axis".into())
                            })? * rho
                    }
                };
                if (*tk - p).dot(s.m[k]) <= tol::LINEAR {
                    return Err(Error::InvalidInput(
                        "a fillet that does not run onto both faces".into(),
                    ));
                }
            }
            (
                vec![
                    along_face(s, 0, p, t[0]),
                    arc(c, radius, t[0], t[1], "blend"),
                    along_face(s, 1, t[1], p),
                ],
                t,
            )
        }
    };
    let mut segments = segments;
    if let Path::Ring { .. } = s.path {
        // A section that touches the axis at a point turns into a pole; one that reaches past it
        // would sweep through itself.
        for sg in &mut segments {
            match &mut sg.segment {
                Segment::Line { a, b } => {
                    for q in [a, b] {
                        if q.x.abs() <= tol::LINEAR {
                            q.x = 0.0;
                        }
                    }
                }
                Segment::Arc { center, .. } if center.x.abs() <= tol::LINEAR => center.x = 0.0,
                _ => {}
            }
        }
        let nearest = segments
            .iter()
            .map(|sg| match &sg.segment {
                Segment::Line { a, b } => a.x.min(b.x),
                Segment::Arc {
                    center,
                    radius,
                    start,
                    sweep,
                } => {
                    let ends = (center.x + radius * start.cos())
                        .min(center.x + radius * (start + sweep).cos());
                    let k = ((start.min(start + sweep) - PI) / TAU).ceil();
                    if PI + k * TAU <= start.max(start + sweep) {
                        center.x - radius
                    } else {
                        ends
                    }
                }
                _ => f64::INFINITY,
            })
            .fold(f64::INFINITY, f64::min);
        if nearest < -tol::LINEAR {
            return Err(Error::Unsupported(
                "a blend about a circular edge that reaches past its axis".into(),
            ));
        }
    }
    let mut reach = [(Vec2::ZERO, 0.0); 2];
    for k in 0..2 {
        let w = t[k].dist(p);
        reach[k] = (unit2(t[k] - p).unwrap_or(s.m[k]), w);
    }
    let mut radius = 0.0f64;
    for sg in &segments {
        radius = radius.max(sg.segment.start().dist(p));
        if let Segment::Arc {
            center, radius: r, ..
        } = sg.segment
        {
            radius = radius.max(center.dist(p) + r);
        }
    }
    Ok(Outline {
        profile: Profile {
            loops: vec![ProfileLoop { segments }],
        },
        reach,
        radius,
    })
}

/// Where the point `q` lies in the section's plane, and whether it lies strictly between the
/// edge's ends along the path.
fn in_section(s: &Section, q: Point3) -> (Vec2, bool) {
    match s.path {
        Path::Prism { frame, length } => {
            let l = frame.to_local(q);
            (
                Vec2::new(l.x, l.y),
                l.z > tol::LINEAR && l.z < length - tol::LINEAR,
            )
        }
        Path::Ring {
            circle,
            t0,
            span,
            full,
        } => {
            let l = circle.to_local(q);
            let r = l.x.hypot(l.y);
            let inside = full || {
                let a = (l.y.atan2(l.x) - t0).rem_euclid(TAU);
                let eps = tol::LINEAR / r.max(tol::LINEAR);
                a > eps && a < span - eps
            };
            (Vec2::new(r, l.z), inside)
        }
    }
}

/// Where a concave blend's strip runs off its face over a convex edge: the face past that edge
/// bounds the body there, so the material the blend adds is trimmed by its surface, within `reach`
/// of `at`, and the trimmed piece lies on that face, `face`.
#[derive(Clone, Copy, Debug)]
struct Spill {
    face: FaceId,
    clip: Clip,
    at: Point3,
    reach: f64,
}

/// Refuses a blend whose strip on either face would cross another edge of that face: the strip
/// would no longer lie on the face, and the result would not be the blend. Edges at the picked
/// edge's own ends are left to how that end is finished. A concave blend's strip may run over a
/// convex edge onto a plane, cylinder or cone, which then trims what the blend adds.
fn check_width(body: &Brep, pk: &Picked, s: &Section, o: &Outline) -> Result<Vec<Spill>> {
    const SAMPLES: usize = 16;
    let picked = body.edge(pk.edge);
    let ends = [picked.start, picked.end];
    let mut spills = Vec::new();
    for k in 0..2 {
        let f = pk.faces[k];
        let (dir, width) = o.reach[k];
        // Edges past a short edge at an end are the end's to trim, by the faces across them.
        let answered: Vec<EdgeId> = ends
            .iter()
            .flat_map(|&v| run::chain(body, pk, s, o, k, v))
            .collect();
        for &lp in &body.face(f).loops {
            let coedges = &body.lp(lp).coedges;
            for &c in coedges {
                let id = body.coedge(c).edge;
                let seam = coedges
                    .iter()
                    .filter(|&&o| body.coedge(o).edge == id)
                    .count()
                    > 1;
                let edge = body.edge(id);
                // An end trims only near its vertex; a strip running over the edge trims all along.
                let runs_over = !s.convex && run::convex(body, c) == Some(true);
                if id == pk.edge
                    || seam
                    || ends.contains(&edge.start)
                    || ends.contains(&edge.end)
                    || (answered.contains(&id) && !runs_over)
                {
                    continue;
                }
                let (t0, t1) = edge.range;
                let mut crossing: Vec<Point3> = Vec::new();
                for i in 0..=SAMPLES {
                    let q = match &edge.curve {
                        Some(cv) => cv.eval(t0 + (t1 - t0) * i as f64 / SAMPLES as f64),
                        None => body.vertex(edge.start).point,
                    };
                    let (q2, along_inside) = in_section(s, q);
                    let reach = (q2 - s.p).dot(dir);
                    let off = (q2 - s.p).cross(dir).abs();
                    if along_inside
                        && reach > tol::LINEAR
                        && reach < width - tol::LINEAR
                        && off < width
                    {
                        crossing.push(q);
                    }
                }
                if crossing.is_empty() {
                    continue;
                }
                let spilled = if runs_over {
                    run::across(body, c).and_then(|g| spill(body, g, &crossing, width, o))
                } else {
                    None
                };
                match spilled {
                    Some(sp) => spills.push(sp),
                    None => {
                        return Err(Error::InvalidInput(format!(
                            "a blend {width} wide does not fit on face {}",
                            face_label(body, f)
                        )));
                    }
                }
            }
        }
    }
    Ok(spills)
}

/// The trim of a concave piece by the face `g` its strip runs onto at the points `crossing` of
/// the edge between them, keeping the side of `g` the body lies on; none for a surface other than
/// a plane, cylinder or cone.
fn spill(body: &Brep, g: FaceId, crossing: &[Point3], width: f64, o: &Outline) -> Option<Spill> {
    let q = crossing[crossing.len() / 2];
    let n = outward(body, g, q);
    let inside = q - n * (0.1 * width);
    let clip = match &body.face(g).surface {
        Surface::Plane(p) => Clip::Plane {
            origin: p.frame.origin,
            keep: -n,
        },
        Surface::Cylinder(c) => about(c.frame, c.radius, 0.0, inside),
        Surface::Cone(c) => about(c.frame, c.radius, c.half_angle.tan(), inside),
        _ => return None,
    };
    let (mut lo, mut hi) = (q, q);
    for p in crossing {
        lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    Some(Spill {
        face: g,
        clip,
        at: lo + (hi - lo) * 0.5,
        reach: 0.5 * lo.dist(hi) + 2.0 * o.radius,
    })
}

/// Refuses blends of two picked edges that share a face but no vertex when their strips on that
/// face would overlap: the two blends would cut into each other, and neither would be a blend.
fn check_apart(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    outlines: &[Outline],
) -> Result<()> {
    match overlap(body, picked, sections, outlines) {
        Some(o) => Err(Error::Unsupported(format!(
            "blends {} and {} wide overlap on face {}",
            o.widths[0],
            o.widths[1],
            face_label(body, picked[o.edges[0]].faces[o.sides[0]])
        ))),
        None => Ok(()),
    }
}

/// Two picked edges whose blends' strips overlap on the face they share: `edges` by index into
/// the picked ones, `sides` the shared face's index in each one's `faces`, and the strips' widths.
struct Overlap {
    edges: [usize; 2],
    sides: [usize; 2],
    widths: [f64; 2],
}

/// The first pair of picked edges sharing a face but no vertex whose strips on it overlap.
fn overlap(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    outlines: &[Outline],
) -> Option<Overlap> {
    const SAMPLES: usize = 16;
    for (i, a) in picked.iter().enumerate() {
        let ea = body.edge(a.edge);
        for (j, b) in picked.iter().enumerate() {
            let eb = body.edge(b.edge);
            if i == j
                || [eb.start, eb.end]
                    .iter()
                    .any(|v| *v == ea.start || *v == ea.end)
            {
                continue;
            }
            // Only edges running side by side can hold strips between them; edges that meet at
            // an angle, or across a corner the blend swallows, are the corners' to resolve.
            let beside = match (sections[i].path, sections[j].path) {
                (Path::Prism { frame: fa, .. }, Path::Prism { frame: fb, .. }) => {
                    fa.z.cross(fb.z).norm() <= 1e-9
                }
                (Path::Ring { circle: ca, .. }, Path::Ring { circle: cb, .. }) => {
                    ca.z.cross(cb.z).norm() <= 1e-9
                        && (cb.origin - ca.origin).cross(ca.z).norm() <= tol::LINEAR
                }
                _ => false,
            };
            if !beside {
                continue;
            }
            for k in 0..2 {
                let Some(l) = (0..2).find(|&l| b.faces[l] == a.faces[k]) else {
                    continue;
                };
                let (dir, wa) = outlines[i].reach[k];
                let wb = outlines[j].reach[l].1;
                let (t0, t1) = eb.range;
                let Some(cv) = &eb.curve else { continue };
                for n in 0..=SAMPLES {
                    let q = cv.eval(t0 + (t1 - t0) * n as f64 / SAMPLES as f64);
                    let (q2, inside) = in_section(&sections[i], q);
                    let reach = (q2 - sections[i].p).dot(dir);
                    let off = (q2 - sections[i].p).cross(dir).abs();
                    if inside && reach > tol::LINEAR && reach < wa + wb - tol::LINEAR && off < wa {
                        return Some(Overlap {
                            edges: [i, j],
                            sides: [k, l],
                            widths: [wa, wb],
                        });
                    }
                }
            }
        }
    }
    None
}

/// Where two fillets' strips overlap on a planar strip between them, the faces beyond, if the
/// round goes there instead: the strip a plane bounded by the two edges and two others, the faces
/// past the edges planes. A ball too large for the strip rolls on those faces and the strip goes,
/// as if it had never been cut.
fn swallowed_strip(body: &Brep, picked: &[Picked], o: &Overlap) -> Option<(FaceId, FaceId)> {
    let [i, j] = o.edges;
    let [k, l] = o.sides;
    let strip = picked[i].faces[k];
    let (a, b) = (picked[i].faces[1 - k], picked[j].faces[1 - l]);
    let plane = |f: FaceId| matches!(body.face(f).surface, Surface::Plane(_));
    let loops = &body.face(strip).loops;
    let bounded = loops.len() == 1 && body.lp(loops[0]).coedges.len() == 4;
    (a != b && plane(a) && plane(b) && plane(strip) && bounded).then_some((a, b))
}

fn face_label(body: &Brep, f: FaceId) -> String {
    let names: Vec<String> = body.face(f).names.iter().map(|n| n.to_string()).collect();
    if names.is_empty() {
        format!("#{}", f.index())
    } else {
        names.join("/")
    }
}

/// The faces with a coedge at `v`.
fn faces_at(body: &Brep, v: VertexId) -> BTreeSet<FaceId> {
    body.coedge_ids()
        .filter(|&c| body.coedge_start(c) == v)
        .map(|c| body.coedge_face(c))
        .collect()
}

/// A point on the picked edge a little way in from its end `v`, to tell which side of a surface
/// through `v` the piece lies on.
fn inward_point(body: &Brep, e: EdgeId, v: VertexId, step: f64) -> Point3 {
    let edge = body.edge(e);
    let curve = edge.curve.as_ref().expect("a picked edge has a curve");
    let (t0, t1) = edge.range;
    let speed = curve.d1(0.5 * (t0 + t1)).1.norm().max(1e-300);
    let dt = (step / speed).min(0.5 * (t1 - t0));
    curve.eval(if v == edge.start { t0 + dt } else { t1 - dt })
}

/// The end of a piece whose edge meets no other picked edge at `v`: square on the one other face
/// there, or trimmed by that face's plane, cylinder or cone.
fn free_end(body: &Brep, pk: &Picked, o: &Outline, v: VertexId) -> Result<End> {
    let unsupported = || {
        Err(Error::Unsupported(
            "a blend whose edge ends where more than one other face, or a face other than a \
             plane, cylinder or cone, meets it"
                .into(),
        ))
    };
    // A face on the same surface as one of the edge's own (a boss's underside flush with the top
    // it overhangs) continues that face past the vertex: the piece ends on the other.
    let others: Vec<FaceId> = faces_at(body, v)
        .into_iter()
        .filter(|f| !pk.faces.contains(f))
        .filter(|&f| {
            !pk.faces
                .iter()
                .any(|&g| boolean::same_domain(&body.face(f).surface, &body.face(g).surface))
        })
        .collect();
    let [f] = others.as_slice() else {
        return unsupported();
    };
    let at = body.vertex(v).point;
    let along = leaving(body, pk.edge, v)
        .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
    let n = outward(body, *f, at);
    if matches!(body.face(*f).surface, Surface::Plane(_)) && parallel(n, along) {
        return Ok(End::Exact);
    }
    let q = inward_point(body, pk.edge, v, o.radius);
    let clip = match &body.face(*f).surface {
        // The face's own plane, not one through the vertex, which lies on it only to tolerance:
        // the trimmed end must coincide with the face it stops on.
        Surface::Plane(p) => {
            let keep = if (q - at).dot(n) >= 0.0 { n } else { -n };
            Clip::Plane {
                origin: p.frame.origin,
                keep,
            }
        }
        Surface::Cylinder(c) => about(c.frame, c.radius, 0.0, q),
        Surface::Cone(c) => about(c.frame, c.radius, c.half_angle.tan(), q),
        _ => return unsupported(),
    };
    Ok(End::Trim(clip))
}

/// The cylinder or cone about `frame.z`, keeping the side of it `q` lies on.
fn about(frame: Frame, r0: f64, slope: f64, q: Point3) -> Clip {
    let l = frame.to_local(q);
    Clip::About {
        origin: frame.origin,
        axis: frame.z,
        r0,
        slope,
        inside: l.x.hypot(l.y) < r0 + slope * l.z,
    }
}

/// Whether picked edges `i` and `j` continue each other at `v`, each end `ei`, `ej` there: their
/// sections there are one, so their pieces meet face to face.
fn continues(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    outlines: &[Outline],
    (i, ei): (usize, usize),
    (j, ej): (usize, usize),
    v: VertexId,
) -> bool {
    let (Some(di), Some(dj)) = (
        leaving(body, picked[i].edge, v),
        leaving(body, picked[j].edge, v),
    ) else {
        return false;
    };
    if !tangent(di, -dj) {
        return false;
    }
    let cap = |k: usize, end: usize| -> Vec<Point3> {
        outlines[k].profile.loops[0]
            .segments
            .iter()
            .map(|sg| sections[k].path.place(sg.segment.start(), end == 1))
            .collect()
    };
    let (a, b) = (cap(i, ei), cap(j, ej));
    a.len() == b.len()
        && a.iter()
            .all(|p| b.iter().any(|q| p.dist(*q) <= tol::LINEAR))
}

/// Two straight pieces that meet at a corner `v` of a plane both their edges bound, each the
/// mirror image of the other across the plane between their other faces, end on that plane.
fn mitre(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    (i, j): (usize, usize),
    v: VertexId,
) -> Result<Option<[Clip; 2]>> {
    let (a, b) = (&picked[i], &picked[j]);
    if !matches!(sections[i].path, Path::Prism { .. })
        || !matches!(sections[j].path, Path::Prism { .. })
    {
        return Ok(None);
    }
    let shared: Vec<FaceId> = a
        .faces
        .iter()
        .copied()
        .filter(|f| b.faces.contains(f))
        .collect();
    let [common] = shared.as_slice() else {
        return Ok(None);
    };
    let wall = |pk: &Picked| {
        pk.faces
            .into_iter()
            .find(|f| f != common)
            .expect("two faces")
    };
    let (wa, wb) = (wall(a), wall(b));
    let plane = |f: FaceId| matches!(body.face(f).surface, Surface::Plane(_));
    if !(plane(*common) && plane(wa) && plane(wb)) {
        return Ok(None);
    }
    let at = body.vertex(v).point;
    let (na, nb) = (outward(body, wa, at), outward(body, wb, at));
    let Some(mid) = (na - nb).unit() else {
        return Ok(None);
    };
    if outward(body, *common, at).dot(mid).abs() > tol::TANGENT {
        return Ok(None);
    }
    let keep = |pk: &Picked| {
        let q = inward_point(body, pk.edge, v, 1.0);
        let keep = if (q - at).dot(mid) >= 0.0 { mid } else { -mid };
        Clip::Plane { origin: at, keep }
    };
    Ok(Some([keep(a), keep(b)]))
}

/// An end that runs on through the one other face at `v` into the air past it and stops square to
/// its edge, clear of that face: the piece crosses the face rather than lying on it.
fn through(body: &Brep, pk: &Picked, o: &Outline, v: VertexId) -> Result<End> {
    let End::Trim(wall) = free_end(body, pk, o, v)? else {
        return Ok(End::Exact);
    };
    let at = body.vertex(v).point;
    let onward = -leaving(body, pk.edge, v)
        .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
    let past = overrun(o, &wall, at, onward)?;
    Ok(End::Trim(Clip::Plane {
        origin: at + onward * past,
        keep: -onward,
    }))
}

/// Whether convex blends `i` and `j`, sharing one face at a vertex of exactly three faces, each run
/// on past `v` through the other's wall into air, so neither can cut material the other keeps.
fn leaves_into_air(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    (i, j): (usize, usize),
    v: VertexId,
) -> bool {
    if !(sections[i].convex && sections[j].convex) || faces_at(body, v).len() != 3 {
        return false;
    }
    let (a, b) = (&picked[i], &picked[j]);
    let shared = a.faces.iter().filter(|f| b.faces.contains(f)).count();
    let wall =
        |pk: &Picked, other: &Picked| pk.faces.into_iter().find(|f| !other.faces.contains(f));
    let (Some(wa), Some(wb)) = (wall(a, b), wall(b, a)) else {
        return false;
    };
    let at = body.vertex(v).point;
    let onward = |pk: &Picked| leaving(body, pk.edge, v).map(|d| -d);
    let (Some(da), Some(db)) = (onward(a), onward(b)) else {
        return false;
    };
    shared == 1
        && da.dot(outward(body, wb, at)) > tol::TANGENT
        && db.dot(outward(body, wa, at)) > tol::TANGENT
}

/// How far a piece must run past its end for its whole section to clear `clip`, with as much again
/// to spare.
fn overrun(o: &Outline, clip: &Clip, at: Point3, along: Vec3) -> Result<f64> {
    let n = match *clip {
        Clip::Plane { keep, .. } => keep,
        Clip::About {
            origin,
            axis,
            slope,
            ..
        } => {
            let rel = at - origin;
            let radial = (rel - axis * rel.dot(axis))
                .unit()
                .ok_or_else(|| Error::Unsupported("a blend ending on an axis".into()))?;
            (radial - axis * slope).unit_or_zero()
        }
    };
    let c = n.dot(along).abs();
    if c < GLANCING {
        return Err(Error::Unsupported(
            "a blend whose edge meets the face it ends on at a glancing angle".into(),
        ));
    }
    Ok(o.radius * ((1.0 - c * c).max(0.0).sqrt() / c + 1.0))
}

/// The cosine below which an edge meets the face it ends on too nearly along it for the face's
/// tangent plane to say how far the piece must run on.
const GLANCING: f64 = 0.05;

/// A straight piece's run past end `k` onto a cylinder or cone it meets at a glancing angle,
/// solved against the surface itself: the tangent plane there would send the piece on without
/// bound, where the curved surface takes in its whole section within a short run. The run is the
/// middle of the first stretch past the end over which every point of the end section lies on
/// the side `clip` takes away, and with it how far back along the piece any point lies on that
/// side. None where the end is not of that kind, or no such stretch exists.
fn glancing_run(s: &Section, o: &Outline, clip: &Clip, k: usize) -> Option<(f64, f64)> {
    let Path::Prism { frame, .. } = s.path else {
        return None;
    };
    let Clip::About {
        origin,
        axis,
        r0,
        slope,
        inside,
    } = *clip
    else {
        return None;
    };
    let onward = if k == 1 { frame.z } else { -frame.z };
    let at = s.path.place(s.p, k == 1);
    let rel = at - origin;
    let radial = (rel - axis * rel.dot(axis)).unit()?;
    if (radial - axis * slope).unit_or_zero().dot(onward).abs() >= GLANCING {
        return None;
    }
    const ARC_SAMPLES: usize = 16;
    let mut cap: Vec<Vec2> = Vec::new();
    for sg in o.profile.loops.iter().flat_map(|l| &l.segments) {
        match sg.segment {
            Segment::Arc {
                center,
                radius,
                start,
                sweep,
            } => cap.extend((0..ARC_SAMPLES).map(|i| {
                let a = start + sweep * i as f64 / ARC_SAMPLES as f64;
                center + Vec2::new(a.cos(), a.sin()) * radius
            })),
            _ => cap.push(sg.segment.start()),
        }
    }
    // Each cap point p + onward·t lies off the surface by f(t) = A·t² + 2B·t + C, negative inside.
    let quadratics: Vec<[f64; 3]> = cap
        .iter()
        .map(|&q| {
            let p = s.path.place(q, k == 1) - origin;
            let (h0, ah) = (p.dot(axis), onward.dot(axis));
            let (rho, arho) = (p - axis * h0, onward - axis * ah);
            let r = r0 + slope * h0;
            [
                arho.dot(arho) - slope * slope * ah * ah,
                rho.dot(arho) - slope * ah * r,
                rho.dot(rho) - r * r,
            ]
        })
        .collect();
    let gone = |t: f64| {
        quadratics.iter().all(|&[a, b, c]| {
            let f = a * t * t + 2.0 * b * t + c;
            if inside { f > 0.0 } else { f < 0.0 }
        })
    };
    let mut roots: Vec<f64> = quadratics
        .iter()
        .flat_map(|&[a, b, c]| {
            let disc = b * b - a * c;
            if a.abs() <= f64::EPSILON || disc < 0.0 {
                return Vec::new();
            }
            let sq = disc.sqrt();
            vec![(-b - sq) / a, (-b + sq) / a]
        })
        .collect();
    let back = roots.iter().fold(0.0f64, |m, &t| m.max(-t));
    roots.retain(|&t| t > 0.0);
    roots.push(0.0);
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|a, b| (*a - *b).abs() <= tol::LINEAR);
    for w in roots.windows(2) {
        let mid = 0.5 * (w[0] + w[1]);
        if w[1] - w[0] > tol::LINEAR && gone(mid) {
            return Some((mid, back));
        }
    }
    let last = *roots.last()?;
    gone(last + o.radius).then_some((last + o.radius, back))
}

/// Where end `k` of a piece lies on its edge, how far past it the piece's material reaches, and
/// how far back along the piece its trim may take material.
fn end_reach(s: &Section, o: &Outline, end: End, k: usize) -> Result<(Point3, f64, f64)> {
    let at = s.path.place(s.p, k == 1);
    let along = match s.path {
        Path::Prism { frame, .. } => frame.z,
        Path::Ring {
            circle, t0, span, ..
        } => {
            let a = if k == 1 { t0 + span } else { t0 };
            circle.z.cross(circle.radial(a))
        }
    };
    let (ext, back) = match end {
        End::Exact => (0.0, 0.0),
        End::Short(d) => (-d, 0.0),
        End::Trim(clip) => match glancing_run(s, o, &clip, k) {
            Some(run) => run,
            None => {
                let ext = overrun(o, &clip, at, along)?;
                (ext, ext)
            }
        },
    };
    Ok((at, ext, back))
}

/// How each piece ends, as the sewing reads it.
struct Ends<'a> {
    sections: &'a [Section],
    outlines: &'a [Outline],
    ends: &'a [[End; 2]],
    beyonds: &'a [[std::result::Result<run::Beyond, String>; 2]],
    /// The piece ends, two at a time, that cross at a crease.
    creases: &'a [[(usize, usize); 2]],
}

/// The tools to sew, with what each end takes off past its edge; or why they cannot be sewn. A
/// piece whose end runs past a short edge is trimmed further by the faces past it. Pieces that
/// cross at a crease end on each other's wall instead of running on through it, and are joined
/// into one tool, so the curve where their blend surfaces cross is an edge of it; each keeps its
/// own blend and side faces by their marks.
fn sewing_tools(
    body: &Brep,
    picked: &[Picked],
    tools: &[Brep],
    e: &Ends,
    feature: &str,
) -> std::result::Result<(Vec<Brep>, Vec<[run::Beyond; 2]>), String> {
    let mut ends = e.ends.to_vec();
    let mut group: Vec<usize> = (0..tools.len()).collect();
    fn root(group: &mut [usize], mut i: usize) -> usize {
        while group[i] != i {
            group[i] = group[group[i]];
            i = group[i];
        }
        i
    }
    for &[(i, ei), (j, ej)] in e.creases {
        for (k, end) in [(i, ei), (j, ej)] {
            let edge = body.edge(picked[k].edge);
            let v = if end == 0 { edge.start } else { edge.end };
            ends[k][end] =
                free_end(body, &picked[k], &e.outlines[k], v).map_err(|x| x.to_string())?;
        }
        let (ri, rj) = (root(&mut group, i), root(&mut group, j));
        group[ri.max(rj)] = ri.min(rj);
    }
    let mut out = Vec::with_capacity(tools.len());
    let mut taken = Vec::with_capacity(tools.len());
    for (i, tool) in tools.iter().enumerate() {
        let [b0, b1] = &e.beyonds[i];
        let b = [b0.clone()?, b1.clone()?];
        if !e.sections[i].convex && b.iter().any(|bk| !bk.gone.is_empty()) {
            return Err("a concave blend that runs on past a short edge".into());
        }
        if b.iter().any(|bk| bk.overhung) {
            return Err("a blend that runs on past a short edge under an overhang".into());
        }
        let tool = if ends[i] == e.ends[i] {
            tool.clone()
        } else {
            let (s, o) = (&e.sections[i], &e.outlines[i]);
            let fresh = piece(s, o, ends[i], feature).map_err(|x| x.to_string())?;
            trimmed(fresh, s, o, ends[i], &e.beyonds[i], feature).map_err(|x| x.to_string())?
        };
        out.push(sew::marked(tool, i));
        taken.push(b);
    }
    for i in 0..out.len() {
        let r = root(&mut group, i);
        if r == i {
            continue;
        }
        let joined = boolean::union(&out[r], &out[i])
            .map_err(|x| format!("joining the tools of a crease: {x}"))?;
        out[r] = joined;
    }
    for i in 0..out.len() {
        let r = root(&mut group, i);
        if r != i {
            out[i] = out[r].clone();
        }
    }
    Ok((out, taken))
}

/// A piece trimmed by the faces past any short edge its contact runs beyond, at either end.
fn trimmed(
    mut tool: Brep,
    s: &Section,
    o: &Outline,
    ends: [End; 2],
    beyond: &[std::result::Result<run::Beyond, String>; 2],
    feature: &str,
) -> Result<Brep> {
    for (k, bk) in beyond.iter().enumerate() {
        // The width check left the edges past this end to it, so an end that cannot be followed
        // there is refused, not left untrimmed.
        let bk = bk.as_ref().map_err(|why| match why.as_str() {
            run::WHOLE_FACE => Error::InvalidInput(why.clone()),
            _ => Error::Unsupported(why.clone()),
        })?;
        if bk.extra.is_empty() {
            continue;
        }
        let (at, ext, _) = end_reach(s, o, ends[k], k)?;
        // `radius` bounds a section by whole circles; the faces past the end lie close to it, so
        // their trims reach only as far as the section's true extent.
        let tight = extent(s, o);
        let reach = 1.25 * (ext.max(0.0) * tight / o.radius + tight);
        for clip in &bk.extra {
            tool = clipped(&tool, clip, at, reach, feature)?;
        }
    }
    Ok(tool)
}

/// A concave piece trimmed by each face its strip runs onto, its faces on those faces marked as a
/// corner tool's are, so the boolean that adds it is told they lie on the body.
fn spilled(body: &Brep, mut tool: Brep, spills: &[Spill], feature: &str) -> Result<Brep> {
    for sp in spills {
        tool = clipped(&tool, &sp.clip, sp.at, sp.reach, feature)?;
        let Some(ix) = body.face_ids().position(|f| f == sp.face) else {
            continue;
        };
        let wall = &body.face(sp.face).surface;
        for f in tool.face_ids().collect::<Vec<_>>() {
            if boolean::same_domain(&tool.face(f).surface, wall) {
                tool.face_mut(f).names =
                    BTreeSet::from([FaceName::new(SIDE_MARK, format!("r_{ix}"))]);
            }
        }
    }
    Ok(tool)
}

/// The largest distance from the edge of any point of a section's outline.
fn extent(s: &Section, o: &Outline) -> f64 {
    let p = s.p;
    let mut far = 0.0f64;
    for sg in o.profile.loops.iter().flat_map(|l| &l.segments) {
        far = far
            .max(sg.segment.start().dist(p))
            .max(sg.segment.end().dist(p));
        if let Segment::Arc {
            center,
            radius,
            start,
            sweep,
        } = sg.segment
        {
            let into = (angle(center - p) - start).rem_euclid(TAU);
            let swept = if sweep >= 0.0 {
                into <= sweep
            } else {
                into - TAU >= sweep
            };
            if swept {
                far = far.max(center.dist(p) + radius);
            }
        }
    }
    far
}

/// The solid of one picked edge's blend material, finished at both ends.
fn piece(s: &Section, o: &Outline, ends: [End; 2], feature: &str) -> Result<Brep> {
    let mut ext = [0.0; 2];
    let mut ats = [Vec3::ZERO; 2];
    let mut back = [0.0; 2];
    for k in 0..2 {
        (ats[k], ext[k], back[k]) = end_reach(s, o, ends[k], k)?;
    }
    let mut tool = match s.path {
        Path::Prism { frame, length } => {
            let len = length + ext[0] + ext[1];
            if len <= tol::LINEAR {
                return Err(Error::Unsupported(
                    "a blend whose corners meet along the edge".into(),
                ));
            }
            sweep::extrude(
                &o.profile,
                &frame.translated(frame.z * -ext[0]),
                len,
                feature,
            )?
        }
        Path::Ring {
            circle,
            t0,
            span,
            full,
        } => {
            let inner = o
                .profile
                .loops
                .iter()
                .flat_map(|l| &l.segments)
                .map(|sg| match sg.segment {
                    Segment::Line { a, b } => a.x.min(b.x),
                    Segment::Arc { center, radius, .. } => center.x - radius,
                    _ => f64::INFINITY,
                })
                .fold(f64::INFINITY, f64::min);
            let meridians = [0, 1].map(|k| match ends[k] {
                End::Trim(Clip::Plane { origin, keep })
                    if keep.dot(circle.z).abs() <= tol::ANGULAR
                        && (origin - circle.origin).dot(keep).abs() <= tol::LINEAR =>
                {
                    // The half plane nearest this end, and how far past the end it lies.
                    let d = circle.z.cross(keep);
                    let at = d.dot(circle.y).atan2(d.dot(circle.x));
                    let end = if k == 0 { t0 } else { t0 + span };
                    let wrap = |x: f64| (x + PI).rem_euclid(TAU) - PI;
                    let past = [at, at + PI]
                        .map(|a| wrap(a - end))
                        .into_iter()
                        .min_by(|x, y| x.abs().total_cmp(&y.abs()))
                        .expect("two half planes");
                    Some(if k == 0 { -past } else { past })
                }
                _ => None,
            });
            let (a0, total) = if full {
                (t0, TAU)
            } else {
                if inner <= tol::LINEAR && ext.iter().any(|&e| e > 0.0) {
                    return Err(Error::Unsupported(
                        "a blend reaching its circular edge's axis that ends on another face"
                            .into(),
                    ));
                }
                let mut turn = ext.map(|e| if e > 0.0 { e / inner } else { 0.0 });
                // An end on a plane through the axis stops exactly at that half plane.
                for (k, m) in meridians.iter().enumerate() {
                    if let Some(m) = m {
                        turn[k] = *m;
                    }
                }
                (t0 - turn[0], span + turn[0] + turn[1])
            };
            if !full && total >= TAU {
                return Err(Error::Unsupported(
                    "a blend on most of a circle that ends on other faces".into(),
                ));
            }
            let f = ring_frame(&circle, a0);
            sweep::revolve(&o.profile, &f, circle.origin, circle.z, total, feature)?
        }
    };
    for k in 0..2 {
        if let End::Trim(clip) = ends[k] {
            if let (Path::Ring { circle, .. }, Clip::Plane { origin, keep }) = (s.path, clip)
                && keep.dot(circle.z).abs() <= tol::ANGULAR
                && (origin - circle.origin).dot(keep).abs() <= tol::LINEAR
            {
                continue;
            }
            // Material past this end lies within its overrun and the section's reach of the
            // vertex. The other end, or a ring's far side, may lie on the wrong side of this end's
            // surface, so only that neighbourhood is trimmed.
            let reach = 1.25 * (ext[k].max(back[k]) + o.radius);
            tool = clipped(&tool, &clip, ats[k], reach, feature)?;
        }
    }
    Ok(tool)
}

/// `tool` less what lies on the wrong side of `clip` within `reach` of `at`.
fn clipped(tool: &Brep, clip: &Clip, at: Point3, reach: f64, feature: &str) -> Result<Brep> {
    let square = Profile::rect(Vec2::ZERO, 2.0 * reach, 2.0 * reach);
    match *clip {
        Clip::Plane { origin, keep } => {
            let o = at - keep * (at - origin).dot(keep);
            let wrong = sweep::extrude(&square, &Frame::from_axis(o, -keep)?, reach, feature)?;
            boolean::difference(tool, &wrong)
        }
        Clip::About {
            origin,
            axis,
            r0,
            slope,
            inside,
        } => {
            let near = sweep::extrude(
                &square,
                &Frame::from_axis(at - axis * reach, axis)?,
                2.0 * reach,
                feature,
            )?;
            let h = (at - origin).dot(axis);
            // The cone runs past the box's ends, by less where its apex lies just past them.
            let span = |k: f64| {
                let (h0, h1) = (h - k * reach, h + k * reach);
                (h0, h1, r0 + slope * h0, r0 + slope * h1)
            };
            let (mut h0, mut h1, mut ra, mut rb) = span(2.0);
            if ra <= tol::LINEAR || rb <= tol::LINEAR {
                (h0, h1, ra, rb) = span(1.25);
            }
            if ra <= tol::LINEAR || rb <= tol::LINEAR {
                return Err(Error::Unsupported(
                    "a blend ending on a cone near its apex".into(),
                ));
            }
            let x = axis.any_perpendicular();
            let frame = Frame::new(origin, x.cross(axis), x)?;
            let pts = [
                Vec2::new(0.0, h0),
                Vec2::new(ra, h0),
                Vec2::new(rb, h1),
                Vec2::new(0.0, h1),
            ];
            let solid = sweep::revolve(
                &Profile::polygon(&pts, &["a", "b", "c", "d"]),
                &frame,
                origin,
                axis,
                TAU,
                feature,
            )?;
            let wrong = if inside {
                boolean::difference(&near, &solid)?
            } else {
                boolean::intersection(&near, &solid)?
            };
            boolean::difference(tool, &wrong)
        }
    }
}

/// The pieces joined into one tool per run of pieces that meet face to face at `joints`, so each
/// run is one boolean with the body. Pieces whose end sections are one are glued along them, since
/// a boolean would have to intersect blend surfaces that touch there tangentially. A piece that
/// does not glue on, as the last piece of a run that closes on itself at a mitre does (its far
/// end's cap would lie on the first piece's), starts a tool of its own, cut after the run as a
/// mitred piece is.
fn joined(joints: &[(usize, usize)], pieces: Vec<(usize, Brep)>) -> Result<Vec<Brep>> {
    let mut left: HashMap<usize, Brep> = pieces.into_iter().collect();
    let mut order: Vec<usize> = left.keys().copied().collect();
    order.sort();
    let mut tools = Vec::new();
    for seed in order {
        let Some(mut tool) = left.remove(&seed) else {
            continue;
        };
        let mut members = vec![seed];
        let mut k = 0;
        while k < members.len() {
            let m = members[k];
            k += 1;
            for &(a, b) in joints {
                let other = if a == m {
                    b
                } else if b == m {
                    a
                } else {
                    continue;
                };
                let Some(next) = left.get(&other) else {
                    continue;
                };
                if let Some(g) = glue(&tool, next) {
                    tool = g;
                    left.remove(&other);
                    members.push(other);
                }
            }
        }
        tools.push(tool);
    }
    Ok(tools)
}

/// An edge of one cap, the edge of the other it coincides with, and whether they run opposite ways.
type EdgePair = (EdgeId, EdgeId, bool);

/// `a` and `b` joined along every planar face of one that the other covers exactly from the other
/// side, with the same straight and circular edges: those faces go, and their edges are shared.
/// `None` when there is no such face, or the joined solid does not check valid.
pub(crate) fn glue(a: &Brep, b: &Brep) -> Option<Brep> {
    let mut caps: Vec<(FaceId, FaceId, Vec<EdgePair>)> = Vec::new();
    for fa in a.face_ids() {
        for fb in b.face_ids() {
            if let Some(m) = matching_caps(a, fa, b, fb) {
                caps.push((fa, fb, m));
            }
        }
    }
    if caps.is_empty() {
        return None;
    }
    // Only the edges a face keeps go across: where caps of one solid meet each other, their
    // shared edge goes with them.
    let used = |src: &Brep, skip: &BTreeSet<FaceId>| -> BTreeSet<EdgeId> {
        src.face_ids()
            .filter(|f| !skip.contains(f))
            .flat_map(|f| src.face(f).loops.clone())
            .flat_map(|lp| src.lp(lp).coedges.clone())
            .map(|c| src.coedge(c).edge)
            .collect()
    };
    let used_a = used(a, &caps.iter().map(|c| c.0).collect());
    let used_b = used(b, &caps.iter().map(|c| c.1).collect());
    let mut a_of_b: HashMap<EdgeId, (EdgeId, bool)> = HashMap::new();
    for (_, _, m) in &caps {
        for &(e_a, e_b, flip) in m {
            if used_b.contains(&e_b) {
                a_of_b.insert(e_b, (e_a, flip));
            }
        }
    }
    let keep_a: BTreeSet<EdgeId> = used_a
        .iter()
        .copied()
        .chain(a_of_b.values().map(|m| m.0))
        .collect();
    let mut out = Brep::new();
    let mut va: Vec<Option<VertexId>> = vec![None; a.vertex_ids().count()];
    let mut vertex_a = |out: &mut Brep, v: VertexId| -> VertexId {
        *va[v.index()].get_or_insert_with(|| out.add_vertex(a.vertex(v).point))
    };
    let mut ea: Vec<Option<EdgeId>> = vec![None; a.edge_ids().count()];
    for e in a.edge_ids().filter(|e| keep_a.contains(e)) {
        let x = a.edge(e);
        let (s0, s1) = (vertex_a(&mut out, x.start), vertex_a(&mut out, x.end));
        ea[e.index()] = Some(match &x.curve {
            Some(c) => out.add_edge(c.clone(), x.range, s0, s1),
            None => out.add_degenerate_edge(s0, x.range),
        });
    }
    let mut eb_to: HashMap<EdgeId, (EdgeId, bool)> = HashMap::new();
    let mut vb_to: HashMap<VertexId, VertexId> = HashMap::new();
    for (&e_b, &(e_a, flip)) in &a_of_b {
        eb_to.insert(e_b, (ea[e_a.index()].expect("kept"), flip));
        let (xa, xb) = (a.edge(e_a), b.edge(e_b));
        let (s, e) = if flip {
            (xa.end, xa.start)
        } else {
            (xa.start, xa.end)
        };
        vb_to.insert(xb.start, vertex_a(&mut out, s));
        vb_to.insert(xb.end, vertex_a(&mut out, e));
    }
    let mut vb: Vec<Option<VertexId>> = vec![None; b.vertex_ids().count()];
    let mut eb: Vec<Option<(EdgeId, bool)>> = Vec::with_capacity(b.edge_ids().count());
    for e in b.edge_ids() {
        if let Some(&m) = eb_to.get(&e) {
            eb.push(Some(m));
            continue;
        }
        if !used_b.contains(&e) {
            eb.push(None);
            continue;
        }
        let x = b.edge(e);
        let mut vertex_b = |v: VertexId| -> VertexId {
            *vb[v.index()].get_or_insert_with(|| match vb_to.get(&v) {
                Some(&w) => w,
                None => out.add_vertex(b.vertex(v).point),
            })
        };
        let (s0, s1) = (vertex_b(x.start), vertex_b(x.end));
        eb.push(Some((
            match &x.curve {
                Some(c) => out.add_edge(c.clone(), x.range, s0, s1),
                None => out.add_degenerate_edge(s0, x.range),
            },
            false,
        )));
    }
    let mut faces = Vec::new();
    for (src, skip, edges) in [
        (a, caps.iter().map(|c| c.0).collect::<Vec<_>>(), None),
        (b, caps.iter().map(|c| c.1).collect::<Vec<_>>(), Some(&eb)),
    ] {
        for f in src.face_ids() {
            if skip.contains(&f) {
                continue;
            }
            let face = src.face(f);
            let nf = out.add_face(face.surface.clone(), face.reversed, face.names.clone());
            for &lp in &face.loops {
                let mut specs = Vec::new();
                for &c in &src.lp(lp).coedges {
                    let co = src.coedge(c);
                    let spec = match edges {
                        None => CoedgeSpec {
                            edge: ea[co.edge.index()].expect("a kept edge"),
                            reversed: co.reversed,
                            pcurve: co.pcurve.clone(),
                        },
                        Some(map) => {
                            let (edge, flip) = map[co.edge.index()].expect("a kept edge");
                            if eb_to.contains_key(&co.edge) {
                                // Where the coedge starts in the face's parameters, which a
                                // projected pcurve continues from along the shared edge.
                                let seed = match &co.pcurve {
                                    PCurve::Affine { origin, dir } => {
                                        let t = src.edge(co.edge).range;
                                        *origin + *dir * if co.reversed { t.1 } else { t.0 }
                                    }
                                    PCurve::Projected { seed } => *seed,
                                };
                                CoedgeSpec {
                                    edge,
                                    reversed: co.reversed != flip,
                                    pcurve: PCurve::Projected { seed },
                                }
                            } else {
                                CoedgeSpec {
                                    edge,
                                    reversed: co.reversed,
                                    pcurve: co.pcurve.clone(),
                                }
                            }
                        }
                    };
                    specs.push(spec);
                }
                out.add_loop(nf, specs);
            }
            faces.push(nf);
        }
    }
    let shell = out.add_shell(faces);
    out.add_solid(vec![shell]);
    check::check(&out).is_empty().then_some(out)
}

/// When planar faces `fa` of `a` and `fb` of `b` lie on one plane facing each other, with one loop
/// each of the same line and circle edges, those edges paired, each with whether it runs the other
/// way.
fn matching_caps(a: &Brep, fa: FaceId, b: &Brep, fb: FaceId) -> Option<Vec<EdgePair>> {
    let (xa, xb) = (a.face(fa), b.face(fb));
    let (Surface::Plane(pa), Surface::Plane(pb)) = (&xa.surface, &xb.surface) else {
        return None;
    };
    let na = if xa.reversed { -pa.frame.z } else { pa.frame.z };
    let nb = if xb.reversed { -pb.frame.z } else { pb.frame.z };
    if xa.loops.len() != 1
        || xb.loops.len() != 1
        || !boolean::same_domain(&xa.surface, &xb.surface)
        || na.dot(nb) >= 0.0
    {
        return None;
    }
    let edges = |s: &Brep, f: &crate::topo::Face| -> Vec<EdgeId> {
        s.lp(f.loops[0])
            .coedges
            .iter()
            .map(|&c| s.coedge(c).edge)
            .collect()
    };
    let (la, lb) = (edges(a, xa), edges(b, xb));
    if la.len() != lb.len() {
        return None;
    }
    let ends = |s: &Brep, e: EdgeId| -> Option<[Point3; 3]> {
        let x = s.edge(e);
        let c = x.curve.as_ref()?;
        if !matches!(c, Curve::Line(_) | Curve::Circle(_)) || x.start == x.end {
            return None;
        }
        Some([
            s.vertex(x.start).point,
            c.eval(0.5 * (x.range.0 + x.range.1)),
            s.vertex(x.end).point,
        ])
    };
    let near = |p: Point3, q: Point3| p.dist(q) <= tol::LINEAR;
    let mut out = Vec::new();
    for &ea in &la {
        let pa = ends(a, ea)?;
        let m = lb.iter().find_map(|&eb| {
            let pb = ends(b, eb)?;
            if !near(pa[1], pb[1]) {
                return None;
            }
            if near(pa[0], pb[0]) && near(pa[2], pb[2]) {
                Some((ea, eb, false))
            } else if near(pa[0], pb[2]) && near(pa[2], pb[0]) {
                Some((ea, eb, true))
            } else {
                None
            }
        })?;
        out.push(m);
    }
    Some(out)
}

/// Piece `i`'s faces named `blend` become `names` and its side faces carry [`SIDE_MARK`]; a face
/// already marked as lying on a body face keeps its mark, and every other face loses its names.
fn marked(mut tool: Brep, i: usize, names: &BTreeSet<FaceName>) -> Brep {
    for f in tool.face_ids().collect::<Vec<_>>() {
        let face = tool.face(f);
        let side = SIDES
            .iter()
            .position(|s| face.names.iter().any(|n| n.face == *s));
        let is_blend = face.names.iter().any(|n| n.face == "blend");
        let on_body = face.names.iter().any(|n| n.feature == SIDE_MARK);
        tool.face_mut(f).names = match (is_blend, side) {
            (true, _) => names.clone(),
            (false, Some(k)) => BTreeSet::from([FaceName::new(SIDE_MARK, format!("{i}_{k}"))]),
            (false, None) if on_body => tool.face(f).names.clone(),
            (false, None) => BTreeSet::new(),
        };
    }
    tool
}

/// Faces named `blend` become `names`; every other face of a tool loses its names, since it lies
/// on the body's faces and never reaches the result on its own.
fn renamed(mut tool: Brep, blend: &str, names: &BTreeSet<FaceName>) -> Brep {
    for f in tool.face_ids().collect::<Vec<_>>() {
        let is_blend = tool.face(f).names.iter().any(|n| n.face == blend);
        tool.face_mut(f).names = if is_blend {
            names.clone()
        } else {
            BTreeSet::new()
        };
    }
    tool
}

/// The sphere that closes a corner where three fillets of `radius` meet on three planes: its
/// centre, `radius` from each plane inside the corner, and the three planes' outward normals,
/// right-handed. Each fillet's spine runs `radius` from both its planes, so all three pass through
/// the centre, and each ends on the section through it: a great circle arc from the foot of the
/// centre on one plane to its foot on the other. The patch is the spherical triangle of the three
/// feet, whatever the angles between the planes.
pub(super) struct Ball {
    centre: Point3,
    radius: f64,
    normals: [Vec3; 3],
    /// The corner, and the directions its three filleted edges leave it in.
    corner: Point3,
    edges: [Vec3; 3],
}

impl Ball {
    /// The direction from the centre to the middle of the patch.
    fn middle(&self) -> Vec3 {
        let [e1, e2, e3] = self.normals;
        (e1 + e2 + e3).unit_or_zero()
    }

    /// The sphere with both poles and its seam off the patch: the poles on the great circle square
    /// to the patch's middle, the seam on the far side.
    fn surface(&self) -> Result<crate::geom::Sphere> {
        let m = self.middle();
        let axis = m
            .cross(self.normals[2])
            .unit()
            .ok_or_else(|| Error::Numerical("a corner whose normals are parallel".into()))?;
        Ok(crate::geom::Sphere {
            frame: Frame::new(self.centre, axis, -m)?,
            radius: self.radius,
        })
    }
}

fn corner_ball(
    body: &Brep,
    picked: &[Picked],
    sections: &[Section],
    at: &[usize],
    v: VertexId,
    radius: f64,
) -> Result<Ball> {
    let faces = faces_at(body, v);
    let corner = body.vertex(v).point;
    let unsupported = || {
        Err(Error::Unsupported(
            "fillets meeting at a vertex other than three convex edges of three planes".into(),
        ))
    };
    if at.len() != 3 || faces.len() != 3 {
        return unsupported();
    }
    let faces: Vec<FaceId> = faces.into_iter().collect();
    for &f in &faces {
        if !matches!(body.face(f).surface, Surface::Plane(_)) {
            return unsupported();
        }
    }
    for &i in at {
        if !sections[i].convex || !matches!(sections[i].path, Path::Prism { .. }) {
            return unsupported();
        }
    }
    let n: Vec<Vec3> = faces.iter().map(|&f| outward(body, f, corner)).collect();
    let det = n[0].cross(n[1]).dot(n[2]);
    if det.abs() <= tol::ANGULAR {
        return unsupported();
    }
    let normals = if det > 0.0 {
        [n[0], n[1], n[2]]
    } else {
        [n[1], n[0], n[2]]
    };
    let [e1, e2, e3] = normals;
    // The point `radius` inside each plane: n_k·(c − corner) = −radius for each k.
    let centre = corner - (e2.cross(e3) + e3.cross(e1) + e1.cross(e2)) * (radius / det.abs());
    let mut edges = [Vec3::ZERO; 3];
    for (k, &i) in at.iter().enumerate() {
        edges[k] = leaving(body, picked[i].edge, v)
            .ok_or_else(|| Error::Numerical("an edge with no tangent".into()))?;
    }
    let ball = Ball {
        centre,
        radius,
        normals,
        corner,
        edges,
    };
    // Each foot must lie on the near side of the patch's middle, so the sphere's poles and seam
    // stay off the patch.
    let m = ball.middle();
    if normals.iter().any(|e| e.dot(m) <= tol::TANGENT) {
        return unsupported();
    }
    Ok(ball)
}

/// The parallelepiped at `origin` spanned by `w`, its faces planes facing out.
fn parallelepiped(origin: Point3, w: [Vec3; 3]) -> Result<Brep> {
    let w = if w[0].cross(w[1]).dot(w[2]) > 0.0 {
        w
    } else {
        [w[1], w[0], w[2]]
    };
    let mut b = Brep::new();
    let at = |c: [usize; 3]| origin + w[0] * c[0] as f64 + w[1] * c[1] as f64 + w[2] * c[2] as f64;
    let mut vs = HashMap::new();
    for a in 0..2 {
        for bb in 0..2 {
            for c in 0..2 {
                vs.insert([a, bb, c], b.add_vertex(at([a, bb, c])));
            }
        }
    }
    let mut edges: HashMap<([usize; 3], [usize; 3]), EdgeId> = HashMap::new();
    let mut faces = Vec::new();
    for j in 0..3 {
        let (k, l) = ((j + 1) % 3, (j + 2) % 3);
        for s in 0..2 {
            let corner = |u: usize, v: usize| {
                let mut c = [0; 3];
                c[j] = s;
                c[k] = u;
                c[l] = v;
                c
            };
            let mut ring = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
            let mut out = w[k].cross(w[l]);
            if s == 0 {
                ring.reverse();
                out = -out;
            }
            let plane = crate::geom::Plane {
                frame: Frame::new(at(ring[0]), out, w[k])?,
            };
            let surface = Surface::Plane(plane);
            let f = b.add_face(surface.clone(), false, BTreeSet::new());
            let mut specs = Vec::new();
            for q in 0..4 {
                let (p0, p1) = (ring[q], ring[(q + 1) % 4]);
                let key = if p0 < p1 { (p0, p1) } else { (p1, p0) };
                let edge = match edges.get(&key) {
                    Some(&e) => e,
                    None => {
                        let (line, len) = crate::geom::Line::through(at(key.0), at(key.1))
                            .ok_or_else(|| Error::Numerical("a flat parallelepiped".into()))?;
                        let e = b.add_edge(Curve::Line(line), (0.0, len), vs[&key.0], vs[&key.1]);
                        edges.insert(key, e);
                        e
                    }
                };
                specs.push(CoedgeSpec {
                    edge,
                    reversed: key.0 != p0,
                    pcurve: PCurve::Projected {
                        seed: surface.project(at(p0)),
                    },
                });
            }
            b.add_loop(f, specs);
            faces.push(f);
        }
    }
    let shell = b.add_shell(faces);
    b.add_solid(vec![shell]);
    Ok(b)
}

/// The material the sphere patch takes off a corner where three fillets meet: the parallelepiped
/// at the sphere's centre bounded by the three fillets' end sections, oversized outward, less the
/// ball.
fn corner_tool(ball: &Ball, feature: &str) -> Result<Brep> {
    let (centre, radius, corner, edges) = (ball.centre, ball.radius, ball.corner, ball.edges);
    // The end sections are the planes through the centre square to each edge; the cell between
    // them is spanned by the lines where two of them meet, each running toward the corner.
    let mut w = [Vec3::ZERO; 3];
    for j in 0..3 {
        let d = edges[(j + 1) % 3].cross(edges[(j + 2) % 3]);
        let d = if d.dot(edges[j]) < 0.0 { d } else { -d };
        // How far along `d` the corner's own coordinate reaches, twice over.
        let reach = (corner - centre).dot(edges[j]) / d.dot(edges[j]);
        w[j] = d * (2.0 * reach.abs().max(radius / d.norm()));
    }
    let cell = parallelepiped(centre, w)?;
    let sphere = ball.surface()?;
    let (axis, seam) = (sphere.frame.z, sphere.frame.x);
    let ball_frame = Frame::new(centre, seam.cross(axis), seam)?;
    let half = Profile {
        loops: vec![ProfileLoop {
            segments: vec![
                NamedSegment::new(
                    Segment::Arc {
                        center: Vec2::ZERO,
                        radius,
                        start: -PI / 2.0,
                        sweep: PI,
                    },
                    "blend",
                ),
                line(Vec2::new(0.0, radius), Vec2::new(0.0, -radius), "axis"),
            ],
        }],
    };
    let solid = sweep::revolve(&half, &ball_frame, centre, axis, TAU, feature)?;
    let corner_name = BTreeSet::from([FaceName::new(feature, "corner")]);
    let solid = renamed(solid, "blend", &corner_name);
    boolean::difference(&cell, &solid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean::csi::signed_distance;
    use crate::{check, measure, mesh};

    fn up(z: f64) -> Frame {
        Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
    }

    fn block() -> Brep {
        sweep::extrude(
            &Profile::rect(Vec2::ZERO, 20.0, 20.0),
            &up(0.0),
            10.0,
            "block",
        )
        .unwrap()
    }

    /// A 20 × 20 × 10 block with a ⌀6 hole down its middle, or a ⌀6 boss 4 high on top, its rim
    /// rounded R1, the fillet built round the rolling ball when `rolling`.
    fn rimmed(boss: bool, rolling: bool) -> Brep {
        let circle = Profile::circle(Vec2::ZERO, 6.0);
        let body = if boss {
            let rod = sweep::extrude(&circle, &up(10.0), 4.0, "rod").unwrap();
            boolean::union(&block(), &rod).unwrap()
        } else {
            let rod = sweep::extrude(&circle, &up(0.0), 12.0, "rod").unwrap();
            boolean::difference(&block(), &rod).unwrap()
        };
        force_rolling(rolling);
        let rounded = fillet(&body, &[("rod.side", "block.end")], 1.0, "rim");
        force_rolling(false);
        rounded.unwrap()
    }

    #[test]
    fn the_rolling_ball_round_a_circular_rim_is_the_exact_torus() {
        // On a circle about both faces' axis the exact blend is a torus; the rolling ball's fitted
        // surface must give the same solid to well within its tenth of the tolerance.
        for boss in [false, true] {
            let exact = rimmed(boss, false);
            let rolled = rimmed(boss, true);
            assert!(check::check(&rolled).is_empty());
            let (a, b) = (measure::volume(&exact), measure::volume(&rolled));
            assert!(
                (a - b).abs() < 1e-9 * a,
                "boss {boss}: torus {a}, rolling ball {b}"
            );
            let (a, b) = (measure::area(&exact), measure::area(&rolled));
            assert!(
                (a - b).abs() < 1e-9 * a,
                "boss {boss}: torus area {a}, rolling ball {b}"
            );
            assert_eq!(exact.face_count(), rolled.face_count());
        }
    }

    /// Every edge of each patch is smooth: the patch and the face across it share a normal there.
    fn assert_tangent_all_round(b: &Brep, patches: &[FaceId]) {
        let normal = |f: FaceId, p: Vec3| {
            let s = &b.face(f).surface;
            let uv = s.project(p);
            let n = s.normal(uv.x, uv.y);
            if b.face(f).reversed { -n } else { n }
        };
        for &f in patches {
            for &lp in &b.face(f).loops {
                for &c in &b.lp(lp).coedges {
                    let e = b.coedge(c).edge;
                    let x = b.edge(e);
                    let Some(curve) = &x.curve else { continue };
                    let other = b
                        .coedge_ids()
                        .find(|&o| o != c && b.coedge(o).edge == e)
                        .map(|o| b.coedge_face(o))
                        .expect("every edge has two faces");
                    for i in 1..34 {
                        let t = x.range.0 + (x.range.1 - x.range.0) * i as f64 / 34.0;
                        let p = curve.eval(t);
                        let angle = normal(f, p).cross(normal(other, p)).norm();
                        assert!(angle < 1e-9, "a crease of {angle} rad at {p:?}");
                        assert!(normal(f, p).dot(normal(other, p)) > 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn four_fillets_alternating_round_a_saddle_close_with_a_patch_tangent_to_each() {
        // A 10 × 10 × 5 boss stands on the block's top, overhanging its +x side by 5. At each end
        // of the overhang's underside edge the boss's side and the block's front cross the z = 10
        // plane, each line turning from concave to convex there: two saddles, mirror images.
        // Filleted R1 all round, each fillet stops a radius short of the saddle, square on the
        // cube about it, and inside the cube the convex fillets run on to the patch. Outside the
        // cubes the fillets cancel, 16 of concave edge against 16 of convex, so the part changes
        // by the two corners alone, each holding the filleted corner where the cube held 4.
        let boss = Profile::rect(Vec2::new(10.0, 0.0), 10.0, 10.0);
        let boss = sweep::extrude(&boss, &up(10.0), 5.0, "o").unwrap();
        let body = boolean::union(&block(), &boss).unwrap();
        let edges = [
            ("o.v_max", "o.start"),
            ("o.v_min", "o.start"),
            ("block.u_max", "block.end"),
            ("block.end", "o.v_max"),
            ("block.end", "o.v_min"),
            ("o.start", "block.u_max"),
        ]
        .map(|(a, b)| EdgeName { a, b, exact: true });
        let b = fillet_named(&body, &edges, 1.0, "f").unwrap();
        assert!(check::check(&b).is_empty());
        let fine = mesh::MeshOptions {
            linear: 0.01,
            angular: 0.2,
        };
        let tris = mesh::tessellate(&b, &fine).unwrap();
        let solid = b.solid_ids().next().unwrap();
        assert!(mesh::report(&b, &tris, solid).watertight());
        let corner = measure::volume(&saddle_corner(1.0).unwrap());
        let want = 4500.0 + 2.0 * (corner - 4.0);
        let got = measure::volume(&b);
        assert!((got - want).abs() < 1e-9 * want, "volume {got} for {want}");
        let of = |kind: &str| {
            b.face_ids()
                .filter(|&f| b.face(f).surface.kind() == kind)
                .count()
        };
        assert_eq!((of("plane"), of("cylinder"), of("nurbs")), (12, 7, 2));
        let patches: Vec<_> = b
            .face_ids()
            .filter(|&f| b.face(f).surface.kind() == "nurbs")
            .collect();
        assert_tangent_all_round(&b, &patches);
        for &f in &patches {
            // Each patch carries all four fillets' names, the underside edge's among them.
            assert_eq!(b.face(f).names.len(), 4);
            assert!(b.face(f).names.iter().any(|n| n.face == "round_5"));
        }
    }

    #[test]
    fn a_rounds_side_faces_lie_on_the_faces_its_edge_joins() {
        // A box 40 × 30 × 20 rounded at radius 5 where its top meets its y = 0 side: the tool's
        // two flat sides, 5 by 40 each, lie on the top and on that side.
        let body = sweep::extrude(
            &Profile::rect(Vec2::new(20.0, 15.0), 40.0, 30.0),
            &Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap(),
            20.0,
            "box",
        )
        .unwrap();
        let picked = pick(&body, &by_surface(&[("box.end", "box.v_min")]), true).unwrap();
        let s = section(&body, &picked[0]).unwrap();
        let o = outline(&s, Kind::Fillet { radius: 5.0 }).unwrap();
        let tool = piece(&s, &o, [End::Exact; 2], "f").unwrap();
        let tool = marked(tool, 0, &BTreeSet::new());
        let (tool, facts) = sides_on_body(tool, &body, &picked, &body);
        assert!(tool.face_ids().all(|f| tool.face(f).names.is_empty()));
        let faces: BTreeSet<FaceId> = facts.on_face.iter().map(|f| f.face).collect();
        assert_eq!(faces, BTreeSet::from(picked[0].faces));
        // Each side is a rectangle: four edges on its face.
        assert_eq!(facts.on_face.len(), 8);
        for f in &facts.on_face {
            let edge = tool.edge(f.edge);
            let c = edge.curve.as_ref().unwrap();
            for t in [edge.range.0, edge.range.1] {
                let d = signed_distance(&body.face(f.face).surface, c.eval(t));
                assert!(d.abs() <= tol::LINEAR, "{d}");
            }
        }
    }

    #[test]
    fn the_end_of_a_threaded_core_chamfers_through_the_strip_its_thread_leaves() {
        // A bolt's shank turned to radius 4 for 10, then a core of radius 3.264 to 52, a 60°
        // thread ridge of pitch 1.2 joined on the core for 32 turns: the core's face left between
        // the turns winds round it, and a chamfer of 0.36 at its end cuts that face.
        use crate::sweep::{Helix, helix};
        let seg = |segment, name: &str| NamedSegment::new(segment, name);
        let line = |a: [f64; 2], b: [f64; 2], name: &str| {
            seg(
                Segment::Line {
                    a: Vec2::new(a[0], a[1]),
                    b: Vec2::new(b[0], b[1]),
                },
                name,
            )
        };
        let (major, pitch): (f64, f64) = (4.0, 1.2);
        let minor = major - 0.6134 * pitch;
        let shank = Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    line([0.0, 0.0], [major, 0.0], "bottom"),
                    line([major, 0.0], [major, 10.0], "plain"),
                    line([major, 10.0], [minor, 10.0], "step"),
                    line([minor, 10.0], [minor, 52.0], "core"),
                    line([minor, 52.0], [0.0, 52.0], "top"),
                    line([0.0, 52.0], [0.0, 0.0], "axis"),
                ],
            }],
        };
        let meridian = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
        let shank = sweep::revolve(&shank, &meridian, Vec3::ZERO, Vec3::Z, TAU, "shank").unwrap();
        let root = minor - 0.1 * pitch;
        let half = |x: f64| (major - x) * (PI / 6.0).tan();
        let span = 2.0 * half(root);
        let z0 = 10.0 + 0.5 * pitch + span / 2.0;
        let turns = ((52.0 - 10.0 - span - 1.5 * pitch) / pitch).floor();
        let ridge = Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    line([root, z0 - half(root)], [major, z0], "a"),
                    line([major, z0], [root, z0 + half(root)], "b"),
                    line([root, z0 + half(root)], [root, z0 - half(root)], "c"),
                ],
            }],
        };
        let motion = Helix {
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            rise: pitch / TAU,
            sign: 1.0,
        };
        let ridge = helix(&ridge, &meridian, &motion, turns * TAU, "hel").unwrap();
        let bolt = crate::boolean::union(&shank, &ridge).unwrap();
        let out = chamfer(&bolt, &[("shank.top", "shank.core")], 0.36, "cha").unwrap();
        assert!(check::check(&out).is_empty());
        let cut = 0.36 * 0.36 / 2.0 * TAU * (minor - 0.12);
        let got = crate::measure::volume(&out);
        let want = crate::measure::volume(&bolt) - cut;
        assert!((got - want).abs() < 1e-6 * want, "{got} against {want}");
    }

    #[test]
    fn a_round_in_the_inside_corner_of_a_union_fills_it() {
        // A plate 4 thick standing on a base 4 thick, both 20 deep: their union is an L, and a
        // round of 3 where the plate's side meets the base's top adds (1 − π/4)·9·20.
        let block = |o: Vec3, w: f64, h: f64, name: &str| {
            sweep::extrude(
                &Profile::rect(Vec2::new(w / 2.0, 10.0), w, 20.0),
                &Frame::new(o, Vec3::Z, Vec3::X).unwrap(),
                h,
                name,
            )
            .unwrap()
        };
        let base = block(Vec3::ZERO, 30.0, 4.0, "base");
        let plate = block(Vec3::new(0.0, 0.0, 4.0), 4.0, 30.0, "plate");
        let l = crate::boolean::union(&base, &plate).unwrap();
        let out = fillet(&l, &[("base.end", "plate.u_max")], 3.0, "fil").unwrap();
        assert!(check::check(&out).is_empty());
        let want = 30.0 * 20.0 * 4.0 + 4.0 * 20.0 * 30.0 + (1.0 - PI / 4.0) * 9.0 * 20.0;
        let got = crate::measure::volume(&out);
        assert!((got - want).abs() < 1e-6, "{got} against {want}");
    }

    #[test]
    fn rounds_of_both_top_edges_must_fit_the_top_between_them() {
        // A box 40 × 30 × 20: its top's two long edges are 30 apart, so rounds of 14.9 leave a
        // strip of top between them, and rounds of 15.1 would cut into each other.
        let body = sweep::extrude(
            &Profile::rect(Vec2::new(20.0, 15.0), 40.0, 30.0),
            &Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X).unwrap(),
            20.0,
            "box",
        )
        .unwrap();
        let edges = [("box.end", "box.v_min"), ("box.end", "box.v_max")];
        let out = fillet(&body, &edges, 14.9, "fil").unwrap();
        assert!(check::check(&out).is_empty());
        assert!(fillet(&body, &edges, 15.1, "fil").is_err());
        // Rounds of 14.9999 leave a strip 0.0002 wide: where each meets it, the boundary is smooth
        // and faces up, though the other round's cylinder passes within a hair of the point.
        let out = fillet(&body, &edges, 14.9999, "fil").unwrap();
        let joint = Point3::new(20.0, 14.9999, 20.0);
        let n = crate::measure::normal_at(&out, joint, 10.0 * tol::LINEAR).unwrap();
        assert!((n - Vec3::Z).norm() < 1e-9, "{n:?}");
    }

    #[test]
    fn a_round_against_a_torus_about_its_edge_s_axis_is_an_exact_torus() {
        // A torus section R20 r5, its tube from 0° to 90°, turned whole, and its outer rim (where
        // the tube meets the flat underside square) rounded R1. Every half plane through the axis
        // cuts the same corner, between the line z = 0 and the tube's circle about (20, 0), so the
        // round is the torus swept by the R1 circle tangent to both: its centre is 1 above the line
        // and 5 − 1 from the tube's centre, at ρ = 20 + √15.
        let seg = |segment, name: &str| NamedSegment::new(segment, name);
        let profile = Profile {
            loops: vec![ProfileLoop {
                segments: vec![
                    seg(
                        Segment::Line {
                            a: Vec2::new(20.0, 0.0),
                            b: Vec2::new(25.0, 0.0),
                        },
                        "bottom",
                    ),
                    seg(
                        Segment::Arc {
                            center: Vec2::new(20.0, 0.0),
                            radius: 5.0,
                            start: 0.0,
                            sweep: std::f64::consts::FRAC_PI_2,
                        },
                        "tube",
                    ),
                    seg(
                        Segment::Line {
                            a: Vec2::new(20.0, 5.0),
                            b: Vec2::new(20.0, 0.0),
                        },
                        "inner",
                    ),
                ],
            }],
        };
        let frame = Frame::new(Vec3::ZERO, -Vec3::Y, Vec3::X).unwrap();
        let body = sweep::revolve(&profile, &frame, Vec3::ZERO, Vec3::Z, TAU, "rv").unwrap();
        let out = fillet(&body, &[("rv.bottom", "rv.tube")], 1.0, "fil").unwrap();
        assert!(check::check(&out).is_empty());
        let rounds: Vec<&Surface> = out
            .face_ids()
            .filter(|&f| has_name(&out, f, "fil.round_0"))
            .map(|f| &out.face(f).surface)
            .collect();
        assert_eq!(rounds.len(), 1);
        let Surface::Torus(t) = rounds[0] else {
            panic!("the round is a {}", rounds[0].kind());
        };
        assert!(
            (t.major - (20.0 + 15f64.sqrt())).abs() < 1e-12,
            "{}",
            t.major
        );
        assert!((t.minor - 1.0).abs() < 1e-12);
        assert!((t.frame.origin.z - 1.0).abs() < 1e-12);
    }
}
