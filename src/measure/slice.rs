//! Where a plane cuts a set of faces, found face by face: each face's surface met by the plane, the
//! curves trimmed to the face and chained across the edges the faces share. It needs no solid, so
//! it slices a sheet as well as a solid's boundary, and it costs no boolean, so many parallel planes
//! cost the faces' intersections with each and little more.

mod critical;

use std::{cell::OnceCell, collections::HashMap, f64::consts::TAU};

use crate::{
    boolean::{
        csi,
        domain::{Domain, Where},
        face_bbox, face_domain, locate_piece,
        ssi::{self, Ssi},
    },
    error::{Error, Result},
    geom::{Curve, Plane, Surface},
    math::{Aabb, Frame, Point3, Vec3},
    measure::SectionEdge,
    tol,
    topo::{Brep, EdgeId, FaceId},
};

/// One curve of a slice: the pieces it runs along, in order, each continuing where the last ends.
/// A closed curve's last piece ends where its first begins.
///
/// A curve the plane cuts across faces runs with the cut region on its left seen from the plane's
/// normal side: where the faces bound a solid, the outer outline of the cut runs counter-clockwise
/// about the normal and each hole clockwise, so the signed area about the normal is the cut's area.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SliceCurve {
    pub edges: Vec<SectionEdge>,
    pub closed: bool,
}

/// Two ends of pieces closer than this, mm, are one point of the slice where no edge they end at
/// carries a wider tolerance of its own: an edge's meeting with the plane is found to about
/// [`tol::LINEAR`], and a fitted intersection curve passes within half of that of both surfaces.
const JOIN: f64 = 10.0 * tol::LINEAR;

/// The faces of an arena made ready to slice by planes square to one normal: each face's trimmed
/// domain and box, and each edge's box, worked out once for every plane.
struct Slicer<'a> {
    b: &'a Brep,
    n: Vec3,
    faces: Vec<Sliced>,
    edges: Vec<(EdgeId, Aabb)>,
}

struct Sliced {
    face: FaceId,
    dom: Domain,
    bb: Aabb,
    /// The lines on the face every closed loop of a cut touching no edge crosses
    /// ([`critical::crossers`]), found when a trace first needs them, then kept for every plane.
    crossers: OnceCell<Vec<critical::Crosser>>,
}

/// A point where the plane meets the edges, and the radius within which the pieces of the slice
/// that end near it end there: the widest tolerance of the edges and vertices met there, and at
/// least [`JOIN`].
#[derive(Clone, Copy, Debug)]
struct Node {
    at: Point3,
    tol: f64,
}

/// The node at `p`, of radius at least `tol`: one already within either radius, else a new one.
fn node_at(nodes: &mut Vec<Node>, p: Point3, tol: f64) -> usize {
    if let Some(k) = nodes.iter().position(|q| q.at.dist(p) <= q.tol.max(tol)) {
        nodes[k].tol = nodes[k].tol.max(tol);
        return k;
    }
    nodes.push(Node { at: p, tol });
    nodes.len() - 1
}

/// A stretch of the slice: its pieces, the points it starts and ends at, and the nodes there.
struct Run {
    edges: Vec<SectionEdge>,
    start: Point3,
    end: Point3,
    from: Option<usize>,
    to: Option<usize>,
    closed: bool,
}

fn reverse(edges: &mut [SectionEdge]) {
    edges.reverse();
    for e in edges {
        e.reversed = !e.reversed;
    }
}

impl Run {
    fn reversed(mut self) -> Run {
        reverse(&mut self.edges);
        std::mem::swap(&mut self.start, &mut self.end);
        std::mem::swap(&mut self.from, &mut self.to);
        self
    }

    /// The point of the run at distance `d` from `p`, its start or with `at_end` its end, along
    /// the run's piece there: that piece's far end if all of it is nearer.
    fn toward(&self, at_end: bool, p: Point3, d: f64) -> Point3 {
        let e = if at_end {
            &self.edges[self.edges.len() - 1]
        } else {
            &self.edges[0]
        };
        let (a, b) = if e.reversed {
            (e.range.1, e.range.0)
        } else {
            e.range
        };
        let (from, to) = if at_end { (b, a) } else { (a, b) };
        let mut gap = |t: f64| e.curve.eval(t).dist(p) - d;
        if gap(to) <= 0.0 {
            return e.curve.eval(to);
        }
        let (mut near, mut far) = (0.5 * (from + to), to);
        for _ in 0..80 {
            if gap(near) <= 0.0 {
                break;
            }
            far = near;
            near = 0.5 * (from + near);
        }
        e.curve.eval(crate::numeric::bisect(&mut gap, near, far))
    }
}

fn edge_of(curve: &Curve, t0: f64, t1: f64) -> SectionEdge {
    SectionEdge {
        curve: curve.clone(),
        range: (t0.min(t1), t0.max(t1)),
        reversed: t1 < t0,
    }
}

/// How far the box `bb` reaches either side of the plane through `o` square to unit `n`.
fn heights(bb: &Aabb, o: Point3, n: Vec3) -> (f64, f64) {
    let c = (bb.min + bb.max) * 0.5;
    let h = (bb.max - bb.min) * 0.5;
    let mid = (c - o).dot(n);
    let r = h.x * n.x.abs() + h.y * n.y.abs() + h.z * n.z.abs();
    (mid - r, mid + r)
}

impl<'a> Slicer<'a> {
    fn new(b: &'a Brep, n: Vec3) -> Slicer<'a> {
        let faces = b
            .face_ids()
            .map(|f| {
                let dom = face_domain(b, f);
                let bb = face_bbox(&dom);
                Sliced {
                    face: f,
                    dom,
                    bb,
                    crossers: OnceCell::new(),
                }
            })
            .collect();
        let edges = b
            .edge_ids()
            .filter_map(|e| {
                let edge = b.edge(e);
                let c = edge.curve.as_ref()?;
                Some((e, c.bbox(edge.range.0, edge.range.1)))
            })
            .collect();
        Slicer { b, n, faces, edges }
    }

    /// The slice by the plane through `o` square to the normal.
    fn at(&self, o: Point3) -> Result<Vec<SliceCurve>> {
        let (b, n) = (self.b, self.n);
        let plane = Surface::Plane(Plane {
            frame: Frame::new(o, n, n.any_perpendicular())?,
        });
        let reaches = |bb: &Aabb, pad: f64| {
            let (lo, hi) = heights(bb, o, n);
            lo <= pad && hi >= -pad
        };
        // Each edge's meetings with the plane, as nodes, and the edges lying in it.
        let mut nodes: Vec<Node> = Vec::new();
        let mut meets: HashMap<EdgeId, Vec<usize>> = HashMap::new();
        let mut lying: Vec<EdgeId> = Vec::new();
        for (e, bb) in &self.edges {
            let te = JOIN.max(b.edge_tolerance(*e));
            if !reaches(bb, te) {
                continue;
            }
            let edge = b.edge(*e);
            let Some(c) = &edge.curve else { continue };
            let hits = csi::curve_surface(c, edge.range, &plane);
            if hits.on_surface {
                lying.push(*e);
                continue;
            }
            let ids: Vec<usize> = hits
                .ts
                .iter()
                .map(|&t| {
                    let p = c.eval(t);
                    // At an end, the vertex's tolerance counts as well.
                    let r = [edge.start, edge.end]
                        .iter()
                        .map(|&v| (b.vertex(v).point, b.vertex_tolerance(v)))
                        .filter(|&(q, tv)| q.dist(p) <= tv.max(te))
                        .fold(te, |r, (_, tv)| r.max(tv));
                    node_at(&mut nodes, p, r)
                })
                .collect();
            if !ids.is_empty() {
                meets.insert(*e, ids);
            }
        }
        for &e in &lying {
            let edge = b.edge(e);
            for v in [edge.start, edge.end] {
                let r = JOIN.max(b.edge_tolerance(e)).max(b.vertex_tolerance(v));
                node_at(&mut nodes, b.vertex(v).point, r);
            }
        }
        let mut runs: Vec<Run> = Vec::new();
        let mut coplanar: Vec<FaceId> = Vec::new();
        for f in &self.faces {
            if !reaches(&f.bb, JOIN) {
                continue;
            }
            let face = b.face(f.face);
            // The nodes on this face's boundary: where its cut curves leave it.
            let mut ends: Vec<usize> = Vec::new();
            for &l in &face.loops {
                for &c in &b.lp(l).coedges {
                    if let Some(ids) = meets.get(&b.coedge(c).edge) {
                        ends.extend(ids);
                    }
                }
            }
            ends.sort_unstable();
            ends.dedup();
            let pad = 1e-3 * f.bb.diagonal() + 1e-4;
            let region = Aabb {
                min: f.bb.min - Vec3::new(pad, pad, pad),
                max: f.bb.max + Vec3::new(pad, pad, pad),
            };
            // A trace starts from the face's boundary meetings, and from the plane's crossings of
            // the lines every closed loop that touches no edge crosses.
            let mut seeds = || {
                let mut s: Vec<Point3> = ends.iter().map(|&k| nodes[k].at).collect();
                let cs = f.crossers.get_or_init(|| critical::crossers(&f.dom, n));
                s.extend(critical::crossings(cs, o, n, &plane));
                s
            };
            let cut = ssi::intersect(&plane, &f.dom.surface, &region, &mut seeds)?;
            let curves = match cut {
                Ssi::Coincident => {
                    coplanar.push(f.face);
                    continue;
                }
                Ssi::Curves { curves, .. } => curves,
            };
            for sc in curves {
                // A curve along which the plane touches the face cuts nothing.
                if sc.tangent {
                    continue;
                }
                let cut = Cut {
                    c: &sc.curve,
                    range: sc.range,
                    closed: sc.closed,
                };
                cut.trim(&ends, &nodes, &f.dom, n, &mut runs);
            }
        }
        let mut out: Vec<SliceCurve> = Vec::new();
        // A face lying in the plane is its own part of the slice: each of its loops, as the face
        // runs it.
        for &f in &coplanar {
            for &l in &b.face(f).loops {
                let edges: Vec<SectionEdge> = b
                    .lp(l)
                    .coedges
                    .iter()
                    .filter_map(|&c| {
                        let co = b.coedge(c);
                        let edge = b.edge(co.edge);
                        Some(SectionEdge {
                            curve: edge.curve.clone()?,
                            range: edge.range,
                            reversed: co.reversed,
                        })
                    })
                    .collect();
                if !edges.is_empty() {
                    out.push(SliceCurve {
                        edges,
                        closed: true,
                    });
                }
            }
        }
        // An edge lying in the plane, of no face lying in it, is where faces either side of the
        // plane meet on it, or a sheet's boundary along it: a stretch of the slice.
        for e in lying {
            let in_coplanar = b
                .edge_uses(e)
                .iter()
                .any(|&c| coplanar.contains(&b.coedge_face(c)));
            if in_coplanar {
                continue;
            }
            let edge = b.edge(e);
            let Some(c) = &edge.curve else { continue };
            let (t0, t1) = edge.range;
            // Turned as the cut curves are, by the outward normal of a face beside it there that
            // stands clear of the plane.
            let (mid, dir) = c.d1(0.5 * (t0 + t1));
            let side = b
                .edge_uses(e)
                .iter()
                .map(|&co| {
                    let face = b.face(b.coedge_face(co));
                    let uv = face.surface.project(mid);
                    let out = face.surface.normal(uv.x, uv.y);
                    let out = if face.reversed { -out } else { out };
                    dir.dot(n.cross(out))
                })
                .max_by(|a, b| a.abs().total_cmp(&b.abs()))
                .unwrap_or(1.0);
            let near = |p: Point3| nodes.iter().position(|q| q.at.dist(p) <= q.tol);
            let (start, end) = (c.eval(t0), c.eval(t1));
            let run = Run {
                edges: vec![edge_of(c, t0, t1)],
                start,
                end,
                from: near(start),
                to: near(end),
                closed: edge.start == edge.end,
            };
            runs.push(if side >= 0.0 { run } else { run.reversed() });
        }
        out.extend(chain(runs, &mut nodes, n));
        Ok(out)
    }
}

/// A curve of the plane's cut of a face's surface, over `range`.
struct Cut<'c> {
    c: &'c Curve,
    range: (f64, f64),
    closed: bool,
}

impl Cut<'_> {
    /// The pieces of the curve inside the face, each split where the curve passes within a node's
    /// radius of one of `ends` and turned to run with the cut region on its left.
    fn trim(&self, ends: &[usize], nodes: &[Node], dom: &Domain, n: Vec3, runs: &mut Vec<Run>) {
        let (c, closed) = (self.c, self.closed);
        let (t0, t1) = self.range;
        let span = t1 - t0;
        let mut cuts: Vec<(f64, usize)> = ends
            .iter()
            .filter_map(|&k| {
                let (d, t) = csi::curve_distance(c, self.range, nodes[k].at);
                (d <= nodes[k].tol).then_some((t, k))
            })
            .collect();
        cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
        cuts.dedup_by(|a, b| (a.0 - b.0).abs() <= 1e-9 * span);
        // The stretches between cuts, as lists of (from, to) parameter ranges with the nodes they
        // start and end at: on a closed curve the last runs on round through the curve's start to
        // the first.
        type Piece = (Vec<(f64, f64)>, Option<usize>, Option<usize>);
        let mut pieces: Vec<Piece> = Vec::new();
        if closed {
            if cuts.is_empty() {
                pieces.push((vec![(t0, t1)], None, None));
            } else {
                for w in cuts.windows(2) {
                    pieces.push((vec![(w[0].0, w[1].0)], Some(w[0].1), Some(w[1].1)));
                }
                let (first, last) = (cuts[0], cuts[cuts.len() - 1]);
                let mut wrap = Vec::new();
                if t1 - last.0 > 1e-12 * span {
                    wrap.push((last.0, t1));
                }
                if first.0 - t0 > 1e-12 * span {
                    wrap.push((t0, first.0));
                }
                if !wrap.is_empty() {
                    pieces.push((wrap, Some(last.1), Some(first.1)));
                }
            }
        } else {
            let mut at: Vec<(f64, Option<usize>)> = vec![(t0, None)];
            at.extend(
                cuts.iter()
                    .filter(|x| x.0 > t0 && x.0 < t1)
                    .map(|&(t, k)| (t, Some(k))),
            );
            at.push((t1, None));
            for w in at.windows(2) {
                if w[1].0 - w[0].0 > 1e-12 * span {
                    pieces.push((vec![(w[0].0, w[1].0)], w[0].1, w[1].1));
                }
            }
        }
        let whole = closed && cuts.is_empty();
        for (piece, from, to) in pieces {
            // The piece's middle, by parameter along all its ranges.
            let len: f64 = piece.iter().map(|r| r.1 - r.0).sum();
            let mut m = 0.5 * len;
            let mut mid_t = piece[0].0;
            for r in &piece {
                if m <= r.1 - r.0 {
                    mid_t = r.0 + m;
                    break;
                }
                m -= r.1 - r.0;
            }
            let start = c.eval(piece[0].0);
            let end = c.eval(piece[piece.len() - 1].1);
            let (mid, dir) = c.d1(mid_t);
            let Where::Inside(uv) = locate_piece(dom, [start, mid, end]) else {
                continue;
            };
            // The cut region lies against the face's outward normal, so it is on the left of
            // n × outward.
            let mut out = dom.surface.normal(uv.x, uv.y);
            if dom.reversed {
                out = -out;
            }
            let forward = dir.dot(n.cross(out)) >= 0.0;
            let run = Run {
                edges: piece.iter().map(|r| edge_of(c, r.0, r.1)).collect(),
                start,
                end,
                from,
                to,
                closed: whole,
            };
            runs.push(if forward { run } else { run.reversed() });
        }
    }
}

/// The runs joined end to end into curves at the nodes they end at.
///
/// Where only two ends meet, they join. Where more meet, as at a vertex where faces pinch, each
/// run arriving is followed by the first run leaving clockwise from it about the normal, seen
/// round the node in the plane: the order the plane meets the faces in round the vertex. A run
/// arrives with the cut on its left, so the cut near the node lies anticlockwise of the run
/// leaving and clockwise of the one arriving, and the tightest turn keeps each wedge of the cut
/// that meets the node its own corner (the face tracing of a planar arrangement; M. de Berg et
/// al., *Computational Geometry*, 3rd ed., 2008, §2.2). Ends left over, as where a sheet's faces
/// are turned against each other, join in pairs.
fn chain(runs: Vec<Run>, nodes: &mut Vec<Node>, n: Vec3) -> Vec<SliceCurve> {
    let mut out = Vec::new();
    let mut open: Vec<Run> = Vec::new();
    for r in runs {
        if r.closed {
            out.push(SliceCurve {
                edges: r.edges,
                closed: true,
            });
        } else {
            open.push(r);
        }
    }
    // Each run's start is end 2k and its end 2k + 1. An end at no edge's meeting, as a sheet's
    // boundary in the plane gives, meets whatever ends lie within JOIN of it.
    let mut at: Vec<usize> = Vec::with_capacity(2 * open.len());
    for r in &open {
        for (node, p) in [(r.from, r.start), (r.to, r.end)] {
            at.push(node.unwrap_or_else(|| node_at(nodes, p, JOIN)));
        }
    }
    let mut here: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (k, &nd) in at.iter().enumerate() {
        here[nd].push(k);
    }
    let mut link: Vec<Option<usize>> = vec![None; at.len()];
    let x = n.any_perpendicular();
    let y = n.cross(x);
    for (nd, ends) in here.iter().enumerate() {
        let (arriving, leaving): (Vec<usize>, Vec<usize>) =
            ends.iter().copied().partition(|&k| k % 2 == 1);
        if ends.len() > 2 && !arriving.is_empty() && !leaving.is_empty() {
            let p = nodes[nd].at;
            // Each run's direction from the node, a short way along it, so that runs leaving
            // tangent to each other are told apart by how they bend.
            let reach = ends
                .iter()
                .map(|&k| {
                    let r = &open[k / 2];
                    let e = if k % 2 == 1 {
                        &r.edges[r.edges.len() - 1]
                    } else {
                        &r.edges[0]
                    };
                    e.curve.bbox(e.range.0, e.range.1).diagonal()
                })
                .fold(f64::INFINITY, f64::min);
            let d = (1e-3 * reach).max(10.0 * nodes[nd].tol).min(0.25 * reach);
            let angle = |k: usize| {
                let v = open[k / 2].toward(k % 2 == 1, p, d) - p;
                v.dot(y).atan2(v.dot(x))
            };
            let leaving: Vec<(usize, f64)> = leaving.iter().map(|&k| (k, angle(k))).collect();
            for &k in &arriving {
                let a = angle(k);
                let next = leaving
                    .iter()
                    .filter(|(m, _)| link[*m].is_none())
                    .map(|&(m, b)| {
                        let turn = (a - b).rem_euclid(TAU);
                        (m, if turn <= 1e-12 { TAU } else { turn })
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((m, _)) = next {
                    link[k] = Some(m);
                    link[m] = Some(k);
                }
            }
        }
        let left: Vec<usize> = ends
            .iter()
            .copied()
            .filter(|&k| link[k].is_none())
            .collect();
        if left.len() == 2 {
            link[left[0]] = Some(left[1]);
            link[left[1]] = Some(left[0]);
        }
    }
    // Each curve from an end joined to nothing, then the closed ones from what is left.
    let mut used = vec![false; open.len()];
    let walk = |first: usize, used: &mut Vec<bool>| {
        let mut edges: Vec<SectionEdge> = Vec::new();
        let (mut forward, mut back) = (0usize, 0usize);
        let mut k = first;
        let mut closed = false;
        loop {
            let r = &open[k / 2];
            used[k / 2] = true;
            let mut piece = r.edges.clone();
            if k % 2 == 1 {
                reverse(&mut piece);
                back += 1;
            } else {
                forward += 1;
            }
            edges.extend(piece);
            match link[k ^ 1] {
                Some(m) if m == first => {
                    closed = true;
                    break;
                }
                Some(m) if !used[m / 2] => k = m,
                _ => break,
            }
        }
        // Run the curve the way most of its runs do.
        if back > forward {
            reverse(&mut edges);
        }
        SliceCurve { edges, closed }
    };
    for k in 0..at.len() {
        if link[k].is_none() && !used[k / 2] {
            out.push(walk(k, &mut used));
        }
    }
    for r in 0..open.len() {
        if !used[r] {
            out.push(walk(2 * r, &mut used));
        }
    }
    out
}

/// Where the plane through `origin` square to `normal` cuts the faces of `b`, solid or sheet, as
/// curves: each piece of each face's surface the plane meets inside the face, chained across the
/// edges the faces share. A solid's boundary gives closed curves; a sheet gives open ones that end
/// on its boundary, and closed ones where the plane misses its boundary.
///
/// Unlike [`super::section_curves`], no boolean is run, so faces in no solid are sliced too, and
/// the cost is each face's intersection with the plane.
///
/// - Where a face's cut has to be traced, it is traced from the face's boundary meetings with the
///   plane and from where the plane crosses the iso-parametric lines through the face's critical
///   points of height along the normal, and its seams: every closed loop of the cut that touches
///   no edge crosses one of those lines, so none is missed.
/// - Pieces join where they end within the tolerance of the edge or vertex they end at, at least
///   ten times [`tol::LINEAR`]. Where more than two meet, as at a vertex where faces pinch, each
///   continues on the next clockwise round the point in the plane.
/// - A face lying in the plane gives each of its loops, as the face runs them, in place of the cut
///   curves of the faces beside it that run along its boundary.
/// - An edge lying in the plane, of no face lying in it, is a piece of the slice: where two faces
///   cross the plane at an edge they share, or a sheet's boundary in the plane.
/// - Where the plane only touches a face, along a curve or at a point, nothing is cut there.
pub fn slice(b: &Brep, origin: Point3, normal: Vec3) -> Result<Vec<SliceCurve>> {
    let n = normal
        .unit()
        .ok_or_else(|| Error::InvalidInput("a plane needs a normal".into()))?;
    Slicer::new(b, n).at(origin)
}

/// [`slice`] by each of the parallel planes through `origin + offset · normal`, the normal taken
/// as a unit vector: the faces' domains, boxes and critical points are worked out once for all of
/// them.
pub fn slices(
    b: &Brep,
    origin: Point3,
    normal: Vec3,
    offsets: &[f64],
) -> Result<Vec<Vec<SliceCurve>>> {
    let n = normal
        .unit()
        .ok_or_else(|| Error::InvalidInput("a plane needs a normal".into()))?;
    let s = Slicer::new(b, n);
    offsets.iter().map(|&d| s.at(origin + n * d)).collect()
}
