//! Faces taken out by closing the loops round them in place, where the gap closes one of the
//! ways below and the region it gains or loses is shown to be the one [`super::general`] finds.
//!
//! The deleted faces fall into pieces joined edge to edge. Round a piece, each neighbouring
//! face's loop is cut where it ran along the piece, and the piece closes by one of:
//!
//! - whole inner loops of planar faces dropped, as a hole's two ends or a boss's foot;
//! - two rings, each a whole loop or the rim of a face with a seam: on one surface the two faces
//!   become one, and on two that meet in a circle each face runs on to the circle;
//! - one chain of two, three or four runs of planar neighbours: two straight edges that face each
//!   other across the piece become one, three planes meet in a new corner, and four either pair
//!   off into such edges or meet in two corners joined by a new edge.
//!
//! The region between the piece and the faces run on is then tested exactly: it must lie strictly
//! on one side of every neighbour's surface ([`sides`]), and on the sides that make it the region
//! the general way adds or cuts ([`Around::as_the_general_way`]); it must stay in the box the
//! general way works in; and no face that keeps its loops may reach it unseen by the checker. The
//! result passes the checker, or the general way runs instead.

use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::{
    boolean::{same_domain, ssi},
    check,
    edit::{
        local::{Rim, Side, Splice, Spliced, Use, Wall, clear_of, sides},
        tweak::meet,
    },
    geom::{Circle, Curve, Line, Surface},
    math::{Aabb, Point3, Vec3},
    measure, tol,
    topo::{Brep, CoedgeId, EdgeId, FaceId, LoopId, VertexId},
};

/// A neighbour's loop cut along a run of a piece's edges, between two coedges that stay.
struct Gap {
    before: CoedgeId,
    after: CoedgeId,
    piece: usize,
}

/// A neighbour's loop that ran along a piece all the way round.
struct Whole {
    face: FaceId,
    lp: LoopId,
    piece: usize,
}

#[derive(Clone)]
enum Ring {
    Whole(FaceId, LoopId),
    Seam(usize),
}

enum Plan {
    /// Whole loops dropped, and the two faces they bounded made one where given.
    Drop(Option<(FaceId, FaceId)>, Option<Rim>),
    /// Two faces of one surface, each cut at its seam, made one: the gaps and the surface's wall.
    Seams([usize; 2], usize),
    /// A chain of gaps closed by planar neighbours.
    Chain {
        gaps: Vec<usize>,
        merges: Vec<(usize, usize)>,
        corners: Vec<(Point3, Vec<usize>)>,
        bridge: Option<[usize; 2]>,
    },
    /// Two rings run on to the circle their surfaces meet in.
    Circle(Circle, [Ring; 2], [usize; 2]),
}

/// What a piece's closing adds or cuts: its box, the rims round it, and its curved faces, each
/// with whether a rim is a meridian of it.
struct Region {
    bbox: Aabb,
    rims: Vec<Rim>,
    round: Vec<(Surface, bool)>,
}

struct Around<'a> {
    s: &'a Brep,
    faces: Vec<FaceId>,
    gone: Vec<bool>,
    piece: Vec<usize>,
    pieces: usize,
    walls: Vec<Wall>,
    wall_of: HashMap<FaceId, usize>,
    gaps: Vec<Gap>,
    wholes: Vec<Whole>,
    bound: Aabb,
}

fn analytic(s: &Surface) -> bool {
    matches!(
        s,
        Surface::Plane(_)
            | Surface::Cylinder(_)
            | Surface::Cone(_)
            | Surface::Sphere(_)
            | Surface::Torus(_)
    )
}

fn outward(s: &Brep, f: FaceId, p: Point3) -> Vec3 {
    let face = s.face(f);
    let uv = face.surface.project(p);
    let n = face.surface.normal(uv.x, uv.y);
    if face.reversed { -n } else { n }
}

fn root(p: &mut [usize], mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

impl<'a> Around<'a> {
    fn new(s: &'a Brep, faces: &[FaceId]) -> Option<Around<'a>> {
        if s.has_tolerances() {
            return None;
        }
        let nf = s.face_ids().count();
        let mut gone = vec![false; nf];
        for &f in faces {
            gone[f.index()] = true;
        }
        if faces.iter().any(|&f| !analytic(&s.face(f).surface)) {
            return None;
        }
        let mut group: Vec<usize> = (0..nf).collect();
        let mut near: Vec<FaceId> = Vec::new();
        for e in s.edge_ids() {
            let uses = s.edge_uses(e);
            let lost = uses
                .iter()
                .filter(|&&c| gone[s.coedge_face(c).index()])
                .count();
            if lost == 0 {
                continue;
            }
            let &[c, d] = uses else { return None };
            let (f, g) = (s.coedge_face(c), s.coedge_face(d));
            match (gone[f.index()], gone[g.index()]) {
                (true, true) => {
                    let (x, y) = (root(&mut group, f.index()), root(&mut group, g.index()));
                    group[x.max(y)] = x.min(y);
                }
                (true, false) => near.push(g),
                _ => near.push(f),
            }
        }
        near.sort();
        near.dedup();
        let mut names: HashMap<usize, usize> = HashMap::new();
        let mut piece = vec![usize::MAX; nf];
        for &f in faces {
            let r = root(&mut group, f.index());
            let n = names.len();
            piece[f.index()] = *names.entry(r).or_insert(n);
        }
        let mut walls: Vec<Wall> = Vec::new();
        let mut wall_of = HashMap::new();
        for &n in &near {
            let face = s.face(n);
            let fine = match &face.surface {
                Surface::Torus(t) => t.major > t.minor,
                Surface::Cone(c) => c.half_angle.tan().abs() >= 1e-9,
                other => analytic(other),
            };
            // The general way refuses these, or runs a cylinder on only between its straight
            // edges: it decides.
            if !fine
                || faces
                    .iter()
                    .any(|&d| same_domain(&s.face(d).surface, &face.surface))
                || sector(s, n, &gone)
            {
                return None;
            }
            let mut at = None;
            for &lp in &face.loops {
                for &c in &s.lp(lp).coedges {
                    let uses = s.edge_uses(s.coedge(c).edge).len();
                    if uses != 2 && !(uses == 1 && s.edge(s.coedge(c).edge).is_degenerate()) {
                        return None;
                    }
                    at = at.or(Some(s.vertex(s.coedge_start(c)).point));
                }
            }
            let at = at?;
            let found = walls.iter().position(|w| {
                same_domain(&w.surface, &face.surface) && {
                    let uv = w.surface.project(at);
                    let n0 = w.surface.normal(uv.x, uv.y);
                    let n0 = if w.reversed { -n0 } else { n0 };
                    n0.dot(outward(s, n, at)) > 0.0
                }
            });
            let k = found.unwrap_or_else(|| {
                walls.push(Wall {
                    surface: face.surface.clone(),
                    reversed: face.reversed,
                });
                walls.len() - 1
            });
            wall_of.insert(n, k);
        }
        let bb = measure::bbox(s);
        let pad = 0.5 * bb.diagonal() + 1.0;
        let pad = Vec3::new(pad, pad, pad);
        let mut around = Around {
            s,
            faces: faces.to_vec(),
            gone,
            piece,
            pieces: names.len(),
            walls,
            wall_of,
            gaps: Vec::new(),
            wholes: Vec::new(),
            bound: Aabb {
                min: bb.min - pad,
                max: bb.max + pad,
            },
        };
        // At most one edge that stays may reach each vertex of a piece's rim: the gap then lies
        // between the two faces that edge parts, or in the one face round the vertex.
        let mut reach: HashMap<VertexId, usize> = HashMap::new();
        for e in s.edge_ids().filter(|&e| !around.cut(e)) {
            let x = s.edge(e);
            *reach.entry(x.start).or_default() += 1;
            if x.end != x.start {
                *reach.entry(x.end).or_default() += 1;
            }
        }
        for e in s.edge_ids().filter(|&e| around.cut(e)) {
            let x = s.edge(e);
            if [x.start, x.end]
                .iter()
                .any(|v| reach.get(v).is_some_and(|&n| n > 1))
            {
                return None;
            }
        }
        for &n in &near {
            for &lp in &s.face(n).loops {
                around.cuts_of(n, lp)?;
            }
        }
        Some(around)
    }

    fn cut(&self, e: EdgeId) -> bool {
        self.s
            .edge_uses(e)
            .iter()
            .any(|&c| self.gone[self.s.coedge_face(c).index()])
    }

    /// The piece across coedge `c`, a neighbour's coedge on a cut edge.
    fn across(&self, c: CoedgeId) -> usize {
        let s = self.s;
        s.edge_uses(s.coedge(c).edge)
            .iter()
            .map(|&o| self.piece[s.coedge_face(o).index()])
            .find(|&p| p != usize::MAX)
            .expect("a cut edge has a deleted face")
    }

    fn cuts_of(&mut self, face: FaceId, lp: LoopId) -> Option<()> {
        let s = self.s;
        let cs = &s.lp(lp).coedges;
        let cut: Vec<bool> = cs.iter().map(|&c| self.cut(s.coedge(c).edge)).collect();
        let n = cs.len();
        if cut.iter().all(|&c| c) {
            let piece = self.across(cs[0]);
            if cs.iter().any(|&c| self.across(c) != piece) {
                return None;
            }
            self.wholes.push(Whole { face, lp, piece });
            return Some(());
        }
        for i in 0..n {
            if cut[i] || !cut[(i + 1) % n] {
                continue;
            }
            let piece = self.across(cs[(i + 1) % n]);
            let mut j = (i + 1) % n;
            while cut[j] {
                if self.across(cs[j]) != piece {
                    return None;
                }
                j = (j + 1) % n;
            }
            self.gaps.push(Gap {
                before: cs[i],
                after: cs[j],
                piece,
            });
        }
        Some(())
    }

    fn wall(&self, c: CoedgeId) -> usize {
        self.wall_of[&self.s.coedge_face(c)]
    }

    fn plane(&self, k: usize) -> Option<(Vec3, f64)> {
        match &self.walls[k].surface {
            Surface::Plane(p) => Some((p.frame.z, p.frame.z.dot(p.frame.origin))),
            _ => None,
        }
    }

    /// The edge a gap's loop leaves the piece by, and the vertex it leaves it at.
    fn spoke(&self, g: usize) -> (EdgeId, VertexId) {
        let c = self.gaps[g].after;
        (self.s.coedge(c).edge, self.s.coedge_start(c))
    }

    /// The ways piece `k` may close; the tests after choose among them.
    fn plans(&self, k: usize) -> Option<Vec<Plan>> {
        let s = self.s;
        let gaps: Vec<usize> = (0..self.gaps.len())
            .filter(|&g| self.gaps[g].piece == k)
            .collect();
        let wholes: Vec<&Whole> = self.wholes.iter().filter(|w| w.piece == k).collect();
        let seam = |g: usize| {
            let gap = &self.gaps[g];
            s.coedge(gap.before).edge == s.coedge(gap.after).edge
        };
        if gaps.is_empty()
            && wholes.iter().all(|w| {
                let face = s.face(w.face);
                matches!(face.surface, Surface::Plane(_)) && face.loops[0] != w.lp
            })
        {
            return Some(vec![Plan::Drop(None, None)]);
        }
        if gaps.iter().all(|&g| seam(g)) {
            let mut rings: Vec<(Ring, usize)> = wholes
                .iter()
                .map(|w| (Ring::Whole(w.face, w.lp), self.wall_of[&w.face]))
                .collect();
            rings.extend(
                gaps.iter()
                    .map(|&g| (Ring::Seam(g), self.wall(self.gaps[g].before))),
            );
            let [(r0, a), (r1, b)] = &rings[..] else {
                return None;
            };
            if a != b {
                return self.circles([r0.clone(), r1.clone()], [*a, *b]);
            }
            return match (r0, r1) {
                (Ring::Whole(f, l0), Ring::Whole(g, l1)) if f != g => {
                    Some(vec![Plan::Drop(Some((*f, *g)), self.between(*l0, *l1, *a))])
                }
                (Ring::Seam(g0), Ring::Seam(g1)) => Some(vec![Plan::Seams([*g0, *g1], *a)]),
                _ => None,
            };
        }
        if !wholes.is_empty() || gaps.iter().any(|&g| seam(g)) {
            return None;
        }
        // The gaps in order round the piece: each one's loop leaves by the edge the next one's
        // arrives by.
        let by_before: HashMap<CoedgeId, usize> =
            gaps.iter().map(|&g| (self.gaps[g].before, g)).collect();
        let mut chain = vec![gaps[0]];
        loop {
            let last = *chain.last().expect("not empty");
            let next = *by_before.get(&s.partner(self.gaps[last].after)?)?;
            if next == chain[0] {
                break;
            }
            if chain.len() > 4 {
                return None;
            }
            chain.push(next);
        }
        let r = chain.len();
        if r != gaps.len() || r < 2 {
            return None;
        }
        let w: Vec<usize> = chain
            .iter()
            .map(|&g| self.wall(self.gaps[g].before))
            .collect();
        let planes: Vec<(Vec3, f64)> = w.iter().map(|&k| self.plane(k)).collect::<Option<_>>()?;
        let chained = |merges: Vec<(usize, usize)>, corners, bridge| Plan::Chain {
            gaps: chain.clone(),
            merges,
            corners,
            bridge,
        };
        let corner = |i: usize, j: usize, k: usize| meet(&[planes[i], planes[j], planes[k]]);
        Some(match r {
            2 if w[0] != w[1] => vec![chained(vec![(0, 1)], Vec::new(), None)],
            3 if w[0] != w[1] && w[1] != w[2] && w[2] != w[0] => {
                vec![chained(
                    Vec::new(),
                    vec![(corner(0, 1, 2)?, vec![0, 1, 2])],
                    None,
                )]
            }
            4 if w[0] == w[2] && w[1] != w[3] => {
                vec![chained(vec![(0, 1), (2, 3)], Vec::new(), None)]
            }
            4 if w[1] == w[3] && w[0] != w[2] => {
                vec![chained(vec![(1, 2), (3, 0)], Vec::new(), None)]
            }
            4 if w[0] != w[2] && w[1] != w[3] => {
                let mut out = Vec::new();
                if let (Some(p), Some(q)) = (corner(0, 2, 1), corner(0, 2, 3)) {
                    let corners = vec![(p, vec![0, 1]), (q, vec![2, 3])];
                    out.push(chained(Vec::new(), corners, Some([w[0], w[2]])));
                }
                if let (Some(p), Some(q)) = (corner(1, 3, 2), corner(1, 3, 0)) {
                    let corners = vec![(p, vec![1, 2]), (q, vec![3, 0])];
                    out.push(chained(Vec::new(), corners, Some([w[1], w[3]])));
                }
                out
            }
            _ => return None,
        })
    }

    /// A ring's one coedge on the piece and its circle, where it is one closed circle.
    fn ring_circle(&self, ring: &Ring) -> Option<(CoedgeId, &Circle)> {
        let s = self.s;
        let c = match ring {
            Ring::Whole(_, lp) => {
                let &[c] = &s.lp(*lp).coedges[..] else {
                    return None;
                };
                c
            }
            Ring::Seam(g) => {
                let gap = &self.gaps[*g];
                let cs = &s.lp(s.coedge(gap.before).lp).coedges;
                let i = cs.iter().position(|&c| c == gap.before)?;
                if cs[(i + 2) % cs.len()] != gap.after {
                    return None;
                }
                cs[(i + 1) % cs.len()]
            }
        };
        let x = s.edge(s.coedge(c).edge);
        match &x.curve {
            Some(Curve::Circle(circle)) if x.start == x.end => Some((c, circle)),
            _ => None,
        }
    }

    /// The straight stretch across the band between two whole loops that are circles about one
    /// axis: what holds the band where no seam crosses it.
    fn between(&self, l0: LoopId, l1: LoopId, wall: usize) -> Option<Rim> {
        let f = self.s.lp(l0).face;
        let (_, c0) = self.ring_circle(&Ring::Whole(f, l0))?;
        let (_, c1) = self.ring_circle(&Ring::Whole(f, l1))?;
        let coaxial = c0.frame.z.cross(c1.frame.z).norm() <= 1e-9
            && (c1.frame.origin - c0.frame.origin).cross(c0.frame.z).norm() <= tol::LINEAR;
        if !coaxial {
            return None;
        }
        let x = c0.frame.x;
        let (line, len) = Line::through(
            c0.frame.origin + x * c0.radius,
            c1.frame.origin + x * c1.radius,
        )?;
        Some(Rim {
            curve: Curve::Line(line),
            range: (0.0, len),
            on: vec![wall],
        })
    }

    fn circles(&self, rings: [Ring; 2], walls: [usize; 2]) -> Option<Vec<Plan>> {
        let (a, b) = (&self.walls[walls[0]].surface, &self.walls[walls[1]].surface);
        let ssi::Ssi::Curves { curves, .. } =
            ssi::intersect(a, b, &self.bound, &mut || Vec::new()).ok()?
        else {
            return None;
        };
        Some(
            curves
                .into_iter()
                .filter(|c| c.closed && !c.tangent)
                .filter_map(|c| match c.curve {
                    Curve::Circle(circle) => Some(Plan::Circle(circle, rings.clone(), walls)),
                    _ => None,
                })
                .collect(),
        )
    }

    /// Lays `plan` into `sp`, and adds the rims and round faces of the region it closes.
    fn apply(
        &self,
        plan: &Plan,
        sp: &mut Splice,
        rims: &mut Vec<Rim>,
        round: &mut Vec<(Surface, bool)>,
        region: &mut Aabb,
    ) -> Option<()> {
        let s = self.s;
        let curve_of = |e: EdgeId| s.edge(e).curve.clone();
        match plan {
            Plan::Drop(join, rim) => {
                if let Some((f, g)) = join {
                    sp.join(*f, *g);
                    let wall = self.wall_of[f];
                    if !matches!(self.walls[wall].surface, Surface::Plane(_)) {
                        rims.push(rim.as_ref().map(|r| Rim {
                            curve: r.curve.clone(),
                            range: r.range,
                            on: r.on.clone(),
                        })?);
                    }
                }
            }
            Plan::Seams([g0, g1], wall) => {
                let (a, b) = (self.spoke(*g0), self.spoke(*g1));
                let (_, gap) = sp.merge(a, b)?;
                sp.merged(self.gaps[*g0].before, self.gaps[*g1].after);
                sp.merged(self.gaps[*g1].before, self.gaps[*g0].after);
                rims.push(Rim {
                    curve: curve_of(a.0)?,
                    range: gap,
                    on: vec![*wall],
                });
            }
            Plan::Chain {
                gaps,
                merges,
                corners,
                bridge,
            } => {
                let r = gaps.len();
                let gap = |i: usize| &self.gaps[gaps[i % r]];
                let walls = |i: usize| vec![self.wall(gap(i).before), self.wall(gap(i + 1).before)];
                for &(i, j) in merges {
                    let (a, b) = (self.spoke(gaps[i]), self.spoke(gaps[j]));
                    let (_, stretch) = sp.merge(a, b)?;
                    sp.merged(gap(i + 1).before, gap(j).after);
                    sp.merged(gap(j + 1).before, gap(i).after);
                    rims.push(Rim {
                        curve: curve_of(a.0)?,
                        range: stretch,
                        on: walls(i),
                    });
                }
                let mut at: Vec<Option<usize>> = vec![None; r];
                let mut ids = Vec::new();
                for (k, (p, spokes)) in corners.iter().enumerate() {
                    let v = sp.add_vertex(*p);
                    ids.push(v);
                    for &i in spokes {
                        let (e, was) = self.spoke(gaps[i]);
                        let stretch = sp.extend(e, was, v)?;
                        at[i] = Some(k);
                        rims.push(Rim {
                            curve: curve_of(e)?,
                            range: stretch,
                            on: walls(i),
                        });
                    }
                }
                let mut edge = None;
                if let Some(on) = bridge {
                    let (line, len) = Line::through(corners[1].0, corners[0].0)?;
                    if len <= 10.0 * tol::LINEAR {
                        return None;
                    }
                    let curve = Curve::Line(line);
                    edge = Some(sp.add_edge(curve.clone(), (0.0, len), ids[1], ids[0]));
                    rims.push(Rim {
                        curve,
                        range: (0.0, len),
                        on: on.to_vec(),
                    });
                }
                if corners.is_empty() {
                    return Some(());
                }
                for i in 0..r {
                    let (from, to) = (at[(i + r - 1) % r]?, at[i]?);
                    let uses = match (from, to, edge) {
                        _ if from == to => Vec::new(),
                        (1, 0, Some(edge)) => vec![Use {
                            edge,
                            reversed: false,
                        }],
                        (0, 1, Some(edge)) => vec![Use {
                            edge,
                            reversed: true,
                        }],
                        _ => return None,
                    };
                    sp.fill(gap(i).before, uses, gap(i).after);
                }
            }
            Plan::Circle(circle, rings, walls) => {
                let (o, z) = (circle.frame.origin, circle.frame.z);
                let mut x: Option<Point3> = None;
                for ring in rings {
                    if let Ring::Seam(g) = ring {
                        let p = s.vertex(self.spoke(*g).1).point - o;
                        let side = (p - z * p.dot(z)).unit()?;
                        let here = o + side * circle.radius;
                        if x.is_some_and(|x| x.dist(here) > tol::LINEAR) {
                            return None;
                        }
                        x = Some(here);
                    }
                }
                let x = x.unwrap_or(o + circle.frame.x * circle.radius);
                let l = circle.frame.to_local(x);
                let t = l.y.atan2(l.x);
                let v = sp.add_vertex(x);
                let curve = Curve::Circle(circle.clone());
                let edge = sp.add_edge(curve.clone(), (t, t + TAU), v, v);
                rims.push(Rim {
                    curve,
                    range: (t, t + TAU),
                    on: walls.to_vec(),
                });
                let mut ways = Vec::new();
                for (ring, &wall) in rings.iter().zip(walls) {
                    let (c, was) = self.ring_circle(ring)?;
                    if was.frame.z.cross(z).norm() > 1e-9
                        || (was.frame.origin - o).cross(z).norm() > tol::LINEAR
                    {
                        return None;
                    }
                    let reversed = (was.frame.z.dot(z) < 0.0) != s.coedge(c).reversed;
                    ways.push(reversed);
                    let surface = &self.walls[wall].surface;
                    let turned = matches!(surface, Surface::Sphere(_) | Surface::Torus(_));
                    let uses = vec![Use { edge, reversed }];
                    match ring {
                        Ring::Whole(face, lp) => {
                            if turned {
                                return None;
                            }
                            sp.new_loop(*face, uses, s.face(*face).loops[0] == *lp);
                            let x = circle.frame.x;
                            let (line, len) = Line::through(
                                was.frame.origin + x * was.radius,
                                o + x * circle.radius,
                            )?;
                            rims.push(Rim {
                                curve: Curve::Line(line),
                                range: (0.0, len),
                                on: vec![wall],
                            });
                        }
                        Ring::Seam(g) => {
                            let (e, at) = self.spoke(*g);
                            let stretch = sp.extend(e, at, v)?;
                            sp.fill(self.gaps[*g].before, uses, self.gaps[*g].after);
                            rims.push(Rim {
                                curve: curve_of(e)?,
                                range: stretch,
                                on: vec![wall],
                            });
                            if turned {
                                round.push((surface.clone(), true));
                                // The face run on is the seam's new stretch turned about the
                                // axis: it lies between the discs at the stretch's heights.
                                let seam = curve_of(e)?;
                                let mut ts = vec![stretch.0, stretch.1];
                                let out = (x - o) - z * (x - o).dot(z);
                                ts.extend(seam.extremes(z, stretch.0, stretch.1));
                                ts.extend(seam.extremes(out, stretch.0, stretch.1));
                                let at: Vec<Vec3> = ts.iter().map(|&t| seam.eval(t) - o).collect();
                                let far = at
                                    .iter()
                                    .map(|p| (*p - z * p.dot(z)).norm())
                                    .fold(0.0, f64::max);
                                let wide = Vec3::new(
                                    far * (1.0 - z.x * z.x).max(0.0).sqrt(),
                                    far * (1.0 - z.y * z.y).max(0.0).sqrt(),
                                    far * (1.0 - z.z * z.z).max(0.0).sqrt(),
                                );
                                for p in &at {
                                    let centre = o + z * p.dot(z);
                                    region.add(centre - wide);
                                    region.add(centre + wide);
                                }
                            }
                        }
                    }
                }
                if ways[0] == ways[1] {
                    return None;
                }
            }
        }
        Some(())
    }

    /// The rims and round faces of piece `k` itself.
    fn piece_rims(
        &self,
        k: usize,
        rims: &mut Vec<Rim>,
        round: &mut Vec<(Surface, bool)>,
        region: &mut Aabb,
    ) -> Option<()> {
        let s = self.s;
        let mut seen: Vec<EdgeId> = Vec::new();
        for &f in self.faces.iter().filter(|f| self.piece[f.index()] == k) {
            let face = s.face(f);
            region.union(&measure::face_bbox(s, f));
            let own = match &face.surface {
                Surface::Plane(_) => None,
                Surface::Cylinder(c) => Some((c.frame.origin, c.frame.z)),
                Surface::Cone(c) => Some((c.frame.origin, c.frame.z)),
                Surface::Sphere(c) => Some((c.frame.origin, c.frame.z)),
                Surface::Torus(c) => Some((c.frame.origin, c.frame.z)),
                _ => return None,
            };
            let (mut straight, mut meridian) = (false, false);
            let mut circles: Vec<&Circle> = Vec::new();
            for &lp in &face.loops {
                for &c in &s.lp(lp).coedges {
                    let e = s.coedge(c).edge;
                    let x = s.edge(e);
                    match (&x.curve, own) {
                        (None, _) => continue,
                        (Some(Curve::Line(_)), _) => straight = true,
                        (Some(Curve::Circle(circle)), Some((o, z))) => {
                            if circle.frame.z.dot(z).abs() <= 1e-9
                                && (circle.frame.origin - o).dot(circle.frame.z).abs()
                                    <= tol::LINEAR
                            {
                                meridian = true;
                            } else if !circles.contains(&circle) {
                                circles.push(circle);
                            }
                        }
                        (Some(Curve::Circle(circle)), None) => {
                            if !circles.contains(&circle) {
                                circles.push(circle);
                            }
                        }
                        _ => return None,
                    }
                    if seen.contains(&e) {
                        continue;
                    }
                    seen.push(e);
                    let on = s
                        .edge_uses(e)
                        .iter()
                        .filter_map(|&o| self.wall_of.get(&s.coedge_face(o)).copied())
                        .collect();
                    rims.push(Rim {
                        curve: x.curve.clone()?,
                        range: x.range,
                        on,
                    });
                }
            }
            if matches!(face.surface, Surface::Sphere(_) | Surface::Torus(_)) {
                round.push((face.surface.clone(), meridian));
                continue;
            }
            round.push((face.surface.clone(), true));
            // A band or disc with no straight edge across it: the stretch across stands in.
            if straight || meridian {
                continue;
            }
            let x = match circles.first() {
                Some(c) => c.frame.x,
                None => continue,
            };
            let ends = match circles[..] {
                [c] if matches!(face.surface, Surface::Plane(_)) => {
                    Some((c.frame.origin + x * c.radius, c.frame.origin))
                }
                [c, d]
                    if c.frame.z.cross(d.frame.z).norm() <= 1e-9
                        && (d.frame.origin - c.frame.origin).cross(c.frame.z).norm()
                            <= tol::LINEAR =>
                {
                    Some((c.frame.origin + x * c.radius, d.frame.origin + x * d.radius))
                }
                _ => None,
            };
            if let Some((p, q)) = ends
                && let Some((line, len)) = Line::through(p, q)
            {
                rims.push(Rim {
                    curve: Curve::Line(line),
                    range: (0.0, len),
                    on: Vec::new(),
                });
            }
        }
        Some(())
    }

    /// `plans`, one for each piece, laid and tested: the solid, or `None` where any test fails.
    fn attempt(&self, plans: &[&Plan]) -> Option<Brep> {
        let mut sp = Splice::new(self.s, &self.faces);
        let mut regions = Vec::new();
        // For each piece, the side of each wall its region lies on.
        let mut lie: Vec<Vec<Side>> = Vec::new();
        for (k, plan) in plans.iter().enumerate() {
            let (mut rims, mut round) = (Vec::new(), Vec::new());
            let mut region = Aabb::EMPTY;
            self.piece_rims(k, &mut rims, &mut round, &mut region)?;
            self.apply(plan, &mut sp, &mut rims, &mut round, &mut region)?;
            let faces: Vec<(&Surface, bool)> = round.iter().map(|(s, m)| (s, *m)).collect();
            lie.push(sides(&self.walls, &rims, &faces)?);
            for rim in &rims {
                region.union(&rim.curve.bbox(rim.range.0, rim.range.1));
            }
            if (0..3)
                .any(|i| region.min[i] < self.bound.min[i] || region.max[i] > self.bound.max[i])
            {
                return None;
            }
            regions.push(Region {
                bbox: region,
                rims,
                round,
            });
        }
        self.as_the_general_way(&lie)?;
        let built = sp.finish()?;
        self.unseen(&built, &regions)?;
        check::check(&built.out).is_empty().then_some(built.out)
    }

    /// `None` unless every piece's region is the one the general way adds or cuts. That way
    /// keeps the material of every wall some piece lies within, all at once, and of any wall some
    /// piece stands out of. A piece that stands out of a wall beside it is material, and is cut
    /// whole when it lies out of every wall of the second kind and in every wall of the first: all
    /// round it is then kept material or the solid's own boundary. A piece within every wall
    /// beside it is a gap, and is filled whole when it lies in every wall of the first kind and,
    /// where there are walls of the second, in one of them.
    fn as_the_general_way(&self, lie: &[Vec<Side>]) -> Option<()> {
        let n = self.walls.len();
        let mut beside = vec![vec![false; n]; self.pieces];
        for (c, k) in self.gaps.iter().map(|g| (g.before, g.piece)).chain(
            self.wholes
                .iter()
                .map(|w| (self.s.lp(w.lp).coedges[0], w.piece)),
        ) {
            beside[k][self.wall(c)] = true;
        }
        let any =
            |side: Side, w: usize| (0..self.pieces).any(|k| beside[k][w] && lie[k][w] == side);
        let within: Vec<bool> = (0..n).map(|w| any(Side::Within, w)).collect();
        let beyond: Vec<bool> = (0..n).map(|w| any(Side::Beyond, w)).collect();
        for k in 0..self.pieces {
            let inside = |w: usize| lie[k][w] == Side::Within;
            if (0..n).any(|w| within[w] && !inside(w)) {
                return None;
            }
            let material = (0..n).any(|w| beside[k][w] && !inside(w));
            let fine = if material {
                (0..n).all(|w| !beyond[w] || !inside(w))
            } else {
                !beyond.contains(&true) || (0..n).any(|w| beyond[w] && inside(w))
            };
            if !fine {
                return None;
            }
        }
        Some(())
    }

    /// `None` where a face that keeps its loops may reach a region in a way the checker does not
    /// search: from another shell, or beside a face run on into the region, whose pair it skips.
    /// A face of a changed face's shell whose box meets a region's is clear of it when the region
    /// lies strictly to one side of the face's surface: a face inside the region joins the changed
    /// faces through faces that cross the region's boundary, which fail this.
    fn unseen(&self, built: &Spliced, regions: &[Region]) -> Option<()> {
        let out = &built.out;
        let pad = Vec3::new(tol::LINEAR, tol::LINEAR, tol::LINEAR);
        let meets = |a: &Aabb, b: &Aabb| {
            (0..3).all(|i| a.min[i] - pad[i] <= b.max[i] && b.min[i] <= a.max[i] + pad[i])
        };
        let shells: Vec<_> = built.changed.iter().map(|&f| out.face(f).shell).collect();
        for f in out.face_ids().filter(|f| !built.changed.contains(f)) {
            let bb = measure::face_bbox(out, f);
            for region in regions.iter().filter(|r| meets(&r.bbox, &bb)) {
                let faces: Vec<(&Surface, bool)> =
                    region.round.iter().map(|(s, m)| (s, *m)).collect();
                if !shells.contains(&out.face(f).shell)
                    || !clear_of(&out.face(f).surface, &region.rims, &faces)
                {
                    return None;
                }
            }
        }
        Some(())
    }
}

/// Whether the general way would run cylinder `n` on only between its straight edges.
fn sector(s: &Brep, n: FaceId, gone: &[bool]) -> bool {
    let face = s.face(n);
    let Surface::Cylinder(c) = &face.surface else {
        return false;
    };
    face.loops.iter().any(|&lp| {
        s.lp(lp).coedges.iter().any(|&co| {
            let e = s.coedge(co).edge;
            let along = |l: &Line| l.dir.cross(c.frame.z).norm() <= 1e-9;
            matches!(&s.edge(e).curve, Some(Curve::Line(l)) if along(l))
                && s.edge_uses(e).iter().any(|&o| s.coedge_face(o) != n)
                && !s
                    .edge_uses(e)
                    .iter()
                    .any(|&o| gone[s.coedge_face(o).index()])
        })
    })
}

/// `s` with `faces` taken out and the loops round them closed in place, where exactly one way of
/// closing them passes every test; `None` leaves the call to the general way.
pub(super) fn direct(s: &Brep, faces: &[FaceId]) -> Option<Brep> {
    let mut faces = faces.to_vec();
    faces.sort();
    faces.dedup();
    let around = Around::new(s, &faces)?;
    let plans: Vec<Vec<Plan>> = (0..around.pieces)
        .map(|k| around.plans(k))
        .collect::<Option<_>>()?;
    let ways: usize = plans.iter().map(Vec::len).product();
    if ways == 0 || ways > 4 {
        return None;
    }
    let mut found = None;
    for way in 0..ways {
        let mut rest = way;
        let chosen: Vec<&Plan> = plans
            .iter()
            .map(|p| {
                let plan = &p[rest % p.len()];
                rest /= p.len();
                plan
            })
            .collect();
        if let Some(out) = around.attempt(&chosen) {
            if found.is_some() {
                return None;
            }
            found = Some(out);
        }
    }
    found
}
