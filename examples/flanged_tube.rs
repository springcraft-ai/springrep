// Builds the landing page's hero part, a flanged tube, and prints each build step's edges, a coarse
// mesh, and the profile it extruded, as JSON: the page draws each step as the operation it is.
use std::f64::consts::TAU;
use std::time::Instant;

use springrep::{
    blend::{chamfer, fillet},
    boolean::{difference, difference_all, union},
    check,
    math::{Frame, Vec2, Vec3},
    measure,
    mesh::{self, MeshOptions},
    profile::Profile,
    sweep::extrude,
    topo::Brep,
};

const LEN: f64 = 56.0;
const FLANGE: f64 = 6.0;

fn at(z: f64) -> Frame {
    Frame::new(Vec3::new(0.0, 0.0, z), Vec3::Z, Vec3::X).unwrap()
}

fn disc(d: f64, z: f64, h: f64, name: &str) -> springrep::Result<Brep> {
    extrude(&Profile::circle(Vec2::ZERO, d), &at(z), h, name)
}

/// Circles swept together, as (centre x, centre y, diameter), drawn at height `z` and swept by `h`
/// (down when negative): what a person would sketch and extrude, as the page animates it.
struct Sweep {
    circles: Vec<(f64, f64, f64)>,
    z: f64,
    h: f64,
}

fn sweep(circles: Vec<(f64, f64, f64)>, z: f64, h: f64) -> Sweep {
    Sweep { circles, z, h }
}

/// One build step: the solid after it, the circles it swept (none for a blend), what it did in a
/// few words, and how long the operation took.
struct Step {
    label: &'static str,
    body: Brep,
    drawn: Vec<Sweep>,
    op: String,
    ms: f64,
}

/// Milliseconds since `clock`, which restarts.
fn lap(clock: &mut Instant) -> f64 {
    let ms = clock.elapsed().as_secs_f64() * 1e3;
    *clock = Instant::now();
    ms
}

fn build() -> springrep::Result<(Brep, Vec<Step>)> {
    let mut steps = vec![];
    let top_z = LEN - FLANGE;
    let mut clock = Instant::now();
    let body = disc(64.0, 0.0, FLANGE, "base")?;
    steps.push(Step {
        label: "extrude flange",
        drawn: vec![sweep(vec![(0.0, 0.0, 64.0)], 0.0, FLANGE)],
        op: format!("extrude {FLANGE} mm"),
        ms: lap(&mut clock),
        body: body.clone(),
    });
    // The tube stands on the flange's top face and stops where the top flange begins.
    let body = union(&body, &disc(28.0, FLANGE, top_z - FLANGE, "tube")?)?;
    steps.push(Step {
        label: "extrude tube",
        drawn: vec![sweep(vec![(0.0, 0.0, 28.0)], FLANGE, top_z - FLANGE)],
        op: format!("extrude {} mm, union", top_z - FLANGE),
        ms: lap(&mut clock),
        body: body.clone(),
    });
    let body = union(&body, &disc(64.0, top_z, FLANGE, "top")?)?;
    steps.push(Step {
        label: "extrude top flange",
        drawn: vec![sweep(vec![(0.0, 0.0, 64.0)], top_z, FLANGE)],
        op: format!("extrude {FLANGE} mm, union"),
        ms: lap(&mut clock),
        body: body.clone(),
    });
    let body = fillet(
        &body,
        &[("base.end", "tube.side"), ("top.start", "tube.side")],
        4.0,
        "root",
    )?;
    steps.push(Step {
        label: "fillet roots",
        drawn: vec![],
        op: "fillet 2 edges r4".into(),
        ms: lap(&mut clock),
        body: body.clone(),
    });
    // Cutting tools run a millimetre past each face so no sliver is left; the page draws the cut
    // from the face it is sketched on.
    let body = difference(&body, &disc(21.0, -1.0, LEN + 2.0, "bore")?)?;
    steps.push(Step {
        label: "cut bore",
        drawn: vec![sweep(vec![(0.0, 0.0, 21.0)], LEN, -LEN)],
        op: "cut bore through".into(),
        ms: lap(&mut clock),
        body: body.clone(),
    });
    let circle: Vec<(f64, f64, f64)> = (0..6)
        .map(|i| {
            let a = TAU * i as f64 / 6.0;
            (24.5 * a.cos(), 24.5 * a.sin(), 6.6)
        })
        .collect();
    let mut holes = vec![];
    for z in [-1.0, top_z - 1.0] {
        for &(x, y, d) in &circle {
            let c = Vec2::new(x, y);
            holes.push(extrude(
                &Profile::circle(c, d),
                &at(z),
                FLANGE + 2.0,
                "bolt",
            )?);
        }
    }
    let refs: Vec<&Brep> = holes.iter().collect();
    let body = difference_all(&body, &refs)?;
    steps.push(Step {
        label: "cut bolt holes",
        drawn: vec![
            sweep(circle.clone(), FLANGE, -FLANGE),
            sweep(circle, LEN, -FLANGE),
        ],
        op: "cut 12 holes".into(),
        ms: lap(&mut clock),
        body: body.clone(),
    });
    let body = chamfer(
        &body,
        &[("base.start", "bore.side"), ("top.end", "bore.side")],
        1.5,
        "lead",
    )?;
    steps.push(Step {
        label: "chamfer bore",
        drawn: vec![],
        op: "chamfer 2 edges 1.5".into(),
        ms: lap(&mut clock),
        body: body.clone(),
    });
    Ok((body, steps))
}

fn main() {
    let t = Instant::now();
    let (b, steps) = build().expect("build");
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let issues = check::check(&b);
    let (area, volume) = measure::area_and_volume(&b);
    let bx = measure::bbox(&b);
    let steps: Vec<String> = steps
        .iter()
        .map(|s| {
            let edges = s
                .body
                .edge_ids()
                .filter(|&e| !s.body.edge(e).is_degenerate())
                .count();
            format!(
                "{{\"label\":\"{}\",\"op\":\"{}\",\"ms\":{:.3},\"issues\":{},\
                 \"faces\":{},\"edge_count\":{edges},\"volume\":{:.6},\"edges\":[{}],{},\
                 \"tool\":{}}}",
                s.label,
                s.op,
                s.ms,
                check::check(&s.body).len(),
                s.body.face_count(),
                measure::volume(&s.body),
                polylines(&s.body).join(","),
                coarse_mesh(&s.body),
                tool_json(&s.drawn)
            )
        })
        .collect();
    println!(
        "{{\"build_ms\":{ms:.3},\"issues\":{},\"volume\":{volume:.6},\"area\":{area:.6},\
         \"bbox\":[[{},{},{}],[{},{},{}]],\"steps\":[{}]}}",
        issues.len(),
        round(bx.min.x),
        round(bx.min.y),
        round(bx.min.z),
        round(bx.max.x),
        round(bx.max.y),
        round(bx.max.z),
        steps.join(","),
    );
}

/// Each edge as points along it. A seam, where a closed face meets itself, is left out: it is a
/// cut in the face's parameters, not an edge anyone would see.
fn polylines(b: &Brep) -> Vec<String> {
    let mut edges = vec![];
    for e in b.edge_ids() {
        let ed = b.edge(e);
        let Some(c) = &ed.curve else { continue };
        let uses = b.edge_uses(e);
        if uses.len() == 2 && b.coedge_face(uses[0]) == b.coedge_face(uses[1]) {
            continue;
        }
        let n = if c.kind() == "line" { 16 } else { 64 };
        let pts: Vec<String> = (0..=n)
            .map(|i| {
                let s = ed.range.0 + (ed.range.1 - ed.range.0) * i as f64 / n as f64;
                let p = c.eval(s);
                format!("[{},{},{}]", round(p.x), round(p.y), round(p.z))
            })
            .collect();
        edges.push(format!("[{}]", pts.join(",")));
    }
    edges
}

/// Positions, then triangles as `[a, b, c, face]`: enough to hide edges behind faces and find
/// where a curved face turns away from the eye.
fn coarse_mesh(b: &Brep) -> String {
    let opt = MeshOptions {
        linear: 0.15,
        angular: 0.3,
    };
    let m = mesh::tessellate(b, &opt).unwrap();
    let pos: Vec<String> = m
        .positions
        .iter()
        .map(|p| format!("{},{},{}", round2(p.x), round2(p.y), round2(p.z)))
        .collect();
    let mut tris = vec![];
    for (k, (_, range)) in m.faces.iter().enumerate() {
        for t in &m.triangles[range.clone()] {
            tris.push(format!("{},{},{},{k}", t[0], t[1], t[2]));
        }
    }
    format!("\"pos\":[{}],\"tri\":[{}]", pos.join(","), tris.join(","))
}

/// Each sweep's circles as closed point lists where they were drawn, with the sweep's direction.
fn tool_json(sweeps: &[Sweep]) -> String {
    if sweeps.is_empty() {
        return "null".into();
    }
    let mut profiles = vec![];
    for s in sweeps {
        for &(x, y, d) in &s.circles {
            let pts: Vec<String> = (0..=64)
                .map(|i| {
                    let a = TAU * i as f64 / 64.0;
                    let (px, py) = (x + d / 2.0 * a.cos(), y + d / 2.0 * a.sin());
                    format!("[{},{},{}]", round(px), round(py), round(s.z))
                })
                .collect();
            profiles.push(format!(
                "{{\"pts\":[{}],\"dir\":[0,0,{}]}}",
                pts.join(","),
                s.h
            ));
        }
    }
    format!("{{\"profiles\":[{}]}}", profiles.join(","))
}

fn round(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
