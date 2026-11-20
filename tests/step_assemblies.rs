//! Several parts in one STEP file, each written once and placed by its instances, and what a read
//! says of each body: its part's name, the uses that place it, and its colour.

use std::f64::consts::{FRAC_PI_2, PI};

use springrep::{
    Error, check,
    math::{Frame, Transform, Vec3},
    measure, primitive,
    step::{self, Instance, Part},
    topo::Brep,
};

fn block() -> Brep {
    primitive::block(&Frame::WORLD, Vec3::new(10.0, 20.0, 30.0), "block").unwrap()
}

fn post() -> Brep {
    primitive::cylinder(&Frame::WORLD, 5.0, 10.0, "post").unwrap()
}

const RED: [f64; 3] = [0.8, 0.1, 0.1];

/// A block, and a post used twice: once stood 40 mm up, clear of the block, once laid along −y
/// at x = 50.
fn written() -> String {
    let (block, post) = (block(), post());
    let parts = [
        Part::new("block", &block, Some(RED)),
        Part::new("post", &post, None),
    ];
    let instances = [
        Instance::new("base", 0, Transform::IDENTITY),
        Instance::new(
            "post-a",
            1,
            Transform::translation(Vec3::new(0.0, 0.0, 40.0)),
        ),
        Instance::new(
            "post-b",
            1,
            Transform::rotation(Vec3::ZERO, Vec3::X, FRAC_PI_2)
                .then(&Transform::translation(Vec3::new(50.0, 0.0, 0.0))),
        ),
    ];
    step::write_assembly("stand", &parts, &instances).unwrap()
}

#[test]
fn an_assembly_reads_back_with_each_instance_where_it_was_placed() {
    let text = written();
    // The post's geometry is in the file once, however often it is used.
    assert_eq!(text.matches("MANIFOLD_SOLID_BREP").count(), 2);
    let import = step::read(text.as_bytes(), "in").unwrap();
    assert!(import.rejected.is_empty(), "{:?}", import.rejected);
    assert!(check::check(&import.brep).is_empty());
    let b = &import.brep;
    // 10 · 20 · 30 and twice π · 5² · 10.
    let want = 6000.0 + 2.0 * PI * 250.0;
    assert!((measure::volume(b) - want).abs() < 1e-6 * want);
    let named: Vec<(&str, &str)> = import
        .bodies
        .iter()
        .map(|x| (x.name.as_deref().unwrap(), x.instance[0].as_str()))
        .collect();
    assert_eq!(
        named,
        [("block", "base"), ("post", "post-a"), ("post", "post-b")]
    );
    assert!(import.bodies.iter().all(|x| x.instance.len() == 1));
    assert_eq!(import.bodies[1].step_id, import.bodies[2].step_id);
    let boxed = |k: usize| measure::solid_bbox(b, import.bodies[k].solid);
    let near = |a: Vec3, e: [f64; 3]| (a - Vec3::new(e[0], e[1], e[2])).norm() < 1e-9;
    assert!(near(boxed(1).min, [-5.0, -5.0, 40.0]) && near(boxed(1).max, [5.0, 5.0, 50.0]));
    // A quarter turn about x takes z to −y: the post lies from y = 0 to −10.
    assert!(near(boxed(2).min, [45.0, -10.0, -5.0]) && near(boxed(2).max, [55.0, 0.0, 5.0]));
}

#[test]
fn a_parts_colour_is_its_bodies_colour_wherever_they_are_placed() {
    let import = step::read(written().as_bytes(), "in").unwrap();
    let colours: Vec<Option<[f64; 3]>> = import.bodies.iter().map(|x| x.colour).collect();
    assert_eq!(colours, [Some(RED), None, None]);
    assert!(import.face_colours.is_empty());
}

#[test]
fn a_face_styled_apart_from_its_body_has_its_own_colour() {
    // As another system writes one face green: a style over the body's, naming the face.
    let text = written();
    let face: String = text
        .lines()
        .find(|l| l.contains("=ADVANCED_FACE("))
        .and_then(|l| l.split('=').next())
        .unwrap()
        .to_string();
    let extra = format!(
        "#900001=DRAUGHTING_PRE_DEFINED_COLOUR('green');\n\
         #900002=FILL_AREA_STYLE_COLOUR('',#900001);\n\
         #900003=FILL_AREA_STYLE('',(#900002));\n\
         #900004=SURFACE_STYLE_FILL_AREA(#900003);\n\
         #900005=SURFACE_SIDE_STYLE('',(#900004));\n\
         #900006=SURFACE_STYLE_USAGE(.BOTH.,#900005);\n\
         #900007=PRESENTATION_STYLE_ASSIGNMENT((#900006));\n\
         #900008=STYLED_ITEM('',(#900007),{face});\n\
         ENDSEC;\nEND-ISO-10303-21;\n"
    );
    let text = text.replace("ENDSEC;\nEND-ISO-10303-21;\n", &extra);
    let import = step::read(text.as_bytes(), "in").unwrap();
    assert_eq!(import.face_colours.len(), 1);
    let (f, colour) = import.face_colours[0];
    assert_eq!(colour, [0.0, 1.0, 0.0]);
    // The first face written is one of the block's.
    assert!(import.brep.solid_faces(import.bodies[0].solid).contains(&f));
    assert_eq!(import.bodies[0].colour, Some(RED));
}

#[test]
fn a_file_of_one_part_names_its_body_and_gives_it_no_use() {
    let text = step::write(&block(), "lone").unwrap();
    let import = step::read(text.as_bytes(), "in").unwrap();
    assert_eq!(import.bodies.len(), 1);
    assert_eq!(import.bodies[0].name.as_deref(), Some("lone"));
    assert!(import.bodies[0].instance.is_empty() && import.bodies[0].colour.is_none());
}

#[test]
fn an_instance_of_a_part_the_file_lacks_is_invalid() {
    let b = block();
    let parts = [Part::new("block", &b, None)];
    let stray = [Instance::new("stray", 1, Transform::IDENTITY)];
    assert!(matches!(
        step::write_assembly("a", &parts, &stray),
        Err(Error::InvalidInput(_))
    ));
}
