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
    written_with(None)
}

/// [`written`], through `write_assembly_with` where `options` are given.
fn written_with(options: Option<&step::WriteOptions>) -> String {
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
    match options {
        Some(o) => step::write_assembly_with("stand", &parts, &instances, o),
        None => step::write_assembly("stand", &parts, &instances),
    }
    .unwrap()
}

#[test]
fn an_assembly_reads_back_with_each_instance_where_it_was_placed() {
    assert_eq!(
        written(),
        written_with(Some(&step::WriteOptions::default()))
    );
    for pcurves in [true, false] {
        let text = written_with(Some(&step::WriteOptions::default().pcurves(pcurves)));
        assert_eq!(text.contains("=PCURVE("), pcurves);
        reads_back_where_placed(&text);
    }
}

fn reads_back_where_placed(text: &str) {
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

#[test]
fn a_relationship_to_another_parts_shape_cannot_place_the_child() {
    let text = written();
    let relation = "=(REPRESENTATION_RELATIONSHIP('','',";
    let mut links = text.lines().filter(|line| line.contains(relation));
    let block = links.next().unwrap();
    let (_, post_endpoints) = links.next().unwrap().split_once(relation).unwrap();
    let (post_rep, _) = post_endpoints.split_once(',').unwrap();
    let (start, endpoints) = block.split_once(relation).unwrap();
    let (_, rest) = endpoints.split_once(',').unwrap();
    let text = text.replacen(block, &format!("{start}{relation}{post_rep},{rest}"), 1);
    let import = step::read(text.as_bytes(), "in").unwrap();
    let instances: Vec<_> = import.bodies.into_iter().map(|b| b.instance).collect();
    let rejected = import.rejected;
    assert_eq!(instances, [["post-a"], ["post-b"]], "{rejected:?}");
    assert_eq!(rejected.len(), 1, "{rejected:?}");
}

#[test]
fn a_relationship_can_place_the_child_in_rep_2() {
    let mut text = written();
    for entity in [
        "=ITEM_DEFINED_TRANSFORMATION('','',",
        "=(REPRESENTATION_RELATIONSHIP('','',",
    ] {
        let line = text.lines().rfind(|line| line.contains(entity)).unwrap();
        let (start, args) = line.split_once(entity).unwrap();
        let (pair, rest) = args.split_once(')').unwrap();
        let (first, second) = pair.split_once(',').unwrap();
        text = text.replacen(line, &format!("{start}{entity}{second},{first}){rest}"), 1);
    }
    reads_back_where_placed(&text);
}

#[test]
fn a_use_with_inconsistent_placement_links_is_rejected() {
    let text = written();
    let entity = "=CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(";
    let mut links = text.lines().filter(|line| line.contains(entity));
    let upright = links.nth(1).unwrap();
    let laid_down = links.next().unwrap().split_once(entity).unwrap().1;
    let (laid_down_rel, _) = laid_down.split_once(',').unwrap();
    let (_, shape) = upright.split_once(',').unwrap();
    for relationship in [laid_down_rel, "$"] {
        let extra = format!("#900001{entity}{relationship},{shape}");
        for links in [format!("{upright}\n{extra}"), format!("{extra}\n{upright}")] {
            let input = text.replacen(upright, &links, 1);
            let import = step::read(input.as_bytes(), "in").unwrap();
            let instances: Vec<_> = import.bodies.into_iter().map(|b| b.instance).collect();
            assert_eq!(instances, [["base"], ["post-b"]], "{:?}", import.rejected);
            assert_eq!(import.rejected.len(), 1, "{:?}", import.rejected);
            if relationship == "$" {
                assert!(import.rejected[0].contains("STEP #900001: unset relationship"));
            }
        }
    }
}

#[test]
fn duplicate_placements_must_agree_within_the_parent_contexts_uncertainty() {
    let text = written();
    let parent_uncertainty = text
        .lines()
        .rfind(|line| line.contains("=UNCERTAINTY_MEASURE_WITH_UNIT("))
        .unwrap();
    let (start, measure) = parent_uncertainty.split_once("LENGTH_MEASURE(").unwrap();
    let (_, rest) = measure.split_once(')').unwrap();
    let text = text.replace(
        parent_uncertainty,
        &format!("{start}LENGTH_MEASURE(0.001){rest}"),
    );

    let link = "=CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(";
    let (upright, args) = text
        .lines()
        .filter_map(|line| line.split_once(link))
        .nth(1)
        .unwrap();
    let (relation, shape) = args.split_once(',').unwrap();
    let (_, endpoints) = text
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{relation}=")))
        .unwrap()
        .split_once("REPRESENTATION_RELATIONSHIP(")
        .unwrap();
    let (endpoints, _) = endpoints.split_once(')').unwrap();

    for z in [40.0005, 40.002] {
        let input = text.replacen(
            &format!("{upright}{link}{args}"),
            &format!(
                "{upright}{link}{args}\n\
                 #900001{link}#900002,{shape}\n\
                 #900002=(REPRESENTATION_RELATIONSHIP({endpoints}) \
                 REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#900003) \
                 SHAPE_REPRESENTATION_RELATIONSHIP());\n\
                 #900003=CARTESIAN_TRANSFORMATION_OPERATOR_3D('','',$,$,#900004,1.,$);\n\
                 #900004=CARTESIAN_POINT('',(0.,0.,{z}));"
            ),
            1,
        );
        let import = step::read(input.as_bytes(), "in").unwrap();
        if z < 40.001 {
            assert!(import.rejected.is_empty(), "{:?}", import.rejected);
            assert_eq!(import.bodies.len(), 3);
            let post = &import.bodies[1];
            assert_eq!(post.instance, ["post-a"]);
            let bounds = measure::solid_bbox(&import.brep, post.solid);
            assert!((bounds.min.z - 40.0).abs() <= springrep::tol::LINEAR);
            let [adjustment] = &import.placement_adjustments[..] else {
                panic!("{:?}", import.placement_adjustments);
            };
            assert_eq!(format!("#{}", adjustment.chosen), relation);
            assert_eq!(adjustment.relationships, (adjustment.chosen, 900002));
            assert!((adjustment.moved - 0.0005).abs() <= springrep::tol::LINEAR);
        } else {
            let instances: Vec<_> = import.bodies.into_iter().map(|b| b.instance).collect();
            assert_eq!(instances, [["base"], ["post-b"]], "{:?}", import.rejected);
            assert_eq!(import.rejected.len(), 1, "{:?}", import.rejected);
            let expected = format!("STEP {relation}, #900002: placements differ by 2.0e-3 mm");
            assert!(
                import.rejected[0].contains(&expected),
                "{:?}",
                import.rejected
            );
            assert!(import.placement_adjustments.is_empty());
        }
    }
}
