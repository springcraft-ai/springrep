//! A solid written with voids that are not voids: bodies outside its outer shell, as some files
//! write a compound of bodies. Each is read as a solid of its own, and the import says so.

use springrep::{boolean, check, measure, step};

/// Two cubes of 10 mm side by side, sharing the face x = 10, written as one `BREP_WITH_VOIDS` whose
/// outer shell is the first and whose one void is the second. A void lies inside its outer shell
/// (ISO 10303-42), so the file holds two bodies: two solids of 10³ = 1000 mm³, the second read
/// from the shell the file lists as a void, and named in `separated`.
#[test]
fn a_void_touching_the_outside_of_its_outer_shell_is_read_as_a_solid_of_its_own() {
    let import = step::read(include_bytes!("fixtures/void-outside-touching.step"), "imp").unwrap();
    assert!(import.rejected.is_empty(), "{:?}", import.rejected);
    assert!(check::check_each(&import.brep).is_empty());
    let solids: Vec<_> = import.brep.solid_ids().collect();
    assert_eq!(solids.len(), 2);
    for s in solids {
        let v = measure::volume(&boolean::solids_of(&import.brep, &[s]));
        assert!((v - 1000.0).abs() < 1e-9, "{v}");
    }
    assert_eq!(import.separated.len(), 1);
    let s = &import.separated[0];
    assert_eq!(
        (s.index, s.step_id, s.shells.as_slice()),
        (0, 300, &[299][..])
    );
    assert_eq!(import.bodies.len(), 2);
    // Together the two cubes are the box 20 × 10 × 10.
    assert!((measure::volume(&import.brep) - 2000.0).abs() < 1e-9);
}
