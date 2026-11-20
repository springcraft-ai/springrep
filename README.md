# springrep

[![crates.io](https://img.shields.io/crates/v/springrep.svg)](https://crates.io/crates/springrep)
[![docs.rs](https://docs.rs/springrep/badge.svg)](https://docs.rs/springrep)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/springcraft-ai/springrep/blob/main/LICENSE)

An exact boundary-representation (B-rep) geometry kernel, in safe Rust with no dependencies.

springrep builds solids from exact analytic and swept surfaces. Every operation returns a solid
that passes the validity checker, or an error.

- **Build:** extrude, revolve and sweep planar profiles, along a path or a helix.
- **Combine:** union, intersection and difference.
- **Finish:** chamfers and fillets, shells and drafts.
- **Measure:** area, volume, mass properties, bounding boxes, closest distances and ray crossings,
  from the exact geometry.
- **Exchange:** read and write STEP AP214, and make watertight triangle meshes for display.
- **History:** every face carries the names of the operations that made it, so a later operation
  selects faces and edges by name.

## Example

A plate with a filleted bore through it:

```rust
use springrep::{
    blend::fillet,
    boolean::difference,
    math::{Frame, Vec2, Vec3},
    measure,
    profile::Profile,
    step,
    sweep::extrude,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = Frame::new(Vec3::ZERO, Vec3::Z, Vec3::X)?;
    let below = Frame::new(Vec3::new(0.0, 0.0, -1.0), Vec3::Z, Vec3::X)?;

    let plate = extrude(&Profile::rect(Vec2::ZERO, 60.0, 40.0), &base, 8.0, "plate")?;
    let bore = extrude(&Profile::circle(Vec2::ZERO, 12.0), &below, 10.0, "bore")?;

    let part = difference(&plate, &bore)?;
    let part = fillet(&part, &[("plate.end", "bore.side")], 1.5, "round")?;

    println!("volume {:.3} mm³", measure::volume(&part));
    println!("area   {:.3} mm²", measure::area(&part));
    let text = step::write(&part, "bracket")?;
    println!("{} faces, {} bytes of STEP", part.face_count(), text.len());
    Ok(())
}
```

More examples are in [`examples/`](https://github.com/springcraft-ai/springrep/tree/main/examples).
Run one with `cargo run --example bracket`.

## Installation

```toml
[dependencies]
springrep = "2"
```

## Documentation

- API reference: <https://docs.rs/springrep>
- Website: <https://springcraft-ai.github.io/springrep/>
- [Design](https://github.com/springcraft-ai/springrep/blob/main/doc/design.md): the
  representation, the tolerance policy, the validity rules, the history rules and the module map.

## Contributing

Read [CONTRIBUTING.md](https://github.com/springcraft-ai/springrep/blob/main/.github/CONTRIBUTING.md)
before you open an issue or a pull request.

## Licence

springrep is under the MIT licence: see
[LICENSE](https://github.com/springcraft-ai/springrep/blob/main/LICENSE).
