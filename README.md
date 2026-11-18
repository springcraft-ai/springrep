# springrep

An exact boundary-representation geometry kernel, in safe Rust with no dependencies.

Website: <https://springcraft-ai.github.io/springrep/>

## Overview

Solids are exact analytic and swept surfaces bounded by curves, held in a typed arena
([`src/topo/`](src/topo/mod.rs)). Sweeps ([`src/sweep.rs`](src/sweep.rs)) build solids from planar
profiles ([`src/profile.rs`](src/profile.rs)), booleans ([`src/boolean/`](src/boolean/mod.rs))
combine them, and chamfers and fillets ([`src/blend.rs`](src/blend.rs)) sew swept blend faces into
the solid, or cut and add the swept blend region where sewing cannot close it. A profile swept along
a helix is screwed about its axis ([`src/sweep/helix.rs`](src/sweep/helix.rs)), and shells and
drafts move planar faces onto new planes ([`src/edit/tweak.rs`](src/edit/tweak.rs)). STEP AP214
comes in and goes out through [`src/step/`](src/step/mod.rs). Every result passes the validity
checker ([`src/check.rs`](src/check.rs)) or comes back as an error. Area, volume, mass properties
and box come from the exact geometry ([`src/measure.rs`](src/measure.rs)), as do closest distances
([`src/distance.rs`](src/distance.rs)) and rays' crossings of faces ([`src/rays.rs`](src/rays.rs));
meshes are watertight by construction ([`src/mesh/`](src/mesh/mod.rs)). Every face carries the names
of the operations that made it ([`src/topo/names.rs`](src/topo/names.rs)).

## Usage

```bash
cargo test -p springrep          # fast unit tests only
```

## Additional Information

See [doc/intro.md](doc/intro.md).

springrep is under the MIT licence: [LICENSE](LICENSE). It has no dependencies.
