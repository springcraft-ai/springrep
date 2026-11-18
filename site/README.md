# site

springrep's landing page. It tells a Rust developer what springrep is, how to start, and how it
scores on [brep-conformance](https://github.com/springcraft-ai/brep-conformance).

## Overview

The page is one HTML file in the Springcraft Alt-1 design system. It loads the design system's
tokens, fonts and logos from that system's published build on the CDN, at an immutable path set in
build.py. It is built from:

- [template.html](template.html) — the page, its styles and its scripts.
- [bracket.rs](../examples/bracket.rs) — the getting-started example, shown on the page as code.
- [data/flanged_tube.json](data/flanged_tube.json) — the animated part in the hero: each build
  step's edges, a coarse mesh that hides edges behind faces, the profiles it swept and what the
  kernel reported, as the [flanged_tube.rs](../examples/flanged_tube.rs) example prints them, with
  each step's new edges and faces marked by [fresh.py](fresh.py). It is committed so the page builds
  without a Rust toolchain.

[build.py](build.py) fills the template's `{{…}}` slots and writes springrep.html into a git-ignored
dist folder here. The crate's manifest excludes this folder, so the site never ships with the crate.
For a host that loads nothing from other origins, such as a sandboxed preview, `--preview` fetches
the CDN files at build time and writes them into springrep.preview.html instead.

The [pages workflow](../.github/workflows/pages.yml) publishes the page to GitHub Pages, with the
crate's `cargo doc` output under `api/`, where the page's API docs link points. A local build has no
`api/` folder, so that link only works on the published site.

The link-preview image (`og:image`) is a 1200×630 PNG of the hero with the finished part and without
the leaderboard. It lives on the CDN under `springrep/`, with a content hash in its name so it can be
cached for good: a new image gets a new name, and the template's `og:image` tag changes with it.

The benchmark figures on the page are specimen data until brep-conformance publishes real results.

## Setup

Python 3 and, to regenerate the hero data, the Rust toolchain that
[rust-toolchain.toml](../rust-toolchain.toml) names.

## Usage

Build the page:

```bash
python3 site/build.py
python3 site/build.py --preview
```

After changing the hero part or springrep itself, regenerate its data first:

```bash
cargo run --release --example flanged_tube | python3 site/fresh.py
```

Check the getting-started example still gives the volume the page and its agent prompt quote
(18276.0016 mm³). It writes `bracket.step` to the current directory:

```bash
cargo run --example bracket
```
