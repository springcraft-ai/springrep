# Contributing

How to propose a change to springrep, and what a change must be for a maintainer to merge it.

Issues and pull requests are welcome. A Springcraft maintainer reviews every pull request, ours
included, and nothing reaches `main` without that review.

## What belongs here

- **A wrong result.** A shape the kernel builds wrongly, a query it answers wrongly, or a sound
  input it refuses. Give the smallest input that shows it.
- **A new operation, or a faster one.** Open an issue first, so the design is agreed before the
  code. [design.md](../doc/design.md) says how the kernel is laid out.
- **The documents.**

A test of [brep-conformance](https://github.com/springcraft-ai/brep-conformance) that springrep
fails is a wrong result, and belongs here. A test that expects the wrong answer belongs in that
repository, and so does a problem with how the suite drives springrep.

## Where the work comes from

springrep is written from published mathematics and from first principles, and every contribution
keeps that true.

- **Your own work, or work you may submit.** You wrote the change, or you have the right to submit
  it under this repository's licence.
- **Nothing from proprietary sources.** No part of a change derives from proprietary source code,
  from decompiled or disassembled code, or from documentation you hold under a non-disclosure
  agreement or a licence that restricts its use. If you have seen such material for the thing you
  are changing, leave that change to someone who has not.
- **Cite what you used.** An algorithm from a paper or a book names it where the code uses it.
- **No proprietary kernel is run.** No workflow, test or benchmark in this repository runs one, and
  no expected value in a test comes from one. An expected value is a closed form, or a figure the
  test derives itself.

## Names

Do not name a proprietary CAD kernel or CAD program, its vendor, or its API, in code, tests,
comments, commit messages or documents. Describe the behaviour instead. The repository holds no
list of such names, plain or hashed, and a change must not add one.

The rule is about CAD software only. Open-source projects are named where it helps, and the tools
the project is built and hosted with are named as usual.

## Files

- No compiled binaries and no vendor SDK files.
- No CAD files outside a folder named `fixtures`. A fixture is a file springrep wrote, or one whose
  terms allow it to be redistributed, with a notice of those terms beside it.
- No secrets. A key or a token in a commit stays in the history after it is removed.

## Dependencies

springrep has no dependencies: it builds from the Rust standard library alone, and a pull request
does not add one. The `no-dependencies` check fails a pull request whose `Cargo.lock` lists any
package but springrep. To propose a dependency, open an issue first and say why the kernel cannot
do without it.

## Licence of contributions

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion
in this repository, as defined in the Apache License, Version 2.0, is licensed under both the
[MIT licence](../LICENSE-MIT) and the [Apache License, Version 2.0](../LICENSE-APACHE), without
any additional terms or conditions.
