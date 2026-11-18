# CLAUDE.md — springrep

Read [`README.md`](README.md) and [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) first. This file is
the rules an agent is most likely to break.

## Name no proprietary CAD product

Never write the name of a proprietary CAD kernel or CAD program, its vendor, or its API: not in
code, tests, comments, commit messages or documents. Describe the behaviour instead. Do not add a
list of such names, plain or hashed, to check against; the rule is enforced by review.

The rule is about CAD software only. Open-source projects are named where it helps, and the tools
the project is built and hosted with are named as usual.

## Nothing from proprietary sources

Do not derive code, tests or documents from proprietary source code, from decompiled code, or from a
CAD vendor's documentation under a non-disclosure agreement or a restrictive licence. Do not quote
the text of a standard. Cite the paper or book an algorithm comes from where the code uses it.

## No proprietary kernel

Never run a proprietary kernel in a workflow, a test or a benchmark, and never take a test's
expected value from one. An expected value is a closed form, or a figure the test derives itself.

## Dependencies and files

springrep has no dependencies: do not add one, not a build or development dependency either. A
check fails a pull request whose `Cargo.lock` lists any package but springrep. No compiled binary,
no vendor SDK file, and no CAD file outside a folder named `fixtures`.
