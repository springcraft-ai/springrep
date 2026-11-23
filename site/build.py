"""Assemble dist/springrep.html from template.html, the hero data and the example.

The design system comes from its published build on the CDN, never from a checkout. With
--preview, those files are fetched now and written into the page, for hosts that load nothing from
other origins; the result goes to dist/springrep.preview.html.

The leaderboard comes from the brep-conformance results site as it stands now, and is written into
the page. With --offline, the copy the last build kept in dist/ is used instead.
"""

import base64
import html
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dist/springrep.html"
PREVIEW = HERE / "dist/springrep.preview.html"
LEADERBOARD_URL = "https://springcraft-ai.github.io/brep-conformance/leaderboard.json"
LEADERBOARD_KEPT = HERE / "dist/leaderboard.json"
GRADES = ("1e-7", "1e-6", "1e-5", "1e-3", "fail", "unsupported", "refused", "timeout", "crash")
# An immutable build of sc-1.1: a new build of the design system gets a new path.
SC = "https://cdn.springcraft.ai/design/sc-1.1/mwl4jqxo"

GH = (
    '<svg class="gh-mark" viewBox="0 0 16 16" aria-hidden="true">'
    '<path d="M8 0c4.42 0 8 3.58 8 8a8.013 '
    "8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 "
    "1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2"
    ".82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92"
    "-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21"
    "-1.61.55-2.33-.66-"
    ".15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 "
    '1.3.01 1.49 0 .21-.15.45-.55.37A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>'
)
KW = r"\b(use|fn|let|pub|mut|Ok|Result|Box|dyn)\b"
FN = r"\b(extrude|difference|fillet|measure::volume|measure::area|step::write)\b"


def fetch(path):
    with urllib.request.urlopen(f"{SC}/{path}") as r:
        return r.read().decode()


def inline_sc(page):
    """The page with sc.css and the logos it links written in, sc.css still in its layer."""
    css = fetch("sc.css")
    imports = re.findall(r'@import\s*"([^"]+)";', css)
    rest = re.sub(r'@import\s*"[^"]+";', "", css)
    layered = "".join(f'@import url("{u}") layer(sc);\n' for u in imports)
    page = page.replace(f'@import url("{SC}/sc.css") layer(sc);', layered + f"@layer sc {{{rest}}}")

    def data_uri(m):
        svg = re.sub(r"<metadata>.*?</metadata>", "", fetch(m.group(2)), flags=re.S)
        data = base64.b64encode(svg.encode()).decode()
        return f'{m.group(1)}="data:image/svg+xml;base64,{data}"'

    return re.sub(rf'(src|href)="{re.escape(SC)}/(assets/[^"]+)"', data_uri, page)


def highlight(src):
    out = []
    for line in html.escape(src, quote=False).splitlines():
        parts = re.split(r'(&quot;.*?&quot;|"[^"]*")', line)
        res = []
        for i, p in enumerate(parts):
            if i % 2:
                res.append(f'<span class="st">{p}</span>')
                continue
            p = re.sub(r"\b(\d+\.\d+)\b", r'<span class="nu">\1</span>', p)
            p = re.sub(KW, r'<span class="kw">\1</span>', p)
            p = re.sub(FN, r'<span class="fn">\1</span>', p)
            res.append(p)
        out.append("".join(res))
    return "\n".join(out)


def drawable(d):
    """Whether the page's fromLive accepts this leaderboard.json; the two must agree."""
    try:
        at = {g: d["grades"].index(g) for g in GRADES if g in d["grades"]}
        kernels = d["kernels"]
        run = next((k for k in kernels if k.get("maintainers")), kernels[0])["run"]
        return (
            d["cases"] > 0
            and datetime.fromisoformat(run["date"].replace("Z", "+00:00")).timestamp() > 0
            and run["cores"]
            and run["tolerance"]
            and all(
                sum(k["counts"][i] for i in at.values()) == d["cases"]
                and all(k["time"][q] >= 0 for q in ("median", "q1", "q3"))
                for k in kernels
            )
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def leaderboard(offline):
    """The results site's leaderboard.json, as text safe inside a script element."""
    if offline:
        text = LEADERBOARD_KEPT.read_text()
    else:
        with urllib.request.urlopen(LEADERBOARD_URL) as r:
            text = r.read().decode()
    d = json.loads(text)
    if not drawable(d):
        sys.exit(f"{LEADERBOARD_URL}: not a board the page can draw")
    LEADERBOARD_KEPT.parent.mkdir(exist_ok=True)
    LEADERBOARD_KEPT.write_text(text)
    return json.dumps(d, separators=(",", ":")).replace("</", "<\\/")


def version():
    """The crate's version, from the package table that opens Cargo.toml."""
    manifest = (HERE.parent / "Cargo.toml").read_text()
    m = re.search(r'^version = "([^"]+)"$', manifest.split("\n[", 1)[0], flags=re.M)
    if not m:
        sys.exit("Cargo.toml: no package version")
    return m.group(1)


def main():
    t = (HERE / "template.html").read_text()
    t = t.replace("{{GH}}", GH)
    t = t.replace("{{SC_CSS}}", f'@import url("{SC}/sc.css") layer(sc);')
    t = t.replace("{{SC}}", SC)
    t = t.replace("{{PART}}", (HERE / "data/flanged_tube.json").read_text().strip())
    t = t.replace("{{VERSION}}", html.escape(version()))
    t = t.replace("{{LEADERBOARD_URL}}", LEADERBOARD_URL)
    t = t.replace("{{LEADERBOARD}}", leaderboard("--offline" in sys.argv[1:]))
    t = t.replace("{{MAIN_RS}}", highlight((HERE.parent / "examples/bracket.rs").read_text()))
    out = OUT
    if "--preview" in sys.argv[1:]:
        t, out = inline_sc(t), PREVIEW
    out.parent.mkdir(exist_ok=True)
    out.write_text(t)
    print(f"{out} ({len(t)} bytes)")


if __name__ == "__main__":
    main()
