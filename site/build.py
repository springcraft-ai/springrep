"""Assemble dist/springrep.html from template.html, the hero data and the example.

The design system comes from its published build on the CDN, never from a checkout. With
--preview, those files are fetched now and written into the page, for hosts that load nothing from
other origins; the result goes to dist/springrep.preview.html.
"""

import base64
import html
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dist/springrep.html"
PREVIEW = HERE / "dist/springrep.preview.html"
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


def version():
    """The crate's version, from Cargo.toml's package table."""
    manifest = (HERE.parent / "Cargo.toml").read_text()
    package = re.search(r"^\[package\][ \t]*$(.*?)(?=^\[|\Z)", manifest, flags=re.M | re.S)
    m = package and re.search(r'^version\s*=\s*"([^"]+)"', package.group(1), flags=re.M)
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
    t = t.replace("{{MAIN_RS}}", highlight((HERE.parent / "examples/bracket.rs").read_text()))
    out = OUT
    if "--preview" in sys.argv[1:]:
        t, out = inline_sc(t), PREVIEW
    out.parent.mkdir(exist_ok=True)
    out.write_text(t)
    print(f"{out} ({len(t)} bytes)")


if __name__ == "__main__":
    main()
