"""Mark which edges and faces each hero build step adds, so the page lights only those.

Reads the flanged_tube example's JSON on stdin and writes data/flanged_tube.json. An edge counts
as old when three points along it lie on the previous step's edges, and a face when all of its
triangles lie on the previous step's mesh: a chamfer or boolean rebuilds the whole solid, so its
ids say nothing about what changed.
"""

import json
import math
import sys
from pathlib import Path

TOL = 0.05
# Two tessellations of one surface sit up to about twice the mesh's 0.15 mm deflection apart.
FACE_TOL = 0.4
OUT = Path(__file__).resolve().parent / "data/flanged_tube.json"


def seg_dist(p, a, b):
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    length2 = sum(x * x for x in ab)
    t = 0 if length2 == 0 else max(0, min(1, sum(ab[i] * ap[i] for i in range(3)) / length2))
    return math.dist(p, [a[i] + ab[i] * t for i in range(3)])


def on(p, edges):
    return any(seg_dist(p, e[i - 1], e[i]) < TOL for e in edges for i in range(1, len(e)))


def mid(a, b):
    return [(a[i] + b[i]) / 2 for i in range(3)]


def probes(e):
    return [e[(len(e) - 1) // 2], mid(e[0], e[1]), mid(e[-2], e[-1])]


def sub3(u, v):
    return [u[i] - v[i] for i in range(3)]


def dot3(u, v):
    return sum(u[i] * v[i] for i in range(3))


def tri_dist(p, a, b, c):
    """Distance from p to triangle abc (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab, ac, ap = sub3(b, a), sub3(c, a), sub3(p, a)
    d1, d2 = dot3(ab, ap), dot3(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return math.dist(p, a)
    bp = sub3(p, b)
    d3, d4 = dot3(ab, bp), dot3(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return math.dist(p, b)
    cp = sub3(p, c)
    d5, d6 = dot3(ab, cp), dot3(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return math.dist(p, c)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return seg_dist(p, a, b)
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return seg_dist(p, a, c)
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return seg_dist(p, b, c)
    s = va + vb + vc
    v, w = vb / s, vc / s
    return math.dist(p, [a[i] + ab[i] * v + ac[i] * w for i in range(3)])


def triangles(s):
    pos, tri = s["pos"], s["tri"]

    def at(i):
        return pos[3 * i : 3 * i + 3]

    return [
        (tri[k + 3], at(tri[k]), at(tri[k + 1]), at(tri[k + 2])) for k in range(0, len(tri), 4)
    ]


def on_mesh(p, tris):
    for _, a, b, c in tris:
        lo = [min(a[i], b[i], c[i]) - FACE_TOL for i in range(3)]
        hi = [max(a[i], b[i], c[i]) + FACE_TOL for i in range(3)]
        if all(lo[i] <= p[i] <= hi[i] for i in range(3)) and tri_dist(p, a, b, c) < FACE_TOL:
            return True
    return False


def fresh_faces(s, prev):
    if prev is None:
        return sorted({f for f, *_ in triangles(s)})
    old = triangles(prev)
    by_face = {}
    for f, a, b, c in triangles(s):
        by_face.setdefault(f, []).append([(a[i] + b[i] + c[i]) / 3 for i in range(3)])
    new = []
    for f, centroids in sorted(by_face.items()):
        sample = centroids[:: max(1, len(centroids) // 24)]
        if any(not on_mesh(p, old) for p in sample):
            new.append(f)
    return new


def main():
    d = json.load(sys.stdin)
    for i, s in enumerate(d["steps"]):
        prev = d["steps"][i - 1]["edges"] if i else []
        s["fresh"] = [0 if all(on(p, prev) for p in probes(e)) else 1 for e in s["edges"]]
        s["fresh_faces"] = fresh_faces(s, d["steps"][i - 1] if i else None)
        print(
            f"{s['label']}: {sum(s['fresh'])} of {len(s['edges'])} edges new, faces "
            f"{s['fresh_faces']} new",
            file=sys.stderr,
        )
    OUT.write_text(json.dumps(d, separators=(",", ":")))


if __name__ == "__main__":
    main()
