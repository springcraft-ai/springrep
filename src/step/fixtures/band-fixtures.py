"""Write the band fixtures of step::tests::a_band_on_a_surface_of_revolution_gets_a_seam.

Each is a tube between z = 0 and 10 about the z axis, written entity by entity from the
ISO 10303-42 definitions: an inner wall that is a cylinder, an outer wall that is a band bounded by
its two rim curves and no seam, two plane ends, and optionally windows cut through the wall between
z = 3 and 7, each a sector between two angles about the axis with its four flat sides. Every point
and control point is in closed form. The Python standard library only; run with

    python3 src/step/fixtures/band-fixtures.py

to write them beside this script. The volumes the test expects, in closed form:

- revolution-band*.step: the outer wall is the revolution of the cubic Bezier r = 10, 13, 13, 10
  at z = 10 t, round a cylinder of radius 6. pi (int_0^10 r^2 dz - 6^2 10) = 967 pi, since
  int_0^10 r^2 dz = 1327. A window from a to b radians takes ((b - a) / 2) int_3^7 (r^2 - 36) dz,
  which for a twelfth of a turn is 5557421 pi / 150000: two leave 66967579 pi / 75000, one leaves
  139492579 pi / 150000.
- cylinder-band-windows.step: the same windows through a tube of radii 6 and 10, its outer wall a
  cylinder. pi (10^2 - 6^2) 10 - 2 (pi / 12) (10^2 - 6^2) 4 = 1792 pi / 3.
- extrusion-band*.step: the outer wall is the extrusion along z of the ellipse of semi-axes 10 and
  8, round a cylinder of radius 5: pi (10 * 8 - 5^2) 10 = 550 pi. The window from 3 pi / 4 to
  5 pi / 4 about the axis takes 4 (80 atan(5/4) - 25 pi / 4), since the sector of the ellipse from
  the polar angle pi - p to pi, with tan p = 1, is (10 * 8 / 2) atan((10 / 8) tan p); leaving
  575 pi - 320 atan(5/4).
- extrusion-nurbs-band*.step: the outer wall is the extrusion of a rational quadratic circle of
  radius 10, its parameter running over [2, 3], round a cylinder of radius 5: pi (10^2 - 5^2) 10 =
  750 pi. A window a twelfth of a turn wide takes (pi / 12) (10^2 - 5^2) 4 = 25 pi: 725 pi.

The rims' vertices are at angle 0 unless said, so the seam that a band is given runs at angle pi,
the middle of the widest gap, unless a window is in the way. A window across angle pi
(revolution-band-window-on-seam, extrusion-band-window, extrusion-nurbs-band-window) makes the seam
run clear of it.
"""

import math
import sys
from pathlib import Path


def real(x):
    """A REAL as ISO 10303-21 writes it: a decimal point always, the exponent after an E."""
    text = repr(float(x))
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += "."
        return mantissa + "E" + exponent
    return text


class Step:
    def __init__(self):
        self.lines = []

    def add(self, entity):
        self.lines.append(f"#{len(self.lines) + 1}={entity};")
        return len(self.lines)

    def point(self, p):
        return self.add("CARTESIAN_POINT('',(%s))" % ",".join(real(x) for x in p))

    def direction(self, d):
        n = math.sqrt(sum(x * x for x in d))
        return self.add("DIRECTION('',(%s))" % ",".join(real(x / n) for x in d))

    def placement(self, origin, z=(0, 0, 1), x=(1, 0, 0)):
        o, zd, xd = self.point(origin), self.direction(z), self.direction(x)
        return self.add(f"AXIS2_PLACEMENT_3D('',#{o},#{zd},#{xd})")

    def vertex(self, p):
        return (self.add(f"VERTEX_POINT('',#{self.point(p)})"), tuple(p))

    def circle(self, centre, r, x=(1, 0, 0)):
        return self.add(f"CIRCLE('',#{self.placement(centre, x=x)},{real(r)})")

    def ellipse(self, centre, a, b):
        return self.add(f"ELLIPSE('',#{self.placement(centre)},{real(a)},{real(b)})")

    def line(self, p, q):
        d = [q[i] - p[i] for i in range(3)]
        n = math.sqrt(sum(x * x for x in d))
        start, dd = self.point(p), self.direction(d)
        return self.add(f"LINE('',#{start},#{self.add(f'VECTOR({chr(39) * 2},#{dd},{real(n)})')})")

    def bezier(self, points):
        ids = ",".join(f"#{self.point(p)}" for p in points)
        return self.add(
            f"B_SPLINE_CURVE_WITH_KNOTS('',3,({ids}),.UNSPECIFIED.,.F.,.F.,(4,4),(0.0,1.0),"
            ".UNSPECIFIED.)"
        )

    def nurbs_circle(self, z, r):
        """A rational quadratic circle of radius r about the z axis at height z, from angle 0, in
        four quarters, its parameter running over [2, 3] (Piegl and Tiller, The NURBS Book, 7.3)."""
        w = math.sqrt(0.5)
        corners = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
        ids = ",".join(f"#{self.point((r * x, r * y, z))}" for x, y in corners)
        weights = ",".join(real(x) for x in [1, w, 1, w, 1, w, 1, w, 1])
        return self.add(
            f"( BOUNDED_CURVE() B_SPLINE_CURVE(2,({ids}),.UNSPECIFIED.,.T.,.F.) "
            "B_SPLINE_CURVE_WITH_KNOTS((3,2,2,2,3),(2.0,2.25,2.5,2.75,3.0),.UNSPECIFIED.) "
            f"CURVE() GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE(({weights})) "
            "REPRESENTATION_ITEM('') )"
        )

    def edge(self, v0, v1, curve):
        return self.add(f"EDGE_CURVE('',#{v0[0]},#{v1[0]},#{curve},.T.)")

    def loop(self, uses):
        oriented = [
            self.add(f"ORIENTED_EDGE('',*,*,#{e},{'.T.' if same else '.F.'})") for e, same in uses
        ]
        return self.add("EDGE_LOOP('',(%s))" % ",".join(f"#{o}" for o in oriented))

    def face(self, surface, same_sense, bounds):
        ids = []
        for outer, uses in bounds:
            kind = "FACE_OUTER_BOUND" if outer else "FACE_BOUND"
            ids.append(self.add(f"{kind}('',#{self.loop(uses)},.T.)"))
        sense = ".T." if same_sense else ".F."
        return self.add(
            "ADVANCED_FACE('',(%s),#%d,%s)" % (",".join(f"#{b}" for b in ids), surface, sense)
        )

    def plane(self, origin, z, x=(1, 0, 0)):
        return self.add(f"PLANE('',#{self.placement(origin, z, x)})")

    def text(self, faces, name):
        shell = self.add("CLOSED_SHELL('',(%s))" % ",".join(f"#{f}" for f in faces))
        brep = self.add(f"MANIFOLD_SOLID_BREP('',#{shell})")
        length = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
        angle = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        solid = self.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        accuracy = self.add(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-7),#{length},"
            "'distance_accuracy_value','')"
        )
        context = self.add(
            f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{accuracy})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid})) REPRESENTATION_CONTEXT('',''))"
        )
        app = self.add("APPLICATION_CONTEXT('automotive design')")
        self.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
        product_context = self.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
        product = self.add(f"PRODUCT('{name}','{name}','',(#{product_context}))")
        formation = self.add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
        definition_context = self.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
        definition = self.add(f"PRODUCT_DEFINITION('design','',#{formation},#{definition_context})")
        shape = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
        rep = self.add(
            f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{self.placement((0, 0, 0))},#{brep}),"
            f"#{context})"
        )
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{rep})")
        header = (
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('springrep test fixture'),'2;1');\n"
            f"FILE_NAME('{name}.step','2026-11-24T00:00:00',(''),(''),'','','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n"
        )
        return header + "\n".join(self.lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"


PROFILE = [(10.0, 0.0), (13.0, 10.0 / 3.0), (13.0, 20.0 / 3.0), (10.0, 10.0)]


def profile_radius(z):
    t = z / 10.0
    r = [p[0] for p in PROFILE]
    return r[0] * (1 - t) ** 3 + 3 * r[1] * t * (1 - t) ** 2 + 3 * r[2] * t * t * (1 - t) + r[3] * t**3


def at(r, phi, z):
    return (r * math.cos(phi), r * math.sin(phi), z)


def tube(name, outer, inner_radius, two_outer=False, top_turn=0.0, windows=()):
    """outer is "revolution", "cylinder", "ellipse" or "nurbs-circle"; windows are (a, b) in
    radians about the axis, each cut between z = 3 and 7."""
    s = Step()
    if outer == "revolution":
        profile = s.bezier([(r, 0.0, z) for r, z in PROFILE])
        axis = s.add(f"AXIS1_PLACEMENT('',#{s.point((0, 0, 0))},#{s.direction((0, 0, 1))})")
        wall = s.add(f"SURFACE_OF_REVOLUTION('',#{profile},#{axis})")
        wall_point = lambda phi, z: at(profile_radius(z), phi, z)  # noqa: E731
        wall_arc = lambda z: s.circle((0, 0, z), profile_radius(z))  # noqa: E731
        wall_line = lambda phi, v0, v1: s.bezier([at(r, phi, z) for r, z in PROFILE])  # noqa: E731
    elif outer == "cylinder":
        wall = s.add(f"CYLINDRICAL_SURFACE('',#{s.placement((0, 0, 0))},10.0)")
        wall_point = lambda phi, z: at(10.0, phi, z)  # noqa: E731
        wall_arc = lambda z: s.circle((0, 0, z), 10.0)  # noqa: E731
        wall_line = lambda phi, v0, v1: s.line(v0[1], v1[1])  # noqa: E731
    elif outer == "ellipse":
        a, b = 10.0, 8.0
        curve = s.ellipse((0, 0, 0), a, b)
        wall = s.add(
            f"SURFACE_OF_LINEAR_EXTRUSION('',#{curve},"
            f"#{s.add(f'VECTOR({chr(39) * 2},#{s.direction((0, 0, 1))},1.0)')})"
        )
        # The ellipse's point at polar angle phi, at distance a b / sqrt((b cos)^2 + (a sin)^2).
        polar = lambda phi: a * b / math.hypot(b * math.cos(phi), a * math.sin(phi))  # noqa: E731
        wall_point = lambda phi, z: at(polar(phi), phi, z)  # noqa: E731
        wall_arc = lambda z: s.ellipse((0, 0, z), a, b)  # noqa: E731
        wall_line = lambda phi, v0, v1: s.line(v0[1], v1[1])  # noqa: E731
    elif outer == "nurbs-circle":
        curve = s.nurbs_circle(0.0, 10.0)
        wall = s.add(
            f"SURFACE_OF_LINEAR_EXTRUSION('',#{curve},"
            f"#{s.add(f'VECTOR({chr(39) * 2},#{s.direction((0, 0, 1))},1.0)')})"
        )
        wall_point = lambda phi, z: at(10.0, phi, z)  # noqa: E731
        wall_arc = lambda z: s.circle((0, 0, z), 10.0)  # noqa: E731
        wall_line = lambda phi, v0, v1: s.line(v0[1], v1[1])  # noqa: E731
    else:
        raise ValueError(outer)

    def rim(z, turn):
        v = s.vertex(wall_point(turn, z))
        if outer == "ellipse":
            return s.edge(v, v, s.ellipse((0, 0, z), 10.0, 8.0))
        if outer == "nurbs-circle":
            return s.edge(v, v, s.nurbs_circle(z, 10.0))
        x = (math.cos(turn), math.sin(turn), 0.0)
        return s.edge(v, v, s.circle((0, 0, z), 10.0, x))

    inner = s.add(f"CYLINDRICAL_SURFACE('',#{s.placement((0, 0, 0))},{real(inner_radius)})")
    outer_bottom = rim(0.0, 0.0)
    outer_top = rim(10.0, top_turn)
    v = s.vertex((inner_radius, 0.0, 0.0))
    inner_bottom = s.edge(v, v, s.circle((0, 0, 0), inner_radius))
    v = s.vertex((inner_radius, 0.0, 10.0))
    inner_top = s.edge(v, v, s.circle((0, 0, 10), inner_radius))

    wall_bounds = [(True, [(outer_bottom, True)]), (two_outer, [(outer_top, False)])]
    inner_bounds = [(True, [(inner_bottom, False)]), (False, [(inner_top, True)])]
    faces = []
    for p1, p2 in windows:
        z1, z2 = 3.0, 7.0
        o = [[s.vertex(wall_point(p, z)) for z in (z1, z2)] for p in (p1, p2)]
        i = [[s.vertex(at(inner_radius, p, z)) for z in (z1, z2)] for p in (p1, p2)]
        o_arc = [s.edge(o[0][k], o[1][k], wall_arc((z1, z2)[k])) for k in (0, 1)]
        i_arc = [s.edge(i[0][k], i[1][k], s.circle((0, 0, (z1, z2)[k]), inner_radius)) for k in (0, 1)]
        o_side = [s.edge(o[j][0], o[j][1], wall_line((p1, p2)[j], o[j][0], o[j][1])) for j in (0, 1)]
        i_side = [s.edge(i[j][0], i[j][1], s.line(i[j][0][1], i[j][1][1])) for j in (0, 1)]
        radial = [[s.edge(i[j][k], o[j][k], s.line(i[j][k][1], o[j][k][1])) for k in (0, 1)] for j in (0, 1)]
        wall_bounds.append(
            (False, [(o_arc[0], False), (o_side[0], True), (o_arc[1], True), (o_side[1], False)])
        )
        inner_bounds.append(
            (False, [(i_arc[0], True), (i_side[1], True), (i_arc[1], False), (i_side[0], False)])
        )
        for j, p in enumerate((p1, p2)):
            normal = (-math.sin(p), math.cos(p), 0.0) if j == 0 else (math.sin(p), -math.cos(p), 0.0)
            side = s.plane((0, 0, 0), normal, (math.cos(p), math.sin(p), 0.0))
            if j == 0:
                uses = [(radial[0][0], False), (i_side[0], True), (radial[0][1], True), (o_side[0], False)]
            else:
                uses = [(radial[1][0], True), (o_side[1], True), (radial[1][1], False), (i_side[1], False)]
            faces.append(s.face(side, True, [(True, uses)]))
        sill = s.plane((0, 0, z1), (0, 0, 1))
        faces.append(
            s.face(sill, True, [(True, [(o_arc[0], True), (radial[1][0], False), (i_arc[0], False), (radial[0][0], True)])])
        )
        lintel = s.plane((0, 0, z2), (0, 0, -1))
        faces.append(
            s.face(lintel, True, [(True, [(o_arc[1], False), (radial[0][1], False), (i_arc[1], True), (radial[1][1], True)])])
        )
    faces.append(s.face(wall, True, wall_bounds))
    faces.append(s.face(inner, False, inner_bounds))
    bottom = s.plane((0, 0, 0), (0, 0, -1))
    faces.append(s.face(bottom, True, [(True, [(outer_bottom, False)]), (False, [(inner_bottom, True)])]))
    top = s.plane((0, 0, 10), (0, 0, 1))
    faces.append(s.face(top, True, [(True, [(outer_top, True)]), (False, [(inner_top, False)])]))
    return s.text(faces, name.replace("-", "_"))


def main():
    here = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    d = math.radians
    two_windows = [(d(30), d(60)), (d(210), d(240))]
    fixtures = {
        "revolution-band": tube("revolution-band", "revolution", 6.0),
        "revolution-band-two-outer": tube("revolution-band-two-outer", "revolution", 6.0, two_outer=True),
        "revolution-band-apart": tube("revolution-band-apart", "revolution", 6.0, top_turn=math.pi / 2),
        "revolution-band-windows": tube(
            "revolution-band-windows", "revolution", 6.0, two_outer=True, windows=two_windows
        ),
        "revolution-band-window-on-seam": tube(
            "revolution-band-window-on-seam", "revolution", 6.0, two_outer=True, windows=[(d(160), d(190))]
        ),
        "cylinder-band-windows": tube(
            "cylinder-band-windows", "cylinder", 6.0, two_outer=True, windows=two_windows
        ),
        "extrusion-band": tube("extrusion-band", "ellipse", 5.0),
        "extrusion-band-window": tube(
            "extrusion-band-window", "ellipse", 5.0, windows=[(3 * math.pi / 4, 5 * math.pi / 4)]
        ),
        "extrusion-nurbs-band": tube("extrusion-nurbs-band", "nurbs-circle", 5.0),
        "extrusion-nurbs-band-window": tube(
            "extrusion-nurbs-band-window", "nurbs-circle", 5.0, windows=[(d(170), d(200))]
        ),
    }
    for name, text in fixtures.items():
        (here / f"{name}.step").write_text(text)


if __name__ == "__main__":
    main()
