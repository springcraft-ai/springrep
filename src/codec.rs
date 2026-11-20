//! A compact binary encoding of built geometry, for build states kept on disk. Numbers are written
//! as their bits, so a decoded solid is the encoded one exactly. A struct is encoded field by field
//! through `fields!`, which names every field without `..`: a field added to the struct and not
//! to its encoding does not compile.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::{
    geom::{
        BSpline, Circle, Cone, Curve, Cylinder, Ellipse, Extrusion, IntersectionCurve, Line,
        NurbsCurve, NurbsSurface, Plane, Revolution, Ruled, Sphere, Surface, Torus,
    },
    math::{Aabb, Frame, Vec2, Vec3},
    topo::{Brep, FaceMemo},
};

/// The solids as bytes that read back as the same solids to the last bit. Faces that share a memo
/// share it again when read. The bytes end in a digest of the rest.
pub fn to_bytes(solids: &[Brep]) -> Vec<u8> {
    let mut w = Writer::default();
    solids.len().put(&mut w);
    for s in solids {
        s.put(&mut w);
    }
    let sum = digest(&w.0);
    sum.put(&mut w);
    w.0
}

/// The solids [`to_bytes`] wrote into `bytes`: `None` for any other bytes, a damaged or shortened
/// copy among them. Bytes from another build of the kernel may read as nothing.
pub fn from_bytes(bytes: &[u8]) -> Option<Vec<Brep>> {
    let (body, sum) = bytes.split_at_checked(bytes.len().checked_sub(8)?)?;
    (u64::from_le_bytes(sum.try_into().ok()?) == digest(body)).then_some(())?;
    let mut r = Reader::new(body);
    let solids = Vec::<Brep>::get(&mut r)?;
    (r.remaining() == 0).then_some(solids)
}

/// The bytes written so far, and the face memos among them by where each lives, so faces that
/// share a memo share it again when read back.
#[derive(Default)]
pub(crate) struct Writer(pub Vec<u8>, pub HashMap<usize, u32>);

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pub memos: Vec<Arc<FaceMemo>>,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader {
            bytes,
            memos: Vec::new(),
        }
    }

    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.bytes.len() {
            return None;
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Some(head)
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len()
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }
}

/// A value written to and read back from bytes. `get` is `None` for bytes no `put` wrote.
pub(crate) trait Codec: Sized {
    fn put(&self, w: &mut Writer);
    fn get(r: &mut Reader) -> Option<Self>;
}

/// [`Codec`] for a struct, field by field.
macro_rules! fields {
    ($t:ident { $($f:ident),* $(,)? }) => {
        impl $crate::codec::Codec for $t {
            fn put(&self, w: &mut $crate::codec::Writer) {
                let $t { $($f),* } = self;
                $($crate::codec::Codec::put($f, w);)*
            }
            fn get(r: &mut $crate::codec::Reader) -> Option<$t> {
                Some($t { $($f: $crate::codec::Codec::get(r)?),* })
            }
        }
    };
}
pub(crate) use fields;

macro_rules! number {
    ($($t:ty),*) => {$(
        impl Codec for $t {
            fn put(&self, w: &mut Writer) {
                w.0.extend_from_slice(&self.to_le_bytes());
            }
            fn get(r: &mut Reader) -> Option<$t> {
                Some(<$t>::from_le_bytes(r.array()?))
            }
        }
    )*};
}
number!(u8, u32, u64, u128);

impl Codec for f64 {
    fn put(&self, w: &mut Writer) {
        self.to_bits().put(w);
    }
    fn get(r: &mut Reader) -> Option<f64> {
        Some(f64::from_bits(u64::get(r)?))
    }
}

impl Codec for usize {
    fn put(&self, w: &mut Writer) {
        (*self as u64).put(w);
    }
    fn get(r: &mut Reader) -> Option<usize> {
        usize::try_from(u64::get(r)?).ok()
    }
}

impl Codec for bool {
    fn put(&self, w: &mut Writer) {
        u8::from(*self).put(w);
    }
    fn get(r: &mut Reader) -> Option<bool> {
        match u8::get(r)? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

impl Codec for String {
    fn put(&self, w: &mut Writer) {
        self.len().put(w);
        w.0.extend_from_slice(self.as_bytes());
    }
    fn get(r: &mut Reader) -> Option<String> {
        let n = usize::get(r)?;
        String::from_utf8(r.take(n)?.to_vec()).ok()
    }
}

/// A length read back, refused when the bytes left could not hold that many items: a damaged
/// length must not allocate without bound.
fn length(r: &mut Reader) -> Option<usize> {
    let n = usize::get(r)?;
    (n <= r.remaining()).then_some(n)
}

impl<T: Codec> Codec for Vec<T> {
    fn put(&self, w: &mut Writer) {
        self.len().put(w);
        for x in self {
            x.put(w);
        }
    }
    fn get(r: &mut Reader) -> Option<Vec<T>> {
        let n = length(r)?;
        (0..n).map(|_| T::get(r)).collect()
    }
}

impl<T: Codec> Codec for Option<T> {
    fn put(&self, w: &mut Writer) {
        match self {
            None => 0u8.put(w),
            Some(x) => {
                1u8.put(w);
                x.put(w);
            }
        }
    }
    fn get(r: &mut Reader) -> Option<Option<T>> {
        match u8::get(r)? {
            0 => Some(None),
            1 => Some(Some(T::get(r)?)),
            _ => None,
        }
    }
}

impl<T: Codec> Codec for Box<T> {
    fn put(&self, w: &mut Writer) {
        (**self).put(w);
    }
    fn get(r: &mut Reader) -> Option<Box<T>> {
        T::get(r).map(Box::new)
    }
}

impl<A: Codec, B: Codec> Codec for (A, B) {
    fn put(&self, w: &mut Writer) {
        self.0.put(w);
        self.1.put(w);
    }
    fn get(r: &mut Reader) -> Option<(A, B)> {
        Some((A::get(r)?, B::get(r)?))
    }
}

impl<T: Codec> Codec for [T; 2] {
    fn put(&self, w: &mut Writer) {
        self[0].put(w);
        self[1].put(w);
    }
    fn get(r: &mut Reader) -> Option<[T; 2]> {
        Some([T::get(r)?, T::get(r)?])
    }
}

impl<T: Codec + Ord> Codec for BTreeSet<T> {
    fn put(&self, w: &mut Writer) {
        self.len().put(w);
        for x in self {
            x.put(w);
        }
    }
    fn get(r: &mut Reader) -> Option<BTreeSet<T>> {
        let n = length(r)?;
        (0..n).map(|_| T::get(r)).collect()
    }
}

impl<K: Codec + Ord, V: Codec> Codec for BTreeMap<K, V> {
    fn put(&self, w: &mut Writer) {
        self.len().put(w);
        for (k, v) in self {
            k.put(w);
            v.put(w);
        }
    }
    fn get(r: &mut Reader) -> Option<BTreeMap<K, V>> {
        let n = length(r)?;
        (0..n).map(|_| Some((K::get(r)?, V::get(r)?))).collect()
    }
}

fields!(Vec2 { x, y });
fields!(Vec3 { x, y, z });
fields!(Frame { origin, x, y, z });
fields!(Aabb { min, max });
fields!(Line { origin, dir });
fields!(Circle { frame, radius });
fields!(Ellipse {
    frame,
    major,
    minor
});
fields!(IntersectionCurve { approx, surfaces });
fields!(Plane { frame });
fields!(Cylinder { frame, radius });
fields!(Cone {
    frame,
    radius,
    half_angle
});
fields!(Sphere { frame, radius });
fields!(Torus {
    frame,
    major,
    minor
});
fields!(Extrusion { curve, dir });
fields!(Revolution {
    curve,
    origin,
    axis
});
fields!(Ruled {
    a,
    a_range,
    b,
    b_range
});

impl Codec for BSpline {
    fn put(&self, w: &mut Writer) {
        self.degree().put(w);
        self.knots().to_vec().put(w);
        self.control_points().to_vec().put(w);
        self.is_periodic().put(w);
    }
    fn get(r: &mut Reader) -> Option<BSpline> {
        let degree = usize::get(r)?;
        let knots = Vec::get(r)?;
        let ctrl = Vec::get(r)?;
        if bool::get(r)? {
            BSpline::periodic(degree, knots, ctrl).ok()
        } else {
            BSpline::new(degree, knots, ctrl).ok()
        }
    }
}

impl Codec for NurbsCurve {
    fn put(&self, w: &mut Writer) {
        self.degree().put(w);
        self.knots().to_vec().put(w);
        self.control_points().to_vec().put(w);
        self.weights().map(<[f64]>::to_vec).put(w);
        self.period().put(w);
    }
    fn get(r: &mut Reader) -> Option<NurbsCurve> {
        NurbsCurve::restore(
            Codec::get(r)?,
            Codec::get(r)?,
            Codec::get(r)?,
            Codec::get(r)?,
            Codec::get(r)?,
        )
        .ok()
    }
}

impl Codec for NurbsSurface {
    fn put(&self, w: &mut Writer) {
        self.degrees().put(w);
        let (ku, kv) = self.knots();
        (ku.to_vec(), kv.to_vec()).put(w);
        self.counts().put(w);
        self.control_points().to_vec().put(w);
        self.weights().map(<[f64]>::to_vec).put(w);
        (self.u_period(), self.v_period()).put(w);
    }
    fn get(r: &mut Reader) -> Option<NurbsSurface> {
        NurbsSurface::restore(
            Codec::get(r)?,
            Codec::get(r)?,
            Codec::get(r)?,
            Codec::get(r)?,
            Codec::get(r)?,
            Codec::get(r)?,
        )
        .ok()
    }
}

impl Codec for Curve {
    fn put(&self, w: &mut Writer) {
        match self {
            Curve::Line(c) => (0u8, c).put_pair(w),
            Curve::Circle(c) => (1u8, c).put_pair(w),
            Curve::Ellipse(c) => (2u8, c).put_pair(w),
            Curve::BSpline(c) => (3u8, c).put_pair(w),
            Curve::Intersection(c) => (4u8, c).put_pair(w),
            Curve::Nurbs(c) => (5u8, c).put_pair(w),
        }
    }
    fn get(r: &mut Reader) -> Option<Curve> {
        Some(match u8::get(r)? {
            0 => Curve::Line(Codec::get(r)?),
            1 => Curve::Circle(Codec::get(r)?),
            2 => Curve::Ellipse(Codec::get(r)?),
            3 => Curve::BSpline(Codec::get(r)?),
            4 => Curve::Intersection(Codec::get(r)?),
            5 => Curve::Nurbs(Codec::get(r)?),
            _ => return None,
        })
    }
}

impl Codec for Surface {
    fn put(&self, w: &mut Writer) {
        match self {
            Surface::Plane(s) => (0u8, s).put_pair(w),
            Surface::Cylinder(s) => (1u8, s).put_pair(w),
            Surface::Cone(s) => (2u8, s).put_pair(w),
            Surface::Sphere(s) => (3u8, s).put_pair(w),
            Surface::Torus(s) => (4u8, s).put_pair(w),
            Surface::Extrusion(s) => (5u8, s).put_pair(w),
            Surface::Revolution(s) => (6u8, s).put_pair(w),
            Surface::Ruled(s) => (7u8, s).put_pair(w),
            Surface::Nurbs(s) => (8u8, s).put_pair(w),
        }
    }
    fn get(r: &mut Reader) -> Option<Surface> {
        Some(match u8::get(r)? {
            0 => Surface::Plane(Codec::get(r)?),
            1 => Surface::Cylinder(Codec::get(r)?),
            2 => Surface::Cone(Codec::get(r)?),
            3 => Surface::Sphere(Codec::get(r)?),
            4 => Surface::Torus(Codec::get(r)?),
            5 => Surface::Extrusion(Codec::get(r)?),
            6 => Surface::Revolution(Codec::get(r)?),
            7 => Surface::Ruled(Codec::get(r)?),
            8 => Surface::Nurbs(Codec::get(r)?),
            _ => return None,
        })
    }
}

/// A variant's tag, then its value.
trait PutPair {
    fn put_pair(self, w: &mut Writer);
}

impl<T: Codec> PutPair for (u8, &T) {
    fn put_pair(self, w: &mut Writer) {
        self.0.put(w);
        self.1.put(w);
    }
}

/// A 64-bit digest of `bytes`, eight at a time: it finds a damaged or truncated file, nothing more.
pub(crate) fn digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ bytes.len() as u64;
    let (words, rest) = bytes.as_chunks::<8>();
    for &c in words {
        h = (h ^ u64::from_le_bytes(c))
            .wrapping_mul(0x0000_0100_0000_01b3)
            .rotate_left(29);
    }
    for &b in rest {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests;
