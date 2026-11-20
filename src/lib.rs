#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(unreachable_pub)]

pub mod blend;
pub mod boolean;
pub mod check;
pub mod codec;
pub mod distance;
pub mod draw;
pub mod edit;
mod error;
pub mod geom;
pub mod loft;
pub mod math;
pub mod measure;
pub mod mesh;
mod numeric;
pub mod overlap;
pub mod pick;
pub mod primitive;
pub mod probe;
pub mod profile;
pub mod rays;
pub mod sew;
pub mod sheet;
pub mod step;
pub mod sweep;
pub mod tol;
pub mod topo;

pub use error::{Error, Result};

/// A digest of the kernel's sources, unit tests aside: two builds with the same one build the same
/// geometry, so a cache of built parts keys on it.
pub const SOURCE: &str = env!("SPRINGREP_SOURCE");

/// The release this build is, as published: what an adapter reports as the kernel's version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
