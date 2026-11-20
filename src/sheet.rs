//! Sheets: open shells of faces with no solid behind them (doc/design.md, "Sheets"). A sheet is
//! made from faces of a solid, turned to face the other way, checked by [`crate::check::check_sheet`]
//! and thickened into a solid.

mod copy;
mod near;
mod thicken;

pub use copy::{flipped, from_faces};
pub use thicken::{Corners, thicken};

#[cfg(test)]
mod tests;
