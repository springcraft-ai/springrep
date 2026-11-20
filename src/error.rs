use std::fmt;

/// Every kernel failure. An operation that cannot produce a valid solid returns one of these rather
/// than a solid that fails [`crate::check`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The caller's input is meaningless: a zero-length extrusion, an open profile, an axis that
    /// crosses the profile.
    InvalidInput(String),
    /// The input is meaningful but this kernel does not build it yet.
    Unsupported(String),
    /// A numerical step failed: a projection that did not converge, a singular system.
    Numerical(String),
    /// The result would not be a valid closed manifold solid; the strings are the checker's issues.
    InvalidResult(Vec<String>),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Numerical(m) => write!(f, "numerical failure: {m}"),
            Error::InvalidResult(issues) => {
                write!(f, "invalid result: {}", issues.join("; "))
            }
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::InvalidInput(msg.into()))
}
