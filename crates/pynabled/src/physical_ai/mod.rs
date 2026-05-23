//! Physical AI bindings for Python (geometry, kinematics, model, dynamics, control, sensor, signal).

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_copy_implementations,
    clippy::doc_markdown,
    clippy::must_use_candidate
)]

pub(crate) mod common;
pub(crate) mod control;
pub(crate) mod dynamics;
pub(crate) mod geometry;
pub(crate) mod kinematics;
pub(crate) mod model;
pub(crate) mod sensor;

#[cfg(feature = "signal")]
pub(crate) mod signal;
