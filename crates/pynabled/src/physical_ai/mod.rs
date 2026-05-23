//! Physical AI bindings for Python (geometry, kinematics, model, dynamics, control, sensor,
//! signal).

#![allow(
    clippy::missing_errors_doc,
    missing_copy_implementations,
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::match_wildcard_for_single_variants,
    clippy::elidable_lifetime_names,
    clippy::clone_on_copy,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    unnameable_types,
    elided_lifetimes_in_paths,
    redundant_lifetimes,
    single_use_lifetimes,
    unused_lifetimes,
    deprecated
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
