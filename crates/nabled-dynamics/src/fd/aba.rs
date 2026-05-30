//! Articulated Body Algorithm forward dynamics entry point.
//!
//! `ForwardDynamicsMethod::Aba` selects this path. Results are validated against
//! CRBA+LU on serial fixtures; delegates to the proven CRBA+LU bias model until
//! spatial ABA passes are fully audited against RNEA gravity conventions.

use nabled_core::scalar::NabledReal;
use nabled_kinematics::chain::ChainSpec;
use nabled_linalg::lu::LuProviderScalar;
use nabled_model::robot::RobotModel;
use ndarray::{Array1, ArrayView1};

use super::crba_lu::crba_lu_forward_dynamics;
use crate::DynamicsError;
use crate::config::DynamicsConfig;

/// Forward dynamics via the ABA API contract (CRBA+LU validated fallback).
pub fn aba_forward_dynamics<T: NabledReal + Default + LuProviderScalar>(
    model: &RobotModel<T>,
    chain: &ChainSpec<T>,
    q: &ArrayView1<'_, T>,
    qd: &ArrayView1<'_, T>,
    tau: &ArrayView1<'_, T>,
    config: &DynamicsConfig<T>,
) -> Result<Array1<T>, DynamicsError> {
    crba_lu_forward_dynamics(model, chain, q, qd, tau, config)
}
