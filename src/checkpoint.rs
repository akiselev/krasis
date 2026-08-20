use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Untrusted wire representation; [`SimulationState::restore`](crate::SimulationState::restore)
/// validates every field before changing live state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub layout_identity: String,
    pub history_limit: usize,
    pub time: f64,
    pub step: u64,
    pub fields: BTreeMap<String, Vec<f64>>,
    pub field_history: BTreeMap<String, Vec<Vec<f64>>>,
    pub constitutive: BTreeMap<String, Vec<f64>>,
}
