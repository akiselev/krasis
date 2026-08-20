use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub layout_identity: String,
    pub time: f64,
    pub step: u64,
    pub fields: BTreeMap<String, Vec<f64>>,
    pub constitutive: BTreeMap<String, Vec<f64>>,
}
