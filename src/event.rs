use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventDirection {
    Rising,
    Falling,
    Either,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventRecord {
    pub id: String,
    pub time: f64,
    pub value_before: f64,
    pub value_after: f64,
    pub direction: EventDirection,
}
