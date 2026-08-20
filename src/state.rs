use crate::{BlockLayout, Checkpoint, KrasisError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FieldId(String);

impl FieldId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionPhase {
    Committed,
    Trial,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct FieldSlot {
    committed: Vec<f64>,
    trial: Option<Vec<f64>>,
    history: VecDeque<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstitutiveSlot {
    pub committed: Vec<f64>,
    trial: Option<Vec<f64>>,
}

impl ConstitutiveSlot {
    pub fn new(committed: Vec<f64>) -> Self {
        Self {
            committed,
            trial: None,
        }
    }

    pub fn trial(&self) -> Option<&[f64]> {
        self.trial.as_deref()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationState {
    phase: TransactionPhase,
    history_limit: usize,
    time: f64,
    step: u64,
    fields: BTreeMap<FieldId, FieldSlot>,
    constitutive: BTreeMap<String, ConstitutiveSlot>,
}

impl SimulationState {
    pub fn new(history_limit: usize) -> Self {
        Self {
            phase: TransactionPhase::Committed,
            history_limit,
            time: 0.0,
            step: 0,
            fields: BTreeMap::new(),
            constitutive: BTreeMap::new(),
        }
    }

    pub fn phase(&self) -> TransactionPhase {
        self.phase
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn insert_field(&mut self, id: FieldId, values: Vec<f64>) -> Result<(), KrasisError> {
        if self.fields.contains_key(&id) {
            return Err(KrasisError::DuplicateField(id.to_string()));
        }
        self.fields.insert(
            id,
            FieldSlot {
                committed: values,
                trial: None,
                history: VecDeque::new(),
            },
        );
        Ok(())
    }

    pub fn insert_constitutive(&mut self, id: impl Into<String>, values: Vec<f64>) {
        self.constitutive
            .insert(id.into(), ConstitutiveSlot::new(values));
    }

    pub fn committed(&self, id: &FieldId) -> Result<&[f64], KrasisError> {
        self.fields
            .get(id)
            .map(|slot| slot.committed.as_slice())
            .ok_or_else(|| KrasisError::UnknownField(id.to_string()))
    }

    pub fn trial(&self, id: &FieldId) -> Result<&[f64], KrasisError> {
        let slot = self
            .fields
            .get(id)
            .ok_or_else(|| KrasisError::UnknownField(id.to_string()))?;
        slot.trial.as_deref().ok_or(KrasisError::NoActiveTrial)
    }

    pub fn begin_trial(&mut self) -> Result<(), KrasisError> {
        if self.phase == TransactionPhase::Trial {
            return Err(KrasisError::TrialAlreadyActive);
        }
        for slot in self.fields.values_mut() {
            slot.trial = Some(slot.committed.clone());
        }
        for slot in self.constitutive.values_mut() {
            slot.trial = Some(slot.committed.clone());
        }
        self.phase = TransactionPhase::Trial;
        Ok(())
    }

    pub fn set_trial(&mut self, id: &FieldId, values: &[f64]) -> Result<(), KrasisError> {
        if self.phase != TransactionPhase::Trial {
            return Err(KrasisError::NoActiveTrial);
        }
        let slot = self
            .fields
            .get_mut(id)
            .ok_or_else(|| KrasisError::UnknownField(id.to_string()))?;
        if values.len() != slot.committed.len() {
            return Err(KrasisError::FieldLength {
                field: id.to_string(),
                actual: values.len(),
                expected: slot.committed.len(),
            });
        }
        slot.trial = Some(values.to_vec());
        Ok(())
    }

    pub fn set_constitutive_trial(&mut self, id: &str, values: &[f64]) -> Result<(), KrasisError> {
        if self.phase != TransactionPhase::Trial {
            return Err(KrasisError::NoActiveTrial);
        }
        let slot = self
            .constitutive
            .get_mut(id)
            .ok_or_else(|| KrasisError::UnknownField(id.to_owned()))?;
        if values.len() != slot.committed.len() {
            return Err(KrasisError::FieldLength {
                field: id.to_owned(),
                actual: values.len(),
                expected: slot.committed.len(),
            });
        }
        slot.trial = Some(values.to_vec());
        Ok(())
    }

    pub fn commit(&mut self, time: f64) -> Result<(), KrasisError> {
        if self.phase != TransactionPhase::Trial {
            return Err(KrasisError::NoActiveTrial);
        }
        for slot in self.fields.values_mut() {
            if self.history_limit > 0 {
                slot.history.push_front(slot.committed.clone());
                slot.history.truncate(self.history_limit);
            }
            slot.committed = slot.trial.take().ok_or(KrasisError::NoActiveTrial)?;
        }
        for slot in self.constitutive.values_mut() {
            slot.committed = slot.trial.take().ok_or(KrasisError::NoActiveTrial)?;
        }
        self.time = time;
        self.step += 1;
        self.phase = TransactionPhase::Committed;
        Ok(())
    }

    pub fn rollback(&mut self) -> Result<(), KrasisError> {
        if self.phase != TransactionPhase::Trial {
            return Err(KrasisError::NoActiveTrial);
        }
        for slot in self.fields.values_mut() {
            slot.trial = None;
        }
        for slot in self.constitutive.values_mut() {
            slot.trial = None;
        }
        self.phase = TransactionPhase::Committed;
        Ok(())
    }

    pub fn checkpoint(&self, layout: &BlockLayout) -> Result<Checkpoint, KrasisError> {
        if self.phase != TransactionPhase::Committed {
            return Err(KrasisError::TrialAlreadyActive);
        }
        Ok(Checkpoint {
            layout_identity: layout.identity().to_owned(),
            time: self.time,
            step: self.step,
            fields: self
                .fields
                .iter()
                .map(|(id, slot)| (id.to_string(), slot.committed.clone()))
                .collect(),
            constitutive: self
                .constitutive
                .iter()
                .map(|(id, slot)| (id.clone(), slot.committed.clone()))
                .collect(),
        })
    }

    pub fn restore(
        &mut self,
        layout: &BlockLayout,
        checkpoint: &Checkpoint,
    ) -> Result<(), KrasisError> {
        if checkpoint.layout_identity != layout.identity() {
            return Err(KrasisError::LayoutMismatch {
                actual: checkpoint.layout_identity.clone(),
                expected: layout.identity().to_owned(),
            });
        }
        for (id, slot) in &mut self.fields {
            let values = checkpoint.fields.get(id.as_str()).ok_or_else(|| {
                KrasisError::MalformedCheckpoint(format!("field `{id}` is missing"))
            })?;
            if values.len() != slot.committed.len() {
                return Err(KrasisError::FieldLength {
                    field: id.to_string(),
                    actual: values.len(),
                    expected: slot.committed.len(),
                });
            }
            slot.committed.clone_from(values);
            slot.trial = None;
            slot.history.clear();
        }
        for (id, slot) in &mut self.constitutive {
            let values = checkpoint.constitutive.get(id).ok_or_else(|| {
                KrasisError::MalformedCheckpoint(format!("constitutive slot `{id}` is missing"))
            })?;
            if values.len() != slot.committed.len() {
                return Err(KrasisError::FieldLength {
                    field: id.clone(),
                    actual: values.len(),
                    expected: slot.committed.len(),
                });
            }
            slot.committed.clone_from(values);
            slot.trial = None;
        }
        self.time = checkpoint.time;
        self.step = checkpoint.step;
        self.phase = TransactionPhase::Committed;
        Ok(())
    }
}
