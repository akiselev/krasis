use crate::{Checkpoint, KrasisError, StateLayout};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TransactionPhase {
    Committed,
    Trial,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct FieldSlot {
    committed: Vec<f64>,
    trial: Option<Vec<f64>>,
    history: VecDeque<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConstitutiveSlot {
    committed: Vec<f64>,
    trial: Option<Vec<f64>>,
}

impl ConstitutiveSlot {
    fn new(committed: Vec<f64>) -> Self {
        Self {
            committed,
            trial: None,
        }
    }

    pub fn committed(&self) -> &[f64] {
        &self.committed
    }

    pub fn trial(&self) -> Option<&[f64]> {
        self.trial.as_deref()
    }
}

/// Layout-bound coupled state with atomic trial, commit, rollback, and restore.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SimulationState {
    layout: StateLayout,
    phase: TransactionPhase,
    history_limit: usize,
    time: f64,
    step: u64,
    fields: BTreeMap<FieldId, FieldSlot>,
    constitutive: BTreeMap<String, ConstitutiveSlot>,
}

impl SimulationState {
    pub fn new(layout: StateLayout, history_limit: usize) -> Self {
        Self {
            layout,
            phase: TransactionPhase::Committed,
            history_limit,
            time: 0.0,
            step: 0,
            fields: BTreeMap::new(),
            constitutive: BTreeMap::new(),
        }
    }

    pub fn layout(&self) -> &StateLayout {
        &self.layout
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
        self.require_committed_structure()?;
        if self.fields.contains_key(&id) {
            return Err(KrasisError::DuplicateField(id.to_string()));
        }
        let block = self
            .layout
            .block(&crate::BlockId::new(id.as_str()))
            .ok_or_else(|| KrasisError::FieldOutsideLayout(id.to_string()))?;
        let expected = block.range().len();
        if values.len() != expected {
            return Err(KrasisError::FieldLength {
                field: id.to_string(),
                actual: values.len(),
                expected,
            });
        }
        require_finite(&format!("field `{id}`"), &values)?;
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

    pub fn insert_constitutive(
        &mut self,
        id: impl Into<String>,
        values: Vec<f64>,
    ) -> Result<(), KrasisError> {
        self.require_committed_structure()?;
        let id = id.into();
        if self.constitutive.contains_key(&id) {
            return Err(KrasisError::DuplicateConstitutive(id));
        }
        require_finite(&format!("constitutive slot `{id}`"), &values)?;
        self.constitutive.insert(id, ConstitutiveSlot::new(values));
        Ok(())
    }

    pub fn committed(&self, id: &FieldId) -> Result<&[f64], KrasisError> {
        self.fields
            .get(id)
            .map(|slot| slot.committed.as_slice())
            .ok_or_else(|| KrasisError::UnknownField(id.to_string()))
    }

    pub fn history(&self, id: &FieldId) -> Result<&VecDeque<Vec<f64>>, KrasisError> {
        self.fields
            .get(id)
            .map(|slot| &slot.history)
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
        self.ensure_complete()?;
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
        require_finite(&format!("trial field `{id}`"), values)?;
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
            .ok_or_else(|| KrasisError::UnknownConstitutive(id.to_owned()))?;
        if values.len() != slot.committed.len() {
            return Err(KrasisError::FieldLength {
                field: id.to_owned(),
                actual: values.len(),
                expected: slot.committed.len(),
            });
        }
        require_finite(&format!("trial constitutive slot `{id}`"), values)?;
        slot.trial = Some(values.to_vec());
        Ok(())
    }

    pub fn commit(&mut self, time: f64) -> Result<(), KrasisError> {
        if self.phase != TransactionPhase::Trial {
            return Err(KrasisError::NoActiveTrial);
        }
        if !time.is_finite() || time < self.time {
            return Err(KrasisError::InvalidCommitTime {
                time,
                current: self.time,
            });
        }

        let field_trials = self
            .fields
            .iter()
            .map(|(id, slot)| {
                slot.trial
                    .clone()
                    .map(|values| (id.clone(), values))
                    .ok_or(KrasisError::NoActiveTrial)
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let constitutive_trials = self
            .constitutive
            .iter()
            .map(|(id, slot)| {
                slot.trial
                    .clone()
                    .map(|values| (id.clone(), values))
                    .ok_or(KrasisError::NoActiveTrial)
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let next_step = self
            .step
            .checked_add(1)
            .ok_or_else(|| KrasisError::MalformedCheckpoint("step counter overflowed".into()))?;

        for (id, slot) in &mut self.fields {
            if self.history_limit > 0 {
                slot.history.push_front(slot.committed.clone());
                slot.history.truncate(self.history_limit);
            }
            slot.committed = field_trials[id].clone();
            slot.trial = None;
        }
        for (id, slot) in &mut self.constitutive {
            slot.committed = constitutive_trials[id].clone();
            slot.trial = None;
        }
        self.time = time;
        self.step = next_step;
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

    pub fn checkpoint(&self) -> Result<Checkpoint, KrasisError> {
        if self.phase != TransactionPhase::Committed {
            return Err(KrasisError::TrialAlreadyActive);
        }
        self.ensure_complete()?;
        Ok(Checkpoint {
            layout_identity: self.layout.identity().to_owned(),
            history_limit: self.history_limit,
            time: self.time,
            step: self.step,
            fields: self
                .fields
                .iter()
                .map(|(id, slot)| (id.to_string(), slot.committed.clone()))
                .collect(),
            field_history: self
                .fields
                .iter()
                .map(|(id, slot)| (id.to_string(), slot.history.iter().cloned().collect()))
                .collect(),
            constitutive: self
                .constitutive
                .iter()
                .map(|(id, slot)| (id.clone(), slot.committed.clone()))
                .collect(),
        })
    }

    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<(), KrasisError> {
        self.require_committed_structure()?;
        if checkpoint.layout_identity != self.layout.identity() {
            return Err(KrasisError::LayoutMismatch {
                actual: checkpoint.layout_identity.clone(),
                expected: self.layout.identity().to_owned(),
            });
        }
        if !checkpoint.time.is_finite() {
            return Err(KrasisError::MalformedCheckpoint(
                "checkpoint time is not finite".into(),
            ));
        }

        let expected_fields: BTreeSet<_> = self
            .layout
            .blocks()
            .iter()
            .map(|block| block.id().as_str().to_owned())
            .collect();
        let checkpoint_fields: BTreeSet<_> = checkpoint.fields.keys().cloned().collect();
        let checkpoint_histories: BTreeSet<_> = checkpoint.field_history.keys().cloned().collect();
        if checkpoint_fields != expected_fields || checkpoint_histories != expected_fields {
            return Err(KrasisError::MalformedCheckpoint(
                "checkpoint fields do not exactly match the state layout".into(),
            ));
        }

        let expected_constitutive: BTreeSet<_> = self.constitutive.keys().cloned().collect();
        let checkpoint_constitutive: BTreeSet<_> =
            checkpoint.constitutive.keys().cloned().collect();
        if checkpoint_constitutive != expected_constitutive {
            return Err(KrasisError::MalformedCheckpoint(
                "checkpoint constitutive slots do not match the active schema".into(),
            ));
        }

        let mut fields = BTreeMap::new();
        for block in self.layout.blocks() {
            let id = FieldId::new(block.id().as_str());
            let values = checkpoint.fields.get(id.as_str()).ok_or_else(|| {
                KrasisError::MalformedCheckpoint(format!("field `{id}` is missing"))
            })?;
            let expected = block.range().len();
            validate_field_values(id.as_str(), values, expected)?;
            let histories = checkpoint.field_history.get(id.as_str()).ok_or_else(|| {
                KrasisError::MalformedCheckpoint(format!("history for field `{id}` is missing"))
            })?;
            if histories.len() > checkpoint.history_limit {
                return Err(KrasisError::MalformedCheckpoint(format!(
                    "history for field `{id}` exceeds the checkpoint limit"
                )));
            }
            for history in histories {
                validate_field_values(id.as_str(), history, expected)?;
            }
            fields.insert(
                id,
                FieldSlot {
                    committed: values.clone(),
                    trial: None,
                    history: histories.iter().cloned().collect(),
                },
            );
        }

        let mut constitutive = BTreeMap::new();
        for (id, current) in &self.constitutive {
            let values = checkpoint.constitutive.get(id).ok_or_else(|| {
                KrasisError::MalformedCheckpoint(format!("constitutive slot `{id}` is missing"))
            })?;
            validate_field_values(id, values, current.committed.len())?;
            constitutive.insert(id.clone(), ConstitutiveSlot::new(values.clone()));
        }

        self.fields = fields;
        self.constitutive = constitutive;
        self.history_limit = checkpoint.history_limit;
        self.time = checkpoint.time;
        self.step = checkpoint.step;
        self.phase = TransactionPhase::Committed;
        Ok(())
    }

    fn require_committed_structure(&self) -> Result<(), KrasisError> {
        if self.phase == TransactionPhase::Trial {
            Err(KrasisError::StructureDuringTrial)
        } else {
            Ok(())
        }
    }

    fn ensure_complete(&self) -> Result<(), KrasisError> {
        for block in self.layout.blocks() {
            if !self
                .fields
                .keys()
                .any(|id| id.as_str() == block.id().as_str())
            {
                return Err(KrasisError::MissingField(block.id().to_string()));
            }
        }
        Ok(())
    }
}

fn validate_field_values(id: &str, values: &[f64], expected: usize) -> Result<(), KrasisError> {
    if values.len() != expected {
        return Err(KrasisError::FieldLength {
            field: id.to_owned(),
            actual: values.len(),
            expected,
        });
    }
    require_finite(id, values)
}

fn require_finite(label: &str, values: &[f64]) -> Result<(), KrasisError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(KrasisError::NonFiniteValue {
            label: label.to_owned(),
            index,
        });
    }
    Ok(())
}
