//! Transactional explicit method-of-lines stepping (SV2-C4).
//!
//! [`ExplicitExecution`] runs Methodus's `explicit_step` over any `methodus::ExplicitOperator`
//! inside the Krasis trial/commit/rollback protocol, with the shape of
//! [`crate::CoupledExecution::attempt_step`]: every attempt opens a trial on the committed
//! [`SimulationState`]; an accepted step sets the trial values and commits them at the step's
//! time; a rejected attempt (an inadmissible stage, or a stage over the CFL maximum) rolls the
//! trial back to the committed state and is retried at Methodus's suggested step within the
//! declared attempt budget. Errors are typed and never retried: a typed evaluation refusal is
//! [`KrasisError::EvaluationRefused`] (logged), a step below the policy minimum is
//! [`KrasisError::ExplicitStepCollapse`], a rejection without a suggestion (`StepPolicy::Fixed`)
//! is [`KrasisError::ExplicitStepRejected`], and a used-up budget is
//! [`KrasisError::ExplicitAttemptBudgetExhausted`].
//!
//! Checkpoints ([`ExplicitCheckpoint`]) bind the committed state and the Methodus
//! [`ExplicitState`] to the operator's [`OperatorIdentity`] and to the identity of the
//! [`ExplicitConfig`] (scheme and step policy, [`explicit_config_identity`]); a checkpoint taken
//! against a different operator or configuration is refused before any state changes.
//!
//! Each accepted step appends an [`ExplicitStepRecord`]: the Methodus [`ExplicitStepReport`]
//! verbatim (step, step limit, scheme, realized CFL, invariant balances) plus, per linear
//! invariant, the balance residual compared against the a priori round-off bound
//! `k·ε·scale`, `k = dimension + 4·stages` (one ε unit per summand of the two invariant sums plus
//! four per stage of stage arithmetic, the budget Methodus's own conservation proof uses). This is
//! arithmetic on the reported numbers only; Krasis computes no physics and names no mesh, and
//! Sinbad maps [`ExplicitExecution::conservation_summary`] into its `DiscreteConservation`
//! evidence. Invariants are reported by index; a caller that knows what each weight vector
//! conserves (Finitum's realization declares only the vectors today) names them through
//! [`ExplicitExecution::with_invariant_labels`], and every balance check and the per-invariant
//! summary then carry the [`InvariantLabel`] (name and conserved component). Single-group
//! explicit only: coupled explicit and multirate stepping are open.

use serde::{Deserialize, Serialize};

use methodus::{
    EvaluationContext, ExplicitConfig, ExplicitOperator, ExplicitRejection, ExplicitState,
    ExplicitStepOutcome, ExplicitStepReport, SolveError, StepPolicy, explicit_step,
};

use crate::coupled::krasis_solve_error;
use crate::{
    Checkpoint, EvaluationRefusal, KrasisError, OperatorIdentity, SimulationState, TransactionPhase,
};

/// Schema tag of [`explicit_config_identity`].
pub const EXPLICIT_CONFIG_IDENTITY_SCHEMA: &str = "krasis-explicit-config/1";

/// Content identity of an explicit configuration: `krasis-explicit-config/1:blake3:<hex>` over
/// the serialized [`ExplicitConfig`] (scheme and step policy, the inputs that determine the
/// accepted trajectory). The attempt budget is not part of it: it bounds how many rejected
/// attempts a step may take, never which state a step accepts.
pub fn explicit_config_identity(config: &ExplicitConfig) -> String {
    let bytes = serde_json::to_vec(config).expect("explicit configuration is serializable");
    format!(
        "{EXPLICIT_CONFIG_IDENTITY_SCHEMA}:blake3:{}",
        blake3::hash(&bytes).to_hex()
    )
}

/// Schema tag of [`InvariantLabel`].
pub const INVARIANT_LABEL_SCHEMA: &str = "krasis-invariant-label/1";

/// The caller-declared meaning of one linear invariant `w_k·y` of an `ExplicitOperator`
/// (`krasis-invariant-label/1`): a name (`"mass"`, `"momentum-x"`, `"energy"`, ...) and, for a
/// system of several conserved components, the index of the component the weight vector sums.
/// Krasis assigns no meaning itself: the labels are telemetry naming, declared through
/// [`ExplicitExecution::with_invariant_labels`], and take no part in the configuration or
/// checkpoint identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvariantLabel {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<usize>,
}

impl InvariantLabel {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            component: None,
        }
    }

    #[must_use]
    pub fn component(name: impl Into<String>, component: usize) -> Self {
        Self {
            name: name.into(),
            component: Some(component),
        }
    }
}

/// One invariant balance of an accepted step checked against the a priori round-off bound.
/// `label` is present only when the execution declared invariant labels
/// ([`ExplicitExecution::with_invariant_labels`]); without them the check serializes exactly as
/// before (W12).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InvariantBalanceCheck {
    /// `|balance_residual| / scale`; `None` when `scale` is zero.
    pub relative_imbalance: Option<f64>,
    /// `k·ε·scale`, `k` being [`ExplicitStepRecord::bound_factor`].
    pub bound: f64,
    /// `|balance_residual| <= bound`.
    pub within_bound: bool,
    /// The declared label of this invariant, when the execution has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<InvariantLabel>,
}

/// Telemetry of one committed explicit step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitStepRecord {
    /// The committed step count after this step (`1` for the first step).
    pub accepted_step: u64,
    pub start_time: f64,
    pub end_time: f64,
    /// Attempts this step took, the accepted one included (`1` without a rejection).
    pub attempts: usize,
    /// `k = dimension + 4·stages` of the balance bound `k·ε·scale`.
    pub bound_factor: f64,
    /// Methodus's report of the accepted attempt, verbatim.
    pub report: ExplicitStepReport,
    /// One check per entry of `report.invariants`, in the same order.
    pub balances: Vec<InvariantBalanceCheck>,
}

/// A rejected attempt, rolled back and retried at `suggested_step` when there is one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitRejectionRecord {
    /// The committed step count the attempt would have produced.
    pub accepted_step: u64,
    /// The committed time the attempt started from.
    pub time: f64,
    /// `1` for the first attempt of the step.
    pub attempt: usize,
    pub attempted_step: f64,
    pub rejection: ExplicitRejection,
    pub suggested_step: Option<f64>,
}

/// An attempt that ran into a typed evaluation failure: rolled back, logged and not retried.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitRefusedAttempt {
    pub refusal: EvaluationRefusal,
    /// The committed time the attempt started from.
    pub time: f64,
    pub final_time: f64,
    /// The retry ceiling the attempt ran under (`None` on a step's first attempt).
    pub retry_ceiling: Option<f64>,
}

/// The balance of one linear invariant over a run of committed steps.
#[derive(Clone, Debug, PartialEq)]
pub struct InvariantConservationSummary {
    /// Position in the operator's `linear_invariants()`.
    pub index: usize,
    /// The declared label, when the execution has one.
    pub label: Option<InvariantLabel>,
    /// Largest `|balance_residual| / scale` over every step; `+∞` when a balance with zero
    /// scale has a nonzero residual.
    pub max_relative_imbalance: f64,
    /// Every step's balance of this invariant lies within its bound.
    pub within_bound: bool,
}

/// Linear-invariant balance over a run of committed steps, for Sinbad's
/// `DiscreteConservation` evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ExplicitConservationSummary {
    pub steps: usize,
    pub invariants: usize,
    /// Largest `|balance_residual| / scale` over every step and invariant; `+∞` when a balance
    /// with zero scale has a nonzero residual.
    pub max_relative_imbalance: f64,
    /// `k·ε`, the relative form of the per-step bound.
    pub relative_bound: f64,
    /// Every balance of every step lies within its bound.
    pub within_bound: bool,
    /// Every step's balance was corrected by declared boundary/source rates.
    pub balance_corrected: bool,
    /// One entry per invariant, in operator order, each named by its label when declared.
    pub per_invariant: Vec<InvariantConservationSummary>,
}

/// Serializable restart data of an [`ExplicitExecution`]: the Krasis committed state and the
/// Methodus explicit state, bound to the operator's content identity and the configuration
/// identity they were produced against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplicitCheckpoint {
    pub operator_identity: String,
    pub config_identity: String,
    pub state: Checkpoint,
    pub integrator: ExplicitState,
}

/// Transactional explicit execution of one group of fields over an `ExplicitOperator`.
#[derive(Debug)]
pub struct ExplicitExecution<Op: ExplicitOperator + OperatorIdentity> {
    operator: Op,
    operator_identity: String,
    config: ExplicitConfig,
    config_identity: String,
    attempt_budget: usize,
    state: SimulationState,
    integrator: ExplicitState,
    invariant_labels: Vec<InvariantLabel>,
    steps: Vec<ExplicitStepRecord>,
    rejections: Vec<ExplicitRejectionRecord>,
    refusals: Vec<ExplicitRefusedAttempt>,
}

impl<Op: ExplicitOperator + OperatorIdentity> ExplicitExecution<Op> {
    /// Binds `operator` and `config` to a committed step-0 `state` whose width is the operator's
    /// dimension. `attempt_budget` is the number of attempts one step may take (the first plus
    /// its retries), at least one. The initial state must be admissible
    /// ([`KrasisError::InadmissibleExplicitState`] otherwise). A state past step 0 is loaded
    /// through [`Self::restore`].
    pub fn new(
        operator: Op,
        state: SimulationState,
        config: ExplicitConfig,
        attempt_budget: usize,
        context: &EvaluationContext,
    ) -> Result<Self, KrasisError> {
        if state.phase() != TransactionPhase::Committed {
            return Err(KrasisError::InvalidCoupling(
                "explicit execution must start from committed state".into(),
            ));
        }
        if state.step() != 0 {
            return Err(KrasisError::InvalidCoupling(
                "nonzero-step state must be loaded through an explicit checkpoint".into(),
            ));
        }
        if attempt_budget == 0 {
            return Err(KrasisError::InvalidCoupling(
                "explicit attempt budget must allow at least one attempt".into(),
            ));
        }
        let has_nan = match config.step_policy {
            StepPolicy::Fixed { step } => step.is_nan(),
            StepPolicy::Cfl {
                target,
                max,
                min_step,
                max_step,
            } => [target, max, min_step, max_step].iter().any(|v| v.is_nan()),
        };
        if has_nan {
            return Err(KrasisError::InvalidCoupling(
                "explicit step policy contains NaN".into(),
            ));
        }
        let values = state.committed_vector()?;
        if values.len() != operator.dimension() {
            return Err(KrasisError::InvalidCoupling(format!(
                "explicit operator has dimension {}, Krasis state width is {}",
                operator.dimension(),
                values.len()
            )));
        }
        let integrator = ExplicitState::initialize(&operator, context, state.time(), values)
            .map_err(explicit_solve_error)?;
        let execution = Self {
            operator_identity: operator.content_identity(),
            config_identity: explicit_config_identity(&config),
            operator,
            config,
            attempt_budget,
            state,
            integrator,
            invariant_labels: Vec::new(),
            steps: Vec::new(),
            rejections: Vec::new(),
            refusals: Vec::new(),
        };
        execution.validate_synchronized()?;
        Ok(execution)
    }

    /// Declares what each of the operator's linear invariants conserves, one label per entry of
    /// `linear_invariants()` in the same order. Every later [`InvariantBalanceCheck`] and the
    /// [`InvariantConservationSummary`] carry the label. Refused
    /// ([`KrasisError::InvalidCoupling`]) when the count differs from the operator's invariant
    /// count, a name is empty, two labels coincide (same name and component), or a step has
    /// already been recorded (labels are declared before stepping, so one run's telemetry is
    /// uniformly named). Labels are not part of the configuration or checkpoint identity: a
    /// checkpoint restores regardless of them.
    pub fn with_invariant_labels(
        mut self,
        labels: Vec<InvariantLabel>,
    ) -> Result<Self, KrasisError> {
        let expected = self.operator.linear_invariants().len();
        if labels.len() != expected {
            return Err(KrasisError::InvalidCoupling(format!(
                "{} invariant labels declared for an operator with {expected} linear invariants",
                labels.len()
            )));
        }
        if !self.steps.is_empty() {
            return Err(KrasisError::InvalidCoupling(
                "invariant labels must be declared before the first explicit step".into(),
            ));
        }
        for (index, label) in labels.iter().enumerate() {
            if label.name.is_empty() {
                return Err(KrasisError::InvalidCoupling(format!(
                    "invariant label {index} has an empty name"
                )));
            }
            if labels[..index].contains(label) {
                return Err(KrasisError::InvalidCoupling(format!(
                    "invariant label {index} (`{}`, component {:?}) repeats an earlier label",
                    label.name, label.component
                )));
            }
        }
        self.invariant_labels = labels;
        Ok(self)
    }

    pub fn operator(&self) -> &Op {
        &self.operator
    }

    /// The declared invariant labels, in operator order; empty when none were declared.
    pub fn invariant_labels(&self) -> &[InvariantLabel] {
        &self.invariant_labels
    }

    /// The operator content identity every checkpoint is bound to.
    pub fn operator_identity(&self) -> &str {
        &self.operator_identity
    }

    pub fn config(&self) -> &ExplicitConfig {
        &self.config
    }

    /// [`explicit_config_identity`] of [`Self::config`].
    pub fn config_identity(&self) -> &str {
        &self.config_identity
    }

    pub fn attempt_budget(&self) -> usize {
        self.attempt_budget
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    pub fn integrator(&self) -> &ExplicitState {
        &self.integrator
    }

    /// Per-step telemetry of the committed trajectory, in step order.
    pub fn step_records(&self) -> &[ExplicitStepRecord] {
        &self.steps
    }

    /// Every rejected (rolled-back) attempt of the committed trajectory, in order.
    pub fn rejections(&self) -> &[ExplicitRejectionRecord] {
        &self.rejections
    }

    /// The transaction log of typed evaluation refusals, in order; not part of the checkpoint
    /// and left alone by [`Self::restore`], as [`crate::CoupledExecution::evaluation_refusals`].
    pub fn evaluation_refusals(&self) -> &[ExplicitRefusedAttempt] {
        &self.refusals
    }

    /// Commits one explicit step from the committed state toward `final_time` (Methodus lands
    /// exactly on it when the remainder fits the step). A rejected attempt is rolled back,
    /// recorded in [`Self::rejections`] and retried at its suggested step until the attempt
    /// budget is used ([`KrasisError::ExplicitAttemptBudgetExhausted`]). On any error the
    /// committed state, the explicit state and the step records are unchanged.
    pub fn advance(
        &mut self,
        context: &EvaluationContext,
        final_time: f64,
    ) -> Result<&ExplicitStepRecord, KrasisError> {
        self.validate_synchronized()?;
        let start_time = self.integrator.time;
        let accepted_step = self.integrator.accepted_steps + 1;
        let mut ceiling = None;
        for attempt in 1..=self.attempt_budget {
            self.state.begin_trial()?;
            let outcome = match explicit_step(
                &self.operator,
                context,
                &self.integrator,
                final_time,
                &self.config,
                ceiling,
            ) {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.state.rollback()?;
                    let error = explicit_solve_error(error);
                    if let KrasisError::EvaluationRefused {
                        code,
                        origin,
                        message,
                    } = &error
                    {
                        self.refusals.push(ExplicitRefusedAttempt {
                            refusal: EvaluationRefusal {
                                code: code.clone(),
                                origin: origin.clone(),
                                message: message.clone(),
                            },
                            time: start_time,
                            final_time,
                            retry_ceiling: ceiling,
                        });
                    }
                    return Err(error);
                }
            };
            match outcome {
                ExplicitStepOutcome::Accepted(accepted) => {
                    if let Err(error) = self.state.set_trial_vector(&accepted.state.values) {
                        self.state.rollback()?;
                        return Err(error);
                    }
                    if let Err(error) = self.state.commit(accepted.state.time) {
                        self.state.rollback()?;
                        return Err(error);
                    }
                    let bound_factor =
                        (self.operator.dimension() + 4 * accepted.report.stages) as f64;
                    let balances = accepted
                        .report
                        .invariants
                        .iter()
                        .enumerate()
                        .map(|(index, balance)| {
                            let bound = bound_factor * f64::EPSILON * balance.scale;
                            let residual = balance.balance_residual.abs();
                            InvariantBalanceCheck {
                                relative_imbalance: (balance.scale > 0.0)
                                    .then(|| residual / balance.scale),
                                bound,
                                within_bound: residual <= bound,
                                label: self.invariant_labels.get(index).cloned(),
                            }
                        })
                        .collect();
                    self.steps.push(ExplicitStepRecord {
                        accepted_step,
                        start_time,
                        end_time: accepted.state.time,
                        attempts: attempt,
                        bound_factor,
                        report: accepted.report,
                        balances,
                    });
                    self.integrator = accepted.state;
                    self.validate_synchronized()?;
                    return Ok(self.steps.last().expect("a step was just recorded"));
                }
                ExplicitStepOutcome::Rejected(rejected) => {
                    self.state.rollback()?;
                    self.rejections.push(ExplicitRejectionRecord {
                        accepted_step,
                        time: start_time,
                        attempt,
                        attempted_step: rejected.attempted_step,
                        rejection: rejected.rejection.clone(),
                        suggested_step: rejected.suggested_step,
                    });
                    let Some(suggested) = rejected.suggested_step else {
                        return Err(KrasisError::ExplicitStepRejected {
                            time: start_time,
                            rejection: rejected.rejection,
                        });
                    };
                    ceiling = Some(suggested);
                }
            }
        }
        Err(KrasisError::ExplicitAttemptBudgetExhausted {
            time: start_time,
            attempts: self.attempt_budget,
        })
    }

    /// Commits steps until the committed time is exactly `final_time` (an output time of the
    /// caller's cadence), returning the records of the steps it committed. A `final_time`
    /// equal to the committed time commits nothing; an earlier one is
    /// [`KrasisError::InvalidCommitTime`]. An error stops at the last committed step.
    pub fn advance_to(
        &mut self,
        context: &EvaluationContext,
        final_time: f64,
    ) -> Result<&[ExplicitStepRecord], KrasisError> {
        let current = self.integrator.time;
        if !(final_time.is_finite() && final_time >= current) {
            return Err(KrasisError::InvalidCommitTime {
                time: final_time,
                current,
            });
        }
        let first = self.steps.len();
        while self.integrator.time < final_time {
            self.advance(context, final_time)?;
        }
        Ok(&self.steps[first..])
    }

    /// The linear-invariant balance of every recorded step against its round-off bound, overall
    /// and per invariant (named by its label when one was declared).
    pub fn conservation_summary(&self) -> ExplicitConservationSummary {
        let invariants = self
            .steps
            .iter()
            .map(|record| record.balances.len())
            .max()
            .unwrap_or(0);
        let mut per_invariant: Vec<InvariantConservationSummary> = (0..invariants)
            .map(|index| InvariantConservationSummary {
                index,
                label: self.invariant_labels.get(index).cloned(),
                max_relative_imbalance: 0.0,
                within_bound: true,
            })
            .collect();
        for record in &self.steps {
            for (index, (check, balance)) in record
                .balances
                .iter()
                .zip(&record.report.invariants)
                .enumerate()
            {
                let relative =
                    check
                        .relative_imbalance
                        .unwrap_or(if balance.balance_residual == 0.0 {
                            0.0
                        } else {
                            f64::INFINITY
                        });
                let entry = &mut per_invariant[index];
                entry.max_relative_imbalance = entry.max_relative_imbalance.max(relative);
                entry.within_bound &= check.within_bound;
            }
        }
        let bound_factor = (self.operator.dimension() + 4 * self.config.scheme.stages()) as f64;
        ExplicitConservationSummary {
            steps: self.steps.len(),
            invariants,
            max_relative_imbalance: per_invariant
                .iter()
                .map(|entry| entry.max_relative_imbalance)
                .fold(0.0, f64::max),
            relative_bound: bound_factor * f64::EPSILON,
            within_bound: per_invariant.iter().all(|entry| entry.within_bound),
            balance_corrected: self
                .steps
                .iter()
                .all(|record| record.report.balance_corrected),
            per_invariant,
        }
    }

    pub fn checkpoint(&self) -> Result<ExplicitCheckpoint, KrasisError> {
        self.validate_synchronized()?;
        Ok(ExplicitCheckpoint {
            operator_identity: self.operator_identity.clone(),
            config_identity: self.config_identity.clone(),
            state: self.state.checkpoint()?,
            integrator: self.integrator.clone(),
        })
    }

    /// Atomically restores the committed and explicit state after validating that the checkpoint
    /// was taken against this operator identity and configuration identity
    /// ([`KrasisError::CheckpointIdentityMismatch`] otherwise) and that its two states agree.
    /// Step records and rejections past the restored step are dropped, so the telemetry
    /// describes the restored trajectory; the evaluation-refusal log is left alone.
    pub fn restore(&mut self, checkpoint: &ExplicitCheckpoint) -> Result<(), KrasisError> {
        for (what, expected, actual) in [
            (
                "operator",
                &self.operator_identity,
                &checkpoint.operator_identity,
            ),
            (
                "configuration",
                &self.config_identity,
                &checkpoint.config_identity,
            ),
        ] {
            if expected != actual {
                return Err(KrasisError::CheckpointIdentityMismatch {
                    what: what.into(),
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
        }
        let mut candidate = self.state.clone();
        candidate.restore(&checkpoint.state)?;
        validate_pair(
            &candidate,
            &checkpoint.integrator,
            self.operator.dimension(),
        )?;
        let step = checkpoint.integrator.accepted_steps;
        self.state = candidate;
        self.integrator = checkpoint.integrator.clone();
        self.steps.retain(|record| record.accepted_step <= step);
        self.rejections
            .retain(|record| record.accepted_step <= step);
        Ok(())
    }

    fn validate_synchronized(&self) -> Result<(), KrasisError> {
        validate_pair(&self.state, &self.integrator, self.operator.dimension())
    }
}

fn validate_pair(
    state: &SimulationState,
    integrator: &ExplicitState,
    dimension: usize,
) -> Result<(), KrasisError> {
    let committed = state.committed_vector()?;
    if state.phase() != TransactionPhase::Committed
        || committed.len() != dimension
        || integrator.values != committed
        || integrator.time != state.time()
        || integrator.accepted_steps != state.step()
    {
        return Err(KrasisError::InvalidCoupling(
            "Krasis committed state and Methodus explicit state are not synchronized".into(),
        ));
    }
    if !integrator.time.is_finite() || integrator.values.iter().any(|value| !value.is_finite()) {
        return Err(KrasisError::InvalidCoupling(
            "explicit state contains non-finite values".into(),
        ));
    }
    Ok(())
}

/// A Methodus explicit-step failure as Krasis reports it: the explicit refusals typed, every
/// other failure through [`krasis_solve_error`] (a typed evaluation failure stays
/// [`KrasisError::EvaluationRefused`]).
fn explicit_solve_error(error: SolveError) -> KrasisError {
    match error {
        SolveError::StepSizeCollapse {
            time,
            step,
            minimum_step,
        } => KrasisError::ExplicitStepCollapse {
            time,
            step,
            minimum_step,
        },
        SolveError::InadmissibleState { index, reason } => {
            KrasisError::InadmissibleExplicitState { index, reason }
        }
        other => krasis_solve_error(other),
    }
}
