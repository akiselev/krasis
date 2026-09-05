//! Reusable checks for Krasis-owned coupled state and composition behavior.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use finitum::{
    PatchCheckReport, RealizationAgreementReport, RealizationPlan, VerificationReportHeader,
};
use methodus::{
    BdfConfig, BdfState, BlockLayout, BlockNonlinearOperator, BlockStrategy, DaeOperator,
    EvaluationContext, NewtonConfig, NonlinearOperator, NumericError, StepOutcome, WorkBudget,
    WorkBudgetReport, WorkCount, bdf_step, check_centered_difference,
    check_solve_strategy_agreement, check_work_budget, solve_blocks, trajectory_error_norms,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{CoupledExecution, EventDirection, FinitumRealization, TransactionalOperator};

/// `/2` (W7): a report binds one Finitum source per realization the operator is built over
/// (`finitum_sources`, in operator order) instead of `/1`'s single optional source, so the same
/// checks cover a [`crate::CoupledSystemOperator`] of N Finitum-backed leaves; every other field
/// and every verdict/number is unchanged.
pub const KRASIS_VERIFICATION_SCHEMA: &str = "krasis-verification/2";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationBinding {
    pub schema: String,
    pub operator_identity: String,
    pub layout_identity: String,
    pub config_identity: String,
    /// One entry per bound Finitum report, in the order the [`FinitumVerificationSource`] holds
    /// them; empty for a check over a generic Methodus operator (cross-block, strategy, event).
    pub finitum_sources: Vec<FinitumSourceBinding>,
    /// `Some(every bound source accepted)` for an execution check; `None` for a generic one.
    pub finitum_verification_accepted: Option<bool>,
}

/// One Finitum-owned report bound to the realization it was recomputed against.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinitumSourceBinding {
    pub realization_identity: String,
    pub verification: VerificationReportHeader,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[error("{code}: {message}")]
pub struct VerificationRefusal {
    pub code: String,
    pub message: String,
}

/// In-process source needed to validate a durable Krasis report against Finitum-owned evidence:
/// one or more Finitum reports, each bound to the identity of the realization it was computed
/// over ([`FinitumRealization::identity`]). An execution check requires every realization its
/// operator is built over ([`TransactionalOperator::realizations`]) to be covered by at least one
/// report (a multi-field leaf may carry one nodal patch per field), and every report to name one
/// of them; [`Self::compose`] joins per-leaf sources for an N-leaf composition.
#[derive(Debug)]
pub struct FinitumVerificationSource {
    entries: Vec<FinitumSourceEntry>,
}

#[derive(Debug)]
struct FinitumSourceEntry {
    realization_identity: String,
    report: FinitumReport,
}

#[derive(Debug)]
enum FinitumReport {
    RealizationAgreement(RealizationAgreementReport),
    Patch(PatchCheckReport),
}

impl FinitumVerificationSource {
    /// Assembly-based realization agreement: single-model `RealizationPlan` evidence only (the
    /// system path has no Finitum counterpart yet), so it binds to a [`FinitumRealization::Plan`].
    pub fn from_realization_agreement(
        report: &RealizationAgreementReport,
        realization: &RealizationPlan,
    ) -> Result<Self, VerificationRefusal> {
        report
            .validate(realization)
            .map_err(finitum_source_refusal)?;
        Ok(Self::single(
            FinitumRealization::Plan(realization).identity(),
            FinitumReport::RealizationAgreement(report.clone()),
        ))
    }

    /// A nodal patch comparison over the realization's mesh, for a single-model plan
    /// (`&RealizationPlan`) or a Finitum reduced system operator (`&ReducedSystemOperator`)
    /// alike; `nodal_values` are one field's vertex-major values with `component_count`
    /// components (a multi-field leaf takes one patch per field, composed with
    /// [`Self::compose`]).
    pub fn check_patch<'a>(
        realization: impl Into<FinitumRealization<'a>>,
        component_count: usize,
        nodal_values: &[f64],
        tolerance: methodus::ComparisonTolerance,
        exact: impl FnMut(&[f64]) -> Vec<f64>,
    ) -> Result<Self, VerificationRefusal> {
        let realization = realization.into();
        let report = finitum::check_nodal_patch(
            realization.mesh(),
            component_count,
            nodal_values,
            tolerance,
            exact,
        )
        .map_err(finitum_source_refusal)?;
        Ok(Self::single(
            realization.identity(),
            FinitumReport::Patch(report),
        ))
    }

    /// Joins sources in the given order (one per leaf, or one per field of a leaf).
    pub fn compose(sources: impl IntoIterator<Item = Self>) -> Self {
        Self {
            entries: sources
                .into_iter()
                .flat_map(|source| source.entries)
                .collect(),
        }
    }

    /// The realization identities this source is bound to, in entry order.
    pub fn realization_identities(&self) -> impl Iterator<Item = &str> {
        self.entries
            .iter()
            .map(|entry| entry.realization_identity.as_str())
    }

    fn single(realization_identity: String, report: FinitumReport) -> Self {
        Self {
            entries: vec![FinitumSourceEntry {
                realization_identity,
                report,
            }],
        }
    }

    /// Recomputes every bound report against the realization it names among `realizations`,
    /// refusing an entry bound elsewhere, a realization no entry covers, or assembly-based
    /// evidence over a reduced system operator; returns the per-entry bindings and whether all
    /// were accepted.
    fn validate_for(
        &self,
        realizations: &[FinitumRealization<'_>],
    ) -> Result<(Vec<FinitumSourceBinding>, bool), VerificationRefusal> {
        if realizations.is_empty() {
            return Err(refusal(
                "KRASIS_VERIFY_FINITUM_SOURCE",
                "the operator is built over no Finitum realization for evidence to bind to",
            ));
        }
        let identities = realizations
            .iter()
            .map(FinitumRealization::identity)
            .collect::<Vec<_>>();
        let mut covered = vec![false; realizations.len()];
        let mut bindings = Vec::with_capacity(self.entries.len());
        let mut all_accepted = true;
        for entry in &self.entries {
            let Some(index) = identities
                .iter()
                .position(|identity| *identity == entry.realization_identity)
            else {
                return Err(refusal(
                    "KRASIS_VERIFY_FINITUM_SOURCE",
                    "Finitum evidence was bound to a different realization identity",
                ));
            };
            covered[index] = true;
            let accepted = match (&entry.report, realizations[index]) {
                (FinitumReport::RealizationAgreement(report), FinitumRealization::Plan(plan)) => {
                    report.validate(plan).map(|validated| validated.accepted)
                }
                (FinitumReport::RealizationAgreement(_), FinitumRealization::ReducedSystem(_)) => {
                    return Err(refusal(
                        "KRASIS_VERIFY_FINITUM_SOURCE",
                        "assembly-based realization agreement is single-model evidence; a \
                         reduced system operator binds nodal-patch evidence",
                    ));
                }
                (FinitumReport::Patch(report), realization) => report
                    .validate(realization.mesh())
                    .map(|validated| validated.accepted),
            }
            .map_err(finitum_source_refusal)?;
            all_accepted &= accepted;
            bindings.push(FinitumSourceBinding {
                realization_identity: entry.realization_identity.clone(),
                verification: entry.header().clone(),
                accepted,
            });
        }
        if let Some(index) = covered.iter().position(|covered| !covered) {
            return Err(refusal(
                "KRASIS_VERIFY_FINITUM_SOURCE",
                format!(
                    "no Finitum evidence is bound to realization `{}`",
                    identities[index]
                ),
            ));
        }
        Ok((bindings, all_accepted))
    }
}

impl FinitumSourceEntry {
    fn header(&self) -> &VerificationReportHeader {
        match &self.report {
            FinitumReport::RealizationAgreement(report) => &report.header,
            FinitumReport::Patch(report) => &report.header,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidatedKrasisVerification {
    pub accepted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDisposition {
    Rejected,
    SolverError,
    UnexpectedAccepted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackIdentityReport {
    pub report_digest: String,
    pub binding: VerificationBinding,
    pub disposition: AttemptDisposition,
    pub checkpoint_before_digest: String,
    pub checkpoint_after_digest: String,
    pub byte_identical: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RestartTrajectoryReport {
    pub report_digest: String,
    pub binding: VerificationBinding,
    pub split_after_steps: usize,
    pub total_steps: usize,
    pub initial_checkpoint_digest: String,
    pub trajectory_l_infinity: f64,
    pub trajectory_l2_time: f64,
    pub continuous_final_checkpoint_digest: String,
    pub restarted_final_checkpoint_digest: String,
    pub final_checkpoint_byte_identical: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossBlockDerivativeCheck {
    pub source_block: String,
    pub target_block: String,
    pub steps: Vec<f64>,
    pub maximum_absolute_errors: Vec<f64>,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossBlockDerivativeReport {
    pub report_digest: String,
    pub binding: VerificationBinding,
    pub checks: Vec<CrossBlockDerivativeCheck>,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum StrategyOutcome {
    Converged {
        state: Vec<f64>,
        work: WorkBudgetReport,
    },
    NotConverged {
        state: Vec<f64>,
        work: WorkBudgetReport,
    },
    Refused {
        message: String,
        work: WorkBudgetReport,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrategyAgreement {
    pub left: BlockStrategy,
    pub right: BlockStrategy,
    pub comparison: methodus::ComparisonReport,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrategyWorkReport {
    pub report_digest: String,
    pub binding: VerificationBinding,
    pub outcomes: Vec<(BlockStrategy, StrategyOutcome)>,
    pub agreements: Vec<StrategyAgreement>,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryReport {
    pub report_digest: String,
    pub binding: VerificationBinding,
    pub accepted_steps: usize,
    pub field_history_depths: Vec<(String, usize)>,
    pub synchronized: bool,
    pub rejected_attempt: AttemptDisposition,
    pub checkpoint_before_rejection_digest: String,
    pub checkpoint_after_rejection_digest: String,
    pub rejection_byte_identical: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventDisposition {
    pub index: usize,
    pub time: f64,
    pub value_before: f64,
    pub value_after: f64,
    pub direction: EventDirection,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventStateReport {
    pub report_digest: String,
    pub binding: VerificationBinding,
    /// This checker exercises a caller-supplied Methodus DAE event surface. Krasis does not yet
    /// persist event records in [`CoupledExecution`].
    pub scope: String,
    pub accepted_step: bool,
    pub events: Vec<EventDisposition>,
    pub input_state_before_digest: String,
    pub input_state_after_digest: String,
    pub input_state_unchanged: bool,
    pub passed: bool,
}

impl RollbackIdentityReport {
    pub fn validate<Op: TransactionalOperator>(
        &self,
        execution: &CoupledExecution<Op>,
        context: &EvaluationContext,
        step: f64,
        config: &BdfConfig,
        finitum: &FinitumVerificationSource,
    ) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
        require_report(
            self,
            &check_rollback_identity(execution, context, step, config, finitum)?,
            self.passed,
        )
    }
}

impl RestartTrajectoryReport {
    #[allow(clippy::too_many_arguments)]
    pub fn validate<Op: TransactionalOperator>(
        &self,
        execution: &CoupledExecution<Op>,
        context: &EvaluationContext,
        step: f64,
        total_steps: usize,
        split_after_steps: usize,
        config: &BdfConfig,
        finitum: &FinitumVerificationSource,
    ) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
        require_report(
            self,
            &check_restart_trajectory(
                execution,
                context,
                step,
                total_steps,
                split_after_steps,
                config,
                finitum,
            )?,
            self.passed,
        )
    }
}

impl CrossBlockDerivativeReport {
    #[allow(clippy::too_many_arguments)]
    pub fn validate<O: BlockNonlinearOperator + ?Sized>(
        &self,
        operator: &O,
        operator_identity: &str,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        steps: &[f64],
        tolerance: methodus::ComparisonTolerance,
    ) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
        require_report(
            self,
            &check_cross_block_derivatives(
                operator,
                operator_identity,
                context,
                state,
                direction,
                steps,
                tolerance,
            )?,
            self.passed,
        )
    }
}

impl StrategyWorkReport {
    #[allow(clippy::too_many_arguments)]
    pub fn validate<O: BlockNonlinearOperator + ?Sized>(
        &self,
        operator: &O,
        operator_identity: &str,
        context: &EvaluationContext,
        initial_state: &[f64],
        config: &NewtonConfig,
        tolerance: methodus::ComparisonTolerance,
        budget: WorkBudget,
    ) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
        require_report(
            self,
            &check_strategy_work(
                operator,
                operator_identity,
                context,
                initial_state,
                config,
                tolerance,
                budget,
            )?,
            self.passed,
        )
    }
}

impl HistoryReport {
    #[allow(clippy::too_many_arguments)]
    pub fn validate<Op: TransactionalOperator>(
        &self,
        execution: &CoupledExecution<Op>,
        context: &EvaluationContext,
        step: f64,
        accepted_steps: usize,
        accepted_config: &BdfConfig,
        rejection_config: &BdfConfig,
        finitum: &FinitumVerificationSource,
    ) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
        require_report(
            self,
            &check_history_and_rejection(
                execution,
                context,
                step,
                accepted_steps,
                accepted_config,
                rejection_config,
                finitum,
            )?,
            self.passed,
        )
    }
}

impl EventStateReport {
    #[allow(clippy::too_many_arguments)]
    pub fn validate_initial<O: DaeOperator + ?Sized>(
        &self,
        operator: &O,
        operator_identity: &str,
        context: &EvaluationContext,
        initial_time: f64,
        initial_values: Vec<f64>,
        step: f64,
        config: &BdfConfig,
    ) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
        require_report(
            self,
            &check_event_state(
                operator,
                operator_identity,
                context,
                initial_time,
                initial_values,
                step,
                config,
            )?,
            self.passed,
        )
    }

    pub fn validate_from_state<O: DaeOperator + ?Sized>(
        &self,
        operator: &O,
        operator_identity: &str,
        context: &EvaluationContext,
        state: BdfState,
        step: f64,
        config: &BdfConfig,
    ) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
        require_report(
            self,
            &check_event_state_from(operator, operator_identity, context, state, step, config)?,
            self.passed,
        )
    }
}

/// Byte-exact rollback of a rejected (or failed) BDF attempt, over any [`TransactionalOperator`]:
/// a single realization or an N-leaf composition. Note that a still-fresh execution has no BDF
/// history, so its first attempt cannot be rejected on an error estimate; prime one accepted
/// step first when `config` is meant to reject.
pub fn check_rollback_identity<Op: TransactionalOperator>(
    execution: &CoupledExecution<Op>,
    context: &EvaluationContext,
    step: f64,
    config: &BdfConfig,
    finitum_verification: &FinitumVerificationSource,
) -> Result<RollbackIdentityReport, VerificationRefusal> {
    validate_bdf_identity_inputs(step, config)?;
    let binding = execution_binding(execution, context, &(step, config), finitum_verification)?;
    let mut candidate = execution.clone();
    let before = checkpoint_bytes(&candidate)?;
    let disposition = match candidate.attempt_step(context, step, config) {
        Ok(StepOutcome::Rejected(_)) => AttemptDisposition::Rejected,
        Err(_) => AttemptDisposition::SolverError,
        Ok(StepOutcome::Accepted(_)) => AttemptDisposition::UnexpectedAccepted,
    };
    let after = checkpoint_bytes(&candidate)?;
    let byte_identical = before == after;
    let passed = binding.finitum_verification_accepted == Some(true)
        && disposition != AttemptDisposition::UnexpectedAccepted
        && byte_identical;
    let mut report = RollbackIdentityReport {
        report_digest: String::new(),
        binding,
        disposition,
        checkpoint_before_digest: digest(&before),
        checkpoint_after_digest: digest(&after),
        byte_identical,
        passed,
    };
    report.report_digest = report_digest(&report)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn check_restart_trajectory<Op: TransactionalOperator>(
    execution: &CoupledExecution<Op>,
    context: &EvaluationContext,
    step: f64,
    total_steps: usize,
    split_after_steps: usize,
    config: &BdfConfig,
    finitum_verification: &FinitumVerificationSource,
) -> Result<RestartTrajectoryReport, VerificationRefusal> {
    validate_bdf_identity_inputs(step, config)?;
    if total_steps == 0 || split_after_steps == 0 || split_after_steps >= total_steps {
        return Err(refusal(
            "KRASIS_VERIFY_INVALID_RESTART_SEQUENCE",
            "restart checking requires 0 < split_after_steps < total_steps",
        ));
    }
    let binding = execution_binding(
        execution,
        context,
        &(step, total_steps, split_after_steps, config),
        finitum_verification,
    )?;
    let initial_checkpoint_digest = digest(&checkpoint_bytes(execution)?);
    let mut continuous = execution.clone();
    let mut restarted_source = execution.clone();
    let mut continuous_states = vec![continuous.integrator().values.clone()];
    let mut restarted_states = vec![restarted_source.integrator().values.clone()];
    for _ in 0..total_steps {
        accept_step(&mut continuous, context, step, config)?;
        continuous_states.push(continuous.integrator().values.clone());
    }
    for _ in 0..split_after_steps {
        accept_step(&mut restarted_source, context, step, config)?;
        restarted_states.push(restarted_source.integrator().values.clone());
    }
    let encoded = checkpoint_bytes(&restarted_source)?;
    let checkpoint = serde_json::from_slice(&encoded).map_err(|error| {
        refusal(
            "KRASIS_VERIFY_CHECKPOINT_DECODE",
            format!("checkpoint did not decode: {error}"),
        )
    })?;
    let mut restarted = execution.clone();
    restarted
        .restore(&checkpoint)
        .map_err(|error| refusal("KRASIS_VERIFY_CHECKPOINT_RESTORE", error.to_string()))?;
    for _ in split_after_steps..total_steps {
        accept_step(&mut restarted, context, step, config)?;
        restarted_states.push(restarted.integrator().values.clone());
    }
    let times = (0..=total_steps)
        .map(|index| execution.integrator().time + index as f64 * step)
        .collect::<Vec<_>>();
    let norms = trajectory_error_norms(&times, &continuous_states, &restarted_states)
        .map_err(|error| refusal("KRASIS_VERIFY_TRAJECTORY", error.to_string()))?;
    let continuous_final_checkpoint = checkpoint_bytes(&continuous)?;
    let restarted_final_checkpoint = checkpoint_bytes(&restarted)?;
    let final_checkpoint_byte_identical = continuous_final_checkpoint == restarted_final_checkpoint;
    let passed = binding.finitum_verification_accepted == Some(true)
        && norms.l_infinity == 0.0
        && norms.l2_time == 0.0
        && final_checkpoint_byte_identical;
    let mut report = RestartTrajectoryReport {
        report_digest: String::new(),
        binding,
        split_after_steps,
        total_steps,
        initial_checkpoint_digest,
        trajectory_l_infinity: norms.l_infinity,
        trajectory_l2_time: norms.l2_time,
        continuous_final_checkpoint_digest: digest(&continuous_final_checkpoint),
        restarted_final_checkpoint_digest: digest(&restarted_final_checkpoint),
        final_checkpoint_byte_identical,
        passed,
    };
    report.report_digest = report_digest(&report)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn check_cross_block_derivatives<O: BlockNonlinearOperator + ?Sized>(
    operator: &O,
    operator_identity: &str,
    context: &EvaluationContext,
    state: &[f64],
    direction: &[f64],
    steps: &[f64],
    tolerance: methodus::ComparisonTolerance,
) -> Result<CrossBlockDerivativeReport, VerificationRefusal> {
    tolerance.validate().map_err(numeric_refusal)?;
    validate_float_slice("cross-block state", state)?;
    validate_float_slice("cross-block direction", direction)?;
    validate_float_slice("cross-block steps", steps)?;
    validate_identity_float("absolute tolerance", tolerance.absolute)?;
    validate_identity_float("relative tolerance", tolerance.relative)?;
    let layout = operator.block_layout();
    if layout.blocks().len() < 2
        || state.len() != layout.dimension()
        || direction.len() != state.len()
    {
        return Err(refusal(
            "KRASIS_VERIFY_INVALID_BLOCK_PROBE",
            "cross-block checking requires at least two blocks and full state/direction vectors",
        ));
    }
    let binding = generic_binding(
        operator_identity,
        layout,
        context,
        &(state, direction, steps, tolerance),
    )?;
    let mut checks = Vec::new();
    for source in layout.blocks() {
        let mut isolated = vec![0.0; layout.dimension()];
        isolated[source.range()].copy_from_slice(&direction[source.range()]);
        if isolated[source.range()].iter().all(|value| *value == 0.0) {
            return Err(refusal(
                "KRASIS_VERIFY_ZERO_BLOCK_DIRECTION",
                format!("source block {} has a zero direction", source.name()),
            ));
        }
        let mut full_derivative = vec![0.0; layout.dimension()];
        operator
            .jacobian_vector_product(context, state, &isolated, &mut full_derivative)
            .map_err(numeric_refusal)?;
        for target in layout
            .blocks()
            .iter()
            .filter(|target| target.name() != source.name())
        {
            let analytic = full_derivative[target.range()].to_vec();
            let mut maximum_absolute_errors = vec![0.0_f64; steps.len()];
            let mut accepted = true;
            for (component, &analytic_component) in analytic.iter().enumerate() {
                let report = check_centered_difference(
                    state,
                    &isolated,
                    &[analytic_component],
                    1,
                    steps,
                    |probe, output| {
                        let mut residual = vec![0.0; layout.dimension()];
                        operator.residual(context, probe, &mut residual)?;
                        output[0] = residual[target.start() + component];
                        Ok(())
                    },
                )
                .map_err(numeric_refusal)?;
                let allowed = tolerance.absolute + tolerance.relative * analytic_component.abs();
                for (maximum, sample) in maximum_absolute_errors.iter_mut().zip(report.samples) {
                    *maximum = maximum.max(sample.maximum_absolute_error);
                    accepted &= sample.maximum_absolute_error <= allowed;
                }
            }
            checks.push(CrossBlockDerivativeCheck {
                source_block: source.name().to_owned(),
                target_block: target.name().to_owned(),
                steps: steps.to_vec(),
                maximum_absolute_errors,
                accepted,
            });
        }
    }
    let passed = !checks.is_empty() && checks.iter().all(|check| check.accepted);
    let mut report = CrossBlockDerivativeReport {
        report_digest: String::new(),
        binding,
        checks,
        passed,
    };
    report.report_digest = report_digest(&report)?;
    Ok(report)
}

pub fn check_strategy_work<O: BlockNonlinearOperator + ?Sized>(
    operator: &O,
    operator_identity: &str,
    context: &EvaluationContext,
    initial_state: &[f64],
    config: &NewtonConfig,
    tolerance: methodus::ComparisonTolerance,
    budget: WorkBudget,
) -> Result<StrategyWorkReport, VerificationRefusal> {
    tolerance.validate().map_err(numeric_refusal)?;
    validate_float_slice("initial strategy state", initial_state)?;
    validate_newton_identity(config)?;
    validate_identity_float("absolute tolerance", tolerance.absolute)?;
    validate_identity_float("relative tolerance", tolerance.relative)?;
    let binding = generic_binding(
        operator_identity,
        operator.block_layout(),
        context,
        &(initial_state, config, tolerance, budget),
    )?;
    let strategies = [
        BlockStrategy::Monolithic,
        BlockStrategy::GaussSeidel,
        BlockStrategy::Jacobi,
    ];
    let mut outcomes = Vec::new();
    for strategy in strategies {
        let counted = CountedOperator::new(operator);
        let solved = solve_blocks(&counted, context, initial_state, strategy, config);
        let counts = counted.counts()?;
        let outcome = match solved {
            Ok(report) => {
                let nonlinear_iterations = u64::try_from(report.trace.len().saturating_sub(1))
                    .map_err(|_| {
                        refusal("KRASIS_VERIFY_WORK_OVERFLOW", "iteration count exceeds u64")
                    })?;
                let work = check_work_budget(
                    WorkCount {
                        operator_evaluations: checked_operator_evaluations(counts)?,
                        linear_iterations: 0,
                        nonlinear_iterations,
                        accepted_steps: 0,
                        rejected_steps: 0,
                    },
                    budget,
                );
                if report.converged {
                    StrategyOutcome::Converged {
                        state: report.state,
                        work,
                    }
                } else {
                    StrategyOutcome::NotConverged {
                        state: report.state,
                        work,
                    }
                }
            }
            Err(error) => StrategyOutcome::Refused {
                message: error.to_string(),
                work: check_work_budget(
                    WorkCount {
                        operator_evaluations: checked_operator_evaluations(counts)?,
                        linear_iterations: 0,
                        nonlinear_iterations: 0,
                        accepted_steps: 0,
                        rejected_steps: 0,
                    },
                    budget,
                ),
            },
        };
        outcomes.push((strategy, outcome));
    }
    let mut agreements = Vec::new();
    for left in 0..outcomes.len() {
        for right in left + 1..outcomes.len() {
            if let (
                StrategyOutcome::Converged {
                    state: left_state, ..
                },
                StrategyOutcome::Converged {
                    state: right_state, ..
                },
            ) = (&outcomes[left].1, &outcomes[right].1)
            {
                agreements.push(StrategyAgreement {
                    left: outcomes[left].0,
                    right: outcomes[right].0,
                    comparison: check_solve_strategy_agreement(left_state, right_state, tolerance)
                        .map_err(numeric_refusal)?,
                });
            }
        }
    }
    let passed = outcomes.iter().all(
        |(_, outcome)| matches!(outcome, StrategyOutcome::Converged { work, .. } if work.accepted),
    ) && agreements.len() == 3
        && agreements
            .iter()
            .all(|agreement| agreement.comparison.accepted);
    let mut report = StrategyWorkReport {
        report_digest: String::new(),
        binding,
        outcomes,
        agreements,
        passed,
    };
    report.report_digest = report_digest(&report)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub fn check_history_and_rejection<Op: TransactionalOperator>(
    execution: &CoupledExecution<Op>,
    context: &EvaluationContext,
    step: f64,
    accepted_steps: usize,
    accepted_config: &BdfConfig,
    rejection_config: &BdfConfig,
    finitum_verification: &FinitumVerificationSource,
) -> Result<HistoryReport, VerificationRefusal> {
    validate_bdf_identity_inputs(step, accepted_config)?;
    validate_bdf_identity_inputs(step, rejection_config)?;
    if accepted_steps == 0 {
        return Err(refusal(
            "KRASIS_VERIFY_EMPTY_HISTORY_SEQUENCE",
            "history checking requires at least one accepted step",
        ));
    }
    let binding = execution_binding(
        execution,
        context,
        &(step, accepted_steps, accepted_config, rejection_config),
        finitum_verification,
    )?;
    let mut candidate = execution.clone();
    let mut synchronized = true;
    for _ in 0..accepted_steps {
        let previous = candidate.integrator().values.clone();
        accept_step(&mut candidate, context, step, accepted_config)?;
        synchronized &= candidate
            .state()
            .history_vector(0)
            .map_err(krasis_refusal)?
            .as_deref()
            == Some(previous.as_slice());
    }
    let checkpoint = candidate.checkpoint().map_err(krasis_refusal)?;
    let field_history_depths = checkpoint
        .state
        .field_history
        .iter()
        .map(|(field, history)| (field.clone(), history.len()))
        .collect::<Vec<_>>();
    synchronized &= field_history_depths
        .iter()
        .all(|(_, depth)| *depth == accepted_steps.min(checkpoint.state.history_limit));
    let before = checkpoint_bytes(&candidate)?;
    let rejected_attempt = match candidate.attempt_step(context, step, rejection_config) {
        Ok(StepOutcome::Rejected(_)) => AttemptDisposition::Rejected,
        Err(_) => AttemptDisposition::SolverError,
        Ok(StepOutcome::Accepted(_)) => AttemptDisposition::UnexpectedAccepted,
    };
    let after = checkpoint_bytes(&candidate)?;
    let rejection_byte_identical = before == after;
    let passed = binding.finitum_verification_accepted == Some(true)
        && synchronized
        && rejected_attempt != AttemptDisposition::UnexpectedAccepted
        && rejection_byte_identical;
    let mut report = HistoryReport {
        report_digest: String::new(),
        binding,
        accepted_steps,
        field_history_depths,
        synchronized,
        rejected_attempt,
        checkpoint_before_rejection_digest: digest(&before),
        checkpoint_after_rejection_digest: digest(&after),
        rejection_byte_identical,
        passed,
    };
    report.report_digest = report_digest(&report)?;
    Ok(report)
}

pub fn check_event_state<O: DaeOperator + ?Sized>(
    operator: &O,
    operator_identity: &str,
    context: &EvaluationContext,
    initial_time: f64,
    initial_values: Vec<f64>,
    step: f64,
    config: &BdfConfig,
) -> Result<EventStateReport, VerificationRefusal> {
    validate_identity_float("initial event time", initial_time)?;
    validate_float_slice("initial event state", &initial_values)?;
    validate_bdf_identity_inputs(step, config)?;
    let state = BdfState::initialize(operator, context, initial_time, initial_values)
        .map_err(numeric_refusal)?;
    check_event_state_from(operator, operator_identity, context, state, step, config)
}

pub fn check_event_state_from<O: DaeOperator + ?Sized>(
    operator: &O,
    operator_identity: &str,
    context: &EvaluationContext,
    state: BdfState,
    step: f64,
    config: &BdfConfig,
) -> Result<EventStateReport, VerificationRefusal> {
    validate_bdf_state_identity(&state)?;
    validate_bdf_identity_inputs(step, config)?;
    if state.values.len() != operator.dimension() {
        return Err(refusal(
            "KRASIS_VERIFY_EVENT_DIMENSION",
            "event state dimension does not match its operator",
        ));
    }
    let binding = VerificationBinding {
        schema: KRASIS_VERIFICATION_SCHEMA.into(),
        operator_identity: require_identity(operator_identity, "operator")?,
        layout_identity: format!("generic-dae-dimension/1:{}", operator.dimension()),
        config_identity: identity(&(context, &state, step, config))?,
        finitum_sources: Vec::new(),
        finitum_verification_accepted: None,
    };
    let before = serde_json::to_vec(&state).map_err(serialization_refusal)?;
    let outcome = bdf_step(operator, context, &state, step, config)
        .map_err(|error| refusal("KRASIS_VERIFY_EVENT_SOLVE", error.to_string()))?;
    let after = serde_json::to_vec(&state).map_err(serialization_refusal)?;
    let input_state_unchanged = after == before;
    let (accepted_step, events) = match outcome {
        StepOutcome::Accepted(accepted) => (
            true,
            accepted
                .events
                .into_iter()
                .map(|event| EventDisposition {
                    index: event.index,
                    time: event.time,
                    value_before: event.value_before,
                    value_after: event.value_after,
                    direction: if event.value_after > event.value_before {
                        EventDirection::Rising
                    } else if event.value_after < event.value_before {
                        EventDirection::Falling
                    } else {
                        EventDirection::Either
                    },
                })
                .collect(),
        ),
        StepOutcome::Rejected(_) => (false, Vec::new()),
    };
    let mut report = EventStateReport {
        report_digest: String::new(),
        binding,
        scope: "generic Methodus DaeOperator event surface; no Krasis CoupledExecution event persistence claim".into(),
        accepted_step,
        events,
        input_state_before_digest: digest(&before),
        input_state_after_digest: digest(&after),
        input_state_unchanged,
        passed: input_state_unchanged,
    };
    report.report_digest = report_digest(&report)?;
    Ok(report)
}

struct CountedOperator<'a, O: ?Sized> {
    inner: &'a O,
    residuals: AtomicU64,
    jvps: AtomicU64,
    overflowed: AtomicBool,
}

impl<'a, O: ?Sized> CountedOperator<'a, O> {
    fn new(inner: &'a O) -> Self {
        Self {
            inner,
            residuals: AtomicU64::new(0),
            jvps: AtomicU64::new(0),
            overflowed: AtomicBool::new(false),
        }
    }

    fn counts(&self) -> Result<(u64, u64), VerificationRefusal> {
        if self.overflowed.load(Ordering::Relaxed) {
            return Err(refusal(
                "KRASIS_VERIFY_WORK_OVERFLOW",
                "operator evaluation counter exceeds u64",
            ));
        }
        Ok((
            self.residuals.load(Ordering::Relaxed),
            self.jvps.load(Ordering::Relaxed),
        ))
    }

    fn increment(&self, counter: &AtomicU64) {
        if counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                count.checked_add(1)
            })
            .is_err()
        {
            self.overflowed.store(true, Ordering::Relaxed);
        }
    }
}

impl<O: BlockNonlinearOperator + ?Sized> NonlinearOperator for CountedOperator<'_, O> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.increment(&self.residuals);
        self.inner.residual(context, state, output)
    }

    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.increment(&self.jvps);
        self.inner
            .jacobian_vector_product(context, state, direction, output)
    }
}

impl<O: BlockNonlinearOperator + ?Sized> BlockNonlinearOperator for CountedOperator<'_, O> {
    fn block_layout(&self) -> &BlockLayout {
        self.inner.block_layout()
    }
}

fn execution_binding<Op: TransactionalOperator, T: Serialize>(
    execution: &CoupledExecution<Op>,
    context: &EvaluationContext,
    config: &T,
    finitum_verification: &FinitumVerificationSource,
) -> Result<VerificationBinding, VerificationRefusal> {
    validate_execution_identity_source(execution)?;
    let (finitum_sources, accepted) =
        finitum_verification.validate_for(&execution.operator().realizations())?;
    Ok(VerificationBinding {
        schema: KRASIS_VERIFICATION_SCHEMA.into(),
        operator_identity: execution.operator().identity().to_owned(),
        layout_identity: execution.state().layout().identity().to_owned(),
        config_identity: identity(&(context, config))?,
        finitum_sources,
        finitum_verification_accepted: Some(accepted),
    })
}

fn generic_binding<T: Serialize>(
    operator_identity: &str,
    layout: &BlockLayout,
    context: &EvaluationContext,
    config: &T,
) -> Result<VerificationBinding, VerificationRefusal> {
    Ok(VerificationBinding {
        schema: KRASIS_VERIFICATION_SCHEMA.into(),
        operator_identity: require_identity(operator_identity, "operator")?,
        layout_identity: identity(layout)?,
        config_identity: identity(&(context, config))?,
        finitum_sources: Vec::new(),
        finitum_verification_accepted: None,
    })
}

fn checked_operator_evaluations(counts: (u64, u64)) -> Result<u64, VerificationRefusal> {
    counts.0.checked_add(counts.1).ok_or_else(|| {
        refusal(
            "KRASIS_VERIFY_WORK_OVERFLOW",
            "operator evaluation count exceeds u64",
        )
    })
}

fn validate_bdf_identity_inputs(step: f64, config: &BdfConfig) -> Result<(), VerificationRefusal> {
    validate_identity_float("BDF step", step)?;
    validate_identity_float("BDF absolute tolerance", config.absolute_tolerance)?;
    validate_identity_float("BDF relative tolerance", config.relative_tolerance)?;
    validate_identity_float("BDF minimum step", config.minimum_step)?;
    validate_identity_float("BDF maximum step", config.maximum_step)?;
    validate_newton_identity(&config.newton)
}

fn validate_newton_identity(config: &NewtonConfig) -> Result<(), VerificationRefusal> {
    validate_identity_float("Newton absolute tolerance", config.absolute_tolerance)?;
    validate_identity_float("Newton relative tolerance", config.relative_tolerance)?;
    validate_identity_float("Newton initial damping", config.initial_damping)?;
    validate_identity_float("Newton minimum damping", config.minimum_damping)
}

fn validate_bdf_state_identity(state: &BdfState) -> Result<(), VerificationRefusal> {
    validate_identity_float("BDF state time", state.time)?;
    validate_float_slice("BDF state values", &state.values)?;
    if let Some(previous) = &state.previous_values {
        validate_float_slice("previous BDF state values", previous)?;
    }
    if let Some(previous_step) = state.previous_step {
        validate_identity_float("previous BDF step", previous_step)?;
    }
    Ok(())
}

fn validate_float_slice(label: &str, values: &[f64]) -> Result<(), VerificationRefusal> {
    for &value in values {
        validate_identity_float(label, value)?;
    }
    Ok(())
}

fn validate_identity_float(label: &str, value: f64) -> Result<(), VerificationRefusal> {
    if !value.is_finite() {
        return Err(refusal(
            "KRASIS_VERIFY_NONFINITE_IDENTITY_INPUT",
            format!("{label} must be finite before identity construction"),
        ));
    }
    if value.to_bits() == (-0.0_f64).to_bits() {
        return Err(refusal(
            "KRASIS_VERIFY_NEGATIVE_ZERO_IDENTITY_INPUT",
            format!("{label} must use canonical positive zero"),
        ));
    }
    Ok(())
}

fn accept_step<Op: TransactionalOperator>(
    execution: &mut CoupledExecution<Op>,
    context: &EvaluationContext,
    step: f64,
    config: &BdfConfig,
) -> Result<(), VerificationRefusal> {
    match execution.attempt_step(context, step, config) {
        Ok(StepOutcome::Accepted(_)) => Ok(()),
        Ok(StepOutcome::Rejected(_)) => Err(refusal(
            "KRASIS_VERIFY_UNEXPECTED_REJECTION",
            "an accepted step was required",
        )),
        Err(error) => Err(krasis_refusal(error)),
    }
}

fn checkpoint_bytes<Op: TransactionalOperator>(
    execution: &CoupledExecution<Op>,
) -> Result<Vec<u8>, VerificationRefusal> {
    validate_execution_identity_source(execution)?;
    let checkpoint = execution.checkpoint().map_err(krasis_refusal)?;
    serde_json::to_vec(&checkpoint).map_err(serialization_refusal)
}

fn validate_execution_identity_source<Op: TransactionalOperator>(
    execution: &CoupledExecution<Op>,
) -> Result<(), VerificationRefusal> {
    let checkpoint = execution.checkpoint().map_err(krasis_refusal)?;
    validate_coupled_checkpoint_identity(&checkpoint)?;
    for realization in execution.operator().realizations() {
        validate_realization_identity_source(realization)?;
    }
    Ok(())
}

fn validate_coupled_checkpoint_identity(
    checkpoint: &crate::CoupledCheckpoint,
) -> Result<(), VerificationRefusal> {
    validate_identity_float("checkpoint time", checkpoint.state.time)?;
    for (field, values) in &checkpoint.state.fields {
        validate_float_slice(&format!("checkpoint field `{field}`"), values)?;
    }
    for (field, histories) in &checkpoint.state.field_history {
        for (level, values) in histories.iter().enumerate() {
            validate_float_slice(
                &format!("checkpoint history `{field}` level {level}"),
                values,
            )?;
        }
    }
    for (slot, values) in &checkpoint.state.constitutive {
        validate_float_slice(&format!("checkpoint constitutive `{slot}`"), values)?;
    }
    validate_bdf_state_identity(&checkpoint.integrator)
}

/// Identity-bearing Finitum values checked finite and positive-zero canonical before hashing: a
/// plan's exposed mesh, element, constraints and stored inputs; a reduced system operator's mesh
/// and constraint set (its quadrature and bound closures are Finitum-internal and enter only
/// through the content digest).
fn validate_realization_identity_source(
    realization: FinitumRealization<'_>,
) -> Result<(), VerificationRefusal> {
    let realization = match realization {
        FinitumRealization::Plan(plan) => plan,
        FinitumRealization::ReducedSystem(reduced) => {
            let mesh = reduced.operator().plan().mesh();
            for (vertex, coordinates) in mesh.vertices().iter().enumerate() {
                validate_float_slice(&format!("realization mesh vertex {vertex}"), coordinates)?;
            }
            for constraint in reduced.constraints().constraints() {
                validate_identity_float("realization constraint offset", constraint.offset)?;
                for dependency in &constraint.dependencies {
                    validate_identity_float("realization constraint weight", dependency.weight)?;
                }
            }
            return Ok(());
        }
    };
    let artifact = realization.artifact();
    for (vertex, coordinates) in artifact.mesh.vertices().iter().enumerate() {
        validate_float_slice(&format!("realization mesh vertex {vertex}"), coordinates)?;
    }
    for (point, quadrature) in artifact.element.quadrature().iter().enumerate() {
        validate_float_slice(
            &format!("realization quadrature point {point}"),
            &quadrature.coordinates,
        )?;
        validate_identity_float("realization quadrature weight", quadrature.weight)?;
        for basis in 0..artifact.element.basis_count() {
            let value = artifact.element.basis_value(point, basis).ok_or_else(|| {
                refusal(
                    "KRASIS_VERIFY_REALIZATION_SOURCE",
                    "Finitum element omitted an expected basis value",
                )
            })?;
            validate_identity_float("realization basis value", value)?;
            let gradient = artifact
                .element
                .basis_gradient(point, basis)
                .ok_or_else(|| {
                    refusal(
                        "KRASIS_VERIFY_REALIZATION_SOURCE",
                        "Finitum element omitted an expected basis gradient",
                    )
                })?;
            validate_float_slice("realization basis gradient", gradient)?;
        }
    }
    for constraint in artifact.constraints.constraints() {
        validate_identity_float("realization constraint offset", constraint.offset)?;
        for dependency in &constraint.dependencies {
            validate_identity_float("realization constraint weight", dependency.weight)?;
        }
    }
    for input in artifact.external_inputs {
        if let finitum::RealizationExternalInput::Stored { values, .. } = input {
            validate_float_slice("realization stored external input", &values)?;
        }
    }
    Ok(())
}

fn identity(value: &impl Serialize) -> Result<String, VerificationRefusal> {
    let bytes = serde_json::to_vec(value).map_err(serialization_refusal)?;
    Ok(digest(&bytes))
}

fn report_digest(report: &impl Serialize) -> Result<String, VerificationRefusal> {
    identity(report)
}

fn require_report<T: PartialEq>(
    observed: &T,
    expected: &T,
    accepted: bool,
) -> Result<ValidatedKrasisVerification, VerificationRefusal> {
    if observed != expected {
        return Err(refusal(
            "KRASIS_VERIFY_REPORT_MISMATCH",
            "verification report does not match recomputed source, inputs, outputs, or identity",
        ));
    }
    Ok(ValidatedKrasisVerification { accepted })
}

fn finitum_source_refusal(error: finitum::FinitumError) -> VerificationRefusal {
    refusal(
        "KRASIS_VERIFY_FINITUM_SOURCE",
        format!("Finitum verification did not validate against its bound source: {error}"),
    )
}

fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn require_identity(identity: &str, label: &str) -> Result<String, VerificationRefusal> {
    if identity.trim().is_empty() {
        Err(refusal(
            "KRASIS_VERIFY_MISSING_IDENTITY",
            format!("{label} identity must be nonempty"),
        ))
    } else {
        Ok(identity.to_owned())
    }
}

fn numeric_refusal(error: NumericError) -> VerificationRefusal {
    refusal("KRASIS_VERIFY_NUMERIC", error.to_string())
}

fn krasis_refusal(error: crate::KrasisError) -> VerificationRefusal {
    refusal("KRASIS_VERIFY_STATE", error.to_string())
}

fn serialization_refusal(error: serde_json::Error) -> VerificationRefusal {
    refusal("KRASIS_VERIFY_SERIALIZATION", error.to_string())
}

fn refusal(code: &str, message: impl Into<String>) -> VerificationRefusal {
    VerificationRefusal {
        code: code.into(),
        message: message.into(),
    }
}
