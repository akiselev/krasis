//! SV2-C4 Krasis explicit transaction: rollback and retry, typed refusals, identity-bound
//! checkpoints and invariant-balance telemetry over plain `&[f64]` test operators.

use std::sync::Arc;

use krasis::{
    BlockId, ExplicitCheckpoint, ExplicitExecution, FieldId, InvariantBalanceCheck, InvariantLabel,
    KrasisError, OperatorIdentity, SimulationState, StateBlock, StateLayout, TransactionPhase,
    explicit_config_identity,
};
use methodus::{
    Admissibility, EvaluationContext, ExplicitConfig, ExplicitOperator, ExplicitRejection,
    ExplicitScheme, ExplicitStepOutcome, NumericError, StepPolicy, explicit_step,
};

/// Periodic first-order upwind advection `u_t + u_x = 0` over cells of the given widths, with
/// a positivity admissible set, an optional leak out of cell 0 (breaking conservation), and an
/// optional typed evaluation refusal once the time passes `refuse_after`.
#[derive(Debug)]
struct Upwind {
    widths: Vec<f64>,
    invariants: Vec<Vec<f64>>,
    leak: f64,
    refuse_after: Option<f64>,
}

impl Upwind {
    fn new(widths: Vec<f64>) -> Self {
        Self {
            invariants: vec![widths.clone()],
            widths,
            leak: 0.0,
            refuse_after: None,
        }
    }

    fn uniform(cells: usize) -> Self {
        Self::new(vec![1.0 / cells as f64; cells])
    }

    fn nonuniform(cells: usize) -> Self {
        Self::new(
            (0..cells)
                .map(|i| {
                    let x = (i as f64 + 0.5) / cells as f64;
                    (1.0 + 0.5 * (2.0 * std::f64::consts::PI * x).sin()) / cells as f64
                })
                .collect(),
        )
    }
}

impl OperatorIdentity for Upwind {
    fn content_identity(&self) -> String {
        format!(
            "test-upwind:{:?}:{}:{:?}",
            self.widths, self.leak, self.refuse_after
        )
    }
}

impl ExplicitOperator for Upwind {
    fn dimension(&self) -> usize {
        self.widths.len()
    }

    fn rhs(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        if self.refuse_after.is_some_and(|after| time > after) {
            return Err(NumericError::Evaluation {
                code: "test.law-unsupported".into(),
                origin: "test-upwind".into(),
                message: "the law refuses past its validity window".into(),
            });
        }
        let n = state.len();
        for i in 0..n {
            let upstream = state[(i + n - 1) % n];
            output[i] = -(state[i] - upstream) / self.widths[i];
        }
        output[0] -= self.leak * state[0];
        Ok(())
    }

    fn stable_step(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        Ok(Some(
            self.widths.iter().copied().fold(f64::INFINITY, f64::min),
        ))
    }

    fn admissibility(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        state: &[f64],
    ) -> Result<Admissibility, NumericError> {
        Ok(match state.iter().position(|&value| value < 0.0) {
            Some(index) => Admissibility::Inadmissible {
                index: Some(index),
                reason: "negative density".into(),
            },
            None => Admissibility::Admissible,
        })
    }

    fn linear_invariants(&self) -> &[Vec<f64>] {
        &self.invariants
    }

    fn invariant_rates(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        output: &mut [f64],
    ) -> Result<bool, NumericError> {
        // Periodic: closed. The leak is deliberately undeclared.
        output[0] = 0.0;
        Ok(true)
    }
}

fn context() -> EvaluationContext {
    EvaluationContext::reproducible()
}

fn cfl(scheme: ExplicitScheme, target: f64, max: f64, min_step: f64) -> ExplicitConfig {
    ExplicitConfig {
        scheme,
        step_policy: StepPolicy::Cfl {
            target,
            max,
            min_step,
            max_step: f64::INFINITY,
        },
    }
}

fn state_of(values: Vec<f64>) -> SimulationState {
    let layout =
        StateLayout::new(vec![StateBlock::new(BlockId::new("u"), 0..values.len())]).unwrap();
    let mut state = SimulationState::new(layout, 1);
    state.insert_field(FieldId::new("u"), values).unwrap();
    state
}

fn square_wave(cells: usize) -> Vec<f64> {
    (0..cells)
        .map(|i| {
            if (cells / 4..cells / 2).contains(&i) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn smooth_profile(cells: usize) -> Vec<f64> {
    (0..cells)
        .map(|i| 1.0 + 0.5 * (2.0 * std::f64::consts::PI * (i as f64 + 0.5) / cells as f64).cos())
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn an_inadmissible_stage_rolls_back_and_the_suggested_step_is_accepted() {
    let cells = 100;
    // Forward Euler at CFL 1.5 drives the square wave's leading edge negative.
    let config = cfl(ExplicitScheme::ForwardEuler, 1.5, 2.0, 1e-6);
    let mut execution = ExplicitExecution::new(
        Upwind::uniform(cells),
        state_of(square_wave(cells)),
        config.clone(),
        4,
        &context(),
    )
    .unwrap();
    let initial = execution.integrator().clone();

    let record = execution.advance(&context(), 1.0).unwrap().clone();
    assert_eq!(record.attempts, 2);
    assert_eq!(record.accepted_step, 1);
    let realized = record.report.cfl_realized.unwrap();
    assert!((realized - 0.75).abs() <= 1e-15, "{realized}");

    let rejections = execution.rejections();
    assert_eq!(rejections.len(), 1);
    let rejection = &rejections[0];
    assert_eq!(rejection.attempt, 1);
    assert_eq!(rejection.time, 0.0);
    assert!(matches!(
        rejection.rejection,
        ExplicitRejection::InadmissibleStage {
            stage: 1,
            index: Some(_),
            ..
        }
    ));
    let attempted = 1.5 * (1.0 / cells as f64);
    assert_eq!(rejection.attempted_step, attempted);
    assert_eq!(rejection.suggested_step, Some(attempted / 2.0));

    // The rejected attempt left nothing behind: the accepted step is exactly Methodus's step
    // from the untouched initial state under the suggested ceiling.
    let ExplicitStepOutcome::Accepted(direct) = explicit_step(
        execution.operator(),
        &context(),
        &initial,
        1.0,
        &config,
        rejection.suggested_step,
    )
    .unwrap() else {
        panic!("the suggested step must be accepted");
    };
    assert_eq!(
        bits(&direct.state.values),
        bits(&execution.integrator().values)
    );
    assert_eq!(direct.report, record.report);
    let state = execution.state();
    assert_eq!(state.phase(), TransactionPhase::Committed);
    assert_eq!(state.step(), 1);
    assert_eq!(state.time(), direct.state.time);
    assert_eq!(
        bits(&state.committed_vector().unwrap()),
        bits(&direct.state.values)
    );
    assert!(direct.state.values.iter().all(|&value| value >= 0.0));
}

#[test]
fn rejected_attempts_are_typed_refusals_and_leave_the_committed_state_bitwise() {
    let cells = 100;
    let context = context();
    let initial = square_wave(cells);
    let cases = [
        // One attempt only: the rejection exhausts the budget.
        (
            cfl(ExplicitScheme::ForwardEuler, 1.5, 2.0, 1e-6),
            1,
            KrasisError::ExplicitAttemptBudgetExhausted {
                time: 0.0,
                attempts: 1,
            },
        ),
        // The halved retry falls below the policy minimum.
        (
            cfl(ExplicitScheme::ForwardEuler, 1.5, 2.0, 0.01),
            4,
            KrasisError::ExplicitStepCollapse {
                time: 0.0,
                step: 1.5 * (1.0 / cells as f64) / 2.0,
                minimum_step: 0.01,
            },
        ),
    ];
    for (config, budget, expected) in cases {
        let mut execution = ExplicitExecution::new(
            Upwind::uniform(cells),
            state_of(initial.clone()),
            config,
            budget,
            &context,
        )
        .unwrap();
        let before = serde_json::to_vec(&execution.checkpoint().unwrap()).unwrap();
        assert_eq!(execution.advance(&context, 1.0).unwrap_err(), expected);
        assert_eq!(
            serde_json::to_vec(&execution.checkpoint().unwrap()).unwrap(),
            before
        );
        assert!(execution.step_records().is_empty());
        assert_eq!(execution.rejections().len(), 1);
        assert!(execution.evaluation_refusals().is_empty());
    }

    // A fixed step never shrinks on its own, so its rejection is not retried.
    let fixed = ExplicitConfig {
        scheme: ExplicitScheme::ForwardEuler,
        step_policy: StepPolicy::Fixed {
            step: 1.5 / cells as f64,
        },
    };
    let mut execution = ExplicitExecution::new(
        Upwind::uniform(cells),
        state_of(initial),
        fixed,
        4,
        &context,
    )
    .unwrap();
    let error = execution.advance(&context, 1.0).unwrap_err();
    assert!(
        matches!(
            error,
            KrasisError::ExplicitStepRejected {
                time: 0.0,
                rejection: ExplicitRejection::InadmissibleStage { stage: 1, .. },
            }
        ),
        "{error:?}"
    );
    assert_eq!(execution.rejections().len(), 1);
    assert_eq!(execution.state().step(), 0);
}

#[test]
fn a_typed_evaluation_refusal_rolls_back_is_logged_and_is_not_retried() {
    let cells = 32;
    let mut operator = Upwind::uniform(cells);
    operator.refuse_after = Some(0.05);
    let mut execution = ExplicitExecution::new(
        operator,
        state_of(smooth_profile(cells)),
        cfl(ExplicitScheme::SspRk3, 0.9, 1.0, 1e-9),
        8,
        &context(),
    )
    .unwrap();
    let error = execution.advance_to(&context(), 1.0).unwrap_err();
    assert_eq!(
        error,
        KrasisError::EvaluationRefused {
            code: "test.law-unsupported".into(),
            origin: "test-upwind".into(),
            message: "the law refuses past its validity window".into(),
        }
    );
    let committed = execution.state().step();
    assert!(committed > 0);
    assert_eq!(execution.step_records().len() as u64, committed);
    assert!(execution.rejections().is_empty());
    let refusals = execution.evaluation_refusals();
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].refusal.code, "test.law-unsupported");
    assert_eq!(refusals[0].time, execution.state().time());
    assert_eq!(refusals[0].retry_ceiling, None);
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);
}

#[test]
fn checkpoint_round_trip_reproduces_the_trajectory_bitwise() {
    let cells = 64;
    let config = cfl(ExplicitScheme::SspRk3, 0.9, 1.0, 1e-12);
    let context = context();
    let mut execution = ExplicitExecution::new(
        Upwind::nonuniform(cells),
        state_of(smooth_profile(cells)),
        config.clone(),
        4,
        &context,
    )
    .unwrap();
    execution.advance_to(&context, 0.5).unwrap();
    assert_eq!(execution.state().time(), 0.5);
    let midpoint = execution.step_records().len();
    let wire = serde_json::to_string(&execution.checkpoint().unwrap()).unwrap();
    execution.advance_to(&context, 1.0).unwrap();
    assert_eq!(execution.state().time(), 1.0);
    let reference = execution.checkpoint().unwrap();
    let reference_records = execution.step_records().to_vec();

    let checkpoint: ExplicitCheckpoint = serde_json::from_str(&wire).unwrap();
    assert_eq!(
        checkpoint.config_identity,
        explicit_config_identity(&config)
    );

    // Restored into a fresh execution.
    let mut restarted = ExplicitExecution::new(
        Upwind::nonuniform(cells),
        state_of(smooth_profile(cells)),
        config,
        4,
        &context,
    )
    .unwrap();
    restarted.restore(&checkpoint).unwrap();
    assert!(restarted.step_records().is_empty());
    restarted.advance_to(&context, 1.0).unwrap();
    assert_eq!(
        serde_json::to_vec(&restarted.checkpoint().unwrap()).unwrap(),
        serde_json::to_vec(&reference).unwrap()
    );
    assert_eq!(
        bits(&restarted.integrator().values),
        bits(&reference.integrator.values)
    );
    assert_eq!(restarted.step_records(), &reference_records[midpoint..]);

    // Restored into the execution that already ran past it: telemetry is truncated to the
    // restored step and the replay is again bitwise.
    execution.restore(&checkpoint).unwrap();
    assert_eq!(execution.step_records(), &reference_records[..midpoint]);
    execution.advance_to(&context, 1.0).unwrap();
    assert_eq!(execution.step_records(), reference_records.as_slice());
    assert_eq!(execution.checkpoint().unwrap(), reference);
}

#[test]
fn a_checkpoint_bound_to_another_identity_is_refused_before_any_change() {
    let cells = 32;
    let context = context();
    let config = cfl(ExplicitScheme::SspRk3, 0.9, 1.0, 1e-12);
    let mut source = ExplicitExecution::new(
        Upwind::uniform(cells),
        state_of(smooth_profile(cells)),
        config.clone(),
        4,
        &context,
    )
    .unwrap();
    source.advance_to(&context, 0.25).unwrap();
    let checkpoint = source.checkpoint().unwrap();

    let other_config = cfl(ExplicitScheme::SspRk2, 0.9, 1.0, 1e-12);
    assert_ne!(
        explicit_config_identity(&other_config),
        explicit_config_identity(&config)
    );
    let mut other = ExplicitExecution::new(
        Upwind::uniform(cells),
        state_of(smooth_profile(cells)),
        other_config,
        4,
        &context,
    )
    .unwrap();
    let before = serde_json::to_vec(&other.checkpoint().unwrap()).unwrap();
    assert_eq!(
        other.restore(&checkpoint).unwrap_err(),
        KrasisError::CheckpointIdentityMismatch {
            what: "configuration".into(),
            expected: other.config_identity().to_owned(),
            actual: checkpoint.config_identity.clone(),
        }
    );
    assert_eq!(
        serde_json::to_vec(&other.checkpoint().unwrap()).unwrap(),
        before
    );

    let mut leaking = Upwind::uniform(cells);
    leaking.leak = 1e-3;
    let mut other = ExplicitExecution::new(
        leaking,
        state_of(smooth_profile(cells)),
        config.clone(),
        4,
        &context,
    )
    .unwrap();
    assert!(matches!(
        other.restore(&checkpoint).unwrap_err(),
        KrasisError::CheckpointIdentityMismatch { what, .. } if what == "operator"
    ));
    assert_eq!(other.state().step(), 0);

    // A checkpoint whose explicit state disagrees with its Krasis state is refused too.
    let mut tampered = checkpoint.clone();
    tampered.integrator.values[0] += 1.0;
    let mut same = ExplicitExecution::new(
        Upwind::uniform(cells),
        state_of(smooth_profile(cells)),
        config,
        4,
        &context,
    )
    .unwrap();
    assert!(matches!(
        same.restore(&tampered).unwrap_err(),
        KrasisError::InvalidCoupling(_)
    ));
    assert_eq!(same.state().step(), 0);
    same.restore(&checkpoint).unwrap();
    assert_eq!(same.checkpoint().unwrap(), checkpoint);
}

#[test]
fn balance_telemetry_is_within_its_bound_when_conservative_and_far_outside_when_leaking() {
    let cells = 64;
    let context = context();
    for scheme in [
        ExplicitScheme::ForwardEuler,
        ExplicitScheme::SspRk2,
        ExplicitScheme::SspRk3,
        ExplicitScheme::SspRk104,
    ] {
        let target = 0.9 * scheme.ssp_coefficient().unwrap();
        let mut execution = ExplicitExecution::new(
            Upwind::nonuniform(cells),
            state_of(smooth_profile(cells)),
            cfl(scheme, target, 6.0, 1e-12),
            4,
            &context,
        )
        .unwrap();
        execution.advance_to(&context, 1.0).unwrap();
        let k = (cells + 4 * scheme.stages()) as f64;
        for record in execution.step_records() {
            assert_eq!(record.bound_factor, k);
            assert_eq!(record.report.scheme, scheme);
            assert!(record.report.cfl_realized.is_some());
            let [check] = record.balances.as_slice() else {
                panic!("one invariant");
            };
            let balance = &record.report.invariants[0];
            assert_eq!(check.bound, k * f64::EPSILON * balance.scale);
            assert!(check.within_bound, "{scheme:?}: {balance:?} vs {check:?}");
        }
        let summary = execution.conservation_summary();
        assert_eq!(summary.steps, execution.step_records().len());
        assert_eq!(summary.invariants, 1);
        assert!(summary.within_bound && summary.balance_corrected);
        assert_eq!(summary.relative_bound, k * f64::EPSILON);
        assert!(summary.max_relative_imbalance <= summary.relative_bound);
        println!(
            "{scheme:?}: {} steps, max relative imbalance {:.3e} (bound {:.3e})",
            summary.steps, summary.max_relative_imbalance, summary.relative_bound
        );
    }

    let mut leaking = Upwind::nonuniform(cells);
    leaking.leak = 1e-3;
    let mut execution = ExplicitExecution::new(
        leaking,
        state_of(smooth_profile(cells)),
        cfl(ExplicitScheme::SspRk3, 0.9, 1.0, 1e-12),
        4,
        &context,
    )
    .unwrap();
    execution.advance_to(&context, 0.25).unwrap();
    for record in execution.step_records() {
        let check = &record.balances[0];
        assert!(!check.within_bound);
        assert!(record.report.invariants[0].balance_residual.abs() > 1e6 * check.bound);
    }
    let summary = execution.conservation_summary();
    assert!(!summary.within_bound);
    assert!(summary.max_relative_imbalance > 1e6 * summary.relative_bound);
}

#[test]
fn construction_refuses_an_unusable_start() {
    let cells = 8;
    let context = context();
    let config = cfl(ExplicitScheme::SspRk3, 0.9, 1.0, 1e-12);
    let mut negative = smooth_profile(cells);
    negative[3] = -1.0;
    assert_eq!(
        ExplicitExecution::new(
            Upwind::uniform(cells),
            state_of(negative),
            config.clone(),
            4,
            &context
        )
        .unwrap_err(),
        KrasisError::InadmissibleExplicitState {
            index: Some(3),
            reason: "negative density".into(),
        }
    );
    assert!(matches!(
        ExplicitExecution::new(
            Upwind::uniform(cells),
            state_of(smooth_profile(cells)),
            config.clone(),
            0,
            &context
        )
        .unwrap_err(),
        KrasisError::InvalidCoupling(_)
    ));
    assert!(matches!(
        ExplicitExecution::new(
            Upwind::uniform(cells + 1),
            state_of(smooth_profile(cells)),
            config.clone(),
            4,
            &context
        )
        .unwrap_err(),
        KrasisError::InvalidCoupling(_)
    ));
    let mut execution = ExplicitExecution::new(
        Upwind::uniform(cells),
        state_of(smooth_profile(cells)),
        config,
        4,
        &context,
    )
    .unwrap();
    assert!(execution.advance_to(&context, 0.0).unwrap().is_empty());
    assert!(matches!(
        execution.advance_to(&context, -1.0).unwrap_err(),
        KrasisError::InvalidCommitTime { .. }
    ));
}

// -------------------------------------------------------------------------------------------
// W13 package 3: named invariants (`InvariantLabel`) over a two-component system
// -------------------------------------------------------------------------------------------

/// Two independent periodic upwind fields `u` and `v` stacked into one state `[u; v]`, each
/// with its own cell-volume mass invariant: a stand-in for a system of several conserved
/// components whose weight vectors Finitum declares and whose meaning the caller names.
#[derive(Debug)]
struct Stacked {
    a: Upwind,
    b: Upwind,
    invariants: Vec<Vec<f64>>,
}

impl Stacked {
    fn new(a: Upwind, b: Upwind) -> Self {
        let (n, m) = (a.widths.len(), b.widths.len());
        let mut first = a.widths.clone();
        first.extend(std::iter::repeat_n(0.0, m));
        let mut second = vec![0.0; n];
        second.extend(b.widths.iter().copied());
        Self {
            a,
            b,
            invariants: vec![first, second],
        }
    }
}

impl OperatorIdentity for Stacked {
    fn content_identity(&self) -> String {
        format!(
            "test-stacked:{}|{}",
            self.a.content_identity(),
            self.b.content_identity()
        )
    }
}

impl ExplicitOperator for Stacked {
    fn dimension(&self) -> usize {
        self.a.dimension() + self.b.dimension()
    }

    fn rhs(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let n = self.a.dimension();
        self.a.rhs(context, time, &state[..n], &mut output[..n])?;
        self.b.rhs(context, time, &state[n..], &mut output[n..])
    }

    fn stable_step(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Option<f64>, NumericError> {
        let n = self.a.dimension();
        let first = self.a.stable_step(context, time, &state[..n])?;
        let second = self.b.stable_step(context, time, &state[n..])?;
        Ok(match (first, second) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        })
    }

    fn admissibility(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Admissibility, NumericError> {
        let n = self.a.dimension();
        match self.a.admissibility(context, time, &state[..n])? {
            Admissibility::Admissible => {}
            other => return Ok(other),
        }
        Ok(match self.b.admissibility(context, time, &state[n..])? {
            Admissibility::Inadmissible { index, reason } => Admissibility::Inadmissible {
                index: index.map(|i| i + n),
                reason,
            },
            admissible => admissible,
        })
    }

    fn linear_invariants(&self) -> &[Vec<f64>] {
        &self.invariants
    }

    fn invariant_rates(
        &self,
        _context: &EvaluationContext,
        _time: f64,
        _state: &[f64],
        output: &mut [f64],
    ) -> Result<bool, NumericError> {
        output[0] = 0.0;
        output[1] = 0.0;
        Ok(true)
    }
}

fn stacked_state(cells: usize) -> SimulationState {
    let layout = StateLayout::new(vec![
        StateBlock::new(BlockId::new("u"), 0..cells),
        StateBlock::new(BlockId::new("v"), cells..2 * cells),
    ])
    .unwrap();
    let mut state = SimulationState::new(layout, 1);
    state
        .insert_field(FieldId::new("u"), smooth_profile(cells))
        .unwrap();
    state
        .insert_field(
            FieldId::new("v"),
            smooth_profile(cells).into_iter().map(|x| 2.0 * x).collect(),
        )
        .unwrap();
    state
}

/// The W12 bytes of an unlabeled balance check: no `label` key.
fn unlabeled_check_bytes(check: &InvariantBalanceCheck) -> String {
    format!(
        r#"{{"relative_imbalance":{},"bound":{},"within_bound":{}}}"#,
        serde_json::to_string(&check.relative_imbalance).unwrap(),
        serde_json::to_string(&check.bound).unwrap(),
        check.within_bound
    )
}

#[test]
fn invariant_labels_name_each_conserved_component_in_every_check_and_the_summary() {
    let cells = 32;
    let context = context();
    let config = cfl(ExplicitScheme::SspRk3, 0.8, 1.0, 1e-12);
    let labels = vec![
        InvariantLabel::component("mass", 0),
        InvariantLabel::component("mass", 1),
    ];
    let mut labeled = ExplicitExecution::new(
        Stacked::new(Upwind::nonuniform(cells), Upwind::uniform(cells)),
        stacked_state(cells),
        config.clone(),
        4,
        &context,
    )
    .unwrap()
    .with_invariant_labels(labels.clone())
    .unwrap();
    assert_eq!(labeled.invariant_labels(), labels.as_slice());
    let mut unlabeled = ExplicitExecution::new(
        Stacked::new(Upwind::nonuniform(cells), Upwind::uniform(cells)),
        stacked_state(cells),
        config.clone(),
        4,
        &context,
    )
    .unwrap();
    assert!(unlabeled.invariant_labels().is_empty());

    labeled.advance_to(&context, 0.5).unwrap();
    unlabeled.advance_to(&context, 0.5).unwrap();
    assert!(labeled.step_records().len() > 1);
    assert_eq!(labeled.step_records().len(), unlabeled.step_records().len());

    for (with, without) in labeled.step_records().iter().zip(unlabeled.step_records()) {
        assert_eq!(with.balances.len(), 2);
        for (index, (check, plain)) in with.balances.iter().zip(&without.balances).enumerate() {
            assert_eq!(check.label.as_ref(), Some(&labels[index]));
            assert!(plain.label.is_none());
            // Labels change nothing but the name: the numbers are the same, and the unlabeled
            // check serializes to its W12 bytes.
            let mut stripped = check.clone();
            stripped.label = None;
            assert_eq!(&stripped, plain);
            assert_eq!(
                serde_json::to_string(plain).unwrap(),
                unlabeled_check_bytes(plain)
            );
            let json = serde_json::to_string(check).unwrap();
            assert!(json.ends_with(&format!(
                r#","label":{{"name":"mass","component":{index}}}}}"#
            )));
            assert_eq!(
                serde_json::from_str::<InvariantBalanceCheck>(&json).unwrap(),
                *check
            );
            assert!(check.within_bound, "{check:?}");
        }
    }

    let summary = labeled.conservation_summary();
    assert_eq!(summary.invariants, 2);
    assert!(summary.within_bound && summary.balance_corrected);
    assert_eq!(summary.per_invariant.len(), 2);
    for (index, entry) in summary.per_invariant.iter().enumerate() {
        assert_eq!(entry.index, index);
        assert_eq!(entry.label.as_ref(), Some(&labels[index]));
        assert!(entry.within_bound);
        assert!(entry.max_relative_imbalance <= summary.relative_bound);
    }
    assert_eq!(
        summary.max_relative_imbalance,
        summary
            .per_invariant
            .iter()
            .map(|entry| entry.max_relative_imbalance)
            .fold(0.0, f64::max)
    );
    let plain = unlabeled.conservation_summary();
    assert!(
        plain
            .per_invariant
            .iter()
            .all(|entry| entry.label.is_none())
    );
    assert_eq!(plain.max_relative_imbalance, summary.max_relative_imbalance);

    // A leaking component is named in the summary; the other stays within bound.
    let mut leaking = Upwind::uniform(cells);
    leaking.leak = 1e-3;
    let mut partial = ExplicitExecution::new(
        Stacked::new(Upwind::nonuniform(cells), leaking),
        stacked_state(cells),
        config.clone(),
        4,
        &context,
    )
    .unwrap()
    .with_invariant_labels(labels.clone())
    .unwrap();
    partial.advance_to(&context, 0.25).unwrap();
    let summary = partial.conservation_summary();
    assert!(!summary.within_bound);
    let [first, second] = summary.per_invariant.as_slice() else {
        panic!("two invariants");
    };
    assert_eq!(first.label.as_ref().unwrap().component, Some(0));
    assert!(first.within_bound);
    assert_eq!(second.label.as_ref().unwrap().component, Some(1));
    assert!(!second.within_bound);
    assert!(second.max_relative_imbalance > 1e6 * summary.relative_bound);

    // Labels take no part in the identities: a labeled checkpoint restores into an unlabeled
    // execution over the same operator and configuration.
    let checkpoint = labeled.checkpoint().unwrap();
    assert_eq!(
        checkpoint.config_identity,
        explicit_config_identity(&config)
    );
    unlabeled.restore(&checkpoint).unwrap();
    assert_eq!(
        unlabeled.state().committed_vector().unwrap(),
        labeled.state().committed_vector().unwrap()
    );
}

#[test]
fn invariant_labels_are_refused_when_they_do_not_fit_the_operator() {
    let cells = 8;
    let context = context();
    let config = cfl(ExplicitScheme::SspRk2, 0.8, 1.0, 1e-12);
    let fresh = || {
        ExplicitExecution::new(
            Stacked::new(Upwind::uniform(cells), Upwind::uniform(cells)),
            stacked_state(cells),
            config.clone(),
            4,
            &context,
        )
        .unwrap()
    };
    let refused = |labels: Vec<InvariantLabel>| {
        let error = fresh().with_invariant_labels(labels).unwrap_err();
        assert!(
            matches!(error, KrasisError::InvalidCoupling(_)),
            "{error:?}"
        );
        error.to_string()
    };
    assert!(refused(vec![InvariantLabel::new("mass")]).contains("1 invariant labels"));
    assert!(
        refused(vec![
            InvariantLabel::new("mass"),
            InvariantLabel::new("momentum"),
            InvariantLabel::new("energy"),
        ])
        .contains("3 invariant labels")
    );
    assert!(
        refused(vec![InvariantLabel::new("mass"), InvariantLabel::new("")]).contains("empty name")
    );
    assert!(
        refused(vec![
            InvariantLabel::component("mass", 0),
            InvariantLabel::component("mass", 0),
        ])
        .contains("repeats")
    );
    // Distinct components of the same name are distinct labels; distinct names without
    // components are too.
    fresh()
        .with_invariant_labels(vec![
            InvariantLabel::component("mass", 0),
            InvariantLabel::component("mass", 1),
        ])
        .unwrap();
    fresh()
        .with_invariant_labels(vec![
            InvariantLabel::new("mass"),
            InvariantLabel::new("energy"),
        ])
        .unwrap();
    // After the first step the run's telemetry is already unlabeled; it stays so.
    let mut stepped = fresh();
    stepped.advance(&context, 1.0).unwrap();
    let error = stepped
        .with_invariant_labels(vec![
            InvariantLabel::new("mass"),
            InvariantLabel::new("energy"),
        ])
        .unwrap_err();
    assert!(error.to_string().contains("before the first explicit step"));
}

// -------------------------------------------------------------------------------------------
// W13 package 1: `OperatorIdentity for finitum::FvRealization` (`finitum-fv:` prefix)
// -------------------------------------------------------------------------------------------

/// A periodic scalar conservation law in the SV2-C2a fixture's shape (decision D1: conserved
/// fields are `DG(order=0)`), realized first order on a periodic line.
const FV_MODEL: &str = r#"
module krasis.w13.fv_identity;
model Burgers1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter k;
  constitutive flux = 0.5 * u * u;
  equation balance on Line { dt(u) + div(flux) = -k * u; }
}
"#;

fn fv_realization(topology: &Arc<finitum::FvTopology>, k: f64) -> finitum::FvRealization {
    use scientia::{
        FvMethodIntent, NumericalFluxIntent, PositivityIntent, ReconstructionIntent,
        SourceTreatment,
    };
    let module = scientia::compile_semantics(FV_MODEL, &quantitas::UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("{diagnostics:#?}"))
        .semantic;
    let system =
        scientia::compile_conservation_law_system(&module, "Burgers1D", "balance", "u").unwrap();
    let intent = FvMethodIntent {
        numerical_flux: NumericalFluxIntent::Hll,
        reconstruction: ReconstructionIntent::Constant,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    };
    let program =
        Arc::new(scientia::lower_conservation_law(&module, &system, &intent, &[]).unwrap());
    finitum::FvRealization::new(
        &system,
        program,
        topology.clone(),
        &finitum::RegionMap::new(),
        &[("k".to_string(), k)].into_iter().collect(),
    )
    .unwrap()
}

#[test]
fn an_fv_realization_carries_the_finitum_fv_identity_and_steps_in_the_explicit_transaction() {
    let cells = 16;
    let ring = Arc::new(
        finitum::FvTopology::from_tensor_box(&finitum::TensorBox {
            axes: vec![finitum::TensorAxis::uniform(0.0, 1.0, cells).periodic()],
        })
        .unwrap(),
    );
    let one = fv_realization(&ring, 0.3);
    let same = fv_realization(&ring, 0.3);
    let other = fv_realization(&ring, 0.4);

    // The identity is the realization digest under the `finitum-fv:` prefix: stable across
    // equal realizations, different when the digest differs.
    let identity = one.content_identity();
    assert_eq!(identity, format!("finitum-fv:{}", one.digest()));
    assert!(identity.starts_with("finitum-fv:blake3:"), "{identity}");
    assert_eq!(identity, same.content_identity());
    assert_ne!(one.digest(), other.digest());
    assert_ne!(identity, other.content_identity());
    assert_eq!(
        other.content_identity(),
        format!("finitum-fv:{}", other.digest())
    );
    // The blanket `&Op` impl Krasis's executions can take reports the same string.
    let by_reference: &dyn OperatorIdentity = &&one;
    assert_eq!(by_reference.content_identity(), identity);

    // The realization is an `ExplicitOperator` with one declared invariant (cell volumes);
    // the Krasis explicit transaction binds its checkpoint to the identity and names the invariant.
    let context = context();
    let config = cfl(ExplicitScheme::SspRk2, 0.5, 1.0, 1e-12);
    assert_eq!(one.linear_invariants().len(), 1);
    let mut execution = ExplicitExecution::new(
        one,
        state_of(smooth_profile(cells)),
        config.clone(),
        4,
        &context,
    )
    .unwrap()
    .with_invariant_labels(vec![InvariantLabel::component("u", 0)])
    .unwrap();
    assert_eq!(execution.operator_identity(), identity);
    execution.advance_to(&context, 0.1).unwrap();
    assert!(!execution.step_records().is_empty());
    let checkpoint = execution.checkpoint().unwrap();
    assert_eq!(checkpoint.operator_identity, identity);
    let summary = execution.conservation_summary();
    assert_eq!(summary.per_invariant.len(), 1);
    assert_eq!(
        summary.per_invariant[0].label,
        Some(InvariantLabel::component("u", 0))
    );
    // The source `-k·u` is a declared rate: the balance is corrected and within bound.
    assert!(summary.balance_corrected, "{summary:?}");
    assert!(summary.within_bound, "{summary:?}");

    // A checkpoint from the other realization is refused by identity before any change.
    let mut other_execution =
        ExplicitExecution::new(other, state_of(smooth_profile(cells)), config, 4, &context)
            .unwrap();
    let error = other_execution.restore(&checkpoint).unwrap_err();
    assert!(matches!(
        error,
        KrasisError::CheckpointIdentityMismatch { ref what, .. } if what == "operator"
    ));
    assert_eq!(other_execution.state().step(), 0);
}
