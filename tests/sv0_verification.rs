use krasis::{
    EventDirection, StrategyOutcome, VerificationRefusal, check_cross_block_derivatives,
    check_event_state, check_event_state_from, check_strategy_work,
};
use methodus::{
    BdfConfig, BdfOrder, BdfState, BlockLayout, BlockNonlinearOperator, BlockSpec,
    ComparisonTolerance, DaeOperator, EvaluationContext, NewtonConfig, NonlinearOperator,
    NumericError, WorkBudget,
};

struct TwoBlockLinear {
    coupling: f64,
    layout: BlockLayout,
}

impl TwoBlockLinear {
    fn new(coupling: f64) -> Self {
        Self {
            coupling,
            layout: BlockLayout::new(vec![
                BlockSpec {
                    name: "left".into(),
                    length: 1,
                    residual_scale: 1.0,
                },
                BlockSpec {
                    name: "right".into(),
                    length: 1,
                    residual_scale: 1.0,
                },
            ])
            .unwrap(),
        }
    }
}

impl NonlinearOperator for TwoBlockLinear {
    fn dimension(&self) -> usize {
        2
    }

    fn residual(
        &self,
        _: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0] + self.coupling * state[1] - 1.0;
        output[1] = self.coupling * state[0] + state[1] - 1.0;
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _: &EvaluationContext,
        _: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = direction[0] + self.coupling * direction[1];
        output[1] = self.coupling * direction[0] + direction[1];
        Ok(())
    }
}

impl BlockNonlinearOperator for TwoBlockLinear {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

#[test]
fn isolated_cross_derivatives_and_counted_strategies_are_reusable() {
    let operator = TwoBlockLinear::new(0.1);
    let context = EvaluationContext::reproducible();
    let tolerance = ComparisonTolerance {
        absolute: 2.0e-8,
        relative: 2.0e-8,
    };
    let derivatives = check_cross_block_derivatives(
        &operator,
        "two-block-linear/1",
        &context,
        &[0.2, -0.3],
        &[0.7, -0.4],
        &[1.0e-3, 5.0e-4],
        tolerance,
    )
    .unwrap();
    assert!(derivatives.passed, "{derivatives:#?}");
    assert!(derivatives.report_digest.starts_with("blake3:"));
    assert!(
        derivatives
            .validate(
                &operator,
                "two-block-linear/1",
                &context,
                &[0.2, -0.3],
                &[0.7, -0.4],
                &[1.0e-3, 5.0e-4],
                tolerance,
            )
            .unwrap()
            .accepted
    );
    assert_eq!(derivatives.checks.len(), 2);
    assert!(
        derivatives
            .checks
            .iter()
            .all(|check| check.source_block != check.target_block)
    );

    let budget = WorkBudget {
        operator_evaluations: 1_000,
        linear_iterations: 0,
        nonlinear_iterations: 100,
        accepted_steps: 0,
        rejected_steps: 0,
    };
    let config = NewtonConfig::default();
    let first = check_strategy_work(
        &operator,
        "two-block-linear/1",
        &context,
        &[0.0, 0.0],
        &config,
        tolerance,
        budget,
    )
    .unwrap();
    let second = check_strategy_work(
        &operator,
        "two-block-linear/1",
        &context,
        &[0.0, 0.0],
        &config,
        tolerance,
        budget,
    )
    .unwrap();
    assert!(first.passed, "{first:#?}");
    assert!(
        first
            .validate(
                &operator,
                "two-block-linear/1",
                &context,
                &[0.0, 0.0],
                &config,
                tolerance,
                budget,
            )
            .unwrap()
            .accepted
    );
    assert!(
        first
            .validate(
                &operator,
                "two-block-linear/1",
                &EvaluationContext::default(),
                &[0.0, 0.0],
                &config,
                tolerance,
                budget,
            )
            .is_err()
    );
    assert_eq!(first, second);
    assert_eq!(first.agreements.len(), 3);
    assert!(serde_json::to_vec(&first).is_ok());

    let tight = check_strategy_work(
        &operator,
        "two-block-linear/1",
        &context,
        &[0.0, 0.0],
        &config,
        tolerance,
        WorkBudget {
            operator_evaluations: 0,
            ..budget
        },
    )
    .unwrap();
    assert!(!tight.passed);
    assert_ne!(first.binding.config_identity, tight.binding.config_identity);

    let renamed = check_strategy_work(
        &operator,
        "two-block-linear/2",
        &context,
        &[0.0, 0.0],
        &config,
        tolerance,
        budget,
    )
    .unwrap();
    assert_ne!(
        first.binding.operator_identity,
        renamed.binding.operator_identity
    );
}

#[test]
fn nonconverged_strategy_is_not_laundered_as_agreement() {
    let report = check_strategy_work(
        &TwoBlockLinear::new(2.0),
        "strong-two-block-linear/1",
        &EvaluationContext::reproducible(),
        &[0.0, 0.0],
        &NewtonConfig {
            max_iterations: 4,
            ..NewtonConfig::default()
        },
        ComparisonTolerance {
            absolute: 1.0e-10,
            relative: 1.0e-10,
        },
        WorkBudget {
            operator_evaluations: 1_000,
            linear_iterations: 0,
            nonlinear_iterations: 100,
            accepted_steps: 0,
            rejected_steps: 0,
        },
    )
    .unwrap();
    assert!(!report.passed);
    assert!(report.outcomes.iter().any(|(_, outcome)| {
        matches!(
            outcome,
            StrategyOutcome::NotConverged { .. } | StrategyOutcome::Refused { .. }
        )
    }));
}

struct UnevenCrossDerivative {
    layout: BlockLayout,
}

impl UnevenCrossDerivative {
    fn new() -> Self {
        Self {
            layout: BlockLayout::new(vec![
                BlockSpec {
                    name: "source".into(),
                    length: 1,
                    residual_scale: 1.0,
                },
                BlockSpec {
                    name: "target".into(),
                    length: 2,
                    residual_scale: 1.0,
                },
            ])
            .unwrap(),
        }
    }
}

impl NonlinearOperator for UnevenCrossDerivative {
    fn dimension(&self) -> usize {
        3
    }

    fn residual(
        &self,
        _: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0];
        output[1] = 1.0e6 * state[0] + state[1];
        output[2] = state[0] + state[2];
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _: &EvaluationContext,
        _: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = direction[0];
        output[1] = 1.0e6 * direction[0] + direction[1];
        // Deliberately wrong only in the small target component.
        output[2] = 1.2 * direction[0] + direction[2];
        Ok(())
    }
}

impl BlockNonlinearOperator for UnevenCrossDerivative {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

#[test]
fn cross_derivative_tolerance_is_applied_per_target_component() {
    let report = check_cross_block_derivatives(
        &UnevenCrossDerivative::new(),
        "uneven-cross-derivative/1",
        &EvaluationContext::reproducible(),
        &[0.25, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
        &[1.0e-4],
        ComparisonTolerance {
            absolute: 1.0e-12,
            relative: 1.0e-5,
        },
    )
    .unwrap();
    assert!(!report.passed);
    assert!(report.checks.iter().any(|check| {
        check.source_block == "source" && check.target_block == "target" && !check.accepted
    }));
}

struct UnitSpeedEvent;

impl DaeOperator for UnitSpeedEvent {
    fn dimension(&self) -> usize {
        1
    }

    fn residual(
        &self,
        _: &EvaluationContext,
        _: f64,
        _: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state_rate[0] - 1.0;
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _: &EvaluationContext,
        _: f64,
        _: &[f64],
        _: &[f64],
        _: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = rate_direction[0];
        Ok(())
    }

    fn event_count(&self) -> usize {
        2
    }

    fn event_values(
        &self,
        _: &EvaluationContext,
        _: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state[0] - 0.5;
        output[1] = 0.5 - state[0];
        Ok(())
    }
}

fn first_order_config(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::One,
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
        minimum_step: step,
        maximum_step: step,
        ..BdfConfig::default()
    }
}

#[test]
fn generic_event_report_is_truthfully_scoped_and_directional() {
    let report = check_event_state(
        &UnitSpeedEvent,
        "unit-speed-event/1",
        &EvaluationContext::reproducible(),
        0.0,
        vec![0.0],
        1.0,
        &first_order_config(1.0),
    )
    .unwrap();
    assert!(report.passed);
    assert!(
        report
            .validate_initial(
                &UnitSpeedEvent,
                "unit-speed-event/1",
                &EvaluationContext::reproducible(),
                0.0,
                vec![0.0],
                1.0,
                &first_order_config(1.0),
            )
            .unwrap()
            .accepted
    );
    assert!(report.accepted_step);
    assert_eq!(report.events.len(), 2);
    assert_eq!(report.events[0].direction, EventDirection::Rising);
    assert_eq!(report.events[1].direction, EventDirection::Falling);
    assert!(
        report
            .scope
            .contains("no Krasis CoupledExecution event persistence claim")
    );
}

struct DecayEvent {
    nonfinite_event: bool,
}

impl DaeOperator for DecayEvent {
    fn dimension(&self) -> usize {
        1
    }
    fn residual(
        &self,
        _: &EvaluationContext,
        _: f64,
        state: &[f64],
        rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = rate[0] + state[0];
        Ok(())
    }
    fn jacobian_vector_product(
        &self,
        _: &EvaluationContext,
        _: f64,
        _: &[f64],
        _: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = state_direction[0] + rate_direction[0];
        Ok(())
    }
    fn event_count(&self) -> usize {
        1
    }
    fn event_values(
        &self,
        _: &EvaluationContext,
        _: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        output[0] = if self.nonfinite_event {
            f64::NAN
        } else {
            state[0] - 0.3
        };
        Ok(())
    }
}

#[test]
fn rejected_event_attempt_emits_nothing_and_nonfinite_events_refuse() {
    let state = BdfState {
        time: 1.0,
        values: vec![0.5],
        previous_values: Some(vec![1.0]),
        previous_step: Some(1.0),
        accepted_steps: 1,
    };
    let strict = BdfConfig {
        order: BdfOrder::Two,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        minimum_step: 0.1,
        maximum_step: 1.0,
        ..BdfConfig::default()
    };
    let report = check_event_state_from(
        &DecayEvent {
            nonfinite_event: false,
        },
        "decay-event/1",
        &EvaluationContext::reproducible(),
        state.clone(),
        1.0,
        &strict,
    )
    .unwrap();
    assert!(report.passed);
    assert!(!report.accepted_step);
    assert!(report.events.is_empty());
    assert!(report.input_state_unchanged);
    assert!(
        report
            .validate_from_state(
                &DecayEvent {
                    nonfinite_event: false,
                },
                "decay-event/1",
                &EvaluationContext::reproducible(),
                state,
                1.0,
                &strict,
            )
            .unwrap()
            .accepted
    );

    let error = check_event_state(
        &DecayEvent {
            nonfinite_event: true,
        },
        "nonfinite-event/1",
        &EvaluationContext::reproducible(),
        0.0,
        vec![1.0],
        0.1,
        &first_order_config(0.1),
    )
    .unwrap_err();
    assert_eq!(error.code, "KRASIS_VERIFY_EVENT_SOLVE");
    let encoded = serde_json::to_vec(&error).unwrap();
    let decoded: VerificationRefusal = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, error);
}

#[test]
fn malformed_probe_identities_and_dimensions_refuse() {
    let operator = TwoBlockLinear::new(0.1);
    assert!(
        check_cross_block_derivatives(
            &operator,
            "",
            &EvaluationContext::reproducible(),
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0e-3],
            ComparisonTolerance {
                absolute: 1.0e-8,
                relative: 1.0e-8,
            },
        )
        .is_err()
    );
    assert!(
        check_cross_block_derivatives(
            &operator,
            "two-block-linear/1",
            &EvaluationContext::reproducible(),
            &[0.0, 0.0],
            &[1.0],
            &[1.0e-3],
            ComparisonTolerance {
                absolute: 1.0e-8,
                relative: 1.0e-8,
            },
        )
        .is_err()
    );
    assert!(
        check_cross_block_derivatives(
            &operator,
            "two-block-linear/1",
            &EvaluationContext::reproducible(),
            &[0.0, 0.0],
            &[0.0, 1.0],
            &[1.0e-3],
            ComparisonTolerance {
                absolute: 1.0e-8,
                relative: 1.0e-8,
            },
        )
        .is_err()
    );
    assert!(
        check_strategy_work(
            &operator,
            "two-block-linear/1",
            &EvaluationContext::reproducible(),
            &[0.0, 0.0],
            &NewtonConfig::default(),
            ComparisonTolerance {
                absolute: 0.0,
                relative: 0.0,
            },
            WorkBudget {
                operator_evaluations: 1_000,
                linear_iterations: 0,
                nonlinear_iterations: 100,
                accepted_steps: 0,
                rejected_steps: 0,
            },
        )
        .is_err()
    );
}
