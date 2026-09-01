//! E6: block-operator composition over one Finitum product space, driven through Krasis's
//! transactional state (`BlockLinearExecution`) via `methodus::solve_minres`.
//!
//! Fixture: the same vector-P2/scalar-P1 saddle-point `MixedOperator` Finitum's own
//! `tests/sv2b_mixed.rs` builds -- a generic `GradientGradient` diagonal block on a vector field
//! coupled to a scalar field by `DivergenceValue` (the `[[A, B^T], [B, 0]]` shape) -- composed
//! into a Krasis `BlockLinearExecution` over `krasis::block_state_layout`. No named physics
//! anywhere in this file, matching `finitum::mixed`'s own convention.
//!
//! `MixedOperator` is representation-only and unconstrained (SV2-B1: no Dirichlet elimination),
//! so its exact kernel is larger than the single `field_b` constant-pressure mode
//! `BlockNullspaceCandidate` declares: every `GradientGradient` stiffness block has zero row
//! sums (a textbook property of a consistent FEM Laplacian, regardless of boundary conditions),
//! so each component-constant mode of `field_a` is *also* an exact kernel vector here --
//! finitum's own `sv2b_mixed.rs` test
//! (`pressure_like_nullspace_candidate_round_trips_and_matches_expected_structure`) makes the
//! same point explicitly (`verify_in_kernel` against the unconstrained operator is `false`).
//!
//! [`minres_reaches_the_expected_solution_with_rollback_restart_and_history_evidence`] avoids
//! that gap honestly: every right-hand side is built as `operator * probe`, which for a
//! symmetric operator always lies in `range(operator) = ker(operator)^perp` exactly, whatever
//! `probe` is -- so MINRES from a zero initial guess, with *no* nullspace projector, converges
//! to a genuine solution, verified by an independently recomputed raw residual.
//! [`nullspace_projector_path_converges_the_declared_reduced_residual`] separately exercises
//! the `BlockNullspaceCandidate` -> `ConstantModeProjector` -> `solve_minres` composition the
//! task requires, verifying exactly what that path's contract promises on an operator whose
//! true kernel is larger than what it declares: convergence of the *projected* residual, not
//! the raw one (using the declared projector on a bigger true kernel would otherwise silently
//! bias the raw solution, which the first test's projector-free construction sidesteps).

use finitum::{
    BlockCoupling, BlockNullspaceCandidate, Cell, CouplingKind, FieldSpec, Mesh, MixedOperator,
    MixedSpace, VertexId,
};
use krasis::{
    BlockId, BlockLinearExecution, FieldId, KrasisError, SimulationState, StateBlock, StateLayout,
    TransactionPhase, block_state_layout,
};
use methodus::{EvaluationContext, LinearOperator, MinresConfig, NullspaceProjector};
use scientia::SymbolId;

const FIELD_A: SymbolId = SymbolId(0);
const FIELD_B: SymbolId = SymbolId(1);

fn unit_square_mesh(subdivisions: usize) -> Mesh {
    let width = subdivisions + 1;
    let vertices = (0..=subdivisions)
        .flat_map(|row| {
            (0..=subdivisions).map(move |column| {
                vec![
                    column as f64 / subdivisions as f64,
                    row as f64 / subdivisions as f64,
                ]
            })
        })
        .collect::<Vec<_>>();
    let cells = (0..subdivisions)
        .flat_map(|row| {
            (0..subdivisions).flat_map(move |column| {
                let lower_left = row * width + column;
                let lower_right = lower_left + 1;
                let upper_left = lower_left + width;
                let upper_right = upper_left + 1;
                [
                    Cell {
                        vertices: vec![
                            VertexId(lower_left),
                            VertexId(lower_right),
                            VertexId(upper_right),
                        ],
                    },
                    Cell {
                        vertices: vec![
                            VertexId(lower_left),
                            VertexId(upper_right),
                            VertexId(upper_left),
                        ],
                    },
                ]
            })
        })
        .collect::<Vec<_>>();
    Mesh::new(2, vertices, cells).unwrap()
}

fn fixture_space(subdivisions: usize) -> MixedSpace {
    MixedSpace::new(
        unit_square_mesh(subdivisions),
        vec![
            FieldSpec {
                symbol: FIELD_A,
                order: 2,
                components: 2,
            },
            FieldSpec {
                symbol: FIELD_B,
                order: 1,
                components: 1,
            },
        ],
    )
    .unwrap()
}

fn fixture_couplings() -> Vec<BlockCoupling> {
    vec![
        BlockCoupling {
            test: FIELD_A,
            trial: FIELD_A,
            kind: CouplingKind::GradientGradient,
            scale: 1.0,
        },
        BlockCoupling {
            test: FIELD_A,
            trial: FIELD_B,
            kind: CouplingKind::DivergenceValue,
            scale: 1.0,
        },
    ]
}

fn zero_state(layout: &StateLayout) -> SimulationState {
    let mut state = SimulationState::new(layout.clone(), 4);
    for block in layout.blocks() {
        state
            .insert_field(
                FieldId::new(block.id().as_str()),
                vec![0.0; block.range().len()],
            )
            .unwrap();
    }
    state
}

fn pseudo_random_vector(dimension: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..dimension)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * bits - 1.0
        })
        .collect()
}

/// A right-hand side guaranteed consistent for `operator` (`operator * probe`, which for a
/// symmetric operator always lies in `range(operator) = ker(operator)^perp`, whatever `probe`
/// is) -- see the module doc comment for why this file never compares against a hand-picked
/// closed form.
fn consistent_right_hand_side(operator: &MixedOperator, seed: u64) -> Vec<f64> {
    let probe = pseudo_random_vector(operator.dimension(), seed);
    let mut rhs = vec![0.0; operator.dimension()];
    operator
        .apply(&EvaluationContext::reproducible(), &probe, &mut rhs)
        .unwrap();
    rhs
}

/// Independent residual recomputation, sharing no state with `solve_minres`'s own internal
/// convergence estimate.
fn residual_norm(operator: &MixedOperator, solution: &[f64], right_hand_side: &[f64]) -> f64 {
    let mut action = vec![0.0; operator.dimension()];
    operator
        .apply(&EvaluationContext::reproducible(), solution, &mut action)
        .unwrap();
    action
        .iter()
        .zip(right_hand_side)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn minres_reaches_the_expected_solution_with_rollback_restart_and_history_evidence() {
    let operator = MixedOperator::new(fixture_space(2), fixture_couplings()).unwrap();
    let (state_layout, state_binding) = block_state_layout(operator.space().layout()).unwrap();
    assert_eq!(state_binding.blocks().count(), 2);

    let mut execution = BlockLinearExecution::new(&operator, zero_state(&state_layout)).unwrap();

    let context = EvaluationContext::reproducible();
    let config = MinresConfig::default();

    // --- Solve 1: converges, commits, becomes the new committed state. ---
    let rhs_1 = consistent_right_hand_side(&operator, 7);
    let report = execution
        .solve(&context, &rhs_1, None, None, &config, 1.0)
        .unwrap();
    assert!(report.converged);
    assert!(residual_norm(&operator, &report.solution, &rhs_1) < 1.0e-6);
    let solution_1 = report.solution.clone();
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        solution_1,
        "the committed state is exactly the reported MINRES solution"
    );
    assert_eq!(execution.state().time(), 1.0);
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);

    let checkpoint_after_solve_1 = execution.checkpoint().unwrap();

    // --- Solve 2: a different consistent right-hand side, also converges and commits. ---
    let rhs_2 = consistent_right_hand_side(&operator, 13);
    let report_2 = execution
        .solve(&context, &rhs_2, None, None, &config, 2.0)
        .unwrap();
    assert!(report_2.converged);
    assert!(residual_norm(&operator, &report_2.solution, &rhs_2) < 1.0e-6);
    let solution_2 = report_2.solution.clone();
    assert_eq!(execution.state().committed_vector().unwrap(), solution_2);
    assert_ne!(solution_1, solution_2);

    // History evidence: the prior committed solution is retained.
    assert_eq!(
        execution.state().history_vector(0).unwrap().unwrap(),
        solution_1
    );

    // --- Rollback evidence: a config too tight to converge in one iteration never commits. ---
    let unreachable = MinresConfig {
        max_iterations: 1,
        absolute_tolerance: 1.0e-14,
        relative_tolerance: 1.0e-14,
    };
    let error = execution
        .solve(&context, &rhs_1, None, None, &unreachable, 3.0)
        .unwrap_err();
    assert!(matches!(error, KrasisError::Solve(_)), "{error:?}");
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        solution_2,
        "a rejected solve must never move the committed state"
    );
    assert_eq!(execution.state().time(), 2.0);
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);

    // --- Restart evidence: restoring the earlier checkpoint returns to solution 1. ---
    execution.restore(&checkpoint_after_solve_1).unwrap();
    assert_eq!(execution.state().committed_vector().unwrap(), solution_1);
    assert_eq!(execution.state().time(), 1.0);
}

/// Exercises the `finitum::BlockNullspaceCandidate` -> `methodus::ConstantModeProjector` ->
/// `methodus::solve_minres` composition end to end through `BlockLinearExecution`. As the
/// module doc comment explains, this fixture's true kernel is larger than the single mode this
/// candidate declares, so the correct verification is convergence of the *projected* residual
/// (exactly what `solve_minres`'s nullspace-projector contract promises), not the raw one.
#[test]
fn nullspace_projector_path_converges_the_declared_reduced_residual() {
    let operator = MixedOperator::new(fixture_space(2), fixture_couplings()).unwrap();
    let (state_layout, _) = block_state_layout(operator.space().layout()).unwrap();
    let mut execution = BlockLinearExecution::new(&operator, zero_state(&state_layout)).unwrap();

    let candidate = BlockNullspaceCandidate::constant(FIELD_B, "field_a carries no Dirichlet data");
    let mode = candidate.resolve(operator.space().layout()).unwrap();

    let context = EvaluationContext::reproducible();
    let rhs = consistent_right_hand_side(&operator, 21);
    let report = execution
        .solve(
            &context,
            &rhs,
            None,
            Some(mode.projector()),
            &MinresConfig::default(),
            1.0,
        )
        .unwrap();
    assert!(report.converged);

    let mut raw_residual = vec![0.0; operator.dimension()];
    operator
        .apply(&context, &report.solution, &mut raw_residual)
        .unwrap();
    for (value, expected) in raw_residual.iter_mut().zip(&rhs) {
        *value -= *expected;
    }
    mode.projector()
        .project(&context, &mut raw_residual)
        .unwrap();
    let projected_residual_norm = raw_residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(
        projected_residual_norm < 1.0e-6,
        "projected residual = {projected_residual_norm}"
    );
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        report.solution
    );
}

#[test]
fn a_non_converged_solve_never_commits_and_state_stays_untouched() {
    let operator = MixedOperator::new(fixture_space(1), fixture_couplings()).unwrap();
    let (layout, _) = block_state_layout(operator.space().layout()).unwrap();
    let mut execution = BlockLinearExecution::new(&operator, zero_state(&layout)).unwrap();
    let before = execution.state().committed_vector().unwrap();

    let rhs = consistent_right_hand_side(&operator, 5);

    let unreachable = MinresConfig {
        max_iterations: 1,
        absolute_tolerance: 1.0e-14,
        relative_tolerance: 1.0e-14,
    };
    let error = execution
        .solve(
            &EvaluationContext::reproducible(),
            &rhs,
            None,
            None,
            &unreachable,
            1.0,
        )
        .unwrap_err();
    assert!(matches!(error, KrasisError::Solve(_)), "{error:?}");
    assert_eq!(execution.state().committed_vector().unwrap(), before);
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);
}

#[test]
fn construction_refuses_a_dimension_mismatched_state() {
    let operator = MixedOperator::new(fixture_space(1), fixture_couplings()).unwrap();
    let small_layout = StateLayout::new(vec![StateBlock::new(BlockId::new("only"), 0..3)]).unwrap();
    let mut state = SimulationState::new(small_layout, 0);
    state
        .insert_field(FieldId::new("only"), vec![0.0; 3])
        .unwrap();
    let error = BlockLinearExecution::new(&operator, state).unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
}

#[test]
fn construction_refuses_a_block_count_mismatch() {
    let operator = MixedOperator::new(fixture_space(1), fixture_couplings()).unwrap();
    let dimension = operator.dimension();
    // Same total width as the operator, but split into one block instead of its two.
    let one_block_layout =
        StateLayout::new(vec![StateBlock::new(BlockId::new("only"), 0..dimension)]).unwrap();
    let mut state = SimulationState::new(one_block_layout, 0);
    state
        .insert_field(FieldId::new("only"), vec![0.0; dimension])
        .unwrap();
    let error = BlockLinearExecution::new(&operator, state).unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
}

#[test]
fn restore_refuses_a_checkpoint_from_a_different_operator_identity() {
    let operator_a = MixedOperator::new(fixture_space(1), fixture_couplings()).unwrap();
    let operator_b = MixedOperator::new(fixture_space(2), fixture_couplings()).unwrap();

    let (layout_a, _) = block_state_layout(operator_a.space().layout()).unwrap();
    let (layout_b, _) = block_state_layout(operator_b.space().layout()).unwrap();

    let mut execution_a = BlockLinearExecution::new(&operator_a, zero_state(&layout_a)).unwrap();
    let execution_b = BlockLinearExecution::new(&operator_b, zero_state(&layout_b)).unwrap();

    let checkpoint_b = execution_b.checkpoint().unwrap();
    let error = execution_a.restore(&checkpoint_b).unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
}
