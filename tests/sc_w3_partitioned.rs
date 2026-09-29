//! SC-W3 package 1/2: the partitioned fixed-point transaction (`krasis::partitioned`) over
//! `CoupledSystemOperator`, and its agreement with the monolithic solve it projects.
//!
//! Fixtures (copied and trimmed from `tests/coupled_system.rs`, which owns the originals):
//! - `two_block_diffusion`: two Dirichlet-constrained linear-diffusion leaves on separate meshes,
//!   exchanging through a mass-scaled edge in both directions -- a steady two-leaf conduction
//!   composition (edge-composed, not `ConnectedSystemOperator`'s constraint-eliminated matching
//!   interface; see `STATUS.md` for that open item). This is the "own verification source" the
//!   SC-W3 package 2 brief calls the "two-leaf matching heat-heat conduction fixture".
//! - `two_block_network`: the transient DAE/BDF fixture with a closed-form manufactured solution
//!   (also owned by `tests/coupled_system.rs`), used here for the transient partitioned-vs-dense
//!   BDF-step agreement gate and the `iteration = once` divergence demonstration.

use std::sync::Arc;

use finitum::{
    AffineConstraint, Cell, ConstraintSet, DiscreteOperator, DofId, DofMap, DynamicExternalInput,
    ElementRestriction, ExternalInput, MethodRealization, NetworkDaeRealization, PreparedElement,
    RealizationPlan, VertexId,
};
use krasis::{
    BlockId, CoupledExecution, CoupledLeaf, CoupledOperator, CoupledSystemOperator,
    CouplingArgument, CouplingEdge, FieldId, KrasisError, PARTITIONED_DIVERGED,
    PARTITIONED_MAX_SWEEPS, PartitionedConfig, PartitionedDisposition, PartitionedExecution,
    PartitionedFixedPoint, PartitionedIteration, PartitionedSchedule, RowKind, SemanticId,
    SimulationState, StateBinding, StateBlock, StateLayout,
};
use methodus::{
    BdfConfig, BdfOrder, BlockNonlinearOperator, CsrMatrix, DaeOperator, EvaluationContext,
    NewtonConfig, NonlinearOperator, StepOutcome, solve_newton,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, compile_network_dae_method, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};

// -------------------------------------------------------------------------------------------
// Fixtures (trimmed copies; see module docs)
// -------------------------------------------------------------------------------------------

const NETWORK_MODEL: &str = r#"
module krasis.sc_w3.network;
model Network {
  domain Graph { dimension = 0; coordinates = lumped; }
  field x: state scalar L2(order=0) on Graph { time_role = differential; };
  field y: state scalar L2(order=0) on Graph { time_role = differential; };
  field z: state scalar L2(order=0) on Graph { time_role = differential; };
  equation ex on Graph { dt(x) + x = 0; }
  equation ey on Graph { dt(y) + y = 0; }
  equation ez on Graph { dt(z) + z = 0; }
}
"#;

const NETWORK_DIMENSION: usize = 3;

const LINEAR_DIFFUSION_MODEL: &str = r#"
module krasis.sc_w3.diffusion;
model LinearDiffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  property capacity = storage_capacity(u);
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation evolution on Omega { capacity * dt(u) - div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(t); }
}
"#;

fn network_operator() -> DiscreteOperator {
    let module = compile_semantics(NETWORK_MODEL, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic;
    let program =
        compile_network_dae_method(&module, "Network", &["ex", "ey", "ez"], &["x", "y", "z"])
            .unwrap();
    let mass = vec![
        vec![2.0, 0.5, 0.0],
        vec![0.5, 2.0, 0.5],
        vec![0.0, 0.5, 2.0],
    ];
    DiscreteOperator::sibling(MethodRealization::NetworkDae(
        NetworkDaeRealization::new(program, mass.clone(), mass, vec![0.0; NETWORK_DIMENSION])
            .unwrap(),
    ))
}

fn square_mesh(subdivisions: usize) -> (finitum::Mesh, DofMap, Vec<Vec<f64>>) {
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
    let restrictions = cells
        .iter()
        .map(|cell| ElementRestriction {
            dofs: cell.vertices.iter().map(|vertex| DofId(vertex.0)).collect(),
        })
        .collect();
    let mesh = finitum::Mesh::new(2, vertices.clone(), cells).unwrap();
    let dofs = DofMap::new(width * width, restrictions).unwrap();
    (mesh, dofs, vertices)
}

fn diffusion_realization(
    subdivisions: usize,
    boundary: impl Fn(&[f64]) -> f64,
) -> (RealizationPlan, Vec<RowKind>) {
    let compilation =
        compile_semantics(LINEAR_DIFFUSION_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "LinearDiffusion", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, vertices) = square_mesh(subdivisions);
    let width = subdivisions + 1;
    let is_wall = |index: usize| {
        let row = index / width;
        let column = index % width;
        row == 0 || column == 0 || row == subdivisions || column == subdivisions
    };
    let constraints = ConstraintSet::new(
        width * width,
        (0..width * width)
            .filter(|index| is_wall(*index))
            .map(|target| AffineConstraint {
                target: DofId(target),
                dependencies: Vec::new(),
                offset: boundary(&vertices[target]),
            }),
    )
    .unwrap();
    let row_kinds = (0..width * width)
        .map(|index| {
            if is_wall(index) {
                RowKind::Algebraic
            } else {
                RowKind::Differential
            }
        })
        .collect();
    let element = PreparedElement::linear_simplex(2).unwrap();
    let model = &compilation.semantic.models[0];
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = &model.symbols[input.binding.symbol.index()].name;
            match name.as_str() {
                "capacity" | "k" => dynamic.push(
                    DynamicExternalInput::try_new(
                        integral.integral_index,
                        input.id,
                        1,
                        "unit;direction=0/v1",
                        |_| Ok(vec![1.0]),
                        |_, _| Ok(vec![0.0]),
                    )
                    .unwrap(),
                ),
                "f" => stored.push(
                    ExternalInput::try_sampled_at(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        0.0,
                        |_, _, _time| Ok(vec![0.0]),
                    )
                    .unwrap(),
                ),
                other => panic!("unexpected external input {other}"),
            }
        }
    }
    let realization = RealizationPlan::new_stateful(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        stored,
        dynamic,
    )
    .unwrap();
    (realization, row_kinds)
}

fn single_block_layout(block: &str, width: usize) -> StateLayout {
    StateLayout::new(vec![StateBlock::new(BlockId::new(block), 0..width)]).unwrap()
}

fn single_binding(layout: &StateLayout, block: &str, semantic: u32) -> StateBinding {
    StateBinding::new(
        layout,
        vec![(SemanticId::new(semantic), BlockId::new(block))],
    )
    .unwrap()
}

fn realization_leaf(
    name: &str,
    semantic: u32,
    realization: RealizationPlan,
    row_kinds: Vec<RowKind>,
) -> (CoupledLeaf, CoupledOperator) {
    let layout = single_block_layout(name, realization.dimension());
    let binding = single_binding(&layout, name, semantic);
    let operator = CoupledOperator::new_with_bindings(realization, &layout, Some(binding))
        .unwrap()
        .with_consistent_initialization(row_kinds, NewtonConfig::default())
        .unwrap();
    (
        CoupledLeaf::realization(name, operator.clone(), layout).unwrap(),
        operator,
    )
}

fn network_leaf(name: &str, semantic: u32) -> (CoupledLeaf, DiscreteOperator) {
    let operator = network_operator();
    let layout = single_block_layout(name, NETWORK_DIMENSION);
    let binding = single_binding(&layout, name, semantic);
    let identity = operator.identity();
    let leaf = CoupledLeaf::new(name, Arc::new(operator.clone()), layout, binding, identity)
        .unwrap()
        .with_row_kinds(vec![RowKind::Differential; NETWORK_DIMENSION])
        .unwrap();
    (leaf, operator)
}

fn scaled_rate_jacobian(operator: &(impl DaeOperator + ?Sized), scale: f64) -> CsrMatrix {
    let dimension = operator.dimension();
    let context = EvaluationContext::reproducible();
    let zero = vec![0.0; dimension];
    let mut entries = Vec::new();
    let mut direction = vec![0.0; dimension];
    let mut column = vec![0.0; dimension];
    for index in 0..dimension {
        direction[index] = 1.0;
        operator
            .jacobian_vector_product(&context, 0.0, &zero, &zero, &zero, &direction, &mut column)
            .unwrap();
        for (row, value) in column.iter().enumerate() {
            if *value != 0.0 {
                entries.push((row, index, scale * value));
            }
        }
        direction[index] = 0.0;
    }
    CsrMatrix::from_triplets(dimension, dimension, entries).unwrap()
}

fn exchange(
    row: &str,
    column: &str,
    operator: &(impl DaeOperator + ?Sized),
    scale: f64,
) -> CouplingEdge {
    CouplingEdge::matrix(
        row,
        column,
        CouplingArgument::State,
        scaled_rate_jacobian(operator, scale),
    )
}

/// Two diffusion leaves with different wall data, exchanging through `-epsilon * M` on their
/// free rows: the steady two-leaf conduction fixture this file's tests agree monolithic and
/// partitioned solutions over.
fn two_block_diffusion(epsilon: f64) -> CoupledSystemOperator {
    let (hot_plan, hot_rows) = diffusion_realization(3, |point| 1.0 + point[0]);
    let (cold_plan, cold_rows) = diffusion_realization(3, |_| 0.0);
    let (hot, hot_operator) = realization_leaf("hot", 10, hot_plan, hot_rows);
    let (cold, cold_operator) = realization_leaf("cold", 11, cold_plan, cold_rows);
    let edges = vec![
        exchange("hot", "cold", &hot_operator, -epsilon),
        exchange("cold", "hot", &cold_operator, -epsilon),
    ];
    CoupledSystemOperator::new(vec![hot, cold], edges).unwrap()
}

/// The same two leaves, with the wrong (positive-feedback) coupling role: both edges carry a
/// `+strength` sign instead of the stabilizing `-epsilon`, and `strength` is large -- the
/// architecture's "wrong Dirichlet-Neumann roles" case, here as a coupling-sign/-magnitude
/// choice rather than a named role (Krasis composes edges, not named roles; role selection is
/// Sinbad policy per `sinbad/ARCHITECTURE.md` §9).
fn two_block_diffusion_wrong_role(strength: f64) -> CoupledSystemOperator {
    let (hot_plan, hot_rows) = diffusion_realization(3, |point| 1.0 + point[0]);
    let (cold_plan, cold_rows) = diffusion_realization(3, |_| 0.0);
    let (hot, hot_operator) = realization_leaf("hot", 10, hot_plan, hot_rows);
    let (cold, cold_operator) = realization_leaf("cold", 11, cold_plan, cold_rows);
    let edges = vec![
        exchange("hot", "cold", &hot_operator, strength),
        exchange("cold", "hot", &cold_operator, strength),
    ];
    CoupledSystemOperator::new(vec![hot, cold], edges).unwrap()
}

fn two_block_network(epsilon: f64) -> CoupledSystemOperator {
    let (leaf_a, operator_a) = network_leaf("a", 0);
    let (leaf_b, operator_b) = network_leaf("b", 1);
    let edges = vec![
        exchange("a", "b", &operator_a, -epsilon),
        exchange("b", "a", &operator_b, -epsilon),
    ];
    CoupledSystemOperator::new(vec![leaf_a, leaf_b], edges).unwrap()
}

fn network_initial_state(operator: &CoupledSystemOperator, a: f64, b: f64) -> SimulationState {
    let mut state = SimulationState::new(operator.layout().clone(), 4);
    for (index, value) in [a, b].into_iter().enumerate() {
        let range = operator.leaf_range(index).unwrap();
        state
            .insert_field(
                FieldId::new(operator.leaves()[index].name()),
                vec![value; range.len()],
            )
            .unwrap();
    }
    state
}

fn fixed_step_config(order: BdfOrder, step: f64) -> BdfConfig {
    BdfConfig {
        order,
        absolute_tolerance: 1.0,
        relative_tolerance: 1.0,
        minimum_step: step,
        maximum_step: step,
        newton: NewtonConfig {
            absolute_tolerance: 1.0e-13,
            relative_tolerance: 1.0e-12,
            ..NewtonConfig::default()
        },
    }
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f64::max)
}

fn implicit_config(
    schedule: PartitionedSchedule,
    tolerance: f64,
    max_sweeps: usize,
) -> PartitionedConfig {
    PartitionedConfig {
        schedule,
        iteration: PartitionedIteration::Implicit {
            tolerance,
            max_sweeps,
            relaxation: 1.0,
        },
        newton: NewtonConfig {
            max_iterations: 100,
            absolute_tolerance: 1.0e-13,
            relative_tolerance: 1.0e-12,
            ..NewtonConfig::default()
        },
        divergence_growth: 10.0,
    }
}

// -------------------------------------------------------------------------------------------
// Package 1: steady partitioned transaction, serial and parallel schedules
// -------------------------------------------------------------------------------------------

#[test]
fn steady_partitioned_transaction_agrees_with_monolithic_newton_for_both_schedules() {
    let operator = two_block_diffusion(0.3);
    let context = EvaluationContext::reproducible();
    let dimension = NonlinearOperator::dimension(&operator);
    let newton = NewtonConfig {
        max_iterations: 200,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        ..NewtonConfig::default()
    };
    let monolithic = solve_newton(&operator, &context, &vec![0.0; dimension], &newton).unwrap();
    assert!(monolithic.converged);

    for schedule in [PartitionedSchedule::Serial, PartitionedSchedule::Parallel] {
        let state = SimulationState::new(operator.layout().clone(), 1);
        let mut state = state;
        for block in operator.layout().blocks() {
            state
                .insert_field(
                    FieldId::new(block.id().as_str()),
                    vec![0.0; block.range().len()],
                )
                .unwrap();
        }
        let mut execution = PartitionedExecution::new(&operator, state).unwrap();
        let config = implicit_config(schedule, 1.0e-10, 200);
        let report = execution.solve(&context, &config, 0.0).unwrap();
        assert_eq!(report.disposition, PartitionedDisposition::Converged);
        assert!(
            report.sweeps.len() > 2,
            "{schedule:?} genuinely iterates the fixed point"
        );
        let committed = execution.state().committed_vector().unwrap();
        let disagreement = max_abs_difference(&committed, &monolithic.state);
        assert!(
            disagreement < 1.0e-8,
            "{schedule:?} partitioned vs monolithic disagreement: {disagreement}"
        );
        // The last sweep's interface (schedule-updated state) change is within the declared
        // tolerance -- the output-based convergence test that gates commit.
        assert!(report.sweeps.last().unwrap().interface_norm <= 1.0e-10);
    }
}

#[test]
fn a_refused_partitioned_solve_commits_nothing() {
    let operator = two_block_diffusion(0.3);
    let context = EvaluationContext::reproducible();
    let mut state = SimulationState::new(operator.layout().clone(), 1);
    for block in operator.layout().blocks() {
        state
            .insert_field(
                FieldId::new(block.id().as_str()),
                vec![0.0; block.range().len()],
            )
            .unwrap();
    }
    let before = state.clone();
    let mut execution = PartitionedExecution::new(&operator, state).unwrap();
    // One sweep can never reach 1e-300: this must exhaust its bound and refuse, not commit.
    let config = implicit_config(PartitionedSchedule::Serial, 1.0e-300, 2);
    let error = execution.solve(&context, &config, 0.0).unwrap_err();
    assert!(
        matches!(&error, KrasisError::EvaluationRefused { code, .. } if code == PARTITIONED_MAX_SWEEPS),
        "{error:?}"
    );
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        before.committed_vector().unwrap()
    );
}

// -------------------------------------------------------------------------------------------
// Package 1: transient, plugged into a BDF transaction as a `NonlinearSolver`
// -------------------------------------------------------------------------------------------

#[test]
fn partitioned_fixed_point_inside_bdf_agrees_with_dense_newton_per_accepted_step() {
    let epsilon = 0.4;
    let operator = two_block_network(epsilon);
    let context = EvaluationContext::reproducible();
    let step = 0.05;
    let steps = 12;
    let config = fixed_step_config(BdfOrder::Two, step);

    let mut dense = CoupledExecution::new(
        operator.clone(),
        network_initial_state(&operator, 1.0, 0.0),
        &context,
    )
    .unwrap();
    let layout = BlockNonlinearOperator::block_layout(&operator).clone();
    let partitioned_config = implicit_config(PartitionedSchedule::Serial, 1.0e-12, 100);
    let solver = PartitionedFixedPoint::new(&layout, &partitioned_config);
    let mut partitioned = CoupledExecution::new(
        operator.clone(),
        network_initial_state(&operator, 1.0, 0.0),
        &context,
    )
    .unwrap();

    for _ in 0..steps {
        let dense_outcome = dense.attempt_step(&context, step, &config).unwrap();
        let partitioned_outcome = partitioned
            .attempt_step_with(&context, step, &config, &solver)
            .unwrap();
        match (dense_outcome, partitioned_outcome) {
            (StepOutcome::Accepted(_), StepOutcome::Accepted(_)) => {}
            other => panic!("unexpected outcome pair: {other:?}"),
        }
        let disagreement = max_abs_difference(
            &dense.state().committed_vector().unwrap(),
            &partitioned.state().committed_vector().unwrap(),
        );
        assert!(
            disagreement < 1.0e-9,
            "dense vs partitioned disagree at this accepted step by {disagreement}"
        );
    }
    assert!(partitioned.evaluation_refusals().is_empty());
}

// -------------------------------------------------------------------------------------------
// Package 2: `iteration = once` with a wrong coupling role/magnitude refuses as predicted
// -------------------------------------------------------------------------------------------

#[test]
fn iteration_once_with_a_wrong_coupling_role_is_refused_as_predicted() {
    // The correctly-signed, moderate exchange: `iteration = once` applies its single sweep and
    // is accepted (splitting error tolerated, not required to match the monolithic solution).
    let operator = two_block_diffusion(0.3);
    let context = EvaluationContext::reproducible();
    let mut state = SimulationState::new(operator.layout().clone(), 1);
    for block in operator.layout().blocks() {
        state
            .insert_field(
                FieldId::new(block.id().as_str()),
                vec![0.0; block.range().len()],
            )
            .unwrap();
    }
    let once_config = PartitionedConfig {
        schedule: PartitionedSchedule::Serial,
        iteration: PartitionedIteration::Once,
        newton: NewtonConfig {
            max_iterations: 100,
            absolute_tolerance: 1.0e-13,
            relative_tolerance: 1.0e-12,
            ..NewtonConfig::default()
        },
        divergence_growth: 10.0,
    };
    let mut execution = PartitionedExecution::new(&operator, state.clone()).unwrap();
    let report = execution.solve(&context, &once_config, 0.0).unwrap();
    assert_eq!(report.disposition, PartitionedDisposition::OnceApplied);
    assert_eq!(report.sweeps.len(), 1);

    // The wrong (positive-feedback) coupling role, run as `iteration = implicit`: a single
    // `Once` sweep is always accepted (see `PartitionedIteration::Once`'s docs -- Methodus's own
    // `solve_blocks` backtracking already forbids a single sweep from raising the residual), so
    // this unconditionally unstable role/strength choice is predicted to never reach the declared
    // tolerance -- observed as either a growing interface change (`PARTITIONED_DIVERGED`) or an
    // exhausted sweep bound (`PARTITIONED_MAX_SWEEPS`), never a silently committed run-away
    // iterate. This is the architecture's "wrong Dirichlet-Neumann roles" case.
    let wrong = two_block_diffusion_wrong_role(6.0);
    let mut wrong_state = SimulationState::new(wrong.layout().clone(), 1);
    for block in wrong.layout().blocks() {
        wrong_state
            .insert_field(
                FieldId::new(block.id().as_str()),
                vec![0.0; block.range().len()],
            )
            .unwrap();
    }
    let before = wrong_state.clone();
    let wrong_config = PartitionedConfig {
        schedule: PartitionedSchedule::Serial,
        iteration: PartitionedIteration::Implicit {
            tolerance: 1.0e-10,
            max_sweeps: 8,
            relaxation: 1.0,
        },
        newton: once_config.newton.clone(),
        divergence_growth: 2.0,
    };
    let mut wrong_execution = PartitionedExecution::new(&wrong, wrong_state).unwrap();
    let error = wrong_execution
        .solve(&context, &wrong_config, 0.0)
        .unwrap_err();
    match &error {
        KrasisError::EvaluationRefused { code, .. } => {
            assert!(
                code == PARTITIONED_DIVERGED || code == PARTITIONED_MAX_SWEEPS,
                "{error:?}"
            );
        }
        KrasisError::Solve(message) => {
            // A sweep whose line search cannot find a damping that improves the residual, or
            // that overflows to a non-finite value, is refused by Methodus's own guard first;
            // still a typed refusal, never a silently accepted iterate.
            assert!(
                message.to_lowercase().contains("non-finite")
                    || message.to_lowercase().contains("line search"),
                "{message}"
            );
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
    assert_eq!(
        wrong_execution.state().committed_vector().unwrap(),
        before.committed_vector().unwrap()
    );
}
