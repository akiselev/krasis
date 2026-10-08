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
//!   BDF-step agreement gate and its `iteration = once` negative control.
//! - `TwoBlockLinear`: a four-unknown linear system with a contractive serial splitting and
//!   irrational coefficients, for the rounding-floor refusal path (no Finitum involved).

use std::sync::Arc;

use finitum::{
    AffineConstraint, Cell, ConstraintSet, DiscreteOperator, DofId, DofMap, DynamicExternalInput,
    ElementRestriction, ExternalInput, MethodRealization, NetworkDaeRealization, PreparedElement,
    RealizationPlan, VertexId,
};
use krasis::{
    BlockId, CoupledExecution, CoupledLeaf, CoupledOperator, CoupledSystemOperator,
    CouplingArgument, CouplingEdge, FieldId, KrasisError, PARTITIONED_DIVERGED,
    PARTITIONED_MAX_SWEEPS, PARTITIONED_REFUSAL_ORIGIN, PartitionedConfig, PartitionedDisposition,
    PartitionedExecution, PartitionedFixedPoint, PartitionedIteration, PartitionedSchedule,
    RowKind, SemanticId, SimulationState, StateBinding, StateBlock, StateLayout, run_partitioned,
};
use methodus::{
    AccelerationMethod, BdfConfig, BdfOrder, BlockLayout, BlockNonlinearOperator, BlockSpec,
    CsrMatrix, DaeOperator, EvaluationContext, NewtonConfig, NonlinearOperator, NumericError,
    SolveError, StepOutcome, solve_newton,
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

/// The same two leaves with the coupling role reversed: both edges carry `+strength` instead of
/// the stabilizing `-epsilon` -- the architecture's "wrong Dirichlet-Neumann roles" case, here as
/// a coupling-sign/-magnitude choice rather than a named role (Krasis composes edges, not named
/// roles; role selection is Sinbad policy per `sinbad/ARCHITECTURE.md` §9). Whether the serial
/// splitting of this fixture contracts depends on the exchange magnitude alone (its
/// error-propagation map is the product of the two off-diagonal blocks, so the sign cancels):
/// its spectral radius is `(strength / s*)^2` with `s*` about 29.4 on this mesh, so strength 6
/// is a convergent configuration and strength 60 is not. The tests assert the regime they use
/// independently of the driver (`serial_schedule_spectral_radius`).
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

/// A committed all-zero state over the operator's own layout.
fn zero_state(operator: &CoupledSystemOperator) -> SimulationState {
    let mut state = SimulationState::new(operator.layout().clone(), 1);
    for block in operator.layout().blocks() {
        state
            .insert_field(
                FieldId::new(block.id().as_str()),
                vec![0.0; block.range().len()],
            )
            .unwrap();
    }
    state
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
        assert!(report.sweeps.last().unwrap().state_change_norm <= 1.0e-10);
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
// Package 1: a `Once` solver inside BDF is accepted by declaration, not as a projection
// -------------------------------------------------------------------------------------------

#[test]
fn a_once_solver_inside_bdf_is_accepted_by_declaration_but_is_not_a_projection_of_dense_newton() {
    let operator = two_block_network(0.4);
    let context = EvaluationContext::reproducible();
    let step = 0.05;
    let config = fixed_step_config(BdfOrder::Two, step);

    let mut dense = CoupledExecution::new(
        operator.clone(),
        network_initial_state(&operator, 1.0, 0.0),
        &context,
    )
    .unwrap();
    let layout = BlockNonlinearOperator::block_layout(&operator).clone();
    let once_config = PartitionedConfig {
        schedule: PartitionedSchedule::Serial,
        iteration: PartitionedIteration::Once,
        newton: implicit_config(PartitionedSchedule::Serial, 1.0e-12, 100).newton,
        divergence_growth: 10.0,
    };
    let solver = PartitionedFixedPoint::new(&layout, &once_config);
    let mut once = CoupledExecution::new(
        operator.clone(),
        network_initial_state(&operator, 1.0, 0.0),
        &context,
    )
    .unwrap();

    let mut worst = 0.0_f64;
    for _ in 0..12 {
        let dense_outcome = dense.attempt_step(&context, step, &config).unwrap();
        let once_outcome = once
            .attempt_step_with(&context, step, &config, &solver)
            .unwrap();
        assert!(
            matches!(
                (&dense_outcome, &once_outcome),
                (StepOutcome::Accepted(_), StepOutcome::Accepted(_))
            ),
            "unexpected outcome pair: {dense_outcome:?} / {once_outcome:?}"
        );
        worst = worst.max(max_abs_difference(
            &dense.state().committed_vector().unwrap(),
            &once.state().committed_vector().unwrap(),
        ));
    }
    assert!(once.evaluation_refusals().is_empty());
    // The negative control for the implicit agreement gate above: one schedule sweep per step
    // is accepted (declared regime), and it is demonstrably not dense Newton.
    assert!(
        worst > 1.0e-9,
        "a single sweep per step must not reproduce dense Newton: worst disagreement {worst:e}"
    );
}

// -------------------------------------------------------------------------------------------
// Package 1: configuration and rounding-floor refusals
// -------------------------------------------------------------------------------------------

#[test]
fn partitioned_acceleration_is_refused_as_invalid_configuration_and_commits_nothing() {
    let operator = two_block_diffusion(0.3);
    let context = EvaluationContext::reproducible();
    let dimension = NonlinearOperator::dimension(&operator);
    let mut config = implicit_config(PartitionedSchedule::Serial, 1.0e-10, 50);
    config.newton.acceleration = Some(AccelerationMethod::Aitken {
        initial_factor: 0.5,
    });
    let error = run_partitioned(&operator, &context, &vec![0.0; dimension], &config).unwrap_err();
    assert!(
        matches!(&error, SolveError::InvalidConfiguration { reason } if reason.contains("acceleration")),
        "{error:?}"
    );

    let state = zero_state(&operator);
    let before = state.committed_vector().unwrap();
    let mut execution = PartitionedExecution::new(&operator, state).unwrap();
    let error = execution.solve(&context, &config, 0.0).unwrap_err();
    assert!(
        matches!(&error, KrasisError::Solve(message) if message.contains("acceleration")),
        "{error:?}"
    );
    assert_eq!(execution.state().committed_vector().unwrap(), before);

    // Methodus's fixed relaxation is refused the same way: Krasis's `relaxation` is the one axis.
    config.newton.acceleration = Some(AccelerationMethod::FixedRelaxation { factor: 0.7 });
    assert!(matches!(
        run_partitioned(&operator, &context, &vec![0.0; dimension], &config),
        Err(SolveError::InvalidConfiguration { .. })
    ));
}

/// A two-block linear system `J x = b` with a contractive serial splitting and irrational
/// coefficients (its residual never reaches an exact zero): rounding-floor scaffolding.
struct TwoBlockLinear {
    layout: BlockLayout,
    jacobian: [[f64; 4]; 4],
    rhs: [f64; 4],
}

impl TwoBlockLinear {
    fn new() -> Self {
        let layout = BlockLayout::new(vec![
            BlockSpec {
                name: "p".into(),
                length: 2,
                residual_scale: 1.0,
            },
            BlockSpec {
                name: "q".into(),
                length: 2,
                residual_scale: 1.0,
            },
        ])
        .unwrap();
        let s = std::f64::consts::SQRT_2 / 3.0;
        let jacobian = [
            [1.0 + std::f64::consts::PI / 10.0, 0.1, s, 0.0],
            [0.1, 1.3, 0.0, s],
            [s, 0.0, 1.1, 0.2],
            [0.0, s, 0.2, 1.0 + std::f64::consts::E / 10.0],
        ];
        Self {
            layout,
            jacobian,
            rhs: [1.0 / 3.0, 2.0 / 3.0, -1.0 / 7.0, 0.5],
        }
    }
}

impl NonlinearOperator for TwoBlockLinear {
    fn dimension(&self) -> usize {
        4
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.jacobian[row]
                .iter()
                .zip(state)
                .map(|(a, x)| a * x)
                .sum::<f64>()
                - self.rhs[row];
        }
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.jacobian[row]
                .iter()
                .zip(direction)
                .map(|(a, d)| a * d)
                .sum();
        }
        Ok(())
    }
}

impl BlockNonlinearOperator for TwoBlockLinear {
    fn block_layout(&self) -> &BlockLayout {
        &self.layout
    }
}

#[test]
fn a_stall_at_the_rounding_floor_is_refused_max_sweeps_never_diverged() {
    let operator = TwoBlockLinear::new();
    let context = EvaluationContext::reproducible();
    // Tolerances no floating-point iterate can meet: the state change must reach exactly zero
    // while Methodus's per-sweep residual threshold is about 1e-300, so the contractive
    // iteration reaches the rounding floor and can then only stall (a failed line search on
    // rounding noise) or exhaust its bound -- never be judged diverging.
    let config = PartitionedConfig {
        schedule: PartitionedSchedule::Serial,
        iteration: PartitionedIteration::Implicit {
            tolerance: 0.0,
            max_sweeps: 80,
            relaxation: 1.0,
        },
        newton: NewtonConfig {
            absolute_tolerance: 0.0,
            relative_tolerance: 1.0e-300,
            ..NewtonConfig::default()
        },
        divergence_growth: 1.5,
    };
    let error = run_partitioned(&operator, &context, &[0.0; 4], &config).unwrap_err();
    match &error {
        SolveError::Numeric(NumericError::Evaluation {
            code,
            origin,
            message,
        }) => {
            assert_eq!(origin, PARTITIONED_REFUSAL_ORIGIN);
            assert_eq!(code, PARTITIONED_MAX_SWEEPS, "{message}");
            // The stall path, not bound exhaustion: a failed line search on rounding noise
            // after a sweep at the floor.
            assert!(
                message.contains("stalled at the rounding floor"),
                "{message}"
            );
        }
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

// -------------------------------------------------------------------------------------------
// Package 1: checkpoints are bound to the operator identity and the state to the layout
// -------------------------------------------------------------------------------------------

#[test]
fn a_partitioned_checkpoint_restores_only_into_the_same_operator() {
    let context = EvaluationContext::reproducible();
    let operator = two_block_diffusion(0.3);
    let mut execution = PartitionedExecution::new(&operator, zero_state(&operator)).unwrap();
    let config = implicit_config(PartitionedSchedule::Serial, 1.0e-10, 50);
    execution.solve(&context, &config, 1.0).unwrap();
    let checkpoint = execution.checkpoint().unwrap();
    assert_eq!(checkpoint.operator_identity, execution.operator_identity());
    assert_eq!(checkpoint.operator_identity, operator.identity());

    // The same operator: restores atomically.
    let mut sibling = PartitionedExecution::new(&operator, zero_state(&operator)).unwrap();
    sibling.restore(&checkpoint).unwrap();
    assert_eq!(
        sibling.state().committed_vector().unwrap(),
        execution.state().committed_vector().unwrap()
    );

    // The same leaves and layout with different edge content is a different system: refused,
    // and the target execution's state is untouched.
    let other = two_block_diffusion(0.5);
    assert_eq!(other.layout().identity(), operator.layout().identity());
    assert_ne!(other.identity(), operator.identity());
    let mut foreign = PartitionedExecution::new(&other, zero_state(&other)).unwrap();
    let before = foreign.state().committed_vector().unwrap();
    let error = foreign.restore(&checkpoint).unwrap_err();
    assert!(
        matches!(&error, KrasisError::InvalidCoupling(message) if message.contains("identity")),
        "{error:?}"
    );
    assert_eq!(foreign.state().committed_vector().unwrap(), before);
}

#[test]
fn a_partitioned_execution_refuses_a_state_over_a_different_layout() {
    let operator = two_block_diffusion(0.3);
    let width = NonlinearOperator::dimension(&operator);
    // Same width and block names, different block boundaries: not the operator's layout.
    let layout = StateLayout::new(vec![
        StateBlock::new(BlockId::new("hot"), 0..width / 2 - 1),
        StateBlock::new(BlockId::new("cold"), width / 2 - 1..width),
    ])
    .unwrap();
    let mut state = SimulationState::new(layout.clone(), 1);
    for block in layout.blocks() {
        state
            .insert_field(
                FieldId::new(block.id().as_str()),
                vec![0.0; block.range().len()],
            )
            .unwrap();
    }
    let error = PartitionedExecution::new(&operator, state).unwrap_err();
    assert!(
        matches!(&error, KrasisError::InvalidCoupling(message) if message.contains("layout")),
        "{error:?}"
    );
}

// -------------------------------------------------------------------------------------------
// Package 2: `iteration = once` is accepted by declaration, and a non-contractive (wrong-role)
// schedule is refused `PARTITIONED_DIVERGED` as predicted
// -------------------------------------------------------------------------------------------

/// The dense coupled Jacobian, assembled column by column from the operator's own
/// Jacobian-vector products at the zero state (both diffusion fixtures are linear).
fn dense_jacobian(operator: &CoupledSystemOperator) -> Vec<Vec<f64>> {
    let context = EvaluationContext::reproducible();
    let dimension = NonlinearOperator::dimension(operator);
    let zero = vec![0.0; dimension];
    let mut jacobian = vec![vec![0.0; dimension]; dimension];
    let mut direction = vec![0.0; dimension];
    let mut column = vec![0.0; dimension];
    for column_index in 0..dimension {
        direction[column_index] = 1.0;
        NonlinearOperator::jacobian_vector_product(
            operator,
            &context,
            &zero,
            &direction,
            &mut column,
        )
        .unwrap();
        for (row, value) in column.iter().enumerate() {
            jacobian[row][column_index] = *value;
        }
        direction[column_index] = 0.0;
    }
    jacobian
}

fn submatrix(
    matrix: &[Vec<f64>],
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) -> Vec<Vec<f64>> {
    rows.map(|row| matrix[row][columns.clone()].to_vec())
        .collect()
}

/// Gaussian elimination with partial pivoting: test scaffolding for the dense block solves of
/// `serial_schedule_spectral_radius`, deliberately independent of Methodus.
fn solve_dense(matrix: &[Vec<f64>], right_hand_side: &[f64]) -> Vec<f64> {
    let dimension = right_hand_side.len();
    let mut a = matrix.to_vec();
    let mut b = right_hand_side.to_vec();
    for pivot_column in 0..dimension {
        let pivot_row = (pivot_column..dimension)
            .max_by(|&left, &right| {
                a[left][pivot_column]
                    .abs()
                    .total_cmp(&a[right][pivot_column].abs())
            })
            .unwrap();
        a.swap(pivot_column, pivot_row);
        b.swap(pivot_column, pivot_row);
        assert!(
            a[pivot_column][pivot_column].abs() > 1.0e-14,
            "singular block"
        );
        for row in (pivot_column + 1)..dimension {
            let factor = a[row][pivot_column] / a[pivot_column][pivot_column];
            if factor != 0.0 {
                let pivot_values = a[pivot_column].clone();
                for (value, pivot_value) in a[row].iter_mut().zip(&pivot_values).skip(pivot_column)
                {
                    *value -= factor * pivot_value;
                }
                b[row] -= factor * b[pivot_column];
            }
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut sum = b[row];
        for column in (row + 1)..dimension {
            sum -= a[row][column] * solution[column];
        }
        solution[row] = sum / a[row][row];
    }
    solution
}

fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}

fn l2(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Spectral radius of the serial schedule's error-propagation map on the second leaf,
/// `G = A_1^{-1} B_10 A_0^{-1} B_01` (block Gauss-Seidel over the dense coupled Jacobian), by
/// power iteration. Independent of `krasis::partitioned`: the serial splitting is contractive
/// exactly when this is below one.
fn serial_schedule_spectral_radius(operator: &CoupledSystemOperator) -> f64 {
    let jacobian = dense_jacobian(operator);
    let first = operator.leaf_range(0).unwrap();
    let second = operator.leaf_range(1).unwrap();
    let a0 = submatrix(&jacobian, first.clone(), first.clone());
    let b01 = submatrix(&jacobian, first.clone(), second.clone());
    let a1 = submatrix(&jacobian, second.clone(), second.clone());
    let b10 = submatrix(&jacobian, second.clone(), first.clone());
    let mut vector: Vec<f64> = (0..second.len()).map(|index| 1.0 + index as f64).collect();
    let norm = l2(&vector);
    vector.iter_mut().for_each(|value| *value /= norm);
    let mut radius = 0.0;
    for _ in 0..400 {
        let first_error: Vec<f64> = solve_dense(&a0, &matvec(&b01, &vector))
            .into_iter()
            .map(|value| -value)
            .collect();
        let image: Vec<f64> = solve_dense(&a1, &matvec(&b10, &first_error))
            .into_iter()
            .map(|value| -value)
            .collect();
        radius = l2(&image);
        vector = image.into_iter().map(|value| value / radius).collect();
    }
    radius
}

#[test]
fn iteration_once_applies_one_accepted_sweep_that_is_not_a_projection_of_the_monolithic_solve() {
    let operator = two_block_diffusion(0.3);
    let context = EvaluationContext::reproducible();
    let dimension = NonlinearOperator::dimension(&operator);
    let newton = implicit_config(PartitionedSchedule::Serial, 1.0e-10, 50).newton;
    let monolithic = solve_newton(&operator, &context, &vec![0.0; dimension], &newton).unwrap();
    assert!(monolithic.converged);

    let once_config = PartitionedConfig {
        schedule: PartitionedSchedule::Serial,
        iteration: PartitionedIteration::Once,
        newton,
        divergence_growth: 10.0,
    };
    let mut execution = PartitionedExecution::new(&operator, zero_state(&operator)).unwrap();
    let report = execution.solve(&context, &once_config, 0.0).unwrap();
    assert_eq!(report.disposition, PartitionedDisposition::OnceApplied);
    assert_eq!(report.sweeps.len(), 1);
    // Accepted by declaration: the single exchange carries its splitting error and is not the
    // monolithic solution (the implicit gate above agrees within 1e-8; this does not).
    let disagreement = max_abs_difference(
        &execution.state().committed_vector().unwrap(),
        &monolithic.state,
    );
    assert!(
        disagreement > 1.0e-8,
        "a once sweep must not reproduce the monolithic solve: disagreement {disagreement:e}"
    );
}

#[test]
fn a_non_contractive_wrong_role_schedule_is_refused_diverged_under_once_and_implicit() {
    let context = EvaluationContext::reproducible();
    // Generous: the refusals below cannot be budget artifacts.
    let budget = 50;

    // The regime, established independently of the driver: the serial splitting's spectral
    // radius scales as (strength / s*)^2 with s* about 29.4 on this mesh. Strength 6 (which the
    // first version of this test used) is a convergent configuration; strength 60 is not.
    let contractive = serial_schedule_spectral_radius(&two_block_diffusion_wrong_role(6.0));
    assert!(
        contractive < 0.05,
        "strength 6 is a convergent configuration: spectral radius {contractive:e}"
    );
    let wrong = two_block_diffusion_wrong_role(60.0);
    let radius = serial_schedule_spectral_radius(&wrong);
    assert!(
        radius > 1.0,
        "strength 60 must be non-contractive: spectral radius {radius:e}"
    );

    let newton = implicit_config(PartitionedSchedule::Serial, 1.0e-10, budget).newton;
    for iteration in [
        PartitionedIteration::Once,
        PartitionedIteration::Implicit {
            tolerance: 1.0e-10,
            max_sweeps: budget,
            relaxation: 1.0,
        },
    ] {
        let config = PartitionedConfig {
            schedule: PartitionedSchedule::Serial,
            iteration: iteration.clone(),
            newton: newton.clone(),
            divergence_growth: 2.0,
        };
        let state = zero_state(&wrong);
        let before = state.committed_vector().unwrap();
        let mut execution = PartitionedExecution::new(&wrong, state).unwrap();
        let error = execution.solve(&context, &config, 0.0).unwrap_err();
        match &error {
            KrasisError::EvaluationRefused {
                code,
                origin,
                message,
            } => {
                assert_eq!(code, PARTITIONED_DIVERGED, "{iteration:?}: {message}");
                assert_eq!(origin, PARTITIONED_REFUSAL_ORIGIN);
                // The mechanism: the whole schedule correction cannot reduce the residual at
                // any admissible damping (Methodus's per-sweep line search), mapped by Krasis.
                assert!(
                    message.contains("no admissible damping"),
                    "{iteration:?}: {message}"
                );
            }
            other => panic!("{iteration:?}: expected PARTITIONED_DIVERGED, got {other:?}"),
        }
        assert_eq!(
            execution.state().committed_vector().unwrap(),
            before,
            "{iteration:?} committed something after a refusal"
        );
    }

    // Positive control: the correctly-signed fixture is contractive and converges under the
    // same budget, so the refusals above are the schedule's, not the budget's.
    let right = two_block_diffusion(0.3);
    assert!(serial_schedule_spectral_radius(&right) < 1.0);
    let mut execution = PartitionedExecution::new(&right, zero_state(&right)).unwrap();
    let report = execution
        .solve(
            &context,
            &implicit_config(PartitionedSchedule::Serial, 1.0e-10, budget),
            0.0,
        )
        .unwrap();
    assert_eq!(report.disposition, PartitionedDisposition::Converged);
    assert!(report.sweeps.len() < budget);
}
