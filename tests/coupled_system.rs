//! W7 batch P / SC-W1: N-block DAE composition across realization plans
//! (`CoupledSystemOperator`) with Newton inside BDF (`CoupledExecution` over `methodus::bdf_step`),
//! the SV7-F2 coupling graph, consistent initialization, checkpoint/restart, and the
//! partitioned-vs-monolithic agreement check. The FC10 cross-dialect acceptance test is ported
//! here (its `CrossDialectOperator` is retired by this composition).
//!
//! Fixtures are Finitum realizations, each its own leaf: a lumped network DAE `M (x' + x) = 0`
//! with a full symmetric positive-definite mass matrix (`finitum::NetworkDaeRealization`) for the
//! transient manufactured-solution, restart and consistent-initialization checks, and Krasis's
//! own GX-D1 Dirichlet-constrained linear diffusion `RealizationPlan`s on separate meshes for the
//! steady partitioned-versus-monolithic agreement check. Cross-leaf edges are mass-matrix scaled
//! exchanges probed from the leaves' own rate Jacobians -- no closed-form matrix is authored --
//! and the manufactured solution follows from the mass matrix cancelling. The transient checks
//! do not use the P1 finite-element leaves because Finitum's single-point P1 quadrature makes
//! their mass matrix rank-deficient (the C11.8 upstream finding, still open): a pure reaction
//! leaf's BDF Newton system is singular on its own, before any composition.

use std::sync::Arc;

use finitum::{
    AffineConstraint, Cell, ConstraintSet, DiscreteOperator, DofId, DofMap, DynamicExternalInput,
    ElementRestriction, ExternalInput, FiniteVolumeFace, FiniteVolumeRealization, Mesh,
    MethodRealization, NetworkDaeRealization, PreparedElement, RealizationPlan, VertexId,
};
use krasis::{
    BlockId, CoupledExecution, CoupledLeaf, CoupledOperator, CoupledSystemOperator,
    CouplingArgument, CouplingEdge, FieldId, KrasisError, RowKind, SemanticId, SimulationState,
    StateBinding, StateBlock, StateLayout, check_strategy_work,
};
use methodus::{
    BdfConfig, BdfOrder, BdfState, BlockNonlinearOperator, BlockStrategy, ComparisonTolerance,
    CsrMatrix, DaeOperator, EvaluationContext, LinearOperator, NewtonConfig, NonlinearOperator,
    StepOutcome, WorkBudget, bdf_step, solve_blocks, solve_newton, verify_dae_jvp,
};
use quantitas::UnitRegistry;
use scientia::{
    AffineMethodKernelSpec, InputSourceRequirement, compile_conservation_law_method,
    compile_network_dae_method, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels,
};

// -------------------------------------------------------------------------------------------
// Fixtures
// -------------------------------------------------------------------------------------------

const NETWORK_MODEL: &str = r#"
module krasis.coupled_system.network;
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
module krasis.coupled_system.diffusion;
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

/// A three-state lumped network `M (x' + x) = 0` with a full SPD mass matrix (so nothing is
/// diagonal by accident): the per-DOF law `x' = -x` holds exactly because the same `M`
/// multiplies both terms.
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

fn square_mesh(subdivisions: usize) -> (Mesh, DofMap, Vec<Vec<f64>>) {
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
    let mesh = Mesh::new(2, vertices.clone(), cells).unwrap();
    let dofs = DofMap::new(width * width, restrictions).unwrap();
    (mesh, dofs, vertices)
}

/// Linear diffusion (`capacity = k = 1`, `f = 0`) with Dirichlet data `boundary(x, y)` pinned
/// on every wall vertex, so the steady solution is nontrivial whenever `boundary` is.
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
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        "unit;direction=0/v1",
                        |_| vec![1.0],
                        |_, _| vec![0.0],
                    )
                    .unwrap(),
                ),
                "f" => stored.push(
                    ExternalInput::sampled(
                        integral.integral_index,
                        input.id,
                        1,
                        &mesh,
                        &element,
                        |_, _| vec![0.0],
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

/// One realization group over a Finitum `RealizationPlan`: a `CoupledOperator` bound to
/// system-level id `semantic` with its row kinds recorded, wrapped as a leaf named `name` whose
/// single block is also named `name`.
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

/// One network leaf named `name` (single block `name`), bound to system-level id `semantic`,
/// every row differential.
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

/// `scale * M_leaf`, with `M_leaf = dF/d(ydot)` probed column by column from the leaf's own
/// DAE JVP at zero state and zero rate (a constrained row's rate column is exactly zero, so a
/// Dirichlet row never receives coupling).
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

/// Two network leaves `a`, `b`, exchanging through `-epsilon * M`:
/// `M (a' + a - epsilon b) = 0`, `M (b' + b - epsilon a) = 0`.
fn two_block_network(epsilon: f64) -> CoupledSystemOperator {
    let (leaf_a, operator_a) = network_leaf("a", 0);
    let (leaf_b, operator_b) = network_leaf("b", 1);
    let edges = vec![
        exchange("a", "b", &operator_a, -epsilon),
        exchange("b", "a", &operator_b, -epsilon),
    ];
    CoupledSystemOperator::new(vec![leaf_a, leaf_b], edges).unwrap()
}

fn initial_state(operator: &CoupledSystemOperator, a: f64, b: f64) -> SimulationState {
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
        // Error control is disabled (every estimate is below 1) so the step sequence is exactly
        // the fixed `step` and the trajectory is a pure BDF1/BDF2 trajectory.
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

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f64::max)
}

// -------------------------------------------------------------------------------------------
// Composition semantics
// -------------------------------------------------------------------------------------------

#[test]
fn composed_residual_is_leaf_residuals_plus_edge_actions_and_the_jvp_is_consistent() {
    let epsilon = 0.3;
    let operator = two_block_network(epsilon);
    let n = NETWORK_DIMENSION;
    assert_eq!(DaeOperator::dimension(&operator), 2 * n);
    assert_eq!(operator.layout().blocks().len(), 2);
    assert_eq!(
        operator.binding().block_for(SemanticId::new(1)),
        Some(&BlockId::new("b"))
    );
    let context = EvaluationContext::reproducible();
    let state = pseudo_random_vector(2 * n, 1);
    let rate = pseudo_random_vector(2 * n, 2);

    let mut composed = vec![0.0; 2 * n];
    DaeOperator::residual(&operator, &context, 0.7, &state, &rate, &mut composed).unwrap();

    // Independent recomputation from a leaf and its probed mass matrix.
    let single = network_operator();
    let mass = scaled_rate_jacobian(&single, 1.0);
    let mut expected = vec![0.0; 2 * n];
    for half in [0..n, n..2 * n] {
        single
            .residual(
                &context,
                0.7,
                &state[half.clone()],
                &rate[half.clone()],
                &mut expected[half],
            )
            .unwrap();
    }
    let mut exchange_value = vec![0.0; n];
    mass.apply(&context, &state[n..2 * n], &mut exchange_value)
        .unwrap();
    for (row, value) in exchange_value.iter().enumerate() {
        expected[row] -= epsilon * value;
    }
    mass.apply(&context, &state[0..n], &mut exchange_value)
        .unwrap();
    for (row, value) in exchange_value.iter().enumerate() {
        expected[n + row] -= epsilon * value;
    }
    assert!(max_abs_difference(&composed, &expected) < 1.0e-14);
    assert!(composed.iter().any(|value| value.abs() > 0.1));

    let discrepancy = verify_dae_jvp(
        &operator,
        &context,
        0.7,
        &state,
        &rate,
        &pseudo_random_vector(2 * n, 3),
        &pseudo_random_vector(2 * n, 4),
        1.0e-6,
    )
    .unwrap();
    assert!(discrepancy < 1.0e-8, "JVP discrepancy {discrepancy}");
}

/// Runs `steps` fixed BDF steps of size `step` through the Krasis transaction from
/// `a = 1, b = 0`.
fn run_fixed_steps(
    operator: &CoupledSystemOperator,
    order: BdfOrder,
    step: f64,
    steps: usize,
) -> CoupledExecution<CoupledSystemOperator> {
    let context = EvaluationContext::reproducible();
    let mut execution = CoupledExecution::new(
        operator.clone(),
        initial_state(operator, 1.0, 0.0),
        &context,
    )
    .unwrap();
    let config = fixed_step_config(order, step);
    for _ in 0..steps {
        match execution.attempt_step(&context, step, &config).unwrap() {
            StepOutcome::Accepted(_) => {}
            StepOutcome::Rejected(rejected) => panic!("unexpected rejection: {rejected:?}"),
        }
    }
    execution
}

/// Manufactured solution: with `a(0) = 1`, `b(0) = 0` at every DOF, the mass matrix cancels
/// and every DOF follows `a = exp(-t) cosh(epsilon t)`, `b = exp(-t) sinh(epsilon t)`.
fn manufactured(operator: &CoupledSystemOperator, epsilon: f64, time: f64) -> Vec<f64> {
    let a = (-time).exp() * (epsilon * time).cosh();
    let b = (-time).exp() * (epsilon * time).sinh();
    let mut exact = vec![0.0; DaeOperator::dimension(operator)];
    exact[operator.leaf_range(0).unwrap()].fill(a);
    exact[operator.leaf_range(1).unwrap()].fill(b);
    exact
}

#[test]
fn newton_inside_bdf_reproduces_the_manufactured_solution_at_first_and_second_order() {
    let epsilon = 0.4;
    let operator = two_block_network(epsilon);
    let final_time = 0.8;
    let exact = manufactured(&operator, epsilon, final_time);

    let error_at = |order: BdfOrder, steps: usize| {
        let execution = run_fixed_steps(&operator, order, final_time / steps as f64, steps);
        assert_eq!(execution.state().step(), steps as u64);
        assert!((execution.state().time() - final_time).abs() < 1.0e-12);
        assert_eq!(execution.integrator().accepted_steps, steps as u64);
        max_abs_difference(&execution.state().committed_vector().unwrap(), &exact)
    };

    let bdf1_coarse = error_at(BdfOrder::One, 16);
    let bdf1_fine = error_at(BdfOrder::One, 32);
    let bdf1_ratio = bdf1_coarse / bdf1_fine;
    assert!(bdf1_coarse < 5.0e-2, "BDF1 error {bdf1_coarse}");
    assert!(
        (1.8..2.2).contains(&bdf1_ratio),
        "BDF1 halving ratio {bdf1_ratio} (errors {bdf1_coarse}, {bdf1_fine})"
    );

    let bdf2_coarse = error_at(BdfOrder::Two, 16);
    let bdf2_fine = error_at(BdfOrder::Two, 32);
    let bdf2_ratio = bdf2_coarse / bdf2_fine;
    assert!(bdf2_coarse < bdf1_coarse);
    assert!(
        (3.4..4.6).contains(&bdf2_ratio),
        "BDF2 halving ratio {bdf2_ratio} (errors {bdf2_coarse}, {bdf2_fine})"
    );

    // Every DOF of every leaf follows the same scalar law: the coupling is exact per DOF.
    let execution = run_fixed_steps(&operator, BdfOrder::Two, final_time / 64.0, 64);
    let committed = execution.state().committed_vector().unwrap();
    let a = &committed[operator.leaf_range(0).unwrap()];
    let b = &committed[operator.leaf_range(1).unwrap()];
    assert!(max_abs_difference(a, &vec![a[0]; a.len()]) < 1.0e-10);
    assert!(max_abs_difference(b, &vec![b[0]; b.len()]) < 1.0e-10);
    assert!(
        b[0] > 0.1,
        "the exchange genuinely moves b off zero: {}",
        b[0]
    );
}

#[test]
fn checkpoint_restart_reproduces_the_composed_trajectory_bitwise() {
    let operator = two_block_network(0.25);
    let context = EvaluationContext::reproducible();
    let step = 0.05;
    let config = fixed_step_config(BdfOrder::Two, step);

    let reference = run_fixed_steps(&operator, BdfOrder::Two, step, 6);
    let reference_final = reference.state().committed_vector().unwrap();

    let mut execution = CoupledExecution::new(
        operator.clone(),
        initial_state(&operator, 1.0, 0.0),
        &context,
    )
    .unwrap();
    for _ in 0..3 {
        assert!(matches!(
            execution.attempt_step(&context, step, &config).unwrap(),
            StepOutcome::Accepted(_)
        ));
    }
    let checkpoint = execution.checkpoint().unwrap();
    assert_eq!(checkpoint.operator_identity, operator.identity());
    assert_eq!(checkpoint.integrator.accepted_steps, 3);
    for _ in 0..3 {
        execution.attempt_step(&context, step, &config).unwrap();
    }
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        reference_final
    );

    // Restore and replay: bitwise the same final state and BDF history.
    execution.restore(&checkpoint).unwrap();
    assert_eq!(execution.state().step(), 3);
    for _ in 0..3 {
        execution.attempt_step(&context, step, &config).unwrap();
    }
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        reference_final
    );
    assert_eq!(execution.integrator(), reference.integrator());

    // A checkpoint from a differently coupled system (another edge content) is refused.
    let other = two_block_network(0.5);
    assert_ne!(other.identity(), operator.identity());
    let mut other_execution =
        CoupledExecution::new(other.clone(), initial_state(&other, 1.0, 0.0), &context).unwrap();
    let error = other_execution.restore(&checkpoint).unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );

    // A Newton policy that cannot converge rolls the transaction back and commits nothing.
    let hopeless = BdfConfig {
        newton: NewtonConfig {
            max_iterations: 1,
            absolute_tolerance: 1.0e-300,
            relative_tolerance: 0.0,
            ..NewtonConfig::default()
        },
        ..fixed_step_config(BdfOrder::One, step)
    };
    let before = execution.checkpoint().unwrap();
    let error = execution
        .attempt_step(&context, step, &hopeless)
        .unwrap_err();
    assert!(matches!(error, KrasisError::Solve(_)), "{error:?}");
    assert_eq!(execution.checkpoint().unwrap(), before);
}

#[test]
fn consistent_initialization_is_solved_over_the_composed_residual() {
    let epsilon = 0.35;
    let operator = two_block_network(epsilon)
        .with_consistent_initialization(NewtonConfig::default())
        .unwrap();
    let context = EvaluationContext::reproducible();
    let state = initial_state(&operator, 1.0, 0.0)
        .committed_vector()
        .unwrap();
    let rate = operator
        .solve_consistent_state_rate(&context, 0.0, &state)
        .unwrap();
    // a' = -a + epsilon b = -1, b' = -b + epsilon a = epsilon: the exchange edge takes part.
    let mut expected = vec![0.0; 2 * NETWORK_DIMENSION];
    expected[operator.leaf_range(0).unwrap()].fill(-1.0);
    expected[operator.leaf_range(1).unwrap()].fill(epsilon);
    assert!(max_abs_difference(&rate, &expected) < 1.0e-9);

    // The mask is part of the identity, and `BdfState::initialize` runs the check.
    let plain = two_block_network(epsilon);
    assert_ne!(plain.identity(), operator.identity());
    assert!(BdfState::initialize(&operator, &context, 0.0, state).is_ok());

    // A leaf without row kinds cannot take part in a composed mask.
    let layout = single_block_layout("bare", NETWORK_DIMENSION);
    let binding = single_binding(&layout, "bare", 7);
    let bare = CoupledLeaf::new(
        "bare",
        Arc::new(network_operator()),
        layout,
        binding,
        "bare-identity",
    )
    .unwrap();
    assert!(bare.row_kinds().is_none());
    let error = CoupledSystemOperator::new(vec![bare], Vec::new())
        .unwrap()
        .with_consistent_initialization(NewtonConfig::default())
        .unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
}

// -------------------------------------------------------------------------------------------
// Partitioned versus monolithic (batch P agreement check, steady view)
// -------------------------------------------------------------------------------------------

/// Two diffusion leaves with different wall data, exchanging through `-epsilon * M` on their
/// free rows: a nontrivial coupled steady solution that monolithic Newton and the leaf-wise
/// Gauss-Seidel/Jacobi strategies of `methodus::solve_blocks` must agree on.
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

#[test]
fn partitioned_gauss_seidel_and_jacobi_agree_with_monolithic_newton() {
    let operator = two_block_diffusion(0.3);
    let context = EvaluationContext::reproducible();
    let dimension = NonlinearOperator::dimension(&operator);
    let config = NewtonConfig {
        max_iterations: 200,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        ..NewtonConfig::default()
    };
    let start = vec![0.0; dimension];

    let monolithic = solve_newton(&operator, &context, &start, &config).unwrap();
    assert!(monolithic.converged);
    let mut residual = vec![0.0; dimension];
    NonlinearOperator::residual(&operator, &context, &monolithic.state, &mut residual).unwrap();
    assert!(residual.iter().all(|value| value.abs() < 1.0e-10));

    // Nontrivial: the cold leaf's interior is heated by the exchange, the hot leaf's interior
    // sits between its wall data (1 and 2) minus what it gives away.
    let cold = &monolithic.state[operator.leaf_range(1).unwrap()];
    assert!(cold.iter().any(|value| *value > 1.0e-3), "{cold:?}");
    let hot = &monolithic.state[operator.leaf_range(0).unwrap()];
    assert!(
        hot.iter().any(|value| *value > 1.0 && *value < 2.0),
        "{hot:?}"
    );

    for strategy in [BlockStrategy::GaussSeidel, BlockStrategy::Jacobi] {
        let partitioned = solve_blocks(&operator, &context, &start, strategy, &config).unwrap();
        assert!(partitioned.converged, "{strategy:?}");
        assert!(
            partitioned.trace.len() > 2,
            "{strategy:?} genuinely iterates the fixed point: {}",
            partitioned.trace.len()
        );
        let difference = max_abs_difference(&partitioned.state, &monolithic.state);
        assert!(difference < 1.0e-9, "{strategy:?} differs by {difference}");
    }

    // The SV0-B4 counted agreement report over the composed operator (the leaf partition is
    // its block layout), identity-bound to the composed system.
    let report = check_strategy_work(
        &operator,
        operator.identity(),
        &context,
        &start,
        &config,
        ComparisonTolerance {
            absolute: 1.0e-9,
            relative: 1.0e-9,
        },
        WorkBudget {
            operator_evaluations: 200_000,
            linear_iterations: 0,
            nonlinear_iterations: 200,
            accepted_steps: 0,
            rejected_steps: 0,
        },
    )
    .unwrap();
    assert!(report.passed, "{report:?}");
    assert_eq!(report.agreements.len(), 3);
}

// -------------------------------------------------------------------------------------------
// Coupling graph (SV7-F2)
// -------------------------------------------------------------------------------------------

#[test]
fn the_coupling_graph_orders_dependencies_first_and_detects_cycles() {
    let (a, op_a) = network_leaf("a", 0);
    let (b, op_b) = network_leaf("b", 1);
    let (c, op_c) = network_leaf("c", 2);

    // No edges: every leaf its own stage, declaration order.
    let independent =
        CoupledSystemOperator::new(vec![a.clone(), b.clone(), c.clone()], Vec::new()).unwrap();
    assert_eq!(independent.graph().stages(), &[vec![0], vec![1], vec![2]]);
    assert!(independent.graph().is_acyclic());
    assert_eq!(independent.graph().nodes(), &["a", "b", "c"]);

    // A chain a <- b <- c is a DAG solved c, b, a.
    let chain = CoupledSystemOperator::new(
        vec![a.clone(), b.clone(), c.clone()],
        vec![
            exchange("a", "b", &op_a, -0.1),
            exchange("b", "c", &op_b, -0.1),
        ],
    )
    .unwrap();
    assert_eq!(chain.graph().stages(), &[vec![2], vec![1], vec![0]]);
    assert!(chain.graph().is_acyclic());
    assert_eq!(chain.graph().dependencies().len(), 2);
    assert_eq!(chain.graph().dependencies()[0].row, 0);
    assert_eq!(chain.graph().dependencies()[0].column, 1);

    // Closing the cycle a <- b <- c <- a makes one fixed-point block.
    let cycle = CoupledSystemOperator::new(
        vec![a.clone(), b.clone(), c.clone()],
        vec![
            exchange("a", "b", &op_a, -0.1),
            exchange("b", "c", &op_b, -0.1),
            exchange("c", "a", &op_c, -0.1),
        ],
    )
    .unwrap();
    assert_eq!(cycle.graph().stages(), &[vec![0, 1, 2]]);
    assert!(!cycle.graph().is_acyclic());

    // A two-way pair plus a one-way consumer: the pair is one stage, solved before its consumer.
    let mixed = CoupledSystemOperator::new(
        vec![a, b, c],
        vec![
            exchange("a", "b", &op_a, -0.1),
            exchange("b", "a", &op_b, -0.1),
            exchange("c", "a", &op_c, -0.1),
        ],
    )
    .unwrap();
    assert_eq!(mixed.graph().stages(), &[vec![0, 1], vec![2]]);
}

#[test]
fn a_one_way_coupling_is_solved_exactly_by_one_ordered_sweep() {
    // Leaf `a` depends on `b` only; `b` is independent. A Gauss-Seidel sweep in declaration
    // order (a then b) needs a second sweep, while the graph's schedule (b then a) is exact
    // after one: the stage order is the sequential schedule.
    let (a, op_a) = network_leaf("a", 0);
    let (b, _) = network_leaf("b", 1);
    let edge = exchange("a", "b", &op_a, -0.5);
    let declared =
        CoupledSystemOperator::new(vec![a.clone(), b.clone()], vec![edge.clone()]).unwrap();
    let scheduled = CoupledSystemOperator::new(vec![b, a], vec![edge]).unwrap();
    assert_eq!(declared.graph().stages(), &[vec![1], vec![0]]);
    assert_eq!(scheduled.graph().stages(), &[vec![0], vec![1]]);

    let context = EvaluationContext::reproducible();
    let config = NewtonConfig {
        max_iterations: 20,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        ..NewtonConfig::default()
    };
    let start = pseudo_random_vector(2 * NETWORK_DIMENSION, 9);
    let sweeps = |operator: &CoupledSystemOperator| {
        solve_blocks(
            operator,
            &context,
            &start,
            BlockStrategy::GaussSeidel,
            &config,
        )
        .unwrap()
        .trace
        .len()
            - 1
    };
    assert_eq!(sweeps(&scheduled), 1);
    assert_eq!(sweeps(&declared), 2);
}

// -------------------------------------------------------------------------------------------
// Refusals and identity
// -------------------------------------------------------------------------------------------

#[test]
fn composition_refuses_malformed_leaves_and_edges() {
    let (a, op_a) = network_leaf("a", 0);
    let (b, _) = network_leaf("b", 1);
    let refused = |result: Result<CoupledSystemOperator, KrasisError>| {
        let error = result.unwrap_err();
        assert!(
            matches!(error, KrasisError::InvalidCoupling(_)),
            "{error:?}"
        );
    };

    // Unknown endpoint, self edge, shape mismatch, duplicate leaf name, no leaves.
    refused(CoupledSystemOperator::new(
        vec![a.clone(), b.clone()],
        vec![exchange("a", "zzz", &op_a, -0.1)],
    ));
    refused(CoupledSystemOperator::new(
        vec![a.clone(), b.clone()],
        vec![exchange("a", "a", &op_a, -0.1)],
    ));
    let (big_plan, big_rows) = diffusion_realization(2, |_| 0.0);
    let (big, big_op) = realization_leaf("big", 2, big_plan, big_rows);
    refused(CoupledSystemOperator::new(
        vec![a.clone(), big],
        vec![exchange("a", "big", &big_op, -0.1)],
    ));
    refused(CoupledSystemOperator::new(
        vec![a.clone(), a.clone()],
        Vec::new(),
    ));
    assert!(matches!(
        CoupledSystemOperator::new(Vec::new(), Vec::new()).unwrap_err(),
        KrasisError::EmptyLayout
    ));

    // Two leaves binding the same system-level id: the SysVarId uniqueness rule.
    let (a_again, _) = network_leaf("a2", 0);
    assert_eq!(
        CoupledSystemOperator::new(vec![a.clone(), a_again], Vec::new()).unwrap_err(),
        KrasisError::StateBindingDuplicateSemanticId(0)
    );

    // Two leaves whose blocks share a block id.
    let layout = single_block_layout("a", NETWORK_DIMENSION);
    let binding = single_binding(&layout, "a", 5);
    let clash = CoupledLeaf::new(
        "other",
        Arc::new(network_operator()),
        layout,
        binding,
        "other",
    )
    .unwrap();
    assert_eq!(
        CoupledSystemOperator::new(vec![a.clone(), clash], Vec::new()).unwrap_err(),
        KrasisError::DuplicateBlock("a".into())
    );

    // Leaf construction: width mismatch, a binding over other blocks, empty name.
    let operator: Arc<dyn DaeOperator> = Arc::new(network_operator());
    let short = single_block_layout("w", NETWORK_DIMENSION - 1);
    let short_binding = single_binding(&short, "w", 1);
    let error = CoupledLeaf::new("w", operator.clone(), short, short_binding, "w").unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
    let other_blocks = single_block_layout("v", NETWORK_DIMENSION);
    let other_binding = single_binding(&other_blocks, "v", 1);
    assert_eq!(
        CoupledLeaf::new(
            "w",
            operator.clone(),
            single_block_layout("w", NETWORK_DIMENSION),
            other_binding,
            "w"
        )
        .unwrap_err(),
        KrasisError::StateBindingLayoutMismatch
    );
    let layout = single_block_layout("w", NETWORK_DIMENSION);
    let binding = single_binding(&layout, "w", 1);
    let error = CoupledLeaf::new(" ", operator, layout, binding, "w").unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );

    // A realization leaf needs a bound coupled operator and its own layout.
    let (plan, _) = diffusion_realization(2, |_| 0.0);
    let layout = single_block_layout("w", plan.dimension());
    let unbound = CoupledOperator::new(plan.clone(), &layout).unwrap();
    let error = CoupledLeaf::realization("w", unbound, layout.clone()).unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
    let bound =
        CoupledOperator::new_with_bindings(plan, &layout, Some(single_binding(&layout, "w", 3)))
            .unwrap();
    let error =
        CoupledLeaf::realization("w", bound.clone(), single_block_layout("w", 1)).unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
    assert!(
        CoupledLeaf::realization("w", bound, layout)
            .unwrap()
            .row_kinds()
            .is_none()
    );

    // Mask length.
    assert!(matches!(
        a.with_row_kinds(vec![RowKind::Differential; NETWORK_DIMENSION + 1])
            .unwrap_err(),
        KrasisError::ConsistentInitializationMaskLength { .. }
    ));
}

#[test]
fn identity_is_canonical_and_covers_leaves_edges_and_edge_content() {
    let baseline = two_block_network(0.25);
    assert_eq!(baseline.identity(), two_block_network(0.25).identity());
    assert!(baseline.identity().starts_with("blake3:"));
    assert_ne!(baseline.identity(), two_block_network(0.5).identity());

    // Edge order does not matter; edge direction does; the leaf name does.
    let (a, op_a) = network_leaf("a", 0);
    let (b, op_b) = network_leaf("b", 1);
    let forward = CoupledSystemOperator::new(
        vec![a.clone(), b.clone()],
        vec![
            exchange("a", "b", &op_a, -0.1),
            exchange("b", "a", &op_b, -0.1),
        ],
    )
    .unwrap();
    let reordered = CoupledSystemOperator::new(
        vec![a.clone(), b.clone()],
        vec![
            exchange("b", "a", &op_b, -0.1),
            exchange("a", "b", &op_a, -0.1),
        ],
    )
    .unwrap();
    assert_eq!(forward.identity(), reordered.identity());
    let one_way = CoupledSystemOperator::new(
        vec![a.clone(), b.clone()],
        vec![exchange("a", "b", &op_a, -0.1)],
    )
    .unwrap();
    assert_ne!(forward.identity(), one_way.identity());
    let (renamed, _) = network_leaf("b2", 1);
    let renamed_system = CoupledSystemOperator::new(vec![a.clone(), renamed], Vec::new()).unwrap();
    let plain_system = CoupledSystemOperator::new(vec![a.clone(), b.clone()], Vec::new()).unwrap();
    assert_ne!(renamed_system.identity(), plain_system.identity());

    // A rate edge is a different relation from a state edge of the same content.
    let rate = CoupledSystemOperator::new(
        vec![a, b],
        vec![CouplingEdge::matrix(
            "a",
            "b",
            CouplingArgument::Rate,
            scaled_rate_jacobian(&op_a, -0.1),
        )],
    )
    .unwrap();
    assert_ne!(rate.identity(), one_way.identity());
    assert_eq!(
        rate.graph().dependencies()[0].argument,
        CouplingArgument::Rate
    );
}

// -------------------------------------------------------------------------------------------
// FC10 port: two distinct Finitum discrete families with explicit cross blocks
// -------------------------------------------------------------------------------------------

const CROSS_DIALECT_SOURCE: &str = r#"
module fixtures.cross_dialect;
model Conservation {
  domain Cells { dimension = 1; coordinates = cartesian; }
  field q: state scalar DG(order=0) on Cells { time_role = differential; };
  property speed = transport_speed(0);
  equation balance on Cells { dt(q) + div(speed * q) = 0; }
}
model Network {
  domain Graph { dimension = 0; coordinates = lumped; }
  field x: state scalar L2(order=0) on Graph { time_role = differential; };
  equation balance on Graph { dt(x) + x = 0; }
}
"#;

fn cross_dialect_blocks() -> (DiscreteOperator, DiscreteOperator) {
    let module = compile_semantics(CROSS_DIALECT_SOURCE, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic;
    let finite_volume = compile_conservation_law_method(
        &module,
        "Conservation",
        "balance",
        "q",
        AffineMethodKernelSpec {
            name: "upwind".into(),
            inputs: vec!["minus".into(), "plus".into()],
            coefficients: vec![1.0, 0.0],
            constant: 0.0,
        },
    )
    .unwrap();
    let network = compile_network_dae_method(&module, "Network", &["balance"], &["x"]).unwrap();
    let finite_volume = MethodRealization::FiniteVolume(
        FiniteVolumeRealization::new(
            finite_volume,
            vec![1.0, 1.0],
            vec![
                FiniteVolumeFace { minus: 0, plus: 1 },
                FiniteVolumeFace { minus: 1, plus: 0 },
            ],
        )
        .unwrap(),
    );
    let network = MethodRealization::NetworkDae(
        NetworkDaeRealization::new(network, vec![vec![1.0]], vec![vec![2.0]], vec![0.0]).unwrap(),
    );
    (
        DiscreteOperator::sibling(finite_volume),
        DiscreteOperator::sibling(network),
    )
}

fn dense(rows: usize, columns: usize, entries: Vec<Vec<f64>>) -> CsrMatrix {
    let mut triplets = Vec::new();
    for (row, values) in entries.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            if *value != 0.0 {
                triplets.push((row, column, *value));
            }
        }
    }
    CsrMatrix::from_triplets(rows, columns, triplets).unwrap()
}

fn cross_dialect_system(fv_from_network: Vec<Vec<f64>>) -> CoupledSystemOperator {
    let (finite_volume, network) = cross_dialect_blocks();
    let fv_layout = single_block_layout("q", 2);
    let fv_binding = single_binding(&fv_layout, "q", 0);
    let net_layout = single_block_layout("x", 1);
    let net_binding = single_binding(&net_layout, "x", 1);
    let fv_identity = finite_volume.identity();
    let net_identity = network.identity();
    let leaves = vec![
        CoupledLeaf::new(
            "fv",
            Arc::new(finite_volume),
            fv_layout,
            fv_binding,
            fv_identity,
        )
        .unwrap(),
        CoupledLeaf::new(
            "net",
            Arc::new(network),
            net_layout,
            net_binding,
            net_identity,
        )
        .unwrap(),
    ];
    let edges = vec![
        CouplingEdge::matrix(
            "fv",
            "net",
            CouplingArgument::State,
            dense(2, 1, fv_from_network),
        ),
        CouplingEdge::matrix(
            "net",
            "fv",
            CouplingArgument::State,
            dense(1, 2, vec![vec![0.25, -0.25]]),
        ),
    ];
    CoupledSystemOperator::new(leaves, edges).unwrap()
}

#[test]
fn fc10_explicit_off_diagonal_blocks_form_a_real_bidirectional_dae() {
    let operator = cross_dialect_system(vec![vec![0.5], vec![-0.5]]);
    let context = EvaluationContext::reproducible();
    let mut residual = vec![0.0; 3];
    DaeOperator::residual(
        &operator,
        &context,
        0.0,
        &[1.0, 0.0, 2.0],
        &[0.1, 0.2, 0.3],
        &mut residual,
    )
    .unwrap();
    // The values FC10's `CrossDialectOperator` produced for the same blocks and cross matrices.
    assert_eq!(residual, vec![2.1, -1.8, 4.55]);
    assert_eq!(operator.graph().stages(), &[vec![0, 1]]);
    let discrepancy = verify_dae_jvp(
        &operator,
        &context,
        0.3,
        &[1.0, 0.0, 2.0],
        &[0.1, 0.2, 0.3],
        &[-0.4, 0.7, 0.2],
        &[0.5, -0.1, 0.6],
        1.0e-6,
    )
    .unwrap();
    assert!(discrepancy < 1.0e-9);
}

#[test]
fn fc10_identity_is_canonical_and_covers_coupling_matrices() {
    let baseline = cross_dialect_system(vec![vec![0.5], vec![-0.5]]);
    assert_eq!(
        baseline.identity(),
        cross_dialect_system(vec![vec![0.5], vec![-0.5]]).identity()
    );
    assert_ne!(
        baseline.identity(),
        cross_dialect_system(vec![vec![0.75], vec![-0.5]]).identity()
    );
}

#[test]
fn fc10_methodus_advances_the_cross_dialect_system_without_method_specific_policy() {
    let operator = cross_dialect_system(vec![vec![0.5], vec![-0.5]]);
    let context = EvaluationContext::reproducible();
    let state = BdfState::initialize(&operator, &context, 0.0, vec![1.0, 0.0, 0.5]).unwrap();
    let config = BdfConfig {
        order: BdfOrder::One,
        minimum_step: 0.1,
        maximum_step: 0.1,
        ..BdfConfig::default()
    };
    let StepOutcome::Accepted(accepted) =
        bdf_step(&operator, &context, &state, 0.1, &config).unwrap()
    else {
        panic!("the first-order fixed step must be accepted")
    };
    assert_eq!(accepted.state.accepted_steps, 1);
    assert_eq!(accepted.state.time, 0.1);
    assert_ne!(accepted.state.values, state.values);

    // Same through the Krasis transaction, with the leaf partition exposed to Methodus.
    let mut fields = SimulationState::new(operator.layout().clone(), 2);
    fields
        .insert_field(FieldId::new("q"), vec![1.0, 0.0])
        .unwrap();
    fields.insert_field(FieldId::new("x"), vec![0.5]).unwrap();
    let mut execution = CoupledExecution::new(operator.clone(), fields, &context).unwrap();
    assert!(matches!(
        execution.attempt_step(&context, 0.1, &config).unwrap(),
        StepOutcome::Accepted(_)
    ));
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        accepted.state.values
    );
    let names: Vec<&str> = operator
        .block_layout()
        .blocks()
        .iter()
        .map(|block| block.name())
        .collect();
    assert_eq!(names, ["fv", "net"]);
}
