//! W7 batch P, Krasis item 3 closing the loop with Finitum `4a6fe65`: a [`CoupledLeaf`] over
//! Finitum's state-dependent, rate-capable `ReducedSystemOperator` (`SystemRealizationPlan` ->
//! `bind_kernels` -> `reduced`), composed across two realization groups (two meshes) and
//! advanced by Newton inside BDF through the Krasis transaction, with the dense and the
//! Newton-Krylov solver hooks.
//!
//! Fixture: two instances of one transient heat model, `hot` (3x3 unit square, unit source)
//! and `cold` (4x4 unit square, no source), both with homogeneous Dirichlet walls; `cold`'s
//! residual gains `-gamma * S * u_hot`, `S` sampling the nearest `hot` vertex at every
//! unconstrained `cold` vertex (a one-way volumetric exchange). The second instance repeats the
//! model's field id, so it is rebound to a system-level id before composing.

use std::collections::BTreeMap;

use finitum::{
    BlockLayout, FieldSource, MeshProfile, PointEvaluation, ReducedSystemOperator, RegionMap,
    RegionTagId, SysVarId, SystemConstitutiveInput, SystemEssentialConstraintRequirement,
    SystemRealizationPlan, TaggedMesh, essential_constraints_from_system, realize,
};
use krasis::{
    AttemptDisposition, BlockId, CoupledExecution, CoupledLeaf, CoupledSystemOperator,
    CouplingArgument, CouplingEdge, FieldId, FinitumVerificationSource, KrasisError,
    OperatorIdentity, RowKind, SemanticId, SimulationState, StateBinding, block_state_layout,
    check_history_and_rejection, check_restart_trajectory, check_rollback_identity,
};
use methodus::{
    BdfConfig, BdfOrder, BdfState, ComparisonTolerance, CsrMatrix, DaeOperator, EvaluationContext,
    ForcingPolicy, GmresConfig, KrylovMethod, NewtonConfig, NewtonKrylovConfig, NewtonKrylovSolver,
    NonlinearSolver, StepOutcome, bdf_step, verify_dae_jvp,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, OperatorSystem, SemanticModel, SymbolId, compile_operator_system,
    compile_semantics,
};

const HEAT_MODEL: &str = r#"
module krasis.coupled_finitum.heat;
model Heat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  property capacity = storage_capacity(u);
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation evolution on Omega { capacity * dt(u) - div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = 0; }
}
"#;

/// System-level id offset for the second instance of the model (`SysVarId`s are dense across
/// every instance; the fixture stands in for system elaboration).
const SECOND_INSTANCE_OFFSET: u32 = 1000;

struct Compiled {
    model: SemanticModel,
    system: OperatorSystem,
}

fn compile() -> Compiled {
    let compilation = compile_semantics(HEAT_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(&compilation.semantic, "Heat", &["evolution"]).unwrap();
    Compiled {
        model: compilation.semantic.models[0].clone(),
        system,
    }
}

fn symbol(model: &SemanticModel, name: &str) -> SymbolId {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name)
        .map(|symbol| symbol.id)
        .unwrap_or_else(|| panic!("model has no symbol {name}"))
}

/// Unit capacity and diffusivity, and the constant volumetric source `source`.
fn constant_constitutive(compiled: &Compiled, source: f64) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &compiled.system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = compiled.model.symbols[input.binding.symbol.index()]
                    .name
                    .as_str();
                let value = match name {
                    "capacity" | "k" => 1.0,
                    "f" => source,
                    other => panic!("unexpected non-basis input {other}"),
                };
                constitutive.push(
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        1,
                        format!("krasis-heat/{name}={value}"),
                        move |_: &PointEvaluation| vec![value],
                        |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                    )
                    .unwrap(),
                );
            }
        }
    }
    constitutive
}

struct HeatInstance {
    tagged: TaggedMesh,
    reduced: ReducedSystemOperator,
    field: SymbolId,
    row_kinds: Vec<RowKind>,
}

impl HeatInstance {
    fn dimension(&self) -> usize {
        self.tagged.mesh.vertices().len()
    }

    fn is_constrained(&self, dof: usize) -> bool {
        self.row_kinds[dof] == RowKind::Algebraic
    }
}

/// One realization group: the heat model on a `subdivisions`-by-`subdivisions` simplex unit
/// square with source `source`, walls eliminated as essential constraints.
fn heat_instance(compiled: &Compiled, subdivisions: usize, source: f64) -> HeatInstance {
    let tagged = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap();
    let field = symbol(&compiled.model, "u");
    let vertex_count = tagged.mesh.vertices().len();
    let layout = BlockLayout::new([(field, vertex_count, 1)]).unwrap();
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), tagged.mesh.clone(), layout).unwrap();
    let operator = plan
        .bind_kernels(constant_constitutive(compiled, source), BTreeMap::new())
        .unwrap();
    let mut requirements = Vec::new();
    let mut region_map = RegionMap::new();
    for block in &compiled.system.blocks {
        for requirement in &block.factorization.essential_constraints {
            region_map.insert(
                requirement.region,
                ["x_min", "x_max", "y_min", "y_max"].map(RegionTagId::new),
            );
            requirements.push(SystemEssentialConstraintRequirement {
                field: block.row,
                requirement: requirement.clone(),
                value: FieldSource::constant([0.0]),
            });
        }
    }
    let constraints =
        essential_constraints_from_system(&operator, &tagged, &region_map, &requirements).unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    let mut row_kinds = vec![RowKind::Differential; vertex_count];
    for constraint in reduced.constraints().constraints() {
        row_kinds[constraint.target.0] = RowKind::Algebraic;
    }
    assert!(row_kinds.contains(&RowKind::Algebraic));
    HeatInstance {
        tagged,
        reduced,
        field,
        row_kinds,
    }
}

/// `scale * S`, `S` the `cold`-by-`hot` matrix sampling the nearest `hot` vertex at every
/// unconstrained `cold` vertex (constrained rows stay pure constraint rows).
fn sampling_matrix(cold: &HeatInstance, hot: &HeatInstance, scale: f64) -> CsrMatrix {
    let hot_vertices = hot.tagged.mesh.vertices();
    let triplets = cold
        .tagged
        .mesh
        .vertices()
        .iter()
        .enumerate()
        .filter(|(row, _)| !cold.is_constrained(*row))
        .map(|(row, point)| {
            let nearest = hot_vertices
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    distance(point, left).total_cmp(&distance(point, right))
                })
                .map(|(column, _)| column)
                .unwrap();
            (row, nearest, scale)
        })
        .collect();
    CsrMatrix::from_triplets(cold.dimension(), hot.dimension(), triplets).unwrap()
}

fn distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(l, r)| (l - r) * (l - r))
        .sum::<f64>()
        .sqrt()
}

struct Fixture {
    operator: CoupledSystemOperator,
    hot: HeatInstance,
    cold: HeatInstance,
}

/// The two-leaf composition; `gamma = 0` composes the two instances with no edge at all.
fn two_heat_leaves(compiled: &Compiled, gamma: f64) -> Fixture {
    let hot = heat_instance(compiled, 3, 1.0);
    let cold = heat_instance(compiled, 4, 0.0);
    let hot_leaf = CoupledLeaf::reduced_system("hot", hot.reduced.clone())
        .unwrap()
        .with_row_kinds(hot.row_kinds.clone())
        .unwrap();
    let cold_leaf = CoupledLeaf::reduced_system("cold", cold.reduced.clone()).unwrap();
    // The second instance of the model repeats `u`'s per-model id: rebind it to its
    // system-level id.
    let rekeyed = StateBinding::new(
        cold_leaf.layout(),
        cold_leaf
            .layout()
            .blocks()
            .iter()
            .map(|block| {
                let semantic = cold_leaf.binding().semantic_for(block.id()).unwrap();
                (
                    SemanticId::new(semantic.as_u32() + SECOND_INSTANCE_OFFSET),
                    block.id().clone(),
                )
            })
            .collect(),
    )
    .unwrap();
    let cold_leaf = cold_leaf
        .with_binding(rekeyed)
        .unwrap()
        .with_row_kinds(cold.row_kinds.clone())
        .unwrap();
    let edges = if gamma == 0.0 {
        Vec::new()
    } else {
        vec![CouplingEdge::matrix(
            "cold",
            "hot",
            CouplingArgument::State,
            sampling_matrix(&cold, &hot, -gamma),
        )]
    };
    // Finitum's system path integrates the P1 mass exactly (degree-4 triangle rule), so the
    // composed index-1 consistent initialization is well posed here; `CoupledExecution::new`
    // runs it through Methodus's `BdfState::initialize`.
    let operator = CoupledSystemOperator::new(vec![hot_leaf, cold_leaf], edges)
        .unwrap()
        .with_consistent_initialization(NewtonConfig {
            absolute_tolerance: 1.0e-13,
            relative_tolerance: 1.0e-12,
            ..NewtonConfig::default()
        })
        .unwrap();
    Fixture {
        operator,
        hot,
        cold,
    }
}

/// A Dirichlet-consistent bump on `hot`, `cold` at rest.
fn initial_values(fixture: &Fixture) -> Vec<f64> {
    let mut values = vec![0.0; fixture.operator.dimension()];
    let hot = fixture.operator.leaf_range(0).unwrap();
    for (vertex, point) in fixture.hot.tagged.mesh.vertices().iter().enumerate() {
        values[hot.start + vertex] =
            (std::f64::consts::PI * point[0]).sin() * (std::f64::consts::PI * point[1]).sin();
    }
    values
}

fn initial_state(fixture: &Fixture) -> SimulationState {
    let values = initial_values(fixture);
    let mut state = SimulationState::new(fixture.operator.layout().clone(), 4);
    for block in fixture.operator.layout().blocks() {
        state
            .insert_field(
                FieldId::new(block.id().as_str()),
                values[block.range()].to_vec(),
            )
            .unwrap();
    }
    state
}

fn fixed_step_config(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::One,
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

const STEP: f64 = 0.02;
const STEPS: usize = 5;

/// `STEPS` fixed BDF1 steps through the Krasis transaction, with the dense Newton path or with
/// `solver`.
fn run_transaction(
    fixture: &Fixture,
    solver: Option<&dyn NonlinearSolver>,
) -> CoupledExecution<CoupledSystemOperator> {
    let context = EvaluationContext::reproducible();
    let mut execution =
        CoupledExecution::new(fixture.operator.clone(), initial_state(fixture), &context).unwrap();
    let config = fixed_step_config(STEP);
    for _ in 0..STEPS {
        let outcome = match solver {
            Some(solver) => execution.attempt_step_with(&context, STEP, &config, solver),
            None => execution.attempt_step(&context, STEP, &config),
        }
        .unwrap();
        match outcome {
            StepOutcome::Accepted(_) => {}
            StepOutcome::Rejected(rejected) => panic!("unexpected rejection: {rejected:?}"),
        }
    }
    assert_eq!(execution.integrator().accepted_steps, STEPS as u64);
    assert!((execution.integrator().time - STEPS as f64 * STEP).abs() <= 1.0e-12);
    execution
}

/// The same `STEPS` steps on one instance alone, straight through Methodus.
fn run_standalone(instance: &HeatInstance, initial: Vec<f64>) -> Vec<f64> {
    let context = EvaluationContext::reproducible();
    let config = fixed_step_config(STEP);
    let mut state = BdfState {
        time: 0.0,
        values: initial,
        previous_values: None,
        previous_step: None,
        accepted_steps: 0,
    };
    for _ in 0..STEPS {
        match bdf_step(&instance.reduced, &context, &state, STEP, &config).unwrap() {
            StepOutcome::Accepted(accepted) => state = accepted.state,
            StepOutcome::Rejected(rejected) => panic!("unexpected rejection: {rejected:?}"),
        }
    }
    state.values
}

fn probe_vector(dimension: usize, seed: f64, scale: f64) -> Vec<f64> {
    (0..dimension)
        .map(|index| scale * ((index as f64 + seed) * 0.618_034).sin())
        .collect()
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f64::max)
}

fn newton_krylov_config() -> (KrylovMethod, NewtonKrylovConfig) {
    (
        KrylovMethod::Gmres(GmresConfig {
            absolute_tolerance: 1.0e-14,
            relative_tolerance: 1.0e-14,
            ..GmresConfig::default()
        }),
        NewtonKrylovConfig {
            absolute_tolerance: 1.0e-13,
            relative_tolerance: 1.0e-12,
            forcing: ForcingPolicy::Constant { forcing: 1.0e-10 },
            ..NewtonKrylovConfig::default()
        },
    )
}

#[test]
fn finitum_reduced_system_leaves_compose_into_one_transient_dae() {
    let compiled = compile();
    let fixture = two_heat_leaves(&compiled, 0.8);
    let operator = &fixture.operator;
    let (hot, cold) = (&fixture.hot, &fixture.cold);
    assert_eq!(operator.dimension(), hot.dimension() + cold.dimension());
    assert_eq!(operator.leaf_range(0).unwrap(), 0..hot.dimension());
    assert_eq!(
        operator.leaf_range(1).unwrap(),
        hot.dimension()..hot.dimension() + cold.dimension()
    );

    // Blocks are namespaced by leaf; the binding carries the per-model field id for the first
    // instance and the system-level id for the second; identities are Finitum's content ids.
    let field = hot.field.0;
    let hot_block = BlockId::new(format!("hot/field_{field}"));
    let cold_block = BlockId::new(format!("cold/field_{field}"));
    assert!(operator.layout().block(&hot_block).is_some());
    assert!(operator.layout().block(&cold_block).is_some());
    assert_eq!(
        operator.binding().semantic_for(&hot_block),
        Some(SemanticId::new(field))
    );
    assert_eq!(
        operator.binding().semantic_for(&cold_block),
        Some(SemanticId::new(field + SECOND_INSTANCE_OFFSET))
    );
    for leaf in operator.leaves() {
        assert!(
            leaf.identity().starts_with("finitum-reduced-system:"),
            "{}",
            leaf.identity()
        );
    }
    assert_ne!(
        operator.leaves()[0].identity(),
        operator.leaves()[1].identity()
    );
    assert!(operator.graph().is_acyclic());
    assert_eq!(operator.graph().stages(), &[vec![0], vec![1]]);

    // Composed residual = leaf residuals + the sampled exchange; the JVP is consistent.
    let context = EvaluationContext::reproducible();
    let dimension = operator.dimension();
    let time = 0.3;
    let state = probe_vector(dimension, 0.4, 1.0);
    let rate = probe_vector(dimension, 2.2, 0.7);
    let mut composed = vec![0.0; dimension];
    operator
        .residual(&context, time, &state, &rate, &mut composed)
        .unwrap();
    let hot_range = operator.leaf_range(0).unwrap();
    let cold_range = operator.leaf_range(1).unwrap();
    let mut expected = vec![0.0; dimension];
    hot.reduced
        .residual(
            time,
            &state[hot_range.clone()],
            &rate[hot_range.clone()],
            &mut expected[hot_range.clone()],
        )
        .unwrap();
    cold.reduced
        .residual(
            time,
            &state[cold_range.clone()],
            &rate[cold_range.clone()],
            &mut expected[cold_range.clone()],
        )
        .unwrap();
    let exchange = sampling_matrix(cold, hot, -0.8);
    let mut sampled = vec![0.0; cold.dimension()];
    methodus::LinearOperator::apply(&exchange, &context, &state[hot_range], &mut sampled).unwrap();
    for (value, contribution) in expected[cold_range].iter_mut().zip(&sampled) {
        *value += contribution;
    }
    assert!(sampled.iter().any(|value| value.abs() > 1.0e-3));
    assert!(max_abs_difference(&composed, &expected) <= 1.0e-13);
    let discrepancy = verify_dae_jvp(
        operator,
        &context,
        time,
        &state,
        &rate,
        &probe_vector(dimension, 1.1, 1.0),
        &probe_vector(dimension, 3.3, 0.8),
        1.0e-5,
    )
    .unwrap();
    assert!(
        discrepancy <= 1.0e-6,
        "composed DAE JVP discrepancy {discrepancy}"
    );

    // The consistent initial rate solved over the composed residual (cross edge included):
    // every differential row of `F(0, y0, ydot)` vanishes, every wall row's rate is zero.
    let initial = initial_values(&fixture);
    let consistent = operator
        .solve_consistent_state_rate(&context, 0.0, &initial)
        .unwrap();
    let mut residual = vec![0.0; dimension];
    operator
        .residual(&context, 0.0, &initial, &consistent, &mut residual)
        .unwrap();
    let mut differential_rows = 0;
    for (row, kind) in hot.row_kinds.iter().chain(&cold.row_kinds).enumerate() {
        match kind {
            RowKind::Differential => {
                differential_rows += 1;
                assert!(
                    residual[row].abs() <= 1.0e-9,
                    "row {row}: {}",
                    residual[row]
                );
            }
            RowKind::Algebraic => assert_eq!(consistent[row], 0.0),
        }
    }
    assert!(differential_rows > 0);
    let cold_start = operator.leaf_range(1).unwrap().start;
    assert!(
        consistent[cold_start..]
            .iter()
            .any(|rate| rate.abs() > 1.0e-3),
        "the exchange drives the cold instance from rest"
    );
}

#[test]
fn newton_inside_bdf_advances_two_finitum_leaves_through_the_transaction() {
    let compiled = compile();

    // Uncoupled: the composition reproduces each instance's own Methodus trajectory.
    let uncoupled = two_heat_leaves(&compiled, 0.0);
    let hot_range = uncoupled.operator.leaf_range(0).unwrap();
    let cold_range = uncoupled.operator.leaf_range(1).unwrap();
    let initial = initial_values(&uncoupled);
    let hot_alone = run_standalone(&uncoupled.hot, initial[hot_range.clone()].to_vec());
    let cold_alone = run_standalone(&uncoupled.cold, initial[cold_range.clone()].to_vec());
    let composed = run_transaction(&uncoupled, None)
        .state()
        .committed_vector()
        .unwrap();
    assert!(max_abs_difference(&composed[hot_range.clone()], &hot_alone) <= 1.0e-10);
    assert!(max_abs_difference(&composed[cold_range.clone()], &cold_alone) <= 1.0e-10);
    assert!(
        max_abs_difference(&hot_alone, &initial[hot_range.clone()]) > 1.0e-3,
        "the hot instance evolved"
    );
    assert!(cold_alone.iter().all(|value| *value == 0.0));

    // Coupled one way: `hot` is unchanged, `cold` is heated, walls stay eliminated, and the
    // dense and Newton-Krylov solver hooks agree.
    let coupled = two_heat_leaves(&compiled, 0.8);
    let dense = run_transaction(&coupled, None);
    let dense_final = dense.state().committed_vector().unwrap();
    assert!(max_abs_difference(&dense_final[hot_range.clone()], &hot_alone) <= 1.0e-10);
    let cold_final = &dense_final[cold_range.clone()];
    assert!(
        cold_final.iter().any(|value| value.abs() > 1.0e-4),
        "the cold instance received heat"
    );
    for (dof, value) in cold_final.iter().enumerate() {
        if coupled.cold.is_constrained(dof) {
            assert!(value.abs() <= 1.0e-12, "cold wall dof {dof} = {value}");
        }
    }
    for (dof, value) in dense_final[hot_range].iter().enumerate() {
        if coupled.hot.is_constrained(dof) {
            assert!(value.abs() <= 1.0e-12, "hot wall dof {dof} = {value}");
        }
    }
    let (method, config) = newton_krylov_config();
    let krylov = NewtonKrylovSolver::new(&method, None, None, &config);
    let inexact = run_transaction(&coupled, Some(&krylov));
    let inexact_final = inexact.state().committed_vector().unwrap();
    let disagreement = max_abs_difference(&inexact_final, &dense_final);
    assert!(
        disagreement <= 1.0e-9,
        "Newton-Krylov vs dense: {disagreement}"
    );

    // A checkpoint of the composition binds to the Finitum content identities: restoring it
    // into the uncoupled composition (different edge set, same leaves) is refused.
    let checkpoint = dense.checkpoint().unwrap();
    let context = EvaluationContext::reproducible();
    let mut other = CoupledExecution::new(
        uncoupled.operator.clone(),
        initial_state(&uncoupled),
        &context,
    )
    .unwrap();
    let error = other.restore(&checkpoint).unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );
}

#[test]
fn a_reduced_system_leaf_refuses_a_binding_over_other_blocks() {
    let compiled = compile();
    let instance = heat_instance(&compiled, 2, 1.0);
    let leaf = CoupledLeaf::reduced_system("hot", instance.reduced.clone()).unwrap();
    let foreign = krasis::StateLayout::new(vec![krasis::StateBlock::new(
        BlockId::new("elsewhere"),
        0..instance.dimension(),
    )])
    .unwrap();
    let binding = StateBinding::new(
        &foreign,
        vec![(SemanticId::new(7), BlockId::new("elsewhere"))],
    )
    .unwrap();
    let error = leaf.with_binding(binding).unwrap_err();
    assert!(
        matches!(error, KrasisError::StateBindingLayoutMismatch),
        "{error:?}"
    );

    // Two instances with the same per-model id cannot compose until one is rebound.
    let twin = CoupledLeaf::reduced_system("cold", instance.reduced.clone()).unwrap();
    let leaf_again = CoupledLeaf::reduced_system("hot", instance.reduced.clone()).unwrap();
    let error = CoupledSystemOperator::new(vec![leaf_again, twin], Vec::new()).unwrap_err();
    assert!(
        matches!(error, KrasisError::StateBindingDuplicateSemanticId(_)),
        "{error:?}"
    );
}

fn bump(point: &[f64]) -> Vec<f64> {
    vec![(std::f64::consts::PI * point[0]).sin() * (std::f64::consts::PI * point[1]).sin()]
}

/// BDF2 with tolerances no step can meet, for the forced rejection after one primed step.
fn rejecting_config(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::Two,
        absolute_tolerance: 1.0e-16,
        relative_tolerance: 1.0e-16,
        minimum_step: 1.0e-8,
        maximum_step: step,
        newton: fixed_step_config(step).newton,
    }
}

/// One nodal patch per leaf over that leaf's own mesh and state slice, composed into one source.
fn patch_sources(
    fixture: &Fixture,
    values: &[f64],
    cold_exact: impl Fn(&[f64]) -> Vec<f64>,
) -> (FinitumVerificationSource, FinitumVerificationSource) {
    let tolerance = ComparisonTolerance {
        absolute: 1.0e-12,
        relative: 1.0e-12,
    };
    let hot = FinitumVerificationSource::check_patch(
        &fixture.hot.reduced,
        1,
        &values[fixture.operator.leaf_range(0).unwrap()],
        tolerance,
        bump,
    )
    .unwrap();
    let cold = FinitumVerificationSource::check_patch(
        &fixture.cold.reduced,
        1,
        &values[fixture.operator.leaf_range(1).unwrap()],
        tolerance,
        cold_exact,
    )
    .unwrap();
    (hot, cold)
}

/// W7 (2026-09-05): the FC7 sources over two `reduced_system` leaves on two meshes -- one
/// Finitum patch per leaf composed into one source; rollback of a forced rejection is bit-exact,
/// a split/restart reproduces the continuous trajectory bit for bit, history stays synchronized,
/// and a source covering only one leaf (or bound to a failed patch) is refused / not accepted.
#[test]
fn fc7_sources_over_two_reduced_system_leaves_roll_back_and_restart_bit_exactly() {
    let compiled = compile();
    let fixture = two_heat_leaves(&compiled, 0.5);
    let context = EvaluationContext::reproducible();
    let execution =
        CoupledExecution::new(fixture.operator.clone(), initial_state(&fixture), &context).unwrap();
    let config = fixed_step_config(STEP);
    let rejecting = rejecting_config(STEP);
    let values = execution.integrator().values.clone();
    let (hot, cold) = patch_sources(&fixture, &values, |_| vec![0.0]);
    assert_eq!(
        hot.realization_identities().collect::<Vec<_>>(),
        vec![fixture.hot.reduced.content_identity()]
    );

    // Evidence over the hot leaf alone leaves the cold realization uncovered.
    let refusal =
        check_restart_trajectory(&execution, &context, STEP, 4, 2, &config, &hot).unwrap_err();
    assert_eq!(refusal.code, "KRASIS_VERIFY_FINITUM_SOURCE");
    assert!(
        refusal
            .message
            .contains(&fixture.cold.reduced.content_identity())
    );

    let source = FinitumVerificationSource::compose([hot, cold]);
    let restart =
        check_restart_trajectory(&execution, &context, STEP, 4, 2, &config, &source).unwrap();
    assert!(restart.passed, "{restart:#?}");
    assert_eq!(restart.trajectory_l_infinity, 0.0);
    assert_eq!(restart.trajectory_l2_time, 0.0);
    assert!(restart.final_checkpoint_byte_identical);
    assert_eq!(restart.binding.schema, "krasis-verification/2");
    assert_eq!(
        restart
            .binding
            .finitum_sources
            .iter()
            .map(|source| source.realization_identity.as_str())
            .collect::<Vec<_>>(),
        vec![
            fixture.hot.reduced.content_identity(),
            fixture.cold.reduced.content_identity()
        ]
    );
    assert!(restart.binding.finitum_sources.iter().all(|s| s.accepted));
    assert!(
        restart
            .validate(&execution, &context, STEP, 4, 2, &config, &source)
            .unwrap()
            .accepted
    );

    let mut primed = execution.clone();
    let outcome = primed.attempt_step(&context, STEP, &config).unwrap();
    assert!(matches!(outcome, StepOutcome::Accepted(_)));
    let rollback = check_rollback_identity(&primed, &context, STEP, &rejecting, &source).unwrap();
    assert!(rollback.passed, "{rollback:#?}");
    assert_eq!(rollback.disposition, AttemptDisposition::Rejected);
    assert!(rollback.byte_identical);
    assert_eq!(
        rollback.checkpoint_before_digest,
        rollback.checkpoint_after_digest
    );

    let history =
        check_history_and_rejection(&execution, &context, STEP, 2, &config, &rejecting, &source)
            .unwrap();
    assert!(history.passed, "{history:#?}");
    assert!(history.synchronized);
    assert_eq!(history.rejected_attempt, AttemptDisposition::Rejected);
    assert!(history.rejection_byte_identical);
    let mut depths = history.field_history_depths.clone();
    depths.sort();
    let mut expected = fixture
        .operator
        .layout()
        .blocks()
        .iter()
        .map(|block| (block.id().to_string(), 2))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(depths, expected);

    // A failed cold patch is recomputed as not accepted and fails the report without refusing.
    let (hot, failed_cold) = patch_sources(&fixture, &values, |_| vec![1.0]);
    let failed = FinitumVerificationSource::compose([hot, failed_cold]);
    let not_accepted =
        check_rollback_identity(&primed, &context, STEP, &rejecting, &failed).unwrap();
    assert_eq!(
        not_accepted.binding.finitum_verification_accepted,
        Some(false)
    );
    assert!(not_accepted.binding.finitum_sources[0].accepted);
    assert!(!not_accepted.binding.finitum_sources[1].accepted);
    assert!(!not_accepted.passed);
    assert!(not_accepted.byte_identical);
}

/// W7 (2026-09-05): `CoupledSystemOperator::initial_state_from` projects each leaf's bound
/// `FieldSource`s over that leaf's own mesh (`NodalContext` per leaf), reproducing the fixture's
/// hand-assembled initial state bitwise, and refuses a missing or unknown block typed.
#[test]
fn initial_state_from_projects_onto_each_leaf_mesh() {
    let compiled = compile();
    let fixture = two_heat_leaves(&compiled, 0.0);
    let layout = fixture.operator.layout();
    assert_eq!(layout.blocks().len(), 2);
    let bindings = layout
        .blocks()
        .iter()
        .map(|block| {
            let source = if block.id().as_str().starts_with("hot/") {
                FieldSource::sampled(bump)
            } else {
                FieldSource::constant([0.0])
            };
            (block.id().clone(), source)
        })
        .collect::<Vec<_>>();
    let state = fixture.operator.initial_state_from(4, &bindings).unwrap();
    assert_eq!(state.committed_vector().unwrap(), initial_values(&fixture));
    assert_eq!(
        state.committed_vector().unwrap(),
        initial_state(&fixture).committed_vector().unwrap()
    );
    let context = EvaluationContext::reproducible();
    let execution = CoupledExecution::new(fixture.operator.clone(), state, &context).unwrap();
    assert_eq!(
        execution.checkpoint().unwrap(),
        CoupledExecution::new(fixture.operator.clone(), initial_state(&fixture), &context)
            .unwrap()
            .checkpoint()
            .unwrap()
    );

    let missing = fixture
        .operator
        .initial_state_from(4, &bindings[..1])
        .unwrap_err();
    assert!(
        matches!(missing, KrasisError::InitialBlockMissing(_)),
        "{missing}"
    );
    let mut unknown = bindings.clone();
    unknown.push((BlockId::new("warm/field_0"), FieldSource::constant([0.0])));
    let unknown = fixture
        .operator
        .initial_state_from(4, &unknown)
        .unwrap_err();
    assert!(
        matches!(unknown, KrasisError::InitialBlockUnknown(_)),
        "{unknown}"
    );
}

/// W7 (2026-09-05, HANDOFF §6 Krasis item): `block_state_layout` reads `FieldBlock::variable`
/// (the system-level `SysVarId`), not the per-model `SymbolId`, so a layout Finitum keys for a
/// second instance binds its blocks to dense system ids without `CoupledLeaf::with_binding`.
/// The honest boundary: Finitum's `SystemRealizationPlan::new` still accepts only the
/// one-instance identity keying, so a second instance of one model cannot yet be realized with
/// its own `SysVarId`s and the fixture's caller-side `with_binding` stays until it can.
#[test]
fn block_state_layout_binds_the_system_variable_and_finitum_still_refuses_a_keyed_plan() {
    let compiled = compile();
    let field = symbol(&compiled.model, "u");
    let keyed = BlockLayout::new_keyed([(SysVarId(field.0 + SECOND_INSTANCE_OFFSET), field, 9, 1)])
        .unwrap();
    let (layout, binding) = block_state_layout(&keyed).unwrap();
    let block = BlockId::new(format!("field_{}", field.0 + SECOND_INSTANCE_OFFSET));
    assert_eq!(layout.blocks()[0].id(), &block);
    assert_eq!(
        binding.semantic_for(&block),
        Some(SemanticId::new(field.0 + SECOND_INSTANCE_OFFSET))
    );
    let identity = BlockLayout::new([(field, 9, 1)]).unwrap();
    let (identity_layout, identity_binding) = block_state_layout(&identity).unwrap();
    assert_eq!(
        identity_layout.blocks()[0].id(),
        &BlockId::new(format!("field_{}", field.0))
    );
    assert_eq!(
        identity_binding.semantic_for(identity_layout.blocks()[0].id()),
        Some(SemanticId::new(field.0))
    );

    let tagged = realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![2, 2],
    })
    .unwrap();
    let refused = SystemRealizationPlan::new(compiled.system.clone(), tagged.mesh.clone(), keyed);
    assert!(
        refused.is_err(),
        "Finitum accepted a keyed one-instance plan"
    );
}
