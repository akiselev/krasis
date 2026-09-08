//! W8 lane K1 (workspace PLAN §6 W8 decision 3, GX-CONTRACTS C12.9 item 6): a typed evaluation
//! failure raised inside a Finitum callback passes through a Krasis transaction unchanged --
//! code, origin and located message -- as `KrasisError::EvaluationRefused`, rolled back bitwise
//! and never retried; the same failure through `initial_state_from` over a
//! `FieldSource::Fallible`; the `check_*` functions report the producer's code; and the
//! `with_consistent_initialization_when_algebraic` opt-in runs the rate solve exactly when a row
//! is algebraic.
//!
//! Fixture: the two-instance transient heat composition of `coupled_finitum_system.rs` (`hot`
//! 3x3 with unit source, `cold` 4x4 at rest, a one-way sampled exchange), with `hot`'s
//! diffusivity bound through `SystemConstitutiveInput::try_new` and refusing, once armed, at
//! every quadrature point of one cell (`REFUSING_CELL`) evaluated after the armed time, with
//! code `RUN_TANGENT_UNAVAILABLE` and origin `Slot("provider/diffusivity")`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use finitum::{
    BlockLayout, CellId, ConstraintSet, FieldSource, InputEvaluationError, InputOrigin,
    MeshProfile, PointEvaluation, ReducedSystemOperator, RegionMap, RegionTagId,
    SystemConstitutiveInput, SystemEssentialConstraintRequirement, SystemRealizationPlan,
    TaggedMesh, essential_constraints_from_system_at, realize,
};
use krasis::{
    AttemptDisposition, BlockId, CoupledExecution, CoupledLeaf, CoupledSystemOperator,
    CouplingArgument, CouplingEdge, EvaluationRefusal, FieldId, FinitumVerificationSource,
    KrasisError, NodalContext, RowKind, SemanticId, SimulationState, StateBinding,
    TransactionPhase, check_cross_block_derivatives, check_history_and_rejection,
    check_restart_trajectory, check_rollback_identity, initial_state_from,
};
use methodus::{
    BdfConfig, BdfOrder, ComparisonTolerance, CsrMatrix, DaeOperator, DenseNewton,
    EvaluationContext, ForcingPolicy, GmresConfig, KrylovMethod, NewtonConfig, NewtonKrylovConfig,
    NewtonKrylovSolver, NonlinearOperator, NonlinearSolver, NumericError, SolveError, SolveReport,
    StepOutcome,
};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, OperatorSystem, SemanticModel, SymbolId, compile_operator_system,
    compile_semantics,
};

const HEAT_MODEL: &str = r#"
module krasis.w8_typed_failures.heat;
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

/// System-level id offset for the second instance of the model (see
/// `coupled_finitum_system.rs`).
const SECOND_INSTANCE_OFFSET: u32 = 1000;

const CODE: &str = "RUN_TANGENT_UNAVAILABLE";
const ORIGIN: &str = "provider/diffusivity";
const MESSAGE: &str = "no tangent for an analytic_provided diffusivity";
const REFUSING_CELL: CellId = CellId(2);

const STEP: f64 = 0.02;
const STEPS: usize = 5;

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

/// The switch and counters behind `hot`'s diffusivity callbacks.
struct Probe {
    /// Bits of the time after which an evaluation in `REFUSING_CELL` refuses (`+inf`: never).
    refuse_after: AtomicU64,
    value_calls: AtomicUsize,
    direction_calls: AtomicUsize,
    value_refusals: AtomicUsize,
    direction_refusals: AtomicUsize,
}

impl Probe {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            refuse_after: AtomicU64::new(f64::INFINITY.to_bits()),
            value_calls: AtomicUsize::new(0),
            direction_calls: AtomicUsize::new(0),
            value_refusals: AtomicUsize::new(0),
            direction_refusals: AtomicUsize::new(0),
        })
    }

    /// Refuse at every evaluation time from now on.
    fn arm(&self) {
        self.refuse_after
            .store((-1.0_f64).to_bits(), Ordering::SeqCst);
    }

    /// Refuse at every evaluation time strictly after `time`.
    fn arm_after(&self, time: f64) {
        self.refuse_after.store(time.to_bits(), Ordering::SeqCst);
    }

    fn disarm(&self) {
        self.refuse_after
            .store(f64::INFINITY.to_bits(), Ordering::SeqCst);
    }

    fn refuses(&self, point: &PointEvaluation) -> bool {
        point.cell == REFUSING_CELL
            && point.time > f64::from_bits(self.refuse_after.load(Ordering::SeqCst))
    }

    fn value_calls(&self) -> usize {
        self.value_calls.load(Ordering::SeqCst)
    }

    fn direction_calls(&self) -> usize {
        self.direction_calls.load(Ordering::SeqCst)
    }

    fn value_refusals(&self) -> usize {
        self.value_refusals.load(Ordering::SeqCst)
    }

    fn direction_refusals(&self) -> usize {
        self.direction_refusals.load(Ordering::SeqCst)
    }
}

fn refusal() -> InputEvaluationError {
    InputEvaluationError::new(CODE, InputOrigin::Slot(ORIGIN.into()), MESSAGE)
}

/// Unit capacity and diffusivity and the constant volumetric source `source`; `k` through the
/// fallible constructor with `probe` when one is given.
fn constitutive(
    compiled: &Compiled,
    source: f64,
    probe: Option<&Arc<Probe>>,
) -> Vec<SystemConstitutiveInput> {
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
                let built = match (name, probe) {
                    ("k", Some(probe)) => {
                        let value_probe = probe.clone();
                        let direction_probe = probe.clone();
                        SystemConstitutiveInput::try_new(
                            block.equation.clone(),
                            integral.integral_index,
                            input.id,
                            1,
                            "krasis-heat/k=1(fallible)",
                            move |point: &PointEvaluation| {
                                value_probe.value_calls.fetch_add(1, Ordering::SeqCst);
                                if value_probe.refuses(point) {
                                    value_probe.value_refusals.fetch_add(1, Ordering::SeqCst);
                                    Err(refusal())
                                } else {
                                    Ok(vec![1.0])
                                }
                            },
                            move |point: &PointEvaluation, _: &PointEvaluation| {
                                direction_probe
                                    .direction_calls
                                    .fetch_add(1, Ordering::SeqCst);
                                if direction_probe.refuses(point) {
                                    direction_probe
                                        .direction_refusals
                                        .fetch_add(1, Ordering::SeqCst);
                                    Err(refusal())
                                } else {
                                    Ok(vec![0.0])
                                }
                            },
                        )
                    }
                    _ => SystemConstitutiveInput::try_new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        1,
                        format!("krasis-heat/{name}={value}"),
                        move |_: &PointEvaluation| Ok(vec![value]),
                        |_: &PointEvaluation, _: &PointEvaluation| Ok(vec![0.0]),
                    ),
                };
                constitutive.push(built.unwrap());
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

/// One realization group on a `subdivisions`-square with source `source`; walls eliminated as
/// essential constraints (algebraic rows) when `constrained`, otherwise no constraint at all
/// (every row differential: a pure Neumann heat problem).
fn heat_instance(
    compiled: &Compiled,
    subdivisions: usize,
    source: f64,
    probe: Option<&Arc<Probe>>,
    constrained: bool,
) -> HeatInstance {
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
        .bind_kernels(constitutive(compiled, source, probe), BTreeMap::new())
        .unwrap();
    let constraints = if constrained {
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
        essential_constraints_from_system_at(&operator, &tagged, &region_map, &requirements, 0.0)
            .unwrap()
    } else {
        ConstraintSet::new(vertex_count, []).unwrap()
    };
    let reduced = operator.reduced(constraints).unwrap();
    let mut row_kinds = vec![RowKind::Differential; vertex_count];
    for constraint in reduced.constraints().constraints() {
        row_kinds[constraint.target.0] = RowKind::Algebraic;
    }
    assert_eq!(row_kinds.contains(&RowKind::Algebraic), constrained);
    HeatInstance {
        tagged,
        reduced,
        field,
        row_kinds,
    }
}

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

fn newton() -> NewtonConfig {
    NewtonConfig {
        absolute_tolerance: 1.0e-13,
        relative_tolerance: 1.0e-12,
        ..NewtonConfig::default()
    }
}

/// The two-leaf composition with exchange `gamma`, `hot`'s diffusivity probed when `probe` is
/// given, both instances constrained or not, consistent initialization recorded through the
/// existing builder or the when-algebraic opt-in.
fn two_heat_leaves(
    compiled: &Compiled,
    gamma: f64,
    probe: Option<&Arc<Probe>>,
    constrained: bool,
    when_algebraic: bool,
) -> Fixture {
    let hot = heat_instance(compiled, 3, 1.0, probe, constrained);
    let cold = heat_instance(compiled, 4, 0.0, None, constrained);
    let hot_leaf = CoupledLeaf::reduced_system("hot", hot.reduced.clone())
        .unwrap()
        .with_row_kinds(hot.row_kinds.clone())
        .unwrap();
    let cold_leaf = CoupledLeaf::reduced_system("cold", cold.reduced.clone()).unwrap();
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
    let edges = vec![CouplingEdge::matrix(
        "cold",
        "hot",
        CouplingArgument::State,
        sampling_matrix(&cold, &hot, -gamma),
    )];
    let operator = CoupledSystemOperator::new(vec![hot_leaf, cold_leaf], edges).unwrap();
    let operator = if when_algebraic {
        operator.with_consistent_initialization_when_algebraic(newton())
    } else {
        operator.with_consistent_initialization(newton())
    }
    .unwrap();
    Fixture {
        operator,
        hot,
        cold,
    }
}

fn bump(point: &[f64]) -> Vec<f64> {
    vec![(std::f64::consts::PI * point[0]).sin() * (std::f64::consts::PI * point[1]).sin()]
}

/// A Dirichlet-consistent bump on `hot`, `cold` at rest.
fn initial_values(fixture: &Fixture) -> Vec<f64> {
    let mut values = vec![0.0; fixture.operator.dimension()];
    let hot = fixture.operator.leaf_range(0).unwrap();
    for (vertex, point) in fixture.hot.tagged.mesh.vertices().iter().enumerate() {
        values[hot.start + vertex] = bump(point)[0];
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
        newton: newton(),
    }
}

/// BDF2 with tolerances no step can meet, for the forced rejection after one primed step.
fn rejecting_config(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::Two,
        absolute_tolerance: 1.0e-16,
        relative_tolerance: 1.0e-16,
        minimum_step: 1.0e-8,
        maximum_step: step,
        newton: newton(),
    }
}

fn checkpoint_bytes(execution: &CoupledExecution<CoupledSystemOperator>) -> Vec<u8> {
    serde_json::to_vec(&execution.checkpoint().unwrap()).unwrap()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(l, r)| (l - r).abs())
        .fold(0.0, f64::max)
}

fn accept(
    execution: &mut CoupledExecution<CoupledSystemOperator>,
    context: &EvaluationContext,
    config: &BdfConfig,
) {
    match execution.attempt_step(context, STEP, config).unwrap() {
        StepOutcome::Accepted(_) => {}
        StepOutcome::Rejected(rejected) => panic!("unexpected rejection: {rejected:?}"),
    }
}

/// How many times one residual evaluation of `hot` reaches the diffusivity value callback, and
/// how many of those refuse at the current arming.
fn per_residual(fixture: &Fixture, probe: &Probe, time: f64) -> (usize, usize) {
    let context = EvaluationContext::reproducible();
    let dimension = fixture.hot.dimension();
    let hot = fixture.operator.leaf_range(0).unwrap();
    let state = initial_values(fixture)[hot].to_vec();
    let rate = vec![0.0; dimension];
    let mut output = vec![0.0; dimension];
    let calls = probe.value_calls();
    let refusals = probe.value_refusals();
    let _ = DaeOperator::residual(
        &fixture.hot.reduced,
        &context,
        time,
        &state,
        &rate,
        &mut output,
    );
    (
        probe.value_calls() - calls,
        probe.value_refusals() - refusals,
    )
}

/// `DenseNewton` counting the implicit solves it is asked for.
struct CountingSolver<'a> {
    inner: DenseNewton<'a>,
    calls: AtomicUsize,
}

impl NonlinearSolver for CountingSolver<'_> {
    fn solve(
        &self,
        operator: &dyn NonlinearOperator,
        context: &EvaluationContext,
        initial_state: &[f64],
    ) -> Result<SolveReport, SolveError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.solve(operator, context, initial_state)
    }
}

fn newton_krylov() -> (KrylovMethod, NewtonKrylovConfig) {
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

fn assert_typed(error: &KrasisError, time: f64) -> EvaluationRefusal {
    let KrasisError::EvaluationRefused {
        code,
        origin,
        message,
    } = error
    else {
        panic!("expected a typed evaluation refusal, got {error:?}");
    };
    assert_eq!(code, CODE);
    assert_eq!(origin, ORIGIN);
    assert!(
        message.starts_with("point (")
            && message.contains(&format!(", t = {time}, "))
            && message.ends_with(&format!("cell {}: {MESSAGE}", REFUSING_CELL.0)),
        "located message: {message}"
    );
    assert_eq!(error.to_string(), format!("{CODE} at {ORIGIN}: {message}"));
    EvaluationRefusal {
        code: code.clone(),
        origin: origin.clone(),
        message: message.clone(),
    }
}

#[test]
fn a_typed_constitutive_refusal_is_the_transaction_outcome_rolled_back_after_one_attempt() {
    let compiled = compile();
    let probe = Probe::new();
    let fixture = two_heat_leaves(&compiled, 0.8, Some(&probe), true, false);
    let context = EvaluationContext::reproducible();
    let config = fixed_step_config(STEP);
    let mut execution =
        CoupledExecution::new(fixture.operator.clone(), initial_state(&fixture), &context).unwrap();
    // One accepted step first, so the refusal interrupts a trajectory with BDF history.
    accept(&mut execution, &context, &config);
    let before = checkpoint_bytes(&execution);
    let refused_time = execution.integrator().time + STEP;

    probe.arm();
    let (_, refusals_per_residual) = per_residual(&fixture, &probe, refused_time);
    assert!(refusals_per_residual >= 1);
    let value_refusals = probe.value_refusals();
    let direction_refusals = probe.direction_refusals();
    let solver = CountingSolver {
        inner: DenseNewton::new(&config.newton),
        calls: AtomicUsize::new(0),
    };
    let error = execution
        .attempt_step_with(&context, STEP, &config, &solver)
        .unwrap_err();
    let refusal = assert_typed(&error, refused_time);

    // Exactly one attempt: one implicit solve was asked for, one residual evaluation reached
    // the callback, no Jacobian action was ever formed (so no Newton iteration and no retry),
    // and the outcome is an error, not a `Rejected` step suggesting a smaller one.
    assert_eq!(solver.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        probe.value_refusals() - value_refusals,
        refusals_per_residual
    );
    assert_eq!(probe.direction_refusals() - direction_refusals, 0);

    // Rolled back bitwise; the transaction log carries the same code and origin; the
    // checkpoint does not carry the log.
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);
    assert_eq!(checkpoint_bytes(&execution), before);
    assert_eq!(execution.integrator().accepted_steps, 1);
    let log = execution.evaluation_refusals();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].refusal, refusal);
    assert_eq!(log[0].time, STEP);
    assert_eq!(log[0].step, STEP);

    // The dense path and the Newton-Krylov path return the identical typed outcome.
    let dense = execution.attempt_step(&context, STEP, &config).unwrap_err();
    assert_eq!(dense, error);
    let (method, nk_config) = newton_krylov();
    let krylov = NewtonKrylovSolver::new(&method, None, None, &nk_config);
    let inexact = execution
        .attempt_step_with(&context, STEP, &config, &krylov)
        .unwrap_err();
    assert_eq!(inexact, error);
    assert_eq!(execution.evaluation_refusals().len(), 3);
    assert_eq!(checkpoint_bytes(&execution), before);

    // Disarmed, the execution continues from the rolled-back state, and the trajectory is the
    // never-refusing composition's, bitwise (the fallible constructor with `Ok` is the
    // infallible one).
    probe.disarm();
    for _ in 1..STEPS {
        accept(&mut execution, &context, &config);
    }
    let clean = two_heat_leaves(&compiled, 0.8, None, true, false);
    let mut reference =
        CoupledExecution::new(clean.operator.clone(), initial_state(&clean), &context).unwrap();
    for _ in 0..STEPS {
        accept(&mut reference, &context, &config);
    }
    assert_eq!(
        bits(&execution.state().committed_vector().unwrap()),
        bits(&reference.state().committed_vector().unwrap())
    );
    assert!(reference.evaluation_refusals().is_empty());
}

#[test]
fn a_typed_refusal_during_consistent_initialization_is_typed_from_coupled_execution_new() {
    let compiled = compile();
    let probe = Probe::new();
    let fixture = two_heat_leaves(&compiled, 0.8, Some(&probe), true, false);
    let context = EvaluationContext::reproducible();
    probe.arm();

    // The composed operator's own Methodus boundary re-emits the typed error unchanged.
    let initial = initial_values(&fixture);
    let mut state = initial.clone();
    let error =
        DaeOperator::make_initial_state_consistent(&fixture.operator, &context, 0.0, &mut state)
            .unwrap_err();
    assert_eq!(error.evaluation_code(), Some(CODE));
    let NumericError::Evaluation { origin, .. } = &error else {
        panic!("{error:?}");
    };
    assert_eq!(origin, ORIGIN);
    assert_eq!(
        state, initial,
        "consistent initialization never adjusts state"
    );
    let error = fixture
        .operator
        .solve_consistent_state_rate(&context, 0.0, &initial)
        .unwrap_err();
    assert_typed(&error, 0.0);

    // ... and so does the execution's constructor, which runs it through Methodus.
    let error = CoupledExecution::new(fixture.operator.clone(), initial_state(&fixture), &context)
        .unwrap_err();
    assert_typed(&error, 0.0);
}

#[test]
fn initial_state_from_evaluates_a_fallible_source_at_the_initial_time_and_refuses_typed() {
    let compiled = compile();
    let fixture = two_heat_leaves(&compiled, 0.8, None, true, false);
    let field = fixture.hot.field.0;
    let hot_block = BlockId::new(format!("hot/field_{field}"));
    let cold_block = BlockId::new(format!("cold/field_{field}"));
    let times = Arc::new(Mutex::new(Vec::new()));
    let seen = times.clone();
    let fallible = FieldSource::fallible(move |point: &[f64], time: f64| {
        seen.lock().unwrap().push(time);
        Ok(bump(point))
    });

    // A time-observed fallible source projects exactly as the baseline, at `t = 0`.
    let sampled = fixture
        .operator
        .initial_state_from(
            4,
            &[
                (
                    hot_block.clone(),
                    FieldSource::fallible(move |coordinates, _time| Ok(bump(coordinates))),
                ),
                (cold_block.clone(), FieldSource::constant([0.0])),
            ],
        )
        .unwrap();
    let projected = fixture
        .operator
        .initial_state_from(
            4,
            &[
                (hot_block.clone(), fallible.clone()),
                (cold_block.clone(), FieldSource::constant([0.0])),
            ],
        )
        .unwrap();
    assert_eq!(
        bits(&projected.committed_vector().unwrap()),
        bits(&sampled.committed_vector().unwrap())
    );
    assert_eq!(
        bits(&projected.committed_vector().unwrap()),
        bits(&initial_values(&fixture))
    );
    assert_eq!(projected.time(), 0.0);
    let hot_leaf = &fixture.operator.leaves()[0];
    let nodal = NodalContext::new(fixture.hot.tagged.mesh.vertices()).unwrap();
    let direct = initial_state_from(
        hot_leaf.layout(),
        &nodal,
        4,
        &[(hot_block.clone(), fallible.clone())],
    )
    .unwrap();
    assert_eq!(
        bits(&direct.committed_vector().unwrap()),
        bits(
            projected
                .committed(&FieldId::new(hot_block.as_str()))
                .unwrap()
        )
    );
    let times = times.lock().unwrap();
    assert_eq!(times.len(), 2 * fixture.hot.dimension());
    assert!(times.iter().all(|time| *time == 0.0));

    // A refusing one is the typed refusal, located at the first refusing vertex in vertex
    // order and at the initial time, with the message formatted as Finitum's own sampling
    // sites format it.
    let refusing = FieldSource::fallible(|point: &[f64], _: f64| {
        if point[0] > 0.6 {
            Err(InputEvaluationError::new(
                "RUN_INITIAL_DATA_UNAVAILABLE",
                InputOrigin::Slot("initial/u".into()),
                "no initial datum here",
            ))
        } else {
            Ok(bump(point))
        }
    });
    let vertex = fixture
        .hot
        .tagged
        .mesh
        .vertices()
        .iter()
        .find(|point| point[0] > 0.6)
        .unwrap();
    let expected = KrasisError::EvaluationRefused {
        code: "RUN_INITIAL_DATA_UNAVAILABLE".into(),
        origin: "initial/u".into(),
        message: format!(
            "point ({}, {}), t = 0: no initial datum here",
            vertex[0], vertex[1]
        ),
    };
    let error = initial_state_from(
        hot_leaf.layout(),
        &nodal,
        4,
        &[(hot_block.clone(), refusing.clone())],
    )
    .unwrap_err();
    assert_eq!(error, expected);
    let error = fixture
        .operator
        .initial_state_from(
            4,
            &[
                (hot_block, refusing),
                (cold_block, FieldSource::constant([0.0])),
            ],
        )
        .unwrap_err();
    assert_eq!(error, expected);
    assert_eq!(
        error.to_string(),
        format!(
            "RUN_INITIAL_DATA_UNAVAILABLE at initial/u: point ({}, {}), t = 0: no initial datum here",
            vertex[0], vertex[1]
        )
    );
}

#[test]
fn check_functions_report_the_producer_code_when_the_operator_refuses() {
    let compiled = compile();
    let probe = Probe::new();
    let fixture = two_heat_leaves(&compiled, 0.8, Some(&probe), true, false);
    let context = EvaluationContext::reproducible();
    let config = fixed_step_config(STEP);
    let execution =
        CoupledExecution::new(fixture.operator.clone(), initial_state(&fixture), &context).unwrap();
    let values = initial_values(&fixture);
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
        |_| vec![0.0],
    )
    .unwrap();
    let source = FinitumVerificationSource::compose([hot, cold]);
    let typed_prefix = format!("{ORIGIN}: point (");

    // Without a refusal the rollback report has no `evaluation_refusal`, and the serialized
    // report carries no such key (every earlier digest is unchanged).
    let plain = check_rollback_identity(&execution, &context, STEP, &config, &source).unwrap();
    assert_eq!(plain.disposition, AttemptDisposition::UnexpectedAccepted);
    assert!(plain.evaluation_refusal.is_none());
    assert!(
        !serde_json::to_string(&plain)
            .unwrap()
            .contains("evaluation_refusal")
    );

    // A refusing operator inside a derivative check: the refusal carries the producer's code.
    probe.arm();
    let dimension = fixture.operator.dimension();
    let direction: Vec<f64> = (0..dimension)
        .map(|index| ((index as f64 + 0.3) * 0.618_034).sin())
        .collect();
    let refusal = check_cross_block_derivatives(
        &fixture.operator,
        fixture.operator.identity(),
        &context,
        &values,
        &direction,
        &[1.0e-3, 1.0e-4],
        tolerance,
    )
    .unwrap_err();
    assert_eq!(refusal.code, CODE);
    assert!(
        refusal.message.starts_with(&typed_prefix),
        "{}",
        refusal.message
    );
    assert!(refusal.message.ends_with(MESSAGE), "{}", refusal.message);

    // The probed attempt of a rollback check: rolled back byte-identically as any solver
    // error, with the typed refusal reported in the report; it round-trips and revalidates.
    let rollback = check_rollback_identity(&execution, &context, STEP, &config, &source).unwrap();
    assert_eq!(rollback.disposition, AttemptDisposition::SolverError);
    assert!(rollback.byte_identical);
    assert!(rollback.passed, "{rollback:#?}");
    let recorded = rollback
        .evaluation_refusal
        .clone()
        .expect("typed refusal recorded");
    assert_eq!(recorded.code, CODE);
    assert_eq!(recorded.origin, ORIGIN);
    assert!(recorded.message.contains(&format!(", t = {STEP}, ")));
    let json = serde_json::to_string(&rollback).unwrap();
    assert!(json.contains("evaluation_refusal"));
    assert_eq!(
        serde_json::from_str::<krasis::RollbackIdentityReport>(&json).unwrap(),
        rollback
    );
    assert!(
        rollback
            .validate(&execution, &context, STEP, &config, &source)
            .unwrap()
            .accepted
    );

    // A refusal on the way to a required accepted step is the check's refusal, typed.
    probe.arm_after(STEP);
    let refusal =
        check_restart_trajectory(&execution, &context, STEP, 4, 2, &config, &source).unwrap_err();
    assert_eq!(refusal.code, CODE);
    assert!(
        refusal.message.starts_with(&typed_prefix),
        "{}",
        refusal.message
    );

    // The history check: one accepted step at `t = STEP`, then the rejection attempt at
    // `2 * STEP` refuses typed and is rolled back byte-identically.
    let history = check_history_and_rejection(
        &execution,
        &context,
        STEP,
        1,
        &config,
        &rejecting_config(STEP),
        &source,
    )
    .unwrap();
    assert!(history.synchronized);
    assert_eq!(history.rejected_attempt, AttemptDisposition::SolverError);
    assert!(history.rejection_byte_identical);
    assert!(history.passed, "{history:#?}");
    let recorded = history
        .evaluation_refusal
        .clone()
        .expect("typed refusal recorded");
    assert_eq!(recorded.code, CODE);
    assert_eq!(recorded.origin, ORIGIN);
    assert!(
        recorded
            .message
            .contains(&format!(", t = {}, ", 2.0 * STEP))
    );
    assert!(
        execution.evaluation_refusals().is_empty(),
        "checks probe a clone"
    );
}

#[test]
fn consistent_initialization_when_algebraic_solves_only_with_an_algebraic_row() {
    let compiled = compile();
    let context = EvaluationContext::reproducible();
    let config = fixed_step_config(STEP);

    for constrained in [true, false] {
        let always_probe = Probe::new();
        let when_probe = Probe::new();
        let always = two_heat_leaves(&compiled, 0.8, Some(&always_probe), constrained, false);
        let when = two_heat_leaves(&compiled, 0.8, Some(&when_probe), constrained, true);
        assert_ne!(always.operator.identity(), when.operator.identity());
        assert!(always.operator.identity().contains(":consistent-init="));
        assert!(when.operator.identity().contains(":consistent-init="));
        let (calls_per_residual, _) = per_residual(&when, &when_probe, 0.0);
        assert!(calls_per_residual >= 1);

        let always_calls = always_probe.value_calls();
        let when_calls = when_probe.value_calls();
        let when_directions = when_probe.direction_calls();
        let mut always_execution =
            CoupledExecution::new(always.operator.clone(), initial_state(&always), &context)
                .unwrap();
        let mut when_execution =
            CoupledExecution::new(when.operator.clone(), initial_state(&when), &context).unwrap();
        // The existing path solved (more than one residual evaluation reached the callback).
        assert!(always_probe.value_calls() - always_calls > calls_per_residual);
        if constrained {
            // An algebraic row: the opt-in solves exactly as the existing path does.
            assert!(when_probe.value_calls() - when_calls > calls_per_residual);
            assert_eq!(
                when_probe.value_calls() - when_calls,
                always_probe.value_calls() - always_calls
            );
            assert_eq!(
                when_probe.direction_calls() - when_directions,
                always_probe.direction_calls()
            );
        } else {
            // All rows differential: exactly one residual evaluation, no Jacobian action, no
            // mass solve.
            assert_eq!(when_probe.value_calls() - when_calls, calls_per_residual);
            assert_eq!(when_probe.direction_calls() - when_directions, 0);
        }

        // Identical committed initial state, identical trajectory.
        assert_eq!(
            bits(&when_execution.state().committed_vector().unwrap()),
            bits(&always_execution.state().committed_vector().unwrap())
        );
        for _ in 0..STEPS {
            accept(&mut always_execution, &context, &config);
            accept(&mut when_execution, &context, &config);
            let always_values = always_execution.state().committed_vector().unwrap();
            let when_values = when_execution.state().committed_vector().unwrap();
            assert!(max_abs_difference(&always_values, &when_values) <= 1.0e-12);
            assert_eq!(bits(&always_values), bits(&when_values));
        }
        assert_eq!(always_execution.integrator().accepted_steps, STEPS as u64);
    }
}
