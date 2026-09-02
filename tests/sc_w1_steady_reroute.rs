//! SC-W1: the steady system runner's target entry point in Krasis.
//!
//! Sinbad's steady runner (`run_system_plan`) today calls Methodus directly with no Krasis
//! transaction: `SystemRealizationPlan` -> `bind_kernels` -> `reduced(constraints)` ->
//! `load_vector` -> `solve_minres`/`solve_gmres`/`solve_conjugate_gradient` from a zero initial
//! guess. Rerouting it through `krasis::BlockLinearExecution` must reproduce today's solutions
//! bit for bit (the SC-W1 gate on 25-stokes and 13-mixed-darcy is "within 1e-12 relative with
//! identical `Completed` outcomes"; this file proves the stronger bitwise statement on the
//! Krasis side for the 25-stokes corpus over the same Finitum path).
//!
//! The fixture mirrors `finitum/tests/sv2b4_system_stokes.rs` (the corpus's own `equation_sign`
//! correction, wall constraints, the auto-derived pressure nullspace candidate verified in the
//! kernel) and is physics-neutral in code: every closure is bound by input shape, never by name.

use std::collections::BTreeMap;
use std::fs;

use finitum::{
    BlockLayout, ConstraintSet, FieldSource, MeshProfile, RegionMap, RegionTagId,
    SystemConstitutiveInput, SystemEssentialConstraintRequirement, SystemRealizationPlan,
    essential_constraints_from_system, quadratic_simplex_dof_map, realize,
};
use krasis::{
    BlockLinearAlgorithm, BlockLinearExecution, BlockLinearSolver, KrasisError, OperatorIdentity,
    SimulationState, TransactionPhase, block_state_layout,
};
use methodus::{
    BlockLinearOperator, ConjugateGradientConfig, ConjugateGradientSymmetryPolicy,
    EvaluationContext, GmresConfig, LinearOperator, MinresConfig, NullspaceProjector,
    OperatorSymmetry, solve_gmres, solve_minres,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, OperatorSystem, RegionId, SymbolId,
    compile_operator_system, compile_semantics,
};

const STOKES_CORPUS: &str = "/projects/sinbad/sinbad/physics/corpus/25-stokes.res";
const MU: f64 = 1.7;

fn stress(strain: &[f64]) -> Vec<f64> {
    strain.iter().map(|value| 2.0 * MU * value).collect()
}

struct CompiledStokes {
    system: OperatorSystem,
    velocity: SymbolId,
    pressure: SymbolId,
}

fn compile_stokes() -> CompiledStokes {
    let source = fs::read_to_string(STOKES_CORPUS).expect("25-stokes.res corpus is readable");
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "StokesFlow",
        &["momentum", "incompressibility"],
    )
    .unwrap();
    let row = |equation: &str| {
        system
            .blocks
            .iter()
            .find(|block| block.equation == equation)
            .expect("compiled block")
            .row
    };
    let velocity = row("momentum");
    let pressure = row("incompressibility");
    CompiledStokes {
        system,
        velocity,
        pressure,
    }
}

fn unit_square(subdivisions: usize) -> finitum::TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![subdivisions, subdivisions],
    })
    .unwrap()
}

fn taylor_hood_layout(mesh: &finitum::Mesh, velocity: SymbolId, pressure: SymbolId) -> BlockLayout {
    let velocity_nodes = quadratic_simplex_dof_map(mesh, 2).unwrap().dof_count() / 2;
    let pressure_nodes = mesh.vertices().len();
    BlockLayout::new([(velocity, velocity_nodes, 2), (pressure, pressure_nodes, 1)]).unwrap()
}

/// Every non-`Basis` primal input, bound by shape: 4 components is the linear stress closure, 2
/// components the body force, anything else zero.
fn stokes_constitutive(system: &OperatorSystem) -> Vec<SystemConstitutiveInput> {
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let components = input.shape.iter().product::<usize>().max(1);
                let binding = if components == 4 {
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "sc-w1-reroute/viscosity",
                        |evaluation: &finitum::PointEvaluation| {
                            stress(
                                evaluation
                                    .values(DerivativeEvaluation::SymmetricGradient)
                                    .expect("active symmetric-gradient input"),
                            )
                        },
                        |_evaluation: &finitum::PointEvaluation,
                         direction: &finitum::PointEvaluation| {
                            stress(
                                direction
                                    .values(DerivativeEvaluation::SymmetricGradient)
                                    .expect("active symmetric-gradient direction"),
                            )
                        },
                    )
                } else if components == 2 {
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "sc-w1-reroute/body-force",
                        |evaluation: &finitum::PointEvaluation| {
                            let x = evaluation.coordinates[0];
                            let y = evaluation.coordinates[1];
                            vec![y - 0.5, 0.5 - x]
                        },
                        move |_evaluation: &finitum::PointEvaluation,
                              _direction: &finitum::PointEvaluation| {
                            vec![0.0; components]
                        },
                    )
                } else {
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        components,
                        "sc-w1-reroute/zero",
                        move |_evaluation: &finitum::PointEvaluation| vec![0.0; components],
                        move |_evaluation: &finitum::PointEvaluation,
                              _direction: &finitum::PointEvaluation| {
                            vec![0.0; components]
                        },
                    )
                };
                constitutive.push(binding.unwrap());
            }
        }
    }
    constitutive
}

fn walls_region_map(region: RegionId) -> RegionMap {
    let mut map = RegionMap::new();
    map.insert(
        region,
        [
            RegionTagId::new("x_min"),
            RegionTagId::new("x_max"),
            RegionTagId::new("y_min"),
            RegionTagId::new("y_max"),
        ],
    );
    map
}

struct SteadyFixture {
    reduced: finitum::ReducedSystemOperator,
    right_hand_side: Vec<f64>,
    nullspace: Option<finitum::BlockNullspaceMode>,
}

/// The steady runner's Finitum path, up to (but not including) the Methodus solve.
fn steady_fixture(subdivisions: usize, equation_sign: BTreeMap<String, f64>) -> SteadyFixture {
    let compiled = compile_stokes();
    let mesh = unit_square(subdivisions);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let operator = plan
        .bind_kernels(stokes_constitutive(&compiled.system), equation_sign)
        .unwrap();
    let momentum_requirement = compiled
        .system
        .blocks
        .iter()
        .find(|block| block.equation == "momentum")
        .unwrap()
        .factorization
        .essential_constraints
        .first()
        .expect("momentum declares one essential-constraint requirement")
        .clone();
    let region_map = walls_region_map(momentum_requirement.region);
    let constraints = essential_constraints_from_system(
        &operator,
        &mesh,
        &region_map,
        &[SystemEssentialConstraintRequirement {
            field: compiled.velocity,
            requirement: momentum_requirement,
            value: FieldSource::constant(vec![0.0, 0.0]),
        }],
    )
    .unwrap();
    let reduced = operator.reduced(constraints).unwrap();
    let nullspace = match operator.nullspace_candidates() {
        [candidate] => {
            let mode = candidate.resolve(operator.layout()).unwrap();
            mode.verify_in_kernel(&reduced, 1.0e-8)
                .unwrap()
                .then_some(mode)
        }
        _ => None,
    };
    let right_hand_side = reduced.load_vector().unwrap();
    SteadyFixture {
        reduced,
        right_hand_side,
        nullspace,
    }
}

fn signed() -> BTreeMap<String, f64> {
    BTreeMap::from([("incompressibility".to_string(), -1.0)])
}

fn zero_state(reduced: &finitum::ReducedSystemOperator) -> SimulationState {
    let (layout, binding) = block_state_layout(reduced.operator().layout()).unwrap();
    assert_eq!(binding.blocks().count(), 2);
    SimulationState::zeroed(layout, 2)
}

#[test]
fn minres_through_krasis_is_bit_identical_to_the_direct_methodus_solve() {
    let fixture = steady_fixture(2, signed());
    let reduced = &fixture.reduced;
    assert_eq!(
        reduced.operator().prove_symmetry(1.0e-9).unwrap(),
        OperatorSymmetry::Symmetric
    );
    assert!(fixture.nullspace.is_some());
    let projector = fixture
        .nullspace
        .as_ref()
        .map(|mode| mode.projector() as &dyn NullspaceProjector);
    let dimension = reduced.rows();
    let context = EvaluationContext::reproducible();
    let config = MinresConfig::default();

    // Today's runner: a direct Methodus call from a zero initial guess.
    let direct = solve_minres(
        reduced,
        None,
        projector,
        &context,
        &fixture.right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap();
    assert!(direct.converged);
    assert!(direct.solution.iter().any(|value| *value != 0.0));

    // The reroute: the same call made inside a Krasis transaction over a zero committed state.
    let mut execution = BlockLinearExecution::new(reduced, zero_state(reduced)).unwrap();
    let rerouted = execution
        .solve(
            &context,
            &fixture.right_hand_side,
            None,
            projector,
            &BlockLinearSolver::Minres(config),
            0.0,
        )
        .unwrap();
    assert_eq!(rerouted.algorithm, BlockLinearAlgorithm::Minres);
    assert_eq!(rerouted.restart_cycles, None);
    assert_eq!(rerouted.report, direct, "bit-identical solution and trace");
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        direct.solution
    );
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);
    assert_eq!(execution.state().step(), 1);

    // The checkpoint binds to the reduced system operator's content identity: the plan digest,
    // every bound constitutive closure, and the constraint set.
    let checkpoint = execution.checkpoint().unwrap();
    assert!(
        reduced
            .content_identity()
            .starts_with("finitum-reduced-system:")
    );
    assert_eq!(checkpoint.operator_identity, execution.operator_identity());
}

#[test]
fn gmres_through_krasis_is_bit_identical_on_the_unsigned_system() {
    // The corpus's own (unsigned) authorship is not symmetric; GMRES is the runner's algorithm
    // for it and takes no projector.
    let fixture = steady_fixture(2, BTreeMap::new());
    let reduced = &fixture.reduced;
    assert_eq!(reduced.symmetry(), OperatorSymmetry::Unknown);
    let dimension = reduced.rows();
    let context = EvaluationContext::reproducible();
    let config = GmresConfig::default();

    let direct = solve_gmres(
        reduced,
        None,
        &context,
        &fixture.right_hand_side,
        &vec![0.0; dimension],
        &config,
    )
    .unwrap();
    assert!(direct.converged);

    let mut execution = BlockLinearExecution::new(reduced, zero_state(reduced)).unwrap();
    let rerouted = execution
        .solve(
            &context,
            &fixture.right_hand_side,
            None,
            None,
            &BlockLinearSolver::Gmres(config),
            0.0,
        )
        .unwrap();
    assert_eq!(rerouted.algorithm, BlockLinearAlgorithm::Gmres);
    assert_eq!(rerouted.restart_cycles, Some(direct.restart_cycles));
    assert_eq!(rerouted.report.solution, direct.solution);
    assert_eq!(rerouted.report.trace, direct.trace);
    assert_eq!(
        execution.state().committed_vector().unwrap(),
        direct.solution
    );
}

#[test]
fn a_methodus_refusal_rolls_back_and_surfaces_typed() {
    // Conjugate gradient with `RequireDeclared` refuses the `Unknown`-symmetry unsigned system
    // exactly as it refuses the runner today; Krasis surfaces the refusal and commits nothing.
    let fixture = steady_fixture(1, BTreeMap::new());
    let reduced = &fixture.reduced;
    let mut execution = BlockLinearExecution::new(reduced, zero_state(reduced)).unwrap();
    let before = execution.state().committed_vector().unwrap();
    let error = execution
        .solve(
            &EvaluationContext::reproducible(),
            &fixture.right_hand_side,
            None,
            None,
            &BlockLinearSolver::ConjugateGradient(ConjugateGradientConfig {
                symmetry_policy: ConjugateGradientSymmetryPolicy::RequireDeclared,
                ..ConjugateGradientConfig::default()
            }),
            0.0,
        )
        .unwrap_err();
    assert!(matches!(error, KrasisError::Solve(_)), "{error:?}");
    assert_eq!(execution.state().committed_vector().unwrap(), before);
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);
    assert_eq!(execution.state().step(), 0);
}

#[test]
fn reduced_system_identity_covers_the_constraint_set() {
    let compiled = compile_stokes();
    let mesh = unit_square(1);
    let layout = taylor_hood_layout(&mesh.mesh, compiled.velocity, compiled.pressure);
    let plan =
        SystemRealizationPlan::new(compiled.system.clone(), mesh.mesh.clone(), layout).unwrap();
    let operator = plan
        .bind_kernels(stokes_constitutive(&compiled.system), signed())
        .unwrap();
    let dimension = operator.dimension();
    let unconstrained = operator
        .reduced(ConstraintSet::new(dimension, Vec::new()).unwrap())
        .unwrap();
    let pinned = operator
        .reduced(
            ConstraintSet::new(
                dimension,
                [finitum::AffineConstraint {
                    target: finitum::DofId(0),
                    dependencies: Vec::new(),
                    offset: 0.0,
                }],
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(unconstrained.block_layout(), pinned.block_layout());
    assert_ne!(unconstrained.content_identity(), pinned.content_identity());
    assert_eq!(
        unconstrained.content_identity(),
        operator
            .reduced(ConstraintSet::new(dimension, Vec::new()).unwrap())
            .unwrap()
            .content_identity()
    );
}
