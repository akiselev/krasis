//! W7 (2026-09-05): the FC7 transient verification sources over `CoupledLeaf::reduced_system`.
//!
//! Sinbad's single-equation transient runner builds its `TransientRestart` evidence over a
//! `RealizationPlan` through `CoupledOperator` (`FinitumVerificationSource::check_patch`,
//! `check_rollback_identity`, `check_restart_trajectory`, `check_history_and_rejection`) and its
//! initial state through `initial_state_from`/`NodalContext`; the single compile path
//! (GX-CONTRACTS C12.6 item c) needs the same sources over Finitum's `ReducedSystemOperator`
//! inside `CoupledExecution<CoupledSystemOperator>`. This file wraps the 02-transient-diffusion
//! corpus model once through each path on the same mesh and proves verdicts identical and the
//! trajectories agreeing to the stated tolerance (the two paths integrate the P1 mass with
//! different exact rules, so the arithmetic is not bitwise).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;

use finitum::{
    BlockLayout, ComponentSelection, DofId, DofMap, DynamicExternalInput, ElementRestriction,
    ExternalInput, FieldSource, Mesh, MeshProfile, PointEvaluation, PreparedElement,
    RealizationPlan, ReducedSystemOperator, RegionMap, RegionTagId, SystemConstitutiveInput,
    SystemEssentialConstraintRequirement, SystemRealizationPlan, TaggedMesh,
    essential_constraints_from_selected, essential_constraints_from_system, realize,
};
use krasis::{
    AttemptDisposition, BlockId, CoupledExecution, CoupledLeaf, CoupledOperator,
    CoupledSystemOperator, FinitumVerificationSource, NodalContext, OperatorIdentity, StateBlock,
    StateLayout, TransactionalOperator, check_history_and_rejection, check_restart_trajectory,
    check_rollback_identity, initial_state_from,
};
use methodus::{BdfConfig, BdfOrder, ComparisonTolerance, EvaluationContext, StepOutcome};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, SemanticModel, SymbolId, compile_operator_system, compile_semantics,
    derive_variational_form, factor_operator, infer_form_requirements, lower_operator_kernels,
};

const CORPUS: &str = "/projects/sinbad/sinbad/physics/corpus/02-transient-diffusion.res";
const MODEL: &str = "TransientDiffusion";
const EQUATION: &str = "evolution";
const WALLS: [&str; 4] = ["x_min", "x_max", "y_min", "y_max"];
const SUBDIVISIONS: usize = 4;
const STEP: f64 = 0.025;
const STEPS: usize = 4;
/// Both paths integrate the P1 mass and stiffness exactly (the plan with the degree-2 midpoint
/// rule, the system path with Finitum's degree-4 rule), so they agree to roundoff.
const TRAJECTORY_TOLERANCE: f64 = 1.0e-10;

fn exact_initial(point: &[f64]) -> Vec<f64> {
    vec![(PI * point[0]).sin() * (PI * point[1]).sin()]
}

fn mesh() -> TaggedMesh {
    realize(&MeshProfile::SimplexBox {
        dimension: 2,
        extent: vec![[0.0, 1.0], [0.0, 1.0]],
        subdivisions: vec![SUBDIVISIONS, SUBDIVISIONS],
    })
    .unwrap()
}

fn model(models: &[SemanticModel]) -> &SemanticModel {
    models
        .iter()
        .find(|model| model.name == MODEL)
        .expect("corpus model")
}

fn symbol(model: &SemanticModel, name: &str) -> SymbolId {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name)
        .map(|symbol| symbol.id)
        .unwrap_or_else(|| panic!("model has no symbol {name}"))
}

fn nodal_dof_map(mesh: &Mesh) -> DofMap {
    let restrictions = mesh
        .cells()
        .iter()
        .map(|cell| ElementRestriction {
            dofs: cell.vertices.iter().map(|vertex| DofId(vertex.0)).collect(),
        })
        .collect();
    DofMap::new(mesh.vertices().len(), restrictions).unwrap()
}

/// Sinbad's single-equation path: `RealizationPlan::new_stateful` over the corpus model with
/// unit capacity/diffusivity, zero source and homogeneous walls, wrapped by `CoupledOperator`.
fn plan_path(tagged: &TaggedMesh) -> (CoupledOperator, StateLayout) {
    let source = fs::read_to_string(CORPUS).expect("02-transient-diffusion.res is readable");
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, MODEL, EQUATION).unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let element = PreparedElement::linear_simplex_with_degree(2, 2).unwrap();
    let dofs = nodal_dof_map(&tagged.mesh);
    let mut region_map = RegionMap::new();
    for requirement in &requirements.essential_constraints {
        region_map.insert(requirement.region, WALLS.map(RegionTagId::new));
    }
    let values = vec![FieldSource::constant([0.0]); requirements.essential_constraints.len()];
    let selection = vec![ComponentSelection::All; requirements.essential_constraints.len()];
    let constraints = essential_constraints_from_selected(
        tagged,
        &dofs,
        &requirements.essential_constraints,
        &region_map,
        &values,
        &selection,
    )
    .unwrap();
    let model = model(&compilation.semantic.models);
    let mut stored = Vec::new();
    let mut dynamic = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let name = model.symbols[input.binding.symbol.index()].name.as_str();
            match name {
                "capacity" | "k" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        format!("corpus-02/{name}=1"),
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
                        &tagged.mesh,
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
        tagged.mesh.clone(),
        element,
        dofs,
        constraints,
        stored,
        dynamic,
    )
    .unwrap();
    let layout = StateLayout::new(vec![StateBlock::new(
        BlockId::new("u"),
        0..realization.dimension(),
    )])
    .unwrap();
    let operator = CoupledOperator::new(realization, &layout).unwrap();
    (operator, layout)
}

/// The system path: `SystemRealizationPlan` -> `bind_kernels` -> `reduced` over the same corpus
/// model, mesh, coefficients and walls.
fn system_path(tagged: &TaggedMesh) -> ReducedSystemOperator {
    let source = fs::read_to_string(CORPUS).expect("02-transient-diffusion.res is readable");
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(&compilation.semantic, MODEL, &[EQUATION]).unwrap();
    let model = model(&compilation.semantic.models);
    let field = symbol(model, "u");
    let layout = BlockLayout::new([(field, tagged.mesh.vertices().len(), 1)]).unwrap();
    let plan = SystemRealizationPlan::new(system.clone(), tagged.mesh.clone(), layout).unwrap();
    let mut constitutive = Vec::new();
    for block in &system.blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                if input.source == InputSourceRequirement::Basis {
                    continue;
                }
                let name = model.symbols[input.binding.symbol.index()].name.as_str();
                let value = match name {
                    "capacity" | "k" => 1.0,
                    "f" => 0.0,
                    other => panic!("unexpected non-basis input {other}"),
                };
                constitutive.push(
                    SystemConstitutiveInput::new(
                        block.equation.clone(),
                        integral.integral_index,
                        input.id,
                        1,
                        format!("corpus-02/{name}={value}"),
                        move |_: &PointEvaluation| vec![value],
                        |_: &PointEvaluation, _: &PointEvaluation| vec![0.0],
                    )
                    .unwrap(),
                );
            }
        }
    }
    let operator = plan.bind_kernels(constitutive, BTreeMap::new()).unwrap();
    let mut requirements = Vec::new();
    let mut region_map = RegionMap::new();
    for block in &system.blocks {
        for requirement in &block.factorization.essential_constraints {
            region_map.insert(requirement.region, WALLS.map(RegionTagId::new));
            requirements.push(SystemEssentialConstraintRequirement {
                field: block.row,
                requirement: requirement.clone(),
                value: FieldSource::constant([0.0]),
            });
        }
    }
    let constraints =
        essential_constraints_from_system(&operator, tagged, &region_map, &requirements).unwrap();
    operator.reduced(constraints).unwrap()
}

fn accepting(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::Two,
        absolute_tolerance: 1.0e12,
        relative_tolerance: 1.0e12,
        minimum_step: step,
        maximum_step: step,
        ..BdfConfig::default()
    }
}

fn rejecting(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::Two,
        absolute_tolerance: 1.0e-16,
        relative_tolerance: 1.0e-16,
        minimum_step: 1.0e-8,
        maximum_step: step,
        ..BdfConfig::default()
    }
}

fn accept<Op: TransactionalOperator>(
    execution: &mut CoupledExecution<Op>,
    context: &EvaluationContext,
    config: &BdfConfig,
) {
    let outcome = execution.attempt_step(context, STEP, config).unwrap();
    assert!(matches!(outcome, StepOutcome::Accepted(_)));
}

fn trajectory<Op: TransactionalOperator>(
    execution: &CoupledExecution<Op>,
    context: &EvaluationContext,
    config: &BdfConfig,
) -> Vec<Vec<f64>> {
    let mut execution = execution.clone();
    let mut states = vec![execution.integrator().values.clone()];
    for _ in 0..STEPS {
        accept(&mut execution, context, config);
        states.push(execution.integrator().values.clone());
    }
    states
}

fn max_relative_difference(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    let scale = left
        .iter()
        .flatten()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    assert!(scale > 0.0);
    left.iter()
        .zip(right)
        .flat_map(|(left, right)| left.iter().zip(right).map(|(l, r)| (l - r).abs()))
        .fold(0.0_f64, f64::max)
        / scale
}

#[test]
fn one_leaf_reduced_system_sources_match_the_coupled_operator_path() {
    let tagged = mesh();
    let (plan_operator, plan_layout) = plan_path(&tagged);
    let reduced = system_path(&tagged);
    let leaf = CoupledLeaf::reduced_system("system", reduced.clone()).unwrap();
    let system_operator = CoupledSystemOperator::new(vec![leaf], Vec::new()).unwrap();
    assert_eq!(system_operator.layout().blocks().len(), 1);
    assert_eq!(system_operator.realizations().len(), 1);
    assert_eq!(
        system_operator.realizations()[0].identity(),
        reduced.content_identity()
    );
    let system_block = system_operator.layout().blocks()[0].id().clone();
    assert_eq!(plan_operator.dimension(), system_operator.dimension());

    // Initial state: one `NodalContext` over the shared mesh, the same `FieldSource` on both
    // layouts, and the composition's own per-leaf projection -- all bitwise equal.
    let nodal = NodalContext::new(tagged.mesh.vertices()).unwrap();
    let initial = FieldSource::sampled(exact_initial);
    let plan_state = initial_state_from(
        &plan_layout,
        &nodal,
        2,
        &[(BlockId::new("u"), initial.clone())],
    )
    .unwrap();
    let system_state = initial_state_from(
        system_operator.layout(),
        &nodal,
        2,
        &[(system_block.clone(), initial.clone())],
    )
    .unwrap();
    let projected = system_operator
        .initial_state_from(2, &[(system_block.clone(), initial)])
        .unwrap();
    assert_eq!(
        plan_state.committed_vector().unwrap(),
        system_state.committed_vector().unwrap()
    );
    assert_eq!(
        system_state.committed_vector().unwrap(),
        projected.committed_vector().unwrap()
    );

    let context = EvaluationContext::reproducible();
    let plan_execution = CoupledExecution::new(plan_operator, plan_state, &context).unwrap();
    let system_execution = CoupledExecution::new(system_operator, system_state, &context).unwrap();

    // Trajectories agree to the stated tolerance (different exact quadrature rules).
    let accepting = accepting(STEP);
    let plan_trajectory = trajectory(&plan_execution, &context, &accepting);
    let system_trajectory = trajectory(&system_execution, &context, &accepting);
    let difference = max_relative_difference(&plan_trajectory, &system_trajectory);
    eprintln!("plan vs system trajectory relative difference {difference:e}");
    assert!(
        difference <= TRAJECTORY_TOLERANCE,
        "plan vs system trajectory relative difference {difference:e}"
    );

    // The independent Finitum evidence: the initial condition against the exact solution at
    // `t = 0`, over the plan and over the reduced system operator through the same constructor.
    let tolerance = ComparisonTolerance {
        absolute: 1.0e-12,
        relative: 1.0e-12,
    };
    let plan_source = FinitumVerificationSource::check_patch(
        plan_execution.operator().realization(),
        1,
        &plan_execution.integrator().values,
        tolerance,
        exact_initial,
    )
    .unwrap();
    let system_source = FinitumVerificationSource::check_patch(
        &reduced,
        1,
        &system_execution.integrator().values,
        tolerance,
        exact_initial,
    )
    .unwrap();

    // Rollback: primed by one accepted step so BDF2 has an error estimate to reject against.
    let rejecting = rejecting(STEP);
    let mut plan_primed = plan_execution.clone();
    accept(&mut plan_primed, &context, &accepting);
    let mut system_primed = system_execution.clone();
    accept(&mut system_primed, &context, &accepting);
    let plan_rollback =
        check_rollback_identity(&plan_primed, &context, STEP, &rejecting, &plan_source).unwrap();
    let system_rollback =
        check_rollback_identity(&system_primed, &context, STEP, &rejecting, &system_source)
            .unwrap();
    assert!(plan_rollback.passed, "{plan_rollback:#?}");
    assert!(system_rollback.passed, "{system_rollback:#?}");
    assert_eq!(plan_rollback.disposition, AttemptDisposition::Rejected);
    assert_eq!(system_rollback.disposition, plan_rollback.disposition);
    assert!(plan_rollback.byte_identical && system_rollback.byte_identical);
    assert_eq!(system_rollback.binding.schema, "krasis-verification/2");
    assert_eq!(system_rollback.binding.finitum_sources.len(), 1);
    assert_eq!(
        system_rollback.binding.finitum_sources[0].realization_identity,
        reduced.content_identity()
    );
    assert_eq!(
        system_rollback.binding.finitum_verification_accepted,
        plan_rollback.binding.finitum_verification_accepted
    );
    assert_eq!(
        system_rollback.binding.finitum_verification_accepted,
        Some(true)
    );

    // Restart: split 2 of 4 steps, restore from the serialized checkpoint, bit-exact on both.
    let plan_restart = check_restart_trajectory(
        &plan_execution,
        &context,
        STEP,
        STEPS,
        2,
        &accepting,
        &plan_source,
    )
    .unwrap();
    let system_restart = check_restart_trajectory(
        &system_execution,
        &context,
        STEP,
        STEPS,
        2,
        &accepting,
        &system_source,
    )
    .unwrap();
    for restart in [&plan_restart, &system_restart] {
        assert!(restart.passed, "{restart:#?}");
        assert_eq!(restart.trajectory_l_infinity, 0.0);
        assert_eq!(restart.trajectory_l2_time, 0.0);
        assert!(restart.final_checkpoint_byte_identical);
    }
    assert!(
        system_restart
            .validate(
                &system_execution,
                &context,
                STEP,
                STEPS,
                2,
                &accepting,
                &system_source
            )
            .unwrap()
            .accepted
    );
    let encoded = serde_json::to_vec(&system_restart).unwrap();
    let decoded: krasis::RestartTrajectoryReport = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, system_restart);

    // History synchronization plus a forced rejection: same verdicts and depths.
    let plan_history = check_history_and_rejection(
        &plan_execution,
        &context,
        STEP,
        2,
        &accepting,
        &rejecting,
        &plan_source,
    )
    .unwrap();
    let system_history = check_history_and_rejection(
        &system_execution,
        &context,
        STEP,
        2,
        &accepting,
        &rejecting,
        &system_source,
    )
    .unwrap();
    for history in [&plan_history, &system_history] {
        assert!(history.passed, "{history:#?}");
        assert!(history.synchronized);
        assert!(history.rejection_byte_identical);
        assert_eq!(history.accepted_steps, 2);
    }
    assert_eq!(
        plan_history.rejected_attempt,
        system_history.rejected_attempt
    );
    assert_eq!(
        plan_history
            .field_history_depths
            .iter()
            .map(|(_, depth)| *depth)
            .collect::<Vec<_>>(),
        system_history
            .field_history_depths
            .iter()
            .map(|(_, depth)| *depth)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        system_history.field_history_depths[0].0,
        system_block.to_string()
    );

    // Evidence bound to the other path's realization is refused on both sides.
    let crossed = check_rollback_identity(&system_primed, &context, STEP, &rejecting, &plan_source)
        .unwrap_err();
    assert_eq!(crossed.code, "KRASIS_VERIFY_FINITUM_SOURCE");
    let crossed = check_rollback_identity(&plan_primed, &context, STEP, &rejecting, &system_source)
        .unwrap_err();
    assert_eq!(crossed.code, "KRASIS_VERIFY_FINITUM_SOURCE");
}
