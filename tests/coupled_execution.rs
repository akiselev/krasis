use finitum::{
    AffineConstraint, Cell, ConstraintSet, DofId, DofMap, DynamicExternalInput, ElementRestriction,
    ExternalInput, Mesh, PreparedElement, RealizationPlan, VertexId,
};
use krasis::{
    BlockId, CoupledCheckpoint, CoupledExecution, CoupledOperator, FieldId, SimulationState,
    StateBlock, StateLayout, TransactionPhase,
};
use methodus::{
    BdfConfig, BdfOrder, BlockNonlinearOperator, EvaluationContext, StepOutcome, verify_dae_jvp,
    verify_jvp,
};
use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, InputSourceRequirement, compile_semantics, derive_variational_form,
    factor_operator, infer_form_requirements, lower_operator_kernels,
};

const MODEL: &str = r#"
module fc7.krasis;
model TransientNonlinear {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  property capacity = storage_capacity(u);
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation evolution on Omega { capacity * dt(u) - div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(t); }
}
"#;

#[test]
fn real_composition_implements_nonlinear_block_and_dae_derivatives() {
    let (operator, _) = coupled_fixture();
    assert_eq!(operator.block_layout().dimension(), operator.dimension());
    assert_eq!(operator.block_layout().blocks()[0].name(), "u");
    let context = EvaluationContext::reproducible();
    let state = interior_vector(0.35);
    let rate = interior_vector(-0.17);
    let state_direction = interior_vector(0.73);
    let rate_direction = interior_vector(-0.41);
    assert!(verify_jvp(&operator, &context, &state, &state_direction, 1.0e-6).unwrap() < 2.0e-9);
    assert!(
        verify_dae_jvp(
            &operator,
            &context,
            0.3,
            &state,
            &rate,
            &state_direction,
            &rate_direction,
            1.0e-6,
        )
        .unwrap()
            < 2.0e-9
    );
}

#[test]
fn accepted_rejected_and_restarted_steps_preserve_transactional_history() {
    let context = EvaluationContext::reproducible();
    let (operator, state) = coupled_fixture();
    let mut execution = CoupledExecution::new(operator.clone(), state, &context).unwrap();
    let fixed = fixed_config(0.05);

    let first = execution.attempt_step(&context, 0.05, &fixed).unwrap();
    assert!(matches!(first, StepOutcome::Accepted(_)));
    assert_eq!(execution.state().phase(), TransactionPhase::Committed);
    assert_eq!(execution.state().step(), 1);
    assert_eq!(
        execution.state().history(&FieldId::new("u")).unwrap()[0],
        interior_vector(1.0)
    );

    let before_rejection = serde_json::to_vec(&execution.checkpoint().unwrap()).unwrap();
    let strict = BdfConfig {
        order: BdfOrder::Two,
        absolute_tolerance: 1.0e-16,
        relative_tolerance: 1.0e-16,
        minimum_step: 1.0e-8,
        maximum_step: 0.05,
        ..BdfConfig::default()
    };
    let rejected = execution.attempt_step(&context, 0.05, &strict).unwrap();
    assert!(matches!(rejected, StepOutcome::Rejected(_)));
    assert_eq!(
        serde_json::to_vec(&execution.checkpoint().unwrap()).unwrap(),
        before_rejection
    );

    let mut continuous = execution.clone();
    for _ in 0..3 {
        let outcome = continuous.attempt_step(&context, 0.05, &fixed).unwrap();
        assert!(matches!(outcome, StepOutcome::Accepted(_)));
    }

    let midpoint = execution.attempt_step(&context, 0.05, &fixed).unwrap();
    assert!(matches!(midpoint, StepOutcome::Accepted(_)));
    let encoded = serde_json::to_vec(&execution.checkpoint().unwrap()).unwrap();
    let checkpoint: CoupledCheckpoint = serde_json::from_slice(&encoded).unwrap();
    let (_, fresh_state) = coupled_fixture();
    let mut restarted = CoupledExecution::new(operator, fresh_state, &context).unwrap();
    restarted.restore(&checkpoint).unwrap();
    for _ in 0..2 {
        let outcome = restarted.attempt_step(&context, 0.05, &fixed).unwrap();
        assert!(matches!(outcome, StepOutcome::Accepted(_)));
    }
    assert_eq!(
        serde_json::to_vec(&continuous.checkpoint().unwrap()).unwrap(),
        serde_json::to_vec(&restarted.checkpoint().unwrap()).unwrap()
    );
}

#[test]
fn checkpoint_refuses_same_size_geometry_and_material_changes() {
    let context = EvaluationContext::reproducible();
    let (source_operator, source_state) = coupled_fixture();
    let source = CoupledExecution::new(source_operator, source_state, &context).unwrap();
    let checkpoint = source.checkpoint().unwrap();

    let (material_operator, material_state) =
        coupled_fixture_variant("k=1+0.2u;direction=0.2du/v1", 0.2, 0.0);
    assert_ne!(source.operator().identity(), material_operator.identity());
    assert_ne!(
        source.operator().realization().digest(),
        material_operator.realization().digest()
    );
    let mut material = CoupledExecution::new(material_operator, material_state, &context).unwrap();
    let material_before = material.checkpoint().unwrap();
    assert!(material.restore(&checkpoint).is_err());
    assert_eq!(material.checkpoint().unwrap(), material_before);

    let (geometry_operator, geometry_state) =
        coupled_fixture_variant("k=1+0.1u;direction=0.1du/v1", 0.1, 0.03);
    assert_ne!(source.operator().identity(), geometry_operator.identity());
    assert_ne!(
        source.operator().realization().digest(),
        geometry_operator.realization().digest()
    );
    let mut geometry = CoupledExecution::new(geometry_operator, geometry_state, &context).unwrap();
    let geometry_before = geometry.checkpoint().unwrap();
    assert!(geometry.restore(&checkpoint).is_err());
    assert_eq!(geometry.checkpoint().unwrap(), geometry_before);
}

fn coupled_fixture() -> (CoupledOperator, SimulationState) {
    coupled_fixture_variant("k=1+0.1u;direction=0.1du/v1", 0.1, 0.0)
}

fn coupled_fixture_variant(
    conductivity_identity: &str,
    conductivity_slope: f64,
    center_shift: f64,
) -> (CoupledOperator, SimulationState) {
    let compilation = compile_semantics(MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "TransientNonlinear", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, constraints) = square_discretization(2, center_shift);
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
                "capacity" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        "capacity=1;direction=0/v1",
                        |_| vec![1.0],
                        |_, _| vec![0.0],
                    )
                    .unwrap(),
                ),
                "k" => dynamic.push(
                    DynamicExternalInput::new(
                        integral.integral_index,
                        input.id,
                        1,
                        conductivity_identity,
                        move |evaluation| {
                            vec![
                                1.0 + conductivity_slope
                                    * evaluation.values(DerivativeEvaluation::Value).unwrap()[0],
                            ]
                        },
                        move |_, direction| {
                            vec![
                                conductivity_slope
                                    * direction.values(DerivativeEvaluation::Value).unwrap()[0],
                            ]
                        },
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
    let layout = StateLayout::new(vec![StateBlock::new(
        BlockId::new("u"),
        0..realization.dimension(),
    )])
    .unwrap();
    let operator = CoupledOperator::new(realization, &layout).unwrap();
    let mut state = SimulationState::new(layout, 2);
    state
        .insert_field(FieldId::new("u"), interior_vector(1.0))
        .unwrap();
    state.insert_constitutive("material", vec![7.0]).unwrap();
    (operator, state)
}

fn fixed_config(step: f64) -> BdfConfig {
    BdfConfig {
        order: BdfOrder::Two,
        absolute_tolerance: 1.0e12,
        relative_tolerance: 1.0e12,
        minimum_step: step,
        maximum_step: step,
        ..BdfConfig::default()
    }
}

fn interior_vector(value: f64) -> Vec<f64> {
    vec![0.0, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0, 0.0]
}

fn square_discretization(subdivisions: usize, center_shift: f64) -> (Mesh, DofMap, ConstraintSet) {
    let width = subdivisions + 1;
    let mut vertices = (0..=subdivisions)
        .flat_map(|row| {
            (0..=subdivisions).map(move |column| {
                vec![
                    column as f64 / subdivisions as f64,
                    row as f64 / subdivisions as f64,
                ]
            })
        })
        .collect::<Vec<_>>();
    vertices[(width * width) / 2][0] += center_shift;
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
    let mesh = Mesh::new(2, vertices, cells).unwrap();
    let dofs = DofMap::new(width * width, restrictions).unwrap();
    let constraints = ConstraintSet::new(
        width * width,
        (0..width * width)
            .filter(|index| {
                let row = index / width;
                let column = index % width;
                row == 0 || column == 0 || row == subdivisions || column == subdivisions
            })
            .map(|target| AffineConstraint {
                target: DofId(target),
                dependencies: Vec::new(),
                offset: 0.0,
            }),
    )
    .unwrap();
    (mesh, dofs, constraints)
}
