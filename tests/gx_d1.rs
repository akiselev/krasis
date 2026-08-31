//! GX-D1: initial-condition projection, `SymbolId`-linked state blocks, and bounded index-1
//! consistent initialization.

use finitum::{
    AffineConstraint, Cell, ConstraintSet, DofId, DofMap, DynamicExternalInput, ElementRestriction,
    ExternalInput, FieldSource, Mesh, PreparedElement, RealizationPlan, VertexId,
};
use krasis::{
    BlockId, CoupledOperator, FieldId, KrasisError, NodalContext, RowKind, SemanticId,
    StateBinding, StateBlock, StateLayout, initial_state_from,
};
use methodus::{DaeOperator, EvaluationContext, NewtonConfig};
use quantitas::UnitRegistry;
use scientia::{
    InputSourceRequirement, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels,
};

// A pure reaction ODE per DOF (`dt(u) + u = 0`): no boundary term, so the factorization has no
// essential constraints and every DOF is genuinely differential. The FEM discretization still
// couples DOFs through the mass matrix `M`, but because the same `M` multiplies both `dt(u)` and
// `u`, the consistent state rate for any `y0` is exactly `-y0` regardless of `M`'s entries.
const LINEAR_REACTION_MODEL: &str = r#"
module krasis.gx_d1;
model LinearReaction {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  equation ode on Omega { dt(u) + u = 0; }
}
"#;

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

/// Realize `LinearReaction` over an unconstrained square mesh (no Dirichlet rows), reusing the
/// same compile/factor/lower/realize pipeline as the coupled-execution fixtures.
fn linear_reaction_realization(
    subdivisions: usize,
) -> (RealizationPlan, StateLayout, Vec<Vec<f64>>) {
    let compilation =
        compile_semantics(LINEAR_REACTION_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "LinearReaction", "ode").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    assert!(
        factorization.essential_constraints.is_empty(),
        "a reaction-only equation must not require essential constraints"
    );
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, vertices) = square_mesh(subdivisions);
    let element = PreparedElement::linear_simplex(2).unwrap();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            assert_eq!(input.source, InputSourceRequirement::Basis);
        }
    }
    let constraints = ConstraintSet::new(dofs.dof_count(), std::iter::empty()).unwrap();
    let realization = RealizationPlan::new(
        requirements,
        factorization,
        kernels,
        mesh,
        element,
        dofs,
        constraints,
        Vec::new(),
    )
    .unwrap();
    let layout = StateLayout::new(vec![StateBlock::new(
        BlockId::new("u"),
        0..realization.dimension(),
    )])
    .unwrap();
    (realization, layout, vertices)
}

// ---------------------------------------------------------------------------------------------
// Part 3: bounded index-1 consistent initialization.
//
// Fixture: linear diffusion `capacity * dt(u) - div(k * grad(u)) = f` with `capacity = k = 1`
// and `f = 0`, Dirichlet-constrained on the boundary (the same equation shape the coupled
// SV0-B4 fixtures use, specialized to its linear case). On a 3x3 grid this has exactly one
// interior DOF (index 4) and eight boundary DOFs pinned to `0`; marking the boundary DOFs
// `Algebraic` and the interior DOF `Differential` reduces the consistent-initialization Newton
// system to one equation, whose solution is exact after a single (linear) Newton step and can
// be recomputed independently from two direct operator calls.
// ---------------------------------------------------------------------------------------------

const LINEAR_DIFFUSION_MODEL: &str = r#"
module krasis.gx_d1.diffusion;
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

fn square_diffusion_discretization(subdivisions: usize) -> (Mesh, DofMap, ConstraintSet) {
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

/// Realize `LinearDiffusion` (`capacity = k = 1`, `f = 0`) over a 3x3 Dirichlet-constrained
/// square mesh: DOF 4 is the sole interior (differential) DOF, DOFs `{0,1,2,3,5,6,7,8}` are the
/// boundary (algebraic) DOFs, all pinned to `0`.
fn linear_diffusion_realization() -> (RealizationPlan, StateLayout) {
    let compilation =
        compile_semantics(LINEAR_DIFFUSION_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "LinearDiffusion", "evolution").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let (mesh, dofs, constraints) = square_diffusion_discretization(2);
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
                        "k=1;direction=0/v1",
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
    let layout = StateLayout::new(vec![StateBlock::new(
        BlockId::new("u"),
        0..realization.dimension(),
    )])
    .unwrap();
    (realization, layout)
}

const INTERIOR_ROW: usize = 4;
const BOUNDARY_ROWS: [usize; 8] = [0, 1, 2, 3, 5, 6, 7, 8];

fn diffusion_mask() -> Vec<RowKind> {
    let mut mask = vec![RowKind::Algebraic; 9];
    mask[INTERIOR_ROW] = RowKind::Differential;
    mask
}

fn consistent_ic() -> Vec<f64> {
    let mut y0 = vec![0.0; 9];
    y0[INTERIOR_ROW] = 1.0;
    y0
}

#[test]
fn consistent_state_rate_matches_the_analytic_solution() {
    let (realization, layout) = linear_diffusion_realization();
    let operator = CoupledOperator::new(realization, &layout)
        .unwrap()
        .with_consistent_initialization(diffusion_mask(), NewtonConfig::default())
        .unwrap();
    let context = EvaluationContext::reproducible();
    let y0 = consistent_ic();

    let state_rate = operator
        .solve_consistent_state_rate(&context, 0.0, &y0)
        .unwrap();

    // Every boundary (algebraic) row's rate is exactly zero.
    for &row in &BOUNDARY_ROWS {
        assert_eq!(state_rate[row], 0.0);
    }

    // Analytic solution, recomputed independently of `solve_consistent_state_rate`: the
    // equation is linear (`capacity = k = 1`, `f = 0`), so its residual at the interior row is
    // affine in the interior rate alone. `residual_at_zero` is the residual with `ydot = 0`;
    // `slope` is `d(residual_row_4)/d(ydot_row_4)` (a scalar Newton step is then exact).
    let mut residual_at_zero = vec![0.0; 9];
    DaeOperator::residual(
        &operator,
        &context,
        0.0,
        &y0,
        &[0.0; 9],
        &mut residual_at_zero,
    )
    .unwrap();
    let mut direction = vec![0.0; 9];
    direction[INTERIOR_ROW] = 1.0;
    let mut slope_output = vec![0.0; 9];
    DaeOperator::jacobian_vector_product(
        &operator,
        &context,
        0.0,
        &y0,
        &[0.0; 9],
        &[0.0; 9],
        &direction,
        &mut slope_output,
    )
    .unwrap();
    let expected_interior_rate = -residual_at_zero[INTERIOR_ROW] / slope_output[INTERIOR_ROW];
    assert!(
        (state_rate[INTERIOR_ROW] - expected_interior_rate).abs() < 1.0e-9,
        "{} != {expected_interior_rate}",
        state_rate[INTERIOR_ROW]
    );

    // Independent confirmation that the interior row's residual vanishes at the solved rate.
    let mut residual = vec![0.0; 9];
    DaeOperator::residual(&operator, &context, 0.0, &y0, &state_rate, &mut residual).unwrap();
    assert!(residual[INTERIOR_ROW].abs() < 1.0e-9);
}

#[test]
fn make_initial_state_consistent_solves_the_rate_and_leaves_state_untouched() {
    let (realization, layout) = linear_diffusion_realization();
    let operator = CoupledOperator::new(realization, &layout)
        .unwrap()
        .with_consistent_initialization(diffusion_mask(), NewtonConfig::default())
        .unwrap();
    let context = EvaluationContext::reproducible();
    let mut state = consistent_ic();
    let before = state.clone();
    DaeOperator::make_initial_state_consistent(&operator, &context, 0.0, &mut state).unwrap();
    assert_eq!(
        state, before,
        "consistent init only solves the (discarded) state rate, never adjusts state"
    );
}

#[test]
fn make_initial_state_consistent_is_a_no_op_without_a_recorded_mask() {
    let (realization, layout) = linear_diffusion_realization();
    let operator = CoupledOperator::new(realization, &layout).unwrap();
    let context = EvaluationContext::reproducible();
    let mut state = consistent_ic();
    let before = state.clone();
    DaeOperator::make_initial_state_consistent(&operator, &context, 0.0, &mut state).unwrap();
    assert_eq!(
        state, before,
        "no mask recorded => identical to the inherited no-op"
    );
}

#[test]
fn consistent_initialization_refuses_an_inconsistent_algebraic_row() {
    let (realization, layout) = linear_diffusion_realization();
    let operator = CoupledOperator::new(realization, &layout)
        .unwrap()
        .with_consistent_initialization(diffusion_mask(), NewtonConfig::default())
        .unwrap();
    let context = EvaluationContext::reproducible();

    // Boundary DOF 0 is pinned to `0` by the Dirichlet constraint; `5.0` violates it, so the
    // algebraic row's residual does not vanish at this initial state.
    let mut y0 = consistent_ic();
    y0[0] = 5.0;

    let error = operator
        .solve_consistent_state_rate(&context, 0.0, &y0)
        .unwrap_err();
    assert!(
        matches!(error, KrasisError::InvalidCoupling(_)),
        "{error:?}"
    );

    // The trait entry point refuses the same way.
    let mut state = y0;
    let error = DaeOperator::make_initial_state_consistent(&operator, &context, 0.0, &mut state)
        .unwrap_err();
    assert!(format!("{error}").contains("algebraic"), "{error}");
}

#[test]
fn consistent_initialization_mask_length_is_validated_at_construction() {
    let (realization, layout) = linear_diffusion_realization();
    let operator = CoupledOperator::new(realization, &layout).unwrap();
    let error = operator
        .with_consistent_initialization(vec![RowKind::Differential; 3], NewtonConfig::default())
        .unwrap_err();
    assert!(
        matches!(
            error,
            KrasisError::ConsistentInitializationMaskLength {
                actual: 3,
                expected: 9
            }
        ),
        "{error:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Part 2: `SymbolId`-linked state blocks.
// ---------------------------------------------------------------------------------------------

#[test]
fn state_binding_is_total_and_injective_and_resolves_both_directions() {
    let layout = StateLayout::new(vec![
        StateBlock::new(BlockId::new("a"), 0..2),
        StateBlock::new(BlockId::new("b"), 2..4),
    ])
    .unwrap();
    let binding = StateBinding::new(
        &layout,
        vec![
            (SemanticId::new(11), BlockId::new("a")),
            (SemanticId::new(22), BlockId::new("b")),
        ],
    )
    .unwrap();
    assert_eq!(
        binding.block_for(SemanticId::new(11)),
        Some(&BlockId::new("a"))
    );
    assert_eq!(
        binding.block_for(SemanticId::new(22)),
        Some(&BlockId::new("b"))
    );
    assert_eq!(
        binding.semantic_for(&BlockId::new("a")),
        Some(SemanticId::new(11))
    );
    assert_eq!(
        binding.semantic_for(&BlockId::new("b")),
        Some(SemanticId::new(22))
    );
    assert_eq!(binding.blocks().count(), 2);
}

#[test]
fn state_binding_refuses_unknown_incomplete_and_duplicate_bindings() {
    let layout = StateLayout::new(vec![
        StateBlock::new(BlockId::new("a"), 0..2),
        StateBlock::new(BlockId::new("b"), 2..4),
    ])
    .unwrap();

    let error = StateBinding::new(&layout, vec![(SemanticId::new(1), BlockId::new("missing"))])
        .unwrap_err();
    assert!(matches!(error, KrasisError::StateBindingUnknownBlock(_)));

    let error =
        StateBinding::new(&layout, vec![(SemanticId::new(1), BlockId::new("a"))]).unwrap_err();
    assert!(matches!(error, KrasisError::StateBindingIncomplete));

    let error = StateBinding::new(
        &layout,
        vec![
            (SemanticId::new(1), BlockId::new("a")),
            (SemanticId::new(2), BlockId::new("a")),
        ],
    )
    .unwrap_err();
    assert!(matches!(error, KrasisError::StateBindingDuplicateBlock(_)));

    let error = StateBinding::new(
        &layout,
        vec![
            (SemanticId::new(1), BlockId::new("a")),
            (SemanticId::new(1), BlockId::new("b")),
        ],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        KrasisError::StateBindingDuplicateSemanticId(1)
    ));
}

#[test]
fn coupled_operator_with_bindings_changes_identity_additively() {
    let (realization, layout, _) = linear_reaction_realization(1);
    let unbound = CoupledOperator::new(realization.clone(), &layout).unwrap();
    assert!(unbound.identity().starts_with("krasis-coupled/1:"));
    assert!(unbound.state_binding().is_none());

    let binding =
        StateBinding::new(&layout, vec![(SemanticId::new(7), BlockId::new("u"))]).unwrap();
    let bound = CoupledOperator::new_with_bindings(realization, &layout, Some(binding)).unwrap();
    assert!(bound.identity().starts_with("krasis-coupled/2:"));
    assert_ne!(unbound.identity(), bound.identity());
    assert_eq!(
        bound.state_binding().unwrap().block_for(SemanticId::new(7)),
        Some(&BlockId::new("u"))
    );
}

#[test]
fn coupled_operator_refuses_a_binding_whose_blocks_do_not_match_the_layout() {
    let (realization, layout, _) = linear_reaction_realization(1);
    let other_layout = StateLayout::new(vec![
        StateBlock::new(BlockId::new("a"), 0..2),
        StateBlock::new(BlockId::new("b"), 2..4),
    ])
    .unwrap();
    let binding = StateBinding::new(
        &other_layout,
        vec![
            (SemanticId::new(1), BlockId::new("a")),
            (SemanticId::new(2), BlockId::new("b")),
        ],
    )
    .unwrap();
    let error =
        CoupledOperator::new_with_bindings(realization, &layout, Some(binding)).unwrap_err();
    assert!(matches!(error, KrasisError::StateBindingLayoutMismatch));
}

// ---------------------------------------------------------------------------------------------
// Part 1: initial-condition projection onto a P1 nodal DOF map.
// ---------------------------------------------------------------------------------------------

fn scalar_layout(vertex_count: usize) -> StateLayout {
    StateLayout::new(vec![StateBlock::new(BlockId::new("u"), 0..vertex_count)]).unwrap()
}

fn sample_vertices() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ]
}

#[test]
fn initial_state_from_projects_constant_and_sampled_sources() {
    let vertices = sample_vertices();
    let layout = scalar_layout(vertices.len());
    let nodal = NodalContext::new(&vertices).unwrap();

    let bindings = vec![(
        BlockId::new("u"),
        FieldSource::sampled(|coordinates| vec![coordinates[0] + 2.0 * coordinates[1]]),
    )];
    let state = initial_state_from(&layout, &nodal, 2, &bindings).unwrap();
    let expected: Vec<f64> = vertices.iter().map(|c| c[0] + 2.0 * c[1]).collect();
    assert_eq!(
        state.committed(&FieldId::new("u")).unwrap(),
        expected.as_slice()
    );

    let bindings = vec![(BlockId::new("u"), FieldSource::constant(vec![3.5]))];
    let state = initial_state_from(&layout, &nodal, 2, &bindings).unwrap();
    assert_eq!(
        state.committed(&FieldId::new("u")).unwrap(),
        [3.5, 3.5, 3.5, 3.5]
    );

    let nodal_values = vec![1.0, 2.0, 3.0, 4.0];
    let bindings = vec![(BlockId::new("u"), FieldSource::nodal(nodal_values.clone()))];
    let state = initial_state_from(&layout, &nodal, 2, &bindings).unwrap();
    assert_eq!(
        state.committed(&FieldId::new("u")).unwrap(),
        nodal_values.as_slice()
    );
}

#[test]
fn initial_state_from_projects_a_vector_valued_block() {
    let vertices = sample_vertices();
    let layout = StateLayout::new(vec![StateBlock::new(
        BlockId::new("velocity"),
        0..vertices.len() * 2,
    )])
    .unwrap();
    let nodal = NodalContext::new(&vertices).unwrap();
    let bindings = vec![(
        BlockId::new("velocity"),
        FieldSource::sampled(|coordinates| vec![coordinates[0], -coordinates[1]]),
    )];
    let state = initial_state_from(&layout, &nodal, 0, &bindings).unwrap();
    let mut expected = Vec::new();
    for coordinates in &vertices {
        expected.push(coordinates[0]);
        expected.push(-coordinates[1]);
    }
    assert_eq!(
        state.committed(&FieldId::new("velocity")).unwrap(),
        expected.as_slice()
    );
}

#[test]
fn initial_state_from_refuses_missing_extra_and_duplicate_blocks() {
    let vertices = sample_vertices();
    let layout = scalar_layout(vertices.len());
    let nodal = NodalContext::new(&vertices).unwrap();

    let error = initial_state_from(&layout, &nodal, 0, &[]).unwrap_err();
    assert!(matches!(error, KrasisError::InitialBlockMissing(_)));

    let bindings = vec![
        (BlockId::new("u"), FieldSource::constant(vec![1.0])),
        (BlockId::new("extra"), FieldSource::constant(vec![1.0])),
    ];
    let error = initial_state_from(&layout, &nodal, 0, &bindings).unwrap_err();
    assert!(matches!(error, KrasisError::InitialBlockUnknown(_)));

    let bindings = vec![
        (BlockId::new("u"), FieldSource::constant(vec![1.0])),
        (BlockId::new("u"), FieldSource::constant(vec![2.0])),
    ];
    let error = initial_state_from(&layout, &nodal, 0, &bindings).unwrap_err();
    assert!(matches!(error, KrasisError::InitialBlockDuplicate(_)));
}

#[test]
fn initial_state_from_refuses_dimension_mismatch_and_non_finite_values() {
    let vertices = sample_vertices();
    let layout = StateLayout::new(vec![StateBlock::new(BlockId::new("u"), 0..5)]).unwrap();
    let nodal = NodalContext::new(&vertices).unwrap();
    let bindings = vec![(BlockId::new("u"), FieldSource::constant(vec![1.0]))];
    let error = initial_state_from(&layout, &nodal, 0, &bindings).unwrap_err();
    assert!(matches!(
        error,
        KrasisError::InitialDimensionMismatch { .. }
    ));

    let layout = scalar_layout(vertices.len());
    let bindings = vec![(BlockId::new("u"), FieldSource::constant(vec![1.0, 2.0]))];
    let error = initial_state_from(&layout, &nodal, 0, &bindings).unwrap_err();
    assert!(matches!(error, KrasisError::FieldLength { .. }));

    let bindings = vec![(BlockId::new("u"), FieldSource::sampled(|_| vec![f64::NAN]))];
    let error = initial_state_from(&layout, &nodal, 0, &bindings).unwrap_err();
    assert!(matches!(error, KrasisError::NonFiniteValue { .. }));
}

#[test]
fn nodal_context_refuses_empty_and_inconsistent_coordinates() {
    let empty: Vec<Vec<f64>> = Vec::new();
    let error = NodalContext::new(&empty).unwrap_err();
    assert!(matches!(error, KrasisError::EmptyNodalContext));

    let inconsistent = vec![vec![0.0, 0.0], vec![1.0]];
    let error = NodalContext::new(&inconsistent).unwrap_err();
    assert!(matches!(
        error,
        KrasisError::InconsistentNodalCoordinates { .. }
    ));
}
