//! W10 Krasis packages 1, 2 and 4 over the real `ConnectionRealizationPlan`-based two-leaf
//! heat-heat fixtures in `tests/support` (matching and `nested_p1`, composed through
//! `ConnectedSystemOperator::new`):
//!
//! - package 1: the connected Jacobian diagonal is Finitum's `diag(Rᵀ J R)` -- `Some` for the
//!   weighted `nested_p1` interface (it was `None` before W10), equal to bounded exhaustive
//!   unit-vector probing of Krasis's own `jacobian_vector_product` composition (with a nonzero
//!   `rate_shift` on the transient fixture), equal to the sum-of-leaf-diagonals convention on
//!   `matching`, and good enough for `JacobiFactory` to precondition GMRES on `nested_p1`;
//! - package 2: the partitioned agreement gate (`sinbad/ARCHITECTURE.md` gate item 9) on the
//!   constraint-eliminated matching fixture and on the `nested_p1` pair, serial and parallel,
//!   under plain iteration and under Aitken, with a refused solve leaving the committed state
//!   bit-identical, plus the transient (BDF) per-accepted-step agreement on the matching pair;
//! - package 4: the block coupling actions equal the masked full JVP and GMRES with the block
//!   Gauss-Seidel factory agrees with the Jacobi-preconditioned solve in no more Krylov
//!   iterations.

mod support;

use krasis::{
    BlockGaussSeidelFactory, ConnectedSystemOperator, CoupledExecution, KrasisError,
    PARTITIONED_MAX_SWEEPS, PartitionedConfig, PartitionedDisposition, PartitionedExecution,
    PartitionedFixedPoint, PartitionedIteration, PartitionedSchedule,
};
use methodus::{
    AccelerationMethod, BdfConfig, BdfOrder, BlockCouplingAction, BlockNonlinearOperator,
    DaeOperator, EvaluationContext, GaussSeidelSweep, GmresConfig, JacobiFactory, JacobianOperator,
    KrylovMethod, NewtonConfig, NewtonKrylovConfig, NewtonKrylovReport, NonlinearOperator,
    OperatorSymmetry, PreconditionerFactory, StepOutcome, solve_newton, solve_newton_krylov,
};
use support::{
    ConnectedFixture, Interface, Storage, affine, max_abs_difference, steady_fixture,
    transient_fixture,
};

const MATCHING: Interface = Interface::Matching { subdivisions: 2 };
const NESTED: Interface = Interface::NestedP1 { coarse: 1, fine: 2 };
const DIRICHLET: [f64; 2] = [300.0, 350.0];

fn newton() -> NewtonConfig {
    NewtonConfig {
        max_iterations: 200,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        ..NewtonConfig::default()
    }
}

fn implicit(
    schedule: PartitionedSchedule,
    acceleration: AccelerationMethod,
    tolerance: f64,
    max_sweeps: usize,
) -> PartitionedConfig {
    PartitionedConfig {
        schedule,
        iteration: PartitionedIteration::Implicit {
            tolerance,
            max_sweeps,
            acceleration,
        },
        newton: newton(),
        divergence_growth: 10.0,
    }
}

fn plain() -> AccelerationMethod {
    AccelerationMethod::FixedRelaxation { factor: 1.0 }
}

fn krylov_config() -> NewtonKrylovConfig {
    NewtonKrylovConfig {
        max_iterations: 50,
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        ..NewtonKrylovConfig::default()
    }
}

fn gmres() -> KrylovMethod {
    KrylovMethod::Gmres(GmresConfig::default())
}

/// `diag(Rᵀ J R)` by bounded exhaustive probing of the connected operator's own
/// `jacobian_vector_product` with every unit direction `e_i` and the rate direction
/// `rate_shift · e_i` (the shifted column the diagonal linearizes).
fn probe_diagonal(
    operator: &ConnectedSystemOperator,
    context: &EvaluationContext,
    time: f64,
    state: &[f64],
    rate: &[f64],
    rate_shift: f64,
) -> Vec<f64> {
    let n = DaeOperator::dimension(operator);
    let mut unit = vec![0.0; n];
    let mut rate_direction = vec![0.0; n];
    let mut output = vec![0.0; n];
    let mut diagonal = vec![0.0; n];
    for probe in 0..n {
        unit[probe] = 1.0;
        rate_direction[probe] = rate_shift;
        DaeOperator::jacobian_vector_product(
            operator,
            context,
            time,
            state,
            rate,
            &unit,
            &rate_direction,
            &mut output,
        )
        .unwrap();
        diagonal[probe] = output[probe];
        unit[probe] = 0.0;
        rate_direction[probe] = 0.0;
    }
    diagonal
}

/// Entrywise agreement to `1e-9` relative; returns the worst relative discrepancy.
fn assert_diagonal_matches(label: &str, diagonal: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(diagonal.len(), expected.len(), "{label}: extent");
    let mut worst = 0.0_f64;
    for (row, (&got, &want)) in diagonal.iter().zip(expected).enumerate() {
        let relative = (got - want).abs() / want.abs().max(1.0);
        worst = worst.max(relative);
        assert!(
            relative <= 1e-9,
            "{label}: row {row}: diagonal {got} != exhaustive JVP probe {want} (relative \
             {relative:.3e})"
        );
    }
    worst
}

/// Finitum's own `system_linearized_diagonal` over the fixture's reduced operators, fed the
/// per-leaf slices of the connection-reduced point exactly as Krasis feeds it.
fn finitum_diagonal(
    fixture: &ConnectedFixture,
    state: &[f64],
    rate: &[f64],
    rate_shift: f64,
) -> Vec<f64> {
    let split = fixture.dimensions[0];
    fixture
        .plan
        .system_linearized_diagonal(
            [&fixture.reduced[0], &fixture.reduced[1]],
            0.0,
            [&state[..split], &state[split..]],
            [&rate[..split], &rate[split..]],
            rate_shift,
        )
        .unwrap()
        .expect("Finitum admits the connected diagonal")
}

fn assert_target_rows_are_identity(fixture: &ConnectedFixture, diagonal: &[f64]) {
    let targets = fixture.targets();
    assert!(!targets.is_empty());
    for target in targets {
        assert_eq!(diagonal[target], 1.0, "target row {target}");
    }
}

/// The affine linearization point every steady test uses (the problem is linear, so the
/// diagonal does not depend on it; the point still exercises the expansion plumbing).
fn steady_point(fixture: &ConnectedFixture) -> Vec<f64> {
    fixture.sampled(|p| affine(p, 300.0, 50.0, 0.0, 0.0))
}

// -------------------------------------------------------------------------------------------
// Package 1: the connected diagonal is Finitum's
// -------------------------------------------------------------------------------------------

#[test]
fn matching_connected_diagonal_is_finitums_and_equals_the_sum_of_leaf_convention() {
    let fixture = steady_fixture(MATCHING, [2.5, 3.0], DIRICHLET);
    assert!(fixture.operator.has_connected_diagonal());
    assert!(!fixture.is_weighted());
    let context = EvaluationContext::reproducible();
    let state = steady_point(&fixture);
    let rate = vec![0.0; fixture.dimension()];
    let rate_shift = 1.0;

    let diagonal =
        DaeOperator::jacobian_diagonal(&fixture.operator, &context, 0.0, &state, &rate, rate_shift)
            .unwrap()
            .expect("matching conduction admits a connected diagonal");
    let expected = probe_diagonal(&fixture.operator, &context, 0.0, &state, &rate, rate_shift);
    let worst = assert_diagonal_matches("matching", &diagonal, &expected);
    eprintln!("matching connected diagonal: worst relative discrepancy vs probing {worst:.3e}");
    assert_target_rows_are_identity(&fixture, &diagonal);
    assert_eq!(
        diagonal,
        finitum_diagonal(&fixture, &state, &rate, rate_shift),
        "the Krasis path is Finitum's connected diagonal, scattered into the leaf ranges"
    );

    // The pre-W10 Krasis convention for weight-1 bijective pairs: the leaves' own reduced
    // diagonals at the connection-expanded point, restricted by the pair's transpose, identity
    // targets. Finitum proved its path equals this to 1e-12; so does Krasis's composition.
    let physical = fixture.pair.expand(&state).unwrap();
    let physical_rate = fixture.pair.expand_homogeneous(&rate).unwrap();
    let mut leaves = Vec::with_capacity(fixture.dimension());
    let mut offset = 0;
    for side in 0..2 {
        let n = fixture.dimensions[side];
        leaves.extend(
            DaeOperator::jacobian_diagonal(
                &fixture.reduced[side],
                &context,
                0.0,
                &physical[offset..offset + n],
                &physical_rate[offset..offset + n],
                rate_shift,
            )
            .unwrap()
            .expect("leaf diagonal"),
        );
        offset += n;
    }
    let mut convention = fixture.pair.restrict_transpose(&leaves).unwrap();
    for target in fixture.targets() {
        convention[target] = 1.0;
    }
    for (row, (&got, &want)) in diagonal.iter().zip(&convention).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "row {row}: connected diagonal {got} != sum-of-leaf-diagonals convention {want}"
        );
    }
    // Non-vacuity: a master row genuinely carries both leaves' contributions.
    let masters = fixture.masters();
    assert!(!masters.is_empty());
    assert!(masters.iter().any(|&master| {
        (diagonal[master] - leaves[master]).abs() > 1e-9 * leaves[master].abs().max(1.0)
    }));
}

#[test]
fn nested_p1_connected_diagonal_is_some_and_matches_exhaustive_probing() {
    let fixture = steady_fixture(NESTED, [2.0, 3.0], DIRICHLET);
    assert!(fixture.operator.has_connected_diagonal());
    // The shape whose diagonal was `None` before W10: weighted interpolation rows.
    assert!(fixture.is_weighted());
    let context = EvaluationContext::reproducible();
    let state = steady_point(&fixture);
    let rate = vec![0.0; fixture.dimension()];
    let rate_shift = 1.0;

    let diagonal =
        DaeOperator::jacobian_diagonal(&fixture.operator, &context, 0.0, &state, &rate, rate_shift)
            .unwrap()
            .expect("nested_p1 conduction admits a connected diagonal since W10");
    let expected = probe_diagonal(&fixture.operator, &context, 0.0, &state, &rate, rate_shift);
    let worst = assert_diagonal_matches("steady nested_p1", &diagonal, &expected);
    eprintln!("nested_p1 connected diagonal: worst relative discrepancy vs probing {worst:.3e}");
    assert_target_rows_are_identity(&fixture, &diagonal);
    assert_eq!(
        diagonal,
        finitum_diagonal(&fixture, &state, &rate, rate_shift)
    );

    // Non-vacuity of the cross term: an interface-adjacent coarse master row differs from the
    // coarse leaf's own reduced diagonal at the same point.
    let physical = fixture.pair.expand(&state).unwrap();
    let coarse = DaeOperator::jacobian_diagonal(
        &fixture.reduced[0],
        &context,
        0.0,
        &physical[..fixture.dimensions[0]],
        &rate[..fixture.dimensions[0]],
        rate_shift,
    )
    .unwrap()
    .unwrap();
    assert!(fixture.masters().iter().any(|&master| {
        (diagonal[master] - coarse[master]).abs() > 1e-9 * coarse[master].abs().max(1.0)
    }));
    // The steady `NonlinearOperator` view (what `JacobiFactory` reads) is the same diagonal at
    // zero rate shift.
    let steady = NonlinearOperator::jacobian_diagonal(&fixture.operator, &context, &state)
        .unwrap()
        .unwrap();
    let steady_expected = probe_diagonal(&fixture.operator, &context, 0.0, &state, &rate, 0.0);
    assert_diagonal_matches("steady nested_p1 (shift 0)", &steady, &steady_expected);
}

#[test]
fn nested_p1_transient_connected_diagonal_sees_rate_shift_at_the_expanded_point() {
    let storage = [
        Storage {
            k0: 2.0,
            k1: 1.0e-3,
            rho: 1.5,
            cp: 0.5,
        },
        Storage {
            k0: 3.0,
            k1: 2.0e-3,
            rho: 2.0,
            cp: 0.75,
        },
    ];
    let fixture = transient_fixture(NESTED, storage, DIRICHLET);
    assert!(fixture.is_weighted());
    let context = EvaluationContext::reproducible();
    let mut state = fixture.sampled(|p| affine(p, 300.0, 20.0, 10.0, -5.0));
    let mut rate = fixture.sampled(|p| affine(p, 5.0, 1.0, 0.25, -0.75));
    // A connection-reduced state owns no meaning at the target rows: hold a deliberately wrong
    // value there, which the composition must never read (it expands through the pair first).
    for target in fixture.targets() {
        state[target] = 0.0;
        rate[target] = 0.0;
    }
    let time = 0.3;
    let targets = fixture.targets();
    let identity_rows: Vec<usize> = targets
        .iter()
        .copied()
        .chain(fixture.essential[0].iter().copied())
        .chain(
            fixture.essential[1]
                .iter()
                .map(|&vertex| fixture.dimensions[0] + vertex),
        )
        .collect();

    let mut diagonals = Vec::new();
    for rate_shift in [0.0, 3.0] {
        let diagonal = DaeOperator::jacobian_diagonal(
            &fixture.operator,
            &context,
            time,
            &state,
            &rate,
            rate_shift,
        )
        .unwrap()
        .expect("storage-bearing nested_p1 conduction admits a connected diagonal");
        let expected = probe_diagonal(&fixture.operator, &context, time, &state, &rate, rate_shift);
        let label = format!("transient nested_p1 at rate_shift {rate_shift}");
        let worst = assert_diagonal_matches(&label, &diagonal, &expected);
        eprintln!("{label}: worst relative discrepancy vs probing {worst:.3e}");
        assert_target_rows_are_identity(&fixture, &diagonal);
        diagonals.push(diagonal);
    }
    // Every storage-bearing row grows with the shift; identity rows stay 1.0.
    for (row, (&unshifted, &shifted)) in diagonals[0].iter().zip(&diagonals[1]).enumerate() {
        if identity_rows.contains(&row) {
            assert_eq!(unshifted, 1.0);
            assert_eq!(shifted, 1.0);
        } else {
            assert!(
                shifted - unshifted > 1e-9 * unshifted.abs().max(1.0),
                "row {row}: diagonal did not grow with rate_shift ({unshifted} -> {shifted})"
            );
        }
    }
}

/// GMRES Newton-Krylov on `fixture` from zero, unpreconditioned and with `JacobiFactory` over
/// the connected diagonal: both converge and agree with each other and with dense Newton to
/// `1e-8`; returns the Krylov iteration counts `[unpreconditioned, jacobi]`.
fn jacobi_krylov_counts(fixture: &ConnectedFixture, label: &str) -> [usize; 2] {
    let context = EvaluationContext::reproducible();
    let zero = vec![0.0; fixture.dimension()];
    let dense = solve_newton(&fixture.operator, &context, &zero, &newton()).unwrap();
    assert!(dense.converged);
    let unpreconditioned = krylov_with(fixture, &context, None);
    let jacobi = krylov_with(fixture, &context, Some(&JacobiFactory));
    let disagreement = max_abs_difference(&jacobi.state, &unpreconditioned.state);
    let dense_disagreement = max_abs_difference(&jacobi.state, &dense.state);
    eprintln!(
        "{label} GMRES: unpreconditioned {} Krylov iterations, Jacobi {}; Jacobi vs \
         unpreconditioned {disagreement:e}, vs dense Newton {dense_disagreement:e}",
        unpreconditioned.linear_iterations(),
        jacobi.linear_iterations()
    );
    assert!(disagreement < 1.0e-8, "{label}: {disagreement}");
    assert!(dense_disagreement < 1.0e-8, "{label}: {dense_disagreement}");
    [
        unpreconditioned.linear_iterations(),
        jacobi.linear_iterations(),
    ]
}

#[test]
fn nested_p1_newton_krylov_with_jacobi_converges_and_agrees_with_the_unpreconditioned_solve() {
    // Finitum's own fixture coefficients (k = 2 coarse, k = 3 fine) and a conductivity
    // contrast (k = 1 coarse, k = 50 fine, a two-decade spread in the diagonal across the
    // interface). With the far faces fully pinned, Jacobi over the connected diagonal costs no
    // more Krylov iterations than no preconditioning on either; the counts are printed.
    for (label, conductivity) in [
        ("nested_p1 k=[2,3]", [2.0, 3.0]),
        ("nested_p1 k=[1,50]", [1.0, 50.0]),
    ] {
        let [unpreconditioned, jacobi] =
            jacobi_krylov_counts(&steady_fixture(NESTED, conductivity, DIRICHLET), label);
        assert!(
            jacobi <= unpreconditioned,
            "{label}: Jacobi took {jacobi} Krylov iterations, unpreconditioned {unpreconditioned}"
        );
    }
}

// -------------------------------------------------------------------------------------------
// Package 2: the partitioned agreement gate on the constraint-eliminated fixtures
// -------------------------------------------------------------------------------------------

/// Serial and parallel `Implicit` solves under `acceleration` agree with monolithic dense
/// Newton within `1e-8`; returns the sweep counts `[serial, parallel]`.
fn assert_partitioned_agreement(
    fixture: &ConnectedFixture,
    acceleration: AccelerationMethod,
    label: &str,
) -> [usize; 2] {
    let context = EvaluationContext::reproducible();
    let zero = vec![0.0; fixture.dimension()];
    let monolithic = solve_newton(&fixture.operator, &context, &zero, &newton()).unwrap();
    assert!(monolithic.converged);
    let mut counts = [0; 2];
    for (index, schedule) in [PartitionedSchedule::Serial, PartitionedSchedule::Parallel]
        .into_iter()
        .enumerate()
    {
        let mut execution =
            PartitionedExecution::new(&fixture.operator, fixture.zero_state()).unwrap();
        let config = implicit(schedule, acceleration, 1.0e-10, 400);
        let report = execution.solve(&context, &config, 0.0).unwrap();
        assert_eq!(report.disposition, PartitionedDisposition::Converged);
        assert_eq!(report.schedule, schedule);
        let evidence = report
            .acceleration
            .as_ref()
            .expect("implicit records its acceleration");
        assert_eq!(evidence.method, acceleration);
        assert_eq!(evidence.trace.len(), report.sweeps.len());
        assert!(evidence.trace.last().unwrap().relaxation_factor.is_none());
        assert!(
            report.sweeps.len() > 2,
            "{label} {schedule:?}: genuinely iterates the fixed point"
        );
        let committed = execution.state().committed_vector().unwrap();
        let disagreement = max_abs_difference(&committed, &monolithic.state);
        eprintln!(
            "{label} {schedule:?} under {acceleration:?}: {} sweeps, partitioned vs monolithic \
             {disagreement:e}",
            report.sweeps.len()
        );
        assert!(
            disagreement < 1.0e-8,
            "{label} {schedule:?}: partitioned vs monolithic disagreement {disagreement}"
        );
        assert!(evidence.trace.last().unwrap().residual_norm <= 1.0e-10);
        counts[index] = report.sweeps.len();
    }
    counts
}

#[test]
fn matching_partitioned_transaction_agrees_with_monolithic_newton_and_refuses_cleanly() {
    let fixture = steady_fixture(MATCHING, [2.5, 3.0], DIRICHLET);
    let plain_counts = assert_partitioned_agreement(&fixture, plain(), "matching");
    // The package 2 gate holds under Aitken as well (package 3's one acceleration axis).
    let aitken_counts = assert_partitioned_agreement(
        &fixture,
        AccelerationMethod::Aitken {
            initial_factor: 1.0,
        },
        "matching",
    );
    eprintln!(
        "matching sweeps [serial, parallel]: plain {plain_counts:?}, Aitken {aitken_counts:?}"
    );

    // A refused solve leaves the committed state bit-identical.
    let context = EvaluationContext::reproducible();
    let mut execution = PartitionedExecution::new(&fixture.operator, fixture.zero_state()).unwrap();
    execution
        .solve(
            &context,
            &implicit(PartitionedSchedule::Serial, plain(), 1.0e-10, 400),
            0.0,
        )
        .unwrap();
    let before = execution.state().committed_vector().unwrap();
    let error = execution
        .solve(
            &context,
            &implicit(PartitionedSchedule::Serial, plain(), 1.0e-300, 2),
            1.0,
        )
        .unwrap_err();
    assert!(
        matches!(&error, KrasisError::EvaluationRefused { code, .. } if code == PARTITIONED_MAX_SWEEPS),
        "{error:?}"
    );
    assert_eq!(execution.state().committed_vector().unwrap(), before);
}

#[test]
fn nested_p1_partitioned_transaction_agrees_with_monolithic_newton_for_both_schedules() {
    let fixture = steady_fixture(NESTED, [2.0, 3.0], DIRICHLET);
    assert!(fixture.is_weighted());
    let counts = assert_partitioned_agreement(&fixture, plain(), "nested_p1");
    eprintln!("nested_p1 sweeps [serial, parallel]: plain {counts:?}");
}

#[test]
fn transient_matching_partitioned_fixed_point_inside_bdf_agrees_with_dense_newton_per_step() {
    let storage = [
        Storage {
            k0: 2.0,
            k1: 1.0e-3,
            rho: 1.5,
            cp: 0.5,
        },
        Storage {
            k0: 3.0,
            k1: 2.0e-3,
            rho: 2.0,
            cp: 0.75,
        },
    ];
    let fixture = transient_fixture(MATCHING, storage, DIRICHLET);
    let context = EvaluationContext::reproducible();
    // A continuous initial temperature (affine in `x`, so the matched traces agree and the
    // algebraic interface rows are consistent), prescribed values at the essential rows.
    let initial = fixture.sampled(|p| affine(p, 300.0, 50.0, 0.0, 0.0));
    let step = 0.01;
    let config = BdfConfig {
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
    };
    let mut dense = CoupledExecution::new(
        fixture.operator.clone(),
        fixture.committed_state(&initial),
        &context,
    )
    .unwrap();
    let layout = BlockNonlinearOperator::block_layout(&fixture.operator).clone();
    let partitioned_config = implicit(PartitionedSchedule::Serial, plain(), 1.0e-12, 200);
    let solver = PartitionedFixedPoint::new(&layout, &partitioned_config);
    let mut partitioned = CoupledExecution::new(
        fixture.operator.clone(),
        fixture.committed_state(&initial),
        &context,
    )
    .unwrap();
    for index in 0..4 {
        let dense_outcome = dense.attempt_step(&context, step, &config).unwrap();
        let partitioned_outcome = partitioned
            .attempt_step_with(&context, step, &config, &solver)
            .unwrap();
        match (dense_outcome, partitioned_outcome) {
            (StepOutcome::Accepted(_), StepOutcome::Accepted(_)) => {}
            other => panic!("step {index}: unexpected outcome pair: {other:?}"),
        }
        let dense_state = dense.state().committed_vector().unwrap();
        let partitioned_state = partitioned.state().committed_vector().unwrap();
        let disagreement = max_abs_difference(&dense_state, &partitioned_state);
        let moved = max_abs_difference(&dense_state, &initial);
        eprintln!("BDF step {index}: dense vs partitioned {disagreement:e}, moved {moved:e}");
        assert!(
            disagreement < 1.0e-9,
            "step {index}: dense vs partitioned disagree by {disagreement}"
        );
        assert!(
            moved > 1.0e-3,
            "step {index}: the trajectory must actually evolve"
        );
    }
    assert!(partitioned.evaluation_refusals().is_empty());
}

// -------------------------------------------------------------------------------------------
// Package 4: block coupling actions and the block Gauss-Seidel factory
// -------------------------------------------------------------------------------------------

#[test]
fn coupled_system_diagonal_is_the_concatenation_of_leaf_diagonals() {
    let fixture = steady_fixture(MATCHING, [2.5, 3.0], DIRICHLET);
    let context = EvaluationContext::reproducible();
    let inner = &fixture.inner;
    let n = NonlinearOperator::dimension(inner);
    let state = steady_point(&fixture);
    let diagonal = NonlinearOperator::jacobian_diagonal(inner, &context, &state)
        .unwrap()
        .expect("every leaf supplies a diagonal");
    let mut unit = vec![0.0; n];
    let mut output = vec![0.0; n];
    for row in 0..n {
        unit[row] = 1.0;
        NonlinearOperator::jacobian_vector_product(inner, &context, &state, &unit, &mut output)
            .unwrap();
        unit[row] = 0.0;
        assert!(
            (diagonal[row] - output[row]).abs() <= 1e-12 * output[row].abs().max(1.0),
            "row {row}: coupled diagonal {} != probe {}",
            diagonal[row],
            output[row]
        );
    }
}

/// Pattern vector of length `n` with no symmetry the coupling could exploit.
fn pattern(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|k| (1.0 + 0.1 * k as f64 + seed) * if k % 2 == 0 { 1.0 } else { -0.7 })
        .collect()
}

#[test]
fn block_coupling_actions_equal_the_masked_jacobian_vector_product_on_both_fixtures() {
    let context = EvaluationContext::reproducible();
    for (label, fixture) in [
        ("matching", steady_fixture(MATCHING, [2.5, 3.0], DIRICHLET)),
        ("nested_p1", steady_fixture(NESTED, [2.0, 3.0], DIRICHLET)),
    ] {
        let operator = &fixture.operator;
        let n = fixture.dimension();
        let point = steady_point(&fixture);
        let layout = BlockNonlinearOperator::block_layout(operator);
        assert_eq!(layout.blocks().len(), 2);
        let factory = BlockGaussSeidelFactory::new(operator, GaussSeidelSweep::Forward);
        let couplings = factory.couplings(&point).unwrap();
        let pairs: Vec<(usize, usize)> = couplings
            .iter()
            .map(|c| (c.row_block(), c.column_block()))
            .collect();
        assert_eq!(pairs, vec![(0, 1), (1, 0)]);
        // Unit-vector columns of the full Jacobian at the point, computed once.
        let mut columns = Vec::with_capacity(n);
        let mut unit = vec![0.0; n];
        for k in 0..n {
            unit[k] = 1.0;
            let mut column = vec![0.0; n];
            NonlinearOperator::jacobian_vector_product(
                operator,
                &context,
                &point,
                &unit,
                &mut column,
            )
            .unwrap();
            unit[k] = 0.0;
            columns.push(column);
        }
        for coupling in &couplings {
            // Krasis declares no transposed structure: nothing is declared.
            assert_eq!(coupling.transpose_of(), None);
            let row = layout.blocks()[coupling.row_block()].range();
            let column = layout.blocks()[coupling.column_block()].range();
            let v = pattern(column.len(), coupling.row_block() as f64);
            let mut action = vec![0.0; row.len()];
            coupling.apply(&context, &v, &mut action).unwrap();

            // (1) The masked full JVP: zero-pad `v` into block j, restrict to block i.
            let mut padded = vec![0.0; n];
            padded[column.clone()].copy_from_slice(&v);
            let mut full = vec![0.0; n];
            NonlinearOperator::jacobian_vector_product(
                operator, &context, &point, &padded, &mut full,
            )
            .unwrap();
            let masked = &full[row.clone()];
            // (2) The same block action assembled from unit-vector columns.
            let mut assembled = vec![0.0; row.len()];
            for (offset, &weight) in v.iter().enumerate() {
                for (target, value) in assembled
                    .iter_mut()
                    .zip(&columns[column.start + offset][row.clone()])
                {
                    *target += weight * value;
                }
            }
            let scale = masked.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
            assert!(
                scale > 1.0,
                "{label} ({}, {}): the coupling action is trivially zero",
                coupling.row_block(),
                coupling.column_block()
            );
            let versus_masked = max_abs_difference(&action, masked) / scale;
            let versus_assembled = max_abs_difference(&action, &assembled) / scale;
            eprintln!(
                "{label} A({}, {})·v: vs masked JVP {versus_masked:.3e}, vs assembled columns \
                 {versus_assembled:.3e} (scale {scale:.3e})",
                coupling.row_block(),
                coupling.column_block()
            );
            assert!(versus_masked <= 1e-12, "{label}: {versus_masked}");
            assert!(versus_assembled <= 1e-12, "{label}: {versus_assembled}");
        }
    }
}

fn krylov_with(
    fixture: &ConnectedFixture,
    context: &EvaluationContext,
    factory: Option<&dyn PreconditionerFactory>,
) -> NewtonKrylovReport {
    let zero = vec![0.0; fixture.dimension()];
    let report = solve_newton_krylov(
        &fixture.operator,
        context,
        &zero,
        &gmres(),
        factory,
        None,
        &krylov_config(),
    )
    .unwrap();
    assert!(report.converged);
    report
}

#[test]
fn gmres_with_block_gauss_seidel_agrees_with_jacobi_and_takes_no_more_krylov_iterations() {
    let fixture = steady_fixture(MATCHING, [2.5, 3.0], DIRICHLET);
    let context = EvaluationContext::reproducible();
    let zero = vec![0.0; fixture.dimension()];
    let dense = solve_newton(&fixture.operator, &context, &zero, &newton()).unwrap();
    assert!(dense.converged);
    let jacobi = krylov_with(&fixture, &context, Some(&JacobiFactory));
    assert!(max_abs_difference(&jacobi.state, &dense.state) < 1.0e-8);

    for (sweep, declared) in [
        (GaussSeidelSweep::Forward, OperatorSymmetry::Nonsymmetric),
        (GaussSeidelSweep::Symmetric, OperatorSymmetry::Unknown),
    ] {
        let factory = BlockGaussSeidelFactory::new(&fixture.operator, sweep);
        // The declaration is Methodus's own over Krasis's undeclared couplings.
        let jacobian = JacobianOperator::new(&fixture.operator, &zero).unwrap();
        let built = PreconditionerFactory::build(&factory, &context, &jacobian, &zero)
            .unwrap()
            .expect("the block Gauss-Seidel factory always builds");
        assert_eq!(built.symmetry(), declared, "{sweep:?}");
        assert_eq!(built.dimension(), fixture.dimension());

        let gauss_seidel = krylov_with(&fixture, &context, Some(&factory));
        let disagreement = max_abs_difference(&gauss_seidel.state, &jacobi.state);
        eprintln!(
            "matching GMRES: Jacobi {} Krylov iterations, block Gauss-Seidel {sweep:?} {}; \
             disagreement {disagreement:e}",
            jacobi.linear_iterations(),
            gauss_seidel.linear_iterations()
        );
        assert!(disagreement < 1.0e-8, "{sweep:?}: {disagreement}");
        assert!(
            gauss_seidel.linear_iterations() <= jacobi.linear_iterations(),
            "{sweep:?}: block Gauss-Seidel took {} Krylov iterations, Jacobi {}",
            gauss_seidel.linear_iterations(),
            jacobi.linear_iterations()
        );
    }
}
