//! SC-W3 package 1: a partitioned fixed-point decomposition over a `methodus::BlockLayout`,
//! usable both as a standalone steady transaction ([`PartitionedExecution`]) and as a
//! [`methodus::NonlinearSolver`] hook ([`PartitionedFixedPoint`]) plugged into
//! [`crate::CoupledExecution::attempt_step_with`] so it runs inside the same BDF transaction as
//! every other solver ([`methodus::BlockNewton`], [`methodus::NewtonKrylovSolver`]).
//!
//! Ownership split: Methodus already owns the per-sweep algorithm
//! ([`methodus::solve_blocks`] with `BlockStrategy::GaussSeidel`/`Jacobi`, one Newton update
//! per block per call with `max_iterations = 1`); Krasis owns running that call repeatedly as a
//! fixed-point iteration, evaluating its own **output-based** convergence between sweeps (the
//! schedule-updated blocks' state -- the exchanged data a `CouplingEdge` or a `ConnectedSystemOperator`
//! eliminated-interface row reads -- rather than the residual norm `solve_blocks` itself checks),
//! applying a fixed relaxation factor to the exchanged data (a vector combination, not a solver
//! algorithm), and refusing typed when the iteration provably diverges or exhausts its bound.
//! Iterate acceleration (Aitken, IQN) is not implemented: Methodus does not yet expose one
//! (`SV7-F3`), so only plain iteration and this fixed relaxation are available; see
//! `STATUS.md` for the recorded cross-repo need.
//!
//! State layout, block ids and checkpoint identity are untouched: this module adds no new
//! operator type and changes no existing identity. A [`PartitionedExecution`] wraps the same
//! [`crate::SimulationState`] trial/commit/rollback every other Krasis transaction uses, and
//! [`PartitionedFixedPoint`] commits nothing itself -- its caller's existing transaction does,
//! exactly as it does for any other `NonlinearSolver`.

use methodus::{
    BlockLayout, BlockNonlinearOperator, BlockStrategy, EvaluationContext, IterationTrace,
    NewtonConfig, NonlinearOperator, NonlinearSolver, NumericError, SolveError, SolveReport,
    solve_blocks,
};
use serde::{Deserialize, Serialize};

use crate::{Checkpoint, KrasisError, SimulationState, TransactionPhase};

/// The refusal code recorded when a sweep's change to the exchanged interface data grows beyond
/// [`PartitionedConfig::divergence_growth`] relative to the previous sweep's: an unconditionally
/// unstable schedule/role choice (the architecture's "wrong Dirichlet-Neumann roles" case), never
/// reported as bare non-convergence. Only `Implicit`'s second and later sweeps can be judged
/// diverging by this test (there is no previous sweep to compare a first, or a `Once`, sweep
/// against); see [`PartitionedIteration::Once`]'s docs for why a single sweep cannot silently
/// diverge under Methodus's backtracked per-sweep algorithm.
pub const PARTITIONED_DIVERGED: &str = "PARTITIONED_DIVERGED";
/// The refusal code recorded when an implicit partitioned iteration exhausts `max_sweeps`
/// without its output-based convergence test passing.
pub const PARTITIONED_MAX_SWEEPS: &str = "PARTITIONED_MAX_SWEEPS";

const REFUSAL_ORIGIN: &str = "krasis partitioned fixed point";

/// Which order a partitioned decomposition visits the leaves of a `BlockLayout` in, each sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionedSchedule {
    /// Gauss-Seidel: each block sees the other blocks' just-updated values within the same sweep.
    Serial,
    /// Jacobi: every block is updated from the sweep's starting values.
    Parallel,
}

impl PartitionedSchedule {
    fn strategy(self) -> BlockStrategy {
        match self {
            Self::Serial => BlockStrategy::GaussSeidel,
            Self::Parallel => BlockStrategy::Jacobi,
        }
    }

    /// Stable lower-case label for reports, mirroring [`methodus::BlockStrategy`]'s naming.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Parallel => "parallel",
        }
    }
}

/// How many fixed-point sweeps a partitioned decomposition runs.
///
/// Only `Implicit` iterated to tolerance is a projection of the same (monolithic) system;
/// `Once` carries the splitting error of an unconverged single exchange, per
/// `sinbad/ARCHITECTURE.md` §9's equivalence-claim paragraph. A `Once` sweep is always accepted:
/// each sweep's per-block Newton step is exact/backtracked by `methodus::solve_blocks` (it
/// refuses `SolveError::LineSearchFailed` before ever returning a worse residual), and Krasis's
/// own divergence test compares a sweep's interface change against the *previous* sweep's, which
/// a lone `Once` sweep has none of. A genuinely unconditionally-unstable one-shot exchange (the
/// architecture's "wrong Dirichlet-Neumann roles" case) therefore only surfaces across repeated
/// application -- an `Implicit` iteration whose interface change grows sweep over sweep
/// (`PARTITIONED_DIVERGED`), or repeated BDF steps of a plugged-in `Once` solver whose state
/// grows without bound until a later step's evaluation hits a typed non-finite refusal. See
/// `STATUS.md` for this recorded limitation of driving the iteration through
/// `methodus::solve_blocks`'s existing (backtracked, block-exact) per-sweep algorithm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PartitionedIteration {
    /// Exactly one schedule sweep, always accepted (see this enum's docs).
    Once,
    /// Repeat sweeps until the sup-norm change of the schedule-updated state between sweeps is
    /// at most `tolerance`, or refuse after `max_sweeps`. `relaxation` in `(0, 1]` is the fixed
    /// damping applied to each sweep's raw update before it becomes the next sweep's input
    /// (`1.0` is plain, undamped iteration).
    Implicit {
        tolerance: f64,
        max_sweeps: usize,
        relaxation: f64,
    },
}

/// Configuration for [`run_partitioned`], [`PartitionedExecution`] and [`PartitionedFixedPoint`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartitionedConfig {
    pub schedule: PartitionedSchedule,
    pub iteration: PartitionedIteration,
    /// Per-sweep block Newton controls (`max_iterations` is overridden to `1`: one sweep is
    /// exactly one `solve_blocks` call over the whole schedule).
    pub newton: NewtonConfig,
    /// A sweep (other than the first) is diverged, and refused `PARTITIONED_DIVERGED`, when its
    /// interface change exceeds `divergence_growth` times the previous sweep's (skipped when the
    /// previous sweep's change is already at the noise floor). Must be finite and `> 1.0`.
    pub divergence_growth: f64,
}

impl PartitionedConfig {
    fn validate(&self) -> Result<(), SolveError> {
        if !self.divergence_growth.is_finite() || self.divergence_growth <= 1.0 {
            return Err(SolveError::InvalidConfiguration {
                reason: "partitioned divergence_growth must be finite and greater than 1.0".into(),
            });
        }
        if let PartitionedIteration::Implicit {
            tolerance,
            max_sweeps,
            relaxation,
        } = &self.iteration
        {
            if !tolerance.is_finite() || *tolerance < 0.0 {
                return Err(SolveError::InvalidConfiguration {
                    reason: "partitioned tolerance must be finite and non-negative".into(),
                });
            }
            if *max_sweeps == 0 {
                return Err(SolveError::InvalidConfiguration {
                    reason: "partitioned max_sweeps must be at least 1".into(),
                });
            }
            if !relaxation.is_finite() || *relaxation <= 0.0 || *relaxation > 1.0 {
                return Err(SolveError::InvalidConfiguration {
                    reason: "partitioned relaxation must be in (0, 1]".into(),
                });
            }
        }
        Ok(())
    }
}

/// One sweep's evidence: the schedule's block-Newton trace over the whole sweep (its first entry
/// is the pre-sweep residual, its last the post-sweep residual -- exactly Methodus's own
/// `solve_blocks(.., max_iterations = 1, ..)` trace), plus the output-based interface measure
/// Krasis evaluates between sweeps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SweepReport {
    pub sweep: usize,
    pub pre_sweep_residual_norm: f64,
    pub post_sweep_residual_norm: f64,
    /// Sup-norm change, over every block the schedule updates, between the sweep's input state
    /// and its raw (pre-relaxation) output: the exchanged interface data's change this sweep.
    pub interface_norm: f64,
    pub block_residual_norms: Vec<(String, f64)>,
    pub newton_trace: Vec<IterationTrace>,
}

/// Outcome of a partitioned decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionedDisposition {
    /// `Implicit` reached its output-based tolerance.
    Converged,
    /// `Once` completed its single sweep without a detected divergence.
    OnceApplied,
}

/// Full evidence for an accepted partitioned decomposition (`run_partitioned` refuses typed
/// instead of returning a diverged or exhausted report -- see [`PARTITIONED_DIVERGED`],
/// [`PARTITIONED_MAX_SWEEPS`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartitionedReport {
    pub schedule: PartitionedSchedule,
    pub relaxation: f64,
    pub disposition: PartitionedDisposition,
    pub sweeps: Vec<SweepReport>,
}

/// An accepted partitioned solve: the state after relaxation, and its report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartitionedSolve {
    pub state: Vec<f64>,
    pub report: PartitionedReport,
}

fn typed_refusal(code: &str, message: String) -> SolveError {
    SolveError::Numeric(NumericError::Evaluation {
        code: code.to_owned(),
        origin: REFUSAL_ORIGIN.to_owned(),
        message,
    })
}

fn sup_norm_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Runs one partitioned decomposition to [`PartitionedConfig::iteration`]'s bound over any
/// operator carrying a [`BlockLayout`]: one `solve_blocks` call per sweep (the algorithm
/// Methodus owns), an output-based (state-change) convergence test and a residual-growth
/// divergence test Krasis owns (see module docs), and fixed relaxation applied to each sweep's
/// raw update before it becomes the next sweep's input.
///
/// Refuses [`PARTITIONED_DIVERGED`] or [`PARTITIONED_MAX_SWEEPS`] (both
/// [`NumericError::Evaluation`], never a silently accepted iterate) instead of returning a
/// diverged or exhausted report.
pub fn run_partitioned<Op>(
    operator: &Op,
    context: &EvaluationContext,
    initial_state: &[f64],
    config: &PartitionedConfig,
) -> Result<PartitionedSolve, SolveError>
where
    Op: BlockNonlinearOperator + ?Sized,
{
    config.validate()?;
    if initial_state.len() != operator.dimension() {
        return Err(SolveError::Numeric(NumericError::DimensionMismatch {
            operation: "partitioned initial state".into(),
            expected: operator.dimension(),
            actual: initial_state.len(),
        }));
    }
    if let Some(index) = initial_state.iter().position(|value| !value.is_finite()) {
        return Err(SolveError::Numeric(NumericError::NonFinite {
            operation: "partitioned initial state".into(),
            index,
        }));
    }

    let max_sweeps = match &config.iteration {
        PartitionedIteration::Once => 1,
        PartitionedIteration::Implicit { max_sweeps, .. } => *max_sweeps,
    };
    let relaxation = match &config.iteration {
        PartitionedIteration::Once => 1.0,
        PartitionedIteration::Implicit { relaxation, .. } => *relaxation,
    };
    let single_sweep = NewtonConfig {
        max_iterations: 1,
        ..config.newton.clone()
    };

    let mut state = initial_state.to_vec();
    let mut sweeps = Vec::with_capacity(max_sweeps);
    let mut previous_interface_norm: Option<f64> = None;
    const NOISE_FLOOR: f64 = 1.0e-12;
    for sweep in 1..=max_sweeps {
        let report = solve_blocks(
            operator,
            context,
            &state,
            config.schedule.strategy(),
            &single_sweep,
        )?;
        let pre = report
            .trace
            .first()
            .expect("solve_blocks records at least one trace entry");
        let post = report
            .trace
            .last()
            .expect("solve_blocks records at least one trace entry");
        let pre_residual = pre.scaled_residual_norm;
        let post_residual = post.scaled_residual_norm;

        let raw_next = report.state;
        let interface_norm = sup_norm_difference(&raw_next, &state);

        // `solve_blocks`'s own backtracking already guarantees the *residual* cannot grow
        // within one sweep (it refuses `LineSearchFailed` first), so the output-based signal
        // Krasis can actually observe diverging is the **exchanged data** growing sweep over
        // sweep: a sweep with no predecessor (`Once`, or `Implicit`'s first sweep) has nothing
        // to compare against and cannot be judged diverging by this test.
        if let Some(previous) = previous_interface_norm {
            if previous > NOISE_FLOOR
                && (!interface_norm.is_finite()
                    || interface_norm > config.divergence_growth * previous)
            {
                return Err(typed_refusal(
                    PARTITIONED_DIVERGED,
                    format!(
                        "sweep {sweep}: the exchanged interface data's change grew from \
                         {previous:e} to {interface_norm:e}, exceeding the declared growth \
                         bound {} ({} schedule)",
                        config.divergence_growth,
                        config.schedule.label(),
                    ),
                ));
            }
        }
        previous_interface_norm = Some(interface_norm);

        let mut relaxed = state.clone();
        for (value, (next, previous)) in relaxed.iter_mut().zip(raw_next.iter().zip(state.iter())) {
            *value = previous + relaxation * (next - previous);
        }
        sweeps.push(SweepReport {
            sweep,
            pre_sweep_residual_norm: pre_residual,
            post_sweep_residual_norm: post_residual,
            interface_norm,
            block_residual_norms: post.block_residual_norms.clone(),
            newton_trace: report.trace,
        });
        state = relaxed;

        match &config.iteration {
            PartitionedIteration::Once => {
                return Ok(PartitionedSolve {
                    state,
                    report: PartitionedReport {
                        schedule: config.schedule,
                        relaxation,
                        disposition: PartitionedDisposition::OnceApplied,
                        sweeps,
                    },
                });
            }
            PartitionedIteration::Implicit { tolerance, .. } => {
                if interface_norm <= *tolerance {
                    return Ok(PartitionedSolve {
                        state,
                        report: PartitionedReport {
                            schedule: config.schedule,
                            relaxation,
                            disposition: PartitionedDisposition::Converged,
                            sweeps,
                        },
                    });
                }
            }
        }
    }

    Err(typed_refusal(
        PARTITIONED_MAX_SWEEPS,
        format!(
            "{} schedule did not reach its declared tolerance within {max_sweeps} sweeps",
            config.schedule.label()
        ),
    ))
}

/// Attaches an externally supplied [`BlockLayout`] to a layout-less nonlinear operator, exactly
/// as [`methodus::BlockNewton`]'s internal view does: the layout is captured at construction
/// time, not derived from whatever operator a caller (e.g. `bdf_step_with`'s implicit-step
/// residual) later passes to [`NonlinearSolver::solve`].
struct LayoutView<'a> {
    inner: &'a dyn NonlinearOperator,
    layout: &'a BlockLayout,
}

impl NonlinearOperator for LayoutView<'_> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn residual(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.inner.residual(context, state, output)
    }

    fn jacobian_vector_product(
        &self,
        context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.inner
            .jacobian_vector_product(context, state, direction, output)
    }
}

impl BlockNonlinearOperator for LayoutView<'_> {
    fn block_layout(&self) -> &BlockLayout {
        self.layout
    }
}

/// [`run_partitioned`] as a [`NonlinearSolver`], so a partitioned fixed-point iteration can run
/// inside a BDF step through [`crate::CoupledExecution::attempt_step_with`], exactly like
/// [`methodus::BlockNewton`] or [`methodus::NewtonKrylovSolver`]: the transaction's existing
/// trial/commit/rollback commits only an accepted step, and a refusal
/// ([`PARTITIONED_DIVERGED`], [`PARTITIONED_MAX_SWEEPS`]) surfaces as
/// [`crate::KrasisError::EvaluationRefused`] with its code and origin intact after rollback,
/// logged in [`crate::CoupledExecution::evaluation_refusals`].
///
/// The per-sweep [`PartitionedReport`] this module builds is not recoverable from the plugged-in
/// call: `methodus::NonlinearSolver::solve` returns only `methodus::SolveReport`, the same
/// generic shape `BlockNewton` and `NewtonKrylovSolver` report through today. `solve` returns
/// every sweep's own Methodus block-Newton trace concatenated in `SolveReport::trace`, so the
/// per-sweep residual evidence survives; the interface norms, schedule and relaxation do not
/// (see `STATUS.md`).
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedFixedPoint<'a> {
    layout: &'a BlockLayout,
    config: &'a PartitionedConfig,
}

impl<'a> PartitionedFixedPoint<'a> {
    #[must_use]
    pub const fn new(layout: &'a BlockLayout, config: &'a PartitionedConfig) -> Self {
        Self { layout, config }
    }
}

impl NonlinearSolver for PartitionedFixedPoint<'_> {
    fn solve(
        &self,
        operator: &dyn NonlinearOperator,
        context: &EvaluationContext,
        initial_state: &[f64],
    ) -> Result<SolveReport, SolveError> {
        let view = LayoutView {
            inner: operator,
            layout: self.layout,
        };
        let solved = run_partitioned(&view, context, initial_state, self.config)?;
        let trace = solved
            .report
            .sweeps
            .into_iter()
            .flat_map(|sweep| sweep.newton_trace)
            .collect();
        Ok(SolveReport {
            state: solved.state,
            converged: true,
            trace,
        })
    }
}

/// Transactional steady partitioned solve over one committed [`SimulationState`], mirroring
/// [`crate::BlockLinearExecution`]'s shape for the nonlinear/partitioned case: a converged (or
/// accepted `Once`) sweep sequence commits at `commit_time`; a refused one rolls back to the
/// prior committed state before returning [`KrasisError::EvaluationRefused`] (typed) or
/// [`KrasisError::Solve`] (any other `solve_blocks` failure).
#[derive(Debug)]
pub struct PartitionedExecution<'op, Op: BlockNonlinearOperator + NonlinearOperator> {
    operator: &'op Op,
    state: SimulationState,
}

impl<'op, Op: BlockNonlinearOperator + NonlinearOperator> PartitionedExecution<'op, Op> {
    /// Binds `operator` to `state`, refusing a dimension mismatch or a state that is not already
    /// complete and in committed phase.
    pub fn new(operator: &'op Op, state: SimulationState) -> Result<Self, KrasisError> {
        if state.phase() != TransactionPhase::Committed {
            return Err(KrasisError::InvalidCoupling(
                "partitioned execution must start from committed state".into(),
            ));
        }
        let width = state.committed_vector()?.len();
        if NonlinearOperator::dimension(operator) != width {
            return Err(KrasisError::InvalidCoupling(format!(
                "partitioned operator has dimension {}, Krasis state width is {width}",
                NonlinearOperator::dimension(operator)
            )));
        }
        if operator.block_layout().dimension() != width {
            return Err(KrasisError::InvalidCoupling(format!(
                "partitioned operator's block layout has dimension {}, Krasis state width is {width}",
                operator.block_layout().dimension()
            )));
        }
        Ok(Self { operator, state })
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    /// Attempts one partitioned decomposition inside a Krasis trial transaction, starting from
    /// the current committed state. See [`run_partitioned`] for the sweep/convergence/divergence
    /// semantics; a refusal rolls back and returns before any state changes.
    pub fn solve(
        &mut self,
        context: &EvaluationContext,
        config: &PartitionedConfig,
        commit_time: f64,
    ) -> Result<PartitionedReport, KrasisError> {
        let initial = self.state.committed_vector()?;
        self.state.begin_trial()?;
        let solved = match run_partitioned(self.operator, context, &initial, config) {
            Ok(solved) => solved,
            Err(error) => {
                self.state.rollback()?;
                return Err(crate::coupled::krasis_solve_error(error));
            }
        };
        if let Err(error) = self.state.set_trial_vector(&solved.state) {
            self.state.rollback()?;
            return Err(error);
        }
        if let Err(error) = self.state.commit(commit_time) {
            self.state.rollback()?;
            return Err(error);
        }
        Ok(solved.report)
    }

    pub fn checkpoint(&self) -> Result<Checkpoint, KrasisError> {
        self.state.checkpoint()
    }

    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<(), KrasisError> {
        let mut candidate = self.state.clone();
        candidate.restore(checkpoint)?;
        self.state = candidate;
        Ok(())
    }
}
