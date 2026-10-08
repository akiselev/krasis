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
//! sup-norm change of the schedule-updated state -- the exchanged data a `CouplingEdge` or a
//! `ConnectedSystemOperator` eliminated-interface row reads -- rather than the residual norm
//! `solve_blocks` itself checks), applying a fixed relaxation factor to that state (a vector
//! combination, not a solver algorithm), and refusing typed when the iteration provably
//! diverges or exhausts its bound.
//!
//! Acceleration: Krasis's `relaxation` is this transaction's only acceleration axis.
//! `NewtonConfig::acceleration` (Methodus's per-call Aitken/fixed relaxation of the partitioned
//! Newton correction) is refused by [`PartitionedConfig`] rather than forwarded: forwarded into
//! every per-sweep `solve_blocks` call it would skip Methodus's backtracking (voiding the
//! line-search divergence observation below) and, because each sweep is a fresh call with
//! `max_iterations = 1`, Aitken's history would reset every sweep and degrade to a fixed factor
//! stacked multiplicatively with `relaxation`. The consumable Methodus shape for the sweep loop
//! is [`methodus::accelerate_fixed_point`] over a [`methodus::FixedPointOperator`] wrapping one
//! sweep; consuming it (and dropping `relaxation`, so one acceleration axis remains) is the next
//! package, recorded in `STATUS.md`.
//!
//! State layout, block ids and checkpoint identity are untouched: this module adds no new
//! operator type and changes no existing identity. A [`PartitionedExecution`] wraps the same
//! [`crate::SimulationState`] trial/commit/rollback every other Krasis transaction uses and
//! binds its checkpoints to the operator's content identity exactly as
//! [`crate::CoupledExecution`] does; [`PartitionedFixedPoint`] commits nothing itself -- its
//! caller's existing transaction does, exactly as it does for any other `NonlinearSolver`.
//!
//! Every refusal this module records is a [`NumericError::Evaluation`] whose `origin` is
//! [`PARTITIONED_REFUSAL_ORIGIN`] and whose `code` is [`PARTITIONED_DIVERGED`] or
//! [`PARTITIONED_MAX_SWEEPS`]; consumers key on that pair.

use methodus::{
    BlockLayout, BlockNonlinearOperator, BlockStrategy, EvaluationContext, IterationTrace,
    NewtonConfig, NonlinearOperator, NonlinearSolver, NumericError, SolveError, SolveReport,
    solve_blocks,
};
use serde::{Deserialize, Serialize};

use crate::{
    Checkpoint, KrasisError, SimulationState, StateLayout, TransactionPhase, TransactionalOperator,
};

/// The refusal code recorded when the partitioned schedule provably diverges, by either of the
/// two observations Krasis makes on the fixed-point iteration it drives:
///
/// - a sweep whose whole schedule correction cannot reduce the residual at any admissible
///   damping: Methodus's per-sweep `solve_blocks` refuses `SolveError::LineSearchFailed`, and
///   inside this driver that *is* the diverging-schedule case. A non-contractive splitting (the
///   architecture's "wrong Dirichlet-Neumann roles" case) has an error-propagation mode whose
///   factor exceeds one, and that mode grows under every damping of the sweep's correction, so
///   the backtracked per-sweep algorithm necessarily fails before an unbounded iterate can form.
///   This is the only observation that can judge a first, or a `Once`, sweep diverging.
/// - a sweep (other than the first) whose change to the schedule-updated state grows beyond
///   [`PartitionedConfig::divergence_growth`] times the previous sweep's.
///
/// Neither observation is made when the previous sweep's change was already at the rounding
/// floor (see [`run_partitioned`]): growth or a failed line search from there is a stall, refused
/// [`PARTITIONED_MAX_SWEEPS`], never divergence. Never reported as bare non-convergence.
pub const PARTITIONED_DIVERGED: &str = "PARTITIONED_DIVERGED";
/// The refusal code recorded when an implicit partitioned iteration does not reach its declared
/// tolerance: it exhausts `max_sweeps`, or it stalls earlier at the rounding floor (the previous
/// sweep changed the state by at most the noise floor and this sweep's correction cannot reduce
/// the residual at any admissible damping, so no further sweep could reach the tolerance).
pub const PARTITIONED_MAX_SWEEPS: &str = "PARTITIONED_MAX_SWEEPS";
/// The `origin` of every refusal this module records ([`NumericError::Evaluation`], surfacing
/// as [`crate::KrasisError::EvaluationRefused`]): consumers key on this string together with
/// the code to attribute a refusal to the partitioned fixed-point transaction rather than to a
/// leaf operator, a constitutive provider or a Methodus solver. Stable; part of the SC-W3
/// contract.
pub const PARTITIONED_REFUSAL_ORIGIN: &str = "krasis partitioned fixed point";

/// The rounding floor of a sweep's state change, in ulps of the larger of `1` and the sup norm
/// of the state the change is measured on: a change at or below this is rounding noise, and
/// growth measured from it is never judged divergence.
const NOISE_FLOOR_ULPS: f64 = 32.0;

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
/// `sinbad/ARCHITECTURE.md` §9's equivalence-claim paragraph. A `Once` sweep is accepted
/// whenever Methodus's backtracked per-sweep algorithm accepts it (at any admissible damping),
/// and it is then accepted **by declaration**, not by any convergence test: its output is one
/// (possibly damped) schedule exchange, demonstrably not the monolithic solution (the SC-W3
/// tests assert the disagreement). It is refused [`PARTITIONED_DIVERGED`] only when its whole
/// schedule correction cannot reduce the residual at any admissible damping (a non-contractive
/// splitting); the sweep-over-sweep growth test cannot judge a lone sweep, which has no
/// predecessor to compare against.
///
/// Inside a BDF transaction ([`PartitionedFixedPoint`]) a `Once` solver is therefore a declared
/// regime: every accepted step carries that sweep's splitting error, so a run using it must be
/// declared as such at the run level (Sinbad's run declaration), and its accepted steps are
/// excluded from agreement claims against the monolithic solve.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PartitionedIteration {
    /// Exactly one schedule sweep, accepted by declaration when the per-sweep algorithm accepts
    /// it (see this enum's docs).
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
    /// exactly one `solve_blocks` call over the whole schedule). `acceleration` must be `None`:
    /// a `Some` is refused as `SolveError::InvalidConfiguration` (see the module docs for why
    /// Krasis's `relaxation` is this transaction's only acceleration axis).
    pub newton: NewtonConfig,
    /// A sweep (other than the first) is diverged, and refused `PARTITIONED_DIVERGED`, when its
    /// state change exceeds `divergence_growth` times the previous sweep's (skipped when the
    /// previous sweep's change is already at the rounding floor). Must be finite and `> 1.0`.
    pub divergence_growth: f64,
}

impl PartitionedConfig {
    fn validate(&self) -> Result<(), SolveError> {
        if !self.divergence_growth.is_finite() || self.divergence_growth <= 1.0 {
            return Err(SolveError::InvalidConfiguration {
                reason: "partitioned divergence_growth must be finite and greater than 1.0".into(),
            });
        }
        if self.newton.acceleration.is_some() {
            return Err(SolveError::InvalidConfiguration {
                reason: "partitioned acceleration is Krasis's relaxation axis until the \
                         sweep-level consumption of methodus::accelerate_fixed_point lands: set \
                         PartitionedConfig.newton.acceleration to None and declare \
                         PartitionedIteration::Implicit::relaxation instead"
                    .into(),
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
/// `solve_blocks(.., max_iterations = 1, ..)` trace), plus the output-based state-change measure
/// Krasis evaluates between sweeps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SweepReport {
    pub sweep: usize,
    pub pre_sweep_residual_norm: f64,
    pub post_sweep_residual_norm: f64,
    /// Sup-norm change of the whole state between the sweep's input and its raw
    /// (pre-relaxation) output: the measure Krasis's convergence and divergence tests evaluate.
    /// Every block the schedule updates contributes, so this is the change of the exchanged
    /// (schedule-updated) data as a whole -- **not** an interface trace or flux measure; no finer
    /// per-DOF trace decomposition exists at this composition level.
    pub state_change_norm: f64,
    pub block_residual_norms: Vec<(String, f64)>,
    pub newton_trace: Vec<IterationTrace>,
}

/// Outcome of a partitioned decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionedDisposition {
    /// `Implicit` reached its output-based tolerance.
    Converged,
    /// `Once` completed its single sweep, accepted by declaration (see
    /// [`PartitionedIteration`]).
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
        origin: PARTITIONED_REFUSAL_ORIGIN.to_owned(),
        message,
    })
}

fn sup_norm_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// [`NOISE_FLOOR_ULPS`] ulps of `max(1, ‖state‖∞)`.
fn noise_floor(state: &[f64]) -> f64 {
    let scale = state
        .iter()
        .fold(1.0_f64, |scale, value| scale.max(value.abs()));
    scale * NOISE_FLOOR_ULPS * f64::EPSILON
}

/// The residual's l2 norm at `state` (the norm Methodus's per-sweep line search measures when
/// every block's residual scale is one), evaluated only to describe a refused sweep.
fn residual_norm<Op>(
    operator: &Op,
    context: &EvaluationContext,
    state: &[f64],
) -> Result<f64, SolveError>
where
    Op: NonlinearOperator + ?Sized,
{
    let mut residual = vec![0.0; operator.dimension()];
    operator.residual(context, state, &mut residual)?;
    Ok(residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt())
}

/// Runs one partitioned decomposition to [`PartitionedConfig::iteration`]'s bound over any
/// operator carrying a [`BlockLayout`]: one `solve_blocks` call per sweep (the algorithm
/// Methodus owns), an output-based (state-change) convergence test and the two divergence
/// observations Krasis owns (see [`PARTITIONED_DIVERGED`]), and fixed relaxation applied to
/// each sweep's raw update before it becomes the next sweep's input.
///
/// Refuses [`PARTITIONED_DIVERGED`] or [`PARTITIONED_MAX_SWEEPS`] (both
/// [`NumericError::Evaluation`] with origin [`PARTITIONED_REFUSAL_ORIGIN`], never a silently
/// accepted iterate) instead of returning a diverged, stalled or exhausted report. The rounding
/// floor both observations respect is 32 ulps (`NOISE_FLOOR_ULPS`) of `max(1, ‖state‖∞)`: a
/// previous sweep that changed the state by at most that is at the floor, from which neither
/// growth nor a failed line search is attributed to the schedule.
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

    let (max_sweeps, relaxation, tolerance) = match &config.iteration {
        PartitionedIteration::Once => (1, 1.0, None),
        PartitionedIteration::Implicit {
            tolerance,
            max_sweeps,
            relaxation,
        } => (*max_sweeps, *relaxation, Some(*tolerance)),
    };
    let single_sweep = NewtonConfig {
        max_iterations: 1,
        ..config.newton.clone()
    };
    let schedule = config.schedule.label();

    let mut state = initial_state.to_vec();
    let mut sweeps = Vec::with_capacity(max_sweeps);
    let mut previous_change: Option<f64> = None;
    for sweep in 1..=max_sweeps {
        let floor = noise_floor(&state);
        let previous_above_floor = previous_change.is_some_and(|previous| previous > floor);
        let report = match solve_blocks(
            operator,
            context,
            &state,
            config.schedule.strategy(),
            &single_sweep,
        ) {
            Ok(report) => report,
            // The whole schedule correction cannot reduce the residual at any admissible
            // damping. From a previous sweep at the rounding floor that is a stall; from
            // anywhere else it is the non-contractive schedule (see `PARTITIONED_DIVERGED`).
            Err(SolveError::LineSearchFailed) => {
                let residual = residual_norm(operator, context, &state)?;
                return Err(match previous_change {
                    Some(previous) if !previous_above_floor => typed_refusal(
                        PARTITIONED_MAX_SWEEPS,
                        format!(
                            "{schedule} schedule stalled at the rounding floor at sweep {sweep} \
                             of {max_sweeps} without reaching its declared tolerance: the \
                             previous sweep changed the state by {previous:e} (floor {floor:e}) \
                             and no admissible damping of this sweep's correction reduces the \
                             residual {residual:e}"
                        ),
                    ),
                    previous => typed_refusal(
                        PARTITIONED_DIVERGED,
                        format!(
                            "sweep {sweep}: no admissible damping of the {schedule} schedule's \
                             whole correction reduces the residual ({residual:e} at the sweep's \
                             input state; previous sweep's state change {}): a non-contractive \
                             splitting, refused before an unbounded iterate can form",
                            previous.map_or_else(
                                || "none, first sweep".to_owned(),
                                |previous| format!("{previous:e}")
                            ),
                        ),
                    ),
                });
            }
            Err(other) => return Err(other),
        };
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
        let state_change_norm = sup_norm_difference(&raw_next, &state);

        if let Some(previous) = previous_change {
            if previous_above_floor
                && (!state_change_norm.is_finite()
                    || state_change_norm > config.divergence_growth * previous)
            {
                return Err(typed_refusal(
                    PARTITIONED_DIVERGED,
                    format!(
                        "sweep {sweep}: the schedule-updated state's change grew from \
                         {previous:e} to {state_change_norm:e}, exceeding the declared growth \
                         bound {} ({schedule} schedule)",
                        config.divergence_growth,
                    ),
                ));
            }
        }
        previous_change = Some(state_change_norm);

        let mut relaxed = state.clone();
        for (value, (next, previous)) in relaxed.iter_mut().zip(raw_next.iter().zip(state.iter())) {
            *value = previous + relaxation * (next - previous);
        }
        sweeps.push(SweepReport {
            sweep,
            pre_sweep_residual_norm: pre_residual,
            post_sweep_residual_norm: post_residual,
            state_change_norm,
            block_residual_norms: post.block_residual_norms.clone(),
            newton_trace: report.trace,
        });
        state = relaxed;

        let disposition = match tolerance {
            None => PartitionedDisposition::OnceApplied,
            Some(tolerance) if state_change_norm <= tolerance => PartitionedDisposition::Converged,
            Some(_) => continue,
        };
        return Ok(PartitionedSolve {
            state,
            report: PartitionedReport {
                schedule: config.schedule,
                relaxation,
                disposition,
                sweeps,
            },
        });
    }

    Err(typed_refusal(
        PARTITIONED_MAX_SWEEPS,
        format!(
            "{schedule} schedule did not reach its declared tolerance within {max_sweeps} sweeps"
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
/// [`crate::KrasisError::EvaluationRefused`] with its code and origin
/// ([`PARTITIONED_REFUSAL_ORIGIN`]) intact after rollback, logged in
/// [`crate::CoupledExecution::evaluation_refusals`].
///
/// `SolveReport::converged` is `true` for every accepted solve: a converged `Implicit`
/// iteration genuinely met its declared output-based tolerance, while an accepted `Once` sweep
/// is accepted by declaration (see [`PartitionedIteration`]) -- the step it feeds is a step of
/// the declared `once` regime, not a projection of the monolithic system, and the run must say
/// so at the run level.
///
/// The per-sweep [`PartitionedReport`] this module builds is not recoverable from the plugged-in
/// call: `methodus::NonlinearSolver::solve` returns only `methodus::SolveReport`, the same
/// generic shape `BlockNewton` and `NewtonKrylovSolver` report through today. `solve` returns
/// every sweep's own Methodus block-Newton trace concatenated in `SolveReport::trace`, so the
/// per-sweep residual evidence survives; the state-change norms, schedule and relaxation do not
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

/// Serializable restart data binding a [`Checkpoint`] to the exact operator a
/// [`PartitionedExecution`] produced it against, mirroring [`crate::BlockLinearCheckpoint`] and
/// [`crate::CoupledCheckpoint`]: the operator's [`TransactionalOperator::identity`] (its content
/// identity -- leaves, edges, elimination), so a checkpoint restores only into an execution over
/// the same system, never into a different operator over the same layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartitionedCheckpoint {
    pub operator_identity: String,
    pub state: Checkpoint,
}

/// Transactional steady partitioned solve over one committed [`SimulationState`], mirroring
/// [`crate::BlockLinearExecution`]'s shape for the nonlinear/partitioned case: a converged (or
/// accepted `Once`) sweep sequence commits at `commit_time`; a refused one rolls back to the
/// prior committed state before returning [`KrasisError::EvaluationRefused`] (typed) or
/// [`KrasisError::Solve`] (any other `solve_blocks` failure, including a refused
/// configuration). Checkpoints are bound to the operator's content identity.
#[derive(Debug)]
pub struct PartitionedExecution<'op, Op: BlockNonlinearOperator + TransactionalOperator> {
    operator: &'op Op,
    operator_identity: String,
    state: SimulationState,
}

impl<'op, Op: BlockNonlinearOperator + TransactionalOperator> PartitionedExecution<'op, Op> {
    /// Binds `operator` to `state`, refusing a state that is not already complete and in
    /// committed phase, a state laid out by anything but the operator's own [`StateLayout`]
    /// (identity compared, as [`crate::CoupledExecution::new`] does), a dimension mismatch, and
    /// a solver [`BlockLayout`] whose blocks do not tile the state layout's blocks exactly (every
    /// solver block must be a run of whole state blocks: the per-block length check
    /// [`crate::BlockLinearExecution::new`] makes, generalized to a per-leaf solver partition
    /// over a possibly multi-block leaf layout).
    pub fn new(operator: &'op Op, state: SimulationState) -> Result<Self, KrasisError> {
        if state.phase() != TransactionPhase::Committed {
            return Err(KrasisError::InvalidCoupling(
                "partitioned execution must start from committed state".into(),
            ));
        }
        if state.layout().identity() != operator.state_layout_identity() {
            return Err(KrasisError::InvalidCoupling(
                "partitioned state layout does not match the operator's state layout".into(),
            ));
        }
        let width = state.committed_vector()?.len();
        if NonlinearOperator::dimension(operator) != width {
            return Err(KrasisError::InvalidCoupling(format!(
                "partitioned operator has dimension {}, Krasis state width is {width}",
                NonlinearOperator::dimension(operator)
            )));
        }
        let solver_layout = operator.block_layout();
        if solver_layout.dimension() != width {
            return Err(KrasisError::InvalidCoupling(format!(
                "partitioned operator's block layout has dimension {}, Krasis state width is {width}",
                solver_layout.dimension()
            )));
        }
        check_blocks_tile_layout(solver_layout, state.layout())?;
        Ok(Self {
            operator,
            operator_identity: operator.identity().to_owned(),
            state,
        })
    }

    /// The operator content identity every checkpoint of this execution is bound to.
    pub fn operator_identity(&self) -> &str {
        &self.operator_identity
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

    pub fn checkpoint(&self) -> Result<PartitionedCheckpoint, KrasisError> {
        Ok(PartitionedCheckpoint {
            operator_identity: self.operator_identity.clone(),
            state: self.state.checkpoint()?,
        })
    }

    /// Atomically restores state after validating it was checkpointed against this exact
    /// operator identity; a mismatch is refused before any state changes.
    pub fn restore(&mut self, checkpoint: &PartitionedCheckpoint) -> Result<(), KrasisError> {
        if checkpoint.operator_identity != self.operator_identity {
            return Err(KrasisError::InvalidCoupling(format!(
                "checkpoint operator identity `{}` does not match `{}`",
                checkpoint.operator_identity, self.operator_identity
            )));
        }
        let mut candidate = self.state.clone();
        candidate.restore(&checkpoint.state)?;
        self.state = candidate;
        Ok(())
    }
}

/// Every solver block must be exactly a run of whole, consecutive state blocks.
fn check_blocks_tile_layout(solver: &BlockLayout, state: &StateLayout) -> Result<(), KrasisError> {
    let mut state_blocks = state.blocks().iter();
    for block in solver.blocks() {
        let range = block.range();
        let mut covered = range.start;
        while covered < range.end {
            let Some(state_block) = state_blocks.next() else {
                return Err(KrasisError::InvalidCoupling(format!(
                    "partitioned solver block `{}` ({}..{}) extends past the last Krasis state block",
                    block.name(),
                    range.start,
                    range.end
                )));
            };
            if state_block.range().start != covered || state_block.range().end > range.end {
                return Err(KrasisError::InvalidCoupling(format!(
                    "partitioned solver block `{}` ({}..{}) does not tile Krasis state block `{}` \
                     ({}..{}): solver blocks must be runs of whole state blocks",
                    block.name(),
                    range.start,
                    range.end,
                    state_block.id(),
                    state_block.range().start,
                    state_block.range().end
                )));
            }
            covered = state_block.range().end;
        }
    }
    if let Some(state_block) = state_blocks.next() {
        return Err(KrasisError::InvalidCoupling(format!(
            "Krasis state block `{}` lies outside every partitioned solver block",
            state_block.id()
        )));
    }
    Ok(())
}
