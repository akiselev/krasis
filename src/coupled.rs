use std::collections::BTreeSet;

use finitum::RealizationPlan;
use methodus::{
    BdfConfig, BdfState, BlockLayout as SolverBlockLayout, BlockNonlinearOperator, BlockSpec,
    DaeOperator, EvaluationContext, NewtonConfig, NonlinearOperator, NonlinearSolver, NumericError,
    SolveError, StepOutcome, bdf_step, bdf_step_with, solve_newton,
};
use serde::{Deserialize, Serialize};

use crate::{
    BlockId, Checkpoint, KrasisError, SimulationState, StateBinding, StateLayout, TransactionPhase,
};

/// Per-row classification for index-1 consistent initialization.
///
/// `Differential` rows contribute their state-rate to the unknown solved by
/// [`CoupledOperator::solve_consistent_state_rate`]. `Algebraic` rows contribute no rate
/// unknown; that row's residual must already vanish at the supplied state (see
/// [`CoupledOperator::solve_consistent_state_rate`] for the exact refusal condition).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowKind {
    Differential,
    Algebraic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ConsistentInitialization {
    pub(crate) mask: Vec<RowKind>,
    pub(crate) newton: NewtonConfig,
}

/// A Krasis-composed DAE operator that [`CoupledExecution`] can enclose in a transaction:
/// it carries a content identity (folded into every checkpoint) and the identity of the
/// [`StateLayout`] its state vector is laid out by.
pub trait TransactionalOperator: DaeOperator + Clone + std::fmt::Debug {
    fn identity(&self) -> &str;
    fn state_layout_identity(&self) -> &str;
}

/// Krasis-owned composition of a realized Finitum action into Methodus contracts.
///
/// Its [`NonlinearOperator`] implementation is the steady view at `t = 0` and `ydot = 0`.
/// Time-dependent boundary, source, or material behavior must use its [`DaeOperator`] view.
#[derive(Clone, Debug)]
pub struct CoupledOperator {
    realization: RealizationPlan,
    state_layout_identity: String,
    block_layout: SolverBlockLayout,
    state_binding: Option<StateBinding>,
    consistent_initialization: Option<ConsistentInitialization>,
    identity: String,
}

impl CoupledOperator {
    pub fn new(
        realization: RealizationPlan,
        state_layout: &StateLayout,
    ) -> Result<Self, KrasisError> {
        Self::new_with_bindings(realization, state_layout, None)
    }

    /// Additive constructor that records a [`StateBinding`] in the operator identity.
    ///
    /// `state_binding`, when present, must name exactly the blocks in `state_layout`.
    pub fn new_with_bindings(
        realization: RealizationPlan,
        state_layout: &StateLayout,
        state_binding: Option<StateBinding>,
    ) -> Result<Self, KrasisError> {
        if realization.dimension() != state_layout.width() {
            return Err(KrasisError::InvalidCoupling(format!(
                "Finitum dimension {} differs from Krasis state width {}",
                realization.dimension(),
                state_layout.width()
            )));
        }
        if let Some(binding) = &state_binding {
            let layout_blocks: BTreeSet<BlockId> = state_layout
                .blocks()
                .iter()
                .map(|block| block.id().clone())
                .collect();
            let binding_blocks: BTreeSet<BlockId> = binding.blocks().cloned().collect();
            if layout_blocks != binding_blocks {
                return Err(KrasisError::StateBindingLayoutMismatch);
            }
        }
        let block_layout = SolverBlockLayout::new(
            state_layout
                .blocks()
                .iter()
                .map(|block| BlockSpec {
                    name: block.id().to_string(),
                    length: block.range().len(),
                    residual_scale: 1.0,
                })
                .collect(),
        )
        .map_err(|error| KrasisError::InvalidCoupling(error.to_string()))?;
        let state_layout_identity = state_layout.identity().to_owned();
        let identity = if let Some(binding) = &state_binding {
            format!(
                "krasis-coupled/2:realization={}:state-layout={state_layout_identity}:state-binding={}",
                realization.digest(),
                binding.identity()
            )
        } else {
            format!(
                "krasis-coupled/1:realization={}:state-layout={state_layout_identity}",
                realization.digest()
            )
        };
        Ok(Self {
            realization,
            state_layout_identity,
            block_layout,
            state_binding,
            consistent_initialization: None,
            identity,
        })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn realization(&self) -> &RealizationPlan {
        &self.realization
    }

    pub fn dimension(&self) -> usize {
        self.realization.dimension()
    }

    pub fn state_binding(&self) -> Option<&StateBinding> {
        self.state_binding.as_ref()
    }

    pub fn state_layout_identity(&self) -> &str {
        &self.state_layout_identity
    }

    /// The per-row differential/algebraic mask recorded by
    /// [`Self::with_consistent_initialization`], if any.
    pub fn row_kinds(&self) -> Option<&[RowKind]> {
        self.consistent_initialization
            .as_ref()
            .map(|config| config.mask.as_slice())
    }

    /// Additively record a per-row differential/algebraic mask and Newton policy for index-1
    /// consistent initialization, folding both into the operator identity.
    ///
    /// Without this, [`DaeOperator::make_initial_state_consistent`] stays the inherited no-op,
    /// identical to today's behavior.
    pub fn with_consistent_initialization(
        mut self,
        mask: Vec<RowKind>,
        newton: NewtonConfig,
    ) -> Result<Self, KrasisError> {
        if mask.len() != self.dimension() {
            return Err(KrasisError::ConsistentInitializationMaskLength {
                actual: mask.len(),
                expected: self.dimension(),
            });
        }
        self.identity = format!(
            "{}:consistent-init={}",
            self.identity,
            consistent_initialization_identity(&mask, &newton)
        );
        self.consistent_initialization = Some(ConsistentInitialization { mask, newton });
        Ok(self)
    }

    /// Solve `F(time, state, ydot) = 0` for `ydot` given a fixed `state`, restricted to rows the
    /// recorded mask marks `Differential`, via dense Newton over an operator wrapper whose
    /// unknown is the reduced state rate.
    ///
    /// An `Algebraic` row never contributes a rate unknown (a semi-explicit index-1 DAE's
    /// algebraic equations do not depend on `ydot`); its returned rate is `0.0`, and this
    /// refuses unless that row's residual already vanishes (within the recorded Newton
    /// configuration's absolute tolerance) at `state` — an inconsistent initial condition on an
    /// algebraic row cannot be repaired by any choice of differential state rate. Requires a
    /// mask recorded by [`Self::with_consistent_initialization`].
    pub fn solve_consistent_state_rate(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &[f64],
    ) -> Result<Vec<f64>, KrasisError> {
        let config = self.consistent_initialization.as_ref().ok_or_else(|| {
            KrasisError::InvalidCoupling(
                "operator has no differential/algebraic mask for consistent initialization".into(),
            )
        })?;
        solve_consistent_state_rate_for(self, &config.mask, &config.newton, context, time, state)
    }

    fn numeric_error(error: finitum::FinitumError) -> NumericError {
        NumericError::Operator {
            message: error.to_string(),
        }
    }
}

/// Solve `F(time, state, ydot) = 0` for `ydot` restricted to the rows `mask` marks
/// `Differential`, by dense Newton over a reduced-rate wrapper of `operator`; algebraic rows
/// keep rate zero and must already vanish at `state` (see
/// [`CoupledOperator::solve_consistent_state_rate`]). Shared by every Krasis-composed operator.
pub(crate) fn solve_consistent_state_rate_for<O: DaeOperator + ?Sized>(
    operator: &O,
    mask: &[RowKind],
    newton: &NewtonConfig,
    context: &EvaluationContext,
    time: f64,
    state: &[f64],
) -> Result<Vec<f64>, KrasisError> {
    {
        let dimension = operator.dimension();
        if mask.len() != dimension {
            return Err(KrasisError::ConsistentInitializationMaskLength {
                actual: mask.len(),
                expected: dimension,
            });
        }
        if state.len() != dimension {
            return Err(KrasisError::InvalidCoupling(format!(
                "state length {} differs from operator dimension {}",
                state.len(),
                dimension
            )));
        }
        if !time.is_finite() {
            return Err(KrasisError::InvalidCoupling(
                "consistent initialization time must be finite".into(),
            ));
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(KrasisError::InvalidCoupling(
                "consistent initialization state must be finite".into(),
            ));
        }

        // The unknown is the state rate restricted to differential rows; an algebraic row
        // never contributes a rate unknown (its residual is defined not to depend on `ydot`,
        // matching a semi-explicit index-1 DAE), so its rate is fixed at zero while solving.
        let differential_rows: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, row)| matches!(row, RowKind::Differential))
            .map(|(index, _)| index)
            .collect();
        let algebraic_rows: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, row)| matches!(row, RowKind::Algebraic))
            .map(|(index, _)| index)
            .collect();

        struct ReducedRateOperator<'a, O: DaeOperator + ?Sized> {
            operator: &'a O,
            time: f64,
            differential_state: &'a [f64],
            differential_rows: &'a [usize],
        }

        impl<O: DaeOperator + ?Sized> ReducedRateOperator<'_, O> {
            fn expand(&self, reduced: &[f64]) -> Vec<f64> {
                let mut full = vec![0.0; self.operator.dimension()];
                for (local, &global) in self.differential_rows.iter().enumerate() {
                    full[global] = reduced[local];
                }
                full
            }

            fn restrict(&self, full: &[f64], output: &mut [f64]) {
                for (local, &global) in self.differential_rows.iter().enumerate() {
                    output[local] = full[global];
                }
            }
        }

        impl<O: DaeOperator + ?Sized> NonlinearOperator for ReducedRateOperator<'_, O> {
            fn dimension(&self) -> usize {
                self.differential_rows.len()
            }

            fn residual(
                &self,
                context: &EvaluationContext,
                state_rate: &[f64],
                output: &mut [f64],
            ) -> Result<(), NumericError> {
                let full_rate = self.expand(state_rate);
                let mut full_output = vec![0.0; self.operator.dimension()];
                DaeOperator::residual(
                    self.operator,
                    context,
                    self.time,
                    self.differential_state,
                    &full_rate,
                    &mut full_output,
                )?;
                self.restrict(&full_output, output);
                Ok(())
            }

            fn jacobian_vector_product(
                &self,
                context: &EvaluationContext,
                state_rate: &[f64],
                direction: &[f64],
                output: &mut [f64],
            ) -> Result<(), NumericError> {
                let full_rate = self.expand(state_rate);
                let full_rate_direction = self.expand(direction);
                let zero_state_direction = vec![0.0; self.operator.dimension()];
                let mut full_output = vec![0.0; self.operator.dimension()];
                DaeOperator::jacobian_vector_product(
                    self.operator,
                    context,
                    self.time,
                    self.differential_state,
                    &full_rate,
                    &zero_state_direction,
                    &full_rate_direction,
                    &mut full_output,
                )?;
                self.restrict(&full_output, output);
                Ok(())
            }
        }

        let wrapper = ReducedRateOperator {
            operator,
            time,
            differential_state: state,
            differential_rows: &differential_rows,
        };
        let initial_guess = vec![0.0; differential_rows.len()];
        let report = solve_newton(&wrapper, context, &initial_guess, newton)
            .map_err(|error| KrasisError::Solve(error.to_string()))?;
        if !report.converged {
            return Err(KrasisError::Solve(
                "consistent initialization Newton solve did not converge".into(),
            ));
        }
        if report.state.iter().any(|value| !value.is_finite()) {
            return Err(KrasisError::Solve(
                "consistent state rate solve produced non-finite values".into(),
            ));
        }

        let mut state_rate = vec![0.0; dimension];
        for (local, &global) in differential_rows.iter().enumerate() {
            state_rate[global] = report.state[local];
        }

        // An algebraic row's residual never depends on `ydot`, so it must already vanish at
        // `state`; a nonzero residual there means the supplied initial condition is not
        // consistent with the algebraic constraint, and no choice of differential state rate
        // can fix it.
        if !algebraic_rows.is_empty() {
            let mut residual = vec![0.0; dimension];
            operator
                .residual(context, time, state, &state_rate, &mut residual)
                .map_err(|error| KrasisError::Solve(error.to_string()))?;
            for &row in &algebraic_rows {
                if residual[row].abs() > newton.absolute_tolerance {
                    return Err(KrasisError::InvalidCoupling(format!(
                        "row {row} is marked algebraic but has residual {} at the supplied \
                         initial state, which exceeds the configured absolute tolerance {}; the \
                         initial condition is not consistent with the algebraic constraint",
                        residual[row], newton.absolute_tolerance
                    )));
                }
            }
        }

        Ok(state_rate)
    }
}

pub(crate) fn consistent_initialization_identity(
    mask: &[RowKind],
    newton: &NewtonConfig,
) -> String {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        mask: &'a [RowKind],
        newton: &'a NewtonConfig,
    }

    let bytes = serde_json::to_vec(&Payload {
        schema: "krasis-consistent-init/1",
        mask,
        newton,
    })
    .expect("consistent-initialization identity payload is serializable");
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}

impl NonlinearOperator for CoupledOperator {
    fn dimension(&self) -> usize {
        self.realization.dimension()
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        // Methodus's time-independent nonlinear contract is the steady view: t = 0, ydot = 0.
        // Time-dependent boundary/material behavior must use the DAE implementation below.
        self.realization
            .residual(0.0, state, &vec![0.0; self.dimension()], output)
            .map_err(Self::numeric_error)
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        let zero = vec![0.0; self.dimension()];
        self.realization
            .jacobian_vector_product(0.0, state, &zero, direction, &zero, output)
            .map_err(Self::numeric_error)
    }
}

impl BlockNonlinearOperator for CoupledOperator {
    fn block_layout(&self) -> &SolverBlockLayout {
        &self.block_layout
    }
}

impl DaeOperator for CoupledOperator {
    fn dimension(&self) -> usize {
        self.realization.dimension()
    }

    fn residual(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.realization
            .residual(time, state, state_rate, output)
            .map_err(Self::numeric_error)
    }

    fn jacobian_vector_product(
        &self,
        _context: &EvaluationContext,
        time: f64,
        state: &[f64],
        state_rate: &[f64],
        state_direction: &[f64],
        rate_direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), NumericError> {
        self.realization
            .jacobian_vector_product(
                time,
                state,
                state_rate,
                state_direction,
                rate_direction,
                output,
            )
            .map_err(Self::numeric_error)
    }

    fn make_initial_state_consistent(
        &self,
        context: &EvaluationContext,
        time: f64,
        state: &mut [f64],
    ) -> Result<(), NumericError> {
        if self.consistent_initialization.is_none() {
            // No mask was recorded at construction: identical to the inherited no-op.
            return Ok(());
        }
        // This never adjusts `state`, only the (discarded) state rate; it exists to validate
        // that a consistent state rate exists at `time`, refusing early rather than at the
        // first BDF step.
        self.solve_consistent_state_rate(context, time, state)
            .map(|_| ())
            .map_err(|error| NumericError::Operator {
                message: error.to_string(),
            })
    }
}

/// Serializable restart data for both Krasis transactions and Methodus BDF history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoupledCheckpoint {
    pub operator_identity: String,
    pub state: Checkpoint,
    pub integrator: BdfState,
}

impl TransactionalOperator for CoupledOperator {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn state_layout_identity(&self) -> &str {
        &self.state_layout_identity
    }
}

/// Transactional execution state for implicit transient solves over any
/// [`TransactionalOperator`]: a single Finitum realization ([`CoupledOperator`], the default)
/// or an N-leaf composition across realization plans ([`crate::CoupledSystemOperator`]).
#[derive(Clone, Debug)]
pub struct CoupledExecution<Op: TransactionalOperator = CoupledOperator> {
    operator: Op,
    state: SimulationState,
    integrator: BdfState,
}

impl<Op: TransactionalOperator> CoupledExecution<Op> {
    pub fn new(
        operator: Op,
        state: SimulationState,
        context: &EvaluationContext,
    ) -> Result<Self, KrasisError> {
        if state.phase() != TransactionPhase::Committed {
            return Err(KrasisError::InvalidCoupling(
                "coupled execution must start from committed state".into(),
            ));
        }
        if state.layout().identity() != operator.state_layout_identity() {
            return Err(KrasisError::InvalidCoupling(
                "state layout does not match the coupled operator".into(),
            ));
        }
        if state.step() != 0 {
            return Err(KrasisError::InvalidCoupling(
                "nonzero-step state must be loaded through a coupled checkpoint".into(),
            ));
        }
        let values = state.committed_vector()?;
        let integrator = BdfState::initialize(&operator, context, state.time(), values)
            .map_err(|error| KrasisError::Solve(error.to_string()))?;
        let execution = Self {
            operator,
            state,
            integrator,
        };
        execution.validate_synchronized()?;
        Ok(execution)
    }

    pub fn operator(&self) -> &Op {
        &self.operator
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    pub fn integrator(&self) -> &BdfState {
        &self.integrator
    }

    /// Attempt one Methodus BDF step inside the Krasis trial transaction, with the dense Newton
    /// solve `bdf_step` runs from `config.newton`.
    pub fn attempt_step(
        &mut self,
        context: &EvaluationContext,
        step: f64,
        config: &BdfConfig,
    ) -> Result<StepOutcome, KrasisError> {
        self.transact_step(|operator, integrator| {
            bdf_step(operator, context, integrator, step, config)
        })
    }

    /// Attempt one Methodus BDF step inside the Krasis trial transaction, with the nonlinear
    /// solve delegated to `solver` (`methodus::bdf_step_with`): a `NewtonKrylovSolver` for a
    /// matrix-free inexact Newton over the step's Jacobian action, a `BlockNewton` for
    /// partitioned Gauss-Seidel/Jacobi iteration over the operator's block layout inside the
    /// step, or `DenseNewton`. `config.newton` is not consulted; the commit/rollback
    /// protocol is exactly [`Self::attempt_step`]'s.
    pub fn attempt_step_with(
        &mut self,
        context: &EvaluationContext,
        step: f64,
        config: &BdfConfig,
        solver: &dyn NonlinearSolver,
    ) -> Result<StepOutcome, KrasisError> {
        self.transact_step(|operator, integrator| {
            bdf_step_with(operator, context, integrator, step, config, solver)
        })
    }

    /// One BDF attempt enclosed in `begin_trial` / `commit` / `rollback`: an accepted step
    /// commits the new values at the step's time; a rejected step or a solver error rolls the
    /// trial back and leaves the committed state and the BDF history untouched.
    fn transact_step(
        &mut self,
        attempt: impl FnOnce(&Op, &BdfState) -> Result<StepOutcome, SolveError>,
    ) -> Result<StepOutcome, KrasisError> {
        self.validate_synchronized()?;
        self.state.begin_trial()?;
        let outcome = match attempt(&self.operator, &self.integrator) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.state.rollback()?;
                return Err(KrasisError::Solve(error.to_string()));
            }
        };
        match &outcome {
            StepOutcome::Accepted(accepted) => {
                if let Err(error) = self.state.set_trial_vector(&accepted.state.values) {
                    self.state.rollback()?;
                    return Err(error);
                }
                if let Err(error) = self.state.commit(accepted.state.time) {
                    self.state.rollback()?;
                    return Err(error);
                }
                self.integrator = accepted.state.clone();
            }
            StepOutcome::Rejected(_) => self.state.rollback()?,
        }
        self.validate_synchronized()?;
        Ok(outcome)
    }

    pub fn checkpoint(&self) -> Result<CoupledCheckpoint, KrasisError> {
        self.validate_synchronized()?;
        Ok(CoupledCheckpoint {
            operator_identity: self.operator.identity().to_owned(),
            state: self.state.checkpoint()?,
            integrator: self.integrator.clone(),
        })
    }

    /// Atomically restore transactional state and BDF history after validating their identity.
    pub fn restore(&mut self, checkpoint: &CoupledCheckpoint) -> Result<(), KrasisError> {
        if checkpoint.operator_identity != self.operator.identity() {
            return Err(KrasisError::InvalidCoupling(format!(
                "checkpoint operator identity `{}` does not match `{}`",
                checkpoint.operator_identity,
                self.operator.identity()
            )));
        }
        let mut candidate = self.state.clone();
        candidate.restore(&checkpoint.state)?;
        validate_pair(
            &candidate,
            &checkpoint.integrator,
            DaeOperator::dimension(&self.operator),
        )?;
        self.state = candidate;
        self.integrator = checkpoint.integrator.clone();
        Ok(())
    }

    fn validate_synchronized(&self) -> Result<(), KrasisError> {
        validate_pair(
            &self.state,
            &self.integrator,
            DaeOperator::dimension(&self.operator),
        )
    }
}

fn validate_pair(
    state: &SimulationState,
    integrator: &BdfState,
    dimension: usize,
) -> Result<(), KrasisError> {
    let committed = state.committed_vector()?;
    if state.phase() != TransactionPhase::Committed
        || committed.len() != dimension
        || integrator.values != committed
        || integrator.time != state.time()
        || integrator.accepted_steps != state.step()
    {
        return Err(KrasisError::InvalidCoupling(
            "Krasis committed state and Methodus BDF state are not synchronized".into(),
        ));
    }
    if integrator.values.iter().any(|value| !value.is_finite()) {
        return Err(KrasisError::InvalidCoupling(
            "BDF state contains non-finite values".into(),
        ));
    }
    match (&integrator.previous_values, integrator.previous_step) {
        (None, None) if integrator.accepted_steps == 0 => {
            if state.history_vector(0)?.is_some() {
                return Err(KrasisError::InvalidCoupling(
                    "initial BDF state unexpectedly has Krasis history".into(),
                ));
            }
        }
        (Some(previous), Some(previous_step))
            if previous.len() == dimension
                && previous.iter().all(|value| value.is_finite())
                && previous_step.is_finite()
                && previous_step > 0.0
                && state.history_vector(0)?.as_deref() == Some(previous.as_slice()) => {}
        _ => {
            return Err(KrasisError::InvalidCoupling(
                "Krasis field history and Methodus BDF history are inconsistent".into(),
            ));
        }
    }
    Ok(())
}
