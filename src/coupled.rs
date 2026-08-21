use finitum::RealizationPlan;
use methodus::{
    BdfConfig, BdfState, BlockLayout as SolverBlockLayout, BlockNonlinearOperator, BlockSpec,
    DaeOperator, EvaluationContext, NonlinearOperator, NumericError, StepOutcome, bdf_step,
};
use serde::{Deserialize, Serialize};

use crate::{Checkpoint, KrasisError, SimulationState, StateLayout, TransactionPhase};

/// Krasis-owned composition of a realized Finitum action into Methodus contracts.
///
/// Its [`NonlinearOperator`] implementation is the steady view at `t = 0` and `ydot = 0`.
/// Time-dependent boundary, source, or material behavior must use its [`DaeOperator`] view.
#[derive(Clone, Debug)]
pub struct CoupledOperator {
    realization: RealizationPlan,
    state_layout_identity: String,
    block_layout: SolverBlockLayout,
    identity: String,
}

impl CoupledOperator {
    pub fn new(
        realization: RealizationPlan,
        state_layout: &StateLayout,
    ) -> Result<Self, KrasisError> {
        if realization.dimension() != state_layout.width() {
            return Err(KrasisError::InvalidCoupling(format!(
                "Finitum dimension {} differs from Krasis state width {}",
                realization.dimension(),
                state_layout.width()
            )));
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
        let identity = format!(
            "krasis-coupled/1:realization={}:state-layout={state_layout_identity}",
            realization.digest()
        );
        Ok(Self {
            realization,
            state_layout_identity,
            block_layout,
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

    fn numeric_error(error: finitum::FinitumError) -> NumericError {
        NumericError::Operator {
            message: error.to_string(),
        }
    }
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
}

/// Serializable restart data for both Krasis transactions and Methodus BDF history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoupledCheckpoint {
    pub operator_identity: String,
    pub state: Checkpoint,
    pub integrator: BdfState,
}

/// Transactional execution state for implicit transient solves.
#[derive(Clone, Debug)]
pub struct CoupledExecution {
    operator: CoupledOperator,
    state: SimulationState,
    integrator: BdfState,
}

impl CoupledExecution {
    pub fn new(
        operator: CoupledOperator,
        state: SimulationState,
        context: &EvaluationContext,
    ) -> Result<Self, KrasisError> {
        if state.phase() != TransactionPhase::Committed {
            return Err(KrasisError::InvalidCoupling(
                "coupled execution must start from committed state".into(),
            ));
        }
        if state.layout().identity() != operator.state_layout_identity {
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

    pub fn operator(&self) -> &CoupledOperator {
        &self.operator
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    pub fn integrator(&self) -> &BdfState {
        &self.integrator
    }

    /// Attempt one Methodus BDF step inside the Krasis trial transaction.
    pub fn attempt_step(
        &mut self,
        context: &EvaluationContext,
        step: f64,
        config: &BdfConfig,
    ) -> Result<StepOutcome, KrasisError> {
        self.validate_synchronized()?;
        self.state.begin_trial()?;
        let outcome = match bdf_step(&self.operator, context, &self.integrator, step, config) {
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
            operator_identity: self.operator.identity.clone(),
            state: self.state.checkpoint()?,
            integrator: self.integrator.clone(),
        })
    }

    /// Atomically restore transactional state and BDF history after validating their identity.
    pub fn restore(&mut self, checkpoint: &CoupledCheckpoint) -> Result<(), KrasisError> {
        if checkpoint.operator_identity != self.operator.identity {
            return Err(KrasisError::InvalidCoupling(format!(
                "checkpoint operator identity `{}` does not match `{}`",
                checkpoint.operator_identity, self.operator.identity
            )));
        }
        let mut candidate = self.state.clone();
        candidate.restore(&checkpoint.state)?;
        validate_pair(
            &candidate,
            &checkpoint.integrator,
            self.operator.dimension(),
        )?;
        self.state = candidate;
        self.integrator = checkpoint.integrator.clone();
        Ok(())
    }

    fn validate_synchronized(&self) -> Result<(), KrasisError> {
        validate_pair(&self.state, &self.integrator, self.operator.dimension())
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
