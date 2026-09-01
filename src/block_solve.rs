//! Transactional MINRES execution over a block-composed `methodus::BlockLinearOperator` (E6).
//!
//! [`BlockLinearExecution`] drives `methodus::solve_minres` over one committed
//! [`SimulationState`], recording every attempt through the same trial/commit/rollback
//! transaction Krasis already uses for BDF stepping (see [`crate::CoupledExecution`]): a
//! converged solve's solution becomes the new committed state; a solve that errors or does not
//! converge is never committed -- its trial is rolled back before a typed refusal is returned.
//!
//! This module names no physics and holds no Finitum type: it drives any
//! `methodus::BlockLinearOperator`. Finitum's `MixedOperator` (`finitum::mixed`) composes
//! exactly that trait over its own `MixedSpace`/`SystemRealizationPlan` machinery, which this
//! module never references.

use serde::{Deserialize, Serialize};

use methodus::{
    BlockLinearOperator, EvaluationContext, LinearSolveReport, MinresConfig, NullspaceProjector,
    OperatorProperties, Preconditioner, solve_minres,
};

use crate::{Checkpoint, KrasisError, SimulationState, TransactionPhase};

/// Serializable restart data binding a [`Checkpoint`] to the exact block operator it was
/// produced against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockLinearCheckpoint {
    pub operator_identity: String,
    pub state: Checkpoint,
}

/// Transactional execution state driving MINRES solves over one committed block state.
#[derive(Debug)]
pub struct BlockLinearExecution<'op, Op: BlockLinearOperator> {
    operator: &'op Op,
    operator_identity: String,
    state: SimulationState,
}

impl<'op, Op: BlockLinearOperator> BlockLinearExecution<'op, Op> {
    /// Binds `operator` to `state`, refusing a dimension or per-block length mismatch and a
    /// state that is not already complete and in committed phase.
    pub fn new(operator: &'op Op, state: SimulationState) -> Result<Self, KrasisError> {
        if state.phase() != TransactionPhase::Committed {
            return Err(KrasisError::InvalidCoupling(
                "block linear execution must start from committed state".into(),
            ));
        }
        // Also validates every layout block has an initialized field.
        let width = state.committed_vector()?.len();
        if operator.rows() != width || operator.columns() != width {
            return Err(KrasisError::InvalidCoupling(format!(
                "block operator is {}x{}, Krasis state width is {width}",
                operator.rows(),
                operator.columns()
            )));
        }
        let operator_layout = operator.block_layout();
        if operator_layout.dimension() != width {
            return Err(KrasisError::InvalidCoupling(format!(
                "block operator layout has dimension {}, Krasis state width is {width}",
                operator_layout.dimension()
            )));
        }
        let operator_blocks = operator_layout.blocks();
        let state_blocks = state.layout().blocks();
        if operator_blocks.len() != state_blocks.len() {
            return Err(KrasisError::InvalidCoupling(format!(
                "block operator declares {} blocks, Krasis state layout declares {}",
                operator_blocks.len(),
                state_blocks.len()
            )));
        }
        for (operator_block, state_block) in operator_blocks.iter().zip(state_blocks) {
            if operator_block.length() != state_block.range().len() {
                return Err(KrasisError::InvalidCoupling(format!(
                    "block operator block `{}` has length {}, Krasis block `{}` has length {}",
                    operator_block.name(),
                    operator_block.length(),
                    state_block.id(),
                    state_block.range().len()
                )));
            }
        }
        let operator_identity = operator_identity(operator);
        Ok(Self {
            operator,
            operator_identity,
            state,
        })
    }

    pub fn operator_identity(&self) -> &str {
        &self.operator_identity
    }

    pub fn state(&self) -> &SimulationState {
        &self.state
    }

    /// Attempts one MINRES solve inside a Krasis trial transaction, starting from the current
    /// committed state as the initial guess. A converged solve commits at `commit_time`; a
    /// solve that errors or does not converge rolls back to the prior committed state and
    /// returns a typed refusal -- this never commits an unconverged solution.
    pub fn solve(
        &mut self,
        context: &EvaluationContext,
        right_hand_side: &[f64],
        preconditioner: Option<&dyn Preconditioner>,
        nullspace_projector: Option<&dyn NullspaceProjector>,
        config: &MinresConfig,
        commit_time: f64,
    ) -> Result<LinearSolveReport, KrasisError> {
        let initial_solution = self.state.committed_vector()?;
        self.state.begin_trial()?;
        let report = match solve_minres(
            self.operator,
            preconditioner,
            nullspace_projector,
            context,
            right_hand_side,
            &initial_solution,
            config,
        ) {
            Ok(report) => report,
            Err(error) => {
                self.state.rollback()?;
                return Err(KrasisError::Solve(error.to_string()));
            }
        };
        if !report.converged {
            self.state.rollback()?;
            return Err(KrasisError::Solve(
                "minres solve did not converge within the configured tolerance".into(),
            ));
        }
        if let Err(error) = self.state.set_trial_vector(&report.solution) {
            self.state.rollback()?;
            return Err(error);
        }
        if let Err(error) = self.state.commit(commit_time) {
            self.state.rollback()?;
            return Err(error);
        }
        Ok(report)
    }

    pub fn checkpoint(&self) -> Result<BlockLinearCheckpoint, KrasisError> {
        Ok(BlockLinearCheckpoint {
            operator_identity: self.operator_identity.clone(),
            state: self.state.checkpoint()?,
        })
    }

    /// Atomically restores state after validating it was checkpointed against this exact
    /// operator identity.
    pub fn restore(&mut self, checkpoint: &BlockLinearCheckpoint) -> Result<(), KrasisError> {
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

/// Identity built only from what [`BlockLinearOperator`] exposes: dimension, declared
/// [`OperatorProperties`], and block layout. This distinguishes operators of different shape or
/// declared symmetry/definiteness/nullspace, but -- unlike a content digest such as
/// `finitum::RealizationPlan::digest` -- it cannot distinguish two operators that share every
/// one of those and differ only in their numerical action (e.g. two `MixedOperator`s over the
/// same mesh and block layout with different coupling scales). Finitum's `MixedOperator`
/// carries no such digest today; see the batch report for a proposed follow-up.
fn operator_identity<Op: BlockLinearOperator>(operator: &Op) -> String {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        rows: usize,
        columns: usize,
        properties: OperatorProperties,
        block_layout: &'a methodus::BlockLayout,
    }

    let payload = Payload {
        schema: "krasis-block-linear/1",
        rows: operator.rows(),
        columns: operator.columns(),
        properties: operator.properties(),
        block_layout: operator.block_layout(),
    };
    let bytes =
        serde_json::to_vec(&payload).expect("block operator identity payload is serializable");
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}
