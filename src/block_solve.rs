//! Transactional Krylov execution over a block-composed `methodus::BlockLinearOperator` (E6,
//! SC-W1 steady-runner target).
//!
//! [`BlockLinearExecution`] drives one Methodus linear algorithm (`solve_conjugate_gradient`,
//! `solve_minres`, or `solve_gmres`, chosen by the caller's [`BlockLinearSolver`] policy) over
//! one committed [`SimulationState`], recording every attempt through the same
//! trial/commit/rollback transaction Krasis already uses for BDF stepping (see
//! [`crate::CoupledExecution`]): a converged solve's solution becomes the new committed state; a
//! solve that errors or does not converge is never committed -- its trial is rolled back before a
//! typed refusal is returned.
//!
//! The Methodus call made here is exactly the call a direct caller would make with the same
//! operator, preconditioner, projector, context, right-hand side, initial guess (the committed
//! state) and configuration, so a steady runner rerouted through this execution reproduces its
//! direct Methodus solution bit for bit (`tests/sc_w1_steady_reroute.rs`).
//!
//! This module names no physics. It is generic over any `methodus::BlockLinearOperator` that
//! also carries a Krasis [`OperatorIdentity`]; the Finitum block operators Krasis composes
//! (`MixedOperator`, `SystemOperator`, `ReducedSystemOperator`) implement it from their own
//! content digests, so a [`BlockLinearCheckpoint`] binds to the operator's numerical content and
//! not only to its shape.

use serde::{Deserialize, Serialize};

use methodus::{
    BlockLinearOperator, ConjugateGradientConfig, EvaluationContext, GmresConfig,
    LinearSolveReport, MinresConfig, NullspaceProjector, OperatorProperties, Preconditioner,
    solve_conjugate_gradient, solve_gmres, solve_minres,
};

use crate::{Checkpoint, KrasisError, SimulationState, TransactionPhase};

/// Content-addressed identity of an operator's numerical action, for binding Krasis checkpoints
/// and reports to the exact operator they were produced against.
///
/// Two operators with the same shape (dimension, block layout, declared properties) but a
/// different action -- a different mesh, coefficient, coupling scale, or constraint set -- must
/// report different identities. Finitum's block operators implement this from their own content
/// digests (`finitum::MixedOperator::digest`, `finitum::SystemOperator::digest`, plus the
/// serialized `ConstraintSet` for the reduced operator).
pub trait OperatorIdentity {
    fn content_identity(&self) -> String;
}

impl OperatorIdentity for finitum::MixedOperator {
    fn content_identity(&self) -> String {
        format!("finitum-mixed:{}", self.digest())
    }
}

impl OperatorIdentity for finitum::SystemOperator {
    fn content_identity(&self) -> String {
        format!("finitum-system:{}", self.digest())
    }
}

impl OperatorIdentity for finitum::ReducedSystemOperator {
    fn content_identity(&self) -> String {
        format!(
            "finitum-reduced-system:{}:constraints={}",
            self.operator().digest(),
            constraint_set_identity(self.constraints())
        )
    }
}

impl<Op: OperatorIdentity + ?Sized> OperatorIdentity for &Op {
    fn content_identity(&self) -> String {
        (**self).content_identity()
    }
}

/// Blake3 over the serialized `ConstraintSet` (targets, dependencies, weights, offsets), which
/// `ConstraintSet::new` has already validated finite; the bytes are hashed as serialized.
fn constraint_set_identity(constraints: &finitum::ConstraintSet) -> String {
    let bytes = serde_json::to_vec(constraints).expect("constraint set is serializable");
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}

/// The Methodus linear algorithm a [`BlockLinearExecution::solve`] runs. Selection is the
/// caller's policy (Sinbad's `SolvePolicy`); Krasis only executes the chosen algorithm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BlockLinearSolver {
    ConjugateGradient(ConjugateGradientConfig),
    Minres(MinresConfig),
    Gmres(GmresConfig),
}

impl BlockLinearSolver {
    pub fn algorithm(&self) -> BlockLinearAlgorithm {
        match self {
            Self::ConjugateGradient(_) => BlockLinearAlgorithm::ConjugateGradient,
            Self::Minres(_) => BlockLinearAlgorithm::Minres,
            Self::Gmres(_) => BlockLinearAlgorithm::Gmres,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockLinearAlgorithm {
    ConjugateGradient,
    Minres,
    Gmres,
}

impl BlockLinearAlgorithm {
    /// Stable lower-case label (`conjugate_gradient`, `minres`, `gmres`) for run receipts.
    pub const fn label(self) -> &'static str {
        match self {
            Self::ConjugateGradient => "conjugate_gradient",
            Self::Minres => "minres",
            Self::Gmres => "gmres",
        }
    }
}

/// A converged, committed solve: the Methodus report exactly as the algorithm returned it, plus
/// the restart-cycle count GMRES reports and the other algorithms do not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockLinearReport {
    pub algorithm: BlockLinearAlgorithm,
    pub report: LinearSolveReport,
    pub restart_cycles: Option<usize>,
}

/// Serializable restart data binding a [`Checkpoint`] to the exact block operator it was
/// produced against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockLinearCheckpoint {
    pub operator_identity: String,
    pub state: Checkpoint,
}

/// Transactional execution state driving Krylov solves over one committed block state.
#[derive(Debug)]
pub struct BlockLinearExecution<'op, Op: BlockLinearOperator + OperatorIdentity> {
    operator: &'op Op,
    operator_identity: String,
    state: SimulationState,
}

impl<'op, Op: BlockLinearOperator + OperatorIdentity> BlockLinearExecution<'op, Op> {
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

    /// Attempts one linear solve inside a Krasis trial transaction, starting from the current
    /// committed state as the initial guess. A converged solve commits at `commit_time`; a
    /// solve that errors or does not converge rolls back to the prior committed state and
    /// returns a typed refusal ([`KrasisError::SolveDidNotConverge`] carries the algorithm and
    /// the iteration count) -- this never commits an unconverged solution.
    ///
    /// A `nullspace_projector` is admitted only with [`BlockLinearSolver::Minres`], the one
    /// Methodus algorithm that takes one; passing it with another algorithm is a typed refusal
    /// rather than a silent drop.
    pub fn solve(
        &mut self,
        context: &EvaluationContext,
        right_hand_side: &[f64],
        preconditioner: Option<&dyn Preconditioner>,
        nullspace_projector: Option<&dyn NullspaceProjector>,
        solver: &BlockLinearSolver,
        commit_time: f64,
    ) -> Result<BlockLinearReport, KrasisError> {
        let algorithm = solver.algorithm();
        if nullspace_projector.is_some() && !matches!(solver, BlockLinearSolver::Minres(_)) {
            return Err(KrasisError::InvalidCoupling(format!(
                "a nullspace projector is only admitted with minres, not {}",
                algorithm.label()
            )));
        }
        let initial_solution = self.state.committed_vector()?;
        self.state.begin_trial()?;
        let outcome = match solver {
            BlockLinearSolver::ConjugateGradient(config) => solve_conjugate_gradient(
                self.operator,
                preconditioner,
                context,
                right_hand_side,
                &initial_solution,
                config,
            )
            .map(|report| (report, None)),
            BlockLinearSolver::Minres(config) => solve_minres(
                self.operator,
                preconditioner,
                nullspace_projector,
                context,
                right_hand_side,
                &initial_solution,
                config,
            )
            .map(|report| (report, None)),
            BlockLinearSolver::Gmres(config) => solve_gmres(
                self.operator,
                preconditioner,
                context,
                right_hand_side,
                &initial_solution,
                config,
            )
            .map(|gmres| {
                let restart_cycles = gmres.restart_cycles;
                (
                    LinearSolveReport {
                        solution: gmres.solution,
                        converged: gmres.converged,
                        trace: gmres.trace,
                    },
                    Some(restart_cycles),
                )
            }),
        };
        let (report, restart_cycles) = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                self.state.rollback()?;
                return Err(KrasisError::Solve(error.to_string()));
            }
        };
        if !report.converged {
            self.state.rollback()?;
            return Err(KrasisError::SolveDidNotConverge {
                algorithm: algorithm.label().to_owned(),
                iterations: report.trace.len().saturating_sub(1),
            });
        }
        if let Err(error) = self.state.set_trial_vector(&report.solution) {
            self.state.rollback()?;
            return Err(error);
        }
        if let Err(error) = self.state.commit(commit_time) {
            self.state.rollback()?;
            return Err(error);
        }
        Ok(BlockLinearReport {
            algorithm,
            report,
            restart_cycles,
        })
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

/// `krasis-block-linear/2`: the operator's [`OperatorIdentity`] content identity joined with
/// what [`BlockLinearOperator`] exposes (dimension, declared [`OperatorProperties`], block
/// layout). `/1` hashed the shape only and could not distinguish two operators over the same
/// layout with different coefficients; `/2` can.
fn operator_identity<Op: BlockLinearOperator + OperatorIdentity>(operator: &Op) -> String {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        content: String,
        rows: usize,
        columns: usize,
        properties: OperatorProperties,
        block_layout: &'a methodus::BlockLayout,
    }

    let payload = Payload {
        schema: "krasis-block-linear/2",
        content: operator.content_identity(),
        rows: operator.rows(),
        columns: operator.columns(),
        properties: operator.properties(),
        block_layout: operator.block_layout(),
    };
    let bytes =
        serde_json::to_vec(&payload).expect("block operator identity payload is serializable");
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}
