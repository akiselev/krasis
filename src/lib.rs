//! Coupled field state and transactional simulation runtime.

mod connected;
pub use connected::ConnectedSystemOperator;
mod binding;
mod block_binding;
mod block_preconditioner;
mod block_solve;
mod checkpoint;
mod coupled;
mod coupled_system;
mod error;
mod event;
mod explicit;
mod initial;
mod layout;
mod partitioned;
mod state;
mod verification;

pub use binding::{SemanticId, StateBinding};
pub use block_binding::block_state_layout;
pub use block_preconditioner::{BlockCoupling, BlockGaussSeidelFactory};
pub use block_solve::{
    BlockLinearAlgorithm, BlockLinearCheckpoint, BlockLinearExecution, BlockLinearReport,
    BlockLinearSolver, OperatorIdentity,
};
pub use checkpoint::Checkpoint;
pub use coupled::{
    CoupledCheckpoint, CoupledExecution, CoupledOperator, EvaluationRefusal, FinitumRealization,
    RefusedAttempt, RowKind, TransactionalOperator,
};
pub use coupled_system::{
    CoupledLeaf, CoupledSystemOperator, CouplingArgument, CouplingDependency, CouplingEdge,
    CouplingGraph,
};
pub use error::KrasisError;
pub use event::{EventDirection, EventRecord};
pub use explicit::{
    EXPLICIT_CONFIG_IDENTITY_SCHEMA, ExplicitCheckpoint, ExplicitConservationSummary,
    ExplicitExecution, ExplicitRefusedAttempt, ExplicitRejectionRecord, ExplicitStepRecord,
    InvariantBalanceCheck, explicit_config_identity,
};
pub use initial::{NodalContext, initial_state_from};
pub use layout::{BlockId, StateBlock, StateLayout};
pub use partitioned::{
    PARTITIONED_ACCELERATION_BREAKDOWN, PARTITIONED_DIVERGED, PARTITIONED_MAX_SWEEPS,
    PARTITIONED_REFUSAL_ORIGIN, PartitionedAcceleration, PartitionedCheckpoint, PartitionedConfig,
    PartitionedDisposition, PartitionedExecution, PartitionedFixedPoint, PartitionedIteration,
    PartitionedReport, PartitionedSchedule, PartitionedSolve, SweepReport, run_partitioned,
};
pub use state::{ConstitutiveSlot, FieldId, SimulationState, TransactionPhase};
pub use verification::{
    AttemptDisposition, CrossBlockDerivativeCheck, CrossBlockDerivativeReport, EventDisposition,
    EventStateReport, FinitumSourceBinding, FinitumVerificationSource, HistoryReport,
    KRASIS_VERIFICATION_SCHEMA, RestartTrajectoryReport, RollbackIdentityReport, StrategyAgreement,
    StrategyOutcome, StrategyWorkReport, ValidatedKrasisVerification, VerificationBinding,
    VerificationRefusal, check_cross_block_derivatives, check_event_state, check_event_state_from,
    check_history_and_rejection, check_restart_trajectory, check_rollback_identity,
    check_strategy_work,
};
