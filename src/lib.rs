//! Coupled field state and transactional simulation runtime.

mod binding;
mod block_binding;
mod block_solve;
mod checkpoint;
mod coupled;
mod coupled_system;
mod error;
mod event;
mod initial;
mod layout;
mod state;
mod verification;

pub use binding::{SemanticId, StateBinding};
pub use block_binding::block_state_layout;
pub use block_solve::{
    BlockLinearAlgorithm, BlockLinearCheckpoint, BlockLinearExecution, BlockLinearReport,
    BlockLinearSolver, OperatorIdentity,
};
pub use checkpoint::Checkpoint;
pub use coupled::{
    CoupledCheckpoint, CoupledExecution, CoupledOperator, RowKind, TransactionalOperator,
};
pub use coupled_system::{
    CoupledLeaf, CoupledSystemOperator, CouplingArgument, CouplingDependency, CouplingEdge,
    CouplingGraph,
};
pub use error::KrasisError;
pub use event::{EventDirection, EventRecord};
pub use initial::{NodalContext, initial_state_from};
pub use layout::{BlockId, StateBlock, StateLayout};
pub use state::{ConstitutiveSlot, FieldId, SimulationState, TransactionPhase};
pub use verification::{
    AttemptDisposition, CrossBlockDerivativeCheck, CrossBlockDerivativeReport, EventDisposition,
    EventStateReport, FinitumVerificationSource, HistoryReport, KRASIS_VERIFICATION_SCHEMA,
    RestartTrajectoryReport, RollbackIdentityReport, StrategyAgreement, StrategyOutcome,
    StrategyWorkReport, ValidatedKrasisVerification, VerificationBinding, VerificationRefusal,
    check_cross_block_derivatives, check_event_state, check_event_state_from,
    check_history_and_rejection, check_restart_trajectory, check_rollback_identity,
    check_strategy_work,
};
