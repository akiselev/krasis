//! Coupled field state and transactional simulation runtime.

mod binding;
mod checkpoint;
mod coupled;
mod error;
mod event;
mod initial;
mod layout;
mod method;
mod state;
mod verification;

pub use binding::{SemanticId, StateBinding};
pub use checkpoint::Checkpoint;
pub use coupled::{CoupledCheckpoint, CoupledExecution, CoupledOperator, RowKind};
pub use error::KrasisError;
pub use event::{EventDirection, EventRecord};
pub use initial::{NodalContext, initial_state_from};
pub use layout::{BlockId, StateBlock, StateLayout};
pub use method::CrossDialectOperator;
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
