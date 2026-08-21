//! Coupled field state and transactional simulation runtime.

mod checkpoint;
mod coupled;
mod error;
mod event;
mod layout;
mod method;
mod state;

pub use checkpoint::Checkpoint;
pub use coupled::{CoupledCheckpoint, CoupledExecution, CoupledOperator};
pub use error::KrasisError;
pub use event::{EventDirection, EventRecord};
pub use layout::{BlockId, StateBlock, StateLayout};
pub use method::CrossDialectOperator;
pub use state::{ConstitutiveSlot, FieldId, SimulationState, TransactionPhase};
