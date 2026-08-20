//! Coupled field state and transactional simulation runtime.

mod checkpoint;
mod error;
mod event;
mod layout;
mod state;

pub use checkpoint::Checkpoint;
pub use error::KrasisError;
pub use event::{EventDirection, EventRecord};
pub use layout::{Block, BlockId, BlockLayout};
pub use state::{ConstitutiveSlot, FieldId, SimulationState, TransactionPhase};
