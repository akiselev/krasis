use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum KrasisError {
    #[error("a coupled state layout must contain at least one block")]
    EmptyLayout,
    #[error("state block id must not be empty")]
    EmptyBlockId,
    #[error("block `{0}` has zero width")]
    EmptyBlock(String),
    #[error("block `{0}` is declared more than once")]
    DuplicateBlock(String),
    #[error("block layout has a gap: expected the next block at {expected}, found {actual}")]
    BlockGap { expected: usize, actual: usize },
    #[error("block `{block}` range {start}..{end} overlaps an existing block")]
    OverlappingBlock {
        block: String,
        start: usize,
        end: usize,
    },
    #[error("field `{0}` already exists")]
    DuplicateField(String),
    #[error("constitutive slot `{0}` already exists")]
    DuplicateConstitutive(String),
    #[error("unknown field `{0}`")]
    UnknownField(String),
    #[error("unknown constitutive slot `{0}`")]
    UnknownConstitutive(String),
    #[error("field `{0}` has no block in the state layout")]
    FieldOutsideLayout(String),
    #[error("state layout field `{0}` has not been initialized")]
    MissingField(String),
    #[error("field `{field}` has length {actual}, expected {expected}")]
    FieldLength {
        field: String,
        actual: usize,
        expected: usize,
    },
    #[error("{label} has a non-finite value at index {index}")]
    NonFiniteValue { label: String, index: usize },
    #[error("a trial transaction is already active")]
    TrialAlreadyActive,
    #[error("no trial transaction is active")]
    NoActiveTrial,
    #[error("state structure cannot change during an active trial")]
    StructureDuringTrial,
    #[error("commit time {time} must be finite and no earlier than {current}")]
    InvalidCommitTime { time: f64, current: f64 },
    #[error("checkpoint layout digest `{actual}` does not match `{expected}`")]
    LayoutMismatch { actual: String, expected: String },
    #[error("checkpoint state is malformed: {0}")]
    MalformedCheckpoint(String),
}
