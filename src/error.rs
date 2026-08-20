use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum KrasisError {
    #[error("block `{0}` has zero width")]
    EmptyBlock(String),
    #[error("block `{block}` range {start}..{end} overlaps an existing block")]
    OverlappingBlock {
        block: String,
        start: usize,
        end: usize,
    },
    #[error("field `{0}` already exists")]
    DuplicateField(String),
    #[error("unknown field `{0}`")]
    UnknownField(String),
    #[error("field `{field}` has length {actual}, expected {expected}")]
    FieldLength {
        field: String,
        actual: usize,
        expected: usize,
    },
    #[error("a trial transaction is already active")]
    TrialAlreadyActive,
    #[error("no trial transaction is active")]
    NoActiveTrial,
    #[error("checkpoint layout digest `{actual}` does not match `{expected}`")]
    LayoutMismatch { actual: String, expected: String },
    #[error("checkpoint state is malformed: {0}")]
    MalformedCheckpoint(String),
}
