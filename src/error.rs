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
    #[error("coupled operator configuration is invalid: {0}")]
    InvalidCoupling(String),
    #[error("coupled numerical solve failed: {0}")]
    Solve(String),
    #[error(
        "{algorithm} did not converge within the configured tolerance after {iterations} iterations"
    )]
    SolveDidNotConverge {
        algorithm: String,
        iterations: usize,
    },
    #[error("nodal context has no vertex coordinates")]
    EmptyNodalContext,
    #[error(
        "nodal coordinate {index} has dimension {actual}, expected {expected} to match every other vertex"
    )]
    InconsistentNodalCoordinates {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("initial-condition bindings are missing block `{0}`")]
    InitialBlockMissing(String),
    #[error("initial-condition bindings name unknown block `{0}`")]
    InitialBlockUnknown(String),
    #[error("initial-condition bindings name block `{0}` more than once")]
    InitialBlockDuplicate(String),
    #[error(
        "block `{block}` has width {width}, which is not a whole number of components over {vertex_count} vertices"
    )]
    InitialDimensionMismatch {
        block: String,
        width: usize,
        vertex_count: usize,
    },
    #[error("block `{0}` binds a field source that cannot be evaluated pointwise")]
    InitialSourceNotPointwise(String),
    #[error("state binding names unknown block `{0}`")]
    StateBindingUnknownBlock(String),
    #[error("state binding names block `{0}` more than once")]
    StateBindingDuplicateBlock(String),
    #[error("state binding names semantic id `{0}` more than once")]
    StateBindingDuplicateSemanticId(u32),
    #[error("state binding does not cover every block in the state layout")]
    StateBindingIncomplete,
    #[error("state binding blocks do not match the coupled operator's state layout")]
    StateBindingLayoutMismatch,
    #[error(
        "consistent-initialization mask has {actual} rows, expected {expected} to match the operator dimension"
    )]
    ConsistentInitializationMaskLength { actual: usize, expected: usize },
}
