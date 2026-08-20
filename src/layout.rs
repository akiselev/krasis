use crate::KrasisError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(String);

impl BlockId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StateBlock {
    id: BlockId,
    range: Range<usize>,
}

impl StateBlock {
    pub fn new(id: BlockId, range: Range<usize>) -> Self {
        Self { id, range }
    }

    pub fn id(&self) -> &BlockId {
        &self.id
    }

    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StateLayout {
    blocks: Vec<StateBlock>,
    width: usize,
    identity: String,
}

impl StateLayout {
    pub fn new(mut blocks: Vec<StateBlock>) -> Result<Self, KrasisError> {
        if blocks.is_empty() {
            return Err(KrasisError::EmptyLayout);
        }
        blocks.sort_by_key(|block| block.range.start);
        let mut previous_end = 0;
        let mut ids = HashSet::new();
        for block in &blocks {
            if block.id.as_str().trim().is_empty() {
                return Err(KrasisError::EmptyBlockId);
            }
            if block.range.is_empty() {
                return Err(KrasisError::EmptyBlock(block.id.to_string()));
            }
            if !ids.insert(block.id.clone()) {
                return Err(KrasisError::DuplicateBlock(block.id.to_string()));
            }
            if block.range.start < previous_end {
                return Err(KrasisError::OverlappingBlock {
                    block: block.id.to_string(),
                    start: block.range.start,
                    end: block.range.end,
                });
            }
            if block.range.start > previous_end {
                return Err(KrasisError::BlockGap {
                    expected: previous_end,
                    actual: block.range.start,
                });
            }
            previous_end = block.range.end;
        }
        let width = blocks.last().map_or(0, |block| block.range.end);
        let identity = blocks
            .iter()
            .map(|block| {
                format!(
                    "{}:{}:{}:{};",
                    block.id.as_str().len(),
                    block.id,
                    block.range.start,
                    block.range.end
                )
            })
            .collect::<Vec<_>>()
            .join("");
        Ok(Self {
            blocks,
            width,
            identity,
        })
    }

    pub fn blocks(&self) -> &[StateBlock] {
        &self.blocks
    }

    pub fn block(&self, id: &BlockId) -> Option<&StateBlock> {
        self.blocks.iter().find(|block| &block.id == id)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
}
