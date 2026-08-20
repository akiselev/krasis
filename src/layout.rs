use crate::KrasisError;
use serde::{Deserialize, Serialize};
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    pub range: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockLayout {
    blocks: Vec<Block>,
    width: usize,
    identity: String,
}

impl BlockLayout {
    pub fn new(mut blocks: Vec<Block>) -> Result<Self, KrasisError> {
        blocks.sort_by_key(|block| block.range.start);
        let mut previous_end = 0;
        for block in &blocks {
            if block.range.is_empty() {
                return Err(KrasisError::EmptyBlock(block.id.to_string()));
            }
            if block.range.start < previous_end {
                return Err(KrasisError::OverlappingBlock {
                    block: block.id.to_string(),
                    start: block.range.start,
                    end: block.range.end,
                });
            }
            previous_end = block.range.end;
        }
        let width = blocks.last().map_or(0, |block| block.range.end);
        let identity = blocks
            .iter()
            .map(|block| format!("{}:{}..{}", block.id, block.range.start, block.range.end))
            .collect::<Vec<_>>()
            .join("|");
        Ok(Self {
            blocks,
            width,
            identity,
        })
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
}
