//! Additive binding of state blocks to opaque semantic identities.
//!
//! Krasis has no dependency on Scientia. [`SemanticId`] mirrors the wire representation of
//! Scientia's **system-level** variable id (`SysVarId`, a dense `u32` allocated by system
//! elaboration across every instance of every model; `sinbad/ARCHITECTURE.md` §2.3-2.4,
//! GX-CONTRACTS C11.8 as amended by C12) so a caller holding one can bind state blocks by
//! semantic identity without Krasis depending on Scientia's crate: construct the matching
//! [`SemanticId`] with [`SemanticId::new`], passing the id's underlying `u32`. Krasis reads it
//! from Finitum's re-keyed `BlockLayout` ([`crate::block_state_layout`]), never from Scientia.
//!
//! The system-level meaning is what makes cross-plan composition sound: two instances of one
//! model share every per-model `SymbolId`, so binding by `SymbolId` would collide the moment a
//! second instance appeared, whereas `SysVarId`s are dense and unique across the whole system.
//! [`crate::CoupledSystemOperator`] therefore refuses a semantic id shared by two leaves. For a
//! single-model system (the implicit one-instance system) the two ids coincide numerically, so
//! every existing binding keeps its value.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{BlockId, KrasisError, StateLayout};

/// Opaque semantic identity for a state block, convention-compatible with Scientia's
/// system-level `SysVarId` (numerically equal to the per-model `SymbolId` for a one-instance
/// system).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SemanticId(u32);

impl SemanticId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Total, injective binding from [`SemanticId`]s to a [`StateLayout`]'s blocks.
///
/// Every block in the layout is bound to exactly one semantic id, and every semantic id names
/// exactly one block: a consumer can therefore resolve either direction unambiguously.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateBinding {
    bindings: BTreeMap<SemanticId, BlockId>,
    identity: String,
}

impl StateBinding {
    pub fn new(
        layout: &StateLayout,
        bindings: Vec<(SemanticId, BlockId)>,
    ) -> Result<Self, KrasisError> {
        let mut map = BTreeMap::new();
        let mut used_blocks = BTreeSet::new();
        for (semantic, block) in bindings {
            if layout.block(&block).is_none() {
                return Err(KrasisError::StateBindingUnknownBlock(block.to_string()));
            }
            if !used_blocks.insert(block.clone()) {
                return Err(KrasisError::StateBindingDuplicateBlock(block.to_string()));
            }
            if map.insert(semantic, block).is_some() {
                return Err(KrasisError::StateBindingDuplicateSemanticId(
                    semantic.as_u32(),
                ));
            }
        }
        if map.len() != layout.blocks().len() {
            return Err(KrasisError::StateBindingIncomplete);
        }
        let identity = bindings_identity(&map);
        Ok(Self {
            bindings: map,
            identity,
        })
    }

    pub fn block_for(&self, semantic: SemanticId) -> Option<&BlockId> {
        self.bindings.get(&semantic)
    }

    pub fn semantic_for(&self, block: &BlockId) -> Option<SemanticId> {
        self.bindings
            .iter()
            .find(|(_, candidate)| *candidate == block)
            .map(|(semantic, _)| *semantic)
    }

    /// Every block named by this binding, in semantic-id order.
    pub fn blocks(&self) -> impl Iterator<Item = &BlockId> {
        self.bindings.values()
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
}

fn bindings_identity(bindings: &BTreeMap<SemanticId, BlockId>) -> String {
    bindings
        .iter()
        .map(|(semantic, block)| {
            format!("{}:{}:{};", semantic.as_u32(), block.as_str().len(), block)
        })
        .collect::<Vec<_>>()
        .join("")
}
