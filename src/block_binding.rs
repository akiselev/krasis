//! `SymbolId`-linked state blocks for a Finitum block-composed product space (E6).
//!
//! Adapts a Finitum `BlockLayout` (e.g. `finitum::MixedSpace::layout()`, the layout underneath
//! a saddle-point `finitum::MixedOperator`) into a Krasis [`StateLayout`] plus a total,
//! injective [`StateBinding`] from each field's Scientia `SymbolId` (mirrored by
//! [`crate::SemanticId`]) to its Krasis [`BlockId`] -- the GX-D1 [`StateBinding`] convention
//! ([`crate::initial_state_from`]) applied to a block-composed space with several fields
//! instead of one realization's single field.

use finitum::BlockLayout as FinitumBlockLayout;

use crate::{BlockId, KrasisError, SemanticId, StateBinding, StateBlock, StateLayout};

/// Builds a Krasis [`StateLayout`] and [`StateBinding`] from a Finitum product-space
/// [`FinitumBlockLayout`]: one Krasis block per Finitum field block, named `field_<variable>`
/// and bound to that field's system-level `SysVarId` (`FieldBlock::variable`, SC-W1: dense
/// across every instance; the identity map `SysVarId(symbol.0)` for a one-instance layout, so
/// nothing changes numerically there), in the same order Finitum declared them.
pub fn block_state_layout(
    layout: &FinitumBlockLayout,
) -> Result<(StateLayout, StateBinding), KrasisError> {
    let blocks = layout
        .blocks()
        .iter()
        .map(|block| {
            StateBlock::new(
                block_id(block.variable.0),
                block.offset..block.offset + block.extent,
            )
        })
        .collect();
    let state_layout = StateLayout::new(blocks)?;
    let bindings = layout
        .blocks()
        .iter()
        .map(|block| {
            (
                SemanticId::new(block.variable.0),
                block_id(block.variable.0),
            )
        })
        .collect();
    let state_binding = StateBinding::new(&state_layout, bindings)?;
    Ok((state_layout, state_binding))
}

fn block_id(variable: u32) -> BlockId {
    BlockId::new(format!("field_{variable}"))
}
