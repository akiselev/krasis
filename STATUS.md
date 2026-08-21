# Krasis status

Updated: 2026-08-21
Milestone: FC11 restart serialization validated

## Implemented

- deterministic, contiguous state layout with identity and schema validation;
- committed/trial field state with explicit begin, commit, and rollback;
- layout-bound fields and bounded committed-state history;
- constitutive state slots participating in the same transaction;
- serializable checkpoints with history and atomic, prevalidated restoration;
- event records separated from state identity.
- canonical flattening of committed/trial/history fields in state-layout order;
- direct Solverang `NonlinearOperator`, `BlockNonlinearOperator`, and `DaeOperator`
  implementations over a real Finitum realization;
- BDF attempts enclosed by field and constitutive trial/commit/rollback transactions;
- coupled checkpoints that atomically validate and restore Krasis state plus Solverang BDF
  values, step-size history, time, and accepted-step identity;
- checkpoint operator identities incorporating Finitum's concrete plan digest and the Krasis
  state layout, with same-size geometry and dynamic-material mismatch refusals.
- `CrossDialectOperator` composes distinct Finitum `DiscreteOperator` families with explicit,
  finite, bidirectional off-diagonal matrices; it implements Solverang DAE, nonlinear, and block
  contracts and rejects same-family or one-way placeholder configurations.

## Boundary

Resolvent owns scientific/coupling meaning, Finitum owns concrete discrete operators,
Krasis owns stateful composition, and Solverang owns the algorithms acting on it.

The cross-dialect and serialized-checkpoint contracts were validated against Resolvent
`57c9b431e77a91d27fe20c4ca206e8b55c3e4cd7` and Finitum
`a39df632b90ceedf779bcceaf7f146433615d743`.

## Validation

Passed on 2026-08-21 with Rust 1.97.0:

```text
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets           # 12 passed, 0 failed
```

## Next

Extend coupling effects or transactional state only from a concrete product case; preserve the
validated checkpoint identity and atomic restore contract.
