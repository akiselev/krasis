# Krasis status

Updated: 2026-08-21
Milestone: FC7 coupled execution complete

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

## Boundary

Resolvent owns scientific/coupling meaning, Finitum owns concrete discrete operators,
Krasis owns stateful composition, and Solverang owns the algorithms acting on it.

## Validation

Passed on 2026-08-21 with Rust 1.97.0:

```text
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets           # 8 passed
```

## Next

FC8's mixed/facet/compatible artifact plans do not yet add global solved actions for Krasis.
Extend coupled layouts or constitutive updates only when a concrete stateful mixed realization
requires them.
