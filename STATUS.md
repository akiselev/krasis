# Krasis status

Updated: 2026-08-21
Milestone: R1 Scientia consumer migration

## Implemented

- deterministic, contiguous state layout with identity and schema validation;
- committed/trial field state with explicit begin, commit, and rollback;
- layout-bound fields and bounded committed-state history;
- constitutive state slots participating in the same transaction;
- serializable checkpoints with history and atomic, prevalidated restoration;
- event records separated from state identity.
- canonical flattening of committed/trial/history fields in state-layout order;
- direct Methodus `NonlinearOperator`, `BlockNonlinearOperator`, and `DaeOperator`
  implementations over a real Finitum realization;
- BDF attempts enclosed by field and constitutive trial/commit/rollback transactions;
- coupled checkpoints that atomically validate and restore Krasis state plus Methodus BDF
  values, step-size history, time, and accepted-step identity;
- checkpoint operator identities incorporating Finitum's concrete plan digest and the Krasis
  state layout, with same-size geometry and dynamic-material mismatch refusals.
- `CrossDialectOperator` composes distinct Finitum `DiscreteOperator` families with explicit,
  finite, bidirectional off-diagonal matrices; it implements Methodus DAE, nonlinear, and block
  contracts and rejects same-family or one-way placeholder configurations.

## Boundary

Scientia owns scientific/coupling meaning, Finitum owns concrete discrete operators,
Krasis owns stateful composition, and Methodus owns the algorithms acting on it.

The numerical dependency moved directly from Solverang to Methodus at Methodus
`d5354abb4dfd197ba5fd66f3742f9820701e4c43`; Krasis has no dependency on the
generalized Solverang constraint engine.

The cross-dialect and serialized-checkpoint contracts were validated against Scientia
`215433962c874dfd86b59ffc6d69f017bba2b95a` and Finitum
`bbc242af14672229294dfb80e48941ba9e6b1ee6`.

## Validation

Passed on 2026-08-21 with Rust 1.97.0:

```text
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --all-targets           # 12 passed, 0 failed
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
cargo test --locked --workspace --doc
git diff --check
```

## Next

Extend coupling effects or transactional state only from a concrete product case; preserve the
validated checkpoint identity and atomic restore contract.
