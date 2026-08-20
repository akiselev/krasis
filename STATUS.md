# Krasis status

Updated: 2026-08-20
Milestone: clean repository bootstrap

## Implemented

- deterministic, contiguous state layout with identity and schema validation;
- committed/trial field state with explicit begin, commit, and rollback;
- layout-bound fields and bounded committed-state history;
- constitutive state slots participating in the same transaction;
- serializable checkpoints with history and atomic, prevalidated restoration;
- event records separated from state identity.

## Boundary

Resolvent owns scientific/coupling meaning, Finitum owns concrete discrete operators,
Krasis owns stateful composition, and Solverang owns the algorithms acting on it.

## Validation

Passed on 2026-08-20 with Rust 1.97.0:

```text
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets           # 4 passed
```

## Next

Connect generated Poisson realization to this transactional state, then add transient
diffusion state and DAE behavior. Introduce Solverang implementations only with the real
stateful residual/JVP composition they represent.
