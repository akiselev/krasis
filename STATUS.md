# Krasis status

Updated: 2026-08-20
Milestone: clean repository bootstrap

## Implemented

- deterministic coupled block layout with overlap validation;
- committed/trial field state with explicit begin, commit, and rollback;
- bounded committed-state history;
- constitutive state slots participating in the same transaction;
- serializable checkpoints and restoration validation;
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
cargo test                         # 1 passed
```

## Next

Wrap one Finitum Poisson operator in Solverang's operator traits, then add transient state
for diffusion before introducing genuinely coupled blocks.
