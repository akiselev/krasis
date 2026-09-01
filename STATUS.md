# Krasis status

Updated: 2026-09-01
Milestone: SV0-B4 reusable coupled verification + GX-D1 initial-condition
projection + E6 transactional block-linear composition

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
- serializable SV0-B4 reports/refusals for byte-exact rollback, checkpoint/restart trajectory
  identity, isolated cross-block derivatives, counted block-strategy agreement and work budgets,
  bounded-history synchronization, and event-state disposition;
- every SV0-B4 report binds operator, state/block-layout, and checker-configuration identities;
  the identity includes the complete Methodus evaluation context, and every versioned report has
  a canonical digest plus a source-aware recomputation validator;
- coupled-execution reports additionally bind a compatible Finitum-owned realization-agreement or
  nodal-patch report to the exact realization identity, recompute it against that realization or
  mesh, and retain its typed header and acceptance;
- event evidence is currently scoped to a caller-supplied Methodus `DaeOperator`; Krasis does not
  yet claim `CoupledExecution` event-record persistence.
- identity-bearing floating-point inputs refuse NaN, either infinity, and negative zero before
  serialization; positive zero is canonical, and solver-error disposition uses finite invalid
  policy rather than a nonfinite sentinel.
- coupled source validation recursively covers checkpoint fields/history/constitutive state,
  Methodus integrator history, and exposed Finitum mesh/element/constraint/stored-input values
  before any checkpoint or report identity is hashed.
- GX-D1 (`e4478f5`): initial-condition projection from Finitum `FieldSource`s onto the DOF map,
  `SymbolId`-linked state blocks (`StateBinding`/`NodalContext`), and reduced-row consistent
  initialization; the authoritative record is `sinbad/docs/simulation-vision/GX-CONTRACTS.md`
  C11.8.
- E6 block-linear composition (`91dfd25`): `block_state_layout` adapts a Finitum product-space
  `BlockLayout` into `StateLayout`/`StateBinding` (the GX-D1 `SymbolId` convention generalized to
  N block-composed fields, no scientia dependency); `BlockLinearExecution`/`BlockLinearCheckpoint`
  drive `methodus::solve_minres` inside the existing trial/commit/rollback transaction — only a
  converged solve ever commits; typed refusals for dimension/block/length mismatches and
  cross-operator checkpoint restores; six acceptance tests compose finitum's real
  vector-P2/scalar-P1 `MixedOperator` saddle-point fixture with rollback/restart/history
  evidence (40 tests total at that head).

## Boundary

Scientia owns scientific/coupling meaning, Finitum owns concrete discrete operators,
Krasis owns stateful composition, and Methodus owns the algorithms acting on it.

The numerical dependency moved directly from Solverang to Methodus; SV0-B4 is validated against
Methodus `b1b10c9f9ff682e562408c5080ad408a9c37a594`. Krasis has no dependency on the
generalized Solverang constraint engine.

The cross-dialect and serialized-checkpoint contracts were validated against Scientia
`0f8d7d65f78d9215385f4912ba79c1be1d979d70`, Finitum SV0-B3
`13f14c0427d3b839e777e9f57086f37ef558592b`, and Resolvent
`5e106e780e44926f8236288d2f76e48dc0283aa9`.

## Validation

Passed on 2026-08-21 with Rust 1.97.0:

```text
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace --all-targets           # 20 passed, 0 failed
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
RUSTDOCFLAGS='-D warnings' cargo test --locked --workspace --doc
git diff --check
```

## Known limits recorded by the 2026-08-30 workspace audit (tree `bb2abe4`)

The first two bullets describe the pre-GX-D1 state and are superseded by
`e4478f5` (C11.8); the third's consistent-initialization half is superseded
(reduced-row initialization landed), while event persistence remains open.

- There is no initial-condition concept: an IC is a caller-assembled
  `Vec<f64>` passed to `insert_field`, with no spatial evaluator or
  projection onto the DOF map.
- State blocks bind to Finitum realizations by `BlockId` name and position,
  not by Scientia `SymbolId`.
- `make_initial_state_consistent` is the default no-op; no DAE consistent
  initialization is performed. `CoupledExecution` does not persist events.

## Next

GX-D1 and the E6 block-linear composition are landed (`e4478f5`, `91dfd25`).
Demand-pulled next work (workspace `PLAN.md` §6):

1. fold `finitum::MixedOperator::digest()` (finitum `96edb6d`) into
   `BlockLinearCheckpoint`'s operator identity — content-addressed instead of
   shape-only — once Krasis binds a `MixedOperator` concretely;
2. block composition over the executable `SystemRealizationPlan` operators
   (finitum `739e2aa`+) as coupled/transient Stokes and the E7 trajectory
   adjoints demand it;
3. DAE index-1 consistent initialization beyond reduced-row, and coupled
   event persistence — still only from a concrete product case.
