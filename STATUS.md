# Krasis status

Updated: 2026-09-01
Milestone: SV0-B4 reusable coupled verification + GX-D1 initial-condition
projection + E6 transactional block-linear composition + W7 SC-W1 steady-runner
target (`BlockLinearExecution` content-addressed, CG/MINRES/GMRES)

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
- W7 / SC-W1 steady-runner target (this commit): `BlockLinearExecution` is now the entry point
  Sinbad's steady system runner reroutes through. `OperatorIdentity` (Krasis trait, implemented
  for `finitum::MixedOperator`, `SystemOperator`, `ReducedSystemOperator` from their own content
  digests plus the serialized `ConstraintSet`) folds the operator's numerical content into the
  `krasis-block-linear/2` checkpoint identity -- `/1` hashed shape only and could not tell two
  operators over one layout apart (STATUS Next item 1, closed). `solve` takes a
  `BlockLinearSolver::{ConjugateGradient, Minres, Gmres}` policy and returns a
  `BlockLinearReport { algorithm, report, restart_cycles }`; non-convergence is the typed
  `KrasisError::SolveDidNotConverge { algorithm, iterations }` after rollback; a nullspace
  projector with a non-MINRES algorithm is refused typed. `SimulationState::zeroed(layout,
  history)` is the zero initial guess a steady solve starts from. `tests/sc_w1_steady_reroute.rs`
  runs the 25-stokes corpus through exactly Sinbad's Finitum path (`SystemRealizationPlan` ->
  `bind_kernels` with the corpus `equation_sign` -> wall constraints -> `reduced` -> verified
  pressure nullspace -> `load_vector`) and proves the Krasis-transacted MINRES and GMRES
  solutions and traces are **bit-identical** to the direct Methodus calls (46 tests at this
  head).

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

W7 (2026-09-01, Rust 1.97.0), per commit; note that `cargo fmt --all` also formats local
path dependencies (sibling repositories), so this wave formats Krasis alone:

```text
cargo fmt -p krasis -- --check
cargo clippy --all-targets -- -D warnings
cargo test                                              # 46 passed, 0 failed (SC-W1 commit)
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps
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

1. ~~fold `finitum::MixedOperator::digest()` into `BlockLinearCheckpoint`'s
   operator identity~~ — landed in W7 (`OperatorIdentity`, `krasis-block-linear/2`);
2. block composition over the executable `SystemRealizationPlan` operators
   (finitum `739e2aa`+) as coupled/transient Stokes and the E7 trajectory
   adjoints demand it;
3. DAE index-1 consistent initialization beyond reduced-row, and coupled
   event persistence — still only from a concrete product case.
4. SC composition (design `sinbad/ARCHITECTURE.md` §8, §12; nothing landed).
   `SemanticId` keeps its `u32` type and comes to mirror Scientia's system-level
   `SysVarId` (a one-line C11.8 amendment). Prerequisite batch P: N-block DAE
   composition over `SystemRealizationPlan` with Newton inside BDF, so 08 can
   execute monolithically. SC-W1: `CoupledSystemOperator` implementing
   `NonlinearOperator`/`DaeOperator`/`BlockNonlinearOperator` over Finitum leaf
   actions and connection realizations (this is SV1-F1 / SV7-F2 / SV4-H1 — the
   IDs are kept), plus the separately gated reroute of Sinbad's steady system
   runner through `BlockLinearExecution` (25-stokes and 13-mixed-darcy must
   reproduce today's solutions within 1e-12 relative before the old path goes).
   The Krasis side of the reroute is landed (W7): Sinbad builds the reduced
   operator exactly as today, then `block_state_layout(reduced.operator()
   .layout())`, `SimulationState::zeroed(layout, history)`,
   `BlockLinearExecution::new(&reduced, state)`, and `solve(ctx, &rhs, None,
   projector, &BlockLinearSolver::Minres(cfg), 0.0)`; the report's `report` is
   the unchanged Methodus `LinearSolveReport`, `restart_cycles` the GMRES
   count, `algorithm.label()` the receipt string. The bitwise agreement on
   25-stokes is proven in `tests/sc_w1_steady_reroute.rs`; 13-mixed-darcy goes
   through the same call with `bind_kernels_with_facets` and is Sinbad's gate.
   `CrossDialectOperator` is retired once `CoupledSystemOperator` covers its FC10
   test. Partitioned-iteration state and output-based convergence evaluation live
   in the fixed-point transaction (SC-W3).
