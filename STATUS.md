# Krasis status

Updated: 2026-09-03
Milestone: SV0-B4 reusable coupled verification + GX-D1 initial-condition
projection + E6 transactional block-linear composition + W7 batch P / SC-W1
(`CoupledSystemOperator` N-leaf DAE composition with Newton inside BDF, Newton-Krylov and
partitioned solver hooks, Finitum `ReducedSystemOperator` leaves, the SV7-F2 coupling graph,
content-addressed `BlockLinearExecution` steady-runner target)

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
- ~~`CrossDialectOperator`~~ retired in W7: `CoupledSystemOperator` covers its FC10 test
  (`tests/coupled_system.rs`, `fc10_*`: the same residual values, JVP check, canonical identity
  over the cross matrices, and Methodus BDF advance). Its two refusals -- same family, one-way
  coupling -- are deliberately dropped: two leaves of one family across plans and one-way (DAG)
  coupling are both legitimate compositions now, and the graph reports them.
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
- W7 batch P + SC-W1 skeleton (this commit): `CoupledSystemOperator` composes N `CoupledLeaf`s
  (each any `methodus::DaeOperator` -- a Finitum realization through `CoupledOperator`, a
  Finitum `DiscreteOperator`, later Finitum's state-dependent `SystemOperator` -- with its own
  `StateLayout`, a `StateBinding` to system-level `SemanticId`s, an opaque content identity and
  optional differential/algebraic row kinds) with `CouplingEdge`s (a `methodus::LinearOperator`
  from one leaf's `State` or `Rate` into another leaf's residual, plus an opaque relation
  identity; `CouplingEdge::matrix` derives it from a `CsrMatrix`'s content). The composed
  operator implements `DaeOperator` (leaf residuals/JVPs plus edge actions on sub-slices;
  events concatenated), `NonlinearOperator` (steady view `t = 0`, `ydot = 0`) and
  `BlockNonlinearOperator` (one Methodus block per leaf, so `solve_blocks` partitions by
  realization group); `with_consistent_initialization(newton)` composes the leaves' row kinds
  into one mask solved over the composed residual (cross edges take part); identity
  `krasis-coupled-system/1` is canonical over leaf names/identities/layouts/bindings/masks and
  sorted edges. `CouplingGraph` (SV7-F2) exposes nodes, typed dependencies and Tarjan SCC
  stages in dependencies-first order (`is_acyclic`). `CoupledExecution<Op = CoupledOperator>`
  is now generic over a `TransactionalOperator` (`CoupledOperator`, `CoupledSystemOperator`),
  so Newton-inside-BDF (`methodus::bdf_step`), checkpoints binding Krasis state to Methodus BDF
  history, restore refusals and rollback of failed steps are the same code for one realization
  and for an N-leaf composition; `CoupledOperator` gains `state_layout_identity()` and
  `row_kinds()`, and the consistent-rate solve is one shared function. `SemanticId` now
  documents the `SysVarId` meaning (C11.8 as amended by C12); `CoupledSystemOperator` refuses a
  semantic id or block id shared by two leaves. `tests/coupled_system.rs` (12 tests):
  composed residual = leaf residuals + edge actions and `verify_dae_jvp`; BDF1/BDF2 through the
  transaction reproduce a manufactured solution with measured halving ratios 2 and 4; bitwise
  checkpoint/restart replay plus cross-identity and failed-Newton rollback refusals; consistent
  initialization over the composed residual; monolithic Newton vs Gauss-Seidel vs Jacobi
  agreement (and the SV0-B4 `check_strategy_work` report) on a nontrivial two-mesh Dirichlet
  diffusion pair; graph stages for DAG/cycle/mixed shapes and a one-sweep exactness check of the
  schedule; refusals; canonical identity; the FC10 port. 54 tests at this head.
- W7 solver hook (this commit): `CoupledExecution::attempt_step_with(ctx, step, config, &dyn
  methodus::NonlinearSolver)` runs Methodus `bdf_step_with` inside the same trial/commit/rollback
  transaction as `attempt_step` (one shared `transact_step`), so a matrix-free
  `NewtonKrylovSolver` (GMRES over the step's Jacobian action) or a partitioned `BlockNewton`
  (`solve_blocks` Gauss-Seidel/Jacobi over `block_layout()` inside the step) drives the same
  `CoupledSystemOperator`; `config.newton` is not consulted on that path. Test: both reproduce the
  dense-Newton BDF2 trajectory of the composed network to 1e-10 and the manufactured solution to
  the same discretization error, and a hopeless solver rolls the transaction back to the exact
  prior checkpoint. 55 tests at this head.
- W7 batch P item 3 closed against Finitum `4a6fe65` (this commit):
  `CoupledLeaf::reduced_system(name, finitum::ReducedSystemOperator)` is the leaf over Finitum's
  state-dependent, rate-capable system operator (`SystemRealizationPlan` -> `bind_kernels` ->
  `reduced`), which implements `methodus::DaeOperator` itself; the leaf's layout is
  `block_state_layout`'s with block ids prefixed `<name>/` (two Finitum leaves share one composed
  layout), its identity is `OperatorIdentity::content_identity` (`finitum-reduced-system:` plan
  digest + constraint set), and no row-kind mask is assumed (`with_row_kinds`). `CoupledLeaf::
  with_binding` re-keys a leaf to system-level ids: a second instance of one model repeats its
  per-model `SymbolId`, which `CoupledSystemOperator::new` refuses until it is rebound (the
  `SysVarId` re-key is Finitum's SC-W1 item; until then the caller supplies the offset).
  `tests/coupled_finitum_system.rs` (3 tests): two instances of a transient heat model on two
  meshes with a one-way sampled exchange edge -- composed residual = Finitum leaf residuals +
  edge and `verify_dae_jvp`; five BDF1 steps through `CoupledExecution<CoupledSystemOperator>`
  reproduce each instance's standalone Methodus trajectory when uncoupled (1e-10), heat the cold
  instance when coupled with walls still eliminated, agree between the dense and Newton-Krylov
  hooks (1e-9), and refuse a checkpoint restore across edge sets; binding/duplicate-id
  refusals. 58 tests at this head.

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
cargo test                                              # 46 passed (SC-W1 commit); 54 passed (batch P commit); 55 passed (solver hook); 58 passed (Finitum leaf)
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

W7 landed (this head): the content-addressed `BlockLinearExecution` steady-runner target
(SC-W1, Krasis side), `CoupledSystemOperator` with the coupling graph and Newton inside BDF
(batch P item 3 / SV1-F1, SV7-F2, SV4-H1 skeleton), `CrossDialectOperator` retired,
`SemanticId` = `SysVarId` convention. Honestly open, with what each waits on:

1. **Transient composition over `SystemRealizationPlan`** -- landed on the Krasis side
   (`CoupledLeaf::reduced_system`, Finitum `4a6fe65`). What remains is Sinbad's: 08 executing
   monolithically from a case file through `CoupledExecution<CoupledSystemOperator>` (batch P
   item 4), and the per-instance `SysVarId` re-key in Finitum's `BlockLayout` so the caller no
   longer has to `with_binding` a second instance of one model.
2. **Newton-Krylov inside BDF** -- landed (Methodus `bf9082f` `bdf_step_with` + `NonlinearSolver`;
   Krasis `attempt_step_with`). Krasis still carries no Newton loop of its own; `BdfState`/
   `StepOutcome` are unchanged so checkpoints keep their shape.
3. **Fixed-point transaction (SC-W3).** The coupling graph and the per-leaf block layout are
   the inputs; partitioned-iteration state, output-based convergence evaluation and
   acceleration wait on Methodus's acceleration hooks (SV7-F3) and on item 2. Not started.
4. **Steady runner reroute in Sinbad (SC-W1 gate).** Krasis side landed; the 25-stokes bitwise
   agreement is proven here (`tests/sc_w1_steady_reroute.rs`); 13-mixed-darcy goes through the
   identical call with `bind_kernels_with_facets` and is Sinbad's gate. API for Sinbad:
   `block_state_layout(reduced.operator().layout())`, `SimulationState::zeroed(layout, history)`,
   `BlockLinearExecution::new(&reduced, state)`, `solve(ctx, &rhs, None, projector,
   &BlockLinearSolver::{ConjugateGradient|Minres|Gmres}(cfg), 0.0)` returning
   `BlockLinearReport { algorithm, report, restart_cycles }` (non-convergence:
   `KrasisError::SolveDidNotConverge { algorithm, iterations }` after rollback).
5. **Upstream (Finitum, still open, C11.8):** the P1 single-point quadrature makes the
   consistent mass matrix rank-deficient; a pure reaction leaf's BDF Newton system is singular
   on its own. The Finitum heat leaves in `tests/coupled_finitum_system.rs` run BDF (mass +
   stiffness is nonsingular) without `with_consistent_initialization`, whose consistent-rate
   solve is over the mass block alone and was not exercised on them; the composed
   consistent-initialization test stays on the network DAE.
6. DAE index-1 consistent initialization beyond reduced-row, and coupled event persistence --
   still only from a concrete product case. `CoupledSystemOperator` concatenates leaf events
   (`event_count`/`event_values`) but `CoupledExecution` still does not persist event records.
