# Krasis status

Updated: 2026-10-19
Milestone: W12 L5 (SV2-C4) -- transactional explicit method-of-lines execution, on top of W10.
Built against Methodus `8bd8333` (W12 L2 explicit SSP-RK API), Finitum `16dbdf3`. Owner gate:
**101 tests across 15 targets** (`--lib` has no unit tests): coupled_execution 5,
coupled_finitum_system 8, coupled_system 13, e6_block_linear 8, fc7_coupled_leaf 1, gx_d1 15,
sc_w1_steady_reroute 4, sc_w3_partitioned 12, sv0_verification 6, transactions 5,
w8_typed_failures 6, w10_connected_composition 11 (the 94 W10 tests, unchanged) and
sv2c_explicit_execution 7 (new); fmt, all-target clippy `-D warnings`, rustdoc `-D warnings`
clean; numerical targets at `CARGO_PROFILE_TEST_OPT_LEVEL=2`. `Cargo.lock` gained only the
transitive entries the sibling path heads now require (cadabra-arrangements/-numeric/
-predicates, libm, slotmap); no Krasis dependency changed.

## SV2-C4 explicit execution (`src/explicit.rs`)

- `ExplicitExecution<Op: methodus::ExplicitOperator + OperatorIdentity>::new(operator, state,
  config: methodus::ExplicitConfig, attempt_budget, cx)`: committed step-0 state whose width is
  the operator dimension, budget >= 1, no NaN in the step policy; the initial state must be
  admissible (`KrasisError::InadmissibleExplicitState{index, reason}`). Later states load via
  `restore`.
- `advance(cx, final_time) -> &ExplicitStepRecord`: each attempt is `begin_trial` ->
  `methodus::explicit_step(.., retry_ceiling)` -> `Accepted`: `set_trial_vector` + `commit(t)`;
  `Rejected`: `rollback`, record an `ExplicitRejectionRecord`, retry at `suggested_step`.
  Typed, never retried: `ExplicitStepRejected{time, rejection}` (no suggestion, `Fixed`),
  `ExplicitStepCollapse{time, step, minimum_step}` (from `SolveError::StepSizeCollapse`),
  `ExplicitAttemptBudgetExhausted{time, attempts}`, `EvaluationRefused` (logged in
  `evaluation_refusals()` as `ExplicitRefusedAttempt`), other `SolveError`s as `Solve(String)`.
  Any error leaves committed state, explicit state and records unchanged.
  `advance_to(cx, t_out) -> &[ExplicitStepRecord]` steps until exactly `t_out` (the caller's
  output cadence; earlier `t_out` is `InvalidCommitTime`).
- Checkpoint `ExplicitCheckpoint{operator_identity, config_identity, state: Checkpoint,
  integrator: methodus::ExplicitState}`. `explicit_config_identity(cfg)` =
  `krasis-explicit-config/1:blake3:<hex>` over the serialized `ExplicitConfig` (scheme + step
  policy; the attempt budget is excluded -- it never changes an accepted state). `restore`
  refuses a different operator or configuration identity
  (`KrasisError::CheckpointIdentityMismatch{what, expected, actual}`) and a checkpoint whose two
  states disagree, before any change; it drops step/rejection records past the restored step
  (the refusal log is left alone, as in `CoupledExecution`).
- Telemetry `ExplicitStepRecord{accepted_step, start_time, end_time, attempts, bound_factor,
  report: methodus::ExplicitStepReport (verbatim: step, step_limit, scheme, ssp_coefficient,
  cfl_realized, balance_corrected, invariants), balances: [InvariantBalanceCheck{
  relative_imbalance: Option, bound, within_bound}]}`; bound `k·ε·scale`, `k = dimension +
  4·stages` (Methodus's a priori budget). `conservation_summary() -> ExplicitConservationSummary
  {steps, invariants, max_relative_imbalance, relative_bound = k·ε, within_bound,
  balance_corrected}` is what Sinbad maps to `DiscreteConservation`. No physics, no mesh.
- Proofs (`tests/sv2c_explicit_execution.rs`, periodic upwind test operator with positivity):
  FE at CFL 1.5 is rejected (`InadmissibleStage`, stage 1), rolled back and accepted at the
  suggested Δt/2, bit-equal to Methodus's direct step from the untouched state; budget 1 ->
  `ExplicitAttemptBudgetExhausted`, halved step below `min_step` -> `ExplicitStepCollapse`,
  `Fixed` rejection -> `ExplicitStepRejected`, each with a byte-identical checkpoint after; a
  typed refusal is logged once, not retried; checkpoint JSON round trip at t=0.5 replays to
  t=1.0 bitwise (fresh execution and the original one, records equal); other config / other
  operator identity / tampered state refused; conservative balances within bound for FE, SSP2,
  SSP3, SSP(10,4) (max relative imbalance ≈1e-15 vs bound 1.5-2.3e-14, 64 cells), leaking
  operator > 1e6 x bound.
- Open: coupled explicit groups, multirate / explicit-fluid-implicit-solid (SV7-F), IMEX;
  telemetry and refusal logs are in memory, not checkpointed; restored state's admissibility is
  not rechecked (the first stage check of the next step catches it).

## W10 summary (2026-10-09, Finitum `a75baeb` / Methodus `a9d97c8`; detail in git `caa9e15`)

- Weighted Jacobi through Finitum's `system_linearized_diagonal` for the one-relation pair shape
  (`ConnectedSystemOperator::has_connected_diagonal()`); sum-of-leaf convention kept only for
  the multi-relation union shape.
- Real matching-fixture partitioned agreement gate; `PartitionedExecution` checkpoints bound to
  operator identity; `NewtonConfig.acceleration` refused inside the transaction; divergence
  rounding floor; `SweepReport::state_change_norm`.
- Sweep-level acceleration through `methodus::accelerate_fixed_point`
  (`PartitionedAcceleration`, fixed relaxation / Aitken).
- Block coupling actions, per-block diagonals and `BlockGaussSeidelFactory` (steady view only;
  cannot serve a BDF stage solve).
- Open: N-way junctions and the multi-relation connected diagonal; reversed-port `new_system`
  pairs fall back; transient nonmatching composition (pending Finitum); exchange-based coupling
  action; an Aitken-breakdown fixture; `PartitionedFixedPoint` reports only `SolveReport`.
- Cross-repo (from W10): Sinbad `preconditioner = "block_gauss_seidel"` via
  `BlockGaussSeidelFactory::new(&operator, GaussSeidelSweep::{Forward,Symmetric})`; partitioned
  policy serializes `PartitionedConfig`; Methodus (optional) owning block Gauss-Seidel
  constructor and `AccelerationBreakdown` trace; Finitum multi-relation diagonal.
- History: SC-W3 (2026-09-18) matching multi-component DAE composition, consistent
  initialization and trace-row classification; SC-W2 (2026-09-17) first matching interface
  (`docs/validation/2026-09-17-sc-w2/`).

## Ownership

Krasis owns coupled field/material state, history, transaction boundaries, events,
checkpoints and composition into Methodus-facing problems. It implements Methodus traits
against Finitum operators. It does not own scientific parsing/form semantics, mesh/basis
assembly, kernel compilation or numerical solver algorithms.

## Implemented capabilities

- Deterministic contiguous layouts and semantic state bindings; committed/trial field and
  constitutive state with explicit begin/commit/rollback and bounded committed history.
- Canonical flattening, serializable checkpoints and atomic prevalidated restoration.
  Checkpoint identity binds layout, concrete operator/geometry/material content and BDF
  history, time, accepted-step identity and prior step sizes.
- Direct Methodus nonlinear, block nonlinear and DAE operator interfaces over Finitum
  realizations. BDF attempts enclose field and constitutive changes in one transaction.
- `CoupledSystemOperator` composes N `CoupledLeaf`s and state/rate residual coupling edges;
  dependency graphs expose Tarjan SCC stages in dependencies-first order. Names and semantic
  ids are unique across leaves. Same-family leaves and one-way dependencies are supported.
- `CoupledLeaf::reduced_system` wraps Finitum's rate-capable `ReducedSystemOperator`, including
  concrete layout and constraint identity. `with_binding` can re-key independently authored
  leaves to system-level semantic ids. These remain independent realization groups; a Finitum
  multi-instance group is not equivalent to an exchange edge.
- `CoupledExecution<Op>` shares Newton/BDF stepping, rollback and checkpoint paths between
  single and composed operators. `attempt_step_with` accepts Methodus nonlinear solver hooks,
  including Newton-Krylov or converged partitioned solves, inside the same transaction.
- Reduced-row consistent initialization assembles differential/algebraic masks across leaves
  and includes cross edges in the residual. `with_consistent_initialization_when_algebraic`
  skips the rate solve only when no row is algebraic; the explicit convention enters identity.
- Initial data projection through `StateBinding`/`NodalContext`, including one nodal context
  per reduced-system leaf. Blocks must be bound exactly once with correct component lengths.
  `initial_state_from` creates a fresh state, samples at that state's initial time (currently
  zero), and has no caller-supplied initial-time parameter. This migration preserves that API.
- `BlockLinearExecution` drives CG/MINRES/GMRES within trial/commit/rollback. Only a converged
  solve commits; invalid policies and cross-operator checkpoint restores refuse. Nullspace
  projection is admitted only with its supported solver convention.
- Identity-bearing floating-point state refuses NaN, infinities and negative zero before
  serialization. Checkpoint/report source validation covers nested fields, history,
  constitutive slots and exposed Finitum inputs/mesh/constraints.

## Verification and error contracts

- SV0-B4/FC7 reports bind exact operator/layout/checker identities and support source-aware
  recomputation: rollback identity, restart trajectory, isolated cross-block derivatives,
  counted strategy agreement/work, synchronized history and event-state disposition.
- `FinitumVerificationSource` binds concrete realization agreement or per-field nodal patch
  reports, including reduced-system leaves. Composed sources retain their constituent
  identities; a passing patch report is not a claim of full realization agreement.
- `krasis-verification/2` and `krasis-block-linear/2` incorporate numerical content rather
  than shape-only identity. Historical schema changes and old test counts remain in Git.
- W8 K1 preserves `NumericError::Evaluation { code, origin, message }` through nonlinear,
  linear, consistent-initialization and transactional boundaries. Callers receive
  `KrasisError::EvaluationRefused`; a typed evaluation refusal is not an ordinary rejected
  step and must not trigger a smaller-dt retry.
- `evaluation_refusals()` records typed attempted-step refusals. Rollback/history reports can
  carry an `EvaluationRefusal`; original producer code, origin and located message survive.
- `FieldSource::Fallible` initial data evaluates at each vertex and the new state's time;
  errors retain point/time with no invented cell identity. A successfully returned NaN still
  follows nonfinite-state rejection, separately from a callback returning a typed error.

## W8 F3 consumer migration

- Migrated 27 constructor/helper calls across seven integration test files:
  six `DynamicExternalInput::try_new`, six `SystemConstitutiveInput::try_new`, four
  `ExternalInput::try_sampled_at`, six `FieldSource::fallible`, four
  `essential_constraints_from_system_at` and one `essential_constraints_from_selected_at`.
- Callback values/tangents are unchanged and wrapped in `Ok`. Stored fixture sources and
  homogeneous constraints explicitly use their initial time `0.0`; runtime constitutive
  callbacks continue receiving Finitum's evaluation point and its actual time.
- Preserved the deliberate `Ok(vec![NaN])` initial-source test. Existing K1 tests still check
  typed provider refusal, original attribution, exact rollback, no retry and initial-time
  evaluation independently of nonfinite-value rejection.
- No retiring constructor/helper calls remain in Krasis source or tests. The final
  `FieldSource::Sampled` match arm and rustdoc link are removed together with Finitum F3.
- No numerical policy, coupling algorithm, public initial-state signature or execution
  identity convention changed in this migration.

## Prescribed-motion checkpoint identity (`39b98f3`)

- Reduced-system content identity consumes Finitum's `realization_digest()`, covering
  prescribed value/rate identity and target descriptors. Static identities keep their exact
  previous bytes; no solver or integration algorithm changed.
- The regression uses two motions with identical initial values but different analytic rates.
  Same-motion restore succeeds; different-motion restore refuses before changing the target
  state/checkpoint. The static identity format is asserted explicitly.
- Full `cargo test -q -p krasis`: all 69 tests passed, none ignored. Focused regression,
  clippy with warnings denied, rustdoc with warnings denied and scoped formatting passed.
  Finitum prescribed-motion prerequisite is committed at `e6d67ee`.

## Fallible patch verification

`FinitumVerificationSource::try_check_patch` delegates to Finitum's fallible nodal checker.
Callback failure retains producer code and origin/location text in `VerificationRefusal`.
The infallible checker wraps its callback in `Ok` and shares the same report path. Tests
compare full rollback report identity between both paths and ensure the first failed vertex
stops callback evaluation while preserving the producer refusal.

## Validation

- W12 L5 (2026-10-19): 101 tests across 15 targets against Finitum `16dbdf3` / Methodus
  `8bd8333`, none failed or ignored; fmt check, all-target clippy and rustdoc with warnings
  denied passed.
- W10 (2026-10-09): 93 tests across 14 targets against Finitum `a75baeb` / Methodus `a9d97c8`,
  none ignored; scoped formatting, all-target all-feature clippy with warnings denied, rustdoc
  with warnings denied and doctests passed (details in the 2026-10-09 entry).
- Before final F3 removal, full 70 tests passed against Finitum `5bd93cc`, none ignored.
  Focused patch forwarding, all-target clippy, rustdoc and scoped formatting passed.
- Final F3 gate against Finitum `23e9fd8`: all 70 tests passed, none failed or ignored;
  all-target clippy with warnings denied, rustdoc with warnings denied, scoped formatting
  and `git diff --check` passed.
- These are local consumer gates; downstream Sinbad integration remains independently gated.

## Current dependency/consumer boundary

- Finitum committed head `23e9fd8` includes F-EVAL, prescribed value/rate lifting and F3;
  fallible constructors and explicit-time `_at` helpers already exist.
- Methodus `4f52d38` supplies typed evaluation errors, candidate BDF rates and
  solver algorithms. Sinbad remains a downstream consumer undergoing W8 migration.
- A Krasis commit alone does not pin sibling path dependencies. Workspace integration
  snapshots record the full federation disposition.
- Sinbad must propagate `EvaluationRefused` directly from initial projection, initialization,
  stepping or steady solve; it must not match strings or retry it as solver nonconvergence.
- Finitum now has multi-instance realization and functional evaluation work. Consuming those
  in Sinbad does not require relocating scientific/kernel execution into Krasis.

## Remaining scope

- General DAE initialization beyond the supported reduced-row/index-1 contracts remains
  demand-driven. A barycenter P1 mass rule can be rank deficient; richer system quadrature
  and the explicit when-algebraic convention have distinct, recorded semantics.
- Coupled event records and the in-memory typed refusal log are not checkpointed. Concatenated
  event functions alone do not establish event persistence.
- Cross-instance physical interfaces, conservative nonmatching transfer and general coupled
  objective derivatives require their owning-layer implementation and end-to-end evidence;
  generic leaf composition alone establishes none of those scientific claims.

Final cross-repository evidence: [September 18 acceptance](../sinbad/docs/validation/2026-09-18-assembly/README.md).
