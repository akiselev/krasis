# Krasis status

2026-09-29 SC-W3 package 1/2 landed (partitioned fixed-point transaction):
`krasis::partitioned` adds a partitioned decomposition over any `methodus::BlockLayout` (serial
Gauss-Seidel / parallel Jacobi schedule), usable as a standalone steady transaction
(`PartitionedExecution`, trial/commit/rollback like `BlockLinearExecution`) or plugged into
`CoupledExecution::attempt_step_with` as a `methodus::NonlinearSolver`
(`PartitionedFixedPoint`), so it shares the existing BDF transaction and rollback path with
`BlockNewton`/`NewtonKrylovSolver`. `ConnectedSystemOperator` gained `BlockNonlinearOperator`
(forwarding to its inner `CoupledSystemOperator`'s leaf-range layout, valid because elimination
preserves dimension and index), so both `ConnectedSystemOperator` and `CoupledSystemOperator`
admit the same `partitioned` decomposition per the brief. Existing 70 owner tests are unchanged;
4 new tests in `tests/sc_w3_partitioned.rs` bring the gate to 74 tests across 12 targets, all
passing with formatting, `cargo check`, strict all-feature clippy, rustdoc and doctests clean.

Ownership split: Methodus's existing `solve_blocks(.., max_iterations = 1, ..)` is called once
per sweep (the per-block Newton-with-backtracking algorithm Methodus already owns); Krasis owns
the outer sweep loop, an **output-based** convergence test (the sup-norm change of the
schedule-updated state between sweeps -- the exchanged interface data, since no finer per-DOF
trace decomposition exists at this composition level), fixed relaxation of that exchanged data
(a vector combination, not a solver algorithm), and two typed refusals recorded as
`NumericError::Evaluation` (surfacing as `KrasisError::EvaluationRefused`, never a silently
accepted iterate): `PARTITIONED_DIVERGED` (a sweep's interface change grows beyond the declared
`divergence_growth` factor relative to the previous sweep's) and `PARTITIONED_MAX_SWEEPS` (an
implicit iteration exhausts its bound without reaching its declared tolerance). Package 2's
agreement gate (`tests/sc_w3_partitioned.rs`): a steady two-leaf conduction fixture's serial and
parallel partitioned solves agree with monolithic Newton within 1e-8; a transient DAE/BDF fixture
plugged in as the BDF solver agrees with the dense-Newton trajectory within 1e-9 per accepted
step; a wrong-role (positive-feedback) coupling run under `iteration = implicit` is refused
typed (`PARTITIONED_DIVERGED` or `PARTITIONED_MAX_SWEEPS`, both observed depending on the
fixture) rather than committed, and a refused solve leaves the committed state bit-identical to
before the attempt.

Recorded findings and open items:
- **A literal `iteration = once` sweep cannot be judged diverging by construction.** Methodus's
  `solve_blocks` already backtracks each sweep to guarantee its own residual does not grow (it
  refuses `SolveError::LineSearchFailed` first), and Krasis's divergence test compares a sweep's
  interface change against the *previous* sweep's, which a lone `Once` sweep has none of. The
  architecture's "wrong Dirichlet-Neumann roles" instability therefore only surfaces across
  repeated application: an `Implicit` iteration (demonstrated above) or, for a plugged-in `Once`
  solver, across repeated BDF steps until a later step's evaluation hits a typed non-finite
  refusal. This is intrinsic to driving the iteration through Methodus's existing per-sweep
  algorithm as the brief directs, not a gap in this package.
- **No iterate acceleration.** Methodus does not yet expose Aitken/IQN acceleration (SV7-F3);
  `PartitionedIteration::Implicit` offers only plain iteration and the fixed relaxation factor
  Krasis itself applies (a vector combination). Cross-repo need: once Methodus lands an
  acceleration primitive over `&[f64]` iterate sequences, `PartitionedConfig` should gain an
  `acceleration` axis consuming it instead of (or in addition to) fixed relaxation.
- **`ConnectedSystemOperator` partitioning is wired but not proven against a real matching
  fixture.** The `BlockNonlinearOperator` impl is real and type-checked (leaf ranges are
  preserved under elimination: `new`/`new_system` already require
  `constraints.dof_count() == inner.dimension()`), but no `ConnectionRealizationPlan`-based
  two-leaf matching-interface fixture exists yet as reusable Krasis test scaffolding, and
  building one from scratch (mesh pairing, `InterfaceRealization`, elimination) was beyond this
  package's time budget. Package 2's "own verification source" is therefore the edge-composed
  `CoupledSystemOperator` two-leaf conduction fixture (`CouplingEdge` exchange), not the
  constraint-eliminated `ConnectedSystemOperator` matching interface; a literal SC-W2-style
  matching heat-heat partitioned agreement gate is open, pending either a Finitum-exported
  reusable matching fixture or a dedicated follow-on package.
- **The transient (`PartitionedFixedPoint`) report is thinner than the steady one.**
  `methodus::NonlinearSolver::solve` returns only the generic `methodus::SolveReport`
  (state, converged, concatenated per-sweep Newton traces); the richer `PartitionedReport`
  (schedule, relaxation, per-sweep interface norms, disposition) is not recoverable from a
  BDF-embedded call, exactly as `BlockNewton` and `NewtonKrylovSolver` cannot report their own
  shape today either. `PartitionedExecution` (steady) returns the full `PartitionedReport`
  directly.
- N-way junctions, general per-stratum matching, and nonmatching transient composition (pending
  Finitum) remain out of scope, as recorded in prior status and `sinbad/ARCHITECTURE.md`.

2026-09-18 bounded implementation accepted:
SHOW-3 / SC-W3: matching multi-component DAE composition, consistent initialization and differential/algebraic trace-row classification. Connected diagonal sums are available for independent leaves and simple matching equivalence classes; unsupported classes refuse. Product trajectory/JVP tests pass; complete owner gate passed: 70 tests across 11 targets, formatting, check, strict clippy, rustdoc and doctests. Final consumer acceptance passed: 210 tests across 35 targets, with a documented external-fixture target retry and unchanged source.


Historical September 17 checkpoint (superseded by the accepted extension above):
SC-W2 first matching-interface implementation:
ConnectedSystemOperator composes checked Finitum affine trace elimination with
CoupledSystemOperator, including state/rate expansion, residual transpose
restriction and the corresponding JVP. Leaf realization identity and extent
must match the proof. Sinbad consumes it inside a steady trial/commit transaction.
Nonmatching, transient-interface strategies and n-way composition are not claimed.
Owner gate: 70 tests across 11 targets, fmt, strict all-feature clippy,
rustdoc and doctests pass with unchanged source hashes. Consumer final acceptance
is pending. Finitum dependency: `2de7dd8cfcf173f364114728c590402ccc4d820c`.
Evidence: `docs/validation/2026-09-17-sc-w2/`.


Updated: 2026-09-29
Milestone: SC-W3 package 1/2, the partitioned fixed-point transaction.

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
