# Krasis status

2026-09-18 implementation in acceptance:
SHOW-3 / SC-W3 candidate: matching multi-component DAE composition, consistent initialization and differential/algebraic trace-row classification. Connected diagonal sums are available for independent leaves and simple matching equivalence classes; unsupported classes refuse. Product trajectory/JVP tests pass; complete owner gate pending.


SC-W2 first matching-interface implementation (2026-09-17):
ConnectedSystemOperator composes checked Finitum affine trace elimination with
CoupledSystemOperator, including state/rate expansion, residual transpose
restriction and the corresponding JVP. Leaf realization identity and extent
must match the proof. Sinbad consumes it inside a steady trial/commit transaction.
Nonmatching, transient-interface strategies and n-way composition are not claimed.
Owner gate: 70 tests across 11 targets, fmt, strict all-feature clippy,
rustdoc and doctests pass with unchanged source hashes. Consumer final acceptance
is pending. Finitum dependency: `2de7dd8cfcf173f364114728c590402ccc4d820c`.
Evidence: `docs/validation/2026-09-17-sc-w2/`.


Updated: 2026-09-17
Milestone: SC-W2 first matching-interface residual and JVP composition.

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
