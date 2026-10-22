# Lane `krasis-fv` (repository: krasis, branch `claude/b13-krasis-fv`, base `d981980`)

Work package: **SV2-C4/SC-W3 leftovers owned by Krasis**: the cross-repository requests Sinbad
filed in W10 and W12.

## Packages

1. **`OperatorIdentity for finitum::FvRealization`** (W12 request): Krasis already depends on
   Finitum and implements `OperatorIdentity` for `finitum::MixedOperator` (`src/block_solve.rs`).
   Add the impl over `FvRealization::digest()` with a `finitum-fv:` prefix, mirroring the mixed
   one, so Sinbad can drop its `FvExplicitOperator` newtype. Test: identity string is stable and
   changes with the realization digest.
2. **Per-step partitioned evidence inside BDF** (W10 request): `PartitionedFixedPoint` inside
   `CoupledExecution` reports only the generic trace today. Record a `PartitionedReport` (the
   type exists in `src/partitioned.rs`) per committed BDF step in the step record (versioned,
   `skip_serializing_if` so steps without partitioning serialize byte-identically), including
   sweeps, residual history, acceleration kind and the per-sweep cost, and expose it from the
   execution's step records. Test: a two-component transient run through the coupled execution
   with a partitioned schedule carries one report per step whose sweep count matches the
   partitioned trace, and the monolithic run serializes unchanged (pin its bytes).
3. **Explicit execution over a per-component invariant layout**: `ExplicitExecution`'s
   conservation summary reports invariants by index; add optional names/components to
   `InvariantBalanceCheck` (versioned, default empty) so a system with several conserved
   components (mass, momentum, energy) reports each by name once Finitum declares them. Test: the
   existing explicit tests unchanged plus a named-invariant case.
4. **STATUS**: a section with what landed and the still-open items (coupled explicit groups,
   multirate, IMEX transaction, which waits on Methodus's IMEX integrator from lane
   `methodus-imex`).

## Files you own

`src/block_solve.rs` (the impl), `src/coupled.rs` / `src/coupled_system.rs` / `src/partitioned.rs`
(per-step report), `src/explicit.rs` (invariant names), `tests/sc_w3_partitioned.rs`,
`tests/sv2c_explicit_execution.rs`, `tests/coupled_execution.rs` (additive), your STATUS section.

## Done means

Packages 1–4 landed, `cargo test --locked -p krasis` green across all targets (the crate's tests
are quick), clippy/rustdoc clean, report §4 with the serde shapes added.

## Refuse rather than guess

Transient nonmatching transfer (`nested_p1`/`planar_p1` are steady-only in Finitum), n-way
junctions and `roles`/explicit serial order are not this lane's; record them as still open.
