# Krasis

Krasis is the stateful coupling layer of the Sinbad stack. It owns field instances,
state layout, trial/commit/rollback, constitutive history, events, and checkpoints,
plus the aggregation of Finitum operators into direct Methodus nonlinear, block, and
DAE implementations. `CoupledExecution` encloses BDF attempts in the state transaction
and checkpoints Krasis state together with Methodus history. Checkpoints bind to Finitum's
concrete realization digest, so same-size mesh, constraint, coefficient, or material changes are
refused. No forwarding adapter or copied numerical contract sits between the owning repositories.
FC10 also provides `CrossDialectOperator`, which composes two distinct Finitum discrete families
with explicit bidirectional off-diagonal derivative blocks and implements Methodus's DAE and
block-nonlinear contracts without named-physics branching.

SV0-B4 adds reusable, serializable verification reports for transactional rollback,
checkpoint/restart trajectory identity, isolated cross-block derivatives, counted block-strategy
agreement, synchronized bounded history, and event-state disposition. Every report binds the
operator, state/block layout, full Methodus evaluation context, and checker configuration
identities. Each versioned report carries a canonical digest and validates only by recomputing
against the exact operator, initial state, inputs, context, and Finitum source. Event checking currently
targets a caller-supplied Methodus `DaeOperator`; it does not claim that `CoupledExecution`
persists event records. Coupled-execution reports require a compatible Finitum-owned report,
bind it to the exact realization identity, source-validate its body against that realization or
its mesh, and retain its typed header and acceptance without duplicating the Finitum schema. The
current adapters cover realization agreement and nodal-patch reports; state-dependent
realizations may use the latter when Finitum truthfully refuses partial-assembly agreement.
Identity-bearing floating-point inputs must be finite. Positive zero is canonical; negative zero
is refused instead of being collapsed by JSON encoding. Solver-error rollback evidence therefore
uses finite, structurally invalid solver policy rather than a nonfinite sentinel.
Coupled checks apply this recursively to checkpoint time, committed fields, field history,
constitutive state, Methodus BDF history, and exposed Finitum realization data such as mesh,
element, constraints, and stored external inputs before hashing.
