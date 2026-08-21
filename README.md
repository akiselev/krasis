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
