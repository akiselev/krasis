# Krasis

Krasis is the stateful coupling layer of the Sinbad stack. It owns field instances,
state layout, trial/commit/rollback, constitutive history, events, and checkpoints,
plus the aggregation of Finitum operators into direct Solverang nonlinear, block, and
DAE implementations. `CoupledExecution` encloses BDF attempts in the state transaction
and checkpoints Krasis state together with Solverang history. Checkpoints bind to Finitum's
concrete realization digest, so same-size mesh, constraint, coefficient, or material changes are
refused. No forwarding adapter or copied numerical contract sits between the owning repositories.
