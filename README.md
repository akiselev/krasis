# Krasis

Krasis is the stateful coupling layer of the Sinbad stack. It owns field instances,
state layout, trial/commit/rollback, constitutive history, events, and checkpoints,
and eventually the aggregation of Finitum operators into Solverang-facing nonlinear
and DAE systems. Those public systems will land with real stateful composition; Krasis
does not expose forwarding adapters as architectural placeholders.
