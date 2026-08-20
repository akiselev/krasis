# Krasis

Krasis is the stateful coupling layer of the Sinbad stack. It owns field and parameter
instances, block layout, trial/commit/rollback, constitutive history, events, checkpoints,
and the aggregation of realized operators into Solverang-facing nonlinear and DAE systems.

The repository starts from a clean transactional state model. The former
`sinbad-scientific-runtime` compatibility graph is not carried forward.

