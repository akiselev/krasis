# Agent instructions

Krasis owns coupled field/material state, history, transaction boundaries, events,
checkpoints, and composition into Solverang-facing problems.

Do not add scientific parsing or form semantics, mesh/basis/assembly implementations,
kernel compilation, or numerical solver algorithms. Implement Solverang-owned traits
directly instead of creating Krasis contract or bridge crates.

Run formatting, clippy with warnings denied, and all tests before handoff. Keep `STATUS.md`
compact and current.

