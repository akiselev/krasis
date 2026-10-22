# Sinbad batch 13 (W13): common brief for every lane

This brief binds every lane of batch 13. The coordinator is the Sinbad coding session
(`session_0163LP3H7ZzKtKkiWy3sDCEv`, 2026-10-22). Your scope is in `docs/notes/batch13/LANE-<lane>.md`
next to this file. Read both fully before touching code. You know nothing else about the batch,
so everything you need is here or in the repositories named below.

W13 continues the W12 finite-volume wave (SV2 Wave C, `sinbad/docs/validation/2026-10-19-sv2c7-scalar-fv/README.md`)
toward SV9 F1 (Euler building blocks) and SV9 E5 stage 1 (conjugate heat transfer on one FV mesh), and
closes W10/W12 cross-repository requests. The user decisions of 2026-10-19 bind: D1 FV method intent
lives in the case `[discretization]` section and `.res` models declare `DG(order=0)`; D2 Finitum owns
TensorBox/periodic meshes; D3 generic numerical fluxes need only `F` and declared wave-speed
information, so Rust contains no physics; D4 named Riemann solvers (HLLC, Roe, ...) exist only as
`.res` catalog modules, never as Rust branches; D5 conjugate heat transfer starts on a single FV mesh;
D6 every nonsmooth derivative is a typed refusal. Reference solutions for FV come from a new
`sinbad-oracle-clawpack` adapter repository.

## Where you work

- Your repository and branch are named in your lane file: `claude/b13-<lane>` in one repository.
  Push only to that branch. No pull requests. Never push to `master`, `main` or another lane's branch.
- Base shas (every lane starts from these; pin siblings to exactly these):

  | repository | sha | GitHub |
  |---|---|---|
  | sinbad | `5341c48` | akiselev/sinbad (private) |
  | scientia | `84188ac` | akiselev/scientia (public) |
  | finitum | `90c4789` | akiselev/finitum (public) |
  | krasis | `d981980` | akiselev/krasis (public) |
  | methodus | `8bd8333` | akiselev/methodus (public) |
  | malleus | `7a43e7f` | akiselev/malleus (public) |
  | quantitas | `0d682ac` | akiselev/quantitas (public) |
  | resolvent | `b604f93` | akiselev/resolvent (public) |
  | artifactum | `2c016f1` | akiselev/artifactum (public) |
  | cadabra3 | `ecdb54b` (branch `main`) | akiselev/cadabra3 (private) |
  | sinbad-oracle-fenicsx | `ca1ddca` | akiselev/sinbad-oracle-fenicsx (public) |

## Setup: sibling repositories by relative path

Every repository here takes its siblings as Cargo path dependencies `../<name>` (see `[dependencies]`
in `Cargo.toml`, transitively). Clone each sibling you need **next to your checkout** (if your
checkout is `/home/user/<repo>`, siblings go to `/home/user/<sibling>`), at the pinned sha:

```sh
cd "$(git rev-parse --show-toplevel)/.."
git clone --depth 1 https://github.com/akiselev/<sibling> <sibling>
git -C <sibling> rev-parse --short HEAD      # must print the pinned sha; if not:
git -C <sibling> fetch --depth 1 origin <sha> && git -C <sibling> checkout -q <sha>
```

Which siblings each repository needs (transitive closure of its path dependencies):
- scientia: malleus, quantitas, resolvent.
- finitum: scientia, malleus, methodus, quantitas, resolvent, cadabra3.
- krasis: finitum, methodus, scientia, malleus, quantitas, resolvent, cadabra3.
- methodus: none.
- sinbad: all of the above plus artifactum, and sinbad-oracle-fenicsx for the recorded-oracle tests.

`cadabra3` is private: call the `add_repo` tool (claude-code-remote) with `owner: akiselev`,
`repo: cadabra3`, `access: read`, then clone exactly as its result says (one clone at a time, a
generous timeout), then `git -C cadabra3 fetch --depth 1 origin ecdb54b && git -C cadabra3 checkout -q ecdb54b`.
Clone one repository at a time; the proxy allows about two concurrent git operations. If a sibling
cannot be fetched, record a blocker in your report and stop rather than substituting a guess.

## What to read

- Your repository's `AGENTS.md` / `CLAUDE.md` / `STATUS.md` (the top block is the latest handoff) and
  the relevant plan file. For Sinbad lanes: `AGENTS.md`, `PLAN.md`, `STATUS.md`,
  `docs/simulation-vision/SV2-METHOD-AND-REALIZATION-PLATFORM.md` (Wave C),
  `docs/simulation-vision/SV9-FULL-SYSTEM-CAMPAIGNS.md` (Waves E/F),
  `docs/simulation-vision/GX-CONTRACTS.md` §C12.16 (the W12 contract record), and the W12 capsule
  `docs/validation/2026-10-19-sv2c7-scalar-fv/README.md`.
- The W12 lane work you build on: Scientia `src/conservation.rs` + `tests/sv2c1_conservation.rs`;
  Finitum `src/fv.rs`, `src/fv_topology.rs`, `src/fv_reconstruction.rs` + `tests/sv2c2a_fv.rs`,
  `tests/sv2c2b_fv.rs`; Krasis `src/explicit.rs`; Methodus `src/explicit.rs`; Sinbad
  `src/finite_volume.rs`, `cases/fv/`, `physics/methods/scalar-conservation.res`,
  `tests/sv2c7_scalar_fv.rs`, `tests/sv2c7_identity_pins.rs`.

## What cannot run here

No Docker, no FEniCSx image, no Wine, no Parasolid, no user datasets. Tests that need them are
compiled at most; never claim they passed. Chromium with Playwright is installed
(`PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`; never run `playwright install`) if a lane needs a
headless browser check.

## Building

- `export CARGO_PROFILE_DEV_DEBUG=0 CARGO_BUILD_JOBS=$(nproc)` (disk is a fixed allowance; one
  target directory).
- While writing: `cargo check --locked -p <crate> --all-targets` only. `cargo clippy --locked
  -p <crate> --all-targets -- -D warnings` and `cargo fmt -p <crate> -- --check` before each push.
- Your own tests once at the end: `cargo test --locked -p <crate> --test <target>` per target you
  own, plus `--lib` if you touched library code. Sinbad integration tests are slow; use
  `CARGO_PROFILE_TEST_OPT_LEVEL=2` for them. Never run the full workspace gate (the consolidation
  does), never `cargo test` of every target.
- Do not commit `Cargo.lock` changes unless your lane adds a dependency (then say so in the report).
- Format only files you changed.

## Rules

- Stay in the files your lane owns. An edit to a file another lane owns is listed under
  "cross-lane edits" in your report with the reason; keep such edits minimal and additive.
- Append-only tables and ledgers (STATUS entries, identity-pin tests, contract records) get new rows
  or sections; never rewrite or delete existing rows. Existing identity pins must stay unchanged;
  if a change necessarily moves a pinned identity, that is a versioned schema bump with a stated
  reason, recorded in the report, never a silent re-pin.
- Repository rules bind: one owner per semantic type; no named-physics dispatch in Rust (physics
  lives in `.res` and case data); typed refusals with stable codes, never NaN or silent fallback;
  tolerances and expectations predeclared with a basis, never widened after viewing a result (if a
  gate fails, report the number, do not move the bar); verification is not validation; generated
  numbers are not independent evidence.
- Never guess a fact you lack (a sibling API, a reference value, a convention): record it as a
  blocker or a request and continue with what is certain.
- `STATUS.md` of your repository: add one section for your lane under the latest wave heading
  (landed truth and exact checks only). Keep it under the file's line cap if one exists by
  condensing older history, never by deleting current limitations.

## Commits

Commit early and often, push after every meaningful step (`git push -u origin claude/b13-<lane>`,
retry with backoff on network errors). Clear messages in the imperative, naming the work package.
End each commit message with the attribution lines your session's instructions give you. Leave the
tree clean at the end.

## Requests to other repositories or lanes

Anything you need changed outside your repository goes in `docs/re-requests/batch13-<lane>.md` on
your branch (the exact call or data you need, the gate it unblocks, the sha you built against), or
the single word "none" if there is nothing. The coordinator files them.

## Final report (your last action)

Commit `docs/notes/batch13-reports/<lane>.md` with exactly these sections and push:

1. **Branch and final sha.**
2. **What landed**, package by package, with the proving test names.
3. **Tests run**: exact commands and counts (passed/failed/ignored) per target.
4. **Contract shapes landed** (schema ids, types, refusal codes) for the coordinator to fold into
   `GX-CONTRACTS.md` C12.
5. **Expected moves**: identities, goldens, pinned numbers that changed, each with its reason.
6. **Cross-lane edits** and files touched outside your ownership.
7. **Contradictions and blockers**: anything that disagrees with the brief, the plan or the code.
8. **Carry-overs**: what is left open, precisely.

## Context limit

When you near your context limit: stop starting new packages, commit, write the report with a
handoff of the half-done package (what is done, what is not, how to continue) and push.
