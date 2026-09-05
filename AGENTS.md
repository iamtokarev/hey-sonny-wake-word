# Working in this repo

A "Hey Sonny" wake-word model: openWakeWord's recipe, Piper-synthesized
positives, two Hugging Face Jobs for the paid work, ONNX out. `README.md` has
the current model and its numbers.

## Where things are

- [`docs/experiment-plan.md`](docs/experiment-plan.md) — every experiment run:
  what it changed, what it measured, what was decided. Read before proposing
  another one; several obvious ideas have already been tried and rejected.
- [`docs/jobs-spec.md`](docs/jobs-spec.md) — running, debugging or costing a
  job: commands, flags, flavors, the dependency header, volumes and secrets.
- [`docs/research/full-run-plan.md`](docs/research/full-run-plan.md) — the
  settled configuration, and which code is ours versus upstream's.
- [`docs/research/upstream-recipe-comparison.md`](docs/research/upstream-recipe-comparison.md)
  — how this recipe differs from upstream's three. Read before calling a
  result good or bad.
- [`docs/research/noise-robustness.md`](docs/research/noise-robustness.md) —
  why the model failed under other speech and music, and what fixed it. Read
  before changing the augmentation pool.
- [`docs/research/openwakeword-baseline.md`](docs/research/openwakeword-baseline.md)
  — upstream's defects and the patch each one needs. Read before touching
  `openwakeword.data`, `train.py`, or any false-accept number.

`docs/` is durable memory. Update a note when it goes stale rather than
appending a correction to it, and keep working notes — attempt logs, scratch
findings, to-do lists — out of it entirely.

## Rules that cost money or correctness

**Jobs spend real money, and the user submits them.** Never run `hf jobs`
unless asked to. Rehearse every code change first — `--steps 400
--refine-steps 100 --limit-fp-frames 60000` for Job B, `--n-train 60
--n-val 20` for Job A — which has caught eight faults for under a dollar
total. Promotion to `iamtokarev/hey-sonny` is always an explicit user
decision, never a step you take because a model looks good.

**Never compare a metric across two runs.** Validation clips differ whenever
the speaker grid changes, so a number from another run's `metrics.json` is not
a baseline. Compare against the `baseline` row that `--baseline-model` scores
inside the same job, on the same clips.

**A false-accept rate is meaningless without its corpus.** The same model at
the same nominal 0.2 FA/h reads 0.458 on upstream's speech-heavy set and 0.773
on the ACAV holdout. Always report which set a number came from.

**Rank by recall at a matched false-accept budget, never at a fixed
threshold.** Models out of this pipeline calibrate differently — three of them
put threshold 0.5 at 0.47, 0.93 and 1.31 FA/h — so a shared threshold compares
different operating points.

**Verify an artifact by exercising it from where a consumer gets it.**
Checksums prove transport, not usability; five published models passed their
round-trip and loaded nowhere.

**The shell is zsh.** Unquoted variables do not word-split, so a variable
holding a flag string arrives as one argument. Write flags out literally in
job submissions. `status` is read-only — don't use it as a variable name.

## Conventions

- openWakeWord is pinned at `368c03716d1e92591906a84949bc477f3a834455`. Record
  that commit in every manifest; PyPI `0.6.0` is not equivalent.
- Both job scripts take flags, not a config file. `--help` is the contract.
- Each script in `scripts/` is a self-contained PEP 723 `uv` script. The
  authoritative dependency header lives in `preflight.py`; copy it rather than
  composing a new one, and re-run preflight after any deliberate change.
- Job A output goes to a **new** dataset repo per data change, never a new
  revision — `-v hf://datasets/...` mounts the default branch and cannot pin a
  revision, so a changed dataset behind an unchanged mount path is a
  reproducibility trap.
