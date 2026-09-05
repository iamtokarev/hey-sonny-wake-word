# `Hey Sonny` wake-word

A "Hey Sonny" wake-word model: openWakeWord's training recipe, Piper-
synthesized positives, and Hugging Face Jobs for the paid runs, exported to a
self-contained ONNX classifier for local inference.



Full experiment log, costs, and decisions: [`docs/experiment-plan.md`](docs/experiment-plan.md).

## How it works

Two Hugging Face Jobs:

1. **`scripts/prepare_job.py`** — generates synthetic positive/adversarial
   speech with Piper, builds a balanced multi-role augmentation pool (speech,
   babble, music, environmental sound), extracts openWakeWord embeddings, and
   publishes everything to a Dataset repo.
2. **`scripts/train_job.py`** — trains the classifier head, evaluates recall at
   a matched false-accept budget (overall, per condition, and on a fixed-SNR
   stress grid), exports ONNX, and optionally promotes the result to a Model
   repo.

Commands, hardware flavors, and costs for both: [`docs/jobs-spec.md`](docs/jobs-spec.md).

## Quickstart

```bash
# try the promoted model — microphone or a wav file
uv run scripts/try_model.py

# compare every published run by recall at a matched FA budget
uv run scripts/compare_runs.py

# score a model against controlled background interference at fixed SNRs
uv run scripts/stress_test.py --clips your_recording.wav
```

Each script in `scripts/` is a self-contained `uv` script — nothing to install
first beyond `uv`, and `hf auth login` while the repos are private.

## Repo layout

| Path | What's there |
| --- | --- |
| `scripts/` | The two Job scripts, plus `preflight.py`, `compare_runs.py`, `stress_test.py`, `try_model.py` |
| `docs/experiment-plan.md` | The experiment log — what each change was, what it cost, what it did |
| `docs/jobs-spec.md` | How to actually run the jobs |
| `docs/research/` | Supporting research: openWakeWord's own defects, how our recipe differs from upstream's, the noise-robustness investigation, the settled full-run configuration |
