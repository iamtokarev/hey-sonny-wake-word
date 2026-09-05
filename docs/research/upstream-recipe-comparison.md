# Upstream's training recipe versus ours

Written 2026-09-02, after the first full Job B run produced recall 0.52 at
0.47 FA/h and the question "is that bad?" turned out to need three reference
points rather than one. Sources are all at the pinned commit `368c037` unless
noted.

There is no single "upstream method". There are three, and they differ from each
other more than ours differs from any of them:

| | **R — released models** | **A — auto-trainer** | **T — tutorial notebook** |
| --- | --- | --- | --- |
| Where | [`docs/models/hey_jarvis.md`](https://github.com/dscripka/openWakeWord/blob/368c037/docs/models/hey_jarvis.md) | `train.py` + [`examples/custom_model.yml`](https://github.com/dscripka/openWakeWord/blob/368c037/examples/custom_model.yml) | [`notebooks/training_models.ipynb`](https://github.com/dscripka/openWakeWord/blob/368c037/notebooks/training_models.ipynb) |
| What it is | What dscripka actually shipped | What the README tells you to run | Teaching material, explicitly *not* for production |
| Reachable by us | No — see [architecture](#architecture) | Yes | Yes |

Ours is a fork of **A**. The relevant question is therefore two questions:
how far are we from A (should be zero, and nearly is), and how far is A from R
(very far, and that is upstream's own doing).

## The bar upstream sets for A

From `automatic_model_training.ipynb` cell 12, upstream's own calibration:

> there are *not* specific guidelines about what these metrics should be in
> practice ... However, from very limited testing the default values in the
> config file (accuracy >= 0.7, recall >= 0.5, false-positive rate <= 0.2 per
> hour) seem to produce models with reasonable performance.

The first full run: **recall 0.52 at 0.47 FA/h**, or 0.35 at 0.09 FA/h. That
sits on the bar, not under it. `train_model`'s `val_recall` is torchmetrics
binary recall on the balanced positive/adversarial validation set at threshold
0.5 — the same quantity we report, so the comparison is direct.

One asymmetry makes ours the harder measurement. `train.py` generates positive
train and positive test from the same generator with the same speaker traversal
and no `max_speakers`, so upstream's `val_recall` is measured on speakers the
model trained on. Ours holds out 12 of 173 speakers. Our 0.52 and their 0.5 are
not the same test.

## Data quantities

| Axis | R (released) | A (default) | Ours | Ratio to R |
| --- | --- | --- | --- | --- |
| Positive clips | ~200,000 | 10,000 | **30,000** (+2,000 val) | 0.15x |
| Adversarial negatives | present, unquantified | 10,000 | **30,000** (+2,000 val) | — |
| Generic negative bank | **~31,000 h** | 2,000 h (ACAV features) | **2,000 h** (same file) | 0.065x |
| Negative composition | ACAV 10k h + Common Voice 11 10k h + podcasts 10k h + FMA 1k h, plus reverberated ACAV | ACAV only | ACAV only | — |
| Background corpus for augmentation | ACAV + FSD50K + FMA | AudioSet `bal_train09` + FMA-small, "recommended to download the entire dataset" | four balanced roles, 4 h each — see below | — |
| Room impulse responses | BIRD (simulated) | MIT environmental, 270 | MIT environmental, 270 | 1x |
| TTS systems | 2 (NVIDIA WaveGlow/LibriTTS **and** VITS/VCTK) | 1 (Piper LibriTTS-R) | 1 (Piper LibriTTS-R, 173 speakers, SLERP-blended pairs) | 0.5x |
| `augmentation_rounds` | — | 1 | 3 | — |
| Positive test data | **manually recorded**, real home, 3–30 ft, near and far field | synthetic, speakers seen in training | synthetic, speakers held out | — |

The README is explicit that both axes scale:

> The number of generated examples required can vary, a minimum of several
> thousand is recommended and **performance seems to increase smoothly with
> increasing dataset size**.

and, of negative data: "This also benefits from scale, and the included models
were all trained with ~30,000 hours".

### The one place we fell below A, not just below R

**Background audio: 4.0 hours** of openWakeWord *sample* zips — a placeholder
that survived into the first paid run, when the auto-training notebook pulls an
AudioSet shard plus FMA-small and tells you to take the whole dataset. This is
also the augmentation upstream rates highest: the tutorial notebook says
outright that "mixing with realistic background data provides the best
results", above reverberation. Corrected in E1; the consequences of the gap are
in [noise-robustness.md](noise-robustness.md).

Two mechanics of `augment_clips` govern how a pool is actually consumed:

- **`AddBackgroundNoise` draws a background file and an SNR per clip**, not per
  batch (`randomize_parameters` calls `random_background` once per batch
  element and samples `snr_in_db` with shape `(batch_size,)`); `mode="per_batch"`
  only decides whether the transform is applied at all. So 30,000 clips at
  p=0.75 get ~22,500 independent draws, and a source's share of the draws is
  its share of the **file count** — hours do not enter. Balance the pool by
  file count, and use `augmentation_rounds` to buy each positive several
  different conditions.
- **Reverberation *is* per batch**: the guard
  `if augmentation_probabilities["RIR"] >= np.random.random()` sits outside the
  per-clip loop, so a batch of 16 gets one RIR or none — about 940 RIR draws
  from 270 files. This is the part `augmentation_batch_size` affects, and the
  reason `custom_model.yml` warns against raising it.

## Recipe details that match

Checked line by line against `train.py`; these are not discrepancies, recorded
so nobody re-checks them:

- **Window.** 32,000 samples / 16 frames. `train.py` computes
  `median_clip_duration + 12000`, floors at 32,000, and snaps to 32,000 when
  within 4,000 — so any phrase with a median under 1.5 s lands on 32,000. The
  released models' 1,536-wide flattened input is 16 x 96 too.
- **Alignment.** `create_fixed_size_clip` sets
  `start = n_samples - (len(x) + end_jitter)` with `end_jitter` uniform on
  0–200 ms: positives are **right-aligned**, ending 0–200 ms before the window
  ends, so the model fires immediately after the phrase. Identical in the
  tutorial notebook's manual `starts` computation. Job A gets this for free
  from `augment_clips`.
- **TTS settings.** `noise_scales=[0.98]`, `noise_scale_ws=[0.98]`,
  `length_scales=[0.75, 1.0, 1.25]` for train; `1.0`/`1.0` for validation.
  Copied exactly.
- **Adversarial texts.** `generate_adversarial_texts(..., include_partial_phrase=1.0, include_input_words=0.2)`. Copied exactly, with a
  configurable share replaced by custom near-miss phrases.
- **Schedule.** 50,000 + 5,000 + 5,000 steps, lr 1e-4 / 1e-5 / 1e-6,
  `max_negative_weight` 1500, `target_false_positives_per_hour` 0.2,
  `layer_size` 32, `model_type` dnn, `batch_n_per_class` 1024/50/50. Ours
  diverges deliberately on the first three — see `experiment-plan.md`.

## Architecture

`docs/models/hey_jarvis.md` reports 102,849 parameters and
`Linear: [1, 1536] -> [1, 64]`, so the released first stage is exactly
`train.Model(layer_dim=64, n_blocks=1)` — twice the `layer_size: 32` that
`custom_model.yml` defaults to and that we inherited.

The released model then adds a second network:

> The second network is trained on a subset of the data ... intended to act as a
> verifier model, only predicting on audio frames that have a score > 0.5 from
> the first model. These models are combined together prior to exporting to the
> final ONNX format.

`train.py` cannot produce this. There is no config key for it, no second
training pass, and `export_model` exports one `nn.Module`. The released
architecture is off the auto-trainer's map entirely.

## Inference-time differences

Upstream's own tutorial reaches an acceptable false-accept rate through the
runtime, not the model. Its small-data model has an FA rate it calls
"unnacceptably high"; adding a **custom verifier model** — a scikit-learn
logistic regression over the same embeddings, trained on ~3 clips of the target
speaker via `openwakeword.train_custom_verifier` — drops it to under 1 per hour.
The notebook also instantiates the detector with
`enable_speex_noise_suppression=True, vad_threshold=0.5`, and the README says
noise suppression "can reduce both false-reject rates and false-accept rates".

We measure a bare ONNX classifier with none of these. That is the right thing
for a model-quality metric and the wrong thing for a deployment estimate — a
real deployment number would need the wrapper included, which this project does
not measure.

## The gaps, and what happened to each

| Gap | Outcome |
| --- | --- |
| Background corpus, 4 h of sample zips | Fixed in E1: four balanced roles including speech, 4 h per role, at 0..20 dB. The largest single win of the campaign. |
| `layer_size` 32 vs the released 64 | Fixed in E2, which went further: 128. Anything ≥ 64 beats 32 substantially; the optimum within 64–512 is below measurement resolution. |
| `augmentation_rounds` 1 | Raised to 3 with E1. |
| Positives 30,000 vs ~200,000 | Tried at 100,000 in E3 and **rejected** — it lost recall at the same step budget. Confounded with two other changes; see `experiment-plan.md`. |
| Negative bank 2,000 h vs ~31,000 h | Not attempted. No prepared feature file exists past ACAV100M, so it needs a new Job A stage over raw corpora, and it targets false accepts, which are already near target. |
| Verifier model | Not attempted. Speaker-specific, needs the user's own recordings, and is a different deliverable. |

## Primary sources

- [`docs/models/hey_jarvis.md`](https://github.com/dscripka/openWakeWord/blob/368c037/docs/models/hey_jarvis.md) — the released recipe
- [`notebooks/automatic_model_training.ipynb`](https://github.com/dscripka/openWakeWord/blob/368c037/notebooks/automatic_model_training.ipynb) — the auto-trainer's own quality bar
- [`notebooks/training_models.ipynb`](https://github.com/dscripka/openWakeWord/blob/368c037/notebooks/training_models.ipynb) — alignment, mixing SNRs, verifier models
- [`openwakeword/data.py`](https://github.com/dscripka/openWakeWord/blob/368c037/openwakeword/data.py) — `augment_clips`, `create_fixed_size_clip`
- [`README.md`](https://github.com/dscripka/openWakeWord/blob/368c037/README.md#training-new-models) — scaling guidance
