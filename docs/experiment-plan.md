# Experiment log

Every experiment run against the "Hey Sonny" model, what it changed, and what
was decided. Started 2026-09-02, when the first full run
(`iamtokarev/hey-sonny@ae315c9c`) produced recall 0.52 at 0.47 FA/h. The
current model is `ea30342b`, at recall 0.715 at 1.0 FA/h.

Read [How we decide](#how-we-decide) before reading any number below, and
[upstream-recipe-comparison.md](research/upstream-recipe-comparison.md) before
calling a result good or bad.

## How we decide

This matters more than any single experiment, because the obvious metric
cannot support a decision.

**Rank by recall at a matched false-accept budget.** For each model, take the
best recall it reaches without exceeding B false accepts per hour, at B in
{0.2, 0.5, 1.0, 2.0}. That is a point on the model's own ROC curve, so it is
comparable across models. `recall_at_fa` in `train_job.py` computes it and
`metrics.json` stores the profile.

Not recall at a fixed threshold: three models out of this pipeline put
threshold 0.5 at 0.47, 0.93 and 1.31 FA/h. Their scores are calibrated
differently, so a shared threshold compares three *different* operating points
and ranks models by their score distribution rather than their quality. On that
metric E0a looked like the best run; at matched cost it was indistinguishable
from the baseline.

**On 2,000 validation positives the standard error is 0.011**, so a paired
comparison resolves to roughly +/-0.03. Treat a single-budget change below 0.03
as noise and one above 0.05 as real. **A margin near the noise floor that holds
at every budget is much stronger evidence than the same margin at one** — it is
four correlated reads of the same curve, and a curve that dominates everywhere
has not merely shifted its threshold.

**False accepts are never a score.** Five events in 10.7 hours has a Poisson 95%
interval of about 0.15 to 1.09 FA/h. They set the budget the recall is read at;
they are not themselves ranked.

**Never compare across runs.** Validation clips change whenever the speaker
grid changes, so another run's `metrics.json` is not a baseline.
`--baseline-model` scores a reference model on the same clips inside the same
job; read the gain from that row.

### The two false-positive sets are not interchangeable

The most important measurement finding of the campaign. Same model, same
nominal 0.2 FA/h budget:

| Set | Corpus | Hours | recall at 0.2 FA/h |
| --- | --- | --- | --- |
| `upstream` | a speech/noise/music mix, unrelated to training | 10.70 | **0.458** |
| `acav_holdout` | unseen ACAV clips — the training negatives' own corpus | 100.0 | 0.773 |

A 0.32 gap. Upstream's set is deliberately speech-heavy and speech is what
actually triggers a wake word, so it is the conservative and transferable
number; the holdout buys resolution and reads optimistically. **Report recall at
a false-accept rate only alongside the corpus it was measured on.**
`--select-on` therefore defaults to `upstream`, and the holdout is used to
compare configurations, where its resolution is what matters and both sets rank
the same way.

The holdout exists because upstream's file alone could not decide anything:
choosing a threshold for 0.2 FA/h on half of it produced **0.327 FA/h** on the
other half, and one fixed model's held-out recall ranged **0.234–0.566** over 12
splits. At 100 h the 0.2 FA/h target allows 20 events instead of 2. Event
grouping is time-based rather than frame-based, because upstream's file is flat
frames turned into stride-1 windows 80 ms apart while ACAV ships pre-windowed at
1.28 s per row — four seconds is 50 frames on one and 3 rows on the other.

The FA target remains a gate, and the final threshold still needs real-room
testing with `scripts/stress_test.py` and `scripts/try_model.py`.

## Bookkeeping

Experiments publish to a separate repo so that `iamtokarev/hey-sonny` always
holds a model somebody chose, not the last one that ran:

- Job B experiments -> `iamtokarev/hey-sonny-experiments`, under `runs/<run-name>/`
- Job A output -> a **new** dataset repo per data change, not a new revision:
  `-v hf://datasets/...:/features` mounts the default branch and cannot pin a
  revision, so a changed dataset behind an unchanged mount path is a
  reproducibility trap
- The winner is promoted to `iamtokarev/hey-sonny` by rerunning Job B with
  `--repo-id iamtokarev/hey-sonny` and the winning flags, from a clean tree

Each run records `config_sha256`, `JOB_ID`, and the resolved input revisions in
its manifest.

---

## E0 — loss weighting schedule

**What changed.** `--escalation-factor` (replacing a hardcoded `weight *= 2`),
`--merge-rule recall-gated`, evaluation of three candidates side by side
(merged, best single, final) instead of one, and `--run-name`.

The baseline escalated the negative weight 1500 -> 3000 -> 6000 because each
sequence's FA/h was above the 0.2 target, and recall fell 0.539 -> 0.532 ->
0.521 as it did. With FA already near target, that trade was not worth making.
The shipped model was also the final checkpoint of sequence 3 — the one trained
under weight 6000 — because 0 of 55 checkpoints cleared all three of
`merge_checkpoints`' percentile gates.

Five runs, ranked by recall at a matched budget on the exact ROC:

| Run | max_negative_weight | FA0.2 | FA0.5 | FA1.0 | FA2.0 | mean |
| --- | --- | --- | --- | --- | --- | --- |
| `v2-escalate` | 1500 -> 6000 | 0.357 | 0.528 | 0.578 | 0.631 | 0.524 |
| `v2-flat1500` | 1500 | 0.380 | 0.521 | 0.579 | 0.621 | 0.525 |
| **`v2-flat500`** | **500** | **0.478** | **0.566** | **0.627** | **0.658** | **0.583** |
| `v2-flat250` | 250 | 0.382 | 0.500 | 0.615 | 0.651 | 0.537 |
| `v2-flat125` | 125 | 0.356 | 0.576 | 0.610 | 0.644 | 0.546 |

**Adopted `--max-negative-weight 500 --escalation-factor 1.0`.** It leads at all
four budgets, its mean beats the baseline by 0.059, and the curve has turned
over on both sides, so the optimum is bracketed rather than guessed.

**Stopping the escalation was not the win; lowering the level was.**
`v2-flat1500` caps the ceiling at 1500 instead of climbing to 6000 and is
indistinguishable from the baseline everywhere. The weight ramps
`linspace(1, weight, steps)` *within* each sequence, so most steps run well
below the ceiling either way.

The `recall-gated` merge rule fires where upstream's found nothing: 5 and 3
checkpoints merged against 0 of 55. Both runs hit the documented fallback ("no
checkpoint met 0.2 FA/h; used the best N by FA"), so no checkpoint's *training*
FA estimate ever met target — worth knowing, and the fallback is why the rule
still produced a model.

### Three measurement faults found while reading these results

All three invalidated a first pass of E0, which was re-run as `v2-*` above.

**Every ONNX published before 2026-09-02 was unusable** — a 3 KB graph with its
weights in an unpublished sibling file. The metrics were computed from the torch
model in-job and remain valid; only the artifacts were bad. Guards and the
general rule are in
[jobs-spec.md](jobs-spec.md#a-checksum-round-trip-is-not-proof-the-model-works).

**An 11-point threshold grid cannot locate a low-FA operating point.** At
`max_negative_weight` 250 the model puts threshold 0.5 at 3.37 FA/h, so its
0.5 FA/h point falls between the 0.95 and 0.99 grid rows and the grid reported
recall 0.086 — a property of the grid, not the model.

**The replacement, a binary search over the exact ROC, was also wrong.** The
event count is *not* monotone in the threshold: lowering it adds a frame, and if
that frame is outside the current refractory window it becomes an event and
shifts the window forward, re-partitioning everything after it. The count can
fall as the threshold falls. The search returned a threshold meeting the budget
rather than the lowest one — 0.880 at recall 0.518 where 0.800 met the same
budget at 0.563. A second bug rode along: the best threshold for a given
false-positive set sits just *above the highest excluded* false positive, not at
the lowest included one.

Both were caught by one invariant, now asserted in `stage_evaluate`: **every
grid threshold is also a candidate for the exact sweep, so the exact reading can
never come out below the grid's.** It did, which is impossible, which is what
made the bugs findable.

---

## E1 — background corpus and augmentation rounds

**What changed.** Four balanced background roles (speech, synthetic babble,
music, environmental) re-cut to 10 s segments and truncated to an equal file
count, `--augmentation-rounds 3`, a parquet ingest branch for AudioSet and
LibriSpeech, and per-condition validation arrays plus a `stress/` tree so the
failing condition is visible to the metric at all.

The original hypothesis was only about *volume* — 4.0 hours of openWakeWord
sample zips against upstream's "download the entire dataset". Measurement
before the run found something worse: the pool had **no speech in it**, and the
model had learned "a second voice means negative". Root cause, evidence and the
per-clip draw mechanics are in
[noise-robustness.md](research/noise-robustness.md).

### `runs/e1-speech-v2` — speech added, SNR left at -10..15 dB

**Recall at <= 1.0 FA/h on the upstream set, per validation condition** (2,000
held-out-speaker clips, augmented under one role each):

| condition | baseline | E1 | change |
| --- | --- | --- | --- |
| clean | 0.915 | 0.795 | -0.120 |
| env | 0.665 | 0.547 | -0.118 |
| music | 0.467 | 0.385 | -0.082 |
| **speech** | **0.092** | **0.269** | **+0.177** |
| **babble** | **0.079** | **0.282** | **+0.203** |
| mixed, at 0.2 / 0.5 / 1.0 / 2.0 | 0.368 / 0.400 / 0.436 / 0.454 | 0.344 / 0.398 / 0.446 / 0.495 | flat |

**Stress grid** — 24 held-out clean positives, share above each model's own
1.0 FA/h threshold:

| | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
| --- | --- | --- | --- | --- | --- |
| speech, baseline -> E1 | 0.67 -> 0.96 | 0.54 -> 0.83 | 0.54 -> 0.83 | 0.33 -> 0.79 | 0.42 -> 0.75 |
| babble, baseline -> E1 | 0.58 -> 0.92 | 0.54 -> 0.96 | 0.63 -> 0.88 | 0.50 -> 0.83 | 0.50 -> 0.88 |
| music, baseline -> E1 | 0.92 -> 0.96 | 0.92 -> 0.96 | 1.00 -> 1.00 | 0.96 -> 0.92 | 0.83 -> 0.79 |
| env, baseline -> E1 | 1.00 -> 1.00 | 1.00 -> 1.00 | 0.96 -> 0.92 | 0.96 -> 1.00 | 0.88 -> 0.92 |

The data fix is real and the gain is on the failing axis. The cost is a
**regression on clean, env and music of 0.08–0.12** and a training-time
`val_fp_per_hr` of 58 at threshold 0.5 against ~10 on v1 data: the model became
trigger-happy on speech, its 1.0 FA/h threshold moved from 0.985 to 0.9994, and
the higher threshold is what cut the easy conditions. The mechanism is the
-10 dB end of the mixing range — a positive buried under babble ten times
louder than the phrase is, to the model, "babble = 1". The released models mixed
at 0..20 dB.

### `runs/e1b-snr0-20` — the same pool at 0..20 dB

Applied to training *and* validation augmentation, so this run's condition
arrays are milder than E1's and the comparison to read is its own paired
`baseline` row.

| upstream set, 10.7 h | baseline | e1b | change |
| --- | --- | --- | --- |
| mixed recall at 0.2 / 0.5 / 1.0 / 2.0 FA/h | 0.493 / 0.530 / 0.574 / 0.605 | **0.536 / 0.615 / 0.659 / 0.708** | +0.04 to +0.10, every budget |
| clean at 1.0 | 0.896 | 0.837 | -0.059 |
| env at 1.0 | 0.743 | 0.683 | -0.060 |
| music at 1.0 | 0.675 | 0.678 | 0 |
| **speech** at 1.0 | 0.236 | **0.545** | **+0.309** |
| **babble** at 1.0 | 0.177 | **0.506** | **+0.329** |
| threshold at 1.0 FA/h | 0.985 | **0.975** | down (E1 was 0.9994) |
| training `val_fp_per_hr` at 0.5 | — | 27 | E1 was 58 |

Stress grid: **1.00 in every cell** for speech, babble, music and env from 20 dB
to 5 dB, 0.92 speech and 0.96 babble at 0 dB, clean 1.00. The baseline in the
same job: speech 0.71 / 0.54 / 0.63 / 0.54 / 0.54, babble 0.67 / 0.50 / 0.63 /
0.46 / 0.46.

The SNR floor did what E1's diagnosis predicted: false-accept pressure halved,
the operating threshold came back below the baseline's, and the easy conditions
recovered most of what E1 lost while speech and babble gained a further 0.28 on
top of E1's gain. Its one regression (clean/env -0.06 on the augmented arrays,
1.00 on clean stress) is the price of hearing the phrase through a
conversation, which was the point.

**Promoted 2026-09-03** as `iamtokarev/hey-sonny@98bbc8b7`, trained on
`iamtokarev/hey-sonny-features-v2b`. The rerun reproduced e1b's numbers exactly
(same seed, same data).

---

## E2 — classifier capacity

**What changed.** `--layer-size`, from the `custom_model.yml` default of 32.
The released `hey_jarvis` first stage is `Linear: [1, 1536] -> [1, 64]`, twice
ours.

The overfitting worry was backwards. The model was **underfitting**, and
capacity was the largest single lever found:

| `layer_size` | upstream @0.2 | acav @0.2 | upstream @1.0 |
| --- | --- | --- | --- |
| 32 | 0.458 | 0.773 | 0.588 |
| 64 | 0.520 | 0.812 | 0.622 |
| 128 | 0.536 | 0.802 | 0.650 |
| 256 | 0.490 | **0.820** | 0.633 |
| 512 | 0.480 | 0.819 | 0.649 |

**The apparent peak at 128 is not established.** Upstream's set resolves two
events at the 0.2 budget, and the 100 h instrument ranks 256 first there. Across
the other budgets 64 through 512 sit within about 0.03 of each other. The
defensible claim is that anything at or above 64 beats 32 substantially, and
that the optimum within 64–512 is below this measurement's resolution. Do not
tune `layer_size` further against upstream's set at the 0.2 budget; it cannot
tell these apart.

Matching upstream was the wrong target anyway: upstream chose 32 for a
10,000-sample default and 64 for a 200,000-sample run, and neither is a
statement about *this* dataset.

### Two unplanned probes

| Probe | upstream @0.2 | acav @0.2 | Verdict |
| --- | --- | --- | --- |
| `x1` batch balance, 200 positives + 200 adversarial vs 1024 ACAV | **0.342** | 0.812 | **rejected** |
| `x2` 100,000 + 10,000 + 10,000 steps | **0.552** | 0.787 | adopt |

**x1 is the clearest demonstration that the corpus decides the answer.** It is
the *best* run on the ACAV holdout at 0.2 FA/h and the *worst* on upstream's
set — worse than the layer-32 baseline it was meant to improve. Raising the
positive share of each batch means fewer distinct negatives per step, which the
in-corpus set cannot see and the out-of-corpus set punishes. A campaign judged
on the holdout alone would have adopted it.

### Where configuration tuning ran out

Eleven full runs. Recall at 0.2 FA/h on the out-of-corpus set:

| Step | Change | @0.2 | Gain |
| --- | --- | --- | --- |
| start | upstream defaults: escalating weight, layer 32, 60k steps | 0.379 | — |
| E0 | flat `max_negative_weight` 500 | 0.458 | +0.079 |
| E2 | `layer_size` 128 | 0.536 | +0.078 |
| x2 | 100k + 10k + 10k steps | 0.552 | +0.016 |
| c2/c3 | capacity and steps combined | 0.564 | +0.012 |

**The gains halve and then halve again, and the last three runs are inside the
measurement's resolution.** Ranked by the mean across all four budgets, the top
six configurations span 0.607 to 0.630 — a spread smaller than the +0.079 that
either of the first two changes bought alone. Combining capacity with steps was
not additive: `c1` (layer 128, 100k steps) scored *below* layer 128 at 60k
steps, and layer 128 at 60k/100k/200k steps reads 0.536/0.508/0.563, which is
not a trend.

Settled configuration: `--max-negative-weight 500 --escalation-factor 1.0
--layer-size 128 --steps 200000 --refine-steps 20000`, best by mean across
budgets at 0.630. Everything after this point is data or architecture.

### A high operating threshold is not, by itself, fragility

That model decides at **0.999** where earlier ones decided at ~0.92, which
looked alarming enough to check rather than assume:

| | threshold | float32 recall | after float16 round-trip |
| --- | --- | --- | --- |
| layer 128 / 200k steps | 0.999017 | 0.561 | 0.544, 0.28 FA/h |
| layer 128 / 60k steps | 0.992063 | 0.420 | 0.414, 0.19 FA/h |

float32 spacing at 0.999 is 6e-8 — about 16,000 representable values between
0.999 and 1.0 — and the positives are not collapsed either, at 1,783 distinct
scores out of 2,000. **There is no precision problem.** The saturated model wins
in float32 *and* in float16, and preferring the older one on robustness grounds
would have cost roughly 0.14 of recall to avoid a problem that does not exist.

The one real effect is small: under float16 the saturated model's false-accept
rate drifts from 0.19 to 0.28 per hour while the older one holds. So **a
quantised deployment must re-pick its threshold on the quantised model** — which
is ordinary practice, not a reason to choose a weaker model.

---

## E3 — 100,000 positives

**What changed.** `--n-train 100000 --max-speakers 316 --val-speakers 22`, on
e1b's recipe otherwise. `max_speakers` must scale as `sqrt(n_train)` or the
extra clips come from the same 173 voices and are repetition rather than
diversity; LibriTTS-R has 904 speakers, so 316 is available.

One change was bundled in: **20% of the adversarial clips are custom phrases**
(`--adversarial-texts`) — name swaps one or two phonemes from "sonny" (honey,
johnny, tony, ronnie, sonic, sunday, money), the other assistants' wake words,
and the openers people say near a device ("hey, so anyway", "okay so"). "hey
sunny" is deliberately absent: it is a homophone of the wake phrase and would
label the phrase itself negative. Bundling broke the one-variable rule
knowingly, on the grounds that the adversarial set is 1/3 of one of six
negative sources.

Published as `iamtokarev/hey-sonny-features-v3`: 115,383 per class with 23,077
custom-phrase negatives, training arrays at `(300000, 16, 96)`. Two Job A
attempts failed first and produced two lasting fixes —
`generate_adversarial_texts` deduplicates and returns far fewer texts than
asked for (21,760 for N=115,383), so the list is now cycled to one slot per
clip; and a transient `cuFFT` failure at 300k rows led to `robust_augment`,
which retries a failed batch on the GPU and then on the CPU, because
`compute_features_from_generator` sizes its memmap up front and cannot tolerate
a missing batch.

**The validation speakers change with the speaker grid**, so v3's
`positive_val*` and stress positives are not the same clips as v2b's. Every
number below is read against the `baseline` row scored in the same job.

### `runs/e3-dnn-v3` and `runs/e3-rnn-v3`

| upstream set, recall at FA/h | 0.2 | 0.5 | 1.0 | 2.0 |
| --- | --- | --- | --- | --- |
| baseline (e1b, on v3 val) | 0.496 | 0.592 | **0.649** | 0.709 |
| e3-dnn-v3 `merged` (selected) | 0.469 | 0.511 | 0.610 | 0.661 |
| e3-rnn-v3 `best_single` (selected) | 0.504 | 0.551 | 0.596 | 0.690 |

By condition at <= 1.0 FA/h, DNN: clean 0.845 -> 0.764, env 0.716 -> 0.636,
music 0.719 -> 0.640, speech 0.503 -> 0.512, babble 0.577 -> 0.573. RNN: clean
0.802, env 0.652, music 0.628, speech 0.492, babble 0.521. ACAV holdout a wash
for both. Stress grids essentially unchanged.

**The model changed character rather than merely losing.** Training-score false
accepts fell from e1b's 27/h to about 5/h and the 1.0 FA/h threshold from 0.975
to 0.55–0.70: the negatives, including the 23k custom near-miss phrases, are
being rejected far more firmly, and recall at the 0.5 score fell with it. The
head that gained 0.056 on v2b loses 0.053 on v3 against the same baseline, so
whatever v3 did, it did to the data and not to one architecture.

### Decision: rejected

100k positives over 316 speakers with 20% custom near-miss negatives, at e1b's
200k steps, loses 0.04–0.05 mixed recall at 1.0 FA/h with either head. The
combination is confounded three ways — more positives *and* harder negatives
*and* the same step budget, meaning a third as many passes per positive. Two
runs would isolate it:

1. Job A `--adversarial-custom-fraction 0` on the same 100k/316 grid — is it
   the phrases? (~$0.90 + Job B $0.40)
2. Job B on v3 with `--steps 600000` — is it passes per positive? (~$1.20)

Both cost more than the whole campaign has spent per gained point, and E4 had
already produced a winner. `features-v3` stays published if this is revisited.

---

## E4 — recurrent head

**What changed.** `--model-type rnn`: upstream's 2-layer bidirectional LSTM of
width 64 over the 16 frames, `out[:, -1]` into one linear unit (~150k
parameters against the DNN's ~200k), in place of the flattened 1536 -> 128 -> 1
MLP. No code change — the head already existed in `train.py`.

The reasoning: the MLP sees the frames as one vector and has no notion of
order, so it cannot separate "the phrase, in order" from "the phrase's
phonemes, some order" — which is exactly what the adversarial and speech
negatives are. Verified locally before spending that `export_onnx` handles the
LSTM at opset 17 with a dynamic batch axis, agreeing with PyTorch to 6e-8.

### `runs/e2-rnn-v2b` — on the v2b features e1b was trained on

| upstream set, recall at FA/h | 0.2 | 0.5 | 1.0 | 2.0 |
| --- | --- | --- | --- | --- |
| e1b (promoted, DNN 128) | 0.536 | 0.615 | 0.659 | 0.708 |
| **e2-rnn-v2b** | **0.593** | **0.643** | **0.715** | **0.751** |

By condition at <= 1.0 FA/h: clean 0.837 -> 0.870, env 0.683 -> 0.727, music
0.678 -> 0.724, speech 0.545 -> 0.624, babble 0.506 -> 0.545. Every condition
gains; the stress grid is unchanged (1.0 in every cell 20–5 dB except
speech@10 at 0.958, one clip; speech@0 0.917, babble@0 0.958). ACAV holdout a
wash. Trained at 56 steps/s against the DNN's 78.

**The hypothesised mechanism did not hold.** Training-score FA is 31.7/h against
e1b's 27/h, so the LSTM does not fire less on negatives at 0.5. It ranks
positives higher relative to the same negatives, which is what the matched-FA
metric rewards. `final` is much worse than `best_single` at 0.2 FA/h (0.433), so
checkpoint selection matters more for this head than for the DNN.

### Promoted 2026-09-04 as `iamtokarev/hey-sonny@ea30342b`

Job B rerun with e1b's flags plus `--model-type rnn` and `--repo-id
iamtokarev/hey-sonny`, 1h 12m on `t4-small`. Same seed and data, so it
reproduced `e2-rnn-v2b` exactly — 0.5925 / 0.643 / 0.7145 / 0.7515, selected
`best_single`, ONNX parity 1.3e-5, 724 KB self-contained. Its `metrics.json`
carries the replaced DNN model as `baseline`, so the comparison travels with the
model card. The previous revision `98bbc8b7` remains in the repo's history.

**Still open:** the live check (`try_model.py` at the 0.5 FA/h threshold, media
playing) has not been run against this revision.

---

## Budget

Approximate, from the job listing:

| Phase | Spend |
| --- | --- |
| E0 + E2 + probes (eleven Job B runs) | ~$1.30 |
| E1 (two Job A pools, two Job B runs) + promotion | ~$0.90 |
| E3 (three Job A attempts, two Job B runs) | ~$2.40 |
| E4 (one Job B run) + promotion | ~$0.90 |
| Preflight and rehearsals | ~$0.15 |
| **Total** | **~$6** |

Roughly a third of E3's spend was two failed Job A attempts. Every Job A or
Job B change gets a rehearsal first — `--steps 400 --refine-steps 100
--limit-fp-frames 60000` for Job B, `--n-train 60 --n-val 20` for Job A — a
pattern that has caught eight faults for under a dollar total.

## Not attempted, and why

- **Enlarging the negative bank** (2,000 h -> more). It targets false accepts,
  which are already near target, and it trades recall down — the wrong direction
  for this failure mode. It is also the most expensive item: no prepared feature
  file exists past ACAV100M, so it needs a new Job A stage over raw corpora.
- **A custom verifier model.** Upstream's own answer for a low false-accept rate
  in a personal deployment, and it needs no GPU — but it is speaker-specific and
  needs recordings of the user's own voice. It belongs after the base model
  settles, and it is a different deliverable.
