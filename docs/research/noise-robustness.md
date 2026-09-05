# Why the model failed when other people talk, and what fixed it

Measured 2026-09-02 against `iamtokarev/hey-sonny@e537d70b`, which worked in a
quiet room and stopped working with music or conversation in it. Fixed
2026-09-03. Everything below came from
[`scripts/stress_test.py`](../../scripts/stress_test.py); re-run it to
reproduce any number here.

## The measurement

Five wake-word clips from macOS `say` voices the model has never seen, each
mixed with one kind of interference at one signal-to-noise ratio, with a second
of the same interference before and after the phrase. The table shows the
**lowest peak score across the five voices** and the strictest measured
false-accept budget that minimum still clears (thresholds from `metrics.json`:
0.9990 / 0.9972 / 0.9853 / 0.9577 for 0.2 / 0.5 / 1.0 / 2.0 FA/h).

| Interference | In Job A's pool? | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB | -5 dB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| environmental sound (FSD50K) | yes, 1,000 clips | 1.000 | 1.000 | 1.000 | 0.996 | 1.000 | 1.000 |
| white noise | no | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 0.181 |
| pink noise | no | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.101 |
| music (FMA) | yes, 200 clips | 1.000 | 0.967 | **0.210** | **0.000** | **0.001** | 0.001 |
| **speech babble** (5 talkers) | **no** | 0.996 | **0.024** | **0.026** | **0.135** | **0.001** | 0.001 |

Each cell draws a different random segment of the interference, so single cells
move between runs — a second run with a fresh seed put one voice at 0.17 under
babble at **20 dB** and music at 0.08 at 10 dB but 0.9999 at 5 dB, depending on
whether the segment had vocals. Read the rows, not the cells: over both runs
babble never cleared any budget at 15 dB or below, and environmental sound,
white and pink noise never failed above -5 dB.

Reverberation through eight of the MIT room impulse responses the model trained
with: every voice above 0.996, and 0.795 at worst with 10 dB of environmental
noise on top. Interference on its own (60 s each of babble, music, pink, white)
never exceeds 0.18, so none of it causes false accepts at any budget.

Read across a row and the pattern is not "noise". The model was robust to what
it trained on in volume, robust even to stationary noise it never saw, and
**collapsed under other speech**: at 15 dB — conversation across the room —
one voice in five already scored 0.02, and by 0 dB every voice was at the
floor. Music, which carries vocals and speech-like harmonic structure, failed
the same way one step later.

The scores did not degrade toward 0.5; they went to 0.001. The model was not
uncertain about a wake word under speech, it was *confident it was a negative*.

## Why: the training data taught exactly this

Three facts, each checked in the code or the manifest:

1. **The positives were never mixed with speech.** Job A's augmentation pool
   was the two openWakeWord sample zips: 1,000 FSD50K environmental clips and
   200 FMA music clips, 4.0 hours, 1,197 files (`manifest.json` of
   `iamtokarev/hey-sonny-features`). No speech corpus. This was a design
   decision — the early data plan specified "environmental sound, household
   noise, music, and human *non-speech*" for the augmentation pool.
2. **The negatives are overwhelmingly speech.** ACAV100M is a speech-heavy
   corpus, it supplies 1,024 of every 1,124 training examples, and its loss is
   weighted up to 500x. Every training batch said: *overlapping speech,
   label 0*.
3. **Nothing in the batch said otherwise.** The only examples with label 1
   were a single voice over environmental sound, music, or silence.

The cheapest rule separating the two classes is "is there more than one
voice?", and the model learned it. That is also why validation could not see
the failure: `positive_val` was augmented from the same speech-free pool, so
recall 0.563 at 0.2 FA/h was measured over conditions the model handled and
said nothing about the one it did not.

Upstream avoids this in all three of its recipes; the omission was ours:

| Recipe | What positives are mixed with | Source |
| --- | --- | --- |
| Released `hey_jarvis` | **ACAV100M** (speech), FSD50K, FMA at 0–20 dB | [`docs/models/hey_jarvis.md`](https://github.com/dscripka/openWakeWord/blob/368c037/docs/models/hey_jarvis.md) |
| Tutorial notebook | FMA, FSD50K, **Common Voice 11** at 5–15 dB | [`training_models.ipynb`](https://github.com/dscripka/openWakeWord/blob/368c037/notebooks/training_models.ipynb) cell 12, 19 |
| Auto-trainer | **AudioSet** (YouTube; speech is its largest class), FMA | [`automatic_model_training.ipynb`](https://github.com/dscripka/openWakeWord/blob/368c037/notebooks/automatic_model_training.ipynb) cell 9, 14 |

Music was a weaker version of the same gap: it *was* in the pool, but at 200
files against 1,000. A role's share of the augmentation draws is its share of
the **file count**, not of its hours — see the per-clip draw in
[upstream-recipe-comparison.md](upstream-recipe-comparison.md) — so the pool
was 17% music, 83% environmental, 0% speech. File-count balance is what the
corrected Job A controls.

## What does not fix it

Checked, so nobody spends time on them:

- **A lower threshold.** Under 10 dB babble the failing voices scored 0.026 and
  0.061. There is no threshold that admits those and rejects anything.
- **A custom verifier model.** It only runs when the base score clears
  `custom_verifier_threshold`, so it cannot rescue a score of 0.02. It remains
  upstream's answer to false accepts, not to this.
- **Silero VAD** (`vad_threshold`). It can only zero predictions.
- **Speex noise suppression.** Linux-only per the README, and it targets
  stationary noise, which the model already handled to 0 dB.
- **Reverberation, capacity, steps, loss weighting.** Reverb passes; the other
  three were swept in [`experiment-plan.md`](../experiment-plan.md) and moved
  recall on the speech-free validation set from 0.379 to 0.564 — the ceiling of
  tuning a model on data that omits the failing condition.

## The fix, and what came after

Two changes, run as `e1-speech-v2` and `e1b-snr0-20` (results in
[experiment-plan.md](../experiment-plan.md)):

1. **Speech in the pool, balanced by file count.** Four roles — speech
   (LibriSpeech), synthetic multi-talker `babble` of 2–5 overlaid speech
   segments, music, environmental sound — re-cut to 10 s segments and
   truncated to an equal count each, with `--augmentation-rounds 3` so every
   positive is seen under three draws.
2. **The mixing floor raised from -10 dB to 0 dB**, the released models' range.
   Speech alone was not enough: at -10 dB a positive is buried under babble ten
   times louder than the phrase, which taught the model to fire on speech
   generally — training false accepts 58/h against 27/h with the floor.

After both, the held-out stress grid reads 1.00 under speech, babble, music and
environmental sound from 20 dB down to 5 dB at the 1.0 FA/h threshold, where
the old model read 0.5–0.6 under speech. The "confident negative" failure is
gone.

Two further levers were tried on the corrected pool (`experiment-plan.md` E3
and E4). The recurrent head gained 0.056 mixed recall at 1.0 FA/h with every
condition up, and was promoted 2026-09-04 as
`iamtokarev/hey-sonny@ea30342b`. 100k positives with 20% custom near-miss
negatives at the same step budget lost 0.04–0.05 with either head; that change
needs its confounds separated before it is tried again.

## Primary sources

- [`openwakeword/data.py`](https://github.com/dscripka/openWakeWord/blob/368c037/openwakeword/data.py) — `augment_clips`: probabilities, `per_batch` mode, the -10..15 dB range, the RIR guard outside the clip loop
- [`openwakeword/model.py`](https://github.com/dscripka/openWakeWord/blob/368c037/openwakeword/model.py) — `predict`: where the verifier, VAD, patience and Speex hooks sit and what each can do to a score
- [`docs/custom_verifier_models.md`](https://github.com/dscripka/openWakeWord/blob/368c037/docs/custom_verifier_models.md) — the verifier's trigger threshold and data recommendations
- [`README.md` §Noise Suppression and VAD](https://github.com/dscripka/openWakeWord/blob/368c037/README.md#noise-suppression-and-voice-activity-detection-vad) — platform limits of Speex; the 5–10 dB false-reject measurement recommendation
