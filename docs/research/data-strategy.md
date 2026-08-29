# Data Strategy for “Hey Sonny”

Research snapshot: 2026-08-24.

## Scope

This note defines the smallest practical data plan for training a personal
“Hey Sonny” model with openWakeWord. It covers data sources, generated data,
augmentation, separation of training and evaluation data, and the licensing
and provenance constraints that affect this project. It does not define model
metrics or deployment-specific tests.

The automated openWakeWord recipe requires five data roles: synthetic target
examples, synthetic adversarial examples, room impulse responses and background
audio, general negative data, and validation data. The current training script
creates separate synthetic train and validation clips, while its supplied
false-positive set is consumed during training for model selection. Sources:
[automatic training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb),
[current training script](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/train.py).

## Proposed data categories

| Category | Role | Default source |
|---|---|---|
| Synthetic positive training audio | Teach the exact “hey sonny” phrase | Piper LibriTTS-R multi-speaker generation; VCTK and L2-ARCTIC as added speaker inventories |
| Synthetic adversarial negatives | Teach close but unwanted phrases and partials | openWakeWord phoneme-overlap generation plus a short manual list |
| General negative training features | Provide large-scale ordinary speech, music, and noise | `davidscripka/openwakeword_features` ACAV100M features |
| Augmentation audio | Mix realistic environmental sound into synthetic clips | License-filtered FSD50K subset; add music only if experiments show it is needed |
| Room impulse responses | Simulate different acoustic spaces | MIT environmental impulse responses used by the upstream recipe |
| Training validation data | Early stopping and candidate selection | Disjoint synthetic clips plus the upstream false-positive validation features |
| Strict held-out evaluation data | Measure final generalization without leakage | Separately recorded positive, confusing-phrase, and long negative audio |

## Source and generation choices

### 1. Positive “Hey Sonny” examples

Use the Piper/LibriTTS-R path already assumed by the openWakeWord automated
pipeline. The maintained Piper sample generator supports a 904-speaker English
generator, speaker-embedding mixtures, several speaking speeds, and stochastic
generation. openWakeWord’s own synthetic-data guidance identifies multi-speaker
speech and random variability as the two important properties for robust
synthetic wake-word data. Sources: [Piper sample generator](https://github.com/rhasspy/piper-sample-generator),
[openWakeWord synthetic-data guidance](https://github.com/dscripka/openWakeWord/blob/main/docs/synthetic_data_generation.md),
[LibriTTS-R Piper model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/libritts_r/medium/MODEL_CARD).

Generate the exact written phrase `hey sonny` across many speakers, speaker
mixtures, generation seeds, and the upstream speaking-speed values. Do not add a
second TTS system initially; the first pilot should establish whether Piper says
“Sonny” consistently and naturally. Listen to a stratified sample before scaling
and reject silent, clipped, severely distorted, mispronounced, or incomplete
clips.

**The real fork is the `.pt` generator path versus the `.onnx` voice path, not
LibriTTS-R versus “Piper voices”.** LibriTTS-R *is* a Piper voice, published in
the standard `rhasspy/piper-voices` inventory in both formats. The `.pt` file is
the VITS training checkpoint, so it exposes the speaker embedding table and
`generate_audio` can interpolate between two speakers; the exported ONNX has no
such handle, and `generate_samples_onnx` iterates discrete speaker IDs with no
`slerp_weights` and no batching. The `.pt` path was chosen for SLERP blending and
GPU batching, and that choice stands.

The speaker inventory is the reason the choice matters. Of 38 English Piper voice
models, 31 are single-speaker; downloading every one of them yields 31 speakers.
LibriTTS is the only place a large speaker count exists. Source:
[`voices.json`](https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json).

### 1a. Added speaker inventories: VCTK and L2-ARCTIC

In scope as of 2026-08-29. Both are Piper voices in the same VITS family, so they
are a second *speaker inventory* rather than a second synthesis architecture —
the distinction
[synthetic-real-data-mixing.md](synthetic-real-data-mixing.md) identifies as the
one that matters. Neither reopens the settled multi-TTS question.

| Model | Speakers | Adds |
|---|---|---|
| `en_GB-vctk-medium` | 109 | British accents, absent from LibriTTS-R’s US audiobook readers |
| `en_US-l2arctic-medium` | 24 | Non-native English (Hindi, Korean, Mandarin, Spanish, Arabic, Vietnamese L1) |

L2-ARCTIC is the more valuable of the two for this project, because the target
speaker’s accent may not be represented anywhere in an American or British voice
inventory, and no amount of LibriTTS-R speaker blending creates a non-native
accent.

Two implementation constraints follow from the code. `main()` rejects a mixed
model list — models must share one file suffix, and only one `.pt` generator is
supported — so these voices cannot be added to the existing LibriTTS-R call.
They run as a separate `.onnx` invocation, which does accept several `--model`
arguments at once because both share that suffix, and the output directories are
merged afterwards. That invocation has no SLERP blending, so its diversity is
capped at 133 discrete speakers rather than a blended space.

Sequencing: generate these only after the LibriTTS-R-only candidate has a
held-out score, and add them as a measured comparison against that baseline on
the same frozen evaluation revision. Adding them to the first run would make the
baseline unmeasurable, and the open question is whether a wider speaker inventory
helps at all — which cannot be answered without the comparison.

**Speaker traversal is the real limit, not the 904 speaker count.** Verified
2026-08-27 against piper-sample-generator 3.2.0. `generate_samples` iterates
`itertools.product(range(num_speakers), range(num_speakers))` and SLERP-blends
each *pair* of speaker embeddings, so every clip is a mixture of two speakers
and no individual speaker can be pinned. Because that product is ordered, the
first `num_speakers` clips all pair speaker 0 with someone else. At the
30,000-clip target with the full 904 speakers, the first element of the pair
never advances past speaker 32: the run explores a narrow anchored slice of the
embedding space while appearing to use all 904 voices.

Cap `max_speakers` so the `n²` pair space is small enough to traverse
completely, choosing `n` near the square root of the clip target. For 30,000
clips, roughly 100–170 speakers gives at least one full pass over all pairs
instead of a single anchored row. The upstream README separately recommends a
value below 904 because the least-represented LibriTTS-R speakers produce
artifacts, so both reasons point the same way. Note also that the generation
settings iterator advances once per *batch*, not per clip, so a large GPU batch
size makes every clip in that batch share one length and noise scale; keep the
batch small relative to the number of setting combinations. Source:
[`generate_samples` speaker and settings iteration](https://github.com/rhasspy/piper-sample-generator/blob/master/piper_sample_generator/__main__.py).

`notebooks/piper_exploration.ipynb` exercises this path at exploration scale and
reconstructs a per-clip manifest of speaker pair and generation settings, which
the generator itself does not record.

For scale, follow the upstream minimum of 20,000 positive training examples for
the first serious candidate; 30,000 is a sensible starting target already used
in this project’s compute plan. A 1,000-example pilot is sufficient to validate
generation, augmentation, and persistence, but not model quality. Source:
[documented custom-model configuration](https://github.com/dscripka/openWakeWord/blob/main/examples/custom_model.yml).

### 2. Generated hard and confusing negatives

Keep both mechanisms in the upstream pipeline:

1. openWakeWord’s English phoneme-overlap generator, which creates similar
   words, partial phrases, and examples containing individual input words.
2. A small manual list chosen for this phrase: `sonny`, `hey honey`,
   `hey johnny`, `hey siri`, `hi sonny`, and `okay sonny`.

The automatic generator deliberately excludes homophones because they are not
acoustically distinguishable from the target. For the same reason, do **not**
label “hey sunny” as a negative: it conflicts with “hey sonny” at the audio
level. Treat that homophone as an unavoidable activation unless later evidence
shows a reliable contextual distinction. Source:
[adversarial generation and use in `train.py`](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/train.py).

Generate adversarial clips with the same speaker and speed diversity as positive
clips. Keep their base text in the manifest so false activations can later be
grouped by phrase. Expand the manual list only from observed false activations;
a large speculative phrase list would increase cost and can teach unintended
boundaries.

### 3. General negative training data

Use the upstream precomputed ACAV100M openWakeWord feature array rather than
building a new large negative corpus. It represents about 2,000 hours of
multilingual speech, music, noise, and real-world sound, already encoded in the
feature format consumed by custom models. The dataset is feature-only, so it is
efficient for training but cannot be listened to or relabeled. Source:
[`davidscripka/openwakeword_features` dataset card](https://huggingface.co/datasets/davidscripka/openwakeword_features/blob/main/README.md).

Use a deterministic subset for pipeline pilots and the full array for the first
serious candidate. Do not add Common Voice, podcasts, or another general speech
corpus until evaluation identifies a concrete gap; the supplied features already
cover the baseline role at useful scale.

### 4. Background audio for augmentation

The upstream notebook demonstrates AudioSet and the Free Music Archive (FMA),
and suggests FSD50K as another source. For this project, prefer a curated FSD50K
subset containing environmental sound, household noise, music, and human
non-speech. Select only clips under CC0 or CC-BY for simpler reuse, retain the
original uploader and license fields, and exclude any clip that contains the
target phrase or hard-negative test phrases. FSD50K contains over 100 hours of
human-labeled sound and publishes per-clip licenses and uploader metadata.
Source: [official FSD50K release](https://zenodo.org/records/4060432).

FMA is optional if music proves underrepresented. Its metadata is CC BY 4.0,
but each audio track retains the artist-selected license, so any selected track
must carry its own license and attribution. Source:
[FMA repository and license notes](https://github.com/mdeff/fma).

Do not make a downloaded-audio AudioSet mirror the default. Google’s official
AudioSet release provides YouTube segment identifiers and precomputed features;
the clips originate in YouTube videos. A third-party archive of extracted audio
therefore has a less straightforward redistribution position than FSD50K.
Source: [official AudioSet download documentation](https://research.google.com/audioset/download.html).

### 5. Room impulse responses

Use the MIT environmental impulse responses referenced by the openWakeWord
automatic notebook. The collection contains 271 real-world impulse responses
measured across everyday spaces and is small enough to keep with the reproducible
input cache. Sources: [MIT IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html),
[openWakeWord download recipe](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb).

The MIT page and Hugging Face mirror do not state an explicit data license.
Cache and use the files privately, record their source and checksum, and do not
republish them in a project Dataset repository until redistribution permission
is clarified.

## Augmentation role and bounds

Use augmentation to vary acoustics, not to replace speaker or phrase diversity.
The current openWakeWord implementation can apply EQ, distortion, pitch shift,
band-stop filtering, colored noise, background mixing, gain, and RIR convolution,
then produces fixed-length 16 kHz clips. Source:
[`augment_clips` implementation](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/data.py).

Start with one augmentation round and preserve the unaugmented base-clip ID for
every derivative. Do not multiply the dataset until a pilot confirms that
speech remains intelligible and the wake phrase is intact. The current code’s
background mixer spans a relatively aggressive -10 to 15 dB SNR range; compare
the upstream default with one more conservative pilot rather than assuming more
noise is always better. A current upstream report associates the documented
mixing range with poor class separation, but it is not yet a resolved upstream
finding, so this remains an empirical question for the project. Sources:
[`data.py`](https://github.com/dscripka/openWakeWord/blob/main/openwakeword/data.py),
[upstream issue #335](https://github.com/dscripka/openWakeWord/issues/335).

## Split and leakage policy

Use three distinct roles:

- **Training:** generated positives, generated adversarial negatives, general
  negative features, and their augmentations.
- **Training validation:** separately generated positive/adversarial clips and
  the upstream 11-hour false-positive feature set. Because `auto_train` uses
  these inputs for early stopping and model selection, neither is a final test.
- **Strict held-out evaluation:** separately recorded “Hey Sonny” utterances,
  confusing phrases, and long-form negative audio from speakers and recording
  sessions absent from training. Do not use this set to select checkpoints,
  tune augmentations, add hard negatives, or choose the operating threshold.

Split base material before augmentation, and keep every derivative of a base
clip in the same split. Use disjoint generation seeds and, where the generator
allows it, disjoint speaker or embedding pools for synthetic train and
validation clips. Because both still come from the same TTS family, treat the
strictly held-out real recordings as the meaningful generalization test.

For duplicate control:

- Compute SHA-256 for every original and generated base clip.
- Assign a stable `base_clip_id`; augmentation outputs inherit it.
- Reject exact duplicates across splits.
- Before freezing the held-out set, run an audio fingerprint or embedding
  similarity check against training sources to find re-encoded near-duplicates.
- Never move a failed held-out example into training while continuing to report
  results against that same held-out version; create a new evaluation revision.

## Provenance, privacy, and licensing flags

Every manifest row should record at least: category and label, exact phrase when
applicable, source repository and revision, source item ID, original uploader
and license where applicable, file checksum, generator/model revision,
generation and augmentation seeds, `base_clip_id`, and split.

| Source/artifact | Current flag for this project |
|---|---|
| Piper sample-generator code | MIT licensed; pin its revision. |
| LibriTTS-R Piper voice/generator | Model card identifies its training dataset as CC BY 4.0; record model revision and attribution. Keep generated clips private until output-redistribution terms have been reviewed. |
| openWakeWord ACAV100M and validation features | Dataset card is CC BY-NC-SA 4.0. Suitable for this private, non-commercial project; do not assume a commercially usable or permissively redistributable downstream dataset/model. |
| FSD50K background clips | Dataset curation is CC-BY, while audio licenses vary per clip. Default to CC0/CC-BY clips and retain attribution metadata. |
| FMA audio | Per-track artist licenses; not one blanket audio license. Optional, and only with per-track filtering and attribution. |
| MIT environmental RIRs | No explicit license found on the source or mirror. Use privately; do not redistribute without clarification. |
| Personally recorded evaluation audio | Treat as private by default because it contains identifiable voices and potentially ambient household speech. Obtain consent from every speaker and avoid recording third-party conversations. |

The TTS code license, voice-model license, generated-audio status, source feature
license, and final model license are separate questions. A private personal-use
workflow avoids unnecessary publication risk, but it does not erase attribution
or non-commercial conditions.

## Settled: one TTS family, no owner voice in the first candidate

Decided 2026-08-28. Piper/VITS remains the only TTS family, and the first serious
candidate trains on synthetic audio alone. This closes the previously open
question about adding a second TTS family. It does not constrain the speaker
inventory within that family: VCTK and L2-ARCTIC were added to scope on
2026-08-29 (§1a) precisely because they widen speaker coverage without
introducing a second synthesis architecture.

The evidence is in
[synthetic-real-data-mixing.md](synthetic-real-data-mixing.md). In short: the
only direct single-TTS versus multi-TTS comparison found in keyword spotting is
null, because the shortcut a model learns is “generated versus recorded” rather
than any individual generator’s signature — a property a second generator
shares. Kokoro is a poor specific choice regardless, offering 28 English voices
against LibriTTS-R’s 904 blended embeddings and a documented weakness on short
utterances. Real recorded audio is the lever that moves false-reject rate, but
it is a second revision: it only pays off with session-matched real negatives,
which do not exist yet.

Two consequences for this note. Augmentation is not a substitute for recording
sessions when real audio is eventually added, contrary to what the augmentation
section might suggest for synthetic clips. And Kokoro, if used at all later,
belongs in *evaluation* as a generator absent from training, not in the training
mix.

## Recommended default plan

1. Generate and audit a 1,000-positive/1,000-validation Piper pilot for the
   exact phrase `hey sonny`.
2. Generate an equal-order adversarial set using openWakeWord’s phonetic
   generator plus the six explicit phrases above.
3. Use a deterministic subset of the upstream ACAV100M features and its 11-hour
   false-positive validation features for the pilot.
4. Build one license-filtered FSD50K background subset and use the upstream MIT
   RIR collection privately; do not add FMA or raw AudioSet initially.
5. Compare one upstream augmentation run with one conservative-background-mix
   run, listening to samples before feature extraction.
6. After the pipeline passes, scale positives to 20,000–30,000 and use the full
   ACAV100M feature array.
7. Freeze an independently recorded held-out evaluation revision before final
   model selection. Keep it outside training storage paths and manifests.
8. Score the LibriTTS-R-only candidate, then generate a VCTK + L2-ARCTIC set via
   the `.onnx` path (§1a) and score a second candidate against the same frozen
   evaluation revision. Keep the two candidates otherwise identical so the
   speaker-inventory effect is the only variable.

## Unresolved empirical questions

- Does Piper pronounce “Sonny” correctly across its speaker range, and which
  speaker embeddings generate unusable artifacts?
- Does capping `max_speakers` for full pair coverage improve held-out recall
  over leaving it unset, or is the anchored slice of the embedding space already
  diverse enough because every clip is a two-speaker blend?
- Does adding the VCTK and L2-ARCTIC speaker inventories improve held-out recall
  over LibriTTS-R alone, and is the gain worth losing SLERP blending for those
  133 speakers? Planned as step 8 of the default plan.
- Do any of L2-ARCTIC’s 24 non-native speakers actually resemble the target
  speaker’s accent, or is the corpus’s L1 coverage the wrong set? This is
  answerable by listening before any training run.
- Which explicit confusing phrases actually cause false activations beyond the
  automatic phonetic negatives?
- Does the upstream -10 to 15 dB background-mixing range help or harm this short
  phrase compared with a more conservative range?
- Is the license-filtered FSD50K subset sufficient, or does evaluation show a
  specific need for more music, television, or conversational backgrounds?
- How large and speaker-diverse must the strictly held-out real positive set be
  before the evaluation metrics stabilize? This should be decided in the
  evaluation-methodology track.
