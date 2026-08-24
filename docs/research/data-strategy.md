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
| Synthetic positive training audio | Teach the exact “hey sonny” phrase | Piper LibriTTS-R multi-speaker generation |
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

## Unresolved empirical questions

- Does Piper pronounce “Sonny” correctly across its speaker range, and which
  speaker embeddings generate unusable artifacts?
- Is one TTS family sufficiently diverse, or does held-out recall justify adding
  a second generator later?
- Which explicit confusing phrases actually cause false activations beyond the
  automatic phonetic negatives?
- Does the upstream -10 to 15 dB background-mixing range help or harm this short
  phrase compared with a more conservative range?
- Is the license-filtered FSD50K subset sufficient, or does evaluation show a
  specific need for more music, television, or conversational backgrounds?
- How large and speaker-diverse must the strictly held-out real positive set be
  before the evaluation metrics stabilize? This should be decided in the
  evaluation-methodology track.
