# Mixing Synthetic and Real Speech for “Hey Sonny”

Research snapshot: 2026-08-28.

## Scope

This note answers one open question from
[`data-strategy.md`](./data-strategy.md): should the positive training set be a
mixture of Piper/LibriTTS-R speech, Kokoro-82M speech, and the project owner’s
own recorded voice, or is one TTS family plus strictly held-out real audio the
better plan?

It covers seven sub-questions: whether TTS-*system* diversity is a real axis
distinct from speaker diversity; how far models trained on synthetic speech
transfer to real speech; what a single speaker’s real recordings buy; what
mixing ratios the literature reports; whether a single-microphone recording set
teaches channel identity instead of the phrase; what happens when the target
accent is outside the TTS inventory; and what mixed data does to the held-out
evaluation set.

It does not revisit augmentation bounds, licensing, negative-data sourcing, or
the metric definitions — those stay in
[`data-strategy.md`](./data-strategy.md) and
[`evaluation-methodology.md`](./evaluation-methodology.md). It does not propose
architecture changes; the classifier stays the upstream DNN head described in
[`openwakeword-baseline.md`](./openwakeword-baseline.md).

Almost all quantitative evidence below comes from open-vocabulary or
multi-keyword KWS systems trained at industrial scale. This project is a
single-phrase personal detector with a 32-unit head on a frozen embedding.
Where a number is transferred across that gap, the text says so.

## Summary of the evidence

| Axis | Direction and size | Strength for this project |
|---|---|---|
| Real vs. synthetic positives | 46.47% → 2.46% FRR at 0.133 FA/h when real data is added | Direct, large |
| Speaker diversity within one TTS | 15.28% → 7.63% FRR going from 1 to 500 real speakers; speakers beat utterances-per-speaker | Direct, large |
| Phrase/text diversity | EER 34.42% → 12.62% going from 500 to 38k synthesized phrases | Large, but open-vocabulary only |
| Frozen pretrained embedding | Cuts the synthetic-to-real accuracy gap from ~32 points to ~2.7 | Direct; this *is* openWakeWord |
| TTS-system identity | 46.50% (best single) vs. 46.47% (two systems combined) | Direct, and essentially null |

## 1. TTS-system diversity is the weakest of the diversity axes

The most direct evidence is [Park et al., *Utilizing TTS Synthesized Data for
Efficient Development of Keyword Spotting
Model*](https://arxiv.org/abs/2407.18879) (Google, SynData4GenAI workshop at
Interspeech 2024). They trained a 320k-parameter “Hey/OK Google” detector on
synthetic data from two architecturally unrelated systems — Virtuoso, a
multilingual speech-text joint model with 726 speaker profiles, and a variant
of AudioLM, an audio language model conditioned on reference audio — and
reported false-reject rate at a fixed 0.133 false accepts per hour.

| Training data | FRR |
|---|---|
| Virtuoso only | 53.10% |
| AudioLM only | 46.50% |
| Virtuoso + AudioLM | 46.47% |
| Virtuoso + real negatives | 17.75% |
| AudioLM + real negatives | 16.59% |
| Virtuoso + AudioLM + real negatives | 17.94% |
| Real data only | 3.17% |

Combining the two systems moved FRR from 46.50% to 46.47% — inside noise. Once
real negative data was added, the two-system mixture (17.94%) was *worse* than
the better single system alone (16.59%). Source:
[Table 4](https://arxiv.org/abs/2407.18879).

This is one keyword, one architecture, one pair of TTS systems, and the paper
does not frame it as a study of system diversity, so it is not a general law.
But it is the only published head-to-head of single-TTS versus multi-TTS
training sets found for KWS, and it does not support the hypothesis. Nothing
found in the search asserts the opposite for KWS.

The axes that the same body of work shows *do* pay:

- **Speaker diversity.** Holding TTS data fixed and adding real positives at 10
  utterances per speaker, FRR fell 15.28% → 14.94% → 9.78% → 9.90% → 7.63% for
  1, 10, 100, 200, and 500 speakers. Holding speakers fixed at 100 and raising
  utterances per speaker from 2 to 200 moved FRR only 10.99% → 7.99% over a
  100× data increase. The paper concludes speaker count matters more than
  utterances per speaker. Source: [Table
  5](https://arxiv.org/abs/2407.18879).
- **Phrase/text diversity.** [Synth4Kws](https://arxiv.org/abs/2407.16840)
  (Google, same workshop) raised unique synthesized phrases from 500 to 38,000
  at a fixed 100 samples per phrase and saw EER fall 34.42% → 12.62% and DET
  AUC 28.90% → 5.93%. This axis mostly does **not** apply here: Synth4Kws is
  user-defined KWS over many keywords, while this project trains one fixed
  phrase. Its analogue here is the adversarial-negative phrase list, not the
  positives.

openWakeWord’s own guidance ranks the axes the same way. Its
[synthetic-data documentation](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/docs/synthetic_data_generation.md)
names exactly two properties as important — “random variability in the
generated speech” and “multi-speaker models” — and spends its diversity section
on high sampling temperatures and spherical interpolation of speaker
embeddings. TTS-system count is not one of its criteria, even though upstream
happened to build on two systems (NVIDIA WaveGlow and VITS).

### Do models learn TTS artifacts as a shortcut?

Yes, demonstrably. [Park et al., *Adversarial training of Keyword Spotting to
Minimize TTS Data Overfitting*](https://arxiv.org/abs/2408.10463) attached a
synthetic/real discriminator to the hidden activations of the same 320k KWS
model. Reading all encoder and decoder layers, it classified synthetic vs. real
with **98.1% accuracy**; even a single encoder layer read alone reached
89.7–96.0%. The authors conclude “there are relatively simple features in the
audio that can differentiate real from synthetic audio.” Suppressing that
information with a gradient-reversal layer improved real-speech FRR by up to
12% relative — and by up to 8% relative even with *no* real positive data at
all, using real negatives as the only real-domain contrast.

[Quintas et al.](https://arxiv.org/abs/2409.12745) corroborate the mechanism at
the representation level: a 2-D PCA of WavLM features separates real from
XTTS-v2 synthetic Speech Commands audio almost linearly, which they call
unexpected given how well those features serve KWS.

Note carefully what this does *not* say. It says a model trained on synthetic
positives learns features specific to *synthesis*. It does not say that adding
a second synthesizer removes them — and the Table 4 result above suggests it
largely does not, because the shortcut being learned is “generated” versus
“recorded,” a property both generators share. The reported cure is real audio
on the other side of the boundary, not more generators.

## 2. The synthetic-to-real gap, and why this project’s architecture shrinks it

The gap is large when a model is trained end-to-end on low-level features and
small when a small head sits on a frozen pretrained embedding. This distinction
matters more than anything else in this note, because openWakeWord *is* the
second case.

[Lin et al., *Training Keyword Spotters with Limited and Synthesized Speech
Data*](https://arxiv.org/abs/2002.01322) (ICASSP 2020) trained on 3,220
Tacotron-2 clips across 92 voices and evaluated on real Speech Commands audio:

| Model | Synthetic only | Equal real data | Full real set (80k) |
|---|---|---|---|
| Full model (~400k params, trained on log-mel) | 56.7% | 88.7% | 97.4% |
| Head model on frozen speech embedding | 92.6% | 95.3% | 97.7% |

Training on synthetic alone costs 32.0 points for the full model and 2.7 points
for the head model. The paper’s conclusion is blunt: “While synthesized speech
does not provide enough useful information to train a full keyword spotter, it
can be used to train a head model on top of a well trained speech embedding
model.”

That embedding is not an analogy for openWakeWord’s — it is the same lineage.
Lin et al. describe an embedding producing “96-dimensional feature vectors, one
every 80 ms,” published at `tfhub.dev/google/speech_embedding/1`. The
[openWakeWord baseline note](./openwakeword-baseline.md) records exactly a
96-wide embedding evaluated at 80 ms intervals with a small DNN head. The
single most reassuring result in this literature is measured on this project’s
own architecture.

Two corroborating results, both extrapolations from adjacent tasks:

- [Quintas et al.](https://arxiv.org/abs/2409.12745) (Speech Commands, XTTS-v2
  voice cloning over 174k Common Voice speakers): a 134k MatchboxNet trained on
  MFCCs scored 98.49% real vs. 92.57% synthetic (5.9 points), while a frozen
  WavLM plus a single linear layer scored 98.03% vs. 96.11% (1.9 points).
- [Hilmes et al.](https://arxiv.org/abs/2407.17997) (ASR, LibriSpeech): the
  real-to-synthetic dev-clean WER penalty was 1.9 points for GMM-HMM, 6.6 for a
  hybrid system, and 11.2 for an attention encoder-decoder. The more the model
  learns its own front end, the more it absorbs synthesis artifacts.

The same Quintas paper carries a warning that applies directly to Piper
generation quality. Unfiltered synthetic data (containing TTS hallucinations)
scored 90.02% with MatchboxNet but only **83.05%** with WavLM features;
ASR-based filtering of mis-synthesized clips recovered it to 96.11%. Bad clips
cost 2.5 points on a raw-feature model and 13 points on a frozen-embedding
model. For this project, that argues the stratified listening pass already
planned in [`data-strategy.md`](./data-strategy.md) is worth more than it
looks, and that an automatic reject filter would be worth more still.

## 3. What the project owner’s own voice buys

The literature offers one direct number for a single speaker. In [Park et
al.](https://arxiv.org/abs/2407.18879) Table 5, adding **one** real speaker with
**10** utterances to a full TTS training set moved FRR from 17.94% to 15.28% —
about 15% relative, from ten recordings. Ten speakers with 100 utterances
between them gave 14.94%, barely better. The curve only bends at 100 speakers
(9.78%).

Read carefully, that result cuts both ways for this project. It shows a
handful of real clips from one person is not nothing. It also shows that most
of the available gain is *speaker-diversity* gain, which one person cannot
supply. The project owner’s voice can move the model from 17.94% to roughly
15.28%; it cannot move it to 7.63%.

Whether that trade is good depends on a question the
[evaluation note](./evaluation-methodology.md) already flags as unresolved:
whether “Hey Sonny” is for one known speaker, a household, or arbitrary
speakers. If it is genuinely one person, deliberately fitting that person’s
voice is the objective, not a bug, and the speaker-independence penalty is
irrelevant. If a household is intended, a model tuned on one voice will
underserve the others and the held-out set must be able to detect that.

Two clarifications about what the personalization literature does and does not
cover:

- **“Personalization” in KWS usually means something else.** [Labrador et al.,
  *Personalizing Keyword Spotting with Speaker
  Information*](https://arxiv.org/abs/2311.03419) conditions the detector on a
  speaker embedding via FiLM rather than retraining on the target speaker’s
  audio, and reports gains concentrated “among underrepresented speaker
  groups.” The [Auto-KWS 2021 challenge](https://arxiv.org/abs/2104.00513)
  defines its task as a device that “can only be awakened by an enrolled
  speaker.” Query-by-example systems such as
  [DONUT](https://arxiv.org/abs/1811.10736) and [on-device
  QbE](https://arxiv.org/abs/1910.05171) match against enrollment audio in an
  embedding space. None of these train classifier weights on a handful of a
  single user’s clips, which is what this project would be doing. **No paper
  was found that directly measures adding one speaker’s home recordings to an
  otherwise synthetic single-phrase wake-word training set.** The Park Table 5
  row is the closest adjacent evidence and it is a single row.
- The gain in Park et al. is measured against a fixed synthetic corpus far
  larger and more diverse than this project’s. Extrapolating its magnitude to a
  30,000-clip Piper set is not supported.

## 4. Mixing ratios and the point of diminishing returns

Two independent sweeps show the same shape: the first slice of real data is
worth far more than the rest.

[Park et al., adversarial paper](https://arxiv.org/abs/2408.10463) Table 3
varies the sampling weight on a 3.8M-utterance real positive pool while holding
7.5M synthetic positives and 14.1M real negatives fixed:

| Real positive weight | Approx. real positives | FRR at 0.133 FA/h |
|---|---|---|
| 0% | 0 | 18.11% |
| 1% | ~38k | 6.83% |
| 5% | ~190k | 3.51% |
| 20% | ~760k | 2.38% |
| 100% | 3.8M | 1.81% |

The first 1% of the real data closes 69% of the total gap. Going from 1% to
100% — a hundredfold increase — closes the remaining 31%.

[Lin et al.](https://arxiv.org/abs/2002.01322) Figure 3 shows the mirror image
from the other end. Adding synthetic data on top of real data helped by 3.0
points absolute at 5 real examples per word, 2.5 points at 10, 0.1 points at
125, and produced “no further improvements” at 1,000 real examples per word.
Synthetic data is a substitute for real data you do not have, and its value
decays to zero as real data accumulates.

[Synth4Kws](https://arxiv.org/abs/2407.16840) adds the one documented case of
mixing actively hurting: against a 50k real-utterance baseline at 11.73% EER,
the best TTS mixture reached 8.19% EER (30.1% relative), but pushing past 50
TTS utterances per phrase degraded results, which the authors attribute to
“extra TTS data might overshadow the real data.” There is a ceiling, and it is
reachable.

**How few real recordings matter?** The honest answer from the literature is
that 10 measurably matter, 100 matter more, 1,000 across many speakers is where
the curve bends, and no source measures the 10-to-100 range for a single
speaker on a single phrase. For this project the binding constraint is not the
literature — it is that one person recording 1,000 utterances of one phrase in
one room supplies almost none of the diversity that made those numbers move.

## 5. The homogeneous-recording-condition trap

This is the strongest reason for caution, and it follows from the project’s own
data layout rather than from a paper about wake words.

If real recorded clips enter the training set *only* as positives, while every
negative comes from the ACAV100M feature array and Piper adversarial clips,
then “recorded on this microphone in this room” becomes a perfect predictor of
the positive label. A binary classifier has no reason to prefer the phrase over
the channel.

That failure mode is documented concretely, though in a neighbouring task.
[Müller et al., *Speech is Silver, Silence is Golden: What do ASVspoof-trained
Models Really Learn?*](https://arxiv.org/abs/2106.12914) (ASVspoof 2021 workshop)
found that in the official ASVspoof 2019/2021 data, leading-silence duration
correlated with the label; a model trained on **nothing but the duration of the
leading silence** reached up to 85% accuracy and 15.1% EER. The dataset
protocol, not the spoofing, was doing the work. A broader survey of the same
phenomenon in voice anti-spoofing and speaker verification is [Sahidullah et
al.](https://arxiv.org/abs/2601.17782) (IEEE JSTSP, 2026), and
[SpurAudio](https://arxiv.org/abs/2605.13672) shows few-shot audio classifiers
degrading sharply when background correlations are broken.

Transferring this to “Hey Sonny” is extrapolation — none of those papers is
about wake words. But the 98.1% synthetic/real discriminator accuracy from
[Park et al.](https://arxiv.org/abs/2408.10463) is measured *inside a KWS
model’s hidden layers*, on exactly the domain distinction at issue, so the
mechanism is established for this model class even if the specific channel
confound is not.

The mitigations follow directly from the mechanism:

1. **Record negatives in the same sessions as positives.** Hard negatives
   (`sonny`, `hey honey`, `hey johnny`, `hi sonny`, `okay sonny`), bare
   fragments, and ordinary conversation, captured on the same microphone in the
   same rooms. This is the single most important control: it breaks the
   correlation between channel and label instead of merely diluting it.
2. **Vary the channel deliberately.** Multiple microphones, multiple rooms,
   multiple distances, multiple days. Sessions recorded back to back on one
   device are one condition however many clips they contain.
3. **Apply the same augmentation pipeline to real and synthetic positives.**
   openWakeWord’s
   [`augment_clips`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/data.py)
   applies RIR convolution, background mixing, EQ, and gain. Augmenting only
   one class would replace the channel shortcut with an augmentation shortcut.
4. **Do not trust augmentation to fix a one-condition set.** Lin et al. tried
   pitch shift, tempo shift, reverberation, and white noise on their synthetic
   clips and saw “no significant improvements,” hypothesizing that the frozen
   embedding had already learned to absorb those distortions. Augmentation adds
   acoustic variation; it does not add a second microphone’s transfer function
   or a second speaker.

## 6. Accent coverage

Both candidate generators are American- and British-English only, and adding
the second one does not widen that.

- The Piper LibriTTS-R generator is `en_US`, built from
  [LibriTTS-R](https://arxiv.org/abs/2305.18802) (585 hours, 2,456 speakers,
  restored from LibriTTS). Piper voices are VITS models — its trainer lives
  under
  [`piper_train/vits/`](https://github.com/rhasspy/piper/blob/master/src/python/piper_train/vits/config.py).
- Kokoro-82M’s [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
  lists 20 American English voices (11F/9M) and 8 British (4F/4M) — 28 English
  voices total, versus LibriTTS-R’s 904 speaker embeddings with pairwise SLERP
  blending. On the diversity axis that the literature says actually matters,
  Kokoro is a large step *down*, not a sideways step.

That an accent gap costs recall is well established in aggregate. The
[“OK Aura” wake-word fairness study](https://arxiv.org/abs/2604.05830)
(SPEAKABLE workshop at LREC 2026) measured predictive disparity across sex, age
and accent and reduced it by 39.94%, 83.65%, and 40.48% respectively using
augmentation and knowledge distillation — a study only worth running because
the disparities were substantial to begin with. It reports disparity
*reductions*, not absolute per-accent FRR, so it establishes direction rather
than magnitude for this project.

The remedy the literature supports is not a second US/UK TTS system. It is
real audio from the target speaker — which is precisely the
[Labrador et al.](https://arxiv.org/abs/2311.03419) finding that personalization
gains concentrate “among underrepresented speaker groups.” If the project
owner’s accent is outside the LibriTTS-R distribution, that raises the value of
their own recordings and lowers the value of Kokoro, because Kokoro does not
cover the gap either.

## 7. Evaluation and leakage

Putting the owner’s voice in training changes what the held-out set can prove.
The existing policy in [`evaluation-methodology.md`](./evaluation-methodology.md)
already requires session-level separation for personal recordings; mixing real
audio into training makes that requirement load-bearing rather than
precautionary.

The reference practice is [Warden’s Speech Commands
dataset](https://arxiv.org/abs/1804.03209), whose splitter hashes the *speaker*
portion of each filename so that every clip from one speaker lands in one split.
The
[`which_set` implementation](https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/examples/speech_commands/input_data.py)
strips everything after `_nohash_` before hashing, so `bobby_nohash_0.wav` and
`bobby_nohash_1.wav` “are always in the same set.” Warden’s stated rationale is
that “keyword-spotting models are much more useful if they’re
speaker-independent.”

That exact rule cannot hold here: if the owner’s voice is in training, it cannot
also be a speaker-disjoint test. The workable substitute is to split one level
down and be explicit about what is being measured:

| Held-out subset | Disjoint from training by | Answers |
|---|---|---|
| Owner, unseen sessions | Session, room, microphone, day | Does it work for me tomorrow, elsewhere? |
| Other speakers | Speaker | Did fitting my voice cost speaker independence? |
| Long negatives | Session and source | Is FA/h still acceptable? |

Three rules make this meaningful:

1. **Split by session, never by clip.** Every clip from one recording session
   goes to exactly one role. A random clip split across a single session
   measures within-session memorization and will report inflated recall.
2. **Freeze sessions before recording more.** Decide which sessions are
   training and which are held out before the next session is recorded, so the
   choice cannot drift toward whatever makes the number look better.
3. **Keep an other-speaker subset even for a personal model.** It is the only
   instrument that can detect the speaker-overfitting this note warns about,
   and it costs a few recordings from a household member or friend. Without
   it, a model that has learned “my voice in my room” and a model that has
   learned “hey sonny” are indistinguishable.

The existing near-duplicate checks in
[`data-strategy.md`](./data-strategy.md) — SHA-256 on base clips, stable
`base_clip_id` inheritance through augmentation, fingerprint comparison before
freezing — apply unchanged and matter more once the same voice appears on both
sides.

## What the openWakeWord code actually does with a mixed dataset

Three implementation facts, verified against pinned commit `368c037`, decide
how a mixture can be assembled at all. All three are project-specific and none
appear in upstream documentation.

**Only the key named `positive` is labeled 1.** `train.py` builds its label
transforms by iterating over `["positive"] + feature_data_files.keys() +
["adversarial_negative"]` and assigns label 1 only when the key is literally
`"positive"`; every other key gets label 0. Adding real recorded positives as a
new entry in `feature_data_files` would therefore train them as **negatives**.
Real positives must either be merged into `positive_features_train.npy` before
`--train_model`, or the label transform must be patched. Source:
[`train.py` label transforms](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py#L838-L849).

**The batch generator reads contiguous slices and does not shuffle.**
`mmap_batch_generator.__next__` takes `self.data[label][counter:counter+n]` for
each class, advances the counter, and wraps to zero at the end of the array. So
appending real clips to the tail of the positive array does not sprinkle them
through training — it produces a short run of batches whose positives are
100% real voice, with every other batch 100% synthetic. Interleave the rows on
disk. Source:
[`mmap_batch_generator`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/data.py#L733-L855).

**The mixing ratio is set per batch, not by corpus size.**
`batch_n_per_class` in
[`custom_model.yml`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/examples/custom_model.yml)
draws 50 positives, 50 adversarial negatives, and 1,024 ACAV100M rows per
batch regardless of array sizes. At 30,000 positives and 50 per batch, the
positive array is traversed once every 600 steps and about 83 times over the
default 50,000 steps. A small set of real clips is therefore seen many times —
the risk is memorization of those specific clips, not their being drowned out.
This is where the project has direct control over the synthetic:real ratio: it
is the proportion of real rows inside `positive_features_train.npy`.

## Recommendation

**Keep Piper/LibriTTS-R as the single TTS family. Do not add Kokoro. Do add a
modest amount of the owner’s own real audio — but only alongside real negatives
recorded in the same sessions, and only after a speaker-disjoint held-out
subset exists.**

The reasoning, in order of how much it moves the decision:

1. **Multi-TTS is the weakest lever available.** The only direct comparison
   found ([Park et al. Table 4](https://arxiv.org/abs/2407.18879)) shows two
   architecturally distinct TTS systems performing identically to the better
   one alone, and worse than it once real negatives are present. The shortcut
   models learn is “generated vs. recorded,” which a second generator shares.
2. **Kokoro is specifically a poor second choice for this phrase.** 28 English
   voices against LibriTTS-R’s 904 blended embeddings is a step down on the axis
   that matters, its voices are fixed packs rather than a sampled embedding
   space, and its own model card documents “weakness on short utterances,
   especially less than 10-20 tokens.” “Hey Sonny” is about four tokens. If a
   second generator is ever tested, the evidence points toward a different
   *speaker inventory*, not a different vocoder.
3. **Real audio is the lever that actually moves FRR**, and the first slice
   does most of the work — 1% of real positives closed 69% of the gap in
   [Table 3](https://arxiv.org/abs/2408.10463). Ten utterances from one speaker
   produced a measurable 15% relative gain in
   [Table 5](https://arxiv.org/abs/2407.18879).
4. **The architecture is already the best-case one.** A small head on a frozen
   embedding took a 32-point synthetic-only penalty down to 2.7 points in
   [Lin et al.](https://arxiv.org/abs/2002.01322), on the same 96-dim/80 ms
   embedding openWakeWord uses. This project starts from the favourable end of
   the synthetic-to-real gap, which lowers the urgency of every fix here.

Concretely, in the order the existing plan runs:

1. **Ship the first serious candidate on Piper alone**, as
   [`data-strategy.md`](./data-strategy.md) already specifies. Establish
   held-out recall and FA/h before changing the data recipe. Everything below
   is a second revision.
2. **Record in sessions, not in bulk.** Target roughly 6–10 sessions across at
   least 2 microphones and 3 rooms. Per session capture perhaps 20–30 “hey
   sonny” utterances at varied pace, volume, and distance, **plus** the hard
   negatives and two to five minutes of ordinary speech and room ambience. The
   negatives are not optional garnish; they are what prevents the channel
   shortcut in §5.
3. **Hold out whole sessions first.** Assign at least a third of sessions to
   calibration and final evaluation before any of them is used for training,
   and record a small other-speaker subset for the speaker-independence check.
4. **Then, and only then, mix.** Merge the remaining real positives into
   `positive_features_train.npy` with rows interleaved, not appended, and put
   the session-matched real negatives into a separate `feature_data_files`
   entry with its own `batch_n_per_class` allocation — where they will correctly
   receive label 0.
5. **Start at roughly 5–10% real rows within the positive array** and compare
   against the Piper-only candidate on the same frozen held-out revision. This
   is a starting point chosen because published curves are steep at the low end
   and flatten quickly, not a value any paper reports for this setting.
6. **Prefer a clip-quality filter over more clips.** The 13-point penalty
   unfiltered synthesis cost a frozen-embedding model in
   [Quintas et al.](https://arxiv.org/abs/2409.12745) is larger than most of
   the mixing gains discussed here. Rejecting mispronounced, truncated, and
   silent Piper clips is cheap and well-supported.

Two things explicitly **not** recommended for this project:

- **Adversarial domain-invariance training.** It is the best-evidenced fix for
  TTS overfitting ([up to 12%
  relative](https://arxiv.org/abs/2408.10463)), but it requires patching the
  upstream training loop with a gradient-reversal layer and a second
  classifier, and its gains were unstable in the mid-range mixtures (−15.1% to
  +5.6% at 5% real positive weight). Disproportionate for an exploratory
  personal project whose architecture already starts from the favourable end of
  the gap.
- **Kokoro as a general-purpose second generator.** If it is used at all, use it
  as an *evaluation* voice set — synthetic held-out positives from a generator
  absent from training are a cheap, if weak, probe for generator-specific
  overfitting. Keep them clearly separate from the real held-out audio, which
  remains the meaningful test.

## Unresolved empirical questions

- Does adding the owner’s real positives improve held-out recall on *other*
  speakers, leave it flat, or reduce it? The Park Table 5 single-speaker row
  suggests a small overall gain, but nothing found measures the speaker-
  independence cost of a one-speaker mixture on a single-phrase model.
- What is the smallest number of real sessions at which the session-to-session
  variance in recall stabilizes? This determines whether 6 sessions or 20 is
  the right recording target, and it can only be answered by recording.
- Does the session-matched real negative set actually suppress the channel
  shortcut? A direct test exists: score the candidate on real audio of the
  owner speaking *unrelated* sentences in a training room. Elevated scores
  there indicate the model learned the channel.
- At what proportion of real rows in the positive array does performance stop
  improving or start degrading? Synth4Kws documents an overshoot point in the
  opposite direction (too much TTS); no source gives one for this direction at
  this scale.
- Does the frozen openWakeWord embedding separate Piper audio from real
  recordings as cleanly as WavLM separates XTTS-v2 audio in
  [Quintas et al.](https://arxiv.org/abs/2409.12745)? This is directly testable
  with the project’s existing feature arrays and a linear probe, and it would
  tell us how much shortcut risk actually exists here rather than in general.
- Would a second *speaker inventory* within the VITS/Piper family — additional
  Piper voices via the generator’s repeatable `--model` argument — deliver the
  speaker-diversity gain that a second TTS architecture does not?

## Primary sources

- [Utilizing TTS Synthesized Data for Efficient Development of Keyword Spotting Model](https://arxiv.org/abs/2407.18879) — Park et al., Google, SynData4GenAI @ Interspeech 2024
- [Adversarial training of Keyword Spotting to Minimize TTS Data Overfitting](https://arxiv.org/abs/2408.10463) — Park et al., Google, SynData4GenAI @ Interspeech 2024
- [Synth4Kws: Synthesized Speech for User Defined Keyword Spotting in Low Resource Environments](https://arxiv.org/abs/2407.16840)
- [Training Keyword Spotters with Limited and Synthesized Speech Data](https://arxiv.org/abs/2002.01322) — Lin et al., ICASSP 2020
- [Enhancing Synthetic Training Data for Speech Commands](https://arxiv.org/abs/2409.12745) — Quintas et al., 2024
- [On the Effect of Purely Synthetic Training Data for Different ASR Architectures](https://arxiv.org/abs/2407.17997) — Hilmes et al., 2024
- [Speech is Silver, Silence is Golden](https://arxiv.org/abs/2106.12914) — Müller et al., ASVspoof 2021 workshop
- [Personalizing Keyword Spotting with Speaker Information](https://arxiv.org/abs/2311.03419) — Labrador et al., 2023
- [“OK Aura, Be Fair With Me”](https://arxiv.org/abs/2604.05830) — López et al., SPEAKABLE @ LREC 2026
- [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209) — Warden, 2018
- [openWakeWord synthetic-data generation guidance](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/docs/synthetic_data_generation.md)
- [openWakeWord `train.py` at pinned commit `368c037`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py)
- [openWakeWord `data.py` at pinned commit `368c037`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/data.py)
- [Piper sample generator](https://github.com/rhasspy/piper-sample-generator)
- [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M) and [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
- [LibriTTS-R](https://arxiv.org/abs/2305.18802)
