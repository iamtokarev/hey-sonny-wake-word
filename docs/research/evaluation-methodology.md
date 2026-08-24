# Evaluation Methodology for “Hey Sonny”

Research snapshot: 2026-08-24.

## Scope

This note defines a practical evaluation protocol for the personal “Hey Sonny”
openWakeWord model. It covers offline evaluation of the complete openWakeWord
audio-to-trigger path, but not integration with any particular device. The goal
is to answer two questions: how often an intended “Hey Sonny” is detected, and
how often unrelated audio causes an activation.

The operating point is inherently a trade-off. Wake-word research commonly
reports false-reject rate at a fixed false-alarm rate per hour rather than
ordinary classification accuracy; published small-footprint systems, for
example, report FRR at 0.1 or 1.0 false alarms/hour. Sources:
[sequence-to-sequence KWS paper](https://arxiv.org/abs/1811.00348),
[attention-based KWS paper](https://arxiv.org/abs/1803.10916).

## What to measure

### Primary metrics

1. **Positive recall** = intended wake-word utterances producing at least one
   matched activation / all eligible intended utterances. Report **miss rate**
   as `1 - recall`. Define and version the matching window around each annotated
   phrase before evaluating; one utterance counts at most once.
2. **False activations per hour (FA/h)** = unmatched activation events / hours
   of negative audio processed. Consecutive above-threshold frames belonging to
   one acoustic incident must count as one event, not many false activations.
3. **Hard-negative activation rate** = hard-negative utterances causing an
   activation / hard-negative utterances tested. Report it overall and by exact
   phrase or category, because a long-audio hourly rate can hide one consistently
   confusing phrase.

Do not use overall accuracy as a release metric: it changes with the arbitrary
ratio of positive and negative examples and says little about a continuously
listening detector. Keep score distributions and the recall-versus-FA/h curve as
diagnostics.

openWakeWord emits one score every 80 ms. Its included models default to a 0.5
threshold, but upstream explicitly recommends selecting a threshold for the
actual use case. Runtime behavior can also use consecutive-frame `patience`, a
debounce interval, VAD, or noise suppression, all of which alter the resulting
trigger events. Therefore the evaluated candidate is not just the ONNX file; it
is the model plus its frozen threshold and complete trigger policy. Sources:
[openWakeWord recommendations](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/README.md#recommendations-for-usage),
[`Model.predict` contract](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/model.py#L209-L351).

### Secondary diagnostics

- Recall by recording session, speaker, speaking style, and acoustic condition.
- FA/h by negative-audio category: conversational speech, media speech, music,
  environmental noise, and mixed audio.
- Peak score and score trace for every positive, miss, hard negative, and false
  activation.
- Detection timing relative to the annotated phrase, useful for finding late or
  anticipatory triggers but not initially a release gate.

## Dataset roles

Keep four roles separate:

| Role | Permitted use |
|---|---|
| Training data | Fit model weights. |
| Upstream training validation | Early stopping, negative weighting, checkpoint selection, and training diagnostics. |
| Calibration set | Select the threshold and any patience, debounce, VAD, or noise-suppression settings. |
| Final evaluation set | One-time evaluation of a frozen model and trigger configuration. |

The synthetic `positive_test` / `negative_test` features and the approximately
11.3-hour false-positive feature set used by `auto_train` are **training
validation**, not independent evaluation. The trainer repeatedly evaluates
candidate checkpoints on them, uses their scores while retaining/merging
checkpoints, and reports the combined model on the same inputs. Its fixed
threshold metrics are useful directional evidence only. Source:
[`auto_train` and validation logic](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py#L221-L358).

Calibration and final evaluation should contain raw 16 kHz PCM audio and be run
through `openwakeword.Model` in streaming-sized chunks. This tests the shared mel
and embedding models, buffering, ONNX classifier, and event policy together;
feature-array evaluation cannot detect failures at those boundaries. The
official `predict_clip` method follows this streaming path. Source:
[`predict_clip`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/model.py#L353-L387).

### Held-out positive audio

Use separately recorded, naturally spoken “Hey Sonny” utterances. Cover the
intended speaker population and several independent sessions, then distribute
examples across ordinary variations such as pace, volume, emphasis, and
background condition. If this is meant only for one person, repeated sessions
from that person matter more than adding unrelated speakers; if household use
is intended, every intended speaker must appear in both calibration and final
evaluation through disjoint sessions.

### Held-out hard negatives

Include short, labeled clips of the confusing phrases identified in the data
strategy, individual fragments such as “hey” and “Sonny,” and normal sentences
containing similar sounds. “Hey Sunny” should not be treated as an ordinary
negative because it is an acoustic homophone; document its behavior as a known
semantic ambiguity instead. Keep hard-negative text, speaker/session, and
condition in the manifest.

### Held-out long negative audio

Use continuous audio with no intended “Hey Sonny”: conversations, podcasts or
TV-like speech, music, silence/room ambience, and mixed environmental sound.
Preserve continuous files and timestamps so event grouping and repeated errors
can be inspected. Short independent negative clips are useful for diagnosis,
but only long continuous audio gives a meaningful FA/h estimate.

## Leakage controls

- Assign split membership before augmentation; all derivatives of one base clip
  stay in one role.
- Separate personal recordings by session, not by randomly splitting clips from
  one recording session.
- Keep source files, speaker or generator identity, phrase, base-clip ID, and
  checksums in the manifest; reject exact duplicates and inspect likely
  near-duplicates across roles.
- Freeze the calibration and final-evaluation revisions before threshold tuning.
- Never add a final-evaluation miss or false activation to training and continue
  calling that same set independent. Mine the error into a future training set,
  then evaluate the next candidate on a new or still-unseen final revision.
- Do not compare many models on the final set and publish only the best result;
  that turns the final set into another model-selection set.

## Threshold and calibration procedure

1. Choose a small, versioned grid of score thresholds and, only if needed,
   a small number of trigger-policy variants.
2. Replay the complete calibration positives, hard negatives, and long negatives
   for every variant.
3. Select the lowest threshold that satisfies the chosen FA/h ceiling, because
   this preserves the most recall among acceptable points. If none satisfies
   the ceiling, return to data/training rather than quietly relaxing the target.
4. Freeze the model hash, threshold, trigger policy, preprocessing settings, and
   event-grouping rule.
5. Run the final set once and report the result even if it fails the gate.

Upstream suggests that below 0.5 FA/h and below 5% miss rate are often reasonable
in practice, but calls this subjective. For this project, use **0.5 FA/h and 90%
overall recall as initial candidate targets**, with 95% recall as the desired
release target. These are starting policy choices, not demonstrated properties
of “Hey Sonny,” and should be revised only from observed use requirements—not to
make a candidate pass. Source:
[openWakeWord project goals](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/README.md#project-goals).

## Sample and duration planning

Avoid claiming more precision than the evaluation exposure supports.

- **Pipeline pilot:** tens of positives and hard negatives plus roughly one hour
  of varied negative audio. This can reveal broken scoring, phrase confusion, or
  event-counting logic; it cannot establish a release-quality FA/h.
- **Candidate calibration:** aim for at least low hundreds of positive
  utterances across multiple sessions and about 10 hours of varied continuous
  negative audio. This is enough to make threshold trade-offs visible, while
  subgroup results may still be only directional.
- **Final evaluation:** aim for a few hundred independent positive utterances
  and at least 24 hours of negative audio spread across sessions and categories.
  Increase exposure when false activations are rare or when conditions are very
  heterogeneous. Dataset diversity and independence matter more than merely
  multiplying near-duplicate clips.

These are planning bands, not universal sample-size claims. Report uncertainty:
use an exact binomial interval for recall and hard-negative rates, and a Poisson
event-rate interval for FA/h. When sessions differ substantially, also report
per-session values or a session-level bootstrap rather than pretending every
frame is independent. NIST recommends exact binomial limits for small samples or
few failures, and models counts in a time interval with a Poisson distribution.
Sources: [NIST exact binomial intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[NIST Poisson distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366j.htm).

Zero observed false activations does not prove the true rate is zero. Under a
constant-rate Poisson assumption, the one-sided 95% upper rate after zero events
is approximately `3 / exposure_hours`; thus 1 quiet hour is weak evidence, while
24 diverse hours is substantially more informative. Treat the assumption
cautiously when false activations cluster by program, speaker, or session.
Source: [NIST zero-failure confidence bound](https://www.itl.nist.gov/div898/handbook/apr/section4/apr451.htm).

## Staged gates

| Stage | Gate |
|---|---|
| Pipeline pilot | Raw-audio streaming evaluation completes reproducibly; event grouping is verified by inspection; positive scores are generally distinguishable from negatives; no release claim is made. |
| Training candidate | Upstream training validation is recorded; ONNX and full-pipeline predictions agree with the expected candidate; candidate proceeds to independent calibration. |
| Calibrated candidate | A threshold/policy is frozen on calibration data; point estimates meet the initial 90% recall and 0.5 FA/h targets; no major hard-negative phrase activates systematically. |
| Release candidate | Frozen configuration is run once on the final set; overall recall reaches the desired 95% target, FA/h stays at or below 0.5, subgroup results show no unacceptable blind spot, and uncertainty intervals are reported. |
| Release | Every miss and false activation has been reviewed; artifacts, dataset revisions, and report are complete. A failed final gate produces a documented non-release candidate, not an adjusted result. |

The first release may retain 90% rather than 95% recall if that trade-off is
explicitly accepted after observing the FA/h curve. The report must preserve the
original desired gate and record the decision; the threshold must not be retuned
on final data.

## Error review

For every miss and false activation, retain the source ID, timestamp, score
trace, matched event boundaries, transcript/phrase if known, session metadata,
and acoustic category. Review audio where privacy permits and assign a concise
cause label: confusing speech, target fragment, media speech, music/noise,
low-SNR target, pronunciation/speaking-style miss, annotation problem, or
pipeline problem. Summarize counts by cause and keep a small set of representative
examples. Error review should drive the next data revision, but those examples
must then leave the independent test role as described above.

## Minimum reporting schema

Each evaluation report should contain:

```yaml
model:
  artifact_sha256: "..."
  openwakeword_revision: "368c037..."
detector:
  threshold: 0.0
  patience_frames: 0
  debounce_seconds: 0.0
  vad_threshold: 0.0
  noise_suppression: false
  event_grouping_rule: "versioned rule name"
data:
  calibration_revision: "..."
  final_revision: "..."
  positive_utterances: 0
  hard_negative_utterances: 0
  long_negative_hours: 0.0
results:
  recall: 0.0
  recall_confidence_interval_95: [0.0, 0.0]
  miss_rate: 0.0
  false_activations: 0
  false_activations_per_hour: 0.0
  fa_per_hour_confidence_interval_95: [0.0, 0.0]
  hard_negative_activation_rate: 0.0
  subgroup_results: {}
decision:
  stage: "pilot|candidate|release"
  passed: false
  rationale: "..."
```

Also retain the threshold sweep from calibration and a machine-readable event
table with one row per intended utterance, miss, and false activation.

## Empirical unknowns

1. Is “Hey Sonny” intended for one known speaker, a household, or arbitrary
   English speakers? This determines the positive evaluation population.
2. Which “Sonny” pronunciation variants are intended positives, and is activation
   on the homophone “Sunny” acceptable?
3. Which negative-audio categories dominate the intended always-listening use,
   and therefore deserve the largest duration in calibration and final sets?
4. Does threshold-only calibration suffice, or does consecutive-frame patience
   materially reduce FA/h without an unacceptable recall loss?
5. How stable are results across recording sessions? The answer determines how
   much additional exposure is needed beyond the initial planning bands.
6. Does the project accept the initial 0.5 FA/h ceiling, or require a stricter
   target after experiencing real false activations?

## Primary sources

- [openWakeWord repository](https://github.com/dscripka/openWakeWord)
- [Pinned training and validation implementation](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py)
- [Pinned streaming inference implementation](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/model.py)
- [openWakeWord metric utilities](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/metrics.py)
- [NIST/SEMATECH statistical methods handbook](https://www.itl.nist.gov/div898/handbook/)
- [Sequence-to-sequence Models for Small-Footprint Keyword Spotting](https://arxiv.org/abs/1811.00348)
- [Attention-based End-to-End Models for Small-Footprint Keyword Spotting](https://arxiv.org/abs/1803.10916)
