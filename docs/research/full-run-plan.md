# Configuration and code boundary

Settled 2026-08-31, after a pilot ran end to end on Colab and upstream
`train.py` was read at commit `368c037`. This note records the configuration
the jobs run with and, more importantly, **which code is ours and which is
upstream's**. The commands that execute it are in
[jobs-spec.md](../jobs-spec.md).

## Settled configuration

| Choice | Value | Reason |
| --- | --- | --- |
| Window | **2 s, 16 frames** | What the shipped openWakeWord models use, and what upstream's own sizing heuristic picks for a phrase this short. Also avoids two hardcoded `16`s in `train.py` (`Model(input_shape=(16, 96))` and the `x_val[:, i:i+16, :]` slice in the positive-test recall path) that silently misbehave at any other size. |
| Positive samples | **30,000** | Requires `max_speakers ≈ 173` per the `sqrt(max_samples)` rule, or the `itertools.product` speaker traversal never advances past the first few speakers. 100,000 was tried and lost — see `experiment-plan.md` E3. |
| False-positive validation set | **upstream's `validation_set_features.npy`** | Makes our false-accept numbers directly comparable to upstream's published models. A room-specific set can be added later as a second measurement, not a replacement. |
| Hardware flavor | per stage — see [jobs-spec.md](../jobs-spec.md#hardware-pick-by-bottleneck) | The two jobs have different bottlenecks: Job A wants vCPU, Job B wants neither a bigger GPU nor more disk. |

## Why the whole ACAV100M file, not a slice

`openwakeword_features_ACAV100M_2000_hrs_16bit.npy` is **already windowed**:
`(5625000, 16, 96)` float16, which at 1.28 s of stride per window is the
advertised 2,000 hours. It is not the flat `(N, 96)` array that `train.py`'s
`f` transform exists to re-cut, so for this file `f` is a no-op. (Read off the
mount during a rehearsal; reasoning from `train.py`'s transform without
checking gives the wrong answer.)

Two consequences:

- **`batch_n_per_class` means what it says.** The realised batch is
  1024 + 50 + 50 = **1,124 examples at 20:1 negatives to positives**. The
  negative weight ramp is therefore doing less work than a 2.3:1 batch would
  need it to, which is the intended design and not an accident to correct.
- **The run makes about eleven passes, not one.** 60,000 steps x 1,024 windows
  is 61.4M draws over 5.625M windows. `mmap_batch_generator` advances a per-file
  counter sequentially and wraps, so those are eleven ordered traversals with
  good read locality — not random sampling. The argument against slicing gets
  stronger with scale: a 200,000-window slice would be traversed about 300
  times instead of eleven.

Mount the full 17.28 GB rather than downloading it. Measured on `t4-small`, the
Hub mount sustains roughly 155 MB/s once warm — about 3.1 MB per step, which is
what sets the ~55 steps/s ceiling.

## Code boundary: reuse the library, own the orchestration

The decision is neither "run `train.py`" nor "write it all". Keep the library's
functions and own only the orchestration around them.

**Import and use unchanged.** `openwakeword.utils.AudioFeatures` and
`compute_features_from_generator` (mandatory — the frozen embedding *is* the
project); `openwakeword.data.{filter_audio_paths, augment_clips, mix_clips_batch,
stack_clips, trim_mmap, mmap_batch_generator, generate_adversarial_texts}`;
`openwakeword.Model` for scoring. `generate_adversarial_texts` in particular is
worth reusing rather than reimplementing.

**Import, but drive ourselves.** `openwakeword.train.Model` is importable — the
argparse block is `__main__`-guarded and the module only pulls `torch`,
`torchinfo`, `torchmetrics`, and `yaml`. Importing it gives us the architecture
(both the DNN and the bidirectional LSTM head), the negative-weight ramp,
hard-example mining, checkpoint collection, and `average_models`:

- the negative weight ramps `np.linspace(1, max_negative_weight, steps)` across
  training, so the model learns the word before it is punished for false
  accepts;
- each batch drops negatives scoring below 0.001 and positives above 0.999
  before the loss, which matters when most of the batch is generic audio the
  model already handles;
- checkpoints are collected, filtered by false-positive rate against a recall
  floor, and the survivors above the 90th percentile are weight-averaged — in
  principle. Measured on the first full run: **zero of 55 checkpoints** cleared
  all three percentile gates, because they move against each other, and the
  model shipped was the unaveraged final one. Reuse the mechanism, not the
  selection rule.

**Do not call `auto_train`.** Call `train_model` three times directly. This is
about thirty lines and it buys two things `auto_train` cannot give us: a correct
`val_set_hrs` (`train_model` takes it as a parameter, `auto_train` overwrites it
with a wrong literal — see
[openwakeword-baseline.md](openwakeword-baseline.md#defects-in-the-training-and-evaluation-path)),
and a stage boundary after each sequence at which to push a checkpoint. That
second point is not optional: an unpushed run that dies loses everything, which
is why Job B checkpoints to a durable bucket mount after each sequence.

**Write ourselves.** Configuration resolution and hashing, run and attempt
manifests, stage `COMPLETE.json` markers, ONNX export, and the final
evaluation — including a correct false-positive event counter, since upstream's
counts frames. `train.py` has none of these.

## Primary sources

- [`train.py` at the pinned commit](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py)
- [`examples/custom_model.yml`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/examples/custom_model.yml)
- [Upstream precomputed features](https://huggingface.co/datasets/davidscripka/openwakeword_features)
- [MIT environmental impulse responses](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses)
- [Hugging Face Jobs pricing and hardware](https://huggingface.co/docs/hub/jobs-pricing)
