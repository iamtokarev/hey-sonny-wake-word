# openWakeWord: what to reuse, and every defect found in it

What this project reuses from openWakeWord's custom-training workflow, and the
compatibility work it has to own. Desk research 2026-08-24; everything under
"Confirmed incompatibilities" and below was found empirically while running the
pipeline against commit `368c037`.

## Upstream status, and why the commit is pinned

openWakeWord is a sound model and inference foundation, but its custom training
workflow is not a supported, reproducible installation on a modern environment.
The repository is not archived and its latest `main` commit is from 2025-12-30,
but the latest release remains
[v0.6.0 from 2024-02-11](https://github.com/dscripka/openWakeWord/releases/tag/v0.6.0),
and the package version in current `main` is *also* `0.6.0` — so a version
number alone does not identify the source being used. The open report that the
[automatic notebook no longer works](https://github.com/dscripka/openWakeWord/issues/296)
is consistent with the dependency drift below. No upstream test exercises
`train.py`; the [tests](https://github.com/dscripka/openWakeWord/tree/368c03716d1e92591906a84949bc477f3a834455/tests)
cover inference and verifier behaviour instead.

So: pin the source commit
[`368c037`](https://github.com/dscripka/openWakeWord/commit/368c03716d1e92591906a84949bc477f3a834455)
and record it in every run manifest. Do not treat PyPI `0.6.0` as equivalent —
current `main` makes TFLite conversion optional and derives the classifier
input shape from the produced features, where released `v0.6.0` always attempts
TFLite conversion.

The three training stages are also not independent in
[`train.py`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py#L598-L910):
before testing the requested stage flag it imports Piper, scans the RIR and
background directories, and samples generated positive WAVs. Splitting
synthesis, features and training across jobs therefore needs phase isolation —
which is what calling the library functions directly buys us.

## Feature and classifier pipeline

The runtime is three parts: a fixed mel-spectrogram model, a fixed Google
speech-embedding model, and the trained wake-word classifier. The embedding
width is 96, and the classifier can be a small fully connected network or a
two-layer bidirectional RNN — both are used here (`--model-type`). For
two-second clips the classifier input is `16 x 96` float features, and the
output is one sigmoid score. It is a binary detector: multiple target phrases
would still map to one positive output. The two-stage architecture documented
for the bundled `hey_jarvis` model is **not** what the generic custom trainer
builds; see
[upstream-recipe-comparison.md](upstream-recipe-comparison.md#architecture).

### ONNX inference contract

The exported `hey_sonny.onnx` is the **classifier**, not a standalone raw-audio
model. It expects one fixed-length sequence of float32 openWakeWord embeddings
and returns one score. Raw input to the complete Python runtime must be 16-bit,
16 kHz PCM; `openwakeword.Model.predict` accumulates it and evaluates the
classifier at 80 ms intervals. The application chooses the activation
threshold; none is embedded in the classifier.

Consequently, validating only `hey_sonny.onnx` with ONNX Runtime is necessary
but insufficient. End-to-end validation must load the shared
`melspectrogram.onnx` and `embedding_model.onnx` assets and run raw PCM through
`openwakeword.Model(..., inference_framework="onnx")`. Both `AudioFeatures` and
`Model` default to `inference_framework="tflite"`, which imports
`ai_edge_litert`; an ONNX-only environment must pass `"onnx"` explicitly or the
default reintroduces the TFLite dependency this project excludes.

### Dependencies and Piper

Current `main` declares Python `>=3.10`, but the `full` extra combines an older
training stack: TensorFlow CPU 2.8.1, ONNX-TF 1.10, TensorFlow Probability 0.16,
Torchaudio `<1`. TensorFlow and ONNX-TF are imported only inside optional TFLite
conversion, so an ONNX-only job installs a tested selective set instead — the
authoritative pinned header lives in `scripts/preflight.py`.

`openwakeword.data` is **not** optional despite living in the `[full]` extra: it
imports `pronouncing`, `audiomentations`, `torch_audiomentations`, `speechbrain`,
`mutagen`, and `acoustics` at module level, so anything using
`filter_audio_paths`, `stack_clips`, `mix_clips_batch` or `trim_mmap` needs all
of them.

The Piper interface has drifted. openWakeWord imports a root-level
`generate_samples.py` and calls a function whose model has a default, matching
the old
[`dscripka/piper-sample-generator` fork](https://github.com/dscripka/piper-sample-generator/blob/f1988a4d54eddb23d99e86f0adfef6226a85acc7/generate_samples.py#L28-L76).
The maintained generator is packaged under
[`piper_sample_generator.__main__`](https://github.com/rhasspy/piper-sample-generator/blob/2971426a55072f7d22fec416ca7800df8bd23207/piper_sample_generator/__main__.py#L29-L76),
requires an explicit `model`, and requires NumPy 2 — not a drop-in replacement
for unmodified `train.py`. Job A uses the maintained one at pinned commit
`2971426a`, cloned rather than installed from PyPI because the wheel drops
`piper_train`, which `__main__.py` imports.

## Confirmed incompatibilities and required patches

Found running the pipeline against `368c037` on Python 3.12–3.13 with NumPy 2,
torch 2.11–2.13 and current torchaudio. Each one blocks the pipeline until
patched.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `uv pip install openwakeword` resolves to nothing installable | `speexdsp-ns==0.1.2` is in `install_requires` on Linux and publishes no wheel past `cp312` | Install with `--no-deps` and supply requirements explicitly, or stay on Python 3.12 where the wheel exists |
| 2 | `AttributeError: module 'torchaudio' has no attribute 'info'` in `get_clip_duration` | `torchaudio.info` was removed from recent torchaudio; `data.py` still calls it | Shim `torchaudio.info` with a `soundfile.info` reader returning `num_frames` and `sample_rate` |
| 3 | `TypeError: _amax() got an unexpected keyword argument 'dim'` in `mix_clips_batch` | Upstream converts the batch to NumPy, then calls `.max(dim=1)` — torch syntax on a NumPy array | Change that one call to `.max(axis=1)` |
| 4 | `IndexError: tuple index out of range` at `utils.py:270`, but only on the last batch | Same line as #3 wraps the index in `torch.from_numpy`. A 1-element torch tensor implements `__index__`, so NumPy reads it as a *scalar* index and drops the leading axis; a `(48000,)` array then reaches `embed_clips`, which requires `(N, samples)` | Drop the wrapper: `error_index = np.where(mixed_clips_batch.max(axis=1) != 0)[0]` |
| 5 | `ImportError: TorchCodec is required for load_with_torchcodec` from `augment_clips` | torchaudio 2.11 delegates `load` to torchcodec. `openwakeword.data` and `torch_audiomentations` call `load` six times and `info` five, and `augment_clips` reaches both for every RIR and background clip | Shim both `torchaudio.load` and `torchaudio.info` onto soundfile — everything on those paths is a plain 16 kHz WAV. Apply unconditionally, not on failure |
| 6 | `ModuleNotFoundError: No module named 'onnxscript'` from `torch.onnx.export` | `torch>=2.9` defaults `export` to the dynamo exporter, which imports `onnxscript`; it is not a torch dependency | Add `onnxscript` to the requirements |
| 7 | `ImportError: cannot import name 'sph_harm' from 'scipy.special'` on importing `openwakeword.data` | `data.py` imports `acoustics` at module level; `acoustics` 0.2.6 is unmaintained, imports `scipy.special.sph_harm` — removed in scipy 1.17 — and declares no upper bound | Pin `scipy<1.17` |

Items 3 and 4 are two defects on one line, and both are upstream defects rather
than version drift:

```python
error_index = torch.from_numpy(np.where(mixed_clips_batch.max(dim=1) != 0)[0])
```

The line sits immediately after the int16 conversion, so it cannot have executed
at this commit under any dependency set — #3 made the silent-clip filter it
guards unreachable, which is why #4 was never found. Fixing #3 alone exposes #4,
and #4 fires only when a batch survives with exactly one clip, i.e. when the clip
count is `1 mod batch_size`. With 897 positives at `batch_size=8` the failure
lands on batch 112 of 113, after every earlier batch has succeeded. That is
consistent with the absence of any upstream test over `train.py`.

```python
a = np.zeros((1, 5), dtype=np.int16)
a[torch.from_numpy(np.array([0]))].shape   # (5,)    — axis collapsed
a[np.array([0])].shape                     # (1, 5)  — correct
```

### `piper-tts` silently disables the GPU

`piper-tts` requires `onnxruntime<2,>=1` — the CPU build. Installed beside
`onnxruntime-gpu` it provides the same `onnxruntime` module and shadows it:
`CUDAExecutionProvider` disappears from `get_available_providers()`,
`AudioFeatures(device="gpu")` falls back to the CPU, and nothing raises.
Feature extraction then runs an order of magnitude slower on hardware that is
being paid for.

The fix is a uv dependency override in the script's PEP 723 metadata, using a
marker that is never true so the requirement is dropped:

```
# [tool.uv]
# override-dependencies = ["onnxruntime ; python_version < '3.0'"]
```

Preflight fails on this rather than reporting it, because the symptom is
invisible at runtime.

### Import order: torch before onnxruntime

onnxruntime registers its CUDA execution provider when first imported, and
locates the CUDA libraries through the ones torch's `nvidia-*` wheels put on the
loader path, so torch is imported first everywhere in this pipeline. The
convention is cheap and plausible, but it is *not* the explanation for the
provider failures observed on 2026-08-31 — those were the `piper-tts` shadowing
above. One earlier failure, on a run with no `piper-tts` and no import-order fix,
remains unexplained: three of four otherwise identical `t4-small` runs passed and
one reported only `['AzureExecutionProvider', 'CPUExecutionProvider']` while
`torch.cuda.is_available()` was still True.

Treat the ordering as a convention and the runtime assertion as the real
protection: every stage that depends on GPU features asserts
`CUDAExecutionProvider` is present rather than trusting it.

## Defects in the training and evaluation path

Unlike the table above, these do not stop the pipeline — they silently change
the numbers it reports, which is worse. Found reading `train.py` and confirmed
by measurement.

| Symptom | Cause | Fix |
|---|---|---|
| Every false-accept rate from `auto_train` is 5.6% optimistic | `val_set_hrs = 11.3` is a hardcoded local with no parameter to override, and is wrong for the very file it describes: `validation_set_features.npy` has shape `(481345, 96)`, and 481,345 x 0.08 s = **10.70 h** | Call `train_model` directly, which takes `val_set_hrs`, and pass the measured duration |
| `get_false_positives` returns frames above threshold, not activation events | The suppression loop reads `len(transitions)` where it means `len(bin_pred)`, so the slice that should swallow a detection's trailing frames is empty or reversed. Measured: one 20-frame activation returns 20 (correct: 1); two activations of 20 and 6 frames return 26 (correct: 2); 100 short activations return 274 (correct: 100) | Reimplement with correct event grouping. Note this makes `auto_train`'s metric and the notebook's accidentally the *same* metric, so the two are at least self-consistent |
| `train.export_model` freezes the ONNX batch axis at 1 | It traces on one example and requests no dynamic axes, so torch 2.13's dynamo exporter pins the batch dimension; batched calls fail with `Got: 64 Expected: 1` — including upstream's own `predict_on_features`, which scores every window of a clip in one go | Call `torch.onnx.export` directly with a dynamic batch axis, then assert the input's batch dimension is symbolic |
| `train.export_model`'s `opset_version=13` is silently ignored | `LayerNormalization` was introduced at opset 17 and has no earlier form; onnxscript logs the downgrade failure, falls back to the ONNX C API, logs that failure too, and returns the model unconverted — the export succeeds and the opset claim is fiction | Export at opset 17 and record the opset the file actually carries |
| `torch.onnx.export` publishes a 3 KB model that loads nowhere | The dynamo exporter defaults to `external_data=True`, writing weights to a sibling `.onnx.data` | `external_data=False`, plus assertions on initializer location and file size. Full account in [jobs-spec.md](../jobs-spec.md#a-checksum-round-trip-is-not-proof-the-model-works) |
| `mmap_batch_generator` in a multi-worker `DataLoader` emits every batch once per worker | The `IterableDataset` is copied into each worker and the generator's per-file cursor starts at zero in all of them, with no sharding; `train.py` sets `num_workers = cpu_count // 2` | Use `num_workers=1` |
| `Model.best_val_fp` is initialised to 1000 and never assigned | `auto_train` gates its negative-weight doubling on `best_val_fp > target_fp_per_hour`, so the doubling is unconditional rather than responsive | Drive `train_model` directly and gate on the rate the previous sequence measured |
| `auto_train`'s checkpoint averaging almost never fires | A checkpoint must clear the 90th percentile on accuracy *and* recall *and* the 10th on false positives at once, but those move against each other, so the intersection is usually empty and the silent fallback ships `self.model` — the last checkpoint of the last sequence, taken under the heaviest negative weight | Gate on one axis and rank on the other; log how many checkpoints were merged |
| Validation recall is measured on speakers seen in training | `train.py` generates positive train and test from the same generator, same speaker traversal, no `max_speakers` | Split the speaker inventory explicitly before generation (`--val-speakers`) |

### Sample-rate mismatch (not a bug, but silent)

Piper's LibriTTS-R generator writes **22050 Hz**. `mix_clips_batch`, the feature
extractor, and `predict_clip` all assume **16 kHz** and none of them resample.
Generated positives must be resampled before use.

This surfaces as a confusing tensor-size error inside `mix_clip` only when clip
lengths happen not to fit the window. When they do fit, there is no error at all
and the model trains on audio playing about 1.38x too fast, corrupting every
feature with nothing to indicate it. Treat the resample as mandatory and assert
the rate rather than trusting it.

## Primary sources

- [openWakeWord repository](https://github.com/dscripka/openWakeWord)
- [Training script at the pinned commit](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py)
- [Custom-model configuration](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/examples/custom_model.yml)
- [Inference implementation](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/model.py)
- [Feature implementation](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/utils.py)
- [Maintained Piper sample generator](https://github.com/rhasspy/piper-sample-generator/tree/2971426a55072f7d22fec416ca7800df8bd23207)
