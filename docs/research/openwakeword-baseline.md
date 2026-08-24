# openWakeWord Technical Baseline

Research snapshot: 2026-08-24.

## Research question

What part of the current openWakeWord custom-training workflow can this project
reuse for a personal **"Hey Sonny"** model, and what minimal compatibility work
must be owned by the project?

## Findings

### Upstream status

openWakeWord remains a suitable model and inference foundation, but its custom
training workflow is not currently a supported, reproducible installation on a
modern environment. The repository is not archived and its latest `main` commit
is from 2025-12-30, but the latest release remains
[v0.6.0 from 2024-02-11](https://github.com/dscripka/openWakeWord/releases/tag/v0.6.0).
The package version in current `main` is still `0.6.0`, so a version number alone
does not identify the source being used. The open upstream report that the
[automatic notebook no longer works](https://github.com/dscripka/openWakeWord/issues/296)
is consistent with the source-level dependency drift described below.
No upstream automated test currently exercises `train.py`; the repository's
[tests](https://github.com/dscripka/openWakeWord/tree/368c03716d1e92591906a84949bc477f3a834455/tests)
cover inference and verifier behavior instead. An end-to-end project smoke test
is therefore part of the compatibility baseline.

The earlier project research was directionally correct, with two important
refinements:

- The project must pin the current source commit
  [`368c037`](https://github.com/dscripka/openWakeWord/commit/368c03716d1e92591906a84949bc477f3a834455),
  not merely install released `v0.6.0`. Current `main` makes TFLite conversion
  optional and derives the classifier input shape from the produced features;
  released `v0.6.0` always attempts TFLite conversion.
- The three training commands are not truly independent in current
  [`train.py`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py#L598-L910).
  Before testing the requested stage flag, the script imports Piper, scans the
  RIR and background directories, and samples generated positive WAV files.
  Separate synthesis, feature, and training Jobs therefore require a small
  phase-isolation patch or must all mount every earlier input and dependency.

### Custom-training path and artifacts

The intended automated path is still useful as the baseline:

1. `--generate_clips` uses Piper to create synthetic positive and adversarial
   negative WAV files for train and validation sets.
2. `--augment_clips` mixes those clips with background audio and room impulse
   responses, then passes them through the frozen openWakeWord feature stack.
3. `--train_model` combines the generated features with general negative
   features, selects candidates using a false-positive validation feature set,
   and exports the selected binary classifier as ONNX.

The required configuration and data roles are defined in the official
[`custom_model.yml`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/examples/custom_model.yml):
target phrases, explicit confusing phrases, sample counts, Piper location,
background and RIR directories, general negative feature arrays, a separate
false-positive validation array, class batch proportions, model size, and the
training target.

The generated working artifacts are four WAV directories
(`positive_train`, `positive_test`, `negative_train`, `negative_test`) and four
memory-mapped feature arrays with matching roles. These live under
`<output_dir>/<model_name>/`. The final classifier is written to
`<output_dir>/<model_name>.onnx`, outside that subdirectory. Upstream logs final
accuracy, recall, and false positives per hour, but does not write a metrics
file or resumable training checkpoint; candidate models are retained in memory
during `auto_train`.

### Feature and classifier pipeline

The runtime is a three-part pipeline: a fixed mel-spectrogram model, a fixed
Google speech-embedding model, and the newly trained wake-word classifier. The
official architecture description and source show that the embedding width is
96 and that the custom classifier can be a small fully connected network or a
two-layer bidirectional RNN. The example configuration uses the DNN with layer
size 32. The special two-stage architecture documented for the bundled
`hey_jarvis` model is **not** what the generic custom trainer builds.
([architecture](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/README.md#model-architecture),
[`AudioFeatures`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/utils.py#L33-L463),
[classifier source](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py#L25-L196))

For the usual two-second clips, the classifier input is normally one batch of
`16 x 96` float features. Current `main` derives the frame count from the
generated feature file, so `16` must be verified rather than hard-coded. The
classifier produces one sigmoid score between 0 and 1. It is a binary detector:
if multiple target phrases are configured, they still map to the same positive
output.

### ONNX inference contract

The exported `hey_sonny.onnx` is the **classifier**, not a standalone raw-audio
model. It expects one fixed-length sequence of float32 openWakeWord embeddings
and returns one score. Raw input to the complete Python runtime must be 16-bit,
16 kHz PCM; `openwakeword.Model.predict` accumulates it and evaluates the
classifier at 80 ms intervals. The application chooses the activation
threshold; no project threshold is embedded in the classifier.
([usage contract](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/README.md#usage),
[`Model` loading and prediction](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/model.py#L32-L373))

Consequently, validating only `hey_sonny.onnx` with ONNX Runtime is necessary
but insufficient. End-to-end validation must also load the shared
`melspectrogram.onnx` and `embedding_model.onnx` assets and run raw PCM through
`openwakeword.Model(..., inference_framework="onnx")`.

### Dependencies and Piper

Current `main` declares Python `>=3.10`, but the `full` extra combines an older
training stack: TensorFlow CPU 2.8.1, ONNX-TF 1.10, TensorFlow Probability 0.16,
Torchaudio `<1`, and related pins. TensorFlow and ONNX-TF are now imported only
inside optional TFLite conversion, so an ONNX-only Job should install a tested,
selective training dependency set rather than `openwakeword[full]`.
([current `setup.py`](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/setup.py))

The project's current root `pyproject.toml` requires Python `>=3.12`. That must
not silently define the training environment: the training container needs its
own tested Python constraint and lock, or the root constraint must be revised
after the compatibility smoke test.

GPU feature extraction explicitly asks ONNX Runtime for
`CUDAExecutionProvider`, so the training image needs a compatible
`onnxruntime-gpu` build and must verify the provider before doing expensive
work. The classifier itself is trained with PyTorch and exported with ONNX
opset 13.

The Piper interface has drifted. openWakeWord imports a root-level
`generate_samples.py` and calls a function whose model has a default. That
matches the old, referenced
[`dscripka/piper-sample-generator` fork](https://github.com/dscripka/piper-sample-generator/blob/f1988a4d54eddb23d99e86f0adfef6226a85acc7/generate_samples.py#L28-L76).
The maintained generator is packaged under
[`piper_sample_generator.__main__`](https://github.com/rhasspy/piper-sample-generator/blob/2971426a55072f7d22fec416ca7800df8bd23207/piper_sample_generator/__main__.py#L29-L76),
requires an explicit `model`, and currently requires NumPy 2. It is not a
drop-in replacement for unmodified `train.py`.

## Project decisions and implications

- Pin openWakeWord source commit `368c037` and record that commit in every run
  manifest. Do not treat PyPI `0.6.0` as equivalent.
- Preserve the upstream feature extractor and default small DNN for the first
  candidate. Architecture changes are out of scope until data and evaluation
  show a concrete need.
- Export ONNX only. Keep TFLite/TensorFlow dependencies outside the training
  environment.
- Add a minimal project-owned compatibility layer that makes synthesis,
  augmentation/feature extraction, and training independently runnable. It
  should lazy-load Piper and avoid requiring RIR, background, or raw WAV inputs
  during a training-only stage.
- For the first smoke test, use the Piper fork expected by openWakeWord, pinned
  to `f1988a4`, with the trusted generator checkpoint pinned and checksummed.
  Its `torch.load` call will probably need the explicit modern-PyTorch
  compatibility argument already used by the maintained generator. Keep WAV
  generation behind a narrow interface so this fork can later be replaced
  without changing the openWakeWord stages.
- Download and pin the shared ONNX feature models explicitly. The final model
  documentation must state that `hey_sonny.onnx` depends on those shared
  preprocessing assets and the openWakeWord streaming logic.
- Capture upstream's logged final metrics into structured run output. Treat
  generated clips and feature arrays as the resumable boundary; unmodified
  `auto_train` itself is restart-from-beginning.

## Unresolved questions requiring a smoke test

1. Does the selective Python/CUDA dependency set resolve together and expose
   both `torch.cuda` and ONNX Runtime's `CUDAExecutionProvider`?
2. Does the pinned Piper fork generate intelligible, correctly pronounced
   "Hey Sonny" WAVs with the selected checkpoint on modern PyTorch?
3. What exact frame count does the generated phrase produce after augmentation,
   and does the exported ONNX input match the feature-array shape?
4. Can each patched stage run with only its declared inputs and safely reuse
   already-persisted artifacts?
5. Does the exported classifier pass ONNX validation and then produce sensible
   scores when loaded through the complete openWakeWord raw-PCM pipeline?
6. Can final validation metrics be captured deterministically into a run
   manifest without changing the upstream training behavior?

## Primary sources

- [openWakeWord repository](https://github.com/dscripka/openWakeWord)
- [Current training script at the pinned commit](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/train.py)
- [Current custom-model configuration](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/examples/custom_model.yml)
- [Current inference implementation](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/model.py)
- [Current feature implementation](https://github.com/dscripka/openWakeWord/blob/368c03716d1e92591906a84949bc477f3a834455/openwakeword/utils.py)
- [openWakeWord v0.6.0 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.6.0)
- [Piper fork expected by openWakeWord](https://github.com/dscripka/piper-sample-generator/tree/f1988a4d54eddb23d99e86f0adfef6226a85acc7)
- [Maintained Piper sample generator](https://github.com/rhasspy/piper-sample-generator/tree/2971426a55072f7d22fec416ca7800df8bd23207)
