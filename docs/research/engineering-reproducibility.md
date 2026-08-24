# Engineering and Reproducibility Contract

Research snapshot: 2026-08-24. Scope: the minimum engineering contract needed
to build, inspect, rerun, and promote the personal **"Hey Sonny"** openWakeWord
model. It does not define model architecture, data selection, evaluation
thresholds, or deployment integration.

## Recommended contract

The executable source of truth should be a small Python package plus command-line
entry points. Configuration files define runs; notebooks only read artifacts and
call package functions. A serious run is identified by immutable source,
environment, input, configuration, and seed references rather than by a Hugging
Face Job ID alone.

The paid training environment is a pinned Docker image. Local Python is for fast
development checks and notebook exploration, but is not claimed to reproduce the
GPU stack. The current root `pyproject.toml` requires Python 3.12 while the
openWakeWord training dependencies need a separately tested Python constraint;
the training Dockerfile and training lock must therefore remain authoritative
until the compatibility smoke test chooses one supported version. This follows
the dependency finding in
[openwakeword-baseline.md](openwakeword-baseline.md#dependencies-and-piper).

Use Git for code and configuration, a private Hugging Face Bucket for mutable
run outputs, revision-pinned Dataset repositories for curated inputs, and a
Model repository for accepted releases. Buckets are intentionally mutable and
non-versioned, whereas Hub repositories have Git history and are intended for
published artifacts ([Hugging Face Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)).

## Minimum repository layout

```text
hey-sonny-wake-word/
  pyproject.toml                 # package and lightweight developer tooling
  uv.lock                        # developer/notebook environment
  configs/
    pilot.yaml
    full.yaml
    evaluation.yaml
  environments/training/
    Dockerfile
    requirements.lock           # exact GPU/training dependency set
  src/hey_sonny/
    pipeline.py                  # stage entry points and orchestration
    openwakeword_adapter.py      # the narrow upstream compatibility seam
    artifacts.py                # manifests, checksums, validation, reuse rules
    evaluation.py               # reusable scoring/report functions
  scripts/
    run.py                       # one CLI for preflight and pipeline stages
    promote.py                   # checked promotion into the Model repository
  tests/
  notebooks/
    01_audio_audit.ipynb
    02_artifact_audit.ipynb
    03_error_analysis.ipynb
  docs/
```

Keep this boundary lean. Do not copy the openWakeWord trainer into several
scripts: isolate the pinned upstream behavior and project patch behind
`openwakeword_adapter.py`. Do not add a workflow orchestrator, experiment
database, DVC, or a second configuration system unless repeated runs expose a
specific need.

## Configuration and environment ownership

- Version the human-authored pilot, full, and evaluation configurations in Git.
- Resolve defaults and command-line overrides before execution, validate the
  result, save it as `resolved-config.yaml`, and hash those exact bytes. Reject
  unknown keys so a typo cannot silently change a paid run.
- Keep secrets and machine-specific mount paths outside configuration and
  manifests.
- Pin openWakeWord, the Piper generator, shared ONNX feature models, and every
  curated Hub input to a full commit/revision. `huggingface_hub` supports
  downloads at a specific commit through `revision`; record the resolved commit,
  not only a branch name ([Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download)).
- Record the immutable container image digest, not just a mutable tag. Docker
  documents that pulling by digest selects the same image even if a tag changes
  ([Docker image digests](https://docs.docker.com/dhi/core-concepts/digests/)).
- Do not promote a model from an uncommitted source tree. A smoke or pilot may
  run dirty for investigation, but its manifest must mark `source.dirty=true`
  and it cannot become a release candidate.

## Run identity and manifest

Use two identifiers:

- `run_id` identifies one intended experiment, for example
  `20260824T120000Z-<git8>-<config8>`.
- `attempt_id` identifies each execution or retry of that run. A retry appends a
  new attempt; it never overwrites the evidence from a failed attempt.

Store the canonical run manifest at `runs/<run-id>/manifest.json`. It should
contain only information needed to reconstruct and audit the run:

| Group | Required fields |
| --- | --- |
| Identity | schema version, run ID, creation time, purpose (`smoke`, `pilot`, `candidate`), parent run if any |
| Source | project Git commit and dirty flag; openWakeWord and Piper full commits |
| Environment | image reference and digest, training lock SHA-256, Python/CUDA/PyTorch/ONNX Runtime versions, detected GPU and ONNX providers |
| Configuration | resolved configuration path and SHA-256; all random seeds and determinism flags actually used |
| Inputs | each Dataset/Model repository ID and resolved commit; Bucket object/prefix plus manifest hash; external asset SHA-256 and license/provenance reference |
| Execution | attempt IDs, HF Job IDs, hardware flavor, stage names, start/end/status, and output prefix |
| Outputs | artifact type, relative path, byte size, SHA-256, producing stage, validation status, and schema/shape where relevant |
| Results | calibration and final revisions, metrics-file hashes, frozen detector-policy hash, gate decision, and failure reason when rejected |

Clip-level provenance belongs in the dataset manifest defined by
[data-strategy.md](data-strategy.md#provenance-privacy-and-licensing-flags), not
repeated inside the run manifest. Never record tokens or other secret values.

Exact bit-for-bit retraining should not be promised until the pilot proves it.
GPU kernels and upstream training behavior may remain nondeterministic even with
recorded seeds. The initial reproducibility claim is therefore: the same inputs
and environment can be reconstructed, the pipeline rerun, and its result judged
against the same evaluation revision and gates.

## Artifact and restart schema

Use the Bucket only as durable working storage:

```text
runs/<run-id>/
  manifest.json
  resolved-config.yaml
  attempts/<attempt-id>.json
  stages/
    synthesis/<stage-key>/
    features/<stage-key>/
    training/<stage-key>/
    calibration/<stage-key>/
    final-evaluation/<stage-key>/
  releases/<release-id>.json
```

Each stage directory contains its output files, `artifacts.json`, and a
`COMPLETE.json` marker written last. `artifacts.json` records input hashes,
output hashes and types, counts/shapes, stage-specific configuration hash, seed,
code/environment identity, and validation results. `COMPLETE.json` points to the
validated artifact-manifest hash. File or directory existence alone never means
that a stage completed.

Define `stage-key` from the stage name plus its relevant resolved configuration,
input artifact hashes, source revisions, environment digest, and seeds. A rerun
may reuse a stage only when its completion marker and every declared output
validate. Otherwise it writes to a new attempt-specific location. This gives
idempotent phase-level recovery without pretending that unmodified
openWakeWord's in-memory `auto_train` can resume mid-training. The durable
boundaries are therefore:

1. generated and audited WAV sets;
2. augmented feature arrays;
3. exported classifier plus captured training metrics;
4. calibration report with frozen detector policy;
5. final held-out evaluation report.

Hugging Face Job disks disappear when the Job ends, while Bucket results persist,
so important outputs must cross one of these validated boundaries before the
next expensive stage ([Jobs result persistence](https://huggingface.co/docs/hub/jobs-manage#persist-your-results)).

## Validation gates

1. **Local static gate:** configuration parsing, formatting/linting, and unit
   tests pass without network or GPU access.
2. **Container preflight:** the exact training image starts; pinned assets match
   checksums; PyTorch sees CUDA; ONNX Runtime exposes
   `CUDAExecutionProvider`; required imports and Bucket write/read-back succeed.
3. **Tiny end-to-end smoke:** create a minimal clip set, extract features, train
   briefly, export ONNX, validate its declared input/output schema, and run raw
   16 kHz PCM through the full openWakeWord ONNX pipeline. Upstream does not test
   custom `train.py`, and the classifier alone is not a raw-audio model
   ([openWakeWord baseline](openwakeword-baseline.md#onnx-inference-contract)).
4. **Persisted pilot:** run the real stage boundaries and production manifest on
   the small pilot; verify counts, shapes, checksums, and completion markers by
   reading them back outside the Job.
5. **Candidate calibration:** score the immutable calibration revision and
   freeze the selected threshold, trigger policy, preprocessing settings, and
   event-grouping rule.
6. **Final evaluation:** run the frozen candidate once on the separate final
   revision, preserve its metrics and error inventory, and apply the gates in
   [evaluation-methodology.md](evaluation-methodology.md#staged-gates). A failed
   final evaluation remains a recorded non-release candidate; it is not retuned.
7. **Promotion read-back:** upload the complete release directory in one Hub
   commit, record its commit SHA, download that commit independently, and verify
   all checksums and end-to-end inference. The Hub client supports both
   revision-pinned downloads and folder uploads as a single repository commit
   ([download guide](https://huggingface.co/docs/huggingface_hub/guides/download),
   [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)).

## Tests, notebooks, and CI

Minimum unit coverage should include configuration validation/canonical hashing,
manifest schema-version checks, checksum verification, stage-key computation,
reuse/retry decisions, audio metadata validation, feature-array shape/dtype
validation, trigger-event grouping, and release inventory construction.
Lightweight integration tests should exercise a stage with tiny fixtures,
corrupted/incomplete artifact rejection, and ONNX loading/inference. The paid
GPU smoke and pilot are explicit manual gates, not ordinary CI jobs.

Notebooks are thin inspectors. They import `hey_sonny`, take a run ID or manifest
path as input, and visualize or play selected data, artifact health, scores, and
errors. They must not contain a second implementation of generation, training,
evaluation, or promotion; modify canonical data; embed credentials; or supply
unrecorded configuration. Keep committed outputs empty by default, except for a
small deliberately reviewed result that helps explain a decision.

Initial CI should do only what is fast and deterministic: install the lightweight
environment, lint, run unit tests and CPU integration fixtures, check that
notebooks parse/import the package, and validate example configurations. A full
training-image build, external downloads, Hugging Face Jobs, and model training
remain manual until their cost and stability are understood. The CI provider and
registry can be chosen when the private Git remote is created.

## Reproduce and promote a model

To reproduce a candidate, check out the manifest's clean Git commit, use the
recorded image digest, resolve all Hub inputs to the recorded commits, verify
Bucket/input manifests and checksums, then run the saved resolved configuration
and seeds under a new attempt ID. Evaluate the result against the same immutable
evaluation revision; compare metrics and error categories rather than assuming
byte-identical weights.

Promotion is a separate checked command. It accepts one validated run ID and
collects only the classifier ONNX file, resolved inference configuration,
required shared-model references and hashes, evaluation summary, run/provenance
manifest, checksums, license information, and model card. It refuses dirty,
incomplete, or failed runs. After independent read-back succeeds, write an
immutable release receipt containing the Model repository commit SHA under the
originating Bucket run; do not delete or rewrite the finalized run evidence.

## Deferred choices

- Exact Python/CUDA/PyTorch/ONNX Runtime pins: choose them after the container
  smoke test, then lock them.
- Git hosting provider, CI provider, and container registry: a private remote is
  required, but no provider affects the current pipeline contract.
- Configuration/schema library: start with the lightest implementation that can
  validate and reject unknown keys.
- Notebook pairing or output-stripping tool: add one only if notebook diffs
  become noisy.
- Mid-training checkpoints: do not patch `auto_train` initially; add checkpoint
  state only if measured training duration makes replaying the training phase
  materially costly.
- Bitwise determinism: investigate only after functional reproducibility and
  evaluation stability are demonstrated.

## Primary sources

- [openWakeWord source pinned by this project](https://github.com/dscripka/openWakeWord/tree/368c03716d1e92591906a84949bc477f3a834455)
- [Hugging Face Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration)
- [Hugging Face Jobs result persistence](https://huggingface.co/docs/hub/jobs-manage#persist-your-results)
- [Hugging Face Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)
- [`huggingface_hub` revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
- [`huggingface_hub` repository uploads](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Docker image digests](https://docs.docker.com/dhi/core-concepts/digests/)
- [Git revision selection](https://git-scm.com/docs/git-rev-parse)
