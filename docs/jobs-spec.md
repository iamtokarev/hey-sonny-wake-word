# HF Jobs execution spec

How the full run executes on Hugging Face Jobs. The configuration and the
reasoning behind it live in [full-run-plan.md](research/full-run-plan.md); this
document is the executable half and does not restate them.

## Shape: two jobs

Split by iteration rate. Job A's output is stable once it succeeds; Job B is
what you re-run to tune the negative weight, the head, and step counts.
Folding them together means paying ~1.5 h of GPU to regenerate 60,000 WAVs on
every hyperparameter change.

| | Job A — prepare | Job B — train |
| --- | --- | --- |
| Owns | generation, augmentation, feature extraction | training, export, evaluation, promotion |
| Writes to | a new `iamtokarev/hey-sonny-features-*` Dataset repo | `iamtokarev/hey-sonny` (Model repo), or the experiments repo |
| Flavor | `t4-medium` | `t4-small` |
| Runs | once per data change | once per hyperparameter change |
| Peak disk | ~10 GB | ~15 GB |

Job B never holds the WAVs, and it mounts ACAV100M rather than downloading it.

## Step 1 — Preflight

[`scripts/preflight.py`](../scripts/preflight.py) runs twice. The first pass on
`cpu-basic` ($0.01/h) is the cheap one: it resolves the header, imports
everything, and round-trips a scratch file through the Hub. The second pass on
the target GPU flavor costs a few cents and checks the things a CPU flavor
cannot — an accelerator is absent on `cpu-basic`, so the CUDA checks there report
rather than assert.

```bash
hf jobs uv run scripts/preflight.py --flavor cpu-basic --timeout 20m -s HF_TOKEN
hf jobs uv run scripts/preflight.py --flavor a10g-large --timeout 20m -s HF_TOKEN -- --skip-hub
```

It reports pass or fail for each of seven checks and keeps going after a
failure, so one run tells you everything that is wrong:

| Check | Catches |
| --- | --- |
| runtime | A base image that is not Python 3.12, which resurrects the `speexdsp-ns` wheel problem |
| imports | A gap in `openwakeword.data`'s six module-level dependencies, 40 minutes before Job A would hit it |
| accelerator | `onnxruntime` shadowing `onnxruntime-gpu`, which silently moves feature extraction to CPU. Verified green on `t4-small` 2026-08-31: torch 2.13 bundles CUDA 13.0, which satisfies `onnxruntime-gpu==1.29.0`, and `AudioFeatures(device="gpu")` selects `CUDAExecutionProvider` |
| shared models and outbound network | The melspectrogram and embedding models are not bundled in the wheel; they come from a GitHub release host that Jobs make no guarantee of reaching |
| augmentation smoke | The whole `augment_clips` -> `compute_features_from_generator` path at three-clip scale, with both augmentation probabilities forced to 1.0. This is how the torchaudio/torchcodec break was found; the Colab pilot never exercised it |
| patch anchors | A patch that would match nothing and no-op silently |
| upstream inputs | A change to the false-positive array's shape, which every false-accept rate depends on |
| hub round-trip | A token without write permission, or a repo that does not exist |

**Done when** both passes exit `COMPLETED` with all checks green. The script
ends by printing a complete PEP 723 block with every dependency pinned to the
version it resolved; paste that into all three job scripts before any paid run.
An unpinned header resolves differently on a later day, and the manifest's
environment record becomes fiction.

## Step 2 — Job A: prepare

```bash
hf jobs uv run scripts/prepare_job.py \
  --flavor t4-medium --timeout 3h --detach \
  -s HF_TOKEN \
  -- --repo-id iamtokarev/hey-sonny-features-v2 --checkpoint-dir /work/ckpt \
     --augmentation-rounds 3 --background-hours-per-role 4
```

Flags, not a config file; `--help` is the contract. The ones that shape the
data (see [noise-robustness.md](research/noise-robustness.md) for why):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--background-sources` | LibriSpeech `other/*` parquet, FMA sample, AudioSet shards, FSD50K sample | `role=SOURCE` entries, role in `speech`, `music`, `env`; a bare SOURCE is `env`. SOURCE is an https zip or `repo_id::path` to a Hub dataset file (zip, tar, or parquet with an `audio` column; AudioSet rows are filtered by `human_labels` to match the role) |
| `--background-hours-per-role` | 4.0 | cap per role after re-cutting into 10 s segments; every role is then truncated to the smallest count, and a `babble` role of 2-5 overlaid speech segments is synthesised, so the four roles are drawn equally |
| `--augmentation-rounds` | 1 | training clips are augmented this many times, each under a fresh background and SNR |
| `--background-snr-min/max` | -10 / 15 | the mixing range `augment_clips` hardcodes; the released models used 0..20 |
| `--stress-positives` | 24 | held-out clean validation clips published under `stress/` |
| `--adversarial-texts` | 18 phrases, see `--help` | synthesised as negatives in place of a share of the phoneme-overlap texts; name swaps near "sonny", other assistants' wake words, room openers. Never a homophone of the phrase |
| `--adversarial-custom-fraction` | 0.2 | share of adversarial clips drawn from `--adversarial-texts`, scattered over random slots so they spread across speaker pairs |
| `--n-train` / `--max-speakers` / `--val-speakers` | 30000 / 173 / 12 | E3 uses 100000 / 316 / 22; keep `max_speakers` near sqrt(n_train + n_val) or the extra clips repeat the same voices |

In order, the script:

1. clones `piper-sample-generator` at its pinned commit and downloads the
   LibriTTS-R `.pt` — the PyPI wheel drops `piper_train`, which `__main__.py`
   imports, so the clone is required;
2. generates `--n-train` positives and `--n-val` validation positives at
   `--max-speakers`, then resamples to 16 kHz — Piper writes 22050 Hz and
   nothing downstream resamples;
3. generates the same counts of adversarial negatives via
   `generate_adversarial_texts`, with `--adversarial-custom-fraction` of the
   texts replaced by `--adversarial-texts`, and resamples them; once each
   class's 16 kHz split is verified against the quotas the 22 kHz originals
   are deleted (26 GB at E3 scale) and a resume treats the split as the record;
4. augments all four sets through `augment_clips` with the MIT RIRs and the
   balanced four-role background pool, training sets `--augmentation-rounds`
   times; a batch whose GPU augmentation raises a cuFFT/CUDA error is retried
   and then run on the CPU (`robust_augment`), because the feature writer
   sizes its memmap up front and cannot tolerate a missing batch;
5. extracts features with `AudioFeatures(device="gpu")` into memory-mapped
   `.npy` arrays at 16 frames x 96 features;
6. augments the validation positives once more per **condition** — `clean`,
   `env`, `music`, `speech`, `babble`, each against one role only — into
   `positive_val_<cond>_features.npy`, and writes `stress/`: 24 clean held-out
   validation WAVs plus 60 s of each role from segments kept out of training;
7. pushes the arrays, the `stress/` tree and a stage manifest to the Dataset
   repo in one commit, then reads that commit back and verifies checksums.

**Done when** the Dataset repo holds `positive_train` and `adversarial_train`
at `(30000 x rounds, 16, 96)`, the two validation arrays and the five condition
arrays at `(2000, 16, 96)`, `stress/manifest.json` with four pools, the
manifest's `pool` entry shows four roles with the same file count, and the
read-back checksums match.

### Confirm the execution provider

`AudioFeatures(device="gpu")` asks onnxruntime for `CUDAExecutionProvider`.
Plain `onnxruntime` does not offer it and silently falls back to CPU — which is
what happened throughout the Colab pilot, on a T4. Assert it and print the
result, so a CPU fallback is visible rather than merely slow:

```python
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
```

`openwakeword` requires `onnxruntime>=1.10,<2`, so `onnxruntime-gpu` installs
beside it and shadows the module. Treat a fallback as a signal to raise vCPU
count rather than as a stage failure: `compute_features_from_generator` takes
`ncpu` and scales with cores.

## Step 3 — Job B: train

```bash
hf jobs uv run scripts/train_job.py \
  --flavor t4-small --timeout 2h --detach \
  -s HF_TOKEN \
  -v hf://datasets/davidscripka/openwakeword_features:/upstream \
  -v hf://datasets/iamtokarev/hey-sonny-features-v2b:/features \
  -v ./out:/out:rw \
  -- --checkpoint-dir /out/jobb --features-repo iamtokarev/hey-sonny-features-v2b \
     --model-type rnn --steps 200000 --refine-steps 20000 \
     --max-negative-weight 500 --escalation-factor 1.0 \
     --baseline-model iamtokarev/hey-sonny
```

Those are the promoted model's flags; `--repo-id` defaults to
`iamtokarev/hey-sonny`, so pass `--repo-id iamtokarev/hey-sonny-experiments
--run-name <name>` for anything that is not a deliberate promotion.

Rehearse first, at about two cents:

```bash
hf jobs uv run scripts/train_job.py \
  --flavor t4-small --timeout 45m --detach \
  -s HF_TOKEN \
  -v hf://datasets/davidscripka/openwakeword_features:/upstream \
  -v hf://datasets/iamtokarev/hey-sonny-features-v2b:/features \
  -v ./out:/out:rw \
  -- --repo-id iamtokarev/hey-sonny-rehearsal --checkpoint-dir /out/jobb-rehearsal \
     --steps 400 --refine-steps 100 --limit-fp-frames 60000
```

The rehearsal exercises every stage including the Hub push, in 90 seconds. It
found three faults the full run would have hit: the ACAV array's real shape, a
frozen ONNX batch axis, and an opset claim the exporter does not honour. The
model it produces is worthless — 400 steps — and `--limit-fp-frames` changes the
false-accept denominator, so read the rehearsal for plumbing, never for quality.

Both jobs take flags, not a config file. There is no `configs/` mount and no
`--config`; `--help` on either script is the contract. The ones worth knowing:
`--baseline-model REPO` scores that repo's ONNX on the same validation sets and
stress grid as a reference row named `baseline` (never published, never
selected) — this is the only valid way to compare two models, since validation
clips differ between feature sets. `--select-condition` ranks candidates on one
of Job A's condition arrays instead of the mixed set. `--model-type rnn` swaps
the flattened MLP head for upstream's 2-layer bidirectional LSTM over the 16
frames; `--layer-size` is then ignored, and `export_onnx` handles the LSTM
(dynamic batch, opset 17).

In order, the script:

1. builds the batch generator over `/upstream/openwakeword_features_ACAV100M_2000_hrs_16bit.npy`
   and the four arrays under `/features`, using upstream's `batch_n_per_class`
   keys — `positive` and `adversarial_negative` are named by `train.py` and the
   generator fails on any other spelling;
2. measures the false-positive validation set's real duration from its array
   shape rather than inheriting `auto_train`'s hardcoded `11.3`:
   `481345 frames x 0.08 s = 10.70 h`;
3. calls `train_model` three times with the sequence schedule from
   `full-run-plan.md`, passing the measured `val_set_hrs`, and writes a
   checkpoint to `/out` after each sequence;
4. selects and averages checkpoints, exports ONNX, and scores the result with a
   corrected event-grouping false-positive counter;
5. pushes the ONNX, metrics, run manifest, and model card to the Model repo in
   one commit, then downloads that commit independently and verifies checksums
   and end-to-end inference on raw 16 kHz PCM.

**Done when** the Model repo commit round-trips, the manifest records
`JOB_ID`, the resolved input commits, the locked dependency versions, and every
seed, and the metrics file reports recall and false-accepts per hour under both
the corrected counter and upstream's frame counter. On a v2 feature set it also
carries `recall_at_fa_by_condition` (recall at each budget per condition),
`stress` (per role and SNR: min, median, and the share of held-out clips above
each budget's threshold) and, with `--baseline-model`, a `baseline` block
measured in the same job.

### A checksum round-trip is not proof the model works

Found 2026-09-02, after five models had been published. `torch.onnx.export` with
the dynamo exporter defaults to `external_data=True`: it writes the graph to
`<name>.onnx` and the weights to a sibling `<name>.onnx.data`. Our export
produced a **3 KB** `hey_sonny.onnx`, and `stage_publish` uploaded only the file
it knew about.

Everything downstream still passed. The ONNX/torch parity check and the
end-to-end `openwakeword.Model` check both ran inside the job, where the sibling
was sitting in the same directory. The read-back verification downloaded every
published file and matched its checksum — correctly, because the bytes that were
uploaded did arrive. What nobody checked was whether the *published set* was
complete, and it was not: the model loads nowhere.

Three guards now, and the third is the one that would have caught it:

1. `external_data=False` on export, so the weights are in the file;
2. assertions that no initializer has `data_location == EXTERNAL` and that the
   file is larger than the weights it must contain (~200 KB here, against 3 KB
   for a graph-only export);
3. **`raw_pcm_check` runs against the downloaded copy**, not the local one, so
   publication is verified by using the artifact rather than by hashing it.

The general rule: verify an artifact by exercising it from where a consumer
would get it. Checksums prove transport, not usability.

## The dependency header

[`scripts/preflight.py`](../scripts/preflight.py) carries the authoritative
header, pinned to the versions its 2026-08-31 run resolved. PEP 723 requires the
block inline, so each job script holds a copy; copy it from preflight rather than
composing a new one, and re-run preflight to regenerate the pins after any
deliberate change.

`openwakeword[full]` is unresolvable on Python 3.12: it pins
`tensorflow-cpu==2.8.1` (Python 3.10 and older), `protobuf<4`, and
`onnx==1.14.0`. The header therefore lists the subset the code actually imports.
The base requirements do resolve — `speexdsp-ns` and `ai-edge-litert` both
publish cp312 manylinux wheels, and neither publishes anything past cp312, which
is why the Colab notebook needed `--no-deps` and this path does not.

Two pins are load-bearing and easy to raise by accident:

- **`scipy==1.16.3`** — `data.py` imports `acoustics` at module level, and
  `acoustics` 0.2.6 imports `scipy.special.sph_harm`, removed in scipy 1.17.
  Raising it re-breaks `import openwakeword.data`.
- **`onnxscript==0.7.1`** — `torch>=2.9` routes `torch.onnx.export` through the
  dynamo exporter, which imports it.

### Keep the working tree off the bucket mount

Measured 2026-09-01. A read-write `--volume` mount is durable, but it is not a
POSIX filesystem and it is not fast. Using it as Job A's working tree produced
both a performance and a correctness problem:

| Working tree | features stage, 160 clips |
| --- | --- |
| `/out` (bucket mount) | 35 s |
| `/work` (local disk) | 6 s |

Worse than slow, reads after writes were **intermittently inconsistent**. In one
run `sources` wrote 270 room impulse responses and `features` listed zero of
them moments later, in the same job. `augment_clips` treats an empty RIR list as
"skip reverberation" rather than an error, so that run produced correctly shaped
feature arrays with the augmentation silently missing. A different run lost a
whole clip directory and failed with `StopIteration` from inside
`compute_features_from_generator`, which says nothing about the cause.

So: generate and augment on local ephemeral disk, and copy only the stage
markers and the finished feature arrays to the mount. The clips are regenerable
from the pinned generator, config and seed, which is the same argument that
retired the Storage Bucket. Assert the augmentation inputs where they are
*used*, not only where they are produced.

## Hardware: pick by bottleneck

Bigger GPUs help exactly one stage. Buy vCPU for Job A and leave Job B alone.

Measured, not estimated: Job A finished in 16 minutes on `t4-medium` for about
$0.16, and Job B trains at 55–70 steps/s on `t4-small`, putting 60,000 steps
near 25 minutes for about $0.20. The paragraphs below argued for `a10g-large`
and `l4x1` before either job had run. Neither is wrong about *where* the
bottleneck is; both overbought against it.

| Stage | Bottleneck | Implication |
| --- | --- | --- |
| Piper generation | GPU | Already fast: 900 clips in 14 s on a T4 |
| Augmentation | CPU — `audiomentations` convolves and resamples on the host | vCPU count sets the pace |
| Feature extraction | GPU with `onnxruntime-gpu`, otherwise CPU | Confirm the provider before blaming hardware |
| Training | Sequential mmap reads over 17.28 GB, at ~3.1 MB per step | GPU is near-idle; a larger one buys nothing. RAM for page cache is the only argument for a bigger flavor, and 11 passes over a 17 GB file defeat any of them |

Feature extraction does reach the GPU — preflight confirmed
`CUDAExecutionProvider` on a T4 — so vCPU count matters only for the
augmentation stage, and a CPU-only flavor is not a candidate.

`a10g-large` gives Job A 12 vCPU and 46 GB RAM for $1.50/h, against `l4x1`'s
8 vCPU for $0.80/h. It finishes sooner for roughly the same total. Weigh
capacity as well as price: on 2026-08-31 `a10g-large` sat in `SCHEDULING` for
over nine minutes while `t4-small` allocated in twenty-five seconds. Queue time
is unbilled, so a long wait costs patience rather than money. Above that —
`l40sx1`, `a100-large`, `rtx-pro-6000` — the extra spend lands on a GPU that is
not the bottleneck in either stage.

Verify flavors and prices with `hf jobs hardware` immediately before submitting;
the published table is mutable.

## Volumes, secrets, and restart

`-s HF_TOKEN` passes the local token, encrypted server-side. With no writable
Hub mount, that token is what makes persistence possible at all, not just the
final promotion.

Hub mounts (`hf://datasets/...`) are read-only and **lazily fetched**, so
mounting ACAV100M keeps a 17.28 GB file off the disk budget until it is read;
`mmap_batch_generator` walks it sequentially, which suits lazy fetch.

A local directory source (`-v ./src:/app/src`) is synced to an auto-created
private `jobs-artifacts` bucket, then mounted. Add `:rw` and the job can write
back; the CLI prints the `hf buckets sync` command to retrieve the results.
That makes `-v ./out:/out:rw` a durable checkpoint target, which is why Job B
writes one after each training sequence rather than only at the end. This
revises an earlier decision to skip durable storage entirely: a dying job no
longer loses everything, at the price of one flag.

Set `--timeout` on every submission. The default is 30 minutes and a training
run silently dies at it.

`JOB_ID` is provided in the container environment; record it as the manifest's
`attempt_id`. `ACCELERATOR`, `CPU_CORES`, and `MEMORY` are provided too, and
belong in the environment block of the manifest.

Sync the source from a clean tree. Promotion from a dirty working tree is not
allowed, since a local-directory mount will happily carry uncommitted edits
into a paid run. This guard belongs at submit time, not in preflight —
preflight runs in the container and has no repository to inspect.
Gate the submission of Jobs A and B on `git status --porcelain` being empty, and
record the commit SHA in the manifest.
