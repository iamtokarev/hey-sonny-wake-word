# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "openwakeword @ git+https://github.com/dscripka/openWakeWord.git@368c03716d1e92591906a84949bc477f3a834455",
#   "torch==2.13.0",
#   "torchaudio==2.11.0",
#   "torchinfo==1.8.0",
#   "torchmetrics==1.9.0",
#   "speechbrain==1.1.1",
#   "audiomentations==0.43.1",
#   "torch-audiomentations==0.12.0",
#   "acoustics==0.2.6",
#   "pronouncing==0.3.0",
#   "mutagen==1.48.1",
#   "pyyaml==6.0.3",
#   # acoustics 0.2.6 imports scipy.special.sph_harm, removed in scipy 1.17.
#   "scipy==1.16.3",
#   "onnxruntime-gpu==1.29.0",
#   "onnxscript==0.7.1",
#   "soundfile==0.14.0",
#   "librosa==0.11.0",
#   "huggingface_hub==1.29.0",
#   "numpy==2.5.2",
#   "requests==2.34.2",
# ]
#
# [tool.uv]
# # Kept identical to Job A even though piper is not imported here: openwakeword
# # requires `onnxruntime>=1.10,<2`, which uv would otherwise install beside
# # onnxruntime-gpu. The CPU build shadows the GPU one, CUDAExecutionProvider
# # disappears, and the ONNX parity check at the end runs on the CPU without
# # saying so. The marker below is never true, so uv drops the requirement.
# override-dependencies = ["onnxruntime ; python_version < '3.0'"]
# ///
"""Job B: train, export, evaluate and publish the Hey Sonny classifier.

Consumes the four feature arrays Job A published, plus upstream's precomputed
ACAV100M negatives and false-positive validation set. Produces an ONNX model,
a metrics file and a run manifest in the Model repo.

Contract, flavors and completion criteria: docs/jobs-spec.md
Configuration and the code boundary: docs/research/full-run-plan.md

This drives `openwakeword.train.Model` rather than calling its `auto_train`.
That buys three things auto_train cannot give: a measured `val_set_hrs` instead
of its hardcoded 11.3, a checkpoint boundary after each training sequence, and
a negative-weight escalation that reads the false-positive rate the run actually
achieved. See `sequence_schedule` for the third.

Stages are restartable, and so are the individual training sequences.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

SR = 16000

# One embedding frame per 8 mel frames of 10 ms each. Job A asserts the same
# relation from the other direction; both must agree or the arrays Job A wrote
# do not fit the model this job builds.
SECONDS_PER_FRAME = 0.08

STAGES = ["inputs", "train", "evaluate", "publish"]

FEATURE_FILES = ("positive_train", "positive_val", "adversarial_train", "adversarial_val")

# Optional per-condition validation positives from Job A (positive_val_<cond>).
# The mixed `positive_val` says how much recall a model has; these say under
# which interference it loses it. The promoted 2026-09-02 model scored 0.56 on
# the mixed set and ~0.02 with a second voice in the room, and nothing in the
# mixed number could have shown that.
VAL_CONDITIONS = ("clean", "env", "music", "speech", "babble")

# Raw-audio stress grid, scored through the exported ONNX at fixed SNRs on the
# `stress/` assets Job A publishes. Roles are background roles; SNR is speech
# RMS over interference RMS, in dB.
STRESS_ROLES = ("speech", "babble", "music", "env")
STRESS_SNRS = (20, 15, 10, 5, 0)


# --------------------------------------------------------------------------
# Preconditions, carried over from Job A. Both were found by preflight rather
# than by reasoning; see openwakeword-baseline.md.
# --------------------------------------------------------------------------

def on_gpu() -> bool:
    """True when this flavor has a CUDA device. ACCELERATOR is coarse: 'cpu' or 'gpu'."""
    return os.environ.get("ACCELERATOR", "none") not in {"none", "cpu", "unset", ""}


def patch_torchaudio() -> None:
    """Route `torchaudio.load` and `torchaudio.info` through soundfile.

    torchaudio 2.11 delegates `load` to torchcodec, which is absent, and has
    removed `info`. This job loads no audio of its own, but importing
    `openwakeword.data` pulls torch_audiomentations, and the end-to-end check
    at the end goes through `openwakeword.Model`.
    """
    import types

    import soundfile
    import torch
    import torchaudio

    def load(path, *args, **kwargs):
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), rate

    def info(path, *args, **kwargs):
        try:
            meta = soundfile.info(str(path))
        except Exception as exc:
            raise RuntimeError(f"could not read audio metadata from {path}") from exc
        return types.SimpleNamespace(
            sample_rate=meta.samplerate, num_frames=meta.frames,
            num_channels=meta.channels, bits_per_sample=16, encoding=meta.subtype)

    torchaudio.load = load
    torchaudio.info = info


def assert_gpu_features() -> str:
    """Fail loudly when the ONNX parity check would silently land on the CPU."""
    import torch          # noqa: F401  keep first
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    if not on_gpu():
        return f"CPU flavor; providers={providers}"
    assert "CUDAExecutionProvider" in providers, (
        f"GPU flavor but onnxruntime offers {providers}.")
    return f"{torch.cuda.get_device_name(0)}; providers={providers}"


# --------------------------------------------------------------------------
# Stage bookkeeping. Same shape as Job A, different payload: this job's
# expensive intermediate is a training checkpoint, not a feature array.
# --------------------------------------------------------------------------

CHECKPOINTED = ("*.COMPLETE.json", "*.pt", "*.onnx", "manifest.json", "metrics.json")


def sync_checkpoint(work: Path, ckpt: Path, direction: str) -> list[str]:
    """Copy stage markers, sequence checkpoints and outputs between local disk
    and the bucket mount.

    The working tree stays on local ephemeral disk. A read-write `--volume`
    mount is durable but is neither POSIX nor fast, and Job A found its
    reads-after-writes intermittently inconsistent; see jobs-spec.md.
    """
    src, dst = (work, ckpt) if direction == "save" else (ckpt, work)
    if not src.is_dir():
        return []
    dst.mkdir(parents=True, exist_ok=True)
    moved = []
    for pattern in CHECKPOINTED:
        for path in sorted(src.glob(pattern)):
            target = dst / path.name
            if direction == "save" and target.is_file() and target.stat().st_size == path.stat().st_size:
                continue
            shutil.copy2(path, target)
            moved.append(path.name)
    return moved


def marker(work: Path, stage: str) -> Path:
    return work / f"{stage}.COMPLETE.json"


def stage_done(work: Path, stage: str) -> bool:
    return marker(work, stage).is_file()


def complete(work: Path, stage: str, **detail) -> None:
    """Write the marker last, so a partial stage never looks finished."""
    payload = {"stage": stage, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **detail}
    marker(work, stage).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"[{stage}] complete: {json.dumps(detail, sort_keys=True, default=str)[:400]}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def build_config(args) -> dict:
    return {
        "repo_id": args.repo_id,
        "features_repo": args.features_repo,
        "model_name": args.model_name,
        "window_frames": args.window_frames,
        "layer_size": args.layer_size,
        "model_type": args.model_type,
        "steps": args.steps,
        "refine_steps": args.refine_steps,
        "max_negative_weight": args.max_negative_weight,
        "escalation_factor": args.escalation_factor,
        "merge_rule": args.merge_rule,
        "merge_top_k": args.merge_top_k,
        "select_fa_gate": args.select_fa_gate,
        "target_fp_per_hour": args.target_fp_per_hour,
        "min_recall": args.min_recall,
        "lr": args.lr,
        "acav_batch_rows": args.acav_batch_rows,
        "positive_per_batch": args.positive_per_batch,
        "adversarial_per_batch": args.adversarial_per_batch,
        "grouping_seconds": GROUPING_SECONDS,
        "acav_holdout_hours": args.acav_holdout_hours,
        "select_on": args.select_on,
        "select_condition": args.select_condition,
        "baseline_model": args.baseline_model,
        "seed": args.seed,
    }


# --------------------------------------------------------------------------
# The corrected false-positive counter
# --------------------------------------------------------------------------

def count_activations_from_indices(above_idx, grouping_window: int = 50) -> int:
    """Event count from the indices of above-threshold frames.

    O(frames above threshold) rather than O(frames), which is what makes the
    exact-ROC sweep below affordable: in the region that matters only a handful
    of the 481,330 frames are above threshold.
    """
    events, last = 0, -(1 << 60)
    for i in above_idx:
        if i > last + grouping_window:
            events += 1
            last = int(i)
    return events


def count_activations(scores, threshold: float, grouping_window: int = 50) -> int:
    """Count activation *events* above `threshold`, not frames above it.

    `openwakeword.metrics.get_false_positives` intends this and does not do it:
    its suppression slice is bounded by `len(transitions)` where it means
    `len(bin_pred)`, so the slice meant to swallow a detection's trailing frames
    is empty or reversed and every frame above threshold is counted separately.
    It also matches only "01" transitions, so a run starting at frame 0 is
    missed entirely.

    `grouping_window` is upstream's own default: 50 frames, 4.0 s at 80 ms per
    frame. Two activations closer together than that are one event, which is the
    right semantics for a wake word -- a device cannot usefully wake twice in
    four seconds.
    """
    import numpy as np

    above = np.asarray(scores).reshape(-1) >= threshold
    return count_activations_from_indices(np.flatnonzero(above), grouping_window)


# --------------------------------------------------------------------------
# Data plumbing
# --------------------------------------------------------------------------

class DeviceWindows:
    """Re-iterable batches of stride-1 windows over a 2-D feature array.

    Upstream materialises the whole windowed false-positive set as one tensor
    and hands it over in a single batch: 481,329 windows of 16x96 float32 is
    2.96 GB, built by a Python list comprehension over half a million slices,
    and copied to the device on every validation pass. This keeps only the
    (481345, 96) base on the device -- 185 MB -- and cuts windows from it there,
    so a validation pass moves nothing across PCIe.

    `train_model` and `_select_best_model` only ever iterate this and call
    `.to(device)` on the members, which is a no-op for tensors already there.
    """

    def __init__(self, array, window: int, batch: int, device, label: float = 0.0):
        import torch

        self.base = torch.from_numpy(array.astype("float32", copy=False)).to(device)
        self.window = window
        self.batch = batch
        self.n = self.base.shape[0] - window + 1
        assert self.n > 0, f"array of {self.base.shape[0]} frames is shorter than the {window}-frame window"
        self.labels = torch.full((batch,), label, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for start in range(0, self.n, self.batch):
            stop = min(start + self.batch, self.n)
            # unfold is a view; the copy happens once, on device, at .contiguous().
            chunk = self.base[start:stop + self.window - 1]
            windows = chunk.unfold(0, self.window, 1).permute(0, 2, 1).contiguous()
            yield windows, self.labels[: windows.shape[0]]


# An activation and the four seconds after it are one event. Expressed in
# seconds because the two false-positive sets step through audio at different
# rates: upstream's file is flat frames turned into stride-1 windows 80 ms
# apart, while ACAV100M ships pre-windowed at 1.28 s per row. Fifty frames and
# three ACAV rows are the same four seconds; using 50 for both would suppress
# 64 seconds of ACAV audio per activation and undercount by an order of
# magnitude.
GROUPING_SECONDS = 4.0


class PrewindowedSet:
    """Batches over an already-windowed feature array, kept off the device.

    Used for the ACAV100M holdout, which is (N, 16, 96) float16 on a lazily
    fetched mount and far too large to hold resident the way `DeviceWindows`
    holds upstream's 185 MB set.
    """

    def __init__(self, array, batch: int, device, label: float = 0.0):
        self.array, self.batch, self.device, self.label = array, batch, device, label
        self.n = int(array.shape[0])

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        import numpy as np
        import torch

        for start in range(0, self.n, self.batch):
            block = np.asarray(self.array[start:start + self.batch], dtype=np.float32)
            x = torch.from_numpy(block).to(self.device)
            yield x, torch.full((x.shape[0],), self.label, dtype=torch.float32,
                                device=self.device)


class EvalSet:
    """A false-positive set, its true duration, and its event-grouping window."""

    def __init__(self, name: str, windows, hours: float, step_seconds: float, detail: dict):
        self.name = name
        self.windows = windows
        self.hours = hours
        self.step_seconds = step_seconds
        self.grouping = max(1, round(GROUPING_SECONDS / step_seconds))
        self.detail = dict(detail, hours=round(hours, 4), grouping=self.grouping,
                           step_seconds=step_seconds)


def window_transform(n_frames: int):
    """Upstream's `f`: reshape a flat 2-D negative batch into model-sized windows.

    A no-op for the published ACAV100M file, which is already `(5625000, 16, 96)`
    -- pre-windowed, not the flat `(N, 96)` the transform exists to reshape. So
    `batch_n_per_class` means what it says: 1024 + 50 + 50 = 1124 examples at
    20:1 negatives to positives. Kept anyway, because it is what makes a flat
    `--acav-file` work, and because dropping it would silently change the batch
    if the array ever ships in the other layout.
    """
    import numpy as np

    def f(x, n=n_frames):
        if n > x.shape[1] or n < x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i + n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    return f


def build_loaders(cfg: dict, paths: dict, device):
    """Assemble the training generator and the two validation sets."""
    import numpy as np
    import torch
    from openwakeword.data import mmap_batch_generator

    n_frames = cfg["window_frames"]

    # Keys are ours to choose -- the generator is key-agnostic -- but `positive`
    # and `adversarial_negative` are what train.py and the config schema name,
    # and the manifest is easier to compare against upstream runs if we match.
    data_files = {
        "ACAV100M_sample": str(paths["acav"]),
        "positive": str(paths["positive_train"]),
        "adversarial_negative": str(paths["adversarial_train"]),
    }
    n_per_class = {
        "ACAV100M_sample": cfg["acav_batch_rows"],
        "positive": cfg["positive_per_batch"],
        "adversarial_negative": cfg["adversarial_per_batch"],
    }
    # Only the flat ACAV array needs reshaping; Job A's arrays are already
    # (N, 16, 96) windows.
    data_transforms = {"ACAV100M_sample": window_transform(n_frames)}
    label_transforms = {
        "positive": lambda x: [1 for _ in x],
        "ACAV100M_sample": lambda x: [0 for _ in x],
        "adversarial_negative": lambda x: [0 for _ in x],
    }

    generator = mmap_batch_generator(
        data_files, n_per_class=n_per_class,
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms)

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, gen):
            self.generator = gen

        def __iter__(self):
            return self.generator

    # num_workers=1 deliberately. An IterableDataset is copied into every
    # worker, and this generator carries its own per-file cursor starting at
    # zero, so N workers emit the same batches N times over. Upstream sets
    # num_workers = cpu_count // 2 and trains on each batch that many times.
    X_train = torch.utils.data.DataLoader(
        IterDataset(generator), batch_size=None, num_workers=1, prefetch_factor=16)

    # Balanced positive/negative validation set. Upstream reads its metrics from
    # whatever the *last* batch was, so this has to stay a single batch.
    pos = np.load(paths["positive_val"])
    neg = np.load(paths["adversarial_val"])
    labels = np.hstack((np.ones(pos.shape[0]), np.zeros(neg.shape[0]))).astype(np.float32)
    X_val = [(torch.from_numpy(np.vstack((pos, neg))).to(device),
              torch.from_numpy(labels).to(device))]

    fp_raw = np.load(paths["false_positive"])
    if cfg.get("limit_fp_frames"):
        fp_raw = fp_raw[: cfg["limit_fp_frames"]]
    upstream_windows = DeviceWindows(fp_raw, n_frames, cfg["fp_batch"], device)

    # The duration auto_train hardcodes as 11.3. Measured, it is 10.70 h for
    # upstream's own file, so every rate auto_train reports is 5.6% optimistic.
    val_set_hrs = len(upstream_windows) * SECONDS_PER_FRAME / 3600
    fp_sets = {"upstream": EvalSet("upstream", upstream_windows, val_set_hrs,
                                   SECONDS_PER_FRAME,
                                   {"source": str(paths["false_positive"]),
                                    "frames": int(fp_raw.shape[0]),
                                    "windows": len(upstream_windows)})}

    # A second, much larger false-positive set, carved off the end of ACAV100M
    # and withheld from training.
    #
    # Why: at 0.2 FA/h over upstream's 10.70 h a model is allowed *two* events.
    # Everything downstream inherits that. Measured 2026-09-02, choosing a
    # threshold for 0.2 FA/h on half that file yields 0.327 FA/h on the other
    # half, and one fixed model's held-out recall ranged 0.234-0.566 across
    # splits. At that resolution the experiments in docs/experiment-plan.md
    # cannot be told apart, which makes this a prerequisite rather than a
    # refinement.
    #
    # ACAV is the training negatives' own corpus, so this measures resolution,
    # not comparability with published openWakeWord figures. Both sets are
    # always reported; neither replaces the other.
    acav = np.load(paths["acav"], mmap_mode="r")
    holdout_hours = cfg.get("acav_holdout_hours", 0.0)
    if holdout_hours and acav.ndim == 3:
        seconds_per_row = acav.shape[1] * SECONDS_PER_FRAME
        rows = int(holdout_hours * 3600 / seconds_per_row)
        cutoff = acav.shape[0] - rows
        assert cutoff > 0, (
            f"holdout of {holdout_hours} h wants {rows} rows but ACAV has "
            f"{acav.shape[0]}")

        # Withhold it from training. Slicing a memmap yields a view, so this
        # costs nothing and the generator's sequential walk simply wraps sooner.
        key = "ACAV100M_sample"
        generator.data[key] = generator.data[key][:cutoff]
        generator.shapes[key] = (cutoff,) + tuple(generator.shapes[key][1:])
        assert generator.data[key].shape[0] == cutoff

        holdout = PrewindowedSet(acav[cutoff:], cfg["fp_batch"], device)
        fp_sets["acav_holdout"] = EvalSet(
            "acav_holdout", holdout, rows * seconds_per_row / 3600, seconds_per_row,
            {"source": str(paths["acav"]), "rows": rows, "train_rows": cutoff})
        print(f"[loaders] ACAV holdout: {rows} rows = "
              f"{rows * seconds_per_row / 3600:.1f} h withheld from training; "
              f"{cutoff} rows remain for training", flush=True)
    elif holdout_hours:
        print(f"[loaders] ACAV is {acav.ndim}-D; holdout needs the pre-windowed "
              f"layout, skipping", flush=True)

    return X_train, X_val, fp_sets, val_set_hrs, {
        "positive_val": list(pos.shape), "adversarial_val": list(neg.shape),
        "fp_frames": int(fp_raw.shape[0]), "fp_windows": len(upstream_windows),
        "val_set_hrs": round(val_set_hrs, 4),
        "fp_sets": {name: st.detail for name, st in fp_sets.items()},
    }


# --------------------------------------------------------------------------
# Stage 1: inputs
# --------------------------------------------------------------------------

def localise_features(paths: dict, local: Path) -> dict:
    """Copy Job A's arrays off the Hub mount onto local disk before training.

    The `/features` volume is fetched lazily, and `mmap_batch_generator` reads
    it 50 rows at a time. With the 184 MB v1 arrays that was invisible; with
    the 553 MB v2 arrays every slice became a network read, training ran at
    1.8 s/step instead of 75 steps/s, and the DataLoader worker died after
    ~12 minutes -- twice, at the same point. ACAV100M stays on its mount: it is
    17 GB, read sequentially, and has served ten runs.
    """
    import time

    local.mkdir(parents=True, exist_ok=True)
    out, copied, started = dict(paths), 0, time.time()
    for key, src in paths.items():
        src = Path(src)
        if key == "stress_dir":
            if src.is_dir():
                dst = local / "stress"
                if not (dst / "manifest.json").is_file():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                out[key] = dst
            continue
        if not src.is_file():
            continue
        dst = local / src.name
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            copied += dst.stat().st_size
        out[key] = dst
    if copied:
        secs = max(time.time() - started, 1e-3)
        print(f"[inputs] copied {copied / 1e6:.0f} MB of features to {local} "
              f"in {secs:.0f} s ({copied / 1e6 / secs:.0f} MB/s)", flush=True)
    return out


def stage_inputs(cfg: dict, work: Path, paths: dict) -> dict:
    import numpy as np

    detail = {"accelerator": assert_gpu_features()}

    missing = [str(p) for k, p in paths.items() if k != "stress_dir" and not Path(p).is_file()]
    assert not missing, (
        f"missing inputs: {missing}. Job A's arrays arrive on the /features "
        f"mount and upstream's on /upstream; check the --volume flags.")

    n_frames = cfg["window_frames"]
    shapes = {}
    for name in FEATURE_FILES:
        arr = np.load(paths[name], mmap_mode="r")
        shapes[name] = list(arr.shape)
        assert arr.ndim == 3 and arr.shape[1:] == (n_frames, 96), (
            f"{name} has shape {arr.shape}, expected (N, {n_frames}, 96). "
            f"Job A and Job B disagree about the window.")

    conditions = [c for c in VAL_CONDITIONS if f"positive_val_{c}" in paths]
    for cond in conditions:
        arr = np.load(paths[f"positive_val_{cond}"], mmap_mode="r")
        shapes[f"positive_val_{cond}"] = list(arr.shape)
        assert arr.shape == tuple(shapes["positive_val"]), (
            f"positive_val_{cond} has shape {arr.shape}, positive_val {shapes['positive_val']}; "
            "the conditions must be the same clips under different backgrounds.")
    detail["val_conditions"] = conditions
    stress_dir = Path(paths.get("stress_dir", "/nonexistent"))
    detail["stress_assets"] = (stress_dir / "manifest.json").is_file()
    if not conditions:
        print("[inputs] no per-condition validation arrays; this is a v1 feature set "
              "and recall will be reported on the mixed set only", flush=True)

    fp = np.load(paths["false_positive"], mmap_mode="r")
    assert fp.ndim == 2 and fp.shape[1] == 96, f"false-positive set has shape {fp.shape}"
    shapes["false_positive"] = list(fp.shape)

    # The published file is 3-D and pre-windowed; a flat 2-D array is also
    # valid and gets reshaped by `window_transform`. Both must end at 96
    # features and, when already windowed, at this job's window length.
    acav = np.load(paths["acav"], mmap_mode="r")
    assert acav.shape[-1] == 96, f"ACAV set has shape {acav.shape}, expected 96 features"
    assert acav.ndim in (2, 3), f"ACAV set has shape {acav.shape}"
    assert acav.ndim == 2 or acav.shape[1] == n_frames, (
        f"ACAV set is pre-windowed at {acav.shape[1]} frames but this job trains "
        f"on {n_frames}. Windowed negatives cannot be re-cut.")
    shapes["acav"] = list(acav.shape)
    detail["acav_pre_windowed"] = acav.ndim == 3
    detail["acav_negatives_per_batch"] = (
        cfg["acav_batch_rows"] if acav.ndim == 3
        else max(0, (cfg["acav_batch_rows"] - n_frames) // n_frames))
    detail["realised_batch"] = (detail["acav_negatives_per_batch"]
                                + cfg["positive_per_batch"] + cfg["adversarial_per_batch"])

    # A lazily-fetched Hub mount serves the 17.28 GB ACAV file on demand. Read a
    # stripe now and time it: the generator will walk about 61M of these rows,
    # and a mount that reads at a few MB/s turns a 20-minute job into an
    # overnight one. Better to see the number in the first 30 seconds.
    started = time.time()
    bytes_per_row = int(np.prod(acav.shape[1:])) * acav.dtype.itemsize
    probe_rows = min(max(1, 50_000_000 // bytes_per_row), acav.shape[0])
    checksum = float(np.asarray(acav[:probe_rows]).astype("float32").sum())
    elapsed = max(time.time() - started, 1e-6)
    mb = probe_rows * bytes_per_row / 1e6
    detail["acav_read_mb_per_s"] = round(mb / elapsed, 1)
    detail["acav_probe_checksum"] = checksum
    print(f"[inputs] ACAV mount reads at {detail['acav_read_mb_per_s']} MB/s "
          f"({mb:.0f} MB in {elapsed:.1f} s)", flush=True)

    detail["shapes"] = shapes
    detail["val_set_hrs"] = round(fp.shape[0] * SECONDS_PER_FRAME / 3600, 4)
    return detail


# --------------------------------------------------------------------------
# Stage 2: train
# --------------------------------------------------------------------------

def sequence_schedule(cfg: dict) -> list[dict]:
    """The three sequences auto_train runs, with the weight escalation fixed.

    auto_train doubles `max_negative_weight` between sequences when
    `self.best_val_fp > target_fp_per_hour`. `best_val_fp` is initialised to
    1000 in `Model.__init__` and never assigned again, so that test is always
    true and the doubling is unconditional. Here the decision is deferred to
    `run_sequences`, which reads the rate the previous sequence actually
    measured.
    """
    steps = cfg["steps"]
    refine = cfg["refine_steps"]
    lr = cfg["lr"]
    return [
        {"name": "seq1", "steps": steps, "lr": lr, "val_from": steps - int(steps * 0.25)},
        {"name": "seq2", "steps": refine, "lr": lr / 10, "val_from": 1},
        {"name": "seq3", "steps": refine, "lr": lr / 100, "val_from": 1},
    ]


def save_sequence(oww, work: Path, name: str, history: dict) -> Path:
    """Persist a sequence boundary as plain tensors.

    Not `Model.save_model`: that pickles the module object, and the `Net` class
    is defined inside `Model.__init__`, so the pickle has no importable path to
    it. State dicts are ordinary tensor mappings and reload anywhere.
    """
    import numpy as np
    import torch

    path = work / f"{name}.pt"
    torch.save({
        "model": oww.model.state_dict(),
        "optimizer": oww.optimizer.state_dict(),
        "best_models": [m.state_dict() for m in oww.best_models],
        "best_model_scores": [{k: float(v) for k, v in s.items()} for s in oww.best_model_scores],
        "history": {k: np.asarray(v).astype(float).tolist() for k, v in history.items()},
    }, path)
    return path


def load_sequence(oww, path: Path) -> None:
    import collections
    import copy as _copy

    import torch

    blob = torch.load(path, map_location=oww.device, weights_only=False)
    oww.model.load_state_dict(blob["model"])
    oww.optimizer.load_state_dict(blob["optimizer"])
    oww.best_models = []
    for state in blob["best_models"]:
        clone = _copy.deepcopy(oww.model)
        clone.load_state_dict(state)
        oww.best_models.append(clone)
    oww.best_model_scores = blob["best_model_scores"]
    oww.history = collections.defaultdict(list, {k: list(v) for k, v in blob["history"].items()})


def escalate(cfg: dict, achieved: float | None) -> bool:
    """Whether to raise the negative weight before the next sequence.

    `escalation_factor` of 1.0 turns this off. That is the interesting setting
    when false accepts are already near target: the first full run escalated
    1500 -> 3000 -> 6000 and paid 0.018 of recall for false accepts it did not
    need to buy.
    """
    return (cfg["escalation_factor"] != 1.0
            and achieved is not None
            and achieved > cfg["target_fp_per_hour"])


def run_sequences(cfg: dict, work: Path, ckpt: Path | None, oww, X_train, X_val, X_val_fp,
                  val_set_hrs: float, force: bool = False) -> dict:
    import numpy as np

    schedule = sequence_schedule(cfg)
    weight = cfg["max_negative_weight"]
    summary = []

    if force:
        # Sequence checkpoints outlive the stage marker, so forcing the stage
        # without clearing them would restore straight back to the end.
        for seq in schedule:
            (work / f"{seq['name']}.pt").unlink(missing_ok=True)

    for i, seq in enumerate(schedule):
        path = work / f"{seq['name']}.pt"
        if path.is_file():
            load_sequence(oww, path)
            achieved = oww.history.get("val_fp_per_hr", [None])[-1]
            print(f"[train] {seq['name']} restored from checkpoint "
                  f"(val_fp_per_hr={achieved})", flush=True)
            summary.append({"sequence": seq["name"], "restored": True,
                            "val_fp_per_hr": None if achieved is None else float(achieved)})
            if i + 1 < len(schedule) and escalate(cfg, achieved):
                weight *= cfg["escalation_factor"]
            continue

        steps = int(seq["steps"])
        # int64 throughout. Upstream uses int64 for sequence 1 but int16 for
        # sequences 2 and 3, which is safe only while `refine_steps` stays under
        # 32767: above that the validation points wrap to negative indices and
        # `step_ndx in val_steps` never matches, so the sequence collects no
        # checkpoints and reports no metrics. Clipped to max_steps-1 as well,
        # because the loop breaks there and upstream's final point is unreachable.
        val_steps = np.unique(np.clip(
            np.linspace(seq["val_from"], steps, 20).astype(np.int64), 1, steps - 1))
        weights = np.linspace(1, weight, steps).tolist()

        print(f"\n[train] {seq['name']}: {steps} steps, lr={seq['lr']:.2e}, "
              f"max_negative_weight={weight}, {len(val_steps)} validation points", flush=True)
        started = time.time()
        oww.train_model(
            X=X_train, X_val=X_val, false_positive_val_data=X_val_fp,
            max_steps=steps, warmup_steps=steps // 5, hold_steps=steps // 3,
            negative_weight_schedule=weights, val_steps=val_steps,
            lr=seq["lr"], val_set_hrs=val_set_hrs)
        elapsed = round(time.time() - started)

        achieved = float(oww.history["val_fp_per_hr"][-1]) if oww.history.get("val_fp_per_hr") else None
        recall = float(oww.history["val_recall"][-1]) if oww.history.get("val_recall") else None
        print(f"[train] {seq['name']} done in {elapsed}s: "
              f"val_fp_per_hr={achieved} val_recall={recall} "
              f"checkpoints={len(oww.best_models)}", flush=True)

        save_sequence(oww, work, seq["name"], oww.history)
        if ckpt:
            sync_checkpoint(work, ckpt, "save")

        summary.append({"sequence": seq["name"], "restored": False, "steps": steps,
                        "lr": seq["lr"], "max_negative_weight": weight,
                        "elapsed_secs": elapsed, "val_fp_per_hr": achieved,
                        "val_recall": recall, "checkpoints": len(oww.best_models)})

        # The escalation auto_train means to make, gated on a real measurement.
        if i + 1 < len(schedule) and escalate(cfg, achieved):
            weight *= cfg["escalation_factor"]
            print(f"[train] false positives above target; raising negative weight to {weight}",
                  flush=True)

    return {"sequences": summary, "n_checkpoints": len(oww.best_models)}


def select_models(oww, cfg: dict) -> tuple[dict, dict]:
    """Produce the three candidates the evaluate stage scores against each other.

    `final` is what upstream ships when its averaging finds nothing, which is
    almost always: on the first full run 0 of 55 checkpoints cleared all three
    percentile gates, so the model published was the last checkpoint of the last
    sequence -- the one trained under the heaviest negative weight, and so the
    worst of the 55 for recall.

    The `recall-gated` rule fixes that by not asking one checkpoint to be top
    decile on three axes that move against each other. It gates on false
    accepts, which is a requirement, then ranks on recall, which is the thing
    being maximised.

    Returning all three rather than picking here is deliberate. Averaging is an
    assumption, not a fact; the evaluate stage costs seconds, so it measures.
    """
    import copy as _copy

    import numpy as np

    candidates = {"final": _copy.deepcopy(oww.model)}
    scores = oww.best_model_scores
    detail = {"rule": cfg["merge_rule"], "candidates": len(oww.best_models)}

    if not oww.best_models or not oww.history.get("val_accuracy"):
        detail["reason"] = "no checkpoints collected"
        return candidates, detail

    if cfg["merge_rule"] == "upstream":
        history = oww.history
        gates = {"accuracy": float(np.percentile(history["val_accuracy"], 90)),
                 "recall": float(np.percentile(history["val_recall"], 90)),
                 "fp_per_hr": float(np.percentile(history.get("val_fp_per_hr", [0]), 10))}
        keep = [i for i, sc in enumerate(scores)
                if sc["val_accuracy"] >= gates["accuracy"]
                and sc["val_recall"] >= gates["recall"]
                and sc.get("val_fp_per_hr", 0) <= gates["fp_per_hr"]]
        detail["gates"] = gates
    else:
        target = cfg["target_fp_per_hour"]
        passing = [i for i, sc in enumerate(scores) if sc.get("val_fp_per_hr", 0) <= target]
        if not passing:
            # Nothing met the target. Fall back to the best decile by false
            # accepts rather than to the last model, and say so.
            k = max(1, len(scores) // 10)
            passing = sorted(range(len(scores)),
                             key=lambda i: scores[i].get("val_fp_per_hr", 0))[:k]
            detail["reason"] = f"no checkpoint met {target} FA/h; used the best {k} by FA"
        detail["passing_fa_gate"] = len(passing)
        keep = sorted(passing, key=lambda i: scores[i]["val_recall"],
                      reverse=True)[: cfg["merge_top_k"]]
        best = keep[0]
        candidates["best_single"] = _copy.deepcopy(oww.best_models[best])
        detail["best_single"] = {k: float(v) for k, v in scores[best].items()}

    detail["merged"] = len(keep)
    if keep:
        candidates["merged"] = oww.average_models(models=[oww.best_models[i] for i in keep])
        detail["merged_steps"] = [int(scores[i]["training_step_ndx"]) for i in keep]
    else:
        detail.setdefault("reason", "no checkpoint cleared the gates")

    return candidates, detail


def stage_train(cfg: dict, work: Path, ckpt: Path | None, paths: dict, state: dict) -> dict:
    import torch
    from openwakeword.train import Model as TrainModel

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, X_val, fp_sets, val_set_hrs, shapes = build_loaders(cfg, paths, device)
    print(f"[train] false-positive validation set: {shapes['fp_windows']} windows, "
          f"{val_set_hrs:.2f} h (auto_train would have used 11.3)", flush=True)

    # Checkpoint gating during training uses upstream's set only. It is 185 MB
    # and lives on the device, so a validation pass moves nothing; the ACAV
    # holdout is ~860 MB streamed from a lazy mount and would be read once per
    # validation point, roughly 50 GB over a run, to inform a decision that the
    # final evaluation re-makes properly against both sets.
    X_val_fp = fp_sets["upstream"].windows

    n_frames = cfg["window_frames"]
    oww = TrainModel(n_classes=1, input_shape=(n_frames, 96), model_type=cfg["model_type"],
                     layer_dim=cfg["layer_size"],
                     seconds_per_example=1280 * n_frames / SR)

    detail = run_sequences(cfg, work, ckpt, oww, X_train, X_val, X_val_fp, val_set_hrs,
                           force=state.get("force_train", False))

    candidates, select_detail = select_models(oww, cfg)
    detail["selection"] = select_detail
    print(f"[train] candidates: {sorted(candidates)}; merged "
          f"{select_detail.get('merged', 0)} of {select_detail['candidates']} checkpoints"
          + (f" ({select_detail['reason']})" if "reason" in select_detail else ""), flush=True)

    torch.save({name: m.state_dict() for name, m in candidates.items()}, work / "candidates.pt")
    state["oww"] = oww
    state["candidates"] = candidates
    state["val_set_hrs"] = val_set_hrs
    detail.update({"val_set_hrs": round(val_set_hrs, 4), "shapes": shapes,
                   "device": str(device)})
    return detail


# --------------------------------------------------------------------------
# Stage 3: evaluate
# --------------------------------------------------------------------------

def score_all(model, windows) -> "object":
    import numpy as np
    import torch

    out = []
    with torch.no_grad():
        for x, _ in windows:
            out.append(model(x).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out)


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# The threshold reported alongside every curve, for continuity with earlier
# runs. It is NOT the comparison metric: see `recall_at_fa`.
COMPARE_AT = 0.5

# Runs are compared by recall at a matched false-accept budget, at each of
# these. Measured 2026-09-02: three models from the same pipeline put threshold
# 0.5 at 0.47, 0.93 and 1.31 FA/h, so "recall at threshold 0.5" compared three
# different operating points and ranked them by calibration rather than by
# quality. Recall at a matched budget is a point on the same ROC curve for
# every model, which is the thing that can be compared.
FA_BUDGETS = [0.2, 0.5, 1.0, 2.0]


def build_curve(fp_scores, pos_scores, val_set_hrs: float, grouping: int) -> list[dict]:
    """Both counters at every threshold.

    The corrected event count is what this project is gated on; upstream's
    frame-above-threshold count is carried alongside so the numbers stay
    comparable with the pilot and with published openWakeWord figures.
    """
    curve = []
    for threshold in THRESHOLDS:
        events = count_activations(fp_scores, threshold, grouping)
        frames = int((fp_scores >= threshold).sum())
        curve.append({
            "threshold": threshold,
            "recall": float((pos_scores >= threshold).mean()),
            "false_accepts_per_hour": round(events / val_set_hrs, 4),
            "frames_above_per_hour_upstream": round(frames / val_set_hrs, 4),
            "events": events, "frames": frames,
        })
    return curve


def at_threshold(curve: list[dict], threshold: float) -> dict:
    return next(row for row in curve if row["threshold"] == threshold)


def recall_at_fa(curve: list[dict], budget: float) -> float | None:
    """Best recall reachable without exceeding `budget` false accepts per hour.

    The comparison metric. Reads a point off the model's own ROC curve rather
    than off a threshold, so two models with different score distributions are
    still compared at the same operating cost. `None` means no threshold on the
    curve buys that budget.
    """
    reachable = [row["recall"] for row in curve
                 if row["false_accepts_per_hour"] <= budget]
    return max(reachable) if reachable else None


def recall_profile(curve: list[dict]) -> dict:
    return {str(b): recall_at_fa(curve, b) for b in FA_BUDGETS}


def exact_profile(fp_scores, pos_scores, val_set_hrs: float, grouping: int) -> dict:
    """Recall at each false-accept budget, read off the exact ROC.

    Two things here were wrong when this was first written, and both produced
    numbers that looked entirely plausible.

    **The event count is not monotone in the threshold, so it cannot be binary
    searched.** Lowering the threshold adds a frame; if that frame is not
    already inside a refractory window it becomes an event and moves the window
    forward, which re-partitions everything after it, so the count can fall as
    the threshold falls. A binary search returns *a* threshold meeting the
    budget rather than the lowest one: on v2-flat500 it returned 0.880
    (recall 0.518) when 0.800 met the same budget at recall 0.563. This sweeps
    instead, descending, keeping the last threshold that fits.

    **The best threshold for a given false-positive set sits just above the
    highest EXCLUDED false positive**, not at the lowest included one. Every
    threshold in between admits the same false positives while catching
    strictly more true ones. Getting this wrong cost one positive in 2,000 and,
    tellingly, made the exact reading come out *below* the grid reading -- which
    cannot happen, since every grid threshold is also a candidate here. That
    impossibility is asserted at the call site.

    The sweep stops after 200 consecutive threshold steps above the largest
    budget; the count would have to fall back by an order of magnitude to
    matter after that.
    """
    import bisect

    import numpy as np

    fp_scores = np.asarray(fp_scores).reshape(-1)
    pos_scores = np.asarray(pos_scores).reshape(-1)
    allowed = {b: b * val_set_hrs for b in FA_BUDGETS}
    cap = max(allowed.values())

    order = np.argsort(-fp_scores, kind="stable")
    ranked = fp_scores[order]
    # k = 0: admit no false positive at all, threshold just above the highest
    # one. Zero events fits every budget, so this is always a valid starting
    # point; the sweep below can only lower the threshold from here. Without it
    # a budget that allows fewer than one event -- 0.2 FA/h on a 1.3 h
    # rehearsal set -- came back None while the grid read 0.0 at 0.99.
    best = {b: float(ranked[0]) for b in FA_BUDGETS} if len(ranked) else {b: None for b in FA_BUDGETS}
    above, fails = [], 0

    for k in range(1, len(order) + 1):
        bisect.insort(above, int(order[k - 1]))
        if k < len(ranked) and ranked[k] == ranked[k - 1]:
            continue                      # the threshold has not actually moved
        events = count_activations_from_indices(above, grouping)
        threshold = float(ranked[k]) if k < len(ranked) else -1.0
        for budget in FA_BUDGETS:
            if events <= allowed[budget]:
                best[budget] = threshold  # descending, so this only ever helps
        if events > cap:
            fails += 1
            if fails > 200:
                break
        else:
            fails = 0

    return {str(b): {
        "recall": None if best[b] is None else float((pos_scores > best[b]).mean()),
        "threshold": best[b]} for b in FA_BUDGETS}


def operating_point(curve: list[dict], target: float) -> dict | None:
    """Lowest threshold meeting the false-accept target -- lowest because recall
    falls monotonically as the threshold rises."""
    meeting = [row for row in curve if row["false_accepts_per_hour"] <= target]
    return min(meeting, key=lambda r: r["threshold"]) if meeting else None


class OnnxScorer:
    """A published ONNX classifier wearing the torch-model interface the
    evaluate stage uses (`to`, `eval`, `__call__` on a feature tensor), so a
    baseline model sits in the candidate table like any checkpoint."""

    def __init__(self, path: Path):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        import numpy as np
        import torch
        arr = x.detach().cpu().numpy().astype("float32")
        out = np.concatenate([self.session.run(None, {self.input: arr[i:i + 4096]})[0].reshape(-1)
                              for i in range(0, arr.shape[0], 4096)]) if arr.shape[0] else np.zeros(0)
        return torch.from_numpy(out)


def stress_rows(onnx_path: Path, stress_dir: Path, thresholds: dict, cfg: dict, label: str) -> dict:
    """Score the held-out positives through the export at fixed SNRs.

    Mirrors scripts/stress_test.py so an in-job number and a local one mean the
    same thing: a second of the interference either side of the phrase, SNR as
    speech RMS over interference RMS, peak-normalised, scored with
    `predict_clip`, and the clip's score is its maximum frame. `clear_frac` is
    the share of positives above each budget's threshold, which is the
    per-condition recall the model would have at that operating point.
    """
    import numpy as np
    import openwakeword
    import openwakeword.utils
    import soundfile

    openwakeword.utils.download_models()
    detector = openwakeword.Model(wakeword_models=[str(onnx_path)], inference_framework="onnx")
    key = next(iter(detector.models))
    rng = np.random.default_rng(cfg["seed"])

    def rms(x) -> float:
        return float(np.sqrt(np.mean(x ** 2)) + 1e-9)

    positives = [soundfile.read(str(p), dtype="float32")[0]
                 for p in sorted((stress_dir / "positives").glob("*.wav"), key=lambda p: int(p.stem))]
    pools = {role: soundfile.read(str(stress_dir / "pools" / f"{role}.wav"), dtype="float32")[0]
             for role in STRESS_ROLES if (stress_dir / "pools" / f"{role}.wav").is_file()}
    assert positives and pools, f"stress assets incomplete under {stress_dir}"

    def score(clip) -> float:
        detector.reset()
        clip = clip / max(1.0, float(np.abs(clip).max()))
        preds = detector.predict_clip((clip * 32767).astype(np.int16), padding=1)
        return max(float(d[key]) for d in preds)

    def mixed(v, pool, snr):
        n = len(v) + 2 * SR
        sig = np.zeros(n, dtype="float32")
        sig[SR:SR + len(v)] = v
        if pool is None:
            return sig
        assert len(pool) > n, "stress pool shorter than a padded positive"
        start = int(rng.integers(0, len(pool) - n))
        return sig + pool[start:start + n] * (rms(v) / 10 ** (snr / 20))

    def summarise(scores) -> dict:
        arr = np.asarray(scores)
        return {"min": round(float(arr.min()), 4), "median": round(float(np.median(arr)), 4),
                "clear_frac": {b: None if t is None else round(float((arr > t).mean()), 3)
                               for b, t in thresholds.items()}}

    rows = {"clean": summarise([score(mixed(v, None, None)) for v in positives])}
    for role, pool in pools.items():
        for snr in STRESS_SNRS:
            rows[f"{role}@{snr}"] = summarise([score(mixed(v, pool, snr)) for v in positives])
    gate = str(cfg["select_fa_gate"])
    print(f"[stress] {label}: clear fraction at <= {gate} FA/h "
          f"(threshold {thresholds.get(gate)}), {len(positives)} held-out positives", flush=True)
    for role in pools:
        line = "  ".join(f"{snr:>2} dB {rows[f'{role}@{snr}']['clear_frac'][gate]}" for snr in STRESS_SNRS)
        print(f"[stress] {label} {role:<7} {line}", flush=True)
    print(f"[stress] {label} clean   {rows['clean']['clear_frac'][gate]}", flush=True)
    return {"thresholds": thresholds, "n_positives": len(positives), "rows": rows}


def stage_evaluate(cfg: dict, work: Path, paths: dict, state: dict) -> dict:
    import numpy as np
    import torch
    from openwakeword.train import Model as TrainModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_frames = cfg["window_frames"]

    oww = state.get("oww")
    if oww is None:
        # Reached after a restart that skipped the train stage.
        oww = TrainModel(n_classes=1, input_shape=(n_frames, 96), model_type=cfg["model_type"],
                         layer_dim=cfg["layer_size"], seconds_per_example=1280 * n_frames / SR)

    candidates = state.get("candidates")
    if candidates is None:
        blob = torch.load(work / "candidates.pt", map_location=device, weights_only=False)
        candidates = {}
        for name, sd in blob.items():
            model = copy.deepcopy(oww.model)
            model.load_state_dict(sd)
            candidates[name] = model

    _, X_val, fp_sets, val_set_hrs, shapes = build_loaders(cfg, paths, device)
    pos = np.load(paths["positive_val"])
    pos_t = torch.from_numpy(pos.astype("float32")).to(device)
    cond_t = {cond: torch.from_numpy(np.load(paths[f"positive_val_{cond}"]).astype("float32")).to(device)
              for cond in VAL_CONDITIONS if f"positive_val_{cond}" in paths}

    # The previously promoted model, scored on exactly the same sets in the
    # same job. Numbers from an older metrics.json were measured on older
    # validation data and cannot be compared with these.
    baseline_onnx = None
    if cfg.get("baseline_model"):
        from huggingface_hub import hf_hub_download
        baseline_onnx = Path(hf_hub_download(cfg["baseline_model"], f"{cfg['model_name']}.onnx"))
        candidates = dict(candidates, baseline=OnnxScorer(baseline_onnx))
        print(f"[evaluate] baseline: {cfg['baseline_model']} ({baseline_onnx.stat().st_size // 1024} KB)",
              flush=True)

    # Every candidate against every false-positive set. Scoring is seconds and
    # assuming that averaging helped, or that one set's resolution is enough,
    # is how the last three wrong numbers happened.
    curves: dict = {}
    exact: dict = {}
    conditions: dict = {}
    for set_name, evalset in fp_sets.items():
        curves[set_name], exact[set_name], conditions[set_name] = {}, {}, {}
        for name in sorted(candidates):
            model = candidates[name].to(device).eval()
            fp_scores = score_all(model, evalset.windows)
            with torch.no_grad():
                pos_scores = model(pos_t).detach().cpu().numpy().reshape(-1)
            curve = build_curve(fp_scores, pos_scores, evalset.hours, evalset.grouping)
            profile = exact_profile(fp_scores, pos_scores, evalset.hours, evalset.grouping)
            # Every grid threshold is also a candidate for the exact sweep, so
            # the exact reading can never be worse than the grid's. If it is,
            # the sweep is broken -- which is how two earlier bugs were caught.
            for budget in FA_BUDGETS:
                grid = recall_at_fa(curve, budget)
                got = profile[str(budget)]["recall"]
                assert grid is None or (got is not None and got >= grid - 1e-9), (
                    f"{set_name}/{name}: exact recall {got} at {budget} FA/h is "
                    f"below the grid's {grid}; the exact sweep is wrong.")
            curves[set_name][name], exact[set_name][name] = curve, profile
            summary = " ".join(
                f"{b}:{'-' if profile[b]['recall'] is None else format(profile[b]['recall'], '.3f')}"
                for b in profile)
            print(f"[evaluate] {set_name:<13} {name:<12} "
                  f"({evalset.hours:.1f} h, group {evalset.grouping}) "
                  f"recall at FA budget {summary}", flush=True)
            # The budget thresholds depend on the false-positive set alone, so
            # each condition is one comparison against them, not a new sweep.
            per_cond = {}
            for cond, arr in cond_t.items():
                with torch.no_grad():
                    cs = model(arr).detach().cpu().numpy().reshape(-1)
                per_cond[cond] = {b: {
                    "recall": None if v["threshold"] is None else float((cs > v["threshold"]).mean()),
                    "threshold": v["threshold"]} for b, v in profile.items()}
            conditions[set_name][name] = per_cond
            if per_cond:
                gate = str(cfg["select_fa_gate"])
                line = " ".join(
                    f"{c}:{'-' if per_cond[c][gate]['recall'] is None else format(per_cond[c][gate]['recall'], '.3f')}"
                    for c in per_cond)
                print(f"[evaluate] {set_name:<13} {name:<12} at <= {gate} FA/h by condition {line}",
                      flush=True)

    primary = cfg["select_on"] if cfg["select_on"] in exact else "upstream"
    # The baseline is a reference row, never the published model.
    trained = {n: p for n, p in exact[primary].items() if n != "baseline"}
    select_cond = cfg.get("select_condition") or "mixed"
    if select_cond != "mixed" and all(select_cond in conditions[primary][n] for n in trained):
        ranking = {n: conditions[primary][n][select_cond] for n in trained}
        print(f"[evaluate] selecting on the {select_cond!r} condition", flush=True)
    else:
        ranking = trained
    selected = choose_candidate_exact(ranking, {n: curves[primary][n] for n in trained}, cfg)
    curve = curves[primary][selected]
    best = candidates[selected].to(device).eval()
    print(f"[evaluate] selected {selected} on the {primary} set "
          f"({fp_sets[primary].hours:.1f} h)", flush=True)
    if "baseline" in exact[primary] and conditions[primary].get(selected):
        gate = str(cfg["select_fa_gate"])
        for cond in conditions[primary][selected]:
            b = conditions[primary]["baseline"][cond][gate]["recall"]
            m = conditions[primary][selected][cond][gate]["recall"]
            print(f"[evaluate] {cond:<7} at <= {gate} FA/h: baseline "
                  f"{'-' if b is None else format(b, '.3f')} -> {selected} "
                  f"{'-' if m is None else format(m, '.3f')}", flush=True)

    onnx_path = work / f"{cfg['model_name']}.onnx"
    export_detail = export_onnx(best, onnx_path, n_frames, cfg["model_name"])
    parity = onnx_parity(onnx_path, best, pos, device)
    end_to_end = raw_pcm_check(onnx_path, cfg)

    stress: dict = {}
    stress_dir = Path(paths.get("stress_dir", "/nonexistent"))
    if (stress_dir / "manifest.json").is_file():
        thresholds = {b: v["threshold"] for b, v in exact[primary][selected].items()}
        stress[selected] = stress_rows(onnx_path, stress_dir, thresholds, cfg, selected)
        if baseline_onnx is not None:
            thresholds = {b: v["threshold"] for b, v in exact[primary]["baseline"].items()}
            stress["baseline"] = stress_rows(baseline_onnx, stress_dir, thresholds, cfg, "baseline")
    else:
        print("[evaluate] no stress/ assets in the feature set; skipping the SNR grid", flush=True)

    return {
        "primary_set": primary,
        "sets": {name: dict(st.detail,
                            curves=curves[name],
                            recall_at_fa={c: {b: v["recall"] for b, v in prof.items()}
                                          for c, prof in exact[name].items()},
                            recall_at_fa_exact=exact[name],
                            conditions=conditions[name])
                 for name, st in fp_sets.items()},
        "val_conditions": sorted(cond_t),
        "recall_at_fa_by_condition": conditions[primary].get(selected, {}),
        "baseline": ({"model": cfg["baseline_model"],
                      "recall_at_fa_exact": exact[primary]["baseline"],
                      "recall_at_fa_by_condition": conditions[primary]["baseline"]}
                     if "baseline" in exact[primary] else None),
        "stress": stress,
        "select_condition": select_cond,
        "curve": curve,
        "curves": curves[primary],
        "selected": selected,
        "compare_at": COMPARE_AT,
        "compare_row": at_threshold(curve, COMPARE_AT),
        "recall_at_fa": {b: v["recall"] for b, v in exact[primary][selected].items()},
        "recall_at_fa_exact": exact[primary][selected],
        "recall_at_fa_grid": recall_profile(curve),
        "recall_at_fa_by_candidate": {n: {b: v["recall"] for b, v in prof.items()}
                                      for n, prof in exact[primary].items()},
        "operating_point": operating_point(curve, cfg["target_fp_per_hour"]),
        "val_set_hrs": round(fp_sets[primary].hours, 4),
        "positive_val_n": int(pos.shape[0]),
        "grouping_window": fp_sets[primary].grouping,
        "onnx_parity_max_abs_diff": parity,
        "onnx_export": export_detail,
        "end_to_end": end_to_end,
        "onnx_sha256": sha256_file(onnx_path),
    }


def choose_candidate_exact(exact: dict, curves: dict, cfg: dict) -> str:
    """Best recall at the selection budget, measured on the exact ROC."""
    key = str(cfg["select_fa_gate"])
    if key not in next(iter(exact.values())):
        return choose_candidate(curves, cfg)
    scored = {name: prof[key]["recall"] for name, prof in exact.items()}
    reachable = {n: r for n, r in scored.items() if r is not None}
    for name in sorted(scored):
        print(f"[evaluate] {name:<12} exact recall at <= {key} FA/h: {scored[name]}", flush=True)
    if not reachable:
        print(f"[evaluate] no candidate reaches {key} FA/h; falling back to the grid",
              flush=True)
        return choose_candidate(curves, cfg)
    return max(reachable, key=lambda n: reachable[n])


def choose_candidate(curves: dict, cfg: dict) -> str:
    """Best recall at a matched false-accept budget.

    Not recall at a fixed threshold. The three candidates come out of the same
    run but not with the same calibration -- in E0a, `final` scored 0.570 recall
    at threshold 0.5 against `merged`'s 0.514, but it was spending 0.93 FA/h to
    do it against merged's 0.47. Comparing at a shared threshold picks whichever
    candidate scores highest overall, which is a statement about its score
    distribution and not about its quality.
    """
    gate = cfg["select_fa_gate"]
    scored = {name: recall_at_fa(curve, gate) for name, curve in curves.items()}
    reachable = {n: r for n, r in scored.items() if r is not None}
    for name in sorted(scored):
        print(f"[evaluate] {name:<12} recall at <= {gate} FA/h: {scored[name]}", flush=True)
    if not reachable:
        print(f"[evaluate] no candidate reaches {gate} FA/h at any threshold; "
              f"choosing the lowest false-accept rate instead", flush=True)
        return min(curves, key=lambda n: min(
            row["false_accepts_per_hour"] for row in curves[n]))
    return max(reachable, key=lambda n: reachable[n])


def export_onnx(model, onnx_path: Path, n_frames: int, output_name: str) -> dict:
    """Export with a dynamic batch axis, at an opset that can hold the model.

    Not `Model.export_model`, which gets both wrong under torch 2.13:

    * It traces on `torch.rand(input_shape)[None,]` and asks for no dynamic
      axes, so the dynamo exporter freezes the batch dimension at 1. Anything
      that scores more than one window per call -- including upstream's own
      `predict_on_features` -- then fails with "Got: 64 Expected: 1".
    * It asks for `opset_version=13`, which cannot represent
      `LayerNormalization` (introduced at 17). onnxscript's converter logs the
      failure, falls back to the ONNX C API, logs *that* failure, and returns
      the unconverted model. The export succeeds and the opset claim is fiction.

    17 is therefore the floor, not a preference. The assertion below is the
    point of the function: a static batch axis has to fail here rather than in
    whatever consumes the model later.
    """
    import onnx
    import onnxruntime
    import torch

    traced = copy.deepcopy(model).to("cpu").eval()
    example = torch.rand(1, n_frames, 96)
    # external_data=False is load-bearing, not a preference. The dynamo
    # exporter defaults to writing weights into a sibling `<name>.onnx.data`,
    # which leaves a 3 KB graph that loads only while that sibling is beside
    # it. Every model published before 2026-09-02 has this defect: the parity
    # and end-to-end checks passed in-job, where the sibling existed, and the
    # published repo held a file no runtime can open.
    kwargs = dict(input_names=["x"], output_names=[output_name], opset_version=17,
                  external_data=False)

    used = "dynamic_axes"
    torch.onnx.export(traced, (example,), str(onnx_path),
                      dynamic_axes={"x": {0: "batch"}, output_name: {0: "batch"}}, **kwargs)
    if isinstance(onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]).get_inputs()[0].shape[0], int):
        # The dynamo exporter takes `dynamic_shapes`; `dynamic_axes` is
        # translated for it on a best-effort basis and can be dropped.
        used = "dynamic_shapes"
        torch.onnx.export(traced, (example,), str(onnx_path),
                          dynamic_shapes={"x": {0: torch.export.Dim("batch")}}, **kwargs)

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    assert not isinstance(shape[0], int), (
        f"exported input shape is {shape}; the batch axis is frozen and the "
        f"model can only score one window per call.")
    assert list(shape[1:]) == [n_frames, 96], f"exported input shape is {shape}"

    # Self-containment, asserted two ways: no initializer may point at an
    # external file, and the file must be big enough to actually hold the
    # weights. A graph-only export is a few KB; this model's parameters alone
    # are ~200 KB.
    model_proto = onnx.load(str(onnx_path))
    external = [init.name for init in model_proto.graph.initializer
                if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL]
    assert not external, (
        f"{onnx_path.name} keeps {len(external)} initializer(s) in an external "
        f"file ({external[:3]}); the published model would not load.")
    size = onnx_path.stat().st_size
    assert size > 100_000, (
        f"{onnx_path.name} is {size} bytes, too small to contain the weights.")
    for stray in onnx_path.parent.glob(onnx_path.name + ".data*"):
        stray.unlink()

    opsets = {d.domain or "ai.onnx": d.version for d in model_proto.opset_import}
    print(f"[evaluate] exported {onnx_path.name}: input {shape}, opsets {opsets}, "
          f"{size / 1024:.0f} KB self-contained (via {used})", flush=True)
    return {"input_shape": [str(d) for d in shape], "opsets": opsets,
            "dynamic_via": used, "bytes": size}


def onnx_parity(onnx_path: Path, torch_model, features, device) -> float:
    """The export is only useful if it computes what the torch model computes."""
    import numpy as np
    import onnxruntime
    import torch

    sample = features[:512].astype("float32")
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    onnx_out = np.concatenate([
        session.run(None, {name: sample[i:i + 64]})[0].reshape(-1)
        for i in range(0, sample.shape[0], 64)])
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(sample).to(device)).cpu().numpy().reshape(-1)

    diff = float(np.abs(onnx_out - torch_out).max())
    assert diff < 1e-4, (
        f"ONNX and torch disagree by {diff}; the exported model is not the "
        f"model that was trained and evaluated.")
    print(f"[evaluate] ONNX/torch parity: max abs diff {diff:.2e}", flush=True)
    return diff


def raw_pcm_check(onnx_path: Path, cfg: dict) -> dict:
    """Load the export the way a deployment does and feed it real PCM.

    This is a plumbing check, not a quality one: it proves the melspectrogram
    and embedding models in front of the classifier accept the export and that
    scores come out finite and in range. Wake-word recall is measured on
    features above.
    """
    import numpy as np
    import openwakeword
    import openwakeword.utils

    openwakeword.utils.download_models()
    detector = openwakeword.Model(wakeword_models=[str(onnx_path)],
                                  inference_framework="onnx")

    rng = np.random.default_rng(cfg["seed"])
    pcm = (rng.normal(0, 0.02, SR * 4) * 32767).astype(np.int16)
    scores = []
    for i in range(0, len(pcm) - 1280, 1280):
        prediction = detector.predict(pcm[i:i + 1280])
        scores.extend(float(v) for v in prediction.values())

    assert scores, "openwakeword.Model produced no predictions"
    assert all(np.isfinite(scores)), "openwakeword.Model produced non-finite scores"
    assert 0.0 <= min(scores) and max(scores) <= 1.0, f"scores out of range: {min(scores)}..{max(scores)}"
    print(f"[evaluate] end-to-end on 4 s of noise: {len(scores)} predictions, "
          f"max score {max(scores):.4f}", flush=True)
    return {"n_predictions": len(scores), "max_score": max(scores),
            "model_keys": sorted(detector.models.keys())}


# --------------------------------------------------------------------------
# Stage 4: publish
# --------------------------------------------------------------------------

MODEL_CARD = """---
license: apache-2.0
library_name: openwakeword
tags:
  - wake-word-detection
  - keyword-spotting
  - onnx
---

# Hey Sonny

An openWakeWord classifier for the phrase **"hey sonny"**, trained on synthetic
Piper LibriTTS-R speech over the frozen openWakeWord melspectrogram and
embedding models. Input is `(1, {frames}, 96)` features, which is
{seconds:.2f} s of 16 kHz audio.

## Operating point

{operating}

False accepts are counted as **activation events** on upstream's
`validation_set_features.npy` ({hours:.2f} h measured from the array shape, not
the 11.3 h `openwakeword.train.auto_train` hardcodes). `metrics.json` also
reports upstream's frame-above-threshold count for every threshold, because
`openwakeword.metrics.get_false_positives` returns that rather than events and
published openWakeWord figures use it.

Of the three candidates scored -- the weight-averaged checkpoints, the single
best checkpoint, and the final model -- **{selected}** was published. All three
curves are in `metrics.json`.

## Provenance

- Attempt: `{attempt}`
- Features: `{features_repo}`
- openWakeWord: `368c03716d1e92591906a84949bc477f3a834455`

Full configuration, seeds, input revisions and environment are in
`manifest.json`; the threshold sweep is in `metrics.json`.
"""


def stage_publish(cfg: dict, work: Path, manifest: dict) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    repo_id = cfg["repo_id"]
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    staging = work / "publish"
    staging.mkdir(parents=True, exist_ok=True)

    onnx_name = f"{cfg['model_name']}.onnx"
    shutil.copy2(work / onnx_name, staging / onnx_name)

    evaluation = manifest["stages"]["evaluate"]
    (staging / "metrics.json").write_text(json.dumps({
        "run_name": manifest.get("run_name"),
        "config_sha256": manifest["config_sha256"],
        "curve": evaluation["curve"],
        "curves": evaluation["curves"],
        "selected": evaluation["selected"],
        "compare_at": evaluation["compare_at"],
        "compare_row": evaluation["compare_row"],
        "recall_at_fa": evaluation["recall_at_fa"],
        "recall_at_fa_exact": evaluation["recall_at_fa_exact"],
        "recall_at_fa_grid": evaluation["recall_at_fa_grid"],
        "recall_at_fa_by_candidate": evaluation["recall_at_fa_by_candidate"],
        "operating_point": evaluation["operating_point"],
        "val_set_hrs": evaluation["val_set_hrs"],
        "grouping_window": evaluation["grouping_window"],
        "primary_set": evaluation["primary_set"],
        "sets": evaluation["sets"],
        "positive_val_n": evaluation["positive_val_n"],
        "onnx_parity_max_abs_diff": evaluation["onnx_parity_max_abs_diff"],
        "selection": manifest["stages"]["train"].get("selection"),
        "metrics_version": 3,
        "val_conditions": evaluation.get("val_conditions", []),
        "recall_at_fa_by_condition": evaluation.get("recall_at_fa_by_condition", {}),
        "baseline": evaluation.get("baseline"),
        "stress": evaluation.get("stress", {}),
        "select_condition": evaluation.get("select_condition", "mixed"),
    }, indent=2, sort_keys=True, default=str))

    operating = evaluation["operating_point"]
    if operating:
        text = (f"Threshold **{operating['threshold']}**: recall "
                f"{operating['recall']:.3f} at "
                f"{operating['false_accepts_per_hour']} false accepts per hour.")
    else:
        text = (f"No threshold reached the target of {cfg['target_fp_per_hour']} "
                f"false accepts per hour. See `metrics.json` for the full sweep.")

    (staging / "README.md").write_text(MODEL_CARD.format(
        frames=cfg["window_frames"], seconds=cfg["window_frames"] * SECONDS_PER_FRAME + 0.72,
        operating=text, hours=evaluation["val_set_hrs"],
        attempt=manifest["attempt_id"], features_repo=cfg["features_repo"],
        selected=evaluation["selected"]))

    # Experiments land under runs/<name>/ so that one repo can hold a series
    # without each run overwriting the last one's metrics.
    run_name = manifest.get("run_name")
    prefix = f"runs/{run_name}" if run_name else ""
    checksums = {f"{prefix}/{p.name}" if prefix else p.name: sha256_file(p)
                 for p in sorted(staging.iterdir()) if p.is_file()}
    manifest["checksums"] = checksums
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))

    # One commit, so a reader never sees a model without its metrics.
    commit = api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="model",
                               path_in_repo=prefix,
                               commit_message=f"Job B {run_name or manifest['attempt_id']}")
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)

    # A successful push is not proof the artifact persisted: read it back.
    mismatched = []
    downloaded = {}
    for filename, digest in checksums.items():
        local = hf_hub_download(repo_id, filename, repo_type="model",
                                revision=revision, force_download=True)
        downloaded[filename] = Path(local)
        if sha256_file(Path(local)) != digest:
            mismatched.append(filename)
    assert not mismatched, f"read-back checksums differ for {mismatched}"

    # Matching checksums prove the bytes arrived. They do not prove the model
    # works: an ONNX file whose weights live in an unpublished sibling passes
    # every checksum and loads nowhere. So run the download.
    published_onnx = next(p for name, p in downloaded.items() if name.endswith(".onnx"))
    end_to_end = raw_pcm_check(published_onnx, cfg)
    print(f"[publish] downloaded model runs: {end_to_end['n_predictions']} predictions",
          flush=True)

    # The manifest cannot carry its own hash, so it is verified by presence
    # rather than by checksum. It is still a completion criterion: without it
    # the commit records no JOB_ID, no input revisions and no seeds.
    published = {f.rfilename for f in api.repo_info(repo_id, repo_type="model",
                                                    revision=revision).siblings}
    manifest_path = f"{prefix}/manifest.json" if prefix else "manifest.json"
    assert manifest_path in published, f"commit {revision} has no {manifest_path}"

    return {"repo_id": repo_id, "revision": revision, "path_in_repo": prefix or ".",
            "verified": sorted(checksums), "downloaded_model_runs": end_to_end,
            "published": sorted(p for p in published if p != ".gitattributes")}


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="/work", help=(
        "Working tree, on local ephemeral disk. Do NOT point this at a bucket "
        "mount; see jobs-spec.md."))
    ap.add_argument("--checkpoint-dir", default="/out/jobb", help=(
        "Durable directory for stage markers and per-sequence checkpoints, so a "
        "crashed run resumes at a sequence boundary rather than at step zero."))
    ap.add_argument("--features-dir", default="/features", help="Job A's output, mounted")
    ap.add_argument("--upstream-dir", default="/upstream", help="davidscripka/openwakeword_features, mounted")
    ap.add_argument("--acav-file", default="openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    ap.add_argument("--fp-file", default="validation_set_features.npy")
    ap.add_argument("--repo-id", default="iamtokarev/hey-sonny")
    ap.add_argument("--features-repo", default="iamtokarev/hey-sonny-features")
    ap.add_argument("--model-name", default="hey_sonny")
    ap.add_argument("--window-frames", type=int, default=16)
    ap.add_argument("--model-type", default="dnn")
    ap.add_argument("--layer-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--refine-steps", type=int, default=5_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-negative-weight", type=float, default=1500)
    ap.add_argument("--escalation-factor", type=float, default=2.0, help=(
        "Multiply the negative weight by this before each later sequence whose "
        "predecessor missed --target-fp-per-hour. 1.0 disables escalation, "
        "which is what you want when false accepts are already near target."))
    ap.add_argument("--merge-rule", choices=("recall-gated", "upstream"), default="recall-gated",
                    help="upstream reproduces auto_train's three-percentile intersection")
    ap.add_argument("--merge-top-k", type=int, default=10,
                    help="recall-gated: checkpoints to average, ranked by recall")
    ap.add_argument("--select-fa-gate", type=float, default=1.0, choices=FA_BUDGETS, help=(
        "The false-accept budget the three candidates are compared at. Each is "
        "scored by the best recall it reaches without exceeding this rate, read "
        "off its own exact ROC -- not at a shared threshold, which would rank "
        "them by score calibration rather than by quality."))
    ap.add_argument("--run-name", help=(
        "Publish under runs/<name>/ instead of the repo root, so one repo can "
        "hold a series of experiments."))
    ap.add_argument("--target-fp-per-hour", type=float, default=0.2)
    ap.add_argument("--min-recall", type=float, default=0.20)
    ap.add_argument("--acav-batch-rows", type=int, default=1024,
                    help="Flat ACAV rows per batch; becomes (rows-16)//16 windows")
    ap.add_argument("--positive-per-batch", type=int, default=50)
    ap.add_argument("--adversarial-per-batch", type=int, default=50)
    ap.add_argument("--acav-holdout-hours", type=float, default=100.0, help=(
        "Hours withheld from the end of ACAV100M as a second, larger "
        "false-positive set. Upstream's file allows only two events at the "
        "0.2 FA/h target, which is not enough resolution to separate two "
        "models; 0 disables."))
    ap.add_argument("--select-on", default="upstream",
                    choices=("acav_holdout", "upstream"), help=(
        "Which false-positive set decides between the three candidates. "
        "`upstream` by default despite its worse resolution: it is a different "
        "corpus from the training negatives, and the ACAV holdout -- unseen "
        "clips, but ACAV -- reads 0.77 where upstream reads 0.46 on the same "
        "model at the same nominal budget. Selecting on the holdout would "
        "optimise toward the training corpus and report the flattering number."))
    ap.add_argument("--select-condition", default="mixed",
                    choices=("mixed",) + VAL_CONDITIONS, help=(
        "Which validation positives rank the candidates: the mixed set, or one "
        "of Job A's per-condition sets when they exist."))
    ap.add_argument("--baseline-model", default=None, help=(
        "A Model repo whose <model-name>.onnx is scored on the same validation "
        "sets and stress grid as a reference row. Never published."))
    ap.add_argument("--fp-batch", type=int, default=16_384,
                    help="Windows per false-positive validation chunk")
    ap.add_argument("--limit-fp-frames", type=int, default=0,
                    help="Truncate the false-positive set. Rehearsal only: it changes the metric")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="Re-run these stages even if their COMPLETE marker exists")
    ap.add_argument("--stop-after", choices=STAGES, help="Stop once this stage finishes")
    args = ap.parse_args()

    cfg = build_config(args)
    cfg["fp_batch"] = args.fp_batch
    cfg["limit_fp_frames"] = args.limit_fp_frames

    patch_torchaudio()

    features_dir = Path(args.features_dir)
    upstream_dir = Path(args.upstream_dir)
    paths = {name: features_dir / f"{name}_features.npy" for name in FEATURE_FILES}
    for cond in VAL_CONDITIONS:
        candidate = features_dir / f"positive_val_{cond}_features.npy"
        if candidate.is_file():
            paths[f"positive_val_{cond}"] = candidate
    paths["stress_dir"] = features_dir / "stress"

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    paths = localise_features(paths, work / "features")
    paths["acav"] = upstream_dir / args.acav_file
    paths["false_positive"] = upstream_dir / args.fp_file
    ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt:
        restored = sync_checkpoint(work, ckpt, "restore")
        print(f"restored from {ckpt}: {restored or 'nothing'}", flush=True)

    manifest = {
        "attempt_id": os.environ.get("JOB_ID", "local"),
        # Deliberately outside `cfg`, and so outside config_sha256: a run name
        # is a label. Two runs with the same configuration must hash alike.
        "run_name": args.run_name,
        "config": cfg,
        "config_sha256": config_hash(cfg),
        "environment": {k: os.environ.get(k, "unset")
                        for k in ("ACCELERATOR", "CPU_CORES", "MEMORY")},
        "openwakeword_commit": "368c03716d1e92591906a84949bc477f3a834455",
        "inputs": {k: str(v) for k, v in paths.items()},
        "stages": {},
    }
    print(json.dumps({"config_sha256": manifest["config_sha256"], **cfg}, indent=2), flush=True)

    state: dict = {}
    for stage in STAGES:
        if stage_done(work, stage) and stage not in args.force:
            manifest["stages"][stage] = json.loads(marker(work, stage).read_text())
            print(f"[{stage}] already complete, skipping", flush=True)
        else:
            started = time.time()
            print(f"\n=== {stage} ===", flush=True)
            if stage == "inputs":
                detail = stage_inputs(cfg, work, paths)
            elif stage == "train":
                state["force_train"] = "train" in args.force
                detail = stage_train(cfg, work, ckpt, paths, state)
            elif stage == "evaluate":
                detail = stage_evaluate(cfg, work, paths, state)
            else:
                detail = stage_publish(cfg, work, manifest)
            detail["elapsed_secs"] = round(time.time() - started)
            complete(work, stage, **detail)
            manifest["stages"][stage] = json.loads(marker(work, stage).read_text())
            if ckpt:
                saved = sync_checkpoint(work, ckpt, "save")
                print(f"checkpointed to {ckpt}: {saved or 'nothing'}", flush=True)

        if args.stop_after == stage:
            print(f"stopping after {stage} as requested", flush=True)
            break

    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    if ckpt:
        sync_checkpoint(work, ckpt, "save")
    print("\n" + json.dumps(manifest["stages"], indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
