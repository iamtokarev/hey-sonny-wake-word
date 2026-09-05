# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "openwakeword @ git+https://github.com/dscripka/openWakeWord.git@368c03716d1e92591906a84949bc477f3a834455",
#   # piper_sample_generator.__main__ imports `piper` and `piper_train`.
#   # piper_train comes from the clone; `piper` comes from piper-tts. The
#   # piper-sample-generator package itself is not installed: it pins
#   # audiomentations==0.33.0 for its own augment.py, which we never call,
#   # and that conflicts with the version openwakeword needs.
#   "piper-tts==1.3.0",
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
#   # AudioSet ships as parquet now, not the .tar the upstream notebook fetches.
#   "pyarrow==22.0.0",
# ]
#
# [tool.uv]
# # piper-tts requires `onnxruntime`, the CPU build. Installed beside
# # onnxruntime-gpu it shadows it, CUDAExecutionProvider disappears, and feature
# # extraction drops to the CPU without raising. The marker below is never true,
# # so uv drops the requirement; onnxruntime-gpu supplies the same `onnxruntime`
# # module and the GPU provider survives.
# override-dependencies = ["onnxruntime ; python_version < '3.0'"]
# ///
"""Job A: generate, augment, and featurise the Hey Sonny training data.

Produces four feature arrays and publishes them to a Dataset repo for Job B to
train against. Runs once per data change; Job B is the one you re-run to tune
hyperparameters.

Contract, flavors and completion criteria: docs/jobs-spec.md
Configuration and the reasoning behind it: docs/research/full-run-plan.md

Stages are restartable. Each writes a COMPLETE marker recording its outputs, and
a rerun skips finished stages unless --force names them. That matters because
the Job disk is ephemeral but a crashed run can be resumed in place through the
read-write volume mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SR = 16000
PIPER_SR = 22050


def embedding_frames(n_samples: int) -> int:
    """Feature frames openWakeWord produces for a window of `n_samples`.

    The melspectrogram uses a 160-sample hop with a 3-frame edge loss, and the
    embedding model consumes 76 melspectrogram frames with a stride of 8. The
    relation is not `n_samples / 1280`: upstream's `seconds_per_example =
    1280 * n_frames / 16000` describes the *stride* an example covers, not the
    audio it needs. Confusing the two yields 20480 samples for a 16-frame
    window, which actually produces 7. Checked against the pilot, where 48000
    samples produced 28 frames.
    """
    melspec_frames = math.ceil(n_samples / 160 - 3)
    return (melspec_frames - 76) // 8 + 1


# 2.00 s exactly, and upstream's default. A range of lengths map to 16 frames
# (31680 < n <= 32960); using upstream's keeps our arrays directly comparable
# with the shipped models.
DEFAULT_TOTAL_LENGTH = 32000

STAGES = ["sources", "generate", "features", "publish"]

# Background pool. Four roles with the same number of files each, because
# torch_audiomentations.AddBackgroundNoise picks `random.choice(files)` per
# clip: the file-count share *is* the draw share. The 2026-09-01 pool was 1,000
# FSD50K + 200 FMA files and no speech at all, and the model it produced
# cannot hear the phrase over other people talking (noise-robustness.md).
BACKGROUND_ROLES = ("speech", "babble", "music", "env")
SOURCE_ROLES = ("speech", "music", "env")     # babble is synthesised from speech
SEGMENT_SECONDS = 10                          # re-cut length; >= the 2 s window
MIN_SEGMENT_SECONDS = 3
STRESS_RESERVE = 6                            # segments per role kept out of training
VAL_CONDITIONS = ("clean", "env", "music", "speech", "babble")


# --------------------------------------------------------------------------
# Configuration. Defaults live here rather than in a separate file: this script
# is versioned in git, and the CLI overrides land in the Job's recorded command,
# so the pair is fully reconstructable from the Job record plus the commit.
# --------------------------------------------------------------------------

def build_config(args) -> dict:
    n_speakers = args.max_speakers
    return {
        "wake_phrase": args.wake_phrase,
        # 16 frames x 96 features. See full-run-plan.md: matches the shipped
        # openWakeWord models and avoids two hardcoded 16s in train.py.
        "window_frames": args.window_frames,
        "total_length": args.total_length,
        "n_train": args.n_train,
        "n_val": args.n_val,
        # Capped near sqrt(clip target): generate_samples walks
        # itertools.product(range(n), range(n)) over speaker pairs, so a cap far
        # above the square root leaves the first element of the pair stuck on the
        # earliest speakers.
        "max_speakers": n_speakers,
        "val_speakers": args.val_speakers,
        "adversarial_texts": list(args.adversarial_texts),
        "adversarial_custom_fraction": args.adversarial_custom_fraction,
        "seed": args.seed,
        "piper_commit": args.piper_commit,
        "piper_model_url": args.piper_model_url,
        "rir_repo": args.rir_repo,
        "background_hours_per_role": args.background_hours_per_role,
        "background_snr_min": args.background_snr_min,
        "background_snr_max": args.background_snr_max,
        "babble_talkers": [2, 5],
        "stress_positives": args.stress_positives,
        "augmentation_rounds": args.augmentation_rounds,
        "augmentation_batch_size": args.augmentation_batch_size,
        "background_sources": args.background_sources,
        "repo_id": args.repo_id,
        # Upstream's generation settings, split so validation is drawn with
        # different noise scales as well as different speakers.
        "gen_train": {"noise_scales": [0.98], "noise_scale_ws": [0.98],
                      "length_scales": [0.75, 1.0, 1.25]},
        "gen_val": {"noise_scales": [1.0], "noise_scale_ws": [1.0],
                    "length_scales": [0.75, 1.0, 1.25]},
    }


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


# --------------------------------------------------------------------------
# Preconditions. Both were found by preflight rather than by reasoning; see
# openwakeword-baseline.md.
# --------------------------------------------------------------------------

def on_gpu() -> bool:
    """True when this flavor has a CUDA device. ACCELERATOR is coarse: 'cpu' or 'gpu'."""
    return os.environ.get("ACCELERATOR", "none") not in {"none", "cpu", "unset", ""}


def patch_torchaudio() -> None:
    """Route `torchaudio.load` and `torchaudio.info` through soundfile.

    torchaudio 2.11 delegates `load` to torchcodec, which is absent, and has
    removed `info`. `augment_clips` reaches both for every RIR and background
    clip. Everything loaded through those paths here is a plain 16 kHz WAV.
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
    """Fail loudly when feature extraction would silently land on the CPU.

    onnxruntime registers CUDAExecutionProvider at import, finding CUDA through
    the libraries torch's nvidia wheels place on the loader path -- hence torch
    first. A missing provider does not raise; AudioFeatures just falls back and
    the stage takes an order of magnitude longer with nothing in the log.
    """
    import torch          # noqa: F401  keep first
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    if not on_gpu():
        return f"CPU flavor; providers={providers}"
    assert "CUDAExecutionProvider" in providers, (
        f"GPU flavor but onnxruntime offers {providers}. Feature extraction "
        "would run on CPU; fix the environment rather than paying for the GPU.")
    return f"{torch.cuda.get_device_name(0)}; providers={providers}"


# --------------------------------------------------------------------------
# Stage bookkeeping
# --------------------------------------------------------------------------

CHECKPOINTED = ("*.COMPLETE.json", "*_features.npy", "manifest.json")


def sync_checkpoint(work: Path, ckpt: Path, direction: str) -> list[str]:
    """Copy stage markers and feature arrays between local disk and the mount.

    Only these travel. The generated WAVs stay local and are regenerated on a
    retry: they are a pure function of the pinned generator, the config and the
    seed, and copying tens of thousands of small files to a bucket costs more
    than making them again.
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
    # The stress assets are a small tree, not a flat file; publish reads them
    # from `work`, so a resumed run needs them back.
    if (src / "stress").is_dir() and not (dst / "stress").is_dir():
        shutil.copytree(src / "stress", dst / "stress")
        moved.append("stress/")
    return moved


def marker(work: Path, stage: str) -> Path:
    return work / f"{stage}.COMPLETE.json"


def stage_done(work: Path, stage: str) -> bool:
    return marker(work, stage).is_file()


def complete(work: Path, stage: str, **detail) -> None:
    """Write the marker last, so a partial stage never looks finished."""
    payload = {"stage": stage, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **detail}
    marker(work, stage).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[{stage}] complete: {json.dumps(detail, sort_keys=True)[:300]}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest) -> None:
    """Stream a file to disk. wget is not guaranteed to exist in the uv image."""
    import requests
    print(f"+ download {url}", flush=True)
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# Stage 1: sources
# --------------------------------------------------------------------------

def to_16k_wav(src: Path, dst: Path) -> None:
    import librosa
    import numpy as np
    import soundfile

    y, _ = librosa.load(str(src), sr=SR, mono=True)
    soundfile.write(str(dst), (y * 32767).astype(np.int16), SR, subtype="PCM_16")


def stage_sources(cfg: dict, work: Path) -> dict:
    """Fetch the generator, the room impulse responses, and background audio."""
    from huggingface_hub import snapshot_download

    piper_dir = work / "piper-sample-generator"
    if not piper_dir.is_dir():
        # The PyPI wheel drops piper_train, which __main__.py imports, so the
        # clone is required even though the package is also a dependency.
        run(["git", "clone", "--quiet", "https://github.com/rhasspy/piper-sample-generator.git",
             str(piper_dir)])
    run(["git", "-C", str(piper_dir), "checkout", "--quiet", cfg["piper_commit"]])

    # Only the weights come from the release. The matching .pt.json config ships
    # in the clone, and generate_samples reads it as f"{model_path}.json" --
    # fetching it from the release URL would overwrite it with a 404 page.
    model_path = piper_dir / "models" / "en_US-libritts_r-medium.pt"
    config_path = Path(str(model_path) + ".json")
    assert config_path.is_file(), f"generator config missing from the clone: {config_path}"
    if not model_path.is_file():
        download(cfg["piper_model_url"], model_path)

    # Room impulse responses: 272 files, ~8 MB, used by augment_clips.
    rir_dir = work / "rirs"
    if not rir_dir.is_dir():
        raw = snapshot_download(cfg["rir_repo"], repo_type="dataset")
        rir_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(Path(raw).rglob("*.wav")):
            to_16k_wav(src, rir_dir / src.name)
    n_rir = len(list(rir_dir.glob("*.wav")))
    assert n_rir > 0, f"no room impulse responses under {rir_dir}"

    # Backgrounds are bounded by hours per role, not by dataset: FSD50K is
    # 34 GB and AudioSet is 2.4 TB, so a subset is mandatory either way. Each
    # source is tagged `role=SOURCE`; raw audio lands under bg_raw/<role> and
    # `_normalise_pool` re-cuts and balances it. Ingest a margin over the cap
    # because re-cutting drops silence and short tails.
    raw_root = work / "bg_raw"
    seconds_cap = cfg["background_hours_per_role"] * 3600 * 1.25 + 120
    ingested = {}
    for role in SOURCE_ROLES:
        role_dir = raw_root / role
        role_dir.mkdir(parents=True, exist_ok=True)
        have = sum(_wav_seconds(p) for p in role_dir.glob("*.wav"))
        for source in [src for r, src in parse_sources(cfg["background_sources"]) if r == role]:
            if have >= seconds_cap:
                break
            have += _ingest_backgrounds(source, role_dir, seconds_cap - have, role=role)
        ingested[role] = round(have / 3600, 2)
        assert have > 0, f"no background audio ingested for role {role!r}"
    print(f"[sources] ingested hours per role: {ingested}", flush=True)

    pool = _normalise_pool(cfg, work)
    return {"piper_commit": cfg["piper_commit"], "piper_model_sha256": sha256_file(model_path),
            "n_rirs": n_rir, "ingested_hours": ingested, "pool": pool,
            "n_backgrounds": sum(v["files"] for v in pool.values())}


def parse_sources(entries: list[str]) -> list[tuple[str, str]]:
    """`role=SOURCE` pairs; a bare SOURCE is `env`, which is what it was before."""
    out = []
    for entry in entries:
        role, sep, source = entry.partition("=")
        if not sep or role not in SOURCE_ROLES:
            role, source = "env", entry
        out.append((role, source))
    return out


def _normalise_pool(cfg: dict, work: Path) -> dict:
    """Re-cut every role into equal-length segments and equalise the file counts.

    Streams file by file so a 4 h role never sits in memory. Each role keeps
    STRESS_RESERVE segments out of training for the stress assets; babble is
    synthesised from speech after the reserve is taken, so the babble stress
    pool never contains a training segment either.
    """
    import numpy as np
    import soundfile

    rng = np.random.default_rng(cfg["seed"] + 1)
    seg = SEGMENT_SECONDS * SR
    cap = int(cfg["background_hours_per_role"] * 3600 / SEGMENT_SECONDS) + STRESS_RESERVE
    pool_root, stress_root = work / "backgrounds", work / "bg_stress"

    def write(role: str, index: int, y, root: Path) -> None:
        (root / role).mkdir(parents=True, exist_ok=True)
        # RMS -20 dBFS unless that would clip; AddBackgroundNoise re-normalises
        # by RMS at mix time, so the level is cosmetic but clipping is not.
        rms = max(float(np.sqrt(np.mean(y ** 2))), 1e-6)
        y = y / max(rms / 0.1, float(np.abs(y).max()) / 0.99)
        soundfile.write(str(root / role / f"{index:05d}.wav"),
                        (y * 32767).astype(np.int16), SR, subtype="PCM_16")

    counts = {}
    for role in SOURCE_ROLES:
        if (pool_root / role).is_dir() and (stress_root / role).is_dir():
            counts[role] = len(list((pool_root / role).glob("*.wav")))
            continue
        buffer, n = np.zeros(0, dtype="float32"), 0
        for src in sorted((work / "bg_raw" / role).glob("*.wav")):
            if n >= cap:
                break
            y, _ = soundfile.read(str(src), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if np.sqrt(np.mean(y ** 2)) < 1e-4:      # silent file; skip rather than amplify it
                continue
            buffer = np.concatenate([buffer, y / np.sqrt(np.mean(y ** 2))])
            while len(buffer) >= seg and n < cap:
                write(role, n, buffer[:seg], pool_root)
                buffer, n = buffer[seg:], n + 1
        if len(buffer) >= MIN_SEGMENT_SECONDS * SR and n < cap:
            write(role, n, buffer, pool_root)
            n += 1
        counts[role] = n

    # Reserve for stress, then balance the three source roles.
    for role in SOURCE_ROLES:
        files = sorted((pool_root / role).glob("*.wav"))
        assert len(files) > STRESS_RESERVE, f"{role}: only {len(files)} segments"
        if not (stress_root / role).is_dir():
            (stress_root / role).mkdir(parents=True)
            for f in files[-STRESS_RESERVE:]:
                shutil.move(str(f), stress_root / role / f.name)
    n = min(len(list((pool_root / r).glob("*.wav"))) for r in SOURCE_ROLES)
    floor = int(0.5 * cfg["background_hours_per_role"] * 3600 / SEGMENT_SECONDS)
    assert n >= max(floor, 1), (
        f"balanced pool would be {n} files per role, under the floor of {floor}; "
        f"per-role counts before balancing: "
        f"{ {r: len(list((pool_root / r).glob('*.wav'))) for r in SOURCE_ROLES} }")
    for role in SOURCE_ROLES:
        files = sorted((pool_root / role).glob("*.wav"))
        for f in rng.permutation(files)[n:] if len(files) > n else []:
            Path(f).unlink()

    # Babble: 2-5 speech segments summed at random offsets. A room with a
    # conversation in it is this, not one clean reader.
    lo, hi = cfg["babble_talkers"]
    for target, src_root, count in ((pool_root, pool_root, n), (stress_root, stress_root, STRESS_RESERVE)):
        if (target / "babble").is_dir():
            continue
        speech = sorted((src_root / "speech").glob("*.wav"))
        for i in range(count):
            k = int(rng.integers(lo, hi + 1))
            mix = np.zeros(seg, dtype="float32")
            for f in rng.choice(speech, size=min(k, len(speech)), replace=False):
                y, _ = soundfile.read(str(f), dtype="float32")
                y = np.resize(y, seg)
                mix += np.roll(y, int(rng.integers(0, seg)))
            write("babble", i, mix, target)

    pool = {role: {"files": len(list((pool_root / role).glob("*.wav"))),
                   "hours": round(len(list((pool_root / role).glob("*.wav"))) * SEGMENT_SECONDS / 3600, 2),
                   "stress_reserve": len(list((stress_root / role).glob("*.wav")))}
            for role in BACKGROUND_ROLES}
    assert len({v["files"] for v in pool.values()}) == 1, f"pool is not balanced: {pool}"
    shutil.rmtree(work / "bg_raw", ignore_errors=True)
    print(f"[sources] balanced pool: {pool}", flush=True)
    return pool


def _wav_seconds(path: Path) -> float:
    import soundfile
    try:
        meta = soundfile.info(str(path))
        return meta.frames / meta.samplerate
    except Exception:
        return 0.0


def _ingest_backgrounds(source: str, out_dir: Path, seconds_needed: float,
                        role: str = "env") -> float:
    """Materialise up to `seconds_needed` of 16 kHz background audio.

    `source` is either an https URL to a zip, or `repo/type/path` naming a file
    in a Hub dataset repo. Both resolve to a named artifact at a fixed path,
    which is more reproducible than a streaming dataset whose order can change
    between runs.
    """
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir.parent / "_bg_staging"
    staging.mkdir(parents=True, exist_ok=True)

    if "::" in source and source.split("::", 1)[1].endswith(".parquet"):
        # Hub audio datasets are parquet now; `shutil.unpack_archive` cannot
        # reach them. Schema confirmed 2026-09-02 on agkphysics/AudioSet:
        # `audio: struct<bytes: binary, path: string>`, 500 rows and ~1.4 h per
        # shard. Addressing one named shard rather than a streaming split keeps
        # the run reproducible -- a streaming order can change between runs.
        import io

        import librosa
        import numpy as np
        import pyarrow.parquet as pq
        import soundfile
        from huggingface_hub import hf_hub_download

        repo_id, filename = source.split("::", 1)
        local = hf_hub_download(repo_id, filename, repo_type="dataset")
        added = 0.0
        pf = pq.ParquetFile(local)
        # AudioSet rows carry `human_labels`; use them so a `music=` shard is
        # music and an `env=` shard has neither speech nor music in it. Datasets
        # without the column (LibriSpeech) pass everything through.
        labelled = "human_labels" in pf.schema_arrow.names
        columns = ["audio", "human_labels"] if labelled else ["audio"]
        kept = skipped_label = 0
        for batch in pf.iter_batches(batch_size=32, columns=columns):
            records = batch.column("audio").to_pylist()
            labels = batch.column("human_labels").to_pylist() if labelled else [None] * len(records)
            for row, (record, tags) in enumerate(zip(records, labels)):
                if added >= seconds_needed:
                    break
                if tags is not None:
                    tags = " ".join(str(t) for t in tags).lower()
                    if (role == "music" and "music" not in tags) or \
                       (role == "env" and ("speech" in tags or "music" in tags)) or \
                       (role == "speech" and "speech" not in tags):
                        skipped_label += 1
                        continue
                kept += 1
                stem = Path(record.get("path") or "").stem or f"{Path(filename).stem}_{kept}"
                dst = out_dir / (stem + ".wav")
                if not dst.is_file():
                    try:
                        raw, rate = soundfile.read(io.BytesIO(record["bytes"]),
                                                   dtype="float32", always_2d=True)
                        mono = librosa.to_mono(raw.T)
                        if rate != SR:
                            mono = librosa.resample(mono, orig_sr=rate, target_sr=SR)
                        soundfile.write(str(dst), (mono * 32767).astype(np.int16),
                                        SR, subtype="PCM_16")
                    except Exception as exc:      # one bad clip is not fatal
                        print(f"  skipped {stem}: {exc}", flush=True)
                        continue
                added += _wav_seconds(dst)
            if added >= seconds_needed:
                break
        shutil.rmtree(staging, ignore_errors=True)
        print(f"  {source} [{role}]: +{added/3600:.2f} h, {kept} rows kept, "
              f"{skipped_label} dropped by label", flush=True)
        return added

    if source.startswith("http"):
        archive = staging / Path(source).name
        if not archive.is_file():
            download(source, archive)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(staging)
        else:
            shutil.unpack_archive(str(archive), str(staging))
    else:
        from huggingface_hub import hf_hub_download
        repo_id, filename = source.split("::", 1)
        local = hf_hub_download(repo_id, filename, repo_type="dataset")
        shutil.unpack_archive(local, str(staging))

    added = 0.0
    audio = sorted(p for p in staging.rglob("*")
                   if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"})
    for src in audio:
        if added >= seconds_needed:
            break
        dst = out_dir / (src.stem + ".wav")
        if not dst.is_file():
            try:
                to_16k_wav(src, dst)
            except Exception as exc:            # a single unreadable clip is not fatal
                print(f"  skipped {src.name}: {exc}", flush=True)
                continue
        added += _wav_seconds(dst)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  {source}: +{added/3600:.2f} h", flush=True)
    return added


# --------------------------------------------------------------------------
# Stage 2: generate, resample, and split by speaker
# --------------------------------------------------------------------------

def speaker_pair(index: int, n_speakers: int) -> tuple[int, int]:
    """Recover the speaker pair for a generated clip from its index.

    generate_samples consumes `itertools.cycle(itertools.product(range(n),
    range(n)))` in order and names files `{index}.wav`, so the mapping is exact
    as long as file_names is left unset. Upstream passes uuid4 names instead,
    which is precisely why its train and validation sets share speakers.
    """
    return divmod(index % (n_speakers * n_speakers), n_speakers)


def stage_generate(cfg: dict, work: Path) -> dict:
    import numpy as np

    import logging

    sys.path.insert(0, str(work / "piper-sample-generator"))
    from piper_sample_generator.__main__ import generate_samples

    # piper_sample_generator.__main__ calls logging.basicConfig(level=DEBUG) at
    # import time, which turns the root logger to DEBUG for the whole process.
    # Every httpx request during publish then dumps its full headers, including
    # signed URLs and access tokens, into the Job log. Put it back.
    logging.getLogger().setLevel(logging.WARNING)

    import openwakeword.data

    model = str(work / "piper-sample-generator" / "models" / "en_US-libritts_r-medium.pt")
    n_speakers = cfg["max_speakers"]
    rng = np.random.default_rng(cfg["seed"])

    # Hold out whole speakers, then send every clip touching one to validation.
    # Splitting clips at random would leave the same voices on both sides and
    # make validation recall a memorisation score.
    held_out = set(rng.choice(n_speakers, size=cfg["val_speakers"], replace=False).tolist())

    # Generating exactly n_train + n_val would under-deliver training clips. The
    # speaker holdout claims every clip touching a held-out speaker in either
    # position, which is 1 - ((n - v)/n)**2 of the run -- 13.4% at 12 of 173, not
    # the n_val/(n_train + n_val) the naive count assumes. Generate enough that
    # the training quota still fills, with a small margin for the tail of the
    # itertools.product cycle.
    val_fraction = 1 - ((n_speakers - cfg["val_speakers"]) / n_speakers) ** 2
    # Count the split exactly rather than estimating it. The index-to-speaker
    # mapping is deterministic, so walking it tells us precisely how many clips
    # to generate for both quotas to fill. A fraction-based estimate with a
    # margin is right on average and wrong on any particular run.
    total, filled_train, filled_val = 0, 0, 0
    while filled_train < cfg["n_train"] or filled_val < cfg["n_val"]:
        a, b = speaker_pair(total, n_speakers)
        if a in held_out or b in held_out:
            filled_val += 1
        else:
            filled_train += 1
        total += 1
    expected_val = filled_val
    print(f"[generate] holding out {cfg['val_speakers']}/{n_speakers} speakers "
          f"({val_fraction:.1%} of clips); generating {total} per class", flush=True)

    custom = list(cfg["adversarial_texts"])
    n_custom = int(round(cfg["adversarial_custom_fraction"] * total)) if custom else 0

    counts = {}
    for kind in ("positive", "adversarial"):
        raw = work / f"{kind}_raw"
        # The 16 kHz split is the record; the 22 kHz originals are removed once
        # it is verified (below). A resume that finds the split complete must
        # not regenerate.
        split_done = all(
            len(list((work / f"{kind}_{role}").glob("*.wav"))) >= cfg[f"n_{role}"]
            for role in ("train", "val"))
        raw_done = raw.is_dir() and len(list(raw.glob("*.wav"))) >= total
        if not split_done and not raw_done:
            raw.mkdir(parents=True, exist_ok=True)
            if kind == "positive":
                text = cfg["wake_phrase"]
                batch = 50
            else:
                # Phoneme-overlap negatives. include_partial_phrase=1.0 keeps
                # fragments like "hey son"; include_input_words=0.2 admits the
                # real words in other combinations, e.g. "hey sunny day".
                text = list(openwakeword.data.generate_adversarial_texts(
                    input_text=cfg["wake_phrase"], N=total,
                    include_partial_phrase=1.0, include_input_words=0.2))
                # It samples with deduplication and returns fewer than N
                # (v3: 21,760 of 115,383). generate_samples cycles whatever
                # list it gets, so cycle it here to a slot per clip first.
                text = [text[i % len(text)] for i in range(total)]
                if n_custom:
                    # generate_samples cycles the list in order, one text per
                    # clip, and speaker_pair(index) walks the speaker grid.
                    # Scattering the phrases over random slots spreads them
                    # across voices instead of the first n_custom pairs.
                    slots = np.random.default_rng(cfg["seed"] + 2).choice(
                        total, size=n_custom, replace=False)
                    for j, k in enumerate(sorted(slots.tolist())):
                        text[k] = custom[j % len(custom)]
                    print(f"[generate] adversarial: {n_custom} of {total} texts are "
                          f"the {len(custom)} custom phrases", flush=True)
                batch = 50 // 7
            settings = cfg["gen_train"]
            generate_samples(
                text=text, output_dir=str(raw), model=model, max_samples=total,
                batch_size=batch, max_speakers=n_speakers,
                slerp_weights=[0.5], length_scales=settings["length_scales"],
                noise_scales=settings["noise_scales"],
                noise_scale_ws=settings["noise_scale_ws"],
                # file_names deliberately unset: default {index}.wav is what
                # makes speaker_pair() exact.
            )

        # Piper writes 22050 Hz. augment_clips raises on any other rate, and
        # nothing downstream resamples, so this is mandatory rather than tidy.
        for role in ("train", "val"):
            (work / f"{kind}_{role}").mkdir(parents=True, exist_ok=True)

        n_train = n_val = 0
        for src in sorted(raw.glob("*.wav"), key=lambda p: int(p.stem)) if raw.is_dir() else []:
            index = int(src.stem)
            a, b = speaker_pair(index, n_speakers)
            is_val = a in held_out or b in held_out
            if is_val and n_val >= cfg["n_val"]:
                continue
            if not is_val and n_train >= cfg["n_train"]:
                continue
            role = "val" if is_val else "train"
            dst = work / f"{kind}_{role}" / f"{index}.wav"
            if not dst.is_file():
                to_16k_wav(src, dst)
            n_val += is_val
            n_train += not is_val

        # Verify against the filesystem, not the counters. The counters record
        # what the loop intended to write; only a listing shows what survived.
        on_disk = {role: len(list((work / f"{kind}_{role}").glob("*.wav")))
                   for role in ("train", "val")}
        if not raw.is_dir():
            # Resumed after the originals were removed: the split is the record.
            n_train, n_val = on_disk["train"], on_disk["val"]
        print(f"[generate] {kind}: counted {{'train': {n_train}, 'val': {n_val}}}, "
              f"on disk {on_disk}", flush=True)
        assert on_disk == {"train": n_train, "val": n_val}, (
            f"{kind}: counters say train={n_train} val={n_val} but the "
            f"filesystem shows {on_disk}")

        counts[kind] = {"train": n_train, "val": n_val}
        # Quotas, not just non-empty: a short split would train on less data than
        # the manifest claims and only show up as a worse model.
        assert n_train >= cfg["n_train"], f"{kind}: {n_train} train clips, wanted {cfg['n_train']}"
        assert n_val >= cfg["n_val"], f"{kind}: {n_val} val clips, wanted {cfg['n_val']}"

        # The 22 kHz originals are dead weight once the split is verified: at
        # 100k clips per class they are ~15 GB the feature stage never reads.
        if raw.is_dir():
            shutil.rmtree(raw)
        print(f"[generate] {kind}: removed {raw.name}; "
              f"{shutil.disk_usage(work).free / 1e9:.1f} GB free under {work}", flush=True)

    return {"counts": counts, "held_out_speakers": sorted(held_out),
            "n_speakers": n_speakers, "generated_per_class": total,
            "val_fraction": round(val_fraction, 4),
            "custom_adversarial": {"phrases": custom, "n_clips": n_custom,
                                   "fraction": cfg["adversarial_custom_fraction"]}}


# --------------------------------------------------------------------------
# Stage 3: augment and extract features
# --------------------------------------------------------------------------

def stage_features(cfg: dict, work: Path) -> dict:
    import numpy as np

    import openwakeword.data
    import openwakeword.utils

    patch_torchaudio()
    print("[features] " + assert_gpu_features(), flush=True)

    # The melspectrogram and embedding models are not bundled in the wheel, so a
    # fresh container has neither. Called every run rather than in `sources`:
    # the stage markers survive on the read-write mount, but the uv environment
    # they would be downloaded into does not.
    openwakeword.utils.download_models()

    rirs = [str(p) for p in sorted((work / "rirs").glob("*.wav"))]
    by_role = {role: [str(p) for p in sorted((work / "backgrounds" / role).glob("*.wav"))]
               for role in BACKGROUND_ROLES}
    backgrounds = [f for files in by_role.values() for f in files]
    # Assert here, not only in `sources`. An empty list is not an error to
    # augment_clips: it just skips reverberation or background mixing and
    # returns quieter, cleaner audio. The arrays still come out the right shape,
    # so a silent miss would only ever show up as a worse model.
    assert rirs, f"no room impulse responses under {work / 'rirs'}"
    for role, files in by_role.items():
        assert files, f"no {role} background clips under {work / 'backgrounds' / role}"
    patch_background_snr(cfg)
    total_length = cfg["total_length"]
    device = "gpu" if on_gpu() else "cpu"
    ncpu = 1 if on_gpu() else max(1, (os.cpu_count() or 2) // 2)

    shapes = {}
    recoveries = {"gpu_retries": 0, "cpu_batches": 0}
    for kind in ("positive", "adversarial"):
        for role in ("train", "val"):
            name = f"{kind}_{role}"
            out = work / f"{name}_features.npy"
            clips = [str(p) for p in sorted((work / name).glob("*.wav"))]
            # An empty list here surfaces as StopIteration from deep inside
            # compute_features_from_generator, which says nothing about the cause.
            print(f"[features] {name}: {len(clips)} clips under {work / name}", flush=True)
            assert clips, (
                f"no clips found under {work / name}. The generate stage reported "
                "writing them, so this is a filesystem visibility problem, not a "
                "generation one -- check whether --work is on a bucket mount.")
            want = len(clips) * (cfg["augmentation_rounds"] if role == "train" else 1)
            if out.is_file() and np.load(out, mmap_mode="r").shape[0] == want:
                shapes[name] = list(np.load(out, mmap_mode="r").shape)
                continue

            # Rounds apply to training data only. Validation stays at one
            # augmentation per clip so its recall remains comparable with
            # earlier runs.
            #
            # Each round is a fresh background file and SNR per clip
            # (AddBackgroundNoise draws both per clip; only the RIR is drawn
            # once per batch), so three rounds show every positive under three
            # different conditions. List multiplication repeats the whole list,
            # so a clip's copies land far apart -- do not sort after multiplying.
            rounds = cfg["augmentation_rounds"] if role == "train" else 1
            expanded = clips * rounds
            generator = robust_augment(
                expanded, recoveries, total_length=total_length,
                batch_size=cfg["augmentation_batch_size"],
                background_clip_paths=backgrounds, RIR_paths=rirs)
            openwakeword.utils.compute_features_from_generator(
                generator, n_total=len(expanded), clip_duration=total_length,
                output_file=str(out), device=device, ncpu=ncpu)

            array = np.load(out, mmap_mode="r")
            assert array.shape[1:] == (cfg["window_frames"], 96), (
                f"{name}: expected (*, {cfg['window_frames']}, 96), got {array.shape}")
            shapes[name] = list(array.shape)
            print(f"[features] {name}: {array.shape}", flush=True)

    # Per-condition validation positives. The mixed `positive_val` above cannot
    # say *which* interference a model fails under; these can, and Job B reads
    # recall at a matched false-accept budget off each one.
    val_clips = [str(p) for p in sorted((work / "positive_val").glob("*.wav"))]
    default_p = {"SevenBandParametricEQ": 0.25, "TanhDistortion": 0.25, "PitchShift": 0.25,
                 "BandStopFilter": 0.25, "AddColoredNoise": 0.25, "AddBackgroundNoise": 0.75,
                 "Gain": 1.0, "RIR": 0.5}
    for cond in VAL_CONDITIONS:
        name = f"positive_val_{cond}"
        out = work / f"{name}_features.npy"
        if out.is_file() and np.load(out, mmap_mode="r").shape[0] == len(val_clips):
            shapes[name] = list(np.load(out, mmap_mode="r").shape)
            continue
        if cond == "clean":
            probs, files = {**default_p, "AddBackgroundNoise": 0.0, "RIR": 0.0,
                            "AddColoredNoise": 0.0}, []
        else:
            probs, files = {**default_p, "AddBackgroundNoise": 1.0}, by_role[cond]
        generator = robust_augment(
            val_clips, recoveries, total_length=total_length,
            batch_size=cfg["augmentation_batch_size"],
            augmentation_probabilities=probs,
            background_clip_paths=files, RIR_paths=rirs)
        openwakeword.utils.compute_features_from_generator(
            generator, n_total=len(val_clips), clip_duration=total_length,
            output_file=str(out), device=device, ncpu=ncpu)
        array = np.load(out, mmap_mode="r")
        assert array.shape == (len(val_clips), cfg["window_frames"], 96), (
            f"{name}: got {array.shape}")
        shapes[name] = list(array.shape)
        print(f"[features] {name}: {array.shape}", flush=True)

    stress = write_stress_assets(cfg, work)
    return {"shapes": shapes, "device": device, "n_rirs": len(rirs),
            "n_backgrounds": len(backgrounds),
            "backgrounds_by_role": {r: len(f) for r, f in by_role.items()},
            "background_snr_db": [cfg["background_snr_min"], cfg["background_snr_max"]],
            "augmentation_recoveries": recoveries, "stress": stress}


def robust_augment(clip_paths, recoveries: dict, batch_size: int, **kwargs):
    """`augment_clips`, resumed past a transient GPU failure.

    Run 6a99bf72 (E3) died 300k clips in with `cuFFT error:
    CUFFT_INTERNAL_ERROR` from julius's band-stop filter: one batch of some
    38,000. compute_features_from_generator sizes its memmap to n_total up
    front and fills it in order, so a generator that stops short leaves zero
    rows at the tail, and one that skips a batch shifts nothing but loses
    rows the manifest still claims. The batch is therefore retried on the
    GPU after the cache is emptied, then run on the CPU (augment_clips picks
    its device from torch.cuda.is_available() per batch). Counts land in
    `recoveries` so the manifest says whether it happened.
    """
    import torch

    import openwakeword.data

    start = 0
    while start < len(clip_paths):
        try:
            for batch in openwakeword.data.augment_clips(
                    clip_paths[start:], batch_size=batch_size, **kwargs):
                yield batch
                start += batch.shape[0]
        except RuntimeError as exc:
            if "cuFFT" not in str(exc) and "CUDA" not in str(exc):
                raise
            end = min(start + batch_size, len(clip_paths))
            print(f"[features] GPU augmentation failed at clip {start}: {exc}", flush=True)
            torch.cuda.empty_cache()
            try:
                batch = next(openwakeword.data.augment_clips(
                    clip_paths[start:end], batch_size=batch_size, **kwargs))
                recoveries["gpu_retries"] += 1
            except RuntimeError as again:
                print(f"[features] retry failed ({again}); running the batch on CPU", flush=True)
                available = torch.cuda.is_available
                torch.cuda.is_available = lambda: False
                try:
                    batch = next(openwakeword.data.augment_clips(
                        clip_paths[start:end], batch_size=batch_size, **kwargs))
                finally:
                    torch.cuda.is_available = available
                recoveries["cpu_batches"] += 1
            yield batch
            start += batch.shape[0]


def patch_background_snr(cfg: dict) -> None:
    """Apply --background-snr-min/max. `augment_clips` hardcodes -10..15 dB in
    the AddBackgroundNoise it builds; it looks the class up on the module at
    call time, so a subclass installed there takes effect. A no-op at the
    library defaults, which are the flag defaults."""
    import torch_audiomentations

    lo, hi = cfg["background_snr_min"], cfg["background_snr_max"]
    if (lo, hi) == (-10, 15):
        return
    base = torch_audiomentations.AddBackgroundNoise

    class AddBackgroundNoiseSNR(base):
        def __init__(self, *args, **kwargs):
            kwargs["min_snr_in_db"], kwargs["max_snr_in_db"] = lo, hi
            super().__init__(*args, **kwargs)

    torch_audiomentations.AddBackgroundNoise = AddBackgroundNoiseSNR
    print(f"[features] background SNR range overridden to {lo}..{hi} dB", flush=True)


def write_stress_assets(cfg: dict, work: Path) -> dict:
    """Small raw-audio set for Job B to score at fixed SNRs: held-out clean
    positives plus a minute of each background role from the training reserve."""
    import numpy as np
    import soundfile

    root = work / "stress"
    if (root / "manifest.json").is_file():
        return json.loads((root / "manifest.json").read_text())
    rng = np.random.default_rng(cfg["seed"] + 2)
    (root / "positives").mkdir(parents=True, exist_ok=True)
    (root / "pools").mkdir(parents=True, exist_ok=True)

    val_clips = sorted((work / "positive_val").glob("*.wav"))
    picks = sorted(rng.choice(val_clips, size=min(cfg["stress_positives"], len(val_clips)),
                              replace=False), key=lambda p: int(p.stem))
    for p in picks:
        shutil.copy2(p, root / "positives" / p.name)

    pools = {}
    for role in BACKGROUND_ROLES:
        parts = [soundfile.read(str(f), dtype="float32")[0]
                 for f in sorted((work / "bg_stress" / role).glob("*.wav"))]
        assert parts, f"no stress reserve for {role}"
        y = np.concatenate(parts)
        soundfile.write(str(root / "pools" / f"{role}.wav"), (y * 32767).astype(np.int16),
                        SR, subtype="PCM_16")
        pools[role] = round(len(y) / SR, 1)

    detail = {"positives": [p.name for p in picks],
              "speaker_pairs": {p.name: speaker_pair(int(p.stem), cfg["max_speakers"]) for p in picks},
              "pool_seconds": pools, "sample_rate": SR}
    (root / "manifest.json").write_text(json.dumps(detail, indent=2, sort_keys=True))
    print(f"[features] stress assets: {len(picks)} positives, pools {pools}", flush=True)
    return detail


# --------------------------------------------------------------------------
# Stage 4: publish and verify
# --------------------------------------------------------------------------

def stage_publish(cfg: dict, work: Path, manifest: dict) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    repo_id = cfg["repo_id"]
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    staging = work / "publish"
    staging.mkdir(parents=True, exist_ok=True)
    checksums = {}
    names = ["positive_train", "positive_val", "adversarial_train", "adversarial_val"]
    names += [f"positive_val_{cond}" for cond in VAL_CONDITIONS]
    for name in names:
        src = work / f"{name}_features.npy"
        dst = staging / f"{name}_features.npy"
        if not dst.is_file():
            shutil.copy2(src, dst)
        checksums[dst.name] = sha256_file(dst)
    if (work / "stress").is_dir():
        if (staging / "stress").is_dir():
            shutil.rmtree(staging / "stress")
        shutil.copytree(work / "stress", staging / "stress")
        for path in sorted((staging / "stress").rglob("*")):
            if path.is_file():
                checksums[str(path.relative_to(staging))] = sha256_file(path)

    manifest["checksums"] = checksums
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # One commit, so a reader never sees a half-published set.
    commit = api.upload_folder(folder_path=str(staging), repo_id=repo_id,
                               repo_type="dataset",
                               commit_message=f"Job A {manifest['attempt_id']}")
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)

    # A successful push is not proof the artifact persisted: read it back.
    mismatched = []
    for filename, digest in checksums.items():
        local = hf_hub_download(repo_id, filename, repo_type="dataset",
                                revision=revision, force_download=True)
        if sha256_file(Path(local)) != digest:
            mismatched.append(filename)
    assert not mismatched, f"read-back checksums differ for {mismatched}"

    return {"repo_id": repo_id, "revision": revision, "files": sorted(checksums)}


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="/work", help=(
        "Working tree, on local ephemeral disk. Do NOT point this at a bucket "
        "mount: reads after writes there proved inconsistent, and a stage that "
        "sees an empty RIR directory degrades silently rather than failing."))
    ap.add_argument("--checkpoint-dir", default="/out/joba", help=(
        "Durable directory for stage markers and feature arrays, so a crashed "
        "run resumes without repeating the expensive stage."))
    ap.add_argument("--repo-id", default="iamtokarev/hey-sonny-features")
    ap.add_argument("--wake-phrase", default="hey sonny")
    ap.add_argument("--n-train", type=int, default=30_000)
    ap.add_argument("--n-val", type=int, default=2_000)
    ap.add_argument("--max-speakers", type=int, default=173, help="~sqrt(n_train + n_val)")
    ap.add_argument("--val-speakers", type=int, default=12, help="Speakers held out of training")
    ap.add_argument("--adversarial-texts", nargs="*", default=[
        # Name swaps one or two phonemes from "sonny", plus the openers people
        # say near a device. "hey sunny" is deliberately absent: it is a
        # homophone of the wake phrase and would label the phrase negative.
        "hey honey", "hey johnny", "hey tony", "hey ronnie", "hey bonnie",
        "hey connie", "hey donnie", "hey sonic", "hey sunday", "hey money",
        "hey siri", "hey google", "hey sam", "hey mommy",
        "hey, so anyway", "okay so", "so funny", "say sorry",
    ], help=(
        "Phrases synthesised as negatives in place of a share of the "
        "phoneme-overlap texts generate_adversarial_texts produces. Pass "
        "nothing to disable."))
    ap.add_argument("--adversarial-custom-fraction", type=float, default=0.2, help=(
        "Share of adversarial clips drawn from --adversarial-texts."))
    ap.add_argument("--augmentation-rounds", type=int, default=1, help=(
        "Augment each training clip this many times; each round draws a "
        "fresh background file and SNR per clip."))
    ap.add_argument("--augmentation-batch-size", type=int, default=16, help=(
        "Clips per augmentation batch. The room impulse response is drawn "
        "once per batch, so smaller means more reverb variety and slower; "
        "custom_model.yml warns against raising it."))
    ap.add_argument("--background-hours-per-role", type=float, default=4.0, help=(
        "Cap per background role after re-cutting into 10 s segments. Roles "
        "are then truncated to the smallest count so each is drawn equally."))
    ap.add_argument("--background-sources", nargs="*", default=[
        # LibriSpeech readers are disjoint from the LibriTTS-R voices Piper
        # synthesises the positives with, so no speaker leaks into the pool.
        "speech=openslr/librispeech_asr::other/validation/0000.parquet",
        "speech=openslr/librispeech_asr::other/test/0000.parquet",
        # The FMA sample is 200 x 30 s = 1.7 h; each AudioSet shard yields
        # ~0.45 h of Music-labelled rows, so six shards reach the 4 h cap.
        "music=https://f002.backblazeb2.com/file/openwakeword-resources/data/fma_sample.zip",
        "music=agkphysics/AudioSet::data/bal_train/00.parquet",
        "music=agkphysics/AudioSet::data/bal_train/02.parquet",
        "music=agkphysics/AudioSet::data/bal_train/03.parquet",
        "music=agkphysics/AudioSet::data/bal_train/04.parquet",
        "music=agkphysics/AudioSet::data/bal_train/05.parquet",
        "music=agkphysics/AudioSet::data/bal_train/06.parquet",
        "env=https://f002.backblazeb2.com/file/openwakeword-resources/data/fsd50k_sample.zip",
        "env=agkphysics/AudioSet::data/bal_train/01.parquet",
        "env=agkphysics/AudioSet::data/bal_train/07.parquet",
    ], help=(
        "'role=SOURCE' with role in speech|music|env; a bare SOURCE is env. "
        "SOURCE is an https zip URL or 'repo_id::path' to a Hub dataset file "
        "(zip, tar, or parquet with an `audio` column). AudioSet parquet rows "
        "are filtered by `human_labels` to match the role."))
    ap.add_argument("--background-snr-min", type=float, default=-10.0, help=(
        "SNR range for mixing backgrounds under clips, dB. Defaults are the "
        "augment_clips values; the released models used 0..20."))
    ap.add_argument("--background-snr-max", type=float, default=15.0)
    ap.add_argument("--stress-positives", type=int, default=24, help=(
        "Held-out clean positives published under stress/ for Job B to score "
        "at fixed SNRs."))
    ap.add_argument("--rir-repo", default="davidscripka/MIT_environmental_impulse_responses")
    # Pinned to what preflight verified on 2026-08-31. A branch name is not a
    # pin: the manifest needs a full commit.
    ap.add_argument("--piper-commit", default="2971426a55072f7d22fec416ca7800df8bd23207")
    ap.add_argument("--piper-model-url", default=(
        "https://github.com/rhasspy/piper-sample-generator/releases/download/"
        "v2.0.0/en_US-libritts_r-medium.pt"))
    ap.add_argument("--window-frames", type=int, default=16)
    ap.add_argument("--total-length", type=int, default=DEFAULT_TOTAL_LENGTH,
                    help="Window in samples; must yield --window-frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="Re-run these stages even if their COMPLETE marker exists")
    ap.add_argument("--stop-after", choices=STAGES, help="Stop once this stage finishes")
    args = ap.parse_args()

    cfg = build_config(args)
    # The window and the frame count have to agree, and the relation is not
    # obvious enough to leave to a comment.
    actual = embedding_frames(cfg["total_length"])
    assert actual == cfg["window_frames"], (
        f"total_length={cfg['total_length']} yields {actual} frames, not "
        f"{cfg['window_frames']}")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if ckpt:
        restored = sync_checkpoint(work, ckpt, "restore")
        print(f"restored from {ckpt}: {restored or 'nothing'}", flush=True)

    manifest = {
        "attempt_id": os.environ.get("JOB_ID", "local"),
        "config": cfg,
        "config_sha256": config_hash(cfg),
        "environment": {k: os.environ.get(k, "unset")
                        for k in ("ACCELERATOR", "CPU_CORES", "MEMORY")},
        "openwakeword_commit": "368c03716d1e92591906a84949bc477f3a834455",
        "stages": {},
    }
    print(json.dumps({"config_sha256": manifest["config_sha256"], **cfg}, indent=2), flush=True)

    for stage in STAGES:
        if stage_done(work, stage) and stage not in args.force:
            manifest["stages"][stage] = json.loads(marker(work, stage).read_text())
            print(f"[{stage}] already complete, skipping", flush=True)
        else:
            started = time.time()
            print(f"\n=== {stage} ===", flush=True)
            if stage == "sources":
                detail = stage_sources(cfg, work)
            elif stage == "generate":
                detail = stage_generate(cfg, work)
            elif stage == "features":
                detail = stage_features(cfg, work)
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

    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("\n" + json.dumps(manifest["stages"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
