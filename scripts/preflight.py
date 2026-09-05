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
#   # acoustics 0.2.6 (latest, unmaintained) imports scipy.special.sph_harm,
#   # which scipy removed in 1.17, and it declares no upper bound. Raising this
#   # pin re-breaks `import openwakeword.data`.
#   "scipy==1.16.3",
#   # onnxscript is required because torch>=2.9 routes torch.onnx.export through
#   # the dynamo exporter.
#   "onnxruntime-gpu==1.29.0",
#   "onnxscript==0.7.1",
#   "soundfile==0.14.0",
#   "librosa==0.11.0",
#   "huggingface_hub==1.29.0",
#   "numpy==2.5.2",
#   "requests==2.34.2",
#   # Job A reads Hub audio datasets (LibriSpeech, AudioSet) as parquet shards.
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
"""Preflight for the Hey Sonny training jobs.

Proves the environment before any paid GPU minute is spent, and emits the exact
pinned dependency block to paste back into every job script. Run it twice: once
on `cpu-basic` for the cheap resolution and Hub checks, then once on the target
GPU flavor, where the accelerator checks actually mean something.

Contract and completion criteria: docs/jobs-spec.md
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import traceback
import uuid

# Every distribution named in the header above, so the pinned block it emits is
# complete rather than a subset someone has to remember to extend.
PINNED = [
    "piper-tts",
    "torch", "torchaudio", "torchinfo", "torchmetrics",
    "speechbrain", "audiomentations", "torch-audiomentations",
    "acoustics", "pronouncing", "mutagen", "pyyaml",
    "onnxruntime-gpu", "onnxscript",
    "soundfile", "librosa", "huggingface_hub",
    "numpy", "requests", "scipy",
]

OWW_COMMIT = "368c03716d1e92591906a84949bc477f3a834455"

PIPER_REPO = "https://github.com/rhasspy/piper-sample-generator.git"
PIPER_MODEL_URL = ("https://github.com/rhasspy/piper-sample-generator/releases/"
                   "download/v2.0.0/en_US-libritts_r-medium.pt")

# Upstream's false-positive validation array. Its real duration is 10.70 h, not
# the 11.3 that `auto_train` hardcodes; every false-accept rate depends on this
# shape, so a change to the file has to fail loudly rather than skew a metric.
FP_REPO = "davidscripka/openwakeword_features"
FP_FILE = "validation_set_features.npy"
FP_SHAPE = (481345, 96)
SECONDS_PER_FRAME = 0.08

# String anchors our patches replace. They cannot drift while the commit is
# pinned, but a cached or re-resolved wheel could still hand us different
# source, and a patch that silently matches nothing is worse than a crash.
PATCH_ANCHORS = {
    "data.py": [
        "torch.from_numpy(np.where(mixed_clips_batch.max(dim=1) != 0)[0])",
    ],
    "train.py": [
        "val_set_hrs = 11.3",
    ],
}

# ACCELERATOR is coarse, not the flavor name: cpu-basic reports 'cpu' and
# t4-small reports 'gpu'. Match the CPU-only values rather than enumerating GPU
# names, which is why this survived that surprise.
CPU_FLAVORS = {"none", "cpu", "unset", ""}


def on_gpu() -> bool:
    """True when this flavor has a CUDA device."""
    return os.environ.get("ACCELERATOR", "none") not in CPU_FLAVORS


results: list[tuple[str, bool, str]] = []


def check(name):
    """Run a check, record pass/fail, and keep going so one run reports everything."""
    def wrap(fn):
        print(f"\n--- {name} ---", flush=True)
        try:
            detail = fn() or ""
            results.append((name, True, detail))
            print(f"PASS  {name}" + (f": {detail}" if detail else ""), flush=True)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            results.append((name, False, detail))
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            print(f"FAIL  {name}: {detail}", flush=True)
        return fn
    return wrap


def runtime():
    import platform
    v = platform.python_version()
    assert v.startswith("3.12."), f"expected Python 3.12.x, got {v}"
    env = {k: os.environ.get(k, "unset") for k in ("JOB_ID", "ACCELERATOR", "CPU_CORES", "MEMORY")}
    return f"python {v}; " + ", ".join(f"{k}={v}" for k, v in env.items())


def imports():
    # openwakeword.data is the expensive one: it imports pronouncing,
    # audiomentations, torch_audiomentations, speechbrain, mutagen and acoustics
    # at module level, so a gap here would otherwise surface deep inside Job A.
    # torch first, and everywhere else in this pipeline too. onnxruntime
    # registers its CUDA execution provider when it is first imported, and it
    # finds the CUDA libraries through the ones torch's nvidia wheels put on the
    # loader path. Import onnxruntime ahead of torch and the provider can fail
    # to register, which does not raise -- it silently drops feature extraction
    # onto the CPU. The old alphabetical `import librosa, onnxruntime, ..., torch`
    # here made GPU availability depend on import order.
    import torch                 # noqa: F401  keep first
    import onnxruntime           # noqa: F401
    import openwakeword          # noqa: F401
    import openwakeword.data     # noqa: F401
    import openwakeword.train    # noqa: F401
    import openwakeword.utils    # noqa: F401
    import librosa, onnxscript, soundfile  # noqa: F401
    return "openwakeword.data, openwakeword.train and the audio stack all import"


def accelerator():
    import torch          # first: loading torch exposes its bundled CUDA libs
    import onnxruntime

    providers = onnxruntime.get_available_providers()
    if not on_gpu():
        flavor = os.environ.get("ACCELERATOR", "none")
        return (f"flavor reports ACCELERATOR={flavor!r}, so the CUDA checks are "
                f"skipped; providers={providers}. Re-run on the target GPU "
                "flavor before the paid run.")

    assert torch.cuda.is_available(), "ACCELERATOR is set but torch.cuda.is_available() is False"
    # AudioFeatures(device="gpu") asks for CUDAExecutionProvider and silently
    # falls back to CPU when it is missing. That fallback cost the Colab pilot
    # its whole feature-extraction budget on a T4.
    assert "CUDAExecutionProvider" in providers, (
        f"onnxruntime offers {providers}; install onnxruntime-gpu or feature "
        "extraction runs on CPU"
    )
    return (f"{torch.cuda.get_device_name(0)}; torch cuda {torch.version.cuda}; "
            f"providers={providers}")


def shared_models():
    # The melspectrogram and embedding models are not bundled in the wheel;
    # download_models() fetches them (plus the pretrained wake-word models, which
    # we do not need but which cost only a few MB). This is also the only
    # outbound-network check: they come from a GitHub release host rather than
    # the Hub, and Jobs make no documented guarantee about reaching it.
    import openwakeword.utils
    openwakeword.utils.download_models()
    F = openwakeword.utils.AudioFeatures(
        inference_framework="onnx",
        device="gpu" if on_gpu() else "cpu",
    )
    return f"melspec and embedding models load; provider={F.onnx_execution_provider}"


def download(url, dest):
    """Stream a file to disk. wget is not guaranteed to exist in the uv image."""
    import requests
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def piper_generation():
    """Generate two clips, and check the filenames Job A's speaker split needs.

    Nothing has run Piper in the Jobs image. It may want the espeak-ng program,
    which the Colab notebook installed with apt and this image may lack.

    The filename assertion matters as much as the audio. Job A recovers each
    clip's speaker pair from its index with `divmod`, which only holds while
    generate_samples names files `{index}.wav` in generation order. That mapping
    is what makes the train/validation split speaker-disjoint, so it is worth an
    explicit check rather than an assumption.
    """
    import subprocess
    import tempfile

    import soundfile

    import openwakeword.data

    work = tempfile.mkdtemp()
    repo = os.path.join(work, "piper-sample-generator")
    # The PyPI wheel drops piper_train, which __main__.py imports, so the clone
    # is needed even though the package is also installed.
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", PIPER_REPO, repo], check=True)
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()

    # Only the weights are a release asset. The matching .pt.json config is
    # committed in the repo, so downloading it returns 404.
    model = os.path.join(repo, "models", "en_US-libritts_r-medium.pt")
    assert os.path.isfile(model + ".json"), f"missing config {model}.json in the clone"
    download(PIPER_MODEL_URL, model)

    sys.path.insert(0, repo)
    from piper_sample_generator.__main__ import generate_samples

    clips = os.path.join(work, "clips")
    generate_samples(
        text="hey sonny", output_dir=clips, model=model,
        max_samples=2, batch_size=2, max_speakers=4,
        slerp_weights=[0.5], length_scales=[1.0],
        noise_scales=[0.98], noise_scale_ws=[0.98],
    )

    names = sorted(os.listdir(clips))
    assert names == ["0.wav", "1.wav"], (
        f"expected index-named files, got {names}. Job A's speaker split "
        "depends on this naming.")
    meta = soundfile.info(os.path.join(clips, "0.wav"))
    assert meta.samplerate == 22050, f"expected 22050 Hz from Piper, got {meta.samplerate}"

    texts = openwakeword.data.generate_adversarial_texts(
        input_text="hey sonny", N=8,
        include_partial_phrase=1.0, include_input_words=0.2)
    assert len(texts) >= 4, f"generate_adversarial_texts returned {len(texts)}: {texts}"

    return (f"piper {commit[:8]}: {names} at {meta.samplerate} Hz, "
            f"{meta.frames/meta.samplerate:.2f} s; adversarial e.g. {texts[:3]}")


def patch_torchaudio():
    """Route `torchaudio.load` and `torchaudio.info` through soundfile.

    torchaudio 2.11 delegates `load` to torchcodec, which is absent and would
    drag in system FFmpeg, and it has removed `info` outright. Between them,
    `openwakeword.data` and `torch_audiomentations` call `load` six times and
    `info` five; `augment_clips` reaches both for every room impulse response
    and background clip. Without this the augmentation stage raises
    `ImportError: TorchCodec is required` and then
    `AttributeError: module 'torchaudio' has no attribute 'info'`. Everything
    this pipeline loads through those paths is a plain 16 kHz WAV, which
    soundfile reads completely. `torchaudio.functional`, the only other
    attribute either package touches, is untouched and still present.

    Applied unconditionally rather than only when the call fails: a shim that
    sometimes engages makes the augmentation stage behave differently on
    different torchaudio releases, and the point of a pinned environment is that
    it does not.
    """
    import types

    import soundfile
    import torch
    import torchaudio

    def load(path, *args, **kwargs):
        # torchaudio.load returns (channels, samples) float32; soundfile is
        # (samples, channels), hence the transpose.
        data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sample_rate

    def info(path, *args, **kwargs):
        try:
            meta = soundfile.info(str(path))
        except Exception as exc:
            # get_clip_duration catches this to skip unreadable files, so the
            # exception type matters as much as the failure.
            raise RuntimeError(f"could not read audio metadata from {path}") from exc
        return types.SimpleNamespace(
            sample_rate=meta.samplerate,
            num_frames=meta.frames,
            num_channels=meta.channels,
            bits_per_sample=16,
            encoding=meta.subtype,
        )

    torchaudio.load = load
    torchaudio.info = info
    return load, info


def augmentation_smoke():
    """Rehearse Job A's core path end to end on three synthetic clips.

    `augment_clips` and `compute_features_from_generator` are the two functions
    Job A leans on hardest, and the Colab pilot exercised neither: it mixed with
    `mix_clips_batch` instead. `augment_clips` calls `torchaudio.load` for room
    impulse responses, and torchaudio has already removed one API under this
    project once (`.info`); in 2.11 `load` routes through torchcodec, which is
    why `patch_torchaudio` exists and is exercised here. Both
    augmentation probabilities are forced to 1.0 so the background-mixing and
    reverberation branches actually run rather than being skipped by chance.
    """
    import tempfile

    import numpy as np
    import scipy.io.wavfile

    import openwakeword.data
    import openwakeword.utils

    patch_torchaudio()

    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)

    def wav(name, secs):
        path = os.path.join(d, name)
        x = (rng.normal(0, 0.05, int(16000 * secs)) * 32767).astype(np.int16)
        scipy.io.wavfile.write(path, 16000, x)
        return path

    clips = [wav(f"clip{i}.wav", 0.6) for i in range(3)]
    backgrounds = [wav(f"bg{i}.wav", 4.0) for i in range(2)]
    rirs = [wav("rir0.wav", 0.25)]

    total_length = 32000  # 2 s at 16 kHz -> 16 frames, the settled window
    always = {
        "SevenBandParametricEQ": 1.0, "TanhDistortion": 1.0, "PitchShift": 1.0,
        "BandStopFilter": 1.0, "AddColoredNoise": 1.0, "AddBackgroundNoise": 1.0,
        "Gain": 1.0, "RIR": 1.0,
    }
    generator = openwakeword.data.augment_clips(
        clips, total_length=total_length, batch_size=2,
        augmentation_probabilities=always,
        background_clip_paths=backgrounds, RIR_paths=rirs,
    )

    out = os.path.join(d, "features.npy")
    openwakeword.utils.compute_features_from_generator(
        generator, n_total=len(clips), clip_duration=total_length,
        output_file=out, device="gpu" if on_gpu() else "cpu", ncpu=1,
    )

    features = np.load(out, mmap_mode="r")
    assert features.shape == (len(clips), 16, 96), (
        f"expected {(len(clips), 16, 96)}, got {features.shape}"
    )
    return f"augment_clips -> compute_features_from_generator produced {features.shape}"


def patch_anchors():
    import openwakeword
    root = os.path.dirname(openwakeword.__file__)
    missing = []
    for filename, anchors in PATCH_ANCHORS.items():
        src = open(os.path.join(root, filename)).read()
        missing += [a for a in anchors if a not in src]
    assert not missing, f"anchors absent from the installed source: {missing}"
    return f"{sum(len(v) for v in PATCH_ANCHORS.values())} patch anchors present at {OWW_COMMIT[:8]}"


def upstream_inputs():
    import numpy as np
    import requests
    from huggingface_hub import hf_hub_url

    # Read only the .npy header rather than pulling 185 MB. The stream is
    # capped as well as ranged, so a proxy that drops the Range header costs a
    # few hundred bytes instead of the whole file.
    url = hf_hub_url(FP_REPO, FP_FILE, repo_type="dataset")
    with requests.get(url, headers={"Range": "bytes=0-255"},
                      stream=True, timeout=60) as resp:
        resp.raise_for_status()
        body = resp.raw.read(256)

    buf = io.BytesIO(body)
    version = np.lib.format.read_magic(buf)
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
        (3, 0): np.lib.format.read_array_header_2_0,
    }
    assert version in readers, f"unrecognised .npy format version {version}"
    shape, _, dtype = readers[version](buf)

    assert shape == FP_SHAPE, f"expected {FP_SHAPE}, got {shape} -- val_set_hrs must be recomputed"
    hours = shape[0] * SECONDS_PER_FRAME / 3600
    return f"{FP_FILE} {shape} {dtype} -> {hours:.2f} h (upstream hardcodes 11.3)"


def hub_roundtrip(repo_id: str):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    who = api.whoami()["name"]
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    payload = uuid.uuid4().bytes * 64
    digest = hashlib.sha256(payload).hexdigest()
    path = f"_preflight/{os.environ.get('JOB_ID', 'local')}.bin"

    api.upload_file(path_or_fileobj=payload, path_in_repo=path,
                    repo_id=repo_id, repo_type="dataset")
    local = hf_hub_download(repo_id, path, repo_type="dataset",
                            force_download=True)
    back = hashlib.sha256(open(local, "rb").read()).hexdigest()
    assert back == digest, f"read-back differs: wrote {digest[:12]}, read {back[:12]}"

    api.delete_file(path, repo_id=repo_id, repo_type="dataset")
    return f"as {who}: wrote and re-read {len(payload)} B in {repo_id}, sha256 {digest[:12]}"


def pinned_block():
    from importlib.metadata import version
    lines = [
        "# /// script",
        '# requires-python = "==3.12.*"',
        "# dependencies = [",
        f'#   "openwakeword @ git+https://github.com/dscripka/openWakeWord.git@{OWW_COMMIT}",',
    ]
    for dist in PINNED:
        lines.append(f'#   "{dist}=={version(dist)}",')
    lines += ["# ]", "# ///"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="iamtokarev/hey-sonny-features",
                    help="Dataset repo to round-trip a scratch file through")
    ap.add_argument("--skip-hub", action="store_true",
                    help="Skip the Hub round-trip (it creates the repo if absent)")
    args = ap.parse_args()

    check("runtime")(runtime)
    check("imports")(imports)
    check("accelerator")(accelerator)
    check("shared models and outbound network")(shared_models)
    check("piper generation")(piper_generation)
    check("augmentation smoke")(augmentation_smoke)
    check("patch anchors")(patch_anchors)
    check("upstream inputs")(upstream_inputs)
    if args.skip_hub:
        print("\n--- hub round-trip ---\nSKIP  --skip-hub was passed", flush=True)
    else:
        check("hub round-trip")(lambda: hub_roundtrip(args.repo))

    failed = [n for n, ok, _ in results if not ok]

    print("\n" + "=" * 68)
    print("Paste this into every job script header:\n")
    print(pinned_block())
    print("=" * 68)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
