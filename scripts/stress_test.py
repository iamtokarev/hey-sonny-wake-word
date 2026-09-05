# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openwakeword @ git+https://github.com/dscripka/openWakeWord.git@368c03716d1e92591906a84949bc477f3a834455",
#   "onnxruntime>=1.17",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "tqdm",
#   "requests",
#   "soundfile",
#   "librosa",
#   "huggingface_hub",
# ]
# ///
"""Score clean wake-word clips under controlled interference, one SNR at a time.

    uv run scripts/stress_test.py                          # macOS: builds clips with `say`
    uv run scripts/stress_test.py --clips a.wav b.wav      # your own recordings
    uv run scripts/stress_test.py --music-dir fma --env-dir fsd50k --rir-dir rirs

`metrics.json` reports one recall figure over positives augmented from the same
pool the model trained on, so it cannot say *which* interference the model
fails under. This does: each row mixes the same phrase with one kind of
interference at one SNR and prints the peak score per voice, the minimum across
voices, and which of the measured false-accept budgets that minimum still
clears. Speech babble and stationary noise are built in; music, environmental
sound and room impulse responses are optional directories of 16 kHz WAVs.

The SNR is speech RMS over interference RMS, with a second of the same
interference before and after the phrase, so it is what a room sounds like
rather than a clip with noise pasted under it.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SR = 16000
PAD = SR                       # 1 s of interference either side of the phrase
SNRS = [20, 15, 10, 5, 0, -5]
SAY_VOICES = ["Samantha", "Daniel", "Karen", "Fred", "Moira"]
BABBLE = [
    "The quarterly report is due on Friday and nobody has started the slides yet",
    "I think we should take the train instead of driving because the traffic will be terrible",
    "Did you see the match last night, that second goal was unbelievable",
    "Remember to buy milk, eggs, bread and something for the cat on the way home",
    "The weather forecast says rain all weekend so the picnic is probably cancelled",
]


def say(text: str, voice: str, out: Path) -> Path:
    if shutil.which("say") is None:
        sys.exit("`say` is macOS only -- pass --clips with your own 16 kHz WAV recordings")
    subprocess.run(["say", "-v", voice, "-o", str(out), text], check=True)
    return out


def load(path, trim: bool = False):
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y = y.astype("float32")
    return librosa.effects.trim(y, top_db=40)[0] if trim else y


def rms(x) -> float:
    import numpy as np
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def pool_from_dir(directory: str | None, seconds: int, rng):
    """Concatenate a random sample of a directory's WAVs into one long buffer."""
    import numpy as np
    if not directory:
        return None
    files = sorted(glob.glob(str(Path(directory) / "**" / "*.wav"), recursive=True))
    if not files:
        sys.exit(f"no .wav files under {directory}")
    parts, total = [], 0
    for f in rng.permutation(files):
        y = load(f)
        parts.append(y)
        total += len(y)
        if total >= seconds * SR:
            break
    pool = np.concatenate(parts)
    return pool / rms(pool)


def build_pools(args, rng, tmp: Path) -> dict:
    import numpy as np
    n = 60 * SR
    pools = {}

    babble = np.zeros(n, dtype="float32")
    for i, (text, voice) in enumerate(zip(BABBLE, SAY_VOICES)):
        if args.babble_dir:
            break
        y = load(say(text, voice, tmp / f"b{i}.aiff"))
        reps = int(np.ceil(n / len(y)))
        babble += np.tile(np.roll(y, i * 3 * SR), reps)[:n]    # 5 talkers, offset
    if args.babble_dir or args.speech_dir:
        src = pool_from_dir(args.babble_dir or args.speech_dir, 120, rng)
        babble = np.zeros(n, dtype="float32")
        for i in range(4):                                  # four offset copies of real speech
            start = rng.integers(0, len(src) - n)
            babble += np.roll(src[start:start + n], i * 3 * SR)
    pools["speech babble"] = babble / rms(babble)
    if args.speech_dir:
        pools["speech"] = pool_from_dir(args.speech_dir, 120, rng)

    white = rng.standard_normal(n).astype("float32")
    pools["white noise"] = white / rms(white)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec)); f[0] = 1
    pink = np.fft.irfft(spec / np.sqrt(f)).astype("float32")
    pools["pink noise"] = pink / rms(pink)

    if args.music_dir:
        pools["music"] = pool_from_dir(args.music_dir, 180, rng)
    if args.env_dir:
        pools["environmental"] = pool_from_dir(args.env_dir, 400, rng)
    return pools


def mix(voice, pool, snr, rng):
    import numpy as np
    n = len(voice) + 2 * PAD
    sig = np.zeros(n, dtype="float32")
    sig[PAD:PAD + len(voice)] = voice
    if pool is None or snr == "clean":
        return sig
    start = rng.integers(0, len(pool) - n)
    return sig + pool[start:start + n] * (rms(voice) / 10 ** (snr / 20))


def reverb(voice, rir_path):
    from scipy.signal import fftconvolve
    h = load(rir_path)[:SR]
    y = fftconvolve(voice, h)[:len(voice) + SR // 2]
    return y / rms(y) * rms(voice)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", help="16 kHz WAVs of the phrase; default: `say` voices")
    ap.add_argument("--model", help="local .onnx; default: the promoted model on the Hub")
    ap.add_argument("--repo", default="iamtokarev/hey-sonny")
    ap.add_argument("--music-dir")
    ap.add_argument("--env-dir")
    ap.add_argument("--babble-dir", help="real speech WAVs to use instead of `say` babble")
    ap.add_argument("--speech-dir", help=(
        "real speech WAVs (e.g. a few minutes of your own non-wake speech): adds a "
        "single-talker 'speech' row and, unless --babble-dir is given, builds the "
        "babble from these instead of `say` voices"))
    ap.add_argument("--json", help="write every row to this file, same shape as metrics.json['stress']")
    ap.add_argument("--rir-dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import openwakeword
    import openwakeword.utils

    rng = np.random.default_rng(args.seed)
    tmp = Path(tempfile.mkdtemp(prefix="stress_"))

    if args.clips:
        voices = {Path(p).stem[:8]: load(p, trim=True) for p in args.clips}
    else:
        voices = {v[:8]: load(say("hey sonny", v, tmp / f"{v}.aiff"), trim=True) for v in SAY_VOICES}

    if args.model:
        onnx, thresholds = args.model, {}
    else:
        from huggingface_hub import hf_hub_download
        onnx = hf_hub_download(args.repo, "hey_sonny.onnx")
        metrics = json.load(open(hf_hub_download(args.repo, "metrics.json")))
        thresholds = {b: v["threshold"] for b, v in metrics["recall_at_fa_exact"].items()}
    openwakeword.utils.download_models(model_names=["hey_jarvis_v0.1"])
    oww = openwakeword.Model(wakeword_models=[onnx], inference_framework="onnx")
    name = Path(onnx).stem

    def score(clip) -> float:
        oww.reset()
        clip = clip / max(1.0, np.abs(clip).max())
        return max(d[name] for d in oww.predict_clip((clip * 32767).astype(np.int16), padding=1))

    if thresholds:
        print("thresholds from metrics.json: " +
              "  ".join(f"FA{b}/h={t:.4f}" for b, t in thresholds.items()))
    head = f"{'cond':>10} | " + " ".join(f"{v:>8}" for v in voices) + " |   min  | clears budget"
    results: dict = {}

    def table(title, rows):
        print(f"\n=== {title} ===\n{head}")
        for label, clips in rows:
            scores = [score(c) for c in clips]
            lo = min(scores)
            ok = [b for b, t in thresholds.items() if lo >= t]
            print(f"{str(label):>10} | " + " ".join(f"{s:8.4f}" for s in scores) +
                  f" | {lo:6.4f} | {', '.join(ok) or '-'}")
            key = "clean" if title == "clean" else f"{title.split(',')[0]}@{label}"
            results[key] = {"min": round(lo, 4), "median": round(float(np.median(scores)), 4),
                            "scores": [round(s, 4) for s in scores],
                            "clear_frac": {b: round(float(np.mean([s > t for s in scores])), 3)
                                           for b, t in thresholds.items()}}

    table("clean", [("clean", [mix(v, None, "clean", rng) for v in voices.values()])])
    for pname, pool in build_pools(args, rng, tmp).items():
        table(f"{pname}, SNR dB", [(s, [mix(v, pool, s, rng) for v in voices.values()]) for s in SNRS])

    if args.rir_dir:
        rirs = sorted(glob.glob(str(Path(args.rir_dir) / "**" / "*.wav"), recursive=True))
        picks = rng.choice(rirs, min(8, len(rirs)), replace=False)
        table("reverb only", [(Path(p).stem[:10], [mix(reverb(v, p), None, "clean", rng)
                                                    for v in voices.values()]) for p in picks])

    print("\n=== 60 s of each interference alone: peak score, frames above each budget ===")
    for pname, pool in build_pools(args, rng, tmp).items():
        oww.reset()
        clip = pool[:60 * SR] / np.abs(pool[:60 * SR]).max() * 0.5
        sc = np.array([d[name] for d in oww.predict_clip((clip * 32767).astype(np.int16), padding=1)])
        print(f"{pname:>14}: peak {sc.max():.4f}  " +
              "  ".join(f"FA{b}:{int((sc >= t).sum())}" for b, t in thresholds.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"model": str(onnx), "thresholds": thresholds, "voices": list(voices), "rows": results},
            indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
