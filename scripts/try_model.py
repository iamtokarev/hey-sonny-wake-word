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
#   "sounddevice",
# ]
# ///
"""Try the Hey Sonny wake-word model from a terminal.

    uv run scripts/try_model.py                    # listen on the microphone
    uv run scripts/try_model.py --wav clip.wav     # score an audio file
    uv run scripts/try_model.py --wav noise.wav --negative   # measure false accepts
    uv run scripts/try_model.py --list-devices

Nothing needs installing first: `uv` resolves the dependencies from the header
above. The model is pulled from the Hugging Face Hub on first run and cached, so
you need to be logged in (`hf auth login`) while the repo is private.

Activations are counted with the project's `refractory-50` rule -- one
activation, then four seconds of silence before another can be counted -- so the
numbers here mean the same thing as the ones in the model's `metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SR = 16000
FRAME = 1280            # 80 ms, the rate openWakeWord scores at
REFRACTORY_SECONDS = 4.0
REPO = "iamtokarev/hey-sonny"


def load_model(args):
    """Fetch the model and its measured thresholds, then build the detector."""
    import openwakeword
    import openwakeword.utils

    if args.model:
        onnx_path, metrics = Path(args.model), None
    else:
        from huggingface_hub import hf_hub_download
        try:
            onnx_path = Path(hf_hub_download(args.repo, "hey_sonny.onnx"))
            metrics = json.load(open(hf_hub_download(args.repo, "metrics.json")))
        except Exception as exc:
            sys.exit(f"Could not fetch {args.repo}: {exc}\n"
                     f"The repo is private -- run `hf auth login`, or pass "
                     f"--model /path/to/hey_sonny.onnx")

    # Pulls the shared melspectrogram and embedding models, which are not
    # bundled in the wheel. Naming one wake word keeps it from downloading the
    # whole pretrained collection.
    openwakeword.utils.download_models(model_names=["hey_jarvis_v0.1"])

    detector = openwakeword.Model(wakeword_models=[str(onnx_path)],
                                  inference_framework="onnx")
    key = next(iter(detector.models))
    return detector, key, metrics, onnx_path


def resolve_threshold(args, metrics):
    """Use the threshold measured for the requested false-accept budget."""
    if args.threshold is not None:
        return args.threshold, "given on the command line"
    if not metrics:
        return 0.5, "default (no metrics.json alongside the model)"
    exact = metrics.get("recall_at_fa_exact") or {}
    entry = exact.get(str(args.fa_budget))
    if not entry or entry.get("threshold") is None:
        return 0.5, "default (that budget is not reachable on this model)"
    corpus = metrics.get("primary_set", "?")
    recall = entry.get("recall")
    return float(entry["threshold"]), (
        f"measured for <= {args.fa_budget} false accepts/hour on the '{corpus}' "
        f"set, where it gave {recall:.3f} recall")


class Activations:
    """Count activations with the refractory rule the metrics use."""

    def __init__(self, threshold: float, refractory_frames: int):
        self.threshold = threshold
        self.refractory = refractory_frames
        self.count = 0
        self._blocked_until = -1

    def feed(self, frame_index: int, score: float) -> bool:
        if score < self.threshold or frame_index < self._blocked_until:
            return False
        self.count += 1
        self._blocked_until = frame_index + self.refractory + 1
        return True


def bar(score: float, width: int = 30) -> str:
    filled = int(round(score * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def run_file(args, detector, key, threshold, refractory):
    import librosa
    import numpy as np

    path = Path(args.wav)
    if not path.is_file():
        sys.exit(f"no such file: {path}")
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    pcm = (audio * 32767).astype(np.int16)
    duration = len(pcm) / SR
    print(f"scoring {path.name}: {duration:.1f} s\n")

    acts = Activations(threshold, refractory)
    scores = []
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        score = float(detector.predict(pcm[i:i + FRAME])[key])
        scores.append(score)
        acts.feed(len(scores) - 1, score)

    if not scores:
        sys.exit("file is shorter than one 80 ms frame")

    # Every contiguous run above the threshold, so a file with several
    # utterances shows all of them. The grouped count below is the one that
    # matches metrics.json -- it suppresses anything within four seconds of a
    # previous hit, which is right for a false-accept rate and wrong for
    # checking whether your voice is detected.
    runs, start = [], None
    for i, sc in enumerate(scores):
        if sc >= threshold and start is None:
            start = i
        elif sc < threshold and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(scores) - 1))

    for lo, hi in runs:
        peak_i = max(range(lo, hi + 1), key=lambda j: scores[j])
        print(f"  DETECTED at {lo * FRAME / SR:6.2f}-{(hi + 1) * FRAME / SR:5.2f} s   "
              f"peak {scores[peak_i]:.4f} at {peak_i * FRAME / SR:.2f} s")

    peak = max(scores)
    print(f"\n  frames scored     {len(scores)}")
    print(f"  peak score        {peak:.4f}   (threshold {threshold:.4f})")
    print(f"  detections        {len(runs)}")
    print(f"  activations       {acts.count}   "
          f"(after {REFRACTORY_SECONDS:.0f} s grouping — the metrics.json rule)")
    if args.negative:
        hours = duration / 3600
        print(f"  false accepts/h   {acts.count / hours:.2f}   "
              f"over {hours:.4f} h")
        print("\n  Note: a few minutes of audio cannot measure a rate near "
              "0.2/hour.\n  One activation here already reads as "
              f"{1 / hours:.1f}/hour.")
    elif not runs:
        print("\n  No activation. If you expected one, try --fa-budget 2.0 for a "
              "lower\n  threshold, or check the recording is 16 kHz mono speech.")


def run_mic(args, detector, key, threshold, refractory):
    import numpy as np
    try:
        import sounddevice as sd
    except Exception as exc:
        sys.exit(f"microphone support needs the sounddevice package: {exc}")

    print(f"listening on {sd.query_devices(args.device, 'input')['name']!r} — "
          f"say \"hey sonny\".  Ctrl-C to stop.\n")
    acts = Activations(threshold, refractory)
    frame_index = 0
    recent = 0.0
    started = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal frame_index, recent
        if status:
            print(f"  (audio status: {status})", file=sys.stderr)
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        score = float(detector.predict(pcm)[key])
        recent = max(recent * 0.7, score)
        if acts.feed(frame_index, score):
            print(f"\r  ** HEY SONNY ** at {time.time() - started:6.1f} s   "
                  f"score {score:.4f}                    ")
        frame_index += 1

    try:
        with sd.InputStream(samplerate=SR, blocksize=FRAME, channels=1,
                            dtype="float32", device=args.device,
                            callback=callback):
            while True:
                print(f"\r  {bar(recent)} {recent:.3f}  "
                      f"activations: {acts.count}   ", end="", flush=True)
                time.sleep(0.08)
    except KeyboardInterrupt:
        elapsed = (time.time() - started) / 3600
        print(f"\n\nstopped after {elapsed * 3600:.0f} s — "
              f"{acts.count} activation(s)"
              + (f", {acts.count / elapsed:.1f}/hour" if elapsed > 0 else ""))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="Score this audio file instead of the microphone")
    ap.add_argument("--negative", action="store_true",
                    help="With --wav: the file contains no wake word, so report a false-accept rate")
    ap.add_argument("--repo", default=REPO, help="Hub model repo")
    ap.add_argument("--model", help="Use a local .onnx instead of downloading")
    ap.add_argument("--fa-budget", default="0.2", choices=("0.2", "0.5", "1.0", "2.0"),
                    help="Pick the threshold measured for this false-accepts-per-hour budget")
    ap.add_argument("--threshold", type=float, help="Override the threshold outright")
    ap.add_argument("--device", help="Input device name or index")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)

    detector, key, metrics, onnx_path = load_model(args)
    threshold, why = resolve_threshold(args, metrics)
    print(f"model      {onnx_path.name}  (wake word key: {key!r})")
    print(f"threshold  {threshold:.4f}  — {why}")
    print(f"grouping   one activation, then {REFRACTORY_SECONDS:.0f} s before another counts\n")

    refractory = int(REFRACTORY_SECONDS * SR / FRAME)
    if args.wav:
        run_file(args, detector, key, threshold, refractory)
    else:
        run_mic(args, detector, key, threshold, refractory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
