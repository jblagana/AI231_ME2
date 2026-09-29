"""Evaluate the VCM on the REAL-HUMAN SLURP eval set (cross-domain probe).

Feeds the SLURP clips (real voices, real rooms) through the TTS-trained model
and reports the cross-domain accuracy — the number that tells us whether the
model actually holds up on real people (Sir's ruling), not just on edge-tts.

Prereq: run `src/make_slurp_eval.py` first (builds data/eval_slurp/manifest.csv),
and the SLURP real audio must be extracted under `data/eval_slurp/audio/`
(see the extract step in the run notes — only the ~504 FLACs in needed_flac.txt).

Usage
-----
    python src/eval_slurp.py --ckpt runs/v1e/vcm_v1.pt
    python src/eval_slurp.py --ckpt runs/v1f/vcm_v1.pt --limit 20   # smoke
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from model import VCM, N_MELS, N_FRAMES  # noqa: E402
from train import wav_to_logmel          # exact training feature path  # noqa: E402
from gen_site_assets import load_mp3     # ffmpeg decode -> (1,T)@16k mono  # noqa: E402
from commands import CLASSES             # 10-class ordering the ckpt expects  # noqa: E402

MANIFEST = REPO / "data" / "eval_slurp" / "manifest.csv"
AUDIO_DIR = REPO / "data" / "eval_slurp" / "audio"


def load_flac(p: Path) -> torch.Tensor:
    """Decode a SLURP FLAC -> (1, T) @16k mono. Reuses the ffmpeg path that
    load_mp3 already uses (the binary handles any input format)."""
    return load_mp3(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="VCM checkpoint (state_dict)")
    ap.add_argument("--limit", type=int, default=0, help="cap clips (0=all); smoke")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"no manifest — run `python src/make_slurp_eval.py` first ({MANIFEST})")
        return 2

    rows = list(csv.DictReader(MANIFEST.open()))
    if args.limit:
        rows = rows[: args.limit]

    # Model: 10 classes (the ckpt's ordering), CPU, eval mode.
    m = VCM(len(CLASSES))
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m.eval()
    print(f"loaded {args.ckpt}  ({sum(p.numel() for p in m.parameters()):,} params)")

    correct = 0
    per_class = defaultdict(lambda: [0, 0])  # class -> [correct, total]
    confusion = Counter()  # (true, pred) for errors
    missing = 0
    for r in rows:
        flac = r["file"]
        cls = r["class"]
        ap_ = AUDIO_DIR / flac
        if not ap_.exists():
            missing += 1
            continue
        wav = load_flac(ap_)
        x = wav_to_logmel(wav).unsqueeze(0)  # (1,1,80,50)
        with torch.no_grad():
            pred = m(x).argmax(1).item()
        per_class[cls][1] += 1
        if pred == CLASSES.index(cls):
            correct += 1
            per_class[cls][0] += 1
        else:
            confusion[(cls, CLASSES[pred])] += 1

    n = sum(v[1] for v in per_class.values())
    print(f"\n== SLURP cross-domain eval: {n} clips ({missing} missing) ==")
    print(f"overall accuracy: {correct}/{n} = {correct/n:.3f}" if n else "no clips evaluated")
    print("\nper-class (real-human):")
    for c in CLASSES:
        if c in per_class:
            cc, tt = per_class[c]
            print(f"  {c:16s} {cc:3d}/{tt:3d} = {cc/tt:.3f}")
    if confusion:
        print("\ntop cross-domain errors (true -> pred):")
        for (t, p), k in confusion.most_common(12):
            print(f"  {t:16s} -> {p:16s} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
