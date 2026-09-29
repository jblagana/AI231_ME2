"""Entropy probe: is the 2-class collapse a strong prior or an argmax of a
flat distribution?

Feeds SLURP clips through the TTS-trained model and checks the softmax
outputs:
  - Low entropy (confident) on the 2-class collapse -> the model has learned
    a strong TTS-specific prior toward play_music / set_reminder.
  - High entropy (uncertain) on the collapse -> the model "doesn't know"
    (flat distribution), and the 2-class collapse is just the argmax of
    noise. Less concerning — the model isn't "knowing wrong", it's
    "not knowing".

Also reports the per-class prediction distribution on SLURP (to confirm the
2-class collapse) and the mean entropy per class.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from model import VCM  # noqa: E402
from train import wav_to_logmel  # noqa: E402
from gen_site_assets import load_mp3  # noqa: E402
from commands import CLASSES  # noqa: E402

MANIFEST = REPO / "data" / "eval_slurp" / "manifest.csv"
AUDIO_DIR = REPO / "data" / "eval_slurp" / "audio" / "slurp_real"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(MANIFEST.open()))
    if args.limit:
        rows = rows[: args.limit]

    m = VCM(len(CLASSES))
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m.eval()

    pred_counts = Counter()
    entropies = []
    correct_entropies = []
    wrong_entropies = []
    n = 0
    for r in rows:
        p = AUDIO_DIR / r["file"]
        if not p.exists():
            continue
        wav = load_mp3(p)
        x = wav_to_logmel(wav).unsqueeze(0)
        with torch.no_grad():
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
            # entropy of the softmax distribution
            ent = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(ent)
        pred_counts[CLASSES[pred]] += 1
        if pred == CLASSES.index(r["class"]):
            correct_entropies.append(ent)
        else:
            wrong_entropies.append(ent)
        n += 1

    print(f"\n== SLURP entropy probe: {n} clips ==")
    print(f"\nprediction distribution (confirming the 2-class collapse):")
    for cls, k in pred_counts.most_common():
        print(f"  {cls:16s} {k:3d}  ({k/n:.1%})")

    print(f"\nentropy (nats):")
    print(f"  all clips  : mean={sum(entropies)/len(entropies):.3f}  "
          f"median={sorted(entropies)[len(entropies)//2]:.3f}")
    if correct_entropies:
        print(f"  correct    : mean={sum(correct_entropies)/len(correct_entropies):.3f}  "
              f"(n={len(correct_entropies)})")
    if wrong_entropies:
        print(f"  wrong      : mean={sum(wrong_entropies)/len(wrong_entropies):.3f}  "
              f"(n={len(wrong_entropies)})")

    # max possible entropy for 10 classes = ln(10) ≈ 2.303
    max_ent = torch.log(torch.tensor(float(len(CLASSES)))).item()
    mean_ent = sum(entropies) / len(entropies)
    print(f"\n  max possible entropy (10 classes): {max_ent:.3f}")
    print(f"  mean entropy / max: {mean_ent/max_ent:.3f}  "
          f"({'confident' if mean_ent/max_ent < 0.5 else 'uncertain' if mean_ent/max_ent < 0.8 else 'very uncertain'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
