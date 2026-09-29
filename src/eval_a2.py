"""A2 — evaluate the two-head VCM checkpoint.

Reports:
  1. command accuracy (vs v1i baseline 0.8618) + per-class breakdown
  2. slot accuracy (parametric clips with a slot value) + per-class breakdown
  3. command confusion matrix (top confusions)

Usage:
  python src/eval_a2.py --ckpt runs/v1a2/vcm_a2_best.pt --data data/raw_v1i
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from slots import extract_slot, PARAMETRIC, SLOT_VOCAB  # noqa: E402
from model_a2 import VCMTwoHead  # noqa: E402
from train_a2 import (VCMDatasetA2, CLASSES, preload_raw,  # noqa: E402
                      precompute_base_features, N_MELS, SR)
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v1a2/vcm_a2_best.pt")
    ap.add_argument("--data", default="data/raw_v1i")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt}"); return 2
    root = Path(args.data)

    ds = VCMDatasetA2(root, args.split, augment=False)
    print(f"eval split={args.split}  n={len(ds)}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    # GPU precompute of base features (3.0 s mel is ~8x the 1.0 s work;
    # ~354 ms/item on CPU -> ~135 min for 11.4k; ~2-5 ms/item on the A100).
    preload_raw(root)
    from torchaudio.transforms import MelSpectrogram
    gpu_mel = MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=320,
                             n_mels=N_MELS, f_min=50.0, f_max=8000.0).to(dev)
    precompute_base_features(ds, dev, gpu_mel)
    va = DataLoader(ds, batch_size=args.batch, num_workers=0)

    m = VCMTwoHead(len(CLASSES)).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()

    cmd_correct = cmd_total = 0
    slot_correct = slot_total = 0
    per_cls = defaultdict(lambda: [0, 0])      # cls -> [correct, total]
    slot_per_cls = defaultdict(lambda: [0, 0])
    confusion = Counter()
    slot_confusion = defaultdict(Counter)

    t0 = time.time()
    with torch.no_grad():
        for x, y_cmd, y_slot in va:
            x, y_cmd, y_slot = x.to(dev), y_cmd.to(dev), y_slot.to(dev)
            cmd_logits, slot_feat = m(x)
            pred = cmd_logits.argmax(1)
            cmd_correct += (pred == y_cmd).sum().item()
            cmd_total += len(y_cmd)
            for cls in PARAMETRIC:
                ci = CLASSES.index(cls)
                mask = (y_cmd == ci) & (y_slot >= 0)
                if mask.any():
                    sl = m.slot_logits(slot_feat[mask], cls)
                    sp = sl.argmax(1)
                    ys = y_slot[mask]
                    slot_correct += (sp == ys).sum().item()
                    slot_total += int(mask.sum().item())
                    slot_per_cls[cls][1] += int(mask.sum().item())
                    slot_per_cls[cls][0] += int((sp == ys).sum().item())
                    sp = sp.cpu()
                    ys = ys.cpu()
                    for i in range(len(ys)):
                        if sp[i] != ys[i]:
                            slot_confusion[cls][(SLOT_VOCAB[cls][ys[i]],
                                                SLOT_VOCAB[cls][sp[i]])] += 1
            pred = pred.cpu()
            y_cmd = y_cmd.cpu()
            for i in range(len(y_cmd)):
                t = CLASSES[y_cmd[i]]
                p = CLASSES[pred[i]]
                per_cls[t][1] += 1
                if t == p:
                    per_cls[t][0] += 1
                else:
                    confusion[(t, p)] += 1
    dt = time.time() - t0

    cmd_acc = cmd_correct / max(cmd_total, 1)
    slot_acc = slot_correct / max(slot_total, 1)
    print(f"\n=== A2 EVAL ({args.split}) ===")
    print(f"command acc: {cmd_acc:.4f}  ({cmd_correct}/{cmd_total})   "
          f"[v1i baseline: 0.8618]")
    print(f"slot acc:    {slot_acc:.4f}  ({slot_correct}/{slot_total})   "
          f"(parametric clips with a slot)")
    print(f"eval time: {dt:.0f}s")

    print("\nper-class command acc:")
    for cls in CLASSES:
        c, t = per_cls[cls]
        print(f"  {cls:18s} {c/t if t else 0:.3f}  ({c}/{t})")

    print("\nper-class slot acc:")
    for cls in PARAMETRIC:
        c, t = slot_per_cls[cls]
        print(f"  {cls:18s} {c/t if t else 0:.3f}  ({c}/{t})")

    print("\ntop command confusions (true -> pred):")
    for (t, p), n in confusion.most_common(10):
        print(f"  {t:18s} -> {p:18s}  {n}")

    print("\ntop slot confusions (class: true -> pred):")
    for cls in PARAMETRIC:
        if slot_confusion[cls]:
            top = slot_confusion[cls].most_common(3)
            for (t, p), n in top:
                print(f"  {cls}: {t!r} -> {p!r}  {n}")

    # save report
    out = ckpt.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split, "n": cmd_total,
        "cmd_acc": cmd_acc, "slot_acc": slot_acc,
        "per_class_cmd": {k: {"acc": v[0]/v[1] if v[1] else 0,
                              "n": v[1]} for k, v in per_cls.items()},
        "per_class_slot": {k: {"acc": v[0]/v[1] if v[1] else 0,
                               "n": v[1]} for k, v in slot_per_cls.items()},
        "top_confusions": {f"{t}->{p}": n for (t, p), n in confusion.most_common(20)},
    }, indent=1))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
