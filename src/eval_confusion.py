"""Per-class confusion breakdown for a trained VCM checkpoint.

Runs the model over the eval split, builds the confusion matrix, and reports
per-class precision/recall/F1 + the top confusion pairs. No noise (clean eval).

Usage:
  python _eval_confusion.py --data data/raw_v1f --ckpt runs/v1f/vcm_v1.pt
  python _eval_confusion.py --data data/raw_v1f --ckpt runs/v1f/vcm_v1.pt --split train
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from commands import CLASSES  # noqa: E402
from model import VCM, N_MELS, N_FRAMES  # noqa: E402
import train as T  # noqa: E402  (reuse wav_to_logmel + load_wav + RAW_CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw_v1f")
    ap.add_argument("--ckpt", default="runs/v1f/vcm_v1.pt")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pool", default="max",
                    choices=["max", "fw_mean", "fw_max"])
    args = ap.parse_args()

    data = Path(args.data)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # accept either a raw state_dict or a {"model": state_dict} wrapper
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model = VCM(len(CLASSES), pool=args.pool)
    missing = model.load_state_dict(sd, strict=False)
    if missing.missing_keys:
        print(f"WARN missing keys: {missing.missing_keys}", file=sys.stderr)
    model.eval()

    # build items from manifest (split + class)
    items = []
    for line in (data / "manifest.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == args.split:
            items.append((data / row["file"], row["class"]))
    print(f"eval items ({args.split}): {len(items)}", flush=True)

    # JuiceFS: per-file np.fromfile is ~100 ms each even cached — preload the
    # eval .raw into RAM (same fix that unblocked training throughput).
    T.preload_raw(data)

    idx = {c: i for i, c in enumerate(CLASSES)}
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=np.int64)  # cm[true][pred]
    wrong = 0
    correct = 0

    def feat(p):
        wav = T.load_wav(p)
        return T.wav_to_logmel(wav)

    with torch.no_grad():
        for s in range(0, len(items), args.batch):
            chunk = items[s:s + args.batch]
            X = torch.stack([feat(p) for p, _ in chunk])
            logits = model(X)
            pred = logits.argmax(1).tolist()
            for (p, cls), pr in zip(chunk, pred):
                t = idx[cls]
                cm[t][pr] += 1
                if pr == t:
                    correct += 1
                else:
                    wrong += 1

    total = correct + wrong
    acc = correct / total if total else 0.0
    print(f"\nACCURACY ({args.split}): {correct}/{total} = {acc:.4f}\n")

    # per-class metrics
    print(f"{'class':<18}{'sup':>6}{'prec':>8}{'rec':>8}{'f1':>8}  top-confused-with")
    tp = cm.diagonal()
    for i, c in enumerate(CLASSES):
        pred_i = cm[:, i].sum()  # predicted as i
        prec = tp[i] / pred_i if pred_i else 0.0
        rec = tp[i] / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sup = cm[i].sum()
        # top confusions: where this class's samples went (off-diagonal, row i)
        row = cm[i].copy()
        row[i] = 0
        top = sorted(zip(row.tolist(), range(n)), reverse=True)[:2]
        topstr = ", ".join(f"{CLASSES[j]}({k})" for k, j in top if k > 0) or "-"
        print(f"{c:<18}{sup:>6}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}  {topstr}")

    # global top confusion pairs (off-diagonal, both directions)
    print("\nTop confusion pairs (true -> pred, count):")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((int(cm[i][j]), CLASSES[i], CLASSES[j]))
    pairs.sort(reverse=True)
    for cnt, a, b in pairs[:12]:
        print(f"  {a} -> {b}: {cnt}")

    # save matrix
    out = Path(args.ckpt).parent / f"confusion_{args.split}.json"
    out.write_text(json.dumps({
        "classes": CLASSES,
        "accuracy": acc,
        "total": total,
        "matrix": cm.tolist(),  # [true][pred]
        "top_pairs": [[a, b, c] for c, a, b in pairs[:12]],
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
