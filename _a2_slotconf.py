"""A2 — per-class slot confusion matrices + wrong-clip sampler.

Runs the A2 checkpoint over the eval split and writes:
  runs/v1a2/slot_confusion.json   full per-class slot confusion (true->pred: n)
  runs/v1a2/wrong_clips.json      sampled wrong clips (slot + command) with
                                  path, phrase, true, pred, confidence

Usage (on n002):
  python _a2_slotconf.py --ckpt runs/v1a2/vcm_a2_best.pt --data data/raw_v1i
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))
from slots import extract_slot, PARAMETRIC, SLOT_VOCAB  # noqa: E402
from model_a2 import VCMTwoHead  # noqa: E402
from train_a2 import (VCMDatasetA2, CLASSES, preload_raw,  # noqa: E402
                      precompute_base_features, N_MELS, SR)
from torchaudio.transforms import MelSpectrogram  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v1a2/vcm_a2_best.pt")
    ap.add_argument("--data", default="data/raw_v1i")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--per-class-slot", type=int, default=4,
                    help="wrong clips to sample per parametric class")
    ap.add_argument("--per-cmd-pair", type=int, default=3,
                    help="wrong clips to sample per top command confusion pair")
    args = ap.parse_args()

    root = Path(args.data)
    ckpt = Path(args.ckpt)
    ds = VCMDatasetA2(root, args.split, augment=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"eval n={len(ds)}  device={dev}", flush=True)
    preload_raw(root)
    gpu_mel = MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=320,
                             n_mels=N_MELS, f_min=50.0, f_max=8000.0).to(dev)
    precompute_base_features(ds, dev, gpu_mel)
    va = DataLoader(ds, batch_size=args.batch, num_workers=0)

    m = VCMTwoHead(len(CLASSES)).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    m.eval()

    # full slot confusion: cls -> true_idx -> pred_idx -> count
    slot_full = {c: defaultdict(Counter) for c in PARAMETRIC}
    # wrong-clip pools: cls -> list of (true_idx, pred_idx, conf, file, phrase)
    slot_wrong = defaultdict(list)
    # command wrong pools: (true,pred) -> list of (conf, file, phrase)
    cmd_wrong = defaultdict(list)
    # command full confusion
    cmd_full = defaultdict(Counter)

    with torch.no_grad():
        for off, (x, y_cmd, y_slot) in enumerate(va):
            x, y_cmd, y_slot = x.to(dev), y_cmd.to(dev), y_slot.to(dev)
            base = off * args.batch
            cmd_logits, slot_feat = m(x)
            pred = cmd_logits.argmax(1)
            cmd_conf = torch.softmax(cmd_logits, 1).max(1).values
            for cls in PARAMETRIC:
                ci = CLASSES.index(cls)
                mask = (y_cmd == ci) & (y_slot >= 0)
                if not mask.any():
                    continue
                sl = m.slot_logits(slot_feat[mask], cls)
                sp = sl.argmax(1)
                sconf = torch.softmax(sl, 1).max(1).values
                ys = y_slot[mask]
                idx = mask.nonzero(as_tuple=True)[0]
                for i in range(len(ys)):
                    t, p, f = int(ys[i]), int(sp[i]), float(sconf[i])
                    slot_full[cls][t][p] += 1
                    if t != p:
                        it = ds.items[base + int(idx[i])]  # (path, cls, phrase, slot)
                        slot_wrong[cls].append((t, p, f, it))
            pred = pred.cpu(); y_cmd = y_cmd.cpu(); cmd_conf = cmd_conf.cpu()
            for i in range(len(y_cmd)):
                t, p = int(y_cmd[i]), int(pred[i])
                cmd_full[t][p] += 1
                if t != p:
                    it = ds.items[base + i]
                    cmd_wrong[(t, p)].append((float(cmd_conf[i]), it))

    # ---- slot confusion json (true label -> {pred label: n}) ----
    out_slot = {}
    for cls in PARAMETRIC:
        vocab = SLOT_VOCAB[cls]
        out_slot[cls] = {}
        for t in range(len(vocab)):
            preds = slot_full[cls].get(t, {})
            if not preds:
                continue
            out_slot[cls][vocab[t]] = {vocab[p]: n for p, n in sorted(
                preds.items(), key=lambda kv: -kv[1])}
    (ckpt.parent / "slot_confusion.json").write_text(
        json.dumps(out_slot, indent=1))
    print(f"wrote {ckpt.parent / 'slot_confusion.json'}")

    # ---- sample wrong clips ----
    samples = []
    for cls in PARAMETRIC:
        pool = slot_wrong[cls]
        # sort by confidence desc (most confident wrong first) — but also keep
        # a spread: take top-2 most confident + 2 random-ish (lowest conf)
        pool_sorted = sorted(pool, key=lambda r: -r[2])
        picks = pool_sorted[:2] + pool_sorted[-2:][::-1]
        seen = set()
        for t, p, f, row in picks:
            if row is None or id(row) in seen:
                continue
            seen.add(id(row))
            samples.append({
                "kind": "slot", "cls": cls,
                "true": SLOT_VOCAB[cls][t], "pred": SLOT_VOCAB[cls][p],
                "slot_conf": round(f, 3),
                "file": str(row[0]),
                "phrase": row[2],
            })
    # top command confusion pairs
    top_pairs = sorted(cmd_wrong.items(),
                       key=lambda kv: -len(kv[1]))[:6]
    for (t, p), pool in top_pairs:
        pool_sorted = sorted(pool, key=lambda r: -r[0])[:args.per_cmd_pair]
        for f, row in pool_sorted:
            samples.append({
                "kind": "cmd",
                "true": CLASSES[t], "pred": CLASSES[p],
                "cmd_conf": round(f, 3),
                "file": str(row[0]),
                "phrase": row[2],
            })
    (ckpt.parent / "wrong_clips.json").write_text(
        json.dumps(samples, indent=1))
    print(f"wrote {ckpt.parent / 'wrong_clips.json'}  ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
