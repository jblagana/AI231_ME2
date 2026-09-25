"""Train VCM v1 on the generated dataset.

Usage:
  python src/train.py --epochs 15 --batch 32 --out runs/v1
  python src/train.py --smoke     # 2 batches, 1 epoch, tiny — pipeline check

Data: data/raw/{split}/{class}/*.mp3 + manifest.jsonl (speaker-disjoint splits).
Augmentation (train only): speed jitter 0.95-1.05, random gain ±6 dB,
SNR 5-25 dB white noise (placeholder for real background-noise dataset).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES  # noqa: E402
from model import VCM, N_MELS, N_FRAMES, SR  # noqa: E402


def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """(1, T) @16k -> (1, 80, 50) log-mel, padded/cropped to fixed length."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[-1] > SR:  # crop to 1 s
        start = random.randint(0, wav.shape[-1] - SR)
        wav = wav[..., start:start + SR]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=320, n_mels=N_MELS,
        f_min=50.0, f_max=8000.0,
    )(wav)
    logmel = torch.clamp(mel, min=1e-5).log10()
    # normalize to ~[0,1] (log10 range is ~[-5, 0])
    logmel = (logmel + 5.0) / 5.0
    if logmel.shape[-1] < N_FRAMES:
        logmel = nn.functional.pad(logmel, (0, N_FRAMES - logmel.shape[-1]))
    return logmel[..., :N_FRAMES]


def load_wav(p: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(p))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav


class VCMDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = root
        self.augment = augment
        self.items = []
        mf = root / "manifest.jsonl"
        if mf.exists():
            for line in mf.open(encoding="utf-8"):
                row = json.loads(line)
                if row["split"] == split:
                    self.items.append((root / row["file"], row["class"]))
        else:
            for cls in CLASSES:
                d = root / split / cls
                if d.exists():
                    for f in sorted(d.glob("*.mp3")):
                        self.items.append((f, cls))
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, cls = self.items[i]
        wav = load_wav(p)
        if self.augment:
            # speed jitter
            rate = random.uniform(0.95, 1.05)
            wav = torchaudio.functional.speed(wav, SR, SR, rate)
            # random gain
            gain = random.uniform(-6.0, 6.0) / 20.0 * 10
            wav = wav * (10 ** (gain / 20.0))
            # white-noise floor (SNR 5-25 dB)
            snr_db = random.uniform(5.0, 25.0)
            sig_p = (wav ** 2).mean().clamp_min(1e-8)
            noise = torch.randn_like(wav)
            noise_p = (noise ** 2).mean().clamp_min(1e-8)
            scale = torch.sqrt(sig_p / (noise_p * 10 ** (snr_db / 10.0)))
            wav = wav + noise * scale
        x = wav_to_logmel(wav)
        return x, CLASSES.index(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    train_ds = VCMDataset(root, "train", augment=True)
    eval_ds = VCMDataset(root, "eval", augment=False)
    if args.smoke:
        train_ds.items = train_ds.items[:64]
        eval_ds.items = eval_ds.items[:32]
        args.epochs = 1
    print(f"train={len(train_ds)}  eval={len(eval_ds)}")
    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("NO DATA — run src/make_dataset.py first"); return 2

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    va = DataLoader(eval_ds, batch_size=args.batch, num_workers=0)

    m = VCM(len(CLASSES))
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(1, args.epochs + 1):
        m.train()
        t0 = time.time()
        tot = cnt = 0
        for x, y in tr:
            opt.zero_grad()
            loss = lossf(m(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            cnt += len(y)
        sched.step()
        # eval
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in va:
                pred = m(x).argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        history.append({"epoch": ep, "train_loss": tot / max(cnt, 1),
                        "eval_acc": acc, "sec": time.time() - t0})
        print(f"ep {ep:2d}  loss={tot/max(cnt,1):.4f}  eval_acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    torch.save(m.state_dict(), out / "vcm_v1.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved -> {out/'vcm_v1.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
