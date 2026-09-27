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
from gen_site_assets import load_mp3  # ffmpeg-based mp3 decode (see below)


# 2026-09-28 (n003 migration): the MelSpectrogram was being CONSTRUCTED on
# every call — measured 117.7 ms/item vs 43.1 ms/item for a shared instance
# (2.7x, n003 CPU, _hpc_diag4.sh). Singleton it.
_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=1024, hop_length=320, n_mels=N_MELS,
    f_min=50.0, f_max=8000.0,
)


def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """(1, T) @16k -> (1, 80, 50) log-mel, padded/cropped to fixed length."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[-1] > SR:  # center-crop to 1 s
        # BUGFIX 2026-09-27 (v1b still ~chance): this was a RANDOM crop, but
        # clips are ~0.39 s (median) — a random window keeps only a random
        # slice of the phrase (often the first word), so the model saw
        # partial words + padding and collapsed to the largest class
        # (media_control, 1000/6900 train). Center-crop keeps the phrase
        # core; for short clips (<=1 s, which is ~all of them) nothing is
        # cropped and the audio is centered in the window below.
        start = (wav.shape[-1] - SR) // 2
        wav = wav[..., start:start + SR]
    mel = _MEL(wav)
    logmel = torch.clamp(mel, min=1e-5).log10()
    # normalize to ~[0,1] (log10 range is ~[-5, 0])
    logmel = (logmel + 5.0) / 5.0
    if logmel.shape[-1] < N_FRAMES:
        # center the audio in the 50-frame window (was left-aligned — the
        # right half of every ~0.39 s clip was dead silence)
        pad = N_FRAMES - logmel.shape[-1]
        logmel = nn.functional.pad(
            logmel, (pad // 2, pad - pad // 2))
    return logmel[..., :N_FRAMES]


def resample_speed(wav: torch.Tensor, rate: float) -> torch.Tensor:
    """(1, T) -> (1, round(T/rate)) at `rate`x speed, pure-torch linear
    resample. ~1-2 ms/item steady-state vs ~3.7 s/item for
    torchaudio.functional.speed
    (measured _probe_resample.py, 2026-09-27 — that one routes through the
    broken torchcodec path too).

    BUGFIX 2026-09-27 (v1 collapse): the first version used
    F.interpolate(size=T2, mode='linear'), which maps the DESTINATION grid
    0..T2-1 onto the SOURCE grid 0..T-1 — i.e. it stretched/compressed
    relative to the original LENGTH, so at rate 0.95 the output was longer
    and slower, phase-drifting the whole clip (cosine corr vs clean ~0.04
    on real clips; v1 trained to 0.176 acc = chance). Correct time-stretch:
    sample the source at `rate`x (source grid 0..T/rate-1). Verified:
    cos_corr 0.99999 on a chirp and >0.99 on 12 real clips, ~1-2 ms/item."""
    T = wav.shape[-1]
    if rate == 1.0:
        return wav
    T2 = int(round(T / rate))
    if T2 <= 1:
        return wav
    src = torch.linspace(0.0, T / rate - 1, T2)
    i0 = src.long().clamp(max=T - 1)
    frac = (src - src.floor()).float()
    return wav[..., i0] * (1 - frac) + wav[..., (i0 + 1).clamp(max=T - 1)] * frac


def load_wav(p: Path) -> torch.Tensor:
    # Prefer pre-decoded .raw siblings (src/decode_wav.py): np.fromfile on
    # raw s16le is ~4 ms/item vs ~0.19 s/item for the per-file ffmpeg
    # subprocess (measured _probe_throughput.py / _probe_wavload.py,
    # 2026-09-27). torchaudio.load is NOT usable in this venv (torchcodec
    # ImportError, measured _probe_wavload.py). Fall back to the ffmpeg mp3
    # decode if no .raw exists yet.
    raw_p = p.with_suffix(".raw")
    if raw_p.exists():
        arr = RAW_CACHE.get(str(raw_p))
        if arr is None:  # preload_raw() not run (e.g. --smoke) — read direct
            arr = (np.fromfile(str(raw_p), dtype=np.int16)
                   / 32768.0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)
    return load_mp3(p)  # imageio_ffmpeg binary (see gen_site_assets)


RAW_CACHE: dict = {}
BASE_FEAT_CACHE: dict = {}  # str(mp3 path) -> base log-mel (1,80,50), no aug


def preload_raw(root: Path) -> int:
    """Load every .raw under root into an in-RAM cache (module-level
    RAW_CACHE, ~1 GB total).

    Why not just warm the page cache: on JuiceFS (n003) even CACHED
    per-file reads are ~100 ms each (measured 2026-09-28: 6900-item epoch
    took >15 min, GPU util 0% the whole time), so the OS cache alone does
    not fix the data path. In RAM, per-item loads are dict lookups (~0 ms)
    and the A100 actually gets fed. On local disk the old ~4 ms path is
    fine, but the cache is still a no-op-ish win (~1 s to build).
    Returns the number of files cached."""
    t0 = time.time()
    n = 0
    for raw_p in sorted(root.rglob("*.raw")):
        key = str(raw_p)
        if key not in RAW_CACHE:
            RAW_CACHE[key] = (np.fromfile(str(raw_p), dtype=np.int16)
                              / 32768.0).astype(np.float32)
        n += 1
    print(f"preload: {n} raw files cached in RAM in {time.time() - t0:.1f}s",
          flush=True)
    return n


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

    def _base_feat(self, p: Path) -> torch.Tensor:
        """Base (un-augmented) log-mel for p, from the RAM cache.

        2026-09-28 (n003 migration): the per-item mel transform cost ~75 ms
        of CPU (per-call torchaudio overhead) — 9 min/epoch of data prep
        starving the A100. Precomputing the base feature once per file
        (13,787 files, ~2 min) and re-augmenting in the mel domain
        (speed jitter = frame resample 0.42 ms/item, gain = +1 dB in log
        domain, noise = additive Gaussian) makes per-item prep ~1 ms."""
        key = str(p)
        feat = BASE_FEAT_CACHE.get(key)
        if feat is None:
            feat = wav_to_logmel(load_wav(p))
            BASE_FEAT_CACHE[key] = feat
        return feat

    def __getitem__(self, i):
        p, cls = self.items[i]
        x = self._base_feat(p)
        if self.augment:
            # speed jitter — resample the mel FRAMES (0.42 ms/item, measured
            # 2026-09-28). Equivalent to time-stretching the audio: frames
            # are evenly spaced in time, so stretching the frame axis is
            # exactly a pitch-preserving speed change for the features.
            rate = random.uniform(0.95, 1.05)
            T = x.shape[-1]
            T2 = int(round(T / rate))
            if T2 > 1:
                src = torch.linspace(0.0, T / rate - 1, T2)
                i0 = src.long().clamp(max=T - 1)
                frac = (src - src.floor()).float()
                x = x[..., i0] * (1 - frac) + x[..., (i0 + 1).clamp(max=T - 1)] * frac
                if x.shape[-1] < N_FRAMES:
                    pad = N_FRAMES - x.shape[-1]
                    x = nn.functional.pad(x, (pad // 2, pad - pad // 2))
                x = x[..., :N_FRAMES]
            # random gain ±6 dB (log-mel is log10: +1 dB = +1/20 on the
            # raw scale, and our features are (log10+5)/5 = dB/20 + 0.5, so
            # a dB gain is just an additive constant)
            x = x + random.uniform(-6.0, 6.0) / 20.0
            # white-noise floor (SNR 5-25 dB) — additive in the feature
            # domain, scaled to the feature's own power (same intent as the
            # waveform version, which is what v1b/v1c trained with)
            snr_db = random.uniform(5.0, 25.0)
            sig_p = x.pow(2).mean().clamp_min(1e-8)
            noise = torch.randn_like(x)
            noise_p = noise.pow(2).mean().clamp_min(1e-8)
            scale = torch.sqrt(sig_p / (noise_p * 10 ** (snr_db / 10.0)))
            x = x + noise * scale
        return x, CLASSES.index(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to warm-start from (e.g. runs/v1/vcm_v1.pt)")
    ap.add_argument("--class-weights", action="store_true",
                    help="inverse-frequency class weights on CrossEntropy "
                         "(v1d: media_control 2000 vs dim_lights 1194 train "
                         "clips skewed v1c toward the big class)")
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

    # Warm the page cache before the first epoch (matters on network FS —
    # see preload_raw docstring). No-op-ish on local disk (~1 s).
    preload_raw(root)

    # Precompute every base log-mel ONCE (RAM, ~43 ms/file -> ~10 min for
    # 13,787 files, one-time). After this, per-item prep is ~1 ms of mel-
    # domain augmentation, so the A100 is fed continuously (measured
    # 2026-09-28: without this, data prep was 75-118 ms/item on CPU and
    # the GPU sat at 0% util the whole epoch).
    t0 = time.time()
    for p, _ in train_ds.items:
        train_ds._base_feat(p)
    for p, _ in eval_ds.items:
        eval_ds._base_feat(p)
    print(f"base features: {len(BASE_FEAT_CACHE)} cached in "
          f"{time.time() - t0:.0f}s", flush=True)

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    va = DataLoader(eval_ds, batch_size=args.batch, num_workers=0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = VCM(len(CLASSES)).to(dev)
    if args.init_from:
        ckpt = Path(args.init_from)
        if not ckpt.exists():
            print(f"--init-from: {ckpt} not found"); return 2
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"warm-start from {ckpt}")
    print(f"device: {dev}")
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.class_weights:
        # inverse-frequency weights: w_c = N / (C * n_c), computed on the
        # TRAIN split (eval must stay untouched). Normalized so the mean
        # weight is 1 (loss scale unchanged, only per-class emphasis shifts).
        counts = torch.zeros(len(CLASSES))
        for _, cls in train_ds.items:
            counts[CLASSES.index(cls)] += 1
        n = len(train_ds)
        w = n / (len(CLASSES) * counts)
        w = (w / w.mean()).to(dev)
        print("class weights:", {CLASSES[i]: round(float(w[i]), 3)
                                 for i in range(len(CLASSES))})
        lossf = nn.CrossEntropyLoss(weight=w)
    else:
        lossf = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(1, args.epochs + 1):
        m.train()
        t0 = time.time()
        tot = cnt = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
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
                x, y = x.to(dev), y.to(dev)
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
