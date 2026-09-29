"""Feature-statistics sanity check: SLURP (real) vs TTS (training domain).

Purpose: the SLURP cross-domain eval (src/eval_slurp.py) reported 0.101
accuracy with the model collapsing to 2 classes (play_music + set_reminder).
Before deciding whether v2 needs real-human DATA or just better FEATURES,
this probes WHERE the collapse happens:

  1. Raw log-mel feature statistics (mean/std/min/max overall + per-mel-band
     mean) — if real-voice features are wildly out of the TTS range, the
     collapse is a feature-mismatch artifact (fixable with normalization).
  2. Model conv-output activations (the 128-channel feature map BEFORE the
     pool) — if the conv features themselves collapse to a tiny region on
     real audio, the collapse is in the model's learned representation, not
     the raw feature scale.

Usage:
    python src/feature_stats.py --ckpt runs/v1e/vcm_v1.pt --n 50
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from model import VCM, N_MELS, N_FRAMES  # noqa: E402
from train import wav_to_logmel, load_wav  # noqa: E402
from gen_site_assets import load_mp3       # noqa: E402
from commands import CLASSES               # noqa: E402

MANIFEST = REPO / "data" / "eval_slurp" / "manifest.csv"
AUDIO_DIR = REPO / "data" / "eval_slurp" / "audio" / "slurp_real"
TRAIN_DIR = REPO / "data" / "raw" / "train"


def load_flac(p: Path) -> torch.Tensor:
    return load_mp3(p)  # ffmpeg handles flac -> (1,T)@16k mono


def sample_slurp(n: int, rng) -> list:
    """Return (path, class) for n SLURP clips (balanced across classes)."""
    rows = list(csv.DictReader(MANIFEST.open()))
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    per = max(1, n // len(by_class))
    out = []
    for cls, rs in by_class.items():
        rs = rng.sample(rs, min(per, len(rs)))
        for r in rs:
            p = AUDIO_DIR / r["file"]
            if p.exists():
                out.append((p, cls))
    return out


def sample_tts(n: int, rng) -> list:
    """Return (path, class) for n TTS clips (balanced across classes).

    Uses the .raw siblings via load_wav (the exact training load path)."""
    by_class = {}
    for cls in CLASSES:
        d = TRAIN_DIR / cls
        if d.exists():
            by_class[cls] = [p for p in d.glob("*.mp3") if p.with_suffix(".raw").exists()]
    per = max(1, n // len(by_class))
    out = []
    for cls, ps in by_class.items():
        ps = rng.sample(ps, min(per, len(ps)))
        for p in ps:
            out.append((p, cls))
    return out


def feats_for(items, loader, rng) -> torch.Tensor:
    """(1,80,50) log-mel for each item, stacked -> (N,80,50)."""
    fs = []
    for p, _ in items:
        wav = loader(p)
        x = wav_to_logmel(wav)  # (1,80,50)
        fs.append(x.squeeze(0))
    return torch.stack(fs)  # (N,80,50)


def conv_features(m: VCM, x: torch.Tensor) -> torch.Tensor:
    """The 128-channel feature map BEFORE the pool: (N,128,F,T)."""
    m.eval()
    with torch.no_grad():
        h = m.conv(x)  # (N,128,F,T)
    return h


def summarize(name: str, f: torch.Tensor, conv: torch.Tensor) -> None:
    f = f.float()
    conv = conv.float()
    print(f"\n== {name} ==")
    print(f"  raw log-mel  : mean={f.mean():.4f}  std={f.std():.4f}  "
          f"min={f.min():.4f}  max={f.max():.4f}")
    # per-mel-band mean (80 values) — the spectral shape
    band = f.mean(dim=(0, 2))  # (80,)
    print(f"  per-band mean: min={band.min():.4f}  max={band.max():.4f}  "
          f"std={band.std():.4f}")
    # silence fraction: frames whose mean energy is near the floor
    frame_energy = f.mean(dim=1)  # (N,50)
    silence_frac = (frame_energy < 0.05).float().mean().item()
    print(f"  silence-frame frac (mean energy < 0.05): {silence_frac:.3f}")
    # conv activation stats — the model's learned representation
    print(f"  conv output  : mean={conv.mean():.4f}  std={conv.std():.4f}  "
          f"min={conv.min():.4f}  max={conv.max():.4f}")
    # per-channel std of the conv output — if real audio collapses, the
    # channels that fire on TTS will be ~0 on real
    ch_std = conv.std(dim=(0, 2, 3))  # (128,)
    print(f"  conv ch-std  : min={ch_std.min():.4f}  max={ch_std.max():.4f}  "
          f"median={ch_std.median():.4f}  (n_channels>0.01: "
          f"{(ch_std > 0.01).sum().item()}/{ch_std.numel()})")
    # how much of the conv activation mass is in the top-10% of channels
    topk = torch.topk(ch_std, max(1, ch_std.numel() // 10)).values
    print(f"  conv top-10% ch-std sum: {topk.sum():.3f}  "
          f"(all-ch sum: {ch_std.sum():.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=50, help="clips per domain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    m = VCM(len(CLASSES))
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m.eval()
    print(f"loaded {args.ckpt}")

    slurp = sample_slurp(args.n, rng)
    tts = sample_tts(args.n, rng)
    print(f"sampled {len(slurp)} SLURP, {len(tts)} TTS")

    fs_s = feats_for(slurp, load_flac, rng)
    fs_t = feats_for(tts, load_wav, rng)
    print(f"raw feats: slurp {tuple(fs_s.shape)}, tts {tuple(fs_t.shape)}")

    # pad to same N for a direct paired comparison (use the smaller)
    n = min(fs_s.shape[0], fs_t.shape[0])
    fs_s, fs_t = fs_s[:n], fs_t[:n]

    summarize("SLURP (real)", fs_s, conv_features(m, fs_s.unsqueeze(1)))
    summarize("TTS (train)", fs_t, conv_features(m, fs_t.unsqueeze(1)))

    # Direct paired comparison of the per-band mean spectral shape
    band_s = fs_s.float().mean(dim=(0, 2))
    band_t = fs_t.float().mean(dim=(0, 2))
    # cosine similarity of the two spectral envelopes
    cos = torch.nn.functional.cosine_similarity(band_s, band_t, dim=0).item()
    # max absolute per-band deviation
    dev = (band_s - band_t).abs()
    print(f"\n== spectral envelope comparison (per-band mean) ==")
    print(f"  cosine similarity (slurp vs tts): {cos:.4f}")
    print(f"  max abs per-band deviation      : {dev.max():.4f}")
    print(f"  mean abs per-band deviation     : {dev.mean():.4f}")
    worst = torch.topk(dev, 5).indices.tolist()
    print(f"  worst bands (mel idx): {worst}  (deviations: "
          f"{[round(dev[i].item(), 3) for i in worst]})")

    # Overall feature-scale ratio (the key question: are real features in
    # the same ballpark as TTS, or off by an order of magnitude?)
    print(f"\n== feature-scale ratio (slurp / tts) ==")
    ms, mt = fs_s.float().mean().item(), fs_t.float().mean().item()
    ss, st = fs_s.float().std().item(), fs_t.float().std().item()
    ps = torch.quantile(fs_s.float().flatten(), 0.95).item()
    pt = torch.quantile(fs_t.float().flatten(), 0.95).item()
    print(f"  mean ratio : {ms / max(mt, 1e-9):.3f}")
    print(f"  std ratio  : {ss / max(st, 1e-9):.3f}")
    print(f"  p95 ratio  : {ps / max(pt, 1e-9):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
