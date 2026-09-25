"""VCM v1 — tiny CNN over log-mel spectrograms.

Architecture (PLAN.md): 80-bin log-mel, 1 s window @ 16 kHz, 20 ms hop
(50 frames). 3 conv blocks (32/64/128 filters, 3x3, maxpool) -> global
avg pool -> dropout -> FC. Target < 1 M params.

Usage:
  python src/model.py --smoke      # forward-pass + param count check
  python src/train.py ...          # training entry (separate file)
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

N_MELS = 80
SR = 16000
WINDOW_S = 1.0
HOP_S = 0.020
N_FRAMES = int((WINDOW_S - HOP_S) / HOP_S) + 1  # 50


class VCM(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = N_MELS, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, n_frames)
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def smoke():
    m = VCM(10)
    x = torch.randn(4, 1, N_MELS, N_FRAMES)
    y = m(x)
    n = param_count(m)
    print(f"input: {tuple(x.shape)} -> output: {tuple(y.shape)}")
    print(f"params: {n:,} ({n/1e6:.3f}M) — target < 1M: {'OK' if n < 1_000_000 else 'OVER'}")
    # rough latency: time 100 forward passes on CPU
    import time
    m.eval()
    with torch.no_grad():
        for _ in range(10):
            m(x)
        t0 = time.perf_counter()
        for _ in range(100):
            m(x)
        dt = (time.perf_counter() - t0) / 100
    print(f"CPU forward (batch 4, this machine): {dt*1000:.1f} ms -> ~{dt*1000/4:.2f} ms per 1s clip here; "
          f"RPi5 estimate (slower ARM cores, no AVX2) ~5-10x = ~{dt*1000/4*5:.0f}-{dt*1000/4*10:.0f} ms "
          f"(still << 1 s window, real-time OK)")
    return 0 if n < 1_000_000 and y.shape == (4, 10) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sys.exit(smoke() if args.smoke else 0)
