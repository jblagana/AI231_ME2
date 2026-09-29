"""VCM v1 — tiny CNN over log-mel spectrograms.

Architecture (PLAN.md): 80-bin log-mel, 1 s window @ 16 kHz, 20 ms hop
(50 frames). 3 conv blocks (32/64/128 filters, 3x3, maxpool) -> global
pool -> dropout -> FC. Target < 1 M params.

Pooling: v1-v1f used AdaptiveAvgPool2d (averages the whole 50-frame time
axis, so a ~10-15-frame verb gets diluted into the boilerplate mean).
v1g switches to AdaptiveMaxPool2d — lets the most salient frame (the verb)
win instead of being averaged out. Same state_dict shape (pools have no
params), so a v1e/v1f checkpoint still loads for a warm-start A/B.
v1h (FrontWeightedPool): max still latches onto the LOUDEST shared cell
(the content word "music", or the shared time-content "today/tomorrow")
and throws the leading word away. FrontWeightedPool keeps every time cell
(front-weighted mean, front-to-back geometric ramp) then max over freq, so
the leading word (interrogative "what" / imperative "set") survives to the
FC. No learnable params -> a v1g checkpoint still loads for a warm-start A/B.

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


class FrontWeightedPool(nn.Module):
    """v1h: front-weighted time pool + max over freq.

    The conv stack downsamples time 8x (50 -> 6 cells) and freq 8x
    (80 -> 10 cells). v1g's AdaptiveMaxPool2d(1) grabs the single LOUDEST
    of the 60 cells — for "play/pause music" that's usually the shared
    content word "music", so both classes collapse onto the same cell and
    the leading word (play vs pause / what vs set) is discarded before the
    FC ever sees it.

    This pool instead:
      1. front-weighted mean over the TIME axis (geometric ramp, front cell
         weight = 1.0, back cell = 1/ratio^(T-1)) — keeps EVERY time cell,
         so the leading word survives, and the ramp tells the FC the front
         is the interrogative/imperative discriminator.
      2. max over the FREQ axis (the most salient frequency band still wins,
         preserving the "salient feature" benefit that made max-pool beat
         avg-pool in v1g).

    No learnable params -> output shape (B, 128, 1, 1) is identical to
    AdaptiveMaxPool2d(1), so a v1g checkpoint loads cleanly for a warm-start
    A/B. `ratio` = front/back weight ratio (default 1.5).
    """

    def __init__(self, ratio: float = 1.5):
        super().__init__()
        self.ratio = ratio

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, F, T)
        T = h.shape[-1]
        # BUGFIX 2026-09-29 (v1h2): the original arange(0, T) ramp was
        # BACK-weighted — the back cell got ratio^(T-1) = 7.6x the front
        # cell, contradicting the docstring. v1h's -0.12 therefore measured
        # back-weighted pooling (tail-biased: the shared content words
        # "music" / "today" sit at the clip end), NOT front-weighting.
        # Flipped so the front cell carries the largest weight (front/back
        # ratio unchanged: ratio^(T-1) : 1; normalization absorbs the scale).
        w = torch.pow(self.ratio, torch.arange(T - 1, -1, -1, device=h.device, dtype=h.dtype))
        w = w / w.sum()  # (T,)  front cell gets the largest weight
        h = h * w.view(1, 1, 1, T)          # front-weighted, keep time dim
        h = h.mean(dim=-1, keepdim=True)    # (B, C, F, 1)  weighted time-mean
        h = h.max(dim=-2, keepdim=True).values  # (B, C, 1, 1)  max over freq
        return h


class FrontBiasedMaxPool(nn.Module):
    """v1j: front-biased max over TIME + max over FREQ.

    Weighted MAX instead of weighted mean (v1h2): scale the time cells by
    the same front-heavy ramp, then max over time. The single loudest cell
    still wins when it is decisively loud (preserves the salience win that
    made v1f(avg) -> v1g(max) +0.08), but when cells are comparable in
    magnitude the FRONT cell breaks the tie — the leading word
    (interrogative "what" / imperative "set") wins without re-introducing
    mean dilution. Note: plain UNWEIGHTED max-over-time + max-over-freq is
    EXACTLY v1g's global max (max is commutative) — the weights are the only
    thing that makes this a new arm. No learnable params -> output shape
    (B, 128, 1, 1) identical to AdaptiveMaxPool2d(1); v1g checkpoints
    warm-start cleanly.
    """

    def __init__(self, ratio: float = 1.5):
        super().__init__()
        self.ratio = ratio

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, F, T)
        T = h.shape[-1]
        # same front-heavy ramp as the fixed FrontWeightedPool (v1h2)
        w = torch.pow(self.ratio, torch.arange(T - 1, -1, -1, device=h.device, dtype=h.dtype))
        w = w / w.sum()  # (T,)  keeps output magnitude near v1g scale
        h = h * w.view(1, 1, 1, T)              # front-biased scale
        h = h.max(dim=-1, keepdim=True).values   # (B, C, F, 1)  weighted time-max
        h = h.max(dim=-2, keepdim=True).values   # (B, C, 1, 1)  max over freq
        return h


class VCM(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = N_MELS, dropout: float = 0.3,
                 pool: str = "max"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # pool: "max" = v1g champion global max (default — v1i unchanged),
        # "fw_mean" = v1h2 fixed front-weighted mean, "fw_max" = v1j
        # front-biased max. All three are param-free with the same output
        # shape (128,1,1) as v1g's AdaptiveMaxPool2d(1), so any v1g
        # checkpoint warm-starts any arm. (v1h's original FrontWeightedPool
        # had a sign bug — its ramp was back-weighted; v1h2 fixes it.)
        self.pool = {"max": nn.AdaptiveMaxPool2d(1),
                     "fw_mean": FrontWeightedPool(),
                     "fw_max": FrontBiasedMaxPool()}[pool]
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
