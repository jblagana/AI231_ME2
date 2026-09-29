"""A2 — two-head VCM: command head + slot head over a 3.0 s window.

Why two heads (PLAN.md A2, 2026-09-30):
  The verb ("set" / "dim" / "call") sits in the first ~0.5 s of the phrase;
  the slot value ("twenty five degrees" / "mom") sits in the LAST spoken
  word. A 1.0 s window (50 frames -> 6 time cells) puts the verb and the
  slot in the SAME 2-3 cells, so a single pooled vector can't separate them.
  A 3.0 s window (150 frames -> 18 time cells) gives a clean spatial split:
    cells 0-3   ~ verb
    cells 6-17  ~ content + slot value
  Head 1 (command) reads the whole window (global max over all 18 cells).
  Head 2 (slot) reads the TAIL cells (global max over the last SLOT_TAIL_CELLS),
  where the slot value (the last spoken word) actually lives.

Architecture (self-contained; does NOT touch model.py / train.py):
  Input (B, 1, 80, 150)
    -> conv stack (32/64/128, 3x3 pad1, BN, ReLU, MaxPool2 x3)
    -> (B, 128, 10, 18)            [10 freq x 18 time cells]
    Head 1: AdaptiveMaxPool2d(1) over all 18 cells -> dropout -> FC(128, 11)
    Head 2: slice tail cells [-SLOT_TAIL_CELLS:] -> AdaptiveMaxPool2d(1)
          -> FC(128, N_c) per parametric class c (6 heads, one per slot class)

Params: ~100 K (conv 92.9 K + cmd FC 1.4 K + slot FC 5.5 K) — << 1 M budget.

Usage:
  python src/model_a2.py --smoke    # forward pass + param count + shape check
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from slots import SLOT_COUNTS, PARAMETRIC  # noqa: E402

N_MELS = 80
SR = 16000
WINDOW_S = 3.0
HOP_S = 0.020
N_FRAMES = int((WINDOW_S - HOP_S) / HOP_S) + 1  # 150

# The slot value is the LAST spoken word of the phrase. Clips are
# left-aligned in the 3.0 s window, so the slot word sits near the end of the
# voiced region — but that end varies by clip length:
#   make_call       last-word-start cell ~1.2  ("call mom" — short clip)
#   dim_lights      ~7.7
#   set_temperature ~10.6
#   set_timer       ~11.0
# A fixed FRONT slice (cells [K:]) would drop make_call's slot (cell 1-3).
# A fixed BACK slice (cells [-K:]) captures the tail where the slot word
# lives for ALL clip lengths: with K=8 (cells 10-17) the slot word is inside
# for every class (verified _a2_cellprobe.py, 2026-09-30). The verb
# (cells 0-3) is excluded, so the slot head doesn't latch onto the shared
# "call" / "set" / "dim" boilerplate.
SLOT_TAIL_CELLS = 8


class VCMTwoHead(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = N_MELS, dropout: float = 0.3,
                 slot_tail_cells: int = SLOT_TAIL_CELLS):
        super().__init__()
        self.slot_tail_cells = slot_tail_cells
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Head 1 — command: global max over ALL time cells (v1g-style), then FC.
        self.cmd_pool = nn.AdaptiveMaxPool2d(1)
        self.cmd_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        # Head 2 — slot: one linear per parametric class, applied to the
        # pooled LATER-cell feature. Only the head matching the clip's command
        # class is used for that clip's slot loss / prediction.
        self.slot_pool = nn.AdaptiveMaxPool2d(1)
        self.slot_heads = nn.ModuleDict({
            cls: nn.Linear(128, n) for cls, n in SLOT_COUNTS.items()
        })

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared conv features: (B,1,F,T) -> (B,128,10,18)."""
        return self.conv(x)

    def forward(self, x: torch.Tensor):
        """Return (cmd_logits, slot_feat).

        cmd_logits: (B, n_classes)
        slot_feat:  (B, 128) — pooled later-cell feature; apply
                    slot_logits(slot_feat, cls) to get a class's slot logits.
        """
        h = self.conv(x)                                  # (B,128,F,T)
        cmd = self.cmd_pool(h).flatten(1)                 # (B,128)
        cmd_logits = self.cmd_fc(cmd)                     # (B,n_classes)
        hs = h[:, :, :, -self.slot_tail_cells:]           # (B,128,F,K) tail cells
        slot_feat = self.slot_pool(hs).flatten(1)         # (B,128)
        return cmd_logits, slot_feat

    def slot_logits(self, slot_feat: torch.Tensor, cls: str) -> torch.Tensor:
        """(B,128) -> (B, N_c) for one parametric class."""
        return self.slot_heads[cls](slot_feat)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def smoke():
    n_classes = 11
    m = VCMTwoHead(n_classes)
    x = torch.randn(4, 1, N_MELS, N_FRAMES)
    cmd_logits, slot_feat = m(x)
    n = param_count(m)

    # conv feature shape check
    with torch.no_grad():
        feat = m.features(x)
    print(f"input: {tuple(x.shape)}")
    print(f"conv features: {tuple(feat.shape)}  (expect (4,128,10,18))")
    print(f"cmd_logits: {tuple(cmd_logits.shape)}  (expect (4,{n_classes}))")
    print(f"slot_feat:  {tuple(slot_feat.shape)}  (expect (4,128))")
    for cls in PARAMETRIC:
        sl = m.slot_logits(slot_feat, cls)
        print(f"  slot[{cls}]: {tuple(sl.shape)}  (expect (4,{SLOT_COUNTS[cls]}))")

    print(f"params: {n:,} ({n/1e6:.3f}M) — target < 1M: {'OK' if n < 1_000_000 else 'OVER'}")
    print(f"  conv: {param_count(m.conv):,}  cmd_fc: {param_count(m.cmd_fc):,}  "
          f"slot_heads: {sum(param_count(h) for h in m.slot_heads.values()):,}")

    # rough latency
    m.eval()
    with torch.no_grad():
        for _ in range(10):
            m(x)
        t0 = time.perf_counter()
        for _ in range(100):
            m(x)
        dt = (time.perf_counter() - t0) / 100
    print(f"CPU forward (batch 4, this machine): {dt*1000:.1f} ms -> "
          f"~{dt*1000/4:.2f} ms per 3s clip here")

    ok = (n < 1_000_000
          and cmd_logits.shape == (4, n_classes)
          and feat.shape == (4, 128, 10, 18)
          and slot_feat.shape == (4, 128))
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sys.exit(smoke() if args.smoke else 0)
