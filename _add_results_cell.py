"""Append the v1d results markdown cell after the progress cell."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
if any("v1d result" in "".join(c["source"]) for c in nb["cells"]):
    print("already present")
    raise SystemExit(0)

MD = """## 9.2 · v1d result (2026-09-28, n003 A100)

| run | where | change | final eval_acc | peak |
|---|---|---|---|---|
| v1 | laptop CPU | (buggy resample + random crop) | 0.176 | — |
| v1b | laptop CPU | resample fixed | 0.183 | — |
| v1c | laptop CPU | + center crop/pad | 0.363 | 0.403 |
| **v1d** | **n003 A100** | **+ class-weighted CE, warm-start v1c** | **0.629** | **0.630** |

Class weights (inverse-freq, mean-normalized, train split only):
media_control 0.67, make_call 0.837, control_lights 0.837, ask_question
0.957, the six 600-clip classes 1.116.

**Reading:** the HPC move itself bought ~29x wall-clock (2.6 s/epoch vs
75 s) — the accuracy jump is the `--class-weights` flag (+0.27), which was
always the next lever; the HPC just made it cheap to run. Still short of
the 0.80 target in PLAN.md — remaining levers: more clips per class
(600-1000 vs the 1000-2000 target), real background noise in the aug,
and longer fine-tunes (now trivial at 2.6 s/epoch)."""

nb["cells"].append({"cell_type": "markdown", "metadata": {},
                    "source": MD.splitlines(keepends=True)})
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("results cell appended — now", len(nb["cells"]), "cells")
