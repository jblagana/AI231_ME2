"""Update v1d numbers to the final (re-run) values: 0.636 final / 0.636 peak,
568s precompute, ~2.7s/epoch."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c["source"])
    orig = src
    src = src.replace(
        "finished all 15 epochs in ~13 min wall time: **final eval_acc 0.629**\n(peak 0.630) vs v1c's 0.363",
        "finished all 15 epochs in ~13 min wall time: **final eval_acc 0.636**\nvs v1c's 0.363")
    src = src.replace(
        "one-time base-feature\nprecompute 597 s (~10 min); then **2.6 s/epoch**",
        "one-time base-feature\nprecompute ~570-600 s (~10 min); then **2.6-3.1 s/epoch**")
    src = src.replace(
        "retrain as **v1d on n003 HPC** (Section 9) — **final 0.629**\n(peak 0.630), +0.27 over v1c",
        "retrain as **v1d on n003 HPC** (Section 9) — **final 0.636**,\n+0.27 over v1c")
    src = src.replace(
        "(n003) = 0.629** (Section 9).",
        "(n003) = 0.636** (Section 9).")
    src = src.replace(
        "| **v1d** | **n003 A100** | **+ class-weighted CE, warm-start v1c** | **0.629** | **0.630** |",
        "| **v1d** | **n003 A100** | **+ class-weighted CE, warm-start v1c** | **0.636** | **0.636** |")
    if src != orig:
        hits += 1
    c["source"] = src.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("cells updated:", hits)
