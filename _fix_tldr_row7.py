"""Section 1 table: add row 7 (v1d on HPC) + update the TL;DR cell (0)."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# --- row 7 in the section-1 issues table ---
c1 = nb["cells"][3]
src = "".join(c1["source"])
anchor = "| H1 | HPC epoch 1 never finished"
assert anchor in src
row7 = """| 7 | v1c plateaued at 0.363 — biggest class (media_control, 1000 clips) out-weighed the small ones (600) | class imbalance in the CE loss (inverse-freq weights were the planned next lever) | `--class-weights` flag (inverse-freq, mean-normalized, train-split only) + retrain as **v1d on n003 HPC** (Section 9) — **final 0.629** (peak 0.630), +0.27 over v1c |
"""
src = src.replace(anchor, row7 + anchor)
c1["source"] = src.splitlines(keepends=True)

# --- TL;DR cell ---
c0 = nb["cells"][0]
src0 = "".join(c0["source"])
old_tldr = """**TL;DR of the whole saga:** v1 (15 epochs) collapsed to ~chance accuracy
(0.176) → diagnosis found the **speed-jitter augmentation was producing
garbage features** (a misaligned-resample bug) → fixed with a correct
manual linear resample (1.2 ms/item, corr 0.99999) → retraining now."""
new_tldr = """**TL;DR of the whole saga:** v1 (15 epochs) collapsed to ~chance accuracy
(0.176) → diagnosis found the **speed-jitter augmentation was producing
garbage features** (a misaligned-resample bug) → fixed (0.183, still bad)
→ second bug: **random 1 s crop** of ~2 s clips (0.363 after center-crop)
→ class imbalance: **v1d with class-weighted CE on the COE HPC A100
(n003) = 0.629** (Section 9). Every issue hit — laptop *and* HPC — is in
the table in Section 1 with its measured root cause and fix."""
assert old_tldr in src0
src0 = src0.replace(old_tldr, new_tldr)
c0["source"] = src0.splitlines(keepends=True)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("TL;DR + row 7 updated")
