"""Append section 9 (HPC migration) to vcm_v1_training.ipynb.

Run from the repo root:  .venv\\Scripts\\python.exe _build_nb_hpc.py
Idempotent-ish: skips if a cell already contains the H9 marker.
"""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
MARKER = "## 9"

nb = json.loads(NB.read_text(encoding="utf-8"))
if any(MARKER in "".join(c["source"]) for c in nb["cells"]):
    print("section 9 already present — nothing to do")
    raise SystemExit(0)

# --- also append the H1-H6 rows to the section-1 issues table (idempotent) ---
ISSUE_ROW = "| H1 |"
for c in nb["cells"]:
    if c["cell_type"] == "markdown" and "## 1 · Issues" in "".join(c["source"]):
        src = "".join(c["source"])
        if ISSUE_ROW not in src:
            src = src.replace(
                "Probe scripts (kept at repo root):",
                """| H1 | HPC epoch 1 never finished (15+ min, GPU 0% util) | JuiceFS per-file reads ~100 ms/item **even cached** (page-cache warm alone didn't fix it) | in-RAM dict of all `.raw` at start (`preload_raw`, ~1 GB, ~60-85 s one-time) → 0.45 ms/item (Section 9) |
| H2 | GPU still 0% after H1; process at 8000% CPU | **train.py never used CUDA** — no `.to(device)` anywhere; v1-v1c were all CPU (laptop *and* HPC) | device plumbing (model/batches/weights → cuda); A100 fwd+bwd measured 69 ms/batch vs ~75 s/epoch CPU |
| H3 | crash: `weight is on cpu, different from other tensors on cuda:0` | class-weights tensor built on CPU, CE loss on GPU (only with `--class-weights`) | `w.to(dev)` (1 line) |
| H4 | epoch still ~10 min — data prep 75-118 ms/item starved the GPU | `MelSpectrogram` **constructed on every call** (117.7 ms/item measured); 43 ms even shared | singleton `_MEL` + precompute every base log-mel once in RAM + re-implement the 3 augmentations in the **mel domain** (frame resample 0.42 ms/item) → per-item prep ~1 ms |
| H5 | checkpoint scp failed (`No such file`) | remote `runs/v1c/` didn't exist — git ignores `*.pt` | `mkdir -p` first, re-scp |
| H6 | `nohup … &` over ssh hangs the session (120 s tool timeout) | ssh waits for the child's stdout/stderr fds | redirect fds to `/dev/null` + verify with a separate ssh (`pgrep`) — the job itself is fine |

Probe scripts (kept at repo root):""")
            c["source"] = src.splitlines(keepends=True)
        break

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = [
md("""## 9 · HPC migration — n002 vs n003, why, and what it cost (2026-09-28)

**Why move:** v1c trained on my laptop (no GPU — the script never even had
a CUDA path) at ~75 s/epoch. The COE HPC nodes have 8x A100-40GB each.
The boss said: pick the better node, move the work there.

**Node comparison (measured live, 2026-09-28 ~01:30):**

| | n002 | n003 |
|---|---|---|
| CPU | 256 cores | 256 cores |
| RAM | 1007 GB (834 free) | 1007 GB (962 free) |
| A100-40GB x8 | all 70-100% busy, 4-38 GB used | **all 8 idle, 0 MB** |
| my conda env | none | `~/.conda/envs/vcm` (torch 2.14+cu130) |
| shared FS | JuiceFS (907 TB free) | JuiceFS (same mount) |

**Verdict: n003** — same box, zero GPU contention, env already there.
n002's GPUs were fully committed to other users at probe time; utilization
fluctuates, so re-check before any future move.

**What was done (all logged below with the issues hit):**
1. `git clone` repo into `~/vcm` on JuiceFS (shared home — both nodes see it)
2. tar + scp the 1.2 GB dataset (mp3 + pre-decoded .raw + manifest)
3. pip: `imageio_ffmpeg`, `matplotlib` into the vcm env
4. scp the v1c checkpoint for warm-start
5. 4 real bugs found & fixed along the way (table below) — each one is a
   row in the §1 issues table too"""),

code("""# H9 · node probe (re-run anytime to see which node is free)
import subprocess, re

def probe(host):
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
         f"jan.rhey.lagana@{host}.ai.internal",
         "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; free -g | sed -n 2p"],
        capture_output=True, text=True, timeout=60)
    gpus = [l for l in r.stdout.splitlines()
            if re.match(r"^\\d+, \\d+ MiB", l)]
    busy = sum(1 for g in gpus if int(g.split(",")[2].strip().rstrip("%")) > 5)
    mem = r.stdout.splitlines()[-1].split()
    return f"{host}: {busy}/8 GPUs busy, RAM {mem[1]}/{mem[0]} GB used"

for h in ("n002", "n003"):
    try:
        print(probe(h))
    except Exception as e:
        print(f"{h}: probe failed ({e})")"""),

md("""### 9.1 · Issues hit during the HPC move (measured, not guessed)

| # | Issue (symptom) | Root cause | Fix / workaround |
|---|---|---|---|
| H1 | first epoch never produced a line for 15+ min; GPU util 0% | **JuiceFS per-file reads are ~100 ms/item even CACHED** (page-cache warm alone did not help — measured: 6900-item epoch still >15 min with GPU idle) | load every `.raw` into an **in-RAM dict** at start (`preload_raw`, ~1 GB, 50-85 s one-time); per-item load drops to 0.45 ms (measured) |
| H2 | GPU still 0% after H1 fix; process at 8000% CPU | **the script never used CUDA** — no `.to(device)` anywhere; v1-v1c all trained on CPU (laptop *and* HPC) | added device plumbing (model, batches, class-weights tensor → cuda). Real A100 fwd+bwd measured at 69 ms/batch vs ~75 s/epoch CPU |
| H3 | crash: `weight is on cpu, different from other tensors on cuda:0` | class-weights tensor built on CPU, CE loss on GPU (only appears with `--class-weights`) | `w.to(dev)` — 1-line fix |
| H4 | epoch still ~10 min: data prep 75-118 ms/item on CPU starved the GPU | `torchaudio.MelSpectrogram` was **constructed on every call** (117.7 ms/item) and the per-item mel is 43 ms even shared — 6900 items = 5 min/epoch of pure prep | (a) singleton `_MEL` (117→43 ms), (b) **precompute every base log-mel once** into RAM (~10 min one-time), (c) re-implement the 3 augmentations in the **mel domain** (frame resample 0.42 ms/item, gain = additive dB, noise = additive Gaussian) → per-item prep ~1 ms |
| H5 | `scp` of the checkpoint failed: `No such file` | remote `runs/v1c/` dir didn't exist (git doesn't track `*.pt`) | `mkdir -p` first, re-scp (385 KB, fine) |
| H6 | `nohup ... &` over ssh hangs the ssh session 120 s (tool timeout) | ssh waits for the child's stdout/stderr fds to close | redirect the nohup'd command's fds to files/`/dev/null` and verify with a *separate* ssh (`pgrep`) — the job itself is fine |

**Net effect (measured):** epoch data-prep ~9 min → ~15 s; A100 fwd+bwd
14.9 s/epoch. 15 epochs + evals should be **~10-15 min total** including
the one-time ~10 min feature precompute, vs ~19 min per run on the laptop
CPU — and the A100 has 100x headroom for bigger experiments (batch 128,
more epochs, bigger models) that were impossible on the laptop.""",
),

code("""# H9 · v1d launch (idempotent — safe to re-run)
import subprocess, re, time

def hpc(*cmd, timeout=90):
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
         "jan.rhey.lagana@n003.ai.internal"] + list(cmd),
        capture_output=True, text=True, timeout=timeout)

r = hpc("pgrep -f 'python src/train.py'")
pid = r.stdout.strip()
if pid:
    print(f"v1d already running (pid {pid}) — tailing log:")
else:
    hpc("cd /mnt/jfs_hpc/home/jan.rhey.lagana/vcm && "
        "nohup bash _train_v1d.sh > /dev/null 2>&1 & sleep 2; echo launched")
    time.sleep(3)
    print("v1d launched on n003 (GPU 1) — class-weighted CE, warm-start v1c")
    print("one-time base-feature precompute takes ~10 min before epoch 1")"""),

code("""# H9 · v1d progress (re-run anytime)
import subprocess, re

r = subprocess.run(
    ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
     "jan.rhey.lagana@n003.ai.internal",
     "tail -25 /mnt/jfs_hpc/home/jan.rhey.lagana/vcm/train_v1d.log"],
    capture_output=True, text=True, timeout=60)
out = r.stdout
lines = [l for l in out.splitlines() if re.match(r"ep\\s+\\d+", l)]
for l in lines[-8:]:
    print(l)
if not lines:
    last = [l for l in out.splitlines() if l.strip()][-2:]
    print("no epochs yet —", " | ".join(last) if last else "log empty")
if "saved ->" in out:
    print("\\nDONE — final line:")
    for l in out.splitlines():
        if "saved" in l: print(l)
"""),
]

nb["cells"].extend(cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"appended {len(cells)} cells — notebook now has {len(nb['cells'])} cells")
