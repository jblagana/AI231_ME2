"""Rewrite the v1d launch cell (23): stdin redirect fixes the ssh hang (H6),
strict pgrep pattern (the wrapper's own cmdline used to match)."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

SRC = '''# H9 · v1d launch (idempotent — safe to re-run)
# NOTE (issue H6): the nohup'd command MUST get `</dev/null` — ssh otherwise
# holds the channel open on the background job's inherited stdin and the
# launch call hangs (measured: 90 s timeout in this very cell, 2026-09-28).
import subprocess, re, time

def hpc(cmd, timeout=90):
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
         "jan.rhey.lagana@n003.ai.internal", cmd],
        capture_output=True, text=True, timeout=timeout)

r = hpc("pgrep -f 'python src/train.py --epochs'")
pid = r.stdout.strip()
if pid:
    print(f"v1d already running (pid {pid}) — check the progress cell")
else:
    hpc("cd /mnt/jfs_hpc/home/jan.rhey.lagana/vcm && "
        "nohup bash _train_v1d.sh > /dev/null 2>&1 < /dev/null & "
        "sleep 2; echo launched", timeout=30)
    time.sleep(3)
    r = hpc("pgrep -f 'python src/train.py --epochs'")
    print("v1d launched on n003 (GPU 1), pid", r.stdout.strip() or "??")
    print("one-time base-feature precompute takes ~10 min before epoch 1")'''

nb["cells"][23] = {"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [],
                   "source": SRC.splitlines(keepends=True)}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("launch cell rewritten")
