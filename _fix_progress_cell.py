"""Fix the progress cell (24): strip the HPC login banner (30 lines) before
parsing, and tail more so the banner can't eat the log."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

SRC = '''# H9 · v1d progress (re-run anytime)
import subprocess, re

r = subprocess.run(
    ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
     "jan.rhey.lagana@n003.ai.internal",
     "tail -60 /mnt/jfs_hpc/home/jan.rhey.lagana/vcm/train_v1d.log"],
    capture_output=True, text=True, timeout=60)
# the ssh login banner is ~30 '#' lines — strip it before parsing
out = "\\n".join(l for l in r.stdout.splitlines()
                 if l.strip() and not l.strip().startswith("#"))
lines = [l for l in out.splitlines() if re.match(r"ep\\s+\\d+", l)]
for l in lines[-8:]:
    print(l)
if not lines:
    last = [l for l in out.splitlines() if l.strip()][-2:]
    print("no epochs yet —", " | ".join(last) if last else "log empty")
if "saved ->" in out:
    print("\\nDONE — final line:")
    for l in out.splitlines():
        if "saved" in l:
            print(l)'''

nb["cells"][24] = {"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [],
                   "source": SRC.splitlines(keepends=True)}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("progress cell fixed")
