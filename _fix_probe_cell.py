"""Rewrite the probe cell (21) with a quoting-proof command, then re-run it."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

SRC = '''# H9 · node probe (re-run anytime to see which node is free)
import subprocess, re

def probe(host):
    cmd = ("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; "
           "grep -m1 Mem /proc/meminfo | tr -s ' '")
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
         f"jan.rhey.lagana@{host}.ai.internal", cmd],
        capture_output=True, text=True, timeout=60)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    gpus = [l for l in lines if re.match(r"^\\d+%$", l.strip())]
    busy = sum(1 for g in gpus if int(g.strip().rstrip("%")) > 5)
    mem = next((l for l in lines if l.strip().startswith("MemTotal")), "?")
    return f"{host}: {busy}/8 GPUs busy, {mem}"

for h in ("n002", "n003"):
    try:
        print(probe(h))
    except Exception as e:
        print(f"{h}: probe failed ({e})")'''

nb["cells"][21] = {"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [],
                   "source": SRC.splitlines(keepends=True)}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("probe cell rewritten")
