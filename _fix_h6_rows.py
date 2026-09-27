"""Update the H6 row in both issues tables (section 1 + section 9.1) with the
refined root cause (inherited stdin, not stdout/stderr)."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

OLD_S1 = "| H6 | `nohup … &` over ssh hangs the session (120 s tool timeout) | ssh waits for the child's stdout/stderr fds | redirect fds to `/dev/null` + verify with a separate ssh (`pgrep`) — the job itself is fine |"
NEW_S1 = "| H6 | `nohup … &` over ssh hangs the launch call (90-120 s timeout) | ssh holds the channel open on the background job's **inherited stdin** (not just stdout/stderr) | `nohup … > /dev/null 2>&1 < /dev/null &` + verify with a separate ssh (`pgrep`) — the job itself is fine |"

OLD_91 = "| H6 | `nohup … &` over ssh hangs the session (120 s tool timeout) | ssh waits for the child's stdout/stderr fds | redirect fds to `/dev/null` + verify with a separate ssh (`pgrep`) — the job itself is fine |"
NEW_91 = "| H6 | `nohup … &` over ssh hangs the launch call (90-120 s timeout) — hit twice, once in the notebook cell itself | ssh holds the channel open on the background job's **inherited stdin** (redirecting stdout/stderr alone is not enough — measured) | `nohup … > /dev/null 2>&1 < /dev/null &` + verify with a separate ssh (`pgrep -f 'python src/train.py --epochs'` — the strict pattern matters, the wrapper's own cmdline matches the loose one) |"

hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c["source"])
    if OLD_S1 in src:
        src = src.replace(OLD_S1, NEW_S1); hits += 1
    if OLD_91 in src:
        src = src.replace(OLD_91, NEW_91); hits += 1
    c["source"] = src.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("H6 rows updated:", hits)
