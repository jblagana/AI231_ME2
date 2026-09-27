"""Fix cell 22 (section 9.1): H6 row exact text + Net effect with measured numbers."""
import json
from pathlib import Path

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
c = nb["cells"][22]
src = "".join(c["source"])

OLD_H6 = "| H6 | `nohup ... &` over ssh hangs the ssh session 120 s (tool timeout) | ssh waits for the child's stdout/stderr fds to close | redirect the nohup'd command's fds to files/`/dev/null` and verify with a *separate* ssh (`pgrep`) — the job itself is fine |"
NEW_H6 = "| H6 | `nohup ... &` over ssh hangs the launch call (90-120 s timeout) — hit twice, once in the notebook cell itself | ssh holds the channel open on the background job's **inherited stdin** (redirecting stdout/stderr alone is not enough — measured) | `nohup ... > /dev/null 2>&1 < /dev/null &` + verify with a separate ssh (`pgrep -f 'python src/train.py --epochs'` — the strict pattern matters, the wrapper's own cmdline matches the loose one) |"
assert OLD_H6 in src
src = src.replace(OLD_H6, NEW_H6)

OLD_NET = """**Net effect (measured):** epoch data-prep ~9 min → ~15 s; A100 fwd+bwd
14.9 s/epoch. 15 epochs + evals should be **~10-15 min total** including
the one-time ~10 min feature precompute, vs ~19 min per run on the laptop
CPU — and the A100 has 100x headroom for bigger experiments (batch 128,
more epochs, bigger models) that were impossible on the laptop."""
NEW_NET = """**Net effect (measured, v1d run 2026-09-28):** one-time base-feature
precompute 597 s (~10 min); then **2.6 s/epoch** on the A100 (vs ~75 s/epoch
CPU on the laptop — 29x). v1d (class-weighted CE, warm-started from v1c)
finished all 15 epochs in ~13 min wall time: **final eval_acc 0.629**
(peak 0.630) vs v1c's 0.363 on the same data — the class-weights flag is
worth +0.27. The A100 also has 100x headroom for the next experiments
(batch 128, more epochs, bigger models) that were impossible on the laptop."""
assert OLD_NET in src
src = src.replace(OLD_NET, NEW_NET)

c["source"] = src.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("cell 22 updated")
