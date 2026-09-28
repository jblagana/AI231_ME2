"""Reconcile the v1f dataset to EXACTLY one clip per intended combo.

Why: fill_gaps.py (v1f) had an index bug — it wrote a missing combo to the
next FREE filename instead of the combo's OWN index, so each re-run re-detected
the same empty index and appended a DUPLICATE instead of filling it. End state:
5384 combos present twice, 5356 intended combos absent (17,444 unique of
22,800). This script fixes both by matching on the (split,class,voice,phrase,
rate) TUPLE, never the filename index.

Steps (idempotent — safe to re-run after an interruption):
  1. build the intended combo set from commands.PHRASES x VOICES x RATES
     (22,800 combos: 11,400/split),
  2. read the manifest, group rows by combo tuple,
  3. for each intended combo: keep the FIRST row whose file exists & is
     non-empty; drop the rest (duplicate rows removed from the manifest;
     their files are left on disk as harmless orphans — training is
     manifest-driven, so orphans are never loaded),
  4. synthesize every intended combo with no valid file, to a fresh free
     filename per class (6 retries),
  5. rewrite manifest.jsonl = kept rows + new rows (clean, no dupes).

Usage:
  python src/reconcile_dataset.py           # full reconcile + synthesize
  python src/reconcile_dataset.py --dry     # report only, no writes
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES, PHRASES  # noqa: E402
import make_dataset  # noqa: E402
from make_dataset import VOICES, RATES, synth_one  # noqa: E402

make_dataset.SEM = asyncio.Semaphore(32)

ROOT = Path("data/raw")
MANIFEST = ROOT / "manifest.jsonl"


def intended_combos():
    """All (split, class, voice, phrase, rate) the dataset should contain."""
    out = []
    for split, voices in (("train", VOICES[:20]), ("eval", VOICES[20:])):
        for cls in CLASSES:
            for voice in voices:
                for phrase in PHRASES[cls]:
                    for rate in RATES:
                        out.append((split, cls, voice, phrase, rate))
    return out


def read_manifest():
    rows = []
    seen_files = set()
    for line in MANIFEST.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d["file"] in seen_files:  # file-level dupe guard
            continue
        seen_files.add(d["file"])
        rows.append(d)
    return rows


def combo_of(d):
    return (d["split"], d["class"], d["voice"], d["phrase"], d["rate"])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    intend = intended_combos()
    intend_set = set(intend)
    rows = read_manifest()

    by_combo = defaultdict(list)
    for d in rows:
        by_combo[combo_of(d)].append(d)

    kept = []            # manifest rows to keep (one per combo, valid file)
    to_drop = []         # duplicate rows removed
    to_synth = []        # (split, cls, voice, phrase, rate) with no valid file
    for combo in intend:
        split, cls, voice, phrase, rate = combo
        group = by_combo.get(combo, [])
        valid = [d for d in group
                 if (ROOT / d["file"]).exists()
                 and (ROOT / d["file"]).stat().st_size > 0]
        valid.sort(key=lambda d: d["file"])
        if valid:
            kept.append(valid[0])
            to_drop.extend(d for d in group if d is not valid[0])
        else:
            to_synth.append(combo)

    extras = [d for d in rows if combo_of(d) not in intend_set]

    print(f"intended combos : {len(intend)}")
    print(f"manifest rows   : {len(rows)}  (unique combos {len(by_combo)})")
    print(f"keep (1/combo)  : {len(kept)}")
    print(f"drop dupes      : {len(to_drop)}")
    print(f"synthesize      : {len(to_synth)}")
    print(f"extra (not in intended): {len(extras)}")
    if args.dry:
        return 0

    # ---- synthesize the missing combos ----
    n_ok = n_fail = 0
    with MANIFEST.open("a", encoding="utf-8") as mf:
        for split, cls, voice, phrase, rate in to_synth:
            d = ROOT / split / cls
            d.mkdir(parents=True, exist_ok=True)
            used = {p.name for p in d.glob("*.mp3")}
            i = 1
            while f"{i:05d}.mp3" in used:
                i += 1
            p = d / f"{i:05d}.mp3"
            ok = await synth_one(voice, phrase, rate, p, retries=6)
            if ok and p.exists() and p.stat().st_size > 0:
                n_ok += 1
                used.add(p.name)
                kept.append({
                    "file": f"{split}/{cls}/{p.name}",
                    "class": cls, "voice": voice,
                    "phrase": phrase, "rate": rate, "split": split,
                })
            else:
                n_fail += 1
                p.unlink(missing_ok=True)
            if (n_ok + n_fail) % 200 == 0:
                print(f"  synth {n_ok + n_fail}/{len(to_synth)} "
                      f"ok={n_ok} fail={n_fail}", flush=True)

    # ---- rewrite the manifest clean (kept already includes new rows) ----
    kept.sort(key=lambda d: (d["split"], d["class"], d["file"]))
    MANIFEST.write_text(
        "\n".join(json.dumps(d) for d in kept) + "\n", encoding="utf-8")

    print(f"DONE: kept={len(kept)}  synthesized ok={n_ok} fail={n_fail} "
          f"dropped_dupes={len(to_drop)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
