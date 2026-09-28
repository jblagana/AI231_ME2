"""Fill manifest gaps left by failed edge-tts synths (v1f, 2026-09-28).

make_dataset.py's resume logic is FILE-based: a combo that failed all
retries leaves a hole in the per-class counter, and a plain re-run just
retries the SAME (voice, phrase, rate) — which fails again when the cause
is a broken voice (v1f: en-IE-EmilyNeural, en-NZ-MollyNeural, en-PH-RosaNeural
lost whole 25-combo blocks to "No audio was received").

This script instead:
  1. reads the manifest,
  2. for each (split, class) derives the FULL expected combo sequence from
     commands.PHRASES (20 voices x phrases x 5 rates, same order as
     make_dataset.py),
  3. finds combos whose expected filename is absent from the manifest,
  4. probes each voice involved in a missing combo (one test synth);
     a dead voice is substituted with a working voice from the SAME split
     (speaker-disjoint property preserved — the replacement is a
     train-only or eval-only voice either way),
  5. synthesizes each missing combo at the NEXT FREE index (no holes,
     never clobbers orphan mp3s from the original decode pass — issue D3),
     8 retries with backoff,
  6. appends to the manifest in the same format.

Usage:
  python src/fill_gaps.py            # fill everything missing
  python src/fill_gaps.py --dry      # list missing combos only
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES, PHRASES  # noqa: E402
import make_dataset  # noqa: E402
from make_dataset import VOICES, RATES, synth_one  # noqa: E402

make_dataset.SEM = asyncio.Semaphore(32)  # gaps are bursty; go faster

MANIFEST = Path("data/raw/manifest.jsonl")
PROBE_TEXT = "play music"  # short, known-good phrase for the voice probe


def expected_combos(split: str, cls: str):
    """Same (voice, phrase, rate) order as make_dataset.py's main loop."""
    voices = VOICES[:20] if split == "train" else VOICES[20:]
    out = []
    for voice in voices:
        for phrase in PHRASES[cls]:
            for rate in RATES:
                out.append((voice, phrase, rate))
    return out


async def probe_voice(voice: str, tmp: Path) -> bool:
    """One test synth; True if the voice actually produces audio."""
    return await synth_one(voice, PROBE_TEXT, "+0%", tmp, retries=2)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    man = {}
    for line in MANIFEST.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        man[e["file"]] = e

    missing = {}  # (split, cls) -> [(voice, phrase, rate)]
    for split in ("train", "eval"):
        for cls in CLASSES:
            have = {f.split("/")[-1] for f in man
                    if f.startswith(f"{split}/{cls}/")}
            combos = expected_combos(split, cls)
            holes = [(v, p, r) for i, (v, p, r) in enumerate(combos)
                     if f"{i+1:05d}.mp3" not in have]
            if holes:
                missing[(split, cls)] = holes

    total = sum(len(v) for v in missing.values())
    print(f"missing combos: {total}")
    for k in sorted(missing):
        print(f"  {k[0]}/{k[1]}: {len(missing[k])}")
    if args.dry:
        return 0

    # ---- voice health check for every voice implicated in a hole ----
    implicated = sorted({v for holes in missing.values()
                         for v, _, _ in holes})
    print(f"probing {len(implicated)} implicated voices...")
    tmp = MANIFEST.parent / "_probe.mp3"
    dead = {}
    for v in implicated:
        ok = await probe_voice(v, tmp)
        tmp.unlink(missing_ok=True)
        dead[v] = not ok
        print(f"  {v}: {'OK' if ok else 'DEAD'}", flush=True)

    voice_pool = {
        "train": [v for v in VOICES[:20] if not dead.get(v)],
        "eval": [v for v in VOICES[20:] if not dead.get(v)],
    }
    sub_count = 0
    n_ok = n_fail = 0
    with MANIFEST.open("a", encoding="utf-8") as mf:
        for (split, cls), holes in sorted(missing.items()):
            d = MANIFEST.parent / split / cls
            d.mkdir(parents=True, exist_ok=True)
            # include DISK files too: the original decode pass left 13 orphan
            # mp3s (on disk, not in manifest — issue D3); never clobber them
            used = {f.split("/")[-1] for f in man
                    if f.startswith(f"{split}/{cls}/")}
            used |= {p.name for p in d.glob("*.mp3")}
            for voice, phrase, rate in holes:
                v = voice
                if dead.get(v):
                    v = voice_pool[split][n_fail % max(len(voice_pool[split]), 1)]
                    sub_count += 1
                i = 1
                while f"{i:05d}.mp3" in used:
                    i += 1
                p = d / f"{i:05d}.mp3"
                ok = await synth_one(v, phrase, rate, p, retries=8)
                if ok:
                    n_ok += 1
                    used.add(p.name)
                    mf.write(json.dumps({
                        "file": f"{split}/{cls}/{p.name}",
                        "class": cls, "voice": v,
                        "phrase": phrase, "rate": rate,
                        "split": split,
                    }) + "\n")
                else:
                    n_fail += 1
            print(f"[{split}] {cls}: filled {len(holes)} "
                  f"(cum ok={n_ok} fail={n_fail} sub={sub_count})", flush=True)

    print(f"DONE: ok={n_ok} fail={n_fail} voice_subs={sub_count}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
