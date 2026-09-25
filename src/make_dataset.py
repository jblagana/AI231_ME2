"""Generate the VCM training set with edge-tts (free Microsoft neural TTS).

Strategy (PLAN.md):
- 10 classes x ~6 phrases x many voices x speed jitter
- Speaker-disjoint split: voices are partitioned train/eval BEFORE generation,
  so the model can't memorize voices.
- Output: data/raw/{split}/{class}/NNNNN.mp3  +  data/raw/manifest.jsonl

Usage:
  python src/make_dataset.py [--voices-per-split 20] [--out data/raw]

edge-tts voices (en-US/en-GB/en-AU/en-IN/en-CA mix for accent diversity):
  male:  Guy, Christopher, Eric, Brian, Thomas, Ryan, Brandon, Andrew, Aaron,
         William, David, Mark, George, Roger, Todd, Paul, EricNeural, ...
  female: Jenny, Aria, Ava, Emma, Mia, Chloe, Lily, Olivia, Hannah, Susan,
         Michelle, Erica, Anna, EmmaNeural, ...
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from commands import CLASSES, PHRASES  # noqa: E402

# 40 voices across 10 English accents (US, GB, AU, IN, CA, ZA, IE, NZ, PH, SG/NG/TZ).
# First half -> train, second half -> eval (deterministic, speaker-disjoint).
# Verified against live edge-tts voice list (2026-09-25, 322 voices available).
VOICES = [
    # train voices (20)
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-ChristopherNeural",
    "en-US-EricNeural", "en-US-AriaNeural", "en-GB-RyanNeural",
    "en-GB-SoniaNeural", "en-GB-ThomasNeural", "en-AU-WilliamMultilingualNeural",
    "en-AU-NatashaNeural", "en-IN-PrabhatNeural", "en-IN-NeerjaNeural",
    "en-CA-LiamNeural", "en-CA-ClaraNeural", "en-ZA-LukeNeural",
    "en-ZA-LeahNeural", "en-IE-ConnorNeural", "en-NZ-MitchellNeural",
    "en-PH-JamesNeural", "en-SG-LunaNeural",
    # eval voices (20) — never seen in training
    "en-US-AndrewNeural", "en-US-AvaNeural", "en-US-BrianNeural",
    "en-US-EmmaNeural", "en-US-MichelleNeural", "en-US-RogerNeural",
    "en-US-SteffanNeural", "en-US-AnaNeural", "en-GB-LibbyNeural",
    "en-GB-MaisieNeural", "en-IN-NeerjaExpressiveNeural",
    "en-NZ-MollyNeural", "en-IE-EmilyNeural", "en-PH-RosaNeural",
    "en-SG-WayneNeural", "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    "en-TZ-ElimuNeural", "en-TZ-ImaniNeural", "en-HK-SamNeural",
]

# edge-tts rate format: "-10%" .. "+10%" (note: bare "0%" is REJECTED — must be "+0%")
RATES = ["-10%", "-5%", "+0%", "+5%", "+10%"]


SEM = asyncio.Semaphore(16)  # 16 concurrent TTS requests (measured: ~1 req/s per stream, 16 streams ~ 240 clips/min)


async def synth_one(voice, text, rate, out_mp3: Path, retries=3):
    import edge_tts
    async with SEM:
        for attempt in range(retries):
            try:
                tts = edge_tts.Communicate(text, voice, rate=rate)
                await tts.save(str(out_mp3))
                if out_mp3.exists() and out_mp3.stat().st_size > 500:
                    return True
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  FAIL {voice} {text!r}: {e}", file=sys.stderr)
                await asyncio.sleep(1 + attempt)
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--per-combo", type=int, default=1,
                    help="clips per (voice, phrase, rate) combo")
    ap.add_argument("--limit-voices", type=int, default=0,
                    help="debug: only first N voices of each split")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {"train": VOICES[:20], "eval": VOICES[20:]}
    if args.limit_voices:
        splits = {k: v[: args.limit_voices] for k, v in splits.items()}

    # Resumable: if a manifest exists, load it and skip already-generated
    # files (restart-safe; the old "w" mode wiped it and forced full regen).
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a crash — drop it
                done.add(e["file"])
        print(f"RESUME: {len(done)} clips already in manifest, skipping them")
    n_ok = len(done)
    n_fail = 0
    with manifest.open("a", encoding="utf-8") as mf:
        for split, voices in splits.items():
            for cls in CLASSES:
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                i = 0
                for voice in voices:
                    for phrase in PHRASES[cls]:
                        for rate in RATES:
                            for _ in range(args.per_combo):
                                i += 1
                                p = d / f"{i:05d}.mp3"
                                if f"{split}/{cls}/{p.name}" in done:
                                    continue  # already generated (resume)
                                ok = await synth_one(voice, phrase, rate, p)
                                if ok:
                                    n_ok += 1
                                    mf.write(json.dumps({
                                        "file": f"{split}/{cls}/{p.name}",
                                        "class": cls, "voice": voice,
                                        "phrase": phrase, "rate": rate,
                                        "split": split,
                                    }) + "\n")
                                else:
                                    n_fail += 1
                print(f"[{split}] {cls}: {i} clips (cum ok={n_ok} fail={n_fail})", flush=True)

    print(f"DONE: {n_ok} ok, {n_fail} failed")


if __name__ == "__main__":
    asyncio.run(main())
