"""Build a REAL-HUMAN eval set from SLURP (cross-domain probe for the VCM).

Why this exists
---------------
v1e/v1f are trained on edge-tts (synthetic) and evaluated on edge-tts (synthetic).
The 0.665 full-eval number is a TTS-on-TTS number. SLURP is a real-human,
real-room smart-home corpus — feeding it through the TTS-trained model gives the
CROSS-DOMAIN accuracy number: how well does the model actually hold up on real
voices, the thing Sir's ruling ("must respond to ANY person") is about.

This script does the CHEAP part only: it downloads the SLURP *text* annotations
(test.jsonl + devel.jsonl — both held-out from SLURP's train split), maps each
entry's `intent` onto one of our 6 covered VCM classes, and selects a balanced
~500-clip eval set. It writes:

    data/eval_slurp/manifest.csv     file,class,slurp_intent,split
    data/eval_slurp/needed_flac.txt  one FLAC basename per line (for extraction)

It does NOT download the 3.9GB audio — that's a separate step (see
HANDOVER / the run notes). The audio tarball is slurp_real.tar.gz from
Zenodo record 4274930; the FLAC basenames here are the exact files to extract.

Mapping (strict-ish, documented)
--------------------------------
Our 10 classes; SLURP covers 6 of them (verified against commands.py 2026-09-29).
The 4 gaps (set_timer, set_temperature, set_reminder, make_call) have no SLURP
intent and are NOT in this eval set — they need Timers-and-Such / Fluent / Ayla.

    play_music    <- play_music
    ask_question  <- weather_query, datetime_query, datetime_convert,
                     qa_factoid, qa_definition, qa_currency, qa_stock,
                     qa_maths, news_query
    control_lights<- iot_hue_lighton, iot_hue_lightoff, iot_wemo_on,
                     iot_wemo_off, hue_lighton, hue_lightoff
    dim_lights    <- iot_hue_lightdim, iot_hue_lightchange, iot_hue_lightup,
                     hue_lightdim, hue_lightup
    set_alarm     <- alarm_set, alarm_query, alarm_remove
    media_control <- audio_volume_mute, audio_volume_up, audio_volume_down,
                     audio_volume_other

Each SLURP entry has N recordings (headset + free-field variants). We pick ONE
recording per entry (deterministic) so each clip is a distinct utterance, not a
dup under a different mic.

Usage
-----
    python src/make_slurp_eval.py                 # build manifest (text only)
    python src/make_slurp_eval.py --target 500    # total clips to select
    python src/make_slurp_eval.py --seed 0        # deterministic selection
"""
import argparse
import csv
import json
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "eval_slurp"
RAW = REPO / "data" / "slurp_text"  # cache the small jsonl downloads here

SLURP_BASE = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/"
SPLITS = ["test.jsonl", "devel.jsonl"]  # both held-out from SLURP train

# Our class -> the SLURP intents that genuinely mean the same task.
INTENT_TO_CLASS = {
    "play_music": ["play_music"],
    "ask_question": [
        "weather_query", "datetime_query", "datetime_convert",
        "qa_factoid", "qa_definition", "qa_currency", "qa_stock",
        "qa_maths", "news_query",
    ],
    "control_lights": [
        "iot_hue_lighton", "iot_hue_lightoff", "iot_wemo_on", "iot_wemo_off",
        "hue_lighton", "hue_lightoff",
    ],
    "dim_lights": [
        "iot_hue_lightdim", "iot_hue_lightchange", "iot_hue_lightup",
        "hue_lightdim", "hue_lightup",
    ],
    "set_alarm": ["alarm_set", "alarm_query", "alarm_remove"],
    "media_control": [
        "audio_volume_mute", "audio_volume_up", "audio_volume_down",
        "audio_volume_other",
    ],
}
CLASS_TO_INTENTS = {c: set(v) for c, v in INTENT_TO_CLASS.items()}


def fetch(split_name: str) -> list:
    """Download + parse one SLURP jsonl split (cached to data/slurp_text/)."""
    cache = RAW / split_name
    if not cache.exists():
        print(f"  downloading {split_name} ...", flush=True)
        data = urllib.request.urlopen(SLURP_BASE + split_name, timeout=60).read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    recs = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    print(f"  {split_name}: {len(recs)} entries", flush=True)
    return recs


def pick_recording(rec: dict) -> str:
    """One FLAC basename per entry (deterministic): prefer free-field (no
    '-headset'), else the first. Keeps each clip a distinct utterance."""
    files = [r["file"] for r in rec.get("recordings", [])]
    if not files:
        return ""
    free = [f for f in files if "headset" not in f]
    return (free or files)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="total clips to select")
    ap.add_argument("--seed", type=int, default=0, help="deterministic selection")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("== SLURP text (held-out splits) ==")
    all_recs = []
    for s in SPLITS:
        all_recs.extend((s[:-6], r) for r in fetch(s))  # strip .jsonl
    print(f"  total held-out entries: {len(all_recs)}")

    # Bucket entries by our class.
    by_class = defaultdict(list)
    unmapped = Counter()
    for split, rec in all_recs:
        intent = rec.get("intent", "")
        cls = next((c for c, iv in CLASS_TO_INTENTS.items() if intent in iv), None)
        if cls is None:
            unmapped[intent] += 1
            continue
        flac = pick_recording(rec)
        if not flac:
            continue
        by_class[cls].append((flac, intent, split))

    # Availability report.
    print("\n== availability per class (held-out, 1 clip/entry) ==")
    for c in INTENT_TO_CLASS:
        print(f"  {c:16s} {len(by_class[c]):5d}")

    # Balanced selection: cap each class at ceil(target/nclasses), take all if short.
    n_classes = len(INTENT_TO_CLASS)
    cap = -(-args.target // n_classes)  # ceil division
    selected = []
    print(f"\n== selecting (cap {cap}/class, target {args.target}) ==")
    for c in INTENT_TO_CLASS:
        pool = by_class[c]
        rng.shuffle(pool)
        take = pool[:cap]
        selected.extend((flac, c, intent, split) for flac, intent, split in take)
        print(f"  {c:16s} {len(take):4d} / {len(pool)}")
    rng.shuffle(selected)

    # Dedupe by FLAC basename (an entry's recording could in theory repeat).
    seen = set()
    dedup = [row for row in selected if not (row[0] in seen or seen.add(row[0]))]

    # Write manifest + needed-flac list.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man = OUT_DIR / "manifest.csv"
    with man.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "slurp_intent", "split"])
        for flac, cls, intent, split in dedup:
            w.writerow([flac, cls, intent, split])
    (OUT_DIR / "needed_flac.txt").write_text("\n".join(sorted({r[0] for r in dedup})) + "\n")

    dist = Counter(r[1] for r in dedup)
    print(f"\n== manifest: {len(dedup)} clips -> {man} ==")
    for c in INTENT_TO_CLASS:
        print(f"  {c:16s} {dist.get(c,0):4d}")
    print(f"\nunique FLAC files to extract: {len({r[0] for r in dedup})}")
    print("next: download slurp_real.tar.gz (Zenodo 4274930), extract needed_flac.txt, run src/eval_slurp.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
