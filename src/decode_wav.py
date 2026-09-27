"""One-time batch decode: data/raw/{split}/{class}/*.mp3 -> sibling .raw
(16 kHz mono s16le, no header) via the imageio_ffmpeg binary.

Why: train.py's per-item ffmpeg subprocess costs ~0.19 s/item (measured
_probe_throughput.py, 2026-09-27) -> ~450 min/epoch. torchaudio.load is
ALSO broken in this venv (torchcodec ImportError, measured _probe_wavload.py),
so the fast load path is np.fromfile on raw s16le: ~4 ms/item (measured).

Usage:
  python src/decode_wav.py            # decode everything missing
  python src/decode_wav.py --smoke    # 20 files only (pipeline check)
Idempotent: existing .raw files are skipped.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()  # same binary gen_site_assets.load_mp3 uses


def decode_one(mp3: Path, raw: Path) -> bool:
    q = subprocess.run(
        [FF, "-y", "-loglevel", "error", "-i", str(mp3),
         "-ar", "16000", "-ac", "1", "-f", "s16le", str(raw)],
        capture_output=True, timeout=60)
    return q.returncode == 0 and raw.exists() and raw.stat().st_size > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    mf = root / "manifest.jsonl"
    rows = [json.loads(l) for l in mf.open(encoding="utf-8")]
    if args.smoke:
        rows = rows[:20]
    todo = []
    for r in rows:
        mp3 = root / r["file"]
        raw = mp3.with_suffix(".raw")
        if mp3.exists() and not raw.exists():
            todo.append((mp3, raw))
    print(f"manifest={len(rows)}  need_decode={len(todo)}", flush=True)
    if not todo:
        print("nothing to do"); return 0

    t0 = time.time()
    ok = fail = 0
    for k, (mp3, raw) in enumerate(todo):
        try:
            if decode_one(mp3, raw):
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {mp3}", flush=True)
        except Exception as e:
            fail += 1
            print(f"  FAIL {mp3}: {e}", flush=True)
        if (k + 1) % 200 == 0:
            rate = (k + 1) / (time.time() - t0)
            eta = (len(todo) - k - 1) / max(rate, 1e-9)
            print(f"  {k+1}/{len(todo)} ok={ok} fail={fail} "
                  f"{rate:.1f}/s eta={eta/60:.1f}min", flush=True)
    print(f"done: ok={ok} fail={fail} in {time.time()-t0:.0f}s", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
