"""Fetch A2 wrong-clip .raw files from n002 and convert to 16 kHz WAV for listening."""
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent
OUT = REPO / "runs" / "v1a2_wrong_audio"
OUT.mkdir(parents=True, exist_ok=True)

samples = json.loads((REPO / "runs" / "v1a2_wrong_clips.json").read_text())
# take all 42 (they're small)
files = sorted({s["file"] for s in samples})
print(f"{len(files)} unique clips")

# one scp for all
remote = "n002.ai.internal"
remote_root = "/mnt/jfs_hpc/home/jan.rhey.lagana/vcm/"
local_tmp = REPO / "runs" / "_raw_fetch"
local_tmp.mkdir(exist_ok=True)

# scp each (fewer round trips: use a single tar over ssh)
rel = " ".join(f.split("data/raw_v1i/", 1)[1] for f in files)
cmd = (f"cd /mnt/jfs_hpc/home/jan.rhey.lagana/vcm/data/raw_v1i && "
       f"tar cf - {rel} | cat")
proc = subprocess.run(["ssh", remote, cmd], capture_output=True)
assert proc.returncode == 0, proc.stderr.decode()[:500]
tar_path = local_tmp / "clips.tar"
tar_path.write_bytes(proc.stdout)
subprocess.run(["tar", "-xf", str(tar_path), "-C", str(local_tmp)], check=True)

# convert each .raw -> .wav (s16le mono 16k)
for s in samples:
    f = s["file"].split("data/raw_v1i/", 1)[1]
    raw = local_tmp / f.replace(".mp3", ".raw")
    if not raw.exists():
        print(f"MISSING {raw}")
        continue
    arr = np.fromfile(str(raw), dtype=np.int16)
    tag = f"{s['cls']}__{s['true'].replace(' ', '_')}__pred_{s['pred'].replace(' ', '_')}"
    wav_p = OUT / f"{tag}_{f.split('/')[-1].replace('.mp3', '.wav')}"
    with wave.open(str(wav_p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(arr.tobytes())
    print(f"{wav_p.name}  ({len(arr)/16000:.2f}s)  {s['phrase']!r}  "
          f"true={s['true']} pred={s['pred']}")
print(f"\n{len(list(OUT.glob('*.wav')))} wavs in {OUT}")
