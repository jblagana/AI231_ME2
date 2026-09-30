import json, wave
from pathlib import Path
import numpy as np
OUT = Path("runs/v1a2_wrong_audio"); OUT.mkdir(exist_ok=True)
samples = json.loads(Path("runs/v1a2_wrong_clips.json").read_text())
n = 0
for s in samples:
    f = s["file"].split("data/raw_v1i/", 1)[1].replace(".mp3", ".raw")
    raw = Path("runs/_raw_fetch") / f
    if not raw.exists():
        print("MISSING", f); continue
    arr = np.fromfile(str(raw), dtype=np.int16)
    kind = s["kind"]
    tag = s["cls"] if kind == "slot" else f"{s['true']}__to__{s['pred']}"
    wav_p = OUT / f"{kind}_{tag}_{f.split('/')[-1].replace('.raw', '.wav')}"
    with wave.open(str(wav_p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(arr.tobytes())
    n += 1
print(n, "wavs written")
