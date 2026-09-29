"""Build a tiny SYNTHETIC 11-class dataset to smoke-test the A2 pipeline
locally (no GPU / no n002 needed). Generates .raw s16le files (random
band-limited noise shaped like speech envelopes) + manifest.jsonl using the
REAL phrases per class, so slots.extract_slot runs on real phrases.

NOT a real dataset — just enough to exercise the data path, slot labels,
forward pass, dual loss, and eval. Run on n002 against data/raw_v1i for the
real thing.
"""
import json
import random
from pathlib import Path

import numpy as np

SR = 16000
CLASSES = [
    "play_music", "ask_weather", "control_lights", "dim_lights",
    "set_timer", "set_alarm", "set_temperature", "media_control",
    "set_reminder", "make_call", "ask_time",
]
# real phrases (subset) per class — must match the manifest vocab
PHRASES = {
    "play_music": ["play music", "play some music", "play a song"],
    "ask_weather": ["what's the weather", "what is the weather", "tell me the weather"],
    "control_lights": ["turn on the lights", "turn off the lights", "lights on"],
    "dim_lights": ["dim the lights to fifty percent", "dim the lights to eighty percent",
                   "dim the lights to thirty percent", "dim the lights", "lower the lights"],
    "set_timer": ["set a timer for five minutes", "set a timer for ten minutes",
                  "set a timer for one minute", "set a timer for thirty minutes"],
    "set_alarm": ["set an alarm for six am", "set an alarm for seven am",
                  "set an alarm for five thirty am", "wake me up at six am"],
    "set_temperature": ["set the temperature to twenty two degrees",
                        "set the temperature to twenty five degrees",
                        "set the thermostat to twenty degrees"],
    "media_control": ["pause", "stop", "next song", "volume up", "volume down"],
    "set_reminder": ["remind me to buy groceries", "remind me to call mom",
                     "remind me at five pm", "add a reminder to water the plants",
                     "remind me about the meeting", "set a reminder for tomorrow"],
    "make_call": ["call mom", "call dad", "call my friend", "call the doctor",
                  "give mom a call", "phone my dad"],
    "ask_time": ["what time is it", "what day is it", "what's today's date"],
}
N_PER = 4  # clips per (class, phrase)
OUT = Path("data/a2_smoke")


def synth_clip(dur_s, seed):
    rng = np.random.RandomState(seed)
    n = int(dur_s * SR)
    # speech-like: amplitude-modulated band-limited noise, with a trailing
    # silent gap so the clip is < 3.0 s (right-alignment test)
    t = np.arange(n) / SR
    f0 = rng.uniform(120, 320)
    env = np.exp(-3 * (t / max(dur_s, 1e-3)))  # decaying envelope
    sig = (np.sin(2 * np.pi * f0 * t) * 0.5
           + rng.randn(n) * 0.5) * env
    sig = sig / (np.abs(sig).max() + 1e-8) * 0.4
    return (sig * 32767).astype(np.int16)


def main():
    if OUT.exists():
        for f in OUT.rglob("*"):
            if f.is_file():
                f.unlink()
    mf = []
    for cls in CLASSES:
        for split in ["train", "eval"]:
            d = OUT / split / cls
            d.mkdir(parents=True, exist_ok=True)
            i = 0
            for phrase in PHRASES[cls]:
                for k in range(N_PER):
                    i += 1
                    # realistic duration 0.5-2.5 s
                    dur = random.uniform(0.5, 2.5)
                    seed = hash((cls, phrase, split, k)) % (2**31)
                    arr = synth_clip(dur, seed)
                    name = f"{i:05d}.raw"
                    (d / name).write_bytes(arr.tobytes())
                    mf.append({"file": f"{split}/{cls}/{name}", "class": cls,
                               "phrase": phrase, "voice": "synth",
                               "rate": "+0%", "split": split})
    (OUT / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in mf) + "\n")
    print(f"wrote {len(mf)} clips -> {OUT}")
    # slot coverage check
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from slots import extract_slot, PARAMETRIC
    n_par = n_slot = 0
    for r in mf:
        if r["class"] in PARAMETRIC:
            n_par += 1
            _, idx = extract_slot(r["class"], r["phrase"])
            if idx >= 0:
                n_slot += 1
    print(f"slot coverage: {n_slot}/{n_par} parametric clips have a slot")


if __name__ == "__main__":
    main()
