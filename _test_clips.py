# Local harness test for the pre-recorded-clip patch (no Pi needed):
# import pi/me2_ui.py with the audio side stubbed, drive apply_fire for
# every command, and assert the segment lists speak_clips would play.
# Run:  python _test_clips.py
import importlib.util
import sys
import types
from pathlib import Path

import wave

HERE = Path(__file__).parent
UI = HERE / "pi" / "me2_ui.py"
CLIPS = HERE / "pi_clips"

# point CLIP_DIR at the local clip dir after import (it's computed at
# module level from __file__ — override it).
spec = importlib.util.spec_from_file_location("me2_ui", UI)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.CLIP_DIR = CLIPS

# stub the audio side: capture segments instead of spawning workers
CAPTURED = []
m._speak_spawn = lambda segs, est: (CAPTURED.append((list(segs), est)), True)[1]
m._speak_synth = lambda text: f"<synth:{text}>"
m._music_play = lambda t: True
m._music_pause = lambda: True
m._music_resume = lambda: True
m._music_kill = lambda: None
m._apply_alsa_volume = lambda v: None
m._apply_alsa_volume.__name__ = "_apply_alsa_volume"

m.DEV["music"]["track"] = "song.mp3"
m.DEV["music"]["playing"] = True
m.DEV["music"]["paused"] = False
m.STATE["weather"] = {"icon": "☀️", "temp": 30.0, "condition": "sunny", "ts": 0}
m.STATE["weather_online"] = True

failures = []


def check(label, segs, want_clips, want_tail):
    wav_paths = [s[0] for s in segs]
    got_clips = [Path(p).name for p in wav_paths if p and p.startswith(str(CLIPS))]
    got_tail = [s[1] for s in segs if s[1] is not None]
    want_tails = [want_tail] if isinstance(want_tail, str) else []
    ok = got_clips == want_clips and got_tail == want_tails
    print(f"{'ok  ' if ok else 'FAIL'} {label:22s} clips={got_clips} tail={got_tail!r}")
    if not ok:
        failures.append(label)


def fire(cmd, slot=None):
    CAPTURED.clear()
    m._LAST_FIRE.clear()  # bypass the 2 s actuation debounce
    m.apply_fire({"cmd": cmd, "slot": slot, "conf": 0.99})
    return CAPTURED[0][0] if CAPTURED else []


# static clips
for cmd, slot, clip in [
    ("PLAY_MUSIC", None, "play_music.wav"),
    ("PAUSE", None, "pause.wav"),
    ("RESUME", None, "resume.wav"),
    ("NEXT", None, "next.wav"),
    ("STOP", None, "stop.wav"),
    ("VOLUME_UP", None, "volume_up.wav"),
    ("VOLUME_DOWN", None, "volume_down.wav"),
    ("LIGHT_ON", None, "light_on.wav"),
    ("LIGHT_OFF", None, "light_off.wav"),
    ("BRIGHTNESS", "20 percent", "brightness_20.wav"),
    ("BRIGHTNESS", "100 percent", "brightness_100.wav"),
    ("COLOR", "red", "color_red.wav"),
    ("COLOR", "green", "color_green.wav"),
    ("TEMPERATURE", "18 degrees", "temp_18.wav"),
    ("TIMER", "1 minute", "timer_60.wav"),
    ("ALARM", "9 PM", "alarm_9pm.wav"),
    ("CALL", None, "call.wav"),
    ("MESSAGE", None, "message.wav"),
    ("CREATE_REMINDER", "study", "reminder_study.wav"),
    ("OUT_OF_SCOPE", None, "oos.wav"),
]:
    segs = fire(cmd, slot)
    check(cmd + (f":{slot}" if slot else ""), segs, [clip], None)

# time: lead-in clip + TTS tail (live clock)
segs = fire("TIME", None)
check("TIME", segs, ["time_lead.wav"], m._time_reply()[1])

# weather online: lead-in + TTS tail
segs = fire("WEATHER", None)
check("WEATHER:online", segs, ["weather_lead.wav"],
      "Sunny, thirty degrees")

# weather offline + stale cache: lead + apology + stale-lead + tail
m.STATE["weather_online"] = False
segs = fire("WEATHER", None)
check("WEATHER:stale", segs,
      ["weather_lead.wav", "no_internet.wav", "stale_lead.wav"],
      "Sunny, thirty degrees")

# weather offline, no cache: lead + apology, no tail
m.STATE["weather"] = None
segs = fire("WEATHER", None)
check("WEATHER:nocache", segs,
      ["weather_lead.wav", "no_internet.wav"], None)

# fallback: missing clip file -> plain TTS
m.CLIP_DIR = HERE / "no_such_dir"
segs = fire("LIGHT_ON")
check("LIGHT_ON:fallback", segs, [], None)
m.CLIP_DIR = CLIPS

# guard scaling: 3-clip weather reply must exceed the old 6.5 s window
m.STATE["weather"] = {"icon": "☀️", "temp": 30.0, "condition": "sunny", "ts": 0}
m.STATE["weather_online"] = False
CAPTURED.clear()
m.apply_fire({"cmd": "WEATHER", "slot": None, "conf": 0.99})
est = CAPTURED[0][1]
print(f"{'ok  ' if est > 6.5 else 'FAIL'} guard est={est:.1f}s (> 6.5 fixed)")
if est <= 6.5:
    failures.append("guard")

# all clips are 48k mono s16
for f in sorted(CLIPS.glob("*.wav")):
    with wave.open(str(f), "rb") as wf:
        assert wf.getframerate() == 48000 and wf.getnchannels() == 1 \
            and wf.getsampwidth() == 2, f
print(f"ok   all {len(list(CLIPS.glob('*.wav')))} clips 48k mono s16")

print()
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("ALL PASS")
