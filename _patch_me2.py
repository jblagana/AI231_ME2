# One-shot: apply the pre-recorded-clip patch to pi/me2_ui.py (pulled from the
# live Pi). Run:  python _patch_me2.py  (syntax-checks + writes pi/me2_ui.py)
import ast
from pathlib import Path

P = Path(__file__).with_name("pi") / "me2_ui.py"
src = P.read_text(encoding="utf-8-sig")


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, f"anchor not unique ({src.count(old)}): {old[:70]!r}"
    src = src.replace(old, new)


# --- 1. clip registry + lookup, after _speak_synth -------------------------
rep('''        print(f"speak: synth failed: {e}", file=sys.stderr)
    return None
''', '''        print(f"speak: synth failed: {e}", file=sys.stderr)
    return None


# ---------------------------------------------------------------- clips
# Pre-recorded reply clips (en-US-RogerNeural, 48k mono s16 WAVs in
# ./clips/, generated 2026-10-05): instant, offline, zero synth latency.
# Every command fires its clip first; a missing clip file degrades to the
# plain-TTS fallback, never to silence.
CLIP_DIR = Path(__file__).resolve().parent / "clips"
CLIP = {
    "PLAY_MUSIC": "play_music", "PAUSE": "pause", "RESUME": "resume",
    "NEXT": "next", "STOP": "stop", "VOLUME_UP": "volume_up",
    "VOLUME_DOWN": "volume_down", "LIGHT_ON": "light_on",
    "LIGHT_OFF": "light_off", "CALL": "call", "MESSAGE": "message",
    "OUT_OF_SCOPE": "oos", "TIME": "time_lead", "WEATHER": "weather_lead",
    "BRIGHTNESS": {"20 percent": "brightness_20",
                   "60 percent": "brightness_60",
                   "100 percent": "brightness_100"},
    "COLOR": {"red": "color_red", "blue": "color_blue",
              "green": "color_green"},
    "TEMPERATURE": {"18 degrees": "temp_18", "22 degrees": "temp_22",
                    "26 degrees": "temp_26"},
    "TIMER": {"10 seconds": "timer_10", "30 seconds": "timer_30",
              "1 minute": "timer_60"},
    "ALARM": {"6 AM": "alarm_6am", "8 AM": "alarm_8am",
              "9 PM": "alarm_9pm"},
    "CREATE_REMINDER": {"drink water": "reminder_drink",
                        "study": "reminder_study",
                        "exercise": "reminder_exercise"},
}


def _clip_name(cmd: str, slot: str) -> str | None:
    """(cmd, slot) -> main clip base name (no .wav), or None => plain TTS."""
    m = CLIP.get(cmd)
    if m is None:
        return None
    if isinstance(m, dict):
        return m.get(slot)
    return m


def _clip_path(name: str) -> str | None:
    p = CLIP_DIR / f"{name}.wav"
    return str(p) if p.exists() else None
''')

# --- 2. _weather_reply: 4-case matrix --------------------------------------
rep('''    w = STATE.get("weather")
    if w and w.get("temp") is not None and STATE.get("weather_online"):
        cond = str(w.get("condition") or "all good").strip().capitalize()
        return (f"{cond}, {w['temp']:.0f}°",
                f"{cond}, {_int_words(w['temp'])} degrees")
    return ("Sorry, no internet connection.",
            "Sorry, no internet connection.")
''', '''    w = STATE.get("weather")
    if w and w.get("temp") is not None and STATE.get("weather_online"):
        cond = str(w.get("condition") or "all good").strip().capitalize()
        return (f"{cond}, {w['temp']:.0f}°",
                f"{cond}, {_int_words(w['temp'])} degrees")
    if w and w.get("temp") is not None:
        # Offline but the last fetch is <=15 min old: the caller speaks
        # apology + stale-lead clips, then this cached data as the tail.
        cond = str(w.get("condition") or "all good").strip().capitalize()
        return (f"{cond}, {w['temp']:.0f}° (15 min old)",
                f"{cond}, {_int_words(w['temp'])} degrees")
    return ("Sorry, no internet connection.",
            "Sorry, no internet connection.")
''')

# --- 3. speak_clips() before def speak -------------------------------------
rep('''def speak(text: str) -> bool:
    """Answer out loud on the M1A. Music ducks (pause -> speak -> resume).
    Best-effort: any failure just means no voice — the chip already has
    the text. Called from apply_fire (STATE["lock"] held); the critical
    sections here stay short (same note as the music section)."""
    import subprocess
    with _TTS["lock"]:
        p = _TTS["proc"]
        if p is not None and p.poll() is None:
            return False  # already talking — don't stack replies
        wav = _speak_synth(text)
        if wav is None:
            return False
        ducked = False
        if DEV["music"]["playing"] and not DEV["music"]["paused"]:
            ducked = _music_pause()
        dev = _music_dev()
        argv = [sys.executable, str(_music_worker()), wav, "0"]
        if dev is not None:
            argv.append(str(dev))
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"speak: spawn failed: {e}", file=sys.stderr)
            if ducked:
                _music_resume()
            return False
        _TTS["proc"] = proc
        # TTS self-trigger window (pi_demo guard): the reply audio hits
        # the M1A mic and the wake gate hears it — live-verified 10-02
        # (one reply drove 3 fake fires, incl. play_music conf 0.9998).
        # pi_demo suppresses wakes + fires while this is in the future.
        STATE["speaking_until"] = time.time() + _SPEAK_GUARD_S
    threading.Thread(target=_speak_monitor, args=(ducked,),
                     daemon=True).start()
    return True
''', '''def _speak_spawn(segs: list, est_s: float) -> bool:
    """Play segs = [(wav_path | None, text | None), ...] in order through
    the M1A worker (None wav = synthesize text via edge-tts/espeak at
    spawn time). Music ducks; the self-trigger guard scales with the
    estimated total reply length (clips are 1.5-3.5 s each; a 2-clip
    reply + TTS tail runs past the old fixed 6.5 s window)."""
    import subprocess
    wavs = []
    for wav, text in segs:
        if wav is None:
            wav = _speak_synth(text)
            if wav is None:
                return False
        wavs.append(wav)
    with _TTS["lock"]:
        p = _TTS["proc"]
        if p is not None and p.poll() is None:
            return False  # already talking — don't stack replies
        ducked = False
        if DEV["music"]["playing"] and not DEV["music"]["paused"]:
            ducked = _music_pause()
        dev = _music_dev()
        argv = [sys.executable, str(_music_worker()), *wavs, "0"]
        if dev is not None:
            argv.append(str(dev))
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"speak: spawn failed: {e}", file=sys.stderr)
            if ducked:
                _music_resume()
            return False
        _TTS["proc"] = proc
        STATE["speaking_until"] = time.time() + max(
            _SPEAK_GUARD_S, est_s + 2.5)
    threading.Thread(target=_speak_monitor, args=(ducked,),
                     daemon=True).start()
    return True


def _clip_dur_s(path: str) -> float:
    try:
        import wave
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 2.0


def speak_clips(names: list, tail: str | None = None) -> bool:
    """Pre-recorded reply: play the named clips (base names, no .wav) in
    order, then an optional TTS tail (live data: time / weather). Every
    named clip missing from disk => plain TTS of the tail (old behavior);
    nothing to say => no voice (chip only). Called from apply_fire
    (STATE["lock"] held). Never raises."""
    try:
        paths = [_clip_path(n) for n in names if n]
        if names and not all(paths):
            paths = []  # partial clip set => full TTS fallback
        segs = [(p, None) for p in paths]
        est = sum(_clip_dur_s(p) for p in paths)
        if tail:
            segs.append((None, tail))
            est += 4.5
        if not segs:
            return False
        return _speak_spawn(segs, est)
    except Exception as e:
        print(f"speak_clips: {e}", file=sys.stderr)
        return False


def speak(text: str) -> bool:
    """Plain TTS reply out loud on the M1A (kept for future free-text
    answers). Same duck/guard/monitor machinery as speak_clips()."""
    return _speak_spawn([(None, text)], 4.5)
''')

# --- 4. apply_fire wiring ---------------------------------------------------
rep('''    if cmd == "OUT_OF_SCOPE":
        # Learned rejection (20th class) — the model's no-action path.
        # Distinct from the confidence gate: the MODEL said "not a command".
        add_chip(f"out of scope ({conf:.2f})")
        return "oos"
''', '''    if cmd == "OUT_OF_SCOPE":
        # Learned rejection (20th class) — the model's no-action path.
        # Distinct from the confidence gate: the MODEL said "not a command".
        add_chip(f"out of scope ({conf:.2f})")
        speak_clips([_clip_name(cmd, slot)])
        return "oos"
''')
rep('''        _apply_alsa_volume(DEV["music"]["vol"])
        add_chip(f"volume {DEV['music']['vol']}")
    elif cmd == "VOLUME_DOWN":
        DEV["music"]["vol"] = max(0, DEV["music"]["vol"] - 20)
        _apply_alsa_volume(DEV["music"]["vol"])
        add_chip(f"volume {DEV['music']['vol']}")
''', '''        _apply_alsa_volume(DEV["music"]["vol"])
        add_chip(f"volume {DEV['music']['vol']}")
        speak_clips([_clip_name(cmd, slot)])
    elif cmd == "VOLUME_DOWN":
        DEV["music"]["vol"] = max(0, DEV["music"]["vol"] - 20)
        _apply_alsa_volume(DEV["music"]["vol"])
        add_chip(f"volume {DEV['music']['vol']}")
        speak_clips([_clip_name(cmd, slot)])
''')
rep('''        if not _music_play(track):  # real audio (instruction 15)
            m["playing"] = False  # fail-soft: don't claim it's playing
            add_chip(f"music: no audio for “{track}”")
''', '''        if not _music_play(track):  # real audio (instruction 15)
            m["playing"] = False  # fail-soft: don't claim it's playing
            add_chip(f"music: no audio for “{track}”")
        else:
            speak_clips([_clip_name(cmd, slot)])
''')
rep('''            m.update(playing=True, paused=False, track=nxt)
            _music_play(nxt)
''', '''            m.update(playing=True, paused=False, track=nxt)
            _music_play(nxt)
            speak_clips([_clip_name(cmd, slot)])
''')
rep('''        m = DEV["music"]
        if m["playing"] and not m["paused"]:
            _music_pause()
            m.update(playing=False, paused=True, user_paused=True)
''', '''        m = DEV["music"]
        if m["playing"] and not m["paused"]:
            _music_pause()
            m.update(playing=False, paused=True, user_paused=True)
            speak_clips([_clip_name(cmd, slot)])
''')
rep('''        if m["track"] and not (m["playing"] and not m["paused"]):
            if _music_resume():
                m.update(playing=True, paused=False, user_paused=False)
                _DUCK.update(awake=False, defer=False)
''', '''        if m["track"] and not (m["playing"] and not m["paused"]):
            if _music_resume():
                m.update(playing=True, paused=False, user_paused=False)
                _DUCK.update(awake=False, defer=False)
                speak_clips([_clip_name(cmd, slot)])
''')
rep('''            m.update(playing=False, paused=False, track=None, user_paused=False)
        else:
            add_chip("nothing playing")
''', '''            m.update(playing=False, paused=False, track=None, user_paused=False)
            speak_clips([_clip_name(cmd, slot)])
        else:
            add_chip("nothing playing")
''')
rep('''    elif cmd == "LIGHT_ON":
        L = DEV["lights"]; L["on"] = True
        if not L["level"]:
            L["level"] = 70
    elif cmd == "LIGHT_OFF":
        DEV["lights"]["on"] = False
    elif cmd == "BRIGHTNESS":
        p = _bright_pct(slot)
        if p is not None:
            L = DEV["lights"]; L["on"] = True; L["level"] = p
    elif cmd == "COLOR":
        c = _lamp_color(slot)
        if c:
            L = DEV["lights"]; L["color"] = c; L["on"] = True
    # --- temperature ---
    elif cmd == "TEMPERATURE":
        d = _temp_deg(slot)
        if d is not None:
            DEV["temp"]["target"] = d
''', '''    elif cmd == "LIGHT_ON":
        L = DEV["lights"]; L["on"] = True
        if not L["level"]:
            L["level"] = 70
        speak_clips([_clip_name(cmd, slot)])
    elif cmd == "LIGHT_OFF":
        DEV["lights"]["on"] = False
        speak_clips([_clip_name(cmd, slot)])
    elif cmd == "BRIGHTNESS":
        p = _bright_pct(slot)
        if p is not None:
            L = DEV["lights"]; L["on"] = True; L["level"] = p
            speak_clips([_clip_name(cmd, slot)])
    elif cmd == "COLOR":
        c = _lamp_color(slot)
        if c:
            L = DEV["lights"]; L["color"] = c; L["on"] = True
            speak_clips([_clip_name(cmd, slot)])
    # --- temperature ---
    elif cmd == "TEMPERATURE":
        d = _temp_deg(slot)
        if d is not None:
            DEV["temp"]["target"] = d
            speak_clips([_clip_name(cmd, slot)])
''')
rep('''    elif cmd == "TIME":
        chip, speak_text = _time_reply(); add_chip(chip); speak(speak_text)
    elif cmd == "WEATHER":
        chip, speak_text = _weather_reply(); add_chip(chip); speak(speak_text)
''', '''    elif cmd == "TIME":
        # Pre-record lead-in (Roger) + TTS tail (live clock).
        chip, speak_text = _time_reply()
        add_chip(chip)
        speak_clips([_clip_name(cmd, slot)], tail=speak_text)
    elif cmd == "WEATHER":
        chip, speak_text = _weather_reply()
        add_chip(chip)
        if STATE.get("weather_online"):
            speak_clips([_clip_name(cmd, slot)], tail=speak_text)
        elif (STATE.get("weather") or {}).get("temp") is not None:
            # Offline + stale cache (<=15 min): apology + stale-lead
            # clips, then the cached data as the TTS/espeak tail.
            speak_clips([_clip_name(cmd, slot), "no_internet",
                         "stale_lead"], tail=speak_text)
        else:
            speak_clips([_clip_name(cmd, slot), "no_internet"])
''')
rep('''            DEV["timer"] = {"until": now + secs, "secs": secs,
                            "mins": secs / 60.0, "done": False}
''', '''            DEV["timer"] = {"until": now + secs, "secs": secs,
                            "mins": secs / 60.0, "done": False}
            speak_clips([_clip_name(cmd, slot)])
''')
rep('''            DEV["alarms"].append(mark)
            DEV["alarms"].sort()
            DEV["alarms"] = DEV["alarms"][:20]
''', '''            DEV["alarms"].append(mark)
            DEV["alarms"].sort()
            DEV["alarms"] = DEV["alarms"][:20]
            speak_clips([_clip_name(cmd, slot)])
''')
rep('''    elif cmd == "CALL":
        add_chip("dialing…")
    elif cmd == "MESSAGE":
        add_chip("opening messages…")
''', '''    elif cmd == "CALL":
        add_chip("dialing…")
        speak_clips([_clip_name(cmd, slot)])
    elif cmd == "MESSAGE":
        add_chip("opening messages…")
        speak_clips([_clip_name(cmd, slot)])
''')
rep('''                DEV["reminders"].insert(0,
                    {"text": slot, "done": False, "ts": time.time()})
                DEV["reminders"] = DEV["reminders"][:20]
''', '''                DEV["reminders"].insert(0,
                    {"text": slot, "done": False, "ts": time.time()})
                DEV["reminders"] = DEV["reminders"][:20]
                speak_clips([_clip_name(cmd, slot)])
''')

ast.parse(src)  # syntax gate
P.write_text(src, encoding="utf-8-sig")  # keep the original BOM
print("patched OK:", P, len(src), "bytes")
