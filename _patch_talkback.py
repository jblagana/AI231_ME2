#!/usr/bin/env python3
"""Talkback fixes (2026-10-05, boss-reported):
1. PAUSE always speaks its clip (was silent when already paused / nothing playing).
2. Same-(cmd,slot) debounce 2.0 s -> 0.5 s (speaker has auto echo cancellation;
   the self-trigger guard still scales with reply length, so 0.5 s is safe).
Idempotent. Run: python _patch_talkback.py
"""
import sys
from pathlib import Path

P = Path(__file__).parent / "pi" / "me2_ui.py"
src = P.read_text(encoding="utf-8-sig")
n = 0

OLD_PAUSE = '''        elif m["paused"]:
            m.update(playing=False, user_paused=True)
        else:
            add_chip("nothing to pause")'''
NEW_PAUSE = '''        elif m["paused"]:
            m.update(playing=False, user_paused=True)
            speak_clips([_clip_name(cmd, slot)])
        else:
            add_chip("nothing to pause")
            speak_clips([_clip_name(cmd, slot)])'''
if OLD_PAUSE in src:
    src = src.replace(OLD_PAUSE, NEW_PAUSE, 1); n += 1
elif NEW_PAUSE in src:
    print("patch 1 (PAUSE) already applied")
else:
    sys.exit("patch 1 anchor not found")

OLD_DB = '''_LAST_FIRE = {}  # (cmd, slot) -> ts, 2 s debounce (ACTUATION guardrail)'''
NEW_DB = '''_LAST_FIRE = {}  # (cmd, slot) -> ts, 0.5 s debounce (was 2 s; speaker has
# auto echo cancellation — the scaling self-trigger guard is the real
# double-fire protection, so a fast re-fire of the same command is safe)'''
if OLD_DB in src:
    src = src.replace(OLD_DB, NEW_DB, 1); n += 1
elif NEW_DB in src:
    print("patch 2 (debounce comment) already applied")
else:
    sys.exit("patch 2 anchor not found")

OLD_GATE = '''    if cmd != "SET_VOLUME" and now - _LAST_FIRE.get((cmd, slot), 0.0) < 2.0:
        return "debounced"'''
NEW_GATE = '''    if cmd != "SET_VOLUME" and now - _LAST_FIRE.get((cmd, slot), 0.0) < 0.5:
        return "debounced"'''
if OLD_GATE in src:
    src = src.replace(OLD_GATE, NEW_GATE, 1); n += 1
elif NEW_GATE in src:
    print("patch 3 (debounce gate) already applied")
else:
    sys.exit("patch 3 anchor not found")

P.write_text(src, encoding="utf-8-sig")  # preserve the original BOM
print(f"applied {n} patch(es) to pi/me2_ui.py")
