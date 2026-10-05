#!/bin/bash
# Remove the fire_beep from pi_demo.py (boss: "fire boop — remove this").
# The M1A speaker has auto echo cancellation and the UI's scaling
# self-trigger guard is the real double-fire protection, so the beep
# (and its 3 s cooldown) are no longer needed.
set -e
cd ~/me2
cp pi_demo.py pi_demo.py.bak_firebeep_1005
python3 - <<'EOF'
from pathlib import Path
p = Path("pi_demo.py")
src = p.read_text(encoding="utf-8-sig")
old = """            fire(res)
            demo.fire_beep(args.device)
            if state_url:
                post(state_url, {"mode": "idle", "t": time.time()})
            cooldown = time.time() + 3.0  # our beep would self-trigger"""
new = """            fire(res)
            # fire_beep removed (2026-10-05): the M1A speaker has auto
            # echo cancellation and the UI's scaling self-trigger guard
            # is the real double-fire protection — the beep only added
            # a 3 s cooldown between commands.
            if state_url:
                post(state_url, {"mode": "idle", "t": time.time()})
            cooldown = time.time() + 0.5"""
assert old in src, "anchor not found"
p.write_text(src.replace(old, new, 1), encoding="utf-8-sig")
print("fire_beep call removed")
EOF
python3 -c 'import ast; ast.parse(open("pi_demo.py", encoding="utf-8-sig").read()); print("syntax OK")'
sudo systemctl restart me2-wake
sleep 2
echo "service: $(systemctl is-active me2-wake)"
grep -n "fire_beep removed\|cooldown = time.time() + 0.5" pi_demo.py
