#!/bin/bash
# Deploy the talkback-patched me2_ui.py on the Pi (run as jan, uses sudo for restart).
set -e
cd ~/me2
test -f me2_ui.py.new || { echo "me2_ui.py.new missing — scp it first"; exit 1; }
python3 -c 'import ast; ast.parse(open("me2_ui.py.new", encoding="utf-8-sig").read()); print("syntax OK")'
cp me2_ui.py me2_ui.py.bak_talkback_1005
mv me2_ui.py.new me2_ui.py
sudo systemctl restart me2-ui
sleep 2
echo "service: $(systemctl is-active me2-ui)"
grep -n "0.5 s debounce\|< 0.5:" me2_ui.py | head -4
grep -n 'add_chip("nothing to pause")' -A1 me2_ui.py | head -4
