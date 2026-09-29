# A2 build notes — 3.0s window + two-head (command + slot)

Goal: build, train, evaluate A2. Deadline Sat 2026-10-03.

## Ground truth (verified from raw_v1i manifest, 22,800 clips, 11 classes)
Class counts: media_control 3000, make_call 2800, control_lights 2400,
dim_lights 2000, play_music 2000, set_alarm 2000, set_reminder 2000,
set_temperature 2000, set_timer 2000, ask_weather 1600, ask_time 1000.

Parametric (slot-bearing) classes + slot vocab (derived from actual phrases):
- dim_lights (5): twenty, thirty, fifty, seventy, eighty  [%]
- set_timer (8): one, two, five, ten, fifteen, twenty, thirty, forty five [min]
- set_alarm (7): five am, five thirty am, six am, seven am, eight am, nine am, six pm
- set_temperature (7): eighteen, twenty, twenty two, twenty four, twenty five, twenty six, twenty eight
- set_reminder (10): buy groceries, call mom, drink water, pay the bills,
  take out the trash, water the plants, the meeting, five pm, tomorrow, next week
- make_call (6): mom, dad, brother, sister, friend, the doctor
Total slot values = 43. Non-parametric (no slot): ask_time, ask_weather,
control_lights, media_control, play_music.

## Architecture (tool-verified forward pass before commit)
Input (B,1,80,150) -> conv stack (32/64/128, 3x3 pad1, maxpool2 x3)
  -> (B,128,10,18)  [10 freq x 18 time cells]
Head 1 command: AdaptiveMaxPool2d(1) over all 18 cells -> FC(128,11)
Head 2 slot (per parametric class c): slice time cells SLOT_CELLS (default 3..18),
  AdaptiveMaxPool2d(1) -> FC(128, N_c)
Params: conv 93,120 + cmd FC 1,419 + slot FC 5,547 = 100,086.

## Loss
CE_cmd (all clips) + 0.5 * CE_slot (only slot-bearing clips, per-class).

## Recipe
v1g (v1k failed): jitter 0.95-1.05, gain +-6 dB, SNR 5-25 dB white noise.
Full retrain from scratch (no warm-start; 50->150 shape breaks conv weights).
50 epochs.

## Files (self-contained, do NOT touch n002's dirty model.py/train.py)
- src/slots.py       (new) taxonomy + phrase->slot extraction + self-test
- src/model_a2.py    (new) VCMTwoHead, self-contained conv stack, smoke
- src/train_a2.py    (new) 150-frame pipeline + dual-loss train, self-contained data utils
- src/eval_a2.py     (new) cmd acc + slot acc + confusion, vs v1i baseline 0.8618

## Train env
n002.ai.internal (A100). Dataset: data/raw_v1i (11-class, .raw s16le + manifest.jsonl).
v1i baseline: 0.8618 cmd acc. Fallback: if A2 < v1i, ship v1i command-only.

## Status
- [ ] verify n002 data (.raw + durations + v1i timing)
- [ ] slots.py + self-test
- [ ] model_a2.py + smoke
- [ ] train_a2.py
- [ ] eval_a2.py
- [ ] commit + push local
- [ ] sync to n002, smoke
- [ ] train (50 ep)
- [ ] eval + report
