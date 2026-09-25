# PLAN — AI231 ME2 Voice Command Model (VCM)

**Deadline: Sat 2026-10-03** · **Status: Day 1 (2026-09-25) — scaffolding + dataset**

## Architecture (v1, ratified by boss 2026-09-25)
- **Features:** log-mel, 80 bins, 16 kHz, ~1 s window / 20 ms hop (Speech-Commands-style)
- **Model:** small CNN (3–4 conv blocks + global pool + FC), **target < 1 M params**
  (budget ceiling 10 M). Real-time on RPi 4/5 is a non-issue at this size.
- **Classes: 10** — spec list as-is; "play music" kept separate (media control = pause/stop/next/volume).
  Parametric commands (dim to X%, timer for X min, alarm, temperature, remind, call)
  are **detected by class only** — slot values are out of scope for a tiny VCM
  (spec says "understand the most common commands"; detection is the defensible read).
- **Export:** PyTorch → ONNX → (Pi demo) — ONNX Runtime on RPi is the portable path.
- **No wake word** — not in the spec; note as future work / group-protocol question.

## Dataset (collective task — our contribution)
- **TTS synthesis** (edge-tts, many voices × speeds) — constraint (7) is on inference,
  not data generation. Human recording won't scale in 7 days.
- **Augmentation:** speed jitter (0.9–1.1), random gain, SNR 0–20 dB background noise
  (Google environmental-noise dataset / MUSAN), light reverb.
- **Split:** speaker-disjoint (some voices train-only, some eval-only) — prevents the
  model from memorizing voices instead of commands.
- **Target:** ~1,000–2,000 clean clips per class (≈10–20k total), 500 eval per class.

## Benchmark (collective task — our proposal, post to group)
Per the 09-24 group protocol (N non-owner evaluators, each command × N, logs required):
1. **Accuracy / macro-F1** on clean held-out (speaker-disjoint)
2. **Robustness:** same eval at SNR 20/10/5 dB
3. **Latency:** p50/p95 inference per 1 s window (CPU, RPi if possible)
4. **Model size:** params + on-disk footprint (ONNX)
5. **Task-completion rate** in the live demo (N evaluators, each command N times)
6. **WER of the recognized command** (optional, only if we add a decoder)

## Timeline
| Day | Milestone |
|---|---|
| 09-25 (Fri) | Scaffolding, env, TTS dataset gen running, model v1 written |
| 09-26 (Sat) | First train on generated data, sanity metrics |
| 09-27 (Sun) | Augmentation pass, retrain, benchmark harness |
| 09-28–29 | RPi demo (mic → VCM → mock device), collect classmate eval data |
| 09-30 | Final train on pooled collective dataset, final numbers |
| 10-01–02 | Demo polish, writeup, buffer |
| 10-03 | **Deadline** |

## Risks
- **Collective dataset/benchmark sync** — the gate. Post our benchmark proposal to
  ai231-me2 group early (boss silent in group for 2 weeks; this is the 2-min unblock).
- **TTS voice uniformity** — synthetic speech is cleaner than real; mitigated by
  noise augmentation + the live-eval benchmark item.
- **RPi availability** — spec allows sharing devices; confirm with group.

## Decision log
- 2026-09-25: repo created (local `C:\Users\Jan\.cline\data\workspaces\chat\AI231_ME2`,
  remote jblagana/AI231_ME2). TTS route + 10 classes + CNN architecture ratified.
  Parametric slots out of scope. No wake word.
