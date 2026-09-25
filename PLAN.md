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
- **Wake word: IN SCOPE** (boss ratified 2026-09-25, post-spec) — separate 2-class
  gate (wake/no-wake) in front of the VCM; matches the 09-22 group protocol (wake
  required while music plays, volume drops to 5%). Implementation lean: tiny 2-class
  CNN reusing the TTS pipeline (fully on-device, no vendor dep) vs Porcupine (free
  personal license, custom wake word) — boss to veto. Benchmark inclusion TBD
  (09-24 group protocol doesn't mention it — flag in BENCHMARK.md as optional).
- **Export:** PyTorch → ONNX → ONNX Runtime (CPU EP, ARM64) on the Pi.
  **TensorRT: NO** — no ARM64/RPi build (x86 + Jetson only); irrelevant at 94K
  params anyway (~1ms/clip). Single inference artifact = ONNX.

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
| 09-28–29 | RPi demo (mic → wake gate → VCM → API-UI mock device), collect classmate eval data |
| 09-30 | Final train on pooled collective dataset, final numbers |
| 10-01–02 | Demo polish, writeup, buffer |
| 10-03 | **Deadline** |

## Risks
- **Collective dataset/benchmark sync** — the gate. Post our benchmark proposal to
  ai231-me2 group early (boss silent in group for 2 weeks; this is the 2-min unblock).
- **TTS voice uniformity** — synthetic speech is cleaner than real; mitigated by
  noise augmentation + the live-eval benchmark item.
- **RPi availability** — RESOLVED (2026-09-25): boss has an **RPi 4B 8GB +
  64GB SanDisk Ultra SD + case with fan**. The SD card is the demo bottleneck
  (not RAM): audio streaming + ONNX Runtime + demo web app all fit easily in
  8GB; SD wear from continuous mic writes is the watch-item (use tmpfs for
  ring buffer, minimal logging).
- **HPC** — boss has access (credentials not yet shared). Local CPU torch is
  fine for 94K params; HPC is for the Day-5 final train on the pooled
  collective dataset if it's big. Ask boss for endpoint/creds when needed.

## Demo (task 5) — API-UI mock device (ratified 2026-09-25)
One local web app: mic → wake gate → VCM → `POST /device/command` → mock
device state + status page. All free, no hardware beyond the Pi:
- **dim_lights** → mock light (brightness % slider) — boss's pick
- **set_timer / set_alarm / set_reminder** → local timer/reminder engine (due-time + toast)
- **set_temperature** → mock thermostat state
- **play_music / media_control** → local audio player (pause/stop/next/volume)
- **make_call** → mock dialer (shows number, no real call)
- **ask_question** → weather/time via public API (note in writeup: "no cloud"
  constrains VCM inference, not the action side)

## Decision log
- 2026-09-25: repo created (local `C:\Users\Jan\.cline\data\workspaces\chat\AI231_ME2`,
  remote jblagana/AI231_ME2). TTS route + 10 classes + CNN architecture ratified.
  Parametric slots out of scope. No wake word.
- 2026-09-25 (handover): wake word IN SCOPE (2-class gate, TBD vs Porcupine);
  TensorRT rejected (no ARM64 build); hardware = RPi4B 8GB (SD card is the
  bottleneck); HPC available for final train; API-UI mock-device demo ratified.
