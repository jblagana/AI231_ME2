# INSTRUCTIONS — AI231 ME2 · tiny on-device Voice Command Model (VCM)

Workspace rule (workspace root `AGENTS.md`, "Instruction log rule"): every user
instruction is logged **verbatim, newest first, before any work on it**. The
Interpretation section is agent-owned; the user may edit it — **a user edit is
the new instruction** (stop, log it, follow it). Re-read this file before each
log update and before `git push`. `done` = **pushed to the remote repo**.

---

## 2026-09-25 — Handover to 'AI231 ME2' session + boss clarifications
Status: in progress (docs being updated; dataset gen still running)
Progress: 60% — ETA ~30 min (docs + push; TTS gen continues in background)
### Instruction (verbatim)
> before u do the following task below, do this: handover the ai231 me2 project
> to another chat session name 'AI231 ME2' under the AI231 chat workspace,
> include all files and other important things weve discussed or uv done, also
> if ever u need access to an hpc for model training-i have access to that, and
> also wakeword is included for the smart device (not sure yet if included in
> becnhmarking), also do we consider tensorrt in addition to onnx. also, i
> already have a rpi4b 8gb ram and 64gb sandisk ultra sd card and a case with
> fan. the smart light with dimming can be an api ui for thats free and what
> else can be an api ui.
### Interpretation (agent — user may edit this section)
- **Handover = docs, not a file move.** The session named "AI231 ME2" already
  exists and its cwd IS this repo (`C:\Users\Jan\.cline\data\workspaces\chat\AI231_ME2`),
  so it sees every file already. The handover is: this entry + PLAN.md
  decisions + HANDOVER.md (state of the world: what's done, what's running,
  what's next, gotchas) + AGENTS.md ground-truth update. Commit + push so the
  session can verify via `git log` that it's on the same page.
- **Hardware (ratified facts):** RPi **4B 8GB** + 64GB SanDisk Ultra SD + case
  with fan. SD card is the demo bottleneck (not RAM) — see PLAN.md risks.
  Demo target is RPi4 (not 5).
- **HPC:** boss has access; **credentials/endpoint not yet shared** — ask him
  when training Day-2 lands. Local CPU torch is fine for 94K params; HPC is
  for the final pooled-dataset train (Day 5) if the collective data is big.
- **Wake word: IN SCOPE now** (boss: "wakeword is included for the smart
  device"). Not in the 10-class spec list — it's a **separate 2-class gate**
  (wake/no-wake) in front of the VCM, per the 09-22 group protocol (music
  edge case: wake required while playing, volume drops to 5%). Benchmark
  inclusion is TBD — group protocol from 09-24 doesn't mention it; flag in
  BENCHMARK.md as an optional item. Implementation choice is a decision:
  tiny 2-class CNN (reuse the TTS pipeline, fully on-device, matches the
  "no ASR/LLM" constraint) vs Porcupine (free personal license, runs on Pi,
  custom wake word) — **muji's lean: 2-class CNN** (consistency + no
  vendor dependency); boss to veto.
- **TensorRT: NO for this hardware** (verified reasoning, not vibes):
  TensorRT has no ARM64/RPi build — it's x86 + Jetson only. On RPi4 the
  portable path is **ONNX Runtime** (CPU EP, ARM64 wheels exist). 94K params
  → ~1ms/clip; TensorRT's speedup is irrelevant at this size. Keep ONNX
  export as the single inference artifact; revisit only if the group
  benchmarks on a Jetson.
- **API-UI demo (task 5) — free, no hardware, all mock-device:**
  - dim_lights → mock light state (brightness %) with a UI slider — boss's pick
  - set_timer / set_alarm / set_reminder → local timer/reminder engine (we
    literally have one: muji's taskstore pattern — due-time + toast)
  - set_temperature → mock thermostat state
  - play_music / media_control → local audio player (pause/stop/next/volume)
  - make_call → mock dialer (shows the number, no real call)
  - ask_question → weather/time via a public API (allowed: "no cloud"
    constrains the VCM inference, not the action side — note this reading
    in the writeup)
  - All commands hit a single `POST /device/command` endpoint on a tiny
    local FastAPI/Flask app + a status page = the "real-world demo".
### Subtasks
- [x] Log this instruction verbatim
- [ ] Update PLAN.md (hardware, wake word, TensorRT, HPC, API-UI demo)
- [ ] Write HANDOVER.md (state of the world for the new session)
- [ ] Update AGENTS.md ground truth
- [ ] Commit + push

## 2026-09-25 — Take over AI231 ME2: start building the VCM
Status: in progress (Day 1 — scaffolding + dataset pipeline)
Progress: ~40% of Day-1 milestone
### Instruction (verbatim)
> Back to AI231 ME2 — I've got the plan ready
> (chips from Agendas session; specs at C:\Users\Jan\Muji\ai231_me2_specs.md,
> deadline Sat 2026-10-03, task #9 "Main agenda")
### Interpretation (agent — user may edit this section)
- Build the individual-work track (tasks 2, 4, 5, 6, 7, 8) inside the deadline;
  contribute proposals to the collective track (tasks 1, 3).
- **Ratified design (see PLAN.md):** TTS-synthesized dataset (edge-tts, 40
  voices, speaker-disjoint train/eval), 10 classes as per spec, log-mel(80) +
  3-block CNN < 1M params, PyTorch → ONNX, no wake word, parametric slots out
  of scope (class detection only).
- Day-1 deliverables: repo + env + dataset generator + model + training script,
  dataset generation running, smoke tests green.
- Collective gate: benchmark proposal needs to be posted to the ai231-me2
  Telegram group (boss has been silent there 2 weeks — his 2-min unblock).

## 2026-09-25 — Repo created
Status: done (pushed)
- Local: `C:\Users\Jan\.cline\data\workspaces\chat\AI231_ME2` (git init, main)
- Remote: jblagana/AI231_ME2 (public, created via API 2026-09-25)
- First push: `aeef261` (Day-1 scaffolding) + BENCHMARK.md
- Layout: `src/` (commands, make_dataset, model, train), `notebooks/`, `data/`
  (gitignored), `runs/` (gitignored checkpoints)
