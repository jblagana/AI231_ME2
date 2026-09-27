# INSTRUCTIONS — AI231 ME2 · tiny on-device Voice Command Model (VCM)

Workspace rule (workspace root `AGENTS.md`, "Instruction log rule"): every user
instruction is logged **verbatim, newest first, before any work on it**. The
Interpretation section is agent-owned; the user may edit it — **a user edit is
the new instruction** (stop, log it, follow it). Re-read this file before each
log update and before `git push`. `done` = **pushed to the remote repo**.

---

## 2026-09-27 — "make sure u log the process in a python notebook so i can check it while ur working. also include there the issues u encountered and the fix or workaround"
Status: in progress — live notebook at notebooks/vcm_v1_training.ipynb (diagnosis + issues + fixes + train)
### Instruction (verbatim)
> make sure u log the process in a python notebook so i can check it while ur working. also include there the issues u encountered and the fix or workaround
### Interpretation (agent — user may edit this section)
- Boss wants a live, checkable notebook of the whole process: issues
  encountered, fixes/workarounds, and the training run — readable while
  muji works. Notebook: `notebooks/vcm_v1_training.ipynb`, executed
  cell-by-cell so outputs land incrementally (boss can open it mid-run).
- Content: (1) env + issues log (torchcodec/broken torchaudio, slow
  decode path, v1 collapse); (2) diagnosis (aug cosine-sim probe,
  no-aug 3-ep climb); (3) speed-jitter root cause + fix; (4) validation;
  (5) full 15-epoch train (background process, progress cell re-runnable).
- Results so far (all in the notebook, sections 6-8):
  - v1b (fixed resample, 15 ep): final 0.183 — resample fix real but NOT
    the whole story.
  - Root cause #2: clips are ~2.14 s (speech ~1.25 s, onset ~0.08 s) but
    the 1.0 s model window used a RANDOM crop — measured: random 1 s crop
    keeps >=50% of speech only 23% of the time (61k trials) → model saw
    partial phrases → collapsed to media_control (largest class, 799/1000
    of 1k eval preds). (An earlier "0.39 s" figure was a probe bug — it
    misread mp3 compressed bytes as audio.)
  - Fix: center-crop + center-pad in `src/train.py`. v1c (15 ep, detached
    pid 15836): ep7 eval_acc 0.385 and climbing (vs 0.183 v1b).

---

## 2026-09-27 — "now back to work" (resume main agenda: AI231 ME2 VCM training)
Status: in progress — fast data path shipped, full train starting after batch decode
### Instruction (verbatim)
> now back to work
### Interpretation (agent — user may edit this section)
- Resumed the main agenda (task 10: train VCM v1). Found the 15-epoch train
  that was launched earlier **was running on the slow data path**:
  probe-measured per-item ffmpeg mp3 decode = 0.19 s AND
  torchaudio.functional.speed (speed jitter) = 3.7 s/item in this venv
  (both route through the broken torchcodec path; torchaudio.load also
  ImportErrors — all three measured in `_probe_*.py`, 2026-09-27).
  Extrapolation: ~450 min/epoch → ~112 h for 15 epochs. Killed it
  (pid 19932) rather than let it grind for 5 days.
- Shipped the fast data path (commit `086e288`):
  - `src/decode_wav.py` — one-time batch mp3 → `.raw` (16 kHz mono s16le,
    no header) at ~14 files/s (~16 min for 13,757 files), idempotent.
  - `train.load_wav` — `np.fromfile` on `.raw` (~4 ms/item measured),
    ffmpeg fallback if `.raw` missing.
  - `train.resample_speed` — pure-torch linear resample for the speed
    jitter (~50 ms/item; length exact, corr 1.0 at rate 1.0).
- New expected epoch time: ~10–15 min (data ~25 min worst case + forward/
  backward on CPU) — will be confirmed by the first real epoch's `sec` in
  history.json. 15 epochs ≈ 3–4 h, not 112 h.
- Sequence: batch decode (running, `_decode_full.log`) → `--smoke` on the
  fast path → full 15-epoch train → eval acc on the speaker-disjoint set.

---

## 2026-09-27 — "the 2class slides is for everything about this project fyi, also i want to see visually (maybe a website) the datasets and the things done and still to do and the datasets i wanna see them or access them there or a few of them"
Status: done (site/ dashboard + committed/pushed)
### Instruction (verbatim)
> the 2class slides is for everything about this project fyi, also i want to see visually (maybe a website) the datasets and the things done and still to do and the datasets i wanna see them or access them there or a few of them
### Interpretation (agent — user may edit this section)
- Built `site/` — a self-contained static dashboard (vanilla HTML/CSS/JS, no
  fetch — works from file:// and the muji preview endpoint). Sections:
  at-a-glance stats (13,787 clips, 10 classes, 94K params, 40 voices,
  80×50 input, ~ms Pi inference), architecture flow diagram
  (mic → wake gate → VCM → mock device) with the three headline "why"
  callouts (separate gate / CNN not CRNN / ONNX), timeline with done/
  today/next status, dataset stats + **30 playable samples** (3 per class,
  20 distinct eval-split voices rotated by class index, each with an
  audio player + log-mel spectrogram PNG rendered with the SAME feature
  pipeline as train.py — 1 s window, 80 mel × 50 frames), and all 7
  WHYS.md entries rendered slide-ready.
- `src/gen_site_assets.py` — regenerates all site assets from
  `data/raw/manifest.jsonl` (eval split only — model never trains on them):
  copies 30 mp3s, renders 30 spectrograms, inlines samples.json into
  index.html from `index_template.html` (the `/*__SAMPLES__*/` marker).
  Re-run after augmentation to refresh samples.
- `src/train.py` fixes (the venv's torchaudio 2.11 routes all decoding
  through torchcodec, whose DLL install is broken — verified: `train.py
  --smoke` crashed on the first load): `load_wav` now decodes via the
  imageio_ffmpeg binary (subprocess, `-ar 16000 -ac 1 -f s16le`); speed
  jitter updated to the 2.11 `speed(wav, orig_freq, factor)` signature.
  Smoke train passes (1 epoch, 64 train / 32 eval items, acc 0.125 =
  chance-level as expected for 1 epoch).
- venv: `pip install imageio-ffmpeg matplotlib` (torchcodec installed then
  removed — broken DLL; left a note to revert load_wav once a working
  codec lands).

---

## 2026-09-27 — "log those reasons on the whys of our architecture, and maybe add that power consumption test"
Status: done (WHYS.md created + committed/pushed)
### Instruction (verbatim)
> so the gate is the model for wakeword and the vcm is the model for the command, got it. also log those reasons on the whys of our architecture, and maybe add that power consumption test to show the gate is really cheap compared to the vcm runs. ill be presenting the design/archi and method in class in 2slides and then a demo.
### Interpretation (agent — user may edit this section)
- Created `WHYS.md` — the "why" source of truth for the design/methodology
  section of the writeup + class presentation. Entries (newest first):
  (1) why the wake gate is a separate model, not a head on the VCM (4
  reasons: run frequency, input windows, data scale, failure budgets);
  (2) why plain CNN not CRNN (no sequence left after 3 maxpools, wrong
  output shape for CTC, ARM-sequential-compute penalty, Pixel keyword-
  spotting precedent; RNN flagged as slot-head fallback); (3) why parametric
  commands are class-only in v1 + Tier-1 joint intent+slot head design
  (finite slot vocab ~35 tokens, strictly additive, no collective conflict);
  (4) why ONNX Runtime on the Pi (50MB vs 2GB, format not framework);
  (5) why TTS not human recording; (6) why 10 classes as-is; (7) why wake
  word in scope (09-22 group protocol).
- Added **benchmark item 6 (proposed, pending boss OK to post to group):**
  power draw — gate (continuous, idle audio) vs VCM (per wake event),
  expected 10–100× cheaper, the quantified justification of the
  bouncer/receptionist split. Method: Pi power-rail meter over a fixed
  10-min script (N wake events + idle audio, both configs).
- Boss is presenting design/archi + method in **2 slides + a demo** in class
  (date TBC) — WHYS.md is the slide source.

---

## 2026-09-25 — "work here" (session cwd = this repo)
Status: in progress
### Instruction (verbatim)
> work here C:\Users\Jan\.cline\data\workspaces\chat\AI231_ME2
### Interpretation (agent — user may edit this section)
- Boss is working out of this repo in the AI231 ME2 session (post-handover).
- First action: check state of the background TTS gen from the handover.
  Found: **gen process (pid 15868) is dead** — log stopped at 18:15
  (`[eval] play_music: 600 clips, cum ok=7498`), manifest has 7,574 entries,
  ~58% of the ~12.9k target. Also found the handover's "resumable, skips
  existing" claim was **wrong** — make_dataset.py rewrites the manifest in
  `"w"` mode and re-synthesizes everything. Fixing: make it actually
  resumable (load manifest, skip done files, append mode), then restart gen.

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
- **Handover = docs, not a file move.** (Correction 2026-09-25: the session did
  NOT exist yet — only this folder did; muji confused the folder with a session.
  The real session "AI231 ME2" was then created under the AI231 workspace with
  cwd = this repo, `C:\Users\Jan\.cline\data\workspaces\chat\AI231_ME2`.)
  The handover is: this entry + PLAN.md
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
