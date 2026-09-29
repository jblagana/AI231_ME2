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

## Robustness changes (ratified 2026-09-28)
Gap-analysis of the spec's real command phrasings vs. v1's assumptions. Four
changes; status below is the live state as of 09-28 (code = `src/`).
1. **Window 1s → 2s.** Spec's real commands run ~1.5–2.5s ("turn on the
   lights", "set a timer for five minutes"); a 1s window truncates them.
   2s covers 9/10 (temperature is the outlier — class-detected only, no slot
   needed). Cost: 50→100 frames, ~2ms/clip on the Pi, still trivial.
   **PENDING code** — `model.py` (`WINDOW_S=1.0`, `N_FRAMES=50`), `train.py`
   (1s center-crop), `gen_site_assets.py` (1s/50-frame viz) all still 1s.
   Note: TTS clips are ~0.39s median, so 2s mostly adds headroom for *live
   human* speech, not the synthetic set.
2. **Classes 10 → 12** (split `ask_question` → `ask_weather` + `ask_time`).
   "what's the weather" and "what time is it" both hit one class, so the demo
   app can't know which to answer. **PENDING code** — `commands.py` `CLASSES`
   still 10; demo app + any trained weights must follow.
   **Split confirmed clean (2026-09-29):** the 13 ask_question phrases already
   divide **8 weather / 5 time**, each half with a strong shared anchor word
   (weather: *weather/forecast/temperature*; time: *time/day/date*). No
   "general" bucket exists — it's a clean 2-way, better than the earlier
   weather/time/general guess. This is the active lever (see Lever analysis).
3. **Multi-phrasing dataset** (3–4+ phrasings/command, many voices, real-human
   eval clips). Single-phrasing data fails the moment a grader says "switch on
   the lights" instead of "turn on the lights". **DONE in code (v1f)** —
   `commands.py` now has 114 phrases / 10 classes (committed `06e7b94`);
   dataset expanded 6,900→11,400 clips/split. Remaining lever: real-human
   eval-session clips (live benchmark, not TTS) to fight TTS voice uniformity.
4. **Writeup note — no-cloud constrains the *model*, not the *action layer*.**
   The demo's weather/time API call is outside the VCM boundary; defensible,
   but needs the one-line note so a grader doesn't read "no cloud" as "no API".
   **ALREADY IN DOC** — see Demo section ("no cloud constrains VCM inference,
   not the action side"); carry it into the writeup.

**Pending before the 09-30 final train:** code edits for #1 (2s window) and
#2 (12 classes) + a retrain. #3 and #4 need no code change.

## Lever analysis (2026-09-29, post-v1g) — where the accuracy goes next

v1g (max-pool) = **0.8355 best / 0.8342 final** (v1f 0.7554 → +0.080). The
confusion audit (9,511/11,400) shows the remaining error is concentrated in
**verb-differing clusters** that the global pool still can't fully separate:
`play_music↔media_control` (302 mutual, #1), `dim↔control_lights` (187),
`make_call↔media` (187), and a diffuse `ask_question` (recall 0.701, scatters
to set_reminder 134 / dim_lights 81 / make_call 64).

**Two levers, in order of cost:**

1. **Merge confusable classes (re-labeling, no arch change).**
   - ~~A1: play_music → media_control~~ — **OFF the table (boss, 2026-09-29).**
     play_music is a distinct outcome: "play <title>" (from a fixed song list),
     "play music" (random available), or "play <unavailable>" (no match) — each
     needs a different app response, so the class must stay separate. This also
     kills the spec's "No.1 → No.8" fold.
   - ~~A2: dim_lights → control_lights~~ — **OFF the table (boss, 2026-09-29).**
   - **Consequence:** the play_music↔media cluster (302) is now the #1 bottleneck
     and is **not fixable by taxonomy** — it can only be attacked by the arch.

2. **Split the diffuse class (re-label + retrain, *improves* the model).**
   - ask_question → ask_weather (8 phrases) + ask_time (5 phrases). Clean 2-way
     (see robustness #2). The time half's bleed into set_reminder/set_alarm is
     the tell — "what day is it **today**" vs "set a reminder for **tomorrow**"
     share the time-content; the leading interrogative ("what") is the real
     discriminator but the global max-pool throws it away (latches on the louder
     content word). Splitting gives each sub-class a tight anchor.
   - **Ceiling (recomputed from the v1g matrix, A off):** even a *perfect*
     ask_question only reaches **~0.868**; at 0.85 recall → **0.8513**, 0.90 →
     **0.8570**. Real, but not the whole story.

3. **Arch — leading-frame attention (the mechanism fix, only lever left past ~0.87).**
   - The max-pool grabs the single loudest cell, which for "play/pause music" is
     the *shared* word "music," so both collapse onto the same cell. A learned
     **frame-level attention** (softmax over frames → weighted sum) lets the model
     read the *verb* frame instead of the loudest noun — and it generalizes to
     play/pause, call/phone, set/remind/wake.
   - **Caveat:** the convs already downsample time 8× (50 → ~6 cells), so true
     verb-frame resolution also needs **finer time resolution** (fewer
     `MaxPool2d(2)` in the time axis). Real arch change, more work than the
     max-pool swap, likely a smaller gain (+0.01–0.03, not +0.08).

**Rejected — "add emphasis on the what" as *audio*:** making "what" louder/slower
in the TTS is a **synthetic artifact**. Real users don't shout "WHAT," so a model
trained on it learns "loud what = question" and won't transfer to a real mic.
The valid version is emphasis in the *model's* attention (Lever 3), not the audio.
**Valid augmentation instead:** add *real* interrogative-led time phrasings
("do you know the time", "could you tell me the time") — legitimate, not an artifact.

**Spotify (app-layer, out of scope for the demo):** the VCM is a pure intent
classifier (no LLM/TTS on device, pre-recorded confirmation clips) — it outputs
the *class*, it does **not** extract the song title. "play <title>" vs "play
music" vs "play <unavailable>" is an **application-layer** decision. Spotify would
need network + OAuth + library resolution — a separate component *after* the VCM,
and it breaks the "edge, no cloud" story. **Demo: VCM classifies → play_music,
a stub plays a confirmation clip.** Title handling = out of scope / app-layer.

**Plan (2026-09-29):** do the ask_question split (Lever 2, cheap, no artifact)
+ add interrogative-led time phrases, retrain v1h. Then decide if the leading-
frame attention arch (Lever 3) is worth the retrain to attack the play_music↔media
cluster that taxonomy can no longer touch.

### A/B results (2026-09-29) — v1i (data) + v1h2/v1j/v1k (pool + recipe arms)

All arms: 30 ep, class-weights, real MUSAN, warm-start from `runs/v1g/vcm_v1.pt`,
same `data/raw_v1f` (11,400 eval), 10 classes. **v1i** is the 11-class
`ask_question`→`ask_weather`+`ask_time` split on the v1g max-pool (separate
data, `raw_v1i`). The three arms isolate *one* variable each vs v1g:

| arm | change vs v1g | eval acc | vs v1g (0.8343) |
|---|---|---|---|
| **v1g** | — (max-pool, v1g recipe) | **0.8343** | baseline |
| **v1i** | ask_question → weather+time (11 cls) | **0.8618** | **+0.0275** ✅ |
| v1h2 | pool `fw_mean` (v1h2 fixed front ramp) | 0.7698 | −0.0645 |
| v1j | pool `fw_max` (front-biased max) | 0.8104 | −0.0239 |
| v1k | recipe `v1k` (SpecAug+SNR0-25+jitter0.9-1.1+reverb) | 0.7353 | −0.0990 ❌ |

**v1i is the clear win and the new shipping model** (+0.0275, ask→set_reminder
134→75, play_music recall 0.711→0.812 as a side effect). The v1h sign bug
(WHYS.md) re-opened the front-weighting question; v1h2/v1j re-test it with the
ramp pointing the right way. **Result: both front pools still lose to v1g's
plain max** (v1h2 −0.0645, v1j −0.0239), and the front pools *hurt* the
play↔media cluster (v1h2 media→play 130→214) — so the "front is the
discriminator" hypothesis is now **dead, correctly measured this time** (the
v1h bug is fixed, the front ramp points the right way, and it still loses).
The learned-attention bet (Lever 3) is re-priced **down**; the path is the
data-side levers (v1i split ✅, v1k robustness recipe). **v1k judged
separately (pre-registered rule: ≥ v1g → adopt for the 10-03 final pooled
train): it FAILED** — 0.7353 (−0.0990), train loss ~1.10 vs v1g's ~0.34
(underfit), ask→set_reminder 134→380, media→play 130→338. The v1k recipe
(SpecAug + SNR 0–25 + jitter 0.9–1.1 + light reverb) is **too aggressive for a
30-ep warm-start** — it rehearsed the final-train augmentation and the model
couldn't absorb it in 30 ep. **Keep the v1g recipe** for the final; if the
10-03 train wants the robustness ingredients, ablate them one at a time (the
SNR 0–25 extension is the one that directly serves the benchmark's SNR-5 item).

## Benchmark (collective task — our proposal, post to group)
Per the 09-24 group protocol (N non-owner evaluators, each command × N, logs required):
1. **Accuracy / macro-F1** on clean held-out (speaker-disjoint)
2. **Robustness:** same eval at SNR 20/10/5 dB
3. **Latency:** p50/p95 inference per 1 s window (CPU, RPi if possible)
4. **Model size:** params + on-disk footprint (ONNX)
5. **Task-completion rate** in the live demo (N evaluators, each command N times)
6. **WER of the recognized command** (optional, only if we add a decoder)
7. **Power draw — gate vs VCM** (proposed 2026-09-27, optional, differentiator):
   average mW of (a) wake gate running continuously on idle audio vs (b) VCM
   per wake event, over a fixed 10-min script (N wake events + idle audio,
   both configs, Pi power-rail meter). Expected: gate ~10–100× cheaper per
   unit time — the quantified justification of the bouncer/receptionist split
   (see WHYS.md).

## Future work (contingency, NOT in scope for demo)
- **Fixed-vocab CTC mini-ASR** — if evaluators use novel phrasings the classifier
  wasn't trained on, a CTC ASR over the ~35-word vocabulary of the 10 commands
  (unique words across all command forms) generalizes to *compositions of known
  words* (e.g. "switch the lights on now") without re-recording, at the cost of
  a text→intent layer. Scale: est. **~100–500K params** (my estimate — order of
  KWS, not ASR). Reference points (tool-verified 2026-09-28):
  - **Keyword spotting (KWS):** Google Speech-Commands CNN class ≈ our 94K VCM;
    `priyadeepjaiswal9c/tiny-kws` = **119K params** (GitHub, verified)
  - **Wav2Small:** wav2vec2 distilled to **72K params** (arXiv:2408.13920, verified)
  - **wav2vec2-base:** ~94M params (prior knowledge, standard figure)
  - **Whisper tiny:** ~39M params (prior knowledge, arXiv:2212.04356)
  Spec-compliance: on-device, no LLM, no cloud — clears constraints (6)/(7);
  only a purist "just pure VCM" reading could object, defensible at KWS-tier
  size. **Decision: do NOT build for the demo** — classifier + multi-phrasing
  TTS covers the demo; this is the documented "next step" line for the writeup.

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
- **RPi availability** — RESOLVED (2026-09-25): **RPi 4B 8GB (BORROWED) +
  64GB SanDisk Ultra SD (boss's, bought 09-27) + iPad 10 20W charger +
  Acer Nitro 5 as dev host** — full spec in **HARDSPEC.md** (canonical).
  Board is unflashed as of 09-27; setup steps there. The SD card is the demo bottleneck
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
