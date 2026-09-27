# WHYS — architecture & method rationale (AI231 ME2 VCM)

The "why" behind each architecture decision. Source of truth for the
design/methodology section of the writeup and for class presentation.
Newest rationale first; each entry: decision → why → what was rejected.

## 2026-09-27 — Why the wake gate is a *separate* model, not a head on the VCM

**Decision:** two models — a tiny 2-class **gate** (wake / no-wake) always on
in front of the VCM; the VCM only runs after a wake fires.

**Why separate (four independent reasons):**
1. **Run frequency.** The wake question must be answered 100% of the time
   (every ~1 s chunk of ambient audio); the intent question only runs *after*
   a wake. If wake were a head on the VCM, the full VCM backbone would have to
   run on every chunk, all day, just to catch the wake word — deleting the
   power-saving split that is the design's point (matters on a Pi running
   off battery/SD continuously).
2. **Different input windows.** The gate needs a sliding window to catch
   "hey muji" wherever it lands in the stream; the VCM needs the command
   window (the ~1 s *after* the wake). One backbone can't be fed both
   windows at once without degrading the gate.
3. **Different training data & scale.** Gate trains on ~1–2k clips
   (only the positive class needs synthesizing — no-wake is a free
   catch-all: silence, music, other speech). VCM trains on ~13.8k. A shared
   backbone gets dragged toward the minority class's features, and the
   gate's false-wake rate — the metric that matters most — becomes hostage
   to the VCM's training dynamics.
4. **Different failure budgets.** A false *wake* costs one wasted VCM run
   (cheap); a false *intent* costs an unwanted device action (visible,
   annoying). The gate must be **conservative** (high threshold, fires
   rarely); the VCM must be **accurate**. One model, one loss, one
   threshold can't hold two risk profiles.

**Terminology note (for the slides):** a *head* is a sub-network (an FC
layer) producing an output; a *class* is a category inside one head's
output. "2-class gate" = one head, two answers. "Multi-head VCM" = several
FC layers side by side.

**Rejected:** 3-head single model (wake + intent + slot) — see reasons 1–4.

## 2026-09-27 — Why a plain CNN (not CRNN / RNN) for the VCM

**Decision:** small CNN (3 conv blocks + global avg-pool + FC), ~94K params.

**Why CNN (four reasons):**
1. **No sequence left to read.** 1 s audio → 50 mel frames → 3 maxpool
   stages (50 → 25 → 12 → 6). By the time an RNN would sit, the temporal
   axis is 6 frames — too short for recurrent modeling to add anything,
   and the global avg-pool already collapses the time axis.
2. **Wrong output shape for CRNN.** CRNN (CNN + RNN + CTC) exists for
   *variable-length sequence* outputs (ASR characters). Our output is one
   label per clip — a sequence-emitting architecture is pure overhead.
3. **Pi-specific compute.** CNNs are dense matrix ops (vectorize well on
   ARM); RNNs are *sequential* compute (timestep t depends on t−1) — the
   worst workload for the Pi's ARM cores. A GRU on 128-dim features alone
   is ~100K params — the entire current model added on top.
4. **Precedent.** Google's on-device keyword-spotting model (Pixel phones)
   is a small CNN over mel spectrograms — same architecture class, same
   ~1 s regime. The field converged here for these reasons.

**Where the RNN instinct is right (documented fallback):** the *slot* head
(Tier 1) is position-sensitive — numbers land at the *end* of the
utterance ("dim the lights to **fifty**"). If plain-CNN slot accuracy
disappoints, a temporal variant (small GRU over the pooled frame sequence)
is the first fix to try, for the slot head specifically — not the intent
head.

## 2026-09-27 — Why parametric commands are class-only in v1, with a Tier-1 upgrade path

**Decision:** v1 detects parametric commands by class only (default slot
values in the rule layer); Tier 1 = joint intent + slot heads.

**Why this is tractable (the key insight):** the number space is
*practically* finite. What humans actually say to a device is a small grid:
brightness {10…100 by 10}, timer minutes {1, 2, 5, 10, 15, 30, 60}, temps
{18, 20, 22, 24, 25, 28}, clock times {5am, 6am, 6:30am, …}. ~30–40 slot
tokens, not a continuum — so it's a second *classification* problem, not
regression or ASR.

**Tier 1 design (stretch goal / future work):** second FC head on the same
backbone (head A → 10 intents, head B → ~35 slot tokens + "no number"),
trained with summed cross-entropy `loss = CE(intent) + CE(slot)`.
Cost: hundreds of extra params (still ~94K total). Data: mostly free —
existing TTS clips already embed numbers; just relabel from the phrase text
(deterministic). Strictly additive over v1 (head A alone = v1 behavior).
Collective contract: no conflict — 10-class list and all 5 benchmark
metrics unchanged; slot accuracy is an extra log column; Day-5 pooled data
needs `phrase_text` in the contribution format (or slot head trains on
labeled clips only).

**Rejected tiers:** (2) sequential slot sub-classifier — same data, more
plumbing, no benefit over joint. (3) ASR for slots (Vosk/Whisper-tiny) —
only truly open-vocabulary tier, but a second model + memory + kills the
"one 94K-param model" story.

## 2026-09-27 — Why ONNX Runtime (and not PyTorch) on the Pi

**Decision:** train in PyTorch → export ONNX → run ONNX Runtime (CPU EP,
ARM64) on the Pi. Single inference artifact = `.onnx`.

**Why:** PyTorch on the Pi is a ~2GB install with slow ARM CPU kernels;
ONNX Runtime is a ~50MB wheel with optimized ARM kernels — real difference
on a 64GB SD card (the demo bottleneck). ONNX is a format, not a framework:
the same file runs on the Pi, the dev machine, and the web. At 94K params
the forward pass is single-digit ms — comfortably real-time for 1 s clips.
Multi-output graphs (Tier 1's two heads) export with zero special handling.
**TensorRT rejected:** no ARM64/RPi build (x86 + Jetson only); irrelevant at
94K params anyway.

## 2026-09-25 — Why TTS (edge-tts) instead of human recording

**Decision:** synthesize the dataset with edge-tts, many voices × speeds.

**Why:** constraint (7) of the spec is on *inference* (on-device), not data
generation. Human recording won't scale to ~10–20k clips in 7 days; TTS
scales in an afternoon. Mitigations for synthetic-speech uniformity: noise
augmentation (SNR 0–20 dB, Google environmental noise / MUSAN), speed
jitter, speaker-disjoint split (some voices train-only, some eval-only —
prevents voice memorization), and the live human-eval benchmark item.

## 2026-09-25 — Why 10 classes as-is, "play music" kept separate

**Decision:** spec's 10-class list unchanged; media control =
pause/stop/next/volume as its own class.

**Why:** the collective benchmark compares models on the *same* class list
(09-24 group protocol) — deviating makes our numbers incomparable. "Play
music" is a command in its own right (start playback), distinct from
controlling playback.

## 2026-09-25 — Why a wake word at all (in scope, boss-ratified post-spec)

**Decision:** wake word in scope; separate 2-class gate in front of the VCM.

**Why:** the 09-22 group protocol requires it — wake word mandatory while
music plays, volume drops to 5% during listening. Without a gate, the VCM
would fire on ambient speech and music lyrics. Implementation choice
(boss to veto): tiny 2-class CNN reusing the TTS pipeline (fully on-device,
no vendor dependency, same code path as the main model) vs Porcupine
(Picovoice, free personal license, battle-tested but a vendor dependency in
an on-device/no-cloud project).

## Proposed benchmark addition (2026-09-27, pending boss OK to post)

**6. Power draw — gate vs VCM** (optional, differentiator): measure
average power consumption (mW) of (a) the wake gate running continuously on
idle audio vs (b) the VCM running per wake event, over a fixed session.
Expected result: gate is ~10–100× cheaper per unit time, which is the
quantified justification of the bouncer/receptionist split. Method: Pi
power rail meter (or `vcgencmd` / PSU draw) over a fixed 10-min script:
N wake events + idle audio, same for both configs.
