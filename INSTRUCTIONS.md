# INSTRUCTIONS — AI231 ME2 · tiny on-device Voice Command Model (VCM)

Workspace rule (workspace root `AGENTS.md`, "Instruction log rule"): every user
instruction is logged **verbatim, newest first, before any work on it**. The
Interpretation section is agent-owned; the user may edit it — **a user edit is
the new instruction** (stop, log it, follow it). Re-read this file before each
log update and before `git push`. `done` = **pushed to the remote repo**.

---

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
Status: done
- Local: `C:\Users\Jan\.cline\data\workspaces\chat\AI231_ME2` (git init, main)
- Remote: jblagana/AI231_ME2 (public)
- Layout: `src/` (commands, make_dataset, model, train), `notebooks/`, `data/`
  (gitignored), `runs/` (gitignored checkpoints)
