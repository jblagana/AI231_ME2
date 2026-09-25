# AGENTS.md — AI231 ME2 workspace

## Instruction log rule
Every user instruction is logged **verbatim, newest first** in `INSTRUCTIONS.md`,
before any work on it. The Interpretation section is agent-owned; a user edit is
the new instruction (stop, log it, follow it). Re-read before each log update and
before `git push`. `done` = **pushed to the remote repo** (jblagana/AI231_ME2).

## Project ground truth
- Specs: `C:\Users\Jan\Muji\ai231_me2_specs.md` (boss-confirmed 2026-09-25)
- Plan + status: `PLAN.md` (re-read at session start; update as you go)
- Deadline: **Sat 2026-10-03**
- Hard constraints: no ASR, no LLM, no cloud at inference, tiny (RPi 4/5 real-time)

## Ops
- Python: `.venv` (CPU torch + edge-tts + soundfile). No GPU on this machine.
- Data lives in `data/` (gitignored); code in `src/`; notebooks in `notebooks/`.
- Commit only this repo's work; commit then push.
