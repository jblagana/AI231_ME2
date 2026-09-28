# DATASETS — classmate dataset compilation (AI231 ME2)

**Logged by:** muji (boss session) · **2026-09-29**
**For:** the AI231 ME2 session — read after v1f training lands, before the
real-human / robustness phase.
**Source:** the shared compilation sheet `AI231_ME2.xlsx` (12 rows, `Datasets`
sheet) + the AI 231 THFY Telegram chat (dataset links) + the two public
classmate repos (fetched and verified 2026-09-29).

## Why this file exists
v1f is trained on the synthetic edge-tts set (40 voices, speaker-disjoint).
The next phase is robustness — getting **real-human** clips of the 10 tasks so
the model responds to ANY person (Sir's ruling). This is the verified map of
what real-human + synthetic data the class group already has, so we don't
re-hunt it.

## Real-human corpora (the ones that matter for robustness)

| Dataset | Contributor | Covers which of the 10 tasks | Notes |
|---|---|---|---|
| **SLURP** | Mark | lighting, music, volume, temperature, alarm, weather, time | Spoken Language Understanding Resource Package — ~23k smart-home commands, real people, real rooms. **The main real corpus.** |
| **Fluent Speech Commands** | Mark | 31 intents incl. lights/music/volume/temp | 97 speakers, 248 phrases, action/object/location slots. |
| **Snips SLU** | Quiel | smart-home appliances (lighting) | Real, English, has speaker demographics. |
| **Timers and Such** ⭐ | Quiel | **TIMER + ALARM specifically** | NeurIPS 2021 benchmark, real human SetTimer/SetAlarm. **Only 2,151 real utterances across all 4 intents.** 12.2GB zip — the one still downloading as of 09-29. |
| **Common Voice** (Mozilla) | King | general / any phrase you filter | 200k recordings, **has Filipino-accent recordings**. CC-licensed, on HuggingFace, filterable by `sentence` column. |
| **GSC v0.02** | Mark + Joven | single words (on/off/stop) + background noise | Real, but single-word only — use the noise set for the noisy condition, not the commands. |

## Synthetic (fill gaps, not the "real" answer)

| Dataset | Contributor | Scale |
|---|---|---|
| **eSpeak NG** | Mark | 10,000 (20 labels × 10 voices × 10 phrases × 5 variations) |
| **Chatterbox TTS** + LibriSpeech + Common Voice refs | Mark | 9,000 (20 labels × 30 refs × 5 phrases × 3 variants; 24 LibriSpeech + 6 Filipino-English refs) |
| **SynTTS-Commands** (cosy2) | Cherry | 14k (cmd 1) + 53k (cmd 8) |
| **jbuchner synthetic** | Joven | one-word, prune to off/on/stop |

## Classmate repos (public, fetched + verified 2026-09-29)
- **Mark's Option B** — `github.com/markandrian30/AI231/tree/main/MEX2/OptionB`
  — 28,800 synthetic WAVs, 150 speakers (134 foreign + 16 Filipino-English),
  19 intents, clean + noisy conditions, 27,956 active after whisper
  filtering. TTS-generated (Chatterbox/Piper) over real reference voices.
  Already splits `WEATHER` and `TIME` into separate intents (our robustness
  change #2 is baked into Mark's schema).
- **Ayla's voice pool** — `github.com/ayla011/ai231-me2-voice-data`
  — the **real-human recording pipeline**: classmates record a fixed prompt
  list, whisper.cpp validates each take on the spot, approved clips go to the
  shared GDrive pool. This is the "50–100 real-human clips/class" mechanism
  for robustness #3, already being built.

## GDrive pools (login-gated — boss's account)
- `drive.google.com/drive/folders/1_GcDuvaRnGlkdtGdoogQI7FflgWf6gF8` — pooled
  real voice records (student-ID speaker labels)
- `drive.google.com/file/d/1miLmuozdojOylqD0xlfOwXAdNsbGYwfN/view` — someone's
  "Frankenstein Dataset"
- `drive.google.com/drive/folders/15MwS2UpPgaAUL-zJVbEV8tl_oQ38avp2` — slop
  outputs (transcription <80%, to be dropped)
- Master index: `docs.google.com/spreadsheets/d/1VFm1-SAdNqtSwOeSHZct23tF6pPTntmj2940sugj61E`
  (the shared compilation — the sheet this file was built from)

## The takeaway (what to actually do)
1. **No public dataset has our 10 commands pre-labeled** — they're our spec,
   not a benchmark. The real-human answer is a **combination**:
   - **SLURP + Fluent Speech Commands + Snips SLU** → lighting/music/volume/temperature
   - **Timers and Such** → TIMER + ALARM (the specific real set; lands with the 12.2GB download)
   - **Common Voice** (filter by `sentence`, has Filipino accents) → novel phrasings + Filipino-accent fill
   - **Ayla's pooled recordings** → real clips of our *exact* phrases (the strongest match)
2. **Synthetic (Mark's Option B + eSpeak NG + SynTTS)** fills the gaps to hit
   50–100 clips/class where real data is thin.
3. **Common Voice filtering is now the fallback, not the primary** — the class
   group already has real-human coverage of all 10 tasks.
4. **GSC v0.02 noise set** is the source for the noisy-condition eval (not the commands).

## Verification status (honest)
- Dataset **names + descriptions**: from the classmates' own annotations in
  the sheet (their verified notes) + the two **public GitHub repos** fetched
  this session.
- **Not re-fetched this turn:** each dataset's URL (the sheet's descriptions
  are detailed enough). If a class's coverage is load-bearing, pull that
  dataset's intent list from source before relying on it.
- **Not openable:** the Google Sheet + GDrive folders (login-gated) — reported
  from the chat + the xlsx the boss uploaded.
