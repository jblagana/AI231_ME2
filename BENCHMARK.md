# Benchmark proposal — AI231 ME2 VCM validation (for the collective)

Proposed by Jan (ai231-me2). Fits the 09-24 group protocol (N non-owner
evaluators, each command N times, output logs required).

## 1. Offline metrics (everyone reports the same 5 numbers)
| Metric | Definition |
|---|---|
| **Accuracy / macro-F1** | On a held-out set, speaker-disjoint (eval voices never in training) |
| **Robustness** | Same eval set at SNR 20 / 10 / 5 dB (background noise mixed in) |
| **Latency** | p50 / p95 inference time per 1 s window, on the target device (RPi 4/5) |
| **Model size** | Param count + on-disk footprint (exported, e.g. ONNX) |
| **WER** | (optional) only if the model outputs text |

## 2. Live demo metric (the one that counts)
- **Task-completion rate:** N non-owner evaluators, each says each of the 10
  commands once (or N times per the group protocol), the demo runs, we log
  (command, predicted class, correct?) → accuracy = correct / total.
- Required log format (one line per trial):
  `evaluator_id, trial, command, audio_file, predicted_class, correct (0/1), latency_ms`

## 3. Rules so the numbers are comparable
1. Everyone evaluates on the **same 10 classes** (spec list, as-is).
2. Eval audio must be from people **not in the training set** (that's the
   whole point of speaker-disjoint splits — if your training data includes the
   evaluators' voices, your accuracy is fake).
3. Report both **clean** and **noisy** numbers — a model that only works in a
   silent room is not a smart-device model.
4. Device: report which RPi (4 or 5) and which runtime (PyTorch / ONNX).

## Our contribution (Jan)
- Dataset: TTS-synthesized, 40 voices across 10 English accents,
  speaker-disjoint 20/20 split, speed jitter + noise augmentation.
  Manifest + generator in this repo (`src/make_dataset.py`).
- Model: 3-block CNN, ~94K params, 80-bin log-mel, 1 s window.
- Benchmark harness: coming (scoring script that ingests the log format above).
