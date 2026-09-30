"""Generate the A2 "wrong clips" dashboard assets (site/index.html section).

Reads runs/v1a2_wrong_clips.json + runs/v1a2_wrong_audio/*.wav (the 42 eval
clips the A2 two-head model got wrong — 24 slot, 18 command), renders each
clip's mel with the EXACT A2 input pipeline (3.0 s right-aligned window,
80 mel x 150 frames, same MelSpectrogram params + (log10+5)/5 norm as
train_a2.wav_to_logmel), and injects the data into site/index_template.html.

Labels come from the JSON (written by the n002 eval run that produced the
checkpoint — the ckpt itself lives on n002 and is not in this tree).

Usage:
  .venv/Scripts/python.exe src/gen_wrong_assets.py
"""
import json
import shutil
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
WRONG_JSON = ROOT / "runs" / "v1a2_wrong_clips.json"
WRONG_AUDIO = ROOT / "runs" / "v1a2_wrong_audio"
WRONG_OUT = SITE / "wrong"

sys.path.insert(0, str(Path(__file__).parent))
from model_a2 import N_MELS, N_FRAMES, SR, SLOT_TAIL_CELLS  # noqa: E402

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=1024, hop_length=320, n_mels=N_MELS,
    f_min=50.0, f_max=8000.0,
)


def load_wav(p: Path) -> torch.Tensor:
    with wave.open(str(p), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        raw = w.readframes(n)
    assert sr == SR, f"{p}: {sr} Hz != {SR}"
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(y).unsqueeze(0)


def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    """Same as train_a2.wav_to_logmel (right-aligned 3.0 s window)."""
    if wav.shape[-1] > SR * 3:
        wav = wav[..., -int(SR * 3):]
    m = _MEL(wav)
    logmel = torch.clamp(m, min=1e-5).log10()
    logmel = (logmel + 5.0) / 5.0
    if logmel.shape[-1] < N_FRAMES:
        pad = N_FRAMES - logmel.shape[-1]
        logmel = torch.nn.functional.pad(logmel, (pad, 0))  # right-align
    return logmel[..., :N_FRAMES]


def render_mel(wav: torch.Tensor, stem: str, title: str, dur: float) -> None:
    lm = wav_to_logmel(wav).squeeze(0).T  # (150, 80)
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=100)
    im = ax.imshow(lm.numpy(), origin="lower", aspect="auto", cmap="viridis")
    # shade the slot head's read region (last SLOT_TAIL_CELLS of 18 cells)
    x0 = N_FRAMES - SLOT_TAIL_CELLS * 10  # 150 - 80 = 70
    ax.axvspan(x0, N_FRAMES, color="#e06c75", alpha=0.18)
    ax.text(x0 + 2, 76, "slot head reads", color="#e06c75", fontsize=8)
    if dur < 3.0:
        # left pad band = added zeros (the dark left band on short clips);
        # exact raw frame count from the same MelSpectrogram (center=True:
        # frames = 1 + N // hop) — read straight off the tensor, not guessed
        raw_frames = _MEL(wav).shape[-1]
        pad = N_FRAMES - raw_frames
        ax.axvspan(0, pad, color="#5aa9ff", alpha=0.25)
        ax.text(pad / 2, 76, f"pad {pad / 50:.2f} s",
                color="#5aa9ff", fontsize=8, ha="center")
    else:
        # tail-cropped: the first (dur - 3.0) s of the clip is off-window
        cropped = dur - 3.0
        ax.text(2, 76, f"first {cropped:.2f} s cropped (off-window)",
                color="#d9a441", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 30, 60, 90, 120, 149])
    ax.set_xticklabels(["0", "0.6", "1.2", "1.8", "2.4", "3.0"])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax, label="log10 power")
    fig.tight_layout()
    fig.savefig(WRONG_OUT / f"mel_{stem}.png")
    plt.close(fig)


def main() -> int:
    clips = json.loads(WRONG_JSON.read_text())
    WRONG_OUT.mkdir(parents=True, exist_ok=True)

    wrong = []
    for c in clips:
        stem = Path(c["file"]).stem
        key = (f"slot_{c['cls']}_{stem}" if c["kind"] == "slot"
               else f"cmd_{c['true']}__to__{c['pred']}_{stem}")
        wav_p = WRONG_AUDIO / f"{key}.wav"
        if not wav_p.exists():
            print(f"  MISSING {wav_p}")
            continue
        shutil.copy2(wav_p, WRONG_OUT / f"{key}.wav")
        wav = load_wav(wav_p)

        dur = wav.shape[-1] / SR  # raw seconds (pre-window)
        row = {
            "key": key, "kind": c["kind"], "phrase": c["phrase"],
            "cmd_true": c["true"],
            "cmd_pred": c.get("pred") if c["kind"] == "cmd" else None,
            "conf": c.get("slot_conf") if c["kind"] == "slot" else c.get("cmd_conf"),
            "dur": round(dur, 3), "over3": dur > 3.0,
        }
        if c["kind"] == "slot":
            row["cls"] = c["cls"]
            row["slot_true"] = c["true"]
            row["slot_pred"] = c["pred"]
            title = (f'"{c["phrase"]}"  ·  slot: {c["true"]} → '
                     f'{c["pred"]}  ({c["slot_conf"]:.2f})')
        else:
            title = (f'"{c["phrase"]}"  ·  command: {c["true"]} → '
                     f'{c["pred"]}  ({c["cmd_conf"]:.2f})')
        render_mel(wav, key, title, dur)
        wrong.append(row)
        print(f"  {key}")

    # slot first (the interesting failures), then cmd; conf desc within
    wrong.sort(key=lambda r: (r["kind"] != "slot", -(r["conf"] or 0)))

    data = {"wrong": wrong,
            "n_wrong": len(wrong),
            "n_slot": sum(1 for r in wrong if r["kind"] == "slot"),
            "n_cmd": sum(1 for r in wrong if r["kind"] == "cmd")}
    (SITE / "wrong.json").write_text(json.dumps(data, indent=1))

    # inject into the template — BOTH markers (wrong data fresh, samples data
    # from the existing samples.json) so index.html stays self-contained.
    template = (SITE / "index_template.html").read_text()
    for marker in ("/*__WRONG__*/", "/*__SAMPLES__*/"):
        if marker not in template:
            raise SystemExit(f"index_template.html lost its {marker} marker")
    samples = (SITE / "samples.json").read_text()
    (SITE / "index.html").write_text(
        template.replace("/*__WRONG__*/", json.dumps(data))
                .replace("/*__SAMPLES__*/", samples.strip()))
    print(f"\n{len(wrong)} wrong clips -> {WRONG_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
