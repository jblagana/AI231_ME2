"""Generate site/ assets for the project dashboard (site/index.html).

Picks 3 eval-split clips per class (deterministic: first 3 in manifest),
copies them to site/samples/, and renders a log-mel spectrogram PNG for each
using the SAME feature pipeline as train.py (torchaudio MelSpectrogram).

Usage:
  .venv/Scripts/python.exe src/gen_site_assets.py
"""
import json
import shutil
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from model import N_MELS, SR  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SITE = ROOT / "site"
SAMPLES = SITE / "samples"
N_SAMPLES = 3


def load_mp3(p: Path) -> torch.Tensor:
    """Decode mp3 -> (1, T) @16k mono float32.

    torchaudio 2.11 in this venv routes all decoding through torchcodec
    (broken install), so decode via the imageio_ffmpeg binary directly.
    """
    import subprocess

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run(
        [ff, "-v", "error", "-i", str(p), "-ar", str(SR), "-ac", "1",
         "-f", "s16le", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(y).unsqueeze(0)


def logmel(wav: torch.Tensor) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=320, n_mels=N_MELS,
        f_min=50.0, f_max=8000.0,
    )(wav)
    return torch.clamp(mel, min=1e-5).log10()


def main() -> int:
    # clean previous assets so stale files never linger
    if SITE.exists():
        for f in SITE.iterdir():
            if f.name not in ("index.html", "index_template.html"):
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in (RAW / "manifest.jsonl").read_text().splitlines()]
    by_class: dict[str, list[dict]] = {}
    for m in manifest:
        if m["split"] == "eval":
            by_class.setdefault(m["class"], []).append(m)

    picked = []
    for ci, cls in enumerate(sorted(by_class)):
        # 3 distinct voices per class, rotated by class index (manifest is
        # voice-sorted, so "first 3 distinct" would be the same 3 voices
        # for every class).
        distinct: list[dict] = []
        seen: set[str] = set()
        for m in by_class[cls]:
            if m["voice"] in seen:
                continue
            seen.add(m["voice"])
            distinct.append(m)
        rot = distinct[ci * 3:] + distinct[: ci * 3]  # rotate by class index
        for m in rot[:N_SAMPLES]:
            src = RAW / m["file"]
            if not src.exists():
                continue
            stem = f"{cls}__{src.stem}"
            shutil.copy2(src, SAMPLES / f"{stem}.mp3")
            wav = load_mp3(src)
            # show the model's actual input: first 1 s window (clips run ~2.5-3 s)
            if wav.shape[-1] > SR:
                wav = wav[..., :SR]
            lm = logmel(wav).squeeze(0).T  # (frames, mels) for plotting
            fig, ax = plt.subplots(figsize=(12, 6.4), dpi=100)
            im = ax.imshow(lm.numpy(), origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(f"{cls}  —  \"{m['phrase']}\"  ({m['voice']})  ·  model input: 1 s window (80 mel x 50 frames)")
            ax.set_xticks(range(0, 50, 10))
            ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8"])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("mel bin")
            fig.colorbar(im, ax=ax, label="log10 power")
            fig.tight_layout()
            fig.savefig(SITE / f"spectro_{stem}.png")
            plt.close(fig)
            picked.append({"class": cls, "stem": stem, "voice": m["voice"], "phrase": m["phrase"]})

    samples_json = json.dumps(picked, indent=1)
    (SITE / "samples.json").write_text(samples_json)
    # inline the data into index.html so the page is self-contained
    # (no fetch — works from file:// and from the muji preview endpoint).
    # index_template.html is the source of truth (keeps the marker);
    # index.html is the generated, ready-to-open output.
    marker = "/*__SAMPLES__*/"
    template = (SITE / "index_template.html").read_text()
    if marker not in template:
        raise SystemExit("index_template.html lost its /*__SAMPLES__*/ marker")
    (SITE / "index.html").write_text(template.replace(marker, samples_json))
    print(f"{len(picked)} samples -> {SAMPLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
