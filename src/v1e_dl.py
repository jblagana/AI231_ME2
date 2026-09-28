"""v1e: MUSAN noise -> 16 kHz mono .npy bank for train.py --real-noise.

Source: huggingface.co/datasets/FluidInference/musan (MUSAN noise subset,
free-sound + sound-bible, ~558 MB, CC-BY-4.0). Output: data/noise16k/*.npy,
each >= 1 s @16 kHz float32.

Decode is soundfile (libsndfile), NOT torchaudio.load — torch 2.14 on n003
routes torchaudio.load through torchcodec, which is not installed in the
vcm env (measured 2026-09-28: every load raised "TorchCodec is required",
first pass kept=0). soundfile reads the same files fine (16 kHz mono PCM_16
verified on noise-free-sound-0000.wav).

Run: python src/v1e_dl.py   (idempotent — skips already-converted files)
"""
import glob
import os

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """1-D linear resample (numpy), no scipy dependency."""
    if sr_in == sr_out:
        return x
    T = len(x)
    T2 = int(round(T * sr_out / sr_in))
    src = np.linspace(0.0, T - 1, T2)
    i0 = np.clip(src.astype(np.int64), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    frac = (src - np.floor(src)).astype(np.float32)
    return x[i0] * (1 - frac) + x[i1] * frac


def main():
    snapshot_download(
        repo_id="FluidInference/musan", repo_type="dataset",
        allow_patterns=["noise/free-sound/*.wav", "noise/sound-bible/*.wav"],
        local_dir=os.path.join(ROOT, "data", "noise_hf"))
    out_dir = os.path.join(ROOT, "data", "noise16k")
    os.makedirs(out_dir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(
        ROOT, "data", "noise_hf", "noise", "**", "*.wav"), recursive=True))
    kept = skipped = 0
    for p in wavs:
        name = os.path.basename(p)[:-4]
        out = os.path.join(out_dir, name + ".npy")
        if os.path.exists(out):
            kept += 1
            continue
        try:
            w, sr = sf.read(p, dtype="float32")
        except Exception as e:
            skipped += 1
            print("LOAD FAIL", p, repr(e)[:100], flush=True)
            continue
        if w.ndim > 1:  # mono
            w = w.mean(axis=1)
        w = resample_linear(w, sr, 16000)
        if len(w) < 16000:  # need >= 1 s
            skipped += 1
            continue
        np.save(out, w.astype(np.float32))
        kept += 1
    print(f"kept={kept} skipped={skipped} -> {out_dir}", flush=True)
    print("NOISE_DL_DONE", flush=True)


if __name__ == "__main__":
    main()
