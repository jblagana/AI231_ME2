"""A2 — train the two-head VCM (command + slot) on the 3.0 s window.

Self-contained: does NOT import model.py / train.py (those stay on the 1.0 s
window for the v1i baseline). Reuses the same fast data-path utilities
(RAM raw cache, singleton MelSpectrogram, base-feature cache, mel-domain
augmentation) but at 150 frames (3.0 s) instead of 50 (1.0 s).

Data: data/raw_v1i/{split}/{class}/*.raw (s16le @16 kHz) + manifest.jsonl
(speaker-disjoint splits; manifest carries the phrase for slot extraction).

Loss:  CE_cmd (all clips) + SLOT_W * CE_slot (parametric clips with a slot).
       Slot labels come from slots.extract_slot(class, phrase); clips without
       a slot value (e.g. "dim the lights", "lower the lights") are excluded
       from the slot loss.

Recipe: v1g (jitter 0.95-1.05, gain +-6 dB, SNR 5-25 dB; real MUSAN noise via
--real-noise). v1k was tested and failed, so v1g is the baseline recipe.

Usage:
  python src/train_a2.py --data data/raw_v1i --epochs 50 --batch 32 \
      --real-noise data/noise16k --out runs/v1a2
  python src/train_a2.py --smoke     # tiny pipeline check
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from slots import extract_slot, PARAMETRIC, SLOT_VOCAB  # noqa: E402
from model_a2 import VCMTwoHead, N_MELS, N_FRAMES, SR  # noqa: E402

# 11 classes, same order as the v1i manifest (single source of truth).
CLASSES = [
    "play_music", "ask_weather", "control_lights", "dim_lights",
    "set_timer", "set_alarm", "set_temperature", "media_control",
    "set_reminder", "make_call", "ask_time",
]

SLOT_W = 0.5  # slot loss weight (PLAN.md A2)
WINDOW_S = 3.0

RAW_CACHE = {}
BASE_FEAT_CACHE = {}
NOISE_BANK = []

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=1024, hop_length=320, n_mels=N_MELS,
    f_min=50.0, f_max=8000.0,
)


def wav_to_logmel(wav: torch.Tensor, mel: torch.nn.Module = None) -> torch.Tensor:
    """(1, T) @16k -> (1, 80, 150) log-mel, RIGHT-aligned in the 3.0 s window.

    Right-aligned (pad on the LEFT) so the slot word — the LAST spoken word of
    the phrase — always sits at the TAIL of the feature map, where the slot
    head reads (model_a2.SLOT_TAIL_CELLS). A left- or center-alignment would
    put a short clip's slot word (e.g. "call mom") in the middle of the map,
    OUTSIDE the tail slice, so the slot head would read silence. The command
    head uses global max pooling (alignment-invariant), so right-alignment
    costs it nothing. Clips longer than 3.0 s keep their LAST 3.0 s (the slot
    word is at the end, so we must not crop the tail)."""
    if mel is None:
        mel = _MEL
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[-1] > SR * WINDOW_S:
        # keep the TAIL (last 3.0 s) — the slot word is the last word
        wav = wav[..., -int(SR * WINDOW_S):]
    m = mel(wav)
    logmel = torch.clamp(m, min=1e-5).log10()
    logmel = (logmel + 5.0) / 5.0
    if logmel.shape[-1] < N_FRAMES:
        pad = N_FRAMES - logmel.shape[-1]
        logmel = nn.functional.pad(logmel, (pad, 0))  # right-align: pad LEFT
    return logmel[..., :N_FRAMES]


def precompute_base_features(ds, dev: torch.device, mel: torch.nn.Module) -> int:
    """Compute every base log-mel ONCE, on `dev` (GPU), into BASE_FEAT_CACHE.

    The 3.0 s mel is ~8x the 1.0 s mel work; on CPU it's ~354 ms/item ->
    ~135 min for 22.8k clips (measured _a2_meltiming.py, 2026-09-30). On the
    A100 the mel matmul is ~100x faster (~2-5 ms/item), so the whole precompute
    is ~2 min. Features are stored on CPU in BASE_FEAT_CACHE (~1.1 GB) so the
    per-item training path stays a dict lookup. Returns the count cached."""
    t0 = time.time()
    n = 0
    for p, _, _, _ in ds.items:
        key = str(p)
        if key in BASE_FEAT_CACHE:
            continue
        w = load_wav(p).to(dev)
        feat = wav_to_logmel(w, mel).cpu()
        BASE_FEAT_CACHE[key] = feat
        n += 1
    print(f"base features: {len(BASE_FEAT_CACHE)} cached (GPU {dev}) in "
          f"{time.time() - t0:.0f}s", flush=True)
    return len(BASE_FEAT_CACHE)


def _resample_mel(x: torch.Tensor, T2: int) -> torch.Tensor:
    """Linearly resample a mel feature's frame axis to T2 frames."""
    T = x.shape[-1]
    if T2 <= 1:
        return x[..., :1]
    if T2 == T:
        return x
    src = torch.linspace(0.0, T - 1, T2)
    i0 = src.long().clamp(max=T - 1)
    i1 = (i0 + 1).clamp(max=T - 1)
    frac = (src - src.floor()).float().view(1, 1, -1)
    return x.index_select(2, i0) * (1 - frac) + x.index_select(2, i1) * frac


def load_wav(p: Path) -> torch.Tensor:
    raw_p = p.with_suffix(".raw")
    if raw_p.exists():
        arr = RAW_CACHE.get(str(raw_p))
        if arr is None:
            arr = (np.fromfile(str(raw_p), dtype=np.int16)
                   / 32768.0).astype(np.float32)
            RAW_CACHE[str(raw_p)] = arr
        return torch.from_numpy(arr).unsqueeze(0)
    raise FileNotFoundError(f"no .raw for {p} (run src/decode_wav.py)")


def preload_raw(root: Path) -> int:
    t0 = time.time()
    n = 0
    for raw_p in sorted(root.rglob("*.raw")):
        key = str(raw_p)
        if key not in RAW_CACHE:
            RAW_CACHE[key] = (np.fromfile(str(raw_p), dtype=np.int16)
                              / 32768.0).astype(np.float32)
        n += 1
    print(f"preload: {n} raw files cached in RAM in {time.time() - t0:.1f}s",
          flush=True)
    return n


def load_noise_bank(noise_dir: Path, max_sec: float = 4.0, mel: torch.nn.Module = None) -> int:
    if mel is None:
        mel = _MEL
    dev = next(mel.buffers()).device  # MelSpectrogram holds buffers only
    t0 = time.time()
    for f in sorted(Path(noise_dir).glob("*.npy")):
        w = torch.from_numpy(np.load(f)).unsqueeze(0).to(dev)
        if w.shape[-1] > int(SR * max_sec):
            w = w[..., :int(SR * max_sec)]
        NOISE_BANK.append(wav_to_logmel(w, mel).cpu())
    print(f"noise bank: {len(NOISE_BANK)} MUSAN clips cached in "
          f"{time.time() - t0:.0f}s", flush=True)
    return len(NOISE_BANK)


class VCMDatasetA2(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = root
        self.augment = augment
        self._jit = (0.95, 1.05)
        self._snr = (5.0, 25.0)
        # items: (path, cls, phrase, slot_idx) — slot_idx folded into the tuple
        # so a single shuffle keeps it aligned with its clip (a parallel
        # slot_labels list would desync after shuffle — caught in smoke).
        self.items = []
        mf = root / "manifest.jsonl"
        if not mf.exists():
            raise FileNotFoundError(f"no manifest at {mf}")
        for line in mf.open(encoding="utf-8"):
            row = json.loads(line)
            if row["split"] != split:
                continue
            cls = row["class"]
            phrase = row.get("phrase", "")
            _, slot_idx = extract_slot(cls, phrase)
            self.items.append((root / row["file"], cls, phrase, slot_idx))
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    @property
    def slot_labels(self):
        return [it[3] for it in self.items]

    def _base_feat(self, p: Path) -> torch.Tensor:
        key = str(p)
        feat = BASE_FEAT_CACHE.get(key)
        if feat is None:
            feat = wav_to_logmel(load_wav(p))
            BASE_FEAT_CACHE[key] = feat
        return feat

    def __getitem__(self, i):
        p, cls, phrase, slot_idx = self.items[i]
        x = self._base_feat(p)
        if self.augment:
            rate = random.uniform(*self._jit)
            T2 = int(round(x.shape[-1] / rate))
            x = _resample_mel(x, T2)
            if x.shape[-1] < N_FRAMES:
                pad = N_FRAMES - x.shape[-1]
                x = nn.functional.pad(x, (pad, 0))  # keep right-alignment
            elif x.shape[-1] > N_FRAMES:
                x = x[..., -N_FRAMES:]              # keep the tail
            x = x[..., :N_FRAMES]
            x = x + random.uniform(-6.0, 6.0) / 20.0
            snr_db = random.uniform(*self._snr)
            if NOISE_BANK:
                nz = random.choice(NOISE_BANK)
                W = N_FRAMES
                if nz.shape[-1] >= W:
                    start = random.randint(0, nz.shape[-1] - W)
                    nz = nz[..., start:start + W]
                else:
                    reps = -(-W // nz.shape[-1])
                    nz = nz.repeat(1, 1, reps)[..., :W]
            else:
                nz = torch.randn(1, N_MELS, N_FRAMES)
            sig_p = x.pow(2).mean().clamp_min(1e-8)
            nz_p = nz.pow(2).mean().clamp_min(1e-8)
            scale = torch.sqrt(sig_p / (nz_p * 10 ** (snr_db / 10.0)))
            x = x + nz * scale
        y_cmd = CLASSES.index(cls)
        return x, y_cmd, slot_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw_v1i")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/v1a2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--real-noise", default=None, metavar="DIR")
    ap.add_argument("--slot-w", type=float, default=SLOT_W)
    args = ap.parse_args()

    root = Path(args.data)
    train_ds = VCMDatasetA2(root, "train", augment=True)
    eval_ds = VCMDatasetA2(root, "eval", augment=False)
    if args.smoke:
        train_ds.items = train_ds.items[:64]
        eval_ds.items = eval_ds.items[:32]
        args.epochs = 1
    print(f"train={len(train_ds)}  eval={len(eval_ds)}")
    if len(train_ds) == 0 or len(eval_ds) == 0:
        print("NO DATA"); return 2

    # slot coverage stats
    def slot_stats(ds):
        n_par = sum(1 for it in ds.items if it[1] in PARAMETRIC)
        n_slot = sum(1 for it in ds.items if it[3] >= 0)
        return n_par, n_slot
    tp, ts = slot_stats(train_ds)
    ep_, es = slot_stats(eval_ds)
    print(f"slot coverage: train {ts}/{tp} parametric clips have a slot "
          f"({100*ts/max(tp,1):.0f}%)")
    print(f"slot coverage: eval  {es}/{ep_} parametric clips have a slot "
          f"({100*es/max(ep_,1):.0f}%)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    if not args.smoke:
        # Full preload only for real runs (22.8k files, minutes on the
        # network FS). Smoke uses 96 items — load them lazily below.
        preload_raw(root)
        # GPU mel (shared by base-feature precompute + noise bank): the 3.0 s
        # mel is ~8x the 1.0 s work (~354 ms/item on CPU -> ~135 min for
        # 22.8k; ~2-5 ms/item on the A100 -> ~2 min).
        from torchaudio.transforms import MelSpectrogram
        gpu_mel = MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=320,
                                 n_mels=N_MELS, f_min=50.0, f_max=8000.0).to(dev)
        precompute_base_features(train_ds, dev, gpu_mel)
        precompute_base_features(eval_ds, dev, gpu_mel)
    if args.real_noise and not args.smoke:
        load_noise_bank(Path(args.real_noise), mel=gpu_mel)
        if not NOISE_BANK:
            print(f"--real-noise: no .npy in {args.real_noise} "
                  f"-> white noise fallback")

    if args.smoke:
        # lazy base features for the 96 smoke items (CPU, ~35 ms each)
        t0 = time.time()
        for p, _, _, _ in train_ds.items:
            train_ds._base_feat(p)
        for p, _, _, _ in eval_ds.items:
            eval_ds._base_feat(p)
        print(f"base features: {len(BASE_FEAT_CACHE)} cached (CPU, smoke) in "
              f"{time.time() - t0:.0f}s", flush=True)
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    va = DataLoader(eval_ds, batch_size=args.batch, num_workers=0)

    m = VCMTwoHead(len(CLASSES)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.class_weights:
        counts = torch.zeros(len(CLASSES))
        for it in train_ds.items:
            counts[CLASSES.index(it[1])] += 1
        n = len(train_ds)
        w = n / (len(CLASSES) * counts)
        w = (w / w.mean()).to(dev)
        print("class weights:", {CLASSES[i]: round(float(w[i]), 3)
                                 for i in range(len(CLASSES))})
        cmd_lossf = nn.CrossEntropyLoss(weight=w)
    else:
        cmd_lossf = nn.CrossEntropyLoss()
    slot_lossf = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_cmd = 0.0
    for ep in range(1, args.epochs + 1):
        m.train()
        t0 = time.time()
        tot = cnt = 0
        slot_tot = slot_cnt = 0
        for x, y_cmd, y_slot in tr:
            x, y_cmd, y_slot = x.to(dev), y_cmd.to(dev), y_slot.to(dev)
            opt.zero_grad()
            cmd_logits, slot_feat = m(x)
            loss = cmd_lossf(cmd_logits, y_cmd)
            # slot loss: per parametric class, over clips that have a slot
            for cls in PARAMETRIC:
                ci = CLASSES.index(cls)
                mask_cls = (y_cmd == ci)
                mask = mask_cls & (y_slot >= 0)
                if mask.any():
                    sl = m.slot_logits(slot_feat[mask], cls)
                    loss = loss + args.slot_w * slot_lossf(sl, y_slot[mask])
                    slot_tot += 1
                    slot_cnt += int(mask.sum().item())
            loss.backward()
            opt.step()
            tot += loss.item() * len(y_cmd)
            cnt += len(y_cmd)
        sched.step()
        # eval: command acc + slot acc
        m.eval()
        cmd_correct = cmd_total = 0
        slot_correct = slot_total = 0
        with torch.no_grad():
            for x, y_cmd, y_slot in va:
                x, y_cmd, y_slot = x.to(dev), y_cmd.to(dev), y_slot.to(dev)
                cmd_logits, slot_feat = m(x)
                pred = cmd_logits.argmax(1)
                cmd_correct += (pred == y_cmd).sum().item()
                cmd_total += len(y_cmd)
                for cls in PARAMETRIC:
                    ci = CLASSES.index(cls)
                    mask = (y_cmd == ci) & (y_slot >= 0)
                    if mask.any():
                        sl = m.slot_logits(slot_feat[mask], cls)
                        sp = sl.argmax(1)
                        slot_correct += (sp == y_slot[mask]).sum().item()
                        slot_total += int(mask.sum().item())
        cmd_acc = cmd_correct / max(cmd_total, 1)
        slot_acc = slot_correct / max(slot_total, 1)
        history.append({"epoch": ep, "train_loss": tot / max(cnt, 1),
                        "eval_cmd_acc": cmd_acc, "eval_slot_acc": slot_acc,
                        "sec": time.time() - t0})
        tag = ""
        if cmd_acc > best_cmd:
            best_cmd = cmd_acc
            torch.save(m.state_dict(), out / "vcm_a2_best.pt")
            tag = "  *best*"
        print(f"ep {ep:2d}  loss={tot/max(cnt,1):.4f}  "
              f"cmd={cmd_acc:.3f}  slot={slot_acc:.3f}  "
              f"({time.time()-t0:.1f}s){tag}", flush=True)

    torch.save(m.state_dict(), out / "vcm_a2.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved -> {out/'vcm_a2.pt'}  (best cmd acc {best_cmd:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
