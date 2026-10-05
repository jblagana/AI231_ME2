# One-shot: multi-wav support for pi/me2_music_worker.py (pulled from the
# live Pi). Run:  python _patch_worker.py  (syntax-checks + writes file)
import ast
from pathlib import Path

P = Path(__file__).with_name("pi") / "me2_music_worker.py"
src = P.read_text(encoding="utf-8-sig")


def rep(old, new, n=1):
    global src
    assert src.count(old) == n, f"anchor not unique ({src.count(old)}): {old[:70]!r}"
    src = src.replace(old, new)


rep('''    if len(argv) < 3:
        print("usage: me2_music_worker.py <wav> <start_sample> [device]",
              file=sys.stderr)
        return 2

    wav, start = argv[1], int(argv[2])
    # argv[3] (device_index) is the PortAudio device index — ignored in
    # the ALSA backend (we hardcode ALSA_DEV). Kept for interface compat.

    if not os.path.isfile(wav):
        print(f"me2_music_worker: no such file: {wav}", file=sys.stderr)
        return 2
''', '''    if len(argv) < 3:
        print("usage: me2_music_worker.py <wav> [<wav> ...] <start_sample> [device]",
              file=sys.stderr)
        return 2

    # Multi-wav (2026-10-05): the TTS reply path plays a pre-recorded
    # clip + a synthesized tail as one worker invocation — argv holds
    # every wav path, then the start_sample (applied to the FIRST wav
    # only; later wavs play in full, back to back, no re-open gap).
    # Disambiguation: wav args are existing files, start_sample is a
    # number — so argv[2] that is a file means "second wav" (the old
    # single-wav form [wav, start, dev] still parses: start is not a file).
    wavs = [argv[1]]
    i = 2
    while i < len(argv) and os.path.isfile(argv[i]):
        wavs.append(argv[i])
        i += 1
    start = int(argv[i])
    # argv[i+1] (device_index) is the PortAudio device index — ignored in
    # the ALSA backend (we hardcode ALSA_DEV). Kept for interface compat.
''')

rep('''    # Load and slice
    try:
        arr, sr = load(wav)
    except Exception as e:
        print(f"me2_music_worker: load failed: {e}", file=sys.stderr)
        return 1
    start = max(0, min(start, len(arr) - 1))

    # Write the slice to a temp wav
    try:
        tmp = write_slice_wav(arr, sr, start)
    except Exception as e:
        print(f"me2_music_worker: write failed: {e}", file=sys.stderr)
        return 1
''', '''    # Load and slice (first wav sliced from start; the rest play in full)
    try:
        arr, sr = load(wavs[0])
        if len(wavs) > 1:
            for w in wavs[1:]:
                extra, sr2 = load(w)
                if sr2 != sr:
                    # resample to the first wav's rate (linear; the TTS
                    # path only ever mixes 48k clips with 48k synth)
                    import numpy as np
                    n = int(len(extra) * sr / float(sr2))
                    x = np.linspace(0.0, len(extra) - 1, n)
                    extra = np.interp(x, np.arange(len(extra)),
                                      extra).astype(np.int16)
                arr = np.concatenate([arr, extra])
    except Exception as e:
        print(f"me2_music_worker: load failed: {e}", file=sys.stderr)
        return 1
    start = max(0, min(start, len(arr) - 1))

    # Write the slice to a temp wav
    try:
        tmp = write_slice_wav(arr, sr, start)
    except Exception as e:
        print(f"me2_music_worker: write failed: {e}", file=sys.stderr)
        return 1
''')

ast.parse(src)
P.write_text(src, encoding="utf-8-sig")
print("worker patched OK:", P, len(src), "bytes")
