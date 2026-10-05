# One-shot: generate the 37 pre-recorded reply clips (edge-tts Roger) as
# 48k mono s16 WAVs in pi_clips/. Run:  python _gen_clips.py
import asyncio
import wave
from pathlib import Path

import edge_tts
import miniaudio
import numpy as np

VOICE = "en-US-RogerNeural"
OUT = Path(__file__).with_name("pi_clips")
OUT.mkdir(exist_ok=True)

CLIPS = [
    # (name, text)
    ("play_music", "Playing music."),
    ("pause", "Music paused."),
    ("resume", "Resuming music."),
    ("next", "Next track."),
    ("stop", "Music stopped."),
    ("volume_up", "Volume up."),
    ("volume_down", "Volume down."),
    ("light_on", "Lights on."),
    ("light_off", "Lights off."),
    ("brightness_20", "Brightness set to twenty percent."),
    ("brightness_60", "Brightness set to sixty percent."),
    ("brightness_100", "Brightness set to one hundred percent."),
    ("color_red", "Color set to red."),
    ("color_blue", "Color set to blue."),
    ("color_green", "Color set to green."),
    ("temp_18", "Temperature set to eighteen degrees."),
    ("temp_22", "Temperature set to twenty two degrees."),
    ("temp_26", "Temperature set to twenty six degrees."),
    ("timer_10", "Timer set to ten seconds."),
    ("timer_30", "Timer set to thirty seconds."),
    ("timer_60", "Timer set to one minute."),
    ("alarm_6am", "Alarm set to six a m."),
    ("alarm_8am", "Alarm set to eight a m."),
    ("alarm_9pm", "Alarm set to nine p m."),
    ("reminder_drink", "Reminder set. Drink water."),
    ("reminder_study", "Reminder set. Study."),
    ("reminder_exercise", "Reminder set. Exercise."),
    ("call", "Calling mom."),
    ("message", "Sending message."),
    ("oos", "Sorry, didn't catch that."),
    ("time_lead", "The time is"),
    ("weather_lead", "The weather is"),
    ("no_internet", "Sorry, no internet connection."),
    ("stale_lead", "but the weather fifteen minutes ago was"),
]


async def one(name: str, text: str):
    mp3 = OUT / f"{name}.mp3"
    await asyncio.wait_for(edge_tts.Communicate(text, VOICE).save(str(mp3)),
                           timeout=30)
    dec = miniaudio.decode_file(str(mp3))
    frames = np.frombuffer(dec.samples, dtype=np.int16)
    if dec.nchannels > 1 and len(frames):
        frames = frames.reshape(-1, dec.nchannels).mean(axis=1)
    sr = int(dec.sample_rate)
    if sr != 48000 and len(frames):
        n = int(len(frames) * 48000.0 / sr)
        x = np.linspace(0.0, len(frames) - 1, n)
        frames = np.interp(x, np.arange(len(frames)), frames).astype(np.int16)
    # trim leading silence (thr 300)
    nz = np.flatnonzero(np.abs(frames) > 300)
    if len(nz):
        frames = frames[max(0, nz[0] - 480):]
    # 180 ms tail
    frames = np.concatenate([frames, np.zeros(180 * 48, dtype=np.int16)])
    # normalize to -3 dB
    peak = int(frames.max())
    if peak > 0:
        frames = (frames * (int(32767 * 0.707) / peak)).astype(np.int16)
    wav = OUT / f"{name}.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(frames.tobytes())
    mp3.unlink()
    return name, len(frames) / 48000.0


async def main():
    rows = await asyncio.gather(*[one(n, t) for n, t in CLIPS])
    bad = []
    for name, dur in sorted(rows, key=lambda r: -r[1]):
        flag = "" if 0.3 <= dur <= 4.0 else "  <-- CHECK"
        if flag:
            bad.append(name)
        print(f"{name:18s} {dur:5.2f}s{flag}")
    print(f"total: {len(rows)} clips")
    if bad:
        print("NEEDS CHECK:", bad)


asyncio.run(main())
