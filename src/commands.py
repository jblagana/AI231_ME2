"""The 10 VCM classes — single source of truth (spec: ai231_me2_specs.md).

Class list follows the spec's ranked top-10 as-is. "Play music" stays its own
class; media control = pause/stop/next/volume variants. Parametric commands
(dim to X%, timer for X min, alarm, temperature, remind, call) are detected by
class only — slot values are out of scope for a tiny VCM.
"""

CLASSES = [
    "play_music",
    "ask_question",
    "control_lights",
    "dim_lights",
    "set_timer",
    "set_alarm",
    "set_temperature",
    "media_control",
    "set_reminder",
    "make_call",
]

# Phrases per class. Variants are what gets fed to the TTS.
# Keep them short (1-4 words) — that's how people actually talk to devices.
PHRASES = {
    "play_music": [
        "play music", "play some music", "play a song", "put on some music",
        "play my playlist", "play music in the living room",
        # v1f expansion (2026-09-28): 1000+ clips/class (PLAN.md target)
        "put on a song", "start playing music", "play some songs",
        "play my favorite playlist",
    ],
    "ask_question": [
        "what's the weather", "what is the weather", "what time is it",
        "what's the temperature outside", "search for the weather",
        "ask about the weather", "what day is it today",
        # v1f expansion
        "what time is it right now", "what's today's date", "what day is it",
        "what's the forecast for today", "tell me the weather",
        "how's the weather outside",
    ],
    "control_lights": [
        "turn on the lights", "turn off the lights", "lights on", "lights off",
        "turn on the light", "turn off the light", "switch on the lights",
        "switch off the lights",
        # v1f expansion
        "turn the lights on", "turn the lights off", "switch on the light",
        "switch off the light",
    ],
    "dim_lights": [
        "dim the lights", "dim the lights to fifty percent", "make the lights dimmer",
        "lower the lights", "dim the lights to thirty percent", "dim the lights to seventy percent",
        # v1f expansion
        "dim the lights to twenty percent", "dim the lights to eighty percent",
        "make the lights lower", "lower the lights a bit",
    ],
    "set_timer": [
        "set a timer for five minutes", "set a timer for ten minutes",
        "set a timer for one minute", "set a timer for thirty minutes",
        "start a timer for two minutes", "set a timer for fifteen minutes",
        # v1f expansion
        "set a timer for twenty minutes", "start a timer for one minute",
        "set a timer for forty five minutes", "start a timer for five minutes",
    ],
    "set_alarm": [
        "set an alarm for six am", "set an alarm for seven am",
        "set an alarm for five thirty am", "wake me up at six am",
        "set an alarm for eight am", "set an alarm for six pm",
        # v1f expansion
        "wake me up at seven am", "set an alarm for nine am",
        "set an alarm for five am", "wake me up at eight am",
    ],
    "set_temperature": [
        "set the temperature to twenty two degrees", "set the temperature to twenty five degrees",
        "set the thermostat to twenty degrees", "set the temperature to eighteen degrees",
        "set the temperature to twenty eight degrees", "set the thermostat to twenty four degrees",
        # v1f expansion
        "set the temperature to twenty degrees", "set the thermostat to twenty two degrees",
        "set the temperature to twenty six degrees", "set the thermostat to eighteen degrees",
    ],
    "media_control": [
        "pause", "stop", "next song", "skip", "volume up", "volume down",
        "louder", "quieter", "pause the music", "stop the music",
        # v1f expansion
        "mute", "unmute", "turn up the volume", "turn down the volume",
        "previous song",
    ],
    "set_reminder": [
        "remind me to buy groceries", "remind me to call mom", "remind me at five pm",
        "add a reminder to water the plants", "remind me about the meeting",
        "set a reminder for tomorrow",
        # v1f expansion
        "remind me to take out the trash", "remind me to pay the bills",
        "set a reminder for next week", "remind me to drink water",
    ],
    "make_call": [
        "call mom", "call dad", "call my mom", "call my dad",
        "call brother", "call sister", "call my friend", "give mom a call",
        # v1f expansion
        "call my sister", "give dad a call", "call the doctor",
        "call my brother", "phone my mom", "phone my dad",
    ],
}

if __name__ == "__main__":
    total = sum(len(v) for v in PHRASES.values())
    for k in CLASSES:
        print(f"{k:20s} {len(PHRASES[k]):3d} phrases")
    print(f"TOTAL: {total} phrases, {len(CLASSES)} classes")
