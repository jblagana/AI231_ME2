"""A2 slot taxonomy — maps (class, phrase) to slot value or None.

The slot head only fires for parametric classes. Non-parametric classes
(ask_time, ask_weather, control_lights, media_control, play_music) have no
slot head at all.

Slot vocabulary (43 values across 6 parametric classes):
  dim_lights:      5 values (percentages)
  set_timer:       8 values (durations)
  set_alarm:       7 values (times)
  set_temperature: 7 values (degrees)
  set_reminder:   10 values (actions/times)
  make_call:       6 values (contacts)
"""
import re

# Classes that have a slot head.
PARAMETRIC = [
    "dim_lights",
    "set_timer",
    "set_alarm",
    "set_temperature",
    "set_reminder",
    "make_call",
]

# Slot vocabulary per parametric class (ordered — index = label).
SLOT_VOCAB = {
    "dim_lights": [
        "twenty", "thirty", "fifty", "seventy", "eighty",
    ],
    "set_timer": [
        "one", "two", "five", "ten", "fifteen", "twenty", "thirty", "forty five",
    ],
    "set_alarm": [
        "five am", "five thirty am", "six am", "seven am", "eight am", "nine am", "six pm",
    ],
    "set_temperature": [
        "eighteen", "twenty", "twenty two", "twenty four", "twenty five", "twenty six", "twenty eight",
    ],
    "set_reminder": [
        "buy groceries", "call mom", "drink water", "pay the bills",
        "take out the trash", "water the plants", "the meeting",
        "five pm", "tomorrow", "next week",
    ],
    "make_call": [
        "mom", "dad", "brother", "sister", "friend", "the doctor",
    ],
}

# Number of slot classes per parametric class.
SLOT_COUNTS = {k: len(v) for k, v in SLOT_VOCAB.items()}

# Total unique slot values.
TOTAL_SLOT_VALUES = sum(SLOT_COUNTS.values())  # 43


def extract_slot(cls: str, phrase: str):
    """Return (slot_label, slot_index) or (None, -1) if no slot.

    slot_label: human-readable string (for logging)
    slot_index: index into SLOT_VOCAB[cls], or -1 if no slot
    """
    if cls not in PARAMETRIC:
        return None, -1

    p = phrase.lower().strip()
    vocab = SLOT_VOCAB[cls]

    if cls == "dim_lights":
        # "dim the lights to X percent"
        m = re.search(r"to\s+(.+?)\s+percent", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        return None, -1

    elif cls == "set_timer":
        # "set a timer for X minutes" / "start a timer for X minutes"
        m = re.search(r"for\s+(.+?)\s+minutes?", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        return None, -1

    elif cls == "set_alarm":
        # "set an alarm for X" / "wake me up at X"
        m = re.search(r"(?:for|at)\s+([a-z]+(?:\s+[a-z]+)*)\s*(?:am|pm)?", p)
        if m:
            raw = m.group(1).strip()
            # Reconstruct the full time expression
            m2 = re.search(r"(?:for|at)\s+(.+)$", p)
            if m2:
                val = m2.group(1).strip()
                if val in vocab:
                    return val, vocab.index(val)
        return None, -1

    elif cls == "set_temperature":
        # "set the temperature to X degrees" / "set the thermostat to X degrees"
        m = re.search(r"to\s+(.+?)\s+degrees?", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        return None, -1

    elif cls == "set_reminder":
        # Various patterns:
        # "remind me to X" / "add a reminder to X" → action
        # "remind me about X" → object
        # "remind me at X" → time
        # "set a reminder for X" → time/object
        m = re.search(r"(?:remind me to|add a reminder to)\s+(.+)$", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        m = re.search(r"remind me about\s+(.+)$", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        m = re.search(r"remind me at\s+(.+)$", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        m = re.search(r"set a reminder for\s+(.+)$", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        return None, -1

    elif cls == "make_call":
        # "call X" / "call my X" / "give X a call" / "phone my X"
        m = re.search(r"(?:call|phone)\s+(?:my\s+)?(.+?)(?:\s+a\s+call)?$", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        m = re.search(r"give\s+(.+?)\s+a\s+call$", p)
        if m:
            val = m.group(1).strip()
            if val in vocab:
                return val, vocab.index(val)
        return None, -1

    return None, -1


def self_test():
    """Verify extraction against all known phrases from the manifest."""
    test_cases = [
        # dim_lights
        ("dim_lights", "dim the lights", None),
        ("dim_lights", "dim the lights to fifty percent", "fifty"),
        ("dim_lights", "dim the lights to eighty percent", "eighty"),
        ("dim_lights", "dim the lights to thirty percent", "thirty"),
        ("dim_lights", "dim the lights to seventy percent", "seventy"),
        ("dim_lights", "dim the lights to twenty percent", "twenty"),
        ("dim_lights", "lower the lights", None),
        ("dim_lights", "lower the lights a bit", None),
        ("dim_lights", "make the lights dimmer", None),
        ("dim_lights", "make the lights lower", None),
        # set_timer
        ("set_timer", "set a timer for five minutes", "five"),
        ("set_timer", "set a timer for ten minutes", "ten"),
        ("set_timer", "set a timer for one minute", "one"),
        ("set_timer", "set a timer for thirty minutes", "thirty"),
        ("set_timer", "start a timer for two minutes", "two"),
        ("set_timer", "set a timer for fifteen minutes", "fifteen"),
        ("set_timer", "set a timer for twenty minutes", "twenty"),
        ("set_timer", "set a timer for forty five minutes", "forty five"),
        # set_alarm
        ("set_alarm", "set an alarm for six am", "six am"),
        ("set_alarm", "set an alarm for seven am", "seven am"),
        ("set_alarm", "set an alarm for five thirty am", "five thirty am"),
        ("set_alarm", "wake me up at six am", "six am"),
        ("set_alarm", "set an alarm for eight am", "eight am"),
        ("set_alarm", "set an alarm for six pm", "six pm"),
        ("set_alarm", "set an alarm for five am", "five am"),
        ("set_alarm", "set an alarm for nine am", "nine am"),
        ("set_alarm", "wake me up at seven am", "seven am"),
        ("set_alarm", "wake me up at eight am", "eight am"),
        # set_temperature
        ("set_temperature", "set the temperature to twenty two degrees", "twenty two"),
        ("set_temperature", "set the temperature to twenty five degrees", "twenty five"),
        ("set_temperature", "set the thermostat to twenty degrees", "twenty"),
        ("set_temperature", "set the temperature to eighteen degrees", "eighteen"),
        ("set_temperature", "set the temperature to twenty eight degrees", "twenty eight"),
        ("set_temperature", "set the thermostat to twenty four degrees", "twenty four"),
        ("set_temperature", "set the thermostat to eighteen degrees", "eighteen"),
        ("set_temperature", "set the temperature to twenty six degrees", "twenty six"),
        # set_reminder
        ("set_reminder", "remind me to buy groceries", "buy groceries"),
        ("set_reminder", "remind me to call mom", "call mom"),
        ("set_reminder", "remind me at five pm", "five pm"),
        ("set_reminder", "add a reminder to water the plants", "water the plants"),
        ("set_reminder", "remind me about the meeting", "the meeting"),
        ("set_reminder", "set a reminder for tomorrow", "tomorrow"),
        ("set_reminder", "remind me to drink water", "drink water"),
        ("set_reminder", "remind me to pay the bills", "pay the bills"),
        ("set_reminder", "remind me to take out the trash", "take out the trash"),
        ("set_reminder", "set a reminder for next week", "next week"),
        # make_call
        ("make_call", "call mom", "mom"),
        ("make_call", "call dad", "dad"),
        ("make_call", "call my mom", "mom"),
        ("make_call", "call my dad", "dad"),
        ("make_call", "call brother", "brother"),
        ("make_call", "call sister", "sister"),
        ("make_call", "call my friend", "friend"),
        ("make_call", "call the doctor", "the doctor"),
        ("make_call", "give mom a call", "mom"),
        ("make_call", "give dad a call", "dad"),
        ("make_call", "phone my dad", "dad"),
        ("make_call", "phone my mom", "mom"),
        # non-parametric
        ("play_music", "play music", None),
        ("media_control", "pause", None),
        ("control_lights", "turn on the lights", None),
        ("ask_weather", "what's the weather", None),
        ("ask_time", "what time is it", None),
    ]

    passed = failed = 0
    for cls, phrase, expected in test_cases:
        label, idx = extract_slot(cls, phrase)
        if expected is None:
            ok = (label is None)
        else:
            ok = (label == expected)
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {cls} / {phrase!r} -> {label!r} (expected {expected!r})")

    print(f"slot self-test: {passed}/{passed + failed} passed")
    if failed:
        print("SLOT EXTRACTION FAILURES — fix before training")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
