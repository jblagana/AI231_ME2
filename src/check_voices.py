import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import edge_tts
from make_dataset import VOICES

async def main():
    vs = [v["ShortName"] for v in await edge_tts.list_voices()]
    missing = [w for w in VOICES if w not in vs]
    print(f"checked {len(VOICES)} voices, {len(vs)} available total")
    if missing:
        print("MISSING:", missing)
        # suggest fixes from same locale
        for m in missing:
            loc = m.split("-")[0] + "-" + m.split("-")[1]
            cands = [v for v in vs if v.startswith(loc) and v not in VOICES][:6]
            print(f"  {m} -> candidates: {cands}")
    else:
        print("ALL 40 VOICES EXIST")

asyncio.run(main())
