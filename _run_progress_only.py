"""Re-run ONLY the progress cell (24) so its stored output shows the final
v1d result (DONE + saved)."""
import json
import time
from pathlib import Path

import jupyter_client

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
km = jupyter_client.KernelManager(kernel_name=nb["metadata"]["kernelspec"]["name"])
km.start_kernel(cwd=str(Path.cwd()))
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=60)
try:
    cell = nb["cells"][24]
    msg_id = kc.execute("".join(cell["source"]))
    out_parts = []
    while True:
        msg = kc.get_iopub_msg(timeout=120)
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mtype = msg["msg_type"]
        if mtype == "stream":
            out_parts.append({"output_type": "stream",
                              "name": msg["content"]["name"],
                              "text": msg["content"]["text"]})
            print(msg["content"]["text"].rstrip(), flush=True)
        elif mtype == "error":
            out_parts.append({"output_type": "error",
                              "ename": msg["content"]["ename"],
                              "evalue": msg["content"]["evalue"],
                              "traceback": msg["content"]["traceback"]})
            print("ERROR:", msg["content"]["evalue"])
        elif mtype == "status" and msg["content"].get("execution_state") == "idle":
            break
    cell["outputs"] = out_parts
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved")
finally:
    km.shutdown_kernel(now=True)
