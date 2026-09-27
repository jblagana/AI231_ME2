"""Execute the new HPC cells (20+) of vcm_v1_training.ipynb and save outputs.

Plain-json version (nbformat write has a type quirk in this env).
Usage: .venv\\Scripts\\python.exe _run_nb_hpc.py
"""
import json
import time
from pathlib import Path

import jupyter_client

NB = Path("notebooks/vcm_v1_training.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
kernel_name = nb["metadata"]["kernelspec"]["name"]

km = jupyter_client.KernelManager(kernel_name=kernel_name)
km.start_kernel(cwd=str(Path.cwd()))
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=60)
try:
    for idx in range(20, len(nb["cells"])):
        cell = nb["cells"][idx]
        if cell["cell_type"] != "code":
            continue
        print(f"--- executing cell {idx}: {cell['source'][0][:70]}", flush=True)
        msg_id = kc.execute("".join(cell["source"]))
        t0 = time.time()
        out_parts = []
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=600)
            except Exception:
                print(f"cell {idx}: TIMEOUT")
                break
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
                print("ERROR:", msg["content"]["evalue"], flush=True)
            elif mtype == "execute_result":
                out_parts.append({"output_type": "execute_result",
                                  "data": msg["content"]["data"],
                                  "execution_count": msg["content"].get(
                                      "execution_count")})
            elif mtype == "status" and msg["content"].get(
                    "execution_state") == "idle":
                break
        cell["outputs"] = out_parts
        cell["execution_count"] = idx - 19
        print(f"    ({time.time() - t0:.1f}s)", flush=True)
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                      encoding="utf-8")
finally:
    km.shutdown_kernel(now=True)
print("saved", NB)
