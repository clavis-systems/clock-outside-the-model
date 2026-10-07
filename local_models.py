"""Local language models on TicToc, in the authors' "with timestamps" setting, through Ollama.

Every message carries its timestamp in square brackets, as in the authors' templates ("[2025-12-07T09:00:20Z]..."),
and the model receives the tool definitions. The model decides by itself; we record whether it calls a tool
(tool_calls in the response, or a <tool_call> block in its text). Differences from the authors' runs: quantized
Ollama models, temperature 0, seed 1, reasoning off, and no timestamp on the response turn itself (it is usually a
few seconds after the last user message, which is timestamped). Only evaluable cases are run.

Usage (Ollama running):
    python local_models.py gemma4:e4b-it-qat data/test.parquet
Writes results/<model>_<split>.jsonl (one line per case; reruns resume) and prints the score.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

from rule import label, load, nar

OLLAMA = "http://127.0.0.1:11434/api/chat"
OPTIONS = {"temperature": 0, "seed": 1, "num_predict": 300}


def messages(history: list[dict]) -> list[dict]:
    out = []
    for m in history:
        stamp = f"[{m['time']}]"
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{"function": {"name": c["function"]["name"],
                                   "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                     for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": stamp, "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": f"{stamp}{m.get('content') or ''}", "tool_name": m.get("name", "")})
        else:
            out.append({"role": m["role"], "content": f"{stamp}{m.get('content') or ''}"})
    return out


def ask(model: str, case: dict) -> dict:
    tools = case["function"] if isinstance(case["function"], list) else json.loads(case["function"])
    body = json.dumps({"model": model, "messages": messages(case["history"]), "tools": tools, "stream": False,
                       "think": False, "options": OPTIONS}).encode()
    request = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as r:
        msg = json.load(r).get("message", {})
    used = bool(msg.get("tool_calls")) or "<tool_call>" in (msg.get("content") or "")
    return {"decision": "tool" if used else "direct", "seconds": round(time.monotonic() - start, 2)}


def main() -> None:
    model, data = sys.argv[1], sys.argv[2]
    cases = [c for c in load(data) if label(c)]
    out = Path("results") / f"{model.replace(':', '_').replace('/', '_')}_{Path(data).stem}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {}
    if out.exists():
        done = {json.loads(r)["id"]: json.loads(r) for r in out.read_text(encoding="utf-8").splitlines() if r}
    with out.open("a", encoding="utf-8") as f:
        for case in cases:
            if case["id"] in done:
                continue
            try:
                row = {"id": case["id"], "label": label(case), **ask(model, case)}
            except Exception as error:                                        # noqa: BLE001
                row = {"id": case["id"], "label": label(case), "decision": "error", "reason": str(error)[:200]}
            done[case["id"]] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
    pairs = [(r["label"], r["decision"]) for r in done.values() if r["decision"] in ("tool", "direct")]
    value, counts = nar(pairs)
    print(json.dumps({"model": model, "data": data, "valid": len(pairs), "errors": len(done) - len(pairs),
                      "normalized_alignment": value, **counts}))


if __name__ == "__main__":
    main()
