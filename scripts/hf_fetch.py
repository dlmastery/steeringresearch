"""Proxy-proof Hugging Face model downloader (urllib + truststore + token).

WHY THIS EXISTS: on this machine an SSL-intercepting proxy breaks the default
huggingface_hub / Xet download transport ("couldn't connect"), but Python's
urllib WITH truststore (OS trust store) reaches HF fine — verified: it pulls
non-gated repos and returns a proper 403 on gated ones (so the transport works;
gating is the only remaining gate). This utility fetches every file of a repo
into models/<repo_id>/ via that working path, so the model can then be loaded
with a LOCAL PATH (no network):

    from steering.model import load_model
    m, t = load_model("models/google/gemma-3-270m-it", quant="none")

USAGE:
    python scripts/hf_fetch.py google/gemma-3-270m-it
    python scripts/hf_fetch.py google/gemma-3-1b-it

PREREQUISITE for gated models (Gemma): accept the license once at the model page
while logged in as the account whose token is cached, e.g.
https://huggingface.co/google/gemma-3-270m-it  → "Acknowledge license".
Otherwise every file 403s.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    print("WARNING: truststore not available; SSL may fail behind a proxy. "
          "pip install truststore")

from huggingface_hub import get_token  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _get(url: str, token: str | None, binary: bool = True):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=60).read()


def fetch(repo_id: str) -> Path:
    token = get_token()
    out = ROOT / "models" / repo_id
    out.mkdir(parents=True, exist_ok=True)

    # 1. list files via the API (works at 200 even behind the proxy).
    api = f"https://huggingface.co/api/models/{repo_id}"
    meta = json.loads(_get(api, token).decode("utf-8"))
    files = [s["rfilename"] for s in meta.get("siblings", [])]
    # skip giant/irrelevant artifacts
    skip = (".gguf", ".pth", ".onnx", ".tflite", ".h5", ".msgpack", ".ot")
    files = [f for f in files if not f.endswith(skip)]
    print(f"{repo_id}: {len(files)} files to fetch -> {out}")

    for f in files:
        dest = out / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip (have) {f}")
            continue
        url = f"https://huggingface.co/{repo_id}/resolve/main/{f}"
        try:
            data = _get(url, token)
            dest.write_bytes(data)
            print(f"  ok  {f}  ({len(data)} bytes)")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"  403 FORBIDDEN {f} — accept the license at "
                      f"https://huggingface.co/{repo_id} (gated). Aborting.")
                raise SystemExit(2)
            print(f"  HTTP {e.code} on {f}: {e}")
            raise
    print(f"DONE -> load with load_model('models/{repo_id}', quant='none')")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for repo in sys.argv[1:]:
        fetch(repo)
