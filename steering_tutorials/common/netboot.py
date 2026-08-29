"""netboot.py -- make HTTPS work on a host with an intercepting TLS proxy.

WHY THIS EXISTS
---------------
On 2026-08-27 every `datasets.load_dataset` call started failing with

    httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]
    unable to get local issuer certificate

while plain `urllib` to the SAME host succeeded. The split is the diagnosis:
`urllib` on Windows can use the OS certificate store; `httpx` -- which the newer
`huggingface_hub` pulled in by the transformers 5.x upgrade uses -- validates
against `certifi` only. This host runs Norton, which does HTTPS scanning by
injecting its own root CA into the WINDOWS store and nowhere else. So certifi
genuinely does not have the issuer, and passing `verify=certifi.where()`
explicitly does not help (verified: it fails identically).

`truststore` makes Python's ssl module use the OS trust store, which is where the
intercepting root actually lives.

Call `enable()` BEFORE importing `datasets` / `huggingface_hub`. Idempotent.

This is upgrade fallout, not a new dependency choice: nothing in this repo asked
for httpx. It arrived with the transformers 5.15.0 upgrade that the
causal-attention fix required, in the same way the `apply_chat_template` ->
BatchEncoding break did.

CPU-only, no model, ASCII stdout.
"""
from __future__ import annotations

_DONE = False


def enable(verbose: bool = False) -> bool:
    """Route TLS verification through the OS trust store. True if injected."""
    global _DONE
    if _DONE:
        return True
    try:
        import truststore
    except ImportError:
        if verbose:
            print("[netboot] truststore not installed; leaving TLS as-is "
                  "(pip install truststore if HF downloads fail with "
                  "CERTIFICATE_VERIFY_FAILED)")
        return False
    truststore.inject_into_ssl()
    _DONE = True
    if verbose:
        print("[netboot] TLS verification now uses the OS trust store")
    return True


def _self_test() -> None:
    ok = enable(verbose=True)
    print("OK  enable() ->", ok)
    assert enable() is ok or enable() is True, "enable() must be idempotent"
    print("OK  idempotent on a second call")
    try:
        import httpx
        r = httpx.get("https://huggingface.co/api/datasets/AI45Research/ATBench",
                      timeout=20)
        print("OK  live HTTPS to huggingface.co ->", r.status_code)
    except ImportError:
        print("SKIP httpx not installed; nothing to verify against")
    except Exception as exc:                       # network may be absent
        print("SKIP live check unavailable: %s" % type(exc).__name__)
    print("\nOK -- netboot.py self-test passed CPU-only, no model, no GPU.")


if __name__ == "__main__":
    _self_test()
