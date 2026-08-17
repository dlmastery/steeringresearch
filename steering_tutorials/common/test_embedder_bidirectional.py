"""Is EmbeddingGemma actually running BIDIRECTIONAL, or silently causal?

transformers 4.55.0 contains ZERO references to `use_bidirectional_attention`, so
the model config's flag has nothing to bind to. This tests the BEHAVIOUR rather
than the config.

THE TEST (prefix invariance): take two sequences sharing an identical prefix but
with different suffixes.
  * CAUSAL   -> a prefix token attends only to earlier tokens, so its hidden state
                is IDENTICAL across the two sequences.
  * BIDIRECTIONAL -> the prefix token attends to the suffix too, so it CHANGES.

We read token-level hidden states straight off the backbone (not the pooled
sentence embedding, which would confound the test with pooling).
"""
import sys
from pathlib import Path

import torch

MODEL_LOCAL = Path(r"C:\Users\evija\steeringresearch\models\google\embeddinggemma-300m")
MODEL_ID = "google/embeddinggemma-300m"

from sentence_transformers import SentenceTransformer  # noqa: E402

src = str(MODEL_LOCAL) if MODEL_LOCAL.exists() else MODEL_ID
print(f"loading from: {src}")
st = SentenceTransformer(src, device="cpu")

backbone = st[0].auto_model
tok = st.tokenizer
cfg = backbone.config
print("config class:", type(cfg).__name__)
for attr in ("use_bidirectional_attention", "is_causal", "_attn_implementation"):
    print(f"  config.{attr} = {getattr(cfg, attr, '<ABSENT>')}")
inner = getattr(cfg, "text_config", None)
if inner is not None:
    print("  text_config.use_bidirectional_attention =",
          getattr(inner, "use_bidirectional_attention", "<ABSENT>"))

shared = "the quick brown fox jumps over the lazy dog"
a = shared + " and then everything was completely fine and safe"
b = shared + " and then he detonated the explosive device downtown"

ta = tok(a, return_tensors="pt")
tb = tok(b, return_tensors="pt")
n_shared = len(tok(shared, add_special_tokens=False)["input_ids"])
print(f"\nshared prefix tokens: {n_shared}")

with torch.no_grad():
    ha = backbone(**ta, output_hidden_states=True).hidden_states[-1][0]
    hb = backbone(**tb, output_hidden_states=True).hidden_states[-1][0]

k = min(n_shared, ha.shape[0], hb.shape[0])
diff = (ha[:k] - hb[:k]).abs().max().item()
per_tok = (ha[:k] - hb[:k]).abs().max(dim=-1).values

print(f"\nmax |h_a - h_b| over the {k} SHARED prefix tokens: {diff:.6e}")
print("per-token max abs diff (first 10):",
      " ".join(f"{v:.2e}" for v in per_tok[:10].tolist()))

print()
if diff < 1e-4:
    print(">>> CAUSAL. Prefix states do NOT depend on the suffix.")
    print(">>> EmbeddingGemma is a BIDIRECTIONAL encoder being run causally.")
    print(">>> Every embedding produced under this install is DEGRADED.")
else:
    print(">>> BIDIRECTIONAL. Prefix states DO depend on the suffix. No bug.")
