"""Did training on the RIGHT distribution actually fix ||v||?

The v3 fix had two independent halves:
  (root)  train the field on ALL token positions, since FlowContext transports
          every position -- the v2 bank was last-token only.
  (guard) integrate_flow(unit_velocity=True), which forces the displacement to
          T*||h|| by construction regardless of how badly ||v|| is trained.

The guard would hide a still-broken field. So measure the RAW ||v|| on the v3
field and compare it to the v2 field, on the SAME states, with the guard OFF.
||v|| should approach 1.0 (the regression target is a unit vector).
"""
import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\Users\evija\steeringresearch")
sys.path.insert(0, str(ROOT))
from steering_tutorials.flas.flow import VelocityField, integrate_flow  # noqa: E402

ART = ROOT / "steering_tutorials/flas/artifacts"
torch.manual_seed(0)

for label, fn in (("v2 (last-token bank)", "flow_v2_LASTTOKEN_SUPERSEDED.pt"),
                  ("v3 (all-position bank)", "flow.pt")):
    ck = torch.load(ART / fn, map_location="cpu", weights_only=False)
    meta, state = ck["meta"], ck["state_dict"]
    vf = VelocityField(hidden_dim=ck["hidden_dim"], concept_dim=ck["concept_dim"],
                       time_dim=ck["time_dim"], width=ck["width"])
    vf.load_state_dict(state)
    vf.eval()
    c = torch.as_tensor(meta["concept_vectors"]["sexual"]).float().reshape(-1)
    hidden = ck["hidden_dim"]
    h = torch.randn(32, hidden) * (5350.0 / (hidden ** 0.5))

    print(f"\n=== {label}  (bank n={meta.get('n_h0_bank','<v2: last-token only>')}) ===")
    for T in (0.02, 0.05, 0.10, 0.15):
        st = {}
        with torch.no_grad():
            x_guard = integrate_flow(vf, h, c, T=T, n_steps=8, norm_relative=True,
                                     unit_velocity=True, stats=st)
            x_raw = integrate_flow(vf, h, c, T=T, n_steps=8, norm_relative=True,
                                   unit_velocity=False)
        raw_norm = sum(st["raw_velocity_norms"]) / len(st["raw_velocity_norms"])
        fin = torch.isfinite(x_raw).all().item()
        rel_guard = ((x_guard - h).norm(dim=-1) / h.norm(dim=-1)).mean().item()
        rel_raw = (((x_raw - h).norm(dim=-1) / h.norm(dim=-1)).mean().item()
                   if fin else float("nan"))
        print(f"  T={T:<5} raw||v||={raw_norm:9.4f}  "
              f"guard rel/T={rel_guard/T:6.4f}  "
              f"UNGUARDED rel/T={rel_raw/T if fin else float('nan'):12.4f} "
              f"finite={fin}")
