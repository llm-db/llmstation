import torch

ATOL = 5e-3
RTOL = 1e-3

peft = torch.load("weights_peft_fa_step30.pt", map_location="cpu", weights_only=True)
v0 = torch.load("weights_v0_fa_step30.pt", map_location="cpu", weights_only=True)
v0s = torch.load("weights_v0s_fa_step30.pt", map_location="cpu", weights_only=True)
v1 = torch.load("weights_v1_fa_step30.pt", map_location="cpu", weights_only=True)
v2 = torch.load("weights_v2_fa_step30.pt", map_location="cpu", weights_only=True)
v3 = torch.load("weights_v3_fa_step30.pt", map_location="cpu", weights_only=True)

def normalize(state_dict, adapter_name):
    """Replace adapter-specific name with canonical placeholder for comparison."""
    return {k.replace(f".{adapter_name}.", ".ADAPTER."): v for k, v in state_dict.items()}

peft_n = normalize(peft, "default")
v0_n = normalize(v0, "ft")
v0s_n = normalize(v0s, "ft")
v1_n = normalize(v1, "ft")
v2_n = normalize(v2, "ft")
v3_n = normalize(v3, "ft")

print(f"peft keys: {len(peft_n)}, v0 keys: {len(v0_n)}, v0s keys: {len(v0s_n)}, v1 keys: {len(v1_n)}, v2 keys: {len(v2_n)}, v3 keys: {len(v3_n)}")
print(f"threshold: atol={ATOL}, rtol={RTOL}")
print()

# --- Weight comparison ---
pairs = [
    ("peft vs v0",  peft_n, v0_n),
    ("peft vs v0s", peft_n, v0s_n),
    ("peft vs v1",  peft_n, v1_n),
    ("peft vs v2",  peft_n, v2_n),
    ("peft vs v3",  peft_n, v3_n),
    ("v2 vs v3",    v2_n,   v3_n),
]
for label, a_dict, b_dict in pairs:
    diffs = []
    fail_count = 0
    for key in sorted(a_dict.keys()):
        a, b = a_dict[key], b_dict.get(key)
        if b is None:
            continue
        diff = (a - b).abs().max().item()
        diffs.append((key, diff))
        if not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
            fail_count += 1

    max_key, max_diff = max(diffs, key=lambda x: x[1])
    avg_diff = sum(d for _, d in diffs) / len(diffs)
    status = "ALL PASS" if fail_count == 0 else f"{fail_count}/{len(diffs)} FAIL"
    print(f"{label}: max_diff={max_diff:.2e}, avg_diff={avg_diff:.2e} | {status}")
    if fail_count > 0:
        print(f"  worst param: {max_key}")

# --- Weight sum comparison ---
print()
sums = {
    "peft": sum(v.sum().item() for v in peft_n.values()),
    "v0":   sum(v.sum().item() for v in v0_n.values()),
    "v0s":  sum(v.sum().item() for v in v0s_n.values()),
    "v1":   sum(v.sum().item() for v in v1_n.values()),
    "v2":   sum(v.sum().item() for v in v2_n.values()),
    "v3":   sum(v.sum().item() for v in v3_n.values()),
}
for name, s in sums.items():
    print(f"weight sum {name}: {s:.10f}")
