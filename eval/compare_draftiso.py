"""对比 draft iso pkl (固定输入, 只变 draft 设备/精度)。
用法: python3 compare_draftiso.py [SOURCE]
"""
import pickle, os, sys, torch

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "npu_fp32"
CONFIGS = [("cpu","fp32"), ("npu","fp32"), ("npu","fp16"), ("npu","bf16")]
def load(ddev, ddt):
    p = f"/home/y50063564/DREAM-S/output/draftiso_{SOURCE}_dev{ddev}_{ddt}.pkl"
    if not os.path.exists(p): return None
    with open(p,"rb") as f: return pickle.load(f)
caps = {f"{d}_{t}": load(d,t) for d,t in CONFIGS}
REF = "npu_fp32"
if caps[REF] is None: REF = next(k for k,v in caps.items() if v)

print(f"SOURCE={SOURCE} 参考={REF}")
names = sorted(set().union(*[set(v.keys()) for v in caps.values() if v]))
ORDER = ["input_hidden","l0_q","l0_k","l0_qkt","l0_attnout","l0_out","l1_out","l2_out","final_hidden","head_out"]
print(f"{'张量':<12}", end="")
for c in CONFIGS: print(f"{'  '+c[0][:3]+'-'+c[1]:>16}", end="")
print()
for name in ORDER:
    if name not in names: continue
    print(f"{name:<12}", end="")
    for c in CONFIGS:
        k = f"{c[0]}_{c[1]}"; v = caps.get(k)
        if v is None or name not in v or v[name] is None:
            print(f"{'—':>16}", end=""); continue
        a = caps[REF][name].double(); b = v[name].double()
        if a.shape != b.shape:
            print(f"{'shape差':>16}", end=""); continue
        d = (a-b).abs()
        mx, mn = d.max().item(), d.mean().item()
        nan = " ⚠️" if torch.isnan(b).any() else ""
        print(f"{mx:>9.1e}/{mn:>6.1e}{nan}", end="")
    print()
print("(格式: maxabs/meanabs, 参考 draft-npu-fp32)")
print("\n=== topk tokens ===")
for c in CONFIGS:
    k=f"{c[0]}_{c[1]}"; v=caps.get(k)
    if v and "topk_tokens" in v:
        print(f"  {c[0]}-{c[1]}: {v['topk_tokens'][0].tolist()}")
