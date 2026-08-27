"""对比 cap_{dev}_{dtype}.pkl 逐层偏差。用法: python3 compare_caps.py
比较: cpu-fp32(参考) vs npu-fp32 vs npu-fp16 vs npu-bf16
"""
import pickle, os, math

def load(dev, dt):
    p = f"/home/y50063564/DREAM-S/output/cap_{dev}_{dt}.pkl"
    if not os.path.exists(p):
        print(f"[缺] {p}")
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

CONFIGS = [("cpu","fp32"), ("npu","fp32"), ("npu","fp16"), ("npu","bf16")]
caps = {f"{d}_{t}": load(d,t) for d,t in CONFIGS}

REF = "cpu_fp32"
if caps[REF] is None:
    print("参考 cpu-fp32 缺失，改用第一个存在的")
    REF = next(k for k,v in caps.items() if v is not None)

def cmp(a, b):
    a = a.double(); b = b.double()
    if a.shape != b.shape:
        return None, None, None, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a - b).abs()
    denom = a.abs().max() if a.abs().max() > 0 else 1
    return d.max().item(), d.mean().item(), d.max().item()/denom, None

print(f"参考基准: {REF}")
# 列出所有捕获名
names = sorted(set().union(*[set(v.keys()) for v in caps.values() if v]))
print(f"捕获张量: {names}\n")
# 只对比有代表性的中间张量
ORDER = ["input_hidden","l0_q","l0_k","l0_qkt","l0_attnout","l0_out","l1_out","l2_out","final_hidden","head_out"]
print(f"{'张量':<14}", end="")
for c in CONFIGS:
    print(f"{'  '+c[0][:3]+'-'+c[1]:>14}", end="")
print()
for name in ORDER:
    if name not in names: continue
    print(f"{name:<14}", end="")
    for c in CONFIGS:
        k = f"{c[0]}_{c[1]}"
        v = caps.get(k)
        if v is None or name not in v:
            print(f"{'—':>14}", end="")
            continue
        mx, mn, rel, err = cmp(caps[REF][name], v[name])
        if err: print(f"{'shape差':>14}", end="")
        elif mx is None: print(f"{'—':>14}", end="")
        else: print(f"{mx:>8.2e}/{mn:>5.1e}", end="")
    print()
print("\n(格式: maxabsdiff/meanabsdiff, 相对参考 dev-dtype)")
print("\n=== topk tokens 对比 ===")
for c in CONFIGS:
    k = f"{c[0]}_{c[1]}"
    v = caps.get(k)
    if v and "topk_tokens" in v:
        print(f"  {c[0]}-{c[1]}: {v['topk_tokens'][0].tolist()}")
