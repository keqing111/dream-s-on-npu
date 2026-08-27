"""对比 CPU/NPU target verification logits。用法: python3 target_verify_compare.py [qid]
输出: 逐位置 argmax 表 + argmax agreement + top1-top2 margin + best_candidate + accept_length
"""
import pickle, sys, torch

QID = sys.argv[1] if len(sys.argv) > 1 else "129"
cpu = pickle.load(open(f"/home/y50063564/DREAM-S/output/target_verify_verify_cpu_q{QID}.pkl","rb"))
npu = pickle.load(open(f"/home/y50063564/DREAM-S/output/target_verify_verify_npu_q{QID}.pkl","rb"))
gen = pickle.load(open(f"/home/y50063564/DREAM-S/output/target_verify_gen_q{QID}.pkl","rb"))  # npu 生成时的验证

tok = None
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf", use_fast=False)
except: pass
def tname(id):
    if tok is None or id < 0: return str(id)
    return f"{id}({tok.decode([id])[:6]})"

print(f"### qid={QID} 固定树验证: CPU-FP32 vs NPU-FP32 ===")
print(f"accept_length: CPU={cpu['accept_length']}  NPU={npu['accept_length']}  gen(NPU初跑)={gen['accept_length']}")
print(f"best_candidate: CPU={cpu['best_candidate']}  NPU={npu['best_candidate']}")

la_c = cpu["argmax"]; la_n = npu["argmax"]
cand = cpu["candidates"]
agree = (la_c == la_n).float().mean().item()
print(f"\nargmax agreement (CPU==NPU): {agree*100:.1f}%")
print(f"形状: argmax {tuple(la_c.shape)} candidates {tuple(cand.shape)}")

# top1-top2 margin (CPU 和 NPU)
m_c = (cpu["top2"][...,0] - cpu["top2"][...,1])
m_n = (npu["top2"][...,0] - npu["top2"][...,1])

print(f"\n{'位置':>4} {'draft':>14} {'CPU argmax':>16} {'NPU argmax':>16} {'same':>5} {'CPU margin':>10} {'NPU margin':>10}")
# 取第一个叶子(或每列)展示前 12 个位置
nleaf = la_c.shape[0]
col = 0
shown = 0
for i in range(la_c.shape[1]):
    if i == 0: continue  # 位置0是bonus/根, 无 draft 验证
    d = cand[0, i].item()
    c = la_c[0, i].item()
    n = la_n[0, i].item()
    same = "✓" if c == n else "✗"
    print(f"{i:>4} {tname(d):>14} {tname(c):>16} {tname(n):>16} {same:>5} {m_c[0,i].item():>10.4f} {m_n[0,i].item():>10.4f}")
    shown += 1
    if shown >= 12: break

# 统计: 有 argmax 翻转的位置数, 以及翻转位置的 margin
flip_pos = (la_c != la_n).any(dim=0).nonzero().flatten().tolist()
print(f"\nargmax 翻转的位置: {flip_pos}")
if flip_pos:
    for i in flip_pos[:10]:
        print(f"  pos{i}: CPU={tname(la_c[0,i].item())} margin={m_c[0,i].item():.4f} | NPU={tname(la_n[0,i].item())} margin={m_n[0,i].item():.4f}")
