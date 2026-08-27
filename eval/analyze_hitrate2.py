"""修正版命中率：pos d 命中率 = P(accept_length >= d)（生存/累积曲线，单调不增）。
并给出 pos1 top-k logits 与 target 期望的对应（覆盖率 / top-1 / target 在 top-k 中的排名）。
"""
import json, sys
from collections import Counter, defaultdict

N = sys.argv[1] if len(sys.argv) > 1 else "10"
D = json.load(open(f"/home/y50063564/DREAM-S/output/spec_collect_{N}.json"))

all_accept = []          # 每步 accept_length
pos1_topk_all = []       # 每步 (tokens, logits)
pos1_tgt = []            # 每步 target 的 pos1 argmax

for s in D:
    for st in s['steps']:
        all_accept.append(st['accept_length'])
        p1 = st['pos1_topk']
        if p1 and p1['tokens']:
            pos1_topk_all.append((p1['tokens'], p1['scores']))
        # target pos1 argmax = 所有叶子共享的 pos0 上下文 argmax
        if st['leaves']:
            pos1_tgt.append(st['leaves'][0]['tgt_argmax'][0])

n_steps = len(all_accept)
print(f"=== 共 {len(D)} 样本, {n_steps} 步 ===")

# 1) 生存/累积命中率: P(accept >= k)
print("\n=== 命中率 P(accept_length >= k) ===")
maxk = max(all_accept) + 1
for k in range(1, maxk + 1):
    cnt = sum(1 for a in all_accept if a >= k)
    print(f"pos{k}: {cnt}/{n_steps} = {cnt/n_steps:.4f}")

# 2) accept_length 分布
print("\n=== accept_length 分布 ===")
dist = Counter(all_accept)
for k in sorted(dist):
    print(f"  accept={k}: {dist[k]} 步 ({dist[k]/n_steps:.3f})")

# 3) pos1 top-k 对应 target 期望
print("\n=== pos1 top-k 与 target 期望的对应 ===")
n1 = len(pos1_topk_all)
hit_top1 = 0; hit_ink = 0; ranks = Counter()
for (toks, scores), tgt in zip(pos1_topk_all, pos1_tgt):
    if tgt in toks:
        hit_ink += 1
        ranks[toks.index(tgt)] += 1
    if toks and toks[0] == tgt:
        hit_top1 += 1
print(f"target 的 pos1 argmax 出现在 draft top-k 中: {hit_ink}/{n1} = {hit_ink/n1:.3f}")
print(f"target == draft top-1: {hit_top1}/{n1} = {hit_top1/n1:.3f}")
print(f"target 在 top-k 中的排名分布: {dict(sorted(ranks.items()))}")

# 4) 与 P(accept>=1) 一致性验证
p_acc1 = sum(1 for a in all_accept if a >= 1) / n_steps
print(f"\n=== 一致性检查 ===")
print(f"P(accept>=1) = {p_acc1:.3f}  vs  pos1 top-k 覆盖率 = {hit_ink/n1:.3f}")
print(f"若相等，则 P(accept>=1) 确实是 pos1 覆盖率（target ∈ draft top-k）")
