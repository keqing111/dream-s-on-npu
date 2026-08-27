"""从 output/spec_collect_{N}.json 统计各位置命中率（边际命中率，非生存率）。
口径:
 A. 叶子级: 每步每深度 d, 所有叶子的 draft token 是否 == target argmax（位置 d-1）
 B. 节点级: 每步每深度 d, 每个树节点(去重)的 draft token 是否 == target argmax
 C. pos1 top-1: draft pos1 top-1 token 是否 == target 在位置0的 argmax
"""
import json, sys
from collections import defaultdict

N = sys.argv[1] if len(sys.argv) > 1 else "10"
D = json.load(open(f"/home/y50063564/DREAM-S/output/spec_collect_{N}.json"))
print(f"=== {len(D)} 样本 逐位置命中率 ===")

# 每样本: 报告 AL + 步数
for s in D:
    print(f"样本{s['sample']}: q={s['question'][:35]} yields={s['yields']} new_tokens={s['new_tokens']} AL={s['al_true']:.3f}")

# ---- 统计 ----
# A. 叶子级: hit[d-1] += (leaf.path[d] == leaf.tgt_argmax[d-1])
leaf_hit = defaultdict(int); leaf_cnt = defaultdict(int)
# B. 节点级
node_hit = defaultdict(int); node_cnt = defaultdict(int)
# C. pos1 top-1
p1_hit = 0; p1_cnt = 0

for s in D:
    for st in s['steps']:
        lvl = st['draft_level_top1']
        pos1 = st['pos1_topk']
        # C: pos1 top-1 token vs target argmax at 位置0 (所有叶子共享 pos0 上下文)
        if st['leaves'] and pos1['tokens']:
            tgt0 = st['leaves'][0]['tgt_argmax'][0]
            if pos1['tokens'][0] == tgt0:
                p1_hit += 1
            p1_cnt += 1
        # 按深度聚合
        dep = st['max_depth']
        for d in range(1, dep + 1):   # d = 树深度(1-based), target 位置 = d-1
            # A. 叶子级
            for leaf in st['leaves']:
                if d < len(leaf['path']) and d-1 < len(leaf['tgt_argmax']):
                    leaf_cnt[d] += 1
                    if leaf['path'][d] == leaf['tgt_argmax'][d-1] and leaf['path'][d] >= 0:
                        leaf_hit[d] += 1
            # B. 节点级: 用该深度的去重 (draft token, tgt argmax) 对
            pairs = set()
            for leaf in st['leaves']:
                if d < len(leaf['path']) and d-1 < len(leaf['tgt_argmax']) and leaf['path'][d] >= 0:
                    pairs.add((leaf['path'][d], leaf['tgt_argmax'][d-1]))
            node_cnt[d] += 1
            if any(x[0] == x[1] for x in pairs):
                node_hit[d] += 1

print(f"\n=== 口径C: pos1 top-1 命中率 = {p1_hit}/{p1_cnt} = {p1_hit/p1_cnt:.3f} ===")
print(f"\n=== 口径A(叶子级) / 口径B(节点级, 该深度任一 proposal 命中) ===")
print(f"{'位置':>6} {'A_hit/cnt':>12} {'A_rate':>8} {'B_hit/cnt':>12} {'B_rate':>8}")
for d in sorted(set(leaf_cnt) | set(node_cnt)):
    a = f"{leaf_hit[d]}/{leaf_cnt[d]}" if leaf_cnt[d] else "-"
    ar = f"{leaf_hit[d]/leaf_cnt[d]:.3f}" if leaf_cnt[d] else "-"
    b = f"{node_hit[d]}/{node_cnt[d]}" if node_cnt[d] else "-"
    br = f"{node_hit[d]/node_cnt[d]:.3f}" if node_cnt[d] else "-"
    print(f"{d:>6} {a:>12} {ar:>8} {b:>12} {br:>8}")

# 附带: 每个样本每个位置 top-1 命中（draft_level_top1 token vs target argmax）
print(f"\n=== 口径D: draft_level_top1 vs target argmax (沿草稿最优路径) ===")
D_hit = defaultdict(int); D_cnt = defaultdict(int)
for s in D:
    for st in s['steps']:
        for lt in st['draft_level_top1']:
            d = lt['depth']
            # target argmax at 该深度: 用叶子中 draft token == lt token 的叶子
            D_cnt[d] += 1
            hit = False
            for leaf in st['leaves']:
                if d < len(leaf['path']) and leaf['path'][d] == lt['token'] and d-1 < len(leaf['tgt_argmax']):
                    hit = (lt['token'] == leaf['tgt_argmax'][d-1])
                    break
            if hit:
                D_hit[d] += 1
for d in sorted(D_cnt):
    print(f"pos{d}: {D_hit[d]}/{D_cnt[d]} = {D_hit[d]/D_cnt[d]:.3f}")
