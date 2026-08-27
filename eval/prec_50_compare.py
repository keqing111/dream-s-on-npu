"""对比 prec50 的 cpu vs npu draft 输出。用法: python3 prec_50_compare.py
"""
import pickle, os, torch
import numpy as np

npu = pickle.load(open("/home/y50063564/DREAM-S/output/prec50_npu.pkl","rb"))
cpu = pickle.load(open("/home/y50063564/DREAM-S/output/prec50_cpu.pkl","rb"))
qids = sorted(npu.keys())
print(f"共 {len(qids)} 样本")

# 1) topk token 一致率
agree_top1 = 0; agree_topk = 0
for q in qids:
    t_n = npu[q]["draft_npu"]["topk_tokens"][0].tolist()
    t_c = cpu[q]["draft_cpu"]["topk_tokens"][0].tolist()
    if t_n[0] == t_c[0]: agree_top1 += 1
    if t_n == t_c: agree_topk += 1
print(f"top1 一致: {agree_top1}/{len(qids)} = {agree_top1/len(qids):.3f}")
print(f"topk 全一致: {agree_topk}/{len(qids)} = {agree_topk/len(qids):.3f}")

# 2) head_out / 各层 的 cpu vs npu 差异 (固定输入)
import collections
diffs = collections.defaultdict(list)
for q in qids:
    a = npu[q]["draft_npu"]; b = cpu[q]["draft_cpu"]
    for k in ["l0_out","l1_out","l2_out","final_hidden","head_out"]:
        if k in a and k in b:
            d = (a[k].double()-b[k].double()).abs()
            diffs[k].append((d.max().item(), d.mean().item()))
print(f"\n=== cpu vs npu draft 逐层差异 (固定NPU输入, max/mean) ===")
print(f"{'张量':<12} {'mean_max':>10} {'mean_mean':>10} {'max_max':>10}")
for k in ["l0_out","l1_out","l2_out","final_hidden","head_out"]:
    if k not in diffs: continue
    mxs = [x[0] for x in diffs[k]]; mns = [x[1] for x in diffs[k]]
    print(f"{k:<12} {np.mean(mxs):>10.2e} {np.mean(mns):>10.2e} {np.max(mxs):>10.2e}")

# 3) top1 token 一致的样本里 head_out 差异 vs 不一致的
h_agree = [diffs['head_out'][i] for i,q in enumerate(qids)
           if npu[q]["draft_npu"]["topk_tokens"][0].tolist()==cpu[q]["draft_cpu"]["topk_tokens"][0].tolist()]
print(f"\nhead_out diff: top1一致样本 mean_max={np.mean([x[0] for x in h_agree]):.2e} (n={len(h_agree)})")

# 4) 有哪些样本 topk 不一致
print(f"\n=== topk 不一致的样本 ===")
for q in qids:
    t_n = npu[q]["draft_npu"]["topk_tokens"][0].tolist()
    t_c = cpu[q]["draft_cpu"]["topk_tokens"][0].tolist()
    if t_n != t_c:
        print(f"  qid={q}: npu={t_n} cpu={t_c}")
