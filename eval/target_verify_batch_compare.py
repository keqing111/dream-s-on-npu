"""批量对比 CPU/NPU target 验证。用法: python3 target_verify_batch_compare.py
"""
import pickle, torch

cpu = pickle.load(open("/home/y50063564/DREAM-S/output/target_verify_batch_verify_cpu.pkl","rb"))
npu = pickle.load(open("/home/y50063564/DREAM-S/output/target_verify_batch_verify_npu.pkl","rb"))
qids = sorted(cpu.keys())
print(f"共 {len(qids)} 样本\n")

total_pos = 0; agree_pos = 0; flip_qids = []
accept_diff = 0; bc_diff = 0
all_margins = []
for q in qids:
    la_c = cpu[q]["argmax"]; la_n = npu[q]["argmax"]
    n_pos = la_c.numel()
    total_pos += n_pos
    agree_pos += (la_c == la_n).sum().item()
    # margin: 所有位置的 min margin (CPU)
    m = (cpu[q]["top2"][...,0] - cpu[q]["top2"][...,1])
    all_margins.append(m.min().item())
    if (la_c != la_n).any():
        flip_qids.append(q)
    if cpu[q]["accept_length"] != npu[q]["accept_length"]:
        accept_diff += 1
    if cpu[q]["best_candidate"] != npu[q]["best_candidate"]:
        bc_diff += 1

print(f"argmax agreement: {agree_pos}/{total_pos} = {agree_pos/total_pos*100:.3f}%")
print(f"argmax 翻转的样本: {len(flip_qids)} 个: {flip_qids}")
print(f"accept_length 不同的样本: {accept_diff} 个")
print(f"best_candidate 不同的样本: {bc_diff} 个")

print(f"\n=== 逐样本 ===")
print(f"{'qid':>5} {'CPU_accept':>10} {'NPU_accept':>10} {'bestC同?':>8} {'min_margin':>10}")
per_q_min = []
for q in qids:
    a_c = cpu[q]["accept_length"]; a_n = npu[q]["accept_length"]
    bc = "✓" if cpu[q]["best_candidate"]==npu[q]["best_candidate"] else "✗"
    m = (cpu[q]["top2"][...,0] - cpu[q]["top2"][...,1]).min().item()
    per_q_min.append(m)
    print(f"{q:>5} {a_c:>10} {a_n:>10} {bc:>8} {m:>10.4f}")

print(f"\n=== 最小 margin 分布 ===")
import numpy as np
margins_sorted = sorted(per_q_min)
print(f"min: {margins_sorted[0]:.4f}  P10: {margins_sorted[1]:.4f}  中位: {np.median(margins_sorted):.4f}")
