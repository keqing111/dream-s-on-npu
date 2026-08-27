import json
from collections import defaultdict

for fname in ["semantic_capture_shuffled.jsonl", "semantic_capture.jsonl"]:
    path = f"/home/y50063564/DREAM-S/eval/data/{fname}"
    try:
        samples = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        print(f"{fname}: 不存在")
        continue
    accept_sum = 0
    num_steps = 0
    accepted_at_pos = defaultdict(int)
    for s in samples:
        for st in s["steps"]:
            L = st.get("accept_length", 0)
            accept_sum += L
            num_steps += 1
            for pos in range(L):
                accepted_at_pos[pos] += 1
    total = sum(accepted_at_pos.values())
    mean_accept = accept_sum / num_steps
    print(f"=== {fname} ===")
    print(f"num_steps={num_steps}, accept_length_sum={accept_sum}")
    print(f"验证: sum(accepted_at_pos)={total} == accept_length_sum={accept_sum}? {total==accept_sum}")
    print(f"每步 accept_length 均值 = {mean_accept:.4f} | 对应 reported AL (mean+1) = {mean_accept+1:.4f}")
    for pos in sorted(accepted_at_pos.keys())[:12]:
        print(f"  位置{pos}: {accepted_at_pos[pos]}/{num_steps} = {accepted_at_pos[pos]/num_steps:.4f}")
    print()
