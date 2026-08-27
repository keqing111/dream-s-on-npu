"""把 output/spec_collect_{N}.json 按"完整树格式"渲染：每步全部候选路径 + pos1 top-k logits。
用法: python3 render_collect.py [N] [样本号] [步范围]
"""
import json, sys, os
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')
from transformers import AutoTokenizer

N = sys.argv[1] if len(sys.argv) > 1 else "10"
sidx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = sys.argv[3] if len(sys.argv) > 3 else "0-1"
D = json.load(open(f"/home/y50063564/DREAM-S/output/spec_collect_{N}.json"))
tok = AutoTokenizer.from_pretrained("/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf", use_fast=False)

def dec(ids):
    ids = [x for x in ids if x >= 0]
    return tok.decode(ids, skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)

s = D[sidx]
print(f"### 样本{sidx}: q={s['question']}")
print(f"### yields={s['yields']} new_tokens={s['new_tokens']} AL={s['al_true']:.3f}")
print(f"### 输出: {s['decoded_output']!r}\n")

if rng == "all":
    sel_steps = s['steps']
else:
    a, b = rng.split("-")
    sel_steps = s['steps'][int(a):int(b)+1]
for si, st in enumerate(sel_steps):
    al = st['accept_length']; bc = st['best_candidate']
    p1 = st['pos1_topk']
    p1text = dec(p1['tokens']) if p1 and p1['tokens'] else "-"
    print("=" * 78)
    print(f"步{si} | accept={al} bc={bc} 叶数={st['num_leaves']} 深={st['max_depth']}")
    print(f"  draft pos1 top-k: tokens={p1['tokens'] if p1 else []} logits={[round(x,3) for x in p1['scores']] if p1 else []}")
    print(f"  pos1 top-k 文本: {p1text!r}")
    print(f"  被采纳路径: {dec(st['accepted_path_tokens'])!r}")
    # 全部候选路径 (去重后的深度1 token 展示 + 叶子)
    paths = st['leaves']
    for li, leaf in enumerate(paths[:0] or []):
        pass
    # 显示去重后的树 token 集（每步完整树存在 json）
    alltoks = sorted(set(x for x in st['draft_tokens_all'] if x >= 0))
    print(f"  整树去重token({len(alltoks)}): {dec(alltoks)!r}")
    # 前几片叶子路径完整打印
    shown = 0
    for li, leaf in enumerate(paths):
        p = [x for x in leaf['path'] if x >= 0]
        if not p: continue
        mark = "   <<< 被采纳" if li == bc else ""
        txt = dec(p)
        print(f"    [{li}] {txt!r}{mark}")
        shown += 1
        if shown >= 8:  # 每步最多展示8条候选（完整在 json）
            print(f"    ... 共 {len(paths)} 条候选路径 (完整见 spec_collect_{N}.json)")
            break
