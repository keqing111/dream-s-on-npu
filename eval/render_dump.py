"""把 output/spec_dump_qid{N}.json 转译为可读的完整树（含 token 解码）。
用法: python3 render_dump.py [qid] [步数范围，如 0-3 | all]
"""
import json, sys
import os
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')
from transformers import AutoTokenizer

qid = sys.argv[1] if len(sys.argv) > 1 else "0"
D = json.load(open(f"/home/y50063564/DREAM-S/output/spec_dump_qid{qid}.json"))
tok = AutoTokenizer.from_pretrained("/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf", use_fast=False)

def dec(ids, keep_ids=False):
    ids = [x for x in ids if x >= 0]
    s = tok.decode(ids, skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)
    if keep_ids:
        return f"{ids} -> {s!r}"
    return repr(s)

print(f"### qid={D['question_id']} input_len={D['input_len']} "
      f"new_tokens={D['new_tokens']} total_ids={D['total_ids']} AL={D['al_reported']:.3f}")
print(f"### EA 输出: {D['decoded_output']!r}\n")

rng = sys.argv[2] if len(sys.argv) > 2 else "0-2"
steps = D['steps']
if rng == "all":
    sel = steps
else:
    a, b = rng.split("-")
    sel = steps[int(a):int(b)+1]

for s in sel:
    st = s['step']; al = s['accept_length']; bc = s['best_candidate']
    nleaf = s['num_leaves']; dep = s['max_depth']
    acc = s['accepted_path_tokens']
    print("=" * 78)
    print(f"步{st}: accept_length={al}  best_candidate=叶子{bc}  叶子数={nleaf}  树深={dep}")
    # 被采纳路径
    print(f"被采纳 token: {dec(acc, keep_ids=True)}")
    # 完整候选路径（所有叶子）
    print(f"整棵树 {len(s['candidate_paths'])} 条候选路径:")
    for li, path in enumerate(s['candidate_paths']):
        mark = "   <<< 被采纳" if li == bc else ""
        print(f"  [{li}] {dec(path)}{mark}")
    print(f"树中去重 token 集: {dec(s['tree_token_set'])}")
