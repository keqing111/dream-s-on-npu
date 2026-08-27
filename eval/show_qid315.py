"""重现 qid315 (AL=10), 展示单步 accept=9 的完整树 + target 验证。
用法: NPU_CARD=6 python3 show_qid315.py
"""
import os, sys, math, random
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
os.environ['FP32_ATTN'] = '1'
from dream_s.model.ea_model import EaModel
import dream_s.model.ea_model as EM
from datasets import load_dataset
from PIL import Image
import io

TRACE = []
_orig_tree = EM.tree_decoding
def _tree_hooked(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores=False):
    res = _orig_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores)
    logits, hidden_state, outputs = res
    TRACE.append({"draft": tree_candidates, "ri": retrieve_indices, "logits": logits})
    return res
EM.tree_decoding = _tree_hooked

_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None and TRACE:
        TRACE[-1]["bc"] = int(res[0]); TRACE[-1]["al"] = int(res[1])
    return res
EM.evaluate_posterior = _ep_hooked

print("加载模型...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()
tok = model.tokenizer

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)
row = ds[315]
img = row["decoded_image"]
if not isinstance(img, Image.Image):
    img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
input_len = inputs.input_ids.shape[1]
print(f"### qid=315 q={row['question'][:60]}")
print(f"### input_len={input_len}")

with torch.no_grad():
    for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                                          max_new_tokens=500, max_length=4096,
                                          is_llama3=False, output_attention_scores=True):
        pass

new_tokens = out.shape[1] - input_len
print(f"\n### yields={len(TRACE)} new_tokens={new_tokens} AL={new_tokens/len(TRACE):.3f}")
print(f"### 输出: {model.tokenizer.decode(out[0,input_len:].tolist(), skip_special_tokens=True)!r}")
print(f"### 生成 token 序列: {out[0,input_len:].tolist()}")

# 展示那一步
st = TRACE[0]
dt = st['draft'][0].tolist(); ri = st['ri']; lg = st['logits']
bc = st['bc']; al = st['al']
print(f"\n### 单步 accept_length={al}  best_candidate={bc}  叶数={ri.shape[0]} 树深={ri.shape[1]}")
# 被采纳路径的 token 序列 + target argmax
path = ri[bc].tolist()
print(f"\n=== 被采纳路径 (best_candidate={bc}) ===")
print(f"路径节点: {path}")
draft_seq = [dt[n] if 0<=n<len(dt) else -1 for n in path]
print(f"draft 序列: {draft_seq}")
print(f"draft 解码: {model.tokenizer.decode([x for x in draft_seq if x>=0], skip_special_tokens=True)!r}")
print(f"\n=== 逐位置验证 (draft vs target argmax) ===")
print(f"{'位置':>4} {'draft token':>22} {'target argmax':>22} {'匹配?':>6}")
for i in range(1, ri.shape[1]):  # 位置1开始 (0是bonus)
    d = draft_seq[i]
    t = lg[bc, i-1].argmax().item()
    match = "✓" if d == t else "✗"
    print(f"{i:>4} {str(d)+' '+tok.decode([d]):>22} {str(t)+' '+tok.decode([t]):>22} {match:>6}")
print(f"\naccept_length={al} = 连续匹配到位置 {al}")
