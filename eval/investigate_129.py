"""复现 qid129, 定位 'loud</s>3.Both' 垃圾提案的步, 结合 DEBUG_MASK headout 探针查精度。
用法: NPU_CARD=6 DEBUG_MASK=1 python3 investigate_129.py
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
    TRACE.append({"draft": tree_candidates[0].tolist()})
    return res
EM.tree_decoding = _tree_hooked

_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None and TRACE:
        TRACE[-1]["al"] = int(res[1])
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
row = ds[129]
img = row["decoded_image"]
if not isinstance(img, Image.Image):
    img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
input_len = inputs.input_ids.shape[1]
print(f"### qid=129 q={row['question'][:60]}", flush=True)

with torch.no_grad():
    for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                                          max_new_tokens=500, max_length=4096,
                                          is_llama3=False, output_attention_scores=True):
        pass

print(f"\n### 每步 draft 提案 + accept ===", flush=True)
garbage_steps = []
for si, t in enumerate(TRACE):
    dtt = t['draft']
    # 树去重 token 解码
    seen = []
    for x in dtt:
        if x >= 0 and x not in seen:
            seen.append(x)
        if len(seen) >= 6: break
    text = tok.decode([x for x in seen if x>=0], skip_special_tokens=False)
    al = t.get('al', -1)
    flag = "  <<< GARBAGE?" if ('</s>' in text or al == 0) else ""
    if al == 0 and 'loud' in text:
        garbage_steps.append(si)
    print(f"步{si}: accept={al} 树前几个: {seen} -> {text!r}{flag}", flush=True)
print(f"\n### 疑似垃圾步: {garbage_steps}", flush=True)
