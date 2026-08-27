"""样本4 在 T=1 下重复跑 N 次，量化 temp=1 采样方差；FP16 vs FP32 分布对比。
用法: QID=4 ROUNDS=5 NPU_CARD=6 python3 repeat_t1.py
"""
import os, sys, math, random, statistics
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
from dream_s.model.ea_model import EaModel
import dream_s.model.ea_model as EM
from datasets import load_dataset
from PIL import Image
import io

QID = int(os.environ.get("QID", "4"))
ROUNDS = int(os.environ.get("ROUNDS", "5"))

TRACE = []
_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None:
        TRACE.append(int(res[1]))
    return res
EM.evaluate_posterior = _ep_hooked

print("加载模型...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)
row = ds[QID]
img = row["decoded_image"]
if not isinstance(img, Image.Image):
    img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
input_len = inputs.input_ids.shape[1]
print(f"### qid={QID} q={row['question'][:50]}  ROUNDS={ROUNDS}")

def run(fp32):
    if fp32: os.environ['FP32_ATTN'] = '1'
    else: os.environ.pop('FP32_ATTN', None)
    TRACE.clear()
    with torch.no_grad():
        for out, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out.shape[1] - input_len
    yields = len(TRACE)
    return new_tokens / yields if yields else 0, yields, new_tokens

for fp32, label in [(False, "FP16"), (True, "FP32")]:
    als = []
    print(f"\n=== {label} (T=1, {ROUNDS} 次) ===", flush=True)
    for r in range(ROUNDS):
        al, yields, newt = run(fp32)
        als.append(al)
        print(f"  round{r+1}: AL={al:.3f} (yields={yields}, new_tokens={newt})", flush=True)
    print(f"  {label}: mean={statistics.mean(als):.3f} std={statistics.stdev(als):.3f} "
          f"min={min(als):.3f} max={max(als):.3f} 范围={max(als)-min(als):.3f}", flush=True)
