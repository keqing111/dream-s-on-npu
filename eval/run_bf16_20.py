"""BF16 跑 MathVista 接受长度最低的 20 条 (T=0), 对比 FP16 基线。
用法: NPU_CARD=6 python3 run_bf16_20.py
"""
import os, sys, json, random
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

QIDS = [int(x) for x in open("/home/y50063564/DREAM-S/eval/qids_low20.txt").read().split(",")]

TRACE = []
_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None:
        TRACE.append(int(res[1]))
    return res
EM.evaluate_posterior = _ep_hooked

print("加载模型 (BF16, 全 BF16 混合精度) ...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()
print("加载完成", flush=True)

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)

# FP32 基线 AL
base = {}
for l in open("/home/y50063564/DREAM-S/output/MathVista-greedy-fp32-accepts.jsonl"):
    r = json.loads(l)
    base[r['qid']] = r['al']

out = {}
for qid in QIDS:
    row = ds[qid]
    img = row["decoded_image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    input_len = inputs.input_ids.shape[1]

    TRACE.clear()
    with torch.no_grad():
        for out_ids, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                                                  max_new_tokens=500, max_length=4096,
                                                  is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out_ids.shape[1] - input_len
    yields = len(TRACE)
    al = new_tokens / yields if yields else 0
    out[qid] = {"al_bf16": al, "al_fp32": base[qid], "accepts": list(TRACE), "yields": yields, "new_tokens": new_tokens}
    print(f"  qid={qid} FP32={base[qid]:.2f} BF16={al:.3f} 变化={al-base[qid]:+.3f}", flush=True)

with open("/home/y50063564/DREAM-S/output/low20_bf16.json", "w") as f:
    json.dump(out, f, indent=1)
# 汇总
als_fp32 = [v["al_fp32"] for v in out.values()]
als_bf16 = [v["al_bf16"] for v in out.values()]
print(f"\n=== 20 条汇总 ===")
print(f"FP32 平均: {sum(als_fp32)/len(als_fp32):.3f}")
print(f"BF16 平均: {sum(als_bf16)/len(als_bf16):.3f}")
print(f"提升: {sum(als_bf16)/len(als_bf16) - sum(als_fp32)/len(als_fp32):+.3f}")
print(f"改善样本数: {sum(1 for v in out.values() if v['al_bf16'] > v['al_fp32'])}/20")
