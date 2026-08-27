"""测试不同 TARGET_LAYER 注入对接受长度的影响（T=0 确定性）。
用法: QIDS="0,1,2" LAYERS="-1,-2,-3,-4" NPU_CARD=6 python3 test_layer_inject.py
"""
import os, sys, math, random
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

QIDS = [int(x) for x in os.environ.get("QIDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
LAYERS = [int(x) for x in os.environ.get("LAYERS", "-1,-2,-3,-4").split(",")]

TRACE = []
_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None:
        p1 = getattr(getattr(model, "ea_layer", None), "_draft_pos1", None)
        nan = bool(p1) and any(not math.isfinite(s) for s in (p1.get("scores") or []))
        TRACE.append({"al": int(res[1]), "nan": nan})
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

# 预构建所有样本
samples = []
for qid in QIDS:
    row = ds[qid]
    img = row["decoded_image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    samples.append((qid, row["question"], inputs))

def run(inputs):
    input_len = inputs.input_ids.shape[1]
    TRACE.clear()
    with torch.no_grad():
        for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out.shape[1] - input_len
    yields = len(TRACE)
    al = new_tokens / yields if yields else 0
    nan = sum(1 for t in TRACE if t.get("nan"))
    return al, nan

# 表头
print(f"\n{'qid':>4}", end="")
for L in LAYERS:
    print(f"  {'layer'+str(L):>10}", end="")
print(f"  {'基线(-1)':>10}")
results = {}
for qid, q, inputs in samples:
    row = [qid]
    for L in LAYERS:
        os.environ['TARGET_LAYER'] = str(L)
        al, nan = run(inputs)
        results[(qid, L)] = (al, nan)
        row.append(f"{al:.2f}")
    row.append(f"{results[(qid,-1)][0]:.2f}")
    print("  ".join(str(x) for x in row), flush=True)

# 汇总
print(f"\n=== 平均 AL ===")
for L in LAYERS:
    als = [results[(q, L)][0] for q, _ in samples]
    nans = sum(results[(q, L)][1] for q, _ in samples)
    print(f"  layer{L}: mean={sum(als)/len(als):.3f}  NaN步={nans}")
