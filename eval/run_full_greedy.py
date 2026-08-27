"""全量 T=0 greedy + FP32 attention 运行，保存每步接受长度。
用法: DATASET=MathVista N=1000 NPU_CARD=6 python3 run_full_greedy.py
      DATASET=ChartQA  N=2500 NPU_CARD=7 python3 run_full_greedy.py
"""
import os, sys, json, random, math, statistics
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
os.environ['FP32_ATTN'] = '1'   # 防 FP16 溢出
from dream_s.model.ea_model import EaModel
import dream_s.model.ea_model as EM
from datasets import load_dataset
from PIL import Image
import io

DATASET = os.environ.get("DATASET", "MathVista")
N = int(os.environ.get("N", "1000"))
TEMP = float(os.environ.get("TEMP", "0.0"))

# 每步接受长度捕获
TRACE = []
_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None:
        TRACE.append(int(res[1]))
    return res
EM.evaluate_posterior = _ep_hooked

print(f"加载模型 (FP16权重 + FP32 attention) ...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()
print(f"加载完成", flush=True)

if DATASET == "MathVista":
    ds = load_dataset("parquet",
        data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
        split="test").shuffle(seed=42).select(range(N))
elif DATASET == "ChartQA":
    ds = load_dataset("parquet",
        data_files={'test': '/home/y50063564/DREAM-S/eval/data/chartqa_test.parquet'},
        split="test").filter(lambda r: r["image"] is not None).shuffle(seed=42).select(range(N))
else:
    raise ValueError(DATASET)

out_file = f"/home/y50063564/DREAM-S/output/{DATASET}-temp{TEMP}-fp32-accepts.jsonl"
f_out = open(out_file, "w")

for i in range(N):
    if DATASET == "MathVista":
        img = ds[i]["decoded_image"]
        q = ds[i]["question"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    else:  # ChartQA
        img = ds[i]["image"]
        q = ds[i]["question"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    input_len = inputs.input_ids.shape[1]

    TRACE.clear()
    with torch.no_grad():
        for out, ts, aas in model.ea_generate(inputs, temperature=TEMP, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out.shape[1] - input_len
    yields = len(TRACE)
    al = new_tokens / yields if yields else 0
    rec = {
        "qid": i, "question": q, "al": round(al, 3),
        "yields": yields, "new_tokens": new_tokens,
        "accepts": list(TRACE),
        "output": model.tokenizer.decode(out[0, input_len:].tolist(), skip_special_tokens=True),
    }
    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{N}", flush=True)

f_out.close()
print(f"=== 保存 {out_file} ===", flush=True)

# ==== 统计 ====
records = [json.loads(l) for l in open(out_file) if l.strip()]
als = [r["al"] for r in records]
print(f"\n=== {DATASET} 全量统计 ({len(records)} 样本, T=0, FP32) ===")
print(f"平均接受长度: {sum(als)/len(als):.3f}")
from collections import Counter
n_steps = 0; acc_pos = Counter()
for r in records:
    for a in r["accepts"]:
        n_steps += 1
        for k in range(1, 9):
            if a >= k:
                acc_pos[k] += 1
print(f"总步数: {n_steps}")
for k in range(1, 9):
    print(f"pos{k} 命中率 P(accept>={k}): {acc_pos[k]}/{n_steps} = {acc_pos[k]/n_steps:.3f}")
