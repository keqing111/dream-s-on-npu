"""T=0(确定性)下对比某样本 FP16 vs FP32 attention 的接受情况。
用法: QID=4 NPU_CARD=6 python3 compare_fp16_fp32.py
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

QID = int(os.environ.get("QID", "4"))

# hooks
def make_hooks():
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
    return TRACE, _ep_hooked

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
print(f"### qid={QID} q={row['question'][:60]}")

def run(fp32):
    TRACE, _ = make_hooks()
    if fp32:
        os.environ['FP32_ATTN'] = '1'
    else:
        os.environ.pop('FP32_ATTN', None)
    with torch.no_grad():
        for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out.shape[1] - input_len
    yields = len(TRACE)
    al = new_tokens/yields
    acc = [t['al'] for t in TRACE]
    nan = sum(1 for t in TRACE if t['nan'])
    txt = model.tokenizer.decode(out[0, input_len:].tolist(), skip_special_tokens=True)
    return al, yields, acc, nan, txt

for fp32, label in [(False, "FP16"), (True, "FP32")]:
    al, yields, acc, nan, txt = run(fp32)
    print(f"\n### {label}: AL={al:.3f} yields={yields} new_tokens={al*yields:.0f} NaN步={nan}")
    print(f"  每步accept: {acc}")
    print(f"  输出: {txt[:150]!r}")
