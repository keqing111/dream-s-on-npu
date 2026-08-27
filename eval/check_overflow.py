"""检查指定样本是否因 FP16 QK^T 溢出(NaN pos1 logits)导致接受长度低。
用法: QIDS="60,97,619" DATASET=MathVista TEMP=0 NPU_CARD=6 python3 check_overflow.py
T=0 时生成确定性，AL 可直接与全量记录对比。
"""
import os, sys, json, math, random
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

QIDS = [int(x) for x in os.environ.get("QIDS", "60,97").split(",")]
DATASET = os.environ.get("DATASET", "MathVista")
TEMP = float(os.environ.get("TEMP", "0"))

TRACE = []
_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None:
        # 此刻 _draft_pos1 是当前树的 pos1 top-k
        p1 = getattr(model, "ea_layer", None)
        p1v = getattr(p1, "_draft_pos1", None) if p1 else None
        nan = bool(p1v) and any(not math.isfinite(s) for s in (p1v.get("scores") or []))
        TRACE.append({"al": res[1], "nan": nan})
    return res
EM.evaluate_posterior = _ep_hooked

print(f"加载模型 (FP16)...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()
print("加载完成", flush=True)

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)

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
        for out, ts, aas in model.ea_generate(inputs, temperature=TEMP, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out.shape[1] - input_len
    yields = len(TRACE)
    accepts = [t.get("al", 0) for t in TRACE]
    nan_steps = sum(1 for t in TRACE if t.get("nan"))
    acc0 = sum(1 for a in accepts if a == 0)
    al = new_tokens / yields if yields else 0
    verdict = "OVERFLOW(溢出)" if nan_steps > 0 else "正常低接受"
    print(f"qid={qid} AL={al:.3f} yields={yields} new_tokens={new_tokens} | NaN步={nan_steps} accept=0步={acc0} ({acc0/max(1,yields):.0%}) [{verdict}]", flush=True)
    print(f"  q={row['question'][:50]}", flush=True)
