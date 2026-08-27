"""精度对照: 同一 qid129 输入, 不同设备/精度跑 draft 第一步, 分层捕获中间张量。
用法: DEVICE=npu DTYPE=fp16 NPU_CARD=6 python3 prec_run.py
      DEVICE=cpu DTYPE=fp32 python3 prec_run.py
"""
import os, sys, random, pickle
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
DEVICE = os.environ.get("DEVICE", "npu")   # npu / cpu
DTYPE = os.environ.get("DTYPE", "fp16")    # fp32 / fp16 / bf16
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
if DTYPE != "fp32" and os.environ.get("FP32_ATTN", "") == "1":
    os.environ['FP32_ATTN'] = '1'   # 仅当显式设置
from dream_s.model.ea_model import EaModel
import dream_s.model.ea_model as EM
from dream_s.model.cnets import DRAFT_CAP
from datasets import load_dataset
from PIL import Image
import io

torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[DTYPE]
dev_map = "npu:0" if DEVICE == "npu" else "cpu"
print(f"### DEVICE={DEVICE} DTYPE={DTYPE} dtype={torch_dtype} FP32_ATTN={os.environ.get('FP32_ATTN','0')}", flush=True)

model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=dev_map)
model.eval()
print("加载完成", flush=True)

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

DRAFT_CAP.clear()
with torch.no_grad():
    gen = model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                            max_new_tokens=500, max_length=4096,
                            is_llama3=False, output_attention_scores=True)
    out, ts, aas = next(gen)   # 第一步
    gen.close()

new_tokens = out.shape[1] - input_len
print(f"第一步 new_tokens={new_tokens}", flush=True)
# 打印 topk
if "topk_tokens" in DRAFT_CAP:
    tt = DRAFT_CAP["topk_tokens"][0].tolist()
    print(f"pos1 top-k tokens: {tt} -> {model.tokenizer.decode([x for x in tt if x>=0])!r}", flush=True)

out_path = f"/home/y50063564/DREAM-S/output/cap_{DEVICE}_{DTYPE}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(DRAFT_CAP, f)
print(f"=== 保存 {out_path}, 捕获 {len(DRAFT_CAP)} 个张量 ===", flush=True)
for k, v in DRAFT_CAP.items():
    print(f"  {k}: {tuple(v.shape)} min={v.min():.3f} max={v.max():.3f}", flush=True)
