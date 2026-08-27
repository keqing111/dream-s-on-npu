"""BF16 数据并行: N卡 各跑 25 条随机 MathVista 样本 (T=0, draft+target 全 BF16)。
用法: CARD=0 NPU_CARD=0 RATIO=1.0 python3 run_bf16_parallel.py  (CARD=0..7)
RATIO=1.0 → 不裁剪视觉 token; RATIO=0.7 → 原始裁剪
每卡独立输出 output/bf16_r{RATIO}_card{CARD}.json
"""
import os, sys, json, random
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)   # 保证各卡生成相同的随机样本
sys.path.insert(0, '/home/y50063564/DREAM-S')
CARD = int(os.environ.get("CARD", "0"))                       # 0..7, 决定取哪 25 条
NGROUP = int(os.environ.get("NGROUP", "8"))                   # 卡数
RATIO = float(os.environ.get("RATIO", "0.7"))                 # 视觉 token 保留比例, 1.0=不裁剪
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", str(CARD))
from dream_s.model.ea_model import EaModel
import dream_s.model.ea_model as EM
from datasets import load_dataset
from PIL import Image
import io

# ---- NGROUP*25 个随机 qid (seed 42, 每进程确定性相同) ----
_qids = random.sample(range(1000), NGROUP*25)                  # 不重复索引
_group = _qids[CARD*25:(CARD+1)*25]
print(f"### CARD={CARD} RATIO={RATIO} 处理 {len(_group)} 条: {_group}", flush=True)

# ---- 每步 accept_length 捕获 (T=0 用) ----
TRACE = []
_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None:
        TRACE.append(int(res[1]))   # res[1] = accept_length (greedy 返回 (bc, al, p))
    return res
EM.evaluate_posterior = _ep_hooked

# ---- 加载模型: draft+target 全 BF16 (训练 mixed_precision 行为) ----
print("加载模型 (BF16) ...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()
model.token_ratio = RATIO   # 设置视觉 token 保留比例 (1.0 = 不裁剪)
print(f"加载完成, token_ratio={RATIO}", flush=True)

# ---- 数据集 (seed 42 shuffle, 与基线一致) ----
ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)

# ---- FP32 基线 (从全量 FP32 结果读, 便于对比) ----
base = {}
for l in open("/home/y50063564/DREAM-S/output/MathVista-greedy-fp32-accepts.jsonl"):
    r = json.loads(l)
    base[r['qid']] = r['al']

# ---- 跑 25 条 ----
out = {}
for qid in _group:
    row = ds[qid]
    img = row["decoded_image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    input_len = inputs.input_ids.shape[1]

    TRACE.clear()
    with torch.no_grad():
        for out_ids, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                                                  max_new_tokens=500, max_length=4096,
                                                  is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out_ids.shape[1] - input_len          # EA 实际生成 token 数 (含 bonus)
    yields = len(TRACE)                                 # 投机步数
    al = new_tokens / yields if yields else 0           # 接受长度 = new_tokens/yields
    b = base.get(qid)
    out[qid] = {"al_bf16": al, "al_fp32": b, "accepts": list(TRACE), "yields": yields, "new_tokens": new_tokens}
    print(f"  CARD{CARD} qid={qid} FP32={b:.2f} BF16={al:.3f} 变化={al-b:+.3f}", flush=True)

with open(f"/home/y50063564/DREAM-S/output/bf16_r{RATIO}_card{CARD}.json", "w") as f:
    json.dump(out, f, indent=1)

# ---- 汇总 ----
_als_b = [v["al_fp32"] for v in out.values() if v["al_fp32"] is not None]
_als_16 = [v["al_bf16"] for v in out.values()]
print(f"=== CARD{CARD} 汇总: FP32均 {sum(_als_b)/len(_als_b):.3f} BF16均 {sum(_als_16)/len(_als_16):.3f} "
      f"提升 {sum(_als_16)/len(_als_16)-sum(_als_b)/len(_als_b):+.3f} 改善 {sum(1 for v in out.values() if v['al_bf16']>v['al_fp32'])}/{len(out)}", flush=True)
