"""TARGET_LAYER 测试: 视觉全开(ratio=1.0) + 全BF16 + T=0, 同一批40样本, 测不同 target hidden 注入层。
用法: LAYER=-1 NPU_CARD=0 python3 run_layer_test.py   (LAYER=-1..-5)
"""
import os, sys, json, random
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
LAYER = int(os.environ.get("LAYER", "-1"))
os.environ['TARGET_LAYER'] = str(LAYER)   # ea_model 读取注入层
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "0")
from dream_s.model.ea_model import EaModel
import dream_s.model.ea_model as EM
from datasets import load_dataset
from PIL import Image
import io

_qids = random.sample(range(1000), 100)
print(f"### LAYER={LAYER} 处理 {len(_qids)} 条", flush=True)

TRACE = []
_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None:
        TRACE.append(int(res[1]))
    return res
EM.evaluate_posterior = _ep_hooked

print(f"加载模型 (BF16, TARGET_LAYER={LAYER}) ...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()
model.token_ratio = 1.0   # 视觉全开, 不裁剪
print("加载完成", flush=True)

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)

base = {}
for l in open("/home/y50063564/DREAM-S/output/MathVista-greedy-fp32-accepts.jsonl"):
    r = json.loads(l)
    base[r['qid']] = r['al']

out = {}
for qid in _qids:
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
    new_tokens = out_ids.shape[1] - input_len
    yields = len(TRACE)
    al = new_tokens / yields if yields else 0
    out[qid] = {"al_layer": al, "al_base": base.get(qid), "yields": yields, "new_tokens": new_tokens}
    print(f"  LAYER{LAYER} qid={qid} base={base.get(qid):.2f} layer{LAYER}={al:.3f} 变化={al-base.get(qid,0):+.3f}", flush=True)

with open(f"/home/y50063564/DREAM-S/output/layer_test_L{LAYER}.json", "w") as f:
    json.dump(out, f, indent=1)
_als = [v["al_layer"] for v in out.values()]
_b = [v["al_base"] for v in out.values() if v["al_base"] is not None]
print(f"=== LAYER{LAYER} 汇总: base均 {sum(_b)/len(_b):.3f} layer均 {sum(_als)/len(_als):.3f} 提升 {sum(_als)/len(_als)-sum(_b)/len(_b):+.3f}", flush=True)
