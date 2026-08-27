"""多样本 draft 隔离: 随机50条, NPU-FP32 target 输出为固定输入, draft 跑 CPU/NPU fp32 对比。
用法:
  MODE=npu NPU_CARD=6 python3 prec_50.py   # 算 source + draft@npu-fp32
  MODE=cpu python3 prec_50.py              # 固定输入 + draft@cpu-fp32
比较脚本随后用 prec_50_compare.py。
"""
import os, sys, random, pickle
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
MODE = os.environ.get("MODE", "npu")
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
from dream_s.model.ea_model import EaModel
from dream_s.model.cnets import DRAFT_CAP
from dream_s.model.utils import initialize_tree, reset_tree_mode
from datasets import load_dataset
from PIL import Image
import io

# MODE: npu(源+dnpu) / cpu(dcpu用npu源) / src_cpu(CPU源+dcpu) / dnpu_cpusrc(dnpu用CPU源)
if MODE in ("npu", "dnpu_cpusrc"):
    DEV = "npu"
elif MODE in ("cpu", "src_cpu"):
    DEV = "cpu"
else:
    raise ValueError(MODE)
dev_map = "npu:0" if DEV == "npu" else "cpu"
print(f"### MODE={MODE} DEV={DEV}", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map=dev_map)
model.eval()
print("加载完成", flush=True)

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)
QIDS = random.sample(range(1000), 50)
print(f"QIDS: {QIDS}", flush=True)

def draft_forward(fixed_embeds, fixed_hidden):
    """用固定输入跑 draft 一层前向 + head + topk, 返回 DRAFT_CAP 快照"""
    model.ea_layer.reset_kv()
    reset_tree_mode(model)
    DRAFT_CAP.clear()
    with torch.no_grad():
        out = model.ea_layer(fixed_hidden, inputs_embeds=fixed_embeds,
                             use_cache=True, output_attention_scores=False)
        out_hidden = out[0]
        last_headout = model.ea_layer.head_weight(out_hidden[:, -1])
        logp = model.ea_layer.logsoftmax(last_headout)
        topk_vals, topk_idx = torch.topk(logp, 4)
        DRAFT_CAP["head_out"] = last_headout.detach().float().cpu()
        DRAFT_CAP["topk_scores"] = topk_vals.detach().float().cpu()
        DRAFT_CAP["topk_tokens"] = topk_idx.detach().float().cpu()
    return dict(DRAFT_CAP)

results = {}
for qi, qid in enumerate(QIDS):
    row = ds[qid]
    img = row["decoded_image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    input_len = inputs.input_ids.shape[1]

    if MODE in ("npu", "src_cpu"):
        # 算 source (embed_model + target) + 捕获 input_hidden / last_hidden_states
        DRAFT_CAP.clear()
        with torch.no_grad():
            initialize_tree(
                inputs.input_ids, model, None, None, model.embed_model,
                inputs.pixel_values, inputs.image_sizes,
                output_draft_attention_scores=True, original_prompt_length=input_len)
        src_embeds = DRAFT_CAP["input_hidden"].float().cpu()
        src_hidden = DRAFT_CAP["last_hidden_states"].float().cpu()
        results[qid] = {"src_embeds": src_embeds, "src_hidden": src_hidden}
        cap = draft_forward(src_embeds.to(model.base_model.device), src_hidden.to(model.base_model.device))
        if MODE == "npu":
            results[qid]["draft_npu"] = cap
            out_pkl = "/home/y50063564/DREAM-S/output/prec50_npu.pkl"
        else:
            results[qid]["draft_cpu"] = cap
            out_pkl = "/home/y50063564/DREAM-S/output/prec50_srccpu_dcpu.pkl"
    else:
        # draft-only: 从指定源加载固定输入
        src_pkl = "/home/y50063564/DREAM-S/output/prec50_npu.pkl" if MODE == "cpu" else "/home/y50063564/DREAM-S/output/prec50_srccpu_dcpu.pkl"
        src = pickle.load(open(src_pkl,"rb"))[qid]
        cap = draft_forward(src["src_embeds"].to(model.base_model.device), src["src_hidden"].to(model.base_model.device))
        if MODE == "cpu":
            results[qid] = {"draft_cpu": cap}
            out_pkl = "/home/y50063564/DREAM-S/output/prec50_npusrc_dcpu.pkl"
        else:
            results[qid] = {"draft_npu": cap}
            out_pkl = "/home/y50063564/DREAM-S/output/prec50_cpusrc_dnpu.pkl"

    if (qi+1) % 10 == 0:
        print(f"  {qi+1}/50 qid={qid}", flush=True)
        with open(out_pkl, "wb") as f:
            pickle.dump(results, f)

with open(out_pkl, "wb") as f:
    pickle.dump(results, f)
print(f"=== 保存 {out_pkl} ({len(results)} 样本) ===", flush=True)
