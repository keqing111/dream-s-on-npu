"""隔离 draft: 固定输入(来自某 source 的 input_hidden + last_hidden_states)，
只变 draft 的设备/精度，分层捕获输出。
用法: SOURCE=cpu_fp32 DDEV=npu DDT=fp16 NPU_CARD=4 python3 prec_draft_iso.py
  SOURCE: cpu_fp32 / npu_fp32 (固定输入来源)
  DDEV:   cpu / npu (draft 设备)
  DDT:    fp32 / fp16 / bf16 (draft 精度)
"""
import os, sys, pickle
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
SOURCE = os.environ.get("SOURCE", "cpu_fp32")
DDEV = os.environ.get("DDEV", "npu")
DDT = os.environ.get("DDT", "fp16")
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
from dream_s.model.ea_model import EaModel
from dream_s.model.cnets import DRAFT_CAP

torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[DDT]
dev_map = "npu:0" if DDEV == "npu" else "cpu"
print(f"### SOURCE={SOURCE} DDEV={DDEV} DDT={DDT}", flush=True)

model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=dev_map)
model.eval()
print("加载完成", flush=True)

# 固定输入
with open(f"/home/y50063564/DREAM-S/output/cap_{SOURCE}.pkl", "rb") as f:
    src = pickle.load(f)
fixed_embeds = src["input_hidden"].to(model.base_model.device).to(torch_dtype)
fixed_hidden = src["last_hidden_states"].to(model.base_model.device).to(torch_dtype)
print(f"fixed_embeds {tuple(fixed_embeds.shape)} fixed_hidden {tuple(fixed_hidden.shape)}", flush=True)

# reset draft 状态
model.ea_layer.reset_kv()
from dream_s.model.utils import reset_tree_mode
reset_tree_mode(model)

DRAFT_CAP.clear()
with torch.no_grad():
    out = model.ea_layer(
        fixed_hidden,
        inputs_embeds=fixed_embeds,
        use_cache=True,
        output_attention_scores=False,
    )
    if isinstance(out, tuple):
        out_hidden = out[0]
    else:
        out_hidden = out
    # head + topk
    last_headout = model.ea_layer.head_weight(out_hidden[:, -1])
    logp = model.ea_layer.logsoftmax(last_headout)
    topk_vals, topk_idx = torch.topk(logp, 4)
    DRAFT_CAP["head_out"] = last_headout.detach().float().cpu()
    DRAFT_CAP["topk_scores"] = topk_vals.detach().float().cpu()
    DRAFT_CAP["topk_tokens"] = topk_idx.detach().float().cpu()

out_path = f"/home/y50063564/DREAM-S/output/draftiso_{SOURCE}_dev{DDEV}_{DDT}.pkl"
with open(out_path, "wb") as f:
    pickle.dump(DRAFT_CAP, f)
print(f"=== 保存 {out_path} ===", flush=True)
if "topk_tokens" in DRAFT_CAP:
    tt = DRAFT_CAP["topk_tokens"][0].tolist()
    print(f"topk: {tt} -> {model.tokenizer.decode([int(x) for x in tt if x>=0])!r}", flush=True)
for k, v in DRAFT_CAP.items():
    print(f"  {k}: {tuple(v.shape)} min={v.min():.3f} max={v.max():.3f}", flush=True)
