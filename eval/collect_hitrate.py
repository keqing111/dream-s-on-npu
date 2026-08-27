"""采集 10 条样本的逐轮数据：draft pos1 top-k logits + 每层 top-1 + target 每层 argmax + 完整树。
输出 output/spec_collect_{N}.json，随后统计各位置命中率。
"""
import os, sys, json, time, random
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

N = int(os.environ.get("N", "10"))
OUT = f"/home/y50063564/DREAM-S/output/spec_collect_{N}.json"

# ---- hooks ----
TRACE = []   # 当前样本的步骤
_orig_tree = EM.tree_decoding
def _tree_hooked(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores=False):
    res = _orig_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores)
    logits, hidden_state, outputs = res
    # 此刻 _draft_pos1 / _draft_level_top1 是当前树刚建好的值（initialize_tree/update_inference_inputs 刚调用过 topK_genrate）
    TRACE.append({
        "draft": tree_candidates, "ri": retrieve_indices, "pos_ids": tree_position_ids, "logits": logits,
        "pos1_topk": getattr(model.ea_layer, "_draft_pos1", None),
        "level_top1": getattr(model.ea_layer, "_draft_level_top1", None),
    })
    return res
EM.tree_decoding = _tree_hooked

_orig_ep = EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None and TRACE:
        TRACE[-1]["bc"] = res[0]; TRACE[-1]["al"] = res[1]
    return res
EM.evaluate_posterior = _ep_hooked

# ---- 模型 ----
print("加载模型 (FP16) ...", flush=True)
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
ds = ds.select(range(N))

all_samples = []
for sidx in range(N):
    row = ds[sidx]
    img = row["decoded_image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    input_len = inputs.input_ids.shape[1]

    TRACE.clear()
    with torch.no_grad():
        for out, ts, aas in model.ea_generate(inputs, temperature=float(os.environ.get("TEMP", "1.0")), top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out.shape[1] - input_len
    yields = len(TRACE)

    # ---- 后处理每步 ----
    steps_out = []
    for t in TRACE:
        dt = t["draft"]; ri = t["ri"]; pos_ids = t["pos_ids"]; lg = t["logits"]
        dtt = dt[0].tolist() if dt is not None else []
        bc = int(t.get("bc", -1)); al = int(t.get("al", -1))
        nleaf = ri.shape[0] if ri is not None else 0
        dep = ri.shape[1] if ri is not None else 0

        # draft pos1 top-k logits (钩子中已按当前树抓取)
        pos1_topk = t.get("pos1_topk") or {"tokens": [], "scores": []}
        # 每层 draft top-1 (钩子中已按当前树抓取)
        lvl = t.get("level_top1") or []
        # 每个叶子的每层 target argmax + draft token
        leaf_data = []
        if ri is not None:
            for leaf in range(nleaf):
                path = ri[leaf].tolist()
                draft_toks = [dtt[n] if 0 <= n < len(dtt) else -1 for n in path]
                tgt_argmax = lg[leaf].argmax(dim=-1).tolist()   # [depth]
                leaf_data.append({"path": draft_toks, "tgt_argmax": tgt_argmax})
        # 被采纳路径
        acc_ids = []
        if al >= 0 and bc >= 0 and bc < nleaf:
            acc_ids = [x for x in leaf_data[bc]["path"][:al+1] if x >= 0]
        steps_out.append({
            "accept_length": al, "best_candidate": bc,
            "num_leaves": nleaf, "max_depth": dep,
            "draft_tokens_all": dtt,
            "tree_position_ids": pos_ids.tolist() if pos_ids is not None else [],
            "pos1_topk": pos1_topk,
            "draft_level_top1": lvl,
            "leaves": leaf_data,
            "accepted_path_tokens": acc_ids,
        })

    all_samples.append({
        "sample": sidx, "question": row["question"], "input_len": input_len,
        "new_tokens": new_tokens, "yields": yields, "al_true": new_tokens/yields,
        "decoded_output": model.tokenizer.decode(out[0, input_len:].tolist(), skip_special_tokens=True),
        "steps": steps_out,
    })
    print(f"  样本{sidx}: yields={yields} new_tokens={new_tokens} AL={new_tokens/yields:.3f}", flush=True)

with open(OUT, "w") as f:
    json.dump(all_samples, f, ensure_ascii=False)
print(f"=== 保存 {OUT} ({N} 样本) ===", flush=True)
