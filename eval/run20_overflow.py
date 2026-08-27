"""对指定 qid 列表跑 T=0 + FP32 attention，采集 pos1 logits + 完整投机树。
计算接受长度 + pos1/2/3/4 接受率。
用法: QIDS_FILE=eval/qids_overflow20.txt NPU_CARD=6 python3 run20_overflow.py
"""
import os, sys, json, math, random
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
os.environ['FP32_ATTN'] = '1'   # 修复溢出
from dream_s.model.ea_model import EaModel
import dream_s.model.ea_model as EM
from datasets import load_dataset
from PIL import Image
import io

QIDS = [int(x) for x in open(os.environ.get("QIDS_FILE", "/home/y50063564/DREAM-S/eval/qids_overflow20.txt")).read().split(",")]
OUT = os.environ.get("OUT", "/home/y50063564/DREAM-S/output/spec_collect_overflow20.json")

TRACE = []
_orig_tree = EM.tree_decoding
def _tree_hooked(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores=False):
    res = _orig_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores)
    logits, hidden_state, outputs = res
    TRACE.append({
        "draft": tree_candidates, "ri": retrieve_indices, "logits": logits,
        "pos1_topk": getattr(model.ea_layer, "_draft_pos1", None),
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

print(f"加载模型 (FP16 权重 + FP32 attention)...", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
model.eval()
print(f"加载完成, 处理 {len(QIDS)} 条: {QIDS}", flush=True)

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)

all_samples = []
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
        for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            pass
    new_tokens = out.shape[1] - input_len
    yields = len(TRACE)
    al = new_tokens/yields if yields else 0

    steps_out = []
    for t in TRACE:
        dt = t["draft"]; ri = t["ri"]; lg = t["logits"]
        dtt = dt[0].tolist() if dt is not None else []
        bc = int(t.get("bc", -1)); al_ = int(t.get("al", -1))
        p1 = t.get("pos1_topk") or {"tokens": [], "scores": []}
        # 完整候选路径 + target argmax
        leaf_data = []
        if ri is not None:
            for leaf in range(ri.shape[0]):
                path = ri[leaf].tolist()
                draft_toks = [dtt[n] if 0 <= n < len(dtt) else -1 for n in path]
                tgt_argmax = lg[leaf].argmax(dim=-1).tolist()
                leaf_data.append({"path": draft_toks, "tgt_argmax": tgt_argmax})
        acc_ids = []
        if al_ >= 0 and bc >= 0 and bc < ri.shape[0]:
            acc_ids = [x for x in leaf_data[bc]["path"][:al_+1] if x >= 0]
        steps_out.append({
            "accept_length": al_, "best_candidate": bc,
            "pos1_topk": p1,
            "leaves": leaf_data,
            "accepted_path_tokens": acc_ids,
        })

    all_samples.append({
        "qid": qid, "question": row["question"], "input_len": input_len,
        "new_tokens": new_tokens, "yields": yields, "al_true": al,
        "decoded_output": model.tokenizer.decode(out[0, input_len:].tolist(), skip_special_tokens=True),
        "steps": steps_out,
    })
    print(f"  qid={qid} AL={al:.3f} yields={yields} new_tokens={new_tokens}", flush=True)

with open(OUT, "w") as f:
    json.dump(all_samples, f, ensure_ascii=False)
print(f"=== 保存 {OUT} ===", flush=True)

# ==== 汇总: 接受长度 + pos1/2/3/4 接受率 ====
from collections import Counter
als = [s['al_true'] for s in all_samples]
print(f"\n=== 汇总 ({len(all_samples)} 样本, T=0, FP32) ===")
print(f"平均接受长度: {sum(als)/len(als):.3f}")
acc_pos = Counter(); n_steps = 0
for s in all_samples:
    for st in s['steps']:
        n_steps += 1
        for k in range(1, 5):
            if st['accept_length'] >= k:
                acc_pos[k] += 1
for k in range(1, 5):
    print(f"pos{k} 接受率 P(accept>={k}): {acc_pos[k]}/{n_steps} = {acc_pos[k]/n_steps:.3f}")
