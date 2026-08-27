"""逐前向 trace: 每步投机树序列 + 命中(accept)的 token + input_ids 增长。seed=42 复现 eval。"""
import os
import sys
import json
import torch
import torch_npu
import random
import numpy as np
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")

QID = int(os.environ.get("QID", "0"))
TRACE = []

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    from dream_s.model.ea_model import EaModel
    import dream_s.model.ea_model as EM
    from datasets import load_dataset
    from PIL import Image
    import io

    _orig_tree = EM.tree_decoding
    def tree_hooked(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores=False):
        res = _orig_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores)
        TRACE.append({"draft_tokens": tree_candidates, "retrieve_indices": retrieve_indices})
        return res
    EM.tree_decoding = tree_hooked

    _orig_eval = EM.evaluate_posterior
    def eval_hooked(logits, candidates, lp):
        res = _orig_eval(logits, candidates, lp)
        if res is not None and TRACE:
            TRACE[-1]["best_candidate"] = res[0]
            TRACE[-1]["accept_length"] = res[1]
            TRACE[-1]["logits_argmax"] = logits.argmax(dim=-1).tolist() if logits.dim()==3 else None
        return res
    EM.evaluate_posterior = eval_hooked

    model = EaModel.from_pretrained(
        base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
        ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()
    tok = model.tokenizer

    ds = load_dataset("parquet",
        data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
        split="test").shuffle(seed=42)
    ds = ds.select(range(QID + 1))
    row = ds[QID]
    img = row["decoded_image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    input_len = inputs.input_ids.shape[1]

    print(f"### qid={QID} question={row['question']} input_len={input_len} image_size={img.size}", flush=True)
    print(f"### prompt: {prompt!r}", flush=True)

    yields = []
    with torch.no_grad():
        for out, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            yields.append(out.shape[1] - input_len)

    new_tokens = out.shape[1] - input_len
    total_ids = len(yields)
    print(f"\n### yields={total_ids} new_tokens={new_tokens} reported AL={new_tokens/total_ids:.3f}", flush=True)

    print(f"\n### 每步投机 trace ===", flush=True)
    acc_sum = 0
    prev_len = 0
    for i, t in enumerate(TRACE):
        dt = t["draft_tokens"]
        dt_toks = dt[0].tolist() if dt is not None else []
        al = int(t.get("accept_length", -1))
        bc = int(t.get("best_candidate", -1))
        ri = t["retrieve_indices"]
        # 被接受的路径 tokens
        path_toks = []
        if al >= 0 and ri is not None and bc >= 0 and bc < ri.shape[0]:
            path = ri[bc].tolist()
            path_toks = [dt_toks[n] if n < len(dt_toks) else -1 for n in path[:al+1]]
            path_toks = [x for x in path_toks if x >= 0]
        accepted_text = tok.decode(path_toks, skip_special_tokens=True)
        # 树里 decode 前几个不同 token
        tree_unique = []
        for x in dt_toks:
            if x not in tree_unique and x != -1:
                tree_unique.append(x)
            if len(tree_unique) >= 8:
                break
        tree_text = tok.decode(tree_unique, skip_special_tokens=True)
        grow = yields[i] - prev_len if i < len(yields) else 0
        prev_len = yields[i]
        acc_sum += al
        print(f"[步{i}] accept={al} input_ids增长={grow} cand_path_len={ri.shape[1] if ri is not None else '?'}", flush=True)
        print(f"   命中token(路径): {path_toks}", flush=True)
        print(f"   命中文本: {accepted_text!r}", flush=True)
        print(f"   树前几个token: {tree_unique}", flush=True)
        print(f"   树文本片段: {tree_text[:80]!r}", flush=True)
    print(f"\n### sum(accept)={acc_sum} 报告AL = {1 + acc_sum/len(TRACE):.3f} vs eval的 {new_tokens/total_ids:.3f}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
