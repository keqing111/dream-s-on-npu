"""采集 target vs draft 语义空间对比：每步草稿树 + target softmax top-k + target 对 draft 提案的概率。"""
import os
import sys
import json
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "7")

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
N = int(os.environ.get("CAPTURE_N", "50"))
OUT = "/home/y50063564/DREAM-S/eval/data/semantic_capture_shuffled.jsonl"
TOPK = 20

CAPTURE = {"steps": [], "pending": None, "best_candidate": None}

def _record_best(orig_eval, logits, candidates, logits_processor):
    res = orig_eval(logits, candidates, logits_processor)
    if res is not None:
        CAPTURE["best_candidate"] = res[0]  # best_candidate index
    return res

def main():
    from dream_s.model.ea_model import EaModel
    import dream_s.model.ea_model as EM
    import pyarrow.parquet as pq
    from PIL import Image
    import io
    from datasets import load_dataset

    # 用和 eval 一致的样本选择: seed=42 shuffle 后取前 N
    ds = load_dataset("parquet",
        data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
        split="test")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(N))

    # monkey-patch tree_decoding: 缓存 draft 树 + target logits（等 evaluate_posterior 给 best_candidate）
    _orig_tree = EM.tree_decoding
    def tree_hooked(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores=False):
        res = _orig_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores)
        logits, hidden_state, outputs = res  # logits: [tree_len, seq, vocab] (target)
        CAPTURE["pending"] = {"logits": logits, "retrieve_indices": retrieve_indices, "draft_tokens": tree_candidates}
        return res
    EM.tree_decoding = tree_hooked

    # monkey-patch evaluate_posterior: 拿 best_candidate，算被接受路径的 target 对齐
    _orig_eval = EM.evaluate_posterior
    def eval_hooked(logits, candidates, logits_processor):
        res = _orig_eval(logits, candidates, logits_processor)
        if res is not None and CAPTURE["pending"] is not None:
            best_candidate, accept_length, sample_p = res
            pending = CAPTURE["pending"]
            step_data = {
                "draft_tokens": pending["draft_tokens"][0].tolist() if pending["draft_tokens"] is not None else [],
                "retrieve_indices": pending["retrieve_indices"].tolist() if pending["retrieve_indices"] is not None else [],
                "best_candidate": int(best_candidate) if torch.is_tensor(best_candidate) else best_candidate,
                "accept_length": int(accept_length),
            }
            logits = pending["logits"]
            if logits is not None and logits.dim() == 3:
                probs = torch.softmax(logits.float(), dim=-1)
                topk_p, topk_i = torch.topk(probs, TOPK, dim=-1)
                step_data["target_topk_ids"] = topk_i.tolist()
                step_data["target_topk_probs"] = [[[round(x, 5) for x in tok] for tok in pos] for pos in topk_p.tolist()]
                # 被接受路径 (best_candidate) 的 target 对齐
                # evaluate_posterior: mask[i] = (candidates[bc, i+1] == argmax(logits[bc, i]))
                # 所以 draft token 在路径位置 j 对应 target logits 位置 j-1
                ri = pending["retrieve_indices"]
                draft_toks = pending["draft_tokens"][0].tolist() if pending["draft_tokens"] is not None else []
                view = []
                bc = int(best_candidate)
                if bc < ri.shape[0]:
                    path = ri[bc].tolist()
                    for j, tree_node in enumerate(path):
                        if j == 0:
                            continue  # 位置 0 是隐式接受的首 token，无直接验证
                        if j >= probs.shape[1]:
                            break
                        tok = draft_toks[tree_node] if tree_node < len(draft_toks) else -1
                        if tok < 0 or tok >= probs.shape[-1]:
                            continue
                        p = probs[bc, j-1, tok].item()   # target logits 位置 j-1
                        argmax_tok = probs[bc, j-1].argmax().item()
                        view.append({
                            "path_pos": j, "tree_node": tree_node, "draft_tok": tok,
                            "target_prob_draft_tok": round(p, 5),
                            "target_argmax": argmax_tok,
                            "target_prob_argmax": round(probs[bc, j-1, argmax_tok].item(), 5),
                            "accepted": j <= int(accept_length),
                        })
                step_data["accepted_path_view"] = view
            CAPTURE["steps"].append(step_data)
            CAPTURE["pending"] = None
        return res
    EM.evaluate_posterior = eval_hooked

    print(f"加载模型 (FP16) ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)

    all_data = []
    for sidx in range(N):
        row = ds[sidx]
        img = row["decoded_image"]
        if not isinstance(img, Image.Image):
            import io as _io
            img = Image.open(_io.BytesIO(img["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [{"type":"image"},{"type":"text","text":row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
        input_len = inputs.input_ids.shape[1]

        CAPTURE["steps"] = []
        with torch.no_grad():
            for out, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                  max_new_tokens=500, max_length=4096,
                                                  is_llama3=False, output_attention_scores=True):
                pass
        all_data.append({
            "sample": sidx, "pid": row["pid"], "question": row["question"],
            "input_len": input_len, "steps": CAPTURE["steps"],
        })
        if (sidx+1) % 10 == 0:
            print(f"  已采集 {sidx+1}/{N}", flush=True)

    with open(OUT, "w") as f:
        for d in all_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"=== 采集完成, 存 {OUT} ({len(all_data)} 样本) ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
