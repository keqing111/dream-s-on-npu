"""rejection 诊断：greedy 模式下，rejection 位置 draft token 与 target token 的概率差。"""
import os
import sys
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '6'

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"

# 全局诊断收集
REJ_STATS = {"total_rej": 0, "near_tie": 0, "diffs": [], "samples": 0}

def evaluate_posterior_instrumented(logits, candidates, logits_processor, _orig=None):
    """greedy 分支的 instrumented 版本：在 rejection 位置记录概率差。"""
    import dream_s.model.utils as U
    if _orig is not None:
        # 直接调原逻辑拿结果
        best_candidate, accept_length, sample_p = _orig(logits, candidates, logits_processor)
    else:
        return None, None, None

    # greedy: rejection 位置 = 第一条路径上 candidates != argmax 的第一处
    argmax_ids = torch.argmax(logits[:, :-1], dim=-1)  # [tree, seq-1]
    target_probs = torch.softmax(logits, dim=-1)        # [tree, seq, vocab]
    # 沿 best_candidate 找第一处 rejection
    row = candidates[best_candidate]  # [seq]
    for i in range(1, min(row.shape[0]-1, argmax_ids.shape[1])):
        draft_tok = row[i]
        target_tok = argmax_ids[best_candidate, i-1]
        if draft_tok != target_tok and draft_tok != -1:
            p_draft = target_probs[best_candidate, i-1, draft_tok].item()
            p_target = target_probs[best_candidate, i-1, target_tok].item()
            REJ_STATS["total_rej"] += 1
            d = abs(p_draft - p_target)
            REJ_STATS["diffs"].append(d)
            if d < 0.003:
                REJ_STATS["near_tie"] += 1
            break
    return best_candidate, accept_length, sample_p

def main():
    from dream_s.model.ea_model import EaModel
    import dream_s.model.utils as U
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    # monkey-patch: ea_model 用 from .utils import * 捕获了 evaluate_posterior，要 patch ea_model 命名空间
    import dream_s.model.ea_model as EM
    _orig_eval = EM.evaluate_posterior
    EM.evaluate_posterior = lambda logits, cand, lp: evaluate_posterior_instrumented(logits, cand, lp, _orig_eval)

    print("加载模型 ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.bfloat16, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)

    table = pq.read_table("/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet")
    for sidx in range(15):
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True,
                                 return_tensors="pt").to(model.base_model.device)
        with torch.no_grad():
            for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.0, top_k=0,
                                                  max_new_tokens=64, max_length=2048,
                                                  is_llama3=False, output_attention_scores=True):
                pass
        REJ_STATS["samples"] += 1
        if (sidx+1) % 5 == 0:
            n = REJ_STATS["total_rej"]
            print(f"  [{sidx+1}/15] rejection={n} near_tie(<0.003)={REJ_STATS['near_tie']} "
                  f"比例={REJ_STATS['near_tie']/max(n,1):.3f}", flush=True)

    n = REJ_STATS["total_rej"]
    diffs = REJ_STATS["diffs"]
    print(f"\n=== rejection 诊断（15 样本, greedy）===", flush=True)
    print(f"总 rejection: {n}, near_tie(<0.003): {REJ_STATS['near_tie']} ({REJ_STATS['near_tie']/max(n,1):.3f})", flush=True)
    if diffs:
        import statistics
        diffs_sorted = sorted(diffs)
        print(f"概率差: 均值 {statistics.mean(diffs):.4f}, 中位数 {statistics.median(diffs):.4f}, "
              f"p90 {diffs_sorted[int(0.9*len(diffs))]:.4f}, p50 {diffs_sorted[int(0.5*len(diffs))]:.4f}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
