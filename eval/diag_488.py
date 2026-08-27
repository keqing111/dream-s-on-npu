"""诊断 AL=50.25 的样本：逐步打印 accept_length / input_ids 增长 / 树深度。"""
import os
import sys
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "4")

QID = int(os.environ.get("QID", "488"))

import random
import numpy as np

def main():
    # 与 eval_llava_npu.py 相同的 seed 设置
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    from dream_s.model.ea_model import EaModel
    import dream_s.model.ea_model as EM
    from datasets import load_dataset
    from PIL import Image
    import io

    # 记录每步
    STATS = {"steps": [], "yields": [], "input_lens": []}
    _orig_eval = EM.evaluate_posterior
    def eval_hooked(logits, candidates, lp):
        res = _orig_eval(logits, candidates, lp)
        if res is not None:
            STATS["steps"].append({
                "accept": int(res[1]),
                "cand_shape": tuple(candidates.shape),
            })
        return res
    EM.evaluate_posterior = eval_hooked

    model = EaModel.from_pretrained(
        base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
        ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()

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
    print(f"qid={QID} question={row['question'][:60]} input_len={input_len}", flush=True)

    with torch.no_grad():
        for out, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                              max_new_tokens=500, max_length=4096,
                                              is_llama3=False, output_attention_scores=True):
            STATS["yields"].append(out.shape[1] - input_len)

    new_tokens = out.shape[1] - input_len
    total_ids = len(STATS["yields"])
    accepts = [s["accept"] for s in STATS["steps"]]
    print(f"\n=== 结果 (seed=42) ===")
    print(f"yields={total_ids}, new_tokens={new_tokens}, eval公式 AL = {new_tokens/total_ids:.3f}")
    print(f"每步 accept_length 序列: {accepts}")
    print(f"每步 input_ids 增长(相对上一步): {[STATS['yields'][0]]+[b-a for a,b in zip(STATS['yields'][:-1], STATS['yields'][1:])]}")
    print(f"candidates 形状集合: {sorted(set(s['cand_shape'] for s in STATS['steps']))}")
    print(f"accept 是否全≤10: {all(a<=10 for a in accepts)}, max accept={max(accepts) if accepts else 0}")
    print(f"sum(accept+1)={sum(a+1 for a in accepts)} vs new_tokens={new_tokens}")
    gen = out[0, input_len:].tolist()
    print(f"生成 token 数(len)= {len(gen)}, 含 eos? {model.tokenizer.eos_token_id in gen}")
    txt = model.tokenizer.decode(gen, skip_special_tokens=True, spaces_between_special_tokens=False)
    print(f"解码输出({len(txt)}字符): {txt[:200]}")
    # 对比 evals 与 yields 数量
    print(f"evals(evaluate_posterior 调用)={len(STATS['steps'])} vs yields={total_ids}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
