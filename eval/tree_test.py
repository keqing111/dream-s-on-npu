"""树配置对比：baseline / 阈值提前停止 / 小树 / 两者，各 30 条。"""
import os
import sys
import math
import time
import json
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '7'

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
N = 30

# (名称, threshold, total_token)
CONFIGS = [
    ("baseline",      1.0, 60),
    ("thr0.7",        0.7, 60),
    ("small20",       1.0, 20),
    ("thr0.7+small20",0.7, 20),
]

def main():
    from dream_s.model.ea_model import EaModel
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    print("加载模型 ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.bfloat16, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)

    table = pq.read_table("/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet")
    rows = []
    for i in range(min(N, table.num_rows)):
        row = {c: table.column(c)[i].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        rows.append((img, row["question"]))

    results = []
    for name, thr, total_tok in CONFIGS:
        # 设置树配置（不重载模型）
        model.ea_layer.threshold = math.log(thr)
        model.ea_layer.total_tokens = total_tok - 1
        model.ea_layer.depth = 8
        model.ea_layer.top_k = 4
        print(f"\n=== {name}: threshold={thr}, total_token={total_tok} ===", flush=True)

        per_tok_speedups, accepts, tree_sizes = [], [], []
        for img, question in rows:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question}]}]
            prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = model.processor(images=img, text=prompt, truncation=True,
                                     return_tensors="pt").to(model.base_model.device)
            input_len = inputs.input_ids.shape[1]

            # naive
            t0 = time.time(); na_tok = 0; na_last = None
            with torch.no_grad():
                for out in model.naive_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                max_new_tokens=200, max_length=2048, is_llama3=False):
                    na_tok += 1; na_last = out
            na_total = time.time() - t0
            na_gen = na_last.shape[1] - input_len

            # ea
            t0 = time.time(); ea_final = input_len; ea_steps = 0
            with torch.no_grad():
                for out, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                      max_new_tokens=200, max_length=2048,
                                                      is_llama3=False, output_attention_scores=True):
                    ea_final = out.shape[1]; ea_steps += 1
            ea_total = time.time() - t0
            ea_gen = ea_final - input_len

            per_tok = (na_total/max(na_gen,1)) / (ea_total/max(ea_gen,1))
            per_tok_speedups.append(per_tok)
            accepts.append(ea_gen / max(ea_steps, 1))   # 每步接受 token 数

        n = len(per_tok_speedups)
        row = {
            "config": name, "threshold": thr, "total_token": total_tok,
            "avg_per_token_speedup": round(sum(per_tok_speedups)/n, 3),
            "avg_accept_len": round(sum(accepts)/n, 2),
        }
        print(json.dumps(row), flush=True)
        results.append(row)

    with open("/home/y50063564/DREAM-S/eval/data/tree_config_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n=== 完成, 结果存 tree_config_results.json ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
