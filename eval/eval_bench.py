"""基准评测：DREAM-S 真实权重，naive vs ea 生成，对比速度+质量，保存输出。"""
import os
import sys
import json
import time
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
DEV = "npu:12"
OUTDIR = "/home/y50063564/DREAM-S/eval/data"
MAX_NEW = 64          # 每个样本生成上限
MAX_LEN = 2048        # 序列上限
SAMPLE_LIMIT = int(os.environ.get("SAMPLE_LIMIT", "0"))  # 0 = 全量

def build_prompt(question, choices=None):
    q = question.strip()
    if choices:
        letters = ["A", "B", "C", "D", "E"]
        for i, c in enumerate(choices):
            if c and str(c).strip():
                q += f"\n{letters[i]}. {c}"
        q += "\nAnswer with the letter."
    return f"USER: <image>\n{q}\nASSISTANT:"

def normalize(s):
    s = s.lower().strip()
    for ch in ".,!?;:\"'":
        s = s.replace(ch, " ")
    return " ".join(s.split())

def exact_match(pred, answer):
    if not answer:
        return None
    p = normalize(str(pred))
    a = normalize(str(answer))
    if not a:
        return None
    if a in p:
        return True
    # 数字答案：取预测中出现的数字比较
    import re
    pa = re.findall(r"\d+\.?\d*", p)
    aa = re.findall(r"\d+\.?\d*", a)
    if aa and pa and aa[0] == pa[0]:
        return True
    return False

def run_generation(model, inputs, method, max_new, max_len):
    """跑一个生成，返回 (最终 input_ids, 耗时秒)。"""
    t0 = time.time()
    last = None
    with torch.no_grad():
        if method == "naive":
            gen = model.naive_generate(inputs, temperature=0.0, top_p=0.0, top_k=0.0,
                                       max_new_tokens=max_new, max_length=max_len)
        else:
            gen = model.ea_generate(inputs, temperature=0.0, top_p=0.0, top_k=0.0,
                                    max_new_tokens=max_new, max_length=max_len)
        for out in gen:
            last = out
    dt = time.time() - t0
    return last, dt

def load_mathvista_parquet(path="/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet"):
    """用 pyarrow 直接读本地 parquet，返回样本 dict 列表（含解码图片）。"""
    import pyarrow.parquet as pq
    from PIL import Image
    import io as _io
    table = pq.read_table(path)
    rows = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        img_dict = row.get("decoded_image")
        img = None
        if isinstance(img_dict, dict) and img_dict.get("bytes"):
            img = Image.open(_io.BytesIO(img_dict["bytes"])).convert("RGB")
        rows.append({**row, "decoded_image": img})
    return rows

def main():
    from dream_s.model.ea_model import EaModel

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"加载 EaModel (draft + target) 到 {DEV} ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=32, depth=6, top_k=4, threshold=1.0,
        torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    print("模型加载完成", flush=True)

    print("加载 MathVista testmini (本地 parquet) ...", flush=True)
    ds = load_mathvista_parquet()
    if SAMPLE_LIMIT > 0:
        ds = ds[:SAMPLE_LIMIT]
    print(f"样本数: {len(ds)}", flush=True)

    results = []
    for i, r in enumerate(ds):
        img = r["decoded_image"]
        question = r["question"]
        answer = r.get("answer")
        choices = r.get("choices")
        prompt = build_prompt(question, choices)

        try:
            inputs = model.processor(images=img, text=prompt, return_tensors="pt").to(DEV)
            na_ids, na_t = run_generation(model, inputs, "naive", MAX_NEW, MAX_LEN)
            ea_ids, ea_t = run_generation(model, inputs, "ea", MAX_NEW, MAX_LEN)
        except Exception as e:
            print(f"  样本 {i} 失败: {type(e).__name__} {str(e)[:80]}", flush=True)
            results.append({"pid": r.get("pid"), "error": str(e)})
            continue

        na_text = model.tokenizer.decode(na_ids[0], skip_special_tokens=True) if na_ids is not None else ""
        ea_text = model.tokenizer.decode(ea_ids[0], skip_special_tokens=True) if ea_ids is not None else ""

        results.append({
            "pid": r.get("pid"),
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "naive_output": na_text,
            "ea_output": ea_text,
            "naive_time": round(na_t, 3),
            "ea_time": round(ea_t, 3),
            "naive_match": exact_match(na_text, answer),
            "ea_match": exact_match(ea_text, answer),
        })

        if (i + 1) % 50 == 0:
            # 实时汇总
            na_ok = sum(1 for x in results if x.get("naive_match") is True)
            ea_ok = sum(1 for x in results if x.get("ea_match") is True)
            scored = sum(1 for x in results if x.get("naive_match") is not None)
            print(f"  [{i+1}/{len(ds)}] naive_acc={na_ok/max(scored,1):.3f} ea_acc={ea_ok/max(scored,1):.3f} "
                  f"naive_t={na_t:.2f}s ea_t={ea_t:.2f}s", flush=True)

        # 定期保存
        if (i + 1) % 100 == 0:
            with open(f"{OUTDIR}/results_{i+1}.jsonl", "w") as f:
                for x in results:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 最终保存 + 汇总
    with open(f"{OUTDIR}/results_all.jsonl", "w") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    scored = [x for x in results if x.get("naive_match") is not None]
    na_ok = sum(1 for x in scored if x["naive_match"] is True)
    ea_ok = sum(1 for x in scored if x["ea_match"] is True)
    na_times = [x["naive_time"] for x in results if "naive_time" in x]
    ea_times = [x["ea_time"] for x in results if "ea_time" in x]
    na_avg = sum(na_times)/len(na_times)
    ea_avg = sum(ea_times)/len(ea_times)

    summary = {
        "n_samples": len(results),
        "n_scored": len(scored),
        "naive_acc": na_ok/len(scored) if scored else None,
        "ea_acc": ea_ok/len(scored) if scored else None,
        "naive_avg_time": round(na_avg, 3),
        "ea_avg_time": round(ea_avg, 3),
        "speedup": round(na_avg/ea_avg, 3) if ea_avg > 0 else None,
        "output_dir": OUTDIR,
    }
    with open(f"{OUTDIR}/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=== 评测完成 ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
