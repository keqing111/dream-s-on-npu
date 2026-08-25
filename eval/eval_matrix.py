"""ratio 测试矩阵：head_ratio × token_ratio，naive vs ea 速度/接受/质量。"""
import os
import sys
import json
import time
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '7'   # 与全量测试的 npu:6 分开

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
OUTDIR = "/home/y50063564/DREAM-S/eval/data"
MAX_NEW = 64
MAX_LEN = 2048
N_SAMPLES = int(os.environ.get("MATRIX_N", "100"))

HEAD_RATIOS = [float(x) for x in os.environ.get("HEAD_RATIOS", "0.5,0.75,1.0").split(",")]
TOKEN_RATIOS = [float(x) for x in os.environ.get("TOKEN_RATIOS", "0.5,0.75,1.0").split(",")]

def truncate_list(lst, num):
    if num not in lst:
        return lst
    return lst[:lst.index(num) + 1]

def load_mathvista(path="/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet"):
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

def run_naive(model, inputs, input_len):
    t0 = time.time(); text = ""
    with torch.no_grad():
        for out_ids in model.naive_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                            max_new_tokens=MAX_NEW, max_length=MAX_LEN, is_llama3=False):
            dids = truncate_list(out_ids[0, input_len:].tolist(), model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(dids, skip_special_tokens=True)
    return text, time.time() - t0

def run_ea(model, inputs, input_len):
    t0 = time.time(); text = ""; steps = 0; final_len = input_len
    with torch.no_grad():
        for out_ids, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                  max_new_tokens=MAX_NEW, max_length=MAX_LEN,
                                                  is_llama3=False, output_attention_scores=True):
            steps += 1
            final_len = out_ids.shape[1]
            dids = truncate_list(out_ids[0, input_len:].tolist(), model.tokenizer.eos_token_id)
            text = model.tokenizer.decode(dids, skip_special_tokens=True)
    accept_len = (final_len - input_len) / max(steps, 1)
    return text, time.time() - t0, accept_len

def main():
    from dream_s.model.ea_model import EaModel
    print(f"加载 EaModel 到 npu:7 ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.bfloat16, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)

    ds = load_mathvista()[:N_SAMPLES]
    os.makedirs(OUTDIR, exist_ok=True)

    all_results = []
    for hr in HEAD_RATIOS:
        for tr in TOKEN_RATIOS:
            model.use_prune_head = (hr < 1.0)
            model.head_ratio = hr
            model.token_ratio = tr
            print(f"\n=== head_ratio={hr} token_ratio={tr} ===", flush=True)

            speedups, accepts, na_times, ea_times = [], [], [], []
            same_output = 0
            for i, r in enumerate(ds):
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": r["question"]}]}]
                try:
                    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = model.processor(images=r["decoded_image"], text=prompt,
                                             truncation=True, return_tensors="pt").to(model.base_model.device)
                except Exception:
                    continue
                input_len = inputs.input_ids.shape[1]
                na_text, na_t = run_naive(model, inputs, input_len)
                ea_text, ea_t, acc = run_ea(model, inputs, input_len)
                speedups.append(na_t/ea_t); accepts.append(acc)
                na_times.append(na_t); ea_times.append(ea_t)
                if na_text.strip() == ea_text.strip():
                    same_output += 1
            n = len(speedups)
            row = {
                "head_ratio": hr, "token_ratio": tr, "n": n,
                "avg_speedup": round(sum(speedups)/n, 3) if n else None,
                "avg_accept_len": round(sum(accepts)/n, 3) if n else None,
                "naive_avg_t": round(sum(na_times)/n, 3) if n else None,
                "ea_avg_t": round(sum(ea_times)/n, 3) if n else None,
                "same_output_ratio": round(same_output/n, 3) if n else None,
            }
            print(json.dumps(row), flush=True)
            all_results.append(row)

    with open(f"{OUTDIR}/matrix_results.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n=== 矩阵完成, 结果存 matrix_results.json ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
