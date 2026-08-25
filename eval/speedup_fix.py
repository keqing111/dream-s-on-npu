"""按论文定义修正加速比：每 token 墙钟时间。tAR=t_naive_total/N_naive, tmethod=t_ea_total/N_ea, speedup=tAR/tmethod."""
import os
import sys
import time
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '6'

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"

def truncate_list(lst, num):
    if num not in lst:
        return lst
    return lst[:lst.index(num) + 1]

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
    print("=== 每 token 归一化的加速比（论文定义）===", flush=True)
    per_token_speedups = []
    total_ratio_speedups = []
    for sidx in [5, 10, 20, 30, 40]:
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
        input_len = inputs.input_ids.shape[1]

        # naive: 数 token（yield 次数），记总时间
        t0 = time.time(); na_tokens = 0; na_last = None
        with torch.no_grad():
            for out_ids in model.naive_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                max_new_tokens=200, max_length=2048, is_llama3=False):
                na_tokens += 1; na_last = out_ids
        na_total = time.time() - t0
        na_gen = na_last.shape[1] - input_len

        # ea: 记生成 token 数和总时间
        t0 = time.time(); ea_final = input_len
        with torch.no_grad():
            for out_ids, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                      max_new_tokens=200, max_length=2048,
                                                      is_llama3=False, output_attention_scores=True):
                ea_final = out_ids.shape[1]
        ea_total = time.time() - t0
        ea_gen = ea_final - input_len

        tAR = na_total / max(na_gen, 1)      # 论文: 标准自回归每 token 耗时
        tmethod = ea_total / max(ea_gen, 1)  # 论文: ea 每 token 耗时
        per_token = tAR / tmethod
        total_ratio = na_total / ea_total
        per_token_speedups.append(per_token)
        total_ratio_speedups.append(total_ratio)
        print(f"样本{sidx}: naive总{na_total:.2f}s/{na_gen}token(每tok {tAR*1000:.0f}ms)  "
              f"ea总{ea_total:.2f}s/{ea_gen}token(每tok {tmethod*1000:.0f}ms)  "
              f"每token加速={per_token:.3f}x  总时间比={total_ratio:.3f}x", flush=True)

    print(f"\n=== 汇总: 每token加速 平均 {sum(per_token_speedups)/len(per_token_speedups):.3f}x | "
          f"总时间比 平均 {sum(total_ratio_speedups)/len(total_ratio_speedups):.3f}x ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
