"""逐样本时间切片：naive vs ea，prefill/init/解码 分段计时。"""
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
    for sidx in [5, 10, 15, 20]:
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
        input_len = inputs.input_ids.shape[1]

        # ---- naive 详细计时 ----
        t_prefill_na = None
        na_tokens = 0
        yields = []
        t0 = time.time()
        prev = t0
        with torch.no_grad():
            for out_ids in model.naive_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                max_new_tokens=200, max_length=2048, is_llama3=False):
                now = time.time()
                yields.append(now - prev)
                prev = now
                na_tokens += 1
        na_total = time.time() - t0

        # ---- ea 详细计时 ----
        ea_steps = 0
        ea_yields = []
        t0 = time.time()
        prev = t0
        with torch.no_grad():
            for out_ids, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                      max_new_tokens=200, max_length=2048,
                                                      is_llama3=False, output_attention_scores=True):
                now = time.time()
                ea_yields.append(now - prev)
                prev = now
                ea_steps += 1
        ea_total = time.time() - t0

        # naive: 首个 yield = prefill + 1st token; 后续 = 每 token
        na_prefill = yields[0] if yields else 0
        na_decode = sum(yields[1:]) if len(yields) > 1 else 0
        # ea: 首个 yield = init; 后续 = 每步
        ea_init = ea_yields[0] if ea_yields else 0
        ea_decode = sum(ea_yields[1:]) if len(ea_yields) > 1 else 0

        print(f"\n样本{sidx}: 输入{input_len}token", flush=True)
        print(f"  naive: 总{na_total:.3f}s | prefill+首token {na_prefill:.3f}s | 解码{na_decode:.3f}s ({na_tokens}token, {na_decode/max(na_tokens-1,1):.4f}s/token)", flush=True)
        print(f"  ea:    总{ea_total:.3f}s | init {ea_init:.3f}s | 解码{ea_decode:.3f}s ({ea_steps}步)", flush=True)
        print(f"  全时加速: {na_total/ea_total:.3f}x | 仅解码加速: {na_decode/ea_decode:.3f}x (排除prefill/init)", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
