"""pruning 开/关对比：token_ratio=0.7（剪枝） vs 1.0（不剪枝），同 30 样本 greedy。"""
import os
import sys
import math
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '6'

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
N = 30

def main():
    from dream_s.model.ea_model import EaModel
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    print("加载模型 (FP16) ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)

    table = pq.read_table("/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet")
    rows = []
    for i in range(min(N, table.num_rows)):
        row = {c: table.column(c)[i].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        rows.append((img, row["question"]))

    for name, tr in [("剪枝 on (ratio=0.7)", 0.7), ("剪枝 off (ratio=1.0)", 1.0)]:
        model.token_ratio = tr
        model.head_ratio = 1.0   # 不剪头
        model.use_prune_head = False
        model.ea_layer.threshold = math.log(1.0)
        model.ea_layer.total_tokens = 59
        accepts = []
        print(f"\n=== {name} ===", flush=True)
        for img, question in rows:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question}]}]
            prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = model.processor(images=img, text=prompt, truncation=True,
                                     return_tensors="pt").to(model.base_model.device)
            input_len = inputs.input_ids.shape[1]
            ea_final = input_len; ea_steps = 0
            with torch.no_grad():
                for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.0, top_k=0,
                                                      max_new_tokens=64, max_length=2048,
                                                      is_llama3=False, output_attention_scores=True):
                    ea_final = out.shape[1]; ea_steps += 1
            accepts.append((ea_final - input_len) / max(ea_steps, 1))
        print(f"  平均接受长度: {sum(accepts)/len(accepts):.2f}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
