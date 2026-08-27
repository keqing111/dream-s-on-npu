"""仔细测 target greedy 生成：数 yield、每步 token、EOS 位置。"""
import os
import sys
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '6'

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"

def main():
    from dream_s.model.ea_model import EaModel
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    print("加载模型 ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)
    tok = model.tokenizer
    print(f"eos_token_id = {tok.eos_token_id}, eos = {tok.convert_ids_to_tokens(tok.eos_token_id)}", flush=True)

    table = pq.read_table("/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet")
    for sidx in [0, 1, 5, 10]:
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [{"type":"image"},{"type":"text","text":row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
        input_len = inputs.input_ids.shape[1]

        yields = []
        with torch.no_grad():
            for out in model.naive_generate(inputs, temperature=0.0, top_p=0.0, top_k=0,
                                            max_new_tokens=32, max_length=2048, is_llama3=False):
                yields.append(out)
        n_yields = len(yields)
        last = yields[-1]
        gen_ids = last[0, input_len:].tolist()
        print(f"\n样本{sidx}: yield 次数={n_yields}, 生成 token ids={gen_ids[:40]}", flush=True)
        print(f"  token 数={len(gen_ids)}, EOS 在 ids 里? {tok.eos_token_id in gen_ids}", flush=True)
        # 如果 yield 次数>生成 token 数，说明有些 yield 没增加 token
        if n_yields != len(gen_ids):
            print(f"  ⚠ yield 次数({n_yields}) != token 数({len(gen_ids)})", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
