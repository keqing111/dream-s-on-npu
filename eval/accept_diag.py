"""诊断 ea_generate 的接受率：每步接受几个 token，解释加速比。"""
import sys
import time
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
DEV = "npu:12"

def main():
    from dream_s.model.ea_model import EaModel
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    print(f"加载 EaModel 到 {DEV} ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=32, depth=6, top_k=4, threshold=1.0,
        torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    print("加载完成", flush=True)

    # 读 MathVista 前几行
    table = pq.read_table("/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet")
    for sidx in range(3):
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img_dict = row["decoded_image"]
        img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
        prompt = f"USER: <image>\n{row['question']}\nASSISTANT:"
        inputs = model.processor(images=img, text=prompt, return_tensors="pt").to(DEV)

        # naive
        t0 = time.time(); last = None
        with torch.no_grad():
            for out in model.naive_generate(inputs, temperature=0, top_p=0, top_k=0, max_new_tokens=64, max_length=2048):
                last = out
        na_t = time.time() - t0
        na_len = last.shape[1]

        # ea: 跟踪每次 yield 的 input_ids 增长 = 接受的 token 数
        t0 = time.time()
        prev_len = inputs['input_ids'].shape[1]
        accepts = []
        steps = 0
        with torch.no_grad():
            for out in model.ea_generate(inputs, temperature=0, top_p=0, top_k=0, max_new_tokens=64, max_length=2048):
                cur_len = out.shape[1]
                accepts.append(cur_len - prev_len)
                prev_len = cur_len
                steps += 1
        ea_t = time.time() - t0
        ea_len = prev_len - inputs['input_ids'].shape[1]

        na_accept = sum(accepts)
        avg_accept = sum(accepts)/len(accepts) if accepts else 0
        print(f"样本{sidx}: naive={na_t:.2f}s/{na_len}token  ea={ea_t:.2f}s/{ea_len}token/{steps}步 "
              f"加速={na_t/ea_t:.2f}x  平均每步接受={avg_accept:.2f}个", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
