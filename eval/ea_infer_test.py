"""投机解码推理冒烟：真实 draft + 真实 target，naive vs ea 对比（NPU）。"""
import sys
import time
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
DEV = "npu:12"

def run_gen(model, inputs, method):
    t0 = time.time()
    last = None
    with torch.no_grad():
        if method == "naive":
            gen = model.naive_generate(inputs, temperature=0.0, top_p=0.0, top_k=0.0,
                                       max_new_tokens=32, max_length=2048)
        else:
            gen = model.ea_generate(inputs, temperature=0.0, top_p=0.0, top_k=0.0,
                                    max_new_tokens=32, max_length=2048)
        for out in gen:
            last = out
    return last, time.time() - t0

def main():
    from dream_s.model.ea_model import EaModel
    print(f"加载 EaModel 到 {DEV} ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=32, depth=6, top_k=4, threshold=1.0,
        torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    print("加载完成", flush=True)

    from PIL import Image
    img = Image.open("/tmp/coco_samples/coco_9.jpg").convert("RGB")
    prompt = "USER: <image>\nWhat objects are in this image? ASSISTANT:"
    inputs = model.processor(images=img, text=prompt, return_tensors="pt").to(DEV)
    print("inputs:", {k: tuple(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}, flush=True)

    print("=== naive_generate (基线自回归) ===", flush=True)
    na_ids, na_t = run_gen(model, inputs, "naive")
    na_text = model.tokenizer.decode(na_ids[0], skip_special_tokens=True)
    print(f"  耗时 {na_t:.2f}s, 输出 {len(na_text)} 字符", flush=True)
    print(f"  文本: {na_text[:150]}", flush=True)

    print("=== ea_generate (投机解码) ===", flush=True)
    ea_ids, ea_t = run_gen(model, inputs, "ea")
    ea_text = model.tokenizer.decode(ea_ids[0], skip_special_tokens=True)
    print(f"  耗时 {ea_t:.2f}s, 输出 {len(ea_text)} 字符", flush=True)
    print(f"  文本: {ea_text[:150]}", flush=True)

    print(f"=== 速度对比: naive {na_t:.2f}s vs ea {ea_t:.2f}s, 加速比 {na_t/ea_t:.2f}x ===", flush=True)
    print("=== PASS: 投机解码 (ea_generate) 真实权重在 NPU 上跑通 ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
