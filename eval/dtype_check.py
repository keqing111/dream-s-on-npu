"""dtype 诊断：FP16 加载 + 打印各组件 dtype + 测接受长度（对比 BF16）。"""
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

    dtype = torch.float16   # 匹配上游 FP16
    print(f"加载模型 dtype={dtype} ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=dtype, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)

    # ---- 打印各组件 dtype ----
    print("\n=== 组件 dtype ===", flush=True)
    emb = model.embed_model
    print(f"target (base_model.language_model): {next(model.base_model.parameters()).dtype}", flush=True)
    print(f"vision_tower: {next(emb.vision_tower.parameters()).dtype}", flush=True)
    print(f"projector linear_1: {emb.multi_modal_projector.linear_1.weight.dtype}", flush=True)
    print(f"draft (ea_layer): {next(model.ea_layer.parameters()).dtype}", flush=True)
    print(f"head_weight: {model.ea_layer.head_weight.weight.dtype if hasattr(model.ea_layer,'head_weight') else 'n/a'}", flush=True)
    print(f"vision_config dtype 目标: {emb.config.vision_config.torch_dtype if hasattr(emb.config.vision_config,'torch_dtype') else 'n/a'}", flush=True)

    # ---- 测 10 样本接受长度 ----
    table = pq.read_table("/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet")
    accepts = []
    for sidx in range(10):
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
        print(f"  样本{sidx} pixel_values dtype: {inputs.pixel_values.dtype}, shape: {tuple(inputs.pixel_values.shape)}", flush=True)
        # pixel_values 转成 vision_dtype
        vision_dtype = next(emb.vision_tower.parameters()).dtype
        inputs.pixel_values = inputs.pixel_values.to(vision_dtype)

        input_len = inputs.input_ids.shape[1]
        ea_final = input_len; ea_steps = 0
        with torch.no_grad():
            for out, ts, aas in model.ea_generate(inputs, temperature=0.0, top_p=0.0, top_k=0,
                                                  max_new_tokens=64, max_length=2048,
                                                  is_llama3=False, output_attention_scores=True):
                ea_final = out.shape[1]; ea_steps += 1
        acc = (ea_final - input_len) / max(ea_steps, 1)
        accepts.append(acc)
    print(f"\n=== FP16 10 样本平均接受长度: {sum(accepts)/len(accepts):.2f} ===", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
