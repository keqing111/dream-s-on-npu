"""诊断: eval 的 average_accept_length 公式 vs 每步 accept_length。"""
import os
import sys
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "4")

def main():
    from dream_s.model.ea_model import EaModel
    import dream_s.model.ea_model as EM
    from datasets import load_dataset
    from PIL import Image
    import io

    # 采集
    STATS = {"yields": 0, "evals": 0, "accept_sum": 0}
    _orig_eval = EM.evaluate_posterior
    def eval_hooked(logits, candidates, lp):
        res = _orig_eval(logits, candidates, lp)
        if res is not None:
            STATS["evals"] += 1
            STATS["accept_sum"] += int(res[1])
        return res
    EM.evaluate_posterior = eval_hooked

    model = EaModel.from_pretrained(
        base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
        ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()

    ds = load_dataset("parquet",
        data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
        split="test").shuffle(seed=42).select(range(3))

    for sidx in range(3):
        row = ds[sidx]
        img = row["decoded_image"]
        messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
        input_len = inputs.input_ids.shape[1]

        STATS["yields"] = 0; STATS["evals"] = 0; STATS["accept_sum"] = 0
        final_len = input_len
        with torch.no_grad():
            for out, ts, aas in model.ea_generate(inputs, temperature=1.0, top_p=0.6, top_k=0,
                                                  max_new_tokens=500, max_length=4096,
                                                  is_llama3=False, output_attention_scores=True):
                STATS["yields"] += 1
                final_len = out.shape[1]
        new_tokens = final_len - input_len
        print(f"样本{sidx}: yields={STATS['yields']} evals={STATS['evals']} "
              f"new_tokens={new_tokens} accept_sum={STATS['accept_sum']}")
        print(f"  eval公式 = {new_tokens}/STATS['yields'] = {new_tokens/STATS['yields']:.3f}")
        print(f"  每步 accept 均值 = {STATS['accept_sum']/STATS['evals']:.3f}")
        print(f"  accept_sum + evals = {STATS['accept_sum']+STATS['evals']} vs new_tokens={new_tokens}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
