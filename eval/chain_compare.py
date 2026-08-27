"""对比 draft 提案链 vs target 输出链（带 prompt），前 N 条样本。"""
import os
import sys
import torch
import torch_npu
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '6'

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
DRAFT = "/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b"
N = int(os.environ.get("CHAIN_N", "5"))

def main():
    from dream_s.model.ea_model import EaModel
    from dream_s.model import utils as U
    import dream_s.model.ea_model as EM
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    # monkey-patch tree_decoding: 抓每步 draft 提案 token
    draft_props_per_sample = {"props": []}
    _orig_tree = EM.tree_decoding
    def tree_decoding_hooked(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores=False):
        res = _orig_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores)
        # draft 贪心路径: retrieve_indices[0] 对应最优路径的 tree 位置
        if tree_candidates is not None and tree_candidates.numel() > 0:
            try:
                greedy_path = tree_candidates[0, retrieve_indices[0]].tolist()
            except Exception:
                greedy_path = []
            draft_props_per_sample["props"].append(greedy_path)
        return res
    EM.tree_decoding = tree_decoding_hooked

    DTYPE = os.environ.get("CHAIN_DTYPE", "fp16")
    torch_dtype_use = torch.float16 if DTYPE == "fp16" else torch.bfloat16
    print(f"加载模型 ({DTYPE}) ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=TARGET, ea_model_path=DRAFT,
        total_token=60, depth=8, top_k=4, threshold=1.0,
        torch_dtype=torch_dtype_use, low_cpu_mem_usage=True, device_map="npu:0")
    model.eval()
    print("加载完成", flush=True)

    table = pq.read_table("/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet")
    tok = model.tokenizer

    for sidx in range(N):
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True,
                                 return_tensors="pt").to(model.base_model.device)
        input_len = inputs.input_ids.shape[1]

        # --- target 输出链（naive greedy）---
        target_ids = None
        with torch.no_grad():
            for out in model.naive_generate(inputs, temperature=0.0, top_p=0.0, top_k=0,
                                            max_new_tokens=32, max_length=2048, is_llama3=False):
                target_ids = out
        target_text = tok.decode(target_ids[0, input_len:].tolist(), skip_special_tokens=True)

        # --- ea 解码过程：每步抓 draft 提案 top1 和 target 验证 ---
        draft_proposals = []   # (步, draft 提案 token 序列)
        accepts_seen = []
        with torch.no_grad():
            for step, (out_ids, ts, aas) in enumerate(model.ea_generate(
                    inputs, temperature=0.0, top_p=0.0, top_k=0,
                    max_new_tokens=32, max_length=2048, is_llama3=False, output_attention_scores=True)):
                # 当前步结束时 input_ids 的增量 = 接受的 token
                cur_len = out_ids.shape[1]
                if step == 0:
                    base_len = cur_len - 1   # 第一步接受 ~1
                    new_acc = out_ids[0, input_len:cur_len]
                else:
                    new_acc = out_ids[0, prev_len:cur_len]
                accepts_seen.append(new_acc.tolist())
                prev_len = cur_len
        # 汇总 target 接受的链
        ea_chain = [t for seg in accepts_seen for t in seg]

        print(f"\n===== 样本{sidx} | pid={row['pid']} =====", flush=True)
        print(f"问题: {row['question'][:80]}", flush=True)
        print(f"Prompt 尾: ...{prompt[-60:] if len(prompt)>60 else prompt}", flush=True)
        print(f"--- target naive 输出链 ---", flush=True)
        print(f"  {target_text[:200]}", flush=True)
        print(f"--- ea 接受链 ---", flush=True)
        ea_text = tok.decode(ea_chain, skip_special_tokens=True)
        print(f"  {ea_text[:200]}", flush=True)
        print(f"--- draft 每步提案链（贪心路径, 每步一行）---", flush=True)
        for pi, prop in enumerate(draft_props_per_sample["props"]):
            prop_text = tok.decode(prop, skip_special_tokens=True)
            print(f"  步{pi}: 提案 [{prop_text[:50]}] tokens={prop[:8]}", flush=True)
        if target_text.strip() == ea_text.strip():
            print(f"  [target 与 ea 输出一致]", flush=True)
        else:
            print(f"  [⚠ target 与 ea 输出不一致]", flush=True)
        draft_props_per_sample["props"].clear()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
