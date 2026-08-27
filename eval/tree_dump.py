"""导出完整树：draft_tokens 的所有节点 + 父子关系 + 每个路径解码。"""
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
    from dream_s.model.kv_cache import initialize_past_key_values
    from dream_s.model.utils import prepare_logits_processor
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
    tok = model.tokenizer

    for sidx in [0]:
        row = {c: table.column(c)[sidx].as_py() for c in table.column_names}
        img = Image.open(io.BytesIO(row["decoded_image"]["bytes"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["question"]}]}]
        prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = model.processor(images=img, text=prompt, truncation=True,
                                 return_tensors="pt").to(model.base_model.device)
        input_len = inputs.input_ids.shape[1]

        # 手动跑初始化树（第一步）
        pkv, pkv_data, cur_len = initialize_past_key_values(model.base_model)
        model.ea_layer.reset_kv()
        model.base_model.model.tree_mask = None

        from dream_s.model.utils import initialize_tree
        out = initialize_tree(inputs.input_ids, model, pkv, None, model.embed_model,
                              inputs.pixel_values if hasattr(inputs, "pixel_values") else None,
                              inputs.image_sizes if hasattr(inputs, "image_sizes") else None,
                              original_prompt_length=input_len)
        draft_input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_state, sample_token, target_score, draft_score, image_start, image_end, text_start, text_end = out

        print(f"\n===== 样本{sidx} 完整树 =====", flush=True)
        print(f"draft_tokens: {tuple(draft_tokens.shape)} (根+60 节点)", flush=True)
        print(f"retrieve_indices: {tuple(retrieve_indices.shape)}", flush=True)
        print(f"tree_mask: {tuple(tree_mask.shape)}", flush=True)
        print(f"tree_position_ids: {tree_position_ids.tolist()}", flush=True)
        print(f"image_start={image_start}, image_end={image_end}, text_start={text_start}, text_end={text_end}", flush=True)

        # 树的节点 token（去掉根）
        tokens = draft_tokens[0].tolist()
        print(f"\n树的节点 tokens（60个）:", flush=True)
        for i, t in enumerate(tokens):
            if i == 0:
                continue  # 根 = sample_token
            d = tok.decode([t], skip_special_tokens=True)
            print(f"  节点{i:2d}: token={t} pos={tree_position_ids[i-1].item() if hasattr(tree_position_ids,'item') else tree_position_ids[i-1]} 解码='{d}'", flush=True)

        # 从 tree_mask 重建父节点：节点 i 的父 = 它的祖先中 pos 少 1 的那个
        tm = tree_mask[0, 0]  # [61, 61]
        parents = [-1] * len(tokens)
        for i in range(1, len(tokens)):
            for j in range(1, len(tokens)):
                if i != j and tm[i, j] and tm[i].bool().sum().item() - tm[j].bool().sum().item() == 1:
                    parents[i] = j
                    break
            # 根 = 0
            if parents[i] == -1 and tm[i, 0]:
                parents[i] = 0
        print(f"\n每个节点的父节点:", flush=True)
        for i in range(1, len(tokens)):
            d = tok.decode([tokens[i]], skip_special_tokens=True)
            pd = tok.decode([tokens[parents[i]]], skip_special_tokens=True) if parents[i] >= 0 else "根"
            print(f"  节点{i:2d} '{d}' <- 父{parents[i]:2d} '{pd}'", flush=True)

        # 输出几条根到叶的路径（贪心 + 前几个分支）
        print(f"\n前 3 条根到叶路径解码:", flush=True)
        # 简单: 用 retrieve_indices 找叶节点路径
        # 叶 = 没有后代的节点
        is_parent = set(parents[1:])
        leaves = [i for i in range(1, len(tokens)) if i not in is_parent]
        leaves = leaves[:3]
        for leaf in leaves:
            # 从叶回溯到根
            path = []
            cur = leaf
            while cur > 0:
                path.append(tokens[cur])
                cur = parents[cur]
            path.reverse()
            text = tok.decode(path, skip_special_tokens=True)
            print(f"  叶{leaf}: [{text[:60]}]", flush=True)

        print(f"\n--- target naive 输出（对比）---", flush=True)
        with torch.no_grad():
            last = None
            for out in model.naive_generate(inputs, temperature=0.0, top_p=0.0, top_k=0,
                                            max_new_tokens=32, max_length=2048, is_llama3=False):
                last = out
        t_text = tok.decode(last[0, input_len:].tolist(), skip_special_tokens=True)
        print(f"  target: {t_text[:150]}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
