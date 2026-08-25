"""数据生成：7B target + COCO 分片 → .pt（复用 ge_data 逻辑）。"""
import os
import sys
import json
import torch
import torch_npu
import pyarrow.parquet as pq
import traceback

TARGET = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
COCO_PQ = "/tmp/coco_0000.parquet"
OUTDIR = "/tmp/train_data"
DEV = "npu:12"
N_SMOKE = 8          # 冒烟跑 8 条
N_TOTAL = None       # None = 全部分片(2932)

# ---------- 从 ge_data_all_llava_mix665k.py 复制的纯函数 ----------
def compute_attention_entropy(attn_weights, eps=1e-8):
    with torch.no_grad():
        entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
        return entropy.mean(dim=1)

def mid_attention_score(attn_tuple, best_layer_idx):
    L = len(attn_tuple)
    B, H, S, K = attn_tuple[0].shape
    score_list = []
    for l in range(L):
        max_per_head = attn_tuple[l].max(dim=-1).values
        score_l = max_per_head.mean(dim=1)
        score_list.append(score_l)
    score_stack = torch.stack(score_list, dim=0).to('cpu')
    best_layer_score = torch.gather(score_stack, dim=0, index=best_layer_idx.unsqueeze(0)).squeeze(0)
    top_layer_score = score_stack[-1]
    return best_layer_score, top_layer_score

def mid_feature_collect_and_score(features_tuple, attn_tuple, eps=1e-8):
    L = int(0.98 * (len(features_tuple)-1))
    B, s, d = features_tuple[0].shape
    features_stack = torch.stack(features_tuple[:L+1], dim=0).to("cpu")
    att_entropy_list = []
    for l in range(L):
        att_entropy = compute_attention_entropy(attn_tuple[l], eps=eps).to("cpu")
        att_entropy_list.append(att_entropy)
    att_entropy_stack = torch.stack(att_entropy_list, dim=0)
    total_metric = att_entropy_stack[1:L-1]
    best_layer_idx = total_metric.argmin(dim=0) + 1
    best_layer_idx_expanded = best_layer_idx.unsqueeze(0).unsqueeze(-1)
    best_features_b = torch.gather(features_stack, dim=0,
                                   index=best_layer_idx_expanded.expand(1, B, s, d)).squeeze(0)
    best_layer_scores, top_layer_scores = mid_attention_score(attn_tuple, best_layer_idx)
    return best_features_b.cpu(), best_layer_scores.cpu(), top_layer_scores.cpu()

def find_subsequence(tensor, subsequence):
    tensor_list = tensor.tolist()
    subseq_list = subsequence if isinstance(subsequence, list) else subsequence.tolist()
    for i in range(len(tensor_list) - len(subseq_list) + 1):
        if tensor_list[i:i+len(subseq_list)] == subseq_list:
            return i
    return -1

def main():
    from transformers import LlavaNextForConditionalGeneration, AutoProcessor
    torch.manual_seed(0)
    print(f"加载 7B target 到 {DEV} ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        TARGET, torch_dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    processor = AutoProcessor.from_pretrained(TARGET)
    print("模型加载完成")

    os.makedirs(OUTDIR, exist_ok=True)
    table = pq.read_table(COCO_PQ, columns=["image_id", "image"])
    total = N_TOTAL if N_TOTAL else table.num_rows
    print(f"COCO 分片共 {table.num_rows} 行, 本次处理 {min(total, table.num_rows)} 行")

    assist_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    assist_len = len(assist_tokens)
    prompt = ("USER: <image>\nWhat is in this image? "
              "ASSISTANT: A photo of a scene with various objects in the image.")

    from PIL import Image
    count = 0
    for i in range(min(total, table.num_rows)):
        img_id = table.column('image_id')[i].as_py()
        img_dict = table.column('image')[i].as_py()
        img_bytes = img_dict.get('bytes')
        if not img_bytes:
            continue
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # 不改图片内容，保持原始尺寸

        inputs = processor(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEV) if hasattr(v, 'to') else v for k, v in inputs.items()}
        seq_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outs = model(**inputs, output_hidden_states=True, output_attentions=True)
        torch.npu.empty_cache()

        mid_feature, _, target_score = mid_feature_collect_and_score(
            outs.hidden_states, outs.attentions)
        inputs_embeds = outs.hidden_states[0].cpu()
        target = outs.hidden_states[-1].cpu()

        # loss_mask: 找 ASSISTANT: 之后的部分
        input_ids_cpu = inputs['input_ids'].cpu()[0]
        pos = find_subsequence(input_ids_cpu, assist_tokens)
        loss_mask = torch.zeros_like(input_ids_cpu)
        if pos >= 0:
            end = min(pos + assist_len + 12, seq_length)  # 固定 12 个 assistant token
            loss_mask[pos + assist_len:end] = 1
        loss_mask = loss_mask.unsqueeze(0)

        # pruning_indices: 按 image attention score 剪枝（ge_data 逻辑）
        last_image_position = (input_ids_cpu == 32000).nonzero()[-1, 0].item()
        image_start, image_end = 5, last_image_position + 1
        img_score = target_score[:, image_start:image_end]
        pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        pruning_indices = []
        for ratio in pruning_ratios:
            if ratio != 1:
                topk = img_score.topk(int((image_end - image_start) * ratio)).indices + image_start
                keep = torch.cat((torch.arange(image_start), topk.squeeze(0),
                                  torch.arange(image_end, seq_length))).sort().values
                pruning_indices.append(keep)
            else:
                pruning_indices.append(torch.arange(seq_length))

        td = {
            "inputs_embeds": inputs_embeds,
            "target": target,
            "hidden_state_mid_a": mid_feature,
            "loss_mask": loss_mask,
            "attention_mask": inputs['attention_mask'].cpu(),
            "pruning_indices": pruning_indices,
            "pruning_ratios": pruning_ratios,
            "image_id": img_id,
        }
        fname = f"{OUTDIR}/data_{count}.ckpt"
        torch.save(td, fname)
        count += 1
        if count % 2 == 0:
            print(f"  [{count}] 已保存 {fname}: inputs_embeds={tuple(inputs_embeds.shape)} "
                  f"target={tuple(target.shape)} loss_mask 非零={int(loss_mask.sum())}", flush=True)
    print(f"=== 数据生成完成: {count} 条 -> {OUTDIR} ===")

import io
if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
