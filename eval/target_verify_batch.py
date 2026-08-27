"""批量 target 验证: 20条样本, 固定树, CPU/NPU target 分别验证。
用法:
  MODE=gen         NPU_CARD=6 python3 target_verify_batch.py  # 生成固定树 + verify_npu
  MODE=verify_cpu  python3 target_verify_batch.py              # CPU 验证固定树
  MODE=verify_npu  NPU_CARD=6 python3 target_verify_batch.py   # NPU 验证固定树
"""
import os, sys, random, pickle
import numpy as np
import torch
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, '/home/y50063564/DREAM-S')
MODE = os.environ.get("MODE", "gen")
QIDS = [int(x) for x in open("/home/y50063564/DREAM-S/eval/qids_verify20.txt").read().split(",")]
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")
from dream_s.model.ea_model import EaModel
from dream_s.model.kv_cache import initialize_past_key_values
from dream_s.model.utils import initialize_tree, reset_tree_mode, tree_decoding, evaluate_posterior
from datasets import load_dataset
from PIL import Image
import io

if MODE in ("gen", "verify_npu"):
    dev_map = "npu:0"
else:
    dev_map = "cpu"
print(f"### MODE={MODE} dev={dev_map} 样本数={len(QIDS)}", flush=True)
model = EaModel.from_pretrained(
    base_model_path="/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf",
    ea_model_path="/home/y50063564/DREAM-S/DREAM-S-llava-v1.6-vicuna-7b",
    total_token=60, depth=8, top_k=4, threshold=1.0,
    torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map=dev_map)
model.eval()
print("加载完成", flush=True)

ds = load_dataset("parquet",
    data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
    split="test").shuffle(seed=42)

results = {}
for qi, QID in enumerate(QIDS):
    row = ds[QID]
    img = row["decoded_image"]
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=img, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)
    input_ids = inputs.input_ids
    input_len = input_ids.shape[1]

    # 复刻 ea_generate 第一步
    model.ea_layer.reset_kv()
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        model.current_length_data.zero_()
    else:
        past_key_values, pv_data, cl_data = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = pv_data
        model.current_length_data = cl_data
    reset_tree_mode(model)

    with torch.no_grad():
        res = initialize_tree(input_ids, model, past_key_values, None, model.embed_model,
                              inputs.pixel_values, inputs.image_sizes,
                              output_draft_attention_scores=True, original_prompt_length=input_len)
        (draft_input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
         logits, hidden_state, sample_token, target_score, draft_score,
         image_start, image_end, text_start, text_end) = res

    if MODE == "gen":
        fixed_tree = {"draft_tokens": draft_tokens.cpu(), "retrieve_indices": retrieve_indices.cpu(),
                      "tree_mask": tree_mask.cpu(), "tree_position_ids": tree_position_ids.cpu()}
        with open(f"/home/y50063564/DREAM-S/output/fixed_tree_q{QID}.pkl", "wb") as f:
            pickle.dump(fixed_tree, f)
    else:
        with open(f"/home/y50063564/DREAM-S/output/fixed_tree_q{QID}.pkl", "rb") as f:
            fixed_tree = pickle.load(f)
        draft_tokens = fixed_tree["draft_tokens"].to(model.base_model.device)
        retrieve_indices = fixed_tree["retrieve_indices"].to(model.base_model.device)
        tree_mask = fixed_tree["tree_mask"].to(model.base_model.device)
        tree_position_ids = fixed_tree["tree_position_ids"].to(model.base_model.device)

    with torch.no_grad():
        model.base_model.model.tree_mask = tree_mask
        vlogits, hidden_new, voutputs = tree_decoding(model, draft_tokens, past_key_values, tree_position_ids, input_ids, retrieve_indices)
        padding = (torch.zeros(1,1,dtype=torch.long)-1).to(model.base_model.device)
        draft_pad = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_pad[0, retrieve_indices]
        best_candidate, accept_length, sample_p = evaluate_posterior(vlogits, candidates, None)

    results[QID] = {
        "logits": vlogits.float().cpu(),
        "argmax": vlogits.argmax(-1).cpu(),
        "top2": torch.topk(vlogits.float(), 2, dim=-1).values.cpu(),
        "candidates": candidates.cpu(),
        "best_candidate": int(best_candidate) if torch.is_tensor(best_candidate) else best_candidate,
        "accept_length": int(accept_length),
    }
    print(f"  qid={QID} accept={accept_length} best_cand={results[QID]['best_candidate']}", flush=True)

with open(f"/home/y50063564/DREAM-S/output/target_verify_batch_{MODE}.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"=== 保存 target_verify_batch_{MODE}.pkl ({len(results)} 样本) ===", flush=True)
