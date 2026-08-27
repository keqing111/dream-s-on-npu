"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import base64
import io
from io import BytesIO
import torch
import os
import io

from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import time

import argparse

import importlib

# try:
#     from ..model.ea_model import EaModel
# except:
#     from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
from transformers import LlavaNextProcessor
from PIL import Image
import requests

import re

# NPU 适配: 物理卡可通过 NPU_CARD 环境变量指定（默认 6）
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = os.environ.get("NPU_CARD", "6")

# 复现协议: 固定随机种子（论文 greedy 复现）
import random
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def truncate_list(lst, num):
    if num not in lst:
        return lst


    first_index = lst.index(num)


    return lst[:first_index + 1]


def find_next_non_image_token(input_ids, image_token_ids):
    image_indices = torch.where(torch.tensor([id in image_token_ids for id in input_ids]))[0]

    if len(image_indices) == 0:
        return -1
    last_image_index = image_indices[-1].item()
    for i in range(last_image_index + 1, len(input_ids)):
        if input_ids[i].item() not in image_token_ids:
            return i

    return -1


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ea-model-path",
    type=str,
    default="//ofa_main_head_mix_final_13B/",
    help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")
parser.add_argument("--base-model-path", type=str, default="",
                    help="path of basemodel, huggingface project or local path")
parser.add_argument(
    "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
)
parser.add_argument(
    "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
)
parser.add_argument("--model-type", type=str, default="vicuna",choices=["llama-2-chat","vicuna","mixtral","llama-3-instruct"])
parser.add_argument(
    "--total-token",
    type=int,
    default=60,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--max-new-token",
    type=int,
    default=500,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=1.0,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--depth",
    type=int,
    default=8,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--topk",
    type=int,
    default=4,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--version",
    type=str,
    default="default",
    help="The version of code.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--layer",
    type=str,
    default="mid_a",
    help="The maximum number of new generated tokens.",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="MathVista",
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--max-length",
    type=int,
    default=4096,
    help="Max sequence length (must exceed input_len + max_new; default 4096 avoids truncating long inputs)",
)
args = parser.parse_args()

# NPU 适配: 直接导入本项目 EaModel（原为 eagle.model.ea_model_head2，已损坏）
import sys
sys.path.insert(0, '/home/y50063564/DREAM-S')
from dream_s.model.ea_model import EaModel

model = EaModel.from_pretrained(
    base_model_path=args.base_model_path,
    ea_model_path=args.ea_model_path,
    total_token=args.total_token,
    depth=args.depth,
    top_k=args.topk,
    threshold=args.threshold,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="npu:0",
)
model.eval()

# ==== TRACE HOOK (注入) ====
import dream_s.model.ea_model as _EM
TRACE = []
_orig_tree = _EM.tree_decoding
def _tree_hooked(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores=False):
    res = _orig_tree(model, tree_candidates, past_key_values, tree_position_ids, input_ids, retrieve_indices, output_draft_attention_scores)
    TRACE.append({"draft": tree_candidates, "ri": retrieve_indices})
    return res
_EM.tree_decoding = _tree_hooked
_orig_ep = _EM.evaluate_posterior
def _ep_hooked(logits, candidates, lp):
    res = _orig_ep(logits, candidates, lp)
    if res is not None and TRACE:
        TRACE[-1]["bc"] = res[0]; TRACE[-1]["al"] = res[1]
    return res
_EM.evaluate_posterior = _ep_hooked
# ==== TRACE HOOK END ====
# warmup(model)

question_file = f"data/question.jsonl"

from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from datasets import load_dataset
import time
import json

sample_num = int(os.environ.get("SAMPLE_NUM", "1000"))
QID = int(os.environ.get("QID", "0"))

if args.dataset == "MMT-Bench":
    ds = load_dataset(path="datasets/test/mmt", data_files = "datasets/test/mmt/MMT-Bench_ALL.tsv", split="train")
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "ScienceQA":
    ds = load_dataset(
        "parquet", 
        data_files={'train': 'datasets/test/scienceqa/data/train-00000-of-00001-1028f23e353fbe3e.parquet', 'test': 'datasets/test/scienceqa/data/test-00000-of-00001-f0e719df791966ff.parquet'},
        split="test"
    )
    ds = ds.filter(lambda record: record["image"] is not None)  
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "ChartQA":
    ds = load_dataset(
        "parquet",
        data_files={'test': '/home/y50063564/DREAM-S/eval/data/chartqa_test.parquet'},
        split="test"
    )
    ds = ds.filter(lambda record: record["image"] is not None)
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "MathVista":
    ds = load_dataset(
        "parquet",
        data_files={'test': '/home/y50063564/DREAM-S/eval/data/mathvista_testmini.parquet'},
        split="test"
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "DocVQA":
    ds = load_dataset(
        "parquet", 
        data_files={'test': 'datasets/test/docvqa/DocVQA/test-00000-of-00006.parquet'},
        split="test"
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "OCRBench":
    ds = load_dataset(
        "parquet", 
        data_files={'test': 'datasets/test/ocr/data/test-00000-of-00011.parquet'},
        split="test"
    )
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(sample_num))

elif args.dataset == "seed":
    with open("/SEED-Bench_v2_level1_2_3.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    print(records["questions"][0])
    ds = records["questions"]


def load_bs64_image(base64_data):
    base64_data = base64_data.replace("\n", "")
    missing_padding = len(base64_data) % 4
    if missing_padding:
        base64_data += "=" * (4 - missing_padding)

    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))
    return image

def load_image(image):
    img_file = io.BytesIO(image)
    img = Image.open(img_file)
    return img

def load_image_path(base_path, image_path):
    try:
        img = Image.open(f"{base_path}/{image_path}")
    except Exception as e:
        img = None 
    return img

os.makedirs("/home/y50063564/DREAM-S/eval/data", exist_ok=True)
answer_file = f"/home/y50063564/DREAM-S/eval/data/{args.dataset}-{args.version}-top{args.topk}-d{args.depth}-total{args.total_token}-temp{args.temperature}.jsonl"
speed_up_avg = []
ar = []
sd = []
adl = []
accept_avg = []
temperature = args.temperature
top_p = 0.6
for i in range(sample_num):
    TRACE.clear()
    if args.dataset == "MMT-Bench":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds["question"][i]}
            ]}
        ]
        imagebase64 = ds["image"][i]
        image = load_bs64_image(imagebase64)
    elif args.dataset == "ScienceQA":
        imagebase64 = ds[i]["image"]["bytes"]
        image = load_image(imagebase64)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds["question"][i]}
            ]}
        ]
    elif args.dataset == "seed":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds[i]["question"]}
            ]}
        ]
        image = load_image_path("datasets/test/seed2/cc3m-image",ds[i]["data_id"])
    elif args.dataset == "ChartQA":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds[i]["question"]}
            ]}
        ]
        image = ds[i]["image"]
        if not isinstance(image, Image.Image):
            image = load_image(image)  # 若为 bytes 再转
    elif args.dataset == "MathVista":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds[i]["question"]}
            ]}
        ]
        image = ds[i]["decoded_image"]   # NPU 适配: 用 parquet 内嵌图片
    elif args.dataset == "DocVQA":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds["question"][i]}
            ]}
        ]
        image = ds["image"][i]
    elif args.dataset == "OCRBench":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ds["question"][i]}
            ]}
        ]
        image = ds["image"][i]

    # 修复: prompt/inputs 计算对所有数据集生效（原缩进在 OCRBench 分支内是 bug）
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = model.processor(images=image, text=prompt, truncation=True, return_tensors="pt").to(model.base_model.device)

    input_ids = inputs.input_ids
    input_ids = torch.as_tensor(input_ids).to("npu:0")   # NPU 适配
    input_len = input_ids.shape[1]
    naive_text = []
    cu_len = input_len
    totaltime=0
    start_time=time.time()
    total_ids=0

    start = time.time()
    for output_ids in model.naive_generate(inputs, temperature=temperature, top_p=top_p,
                                        max_new_tokens=args.max_new_token, max_length=args.max_length,
                                        is_llama3=args.model_type=="llama-3-instruct"):
        totaltime += (time.time() - start_time)
        total_ids+=1
        decode_ids = output_ids[0, input_len:].tolist()
        decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
        text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                        clean_up_tokenization_spaces=True, )
        cu_len = output_ids.shape[1]
        start_time = time.time()
    ar_time = totaltime


    totaltime=0
    start_time=time.time()
    total_ids=0
    cu_len=input_len
    ea_text=[]
    oo = 0

    final_target_score = None
    final_all_attention_scores = None

    for output_id, target_score, all_attention_scores in model.ea_generate(inputs, temperature=temperature, top_p=top_p,
                                        max_new_tokens=args.max_new_token, max_length=args.max_length,
                                        is_llama3=args.model_type=="llama-3-instruct", output_attention_scores=True):
        totaltime+=(time.time()-start_time)
        total_ids+=1
        decode_ids = output_id[0, input_len:].tolist()
        decode_ids = truncate_list(decode_ids, model.tokenizer.eos_token_id)
        if args.model_type == "llama-3-instruct":
            decode_ids = truncate_list(decode_ids, model.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        text = model.tokenizer.decode(decode_ids, skip_special_tokens=True, spaces_between_special_tokens=False,
                                        clean_up_tokenization_spaces=True, )

        cu_len = output_id.shape[1]
        new_tokens = cu_len - input_len
        start_time = time.time()

        final_target_score = target_score
        final_all_attention_scores = all_attention_scores

    sd_time = totaltime
    decoded_output = text

    # ==== COMPLETE DUMP (注入) ====
    dump = []
    if TRACE and i == QID:
        al_list = [int(t.get("al", -1)) for t in TRACE]
        print(f"### TRACE: yields={total_ids} new_tokens={new_tokens} evalAL={new_tokens/total_ids:.3f}", flush=True)
        for si, t in enumerate(TRACE):
            dt = t["draft"]; ri = t["ri"]
            dtt = dt[0].tolist() if dt is not None else []
            bc = int(t.get("bc", -1)); al = int(t.get("al", -1))
            # 完整候选路径: 每个叶子 = retrieve_indices 的一行 -> 该叶子的 token 路径
            cand_paths = []   # [[token ids], ...]
            if ri is not None:
                for leaf in range(ri.shape[0]):
                    path = [dtt[n] if 0 <= n < len(dtt) else -1 for n in ri[leaf].tolist()]
                    cand_paths.append(path)
            # 被采纳路径
            acc_path_ids = []
            if al >= 0 and ri is not None and bc >= 0 and bc < ri.shape[0]:
                acc_path_ids = cand_paths[bc][:al+1]
            # 整棵树的去重 token 集
            tree_tokens = sorted(set(x for x in dtt if x >= 0))
            step_dump = {
                "step": si,
                "accept_length": al,
                "best_candidate": bc,
                "num_leaves": ri.shape[0] if ri is not None else 0,
                "max_depth": ri.shape[1] if ri is not None else 0,
                "draft_tokens_all": dtt,
                "tree_token_set": tree_tokens,
                "candidate_paths": cand_paths,
                "accepted_path_tokens": acc_path_ids,
            }
            dump.append(step_dump)
            print(f"[步{si}] accept={al} 叶数={step_dump['num_leaves']} 深={step_dump['max_depth']} 采纳路径={model.tokenizer.decode(acc_path_ids, skip_special_tokens=True)!r}", flush=True)
        dump_file = f"/home/y50063564/DREAM-S/output/spec_dump_qid{QID}.json"
        with open(dump_file, "w") as f:
            json.dump({"question_id": i, "input_len": input_len,
                       "decoded_output": decoded_output,
                       "new_tokens": new_tokens, "total_ids": total_ids,
                       "al_reported": new_tokens/total_ids,
                       "steps": dump}, f, ensure_ascii=False, indent=1)
        print(f"### 完整 dump 已保存: {dump_file} ({len(dump)} 步)", flush=True)
    # ==== COMPLETE DUMP END ====


    record = {
        "question_id": i,
        "decoded_output": decoded_output,
        # "accept_length_list": accept_length_list,
        "average_accept_length": f"{new_tokens/total_ids:.2f}",
        "speedup": ar_time / sd_time
    }

    print('record: ', record, flush=True)
    with open(answer_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    speed_up_avg.append(ar_time / sd_time)
    ar.append(ar_time)
    sd.append(sd_time)
    adl.append(new_tokens/total_ids)
    # accept_avg.append(avg_accept_length)
print('average speed up: ', sum(ar)/sum(sd))
print('max speedup: ', max(speed_up_avg))
print(f'average draft length: {sum(adl)/len(adl):.2f}, max draft length: {max(adl):.2f}')