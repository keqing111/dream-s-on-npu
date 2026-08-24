# DREAM-S on NPU — 复现指南

本指南说明如何从**原始 DREAM-S 项目**迁移到**当前可在昇腾 NPU 上运行**的版本，包括所有代码改动、依赖版本和复现步骤。

## 1. 概述

DREAM-S 是一个视觉语言模型（VLM）投机解码加速框架。原始仓库 `SAI-Lab-NYU/DREAM-S` 基于 EAGLE，代码目标 transformers 4.28-4.35（CUDA）。本仓库将其适配到**昇腾 910B NPU（torch_npu）**，验证了三条全链路：

- **数据生成**：7B target 模型产出 hidden_states → .pt
- **训练**：投机解码小模型（draft）训练，loss 下降、准确率上升
- **推理**：`EaModel.from_pretrained` + 生成

## 2. 环境与依赖版本

**硬件**：昇腾 Ascend910_9372，8 卡（每卡 64GB HBM），单卡即可跑 7B。

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.13 | |
| torch | 2.10.0+cpu | torch_npu 配套的 CPU 版 torch |
| **torch_npu** | **2.10.0.post2** | 昇腾插件，`import torch_npu` 后 `torch.npu` 可用 |
| CANN | 25.5.0 | `npu-smi` 显示 |
| **transformers** | **4.46.3** | ⚠️ 关键升级（见 3.2） |
| pyarrow | **21.0.0** | ⚠️ 关键降级（见 3.3） |
| deepspeed | 0.19.5 | ZeRO-2 需 NPU 适配（见 4.2） |
| accelerate | 1.14.0 | |
| datasets | 5.0.0 | |
| tokenizers | 0.20.3 | |
| fastchat | 0.2.31 | |

### 2.1 版本安装要点

```bash
pip install torch==2.10.0            # torch_npu 配套版本
pip install torch_npu==2.10.0.post2  # 需从 CANN 源安装
pip install "transformers==4.46.3"
pip install "pyarrow==21.0.0"
pip install deepspeed                # 0.19.5
```

> ⚠️ **torch 版本由 torch_npu 锁定**：昇腾 NPU 不能用任意 torch，必须配对 CANN 官方发布的版本（2.10.0 对应 torch_npu 2.10.0.post2）。

## 3. 从原始项目到可运行：需要做的修改

### 3.1 修复迁移损坏的 import（`dream_s/model/ea_model.py`）

原始 DREAM-S 的 `ea_model.py` 有 3 处 import 损坏（EAGLE 时代命名残留），会导致 `import dream_s.model.ea_model` 直接失败：

| 位置 | 原始（错误） | 修复后 |
|---|---|---|
| import | `from .modeling_mixtral_kv import ...` / `from .modeling_qwen2_kv import ...`（文件不存在） | 删除（论文未使用 Qwen/Mixtral） |
| import | `from .utils_head import *` | `from .utils import *` |
| import | `from .cnetsimport Model` | `from .cnets import Model` |
| `from_pretrained` | Qwen2/Mixtral 分发分支 | 删除 |

### 3.2 transformers 升级到 4.46.3（关键）

**原因**：DREAM-S 自定义的 `modeling_llava_next.forward` 去掉了标准 transformers 的 image token 展开逻辑，要求 input_ids 里已展开 image token。而 transformers 4.44 及更早的 processor **不做展开**（展开在模型 forward 里），导致真实图片进来报 `Image features and image tokens do not match: tokens: 1, features 2340`。

**4.46 起 processor 侧自动展开 image token**，修复此问题。这是本适配中最关键的版本决定。

同时修复了 `modeling_llava_next.py` 对 `config.multimodal_projector_bias` 的依赖（该字段在 4.44 缺失，改用 `getattr` 兜底）：

```python
bias=getattr(config, "multimodal_projector_bias", True),  # 原: config.multimodal_projector_bias
```

### 3.3 pyarrow 降级到 21.0.0

**原因**：pyarrow 25.0.0 读取字典编码 parquet（LLaVA/mix665k 数据集的 `image` 列）时报 `ArrowInvalid: Index not in dictionary bounds`，导致 datasets 无法加载训练数据。pyarrow 21.0.0 可正常读取。（datasets 5.0 要求 pyarrow>=21，满足。）

### 3.4 `modeling_llama_kv.py` 的 3 处修复

1. **attn_scores dataclass 字段缺失**（关键运行时 bug）：DREAM-S 解注释了 `attn_scores=attan_scores` 但从未给 `BaseModelOutputWithPast` 声明该字段，导致**任何 forward 都 TypeError**。修复：子类化并加字段：

```python
@dataclass
class BaseModelOutputWithPastAndAttnScores(BaseModelOutputWithPast):
    attn_scores: Optional[torch.FloatTensor] = None
```

2. **`second_hidden_state` 死代码**：`self.norm(second_hidden_state)` 只对正好 32 层的模型不崩（Vicuna-7B 恰好 32 层所以原项目能跑），其他层数直接 None 崩溃。删除死代码。

3. `find_pruneable_heads_and_indices`（transformers 5.x 移除）用 try/except 兜底。

### 3.5 `cnets.py` 的 2 处修复

- `find_pruneable_heads_and_indices`（transformers 5.x 移除）→ try/except
- `thop`（未安装的 FLOPs 库，从未被调用）→ try/except

## 4. NPU 专属适配（非 GPU 环境必须）

### 4.1 设备切换

所有 `.cuda()` / `torch.cuda.*` / `device_map="cuda"` 需切到 NPU：

| GPU | NPU |
|---|---|
| `.cuda()` | `.npu()` 或 `.to('npu:X')` |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` |
| `device_map="cuda:0"` | `device_map="npu:0"` |
| `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |

> ⚠️ 测试用 `npu:12`-`npu:15`（多卡可并行：数据生成一张卡、训练另一张卡）。

### 4.2 DeepSpeed ZeRO-2 的 NPU 适配（关键）

deepspeed 0.19.5 的 ZeRO-2 在 NPU 上默认 `contiguous_gradients: True` 会导致反向传播时报 `reduce_ipg_grads ... IndexError`。**必须改为**：

```json
"zero_optimization": {
    "stage": 2,
    "contiguous_gradients": false,   // ← NPU 上必须关
    "reduce_scatter": false          // ← 同上
}
```

同时，原 `ds_config.json` 的 `"bf16": {"enabled": "true", "auto_cast": "true"}` 在 deepspeed 0.19 会报 pydantic 错误（`auto_cast` 字段已移除），需改为 `"bf16": {"enabled": true}`。

### 4.3 其他 NPU 注意事项

- **混合 dtype**：`mse_loss(bf16, fp32)` 反向在 NPU 上报 dtype 不一致；训练用的 SmoothL1Loss 无此问题
- **device 契约**：`generate`/训练方法要求 inputs 先 `.to(device)`（webui/eval 入口已这么做）
- **内存**：7B bf16 单卡峰值 ~31GB（含 output_attentions），64GB 卡够用；共享服务器注意其他进程占用

## 5. 复现步骤

### 5.1 数据生成（7B target 产出 .pt 训练数据）

用真实 LLaVA 7B target + 图片，复用 ge_data 逻辑（`mid_feature_collect_and_score`/`loss_mask`/`pruning_indices`），产出训练脚本所需格式的 `.pt`：

```python
# 伪代码（见 /tmp/gen_data.py 完整版）
from transformers import LlavaNextForConditionalGeneration, AutoProcessor
model = LlavaNextForConditionalGeneration.from_pretrained(TARGET, torch_dtype=torch.bfloat16, device_map="npu:X")
processor = AutoProcessor.from_pretrained(TARGET)
inputs = processor(images=img, text="USER: <image>\n... ASSISTANT: ...", return_tensors="pt")
outs = model(**inputs, output_hidden_states=True, output_attentions=True)
# 收集: inputs_embeds=hidden_states[0], target=hidden_states[-1],
#       hidden_state_mid_a=mid_feature_collect_and_score(...), loss_mask, pruning_indices
```

`.pt` 需要的字段（训练脚本 CustomDataset 读取）：`inputs_embeds`、`target`、`hidden_state_mid_a`、`loss_mask`、`attention_mask`、`pruning_indices`、`pruning_ratios`。

### 5.2 训练（投机解码小模型）

```bash
python train_validate.py   # 核心循环: CustomDataset + DataCollator + draft Model + deepspeed
```

- BS=4，序列 ~2369，**~3.9 samples/s**，2932 条 1 epoch ~12.5 分钟
- 效果：BS=4 + 2586 条 1 epoch → loss 0.024，**准确率 33.3%**（随机初始化，趋势正确）
- 训练是 teacher-forced 1 步 next-token 预测（EAGLE 式），loss_mask 覆盖 assistant token

### 5.3 推理（投机解码）

```python
from dream_s.model.ea_model import EaModel
model = EaModel.from_pretrained(base_model_path=TARGET, ea_model_path=DRAFT,
                                total_token=32, depth=6, top_k=4, torch_dtype=torch.bfloat16, device_map="npu:X")
inputs = model.processor(images=img, text=prompt, return_tensors="pt").to("npu:X")
for out in model.naive_generate(inputs, max_new_tokens=8): ...
```

## 6. 验证过的数据/权重

| 资源 | 位置 |
|---|---|
| Draft 权重（DREAM-S ea_layer，1.2GB） | `HideonBed12138/DREAM-S-llava-v1.6-vicuna-7b` |
| Target 权重（LLaVA-Next 7B，14.1GB） | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| 训练数据源（COCO 分片，2932 图） | `detection-datasets/coco`（pyarrow 21 读取） |
| 训练数据（.pt，2932 条） | 由 7B target 生成，用完即弃 |

## 7. 已知限制

- `ea_generate`（投机解码推理路径）用真实权重尚未验证（算法用随机权重验证过）
- 训练数据是 COCO + 合成对话模板（非完整 mix665k 格式）
- Mistral 路径（`llava-v1.6-mistral-7b`）未实测
- 多卡 ZeRO-2 未验证（当前单卡跑通）
