"""训练流程验证：真实数据 + 真实模型 + deepspeed，检查 loss 和准确率（NPU 13）。"""
import os
import sys
import glob
import torch
import torch_npu
import torch.nn as nn
import deepspeed
import traceback
sys.path.insert(0, '/home/y50063564/DREAM-S')
sys.path.insert(0, '/home/y50063564/DREAM-S/dream_s/model')

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29600'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '13'

MAX_LEN = 5120
BS = 4
EPOCHS = 1
N_FILES = 200000

def main():
    from cnets import Model
    from configs import EConfig

    print("加载 head (lm_head) ...")
    import json
    from safetensors import safe_open
    basepath = "/home/y50063564/DREAM-S/llava-v1.6-vicuna-7b-hf"
    with open(os.path.join(basepath, "model.safetensors.index.json")) as f:
        index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["language_model.lm_head.weight"]
    with safe_open(os.path.join(basepath, head_path), framework="pt", device="cpu") as f:
        tensor = f.get_slice("language_model.lm_head.weight")[:, :].float()
    head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
    head.weight.data = tensor

    print("加载训练数据 ...")
    files = sorted(glob.glob("/tmp/train_data/*.ckpt"))[:N_FILES]
    print(f"  使用 {len(files)} 条")
    dataset = CustomDataset(files)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=True,
                                         collate_fn=DataCollatorWithPadding(), num_workers=0)

    print("构建 draft (cnets.Model) + deepspeed ...")
    config = EConfig.from_pretrained("/home/y50063564/DREAM-S/dream_s/train/vicuna_7B_config.json")
    model = Model(config, path=basepath, load_emb=True)
    criterion = nn.SmoothL1Loss(reduction="none")

    ds_cfg = {
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 5e-5, "weight_decay": 0.0, "adam_w_mode": True, "betas": [0.9, 0.95]}},
        "scheduler": {"type": "WarmupDecayLR", "params": {"warmup_min_lr": 0, "warmup_max_lr": 5e-5, "warmup_num_steps": 2, "total_num_steps": 100}},
        "zero_optimization": {"stage": 2, "contiguous_gradients": False, "reduce_scatter": False},
        "gradient_accumulation_steps": 1, "gradient_clipping": 0.5, "train_micro_batch_size_per_gpu": BS,
    }

    deepspeed.init_distributed()
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_cfg)
    head_engine, _, _, _ = deepspeed.initialize(model=head, model_parameters=head.parameters(), config=ds_cfg)
    for p in head.parameters():
        p.requires_grad = False

    print("=== 训练循环 ===")
    import time as _time
    t0 = _time.time()
    for epoch in range(EPOCHS):
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        for bi, data in enumerate(loader):
            model_engine.zero_grad()
            rank = torch.distributed.get_rank()
            inputs_embeds = torch.autograd.Variable(data["inputs_embeds"].to(torch.bfloat16).to(rank), requires_grad=True)
            last_hidden_state = torch.autograd.Variable(data["hidden_states"].to(torch.bfloat16).to(rank), requires_grad=True)
            predict, all_hidden_states = model_engine(
                last_hidden_state.to(rank), inputs_embeds=inputs_embeds.to(rank),
                attention_mask=data["attention_mask"].to(rank), output_hidden_states=True, last_head_ratio=1)
            mid_predict = all_hidden_states[-2]
            with torch.no_grad():
                target_head = head_engine(data["target"].to(rank).to(torch.bfloat16))
                target_p = nn.Softmax(dim=2)(target_head).detach()
            loss_mask = data["loss_mask"][:, :, None].to(rank)
            vloss, ploss, out_head = compute_loss(head_engine, criterion, data["target"], target_p, predict, loss_mask)
            mid_vloss = compute_mid_loss(criterion, data["hidden_states_mid"], mid_predict, loss_mask)
            loss = 0.1*vloss + 1.0*ploss + mid_vloss
            model_engine.backward(loss)
            model_engine.step()

            with torch.no_grad():
                _, pred = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((pred == target) * loss_mask.squeeze()).sum().item()
            total_loss += loss.item(); total_correct += cc; total_tokens += ct
            if bi % 10 == 0:
                dt = _time.time() - t0
                print(f"  epoch{epoch} batch{bi}: vloss={vloss.item():.4f} ploss={ploss.item():.4f} "
                      f"mid={mid_vloss.item():.4f} loss={loss.item():.4f} acc={cc/max(ct,1):.4f} "
                      f"{(bi+1+epoch*len(loader))/dt:.2f} batch/s", flush=True)
        print(f"epoch{epoch} 完成: avg_loss={total_loss/max(bi+1,1):.4f} acc={total_correct/max(total_tokens,1):.4f} "
              f"用时 {(_time.time()-t0)/60:.1f} 分钟", flush=True)
    print(f"=== 训练验证完成, 总耗时 {(_time.time()-t0)/60:.1f} 分钟 ===")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.load(self.data[index], weights_only=False)
        new_data = {}
        hidden_state = data['target'][:MAX_LEN]
        inputs_embeds = data['inputs_embeds'][:MAX_LEN]
        hidden_state_mid = data['hidden_state_mid_a'][:MAX_LEN]
        loss_mask = data["loss_mask"][:MAX_LEN]
        indices = data["pruning_indices"]
        ratios = data["pruning_ratios"]
        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        target = hidden_state
        hidden_state = hidden_state[:,:-1,:]
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["hidden_state_mid"] = hidden_state_mid
        new_data["inputs_embeds"] = inputs_embeds
        new_data["ratios"] = ratios
        new_data["indices"] = indices
        return new_data

class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        return torch.cat((intensors, torch.zeros(B, N - n, S)), dim=1)
    def __call__(self, features):
        max_length = max(item['hidden_state_mid'].shape[1] for item in features)
        batch_inputs_embeds = torch.cat([self.paddingtensor(item['inputs_embeds'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length-1) for item in features])
        batch_hidden_states_mid = torch.cat([self.paddingtensor(item['hidden_state_mid'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor([item['loss_mask'] + [0]*(max_length-len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor([item['attention_mask'] + [0]*(max_length-len(item['attention_mask'])) for item in features])
        return {
            "inputs_embeds": batch_inputs_embeds,
            "hidden_states": batch_hidden_states,
            "hidden_states_mid": batch_hidden_states_mid,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "indices": [item["indices"] for item in features],
            "ratios": features[0]["ratios"],
            "max_length": max_length,
        }

def compute_loss(head_engine, criterion, target, target_p, predict, loss_mask, rank=0):
    out_head = head_engine(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0]*loss_mask.shape[1] + 1e-5)
    vloss = criterion(predict, target.to(rank))
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0]*loss_mask.shape[1] + 1e-5)
    return vloss, ploss, out_head

def compute_mid_loss(criterion, target, predict, loss_mask, rank=0):
    vloss = criterion(predict, target.to(rank))
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0]*loss_mask.shape[1] + 1e-5)
    return vloss

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
