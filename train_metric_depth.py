# train_metric_depth.py
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb                                           # ← 新增

from vggt.models.vggt import VGGT
from metric_adapter import MetricDepthAdapter
from dataset_nyu import NYUv2Dataset

CONFIG = {
    "mat_path": "/root/autodl-tmp/nyu_depth_v2_labeled_baidu.mat",
    "img_size": 518,
    "max_depth": 10.0,
    "batch_size": 4,
    "num_epochs": 30,
    "lr_adapter":    5e-3,                             # ← 改：adapter 学习率
    "lr_depth_head": 1e-7,                             # ← 新增：depth_head 用更小的 lr
    "weight_decay": 1e-4,
    "save_dir": "./checkpoints",
    "log_interval": 20,
    "val_interval": 1,
    "wandb_project": "vggt-metric-depth",              # ← 新增
    "wandb_run_name": "depth-head-finetune-v1",        # ← 新增
}


def is_ddp():
    return "RANK" in os.environ

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process():
    if is_ddp():
        return dist.get_rank() == 0
    return True


def silog_loss(pred, target, valid_mask, lamda=0.85):
    pred_v   = pred[valid_mask].clamp(min=1e-4)
    target_v = target[valid_mask].clamp(min=1e-4)
    log_diff = torch.log(pred_v) - torch.log(target_v)
    return torch.sqrt((log_diff**2).mean() - lamda * (log_diff.mean()**2) + 1e-8)

def l1_loss(pred, target, valid_mask):
    return F.l1_loss(pred[valid_mask], target[valid_mask])

def compute_metrics(pred, target, valid_mask):
    p = pred[valid_mask].clamp(min=1e-4)
    t = target[valid_mask].clamp(min=1e-4)
    thresh = torch.max(p / t, t / p)
    return {
        "d1":      (thresh < 1.25).float().mean().item(),
        "d2":      (thresh < 1.25**2).float().mean().item(),
        "abs_rel": ((p - t).abs() / t).mean().item(),
        "rmse":    torch.sqrt(((p - t)**2).mean()).item(),
    }


def main():
    use_ddp = is_ddp()
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")

    dtype = torch.bfloat16

    # ── wandb 初始化（只在主进程）──────────────────── ← 新增
    if is_main_process():
        wandb.init(
            project=CONFIG["wandb_project"],
            name=CONFIG["wandb_run_name"],
            config=CONFIG,
        )

    # ── 1. 加载 VGGT，全部冻结，然后单独解冻 depth_head ──── ← 改
    if is_main_process():
        print("Loading VGGT...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).to(dtype)
    model.eval()

    for p in model.parameters():                       # 先全冻结
        p.requires_grad = False

    for p in model.depth_head.parameters():            # ← 新增：只解冻 depth_head
        p.requires_grad = True
    model.depth_head.train()                           # ← 新增：切训练模式

    if is_main_process():
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total:     {total/1e6:.1f}M params")
        print(f"Trainable: {trainable/1e6:.1f}M params (depth_head only)")

    # ── 2. Adapter ──
    adapter = MetricDepthAdapter(max_depth=CONFIG["max_depth"]).to(device)

    # ── 3. DDP：model 和 adapter 都要包装 ──────────── ← 改
    if use_ddp:
        model_ddp   = DDP(model,   device_ids=[local_rank], find_unused_parameters=True)
        adapter_fwd = DDP(adapter, device_ids=[local_rank])
    else:
        model_ddp   = model
        adapter_fwd = adapter

    # ── 4. 数据集 ──
    train_dataset = NYUv2Dataset(CONFIG["mat_path"], split="train", img_size=CONFIG["img_size"])
    val_dataset   = NYUv2Dataset(CONFIG["mat_path"], split="val",   img_size=CONFIG["img_size"])

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── 5. 分组学习率优化器 ────────────────────────── ← 改
    optimizer = torch.optim.AdamW([
        {"params": model.depth_head.parameters(), "lr": CONFIG["lr_depth_head"]},
        {"params": adapter.parameters(),           "lr": CONFIG["lr_adapter"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    best_abs_rel = float("inf")
    global_step  = 0                                   # ← 新增：wandb x 轴

    for epoch in range(CONFIG["num_epochs"]):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.depth_head.train()                       # ← 新增：每 epoch 重置训练模式
        adapter_fwd.train()
        total_loss  = 0.0
        total_silog = 0.0                              # ← 新增
        total_l1    = 0.0                              # ← 新增

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())
        ):
            images   = batch["image"].unsqueeze(1).to(device).to(dtype)
            gt_depth = batch["depth"].to(device).float()

            # ── aggregator 不需要梯度，depth_head 需要 ── ← 改
            with torch.cuda.amp.autocast(dtype=dtype):
                with torch.no_grad():                  # aggregator 冻结，不存梯度
                    agg_tokens, ps_idx = (
                        model_ddp.module.aggregator(images)
                        if use_ddp else model.aggregator(images)
                    )
                depth_map, _ = (                       # depth_head 解冻，保留梯度
                    model_ddp.module.depth_head(agg_tokens, images, ps_idx)
                    if use_ddp else model.depth_head(agg_tokens, images, ps_idx)
                )

            affine_depth = depth_map.squeeze(-1).float()
            metric_depth = adapter_fwd(affine_depth)

            pred_flat   = metric_depth.reshape(-1)
            target_flat = gt_depth.reshape(-1)
            mask_flat   = (target_flat > 0.1) & (target_flat < CONFIG["max_depth"])

            if mask_flat.sum() == 0:
                continue

            loss_silog = silog_loss(pred_flat, target_flat, mask_flat)
            loss_l1    = l1_loss(pred_flat, target_flat, mask_flat)
            loss       = loss_silog + 0.1 * loss_l1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.depth_head.parameters(), 1.0)  # ← 新增
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_silog += loss_silog.item()           # ← 新增
            total_l1    += loss_l1.item()              # ← 新增
            global_step += 1                           # ← 新增

            if is_main_process() and (step + 1) % CONFIG["log_interval"] == 0:
                print(f"  step {step+1}: loss={loss.item():.4f} "
                      f"(silog={loss_silog.item():.4f}, l1={loss_l1.item():.4f})")
                # ── wandb step 级别记录 ──────────────── ← 新增
                wandb.log({
                    "train/loss_step":       loss.item(),
                    "train/silog_step":      loss_silog.item(),
                    "train/l1_step":         loss_l1.item(),
                    "adapter/scale":         adapter.log_scale.exp().item(),
                    "adapter/shift":         adapter.shift.item(),
                    "train/depth_head_grad": sum(
                        p.grad.norm().item() ** 2
                        for p in model.depth_head.parameters()
                        if p.grad is not None
                    ) ** 0.5,
                }, step=global_step)

        scheduler.step()

        if is_main_process():
            n = len(train_loader)
            print(f"Epoch {epoch+1} avg loss: {total_loss/n:.4f}")
            # ── wandb epoch 级别记录 ─────────────────── ← 新增
            wandb.log({
                "train/loss_epoch":      total_loss  / n,
                "train/silog_epoch":     total_silog / n,
                "train/l1_epoch":        total_l1    / n,
                "train/lr_depth_head":   optimizer.param_groups[0]["lr"],
                "train/lr_adapter":      optimizer.param_groups[1]["lr"],
                "epoch": epoch + 1,
            }, step=global_step)

        # ── 验证 ──
        if is_main_process() and (epoch + 1) % CONFIG["val_interval"] == 0:
            model.depth_head.eval()                    # ← 新增：切回 eval
            adapter.eval()
            all_metrics = {"d1": [], "d2": [], "abs_rel": [], "rmse": []}

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    images   = batch["image"].unsqueeze(1).to(device).to(dtype)
                    gt_depth = batch["depth"].to(device).float()

                    with torch.cuda.amp.autocast(dtype=dtype):
                        agg_tokens, ps_idx = model.aggregator(images)
                        depth_map, _       = model.depth_head(agg_tokens, images, ps_idx)

                    affine_depth = depth_map.squeeze(-1).float()
                    metric_depth = adapter(affine_depth)

                    pred_flat   = metric_depth.reshape(-1)
                    target_flat = gt_depth.reshape(-1)
                    mask_flat   = (target_flat > 0.1) & (target_flat < CONFIG["max_depth"])

                    if mask_flat.sum() == 0:
                        continue

                    m = compute_metrics(pred_flat, target_flat, mask_flat)
                    for k, v in m.items():
                        all_metrics[k].append(v)

            avg = {k: sum(v)/len(v) for k, v in all_metrics.items()}
            print(f"\n📊 Val (Epoch {epoch+1}): "
                  f"AbsRel={avg['abs_rel']:.4f}  RMSE={avg['rmse']:.4f}  "
                  f"δ1={avg['d1']:.4f}  δ2={avg['d2']:.4f}")

            # ── wandb 验证指标 ───────────────────────── ← 新增
            wandb.log({
                "val/abs_rel": avg["abs_rel"],
                "val/rmse":    avg["rmse"],
                "val/d1":      avg["d1"],
                "val/d2":      avg["d2"],
            }, step=global_step)

            if avg["abs_rel"] < best_abs_rel:
                best_abs_rel = avg["abs_rel"]
                torch.save({
                    "epoch":            epoch + 1,
                    "depth_head_state": model.depth_head.state_dict(),  # ← 新增
                    "adapter_state":    adapter.state_dict(),
                    "metrics":          avg,
                }, os.path.join(CONFIG["save_dir"], "best_ckpt.pth"))
                print(f"   ✅ Saved best (AbsRel={best_abs_rel:.4f})")
                wandb.run.summary["best_abs_rel"] = best_abs_rel        # ← 新增
                wandb.run.summary["best_epoch"]   = epoch + 1           # ← 新增

        if use_ddp:
            dist.barrier()

    if is_main_process():
        wandb.finish()                                 # ← 新增

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()