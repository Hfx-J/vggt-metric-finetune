# train_scale_depth.py
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

from vggt.models.vggt import VGGT
from vggt.models.vggt_scale import VGGTScaleDepth
from dataset_nyu import NYUv2Dataset

CONFIG = {
    "mat_path":           "/root/autodl-tmp/nyu_depth_v2_labeled_baidu.mat",
    "img_size":           518,
    "max_depth":          10.0,
    "batch_size":         4,            # depth_head 微调显存开销大，从 2 开始
    "num_epochs":         50,
    "lr_new":             1e-3,         # scale_token / scale_mlp / fuse_conv
    "lr_depth_head":      1e-6,         # depth_head 微调学习率（比新增模块低 100x）
    "weight_decay":       1e-4,
    "freeze_depth_head":  False,        # ← True = 冻结 depth_head；False = 微调
    "save_dir":           "./checkpoints_scale",
    "log_interval":       20,
    "val_interval":       1,
    "scale_per_frame":    False,
    "lambda_l1":          0.1,
    "wandb_project":      "vggt-scale-depth",
    "wandb_run_name":     "scale-token-dpt-finetune-v1",
}


# ── DDP helpers ───────────────────────────────────────────────────────────────

def is_ddp():
    return "RANK" in os.environ

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main():
    return dist.get_rank() == 0 if is_ddp() else True


# ── Loss & metrics ────────────────────────────────────────────────────────────

def silog_loss(pred, target, mask, lamda=0.85):
    p = pred[mask].clamp(min=1e-4)
    t = target[mask].clamp(min=1e-4)
    d = torch.log(p) - torch.log(t)
    return torch.sqrt((d ** 2).mean() - lamda * (d.mean() ** 2) + 1e-8)

def l1_loss(pred, target, mask):
    return F.l1_loss(pred[mask], target[mask])

def compute_metrics(pred, target, mask):
    p = pred[mask].clamp(min=1e-4)
    t = target[mask].clamp(min=1e-4)
    r = torch.max(p / t, t / p)
    return {
        "d1":      (r < 1.25).float().mean().item(),
        "d2":      (r < 1.25 ** 2).float().mean().item(),
        "abs_rel": ((p - t).abs() / t).mean().item(),
        "rmse":    torch.sqrt(((p - t) ** 2).mean()).item(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    use_ddp = is_ddp()
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")

    dtype = torch.bfloat16

    if is_main():
        wandb.init(
            project=CONFIG["wandb_project"],
            name=CONFIG["wandb_run_name"],
            config=CONFIG,
        )

    # ── 1. 加载预训练 VGGT，构建 VGGTScaleDepth ──────────────────────────────
    if is_main():
        print("Loading pretrained VGGT...")
    pretrained = VGGT.from_pretrained("facebook/VGGT-1B").to(device).to(dtype)

    if is_main():
        print(f"Building VGGTScaleDepth "
              f"(freeze_depth_head={CONFIG['freeze_depth_head']})...")
    model = VGGTScaleDepth(
        pretrained_vggt=pretrained,
        embed_dim=1024,
        freeze_backbone=True,
        freeze_depth_head=CONFIG["freeze_depth_head"],   # ← 新增
        scale_per_frame=CONFIG["scale_per_frame"],
        img_size=CONFIG["img_size"],
    ).to(device)

    del pretrained

    # ── dtype 设置 ─────────────────────────────────────────────────────────
    # aggregator      → bfloat16（推理加速）
    # depth_head      → float32（LayerNorm 需要；微调时梯度精度更高）
    # scale_mlp/fuse  → float32
    model.aggregator.to(dtype)
    model.depth_head.to(torch.float32)
    model.scale_mlp.to(torch.float32)
    model.fuse_conv.to(torch.float32)

    if is_main():
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total:     {total / 1e6:.1f}M params")
        print(f"Trainable: {trainable / 1e6:.3f}M params")
        for g in (model.module if use_ddp else model).trainable_parameter_groups():
            n = sum(p.numel() for p in g["params"])
            print(f"  {g['name']:20s}: {n / 1e6:.3f}M  lr={g['lr']}")

    # ── 2. DDP ───────────────────────────────────────────────────────────────
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if use_ddp else model

    # ── 3. 数据集 ────────────────────────────────────────────────────────────
    train_dataset = NYUv2Dataset(
        CONFIG["mat_path"], split="train", img_size=CONFIG["img_size"]
    )
    val_dataset = NYUv2Dataset(
        CONFIG["mat_path"], split="val", img_size=CONFIG["img_size"]
    )

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        sampler=val_sampler, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── 4. 优化器 ─────────────────────────────────────────────────────────────
    # trainable_parameter_groups() 会根据 freeze_depth_head 自动决定
    # 是否把 depth_head 参数加入优化器，并使用 lr_depth_head 的学习率。
    optimizer = torch.optim.AdamW(
        raw_model.trainable_parameter_groups(
            depth_head_lr=CONFIG["lr_depth_head"]
        ),
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    best_abs_rel = float("inf")
    global_step  = 0

    for epoch in range(CONFIG["num_epochs"]):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss  = 0.0
        total_silog = 0.0
        total_l1    = 0.0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not is_main())
        ):
            images   = batch["image"].unsqueeze(1).to(device).to(dtype)  # [B,1,3,H,W]
            gt_depth = batch["depth"].to(device).float()                  # [B,H,W]

            with torch.cuda.amp.autocast(dtype=dtype):
                out = model(images)

            pred_depth = out["depth"].squeeze(1).squeeze(-1).float()

            pred_flat   = pred_depth.reshape(-1)
            target_flat = gt_depth.reshape(-1)
            mask_flat   = (target_flat > 0.1) & (target_flat < CONFIG["max_depth"])

            if mask_flat.sum() == 0:
                continue

            loss_silog = silog_loss(pred_flat, target_flat, mask_flat)
            loss_l1    = l1_loss(pred_flat, target_flat, mask_flat)
            loss       = loss_silog + CONFIG["lambda_l1"] * loss_l1

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            total_loss  += loss.item()
            total_silog += loss_silog.item()
            total_l1    += loss_l1.item()
            global_step += 1

            if is_main() and (step + 1) % CONFIG["log_interval"] == 0:
                scale_val = out["scale_factor"].float().mean().item()

                log_dict = {
                    "train/loss_step":  loss.item(),
                    "train/silog_step": loss_silog.item(),
                    "train/l1_step":    loss_l1.item(),
                    "train/scale_mean": scale_val,
                    "train/scale_std":  out["scale_factor"].float().std().item(),
                    "grad/scale_token": (
                        raw_model.aggregator.scale_token.grad.norm().item()
                        if raw_model.aggregator.scale_token.grad is not None else 0.0
                    ),
                }

                # 当 depth_head 参与训练时，额外监控其梯度范数
                if not CONFIG["freeze_depth_head"]:
                    dpt_grads = [
                        p.grad.norm().item()
                        for p in raw_model.depth_head.parameters()
                        if p.grad is not None
                    ]
                    if dpt_grads:
                        log_dict["grad/depth_head"] = sum(dpt_grads) / len(dpt_grads)

                print(f"  step {step + 1}: loss={loss.item():.4f} "
                      f"(silog={loss_silog.item():.4f}, l1={loss_l1.item():.4f}) "
                      f"scale={scale_val:.4f}")
                wandb.log(log_dict, step=global_step)

        scheduler.step()

        if is_main():
            n = len(train_loader)
            print(f"Epoch {epoch + 1} avg loss: {total_loss / n:.4f}")
            wandb.log({
                "train/loss_epoch":  total_loss  / n,
                "train/silog_epoch": total_silog / n,
                "train/l1_epoch":    total_l1    / n,
                "train/lr_new":      optimizer.param_groups[0]["lr"],
                "train/lr_dpt":      optimizer.param_groups[-1]["lr"],
                "epoch":             epoch + 1,
            }, step=global_step)

        # ── 验证 ─────────────────────────────────────────────────────────
        if is_main() and (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()
            all_metrics = {"d1": [], "d2": [], "abs_rel": [], "rmse": []}
            scale_list  = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    images   = batch["image"].unsqueeze(1).to(device).to(dtype)
                    gt_depth = batch["depth"].to(device).float()

                    with torch.cuda.amp.autocast(dtype=dtype):
                        out = model(images)

                    pred_depth = out["depth"].squeeze(1).squeeze(-1).float()
                    scale_list.append(out["scale_factor"].float().mean().item())

                    pred_flat   = pred_depth.reshape(-1)
                    target_flat = gt_depth.reshape(-1)
                    mask_flat   = (target_flat > 0.1) & (target_flat < CONFIG["max_depth"])

                    if mask_flat.sum() == 0:
                        continue

                    m = compute_metrics(pred_flat, target_flat, mask_flat)
                    for k, v in m.items():
                        all_metrics[k].append(v)

            avg       = {k: sum(v) / len(v) for k, v in all_metrics.items()}
            avg_scale = sum(scale_list) / len(scale_list)

            print(f"\n📊 Val (Epoch {epoch + 1}): "
                  f"AbsRel={avg['abs_rel']:.4f}  RMSE={avg['rmse']:.4f}  "
                  f"δ1={avg['d1']:.4f}  δ2={avg['d2']:.4f}  "
                  f"scale={avg_scale:.4f}")

            wandb.log({
                "val/abs_rel":   avg["abs_rel"],
                "val/rmse":      avg["rmse"],
                "val/d1":        avg["d1"],
                "val/d2":        avg["d2"],
                "val/scale_avg": avg_scale,
            }, step=global_step)

            if avg["abs_rel"] < best_abs_rel:
                best_abs_rel = avg["abs_rel"]

                ckpt = {
                    "epoch":            epoch + 1,
                    "scale_token":      raw_model.aggregator.scale_token.data,
                    "scale_mlp_state":  raw_model.scale_mlp.state_dict(),
                    "fuse_conv_state":  raw_model.fuse_conv.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "metrics":          avg,
                    "config":           CONFIG,
                }

                # depth_head 微调时一并保存，冻结时无需保存（不变）
                if not CONFIG["freeze_depth_head"]:
                    ckpt["depth_head_state"] = raw_model.depth_head.state_dict()

                torch.save(ckpt, os.path.join(CONFIG["save_dir"], "best_ckpt.pth"))
                print(f"   ✅ Saved best (AbsRel={best_abs_rel:.4f})")
                wandb.run.summary["best_abs_rel"] = best_abs_rel
                wandb.run.summary["best_epoch"]   = epoch + 1

        if use_ddp:
            dist.barrier()

    if is_main():
        wandb.finish()

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()