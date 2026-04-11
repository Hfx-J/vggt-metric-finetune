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
    "batch_size":         4,
    "num_epochs":         30,
    # ── Learning rates ────────────────────────────────────────────────────
    "lr_new":             1e-3,   # scale_token / scale_mlp / fuse_conv
    "lr_lora":            6e-4,   # LoRA A/B matrices in the backbone
    "lr_depth_head":      2e-6,   # depth_head fine-tune (100× smaller)
    "weight_decay":       1e-4,
    # ── Freeze flags ─────────────────────────────────────────────────────
    "freeze_depth_head":  False,  # True = freeze depth_head entirely
    # ── LoRA config ───────────────────────────────────────────────────────
    "lora_rank":          8,      # 0 = disable LoRA (backbone fully frozen)
    "lora_alpha":         16.0,   # scaling = alpha / rank  →  2.0
    "lora_dropout":       0.05,
    # Which sub-modules inside each transformer block to adapt.
    # For a pure attention-only LoRA remove "mlp.fc1" / "mlp.fc2".
    "lora_targets": ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
    # ── Misc ──────────────────────────────────────────────────────────────
    "save_dir":           "./checkpoints_scale",
    "log_interval":       20,
    "val_interval":       1,
    "scale_per_frame":    False,
    "lambda_l1":          0.1,
    "wandb_project":      "vggt-scale-depth",
    "wandb_run_name":     "scale-token-lora-dpt-finetune-v1",
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
        print(
            f"Building VGGTScaleDepth ("
            f"lora_rank={CONFIG['lora_rank']}, "
            f"freeze_depth_head={CONFIG['freeze_depth_head']})..."
        )
    model = VGGTScaleDepth(
        pretrained_vggt=pretrained,
        embed_dim=1024,
        freeze_backbone=True,           # non-LoRA aggregator weights frozen
        freeze_depth_head=CONFIG["freeze_depth_head"],
        scale_per_frame=CONFIG["scale_per_frame"],
        img_size=CONFIG["img_size"],
        lora_rank=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        lora_targets=CONFIG["lora_targets"],
    ).to(device)

    del pretrained

    # ── dtype partitioning ────────────────────────────────────────────────
    # • aggregator (incl. LoRA layers) → bfloat16 for throughput
    # • depth_head / scale_mlp / fuse_conv → float32 for LayerNorm stability
    # Note: LoRALinear stores weight frozen in bfloat16 but keeps A/B in
    # float32 to avoid under-flow in small gradients.  The forward() cast
    # handles dtype promotion automatically.
    model.aggregator.to(dtype)
    model.depth_head.to(torch.float32)
    model.scale_mlp.to(torch.float32)
    model.fuse_conv.to(torch.float32)

    if is_main():
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params    : {total     / 1e6:.1f}M")
        print(f"Trainable params: {trainable / 1e6:.3f}M  "
              f"({100 * trainable / total:.2f}%)")
        raw = model
        for g in raw.trainable_parameter_groups(
            depth_head_lr=CONFIG["lr_depth_head"],
            lora_lr=CONFIG["lr_lora"],
        ):
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
    optimizer = torch.optim.AdamW(
        raw_model.trainable_parameter_groups(
            depth_head_lr=CONFIG["lr_depth_head"],
            lora_lr=CONFIG["lr_lora"],
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

                # ── gradient norm helpers ──────────────────────────────
                def _grad_norm(params):
                    gs = [p.grad.norm().item() for p in params if p.grad is not None]
                    return sum(gs) / len(gs) if gs else 0.0

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

                # LoRA gradient norms (A and B separately)
                if CONFIG["lora_rank"] > 0:
                    from vggt.models.vggt_scale import LoRALinear
                    lora_modules = [
                        m for m in raw_model.aggregator.modules()
                        if isinstance(m, LoRALinear)
                    ]
                    log_dict["grad/lora_A"] = _grad_norm(
                        [m.lora_A for m in lora_modules]
                    )
                    log_dict["grad/lora_B"] = _grad_norm(
                        [m.lora_B for m in lora_modules]
                    )

                if not CONFIG["freeze_depth_head"]:
                    log_dict["grad/depth_head"] = _grad_norm(
                        raw_model.depth_head.parameters()
                    )

                print(
                    f"  step {step + 1}: loss={loss.item():.4f} "
                    f"(silog={loss_silog.item():.4f}, l1={loss_l1.item():.4f}) "
                    f"scale={scale_val:.4f}"
                )
                wandb.log(log_dict, step=global_step)

        scheduler.step()

        if is_main():
            n = len(train_loader)
            print(f"Epoch {epoch + 1} avg loss: {total_loss / n:.4f}")

            # find lr for each group by name
            lr_by_name = {g["name"]: g["lr"] for g in optimizer.param_groups
                          if "name" in g}
            wandb.log({
                "train/loss_epoch":  total_loss  / n,
                "train/silog_epoch": total_silog / n,
                "train/l1_epoch":    total_l1    / n,
                "train/lr_new":      lr_by_name.get("scale_token", 0.0),
                "train/lr_lora":     lr_by_name.get("lora",        0.0),
                "train/lr_dpt":      lr_by_name.get("depth_head",  0.0),
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

            print(
                f"\n📊 Val (Epoch {epoch + 1}): "
                f"AbsRel={avg['abs_rel']:.4f}  RMSE={avg['rmse']:.4f}  "
                f"δ1={avg['d1']:.4f}  δ2={avg['d2']:.4f}  "
                f"scale={avg_scale:.4f}"
            )

            wandb.log({
                "val/abs_rel":   avg["abs_rel"],
                "val/rmse":      avg["rmse"],
                "val/d1":        avg["d1"],
                "val/d2":        avg["d2"],
                "val/scale_avg": avg_scale,
            }, step=global_step)

            if avg["abs_rel"] < best_abs_rel:
                best_abs_rel = avg["abs_rel"]

                # ── Compact checkpoint: only adapted weights ──────────
                ckpt = {
                    "epoch":           epoch + 1,
                    "lora_rank":       CONFIG["lora_rank"],
                    "lora_alpha":      CONFIG["lora_alpha"],
                    # lora_state_dict includes LoRA A/B + scale_token
                    # + scale_mlp + fuse_conv (+ depth_head if trained)
                    "adapter_state":   raw_model.lora_state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics":         avg,
                    "config":          CONFIG,
                }

                torch.save(
                    ckpt,
                    os.path.join(CONFIG["save_dir"], "best_ckpt.pth")
                )
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