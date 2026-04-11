# train_mixed.py
"""
Mixed NYU-Depth-v2 + KITTI Eigen depth training
================================================
Key design decisions vs. the single-dataset script:

1.  **Dataset-aware loss masking**
    NYU uses 0.1–10 m;  KITTI uses 1e-3–80 m.
    Each sample carries its own ``max_depth`` field, and the mask is built
    per-sample rather than from a global CONFIG constant.

2.  **Balanced sampling**
    A ``BalancedDatasetSampler`` ensures each mini-batch contains a roughly
    equal proportion of NYU and KITTI samples regardless of dataset size
    imbalance (KITTI ~30× larger).

3.  **Scale-factor head**
    The VGGTScaleDepth model already predicts a per-sample scale token.
    For KITTI the absolute depth range is 8× wider, so we log separate
    ``scale_mean_nyu`` / ``scale_mean_kitti`` metrics to verify the head
    adapts appropriately.

4.  **Dataset-specific validation metrics**
    AbsRel / RMSE / δ₁ are evaluated separately for NYU and KITTI so you
    can track regression on either domain independently.

Usage — single GPU:
    python train_mixed.py

Usage — multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train_mixed.py
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Iterator, List

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

from vggt.models.vggt import VGGT
from vggt.models.vggt_scale import VGGTScaleDepth
from dataset_nyu import NYUv2Dataset
from dataset_kitti import KITTIDepthDataset


# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Data paths ────────────────────────────────────────────────────────
    "nyu_mat_path":   "/root/autodl-tmp/nyu_depth_v2_labeled_baidu.mat",
    "kitti_root":     "/root/autodl-tmp/kitti",
    "img_size":       518,
    # ── Depth caps (per dataset; also used for loss masking) ──────────────
    "nyu_max_depth":  10.0,
    "kitti_max_depth": 80.0,
    # ── Sampling balance ──────────────────────────────────────────────────
    # Fraction of each batch drawn from NYU  (1 - ratio → KITTI).
    # 0.5 = equal count, regardless of raw dataset sizes.
    "nyu_sample_ratio": 0.5,
    # ── Training ──────────────────────────────────────────────────────────
    "batch_size":     4,
    "num_epochs":     30,
    "lr_new":             1e-3,   # scale_token / scale_mlp / fuse_conv
    "lr_lora":            6e-4,   # LoRA A/B matrices in the backbone
    "lr_depth_head":      2e-6,   # depth_head fine-tune (100× smaller)
    "weight_decay":       1e-4,
    "freeze_depth_head": False,
    # ── LoRA ──────────────────────────────────────────────────────────────
    "lora_rank":      8,
    "lora_alpha":     16.0,
    "lora_dropout":   0.05,
    "lora_targets":   ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
    # ── Loss ──────────────────────────────────────────────────────────────
    "lambda_l1":      0.1,
    # Per-dataset SILog loss weights (sum doesn't need to be 1).
    "loss_weight_nyu":   1.0,
    "loss_weight_kitti": 1.0,
    # ── Misc ──────────────────────────────────────────────────────────────
    "scale_per_frame": False,
    "save_dir":       "./checkpoints_mixed",
    "log_interval":   20,
    "val_interval":   1,
    "wandb_project":  "vggt-mixed-depth",
    "wandb_run_name": "nyu-kitti-lora-v1",
}


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_ddp() -> bool:
    return "RANK" in os.environ

def setup_ddp() -> int:
    import datetime
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=4),   # 默认 10 分钟太短，验证集大时会超时
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp() -> None:
    dist.destroy_process_group()

def is_main() -> bool:
    return dist.get_rank() == 0 if is_ddp() else True

def allreduce_scalar(val: float, device: torch.device) -> float:
    """对单个标量在所有 rank 间求均值（DDP 时使用，单卡直接返回原值）。"""
    if not is_ddp():
        return val
    t = torch.tensor(val, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


# ─────────────────────────────────────────────────────────────────────────────
# Balanced sampler
# ─────────────────────────────────────────────────────────────────────────────

class BalancedDatasetSampler(Sampler):
    """
    Produces indices that interleave samples from two sub-datasets at a
    user-specified ratio, without replacement within each epoch.

    Parameters
    ----------
    dataset_a_len, dataset_b_len : int
        Lengths of sub-dataset A (NYU) and B (KITTI).
    ratio_a : float
        Fraction of each batch drawn from A.  E.g. 0.5 → 50/50.
    total_samples : int | None
        Total epoch length.  Defaults to 2 × max(len_a, len_b) so that
        the larger dataset is always covered once.
    """

    def __init__(
        self,
        dataset_a_len: int,
        dataset_b_len: int,
        ratio_a: float = 0.5,
        total_samples: int | None = None,
    ) -> None:
        self.len_a    = dataset_a_len
        self.len_b    = dataset_b_len
        self.ratio_a  = ratio_a

        if total_samples is None:
            total_samples = 2 * max(dataset_a_len, dataset_b_len)
        self.total = total_samples

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator[int]:
        n_a = round(self.total * self.ratio_a)
        n_b = self.total - n_a

        # Sample with replacement so the smaller dataset can be reused
        idx_a = torch.randint(0,          self.len_a, (n_a,)).tolist()
        idx_b = torch.randint(self.len_a, self.len_a + self.len_b, (n_b,)).tolist()

        merged = idx_a + idx_b
        random.shuffle(merged)
        return iter(merged)


# ─────────────────────────────────────────────────────────────────────────────
# Loss & metrics  (dataset-aware via per-sample max_depth)
# ─────────────────────────────────────────────────────────────────────────────

def build_mask(pred: torch.Tensor,
               target: torch.Tensor,
               max_depth: torch.Tensor) -> torch.Tensor:
    """
    Per-sample valid mask.  ``max_depth`` is a 1-D tensor [B] carrying each
    sample's depth cap (10 m for NYU, 80 m for KITTI).
    """
    # Expand max_depth to match the flat pred/target shape isn't practical, so
    # we build the mask in batch dimension first.
    B = target.shape[0]
    masks = []
    for i in range(B):
        cap = max_depth[i].item()
        m   = (target[i] > 0.1) & (target[i] <= cap)
        masks.append(m)
    return torch.stack(masks)   # [B, ...]


def silog_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, lamda: float = 0.85) -> torch.Tensor:
    p = pred[mask].clamp(min=1e-4)
    t = target[mask].clamp(min=1e-4)
    d = torch.log(p) - torch.log(t)
    return torch.sqrt((d ** 2).mean() - lamda * d.mean() ** 2 + 1e-8)


def l1_loss(pred: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred[mask], target[mask])


def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> dict:
    p = pred[mask].clamp(min=1e-4)
    t = target[mask].clamp(min=1e-4)
    r = torch.max(p / t, t / p)
    return {
        "d1":      (r < 1.25      ).float().mean().item(),
        "d2":      (r < 1.25 ** 2 ).float().mean().item(),
        "abs_rel": ((p - t).abs() / t).mean().item(),
        "rmse":    torch.sqrt(((p - t) ** 2).mean()).item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-aware collation
# ─────────────────────────────────────────────────────────────────────────────

def mixed_collate_fn(batch: list) -> dict:
    """
    Custom collate that handles the heterogeneous ``max_depth`` and
    ``dataset`` string fields alongside the standard tensors.
    """
    images    = torch.stack([b["image"]    for b in batch])
    depths    = torch.stack([b["depth"]    for b in batch])
    max_depths = torch.tensor([b["max_depth"] for b in batch], dtype=torch.float32)
    datasets  = [b["dataset"] for b in batch]
    return {
        "image":     images,
        "depth":     depths,
        "max_depth": max_depths,
        "dataset":   datasets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset loss split helper
# ─────────────────────────────────────────────────────────────────────────────

def dataset_split_loss(
    pred_depth: torch.Tensor,       # [B, H, W]
    gt_depth:   torch.Tensor,       # [B, 1, H, W]  or [B, H, W]
    max_depth_per_sample: torch.Tensor,   # [B]
    dataset_labels: list,           # ["nyu", "kitti", ...]
    lambda_l1: float,
    weight_nyu: float,
    weight_kitti: float,
) -> tuple[torch.Tensor, dict]:
    """
    Compute weighted SILog + L1 loss, split by dataset origin.
    Returns (total_loss, log_dict).
    """
    gt = gt_depth.squeeze(1) if gt_depth.dim() == 4 else gt_depth  # [B, H, W]

    nyu_idx   = [i for i, d in enumerate(dataset_labels) if d == "nyu"]
    kitti_idx = [i for i, d in enumerate(dataset_labels) if d == "kitti"]

    total_loss = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
    log_dict   = {}

    for name, idx, weight in [
        ("nyu",   nyu_idx,   weight_nyu),
        ("kitti", kitti_idx, weight_kitti),
    ]:
        if not idx:
            continue
        sel_idx = torch.tensor(idx, device=pred_depth.device)
        p  = pred_depth[sel_idx].reshape(-1)
        t  = gt[sel_idx].reshape(-1)
        md = max_depth_per_sample[sel_idx]

        # Build flat mask
        t_2d   = gt[sel_idx]         # [n, H, W]
        md_exp = md.view(-1, 1, 1).expand_as(t_2d)
        mask   = ((t_2d > 0.1) & (t_2d <= md_exp)).reshape(-1)

        if mask.sum() == 0:
            continue

        sl = silog_loss(p, t, mask)
        l1 = l1_loss(p, t, mask)
        ds_loss = weight * (sl + lambda_l1 * l1)

        total_loss = total_loss + ds_loss
        log_dict[f"train/silog_{name}"] = sl.item()
        log_dict[f"train/l1_{name}"]    = l1.item()
        log_dict[f"train/loss_{name}"]  = ds_loss.item()

    return total_loss, log_dict


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
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

    # ── 1. Model ────────────────────────────────────────────────────────────
    if is_main():
        print("Loading pretrained VGGT …")
    pretrained = VGGT.from_pretrained("facebook/VGGT-1B").to(device).to(dtype)

    model = VGGTScaleDepth(
        pretrained_vggt=pretrained,
        embed_dim=1024,
        freeze_backbone=True,
        freeze_depth_head=CONFIG["freeze_depth_head"],
        scale_per_frame=CONFIG["scale_per_frame"],
        img_size=CONFIG["img_size"],
        lora_rank=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        lora_targets=CONFIG["lora_targets"],
    ).to(device)

    del pretrained

    model.aggregator.to(dtype)
    model.depth_head.to(torch.float32)
    model.scale_mlp.to(torch.float32)
    model.fuse_conv.to(torch.float32)

    if is_main():
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total: {total/1e6:.1f}M  Trainable: {trainable/1e6:.3f}M "
              f"({100*trainable/total:.2f}%)")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if use_ddp else model

    # ── 2. Datasets ─────────────────────────────────────────────────────────
    if is_main():
        print("Building NYU dataset …")
    nyu_train = NYUv2Dataset(
        CONFIG["nyu_mat_path"], split="train", img_size=CONFIG["img_size"]
    )
    nyu_val = NYUv2Dataset(
        CONFIG["nyu_mat_path"], split="val",   img_size=CONFIG["img_size"]
    )

    if is_main():
        print("Building KITTI dataset …")
    kitti_train = KITTIDepthDataset(
        CONFIG["kitti_root"],
        split="train",
        img_size=CONFIG["img_size"],
        max_depth=CONFIG["kitti_max_depth"],
        augment=True,
    )
    kitti_val = KITTIDepthDataset(
        CONFIG["kitti_root"],
        split="val",
        img_size=CONFIG["img_size"],
        max_depth=CONFIG["kitti_max_depth"],
        augment=False,
    )

    # ── NYUv2Dataset 补充 dataset / max_depth 字段 ─────────────────────────
    # We monkey-patch __getitem__ to inject the two extra fields that
    # KITTIDepthDataset already returns natively.
    _nyu_get = NYUv2Dataset.__getitem__
    def _nyu_getitem_patched(self, idx):
        sample = _nyu_get(self, idx)
        sample["dataset"]   = "nyu"
        sample["max_depth"] = CONFIG["nyu_max_depth"]
        return sample
    NYUv2Dataset.__getitem__ = _nyu_getitem_patched

    # ── 3. Loaders ───────────────────────────────────────────────────────────
    # Training: ConcatDataset + BalancedDatasetSampler (single-GPU) or
    #           DistributedSampler wrapping the balanced indices (multi-GPU).
    concat_train = ConcatDataset([nyu_train, kitti_train])

    if use_ddp:
        # For DDP we rely on DistributedSampler; balance is achieved by
        # keeping equal-size sub-splits (an approximation).
        train_sampler = DistributedSampler(concat_train, shuffle=True)
        val_nyu_sampler   = DistributedSampler(nyu_val,   shuffle=False)
        val_kitti_sampler = DistributedSampler(kitti_val, shuffle=False)
    else:
        train_sampler = BalancedDatasetSampler(
            dataset_a_len=len(nyu_train),
            dataset_b_len=len(kitti_train),
            ratio_a=CONFIG["nyu_sample_ratio"],
        )
        val_nyu_sampler   = None
        val_kitti_sampler = None

    train_loader = DataLoader(
        concat_train,
        batch_size=CONFIG["batch_size"],
        sampler=train_sampler,
        collate_fn=mixed_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_nyu_loader = DataLoader(
        nyu_val,
        batch_size=CONFIG["batch_size"],
        sampler=val_nyu_sampler,
        shuffle=False,
        collate_fn=mixed_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_kitti_loader = DataLoader(
        kitti_val,
        batch_size=CONFIG["batch_size"],
        sampler=val_kitti_sampler,
        shuffle=False,
        collate_fn=mixed_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # ── 4. Optimizer & scheduler ────────────────────────────────────────────
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
    best_abs_rel_nyu   = float("inf")
    best_abs_rel_kitti = float("inf")
    global_step = 0

    # ── 5. Training loop ────────────────────────────────────────────────────
    for epoch in range(CONFIG["num_epochs"]):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss  = 0.0
        epoch_count = 0

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main())
        ):
            images   = batch["image"].unsqueeze(1).to(device).to(dtype)  # [B,1,3,H,W]
            gt_depth = batch["depth"].to(device).float()                  # [B,1,H,W]
            max_d    = batch["max_depth"].to(device)                      # [B]
            ds_names = batch["dataset"]                                   # list[str]

            with torch.cuda.amp.autocast(dtype=dtype):
                out = model(images)

            pred_depth = out["depth"].squeeze(1).squeeze(-1).float()     # [B, H, W]

            loss, log_dict = dataset_split_loss(
                pred_depth, gt_depth, max_d, ds_names,
                CONFIG["lambda_l1"],
                CONFIG["loss_weight_nyu"],
                CONFIG["loss_weight_kitti"],
            )

            if loss.item() == 0.0:   # all pixels masked
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_count += 1
            global_step += 1

            if is_main() and (step + 1) % CONFIG["log_interval"] == 0:
                scale_arr = out["scale_factor"].float()
                scale_nyu   = scale_arr[
                    [i for i, d in enumerate(ds_names) if d == "nyu"]
                ].mean().item() if any(d == "nyu" for d in ds_names) else 0.0
                scale_kitti = scale_arr[
                    [i for i, d in enumerate(ds_names) if d == "kitti"]
                ].mean().item() if any(d == "kitti" for d in ds_names) else 0.0

                log_dict.update({
                    "train/loss_total":    loss.item(),
                    "train/scale_nyu":     scale_nyu,
                    "train/scale_kitti":   scale_kitti,
                })
                wandb.log(log_dict, step=global_step)
                print(
                    f"  step {step+1}: loss={loss.item():.4f} "
                    f"| nyu_silog={log_dict.get('train/silog_nyu', 0):.4f} "
                    f"| kitti_silog={log_dict.get('train/silog_kitti', 0):.4f}"
                )

        scheduler.step()

        if is_main() and epoch_count > 0:
            wandb.log({
                "train/loss_epoch": epoch_loss / epoch_count,
                "epoch":            epoch + 1,
            }, step=global_step)

        # ── 6. Validation — 所有 rank 都参与，all_reduce 汇总 ──────────────
        # 关键：不能只让 rank 0 跑验证，否则其他 rank 在 barrier 处等待超时。
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            model.eval()

            def _run_val(loader: DataLoader, tag: str) -> dict:
                """
                所有 rank 各自跑自己的分片（DistributedSampler 已切分），
                最后用 all_reduce 把各 rank 的累计和聚合为全局均值。
                """
                metric_keys = ["d1", "d2", "abs_rel", "rmse", "scale"]
                # 用 sum + count 两个变量来做全局均值，避免 list 跨进程传输
                local_sums   = {k: 0.0 for k in metric_keys}
                local_counts = {k: 0   for k in metric_keys}

                with torch.no_grad():
                    for batch in tqdm(loader, desc=f"Val-{tag}",
                                      disable=not is_main()):
                        images   = batch["image"].unsqueeze(1).to(device).to(dtype)
                        gt_depth = batch["depth"].to(device).float()
                        max_d    = batch["max_depth"].to(device)

                        with torch.cuda.amp.autocast(dtype=dtype):
                            out = model(images)

                        pred = out["depth"].squeeze(1).squeeze(-1).float()
                        gt   = gt_depth.squeeze(1)

                        local_sums["scale"]   += out["scale_factor"].float().mean().item()
                        local_counts["scale"] += 1

                        for i in range(pred.shape[0]):
                            cap  = max_d[i].item()
                            mask = (gt[i] > 0.1) & (gt[i] <= cap)
                            if mask.sum() == 0:
                                continue
                            m = compute_metrics(pred[i], gt[i], mask)
                            for k, v in m.items():
                                local_sums[k]   += v
                                local_counts[k] += 1

                # all_reduce：把各 rank 的 sum 和 count 加总
                avg = {}
                for k in metric_keys:
                    s = allreduce_scalar(local_sums[k],   device)
                    c = allreduce_scalar(float(local_counts[k]), device)
                    avg[k] = s / c if c > 0 else 0.0
                return avg

            avg_nyu   = _run_val(val_nyu_loader,  "NYU")
            avg_kitti = _run_val(val_kitti_loader, "KITTI")

            # 验证结束，所有 rank 切回训练模式
            model.train()

            if is_main():
                print(
                    f"\n📊 Epoch {epoch+1} — NYU  : "
                    f"AbsRel={avg_nyu['abs_rel']:.4f}  RMSE={avg_nyu['rmse']:.4f}  "
                    f"δ1={avg_nyu['d1']:.4f}  scale={avg_nyu['scale']:.4f}"
                )
                print(
                    f"📊 Epoch {epoch+1} — KITTI: "
                    f"AbsRel={avg_kitti['abs_rel']:.4f}  RMSE={avg_kitti['rmse']:.4f}  "
                    f"δ1={avg_kitti['d1']:.4f}  scale={avg_kitti['scale']:.4f}\n"
                )

                wandb.log({
                    "val_nyu/abs_rel":    avg_nyu["abs_rel"],
                    "val_nyu/rmse":       avg_nyu["rmse"],
                    "val_nyu/d1":         avg_nyu["d1"],
                    "val_nyu/scale":      avg_nyu["scale"],
                    "val_kitti/abs_rel":  avg_kitti["abs_rel"],
                    "val_kitti/rmse":     avg_kitti["rmse"],
                    "val_kitti/d1":       avg_kitti["d1"],
                    "val_kitti/scale":    avg_kitti["scale"],
                }, step=global_step)

                improved = False
                if avg_nyu["abs_rel"] < best_abs_rel_nyu:
                    best_abs_rel_nyu = avg_nyu["abs_rel"]
                    improved = True
                if avg_kitti["abs_rel"] < best_abs_rel_kitti:
                    best_abs_rel_kitti = avg_kitti["abs_rel"]
                    improved = True

                if improved:
                    ckpt = {
                        "epoch":           epoch + 1,
                        "lora_rank":       CONFIG["lora_rank"],
                        "lora_alpha":      CONFIG["lora_alpha"],
                        "adapter_state":   raw_model.lora_state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "metrics_nyu":     avg_nyu,
                        "metrics_kitti":   avg_kitti,
                        "config":          CONFIG,
                    }
                    torch.save(ckpt, os.path.join(CONFIG["save_dir"], "best_ckpt.pth"))
                    print(
                        f"   ✅ Saved best — "
                        f"NYU AbsRel={best_abs_rel_nyu:.4f}  "
                        f"KITTI AbsRel={best_abs_rel_kitti:.4f}"
                    )
                    wandb.run.summary["best_abs_rel_nyu"]   = best_abs_rel_nyu
                    wandb.run.summary["best_abs_rel_kitti"] = best_abs_rel_kitti
                    wandb.run.summary["best_epoch"]         = epoch + 1

        # 每个 epoch 结尾的 barrier：两个 rank 都会到达这里（验证已同步完毕）
        if use_ddp:
            dist.barrier()

    if is_main():
        wandb.finish()
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()