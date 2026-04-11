# eval_depth.py
"""
微调后 VGGTScaleDepth 评估脚本（含对比模型）
=============================================
功能：
  1. 加载 best_ckpt.pth 中保存的 adapter_state
  2. 在 NYU-Depth-v2 和 KITTI val 集上分别评估
  3. 输出学术标准的 7 项深度指标
  4. 可视化对比图（RGB / GT / Pred / Error map）
  5. 与「未微调 baseline」及第三方对比模型比较，量化微调收益

对比模型（通过 --compare 指定，可多选）：
  depth_anything_v2   Depth Anything V2 Metric (indoor/outdoor 自动切换)
                      · 正确 HF repo:
                          depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
                          depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
                      · 依赖: pip install transformers>=4.40
  map_anything        MAPAnything (Facebook Research, monocular metric depth)
                      · 正确 HF repo: facebook/map-anything
                      · 依赖: pip install git+https://github.com/facebookresearch/map-anything

  ⚠️  原代码中 "Depth-Anything-V3-Large-Indoor/Outdoor" 根本不存在于 HuggingFace；
      "Depth Anything 3 (DA3)" 使用 depth_anything_3.api.DepthAnything3 自定义 API，
      与 AutoModelForDepthEstimation 完全不兼容，已替换为 DAv2 Metric。
      旧参数 --compare depth_anything_v3 会自动映射到 depth_anything_v2。

用法：
  # 只评估微调模型
  python eval_depth.py --ckpt ./checkpoints_scale/best_ckpt.pth

  # 加入所有对比模型
  python eval_depth.py --ckpt ./checkpoints_scale/best_ckpt.pth \
      --baseline \
      --compare depth_anything_v2 map_anything

  # 只评估 NYU，并保存可视化
  python eval_depth.py --ckpt ./checkpoints_scale/best_ckpt.pth \
      --dataset nyu --vis --vis_dir ./vis_output \
      --compare depth_anything_v2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── 可视化依赖（可选）────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from vggt.models.vggt import VGGT
from vggt.models.vggt_scale import VGGTScaleDepth
from dataset_nyu import NYUv2Dataset
from dataset_kitti import KITTIDepthDataset


# ─────────────────────────────────────────────────────────────────────────────
# 路径配置（与训练脚本保持一致）
# ─────────────────────────────────────────────────────────────────────────────
DATA_CFG = {
    "nyu_mat_path":    "/root/autodl-tmp/nyu_depth_v2_labeled_baidu.mat",
    "kitti_root":      "/root/autodl-tmp/kitti",
    "img_size":        518,
    "nyu_max_depth":   10.0,
    "kitti_max_depth": 80.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# 通用对比模型接口
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonModel(nn.Module):
    """
    把任意第三方深度模型包装成与 VGGTScaleDepth 相同的调用接口：
        out = model(images)   # images: [B,1,3,H,W]  float32  (T=1 单帧)
        out["depth"]          # [B,1,H,W,1]  float32
    子类只需实现 _infer_single(rgb: Tensor [B,3,H,W] float32) -> Tensor [B,H,W]
    注意：evaluate() 保证传入 float32；子类写上 rgb.float() 更安全。
    """

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: [B, T=1, 3, H, W]  float32
        rgb   = images[:, 0]                        # [B,3,H,W]
        depth = self._infer_single(rgb)             # [B,H,W]
        depth = depth.unsqueeze(1).unsqueeze(-1)    # [B,1,H,W,1]
        return {"depth": depth}

    def _infer_single(self, rgb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Depth Anything V2 Metric
# ─────────────────────────────────────────────────────────────────────────────

def build_depth_anything_v2(
    device:   torch.device,
    dtype:    torch.dtype,
    img_size: int = 518,
) -> ComparisonModel:
    """
    Depth Anything V2 Metric（via HuggingFace transformers）。

    正确的 HF repo ID：
      indoor : depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf   (max_depth=20)
      outdoor: depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf  (max_depth=80)

    依赖：
        pip install transformers>=4.40
    """
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    _ID_INDOOR  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    _ID_OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

    class DepthAnythingV2Wrapper(ComparisonModel):
        def __init__(self):
            super().__init__()
            print(f"  [DA-V2] Loading indoor  model: {_ID_INDOOR}")
            self.model_indoor  = AutoModelForDepthEstimation.from_pretrained(
                _ID_INDOOR).to(device).to(torch.float32).eval()
            self.proc_indoor   = AutoImageProcessor.from_pretrained(_ID_INDOOR)

            print(f"  [DA-V2] Loading outdoor model: {_ID_OUTDOOR}")
            self.model_outdoor = AutoModelForDepthEstimation.from_pretrained(
                _ID_OUTDOOR).to(device).to(torch.float32).eval()
            self.proc_outdoor  = AutoImageProcessor.from_pretrained(_ID_OUTDOOR)

            self._target_size = (img_size, img_size)
            self._max_depth   = 10.0   # 初始值，evaluate() 会通过 set_max_depth 注入

        def set_max_depth(self, max_depth: float) -> None:
            self._max_depth = max_depth

        def _run_head(self, rgb_bhwc_np: np.ndarray,
                      model, processor) -> torch.Tensor:
            """uint8 numpy [B,H,W,3] → metric depth Tensor [B,H,W] on device"""
            inputs = processor(
                images=[rgb_bhwc_np[i] for i in range(len(rgb_bhwc_np))],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)
            depth = out.predicted_depth    # [B, H', W']，单位：米
            depth = F.interpolate(
                depth.unsqueeze(1).float(),
                size=self._target_size,
                mode="bilinear", align_corners=False,
            ).squeeze(1)                   # [B,H,W]
            return depth.to(device)        # 确保在正确 device 上

        def _infer_single(self, rgb: torch.Tensor) -> torch.Tensor:
            rgb = rgb.float()   # 确保 float32

            mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=rgb.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225],
                                 device=rgb.device).view(1, 3, 1, 1)
            rgb_u8 = ((rgb * std + mean).clamp(0, 1)
                      .permute(0, 2, 3, 1).mul(255)
                      .byte().cpu().numpy())         # [B,H,W,3] uint8

            # DAv2 indoor 训练在 max_depth=20，outdoor 训练在 max_depth=80
            use_outdoor = self._max_depth > 15.0
            if use_outdoor:
                return self._run_head(rgb_u8, self.model_outdoor, self.proc_outdoor)
            else:
                return self._run_head(rgb_u8, self.model_indoor,  self.proc_indoor)

    return DepthAnythingV2Wrapper()


# ─────────────────────────────────────────────────────────────────────────────
# MAPAnything
# ─────────────────────────────────────────────────────────────────────────────

def build_map_anything(
    device:   torch.device,
    dtype:    torch.dtype,
    img_size: int = 518,
) -> ComparisonModel:
    """
    Facebook Research MAPAnything（metric monocular depth）。

    正确 HF repo: facebook/map-anything
    依赖（先 clone 再安装）：
        git clone https://github.com/facebookresearch/map-anything
        cd map-anything && pip install -e .

    参考：
        https://github.com/facebookresearch/map-anything
        https://huggingface.co/facebook/map-anything
    """
    try:
        from mapanything.utils.inference import load_model, run_monocular_depth
    except ImportError:
        print("  [MAPAnything] ⚠️  mapanything 包未安装。请执行：")
        print("      git clone https://github.com/facebookresearch/map-anything")
        print("      cd map-anything && pip install -e .")
        raise

    _HF_REPO = "facebook/map-anything"

    class MAPAnythingWrapper(ComparisonModel):
        def __init__(self):
            super().__init__()
            print(f"  [MAPAnything] Loading model: {_HF_REPO}")
            # load_model 接受 HF repo ID 或本地路径
            self.model = load_model(_HF_REPO, device=device)
            self.model.eval()
            self._target_size = (img_size, img_size)

        def _infer_single(self, rgb: torch.Tensor) -> torch.Tensor:
            rgb = rgb.float()   # 确保 float32

            mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=rgb.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225],
                                 device=rgb.device).view(1, 3, 1, 1)
            # 逆归一化 → uint8 numpy [B,H,W,3]
            rgb_u8 = ((rgb * std + mean).clamp(0, 1)
                      .permute(0, 2, 3, 1).mul(255)
                      .byte().cpu().numpy())

            # run_monocular_depth：接受 uint8 numpy 列表，返回 [B,H,W] float32 tensor
            depth = run_monocular_depth(
                self.model,
                images=rgb_u8,           # [B,H,W,3]
                device=device,
            )                            # [B,H,W]  单位：米

            depth = F.interpolate(
                depth.unsqueeze(1).float(),
                size=self._target_size,
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            return depth.to(device)

    return MAPAnythingWrapper()


# ─────────────────────────────────────────────────────────────────────────────
# 对比模型注册表（新增模型在这里登记即可）
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_REGISTRY: Dict[str, callable] = {
    "depth_anything_v2": build_depth_anything_v2,
    "map_anything":       build_map_anything,
    # 未来可继续添加：
    # "unidepth_v2":      build_unidepth_v2,
    # "metric3d_v2":      build_metric3d_v2,
}

# 旧名称兼容（depth_anything_v3 不存在，自动映射到 v2）
_COMPAT_ALIASES: Dict[str, str] = {
    "depth_anything_v3": "depth_anything_v2",
}

DISPLAY_NAMES: Dict[str, str] = {
    "depth_anything_v2": "DepthAnything-V2",
    "depth_anything_v3": "DepthAnything-V2",   # compat alias
    "map_anything":       "MAPAnything",
}


# ─────────────────────────────────────────────────────────────────────────────
# 7 项标准指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred:  torch.Tensor,
                    gt:    torch.Tensor,
                    mask:  torch.Tensor) -> Dict[str, float]:
    p = pred[mask].clamp(min=1e-4)
    t = gt[mask].clamp(min=1e-4)

    thresh   = torch.max(p / t, t / p)
    abs_rel  = ((p - t).abs() / t).mean().item()
    sq_rel   = (((p - t) ** 2) / t).mean().item()
    rmse     = torch.sqrt(((p - t) ** 2).mean()).item()
    rmse_log = torch.sqrt(
        ((torch.log(p) - torch.log(t)) ** 2).mean()
    ).item()
    d1 = (thresh < 1.25     ).float().mean().item()
    d2 = (thresh < 1.25 ** 2).float().mean().item()
    d3 = (thresh < 1.25 ** 3).float().mean().item()

    return {
        "AbsRel":  abs_rel,  "SqRel":   sq_rel,
        "RMSE":    rmse,     "RMSElog": rmse_log,
        "δ<1.25":  d1,       "δ<1.25²": d2,  "δ<1.25³": d3,
    }


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def _denorm(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    img  = (tensor.float() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def save_vis(rgb:        torch.Tensor,
             gt_depth:   torch.Tensor,
             pred_depth: torch.Tensor,
             max_depth:  float,
             save_path:  str) -> None:
    if not HAS_MPL:
        return

    rgb_np  = _denorm(rgb)
    gt_np   = gt_depth.cpu().numpy()
    pred_np = pred_depth.cpu().numpy()

    valid  = (gt_np > 0.001) & (gt_np <= max_depth)
    err_np = np.zeros_like(gt_np)
    err_np[valid] = np.abs(pred_np[valid] - gt_np[valid]) / (gt_np[valid] + 1e-6)

    vmin, vmax = 0.0, max_depth
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].imshow(rgb_np);                              axes[0].set_title("RGB")
    axes[1].imshow(gt_np,   cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("GT Depth")
    axes[2].imshow(pred_np, cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[2].set_title("Pred Depth")
    im = axes[3].imshow(err_np, cmap="hot", vmin=0, vmax=0.3)
    axes[3].set_title("Abs Rel Error")
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 模型构建（VGGTScaleDepth）
# ─────────────────────────────────────────────────────────────────────────────

def build_model(ckpt_path: Optional[str],
                device:    torch.device,
                dtype:     torch.dtype) -> VGGTScaleDepth:
    print("Loading pretrained VGGT …")
    pretrained = VGGT.from_pretrained("facebook/VGGT-1B").to(device).to(dtype)

    lora_rank    = 8
    lora_alpha   = 16.0
    lora_targets = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")

    if ckpt_path is not None:
        ckpt         = torch.load(ckpt_path, map_location="cpu")
        lora_rank    = ckpt.get("lora_rank",  lora_rank)
        lora_alpha   = ckpt.get("lora_alpha", lora_alpha)
        cfg          = ckpt.get("config", {})
        lora_targets = cfg.get("lora_targets", lora_targets)

    model = VGGTScaleDepth(
        pretrained_vggt=pretrained,
        embed_dim=1024,
        freeze_backbone=True,
        freeze_depth_head=False,
        scale_per_frame=False,
        img_size=DATA_CFG["img_size"],
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        lora_targets=lora_targets,
    ).to(device)

    del pretrained
    model.aggregator.to(dtype)
    model.depth_head.to(torch.float32)
    model.scale_mlp.to(torch.float32)
    model.fuse_conv.to(torch.float32)

    if ckpt_path is not None:
        missing, unexpected = model.load_state_dict(
            ckpt["adapter_state"], strict=False
        )
        print(f"Loaded adapter from {ckpt_path}")
        if missing:
            print(f"  [警告] missing keys ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"  [警告] unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    else:
        print("  → Baseline mode: no adapter loaded")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 评估核心（通用，兼容 VGGTScaleDepth 和 ComparisonModel）
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    max_depth: float,
    device:    torch.device,
    dtype:     torch.dtype,
    vis:       bool = False,
    vis_dir:   str  = "./vis",
    vis_every: int  = 50,
    tag:       str  = "eval",
) -> Dict[str, float]:

    is_cmp = isinstance(model, ComparisonModel)

    # 对比模型的室内/室外切换：把 max_depth 注入（如有 set_max_depth 方法）
    if is_cmp and hasattr(model, "set_max_depth"):
        model.set_max_depth(max_depth)

    all_metrics: List[Dict[str, float]] = []
    vis_saved = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=tag)):
        gt_depth = batch["depth"].to(device).float().squeeze(1)   # [B,H,W]

        # ── 对比模型：float32 输入，不走 autocast ──────────────────────────
        # 理由：HF transformers / mapanything 内部是 float32；
        #       bfloat16 autocast 会与之冲突。
        if is_cmp:
            images = batch["image"].unsqueeze(1).to(device).float()
            out    = model(images)
        else:
            images = batch["image"].unsqueeze(1).to(device).to(dtype)
            with torch.cuda.amp.autocast(dtype=dtype):
                out = model(images)

        pred = out["depth"].squeeze(1).squeeze(-1).float()         # [B,H,W]

        # ── 尺度对齐（仅对比模型）：median scaling ────────────────────────
        if is_cmp:
            for i in range(pred.shape[0]):
                mask_i = (gt_depth[i] > 0.001) & (gt_depth[i] <= max_depth)
                if mask_i.sum() > 10:
                    scale = (gt_depth[i][mask_i].median() /
                             pred[i][mask_i].clamp(1e-4).median())
                    pred[i] = pred[i] * scale

        for i in range(pred.shape[0]):
            gt_i   = gt_depth[i]
            pred_i = pred[i]
            mask   = (gt_i > 0.001) & (gt_i <= max_depth)

            if mask.sum() < 10:
                continue

            m = compute_metrics(
                pred_i.reshape(-1), gt_i.reshape(-1), mask.reshape(-1)
            )
            all_metrics.append(m)

            if vis and HAS_MPL and (batch_idx % vis_every == 0) and i == 0:
                save_vis(
                    rgb=batch["image"][i],
                    gt_depth=gt_i,
                    pred_depth=pred_i.clamp(0, max_depth),
                    max_depth=max_depth,
                    save_path=os.path.join(
                        vis_dir, tag, f"batch{batch_idx:05d}.png"
                    ),
                )
                vis_saved += 1

    if not all_metrics:
        print(f"[{tag}] 没有有效样本，请检查 mask 条件或深度文件。")
        return {}

    avg = aggregate_metrics(all_metrics)
    if vis and vis_saved:
        print(f"  → 已保存 {vis_saved} 张可视化至 {vis_dir}/{tag}/")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# 打印指标表格
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: Dict[str, Dict[str, float]]) -> None:
    metric_order = ["AbsRel", "SqRel", "RMSE", "RMSElog",
                    "δ<1.25", "δ<1.25²", "δ<1.25³"]
    lower_better = {"AbsRel", "SqRel", "RMSE", "RMSElog"}

    col_w  = 11
    name_w = 30
    header = f"{'Model':<{name_w}}" + "".join(
        f"{m:>{col_w}}" for m in metric_order
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for name, metrics in results.items():
        if not metrics:
            continue
        row = f"{name:<{name_w}}"
        for m in metric_order:
            v     = metrics.get(m, float("nan"))
            arrow = "↓" if m in lower_better else "↑"
            row  += f"{v:>{col_w - 1}.4f}{arrow}"
        print(row)

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 改进率计算
# ─────────────────────────────────────────────────────────────────────────────

def print_improvement(ft:  Dict[str, float],
                      ref: Dict[str, float],
                      tag: str,
                      ref_name: str = "baseline") -> None:
    lower_better = {"AbsRel", "SqRel", "RMSE", "RMSElog"}
    print(f"\n  改进率 [{tag}]  vs {ref_name}  （正数 = 微调更好）")
    for k, v_ft in ft.items():
        v_ref = ref.get(k, float("nan"))
        if v_ref == 0 or v_ref != v_ref:
            continue
        if k in lower_better:
            delta = (v_ref - v_ft) / (abs(v_ref) + 1e-9) * 100
        else:
            delta = (v_ft - v_ref) / (abs(v_ref) + 1e-9) * 100
        sign = "✅" if delta > 0 else "❌"
        print(f"    {k:<12}: {delta:+.2f}%  {sign}")


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader 构建
# ─────────────────────────────────────────────────────────────────────────────

def _collate(batch):
    images   = torch.stack([b["image"]      for b in batch])
    depths   = torch.stack([b["depth"]      for b in batch])
    max_deps = torch.tensor([b["max_depth"] for b in batch])
    return {"image": images, "depth": depths, "max_depth": max_deps}


def build_loaders(dataset: str, batch_size: int = 4):
    loaders = {}

    if dataset in ("nyu", "both"):
        _orig = NYUv2Dataset.__getitem__
        def _patched(self, idx):
            s = _orig(self, idx)
            s.setdefault("max_depth", DATA_CFG["nyu_max_depth"])
            return s
        NYUv2Dataset.__getitem__ = _patched

        nyu_val = NYUv2Dataset(
            DATA_CFG["nyu_mat_path"], split="val",
            img_size=DATA_CFG["img_size"]
        )
        loaders["nyu"] = DataLoader(
            nyu_val, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=_collate,
        )

    if dataset in ("kitti", "both"):
        kitti_val = KITTIDepthDataset(
            DATA_CFG["kitti_root"], split="val",
            img_size=DATA_CFG["img_size"],
            max_depth=DATA_CFG["kitti_max_depth"],
            augment=False,
        )
        loaders["kitti"] = DataLoader(
            kitti_val, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=_collate,
        )

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VGGTScaleDepth 深度评估")
    parser.add_argument("--ckpt",       required=True,
                        help="微调 checkpoint 路径（best_ckpt.pth）")
    parser.add_argument("--dataset",    default="both",
                        choices=["nyu", "kitti", "both"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--baseline",   action="store_true",
                        help="跑未微调的 pretrained 作为对比基线")
    parser.add_argument(
        "--compare",
        nargs="+",
        choices=list(COMPARISON_REGISTRY.keys()) + list(_COMPAT_ALIASES.keys()),
        default=[],
        metavar="MODEL",
        help=(
            "加入第三方对比模型，可多选，空格分隔。\n"
            f"可选值：{', '.join(COMPARISON_REGISTRY.keys())}\n"
            "注：depth_anything_v3 不存在，会自动映射到 depth_anything_v2。"
        ),
    )
    parser.add_argument("--vis",        action="store_true")
    parser.add_argument("--vis_dir",    default="./vis_output")
    parser.add_argument("--vis_every",  type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    if args.vis and not HAS_MPL:
        print("[警告] matplotlib 未安装，--vis 参数将被忽略。")

    # 处理 alias（depth_anything_v3 → depth_anything_v2，去重）
    compare_resolved: List[str] = []
    for m in (args.compare or []):
        real = _COMPAT_ALIASES.get(m, m)
        if m != real:
            print(f"[警告] --compare {m} 不存在，已自动映射到 {real}。")
        if real not in compare_resolved:
            compare_resolved.append(real)

    # ── DataLoader ────────────────────────────────────────────────────────
    print(f"\n加载数据集（{args.dataset}）…")
    loaders = build_loaders(args.dataset, args.batch_size)

    results: Dict[str, Dict[str, float]] = {}

    # ── 1. 微调模型 ───────────────────────────────────────────────────────
    print(f"\n[Step 1] 评估微调模型: {args.ckpt}")
    ft_model = build_model(args.ckpt, device, dtype)

    for ds_name, loader in loaders.items():
        max_d = DATA_CFG[f"{ds_name}_max_depth"]
        tag   = f"{ds_name.upper()} (ours-finetuned)"
        print(f"\n  → {tag}  [{len(loader.dataset)} samples]")
        results[tag] = evaluate(
            ft_model, loader, max_d, device, dtype,
            vis=args.vis, vis_dir=args.vis_dir,
            vis_every=args.vis_every, tag=f"{ds_name}_ft",
        )

    del ft_model
    torch.cuda.empty_cache()

    # ── 2. Baseline（可选）───────────────────────────────────────────────
    if args.baseline:
        print(f"\n[Step 2] 评估 pretrained baseline …")
        bl_model = build_model(None, device, dtype)

        for ds_name, loader in loaders.items():
            max_d = DATA_CFG[f"{ds_name}_max_depth"]
            tag   = f"{ds_name.upper()} (ours-pretrained)"
            print(f"\n  → {tag}  [{len(loader.dataset)} samples]")
            results[tag] = evaluate(
                bl_model, loader, max_d, device, dtype,
                vis=False, tag=f"{ds_name}_bl",
            )

        del bl_model
        torch.cuda.empty_cache()

    # ── 3. 第三方对比模型（可选）─────────────────────────────────────────
    for step_idx, model_key in enumerate(compare_resolved, start=3):
        print(f"\n[Step {step_idx}] 评估对比模型: {model_key}")
        builder   = COMPARISON_REGISTRY[model_key]
        cmp_model = builder(device, dtype, img_size=DATA_CFG["img_size"])

        for ds_name, loader in loaders.items():
            max_d   = DATA_CFG[f"{ds_name}_max_depth"]
            display = DISPLAY_NAMES.get(model_key, model_key)
            tag     = f"{ds_name.upper()} ({display})"
            print(f"\n  → {tag}  [{len(loader.dataset)} samples]")
            results[tag] = evaluate(
                cmp_model, loader, max_d, device, dtype,
                vis=False, tag=f"{ds_name}_{model_key}",
            )

        del cmp_model
        torch.cuda.empty_cache()

    # ── 4. 打印结果表格 ───────────────────────────────────────────────────
    print_table(results)

    # ── 5. 改进率（微调 vs 各参照模型）──────────────────────────────────
    for ds_name in loaders:
        ft_key = f"{ds_name.upper()} (ours-finetuned)"
        if ft_key not in results:
            continue

        bl_key = f"{ds_name.upper()} (ours-pretrained)"
        if bl_key in results:
            print_improvement(results[ft_key], results[bl_key],
                               tag=ds_name.upper(), ref_name="pretrained-baseline")

        for model_key in compare_resolved:
            display = DISPLAY_NAMES.get(model_key, model_key)
            cmp_key = f"{ds_name.upper()} ({display})"
            if cmp_key in results:
                print_improvement(results[ft_key], results[cmp_key],
                                   tag=ds_name.upper(), ref_name=display)

    print("\n✅ 评估完成。\n")


if __name__ == "__main__":
    main()