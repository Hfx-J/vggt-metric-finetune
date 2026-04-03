# dataset_kitti.py
"""
KITTI Depth Dataset — 适配实际目录结构
======================================

实际布局：
  /autodl-tmp/kitti/
  ├── raw/                                      ← RGB 原图
  │   └── 2011_09_26/
  │       └── 2011_09_26_drive_0001_sync/
  │           └── image_02/data/
  │                   0000000000.png
  └── train/   (或 val/)                        ← 深度 GT (16-bit PNG)
      └── 2011_09_26_drive_0001_sync/
          └── proj_depth/groundtruth/image_02/
                  0000000000.png

深度 PNG 编码：uint16，value / 256.0 = 深度 (米)
有效范围：1e-3 m … 80 m (Eigen 评测惯例)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class KITTIDepthDataset(Dataset):
    """
    Parameters
    ----------
    kitti_root : str
        数据集根目录，下面应有 raw/  train/  val/ 三个子目录。
    split : "train" | "val"
        选择读取哪个分区；直接对应同名子目录。
    img_size : int
        输出的正方形分辨率（image 和 depth 同步缩放）。
    max_depth : float
        深度上限（米），超出部分在 mask 中置为无效。
    augment : bool
        训练时开启随机翻转 + 色彩抖动；val 时自动关闭。
    """

    MIN_DEPTH_M: float = 1e-3
    # KITTI image_02 可能是 .png 也可能是 .jpg，按此顺序尝试
    _RGB_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        kitti_root: str,
        split: str = "train",
        img_size: int = 518,
        max_depth: float = 80.0,
        augment: bool = True,
    ) -> None:
        super().__init__()

        assert split in ("train", "val"), f"split 只支持 'train'/'val'，收到: {split!r}"

        self.root       = Path(kitti_root)
        self.raw_root   = self.root / "raw"
        self.depth_root = self.root / split      # train/ 或 val/
        self.split      = split
        self.img_size   = img_size
        self.max_depth  = max_depth
        self.augment    = augment and (split == "train")

        # 验证目录存在
        for d, label in [(self.raw_root, "raw/"), (self.depth_root, f"{split}/")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"目录不存在: {d}  ← 请检查 kitti_root 路径是否正确"
                )

        self.samples: List[Tuple[Path, Path]] = self._discover_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"在 {self.depth_root} 下未找到任何有效样本，"
                "请确认 train/val 目录包含 proj_depth/groundtruth/image_02/*.png"
            )

        print(
            f"[KITTIDepthDataset] split={split}  "
            f"samples={len(self.samples)}  img_size={img_size}  max_depth={max_depth}m"
        )

        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )
        # hue 在旧版 torchvision PIL 后端有 uint8 负值溢出 bug，去掉 hue。
        # jitter 在 ToTensor 之后、对 float tensor 调用，绕开 PIL 路径。
        self._jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15
        )

    # ------------------------------------------------------------------
    # 文件发现
    # ------------------------------------------------------------------

    def _discover_samples(self) -> List[Tuple[Path, Path]]:
        """
        遍历 {split}/<drive>/proj_depth/groundtruth/image_02/*.png，
        并从 raw/<date>/<drive>/image_02/data/<frame>.[png|jpg] 找对应 RGB。
        """
        depth_paths = sorted(
            self.depth_root.rglob("proj_depth/groundtruth/image_02/*.png")
        )

        samples: List[Tuple[Path, Path]] = []
        missing_rgb = 0

        for d_path in depth_paths:
            rgb_path = self._depth_to_rgb(d_path)
            if rgb_path is None:
                missing_rgb += 1
                continue
            samples.append((rgb_path, d_path))

        if missing_rgb:
            print(
                f"  [警告] {missing_rgb} 个深度图未找到对应 RGB，已跳过。"
                "（检查 raw/ 子目录结构是否为 raw/<date>/<drive>/image_02/data/）"
            )

        return samples

    def _depth_to_rgb(self, depth_path: Path) -> Optional[Path]:
        """
        从深度图路径推导 RGB 路径。

        depth_path 示例：
          .../train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000000.png

        对应 RGB：
          .../raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.[png|jpg]

        日期前缀（2011_09_26）从 drive 名称前三段下划线拼接而来。
        """
        parts = depth_path.parts
        try:
            proj_idx = parts.index("proj_depth")
        except ValueError:
            return None

        drive = parts[proj_idx - 1]           # e.g. "2011_09_26_drive_0001_sync"
        frame = depth_path.stem               # e.g. "0000000000"

        # 日期 = drive 名称前三个下划线分隔段
        date_parts = drive.split("_")
        if len(date_parts) < 3:
            return None
        date = "_".join(date_parts[:3])       # e.g. "2011_09_26"

        rgb_dir = self.raw_root / date / drive / "image_02" / "data"

        for ext in self._RGB_EXTS:
            candidate = rgb_dir / f"{frame}{ext}"
            if candidate.exists():
                return candidate

        return None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rgb_path, depth_path = self.samples[idx]

        # ── 读取 RGB ──────────────────────────────────────────────────
        img = Image.open(rgb_path).convert("RGB")

        # ── 读取深度 (uint16 PNG → float32 米) ──────────────────────
        depth_np = np.array(Image.open(depth_path), dtype=np.float32) / 256.0

        # ── 同步空间增强（img 和 depth 必须完全一致）────────────────
        if self.augment and torch.rand(1).item() > 0.5:
            img      = TF.hflip(img)
            depth_np = depth_np[:, ::-1].copy()

        # ── RGB resize → tensor ──────────────────────────────────────
        img_tensor = TF.to_tensor(
            TF.resize(img, (self.img_size, self.img_size), antialias=True)
        )                                           # float32 [3,H,W]  0-1

        # ── 色彩抖动在 tensor 上执行（规避 PIL 后端 uint8 溢出 bug）──
        if self.augment:
            img_tensor = self._jitter(img_tensor)

        # ── ImageNet 归一化 ───────────────────────────────────────────
        img_tensor = self._normalize(img_tensor)

        # ── Depth resize（保留稀疏有效性）────────────────────────────
        # 双线性缩放后把原本无效像素（LiDAR 空洞）重新清零，
        # 防止插值把洞"填上"产生伪深度值。
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        valid_t = (depth_t > self.MIN_DEPTH_M) & (depth_t <= self.max_depth)

        depth_r = torch.nn.functional.interpolate(
            depth_t, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        valid_r = torch.nn.functional.interpolate(
            valid_t.float(), size=(self.img_size, self.img_size),
            mode="nearest",
        ).bool()

        depth_r[~valid_r] = 0.0
        depth_tensor = depth_r.squeeze(0)           # [1, img_size, img_size]

        return {
            "image":     img_tensor,      # [3, img_size, img_size]
            "depth":     depth_tensor,    # [1, img_size, img_size]
            "dataset":   "kitti",
            "max_depth": self.max_depth,
        }