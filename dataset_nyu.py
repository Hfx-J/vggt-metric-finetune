# dataset_nyu.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class NYUv2Dataset(Dataset):
    def __init__(self, mat_path, split="train", img_size=518):
        super().__init__()
        self.img_size = img_size

        print(f"Loading NYUv2 from {mat_path}...")
        with h5py.File(mat_path, 'r') as f:
            images_raw = np.array(f['images'])  # 先读出来看shape
            depths_raw = np.array(f['depths'])

            print(f"Raw images shape: {images_raw.shape}")  # 打印出来确认
            print(f"Raw depths shape: {depths_raw.shape}")

            # ── 根据实际 shape 自动选择正确的 transpose ──
            # 标准 h5py 读出来通常是 (3, 640, 480, 1449)
            # 有些版本是 (1449, 480, 640, 3) 已经是正确顺序
            if images_raw.ndim == 4:
                if images_raw.shape[0] == 3:
                    # (3, W, H, N) → (N, H, W, 3)
                    images = images_raw.transpose(3, 2, 1, 0)
                elif images_raw.shape[-1] == 3:
                    # (N, H, W, 3) 已经是正确顺序
                    images = images_raw
                elif images_raw.shape[1] == 3:
                    # (N, 3, H, W) → (N, H, W, 3)
                    images = images_raw.transpose(0, 2, 3, 1)
                else:
                    raise ValueError(f"Unrecognized images shape: {images_raw.shape}")
            else:
                raise ValueError(f"Expected 4D images array, got {images_raw.ndim}D")

            if depths_raw.ndim == 3:
                if depths_raw.shape[0] != images.shape[0]:
                    # (H, W, N) → (N, H, W)
                    depths = depths_raw.transpose(2, 1, 0)
                else:
                    # (N, H, W) 已经正确
                    depths = depths_raw
            else:
                raise ValueError(f"Expected 3D depths array, got {depths_raw.ndim}D")

            print(f"After transpose - images: {images.shape}, depths: {depths.shape}")
            # 正确结果应该是：images (1449, H, W, 3), depths (1449, H, W)

        # 按官方划分
        n_total = images.shape[0]
        n_train = int(n_total * 0.55)  # 约795张

        if split == "train":
            self.images = images[:n_train]
            self.depths = depths[:n_train]
        else:
            self.images = images[n_train:]
            self.depths = depths[n_train:]

        print(f"Loaded {len(self.images)} samples for {split}")

        self.img_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)       # [H, W, 3]
        depth = self.depths[idx].astype(np.float32)   # [H, W]

        img_tensor = self.img_transform(img)           # [3, 518, 518]

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False
        ).squeeze(0)  # [1, 518, 518]

        return {
            "image": img_tensor,
            "depth": depth_tensor,
        }