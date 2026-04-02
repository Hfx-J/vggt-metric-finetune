# infer_metric_depth.py
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from metric_adapter import MetricDepthAdapter
import numpy as np

device = "cuda"
dtype = torch.bfloat16

# 加载模型
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).to(dtype)
model.eval()

# 加载 adapter
adapter = MetricDepthAdapter(max_depth=10.0).to(device)
ckpt = torch.load("./checkpoints/best_adapter.pth")
adapter.load_state_dict(ckpt["adapter_state"])
adapter.eval()

# 推理
image_paths = ["your_image.jpg"]
images = load_and_preprocess_images(image_paths).to(device)  # [S, 3, H, W]

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images_batch = images.unsqueeze(0)  # [1, S, 3, H, W]
        agg_tokens, ps_idx = model.aggregator(images_batch)
        depth_map, depth_conf = model.depth_head(agg_tokens, images_batch, ps_idx)

    affine_depth = depth_map.squeeze(2).float()   # [1, S, H, W]
    metric_depth = adapter(affine_depth)           # [1, S, H, W]，单位米

print(f"Depth range: {metric_depth.min():.2f}m ~ {metric_depth.max():.2f}m")