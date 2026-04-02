# metric_adapter.py
# 放在 vggt/ 根目录下（和 requirements.txt 同级）

import torch
import torch.nn as nn

class MetricDepthAdapter(nn.Module):
    """
    在冻结的 VGGT 仿射不变深度之上，学习 metric scale。
    原理：metric_depth = exp(log_scale) * affine_depth + shift
    """
    def __init__(self, max_depth=10.0):
        super().__init__()
        self.max_depth = max_depth
        # 可学习参数：尺度（log空间初始化为0，即scale=1）
        self.log_scale = nn.Parameter(torch.zeros(1))
        # 可学习参数：偏移（初始化为0）
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, affine_depth):
        """
        Args:
            affine_depth: [B, S, 1, H, W] 或 [B, S, H, W]，来自 model.depth_head 的原始输出
        Returns:
            metric_depth: 同shape，单位为米
        """
        metric = torch.exp(self.log_scale) * affine_depth + self.shift
        # 截断到合理范围，防止训练初期爆炸
        return metric.clamp(min=0.001, max=self.max_depth)