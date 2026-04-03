# vggt_scale.py  (place at: vggt/models/vggt_scale.py)
#
# Adds:
#   1. A learnable scale_token to VGGT's alternating attention layers.
#   2. LoRA (Low-Rank Adaptation) injected into the aggregator backbone,
#      so the entire VGGT transformer can be fine-tuned at low cost.
#
# Token sequence after modification:
#   [camera×1 | register×4 | **scale×1** | patches...]
#                                ↑ patch_start_idx shifts from 5 → 6
#
# LoRA target layers (per transformer block, both frame_blocks & global_blocks):
#   attn.qkv  · attn.proj  · mlp.fc1  · mlp.fc2
#
# Only LoRA A/B matrices + scale_token + scale_mlp + fuse_conv (+ optionally
# depth_head) are trainable; all original weights stay frozen.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.models.vggt import VGGT


# ══════════════════════════════════════════════════════════════════════════════
# 0.  LoRA primitives
# ══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adaptation:

        y = x @ (W + scale · B @ A)ᵀ + bias

    where  scale = alpha / rank,  A ∈ R^{rank×in},  B ∈ R^{out×rank}.
    The original weight W is frozen; only A and B are trainable.

    Args:
        linear   : the pretrained nn.Linear to wrap
        rank     : LoRA rank  r  (typical: 4, 8, 16, 32)
        alpha    : LoRA scaling alpha (typical: same as rank, or 2×rank)
        dropout  : dropout on the low-rank path (default 0.0)
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.rank         = rank
        self.scale        = alpha / rank

        # Freeze original weight
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA matrices  — always float32 for numerical stability
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Kaiming-uniform init for A  (matches the original LoRA paper)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast A/B to match input dtype (bfloat16 / float32)
        lora_out = (
            self.dropout(x)
            @ self.lora_A.to(x.dtype).T
            @ self.lora_B.to(x.dtype).T
        ) * self.scale
        return F.linear(x, self.weight, self.bias) + lora_out

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear with W' = W + scale·B@A (merged, inference-ready)."""
        merged_w = self.weight + self.scale * (self.lora_B @ self.lora_A)
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        lin.weight.data = merged_w.detach()
        if self.bias is not None:
            lin.bias.data = self.bias.data.clone()
        return lin

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


def _inject_lora_into_block(
    block: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_names: Tuple[str, ...] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
) -> int:
    """
    Replace matching sub-modules in `block` with LoRALinear wrappers in-place.
    Returns the number of layers replaced.
    """
    replaced = 0
    for target in target_names:
        parts  = target.split(".")
        parent = block
        for p in parts[:-1]:
            parent = getattr(parent, p, None)
            if parent is None:
                break
        if parent is None:
            continue
        leaf_name = parts[-1]
        leaf      = getattr(parent, leaf_name, None)
        if isinstance(leaf, nn.Linear):
            setattr(parent, leaf_name, LoRALinear(leaf, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced


def inject_lora(
    aggregator: "AggregatorWithScaleToken",
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_names: Tuple[str, ...] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
) -> int:
    """
    Inject LoRA into every frame_block and global_block of the aggregator.

    Args:
        aggregator   : AggregatorWithScaleToken instance
        rank         : LoRA rank
        alpha        : LoRA alpha (scaling = alpha / rank)
        dropout      : dropout on the LoRA path
        target_names : dotted paths of sub-modules to replace inside each block

    Returns:
        total number of LoRALinear layers created
    """
    total = 0
    for block_list in (aggregator.frame_blocks, aggregator.global_blocks):
        for block in block_list:
            total += _inject_lora_into_block(block, rank, alpha, dropout, target_names)
    return total


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA A/B weights (and scale_token) — compact checkpoint."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k or "scale_token" in k
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  AggregatorWithScaleToken
# ══════════════════════════════════════════════════════════════════════════════

class AggregatorWithScaleToken(Aggregator):
    """
    Subclass of Aggregator that injects one extra learnable scale_token
    between the register tokens and the patch tokens.

    Token layout (per frame, before flattening):
        idx 0      → camera_token
        idx 1..4   → register_tokens  (num_register_tokens = 4)
        idx 5      → scale_token      ← NEW
        idx 6..    → patch_tokens

    Usage:
        new_agg = AggregatorWithScaleToken.from_pretrained_aggregator(
            pretrained_vggt.aggregator
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        embed_dim = kwargs.get("embed_dim", 1024)

        # Scale token — same convention as camera_token: shape [1, 2, 1, D]
        self.scale_token = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        nn.init.normal_(self.scale_token, std=1e-6)

        self.scale_token_idx = self.patch_start_idx  # = 5
        self.patch_start_idx = self.patch_start_idx + 1  # = 6

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained_aggregator(
        cls,
        pretrained_agg: Aggregator,
        img_size: int = 518,
        patch_embed: str = "dinov2_vitl14_reg",
    ) -> "AggregatorWithScaleToken":
        embed_dim           = pretrained_agg.camera_token.shape[-1]
        depth               = pretrained_agg.depth
        patch_size          = pretrained_agg.patch_size
        num_register_tokens = pretrained_agg.register_token.shape[2]
        aa_order            = pretrained_agg.aa_order
        aa_block_size       = pretrained_agg.aa_block_size
        has_rope            = pretrained_agg.rope is not None

        first_block = pretrained_agg.frame_blocks[0]
        num_heads   = getattr(
            getattr(first_block, "attn", first_block), "num_heads", 16
        )

        new_agg = cls(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_register_tokens=num_register_tokens,
            patch_embed=patch_embed,
            aa_order=aa_order,
            aa_block_size=aa_block_size,
            rope_freq=100 if has_rope else -1,
        )

        missing, unexpected = new_agg.load_state_dict(
            pretrained_agg.state_dict(), strict=False
        )
        assert all("scale_token" in k for k in missing), (
            f"Unexpected missing keys (not scale_token): "
            f"{[k for k in missing if 'scale_token' not in k]}"
        )

        return new_agg

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P_patch, C = patch_tokens.shape

        camera_token_e   = slice_expand_and_flatten(self.camera_token,   B, S)
        register_token_e = slice_expand_and_flatten(self.register_token, B, S)
        scale_token_e    = slice_expand_and_flatten(self.scale_token,    B, S)

        tokens = torch.cat(
            [camera_token_e, register_token_e, scale_token_e, patch_tokens], dim=1
        )

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size,
                device=images.device
            )
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2,
                device=images.device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape

        frame_idx  = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = (
                        self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos
                        )
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = (
                        self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat(
                    [frame_intermediates[i], global_intermediates[i]], dim=-1
                )
                output_list.append(concat_inter)

        del concat_inter, frame_intermediates, global_intermediates

        return output_list, self.patch_start_idx


# ══════════════════════════════════════════════════════════════════════════════
# 2.  New head modules
# ══════════════════════════════════════════════════════════════════════════════

class ScaleMLP(nn.Module):
    """
    Estimates a positive scale factor from the scale_token's feature vector.
    Input : [B, 2·embed_dim]   Output: [B, 1]  (positive via Softplus)
    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FuseConv(nn.Module):
    """
    Fuses relative depth + scale into absolute depth via a small residual conv.
    Input:  depth_rel [N,1,H,W],  scale_map [N,1,H,W]
    Output: [N,1,H,W]
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1,  kernel_size=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, depth_rel: torch.Tensor, scale_map: torch.Tensor) -> torch.Tensor:
        scaled  = depth_rel * scale_map
        refined = self.conv(torch.cat([scaled, scale_map], dim=1))
        return scaled + refined


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VGGTScaleDepth — main model
# ══════════════════════════════════════════════════════════════════════════════

class VGGTScaleDepth(nn.Module):
    """
    Wraps a pretrained VGGT and adds absolute-scale depth estimation,
    with optional LoRA fine-tuning of the aggregator backbone.

    Architecture changes vs. stock VGGT
    ─────────────────────────────────────
    • aggregator   → AggregatorWithScaleToken  (scale_token added)
    • [optional]     LoRA injected into every frame_block / global_block
    • scale_mlp    — new, always trainable
    • fuse_conv    — new, always trainable
    • depth_head   — trainable when freeze_depth_head=False

    LoRA trainable params (approximate, rank=8, ViT-L)
    ─────────────────────────────────────────────────────
      24 layers × (qkv 3×1024 + proj 1024 + fc1 4096 + fc2 4096)
      × 2 (frame + global) × rank × 2  ≈  ~10 M extra params

    Args
    ────
    pretrained_vggt   : loaded VGGT instance
    embed_dim         : must match pretrained model (default 1024)
    freeze_backbone   : freeze non-LoRA aggregator weights
    freeze_depth_head : freeze depth_head weights
    scale_per_frame   : predict one scale per frame vs. per scene
    img_size          : positional embedding image size (default 518)
    lora_rank         : LoRA rank r (0 = disable LoRA)
    lora_alpha        : LoRA alpha scaling  (default = lora_rank)
    lora_dropout      : dropout on LoRA path (default 0.05)
    lora_targets      : which sub-layers to inject (dotted paths)
    """

    def __init__(
        self,
        pretrained_vggt: VGGT,
        embed_dim: int = 1024,
        freeze_backbone: bool = True,
        freeze_depth_head: bool = True,
        scale_per_frame: bool = False,
        img_size: int = 518,
        # ── LoRA ──────────────────────────────────────────────────────────
        lora_rank: int = 8,
        lora_alpha: Optional[float] = None,
        lora_dropout: float = 0.05,
        lora_targets: Tuple[str, ...] = (
            "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"
        ),
    ):
        super().__init__()

        assert pretrained_vggt.depth_head is not None, (
            "VGGT must be loaded with enable_depth=True."
        )

        self.lora_rank = lora_rank

        # ── Replace aggregator with scale-token version ───────────────────
        self.aggregator = AggregatorWithScaleToken.from_pretrained_aggregator(
            pretrained_vggt.aggregator, img_size=img_size
        )
        self.scale_token_idx = self.aggregator.scale_token_idx  # = 5

        # ── Keep pretrained depth head ────────────────────────────────────
        self.depth_head = pretrained_vggt.depth_head

        # ── New trainable modules ─────────────────────────────────────────
        dim_agg = 2 * embed_dim
        self.scale_mlp       = ScaleMLP(dim_in=dim_agg)
        self.fuse_conv       = FuseConv()
        self.scale_per_frame = scale_per_frame

        # ── Freeze non-LoRA aggregator weights (except scale_token) ───────
        if freeze_backbone:
            for name, p in self.aggregator.named_parameters():
                if name != "scale_token":
                    p.requires_grad = False

        # ── Inject LoRA into aggregator backbone ──────────────────────────
        if lora_rank > 0:
            alpha   = lora_alpha if lora_alpha is not None else float(lora_rank)
            n_lora  = inject_lora(
                self.aggregator,
                rank=lora_rank,
                alpha=alpha,
                dropout=lora_dropout,
                target_names=lora_targets,
            )
            # LoRALinear keeps original weight frozen, but marks A/B trainable
            # — re-enable requires_grad only on LoRA matrices
            for name, p in self.aggregator.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    p.requires_grad = True

        # ── depth_head: freeze or unfreeze ───────────────────────────────
        for p in self.depth_head.parameters():
            p.requires_grad = not freeze_depth_head

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images : [S, 3, H, W] or [B, S, 3, H, W], values ∈ [0, 1]

        Returns dict:
            depth         [B, S, H, W, 1]
            depth_conf    [B, S, H, W]
            scale_factor  [B, 1] or [B*S, 1]
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        B, S = images.shape[:2]

        # ── 1. Modified aggregator ────────────────────────────────────────
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        # ── 2. Scale token → scale factor ────────────────────────────────
        last_output     = aggregated_tokens_list[-1].float()           # [B, S, P, 2C]
        scale_tok_feats = last_output[:, :, self.scale_token_idx, :]   # [B, S, 2C]

        if self.scale_per_frame:
            scale_input  = scale_tok_feats.reshape(B * S, -1)
            scale_factor = self.scale_mlp(scale_input)    # [B*S, 1]
        else:
            scale_input  = scale_tok_feats.mean(dim=1)    # [B,   2C]
            scale_factor = self.scale_mlp(scale_input)    # [B,   1]

        # ── 3. Relative depth from depth_head ────────────────────────────
        with torch.cuda.amp.autocast(enabled=False):
            depth_rel, depth_conf = self.depth_head(
                [t.float() for t in aggregated_tokens_list],
                images=images.float(),
                patch_start_idx=patch_start_idx,
            )

        # ── 4. Fuse scale + relative depth ────────────────────────────────
        H, W       = depth_rel.shape[2], depth_rel.shape[3]
        depth_flat = depth_rel.squeeze(-1).reshape(B * S, 1, H, W)

        if self.scale_per_frame:
            scale_flat = scale_factor.reshape(B * S, 1, 1, 1).expand(B * S, 1, H, W)
        else:
            scale_flat = (
                scale_factor
                .unsqueeze(1).expand(B, S, 1)
                .reshape(B * S, 1, 1, 1)
                .expand(B * S, 1, H, W)
            )

        depth_abs = self.fuse_conv(depth_flat, scale_flat)   # [B*S, 1, H, W]
        depth_out = depth_abs.reshape(B, S, H, W, 1)

        return {
            "depth":        depth_out,
            "depth_conf":   depth_conf,
            "scale_factor": scale_factor,
        }

    # ── Optimizer helper ──────────────────────────────────────────────────────

    def trainable_parameter_groups(
        self,
        depth_head_lr: float = 1e-6,
        lora_lr: float = 1e-4,
    ) -> list:
        """
        Returns parameter groups for AdamW.

        Recommended learning rates
        ───────────────────────────
          scale_token / scale_mlp / fuse_conv : 1e-3
          LoRA A/B matrices                   : 1e-4
          depth_head (fine-tune)              : 1e-6
        """
        groups = [
            {
                "params": [self.aggregator.scale_token],
                "lr":     1e-3,
                "name":   "scale_token",
            },
            {
                "params": list(self.scale_mlp.parameters()),
                "lr":     1e-3,
                "name":   "scale_mlp",
            },
            {
                "params": list(self.fuse_conv.parameters()),
                "lr":     1e-3,
                "name":   "fuse_conv",
            },
        ]

        # LoRA parameters from the aggregator backbone
        if self.lora_rank > 0:
            lora_params = [
                p for n, p in self.aggregator.named_parameters()
                if ("lora_A" in n or "lora_B" in n) and p.requires_grad
            ]
            if lora_params:
                groups.append({
                    "params": lora_params,
                    "lr":     lora_lr,
                    "name":   "lora",
                })

        # depth_head — only when unfrozen
        depth_head_params = [
            p for p in self.depth_head.parameters() if p.requires_grad
        ]
        if depth_head_params:
            groups.append({
                "params": depth_head_params,
                "lr":     depth_head_lr,
                "name":   "depth_head",
            })

        return groups

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def lora_state_dict(self) -> dict:
        """
        Returns a compact state dict containing only:
          • LoRA A/B matrices (aggregator)
          • scale_token
          • scale_mlp
          • fuse_conv
          • depth_head  (if trainable)
        Suitable for saving as a diff on top of the base VGGT weights.
        """
        sd = {}
        for k, v in self.state_dict().items():
            keep = (
                "lora_A" in k
                or "lora_B" in k
                or "scale_token" in k
                or k.startswith("scale_mlp.")
                or k.startswith("fuse_conv.")
            )
            if not keep:
                # include depth_head only when it was fine-tuned
                depth_head_trainable = any(
                    p.requires_grad for p in self.depth_head.parameters()
                )
                if depth_head_trainable and k.startswith("depth_head."):
                    keep = True
            if keep:
                sd[k] = v
        return sd

    def merge_lora(self):
        """
        Merge LoRA weights into the original weight matrices in-place
        (for inference without LoRALinear overhead).
        After calling this, all LoRALinear layers become plain nn.Linear.
        """
        if self.lora_rank <= 0:
            return
        for block_list in (
            self.aggregator.frame_blocks,
            self.aggregator.global_blocks,
        ):
            for block in block_list:
                for name, module in list(block.named_modules()):
                    if isinstance(module, LoRALinear):
                        # Navigate to parent
                        parts  = name.split(".")
                        parent = block
                        for p in parts[:-1]:
                            parent = getattr(parent, p)
                        setattr(parent, parts[-1], module.merge())
        self.lora_rank = 0  # mark as merged


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Loss
# ══════════════════════════════════════════════════════════════════════════════

def scale_depth_loss(
    depth_pred:   torch.Tensor,
    depth_gt:     torch.Tensor,
    scale_factor: torch.Tensor,
    scale_gt:     torch.Tensor = None,
    conf:         torch.Tensor = None,
    lambda_scale: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = depth_pred.squeeze(-1)
    gt   = depth_gt.squeeze(-1)
    mask = (gt > eps) & (pred > eps)

    w = conf[mask] if conf is not None else torch.ones(mask.sum(), device=pred.device)

    log_diff = torch.log(pred[mask] + eps) - torch.log(gt[mask] + eps)
    silog    = (w * log_diff ** 2).sum() / w.sum() \
             - 0.5 * ((w * log_diff).sum() / w.sum()) ** 2

    loss = silog
    if scale_gt is not None:
        loss = loss + lambda_scale * torch.abs(scale_factor - scale_gt).mean()

    return loss


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading pretrained VGGT...")
    pretrained = VGGT.from_pretrained("facebook/VGGT-1B")
    pretrained = pretrained.cuda().eval()

    print("Building VGGTScaleDepth (lora_rank=8, freeze_depth_head=False)...")
    model = VGGTScaleDepth(
        pretrained_vggt=pretrained,
        embed_dim=1024,
        freeze_backbone=True,
        freeze_depth_head=False,
        scale_per_frame=False,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
    ).cuda()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total:     {total  / 1e6:.1f}M")
    print(f"Trainable: {trainable / 1e6:.3f}M")
    for g in model.trainable_parameter_groups():
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:20s}: {n / 1e6:.3f}M  lr={g['lr']}")

    print(f"\nscale_token_idx : {model.scale_token_idx}")
    print(f"patch_start_idx : {model.aggregator.patch_start_idx}")

    # Count LoRA layers injected
    lora_count = sum(
        1 for m in model.aggregator.modules() if isinstance(m, LoRALinear)
    )
    print(f"LoRA layers     : {lora_count}")

    dummy = torch.rand(2, 4, 3, 518, 518).cuda().to(torch.bfloat16)
    with torch.no_grad():
        out = model(dummy)

    print("\ndepth      :", out["depth"].shape)
    print("depth_conf :", out["depth_conf"].shape)
    print("scale      :", out["scale_factor"].shape)
    print("scale values:", out["scale_factor"].squeeze())

    lora_sd = model.lora_state_dict()
    print(f"\nLoRA checkpoint keys: {len(lora_sd)}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(), weight_decay=1e-4
    )
    print("Optimizer built successfully.")
    print("\nDone.")