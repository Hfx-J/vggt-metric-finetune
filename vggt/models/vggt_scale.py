# vggt_scale.py  (place at: vggt/models/vggt_scale.py)
#
# Adds a learnable scale_token to VGGT's alternating attention layers.
# Token sequence after modification:
#   [camera×1 | register×4 | **scale×1** | patches...]
#                                ↑ patch_start_idx shifts from 5 → 6
#
# The scale token participates in BOTH frame-attention and global-attention,
# so by the last layer it has aggregated info from every frame in the scene.
# Its final representation is fed into ScaleMLP → scale_factor.

import torch
import torch.nn as nn
from typing import List, Tuple

from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.models.vggt import VGGT


# ──────────────────────────────────────────────────────────────────────────────
# 1. AggregatorWithScaleToken
# ──────────────────────────────────────────────────────────────────────────────

class AggregatorWithScaleToken(Aggregator):
    """
    Subclass of Aggregator that injects one extra learnable scale_token
    between the register tokens and the patch tokens.

    Token layout (per frame, before flattening):
        idx 0      → camera_token
        idx 1..4   → register_tokens  (num_register_tokens = 4)
        idx 5      → scale_token      ← NEW
        idx 6..    → patch_tokens

    All existing _process_frame_attention / _process_global_attention methods
    are inherited unchanged. The scale token rides along as a regular token
    and participates in every attention block automatically.

    Usage:
        new_agg = AggregatorWithScaleToken.from_pretrained_aggregator(
            pretrained_vggt.aggregator
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        embed_dim = kwargs.get("embed_dim", 1024)

        # Scale token — same convention as camera_token: shape [1, 2, 1, D]
        #   dim-1 index 0 : representation used for the FIRST frame
        #   dim-1 index 1 : representation used for ALL OTHER frames
        # slice_expand_and_flatten() handles the per-frame expansion.
        self.scale_token = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
        nn.init.normal_(self.scale_token, std=1e-6)

        # Where the scale token lives in the token sequence
        self.scale_token_idx = self.patch_start_idx  # = 5

        # Patches now start one position later
        self.patch_start_idx = self.patch_start_idx + 1  # = 6

    # ── Factory: build from a pretrained Aggregator ──────────────────────────

    @classmethod
    def from_pretrained_aggregator(
        cls,
        pretrained_agg: Aggregator,
        img_size: int = 518,
        patch_embed: str = "dinov2_vitl14_reg",
    ) -> "AggregatorWithScaleToken":
        """
        Builds an AggregatorWithScaleToken whose weights are copied from
        a pretrained Aggregator.  Only `scale_token` starts from scratch.

        Args:
            pretrained_agg : the .aggregator attribute of a pretrained VGGT model
            img_size       : must match what VGGT was trained with (default 518)
            patch_embed    : patch-embed type string used in the original model
        """
        embed_dim           = pretrained_agg.camera_token.shape[-1]
        depth               = pretrained_agg.depth
        patch_size          = pretrained_agg.patch_size
        num_register_tokens = pretrained_agg.register_token.shape[2]
        aa_order            = pretrained_agg.aa_order
        aa_block_size       = pretrained_agg.aa_block_size
        has_rope            = pretrained_agg.rope is not None

        first_block = pretrained_agg.frame_blocks[0]
        num_heads = getattr(
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
        """
        Identical to Aggregator.forward() except scale_token is prepended
        between register_token and patch_tokens.

        Returns:
            output_list     : list of [B, S, P+1, 2·C] tensors
            patch_start_idx : 6
        """
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


# ──────────────────────────────────────────────────────────────────────────────
# 2. New head modules
# ──────────────────────────────────────────────────────────────────────────────

class ScaleMLP(nn.Module):
    """
    Estimates a positive scale factor from the scale_token's feature vector.

    Input : [B, 2·embed_dim]
    Output: [B, 1]  (positive scalar via Softplus)
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

    Input:
        depth_rel  [N, 1, H, W]
        scale_map  [N, 1, H, W]
    Output:
        [N, 1, H, W]
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(
        self,
        depth_rel: torch.Tensor,
        scale_map: torch.Tensor,
    ) -> torch.Tensor:
        scaled  = depth_rel * scale_map
        refined = self.conv(torch.cat([scaled, scale_map], dim=1))
        return scaled + refined


# ──────────────────────────────────────────────────────────────────────────────
# 3. VGGTScaleDepth — the main model
# ──────────────────────────────────────────────────────────────────────────────

class VGGTScaleDepth(nn.Module):
    """
    Wraps a pretrained VGGT and adds absolute-scale depth estimation.

    Architecture changes vs. stock VGGT:
        • Aggregator       → AggregatorWithScaleToken  (scale_token added)
        • scale_mlp        — new, always trainable
        • fuse_conv        — new, always trainable
        • depth_head       — trainable when freeze_depth_head=False

    Args:
        pretrained_vggt   : loaded VGGT instance
        embed_dim         : must match pretrained model (default 1024)
        freeze_backbone   : freeze aggregator weights except scale_token
        freeze_depth_head : freeze depth_head weights (set False to finetune)
        scale_per_frame   : predict one scale per frame vs. per scene
        img_size          : image size for positional embedding (default 518)
    """

    def __init__(
        self,
        pretrained_vggt: VGGT,
        embed_dim: int = 1024,
        freeze_backbone: bool = True,
        freeze_depth_head: bool = True,
        scale_per_frame: bool = False,
        img_size: int = 518,
    ):
        super().__init__()

        assert pretrained_vggt.depth_head is not None, (
            "VGGT must be loaded with enable_depth=True."
        )

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

        # ── Freeze pretrained aggregator weights (except scale_token) ─────
        if freeze_backbone:
            for name, p in self.aggregator.named_parameters():
                if name != "scale_token":
                    p.requires_grad = False

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
            scale_input  = scale_tok_feats.mean(dim=1)    # [B, 2C]
            scale_factor = self.scale_mlp(scale_input)    # [B, 1]

        # ── 3. Relative depth from depth_head ────────────────────────────
        # Cast to float32: depth_head's LayerNorm requires float32 inputs.
        with torch.cuda.amp.autocast(enabled=False):
            depth_rel, depth_conf = self.depth_head(
                [t.float() for t in aggregated_tokens_list],
                images=images.float(),
                patch_start_idx=patch_start_idx,
            )
        # depth_rel : [B, S, H, W, 1]

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
        backbone_lr: float = 1e-5,
        depth_head_lr: float = 1e-5,
    ) -> list:
        """
        Returns parameter groups for AdamW.

        Learning rate guidelines:
            scale_token / scale_mlp / fuse_conv : 1e-3
            depth_head (finetune)               : 1e-5
            backbone blocks (optional)          : 1e-5
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

        # depth_head — only added when it has trainable parameters
        depth_head_params = [p for p in self.depth_head.parameters() if p.requires_grad]
        if depth_head_params:
            groups.append({
                "params": depth_head_params,
                "lr":     depth_head_lr,
                "name":   "depth_head",
            })

        # Uncomment to fine-tune backbone blocks at lower LR:
        # groups.append({
        #     "params": list(self.aggregator.frame_blocks.parameters()) +
        #               list(self.aggregator.global_blocks.parameters()),
        #     "lr":     backbone_lr,
        #     "name":   "backbone_blocks",
        # })

        return groups


# ──────────────────────────────────────────────────────────────────────────────
# 4. Loss
# ──────────────────────────────────────────────────────────────────────────────

def scale_depth_loss(
    depth_pred:   torch.Tensor,
    depth_gt:     torch.Tensor,
    scale_factor: torch.Tensor,
    scale_gt:     torch.Tensor = None,
    conf:         torch.Tensor = None,
    lambda_scale: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Combines SILog + optional scale L1 + optional confidence weighting.

    Args:
        depth_pred   [B, S, H, W, 1]
        depth_gt     [B, S, H, W, 1]   (values <= 0 treated as invalid)
        scale_factor [B, 1] or [B*S, 1]
        scale_gt     [B, 1] optional
        conf         [B, S, H, W] optional
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# 5. Minimal sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading pretrained VGGT...")
    pretrained = VGGT.from_pretrained("facebook/VGGT-1B")
    pretrained = pretrained.cuda().eval()

    print("Building VGGTScaleDepth (freeze_depth_head=False)...")
    model = VGGTScaleDepth(
        pretrained_vggt=pretrained,
        embed_dim=1024,
        freeze_backbone=True,
        freeze_depth_head=False,
        scale_per_frame=False,
    ).cuda()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total:     {total / 1e6:.1f}M")
    print(f"Trainable: {trainable / 1e6:.3f}M")
    for g in model.trainable_parameter_groups():
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:20s}: {n / 1e6:.3f}M  lr={g['lr']}")

    print(f"scale_token_idx : {model.scale_token_idx}")
    print(f"patch_start_idx : {model.aggregator.patch_start_idx}")

    dummy = torch.rand(2, 4, 3, 518, 518).cuda()
    with torch.no_grad():
        out = model(dummy)

    print("depth      :", out["depth"].shape)
    print("depth_conf :", out["depth_conf"].shape)
    print("scale      :", out["scale_factor"].shape)
    print("scale values:", out["scale_factor"].squeeze())

    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(), weight_decay=1e-4
    )
    print("Optimizer built successfully.")
    print("\nDone.")