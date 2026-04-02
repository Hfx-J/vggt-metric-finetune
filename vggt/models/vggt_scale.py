# vggt_scale_depth_v2.py
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
from torch.utils.checkpoint import checkpoint

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
        # Infer architecture from stored attributes
        embed_dim          = pretrained_agg.camera_token.shape[-1]
        depth              = pretrained_agg.depth
        patch_size         = pretrained_agg.patch_size
        num_register_tokens = pretrained_agg.register_token.shape[2]
        aa_order           = pretrained_agg.aa_order
        aa_block_size      = pretrained_agg.aa_block_size
        has_rope           = pretrained_agg.rope is not None

        # Infer num_heads from the first frame block
        first_block = pretrained_agg.frame_blocks[0]
        # Most block implementations store num_heads on the attn sub-module
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

        # Copy all pretrained weights; scale_token will be missing from the
        # pretrained state dict → reported in `missing_keys` but that's fine.
        missing, unexpected = new_agg.load_state_dict(
            pretrained_agg.state_dict(), strict=False
        )
        assert all("scale_token" in k for k in missing), (
            f"Unexpected missing keys (not scale_token): {[k for k in missing if 'scale_token' not in k]}"
        )

        return new_agg

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Identical to Aggregator.forward() except scale_token is prepended
        between register_token and patch_tokens.

        Returns:
            output_list     : list of [B, S, P+1, 2·C] tensors   (P+1 because of scale_token)
            patch_start_idx : 6 (updated to reflect the extra scale token slot)
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)

        # Patch embedding
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P_patch, C = patch_tokens.shape

        # Expand special tokens → [B*S, *, C]
        camera_token_e   = slice_expand_and_flatten(self.camera_token,   B, S)
        register_token_e = slice_expand_and_flatten(self.register_token, B, S)
        scale_token_e    = slice_expand_and_flatten(self.scale_token,    B, S)  # NEW

        # Concatenate: [camera | register | scale | patches]
        tokens = torch.cat(
            [camera_token_e, register_token_e, scale_token_e, patch_tokens], dim=1
        )

        # Rotary position embedding
        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size,
                device=images.device
            )
            # Shift patch positions by +1 (away from 0-origin), then zero-pad
            # for all special tokens (camera, register, scale).
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2,
                device=images.device, dtype=pos.dtype
            )
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape   # P = 1 + 4 + 1 + P_patch = 6 + P_patch

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
                # [B, S, P, 2C]  (frame ‖ global concatenation, same as original)
                concat_inter = torch.cat(
                    [frame_intermediates[i], global_intermediates[i]], dim=-1
                )
                output_list.append(concat_inter)

        del concat_inter, frame_intermediates, global_intermediates

        # patch_start_idx = 6, correctly signals DPT head where patches begin
        return output_list, self.patch_start_idx


# ──────────────────────────────────────────────────────────────────────────────
# 2. New head modules
# ──────────────────────────────────────────────────────────────────────────────

class ScaleMLP(nn.Module):
    """
    Estimates a positive scale factor from the scale_token's feature vector.

    Input : [B, 2·embed_dim]  (the scale token's last-layer representation,
                               after concatenating frame and global features)
    Output: [B, 1]            (positive scalar via Softplus)
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
        return self.net(x)   # [B, 1]


class FuseConv(nn.Module):
    """
    Fuses relative depth + scale into an absolute depth map using a small
    residual conv.  Residual design means the conv starts as a near-identity
    (predicted increment ≈ 0 at init), so training is stable from day one.

    Input:
        depth_rel  [N, 1, H, W]  relative depth from DPT head
        scale_map  [N, 1, H, W]  broadcasted scale factor
    Output:
        [N, 1, H, W]  absolute depth
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
        # Zero-init last conv so the residual starts as identity
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(
        self,
        depth_rel: torch.Tensor,
        scale_map: torch.Tensor,
    ) -> torch.Tensor:
        scaled  = depth_rel * scale_map
        refined = self.conv(torch.cat([scaled, scale_map], dim=1))
        return scaled + refined   # residual


# ──────────────────────────────────────────────────────────────────────────────
# 3. VGGTScaleDepth — the main model
# ──────────────────────────────────────────────────────────────────────────────

class VGGTScaleDepth(nn.Module):
    """
    Wraps a pretrained VGGT and adds absolute-scale depth estimation.

    Architecture changes vs. stock VGGT:
        • Aggregator  → AggregatorWithScaleToken  (scale_token added, rest frozen)
        • scale_mlp   — new, trainable
        • fuse_conv   — new, trainable
        • depth_head  — unchanged, frozen by default

    Trainable parameter count is tiny (< 1M for embed_dim=1024).

    Args:
        pretrained_vggt  : loaded VGGT instance (weights already on device)
        embed_dim        : must match pretrained model (default 1024 for VGGT-1B)
        freeze_backbone  : freeze depth_head and all aggregator weights except
                           scale_token  (recommended for initial experiments)
        scale_per_frame  : if True, predict one scale per frame rather than
                           one per scene/batch
        img_size         : image size used for positional embedding init (default 518)
    """

    def __init__(
        self,
        pretrained_vggt: VGGT,
        embed_dim: int = 1024,
        freeze_backbone: bool = True,
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
        dim_agg = 2 * embed_dim          # aggregator outputs concat(frame, global) = 2C
        self.scale_mlp  = ScaleMLP(dim_in=dim_agg)
        self.fuse_conv  = FuseConv()
        self.scale_per_frame = scale_per_frame

        # ── Freeze pretrained weights ─────────────────────────────────────
        if freeze_backbone:
            # Freeze depth_head entirely
            for p in self.depth_head.parameters():
                p.requires_grad = False

            # Freeze all aggregator weights EXCEPT the new scale_token
            for name, p in self.aggregator.named_parameters():
                if name != "scale_token":
                    p.requires_grad = False

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images : [S, 3, H, W] or [B, S, 3, H, W], values ∈ [0, 1]

        Returns dict:
            depth         [B, S, H, W, 1]   absolute depth
            depth_conf    [B, S, H, W]       confidence from DPT head
            scale_factor  [B, 1] or [B, S, 1]
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        B, S = images.shape[:2]

        # ── 1. Run modified aggregator ────────────────────────────────────
        # output_list : list of [B, S, P, 2C], P = 6 + num_patches
        # patch_start_idx = 6
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        # ── 2. Extract scale token features from the last layer ───────────
        # last_output : [B, S, P, 2C]
        last_output = aggregated_tokens_list[-1]

        # Scale token is at fixed position self.scale_token_idx in every frame
        # shape: [B, S, 2C]
        scale_tok_feats = last_output[:, :, self.scale_token_idx, :]

        if self.scale_per_frame:
            # One scale per (batch, frame) — reshape [B*S, 2C] → [B*S, 1]
            scale_input  = scale_tok_feats.reshape(B * S, -1)
            scale_factor = self.scale_mlp(scale_input)          # [B*S, 1]
        else:
            # One scale per scene — mean-pool over S frames first
            scale_input  = scale_tok_feats.mean(dim=1)          # [B, 2C]
            scale_factor = self.scale_mlp(scale_input)          # [B, 1]

        # ── 3. Relative depth from frozen DPT head ────────────────────────
        with torch.cuda.amp.autocast(enabled=False):
            depth_rel, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
        # depth_rel : [B, S, H, W, 1]

        # ── 4. Fuse scale + relative depth ────────────────────────────────
        H, W = depth_rel.shape[2], depth_rel.shape[3]

        # Flatten to [B*S, 1, H, W] for conv
        depth_flat = depth_rel.squeeze(-1).reshape(B * S, 1, H, W)

        if self.scale_per_frame:
            scale_flat = scale_factor.reshape(B * S, 1, 1, 1).expand(B * S, 1, H, W)
        else:
            scale_flat = (
                scale_factor                                     # [B, 1]
                .unsqueeze(1).expand(B, S, 1)                   # [B, S, 1]
                .reshape(B * S, 1, 1, 1)
                .expand(B * S, 1, H, W)
            )

        depth_abs  = self.fuse_conv(depth_flat, scale_flat)     # [B*S, 1, H, W]
        depth_out  = depth_abs.reshape(B, S, H, W, 1)

        return {
            "depth":        depth_out,
            "depth_conf":   depth_conf,
            "scale_factor": scale_factor,
        }

    # ── Optimizer helper ──────────────────────────────────────────────────────

    def trainable_parameter_groups(self, backbone_lr: float = 1e-5) -> list:
        """
        Returns parameter groups suitable for AdamW.

        Suggested learning rates:
            new params (scale_token, scale_mlp, fuse_conv) : 1e-3
            backbone fine-tune (optional)                   : 1e-5
        """
        return [
            {
                "params":  [self.aggregator.scale_token],
                "lr":      1e-3,
                "name":    "scale_token",
            },
            {
                "params":  list(self.scale_mlp.parameters()),
                "lr":      1e-3,
                "name":    "scale_mlp",
            },
            {
                "params":  list(self.fuse_conv.parameters()),
                "lr":      1e-3,
                "name":    "fuse_conv",
            },
            # Uncomment to fine-tune backbone blocks at lower LR:
            # {
            #     "params":  list(self.aggregator.frame_blocks.parameters()) +
            #                list(self.aggregator.global_blocks.parameters()),
            #     "lr":      backbone_lr,
            #     "name":    "backbone_blocks",
            # },
        ]


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
    Combines:
        SILog loss  — scale-invariant log depth, standard for monocular depth
        Scale L1    — optional direct scale supervision when GT is available
        Confidence  — optional confidence weighting from DPT head

    Args:
        depth_pred   [B, S, H, W, 1]
        depth_gt     [B, S, H, W, 1]   (values <= 0 are treated as invalid)
        scale_factor [B, 1] or [B*S, 1]
        scale_gt     [B, 1] optional
        conf         [B, S, H, W]  optional confidence weights
    """
    pred = depth_pred.squeeze(-1)   # [B, S, H, W]
    gt   = depth_gt.squeeze(-1)

    mask = (gt > eps) & (pred > eps)

    if conf is not None:
        w = conf[mask]
    else:
        w = torch.ones(mask.sum(), device=pred.device)

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

    print("Building VGGTScaleDepth...")
    model = VGGTScaleDepth(
        pretrained_vggt=pretrained,
        embed_dim=1024,
        freeze_backbone=True,
        scale_per_frame=False,
    ).cuda()

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,}  /  Total: {total:,}  "
          f"({100 * trainable / total:.2f}%)")

    # Token layout check
    print(f"scale_token_idx : {model.scale_token_idx}")   # should be 5
    print(f"patch_start_idx : {model.aggregator.patch_start_idx}")   # should be 6

    # Forward
    dummy = torch.rand(2, 4, 3, 518, 518).cuda()  # B=2, S=4 frames
    with torch.no_grad():
        out = model(dummy)

    print("depth      :", out["depth"].shape)        # [2, 4, H, W, 1]
    print("depth_conf :", out["depth_conf"].shape)   # [2, 4, H, W]
    print("scale      :", out["scale_factor"].shape) # [2, 1]
    print("scale values:", out["scale_factor"].squeeze())

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.trainable_parameter_groups(),
        weight_decay=1e-4,
    )
    print("Optimizer built successfully.")
    print("\nDone.")