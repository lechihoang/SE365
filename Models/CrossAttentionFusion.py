import torch
import torch.nn as nn

from Models.unfreeze import freeze_all, unfreeze_text_backbone, unfreeze_image_backbone


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion (token <-> patch).

    Text tokens (B, T, d_t) attend to image patch tokens (B, P, d_i),
    and image patches attend back to text tokens — bidirectional, with
    proper padding masks on both sides. This is real cross-attention, not
    the degenerate 1-vector-per-modality variant.
    """
    def __init__(self, text_model, image_model, num_factors=5,
                 unfreeze_text_layers=0, unfreeze_image_layers=0, num_heads=8):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model

        freeze_all(text_model, image_model)
        unfreeze_text_backbone(text_model, unfreeze_text_layers)
        unfreeze_image_backbone(image_model, unfreeze_image_layers)

        text_dim  = self.text_model.encoder.config.hidden_size
        image_dim = self.image_model.encoder.num_features
        hidden = 512

        # Keep these names so old checkpoints partially map.
        self.text_proj  = nn.Linear(text_dim,  hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.cross_attn_t2i = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=0.1)
        self.cross_attn_i2t = nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=0.1)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_factors)
        )

    @staticmethod
    def _masked_mean(x, keep_mask):
        # x: (B, L, D), keep_mask: (B, L) bool, True == real
        m = keep_mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, input_ids, attention_mask, pixel_values, num_images=None):
        # Token-level text features + pad mask (True == PAD).
        _, _, text_tokens, text_pad = self.text_model(input_ids, attention_mask, return_tokens=True)
        # Patch-level image features + valid mask (True == real patch).
        image_patches, patch_mask = self.image_model.forward_features(pixel_values, num_images=num_images)

        text_tokens   = text_tokens.float()
        image_patches = image_patches.float()

        t = self.text_proj(text_tokens)     # (B, T, hidden)
        i = self.image_proj(image_patches)  # (B, P, hidden)

        # key_padding_mask: True == position is PAD and must be ignored.
        t_kpm = text_pad                         # (B, T) True==PAD
        i_kpm = ~patch_mask                      # (B, P) True==PAD

        t_out, _ = self.cross_attn_t2i(query=t, key=i, value=i, key_padding_mask=i_kpm)
        i_out, _ = self.cross_attn_i2t(query=i, key=t, value=t, key_padding_mask=t_kpm)

        # Masked mean pool over the attended representations.
        t_keep = ~t_kpm
        t_pooled = self._masked_mean(t_out, t_keep)
        i_pooled = self._masked_mean(i_out, patch_mask)

        fused = torch.cat([t_pooled, i_pooled], dim=1)
        return self.head(fused)
