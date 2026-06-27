import torch
import torch.nn as nn
import timm


class ImageModel(nn.Module):
    def __init__(self, model_name='convnext_base', num_factors=5):
        super(ImageModel, self).__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)

        self.fc = nn.Sequential(
            nn.Linear(self.encoder.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.factor_head = nn.Linear(256, num_factors)

    def forward(self, pixel_values, num_images=None):
        # Unify 4D to 5D for consistent processing
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)

        B, N, C, H, W = pixel_values.shape
        features = self.encoder(pixel_values.reshape(B * N, C, H, W)).reshape(B, N, -1)
        
        # Average pooling
        if num_images is not None:
            mask = (torch.arange(N, device=pixel_values.device).expand(B, N) < num_images.unsqueeze(1)).float().unsqueeze(-1)
            features = (features * mask).sum(dim=1) / num_images.float().clamp(min=1).unsqueeze(1)
        else:
            features = features.mean(dim=1)

        features = features.to(torch.float32)
        return self.factor_head(self.fc(features)), features

    # ------------------------------------------------------------------
    # Patch-token path (used by CrossAttentionFusion).
    # Returns image patch tokens aggregated across the real images of
    # each sample, plus a validity mask for the padded positions.
    # ------------------------------------------------------------------
    def forward_features(self, pixel_values, num_images=None):
        """Return (patches, patch_mask).

        patches:    (B, P, D) - patch tokens pooled across real images
        patch_mask: (B, P) bool - True for real patch positions

        Supports two timm output layouts:
        - 4D spatial map  (B*N, D, h, w) or (B*N, h, w, D): ConvNeXt, Swin, EfficientNet
        - 3D token sequence (B*N, num_tokens, D): ViT, SigLIP and other isotropic models
        """
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        B, N, C, H, W = pixel_values.shape
        if num_images is None:
            num_images = torch.full((B,), N, dtype=torch.long, device=pixel_values.device)

        # Raw features for every image slot (incl. padding images).
        feats = self.encoder.forward_features(pixel_values.view(B * N, C, H, W))

        D = self.encoder.num_features

        if feats.dim() == 3:
            # ViT / SigLIP: output is (B*N, num_tokens, D) — token sequence.
            # Treat each token as a "patch" position directly.
            P = feats.shape[1]
            feats = feats.reshape(B, N, P, D)
        elif feats.dim() == 4:
            # ConvNeXt / Swin / EfficientNet: spatial feature map.
            # Normalise to channels-first (B*N, D, h, w).
            #   Swin         -> (B*N, H, W, D) channels-last  [D == last dim]
            #   ConvNeXt     -> (B*N, D, H, W) channels-first [D == dim 1]
            #   EfficientNet -> (B*N, D, H, W) channels-first
            if feats.shape[1] != D and feats.shape[-1] == D:
                feats = feats.permute(0, 3, 1, 2).contiguous()
            h, w = feats.shape[2], feats.shape[3]
            P = h * w
            feats = feats.reshape(B, N, P, D)
        else:
            raise ValueError(
                f"[ImageModel.forward_features] Unexpected feature ndim={feats.dim()} "
                f"from {type(self.encoder).__name__}. Expected 3 (ViT) or 4 (CNN)."
            )

        # Average patch tokens across the REAL images of each sample
        # (ignore padding/black images). Result: (B, P, D).
        img_mask = (torch.arange(N, device=pixel_values.device).expand(B, N) < num_images.unsqueeze(1)).float().unsqueeze(-1)  # (B, N, 1)
        feats = (feats * img_mask.unsqueeze(-1)).sum(dim=1) / num_images.float().clamp(min=1).unsqueeze(-1).unsqueeze(-1)
        feats = feats.to(torch.float32)

        patch_mask = torch.ones(B, P, dtype=torch.bool, device=feats.device)
        return feats, patch_mask
