import torch
import torch.nn as nn

from Models.unfreeze import freeze_all, unfreeze_text_backbone, unfreeze_image_backbone


class GatedCrossModalFusion(nn.Module):
    """
    Gated Cross-Modal Fusion.
    Each modality is conditioned on the other, then gated.
    """
    def __init__(self, text_model, image_model, num_factors=5,
                 unfreeze_text_layers=0, unfreeze_image_layers=0):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model

        freeze_all(text_model, image_model)
        unfreeze_text_backbone(text_model, unfreeze_text_layers)
        unfreeze_image_backbone(image_model, unfreeze_image_layers)

        text_dim  = self.text_model.encoder.config.hidden_size
        image_dim = self.image_model.encoder.num_features
        hidden = 512

        self.text_from_image  = nn.Linear(image_dim, text_dim)
        self.image_from_text  = nn.Linear(text_dim,  image_dim)
        self.text_proj  = nn.Linear(text_dim,  hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.gate       = nn.Linear(text_dim + image_dim, hidden)

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, num_factors)
        )

    def forward(self, input_ids, attention_mask, pixel_values, num_images=None):
        _, text_feat  = self.text_model(input_ids, attention_mask)
        _, image_feat = self.image_model(pixel_values, num_images=num_images)
        text_feat  = text_feat.float()
        image_feat = image_feat.float()

        text_enh  = text_feat  + torch.tanh(self.text_from_image(image_feat))
        image_enh = image_feat + torch.tanh(self.image_from_text(text_feat))

        g = torch.sigmoid(self.gate(torch.cat([text_enh, image_enh], dim=1)))
        fused = g * self.text_proj(text_enh) + (1 - g) * self.image_proj(image_enh)
        return self.head(fused)
