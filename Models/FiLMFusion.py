import torch
import torch.nn as nn

from Models.unfreeze import freeze_all, unfreeze_text_backbone, unfreeze_image_backbone


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Text generates gamma and beta to modulate image features:
        modulated_image = gamma * image + beta
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

        self.film_gamma = nn.Linear(text_dim, image_dim)
        self.film_beta  = nn.Linear(text_dim, image_dim)
        # Standard FiLM init: gamma=1, beta=0 so that at initialization
        # modulated_image = 1 * image + 0 = image (identity modulation).
        # The default Linear init drives gamma ~ N(0, sqrt(1/text_dim)) which
        # near-zeroes the image branch at start and biases the comparison.
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.head = nn.Sequential(
            nn.Linear(text_dim + image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_factors)
        )

    def forward(self, input_ids, attention_mask, pixel_values, num_images=None):
        _, text_feat  = self.text_model(input_ids, attention_mask)
        _, image_feat = self.image_model(pixel_values, num_images=num_images)
        text_feat  = text_feat.float()
        image_feat = image_feat.float()

        gamma = self.film_gamma(text_feat)
        beta  = self.film_beta(text_feat)
        modulated = gamma * image_feat + beta

        fused = torch.cat([text_feat, modulated], dim=1)
        return self.head(fused)
