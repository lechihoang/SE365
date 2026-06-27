import torch
import torch.nn as nn

from Models.unfreeze import freeze_all, unfreeze_text_backbone, unfreeze_image_backbone


class FusionModel(nn.Module):
    def __init__(self, text_model, image_model, num_factors=5, unfreeze_text_layers=0, unfreeze_image_layers=0):
        super(FusionModel, self).__init__()
        self.text_model = text_model
        self.image_model = image_model

        # Đóng băng trọng số mặc định, sau đó mở khóa chọn lọc
        freeze_all(text_model, image_model)
        unfreeze_text_backbone(text_model, unfreeze_text_layers)
        unfreeze_image_backbone(image_model, unfreeze_image_layers)

        fusion_size = self.text_model.encoder.config.hidden_size + self.image_model.encoder.num_features

        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.factor_head = nn.Linear(256, num_factors)

    def forward(self, input_ids, attention_mask, pixel_values, num_images=None):
        # Không dùng torch.no_grad() để Autograd có thể truyền ngược qua các layer đã được "tan băng"
        _, text_features = self.text_model(input_ids, attention_mask)
        # ImageModel tự xử lý 5D tensor và tính Average Pooling
        _, image_features = self.image_model(pixel_values, num_images=num_images)

        fused_features = torch.cat((text_features.to(torch.float32), image_features.to(torch.float32)), dim=1)
        return self.factor_head(self.fusion_fc(fused_features))
