import torch
import torch.nn as nn
from transformers import AutoModel


class TextModel(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_factors=5):
        super(TextModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name).float()

        self.fc = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.factor_head = nn.Linear(256, num_factors)

    def forward(self, input_ids, attention_mask, return_tokens=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Rút trích feature an toàn không cần if-else rườm rà
        features = outputs.pooler_output if getattr(outputs, 'pooler_output', None) is not None else outputs.last_hidden_state[:, 0, :]
        features = features.to(torch.float32)

        if return_tokens:
            # Token-level path used by CrossAttentionFusion.
            # tokens: (B, T, hidden), pad_mask: (B, T) bool with True == PAD
            tokens = outputs.last_hidden_state.to(torch.float32)
            pad_mask = (attention_mask == 0)
            return self.factor_head(self.fc(features)), features, tokens, pad_mask

        return self.factor_head(self.fc(features)), features
