"""Shared helpers for the fusion modules.

The five fusion architectures (Concat, GMU, Gated Cross-Modal, FiLM,
Cross-Attention) all freeze the two backbones and then *selectively
unfreeze* the last N text/image layers. The exact attribute path differs
by family:

- Text (HuggingFace Transformers):
    <BaseModel>.encoder.encoder.layer   # RoBERTa / BERT / XLM-R / PhoBERT / ViSoBERT / DeBERTa
    <BaseModel>.encoder.layer           # some models expose the layer list directly

- Image (timm):
    encoder.stages[-1].blocks           # ConvNeXt
    encoder.layers[-1].blocks           # Swin
    encoder.blocks[-1]                  # EfficientNet (a Sequential of sub-blocks)

Previously each fusion file hard-coded only the first path of each side,
which silently skipped the unfreeze step for Swin and EfficientNet and for
any non-canonical text encoder. Centralising the logic here removes the
duplication and guarantees a ``UserWarning`` is raised whenever a backbone
cannot be matched, instead of failing quietly.

Note: the helpers are intentionally named ``unfreeze_text_backbone`` /
``unfreeze_image_backbone`` (not ``unfreeze_text_layers``) so they do not
shadow the same-named ``__init__`` parameters of the fusion classes.
"""

import warnings


def freeze_all(*models):
    """Set ``requires_grad = False`` on every parameter of each model."""
    for model in models:
        for param in model.parameters():
            param.requires_grad = False


def unfreeze_text_backbone(text_model, n):
    """Unfreeze the last ``n`` transformer layers of a HuggingFace text model.

    The text model is expected to expose its backbone as ``text_model.encoder``
    (this is how ``TextModel`` wraps ``AutoModel``).
    """
    if n <= 0:
        return
    enc = text_model.encoder
    layers = None
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
        layers = enc.encoder.layer
    elif hasattr(enc, "layer"):
        layers = enc.layer
    else:
        warnings.warn(
            f"[unfreeze] Cannot unfreeze text layers for {type(enc).__name__} "
            f"— no known layer container."
        )
        return
    for layer in layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True


def unfreeze_image_backbone(image_model, n):
    """Unfreeze the last ``n`` blocks of a timm image backbone.

    The image model is expected to expose its backbone as
    ``image_model.encoder`` (this is how ``ImageModel`` wraps a ``timm``
    model). ConvNeXt, Swin and EfficientNet use different attribute paths,
    all handled here.
    """
    if n <= 0:
        return
    enc = image_model.encoder
    blocks = []
    if hasattr(enc, "stages"):                                    # ConvNeXt
        blocks = enc.stages[-1].blocks
    elif hasattr(enc, "layers") and hasattr(enc.layers[-1], "blocks"):  # Swin
        blocks = enc.layers[-1].blocks
    elif hasattr(enc, "blocks"):                                  # EfficientNet
        last = enc.blocks[-1]
        blocks = list(last) if hasattr(last, "__len__") else [last]
    else:
        warnings.warn(
            f"[unfreeze] Cannot unfreeze image layers for {type(enc).__name__} "
            f"— no known block container."
        )
        return
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True
