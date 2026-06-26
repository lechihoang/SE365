"""
Grad-CAM explainer for the image branch of the multimodal model.

Provides manual Grad-CAM computation (no pytorch-grad-cam dependency) with
full control over multi-image hook isolation. Generates per-target heatmaps,
overlays, and 5-target comparison figures.

Usage:
    from xai.utils import load_model, load_single_sample
    from xai.gradcam_explainer import GradCAMExplainer

    model, config = load_model(exp_dir)
    sample = load_single_sample(csv_path, idx=0, tokenizer=tok, image_processor=proc)

    explainer = GradCAMExplainer(model, device, output_dir='./xai_output/gradcam')
    results = explainer.explain_sample(sample, sample_id='sample_000')
"""

import os
import math
import json
import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from xai.config import (
    TARGET_NAMES,
    FACTOR_NAMES,
    DISPLAY_NAMES,
    INDEX_TO_FACTOR,
    IMAGE_FEATURE_DIM,
    NUM_TARGETS,
    DEFAULT_DPI,
    THESIS_DPI,
    COLOR_SCHEMES,
)

# ══════════════════════════════════════════════════════════════════════════════
# Feature Map Normalization — try import from xai.utils, define locally as fallback
# ══════════════════════════════════════════════════════════════════════════════

try:
    from xai.utils import normalize_feature_map_to_bchw
except ImportError:
    def normalize_feature_map_to_bchw(
        tensor: torch.Tensor,
        expected_channels: int = IMAGE_FEATURE_DIM,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convert a spatial feature map from any format to [B, C, H, W].

        Handles all common formats produced by timm vision transformers:
          - [B, C, H, W]  -- already correct (ConvNeXt, some configs)
          - [B, H, W, C]  -- channel-last (Swin-B default in timm)
          - [B, N, C]     -- sequence format where N = H*W
          - [B, C, N]     -- transposed sequence format

        Args:
            tensor: Feature map tensor in any of the above formats.
            expected_channels: Number of channels to detect (default: 1024 for Swin-B).

        Returns:
            Tuple of (normalized_tensor [B, C, H, W], metadata dict).

        Raises:
            ValueError: If the tensor format cannot be determined.
        """
        shape = tensor.shape
        ndim = len(shape)
        meta = {
            'raw_shape': list(shape),
            'detected_format': None,
            'reshape_required': False,
            'reshape_rule': None,
            'target_shape': None,
        }

        if ndim == 4:
            B, d1, d2, d3 = shape

            if d1 == expected_channels and d2 > 1 and d3 > 1:
                meta['detected_format'] = 'BCHW'
                meta['target_shape'] = list(shape)
                return tensor, meta

            if d3 == expected_channels and d1 > 1 and d2 > 1:
                out = tensor.permute(0, 3, 1, 2).contiguous()
                meta['detected_format'] = 'BHWC'
                meta['reshape_required'] = True
                meta['reshape_rule'] = 'permute(0, 3, 1, 2)'
                meta['target_shape'] = list(out.shape)
                return out, meta

        elif ndim == 3:
            B, d1, d2 = shape

            if d2 == expected_channels:
                N = d1
                H = W = int(math.sqrt(N))
                if H * W == N:
                    out = tensor.permute(0, 2, 1).reshape(B, expected_channels, H, W).contiguous()
                    meta['detected_format'] = 'BNC'
                    meta['reshape_required'] = True
                    meta['reshape_rule'] = f'permute(0,2,1).reshape(B,{expected_channels},{H},{W})'
                    meta['target_shape'] = list(out.shape)
                    return out, meta

            if d1 == expected_channels:
                N = d2
                H = W = int(math.sqrt(N))
                if H * W == N:
                    out = tensor.reshape(B, expected_channels, H, W).contiguous()
                    meta['detected_format'] = 'BCN'
                    meta['reshape_required'] = True
                    meta['reshape_rule'] = f'reshape(B,{expected_channels},{H},{W})'
                    meta['target_shape'] = list(out.shape)
                    return out, meta

        raise ValueError(
            f'Cannot normalize feature map with shape {list(shape)} and '
            f'expected_channels={expected_channels}. '
            f'Supported formats: [B,C,H,W], [B,H,W,C], [B,N,C], [B,C,N].'
        )


# ══════════════════════════════════════════════════════════════════════════════
# MultiTargetScoreWrapper — wraps multimodal model to expose a single score
# ══════════════════════════════════════════════════════════════════════════════

class MultiTargetScoreWrapper(nn.Module):
    """Wraps a multimodal model so that forward(pixel_values) returns a single
    target score [B, 1], with text inputs fixed as buffers.

    This is useful when an external library (e.g., pytorch-grad-cam) expects
    a model with a single-tensor input and output.  For the manual Grad-CAM
    in this module it is not strictly needed, but is provided for
    compatibility and potential future use.

    Args:
        multimodal_model: The full CrossAttentionFusion (or similar) model.
        fixed_input_ids: Text input_ids [1, seq_len] to hold constant.
        fixed_attention_mask: Text attention_mask [1, seq_len].
        score_index: Which of the 5 target scores to extract (0-4).
        fixed_num_images: Optional num_images tensor [1].
    """

    def __init__(
        self,
        multimodal_model: nn.Module,
        fixed_input_ids: torch.Tensor,
        fixed_attention_mask: torch.Tensor,
        score_index: int,
        fixed_num_images: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = multimodal_model
        self.score_index = score_index

        # Store text inputs as buffers (not parameters) so they are not updated
        self.register_buffer('fixed_input_ids', fixed_input_ids.detach())
        self.register_buffer('fixed_attention_mask', fixed_attention_mask.detach())
        if fixed_num_images is not None:
            self.register_buffer('fixed_num_images', fixed_num_images.detach())
        else:
            self.fixed_num_images = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass with only pixel_values as variable input.

        Args:
            pixel_values: Image tensor [B, N, C, H, W] or [B, C, H, W].

        Returns:
            Selected score tensor [B, 1].
        """
        output = self.model(
            input_ids=self.fixed_input_ids,
            attention_mask=self.fixed_attention_mask,
            pixel_values=pixel_values,
            num_images=self.fixed_num_images,
        )
        # Fusion models return tensor, unimodal return tuple
        preds = output[0] if isinstance(output, tuple) else output
        return preds[:, self.score_index : self.score_index + 1]


# ══════════════════════════════════════════════════════════════════════════════
# SwinReshapeTransform — reshape_transform for pytorch-grad-cam compatibility
# ══════════════════════════════════════════════════════════════════════════════

class SwinReshapeTransform:
    """Reshape transform for Swin Transformer feature maps.

    Converts feature maps from various timm formats to [B, C, H, W],
    suitable for use as the ``reshape_transform`` parameter in
    pytorch-grad-cam (if one chooses to use that library instead of
    the manual implementation in this module).

    Handles:
      - [B, H, W, C] -> permute to [B, C, H, W]
      - [B, N, C]    -> reshape to [B, C, H, W] (assumes N = H*W, square)
      - [B, C, H, W] -> passthrough
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor to [B, C, H, W].

        Args:
            tensor: Feature map in [B, H, W, C], [B, N, C], or [B, C, H, W].

        Returns:
            Tensor in [B, C, H, W] format.
        """
        if tensor.ndim == 4:
            B, d1, d2, d3 = tensor.shape
            # [B, H, W, C] — channel-last
            if d3 > d1 and d3 > d2:
                return tensor.permute(0, 3, 1, 2).contiguous()
            # [B, C, H, W] — already correct
            return tensor

        if tensor.ndim == 3:
            B, N, C = tensor.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                raise ValueError(
                    f'Cannot reshape [B={B}, N={N}, C={C}] to spatial: '
                    f'N={N} is not a perfect square.'
                )
            return tensor.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        raise ValueError(f'Unexpected tensor ndim={tensor.ndim}, shape={tensor.shape}')


# ══════════════════════════════════════════════════════════════════════════════
# Manual Grad-CAM Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_gradcam_for_image(
    model: nn.Module,
    sample: Dict[str, Any],
    target_idx: int,
    image_idx: int,
    target_layer: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a specific image and target score.

    Uses manual hook-based Grad-CAM to handle multi-image input properly.
    The full model forward pass processes ALL images (preserving multi-image
    context), but hooks capture activations/gradients only for the specific
    image_idx.

    Algorithm:
      1. Register forward hook on target_layer to capture activations.
      2. Register backward hook on target_layer to capture gradients.
      3. Forward pass through the full model.
      4. Backward pass on the selected target score.
      5. Slice out activations and gradients for image_idx.
      6. Compute alpha_k = global_avg_pool(gradients).
      7. Compute cam = ReLU(sum(alpha_k * activations)).
      8. Normalize to [0, 1].

    Args:
        model: The full multimodal model (e.g., CrossAttentionFusion).
        sample: Sample dict from load_single_sample().
        target_idx: Index of the target score (0-4).
        image_idx: Index of the image within the multi-image batch (0-based).
        target_layer: The layer to hook for Grad-CAM (e.g., encoder.norm).
        device: Torch device.

    Returns:
        Grad-CAM heatmap as numpy array [H, W] normalized to [0, 1].
    """
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Enable gradients for this pass (model is in eval mode)
        model.zero_grad()

        with torch.enable_grad():
            # Ensure pixel_values requires grad for backward to flow
            pixel_values = sample['pixel_values'].clone().detach().requires_grad_(True)

            output = model(
                input_ids=sample['input_ids'],
                attention_mask=sample['attention_mask'],
                pixel_values=pixel_values,
                num_images=sample.get('num_images'),
            )
            # Fusion models return tensor, unimodal return tuple
            preds = output[0] if isinstance(output, tuple) else output

            # Select the target score and backward
            target_score = preds[0, target_idx]
            target_score.backward(retain_graph=False)

        # ── Extract activations and gradients for the specific image ──
        if 'value' not in activations or 'value' not in gradients:
            raise RuntimeError(
                f'Hooks did not capture activations/gradients. '
                f'Check that target_layer is in the forward path.'
            )

        act = activations['value']   # [B*N, H, W, C] or [B*N, C, H, W] etc.
        grad = gradients['value']    # same shape

        # Normalize to BCHW using the utility
        act_bchw, act_meta = normalize_feature_map_to_bchw(act, expected_channels=IMAGE_FEATURE_DIM)
        grad_bchw, _ = normalize_feature_map_to_bchw(grad, expected_channels=IMAGE_FEATURE_DIM)

        # The encoder processes B*N images. Slice out the target image.
        # pixel_values is [1, N, C, H, W], so B=1, and images are indexed 0..N-1.
        cam_act = act_bchw[image_idx]    # [C, H, W]
        cam_grad = grad_bchw[image_idx]  # [C, H, W]

        # alpha_k = global average pool of gradients per channel
        alpha = cam_grad.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination + ReLU
        cam = torch.relu((alpha * cam_act).sum(dim=0))  # [H, W]

        # Normalize to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    finally:
        # Always clean up hooks and gradients
        handle_fwd.remove()
        handle_bwd.remove()
        model.zero_grad()
        activations.clear()
        gradients.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Gradient Diagnostics — diagnose target similarity
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_target_gradients(
    model: nn.Module,
    sample: Dict[str, Any],
    target_layer: nn.Module,
    device: torch.device,
    image_idx: int = 0,
) -> Dict[str, Any]:
    """Diagnose why Grad-CAM heatmaps may look similar across targets.

    Runs backward for each of the 5 targets and compares:
    - gradient statistics (mean, std, abs_max)
    - pairwise cosine similarity between target gradients
    - raw CAM similarity before normalization

    Args:
        model: The full multimodal model.
        sample: Sample dict from load_single_sample().
        target_layer: The layer hooked for Grad-CAM.
        device: Torch device.
        image_idx: Which image to diagnose.

    Returns:
        Dictionary with diagnostic results.
    """
    import torch.nn.functional as F

    grad_vectors = []
    raw_cams = []
    grad_stats = []

    for t_idx in range(NUM_TARGETS):
        gradients = {}

        def bwd_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0].detach()

        handle = target_layer.register_full_backward_hook(bwd_hook)
        model.zero_grad()

        try:
            with torch.enable_grad():
                pv = sample['pixel_values'].clone().detach().requires_grad_(True)
                output = model(
                    input_ids=sample['input_ids'],
                    attention_mask=sample['attention_mask'],
                    pixel_values=pv,
                    num_images=sample.get('num_images'),
                )
                preds = output[0] if isinstance(output, tuple) else output
                preds[0, t_idx].backward(retain_graph=False)

            if 'value' in gradients:
                g = gradients['value']
                # normalize to BCHW
                g_bchw, _ = normalize_feature_map_to_bchw(g, IMAGE_FEATURE_DIM)
                g_img = g_bchw[image_idx]  # [C, H, W]

                flat = g_img.flatten().cpu()
                grad_vectors.append(flat)

                grad_stats.append({
                    'target': FACTOR_NAMES[t_idx],
                    'grad_mean': float(flat.mean()),
                    'grad_std': float(flat.std()),
                    'grad_abs_mean': float(flat.abs().mean()),
                    'grad_abs_max': float(flat.abs().max()),
                    'nonzero_ratio': float((flat.abs() > 1e-10).float().mean()),
                    'predicted_score': float(preds[0, t_idx].detach().cpu()),
                })

                # also compute raw CAM (before normalization)
                # reuse activations from a quick forward
                cam = compute_gradcam_for_image(
                    model, sample, t_idx, image_idx, target_layer, device
                )
                raw_cams.append(cam)
        finally:
            handle.remove()
            model.zero_grad()
            gradients.clear()

    # Compute pairwise cosine similarity of gradients
    n = len(grad_vectors)
    grad_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos = F.cosine_similarity(
                grad_vectors[i].unsqueeze(0),
                grad_vectors[j].unsqueeze(0),
            ).item()
            grad_sim[i, j] = cos

    # Compute pairwise correlation of raw CAMs
    cam_corr = np.zeros((n, n))
    flat_cams = [c.flatten() for c in raw_cams]
    for i in range(n):
        for j in range(n):
            if np.std(flat_cams[i]) > 1e-10 and np.std(flat_cams[j]) > 1e-10:
                cam_corr[i, j] = np.corrcoef(flat_cams[i], flat_cams[j])[0, 1]
            else:
                cam_corr[i, j] = 1.0 if i == j else 0.0

    # Raw CAM statistics
    cam_stats = []
    for t_idx, cam in enumerate(raw_cams):
        cam_stats.append({
            'target': FACTOR_NAMES[t_idx],
            'cam_min': float(cam.min()),
            'cam_max': float(cam.max()),
            'cam_mean': float(cam.mean()),
            'cam_std': float(cam.std()),
        })

    return {
        'grad_stats': grad_stats,
        'grad_similarity_matrix': grad_sim,
        'cam_correlation_matrix': cam_corr,
        'cam_stats': cam_stats,
        'raw_cams': raw_cams,
        'factor_names': FACTOR_NAMES,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Overlay CAM on Image
# ══════════════════════════════════════════════════════════════════════════════

def overlay_cam_on_image(
    cam: np.ndarray,
    original_image: Image.Image,
    image_size: int = 224,
    colormap_name: str = 'jet',
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an original image.

    Does NOT depend on pytorch-grad-cam's show_cam_on_image to avoid
    a mandatory dependency.

    Args:
        cam: Raw Grad-CAM heatmap [H, W] normalized to [0, 1].
        original_image: Original PIL Image.
        image_size: Target size for the overlay (square).
        colormap_name: Matplotlib colormap name (default: 'jet').
        alpha: Blending weight for the heatmap (0 = image only, 1 = heatmap only).

    Returns:
        Overlay as numpy array [image_size, image_size, 3] uint8.
    """
    import matplotlib.cm as cm

    # Resize original image
    img_resized = original_image.convert('RGB').resize(
        (image_size, image_size), Image.BILINEAR
    )
    img_np = np.array(img_resized, dtype=np.float32) / 255.0  # [H, W, 3]

    # Resize CAM to match image
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8), mode='L')
    cam_resized = np.array(
        cam_pil.resize((image_size, image_size), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0

    # Apply colormap
    colormap = cm.get_cmap(colormap_name)
    heatmap = colormap(cam_resized)[:, :, :3]  # [H, W, 3], drop alpha channel

    # Blend
    overlay = (1 - alpha) * img_np + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# 5-Target Comparison Figure
# ══════════════════════════════════════════════════════════════════════════════

def create_5target_comparison(
    original_images: List[Image.Image],
    all_cams: Dict[int, List[np.ndarray]],
    target_names: List[str],
    sample_id: str,
    save_path: str,
    dpi: int = DEFAULT_DPI,
) -> str:
    """Create a comparison figure with Grad-CAM heatmaps for all 5 targets.

    Layout:
      - Rows: one per image (handles single-image and multi-image reviews)
      - Columns: [Original, target0_overlay, target1_overlay, ..., target4_overlay]

    Args:
        original_images: List of PIL Images (one per real image in the review).
        all_cams: Dict mapping image_idx -> list of 5 cam arrays [H, W].
                  all_cams[img_idx][target_idx] = cam numpy array.
        target_names: List of 5 display names for the targets.
        sample_id: Identifier string for the sample (used in title).
        save_path: File path to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        The save_path for confirmation.
    """
    import matplotlib.pyplot as plt

    n_images = len(original_images)
    n_cols = 1 + NUM_TARGETS  # Original + 5 targets

    fig, axes = plt.subplots(
        n_images, n_cols,
        figsize=(3 * n_cols, 3 * n_images),
        squeeze=False,
    )

    for img_idx in range(n_images):
        pil_img = original_images[img_idx]

        # Original image column
        img_resized = pil_img.convert('RGB').resize((224, 224), Image.BILINEAR)
        axes[img_idx, 0].imshow(np.array(img_resized))
        if img_idx == 0:
            axes[img_idx, 0].set_title('Original', fontsize=10, fontweight='bold')
        if n_images > 1:
            axes[img_idx, 0].set_ylabel(f'Image {img_idx}', fontsize=9)
        axes[img_idx, 0].axis('off')

        # Grad-CAM overlay columns
        cams_for_image = all_cams.get(img_idx, [None] * NUM_TARGETS)
        for t_idx in range(NUM_TARGETS):
            ax = axes[img_idx, t_idx + 1]
            cam = cams_for_image[t_idx] if t_idx < len(cams_for_image) else None

            if cam is not None:
                overlay = overlay_cam_on_image(cam, pil_img)
                ax.imshow(overlay)
            else:
                ax.imshow(np.array(img_resized))
                ax.text(
                    0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='red',
                )

            if img_idx == 0:
                display_name = target_names[t_idx] if t_idx < len(target_names) else f'Target {t_idx}'
                ax.set_title(display_name, fontsize=9, fontweight='bold')
            ax.axis('off')

    fig.suptitle(
        f'Grad-CAM: {sample_id}',
        fontsize=13, fontweight='bold', y=1.02,
    )
    fig.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'[XAI] Saved 5-target comparison: {save_path}')
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# Find Target Layer — auto-detect the best layer for Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════

def find_target_layer(model: nn.Module) -> nn.Module:
    """Find the best target layer in the image encoder for Grad-CAM.

    Tries layers in priority order:
      1. model.image_model.encoder.norm  (Swin-B: produces [B, 7, 7, 1024])
      2. model.image_model.encoder.layers[-1]  (some ViT variants)
      3. model.image_model.encoder.stages[-1]  (ConvNeXt / older timm)

    Verifies the selected layer produces spatial features via a dummy forward pass.

    Args:
        model: The full multimodal model with an ``image_model.encoder`` attribute.

    Returns:
        The target layer module.

    Raises:
        RuntimeError: If no suitable target layer is found.
    """
    encoder = model.image_model.encoder

    # Priority 1: encoder.norm (Swin-B default)
    if hasattr(encoder, 'norm') and isinstance(encoder.norm, nn.Module):
        layer = encoder.norm
        print(f'[XAI] Target layer: image_model.encoder.norm ({type(layer).__name__})')

        # Verify it produces spatial features
        if _verify_target_layer(model, layer):
            return layer
        print('[XAI] WARNING: encoder.norm did not produce spatial features, trying fallbacks.')

    # Priority 2: encoder.layers[-1]
    if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
        layer = encoder.layers[-1]
        print(f'[XAI] Target layer fallback: image_model.encoder.layers[-1] ({type(layer).__name__})')
        if _verify_target_layer(model, layer):
            return layer

    # Priority 3: encoder.stages[-1]
    if hasattr(encoder, 'stages') and len(encoder.stages) > 0:
        layer = encoder.stages[-1]
        print(f'[XAI] Target layer fallback: image_model.encoder.stages[-1] ({type(layer).__name__})')
        if _verify_target_layer(model, layer):
            return layer

    raise RuntimeError(
        'Could not find a suitable target layer for Grad-CAM. '
        'Tried: encoder.norm, encoder.layers[-1], encoder.stages[-1]. '
        f'Available attributes: {[n for n, _ in encoder.named_children()]}'
    )


def _verify_target_layer(model: nn.Module, layer: nn.Module) -> bool:
    """Verify that a target layer produces spatial features via a dummy forward pass.

    Args:
        model: The full multimodal model.
        layer: The candidate target layer.

    Returns:
        True if the layer produces a feature map that can be normalized to BCHW.
    """
    captured = {}

    def hook_fn(module, input, output):
        captured['output'] = output

    handle = layer.register_forward_hook(hook_fn)

    try:
        device = next(model.parameters()).device
        dummy_pixel = torch.randn(1, 1, 3, 224, 224, device=device)

        with torch.no_grad():
            # Run just the image encoder to check the layer output
            model.image_model(dummy_pixel)

        if 'output' not in captured:
            return False

        output = captured['output']
        if isinstance(output, tuple):
            output = output[0]

        # Check if it can be normalized to BCHW
        try:
            normalized, meta = normalize_feature_map_to_bchw(output, IMAGE_FEATURE_DIM)
            H, W = normalized.shape[2], normalized.shape[3]
            print(f'[XAI]   Verified: {meta["detected_format"]} -> [B, {IMAGE_FEATURE_DIM}, {H}, {W}]')
            return H > 1 and W > 1
        except ValueError:
            return False

    except Exception as e:
        print(f'[XAI] WARNING: Target layer verification failed: {e}')
        return False

    finally:
        handle.remove()
        captured.clear()


# ══════════════════════════════════════════════════════════════════════════════
# GradCAMExplainer — main entry point
# ══════════════════════════════════════════════════════════════════════════════

class GradCAMExplainer:
    """High-level Grad-CAM explainer for the image branch.

    Generates heatmaps, overlays, comparison figures, and metadata for
    individual samples. Handles single-image and multi-image reviews.

    Args:
        model: The full multimodal model (e.g., CrossAttentionFusion), in eval mode.
        device: Torch device.
        output_dir: Base directory for saving artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = './xai_output/gradcam',
    ):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.target_layer = find_target_layer(model)

        os.makedirs(output_dir, exist_ok=True)
        print(f'[XAI] GradCAMExplainer initialized. Output dir: {output_dir}')

    def explain_sample(
        self,
        sample: Dict[str, Any],
        sample_id: str,
    ) -> Dict[str, Any]:
        """Generate Grad-CAM explanations for all targets and all images in a sample.

        Args:
            sample: Sample dict from load_single_sample().
            sample_id: Unique identifier for the sample (e.g., 'sample_042').

        Returns:
            Dictionary with:
              - 'sample_id': str
              - 'num_images': int
              - 'target_names': list of str
              - 'cams': dict mapping (image_idx, target_idx) -> cam [H, W] numpy
              - 'overlays': dict mapping (image_idx, target_idx) -> overlay [H, W, 3] uint8
              - 'comparison_path': str (path to saved comparison figure)
              - 'individual_paths': dict mapping (image_idx, target_idx) -> saved overlay path
              - 'metadata': dict with generation info
        """
        sample_dir = os.path.join(self.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        num_images = sample.get('num_real_images', 1)
        original_images = sample.get('loaded_images', [])

        if not original_images:
            print(f'[XAI] WARNING: No loaded_images in sample. Using placeholder.')
            original_images = [Image.new('RGB', (224, 224), color='gray')]
            num_images = 1

        print(f'[XAI] Explaining {sample_id}: {num_images} image(s), {NUM_TARGETS} targets')

        all_cams: Dict[int, List[np.ndarray]] = {}
        all_overlays: Dict[Tuple[int, int], np.ndarray] = {}
        individual_paths: Dict[Tuple[int, int], str] = {}
        cam_store: Dict[Tuple[int, int], np.ndarray] = {}

        for img_idx in range(num_images):
            cams_for_image = []
            for t_idx in range(NUM_TARGETS):
                print(
                    f'[XAI]   Computing Grad-CAM: image {img_idx}, '
                    f'target {t_idx} ({FACTOR_NAMES[t_idx]})'
                )

                cam = compute_gradcam_for_image(
                    model=self.model,
                    sample=sample,
                    target_idx=t_idx,
                    image_idx=img_idx,
                    target_layer=self.target_layer,
                    device=self.device,
                )
                cams_for_image.append(cam)
                cam_store[(img_idx, t_idx)] = cam

                # Generate overlay
                overlay = overlay_cam_on_image(cam, original_images[img_idx])
                all_overlays[(img_idx, t_idx)] = overlay

                # Save individual overlay
                overlay_filename = f'gradcam_img{img_idx}_{FACTOR_NAMES[t_idx]}.png'
                overlay_path = os.path.join(sample_dir, overlay_filename)
                Image.fromarray(overlay).save(overlay_path)
                individual_paths[(img_idx, t_idx)] = overlay_path

            all_cams[img_idx] = cams_for_image

        # Save raw CAM values as .npz
        cam_arrays = {
            f'cam_img{img_idx}_{FACTOR_NAMES[t_idx]}': cam_store[(img_idx, t_idx)]
            for img_idx, t_idx in cam_store
        }
        raw_path = os.path.join(sample_dir, 'raw_cams.npz')
        np.savez(raw_path, **cam_arrays)
        print(f'[XAI] Saved raw CAMs: {raw_path}')

        # Create 5-target comparison figure
        comparison_path = os.path.join(sample_dir, 'gradcam_5target_comparison.png')
        create_5target_comparison(
            original_images=original_images[:num_images],
            all_cams=all_cams,
            target_names=DISPLAY_NAMES,
            sample_id=sample_id,
            save_path=comparison_path,
            dpi=DEFAULT_DPI,
        )

        # Generate metadata
        metadata = {
            'sample_id': sample_id,
            'sample_idx': sample.get('sample_idx', None),
            'num_images': num_images,
            'num_targets': NUM_TARGETS,
            'target_names': TARGET_NAMES,
            'display_names': DISPLAY_NAMES,
            'target_layer': str(type(self.target_layer).__name__),
            'device': str(self.device),
            'timestamp': datetime.datetime.now().isoformat(),
            'cam_shape': list(cam_store.get((0, 0), np.array([])).shape),
            'artifacts': {
                'comparison': comparison_path,
                'raw_cams': raw_path,
                'individual_overlays': {
                    f'img{k[0]}_{FACTOR_NAMES[k[1]]}': v
                    for k, v in individual_paths.items()
                },
            },
        }

        # Save metadata
        meta_path = os.path.join(sample_dir, 'metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved metadata: {meta_path}')

        return {
            'sample_id': sample_id,
            'num_images': num_images,
            'target_names': TARGET_NAMES,
            'cams': cam_store,
            'overlays': all_overlays,
            'comparison_path': comparison_path,
            'individual_paths': individual_paths,
            'metadata': metadata,
        }

    def explain_single_target(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        target_idx: int,
        image_idx: int = 0,
    ) -> Dict[str, Any]:
        """Generate Grad-CAM explanation for a single target and image.

        Lighter-weight alternative to explain_sample() when only one
        target/image combination is needed.

        Args:
            sample: Sample dict from load_single_sample().
            sample_id: Unique identifier for the sample.
            target_idx: Index of the target score (0-4).
            image_idx: Index of the image (0-based, default 0).

        Returns:
            Dictionary with:
              - 'cam': numpy array [H, W]
              - 'overlay': numpy array [H, W, 3] uint8
              - 'overlay_path': str (path to saved overlay)
              - 'target_name': str
              - 'factor_name': str
        """
        sample_dir = os.path.join(self.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        original_images = sample.get('loaded_images', [])
        if not original_images or image_idx >= len(original_images):
            print(f'[XAI] WARNING: image_idx={image_idx} out of range. Using placeholder.')
            pil_img = Image.new('RGB', (224, 224), color='gray')
        else:
            pil_img = original_images[image_idx]

        factor_name = FACTOR_NAMES[target_idx]
        print(
            f'[XAI] Computing Grad-CAM: {sample_id}, '
            f'image {image_idx}, target {target_idx} ({factor_name})'
        )

        cam = compute_gradcam_for_image(
            model=self.model,
            sample=sample,
            target_idx=target_idx,
            image_idx=image_idx,
            target_layer=self.target_layer,
            device=self.device,
        )

        overlay = overlay_cam_on_image(cam, pil_img)

        overlay_filename = f'gradcam_img{image_idx}_{factor_name}.png'
        overlay_path = os.path.join(sample_dir, overlay_filename)
        Image.fromarray(overlay).save(overlay_path)
        print(f'[XAI] Saved overlay: {overlay_path}')

        return {
            'cam': cam,
            'overlay': overlay,
            'overlay_path': overlay_path,
            'target_name': TARGET_NAMES[target_idx],
            'factor_name': factor_name,
            'display_name': DISPLAY_NAMES[target_idx],
            'image_idx': image_idx,
            'target_idx': target_idx,
        }
