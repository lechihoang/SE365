"""
SHAP-based explainer for the fused embedding of the multimodal model.

Phase 4 of the XAI pipeline. Uses SHAP (SHapley Additive exPlanations) to
attribute each of the 1024 fused-vector dimensions to the 5 predicted quality
scores.  Because CrossAttentionFusion concatenates text-origin (dims 0:512)
and image-origin (dims 512:1024) features, SHAP values can be aggregated by
modality to answer "how much did text vs. image information contribute to
each score?"

NOTE on naming: We use "text-origin" and "image-origin" rather than plain
"text" / "image" because the fused features have already gone through
cross-attention.  t_out (text-origin) is the result of text queries attending
to image keys/values, so it contains information from *both* modalities.

Usage:
    from xai.utils import load_model
    from xai.shap_explainer import SHAPExplainer

    model, config = load_model(exp_dir)
    explainer = SHAPExplainer(model, device, output_dir='./xai_output/shap')
    embeddings = explainer.extract_embeddings(dataloader, max_samples=200)
    results = explainer.explain_sample(sample_fused, sample_id='sample_000',
                                        background=background)
"""

import os
import json
import datetime
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from xai.config import (
    TARGET_NAMES,
    FACTOR_NAMES,
    DISPLAY_NAMES,
    INDEX_TO_FACTOR,
    NUM_TARGETS,
    FUSED_DIM,
    CROSS_ATTN_HIDDEN_DIM,
    DEFAULT_DPI,
    COLOR_SCHEMES,
    DEFAULT_SEED,
)

# ── Graceful import of shap ─────────────────────────────────────────────────
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _check_shap() -> None:
    """Raise ImportError with install instructions if shap is missing."""
    if not _SHAP_AVAILABLE:
        raise ImportError(
            'The `shap` library is required for Phase 4 but is not installed.\n'
            'Install it with:  pip install shap\n'
            'For GPU support:  pip install shap --no-build-isolation'
        )


# ══════════════════════════════════════════════════════════════════════════════
# 1. FusionHeadWrapper — wraps model.head for a single target score
# ══════════════════════════════════════════════════════════════════════════════

class FusionHeadWrapper(nn.Module):
    """Wraps the fusion head (nn.Sequential) to output a single target score.

    Takes the 1024-dim fused embedding as input and returns the predicted
    value for one specific quality factor (e.g., food_score).

    The wrapper is always kept in eval mode so that Dropout layers inside
    the head are disabled, producing deterministic outputs for SHAP.

    Args:
        head_sequential: The ``model.head`` nn.Sequential module.
        score_index: Which of the 5 target scores to extract (0-4).
    """

    def __init__(self, head_sequential: nn.Sequential, score_index: int):
        super().__init__()
        if not 0 <= score_index < NUM_TARGETS:
            raise ValueError(
                f'score_index must be in [0, {NUM_TARGETS - 1}], got {score_index}'
            )
        self.head = head_sequential
        self.score_index = score_index
        # Ensure eval mode so Dropout is disabled
        self.eval()

    def forward(self, fused_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head, selecting one score column.

        Args:
            fused_embedding: Fused feature vector [B, 1024].

        Returns:
            Selected score tensor [B, 1].
        """
        all_scores = self.head(fused_embedding)  # [B, 5]
        return all_scores[:, self.score_index : self.score_index + 1]  # [B, 1]

    def train(self, mode: bool = True):
        """Override train() to always stay in eval mode."""
        return super().train(False)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Extract fused embeddings from dataloader
# ══════════════════════════════════════════════════════════════════════════════

def extract_fused_embeddings(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract fused embeddings by hooking model.head input during inference.

    Registers a forward_pre_hook on ``model.head`` to capture the 1024-dim
    fused vector that is the input to the classification head.  Runs the
    model in inference mode (no gradients) over the dataloader.

    Args:
        model: The full multimodal model (e.g., CrossAttentionFusion), in
               eval mode.
        dataloader: PyTorch DataLoader yielding batches with keys
                    ``input_ids``, ``attention_mask``, ``pixel_values``,
                    ``num_images``, and ``labels`` (or target tensor).
        device: Torch device.
        max_samples: If set, stop after collecting this many samples.

    Returns:
        Tuple of:
          - fused_tensor [N, 1024]: The captured fused embeddings.
          - labels [N, 5]: Ground-truth labels.
          - predictions [N, 5]: Model predictions.
    """
    fused_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    pred_list: List[torch.Tensor] = []
    collected = 0

    captured = {}

    def _hook_fn(module, args):
        """forward_pre_hook: capture the input to model.head."""
        # args is a tuple; the first element is the fused vector [B, 1024]
        if isinstance(args, tuple) and len(args) > 0:
            captured['fused'] = args[0].detach()
        elif isinstance(args, torch.Tensor):
            captured['fused'] = args.detach()

    hook_handle = model.head.register_forward_pre_hook(_hook_fn)

    try:
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                num_images = batch.get('num_images')
                if num_images is not None:
                    num_images = num_images.to(device)

                # Labels
                labels = batch.get('labels', batch.get('targets', None))
                if labels is None:
                    # Try to construct from individual score columns
                    labels = torch.stack([
                        batch[name] for name in TARGET_NAMES
                    ], dim=1) if TARGET_NAMES[0] in batch else None

                # Forward pass
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    num_images=num_images,
                )
                preds = output[0] if isinstance(output, tuple) else output

                # Collect captured fused embeddings
                if 'fused' in captured:
                    fused_list.append(captured['fused'].cpu())
                    captured.clear()

                pred_list.append(preds.detach().cpu())
                if labels is not None:
                    label_list.append(labels.cpu())

                collected += input_ids.size(0)
                if max_samples is not None and collected >= max_samples:
                    break

    finally:
        hook_handle.remove()
        captured.clear()

    # Concatenate
    fused_tensor = torch.cat(fused_list, dim=0)
    predictions = torch.cat(pred_list, dim=0)

    if label_list:
        labels_tensor = torch.cat(label_list, dim=0)
    else:
        labels_tensor = torch.zeros(fused_tensor.size(0), NUM_TARGETS)

    # Trim to max_samples
    if max_samples is not None and fused_tensor.size(0) > max_samples:
        fused_tensor = fused_tensor[:max_samples]
        labels_tensor = labels_tensor[:max_samples]
        predictions = predictions[:max_samples]

    print(f'[XAI] Extracted {fused_tensor.size(0)} fused embeddings, '
          f'shape: {list(fused_tensor.shape)}')

    return fused_tensor, labels_tensor, predictions


# ══════════════════════════════════════════════════════════════════════════════
# 3. Select background samples for SHAP
# ══════════════════════════════════════════════════════════════════════════════

def select_background(
    fused_tensor: torch.Tensor,
    n_background: int = 100,
    seed: int = DEFAULT_SEED,
) -> Tuple[torch.Tensor, List[int]]:
    """Randomly sample background embeddings for SHAP baseline.

    Args:
        fused_tensor: All fused embeddings [N, 1024].
        n_background: Number of background samples to select.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
          - background [n_bg, 1024]: Selected background embeddings.
          - indices: List of selected sample indices.
    """
    n_total = fused_tensor.size(0)
    n_bg = min(n_background, n_total)

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=n_bg, replace=False).tolist()

    background = fused_tensor[indices]

    print(f'[XAI] Selected {n_bg} background samples '
          f'(from {n_total} total, seed={seed})')

    return background, indices


# ══════════════════════════════════════════════════════════════════════════════
# 4. Compute SHAP values
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(
    wrapper: FusionHeadWrapper,
    background: torch.Tensor,
    samples: torch.Tensor,
    method: str = 'deep',
) -> Tuple[np.ndarray, float]:
    """Compute SHAP values for fused embedding dimensions.

    Uses DeepExplainer (gradient-based, fast) to explain how each of the
    1024 fused dimensions contributes to a single target score.

    IMPORTANT: DeepExplainer requires PyTorch tensors (not numpy arrays)
    for both background and samples.  Do NOT wrap the computation in
    ``torch.no_grad()`` — DeepExplainer needs gradients to flow.

    Args:
        wrapper: FusionHeadWrapper targeting one score index.
        background: Background embeddings [n_bg, 1024] as PyTorch tensor.
        samples: Sample embeddings to explain [N, 1024] as PyTorch tensor.
        method: SHAP method. Currently only 'deep' (DeepExplainer) is
                supported.

    Returns:
        Tuple of:
          - shap_values [N, 1024]: SHAP value for each fused dimension.
          - base_value: Expected model output over the background set.

    Raises:
        ImportError: If shap is not installed.
        ValueError: If method is not supported.
    """
    _check_shap()

    if method != 'deep':
        raise ValueError(
            f'Unsupported SHAP method: {method!r}. '
            f'Currently only "deep" (DeepExplainer) is supported.'
        )

    # Ensure wrapper is in eval mode (dropout disabled)
    wrapper.eval()

    # DeepExplainer needs tensors on the same device as the model
    device = next(wrapper.parameters()).device
    bg = background.to(device)
    s = samples.to(device)

    # Create DeepExplainer — DO NOT use torch.no_grad() here
    explainer = shap.DeepExplainer(wrapper, bg)
    raw_shap = explainer.shap_values(s)

    # Handle return format: may be a list (one per output) or ndarray
    if isinstance(raw_shap, list):
        # DeepExplainer with single output [B, 1] may return a list of length 1
        shap_vals = raw_shap[0]
    else:
        shap_vals = raw_shap

    # Convert to numpy if tensor
    if isinstance(shap_vals, torch.Tensor):
        shap_vals = shap_vals.detach().cpu().numpy()
    elif not isinstance(shap_vals, np.ndarray):
        shap_vals = np.array(shap_vals)

    # Squeeze trailing dim if shape is [N, 1024, 1]
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals.squeeze(-1)

    # Base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[0])
    else:
        base_value = float(base_value)

    return shap_vals, base_value


# ══════════════════════════════════════════════════════════════════════════════
# 5. Modality contribution analysis
# ══════════════════════════════════════════════════════════════════════════════

def modality_contribution(
    shap_values: np.ndarray,
    text_dim: int = CROSS_ATTN_HIDDEN_DIM,
) -> Dict[str, Any]:
    """Compute text-origin vs image-origin contribution from SHAP values.

    Splits the 1024-dim SHAP values into:
      - text-origin (dims 0:512): cross-attended text features
      - image-origin (dims 512:1024): cross-attended image features

    Works for both single samples [1024] and batches [N, 1024].

    Args:
        shap_values: SHAP values, shape [1024] or [N, 1024].
        text_dim: Number of text-origin dimensions (default 512).

    Returns:
        Dictionary with:
          - text_abs: Sum of |SHAP| for text-origin dims.
          - image_abs: Sum of |SHAP| for image-origin dims.
          - text_pct: Text-origin percentage of total |SHAP|.
          - image_pct: Image-origin percentage of total |SHAP|.
          - text_signed: Sum of signed SHAP for text-origin dims.
          - image_signed: Sum of signed SHAP for image-origin dims.
    """
    is_single = shap_values.ndim == 1
    if is_single:
        sv = shap_values[np.newaxis, :]  # [1, 1024]
    else:
        sv = shap_values  # [N, 1024]

    text_sv = sv[:, :text_dim]       # [N, 512]
    image_sv = sv[:, text_dim:]      # [N, 512]

    text_abs = np.abs(text_sv).sum(axis=1)    # [N]
    image_abs = np.abs(image_sv).sum(axis=1)  # [N]
    total_abs = text_abs + image_abs

    # Avoid division by zero
    safe_total = np.where(total_abs > 1e-12, total_abs, 1.0)
    text_pct = (text_abs / safe_total) * 100.0
    image_pct = (image_abs / safe_total) * 100.0

    text_signed = text_sv.sum(axis=1)    # [N]
    image_signed = image_sv.sum(axis=1)  # [N]

    if is_single:
        return {
            'text_abs': float(text_abs[0]),
            'image_abs': float(image_abs[0]),
            'text_pct': float(text_pct[0]),
            'image_pct': float(image_pct[0]),
            'text_signed': float(text_signed[0]),
            'image_signed': float(image_signed[0]),
        }
    else:
        return {
            'text_abs': text_abs.tolist(),
            'image_abs': image_abs.tolist(),
            'text_pct': text_pct.tolist(),
            'image_pct': image_pct.tolist(),
            'text_signed': text_signed.tolist(),
            'image_signed': image_signed.tolist(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Additivity check
# ══════════════════════════════════════════════════════════════════════════════

def additivity_check(
    wrapper: FusionHeadWrapper,
    sample: torch.Tensor,
    shap_values: np.ndarray,
    base_value: float,
    tolerance: float = 0.05,
) -> Tuple[bool, float]:
    """Verify the SHAP additivity property: f(x) ≈ E[f(x)] + sum(SHAP).

    The fundamental SHAP property states that the model prediction for a
    sample should equal the base value (expected prediction over background)
    plus the sum of all SHAP values for that sample.

    Args:
        wrapper: FusionHeadWrapper targeting one score index.
        sample: Single fused embedding [1024] or [1, 1024].
        shap_values: SHAP values for this sample [1024] or [1, 1024].
        base_value: Expected model output (from DeepExplainer).
        tolerance: Maximum allowable absolute error.

    Returns:
        Tuple of:
          - passes: Whether the check passes (error < tolerance).
          - error: Absolute difference between prediction and SHAP sum.
    """
    wrapper.eval()
    device = next(wrapper.parameters()).device

    # Ensure correct shapes
    if sample.ndim == 1:
        sample = sample.unsqueeze(0)
    sample = sample.to(device)

    with torch.no_grad():
        prediction = wrapper(sample).item()

    sv = shap_values.flatten()
    shap_sum = base_value + float(sv.sum())

    error = abs(prediction - shap_sum)
    passes = error < tolerance

    if not passes:
        print(f'[XAI] WARNING: Additivity check failed. '
              f'prediction={prediction:.4f}, base+sum(shap)={shap_sum:.4f}, '
              f'error={error:.4f} > tolerance={tolerance}')
    else:
        print(f'[XAI] Additivity check passed. '
              f'error={error:.6f} < tolerance={tolerance}')

    return passes, error


# ══════════════════════════════════════════════════════════════════════════════
# 7. Modality contribution bar chart (all 5 targets)
# ══════════════════════════════════════════════════════════════════════════════

def plot_modality_contribution(
    contributions,
    target_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
):
    """Plot grouped bar chart of text-origin vs image-origin contribution.

    Creates a figure with 5 grouped bars (one per target), each showing
    the text-origin and image-origin percentage contributions determined
    by SHAP value magnitudes.

    Args:
        contributions: Either a list of 5 dicts (each with 'text_pct' and
                       'image_pct' keys), or a dict mapping factor_name ->
                       contribution dict.
        target_names: Display names for the targets. If None, uses
                      DISPLAY_NAMES from config.
        save_path: If provided, save figure to this path. If None, only
                   return the Figure without saving.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    # Accept both dict (factor_name -> contrib) and list formats
    if isinstance(contributions, dict):
        ordered_keys = list(contributions.keys())
        contrib_list = [contributions[k] for k in ordered_keys]
        if target_names is None:
            target_names = ordered_keys
    else:
        contrib_list = contributions

    if target_names is None:
        target_names = DISPLAY_NAMES

    n_targets = len(contrib_list)
    text_pcts = [c['text_pct'] for c in contrib_list]
    image_pcts = [c['image_pct'] for c in contrib_list]

    x = np.arange(n_targets)
    width = 0.35

    text_color = COLOR_SCHEMES['modality_colors']['text']
    image_color = COLOR_SCHEMES['modality_colors']['image']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_text = ax.bar(x - width / 2, text_pcts, width,
                       label='Text-origin', color=text_color, edgecolor='white')
    bars_image = ax.bar(x + width / 2, image_pcts, width,
                        label='Image-origin', color=image_color, edgecolor='white')

    for bar in bars_text:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    for bar in bars_image:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    ax.set_xlabel('Quality Factor', fontsize=12)
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_title('SHAP Modality Contribution: Text-Origin vs Image-Origin',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, fontsize=10)
    ax.set_ylim(0, max(max(text_pcts), max(image_pcts)) * 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'[XAI] Saved modality contribution chart: {save_path}')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. Single-target modality bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_modality_single_target(
    text_pct: float,
    image_pct: float,
    target_name: str,
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
):
    """Plot single bar chart showing text-origin vs image-origin contribution.

    Args:
        text_pct: Text-origin contribution percentage.
        image_pct: Image-origin contribution percentage.
        target_name: Display name for the target (e.g., 'Food Score').
        save_path: If provided, save figure. If None, only return Figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    text_color = COLOR_SCHEMES['modality_colors']['text']
    image_color = COLOR_SCHEMES['modality_colors']['image']

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(
        ['Text-origin', 'Image-origin'],
        [text_pct, image_pct],
        color=[text_color, image_color],
        edgecolor='white',
        width=0.5,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.set_ylabel('Contribution (%)', fontsize=11)
    ax.set_title(f'SHAP Modality Contribution: {target_name}',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(text_pct, image_pct) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'[XAI] Saved single-target chart: {save_path}')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9. SHAPExplainer — High-Level Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """High-level SHAP explainer for the fused embedding space.

    Orchestrates SHAP value computation, modality contribution analysis,
    visualization, and artifact saving.  Analogous to GradCAMExplainer
    (Phase 2) and AttentionExplainer (Phase 3).

    The explainer operates on the 1024-dim fused vector that feeds into
    ``model.head``.  It wraps the head for each of the 5 targets and uses
    DeepExplainer to compute per-dimension SHAP values.

    Args:
        model: The full multimodal model (e.g., CrossAttentionFusion),
               in eval mode.
        device: Torch device.
        output_dir: Base directory for saving artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = './xai_output/shap',
    ):
        _check_shap()

        self.model = model
        self.device = device
        self.output_dir = output_dir

        # Ensure model is in eval mode
        self.model.eval()

        os.makedirs(output_dir, exist_ok=True)
        print(f'[XAI] SHAPExplainer initialized. Output dir: {output_dir}')

    def extract_embeddings(
        self,
        dataloader,
        max_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract fused embeddings from the dataloader.

        Convenience wrapper around :func:`extract_fused_embeddings`.

        Args:
            dataloader: PyTorch DataLoader.
            max_samples: If set, stop after this many samples.

        Returns:
            Fused embedding tensor [N, 1024].
        """
        fused, labels, preds = extract_fused_embeddings(
            self.model, dataloader, self.device, max_samples=max_samples,
        )
        # Store labels and predictions for potential later use
        self._last_labels = labels
        self._last_predictions = preds
        return fused

    def explain_sample(
        self,
        sample_fused: torch.Tensor,
        sample_id: str,
        background: torch.Tensor,
    ) -> Dict[str, Any]:
        """Generate full SHAP explanation for a single sample.

        For each of the 5 target scores:
          1. Create FusionHeadWrapper for the target
          2. Compute SHAP values via DeepExplainer
          3. Compute modality contribution (text-origin vs image-origin)
          4. Run additivity check

        Then:
          5. Generate modality contribution bar chart (all 5 targets)
          6. Save raw SHAP values as .npz
          7. Save modality contribution as JSON
          8. Save metadata JSON

        Output folder structure::

            {output_dir}/{sample_id}/
              shap_modality_contribution.png
              shap_modality_contribution.json
              raw_shap_values.npz
              metadata.json

        Args:
            sample_fused: Fused embedding for the sample [1024] or [1, 1024].
            sample_id: Unique identifier (e.g., 'sample_042').
            background: Background embeddings [n_bg, 1024].

        Returns:
            Dictionary with:
              - sample_id: str
              - shap_values: dict mapping factor_name -> [1024] ndarray
              - base_values: dict mapping factor_name -> float
              - contributions: list of 5 modality contribution dicts
              - additivity: dict mapping factor_name -> (passes, error)
              - paths: dict with all artifact paths
        """
        sample_dir = os.path.join(self.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Ensure sample is [1, 1024]
        if sample_fused.ndim == 1:
            sample_fused = sample_fused.unsqueeze(0)

        print(f'[XAI] Explaining {sample_id} (SHAP analysis, {NUM_TARGETS} targets)')

        shap_values_dict: Dict[str, np.ndarray] = {}
        base_values_dict: Dict[str, float] = {}
        contributions_list: List[Dict[str, Any]] = []
        additivity_dict: Dict[str, Tuple[bool, float]] = {}

        for t_idx in range(NUM_TARGETS):
            factor = FACTOR_NAMES[t_idx]
            display = DISPLAY_NAMES[t_idx]
            print(f'[XAI]   Target {t_idx} ({factor}): computing SHAP values...')

            # Create wrapper for this target
            wrapper = FusionHeadWrapper(self.model.head, t_idx)
            wrapper.to(self.device)
            wrapper.eval()

            # Compute SHAP values
            sv, base_val = compute_shap_values(
                wrapper=wrapper,
                background=background,
                samples=sample_fused,
                method='deep',
            )

            # sv is [1, 1024], squeeze to [1024]
            sv_single = sv[0] if sv.ndim > 1 else sv

            shap_values_dict[factor] = sv_single
            base_values_dict[factor] = base_val

            # Modality contribution
            contrib = modality_contribution(sv_single)
            contributions_list.append(contrib)
            print(f'[XAI]     text-origin: {contrib["text_pct"]:.1f}%, '
                  f'image-origin: {contrib["image_pct"]:.1f}%')

            # Additivity check
            passes, error = additivity_check(
                wrapper=wrapper,
                sample=sample_fused,
                shap_values=sv_single,
                base_value=base_val,
            )
            additivity_dict[factor] = (passes, error)

        # ── Generate modality contribution chart (all 5 targets) ─────────
        chart_path = os.path.join(sample_dir, 'shap_modality_contribution.png')
        plot_modality_contribution(
            contributions=contributions_list,
            target_names=DISPLAY_NAMES,
            save_path=chart_path,
            dpi=DEFAULT_DPI,
        )

        # ── Save raw SHAP values as .npz ─────────────────────────────────
        npz_path = os.path.join(sample_dir, 'raw_shap_values.npz')
        npz_data = {factor: shap_values_dict[factor] for factor in FACTOR_NAMES}
        np.savez(npz_path, **npz_data)
        print(f'[XAI] Saved raw SHAP values: {npz_path}')

        # ── Save modality contribution as JSON ───────────────────────────
        contrib_json_path = os.path.join(
            sample_dir, 'shap_modality_contribution.json'
        )
        contrib_json = {}
        for t_idx, factor in enumerate(FACTOR_NAMES):
            c = contributions_list[t_idx]
            contrib_json[factor] = {
                'display_name': DISPLAY_NAMES[t_idx],
                'text_pct': round(c['text_pct'], 2),
                'image_pct': round(c['image_pct'], 2),
                'text_abs': round(c['text_abs'], 6),
                'image_abs': round(c['image_abs'], 6),
                'text_signed': round(c['text_signed'], 6),
                'image_signed': round(c['image_signed'], 6),
            }
        with open(contrib_json_path, 'w', encoding='utf-8') as f:
            json.dump(contrib_json, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved modality contribution: {contrib_json_path}')

        # ── Save metadata JSON ───────────────────────────────────────────
        meta_path = os.path.join(sample_dir, 'metadata.json')
        metadata = {
            'sample_id': sample_id,
            'fused_dim': FUSED_DIM,
            'text_origin_dims': f'0:{CROSS_ATTN_HIDDEN_DIM}',
            'image_origin_dims': f'{CROSS_ATTN_HIDDEN_DIM}:{FUSED_DIM}',
            'num_targets': NUM_TARGETS,
            'target_names': TARGET_NAMES,
            'factor_names': FACTOR_NAMES,
            'display_names': DISPLAY_NAMES,
            'shap_method': 'DeepExplainer',
            'background_size': int(background.size(0)),
            'base_values': {f: round(v, 6) for f, v in base_values_dict.items()},
            'additivity_checks': {
                f: {'passes': bool(p), 'error': round(e, 6)}
                for f, (p, e) in additivity_dict.items()
            },
            'device': str(self.device),
            'timestamp': datetime.datetime.now().isoformat(),
            'artifacts': {
                'modality_chart': chart_path,
                'modality_json': contrib_json_path,
                'raw_shap': npz_path,
            },
            'note': (
                'Text-origin dims (0:512) come from cross-attention where '
                'text queries attend to image. Image-origin dims (512:1024) '
                'come from cross-attention where image queries attend to text. '
                'Both channels contain mixed-modality information.'
            ),
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved metadata: {meta_path}')

        print(f'[XAI] SHAP explanation complete for {sample_id}')

        return {
            'sample_id': sample_id,
            'shap_values': shap_values_dict,
            'base_values': base_values_dict,
            'contributions': contributions_list,
            'additivity': additivity_dict,
            'paths': {
                'sample_dir': sample_dir,
                'modality_chart': chart_path,
                'modality_json': contrib_json_path,
                'raw_shap': npz_path,
                'metadata': meta_path,
            },
        }

    def explain_batch(
        self,
        samples_fused: torch.Tensor,
        sample_ids: List[str],
        background: torch.Tensor,
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for a batch of samples.

        Calls :meth:`explain_sample` for each sample, collects results,
        and computes aggregate statistics across the batch.

        Args:
            samples_fused: Fused embeddings [N, 1024].
            sample_ids: List of N sample identifiers.
            background: Background embeddings [n_bg, 1024].

        Returns:
            Dictionary with:
              - sample_results: list of per-sample result dicts.
              - aggregate: dict with mean contributions per target.
              - aggregate_chart_path: path to aggregate contribution chart.
        """
        n_samples = samples_fused.size(0)
        assert len(sample_ids) == n_samples, (
            f'sample_ids length ({len(sample_ids)}) != '
            f'samples_fused size ({n_samples})'
        )

        print(f'[XAI] Explaining batch of {n_samples} samples')

        sample_results: List[Dict[str, Any]] = []

        for i in range(n_samples):
            result = self.explain_sample(
                sample_fused=samples_fused[i],
                sample_id=sample_ids[i],
                background=background,
            )
            sample_results.append(result)

        # ── Aggregate contributions across batch ─────────────────────────
        # Mean text_pct and image_pct for each target across all samples
        aggregate_contribs: List[Dict[str, Any]] = []
        for t_idx in range(NUM_TARGETS):
            text_pcts = [r['contributions'][t_idx]['text_pct'] for r in sample_results]
            image_pcts = [r['contributions'][t_idx]['image_pct'] for r in sample_results]
            aggregate_contribs.append({
                'text_pct': float(np.mean(text_pcts)),
                'image_pct': float(np.mean(image_pcts)),
                'text_pct_std': float(np.std(text_pcts)),
                'image_pct_std': float(np.std(image_pcts)),
            })

        # Aggregate chart
        agg_chart_path = os.path.join(
            self.output_dir, 'shap_aggregate_modality_contribution.png'
        )
        plot_modality_contribution(
            contributions=aggregate_contribs,
            target_names=DISPLAY_NAMES,
            save_path=agg_chart_path,
            dpi=DEFAULT_DPI,
        )

        # Save aggregate JSON
        agg_json_path = os.path.join(
            self.output_dir, 'shap_aggregate_modality_contribution.json'
        )
        agg_json = {}
        for t_idx, factor in enumerate(FACTOR_NAMES):
            c = aggregate_contribs[t_idx]
            agg_json[factor] = {
                'display_name': DISPLAY_NAMES[t_idx],
                'mean_text_pct': round(c['text_pct'], 2),
                'mean_image_pct': round(c['image_pct'], 2),
                'std_text_pct': round(c['text_pct_std'], 2),
                'std_image_pct': round(c['image_pct_std'], 2),
                'n_samples': n_samples,
            }
        with open(agg_json_path, 'w', encoding='utf-8') as f:
            json.dump(agg_json, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved aggregate contribution: {agg_json_path}')

        print(f'[XAI] Batch explanation complete ({n_samples} samples)')

        return {
            'sample_results': sample_results,
            'aggregate': aggregate_contribs,
            'aggregate_chart_path': agg_chart_path,
            'aggregate_json_path': agg_json_path,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 10. Ablation Check — zero-out modality sanity check
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation_check(
    model_head: nn.Sequential,
    sample_fused: torch.Tensor,
    background_mean: torch.Tensor,
    text_dim: int = CROSS_ATTN_HIDDEN_DIM,
) -> Dict[str, Any]:
    """Zero-out ablation to verify modality importance direction.

    Independently zeroes out text-origin and image-origin dimensions,
    replacing them with the background mean.  Compares the resulting
    predictions to the full prediction to quantify each modality's
    contribution via prediction drop.

    This is a model-based sanity check (not SHAP-based) that should
    directionally agree with the SHAP modality contributions.

    Args:
        model_head: The ``model.head`` nn.Sequential module.
        sample_fused: Single fused embedding [1024] or [1, 1024].
        background_mean: Mean of background embeddings [1024] or [1, 1024].
        text_dim: Number of text-origin dimensions (default 512).

    Returns:
        Dictionary with per-target prediction drops:
          - full_prediction: [5] full model prediction
          - text_zeroed_prediction: [5] prediction with text dims replaced
          - image_zeroed_prediction: [5] prediction with image dims replaced
          - text_zeroed_drop: [5] prediction drop from zeroing text
          - image_zeroed_drop: [5] prediction drop from zeroing image
          - text_drop_pct: [5] percentage drop from zeroing text
          - image_drop_pct: [5] percentage drop from zeroing image
    """
    # Ensure correct shapes
    if sample_fused.ndim == 1:
        sample_fused = sample_fused.unsqueeze(0)
    if background_mean.ndim == 1:
        background_mean = background_mean.unsqueeze(0)

    device = next(model_head.parameters()).device
    sample = sample_fused.to(device)
    bg_mean = background_mean.to(device)

    model_head.eval()

    with torch.no_grad():
        # Full prediction
        full_pred = model_head(sample).squeeze(0).cpu().numpy()  # [5]

        # Zero out text-origin dims (replace with background mean)
        text_zeroed = sample.clone()
        text_zeroed[:, :text_dim] = bg_mean[:, :text_dim]
        text_zeroed_pred = model_head(text_zeroed).squeeze(0).cpu().numpy()

        # Zero out image-origin dims (replace with background mean)
        image_zeroed = sample.clone()
        image_zeroed[:, text_dim:] = bg_mean[:, text_dim:]
        image_zeroed_pred = model_head(image_zeroed).squeeze(0).cpu().numpy()

    # Compute drops
    text_drop = full_pred - text_zeroed_pred     # positive = text contributed positively
    image_drop = full_pred - image_zeroed_pred   # positive = image contributed positively

    # Percentage drops (relative to full prediction magnitude)
    safe_full = np.where(np.abs(full_pred) > 1e-8, np.abs(full_pred), 1.0)
    text_drop_pct = (np.abs(text_drop) / safe_full) * 100.0
    image_drop_pct = (np.abs(image_drop) / safe_full) * 100.0

    result = {
        'full_prediction': full_pred.tolist(),
        'text_zeroed_prediction': text_zeroed_pred.tolist(),
        'image_zeroed_prediction': image_zeroed_pred.tolist(),
        'text_zeroed_drop': text_drop.tolist(),
        'image_zeroed_drop': image_drop.tolist(),
        'text_drop_pct': text_drop_pct.tolist(),
        'image_drop_pct': image_drop_pct.tolist(),
        'factor_names': FACTOR_NAMES,
        'display_names': DISPLAY_NAMES,
    }

    print('[XAI] Ablation check results:')
    for t_idx in range(NUM_TARGETS):
        print(f'[XAI]   {FACTOR_NAMES[t_idx]:>8s}: '
              f'full={full_pred[t_idx]:.3f}, '
              f'text_zeroed={text_zeroed_pred[t_idx]:.3f} '
              f'(drop={text_drop_pct[t_idx]:.1f}%), '
              f'image_zeroed={image_zeroed_pred[t_idx]:.3f} '
              f'(drop={image_drop_pct[t_idx]:.1f}%)')

    return result
