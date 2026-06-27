"""
LIME-based explainer for text and image modalities of the multimodal model.

Phase 5 of the XAI pipeline. Uses LIME (Local Interpretable Model-agnostic
Explanations) to generate per-feature importance explanations for both the
text (word-level) and image (superpixel-level) inputs independently.

Unlike Phase 4 (SHAP on fused embeddings), LIME operates on the raw inputs:
it perturbs words in the review text and superpixels in the food images,
observing how the model's predicted quality score changes. This provides
human-interpretable explanations: "which words / image regions most
influenced the predicted food_score?"

Because the model outputs raw regression scores (not probabilities), we
apply sigmoid to convert each target score into a pseudo-classification
probability p_high, then return [1-p_high, p_high] as the two-column output
that LIME's classification explainer expects.

Usage:
    from xai.utils import load_model, load_single_sample, get_tokenizer, get_image_processor
    from xai.lime_explainer import LIMEExplainer

    model, config = load_model(exp_dir)
    tokenizer = get_tokenizer()
    image_processor = get_image_processor()
    sample = load_single_sample(csv_path, idx=0, tokenizer=tokenizer,
                                 image_processor=image_processor)

    explainer = LIMEExplainer(model, tokenizer, image_processor, device,
                               output_dir='./xai_output/lime')
    results = explainer.explain_sample(sample, sample_id='sample_000')
"""

import os
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
    NUM_TARGETS,
    DEFAULT_DPI,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_IMAGES,
    DEFAULT_SEED,
    COLOR_SCHEMES,
)

# ── Graceful import of lime ────────────────────────────────────────────────
try:
    from lime.lime_image import LimeImageExplainer
    from lime.lime_text import LimeTextExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

try:
    from skimage.segmentation import mark_boundaries
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


def _check_lime() -> None:
    """Raise ImportError with install instructions if lime is missing."""
    if not _LIME_AVAILABLE:
        raise ImportError(
            'The `lime` library is required for Phase 5 but is not installed.\n'
            'Install it with:  pip install lime\n'
            'You may also need scikit-image:  pip install scikit-image'
        )


def _check_skimage() -> None:
    """Raise ImportError with install instructions if skimage is missing."""
    if not _SKIMAGE_AVAILABLE:
        raise ImportError(
            'The `scikit-image` library is required for LIME image explanations.\n'
            'Install it with:  pip install scikit-image'
        )


# ══════════════════════════════════════════════════════════════════════════════
# 1. ImageLimePredictFn — callable for LIME image explanations
# ══════════════════════════════════════════════════════════════════════════════

class ImageLimePredictFn:
    """Callable that LIME's image explainer calls with perturbed numpy images.

    LIME generates N perturbed versions of the original image (masking out
    different superpixels) and passes them as a numpy array [N, H, W, 3].
    This callable preprocesses each image through the same TimmProcessor
    used during training, places the perturbed image at index 0 of the
    multi-image tensor (keeping other images fixed), runs the model, and
    returns pseudo-classification probabilities.

    Args:
        model: The full multimodal model in eval mode.
        fixed_input_ids: Text input_ids from the sample [1, seq_len].
        fixed_attention_mask: Text attention_mask from the sample [1, seq_len].
        fixed_pixel_values: Original pixel_values [1, N_img, C, H, W].
        fixed_num_images: Number of real images tensor [1].
        score_index: Which of the 5 target scores to explain (0-4).
        device: Torch device.
        image_processor: TimmProcessor or AutoImageProcessor instance.
        batch_size: Internal batch size to avoid GPU OOM.
    """

    def __init__(
        self,
        model: nn.Module,
        fixed_input_ids: torch.Tensor,
        fixed_attention_mask: torch.Tensor,
        fixed_pixel_values: torch.Tensor,
        fixed_num_images: torch.Tensor,
        score_index: int,
        device: torch.device,
        image_processor,
        batch_size: int = 32,
    ):
        self.model = model
        self.fixed_input_ids = fixed_input_ids
        self.fixed_attention_mask = fixed_attention_mask
        self.fixed_pixel_values = fixed_pixel_values
        self.fixed_num_images = fixed_num_images
        self.score_index = score_index
        self.device = device
        self.image_processor = image_processor
        self.batch_size = batch_size

    def __call__(self, images_np: np.ndarray) -> np.ndarray:
        """Process perturbed images and return pseudo-classification probs.

        Args:
            images_np: Perturbed images from LIME [N, H, W, 3].
                       May be uint8 [0, 255] or float [0.0, 1.0].

        Returns:
            Probability array [N, 2] where column 0 is P(low) and
            column 1 is P(high) for the target score.
        """
        n_samples = images_np.shape[0]
        all_probs = []

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_np = images_np[start:end]
            batch_size = end - start

            # Convert numpy images to PIL
            pil_images = []
            for i in range(batch_size):
                img = batch_np[i]
                # Handle both uint8 [0, 255] and float [0, 1] ranges
                if img.dtype in (np.float32, np.float64):
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img, mode='RGB'))

            # Process through image_processor (same transform as training)
            processed = self.image_processor(pil_images, return_tensors='pt')
            perturbed_pixels = processed['pixel_values']  # [batch_size, C, H, W]

            # Build multi-image tensor: perturbed at index 0, rest from fixed
            # fixed_pixel_values is [1, N_img, C, H, W]
            n_img = self.fixed_pixel_values.shape[1]
            c, h, w = perturbed_pixels.shape[1], perturbed_pixels.shape[2], perturbed_pixels.shape[3]

            # Start with fixed images repeated for the batch
            pixel_batch = self.fixed_pixel_values.expand(batch_size, -1, -1, -1, -1).clone()
            # Replace index 0 with perturbed images
            pixel_batch[:, 0, :, :, :] = perturbed_pixels

            pixel_batch = pixel_batch.to(self.device)

            # Repeat fixed text inputs for the batch
            input_ids = self.fixed_input_ids.expand(batch_size, -1).to(self.device)
            attention_mask = self.fixed_attention_mask.expand(batch_size, -1).to(self.device)
            num_images = self.fixed_num_images.expand(batch_size).to(self.device)

            # Run model
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_batch,
                    num_images=num_images,
                )
                preds = output[0] if isinstance(output, tuple) else output

            # Extract target score and apply sigmoid for pseudo-classification
            target_scores = preds[:, self.score_index]  # [batch_size]
            p_high = torch.sigmoid(target_scores).cpu().numpy()  # [batch_size]
            probs = np.stack([1.0 - p_high, p_high], axis=1)  # [batch_size, 2]
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)  # [N, 2]


# ══════════════════════════════════════════════════════════════════════════════
# 2. TextLimePredictFn — callable for LIME text explanations
# ══════════════════════════════════════════════════════════════════════════════

class TextLimePredictFn:
    """Callable that LIME's text explainer calls with perturbed text strings.

    LIME generates N perturbed versions of the original text (masking out
    different words) and passes them as a list of strings. This callable
    tokenizes each string, pairs it with the fixed image inputs, runs the
    model, and returns pseudo-classification probabilities.

    Args:
        model: The full multimodal model in eval mode.
        fixed_pixel_values: Original pixel_values [1, N_img, C, H, W].
        fixed_num_images: Number of real images tensor [1].
        tokenizer: HuggingFace tokenizer (e.g., PhoBERT).
        max_length: Maximum token sequence length.
        score_index: Which of the 5 target scores to explain (0-4).
        device: Torch device.
        batch_size: Internal batch size to avoid GPU OOM.
    """

    def __init__(
        self,
        model: nn.Module,
        fixed_pixel_values: torch.Tensor,
        fixed_num_images: torch.Tensor,
        tokenizer,
        max_length: int,
        score_index: int,
        device: torch.device,
        batch_size: int = 32,
    ):
        self.model = model
        self.fixed_pixel_values = fixed_pixel_values
        self.fixed_num_images = fixed_num_images
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.score_index = score_index
        self.device = device
        self.batch_size = batch_size

    def __call__(self, text_list: List[str]) -> np.ndarray:
        """Process perturbed texts and return pseudo-classification probs.

        Args:
            text_list: List of N perturbed text strings.

        Returns:
            Probability array [N, 2] where column 0 is P(low) and
            column 1 is P(high) for the target score.
        """
        n_samples = len(text_list)
        all_probs = []

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_texts = text_list[start:end]
            batch_size = end - start

            # Tokenize all texts in the batch
            encoded = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            )

            input_ids = encoded['input_ids'].to(self.device)        # [batch_size, seq_len]
            attention_mask = encoded['attention_mask'].to(self.device)  # [batch_size, seq_len]

            # Repeat fixed images for the batch
            pixel_values = self.fixed_pixel_values.expand(
                batch_size, -1, -1, -1, -1
            ).to(self.device)
            num_images = self.fixed_num_images.expand(batch_size).to(self.device)

            # Run model
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    num_images=num_images,
                )
                preds = output[0] if isinstance(output, tuple) else output

            # Extract target score and apply sigmoid for pseudo-classification
            target_scores = preds[:, self.score_index]  # [batch_size]
            p_high = torch.sigmoid(target_scores).cpu().numpy()  # [batch_size]
            probs = np.stack([1.0 - p_high, p_high], axis=1)  # [batch_size, 2]
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)  # [N, 2]


# ══════════════════════════════════════════════════════════════════════════════
# 3. run_lime_image — generate LIME image explanation
# ══════════════════════════════════════════════════════════════════════════════

def run_lime_image(
    model: nn.Module,
    sample: Dict[str, Any],
    score_index: int,
    image_processor,
    device: torch.device,
    num_samples: int = 1000,
    seed: int = DEFAULT_SEED,
):
    """Generate a LIME explanation for the first image of a sample.

    Creates an ImageLimePredictFn, builds a LimeImageExplainer, and calls
    explain_instance on the first real image (resized to 224x224).

    Args:
        model: The full multimodal model in eval mode.
        sample: Output from load_single_sample().
        score_index: Which of the 5 target scores to explain (0-4).
        image_processor: TimmProcessor or AutoImageProcessor.
        device: Torch device.
        num_samples: Number of perturbed images for LIME (default 1000).
        seed: Random seed for reproducibility.

    Returns:
        LIME ImageExplanation object.

    Raises:
        ImportError: If lime is not installed.
        ValueError: If the sample has no real images.
    """
    _check_lime()

    if sample['num_real_images'] < 1:
        raise ValueError(
            'Cannot run LIME image explanation: sample has no real images.'
        )

    # Get original image as numpy [H, W, 3] uint8
    original_image = sample['loaded_images'][0].copy()
    original_image = original_image.resize((224, 224), Image.BILINEAR)
    image_np = np.array(original_image).astype(np.uint8)  # [224, 224, 3]

    # Create predict function
    predict_fn = ImageLimePredictFn(
        model=model,
        fixed_input_ids=sample['input_ids'],
        fixed_attention_mask=sample['attention_mask'],
        fixed_pixel_values=sample['pixel_values'],
        fixed_num_images=sample['num_images'],
        score_index=score_index,
        device=device,
        image_processor=image_processor,
    )

    # Create LIME explainer and explain
    explainer = LimeImageExplainer(random_state=seed)

    print(f'[XAI] Running LIME image explanation '
          f'(target={FACTOR_NAMES[score_index]}, num_samples={num_samples})...')

    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        random_seed=seed,
    )

    print(f'[XAI] LIME image explanation complete for {FACTOR_NAMES[score_index]}')

    return explanation


# ══════════════════════════════════════════════════════════════════════════════
# 4. run_lime_text — generate LIME text explanation
# ══════════════════════════════════════════════════════════════════════════════

def run_lime_text(
    model: nn.Module,
    sample: Dict[str, Any],
    score_index: int,
    tokenizer,
    device: torch.device,
    num_features: int = 10,
    num_samples: int = 500,
    seed: int = DEFAULT_SEED,
):
    """Generate a LIME explanation for the text of a sample.

    Creates a TextLimePredictFn, builds a LimeTextExplainer with
    split_expression=r'\\s+' (matching Vietnamese syllable boundaries),
    and calls explain_instance.

    Args:
        model: The full multimodal model in eval mode.
        sample: Output from load_single_sample().
        score_index: Which of the 5 target scores to explain (0-4).
        tokenizer: HuggingFace tokenizer (e.g., PhoBERT).
        device: Torch device.
        num_features: Number of top features to include (default 10).
        num_samples: Number of perturbed texts for LIME (default 500).
        seed: Random seed for reproducibility.

    Returns:
        LIME TextExplanation object.

    Raises:
        ImportError: If lime is not installed.
    """
    _check_lime()

    original_text = sample['text']

    # Create predict function
    predict_fn = TextLimePredictFn(
        model=model,
        fixed_pixel_values=sample['pixel_values'],
        fixed_num_images=sample['num_images'],
        tokenizer=tokenizer,
        max_length=DEFAULT_MAX_LENGTH,
        score_index=score_index,
        device=device,
    )

    # Create LIME text explainer with Vietnamese syllable splitting
    explainer = LimeTextExplainer(
        split_expression=r'\s+',
        random_state=seed,
    )

    print(f'[XAI] Running LIME text explanation '
          f'(target={FACTOR_NAMES[score_index]}, num_samples={num_samples}, '
          f'num_features={num_features})...')

    explanation = explainer.explain_instance(
        original_text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,),
    )

    print(f'[XAI] LIME text explanation complete for {FACTOR_NAMES[score_index]}')

    return explanation


# ══════════════════════════════════════════════════════════════════════════════
# 5. save_lime_image_explanation — save image explanation artifacts
# ══════════════════════════════════════════════════════════════════════════════

def save_lime_image_explanation(
    explanation,
    original_image: Image.Image,
    save_dir: str,
    sample_id: str,
    target_idx: int,
    factor_name: str,
    dpi: int = DEFAULT_DPI,
) -> Dict[str, str]:
    """Save LIME image explanation artifacts: overlays, weights JSON.

    Generates three overlay images (positive-only, negative-only, combined)
    using superpixel boundaries and saves the superpixel weights as JSON.

    Args:
        explanation: LIME ImageExplanation object.
        original_image: Original PIL image (will be resized to 224x224).
        save_dir: Directory to save artifacts into.
        sample_id: Sample identifier for filenames.
        target_idx: Target index (0-4) for filenames.
        factor_name: Factor name (e.g., 'food') for filenames.
        dpi: Resolution for saved figures.

    Returns:
        Dictionary of saved file paths with keys: 'positive_overlay',
        'negative_overlay', 'combined_overlay', 'weights_json'.
    """
    _check_skimage()
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Resize original image to match LIME's working size
    original_resized = original_image.copy().resize((224, 224), Image.BILINEAR)
    original_np = np.array(original_resized).astype(np.uint8)

    # Determine the label to use for get_image_and_mask
    # LIME may use label 1 (high class) from our pseudo-classification
    available_labels = explanation.top_labels
    label = 1 if 1 in available_labels else available_labels[0]

    paths = {}

    # ── Positive superpixels overlay ──────────────────────────────────────
    temp_pos, mask_pos = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )
    img_pos = mark_boundaries(temp_pos / 255.0, mask_pos)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_pos)
    ax.set_title(f'LIME Positive Regions — {DISPLAY_NAMES[target_idx]}',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    pos_path = os.path.join(
        save_dir,
        f'{sample_id}_lime_image_{factor_name}_positive.png',
    )
    fig.savefig(pos_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    paths['positive_overlay'] = pos_path
    print(f'[XAI] Saved: {pos_path}')

    # ── Negative superpixels overlay ──────────────────────────────────────
    temp_neg, mask_neg = explanation.get_image_and_mask(
        label,
        positive_only=False,
        negative_only=True,
        num_features=10,
        hide_rest=False,
    )
    img_neg = mark_boundaries(temp_neg / 255.0, mask_neg)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_neg)
    ax.set_title(f'LIME Negative Regions — {DISPLAY_NAMES[target_idx]}',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    neg_path = os.path.join(
        save_dir,
        f'{sample_id}_lime_image_{factor_name}_negative.png',
    )
    fig.savefig(neg_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    paths['negative_overlay'] = neg_path
    print(f'[XAI] Saved: {neg_path}')

    # ── Combined overlay (positive green, negative red) ───────────────────
    temp_combined, mask_combined = explanation.get_image_and_mask(
        label,
        positive_only=False,
        num_features=10,
        hide_rest=False,
    )
    img_combined = mark_boundaries(temp_combined / 255.0, mask_combined)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_combined)
    ax.set_title(f'LIME Combined Regions — {DISPLAY_NAMES[target_idx]}',
                 fontsize=11, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    combined_path = os.path.join(
        save_dir,
        f'{sample_id}_lime_image_{factor_name}_combined.png',
    )
    fig.savefig(combined_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    paths['combined_overlay'] = combined_path
    print(f'[XAI] Saved: {combined_path}')

    # ── Superpixel weights as JSON ────────────────────────────────────────
    local_exp = explanation.local_exp.get(label, [])
    weights = {
        'label': int(label),
        'factor_name': factor_name,
        'display_name': DISPLAY_NAMES[target_idx],
        'intercept': float(explanation.intercept.get(label, 0.0)),
        'score': float(explanation.score) if hasattr(explanation, 'score') else None,
        'superpixel_weights': [
            {'superpixel_id': int(sp_id), 'weight': float(weight)}
            for sp_id, weight in sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)
        ],
    }

    weights_path = os.path.join(
        save_dir,
        f'{sample_id}_lime_image_{factor_name}_weights.json',
    )
    with open(weights_path, 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    paths['weights_json'] = weights_path
    print(f'[XAI] Saved: {weights_path}')

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 6. save_lime_text_explanation — save text explanation artifacts
# ══════════════════════════════════════════════════════════════════════════════

def save_lime_text_explanation(
    explanation,
    save_dir: str,
    sample_id: str,
    target_idx: int,
    factor_name: str,
    dpi: int = DEFAULT_DPI,
) -> Dict[str, str]:
    """Save LIME text explanation artifacts: bar chart, weights JSON, HTML.

    Generates a horizontal bar chart of word importances (green for positive
    contribution, red for negative), saves word weights as JSON, and saves
    the LIME HTML visualization.

    Args:
        explanation: LIME TextExplanation object.
        save_dir: Directory to save artifacts into.
        sample_id: Sample identifier for filenames.
        target_idx: Target index (0-4) for filenames.
        factor_name: Factor name (e.g., 'food') for filenames.
        dpi: Resolution for saved figures.

    Returns:
        Dictionary of saved file paths with keys: 'bar_chart',
        'weights_json', 'html'.
    """
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Use label 1 (high class) for text explanations
    label = 1
    local_exp = explanation.as_list(label=label)

    paths = {}

    # ── Horizontal bar chart of word importances ──────────────────────────
    if local_exp:
        words = [pair[0] for pair in local_exp]
        weights_vals = [pair[1] for pair in local_exp]

        # Sort by absolute weight for display
        sorted_pairs = sorted(
            zip(words, weights_vals),
            key=lambda x: abs(x[1]),
            reverse=False,  # smallest at top for horizontal bar
        )
        sorted_words = [p[0] for p in sorted_pairs]
        sorted_weights = [p[1] for p in sorted_pairs]

        # Color: green for positive, red for negative
        colors = ['#4CAF50' if w >= 0 else '#F44336' for w in sorted_weights]

        fig_height = max(4, len(sorted_words) * 0.4 + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        y_pos = np.arange(len(sorted_words))
        ax.barh(y_pos, sorted_weights, color=colors, edgecolor='white', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_words, fontsize=10)
        ax.set_xlabel('Weight (contribution to prediction)', fontsize=11)
        ax.set_title(
            f'LIME Word Importance — {DISPLAY_NAMES[target_idx]}',
            fontsize=12,
            fontweight='bold',
        )
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
        ax.grid(axis='x', alpha=0.3)

        fig.tight_layout()

        bar_path = os.path.join(
            save_dir,
            f'{sample_id}_lime_text_{factor_name}_bar.png',
        )
        fig.savefig(bar_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        paths['bar_chart'] = bar_path
        print(f'[XAI] Saved: {bar_path}')
    else:
        print(f'[XAI] WARNING: No word importances for {factor_name}')

    # ── Word weights as JSON ──────────────────────────────────────────────
    weights_data = {
        'label': label,
        'factor_name': factor_name,
        'display_name': DISPLAY_NAMES[target_idx],
        'intercept': float(explanation.intercept.get(label, 0.0))
            if hasattr(explanation, 'intercept') and isinstance(explanation.intercept, dict)
            else float(explanation.intercept[label]) if hasattr(explanation, 'intercept') else None,
        'score': float(explanation.score) if hasattr(explanation, 'score') else None,
        'word_weights': [
            {'word': word, 'weight': float(weight)}
            for word, weight in sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)
        ],
    }

    weights_path = os.path.join(
        save_dir,
        f'{sample_id}_lime_text_{factor_name}_weights.json',
    )
    with open(weights_path, 'w', encoding='utf-8') as f:
        json.dump(weights_data, f, indent=2, ensure_ascii=False)
    paths['weights_json'] = weights_path
    print(f'[XAI] Saved: {weights_path}')

    # ── HTML visualization ────────────────────────────────────────────────
    try:
        html_content = explanation.as_html(labels=(label,))
        html_path = os.path.join(
            save_dir,
            f'{sample_id}_lime_text_{factor_name}.html',
        )
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        paths['html'] = html_path
        print(f'[XAI] Saved: {html_path}')
    except Exception as e:
        print(f'[XAI] WARNING: Could not save HTML for {factor_name}: {e}')

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 7. LIMEExplainer — High-Level Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class LIMEExplainer:
    """High-level LIME explainer for both text and image modalities.

    Orchestrates LIME explanation generation for all (or specified) target
    scores, saves visualization artifacts and metadata. Analogous to
    GradCAMExplainer (Phase 2), AttentionExplainer (Phase 3), and
    SHAPExplainer (Phase 4).

    Args:
        model: The full multimodal model (e.g., CrossAttentionFusion),
               in eval mode.
        tokenizer: HuggingFace tokenizer (e.g., PhoBERT).
        image_processor: Image processor (TimmProcessor or AutoImageProcessor).
        device: Torch device.
        output_dir: Base directory for saving artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        image_processor,
        device: torch.device,
        output_dir: str = './xai_output/lime',
    ):
        _check_lime()

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.output_dir = output_dir

        # Ensure model is in eval mode
        self.model.eval()

        os.makedirs(output_dir, exist_ok=True)
        print(f'[XAI] LIMEExplainer initialized. Output dir: {output_dir}')

    def explain_sample(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        targets: Optional[List[int]] = None,
        do_image: bool = True,
        do_text: bool = True,
        num_image_samples: int = 1000,
        num_text_samples: int = 500,
        seed: int = DEFAULT_SEED,
    ) -> Dict[str, Any]:
        """Generate full LIME explanation for a single sample.

        For each target (or specified targets):
          1. If do_image and sample has images: run_lime_image, save artifacts
          2. If do_text: run_lime_text, save artifacts

        Saves sample metadata JSON summarizing all generated artifacts.

        Output folder structure::

            {output_dir}/{sample_id}/
              {sample_id}_lime_image_{factor}_positive.png
              {sample_id}_lime_image_{factor}_negative.png
              {sample_id}_lime_image_{factor}_combined.png
              {sample_id}_lime_image_{factor}_weights.json
              {sample_id}_lime_text_{factor}_bar.png
              {sample_id}_lime_text_{factor}_weights.json
              {sample_id}_lime_text_{factor}.html
              metadata.json

        Args:
            sample: Output from load_single_sample().
            sample_id: Unique identifier (e.g., 'sample_042').
            targets: List of target indices to explain (0-4). If None,
                     explains all 5 targets.
            do_image: Whether to generate image LIME explanations.
            do_text: Whether to generate text LIME explanations.
            num_image_samples: Number of perturbed images for LIME.
            num_text_samples: Number of perturbed texts for LIME.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with:
              - sample_id: str
              - image_explanations: dict mapping factor_name -> explanation data
              - text_explanations: dict mapping factor_name -> explanation data
              - paths: dict with all artifact paths per target
              - metadata_path: path to metadata JSON
        """
        sample_dir = os.path.join(self.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        if targets is None:
            targets = list(range(NUM_TARGETS))

        has_images = sample['num_real_images'] > 0

        print(f'[XAI] Explaining {sample_id} (LIME analysis, '
              f'{len(targets)} targets, image={do_image and has_images}, '
              f'text={do_text})')

        image_results: Dict[str, Any] = {}
        text_results: Dict[str, Any] = {}
        all_paths: Dict[str, Dict[str, Any]] = {}

        for t_idx in targets:
            if t_idx < 0 or t_idx >= NUM_TARGETS:
                print(f'[XAI] WARNING: Skipping invalid target index {t_idx}')
                continue

            factor = FACTOR_NAMES[t_idx]
            display = DISPLAY_NAMES[t_idx]
            target_paths: Dict[str, Any] = {}

            print(f'[XAI] === Target {t_idx} ({factor}: {display}) ===')

            # ── Image LIME ────────────────────────────────────────────────
            if do_image and has_images:
                try:
                    img_explanation = run_lime_image(
                        model=self.model,
                        sample=sample,
                        score_index=t_idx,
                        image_processor=self.image_processor,
                        device=self.device,
                        num_samples=num_image_samples,
                        seed=seed,
                    )

                    img_paths = save_lime_image_explanation(
                        explanation=img_explanation,
                        original_image=sample['loaded_images'][0],
                        save_dir=sample_dir,
                        sample_id=sample_id,
                        target_idx=t_idx,
                        factor_name=factor,
                        dpi=DEFAULT_DPI,
                    )

                    image_results[factor] = {
                        'explanation': img_explanation,
                        'paths': img_paths,
                    }
                    target_paths['image'] = img_paths

                except Exception as e:
                    print(f'[XAI] ERROR in LIME image for {factor}: {e}')
                    image_results[factor] = {'error': str(e)}

            elif do_image and not has_images:
                print(f'[XAI] Skipping image LIME for {factor}: no real images')

            # ── Text LIME ─────────────────────────────────────────────────
            if do_text:
                try:
                    txt_explanation = run_lime_text(
                        model=self.model,
                        sample=sample,
                        score_index=t_idx,
                        tokenizer=self.tokenizer,
                        device=self.device,
                        num_features=10,
                        num_samples=num_text_samples,
                        seed=seed,
                    )

                    txt_paths = save_lime_text_explanation(
                        explanation=txt_explanation,
                        save_dir=sample_dir,
                        sample_id=sample_id,
                        target_idx=t_idx,
                        factor_name=factor,
                        dpi=DEFAULT_DPI,
                    )

                    text_results[factor] = {
                        'explanation': txt_explanation,
                        'paths': txt_paths,
                    }
                    target_paths['text'] = txt_paths

                except Exception as e:
                    print(f'[XAI] ERROR in LIME text for {factor}: {e}')
                    text_results[factor] = {'error': str(e)}

            all_paths[factor] = target_paths

        # ── Save metadata JSON ────────────────────────────────────────────
        metadata = {
            'sample_id': sample_id,
            'sample_idx': sample.get('sample_idx', None),
            'csv_path': sample.get('csv_path', None),
            'text': sample['text'],
            'num_real_images': sample['num_real_images'],
            'targets_explained': [FACTOR_NAMES[t] for t in targets],
            'do_image': do_image,
            'do_text': do_text,
            'num_image_samples': num_image_samples,
            'num_text_samples': num_text_samples,
            'seed': seed,
            'device': str(self.device),
            'timestamp': datetime.datetime.now().isoformat(),
            'torch_version': torch.__version__,
            'target_names': TARGET_NAMES,
            'factor_names': FACTOR_NAMES,
            'display_names': DISPLAY_NAMES,
            'artifacts': {},
        }

        # Collect artifact paths into metadata (exclude explanation objects)
        for factor in FACTOR_NAMES:
            factor_artifacts = {}
            if factor in image_results and 'paths' in image_results[factor]:
                factor_artifacts['image'] = image_results[factor]['paths']
            if factor in text_results and 'paths' in text_results[factor]:
                factor_artifacts['text'] = text_results[factor]['paths']
            if factor_artifacts:
                metadata['artifacts'][factor] = factor_artifacts

        meta_path = os.path.join(sample_dir, 'metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved metadata: {meta_path}')

        print(f'[XAI] LIME explanation complete for {sample_id}')

        return {
            'sample_id': sample_id,
            'image_explanations': image_results,
            'text_explanations': text_results,
            'paths': all_paths,
            'metadata_path': meta_path,
        }
