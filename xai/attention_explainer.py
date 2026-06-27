"""
Attention-based explainer for the text branch (PhoBERT) of the multimodal model.

Phase 3 of the XAI pipeline. Extracts self-attention weights from PhoBERT,
aggregates them across layers/heads, and visualizes token-level importance
for Vietnamese restaurant review quality assessment.

NOTE on cross-attention: The CrossAttentionFusion model now uses token-level ×
patch-level cross-attention (Q=[B,T,512], K=[B,P,512] → attention [B,8,T,P]).
Cross-attention weights are informative and visualizable. This module currently
focuses on PhoBERT self-attention; cross-attention visualization is planned for
a future Phase 3 update.

Usage:
    from xai.utils import load_model, load_single_sample
    from xai.attention_explainer import AttentionExplainer

    model, config = load_model(exp_dir)
    sample = load_single_sample(csv_path, idx=0, tokenizer=tok, image_processor=proc)

    explainer = AttentionExplainer(model, tokenizer, device, output_dir='./xai_output/attention')
    results = explainer.explain_sample(sample, sample_id='sample_000')
"""

import os
import json
import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from xai.config import (
    TARGET_NAMES,
    FACTOR_NAMES,
    DISPLAY_NAMES,
    NUM_TARGETS,
    PHOBERT_NUM_LAYERS,
    PHOBERT_NUM_HEADS,
    CROSS_ATTN_HIDDEN_DIM,
    DEFAULT_DPI,
    THESIS_DPI,
    COLOR_SCHEMES,
)


# ══════════════════════════════════════════════════════════════════════════════
# Special token sets for PhoBERT (RoBERTa-based)
# ══════════════════════════════════════════════════════════════════════════════

_PHOBERT_SPECIAL_TOKENS = {'<s>', '</s>', '<pad>', '<unk>', '<mask>'}


def _is_special_token(token: str) -> bool:
    """Check whether a token is a PhoBERT special token."""
    return token in _PHOBERT_SPECIAL_TOKENS


# ══════════════════════════════════════════════════════════════════════════════
# 1. Extract PhoBERT Self-Attention
# ══════════════════════════════════════════════════════════════════════════════

def extract_phobert_attention(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> Dict[str, Any]:
    """Extract self-attention weights from all 12 PhoBERT layers.

    Bypasses TextModel.forward() and calls model.text_model.encoder directly
    with output_attentions=True. Assumes eager attention is already enabled
    (via xai.utils.enable_eager_attention or manual patching in notebook).

    Args:
        model: The full multimodal model (e.g., CrossAttentionFusion).
               Must have model.text_model.encoder (HuggingFace RoBERTa).
        input_ids: Token IDs tensor [1, seq_len].
        attention_mask: Attention mask tensor [1, seq_len].
        tokenizer: HuggingFace tokenizer for decoding token IDs.

    Returns:
        Dictionary with:
          - 'attentions': numpy array [12, 12, seq_len, seq_len] (float32)
                          axes: [layer, head, query_token, key_token]
          - 'tokens': list of decoded token strings (length = seq_len)
          - 'seq_len': int, actual sequence length (excluding padding)
    """
    # Access the HuggingFace encoder through CrossAttentionFusion
    encoder = model.text_model.encoder

    with torch.no_grad():
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    # outputs.attentions is a tuple of 12 tensors, each [B, 12, L, L]
    raw_attentions = outputs.attentions
    if raw_attentions is None:
        raise RuntimeError(
            'encoder returned None attentions. Ensure eager attention is enabled '
            '(call xai.utils.enable_eager_attention before extraction).'
        )

    # Stack to [12, 12, L, L] (squeeze batch dim, which should be 1)
    stacked = torch.stack(raw_attentions, dim=0).squeeze(1)  # [12, B=1, 12, L, L] -> [12, 12, L, L]
    # If batch > 1, this only works for batch=1; the squeeze handles it
    if stacked.dim() == 5:
        stacked = stacked[:, 0, :, :, :]  # take first sample

    # Trim to actual sequence length (exclude padding)
    mask = attention_mask[0]  # [seq_len]
    seq_len = int(mask.sum().item())
    trimmed = stacked[:, :, :seq_len, :seq_len]  # [12, 12, seq_len, seq_len]

    # Decode tokens
    token_ids = input_ids[0, :seq_len].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f'[XAI] Extracted attention: {PHOBERT_NUM_LAYERS} layers, '
          f'{PHOBERT_NUM_HEADS} heads, seq_len={seq_len}')

    return {
        'attentions': trimmed.cpu().numpy().astype(np.float32),
        'tokens': tokens,
        'seq_len': seq_len,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Aggregate Attention Across Layers/Heads
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_attention(
    attentions: np.ndarray,
    strategy: str = 'last_layer_mean',
) -> np.ndarray:
    """Aggregate multi-head, multi-layer attention into a single [L, L] matrix.

    Args:
        attentions: Attention tensor [12, 12, L, L] (layers, heads, query, key).
        strategy: Aggregation strategy. One of:
            - 'last_layer_mean': Mean over 12 heads of layer 11.
            - 'last_4_layers_mean': Mean over layers 8-11, all heads.
            - 'attention_rollout': Multiply attention across layers with
              residual mixing (Abnar & Zuidema, 2020).

    Returns:
        Aggregated attention matrix [L, L].

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy == 'last_layer_mean':
        # Layer 11 (0-indexed), mean across 12 heads
        return attentions[PHOBERT_NUM_LAYERS - 1].mean(axis=0)  # [L, L]

    elif strategy == 'last_4_layers_mean':
        # Layers 8, 9, 10, 11 — mean across both layers and heads
        return attentions[8:12].mean(axis=(0, 1))  # [L, L]

    elif strategy == 'attention_rollout':
        # Attention rollout: iteratively multiply attention matrices
        # with residual connection mixing (0.5 * I + 0.5 * A)
        num_layers, num_heads, seq_len, _ = attentions.shape
        rollout = np.eye(seq_len, dtype=np.float32)

        for layer_idx in range(num_layers):
            # Average across heads for this layer
            attn_heads_mean = attentions[layer_idx].mean(axis=0)  # [L, L]
            # Add residual connection: 0.5 * identity + 0.5 * attention
            attn_with_residual = 0.5 * np.eye(seq_len, dtype=np.float32) + 0.5 * attn_heads_mean
            # Re-normalize rows to sum to 1
            row_sums = attn_with_residual.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            attn_with_residual = attn_with_residual / row_sums
            # Matrix multiply
            rollout = rollout @ attn_with_residual

        return rollout  # [L, L]

    else:
        raise ValueError(
            f'Unknown aggregation strategy: {strategy!r}. '
            f'Choose from: last_layer_mean, last_4_layers_mean, attention_rollout'
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLS Token Importance
# ══════════════════════════════════════════════════════════════════════════════

def cls_token_importance(
    attention_matrix: np.ndarray,
    tokens: List[str],
) -> Dict[str, Any]:
    """Extract how much the CLS token attends to each other token.

    Row 0 of the attention matrix represents the CLS token's attention
    distribution over all tokens. This function extracts that row,
    excludes special tokens, and returns sorted importances.

    Args:
        attention_matrix: Aggregated attention [L, L].
        tokens: List of token strings (length L).

    Returns:
        Dictionary with:
          - 'importances': list of (token, importance) tuples, sorted descending.
          - 'raw_vector': numpy array [L] of CLS attention to all tokens.
          - 'special_token_indices': list of indices that are special tokens.
    """
    cls_attn = attention_matrix[0]  # [L] — CLS attention to all tokens
    special_indices = []
    importances = []

    for i, token in enumerate(tokens):
        if _is_special_token(token):
            special_indices.append(i)
        else:
            importances.append((token, float(cls_attn[i])))

    # Sort by importance descending
    importances.sort(key=lambda x: x[1], reverse=True)

    return {
        'importances': importances,
        'raw_vector': cls_attn,
        'special_token_indices': special_indices,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Merge Subword Tokens to Whole Words
# ══════════════════════════════════════════════════════════════════════════════

def _is_continuation_token(token: str) -> bool:
    """Check if a token is a subword continuation.

    PhoBERT uses two possible conventions depending on the tokenizer version:
      - '@@' suffix: continuation marker (e.g., 'pho@@', 'bert')
      - No 'Ġ' (space) prefix: continuation (e.g., 'Ġpho', 'bert')
        where 'Ġ' indicates a word start.

    Returns True if this token continues the previous word.
    """
    if _is_special_token(token):
        return False
    # Check @@ convention: a token ending with @@ means the NEXT token continues
    # So the continuation token is the one that does NOT start a new word.
    # With Ġ convention: tokens starting with Ġ are word starters; others continue.
    if token.startswith('Ġ'):  # Ġ (U+0120) — word-start marker
        return False
    # If no Ġ prefix and not special, it could be a continuation in Ġ-mode.
    # But we also need to check if the tokenizer uses @@ mode.
    # We handle this in merge_subword_attention by detecting the mode.
    return True


def _detect_subword_mode(tokens: List[str]) -> str:
    """Detect which subword convention the tokenizer uses.

    Returns 'at_sign' for @@ convention, 'g_prefix' for Ġ convention,
    or 'none' if neither pattern is found.
    """
    has_at_sign = any('@@' in t for t in tokens if not _is_special_token(t))
    has_g_prefix = any(t.startswith('Ġ') for t in tokens if not _is_special_token(t))

    if has_at_sign:
        return 'at_sign'
    elif has_g_prefix:
        return 'g_prefix'
    return 'none'


def merge_subword_attention(
    token_importances: List[Tuple[str, float]],
    tokens: List[str],
    strategy: str = 'mean',
) -> List[Tuple[str, float]]:
    """Merge subword token importances back to whole Vietnamese words.

    PhoBERT uses BPE tokenization that splits Vietnamese words into subwords.
    This function reconstructs whole words and aggregates their importances.

    Handles two subword conventions:
      - @@ mode: 'pho@@' + 'bert' -> 'phobert' (PhoBERT v1 style)
      - Ġ mode: 'Ġpho' + 'bert' -> 'pho bert' (RoBERTa/PhoBERT v2 style)

    Args:
        token_importances: List of (token, importance) tuples from cls_token_importance.
        tokens: Full token list (used to detect subword mode and ordering).
        strategy: Aggregation for subword importances. One of 'mean', 'max', 'sum'.

    Returns:
        List of (word, importance) tuples, sorted descending by importance.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy not in ('mean', 'max', 'sum'):
        raise ValueError(f'Unknown merge strategy: {strategy!r}. Choose from: mean, max, sum')

    # Build a map from token text to importance for lookup
    # Note: token_importances already excludes special tokens
    if not token_importances:
        return []

    mode = _detect_subword_mode(tokens)

    # Build ordered list of (token, importance) following the original sequence
    # Filter out special tokens from the full token list
    non_special = [(t, i) for i, t in enumerate(tokens) if not _is_special_token(t)]

    # Build importance map by index for precise lookup
    # token_importances is sorted by importance, but we need original order for merging
    importance_by_token_index = {}
    imp_iter = iter(token_importances)
    imp_map = {t: v for t, v in token_importances}

    # Re-derive importances in sequence order
    ordered_pairs = []
    for token, orig_idx in non_special:
        imp = imp_map.get(token, 0.0)
        ordered_pairs.append((token, imp))

    # But there might be duplicate tokens with different importances.
    # Safer approach: use the raw CLS attention vector ordering.
    # Since token_importances is already sorted, we rebuild from the full tokens list.
    # Actually, token_importances has (token_str, importance) and may have duplicates.
    # We need to match by position, not by token text.
    # Let's rebuild from scratch using the original token order.

    # Rebuild: get cls_attn values in sequence order from the importance list
    # token_importances is sorted by value; we need positional ordering.
    # The cleanest approach: caller should pass raw_vector, but the API uses
    # the importances list. We'll work with the sequence of non-special tokens.

    # Group into words based on subword mode
    words: List[Tuple[str, List[float]]] = []

    if mode == 'at_sign':
        # @@ marks that current token is prefix of next: 'pho@@' + 'bert' -> 'phobert'
        current_word_parts: List[str] = []
        current_importances: List[float] = []

        for token, imp in ordered_pairs:
            if token.endswith('@@'):
                current_word_parts.append(token[:-2])  # strip @@
                current_importances.append(imp)
            else:
                current_word_parts.append(token)
                current_importances.append(imp)
                # Word complete
                words.append((''.join(current_word_parts), current_importances))
                current_word_parts = []
                current_importances = []

        # Handle trailing subwords (shouldn't happen but be safe)
        if current_word_parts:
            words.append((''.join(current_word_parts), current_importances))

    elif mode == 'g_prefix':
        # Ġ prefix marks word start: 'Ġpho' starts new word, 'bert' continues
        current_word_parts: List[str] = []
        current_importances: List[float] = []

        for token, imp in ordered_pairs:
            if token.startswith('Ġ'):
                # Save previous word if exists
                if current_word_parts:
                    words.append((''.join(current_word_parts), current_importances))
                current_word_parts = [token[1:]]  # strip Ġ
                current_importances = [imp]
            else:
                # Continuation of current word
                if not current_word_parts:
                    # First token without Ġ — treat as new word
                    current_word_parts = [token]
                    current_importances = [imp]
                else:
                    current_word_parts.append(token)
                    current_importances.append(imp)

        if current_word_parts:
            words.append((''.join(current_word_parts), current_importances))

    else:
        # No subword markers detected — each token is a word
        for token, imp in ordered_pairs:
            words.append((token, [imp]))

    # Aggregate importances per word
    merged: List[Tuple[str, float]] = []
    for word_text, imp_list in words:
        if not imp_list:
            continue
        if strategy == 'mean':
            agg = float(np.mean(imp_list))
        elif strategy == 'max':
            agg = float(np.max(imp_list))
        else:  # sum
            agg = float(np.sum(imp_list))
        merged.append((word_text, agg))

    # Sort descending by importance
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 5. Attention Heatmap Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    title: str = 'Attention Heatmap',
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
) -> Any:
    """Plot attention matrix as a heatmap.

    Handles long sequences gracefully:
      - <= 30 tokens: full annotations with token labels
      - 31-60 tokens: token labels on axes but no cell annotations
      - > 60 tokens: skip heatmap entirely and print a warning

    Uses DejaVu Sans font for Vietnamese diacritic support.

    Args:
        attention_matrix: Attention weights [L, L].
        tokens: List of token strings (length L).
        title: Plot title.
        save_path: If provided, save figure to this path.
        dpi: Figure resolution.
        figsize: Optional (width, height). Auto-computed if None.
        cmap: Colormap name. Defaults to 'magma' from COLOR_SCHEMES.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    seq_len = len(tokens)

    if seq_len > 60:
        print(f'[XAI] Skipping heatmap ({seq_len} tokens > 60 limit): {title}')
        return None

    if cmap is None:
        cmap = COLOR_SCHEMES['attention_cmap']

    # Set font for Vietnamese diacritics
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    if figsize is None:
        size = max(6, seq_len * 0.35)
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn if available, fall back to matplotlib imshow
    try:
        import seaborn as sns
        annot = seq_len <= 30
        fmt_str = '.2f' if annot else ''
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap=cmap,
            annot=annot,
            fmt=fmt_str,
            square=True,
            ax=ax,
            cbar_kws={'shrink': 0.8},
            linewidths=0.5 if seq_len <= 20 else 0,
        )
    except ImportError:
        im = ax.imshow(attention_matrix, cmap=cmap, aspect='equal')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Key tokens', fontsize=10)
    ax.set_ylabel('Query tokens', fontsize=10)
    ax.tick_params(axis='both', labelsize=7)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'[XAI] Saved heatmap: {save_path}')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLS Importance Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_cls_importance_bar(
    tokens: List[str],
    importances: List[float],
    title: str = 'CLS Token Importance',
    save_path: Optional[str] = None,
    top_k: int = 15,
    dpi: int = DEFAULT_DPI,
    figsize: Optional[Tuple[float, float]] = None,
    color: Optional[str] = None,
) -> Any:
    """Plot horizontal bar chart of CLS attention importance per token.

    Shows the top-k most attended tokens sorted in descending order,
    with value labels on each bar.

    Args:
        tokens: List of token strings.
        importances: List of importance values (same length as tokens).
        title: Plot title.
        save_path: If provided, save figure to this path.
        top_k: Number of top tokens to display.
        dpi: Figure resolution.
        figsize: Optional (width, height). Auto-computed if None.
        color: Bar color. Defaults to the text modality color.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    # Set font for Vietnamese diacritics
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    if color is None:
        color = COLOR_SCHEMES['modality_colors']['text']

    # Pair, sort descending, take top_k
    paired = list(zip(tokens, importances))
    paired.sort(key=lambda x: x[1], reverse=True)
    paired = paired[:top_k]

    if not paired:
        print(f'[XAI] No tokens to plot for: {title}')
        return None

    # Reverse for bottom-to-top display (highest at top)
    display_tokens = [p[0] for p in reversed(paired)]
    display_values = [p[1] for p in reversed(paired)]

    if figsize is None:
        figsize = (8, max(4, len(display_tokens) * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(display_tokens)), display_values, color=color, edgecolor='white')

    # Value labels on bars
    for bar, val in zip(bars, display_values):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}',
            va='center', ha='left', fontsize=8,
        )

    ax.set_yticks(range(len(display_tokens)))
    ax.set_yticklabels(display_tokens, fontsize=9)
    ax.set_xlabel('Attention Weight', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(display_values) * 1.2 if display_values else 1.0)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'[XAI] Saved bar chart: {save_path}')

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. Attention Sink Diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def compute_attention_sink_ratio(
    cls_importance: np.ndarray,
    tokens: List[str],
) -> Dict[str, Any]:
    """Compute what fraction of CLS attention goes to special tokens.

    Attention sinks (Xiao et al., 2023) describe the phenomenon where
    transformer models allocate disproportionate attention to the CLS and
    SEP tokens. A high sink ratio (> 0.3) may indicate the model is not
    distributing attention meaningfully across content tokens.

    Args:
        cls_importance: CLS attention vector [L] (raw_vector from cls_token_importance).
        tokens: List of token strings (length L).

    Returns:
        Dictionary with:
          - 'sink_ratio': fraction of CLS attention on special tokens.
          - 'special_attention': dict mapping special token -> its attention weight.
          - 'content_attention_sum': total attention on non-special tokens.
          - 'warning': str or None. Warning message if sink_ratio > 0.3.
    """
    special_attn = {}
    content_sum = 0.0

    for i, token in enumerate(tokens):
        weight = float(cls_importance[i])
        if _is_special_token(token):
            special_attn[f'{token}@{i}'] = weight
        else:
            content_sum += weight

    special_sum = sum(special_attn.values())
    total = special_sum + content_sum
    sink_ratio = special_sum / total if total > 1e-12 else 0.0

    warning = None
    if sink_ratio > 0.3:
        warning = (
            f'High attention sink ratio: {sink_ratio:.3f}. '
            f'{sink_ratio * 100:.1f}% of CLS attention goes to special tokens '
            f'(<s>, </s>). Content token attention may be unreliable.'
        )
        print(f'[XAI] WARNING: {warning}')

    return {
        'sink_ratio': float(sink_ratio),
        'special_attention': special_attn,
        'content_attention_sum': float(content_sum),
        'warning': warning,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. AttentionExplainer — High-Level Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class AttentionExplainer:
    """High-level attention explainer for the PhoBERT text branch.

    Orchestrates attention extraction, aggregation, visualization, and
    artifact saving for individual samples. Analogous to GradCAMExplainer
    from Phase 2.

    Args:
        model: The full multimodal model (e.g., CrossAttentionFusion), in eval mode.
               Must have eager attention already enabled.
        tokenizer: HuggingFace tokenizer for PhoBERT.
        device: Torch device.
        output_dir: Base directory for saving artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        output_dir: str = './xai_output/attention',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        print(f'[XAI] AttentionExplainer initialized. Output dir: {output_dir}')

    def explain_sample(
        self,
        sample: Dict[str, Any],
        sample_id: str,
    ) -> Dict[str, Any]:
        """Generate full attention-based explanation for a single sample.

        Pipeline:
          1. Extract attention from all 12 PhoBERT layers
          2. Aggregate with last_layer_mean and last_4_layers_mean strategies
          3. Compute CLS token importance (subword level)
          4. Merge subwords to whole Vietnamese words
          5. Generate heatmaps and bar charts
          6. Compute attention sink diagnostic
          7. Save all artifacts

        Output folder structure:
          {output_dir}/{sample_id}/
            attention_layer11_mean_heatmap.png
            attention_last4layers_mean_heatmap.png
            cls_importance_subword_bar.png
            cls_importance_word_bar.png
            raw_attention.npz
            tokens.json
            word_importance.json
            topk_tokens.json
            metadata.json

        Args:
            sample: Sample dict from xai.utils.load_single_sample().
            sample_id: Unique identifier (e.g., 'sample_042').

        Returns:
            Dictionary with all results and artifact paths.
        """
        sample_dir = os.path.join(self.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        print(f'[XAI] Explaining {sample_id} (attention analysis)')

        # ── Step 1: Extract attention ─────────────────────────────────────
        attn_result = extract_phobert_attention(
            model=self.model,
            input_ids=sample['input_ids'],
            attention_mask=sample['attention_mask'],
            tokenizer=self.tokenizer,
        )
        attentions = attn_result['attentions']  # [12, 12, L, L]
        tokens = attn_result['tokens']
        seq_len = attn_result['seq_len']

        # ── Step 2: Aggregate attention ───────────────────────────────────
        strategies = {
            'last_layer_mean': 'attention_layer11_mean_heatmap.png',
            'last_4_layers_mean': 'attention_last4layers_mean_heatmap.png',
        }

        aggregated_matrices = {}
        heatmap_paths = {}

        for strat_name, heatmap_filename in strategies.items():
            agg_matrix = aggregate_attention(attentions, strategy=strat_name)
            aggregated_matrices[strat_name] = agg_matrix

            # Generate heatmap
            heatmap_path = os.path.join(sample_dir, heatmap_filename)
            plot_attention_heatmap(
                attention_matrix=agg_matrix,
                tokens=tokens,
                title=f'{sample_id}: {strat_name.replace("_", " ").title()}',
                save_path=heatmap_path,
                dpi=DEFAULT_DPI,
            )
            heatmap_paths[strat_name] = heatmap_path

        # ── Step 3: CLS importance (subword level) ────────────────────────
        # Use last_layer_mean as the primary strategy for CLS analysis
        primary_matrix = aggregated_matrices['last_layer_mean']
        cls_result = cls_token_importance(primary_matrix, tokens)

        # Subword-level bar chart
        sub_tokens = [t for t, _ in cls_result['importances']]
        sub_importances = [v for _, v in cls_result['importances']]

        subword_bar_path = os.path.join(sample_dir, 'cls_importance_subword_bar.png')
        plot_cls_importance_bar(
            tokens=sub_tokens,
            importances=sub_importances,
            title=f'{sample_id}: CLS Importance (Subword)',
            save_path=subword_bar_path,
            top_k=15,
            dpi=DEFAULT_DPI,
        )

        # ── Step 4: Merge subwords to words ───────────────────────────────
        word_importances = merge_subword_attention(
            token_importances=cls_result['importances'],
            tokens=tokens,
            strategy='mean',
        )

        # Word-level bar chart
        word_tokens = [w for w, _ in word_importances]
        word_values = [v for _, v in word_importances]

        word_bar_path = os.path.join(sample_dir, 'cls_importance_word_bar.png')
        plot_cls_importance_bar(
            tokens=word_tokens,
            importances=word_values,
            title=f'{sample_id}: CLS Importance (Word-Level)',
            save_path=word_bar_path,
            top_k=15,
            dpi=DEFAULT_DPI,
        )

        # ── Step 5: Attention sink diagnostic ─────────────────────────────
        sink_result = compute_attention_sink_ratio(
            cls_importance=cls_result['raw_vector'],
            tokens=tokens,
        )

        # ── Step 6: Save raw data ─────────────────────────────────────────
        # Raw attention in float16 to save space
        raw_attn_path = os.path.join(sample_dir, 'raw_attention.npz')
        np.savez(
            raw_attn_path,
            attentions=attentions.astype(np.float16),
        )
        print(f'[XAI] Saved raw attention: {raw_attn_path}')

        # Tokens JSON
        tokens_path = os.path.join(sample_dir, 'tokens.json')
        with open(tokens_path, 'w', encoding='utf-8') as f:
            json.dump({
                'tokens': tokens,
                'seq_len': seq_len,
                'text': sample.get('text', ''),
            }, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved tokens: {tokens_path}')

        # Word importance JSON
        word_imp_path = os.path.join(sample_dir, 'word_importance.json')
        with open(word_imp_path, 'w', encoding='utf-8') as f:
            json.dump({
                'word_importances': [{'word': w, 'importance': v} for w, v in word_importances],
                'merge_strategy': 'mean',
                'attention_strategy': 'last_layer_mean',
            }, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved word importance: {word_imp_path}')

        # Top-k tokens JSON (subword level)
        topk_path = os.path.join(sample_dir, 'topk_tokens.json')
        top_k_data = [
            {'token': t, 'importance': float(v), 'rank': i + 1}
            for i, (t, v) in enumerate(cls_result['importances'][:20])
        ]
        with open(topk_path, 'w', encoding='utf-8') as f:
            json.dump(top_k_data, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved top-k tokens: {topk_path}')

        # ── Step 7: Metadata ──────────────────────────────────────────────
        metadata = {
            'sample_id': sample_id,
            'sample_idx': sample.get('sample_idx', None),
            'text': sample.get('text', ''),
            'seq_len': seq_len,
            'num_layers': PHOBERT_NUM_LAYERS,
            'num_heads': PHOBERT_NUM_HEADS,
            'aggregation_strategies': list(strategies.keys()),
            'sink_ratio': sink_result['sink_ratio'],
            'sink_warning': sink_result['warning'],
            'num_words_after_merge': len(word_importances),
            'device': str(self.device),
            'timestamp': datetime.datetime.now().isoformat(),
            'artifacts': {
                'heatmaps': heatmap_paths,
                'subword_bar': subword_bar_path,
                'word_bar': word_bar_path,
                'raw_attention': raw_attn_path,
                'tokens': tokens_path,
                'word_importance': word_imp_path,
                'topk_tokens': topk_path,
            },
            'cross_attention_note': (
                'Cross-attention now uses token×patch attention (Q=[B,T,512], K=[B,P,512] '
                '-> [B,8,T,P] weights). Cross-attention visualization is planned for Phase 3 update. '
                'This module currently analyzes PhoBERT self-attention only.'
            ),
        }

        meta_path = os.path.join(sample_dir, 'metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved metadata: {meta_path}')

        print(f'[XAI] Attention explanation complete for {sample_id}')

        return {
            'sample_id': sample_id,
            'attentions': attentions,
            'tokens': tokens,
            'seq_len': seq_len,
            'aggregated': aggregated_matrices,
            'cls_importance': cls_result,
            'word_importances': word_importances,
            'sink_diagnostic': sink_result,
            'paths': {
                'sample_dir': sample_dir,
                'heatmaps': heatmap_paths,
                'subword_bar': subword_bar_path,
                'word_bar': word_bar_path,
                'raw_attention': raw_attn_path,
                'tokens': tokens_path,
                'word_importance': word_imp_path,
                'topk_tokens': topk_path,
                'metadata': meta_path,
            },
            'metadata': metadata,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 9. Tokenization Inspector (Debug Utility)
# ══════════════════════════════════════════════════════════════════════════════

def inspect_tokenization(
    tokenizer,
    text: str,
    max_display: int = 50,
) -> List[Dict[str, Any]]:
    """Inspect how PhoBERT tokenizes a Vietnamese text.

    Useful for debugging subword merging and understanding token boundaries.
    Prints a formatted table and returns structured data.

    Args:
        tokenizer: HuggingFace tokenizer (PhoBERT).
        text: Vietnamese text to tokenize.
        max_display: Maximum number of tokens to display.

    Returns:
        List of dicts, each with: token, token_id, is_special, is_subword, position.
    """
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    token_ids = encoding['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    mode = _detect_subword_mode(tokens)

    results = []
    display_count = min(len(tokens), max_display)

    print(f'[XAI] Tokenization inspection (mode={mode}):')
    print(f'[XAI]   Text: {text[:100]}{"..." if len(text) > 100 else ""}')
    print(f'[XAI]   Total tokens: {len(tokens)}')
    print(f'[XAI]   {"Pos":<5} {"Token":<25} {"ID":<8} {"Special":<9} {"Subword":<9}')
    print(f'[XAI]   {"---":<5} {"-----":<25} {"--":<8} {"-------":<9} {"-------":<9}')

    for i in range(display_count):
        token = tokens[i]
        token_id = token_ids[i]
        is_special = _is_special_token(token)

        # Determine if subword based on detected mode
        if is_special:
            is_subword = False
        elif mode == 'at_sign':
            # In @@ mode, a token without @@ that follows a @@ token is a continuation
            is_subword = (i > 0 and not _is_special_token(tokens[i - 1])
                          and tokens[i - 1].endswith('@@'))
        elif mode == 'g_prefix':
            is_subword = not token.startswith('Ġ') and i > 0
        else:
            is_subword = False

        entry = {
            'position': i,
            'token': token,
            'token_id': token_id,
            'is_special': is_special,
            'is_subword': is_subword,
        }
        results.append(entry)

        print(f'[XAI]   {i:<5} {token:<25} {token_id:<8} {str(is_special):<9} {str(is_subword):<9}')

    if len(tokens) > max_display:
        print(f'[XAI]   ... ({len(tokens) - max_display} more tokens)')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 10. Cross-Attention Extraction & Visualization
# ══════════════════════════════════════════════════════════════════════════════

def extract_cross_attention(
    model: nn.Module,
    sample: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:
    """Extract cross-attention weights from CrossAttentionFusion.

    Calls the model's sub-modules directly to capture the attention
    weights that ``model.forward()`` normally discards.

    Args:
        model: The full multimodal model (CrossAttentionFusion) in eval mode.
        sample: Sample dict from ``load_single_sample()``.
        tokenizer: HuggingFace tokenizer for decoding token IDs.

    Returns:
        Dictionary with:
          - 't2i_attn': numpy array — text-to-image attention [T_real, P]
          - 'i2t_attn': numpy array — image-to-text attention [P, T_real]
          - 'tokens': list of decoded token strings (trimmed, no padding)
          - 'seq_len': int — actual text token count
          - 'num_patches': int — number of image patches (e.g. 49)
    """
    model.eval()
    with torch.no_grad():
        _, _, text_tokens, text_pad = model.text_model(
            sample['input_ids'], sample['attention_mask'], return_tokens=True,
        )
        image_patches, patch_mask = model.image_model.forward_features(
            sample['pixel_values'], num_images=sample.get('num_images'),
        )

        t = model.text_proj(text_tokens.float())
        i = model.image_proj(image_patches.float())

        t_kpm = text_pad
        i_kpm = ~patch_mask

        _, t2i_attn = model.cross_attn_t2i(
            query=t, key=i, value=i, key_padding_mask=i_kpm,
        )
        _, i2t_attn = model.cross_attn_i2t(
            query=i, key=t, value=t, key_padding_mask=t_kpm,
        )

    # Trim to real token length (remove padding)
    mask = sample['attention_mask'][0]
    seq_len = int(mask.sum().item())
    token_ids = sample['input_ids'][0, :seq_len].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # t2i_attn: [B, T, P] (head-averaged) or [B, num_heads, T, P]
    t2i = t2i_attn[0].cpu().numpy()  # drop batch dim
    i2t = i2t_attn[0].cpu().numpy()

    # If head-averaged (2-D after batch drop), keep as-is; trim T to real tokens
    if t2i.ndim == 2:
        t2i = t2i[:seq_len, :]      # [T_real, P]
        i2t = i2t[:, :seq_len]      # [P, T_real]
    elif t2i.ndim == 3:
        t2i = t2i[:, :seq_len, :]   # [H, T_real, P]
        i2t = i2t[:, :, :seq_len]   # [H, P, T_real]
        # Average over heads for primary visualizations
        t2i = t2i.mean(axis=0)      # [T_real, P]
        i2t = i2t.mean(axis=0)      # [P, T_real]

    num_patches = t2i.shape[1]

    print(f'[XAI] Extracted cross-attention: T={seq_len}, P={num_patches}')

    return {
        't2i_attn': t2i.astype(np.float32),
        'i2t_attn': i2t.astype(np.float32),
        'tokens': tokens,
        'seq_len': seq_len,
        'num_patches': num_patches,
    }


def plot_cross_attention_heatmap(
    attn_matrix: np.ndarray,
    tokens: List[str],
    title: str = 'Cross-Attention: Text → Image',
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    max_tokens: int = 60,
) -> Any:
    """Plot cross-attention matrix as a heatmap (tokens × patches).

    Args:
        attn_matrix: Attention weights [T, P].
        tokens: List of token strings (length T).
        title: Plot title.
        save_path: If provided, save figure to this path.
        dpi: Figure resolution.
        max_tokens: Skip heatmap if token count exceeds this.

    Returns:
        matplotlib Figure object, or None if skipped.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    T, P = attn_matrix.shape
    if T > max_tokens:
        print(f'[XAI] Skipping cross-attention heatmap ({T} tokens > {max_tokens} limit)')
        return None

    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    H = W = int(np.sqrt(P))
    patch_labels = [f'{r},{c}' for r in range(H) for c in range(W)] if H * W == P else [str(j) for j in range(P)]

    fig_w = max(8, P * 0.25)
    fig_h = max(6, T * 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    try:
        import seaborn as sns
        sns.heatmap(
            attn_matrix, xticklabels=patch_labels, yticklabels=tokens,
            cmap='viridis', ax=ax, cbar_kws={'shrink': 0.6},
        )
    except ImportError:
        im = ax.imshow(attn_matrix, aspect='auto', cmap='viridis')
        ax.set_xticks(range(P))
        ax.set_xticklabels(patch_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(T))
        ax.set_yticklabels(tokens, fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.6)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(f'Image Patches ({H}×{W})', fontsize=10)
    ax.set_ylabel('Text Tokens', fontsize=10)
    ax.tick_params(axis='x', labelsize=5, rotation=90)
    ax.tick_params(axis='y', labelsize=6)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'[XAI] Saved cross-attention heatmap: {save_path}')

    return fig


def plot_patch_importance(
    i2t_attn: np.ndarray,
    original_image,
    title: str = 'Patch Importance (from Cross-Attention)',
    save_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
) -> Any:
    """Overlay patch-level importance on the original image.

    Importance = mean attention each patch receives from all text tokens.

    Args:
        i2t_attn: Image-to-text attention [P, T] — each patch's attention to tokens.
        original_image: PIL Image (original review image).
        title: Plot title.
        save_path: If provided, save figure.
        dpi: Figure resolution.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from PIL import Image

    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    P = i2t_attn.shape[0]
    H = W = int(np.sqrt(P))
    if H * W != P:
        print(f'[XAI] WARNING: P={P} is not a perfect square, skipping patch overlay')
        return None

    # Importance per patch = sum of attention that patch distributes to text tokens
    patch_importance = i2t_attn.sum(axis=1)  # [P]
    patch_map = patch_importance.reshape(H, W)

    # Normalize to [0, 1]
    pmin, pmax = patch_map.min(), patch_map.max()
    if pmax - pmin > 1e-8:
        patch_map = (patch_map - pmin) / (pmax - pmin)
    else:
        patch_map = np.zeros_like(patch_map)

    img_resized = original_image.convert('RGB').resize((224, 224), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Upsample patch_map to image size
    from PIL import Image as PILImage
    patch_pil = PILImage.fromarray((patch_map * 255).astype(np.uint8), mode='L')
    patch_upsampled = np.array(
        patch_pil.resize((224, 224), PILImage.BILINEAR), dtype=np.float32,
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(patch_map, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Patch Importance ({H}×{W})', fontsize=11)
    axes[1].axis('off')

    overlay = 0.5 * img_np + 0.5 * plt.cm.hot(patch_upsampled)[:, :, :3]
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay on Image', fontsize=11)
    axes[2].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'[XAI] Saved patch importance: {save_path}')

    return fig


class CrossAttentionExplainer:
    """Generates cross-attention visualizations between text tokens and image patches.

    Extracts the attention weights from ``cross_attn_t2i`` and ``cross_attn_i2t``
    in CrossAttentionFusion and produces:
      - Token × Patch heatmap
      - Patch importance overlay on original image
      - Top-K token-patch pairs
      - Raw attention matrices (.npy)
      - Summary statistics (JSON)

    Args:
        model: The full multimodal model (CrossAttentionFusion), in eval mode.
        tokenizer: HuggingFace tokenizer for PhoBERT.
        device: Torch device.
        output_dir: Base directory for saving artifacts.
    """

    def __init__(self, model: nn.Module, tokenizer, device: torch.device,
                 output_dir: str = './xai_output/cross_attention'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f'[XAI] CrossAttentionExplainer initialized. Output dir: {output_dir}')

    def explain_sample(self, sample: Dict[str, Any], sample_id: str,
                       top_k: int = 20) -> Dict[str, Any]:
        """Generate cross-attention explanation for a single sample.

        Args:
            sample: Sample dict from ``load_single_sample()``.
            sample_id: Unique identifier (e.g., 'sample_0042').
            top_k: Number of top token-patch pairs to extract.

        Returns:
            Dictionary with all results and artifact paths.
        """
        sample_dir = os.path.join(self.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        print(f'[XAI] Cross-attention explanation for {sample_id}')

        # ── Extract cross-attention ──────────────────────────────────────
        ca = extract_cross_attention(self.model, sample, self.tokenizer)
        t2i = ca['t2i_attn']  # [T_real, P]
        i2t = ca['i2t_attn']  # [P, T_real]
        tokens = ca['tokens']
        seq_len = ca['seq_len']
        num_patches = ca['num_patches']

        # ── Save raw attention matrices ──────────────────────────────────
        raw_path = os.path.join(sample_dir, 'cross_attention_raw.npz')
        np.savez(raw_path, t2i_attn=t2i, i2t_attn=i2t)
        print(f'[XAI] Saved: {raw_path}')

        # ── Token × Patch heatmap ────────────────────────────────────────
        heatmap_path = os.path.join(sample_dir, 'token_patch_heatmap.png')
        plot_cross_attention_heatmap(
            t2i, tokens,
            title=f'{sample_id}: Text→Image Cross-Attention',
            save_path=heatmap_path, dpi=DEFAULT_DPI,
        )

        # ── Patch importance overlay ─────────────────────────────────────
        overlay_path = os.path.join(sample_dir, 'patch_importance.png')
        original_images = sample.get('loaded_images', [])
        if original_images:
            plot_patch_importance(
                i2t, original_images[0],
                title=f'{sample_id}: Patch Importance',
                save_path=overlay_path, dpi=DEFAULT_DPI,
            )
        else:
            overlay_path = None

        # ── Top-K token-patch pairs ──────────────────────────────────────
        # Flatten t2i to find highest attention pairs
        flat = t2i.flatten()
        topk_indices = np.argsort(flat)[::-1][:top_k]
        topk_pairs = []
        for idx in topk_indices:
            t_idx = int(idx // num_patches)
            p_idx = int(idx % num_patches)
            H_p = W_p = int(np.sqrt(num_patches))
            topk_pairs.append({
                'rank': len(topk_pairs) + 1,
                'token_idx': t_idx,
                'token': tokens[t_idx] if t_idx < len(tokens) else '<PAD>',
                'patch_idx': p_idx,
                'patch_row': p_idx // W_p if H_p * W_p == num_patches else -1,
                'patch_col': p_idx % W_p if H_p * W_p == num_patches else -1,
                'attention': float(flat[idx]),
            })

        topk_path = os.path.join(sample_dir, 'token_patch_topk.json')
        with open(topk_path, 'w', encoding='utf-8') as f:
            json.dump(topk_pairs, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved: {topk_path}')

        # ── Summary statistics ───────────────────────────────────────────
        # Per-token importance = how much each token attends to image overall
        token_importance = t2i.sum(axis=1)  # [T_real]
        # Per-patch importance = how much each patch receives from text
        patch_importance = t2i.sum(axis=0)  # [P]

        # Entropy per token (how spread is attention across patches)
        eps = 1e-10
        t2i_safe = np.clip(t2i, eps, None)
        token_entropy = -(t2i_safe * np.log2(t2i_safe)).sum(axis=1)  # [T_real]

        summary = {
            'sample_id': sample_id,
            'seq_len': seq_len,
            'num_patches': num_patches,
            'mean_attention': float(t2i.mean()),
            'max_attention': float(t2i.max()),
            'min_attention': float(t2i.min()),
            'mean_token_entropy': float(token_entropy.mean()),
            'top_5_tokens': [
                {'token': tokens[i], 'importance': float(token_importance[i])}
                for i in np.argsort(token_importance)[::-1][:5]
                if i < len(tokens)
            ],
            'top_5_patches': [
                {'patch_idx': int(i), 'importance': float(patch_importance[i])}
                for i in np.argsort(patch_importance)[::-1][:5]
            ],
            'timestamp': datetime.datetime.now().isoformat(),
        }

        summary_path = os.path.join(sample_dir, 'cross_attention_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'[XAI] Saved: {summary_path}')

        paths = {
            'sample_dir': sample_dir,
            'raw_attention': raw_path,
            'heatmap': heatmap_path,
            'patch_importance': overlay_path,
            'topk_pairs': topk_path,
            'summary': summary_path,
        }

        # ── Upgraded thesis-quality visualizations ───────────────────────
        if original_images:
            try:
                tok2p = plot_token_to_patches(
                    t2i, tokens, original_images[0], sample_id, sample_dir,
                    top_k_tokens=5, top_k_patches=5, dpi=DEFAULT_DPI,
                )
                paths['token_to_patch_figs'] = tok2p
            except Exception as e:
                print(f'[XAI] WARNING: token→patch viz failed: {e}')

            try:
                p2t = plot_patch_to_tokens(
                    i2t, tokens, original_images[0], sample_id, sample_dir,
                    top_k_patches=5, top_k_tokens=5, dpi=DEFAULT_DPI,
                )
                paths['patch_to_token_figs'] = p2t
            except Exception as e:
                print(f'[XAI] WARNING: patch→token viz failed: {e}')

        try:
            topk_hm = plot_topk_heatmap(
                t2i, tokens, sample_id,
                save_path=os.path.join(sample_dir, 'topk_token_patch_heatmap.png'),
                top_k_tokens=10, top_k_patches=10, dpi=DEFAULT_DPI,
            )
            paths['topk_heatmap'] = topk_hm
        except Exception as e:
            print(f'[XAI] WARNING: top-K heatmap failed: {e}')

        try:
            bip = plot_bipartite_graph(
                t2i, tokens, sample_id,
                save_path=os.path.join(sample_dir, 'token_patch_bipartite_graph.png'),
                top_k_edges=15, dpi=DEFAULT_DPI,
            )
            paths['bipartite_graph'] = bip
        except Exception as e:
            print(f'[XAI] WARNING: bipartite graph failed: {e}')

        num_files = len([f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))])
        summary['num_generated_files'] = num_files
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f'[XAI] Cross-attention explanation complete for {sample_id} ({num_files} files)')

        return {
            'sample_id': sample_id,
            't2i_attn': t2i,
            'i2t_attn': i2t,
            'tokens': tokens,
            'topk_pairs': topk_pairs,
            'summary': summary,
            'paths': paths,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 11. Thesis-Quality Cross-Attention Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def _patch_grid_coords(num_patches: int):
    """Return (H, W) for a square patch grid. Raises if not square."""
    H = W = int(np.sqrt(num_patches))
    if H * W != num_patches:
        raise ValueError(f'P={num_patches} is not a perfect square')
    return H, W


def _highlight_patches_on_image(
    img_np: np.ndarray, H: int, W: int, patch_indices: List[int],
    scores: Optional[List[float]] = None, img_size: int = 224,
) -> np.ndarray:
    """Draw colored rectangles on image for given patch indices."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_np)
    ph, pw = img_size // H, img_size // W
    cmap = plt.cm.plasma
    max_s = max(scores) if scores and max(scores) > 0 else 1.0
    for rank, pidx in enumerate(patch_indices):
        r, c = pidx // W, pidx % W
        s = scores[rank] / max_s if scores else 1.0
        color = cmap(s)
        rect = mpatches.Rectangle((c * pw, r * ph), pw, ph,
                                   linewidth=2, edgecolor=color,
                                   facecolor=(*color[:3], 0.35))
        ax.add_patch(rect)
        ax.text(c * pw + pw // 2, r * ph + ph // 2, f'{pidx}',
                ha='center', va='center', fontsize=7, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.6))
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return buf


def plot_token_to_patches(
    t2i_attn: np.ndarray, tokens: List[str], original_image,
    sample_id: str, save_dir: str, top_k_tokens: int = 5,
    top_k_patches: int = 5, dpi: int = DEFAULT_DPI,
) -> List[str]:
    """For each top-attended token, show which image patches it attends to.

    Generates one figure per token + a grid summary figure.

    Returns list of saved file paths.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)
    T, P = t2i_attn.shape
    H, W = _patch_grid_coords(P)
    img_size = 224
    ph, pw = img_size // H, img_size // W

    img_resized = original_image.convert('RGB').resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    token_total = t2i_attn.sum(axis=1)
    top_token_idxs = np.argsort(token_total)[::-1][:top_k_tokens]
    saved = []

    per_token_images = []
    per_token_labels = []

    for rank, tidx in enumerate(top_token_idxs):
        if tidx >= len(tokens):
            continue
        tok = tokens[tidx]
        row = t2i_attn[tidx]
        top_pidxs = np.argsort(row)[::-1][:top_k_patches]
        top_scores = [float(row[p]) for p in top_pidxs]

        fig, axes = plt.subplots(1, top_k_patches + 1, figsize=(3 * (top_k_patches + 1), 3.5))
        axes[0].imshow(img_np)
        axes[0].set_title(f'Token: "{tok}"', fontsize=10, fontweight='bold')
        axes[0].axis('off')

        for i, (pidx, sc) in enumerate(zip(top_pidxs, top_scores)):
            r, c = int(pidx) // W, int(pidx) % W
            y0, x0 = r * ph, c * pw
            crop = img_np[y0:y0+ph, x0:x0+pw]
            axes[i+1].imshow(crop)
            axes[i+1].set_title(f'P{pidx} ({r},{c})\n{sc:.4f}', fontsize=8)
            axes[i+1].axis('off')

        fig.suptitle(f'{sample_id} — Token "{tok}" → Top Patches', fontsize=11, fontweight='bold')
        fig.tight_layout()
        safe_tok = tok.replace('/', '_').replace('\\', '_')[:20]
        fname = f'token_overlay_{rank+1}_{safe_tok}.png'
        fpath = os.path.join(save_dir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches='tight', facecolor='white')

        per_token_images.append(
            _highlight_patches_on_image(img_np, H, W, [int(p) for p in top_pidxs], top_scores, img_size)
        )
        per_token_labels.append(tok)
        plt.close(fig)
        saved.append(fpath)

    if per_token_images:
        n = len(per_token_images)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, img_buf, label in zip(axes, per_token_images, per_token_labels):
            ax.imshow(img_buf)
            ax.set_title(f'"{label}"', fontsize=10, fontweight='bold')
            ax.axis('off')
        fig.suptitle(f'{sample_id} — Top Tokens → Patch Overlay', fontsize=12, fontweight='bold')
        fig.tight_layout()
        grid_path = os.path.join(save_dir, 'top_tokens_patch_overlay_grid.png')
        fig.savefig(grid_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved.append(grid_path)

    return saved


def plot_patch_to_tokens(
    i2t_attn: np.ndarray, tokens: List[str], original_image,
    sample_id: str, save_dir: str, top_k_patches: int = 5,
    top_k_tokens: int = 5, dpi: int = DEFAULT_DPI,
) -> List[str]:
    """For each top-attended patch, show which text tokens it attends to.

    Returns list of saved file paths.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)
    P, T = i2t_attn.shape
    H, W = _patch_grid_coords(P)
    img_size = 224
    ph, pw = img_size // H, img_size // W

    img_resized = original_image.convert('RGB').resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    patch_total = i2t_attn.sum(axis=1)
    top_pidxs = np.argsort(patch_total)[::-1][:top_k_patches]
    saved = []

    for rank, pidx in enumerate(top_pidxs):
        pidx = int(pidx)
        r, c = pidx // W, pidx % W
        y0, x0 = r * ph, c * pw
        crop = img_np[y0:y0+ph, x0:x0+pw]
        row = i2t_attn[pidx]
        top_tidxs = np.argsort(row)[::-1][:top_k_tokens]
        top_tok_scores = [(tokens[ti] if ti < len(tokens) else '?', float(row[ti]))
                          for ti in top_tidxs]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                                  gridspec_kw={'width_ratios': [1, 1, 1.5]})
        axes[0].imshow(img_np)
        rect = plt.Rectangle((x0, y0), pw, ph, linewidth=3,
                               edgecolor='lime', facecolor=(0, 1, 0, 0.25))
        axes[0].add_patch(rect)
        axes[0].set_title(f'Patch {pidx} ({r},{c})', fontsize=10, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(crop)
        axes[1].set_title('Zoomed Patch', fontsize=10)
        axes[1].axis('off')

        y_pos = range(len(top_tok_scores))
        bars_labels = [t for t, _ in reversed(top_tok_scores)]
        bars_vals = [s for _, s in reversed(top_tok_scores)]
        colors = [COLOR_SCHEMES['modality_colors']['text']] * len(bars_vals)
        axes[2].barh(y_pos, bars_vals, color=colors, edgecolor='white')
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(bars_labels, fontsize=9)
        axes[2].set_xlabel('Attention', fontsize=9)
        axes[2].set_title('Top Tokens', fontsize=10, fontweight='bold')
        for bar, val in zip(axes[2].patches, bars_vals):
            axes[2].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                         f'{val:.4f}', va='center', fontsize=7)

        fig.suptitle(f'{sample_id} — Patch {pidx} → Token Explanation', fontsize=11, fontweight='bold')
        fig.tight_layout()
        fpath = os.path.join(save_dir, f'patch_{pidx}_token_explanation.png')
        fig.savefig(fpath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved.append(fpath)

    return saved


def plot_topk_heatmap(
    t2i_attn: np.ndarray, tokens: List[str],
    sample_id: str, save_path: str,
    top_k_tokens: int = 10, top_k_patches: int = 10,
    dpi: int = DEFAULT_DPI,
) -> str:
    """Plot a readable heatmap of only the top-K tokens × top-K patches."""
    import matplotlib.pyplot as plt

    T, P = t2i_attn.shape
    H, W = _patch_grid_coords(P)

    tok_imp = t2i_attn.sum(axis=1)
    top_tidxs = np.argsort(tok_imp)[::-1][:min(top_k_tokens, T)]

    patch_imp = t2i_attn.sum(axis=0)
    top_pidxs = np.argsort(patch_imp)[::-1][:min(top_k_patches, P)]

    sub = t2i_attn[np.ix_(top_tidxs, top_pidxs)]
    row_labels = [tokens[i] if i < len(tokens) else '?' for i in top_tidxs]
    col_labels = [f'P{p}({p//W},{p%W})' for p in top_pidxs]

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels)*0.8), max(4, len(row_labels)*0.5)))
    try:
        import seaborn as sns
        sns.heatmap(sub, xticklabels=col_labels, yticklabels=row_labels,
                    cmap='viridis', annot=True, fmt='.3f', ax=ax,
                    cbar_kws={'shrink': 0.7})
    except ImportError:
        im = ax.imshow(sub, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)

    ax.set_title(f'{sample_id} — Top-{len(row_labels)} Tokens × Top-{len(col_labels)} Patches',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Image Patches', fontsize=9)
    ax.set_ylabel('Text Tokens', fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[XAI] Saved top-K heatmap: {save_path}')
    return save_path


def plot_bipartite_graph(
    t2i_attn: np.ndarray, tokens: List[str],
    sample_id: str, save_path: str,
    top_k_edges: int = 15, dpi: int = DEFAULT_DPI,
) -> str:
    """Draw a bipartite graph: tokens (left) ↔ patches (right), edges = attention."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    T, P = t2i_attn.shape
    H, W = _patch_grid_coords(P)

    flat = t2i_attn.flatten()
    topk_flat = np.argsort(flat)[::-1][:top_k_edges]

    left_set = set()
    right_set = set()
    edges = []
    for idx in topk_flat:
        ti = int(idx // P)
        pi = int(idx % P)
        left_set.add(ti)
        right_set.add(pi)
        edges.append((ti, pi, float(flat[idx])))

    left_list = sorted(left_set)
    right_list = sorted(right_set)
    left_y = {v: i for i, v in enumerate(left_list)}
    right_y = {v: i for i, v in enumerate(right_list)}

    fig_h = max(4, max(len(left_list), len(right_list)) * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    max_score = max(e[2] for e in edges) if edges else 1.0
    for ti, pi, sc in edges:
        lw = 1 + 4 * (sc / max_score)
        alpha = 0.3 + 0.7 * (sc / max_score)
        ax.plot([0.15, 0.85], [left_y[ti], right_y[pi]],
                color='steelblue', linewidth=lw, alpha=alpha)

    for ti in left_list:
        tok = tokens[ti] if ti < len(tokens) else '?'
        ax.plot(0.15, left_y[ti], 'o', color=COLOR_SCHEMES['modality_colors']['text'],
                markersize=10, zorder=5)
        ax.text(0.12, left_y[ti], tok, ha='right', va='center', fontsize=8, fontweight='bold')

    for pi in right_list:
        r, c = pi // W, pi % W
        ax.plot(0.85, right_y[pi], 's', color=COLOR_SCHEMES['modality_colors']['image'],
                markersize=10, zorder=5)
        ax.text(0.88, right_y[pi], f'P{pi}({r},{c})', ha='left', va='center', fontsize=8)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, max(len(left_list), len(right_list)) - 0.5)
    ax.invert_yaxis()
    ax.axis('off')

    text_patch = mpatches.Patch(color=COLOR_SCHEMES['modality_colors']['text'], label='Text Tokens')
    img_patch = mpatches.Patch(color=COLOR_SCHEMES['modality_colors']['image'], label='Image Patches')
    ax.legend(handles=[text_patch, img_patch], loc='lower center', ncol=2, fontsize=9)

    ax.set_title(f'{sample_id} — Token↔Patch Bipartite (Top-{top_k_edges})',
                 fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[XAI] Saved bipartite graph: {save_path}')
    return save_path
