"""
Load XAI artifacts from disk for the AI Agent.

Reads JSON metadata from each XAI phase (Grad-CAM, Attention,
Cross-Attention, SHAP, LIME) and returns a structured evidence
dictionary.  Missing files are handled gracefully — the loader
records which artifacts are available and which are absent.

Also resolves paths to visual artifacts (PNG figures) for report
integration.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional


_FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']

# Tokens that are tokenizer noise — filter these from attention display
_NOISE_PATTERN = re.compile(
    r'^[^\w]|@@$|^\d+@@|^[a-zA-Z]{1,2}$|^<[^>]+>$', re.UNICODE)


def _load_json(path: str) -> Optional[Any]:
    """Load a JSON file. Returns None on any failure."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _is_noisy_token(token: str) -> bool:
    """Check if a token is likely tokenizer noise (subword fragments,
    single characters, special tokens)."""
    if not token:
        return True
    # Special tokens
    if token in {'<s>', '</s>', '<pad>', '<unk>', '<mask>'}:
        return True
    # Strip BPE markers for length check
    clean = token.replace('@@', '').replace('Ġ', '')
    if len(clean) < 2:
        return True
    # Pure numbers or number+@@ fragments
    if clean.isdigit():
        return True
    # Still has @@ suffix — it's a subword fragment
    if token.endswith('@@'):
        return True
    return False


def _resolve_path(base: str, filename: str) -> Optional[str]:
    """Return the path if it exists on disk, else None."""
    p = os.path.join(base, filename)
    return p if os.path.isfile(p) else None


class EvidenceLoader:
    """Loads XAI evidence artifacts for a given sample."""

    def load(
        self,
        sample_id: str,
        xai_dir: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load all available evidence for a sample.

        Returns:
            Dict with keys per XAI phase, plus _missing, _warnings,
            and visual_artifacts.
        """
        evidence: Dict[str, Any] = {}
        missing: List[str] = []
        warnings: List[str] = []

        # ── Grad-CAM ────────────────────────────────────────────────
        gc_dir = os.path.join(xai_dir, 'gradcam', sample_id)
        gc_meta = _load_json(os.path.join(gc_dir, 'metadata.json'))
        if gc_meta:
            evidence['gradcam'] = gc_meta
        else:
            missing.append('gradcam')

        # ── Attention ───────────────────────────────────────────────
        at_dir = os.path.join(xai_dir, 'attention', sample_id)
        attn_topk = _load_json(os.path.join(at_dir, 'topk_tokens.json'))
        attn_words = _load_json(os.path.join(at_dir, 'word_importance.json'))

        # Filter noisy tokens
        if attn_topk:
            attn_topk = [e for e in attn_topk
                         if not _is_noisy_token(e.get('token', ''))]
        if attn_words:
            items = attn_words.get('word_importances', [])
            attn_words['word_importances'] = [
                w for w in items
                if not _is_noisy_token(w.get('word', ''))]

        if attn_topk or attn_words:
            evidence['attention'] = {
                'topk_tokens': attn_topk,
                'word_importance': attn_words,
            }
        else:
            missing.append('attention')

        # ── Cross-Attention ─────────────────────────────────────────
        ca_dir = os.path.join(xai_dir, 'cross_attention', sample_id)
        ca_summary = _load_json(
            os.path.join(ca_dir, 'cross_attention_summary.json'))
        ca_topk = _load_json(
            os.path.join(ca_dir, 'token_patch_topk.json'))
        if ca_topk:
            ca_topk = [e for e in ca_topk
                       if not _is_noisy_token(e.get('token', ''))]
        if ca_summary or ca_topk:
            evidence['cross_attention'] = {
                'summary': ca_summary,
                'topk_pairs': ca_topk,
            }
        else:
            missing.append('cross_attention')

        # ── SHAP ────────────────────────────────────────────────────
        shap_contrib = _load_json(
            os.path.join(xai_dir, 'shap', sample_id,
                         'shap_modality_contribution.json'))
        if shap_contrib:
            evidence['shap'] = shap_contrib
        else:
            missing.append('shap')

        # ── LIME ────────────────────────────────────────────────────
        lime_dir = os.path.join(xai_dir, 'lime', sample_id)
        lime_text: Dict[str, Any] = {}
        for factor in _FACTOR_NAMES:
            path = os.path.join(
                lime_dir, f'{sample_id}_lime_text_{factor}_weights.json')
            data = _load_json(path)
            if data is not None:
                lime_text[factor] = data
        if lime_text:
            evidence['lime'] = {'text_weights': lime_text}
        else:
            missing.append('lime')

        # ── Case study metadata ─────────────────────────────────────
        if case_id:
            cs_meta = _load_json(
                os.path.join(xai_dir, 'case_studies', case_id,
                             'metadata.json'))
            if cs_meta:
                evidence['case_study'] = cs_meta
            else:
                warnings.append(f'Case study {case_id} metadata not found')

        # ── Visual artifact paths ───────────────────────────────────
        visuals: Dict[str, Optional[str]] = {}
        visuals['gradcam_food'] = _resolve_path(
            gc_dir, 'gradcam_img0_food.png')
        visuals['gradcam_overall'] = _resolve_path(
            gc_dir, 'gradcam_img0_overall.png')
        visuals['gradcam_comparison'] = _resolve_path(
            gc_dir, 'gradcam_5target_comparison.png')
        visuals['attention_heatmap'] = _resolve_path(
            at_dir, 'attention_layer11_mean_heatmap.png')
        visuals['attention_word_bar'] = _resolve_path(
            at_dir, 'cls_importance_word_bar.png')
        visuals['cross_attention_heatmap'] = _resolve_path(
            ca_dir, 'topk_token_patch_heatmap.png')
        visuals['cross_attention_bipartite'] = _resolve_path(
            ca_dir, 'token_patch_bipartite_graph.png')
        visuals['cross_attention_overlay'] = _resolve_path(
            ca_dir, 'top_tokens_patch_overlay_grid.png')
        visuals['cross_attention_patch_importance'] = _resolve_path(
            ca_dir, 'patch_importance.png')
        visuals['shap_chart'] = _resolve_path(
            os.path.join(xai_dir, 'shap', sample_id),
            'shap_modality_contribution.png')
        for factor in _FACTOR_NAMES:
            visuals[f'lime_text_{factor}'] = _resolve_path(
                lime_dir, f'{sample_id}_lime_text_{factor}_bar.png')
            visuals[f'lime_image_{factor}'] = _resolve_path(
                lime_dir,
                f'{sample_id}_lime_image_{factor}_positive.png')
        if case_id:
            cs_dir = os.path.join(xai_dir, 'case_studies', case_id)
            visuals['combined_core'] = _resolve_path(
                cs_dir, 'combined_figure_target0_food.png')
            visuals['combined_cross_attention'] = _resolve_path(
                cs_dir, 'combined_cross_attention_figure.png')

        evidence['visual_artifacts'] = {
            k: v for k, v in visuals.items() if v is not None}
        evidence['_missing'] = missing
        evidence['_warnings'] = warnings

        return evidence
