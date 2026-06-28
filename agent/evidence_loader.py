"""
Load XAI artifacts from disk for the AI Agent.

Reads JSON metadata from each XAI phase (Grad-CAM, Attention,
Cross-Attention, SHAP, LIME) and returns a structured evidence
dictionary.  Missing files are handled gracefully — the loader
records which artifacts are available and which are absent.
"""

import os
import json
from typing import Dict, Any, List, Optional


# Factor names matching xai/config.py
_FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file. Returns None on any failure."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


class EvidenceLoader:
    """Loads XAI evidence artifacts for a given sample.

    Checks each phase directory for expected JSON files and returns
    a structured dict with available evidence and warnings for
    missing artifacts.
    """

    def load(
        self,
        sample_id: str,
        xai_dir: str,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load all available evidence for a sample.

        Args:
            sample_id: e.g. 'sample_0042'.
            xai_dir: Base XAI directory (e.g. '{exp_dir}/xai').
            case_id: Optional case study ID for loading case metadata.

        Returns:
            Dict with keys: gradcam, attention, cross_attention, shap,
            lime, case_study, _missing, _warnings.
        """
        evidence: Dict[str, Any] = {}
        missing: List[str] = []
        warnings: List[str] = []

        # Grad-CAM
        gc_meta = _load_json(
            os.path.join(xai_dir, 'gradcam', sample_id, 'metadata.json'))
        if gc_meta:
            evidence['gradcam'] = gc_meta
        else:
            missing.append('gradcam/metadata.json')

        # PhoBERT Attention
        attn_topk = _load_json(
            os.path.join(xai_dir, 'attention', sample_id, 'topk_tokens.json'))
        attn_words = _load_json(
            os.path.join(xai_dir, 'attention', sample_id,
                         'word_importance.json'))
        if attn_topk or attn_words:
            evidence['attention'] = {
                'topk_tokens': attn_topk,
                'word_importance': attn_words,
            }
        else:
            missing.append('attention/topk_tokens.json')

        # Cross-Attention
        ca_summary = _load_json(
            os.path.join(xai_dir, 'cross_attention', sample_id,
                         'cross_attention_summary.json'))
        ca_topk = _load_json(
            os.path.join(xai_dir, 'cross_attention', sample_id,
                         'token_patch_topk.json'))
        if ca_summary or ca_topk:
            evidence['cross_attention'] = {
                'summary': ca_summary,
                'topk_pairs': ca_topk,
            }
        else:
            missing.append('cross_attention/summary.json')

        # SHAP
        shap_contrib = _load_json(
            os.path.join(xai_dir, 'shap', sample_id,
                         'shap_modality_contribution.json'))
        if shap_contrib:
            evidence['shap'] = shap_contrib
        else:
            missing.append('shap/modality_contribution.json')

        # LIME — per-factor text weights
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
            missing.append('lime/text_weights')

        # Case study metadata
        if case_id:
            cs_meta = _load_json(
                os.path.join(xai_dir, 'case_studies', case_id,
                             'metadata.json'))
            if cs_meta:
                evidence['case_study'] = cs_meta
            else:
                warnings.append(f'Case study {case_id} metadata not found')

        evidence['_missing'] = missing
        evidence['_warnings'] = warnings

        return evidence
