"""
Compress raw XAI evidence into concise text blocks for the LLM prompt.

Takes the structured evidence dict from EvidenceLoader and produces
a compact text representation that fits within LLM token budgets.
Uses Top-K selection for attention, cross-attention, and LIME.

Important: uses "text-origin" / "image-origin" (not "pure text" /
"pure image") because cross-attention mixes modality information.
"""

from typing import Dict, Any, List, Optional

from agent.config import AgentConfig


# Factor display names matching xai/config.py
_FACTOR_DISPLAY = {
    'food': 'Food Score',
    'price': 'Price Score',
    'atmos': 'Atmosphere Score',
    'service': 'Service Score',
    'overall': 'Overall Satisfaction',
}
_FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']


class EvidenceBuilder:
    """Builds compressed evidence text from raw XAI artifacts."""

    def __init__(self, config: Optional[AgentConfig] = None):
        cfg = config or AgentConfig()
        self.top_k_tokens = cfg.top_k_tokens
        self.top_k_pairs = cfg.top_k_pairs
        self.top_k_lime = cfg.top_k_lime

    def build(self, evidence: Dict[str, Any]) -> Dict[str, str]:
        """Convert evidence dict into per-method text blocks.

        Returns dict with keys: gradcam, attention, cross_attention,
        shap, lime, missing_summary.  Each value is a short text
        string suitable for inclusion in an LLM prompt.
        """
        blocks: Dict[str, str] = {}
        blocks['gradcam'] = self._build_gradcam(evidence.get('gradcam'))
        blocks['attention'] = self._build_attention(evidence.get('attention'))
        blocks['cross_attention'] = self._build_cross_attention(
            evidence.get('cross_attention'))
        blocks['shap'] = self._build_shap(evidence.get('shap'))
        blocks['lime'] = self._build_lime(evidence.get('lime'))

        missing = evidence.get('_missing', [])
        if missing:
            blocks['missing_summary'] = (
                'Missing evidence: ' + ', '.join(missing))
        else:
            blocks['missing_summary'] = 'All XAI evidence available.'

        return blocks

    # ── Per-method builders ──────────────────────────────────────────

    def _build_gradcam(self, data: Optional[Dict]) -> str:
        if not data:
            return 'Grad-CAM evidence not available.'
        layer = data.get('target_layer', 'unknown')
        n_img = data.get('num_images', 1)
        n_tgt = data.get('num_targets', 5)
        return (
            f'Grad-CAM attached to {layer}. '
            f'{n_img} image(s) analyzed, {n_tgt} targets.'
        )

    def _build_attention(self, data: Optional[Dict]) -> str:
        if not data:
            return 'PhoBERT attention evidence not available.'
        topk = data.get('topk_tokens')
        if not topk:
            words = data.get('word_importance', {})
            items = words.get('word_importances', [])
            if items:
                topk = [{'token': w['word'], 'importance': w['importance']}
                        for w in items[:self.top_k_tokens]]
        if not topk:
            return 'Attention data present but no top tokens found.'

        lines = [f'Top tokens by CLS attention (PhoBERT):']
        for i, entry in enumerate(topk[:self.top_k_tokens]):
            tok = entry.get('token', '?')
            imp = entry.get('importance', 0)
            lines.append(f'  {i+1}. {tok} ({imp:.4f})')
        return '\n'.join(lines)

    def _build_cross_attention(self, data: Optional[Dict]) -> str:
        if not data:
            return 'Cross-attention evidence not available.'
        parts: List[str] = []

        summary = data.get('summary')
        if summary:
            ent = summary.get('mean_token_entropy', 0)
            parts.append(f'Mean token entropy: {ent:.3f} bits')
            top_toks = summary.get('top_5_tokens', [])
            if top_toks:
                tok_str = ', '.join(
                    f'{t["token"]} ({t["importance"]:.4f})'
                    for t in top_toks[:5])
                parts.append(f'Top tokens (by total attention): {tok_str}')

        topk = data.get('topk_pairs')
        if topk:
            parts.append('Top token→patch pairs:')
            for entry in topk[:self.top_k_pairs]:
                tok = entry.get('token', '?')
                pr = entry.get('patch_row', -1)
                pc = entry.get('patch_col', -1)
                att = entry.get('attention', 0)
                parts.append(
                    f'  {tok} → Patch({pr},{pc}) attn={att:.4f}')

        return '\n'.join(parts) if parts else (
            'Cross-attention data present but empty.')

    def _build_shap(self, data: Optional[Dict]) -> str:
        if not data:
            return 'SHAP modality contribution not available.'
        lines = ['SHAP modality contribution (text-origin vs image-origin):']
        for factor in _FACTOR_NAMES:
            entry = data.get(factor)
            if not entry:
                continue
            tp = entry.get('text_pct', 0)
            ip = entry.get('image_pct', 0)
            ts = entry.get('text_signed', None)
            ims = entry.get('image_signed', None)
            line = f'  {_FACTOR_DISPLAY.get(factor, factor)}: ' \
                   f'text-origin {tp:.1f}%, image-origin {ip:.1f}%'
            if ts is not None and ims is not None:
                line += f' (text signed={ts:+.4f}, image signed={ims:+.4f})'
            lines.append(line)
        return '\n'.join(lines)

    def _build_lime(self, data: Optional[Dict]) -> str:
        if not data:
            return 'LIME evidence not available.'
        text_w = data.get('text_weights', {})
        if not text_w:
            return 'LIME data present but no text weights found.'
        lines = ['LIME text evidence (top positive/negative words):']
        for factor in _FACTOR_NAMES:
            fw = text_w.get(factor)
            if not fw:
                continue
            # fw is either a list of [word, weight] or a dict
            if isinstance(fw, list):
                items = fw
            elif isinstance(fw, dict):
                items = fw.get('weights', fw.get('features', []))
            else:
                continue
            if not items:
                continue
            pos = [(w, s) for w, s in items if s > 0][:self.top_k_lime]
            neg = [(w, s) for w, s in items if s < 0][:self.top_k_lime]
            pos_str = ', '.join(f'{w} (+{s:.3f})' for w, s in pos) or '(none)'
            neg_str = ', '.join(f'{w} ({s:.3f})' for w, s in neg) or '(none)'
            lines.append(
                f'  {_FACTOR_DISPLAY.get(factor, factor)}: '
                f'positive=[{pos_str}], negative=[{neg_str}]')
        return '\n'.join(lines)
