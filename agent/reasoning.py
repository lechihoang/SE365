"""
Pre-LLM reasoning layer: builds a structured reasoning graph from
XAI evidence before sending it to the language model.

The reasoning graph ranks evidence, detects conflicts, and produces
an intermediate representation that the LLM verbalizes — rather than
letting the LLM invent reasoning from scratch.

Pipeline:
  Evidence (loader) → Reasoning Graph → Prompt → LLM → Report
"""

from typing import Dict, Any, List, Optional

_FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']
_FACTOR_DISPLAY = {
    'food': 'Food Score', 'price': 'Price Score',
    'atmos': 'Atmosphere Score', 'service': 'Service Score',
    'overall': 'Overall Satisfaction',
}

# Keywords per target for matching review text to targets
_TARGET_KEYWORDS = {
    'food': ['ngon', 'dở', 'ăn', 'món', 'thức', 'vị', 'thơm', 'tươi',
             'nóng', 'nguội', 'dai', 'chắc', 'ngọt', 'béo', 'mặn',
             'cay', 'bánh', 'cơm', 'phở', 'bún', 'chả', 'ghẹ', 'bò',
             'gà', 'heo', 'cá', 'tôm', 'rau', 'nước', 'lèo', 'topping'],
    'price': ['giá', 'tiền', 'đắt', 'rẻ', 'mắc', 'hợp lý', 'worth',
              'bình dân', 'phải chăng', 'xứng đáng', 'chi phí'],
    'atmos': ['không gian', 'view', 'đẹp', 'sạch', 'thoáng', 'ồn',
              'trang trí', 'decor', 'nội thất', 'ghế', 'bàn', 'sang'],
    'service': ['phục vụ', 'nhân viên', 'thái độ', 'nhanh', 'chậm',
                'nhiệt tình', 'chu đáo', 'order', 'gọi món', 'đợi'],
    'overall': ['tổng', 'chung', 'hài lòng', 'quay lại', 'recommend',
                'giới thiệu', 'trải nghiệm', 'ok', 'ổn'],
}


def _strength(value: float, thresholds: tuple = (0.3, 0.6)) -> str:
    """Classify a normalized value into low/moderate/high."""
    if value < thresholds[0]:
        return 'low'
    elif value < thresholds[1]:
        return 'moderate'
    return 'high'


def _text_mentions_target(text: str, target: str) -> List[str]:
    """Return keywords from the review that relate to a target."""
    text_lower = text.lower()
    return [kw for kw in _TARGET_KEYWORDS.get(target, [])
            if kw in text_lower]


def build_reasoning_graph(
    predictions: Dict[str, float],
    evidence: Dict[str, Any],
    review_text: str = '',
) -> Dict[str, Any]:
    """Build a structured reasoning graph from evidence.

    For each of the 5 targets, produces:
    - supporting_evidence: ranked list of evidence items
    - contradicting_evidence: conflicts detected
    - missing_evidence: which methods have no data
    - evidence_strength: overall strength assessment
    - review_keywords: target-relevant words found in review
    - interpretation_hint: pre-computed reasoning hint for the LLM

    Also produces:
    - agreement_matrix: per-target per-method strength summary
    """
    from agent.output_schema import score_to_level

    # Map prediction keys to factor names
    _pred_map = {
        'food_score': 'food', 'price_score': 'price',
        'atmosphere_score': 'atmos', 'service_score': 'service',
        'overall_satisfaction': 'overall',
    }
    pred_by_factor = {}
    for k, v in predictions.items():
        factor = _pred_map.get(k)
        if factor:
            pred_by_factor[factor] = v

    # Pre-extract evidence components
    attn_data = evidence.get('attention', {})
    topk_tokens = attn_data.get('topk_tokens', []) or []
    word_imp = attn_data.get('word_importance', {})
    word_items = word_imp.get('word_importances', []) if word_imp else []

    ca_data = evidence.get('cross_attention', {})
    ca_pairs = ca_data.get('topk_pairs', []) or []
    ca_summary = ca_data.get('summary', {}) or {}

    shap_data = evidence.get('shap', {})
    lime_data = evidence.get('lime', {})
    lime_text = lime_data.get('text_weights', {}) if lime_data else {}

    gc_data = evidence.get('gradcam', {})

    visuals = evidence.get('visual_artifacts', {})

    targets = {}
    agreement_rows = []

    for factor in _FACTOR_NAMES:
        score = pred_by_factor.get(factor, 0)
        level = score_to_level(score)
        keywords = _text_mentions_target(review_text, factor)

        supporting = []
        contradicting = []
        missing_ev = []

        # ── Attention evidence ──────────────────────────────────
        attn_strength = 'missing'
        relevant_tokens = []
        for entry in (topk_tokens or [])[:10]:
            tok = entry.get('token', '')
            imp = entry.get('importance', 0)
            tok_lower = tok.lower().replace('Ġ', '')
            if any(kw in tok_lower for kw in
                   _TARGET_KEYWORDS.get(factor, [])):
                relevant_tokens.append(f'{tok} ({imp:.4f})')
        if relevant_tokens:
            attn_strength = 'high' if len(relevant_tokens) >= 2 else 'moderate'
            supporting.append({
                'method': 'attention',
                'evidence': f'Relevant tokens: {", ".join(relevant_tokens[:5])}',
                'strength': attn_strength,
                'rank': 1,
            })
        elif topk_tokens:
            attn_strength = 'weak'
            supporting.append({
                'method': 'attention',
                'evidence': 'Tokens available but none directly relevant',
                'strength': 'weak', 'rank': 5,
            })
        else:
            missing_ev.append('attention')

        # ── Cross-Attention evidence ────────────────────────────
        ca_strength = 'missing'
        relevant_ca = []
        for pair in (ca_pairs or [])[:10]:
            tok = pair.get('token', '')
            tok_lower = tok.lower().replace('Ġ', '')
            if any(kw in tok_lower for kw in
                   _TARGET_KEYWORDS.get(factor, [])):
                pr = pair.get('patch_row', -1)
                pc = pair.get('patch_col', -1)
                att = pair.get('attention', 0)
                relevant_ca.append(
                    f'{tok}→Patch({pr},{pc}) score={att:.4f}')
        if relevant_ca:
            ca_strength = 'high' if len(relevant_ca) >= 2 else 'moderate'
            supporting.append({
                'method': 'cross_attention',
                'evidence': '; '.join(relevant_ca[:3]),
                'strength': ca_strength, 'rank': 2,
            })
        elif ca_pairs:
            ca_strength = 'weak'
        else:
            missing_ev.append('cross_attention')

        # ── SHAP evidence ──────────────────────────────────────
        shap_strength = 'missing'
        shap_entry = shap_data.get(factor, {})
        if shap_entry:
            tp = shap_entry.get('text_pct', 50)
            ip = shap_entry.get('image_pct', 50)
            ts = shap_entry.get('text_signed', None)
            dominance = ('text-origin dominant' if tp > 60
                         else 'image-origin dominant' if ip > 60
                         else 'balanced')
            shap_str = f'text-origin {tp:.1f}%, image-origin {ip:.1f}% ({dominance})'
            if ts is not None:
                shap_str += f', text signed={ts:+.4f}'
            shap_strength = 'high'
            supporting.append({
                'method': 'shap',
                'evidence': shap_str,
                'strength': 'high', 'rank': 2,
            })
            # Conflict: text-origin high but no text keywords
            if tp > 70 and not keywords:
                contradicting.append({
                    'type': 'shap_text_mismatch',
                    'detail': (f'SHAP shows {tp:.0f}% text-origin '
                               f'but review has no {factor} keywords'),
                })
        else:
            missing_ev.append('shap')

        # ── LIME evidence ──────────────────────────────────────
        lime_strength = 'missing'
        lime_factor = lime_text.get(factor)
        if lime_factor:
            items = (lime_factor if isinstance(lime_factor, list)
                     else lime_factor.get('weights',
                          lime_factor.get('features', [])))
            if items:
                pos = [(w, s) for w, s in items if s > 0][:3]
                neg = [(w, s) for w, s in items if s < 0][:3]
                if pos or neg:
                    parts = []
                    if pos:
                        parts.append('positive: ' + ', '.join(
                            f'{w}(+{s:.3f})' for w, s in pos))
                    if neg:
                        parts.append('negative: ' + ', '.join(
                            f'{w}({s:.3f})' for w, s in neg))
                    lime_strength = ('moderate' if len(pos) + len(neg) >= 2
                                     else 'weak')
                    supporting.append({
                        'method': 'lime',
                        'evidence': '; '.join(parts),
                        'strength': lime_strength, 'rank': 3,
                    })
                else:
                    lime_strength = 'weak'
            else:
                lime_strength = 'weak'
        else:
            missing_ev.append('lime')

        # ── Grad-CAM evidence ──────────────────────────────────
        gc_strength = 'missing'
        if gc_data:
            gc_strength = 'weak'
            gc_path = visuals.get(f'gradcam_{factor}')
            if gc_path:
                gc_strength = 'moderate'
                supporting.append({
                    'method': 'gradcam',
                    'evidence': f'Overlay available at {gc_path}',
                    'strength': 'moderate', 'rank': 4,
                })
        else:
            missing_ev.append('gradcam')

        # Sort supporting evidence by rank
        supporting.sort(key=lambda x: x.get('rank', 99))

        # Overall evidence strength
        strengths = [attn_strength, ca_strength, shap_strength,
                     lime_strength, gc_strength]
        high_count = strengths.count('high')
        mod_count = strengths.count('moderate')
        if high_count >= 2:
            overall_strength = 'high'
        elif high_count >= 1 or mod_count >= 2:
            overall_strength = 'moderate'
        elif any(s not in ('missing', 'weak') for s in strengths):
            overall_strength = 'weak'
        else:
            overall_strength = 'very_weak'

        # Interpretation hint
        hint_parts = []
        if keywords:
            hint_parts.append(
                f'Review mentions {factor}-related words: '
                f'{", ".join(keywords[:5])}')
        if shap_entry:
            tp = shap_entry.get('text_pct', 50)
            if tp > 60:
                hint_parts.append('SHAP confirms text-origin dominance')
            elif tp < 40:
                hint_parts.append('SHAP shows image-origin dominance')
        if not hint_parts:
            hint_parts.append(
                'No strong direct evidence for this target')

        targets[factor] = {
            'target': factor,
            'display_name': _FACTOR_DISPLAY[factor],
            'prediction': score,
            'level': level,
            'review_keywords': keywords,
            'supporting_evidence': supporting,
            'contradicting_evidence': contradicting,
            'missing_evidence': missing_ev,
            'evidence_strength': overall_strength,
            'interpretation_hint': '. '.join(hint_parts) + '.',
        }

        agreement_rows.append({
            'target': factor,
            'gradcam': gc_strength,
            'attention': attn_strength,
            'cross_attention': ca_strength,
            'shap': shap_strength,
            'lime': lime_strength,
            'overall_agreement': overall_strength,
        })

    return {
        'targets': targets,
        'agreement_matrix': agreement_rows,
    }
