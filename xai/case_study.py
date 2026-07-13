"""
Case study selection and analysis for the multimodal XAI pipeline.

Phase 6 of the XAI pipeline. Selects representative samples across seven
case types (correct, high_error, text_dominant, image_dominant, conflict,
difficult, agreement), assembles multi-method explanation figures from
Phases 2-5 artifacts, and generates structured metadata and analysis text.

When a model is provided, the ``XAIOrchestrator`` can generate missing
artifacts on-demand by calling the Phase 2-5 explainer modules.  When no
model is given, the module operates as a **pure consumer** of existing
artifacts (backward-compatible).

Usage (consumer-only, existing behavior):
    from xai.case_study import CaseStudyRunner

    runner = CaseStudyRunner(
        exp_dir='experiments/EXP_060A_bestsequential_full_configuration',
        dataset_csv='data/text/test.csv',
        image_dir='data/image',
    )
    results = runner.run()

Usage (with on-demand generation):
    runner = CaseStudyRunner(
        exp_dir='experiments/EXP_060A_bestsequential_full_configuration',
        dataset_csv='data/text/test.csv',
        image_dir='data/image',
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        shap_background=background_tensor,
    )
    results = runner.run()
"""

import os
import ast
import json
import time
import hashlib
import datetime
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from xai.config import (
    TARGET_NAMES, FACTOR_NAMES, DISPLAY_NAMES, FACTOR_TO_DISPLAY,
    FACTOR_TO_INDEX, INDEX_TO_FACTOR, NUM_TARGETS,
    DEFAULT_DPI, THESIS_DPI, COLOR_SCHEMES, SCORE_RANGE, BEST_EXP_ID,
)

logger = logging.getLogger('xai')

# ── Case type definitions ────────────────────────────────────────────────────

CASE_TYPES = [
    'conflict', 'high_error', 'text_dominant', 'image_dominant',
    'difficult', 'agreement', 'correct',
]
CASE_PRIORITY = {ct: i for i, ct in enumerate(CASE_TYPES)}
CASE_TARGET_COUNTS = {
    'conflict': (1, 3), 'high_error': (2, 4), 'text_dominant': (1, 3),
    'image_dominant': (1, 3), 'difficult': (1, 2), 'agreement': (1, 2),
    'correct': (2, 4),
}

# Vietnamese aspect keywords for text richness scoring
ASPECT_KEYWORDS = {
    'food': ['ngon', 'do an', 'mon', 'thuc an', 'banh', 'com', 'pho',
             'bun', 'cha', 'nem', 'nuoc', 'topping', 'vi', 'chat luong'],
    'price': ['gia', 're', 'dat', 'tien', 'chi phi', 'hop ly',
              'phai chang', 'mac', 'binh dan', 'worth'],
    'atmos': ['khong gian', 'view', 'dep', 'sach', 'thoang', 'am cung',
              'trang tri', 'decor', 'noi that', 'ban ghe', 'sang trong'],
    'service': ['phuc vu', 'nhan vien', 'thai do', 'nhanh', 'cham',
                'nhiet tinh', 'tu van', 'order', 'goi mon', 'doi'],
    'overall': ['tong', 'chung', 'hai long', 'quay lai', 'gioi thieu',
                'recommend', 'trai nghiem'],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Artifact Checking
# ══════════════════════════════════════════════════════════════════════════════

def check_sample_artifacts(sample_id: str, xai_dir: str) -> Dict[str, Any]:
    """Check which XAI phases have actual artifact files for a given sample.

    Checks for specific expected files, not just directory existence.

    Args:
        sample_id: Sample identifier, e.g. 'sample_0042'.
        xai_dir: Base XAI directory, e.g. '{exp_dir}/xai'.

    Returns:
        Dict with per-phase availability (bool), file paths found,
        and 'completeness' (float 0.0-1.0).
    """
    result = {}
    found = 0

    def _check(phase, patterns):
        """Check if any of the expected files exist for a phase."""
        phase_dir = os.path.join(xai_dir, phase, sample_id)
        if not os.path.isdir(phase_dir):
            return False, []
        existing = []
        for pat in patterns:
            p = os.path.join(phase_dir, pat)
            if os.path.isfile(p):
                existing.append(p)
        return len(existing) > 0, existing

    # Grad-CAM: at least one overlay file
    gc_patterns = [f'gradcam_img0_{f}.png' for f in FACTOR_NAMES]
    gc_patterns.append('gradcam_5target_comparison.png')
    gc_ok, gc_files = _check('gradcam', gc_patterns)
    result['gradcam'] = gc_ok

    # Attention: word importance bar or heatmap
    at_patterns = ['cls_importance_word_bar.png', 'cls_importance_subword_bar.png',
                   'attention_layer11_mean_heatmap.png', 'word_importance.json']
    at_ok, at_files = _check('attention', at_patterns)
    result['attention'] = at_ok

    # SHAP: modality contribution chart or JSON
    sh_patterns = ['shap_modality_contribution.png', 'shap_modality_contribution.json']
    sh_ok, sh_files = _check('shap', sh_patterns)
    result['shap'] = sh_ok

    # LIME: at least one image or text explanation
    lm_patterns = [f'{sample_id}_lime_image_{f}_positive.png' for f in FACTOR_NAMES]
    lm_patterns += [f'{sample_id}_lime_text_{f}_bar.png' for f in FACTOR_NAMES]
    lm_ok, lm_files = _check('lime', lm_patterns)
    result['lime'] = lm_ok

    # Cross-Attention: token-patch visualization artifacts
    ca_patterns = [
        'cross_attention_raw.npz', 'cross_attention_summary.json',
        'topk_token_patch_heatmap.png', 'token_patch_bipartite_graph.png',
        'top_tokens_patch_overlay_grid.png', 'top_patches_token_rankings.png',
    ]
    ca_ok, ca_files = _check('cross_attention', ca_patterns)
    result['cross_attention'] = ca_ok

    found = sum(int(result[p]) for p in
                ['gradcam', 'attention', 'shap', 'lime', 'cross_attention'])
    result['completeness'] = found / 5
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAP Contribution Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_shap_contribution(
    sample_id: str, shap_dir: str,
) -> Optional[Dict[str, Any]]:
    """Load SHAP modality contribution JSON for a sample.

    Returns dict keyed by factor name with 'text_pct', 'image_pct',
    'text_signed', 'image_signed', etc. None if not found.
    """
    json_path = os.path.join(shap_dir, sample_id,
                             'shap_modality_contribution.json')
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f'[XAI] WARNING: Failed to load SHAP for {sample_id}: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Selection Score Computation
# ══════════════════════════════════════════════════════════════════════════════

def _aspect_keyword_coverage(text: str) -> float:
    """Fraction of 5 aspects whose keywords appear in review text."""
    text_lower = text.lower()
    return sum(
        any(kw in text_lower for kw in ASPECT_KEYWORDS.get(f, []))
        for f in FACTOR_NAMES
    ) / len(FACTOR_NAMES)


def _get_errors(row: pd.Series) -> List[float]:
    """Extract per-factor absolute errors from prediction row."""
    return [row.get(f'absolute_error_{f}', 0.0) for f in FACTOR_NAMES]


def compute_selection_score(
    row: pd.Series, text: str, num_images: int,
    shap_data: Optional[Dict[str, Any]],
    artifact_info: Dict[str, Any], case_type: str,
) -> float:
    """Compute multi-criteria selection score for a candidate sample.

    Score = PredictionQuality * VisualRichness * TextRichness
          * MultimodalBalance * ExplanationCompleteness
    """
    errors = _get_errors(row)
    mean_error, max_error = float(np.mean(errors)), float(np.max(errors))

    # PredictionQuality
    if case_type in ('correct', 'agreement'):
        pq = max(0.01, 1.0 / (1.0 + mean_error))
    elif case_type == 'high_error':
        pq = max_error
    else:
        pq = 1.0

    # VisualRichness
    vr = max(1, num_images)

    # TextRichness
    tr = max(0.1, min(1.0, len(text) / 100.0)
             * max(0.1, _aspect_keyword_coverage(text)))

    # MultimodalBalance
    if shap_data and case_type in ('agreement', 'correct'):
        bals = [1.0 - abs(shap_data[f].get('text_pct', 50) / 100 - 0.5)
                for f in FACTOR_NAMES if f in shap_data]
        mb = float(np.mean(bals)) if bals else 0.5
    elif shap_data and case_type == 'text_dominant':
        mb = float(np.mean([shap_data[f].get('text_pct', 50) / 100
                            for f in FACTOR_NAMES if f in shap_data]) or 0.5)
    elif shap_data and case_type == 'image_dominant':
        mb = float(np.mean([shap_data[f].get('image_pct', 50) / 100
                            for f in FACTOR_NAMES if f in shap_data]) or 0.5)
    else:
        mb = 1.0

    ec = artifact_info.get('completeness', 0.0)
    # Heavily penalize samples with no artifacts at all
    if ec == 0.0:
        return 0.0

    # Bonus for cross-attention availability (multiplicative boost)
    ca_boost = 1.2 if artifact_info.get('cross_attention', False) else 1.0

    return float(pq * vr * tr * mb * ec * ca_boost)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Case Selection
# ══════════════════════════════════════════════════════════════════════════════

def _get_text_and_images(df: pd.DataFrame, idx: int) -> Tuple[str, int]:
    """Extract review text and image count from dataset row."""
    row = df.iloc[idx]
    text = str(row.get('comment_clean', ''))
    try:
        urls = ast.literal_eval(str(row.get('image_url', '[]')))
        n = len(urls) if isinstance(urls, list) else 1
    except (ValueError, SyntaxError):
        n = 1
    return text, min(n, 4)


def select_cases(
    pred_df: pd.DataFrame, dataset_df: pd.DataFrame, xai_dir: str,
) -> Dict[str, List[Tuple[int, float, str]]]:
    """Select candidate samples for each case type with progressive relaxation.

    De-duplicates by CASE_PRIORITY (conflict > high_error > ... > correct).

    Returns dict mapping case_type -> list of (sample_idx, score, reason).
    """
    n = min(len(pred_df), len(dataset_df))
    shap_dir = os.path.join(xai_dir, 'shap')

    # Pre-compute per-sample data
    sd = {}
    for idx in range(n):
        sid = f'sample_{idx:04d}'
        text, n_img = _get_text_and_images(dataset_df, idx)
        sd[idx] = {
            'row': pred_df.iloc[idx], 'sid': sid, 'text': text,
            'n_img': n_img,
            'shap': load_shap_contribution(sid, shap_dir),
            'art': check_sample_artifacts(sid, xai_dir),
        }

    with_art = sum(1 for d in sd.values() if d['art']['completeness'] > 0)
    print(f'[XAI] Pre-computed {n} samples ({with_art} with XAI artifacts)')

    def _score(idx: int, ct: str) -> float:
        d = sd[idx]
        return compute_selection_score(
            d['row'], d['text'], d['n_img'], d['shap'], d['art'], ct)

    raw: Dict[str, List[Tuple[int, float, str]]] = {}

    # Helper: skip samples with zero artifacts
    def _has_artifacts(idx):
        return sd[idx]['art']['completeness'] > 0

    # CORRECT: all errors < threshold
    for thr in [0.3, 0.5, 0.8]:
        cands = []
        for idx, d in sd.items():
            if not _has_artifacts(idx):
                continue
            errs = _get_errors(d['row'])
            if all(e < thr for e in errs):
                cands.append((idx, _score(idx, 'correct'),
                              f'All errors < {thr:.1f} (mean={np.mean(errs):.3f})'))
        if len(cands) >= CASE_TARGET_COUNTS['correct'][0]:
            break
    raw['correct'] = sorted(cands, key=lambda x: -x[1])

    # HIGH_ERROR: max error > threshold
    for thr in [2.0, 1.5, 1.0]:
        cands = []
        for idx, d in sd.items():
            if not _has_artifacts(idx):
                continue
            errs = _get_errors(d['row'])
            mx = max(errs)
            if mx > thr:
                wf = FACTOR_NAMES[int(np.argmax(errs))]
                cands.append((idx, _score(idx, 'high_error'),
                              f'Max error {mx:.3f} > {thr:.1f} (worst: {wf})'))
        if len(cands) >= CASE_TARGET_COUNTS['high_error'][0]:
            break
    raw['high_error'] = sorted(cands, key=lambda x: -x[1])

    # TEXT_DOMINANT: SHAP text_pct > threshold for majority of targets
    for thr in [65.0, 55.0, 50.0]:
        cands = []
        for idx, d in sd.items():
            if not _has_artifacts(idx):
                continue
            sh = d['shap']
            if not sh:
                continue
            cnt = sum(1 for f in FACTOR_NAMES
                      if f in sh and sh[f].get('text_pct', 0) > thr)
            if cnt >= 3:
                avg = np.mean([sh[f].get('text_pct', 50)
                               for f in FACTOR_NAMES if f in sh])
                cands.append((idx, _score(idx, 'text_dominant'),
                              f'Text dominant {cnt}/5 (mean={avg:.1f}%, thr={thr:.0f}%)'))
        if len(cands) >= CASE_TARGET_COUNTS['text_dominant'][0]:
            break
    raw['text_dominant'] = sorted(cands, key=lambda x: -x[1])

    # IMAGE_DOMINANT: SHAP image_pct > threshold for majority
    for thr in [55.0, 50.0, 45.0]:
        cands = []
        for idx, d in sd.items():
            if not _has_artifacts(idx):
                continue
            sh = d['shap']
            if not sh:
                continue
            cnt = sum(1 for f in FACTOR_NAMES
                      if f in sh and sh[f].get('image_pct', 0) > thr)
            if cnt >= 3:
                avg = np.mean([sh[f].get('image_pct', 50)
                               for f in FACTOR_NAMES if f in sh])
                cands.append((idx, _score(idx, 'image_dominant'),
                              f'Image dominant {cnt}/5 (mean={avg:.1f}%, thr={thr:.0f}%)'))
        if len(cands) >= CASE_TARGET_COUNTS['image_dominant'][0]:
            break
    raw['image_dominant'] = sorted(cands, key=lambda x: -x[1])

    # CONFLICT: signed SHAP text/image push opposite + moderate error
    cands = []
    for idx, d in sd.items():
        if not _has_artifacts(idx):
            continue
        sh = d['shap']
        if not sh:
            continue
        me = float(np.mean(_get_errors(d['row'])))
        if me < 0.3:
            continue
        cf = [f for f in FACTOR_NAMES if f in sh
              and sh[f].get('text_signed', 0) * sh[f].get('image_signed', 0) < 0]
        if cf:
            s = _score(idx, 'conflict') * len(cf)
            cands.append((idx, s,
                          f'Text-image conflict on {", ".join(cf)} (me={me:.3f})'))
    raw['conflict'] = sorted(cands, key=lambda x: -x[1])

    # DIFFICULT: no keywords for a target but model predicts well
    cands = []
    for idx, d in sd.items():
        if not _has_artifacts(idx):
            continue
        tl = d['text'].lower()
        errs = _get_errors(d['row'])
        df_facts = [f for fi, f in enumerate(FACTOR_NAMES)
                    if not any(kw in tl for kw in ASPECT_KEYWORDS.get(f, []))
                    and errs[fi] < 1.0]
        if df_facts:
            s = _score(idx, 'difficult') * len(df_facts)
            cands.append((idx, s,
                          f'No keywords for {", ".join(df_facts)}, predicted well'))
    raw['difficult'] = sorted(cands, key=lambda x: -x[1])

    # AGREEMENT: low error + balanced SHAP
    cands = []
    for idx, d in sd.items():
        if not _has_artifacts(idx):
            continue
        me = float(np.mean(_get_errors(d['row'])))
        sh = d['shap']
        if me > 0.5 or not sh:
            continue
        bals = [1.0 - abs(sh[f].get('text_pct', 50) / 100 - 0.5)
                for f in FACTOR_NAMES if f in sh]
        mb = float(np.mean(bals)) if bals else 0.0
        if mb > 0.7:
            cands.append((idx, _score(idx, 'agreement'),
                          f'Low error ({me:.3f}) + balanced SHAP ({mb:.3f})'))
    raw['agreement'] = sorted(cands, key=lambda x: -x[1])

    # De-duplicate by priority
    assigned: set = set()
    final: Dict[str, List[Tuple[int, float, str]]] = {}
    for ct in CASE_TYPES:
        sel = []
        for idx, score, reason in raw.get(ct, []):
            if idx in assigned:
                continue
            sel.append((idx, score, reason))
            assigned.add(idx)
            if len(sel) >= CASE_TARGET_COUNTS[ct][1]:
                break
        final[ct] = sel

    total = sum(len(v) for v in final.values())
    print(f'[XAI] Selection complete: {total} cases')
    for ct in CASE_TYPES:
        mn, mx = CASE_TARGET_COUNTS[ct]
        c = len(final[ct])
        print(f'[XAI]   {ct:16s}: {c} (target {mn}-{mx})'
              f' [{"OK" if c >= mn else "BELOW_MIN"}]')
    return final


# ══════════════════════════════════════════════════════════════════════════════
# 5. Combined Figure Generation
# ══════════════════════════════════════════════════════════════════════════════

def _load_artifact_image(path: str) -> Optional[np.ndarray]:
    """Load a PNG artifact as numpy array. None on failure."""
    if not os.path.isfile(path):
        return None
    try:
        return np.array(Image.open(path).convert('RGB'))
    except Exception as e:
        print(f'[XAI] WARNING: Failed to load {path}: {e}')
        return None


def _load_original_image(
    dataset_row: pd.Series, image_dir: str, image_idx: int = 0,
) -> Optional[np.ndarray]:
    """Load original review image from cache directory."""
    try:
        urls = ast.literal_eval(str(dataset_row.get('image_url', '[]')))
        if not isinstance(urls, list) or image_idx >= len(urls):
            return None
        url_hash = hashlib.md5(urls[image_idx].encode('utf-8')).hexdigest()
        local = os.path.join(image_dir, f'{url_hash}.jpg')
        if os.path.isfile(local):
            return np.array(Image.open(local).convert('RGB'))
    except Exception as e:
        print(f'[XAI] WARNING: Failed to load original image: {e}')
    return None


def _placeholder(ax, msg: str) -> None:
    """Render a grey 'not available' placeholder on axes."""
    ax.set_facecolor('#f0f0f0')
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=12, color='#888888', style='italic')
    ax.set_xticks([])
    ax.set_yticks([])


def create_combined_figure(
    sample_id: str, xai_dir: str, target_idx: int, factor_name: str,
    dataset_row: pd.Series, pred_row: pd.Series,
    shap_data: Optional[Dict[str, Any]], image_dir: str,
    save_path: str, dpi: int = THESIS_DPI,
) -> str:
    """Create a 3x2 combined multi-method explanation figure (12x14 in).

    Row 1: Original image | Grad-CAM overlay
    Row 2: Attention word bar | SHAP modality bar
    Row 3: LIME image (positive) | LIME text bar
    Missing artifacts show placeholder panels.

    Returns save_path on success.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10

    dn = DISPLAY_NAMES[target_idx]
    yt = pred_row.get(f'y_true_{factor_name}', float('nan'))
    yp = pred_row.get(f'y_pred_{factor_name}', float('nan'))
    err = pred_row.get(f'absolute_error_{factor_name}', float('nan'))

    # Locate artifacts
    def _art(phase, filename):
        return os.path.join(xai_dir, phase, sample_id, filename)

    arts = {
        'original': _load_original_image(dataset_row, image_dir),
        'gradcam':  _load_artifact_image(_art('gradcam', f'gradcam_img0_{factor_name}.png')),
        'attention': _load_artifact_image(_art('attention', 'cls_importance_word_bar.png')),
        'shap':     _load_artifact_image(_art('shap', 'shap_modality_contribution.png')),
        'lime_img': _load_artifact_image(_art('lime', f'{sample_id}_lime_image_{factor_name}_positive.png')),
        'lime_txt': _load_artifact_image(_art('lime', f'{sample_id}_lime_text_{factor_name}_bar.png')),
    }

    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.05, 1, 1, 1],
                  hspace=0.25, wspace=0.15)

    # Title
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis('off')
    ax_t.text(0.5, 0.5,
              f'{sample_id}  |  {dn}  |  True: {yt:.2f}  Pred: {yp:.2f}  Error: {err:.3f}',
              transform=ax_t.transAxes, ha='center', va='center',
              fontsize=13, fontweight='bold')

    # Panel rendering helper
    panels = [
        (gs[1, 0], arts['original'],  'Original Image',             'Original image\nnot available'),
        (gs[1, 1], arts['gradcam'],   f'Grad-CAM: {dn}',           f'Grad-CAM ({factor_name})\nnot available'),
        (gs[2, 0], arts['attention'], 'Attention: Word Importance',  'Attention\nnot available'),
        (gs[2, 1], arts['shap'],      'SHAP: Modality Contribution','SHAP\nnot available'),
        (gs[3, 0], arts['lime_img'],  f'LIME Image: {dn}',         f'LIME image\nnot available'),
        (gs[3, 1], arts['lime_txt'],  f'LIME Text: {dn}',          f'LIME text\nnot available'),
    ]
    for spec, img, title, fallback in panels:
        ax = fig.add_subplot(spec)
        if img is not None:
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
        else:
            _placeholder(ax, fallback)
        ax.axis('off')

    # Footer
    review = str(dataset_row.get('comment_clean', ''))
    if len(review) > 300:
        review = review[:297] + '...'
    pred_line = '  |  '.join(
        f'{DISPLAY_NAMES[i]}: {pred_row.get(f"y_true_{f}", 0):.1f}'
        f'->{pred_row.get(f"y_pred_{f}", 0):.1f}'
        for i, f in enumerate(FACTOR_NAMES))
    fig.text(0.5, 0.01, f'Review: {review}\n{pred_line}',
             ha='center', va='bottom', fontsize=8, wrap=True,
             style='italic', color='#444444')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[XAI] Saved combined figure: {save_path}')
    return save_path


def create_cross_attention_figure(
    sample_id: str, xai_dir: str,
    dataset_row: pd.Series, pred_row: pd.Series,
    image_dir: str, save_path: str, dpi: int = THESIS_DPI,
) -> Optional[str]:
    """Create a combined cross-attention visualization figure.

    Assembles token-patch overlay grid, top-K heatmap, bipartite graph,
    and patch importance into a single thesis-ready figure.

    Returns save_path on success, None if no cross-attention artifacts exist.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10

    ca_dir = os.path.join(xai_dir, 'cross_attention', sample_id)
    if not os.path.isdir(ca_dir):
        return None

    arts = {
        'original':  _load_original_image(dataset_row, image_dir),
        'overlay_grid': _load_artifact_image(
            os.path.join(ca_dir, 'top_tokens_patch_overlay_grid.png')),
        'topk_heatmap': _load_artifact_image(
            os.path.join(ca_dir, 'topk_token_patch_heatmap.png')),
        'bipartite': _load_artifact_image(
            os.path.join(ca_dir, 'token_patch_bipartite_graph.png')),
        'patch_importance': _load_artifact_image(
            os.path.join(ca_dir, 'patch_importance.png')),
    }

    has_any = any(v is not None for k, v in arts.items() if k != 'original')
    if not has_any:
        return None

    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.05, 1, 1, 1],
                  hspace=0.25, wspace=0.15)

    # Title
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis('off')
    ax_t.text(0.5, 0.5,
              f'{sample_id}  |  Cross-Attention: Token × Patch Analysis',
              transform=ax_t.transAxes, ha='center', va='center',
              fontsize=14, fontweight='bold')

    panels = [
        (gs[1, 0], arts['original'],       'Original Image',
         'Original image\nnot available'),
        (gs[1, 1], arts['patch_importance'], 'Patch Importance Overlay',
         'Patch importance\nnot available'),
        (gs[2, :], arts['overlay_grid'],    'Top Tokens → Patch Overlay Grid',
         'Token→Patch overlay\nnot available'),
        (gs[3, 0], arts['topk_heatmap'],    'Top-K Token × Patch Heatmap',
         'Top-K heatmap\nnot available'),
        (gs[3, 1], arts['bipartite'],       'Token ↔ Patch Bipartite Graph',
         'Bipartite graph\nnot available'),
    ]
    for spec, img, title, fallback in panels:
        ax = fig.add_subplot(spec)
        if img is not None:
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
        else:
            _placeholder(ax, fallback)
        ax.axis('off')

    review = str(dataset_row.get('comment_clean', ''))
    if len(review) > 300:
        review = review[:297] + '...'
    fig.text(0.5, 0.01, f'Review: {review}',
             ha='center', va='bottom', fontsize=8, wrap=True,
             style='italic', color='#444444')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[XAI] Saved cross-attention figure: {save_path}')
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# 6. Metadata Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_case_metadata(
    case_id: str, case_type: str, sample_id: str,
    dataset_row: pd.Series, pred_row: pd.Series,
    shap_data: Optional[Dict[str, Any]],
    artifact_info: Dict[str, Any], selection_reason: str,
) -> Dict[str, Any]:
    """Generate comprehensive metadata JSON for a case study."""
    preds, gt, errs = {}, {}, {}
    for i, f in enumerate(FACTOR_NAMES):
        d = DISPLAY_NAMES[i]
        preds[d] = float(pred_row.get(f'y_pred_{f}', 0))
        gt[d] = float(pred_row.get(f'y_true_{f}', 0))
        errs[d] = float(pred_row.get(f'absolute_error_{f}', 0))

    ev = list(errs.values())
    return {
        'case_id': case_id, 'case_type': case_type, 'sample_id': sample_id,
        'selection_reason': selection_reason,
        'timestamp': datetime.datetime.now().isoformat(),
        'review_text': str(dataset_row.get('comment_clean', '')),
        'predictions': preds, 'ground_truth': gt, 'absolute_errors': errs,
        'summary_stats': {
            'mean_absolute_error': round(float(np.mean(ev)), 4),
            'max_absolute_error': round(float(np.max(ev)), 4),
            'worst_factor': DISPLAY_NAMES[int(np.argmax(ev))],
        },
        'artifact_availability': {
            k: artifact_info.get(k, False)
            for k in ['gradcam', 'attention', 'cross_attention',
                       'shap', 'lime', 'completeness']
        },
        'shap_modality_contribution': shap_data or {},
        'target_names': TARGET_NAMES, 'factor_names': FACTOR_NAMES,
        'display_names': DISPLAY_NAMES, 'score_range': list(SCORE_RANGE),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Analysis Text Generation
# ══════════════════════════════════════════════════════════════════════════════

_CASE_DESCRIPTIONS = {
    'correct': (
        'This sample demonstrates accurate prediction across all quality '
        'aspects. The model successfully captured review sentiment and visual '
        'cues to produce predictions close to ground truth.'),
    'high_error': (
        'This sample shows significant prediction error. This failure case '
        'reveals limitations in how the model processes either the textual '
        'review or visual content. Potential causes include ambiguous '
        'language, misleading visuals, or annotation noise.'),
    'text_dominant': (
        'SHAP analysis reveals that text-origin features dominate the '
        'prediction for a majority of quality targets. The review text '
        'carries more explanatory weight than the visual content.'),
    'image_dominant': (
        'SHAP analysis shows that image-origin features dominate the '
        'prediction for a majority of quality targets. Visual information '
        '(food presentation, venue appearance) is the primary driver.'),
    'conflict': (
        'This sample exhibits inter-modal conflict: signed SHAP contributions '
        'from text-origin and image-origin features push in opposite '
        'directions, indicating contradictory text and image signals.'),
    'difficult': (
        'The review text lacks explicit keywords for one or more quality '
        'aspects, yet the model produces reasonable predictions. This suggests '
        'reliance on implicit cues or cross-modal information transfer.'),
    'agreement': (
        'This sample shows low prediction error and balanced modality '
        'contribution. Text and image features contribute roughly equally, '
        'representing an ideal case of multimodal agreement.'),
}


def generate_analysis_text(
    case_id: str, case_type: str, metadata: Dict[str, Any],
) -> str:
    """Generate structured Markdown analysis text for a case study."""
    m = metadata
    stats = m.get('summary_stats', {})
    preds = m.get('predictions', {})
    gt = m.get('ground_truth', {})
    errs = m.get('absolute_errors', {})
    shap_c = m.get('shap_modality_contribution', {})
    arts = m.get('artifact_availability', {})
    review = m.get('review_text', '')

    lines = [
        f'# Case Study: {case_id}', '',
        f'**Case Type:** {case_type}',
        f'**Sample:** {m.get("sample_id", "")}',
        f'**Selection Reason:** {m.get("selection_reason", "")}', '',
        '## Prediction Summary', '',
        '| Target | Ground Truth | Prediction | Error |',
        '|--------|-------------|------------|-------|',
    ]
    for dn in DISPLAY_NAMES:
        lines.append(f'| {dn} | {gt.get(dn, 0):.2f} | {preds.get(dn, 0):.2f} '
                      f'| {errs.get(dn, 0):.3f} |')

    lines.extend([
        '', f'**Mean Error:** {stats.get("mean_absolute_error", 0):.4f}  |  '
        f'**Max Error:** {stats.get("max_absolute_error", 0):.4f}  |  '
        f'**Worst:** {stats.get("worst_factor", "")}', '',
        '## Analysis', '', _CASE_DESCRIPTIONS.get(case_type, ''),
    ])

    # SHAP detail for relevant case types
    if shap_c and case_type in ('correct', 'text_dominant', 'image_dominant',
                                 'conflict', 'agreement'):
        lines.append('')
        if case_type == 'conflict':
            lines.append('### Conflict Details')
            for f in FACTOR_NAMES:
                if f not in shap_c:
                    continue
                ts = shap_c[f].get('text_signed', 0)
                ims = shap_c[f].get('image_signed', 0)
                if ts * ims < 0:
                    lines.append(
                        f'- **{FACTOR_TO_DISPLAY.get(f, f)}:** Text '
                        f'{"up" if ts > 0 else "down"} ({ts:+.4f}), Image '
                        f'{"up" if ims > 0 else "down"} ({ims:+.4f})')
        else:
            lines.append('### Modality Contribution')
            for f in FACTOR_NAMES:
                if f not in shap_c:
                    continue
                tp = shap_c[f].get('text_pct', 0)
                ip = shap_c[f].get('image_pct', 0)
                lines.append(
                    f'- **{FACTOR_TO_DISPLAY.get(f, f)}:** '
                    f'Text {tp:.1f}% | Image {ip:.1f}%')

    # Cross-attention interpretation
    if arts.get('cross_attention'):
        lines.extend([
            '', '### Cross-Attention (Token × Patch)', '',
            'The cross-attention mechanism reveals which text tokens attend to which '
            'image patches. Token-to-patch overlays and the bipartite graph show the '
            'strongest cross-modal connections. Manual interpretation of the '
            'generated figures is recommended for thesis discussion.',
        ])

    # Artifact availability
    lines.extend([
        '', '## Available XAI Artifacts', '',
        *[f'- {p.replace("_", " ").title()}: '
          f'{"Available" if arts.get(p) else "Missing"}'
          for p in ['gradcam', 'attention', 'cross_attention', 'shap', 'lime']],
        f'- Completeness: {arts.get("completeness", 0):.0%}',
        '', '## Review Text', '',
        f'> {review[:500]}{"..." if len(review) > 500 else ""}', '',
    ])
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 8. XAI Orchestrator — On-Demand Artifact Generation
# ══════════════════════════════════════════════════════════════════════════════

class XAIOrchestrator:
    """Generates missing XAI artifacts by calling Phase 2-5 explainers.

    Wraps GradCAMExplainer (Phase 2), AttentionExplainer (Phase 3),
    SHAPExplainer (Phase 4), and LIMEExplainer (Phase 5).  Explainers
    are lazily initialized on first use to avoid heavy imports unless
    they are actually needed.

    Args:
        model: The loaded multimodal model (e.g., CrossAttentionFusion).
        tokenizer: HuggingFace tokenizer for the text branch.
        image_processor: HuggingFace image processor for the vision branch.
        device: torch.device for inference.
        xai_dir: Base XAI directory (e.g. '{exp_dir}/xai').
        shap_background: Optional [N, 1024] tensor of fused embeddings for
            SHAP baseline.  If None, SHAP generation is skipped.
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        device,
        xai_dir: str,
        shap_background=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.xai_dir = xai_dir
        self.shap_background = shap_background  # [N, 1024] tensor, optional

        # Lazy-init explainers (created on first use)
        self._gradcam = None
        self._attention = None
        self._cross_attention = None
        self._shap = None
        self._lime = None
        self._attention_patched = False

    # ── Public API ──────────────────────────────────────────────────────

    def ensure_artifacts(
        self, sample: Dict[str, Any], sample_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Check and generate all missing artifacts for a sample.

        For each phase (gradcam, attention, shap, lime), checks whether
        artifacts already exist on disk.  If not, attempts to generate them
        using the corresponding explainer.

        Args:
            sample: Sample dict from load_single_sample().
            sample_id: Unique identifier, e.g. 'sample_0042'.

        Returns:
            Generation log dict mapping phase name to status info::

                {phase: {'status': 'existing'|'generated'|'failed',
                         'time_s': float (only if generated),
                         'error': str (only if failed)}}
        """
        log: Dict[str, Dict[str, Any]] = {}
        art = check_sample_artifacts(sample_id, self.xai_dir)

        for phase in ['gradcam', 'attention', 'cross_attention', 'shap', 'lime']:
            if art[phase]:
                log[phase] = {'status': 'existing'}
            else:
                try:
                    t0 = time.time()
                    self._generate(phase, sample, sample_id)
                    elapsed = round(time.time() - t0, 1)
                    log[phase] = {'status': 'generated', 'time_s': elapsed}
                    print(f'[XAI] Generated {phase} for {sample_id} '
                          f'({elapsed}s)')
                except Exception as e:
                    log[phase] = {'status': 'failed', 'error': str(e)}
                    print(f'[XAI] WARNING: Failed to generate {phase} '
                          f'for {sample_id}: {e}')

        return log

    # ── Private: per-phase generation ───────────────────────────────────

    def _generate(self, phase: str, sample: Dict[str, Any],
                  sample_id: str) -> None:
        """Dispatch generation to the appropriate explainer."""
        if phase == 'gradcam':
            self._generate_gradcam(sample, sample_id)
        elif phase == 'attention':
            self._generate_attention(sample, sample_id)
        elif phase == 'cross_attention':
            self._generate_cross_attention(sample, sample_id)
        elif phase == 'shap':
            self._generate_shap(sample, sample_id)
        elif phase == 'lime':
            self._generate_lime(sample, sample_id)
        else:
            raise ValueError(f'Unknown XAI phase: {phase}')

    def _generate_gradcam(self, sample: Dict[str, Any],
                          sample_id: str) -> None:
        """Generate Grad-CAM artifacts (Phase 2)."""
        if self._gradcam is None:
            from xai.gradcam_explainer import GradCAMExplainer
            output_dir = os.path.join(self.xai_dir, 'gradcam')
            os.makedirs(output_dir, exist_ok=True)
            self._gradcam = GradCAMExplainer(
                self.model, self.device, output_dir)
            print('[XAI] Initialized GradCAMExplainer (on-demand)')
        self._gradcam.explain_sample(sample, sample_id)

    def _generate_attention(self, sample: Dict[str, Any],
                            sample_id: str) -> None:
        """Generate attention artifacts (Phase 3).

        Patches sdpa -> eager attention on first call so that
        output_attentions=True works correctly.
        """
        if self._attention is None:
            # Patch sdpa -> eager before first use
            if not self._attention_patched:
                from xai.utils import enable_eager_attention
                enable_eager_attention(self.model)
                self._attention_patched = True

            from xai.attention_explainer import AttentionExplainer
            output_dir = os.path.join(self.xai_dir, 'attention')
            os.makedirs(output_dir, exist_ok=True)
            self._attention = AttentionExplainer(
                self.model, self.tokenizer, self.device, output_dir)
            print('[XAI] Initialized AttentionExplainer (on-demand)')
        self._attention.explain_sample(sample, sample_id)

    def _generate_cross_attention(self, sample: Dict[str, Any],
                                  sample_id: str) -> None:
        """Generate cross-attention artifacts (Phase 3 — token×patch)."""
        if self._cross_attention is None:
            if not self._attention_patched:
                from xai.utils import enable_eager_attention
                enable_eager_attention(self.model)
                self._attention_patched = True

            from xai.attention_explainer import CrossAttentionExplainer
            output_dir = os.path.join(self.xai_dir, 'cross_attention')
            os.makedirs(output_dir, exist_ok=True)
            self._cross_attention = CrossAttentionExplainer(
                self.model, self.tokenizer, self.device, output_dir)
            print('[XAI] Initialized CrossAttentionExplainer (on-demand)')
        self._cross_attention.explain_sample(sample, sample_id)

    def _generate_shap(self, sample: Dict[str, Any],
                       sample_id: str) -> None:
        """Generate SHAP artifacts (Phase 4).

        Extracts a single-sample fused embedding by hooking model.head,
        then calls SHAPExplainer.explain_sample with the background set.
        """
        if self.shap_background is None:
            raise RuntimeError(
                'SHAP background not provided -- cannot generate SHAP '
                'artifacts on-demand without a background embedding set')

        if self._shap is None:
            from xai.shap_explainer import SHAPExplainer
            output_dir = os.path.join(self.xai_dir, 'shap')
            os.makedirs(output_dir, exist_ok=True)
            self._shap = SHAPExplainer(
                self.model, self.device, output_dir)
            print('[XAI] Initialized SHAPExplainer (on-demand)')

        # Extract fused embedding for this single sample via forward hook
        import torch
        captured: Dict[str, Any] = {}

        def hook_fn(module, args):
            if isinstance(args, tuple) and len(args) > 0:
                captured['fused'] = args[0].detach()
            elif isinstance(args, torch.Tensor):
                captured['fused'] = args.detach()

        handle = self.model.head.register_forward_pre_hook(hook_fn)
        try:
            with torch.no_grad():
                self.model(
                    input_ids=sample['input_ids'],
                    attention_mask=sample['attention_mask'],
                    pixel_values=sample['pixel_values'],
                    num_images=sample.get('num_images'),
                )
            if 'fused' not in captured:
                raise RuntimeError(
                    'Forward hook did not capture fused embedding')
            fused = captured['fused'].cpu()
        finally:
            handle.remove()

        self._shap.explain_sample(fused[0], sample_id, self.shap_background)

    def _generate_lime(self, sample: Dict[str, Any],
                       sample_id: str) -> None:
        """Generate LIME artifacts (Phase 5).

        Uses reduced perturbation counts (500 image, 300 text) for faster
        on-demand generation compared to full batch runs.
        """
        if self._lime is None:
            from xai.lime_explainer import LIMEExplainer
            output_dir = os.path.join(self.xai_dir, 'lime')
            os.makedirs(output_dir, exist_ok=True)
            self._lime = LIMEExplainer(
                self.model, self.tokenizer, self.image_processor,
                self.device, output_dir)
            print('[XAI] Initialized LIMEExplainer (on-demand)')
        self._lime.explain_sample(
            sample, sample_id,
            num_image_samples=500,
            num_text_samples=300,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9. CaseStudyRunner
# ══════════════════════════════════════════════════════════════════════════════

class CaseStudyRunner:
    """High-level runner for Phase 6: Case Study Selection and Analysis.

    Operates in two modes:

    * **Consumer-only** (model=None): reads existing Phase 2-5 artifacts.
      Missing panels show placeholders. This is the original behavior.
    * **Orchestrator** (model provided): uses ``XAIOrchestrator`` to
      generate missing artifacts on-demand before creating figures.

    Args:
        exp_dir: Path to experiment directory.
        dataset_csv: Path to dataset CSV (test.csv or val.csv).
        image_dir: Path to cached image directory.
        split: Data split name ('test' or 'val').
        model: Optional loaded multimodal model.  If provided, enables
            on-demand artifact generation via XAIOrchestrator.
        tokenizer: HuggingFace tokenizer (required if model is provided).
        image_processor: HuggingFace image processor (required if model
            is provided).
        device: torch.device for inference (required if model is provided).
        shap_background: Optional [N, 1024] tensor for SHAP baselines.
            If None, SHAP generation is skipped but other phases can
            still be generated on-demand.
    """

    def __init__(self, exp_dir: str, dataset_csv: str, image_dir: str,
                 split: str = 'test',
                 model=None, tokenizer=None, image_processor=None,
                 device=None, shap_background=None):
        self.exp_dir = exp_dir
        self.dataset_csv = dataset_csv
        self.image_dir = image_dir
        self.split = split
        self.xai_dir = os.path.join(exp_dir, 'xai')
        self.output_dir = os.path.join(self.xai_dir, 'case_studies')
        self.pred_csv = os.path.join(
            exp_dir, 'test_predictions.csv' if split == 'test'
            else 'predictions.csv')

        # Build orchestrator if model is available
        if model is not None:
            self.orchestrator: Optional[XAIOrchestrator] = XAIOrchestrator(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                device=device,
                xai_dir=self.xai_dir,
                shap_background=shap_background,
            )
            mode = 'orchestrator (on-demand generation enabled)'
        else:
            self.orchestrator = None
            mode = 'consumer-only (existing artifacts)'

        print(f'[XAI] CaseStudyRunner initialized')
        print(f'[XAI]   Experiment : {exp_dir}')
        print(f'[XAI]   Dataset    : {dataset_csv}')
        print(f'[XAI]   Predictions: {self.pred_csv}')
        print(f'[XAI]   Output     : {self.output_dir}')
        print(f'[XAI]   Mode       : {mode}')

    def run(self) -> Dict[str, Any]:
        """Run the full case study pipeline. Returns results dict."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Step 1: Load data
        print('[XAI] Step 1: Loading data...')
        for p, label in [(self.pred_csv, 'Prediction CSV'),
                         (self.dataset_csv, 'Dataset CSV')]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f'{label} not found: {p}')

        pred_df = pd.read_csv(self.pred_csv)
        dataset_df = pd.read_csv(self.dataset_csv)
        print(f'[XAI]   Predictions: {len(pred_df)} | Dataset: {len(dataset_df)}')

        # Step 2: Select cases
        print('[XAI] Step 2: Selecting cases...')
        selections = select_cases(pred_df, dataset_df, self.xai_dir)

        # Step 3: Build manifest
        print('[XAI] Step 3: Generating manifests...')
        manifest = self._build_manifest(selections, pred_df)
        self._save_manifests(manifest, dataset_df, pred_df)

        # Step 4: Generate per-case artifacts
        print('[XAI] Step 4: Generating per-case artifacts...')
        cases = self._generate_cases(manifest, dataset_df, pred_df)

        # Step 5: Index and summary
        print('[XAI] Step 5: Writing index and summary...')
        idx_path = self._write_index(cases, manifest, pred_df)
        sum_path = self._write_summary(cases, manifest, pred_df)

        # Step 6: Selection log
        print('[XAI] Step 6: Saving selection log...')
        log_path = self._save_selection_log(selections, pred_df, dataset_df)

        total = len(cases)
        print(f'[XAI] Phase 6 complete: {total} case studies generated')
        print(f'[XAI] Output: {self.output_dir}')

        return {
            'selections': {ct: list(sels) for ct, sels in selections.items()},
            'cases': cases, 'output_dir': self.output_dir,
            'index_csv': idx_path, 'summary_md': sum_path,
            'selection_log': log_path, 'total_cases': total,
        }

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_manifest(self, selections, pred_df):
        """Build manifest row list from selections."""
        rows = []
        seq = {ct: 0 for ct in CASE_TYPES}
        for ct in CASE_TYPES:
            for idx, score, reason in selections.get(ct, []):
                seq[ct] += 1
                errs = _get_errors(pred_df.iloc[idx])
                rows.append({
                    'case_id': f'case_{ct}_{seq[ct]:03d}',
                    'case_type': ct, 'sample_id': f'sample_{idx:04d}',
                    'sample_idx': idx, 'split': self.split,
                    'mean_error': round(float(np.mean(errs)), 4),
                    'score': round(score, 4), 'reason': reason,
                })
        return rows

    def _save_manifests(self, manifest, dataset_df, pred_df):
        """Save CSV, JSON, and preview MD manifests."""
        csv_p = os.path.join(self.output_dir, 'sample_manifest.csv')
        pd.DataFrame(manifest).to_csv(csv_p, index=False, encoding='utf-8')
        print(f'[XAI]   Saved: {csv_p}')

        json_p = os.path.join(self.output_dir, 'sample_manifest.json')
        with open(json_p, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f'[XAI]   Saved: {json_p}')

        md_p = os.path.join(self.output_dir, 'sample_manifest_preview.md')
        lines = ['# Case Study Sample Manifest', '',
                 f'Generated: {datetime.datetime.now().isoformat()}',
                 f'Total cases: {len(manifest)}', '']
        for e in manifest:
            idx = e['sample_idx']
            dr = dataset_df.iloc[idx]
            pr = pred_df.iloc[idx]
            rev = str(dr.get('comment_clean', ''))[:200]
            lines.extend([
                f'## {e["case_id"]}', '',
                f'- **Sample:** {e["sample_id"]} (idx {idx})',
                f'- **Type:** {e["case_type"]}',
                f'- **Reason:** {e["reason"]}', '',
                f'**Review:** {rev}...', '',
                '| Target | True | Pred | Error |',
                '|--------|------|------|-------|',
            ])
            for i, f in enumerate(FACTOR_NAMES):
                lines.append(
                    f'| {DISPLAY_NAMES[i]} | {pr.get(f"y_true_{f}", 0):.2f} '
                    f'| {pr.get(f"y_pred_{f}", 0):.2f} '
                    f'| {pr.get(f"absolute_error_{f}", 0):.3f} |')
            lines.extend(['', '---', ''])
        with open(md_p, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'[XAI]   Saved: {md_p}')

    def _load_sample_for_orchestrator(
        self, idx: int, dataset_df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """Load a single sample dict for on-demand XAI generation.

        Uses the same data loading path as the Phase 2-5 scripts.
        Returns None if loading fails (sample will use existing artifacts
        or placeholders).
        """
        try:
            from xai.utils import load_single_sample
            sample = load_single_sample(
                csv_path=self.dataset_csv,
                idx=idx,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                image_dir=self.image_dir,
                device=self.device,
            )
            return sample
        except ImportError:
            print('[XAI] WARNING: xai.utils.load_single_sample not available')
            return None
        except Exception as e:
            print(f'[XAI] WARNING: Failed to load sample {idx} for '
                  f'orchestrator: {e}')
            return None

    @property
    def tokenizer(self):
        """Access tokenizer from orchestrator (or None)."""
        return self.orchestrator.tokenizer if self.orchestrator else None

    @property
    def image_processor(self):
        """Access image_processor from orchestrator (or None)."""
        return self.orchestrator.image_processor if self.orchestrator else None

    @property
    def device(self):
        """Access device from orchestrator (or None)."""
        return self.orchestrator.device if self.orchestrator else None

    def _generate_cases(self, manifest, dataset_df, pred_df):
        """Generate figures, metadata, analysis for each case.

        When an orchestrator is available, attempts to generate any missing
        XAI artifacts before creating combined figures.  Generation failures
        are logged but do not prevent figure creation (missing panels show
        placeholders).
        """
        cases = []
        generation_log: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for e in manifest:
            cid, ct, idx = e['case_id'], e['case_type'], e['sample_idx']
            sid = e['sample_id']
            print(f'[XAI]   Processing {cid} ({sid})...')

            case_dir = os.path.join(self.output_dir, cid)
            os.makedirs(case_dir, exist_ok=True)

            dr = dataset_df.iloc[idx]
            pr = pred_df.iloc[idx]

            # On-demand artifact generation (if orchestrator available)
            if self.orchestrator is not None:
                sample = self._load_sample_for_orchestrator(idx, dataset_df)
                if sample is not None:
                    gen_log = self.orchestrator.ensure_artifacts(sample, sid)
                    generation_log[cid] = gen_log
                    generated = [p for p, v in gen_log.items()
                                 if v['status'] == 'generated']
                    failed = [p for p, v in gen_log.items()
                              if v['status'] == 'failed']
                    if generated:
                        print(f'[XAI]   Generated: {", ".join(generated)}')
                    if failed:
                        print(f'[XAI]   Failed: {", ".join(failed)}')
                else:
                    print(f'[XAI]   Skipped on-demand generation '
                          f'(sample load failed)')

            # Re-check artifacts after potential generation
            shap = load_shap_contribution(
                sid, os.path.join(self.xai_dir, 'shap'))
            ainfo = check_sample_artifacts(sid, self.xai_dir)

            # Combined core figures for each target
            fig_paths = []
            for ti, fn in enumerate(FACTOR_NAMES):
                fp = os.path.join(case_dir,
                                  f'combined_figure_target{ti}_{fn}.png')
                try:
                    create_combined_figure(
                        sid, self.xai_dir, ti, fn, dr, pr, shap,
                        self.image_dir, fp, THESIS_DPI)
                    fig_paths.append(fp)
                except Exception as ex:
                    print(f'[XAI] WARNING: Figure failed for {cid}/{fn}: {ex}')

            # Cross-attention figure (separate to avoid overcrowding)
            ca_fig_path = os.path.join(
                case_dir, 'combined_cross_attention_figure.png')
            try:
                ca_result = create_cross_attention_figure(
                    sid, self.xai_dir, dr, pr, self.image_dir,
                    ca_fig_path, THESIS_DPI)
                if ca_result:
                    fig_paths.append(ca_fig_path)
            except Exception as ex:
                print(f'[XAI] WARNING: Cross-attention figure failed '
                      f'for {cid}: {ex}')

            # Metadata (include generation log if available)
            meta = generate_case_metadata(
                cid, ct, sid, dr, pr, shap, ainfo, e['reason'])
            if cid in generation_log:
                meta['generation_log'] = generation_log[cid]
            mp = os.path.join(case_dir, 'metadata.json')
            with open(mp, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # Analysis
            analysis = generate_analysis_text(cid, ct, meta)
            ap = os.path.join(case_dir, 'analysis.md')
            with open(ap, 'w', encoding='utf-8') as f:
                f.write(analysis)

            cases.append({
                'case_id': cid, 'case_type': ct, 'sample_id': sid,
                'sample_idx': idx, 'figure_paths': fig_paths,
                'metadata_path': mp, 'analysis_path': ap,
            })

        # Save generation log
        if generation_log:
            self._save_generation_log(generation_log)

        return cases

    def _save_generation_log(
        self, generation_log: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> str:
        """Save on-demand generation log to JSON."""
        # Compute summary statistics
        total_generated = 0
        total_failed = 0
        total_existing = 0
        for case_log in generation_log.values():
            for phase_info in case_log.values():
                status = phase_info.get('status', '')
                if status == 'generated':
                    total_generated += 1
                elif status == 'failed':
                    total_failed += 1
                elif status == 'existing':
                    total_existing += 1

        log_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'summary': {
                'total_cases': len(generation_log),
                'artifacts_existing': total_existing,
                'artifacts_generated': total_generated,
                'artifacts_failed': total_failed,
            },
            'per_case': generation_log,
        }
        p = os.path.join(self.output_dir, 'generation_log.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f'[XAI]   Saved generation log: {p}')
        print(f'[XAI]   Artifacts: {total_existing} existing, '
              f'{total_generated} generated, {total_failed} failed')
        return p

    def _write_index(self, cases, manifest, pred_df):
        """Write case_study_index.csv."""
        rows = []
        mmap = {e['case_id']: e for e in manifest}
        for c in cases:
            errs = _get_errors(pred_df.iloc[c['sample_idx']])
            rows.append({
                'case_id': c['case_id'], 'case_type': c['case_type'],
                'sample_id': c['sample_id'], 'split': self.split,
                'mean_absolute_error': round(float(np.mean(errs)), 4),
                'key_finding': mmap.get(c['case_id'], {}).get('reason', ''),
                'figure_path': c['figure_paths'][0] if c['figure_paths'] else '',
            })
        p = os.path.join(self.output_dir, 'case_study_index.csv')
        pd.DataFrame(rows).to_csv(p, index=False, encoding='utf-8')
        print(f'[XAI]   Saved: {p}')
        return p

    def _write_summary(self, cases, manifest, pred_df):
        """Write case_study_summary.md."""
        lines = [
            '# Case Study Summary', '',
            f'**Experiment:** {os.path.basename(self.exp_dir)}',
            f'**Split:** {self.split}',
            f'**Generated:** {datetime.datetime.now().isoformat()}',
            f'**Total Cases:** {len(cases)}', '',
            '## Case Type Distribution', '',
            '| Case Type | Count | Target |',
            '|-----------|-------|--------|',
        ]
        for ct in CASE_TYPES:
            n = sum(1 for c in cases if c['case_type'] == ct)
            mn, mx = CASE_TARGET_COUNTS[ct]
            lines.append(f'| {ct} | {n} | {mn}-{mx} |')

        lines.extend(['', '## Case Details', ''])
        mmap = {e['case_id']: e for e in manifest}
        for c in cases:
            errs = _get_errors(pred_df.iloc[c['sample_idx']])
            me = float(np.mean(errs))
            r = mmap.get(c['case_id'], {}).get('reason', '')
            lines.extend([
                f'### {c["case_id"]}', '',
                f'- **Sample:** {c["sample_id"]}',
                f'- **Type:** {c["case_type"]}',
                f'- **Mean Error:** {me:.4f}',
                f'- **Reason:** {r}',
                f'- **Figures:** {len(c["figure_paths"])}', '',
            ])

        p = os.path.join(self.output_dir, 'case_study_summary.md')
        with open(p, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'[XAI]   Saved: {p}')
        return p

    def _save_selection_log(self, selections, pred_df, dataset_df):
        """Save selection_log.json with all thresholds and decisions."""
        log = {
            'timestamp': datetime.datetime.now().isoformat(),
            'exp_dir': self.exp_dir, 'dataset_csv': self.dataset_csv,
            'pred_csv': self.pred_csv, 'split': self.split,
            'num_predictions': len(pred_df),
            'num_dataset_rows': len(dataset_df),
            'case_types': {
                ct: {
                    'num_candidates': len(selections.get(ct, [])),
                    'target_range': list(CASE_TARGET_COUNTS[ct]),
                    'selected': [
                        {'sample_idx': i, 'score': round(s, 4), 'reason': r}
                        for i, s, r in selections.get(ct, [])
                        [:CASE_TARGET_COUNTS[ct][1]]
                    ],
                } for ct in CASE_TYPES
            },
        }
        p = os.path.join(self.output_dir, 'selection_log.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f'[XAI]   Saved: {p}')
        return p
