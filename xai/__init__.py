"""
XAI Infrastructure Package for Multimodal Restaurant Review Quality Assessment.

Provides reusable utilities for all XAI phases (Grad-CAM, Attention, SHAP, LIME,
Case Studies, Report Generation, Thesis Visualization).

Usage:
    from xai.config import TARGET_NAMES, FACTOR_NAMES, DISPLAY_NAMES
    from xai.utils import load_model, load_single_sample, get_prediction
"""

from xai.config import (
    TARGET_NAMES,
    TARGET_INDICES,
    FACTOR_NAMES,
    DISPLAY_NAMES,
    LABEL_COLS,
    FACTOR_TO_DISPLAY,
    INDEX_TO_FACTOR,
    FACTOR_TO_INDEX,
    SCORE_RANGE,
    DEFAULT_SEED,
    DEFAULT_DPI,
    THESIS_DPI,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_IMAGES,
    BEST_TEXT_MODEL,
    BEST_IMAGE_MODEL,
    BEST_FUSION_TYPE,
    BEST_EXP_ID,
    TEXT_FEATURE_DIM,
    IMAGE_FEATURE_DIM,
    CROSS_ATTN_HIDDEN_DIM,
    FUSED_DIM,
    NUM_TARGETS,
    PHOBERT_NUM_LAYERS,
    PHOBERT_NUM_HEADS,
    COLOR_SCHEMES,
)

from xai.utils import (
    get_device,
    set_seed,
    get_tokenizer,
    get_image_processor,
    load_model,
    load_single_sample,
    get_prediction,
    save_figure,
    save_raw_values,
    enable_eager_attention,
    normalize_feature_map_to_bchw,
)

from xai.attention_explainer import (
    extract_phobert_attention,
    aggregate_attention,
    cls_token_importance,
    merge_subword_attention,
    plot_attention_heatmap,
    plot_cls_importance_bar,
    compute_attention_sink_ratio,
    AttentionExplainer,
    inspect_tokenization,
)

try:
    from xai.shap_explainer import (
        FusionHeadWrapper,
        extract_fused_embeddings,
        select_background,
        compute_shap_values,
        modality_contribution,
        additivity_check,
        plot_modality_contribution,
        plot_modality_single_target,
        SHAPExplainer,
        run_ablation_check,
    )
except ImportError:
    # shap library not installed — Phase 4 functions unavailable
    pass
