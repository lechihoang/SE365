"""
Centralized constants for all XAI phases (1-8).

All target names, indices, display names, color schemes, and model dimensions
are defined here. No phase should hardcode these values.
"""

# ── Target definitions (from src/dataset.py) ──────────────────────────────────
TARGET_NAMES = [
    'food_score',
    'price_score',
    'atmosphere_score',
    'service_score',
    'overall_satisfaction',
]

TARGET_INDICES = {
    'food_score': 0,
    'price_score': 1,
    'atmosphere_score': 2,
    'service_score': 3,
    'overall_satisfaction': 4,
}

# Short names used in test.py, Trainer.py
FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']

# Human-readable display names
DISPLAY_NAMES = ['Food Score', 'Price Score', 'Atmosphere Score',
                 'Service Score', 'Overall Satisfaction']

# Column names in dataset CSV (same as TARGET_NAMES)
LABEL_COLS = [
    'food_score',
    'price_score',
    'atmosphere_score',
    'service_score',
    'overall_satisfaction',
]

# Bidirectional mappings
FACTOR_TO_DISPLAY = {
    'food': 'Food Score',
    'price': 'Price Score',
    'atmos': 'Atmosphere Score',
    'service': 'Service Score',
    'overall': 'Overall Satisfaction',
}

INDEX_TO_FACTOR = {0: 'food', 1: 'price', 2: 'atmos', 3: 'service', 4: 'overall'}
FACTOR_TO_INDEX = {'food': 0, 'price': 1, 'atmos': 2, 'service': 3, 'overall': 4}

# Score range (from dataset annotation)
SCORE_RANGE = (1, 10)

# ── Reproducibility ───────────────────────────────────────────────────────────
DEFAULT_SEED = 42

# ── Visualization ─────────────────────────────────────────────────────────────
DEFAULT_DPI = 150
THESIS_DPI = 300

# ── Data defaults (from Config.py / src/dataset.py) ──────────────────────────
DEFAULT_MAX_LENGTH = 256
DEFAULT_MAX_IMAGES = 4

# ── Best model configuration (from Phase 6 experiments) ──────────────────────
BEST_TEXT_MODEL = 'vinai/phobert-base-v2'
BEST_IMAGE_MODEL = 'swin_base_patch4_window7_224'
BEST_FUSION_TYPE = 'cross_attention'
BEST_EXP_ID = 'EXP_060A_bestsequential_full_configuration'

# ── Dimension constants (best model: Swin-B + PhoBERT + CrossAttention) ──────
TEXT_FEATURE_DIM = 768       # PhoBERT hidden_size
IMAGE_FEATURE_DIM = 1024     # Swin-B num_features
CROSS_ATTN_HIDDEN_DIM = 512  # CrossAttentionFusion projection dim
FUSED_DIM = 1024             # After cross-attention concat: 512 + 512
NUM_TARGETS = 5

# ── PhoBERT architecture ─────────────────────────────────────────────────────
PHOBERT_NUM_LAYERS = 12
PHOBERT_NUM_HEADS = 12

# ── Color scheme (consistent across all phases) ──────────────────────────────
COLOR_SCHEMES = {
    'gradcam_cmap': 'jet',
    'attention_cmap': 'magma',
    'shap_positive': '#FF4444',
    'shap_negative': '#4444FF',
    'modality_colors': {'text': '#1b9e77', 'image': '#d95f02'},
    'target_colors': {
        'food': '#E53935',
        'price': '#43A047',
        'atmos': '#1E88E5',
        'service': '#FB8C00',
        'overall': '#8E24AA',
    },
    'bar_gt': '#2196F3',
    'bar_pred': '#FF5722',
}
