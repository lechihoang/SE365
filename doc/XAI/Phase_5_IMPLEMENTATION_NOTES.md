# Phase 5: LIME Local Explanation — Implementation Notes

## 1. What Was Implemented

| Item | Status |
|---|---|
| `xai/lime_explainer.py` | Implemented |
| `ImageLimePredictFn` class | Implemented — wraps model for image perturbations |
| `TextLimePredictFn` class | Implemented — wraps model for text perturbations |
| `run_lime_image()` | Implemented — orchestrates single image LIME explanation |
| `run_lime_text()` | Implemented — orchestrates single text LIME explanation |
| `save_lime_image_explanation()` | Implemented — saves superpixel overlays + weights JSON |
| `save_lime_text_explanation()` | Implemented — saves word bar chart + weights JSON + HTML |
| `LIMEExplainer` class | Implemented — high-level orchestrator for full sample explanation |
| Phase 5 notebook | Implemented with 5-sample processing |
| Sample manifest files | Generated: CSV, JSON, Markdown preview |

## 2. Proposal Deviations

### 2.1 Reduced sample count

- **Proposal:** 5-10 case study samples as minimum viable scope.
- **Actual:** 5 samples as default (`NUM_LIME_SAMPLES = 5`).
- **Why:** LIME is computationally expensive (1000 image perturbations + 500 text perturbations per sample per target = many forward passes). 5 samples × 5 targets × 2 modalities is already substantial. Can be increased by changing the config variable.

### 2.2 No stability analysis in default run

- **Proposal:** Run with 3 different seeds to assess stability.
- **Actual:** Single seed (42) by default. Stability analysis can be added as an optional notebook cell.
- **Why:** Running 3 seeds triples the computation time. For 5 samples × 5 targets × 2 modalities × 3 seeds, the total would be ~750 LIME runs. A single-seed run is sufficient for thesis case studies.

### 2.3 Image resized to 224×224 before LIME

- **Proposal:** Discusses whether to resize or not.
- **Actual:** Resize to 224×224 before passing to LIME.
- **Why:** Matching the model's native resolution ensures superpixel masks align with what the model processes. Large original images would create unnecessarily fine superpixel segmentation.

## 3. Technical Decisions

### 3.1 Sigmoid pseudo-classification mapping
LIME expects classification-like [N, num_classes] output. The regression score is mapped through sigmoid: `p_high = sigmoid(score)`, output = `[1-p_high, p_high]`. This creates a pseudo two-class output where "high" = high target score.

### 3.2 Batch processing for GPU memory
Both predict functions process perturbations in internal batches (default 32) to avoid GPU OOM. With num_samples=1000, this means ~31 batches per LIME run.

### 3.3 Vietnamese text splitting
LIME splits text by whitespace (`split_expression=r'\s+'`), which naturally produces Vietnamese syllables. This aligns with PhoBERT's syllable-level tokenization.

### 3.4 Multi-image handling
For image LIME, only the first real image (index 0) is perturbed. All other images remain fixed. This isolates the visual contribution of the primary image.

### 3.5 Graceful lime import
The module handles ImportError for the lime package gracefully, printing install instructions.

## 4. Assumptions

- Model is CrossAttentionFusion in eval mode
- Image processor is TimmProcessor (from get_image_processor())
- Tokenizer is PhoBERT (from get_tokenizer())
- max_length=256 for tokenization (matches training)
- max_images=4 for multi-image (matches dataset)
- LIME's hide_color=0 (black) for hidden superpixels
- LIME's default segmentation (quickshift) for superpixels

## 5. Files Created

| File | Description |
|---|---|
| `xai/lime_explainer.py` | Core LIME module with predict functions, runners, saving, and LIMEExplainer class |
| `xai/notebooks/Phase5_LIME.ipynb` | Interactive notebook with 5-sample LIME analysis |
| `doc/XAI/Phase_5_IMPLEMENTATION_NOTES.md` | This document |

## 6. Files Modified

None.

## 7. Reusable Components for Later Phases

| Component | Reusable by |
|---|---|
| `ImageLimePredictFn` | Phase 6 (Case Studies) — image perturbation analysis |
| `TextLimePredictFn` | Phase 6 — text perturbation analysis |
| `save_lime_image_explanation()` | Phase 8 (Thesis Visualization) — publication figures |
| `save_lime_text_explanation()` | Phase 8 — text importance figures |
| `LIMEExplainer` | Phase 6 — batch LIME for case studies |

## 8. Known Limitations

1. **Computational cost:** Each image LIME run requires ~1000 model forward passes. Each text LIME requires ~500. Total for 5 samples × 5 targets × 2 modalities ≈ 37,500 forward passes.
2. **Instability:** LIME results can vary across random seeds. A single seed is used for reproducibility but may not capture the full picture.
3. **Sigmoid mapping is a heuristic:** The regression-to-pseudo-classification conversion is not theoretically grounded. It works in practice because LIME only needs monotonic input-output relationships.
4. **Single image explained:** For multi-image reviews, only the first image is explained. Other images remain fixed.
5. **Superpixel granularity:** The default quickshift segmentation may not align with semantic boundaries (food vs. background).

## 9. Suggestions for Phase 6

- Load LIME word weights and Attention top tokens for the same sample to compute cross-method agreement
- Load LIME superpixel maps and Grad-CAM heatmaps to compute spatial overlap
- Use LIME stability analysis (multi-seed) for selected case studies
- Combine LIME, Grad-CAM, Attention, and SHAP results into unified case study panels
