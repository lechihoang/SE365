# Phase 4: Fusion-level SHAP Analysis — Implementation Notes

## 1. What Was Implemented

| Item | Status |
|---|---|
| `xai/shap_explainer.py` | Implemented |
| `FusionHeadWrapper` class | Implemented — wraps model.head + score_index |
| `extract_fused_embeddings()` | Implemented — forward_pre_hook on model.head |
| `select_background()` | Implemented — random sample from validation embeddings |
| `compute_shap_values()` | Implemented — DeepExplainer primary |
| `modality_contribution()` | Implemented — text-origin (0:512) vs image-origin (512:1024) |
| `additivity_check()` | Implemented — verifies prediction ≈ base + sum(shap) |
| `plot_modality_contribution()` | Implemented — grouped bar chart, 5 targets |
| `SHAPExplainer` class | Implemented — high-level orchestrator |
| `run_ablation_check()` | Implemented — zero-out text/image dims |
| Phase 4 notebook | Implemented with 15-sample batch processing |

## 2. Proposal Deviations

### 2.1 Experiment ID: EXP_060A instead of EXP_050C

- **Proposal:** References `EXP_050C` throughout.
- **Actual:** Uses `EXP_060A_bestsequential_full_configuration` everywhere.
- **Why:** EXP_060A is the current best experiment used by all previous phases. EXP_050C is outdated.

### 2.2 No separate `shap_config.py`

- **Proposal:** Create `xai/shap_config.py` for SHAP-specific configuration.
- **Actual:** SHAP constants (text_dim=512, background_size=100) are defined in `shap_explainer.py` itself, with overridable parameters.
- **Why:** Phases 2 and 3 established the pattern of keeping method-specific constants inside the explainer module, not in separate config files. The shared constants (TARGET_NAMES, FACTOR_NAMES, etc.) are already in `xai/config.py`.

### 2.3 No modality-level KernelExplainer

- **Proposal:** Implement 2-feature KernelExplainer as supplementary validation.
- **Actual:** Implemented ablation check instead (zero-out text/image dims).
- **Why:** Ablation provides cleaner modality importance evidence without the complexity of a second SHAP explainer. It directly answers "what happens when text/image features are removed?" which is more intuitive for thesis presentation. The ablation check is also faster and does not require the shap library.

### 2.4 No beeswarm plots

- **Proposal:** Generate beeswarm/summary plots at dataset level.
- **Actual:** Focus on modality contribution charts and per-sample waterfall-style analysis.
- **Why:** Beeswarm plots for 1024 anonymous dimensions provide little interpretable insight. The key thesis question is "which modality dominates per target?" — answered by the modality contribution summary chart.

## 3. Technical Decisions

### 3.1 Forward pre-hook on model.head
The fused vector is captured using `model.head.register_forward_pre_hook()`. The pre-hook receives `(module, input_args)` where `input_args[0]` is the fused tensor [B, 1024]. This is the cleanest approach since `torch.cat` is a functional call that can't be hooked.

### 3.2 DeepExplainer as primary method
DeepExplainer is fast (gradient-based, no sampling) and well-suited for the MLP head. The head is a standard `Linear→ReLU→Dropout→Linear→ReLU→Linear` that DeepExplainer handles natively.

### 3.3 Terminology: "text-origin" and "image-origin"
All outputs use "text-origin" and "image-origin" rather than "text" and "image" to acknowledge that cross-attended features contain information from both modalities.

### 3.4 Graceful shap import
The module handles `ImportError` for the `shap` library gracefully, printing an install instruction.

## 4. Assumptions

- Model is CrossAttentionFusion with head = Sequential(1024→512→ReLU→DO→512→256→ReLU→256→5)
- Fused vector is [B, 1024] with text-origin in dims 0:512, image-origin in dims 512:1024
- model.eval() is called before any SHAP computation (disables Dropout)
- Background samples are from the validation set (not training set) to avoid data leakage
- DeepExplainer handles the MLP head correctly (all operations are differentiable)

## 5. Files Created

| File | Description |
|---|---|
| `xai/shap_explainer.py` | Core SHAP module: FusionHeadWrapper, embedding extraction, SHAP computation, modality analysis, ablation, plotting |
| `xai/notebooks/Phase4_SHAP.ipynb` | Interactive notebook with embedding extraction, SHAP analysis, modality contribution, ablation |
| `doc/XAI/Phase_4_IMPLEMENTATION_NOTES.md` | This document |

## 6. Files Modified

None. No existing files were modified.

## 7. Reusable Components for Later Phases

| Component | Reusable by |
|---|---|
| `extract_fused_embeddings()` | Phase 6 (Case Studies) — extract embeddings for selected samples |
| `modality_contribution()` | Phase 6 — compute text/image balance for case study panels |
| `FusionHeadWrapper` | Phase 5 (LIME) — if LIME is applied at the fusion level |
| `plot_modality_contribution()` | Phase 8 (Thesis Visualization) — publication-quality modality charts |
| `run_ablation_check()` | Phase 6 — ablation as cross-method validation |

## 8. Known Limitations

1. **Cross-attention mixing:** Text-origin features contain image information and vice versa. The grouping is by query origin, not by pure modality content.
2. **DeepExplainer approximation:** SHAP values are approximate (DeepLIFT-based), not exact Shapley values. Acknowledged in thesis methodology.
3. **Background sensitivity:** Results depend on background sample selection. The 100-sample random selection from validation set is standard but not exhaustive.
4. **No per-feature interpretation:** Individual dimensions of the 1024-d fused vector don't have semantic labels. Only grouped modality analysis is interpretable.

## 9. Future Improvements

1. **Background stability test:** Compare results with 50, 100, 200 background samples.
2. **Error-stratified analysis:** Compare modality contribution for correct vs. incorrect predictions.
3. **Per-target waterfall plots:** Using shap.Explanation objects for detailed feature-level views.

## 10. Migration Note

The proposal references `EXP_050C` as the best experiment. This has been replaced with `EXP_060A_bestsequential_full_configuration` throughout the implementation to match all previous phases (1-3) which already use this experiment. The model architecture and fusion type remain identical (CrossAttentionFusion + LogCosh).
