# Phase 5: LIME Local Explanation — Implementation Notes (V2)

Updated for the **token × patch CrossAttentionFusion** architecture on branch `xai-v3`.

---

## 1. Proposal Compliance

| Item | Status | Notes |
|---|---|---|
| `xai/lime_explainer.py` | Unchanged | All classes and functions correct for new architecture |
| `ImageLimePredictFn` class | Unchanged | Wraps model for image perturbations |
| `TextLimePredictFn` class | Unchanged | Wraps model for text perturbations |
| `run_lime_image()` | Unchanged | Orchestrates single image LIME explanation |
| `run_lime_text()` | Unchanged | Orchestrates single text LIME explanation |
| `save_lime_image_explanation()` | Unchanged | Saves superpixel overlays + weights JSON |
| `save_lime_text_explanation()` | Unchanged | Saves word bar chart + weights JSON + HTML |
| `LIMEExplainer` class | Unchanged | High-level orchestrator |
| Phase 5 notebook | **Updated** | Branch → `xai-v3` |

---

## 2. Differences from V1

### 2.1 Branch updated

All notebook clone cells use `xai-v3` (was `xai-v2`).

### 2.2 Sample count increased

`NUM_LIME_SAMPLES` changed from 5 to 15 for consistency with Phase 2 (Grad-CAM), Phase 3 (Attention), and Phase 4 (SHAP), which all process 15 samples.

### 2.3 Zero code changes to `lime_explainer.py`

LIME is **architecture-agnostic** — it treats the model as a black box. The predict functions (`ImageLimePredictFn`, `TextLimePredictFn`) call `model(input_ids, attention_mask, pixel_values, num_images)` which is the **unchanged** external API of CrossAttentionFusion.

---

## 3. Cross-Attention V3 Compatibility

### Why `lime_explainer.py` needed no changes

LIME wraps the model as a black box:

1. **Image LIME:** Perturbs superpixels, keeps text fixed, calls `model(...)`, reads `preds[:, score_index]`
2. **Text LIME:** Perturbs words, keeps images fixed, calls `model(...)`, reads `preds[:, score_index]`

The model's forward signature `(input_ids, attention_mask, pixel_values, num_images) → preds` is unchanged. LIME never accesses model internals (no hooks, no attention weights, no fused embeddings). The architecture redesign is invisible to LIME.

### What IS different (behavioral, not code)

The model's predictions may be slightly different for the same input because the token×patch cross-attention produces richer fused representations. This means LIME explanations generated with V2 may differ from V1 — but this is expected and correct (the model is genuinely different).

---

## 4. Engineering Decisions

- Sigmoid pseudo-classification mapping for regression scores
- Batch processing (default 32) for GPU memory management
- Vietnamese text splitting via `split_expression=r'\s+'`
- Only first image explained for multi-image reviews
- `.contiguous()` after `.expand()` in TextLimePredictFn (tensor safety)
- Graceful `lime` import with install instructions

---

## 5. Compatibility with Future Phases

| Phase | Status | Notes |
|---|---|---|
| Phase 6 (Case Study) | Compatible | Artifact format unchanged |
| Phase 7 (Report) | Compatible | Metadata format unchanged |
| Phase 8 (Thesis Viz) | Compatible | Figure format unchanged |

---

## 6. Remaining Limitations

1. **Computational cost:** ~1000 image + ~500 text perturbations per sample per target.
2. **Instability:** Results vary across random seeds.
3. **Sigmoid mapping heuristic:** Not theoretically grounded for regression.
4. **Single image explained:** Only first image in multi-image reviews.
5. **Superpixel granularity:** Default quickshift may not align with semantic boundaries.

---

## 7. Summary

### V2 changes
- Notebook: branch → `xai-v3`
- Python module: **zero changes**

### Ready for use?
**Yes.** LIME is architecture-agnostic. The model's external API is unchanged. Re-run on Colab to generate explanations with the new architecture's predictions.
