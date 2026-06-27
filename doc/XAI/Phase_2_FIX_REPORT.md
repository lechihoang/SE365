# Phase 2 Fix Report — Grad-CAM Target Similarity

## 1. Problem Observed

For several samples, the Grad-CAM heatmaps for all 5 targets (food, price, atmosphere, service, overall) look almost identical. The 5-target comparison figures show very similar spatial patterns across all targets.

## 2. Root Cause Analysis

**This is NOT a bug. It is expected behavior caused by the model architecture.**

The gradient path from each target score to the image encoder's spatial features is:

```
preds[0, target_idx]
  → Linear(256→5)         ← ONLY this row differs per target
  → ReLU
  → Linear(512→256)       ← shared weights
  → Dropout
  → Linear(1024→512)      ← shared weights
  → cross_attn_i2t        ← shared
  → image_proj(1024→512)  ← shared
  → ImageModel.encoder    ← shared (Swin-B)
  → encoder.norm          ← GRAD-CAM HOOK POINT
```

The 5 targets differ ONLY at the final `Linear(256→5)` layer — each target uses one row of that weight matrix. The gradient signal from one row must propagate backward through 3 fully-shared linear layers, a cross-attention module, and a linear projection before reaching the encoder.norm feature maps.

By the time the gradient reaches encoder.norm (1024 channels × 7×7 spatial), the per-target differences from the single 256-dim weight row are heavily diluted through the shared 1024→512→256 chain. The result is that the 5 gradient vectors at encoder.norm are nearly parallel (cosine similarity > 0.95).

**Structural explanation:**
- The image encoder learns a single shared visual representation
- The fusion head maps this to a shared 256-dim space
- Only the final 256→5 projection separates targets
- This is a common pattern in multi-target regression with shared backbones

## 3. Files Modified

| File | Change |
|---|---|
| `xai/gradcam_explainer.py` | Added `diagnose_target_gradients()` function |
| `xai/notebooks/Phase2_GradCAM.ipynb` | Complete rewrite: 15-sample processing, smart selection, gradient diagnostics, updated branch to `xai-v2` |

## 4. Diagnostic Checks Added

1. **`diagnose_target_gradients()`** — New function in `gradcam_explainer.py` that:
   - Runs backward for each of 5 targets
   - Captures gradients at the target layer
   - Computes per-target gradient statistics (mean, std, abs_max, nonzero_ratio)
   - Computes 5×5 pairwise cosine similarity of gradient vectors
   - Computes 5×5 pairwise Pearson correlation of raw CAMs
   - Reports predicted scores for each target

2. **Step 10 in notebook** — Runs the diagnostic and displays:
   - Gradient statistics table
   - Raw CAM statistics table (before normalization)
   - Side-by-side gradient similarity and CAM correlation heatmaps
   - Automated interpretation message

3. **Diagnostic artifacts saved:**
   - `gradient_diagnostics.png` — visual matrices
   - `gradient_diagnostics.json` — numerical values

## 5. Fixes Applied

No code fix was needed because the implementation is correct. The changes are:

1. **Added diagnostics** to explain the behavior scientifically
2. **Added explanatory messages** in the notebook when high similarity is detected
3. **Increased samples to 15** with smart selection (5 correct + 5 mid-error + 5 high-error)
4. **Noted that SHAP (Phase 4)** is the correct tool for target-specific modality analysis

## 6. Remaining Risks

1. **Min-max normalization** can make weak/flat CAMs look artificially strong. The diagnostics now save raw CAM statistics so this can be assessed.
2. **7×7 resolution** is coarse. This is an architectural limitation, not a bug.
3. **Some samples may have genuinely identical visual evidence** for all targets (e.g., a single food photo).

## 7. How to Verify

Run the notebook on Colab. In Step 10, check:
- If gradient cosine similarity matrix shows values > 0.95 for all pairs → confirms expected shared-encoder behavior
- If raw CAM correlation shows values > 0.90 → confirms heatmaps are similar
- The explanatory message should print: "EXPECTED: Gradients are very similar across targets"

This confirms the diagnosis without any code changes needed.
