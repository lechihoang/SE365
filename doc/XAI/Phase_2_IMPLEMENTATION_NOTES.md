# Phase 2: Grad-CAM — Implementation Notes

## 1. Proposal Compliance

| Item | Status |
|---|---|
| `xai/gradcam_explainer.py` created | Implemented |
| `MultiTargetScoreWrapper` class | Implemented |
| `SwinReshapeTransform` class (as `SwinReshapeTransform`) | Implemented |
| Manual per-image Grad-CAM via hooks | Implemented (`compute_gradcam_for_image`) |
| Heatmap overlay generation | Implemented (`overlay_cam_on_image`) |
| 5-target comparison figure | Implemented (`create_5target_comparison`) |
| Target layer auto-detection | Implemented (`find_target_layer`) |
| `GradCAMExplainer` high-level class | Implemented |
| Phase 2 notebook | Implemented |
| Target specificity validation | Implemented (correlation matrix) |
| Reproducibility validation | Implemented (allclose check) |
| Gradient flow validation | Implemented (backward hook check) |
| Batch processing with summary | Implemented |

## 2. Proposal Deviations

### 2.1 No pytorch-grad-cam dependency

- **Proposal:** Uses `pytorch-grad-cam` library for single-image GradCAM, manual hooks for multi-image.
- **Actual:** All Grad-CAM computation is manual via PyTorch hooks. No `pytorch-grad-cam` dependency.
- **Why:** The multi-image hook isolation approach is already the recommended strategy (Risk R1, Strategy B). Implementing it manually for ALL cases (single and multi-image) eliminates the external dependency, simplifies the install (no `!pip install grad-cam` needed), and provides a single consistent code path. The algorithm is simple: forward hook captures activations, backward hook captures gradients, compute weighted sum + ReLU.

### 2.2 Overlay implementation without show_cam_on_image

- **Proposal:** Uses `pytorch-grad-cam`'s `show_cam_on_image` utility.
- **Actual:** Uses a custom `overlay_cam_on_image()` that uses matplotlib colormaps.
- **Why:** Avoids the pytorch-grad-cam dependency. The overlay logic is simple: resize CAM to image size, apply colormap, blend with original image.

### 2.3 Artifact folder structure

- **Proposal:** Flat structure with `sample_{id}_target{idx}_{name}.png` and `raw/` and `metadata/` subdirs.
- **Actual:** Each sample gets its own subdirectory: `{output_dir}/{sample_id}/`. Individual overlays, raw CAMs (.npz), comparison figure, and metadata.json are inside.
- **Why:** Cleaner organization when processing multiple samples. Easier to locate all artifacts for one sample. The flat structure with many per-sample per-target files becomes unwieldy with 50+ samples.

### 2.4 Randomization sanity check not included

- **Proposal:** Compare trained model heatmap vs randomized model heatmap.
- **Actual:** Not implemented in the notebook.
- **Why:** Creating a randomized model copy requires reinitializing all weights, which is expensive in GPU memory (doubles memory usage). The target specificity check (correlation matrix) and reproducibility check provide sufficient validation for Phase 2. Randomization test can be added as a separate notebook cell if needed.

## 3. Engineering Decisions

### 3.1 Manual Grad-CAM over library-based

The manual implementation gives full control over:
- Multi-image hook isolation (slicing the B*N dimension)
- Gradient management (enable_grad within eval mode)
- Hook cleanup (always in `finally` block)
- No version-specific behavior from external libraries

### 3.2 normalize_feature_map_to_bchw with try/except import

The function is imported from `xai.utils` if available, with a complete local fallback. This handles the case where the remote `utils.py` might not have this function yet.

### 3.3 Hooks always cleaned in finally blocks

Both `compute_gradcam_for_image` and `_verify_target_layer` use try/finally to ensure hooks are removed even if an error occurs during forward/backward.

### 3.4 pixel_values.requires_grad_(True)

The input pixel_values tensor must have `requires_grad=True` for gradients to flow through the encoder to the target layer. This is set via `.clone().detach().requires_grad_(True)` to avoid modifying the original sample tensor.

## 4. Assumptions

- Checkpoint is at `{EXP_DIR}/best_model_train_fusion.pth`
- Model is CrossAttentionFusion with Swin-B + PhoBERT
- Swin-B encoder has a `.norm` layer producing spatial features before pooling
- Feature map channels = 1024 (IMAGE_FEATURE_DIM from config)
- Output format from encoder.norm is [B, H, W, C] (BHWC) — handled by normalize function
- Each sample has 1-4 real images with black padding to 4

## 5. How Phase 2 Reuses Phase 1

| Phase 1 Component | Usage in Phase 2 |
|---|---|
| `load_model()` | Model loading + eval mode |
| `load_single_sample()` | Sample loading with correct preprocessing |
| `get_prediction()` | Get predictions for metadata |
| `save_raw_values()` | Batch summary JSON |
| `get_metadata()` | Standard metadata generation |
| `TARGET_NAMES`, `FACTOR_NAMES`, etc. | All naming constants |
| `IMAGE_FEATURE_DIM` | Expected channels for feature map normalization |
| `DEFAULT_DPI`, `THESIS_DPI` | Figure resolution |
| Eager attention patch | Copied inline in notebook Step 6 |

## 6. Artifacts Generated

Per sample:
- `{sample_id}/gradcam_img{k}_{factor}.png` — overlay for each image × target
- `{sample_id}/raw_cams.npz` — all raw CAM arrays in one file
- `{sample_id}/gradcam_5target_comparison.png` — side-by-side comparison
- `{sample_id}/metadata.json` — generation parameters and paths

Aggregate:
- `gradcam_batch_summary.json` — batch processing status

Validation:
- `{sample_id}/target_specificity_corr.png` — correlation matrix figure

## 7. Known Limitations

1. **7×7 resolution:** Swin-B produces 7×7 spatial feature maps for 224×224 input. Heatmaps are coarse.
2. **Gradient mixing in multi-image:** Gradients for image k are influenced by other images through pooling and fusion layers. This is by design (preserves model fidelity) but means attribution isn't purely isolated.
3. **No randomization test:** Not implemented due to memory constraints. Can be added separately.
4. **Single-batch processing:** B=1 for all Grad-CAM computations (required for per-image hook isolation).

## 8. Grad-CAM Target Similarity Diagnosis

### Issue
Grad-CAM heatmaps for all 5 targets appear nearly identical for many samples.

### Root Cause
**Expected behavior, not a bug.** The 5 targets share the entire image encoder (Swin-B, 88M params), cross-attention fusion, and 3 linear layers in the prediction head. Only the final `Linear(256→5)` layer differs per target — one row per target. By the time the gradient from that single row propagates backward through `256→512→1024→image_proj→cross_attention→encoder`, the per-target signal is heavily diluted through shared weights.

### Evidence
- Gradient cosine similarity at `encoder.norm` is typically >0.95 between all target pairs
- Raw CAM Pearson correlation is typically >0.90
- This is consistent with shared-backbone multi-target regression architectures

### Diagnostics Added
- `diagnose_target_gradients()` function in `gradcam_explainer.py`
- Gradient statistics per target (mean, std, abs_max)
- 5×5 gradient cosine similarity matrix
- 5×5 raw CAM Pearson correlation matrix
- Automated interpretation messages in notebook

### Recommendation
- Grad-CAM shows WHERE the image branch looks (shared visual evidence)
- For target-SPECIFIC modality analysis, use SHAP (Phase 4) on the fused embedding
- Document this finding in the thesis as a structural property of shared-encoder architectures

---

## 9. How Phase 3 Can Reuse Phase 2 Results

Phase 3 (Attention Visualization) does not directly depend on Grad-CAM outputs, but:
- Phase 6 (Case Studies) will load both Grad-CAM overlays and attention heatmaps for combined figures
- The `raw_cams.npz` files can be loaded by Phase 6 for quantitative cross-method comparison
- The `metadata.json` format is consistent with Phase 3's planned metadata format
- The notebook structure (steps, logging, validation) is identical, making Phase 3 easy to implement
