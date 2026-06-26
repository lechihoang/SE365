# Phase 2: Grad-CAM for Image Branch -- Implementation Proposal

---

## 1. Purpose

### Research Motivation

Grad-CAM (Gradient-weighted Class Activation Mapping) answers the fundamental question: **"Where in the image did the model look when predicting each target score?"** This is the single most important visual explanation tool for the image branch of the multimodal restaurant review quality assessment system.

The best-performing model (Swin-B + PhoBERT + CrossAttentionFusion + LogCosh, Val Mean MAE 1.1079) processes up to 4 restaurant/food images per review through a Swin-B encoder, producing a 1024-dimensional pooled feature vector. Without Grad-CAM, it is impossible to determine whether the model's image branch is attending to semantically relevant regions (food presentation, restaurant interior, table setting) or exploiting spurious correlations (watermarks, lighting artifacts, image borders).

### Engineering Motivation

Grad-CAM is the most mature, well-understood, and library-supported spatial attribution method for deep neural networks. By implementing it as the second phase (immediately after Phase 1 Infrastructure), all subsequent phases (Attention, SHAP, LIME, Case Studies) can reference and compare against Grad-CAM heatmaps as a spatial baseline.

### Thesis Motivation

A thesis defense requires concrete visual evidence that the image branch contributes meaningful information to each of the 5 target scores. Grad-CAM produces publication-ready heatmap overlays that directly answer examiner questions such as:

- "Why did the model assign a low food_score?"
- "Does the model actually use restaurant interior information for atmosphere_score?"
- "Is the model looking at the food or the background?"

Phase 2 produces the primary figures for the image-branch results section and is referenced by every subsequent XAI phase.

---

## 2. Objectives

### Research Objectives

1. **RO-1:** Generate target-specific spatial attribution maps for each of the 5 regression targets (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`), demonstrating that different targets activate different image regions.

2. **RO-2:** Produce per-image Grad-CAM explanations for multi-image reviews, showing which spatial regions within each individual review image contribute to each target score.

3. **RO-3:** Validate that Grad-CAM explanations are target-specific (different targets produce different heatmaps for the same image) and model-dependent (randomized weights produce degraded heatmaps).

### Engineering Objectives

1. **EO-1:** Create a reusable `GradCAMExplainer` module that wraps the full multimodal model and generates Grad-CAM heatmaps for any target score index and any individual image within a multi-image review.

2. **EO-2:** Implement wrapper classes (`MultiTargetScoreWrapper`, `SingleImageGradCAMWrapper`) that correctly handle the CrossAttentionFusion forward pass, fixed text inputs, and per-image gradient isolation.

3. **EO-3:** Produce all output artifacts (heatmap PNGs, raw activation NPYs, comparison figures, metadata JSONs) in the standardized folder structure defined by Phase 1 Infrastructure.

4. **EO-4:** Create a self-contained Jupyter notebook (`Phase2_GradCAM.ipynb`) that demonstrates the complete Grad-CAM pipeline from checkpoint loading to publication-ready figures.

### Expected Contributions

- First visual evidence that the image branch of the multimodal system encodes target-specific spatial information.
- Per-image attribution maps that reveal whether multi-image aggregation dilutes or preserves spatial evidence.
- Failure case identification (model shortcuts, background attention, watermark sensitivity).
- Reusable infrastructure for Phase 6 (Case Studies) and Phase 8 (Thesis Visualization).

---

## 3. Inputs

### Checkpoint

| Item | Path | Description |
|------|------|-------------|
| Best model checkpoint | `experiments/EXP_XXX/best_model_train_fusion.pth` | Contains `model_state_dict` for CrossAttentionFusion with Swin-B + PhoBERT |
| Checkpoint metadata | `experiments/EXP_XXX/config.json` or command-line args | Records `image_model_name`, `text_model_name`, `fusion_type`, etc. |

### Data

| Item | Path | Description |
|------|------|-------------|
| Test CSV | `data/text/test.csv` | Contains `comment_clean`, `image_url`, and 5 target scores |
| Validation CSV | `data/text/val.csv` | Alternative source for sample selection |
| Image directory | `data/image/` | Pre-downloaded images stored as `{md5_hash}.jpg` |

### Phase 1 Infrastructure

| Item | Path | Description |
|------|------|-------------|
| `load_model()` | `xai/utils.py` | Loads CrossAttentionFusion from checkpoint |
| `load_single_sample()` | `xai/utils.py` | Loads one sample with all preprocessing |
| `get_prediction()` | `xai/utils.py` | Runs forward pass and returns predictions |
| `save_figure()` | `xai/utils.py` | Saves matplotlib figures with consistent DPI and formatting |
| `save_raw_values()` | `xai/utils.py` | Saves numpy arrays with consistent naming |
| `TARGET_NAMES` | `xai/config.py` | `['food_score', 'price_score', 'atmosphere_score', 'service_score', 'overall_satisfaction']` |
| `TARGET_INDICES` | `xai/config.py` | `{name: index}` mapping |

### External Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pytorch-grad-cam` | >= 1.4.8 | GradCAM implementation, `show_cam_on_image` utility |
| `timm` | (existing) | Swin-B model loading |
| `torch` | (existing) | Gradient computation |
| `matplotlib` | (existing) | Figure generation |
| `numpy` | (existing) | Raw activation storage |
| `PIL` / `Pillow` | (existing) | Image loading and resizing |

---

## 4. Outputs

### Per Sample, Per Target

| Artifact | Format | Naming | Description |
|----------|--------|--------|-------------|
| Heatmap overlay | PNG | `sample_{id}_target{idx}_{name}.png` | Grad-CAM heatmap overlaid on original image (single-image reviews) |
| Per-image heatmap overlay | PNG | `sample_{id}_target{idx}_{name}_img{k}.png` | Grad-CAM heatmap for image `k` in multi-image reviews |
| Raw activation map | NPY | `raw/sample_{id}_target{idx}_cam.npy` | Raw Grad-CAM activation values, shape `[7, 7]` |
| Per-image raw activation | NPY | `raw/sample_{id}_target{idx}_img{k}_cam.npy` | Raw activation for image `k`, shape `[7, 7]` |

### Per Sample (Cross-Target)

| Artifact | Format | Naming | Description |
|----------|--------|--------|-------------|
| 5-target comparison figure | PNG | `sample_{id}_comparison_5targets.png` | Single figure showing all 5 target heatmaps side by side for visual comparison |
| Multi-image 5-target grid | PNG | `sample_{id}_comparison_5targets_all_images.png` | Grid: rows = images, columns = targets (for multi-image reviews) |
| Metadata | JSON | `metadata/sample_{id}_gradcam_metadata.json` | Records predictions, true labels, sample info, generation parameters |

### Aggregate

| Artifact | Format | Naming | Description |
|----------|--------|--------|-------------|
| Batch summary | JSON | `metadata/gradcam_batch_summary.json` | Lists all processed samples, their paths, timing, and status |

---

## 5. Architecture Attachment Point

### Where Grad-CAM Hooks Into the Model

Grad-CAM requires two things from a neural network:
1. **Forward hook:** Captures the spatial feature map (activations) at a specific layer.
2. **Backward hook:** Captures the gradients of the target output with respect to those activations.

In the current architecture, the attachment point is inside the Swin-B image encoder, specifically **before global average pooling destroys spatial information**.

### Full Model Data Flow

```
Input:
  input_ids [B, L]           --> TextModel.encoder (PhoBERT) --> text_feat [B, 768]
  attention_mask [B, L]      /
  pixel_values [B, N, 3, 224, 224] --> ImageModel.encoder (Swin-B) --> [B*N, 1024] --> reshape --> [B, N, 1024]
                                                                                     --> masked avg pool --> image_feat [B, 1024]

CrossAttentionFusion:
  text_proj(text_feat)  --> t [B, 1, 512]
  image_proj(image_feat) --> i [B, 1, 512]
  cross_attn_t2i(q=t, k=i, v=i) --> t_out [B, 1, 512]
  cross_attn_i2t(q=i, k=t, v=t) --> i_out [B, 1, 512]
  cat(t_out, i_out) --> fused [B, 1024]
  head(fused) --> preds [B, 5]
```

### The Critical Layer: `model.image_model.encoder.norm`

Inside Swin-B (`swin_base_patch4_window7_224` via timm), the forward pass proceeds as:

```
Input [B, 3, 224, 224]
  --> patch_embed --> [B, 56*56, 128]
  --> layers[0] (Stage 1) --> [B, 28*28, 256]
  --> layers[1] (Stage 2) --> [B, 14*14, 512]
  --> layers[2] (Stage 3) --> [B, 7*7, 1024]
  --> layers[3] (Stage 4, last stage) --> [B, 7*7, 1024]   <-- spatial features preserved
  --> norm (LayerNorm) --> [B, 49, 1024]                    <-- GRAD-CAM TARGET LAYER
  --> global_pool (mean over spatial dim) --> [B, 1024]     <-- spatial info destroyed
```

The target layer for Grad-CAM is **`model.image_model.encoder.norm`** (the final LayerNorm applied after the last stage but before global average pooling). This layer:

- Preserves full spatial resolution: `[B, 49, 1024]` (equivalently `[B, 7, 7, 1024]`)
- Contains the richest, most semantically meaningful features (deepest layer)
- Is the last point where spatial information exists before pooling collapses it to `[B, 1024]`

### Wrapper Design Principle

Grad-CAM via `pytorch-grad-cam` expects a model that takes an image tensor and produces a scalar or vector output. The multimodal model takes text + image. The solution is a **wrapper** that:

1. **Fixes text inputs:** `input_ids` and `attention_mask` are stored as buffers in the wrapper. They do not vary during Grad-CAM computation.
2. **Accepts only pixel_values:** The wrapper's `forward(pixel_values)` method calls the full multimodal model with the fixed text + variable image.
3. **Returns a specific target score:** The wrapper selects `preds[:, score_index]` to produce the scalar that Grad-CAM backpropagates from.

This ensures that gradients flow from the selected target score, through the fusion pipeline, through the image projection, back to the Swin-B encoder's spatial feature maps -- while the text branch provides its contextual contribution but receives no gradient updates (its inputs are fixed).

### Multi-Image Design Principle

The model processes up to 4 images per review via `ImageModel.forward()`:
1. Reshape `[B, N, C, H, W]` to `[B*N, C, H, W]`
2. Pass all `B*N` images through the encoder
3. Reshape back to `[B, N, features_dim]`
4. Apply masked average pooling to get `[B, features_dim]`

For Grad-CAM, we need per-image spatial attribution. The wrapper must:
1. Process ALL images normally through the full model (preserving the correct multi-image context)
2. Register a hook on the encoder to capture the spatial feature map for a SPECIFIC image index
3. Backpropagate from the target score
4. Collect gradients only for that specific image's feature map

This is detailed fully in Section 6 and Section 12 (Risk R1).

---

## 6. Detailed Implementation Plan

### Step 1: Create `xai/gradcam_explainer.py`

This is the primary module for Phase 2. It contains all Grad-CAM logic.

**Responsibility:** Provide a high-level API for generating Grad-CAM explanations for any sample, any target, and any individual image in a multi-image review.

### Step 2: Implement `RegressionScoreTarget` Class

`pytorch-grad-cam` provides `ClassifierOutputTarget` which expects a class index and calls `output[class_index]` on the model output. For regression, the model outputs a scalar (from the wrapper), and the target should simply return that scalar.

**Design:**
- Inherits from or follows the interface expected by `pytorch-grad-cam`'s `targets` parameter
- `__call__(model_output)` returns `model_output[0, score_index]` or simply the scalar output
- Since the wrapper already selects the target score and returns `[B]`, the target can be `None` (pytorch-grad-cam uses the full output when targets=None) OR a simple lambda/class that returns the scalar

**Decision:** Use `targets=None` if the wrapper returns shape `[B, 1]`, or implement a minimal `RegressionScoreTarget` that returns `model_output.squeeze()`. The wrapper must ensure pytorch-grad-cam receives the correct scalar to backpropagate.

### Step 3: Implement `MultiTargetScoreWrapper` Class

**Purpose:** Wraps the full CrossAttentionFusion model so that `pytorch-grad-cam` sees a standard `nn.Module` with `forward(pixel_values) -> score`.

**Interface:**
```
class MultiTargetScoreWrapper(nn.Module):
    __init__(self, multimodal_model, fixed_input_ids, fixed_attention_mask, score_index, fixed_num_images=None)
    forward(self, pixel_values) -> Tensor [B, 1]
```

**Behavior:**
1. Store `fixed_input_ids`, `fixed_attention_mask`, `fixed_num_images` as registered buffers (not parameters, to avoid optimizer inclusion).
2. In `forward()`:
   - Expand fixed text inputs to match batch size of `pixel_values` if needed (for batch Grad-CAM, though typically B=1 for XAI).
   - Call `self.multimodal_model(input_ids=..., attention_mask=..., pixel_values=pixel_values, num_images=...)`.
   - Select `preds[:, self.score_index].unsqueeze(-1)` to return `[B, 1]`.
3. The model must be in `eval()` mode.
4. All parameters except those needed for gradient computation should be handled appropriately. Note: `pytorch-grad-cam` manages gradient enabling internally.

### Step 4: Implement `SingleImageGradCAMWrapper` Class

**Purpose:** Enables per-image Grad-CAM for multi-image reviews. Instead of explaining the pooled feature from all images, this wrapper isolates the spatial feature map of ONE specific image.

**Interface:**
```
class SingleImageGradCAMWrapper(nn.Module):
    __init__(self, multimodal_model, fixed_input_ids, fixed_attention_mask, score_index, image_index, num_images, all_pixel_values)
    forward(self, single_image) -> Tensor [B, 1]
```

**Behavior:**
1. Store `all_pixel_values` (all images for the review), `image_index` (which image to explain), and other fixed inputs.
2. In `forward(single_image)`:
   - Create a copy of `all_pixel_values`.
   - Replace the image at `image_index` with `single_image` (the input to this forward call).
   - Call the full multimodal model with the reconstructed `pixel_values` tensor.
   - Return the selected target score.
3. Grad-CAM hooks are registered on the encoder layers. Since `single_image` flows through the encoder at position `image_index` within the `B*N` batch, gradients will flow specifically through the spatial features of that image.

**Alternative approach (simpler, recommended):**

Instead of the complex wrapper above, use a **hook-based approach**:
1. Use `MultiTargetScoreWrapper` as-is (processes all images normally).
2. Before calling pytorch-grad-cam, register a custom forward hook on `model.image_model.encoder.norm` that:
   - Captures the full output `[B*N, 49, 1024]`
   - Extracts only the slice for the target image index: `activations[image_index]` -> `[49, 1024]`
3. Similarly, register a backward hook that captures gradients only for the target image index.
4. After pytorch-grad-cam runs, compute the Grad-CAM map manually from the captured per-image activations and gradients.

**Final Decision:** Use Strategy B from Risk R1 -- the hook-based approach. Process all images through the full model to preserve multi-image context, but capture activations and gradients for a specific image index via hooks. This is detailed in Risk R1.

### Step 5: Implement `SwinTransformerReshapeTransform` Class

**Purpose:** `pytorch-grad-cam` expects feature maps in channels-first format `[B, C, H, W]`. Swin-B's norm layer outputs `[B, 49, 1024]` (or `[B, 7, 7, 1024]`), which is channels-last.

**Interface:**
```
class SwinTransformerReshapeTransform:
    __call__(self, tensor) -> Tensor [B, C, H, W]
```

**Behavior:**
1. If input shape is `[B, tokens, channels]` (e.g., `[B, 49, 1024]`):
   - Compute spatial dimensions: `H = W = int(sqrt(tokens))` = 7
   - Reshape to `[B, H, W, C]`
   - Permute to `[B, C, H, W]` -> `[B, 1024, 7, 7]`
2. If input shape is `[B, H, W, C]`:
   - Permute to `[B, C, H, W]`
3. This transform is passed to `GradCAM(model=..., target_layers=[...], reshape_transform=reshape_fn)`.

### Step 6: Identify and Validate Target Layer

**Action:** Programmatically identify the correct target layer for Grad-CAM.

**Procedure:**
1. Load the model via Phase 1's `load_model()`.
2. Access the target layer: `target_layer = model.image_model.encoder.norm`
3. Verify by running a dummy forward pass and inspecting the hook output shape.
4. Confirm it is `[B*N, 49, 1024]` or `[B*N, 7, 7, 1024]` depending on timm version.
5. If `encoder.norm` does not produce spatial output (some timm versions may apply norm after pooling), fall back to `model.image_model.encoder.layers[-1].blocks[-1].norm2`.

**Validation:**
- Register a temporary forward hook on the candidate layer.
- Run a forward pass with a dummy input of shape `[1, 4, 3, 224, 224]`.
- Verify the hook captures a tensor with a spatial dimension that equals 49 (7x7) or has H=W=7.

### Step 7: Implement `GradCAMExplainer` Class

**Purpose:** High-level orchestrator that generates all Grad-CAM artifacts for a given sample.

**Interface:**
```
class GradCAMExplainer:
    __init__(self, model, device, output_dir)
    explain_sample(self, sample, sample_id) -> dict
    explain_single_target(self, sample, sample_id, target_idx) -> dict
    explain_all_targets(self, sample, sample_id) -> dict
    create_comparison_figure(self, sample_id, heatmaps, original_images) -> path
```

**`explain_sample` workflow:**
1. Load sample using Phase 1's `load_single_sample()`.
2. Determine `num_images` (how many real images in this review).
3. Get model predictions using Phase 1's `get_prediction()`.
4. For each target index (0-4):
   a. For each real image (0 to num_images-1):
      - Create or configure the appropriate wrapper.
      - Run Grad-CAM to get the activation map `[7, 7]`.
      - Overlay on the original image to create heatmap PNG.
      - Save raw activation as NPY.
   b. If single-image review, save directly as `sample_{id}_target{idx}_{name}.png`.
   c. If multi-image review, save as `sample_{id}_target{idx}_{name}_img{k}.png`.
5. Create 5-target comparison figure.
6. Save metadata JSON.
7. Return dict of all artifact paths and key values.

### Step 8: Implement Heatmap Overlay Generation

**Procedure:**
1. Obtain raw Grad-CAM map: shape `[7, 7]`, values in `[0, 1]` (normalized by pytorch-grad-cam).
2. Load original image as RGB array, resize to 224x224, normalize to `[0, 1]` float range.
3. Call `show_cam_on_image(rgb_image_float01, grayscale_cam, use_rgb=True)` from `pytorch-grad-cam`.
4. The result is a `[224, 224, 3]` uint8 array with the heatmap overlay.
5. Save using PIL or matplotlib with consistent DPI (150 or 300 for thesis quality).

**Critical detail:** The `rgb_image_float01` must be the ORIGINAL image normalized to `[0, 1]`, NOT the preprocessed tensor (which has channel normalization applied). Load the original PIL image, resize to 224x224, convert to numpy float32 / 255.

### Step 9: Implement 5-Target Comparison Figure

**Design:**
- For single-image reviews: 1 row, 6 columns (original + 5 target heatmaps).
- For multi-image reviews: `num_images` rows, 6 columns (original_k + 5 target heatmaps for image k).
- Column titles: `"Original"`, `"food_score"`, `"price_score"`, `"atmosphere_score"`, `"service_score"`, `"overall_satisfaction"`.
- Row titles (multi-image): `"Image 1"`, `"Image 2"`, etc.
- Figure title: `"Sample {id} -- Grad-CAM Target Comparison"`.
- Include predicted and true scores as subtitle text.
- Use `matplotlib.pyplot.subplots()` with `figsize` scaled to number of rows.
- Save as PNG at 300 DPI for thesis quality.

### Step 10: Implement Metadata JSON Generation

**Contents of `sample_{id}_gradcam_metadata.json`:**

```json
{
  "sample_id": 42,
  "num_images": 3,
  "predictions": {
    "food_score": 3.45,
    "price_score": 2.89,
    "atmosphere_score": 4.12,
    "service_score": 3.67,
    "overall_satisfaction": 3.78
  },
  "true_labels": {
    "food_score": 4.0,
    "price_score": 3.0,
    "atmosphere_score": 4.0,
    "service_score": 4.0,
    "overall_satisfaction": 4.0
  },
  "target_layer": "image_model.encoder.norm",
  "cam_spatial_shape": [7, 7],
  "feature_channels": 1024,
  "timestamp": "2026-06-26T14:30:00",
  "model_checkpoint": "experiments/EXP_XXX/best_model_train_fusion.pth",
  "artifacts": {
    "heatmaps": ["sample_42_target0_food_score_img0.png", "..."],
    "raw_activations": ["raw/sample_42_target0_img0_cam.npy", "..."],
    "comparison_figure": "sample_42_comparison_5targets.png"
  }
}
```

### Step 11: Implement Raw Activation Saving

**Procedure:**
1. After pytorch-grad-cam generates `grayscale_cam` (shape `[1, H, W]` or `[H, W]`):
   - Save the raw `[7, 7]` activation map BEFORE upsampling as `.npy`.
   - This preserves the exact Grad-CAM activation values for later quantitative analysis.
2. Use `numpy.save()` with the standardized naming convention.
3. These raw files are used in Phase 6 (Case Studies) for quantitative comparison and in Phase 8 (Thesis Visualization) for custom plots.

### Step 12: Implement Batch Processing

**Purpose:** Process multiple samples efficiently.

**Procedure:**
1. Accept a list of sample IDs (or indices into test/val CSV).
2. For each sample:
   a. Load sample.
   b. Call `explain_sample()`.
   c. Log progress (sample ID, time taken, number of images).
   d. Handle errors gracefully (skip sample, log error, continue).
3. After all samples: generate `gradcam_batch_summary.json`.

**Memory management:**
- Process one sample at a time (B=1 for Grad-CAM).
- Call `torch.cuda.empty_cache()` between samples if on GPU.
- Delete intermediate tensors explicitly.
- The Grad-CAM context manager (`with GradCAM(...) as cam:`) handles hook cleanup automatically.

### Step 13: Create Notebook `xai/notebooks/Phase2_GradCAM.ipynb`

Detailed cell-by-cell design in Section 9.

---

## 7. Required Code Files

### New Files

| File | Responsibility |
|------|----------------|
| `xai/gradcam_explainer.py` | Core module: `GradCAMExplainer`, `MultiTargetScoreWrapper`, `SingleImageGradCAMWrapper`, `SwinTransformerReshapeTransform`, `RegressionScoreTarget`, all heatmap generation and saving logic |
| `xai/notebooks/Phase2_GradCAM.ipynb` | Interactive notebook demonstrating the full Grad-CAM pipeline with visualizations |

### Modified Files

| File | Modification |
|------|-------------|
| `xai/config.py` | Add Grad-CAM-specific constants: `GRADCAM_TARGET_LAYER_NAME`, `GRADCAM_SPATIAL_SIZE`, `GRADCAM_OUTPUT_SUBDIR`, `GRADCAM_DPI` |
| `xai/utils.py` | Add helper `get_original_image_rgb(sample, image_index)` that returns the original image as `[H, W, 3]` float32 in `[0, 1]` range for heatmap overlay |

### Dependencies on Phase 1 Files

| File | Functions Used |
|------|----------------|
| `xai/utils.py` | `load_model()`, `load_single_sample()`, `get_prediction()`, `save_figure()`, `save_raw_values()` |
| `xai/config.py` | `TARGET_NAMES`, `TARGET_INDICES`, `FACTOR_NAMES`, `EXP_DIR`, `DEVICE` |

---

## 8. Folder Structure

```
experiments/EXP_XXX/xai/gradcam/
├── sample_{id}_target0_food_score.png
├── sample_{id}_target1_price_score.png
├── sample_{id}_target2_atmosphere_score.png
├── sample_{id}_target3_service_score.png
├── sample_{id}_target4_overall_satisfaction.png
├── sample_{id}_target0_food_score_img0.png          # multi-image: per-image heatmap
├── sample_{id}_target0_food_score_img1.png
├── sample_{id}_target0_food_score_img2.png
├── ...
├── sample_{id}_comparison_5targets.png               # all 5 targets side by side
├── sample_{id}_comparison_5targets_all_images.png     # grid for multi-image reviews
├── raw/
│   ├── sample_{id}_target0_cam.npy                   # raw [7,7] activation (single-image)
│   ├── sample_{id}_target0_img0_cam.npy              # raw [7,7] activation (per-image)
│   ├── sample_{id}_target0_img1_cam.npy
│   ├── sample_{id}_target1_cam.npy
│   └── ...
└── metadata/
    ├── sample_{id}_gradcam_metadata.json             # per-sample metadata
    └── gradcam_batch_summary.json                     # batch processing summary
```

### Naming Convention Rules

- `{id}` is the integer index of the sample in the test/val dataset (zero-padded to 4 digits: `0042`).
- `{idx}` is the target index (0-4).
- `{name}` is the target name (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`).
- `{k}` is the image index within the review (0-indexed).
- Single-image reviews use `sample_{id}_target{idx}_{name}.png` (no `_img{k}` suffix).
- Multi-image reviews use `sample_{id}_target{idx}_{name}_img{k}.png`.
- All paths are relative to `experiments/EXP_XXX/xai/gradcam/`.

---

## 9. Notebook Design (Cell by Cell)

### Cell 1: Setup and Imports

**Purpose:** Import all dependencies, configure paths, set device.

**Contents:**
- Import `torch`, `numpy`, `matplotlib.pyplot`, `PIL.Image`, `os`, `json`.
- Import `pytorch_grad_cam` classes: `GradCAM`, `show_cam_on_image`.
- Import Phase 1 utilities: `load_model`, `load_single_sample`, `get_prediction`.
- Import Phase 2 module: `GradCAMExplainer`, wrapper classes.
- Import Phase 1 config: `TARGET_NAMES`, `TARGET_INDICES`, `EXP_DIR`.
- Set `device`, `matplotlib` inline mode, random seed.
- Print library versions for reproducibility.

**Expected output:** Version printout, device confirmation.

### Cell 2: Load Model

**Purpose:** Load the best-performing CrossAttentionFusion model from checkpoint.

**Contents:**
- Call `load_model()` from Phase 1.
- Set model to `eval()` mode.
- Print model architecture summary (number of parameters, encoder types).

**Expected output:** Model loaded confirmation, parameter count.

### Cell 3: Load a Single Sample

**Purpose:** Load one test sample for demonstration.

**Contents:**
- Select a sample ID (choose one with multiple images and interesting predictions).
- Call `load_single_sample(sample_id)`.
- Display the original image(s) using matplotlib.
- Print the text review (comment_clean).
- Print true labels and model predictions for all 5 targets.
- Print `num_images` count.

**Expected output:** Image display, text printout, prediction table.

### Cell 4: Identify Target Layer

**Purpose:** Verify the Grad-CAM target layer programmatically.

**Contents:**
- Access `model.image_model.encoder.norm`.
- Register a temporary forward hook.
- Run a dummy forward pass.
- Print the captured tensor shape (should be `[B*N, 49, 1024]` or similar).
- Confirm spatial dimensions are 7x7.
- Remove the temporary hook.

**Expected output:** Shape printout, confirmation message.

### Cell 5: Create Wrapper and Generate Single Heatmap

**Purpose:** Demonstrate the basic Grad-CAM pipeline for one target, one image.

**Contents:**
- Create `MultiTargetScoreWrapper` for `food_score` (index 0).
- Instantiate `SwinTransformerReshapeTransform`.
- Identify `target_layer` in the wrapper's model hierarchy.
- Create `GradCAM` object with `target_layers=[target_layer]` and `reshape_transform=reshape_fn`.
- Run `cam(input_tensor=..., targets=None)`.
- Get `grayscale_cam` of shape `[1, 7, 7]` -> `[7, 7]`.
- Load original image as RGB float [0,1].
- Call `show_cam_on_image()`.
- Display: original image, raw heatmap, overlay.

**Expected output:** Three-panel matplotlib figure.

### Cell 6: Generate All 5 Targets for One Image

**Purpose:** Show target specificity by generating heatmaps for all 5 targets on the same image.

**Contents:**
- Loop over target indices 0-4.
- For each target, create wrapper, run Grad-CAM, generate overlay.
- Display all 5 heatmaps in a single row with target names as titles.
- Add original image as the first panel.

**Expected output:** 6-panel figure (original + 5 target heatmaps). Visual evidence that different targets produce different heatmaps.

### Cell 7: Per-Image Grad-CAM (Multi-Image Review)

**Purpose:** Demonstrate per-image Grad-CAM for a multi-image review sample.

**Contents:**
- Select a sample with num_images >= 2.
- For one target (e.g., `food_score`):
  - Loop over each real image (0 to num_images-1).
  - Run per-image Grad-CAM using the hook-based approach.
  - Generate overlay for each image.
- Display: one row per image, showing original and heatmap.

**Expected output:** Multi-row figure showing per-image heatmaps.

### Cell 8: Full Sample Explanation Using GradCAMExplainer

**Purpose:** Use the high-level `GradCAMExplainer` to generate all artifacts.

**Contents:**
- Instantiate `GradCAMExplainer(model, device, output_dir)`.
- Call `explainer.explain_sample(sample, sample_id)`.
- Print the returned artifact dict.
- Display the generated comparison figure.

**Expected output:** Artifact paths printed, comparison figure displayed.

### Cell 9: Batch Processing (Multiple Samples)

**Purpose:** Process several samples to demonstrate batch capability.

**Contents:**
- Select 5-10 diverse sample IDs (covering different num_images counts, different score ranges).
- Loop over sample IDs, call `explainer.explain_sample()` for each.
- Print timing per sample.
- Load and display the batch summary JSON.

**Expected output:** Progress log, timing summary.

### Cell 10: Sanity Check -- Target Specificity

**Purpose:** Validate that different targets produce measurably different heatmaps.

**Contents:**
- For one sample, load all 5 raw Grad-CAM NPY files.
- Compute pairwise Pearson correlation between the 5 heatmaps (flatten to 49-d vectors).
- Display correlation matrix as a heatmap.
- Interpret: low correlation between food_score and atmosphere_score heatmaps = good target specificity; high correlation everywhere = potential head collapse.

**Expected output:** 5x5 correlation matrix heatmap.

### Cell 11: Sanity Check -- Randomized Model

**Purpose:** Verify that Grad-CAM explanations depend on trained weights.

**Contents:**
- Create a randomized copy of the model (reinitialize weights).
- Run Grad-CAM on the same sample, same target.
- Display: trained model heatmap vs. randomized model heatmap side by side.
- Compute correlation between the two.

**Expected output:** Visual comparison showing degraded/different heatmaps from randomized model.

### Cell 12: Discussion and Interpretation

**Purpose:** Markdown cell with qualitative interpretation guidelines.

**Contents:**
- How to interpret heatmaps for each target (food = food presentation, atmosphere = interior, etc.).
- What diffuse/uniform heatmaps mean.
- What background-focused heatmaps mean (potential shortcuts).
- Limitations of 7x7 resolution.
- Connection to thesis sections.

**Expected output:** Rendered markdown.

---

## 10. Algorithm

### Pseudo-Workflow: Single Image, Single Target

```
FUNCTION generate_gradcam(model, sample, target_idx):
    SET model to eval mode
    EXTRACT input_ids, attention_mask, pixel_values, num_images from sample
    
    CREATE wrapper = MultiTargetScoreWrapper(
        model, input_ids, attention_mask, target_idx, num_images
    )
    
    SET target_layer = wrapper.multimodal_model.image_model.encoder.norm
    SET reshape_fn = SwinTransformerReshapeTransform()
    
    CREATE cam = GradCAM(
        model=wrapper,
        target_layers=[target_layer],
        reshape_transform=reshape_fn
    )
    
    SET grayscale_cam = cam(input_tensor=pixel_values, targets=None)
    # grayscale_cam shape: [B, H_cam, W_cam] = [1, 7, 7]
    SET grayscale_cam = grayscale_cam[0]  # [7, 7]
    
    LOAD original_image as RGB float in [0, 1], resized to 224x224
    SET overlay = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    
    SAVE overlay as PNG
    SAVE grayscale_cam as NPY
    
    RETURN grayscale_cam, overlay
```

### Pseudo-Workflow: Per-Image Grad-CAM (Multi-Image Review)

```
FUNCTION generate_per_image_gradcam(model, sample, target_idx, image_index):
    SET model to eval mode
    EXTRACT input_ids, attention_mask, pixel_values, num_images from sample
    # pixel_values shape: [1, N, 3, 224, 224]
    
    ASSERT image_index < num_images  # never explain padding images
    
    # Strategy: Use hook-based isolation
    # The encoder processes B*N images. Image at index `image_index`
    # corresponds to index `batch_idx * N + image_index` in the B*N batch.
    
    SET activation_store = {}
    SET gradient_store = {}
    
    DEFINE forward_hook(module, input, output):
        # output shape: [B*N, 49, 1024] or [B*N, 7, 7, 1024]
        activation_store['value'] = output
    
    DEFINE backward_hook(module, grad_input, grad_output):
        gradient_store['value'] = grad_output[0]
    
    REGISTER forward_hook on model.image_model.encoder.norm
    REGISTER backward_hook on model.image_model.encoder.norm
    
    # Full forward pass
    SET preds = model(input_ids, attention_mask, pixel_values, num_images)
    SET target_score = preds[0, target_idx]
    
    # Backward pass
    model.zero_grad()
    target_score.backward(retain_graph=True)
    
    # Extract per-image activations and gradients
    SET full_activations = activation_store['value']   # [B*N, 49, 1024]
    SET full_gradients = gradient_store['value']        # [B*N, 49, 1024]
    
    SET img_activations = full_activations[image_index]  # [49, 1024]
    SET img_gradients = full_gradients[image_index]      # [49, 1024]
    
    # Compute Grad-CAM manually
    SET weights = img_gradients.mean(dim=0)  # [1024] -- GAP over spatial dim
    SET cam = (img_activations * weights).sum(dim=-1)  # [49]
    SET cam = ReLU(cam)
    SET cam = cam.reshape(7, 7)
    SET cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize to [0,1]
    
    REMOVE hooks
    
    LOAD original_image[image_index] as RGB float [0,1]
    SET overlay = show_cam_on_image(original_image, cam.numpy(), use_rgb=True)
    
    SAVE overlay as PNG
    SAVE cam as NPY
    
    RETURN cam, overlay
```

### Pseudo-Workflow: Full Sample Explanation

```
FUNCTION explain_sample(model, sample, sample_id):
    SET results = {}
    
    EXTRACT num_images from sample
    GET predictions = model forward pass
    GET true_labels from sample
    
    FOR target_idx in [0, 1, 2, 3, 4]:
        SET target_name = TARGET_NAMES[target_idx]
        
        IF num_images == 1:
            cam, overlay = generate_gradcam(model, sample, target_idx)
            SAVE overlay as sample_{id}_target{idx}_{name}.png
            SAVE cam as raw/sample_{id}_target{idx}_cam.npy
        ELSE:
            FOR img_idx in range(num_images):
                cam, overlay = generate_per_image_gradcam(
                    model, sample, target_idx, img_idx
                )
                SAVE overlay as sample_{id}_target{idx}_{name}_img{k}.png
                SAVE cam as raw/sample_{id}_target{idx}_img{k}_cam.npy
        
        STORE results for this target
    
    CREATE comparison figure (5 targets side by side)
    SAVE as sample_{id}_comparison_5targets.png
    
    IF num_images > 1:
        CREATE multi-image comparison grid
        SAVE as sample_{id}_comparison_5targets_all_images.png
    
    SAVE metadata JSON
    
    RETURN results
```

---

## 11. Validation

### V1: Sanity Check -- Target Specificity

**Test:** Generate Grad-CAM heatmaps for all 5 targets on the same image. Compute pairwise Pearson correlation between the 5 flattened heatmaps.

**Expected result:** Correlation should NOT be 1.0 for all pairs. Different targets should produce at least some different spatial patterns. If all 5 heatmaps are identical (correlation ~ 1.0 for all pairs), this indicates head collapse or insufficient target differentiation in the image branch.

**Pass criterion:** At least one pair of targets has Pearson r < 0.9 for a meaningful sample.

### V2: Sanity Check -- Model Dependency (Randomization Test)

**Test:** Reinitialize the model with random weights (keeping architecture identical). Generate Grad-CAM on the same sample.

**Expected result:** The heatmap from the randomized model should be qualitatively different from the trained model. It should look more uniform or random.

**Pass criterion:** Pearson correlation between trained and randomized heatmaps is below 0.7 for at least 80% of test samples.

### V3: Qualitative Check -- Semantic Alignment

**Test:** For samples with clear visual content (well-lit food photos, visible restaurant interiors), visually inspect whether:
- `food_score` heatmaps highlight food areas
- `atmosphere_score` heatmaps highlight interior/ambiance areas
- `price_score` heatmaps are diffuse (price is typically text-driven)

**Pass criterion:** At least 50% of inspected samples show semantically plausible heatmaps for `food_score` and `atmosphere_score`.

### V4: Technical Check -- Shape Consistency

**Test:** Verify all raw CAM files have shape `[7, 7]`, all overlay PNGs have size 224x224 (or original image size), all values in raw CAM are in `[0, 1]`.

**Pass criterion:** 100% of generated files pass shape and value range checks.

### V5: Reproducibility Check

**Test:** Run Grad-CAM twice on the same sample with the same checkpoint. Compare outputs.

**Expected result:** Identical results (model is in eval mode, no stochastic components).

**Pass criterion:** Raw CAM arrays are identical (numpy allclose with atol=1e-7).

### V6: Technical Check -- Gradient Flow

**Test:** Before running Grad-CAM, verify that the gradient of the target score with respect to the target layer's output is non-zero.

**Procedure:** Register a backward hook, backpropagate from the target score, check `gradient.abs().sum() > 0`.

**Pass criterion:** Non-zero gradient for all 5 targets, for all real images.

### V7: Multi-Image Consistency

**Test:** For a multi-image review, verify that:
1. Padding images (black images beyond `num_images`) are never processed by Grad-CAM.
2. Each real image produces a distinct heatmap.
3. The number of generated heatmap files equals `num_images * 5` (per target per image).

**Pass criterion:** File count matches expectation, no files generated for padding images.

---

## 12. Risks -- Fully Analyzed

### R1: Multi-Image Grad-CAM Strategy

**Problem:** The model processes up to 4 images per review via masked average pooling in `ImageModel.forward()`. Grad-CAM generates one heatmap per forward pass. How should we generate per-image spatial explanations?

**Why it happens:** The `ImageModel` reshapes `[B, N, C, H, W]` to `[B*N, C, H, W]`, passes all through the encoder, reshapes back to `[B, N, features_dim]`, then applies masked average pooling to get `[B, features_dim]`. Spatial information is lost at pooling. The pooled feature carries no per-image spatial identity.

#### Strategy A: Explain the Pooled Feature Only

- **Implementation:** Run standard Grad-CAM on the full model. The target layer captures `[B*N, 49, 1024]` feature maps for all images simultaneously. The pooling layer mixes them. The resulting Grad-CAM heatmap is a gradient-weighted average across all images.
- **Advantages:** Simple, uses model as-is, no wrapper complexity.
- **Disadvantages:** Cannot localize which image matters. The heatmap is an aggregate over all N images' spatial features, making it uninterpretable as a per-image explanation. Cannot produce meaningful overlays because the heatmap does not correspond to any single image's spatial layout.
- **Engineering trade-off:** Low implementation cost, but scientifically inadequate for a thesis.
- **Research trade-off:** Fails to answer "which image region in which photo matters?" -- a core thesis question for multi-image reviews.

#### Strategy B: Per-Image Grad-CAM via Hook Isolation

- **Implementation:** Process all images through the full model normally (preserving multi-image context and masked average pooling). Register hooks on the encoder's norm layer. After the forward and backward pass, extract activations and gradients for a SPECIFIC image index from the `[B*N, 49, 1024]` tensors. Compute Grad-CAM manually from the per-image slice.
- **Advantages:**
  - Preserves the full multi-image forward pass context (masked average pooling is computed correctly over all images).
  - Gradients reflect the true model behavior: how changing one image's features would affect the target score within the multi-image context.
  - Per-image spatial explanation is produced.
  - No model modification needed.
- **Disadvantages:**
  - Requires manual Grad-CAM computation (cannot use `pytorch-grad-cam`'s built-in pipeline directly for the per-image case).
  - Hook management adds complexity.
  - Gradients for one image are influenced by the other images through the pooling and fusion layers.
- **Engineering trade-off:** Moderate implementation cost. Hook-based approach is well-established in PyTorch.
- **Research trade-off:** The gradient for image k reflects how image k contributes to the target score WITHIN the context of all other images. This is the correct multimodal interpretation.

#### Strategy C: Explain Only the First/Primary Image

- **Implementation:** Take only the first image from each review. Run Grad-CAM as if it were a single-image model.
- **Advantages:** Simplest possible implementation.
- **Disadvantages:** Ignores images 2-4 entirely. Violates the multi-image design of the dataset. Produces incomplete explanations for reviews with multiple food/restaurant photos.
- **Engineering trade-off:** Minimal effort, maximum information loss.
- **Research trade-off:** Unacceptable for a thesis that claims to handle multi-image reviews.

#### Strategy D: Process Each Image Independently (Bypass Pooling)

- **Implementation:** Create a wrapper that processes one image through the encoder, skips the multi-image pooling, and feeds the single-image feature directly into the fusion pipeline.
- **Advantages:** Clean per-image heatmap.
- **Disadvantages:** Changes the model's forward pass. The fusion layer receives a feature from one image instead of the pooled feature from all images. The model's behavior is no longer faithful to the trained computation. The explanation does not reflect what the model actually does during inference.
- **Engineering trade-off:** Moderate effort, but produces unfaithful explanations.
- **Research trade-off:** Violates post-hoc explanation fidelity. A thesis examiner could challenge: "This explanation was generated from a different model computation than the one that produced the prediction."

**FINAL DECISION: Strategy B -- Per-Image Grad-CAM via Hook Isolation.**

**Reason:** Strategy B is the only approach that preserves the full model computation (all images, masked average pooling, cross-attention fusion) while still producing per-image spatial explanations. The gradients are faithful to the trained model's behavior. The implementation complexity is manageable with PyTorch hooks. For single-image reviews (num_images=1), Strategy B reduces to standard Grad-CAM with no additional complexity.

**Implementation plan:**
1. For single-image reviews: use `MultiTargetScoreWrapper` + `pytorch-grad-cam` directly.
2. For multi-image reviews: use the hook-based approach with manual Grad-CAM computation (forward hook + backward hook on encoder.norm, extract per-image slice, compute weighted sum + ReLU).

---

### R2: Swin-B Target Layer Selection

**Problem:** Which layer in the Swin-B encoder should be the Grad-CAM target? Multiple candidates exist.

**Why it happens:** Swin-B in timm has a deep hierarchy: `patch_embed -> layers[0..3] -> norm -> global_pool`. Different layers capture different abstraction levels.

#### Candidate A: `model.image_model.encoder.norm`

- **Description:** The final LayerNorm applied after the last Swin-B stage, before global average pooling.
- **Output shape:** `[B, 49, 1024]` (tokens x channels) or equivalently `[B, 7, 7, 1024]` (H x W x channels).
- **Advantages:**
  - Deepest layer with spatial information preserved.
  - Contains the most semantically rich features (high-level object/scene detectors).
  - Standard choice in pytorch-grad-cam's Swin Transformer examples.
  - Simple to access: one attribute.
- **Disadvantages:**
  - LayerNorm may slightly modify the feature magnitudes compared to the raw stage output.
  - 7x7 spatial resolution is coarse.

#### Candidate B: `model.image_model.encoder.layers[-1].blocks[-1].norm2`

- **Description:** The last normalization layer inside the final Swin-B block, before the norm layer.
- **Output shape:** Same spatial dimensions `[B, 49, 1024]`.
- **Advantages:**
  - Captures features immediately before the final norm, potentially more "raw" activations.
- **Disadvantages:**
  - More complex path to specify.
  - Functionally very similar to Candidate A (the difference is one LayerNorm application).
  - Less conventional.

#### Candidate C: `model.image_model.encoder.layers[-2]` (Second-to-last stage)

- **Description:** The output of Stage 3 (index 2), which has spatial resolution 14x14 and 512 channels.
- **Output shape:** `[B, 196, 512]` or `[B, 14, 14, 512]`.
- **Advantages:**
  - Higher spatial resolution (14x14 vs 7x7).
  - May capture mid-level features (textures, patterns) better.
- **Disadvantages:**
  - Features are less semantically rich (not the final representation).
  - Gradients must flow through the entire last stage, potentially diluting the signal.
  - Non-standard choice.

**FINAL DECISION: Candidate A -- `model.image_model.encoder.norm`.**

**Reason:** This is the standard target layer for Swin Transformers in Grad-CAM. It is the deepest layer that preserves spatial information, contains the most semantically meaningful features, and is recommended by the `pytorch-grad-cam` library's documentation for Vision Transformers. The 7x7 resolution is adequate for coarse localization (food region vs. background vs. interior), which is the primary thesis question.

**Fallback:** If `encoder.norm` does not produce spatially interpretable output in the specific timm version used, fall back to `encoder.layers[-1].blocks[-1].norm2`. This should be checked during Step 6 (Target Layer Validation).

---

### R3: Feature Map Shape (Channels-Last vs. Channels-First)

**Problem:** `pytorch-grad-cam` expects feature maps in channels-first format `[B, C, H, W]`. Swin-B in timm outputs channels-last format `[B, H*W, C]` or `[B, H, W, C]`.

**Why it happens:** Swin Transformer operates on token sequences `[B, num_patches, embedding_dim]`, unlike CNNs which use `[B, C, H, W]`. The timm implementation preserves this token-based format through the norm layer.

#### Strategy A: Use `reshape_transform` parameter

- **Implementation:** `pytorch-grad-cam` supports a `reshape_transform` callable that converts the activation tensor to `[B, C, H, W]` format before computing Grad-CAM.
- **Advantages:** Clean, built-in solution. No model modification needed.
- **Disadvantages:** Must be implemented correctly for the specific output format.

#### Strategy B: Modify the model to output channels-first

- **Implementation:** Add a permute operation after the norm layer.
- **Advantages:** Eliminates the need for reshape_transform.
- **Disadvantages:** Modifies the model architecture, which could break checkpoint loading or other downstream components.

**FINAL DECISION: Strategy A -- Use `reshape_transform`.**

**Implementation:**
```
class SwinTransformerReshapeTransform:
    def __call__(self, tensor):
        # Input: [B, H*W, C] or [B, H, W, C]
        if tensor.dim() == 3:
            B, tokens, C = tensor.shape
            H = W = int(tokens ** 0.5)
            result = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif tensor.dim() == 4:
            result = tensor.permute(0, 3, 1, 2)
        return result
        # Output: [B, C, H, W] = [B, 1024, 7, 7]
```

This transform is passed to `GradCAM(reshape_transform=SwinTransformerReshapeTransform())`.

---

### R4: Grad-CAM on Regression Output

**Problem:** `pytorch-grad-cam` was originally designed for classification. The `ClassifierOutputTarget` class selects a class index from the output logits. Our model outputs 5 regression scores, not class probabilities.

**Why it happens:** The standard Grad-CAM pipeline calls `targets[i](output)` to get the scalar to backpropagate from. `ClassifierOutputTarget(class_idx)` returns `output[0, class_idx]`. For regression, the wrapper already selects the target score index.

#### Strategy A: Use `targets=None`

- **Description:** When `targets=None`, `pytorch-grad-cam` uses the sum of the model output as the backpropagation target.
- **Applicability:** If the wrapper returns `[B, 1]` (a single scalar per batch element), `targets=None` effectively uses that scalar, which is correct.
- **Advantages:** Simplest, no custom target class needed.
- **Disadvantages:** Only works if the wrapper's output is already the desired single target score.

#### Strategy B: Use `ClassifierOutputTarget(0)`

- **Description:** Since the wrapper returns `[B, 1]`, use `ClassifierOutputTarget(0)` to select index 0.
- **Advantages:** Explicit, uses the library's built-in class.
- **Disadvantages:** Semantically misleading name ("classifier" for regression).

#### Strategy C: Implement `RegressionScoreTarget`

- **Description:** Custom target class that simply returns the model output as-is (since the wrapper already selects the correct score).
- **Implementation:**
  ```
  class RegressionScoreTarget:
      def __call__(self, model_output):
          return model_output  # model_output is already [B] or scalar
  ```
- **Advantages:** Semantically clear, self-documenting.
- **Disadvantages:** Minimal extra code.

**FINAL DECISION: Strategy A for the simple case (wrapper returns `[B, 1]`, use `targets=None`). If this causes issues with `pytorch-grad-cam`'s internal handling, fall back to Strategy C with a custom `RegressionScoreTarget`.**

**Reason:** The `MultiTargetScoreWrapper` already selects the target score. When `targets=None`, `pytorch-grad-cam` sums the output, which for a `[B, 1]` tensor is equivalent to selecting the single value. This is the simplest correct approach. Document both options in the code so the implementer can switch if needed.

---

### R5: Resolution Limitations (7x7 Heatmap)

**Problem:** The Grad-CAM heatmap at the target layer has spatial resolution of only 7x7 (49 spatial positions for a 224x224 input image). This is a 32x downsampling factor. The upsampled heatmap is coarse.

**Why it happens:** Swin-B with patch size 4 and 4 stages of 2x downsampling reduces spatial dimensions by a factor of 32: 224 / 32 = 7.

#### Impact Assessment

- **Positive:** 7x7 resolution is sufficient to distinguish between major image regions: food area vs. background, table setting vs. wall, dish presentation vs. surrounding decor. For restaurant review images, this level of coarseness is acceptable because the semantically relevant regions (food, interior, table) are typically large.
- **Negative:** Cannot provide pixel-level or fine-grained localization. Cannot distinguish between specific food items on a plate. Cannot identify small details like watermarks or text overlays at high precision.

#### Possible Mitigations

1. **Grad-CAM++:** Uses second-order gradients for potentially better localization. However, the improvement is marginal at 7x7 base resolution and adds complexity.
2. **Layer-CAM or Eigen-CAM:** Alternative attribution methods available in `pytorch-grad-cam`. Could be explored in future work.
3. **Using an earlier layer (14x14):** Higher spatial resolution but lower semantic quality (Risk R2, Candidate C). Trade-off is not favorable.
4. **Using HiResCAM:** Available in `pytorch-grad-cam`, computes element-wise product instead of GAP+weighted-sum. May produce sharper maps but is less established.

**FINAL DECISION: Accept 7x7 resolution as adequate for this thesis.**

**Reason:** The thesis question is "does the model look at food presentation for food_score?" -- not "which pixel of the food plate matters most?" 7x7 spatial resolution is sufficient to answer coarse spatial questions. The limitation should be explicitly acknowledged in the thesis limitations section.

**Thesis language:** "Grad-CAM heatmaps provide coarse spatial localization at 7x7 resolution (for 224x224 input images), sufficient to identify broad regions of interest but not fine-grained pixel-level attribution. This is a known limitation of operating at the final encoder stage."

---

### R6: Black Padding Images

**Problem:** Reviews with fewer than 4 images are padded with black (all-zero) PIL images of size 224x224. Running Grad-CAM on these padding images would produce meaningless results and waste computation.

**Why it happens:** The `MultimodalDataset.__getitem__()` in `src/dataset.py` creates `Image.new('RGB', (224, 224), color='black')` for missing images. The `num_images` tensor records how many real images exist.

#### Impact

- Padding images carry no visual information.
- The encoder still processes them (they produce features, which are masked to zero during average pooling via the `num_images` mask).
- Grad-CAM gradients through padding images are non-informative: the activations are near-zero (black input produces near-zero features in deep layers), and the mask ensures they do not contribute to the pooled representation.

**FINAL DECISION: Never run Grad-CAM on padding images.**

**Implementation:**
1. In `GradCAMExplainer.explain_sample()`, read `num_images` from the sample.
2. Loop only over `range(num_images)`, not `range(max_images)`.
3. Assert `image_index < num_images` before per-image Grad-CAM.
4. Log a warning if `num_images == 0` (should not happen in valid data).

---

### R7: Memory Pressure from Multiple Grad-CAM Runs

**Problem:** Each sample requires up to `5 targets * N images = 20` Grad-CAM computations. Each computation involves a full forward and backward pass through the multimodal model. For Swin-B + PhoBERT + CrossAttentionFusion, this is significant GPU memory.

**Why it happens:** Grad-CAM requires gradient computation, which means the full computational graph must be built (no `torch.no_grad()`). Swin-B has ~88M parameters, PhoBERT has ~135M parameters.

#### Mitigation Strategies

1. **Process one sample at a time (B=1):** Already required by the per-image approach.
2. **Explicit garbage collection:** Call `torch.cuda.empty_cache()` and `gc.collect()` between samples.
3. **Use context manager:** `with GradCAM(...) as cam:` ensures hooks are cleaned up.
4. **Delete intermediate tensors:** Explicitly `del` large tensors after saving results.
5. **CPU fallback:** For machines without sufficient GPU memory, support CPU computation (slower but functional).

**FINAL DECISION: Process B=1, use context managers, explicit cache clearing between samples, document CPU fallback.**

---

### R8: Target Layer Access Through Wrapper

**Problem:** When using `MultiTargetScoreWrapper`, the `target_layer` must be accessible through the wrapper's attribute hierarchy. `pytorch-grad-cam` accesses layers via the model object passed to it.

**Why it happens:** `pytorch-grad-cam` registers hooks on `target_layers` which must be sub-modules of the `model` argument. If the wrapper does not expose the inner model's layers correctly, hooks will fail.

#### Solution

The `MultiTargetScoreWrapper` stores the multimodal model as `self.multimodal_model`. The target layer is accessed as:

```
wrapper.multimodal_model.image_model.encoder.norm
```

This is a valid sub-module of `wrapper` (through the `nn.Module` hierarchy). `pytorch-grad-cam` will find it when traversing the module tree.

**Verification:** After creating the wrapper, confirm that `target_layer` appears in `dict(wrapper.named_modules())`.

**FINAL DECISION: Access via `wrapper.multimodal_model.image_model.encoder.norm`. Verify with `named_modules()` in Step 6.**

---

### R9: Frozen Parameters and Gradient Computation

**Problem:** In `CrossAttentionFusion.__init__()`, all text and image model parameters are frozen (`requires_grad = False`). Some image encoder layers may be unfrozen (`unfreeze_image_layers`). Grad-CAM needs gradients to flow through the target layer.

**Why it happens:** The model was trained with most parameters frozen. After loading from checkpoint, the `requires_grad` flags reflect the training configuration.

#### Analysis

- `pytorch-grad-cam` internally calls `model.eval()` and manages gradient computation.
- Crucially, `pytorch-grad-cam` enables gradients on the TARGET LAYER's output via hooks, regardless of whether the layer's parameters have `requires_grad = True/False`.
- The forward hook captures activations (does not require `requires_grad` on parameters).
- The backward hook captures gradients of the LOSS with respect to the ACTIVATIONS (not parameters). Gradient flow through activations is controlled by the computational graph, not by `requires_grad` on parameters.
- However, if ALL intermediate operations between the target layer and the output have no gradient path (all parameters frozen AND no gradient-carrying operations), gradients may not flow.
- In practice: the layers between `encoder.norm` and the output (global_pool, image_proj, cross-attention, head) have trainable parameters (the fusion layers are not frozen). Therefore, gradients WILL flow from the output back through these layers to the encoder's norm output.

**FINAL DECISION: No action needed. Gradient flow is ensured by the trainable fusion layers between the target layer and the output. Verify with the gradient flow test (Validation V6).**

---

## 13. Best Practices

### BP1: Deterministic Execution

- Set `torch.manual_seed(42)` before any Grad-CAM computation.
- Use `model.eval()` to disable dropout and batch norm training behavior.
- Disable CUDA benchmark mode: `torch.backends.cudnn.benchmark = False`.
- Set `torch.backends.cudnn.deterministic = True`.
- All Grad-CAM results should be bit-reproducible across runs with the same seed and checkpoint.

### BP2: Artifact Naming Consistency

- Use the naming conventions defined in Section 8 throughout all code.
- Zero-pad sample IDs to 4 digits: `sample_0042` not `sample_42`.
- Use target names as defined in `xai/config.py:TARGET_NAMES`, never hardcoded strings.
- Include target index in filenames for unambiguous sorting.

### BP3: Figure Quality for Thesis

- Use 300 DPI for all saved PNGs intended for thesis figures.
- Use 150 DPI for exploratory/debugging figures.
- Set consistent colormap: `jet` or `turbo` for heatmaps (matching pytorch-grad-cam defaults).
- Use consistent figure sizes: single heatmap 4x4 inches, comparison figure 24x4 inches (6 panels).
- Include axis labels and titles in all figures.
- Use a consistent font size: 12pt for axis labels, 14pt for titles.

### BP4: Logging

- Log every sample processed: sample ID, num_images, time taken, number of artifacts generated.
- Log any warnings (e.g., uniform heatmap, zero gradient, unexpected tensor shapes).
- Use Python's `logging` module, not `print()`, in the module code.
- In the notebook, `print()` is acceptable for interactive display.

### BP5: Error Handling

- Catch and log exceptions during per-sample processing without crashing the batch.
- If a sample fails (e.g., corrupt image, unexpected shape), log the error and continue.
- Record failed samples in the batch summary JSON.
- Never silently skip failures.

### BP6: Memory Management

- Use `torch.cuda.empty_cache()` between samples on GPU.
- Use `with GradCAM(...) as cam:` context managers to ensure hook cleanup.
- Delete large intermediate tensors (`del grayscale_cam, overlay`) after saving.
- Monitor GPU memory usage if processing many samples.

### BP7: Checkpoint Handling

- Always load the checkpoint using Phase 1's `load_model()` function for consistency.
- Never modify the model state dict or architecture after loading.
- Record the checkpoint path in every metadata JSON for traceability.

### BP8: Raw Value Preservation

- Always save the raw `[7, 7]` Grad-CAM activation map as `.npy` alongside the visual overlay.
- This enables later quantitative analysis (Phase 6, 8) without regenerating heatmaps.
- Raw values are in `[0, 1]` range after normalization by pytorch-grad-cam.
- The file size is negligible (7x7x8 bytes = ~400 bytes per file).

### BP9: Original Image Recovery

- For heatmap overlay, always use the ORIGINAL image (loaded from disk, resized to 224x224, normalized to `[0, 1]` float), NOT the preprocessed tensor.
- The preprocessed tensor has channel normalization (ImageNet mean/std subtraction) applied, which distorts colors.
- Store the image loading logic in `xai/utils.py` as a shared utility.

### BP10: Configuration Management

- All Grad-CAM parameters (target layer name, DPI, spatial size, colormap) should be defined in `xai/config.py`.
- Do not hardcode these values in `gradcam_explainer.py` or the notebook.
- This enables easy modification for different backbones or experiments.

---

## 14. Deliverables

### Code Deliverables

| # | Deliverable | Format | Location |
|---|-------------|--------|----------|
| D1 | `GradCAMExplainer` module | Python (.py) | `xai/gradcam_explainer.py` |
| D2 | Phase 2 notebook | Jupyter (.ipynb) | `xai/notebooks/Phase2_GradCAM.ipynb` |
| D3 | Config updates | Python (.py) | `xai/config.py` (modified) |
| D4 | Utils updates | Python (.py) | `xai/utils.py` (modified) |

### Artifact Deliverables (Per Sample)

| # | Deliverable | Format | Count per Sample |
|---|-------------|--------|-----------------|
| D5 | Heatmap overlay PNGs | PNG | 5 (single-image) or 5 * num_images (multi-image) |
| D6 | Raw activation maps | NPY | 5 (single-image) or 5 * num_images (multi-image) |
| D7 | 5-target comparison figure | PNG | 1 (single-image) or 2 (multi-image: per-image + grid) |
| D8 | Metadata JSON | JSON | 1 |

### Aggregate Deliverables

| # | Deliverable | Format | Count |
|---|-------------|--------|-------|
| D9 | Batch summary JSON | JSON | 1 |
| D10 | Sanity check: correlation matrix figure | PNG | 1 per tested sample |
| D11 | Sanity check: randomization comparison figure | PNG | 1 per tested sample |

### Documentation Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| D12 | Notebook markdown cells | Interpretation guidelines, figure captions, connection to thesis |
| D13 | Code docstrings | All public classes and functions documented |

---

## 15. Thesis Usage

### Results Section

1. **Target-specific heatmaps (Figure X):** Show the 5-target comparison figure for 2-3 representative samples. This is the primary evidence that the image branch encodes target-specific spatial information. Caption should emphasize that different targets produce different spatial attention patterns.

2. **Per-image attribution (Figure X+1):** For a multi-image review, show heatmaps across all images for one target. Demonstrate that the model distributes attention across multiple food/restaurant photos differently.

3. **Quantitative target specificity (Table X):** Report the pairwise Pearson correlation matrix between the 5 target heatmaps, averaged over N test samples. Low correlation between food_score and atmosphere_score heatmaps supports the claim that the model learns target-specific visual features.

### Discussion Section

4. **Semantic alignment discussion:** Interpret whether food_score heatmaps actually highlight food presentation, atmosphere_score highlights interior, etc. Discuss cases where this alignment holds and cases where it fails.

5. **Failure case analysis (Figure X+2):** Show samples where Grad-CAM highlights background, watermarks, or irrelevant regions. Discuss potential model shortcuts and their implications for model reliability.

6. **Image-branch contribution:** Connect Grad-CAM findings to Phase 4 (SHAP) results about modality dominance. If SHAP shows image contribution is low for price_score, and Grad-CAM shows diffuse heatmaps for price_score, the evidence is mutually reinforcing.

7. **Resolution limitation discussion:** Acknowledge the 7x7 coarseness in the limitations subsection.

### Case Studies (Phase 6)

8. **Multi-method case study:** For selected samples, combine Grad-CAM heatmaps with attention maps (Phase 3), SHAP values (Phase 4), and LIME results (Phase 5) to provide a comprehensive multi-level explanation.

### Defense Presentation

9. **Slide: "Where does the model look?"** -- Show 5-target comparison figure with clear annotations.
10. **Slide: "Is the model using valid evidence?"** -- Show good case (food highlighted for food_score) vs. failure case (background highlighted).
11. **Prepared defense answers:**
    - "Where exactly is Grad-CAM attached?" -> "To the final LayerNorm of the Swin-B encoder, before global average pooling. This layer outputs 7x7 spatial feature maps with 1024 channels."
    - "How do you handle multiple images?" -> "I process all images through the full model to preserve the masked average pooling context, then extract per-image activations and gradients via hooks."
    - "Does Grad-CAM prove the model uses food presentation?" -> "No. It provides target-linked supportive evidence, not causal proof. I validate it with perturbation methods in Phase 5."

### Journal Paper

12. **Method section:** Describe the wrapper design, target layer selection, and per-image hook isolation strategy.
13. **Results section:** Report target specificity quantitatively (correlation matrix) and qualitatively (representative heatmaps).
14. **Supplementary material:** Include additional heatmaps for diverse samples in an appendix or supplementary PDF.

---

## 16. Phase Completion Checklist

### Infrastructure

- [ ] `xai/gradcam_explainer.py` created with all classes: `GradCAMExplainer`, `MultiTargetScoreWrapper`, `SingleImageGradCAMWrapper` (or hook-based equivalent), `SwinTransformerReshapeTransform`, `RegressionScoreTarget`.
- [ ] `xai/config.py` updated with Grad-CAM constants.
- [ ] `xai/utils.py` updated with `get_original_image_rgb()` helper.
- [ ] All imports resolve without errors.
- [ ] `pytorch-grad-cam` installed and importable.

### Target Layer Validation

- [ ] `model.image_model.encoder.norm` verified as producing `[B*N, 49, 1024]` or equivalent spatial output.
- [ ] Reshape transform verified: output is `[B, 1024, 7, 7]` after transform.
- [ ] Gradient flow verified: non-zero gradient from each target score to the target layer output.

### Single-Image Grad-CAM

- [ ] `MultiTargetScoreWrapper` correctly wraps the full model and returns `[B, 1]` for a selected target.
- [ ] `GradCAM` object successfully generates `grayscale_cam` of shape `[1, 7, 7]`.
- [ ] `show_cam_on_image` produces a valid 224x224 RGB overlay image.
- [ ] Overlay PNG saved to correct path with correct naming.
- [ ] Raw activation NPY saved with shape `[7, 7]` and values in `[0, 1]`.

### Multi-Image Grad-CAM

- [ ] Per-image hook-based Grad-CAM generates correct per-image heatmaps.
- [ ] Padding images (beyond `num_images`) are never processed.
- [ ] Per-image heatmaps saved with `_img{k}` suffix.
- [ ] All real images in a multi-image review produce non-trivial heatmaps.

### Comparison Figures

- [ ] 5-target comparison figure generated for each sample: 6 columns (original + 5 targets).
- [ ] Multi-image comparison grid generated for multi-image samples.
- [ ] All figures saved at 300 DPI.
- [ ] Titles and labels are correct and readable.

### Metadata

- [ ] Per-sample metadata JSON contains: sample_id, num_images, predictions, true_labels, target_layer, artifacts list, timestamp, checkpoint path.
- [ ] Batch summary JSON contains: all processed sample IDs, timing, success/failure status.

### Sanity Checks

- [ ] Target specificity test: at least one pair of targets has Pearson r < 0.9 for a meaningful sample.
- [ ] Randomization test: trained vs. randomized model heatmaps have Pearson r < 0.7.
- [ ] Reproducibility test: two runs on the same sample produce identical results (allclose atol=1e-7).
- [ ] Shape consistency: all raw CAM files have shape `[7, 7]`, all overlays are 224x224.

### Notebook

- [ ] All 12 cells execute without errors.
- [ ] Cell outputs are saved in the notebook.
- [ ] Markdown cells provide interpretation guidance.
- [ ] Notebook is self-contained (can be run from scratch with only the checkpoint and data).

### Integration

- [ ] Output folder structure matches Section 8 specification exactly.
- [ ] All file paths use Phase 1's `EXP_DIR` and subdirectory conventions.
- [ ] Artifacts are discoverable by Phase 6 (Case Study) and Phase 8 (Thesis Visualization).

---

*End of Phase 2: Grad-CAM Implementation Proposal*
