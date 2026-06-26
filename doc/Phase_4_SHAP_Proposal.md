# Phase 4: Fusion-level SHAP Analysis

## Implementation Proposal for Multimodal Vietnamese Restaurant Review Quality Assessment

**Target Model:** Swin-B + PhoBERT + CrossAttentionFusion + LogCosh (EXP_050C, Mean MAE 1.1079)

---

## 1. Purpose

### Why this phase exists

SHAP (SHapley Additive exPlanations) analysis at the fusion level exists to answer the central multimodal research question that no other XAI method in this pipeline can answer: **which modality -- image or text -- drove the prediction?**

Grad-CAM (Phase 2) localizes spatial evidence in the image branch. Attention Visualization (Phase 3) reveals token interaction patterns in the text branch. But neither method can quantify the relative contribution of image versus text to the final predicted score. Only SHAP, applied at the fusion level, provides this decomposition.

### Research motivation

In a multimodal system predicting five quality aspects (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`), the central scientific question is whether the model genuinely uses both modalities and whether different targets rely on different modality balances. For example, one expects `atmosphere_score` to be more image-driven (restaurant interior, lighting, decor) while `price_score` should be more text-driven (explicit price mentions in reviews). SHAP is the only method in this pipeline that can provide quantitative evidence for these hypotheses.

### Engineering motivation

SHAP values decompose each prediction into per-feature additive contributions relative to a background baseline. When features are grouped by modality origin, this immediately yields modality contribution percentages. The fusion vector in CrossAttentionFusion is 1024-dimensional with a known structure: dimensions 0:512 originate from the text-attended cross-attention output (`t_out`), and dimensions 512:1024 originate from the image-attended cross-attention output (`i_out`). This clean structural split makes modality grouping natural and well-defined.

### Critical distinction from ConcatFusion

The existing XAI handbook (`doc/Explainable_AI_for_Multimodal_Product_Quality_Assessment.md`) documents SHAP for ConcatFusion where `fused = cat[text_feat(768), image_feat(1024)]` yields a 1792-dimensional vector with raw encoder features. **This proposal is specifically designed for CrossAttentionFusion**, where the fusion vector structure is fundamentally different:

- **ConcatFusion:** dims 0:768 = raw PhoBERT features, dims 768:1792 = raw Swin-B features
- **CrossAttentionFusion:** dims 0:512 = text features AFTER cross-attending to image (`t_out`), dims 512:1024 = image features AFTER cross-attending to text (`i_out`)

The cross-attended features contain information from both modalities. Grouping dims 0:512 as "text-origin" and dims 512:1024 as "image-origin" is a reasonable approximation that must be acknowledged as such in all reporting.

---

## 2. Objectives

### Research objectives

1. Quantify per-target modality contribution (text-origin vs image-origin) using SHAP value aggregation across the 1024-dimensional fused embedding.
2. Determine the modality dominance profile for each of the five targets: is `food_score` balanced? Is `price_score` text-dominant? Is `atmosphere_score` image-dominant?
3. Compare modality contributions between correct predictions and incorrect predictions to identify whether modality imbalance correlates with prediction error.
4. Generate publication-quality evidence for the thesis-level claim: "The multimodal system genuinely uses both modalities, with target-specific modality balance."

### Engineering objectives

1. Implement `xai/shap_explainer.py` with a `FusionHeadWrapper` that maps the fused embedding to a single target score.
2. Implement an embedding extraction pipeline that captures the fused vector (1024-d) before the prediction head, using a forward hook on the `torch.cat` operation in `CrossAttentionFusion.forward()`.
3. Implement both DeepExplainer (primary, fast) and modality-level KernelExplainer (supplementary, fast 2-feature version) approaches.
4. Produce reproducible SHAP artifacts: raw `.npy` values, background embeddings `.pt`, summary CSV, and publication-quality PNG plots.
5. Store all artifacts under `experiments/EXP_050C/xai/shap/` in a structured folder hierarchy.

### Expected contributions

- A per-target modality contribution summary table suitable for thesis Results chapter.
- Waterfall plots for selected case study samples showing per-feature contributions.
- Beeswarm/summary plots at dataset level showing which fused dimensions matter most.
- A modality contribution chart comparing all five targets side-by-side.
- Quantitative evidence that CrossAttentionFusion does not exhibit branch collapse (one modality contributing near-zero).

---

## 3. Inputs

### Checkpoint

- **Path:** `experiments/EXP_050C/checkpoints/best_model.pth` (or equivalent checkpoint path for the best CrossAttentionFusion + LogCosh model)
- **Contents:** Full `CrossAttentionFusion` state dict, including `text_model`, `image_model`, `text_proj`, `image_proj`, `cross_attn_t2i`, `cross_attn_i2t`, and `head` parameters.
- **Usage:** Load once, set to `model.eval()`, freeze all parameters, never modify.

### Model architecture files

- `Models/CrossAttentionFusion.py` -- the target model class
- `Models/TextModel.py` -- text encoder wrapper (PhoBERT `vinai/phobert-base-v2`, hidden_size=768)
- `Models/ImageModel.py` -- image encoder wrapper (Swin-B `swin_base_patch4_window7_224`, num_features=1024)

### Dataset files

- `data/text/train.csv` -- training split for background sample extraction
- `data/text/val.csv` -- validation split for background sample extraction and evaluation samples
- `data/text/test.csv` -- test split for final SHAP analysis
- `data/image/` -- image directory (MD5-hashed filenames)

### Dataset class

- `src/dataset.py` -- `MultimodalDataset` or `AdvancedMultimodalDataset` class for data loading

### Configuration

- `Config.py` -- argument definitions for experiment configuration
- `xai/config.py` -- XAI-specific configuration (from Phase 1 Infrastructure)
- `xai/utils.py` -- XAI utility functions (from Phase 1 Infrastructure)

### Dependencies

- `shap` -- SHAP library (DeepExplainer, KernelExplainer, plotting)
- `torch` -- PyTorch for model inference and hook registration
- `numpy` -- numerical operations on SHAP arrays
- `matplotlib` -- plotting
- `pandas` -- summary CSV generation

---

## 4. Outputs

### Per-sample per-target outputs

For each explained sample and each of the 5 targets:

| Artifact | Format | Description |
|---|---|---|
| SHAP values vector | `.npy` [1024] | Raw SHAP values for each fused dimension |
| Waterfall plot | `.png` | Top-k features pushing prediction above/below baseline |
| Bar plot | `.png` | Top-k features by absolute SHAP magnitude |
| Modality contribution | included in summary CSV | text-origin %, image-origin %, signed sums |

### Dataset-level outputs

| Artifact | Format | Description |
|---|---|---|
| Modality contribution summary | `.csv` | Per-target: text_pct, image_pct, text_signed_mean, image_signed_mean across all samples |
| Modality contribution chart (all targets) | `.png` | Side-by-side bar chart of modality % for all 5 targets |
| Per-target modality contribution chart | `.png` x 5 | Individual charts per target |
| Beeswarm/summary plot | `.png` x 5 | Dataset-level feature importance distribution per target |
| Background fused embeddings | `.pt` [N_bg, 1024] | Saved for reproducibility |
| All sample fused embeddings | `.pt` [N_samples, 1024] | All explained sample embeddings |
| SHAP configuration metadata | `.json` | Background size, explainer type, model checkpoint, timestamp |

---

## 5. Architecture Attachment Point

### Precise attachment location

SHAP attaches to the **prediction head** of `CrossAttentionFusion`, not to the full model. The attachment point is precisely defined as follows:

```
Full CrossAttentionFusion forward pass:
  text_feat [B, 768]  ---> text_proj ---> t [B, 1, 512]
  image_feat [B, 1024] --> image_proj --> i [B, 1, 512]
  
  cross_attn_t2i(Q=t, K=i, V=i) --> t_out [B, 1, 512]
  cross_attn_i2t(Q=i, K=t, V=t) --> i_out [B, 1, 512]
  
  fused = cat[t_out.squeeze(1), i_out.squeeze(1)] --> [B, 1024]
                                                        ^
                                                        |
                                              SHAP ATTACHMENT POINT
                                              (hook captures here)
                                                        |
                                                        v
  head(fused):
    Linear(1024, 512) --> ReLU --> Dropout(0.2) --> Linear(512, 256) --> ReLU --> Linear(256, 5)
                                                                                      |
                                                                            output [B, 5]
```

### What the SHAP wrapper explains

The `FusionHeadWrapper` takes the fused vector `[B, 1024]` as input and returns a single scalar `[B, 1]` for one selected target index. SHAP computes how each of the 1024 fused dimensions contributes to that scalar output, relative to the background distribution.

### Why this attachment point

1. **Dimensionality is tractable:** 1024 features is manageable for DeepExplainer (gradient-based, fast). KernelExplainer would be slow at 1024 dimensions but the 2-feature modality-level grouping fallback is extremely fast.

2. **The head is a clean differentiable function:** `Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear` is a standard MLP that DeepExplainer handles natively.

3. **Feature grouping is meaningful:** The first 512 dimensions are text-origin (from `t_out`), the last 512 are image-origin (from `i_out`). This provides a natural modality decomposition.

4. **Isolation from encoder complexity:** By explaining only the head, we avoid the computational cost and complexity of backpropagating SHAP through Swin-B and PhoBERT encoders. The encoders are used only to extract fused embeddings.

### Architecture diagram

```
Input Data (text + images)
        |
        v
[Full CrossAttentionFusion Model]
        |
   fused vector [B, 1024]  <--- extracted via hook, stored as tensor
        |
        v
[FusionHeadWrapper(head, score_index)]  <--- SHAP explains THIS
        |
        v
   scalar score [B, 1]     <--- one target at a time
```

---

## 6. Detailed Implementation Plan

### Step 1: Create `xai/shap_explainer.py`

This is the primary implementation file. It contains all SHAP-related functionality.

#### 1.1 FusionHeadWrapper class

**Responsibility:** Wraps the prediction head (`self.head` from `CrossAttentionFusion`) to accept fused embeddings and return a single target score.

**Specification:**
- Inherits from `nn.Module`.
- Constructor takes: `head_sequential` (the `nn.Sequential` head from the model), `score_index` (int, 0-4).
- `forward(self, fused_embedding)`:
  - Input: `fused_embedding` of shape `[B, 1024]`.
  - Passes through `head_sequential` to get `[B, 5]`.
  - Selects column `score_index` and returns `[B, 1]`.
- Must call `self.eval()` and `torch.no_grad()` is NOT used (DeepExplainer needs gradients).
- Dropout must be disabled via `eval()` mode.

**Critical implementation detail for CrossAttentionFusion:** Unlike ConcatFusion which has separate `fusion_fc` and `factor_head` modules, CrossAttentionFusion has a single `self.head` Sequential:
```
head = Sequential(
    Linear(1024, 512),  # [0]
    ReLU(),             # [1]
    Dropout(0.2),       # [2]
    Linear(512, 256),   # [3]
    ReLU(),             # [4]
    Linear(256, 5)      # [5]
)
```
The wrapper must accept this single Sequential directly.

#### 1.2 extract_fused_embeddings function

**Responsibility:** Run the full model on a dataloader to capture fused vectors before the head.

**Specification:**
- Input: `model` (CrossAttentionFusion, in eval mode), `dataloader`, `device`, optional `max_samples`.
- Uses a forward hook registered on the point just before `self.head(fused)` is called.

**Hook strategy:** Register a forward pre-hook on `model.head` (the Sequential module). The pre-hook receives `(module, input_args)` where `input_args[0]` is the fused tensor `[B, 1024]`. Store it in a list.

**Alternative strategy:** Modify the forward method temporarily to return both the prediction and the fused vector. This is less clean but simpler.

**Recommended strategy:** Forward hook on `model.head`.

**Implementation pseudocode:**
```
function extract_fused_embeddings(model, dataloader, device, max_samples=None):
    model.eval()
    all_fused = []
    all_labels = []
    all_preds = []
    
    captured = {}
    
    define hook_fn(module, input_args):
        captured['fused'] = input_args[0].detach().cpu()
    
    hook = model.head.register_forward_pre_hook(hook_fn)
    
    with torch.no_grad():
        for batch in dataloader:
            move batch to device
            preds = model(input_ids, attention_mask, pixel_values, num_images)
            all_fused.append(captured['fused'])
            all_labels.append(batch['factor_scores'])
            all_preds.append(preds.detach().cpu())
            
            if max_samples and total collected >= max_samples:
                break
    
    hook.remove()
    return cat(all_fused), cat(all_labels), cat(all_preds)
```

**Output:** Tensor of shape `[N, 1024]`, tensor of shape `[N, 5]` (labels), tensor of shape `[N, 5]` (predictions).

#### 1.3 select_background_samples function

**Responsibility:** Select a representative subset of fused embeddings for the SHAP background set.

**Specification:**
- Input: `all_fused` tensor `[N, 1024]`, `n_background` (default 100).
- Strategy: Random sampling from the validation set.
- Output: `background_fused` tensor `[n_background, 1024]`.
- Seeds the random number generator for reproducibility.
- Saves the selected indices for reproducibility.

#### 1.4 compute_shap_deep function (primary method)

**Responsibility:** Compute SHAP values using DeepExplainer on the fusion head wrapper.

**Specification:**
- Input: `wrapper` (FusionHeadWrapper), `background_fused` `[N_bg, 1024]`, `sample_fused` `[N_samples, 1024]`.
- Creates `shap.DeepExplainer(wrapper, background_fused)`.
- Calls `explainer.shap_values(sample_fused)`.
- Handles return format: DeepExplainer may return a list (for multi-output) or a single array. Since the wrapper outputs `[B, 1]`, expect a single array of shape `[N_samples, 1024]`.
- Returns: `shap_values` array `[N_samples, 1024]`, `base_value` scalar (expected value).

**Critical note:** DeepExplainer requires PyTorch tensors, not numpy arrays. Both `background_fused` and `sample_fused` must be PyTorch tensors with `requires_grad=False` on the appropriate device.

#### 1.5 compute_shap_modality_level function (supplementary fast method)

**Responsibility:** Compute modality-level SHAP using KernelExplainer with only 2 super-features.

**Specification:**
- Input: `wrapper` (FusionHeadWrapper), `background_fused` `[N_bg, 1024]`, `sample_fused` `[N_samples, 1024]`.
- Creates a modality-level prediction function that:
  1. Accepts a 2-feature input: `[text_block_mean, image_block_mean]`.
  2. Internally reconstructs a full 1024-d vector by broadcasting: dims 0:512 all set to `text_block_mean`, dims 512:1024 all set to `image_block_mean`.
  3. Actually, a better approach: for each perturbation, replace the text block (0:512) with the corresponding background block mean or the sample block, and similarly for the image block (512:1024). This is a grouped-feature approach.
- **Refined specification for modality-level KernelExplainer:**
  1. Define a function `modality_predict(binary_mask_2d)` where each row is `[text_present, image_present]` with values 0 or 1.
  2. When `text_present=1`, use the sample's dims 0:512. When `text_present=0`, use the background mean dims 0:512.
  3. When `image_present=1`, use the sample's dims 512:1024. When `image_present=0`, use the background mean dims 512:1024.
  4. Pass the reconstructed 1024-d vector through the wrapper.
- Uses `shap.KernelExplainer` with the modality-level predict function and a background of `[[mean_text_block, mean_image_block]]`.
- This produces exactly 2 SHAP values per sample: one for text-origin, one for image-origin.
- Extremely fast because only 2 features.

#### 1.6 modality_contribution function

**Responsibility:** Compute modality contribution percentages from per-feature SHAP values.

**Specification:**
- Input: `shap_values` array `[1024]` (single sample) or `[N, 1024]` (batch), `text_dim=512`.
- Computes:
  - `text_abs = sum(|shap_values[0:text_dim]|)`
  - `image_abs = sum(|shap_values[text_dim:]|)`
  - `total = text_abs + image_abs + 1e-8`
  - `text_pct = 100 * text_abs / total`
  - `image_pct = 100 * image_abs / total`
  - `text_signed = sum(shap_values[0:text_dim])`
  - `image_signed = sum(shap_values[text_dim:])`
- Returns: dictionary with keys `text_pct`, `image_pct`, `text_signed`, `image_signed`, `text_abs`, `image_abs`.

**Naming convention:** The dimensions are labeled "text-origin" and "image-origin" (not "text" and "image") to acknowledge that cross-attended features contain information from both modalities.

#### 1.7 Plotting functions

All plotting functions follow these conventions:
- DPI: 200 for saved figures, 300 for publication figures.
- Color scheme: consistent across all XAI phases. Text-origin = `#1b9e77` (teal), Image-origin = `#d95f02` (orange).
- Font: Use matplotlib default or configure via `xai/config.py`.
- All plots include descriptive titles with target name.
- All plots include axis labels.
- All plots use `plt.tight_layout()` before saving.
- All plots call `plt.close()` after saving to prevent memory leaks.

**7a. plot_waterfall(shap_values_1d, base_value, feature_data_1d, sample_id, target_idx, target_name, save_path)**
- Uses `shap.plots.waterfall()` with a `shap.Explanation` object.
- Shows top contributing features pushing the prediction above/below baseline.
- File name: `sample_{id}_target{idx}_{name}_waterfall.png`

**7b. plot_bar(shap_values_1d, feature_data_1d, sample_id, target_idx, target_name, save_path)**
- Uses `shap.plots.bar()` with a `shap.Explanation` object.
- Shows top features by absolute SHAP value.
- File name: `sample_{id}_target{idx}_{name}_bar.png`

**7c. plot_beeswarm(shap_values_2d, feature_data_2d, target_idx, target_name, save_path)**
- Uses `shap.plots.beeswarm()` with dataset-level SHAP values.
- Shows distribution of SHAP values across samples for each feature.
- File name: `dataset_level_beeswarm_target{idx}_{name}.png`

**7d. plot_modality_contribution(text_pct, image_pct, target_idx, target_name, save_path)**
- Custom matplotlib bar chart showing text-origin vs image-origin contribution for one target.
- Includes percentage labels on bars.
- File name: `modality_contribution_target{idx}_{name}.png`

**7e. plot_modality_contribution_summary(contributions_dict, save_path)**
- Grouped bar chart showing modality contribution across all 5 targets.
- X-axis: target names. Two bars per target: text-origin and image-origin.
- File name: `modality_contribution_summary_all_targets.png`

**7f. plot_modality_contribution_heatmap(contributions_df, save_path)**
- Heatmap with targets on Y-axis, modalities on X-axis, values as percentages.
- Alternative visualization to the grouped bar chart.
- File name: `modality_contribution_heatmap.png`

### Step 2: Embedding extraction strategy

#### 2.1 Hook-based extraction (recommended)

Register a `register_forward_pre_hook` on `model.head`. When `model.forward()` reaches `self.head(fused)`, the pre-hook fires and captures `fused` before the head processes it.

**Why pre-hook on `model.head` and not a regular hook on the cat operation:**
- `torch.cat` is a functional call, not a module, so hooks cannot be registered on it directly.
- `model.head` is an `nn.Sequential` module, so `register_forward_pre_hook` captures its input, which is exactly the fused vector.

**Verification:** After extraction, verify that:
1. Shape is `[B, 1024]` for every batch.
2. Running the fused vector through `model.head` reproduces the model's original predictions (within floating point tolerance).

#### 2.2 Batch extraction workflow

```
For each split (train, val, test):
    Create dataloader with batch_size (e.g., 32)
    Extract all fused embeddings
    Save as .pt file
```

Expected output sizes (approximate, depends on dataset size):
- `background_fused.pt`: `[100, 1024]` (selected from validation set)
- `val_fused_embeddings.pt`: `[N_val, 1024]`
- `test_fused_embeddings.pt`: `[N_test, 1024]`

### Step 3: Background sample selection

#### 3.1 Selection procedure

1. Extract all fused embeddings from the validation set.
2. Set random seed to 42.
3. Randomly select 100 indices without replacement.
4. Extract the corresponding rows as the background set.
5. Save the background set as `background_fused.pt`.
6. Save the selected indices as `background_indices.json`.

#### 3.2 Why 100 samples

- SHAP documentation recommends 100-200 background samples for DeepExplainer.
- Too few (< 50): unstable base value and SHAP values.
- Too many (> 200): diminishing returns and increased memory.
- 100 is a proven default that balances stability and efficiency.

#### 3.3 Stability verification

Run SHAP with background sizes 50, 100, and 200. Compare modality contribution percentages. If the difference between 100 and 200 is less than 2 percentage points, 100 is sufficient.

### Step 4: SHAP explainer selection

#### 4.1 Primary: DeepExplainer

- **How it works:** Uses a modified backpropagation (DeepLIFT-based) to compute SHAP values analytically. Does not require perturbation sampling.
- **Speed:** Very fast. For a 1024-input, 1-output MLP with 100 background samples, computing SHAP for 1 sample takes < 1 second.
- **Accuracy:** Approximate, but excellent for differentiable neural networks.
- **Requirements:** PyTorch model, differentiable activations, no non-differentiable operations in the head.
- **Compatibility check for CrossAttentionFusion head:** The head uses `Linear`, `ReLU`, `Dropout` (disabled in eval mode). All are differentiable and compatible with DeepExplainer.

#### 4.2 Supplementary: KernelExplainer with modality-level grouping

- **How it works:** Model-agnostic perturbation method based on weighted linear regression.
- **Speed at 1024 features:** Extremely slow (prohibitive). Each sample requires O(2^1024) coalitions, approximated by sampling, but even with sampling, 1024 features is too many.
- **Speed at 2 features (modality-level):** Extremely fast. Only 2^2 = 4 coalitions are exact.
- **Usage:** Provides a direct "text-origin vs image-origin" answer as a validation of DeepExplainer grouping.

#### 4.3 Rejected alternatives

- **GradientExplainer:** A compromise between DeepExplainer and KernelExplainer. Uses expected gradients. Could be used if DeepExplainer fails, but DeepExplainer is expected to work well for this MLP head.
- **Exact Shapley:** Computationally impossible for 1024 features.
- **PermutationExplainer:** Slower than DeepExplainer for this use case.

#### 4.4 Final decision

Use **DeepExplainer** as the primary method for per-feature SHAP values (1024 values per sample per target). Use **modality-level KernelExplainer** (2 features) as a fast supplementary validation that confirms the grouped results from DeepExplainer.

### Step 5: Per-target analysis workflow

For each of the 5 targets, the complete workflow is:

```
For target_idx in [0, 1, 2, 3, 4]:
    target_name = ["food_score", "price_score", "atmosphere_score", 
                    "service_score", "overall_satisfaction"][target_idx]
    
    1. Create FusionHeadWrapper(model.head, score_index=target_idx)
    2. Set wrapper to eval mode
    3. Create DeepExplainer(wrapper, background_fused)
    4. For each sample to explain:
        a. Compute shap_values = explainer.shap_values(sample_fused)
        b. Compute modality_contribution(shap_values)
        c. Run additivity check
        d. Generate waterfall plot
        e. Generate bar plot
        f. Save raw shap_values as .npy
    5. Aggregate all sample contributions for this target
    6. Generate beeswarm plot
    7. Generate modality contribution chart for this target
    8. Save per-target summary row to CSV

After all targets:
    Generate modality_contribution_summary_all_targets.png
    Generate modality_contribution_heatmap.png
    Save modality_contribution_summary.csv
```

### Step 6: Modality-level SHAP (fast alternative)

This is the 2-feature KernelExplainer approach described in Step 1.5.

**When to use:**
- As a quick validation of DeepExplainer results.
- When computing full per-feature SHAP is too slow for a large number of samples.
- For generating a simple "text vs image" summary that is interpretable without embedding dimension knowledge.

**Implementation pseudocode:**
```
function compute_modality_shap(wrapper, background_fused, sample_fused, text_dim=512):
    background_mean = background_fused.mean(dim=0)  # [1024]
    
    define predict_fn(modality_mask_2d):
        results = []
        for mask in modality_mask_2d:  # mask is [text_flag, image_flag]
            input_vec = background_mean.clone()
            if mask[0] == 1:  # text-origin present
                input_vec[0:text_dim] = sample_fused[0:text_dim]
            if mask[1] == 1:  # image-origin present
                input_vec[text_dim:] = sample_fused[text_dim:]
            pred = wrapper(input_vec.unsqueeze(0))
            results.append(pred.item())
        return numpy.array(results).reshape(-1, 1)
    
    bg_modality = numpy.array([[0.5, 0.5]])  # or use background stats
    explainer = shap.KernelExplainer(predict_fn, bg_modality)
    shap_values = explainer.shap_values(numpy.array([[1.0, 1.0]]))
    return shap_values  # [1, 2] -- one value for text, one for image
```

---

## 7. Required Code Files

### New files to create

| File | Responsibility |
|---|---|
| `xai/shap_explainer.py` | Core SHAP implementation: FusionHeadWrapper, embedding extraction, SHAP computation (Deep and Kernel), modality contribution analysis, all plotting functions |
| `xai/shap_config.py` | SHAP-specific configuration constants: text_dim, background_size, target_names, color palette, plot DPI, file naming templates |
| `notebook/SHAP_Phase4_Analysis.ipynb` | Interactive analysis notebook: load checkpoint, extract embeddings, compute SHAP, generate plots, modality analysis |

### Files to modify

| File | Modification |
|---|---|
| `xai/__init__.py` | Add imports for `shap_explainer` module (if `xai/` package exists from Phase 1) |
| `xai/config.py` | Add SHAP-specific configuration entries (target names, color palette, dimension splits) |

### Files referenced but NOT modified

| File | Usage |
|---|---|
| `Models/CrossAttentionFusion.py` | Load model architecture, access `model.head` for wrapper |
| `Models/TextModel.py` | Loaded as part of CrossAttentionFusion |
| `Models/ImageModel.py` | Loaded as part of CrossAttentionFusion |
| `src/dataset.py` | Load dataset for embedding extraction |
| `Config.py` | Reference for experiment configuration |
| `Trainer.py` | Reference for loss function (not used during SHAP) |

### Detailed file specification: `xai/shap_explainer.py`

```
Module: xai/shap_explainer.py

Classes:
    FusionHeadWrapper(nn.Module)
        __init__(self, head_sequential, score_index)
        forward(self, fused_embedding) -> [B, 1]

Functions:
    extract_fused_embeddings(model, dataloader, device, max_samples=None)
        -> (fused_tensor [N, 1024], labels [N, 5], preds [N, 5])
    
    select_background(fused_tensor, n_background=100, seed=42)
        -> (background_tensor [n_bg, 1024], selected_indices list)
    
    compute_shap_deep(wrapper, background_fused, sample_fused)
        -> (shap_values [N, 1024], base_value float)
    
    compute_shap_modality_level(wrapper, background_fused, sample_fused, text_dim=512)
        -> (shap_values [N, 2], base_value float)
    
    modality_contribution(shap_values_1d, text_dim=512)
        -> dict{text_pct, image_pct, text_signed, image_signed, text_abs, image_abs}
    
    batch_modality_contribution(shap_values_2d, text_dim=512)
        -> dict{mean_text_pct, mean_image_pct, std_text_pct, std_image_pct, ...}
    
    additivity_check(wrapper, sample_fused, shap_values, base_value, tolerance=0.01)
        -> (passes bool, max_error float)
    
    plot_waterfall(shap_values_1d, base_value, feature_data, sample_id, 
                   target_idx, target_name, save_dir)
    
    plot_bar(shap_values_1d, feature_data, sample_id, target_idx, target_name, save_dir)
    
    plot_beeswarm(shap_values_2d, feature_data_2d, target_idx, target_name, save_dir)
    
    plot_modality_contribution(text_pct, image_pct, target_idx, target_name, save_dir)
    
    plot_modality_contribution_summary(contributions_dict, save_dir)
    
    plot_modality_contribution_heatmap(contributions_df, save_dir)
    
    save_shap_metadata(config_dict, save_dir)
```

---

## 8. Folder Structure

```
experiments/EXP_050C/xai/shap/
|
+-- sample_{id}_target0_food_score_waterfall.png
+-- sample_{id}_target0_food_score_bar.png
+-- sample_{id}_target1_price_score_waterfall.png
+-- sample_{id}_target1_price_score_bar.png
+-- sample_{id}_target2_atmosphere_score_waterfall.png
+-- sample_{id}_target2_atmosphere_score_bar.png
+-- sample_{id}_target3_service_score_waterfall.png
+-- sample_{id}_target3_service_score_bar.png
+-- sample_{id}_target4_overall_satisfaction_waterfall.png
+-- sample_{id}_target4_overall_satisfaction_bar.png
|
+-- modality_contribution_target0_food_score.png
+-- modality_contribution_target1_price_score.png
+-- modality_contribution_target2_atmosphere_score.png
+-- modality_contribution_target3_service_score.png
+-- modality_contribution_target4_overall_satisfaction.png
+-- modality_contribution_summary_all_targets.png
+-- modality_contribution_heatmap.png
+-- modality_contribution_summary.csv
|
+-- dataset_level_beeswarm_target0_food_score.png
+-- dataset_level_beeswarm_target1_price_score.png
+-- dataset_level_beeswarm_target2_atmosphere_score.png
+-- dataset_level_beeswarm_target3_service_score.png
+-- dataset_level_beeswarm_target4_overall_satisfaction.png
|
+-- raw/
|   +-- shap_values_sample_{id}_target0.npy
|   +-- shap_values_sample_{id}_target1.npy
|   +-- shap_values_sample_{id}_target2.npy
|   +-- shap_values_sample_{id}_target3.npy
|   +-- shap_values_sample_{id}_target4.npy
|   +-- background_fused.pt
|   +-- background_indices.json
|   +-- val_fused_embeddings.pt
|   +-- val_labels.pt
|   +-- val_predictions.pt
|   +-- test_fused_embeddings.pt
|   +-- test_labels.pt
|   +-- test_predictions.pt
|
+-- metadata/
    +-- shap_config.json
    +-- additivity_check_results.json
    +-- background_stability_report.json
```

### Naming conventions

- `sample_{id}`: the dataset index of the sample (zero-indexed integer).
- `target{idx}`: 0=food, 1=price, 2=atmosphere, 3=service, 4=overall.
- Target names in file names use underscores: `food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`.
- All PNG files use 200 DPI unless marked as publication-quality (300 DPI).
- All tensor files use `.pt` (PyTorch) format.
- All numerical arrays use `.npy` (NumPy) format.
- All metadata uses `.json` format.

---

## 9. Notebook Design

### Notebook: `notebook/SHAP_Phase4_Analysis.ipynb`

#### Cell 1: Header and Configuration

**Type:** Markdown

**Content:** Title, phase description, experiment identifier, date, model description.

#### Cell 2: Imports and Setup

**Type:** Code

**Actions:**
- Import standard libraries: `torch`, `numpy`, `pandas`, `matplotlib`, `json`, `os`.
- Import `shap`.
- Import project modules: `Models/CrossAttentionFusion`, `Models/TextModel`, `Models/ImageModel`, `src/dataset`, `xai/shap_explainer`.
- Set random seeds (42).
- Set device (cuda/cpu).
- Define experiment path: `experiments/EXP_050C/`.
- Define output directory: `experiments/EXP_050C/xai/shap/`.
- Create output directories.
- Define target names list: `['food_score', 'price_score', 'atmosphere_score', 'service_score', 'overall_satisfaction']`.

**Expected output:** Print device, paths, confirmation of directory creation.

#### Cell 3: Load Model and Checkpoint

**Type:** Code

**Actions:**
- Instantiate TextModel with `vinai/phobert-base-v2`.
- Instantiate ImageModel with `swin_base_patch4_window7_224`.
- Instantiate CrossAttentionFusion with text_model, image_model, num_factors=5.
- Load checkpoint state dict.
- Set model to `eval()` mode.
- Move to device.
- Print model architecture summary (number of parameters, head structure).

**Expected output:** "Model loaded successfully. Head: Sequential(Linear(1024, 512), ReLU, Dropout(0.2), Linear(512, 256), ReLU, Linear(256, 5))". Confirmation that model is in eval mode.

**Parameters:**
- `checkpoint_path`: path to `best_model.pth`.
- `text_model_name`: `'vinai/phobert-base-v2'`.
- `image_model_name`: `'swin_base_patch4_window7_224'`.

#### Cell 4: Load Dataset and Create DataLoaders

**Type:** Code

**Actions:**
- Load tokenizer for PhoBERT.
- Load image processor for Swin-B.
- Create validation dataset and dataloader.
- Create test dataset and dataloader (or a subset for SHAP analysis).
- Print dataset sizes.

**Parameters:**
- `val_csv`: `'data/text/val.csv'`.
- `test_csv`: `'data/text/test.csv'`.
- `batch_size`: 32.
- `max_length`: 256.
- `image_dir`: `'data/image'`.

**Expected output:** Dataset sizes, number of batches.

#### Cell 5: Extract Fused Embeddings

**Type:** Code

**Actions:**
- Call `extract_fused_embeddings(model, val_loader, device)` to get validation embeddings.
- Call `extract_fused_embeddings(model, test_loader, device)` to get test embeddings (or a subset).
- Verify shapes: `[N_val, 1024]`, `[N_test, 1024]`.
- Verify prediction reproduction: compare `model.head(fused)` with original model predictions. Print max absolute error.
- Save embeddings to `.pt` files.

**Expected output:** "Validation embeddings: [N_val, 1024]". "Test embeddings: [N_test, 1024]". "Max prediction reproduction error: < 1e-5". Confirmation of saved files.

#### Cell 6: Select Background Samples

**Type:** Code

**Actions:**
- Call `select_background(val_fused, n_background=100, seed=42)`.
- Save background to `background_fused.pt`.
- Save indices to `background_indices.json`.
- Print background statistics: mean, std, min, max across dimensions.

**Expected output:** "Background samples: [100, 1024]". Statistics summary.

#### Cell 7: SHAP Analysis -- Single Sample Demo

**Type:** Code

**Actions:**
- Select one representative sample from the test set.
- For each of the 5 targets:
  - Create FusionHeadWrapper.
  - Create DeepExplainer.
  - Compute SHAP values.
  - Run additivity check.
  - Compute modality contribution.
  - Print results.
- Display waterfall plot for `overall_satisfaction`.

**Expected output:** For each target: base_value, modality contribution (text-origin %, image-origin %), additivity check pass/fail. One waterfall plot displayed inline.

**Purpose:** Quick sanity check before running full analysis.

#### Cell 8: SHAP Analysis -- All Targets, Multiple Samples

**Type:** Code

**Actions:**
- Select N samples for detailed analysis (e.g., 20-50 samples from test set, including correct and incorrect predictions).
- For each target (loop over 5):
  - Create FusionHeadWrapper.
  - Create DeepExplainer.
  - For each sample:
    - Compute SHAP values.
    - Run additivity check.
    - Compute modality contribution.
    - Save raw SHAP values to `.npy`.
    - Generate waterfall plot (for selected samples).
    - Generate bar plot (for selected samples).
  - Aggregate modality contributions across samples.
  - Generate beeswarm plot.
  - Generate modality contribution chart.
- Generate modality contribution summary chart (all targets).
- Generate modality contribution heatmap.
- Save `modality_contribution_summary.csv`.

**Expected output:** Progress bars, saved file confirmations, inline display of summary chart.

**Parameters:**
- `n_explain_samples`: 20 (for detailed plots), up to 200 (for beeswarm/aggregate).
- `n_plot_samples`: 5 (number of samples to generate individual waterfall/bar plots for).

#### Cell 9: Background Stability Check

**Type:** Code

**Actions:**
- Run SHAP with background sizes 50, 100, 200 on the same 5 samples.
- Compare modality contribution percentages.
- Generate a stability comparison table.

**Expected output:** Table showing that modality contribution changes by < 2 percentage points between 100 and 200 background samples.

#### Cell 10: Modality-Level KernelExplainer Validation

**Type:** Code

**Actions:**
- Run modality-level KernelExplainer (2 features) on the same samples.
- Compare text-origin vs image-origin results with DeepExplainer grouped results.
- Print comparison table.

**Expected output:** Agreement within 5 percentage points between DeepExplainer grouping and KernelExplainer modality-level results.

#### Cell 11: Additivity Verification Summary

**Type:** Code

**Actions:**
- Aggregate all additivity check results.
- Compute: percentage of samples passing (error < 0.01).
- Print summary statistics.

**Expected output:** "Additivity check: X% of samples pass with tolerance 0.01". Maximum error across all samples.

#### Cell 12: Results Summary and Interpretation

**Type:** Markdown + Code

**Actions:**
- Display the modality contribution summary table.
- Display the summary chart.
- Write interpretation of per-target modality balance.
- Compare with expected hypotheses (atmosphere = image-dominant, price = text-dominant, etc.).

**Expected output:** Summary table, interpretation text, comparison with hypotheses.

#### Cell 13: Save Metadata

**Type:** Code

**Actions:**
- Save `shap_config.json` with all configuration parameters.
- Save `additivity_check_results.json` with aggregate results.
- Save `background_stability_report.json`.
- Print completion message with artifact count.

**Expected output:** Confirmation of all saved files. Final artifact count.

---

## 10. Algorithm

### High-level algorithm

```
ALGORITHM: Phase 4 Fusion-level SHAP Analysis

INPUT:
    checkpoint_path       -- path to trained CrossAttentionFusion model
    val_dataloader        -- validation set dataloader
    test_dataloader       -- test set dataloader
    n_background = 100    -- number of background samples
    n_explain = 50        -- number of samples to explain
    targets = [0,1,2,3,4] -- target indices
    text_dim = 512        -- text-origin dimension boundary in fused vector

OUTPUT:
    SHAP values, modality contributions, plots, summary CSV

PROCEDURE:

STEP 1: MODEL PREPARATION
    Load CrossAttentionFusion from checkpoint
    Set model.eval()
    Move to device
    Verify model.head is Sequential(1024->512->ReLU->DO->512->256->ReLU->256->5)

STEP 2: EMBEDDING EXTRACTION
    Register forward pre-hook on model.head to capture input tensor
    For each batch in val_dataloader:
        Run model.forward(input_ids, attention_mask, pixel_values, num_images)
        Collect captured fused vectors [B, 1024]
        Collect labels [B, 5]
        Collect predictions [B, 5]
    Concatenate into val_fused [N_val, 1024], val_labels [N_val, 5], val_preds [N_val, 5]
    Remove hook
    
    VERIFICATION: For random subset of 10 samples:
        Assert model.head(val_fused[i]) == val_preds[i] within tolerance 1e-5
    
    Repeat for test set if needed
    Save all embeddings to .pt files

STEP 3: BACKGROUND SELECTION
    Set random seed = 42
    Select 100 random indices from val_fused
    background = val_fused[selected_indices]  -- [100, 1024]
    Save background and indices

STEP 4: PER-TARGET SHAP COMPUTATION
    For each target_idx in [0, 1, 2, 3, 4]:
        target_name = targets[target_idx]
        
        4a. Create wrapper
            wrapper = FusionHeadWrapper(model.head, target_idx)
            wrapper.eval()
        
        4b. Create explainer
            explainer = shap.DeepExplainer(wrapper, background)
        
        4c. Compute SHAP for each sample
            sample_shap_values = []  -- will be [n_explain, 1024]
            sample_contributions = []
            
            For sample_idx in range(n_explain):
                sample = test_fused[sample_idx].unsqueeze(0)  -- [1, 1024]
                shap_vals = explainer.shap_values(sample)     -- [1, 1024]
                shap_vals = shap_vals[0]                      -- [1024]
                
                ADDITIVITY CHECK:
                    pred = wrapper(sample).item()
                    reconstructed = base_value + sum(shap_vals)
                    error = |pred - reconstructed|
                    Assert error < tolerance (0.01)
                    Log result
                
                MODALITY CONTRIBUTION:
                    text_abs = sum(|shap_vals[0:text_dim]|)
                    image_abs = sum(|shap_vals[text_dim:]|)
                    total = text_abs + image_abs + 1e-8
                    text_pct = 100 * text_abs / total
                    image_pct = 100 * image_abs / total
                    text_signed = sum(shap_vals[0:text_dim])
                    image_signed = sum(shap_vals[text_dim:])
                
                Save shap_vals to .npy
                Append to sample_shap_values
                Append contribution dict to sample_contributions
                
                If sample_idx in selected_plot_samples:
                    Generate waterfall plot
                    Generate bar plot
        
        4d. Aggregate across samples
            mean_text_pct = mean(all text_pct for this target)
            mean_image_pct = mean(all image_pct for this target)
            std_text_pct = std(all text_pct for this target)
            std_image_pct = std(all image_pct for this target)
        
        4e. Generate dataset-level plots
            Generate beeswarm plot from sample_shap_values [n_explain, 1024]
            Generate modality contribution chart
        
        4f. Store target summary
            summary[target_name] = {mean_text_pct, mean_image_pct, std, ...}

STEP 5: CROSS-TARGET SUMMARY
    Generate modality_contribution_summary_all_targets.png
    Generate modality_contribution_heatmap.png
    Save modality_contribution_summary.csv with columns:
        target, mean_text_origin_pct, mean_image_origin_pct,
        std_text_origin_pct, std_image_origin_pct,
        mean_text_signed, mean_image_signed

STEP 6: MODALITY-LEVEL VALIDATION (optional)
    For each target:
        Run KernelExplainer with 2 super-features
        Compare with grouped DeepExplainer results
    Report agreement

STEP 7: SAVE METADATA
    Save shap_config.json:
        checkpoint_path, background_size, n_explain,
        text_dim, explainer_type, model_architecture,
        target_names, timestamp, seed
    Save additivity_check_results.json
    Save background_stability_report.json (if run)

STEP 8: COMPLETION VERIFICATION
    Verify all expected files exist
    Print artifact count and summary
```

### Computational complexity estimate

| Operation | Approximate time | Notes |
|---|---|---|
| Embedding extraction (val set) | 2-5 minutes | Depends on dataset size, GPU |
| Embedding extraction (test set) | 2-5 minutes | Depends on dataset size, GPU |
| DeepExplainer creation (per target) | < 1 second | One-time setup |
| SHAP per sample per target | 0.1-0.5 seconds | DeepExplainer on 1024-d MLP |
| 50 samples x 5 targets | 25-125 seconds | Total SHAP computation |
| Plotting per sample per target | 1-2 seconds | Waterfall + bar |
| Beeswarm per target | 2-5 seconds | Dataset-level |
| Total estimated time | 15-30 minutes | Including all I/O and plotting |

---

## 11. Validation

### 11.1 Additivity check (quantitative, mandatory)

**Specification:** For every sample and every target, verify:

```
prediction == base_value + sum(shap_values)
```

within a tolerance of 0.01 (on the original target scale, e.g., scores typically range 1-10).

**Procedure:**
1. For sample `x` and target `t`:
   - `pred = wrapper(x).item()`
   - `reconstructed = explainer.expected_value + sum(shap_values)`
   - `error = |pred - reconstructed|`
2. Log all errors.
3. Report: percentage passing, mean error, max error.

**Acceptance criteria:** At least 95% of samples pass with tolerance 0.01. If any sample has error > 0.1, investigate the explainer configuration.

### 11.2 Prediction reproduction check (quantitative, mandatory)

**Specification:** Verify that the extracted fused embeddings, when passed through `model.head`, reproduce the original model predictions.

**Procedure:**
1. For 10 random samples from the extracted embeddings:
   - `head_pred = model.head(fused_embedding)`
   - `model_pred = model(input_ids, attention_mask, pixel_values, num_images)`
   - `error = |head_pred - model_pred|`
2. All errors must be < 1e-5 (floating point tolerance).

**Purpose:** Confirms that the hook correctly captures the fused vector.

### 11.3 Modality contribution sanity check (qualitative)

**Specification:** Verify that modality contributions are sensible:
1. Neither text-origin nor image-origin should consistently dominate across ALL 5 targets (which would suggest branch collapse).
2. Modality contributions should vary across targets.
3. Expected patterns:
   - `atmosphere_score`: image-origin should contribute more (visual ambiance cues).
   - `price_score`: text-origin should contribute more (explicit price mentions in reviews).
   - `food_score`: should be balanced (food presentation + food description).
   - `service_score`: text-origin should contribute more (service is described in text).
   - `overall_satisfaction`: should be balanced.

**If expectations are violated:** This is a finding, not a bug. Document and investigate. It may reveal that the model learned different patterns than expected.

### 11.4 Cross-method validation (qualitative)

**Specification:** Compare SHAP modality contribution with ablation results.
1. Zero out text-origin features (dims 0:512) in the fused vector. Run through head. Observe prediction change.
2. Zero out image-origin features (dims 512:1024). Run through head. Observe prediction change.
3. If text-origin SHAP contribution is 70% for `price_score`, then zeroing text-origin features should cause a larger prediction change for `price_score` than zeroing image-origin features.

**Purpose:** Ablation provides independent evidence of modality importance. Agreement with SHAP strengthens both results. Disagreement should be investigated and reported.

### 11.5 Background stability check (quantitative)

**Specification:** Run SHAP with background sizes 50, 100, 200 on 5 identical samples. Compare modality contribution percentages.

**Acceptance criteria:** The difference in modality contribution between background size 100 and 200 should be less than 2 percentage points for all targets.

### 11.6 DeepExplainer vs KernelExplainer agreement (quantitative)

**Specification:** Compare modality contribution from DeepExplainer (summed over feature groups) with modality-level KernelExplainer (2 features) on the same samples.

**Acceptance criteria:** Agreement within 5 percentage points. If disagreement is larger, investigate and report. DeepExplainer uses gradient-based approximation while KernelExplainer uses perturbation, so some disagreement is expected.

### 11.7 Reproducibility check (mandatory)

**Specification:** Run the entire SHAP pipeline twice with identical seeds and background. Verify that:
1. All SHAP values are identical (bitwise, since DeepExplainer is deterministic given fixed inputs).
2. All modality contributions are identical.
3. All plots are visually identical.

---

## 12. Risks -- Fully Analyzed

### R1: SHAP computation speed on 1024 dimensions

**Problem:** KernelExplainer on 1024 features is prohibitively slow because it must evaluate the model at many perturbed input combinations. The number of evaluations scales poorly with the number of features.

**Why it happens:** KernelExplainer estimates SHAP values by solving a weighted linear regression over sampled feature coalitions. For M features, the number of possible coalitions is 2^M. With M=1024, even aggressive sampling (e.g., 2*M+2048 evaluations) requires thousands of model evaluations per sample per target.

**Possible implementation strategies:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A: DeepExplainer | Very fast (gradient-based, single forward+backward pass). Native PyTorch support. No sampling needed. | Approximate. Assumes additive decomposition compatible with DeepLIFT. May not perfectly match Shapley axioms. |
| B: KernelExplainer with modality-level grouping (2 features) | Exact Shapley values for 2 features. Very fast (4 evaluations). Directly answers "text vs image." | Loses all per-feature detail. Cannot show which specific fused dimensions matter. |
| C: PCA dimensionality reduction | Reduces 1024 features to 16-32 principal components. Makes KernelExplainer tractable. | Loses direct interpretability. PCA dimensions do not correspond to modality groups. Adds complexity. |
| D: K-means feature clustering | Groups correlated features. Reduces effective dimensionality. | May split modality groups arbitrarily. Adds tuning complexity. |
| E: GradientExplainer | Faster than Kernel, uses expected gradients. Works with PyTorch. | Still approximate. Less established than DeepExplainer for this use case. |

**Engineering trade-offs:**
- Strategy A (DeepExplainer) is the fastest and most practical. Its approximation quality is well-studied for MLP-type heads. The head `Linear->ReLU->Dropout->Linear->ReLU->Linear` is a textbook MLP that DeepExplainer handles well.
- Strategy B provides an independent validation of modality-level results.
- Strategies C and D add complexity without clear benefits for the modality contribution question.

**Research trade-offs:**
- DeepExplainer SHAP values are not exact Shapley values but are widely accepted in the XAI literature as practical approximations.
- The thesis should acknowledge the approximation in the methodology section.

**FINAL DECISION:** Strategy A (DeepExplainer) as the primary method for per-feature SHAP values. Strategy B (modality-level KernelExplainer with 2 features) as a fast supplementary validation. No PCA or clustering.

**Reason:** DeepExplainer provides fast, per-feature SHAP values that enable both detailed feature analysis and grouped modality analysis. The modality-level KernelExplainer serves as an independent cross-check. The combination answers both "which modality dominates?" (simple answer) and "which fused dimensions matter?" (detailed answer) without excessive computation.

---

### R2: Cross-attended features are not pure modality features

**Problem:** In CrossAttentionFusion, dimensions 0:512 are text features AFTER attending to image (`t_out`), meaning they contain information from BOTH modalities. Similarly, dimensions 512:1024 are image features AFTER attending to text (`i_out`). Grouping 0:512 as "text contribution" is an approximation.

**Why it happens:** Cross-attention is designed to let modalities inform each other. The query comes from one modality, but the keys and values come from the other. The output is a mixture. Specifically:
- `t_out = cross_attn_t2i(Q=text_proj, K=image_proj, V=image_proj)` -- text queries attending to image content.
- `i_out = cross_attn_i2t(Q=image_proj, K=text_proj, V=text_proj)` -- image queries attending to text content.

So `t_out` is actually "text-query attending to image information" and `i_out` is "image-query attending to text information." The modality origin of the query defines which group the output belongs to, but the information content is cross-modal.

**Possible implementation strategies:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A: Group by query origin, acknowledge approximation | Simple. Preserves the 0:512/512:1024 structure. Maps to "text-origin" and "image-origin" terminology. Widely used in multimodal XAI literature. | Does not capture the cross-modal mixing. May understate the true contribution of the attended modality. |
| B: Run SHAP on raw encoder features before cross-attention | Explains pure modality features (768 text + 1024 image). No cross-modal mixing issue. | Different attachment point (before fusion, not after). Requires a different wrapper. Does not explain the cross-attention mechanism itself. Larger dimensionality (1792 vs 1024). |
| C: Use ablation as complementary evidence | Zero out text-origin or image-origin features and measure prediction drop. Independent of SHAP. | Not SHAP-based. Provides different (complementary) information. Does not decompose per-feature. |
| D: Compute SHAP on a ConcatFusion version of the model for comparison | Compare CrossAttention SHAP with ConcatFusion SHAP to understand cross-modal mixing effects. | Requires a separate trained model. Not the best model. Adds experiment complexity. |

**Engineering trade-offs:**
- Strategy A is the simplest and most directly useful. The naming convention "text-origin" vs "image-origin" (instead of "text" vs "image") explicitly communicates the approximation.
- Strategy B provides cleaner modality separation but at a different architectural level and with higher dimensionality.
- Strategy C is orthogonal and complementary.

**Research trade-offs:**
- The thesis can discuss the cross-modal mixing as a known limitation, which actually demonstrates research maturity.
- The modality contribution percentages should be presented with a caveat about cross-attention mixing.
- This limitation is inherent to all cross-attention fusion architectures and is not unique to this project.

**FINAL DECISION:** Strategy A (group by query origin, acknowledge approximation) combined with Strategy C (ablation as complementary evidence).

**Implementation details:**
- Label dimensions 0:512 as "text-origin" and 512:1024 as "image-origin" in all outputs.
- Include a standard caveat in all plots and reports: "In CrossAttentionFusion, text-origin features (dims 0:512) are text queries that attended to image content, and image-origin features (dims 512:1024) are image queries that attended to text content. Grouping is by query origin modality."
- Run ablation (zero-out text-origin, zero-out image-origin) to provide independent evidence of modality importance. Include ablation results alongside SHAP results.

**Reason:** Acknowledging the approximation is scientifically honest and demonstrates understanding of the architecture. The ablation supplement provides independent evidence without the cross-modal mixing confound (at the level of the fused vector). The combination is stronger than either method alone.

---

### R3: Background sample selection

**Problem:** SHAP values are relative to a background distribution. Different background sets can produce different SHAP values and modality contributions. The choice of background samples affects the base value (expected output) and the direction/magnitude of all attributions.

**Why it happens:** SHAP explains deviation from an expected baseline. If the background is not representative of the data distribution, the baseline becomes misleading and SHAP values become unreliable.

**Possible implementation strategies:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A: Random sample from validation set | Simple. Representative if validation set is representative. | May include outliers. Different random draws may give slightly different results. |
| B: K-medoids clustering, select cluster centroids | Maximally diverse background. Resistant to outliers. | Requires clustering step. Over-engineering for this use case. |
| C: Use the full validation set as background | Most representative possible. | Slow for DeepExplainer (memory scales with background size). May exceed GPU memory. |
| D: Use training set statistics (mean vector) as single background sample | Very fast. Simplest possible. | Extremely crude. May not capture distribution shape. Unstable SHAP values. |

**Engineering trade-offs:**
- Strategy A with 100 samples is the standard practice in SHAP documentation and research papers.
- Stability verification (comparing 50, 100, 200) provides confidence that 100 is sufficient.
- Saving the exact background set and indices ensures reproducibility.

**Research trade-offs:**
- The thesis should state the background selection method clearly.
- Using the validation set (not training set) avoids data leakage concerns.
- The stability check is publishable evidence of methodology robustness.

**FINAL DECISION:** Strategy A (random sample of 100 from validation set fused embeddings). Verify stability with 50 and 200. Save background set and indices for reproducibility.

**Reason:** Standard practice, simple, reproducible, and verified stable. The stability check adds rigor without adding significant computation.

---

### R4: Dropout in head during SHAP computation

**Problem:** The CrossAttentionFusion head contains `Dropout(0.2)` at index [2] of the Sequential. If the model is in training mode during SHAP computation, dropout introduces stochasticity: the same input produces different outputs on different calls. This makes SHAP values non-reproducible and violates the deterministic function assumption.

**Why it happens:** Dropout randomly zeroes neurons during training to prevent overfitting. If `model.eval()` is not called, dropout remains active during inference.

**Possible implementation strategies:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A: Always use model.eval() | Standard practice. Disables dropout and batchnorm training behavior. Deterministic outputs. | None for this use case. |
| B: Replace dropout with identity layer for SHAP | Explicit removal. Guarantees no stochastic behavior. | Modifies the model. Unnecessary if eval() is used. |

**FINAL DECISION:** Strategy A. Always call `model.eval()` before any SHAP computation. Verify determinism by running the same sample twice and checking that outputs are identical.

**Reason:** `model.eval()` is the standard and sufficient approach. The additivity check will also fail if dropout is active, providing an additional safety net.

---

### R5: SHAP for regression vs classification

**Problem:** SHAP explanation for regression output is fundamentally different from classification. In classification, SHAP explains log-odds or probabilities. In regression, SHAP explains the raw scalar output directly. This affects the wrapper design and interpretation.

**Why it happens:** The model predicts raw scores (e.g., food_score on a 1-10 scale), not class probabilities. The `FusionHeadWrapper` returns a single scalar per sample, not a probability distribution.

**Possible implementation strategies:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A: Explain raw scalar output directly | Natural for regression. SHAP values are in the same units as the target score. Interpretable: "this feature increased the food_score by 0.3 points." | No normalization. Scale differs across targets. |
| B: Normalize output to [0,1] before SHAP | Makes SHAP values comparable across targets. | Changes the function being explained. Less interpretable: "this feature increased the normalized score by 0.03" does not map to a human-understandable scale. |

**FINAL DECISION:** Strategy A. Explain the raw scalar output directly for each target. SHAP values will be in the same units as the predicted scores.

**Reason:** Raw scalar output is the most interpretable for regression. The base value will be a meaningful average score (e.g., "base food_score = 6.5"), and SHAP values show how much each feature moves the score from that base. When comparing across targets, use normalized modality contribution percentages (which are already unit-free).

---

### R6: Memory constraints for embedding storage and SHAP computation

**Problem:** Storing fused embeddings for many samples and computing SHAP values on them requires memory management. For a dataset of 10,000 samples, fused embeddings require 10000 x 1024 x 4 bytes = ~40 MB, which is manageable. But GPU memory during DeepExplainer computation depends on background size and batch processing.

**Why it happens:** DeepExplainer internally processes background samples during setup and processes input samples during `shap_values()`. For 100 background samples of dimension 1024, the internal representation is moderate but can grow if many samples are explained at once.

**Possible implementation strategies:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A: Extract embeddings in batches, store as .pt files, process SHAP on CPU | Simple. Low GPU memory. Reproducible. | CPU SHAP is slower than GPU SHAP. |
| B: Extract embeddings in batches, process SHAP on GPU | Fast. Uses GPU acceleration. | Higher GPU memory. May require batch-processing SHAP too. |
| C: Process everything end-to-end on GPU | Fastest possible. | Highest memory requirement. May OOM on consumer GPUs. |

**Engineering trade-offs:**
- Strategy A is safest and most portable across hardware.
- Strategy B is faster and feasible for the MLP head (small model, low memory).
- The MLP head (`Linear(1024->512)->ReLU->Dropout->Linear(512->256)->ReLU->Linear(256->5)`) is very small. Even on CPU, SHAP computation for this head is fast.

**FINAL DECISION:** Strategy B. Extract embeddings on GPU in batches, then compute SHAP on GPU if memory permits, falling back to CPU if needed. The head is small enough that either works well.

**Reason:** The MLP head is tiny (~800K parameters). Memory is not a serious concern for this architecture. The main bottleneck is embedding extraction (which runs the full model), not SHAP computation.

---

### R7: SHAP library version compatibility

**Problem:** The `shap` library has gone through several API changes. DeepExplainer behavior, return formats, and plotting APIs may differ between versions. The `shap.Explanation` object and `shap.plots.waterfall()` API were introduced in later versions.

**Why it happens:** SHAP is an actively maintained library with breaking changes between major versions.

**Possible implementation strategies:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A: Pin to a known working version (e.g., shap>=0.42, <0.45) | Reproducible. Known behavior. | May miss newer features or fixes. |
| B: Write version-agnostic code with fallbacks | Works across versions. | More complex code. Harder to test. |
| C: Use latest version and adapt | Latest features. | May break on future updates. |

**FINAL DECISION:** Strategy A. Pin to `shap>=0.42` and test with the pinned version. Document the version in `shap_config.json` metadata.

**Implementation notes:**
- `shap.DeepExplainer` is stable across versions 0.40+.
- `shap.Explanation` object and `shap.plots.waterfall` require `shap>=0.40`.
- `shap.plots.beeswarm` requires `shap>=0.40`.
- Record `shap.__version__` in metadata.

---

### R8: Multi-output SHAP return format ambiguity

**Problem:** When the FusionHeadWrapper returns `[B, 1]` (a 2D tensor with 1 column), DeepExplainer may return SHAP values in different formats depending on how the output is interpreted:
- If interpreted as single output: returns `numpy.ndarray` of shape `[N, 1024]`.
- If interpreted as multi-output with 1 class: returns a list of length 1, where `list[0]` has shape `[N, 1024]`.

**Why it happens:** SHAP's DeepExplainer checks the output dimensionality and may wrap results in a list for multi-dimensional outputs.

**FINAL DECISION:** Handle both cases in `compute_shap_deep()`. Check if the return value is a list. If so, extract `[0]`. If it is already a numpy array, use directly. Always verify the final shape is `[N, 1024]`.

**Implementation pseudocode:**
```
shap_values = explainer.shap_values(sample_fused)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
assert shap_values.shape == (n_samples, 1024)
```

---

## 13. Best Practices

### Deterministic execution

1. Set `torch.manual_seed(42)`, `numpy.random.seed(42)`, and `random.seed(42)` at the start of every script and notebook.
2. Use `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` during SHAP computation.
3. Always call `model.eval()` before any SHAP-related computation.
4. Never use `torch.no_grad()` around the wrapper during DeepExplainer computation -- DeepExplainer needs gradients.

### Artifact naming

1. Follow the naming convention defined in Section 8 consistently.
2. Include target index AND target name in file names for unambiguous identification.
3. Use zero-padded sample IDs if dataset size exceeds 999 samples (e.g., `sample_0042`).

### Checkpoint handling

1. Load checkpoint once at the start. Never modify model weights during SHAP.
2. Verify the checkpoint loads correctly by comparing predictions on a known sample.
3. Store the checkpoint path in metadata for traceability.

### Memory optimization

1. Extract embeddings in batches (batch_size=32). Do not load the entire dataset at once.
2. Move extracted embeddings to CPU immediately after capturing via hook.
3. Delete the dataloader and intermediate tensors after extraction is complete.
4. For SHAP computation, process samples one at a time or in small batches if memory is constrained.

### Logging

1. Use Python `logging` module, not `print()`, for production code.
2. Log: sample ID, target, SHAP computation time, additivity error, modality contribution.
3. Save logs alongside artifacts for debugging.

### Figure consistency

1. Use identical color palette across all SHAP plots and all XAI phases:
   - Text-origin: `#1b9e77` (teal)
   - Image-origin: `#d95f02` (orange)
2. Use consistent figure sizes: waterfall and bar plots at (10, 6), summary plots at (12, 8), modality charts at (8, 5).
3. Use consistent font sizes: title 14pt, axis labels 12pt, tick labels 10pt.
4. Always include target name in plot titles.
5. Save all plots with `bbox_inches='tight'` to avoid clipping.

### Configuration management

1. All configurable parameters (background size, text_dim, color palette, DPI, tolerance thresholds) should be defined in `xai/shap_config.py` or passed as arguments, never hardcoded in the computation functions.
2. Save all configuration to `shap_config.json` at the end of the pipeline.
3. Include library versions (`shap`, `torch`, `numpy`) in metadata.

### Data integrity

1. Always save raw SHAP values as `.npy` files alongside visual plots.
2. Save fused embeddings, labels, and predictions as `.pt` files.
3. Include background samples and indices for full reproducibility.
4. Any researcher should be able to reproduce all plots from the saved raw values without re-running the model.

### Batch processing pattern

1. When generating plots for many samples, use a progress bar (`tqdm`).
2. Flush matplotlib figures after each save to prevent memory accumulation.
3. Process one target at a time to limit active memory.
4. Save intermediate results frequently so that a crash does not lose all progress.

---

## 14. Deliverables

### Code deliverables

| Deliverable | Path | Description |
|---|---|---|
| SHAP explainer module | `xai/shap_explainer.py` | All classes and functions for SHAP analysis |
| SHAP configuration | `xai/shap_config.py` | Configuration constants |
| Analysis notebook | `notebook/SHAP_Phase4_Analysis.ipynb` | Complete interactive analysis |

### Per-sample artifacts (for each selected sample, for each of 5 targets)

| Artifact | Format | Quantity |
|---|---|---|
| Raw SHAP values | `.npy` [1024] | n_samples x 5 targets |
| Waterfall plot | `.png` | n_plot_samples x 5 targets |
| Bar plot | `.png` | n_plot_samples x 5 targets |

### Per-target artifacts (for each of 5 targets)

| Artifact | Format | Quantity |
|---|---|---|
| Modality contribution chart | `.png` | 5 |
| Beeswarm/summary plot | `.png` | 5 |

### Dataset-level artifacts

| Artifact | Format | Quantity |
|---|---|---|
| Modality contribution summary (all targets) | `.png` | 1 |
| Modality contribution heatmap | `.png` | 1 |
| Modality contribution summary CSV | `.csv` | 1 |

### Raw data artifacts

| Artifact | Format | Description |
|---|---|---|
| Background fused embeddings | `.pt` [100, 1024] | 1 file |
| Background indices | `.json` | 1 file |
| Validation fused embeddings | `.pt` [N_val, 1024] | 1 file |
| Validation labels | `.pt` [N_val, 5] | 1 file |
| Validation predictions | `.pt` [N_val, 5] | 1 file |
| Test fused embeddings | `.pt` [N_test, 1024] | 1 file |
| Test labels | `.pt` [N_test, 5] | 1 file |
| Test predictions | `.pt` [N_test, 5] | 1 file |

### Metadata artifacts

| Artifact | Format | Description |
|---|---|---|
| SHAP configuration | `.json` | Full config including versions, paths, parameters |
| Additivity check results | `.json` | Pass rate, mean error, max error per target |
| Background stability report | `.json` | Comparison across background sizes |

### Summary deliverables for thesis

| Deliverable | Description |
|---|---|
| Modality contribution summary table | Table with text-origin % and image-origin % per target, with standard deviations |
| Modality contribution summary chart | Publication-quality grouped bar chart |
| 2-3 sample waterfall plots | For case study section |
| Modality dominance profile | Textual summary of per-target modality balance |

---

## 15. Thesis Usage

### Central thesis question answered

Phase 4 directly answers the thesis-level question: **"Was the model more influenced by image or text?"**

This is the only phase that provides quantitative evidence for modality contribution. The answer is not a single number but a per-target profile that shows the model uses modalities differently for different quality aspects.

### Per-target modality profile (expected, subject to empirical verification)

| Target | Expected text-origin % | Expected image-origin % | Reasoning |
|---|---|---|---|
| `food_score` | 40-55% | 45-60% | Balanced: food description in text + food presentation in image |
| `price_score` | 55-75% | 25-45% | Text-dominant: reviews explicitly mention price, images rarely show prices |
| `atmosphere_score` | 30-45% | 55-70% | Image-dominant: restaurant interior, lighting, decor are visual |
| `service_score` | 55-75% | 25-45% | Text-dominant: service quality is described in words, rarely visible |
| `overall_satisfaction` | 45-55% | 45-55% | Balanced: overall judgment integrates both visual and textual evidence |

**Important:** These are hypothesized expectations. Actual results may differ and should be reported honestly regardless of whether they match expectations. Unexpected results are research findings, not failures.

### Results chapter usage

1. **Modality contribution summary table:** A central table in the Results chapter showing mean text-origin and image-origin percentages per target. This table quantifies the multimodal behavior of the model.

2. **Modality contribution chart:** A figure showing the side-by-side comparison. This is likely to be one of the most-cited figures in the thesis.

3. **Interpretation paragraph:** For each target, discuss whether the modality balance aligns with domain knowledge (e.g., "atmosphere_score is indeed image-dominant, consistent with the expectation that restaurant ambiance is primarily a visual attribute").

### Discussion chapter usage

1. **Cross-method validation:** Compare SHAP modality contributions with Grad-CAM findings (Phase 2) and attention findings (Phase 3). If SHAP shows image dominance for atmosphere_score, Grad-CAM should highlight interior/ambiance regions, and attention should show less strong activation for atmosphere-related text tokens.

2. **Branch collapse analysis:** If any target shows > 90% contribution from one modality, discuss whether this indicates branch collapse or is a genuine data-driven pattern.

3. **Limitations paragraph:** Discuss the cross-attended feature approximation (R2). Acknowledge that "text-origin" features in CrossAttentionFusion contain image information and vice versa.

### Case study usage (Phase 6 integration)

1. Waterfall plots for 2-3 selected samples showing how features push the prediction above or below the baseline.
2. Modality contribution for these specific samples, compared with Grad-CAM overlays and attention heatmaps from the same samples.
3. One "correct prediction" case study and one "incorrect prediction" case study, showing how modality contribution differs.

### Defense presentation usage

1. **Key slide:** Modality contribution summary chart with all 5 targets. This answers the committee's likely question: "Does your model actually use both modalities?"
2. **Supporting slide:** One waterfall plot for a representative sample, showing the SHAP decomposition.
3. **Defense talking point:** "SHAP analysis at the fusion level revealed that the model uses both modalities but with target-specific balance. For atmosphere_score, image-origin features contributed X%, consistent with the visual nature of ambiance. For price_score, text-origin features contributed Y%, consistent with explicit price mentions in reviews."

### Journal paper usage

1. Modality contribution table and chart are publication-ready.
2. The cross-attention grouping caveat demonstrates methodological rigor.
3. Per-target modality profiles contribute to the understanding of multimodal fusion behavior in quality assessment tasks.
4. The comparison between DeepExplainer and modality-level KernelExplainer validates the methodology.

---

## 16. Phase Completion Checklist

### Infrastructure

- [ ] `xai/shap_explainer.py` exists and contains all specified classes and functions.
- [ ] `xai/shap_config.py` exists with all configuration constants.
- [ ] `notebook/SHAP_Phase4_Analysis.ipynb` exists and runs end-to-end without errors.

### Model and data preparation

- [ ] Checkpoint loads successfully and `model.eval()` is confirmed.
- [ ] Forward hook on `model.head` correctly captures fused embeddings of shape `[B, 1024]`.
- [ ] Prediction reproduction check passes: `|model.head(fused) - model(inputs)| < 1e-5`.
- [ ] Validation set fused embeddings extracted and saved to `val_fused_embeddings.pt`.
- [ ] Test set fused embeddings extracted and saved to `test_fused_embeddings.pt`.

### Background selection

- [ ] Background set of 100 samples selected from validation embeddings.
- [ ] Background saved to `background_fused.pt`.
- [ ] Background indices saved to `background_indices.json`.
- [ ] Background stability check completed (50 vs 100 vs 200): difference < 2 percentage points.

### SHAP computation

- [ ] DeepExplainer runs without errors for all 5 targets.
- [ ] SHAP values computed for at least 20 samples per target.
- [ ] All raw SHAP values saved as `.npy` files.
- [ ] Additivity check passes for at least 95% of samples with tolerance 0.01.
- [ ] Modality contribution computed for all explained samples.

### Modality-level validation

- [ ] KernelExplainer (2-feature) runs for at least 5 samples per target.
- [ ] Agreement with DeepExplainer grouping within 5 percentage points documented.

### Ablation validation

- [ ] Zero-out text-origin ablation completed for all 5 targets.
- [ ] Zero-out image-origin ablation completed for all 5 targets.
- [ ] Ablation results consistent with SHAP modality contribution direction.

### Plots and figures

- [ ] Waterfall plots generated for at least 5 selected samples, all 5 targets.
- [ ] Bar plots generated for at least 5 selected samples, all 5 targets.
- [ ] Beeswarm plots generated for all 5 targets.
- [ ] Per-target modality contribution charts generated for all 5 targets.
- [ ] Cross-target modality contribution summary chart generated.
- [ ] Modality contribution heatmap generated.
- [ ] All plots use consistent color palette and font sizes.
- [ ] All plots include descriptive titles with target names.

### Summary outputs

- [ ] `modality_contribution_summary.csv` exists with columns: target, mean_text_origin_pct, mean_image_origin_pct, std_text_origin_pct, std_image_origin_pct, mean_text_signed, mean_image_signed.
- [ ] Summary CSV contains rows for all 5 targets.
- [ ] Modality contribution percentages are reasonable (no target at 100%/0% unless branch collapse is a genuine finding).

### Metadata and reproducibility

- [ ] `shap_config.json` saved with: checkpoint_path, background_size, n_explain, text_dim, explainer_type, shap_version, torch_version, timestamp, seed.
- [ ] `additivity_check_results.json` saved with: per-target pass rate, mean error, max error.
- [ ] `background_stability_report.json` saved with: comparison across background sizes.
- [ ] All artifacts stored under `experiments/EXP_050C/xai/shap/` following the defined folder structure.

### Reproducibility verification

- [ ] Running the notebook twice with identical seeds produces identical SHAP values.
- [ ] Running the notebook twice produces identical modality contribution percentages.
- [ ] All raw values (.npy, .pt, .json) can regenerate all plots without re-running the model.

### Interpretation and thesis readiness

- [ ] Per-target modality dominance profile documented (text-dominant, image-dominant, or balanced for each target).
- [ ] Results compared with domain expectations and discrepancies discussed.
- [ ] Cross-attention grouping caveat documented in interpretive text.
- [ ] At least one "correct prediction" and one "incorrect prediction" sample analyzed for case study usage.
- [ ] Summary table ready for thesis Results chapter insertion.
- [ ] Summary chart ready for thesis figure insertion (publication DPI: 300).
