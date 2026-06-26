# Phase 5: LIME Local Explanation for Multimodal Quality Assessment

## Implementation Proposal Document

**Phase:** 5 of 8
**XAI Method:** LIME (Local Interpretable Model-agnostic Explanations)
**Target:** Full multimodal system (black-box wrapper)
**Best Model:** Swin-B + PhoBERT (`vinai/phobert-base-v2`) + CrossAttentionFusion + LogCosh
**Priority:** LOWEST among the four XAI methods -- can be moved to appendix or skipped if time-constrained
**Status:** Specification Complete -- Ready for Implementation

---

## 1. Purpose

### 1.1 Why This Phase Exists

Phases 2, 3, and 4 explain the multimodal system by looking *inside* the model architecture:

- **Phase 2 (Grad-CAM):** uses gradients flowing back through the Swin-B encoder to highlight spatial image regions that support each target score.
- **Phase 3 (Attention):** extracts self-attention weight matrices from PhoBERT to visualize token-to-token information flow within the text encoder.
- **Phase 4 (SHAP):** measures the contribution of fused embedding features (image dimensions 0:1024, text dimensions 1024:1792) to each prediction target.

All three methods are *model-internals-based*. They require access to gradients, attention tensors, or intermediate embeddings. A thesis examiner can legitimately ask: "What happens if you test the model from the outside? If you remove a food-related image region, does the food score actually drop? If you remove the word 'ngon' from the review, does the food score decrease?"

LIME answers these questions. It treats the entire multimodal system as a black box, perturbs human-interpretable input units (superpixels for images, words for text), observes how predictions change, and fits a local linear surrogate to quantify which input units matter most near a specific sample.

Phase 5 provides **perturbation-based local validation** of findings from the other three methods. It is the only XAI method in this system that operates entirely from the input side without requiring any internal model access.

### 1.2 Research Motivation

1. **Independent validation of gradient-based findings:** Grad-CAM may highlight a region because of gradient artifacts or batch normalization effects. If LIME independently confirms that the same region locally affects the prediction when removed, the evidence is substantially strengthened.
2. **Independent validation of attention-based findings:** Phase 3 explicitly documents that "attention is not explanation." LIME provides the perturbation-based complement: if attention highlights the word "ngon" and LIME confirms that removing "ngon" locally decreases `food_score`, then the two methods mutually reinforce each other.
3. **Model-agnostic validation:** LIME does not depend on the model being differentiable or having accessible internal representations. It validates that the model, viewed as a function from inputs to outputs, behaves as expected.
4. **Multi-method triangulation for thesis defense:** A thesis claiming explainability with only gradient-based methods is vulnerable. Adding perturbation-based evidence (LIME) alongside gradient-based (Grad-CAM), attention-based (Phase 3), and attribution-based (SHAP) evidence creates a four-level explanatory framework that examiners will find difficult to challenge.

### 1.3 Engineering Motivation

1. **Reuse of Phase 1 infrastructure:** LIME wrappers build directly on `load_model()`, `load_single_sample()`, `get_prediction()`, `get_image_processor()`, and `get_tokenizer()` from `xai/utils.py`.
2. **Simple integration:** The `lime` Python package provides `LimeImageExplainer` and `LimeTextExplainer` out of the box. The main engineering effort is writing correct prediction wrapper functions that bridge LIME's expected interface with the multimodal model.
3. **Artifact reuse:** LIME outputs (superpixel weight overlays, word importance bar charts, HTML reports) directly serve Phase 6 (Case Studies), Phase 7 (Report Generation), and Phase 8 (Thesis Visualization).

### 1.4 Priority and Scope Decision

LIME is the **lowest priority** XAI method in this thesis. The core explanatory framework is already complete with Grad-CAM (image evidence), Attention (text information flow), and SHAP (fusion-level contribution). LIME adds valuable perturbation-based validation but is not essential for the thesis argument.

**Minimum viable scope:** Run LIME on 5--10 carefully selected case study samples, generating both image and text explanations for all 5 target scores. This produces sufficient material for one thesis subsection and 2--3 defense slides.

**If time-constrained:** LIME results can be moved entirely to the thesis appendix without weakening the core claim. The defense answer becomes: "I prioritized Grad-CAM, Attention, and SHAP for the main analysis. LIME perturbation checks are available in the appendix for selected samples."

---

## 2. Objectives

### 2.1 Research Objectives

| ID | Objective | Success Criterion |
|----|-----------|-------------------|
| R1 | Image LIME: identify which superpixel regions locally support or oppose each target score | For food-related samples, food-presentation superpixels have positive LIME weights for `food_score` |
| R2 | Text LIME: identify which Vietnamese words locally support or oppose each target score | The word "ngon" has positive weight for `food_score`; "gia" and "cao" have negative weight for `price_score` |
| R3 | Cross-validate LIME image results with Grad-CAM (Phase 2) findings | Positive LIME superpixels visually overlap with Grad-CAM hot regions for the same sample and target |
| R4 | Cross-validate LIME text results with Attention (Phase 3) findings | LIME top-importance words appear among Attention top-attended tokens for the same sample |
| R5 | Demonstrate that the model reacts to expected perturbations | Removing known important cues (food words, food image regions) changes the corresponding target score |
| R6 | Assess LIME explanation stability across random seeds | Top-K features appearing in 2/3 or 3/3 runs indicate stable explanations |

### 2.2 Engineering Objectives

| ID | Objective | Success Criterion |
|----|-----------|-------------------|
| E1 | Create `xai/lime_explainer.py` with image and text LIME wrapper functions | Module runs without error, produces valid LIME explanation objects |
| E2 | Implement correct regression-to-pseudo-classification conversion | Sigmoid mapping produces [N, 2] output compatible with `LimeImageExplainer` and `LimeTextExplainer` |
| E3 | Implement image preprocessing that exactly matches training pipeline | LIME wrapper produces identical model output as direct inference for the unperturbed original image |
| E4 | Handle multi-image reviews correctly in image LIME | First real image is explained; remaining images are kept fixed |
| E5 | Handle Vietnamese syllable-level text splitting in text LIME | Default whitespace splitting produces syllable-level features consistent with PhoBERT tokenization |
| E6 | Generate per-sample, per-target artifacts: superpixel overlays, word bar charts, HTML, JSON | All artifacts saved to `experiments/EXP_XXX/xai/lime/` following the folder convention |
| E7 | Support reproducibility via fixed random seeds | Same seed produces identical LIME explanations |
| E8 | Create notebook `xai/notebooks/Phase5_LIME.ipynb` | Notebook runs end-to-end, generates all artifacts for selected samples |

### 2.3 Expected Contributions

1. First perturbation-based local explanation for this multimodal Vietnamese review quality assessment system.
2. Documented cross-validation between LIME and Grad-CAM for image explanations.
3. Documented cross-validation between LIME and Attention for text explanations.
4. Reusable LIME prediction wrappers that correctly bridge the `lime` package interface with the CrossAttentionFusion model.

---

## 3. Inputs

### 3.1 Model Checkpoint

| Input | Path | Description |
|-------|------|-------------|
| Best model checkpoint | `experiments/EXP_060A_bestsequential_full_configuration/best_model_train_fusion.pth` | Trained CrossAttentionFusion weights. Keys: `model_state_dict`, `args`, `best_mean_mae`, `epoch`, etc. |

### 3.2 Dataset Files

| Input | Path | Description |
|-------|------|-------------|
| Test CSV | `data/text/test.csv` | Columns: `comment_clean`, `image_url`, `food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction` |
| Validation CSV | `data/text/val.csv` | Alternative sample source |
| Image cache | `data/image/` | JPEG files named by MD5 hash of original URL |

### 3.3 Phase 1 Infrastructure

| Input | Path | Description |
|-------|------|-------------|
| XAI utilities | `xai/utils.py` | `load_model()`, `load_single_sample()`, `get_prediction()`, `get_image_processor()`, `get_tokenizer()`, `save_figure()`, `save_raw_values()`, `set_seed()`, `get_device()` |
| XAI config | `xai/config.py` | `TARGET_NAMES`, `TARGET_INDICES`, `FACTOR_NAMES`, `DISPLAY_NAMES`, `COLOR_SCHEMES`, `DEFAULT_SEED`, `DEFAULT_DPI`, `THESIS_DPI`, `DEFAULT_MAX_LENGTH`, `DEFAULT_MAX_IMAGES`, `BEST_TEXT_MODEL`, `BEST_IMAGE_MODEL`, `SCORE_RANGE`, `INDEX_TO_FACTOR`, `FACTOR_TO_INDEX`, `FACTOR_TO_DISPLAY` |

### 3.4 Phase 2 and Phase 3 Results (for Cross-Validation)

| Input | Path | Description |
|-------|------|-------------|
| Grad-CAM heatmaps | `experiments/EXP_XXX/xai/gradcam/sample_{id}_target{idx}_*.png` | Heatmap overlays for visual comparison |
| Attention results | `experiments/EXP_XXX/xai/attention/sample_{id}_*.json` | Token importance rankings for textual comparison |

### 3.5 Model Configuration Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Text encoder | `vinai/phobert-base-v2` | PhoBERT, 768-dim, syllable-level BPE |
| Image encoder | `swin_base_patch4_window7_224` | Swin-B via timm, 1024-dim features |
| Fusion type | `cross_attention` | CrossAttentionFusion with bidirectional cross-attention |
| Max text length | 256 | `Config.py` default |
| Max images | 4 | `MultimodalDataset`, padded with black 224x224 |
| Image resolution | 224x224 | Swin-B input requirement |
| Num targets | 5 | food(0), price(1), atmosphere(2), service(3), overall(4) |

### 3.6 External Library

| Library | Version | Usage |
|---------|---------|-------|
| `lime` | latest stable (>=0.2.0) | `lime.lime_image.LimeImageExplainer`, `lime.lime_text.LimeTextExplainer` |
| `scikit-image` | required by `lime` | Superpixel segmentation (quickshift or SLIC) |

---

## 4. Outputs

### 4.1 Python Module

| Output | Path | Description |
|--------|------|-------------|
| LIME explainer module | `xai/lime_explainer.py` | Contains `ImageLimePredictFn`, `TextLimePredictFn`, `run_lime_image()`, `run_lime_text()`, `save_lime_image_explanation()`, `save_lime_text_explanation()`, `save_lime_text_html()` |

### 4.2 Notebook

| Output | Path | Description |
|--------|------|-------------|
| LIME analysis notebook | `xai/notebooks/Phase5_LIME.ipynb` | End-to-end LIME analysis for selected case study samples |

### 4.3 Per-Sample Artifacts (for Each Sample and Each Target)

| Artifact | Path Pattern | Format | Description |
|----------|-------------|--------|-------------|
| Positive superpixel overlay | `experiments/EXP_XXX/xai/lime/sample_{id}_target{idx}_{name}_image_positive.png` | PNG (300 DPI) | Superpixels with positive LIME weights highlighted on original image |
| Negative superpixel overlay | `experiments/EXP_XXX/xai/lime/sample_{id}_target{idx}_{name}_image_negative.png` | PNG (300 DPI) | Superpixels with negative LIME weights highlighted on original image |
| Combined superpixel overlay | `experiments/EXP_XXX/xai/lime/sample_{id}_target{idx}_{name}_image_combined.png` | PNG (300 DPI) | Both positive and negative superpixels with color coding |
| Word importance bar chart | `experiments/EXP_XXX/xai/lime/sample_{id}_target{idx}_{name}_text_bar.png` | PNG (300 DPI) | Horizontal bar chart of top word weights (green = positive, red = negative) |
| Interactive text HTML | `experiments/EXP_XXX/xai/lime/sample_{id}_target{idx}_{name}_text.html` | HTML | `explanation.as_html()` output with highlighted words |
| Superpixel weights | `experiments/EXP_XXX/xai/lime/raw/sample_{id}_target{idx}_image_weights.json` | JSON | Mapping: superpixel_id -> weight |
| Word weights | `experiments/EXP_XXX/xai/lime/raw/sample_{id}_target{idx}_text_weights.json` | JSON | Mapping: word -> weight |
| LIME config | `experiments/EXP_XXX/xai/lime/raw/sample_{id}_target{idx}_lime_config.json` | JSON | `num_samples`, `num_features`, `seed`, `segmentation_fn`, timestamp |
| Sample metadata | `experiments/EXP_XXX/xai/lime/metadata/sample_{id}_lime_metadata.json` | JSON | Sample index, text, predictions, ground truth, number of images, image paths |

### 4.4 Stability Analysis Artifacts (When Multi-Seed Runs Are Performed)

| Artifact | Path Pattern | Format | Description |
|----------|-------------|--------|-------------|
| Stability report | `experiments/EXP_XXX/xai/lime/stability/sample_{id}_target{idx}_stability.json` | JSON | Per-seed top-K features, consensus features, Jaccard similarity across seeds |

---

## 5. Architecture Attachment Point

### 5.1 Black-Box Wrapping Strategy

LIME is fundamentally different from Grad-CAM, Attention, and SHAP in its attachment point. It does **not** attach to any internal layer, tensor, or gradient. Instead, it wraps the **entire model** as a black-box function:

```
LIME Image Attachment:
======================
                                     FIXED (frozen during perturbation)
                                     ┌─────────────────────────────┐
Perturbed Images [N, H, W, 3]       │  input_ids [1, 256]         │
  │                                  │  attention_mask [1, 256]    │
  │  (numpy -> PIL -> TimmProcessor) │  num_images [1]             │
  │                                  └──────────┬──────────────────┘
  v                                             │
┌──────────────────────────────────────────────────────────────────┐
│                    CrossAttentionFusion                          │
│                                                                  │
│  ImageModel(Swin-B)                TextModel(PhoBERT)            │
│  pixel_values -> [B, 1024]         input_ids -> [B, 768]        │
│                    │                         │                    │
│                    v                         v                    │
│              image_proj(512)           text_proj(512)             │
│                    │                         │                    │
│                    v                         v                    │
│              cross_attn_t2i          cross_attn_i2t               │
│                    │                         │                    │
│                    └─────────┬───────────────┘                    │
│                              v                                    │
│                    cat -> [B, 1024] -> head -> [B, 5]             │
└──────────────────────────────────────────────┬───────────────────┘
                                               │
                                               v
                                    score[score_index]
                                               │
                                               v
                                    sigmoid -> [1-p, p] -> [N, 2]
                                    (pseudo-classification output)


LIME Text Attachment:
=====================
                                     FIXED (frozen during perturbation)
                                     ┌─────────────────────────────┐
Perturbed Text Strings [N]           │  pixel_values [1, 4, 3, 224, 224] │
  │                                  │  num_images [1]             │
  │  (tokenize with PhoBERT)         └──────────┬──────────────────┘
  │                                             │
  v                                             │
┌──────────────────────────────────────────────────────────────────┐
│                    CrossAttentionFusion                          │
│    (same forward pass as above)                                  │
└──────────────────────────────────────────────┬───────────────────┘
                                               │
                                               v
                                    score[score_index]
                                               │
                                               v
                                    sigmoid -> [1-p, p] -> [N, 2]
```

### 5.2 Key Architectural Considerations

1. **Image LIME perturbs at the numpy/PIL level** (before TimmProcessor), not at the tensor level. The predict function must convert LIME's numpy images back through the full preprocessing pipeline.

2. **Text LIME perturbs at the string level** (before PhoBERT tokenizer), not at the token level. LIME removes words from the string, and the predict function must re-tokenize each perturbed string.

3. **Multi-image handling:** The model expects `pixel_values [B, N, 3, 224, 224]` with up to 4 images. For Image LIME, only the primary image (first real image) is perturbed. The remaining images are held fixed at their original preprocessed values. The perturbed primary image is placed at index 0 in the multi-image tensor.

4. **CrossAttentionFusion forward signature:** `forward(input_ids, attention_mask, pixel_values, num_images=None)`. The LIME wrappers must provide all four arguments.

5. **Regression output conversion:** The model outputs `[B, 5]` regression scores. LIME expects classification-like `[N, num_classes]` probabilities. The wrapper applies sigmoid to the target score to create a pseudo two-class output.

### 5.3 Comparison with Other Phases' Attachment Points

| Phase | Method | Attachment Level | What Is Accessed |
|-------|--------|-----------------|-----------------|
| 2 | Grad-CAM | Swin-B internal spatial feature map | Gradients + activations `[B, 1024, 7, 7]` |
| 3 | Attention | PhoBERT internal attention matrices | `output_attentions=True` -> `tuple(12 x [B, 12, L, L])` |
| 4 | SHAP | Fusion MLP input (fused embedding) | `[B, 1024]` after cross-attention concatenation |
| **5** | **LIME** | **Full model black-box wrapper** | **Input numpy images or text strings -> output scores** |

LIME is the only method that operates entirely from the input space. This independence is precisely what makes it valuable as a cross-validation tool.

---

## 6. Detailed Implementation Plan

### Step A: Create `xai/lime_explainer.py` -- Module Structure

**Purpose:** Single Python module containing all LIME-specific code. Imports infrastructure from `xai/utils.py` and constants from `xai/config.py`.

**Module-level imports:**
- `numpy`, `torch`, `PIL.Image`
- `lime.lime_image.LimeImageExplainer`
- `lime.lime_text.LimeTextExplainer`
- `matplotlib.pyplot`, `matplotlib.colors`
- `skimage.segmentation.mark_boundaries`
- `json`, `os`, `logging`, `datetime`
- From `xai.config`: `TARGET_NAMES`, `TARGET_INDICES`, `INDEX_TO_FACTOR`, `FACTOR_TO_DISPLAY`, `SCORE_RANGE`, `DEFAULT_SEED`, `THESIS_DPI`, `COLOR_SCHEMES`
- From `xai.utils`: `save_figure`, `save_raw_values`, `get_device`

**Module-level logger:** `logger = logging.getLogger(__name__)`

---

### Step B: Implement `ImageLimePredictFn` Class

**Purpose:** Callable class that LIME calls with batches of perturbed numpy images and receives back pseudo-classification probabilities.

**Constructor parameters:**
- `model`: the loaded CrossAttentionFusion model (already in eval mode, on device)
- `fixed_input_ids`: `torch.Tensor [1, 256]` -- the tokenized text for this sample
- `fixed_attention_mask`: `torch.Tensor [1, 256]` -- the attention mask for this sample
- `fixed_pixel_values`: `torch.Tensor [1, max_images, 3, 224, 224]` -- all images for this sample (preprocessed)
- `fixed_num_images`: `torch.Tensor [1]` -- actual number of real images
- `score_index`: `int` -- which of the 5 targets to explain (0-4)
- `device`: `torch.device`
- `image_processor`: the TimmProcessor instance (from Phase 1 `get_image_processor()`)
- `batch_size`: `int = 64` -- internal batch size for model inference on perturbed samples

**`__call__(self, images_np)` method:**

Input: `images_np` is a numpy array of shape `[N, H, W, 3]` provided by LIME. Values are typically in `[0, 255]` uint8 range, matching the original image passed to `explain_instance()`.

Processing steps:
1. Initialize output array: `results = np.zeros((N, 2))`
2. Process in internal batches of size `batch_size` to avoid GPU OOM:
   a. For each batch of perturbed images:
      - Convert each numpy image `[H, W, 3]` to a PIL Image (RGB). Handle both uint8 `[0, 255]` and float `[0.0, 1.0]` ranges: if `images_np.max() > 1.0`, treat as uint8; otherwise, multiply by 255 and cast to uint8 before creating PIL Image.
      - Apply `image_processor` (TimmProcessor) to each PIL image to get a tensor `[3, 224, 224]`.
      - Stack into `[batch_n, 3, 224, 224]`.
      - Build multi-image tensor: create `[batch_n, max_images, 3, 224, 224]` by placing the perturbed image at index 0 and copying fixed images at indices 1 through `max_images-1` from `fixed_pixel_values`.
      - Repeat `fixed_input_ids` to `[batch_n, 256]`.
      - Repeat `fixed_attention_mask` to `[batch_n, 256]`.
      - Repeat `fixed_num_images` to `[batch_n]`.
      - Move all tensors to `device`.
      - Run model with `torch.no_grad()`:
        ```
        preds = model(input_ids, attention_mask, pixel_values, num_images)
        ```
      - Extract target score: `score = preds[:, score_index]` -> `[batch_n]`
      - Apply sigmoid: `p_high = torch.sigmoid(score)` -> `[batch_n]`
      - Build two-column output: `two_col = torch.stack([1 - p_high, p_high], dim=1)` -> `[batch_n, 2]`
      - Copy to results array.
3. Return: `results` as `numpy.ndarray [N, 2]`

**Critical implementation notes:**
- The image preprocessing must be identical to training. The TimmProcessor uses `timm.data.create_transform(**data_config, is_training=False)`. This includes specific normalization constants (ImageNet mean/std for Swin-B), resize, and center crop. LIME provides raw (potentially resized) numpy images; the predict function must convert them to PIL and then apply TimmProcessor.
- GPU memory management: with `num_samples=1000` perturbations, processing all at once with the full multimodal model will OOM. The internal `batch_size=64` processes perturbations in chunks.
- The model must remain in `eval()` mode throughout. Verify once in the constructor: `assert not model.training`.

---

### Step C: Implement `TextLimePredictFn` Class

**Purpose:** Callable class that LIME calls with lists of perturbed text strings and receives back pseudo-classification probabilities.

**Constructor parameters:**
- `model`: the loaded CrossAttentionFusion model (eval mode, on device)
- `fixed_pixel_values`: `torch.Tensor [1, max_images, 3, 224, 224]` -- fixed images for this sample
- `fixed_num_images`: `torch.Tensor [1]` -- actual number of real images
- `tokenizer`: the PhoBERT tokenizer (from Phase 1 `get_tokenizer()`)
- `max_length`: `int = 256` -- must match training tokenization
- `score_index`: `int` -- which target to explain (0-4)
- `device`: `torch.device`
- `batch_size`: `int = 64` -- internal batch size

**`__call__(self, text_list)` method:**

Input: `text_list` is a list of N strings. Each string is a version of the original review with some words removed by LIME.

Processing steps:
1. Initialize output array: `results = np.zeros((len(text_list), 2))`
2. Process in internal batches:
   a. For each batch of text strings:
      - Tokenize all strings at once with the PhoBERT tokenizer:
        ```
        enc = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        ```
      - **Critical:** Use `padding='max_length'` (not `padding=True` or `padding='longest'`) to match training behavior in `MultimodalDataset.__getitem__()`. Different padding strategies produce different `attention_mask` tensors, which may produce subtly different predictions from PhoBERT.
      - Repeat `fixed_pixel_values` to `[batch_n, max_images, 3, 224, 224]`.
      - Repeat `fixed_num_images` to `[batch_n]`.
      - Move all tensors to `device`.
      - Run model with `torch.no_grad()`.
      - Extract target score, apply sigmoid, build two-column output (same as image wrapper).
      - Copy to results array.
3. Return: `results` as `numpy.ndarray [N, 2]`

**Critical implementation notes:**
- LIME perturbs text by removing words (replacing them with `UNKWORDZ` or simply removing them from the string). The tokenizer then processes the modified string. This means the tokenization of perturbed strings will differ from the original -- different number of tokens, different token IDs, different attention masks. This is correct behavior for LIME.
- The `max_length=256` truncation ensures that even if LIME produces a very short string (few words remaining), the padding makes the tensor shape `[batch_n, 256]`, matching the model's expected input.

---

### Step D: Implement `run_lime_image()` Function

**Purpose:** Orchestrate a single Image LIME explanation for one sample and one target.

**Signature:**
```
run_lime_image(
    model,
    sample,             # dict from load_single_sample()
    score_index,        # int, 0-4
    image_processor,    # TimmProcessor instance
    device,             # torch.device
    num_samples=1000,   # number of LIME perturbations
    seed=42,            # random seed for reproducibility
    batch_size=64,      # internal batch size for predict_fn
) -> lime_image.ImageExplanation
```

**Implementation steps:**
1. Extract the primary image for LIME:
   - Get the original image from the sample. The sample from `load_single_sample()` contains `pixel_values [max_images, 3, 224, 224]` (preprocessed tensors) and access to the original image files.
   - For LIME, we need the original image as a numpy array `[H, W, 3]` in uint8 `[0, 255]` format. This is the image **before** TimmProcessor transforms.
   - The `load_single_sample()` function should also provide the raw PIL image or the image path. If not, load the primary image from the image URL/path using the same MD5-hashed filename logic as `MultimodalDataset._load_image()`.
   - Convert PIL image to numpy array: `image_np = np.array(pil_image)` -> `[H_orig, W_orig, 3]`, uint8.
   - **Important:** Do NOT resize the image to 224x224 before passing to LIME. LIME will pass the image at whatever resolution it receives. The predict function handles all resizing via TimmProcessor internally. However, if the original image is very large (e.g., 4000x3000), this will create very large superpixel masks and slow down LIME. **Decision:** Resize the image to 224x224 before passing to LIME. This ensures superpixel masks are at the model's native resolution, and the TimmProcessor in the predict function will handle it correctly since 224x224 is the expected input size.

2. Create the predict function:
   ```
   predict_fn = ImageLimePredictFn(
       model=model,
       fixed_input_ids=sample['input_ids'].unsqueeze(0),
       fixed_attention_mask=sample['attention_mask'].unsqueeze(0),
       fixed_pixel_values=sample['pixel_values'].unsqueeze(0),
       fixed_num_images=sample['num_images'].unsqueeze(0),
       score_index=score_index,
       device=device,
       image_processor=image_processor,
       batch_size=batch_size,
   )
   ```

3. Create the LIME image explainer:
   ```
   explainer = LimeImageExplainer()
   ```

4. Run `explain_instance`:
   ```
   explanation = explainer.explain_instance(
       image_np,                    # [224, 224, 3] numpy array
       predict_fn,                  # callable
       top_labels=2,                # explain both pseudo-classes
       hide_color=0,                # black for hidden superpixels
       num_samples=num_samples,     # default 1000
       random_seed=seed,            # for reproducibility
   )
   ```

5. Return the `explanation` object.

**Notes on `hide_color`:** When LIME hides a superpixel, it replaces those pixels with `hide_color`. Using `0` (black) means hidden regions become black pixels. After TimmProcessor normalization (ImageNet mean/std), black pixels become approximately `[-2.1, -1.8, -2.0]` in the normalized tensor, which represents a substantial deviation from typical food/restaurant images. This is acceptable because it tests: "what happens if this region contained no information?"

---

### Step E: Implement `run_lime_text()` Function

**Purpose:** Orchestrate a single Text LIME explanation for one sample and one target.

**Signature:**
```
run_lime_text(
    model,
    sample,             # dict from load_single_sample()
    score_index,        # int, 0-4
    tokenizer,          # PhoBERT tokenizer
    device,             # torch.device
    num_features=10,    # number of top features to select
    num_samples=500,    # number of LIME perturbations
    seed=42,            # random seed
    max_length=256,     # tokenizer max length
    batch_size=64,      # internal batch size
) -> lime_text.TextExplanation
```

**Implementation steps:**
1. Extract the original text from the sample:
   - Get `text = sample['original_text']` (the raw `comment_clean` string before tokenization).
   - If `load_single_sample()` does not return the original text string, reconstruct it from the dataframe row or pass it separately.

2. Create the predict function:
   ```
   predict_fn = TextLimePredictFn(
       model=model,
       fixed_pixel_values=sample['pixel_values'].unsqueeze(0),
       fixed_num_images=sample['num_images'].unsqueeze(0),
       tokenizer=tokenizer,
       max_length=max_length,
       score_index=score_index,
       device=device,
       batch_size=batch_size,
   )
   ```

3. Create the LIME text explainer:
   ```
   text_explainer = LimeTextExplainer(
       class_names=['low', 'high'],
       split_expression=r'\s+',     # whitespace splitting for Vietnamese syllables
       random_state=seed,
   )
   ```

4. Run `explain_instance`:
   ```
   explanation = text_explainer.explain_instance(
       text,                        # original review string
       predict_fn,                  # callable
       num_features=num_features,   # top features to report
       num_samples=num_samples,     # default 500
       labels=(1,),                 # explain the "high" pseudo-class
   )
   ```

5. Return the `explanation` object.

**Notes on `split_expression`:** Vietnamese text is space-separated at the syllable level. PhoBERT tokenizes at the syllable level using BPE. Using whitespace splitting means LIME features are syllables, which aligns with the model's tokenization granularity. Compound words like "nha hang" (restaurant) will be split into "nha" and "hang" as separate LIME features. This is acceptable because PhoBERT processes them as separate tokens too.

**Notes on `num_samples`:** Text LIME uses 500 perturbations by default (less than image LIME's 1000) because text perturbation spaces are typically smaller (number of words in a review << number of superpixels in an image), so fewer samples are needed for a stable local surrogate.

---

### Step F: Implement Visualization Functions

#### F1: `save_lime_image_explanation()`

**Signature:**
```
save_lime_image_explanation(
    explanation,        # LIME ImageExplanation object
    image_np,           # original image [H, W, 3] numpy
    sample_id,          # int or str
    score_index,        # int, 0-4
    save_dir,           # path to experiments/EXP_XXX/xai/lime/
    dpi=300,
)
```

**Implementation:**
1. Get positive-only mask:
   ```
   temp, mask = explanation.get_image_and_mask(
       label=1,                    # "high" pseudo-class
       positive_only=True,
       num_features=5,             # top 5 positive superpixels
       hide_rest=False,
   )
   ```
2. Overlay boundaries on original image using `skimage.segmentation.mark_boundaries()`.
3. Save as `sample_{id}_target{idx}_{name}_image_positive.png` at thesis DPI.
4. Get negative-only mask:
   ```
   temp, mask = explanation.get_image_and_mask(
       label=1,
       positive_only=False,
       negative_only=True,
       num_features=5,
   )
   ```
5. Save as `sample_{id}_target{idx}_{name}_image_negative.png`.
6. Get combined (positive + negative) visualization:
   ```
   temp, mask = explanation.get_image_and_mask(
       label=1,
       positive_only=False,
       num_features=10,
       hide_rest=False,
   )
   ```
7. Save as `sample_{id}_target{idx}_{name}_image_combined.png`.
8. Save superpixel weights to JSON:
   ```
   weights = dict(explanation.local_exp[1])  # label=1 weights
   ```
   Save as `raw/sample_{id}_target{idx}_image_weights.json`.

Where `{name}` is the factor short name from `INDEX_TO_FACTOR[score_index]` (e.g., "food", "price", "atmos", "service", "overall").

#### F2: `save_lime_text_explanation()`

**Signature:**
```
save_lime_text_explanation(
    explanation,        # LIME TextExplanation object
    sample_id,
    score_index,
    save_dir,
    num_features=10,    # number of top features in bar chart
    dpi=300,
)
```

**Implementation:**
1. Extract word weights from explanation:
   ```
   word_weights = explanation.as_list(label=1)
   ```
   This returns a list of `(word, weight)` tuples sorted by absolute weight.
2. Take top `num_features` by absolute weight.
3. Create horizontal bar chart:
   - Sort by weight value (not absolute).
   - Color positive weights green, negative weights red.
   - X-axis: LIME weight. Y-axis: Vietnamese word (syllable).
   - Title: `"LIME Text: {DISPLAY_NAME} (Sample {id})"`.
   - Use `plt.barh()` with color conditional on sign.
   - Ensure Vietnamese characters render correctly (use a font that supports Vietnamese diacritics, or rely on matplotlib's default Unicode support).
4. Save as `sample_{id}_target{idx}_{name}_text_bar.png` at thesis DPI.
5. Save word weights to JSON:
   ```
   weights_dict = {word: float(weight) for word, weight in word_weights}
   ```
   Save as `raw/sample_{id}_target{idx}_text_weights.json`.

#### F3: `save_lime_text_html()`

**Signature:**
```
save_lime_text_html(
    explanation,
    sample_id,
    score_index,
    save_dir,
)
```

**Implementation:**
1. Generate HTML: `html_str = explanation.as_html(labels=(1,))`
2. Save as `sample_{id}_target{idx}_{name}_text.html`.
3. This HTML file provides an interactive view with color-coded words that can be opened in a browser.

#### F4: `save_lime_config()`

**Signature:**
```
save_lime_config(
    sample_id,
    score_index,
    save_dir,
    num_samples,
    num_features,
    seed,
    lime_type,          # 'image' or 'text'
    hide_color=None,    # only for image LIME
)
```

**Implementation:** Save a JSON file to `raw/sample_{id}_target{idx}_lime_config.json` containing:
- `lime_type` (image/text)
- `num_samples`
- `num_features`
- `seed`
- `hide_color` (image only)
- `split_expression` (text only)
- `timestamp` (ISO format)
- `score_index`
- `target_name`

#### F5: `save_lime_metadata()`

**Signature:**
```
save_lime_metadata(
    sample_id,
    sample,             # dict from load_single_sample()
    predictions,        # dict from get_prediction()
    save_dir,
)
```

**Implementation:** Save a JSON file to `metadata/sample_{id}_lime_metadata.json` containing:
- `sample_index`: int
- `original_text`: the raw review text
- `num_images`: actual number of real images
- `predictions`: dict of target_name -> predicted_value
- `ground_truth`: dict of target_name -> true_value
- `absolute_errors`: dict of target_name -> |pred - true|

---

### Step G: Handle Regression Output for LIME (Sigmoid Mapping)

**Problem:** `LimeImageExplainer` and `LimeTextExplainer` expect the `classifier_fn` to return an array of shape `[N, num_classes]` where values behave like probabilities (sum approximately to 1 per row, values in [0, 1]). The model outputs regression scores in approximately [1, 10].

**Solution: Sigmoid pseudo-classification mapping.**

For each perturbed sample:
1. Get the raw regression score for the target: `s = preds[:, score_index]`
2. Apply sigmoid: `p = sigmoid(s) = 1 / (1 + exp(-s))`
3. Return two columns: `[1 - p, p]`

**Why sigmoid works:**
- Sigmoid maps any real value to (0, 1).
- For scores in [1, 10], sigmoid(1) = 0.73, sigmoid(5) = 0.99, sigmoid(10) = 0.99995. This means most scores will produce p_high close to 1.
- The key for LIME is not the absolute probability values but the **relative changes** across perturbations. If removing a food region decreases the score from 7.5 to 6.8, the sigmoid output changes from 0.99945 to 0.99889. The change is small but consistent, and LIME's linear surrogate can detect it.
- **Potential issue:** For scores > 3, sigmoid saturates and relative changes become very small. This can reduce LIME's sensitivity.
- **Mitigation:** An alternative is to normalize scores first: `s_norm = (s - min_score) / (max_score - min_score)` where `min_score=1`, `max_score=10`, then apply sigmoid to `s_norm * 6 - 3` (maps [0,1] to [-3, 3] for sigmoid). This spreads the sigmoid across a more sensitive range.
- **Decision for this thesis:** Use direct sigmoid without pre-normalization. The default approach is standard in the LIME regression literature, and the documentation explicitly warns that LIME is used as a "local sanity check," not as a precise quantitative tool. If results appear insensitive (all perturbations produce nearly identical pseudo-probabilities), switch to the normalized sigmoid variant.

**Implementation in both `ImageLimePredictFn` and `TextLimePredictFn`:**
```
score = preds[:, self.score_index]              # [batch_n]
p_high = torch.sigmoid(score)                    # [batch_n]
two_col = torch.stack([1 - p_high, p_high], dim=1)  # [batch_n, 2]
return two_col.cpu().numpy()
```

---

### Step H: Vietnamese Text Handling for LIME Text

**Problem:** LIME's `LimeTextExplainer` splits text into features using a `split_expression` parameter. The default is `r'\s+'` (whitespace). For Vietnamese text, this produces syllable-level features.

**Analysis:**

Vietnamese is a syllable-separated language. Compound words are written with spaces between syllables:
- "nha hang" = "restaurant" (two syllables)
- "khong gian" = "atmosphere" (two syllables)
- "nhan vien" = "staff" (two syllables)
- "gia ca" = "price" (two syllables)

PhoBERT (`vinai/phobert-base-v2`) tokenizes at the syllable level with BPE. Each Vietnamese syllable typically becomes one or a small number of BPE tokens. Therefore, LIME's whitespace splitting produces features at approximately the same granularity as PhoBERT's tokenization.

**Example:**
```
Original text:  "Do an ngon nhung gia hoi cao"
LIME features:  ["Do", "an", "ngon", "nhung", "gia", "hoi", "cao"]
PhoBERT tokens: ["Do", "an", "ngon", "nhung", "gia", "hoi", "cao"]
                (approximately -- BPE may split rare syllables further)
```

**Trade-off:** Compound words like "nha hang" are treated as two independent features. LIME cannot test the removal of "nha hang" as a unit. However, this matches PhoBERT's processing, and individual syllable importance is still informative (e.g., "ngon" alone is a strong food quality signal).

**Decision:** Use default whitespace splitting (`split_expression=r'\s+'`). Document in the thesis that LIME features are syllable-level, consistent with PhoBERT's tokenization granularity.

---

### Step I: Multi-Image Handling for Image LIME

**Problem:** Reviews can contain up to 4 images. LIME perturbs one image at a time by manipulating superpixels. How should multi-image reviews be handled?

**Strategy analysis:**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| A: First image only | Explain only the first real image; keep others fixed | Simple, fast, consistent with Phase 2 primary analysis | Misses contribution of other images |
| B: All images separately | Run LIME on each real image independently, keeping the others fixed | Comprehensive | Expensive (N_images x num_samples forward passes per target) |
| C: Highest Grad-CAM image | Use Phase 2 results to select the image with highest Grad-CAM activation | Focuses on most relevant image | Requires Phase 2 to be complete first; introduces dependency |

**Decision:**
- **Primary approach (Strategy A):** Explain only the first real image. This is consistent with Phase 2 (Grad-CAM) primary analysis, which also focuses on the first image by default. The first image is typically the most representative food/restaurant photo in the review.
- **Extended approach (Strategy B):** For 2--3 selected case studies where multi-image analysis is desired, run LIME on each real image separately. This is optional and only performed if time permits.

**Implementation:** In `ImageLimePredictFn.__call__()`, the perturbed image is always placed at position 0 in the multi-image tensor. Images at positions 1 through `max_images-1` are copied from `fixed_pixel_values` without modification. The `fixed_num_images` tensor is passed unchanged.

---

### Step J: Original Image Retrieval for LIME

**Problem:** LIME requires the original image as a numpy array `[H, W, 3]`. The current `load_single_sample()` returns preprocessed tensors `[max_images, 3, 224, 224]` which have already been through TimmProcessor (normalized, resized). LIME needs the image **before** preprocessing.

**Solution:** Add a helper function or modify `load_single_sample()` to also return the raw PIL images.

**`load_original_images(csv_path, sample_index, image_dir, max_images=4)` function:**
1. Read the row from the CSV at `sample_index`.
2. Parse `image_url` column to get the list of image URLs.
3. For each URL (up to `max_images`):
   a. Compute MD5 hash: `hashlib.md5(url.encode('utf-8')).hexdigest()`
   b. Construct local path: `os.path.join(image_dir, f"{hash}.jpg")`
   c. Load as PIL image: `Image.open(local_path).convert('RGB')`
   d. If file not found, create black image: `Image.new('RGB', (224, 224), color='black')`
4. Return list of PIL images and `num_real_images` count.

**For LIME:** Take the first real PIL image, resize to 224x224 (`pil_image.resize((224, 224), Image.BILINEAR)`), convert to numpy: `np.array(pil_image)` -> `[224, 224, 3]` uint8.

This function replicates the image loading logic from `MultimodalDataset._load_image()` in `src/dataset.py`.

---

### Step K: Stability Analysis Function

**Purpose:** Run LIME with multiple seeds and assess explanation consistency.

**`run_lime_stability_analysis()` function:**

**Signature:**
```
run_lime_stability_analysis(
    model, sample, score_index, ...,
    seeds=[42, 123, 456],
    top_k=5,                    # number of top features to compare
    lime_type='text',           # or 'image'
) -> dict
```

**Implementation:**
1. For each seed in `seeds`:
   a. Run LIME (image or text) with that seed.
   b. Extract top-K features by absolute weight.
   c. Store the feature list and weights.
2. Compute consensus:
   - **Consensus features:** features appearing in 2/3 or 3/3 seed runs.
   - **Jaccard similarity:** pairwise Jaccard index between top-K feature sets.
   - **Rank correlation:** Spearman rank correlation of weights for common features across seeds.
3. Return a dict containing:
   - Per-seed top-K features and weights.
   - Consensus features with their average weight.
   - Pairwise Jaccard similarities.
   - Overall stability score: fraction of consensus features out of top-K.
4. Save to `stability/sample_{id}_target{idx}_stability.json`.

---

## 7. Required Code Files

| File | Location | Responsibility |
|------|----------|---------------|
| `lime_explainer.py` | `xai/lime_explainer.py` | All LIME-specific code: `ImageLimePredictFn`, `TextLimePredictFn`, `run_lime_image()`, `run_lime_text()`, `save_lime_image_explanation()`, `save_lime_text_explanation()`, `save_lime_text_html()`, `save_lime_config()`, `save_lime_metadata()`, `load_original_images()`, `run_lime_stability_analysis()` |
| Phase 5 notebook | `xai/notebooks/Phase5_LIME.ipynb` | End-to-end LIME analysis notebook. Loads model, selects samples, runs image and text LIME, generates all visualizations, performs stability analysis, cross-validates with Phase 2 and Phase 3 results. |

**No modifications to existing files.** Phase 5 only adds new files and imports from `xai/utils.py` and `xai/config.py` (created in Phase 1).

---

## 8. Folder Structure

After Phase 5 completion, the following artifacts are added:

```
SE365/                                              (project root)
|-- xai/                                            (Phase 1, existing)
|   |-- __init__.py
|   |-- config.py
|   |-- utils.py
|   |-- lime_explainer.py                           (NEW -- Phase 5)
|   |-- notebooks/
|       |-- Phase1_Infrastructure_Verification.ipynb (Phase 1)
|       |-- Phase2_GradCAM.ipynb                     (Phase 2)
|       |-- Phase3_Attention.ipynb                   (Phase 3)
|       |-- Phase4_SHAP.ipynb                        (Phase 4)
|       |-- Phase5_LIME.ipynb                        (NEW -- Phase 5)
|
|-- experiments/
    |-- EXP_060A_bestsequential_full_configuration/
        |-- xai/
            |-- lime/                               (NEW -- Phase 5)
            |   |-- sample_{id}_target0_food_image_positive.png
            |   |-- sample_{id}_target0_food_image_negative.png
            |   |-- sample_{id}_target0_food_image_combined.png
            |   |-- sample_{id}_target0_food_text_bar.png
            |   |-- sample_{id}_target0_food_text.html
            |   |-- sample_{id}_target1_price_image_positive.png
            |   |-- sample_{id}_target1_price_image_negative.png
            |   |-- sample_{id}_target1_price_image_combined.png
            |   |-- sample_{id}_target1_price_text_bar.png
            |   |-- sample_{id}_target1_price_text.html
            |   |-- sample_{id}_target2_atmos_image_positive.png
            |   |-- sample_{id}_target2_atmos_image_negative.png
            |   |-- sample_{id}_target2_atmos_image_combined.png
            |   |-- sample_{id}_target2_atmos_text_bar.png
            |   |-- sample_{id}_target2_atmos_text.html
            |   |-- sample_{id}_target3_service_image_positive.png
            |   |-- sample_{id}_target3_service_image_negative.png
            |   |-- sample_{id}_target3_service_image_combined.png
            |   |-- sample_{id}_target3_service_text_bar.png
            |   |-- sample_{id}_target3_service_text.html
            |   |-- sample_{id}_target4_overall_image_positive.png
            |   |-- sample_{id}_target4_overall_image_negative.png
            |   |-- sample_{id}_target4_overall_image_combined.png
            |   |-- sample_{id}_target4_overall_text_bar.png
            |   |-- sample_{id}_target4_overall_text.html
            |   |-- raw/
            |   |   |-- sample_{id}_target0_image_weights.json
            |   |   |-- sample_{id}_target0_text_weights.json
            |   |   |-- sample_{id}_target0_lime_config.json
            |   |   |-- sample_{id}_target1_image_weights.json
            |   |   |-- sample_{id}_target1_text_weights.json
            |   |   |-- sample_{id}_target1_lime_config.json
            |   |   |-- ... (for all 5 targets)
            |   |-- metadata/
            |   |   |-- sample_{id}_lime_metadata.json
            |   |-- stability/
            |       |-- sample_{id}_target{idx}_stability.json
            |
            |-- (infrastructure/, gradcam/, attention/, shap/ -- from prior phases)
```

**Naming convention:** `sample_{id}_target{idx}_{name}_{type}_{variant}.{ext}` where:
- `{id}` = dataset row index (integer)
- `{idx}` = target index (0-4)
- `{name}` = factor short name from `INDEX_TO_FACTOR` (food, price, atmos, service, overall)
- `{type}` = image or text
- `{variant}` = positive, negative, combined, bar, etc.
- `{ext}` = png, html, json

---

## 9. Notebook Design

### `xai/notebooks/Phase5_LIME.ipynb`

#### Cell 1: Title and Documentation
**Type:** Markdown
**Content:**
- Phase 5 title: "LIME Local Explanation for Multimodal Quality Assessment"
- Brief description of LIME methodology
- Note on phase priority (lowest among XAI methods)
- Link to Phase 1 infrastructure verification

#### Cell 2: Environment Setup and Imports
**Type:** Code
**Content:**
- Import standard libraries: `numpy`, `torch`, `os`, `json`, `warnings`
- Import `PIL.Image`, `matplotlib.pyplot`
- Import from `xai.config`: target names, indices, display names, color schemes, seeds, DPI, model config constants
- Import from `xai.utils`: `load_model`, `load_single_sample`, `get_prediction`, `get_image_processor`, `get_tokenizer`, `set_seed`, `get_device`, `save_figure`, `save_raw_values`
- Import from `xai.lime_explainer`: `run_lime_image`, `run_lime_text`, `save_lime_image_explanation`, `save_lime_text_explanation`, `save_lime_text_html`, `save_lime_config`, `save_lime_metadata`, `load_original_images`, `run_lime_stability_analysis`
- `warnings.filterwarnings('ignore')`
- Print library versions

**Expected output:** Clean import, version confirmation

#### Cell 3: Configuration
**Type:** Code
**Content:**
- Set experiment path: `EXP_DIR = 'experiments/EXP_060A_bestsequential_full_configuration'`
- Set checkpoint path: `CHECKPOINT_PATH = os.path.join(EXP_DIR, 'best_model_train_fusion.pth')`
- Set CSV path: `CSV_PATH = 'data/text/test.csv'` (or val.csv)
- Set image directory: `IMAGE_DIR = 'data/image'`
- Set LIME output directory: `LIME_DIR = os.path.join(EXP_DIR, 'xai', 'lime')`
- Create output subdirectories: `raw/`, `metadata/`, `stability/`
- Set LIME hyperparameters:
  ```
  IMAGE_NUM_SAMPLES = 1000
  TEXT_NUM_SAMPLES = 500
  TEXT_NUM_FEATURES = 10
  PRIMARY_SEED = 42
  STABILITY_SEEDS = [42, 123, 456]
  BATCH_SIZE = 64
  ```
- Select sample indices for case study:
  ```
  SAMPLE_INDICES = [0, 15, 42, 100, 200]  # adjust based on dataset
  ```
  Criteria for sample selection (documented in markdown cell):
  - Include at least 1 high-scoring sample (most targets > 7)
  - Include at least 1 low-scoring sample (most targets < 4)
  - Include at least 1 sample with mixed scores (high food, low price)
  - Include at least 1 sample with multiple real images
  - Include at least 1 sample with an interesting/long review text

**Expected output:** Configuration printed, directories created

#### Cell 4: Model and Preprocessor Loading
**Type:** Code
**Content:**
- `device = get_device()`
- `set_seed(PRIMARY_SEED)`
- `model = load_model(CHECKPOINT_PATH, device)`
- `tokenizer = get_tokenizer()`
- `image_processor = get_image_processor()`
- Verify model is in eval mode: `assert not model.training`
- Print device, model type confirmation

**Expected output:** Model loaded successfully, device confirmed

#### Cell 5: Sample Selection and Preview
**Type:** Code
**Content:**
- Load dataframe: `df = pd.read_csv(CSV_PATH)`
- For each sample index in `SAMPLE_INDICES`:
  - Print the review text (first 200 chars)
  - Print ground truth scores for all 5 targets
  - Print number of images
  - Load and display the primary image as a thumbnail
- Load each sample using `load_single_sample()`
- Compute and display model predictions using `get_prediction()`
- Show prediction vs ground truth table

**Expected output:** Sample previews with text, images, and predictions

#### Cell 6: Image LIME -- Single Sample Walkthrough
**Type:** Markdown + Code
**Content:**
- Markdown: Explain the Image LIME process step by step
- Code: For one sample (e.g., `SAMPLE_INDICES[0]`), one target (e.g., `food_score`):
  1. Load original PIL image using `load_original_images()`
  2. Display original image
  3. Run `run_lime_image()` with verbose output
  4. Display positive superpixel overlay
  5. Display negative superpixel overlay
  6. Display combined overlay
  7. Print top superpixel weights
  8. Save all artifacts

**Expected output:** Three image overlays, superpixel weight values, saved artifacts

#### Cell 7: Text LIME -- Single Sample Walkthrough
**Type:** Markdown + Code
**Content:**
- Markdown: Explain the Text LIME process step by step
- Code: For the same sample and target as Cell 6:
  1. Extract original text
  2. Display original text
  3. Run `run_lime_text()` with verbose output
  4. Display word importance bar chart
  5. Print word weights table
  6. Save bar chart, HTML, and JSON artifacts

**Expected output:** Bar chart, word weights, saved artifacts

#### Cell 8: Batch LIME -- All Samples, All Targets
**Type:** Code
**Content:**
- For each sample in `SAMPLE_INDICES`:
  - Load sample and original images
  - For each target index in 0..4:
    - Run `run_lime_image()` (with `IMAGE_NUM_SAMPLES`, `PRIMARY_SEED`)
    - Save all image LIME artifacts
    - Run `run_lime_text()` (with `TEXT_NUM_SAMPLES`, `PRIMARY_SEED`)
    - Save all text LIME artifacts
    - Save LIME config JSON
  - Save sample metadata JSON
  - Print progress: "Completed sample {id}: {elapsed_time}s"
- Print total elapsed time

**Expected output:** All artifacts generated for all samples and targets, progress log

#### Cell 9: Stability Analysis
**Type:** Markdown + Code
**Content:**
- Markdown: Explain why stability matters for LIME (stochastic perturbation sampling)
- Code: For 2--3 selected samples and 2--3 selected targets:
  - Run `run_lime_stability_analysis()` with `STABILITY_SEEDS = [42, 123, 456]`
  - Print consensus features for each sample/target
  - Print pairwise Jaccard similarities
  - Print stability score (fraction of features in consensus)
  - Save stability reports

**Expected output:** Stability metrics, consensus feature lists

#### Cell 10: Cross-Validation with Grad-CAM (Phase 2)
**Type:** Markdown + Code
**Content:**
- Markdown: "Do LIME positive superpixels overlap with Grad-CAM hot regions?"
- Code: For 2--3 selected samples:
  - Load Grad-CAM heatmap PNG from `experiments/EXP_XXX/xai/gradcam/`
  - Load LIME positive superpixel overlay PNG
  - Display side by side using `matplotlib.pyplot.subplot(1, 2, ...)`
  - Qualitative comparison: document whether high-attention Grad-CAM regions correspond to positive LIME superpixels

**Expected output:** Side-by-side comparison figures, written analysis

#### Cell 11: Cross-Validation with Attention (Phase 3)
**Type:** Markdown + Code
**Content:**
- Markdown: "Do LIME top-importance words match Attention top-attended tokens?"
- Code: For 2--3 selected samples:
  - Load Attention token importance from `experiments/EXP_XXX/xai/attention/` JSON
  - Load LIME text weights from `raw/sample_{id}_target{idx}_text_weights.json`
  - Compare top-K lists: compute overlap, rank correlation
  - Display comparison table

**Expected output:** Overlap metrics, comparison tables

#### Cell 12: Per-Target Analysis
**Type:** Markdown + Code
**Content:**
- Markdown: "Does LIME show target-specific behavior?"
- Code: For one sample with varied scores:
  - Show LIME text bar charts for all 5 targets side by side
  - Analyze: does "ngon" have highest weight for `food_score`? Does "gia" have highest weight for `price_score`?
  - Show LIME image overlays for `food_score` vs `atmosphere_score` -- do different regions light up?

**Expected output:** Multi-target comparison figures, written analysis

#### Cell 13: Summary and Findings
**Type:** Markdown
**Content:**
- Summary of key findings:
  - Which words consistently appear as top LIME features across samples?
  - Which image regions are consistently highlighted?
  - Agreement/disagreement with Grad-CAM and Attention
  - Stability assessment
  - Target-specific behavior analysis
- Limitations:
  - Local explanations only
  - Syllable-level features, not compound words
  - Computational cost
  - Sensitivity to segmentation (image) and perturbation sampling

---

## 10. Algorithm

### 10.1 Image LIME Algorithm

```
ALGORITHM: Image LIME for Multimodal Model
==========================================

INPUT:
  model          -- trained CrossAttentionFusion (eval mode, on device)
  sample         -- dict with input_ids, attention_mask, pixel_values, num_images
  original_image -- PIL Image (first real image, pre-preprocessing)
  score_index    -- target to explain (0-4)
  num_samples    -- number of perturbations (default 1000)
  seed           -- random seed (default 42)

OUTPUT:
  explanation    -- LIME ImageExplanation object
  artifacts      -- saved PNGs, JSONs

PROCEDURE:
  1. PREPARE image
     a. Resize original_image to (224, 224)
     b. Convert to numpy array [224, 224, 3] uint8

  2. CONSTRUCT predict_fn = ImageLimePredictFn(
       model, fixed_text, fixed_images, score_index, image_processor)

  3. VERIFY predict_fn on original image
     a. original_pred = predict_fn(image_np[np.newaxis, ...])
     b. direct_pred = model(sample inputs) with get_prediction()
     c. Assert |sigmoid(direct_pred[score_index]) - original_pred[0, 1]| < 0.01
     d. If assertion fails, diagnose preprocessing mismatch

  4. CREATE explainer = LimeImageExplainer()

  5. GENERATE explanation
     a. explanation = explainer.explain_instance(
          image_np,
          predict_fn,
          top_labels=2,
          hide_color=0,
          num_samples=num_samples,
          random_seed=seed
        )
     b. LIME internally:
        i.   Segment image into superpixels using quickshift
        ii.  Generate num_samples binary masks over superpixels
        iii. For each mask: hide masked superpixels, run predict_fn
        iv.  Fit weighted linear regression near original
        v.   Extract per-superpixel coefficients

  6. EXTRACT results
     a. positive_mask = explanation.get_image_and_mask(label=1, positive_only=True)
     b. negative_mask = explanation.get_image_and_mask(label=1, negative_only=True)
     c. combined_mask = explanation.get_image_and_mask(label=1, positive_only=False)
     d. weights = dict(explanation.local_exp[1])

  7. SAVE artifacts
     a. save_lime_image_explanation(explanation, image_np, ...)
     b. save_lime_config(...)

  8. RETURN explanation
```

### 10.2 Text LIME Algorithm

```
ALGORITHM: Text LIME for Multimodal Model
==========================================

INPUT:
  model          -- trained CrossAttentionFusion (eval mode, on device)
  sample         -- dict with input_ids, attention_mask, pixel_values, num_images
  original_text  -- raw review string (comment_clean)
  score_index    -- target to explain (0-4)
  num_features   -- top features to select (default 10)
  num_samples    -- number of perturbations (default 500)
  seed           -- random seed (default 42)

OUTPUT:
  explanation    -- LIME TextExplanation object
  artifacts      -- saved PNGs, HTMLs, JSONs

PROCEDURE:
  1. CONSTRUCT predict_fn = TextLimePredictFn(
       model, fixed_images, fixed_num_images, tokenizer,
       max_length=256, score_index, device)

  2. VERIFY predict_fn on original text
     a. original_pred = predict_fn([original_text])
     b. direct_pred = get_prediction(model, sample)
     c. Assert |sigmoid(direct_pred[target_name]) - original_pred[0, 1]| < 0.01

  3. CREATE text_explainer = LimeTextExplainer(
       class_names=['low', 'high'],
       split_expression=r'\s+',
       random_state=seed
     )

  4. GENERATE explanation
     a. explanation = text_explainer.explain_instance(
          original_text,
          predict_fn,
          num_features=num_features,
          num_samples=num_samples,
          labels=(1,)
        )
     b. LIME internally:
        i.   Split text into words using split_expression
        ii.  Generate num_samples binary masks over words
        iii. For each mask: reconstruct text with only unmasked words
        iv.  Run predict_fn on perturbed texts
        v.   Fit weighted sparse linear regression
        vi.  Extract per-word coefficients

  5. EXTRACT results
     a. word_weights = explanation.as_list(label=1)
     b. html_str = explanation.as_html(labels=(1,))

  6. SAVE artifacts
     a. save_lime_text_explanation(explanation, ...)
     b. save_lime_text_html(explanation, ...)
     c. save_lime_config(...)

  7. RETURN explanation
```

### 10.3 Full Phase 5 Orchestration Algorithm

```
ALGORITHM: Phase 5 Complete LIME Analysis
==========================================

INPUT:
  checkpoint_path
  csv_path
  image_dir
  sample_indices = [list of 5-10 dataset indices]
  seeds = [42, 123, 456]

PROCEDURE:
  1. LOAD model, tokenizer, image_processor (Phase 1 infrastructure)
  2. SET seed(42)

  3. FOR each sample_index in sample_indices:
     a. LOAD sample = load_single_sample(csv_path, sample_index, ...)
     b. LOAD original_images = load_original_images(csv_path, sample_index, ...)
     c. COMPUTE predictions = get_prediction(model, sample, device)
     d. SAVE sample metadata

     e. FOR each score_index in [0, 1, 2, 3, 4]:
        i.   RUN image_explanation = run_lime_image(
               model, sample, score_index,
               image_processor, device,
               num_samples=1000, seed=42)
        ii.  SAVE image artifacts
        iii. RUN text_explanation = run_lime_text(
               model, sample, score_index,
               tokenizer, device,
               num_features=10, num_samples=500, seed=42)
        iv.  SAVE text artifacts

  4. STABILITY ANALYSIS
     a. SELECT 2-3 samples and 2-3 targets
     b. FOR each selected (sample, target):
        c. RUN stability analysis with seeds [42, 123, 456]
        d. SAVE stability report

  5. CROSS-VALIDATION
     a. LOAD Phase 2 Grad-CAM results
     b. LOAD Phase 3 Attention results
     c. COMPARE and document agreement/disagreement

  6. GENERATE summary statistics and findings
```

---

## 11. Validation

### 11.1 Preprocessing Consistency Check

**Test:** For the unperturbed original image, verify that the LIME predict function produces the same output as direct model inference.

**Procedure:**
1. Load a sample using `load_single_sample()`.
2. Get direct prediction: `direct_pred = get_prediction(model, sample, device)`.
3. Load the original image as numpy array.
4. Run through `ImageLimePredictFn.__call__([image_np])`.
5. Compare: `|sigmoid(direct_pred[target]) - lime_pred[0, 1]| < 0.01`.
6. If this fails, there is a preprocessing mismatch between the LIME wrapper and the training pipeline. Debug by comparing intermediate tensors.

**Do the same for text:** Run `TextLimePredictFn.__call__([original_text])` and compare with direct prediction.

### 11.2 LIME Stability Check

**Test:** Run LIME with 3 different seeds (42, 123, 456) and verify that top-K features are consistent.

**Criteria:**
- For text LIME: at least 3 of top-5 words should appear in 2/3 or 3/3 runs (60%+ Jaccard similarity for top-5).
- For image LIME: at least 3 of top-5 superpixels should appear in 2/3 or 3/3 runs.
- If stability is poor (<50% overlap), increase `num_samples` to 2000 and re-test.

### 11.3 Sanity Check -- Expected Word Behavior

**Test:** For samples containing known sentiment words, verify LIME assigns expected sign:

| Word | Expected Target | Expected LIME Sign |
|------|----------------|-------------------|
| `ngon` (delicious) | `food_score` | Positive |
| `do an ngon` (delicious food) | `food_score` | Positive |
| `gia` (price) | `price_score` | Negative or context-dependent |
| `cao` (high/expensive) | `price_score` | Negative |
| `dep` (beautiful) | `atmosphere_score` | Positive |
| `nhan vien` (staff) | `service_score` | Context-dependent |
| `tot` (good) | Multiple targets | Positive |

If LIME assigns the opposite sign (e.g., "ngon" has negative weight for `food_score`), investigate:
- Is the model prediction correct for this sample?
- Is the sigmoid mapping producing sufficient sensitivity?
- Is the perturbation count too low?

### 11.4 Sanity Check -- Expected Image Region Behavior

**Test:** For a sample with a clear food photo, verify that:
- Positive LIME superpixels for `food_score` overlap with the food item.
- Negative LIME superpixels for `food_score` are in background regions.
- For `atmosphere_score`, positive superpixels shift toward interior/ambiance regions.

### 11.5 Cross-Method Consistency

**Test:** Compare LIME results with Phase 2 and Phase 3 results for the same sample and target.

Image cross-validation:
- Overlap between Grad-CAM hot regions (top 20% activation) and LIME positive superpixels.
- Method: visual side-by-side comparison (qualitative). Quantitative comparison is optional: compute percentage of LIME positive superpixel area that overlaps with Grad-CAM top-20% region.

Text cross-validation:
- Overlap between Attention top-attended tokens and LIME top-importance words.
- Method: compute intersection of top-K lists.
- Expected: partial overlap. Complete overlap is unlikely because Attention measures information flow (internal) while LIME measures perturbation response (external).

### 11.6 Reproducibility Check

**Test:** Run the entire Phase 5 notebook twice with the same seed. Verify all output artifacts are bit-for-bit identical.

**Procedure:**
1. Run notebook with seed=42.
2. Copy all output artifacts to a temporary directory.
3. Delete output artifacts.
4. Run notebook again with seed=42.
5. Compare: `diff` all JSON files, `md5sum` all PNG files.
6. Expected: identical.

---

## 12. Risks -- Fully Analyzed

### R1: LIME Computational Cost

**Problem:** Each LIME explanation requires hundreds to thousands of model forward passes per sample per target. The multimodal model (Swin-B + PhoBERT + CrossAttentionFusion) is large and slow.

**Why it happens:** LIME generates N perturbed versions of the input, runs all N through the model, collects predictions, and fits a local linear surrogate. For Image LIME with `num_samples=1000` and batch_size=64, this means ~16 forward passes of the full multimodal model per target per sample. With 5 targets and 10 samples, this is 800 batches of 64 = 51,200 model forward passes.

**Estimation:**
- Single forward pass on GPU (batch=64): ~0.5--1.0 seconds for the full multimodal model.
- Per sample per target (1000 perturbations, batch=64): ~16 batches x ~0.75s = ~12 seconds.
- Per sample (5 targets, image + text): ~12s x 5 (image) + ~8s x 5 (text) = ~100 seconds on GPU.
- 10 samples total: ~1000 seconds = ~17 minutes on GPU.
- On CPU: multiply by ~10x = ~170 minutes.

**Possible strategies:**

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| A: Full 1000 samples | Standard LIME with num_samples=1000 | Stable explanations | ~17 min on GPU for 10 samples |
| B: Reduced 500 samples | Use num_samples=500 | ~50% faster | Less stable, may miss subtle features |
| C: Selected case studies only (5-10 samples) | Only explain curated samples, not full dataset | Tractable, focused on thesis needs | Cannot generalize findings |
| D: Skip LIME entirely | Rely on SHAP + Grad-CAM + Attention | Maximum time savings | Lose perturbation-based validation |
| E: Reduce targets | Only explain food_score and overall_satisfaction (2 of 5) | ~60% faster | Miss target-specific analysis |

**Engineering trade-offs:** Strategy A produces the most stable results but is only feasible on GPU. Strategy C reduces total samples but maintains explanation quality per sample. Strategy D is viable for thesis (SHAP + Grad-CAM + Attention is already a strong three-method framework) but loses the "perturbation-based validation" argument.

**Research trade-offs:** More samples with fewer perturbations (B) may produce noisier explanations, weakening the validation argument. Fewer samples with full perturbations (C) limits generalizability but each individual explanation is trustworthy.

**FINAL DECISION:** Strategy C -- Run LIME on 5--10 carefully selected case study samples, with `num_samples=1000` for Image LIME and `num_samples=500` for Text LIME. Explain all 5 targets per sample. Fix seed for reproducibility. Run 3-seed stability analysis on 2--3 selected (sample, target) pairs. Total estimated time on GPU: ~15--20 minutes. On CPU: ~2.5--3 hours, which is acceptable for a one-time thesis analysis.

**Reason:** This balances explanation quality (1000 perturbations for image), coverage (all 5 targets), and tractability (5--10 samples). The thesis presents LIME as "local case study validation," not as a dataset-wide analysis tool, so limited sample coverage is methodologically appropriate.

---

### R2: Regression Adaptation for LIME

**Problem:** LIME's `LimeImageExplainer` and `LimeTextExplainer` expect the prediction function to return classification-like probabilities `[N, num_classes]`. The model outputs regression scores in approximately [1, 10].

**Why it happens:** LIME was designed for classification. The `explain_instance()` method internally expects `classifier_fn` output shape `[N, num_classes]` where values are interpretable as class probabilities.

**Possible strategies:**

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| A: Sigmoid mapping | `score -> sigmoid(score) -> [1-p, p]` | Simple, standard practice, works with Image/Text explainers | Sigmoid saturates for scores > 3, reducing sensitivity |
| B: Score normalization + sigmoid | `score -> (s-1)/9 -> sigmoid(6*s_norm - 3) -> [1-p, p]` | Better sensitivity spread across score range | More complex, harder to interpret sigmoid threshold |
| C: Direct regression wrapper | Return `[N, 1]` regression scores directly | No mapping needed | `LimeImageExplainer` does not support regression mode; `LimeTabularExplainer` does but cannot segment images |
| D: Binned classification | Map scores to bins (e.g., [1-3]=low, [4-6]=mid, [7-10]=high) | LIME works naturally with multi-class output | Loses regression resolution, arbitrary bin boundaries |

**Engineering trade-offs:** Strategy A is simplest to implement and is widely used in the LIME-for-regression literature. The sigmoid saturation concern is mitigated by the fact that LIME measures *relative* changes: even if sigmoid(7.5) and sigmoid(6.8) are both close to 1.0, the difference is consistent and detectable by LIME's linear surrogate. Strategy B addresses the saturation issue but introduces arbitrary normalization parameters.

**Research trade-offs:** Strategy A is defensible: "We applied sigmoid to convert regression scores to pseudo-probabilities, following standard practice for LIME with regression models." Strategy C would require switching to `LimeTabularExplainer`, which does not provide superpixel-based image explanations or word-level text explanations.

**FINAL DECISION:** Strategy A (direct sigmoid). Use `torch.sigmoid(score)` to create a two-column `[1-p, p]` output. If initial results show poor sensitivity (all perturbations produce nearly identical pseudo-probabilities), implement Strategy B as a fallback. Document the mapping in the thesis methods section:

> "Since LIME expects classification-like output, we mapped each regression target score through a sigmoid function to produce a pseudo-probability, consistent with standard practice for applying LIME to regression models. The local surrogate then models perturbation effects on this pseudo-probability."

---

### R3: LIME Instability Across Runs

**Problem:** LIME explanations can vary significantly with different random seeds because the perturbation sampling is stochastic.

**Why it happens:** LIME generates perturbations by randomly sampling binary masks over interpretable features (superpixels or words). Different random seeds produce different sets of perturbations, which lead to different local surrogate models and different feature rankings.

**Possible strategies:**

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| A: Fix single seed | Use seed=42 for all runs | Perfectly reproducible | May miss variability; results depend on one sampling |
| B: Multi-seed consensus | Run seeds [42, 123, 456], report features in 2/3+ | More robust, reveals variability | 3x computational cost |
| C: Increase num_samples | Use 5000+ perturbations for more stable single-seed run | Reduces variance within one run | Very slow; 5x cost for image LIME |

**Engineering trade-offs:** Strategy B is 3x the cost of Strategy A but provides valuable stability evidence for the thesis. Strategy C is prohibitively expensive for the multimodal model.

**Research trade-offs:** A thesis that reports LIME with only one seed is vulnerable to the criticism "how do you know this isn't random?" Reporting 3-seed consensus demonstrates methodological rigor.

**FINAL DECISION:** Primary analysis uses fixed seed=42 for reproducibility. Additionally, for 2--3 selected (sample, target) pairs, run all three seeds [42, 123, 456] and report consensus. Features appearing in 2/3 or 3/3 runs are labeled "stable." Features appearing in only 1/3 are labeled "unstable." The thesis reports: "We verified LIME stability by repeating the analysis with three different seeds. X% of top-5 features were consistently identified, indicating [stable/moderately stable/unstable] explanations."

---

### R4: Vietnamese Text Segmentation for LIME Text

**Problem:** LIME splits text into features using a configurable `split_expression`. For Vietnamese, compound words span multiple space-separated syllables.

**Why it happens:** Vietnamese is a monosyllabic-rooted language where multi-syllable words are written with spaces: "nha hang" (restaurant), "khong gian" (atmosphere), "nhan vien" (staff).

**Possible strategies:**

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| A: Whitespace splitting | Use default `r'\s+'` split | Simple; aligns with PhoBERT syllable-level tokenization | Compound words split into syllables; cannot test "nha hang" as unit |
| B: Vietnamese word segmenter | Use VnCoreNLP or underthesea for word segmentation | Linguistically correct compound words | Adds heavy dependency; segmentation errors introduce noise; LIME features no longer align with PhoBERT tokens |
| C: Custom split with known compounds | Manually define a list of compound words to keep together | Targeted improvement for known aspect terms | Does not scale; ad hoc |

**Analysis:**

PhoBERT tokenizes at the syllable level with BPE. When LIME removes a word (e.g., "hang" from "nha hang"), PhoBERT still processes the remaining syllables individually. This means LIME's whitespace splitting is actually well-aligned with the model's tokenization:
- If "hang" is removed, PhoBERT tokenizes the remaining text without "hang" -- the same as if the model never saw that syllable.
- If a Vietnamese word segmenter were used and "nha hang" were treated as one feature, LIME would remove both syllables together. This tests a different perturbation (removing the entire concept "restaurant") which is valid but not what PhoBERT sees internally.

**Research trade-off:** Syllable-level features provide finer-grained importance (which syllable matters more?) but cannot express compound word importance. For the thesis, the granularity is acceptable because: (a) most aspect-bearing words in Vietnamese reviews are recognizable at the syllable level ("ngon", "dat", "dep", "sach"), and (b) the thesis can state: "LIME features are Vietnamese syllables, consistent with PhoBERT's BPE tokenization."

**FINAL DECISION:** Strategy A (whitespace splitting). Document that features are syllable-level. If an interesting compound word analysis is needed (e.g., "nha hang"), manually note that both syllables should be considered together in the interpretation.

---

### R5: Multi-Image LIME

**Problem:** The model supports up to 4 images per review with masked average pooling. LIME perturbs one image at a time. How should multi-image reviews be handled?

**Why it happens:** `ImageModel.forward()` in `Models/ImageModel.py` processes all images through Swin-B, then applies masked average pooling using `num_images`. LIME's `LimeImageExplainer` works on a single image.

**Possible strategies:**

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| A: First image only | Explain first real image, keep others fixed | Simple, consistent with Phase 2 (Grad-CAM) primary analysis | Misses other images' contributions |
| B: All images separately | Run LIME on each real image independently | Comprehensive coverage | N_images x cost; results for one image depend on others being fixed |
| C: Highest Grad-CAM image | Select the image with strongest Grad-CAM activation for this target | Focuses on most important image | Requires Phase 2 results; introduces Phase 2 dependency |
| D: Concatenated image | Stitch images into a 2x2 grid, run LIME on the composite | Tests all images at once | Superpixels span across different images; hard to interpret |

**Engineering trade-offs:** Strategy A is O(1) per sample. Strategy B is O(N_images) per sample but gives richer analysis. Strategy C adds a dependency on Phase 2 being complete. Strategy D creates interpretation ambiguity.

**Research trade-offs:** The thesis focuses on whether the model uses image evidence at all, not on fine-grained per-image analysis. Strategy A answers this question for the primary image. Strategy B would be ideal for a dedicated multi-image analysis but is out of scope for LIME's role as a "local sanity check."

**FINAL DECISION:** Strategy A for the default implementation: explain only the first real image, keeping remaining images fixed. For 1--2 selected multi-image case studies (where the review has 3--4 real images), optionally apply Strategy B (explain each image separately) to test whether different images support different targets. This extended analysis is optional and only performed if time permits.

---

### R6: Image Preprocessing Consistency

**Problem:** LIME provides perturbed images as numpy arrays. These must be preprocessed identically to the training pipeline before being fed to the model.

**Why it happens:** The training pipeline uses `TimmProcessor` (which wraps `timm.data.create_transform` with `is_training=False`). This applies specific normalization (ImageNet mean/std), resize, and center crop. If the LIME predict function applies different preprocessing, the model receives out-of-distribution inputs and explanations become unreliable.

**Possible strategies:**

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| S1: numpy -> PIL -> TimmProcessor | Convert LIME's numpy array to PIL, then apply TimmProcessor | Exact match with training pipeline | Requires numpy-to-PIL conversion per image per perturbation |
| S2: Manual normalization | Apply ImageNet normalization directly to numpy tensor | Avoids PIL conversion overhead | Must manually replicate all TimmProcessor steps; error-prone |
| S3: Skip preprocessing | Normalize LIME's numpy to [0,1] only | Fast | Completely wrong; model receives incorrectly normalized inputs |

**Engineering trade-offs:** S1 is 10--20% slower than S2 due to PIL conversion overhead, but it guarantees identical preprocessing. S2 risks subtle differences in resize interpolation, normalization order, or crop behavior.

**Research trade-offs:** Preprocessing mismatch would silently produce incorrect explanations. The validation check (Section 11.1) catches this, but S1 prevents the problem entirely.

**FINAL DECISION:** Strategy S1 -- Convert numpy to PIL, then apply TimmProcessor. The overhead is negligible compared to the model forward pass time. Validate with the preprocessing consistency check (Section 11.1) before running full analysis.

---

### R7: GPU Memory Management

**Problem:** LIME generates up to 1000 perturbed samples for image LIME. Processing all 1000 through the multimodal model at once will exceed GPU memory.

**Why it happens:** Each perturbed sample requires `pixel_values [1, 4, 3, 224, 224]` (~2.4 MB float32) + `input_ids [1, 256]` + model activations. With 1000 samples, the batch would require ~2.4 GB just for image tensors plus model activation memory.

**Possible strategies:**

| Strategy | Description | Advantage | Disadvantage |
|----------|-------------|-----------|--------------|
| A: Internal batching (batch_size=64) | Process perturbations in chunks of 64 | Manageable GPU memory (~300 MB per batch) | Slightly slower due to multiple forward passes |
| B: Small batch (batch_size=16) | Process in very small chunks | Minimal GPU memory | Much slower |
| C: CPU inference | Run LIME on CPU | No GPU memory issue | 5--10x slower |

**FINAL DECISION:** Strategy A with `batch_size=64` as default. For GPUs with less than 8 GB VRAM (common on Google Colab free tier), reduce to `batch_size=32` or `batch_size=16`. The batch_size parameter is exposed in the function signature so users can adjust it.

---

### R8: LIME Label Interpretation for Regression

**Problem:** LIME's `explain_instance()` returns explanations indexed by "label" (class index). With the sigmoid pseudo-classification, label=0 is "low" and label=1 is "high." Which label should be used for explanation?

**Why it happens:** The two-column output `[1-p, p]` represents two pseudo-classes. Features that increase the score have positive weights for label=1 ("high") and negative weights for label=0 ("low").

**Decision:** Always explain label=1 ("high"). A positive LIME weight for label=1 means "this feature locally increases the target score." A negative weight means "this feature locally decreases the target score." This provides intuitive interpretation:
- Positive weight for "ngon" for `food_score` = "ngon" locally increases food score.
- Negative weight for "cao" for `price_score` = "cao" locally decreases price score.

Document this convention clearly in the notebook and in all saved metadata.

---

## 13. Best Practices

### 13.1 Deterministic Execution

- Always call `set_seed(seed)` before each LIME run, not just at notebook start. LIME's internal randomness is controlled via `random_seed` (image) or `random_state` (text) parameters, but surrounding code may also use random functions.
- Pass `random_seed` to `LimeImageExplainer.explain_instance()` and `random_state` to `LimeTextExplainer.__init__()`.
- Save the seed in `lime_config.json` for each explanation.

### 13.2 Model State

- Verify `model.training == False` before every LIME run. A model in training mode with dropout produces non-deterministic predictions.
- Use `torch.no_grad()` in all predict functions to prevent gradient computation and save memory.
- Never modify model parameters during LIME analysis.

### 13.3 Memory Optimization

- Use internal batching (batch_size=64) in predict functions to avoid GPU OOM.
- Delete LIME explanation objects after saving artifacts if running many samples to free memory: `del explanation; gc.collect()`.
- Use `torch.cuda.empty_cache()` between samples if GPU memory is tight.

### 13.4 Artifact Naming and Organization

- Follow the naming convention established in Phase 1: `sample_{id}_target{idx}_{name}_{type}_{variant}.{ext}`.
- Use `INDEX_TO_FACTOR` from `xai/config.py` for factor short names; never hardcode.
- Create output directories with `os.makedirs(path, exist_ok=True)` at the start of the notebook.
- Save both visual artifacts (PNG for thesis figures) and raw data artifacts (JSON for numerical analysis).

### 13.5 Figure Quality

- Use `THESIS_DPI = 300` for all saved figures (defined in `xai/config.py`).
- Use `plt.tight_layout()` before saving to prevent label clipping.
- For image overlays, use `mark_boundaries()` from `skimage.segmentation` with visible boundary colors.
- For text bar charts, ensure Vietnamese diacritical characters render correctly. Test with a sample containing `a`, `o`, `u` with diacritics.
- Use consistent color scheme from `COLOR_SCHEMES` in `xai/config.py`:
  - Green for positive LIME weights.
  - Red for negative LIME weights.
  - Target-specific colors for multi-target comparisons.

### 13.6 Logging and Progress Tracking

- Use Python `logging` module with INFO level for progress and WARNING for issues.
- Log start/end time for each LIME explanation.
- Log total forward passes performed.
- Print progress in the notebook: "Processing sample {id}, target {idx} ({name})..."

### 13.7 Error Handling

- Catch and log (but do not crash on) individual sample failures:
  - Missing image files: fall back to black image, log warning.
  - Empty text after perturbation: LIME may produce a text with all words removed. The tokenizer should handle empty strings gracefully (producing only special tokens).
  - Model inference errors: log the error, skip this (sample, target) pair, continue.
- Save partial results: if the notebook crashes mid-way, artifacts saved so far are not lost.

### 13.8 Vietnamese-Specific Considerations

- Verify that matplotlib renders Vietnamese characters correctly. If not, install a font that supports Vietnamese (e.g., `DejaVu Sans`) and set `plt.rcParams['font.family']` appropriately.
- In word importance bar charts, display the full Vietnamese syllable with diacritics, not transliterated or ASCII approximations.
- When saving JSON files with Vietnamese text, use `json.dump(..., ensure_ascii=False)` to preserve Unicode characters.

---

## 14. Deliverables

### 14.1 Code

| Deliverable | Path | Description |
|-------------|------|-------------|
| LIME explainer module | `xai/lime_explainer.py` | Complete module with all classes and functions specified in Section 6 |
| LIME notebook | `xai/notebooks/Phase5_LIME.ipynb` | End-to-end analysis notebook with all cells specified in Section 9 |

### 14.2 Image LIME Artifacts (Per Sample, Per Target)

| Deliverable | Count (10 samples x 5 targets) | Description |
|-------------|-------------------------------|-------------|
| Positive superpixel overlays | 50 PNGs | Superpixels with positive weight highlighted |
| Negative superpixel overlays | 50 PNGs | Superpixels with negative weight highlighted |
| Combined superpixel overlays | 50 PNGs | Both positive and negative, color-coded |
| Superpixel weight JSONs | 50 JSONs | Raw superpixel_id -> weight mappings |

### 14.3 Text LIME Artifacts (Per Sample, Per Target)

| Deliverable | Count (10 samples x 5 targets) | Description |
|-------------|-------------------------------|-------------|
| Word importance bar charts | 50 PNGs | Top-10 words by absolute LIME weight |
| Interactive HTML reports | 50 HTMLs | LIME's as_html() output with color-coded words |
| Word weight JSONs | 50 JSONs | Raw word -> weight mappings |

### 14.4 Configuration and Metadata

| Deliverable | Count | Description |
|-------------|-------|-------------|
| LIME config JSONs | 100 (50 image + 50 text) | Hyperparameters used for each explanation |
| Sample metadata JSONs | 10 | Ground truth, predictions, sample info |
| Stability reports | 4--6 (2--3 pairs x text/image) | Multi-seed consensus analysis |

### 14.5 Cross-Validation Figures

| Deliverable | Count | Description |
|-------------|-------|-------------|
| Grad-CAM vs LIME side-by-side | 2--3 | Image explanation cross-validation |
| Attention vs LIME comparison tables | 2--3 | Text explanation cross-validation |
| Per-target comparison panels | 2--3 | Multi-target LIME visualization |

### 14.6 Summary

**Total deliverables:** ~320 artifact files + 2 code files.

---

## 15. Thesis Usage

### 15.1 Results Chapter

LIME results support a dedicated subsection titled (example): "Perturbation-Based Local Validation with LIME." This subsection:

1. **Introduces LIME methodology** (2--3 paragraphs) explaining the local surrogate approach, sigmoid regression adaptation, and Vietnamese syllable-level features.
2. **Presents image LIME case studies** (2--3 figures): superpixel overlays showing that food-related regions positively support `food_score`, ambiance-related regions support `atmosphere_score`, etc.
3. **Presents text LIME case studies** (2--3 figures): word importance bar charts showing that aspect-specific words have expected LIME weights.
4. **Reports stability** (1 table or paragraph): consensus features across 3 seeds.

### 15.2 Discussion Chapter

LIME contributes to three discussion points:

1. **Multi-method agreement:** "LIME image explanations aligned with Grad-CAM spatial evidence for X of Y samples, confirming that gradient-based and perturbation-based methods converge."
2. **Multi-method disagreement as diagnostic:** "In sample Z, LIME and Grad-CAM disagreed on the primary food region. This may indicate that the Grad-CAM gradient signal is influenced by batch normalization artifacts, while LIME's perturbation test directly measures prediction sensitivity."
3. **Model faithfulness:** "The model decreases `price_score` when price-related words are removed (LIME evidence), increases `food_score` when food-presentation regions are present (LIME + Grad-CAM evidence), and allocates more fusion weight to text for price-related predictions (SHAP evidence). Together, these findings demonstrate that the model has learned interpretable quality signals."

### 15.3 Case Studies (Phase 6)

LIME artifacts are integrated into comprehensive case study panels in Phase 6. Each case study sample shows:
- Original image + Grad-CAM heatmap + LIME superpixel overlay (three images in a row)
- Original text + Attention heatmap + LIME word bar chart (three visualizations in a row)
- SHAP modality contribution bar (below)
- Prediction vs ground truth table (below)

### 15.4 Defense Presentation

LIME content for 2--3 defense slides:

**Slide 1: "Perturbation-Based Validation"**
- Show one image LIME example: original image -> positive superpixels -> negative superpixels
- Caption: "Removing food-presentation regions locally decreases food_score by X points"

**Slide 2: "Text Evidence Validation"**
- Show one text LIME bar chart
- Caption: "Removing 'ngon' locally decreases food_score; removing 'gia cao' locally increases price_score"

**Slide 3: "Multi-Method Agreement"**
- Side-by-side: Grad-CAM heatmap vs LIME superpixels for same sample
- Caption: "Gradient-based (Grad-CAM) and perturbation-based (LIME) methods identify consistent food-related regions"

### 15.5 Journal Paper

For a journal paper, LIME provides the "perturbation validation" paragraph that reviewers expect in XAI papers. Key sentence:

> "To validate our gradient-based explanations, we applied LIME as an independent perturbation-based sanity check. Image LIME confirmed that food-presentation superpixels locally support the food quality score, while text LIME confirmed that aspect-specific Vietnamese words (e.g., 'ngon' for food quality, 'gia' for price) exhibit expected local importance patterns."

### 15.6 Thesis Appendix (If Time-Constrained)

If LIME is deprioritized, all results move to the thesis appendix:
- Appendix section: "LIME Local Perturbation Analysis"
- Include 2--3 representative examples
- Reference from main text: "Additional perturbation-based validation using LIME is presented in Appendix X."
- The core thesis argument (Grad-CAM + Attention + SHAP) remains intact.

---

## 16. Phase Completion Checklist

### 16.1 Code Completion

- [ ] `xai/lime_explainer.py` created and importable
- [ ] `ImageLimePredictFn` class implemented with correct preprocessing
- [ ] `TextLimePredictFn` class implemented with correct tokenization
- [ ] `run_lime_image()` function returns valid `ImageExplanation` object
- [ ] `run_lime_text()` function returns valid `TextExplanation` object
- [ ] `save_lime_image_explanation()` generates positive, negative, and combined PNGs
- [ ] `save_lime_text_explanation()` generates word importance bar chart PNG
- [ ] `save_lime_text_html()` generates interactive HTML
- [ ] `save_lime_config()` saves hyperparameter JSON
- [ ] `save_lime_metadata()` saves sample metadata JSON
- [ ] `load_original_images()` correctly loads pre-preprocessing PIL images
- [ ] `run_lime_stability_analysis()` runs multi-seed analysis and reports consensus

### 16.2 Validation

- [ ] Preprocessing consistency check passes: LIME predict_fn output matches direct inference within tolerance < 0.01
- [ ] Text preprocessing check passes: TextLimePredictFn output matches direct inference within tolerance < 0.01
- [ ] Stability check: top-5 LIME text features show >= 60% Jaccard overlap across 3 seeds for at least 2 of 3 tested (sample, target) pairs
- [ ] Sanity check: "ngon" has positive LIME weight for `food_score` in samples where it appears
- [ ] Sanity check: food-related superpixels have positive LIME weight for `food_score`
- [ ] Reproducibility check: two identical seed runs produce identical artifacts

### 16.3 Artifacts

- [ ] Image LIME artifacts generated for all selected samples (5--10) and all 5 targets
- [ ] Text LIME artifacts generated for all selected samples (5--10) and all 5 targets
- [ ] LIME config JSONs saved for every explanation
- [ ] Sample metadata JSONs saved for every sample
- [ ] Stability reports saved for 2--3 (sample, target) pairs
- [ ] Cross-validation figures (LIME vs Grad-CAM, LIME vs Attention) generated for 2--3 samples

### 16.4 Notebook

- [ ] `xai/notebooks/Phase5_LIME.ipynb` runs end-to-end without error
- [ ] All 13 cells produce expected output
- [ ] Vietnamese characters render correctly in all figures
- [ ] All artifacts are saved to the correct paths following naming conventions
- [ ] Notebook includes written analysis and findings in markdown cells

### 16.5 Quality Assurance

- [ ] All PNGs are at 300 DPI (THESIS_DPI)
- [ ] All JSONs use `ensure_ascii=False` for Vietnamese text
- [ ] All file paths follow the naming convention from `xai/config.py`
- [ ] No hardcoded paths or magic numbers in the code (all from config)
- [ ] Memory management: no GPU OOM during full notebook execution
- [ ] Error handling: individual sample failures do not crash the notebook

### 16.6 Documentation

- [ ] Module-level docstring in `lime_explainer.py` explaining the module's purpose
- [ ] Function-level docstrings for all public functions with parameter descriptions
- [ ] Inline comments explaining the sigmoid mapping rationale
- [ ] Inline comments explaining the Vietnamese syllable splitting decision
- [ ] Notebook includes markdown cells documenting methodology, findings, and limitations

---

## Appendix A: Expected Thesis Method Paragraph

> LIME (Local Interpretable Model-agnostic Explanations) was employed as a perturbation-based local validation method to independently verify the findings from gradient-based (Grad-CAM) and attribution-based (SHAP) analyses. For image explanations, the input image was segmented into superpixels and subsets of segments were randomly hidden while the text input remained fixed. For text explanations, subsets of Vietnamese syllables were removed while the image input remained fixed. A local weighted linear surrogate model was fitted near each original sample to estimate the local importance of each interpretable feature. Since the model produces regression outputs, the prediction scores were mapped through a sigmoid function to create pseudo-classification probabilities suitable for the LIME framework. All text features were defined at the Vietnamese syllable level, consistent with the PhoBERT tokenizer's BPE granularity. Stability of LIME explanations was verified by repeating the analysis with three different random seeds (42, 123, 456) and reporting features that appeared consistently in the majority of runs.

## Appendix B: Expected Thesis Results Paragraph

> LIME image explanations for representative samples confirmed that food-presentation superpixels locally supported higher food quality scores, while background and non-food regions exhibited neutral or negative local importance. LIME text explanations revealed that aspect-specific Vietnamese syllables such as "ngon" (delicious) positively contributed to food_score predictions, while "gia" (price) and "cao" (high/expensive) negatively contributed to price_score predictions. These perturbation-based findings were consistent with the Grad-CAM spatial evidence (Phase 2) and the self-attention information flow patterns (Phase 3), strengthening the claim that the multimodal model has learned interpretable, aspect-specific quality signals from Vietnamese restaurant reviews.

## Appendix C: Defense Q&A Preparation

| Question | Strong Answer |
|----------|---------------|
| Why use LIME if you already have Grad-CAM and SHAP? | LIME provides independent perturbation-based validation. Grad-CAM uses internal gradients; SHAP uses feature attribution on embeddings; LIME tests from the input side by removing human-interpretable units. Agreement across methods strengthens the explanatory claim. |
| How did you adapt LIME to regression? | We mapped each regression score through a sigmoid function to create a two-column pseudo-probability output, following standard practice for LIME with regression models. |
| Why use syllable-level features for Vietnamese? | PhoBERT tokenizes Vietnamese text at the syllable level using BPE. Syllable-level LIME features align with the model's tokenization granularity, ensuring that removing a LIME feature corresponds to removing a meaningful unit from the model's perspective. |
| Is LIME stable for your model? | We verified stability by running LIME with three different random seeds. For the tested samples, X% of top-5 features appeared consistently across runs, indicating [stable/moderately stable] explanations. |
| What if LIME and Grad-CAM disagree? | Disagreement is diagnostic, not embarrassing. LIME measures local perturbation response (external); Grad-CAM measures gradient-weighted spatial activation (internal). Disagreement may reveal that the model uses a region for reasons not captured by simple occlusion, or that gradient flow through batch normalization creates artifacts. We document both agreements and disagreements transparently. |
| What are LIME's limitations? | LIME explanations are local (valid near one sample, not globally), sensitive to the number of perturbation samples, dependent on the segmentation algorithm for images, and computationally expensive. We mitigate these by using 1000 perturbations for images, verifying stability with multiple seeds, and clearly framing results as local evidence rather than global conclusions. |
