# Phase 1: XAI Infrastructure Setup -- Implementation Proposal

---

## 1. Purpose

### Why This Phase Exists

Phase 1 is the foundational layer upon which all subsequent XAI phases (Grad-CAM, Attention, SHAP, LIME, Case Studies, Report Generation, Thesis Visualization) depend. No XAI method can be implemented without first guaranteeing that the model loads correctly, that preprocessing reproduces training-time behavior exactly, that single-sample inference yields numerically verified predictions, and that intermediate tensors (text features, image features, attention weights, spatial feature maps) can be reliably extracted.

### Research Motivation

The thesis investigates explainability of a multimodal deep learning system for Vietnamese restaurant review quality assessment. Before any explanation can be trusted, the inference pipeline itself must be verified to be deterministic, reproducible, and numerically consistent with the training pipeline. A prediction that differs from the training-time output invalidates any downstream explanation. Phase 1 establishes this trust boundary.

### Engineering Motivation

The current codebase separates model building (main.py, test.py) from experiment management (Trainer.py, Config.py) and data handling (src/dataset.py). XAI requires a different operational mode: single-sample inference with intermediate tensor extraction, hook-based feature map capture, and attention retrieval. These capabilities do not exist in the current codebase and must be built as clean, reusable utilities that all subsequent phases import.

The existing demo notebook (`notebook/demo_single_sample_exp060A.ipynb`) proves the concept but contains inline code that is not reusable. Phase 1 extracts this logic into importable Python modules within the `xai/` package.

---

## 2. Objectives

### Research Objectives

1. **RO-1:** Establish a verified, deterministic inference pipeline for the best model configuration (Swin-B + PhoBERT + CrossAttentionFusion + LogCosh) that produces predictions matching the training pipeline within numerical tolerance.
2. **RO-2:** Confirm that intermediate representations (text features [B, 768], image features [B, 1024]) maintain expected shapes and value ranges.
3. **RO-3:** Verify that the text encoder (PhoBERT) supports `output_attentions=True` and that the image encoder (Swin-B via timm) produces accessible spatial feature maps before global pooling.
4. **RO-4:** Guarantee full reproducibility via seed control across all random sources (Python, NumPy, PyTorch, CUDA).

### Engineering Objectives

1. **EO-1:** Create a self-contained `xai/` Python package with utility modules (`utils.py`, `config.py`, `__init__.py`) that all subsequent phases import.
2. **EO-2:** Implement `load_model()` that reconstructs the full CrossAttentionFusion model and loads a checkpoint, with device auto-detection (CUDA, MPS, CPU).
3. **EO-3:** Implement `load_single_sample()` that reads one row from a CSV, tokenizes the text, processes all images (with correct padding), and returns ready-to-use tensors.
4. **EO-4:** Implement `get_prediction()` that runs inference and returns predictions as a dictionary keyed by target name.
5. **EO-5:** Implement artifact-saving utilities (`save_figure()`, `save_raw_values()`) with consistent naming, DPI control, and metadata.
6. **EO-6:** Create a verification notebook that validates all infrastructure components end-to-end.

### Expected Contributions

- A reusable XAI utility package that eliminates code duplication across Phases 2--8.
- Numerical verification that the XAI pipeline produces identical predictions to the training pipeline.
- Documented extraction procedures for intermediate tensors required by Grad-CAM (spatial feature maps), Attention (attention matrices), SHAP (fused embeddings), and LIME (prediction functions).

---

## 3. Inputs

### Checkpoint File

| Input | Path | Description |
|---|---|---|
| Fusion checkpoint | `experiments/EXP_060A_bestsequential_full_configuration/best_model_train_fusion.pth` | Best model weights. Contains keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_loss`, `best_mean_mae`, `args`. |

### Configuration File

| Input | Path | Description |
|---|---|---|
| Experiment config | `experiments/EXP_060A_.../config.yaml` (or `config.json`) | Saved `vars(args)` containing `text_model_name`, `image_model_name`, `fusion_type`, `loss_fn`, `max_length`, `seed`, etc. |

### Data Files

| Input | Path | Description |
|---|---|---|
| Validation CSV | `./data/text/val.csv` | Columns: `comment_clean`, `image_url`, `food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction` |
| Test CSV | `./data/text/test.csv` | Same schema as val.csv |
| Image cache | `./data/image/` | JPEG files named by MD5 hash of original URL |

### Existing Predictions

| Input | Path | Description |
|---|---|---|
| Validation predictions | `experiments/EXP_060A_.../predictions.csv` | Columns: `index`, `split`, `y_true_{factor}`, `y_pred_{factor}`, `absolute_error_{factor}` for factor in [food, price, atmos, service, overall] |
| Test predictions | `experiments/EXP_060A_.../test_predictions.csv` | Same schema, split=test |

### Source Code Files

| Input | Path | Description |
|---|---|---|
| TextModel | `Models/TextModel.py` | PhoBERT wrapper. Returns `(factor_head_output, raw_features)` where `raw_features` is `[B, 768]`. |
| ImageModel | `Models/ImageModel.py` | Swin-B via timm. Returns `(factor_head_output, raw_features)` where `raw_features` is `[B, 1024]`. Handles multi-image with masked average pooling. |
| CrossAttentionFusion | `Models/CrossAttentionFusion.py` | Bidirectional cross-attention. Takes `text_feat [B, 768]` and `image_feat [B, 1024]`, projects to 512, cross-attends, concatenates to `[B, 1024]`, head outputs `[B, 5]`. |
| TimmProcessor | `main.py` (class `TimmProcessor`) | Wraps `timm.data.create_transform` for image preprocessing. Must be replicated exactly. |
| Dataset | `src/dataset.py` | `MultimodalDataset` with MD5-based image loading, max 4 images, black-image padding. |

### Pre-trained Model Weights (Downloaded at Runtime)

| Input | Source | Description |
|---|---|---|
| PhoBERT | `vinai/phobert-base-v2` (HuggingFace) | Text encoder. 768-dim hidden size, 12 layers, 12 heads. |
| Swin-B | `swin_base_patch4_window7_224` (timm) | Image encoder. 1024-dim features, patch size 4, window size 7. |

---

## 4. Outputs

### Python Modules

| Output | Path | Description |
|---|---|---|
| XAI config | `xai/config.py` | Constants: target names, indices, display names, score ranges, seed, DPI, color schemes. |
| XAI utilities | `xai/utils.py` | Functions: `load_model()`, `load_single_sample()`, `get_prediction()`, `save_figure()`, `save_raw_values()`, `get_image_processor()`, `get_tokenizer()`, `set_seed()`, `get_device()`. |
| Package init | `xai/__init__.py` | Exposes public API from `config` and `utils`. |

### Verification Notebook

| Output | Path | Description |
|---|---|---|
| Infrastructure notebook | `xai/notebooks/Phase1_Infrastructure_Verification.ipynb` | End-to-end verification of all infrastructure components. |

### Verified Outputs (Saved Within Notebook Execution)

| Output | Path | Description |
|---|---|---|
| Verification report | `experiments/EXP_060A_.../xai/infrastructure/verification_report.json` | JSON containing: model loaded (bool), prediction match (bool), tolerance, tensor shapes, device info, timestamp. |
| Sample prediction | `experiments/EXP_060A_.../xai/infrastructure/sample_prediction.json` | Ground truth, predictions, absolute errors for the verified sample. |

---

## 5. Architecture Attachment Point

Phase 1 infrastructure connects to every level of the model architecture. It does not perform XAI analysis itself but exposes the attachment points that Phases 2--8 will use.

```
Input Layer                    Infrastructure Function
-------------------------------------------------------------
Text Input (comment_clean)  -> load_single_sample() -> tokenizer
  |                                                      |
  v                                                      v
PhoBERT Encoder             <- get_tokenizer()      <- AutoTokenizer
  |                            
  |-- pooler_output [B,768] <- Extracted via model.text_model()
  |-- attentions            <- Phase 3 will use output_attentions=True
  |                            (Phase 1 VERIFIES this works)
  v
                            CrossAttentionFusion.forward()
                              |
Image Input (image_url)     -> load_single_sample() -> image_processor
  |                                                      |
  v                                                      v
Swin-B Encoder              <- get_image_processor() <- TimmProcessor
  |
  |-- spatial feature map   <- Phase 2 will use hooks
  |   [B*N, 1024, 7, 7]       (Phase 1 VERIFIES this shape)
  |-- pooled [B, 1024]     <- Extracted via model.image_model()
  v
Cross-Attention Fusion
  |-- text_proj(768->512)
  |-- image_proj(1024->512)
  |-- cross_attn_t2i, cross_attn_i2t
  |-- concat [B, 1024]
  v
Prediction Head
  |-- Linear(1024->512)->ReLU->Dropout->Linear(512->256)->ReLU->Linear(256->5)
  v
Output [B, 5]               <- get_prediction() returns dict
  |-- idx 0: food_score
  |-- idx 1: price_score
  |-- idx 2: atmosphere_score
  |-- idx 3: service_score
  |-- idx 4: overall_satisfaction
```

### Key Attachment Point Details for Subsequent Phases

| Phase | Attachment Point | How Phase 1 Enables It |
|---|---|---|
| Phase 2 (Grad-CAM) | Swin-B last stage spatial feature map `[B*N, 1024, 7, 7]` | Phase 1 verifies shape via forward hook on `model.image_model.encoder` |
| Phase 3 (Attention) | PhoBERT attention matrices `tuple(12 x [B, 12, L, L])` | Phase 1 verifies `output_attentions=True` on `model.text_model.encoder` |
| Phase 4 (SHAP) | Fused embedding before prediction head | Phase 1 provides extraction of `text_features` and `image_features` |
| Phase 5 (LIME) | Full model predict function | Phase 1 provides `get_prediction()` as foundation for LIME predict_fn wrappers |
| Phase 6 (Case Studies) | All of the above | Phase 1 infrastructure is the base for case study pipelines |
| Phase 7 (Reports) | Saved artifacts | Phase 1 defines `save_figure()` and `save_raw_values()` conventions |
| Phase 8 (Thesis Viz) | Consistent styling | Phase 1 defines `COLOR_SCHEMES`, `DEFAULT_DPI`, display name conventions |

---

## 6. Detailed Implementation Plan

### Step 1: Create `xai/__init__.py`

**Purpose:** Make `xai/` an importable Python package.

**Contents:** Import and expose the public API from `config` and `utils` submodules.

**Exports:**
- From `xai.config`: `TARGET_NAMES`, `TARGET_INDICES`, `FACTOR_NAMES`, `DISPLAY_NAMES`, `LABEL_COLS`, `SCORE_RANGE`, `DEFAULT_SEED`, `DEFAULT_DPI`, `THESIS_DPI`, `COLOR_SCHEMES`
- From `xai.utils`: `load_model`, `load_single_sample`, `get_prediction`, `save_figure`, `save_raw_values`, `get_image_processor`, `get_tokenizer`, `set_seed`, `get_device`

**Implementation detail:** Use explicit imports, not wildcard. Include a module-level docstring explaining that this is the XAI infrastructure package for the multimodal quality assessment system.

### Step 2: Create `xai/config.py`

**Purpose:** Centralize all constants used across Phases 1--8 so that no phase hardcodes target names, indices, or styling.

**Constants to define:**

```
TARGET_NAMES = ['food_score', 'price_score', 'atmosphere_score', 
                'service_score', 'overall_satisfaction']

TARGET_INDICES = {
    'food_score': 0,
    'price_score': 1,
    'atmosphere_score': 2,
    'service_score': 3,
    'overall_satisfaction': 4,
}

FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']

DISPLAY_NAMES = ['Food Score', 'Price Score', 'Atmosphere Score', 
                 'Service Score', 'Overall Satisfaction']

LABEL_COLS = ['food_score', 'price_score', 'atmosphere_score', 
              'service_score', 'overall_satisfaction']

FACTOR_TO_DISPLAY = {
    'food': 'Food Score',
    'price': 'Price Score',
    'atmos': 'Atmosphere Score',
    'service': 'Service Score',
    'overall': 'Overall Satisfaction',
}

INDEX_TO_FACTOR = {0: 'food', 1: 'price', 2: 'atmos', 3: 'service', 4: 'overall'}
FACTOR_TO_INDEX = {'food': 0, 'price': 1, 'atmos': 2, 'service': 3, 'overall': 4}

SCORE_RANGE = (1, 10)

DEFAULT_SEED = 42
DEFAULT_DPI = 150
THESIS_DPI = 300

DEFAULT_MAX_LENGTH = 256
DEFAULT_MAX_IMAGES = 4

# Best model configuration
BEST_TEXT_MODEL = 'vinai/phobert-base-v2'
BEST_IMAGE_MODEL = 'swin_base_patch4_window7_224'
BEST_FUSION_TYPE = 'cross_attention'
BEST_EXP_ID = 'EXP_060A_bestsequential_full_configuration'

# Dimension constants for the best model
TEXT_FEATURE_DIM = 768
IMAGE_FEATURE_DIM = 1024
CROSS_ATTN_HIDDEN_DIM = 512
FUSED_DIM = 1024  # After cross-attention concat: 512 + 512
NUM_TARGETS = 5

# PhoBERT architecture constants
PHOBERT_NUM_LAYERS = 12
PHOBERT_NUM_HEADS = 12

# Color scheme for consistent visualization across all phases
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
```

### Step 3: Create `xai/utils.py`

**Purpose:** Provide all reusable infrastructure functions for XAI analysis.

**Functions to implement:**

#### 3a. `get_device()`

- Signature: `get_device() -> torch.device`
- Logic: Return `cuda` if `torch.cuda.is_available()`, then `mps` if `torch.backends.mps.is_available()`, else `cpu`.
- Must match the logic in `main.py` lines 51--54 and `test.py` lines 35--38 exactly.

#### 3b. `set_seed(seed: int = 42)`

- Signature: `set_seed(seed: int = DEFAULT_SEED) -> None`
- Logic: Replicate exactly the `set_seed()` function from `main.py` lines 24--34:
  - `random.seed(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)`
  - If CUDA available: `torch.cuda.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`

#### 3c. `get_tokenizer(text_model_name: str)`

- Signature: `get_tokenizer(text_model_name: str = BEST_TEXT_MODEL) -> AutoTokenizer`
- Logic: `AutoTokenizer.from_pretrained(text_model_name)`
- Returns the HuggingFace tokenizer.

#### 3d. `get_image_processor(image_model_name: str)`

- Signature: `get_image_processor(image_model_name: str = BEST_IMAGE_MODEL) -> callable`
- Logic: Replicate the fallback chain from `main.py` lines 62--69:
  1. Try `AutoImageProcessor.from_pretrained(image_model_name)`
  2. If fails and `'siglip'` in name: `AutoImageProcessor.from_pretrained('google/siglip-base-patch16-256')`
  3. Else: instantiate `TimmProcessor(image_model_name)`
- Must include the `TimmProcessor` class definition (replicated from `main.py` lines 13--22) within `utils.py` or as a private helper class.
- The `TimmProcessor.__call__` must use `is_training=False` for the transform, exactly as in the existing code.

#### 3e. `load_model(exp_dir, device, text_model_name, image_model_name, fusion_type)`

- Signature: `load_model(exp_dir: str, device: torch.device = None, text_model_name: str = None, image_model_name: str = None, fusion_type: str = None) -> tuple[nn.Module, dict]`
- Logic:
  1. If `device` is None, call `get_device()`.
  2. Attempt to load config from `exp_dir/config.yaml` (try yaml first, then json, then checkpoint `args` key). Use this to fill in `text_model_name`, `image_model_name`, `fusion_type` if not explicitly provided.
  3. Build `TextModel(model_name=text_model_name)`.
  4. Build `ImageModel(model_name=image_model_name)`.
  5. Build the fusion model based on `fusion_type`:
     - `'cross_attention'` -> `CrossAttentionFusion(text_model=text_model, image_model=image_model)`
     - `'gmu'` -> `GMUFusion(...)`
     - `'gated_cross'` -> `GatedCrossModalFusion(...)`
     - `'film'` -> `FiLMFusion(...)`
     - default -> `FusionModel(...)`
  6. Locate checkpoint: `exp_dir/best_model_train_fusion.pth`.
  7. Load checkpoint using the same logic as `test.py` `load_ckpt()`: handle both `'model_state_dict'` key and raw state_dict.
  8. Call `model.load_state_dict(state_dict)`.
  9. Move to device, set `model.eval()`.
  10. Return `(model, config_dict)`.
- **Critical:** The model construction must match `test.py` lines 70--87 exactly. No extra arguments (like `unfreeze_text_layers` or `unfreeze_image_layers`) because the checkpoint was saved with specific parameter structure.

#### 3f. `load_single_sample(csv_path, idx, tokenizer, image_processor, max_length, max_images, image_dir, device)`

- Signature: `load_single_sample(csv_path: str, idx: int, tokenizer, image_processor, max_length: int = 256, max_images: int = 4, image_dir: str = './data/image', device: torch.device = None) -> dict`
- Logic:
  1. Read CSV into DataFrame.
  2. Extract row at `idx`.
  3. Extract `comment_clean` as string.
  4. Tokenize: `tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')`.
  5. Parse `image_url` using `ast.literal_eval()` with try/except fallback to single-element list (exactly as `src/dataset.py` lines 57--59).
  6. Load images using MD5 hash lookup in `image_dir` (replicate `_load_image()` from `src/dataset.py` lines 28--40). For XAI, do NOT download from URL; use local cache only; if not found, use black placeholder image.
  7. Pad to `max_images` with black images `Image.new('RGB', (224, 224), color='black')`.
  8. Process images: `image_processor(images_list, return_tensors='pt')['pixel_values']` to get `[max_images, C, H, W]`.
  9. Compute `num_images = min(len(image_urls), max_images)`.
  10. Extract ground truth: `torch.tensor([row[col] for col in LABEL_COLS], dtype=torch.float)`.
  11. Add batch dimension to all tensors: `input_ids [1, seq_len]`, `attention_mask [1, seq_len]`, `pixel_values [1, max_images, C, H, W]`, `num_images [1]`.
  12. Move all tensors to device.
  13. Return dictionary:
      ```
      {
          'input_ids': tensor [1, seq_len],
          'attention_mask': tensor [1, seq_len],
          'pixel_values': tensor [1, max_images, C, H, W],
          'num_images': tensor [1],
          'factor_scores': tensor [5],
          'text': str,
          'image_urls': list[str],
          'loaded_images': list[PIL.Image],  # Original PIL images before transform
          'num_real_images': int,
          'sample_idx': int,
          'csv_path': str,
      }
      ```
- **Critical:** The preprocessing must produce byte-identical tensors to `MultimodalDataset.__getitem__()` from `src/dataset.py`. The tokenizer call, image processing call, and padding logic must match exactly.

#### 3g. `get_prediction(model, sample_dict)`

- Signature: `get_prediction(model: nn.Module, sample_dict: dict) -> dict`
- Logic:
  1. Extract `input_ids`, `attention_mask`, `pixel_values`, `num_images` from `sample_dict`.
  2. Run inference under `torch.no_grad()`:
     ```
     output = model(input_ids=..., attention_mask=..., pixel_values=..., num_images=...)
     preds = output[0] if isinstance(output, tuple) else output
     ```
     This matches `test.py` lines 114--116 exactly.
  3. Convert predictions to numpy: `preds.cpu().numpy().flatten()` to get shape `(5,)`.
  4. Build result dictionary:
     ```
     {
         'predictions': {
             'food_score': float,
             'price_score': float,
             'atmosphere_score': float,
             'service_score': float,
             'overall_satisfaction': float,
         },
         'predictions_array': np.ndarray shape (5,),
         'ground_truth': {
             'food_score': float,
             ...
         },
         'ground_truth_array': np.ndarray shape (5,),
         'absolute_errors': {
             'food_score': float,
             ...
         },
         'mean_mae': float,
     }
     ```
  5. Return the dictionary.

#### 3h. `save_figure(fig, save_path, dpi, close)`

- Signature: `save_figure(fig: matplotlib.figure.Figure, save_path: str, dpi: int = DEFAULT_DPI, close: bool = True) -> str`
- Logic:
  1. Create parent directory if it does not exist: `os.makedirs(os.path.dirname(save_path), exist_ok=True)`.
  2. Save figure: `fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')`.
  3. If `close`: `plt.close(fig)`.
  4. Print confirmation: `f'Saved figure: {save_path}'`.
  5. Return `save_path`.

#### 3i. `save_raw_values(data, save_path)`

- Signature: `save_raw_values(data: dict, save_path: str) -> str`
- Logic:
  1. Create parent directory if it does not exist.
  2. Determine format from extension:
     - `.json`: `json.dump(data, f, indent=2, default=str)`. Convert numpy arrays to lists using a custom encoder.
     - `.npy`: `np.save(save_path, data)`.
     - `.npz`: `np.savez(save_path, **data)`.
     - `.csv`: `pd.DataFrame(data).to_csv(save_path, index=False)`.
  3. Print confirmation.
  4. Return `save_path`.

### Step 4: Verify Model Loading

**Purpose:** Confirm that the checkpoint loads without errors and that the resulting model has the expected architecture.

**Verification steps:**
1. Call `load_model(exp_dir)`.
2. Assert model class is `CrossAttentionFusion`.
3. Assert `model.text_model.encoder.config.hidden_size == 768`.
4. Assert `model.image_model.encoder.num_features == 1024`.
5. Assert the prediction head final layer output size is 5: check `model.head[-1].out_features == 5`.
6. Count total parameters and compare to expected (should be consistent across loads).
7. Verify model is in eval mode: `assert not model.training`.

### Step 5: Verify Single-Sample Inference

**Purpose:** Confirm that `load_single_sample()` followed by `get_prediction()` produces a valid output.

**Verification steps:**
1. Load a sample from the validation or test set.
2. Verify tensor shapes:
   - `input_ids`: `[1, 256]`
   - `attention_mask`: `[1, 256]`
   - `pixel_values`: `[1, 4, 3, 224, 224]`
   - `num_images`: `[1]`
3. Run `get_prediction()`.
4. Verify output has 5 values.
5. Verify all predictions are within a reasonable range (e.g., 0 to 12 for scores on a 1--10 scale; allow some headroom since regression outputs are not clipped).

### Step 6: Verify Output Matches predictions.csv

**Purpose:** Numerically validate that the XAI infrastructure reproduces the same predictions as the training pipeline.

**Verification steps:**
1. Load `predictions.csv` (or `test_predictions.csv`) from the experiment directory.
2. Select the same sample by index.
3. Extract `y_pred_food`, `y_pred_price`, `y_pred_atmos`, `y_pred_service`, `y_pred_overall` from the CSV.
4. Run `get_prediction()` on that sample.
5. Compare each prediction value. Tolerance: absolute difference < 1e-4.
6. If tolerance is not met, investigate:
   - AMP (mixed precision) during training vs. full precision during inference.
   - Device differences (GPU training vs. CPU inference).
   - Image preprocessing differences.
7. Report the maximum absolute difference across all 5 targets and all verified samples.
8. Verify at least 3 samples to confirm the pattern holds.

**Why 1e-4 tolerance:** Floating-point differences arise from: (a) GPU vs. CPU arithmetic, (b) cudnn non-determinism during validation (though this should be minimal with deterministic=True), (c) potential AMP during validation. A tolerance of 1e-4 is strict enough to catch preprocessing bugs but lenient enough to accommodate legitimate floating-point variation.

### Step 7: Extract and Verify Intermediate Tensors

**Purpose:** Confirm that text features and image features can be extracted with expected shapes, which Phases 2--5 depend on.

**Verification steps:**

#### 7a. Text Features
1. Run `model.text_model(input_ids, attention_mask)` and capture the returned `(_, text_features)`.
2. Assert `text_features.shape == (1, 768)`.
3. Verify values are finite: `assert torch.isfinite(text_features).all()`.

#### 7b. Image Features
1. Run `model.image_model(pixel_values, num_images=num_images)` and capture `(_, image_features)`.
2. Assert `image_features.shape == (1, 1024)`.
3. Verify values are finite.

#### 7c. Combined Feature Extraction Consistency
1. Run the full model forward pass.
2. Separately extract text_features and image_features.
3. Manually replicate the fusion forward pass:
   - `t = model.text_proj(text_features.float()).unsqueeze(1)`
   - `i = model.image_proj(image_features.float()).unsqueeze(1)`
   - Run cross-attention manually.
   - Concatenate and pass through head.
4. Compare the manual result with the full forward pass result. They must match within 1e-6.

### Step 8: Verify Text Encoder Attention Extraction

**Purpose:** Confirm that PhoBERT supports `output_attentions=True`, which Phase 3 (Attention Visualization) requires.

**Verification steps:**
1. Access the text encoder directly: `model.text_model.encoder`.
2. Run forward pass with `output_attentions=True, return_dict=True`:
   ```
   text_outputs = model.text_model.encoder(
       input_ids=input_ids,
       attention_mask=attention_mask,
       output_attentions=True,
       return_dict=True,
   )
   ```
3. Assert `text_outputs.attentions` is not None.
4. Assert `len(text_outputs.attentions) == 12` (PhoBERT has 12 layers).
5. Assert each attention tensor has shape `[1, 12, L, L]` where L = max_length (256).
6. Assert attention values are non-negative (they are softmax outputs).
7. Assert attention rows sum to approximately 1.0 (within tolerance for floating-point).
8. Save a sample attention matrix shape and value range to the verification report.

### Step 9: Verify Image Encoder Spatial Feature Map Shape via Hook

**Purpose:** Confirm the Swin-B spatial feature map shape before global pooling, which Phase 2 (Grad-CAM) requires.

**Verification steps:**
1. Register a forward hook on the appropriate layer of `model.image_model.encoder`. For Swin-B in timm, the target is the last norm layer or the last stage that produces spatial output. The hook should capture the output tensor.
2. Candidate hook targets to investigate (in order of preference):
   - `model.image_model.encoder.norm` (the final LayerNorm before pooling)
   - `model.image_model.encoder.layers[-1]` or `model.image_model.encoder.stages[-1]`
   - Alternatively, use `model.image_model.encoder.forward_features()` which returns the feature map before pooling.
3. Run a forward pass with a single image `[1, 3, 224, 224]` through the image encoder.
4. Capture the spatial feature map from the hook.
5. Verify its shape. Expected: `[1, 1024, 7, 7]` for Swin-B with 224x224 input. However, Swin-B in timm may produce `[1, 7*7, 1024]` (sequence format) rather than `[1, 1024, 7, 7]` (spatial format). The verification must determine the exact shape and document whether reshaping is needed.
6. Record the exact layer name, output shape, and any required reshape in the verification report. This information is critical for Phase 2.
7. Remove the hook after verification.

### Step 10: Create Verification Notebook

**Purpose:** Provide a runnable, cell-by-cell notebook that validates all infrastructure components and serves as documentation.

**Location:** `xai/notebooks/Phase1_Infrastructure_Verification.ipynb`

**Design:** See Section 9 (Notebook Design) for cell-by-cell specification.

---

## 7. Required Code Files

| File | Location | Responsibility |
|---|---|---|
| `__init__.py` | `xai/__init__.py` | Package initialization. Exposes public API. |
| `config.py` | `xai/config.py` | All constants: target names, indices, display names, label columns, score range, seed, DPI, color schemes, best model config, dimension constants. |
| `utils.py` | `xai/utils.py` | Core utility functions: model loading, sample loading, prediction, artifact saving, processor/tokenizer creation, seed setting, device detection. Contains private `TimmProcessor` class. |
| Verification notebook | `xai/notebooks/Phase1_Infrastructure_Verification.ipynb` | End-to-end verification of all infrastructure. Generates verification report. |

---

## 8. Folder Structure

After Phase 1 completion, the project tree adds:

```
SE365/                              (project root)
|-- Models/                         (existing, unchanged)
|-- src/                            (existing, unchanged)
|-- main.py                         (existing, unchanged)
|-- test.py                         (existing, unchanged)
|-- Config.py                       (existing, unchanged)
|-- Trainer.py                      (existing, unchanged)
|-- notebook/                       (existing, unchanged)
|   |-- demo_single_sample_exp060A.ipynb
|
|-- xai/                            (NEW - Phase 1)
|   |-- __init__.py                 
|   |-- config.py                   
|   |-- utils.py                    
|   |-- notebooks/                  
|       |-- Phase1_Infrastructure_Verification.ipynb
|
|-- experiments/
    |-- EXP_060A_.../
        |-- best_model_train_fusion.pth   (existing)
        |-- config.yaml                    (existing)
        |-- predictions.csv                (existing)
        |-- xai/                           (NEW - Phase 1)
            |-- infrastructure/
                |-- verification_report.json
                |-- sample_prediction.json
```

Convention for subsequent phases (defined here, created later):
```
experiments/EXP_060A_.../xai/
|-- infrastructure/     (Phase 1)
|-- gradcam/            (Phase 2)
|-- attention/          (Phase 3)
|-- shap/               (Phase 4)
|-- lime/               (Phase 5)
|-- case_studies/        (Phase 6)
|-- reports/            (Phase 7)
|-- thesis_figures/     (Phase 8)
|-- raw_values/         (shared raw data)
```

---

## 9. Notebook Design

### Phase1_Infrastructure_Verification.ipynb

**Notebook environment:** Google Colab (with GPU runtime recommended, CPU fallback supported).

---

#### Cell 0: Title and Description (Markdown)

Content: Title, purpose statement, experiment ID, model configuration summary table (Image Backbone, Text Backbone, Fusion, Loss), link to Phase 1 specification. Note that this notebook verifies infrastructure only and does not perform XAI analysis.

---

#### Cell 1: Environment Setup

- Mount Google Drive (if Colab).
- Clone repository (if Colab).
- Install dependencies from `requirements.txt`.
- Extract data from Drive (same pattern as `demo_single_sample_exp060A.ipynb` cells 2--4).

---

#### Cell 2: Imports and Path Configuration

- Import standard libraries: `os`, `sys`, `json`, `warnings`.
- Set `PROJECT_ROOT`, `DRIVE_ROOT`, `EXP_ID`, `EXP_DIR`.
- Add `PROJECT_ROOT` to `sys.path`.
- Change working directory to `PROJECT_ROOT`.
- Import from `xai.config`: all constants.
- Import from `xai.utils`: all functions.
- Print all paths for verification.

---

#### Cell 3: Set Seed and Device (Verification V1)

- Call `set_seed(DEFAULT_SEED)`.
- Call `get_device()`.
- Print device.
- **Assertion:** Device is one of `cuda`, `mps`, `cpu`.
- Print: "V1: Seed and device -- PASSED".

---

#### Cell 4: Load Model (Verification V2)

- Call `model, config = load_model(EXP_DIR)`.
- **Assertion:** `isinstance(model, CrossAttentionFusion)`.
- **Assertion:** `model.text_model.encoder.config.hidden_size == 768`.
- **Assertion:** `model.image_model.encoder.num_features == 1024`.
- **Assertion:** `not model.training` (eval mode).
- Print model class name.
- Print total parameter count: `sum(p.numel() for p in model.parameters())`.
- Print checkpoint `best_mean_mae` from config/checkpoint.
- Print: "V2: Model loading -- PASSED".

---

#### Cell 5: Load Tokenizer and Image Processor (Verification V3)

- Call `tokenizer = get_tokenizer(config['text_model_name'])`.
- Call `image_processor = get_image_processor(config['image_model_name'])`.
- Verify tokenizer vocabulary size is > 0.
- Tokenize a test string `"Xin chao"` and verify `input_ids` is non-empty.
- Process a dummy black PIL image and verify `pixel_values` shape is `[1, 3, 224, 224]`.
- Print: "V3: Tokenizer and image processor -- PASSED".

---

#### Cell 6: Load Single Sample (Verification V4)

- Determine data split (test or val, same logic as demo notebook).
- Call `sample = load_single_sample(csv_path, idx=0, tokenizer=tokenizer, image_processor=image_processor, image_dir=IMAGE_DIR, device=device)`.
- **Assertion:** `sample['input_ids'].shape == (1, 256)`.
- **Assertion:** `sample['attention_mask'].shape == (1, 256)`.
- **Assertion:** `sample['pixel_values'].shape == (1, 4, 3, 224, 224)`.
- **Assertion:** `sample['num_images'].shape == (1,)`.
- **Assertion:** `sample['factor_scores'].shape == (5,)`.
- **Assertion:** `len(sample['text']) > 0`.
- Print sample text (truncated to 100 chars).
- Print number of real images.
- Print ground truth scores.
- Print: "V4: Single sample loading -- PASSED".

---

#### Cell 7: Run Inference (Verification V5)

- Call `result = get_prediction(model, sample)`.
- **Assertion:** `len(result['predictions']) == 5`.
- **Assertion:** all predictions are finite (not NaN, not Inf).
- Print predictions dictionary.
- Print ground truth dictionary.
- Print absolute errors dictionary.
- Print mean MAE.
- Print: "V5: Single-sample inference -- PASSED".

---

#### Cell 8: Numerical Match Against predictions.csv (Verification V6)

- Load predictions CSV from experiment directory.
- Select the same sample index.
- Extract stored predictions.
- Compare with `get_prediction()` output.
- Compute maximum absolute difference across all 5 targets.
- **Assertion:** max difference < 1e-4 (if same device) or < 1e-3 (if cross-device).
- Repeat for at least 2 additional samples (indices 1 and 2).
- Print per-sample, per-target differences in a table.
- Print: "V6: Numerical consistency -- PASSED (max_diff=X.XXXX)".

---

#### Cell 9: Extract Intermediate Tensors (Verification V7)

- Run `_, text_features = model.text_model(sample['input_ids'], sample['attention_mask'])`.
- **Assertion:** `text_features.shape == (1, 768)`.
- **Assertion:** `torch.isfinite(text_features).all()`.
- Run `_, image_features = model.image_model(sample['pixel_values'], num_images=sample['num_images'])`.
- **Assertion:** `image_features.shape == (1, 1024)`.
- **Assertion:** `torch.isfinite(image_features).all()`.
- Print feature statistics: mean, std, min, max for both.
- Print: "V7: Intermediate tensor extraction -- PASSED".

---

#### Cell 10: Verify PhoBERT Attention Extraction (Verification V8)

- Access `model.text_model.encoder`.
- Run forward pass with `output_attentions=True, return_dict=True`.
- **Assertion:** `text_outputs.attentions is not None`.
- **Assertion:** `len(text_outputs.attentions) == 12`.
- **Assertion:** `text_outputs.attentions[0].shape == (1, 12, 256, 256)`.
- **Assertion:** attention values are >= 0 (softmax output).
- Check row sum: `attn_sum = text_outputs.attentions[-1][0, 0].sum(dim=-1)`, verify close to 1.0.
- Print attention shape summary.
- Print: "V8: PhoBERT attention extraction -- PASSED".

---

#### Cell 11: Verify Swin-B Spatial Feature Map via Hook (Verification V9)

- Define a hook storage dict.
- Define hook function that captures the output.
- Register hook on candidate layers of `model.image_model.encoder`.
- Run forward pass with a single image: reshape `sample['pixel_values']` from `[1, 4, 3, 224, 224]` to process only the first image `[1, 3, 224, 224]`.
- Capture feature map from hook.
- Print captured shape.
- Determine if reshaping is needed (e.g., from `[1, 49, 1024]` to `[1, 1024, 7, 7]`).
- **Assertion:** Feature map has 1024 channels and spatial extent 7x7 (after possible reshape).
- Document the exact layer name and any reshape requirement.
- Remove hook.
- Print: "V9: Swin-B spatial feature map -- PASSED (shape=..., layer=...)".

---

#### Cell 12: Forward-Pass Consistency Check (Verification V10)

- Run full model forward pass to get predictions.
- Separately extract text_features and image_features.
- Manually compute the fusion forward pass step by step:
  1. `t = model.text_proj(text_features.float()).unsqueeze(1)`
  2. `i = model.image_proj(image_features.float()).unsqueeze(1)`
  3. `t_out, _ = model.cross_attn_t2i(query=t, key=i, value=i)`
  4. `i_out, _ = model.cross_attn_i2t(query=i, key=t, value=t)`
  5. `fused = torch.cat([t_out.squeeze(1), i_out.squeeze(1)], dim=1)`
  6. `manual_preds = model.head(fused)`
- Compare `manual_preds` with the full forward pass result.
- **Assertion:** max absolute difference < 1e-6.
- Print: "V10: Forward-pass consistency -- PASSED".

---

#### Cell 13: Save Verification Report

- Compile all verification results into a dictionary.
- Save to `experiments/EXP_060A_.../xai/infrastructure/verification_report.json`.
- Save sample prediction to `experiments/EXP_060A_.../xai/infrastructure/sample_prediction.json`.
- Print final summary:
  ```
  ===================================
  PHASE 1 INFRASTRUCTURE VERIFICATION
  ===================================
  V1  Seed and device           : PASSED
  V2  Model loading             : PASSED
  V3  Tokenizer/processor       : PASSED
  V4  Single sample loading     : PASSED
  V5  Single-sample inference   : PASSED
  V6  Numerical consistency     : PASSED (max_diff=...)
  V7  Intermediate tensors      : PASSED
  V8  PhoBERT attention         : PASSED
  V9  Swin-B feature map        : PASSED (shape=..., layer=...)
  V10 Forward-pass consistency  : PASSED
  ===================================
  All verifications passed. Phase 1 infrastructure is ready.
  ===================================
  ```

---

#### Cell 14: Summary and Next Steps (Markdown)

- Summarize what was verified.
- List the documented findings (feature map shape, attention shape, layer names).
- State that Phases 2--8 can now import from `xai.utils` and `xai.config`.
- List which specific findings feed into which phase.

---

## 10. Algorithm

### Pseudo-Workflow for Infrastructure Setup

```
ALGORITHM: Phase 1 -- XAI Infrastructure Setup and Verification

INPUT:
  - Project codebase (Models/, src/, main.py, test.py, Config.py, Trainer.py)
  - Experiment directory with checkpoint, config, predictions
  - Data directory with CSVs and image cache

OUTPUT:
  - xai/ Python package (config.py, utils.py, __init__.py)
  - Verification notebook with all checks passing
  - verification_report.json

PROCEDURE:

1. CREATE_PACKAGE:
   1.1  Create xai/ directory at project root
   1.2  Create xai/__init__.py with public API exports
   1.3  Create xai/config.py with all constants
   1.4  Create xai/utils.py with all utility functions
   1.5  Create xai/notebooks/ directory

2. IMPLEMENT_UTILITIES:
   2.1  Implement get_device() -- detect CUDA > MPS > CPU
   2.2  Implement set_seed(seed) -- replicate main.py exactly
   2.3  Implement get_tokenizer(name) -- AutoTokenizer wrapper
   2.4  Implement get_image_processor(name) -- with TimmProcessor fallback
   2.5  Implement load_model(exp_dir, ...) -- config loading + model build + checkpoint load
   2.6  Implement load_single_sample(csv, idx, ...) -- replicate dataset.py preprocessing
   2.7  Implement get_prediction(model, sample) -- inference + result formatting
   2.8  Implement save_figure(fig, path, dpi) -- matplotlib saving utility
   2.9  Implement save_raw_values(data, path) -- JSON/numpy saving utility

3. VERIFY_MODEL_LOADING:
   3.1  Call load_model(EXP_DIR)
   3.2  ASSERT model class is CrossAttentionFusion
   3.3  ASSERT text encoder hidden_size == 768
   3.4  ASSERT image encoder num_features == 1024
   3.5  ASSERT model is in eval mode

4. VERIFY_SINGLE_SAMPLE:
   4.1  Call load_single_sample(csv, idx=0, ...)
   4.2  ASSERT input_ids shape == [1, 256]
   4.3  ASSERT pixel_values shape == [1, 4, 3, 224, 224]
   4.4  Call get_prediction(model, sample)
   4.5  ASSERT output has 5 finite values

5. VERIFY_NUMERICAL_CONSISTENCY:
   5.1  Load predictions.csv from experiment directory
   5.2  FOR sample_idx IN [0, 1, 2]:
   5.2.1    Load sample at sample_idx
   5.2.2    Run get_prediction()
   5.2.3    Extract stored prediction from CSV
   5.2.4    COMPUTE absolute difference per target
   5.2.5    ASSERT max_diff < TOLERANCE (1e-4 same device, 1e-3 cross-device)
   5.3  RECORD max_diff across all samples and targets

6. VERIFY_INTERMEDIATE_TENSORS:
   6.1  Run model.text_model() -> assert text_features shape [1, 768]
   6.2  Run model.image_model() -> assert image_features shape [1, 1024]
   6.3  ASSERT all values are finite

7. VERIFY_ATTENTION_EXTRACTION:
   7.1  Run model.text_model.encoder(output_attentions=True)
   7.2  ASSERT attentions tuple has 12 elements
   7.3  ASSERT each element shape [1, 12, L, L]
   7.4  ASSERT values >= 0 and rows sum to ~1.0

8. VERIFY_SPATIAL_FEATURE_MAP:
   8.1  Register forward hook on Swin-B encoder
   8.2  Run single-image forward pass
   8.3  Capture feature map output
   8.4  DETERMINE exact shape (expected ~[1, 1024, 7, 7] or ~[1, 49, 1024])
   8.5  DOCUMENT layer name, shape, reshape requirement
   8.6  Remove hook

9. VERIFY_FORWARD_CONSISTENCY:
   9.1  Run full model forward pass -> preds_full
   9.2  Extract text_features, image_features separately
   9.3  Manually replicate fusion: project, cross-attend, concat, head
   9.4  ASSERT manual_preds matches preds_full within 1e-6

10. SAVE_REPORT:
    10.1 Compile all verification results
    10.2 Save verification_report.json
    10.3 Save sample_prediction.json
    10.4 Print final summary table

END PROCEDURE
```

---

## 11. Validation

### Sanity Checks

| Check | Method | Expected Result |
|---|---|---|
| Model loads without error | `load_model()` returns without exception | Model object + config dict |
| Model is correct class | `isinstance(model, CrossAttentionFusion)` | True |
| Model is in eval mode | `model.training` | False |
| Tokenizer works | Tokenize a known Vietnamese string | Non-empty input_ids |
| Image processor works | Process a 224x224 PIL image | Tensor of shape [1, 3, 224, 224] |
| Prediction has 5 outputs | `len(predictions) == 5` | True |
| Predictions are finite | `np.isfinite(preds).all()` | True |
| Predictions are in reasonable range | All values between -2 and 15 | True (no extreme outliers) |

### Quantitative Validation

| Check | Method | Acceptance Criterion |
|---|---|---|
| Prediction matches CSV | Compare with predictions.csv | Max absolute difference < 1e-4 (same device) |
| Cross-device consistency | Compare GPU vs CPU inference | Max absolute difference < 1e-3 |
| Forward-pass decomposition | Manual fusion vs. full forward | Max absolute difference < 1e-6 |
| Multi-sample consistency | Verify 3+ samples | All samples pass tolerance |

### Reproducibility Checks

| Check | Method | Expected Result |
|---|---|---|
| Deterministic output | Run same sample twice | Identical predictions (bit-for-bit on same device) |
| Seed independence | Different seeds produce same eval-mode output (no dropout) | Identical predictions |
| Device annotation | Record which device was used | Stored in verification_report.json |

### Shape Verification

| Tensor | Expected Shape | Source |
|---|---|---|
| input_ids | [1, 256] | Tokenizer with max_length=256 |
| attention_mask | [1, 256] | Tokenizer with max_length=256 |
| pixel_values | [1, 4, 3, 224, 224] | Image processor, 4 images padded |
| num_images | [1] | Scalar tensor |
| text_features | [1, 768] | TextModel encoder |
| image_features | [1, 1024] | ImageModel encoder |
| attention per layer | [1, 12, 256, 256] | PhoBERT with output_attentions |
| spatial feature map | [1, 1024, 7, 7] or equivalent | Swin-B last stage (verify via hook) |
| predictions | [1, 5] | CrossAttentionFusion output |

---

## 12. Risks

### R1: Checkpoint Loading Mismatch

**Problem:** The saved `state_dict` keys may not match the model architecture built at inference time, causing `load_state_dict()` to fail with missing or unexpected keys.

**Why it happens:** The checkpoint was saved during training where the model may have been constructed with specific `unfreeze_text_layers` or `unfreeze_image_layers` arguments. However, these arguments only affect `requires_grad` flags, not the architecture itself. A more dangerous scenario is if the model was wrapped in `DataParallel` or `DistributedDataParallel` during training, which prepends `module.` to all keys.

**Possible strategies:**

| Strategy | Description |
|---|---|
| S1: Exact reconstruction | Build model with identical arguments as training, load state_dict directly |
| S2: Flexible loading | Strip `module.` prefix if present, use `strict=False` |
| S3: Config-driven | Read args from checkpoint or config file to reconstruct identically |

**Advantages and disadvantages:**

- S1: Simplest and most reliable. Disadvantage: requires knowing exact construction args.
- S2: Handles DataParallel wrapping. Disadvantage: `strict=False` silently ignores missing keys, hiding real bugs.
- S3: Most robust. Disadvantage: slightly more complex implementation.

**Engineering trade-offs:** S3 requires parsing config files, but the config is already saved by Trainer.py. S2 is dangerous because `strict=False` could mask genuine architecture mismatches.

**Research trade-offs:** Any mismatch in loaded weights invalidates all subsequent XAI analysis. Strictness is critical.

**Recommended implementation:** Use S3 (config-driven reconstruction) with S1 as the loading call. Read `text_model_name`, `image_model_name`, and `fusion_type` from the experiment config. Build the model using the same constructor pattern as `test.py`. Use `strict=True` for `load_state_dict()`. If keys have `module.` prefix, strip it before loading but log a warning.

**Final decision:** Config-driven reconstruction with strict loading and `module.` prefix stripping as a documented fallback.

**Reason:** This matches the existing `test.py` pattern, ensures architecture fidelity, and will catch genuine mismatches while handling the one known prefix issue.

---

### R2: Image Processor Inconsistency Between Training and XAI

**Problem:** If the XAI pipeline uses a different image processor than was used during training, the pixel values fed to the model will differ, causing incorrect predictions and invalidating all explanations.

**Why it happens:** The codebase has a fallback chain: try `AutoImageProcessor.from_pretrained()`, then check for SigLIP, then fall back to `TimmProcessor`. For Swin-B (`swin_base_patch4_window7_224`), `AutoImageProcessor.from_pretrained()` will fail because Swin-B in timm is not a HuggingFace model, so the `TimmProcessor` fallback will activate. If the XAI code does not replicate this fallback exactly, it will use a different transform (different normalization, resize, crop).

**Possible strategies:**

| Strategy | Description |
|---|---|
| S1: Copy TimmProcessor class | Replicate the exact TimmProcessor class in xai/utils.py |
| S2: Import from main.py | Import TimmProcessor from main.py |
| S3: Replicate the full fallback chain | Implement the same try/except logic as main.py |

**Advantages and disadvantages:**

- S1: Self-contained, no external dependency. Disadvantage: code duplication.
- S2: No duplication. Disadvantage: main.py is not structured as a library; importing from it triggers argparse which fails without CLI args.
- S3: Most faithful to training. Disadvantage: requires both TimmProcessor class and the fallback logic.

**Engineering trade-offs:** S2 fails because `main.py` calls `get_args()` at module level (it does not, but the TimmProcessor is a local class). Actually, TimmProcessor is defined at module level in main.py before the `main()` function, so importing it would work: `from main import TimmProcessor`. However, this creates a fragile dependency and `main.py` also imports `Trainer` and other modules.

**Research trade-offs:** Any difference in image preprocessing directly corrupts all XAI outputs. This is a zero-tolerance risk.

**Recommended implementation:** Use S3: replicate the full fallback chain in `xai/utils.py`, including a private `_TimmProcessor` class that is a direct copy of the `TimmProcessor` from `main.py`. Add a comment linking to the source. The `get_image_processor()` function implements the same try/except chain.

**Final decision:** Replicate the full fallback chain with a private `_TimmProcessor` class in `xai/utils.py`.

**Reason:** Self-containment avoids fragile cross-module imports, and the exact replication ensures training-time consistency. The numerical verification in Step 6 will catch any discrepancy.

---

### R3: Swin-B Spatial Feature Map Shape Uncertainty

**Problem:** Grad-CAM (Phase 2) requires spatial feature maps of shape `[B, C, H, W]`. Swin-B in timm may produce feature maps in different formats depending on the specific model variant and timm version: `[B, C, H, W]`, `[B, H*W, C]` (sequence format), or `[B, H, W, C]` (channel-last).

**Why it happens:** Swin Transformer operates on patches and windows internally. The output format of intermediate stages depends on whether the model uses `channels_last` memory format, whether `output_fmt` is set, and the specific timm version. The `forward_features()` method may reshape outputs differently than raw stage outputs.

**Possible strategies:**

| Strategy | Description |
|---|---|
| S1: Hook on the norm layer | Register hook on `model.image_model.encoder.norm`, which is applied before pooling. Captures the feature tensor in whatever format it is in. |
| S2: Hook on the last stage | Register hook on the last stage/layer block. |
| S3: Use `forward_features()` | Call `model.image_model.encoder.forward_features(x)` which returns features before pooling. |
| S4: Inspect and adapt | Run a probe forward pass, capture the output, detect the format, and reshape accordingly. |

**Advantages and disadvantages:**

- S1: Clean attachment point, but the norm layer may output `[B, H*W, C]` or `[B, H, W, C]`.
- S2: Stage outputs may be in window format, harder to interpret.
- S3: Explicit API, but the return shape still needs verification. Also, this bypasses the hook mechanism that pytorch-grad-cam uses.
- S4: Adaptive, handles any format. Slightly more complex but robust.

**Engineering trade-offs:** Phase 2 (Grad-CAM) will use `pytorch-grad-cam`, which expects `[B, C, H, W]` or can handle other formats with a reshape transform. The critical thing is to know the exact shape so Phase 2 can configure the reshape.

**Research trade-offs:** An incorrect feature map shape leads to meaningless Grad-CAM heatmaps. This must be resolved definitively in Phase 1.

**Recommended implementation:** Use S4 (inspect and adapt). In the verification notebook, register a hook on multiple candidate layers (`encoder.norm`, `encoder.layers[-1].blocks[-1]`, etc.), run a forward pass, capture all outputs, and document:
1. The exact layer that produces the spatial feature map.
2. The exact shape of that output.
3. Whether reshaping from `[B, H*W, C]` to `[B, C, H, W]` is needed.

Record these findings in `verification_report.json` with keys `gradcam_target_layer`, `feature_map_shape`, `reshape_required`, `reshape_from`, `reshape_to`.

**Final decision:** Probe-based shape inspection with results documented in the verification report. Phase 2 will read this report to configure Grad-CAM correctly.

**Reason:** This is the only approach that provides ground truth about the actual tensor shapes, rather than relying on assumptions about timm internals that may change across versions.

---

### R4: PhoBERT Attention Extraction Compatibility

**Problem:** PhoBERT (`vinai/phobert-base-v2`) must support `output_attentions=True` in its forward pass. If it does not, Phase 3 (Attention Visualization) cannot proceed.

**Why it happens:** PhoBERT is based on RoBERTa, which is implemented in HuggingFace transformers and supports `output_attentions=True`. However, the model is accessed through `TextModel.encoder` which is an `AutoModel`, and the `TextModel.forward()` method does not pass `output_attentions` to the encoder. For attention extraction, the encoder must be called directly.

**Possible strategies:**

| Strategy | Description |
|---|---|
| S1: Call encoder directly | Bypass TextModel.forward() and call model.text_model.encoder() with output_attentions=True |
| S2: Modify TextModel.forward() | Add output_attentions parameter to TextModel |
| S3: Register hooks | Use hooks on attention layers to capture weights |

**Advantages and disadvantages:**

- S1: No code modification to existing models. Works because the encoder is a standard HuggingFace AutoModel. Disadvantage: bypasses the TextModel wrapper, so must ensure the same inputs are passed.
- S2: Modifies the training codebase. Disadvantage: risks breaking existing training/test code; changes must be backward-compatible.
- S3: Framework-independent. Disadvantage: more complex, fragile if internal layer names change.

**Engineering trade-offs:** S1 is cleanest because it uses the public HuggingFace API without modifying the existing codebase. The encoder is a standard RoBERTa model that definitely supports `output_attentions`.

**Research trade-offs:** All three strategies yield the same attention tensors. S1 is the least invasive.

**Recommended implementation:** Use S1. Access `model.text_model.encoder` and call it directly with `output_attentions=True, return_dict=True`. Verify in Phase 1 that this works and document the exact call signature.

**Final decision:** Direct encoder call with `output_attentions=True`.

**Reason:** No modification to existing code. PhoBERT/RoBERTa via HuggingFace transformers has native support for this parameter. Phase 1 will verify it works and Phase 3 will use the verified pattern.

---

### R5: Multi-Image Handling During Single-Sample XAI

**Problem:** Each review can have up to 4 images. The ImageModel processes all images via `[B*N, C, H, W]`, applies the encoder, reshapes to `[B, N, features]`, and uses masked average pooling. For XAI methods like Grad-CAM, which operate on individual images, it is unclear whether to explain the first image, all images, or the highest-contribution image.

**Why it happens:** The architecture was designed for multi-image aggregation during training. XAI methods like Grad-CAM need a single spatial feature map. The masked average pooling merges all images into one vector before fusion, so the per-image gradient signal is diluted.

**Possible strategies:**

| Strategy | Description |
|---|---|
| S1: First image only | Always explain the first image in the review |
| S2: All images | Generate Grad-CAM for every image, display as a grid |
| S3: Highest-contribution image | Use gradient magnitude or feature norm to select the most influential image |
| S4: All real images (skip padding) | Explain all non-padded images, skip black placeholders |

**Advantages and disadvantages:**

- S1: Simplest implementation. Disadvantage: misses important evidence from other images; biased by image ordering.
- S2: Most thorough. Disadvantage: more computation; padding images produce noise.
- S3: Most informative. Disadvantage: requires an additional selection step; the selection criterion itself needs validation.
- S4: Balanced. Disadvantage: variable number of outputs per sample; still does not rank importance.

**Engineering trade-offs:** S4 is the most practical for the thesis because it explains all real evidence without wasting computation on padding. The implementation in `load_single_sample()` already tracks `num_real_images`, so filtering is straightforward.

**Research trade-offs:** For the thesis, showing all real images with individual Grad-CAM maps is more scientifically complete than showing only one. However, the primary analysis should focus on image-level contribution ranking.

**Recommended implementation:** Use S4 (all real images, skip padding) as the default in Phase 1 infrastructure. The `load_single_sample()` function already returns `num_real_images` and `loaded_images` (original PIL images). Phase 2 (Grad-CAM) will iterate over images `0:num_real_images` and generate one heatmap per real image per target.

**Final decision:** Explain all real images (1 to `num_real_images`), skip padding. Phase 1 ensures the infrastructure returns the necessary metadata. Phase 2 implements the per-image loop.

**Reason:** This approach is scientifically complete, implementable with the existing data structure, and avoids arbitrary selection bias. It naturally handles single-image reviews (most common) and multi-image reviews.

---

### R6: Device Compatibility (Colab GPU vs. CPU Fallback)

**Problem:** The XAI pipeline must work on Colab with GPU, Colab without GPU (CPU-only), and potentially local machines. Numerical outputs may differ slightly between devices due to floating-point arithmetic differences.

**Why it happens:** CUDA and CPU use different underlying math libraries. Even with `torch.backends.cudnn.deterministic = True`, GPU and CPU will produce slightly different results for the same computation. Colab free tier may not always provide GPU access.

**Possible strategies:**

| Strategy | Description |
|---|---|
| S1: GPU-only | Require GPU, fail if not available |
| S2: Auto-detect with tolerance | Auto-detect device, accept wider tolerance for cross-device comparisons |
| S3: Force CPU | Always use CPU for reproducibility |

**Advantages and disadvantages:**

- S1: Simplest, fastest. Disadvantage: fails when GPU is unavailable (Colab free tier).
- S2: Most flexible. Disadvantage: must define and document acceptable tolerance.
- S3: Most reproducible. Disadvantage: much slower, especially for SHAP/LIME which require many forward passes.

**Engineering trade-offs:** S2 is the only practical option. The verification report should record which device was used, and the numerical tolerance should be:
- Same-device comparison (e.g., GPU training + GPU XAI): < 1e-4
- Cross-device comparison (e.g., GPU training + CPU XAI): < 1e-3

**Research trade-offs:** Minor numerical differences between GPU and CPU do not affect XAI interpretation quality. A Grad-CAM heatmap that shifts by 1e-4 in prediction value produces visually identical overlays.

**Recommended implementation:** Use S2. Auto-detect device via `get_device()`. Record device in all output metadata. Use device-aware tolerance for numerical verification. Print a warning if running on CPU when GPU data was generated.

**Final decision:** Auto-detect with device-aware tolerance. Document device in all artifacts.

**Reason:** Maximizes accessibility (works on any Colab tier) while maintaining scientific rigor through recorded metadata and appropriate tolerance.

---

### R7: Numerical Precision Differences Between Training and Inference

**Problem:** Training may have used AMP (Automatic Mixed Precision, `--use_amp` flag) which performs some computations in float16. Inference in the XAI pipeline runs in float32. This can cause small but measurable prediction differences.

**Why it happens:** AMP uses float16 for forward pass computations to save memory and increase speed. The saved predictions in `predictions.csv` were generated during validation, which also used AMP if enabled. The XAI pipeline runs with `torch.no_grad()` in float32, producing slightly different results.

**Possible strategies:**

| Strategy | Description |
|---|---|
| S1: Match AMP behavior | Wrap XAI inference in `torch.cuda.amp.autocast()` to match training |
| S2: Accept float32 differences | Run in float32 and accept small differences |
| S3: Re-generate predictions | Run the model in float32 to produce new reference predictions |

**Advantages and disadvantages:**

- S1: Most accurate match. Disadvantage: AMP behavior may vary with GPU hardware; adds complexity.
- S2: Simplest. Disadvantage: differences may exceed 1e-4 for some samples.
- S3: Eliminates the discrepancy entirely. Disadvantage: requires running full evaluation again.

**Engineering trade-offs:** S2 is practical if the tolerance is set correctly. The key insight is that `test.py` line 114 uses `torch.cuda.amp.autocast(enabled=use_amp)` during test inference, so `test_predictions.csv` was generated with AMP if `use_amp=True`. The XAI pipeline should check whether AMP was used (from the config) and match that behavior.

**Research trade-offs:** The XAI analysis cares about the model's behavior, not the exact training-time prediction. As long as the model weights are identical and the preprocessing is identical, small float32 vs float16 differences in predictions are irrelevant to explanation quality.

**Recommended implementation:** Use S2 with adaptive tolerance. Check the config for `use_amp`. If AMP was used during training, widen the acceptable tolerance to 1e-3. If AMP was not used, keep tolerance at 1e-4. Optionally, add the ability to run with autocast for exact matching, but default to float32 for simplicity and stability.

**Final decision:** Float32 inference with adaptive tolerance based on training AMP setting. Record AMP status in verification report.

**Reason:** Float32 is more numerically stable for XAI methods (especially gradient-based ones like Grad-CAM) and avoids AMP-related edge cases. The small prediction difference does not affect explanation quality.

---

## 13. Best Practices

### Logging

- All utility functions should print their operations with prefixed timestamps: `[XAI] Loading model from ...`, `[XAI] Loaded sample idx=42 with 3 images`.
- Use Python's `logging` module at INFO level for operations and WARNING level for fallbacks (e.g., "Using TimmProcessor fallback for image processing").
- The verification notebook should print a clear PASSED/FAILED status for each check.

### Deterministic Execution

- Always call `set_seed()` before any inference.
- Always set `model.eval()` before any inference (disables dropout).
- Use `torch.no_grad()` for all inference operations.
- Record the seed value in all output metadata.

### Artifact Naming Convention

- Figures: `{method}_{target}_{descriptor}.png` (e.g., `gradcam_food_overlay.png`)
- Raw values: `{method}_{target}_{descriptor}.json` or `.npy`
- Reports: `{phase}_{descriptor}.json`
- All artifacts include the experiment ID in their parent directory path, not in the filename.

### Checkpoint Handling

- Never modify the checkpoint file.
- Load with `map_location=device` to handle cross-device loading.
- After loading, verify the model by running at least one inference pass.
- Record the checkpoint `best_mean_mae` in all output metadata for traceability.

### Memory Optimization

- For single-sample inference, use batch size 1. No DataLoader overhead.
- Release intermediate tensors when not needed: `del tensor; torch.cuda.empty_cache()`.
- When extracting features for SHAP background sets (Phase 4), process in batches and save to disk rather than holding all in memory.

### Configuration Management

- All phase-specific parameters should be defined at the top of each notebook as named constants.
- Cross-phase constants live in `xai/config.py` and are imported, never redefined.
- Experiment-specific paths are derived from `EXP_DIR`, never hardcoded.

### Figure Consistency

- All figures use the color schemes defined in `xai/config.py`.
- All figures are saved at `DEFAULT_DPI` (150) for drafts and `THESIS_DPI` (300) for final thesis.
- All figures use `bbox_inches='tight'` and `facecolor='white'`.
- All figures include descriptive titles and axis labels.
- Font sizes: title=14, axis labels=12, tick labels=10.

### Code Quality

- Type hints on all public functions.
- Docstrings with Args, Returns, and Raises sections.
- No global state except constants in `config.py`.
- All file paths accept both forward slashes and backslashes (use `os.path.join()`).

---

## 14. Deliverables

### Python Package

| Deliverable | Path | Type |
|---|---|---|
| Package init | `xai/__init__.py` | Python module |
| Constants | `xai/config.py` | Python module |
| Utilities | `xai/utils.py` | Python module |

### Notebook

| Deliverable | Path | Type |
|---|---|---|
| Verification notebook | `xai/notebooks/Phase1_Infrastructure_Verification.ipynb` | Jupyter notebook |

### Verification Artifacts

| Deliverable | Path | Type |
|---|---|---|
| Verification report | `experiments/EXP_060A_.../xai/infrastructure/verification_report.json` | JSON |
| Sample prediction | `experiments/EXP_060A_.../xai/infrastructure/sample_prediction.json` | JSON |

### Documented Findings

The verification report JSON must contain at minimum:

```
{
  "phase": "Phase 1: Infrastructure",
  "experiment_id": "EXP_060A_bestsequential_full_configuration",
  "timestamp": "...",
  "device": "cuda" or "cpu",
  "seed": 42,
  "model_class": "CrossAttentionFusion",
  "text_model": "vinai/phobert-base-v2",
  "image_model": "swin_base_patch4_window7_224",
  "checkpoint_best_mean_mae": 1.1079,
  "total_parameters": ...,
  "verifications": {
    "v1_seed_device": true,
    "v2_model_loading": true,
    "v3_tokenizer_processor": true,
    "v4_single_sample": true,
    "v5_inference": true,
    "v6_numerical_consistency": {
      "passed": true,
      "max_diff": ...,
      "tolerance": ...,
      "num_samples_verified": 3
    },
    "v7_intermediate_tensors": {
      "text_features_shape": [1, 768],
      "image_features_shape": [1, 1024]
    },
    "v8_attention_extraction": {
      "passed": true,
      "num_layers": 12,
      "num_heads": 12,
      "attention_shape_per_layer": [1, 12, 256, 256]
    },
    "v9_spatial_feature_map": {
      "passed": true,
      "target_layer_name": "...",
      "raw_shape": [...],
      "reshape_required": true/false,
      "target_shape": [1, 1024, 7, 7]
    },
    "v10_forward_consistency": {
      "passed": true,
      "max_diff": ...
    }
  }
}
```

---

## 15. Thesis Usage

### How Phase 1 Outputs Support the Thesis

#### Chapter: Methodology

Phase 1 infrastructure enables the thesis methodology section to state:

> "All explainability analyses were conducted using a verified inference pipeline that reproduces the training-time predictions within a tolerance of [X]. The XAI infrastructure ensures deterministic execution through fixed random seeds, eval-mode inference, and consistent preprocessing. The pipeline was validated by comparing predictions from the XAI utility functions against the stored predictions from the best training run (EXP_060A), confirming numerical consistency across [N] samples."

#### Chapter: Implementation

The `xai/` package structure demonstrates software engineering quality:

> "The explainability module was implemented as a self-contained Python package (`xai/`) with centralized configuration (`xai/config.py`), reusable utility functions (`xai/utils.py`), and method-specific modules for each XAI technique. This modular design ensured consistency across all explanation methods and facilitated reproducible analysis."

#### Chapter: Results

The verification report provides evidence of pipeline reliability:

> "Prior to XAI analysis, the inference pipeline was validated through ten verification checks including model loading, tensor shape verification, numerical consistency against stored predictions, attention extraction compatibility, and spatial feature map shape confirmation. All checks passed, establishing the trustworthiness of subsequent explanations."

#### Chapter: Appendix

The verification notebook serves as reproducibility documentation that thesis reviewers can inspect.

#### Defense Presentation

The verification summary table (V1--V10) can be shown as a slide to demonstrate methodological rigor. Key numbers to present:
- Total parameters loaded.
- Numerical tolerance achieved.
- Feature map shapes verified.
- Attention layers confirmed.

#### Journal Paper

If extended to a publication, the infrastructure verification section demonstrates the reproducibility standard expected in top venues.

---

## 16. Phase Completion Checklist

Every item below is measurable and verifiable. Phase 1 is complete when all items are checked.

### Code Deliverables

- [ ] `xai/__init__.py` exists and imports pass without error.
- [ ] `xai/config.py` exists and all constants are defined (TARGET_NAMES, TARGET_INDICES, FACTOR_NAMES, DISPLAY_NAMES, LABEL_COLS, SCORE_RANGE, DEFAULT_SEED, DEFAULT_DPI, THESIS_DPI, COLOR_SCHEMES, dimension constants).
- [ ] `xai/utils.py` exists and contains all specified functions: `get_device()`, `set_seed()`, `get_tokenizer()`, `get_image_processor()`, `load_model()`, `load_single_sample()`, `get_prediction()`, `save_figure()`, `save_raw_values()`.
- [ ] `from xai.config import TARGET_NAMES` works from the project root.
- [ ] `from xai.utils import load_model` works from the project root.

### Verification Checks (all must pass)

- [ ] V1: `set_seed()` and `get_device()` execute without error.
- [ ] V2: `load_model()` returns a `CrossAttentionFusion` model in eval mode with correct dimensions (text_dim=768, image_dim=1024).
- [ ] V3: `get_tokenizer()` and `get_image_processor()` produce valid tokenizer and processor.
- [ ] V4: `load_single_sample()` returns tensors with correct shapes: input_ids [1, 256], attention_mask [1, 256], pixel_values [1, 4, 3, 224, 224], num_images [1], factor_scores [5].
- [ ] V5: `get_prediction()` returns 5 finite prediction values.
- [ ] V6: Predictions match `predictions.csv` within tolerance (< 1e-4 same device, < 1e-3 cross-device) for at least 3 samples.
- [ ] V7: Text features shape is [1, 768] and image features shape is [1, 1024], all finite.
- [ ] V8: PhoBERT attention extraction works: 12 layers, each [1, 12, L, L], values >= 0, rows sum to ~1.0.
- [ ] V9: Swin-B spatial feature map shape is documented (expected: 1024 channels, 7x7 spatial or equivalent). The exact layer name and any required reshape are recorded.
- [ ] V10: Manual forward-pass decomposition matches full forward pass within 1e-6.

### Artifacts

- [ ] `xai/notebooks/Phase1_Infrastructure_Verification.ipynb` exists and runs end-to-end without error.
- [ ] `experiments/EXP_060A_.../xai/infrastructure/verification_report.json` is generated with all verification results.
- [ ] `experiments/EXP_060A_.../xai/infrastructure/sample_prediction.json` is generated.

### Documentation

- [ ] The verification report records the device used, the seed, the model class, and the checkpoint MAE.
- [ ] The Swin-B feature map findings (layer name, shape, reshape requirement) are documented in the verification report and will be consumed by Phase 2.
- [ ] The PhoBERT attention extraction findings (num_layers, num_heads, shape) are documented and will be consumed by Phase 3.

### Integration Readiness

- [ ] Phase 2 (Grad-CAM) can import `from xai.utils import load_model, load_single_sample, get_prediction, save_figure` and use them directly.
- [ ] Phase 3 (Attention) can access PhoBERT attention matrices via the documented pattern.
- [ ] Phase 4 (SHAP) can extract text_features and image_features via the documented pattern.
- [ ] Phase 5 (LIME) can wrap `get_prediction()` to build LIME predict functions.
- [ ] All subsequent phases can import constants from `xai.config`.
