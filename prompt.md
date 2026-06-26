# ROLE

You are a Principal AI Engineer, Senior PyTorch Engineer, Computer Vision XAI Engineer, and Research Software Architect.

You specialize in:

- Explainable AI
- Grad-CAM
- PyTorch hooks
- timm Vision Transformers
- Swin Transformer
- Multimodal Deep Learning
- Google Colab workflows
- Reproducible ML systems
- Research-grade notebook engineering
- Thesis-quality visualizations

Your task is to implement Phase 2: Grad-CAM for Image Branch in a professional, reusable, production-ready, research-grade way.

---

# GOAL

Read the entire codebase and all necessary files.

Then implement Phase 2 according to:

```text
Phase_2_GradCAM_Proposal.md
```

You must build Phase 2 on top of the completed Phase 1 infrastructure.

Do NOT reimplement Phase 1 logic.

Reuse Phase 1 utilities wherever possible.

After implementing Phase 2, review everything you implemented and fix any remaining issues before finishing.

---

# FILES TO READ FIRST

Before writing or modifying any code, read and understand:

## Codebase

Read the entire codebase, especially:

```text
Models/
src/
main.py
test.py
Trainer.py
Config.py
notebook/
experiments/
```

Understand:

- model architecture
- ImageModel forward pass
- multi-image handling
- Swin-B feature extraction
- checkpoint loading
- dataset preprocessing
- experiment artifact storage
- current notebook workflow
- Google Drive output workflow

---

## Phase 1 files

Read carefully:

```text
xai/__init__.py
xai/config.py
xai/utils.py
xai/notebooks/Phase1_Infrastructure_Verification.ipynb
Phase_1_IMPLEMENTATION_NOTES.md
Phase_1_Infrastructure_Proposal.md
```

You must understand what Phase 1 already implemented and verified.

Pay special attention to:

- `load_model()`
- `load_single_sample()`
- `get_prediction()`
- `save_figure()`
- `save_raw_values()`
- `get_metadata()`
- `enable_eager_attention()`
- `normalize_feature_map_to_bchw()`
- target names and target indices
- artifact naming conventions
- Google Drive output paths
- feature map shape verification
- Swin-B output format handling

---

## Phase 2 proposal

Read carefully:

```text
Phase_2_GradCAM_Proposal.md
```

This is the implementation specification for Phase 2.

Follow it closely.

If the proposal conflicts with the actual codebase or Phase 1 implementation, follow the codebase and Phase 1 implementation, then document the deviation in:

```text
Phase_2_IMPLEMENTATION_NOTES.md
```

---

# IMPORTANT CONTEXT

The current best model is the multimodal configuration:

```text
Image Backbone: Swin-B
Text Backbone: PhoBERT
Fusion: Cross-Attention
Loss: Log-Cosh
Target outputs: 5 regression scores
```

Targets:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

Grad-CAM must be run one target at a time.

For example:

```text
target_idx = 0 -> food_score
target_idx = 1 -> price_score
target_idx = 2 -> atmosphere_score
target_idx = 3 -> service_score
target_idx = 4 -> overall_satisfaction
```

---

# IMPLEMENTATION REQUIREMENTS

## 1. Reuse Phase 1 infrastructure

Do not duplicate code that already exists in Phase 1.

Use:

```python
from xai.config import ...
from xai.utils import ...
```

where appropriate.

Do not re-create:

- target mappings
- model loading
- sample loading
- prediction formatting
- figure saving
- raw value saving
- feature map normalization
- metadata creation

unless the Phase 1 implementation is missing something required for Phase 2.

---

## 2. Create Phase 2 Grad-CAM module

Create:

```text
xai/gradcam_explainer.py
```

This file should contain reusable code for Grad-CAM.

It should not be notebook-only logic.

At minimum, implement:

```text
GradCAMExplainer
MultiTargetScoreWrapper
SwinTransformerReshapeTransform
RegressionScoreTarget, if needed
manual per-image Grad-CAM helpers
heatmap overlay helpers
comparison figure helpers
metadata saving helpers
```

Use type hints and docstrings.

---

## 3. Multi-image handling

The model supports up to 4 images per review.

Do NOT explain only the first image.

Do NOT explain padded black images.

Implement Grad-CAM for all real images:

```text
for image_idx in range(num_real_images):
    generate Grad-CAM for that image
```

Use the multi-image strategy approved by the proposal:

```text
Process all images through the full multimodal model to preserve context,
but isolate activations and gradients for each real image via hooks.
```

This ensures the explanation reflects the real inference computation.

---

## 4. Target-specific Grad-CAM

Every Grad-CAM run must be target-specific.

Do NOT backpropagate from all 5 outputs at once.

Use exactly one target per run:

```python
score = output[:, target_idx]
score.backward()
```

Generate heatmaps separately for:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

---

## 5. Keep text fixed

When explaining image branch using Grad-CAM:

- text input must remain fixed
- attention mask must remain fixed
- only image evidence is analyzed

This avoids multimodal confounding.

Document this behavior in code comments and metadata.

---

## 6. Swin-B feature map format

Phase 1 already found/handled that Swin-B may output feature maps as:

```text
[B, H, W, C]
```

or

```text
[B, N, C]
```

or

```text
[B, C, H, W]
```

Reuse Phase 1's:

```python
normalize_feature_map_to_bchw()
```

Do not assume a fixed shape.

Grad-CAM logic must support the actual feature format verified in Phase 1.

---

## 7. Target layer detection

Implement robust target layer selection.

Preferred target layer:

```python
model.image_model.encoder.norm
```

But if this does not exist or does not return a valid spatial feature map, fallback to suitable Swin-B layers detected from the codebase.

The selected layer and feature map metadata must be saved into output metadata.

---

## 8. Artifact output location

Follow the existing project and Drive workflow.

Generated Phase 2 artifacts should be saved under the experiment folder on Drive, for example:

```text
/content/drive/MyDrive/SE365/experiments/EXP_060A_bestsequential_full_configuration/xai/gradcam/
```

Use the current project’s path conventions.

Do not invent a conflicting folder structure.

At minimum create:

```text
gradcam/
raw/
metadata/
figures/
```

or the structure already defined in Phase 2 proposal.

---

## 9. Required output artifacts

For each processed sample, save:

```text
Per-image Grad-CAM overlay PNG
Per-target Grad-CAM overlay PNG
Raw CAM .npy
5-target comparison figure
Multi-image grid figure, if sample has multiple images
Metadata JSON
```

Also save:

```text
gradcam_batch_summary.json
```

All saved paths must be printed in the notebook immediately after saving.

Example:

```text
Saved:
/content/drive/MyDrive/SE365/experiments/.../xai/gradcam/sample_0042_target0_food_score_img0.png
```

---

## 10. Notebook implementation

Create:

```text
xai/notebooks/Phase2_GradCAM.ipynb
```

The notebook must follow the same professional style as Phase 1 notebook.

It must include:

1. Title and explanation
2. Environment setup
3. Path configuration
4. Imports
5. Seed and device setup
6. Load model
7. Load sample
8. Show sample text and images
9. Verify Grad-CAM target layer
10. Generate Grad-CAM for one target
11. Generate Grad-CAM for all 5 targets
12. Generate Grad-CAM for multi-image sample
13. Save all artifacts
14. Run target-specific sanity check
15. Run reproducibility sanity check
16. Save batch summary
17. Final PASS/FAIL summary

Every section must print progress logs.

---

## 11. Logging requirements

Every major notebook cell must print:

```text
============================================================
Phase 2 — Step X/Y — Step Name
============================================================
```

Whenever a file is saved, print:

```text
Saved:
<absolute path>
```

Every error should be readable and actionable.

Do not allow silent failures.

---

## 12. Validation requirements

Implement validation checks:

### Technical validation

- model loads correctly
- sample loads correctly
- Grad-CAM target layer found
- feature map normalized to `[B, C, H, W]`
- raw CAM shape is valid
- CAM values are finite
- CAM values are normalized to `[0, 1]`
- no CAM generated for padding images
- file count matches expectation

### Target-specific validation

For one sample, compute pairwise correlation between 5 target heatmaps.

If all heatmaps are nearly identical, print warning.

### Reproducibility validation

Run the same Grad-CAM twice on the same sample and target.

Compare raw CAM arrays using `np.allclose`.

Save result in metadata.

---

## 13. Error handling

Handle these cases gracefully:

- `pytorch-grad-cam` not installed
- checkpoint missing
- config missing
- no test CSV found
- image file missing
- sample has only black placeholder image
- target layer not found
- gradient is zero
- CAM contains NaN or Inf
- GPU out of memory

When recoverable, print warning and continue.

When unrecoverable, raise a clear error message.

---

## 14. Do not break existing code

Do NOT modify existing training, evaluation, or experiment notebooks unless absolutely necessary.

If you need to modify Phase 1 utilities, keep changes backward-compatible.

Do not break Phase 1 notebook.

---

## 15. Install dependency if needed

If `pytorch-grad-cam` is missing, the Phase 2 notebook should include a Colab-safe install cell:

```python
!pip install grad-cam
```

But do not force reinstall if already installed.

---

## 16. Implementation notes

After finishing implementation, create:

```text
Phase_2_IMPLEMENTATION_NOTES.md
```

This file must contain:

1. What was implemented exactly as specified
2. What deviated from the proposal
3. Why deviations were necessary
4. Engineering decisions
5. Assumptions
6. How Phase 2 reuses Phase 1
7. What artifacts are generated
8. Known limitations
9. How Phase 3 can reuse Phase 2 results

---

## 17. Fix report

If you had to fix any existing issue in Phase 1 utilities or notebook, create/update:

```text
Phase_2_FIX_REPORT.md
```

Include:

- root cause
- files modified
- fix applied
- compatibility impact
- future phase impact

---

# QUALITY REQUIREMENTS

The implementation must be:

```text
research-grade
industry-grade
reproducible
maintainable
notebook-friendly
Colab-friendly
clear enough for thesis defense
```

Code must include:

- type hints
- docstrings
- readable function names
- no unnecessary global state
- no duplicate Phase 1 logic
- consistent artifact names
- consistent metadata
- robust path handling
- explicit logs
- clear failure messages

---

# SELF REVIEW AFTER IMPLEMENTATION

After implementing Phase 2, perform a complete static review.

Do NOT just stop after writing files.

Review:

- all imports
- all paths
- notebook cell order
- shape assumptions
- target index handling
- multi-image handling
- hook registration and removal
- gradient clearing
- file saving
- metadata completeness
- compatibility with Phase 1
- compatibility with future Phase 3
- whether the notebook can be run from top to bottom in Colab

Fix any issue you find.

Repeat review until no obvious problems remain.

Do not run training.

Do not change the trained model.

Do not regenerate experiments.

Only implement Phase 2 XAI Grad-CAM infrastructure and notebook.
