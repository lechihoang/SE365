# ROLE

You are a Principal XAI Engineer, Senior PyTorch Debugging Engineer, and Research Software Architect.

You specialize in:

- Grad-CAM
- PyTorch hooks
- timm Swin Transformer
- Multimodal Deep Learning
- Regression XAI
- Debugging attribution methods
- Research-grade notebook engineering

---

# GOAL

Read the entire codebase and all Phase 2 implementation files.

Diagnose why the Grad-CAM heatmaps for all 5 targets:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

look almost identical.

Then fix the implementation if there is a bug or design issue.

If the root cause cannot be diagnosed purely from code inspection, add diagnostic logging and sanity-check cells to the notebook so the cause can be identified when the notebook runs.

After fixing, update the Phase 2 notebook so it runs Grad-CAM for **15 samples** instead of only 3 samples.

---

# FILES TO READ FIRST

Read the entire codebase.

Especially read:

```text
Models/ImageModel.py
Models/TextModel.py
Models/CrossAttentionFusion.py
Models/FusionModel.py
Models/GMUFusion.py
Models/GatedCrossModalFusion.py
Models/FiLMFusion.py
src/dataset.py
main.py
test.py
xai/config.py
xai/utils.py
xai/gradcam_explainer.py
xai/notebooks/Phase2_GradCAM.ipynb
Phase_2_GradCAM_Proposal.md
Phase_2_IMPLEMENTATION_NOTES.md
Phase_2_FIX_REPORT.md
Phase_1_IMPLEMENTATION_NOTES.md
```

Understand exactly:

- how ImageModel processes multi-image inputs
- where Swin-B pooling happens
- what tensor shape the target layer returns
- whether hooks are attached before or after spatial pooling
- how gradients flow from each target score back to the image encoder
- whether each target actually produces different gradients
- whether CAM normalization hides differences
- whether CAMs are accidentally reused across targets

---

# CURRENT OBSERVATION

Grad-CAM runs successfully.

However, for several samples, the heatmaps for:

```text
Food
Price
Atmosphere
Service
Overall Satisfaction
```

look almost identical.

This may mean one of the following:

1. Implementation bug.
2. Wrong target layer.
3. Target-specific backward is not actually target-specific.
4. CAMs are being reused accidentally.
5. Gradients from all 5 outputs to the image branch are almost identical.
6. Final Swin-B layer is too coarse or too close to global pooling.
7. Min-max normalization makes weak target CAMs look artificially strong.
8. The selected sample genuinely uses the same visual evidence for all targets.

Your task is to determine which explanation is most likely.

---

# DIAGNOSTIC PRIORITY

First, try to diagnose by reading the code.

Only if static code inspection cannot prove the cause, add diagnostic logging/cells to the notebook.

Do not blindly add logs before understanding the code.

---

# REQUIRED DIAGNOSTIC CHECKS

Add the following diagnostic checks to the Phase 2 notebook and/or `gradcam_explainer.py` as needed.

## 1. Target score check

For each sample, print the model predictions:

```text
food_score
price_score
atmosphere_score
service_score
overall_satisfaction
```

Verify that the 5 predicted scores are not all identical.

---

## 2. Target-specific backward check

For each target index:

```python
target_score = preds[0, target_idx]
target_score.backward()
```

Print:

```text
target_idx
target_name
target_score
```

Verify the correct target is selected each time.

---

## 3. Gradient statistics per target

For each target, compute gradient stats at the hooked image layer:

```text
grad_mean
grad_std
grad_abs_mean
grad_abs_max
nonzero_ratio
```

If all gradient stats are identical across 5 targets, investigate why.

---

## 4. Gradient similarity matrix

For one sample and one image, compute cosine similarity between flattened gradients of all 5 targets.

Save/print a 5×5 matrix:

```text
Gradient Similarity Matrix
```

Interpretation:

- similarity ≈ 1.0 for all pairs means image-branch gradients are nearly identical across targets
- similarity clearly below 1.0 means targets are different, but visualization may hide differences

---

## 5. Raw CAM similarity matrix

Compute pairwise correlation between raw CAM arrays for all 5 targets.

Save/print a 5×5 matrix:

```text
Raw CAM Correlation Matrix
```

Do this before overlay and before any visual formatting.

---

## 6. Raw CAM value range

For each target, print:

```text
cam_min
cam_max
cam_mean
cam_std
```

This checks whether min-max normalization makes weak/flat CAMs look artificially strong.

---

## 7. Target layer comparison

Try multiple candidate target layers:

```text
image_model.encoder.norm
image_model.encoder.layers[-1]
image_model.encoder.layers[-1].blocks[-1]
image_model.encoder.layers[-1].blocks[-1].norm2
```

Use only those that exist in the actual code.

For each candidate layer, generate CAMs and compute target similarity.

Select the best layer based on:

- valid spatial feature map
- nonzero gradients
- target-specific differences
- semantic plausibility
- stability

Do not assume `encoder.norm` is always best.

---

## 8. CAM reuse bug check

Verify that the loop creates a new CAM for every:

```text
image_idx
target_idx
```

Check that:

- CAM arrays are not the same object
- CAM arrays are not overwritten
- saved filenames are unique
- dictionary keys include both image_idx and target_idx

---

## 9. Multi-image indexing check

Verify that when the model flattens images as `[B*N, C, H, W]`, the code slices the correct image index.

For current batch size B=1, image index mapping is:

```text
flat_index = image_idx
```

But implement it explicitly and document it.

If later batch size > 1, use:

```text
flat_index = batch_idx * max_images + image_idx
```

---

# FIX REQUIREMENTS

After diagnosis, fix the root issue.

Potential fixes may include:

## If target layer is too late/coarse

Update target layer selection to prefer the layer that gives the best target-specific Grad-CAM.

Document why the new layer is better.

---

## If gradients are actually identical

Do not fake target specificity.

Instead:

- keep the implementation correct
- add diagnostics showing gradient similarity
- explain in notes that the image branch provides similar visual evidence for these targets
- recommend using SHAP or text attribution for target-level differences

---

## If normalization hides differences

Save both:

```text
normalized CAM
raw CAM
```

Also add optional fixed-scale visualization or side-by-side raw statistic reporting.

---

## If CAMs are accidentally reused

Fix the loop/storage/saving logic.

---

## If wrong target is selected

Fix target indexing immediately.

---

# NOTEBOOK UPDATE REQUIREMENT

Update:

```text
xai/notebooks/Phase2_GradCAM.ipynb
```

so that it runs Grad-CAM for **15 samples** instead of 3.

The notebook should define:

```python
NUM_GRADCAM_SAMPLES = 15
```

and use this value consistently.

Do not hardcode 15 in multiple places.

---

# SAMPLE SELECTION REQUIREMENT

Do not simply use the first 15 rows blindly if there is a better existing sample selection method.

Implement a reasonable selection strategy:

1. Include several high-confidence correct samples.
2. Include several high-error samples.
3. Include several multi-image samples.
4. Include several samples with different dominant visual content if possible.

If the notebook cannot implement this robustly yet, fall back to the first 15 test samples but clearly document this limitation.

---

# OUTPUT REQUIREMENTS

After the fix, the notebook should save:

```text
gradcam outputs for 15 samples
diagnostic summary json
gradient similarity matrices
raw CAM similarity matrices
target layer comparison results
updated batch summary json
```

All outputs should be saved under the existing Phase 2 output folder, for example:

```text
/content/drive/MyDrive/SE365/experiments/EXP_060A_bestsequential_full_configuration/xai/gradcam/
```

Every saved file must print:

```text
Saved:
<absolute path>
```

---

# IMPLEMENTATION NOTES

After finishing, update or create:

```text
Phase_2_IMPLEMENTATION_NOTES.md
```

Add a new section:

```text
## Grad-CAM Target Similarity Diagnosis
```

Explain:

- whether the issue was a bug or an expected behavior
- what evidence supports the conclusion
- which diagnostics were added
- which target layer was selected
- whether target-specificity improved
- remaining limitations

---

# FIX REPORT

Create or update:

```text
Phase_2_FIX_REPORT.md
```

Include:

1. Problem observed
2. Root cause analysis
3. Files modified
4. Diagnostic checks added
5. Fixes applied
6. Remaining risks
7. How to verify the fix by running the notebook

---

# QUALITY REQUIREMENTS

The implementation must be:

- professional
- reproducible
- Colab-friendly
- notebook-friendly
- research-grade
- thesis-ready

Do not break Phase 1.

Do not break existing experiment code.

Do not modify model weights.

Do not retrain anything.

Do not use test results to select a new model.

Only debug and improve the Grad-CAM implementation.

---

# FINAL SELF-REVIEW

After making changes, perform a complete static review.

Check:

- imports
- paths
- notebook cell order
- target indexing
- target layer selection
- hook cleanup
- gradient zeroing
- CAM storage keys
- CAM file names
- metadata completeness
- diagnostics saving
- 15-sample execution logic
- compatibility with Phase 1 utilities
- compatibility with future XAI phases

Fix any issue found before finishing.
