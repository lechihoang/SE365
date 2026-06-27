# Phase 6 Fix Report — Missing Artifact Panels

## 1. Root Cause

Two interconnected issues:

### Issue A: Shallow artifact checking
`check_sample_artifacts()` only checked whether phase **directories** existed (`os.path.isdir()`), not whether actual artifact **files** were present inside them. An empty directory would pass the check.

### Issue B: No artifact filter in sample selection
The selection functions (`select_correct_cases`, etc.) did not filter out samples with zero XAI artifacts. Samples were selected purely based on prediction error characteristics. Since most samples in the prediction CSV have NOT been processed by Phases 2-5 (only 15 samples per phase), the majority of selected samples had no artifacts at all.

### Combined effect
Phase 6 selected samples based on error profiles → most had no XAI artifacts → `create_combined_figure()` could not find any artifact files → all panels showed "not available."

## 2. Files Modified

| File | Changes |
|---|---|
| `xai/case_study.py` | `check_sample_artifacts()`: now checks for specific artifact files (not just directories). `compute_selection_score()`: returns 0.0 for zero-artifact samples. All 7 case type selectors: added `_has_artifacts()` filter to skip samples with no XAI outputs. |
| `xai/notebooks/Phase6_CaseStudy.ipynb` | Step 7: added artifact availability diagnostic table showing per-sample artifact status before pipeline runs. |
| `doc/XAI/Phase_6_FIX_REPORT.md` | This document. |

## 3. Artifact Filename Patterns (Verified)

The filename patterns in `create_combined_figure()` are **correct** — they match what Phases 2-5 actually generate:

| Panel | Expected filename | Phase that creates it | Verified |
|---|---|---|---|
| Grad-CAM | `gradcam_img0_{factor}.png` | Phase 2 (line 825) | Correct |
| Attention | `cls_importance_word_bar.png` | Phase 3 (line 792) | Correct |
| SHAP | `shap_modality_contribution.png` | Phase 4 (line 786) | Correct |
| LIME Image | `{sample_id}_lime_image_{factor}_positive.png` | Phase 5 (line 527) | Correct |
| LIME Text | `{sample_id}_lime_text_{factor}_bar.png` | Phase 5 (line 688) | Correct |

## 4. Fixes Applied

### Fix 1: Deep artifact checking
`check_sample_artifacts()` now checks for specific files:
- Grad-CAM: any `gradcam_img0_{factor}.png` or `gradcam_5target_comparison.png`
- Attention: `cls_importance_word_bar.png` or `attention_layer11_mean_heatmap.png`
- SHAP: `shap_modality_contribution.png` or `.json`
- LIME: any `{sample_id}_lime_image_{factor}_positive.png` or `_text_{factor}_bar.png`

### Fix 2: Zero-score for artifact-less samples
`compute_selection_score()` returns `0.0` when `completeness == 0`, ensuring these samples can never be selected over samples with artifacts.

### Fix 3: Artifact filter in all selectors
Added `if not _has_artifacts(idx): continue` to all 7 case type selection loops. This skips samples with no XAI artifacts entirely during candidate generation.

### Fix 4: Diagnostic table in notebook
Step 7 now prints a per-sample artifact availability table before running the pipeline, allowing the user to see which samples are eligible.

## 5. Remaining Limitations

1. If Phases 2-5 were only run on 15 samples each, and those 15 are the same across phases, only those 15 can be selected as case studies. If they're different, the intersection may be smaller.
2. The `_load_original_image()` function still uses the MD5 hash + `.jpg` extension. If images are stored with different extensions, they won't be found. This is consistent with `src/dataset.py` which also uses `.jpg`.

## 6. How to Verify

1. Run the notebook. Step 7 should print the artifact diagnostic table.
2. Only samples with `OK` in at least one column should appear in the selection.
3. Combined figures should have populated panels for available phases and clean placeholders only for genuinely missing phases.
4. `selection_log.json` should show which samples were selected and their artifact completeness.

## 7. Example Before/After

**Before (broken):**
```
sample_0000: GradCAM=OK  Attn=OK  SHAP=OK  LIME=OK  → selected
sample_0050: GradCAM=--  Attn=--  SHAP=--  LIME=--  → also selected (error matched)
sample_0100: GradCAM=--  Attn=--  SHAP=--  LIME=--  → also selected
→ Result: many "not available" panels
```

**After (fixed):**
```
sample_0000: GradCAM=OK  Attn=OK  SHAP=OK  LIME=OK  → selected (score=0.87)
sample_0050: GradCAM=--  Attn=--  SHAP=--  LIME=--  → SKIPPED (score=0.0)
sample_0100: GradCAM=--  Attn=--  SHAP=--  LIME=--  → SKIPPED (score=0.0)
→ Result: panels populated for all available phases
```
