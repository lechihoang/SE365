# Phase 6: Case Study Selection and Analysis

## Implementation Proposal

---

# 1. Purpose

### Why This Phase Exists

Case studies transform abstract XAI metrics into concrete, human-interpretable examples. Phases 2 through 5 produce explanations across the entire dataset: Grad-CAM heatmaps, attention maps, SHAP modality contributions, and LIME perturbation results. However, raw XAI artifacts alone do not tell a scientific story. A thesis examiner will always ask:

> "Show me a specific example where the model used the right evidence."
>
> "Show me an example where image and text disagreed."
>
> "Show me a failure case and explain why the model failed."

Phase 6 exists to answer these questions systematically. It selects representative samples across seven distinct case types, assembles multi-method explanation panels for each, and generates structured analysis text. The result is a curated portfolio of 10 to 17 case studies that demonstrate the model's reasoning across success, failure, agreement, conflict, and modality dominance scenarios.

### Research Motivation

Multi-method XAI is strongest when demonstrated on carefully chosen examples that showcase different model behaviors. Selecting only correct predictions would be cherry-picking. Selecting only failures would be misleading. A balanced case study portfolio proves scientific rigor and helps the thesis argue that the model has been examined from multiple angles.

### Engineering Motivation

Without an automated selection pipeline, case studies are chosen manually, which introduces selection bias, is not reproducible, and does not scale. Phase 6 provides a deterministic, criteria-based selection process that produces consistent results across experiment runs.

---

# 2. Objectives

### Research Objectives

1. Demonstrate that the XAI framework can explain model behavior across diverse scenarios: correct predictions, failures, modality dominance, and inter-modal conflicts.
2. Identify specific patterns in how the Swin-B + PhoBERT + CrossAttentionFusion model uses image and text evidence for different quality aspects.
3. Provide concrete examples for thesis defense that illustrate both model strengths and limitations.
4. Show how multiple XAI methods (Grad-CAM, Attention, SHAP, optionally LIME) converge or diverge on the same sample.

### Engineering Objectives

1. Algorithmically select 10 to 17 samples matching seven predefined case type criteria.
2. Generate combined multi-method explanation figures suitable for thesis inclusion (12x10 inches, 300 DPI, Vietnamese-capable fonts).
3. Produce structured metadata JSON and analysis text for each case study.
4. Create a master index CSV and summary document linking all cases.
5. Ensure the entire pipeline is reproducible and deterministic given the same experiment artifacts.

### Expected Contributions

- A case study selection algorithm that other multimodal XAI projects can adapt.
- A combined figure generator that assembles cross-method explanations into single thesis-ready panels.
- A template-based analysis generator that produces structured natural-language summaries of model behavior.

---

# 3. Inputs

### Prediction Data

| File | Source | Contents |
|---|---|---|
| `experiments/EXP_XXX/test_predictions.csv` | `test.py` | Per-sample: `index`, `split`, `y_true_{factor}`, `y_pred_{factor}`, `absolute_error_{factor}` for all 5 targets |
| `experiments/EXP_XXX/predictions.csv` | `Trainer.py` | Same format for the validation set |

Column names for factors: `food`, `price`, `atmos`, `service`, `overall`.

### Dataset CSVs

| File | Key Columns |
|---|---|
| `data/text/test.csv` or `data/text/val.csv` | `comment_clean`, `image_url`, `food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction` |

The `image_url` column contains a Python list literal of up to 4 image URLs per review, parsed via `ast.literal_eval()` in `src/dataset.py`.

### XAI Artifacts from Previous Phases

| Phase | Artifact Directory | Key Files |
|---|---|---|
| Phase 2 (Grad-CAM) | `experiments/EXP_XXX/xai/gradcam/` | `sample_{idx}_target{t}_{factor}_overlay.png`, `sample_{idx}_target{t}_{factor}_raw.npy` |
| Phase 3 (Attention) | `experiments/EXP_XXX/xai/attention/` | `sample_{idx}_token_heatmap.png`, `sample_{idx}_cls_importance.npy`, `sample_{idx}_word_importance.json` |
| Phase 4 (SHAP) | `experiments/EXP_XXX/xai/shap/` | `sample_{idx}_shap_values.npy`, `sample_{idx}_modality_contribution.json` |
| Phase 5 (LIME) | `experiments/EXP_XXX/xai/lime/` (if completed) | `sample_{idx}_target{t}_lime_image.png`, `sample_{idx}_target{t}_lime_text.json` |

### SHAP Modality Contribution Data

The SHAP modality contribution JSON (from Phase 4) contains per-sample, per-target:

```json
{
  "sample_id": 42,
  "contributions": {
    "food":    {"text_pct": 0.38, "image_pct": 0.62},
    "price":   {"text_pct": 0.71, "image_pct": 0.29},
    "atmos":   {"text_pct": 0.25, "image_pct": 0.75},
    "service": {"text_pct": 0.55, "image_pct": 0.45},
    "overall": {"text_pct": 0.42, "image_pct": 0.58}
  }
}
```

### Model Checkpoint

| File | Purpose |
|---|---|
| `experiments/EXP_XXX/best_model_train_fusion.pth` | Best model checkpoint (Swin-B + PhoBERT + CrossAttentionFusion + LogCosh) |

### Configuration

| File | Purpose |
|---|---|
| `experiments/EXP_XXX/config.yaml` or `config.json` | Experiment hyperparameters and architecture settings |

---

# 4. Outputs

### Per Case Study

Each case study directory contains:

| File | Format | Description |
|---|---|---|
| `combined_figure_target{t}_{factor}.png` | PNG, 12x10 in, 300 DPI | Multi-method explanation panel for one target |
| `metadata.json` | JSON | All quantitative data for this case |
| `analysis.md` | Markdown | Structured analysis text |
| `individual_views/` | Directory | Symlinks or copies of individual XAI artifacts |

### Summary Outputs

| File | Format | Description |
|---|---|---|
| `case_study_index.csv` | CSV | Master list of all selected cases with columns: `case_id`, `case_type`, `sample_id`, `split`, `mean_absolute_error`, `key_finding`, `figure_path` |
| `case_study_summary.md` | Markdown | Combined narrative across all case types |

---

# 5. Architecture Attachment Point

Phase 6 does **NOT** attach to any layer of the model architecture. It does not modify the model, does not run inference, and does not compute gradients.

Phase 6 is a **pure consumer** of outputs from Phases 2 through 5. It reads:

1. Prediction CSVs (from `test.py` or `Trainer.py`)
2. XAI artifact files (from Phases 2-5)
3. Dataset CSVs (for review text and image URLs)

And it **organizes** them into:

1. Selected case study samples (based on quantitative criteria)
2. Combined explanation figures (assembled from individual XAI artifacts)
3. Structured analysis text (generated from templates and metadata)

```
Phase 2 (Grad-CAM) ──────┐
Phase 3 (Attention) ─────┤
Phase 4 (SHAP) ──────────┼──▶ Phase 6 (Case Study) ──▶ Combined Figures
Phase 5 (LIME) ──────────┤                            Metadata JSONs
Prediction CSVs ─────────┤                            Analysis Text
Dataset CSVs ────────────┘                            Master Index
```

---

# 6. Detailed Implementation Plan

## A. Case Study Selector: `xai/case_study_selector.py`

This module contains all selection logic. It reads prediction data and XAI artifacts, applies criteria for each case type, and returns lists of candidate sample indices.

### A.1. `load_predictions(exp_dir, split='test')`

**Responsibility:** Load the prediction CSV into a pandas DataFrame.

**Inputs:**
- `exp_dir`: path to experiment directory (e.g., `experiments/EXP_060A`)
- `split`: `'test'` loads `test_predictions.csv`, `'val'` loads `predictions.csv`

**Returns:** DataFrame with columns: `index`, `split`, `y_true_food`, `y_true_price`, `y_true_atmos`, `y_true_service`, `y_true_overall`, `y_pred_food`, `y_pred_price`, `y_pred_atmos`, `y_pred_service`, `y_pred_overall`, `absolute_error_food`, `absolute_error_price`, `absolute_error_atmos`, `absolute_error_service`, `absolute_error_overall`, `max_error` (computed), `mean_error` (computed).

**Processing steps:**
1. Read the CSV from `{exp_dir}/test_predictions.csv` or `{exp_dir}/predictions.csv`.
2. Compute `max_error` = max of all 5 `absolute_error_{factor}` columns per row.
3. Compute `mean_error` = mean of all 5 `absolute_error_{factor}` columns per row.
4. Return the DataFrame.

### A.2. `load_shap_contributions(shap_dir)`

**Responsibility:** Load SHAP modality contribution data for all samples that have it.

**Inputs:**
- `shap_dir`: path to `experiments/EXP_XXX/xai/shap/`

**Returns:** Dictionary mapping `sample_id` to modality contribution dict. Each entry has per-target `text_pct` and `image_pct`.

**Processing steps:**
1. Glob for `*_modality_contribution.json` files in `shap_dir`.
2. Parse each JSON file.
3. Extract `sample_id` from the filename pattern.
4. Return the dictionary.

### A.3. `load_dataset_text(csv_path)`

**Responsibility:** Load the dataset CSV to access review text and image URLs.

**Inputs:**
- `csv_path`: path to `data/text/test.csv` or `data/text/val.csv`

**Returns:** DataFrame with at minimum: `comment_clean`, `image_url`.

### A.4. `check_xai_artifacts(sample_id, exp_dir, required_phases)`

**Responsibility:** Verify that a candidate sample has all required XAI artifacts from specified phases.

**Inputs:**
- `sample_id`: integer index
- `exp_dir`: experiment directory path
- `required_phases`: list of phase names, e.g., `['gradcam', 'attention', 'shap']`

**Returns:** Boolean. `True` only if ALL required artifact files exist for this sample.

**Processing steps:**
1. For each phase, check the expected file pattern:
   - `gradcam`: at least one `sample_{sample_id}_target*_overlay.png` exists
   - `attention`: `sample_{sample_id}_token_heatmap.png` or `sample_{sample_id}_word_importance.json` exists
   - `shap`: `sample_{sample_id}_modality_contribution.json` exists
   - `lime`: `sample_{sample_id}_target*_lime_image.png` exists (optional check)
2. Return `True` if all required phases have artifacts; `False` otherwise.

### A.5. `select_correct_cases(df, threshold=0.3, n=3)`

**Responsibility:** Find samples where the model predicts ALL 5 targets accurately.

**Criteria:** `max_error < threshold` (all 5 absolute errors are below the threshold).

**Algorithm:**
1. Filter DataFrame where `max_error < threshold`.
2. Sort by `mean_error` ascending (best predictions first).
3. Return the top `n` indices.
4. **Fallback:** If fewer than 2 samples match, relax threshold to 0.5, then 0.8, then 1.0. Document the final threshold used.

### A.6. `select_high_error_cases(df, threshold=2.0, n=3)`

**Responsibility:** Find samples where the model fails significantly on at least one target.

**Criteria:** `max_error > threshold` (at least one absolute error exceeds the threshold).

**Algorithm:**
1. Filter DataFrame where `max_error > threshold`.
2. Sort by `max_error` descending (worst predictions first).
3. Return the top `n` indices.
4. **Fallback:** If fewer than 2 samples match, relax threshold to 1.5, then 1.0.
5. For diversity: try to select samples that fail on different targets (e.g., one fails on `food`, another on `price`).

**Diversity sub-algorithm:**
1. Group high-error samples by which target has the highest error.
2. Select one sample from each group (up to `n` groups).
3. If fewer than `n` groups, fill remaining slots with the highest overall errors.

### A.7. `select_text_dominant_cases(shap_contributions, threshold=0.70, n=2)`

**Responsibility:** Find samples where text modality dominates the prediction.

**Criteria:** SHAP `text_pct > threshold` for at least 3 out of 5 targets.

**Algorithm:**
1. For each sample in `shap_contributions`, count how many targets have `text_pct > threshold`.
2. Filter samples where count >= 3.
3. Sort by average `text_pct` across all 5 targets, descending.
4. Return the top `n` indices.
5. **Fallback:** If fewer than 1 sample matches, relax threshold to 0.60, then 0.55.

### A.8. `select_image_dominant_cases(shap_contributions, threshold=0.60, n=2)`

**Responsibility:** Find samples where image modality dominates the prediction.

**Criteria:** SHAP `image_pct > threshold` for at least 3 out of 5 targets.

**Algorithm:**
1. Mirror of `select_text_dominant_cases` but for `image_pct`.
2. Filter samples where count >= 3.
3. Sort by average `image_pct` across all 5 targets, descending.
4. Return the top `n` indices.
5. **Fallback:** If fewer than 1 sample matches, relax threshold to 0.55, then 0.50.

### A.9. `select_conflict_cases(df, shap_contributions, text_data, n=3)`

**Responsibility:** Find samples where image and text evidence point in opposing directions.

**Criteria:** A conflict is detected when:
- SHAP shows one modality pushing a target score UP (positive signed contribution) while the other pushes it DOWN (negative signed contribution), AND
- the sample has moderate-to-high error on that target (absolute_error > 0.8), suggesting the model struggled with the conflicting signals.

**Algorithm:**
1. **Primary heuristic (SHAP-based):** For samples with signed SHAP data, identify cases where the signed text contribution and signed image contribution have opposite signs for the same target, AND the magnitude difference is large (both > 0.2 of total absolute contribution).
2. **Secondary heuristic (keyword + error):** If signed SHAP is not available, use a keyword-based approach:
   - Define positive Vietnamese sentiment keywords: `['ngon', 'tuyệt vời', 'tuyệt', 'xuất sắc', 'đẹp', 'sạch', 'thơm', 'nóng', 'tươi', 'thích', 'ưng', 'hài lòng', 'ok', 'ổn', 'tốt', 'hay', 'vui', 'nhanh', 'chu đáo', 'nhiệt tình', 'rẻ', 'hợp lý', 'xứng đáng']`
   - Define negative Vietnamese sentiment keywords: `['dở', 'tệ', 'chán', 'bẩn', 'hôi', 'lạnh', 'nguội', 'tanh', 'đắt', 'mắc', 'quá giá', 'chậm', 'lâu', 'thái độ', 'hỗn', 'cọc', 'ồn', 'chật', 'nóng', 'thiếu', 'ít', 'nhỏ', 'không ngon', 'không tốt', 'thất vọng']`
   - Score the review text for overall sentiment direction.
   - Compare text sentiment direction with prediction error direction (predicted higher or lower than true).
   - Select samples where text sentiment is positive but the model under-predicted, or text sentiment is negative but the model over-predicted, suggesting image evidence opposed the text.
3. Sort candidates by the strength of the conflict signal.
4. Return the top `n` indices.
5. **Fallback:** If the primary heuristic yields fewer than 1 sample, fall back to secondary. If secondary also yields fewer than 1, select from high-error cases and mark as "potential conflict -- manual verification needed."

### A.10. `select_difficult_cases(df, text_data, n=2)`

**Responsibility:** Find samples where the review text does not mention certain aspects, yet the model must still predict those scores.

**Criteria:** The review text does not contain keywords related to a specific target, but the model's prediction for that target has low-to-moderate error (the model inferred the score from other evidence, likely image).

**Algorithm:**
1. Define keyword sets per target:
   - `food`: `['ăn', 'món', 'đồ ăn', 'thức ăn', 'ngon', 'dở', 'nấu', 'vị', 'mùi', 'thơm', 'tanh', 'tươi', 'nóng', 'nguội', 'bánh', 'cơm', 'phở', 'bún', 'mì', 'gà', 'bò', 'heo', 'cá', 'tôm', 'rau', 'trái cây', 'kem', 'nước', 'trà', 'cà phê', 'bia', 'rượu']`
   - `price`: `['giá', 'tiền', 'đắt', 'mắc', 'rẻ', 'hợp lý', 'xứng đáng', 'quá giá', 'chấp nhận', 'phải chăng', 'đáng tiền', 'bình dân']`
   - `service`: `['nhân viên', 'phục vụ', 'thái độ', 'nhanh', 'chậm', 'lâu', 'chu đáo', 'nhiệt tình', 'hỗn', 'cọc', 'niềm nở', 'thân thiện', 'chuyên nghiệp', 'order', 'gọi món']`
   - `atmosphere/atmos`: `['không gian', 'không khí', 'view', 'đẹp', 'sạch', 'bẩn', 'ồn', 'yên tĩnh', 'thoáng', 'chật', 'trang trí', 'nội thất', 'ánh sáng', 'điều hòa', 'máy lạnh', 'ghế', 'bàn', 'wifi', 'parking', 'đỗ xe']`
2. For each sample, check which target keyword sets are absent from `comment_clean` (case-insensitive, normalized).
3. A "difficult" sample is one where at least 1 target's keywords are completely absent from the text, AND the model's absolute error for that target is < 1.0 (the model still made a reasonable prediction without textual evidence).
4. Prefer samples where more targets lack keyword coverage.
5. Sort by number of missing-keyword targets descending, then by mean error ascending (successful inference from context).
6. Return the top `n` indices.

### A.11. `select_agreement_cases(df, shap_contributions, n=2)`

**Responsibility:** Find samples where both modalities provide consistent, balanced evidence and the model predicts accurately.

**Criteria:**
- `mean_error < 0.5` (good overall prediction)
- SHAP contributions are balanced: for at least 3 targets, `0.35 < text_pct < 0.65` (neither modality dominates)

**Algorithm:**
1. Filter for low mean error.
2. Filter for balanced SHAP contributions.
3. Sort by how close the average text_pct is to 0.50 (most balanced first).
4. Return the top `n` indices.
5. **Fallback:** Relax error threshold to 0.8 and balance window to `0.30 < text_pct < 0.70`.

### A.12. `run_case_selection(exp_dir, dataset_csv_path, split='test', required_phases=['gradcam', 'attention', 'shap'])`

**Responsibility:** Orchestrate the full selection pipeline.

**Algorithm:**
1. Load predictions via `load_predictions()`.
2. Load SHAP contributions via `load_shap_contributions()`.
3. Load dataset text via `load_dataset_text()`.
4. Run each selection function.
5. For each candidate, verify XAI artifacts exist via `check_xai_artifacts()`.
6. Remove duplicates (if a sample qualifies for multiple case types, assign it to the most informative type using priority: conflict > high_error > text_dominant > image_dominant > difficult > agreement > correct).
7. If any case type has zero samples after filtering, log a warning and document it.
8. Return a dictionary mapping case_type to list of selected sample indices.
9. Save the selection results to `case_study_index.csv`.

**De-duplication priority rationale:** Conflict cases are rarest and most scientifically interesting. High-error cases demonstrate failure analysis. Dominance cases demonstrate modality contribution. Difficult and agreement cases are supplementary. Correct cases are easiest to replenish.

---

## B. Combined Figure Generator: `xai/case_study_figure.py`

This module assembles individual XAI artifacts into single thesis-ready combined figures.

### B.1. `configure_matplotlib_vietnamese()`

**Responsibility:** Configure matplotlib to render Vietnamese diacritics correctly.

**Implementation:**
1. Set `matplotlib.rcParams['font.family']` to a Unicode-complete font: try `'Noto Sans'` first, fall back to `'DejaVu Sans'`, then `'Arial'`.
2. Set `matplotlib.rcParams['axes.unicode_minus'] = False`.
3. Test rendering by creating a small figure with Vietnamese text (e.g., "Thức ăn ngon, giá hợp lý") and verifying no missing-glyph boxes appear.
4. Log the font actually used.

### B.2. `load_original_image(image_url, image_dir)`

**Responsibility:** Load the original restaurant image from local cache or URL.

**Implementation:**
1. Compute MD5 hash of the URL (matching the logic in `src/dataset.py` line 30-31).
2. Check `{image_dir}/{url_hash}.jpg`.
3. If exists, load and return as PIL Image.
4. If not, attempt to download from URL (with timeout=5).
5. If download fails, return a placeholder image with text "Image not available".

### B.3. `create_combined_figure(sample_id, exp_dir, target_idx, target_name, dataset_row, prediction_row, shap_contribution, image_dir, output_path, include_lime=False)`

**Responsibility:** Create a single combined figure assembling all XAI evidence for one sample and one target.

**Figure layout:**

```
┌─────────────────────────────────────────────────────────┐
│  Title: Case {case_id} | Sample {sample_id}             │
│  Target: {target_name} | True: {y_true} | Pred: {y_pred} │
│  Error: {abs_error} | Case Type: {case_type}             │
├────────────────────────┬────────────────────────────────┤
│  Row 1, Col 1:         │  Row 1, Col 2:                 │
│  Original Image        │  Grad-CAM Overlay              │
│  (first/primary image) │  (for this target)             │
├────────────────────────┴────────────────────────────────┤
│  Row 2:                                                  │
│  Left: Attention Token Importance Bar Chart              │
│  Right: SHAP Modality Contribution Bar (text% vs img%)   │
├─────────────────────────────────────────────────────────┤
│  Row 3 (optional, if include_lime=True):                 │
│  Left: LIME Image Explanation                            │
│  Right: LIME Text Word Importance                        │
├─────────────────────────────────────────────────────────┤
│  Footer: Review text (truncated to 200 chars)            │
│  All 5 predictions: food=X, price=X, atmos=X, ...        │
└─────────────────────────────────────────────────────────┘
```

**Implementation steps:**
1. Call `configure_matplotlib_vietnamese()`.
2. Determine figure grid: 3 rows (or 4 if LIME included) x 2 columns.
3. Row 1: Load original image. Load Grad-CAM overlay PNG. Display side by side.
4. Row 2, Left: Load attention word importance data. Create horizontal bar chart of top-10 most important words (after subword merging). Use warm colormap for positive importance.
5. Row 2, Right: Load SHAP modality contribution for this target. Create grouped bar chart: one bar for `text_pct`, one for `image_pct`, with percentage labels.
6. Row 3 (optional): Load LIME image explanation. Load LIME text word importance. Display.
7. Add title with sample metadata (sample ID, target name, true score, predicted score, absolute error).
8. Add footer with the first 200 characters of `comment_clean` and all 5 predictions.
9. Set figure size to `(12, 10)` inches (or `(12, 13)` if LIME row included).
10. Save at 300 DPI to `output_path`.
11. Close the figure to free memory.

### B.4. `create_all_target_figures(sample_id, exp_dir, case_type, dataset_row, prediction_row, shap_contribution, image_dir, output_dir, targets_to_show='key')`

**Responsibility:** Generate combined figures for multiple targets for one case study.

**Target selection logic (`targets_to_show`):**
- `'all'`: Generate for all 5 targets.
- `'key'`: Generate for the 2 most informative targets based on case type:
  - `correct`: target with lowest error + `overall`
  - `high_error`: target with highest error + `overall`
  - `text_dominant`: target with highest `text_pct` + `overall`
  - `image_dominant`: target with highest `image_pct` + `overall`
  - `conflict`: target with the conflict signal + `overall`
  - `difficult`: target with missing keywords + `overall`
  - `agreement`: `food` + `overall` (representative pair)

### B.5. `create_multi_target_comparison(sample_id, exp_dir, output_path)`

**Responsibility:** Create a single overview figure showing Grad-CAM heatmaps for all 5 targets side-by-side, demonstrating that the model uses different visual evidence for different quality aspects.

**Figure layout:**
```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ food     │ price    │ atmos    │ service  │ overall  │
│ GradCAM  │ GradCAM  │ GradCAM  │ GradCAM  │ GradCAM  │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

Figure size: `(20, 4)` inches, 300 DPI.

---

## C. Case Study Metadata Generator: `xai/case_study_metadata.py`

### C.1. `generate_metadata(sample_id, case_type, exp_dir, dataset_row, prediction_row, shap_contribution)`

**Responsibility:** Create a comprehensive JSON metadata file for one case study.

**Output JSON structure:**

```json
{
  "case_id": "case_correct_001",
  "case_type": "correct",
  "sample_id": 42,
  "split": "test",
  "review_text": "Quán rất ngon, nhân viên phục vụ nhiệt tình...",
  "review_text_length": 85,
  "image_urls": ["https://...", "https://..."],
  "num_images": 2,
  "ground_truth": {
    "food": 4.0,
    "price": 3.5,
    "atmos": 4.0,
    "service": 4.5,
    "overall": 4.0
  },
  "predicted": {
    "food": 3.92,
    "price": 3.61,
    "atmos": 3.85,
    "service": 4.38,
    "overall": 3.95
  },
  "absolute_errors": {
    "food": 0.08,
    "price": 0.11,
    "atmos": 0.15,
    "service": 0.12,
    "overall": 0.05
  },
  "max_error": 0.15,
  "mean_error": 0.102,
  "shap_modality_contribution": {
    "food":    {"text_pct": 0.38, "image_pct": 0.62},
    "price":   {"text_pct": 0.71, "image_pct": 0.29},
    "atmos":   {"text_pct": 0.25, "image_pct": 0.75},
    "service": {"text_pct": 0.55, "image_pct": 0.45},
    "overall": {"text_pct": 0.42, "image_pct": 0.58}
  },
  "top_attention_tokens": ["ngon", "phuc_vu", "nhiet_tinh", "khong_gian", "ok"],
  "gradcam_focus_description": {
    "food": "Grad-CAM highlights the food dish in the center of the image",
    "atmos": "Grad-CAM highlights the restaurant interior and seating area"
  },
  "xai_artifacts": {
    "gradcam_overlay_food": "xai/gradcam/sample_42_target0_food_overlay.png",
    "gradcam_overlay_overall": "xai/gradcam/sample_42_target4_overall_overlay.png",
    "attention_heatmap": "xai/attention/sample_42_token_heatmap.png",
    "shap_contribution": "xai/shap/sample_42_modality_contribution.json"
  },
  "combined_figures": [
    "xai/case_studies/case_correct_001/combined_figure_target0_food.png",
    "xai/case_studies/case_correct_001/combined_figure_target4_overall.png"
  ],
  "selection_criteria": {
    "threshold_used": 0.3,
    "criterion_description": "All 5 absolute errors below 0.3"
  }
}
```

---

## D. Analysis Text Generator: `xai/case_study_analysis.py`

### D.1. Template System

This module generates structured natural-language analysis for each case study based on the metadata and XAI evidence. It uses a template system with placeholders that are filled from the metadata JSON.

### D.2. Templates by Case Type

**Template: Correct Prediction**

```
## Case Study: {case_id}

### Summary
The model correctly predicted all 5 quality aspects for this restaurant review with a maximum
absolute error of {max_error:.2f}. The mean absolute error across all targets was {mean_error:.3f}.

### Prediction Overview
| Aspect | Ground Truth | Predicted | Absolute Error |
|--------|-------------|-----------|---------------|
| Food | {gt_food} | {pred_food} | {err_food:.3f} |
| Price | {gt_price} | {pred_price} | {err_price:.3f} |
| Atmosphere | {gt_atmos} | {pred_atmos} | {err_atmos:.3f} |
| Service | {gt_service} | {pred_service} | {err_service:.3f} |
| Overall | {gt_overall} | {pred_overall} | {err_overall:.3f} |

### Image Evidence (Grad-CAM)
For the {primary_target} target, Grad-CAM highlighted {gradcam_focus_description}. This suggests
the image encoder correctly identified visual cues relevant to {primary_target} assessment.

### Text Evidence (Attention)
The attention mechanism highlighted the following tokens as most interactive:
{top_tokens_formatted}. These tokens are semantically consistent with the predicted scores.

### Modality Contribution (SHAP)
For {primary_target}, SHAP analysis showed {text_pct:.0%} text contribution and
{image_pct:.0%} image contribution. {modality_dominance_statement}

### Interpretation
This case demonstrates that the model uses {modality_balance_description} to arrive at accurate
predictions. The alignment between Grad-CAM visual evidence, attention-highlighted tokens, and
SHAP modality contributions supports the conclusion that the model's reasoning is consistent
with human interpretation for this sample.
```

**Template: High-Error Case**

```
## Case Study: {case_id}

### Summary
The model produced a significant prediction error on this sample. The highest error was
{max_error:.2f} on the {worst_target} target (predicted {pred_worst:.2f} vs true {gt_worst:.2f}).

### Prediction Overview
[same table as correct template]

### Error Analysis
The model {error_direction} the {worst_target} score by {max_error:.2f} points.

### Image Evidence (Grad-CAM)
Grad-CAM for {worst_target} showed {gradcam_focus_description}. {gradcam_diagnosis}: the model
{gradcam_interpretation}.

### Text Evidence (Attention)
The attention mechanism highlighted: {top_tokens_formatted}. {attention_diagnosis}: {attention_interpretation}.

### Modality Contribution (SHAP)
SHAP showed {text_pct:.0%} text and {image_pct:.0%} image contribution for {worst_target}.
{shap_diagnosis}.

### Possible Failure Causes
{failure_causes_list}

### Interpretation
This failure case reveals {failure_interpretation}. This finding suggests areas for model
improvement, such as {improvement_suggestion}.
```

**Template: Image-Text Conflict**

```
## Case Study: {case_id}

### Summary
This sample exhibits a conflict between image and text evidence for the {conflict_target} target.
The model predicted {pred_conflict:.2f} against a true score of {gt_conflict:.2f}.

### Conflict Evidence
- **Image evidence direction:** {image_evidence_direction}
- **Text evidence direction:** {text_evidence_direction}
- **SHAP text contribution:** {text_pct:.0%} ({text_direction})
- **SHAP image contribution:** {image_pct:.0%} ({image_direction})

### Image Evidence (Grad-CAM)
Grad-CAM highlighted {gradcam_focus_description}, suggesting {gradcam_interpretation}.

### Text Evidence (Attention + Keywords)
The review text contains {text_sentiment_keywords}. Attention highlighted {top_tokens_formatted},
indicating {attention_interpretation}.

### Resolution
The fusion module resolved this conflict by {resolution_description}. The final prediction
{resolution_assessment}.

### Interpretation
This case demonstrates how the CrossAttentionFusion module handles contradictory evidence from
image and text modalities. {scientific_insight}.
```

**Template: Text-Dominant**

```
## Case Study: {case_id}

### Summary
Text evidence dominated the model's prediction for this sample, with an average text contribution
of {avg_text_pct:.0%} across all targets.

### Modality Dominance
| Target | Text % | Image % |
|--------|--------|---------|
[table of contributions per target]

### Text Evidence
The review text reads: "{review_text_truncated}"
Key attention tokens: {top_tokens_formatted}.
The text contains {text_evidence_description}, which strongly drives the prediction.

### Image Evidence
Despite lower image contribution, Grad-CAM showed {gradcam_focus_description}.
The image evidence was {image_evidence_assessment}.

### Interpretation
Text dominance in this sample is {expected_or_unexpected} because {reasoning}.
```

**Template: Image-Dominant**

Similar structure to text-dominant, with inverted focus: emphasize what visual evidence the model used and why text contributed less.

**Template: Difficult Sample**

```
## Case Study: {case_id}

### Summary
The review text does not mention {missing_aspects} explicitly, yet the model predicted
{missing_aspect_scores} with errors of {missing_aspect_errors}.

### Missing Textual Evidence
Keywords searched but not found: {missing_keywords_per_aspect}.
The review text: "{review_text_truncated}"

### How the Model Compensated
Without explicit textual cues for {missing_aspects}, the model relied on:
- **Image evidence:** {gradcam_description} (SHAP image contribution: {image_pct:.0%})
- **Contextual text cues:** {contextual_text_evidence}
- **Cross-modal inference:** {cross_modal_description}

### Interpretation
This case demonstrates the model's ability to infer {missing_aspects} quality from {inference_source}.
```

**Template: Agreement**

```
## Case Study: {case_id}

### Summary
Both image and text evidence provide consistent, balanced support for the model's accurate
prediction. Mean error: {mean_error:.3f}. Average modality balance: {avg_text_pct:.0%} text
/ {avg_image_pct:.0%} image.

### Evidence Alignment
[describe how Grad-CAM, attention, and SHAP all tell a consistent story]

### Interpretation
This case represents an ideal multimodal prediction scenario where both modalities contribute
complementary evidence without conflict.
```

### D.3. `generate_analysis(case_id, case_type, metadata, template_dir=None)`

**Responsibility:** Fill the appropriate template with data from the metadata JSON.

**Algorithm:**
1. Select the template based on `case_type`.
2. Extract all required values from `metadata`.
3. Compute derived descriptions:
   - `modality_dominance_statement`: "Text dominated" / "Image dominated" / "Both modalities contributed roughly equally"
   - `error_direction`: "overestimated" / "underestimated"
   - `gradcam_diagnosis`: semi-automated from Grad-CAM focus region description
   - `failure_causes_list`: generated from a decision tree based on error magnitude, modality contribution, and text content
4. Fill template placeholders.
5. Write to `analysis.md` in the case study directory.

### D.4. Failure Cause Decision Tree

For high-error cases, generate candidate failure causes from the following decision tree:

```
IF max_error > 2.0:
  IF text_pct > 0.70:
    -> "Strong text dominance may have overridden visual evidence"
  IF image_pct > 0.70:
    -> "Strong image dominance may have overridden textual evidence"
  IF review_text contains sarcasm indicators (positive words + low true score):
    -> "Possible sarcasm in review text that the model did not detect"
  IF num_images == 0 or images are all black:
    -> "Missing or low-quality images may have degraded image evidence"
  IF comment_length < 20:
    -> "Very short review provides insufficient textual evidence"
  IF error_direction == 'overestimated' AND target == 'price':
    -> "Model may have misinterpreted price-related sentiment"
  DEFAULT:
    -> "Complex interaction between image and text evidence led to prediction error"
```

---

## E. Notebook: `xai/notebooks/Phase6_CaseStudy.ipynb`

Detailed cell-by-cell design is provided in Section 9.

---

# 7. Required Code Files

| File | Responsibility |
|---|---|
| `xai/case_study_selector.py` | All case selection algorithms. Reads prediction CSVs, SHAP contributions, and dataset text. Returns selected sample indices per case type. Contains functions A.1 through A.12. |
| `xai/case_study_figure.py` | Combined figure generator. Reads individual XAI artifact images and data, assembles them into multi-panel thesis-ready figures. Contains functions B.1 through B.5. |
| `xai/case_study_metadata.py` | Metadata JSON generator. Collects all quantitative data for a case study into a structured JSON file. Contains function C.1. |
| `xai/case_study_analysis.py` | Template-based analysis text generator. Fills markdown templates with case study data to produce structured natural-language summaries. Contains functions D.1 through D.4. |
| `xai/case_study_runner.py` | Orchestration script. Calls selector, then iterates over selected cases to generate figures, metadata, and analysis. Produces the master index CSV and summary document. |
| `xai/notebooks/Phase6_CaseStudy.ipynb` | Interactive notebook for exploring, verifying, and refining case studies. |

### Dependency Map

```
case_study_runner.py
    ├── case_study_selector.py
    │     ├── reads: predictions.csv / test_predictions.csv
    │     ├── reads: xai/shap/*_modality_contribution.json
    │     └── reads: dataset CSV (comment_clean, image_url)
    ├── case_study_figure.py
    │     ├── reads: xai/gradcam/*_overlay.png
    │     ├── reads: xai/attention/*_word_importance.json
    │     ├── reads: xai/shap/*_modality_contribution.json
    │     ├── reads: xai/lime/*_lime_image.png (optional)
    │     └── reads: original images from image_dir
    ├── case_study_metadata.py
    │     └── reads: all of the above
    └── case_study_analysis.py
          └── reads: metadata.json
```

---

# 8. Folder Structure

```
experiments/EXP_XXX/xai/case_studies/
├── case_correct_001/
│   ├── combined_figure_target0_food.png
│   ├── combined_figure_target4_overall.png
│   ├── multi_target_gradcam_comparison.png
│   ├── metadata.json
│   ├── analysis.md
│   └── individual_views/
│       ├── gradcam_target0_food_overlay.png  (symlink or copy)
│       ├── gradcam_target4_overall_overlay.png
│       ├── attention_token_heatmap.png
│       └── shap_modality_contribution.json
├── case_correct_002/
│   └── (same structure)
├── case_correct_003/
│   └── (same structure)
├── case_higherror_001/
│   └── (same structure, with focus on error target)
├── case_higherror_002/
│   └── (same structure)
├── case_higherror_003/
│   └── (same structure)
├── case_conflict_001/
│   └── (same structure, with conflict evidence highlighted)
├── case_conflict_002/
│   └── (same structure)
├── case_conflict_003/
│   └── (same structure)
├── case_textdominant_001/
│   └── (same structure)
├── case_textdominant_002/
│   └── (same structure)
├── case_imagedominant_001/
│   └── (same structure)
├── case_imagedominant_002/
│   └── (same structure)
├── case_difficult_001/
│   └── (same structure)
├── case_difficult_002/
│   └── (same structure)
├── case_agreement_001/
│   └── (same structure)
├── case_agreement_002/
│   └── (same structure)
├── case_study_index.csv
├── case_study_summary.md
└── selection_log.json
```

### File Naming Convention

- Case directories: `case_{type}_{sequence:03d}/` where type is one of `correct`, `higherror`, `conflict`, `textdominant`, `imagedominant`, `difficult`, `agreement`.
- Combined figures: `combined_figure_target{idx}_{factor_name}.png` where `idx` is 0-4 and `factor_name` matches `['food', 'price', 'atmos', 'service', 'overall']`.
- Multi-target comparison: `multi_target_gradcam_comparison.png`.
- Selection log: `selection_log.json` records all thresholds used, fallback actions taken, and de-duplication decisions.

### Expected Case Count

| Case Type | Target Count | Min | Max |
|---|---|---|---|
| correct | 3 | 2 | 4 |
| high_error | 3 | 2 | 4 |
| conflict | 3 | 1 | 3 |
| text_dominant | 2 | 1 | 3 |
| image_dominant | 2 | 1 | 3 |
| difficult | 2 | 1 | 2 |
| agreement | 2 | 1 | 2 |
| **Total** | **17** | **10** | **21** |

---

# 9. Notebook Design: `xai/notebooks/Phase6_CaseStudy.ipynb`

### Cell 1: Configuration and Imports

**Type:** Code

**Content:**
- Import `pandas`, `numpy`, `matplotlib.pyplot`, `json`, `os`, `sys`, `PIL.Image`, `IPython.display`
- Add parent directory to `sys.path` for module imports
- Import `case_study_selector`, `case_study_figure`, `case_study_metadata`, `case_study_analysis`
- Define experiment configuration:
  ```
  EXP_DIR = '../experiments/EXP_060A'
  DATASET_CSV = '../data/text/test.csv'
  IMAGE_DIR = '../data/image'
  SPLIT = 'test'
  REQUIRED_PHASES = ['gradcam', 'attention', 'shap']
  ```
- Define target names: `FACTOR_NAMES = ['food', 'price', 'atmos', 'service', 'overall']`
- Define target indices: `FACTOR_INDICES = {name: idx for idx, name in enumerate(FACTOR_NAMES)}`

**Expected output:** Configuration summary printed to stdout.

### Cell 2: Load Prediction Data

**Type:** Code

**Content:**
- Call `load_predictions(EXP_DIR, SPLIT)` to get the predictions DataFrame.
- Display `df.describe()` for error columns.
- Display histogram of `max_error` to understand error distribution.

**Expected output:** Summary statistics table and error distribution plot.

### Cell 3: Load SHAP Contributions

**Type:** Code

**Content:**
- Call `load_shap_contributions(os.path.join(EXP_DIR, 'xai', 'shap'))`.
- Print number of samples with SHAP data.
- Display average modality contributions across all samples as a summary bar chart.

**Expected output:** Count of available SHAP samples, summary bar chart.

### Cell 4: Load Dataset Text

**Type:** Code

**Content:**
- Call `load_dataset_text(DATASET_CSV)`.
- Display a few sample rows showing `comment_clean` and `image_url`.

**Expected output:** Sample rows from the dataset.

### Cell 5: Run Case Selection -- Correct Cases

**Type:** Code

**Content:**
- Call `select_correct_cases(df, threshold=0.3, n=3)`.
- Display selected sample indices and their error profiles.
- Verify XAI artifacts exist for each selected sample.
- Print the review text for each selected sample.

**Expected output:** Table of 3 selected correct-prediction samples with full error details.

### Cell 6: Run Case Selection -- High-Error Cases

**Type:** Code

**Content:**
- Call `select_high_error_cases(df, threshold=2.0, n=3)`.
- Display selected samples with their worst-target errors.
- Print the review text for context.

**Expected output:** Table of 3 high-error samples.

### Cell 7: Run Case Selection -- Text-Dominant Cases

**Type:** Code

**Content:**
- Call `select_text_dominant_cases(shap_contributions, threshold=0.70, n=2)`.
- Display SHAP contributions per target for selected samples.

**Expected output:** Table of 2 text-dominant samples with per-target modality percentages.

### Cell 8: Run Case Selection -- Image-Dominant Cases

**Type:** Code

**Content:**
- Call `select_image_dominant_cases(shap_contributions, threshold=0.60, n=2)`.
- Display SHAP contributions per target.

**Expected output:** Table of 2 image-dominant samples.

### Cell 9: Run Case Selection -- Conflict Cases

**Type:** Code

**Content:**
- Call `select_conflict_cases(df, shap_contributions, text_data, n=3)`.
- Display conflict evidence for each selected sample.
- Show which target has the conflict and in which direction.

**Expected output:** Table of up to 3 conflict samples with conflict analysis.

### Cell 10: Run Case Selection -- Difficult Cases

**Type:** Code

**Content:**
- Call `select_difficult_cases(df, text_data, n=2)`.
- Display which aspect keywords are missing from each review.
- Show the model's prediction accuracy on those aspects.

**Expected output:** Table of 2 difficult samples with missing-keyword analysis.

### Cell 11: Run Case Selection -- Agreement Cases

**Type:** Code

**Content:**
- Call `select_agreement_cases(df, shap_contributions, n=2)`.
- Display balanced modality contribution evidence.

**Expected output:** Table of 2 agreement samples.

### Cell 12: Selection Summary and De-duplication

**Type:** Code

**Content:**
- Aggregate all selections.
- Apply de-duplication using the priority scheme from Section A.12.
- Display final `case_study_index.csv` as a formatted table.
- Print total case count and verify it is in the 10-17 range.

**Expected output:** Final case study index table.

### Cell 13: Generate Combined Figure -- Example (Correct Case)

**Type:** Code

**Content:**
- Select the first correct case.
- Call `create_combined_figure()` for `target_idx=0` (food) and `target_idx=4` (overall).
- Display the generated figures inline.
- Allow visual inspection before batch generation.

**Expected output:** Two combined figure PNGs displayed inline.

### Cell 14: Generate Combined Figure -- Example (High-Error Case)

**Type:** Code

**Content:**
- Select the first high-error case.
- Generate combined figure for the worst-error target.
- Display inline.

**Expected output:** One combined figure PNG for the error case.

### Cell 15: Generate Combined Figure -- Example (Conflict Case)

**Type:** Code

**Content:**
- Select the first conflict case.
- Generate combined figure for the conflict target.
- Display inline with conflict evidence annotations.

**Expected output:** One combined figure PNG for the conflict case.

### Cell 16: Batch Generate All Case Study Artifacts

**Type:** Code

**Content:**
- Loop over all selected case studies.
- For each case study:
  1. Create the case directory.
  2. Generate combined figures via `create_all_target_figures()`.
  3. Generate metadata JSON via `generate_metadata()`.
  4. Generate analysis text via `generate_analysis()`.
  5. Copy/symlink individual XAI artifacts into `individual_views/`.
- Print progress bar (tqdm).
- Report any failures.

**Expected output:** Progress log showing all case studies generated.

### Cell 17: Generate Multi-Target Grad-CAM Comparison (Selected Cases)

**Type:** Code

**Content:**
- For 2-3 key cases (one correct, one high-error, one conflict), generate the 5-target Grad-CAM comparison figure.
- Display inline.

**Expected output:** Multi-target comparison figures.

### Cell 18: Generate Case Study Summary Document

**Type:** Code

**Content:**
- Read all metadata JSONs.
- Compile `case_study_summary.md` with:
  - Executive summary (total cases, types, key findings).
  - One paragraph per case type summarizing patterns.
  - Key figures referenced by path.
- Write summary file.

**Expected output:** Summary markdown file generated.

### Cell 19: Validation Checks

**Type:** Code

**Content:**
- Verify all expected files exist in each case directory.
- Verify metadata JSON is valid and complete.
- Verify combined figures are non-zero size and readable by PIL.
- Verify analysis.md is non-empty.
- Cross-check that prediction data in metadata matches `test_predictions.csv`.
- Report pass/fail for each check.

**Expected output:** Validation report with pass/fail for each case study.

### Cell 20: Thesis-Ready Figure Selection

**Type:** Code

**Content:**
- Select 3-5 figures recommended for the Results chapter.
- Select additional figures for the Appendix.
- Display the recommended figures with suggested captions.
- Print suggested caption text for each figure.

**Expected output:** Curated figure selection with thesis captions.

### Cell 21: Markdown (Closing Notes)

**Type:** Markdown

**Content:**
- Phase 6 completion summary.
- Links to generated artifacts.
- Notes on any fallback thresholds used or unusual findings.

---

# 10. Algorithm

### Master Algorithm: Case Study Pipeline

```
PROCEDURE run_phase6_pipeline(exp_dir, dataset_csv, image_dir, split):

    # ═══════════════════════════════════════════
    # STEP 1: LOAD ALL DATA SOURCES
    # ═══════════════════════════════════════════
    
    predictions_df = load_predictions(exp_dir, split)
        → Columns: index, y_true_{factor}, y_pred_{factor}, absolute_error_{factor}
        → Add: max_error = MAX(absolute_error_food, ..., absolute_error_overall)
        → Add: mean_error = MEAN(absolute_error_food, ..., absolute_error_overall)
    
    shap_contributions = load_shap_contributions(exp_dir + '/xai/shap/')
        → Dict: sample_id → {target: {text_pct, image_pct}}
    
    text_data = load_dataset_text(dataset_csv)
        → Columns: comment_clean, image_url

    # ═══════════════════════════════════════════
    # STEP 2: SELECT CANDIDATES PER CASE TYPE
    # ═══════════════════════════════════════════
    
    candidates = {}
    
    candidates['correct'] = select_correct_cases(predictions_df, threshold=0.3, n=3)
        → IF count < 2: relax to 0.5, then 0.8, then 1.0
        → Sort by mean_error ascending
    
    candidates['high_error'] = select_high_error_cases(predictions_df, threshold=2.0, n=3)
        → IF count < 2: relax to 1.5, then 1.0
        → Diversify across failed targets
        → Sort by max_error descending
    
    candidates['text_dominant'] = select_text_dominant_cases(shap_contributions, threshold=0.70, n=2)
        → Count targets where text_pct > threshold
        → IF count < 1: relax to 0.60, then 0.55
    
    candidates['image_dominant'] = select_image_dominant_cases(shap_contributions, threshold=0.60, n=2)
        → Count targets where image_pct > threshold
        → IF count < 1: relax to 0.55, then 0.50
    
    candidates['conflict'] = select_conflict_cases(predictions_df, shap_contributions, text_data, n=3)
        → Primary: signed SHAP opposite directions + moderate error
        → Secondary: keyword sentiment vs error direction
        → IF count < 1: mark as manual-verification-needed
    
    candidates['difficult'] = select_difficult_cases(predictions_df, text_data, n=2)
        → Missing aspect keywords + low error on that aspect
    
    candidates['agreement'] = select_agreement_cases(predictions_df, shap_contributions, n=2)
        → Low error + balanced SHAP contributions

    # ═══════════════════════════════════════════
    # STEP 3: VERIFY XAI ARTIFACTS
    # ═══════════════════════════════════════════
    
    FOR each case_type in candidates:
        FOR each sample_id in candidates[case_type]:
            IF NOT check_xai_artifacts(sample_id, exp_dir, ['gradcam', 'attention', 'shap']):
                REMOVE sample_id from candidates[case_type]
                LOG warning: "Sample {sample_id} missing XAI artifacts, excluded"

    # ═══════════════════════════════════════════
    # STEP 4: DE-DUPLICATE
    # ═══════════════════════════════════════════
    
    priority_order = ['conflict', 'high_error', 'text_dominant', 
                      'image_dominant', 'difficult', 'agreement', 'correct']
    
    assigned_samples = SET()
    final_selection = {}
    
    FOR case_type in priority_order:
        final_selection[case_type] = []
        FOR sample_id in candidates[case_type]:
            IF sample_id NOT IN assigned_samples:
                final_selection[case_type].append(sample_id)
                assigned_samples.add(sample_id)
    
    total_cases = SUM(len(v) for v in final_selection.values())
    ASSERT 10 <= total_cases <= 21
    LOG "Total case studies selected: {total_cases}"

    # ═══════════════════════════════════════════
    # STEP 5: GENERATE ARTIFACTS FOR EACH CASE
    # ═══════════════════════════════════════════
    
    case_index_rows = []
    
    FOR case_type in final_selection:
        FOR seq, sample_id in enumerate(final_selection[case_type], start=1):
            case_id = f"case_{case_type}_{seq:03d}"
            case_dir = f"{exp_dir}/xai/case_studies/{case_id}/"
            CREATE_DIRECTORY(case_dir)
            
            # Retrieve data for this sample
            dataset_row = text_data.iloc[sample_id]
            prediction_row = predictions_df[predictions_df['index'] == sample_id].iloc[0]
            shap_data = shap_contributions.get(sample_id, None)
            
            # 5a. Generate metadata JSON
            metadata = generate_metadata(
                sample_id, case_type, exp_dir,
                dataset_row, prediction_row, shap_data
            )
            WRITE_JSON(metadata, f"{case_dir}/metadata.json")
            
            # 5b. Determine which targets to generate figures for
            key_targets = determine_key_targets(case_type, prediction_row, shap_data)
            
            # 5c. Generate combined figures
            FOR target_idx, target_name in key_targets:
                output_path = f"{case_dir}/combined_figure_target{target_idx}_{target_name}.png"
                create_combined_figure(
                    sample_id, exp_dir, target_idx, target_name,
                    dataset_row, prediction_row, shap_data,
                    image_dir, output_path
                )
            
            # 5d. Generate multi-target Grad-CAM comparison
            create_multi_target_comparison(
                sample_id, exp_dir,
                f"{case_dir}/multi_target_gradcam_comparison.png"
            )
            
            # 5e. Generate analysis text
            analysis_text = generate_analysis(case_id, case_type, metadata)
            WRITE_FILE(analysis_text, f"{case_dir}/analysis.md")
            
            # 5f. Copy individual XAI artifacts
            copy_individual_views(sample_id, exp_dir, case_dir)
            
            # 5g. Add to index
            case_index_rows.append({
                'case_id': case_id,
                'case_type': case_type,
                'sample_id': sample_id,
                'split': split,
                'mean_error': prediction_row['mean_error'],
                'max_error': prediction_row['max_error'],
                'key_finding': summarize_key_finding(case_type, metadata),
                'figure_paths': [list of combined figure paths]
            })

    # ═══════════════════════════════════════════
    # STEP 6: SAVE MASTER INDEX AND SUMMARY
    # ═══════════════════════════════════════════
    
    WRITE_CSV(case_index_rows, f"{exp_dir}/xai/case_studies/case_study_index.csv")
    
    summary_text = compile_case_study_summary(final_selection, case_index_rows)
    WRITE_FILE(summary_text, f"{exp_dir}/xai/case_studies/case_study_summary.md")
    
    selection_log = {
        'timestamp': current_timestamp,
        'exp_dir': exp_dir,
        'split': split,
        'thresholds_used': {all threshold values, including fallbacks},
        'total_cases': total_cases,
        'per_type_counts': {case_type: len(samples) for each type},
        'excluded_samples': [list of samples excluded due to missing artifacts],
        'deduplication_decisions': [list of reassignment decisions]
    }
    WRITE_JSON(selection_log, f"{exp_dir}/xai/case_studies/selection_log.json")
    
    RETURN final_selection
```

### Key Target Determination Sub-Algorithm

```
FUNCTION determine_key_targets(case_type, prediction_row, shap_data):
    
    factor_names = ['food', 'price', 'atmos', 'service', 'overall']
    
    IF case_type == 'correct':
        best_target_idx = ARGMIN(prediction_row[absolute_error_{factor}] for factor in factor_names)
        RETURN [(best_target_idx, factor_names[best_target_idx]), (4, 'overall')]
    
    ELIF case_type == 'high_error':
        worst_target_idx = ARGMAX(prediction_row[absolute_error_{factor}] for factor in factor_names)
        RETURN [(worst_target_idx, factor_names[worst_target_idx]), (4, 'overall')]
    
    ELIF case_type == 'text_dominant':
        IF shap_data is not None:
            most_text_idx = ARGMAX(shap_data[factor]['text_pct'] for factor in factor_names)
            RETURN [(most_text_idx, factor_names[most_text_idx]), (4, 'overall')]
        RETURN [(0, 'food'), (4, 'overall')]
    
    ELIF case_type == 'image_dominant':
        IF shap_data is not None:
            most_image_idx = ARGMAX(shap_data[factor]['image_pct'] for factor in factor_names)
            RETURN [(most_image_idx, factor_names[most_image_idx]), (4, 'overall')]
        RETURN [(0, 'food'), (4, 'overall')]
    
    ELIF case_type == 'conflict':
        # Return the target with the strongest conflict signal
        conflict_target_idx = identify_conflict_target(prediction_row, shap_data)
        RETURN [(conflict_target_idx, factor_names[conflict_target_idx]), (4, 'overall')]
    
    ELIF case_type == 'difficult':
        # Return the target with missing keywords
        missing_target_idx = identify_missing_keyword_target(prediction_row, text)
        RETURN [(missing_target_idx, factor_names[missing_target_idx]), (4, 'overall')]
    
    ELIF case_type == 'agreement':
        RETURN [(0, 'food'), (4, 'overall')]
```

---

# 11. Validation

### 11.1. Structural Validation

| Check | Method | Pass Criterion |
|---|---|---|
| Every case type has at least 1 case | Count cases per type in `case_study_index.csv` | All 7 types have count >= 1 |
| Total cases in range | Count rows in `case_study_index.csv` | 10 <= total <= 21 |
| No duplicate samples across case types | Check `sample_id` uniqueness in index | No duplicates |
| All case directories exist | Glob `case_studies/case_*/` | Count matches index row count |

### 11.2. File Completeness Validation

| Check | Method | Pass Criterion |
|---|---|---|
| Each case has `metadata.json` | Check file existence | All cases pass |
| Each case has at least 1 combined figure | Check `combined_figure_*.png` existence | All cases pass |
| Each case has `analysis.md` | Check file existence | All cases pass |
| Combined figures are valid PNGs | Open with PIL and verify `.size` | All images have non-zero dimensions |
| Combined figures are correct size | Check PIL `.size` against expected DPI | Width ~ 3600 px (12 in * 300 DPI) |

### 11.3. Metadata Consistency Validation

| Check | Method | Pass Criterion |
|---|---|---|
| `sample_id` in metadata matches prediction CSV | Cross-reference `sample_id` with `test_predictions.csv` `index` column | All match |
| Ground truth scores in metadata match dataset CSV | Compare `ground_truth` dict with corresponding row in dataset CSV | All 5 values match |
| Predicted scores match prediction CSV | Compare `predicted` dict with `y_pred_{factor}` columns | All 5 values match (tolerance 1e-4) |
| Absolute errors are correctly computed | Verify `abs(ground_truth - predicted) == absolute_errors` for each target | All within tolerance 1e-4 |
| SHAP contributions sum to ~1.0 | Verify `text_pct + image_pct` for each target | Sum within [0.98, 1.02] |

### 11.4. Selection Criteria Validation

| Check | Method | Pass Criterion |
|---|---|---|
| Correct cases: all errors below threshold | Verify `max_error < threshold` for each correct case | All pass |
| High-error cases: at least one error above threshold | Verify `max_error > threshold` for each high-error case | All pass |
| Text-dominant: text_pct above threshold | Verify SHAP text_pct condition | All pass |
| Image-dominant: image_pct above threshold | Verify SHAP image_pct condition | All pass |
| Difficult: keywords missing from text | Verify keyword absence in `comment_clean` | All pass |

### 11.5. Visual Quality Validation (Manual)

- Open 3-5 combined figures and verify:
  - Vietnamese text renders correctly (no missing-glyph boxes).
  - Grad-CAM overlay is visible and aligned with the original image.
  - Attention bar chart shows meaningful token labels (not raw subword tokens).
  - SHAP bar chart shows correct percentages and they sum to approximately 100%.
  - Title, footer, and labels are readable at thesis-print scale.

### 11.6. Reproducibility Check

- Run the entire pipeline twice with the same inputs.
- Verify that `case_study_index.csv` is identical both times.
- Verify that `metadata.json` files are identical.
- Combined figure PNGs may differ slightly due to matplotlib rendering, but the data content (which samples, which targets, which values) must be identical.

---

# 12. Risks -- FULLY ANALYZED

## R1: Not Enough Samples Matching Criteria

### Problem
Some case types may have very few or zero matching samples in the test set. For example, the "correct prediction" criterion (all 5 errors < 0.3) may be too strict for a model with mean MAE of 1.1079. Similarly, "high-error" with threshold 2.0 may have very few samples if the model is consistently moderate.

### Why It Happens
- The model's error distribution determines how many samples fall into each category.
- A mean MAE of 1.1079 means the average error per target is around 1.1, so errors below 0.3 across all 5 targets simultaneously are rare.
- The test set is typically smaller than the validation set, further reducing candidate pools.

### Strategy A: Strict Thresholds with Progressive Relaxation
Use the ideal thresholds first. If fewer than 2 candidates are found, relax progressively:
- Correct: 0.3 -> 0.5 -> 0.8 -> 1.0
- High-error: 2.0 -> 1.5 -> 1.0

**Advantage:** Preserves the most meaningful cases when available; gracefully degrades.
**Disadvantage:** Relaxed thresholds weaken the claim that the case truly represents its type.

### Strategy B: Use Validation Set Instead of Test Set
The validation set is larger (Trainer.py produces `predictions.csv`). Use it as a secondary pool.

**Advantage:** More candidates available.
**Disadvantage:** Validation set was seen during model selection (early stopping based on validation MAE), so predictions may be slightly optimistic.

### Strategy C: Accept Fewer Cases
If a case type has only 1 sample, include it. If 0, document that no sample met the criterion and explain what that means for the model.

**Advantage:** Honest reporting; zero-sample case types are themselves an interesting finding.
**Disadvantage:** The thesis may have fewer illustrative examples for that type.

### Engineering Trade-offs
- Strategy A is the safest engineering choice because it is automated and deterministic.
- Strategy B requires loading a second prediction file but is straightforward.
- Strategy C is acceptable for rare case types but may leave gaps in the narrative.

### Research Trade-offs
- For a thesis, having at least 1 example per case type is preferable. Zero examples for "conflict" would weaken the cross-modal analysis chapter.
- Relaxed thresholds should be clearly documented so examiners know the criteria were adjusted.

### FINAL DECISION
Use **Strategy A as primary, with Strategy C as fallback**. Start with strict thresholds. Relax progressively if fewer than 2 samples are found. If a case type has 0 samples even after maximum relaxation, accept zero and document it as a finding. The `selection_log.json` file records all threshold adjustments. Do NOT use the validation set; keep all case studies from the test set for scientific cleanliness.

### Reason
Test set predictions are the most honest measure of model behavior. Progressive relaxation is fully automated and requires no manual intervention. Documenting the final thresholds used is sufficient for thesis transparency.

---

## R2: Conflict Detection Without Sentiment Analysis

### Problem
Detecting image-text conflict requires knowing the sentiment direction of both modalities. The dataset does not include explicit sentiment labels. Text sentiment must be inferred, and "image sentiment" is not directly labeled.

### Why It Happens
- The dataset contains `comment_clean` text and `image_url` images, but no per-modality sentiment annotations.
- Vietnamese text sentiment analysis requires understanding of Vietnamese-specific words, negation patterns (e.g., "khong ngon" = not delicious), and contrast words (e.g., "nhung" = but).
- Image sentiment is not well-defined outside the model's learned representation.

### Strategy A: Keyword Heuristic
Use Vietnamese sentiment keyword lists to score the review text as positive or negative. Compare with prediction error direction.

**Advantage:** Simple to implement, no external dependencies, covers common Vietnamese sentiment expressions.
**Disadvantage:** Misses sarcasm, complex negation, and implicit sentiment. Keyword lists may be incomplete.

### Strategy B: SHAP Signed Contribution Direction
If signed SHAP values are available (not just absolute), check whether the text contribution pushes a target score UP while the image contribution pushes it DOWN (or vice versa).

**Advantage:** Uses the model's own internal evidence, which is directly relevant to the model's decision process. Does not require external sentiment analysis.
**Disadvantage:** Requires Phase 4 to save signed SHAP values (not just magnitude). If Phase 4 only saved absolute values, this strategy is unavailable.

### Strategy C: Manual Selection from High-Error Cases
Present high-error cases to the researcher and let them manually identify conflicts.

**Advantage:** Most accurate conflict identification.
**Disadvantage:** Not reproducible, introduces selection bias, does not scale.

### Engineering Trade-offs
- Strategy A is the fastest to implement but least precise.
- Strategy B is the most scientifically rigorous but depends on Phase 4 output format.
- Strategy C is best for a small number of cases but violates the algorithmic selection principle.

### Research Trade-offs
- For thesis defense, the conflict detection method must be explainable and reproducible.
- A combined A+B approach covers more cases and is stronger scientifically.

### FINAL DECISION
Use **Strategy A + B combined**. First, check for signed SHAP contribution direction disagreement (Strategy B) as the primary signal. If signed SHAP is not available for a sample, fall back to the keyword heuristic (Strategy A). Additionally, require that the sample has moderate-to-high error on the conflict target (absolute_error > 0.8) to filter out false positives where the modalities disagree internally but the model still predicts correctly.

### Reason
The combined approach maximizes recall (more conflict candidates) while the error filter reduces false positives. The keyword lists are explicitly documented, making the approach reproducible. If Phase 4 saved signed SHAP values, Strategy B provides the strongest signal.

---

## R3: Vietnamese Font Rendering in Combined Figures

### Problem
Matplotlib's default fonts may not include complete Vietnamese diacritical marks (e.g., o with horn and grave: o). This causes missing-glyph boxes or mojibake in figure titles, axis labels, and footer text.

### Why It Happens
- Matplotlib uses system fonts and its internal font catalog.
- Many common fonts (e.g., `serif`, `monospace`) do not include Vietnamese characters with combined diacritics.
- The `comment_clean` text in the dataset is full Vietnamese text with marks like a, e, o, u, and d.

### Strategy A: DejaVu Sans (Matplotlib Default)
DejaVu Sans has broad Unicode coverage and is bundled with matplotlib.

**Advantage:** No additional font installation required.
**Disadvantage:** Vietnamese support is incomplete for some combined diacritical marks. The glyph coverage depends on the specific DejaVu Sans version.

### Strategy B: Configure Noto Sans or Arial
Set `rcParams` to use a known Vietnamese-capable font such as Noto Sans or Arial.

**Advantage:** Noto Sans has excellent Vietnamese coverage. Arial also covers Vietnamese well.
**Disadvantage:** Requires the font to be installed on the system. Noto Sans may need manual installation on some environments.

### Strategy C: English Labels Only in Figures, Vietnamese in Captions
Use English for all figure text (axis labels, titles). Include Vietnamese review text only in the thesis caption or analysis markdown, not in the figure itself.

**Advantage:** Eliminates the font rendering problem entirely.
**Disadvantage:** The figure loses context because the review text is not shown inline. This weakens the combined figure's self-containedness.

### Engineering Trade-offs
- Strategy B is the best balance of reliability and readability.
- Strategy C is the safest fallback but reduces figure informativeness.

### Research Trade-offs
- Showing Vietnamese text in the figure is important for thesis context, especially for a Vietnamese restaurant review quality assessment project.
- Examiners may want to read the review text directly in the figure.

### FINAL DECISION
Use **Strategy B**. Configure matplotlib with `'Noto Sans'` as the primary font. Add a font verification function (`configure_matplotlib_vietnamese()`) that tests rendering with a sample Vietnamese string before generating any figures. If Noto Sans is not available, fall back to `'Arial'`, then to `'DejaVu Sans'`. If all fail the rendering test, truncate Vietnamese text in figures to ASCII-safe characters and add a warning to the selection log.

### Reason
Noto Sans is the industry standard for multi-script rendering and is freely available from Google Fonts. The verification function catches problems early rather than discovering mojibake in generated figures.

---

## R4: Inconsistent XAI Artifacts Across Phases

### Problem
Some samples may have Grad-CAM heatmaps from Phase 2 but not SHAP contributions from Phase 4, or vice versa. This happens when Phases 2-5 were run on different sample subsets, or when certain phases failed on specific samples.

### Why It Happens
- Different phases may have been run independently with different sample lists.
- Some phases may fail on specific samples due to edge cases (e.g., SHAP timeout on a sample with very long text, Grad-CAM failure on a sample with all-black images).
- Phases may have been run at different times with different code versions.

### Strategy A: Only Select Samples with All Required Artifacts
Filter candidates to those with complete XAI coverage before case selection.

**Advantage:** Every case study has a complete multi-method explanation panel. No missing panels.
**Disadvantage:** Reduces the candidate pool, potentially excluding interesting samples.

### Strategy B: Allow Partial Artifacts with Fallback Panels
Accept samples with some missing artifacts. Use placeholder text or blank panels in the combined figure where artifacts are missing.

**Advantage:** Larger candidate pool.
**Disadvantage:** Combined figures look inconsistent. Thesis examiners may question why some methods are missing.

### Strategy C: Re-run Missing Phases on Candidate Samples
After case selection, identify which candidates have missing artifacts. Re-run the missing phases only for those specific samples.

**Advantage:** Complete coverage for all selected cases.
**Disadvantage:** Requires re-running inference and XAI computation, which has computational cost and may introduce dependency on Phases 2-5 being runnable.

### Engineering Trade-offs
- Strategy A is simplest and safest.
- Strategy C is most thorough but adds a dependency on earlier phases being re-runnable.

### Research Trade-offs
- Complete multi-method panels are much stronger for thesis defense.
- If the candidate pool is too small with Strategy A, Strategy C becomes necessary.

### FINAL DECISION
Use **Strategy A as primary**. Only select case studies from samples that have ALL required XAI artifacts (Grad-CAM, Attention, SHAP). LIME artifacts are optional (include in the combined figure if available, omit the LIME row if not). If Strategy A yields fewer than 10 total case studies across all types, escalate to Strategy C: identify the specific missing artifacts and request the user to re-run those phases on the candidate samples before finalizing Phase 6.

### Reason
Strategy A keeps Phase 6 as a pure consumer of previous phase outputs with no model inference. This is architecturally clean and avoids re-introducing model dependencies. The LIME-optional policy reflects that Phase 5 may not be completed before Phase 6.

---

## R5: Case Study Selection Bias

### Problem
Manually choosing examples that make the model look good is cherry-picking, which undermines scientific credibility. Even with algorithmic selection, there is a risk that criteria are tuned to favor impressive results.

### Why It Happens
- Researchers naturally gravitate toward examples that support their narrative.
- Error thresholds and sample counts are free parameters that can be adjusted to get desired results.
- Without explicit inclusion of failure cases, the portfolio appears biased.

### Strategy A: Fully Algorithmic Selection
Use deterministic criteria applied uniformly across the dataset. Document all thresholds, fallbacks, and de-duplication decisions in `selection_log.json`.

**Advantage:** Reproducible, transparent, auditable.
**Disadvantage:** May select cases that are not visually compelling or easy to explain.

### Strategy B: Algorithmic Selection with Manual Override
Algorithm selects candidates; researcher can swap individual cases.

**Advantage:** Balances scientific rigor with narrative quality.
**Disadvantage:** Manual overrides reintroduce bias. Must be documented transparently.

### Strategy C: Random Selection within Criteria
From all samples matching each criterion, select randomly (with a fixed seed).

**Advantage:** Eliminates any remaining selection bias.
**Disadvantage:** May miss the most illustrative examples.

### Engineering Trade-offs
- Strategy A is the strongest for reproducibility.
- The sorting criteria within each selection function (e.g., "sort by mean_error ascending") do introduce a preference, but it is explicit and documented.

### Research Trade-offs
- Thesis examiners respect algorithmic selection with documented criteria.
- Including BOTH success cases AND failure cases is the strongest defense against cherry-picking accusations.

### FINAL DECISION
Use **Strategy A**. All selection is algorithmic with documented criteria. The portfolio MUST include both correct-prediction cases AND high-error cases. The `selection_log.json` records every threshold, fallback, and de-duplication decision. No manual overrides. If a researcher wants to replace a specific case, they must add a manual override log entry in `selection_log.json` with justification.

### Reason
Full algorithmic selection with a balanced portfolio (success + failure + diverse case types) is the strongest defense against cherry-picking. The selection log provides full auditability for thesis examiners.

---

## R6: Combined Figure Visual Clarity at Thesis Print Scale

### Problem
A combined figure with multiple panels must remain readable when printed in a thesis at A4 size. Small text, thin lines, or low-contrast elements may become illegible.

### Why It Happens
- Figures designed on screen at 100% zoom may lose readability at print scale.
- Vietnamese text with diacritics requires slightly larger font sizes for legibility.
- Bar charts with many thin bars or small labels become unreadable at reduced print sizes.

### Strategy A: Large Figure Size with High DPI
Use 12x10 inches at 300 DPI, resulting in 3600x3000 pixel images.

**Advantage:** Plenty of space for all panels. Standard thesis figure quality.
**Disadvantage:** Large file sizes (potentially 5-15 MB per figure).

### Strategy B: Smaller Figures with Fewer Panels
Reduce to 8x6 inches, include only 2 panels instead of 4-6.

**Advantage:** Each panel gets more space. Smaller files.
**Disadvantage:** Loses the multi-method comparison in a single figure.

### FINAL DECISION
Use **Strategy A**. Generate at 12x10 inches, 300 DPI. Set minimum font size to 10pt for all text, 12pt for titles. Use high-contrast color palettes (not pastel). Verify readability by viewing the PNG at 50% zoom (simulating print reduction). The large file size is acceptable for thesis figures.

### Reason
The value of Phase 6 is the combined multi-method figure. Reducing panel count defeats the purpose. 300 DPI at 12x10 is standard for academic publications.

---

## R7: SHAP Contribution Data Format Dependency

### Problem
Phase 6 reads SHAP modality contribution data from Phase 4 artifacts. If Phase 4 stored this data in a different format than expected (different file naming, different JSON structure, or stored as NumPy arrays instead of JSON), Phase 6 will fail to load it.

### Why It Happens
- Phases are implemented independently by different coding sessions.
- File naming conventions may drift between phase implementations.
- Phase 4 may store SHAP values as raw NumPy arrays without pre-computed modality percentages.

### Strategy A: Define a Strict Contract
Specify exact file naming and JSON structure in both Phase 4 and Phase 6 proposals.

**Advantage:** No ambiguity.
**Disadvantage:** Requires Phase 4 to be (re)implemented to match this contract.

### Strategy B: Flexible Loader with Fallbacks
Phase 6's loader tries multiple file patterns and JSON structures. If modality percentages are not pre-computed, compute them from raw SHAP values.

**Advantage:** Robust to Phase 4 implementation variations.
**Disadvantage:** More complex loader code.

### FINAL DECISION
Use **Strategy B**. The `load_shap_contributions()` function should:
1. First, try to load `sample_{idx}_modality_contribution.json` (expected format).
2. If not found, try to load raw SHAP values from `sample_{idx}_shap_values.npy` and compute modality percentages using the known dimension split (image dims 0:1024, text dims 1024:1792 for Swin-B + PhoBERT, or read the dimension split from a config file).
3. If neither is found, mark the sample as having no SHAP data and exclude it from SHAP-dependent case types.

### Reason
Flexible loading is more robust and does not require retroactive changes to Phase 4. Computing modality percentages from raw SHAP values is straightforward given the known architecture dimensions.

---

## R8: Cross-Attention Fusion Complicates SHAP Dimension Splitting

### Problem
The best model uses `CrossAttentionFusion`, which projects text and image features into a shared `hidden=512` space before cross-attention and concatenation. This means the fusion vector is `[B, 1024]` (512 + 512) rather than `[B, 1792]` (768 + 1024) as in the Concat fusion. The SHAP modality split must use `0:512` for text-originated features and `512:1024` for image-originated features.

### Why It Happens
The `CrossAttentionFusion` architecture (in `Models/CrossAttentionFusion.py`) applies `text_proj` (768 -> 512) and `image_proj` (1024 -> 512) before cross-attention. The fused vector is `[t_out.squeeze(1), i_out.squeeze(1)]` = `[B, 1024]`.

### Strategy A: Phase 4 Handles the Dimension Split Based on Architecture
Phase 4's SHAP implementation should detect the fusion type and use the correct dimension split.

**Advantage:** Phase 6 receives pre-computed modality percentages regardless of architecture.
**Disadvantage:** Phase 4 must be architecture-aware.

### Strategy B: Phase 6 Reads the Architecture Config
Phase 6 reads the experiment's `config.yaml` to determine `fusion_type`, then uses the correct dimension split.

**Advantage:** Phase 6 can recompute modality percentages even if Phase 4 stored only raw SHAP values.
**Disadvantage:** Phase 6 needs architecture knowledge, which violates the "pure consumer" design.

### FINAL DECISION
Use **Strategy A as primary, Strategy B as fallback**. Phase 4 should store pre-computed modality percentages in the JSON, making Phase 6 architecture-agnostic. If Phase 4 only stored raw SHAP values, Phase 6 reads `config.yaml` to determine the fusion type and computes the split:
- `fusion_type == 'concat'`: text = `0:text_dim`, image = `text_dim:text_dim+image_dim`
- `fusion_type == 'cross_attention'`: text = `0:512`, image = `512:1024`
- Other fusion types: read from a fusion-type-to-split mapping table.

### Reason
This layered approach makes Phase 6 robust to different Phase 4 implementations while maintaining the preferred "pure consumer" design when possible.

---

# 13. Best Practices

### 13.1. Deterministic Selection
- All selection functions use deterministic sorting (no random sampling).
- When multiple samples have identical scores, break ties by sample index (lower index first) to ensure reproducibility.
- Record the random seed (if any randomness is introduced) in `selection_log.json`.

### 13.2. Artifact Naming Consistency
- Follow the naming convention established in Phases 2-5 exactly.
- Case study directory names use lowercase with underscores: `case_correct_001`, not `Case-Correct-1`.
- Figure filenames include both target index and target name for easy identification: `combined_figure_target0_food.png`.

### 13.3. Logging and Auditability
- `selection_log.json` records: timestamp, all thresholds (initial and final after relaxation), candidate counts before and after filtering, de-duplication decisions, and any warnings.
- Every function that applies a threshold should log whether the original or relaxed threshold was used.

### 13.4. Memory Management
- Close matplotlib figures immediately after saving (`plt.close(fig)` or `plt.close('all')`).
- Do not hold all case study images in memory simultaneously.
- Process case studies one at a time in the batch generation loop.

### 13.5. Figure Consistency
- Use consistent color palettes across all case studies:
  - Grad-CAM: `jet` or `turbo` colormap (matching Phase 2).
  - Attention bars: warm-to-cool gradient based on importance.
  - SHAP modality bars: fixed colors (e.g., blue for text, orange for image), used consistently across all case studies.
- Use consistent figure dimensions (12x10 inches) and font sizes (10pt minimum, 12pt titles).
- Use consistent axis limits for SHAP modality bars (0% to 100%).

### 13.6. Text Handling
- Truncate `comment_clean` to 200 characters in figure footers with "..." suffix.
- Do not truncate in `metadata.json` (store full text).
- Handle empty or NaN text gracefully (use placeholder: "[No review text available]").

### 13.7. Error Handling
- If a specific XAI artifact file is missing or corrupted, skip that panel in the combined figure and add a "Not Available" placeholder.
- Log all skipped panels in the selection log.
- Never let a single missing file crash the entire pipeline.

### 13.8. Configuration Management
- All thresholds, file paths, and parameters should be configurable via a dictionary or config file, not hardcoded.
- Default values should be reasonable (as specified in this proposal).
- The notebook should expose all key parameters in Cell 1 for easy adjustment.

### 13.9. Checkpoint Immutability
- Phase 6 NEVER modifies the model checkpoint or any Phase 2-5 artifacts.
- All outputs go exclusively into `experiments/EXP_XXX/xai/case_studies/`.

### 13.10. Thesis Integration Preparation
- Each combined figure should be self-contained: a thesis reader should understand the case from the figure alone.
- Analysis markdown should include suggested thesis captions for each figure.
- The summary document should include suggested section placement (Results chapter vs. Appendix).

---

# 14. Deliverables

### Per Case Study (10-17 cases)

| Deliverable | Format | Count per Case |
|---|---|---|
| Combined explanation figure | PNG (12x10 in, 300 DPI) | 2 per case (key target + overall) |
| Multi-target Grad-CAM comparison | PNG (20x4 in, 300 DPI) | 1 per case |
| Metadata JSON | JSON | 1 per case |
| Analysis markdown | Markdown | 1 per case |
| Individual XAI view copies | PNG, JSON, NPY | Variable (3-6 files) |

### Summary Deliverables

| Deliverable | Format | Count |
|---|---|---|
| Case study master index | CSV | 1 |
| Case study summary narrative | Markdown | 1 |
| Selection log | JSON | 1 |

### Source Code

| Deliverable | Format | Count |
|---|---|---|
| `xai/case_study_selector.py` | Python module | 1 |
| `xai/case_study_figure.py` | Python module | 1 |
| `xai/case_study_metadata.py` | Python module | 1 |
| `xai/case_study_analysis.py` | Python module | 1 |
| `xai/case_study_runner.py` | Python module | 1 |
| `xai/notebooks/Phase6_CaseStudy.ipynb` | Jupyter notebook | 1 |

### Approximate Total Artifact Count

- 10-17 case directories, each containing 5-9 files.
- Total files: approximately 80-150 files.
- Total disk usage: approximately 200-500 MB (primarily PNG figures).

---

# 15. Thesis Usage

### Results Chapter (3-5 Key Case Studies)

Select the 3-5 most compelling case studies for the main Results chapter:

1. **One correct-prediction case** demonstrating aligned multi-method evidence. Use the combined figure to show that Grad-CAM, attention, and SHAP all point to sensible reasoning. This is the "everything works" showcase.

2. **One high-error case** demonstrating failure analysis. Show what went wrong using XAI evidence. This demonstrates scientific honesty and the diagnostic value of XAI.

3. **One conflict case** demonstrating cross-modal disagreement. Show how the fusion module resolves contradictory image and text evidence. This is the most novel contribution for a multimodal XAI thesis.

4. **One text-dominant or image-dominant case** demonstrating modality imbalance. Show how SHAP quantifies the dominance and what it means for the prediction.

5. **One difficult case** (optional) demonstrating inference from incomplete evidence. Show how the model predicts aspects not mentioned in the text.

### Appendix

Include all remaining case studies (5-12 additional) in the Appendix with their combined figures and analysis text. This provides comprehensive evidence without overwhelming the main narrative.

### Discussion Chapter

Use specific case studies to support discussion points:

- **Conflict cases** support the discussion of "fusion module effectiveness" and "cross-modal reasoning."
- **High-error cases** support the discussion of "model limitations" and "failure modes."
- **Text-dominant / image-dominant cases** support the discussion of "modality balance" and "when each modality matters."
- **Difficult cases** support the discussion of "implicit reasoning" and "contextual inference."

### Defense Slides (2-3 Carefully Chosen Cases)

Select 2-3 cases that can be explained verbally in under 2 minutes each:

1. **The success story:** A correct prediction with clear, intuitive XAI evidence that the audience can immediately understand. Choose a sample with recognizable food imagery and clear sentiment text.

2. **The interesting failure:** A high-error or conflict case that reveals something about how the model works (or fails). This shows research depth.

3. **The modality story:** A case where image and text contribute differently, demonstrating the thesis's core contribution of multimodal explainability.

### Recommended Figure Captions

For the Results chapter, use captions of this form:

> **Figure X.** Multi-method explanation for a correct prediction case (Sample {id}). (a) Original restaurant image. (b) Grad-CAM overlay for {target}, highlighting {region description}. (c) Top-10 attention tokens from PhoBERT, showing emphasis on {key tokens}. (d) SHAP modality contribution showing {text_pct}% text and {image_pct}% image influence. The model correctly predicted {target} = {pred:.1f} against ground truth {true:.1f} (error = {error:.2f}).

For conflict cases:

> **Figure X.** Cross-modal conflict case (Sample {id}). Despite positive visual evidence from {image description}, the review text contains negative cues ("{negative text}"), creating conflicting signals for {target}. SHAP analysis reveals {text_pct}% text contribution with negative direction, while {image_pct}% image contribution is positive. The CrossAttention fusion resolved this by {resolution}, yielding a prediction of {pred:.1f} against ground truth {true:.1f}.

---

# 16. Phase Completion Checklist

### Pre-Requisites

- [ ] Phase 2 (Grad-CAM) artifacts exist for the target experiment
- [ ] Phase 3 (Attention) artifacts exist for the target experiment
- [ ] Phase 4 (SHAP) artifacts exist with modality contribution data
- [ ] `test_predictions.csv` or `predictions.csv` exists in the experiment directory
- [ ] Dataset CSV (`data/text/test.csv` or `data/text/val.csv`) is accessible
- [ ] Image directory (`data/image/`) contains cached images
- [ ] Vietnamese-capable font (Noto Sans or Arial) is installed on the system

### Implementation

- [ ] `xai/case_study_selector.py` implemented and tested
- [ ] `xai/case_study_figure.py` implemented and tested
- [ ] `xai/case_study_metadata.py` implemented and tested
- [ ] `xai/case_study_analysis.py` implemented and tested
- [ ] `xai/case_study_runner.py` implemented and tested
- [ ] Vietnamese font rendering verified with test figure

### Execution

- [ ] Case selection completed: at least 1 case per type, 10-17 total
- [ ] `case_study_index.csv` generated and verified
- [ ] `selection_log.json` generated with all threshold decisions
- [ ] All combined figures generated and visually inspected
- [ ] All metadata JSONs generated and cross-checked against prediction CSV
- [ ] All analysis markdown files generated and reviewed for coherence

### Validation

- [ ] Structural validation passed (all expected files exist)
- [ ] Metadata consistency validation passed (predictions match CSV data)
- [ ] Selection criteria validation passed (each case meets its type's criteria)
- [ ] Visual quality validation passed (Vietnamese renders, panels are readable)
- [ ] Reproducibility check passed (re-running produces identical selection)

### Thesis Readiness

- [ ] 3-5 key case studies selected for Results chapter
- [ ] Remaining cases assigned to Appendix
- [ ] Figure captions drafted for key cases
- [ ] Discussion chapter case references identified
- [ ] 2-3 defense slide cases selected
- [ ] `case_study_summary.md` reviewed and finalized

### Documentation

- [ ] All thresholds and criteria documented in `selection_log.json`
- [ ] Any fallback thresholds used are noted
- [ ] Any case types with zero samples are documented with explanation
- [ ] Notebook `Phase6_CaseStudy.ipynb` runs end-to-end without errors

---

*End of Phase 6 Implementation Proposal*
