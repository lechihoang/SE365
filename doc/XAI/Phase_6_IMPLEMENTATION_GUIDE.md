# Phase 6: Case Study Selection & Analysis -- Implementation Guide

This guide summarizes what changed from the original Phase 6 proposal and provides
the implementation-ready specification for the developer.

---

## 1. Major Changes from Original Proposal

| Area | Original Proposal | Updated Specification |
|---|---|---|
| LIME status | Optional (`include_lime=False` default) | First-class: Phase 5 is complete, LIME included in all combined figures |
| Combined figure size | 12x10 inches (3 rows, no LIME) | 12x14 inches (6 rows, includes LIME row) |
| XAI methods in figure | 3 (Grad-CAM, Attention, SHAP) | 4 (Grad-CAM, Attention, SHAP, LIME) |
| Sample selection | Simple threshold-based per case type | Multi-criteria ranking with `SelectionScore` formula |
| Sample manifest | Not specified | Required: CSV + JSON + preview Markdown |
| Sample display | Minimal table | Full inline display: images, text, predictions, case type rationale |
| Experiment ID | `EXP_XXX` placeholder | `EXP_060A_bestsequential_full_configuration` |
| Artifact paths | Old naming convention (`sample_{idx}_target{t}_{factor}_overlay.png`) | New per-sample subdirectory structure (`{phase}/{sample_id}/`) |
| Required phases | `['gradcam', 'attention', 'shap']` | `['gradcam', 'attention', 'shap', 'lime']` (LIME now required, graceful fallback if missing) |

---

## 2. Phase Outputs Consumed by Phase 6

All paths are relative to `{EXP_DIR}` = `experiments/EXP_060A_bestsequential_full_configuration`.

### Phase 2 -- Grad-CAM

Directory: `{EXP_DIR}/xai/gradcam/{sample_id}/`

| File | Description |
|---|---|
| `gradcam_img{k}_{factor}.png` | Per-image per-target heatmap overlay |
| `raw_cams.npz` | Raw CAM arrays |
| `gradcam_5target_comparison.png` | 5-target side-by-side comparison |
| `metadata.json` | Phase 2 run metadata |

### Phase 3 -- Attention

Directory: `{EXP_DIR}/xai/attention/{sample_id}/`

| File | Description |
|---|---|
| `attention_layer11_mean_heatmap.png` | Token-level attention heatmap (last layer) |
| `attention_last4layers_mean_heatmap.png` | Mean of last 4 layers |
| `cls_importance_subword_bar.png` | Subword-level CLS importance bar chart |
| `cls_importance_word_bar.png` | Word-level CLS importance bar chart |
| `raw_attention.npz` | Raw attention tensor `[12, 12, L, L]` |
| `tokens.json` | Token list |
| `word_importance.json` | Merged word importance scores |
| `topk_tokens.json` | Top-K important tokens |
| `metadata.json` | Phase 3 run metadata |

### Phase 4 -- SHAP

Directory: `{EXP_DIR}/xai/shap/{sample_id}/`

| File | Description |
|---|---|
| `shap_modality_contribution.png` | 5-target modality contribution chart |
| `shap_modality_contribution.json` | `text_pct` / `image_pct` per target |
| `raw_shap_values.npz` | Raw SHAP values per target |
| `metadata.json` | Phase 4 run metadata |

Aggregate file: `{EXP_DIR}/xai/shap/shap_aggregate_modality_contribution.json`

### Phase 5 -- LIME

Directory: `{EXP_DIR}/xai/lime/{sample_id}/`

| File | Description |
|---|---|
| `{sample_id}_lime_image_{factor}_positive.png` | Positive superpixel overlay |
| `{sample_id}_lime_image_{factor}_negative.png` | Negative superpixel overlay |
| `{sample_id}_lime_image_{factor}_combined.png` | Combined superpixel overlay |
| `{sample_id}_lime_image_{factor}_weights.json` | Superpixel weights |
| `{sample_id}_lime_text_{factor}_bar.png` | Text word importance bar chart |
| `{sample_id}_lime_text_{factor}_weights.json` | Text word weights |
| `{sample_id}_lime_text_{factor}.html` | Interactive LIME HTML explanation |
| `metadata.json` | Phase 5 run metadata |

---

## 3. New Phase 6 Outputs

All outputs go to `{EXP_DIR}/xai/case_studies/`.

| Output | Format | Description |
|---|---|---|
| `case_{type}_{seq:03d}/combined_figure_target{idx}_{factor}.png` | PNG 12x14 in, 300 DPI | Combined multi-method figure (all 4 XAI methods) |
| `case_study_index.csv` | CSV | Master list: `case_id`, `case_type`, `sample_id`, `split`, `mean_error`, `key_finding`, `figure_path` |
| `case_study_summary.md` | Markdown | Narrative summary across all case types |
| `case_{type}_{seq:03d}/metadata.json` | JSON | Per-case quantitative data |
| `case_{type}_{seq:03d}/analysis.md` | Markdown | Per-case structured analysis text |
| `selection_log.json` | JSON | All thresholds, fallbacks, de-duplication decisions |
| `sample_manifest.csv` | CSV | Selected sample manifest |
| `sample_manifest.json` | JSON | Same manifest in JSON format |
| `sample_manifest_preview.md` | Markdown | Human-readable manifest preview |

---

## 4. Combined Figure Layout

The combined figure now has 6 content rows to accommodate all 4 XAI methods.

```
+-------------------------------------------------------------+
| Title: Case ID | Sample ID | Target | True/Pred/Error       |
+-----------------------------+-------------------------------+
| Row 1: Original Image       | Row 1: Grad-CAM Overlay       |
|                              |        (for this target)       |
+-----------------------------+-------------------------------+
| Row 2: Attention             | Row 2: SHAP Modality           |
|   Word Importance Bar        |   Contribution Bar             |
+-----------------------------+-------------------------------+
| Row 3: LIME Image            | Row 3: LIME Text               |
|   (positive superpixels)     |   Word Importance Bar          |
+-----------------------------+-------------------------------+
| Footer: Review text (full) + all 5 predictions + ground truth|
+-------------------------------------------------------------+
```

**Specifications:**
- Figure size: `(12, 14)` inches -- 6 content rows with LIME
- DPI: 300 for thesis, 150 for drafts
- Minimum font size: 10pt for labels, 12pt for titles
- Vietnamese font: DejaVu Sans (primary), with Noto Sans / Arial as alternates
- Grad-CAM colormap: `jet` (from `COLOR_SCHEMES['gradcam_cmap']`)
- SHAP modality bars: text = `#1b9e77`, image = `#d95f02` (from `COLOR_SCHEMES['modality_colors']`)

---

## 5. Sample Selection Ranking Formula

Instead of simple threshold filtering, use a multiplicative ranking score to pick
the best candidates within each case type.

```
SelectionScore = PredictionQuality
               x VisualRichness
               x TextRichness
               x MultimodalBalance
               x ExplanationCompleteness
```

### Component Definitions

**PredictionQuality** -- varies by case type:
- For `correct` / `agreement`: higher quality = lower mean_error
- For `high_error`: higher quality = higher max_error (most illustrative failures)
- For other types: neutral (set to 1.0)

**VisualRichness:**
```python
VisualRichness = num_images * image_diversity_score
```
Where `image_diversity_score` reflects whether images show distinct content (default 1.0
if not computed; can be enhanced later with image embedding similarity).

**TextRichness:**
```python
TextRichness = min(1.0, text_length / 100) * aspect_keyword_coverage
```
Where `aspect_keyword_coverage` = fraction of the 5 aspects whose keywords appear in
the review text.

**MultimodalBalance:**
```python
MultimodalBalance = 1 - abs(text_pct - 0.50)
```
Closer to 50/50 SHAP split = higher score. Ranges from 0.5 (one modality 100%) to
1.0 (perfect balance). Used for `agreement` cases; set to 1.0 for dominance cases.

**ExplanationCompleteness:**
```python
ExplanationCompleteness = num_phases_with_artifacts / 4
```
Fraction of phases (Grad-CAM, Attention, SHAP, LIME) that have artifacts for this sample.
Values: 0.25, 0.50, 0.75, 1.0.

---

## 6. Sample Display Requirements

Every selected sample MUST immediately show the following in the notebook and manifest:

1. **Sample index** -- `sample_{idx:04d}` format
2. **Review text** -- full `comment_clean`, not truncated
3. **All images** -- displayed inline in the notebook (loaded from cache or URL)
4. **Prediction vs ground truth table** -- all 5 targets with true, predicted, and error
5. **Case type assignment** -- which case type this sample was selected for
6. **Selection rationale** -- why this sample was chosen (e.g., "lowest mean_error among correct predictions", "strongest text-image conflict on price target")

The sample manifest files (CSV, JSON, preview Markdown) must contain all of the above
in a format that can be reviewed before proceeding with figure generation.

---

## 7. Implementation Notes for the Developer

### Architecture

Phase 6 is a **pure consumer** -- no model inference, no gradient computation, no
checkpoint loading. It only reads artifacts from Phases 2-5 and assembles them.

### Shared Infrastructure

- `xai/utils.py` -- use `create_xai_metadata()` for consistent metadata generation
- `xai/config.py` -- all constants come from here:
  - `FACTOR_NAMES`, `DISPLAY_NAMES`, `FACTOR_TO_DISPLAY`
  - `FACTOR_TO_INDEX`, `INDEX_TO_FACTOR`
  - `COLOR_SCHEMES`, `DEFAULT_DPI`, `THESIS_DPI`
  - `BEST_EXP_ID` = `'EXP_060A_bestsequential_full_configuration'`

### Notebook Workflow

Follows the same pattern as Phases 1-5:
1. Clone repo / mount Drive (Colab)
2. Configure experiment paths
3. Load prediction data + dataset CSV
4. Run selection algorithm
5. Display manifest for review
6. Generate combined figures
7. Generate metadata + analysis
8. Validation checks

### Output Directory

```
{EXP_DIR}/xai/case_studies/
  +-- case_correct_001/
  +-- case_higherror_001/
  +-- case_conflict_001/
  +-- case_textdominant_001/
  +-- case_imagedominant_001/
  +-- case_difficult_001/
  +-- case_agreement_001/
  +-- case_study_index.csv
  +-- case_study_summary.md
  +-- selection_log.json
  +-- sample_manifest.csv
  +-- sample_manifest.json
  +-- sample_manifest_preview.md
```

### Missing Artifact Handling

Not all samples will have artifacts from all 4 phases. Handle gracefully:
- Check each phase directory before attempting to load
- If a phase's artifacts are missing, omit that row from the combined figure
- Adjust figure height: 12x14 (all 4 methods), 12x10 (3 methods, no LIME)
- Log missing artifacts in `selection_log.json`
- `ExplanationCompleteness` in the ranking formula naturally penalizes incomplete samples

### Memory Management

- Close matplotlib figures immediately after saving: `plt.close(fig)`
- Process one case study at a time in the batch loop
- Do not hold all images in memory simultaneously

---

## 8. Cross-Phase Consistency Checklist

These conventions MUST match across all phases. Any deviation will cause Phase 6 to
fail when loading artifacts.

| Convention | Value | Source |
|---|---|---|
| `sample_id` format | `sample_{idx:04d}` | All phases use 4-digit zero-padded index |
| Factor names (short) | `['food', 'price', 'atmos', 'service', 'overall']` | `xai/config.py` `FACTOR_NAMES` |
| Display names | `['Food Score', 'Price Score', 'Atmosphere Score', 'Service Score', 'Overall Satisfaction']` | `xai/config.py` `DISPLAY_NAMES` |
| DPI (draft) | 150 | `xai/config.py` `DEFAULT_DPI` |
| DPI (thesis) | 300 | `xai/config.py` `THESIS_DPI` |
| Grad-CAM colormap | `jet` | `COLOR_SCHEMES['gradcam_cmap']` |
| Attention colormap | `magma` | `COLOR_SCHEMES['attention_cmap']` |
| SHAP positive color | `#FF4444` | `COLOR_SCHEMES['shap_positive']` |
| SHAP negative color | `#4444FF` | `COLOR_SCHEMES['shap_negative']` |
| Modality text color | `#1b9e77` | `COLOR_SCHEMES['modality_colors']['text']` |
| Modality image color | `#d95f02` | `COLOR_SCHEMES['modality_colors']['image']` |
| Target colors | food=`#E53935`, price=`#43A047`, atmos=`#1E88E5`, service=`#FB8C00`, overall=`#8E24AA` | `COLOR_SCHEMES['target_colors']` |
| Score range | (1, 10) | `xai/config.py` `SCORE_RANGE` |
| Experiment ID | `EXP_060A_bestsequential_full_configuration` | `xai/config.py` `BEST_EXP_ID` |

---

## 9. Quick Reference: Case Types and Counts

| Case Type | Target Count | Selection Priority (de-dup) |
|---|---|---|
| `conflict` | 1-3 | 1 (highest) |
| `high_error` | 2-4 | 2 |
| `text_dominant` | 1-3 | 3 |
| `image_dominant` | 1-3 | 4 |
| `difficult` | 1-2 | 5 |
| `agreement` | 1-2 | 6 |
| `correct` | 2-4 | 7 (lowest) |
| **Total** | **10-21** | |

If a sample qualifies for multiple case types, assign it to the type with the
highest priority (lowest number). This preserves the rarest, most scientifically
interesting cases.

---

*End of Phase 6 Implementation Guide*
