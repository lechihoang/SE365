# Phase 7: XAI Report Generation — Implementation Proposal

---

## 1. Purpose

### Why this phase exists

Phases 2 through 6 of the XAI pipeline produce a large collection of artifacts: Grad-CAM heatmap PNGs, attention token heatmaps, SHAP value arrays, LIME superpixel masks, case study combined figures, raw numerical data in NPY and JSON formats, and CSV summaries. These artifacts are stored across multiple subdirectories under `experiments/EXP_XXX/xai/` and are individually useful, but they do not form a coherent, thesis-ready narrative by themselves.

Without automated report generation, the thesis author must manually:

- Locate every relevant figure across `gradcam/`, `attention/`, `shap/`, `lime/`, `case_studies/`, and `raw_values/` subdirectories.
- Manually compute aggregate statistics from raw numerical files.
- Manually construct comparison tables between XAI methods.
- Write description paragraphs for each figure and table.
- Ensure that every figure reference in the text actually corresponds to an existing file.
- Repeat the entire process whenever XAI is re-run with different samples, a different checkpoint, or updated parameters.

This phase exists to eliminate that manual burden and ensure that the full XAI analysis is presented consistently, completely, and reproducibly. Report generation is the bridge between raw XAI artifacts and thesis writing.

### Research motivation

A strong thesis defense requires not only correct individual XAI visualizations but also:

1. Aggregate statistics across samples (e.g., mean modality contribution across all analyzed samples).
2. Cross-method comparison (e.g., do Grad-CAM focus regions agree with LIME positive superpixels?).
3. Systematic case study presentation (correct predictions, high-error predictions, edge cases).
4. Method limitation discussion grounded in the actual results.

These are impossible to produce reliably through manual copy-paste workflows. An automated report generator guarantees that every thesis claim about XAI results is backed by verifiable, reproducible data.

### Engineering motivation

When any upstream XAI phase is re-executed (e.g., Grad-CAM is re-run with different sample indices, or SHAP is recomputed with a larger background set), the report must regenerate cleanly from the updated artifacts. Manual report maintenance would be error-prone and would violate reproducibility standards. An automated generator reads current artifacts and produces a fresh report each time.

---

## 2. Objectives

### Research objectives

1. Produce a comprehensive Markdown report (`xai_report.md`) that synthesizes findings from all completed XAI phases into a unified narrative.
2. Generate aggregate statistical summaries: modality contribution means, standard deviations, min/max across all analyzed samples, broken down by target (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`).
3. Produce cross-method comparison tables that systematically document agreement and disagreement between Grad-CAM, attention visualization, SHAP, and LIME for each case study sample.
4. Create thesis-ready text blocks with proper academic phrasing, correct citations (Selvaraju et al. 2017, Lundberg & Lee 2017, Ribeiro et al. 2016), and figure/table captions.
5. Present all case study analyses (correct predictions, high-error predictions, edge cases) with their combined figures and analytical commentary in a single coherent document.

### Engineering objectives

1. Implement a `ReportGenerator` class that scans XAI artifact directories, determines which phases completed successfully, and generates the report from available data.
2. Support graceful degradation: if LIME artifacts are missing, the report generates without the LIME section and explicitly notes "Phase 5 (LIME) not completed."
3. Ensure all figure references in the report correspond to existing files; never reference a PNG that does not exist.
4. Copy key figures into a `report_figures/` directory with thesis-quality descriptive naming.
5. Generate structured CSV and Markdown tables in `summary_tables/` for direct inclusion in LaTeX.
6. Include timestamps and artifact checksums for reproducibility tracking.

### Expected contributions

- A single command regenerates the entire XAI analysis report from current artifacts.
- The report serves as the first draft of the thesis XAI Results chapter.
- Summary tables can be directly converted to LaTeX `\begin{tabular}` blocks.
- The report structure matches standard thesis chapter organization: Methods, Results, Discussion, Limitations.

---

## 3. Inputs

### Artifact directories

All inputs are read from the experiment's XAI output directory. For the primary experiment (e.g., `EXP_060A`), the input structure is:

```
experiments/EXP_060A/xai/
├── gradcam/
│   ├── heatmap_sampleXXX_food.png
│   ├── heatmap_sampleXXX_price.png
│   ├── heatmap_sampleXXX_atmos.png
│   ├── heatmap_sampleXXX_service.png
│   ├── heatmap_sampleXXX_overall.png
│   ├── raw_cam_sampleXXX_food.npy
│   ├── raw_cam_sampleXXX_price.npy
│   ├── ...
│   └── gradcam_metadata.json
├── attention/
│   ├── token_heatmap_sampleXXX.png
│   ├── cls_importance_sampleXXX.json
│   ├── word_importance_sampleXXX.json
│   └── attention_metadata.json
├── shap/
│   ├── shap_values_sampleXXX_targetY.npy
│   ├── modality_contribution_summary.csv
│   ├── waterfall_sampleXXX_targetY.png
│   ├── bar_sampleXXX_targetY.png
│   └── shap_metadata.json
├── lime/                                    (may not exist)
│   ├── superpixel_sampleXXX_food.png
│   ├── word_weights_sampleXXX_food.json
│   └── lime_metadata.json
├── case_studies/
│   ├── combined_correct_sampleXXX.png
│   ├── combined_higherror_sampleXXX.png
│   ├── case_study_metadata.json
│   └── analysis_sampleXXX.txt
└── raw_values/
    ├── gradcam_summary.csv
    ├── attention_summary.csv
    ├── shap_raw_all.npz
    └── sample_metadata.json
```

### Experiment configuration

- `experiments/EXP_060A/config.yaml` or `config.json` — experiment hyperparameters (text_model_name, image_model_name, fusion_type, loss_fn, etc.).
- `experiments/EXP_060A/test_metrics.json` — test set performance metrics (MAE, RMSE, R2 per target).
- `experiments/EXP_060A/test_predictions.csv` — per-sample predictions and errors.

### Optional comparison experiment

- `experiments/EXP_012/` — baseline experiment (ConvNeXt + XLM-RoBERTa + Concat + MSE) for comparison if available.

### Model architecture constants

These are derived from the codebase and hardcoded in the report generator configuration:

| Constant | Value | Source |
|---|---|---|
| Factor names | `['food', 'price', 'atmos', 'service', 'overall']` | `test.py` line 125 |
| Display names | `['Food Score', 'Price Score', 'Atmosphere Score', 'Service Score', 'Overall Satisfaction']` | Thesis convention |
| Target indices | `0=food, 1=price, 2=atmos, 3=service, 4=overall` | Model output order |
| Text dim (PhoBERT) | 768 | `TextModel.encoder.config.hidden_size` |
| Image dim (Swin-B) | 1024 | `ImageModel.encoder.num_features` |
| Fusion dim (CrossAttention) | 1024 (hidden=512, fused=512*2) | `CrossAttentionFusion.py` line 60 |
| Concatenation order | text first, image second | `FusionModel.py` line 47 |

---

## 4. Outputs

### Primary outputs

| Output | Format | Description |
|---|---|---|
| `xai_report.md` | Markdown | Comprehensive XAI analysis report (~2000-4000 lines) |
| `thesis_text_blocks.md` | Markdown | Ready-to-paste thesis paragraphs with citations |
| `summary_tables/modality_contribution_by_target.csv` | CSV | Modality contribution statistics per target |
| `summary_tables/modality_contribution_by_target.md` | Markdown | Same data as Markdown table |
| `summary_tables/cross_method_agreement.csv` | CSV | Agreement matrix between XAI methods per sample |
| `summary_tables/case_study_summary.csv` | CSV | Summary row per case study sample |
| `summary_tables/xai_method_comparison.md` | Markdown | Method strengths and limitations table |
| `report_figures/fig_gradcam_5target_comparison.png` | PNG | Copy of representative Grad-CAM across 5 targets |
| `report_figures/fig_shap_modality_all_targets.png` | PNG | Copy of SHAP modality contribution chart |
| `report_figures/fig_attention_example.png` | PNG | Copy of representative attention heatmap |
| `report_figures/fig_case_study_correct_NNN.png` | PNG | Copies of case study combined figures |
| `report_figures/fig_case_study_higherror_NNN.png` | PNG | Copies of high-error case study figures |

### Output location

All outputs are written to:

```
experiments/EXP_XXX/xai/
├── xai_report.md
├── thesis_text_blocks.md
├── summary_tables/
│   ├── modality_contribution_by_target.csv
│   ├── modality_contribution_by_target.md
│   ├── cross_method_agreement.csv
│   ├── case_study_summary.csv
│   └── xai_method_comparison.md
└── report_figures/
    ├── fig_gradcam_5target_comparison.png
    ├── fig_shap_modality_all_targets.png
    ├── fig_attention_example.png
    ├── fig_case_study_correct_001.png
    ├── fig_case_study_higherror_001.png
    └── ...
```

---

## 5. Architecture Attachment Point

Phase 7 does not attach to the model architecture. It does not load the trained model, does not perform forward passes, and does not compute gradients.

Phase 7 is a **pure post-processing phase** that reads artifacts produced by Phases 2-6 and organizes them into structured documents and tables.

The attachment point is conceptual: the report generator must understand the model architecture to correctly describe it in the report text. Specifically, it must know:

- The best model is Swin-B + PhoBERT + CrossAttentionFusion + LogCosh.
- Grad-CAM attaches to the last spatial feature maps of Swin-B (`[B, 1024, 7, 7]`).
- Attention visualization extracts from PhoBERT's self-attention layers (`output_attentions=True`).
- SHAP operates on the fused embedding passed to the CrossAttentionFusion prediction head.
- LIME perturbs image superpixels and text words while holding the other modality fixed.

This architectural knowledge is encoded as configuration constants in the report generator, not derived dynamically from the model.

---

## 6. Detailed Implementation Plan

### A. Create `xai/report_generator.py`

This is the primary module for Phase 7. It contains the `ReportGenerator` class and all supporting functions.

#### A.1. `ReportGenerator.__init__(self, exp_dir, exp_id, config=None)`

**Parameters:**
- `exp_dir` (str): Path to the experiment root directory (e.g., `experiments/EXP_060A`).
- `exp_id` (str): Experiment identifier string (e.g., `"EXP_060A"`).
- `config` (dict, optional): Experiment configuration. If None, loaded from `config.yaml` or `config.json` in `exp_dir`.

**Initialization steps:**

1. Set `self.xai_dir = os.path.join(exp_dir, 'xai')`. Verify it exists; raise `FileNotFoundError` if not.
2. Set paths for all XAI subdirectories:
   - `self.gradcam_dir = os.path.join(self.xai_dir, 'gradcam')`
   - `self.attention_dir = os.path.join(self.xai_dir, 'attention')`
   - `self.shap_dir = os.path.join(self.xai_dir, 'shap')`
   - `self.lime_dir = os.path.join(self.xai_dir, 'lime')`
   - `self.case_studies_dir = os.path.join(self.xai_dir, 'case_studies')`
   - `self.raw_values_dir = os.path.join(self.xai_dir, 'raw_values')`
3. Scan each subdirectory for existence and artifact count. Store results in `self.available_phases` dict:
   ```
   {
       'gradcam': True/False,
       'attention': True/False,
       'shap': True/False,
       'lime': True/False,
       'case_studies': True/False,
   }
   ```
4. Load experiment config from `exp_dir/config.yaml` or `exp_dir/config.json`. Extract:
   - `self.text_model_name` (e.g., `"vinai/phobert-base-v2"`)
   - `self.image_model_name` (e.g., `"swin_base_patch4_window7_224"`)
   - `self.fusion_type` (e.g., `"cross_attention"`)
   - `self.loss_fn` (e.g., `"logcosh"`)
5. Load test metrics from `exp_dir/test_metrics.json` if available. Store as `self.test_metrics`.
6. Set output paths:
   - `self.report_path = os.path.join(self.xai_dir, 'xai_report.md')`
   - `self.thesis_blocks_path = os.path.join(self.xai_dir, 'thesis_text_blocks.md')`
   - `self.tables_dir = os.path.join(self.xai_dir, 'summary_tables')`
   - `self.figures_dir = os.path.join(self.xai_dir, 'report_figures')`
7. Create output directories: `os.makedirs(self.tables_dir, exist_ok=True)`, same for `self.figures_dir`.
8. Set architecture constants:
   ```python
   self.factor_names = ['food', 'price', 'atmos', 'service', 'overall']
   self.display_names = ['Food Score', 'Price Score', 'Atmosphere Score',
                         'Service Score', 'Overall Satisfaction']
   self.num_targets = 5
   ```
9. Log initialization summary: which phases are available, experiment config, artifact counts.

#### A.2. `ReportGenerator.generate_full_report(self)`

**Returns:** Path to the generated `xai_report.md`.

**Steps:**

1. Initialize `report_lines = []` (list of strings).
2. Call each section generator in order, appending returned lines:
   - `self._generate_header()` — report title, generation timestamp, experiment info.
   - `self._generate_executive_summary()` — model config, best metrics, available XAI methods.
   - `self._generate_gradcam_section()` — if `self.available_phases['gradcam']`.
   - `self._generate_attention_section()` — if `self.available_phases['attention']`.
   - `self._generate_shap_section()` — if `self.available_phases['shap']`.
   - `self._generate_lime_section()` — if `self.available_phases['lime']`.
   - `self._generate_cross_method_comparison()` — if at least 2 methods available.
   - `self._generate_case_studies_section()` — if `self.available_phases['case_studies']`.
   - `self._generate_limitations_section()` — always generated.
3. Join all lines with newlines and write to `self.report_path`.
4. Call `self._generate_summary_tables()`.
5. Call `self._copy_report_figures()`.
6. Call `self._generate_thesis_text_blocks()`.
7. Return `self.report_path`.

#### A.3. Section generators — detailed specifications

##### `_generate_header(self) -> list[str]`

Produce:
```markdown
# XAI Analysis Report — {exp_id}

**Generated:** {ISO 8601 timestamp}
**Experiment:** {exp_id}
**Model:** {image_model_name} + {text_model_name} + {fusion_type} + {loss_fn}
**Report Generator Version:** 1.0

---
```

##### `_generate_executive_summary(self) -> list[str]`

Content:
1. Model architecture description (one paragraph): image encoder, text encoder, fusion method, loss function. Reference `CrossAttentionFusion.py` architecture: text and image features are projected to 512-dim hidden space, bidirectional cross-attention is applied, and the concatenated cross-attended outputs (1024-dim) pass through the MLP head.
2. Test set performance table (from `test_metrics.json`):
   ```
   | Target | MAE | RMSE | R2 |
   |---|---:|---:|---:|
   | Food | ... | ... | ... |
   ```
3. XAI methods applied: list which phases completed with checkmark notation.
4. Number of samples analyzed per method.
5. Mean modality contribution across all samples (if SHAP data available): e.g., "On average, image features contributed 58.3% and text features 41.7% to the overall satisfaction prediction."

##### `_generate_gradcam_section(self) -> list[str]`

1. Section header: `## Grad-CAM Analysis (Phase 2)`.
2. Method description paragraph: Grad-CAM applied to the final spatial feature maps of Swin-B before global pooling. Heatmaps generated for each of the 5 target scores. Cite Selvaraju et al. (2017).
3. For each analyzed sample (read from `gradcam_metadata.json`):
   - Embed the 5-target heatmap grid figure if a combined PNG exists, or embed individual heatmaps.
   - Include the sample's review text (quoted, in Vietnamese with annotation).
   - Include ground truth and predicted scores.
   - Note which image regions were highlighted for each target.
4. Domain consistency assessment paragraph: summarize whether food-related targets focused on food regions, atmosphere-related targets focused on interior/environment regions, etc. Derive this from metadata or analysis text files.
5. If more than 5 samples analyzed, show a representative subset and note the total count.

**Graceful degradation:** If `gradcam/` directory does not exist or contains no PNGs, emit:
```markdown
## Grad-CAM Analysis (Phase 2)

*Phase 2 (Grad-CAM) has not been completed. No Grad-CAM artifacts found.*
```

##### `_generate_attention_section(self) -> list[str]`

1. Section header: `## Attention Analysis (Phase 3)`.
2. Method description: attention weights extracted from PhoBERT with `output_attentions=True`. Aggregation strategy used (e.g., last-layer mean over heads). Cite transformer attention literature.
3. For each analyzed sample:
   - Embed token heatmap PNG.
   - List top-5 most attended tokens (from `word_importance_sampleXXX.json`).
   - Note CLS attention distribution (from `cls_importance_sampleXXX.json`).
4. Aggregated findings: common high-attention token patterns across samples (e.g., food-related tokens, price-related tokens, sentiment modifiers).

**Graceful degradation:** Same pattern — note "Phase 3 (Attention) not completed" if absent.

##### `_generate_shap_section(self) -> list[str]`

1. Section header: `## SHAP Analysis (Phase 4)`.
2. Method description: SHAP applied to the fused embedding and prediction head. DeepExplainer or KernelExplainer used. Background set description. Cite Lundberg & Lee (2017).
3. Modality contribution table (from `modality_contribution_summary.csv`):
   ```
   | Target | Text % (mean) | Text % (std) | Image % (mean) | Image % (std) | N_samples |
   |---|---:|---:|---:|---:|---:|
   | Food Score | ... | ... | ... | ... | ... |
   | Price Score | ... | ... | ... | ... | ... |
   ```
4. Include min and max percentages to show the range of variation.
5. For representative samples, embed waterfall plots and bar plots.
6. Per-target analysis: which targets are more image-driven vs. text-driven? Is `price_score` more text-driven than `food_score`? These are key thesis findings.

**Graceful degradation:** If absent, note "Phase 4 (SHAP) not completed."

##### `_generate_lime_section(self) -> list[str]`

1. Section header: `## LIME Analysis (Phase 5)`.
2. Method description: LIME image applied with text fixed; LIME text applied with image fixed. Cite Ribeiro et al. (2016).
3. For each analyzed sample:
   - Embed superpixel mask PNGs.
   - List top positive and negative word weights.
4. Cross-validation with other methods: do LIME positive superpixels overlap with Grad-CAM hot regions? Do LIME top words align with attention top tokens?

**Graceful degradation:** If absent, note "Phase 5 (LIME) not completed. LIME analysis was not performed for this experiment."

##### `_generate_cross_method_comparison(self) -> list[str]`

1. Section header: `## Cross-Method Comparison`.
2. For each case study sample where multiple methods produced results, create a comparison entry:
   - **Image evidence agreement:** Does Grad-CAM focus region overlap with LIME positive superpixels? Qualitative assessment: "strong agreement", "partial agreement", "disagreement".
   - **Text evidence agreement:** Do attention top tokens match LIME top words? Qualitative assessment.
   - **Modality dominance consistency:** Does SHAP modality contribution align with the qualitative impression from Grad-CAM (strong image focus) vs. attention (strong text evidence)?
3. Produce a summary agreement table:
   ```
   | Sample | GradCAM-LIME Image | Attention-LIME Text | SHAP Dominance | Overall |
   |---|---|---|---|---|
   | sample_042 | agree | agree | image-dominant | consistent |
   | sample_187 | partial | agree | text-dominant | partially consistent |
   ```
4. Discussion paragraph: what cross-method consistency tells us about model reliability.

**Graceful degradation:** Requires at least 2 methods completed. If only 1 method available, emit: "Cross-method comparison requires at least two completed XAI phases. Only {method_name} is currently available."

##### `_generate_case_studies_section(self) -> list[str]`

1. Section header: `## Case Studies (Phase 6)`.
2. For each case study (read from `case_study_metadata.json`):
   - Embed the combined figure (`combined_correct_sampleXXX.png` or `combined_higherror_sampleXXX.png`).
   - Include the full analysis text from `analysis_sampleXXX.txt`.
   - Include the sample's review text, ground truth, prediction, and error.
   - Tag the case study type: "correct prediction", "high-error prediction", "edge case".
3. Summary paragraph synthesizing patterns across case studies.

**Graceful degradation:** If `case_studies/` directory is empty or missing, note "Phase 6 (Case Studies) not completed."

##### `_generate_limitations_section(self) -> list[str]`

Always generated. Content:

1. Per-method limitations:
   - Grad-CAM: coarse resolution (7x7 upsampled to 224x224), not causal, sensitive to target layer choice.
   - Attention: shows interaction, not guaranteed causal importance (Jain & Wallace, 2019).
   - SHAP: embedding dimensions are not human-semantic; background set choice affects results.
   - LIME: local only, sensitive to perturbation sampling, computationally expensive.
2. Cross-method limitations: methods explain different objects (spatial maps, token interactions, fused features, perturbation responses), so disagreement does not necessarily indicate error.
3. Scope limitation: XAI was applied to a selected subset of samples, not the entire dataset.
4. Vietnamese text specificity: subword tokenization of Vietnamese text requires careful aggregation for human-readable presentation.

#### A.4. `_generate_summary_tables(self)`

Creates CSV and Markdown files in `self.tables_dir`.

##### Table 1: Modality contribution by target

**Source:** Read `shap/modality_contribution_summary.csv`.

**Process:**
1. Load the CSV.
2. Group by target (0-4).
3. Compute per-target statistics: mean, std, min, max for both text_pct and image_pct.
4. Write `modality_contribution_by_target.csv` with columns: `target, target_display_name, text_pct_mean, text_pct_std, text_pct_min, text_pct_max, image_pct_mean, image_pct_std, image_pct_min, image_pct_max, n_samples`.
5. Write `modality_contribution_by_target.md` as a formatted Markdown table.

**Graceful degradation:** If SHAP summary CSV not found, skip this table and log a warning.

##### Table 2: Cross-method agreement

**Source:** Cross-reference artifacts from `gradcam/`, `attention/`, `shap/`, `lime/`, and `case_studies/`.

**Process:**
1. For each sample that appears in case study metadata:
   - Check if Grad-CAM artifacts exist for that sample.
   - Check if attention artifacts exist.
   - Check if SHAP artifacts exist.
   - Check if LIME artifacts exist.
2. For samples with multiple methods, read the analysis text and metadata to extract agreement qualifications.
3. If automated agreement scoring was performed in Phase 6, read those scores.
4. If no automated scoring exists, populate the agreement column with "manual review needed".
5. Write `cross_method_agreement.csv`.

##### Table 3: Case study summary

**Source:** `case_studies/case_study_metadata.json`.

**Process:**
1. Read metadata for all case study samples.
2. Create one row per sample with columns: `sample_id, case_type, review_text_preview (first 80 chars), ground_truth (5 scores), prediction (5 scores), mean_absolute_error, dominant_modality (from SHAP if available), gradcam_available, attention_available, shap_available, lime_available`.
3. Write `case_study_summary.csv`.

##### Table 4: XAI method comparison

**Process:** Generate a static comparison table (content is architecture-specific, not data-dependent).

```markdown
| Method | Explains | Attachment Point | Resolution | Causal Strength | Key Citation |
|---|---|---|---|---|---|
| Grad-CAM | Image regions | Swin-B last spatial maps [B,1024,7,7] | Coarse (7x7 -> 224x224) | Moderate (gradient-linked) | Selvaraju et al. 2017 |
| Attention | Token interactions | PhoBERT self-attention [B,12,L,L] | Token-level | Weak (not causal) | Jain & Wallace 2019 |
| SHAP | Feature contributions | Fusion head input | Feature-level (grouped by modality) | Strong (Shapley axioms) | Lundberg & Lee 2017 |
| LIME | Local perturbation response | Full model (superpixels/words) | Superpixel/word-level | Moderate (local fidelity) | Ribeiro et al. 2016 |
```

Write to `xai_method_comparison.md`.

#### A.5. `_copy_report_figures(self)`

Copies key figures from their original artifact directories into `report_figures/` with thesis-quality descriptive names.

**Mapping rules:**

1. If a 5-target Grad-CAM comparison grid exists for the first case study sample, copy it as `fig_gradcam_5target_comparison.png`.
2. If individual Grad-CAM heatmaps exist but no grid, create a note in the report that individual files should be referenced.
3. Copy the first SHAP modality contribution chart as `fig_shap_modality_all_targets.png`.
4. Copy the first attention heatmap as `fig_attention_example.png`.
5. Copy all case study combined figures as `fig_case_study_{type}_{NNN}.png` where `type` is `correct` or `higherror` and `NNN` is a zero-padded sequential number.
6. Copy LIME superpixel examples if available.

**Implementation detail:** Use `shutil.copy2()` to preserve timestamps. Log each copy operation.

**Naming convention:** All report figure filenames use the prefix `fig_` followed by the method name, followed by a descriptive suffix. No numerical figure numbers — the thesis LaTeX will assign those.

#### A.6. `_generate_thesis_text_blocks(self)`

Generates `thesis_text_blocks.md` containing ready-to-paste paragraphs organized by thesis section.

##### Block 1: Method description — Grad-CAM

```
To localize visually salient regions associated with each predicted quality score,
Gradient-weighted Class Activation Mapping (Grad-CAM) [Selvaraju et al., 2017] was
applied to the final spatial feature maps of the Swin-B image encoder. For a selected
target output y^t, the gradients of the scalar output with respect to the feature maps
of shape [B, 1024, 7, 7] were globally averaged to obtain channel importance weights.
The weighted combination was passed through a ReLU activation and upsampled to the
original image resolution (224x224) to produce a class activation heatmap. Separate
heatmaps were generated for each of the five target scores, enabling comparison of
visual evidence across food quality, price, atmosphere, service, and overall
satisfaction predictions.
```

##### Block 2: Method description — Attention

```
Attention weights were extracted from the PhoBERT text encoder by enabling
output_attentions=True during inference. For each input review, the self-attention
matrices from the last transformer layer were averaged across all 12 attention heads
to produce a token-level attention heatmap. This aggregation strategy (last-layer
mean over heads) balances stability with task relevance, as later layers in BERT-family
models tend to capture more task-specific information. Token-level importance was
derived from the attention weights directed toward the [CLS] token, and subword tokens
were aggregated back to word level for human-facing visualization.
```

##### Block 3: Method description — SHAP

```
SHAP (SHapley Additive exPlanations) [Lundberg and Lee, 2017] was applied to the
prediction head of the CrossAttentionFusion model to quantify feature-level and
modality-level contributions to each predicted score. The fused representation, formed
by concatenating the cross-attended text and image outputs (dimension 1024), served as
the input to the SHAP explainer. Modality-level contributions were computed by
aggregating the absolute SHAP values within each modality's dimensional segment.
This analysis directly addresses the research question of whether the model's
predictions are primarily driven by visual or textual evidence.
```

##### Block 4: Method description — LIME

```
LIME (Local Interpretable Model-agnostic Explanations) [Ribeiro et al., 2016] was
used as a local perturbation-based validation method for both image and text modalities.
For image LIME, the input image was segmented into superpixels, and subsets of segments
were randomly masked while the text input was held fixed. For text LIME, subsets of words
were removed while the image was held fixed. In both cases, a sparse local linear
surrogate was fitted to approximate the model's local response, yielding per-superpixel
and per-word importance scores.
```

##### Block 5: Results description — Modality contribution

Template (populated with actual data if SHAP summary available):

```
SHAP-based modality contribution analysis revealed that, on average, image features
accounted for {image_mean:.1f}% (SD = {image_std:.1f}%) of the total absolute SHAP
magnitude for the overall satisfaction prediction, while text features contributed
{text_mean:.1f}% (SD = {text_std:.1f}%). This pattern was {consistent/variable}
across the five target scores: {per-target summary}. Notably, the price score showed
the highest text contribution ({price_text_mean:.1f}%), consistent with the expectation
that price judgments are more explicitly expressed in review text.
```

##### Block 6: Discussion paragraph — Cross-method consistency

Template:

```
The multi-level explainability framework produced largely consistent evidence across
methods. Grad-CAM highlighted food presentation and restaurant interior regions that
aligned with the predictions' target scores, while attention analysis identified
aspect-bearing tokens (e.g., food quality terms for food_score, price-related phrases
for price_score). SHAP provided quantitative confirmation of modality dominance, and
LIME perturbation experiments validated that removing highlighted regions or words
produced expected changes in predictions. In {N} of {total} case study samples, all
four methods produced consistent explanations. In {M} cases, partial disagreement
was observed, primarily between Grad-CAM spatial localization and LIME superpixel
boundaries, which is expected given the methods' different spatial granularity.
```

##### Block 7: Figure captions

Generate a caption for each report figure, following academic convention:

```
**Figure {placeholder}.** Grad-CAM visualization for five target scores on a
representative sample. Warm colors indicate image regions with positive gradient-weighted
activation. The food_score heatmap focused on {observed_region}, while the
atmosphere_score heatmap highlighted {observed_region}.
```

All captions use `{placeholder}` for figure numbers since LaTeX will assign them.

#### A.7. Graceful degradation implementation

Every section generator follows this pattern:

```
def _generate_X_section(self):
    lines = []
    if not self.available_phases['X']:
        lines.append(f'## {section_title}')
        lines.append('')
        lines.append(f'*Phase {phase_number} ({method_name}) has not been completed. '
                     f'No {method_name} artifacts found in {self.X_dir}.*')
        lines.append('')
        return lines
    # ... normal generation ...
```

Rules:
- Never raise an exception for missing artifacts.
- Always check `os.path.exists()` before reading any file.
- If a metadata JSON is corrupted or missing keys, log a warning and use default values.
- If a PNG referenced in metadata does not exist on disk, omit it from the report and add a note: `*(Figure not found: {filename})*`.
- If a CSV has no data rows, note "No data available" in the corresponding table.

### B. Create `xai/notebooks/Phase7_Report.ipynb`

This notebook provides an interactive interface for report generation and allows manual inspection of intermediate results before generating the final report.

See Section 9 (Notebook Design) for cell-by-cell specification.

---

## 7. Required Code Files

### New files

| File | Responsibility |
|---|---|
| `xai/report_generator.py` | Main module. Contains `ReportGenerator` class with all section generators, table generators, figure copier, and thesis text block generator. |
| `xai/notebooks/Phase7_Report.ipynb` | Interactive notebook for report generation. Loads the `ReportGenerator`, displays intermediate data, and generates the full report. |

### Files read but not modified

| File | How used |
|---|---|
| `experiments/EXP_XXX/config.yaml` or `config.json` | Read experiment configuration |
| `experiments/EXP_XXX/test_metrics.json` | Read test performance metrics |
| `experiments/EXP_XXX/test_predictions.csv` | Read per-sample predictions for case study metadata |
| `experiments/EXP_XXX/xai/gradcam/gradcam_metadata.json` | Read Grad-CAM artifact metadata |
| `experiments/EXP_XXX/xai/attention/attention_metadata.json` | Read attention artifact metadata |
| `experiments/EXP_XXX/xai/shap/shap_metadata.json` | Read SHAP artifact metadata |
| `experiments/EXP_XXX/xai/shap/modality_contribution_summary.csv` | Read modality contribution data |
| `experiments/EXP_XXX/xai/lime/lime_metadata.json` | Read LIME artifact metadata |
| `experiments/EXP_XXX/xai/case_studies/case_study_metadata.json` | Read case study metadata |
| `experiments/EXP_XXX/xai/case_studies/analysis_sampleXXX.txt` | Read per-sample analysis text |
| `experiments/EXP_XXX/xai/raw_values/sample_metadata.json` | Read raw value metadata |

### Module dependencies

```python
# Standard library
import os
import json
import csv
import shutil
import datetime
import hashlib
import logging
from pathlib import Path
from collections import defaultdict

# Third-party
import numpy as np
import pandas as pd
import yaml  # for config.yaml reading
```

No PyTorch, no model imports, no GPU requirements. This module runs on CPU only.

---

## 8. Folder Structure

### Complete output structure after Phase 7

```
experiments/EXP_XXX/xai/
├── gradcam/                                    # Phase 2 artifacts (input)
│   ├── heatmap_sampleXXX_food.png
│   ├── ...
│   └── gradcam_metadata.json
├── attention/                                  # Phase 3 artifacts (input)
│   ├── token_heatmap_sampleXXX.png
│   ├── ...
│   └── attention_metadata.json
├── shap/                                       # Phase 4 artifacts (input)
│   ├── shap_values_sampleXXX_targetY.npy
│   ├── modality_contribution_summary.csv
│   ├── ...
│   └── shap_metadata.json
├── lime/                                       # Phase 5 artifacts (input, optional)
│   ├── superpixel_sampleXXX_food.png
│   ├── ...
│   └── lime_metadata.json
├── case_studies/                                # Phase 6 artifacts (input)
│   ├── combined_correct_sampleXXX.png
│   ├── combined_higherror_sampleXXX.png
│   ├── case_study_metadata.json
│   └── analysis_sampleXXX.txt
├── raw_values/                                  # Phases 2-6 raw data (input)
│   ├── gradcam_summary.csv
│   ├── attention_summary.csv
│   ├── shap_raw_all.npz
│   └── sample_metadata.json
├── xai_report.md                               # Phase 7 output: full report
├── thesis_text_blocks.md                        # Phase 7 output: thesis paragraphs
├── summary_tables/                              # Phase 7 output
│   ├── modality_contribution_by_target.csv
│   ├── modality_contribution_by_target.md
│   ├── cross_method_agreement.csv
│   ├── case_study_summary.csv
│   └── xai_method_comparison.md
└── report_figures/                              # Phase 7 output
    ├── fig_gradcam_5target_comparison.png
    ├── fig_shap_modality_all_targets.png
    ├── fig_attention_example.png
    ├── fig_case_study_correct_001.png
    ├── fig_case_study_higherror_001.png
    └── ...
```

### Naming conventions

- Phase 7 outputs use the prefix `fig_` for copied report figures.
- Summary tables use descriptive snake_case names.
- No numerical prefixes on files (thesis assigns figure/table numbers).
- All Phase 7 outputs are in `xai_report.md`, `thesis_text_blocks.md`, `summary_tables/`, and `report_figures/`. Phase 7 never modifies files in `gradcam/`, `attention/`, `shap/`, `lime/`, `case_studies/`, or `raw_values/`.

---

## 9. Notebook Design

### `xai/notebooks/Phase7_Report.ipynb`

#### Cell 1: Markdown — Title and description

```markdown
# Phase 7: XAI Report Generation

Generate a comprehensive XAI analysis report from all available artifacts.

**Experiment:** EXP_060A (Swin-B + PhoBERT + CrossAttention + LogCosh)
**Prerequisites:** At least one XAI phase (2-6) must be completed.
```

#### Cell 2: Setup and imports

```python
import os
import sys
import json

# Paths
PROJECT_ROOT = '/content/SE365'
DRIVE_ROOT = '/content/drive/MyDrive/SE365'
EXP_ID = 'EXP_060A_bestsequential_full_configuration'
EXP_DIR = f'{DRIVE_ROOT}/experiments/{EXP_ID}'

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
```

**Expected output:** Path confirmation, no errors.

#### Cell 3: Scan available XAI artifacts

```python
from xai.report_generator import ReportGenerator

generator = ReportGenerator(exp_dir=EXP_DIR, exp_id=EXP_ID)
print("Available XAI phases:")
for phase, available in generator.available_phases.items():
    status = "COMPLETED" if available else "NOT FOUND"
    print(f"  {phase:<15s}: {status}")
```

**Expected output:** A summary of which phases have artifacts available. At minimum, Grad-CAM and SHAP should show as completed.

#### Cell 4: Markdown — Inspect experiment configuration

```markdown
## Experiment Configuration
```

#### Cell 5: Display experiment config and test metrics

```python
print(f"Text Model:  {generator.text_model_name}")
print(f"Image Model: {generator.image_model_name}")
print(f"Fusion:      {generator.fusion_type}")
print(f"Loss:        {generator.loss_fn}")

if generator.test_metrics:
    print("\nTest Set Performance:")
    for key, val in generator.test_metrics.items():
        print(f"  {key}: {val:.4f}")
```

**Expected output:** Full configuration display and test metrics table.

#### Cell 6: Markdown — Preview modality contribution data

```markdown
## Preview: Modality Contribution Data (SHAP)
```

#### Cell 7: Preview SHAP modality contribution

```python
import pandas as pd

shap_csv = os.path.join(generator.shap_dir, 'modality_contribution_summary.csv')
if os.path.isfile(shap_csv):
    df_modality = pd.read_csv(shap_csv)
    print(f"Modality contribution data: {len(df_modality)} rows")
    display(df_modality.describe())
    
    # Preview per-target averages
    for target_idx, target_name in enumerate(generator.display_names):
        subset = df_modality[df_modality['target_index'] == target_idx]
        if len(subset) > 0:
            print(f"\n{target_name}:")
            print(f"  Image %: {subset['image_pct'].mean():.1f} +/- {subset['image_pct'].std():.1f}")
            print(f"  Text  %: {subset['text_pct'].mean():.1f} +/- {subset['text_pct'].std():.1f}")
else:
    print("SHAP modality contribution data not available.")
```

**Expected output:** Per-target modality contribution statistics.

#### Cell 8: Markdown — Preview case studies

```markdown
## Preview: Case Studies
```

#### Cell 9: Preview case study metadata

```python
cs_meta_path = os.path.join(generator.case_studies_dir, 'case_study_metadata.json')
if os.path.isfile(cs_meta_path):
    with open(cs_meta_path, 'r', encoding='utf-8') as f:
        cs_metadata = json.load(f)
    print(f"Case studies found: {len(cs_metadata.get('samples', []))}")
    for sample in cs_metadata.get('samples', []):
        print(f"  Sample {sample['sample_id']}: {sample['case_type']} "
              f"(MAE: {sample.get('mean_absolute_error', 'N/A')})")
else:
    print("Case study metadata not found.")
```

**Expected output:** List of case study samples with types and error values.

#### Cell 10: Markdown — Generate report

```markdown
## Generate Full Report
```

#### Cell 11: Generate the full report

```python
report_path = generator.generate_full_report()
print(f"\nReport generated: {report_path}")

# Print artifact summary
print(f"\nGenerated artifacts:")
for root, dirs, files in os.walk(generator.xai_dir):
    level = root.replace(generator.xai_dir, '').count(os.sep)
    indent = '  ' * level
    basename = os.path.basename(root)
    if level == 0:
        basename = 'xai/'
    print(f'{indent}{basename}/')
    for f in sorted(files):
        if root.endswith('summary_tables') or root.endswith('report_figures') or f in ['xai_report.md', 'thesis_text_blocks.md']:
            print(f'{indent}  {f}')
```

**Expected output:** Report path and list of generated artifacts.

#### Cell 12: Markdown — Preview generated report

```markdown
## Preview: Generated Report (first 200 lines)
```

#### Cell 13: Preview report content

```python
with open(report_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total report lines: {len(lines)}")
print(f"\n{'='*60}")
for line in lines[:200]:
    print(line, end='')
```

**Expected output:** First 200 lines of the generated report.

#### Cell 14: Markdown — Preview thesis text blocks

```markdown
## Preview: Thesis Text Blocks
```

#### Cell 15: Preview thesis text blocks

```python
thesis_path = generator.thesis_blocks_path
if os.path.isfile(thesis_path):
    with open(thesis_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(content[:3000])
else:
    print("Thesis text blocks not generated.")
```

**Expected output:** First 3000 characters of thesis text blocks.

#### Cell 16: Markdown — Preview summary tables

```markdown
## Preview: Summary Tables
```

#### Cell 17: Display summary tables

```python
tables_dir = generator.tables_dir

for table_file in sorted(os.listdir(tables_dir)):
    filepath = os.path.join(tables_dir, table_file)
    print(f"\n{'='*60}")
    print(f"TABLE: {table_file}")
    print(f"{'='*60}")
    
    if table_file.endswith('.csv'):
        df = pd.read_csv(filepath)
        display(df)
    elif table_file.endswith('.md'):
        with open(filepath, 'r', encoding='utf-8') as f:
            print(f.read())
```

**Expected output:** All summary tables rendered in the notebook.

#### Cell 18: Markdown — Report figures inventory

```markdown
## Report Figures Inventory
```

#### Cell 19: List and display report figures

```python
from IPython.display import Image, display as ipy_display

figures_dir = generator.figures_dir
fig_files = sorted([f for f in os.listdir(figures_dir) if f.endswith('.png')])
print(f"Report figures: {len(fig_files)}")

for fig_file in fig_files:
    fig_path = os.path.join(figures_dir, fig_file)
    print(f"\n--- {fig_file} ---")
    ipy_display(Image(filename=fig_path, width=600))
```

**Expected output:** All report figures displayed inline.

#### Cell 20: Markdown — Completion summary

```markdown
## Phase 7 Complete

**Artifacts generated:**
- `xai_report.md` — full XAI analysis report
- `thesis_text_blocks.md` — ready-to-paste thesis paragraphs
- `summary_tables/` — CSV and Markdown tables
- `report_figures/` — thesis-quality figure copies

**Next step:** Phase 8 (Thesis Visualization) for publication-quality figure refinement.
```

#### Cell 21: Completion verification

```python
expected_files = [
    generator.report_path,
    generator.thesis_blocks_path,
    os.path.join(generator.tables_dir, 'xai_method_comparison.md'),
]

print("Completion check:")
all_ok = True
for fpath in expected_files:
    exists = os.path.isfile(fpath)
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print(f"  [{status}] {os.path.basename(fpath)}")

if all_ok:
    print("\nPhase 7 COMPLETE.")
else:
    print("\nPhase 7 INCOMPLETE — check missing files above.")
```

**Expected output:** All expected files exist; "Phase 7 COMPLETE."

---

## 10. Algorithm

### Main workflow pseudocode

```
FUNCTION generate_full_report(exp_dir, exp_id):
    # Step 1: Initialize
    xai_dir = exp_dir / "xai"
    VERIFY xai_dir exists
    
    # Step 2: Scan available phases
    available = {}
    FOR each phase IN [gradcam, attention, shap, lime, case_studies]:
        phase_dir = xai_dir / phase
        IF phase_dir exists AND contains expected files:
            available[phase] = True
            LOG "Found {count} artifacts in {phase}/"
        ELSE:
            available[phase] = False
            LOG "Phase {phase} not found"
    
    # Step 3: Load experiment config
    config = LOAD config.yaml OR config.json FROM exp_dir
    metrics = LOAD test_metrics.json FROM exp_dir (if exists)
    
    # Step 4: Generate report sections
    report = []
    report += generate_header(exp_id, config, timestamp=NOW)
    report += generate_executive_summary(config, metrics, available)
    
    IF available['gradcam']:
        metadata = LOAD gradcam/gradcam_metadata.json
        report += generate_gradcam_section(metadata, xai_dir)
    ELSE:
        report += generate_missing_phase_note("Grad-CAM", 2)
    
    IF available['attention']:
        metadata = LOAD attention/attention_metadata.json
        report += generate_attention_section(metadata, xai_dir)
    ELSE:
        report += generate_missing_phase_note("Attention", 3)
    
    IF available['shap']:
        metadata = LOAD shap/shap_metadata.json
        modality_csv = LOAD shap/modality_contribution_summary.csv
        report += generate_shap_section(metadata, modality_csv, xai_dir)
    ELSE:
        report += generate_missing_phase_note("SHAP", 4)
    
    IF available['lime']:
        metadata = LOAD lime/lime_metadata.json
        report += generate_lime_section(metadata, xai_dir)
    ELSE:
        report += generate_missing_phase_note("LIME", 5)
    
    # Step 5: Cross-method comparison (needs >= 2 methods)
    completed_methods = [k for k, v in available.items() 
                         if v and k != 'case_studies']
    IF len(completed_methods) >= 2:
        report += generate_cross_method_comparison(available, xai_dir)
    ELSE:
        report += generate_insufficient_methods_note(completed_methods)
    
    IF available['case_studies']:
        cs_metadata = LOAD case_studies/case_study_metadata.json
        report += generate_case_studies_section(cs_metadata, xai_dir)
    ELSE:
        report += generate_missing_phase_note("Case Studies", 6)
    
    report += generate_limitations_section()
    
    # Step 6: Write report
    WRITE report TO xai_dir / "xai_report.md"
    
    # Step 7: Generate summary tables
    IF available['shap']:
        generate_modality_contribution_table(modality_csv, tables_dir)
    IF available['case_studies'] AND len(completed_methods) >= 2:
        generate_cross_method_agreement_table(cs_metadata, available, tables_dir)
    generate_case_study_summary_table(cs_metadata, tables_dir)
    generate_method_comparison_table(tables_dir)
    
    # Step 8: Copy report figures
    copy_report_figures(available, xai_dir, figures_dir)
    
    # Step 9: Generate thesis text blocks
    generate_thesis_text_blocks(config, metrics, modality_csv, thesis_path)
    
    RETURN report_path
```

### Modality contribution aggregation pseudocode

```
FUNCTION compute_modality_contribution_stats(modality_csv_path):
    df = READ_CSV(modality_csv_path)
    
    results = []
    FOR target_idx IN [0, 1, 2, 3, 4]:
        target_name = display_names[target_idx]
        subset = df WHERE target_index == target_idx
        
        IF len(subset) == 0:
            CONTINUE
        
        row = {
            'target': target_name,
            'text_pct_mean': MEAN(subset.text_pct),
            'text_pct_std': STD(subset.text_pct),
            'text_pct_min': MIN(subset.text_pct),
            'text_pct_max': MAX(subset.text_pct),
            'image_pct_mean': MEAN(subset.image_pct),
            'image_pct_std': STD(subset.image_pct),
            'image_pct_min': MIN(subset.image_pct),
            'image_pct_max': MAX(subset.image_pct),
            'n_samples': len(subset),
        }
        results.APPEND(row)
    
    RETURN results
```

### Cross-method agreement pseudocode

```
FUNCTION assess_cross_method_agreement(sample_id, available, xai_dir):
    agreement = {}
    
    # Image agreement: Grad-CAM vs LIME
    IF available['gradcam'] AND available['lime']:
        gradcam_meta = LOAD gradcam metadata for sample_id
        lime_meta = LOAD lime metadata for sample_id
        
        # Compare focus regions
        # If both highlight similar image areas: "agree"
        # If partial overlap: "partial"
        # If different areas: "disagree"
        agreement['gradcam_lime_image'] = ASSESS_OVERLAP(
            gradcam_focus_region, lime_positive_superpixels
        )
    
    # Text agreement: Attention vs LIME
    IF available['attention'] AND available['lime']:
        attention_top_tokens = LOAD word_importance for sample_id
        lime_top_words = LOAD lime word weights for sample_id
        
        top_k = 5
        attention_set = SET(attention_top_tokens[:top_k])
        lime_set = SET(lime_top_words[:top_k])
        overlap = len(attention_set INTERSECT lime_set) / top_k
        
        IF overlap >= 0.6:
            agreement['attention_lime_text'] = 'agree'
        ELIF overlap >= 0.2:
            agreement['attention_lime_text'] = 'partial'
        ELSE:
            agreement['attention_lime_text'] = 'disagree'
    
    # Modality dominance: SHAP vs qualitative
    IF available['shap']:
        shap_modality = LOAD modality contribution for sample_id
        IF shap_modality.image_pct > 60:
            agreement['shap_dominance'] = 'image-dominant'
        ELIF shap_modality.text_pct > 60:
            agreement['shap_dominance'] = 'text-dominant'
        ELSE:
            agreement['shap_dominance'] = 'balanced'
    
    RETURN agreement
```

### Figure copy pseudocode

```
FUNCTION copy_report_figures(available, xai_dir, figures_dir):
    copied = []
    
    IF available['gradcam']:
        # Find the first 5-target comparison grid
        gradcam_dir = xai_dir / "gradcam"
        grid_candidates = GLOB(gradcam_dir, "*5target*" OR "*comparison*")
        IF grid_candidates:
            COPY grid_candidates[0] TO figures_dir / "fig_gradcam_5target_comparison.png"
            copied.APPEND("fig_gradcam_5target_comparison.png")
        ELSE:
            # Copy individual heatmaps for the first sample
            FOR target IN factor_names:
                heatmap = FIND first heatmap matching target
                IF heatmap:
                    COPY heatmap TO figures_dir / f"fig_gradcam_{target}_example.png"
                    copied.APPEND(f"fig_gradcam_{target}_example.png")
    
    IF available['shap']:
        shap_dir = xai_dir / "shap"
        modality_charts = GLOB(shap_dir, "*modality*")
        IF modality_charts:
            COPY modality_charts[0] TO figures_dir / "fig_shap_modality_all_targets.png"
            copied.APPEND("fig_shap_modality_all_targets.png")
        
        waterfall_examples = GLOB(shap_dir, "*waterfall*")
        IF waterfall_examples:
            COPY waterfall_examples[0] TO figures_dir / "fig_shap_waterfall_example.png"
            copied.APPEND("fig_shap_waterfall_example.png")
    
    IF available['attention']:
        attention_dir = xai_dir / "attention"
        heatmaps = GLOB(attention_dir, "*heatmap*")
        IF heatmaps:
            COPY heatmaps[0] TO figures_dir / "fig_attention_example.png"
            copied.APPEND("fig_attention_example.png")
    
    IF available['case_studies']:
        cs_dir = xai_dir / "case_studies"
        combined_figs = GLOB(cs_dir, "combined_*")
        counter_correct = 1
        counter_error = 1
        FOR fig IN combined_figs:
            IF "correct" IN fig:
                dest = f"fig_case_study_correct_{counter_correct:03d}.png"
                counter_correct += 1
            ELSE:
                dest = f"fig_case_study_higherror_{counter_error:03d}.png"
                counter_error += 1
            COPY fig TO figures_dir / dest
            copied.APPEND(dest)
    
    LOG f"Copied {len(copied)} figures to {figures_dir}"
    RETURN copied
```

---

## 11. Validation

### V1: Report renders correctly as Markdown

**Test:** Open `xai_report.md` in a Markdown renderer (VS Code preview, GitHub, Colab markdown cell). Verify:
- All headers render as proper headings (H1, H2, H3).
- All tables render with correct column alignment.
- All image references use correct relative paths.
- No broken Markdown syntax (unclosed code blocks, mismatched backticks).

**Automated check:** Parse the Markdown for common syntax errors:
- Every `![` has a matching `]()`.
- Every table has matching `|` counts per row.
- Every code block ` ``` ` is closed.

### V2: All referenced figures exist

**Test:** Extract every image reference from `xai_report.md` (regex `!\[.*?\]\((.*?)\)`). For each path, verify `os.path.exists(resolved_path)` returns True.

**Automated check:** The report generator should perform this validation before writing the final report. If a referenced figure does not exist, replace the reference with `*(Figure not found: filename)*`.

### V3: Tables have correct values matching raw data

**Test:** For the modality contribution table:
1. Read `shap/modality_contribution_summary.csv` directly.
2. Manually compute mean, std, min, max for each target.
3. Compare with values in `summary_tables/modality_contribution_by_target.csv`.
4. All values must match to 4 decimal places.

**Automated check:** The report generator can include a `verify_tables()` method that recomputes statistics from raw data and asserts equality with generated tables.

### V4: Thesis text blocks are grammatically correct English

**Test:** Read `thesis_text_blocks.md`. Verify:
- No Vietnamese text appears in the English paragraphs (except quoted review text).
- Citations follow the format: `[Author et al., Year]`.
- Sentences are complete and grammatically well-formed.
- Technical terms are used correctly (e.g., "Shapley values" not "shapley values").

**Manual check:** This requires human review. The generated text should be proofread before thesis submission.

### V5: Report regeneration produces identical output

**Test:** Run the report generator twice on the same artifacts without any changes. The output files should be byte-identical except for the generation timestamp. If timestamps are excluded from comparison, the rest of the content must be identical.

**Automated check:** Compare file hashes (excluding timestamp lines).

### V6: Graceful degradation works correctly

**Test:** Temporarily rename one artifact directory (e.g., rename `lime/` to `lime_backup/`). Run the report generator. Verify:
- The report generates without errors.
- The LIME section contains the "not completed" note.
- All other sections are unaffected.
- Restore the directory and verify the full report generates correctly again.

### V7: Figure naming consistency

**Test:** Verify all files in `report_figures/` follow the naming convention `fig_{method}_{description}.png`. No files should have numerical prefixes or ambiguous names.

---

## 12. Risks — Fully Analyzed

### R1: Missing artifacts from incomplete phases

**Problem:** Not all XAI phases (2-6) may be completed before report generation is attempted. For example, LIME (Phase 5) is computationally expensive and may be skipped for early iterations.

**Why it happens:** The XAI pipeline is designed to be executed incrementally. Phase 7 may be run after only Phases 2 and 4, for instance.

**Possible strategies:**

| Strategy | Description |
|---|---|
| A. Require all phases | Refuse to generate report unless all phases are complete |
| B. Graceful degradation | Generate report with available phases, mark missing ones |
| C. Partial report modes | Generate separate mini-reports per phase |

**Advantages and disadvantages:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A | Guarantees completeness | Blocks progress; impractical during iterative development |
| B | Allows incremental progress; report improves as phases complete | Report may look sparse initially; need to track what is missing |
| C | Each report is self-contained | Duplicates boilerplate; no unified cross-method analysis |

**Engineering trade-off:** Strategy A is the safest for a final thesis but blocks iterative workflow. Strategy C fragments the narrative. Strategy B provides the best workflow for research development.

**Research trade-off:** A thesis examiner expects a complete report, but during development, partial reports are more useful than no report.

**Recommended implementation:** Strategy B — Graceful degradation.

**FINAL DECISION:** Implement graceful degradation. Each section generator checks for artifact existence before generating content. Missing sections are noted with "Phase X ({method_name}) has not been completed. No artifacts found." The report is valid and renderable regardless of which phases are complete. The executive summary clearly states which phases are available.

**Reason:** This allows the report to be generated and reviewed incrementally, with the thesis author able to see the growing narrative as each phase completes. The final thesis submission will have all phases complete, producing a full report.

---

### R2: Report becomes stale after re-running XAI

**Problem:** If Grad-CAM is re-run with different sample indices or different parameters, the report may reference old figures or contain outdated statistics. The user might forget to regenerate the report after updating artifacts.

**Why it happens:** The report is a derived artifact, not an input to other phases. There is no automatic dependency tracking between Phase 2-6 outputs and Phase 7.

**Possible strategies:**

| Strategy | Description |
|---|---|
| A. Timestamp-based staleness detection | Compare report timestamp with latest artifact modification time |
| B. Checksum-based tracking | Store checksums of all input artifacts; warn if changed |
| C. Always regenerate | Report generator always regenerates from scratch, no caching |
| D. Git-based tracking | Track artifact versions via git commits |

**Advantages and disadvantages:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A | Simple; detects most staleness | File timestamps can be unreliable across OS/drive mounts |
| B | Precise; detects content changes even if timestamps preserved | Adds complexity; checksum file must be maintained |
| C | Always correct; no staleness possible | Slightly slower (but report generation is fast, ~seconds) |
| D | Full version history | Artifacts may be too large for git; adds git workflow burden |

**Engineering trade-off:** Checksum tracking adds a maintenance burden. Timestamp detection can fail on Google Drive or mounted filesystems. Always-regenerate is simplest.

**Research trade-off:** Reproducibility benefits from checksums, but the overhead is not justified for a single-experiment thesis pipeline.

**Recommended implementation:** Strategy C (always regenerate) combined with a lightweight checksum log for reproducibility tracking.

**FINAL DECISION:** The report generator always regenerates from current artifacts. It never reads a cached previous report. The generation timestamp is included in the report header. Additionally, after generation, a `report_checksums.json` is written listing the MD5 hash of every input artifact that was read. This allows future verification that the report matches its source data, without introducing dependency-tracking complexity.

**Reason:** Report generation takes only seconds (it reads files and writes Markdown, no model inference). Always regenerating eliminates staleness risk entirely. The checksum log provides reproducibility without blocking the workflow.

---

### R3: Figure naming conflicts with thesis

**Problem:** The thesis may need different figure numbering than the report. If report figures are named `figure_1.png`, `figure_2.png`, they will conflict with LaTeX `\begin{figure}` auto-numbering.

**Why it happens:** LaTeX assigns figure numbers based on document order (`\caption` placement). Hardcoded figure numbers in filenames create a disconnect.

**Possible strategies:**

| Strategy | Description |
|---|---|
| A. Numbered filenames | Name files `figure_01.png`, `figure_02.png` |
| B. Descriptive filenames | Name files `fig_gradcam_5target_comparison.png` |
| C. Both | Use descriptive names with an optional index prefix |

**Advantages and disadvantages:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A | Easy to reference by number | Numbers may change if figures are reordered; conflicts with LaTeX |
| B | Self-documenting; no conflict with LaTeX numbering | Longer filenames; must remember names |
| C | Covers both use cases | Redundant; index prefix may still conflict |

**Engineering trade-off:** Descriptive names are slightly longer but eliminate all numbering conflicts.

**Research trade-off:** None. Descriptive names are standard practice in research projects.

**Recommended implementation:** Strategy B — Descriptive filenames.

**FINAL DECISION:** All report figures use descriptive names with the prefix `fig_` followed by method and description. Example: `fig_gradcam_5target_comparison.png`, not `figure_3.png`. The thesis LaTeX will assign `\label{fig:gradcam-5target}` and `\ref{fig:gradcam-5target}` independently of the filename.

**Reason:** This is standard academic practice and avoids all conflicts with document-level numbering.

---

### R4: English vs Vietnamese in report

**Problem:** The review text in the dataset is Vietnamese, but the thesis and report are written in English. Vietnamese text must appear in the report for case studies but should be properly contextualized.

**Why it happens:** This is a Vietnamese restaurant review dataset. The model processes Vietnamese text with PhoBERT. Showing the original text is essential for understanding attention patterns and LIME word importance.

**Possible strategies:**

| Strategy | Description |
|---|---|
| A. English-only report | Translate all Vietnamese text to English |
| B. Vietnamese-only examples | Show Vietnamese text without translation |
| C. Bilingual | Show Vietnamese original with English annotation |

**Advantages and disadvantages:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A | Universally readable | Loses connection to actual model input; translations may be imprecise |
| B | Authentic; shows what the model actually processed | Non-Vietnamese readers cannot understand the text |
| C | Complete; respects both audiences | Slightly verbose; requires translation effort |

**Engineering trade-off:** Strategy C requires maintaining translations, but for a small number of case study samples this is manageable.

**Research trade-off:** Showing the original Vietnamese text is scientifically important because it is the actual model input. Translation provides accessibility.

**Recommended implementation:** Strategy C — Bilingual with English annotation.

**FINAL DECISION:** The report is written in English. Vietnamese review text is quoted in its original form (inside blockquotes or code blocks). Key Vietnamese terms that appear in attention/LIME analysis are annotated with English translations inline: e.g., "`ngon` (delicious)", "`gia hoi cao` (somewhat expensive price)". Full sentence translations are not provided unless the text is central to the analysis narrative.

**Reason:** This preserves authenticity while keeping the report accessible. The thesis examiner can see the actual model input while understanding the semantic content through annotations.

---

### R5: Report length becomes unwieldy

**Problem:** If many samples are analyzed across all methods, the report could grow to thousands of lines, making it difficult to navigate and use as a thesis draft.

**Why it happens:** Each sample generates multiple figures and analysis paragraphs across 4-5 methods. With 10+ samples per method, the report can easily exceed 5000 lines.

**Possible strategies:**

| Strategy | Description |
|---|---|
| A. Include everything | Show all samples in the main report |
| B. Representative subset | Show 2-3 representative samples per section; reference appendix for rest |
| C. Summary + detailed appendix | Main report has summaries; separate appendix file has full details |

**Advantages and disadvantages:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A | Complete; no information lost | Too long for practical use; hard to navigate |
| B | Balanced; thesis-appropriate length | Must choose representative samples carefully |
| C | Clean separation of summary and details | Two files to maintain; cross-references more complex |

**Engineering trade-off:** Strategy B is simplest to implement and produces the most thesis-ready output.

**Research trade-off:** A thesis chapter should show representative examples in the main text and mention that additional examples are available. Strategy B matches this convention.

**Recommended implementation:** Strategy B.

**FINAL DECISION:** Each method section shows at most 3 representative samples in full detail (with figures and analysis). If more samples are available, the section notes: "Analysis was performed on N total samples. Full results are available in the individual phase artifact directories." The cross-method comparison and case study sections include all case study samples (typically 4-8 total from Phase 6).

**Reason:** This produces a report of manageable length (~1500-3000 lines) that directly serves as a thesis chapter draft, while preserving full data access through the underlying artifact directories.

---

### R6: Inconsistency between report text and actual figures

**Problem:** The report might describe observations (e.g., "Grad-CAM highlighted the food region") that do not match the actual figure content if the text is generated from templates rather than from actual artifact analysis.

**Why it happens:** Report text blocks are partially templated. While data-driven sections (tables, statistics) are computed from artifacts, qualitative descriptions may rely on metadata annotations from Phase 6 rather than automated image analysis.

**Possible strategies:**

| Strategy | Description |
|---|---|
| A. Fully templated text | Use generic descriptions that are always safe |
| B. Metadata-driven text | Use Phase 6 analysis annotations as the text source |
| C. Manual review required | Generate conservative text and flag sections for human review |

**Advantages and disadvantages:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A | Never incorrect; always safe | Generic; does not add analytical value |
| B | Specific; leverages Phase 6 work | Depends on Phase 6 quality; may propagate errors |
| C | Honest about limitations | Requires additional human effort |

**Engineering trade-off:** Strategy B leverages the investment made in Phase 6 case study analysis. Strategy C adds transparency.

**Research trade-off:** Thesis text should be reviewed by the author regardless. The report is a draft, not a final submission.

**Recommended implementation:** Combination of B and C.

**FINAL DECISION:** Qualitative descriptions are sourced from Phase 6 analysis text files (`analysis_sampleXXX.txt`) when available. When no analysis text exists, the report uses conservative templated descriptions and marks them with `[REVIEW NEEDED]` tags. All quantitative values (statistics, percentages) are computed directly from raw data and are never templated. The thesis author is expected to review and refine qualitative descriptions before thesis submission.

**Reason:** This maximizes the value of the automated report while being honest about the boundary between automated and human-reviewed content.

---

### R7: Artifact directory structure changes between phases

**Problem:** If the artifact directory structure from Phases 2-6 does not exactly match what the report generator expects (e.g., different file naming conventions, missing metadata JSON files), the report generator may fail or produce incomplete output.

**Why it happens:** Phases 2-6 are implemented independently, and minor naming or structural inconsistencies can accumulate.

**Possible strategies:**

| Strategy | Description |
|---|---|
| A. Strict schema enforcement | Define exact expected filenames; fail on mismatch |
| B. Flexible pattern matching | Use glob patterns and fuzzy matching to find artifacts |
| C. Discovery-based scanning | Scan directories and infer artifact types from content |

**Advantages and disadvantages:**

| Strategy | Advantages | Disadvantages |
|---|---|---|
| A | Predictable; catches inconsistencies early | Brittle; any naming change breaks the report |
| B | Robust to minor variations | May match wrong files if patterns are too loose |
| C | Most flexible | Complex; may misidentify files |

**Engineering trade-off:** Strategy B with well-defined patterns is the best balance of robustness and correctness.

**Research trade-off:** None significant. Artifact discovery is an engineering concern.

**Recommended implementation:** Strategy B — Flexible pattern matching with documented conventions.

**FINAL DECISION:** The report generator uses glob patterns to find artifacts (e.g., `heatmap_*_food.png` for Grad-CAM food heatmaps). Pattern conventions are documented in the module docstring and in the Phase 1 infrastructure proposal. If a metadata JSON file exists, it is used as the primary source of artifact locations. If it does not exist, the generator falls back to pattern-based discovery. This two-tier approach handles both well-structured (metadata-driven) and loosely-structured (pattern-discovered) artifact directories.

**Reason:** Metadata JSON files are the clean path and should be the primary interface. Pattern-based fallback handles the case where earlier phases were run without full metadata generation.

---

## 13. Best Practices

### Reproducibility

1. **Always regenerate from artifacts.** Never manually edit `xai_report.md` and expect the edits to persist. The report generator will overwrite on regeneration. Manual edits belong in `thesis_text_blocks.md` or in the thesis LaTeX files.
2. **Checksum logging.** After report generation, write `report_checksums.json` containing the MD5 hash of every input file that was read. This allows future verification that the report matches its sources.
3. **Timestamp inclusion.** The report header includes the ISO 8601 generation timestamp. This allows tracking when the report was last generated relative to artifact modification dates.

### Logging

1. Use Python `logging` module at INFO level for normal operations and WARNING level for missing artifacts or degraded sections.
2. Log every file read operation with the file path and size.
3. Log every file write operation with the output path.
4. Log artifact counts per phase during initialization.
5. Do not use `print()` in the module; use `logging.getLogger(__name__)`.

### Figure consistency

1. All copied figures maintain their original resolution and DPI. The report generator does not resize or reprocess figures.
2. Figure filenames use lowercase ASCII with underscores. No spaces, no special characters.
3. Figure references in the Markdown report use relative paths from the report file location: `![caption](report_figures/fig_gradcam_5target_comparison.png)`.

### Configuration management

1. Architecture constants (factor names, display names, target indices) are defined once in the `ReportGenerator.__init__` method and referenced throughout.
2. The report generator does not hardcode experiment-specific values (e.g., specific MAE numbers). All values are read from artifacts.
3. The only hardcoded content is: method descriptions, citation text, and the method comparison table (which is architecture-specific but not data-dependent).

### Error handling

1. Every file read is wrapped in try/except. On failure, log a warning and continue with a default value or skip the section.
2. JSON parsing failures produce a warning and a `{}` default, not a crash.
3. CSV parsing failures produce a warning and an empty DataFrame, not a crash.
4. The report generator should never raise an unhandled exception. The worst case is a report with many "not available" notes, which is still valid and informative.

### Memory efficiency

1. Do not load NPY files into memory unless their values are needed for aggregation. Most NPY files are only referenced by path, not loaded.
2. Do not load PNG files into memory. The report generator only copies file paths and embeds them as Markdown image references.
3. If loading SHAP value arrays for aggregation, process one target at a time and release memory between targets.

### Code organization

1. The `ReportGenerator` class should be the only public API of the module. All helper functions should be private methods (prefixed with `_`).
2. Each section generator is a separate method, making it easy to modify one section without affecting others.
3. The module should have no side effects on import. All file I/O happens only when `generate_full_report()` is called.

---

## 14. Deliverables

### After Phase 7 completion, the following artifacts will exist:

#### Primary report files

| Artifact | Format | Size estimate | Description |
|---|---|---|---|
| `xai/xai_report.md` | Markdown | 1500-3000 lines | Full XAI analysis report |
| `xai/thesis_text_blocks.md` | Markdown | 300-600 lines | Ready-to-paste thesis paragraphs |
| `xai/report_checksums.json` | JSON | ~2 KB | MD5 checksums of all input artifacts |

#### Summary tables

| Artifact | Format | Description |
|---|---|---|
| `xai/summary_tables/modality_contribution_by_target.csv` | CSV | Per-target modality contribution statistics |
| `xai/summary_tables/modality_contribution_by_target.md` | Markdown | Same as above, formatted as Markdown table |
| `xai/summary_tables/cross_method_agreement.csv` | CSV | Agreement assessment between methods per sample |
| `xai/summary_tables/case_study_summary.csv` | CSV | One row per case study with key metrics |
| `xai/summary_tables/xai_method_comparison.md` | Markdown | Method strengths and limitations comparison table |

#### Report figures (copies)

| Artifact | Format | Description |
|---|---|---|
| `xai/report_figures/fig_gradcam_5target_comparison.png` | PNG | Representative Grad-CAM across 5 targets |
| `xai/report_figures/fig_shap_modality_all_targets.png` | PNG | SHAP modality contribution chart |
| `xai/report_figures/fig_shap_waterfall_example.png` | PNG | SHAP waterfall plot example |
| `xai/report_figures/fig_attention_example.png` | PNG | Representative attention heatmap |
| `xai/report_figures/fig_case_study_correct_NNN.png` | PNG | Case study combined figures (correct predictions) |
| `xai/report_figures/fig_case_study_higherror_NNN.png` | PNG | Case study combined figures (high-error predictions) |
| `xai/report_figures/fig_lime_image_example.png` | PNG | LIME superpixel example (if Phase 5 completed) |
| `xai/report_figures/fig_lime_text_example.png` | PNG | LIME text importance example (if Phase 5 completed) |

#### Code files

| Artifact | Format | Description |
|---|---|---|
| `xai/report_generator.py` | Python | Report generation module |
| `xai/notebooks/Phase7_Report.ipynb` | Jupyter | Interactive report generation notebook |

---

## 15. Thesis Usage

### Results chapter

- `xai_report.md` serves as the first draft of the "XAI Results" section of the thesis. The structure (Grad-CAM Analysis, Attention Analysis, SHAP Analysis, LIME Analysis, Cross-Method Comparison, Case Studies, Limitations) maps directly to thesis subsections.
- The modality contribution table (`modality_contribution_by_target.csv`) becomes a key thesis table showing the balance between image and text evidence across prediction targets.
- Case study figures become the central visual evidence in the thesis results section.

### Discussion chapter

- The cross-method comparison section provides material for the "Discussion" chapter, specifically the subsection on multi-method consistency and disagreement.
- The limitations section provides the foundation for the "Limitations" subsection of the discussion.
- Thesis text blocks in `thesis_text_blocks.md` can be directly pasted into the LaTeX document with minimal modification.

### Case Studies

- Each case study in the report corresponds to one thesis case study subsection. The combined figure, analysis text, and cross-method comparison provide a complete analytical narrative.
- The case study summary table provides an overview for the thesis results summary.

### Thesis figures and tables

- All files in `report_figures/` can be copied directly to the thesis `figures/` directory. Their descriptive names make them easy to reference in LaTeX: `\includegraphics{figures/fig_gradcam_5target_comparison.png}`.
- All `.md` tables in `summary_tables/` can be converted to LaTeX `tabular` format using standard tools (e.g., `pandoc` or manual conversion).
- All `.csv` files can be imported into LaTeX using the `csvsimple` or `pgfplotstable` packages.

### Defense presentation

- The executive summary section of the report provides a ready-made narrative for the XAI portion of the defense presentation.
- Key figures (Grad-CAM comparison, SHAP modality chart, attention example, case study combined figure) can be extracted as presentation slides.
- The method comparison table serves as a defense slide comparing XAI approaches.

### Journal paper

- The modality contribution statistics provide quantitative results for a journal paper's results section.
- The cross-method comparison provides material for a journal paper's discussion of explanation reliability.
- Thesis text blocks can be adapted for journal format by shortening and adjusting citations to the journal style.

---

## 16. Phase Completion Checklist

### Infrastructure

- [ ] `xai/report_generator.py` module exists and is importable.
- [ ] `ReportGenerator` class can be instantiated with `exp_dir` and `exp_id` parameters.
- [ ] Initialization correctly scans XAI subdirectories and identifies available phases.
- [ ] Initialization loads experiment config and test metrics without errors.
- [ ] Output directories (`summary_tables/`, `report_figures/`) are created automatically.

### Report generation

- [ ] `generate_full_report()` produces `xai_report.md` without errors.
- [ ] Report renders correctly as Markdown (verified in VS Code, GitHub, or Colab).
- [ ] Report header includes correct experiment ID, model configuration, and generation timestamp.
- [ ] Executive summary includes test set performance metrics.
- [ ] Executive summary lists which XAI phases are available.

### Section completeness

- [ ] Grad-CAM section includes method description, representative heatmaps, and domain consistency assessment (or "not completed" note).
- [ ] Attention section includes method description, token heatmaps, and top-token analysis (or "not completed" note).
- [ ] SHAP section includes method description, modality contribution table, and waterfall examples (or "not completed" note).
- [ ] LIME section includes method description, superpixel examples, and word importance (or "not completed" note).
- [ ] Cross-method comparison section compares at least 2 methods (or notes insufficient methods).
- [ ] Case studies section includes combined figures and analysis text (or "not completed" note).
- [ ] Limitations section is always present and covers all methods.

### Summary tables

- [ ] `modality_contribution_by_target.csv` exists and contains correct per-target statistics.
- [ ] `modality_contribution_by_target.md` exists as a formatted Markdown table.
- [ ] `cross_method_agreement.csv` exists (if applicable).
- [ ] `case_study_summary.csv` exists (if applicable).
- [ ] `xai_method_comparison.md` exists with correct method descriptions.
- [ ] All CSV values match raw data (verified by manual spot-check or automated test).

### Report figures

- [ ] `report_figures/` directory contains copies of key figures with descriptive names.
- [ ] Every figure referenced in `xai_report.md` exists at the referenced path.
- [ ] No figure file in `report_figures/` is unreferenced in the report.
- [ ] All figure filenames follow the `fig_{method}_{description}.png` convention.

### Thesis text blocks

- [ ] `thesis_text_blocks.md` exists and contains method description paragraphs.
- [ ] Results description paragraphs include actual statistics (not template placeholders).
- [ ] Discussion paragraphs are grammatically correct English.
- [ ] All citations use the format `[Author et al., Year]`.
- [ ] Figure captions use `{placeholder}` for figure numbers.

### Graceful degradation

- [ ] Report generates successfully when only 1 XAI phase is complete.
- [ ] Report generates successfully when all XAI phases are complete.
- [ ] Missing phases are clearly noted in the report text.
- [ ] No unhandled exceptions occur for any combination of available/missing phases.

### Reproducibility

- [ ] `report_checksums.json` is generated after report completion.
- [ ] Running the generator twice on unchanged artifacts produces identical reports (except timestamp).
- [ ] The notebook (`Phase7_Report.ipynb`) runs end-to-end in Google Colab without errors.

### Integration

- [ ] Report content is consistent with Phase 1 infrastructure naming conventions.
- [ ] Report figure names do not conflict with existing Phase 2-6 artifact names.
- [ ] Report can be regenerated after any Phase 2-6 is re-run.
- [ ] The report generator does not modify any Phase 2-6 artifact files.
