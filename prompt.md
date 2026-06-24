# ROLE

You are a Senior Machine Learning Research Engineer, Data Visualization Specialist, and Research Reproducibility Engineer.

You specialize in:

- PyTorch experiment analysis
- Multimodal deep learning experiment tracking
- Research leaderboard generation
- Ablation study visualization
- Google Colab workflows
- Google Drive artifact management
- pandas, matplotlib, seaborn-free visualization
- Jupyter Notebook generation
- Scientific reporting for thesis projects

Your task is to understand the existing codebase and experiment artifact structure, then create a complete notebook that automatically reads experiment outputs and generates tables/figures for thesis reporting.

---

# GOAL

Read the entire codebase and understand:

- how the project is implemented
- how experiments are named
- how the 21 experiments were trained
- how experiment folders are structured
- what files each experiment produces
- where outputs are stored in Google Drive
- what metrics are available in `metrics.json`
- which experiments belong to which phase
- which experiments are validation-only
- which Phase 6 experiments also contain test metrics and test figures

Then create a Jupyter notebook file:

```text
notebooks/generate_experiment_leaderboard.ipynb
```

This notebook must be runnable in Google Colab.

The notebook should automatically read all experiment folders from:

```python
DRIVE_ROOT = "/content/drive/MyDrive/SE365"
EXPERIMENTS_DIR = f"{DRIVE_ROOT}/experiments"
REPORTS_DIR = f"{DRIVE_ROOT}/reports"
```

Then generate the following outputs.

---

# REQUIRED FIGURES

The notebook must generate exactly these 7 figures:

```text
1. Overall Leaderboard
2. Image Backbone Comparison
3. Text Backbone Comparison
4. Fusion Comparison
5. Loss Comparison
6. Performance Evolution Across Phases
7. Top-3 Radar Chart
```

Recommended save paths:

```text
/content/drive/MyDrive/SE365/reports/figures/01_overall_leaderboard.png
/content/drive/MyDrive/SE365/reports/figures/02_image_backbone_comparison.png
/content/drive/MyDrive/SE365/reports/figures/03_text_backbone_comparison.png
/content/drive/MyDrive/SE365/reports/figures/04_fusion_comparison.png
/content/drive/MyDrive/SE365/reports/figures/05_loss_comparison.png
/content/drive/MyDrive/SE365/reports/figures/06_performance_evolution.png
/content/drive/MyDrive/SE365/reports/figures/07_top3_radar_chart.png
```

---

# REQUIRED TABLES

The notebook must generate exactly these 4 tables:

```text
1. Full Experiment Leaderboard
2. Ablation Summary
3. Validation vs Test Comparison
4. Improvement vs Baseline
```

Recommended save paths:

```text
/content/drive/MyDrive/SE365/reports/tables/full_experiment_leaderboard.csv
/content/drive/MyDrive/SE365/reports/tables/full_experiment_leaderboard.xlsx

/content/drive/MyDrive/SE365/reports/tables/ablation_summary.csv
/content/drive/MyDrive/SE365/reports/tables/ablation_summary.xlsx

/content/drive/MyDrive/SE365/reports/tables/validation_vs_test_comparison.csv
/content/drive/MyDrive/SE365/reports/tables/validation_vs_test_comparison.xlsx

/content/drive/MyDrive/SE365/reports/tables/improvement_vs_baseline.csv
/content/drive/MyDrive/SE365/reports/tables/improvement_vs_baseline.xlsx
```

Also generate:

```text
/content/drive/MyDrive/SE365/reports/experiment_report.md
```

summarizing key results.

---

# CONTEXT

The project is an Explainable Multimodal Deep Learning system for Vietnamese restaurant review quality regression.

Each experiment folder is located under:

```text
/content/drive/MyDrive/SE365/experiments/<EXPERIMENT_ID>/
```

Example experiment folders may look like:

```text
EXP_010_text_only_xlmr_mse
EXP_011_image_only_convnext_meanpool_mse
EXP_012_multimodal_convnext_xlmr_concat_mse
EXP_020B_swinb_xlmr_concat_mse
EXP_020D_efficientnetb3_xlmr_concat_mse
EXP_030B_bestimage_phobert_concat_mse
EXP_030D_bestimage_visobert_concat_mse
EXP_040B_bestimage_besttext_gmu_mse
EXP_040C_bestimage_besttext_gatedcrossmodal_mse
EXP_041A_bestimage_besttext_film_mse
EXP_041B_bestimage_besttext_crossattention_mse
EXP_050B_bestfusion_huber
EXP_050C_bestfusion_logcosh
EXP_051D_bestfusion_uncertaintyweighted
EXP_060A_bestsequential_full_configuration
EXP_060B_swinb_visobert_gmu_uncertainty
EXP_060C_efficientnetb3_phobert_film_huber
EXP_060D_efficientnetb3_visobert_crossattention_logcosh
EXP_060E_convnext_phobert_gatedcrossmodal_autoweight
```

The exact folder list must be discovered automatically from Drive.

Do not hard-code the number of experiments.

---

# EXPERIMENT FOLDER CONTENTS

Each experiment folder may contain:

```text
config.yaml
metrics.json
predictions.csv
train.log
best_model_train_text.pth
best_model_train_fusion.pth
```

Some Phase 6 experiments may additionally contain test outputs such as:

```text
test_metrics.json
test_predictions.csv
test_scatter_pred_vs_true.png
test_mae_aspects.png
test_error_distributions.png
```

The notebook must handle missing files gracefully.

If a folder does not contain `metrics.json`, skip it and report it in a warning table.

If a folder contains `test_metrics.json`, include it in the validation-vs-test comparison.

If a folder does not contain `test_metrics.json`, leave test metric columns as `NaN`.

---

# METRICS FORMAT

A typical `metrics.json` may contain keys such as:

```json
{
  "loss": 2.8099678943031714,
  "mae_food": 1.257565975189209,
  "rmse_food": 1.745107650756836,
  "r2_food": 0.4213736057281494,
  "mae_price": 1.273634433746338,
  "rmse_price": 1.7648751735687256,
  "r2_price": 0.30262500047683716,
  "mae_atmos": 1.2458659410476685,
  "rmse_atmos": 1.6386243104934692,
  "r2_atmos": 0.3081444501876831,
  "mae_service": 1.2945499420166016,
  "rmse_service": 1.7338619232177734,
  "r2_service": 0.4136648178100586,
  "mae_overall": 1.0923101902008057,
  "rmse_overall": 1.484327793121338,
  "r2_overall": 0.4589042067527771,
  "mean_mae": 1.2327852964401245,
  "overall_mae": 1.0923101902008057,
  "aspect_mae": 1.2679040729999542
}
```

The notebook must handle possible key variations:

```text
mean_mae
overall_mae
mae_overall
aspect_mae
mae_food
mae_price
mae_atmos
mae_service
rmse_overall
r2_overall
```

If `overall_mae` is missing, use `mae_overall`.

If `mean_mae` is missing, compute it from available target MAEs.

If `aspect_mae` is missing, compute it from food/price/atmos/service MAEs.

---

# REQUIRED LOGIC

## 1. Auto-discover experiments

The notebook must scan:

```python
EXPERIMENTS_DIR
```

and find all subfolders.

For each folder:

- read `metrics.json`
- optionally read `test_metrics.json`
- optionally read `config.yaml`
- infer components from experiment name and config

---

## 2. Infer experiment metadata

For each experiment, infer:

```text
experiment_id
phase
image_backbone
text_backbone
fusion_method
loss_function
is_unimodal_text
is_unimodal_image
has_test_metrics
```

Use both:

1. `config.yaml` if available
2. experiment folder name fallback

Examples:

```text
EXP_020D_efficientnetb3_xlmr_concat_mse
```

should infer:

```text
phase = Image Backbone Ablation
image_backbone = EfficientNet-B3
text_backbone = XLM-R
fusion_method = Concat
loss_function = MSE
```

```text
EXP_040B_bestimage_besttext_gmu_mse
```

should infer:

```text
phase = Fusion Ablation
fusion_method = GMU
loss_function = MSE
```

```text
EXP_060E_convnext_phobert_gatedcrossmodal_autoweight
```

should infer:

```text
phase = Promising Combination
image_backbone = ConvNeXt
text_backbone = PhoBERT
fusion_method = Gated Cross-Modal
loss_function = AutoWeight
```

The notebook should include robust helper functions for parsing folder names.

---

## 3. Full Experiment Leaderboard

Create a dataframe sorted by:

```text
mean_mae ascending
```

Columns should include:

```text
rank
experiment_id
phase
image_backbone
text_backbone
fusion_method
loss_function
mean_mae
overall_mae
aspect_mae
mae_food
mae_price
mae_service
mae_atmos
rmse_overall
r2_overall
has_test_metrics
```

Save to CSV and Excel.

Also display top 10 in notebook.

---

## 4. Overall Leaderboard Figure

Create a horizontal bar chart of top 15 experiments by `mean_mae`.

Use matplotlib only.

Do not use seaborn.

Annotate each bar with the `mean_mae` value.

Save as PNG.

---

## 5. Image Backbone Comparison

Use experiments from image ablation phase.

Include experiments whose IDs contain:

```text
EXP_020
```

or whose phase is inferred as:

```text
Image Backbone Ablation
```

Compare `mean_mae` by image backbone.

If multiple experiments have the same backbone, keep the best one.

Save:

```text
image_backbone_comparison.csv
image_backbone_comparison.png
```

---

## 6. Text Backbone Comparison

Use experiments from text ablation phase.

Include experiments whose IDs contain:

```text
EXP_030
```

or whose phase is inferred as:

```text
Text Backbone Ablation
```

Compare `mean_mae` by text backbone.

If multiple experiments have the same text backbone, keep the best one.

Save:

```text
text_backbone_comparison.csv
text_backbone_comparison.png
```

---

## 7. Fusion Comparison

Use experiments from fusion phase.

Include experiments whose IDs contain:

```text
EXP_040
EXP_041
```

or whose phase is inferred as fusion-related.

Compare `mean_mae` by fusion method.

If multiple experiments use the same fusion method, keep the best one.

Save:

```text
fusion_comparison.csv
fusion_comparison.png
```

---

## 8. Loss Comparison

Use experiments from loss phase.

Include experiments whose IDs contain:

```text
EXP_050
EXP_051
```

or whose phase is inferred as loss-related.

Compare `mean_mae` by loss function.

If multiple experiments use the same loss, keep the best one.

Save:

```text
loss_comparison.csv
loss_comparison.png
```

---

## 9. Performance Evolution Across Phases

Create one table and one line plot showing best `mean_mae` by phase.

Phases should be ordered logically:

```text
Baseline
Image Ablation
Text Ablation
Fusion Ablation
Loss Ablation
Promising Combination
```

For each phase, select the best experiment by `mean_mae`.

Save:

```text
performance_evolution.csv
performance_evolution.png
```

---

## 10. Top-3 Radar Chart

Take top 3 experiments by `mean_mae`.

Plot a radar chart over five target MAEs:

```text
overall
food
price
service
atmosphere
```

Since lower MAE is better, either:

- plot inverted normalized score, or
- clearly label that lower value is better.

Prefer normalized score:

```text
score = 1 - normalized_mae
```

Save:

```text
top3_radar_chart.png
```

---

## 11. Ablation Summary Table

Create a table summarizing each ablation phase:

```text
phase
reference_experiment
best_experiment
reference_mean_mae
best_mean_mae
absolute_improvement
relative_improvement_percent
interpretation
```

For example:

```text
Image Ablation:
reference = EXP_012_multimodal_convnext_xlmr_concat_mse
best = best among EXP_020*
```

Text Ablation:

```text
reference = best image config with XLM-R
best = best among EXP_030*
```

Fusion Ablation:

```text
reference = concat
best = best among EXP_040* and EXP_041*
```

Loss Ablation:

```text
reference = MSE
best = best among EXP_050* and EXP_051*
```

Promising Combination:

```text
reference = best sequential config
best = best among EXP_060*
```

Save to CSV and Excel.

---

## 12. Validation vs Test Comparison

For experiments with `test_metrics.json`, create:

```text
experiment_id
phase
val_mean_mae
test_mean_mae
val_overall_mae
test_overall_mae
generalization_gap_mean_mae
generalization_gap_overall_mae
```

Save to CSV and Excel.

Create a bar chart if at least two experiments have test metrics.

Save:

```text
validation_vs_test_comparison.png
```

---

## 13. Improvement vs Baseline

Use:

```text
EXP_012_multimodal_convnext_xlmr_concat_mse
```

as the main baseline if it exists.

Otherwise use the earliest multimodal concat MSE experiment.

For every experiment, compute:

```text
absolute_improvement = baseline_mean_mae - experiment_mean_mae
relative_improvement_percent = absolute_improvement / baseline_mean_mae * 100
```

Save:

```text
improvement_vs_baseline.csv
improvement_vs_baseline.xlsx
improvement_vs_baseline.png
```

---

# OUTPUT FOLDER STRUCTURE

The notebook should create:

```text
/content/drive/MyDrive/SE365/reports/
├── figures/
│   ├── 01_overall_leaderboard.png
│   ├── 02_image_backbone_comparison.png
│   ├── 03_text_backbone_comparison.png
│   ├── 04_fusion_comparison.png
│   ├── 05_loss_comparison.png
│   ├── 06_performance_evolution.png
│   ├── 07_top3_radar_chart.png
│   ├── validation_vs_test_comparison.png
│   └── improvement_vs_baseline.png
│
├── tables/
│   ├── full_experiment_leaderboard.csv
│   ├── full_experiment_leaderboard.xlsx
│   ├── ablation_summary.csv
│   ├── ablation_summary.xlsx
│   ├── validation_vs_test_comparison.csv
│   ├── validation_vs_test_comparison.xlsx
│   ├── improvement_vs_baseline.csv
│   └── improvement_vs_baseline.xlsx
│
└── experiment_report.md
```

---

# NOTEBOOK REQUIREMENTS

The generated notebook must include:

## Section 1: Setup

- Mount Google Drive.
- Define paths.
- Create report folders.
- Import required libraries.

## Section 2: Experiment Discovery

- Discover experiment folders.
- Load metrics.
- Load test metrics if available.
- Load config if available.
- Build dataframe.

## Section 3: Metadata Parsing

- Infer phase, image backbone, text backbone, fusion, and loss.

## Section 4: Leaderboard

- Build full leaderboard.
- Save CSV/XLSX.
- Plot overall leaderboard.

## Section 5: Ablation Analysis

- Image comparison.
- Text comparison.
- Fusion comparison.
- Loss comparison.

## Section 6: Performance Evolution

- Best result per phase.
- Line plot.

## Section 7: Top-3 Radar Chart

- Radar chart for top 3.

## Section 8: Validation vs Test

- Use `test_metrics.json` where available.

## Section 9: Improvement vs Baseline

- Compare all experiments against baseline.

## Section 10: Markdown Report

- Generate `experiment_report.md`.
- Include summary of best model, best phase results, baseline improvement, and test comparison.

## Section 11: Logic Checks

Before finishing, include notebook cells that check:

- number of discovered experiments
- number of loaded metrics
- missing metrics files
- duplicated experiment IDs
- missing required metric columns
- whether baseline exists
- whether Phase 6 test metrics exist
- whether output files were successfully saved

---

# CONSTRAINTS

Do NOT run the notebook.

Only create the `.ipynb` file.

After creating the notebook, inspect its own logic.

Check for:

- path errors
- missing imports
- invalid JSON/YAML handling
- invalid dataframe columns
- divide-by-zero risk
- missing metric key handling
- matplotlib save path issues
- radar chart logic
- baseline fallback logic
- empty dataframe handling
- Excel writer dependency issues

If there is any issue, fix the notebook before finalizing.

Do not assume all experiments have the same metric keys.

Do not assume all experiments have test metrics.

Do not assume all configs exist.

Do not use seaborn.

Use matplotlib and pandas.

Make the notebook robust and readable.

---

# FORMAT PRINCIPLE

The notebook must be written as a professional research analysis notebook.

It should be easy to read, easy to rerun, and easy to modify.

Use Markdown cells to explain:

- what each section does
- what output is generated
- how to interpret each chart/table

Use code cells with clear comments.

Make all paths configurable at the top.

The final notebook should allow me to run all cells in Google Colab and automatically generate all required tables and figures into:

```text
/content/drive/MyDrive/SE365/reports/
```
