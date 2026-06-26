# Phase 8: Thesis-Ready Visualization -- Implementation Proposal

---

## 1. Purpose

### Why This Phase Exists

Phases 2 through 7 produce draft-quality XAI visualizations at 150 DPI with inconsistent styling, ad hoc color choices, variable font sizes, and no unified layout. These figures are suitable for iterative development and debugging, but they are not suitable for insertion into a thesis document, a journal submission, or a defense presentation slide deck. Phase 8 exists to transform every draft XAI artifact into a publication-ready, thesis-grade figure with enforced consistency, professional typography, high resolution, correct Vietnamese text rendering, and standardized sizing.

### Research Motivation

A thesis examining explainability of a multimodal deep learning system for Vietnamese restaurant review quality assessment must present visual evidence that is clear, credible, and reproducible. Figures that use inconsistent fonts, misaligned color scales, unreadable text, or pixelated images undermine the scientific credibility of the work. Publication-ready figures allow the examiner to focus on the explanation content rather than the presentation artifacts. Additionally, journals require specific figure resolutions (typically 300 DPI minimum), colorblind-accessible palettes, and grayscale-safe designs. Phase 8 fulfills all of these requirements systematically.

### Engineering Motivation

Without a centralized style configuration, each phase independently makes styling decisions. Phase 2 may use `'jet'` colormap while Phase 5 uses `'hot'`. Phase 3 may set font size 12 while Phase 4 uses 8. Phase 6 case study panels may use different color assignments for modalities than Phase 4 SHAP charts. This inconsistency is not only visually distracting but operationally expensive: fixing it manually for each figure in each phase is error-prone and non-reproducible.

Phase 8 solves this by:
1. Defining a single source of truth for all visual styling parameters.
2. Providing figure-generation functions that read pre-computed raw XAI data (NPY, JSON, CSV) and render them with thesis-grade formatting.
3. Generating both thesis-format and slide-format variants from the same data.
4. Producing a machine-readable figure catalog that maps each figure to its source data, sample ID, target, and XAI method.

This phase does NOT re-run any XAI computation. It only re-renders pre-computed data.

---

## 2. Objectives

### Research Objectives

1. **RO-1:** Ensure that all XAI evidence figures in the thesis use identical visual conventions, so that a reader can compare Grad-CAM, attention, SHAP, and LIME results without being distracted by stylistic differences.
2. **RO-2:** Guarantee that Vietnamese diacritical marks (a, a, d, e, o, o, u and all tone marks) render correctly in all token-level visualizations (attention bar charts, attention heatmaps, LIME text explanations).
3. **RO-3:** Produce figures that remain interpretable when printed in grayscale, satisfying journal submission requirements.
4. **RO-4:** Generate multi-panel composite figures that combine multiple XAI methods for the same sample, enabling side-by-side comparison within a single thesis figure.
5. **RO-5:** Create a modality contribution profile across all five targets (food_score, price_score, atmosphere_score, service_score, overall_satisfaction) as a summary visualization suitable for the Results chapter.

### Engineering Objectives

1. **EO-1:** Create `xai/thesis_style.py` as the single source of truth for all matplotlib rcParams, colormaps, font configurations, figure sizes, and color palettes.
2. **EO-2:** Create `xai/thesis_figures.py` containing one rendering function per figure type, each reading raw XAI data from disk and producing PNG (300 DPI) and PDF outputs.
3. **EO-3:** Implement Vietnamese font detection and fallback logic that works across Windows, Linux, and Colab environments.
4. **EO-4:** Generate slide-format variants (larger fonts, wider spacing, white backgrounds) with a `_slide` suffix.
5. **EO-5:** Produce a `figure_catalog.csv` that catalogs every generated figure with metadata.
6. **EO-6:** Create a Phase 8 notebook that regenerates all thesis figures from pre-computed data in a single run.

### Expected Contributions

- A reproducible, single-command pipeline that transforms all raw XAI outputs into thesis-ready figures.
- A visual style guide embedded in code that enforces consistency without manual intervention.
- Figures that are directly insertable into LaTeX documents (via `\includegraphics`) and PowerPoint/Google Slides (via PNG).
- A figure catalog that accelerates thesis writing by providing instant lookup of the correct figure for any discussion point.

---

## 3. Inputs

### Raw XAI Data from Previous Phases

All inputs are pre-computed artifacts stored in the experiment XAI directory. Phase 8 reads raw numerical data, NOT draft-quality PNG figures.

| Input | Path Pattern | Format | Source Phase | Description |
|---|---|---|---|---|
| Grad-CAM heatmaps | `experiments/EXP_060A_.../xai/gradcam/gradcam_sample{id}_target{idx}.npy` | NPY `[H, W]` normalized to [0,1] | Phase 2 | Per-target Grad-CAM activation maps |
| Grad-CAM metadata | `experiments/EXP_060A_.../xai/gradcam/gradcam_metadata.json` | JSON | Phase 2 | Sample IDs, target indices, image paths, prediction values |
| Attention weights | `experiments/EXP_060A_.../xai/attention/attention_sample{id}.npz` | NPZ | Phase 3 | CLS attention weights, token-to-token matrices, token lists |
| Attention metadata | `experiments/EXP_060A_.../xai/attention/attention_metadata.json` | JSON | Phase 3 | Token lists, layer/head info, sample IDs |
| SHAP values | `experiments/EXP_060A_.../xai/shap/shap_values_sample{id}.npy` | NPY `[num_targets, fused_dim]` | Phase 4 | Per-target SHAP values on fused embedding |
| SHAP modality summary | `experiments/EXP_060A_.../xai/shap/shap_modality_summary.csv` | CSV | Phase 4 | Text% vs Image% per target, aggregated across samples |
| SHAP base values | `experiments/EXP_060A_.../xai/shap/shap_base_values.npy` | NPY `[num_targets]` | Phase 4 | SHAP expected values per target |
| LIME image weights | `experiments/EXP_060A_.../xai/lime/lime_image_sample{id}_target{idx}.npz` | NPZ | Phase 5 | Superpixel masks and weights |
| LIME text weights | `experiments/EXP_060A_.../xai/lime/lime_text_sample{id}_target{idx}.json` | JSON | Phase 5 | Word-weight pairs |
| Case study metadata | `experiments/EXP_060A_.../xai/case_studies/case_study_manifest.json` | JSON | Phase 6 | List of case studies with sample IDs, case type, relevant targets |
| Case study reports | `experiments/EXP_060A_.../xai/case_studies/case_{type}_sample{id}.json` | JSON | Phase 6 | Combined XAI results for each case study |
| Phase 7 report data | `experiments/EXP_060A_.../xai/reports/report_data.json` | JSON | Phase 7 | Aggregated statistics, cross-method comparisons |

### Original Images and Text

| Input | Path | Description |
|---|---|---|
| Image cache | `./data/image/` | JPEG files named by MD5 hash of original URL |
| Validation CSV | `./data/text/val.csv` | Contains `comment_clean`, `image_url`, ground truth scores |
| Test CSV | `./data/text/test.csv` | Same schema |

### Infrastructure

| Input | Path | Description |
|---|---|---|
| XAI config | `xai/config.py` | Constants: `TARGET_NAMES`, `FACTOR_NAMES`, `DISPLAY_NAMES`, `COLOR_SCHEMES`, etc. |
| XAI utilities | `xai/utils.py` | `load_single_sample()` for loading original images and text |

### Predictions

| Input | Path | Description |
|---|---|---|
| Validation predictions | `experiments/EXP_060A_.../predictions.csv` | Ground truth and predicted values for all validation samples |
| Test predictions | `experiments/EXP_060A_.../test_predictions.csv` | Same for test set |

---

## 4. Outputs

### Python Modules

| Output | Path | Description |
|---|---|---|
| Style configuration | `xai/thesis_style.py` | Central style dict, `apply_thesis_style()`, `apply_slide_style()`, colormap registry, font configuration, figure size classes |
| Figure generators | `xai/thesis_figures.py` | One function per figure type. Each loads raw data, applies style, renders figure, saves PNG+PDF |

### Thesis-Quality Figures

All saved under `experiments/EXP_060A_.../xai/thesis_figures/`. Both PNG (300 DPI, raster) and PDF (vector) formats where applicable. See Section 8 for complete folder structure.

### Slide-Format Figures

Same figures re-rendered with larger fonts and slide-optimized sizing. Saved under `experiments/EXP_060A_.../xai/thesis_figures/slides/`.

### Figure Catalog

| Output | Path | Description |
|---|---|---|
| Figure catalog | `experiments/EXP_060A_.../xai/thesis_figures/figure_catalog.csv` | Columns: `filename`, `figure_type`, `format`, `sample_id`, `target`, `xai_method`, `description`, `source_phase`, `width_inches`, `height_inches`, `dpi` |

### Style Config Export

| Output | Path | Description |
|---|---|---|
| Style config JSON | `experiments/EXP_060A_.../xai/thesis_figures/style_config.json` | Serialized version of all style parameters for reproducibility documentation |

### Notebook

| Output | Path | Description |
|---|---|---|
| Phase 8 notebook | `xai/notebooks/Phase8_ThesisVisualization.ipynb` | Runnable notebook that regenerates all thesis figures |

---

## 5. Architecture Attachment Point

Phase 8 has **no model attachment**. It does not load, run, or interact with the trained model in any way. This phase operates entirely on pre-computed XAI artifacts saved to disk by Phases 2 through 7.

The relationship to the model architecture is indirect: Phase 8 renders visualizations that were computed at specific architectural attachment points by earlier phases:

```
Phase 8 reads artifacts FROM:

  Phase 2 (Grad-CAM)      -> Swin-B last spatial feature map [B, 1024, 7, 7]
  Phase 3 (Attention)      -> PhoBERT attention matrices [12 layers x [B, 12, L, L]]
  Phase 4 (SHAP)           -> Cross-Attention fused embedding [B, 1024] -> head
  Phase 5 (LIME)           -> Full model predict function (image superpixels / text words)
  Phase 6 (Case Studies)   -> Combined multi-method results per sample
  Phase 7 (Reports)        -> Aggregated cross-sample statistics

Phase 8 attachment: NONE (pure visualization of saved data)
```

The only codebase interaction is importing constants from `xai/config.py` (target names, display names, color schemes, factor names) and utility functions from `xai/utils.py` (for loading original images and text when compositing overlay figures).

---

## 6. Detailed Implementation Plan

### A. Create `xai/thesis_style.py` -- Central Style Configuration

This module is the single source of truth for all visual styling across thesis figures. No figure-generation function should contain hardcoded style values.

#### A1. THESIS_STYLE Dictionary

Define a dictionary `THESIS_STYLE` containing all matplotlib rcParams overrides for thesis-format output:

```
THESIS_STYLE = {
    # Resolution
    'figure.dpi': 300,

    # Default figure size (inches) -- overridden per figure type
    'figure.figsize': (7, 5),

    # Font
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Noto Serif', 'Times New Roman', 'serif'],
    'font.size': 10,

    # Axes
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.8,
    'axes.grid': False,

    # Tick marks
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,

    # Legend
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',

    # Lines and patches
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,

    # Figure background
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}
```

#### A2. SLIDE_STYLE Dictionary

Define a separate dictionary `SLIDE_STYLE` with presentation-optimized parameters:

```
SLIDE_STYLE = {
    'figure.dpi': 150,
    'figure.figsize': (10, 6),
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Noto Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
}
```

#### A3. FIGURE_SIZES Dictionary

Define standard figure size classes (width, height in inches) that each figure function references:

```
FIGURE_SIZES = {
    'single_column':   (3.5, 2.8),    # Single-column journal figure
    'double_column':   (7.0, 4.0),    # Double-column journal figure
    'full_page':       (7.0, 9.0),    # Full-page thesis figure
    'thesis_wide':     (6.5, 4.0),    # Standard thesis-width figure
    'thesis_tall':     (6.5, 6.0),    # Tall thesis figure
    'panel_5col':      (14.0, 3.0),   # 5-target Grad-CAM panel (1x6 grid)
    'panel_2x2':       (7.0, 7.0),    # 2x2 case study panel
    'panel_3x2':       (7.0, 10.0),   # 3x2 case study panel
    'panel_2x3':       (10.5, 7.0),   # 2x3 comparison panel
    'heatmap_square':  (6.0, 5.5),    # Square attention heatmap
    'bar_horizontal':  (6.5, 4.5),    # Horizontal bar chart
    'radar':           (5.5, 5.5),    # Radar/spider chart
    'slide_standard':  (10.0, 6.0),   # Slide figure
    'slide_wide':      (12.0, 5.0),   # Wide slide figure
}
```

#### A4. COLOR_PALETTE -- Colorblind-Friendly and Grayscale-Safe

Define a colorblind-friendly palette. Use the Okabe-Ito palette as the primary categorical palette, which is designed for color vision deficiency accessibility:

```
COLORBLIND_PALETTE = {
    'orange':     '#E69F00',
    'sky_blue':   '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'purple':     '#CC79A7',
    'black':      '#000000',
}
```

Define modality-specific colors using distinct, colorblind-safe assignments:

```
MODALITY_COLORS = {
    'text':  '#0072B2',    # Blue (Okabe-Ito blue)
    'image': '#D55E00',    # Vermillion (Okabe-Ito vermillion)
}
```

Define target-specific colors using the five most distinguishable colors from an extended colorblind-safe set:

```
TARGET_COLORS = {
    'food':     '#E69F00',    # Orange
    'price':    '#009E73',    # Green
    'atmos':    '#56B4E9',    # Sky blue
    'service':  '#CC79A7',    # Purple
    'overall':  '#D55E00',    # Vermillion
}
```

Define method-specific colormaps:

```
METHOD_CMAPS = {
    'gradcam':     'inferno',     # Sequential, perceptually uniform, grayscale-safe
    'attention':   'YlOrRd',      # Sequential warm, good for attention weights
    'shap_pos':    '#D55E00',     # Vermillion for positive SHAP
    'shap_neg':    '#0072B2',     # Blue for negative SHAP
    'shap_cmap':   'RdBu_r',     # Diverging for SHAP (reversed: red=positive, blue=negative)
    'lime_pos':    '#009E73',     # Green for positive LIME
    'lime_neg':    '#D55E00',     # Vermillion for negative LIME
}
```

**Design decision on Grad-CAM colormap:** Use `'inferno'` instead of `'jet'`. Rationale: `'jet'` is not perceptually uniform and creates misleading intensity bands in grayscale. `'inferno'` is perceptually uniform, prints well in grayscale (monotonically increasing luminance), and is colorblind-friendly. This overrides the `'jet'` default from `xai/config.py` `COLOR_SCHEMES['gradcam_cmap']` specifically for thesis figures. Draft figures (Phases 2-5) may continue using `'jet'` for quick inspection, but thesis figures must use `'inferno'`.

#### A5. HATCH_PATTERNS Dictionary

For grayscale-safe bar charts, define hatching patterns to supplement color:

```
HATCH_PATTERNS = {
    'text':  '///',
    'image': '...',
    'food':  '',
    'price': '///',
    'atmos': '\\\\\\',
    'service': '...',
    'overall': 'xxx',
}
```

#### A6. `apply_thesis_style()` Function

- Signature: `apply_thesis_style() -> None`
- Logic:
  1. Call `matplotlib.rcParams.update(THESIS_STYLE)`.
  2. Call `_configure_vietnamese_font()` (see A8).
  3. Print confirmation: `"Thesis style applied: 300 DPI, serif font, 10pt base"`.

#### A7. `apply_slide_style()` Function

- Signature: `apply_slide_style() -> None`
- Logic:
  1. Call `matplotlib.rcParams.update(SLIDE_STYLE)`.
  2. Call `_configure_vietnamese_font()`.
  3. Print confirmation: `"Slide style applied: 150 DPI, sans-serif font, 14pt base"`.

#### A8. `_configure_vietnamese_font()` Private Function

- Signature: `_configure_vietnamese_font() -> str`
- Logic:
  1. Define a test string containing all challenging Vietnamese characters: `"Dong an ngon nhung gia hoi cao. Khong gian dep. Dich vu tot."` (with full diacritics: `"Do an ngon nhung gia hoi cao. Khong gian dep. Dich vu tot."` -- the actual Vietnamese: `"Dò ăn ngon nhuưng giá hoơi cao. Không gian dẹp. Dịch vụ tót."`).
  2. Attempt font resolution in priority order:
     - **Priority 1:** `'DejaVu Sans'` -- bundled with matplotlib, supports Vietnamese diacritics. Test by checking `matplotlib.font_manager.findfont('DejaVu Sans')` does not return a fallback.
     - **Priority 2:** `'Noto Sans'` or `'Noto Serif'` -- Google's pan-Unicode font. May be installed on the system.
     - **Priority 3:** `'Arial Unicode MS'` -- common on Windows systems with Office installed.
     - **Priority 4:** matplotlib default -- as a last resort.
  3. For each candidate, attempt to render the test string on a throwaway figure. Catch any font substitution warnings from matplotlib.
  4. Set `matplotlib.rcParams['font.sans-serif']` or `matplotlib.rcParams['font.serif']` to include the validated font at the front of the list.
  5. Return the name of the font that was successfully configured.
  6. Print the result: `f"Vietnamese font configured: {font_name}"`.

#### A9. `get_cmap(method)` Function

- Signature: `get_cmap(method: str) -> matplotlib.colors.Colormap`
- Logic: Look up `method` in `METHOD_CMAPS` and return `matplotlib.colormaps[cmap_name]`. For SHAP, return the diverging colormap. For Grad-CAM and attention, return the sequential colormap.

#### A10. `get_figure_size(size_class, style)` Function

- Signature: `get_figure_size(size_class: str, style: str = 'thesis') -> tuple[float, float]`
- Logic:
  1. If `style == 'slide'`, map each thesis size class to its slide equivalent (use `slide_standard` or `slide_wide` as appropriate).
  2. If `style == 'thesis'`, look up `FIGURE_SIZES[size_class]` directly.
  3. Return `(width, height)` in inches.

#### A11. `save_thesis_figure(fig, base_path, formats, dpi, catalog_entry)` Function

- Signature: `save_thesis_figure(fig, base_path: str, formats: list[str] = ['png', 'pdf'], dpi: int = 300, catalog_entry: dict = None) -> list[str]`
- Logic:
  1. For each format in `formats`:
     - Save `fig.savefig(base_path + '.' + fmt, dpi=dpi, bbox_inches='tight', facecolor='white')`.
  2. If `catalog_entry` is not None, append it to the in-memory catalog list (module-level list `_CATALOG_ENTRIES`).
  3. Return list of saved file paths.

#### A12. `export_figure_catalog(output_path)` Function

- Signature: `export_figure_catalog(output_path: str) -> str`
- Logic:
  1. Convert `_CATALOG_ENTRIES` list to a pandas DataFrame.
  2. Save as CSV to `output_path`.
  3. Return `output_path`.

---

### B. Create `xai/thesis_figures.py` -- Figure Generation Functions

Each function below follows the same contract:
1. Accept an experiment directory path and relevant identifiers (sample_id, target_idx).
2. Load raw XAI data from the experiment's `xai/` subdirectory.
3. Load original images or text via `xai.utils.load_single_sample()` when needed for overlays.
4. Apply thesis style (caller must have called `apply_thesis_style()` first).
5. Create the figure using matplotlib.
6. Save to `thesis_figures/` subdirectory.
7. Return the figure object and the list of saved paths.
8. Register the figure in the catalog.

#### B1. `fig_gradcam_5target_panel(sample_id, exp_dir, csv_path, image_dir)`

**Purpose:** Create a single composite figure showing the original image alongside Grad-CAM overlays for all 5 targets.

**Layout:** 1 row x 6 columns: [Original Image | Food | Price | Atmos | Service | Overall]

**Figure size:** `panel_5col` = (14, 3) inches.

**Implementation steps:**
1. Load original image for `sample_id` from CSV and image cache.
2. For each target index 0 through 4:
   a. Load Grad-CAM heatmap from `xai/gradcam/gradcam_sample{sample_id}_target{idx}.npy`.
   b. Normalize to [0, 1] if not already.
   c. Resize heatmap to original image dimensions using bilinear interpolation.
3. Create figure with 6 subplots using `plt.subplots(1, 6, figsize=...)`.
4. Subplot 0: Display original image. Title: `"Original"`.
5. Subplots 1-5: Display original image with Grad-CAM overlay using `'inferno'` colormap at `alpha=0.5`. Title: display name of each target (e.g., `"Food Score"`).
6. Add a single shared colorbar at the right side of the figure. Colorbar label: `"Activation Intensity"`.
7. Remove individual axis ticks from all subplots.
8. Add a super-title with sample ID information.
9. Save as PNG (300 DPI) and PDF.
10. Register in catalog: `figure_type='gradcam_5target_panel'`, `xai_method='gradcam'`.

**Save path:** `thesis_figures/gradcam/fig_gradcam_5target_sample{id}.{png,pdf}`

#### B2. `fig_gradcam_multi_image(sample_id, exp_dir, target_idx, csv_path, image_dir)`

**Purpose:** For multi-image reviews, show Grad-CAM for each real image (up to 4) for a single target.

**Layout:** 1 row x N columns where N = number of real images for this review.

**Figure size:** Dynamically computed: `(3.5 * N, 3.0)` inches.

**Implementation steps:**
1. Load sample metadata to determine `num_real_images`.
2. For each real image (index 0 to num_real_images - 1):
   a. Load the original image.
   b. Load the per-image Grad-CAM heatmap: `gradcam_sample{id}_img{img_idx}_target{target_idx}.npy`.
   c. Overlay heatmap on image.
3. Create subplots: `plt.subplots(1, N)`.
4. Title each subplot: `f"Image {img_idx+1}"`.
5. Add super-title: `f"Grad-CAM for {DISPLAY_NAMES[target_idx]} (Sample {sample_id})"`.
6. Add shared colorbar.
7. Save as PNG only (contains photographic content; PDF would be excessively large).

**Save path:** `thesis_figures/gradcam/fig_gradcam_multiimage_sample{id}_target{idx}.png`

#### B3. `fig_attention_token_bar(sample_id, exp_dir)`

**Purpose:** Horizontal bar chart showing the top-15 tokens by CLS attention weight, with Vietnamese text labels.

**Layout:** Single horizontal bar chart.

**Figure size:** `bar_horizontal` = (6.5, 4.5) inches.

**Implementation steps:**
1. Load attention data from `xai/attention/attention_sample{id}.npz`.
2. Extract CLS-to-token attention weights (row 0 of the last-layer mean-across-heads attention matrix). Shape: `[L]`.
3. Load the token list from the metadata.
4. Exclude special tokens (`<s>`, `</s>`, `<pad>`).
5. Sort remaining tokens by attention weight in descending order.
6. Select top 15 tokens.
7. Create horizontal bar chart using `plt.barh()`.
8. Color-code bars by aspect relevance (optional but recommended):
   - Use TARGET_COLORS for tokens matching known aspect keywords.
   - Food-related: tokens containing `ngon`, `do an`, `mon`, `banh`, etc.
   - Price-related: `gia`, `dat`, `re`, `tien`, etc.
   - Atmosphere-related: `khong gian`, `dep`, `sach`, `am cung`, etc.
   - Service-related: `phuc vu`, `nhan vien`, `thai do`, etc.
   - For tokens not matching any category, use neutral gray `'#666666'`.
9. Add Vietnamese token labels on the y-axis. Ensure proper rendering with configured font.
10. X-axis label: `"CLS Attention Weight"`.
11. Y-axis label: `"Token"`.
12. Title: `f"Top-15 Tokens by CLS Attention (Sample {sample_id})"`.
13. Add a small legend mapping colors to aspect categories.
14. Save as PNG and PDF.

**Save path:** `thesis_figures/attention/fig_attention_bar_sample{id}.{png,pdf}`

#### B4. `fig_attention_heatmap(sample_id, exp_dir, layer_idx, num_display_tokens)`

**Purpose:** Token-to-token attention heatmap for a specific layer, showing interaction patterns.

**Layout:** Square heatmap with Vietnamese token labels on both axes.

**Figure size:** `heatmap_square` = (6.0, 5.5) inches.

**Parameters:**
- `layer_idx`: which transformer layer to visualize (default: 11, the last PhoBERT layer).
- `num_display_tokens`: maximum tokens to display (default: 30, to keep labels readable).

**Implementation steps:**
1. Load attention matrix from `xai/attention/attention_sample{id}.npz`.
2. Extract the specified layer's attention, averaged across heads: shape `[L, L]`.
3. Load token list from metadata.
4. Trim to `num_display_tokens` (from position 0 up to `num_display_tokens`, excluding padding tokens).
5. Trim the attention matrix accordingly: `attn[:num_display_tokens, :num_display_tokens]`.
6. Create figure and axis.
7. Use `sns.heatmap()` or `plt.imshow()` with `'YlOrRd'` colormap.
8. Set x-tick labels to tokens, rotated 45 degrees, horizontally aligned to the right.
9. Set y-tick labels to tokens (not rotated).
10. Add colorbar with label: `"Attention Weight"`.
11. Title: `f"Attention Heatmap (Layer {layer_idx+1}, Sample {sample_id})"`.
12. Adjust layout to prevent label clipping.
13. Save as PNG and PDF.

**Save path:** `thesis_figures/attention/fig_attention_heatmap_sample{id}.{png,pdf}`

#### B5. `fig_shap_modality_summary(exp_dir)`

**Purpose:** Grouped bar chart showing text% vs image% modality contribution across all 5 targets, aggregated across samples.

**Layout:** Grouped bar chart with 5 groups (one per target) and 2 bars per group (Text, Image).

**Figure size:** `thesis_wide` = (6.5, 4.0) inches.

**Implementation steps:**
1. Load `xai/shap/shap_modality_summary.csv`.
2. Extract columns: target name, text_percentage, image_percentage. If standard deviation columns exist (from multiple samples), also extract those for error bars.
3. Set bar positions: 5 groups, 2 bars each, with appropriate spacing.
4. Plot text bars using `MODALITY_COLORS['text']` with hatch pattern `HATCH_PATTERNS['text']`.
5. Plot image bars using `MODALITY_COLORS['image']` with hatch pattern `HATCH_PATTERNS['image']`.
6. Add error bars if available (using standard error or standard deviation).
7. X-axis labels: `DISPLAY_NAMES` for each target.
8. Y-axis label: `"Modality Contribution (%)"`.
9. Set y-axis range: `[0, 100]` with gridlines at 25% intervals.
10. Add horizontal line at 50% (dashed, gray) to indicate equal contribution.
11. Add value labels above each bar (e.g., `"62.3%"`).
12. Legend: `["Text", "Image"]` with corresponding colors and hatches.
13. Title: `"Modality Contribution by Target Score"`.
14. Save as PNG and PDF.

**Save path:** `thesis_figures/shap/fig_shap_modality_summary.{png,pdf}`

#### B6. `fig_shap_waterfall(sample_id, exp_dir, target_idx)`

**Purpose:** SHAP waterfall plot for a single sample and target, showing top contributing features.

**Layout:** Standard SHAP waterfall (horizontal bars from base value to predicted value).

**Figure size:** `thesis_wide` = (6.5, 4.0) inches.

**Implementation steps:**
1. Load SHAP values from `xai/shap/shap_values_sample{id}.npy`.
2. Load base values from `xai/shap/shap_base_values.npy`.
3. Extract the SHAP values for `target_idx`: shape `[fused_dim]` (e.g., `[1024]` for CrossAttentionFusion).
4. Create feature labels:
   - For CrossAttentionFusion: the fused embedding is `[text_proj_out(512) ; image_proj_out(512)]` after cross-attention. Label the first 512 dimensions as `"T2I_dim_{i}"` and the next 512 as `"I2T_dim_{i}"`.
   - Alternatively, group features into `"Text-attended"` and `"Image-attended"` segments and show the top-N individual features plus grouped modality contributions.
5. Select top 15 features by absolute SHAP value.
6. Use the `shap` library's `shap.plots.waterfall()` if available and compatible with thesis styling, OR implement a custom waterfall using matplotlib:
   a. Sort the top features by SHAP value magnitude.
   b. Draw horizontal bars from the cumulative base, colored red (positive) or blue (negative).
   c. Add connecting lines between bars.
   d. Label each bar with the feature name and SHAP value.
   e. Mark the base value and final predicted value.
7. Title: `f"SHAP Waterfall for {DISPLAY_NAMES[target_idx]} (Sample {sample_id})"`.
8. Save as PNG and PDF.

**Save path:** `thesis_figures/shap/fig_shap_waterfall_sample{id}_target{idx}.{png,pdf}`

#### B7. `fig_case_study_panel(sample_id, case_type, exp_dir, csv_path, image_dir)`

**Purpose:** Multi-panel composite figure combining all XAI methods for one case study sample. This is the flagship thesis figure that demonstrates the multi-level explainability approach.

**Layout options based on available data:**
- If all 4 methods available: 2x3 grid using `panel_2x3` = (10.5, 7.0) inches.
- If 3 methods available: 2x2 grid using `panel_2x2` = (7.0, 7.0) inches.

**Panel assignments (2x3 layout):**
- Panel A (top-left): Original image + Grad-CAM overlay for the primary target.
- Panel B (top-center): Attention token bar chart (top-10 tokens, compact).
- Panel C (top-right): SHAP modality contribution (text% vs image%) for this sample across 5 targets.
- Panel D (bottom-left): LIME image explanation (positive/negative superpixels).
- Panel E (bottom-center): Prediction vs Ground Truth comparison table or bar chart.
- Panel F (bottom-right): LIME text explanation (top positive/negative words).

**Implementation steps:**
1. Load case study metadata from `xai/case_studies/case_{case_type}_sample{id}.json`.
2. Create figure with appropriate grid using `plt.subplots()` or `matplotlib.gridspec.GridSpec` for fine-grained control.
3. For each panel:
   a. Load the relevant raw data.
   b. Render using a compact version of the corresponding single-figure function.
   c. Add panel label (A, B, C, D, E, F) in the top-left corner of each subplot, bold, 12pt.
4. Add super-title: `f"Case Study: {case_type_display} (Sample {sample_id})"`.
   - `case_type_display` mapping: `'correct'` -> `'Correct Prediction'`, `'high_error'` -> `'High Error Case'`, `'conflict'` -> `'Modality Conflict'`.
5. Adjust spacing: `plt.subplots_adjust(hspace=0.3, wspace=0.25)`.
6. Save as PNG and PDF.

**Save path:** `thesis_figures/case_studies/fig_case_{case_type}_{sample_id}.{png,pdf}`

#### B8. `fig_modality_profile_radar(exp_dir)`

**Purpose:** Radar/spider chart showing the modality balance (text% contribution) across all 5 targets. Alternatively, if radar readability is poor, fall back to a grouped bar chart.

**Layout:** Single radar chart.

**Figure size:** `radar` = (5.5, 5.5) inches.

**Implementation steps:**
1. Load `xai/shap/shap_modality_summary.csv`.
2. Extract text_percentage for each target.
3. Create radar chart:
   a. Define 5 axes corresponding to 5 targets.
   b. Plot text contribution as one polygon (filled, semi-transparent blue).
   c. Plot image contribution as another polygon (filled, semi-transparent vermillion).
   d. Add radial gridlines at 20%, 40%, 60%, 80%, 100%.
   e. Label each axis with the display name.
4. Add legend: `["Text Contribution", "Image Contribution"]`.
5. Title: `"Modality Contribution Profile Across Targets"`.
6. Save as PNG and PDF.
7. **Fallback:** If the radar chart is difficult to read (determined by manual inspection during notebook execution), generate a grouped bar chart instead using the same data. The bar chart version follows the same pattern as `fig_shap_modality_summary` but uses a polar coordinate system is replaced by a standard Cartesian chart with a different visual arrangement (e.g., stacked bars to 100%).

**Save path:** `thesis_figures/shap/fig_modality_profile_radar.{png,pdf}`

#### B9. `fig_cross_method_comparison(sample_id, exp_dir, target_idx, csv_path, image_dir)`

**Purpose:** Side-by-side comparison of image explanations (Grad-CAM vs LIME Image) and text explanations (Attention vs LIME Text) for the same sample and target.

**Layout:** 2 rows x 2 columns.
- Row 1: Image explanations. [Grad-CAM Overlay | LIME Image]
- Row 2: Text explanations. [Attention Bar Chart | LIME Text Bar Chart]

**Figure size:** `panel_2x2` = (7.0, 7.0) inches.

**Implementation steps:**
1. Load Grad-CAM heatmap, LIME image weights, attention weights, and LIME text weights for the specified sample and target.
2. Panel (0,0): Grad-CAM overlay on original image. Title: `"Grad-CAM"`.
3. Panel (0,1): LIME image with positive superpixels highlighted in green, negative in red. Title: `"LIME Image"`.
4. Panel (1,0): Compact attention bar chart (top-10 tokens). Title: `"Attention"`.
5. Panel (1,1): LIME text bar chart (top-10 words by weight, green=positive, red=negative). Title: `"LIME Text"`.
6. Super-title: `f"Cross-Method Comparison for {DISPLAY_NAMES[target_idx]} (Sample {sample_id})"`.
7. Save as PNG (not PDF, due to photographic content).

**Save path:** `thesis_figures/comparison/fig_cross_method_sample{id}_target{idx}.png`

#### B10. `fig_architecture_xai_diagram(exp_dir)`

**Purpose:** Recreate the model architecture diagram with XAI attachment points clearly marked, suitable for the Methodology chapter.

**Layout:** Horizontal flowchart showing the data flow from inputs through encoders, fusion, and prediction head, with XAI attachment points annotated.

**Figure size:** `thesis_wide` = (6.5, 4.0) inches.

**Implementation approach:** Use matplotlib patches and arrows to construct the diagram programmatically. Do NOT rely on mermaid or external rendering.

**Implementation steps:**
1. Draw rectangular blocks for each component:
   - Input Images box
   - Swin-B Encoder box
   - Image Embedding `[B, 1024]` box
   - Input Text box
   - PhoBERT Encoder box
   - Text Embedding `[B, 768]` box
   - Cross-Attention Fusion box
   - Fused Embedding `[B, 1024]` box
   - MLP Prediction Head box
   - 5 output boxes for each target score
2. Draw arrows showing data flow.
3. Add XAI attachment point annotations:
   - Grad-CAM: dashed arrow from Swin-B to a side annotation box `"Grad-CAM [B, 1024, 7, 7]"`.
   - Attention: dashed arrow from PhoBERT to `"Attention [12, 12, L, L]"`.
   - SHAP: dashed arrow from Fused Embedding to `"SHAP [B, 1024]"`.
   - LIME: dashed arrows from Input Images and Input Text to `"LIME Perturbation"`.
4. Use consistent colors: blue for text path, vermillion for image path, gray for fusion path, dashed lines for XAI.
5. Title: `"Model Architecture with XAI Attachment Points"`.
6. Save as PDF (vector, no photographic content) and PNG.

**Save path:** `thesis_figures/comparison/fig_architecture_xai.{png,pdf}`

#### B11. `fig_prediction_comparison_bar(sample_id, exp_dir)`

**Purpose:** Simple grouped bar chart comparing ground truth and predicted values for all 5 targets for a single sample.

**Layout:** 5 groups x 2 bars (Ground Truth, Prediction).

**Figure size:** `thesis_wide` = (6.5, 4.0) inches.

**Implementation steps:**
1. Load predictions and ground truth for the sample from predictions.csv or case study metadata.
2. Plot grouped bars: Ground Truth in blue, Prediction in vermillion.
3. Add error annotation (absolute error) above each prediction bar.
4. X-axis: target display names. Y-axis: `"Score"` with range [0, 10].
5. Title: `f"Prediction vs Ground Truth (Sample {sample_id})"`.
6. Save as PNG and PDF.

**Save path:** `thesis_figures/case_studies/fig_prediction_bar_sample{id}.{png,pdf}`

---

### C. Create Slide-Format Variants

For every figure generated by functions B1 through B11, create a corresponding slide-format variant.

**Implementation approach:**
1. Each figure function accepts an optional `style` parameter: `'thesis'` (default) or `'slide'`.
2. When `style='slide'`:
   - Use `SLIDE_STYLE` rcParams instead of `THESIS_STYLE`.
   - Use `FIGURE_SIZES['slide_standard']` or `FIGURE_SIZES['slide_wide']` instead of the thesis size.
   - Increase all font sizes proportionally.
   - Increase line widths.
   - Set DPI to 150.
3. Save with `_slide` suffix: `fig_shap_modality_summary_slide.png`.
4. Slide figures are saved as PNG only (not PDF).

The notebook (Section E) iterates over all figure functions twice: once with `style='thesis'`, once with `style='slide'`.

---

### D. Create Figure Catalog

#### D1. Catalog Structure

The figure catalog is a CSV file with the following columns:

| Column | Type | Description |
|---|---|---|
| `filename` | str | Filename without directory path (e.g., `fig_gradcam_5target_sample42.png`) |
| `filepath` | str | Relative path from experiment directory |
| `figure_type` | str | One of: `gradcam_5target_panel`, `gradcam_multi_image`, `attention_bar`, `attention_heatmap`, `shap_modality_summary`, `shap_waterfall`, `shap_radar`, `case_study_panel`, `cross_method_comparison`, `architecture_diagram`, `prediction_bar` |
| `format` | str | `png` or `pdf` |
| `sample_id` | int or blank | Sample index from the dataset, blank for aggregate figures |
| `target` | str or blank | Factor name (food, price, atmos, service, overall), blank if all targets |
| `xai_method` | str | `gradcam`, `attention`, `shap`, `lime`, `combined`, or `architecture` |
| `description` | str | Human-readable description for thesis writing |
| `source_phase` | str | Which XAI phases contributed data: `Phase 2`, `Phase 3`, etc. |
| `width_inches` | float | Figure width |
| `height_inches` | float | Figure height |
| `dpi` | int | Resolution |
| `style` | str | `thesis` or `slide` |
| `latex_label` | str | Suggested LaTeX label: `fig:gradcam_5target_42` |
| `latex_caption` | str | Suggested LaTeX caption text |

#### D2. Catalog Generation

As each figure function saves its output, it registers an entry in the module-level `_CATALOG_ENTRIES` list. At the end of the notebook, `export_figure_catalog()` writes the accumulated catalog to CSV.

---

### E. Create Notebook: `xai/notebooks/Phase8_ThesisVisualization.ipynb`

See Section 9 for cell-by-cell design.

---

## 7. Required Code Files

| File | Location | Responsibility |
|---|---|---|
| `thesis_style.py` | `xai/thesis_style.py` | Central style configuration. Contains: `THESIS_STYLE` dict, `SLIDE_STYLE` dict, `FIGURE_SIZES` dict, `COLORBLIND_PALETTE`, `MODALITY_COLORS`, `TARGET_COLORS`, `METHOD_CMAPS`, `HATCH_PATTERNS`, `apply_thesis_style()`, `apply_slide_style()`, `_configure_vietnamese_font()`, `get_cmap()`, `get_figure_size()`, `save_thesis_figure()`, `export_figure_catalog()`. |
| `thesis_figures.py` | `xai/thesis_figures.py` | Figure generation functions. Contains: `fig_gradcam_5target_panel()`, `fig_gradcam_multi_image()`, `fig_attention_token_bar()`, `fig_attention_heatmap()`, `fig_shap_modality_summary()`, `fig_shap_waterfall()`, `fig_case_study_panel()`, `fig_modality_profile_radar()`, `fig_cross_method_comparison()`, `fig_architecture_xai_diagram()`, `fig_prediction_comparison_bar()`. Each function loads raw data, renders with thesis style, saves PNG+PDF, registers in catalog. |
| Phase 8 notebook | `xai/notebooks/Phase8_ThesisVisualization.ipynb` | Runnable notebook that imports `thesis_style` and `thesis_figures`, applies style, generates all figures for selected samples and all targets, exports catalog. |

---

## 8. Folder Structure

After Phase 8 completion, the thesis figures directory is organized as follows:

```
experiments/EXP_060A_bestsequential_full_configuration/xai/thesis_figures/
|
|-- gradcam/
|   |-- fig_gradcam_5target_sample{id}.png
|   |-- fig_gradcam_5target_sample{id}.pdf
|   |-- fig_gradcam_multiimage_sample{id}_target{idx}.png
|   |-- ...  (one per sample x target combination selected)
|
|-- attention/
|   |-- fig_attention_bar_sample{id}.png
|   |-- fig_attention_bar_sample{id}.pdf
|   |-- fig_attention_heatmap_sample{id}.png
|   |-- fig_attention_heatmap_sample{id}.pdf
|   |-- ...
|
|-- shap/
|   |-- fig_shap_modality_summary.png
|   |-- fig_shap_modality_summary.pdf
|   |-- fig_shap_waterfall_sample{id}_target{idx}.png
|   |-- fig_shap_waterfall_sample{id}_target{idx}.pdf
|   |-- fig_modality_profile_radar.png
|   |-- fig_modality_profile_radar.pdf
|   |-- ...
|
|-- case_studies/
|   |-- fig_case_correct_{sample_id}.png
|   |-- fig_case_correct_{sample_id}.pdf
|   |-- fig_case_higherror_{sample_id}.png
|   |-- fig_case_higherror_{sample_id}.pdf
|   |-- fig_case_conflict_{sample_id}.png
|   |-- fig_case_conflict_{sample_id}.pdf
|   |-- fig_prediction_bar_sample{id}.png
|   |-- fig_prediction_bar_sample{id}.pdf
|   |-- ...
|
|-- comparison/
|   |-- fig_cross_method_sample{id}_target{idx}.png
|   |-- fig_architecture_xai.png
|   |-- fig_architecture_xai.pdf
|
|-- slides/
|   |-- fig_gradcam_5target_sample{id}_slide.png
|   |-- fig_attention_bar_sample{id}_slide.png
|   |-- fig_shap_modality_summary_slide.png
|   |-- fig_case_correct_{sample_id}_slide.png
|   |-- fig_cross_method_sample{id}_target{idx}_slide.png
|   |-- fig_architecture_xai_slide.png
|   |-- ...
|
|-- figure_catalog.csv
|-- style_config.json
```

**Naming conventions:**
- All filenames start with `fig_` prefix.
- Method name follows: `gradcam`, `attention`, `shap`, `case`, `cross_method`, `architecture`, `prediction`.
- Sample and target identifiers are appended with underscore separators.
- Slide variants add `_slide` suffix before extension.
- PDF files are generated only for non-photographic figures (charts, diagrams). Figures containing photographic image content (Grad-CAM overlays, LIME image, cross-method comparison) are PNG only at 300 DPI.

---

## 9. Notebook Design

### Notebook: `xai/notebooks/Phase8_ThesisVisualization.ipynb`

#### Cell 1: Title and Description (Markdown)

```
# Phase 8: Thesis-Ready Visualization
Generate all publication-quality figures from pre-computed XAI data.
- Input: Raw XAI artifacts from Phases 2-7
- Output: Thesis-quality PNG/PDF figures, slide variants, figure catalog
- No model loading required
```

#### Cell 2: Configuration Parameters (Code)

Define all configurable parameters at the top of the notebook:

```
EXP_DIR = '../experiments/EXP_060A_bestsequential_full_configuration'
CSV_PATH = '../data/text/val.csv'
IMAGE_DIR = '../data/image'

# Samples to generate figures for (from case studies)
CASE_STUDY_SAMPLES = {
    'correct': [<sample_id_1>, <sample_id_2>],
    'high_error': [<sample_id_3>],
    'conflict': [<sample_id_4>],
}

# Additional samples for individual method figures
GRADCAM_SAMPLES = [<list of sample IDs>]
ATTENTION_SAMPLES = [<list of sample IDs>]
SHAP_SAMPLES = [<list of sample IDs>]

# Targets to generate per-target figures for
TARGETS_TO_VISUALIZE = [0, 1, 2, 3, 4]  # All 5 targets

# Style
GENERATE_SLIDES = True
```

#### Cell 3: Imports (Code)

```
import sys
sys.path.insert(0, '..')

from xai.thesis_style import (
    apply_thesis_style, apply_slide_style,
    export_figure_catalog, FIGURE_SIZES
)
from xai.thesis_figures import (
    fig_gradcam_5target_panel, fig_gradcam_multi_image,
    fig_attention_token_bar, fig_attention_heatmap,
    fig_shap_modality_summary, fig_shap_waterfall,
    fig_case_study_panel, fig_modality_profile_radar,
    fig_cross_method_comparison, fig_architecture_xai_diagram,
    fig_prediction_comparison_bar
)
from xai.config import DISPLAY_NAMES, FACTOR_NAMES
```

#### Cell 4: Apply Thesis Style (Code)

```
apply_thesis_style()
print("Style applied. Ready to generate thesis figures.")
```

Expected output: Confirmation of style application and Vietnamese font detection.

#### Cell 5: Vietnamese Font Test (Code)

Render a test figure containing Vietnamese text to visually confirm diacritical mark rendering:

```
# Create a small test figure with Vietnamese text
test_strings = [
    "Do an ngon nhung gia hoi cao",    # with actual Vietnamese diacritics
    "Khong gian dep, dich vu tot",      # with actual Vietnamese diacritics
    "Nha hang sach se, gia ca hop ly",  # with actual Vietnamese diacritics
]
# Render in a simple plot, display inline, verify rendering
```

Expected output: A small figure displaying Vietnamese text with all diacritics correctly rendered.

#### Cell 6: Generate Architecture Diagram (Code + Markdown Header)

```
## Architecture Diagram
```
Call `fig_architecture_xai_diagram(EXP_DIR)`. Display inline.

Expected output: Architecture diagram with XAI attachment points.

#### Cell 7: Generate Grad-CAM Figures (Code + Markdown Header)

```
## Grad-CAM Visualizations
```

Loop over `GRADCAM_SAMPLES`:
1. Call `fig_gradcam_5target_panel(sample_id, EXP_DIR, CSV_PATH, IMAGE_DIR)` for each sample.
2. Call `fig_gradcam_multi_image(sample_id, EXP_DIR, target_idx, CSV_PATH, IMAGE_DIR)` for multi-image samples.

Display the first figure inline as a quality check.

Expected output: One 5-target panel per sample, multi-image panels for reviews with multiple images.

#### Cell 8: Generate Attention Figures (Code + Markdown Header)

```
## Attention Visualizations
```

Loop over `ATTENTION_SAMPLES`:
1. Call `fig_attention_token_bar(sample_id, EXP_DIR)` for each sample.
2. Call `fig_attention_heatmap(sample_id, EXP_DIR, layer_idx=11)` for each sample.

Display one of each inline.

Expected output: Token bar charts and heatmaps with correctly rendered Vietnamese tokens.

#### Cell 9: Generate SHAP Figures (Code + Markdown Header)

```
## SHAP Visualizations
```

1. Call `fig_shap_modality_summary(EXP_DIR)` -- single aggregate figure.
2. Call `fig_modality_profile_radar(EXP_DIR)` -- radar chart.
3. Loop over `SHAP_SAMPLES` and targets: call `fig_shap_waterfall(sample_id, EXP_DIR, target_idx)`.

Display modality summary and radar inline.

Expected output: Modality summary bar chart, radar chart, waterfall plots.

#### Cell 10: Generate Case Study Panels (Code + Markdown Header)

```
## Case Study Composite Panels
```

Loop over `CASE_STUDY_SAMPLES`:
1. For each `(case_type, sample_ids)` pair, call `fig_case_study_panel(sample_id, case_type, EXP_DIR, CSV_PATH, IMAGE_DIR)`.
2. Call `fig_prediction_comparison_bar(sample_id, EXP_DIR)` for each.

Display one case study panel inline.

Expected output: Multi-panel composite figures combining all XAI methods.

#### Cell 11: Generate Cross-Method Comparison Figures (Code + Markdown Header)

```
## Cross-Method Comparisons
```

Select 2-3 representative samples. For each:
1. Call `fig_cross_method_comparison(sample_id, EXP_DIR, target_idx, CSV_PATH, IMAGE_DIR)`.

Display one comparison inline.

Expected output: Side-by-side Grad-CAM vs LIME, Attention vs LIME Text.

#### Cell 12: Generate Slide Variants (Code + Markdown Header)

```
## Slide Format Variants
```

1. Call `apply_slide_style()`.
2. Re-run all figure functions with `style='slide'`.
3. Call `apply_thesis_style()` to restore thesis style.

Expected output: Slide-format figures saved to `slides/` subdirectory.

#### Cell 13: Export Figure Catalog (Code + Markdown Header)

```
## Figure Catalog
```

1. Call `export_figure_catalog(os.path.join(EXP_DIR, 'xai/thesis_figures/figure_catalog.csv'))`.
2. Display the catalog as a DataFrame.

Expected output: CSV file with all figure metadata. DataFrame display showing all entries.

#### Cell 14: Export Style Config (Code)

1. Serialize `THESIS_STYLE`, `FIGURE_SIZES`, `COLORBLIND_PALETTE`, `MODALITY_COLORS`, `TARGET_COLORS`, `METHOD_CMAPS` to JSON.
2. Save to `xai/thesis_figures/style_config.json`.

Expected output: JSON file documenting all style parameters.

#### Cell 15: Validation Summary (Code + Markdown)

```
## Validation Summary
```

1. Count total figures generated (thesis + slide).
2. Verify all expected files exist.
3. Check DPI of a sample PNG (via PIL.Image.open and checking info['dpi']).
4. Report any missing figures or errors.

Expected output: Summary table showing figure counts, DPI verification, and any issues.

---

## 10. Algorithm

### Master Algorithm: Thesis Figure Generation Pipeline

```
ALGORITHM: GenerateThesisFigures

INPUT:
    exp_dir         -- experiment directory containing xai/ subdirectory
    csv_path        -- path to validation/test CSV
    image_dir       -- path to image cache
    sample_lists    -- dictionaries mapping figure types to sample IDs
    target_indices  -- list of target indices to process (default: [0,1,2,3,4])

OUTPUT:
    thesis_figures/ directory with all PNG/PDF figures
    figure_catalog.csv
    style_config.json

PROCEDURE:

1. INITIALIZE STYLE
   1.1  Load THESIS_STYLE dictionary
   1.2  Apply to matplotlib rcParams globally
   1.3  Detect and configure Vietnamese-capable font
   1.4  Verify font by rendering test string
   1.5  If font test fails, log warning and fall back to next candidate
   1.6  Initialize empty catalog list

2. CREATE OUTPUT DIRECTORIES
   2.1  Create thesis_figures/ and all subdirectories:
        gradcam/, attention/, shap/, case_studies/, comparison/, slides/

3. GENERATE ARCHITECTURE DIAGRAM
   3.1  Call fig_architecture_xai_diagram(exp_dir)
   3.2  Register in catalog

4. FOR EACH sample_id IN gradcam_sample_list:
   4.1  LOAD original images for sample_id from csv_path + image_dir
   4.2  FOR EACH target_idx IN target_indices:
        4.2.1  LOAD gradcam heatmap from xai/gradcam/gradcam_sample{id}_target{idx}.npy
        4.2.2  IF file not found: LOG warning, SKIP
   4.3  CALL fig_gradcam_5target_panel(sample_id, exp_dir, csv_path, image_dir)
   4.4  IF sample has multiple real images:
        4.4.1  FOR EACH target_idx:
               CALL fig_gradcam_multi_image(sample_id, exp_dir, target_idx, csv_path, image_dir)
   4.5  Register all generated figures in catalog

5. FOR EACH sample_id IN attention_sample_list:
   5.1  LOAD attention data from xai/attention/attention_sample{id}.npz
   5.2  IF file not found: LOG warning, SKIP
   5.3  CALL fig_attention_token_bar(sample_id, exp_dir)
   5.4  CALL fig_attention_heatmap(sample_id, exp_dir, layer_idx=11)
   5.5  Register in catalog

6. GENERATE AGGREGATE SHAP FIGURES
   6.1  LOAD xai/shap/shap_modality_summary.csv
   6.2  IF file not found: LOG warning, SKIP
   6.3  CALL fig_shap_modality_summary(exp_dir)
   6.4  CALL fig_modality_profile_radar(exp_dir)
   6.5  Register in catalog

7. FOR EACH sample_id IN shap_sample_list:
   7.1  FOR EACH target_idx IN target_indices:
        7.1.1  LOAD SHAP values from xai/shap/shap_values_sample{id}.npy
        7.1.2  IF file not found: LOG warning, SKIP
        7.1.3  CALL fig_shap_waterfall(sample_id, exp_dir, target_idx)
   7.2  Register in catalog

8. FOR EACH (case_type, sample_ids) IN case_study_samples:
   8.1  FOR EACH sample_id IN sample_ids:
        8.1.1  LOAD case study report from xai/case_studies/
        8.1.2  IF file not found: LOG warning, SKIP
        8.1.3  CALL fig_case_study_panel(sample_id, case_type, exp_dir, csv_path, image_dir)
        8.1.4  CALL fig_prediction_comparison_bar(sample_id, exp_dir)
   8.2  Register in catalog

9. FOR EACH sample_id IN comparison_sample_list:
   9.1  FOR EACH target_idx IN selected_targets:
        9.1.1  CALL fig_cross_method_comparison(sample_id, exp_dir, target_idx, csv_path, image_dir)
   9.2  Register in catalog

10. GENERATE SLIDE VARIANTS
    10.1  Apply SLIDE_STYLE to rcParams
    10.2  REPEAT steps 3-9 with style='slide'
    10.3  Save all slide figures to slides/ subdirectory with _slide suffix
    10.4  Restore THESIS_STYLE to rcParams

11. EXPORT CATALOG
    11.1  Convert catalog list to DataFrame
    11.2  Save to figure_catalog.csv
    11.3  Print summary: total figures, by type, by format

12. EXPORT STYLE CONFIG
    12.1  Serialize all style dictionaries to JSON
    12.2  Save to style_config.json

13. VALIDATE
    13.1  FOR EACH expected figure file:
          13.1.1  Assert file exists
          13.1.2  Assert file size > 0
    13.2  Spot-check DPI on 3 random PNG files
    13.3  Report validation results
```

---

## 11. Validation

### V1. Rendering Correctness

**Check:** All figures render without Python errors.
**Method:** Run the entire Phase 8 notebook from top to bottom. Any cell that raises an exception indicates a rendering failure.
**Pass criterion:** Zero exceptions.

### V2. Vietnamese Diacritics

**Check:** Vietnamese text displays correctly in all token-level figures.
**Method:**
1. Inspect `fig_attention_token_bar` and `fig_attention_heatmap` figures manually.
2. Verify that characters such as `d` (d with stroke), `a` (a with breve), `o` (o with horn), `u` (u with horn), and all six tones display without replacement boxes or question marks.
3. Compare rendered tokens against the original `comment_clean` text from the CSV.
**Pass criterion:** No glyph substitution or missing characters in any figure containing Vietnamese text.

### V3. DPI Verification

**Check:** All thesis figures are saved at 300 DPI.
**Method:** Open 5 random PNG files using PIL/Pillow and read the DPI metadata: `Image.open(path).info.get('dpi')`.
**Pass criterion:** All report `(300, 300)`.

### V4. Color Consistency

**Check:** All figures use the same color assignments for the same semantic concepts.
**Method:**
1. Verify that text modality is always `MODALITY_COLORS['text']` (#0072B2) across all figures.
2. Verify that image modality is always `MODALITY_COLORS['image']` (#D55E00).
3. Verify that target-specific colors match `TARGET_COLORS` in every figure that uses them.
4. Perform this check by visual inspection of at least one figure from each of the 6 subdirectories.
**Pass criterion:** Zero color inconsistencies.

### V5. Grayscale Readability

**Check:** Key figures remain interpretable when converted to grayscale.
**Method:**
1. Select 3 critical figures: `fig_shap_modality_summary`, `fig_gradcam_5target_panel`, `fig_attention_token_bar`.
2. Convert each to grayscale programmatically: `ImageOps.grayscale(Image.open(path))`.
3. Visually inspect whether bars, heatmap regions, and chart elements remain distinguishable.
**Pass criterion:** All elements distinguishable. Hatch patterns provide secondary encoding for bar charts.

### V6. Figure Size Verification

**Check:** Figure dimensions match the specified size classes.
**Method:** For each figure type, open the PNG and compute actual width/height in inches: `pixels / dpi`.
**Pass criterion:** Within 0.1 inches of specified size for each dimension.

### V7. Catalog Completeness

**Check:** The figure catalog lists every generated figure.
**Method:**
1. List all files in `thesis_figures/` recursively.
2. Compare against `figure_catalog.csv` entries.
3. Identify any files not in the catalog, or catalog entries without corresponding files.
**Pass criterion:** One-to-one correspondence (excluding `figure_catalog.csv` and `style_config.json` themselves).

### V8. PDF vs PNG Appropriateness

**Check:** PDF files are generated for charts/diagrams, PNG for photographic-content figures.
**Method:** Verify from the catalog that:
- `fig_shap_modality_summary` has both PNG and PDF.
- `fig_architecture_xai` has both PNG and PDF.
- `fig_gradcam_5target_panel` has PNG and PDF (PDF with embedded raster is acceptable).
- `fig_cross_method_comparison` has PNG only.
**Pass criterion:** Format assignments match the specification.

### V9. Reproducibility

**Check:** Running the notebook twice produces identical output.
**Method:**
1. Run the notebook once and compute MD5 checksums of all thesis figure PNGs.
2. Delete the thesis_figures directory.
3. Run the notebook again.
4. Compute MD5 checksums of all new PNGs.
5. Compare checksums.
**Pass criterion:** All checksums match (identical file content). Note: PDF checksums may differ due to timestamp metadata; check PNG only.

### V10. No Model Loading

**Check:** Phase 8 does not load the trained model or run any inference.
**Method:** Search `thesis_figures.py` and the notebook for any calls to `load_model()`, `torch.load()`, `model.forward()`, or `model.eval()`.
**Pass criterion:** No model loading or inference calls found. The only import from `xai/utils.py` should be `load_single_sample()` (for loading original images/text) and utility constants.

---

## 12. Risks

### R1: Vietnamese Font Rendering Failure

**Problem:** Matplotlib may not render Vietnamese diacritical marks (`a breve`, `a circumflex`, `d stroke`, `e circumflex`, `o circumflex`, `o horn`, `u horn`, plus six tone marks) correctly. This manifests as replacement boxes (tofu), question marks, or incorrectly positioned combining characters.

**Why it happens:** Matplotlib uses its own font management system separate from the OS. Many matplotlib-bundled fonts do not include Vietnamese glyphs. The default `'DejaVu Sans'` font DOES support Vietnamese, but if rcParams override the font family to one that lacks Vietnamese support (e.g., certain `'serif'` fonts), rendering will fail.

**Possible strategies:**

- **Strategy A: Use DejaVu Sans/Serif (matplotlib-bundled).** DejaVu fonts are bundled with every matplotlib installation and include Latin Extended-A/B character blocks covering all Vietnamese characters. Set `font.serif: ['DejaVu Serif']` for thesis or `font.sans-serif: ['DejaVu Sans']` for slides.
  - *Advantage:* Guaranteed available on every platform (Windows, Linux, macOS, Colab). Zero installation required.
  - *Disadvantage:* DejaVu Serif may not match the thesis template's required font (e.g., Times New Roman). However, for figures embedded in LaTeX, this is typically acceptable.

- **Strategy B: Install and use Noto fonts (Google).** Download `Noto Serif` or `Noto Sans` and register them with matplotlib's font manager via `matplotlib.font_manager.fontManager.addfont()`.
  - *Advantage:* Professional quality, extensive Unicode coverage, available in both serif and sans-serif. Matches the modern typography standard.
  - *Disadvantage:* Requires font installation, which may not be possible on Colab without additional setup. Adds a dependency.

- **Strategy C: Use PIL/Pillow for text rendering.** For figures where matplotlib font rendering fails, render text separately using Pillow with a system font, then composite onto the matplotlib figure.
  - *Advantage:* Uses OS-level font rendering which typically handles Vietnamese correctly.
  - *Disadvantage:* Extremely complex, breaks the matplotlib workflow, makes figures harder to reproduce and modify.

- **Strategy D: Transliterate Vietnamese to ASCII.** Replace diacritical marks with their ASCII equivalents (e.g., `do an` -> `do an`).
  - *Advantage:* Eliminates the font problem entirely.
  - *Disadvantage:* Loses linguistic authenticity. Vietnamese words become ambiguous without diacritics (e.g., `ma` can mean ghost, mother, tomb, rice seedling, horse, or cheek depending on the tone mark). This is scientifically unacceptable for a thesis on Vietnamese text processing.

**Engineering trade-offs:** Strategy A is zero-cost and zero-risk but may produce a slightly different serif face than the thesis body text. Strategy B produces the best visual result but adds complexity. Strategy C is over-engineered. Strategy D is a non-starter for a Vietnamese NLP thesis.

**Research trade-offs:** Authentic Vietnamese rendering is essential for a thesis claiming to analyze Vietnamese restaurant reviews. Strategy D undermines the thesis claim. Strategies A and B are both research-acceptable.

**FINAL DECISION:** Use Strategy A (DejaVu Sans/Serif) as the primary font. DejaVu is guaranteed available in every matplotlib installation and renders all Vietnamese characters correctly. For the serif variant (thesis), use `'DejaVu Serif'`; for sans-serif variant (slides), use `'DejaVu Sans'`. The `_configure_vietnamese_font()` function tests rendering of a Vietnamese string and falls back through the priority list: DejaVu -> Noto -> Arial Unicode MS -> matplotlib default. Never use Strategy D.

**Reason:** Zero installation overhead, cross-platform compatibility, and complete Vietnamese character coverage make DejaVu the only risk-free choice. The visual difference between DejaVu Serif and Times New Roman is negligible in embedded figures.

---

### R2: Figure Size Inconsistency Across Figure Types

**Problem:** Different figure types have different aspect ratios and content densities. A 5-column Grad-CAM panel needs a wide format, while a radar chart needs a square format, and a case study panel needs a tall format. Without standardization, figures will have inconsistent white space, label truncation, and scaling artifacts.

**Why it happens:** Matplotlib auto-sizing is unreliable when figures have complex layouts. Setting a uniform `figsize` for all figures causes some to be too wide and others too tall.

**Possible strategies:**

- **Strategy A: One size fits all.** Use `(7, 5)` for everything.
  - *Advantage:* Simple. Consistent page appearance.
  - *Disadvantage:* 5-column Grad-CAM panels become unreadably compressed. Radar charts have excessive horizontal white space. Case study panels truncate labels.

- **Strategy B: Size classes per figure type.** Define a `FIGURE_SIZES` dictionary mapping each figure type to an optimal `(width, height)`.
  - *Advantage:* Each figure type gets the aspect ratio and dimensions that best suit its content. No truncation or compression artifacts.
  - *Disadvantage:* Slightly more complex to implement. Thesis layout must accommodate different figure widths (but LaTeX handles this naturally with `\includegraphics[width=\textwidth]{...}`).

- **Strategy C: Dynamic sizing based on content.** Compute figure size at runtime based on number of elements (tokens, targets, panels).
  - *Advantage:* Adapts perfectly to varying content.
  - *Disadvantage:* Non-deterministic sizing makes it harder to maintain consistency across notebook runs. Over-engineered for a fixed set of figure types.

**FINAL DECISION:** Strategy B. Define `FIGURE_SIZES` with named size classes. Each figure function specifies which size class it uses. This balances flexibility with consistency. The size classes are: `single_column` (3.5x2.8), `double_column` (7x4), `full_page` (7x9), `thesis_wide` (6.5x4), `panel_5col` (14x3), `panel_2x2` (7x7), `panel_3x2` (7x10), `panel_2x3` (10.5x7), `heatmap_square` (6x5.5), `bar_horizontal` (6.5x4.5), `radar` (5.5x5.5), `slide_standard` (10x6), `slide_wide` (12x5).

**Reason:** The figure set is well-defined and fixed. Each type has known content structure. Named size classes make the intent clear and ensure reproducibility.

---

### R3: Color Scheme Accessibility (Colorblind Safety)

**Problem:** Approximately 8% of males and 0.5% of females have some form of color vision deficiency. Thesis figures that rely solely on red-green color distinctions will be inaccessible to these readers. Some journals also require grayscale-safe figures.

**Why it happens:** Default matplotlib color cycles and commonly used palettes (e.g., red/green for positive/negative SHAP) use colors that are indistinguishable under protanopia or deuteranopia.

**Possible strategies:**

- **Strategy A: Okabe-Ito palette.** Use the Okabe-Ito colorblind-friendly palette, which was specifically designed for universal color vision accessibility.
  - *Advantage:* Proven accessibility. Used in major publications. Distinguishable under all common color vision deficiencies.
  - *Disadvantage:* Limited to 8 colors. Some colors (yellow) may be less visible on white backgrounds.

- **Strategy B: Viridis-family colormaps.** Use `viridis`, `plasma`, `inferno`, or `magma` for sequential data.
  - *Advantage:* Perceptually uniform and colorblind-safe. Monotonically increasing luminance ensures grayscale safety.
  - *Disadvantage:* Not suitable for categorical data (only sequential).

- **Strategy C: Color + secondary encoding.** Use color for primary encoding but add hatch patterns, marker shapes, or line styles as secondary encoding.
  - *Advantage:* Figures remain readable even in pure grayscale or under any color vision deficiency.
  - *Disadvantage:* Slightly more complex rendering. Hatching can look busy.

**FINAL DECISION:** Combine Strategies A, B, and C. Use Okabe-Ito for categorical comparisons (modality colors, target colors). Use `inferno` for Grad-CAM heatmaps and `YlOrRd` for attention (both are perceptually uniform). Use `RdBu_r` for SHAP diverging values (red-blue is safe for most color vision deficiencies). Add hatch patterns to all bar charts as secondary encoding. This triple-layered approach ensures accessibility under all conditions.

**Reason:** A thesis should be readable by everyone. The combination of colorblind-safe palette + perceptually uniform colormaps + hatch patterns provides maximum accessibility with minimal visual complexity.

---

### R4: Grad-CAM Heatmap Colorbar Inconsistency

**Problem:** Different Grad-CAM figures may use different color scales if the raw heatmap values are not consistently normalized. This makes cross-figure comparison misleading: a medium-intensity region in one figure may appear identical to a high-intensity region in another.

**Why it happens:** Grad-CAM heatmaps from Phase 2 may be saved with different value ranges depending on the sample and target. Matplotlib's `imshow()` auto-scales colors to the data range of each individual subplot unless explicitly controlled.

**Possible strategies:**

- **Strategy A: Normalize all heatmaps to [0, 1] before visualization.** Divide each heatmap by its maximum value.
  - *Advantage:* Full color range used in every subplot. Easy to implement.
  - *Disadvantage:* Cross-figure comparison is still misleading because a value of 0.8 in one figure may correspond to a very different absolute activation than 0.8 in another.

- **Strategy B: Use a fixed global vmin/vmax across all Grad-CAM figures.** Compute the global maximum activation across all heatmaps and use that as the shared `vmax`.
  - *Advantage:* True cross-figure comparability.
  - *Disadvantage:* Some figures may appear very faint if their maximum activation is much lower than the global maximum. Requires a pre-scan of all heatmaps.

- **Strategy C: Normalize per-figure (within the 5-target panel) but not across figures.** Within a single `fig_gradcam_5target_panel`, use the same `vmax` for all 5 targets.
  - *Advantage:* Cross-target comparison within a single sample is valid. Colorbar is meaningful. Each figure uses its full range.
  - *Disadvantage:* Cross-sample comparison still requires caution. But this is the standard practice in Grad-CAM literature.

**FINAL DECISION:** Strategy C. Within each `fig_gradcam_5target_panel`, compute `vmax = max(all 5 heatmaps' max values)` and `vmin = 0`. Use this shared range for all 5 subplots and the shared colorbar. This makes cross-target comparison within a sample valid while preserving visual clarity. Add a caption note: "Colorbar range is normalized within each panel."

**Reason:** Strategy C is the standard practice in Grad-CAM literature. Cross-sample comparison of absolute activation values is generally not meaningful (different images have different content), so per-panel normalization is scientifically appropriate.

---

### R5: PDF vs PNG Format Selection

**Problem:** LaTeX thesis documents prefer PDF figures for charts and diagrams (vector, resolution-independent, smaller file size), but Grad-CAM overlays and LIME image explanations contain photographic content that makes PDF files excessively large and slow to compile.

**Why it happens:** PDF stores vector graphics efficiently but stores raster images as embedded bitmaps. A Grad-CAM overlay figure saved as PDF embeds the full-resolution photograph, resulting in a 5-20 MB file versus a 200 KB PNG.

**Possible strategies:**

- **Strategy A: Save everything as PNG only.** Simple and consistent.
  - *Advantage:* Simple implementation. All figures handled uniformly.
  - *Disadvantage:* Charts and text in PNG have fixed resolution. Zooming reveals pixelation. Larger file sizes for simple charts.

- **Strategy B: Save everything as PDF only.** LaTeX handles PDFs natively.
  - *Advantage:* Vector graphics for charts. Single format.
  - *Disadvantage:* Photographic figures become enormous (5-20 MB each). LaTeX compilation slows significantly.

- **Strategy C: Dual format -- charts as PDF + PNG, photos as PNG only.** Each figure function determines the appropriate format(s).
  - *Advantage:* Optimal file sizes. Best quality for each content type. Thesis author can choose the appropriate format for each `\includegraphics`.
  - *Disadvantage:* Slightly more complex. Must track which formats each figure supports.

**FINAL DECISION:** Strategy C. Each figure function specifies its default format list:
- Charts, bar plots, diagrams: `['png', 'pdf']` (both formats).
- Photographic overlays (Grad-CAM, LIME image, cross-method comparison): `['png']` (PNG at 300 DPI only).
- Case study panels (mixed content): `['png', 'pdf']` (PDF will embed raster, but the composite is primarily chart content).

The `save_thesis_figure()` function accepts a `formats` parameter, and each figure function passes its appropriate format list.

**Reason:** This is standard practice in academic publishing. The thesis author inserts the PDF version for charts (crisp text and lines at any zoom) and the PNG version for photographic figures.

---

### R6: SHAP Waterfall Feature Labels

**Problem:** For CrossAttentionFusion, the fused embedding has dimension 1024 (512 from text-to-image cross-attention + 512 from image-to-text cross-attention). Individual embedding dimensions do not correspond to human-interpretable features. Labeling SHAP waterfall bars as `"dim_0"`, `"dim_1"`, etc. provides no insight.

**Why it happens:** Unlike tabular data where each feature has a name, neural network embeddings are distributed representations. A single dimension encodes a complex combination of learned features.

**Possible strategies:**

- **Strategy A: Label as raw dimension indices.** `"dim_0"` through `"dim_1023"`.
  - *Advantage:* Technically accurate. No overclaiming.
  - *Disadvantage:* Uninterpretable for the reader.

- **Strategy B: Label as modality-grouped dimensions.** `"T2I_dim_{i}"` for dimensions 0-511, `"I2T_dim_{i}"` for dimensions 512-1023.
  - *Advantage:* Shows which cross-attention direction produced each important feature. More informative than raw indices.
  - *Disadvantage:* Still does not explain what the dimension represents semantically.

- **Strategy C: Show only grouped modality contributions.** Instead of individual dimension bars, show two aggregate bars: total text-attended contribution and total image-attended contribution.
  - *Advantage:* Directly answers the thesis question of modality balance. Clear and interpretable.
  - *Disadvantage:* Loses the fine-grained waterfall structure that shows how individual features push the prediction.

- **Strategy D: Hybrid waterfall.** Show top-10 individual dimensions with modality labels, plus two additional grouped bars showing the sum of remaining text-attended and image-attended contributions.
  - *Advantage:* Preserves the waterfall structure while providing interpretable grouping.
  - *Disadvantage:* Non-standard SHAP visualization.

**FINAL DECISION:** Strategy B for the individual waterfall plot, supplemented by Strategy C in the modality summary chart (`fig_shap_modality_summary`). The waterfall uses `"T2I_dim_{i}"` and `"I2T_dim_{i}"` labels to identify which cross-attention direction produced each feature, while the modality summary provides the grouped interpretation. This combination satisfies both the fine-grained and high-level analysis needs of the thesis.

**Reason:** The thesis needs both levels of analysis. The waterfall shows the SHAP computation is valid (additivity, base value, individual contributions), while the modality summary answers the research question. Using both avoids overclaiming at either level.

---

### R7: Slide Font Size Scaling

**Problem:** Figures designed for thesis print (10pt font at 7-inch width) are unreadable when projected on a screen during a defense presentation. Slide figures need proportionally larger fonts, thicker lines, and simpler layouts.

**Why it happens:** Print and projection have fundamentally different viewing conditions. Print is viewed at arm's length; slides are viewed from meters away.

**Possible strategies:**

- **Strategy A: Scale all rcParams uniformly.** Multiply all font sizes by 1.4x.
  - *Advantage:* Simple.
  - *Disadvantage:* Some elements (tick labels) may become too large relative to the figure content.

- **Strategy B: Define separate SLIDE_STYLE with hand-tuned parameters.** Each parameter is individually optimized for slide readability.
  - *Advantage:* Best visual result. Fine-grained control.
  - *Disadvantage:* More configuration to maintain.

**FINAL DECISION:** Strategy B. Define `SLIDE_STYLE` as a separate dictionary with hand-tuned values: base font 14pt (vs 10pt), title 18pt (vs 12pt), line width 2.5 (vs 1.5), figure size 10x6 (vs 6.5x4). Each figure function accepts a `style` parameter and uses the appropriate size class and rcParams.

**Reason:** Defense presentations are a critical deliverable. Hand-tuned slide parameters produce significantly better results than uniform scaling.

---

### R8: Memory Pressure from Batch Figure Generation

**Problem:** Generating all thesis figures in a single notebook run may accumulate matplotlib figure objects in memory, especially for Grad-CAM overlays that contain full-resolution images.

**Why it happens:** Matplotlib keeps figure objects in memory until explicitly closed. A typical Grad-CAM overlay figure may use 50-100 MB of RAM. Generating 50+ figures without closing them can exceed Colab's 12 GB RAM limit.

**Possible strategies:**

- **Strategy A: Close figures immediately after saving.** Call `plt.close(fig)` after each save.
  - *Advantage:* Minimal memory footprint.
  - *Disadvantage:* Figures cannot be displayed inline in the notebook after closing. Must choose between display and memory.

- **Strategy B: Display then close.** Call `plt.show()` (which displays inline in Jupyter), then `plt.close(fig)`.
  - *Advantage:* Figures are visible in the notebook output. Memory is released after display.
  - *Disadvantage:* `plt.show()` may not release memory immediately in all notebook environments.

- **Strategy C: Generate in batches with explicit gc.** Process figures in batches of 10, call `gc.collect()` between batches.
  - *Advantage:* Balances memory with throughput.
  - *Disadvantage:* Adds complexity.

**FINAL DECISION:** Strategy A as the default behavior in `save_thesis_figure()`. The `close` parameter defaults to `True`, calling `plt.close(fig)` after saving. For inline display in the notebook, the user can set `close=False` for selected figures and manually close them after inspection. The notebook displays only the first figure from each category as a quality check, closing all others immediately.

**Reason:** Memory safety is more important than inline display. The saved PNG files can always be viewed afterward. This prevents Colab crashes during batch figure generation.

---

## 13. Best Practices

### BP1: Deterministic Figure Generation

- Always set `matplotlib.rcParams` at the start of the notebook, before any figure creation.
- Use explicit figure sizes (never rely on auto-sizing).
- Use explicit `vmin`/`vmax` for all colormaps.
- Use `bbox_inches='tight'` for all `savefig` calls to prevent label truncation.

### BP2: Font Consistency

- Never specify fonts inline in individual figure functions. Always read from `THESIS_STYLE` or `SLIDE_STYLE`.
- Test Vietnamese rendering once at notebook start; do not test per-figure.
- If a font change is needed, modify `thesis_style.py` and re-run the entire notebook.

### BP3: Color Consistency

- Never hardcode hex color values in figure functions. Always reference `MODALITY_COLORS`, `TARGET_COLORS`, or `COLORBLIND_PALETTE` from `thesis_style.py`.
- If a new color is needed, add it to the palette first, then reference it.

### BP4: Artifact Naming

- All thesis figure filenames start with `fig_`.
- Use underscores as separators, not hyphens.
- Include sample_id and target_idx in filenames where applicable.
- Slide variants append `_slide` before the file extension.
- Never include spaces or special characters in filenames.

### BP5: Memory Management

- Close every figure immediately after saving unless actively displaying it.
- Do not store figure objects in lists or dictionaries.
- Run `gc.collect()` after processing each batch of 10+ figures.

### BP6: Error Handling

- If a raw data file is missing (e.g., Grad-CAM heatmap for a specific sample was not computed in Phase 2), log a warning and skip that figure. Do NOT raise an exception that halts the entire notebook.
- Use `os.path.exists()` checks before loading any raw data file.
- Track skipped figures in the catalog with a `status` column: `'generated'` or `'skipped:missing_data'`.

### BP7: Logging

- Print a one-line summary after each figure is saved: `f"Saved: {filename} ({width}x{height} in, {dpi} DPI)"`.
- At the end of the notebook, print a summary table: total generated, total skipped, total size in MB.

### BP8: Configuration Over Code

- All tunable parameters (DPI, font sizes, figure sizes, color values, sample IDs, target indices) are defined in configuration dictionaries or notebook-level variables.
- No hardcoded values in loop bodies or function internals.

### BP9: Reproducibility Documentation

- Export `style_config.json` at the end of each run. This file documents every visual parameter used, enabling exact reproduction of the figures.
- Include the matplotlib version and font configuration in the style config.

### BP10: LaTeX Integration

- Use `\includegraphics[width=\textwidth]{figures/fig_gradcam_5target_sample42.pdf}` for full-width figures.
- Use `\includegraphics[width=0.48\textwidth]{...}` for side-by-side figures.
- The `figure_catalog.csv` includes suggested `latex_label` and `latex_caption` columns to accelerate thesis writing.

---

## 14. Deliverables

### Primary Deliverables

| Deliverable | Type | Description |
|---|---|---|
| `xai/thesis_style.py` | Python module | Central style configuration with all palettes, sizes, fonts, and style-application functions |
| `xai/thesis_figures.py` | Python module | 11 figure-generation functions producing thesis-quality output |
| `xai/notebooks/Phase8_ThesisVisualization.ipynb` | Jupyter notebook | Runnable end-to-end figure generation pipeline |

### Generated Artifacts

| Deliverable | Type | Count (approximate) | Description |
|---|---|---|---|
| Grad-CAM 5-target panels | PNG + PDF | 3-5 panels (one per case study sample) | Full 5-target comparison per sample |
| Grad-CAM multi-image figures | PNG | Variable (multi-image reviews only) | Per-image Grad-CAM for multi-image reviews |
| Attention bar charts | PNG + PDF | 3-5 | Top-15 token importance with Vietnamese labels |
| Attention heatmaps | PNG + PDF | 3-5 | Token-to-token attention matrices |
| SHAP modality summary | PNG + PDF | 1 | Aggregate text vs image contribution |
| SHAP modality radar | PNG + PDF | 1 | Radar profile across 5 targets |
| SHAP waterfall plots | PNG + PDF | 5-15 (samples x targets) | Per-sample per-target SHAP decomposition |
| Case study panels | PNG + PDF | 3-5 | Multi-method composite figures |
| Prediction bar charts | PNG + PDF | 3-5 | Ground truth vs prediction comparison |
| Cross-method comparisons | PNG | 2-3 | Side-by-side Grad-CAM/LIME, Attention/LIME |
| Architecture diagram | PNG + PDF | 1 | Model architecture with XAI points |
| Slide variants | PNG | All of the above | Slide-format versions of all figures |

### Metadata Artifacts

| Deliverable | Type | Description |
|---|---|---|
| `figure_catalog.csv` | CSV | Complete catalog of all generated figures with metadata, descriptions, and LaTeX suggestions |
| `style_config.json` | JSON | Serialized style configuration for reproducibility |

---

## 15. Thesis Usage

### Results Chapter

- **fig_shap_modality_summary**: Primary evidence for the modality contribution research question. Shows whether text or image dominates each target score. Directly supports claims like: "Text contribution was highest for price_score (XX%), while image contribution dominated atmosphere_score (YY%)."
- **fig_modality_profile_radar**: Summary visualization showing the modality balance pattern across all 5 targets. Suitable for a prominent position in the Results section.
- **fig_gradcam_5target_panel**: Evidence that the image branch attends to semantically relevant regions (food presentation for food_score, interior for atmosphere_score). Supports: "The model correctly localized food presentation features when predicting food_score."
- **fig_attention_token_bar**: Evidence that the text branch attends to aspect-relevant tokens. Supports: "CLS attention was concentrated on sentiment-bearing tokens such as 'ngon' and price-related tokens such as 'gia'."
- **fig_shap_waterfall**: Detailed decomposition of individual predictions. Supports: "For sample X, the text-attended features contributed positively while image-attended features had a negative effect on price_score."

### Discussion Chapter

- **fig_cross_method_comparison**: Evidence of multi-method agreement or disagreement. Supports: "Grad-CAM and LIME Image agreed on food region importance for food_score, but showed different patterns for atmosphere_score, suggesting that gradient-based and perturbation-based methods capture complementary visual evidence."
- **fig_case_study_panel (high_error)**: Analysis of failure cases. Supports: "In high-error cases, the image branch focused on irrelevant background regions while the text branch correctly identified sentiment terms, suggesting modality conflict."
- **fig_case_study_panel (conflict)**: Analysis of modality disagreement. Supports: "When text and image evidence conflicted, the model tended to follow the text branch, consistent with the higher text contribution observed in SHAP analysis."

### Methodology Chapter

- **fig_architecture_xai**: Architecture diagram showing where each XAI method attaches to the model. Essential for the methodology description.

### Case Studies Section

- **fig_case_study_panel (correct)**: Exemplar of a well-explained correct prediction. Shows how all four XAI methods provide complementary evidence.
- **fig_prediction_comparison_bar**: Visual summary of prediction accuracy for discussed samples.

### Defense Presentation

- All slide-format variants are directly insertable into PowerPoint or Google Slides.
- Key slides:
  1. Architecture diagram with XAI points (methodology overview).
  2. SHAP modality summary (main finding).
  3. One case study panel per case type (evidence quality).
  4. Cross-method comparison (multi-method validation).

### Journal Paper

- Figures saved as PDF (charts) and high-DPI PNG (overlays) meet journal submission requirements.
- Grayscale-safe design satisfies journals that print in black and white.
- The figure catalog accelerates selection of the most impactful figures for a concise journal paper.

---

## 16. Phase Completion Checklist

### Code Deliverables

- [ ] `xai/thesis_style.py` exists and contains: `THESIS_STYLE`, `SLIDE_STYLE`, `FIGURE_SIZES`, `COLORBLIND_PALETTE`, `MODALITY_COLORS`, `TARGET_COLORS`, `METHOD_CMAPS`, `HATCH_PATTERNS`, `apply_thesis_style()`, `apply_slide_style()`, `_configure_vietnamese_font()`, `get_cmap()`, `get_figure_size()`, `save_thesis_figure()`, `export_figure_catalog()`.
- [ ] `xai/thesis_figures.py` exists and contains all 11 figure-generation functions: `fig_gradcam_5target_panel`, `fig_gradcam_multi_image`, `fig_attention_token_bar`, `fig_attention_heatmap`, `fig_shap_modality_summary`, `fig_shap_waterfall`, `fig_case_study_panel`, `fig_modality_profile_radar`, `fig_cross_method_comparison`, `fig_architecture_xai_diagram`, `fig_prediction_comparison_bar`.
- [ ] `xai/notebooks/Phase8_ThesisVisualization.ipynb` exists and runs end-to-end without errors.

### Output Artifacts

- [ ] `thesis_figures/gradcam/` contains at least 3 five-target panel figures.
- [ ] `thesis_figures/attention/` contains at least 3 bar charts and 3 heatmaps.
- [ ] `thesis_figures/shap/` contains modality summary, radar chart, and at least 5 waterfall plots.
- [ ] `thesis_figures/case_studies/` contains at least one panel per case type (correct, high_error, conflict).
- [ ] `thesis_figures/comparison/` contains at least 2 cross-method comparisons and 1 architecture diagram.
- [ ] `thesis_figures/slides/` contains slide variants of all key figures.
- [ ] `figure_catalog.csv` exists and has one entry per generated figure.
- [ ] `style_config.json` exists and documents all style parameters.

### Quality Validation

- [ ] All thesis figures are 300 DPI (verified by PIL metadata check on at least 5 files).
- [ ] All slide figures are 150 DPI.
- [ ] Vietnamese diacritics render correctly in all attention bar charts and heatmaps (manual visual inspection).
- [ ] Color scheme is consistent across all figures (MODALITY_COLORS verified in at least 3 figures).
- [ ] At least 3 critical figures (SHAP summary, Grad-CAM panel, attention bar) pass grayscale readability test.
- [ ] Figure dimensions match FIGURE_SIZES specifications (within 0.1 inch tolerance, verified on at least 5 figures).
- [ ] Running the notebook twice produces identical PNG outputs (MD5 checksum verification).

### No-Model Verification

- [ ] `thesis_figures.py` contains zero calls to `load_model()`, `torch.load()`, `model.forward()`, or `model.eval()`.
- [ ] The notebook does not import any model classes (`CrossAttentionFusion`, `FusionModel`, etc.).
- [ ] All data is loaded from pre-computed NPY, JSON, and CSV files only.

### Catalog Verification

- [ ] Every file in `thesis_figures/` (excluding `figure_catalog.csv` and `style_config.json`) has a corresponding entry in `figure_catalog.csv`.
- [ ] Every entry in `figure_catalog.csv` has a corresponding file on disk.
- [ ] All `latex_label` values are unique.
- [ ] All `description` values are non-empty.

### Integration Verification

- [ ] At least one figure from each subdirectory (gradcam, attention, shap, case_studies, comparison) can be successfully included in a LaTeX document using `\includegraphics`.
- [ ] Slide figures are usable in a PowerPoint presentation at 16:9 aspect ratio.
