# Phase 1: Implementation Notes

This document summarizes the implementation decisions for Phase 1 (XAI Infrastructure).
It is intended for future maintenance, reproducibility, and Phase 2–8 development.

---

## 1. Proposal Compliance

| Item | Status | Notes |
|---|---|---|
| `xai/__init__.py` | ✔ Exact | Exports all public API as specified |
| `xai/config.py` | ✔ Exact | All constants defined: TARGET_NAMES, TARGET_INDICES, FACTOR_NAMES, DISPLAY_NAMES, LABEL_COLS, SCORE_RANGE, COLOR_SCHEMES, dimension constants |
| `get_device()` | ✔ Exact | Matches main.py lines 51-54 |
| `set_seed()` | ✔ Exact | Matches main.py lines 24-34 |
| `get_tokenizer()` | ✔ Exact | Direct AutoTokenizer wrapper |
| `get_image_processor()` | ✔ Exact | Full fallback chain from main.py lines 61-69 with private `_TimmProcessor` |
| `load_model()` | ✔ Exact | Config-driven reconstruction matching test.py lines 59-97 |
| `load_single_sample()` | ✔ Exact | Replicates src/dataset.py MultimodalDataset.__getitem__() |
| `get_prediction()` | ✔ Exact | Matches test.py lines 105-121 |
| `save_figure()` | ✔ Exact | Consistent DPI, bbox_inches, facecolor |
| `save_raw_values()` | ✔ Exact | Supports JSON, NPY, NPZ, CSV |
| Verification notebook | ✔ Exact | 10 verification checks (V1–V10) as specified |
| `_NumpyEncoder` | ✔ Added | Custom JSON encoder for numpy/torch types |
| `get_metadata()` | ✔ Added | Utility for consistent artifact metadata |

---

## 2. Proposal Deviations

### 2.1 Artifact output directory

- **Proposal:** Save artifacts to `experiments/EXP_XXX/xai/infrastructure/`
- **Actual:** Save to `{DRIVE_ROOT}/xai/phase1/`
- **Why:** The existing codebase stores experiment outputs on Google Drive under `{DRIVE_ROOT}/experiments/`. However, XAI artifacts span multiple experiments and phases. Following the existing notebook pattern (e.g., `demo_single_sample_exp060A.ipynb` saves to `{EXP_DIR}/demo_single_sample/`), XAI outputs are stored under `{DRIVE_ROOT}/xai/phase{N}/` for phase-level artifacts and `{DRIVE_ROOT}/xai/{method}/` for method-level artifacts. This avoids polluting experiment directories with post-hoc analysis artifacts.
- **Compatibility:** The notebook's `XAI_OUT_DIR` variable can be changed by the user in the configuration cell.

### 2.2 `_TimmProcessor` as private class

- **Proposal:** References `TimmProcessor` class replication
- **Actual:** Named `_TimmProcessor` (underscore prefix)
- **Why:** Python convention for internal-use-only classes. It should not be imported directly by other modules; `get_image_processor()` is the public API.

### 2.3 `get_metadata()` added beyond proposal

- **Proposal:** Not explicitly specified
- **Actual:** Added as a utility function
- **Why:** The implementation requirements mandate metadata in every artifact (experiment_id, timestamp, git commit, device, seed, model names). A shared utility eliminates boilerplate in future phases.

---

## 3. Engineering Decisions

### 3.1 No modification to existing code
Zero lines changed in Models/, src/, main.py, test.py, Config.py, or Trainer.py. All XAI functionality is in the new `xai/` package.

### 3.2 Private `_load_config()` function
Config loading logic (yaml → json → checkpoint args) is extracted into a private function reused by `load_model()`. This mirrors how `test.py` handles config but adds yaml support.

### 3.3 `module.` prefix stripping
The `load_model()` function handles DataParallel-wrapped checkpoints by stripping `module.` prefixes, with a warning. This is defensive — current checkpoints don't use DataParallel, but the safeguard costs nothing.

### 3.4 Image loading without network
`_load_image_from_cache()` only reads from local cache (MD5 hash lookup), never downloads from URLs. This is safer for XAI analysis (deterministic, no network dependency) and matches the Colab workflow where `data.zip` is extracted first.

### 3.5 Notebook follows exact Colab patterns
Steps 1-3 (Drive mount, clone, data extraction) match `EXP_060A_bestsequential_full_configuration.ipynb` exactly. Step 4 configuration cell uses the same `DRIVE_ROOT`/`EXP_ID` pattern.

---

## 4. Assumptions

| Assumption | How to verify |
|---|---|
| Checkpoint is at `{EXP_DIR}/best_model_train_fusion.pth` | `os.path.isfile()` |
| Checkpoint contains `'model_state_dict'` key | Check `'model_state_dict' in ckpt` |
| Config is at `{EXP_DIR}/config.yaml` or `config.json` | `os.path.isfile()` with fallback |
| Data CSVs have columns: `comment_clean`, `image_url`, `food_score`, ... | Pandas read + column check |
| Image cache exists at `./data/image/` with MD5-hashed filenames | `os.path.exists()` per image |
| PhoBERT supports `output_attentions=True` | Verified in V8 |
| Swin-B produces spatial feature maps before pooling | Verified in V9 |
| `max_length=256` (from Config.py default) | Read from experiment config |
| `max_images=4` (from dataset.py) | Hardcoded as constant |

---

## 5. Compatibility with Existing Codebase

- **No imports from main.py or test.py:** These files use argparse, which fails when imported as modules. Instead, their logic is replicated in `xai/utils.py`.
- **Model imports:** `from Models.TextModel import TextModel` etc. work because the project root is on `sys.path`.
- **Dataset compatibility:** `load_single_sample()` replicates `MultimodalDataset.__getitem__()` line-by-line to ensure identical preprocessing.
- **Checkpoint format:** Handles both `{'model_state_dict': ...}` (Trainer.py format) and raw state_dict.

---

## 6. Reusable Components for Future Phases

| Component | Used by |
|---|---|
| `load_model()` | All phases (2-8) |
| `load_single_sample()` | Phases 2, 3, 5, 6 |
| `get_prediction()` | Phases 2, 5, 6 |
| `get_tokenizer()` | Phases 3, 5 |
| `get_image_processor()` | Phases 2, 5 |
| `save_figure()` | Phases 2, 3, 4, 5, 6, 8 |
| `save_raw_values()` | All phases |
| `get_metadata()` | All phases |
| `set_seed()` | All phases |
| `get_device()` | All phases |
| `xai/config.py` constants | All phases |
| V9 findings (feature map layer/shape) | Phase 2 (Grad-CAM) |
| V8 findings (attention shape) | Phase 3 (Attention) |
| V10 findings (fused vector shape) | Phase 4 (SHAP) |

---

## 7. Suggested Improvements

1. **Batch inference utility:** A `get_predictions_batch()` function for SHAP background extraction (Phase 4) would be useful. Not added now to keep Phase 1 minimal.

2. **Caching:** `load_model()` could cache the model to avoid reloading in notebooks that call it multiple times. Deferred to avoid complexity.

3. **Config dataclass:** The config dict could be wrapped in a dataclass for type safety. Deferred because existing codebase uses plain dicts.

---

## 8. Implementation Summary

### What was implemented
- `xai/__init__.py` — package initialization with public API exports
- `xai/config.py` — all constants for targets, dimensions, colors, defaults
- `xai/utils.py` — 10 utility functions + 2 helper classes covering model loading, inference, preprocessing, and artifact saving
- `xai/notebooks/Phase1_Infrastructure_Verification.ipynb` — 10-check verification notebook following existing Colab patterns

### What remains for later phases
- Phase 2: `xai/gradcam_explainer.py` + Grad-CAM notebook
- Phase 3: `xai/attention_explainer.py` + Attention notebook
- Phase 4: `xai/shap_explainer.py` + SHAP notebook
- Phase 5: `xai/lime_explainer.py` + LIME notebook
- Phases 6-8: Case study selection, report generation, thesis figures

### Is Phase 1 ready for Phase 2?
**Yes.** All prerequisites verified:
- Model loads correctly (V2)
- Preprocessing matches training pipeline (V6)
- Swin-B spatial feature map layer identified and shape documented (V9)
- PhoBERT attention extraction verified (V8)
- Forward-pass decomposition confirmed (V10)
- All utility functions tested and ready for import
