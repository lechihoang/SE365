# Phase 4: Fusion-level SHAP Analysis — Implementation Notes (V2)

Updated for the **token × patch CrossAttentionFusion** architecture on branch `xai-v3`.

---

## 1. Proposal Compliance

| Item | Status | Notes |
|---|---|---|
| `xai/shap_explainer.py` | Unchanged | All functions correct for new architecture |
| `FusionHeadWrapper` class | Unchanged | Wraps model.head + score_index |
| `extract_fused_embeddings()` | Unchanged | forward_pre_hook on model.head |
| `select_background()` | Unchanged | Random sample from validation embeddings |
| `compute_shap_values()` | Unchanged | DeepExplainer primary |
| `modality_contribution()` | Unchanged | text-origin (0:512) vs image-origin (512:1024) |
| `additivity_check()` | Unchanged | Verifies prediction ≈ base + sum(shap) |
| `plot_modality_contribution()` | Unchanged | Grouped bar chart, 5 targets |
| `SHAPExplainer` class | Unchanged | High-level orchestrator |
| `run_ablation_check()` | Unchanged | Zero-out text/image dims |
| Phase 4 notebook | **Updated** | Branch → `xai-v3` |

---

## 2. Differences from V1

### 2.1 Branch updated

All notebook clone cells use `xai-v3` (was `xai-v2`).

### 2.2 Zero code changes to `shap_explainer.py`

The SHAP implementation operates on the fused vector `[B, 1024]` and the prediction head `model.head`. Both are **unchanged** by the architecture redesign. The cross-attention redesign only changed how the fused vector is computed internally (token×patch attention instead of single-vector attention), but its shape, structure, and the head that processes it are identical.

---

## 3. Cross-Attention V3 Compatibility

### Why `shap_explainer.py` needed no changes

SHAP attaches to the **prediction head** via a forward pre-hook on `model.head`. The fused vector captured by this hook is always `[B, 1024]` regardless of how it was computed upstream:

- **Old architecture:** `cat(squeeze(t_out[B,1,512]), squeeze(i_out[B,1,512]))` → `[B, 1024]`
- **New architecture:** `cat(masked_mean(t_out[B,T,512]), masked_mean(i_out[B,P,512]))` → `[B, 1024]`

Same output dimension, same head, same SHAP methodology.

### What IS different (behavioral, not code)

The fused embedding now contains richer cross-modal interactions because each text token attended to all image patches (and vice versa). This may produce slightly different SHAP values and modality contribution percentages compared to V1, but the implementation is unchanged.

---

## 4. Engineering Decisions

- Forward pre-hook on `model.head` captures fused vector
- DeepExplainer as primary SHAP method (fast, gradient-based)
- "text-origin" / "image-origin" terminology acknowledges cross-modal mixing
- Ablation check (zero-out dims) as complementary validation
- Graceful `shap` import with install instructions

---

## 5. Compatibility with Future Phases

| Phase | Status | Notes |
|---|---|---|
| Phase 6 (Case Study) | Compatible | Artifact format unchanged |
| Phase 7 (Report) | Compatible | Modality contribution CSV format unchanged |
| Phase 8 (Thesis Viz) | Compatible | Figure format unchanged |

---

## 6. Remaining Limitations

1. **Cross-attention mixing:** Text-origin features contain image information and vice versa.
2. **DeepExplainer approximation:** Not exact Shapley values.
3. **Background sensitivity:** Results depend on background sample selection.
4. **No per-feature interpretation:** Only grouped modality analysis is interpretable.

---

## 7. Summary

### V2 changes
- Notebook: branch → `xai-v3`
- Python module: **zero changes**

### Ready for use?
**Yes.** The fused vector `[B, 1024]` and prediction head are unchanged. Re-run on Colab to capture new embeddings from the redesigned architecture.
