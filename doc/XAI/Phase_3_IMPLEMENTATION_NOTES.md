# Phase 3: Attention Visualization — Implementation Notes (V3)

Updated to include **cross-attention visualization** between text tokens and image patches.

---

## 1. Audit Result

### A. Does CrossAttentionFusion expose cross-attention weights?

**Yes, but they are discarded.** In `CrossAttentionFusion.forward()` (lines 67-68):
```python
t_out, _ = self.cross_attn_t2i(query=t, key=i, value=i, key_padding_mask=i_kpm)
i_out, _ = self.cross_attn_i2t(query=i, key=t, value=t, key_padding_mask=t_kpm)
```
The `_` discards the attention weights. `nn.MultiheadAttention` returns weights as the second value by default (`need_weights=True`), so the weights are computed but not stored.

### B. Was cross-attention visualization implemented before?

**No.** The Phase 3 notebook Step 14 extracted cross-attention weights in a standalone verification cell but produced no saved artifacts or visualizations. The `AttentionExplainer` class generated only PhoBERT self-attention outputs.

### C. Are weights recoverable without architecture changes?

**Yes.** The weights can be extracted by calling `model.cross_attn_t2i(...)` directly with the projected features and capturing the second return value. No architecture modification is needed.

---

## 2. What Was Implemented

### New classes and functions in `xai/attention_explainer.py`

| Item | Description |
|---|---|
| `extract_cross_attention()` | Extracts T×P cross-attention weights from both `cross_attn_t2i` and `cross_attn_i2t`, trims padding, returns head-averaged matrices |
| `plot_cross_attention_heatmap()` | Renders token × patch heatmap with patch grid labels |
| `plot_patch_importance()` | Overlays patch-level importance on the original image (3-panel: original, patch map, overlay) |
| `CrossAttentionExplainer` | High-level orchestrator generating all cross-attention artifacts per sample |

### New exports in `xai/__init__.py`

```python
extract_cross_attention, plot_cross_attention_heatmap,
plot_patch_importance, CrossAttentionExplainer
```

### New notebook cells (Steps 14b, 14c, 14d)

| Cell | Description |
|---|---|
| Step 14b | Extract cross-attention, validate shapes/sums, generate T×P heatmap |
| Step 14c | Patch importance overlay on original image, top-10 token-patch pairs, entropy statistics |
| Step 14d | Batch processing: generate cross-attention artifacts for all 15 samples |

### Per-sample artifacts generated

```
cross_attention/{sample_id}/
├── cross_attention_raw.npz        # t2i_attn [T, P] + i2t_attn [P, T]
├── token_patch_heatmap.png        # Text tokens × Image patches heatmap
├── patch_importance.png           # 3-panel: original + patch map + overlay
├── token_patch_topk.json          # Top-20 token-patch pairs with scores
└── cross_attention_summary.json   # Statistics: mean/max/entropy, top-5 tokens, top-5 patches
```

---

## 3. Architecture — No Changes Required

The cross-attention weights are extracted by calling the sub-modules directly:

```python
with torch.no_grad():
    _, _, text_tokens, text_pad = model.text_model(..., return_tokens=True)
    image_patches, patch_mask = model.image_model.forward_features(...)
    t = model.text_proj(text_tokens.float())
    i = model.image_proj(image_patches.float())
    _, t2i_attn = model.cross_attn_t2i(query=t, key=i, value=i, key_padding_mask=~patch_mask)
```

This bypasses `model.forward()` which discards the weights. The computation is identical — same projections, same attention modules, same key_padding_masks. Zero architecture modification.

---

## 4. Engineering Decisions

### 4.1 Head averaging

`nn.MultiheadAttention` returns `[B, T, P]` (averaged over heads) by default. The implementation handles both averaged and per-head formats, falling back to head averaging for visualization.

### 4.2 Padding trimming

Cross-attention matrices are trimmed to `[T_real, P]` where `T_real` = actual tokens (excluding padding). Patches are not trimmed because `patch_mask` is always all-True in the current architecture.

### 4.3 Patch importance via i2t_attn

Patch importance is computed from `i2t_attn` (image→text direction): `importance[p] = sum(i2t_attn[p, :])`. This measures how much each patch distributes attention to text tokens — patches that strongly attend to content words are more "engaged."

### 4.4 Backward compatibility

All existing PhoBERT self-attention outputs are preserved unchanged. Cross-attention outputs go to a separate `cross_attention/` directory. No existing artifact format or API was modified.

---

## 5. Compatibility with Future Phases

| Phase | Status | Notes |
|---|---|---|
| Phase 6 (Case Study) | Compatible | New artifacts in `cross_attention/` can be loaded for combined figures |
| Phase 7 (Report) | Compatible | Summary JSON provides statistics for report tables |
| Phase 8 (Thesis Viz) | Compatible | Heatmap and overlay PNGs are thesis-quality (configurable DPI) |

---

## 6. Remaining Limitations

1. **Head averaging:** Individual head patterns are lost in the averaged view. Per-head visualization could reveal specialized heads.
2. **Not target-specific:** Cross-attention is computed once during the forward pass, not per-target. SHAP (Phase 4) provides target-specific modality analysis.
3. **Patch labels are grid indices:** `(row, col)` labels don't carry semantic meaning. Combining with Grad-CAM overlays (Phase 6) provides richer interpretation.
4. **Long sequences:** The T×P heatmap is skipped for sequences > 60 tokens. The summary JSON and top-K pairs are always generated.

---

## 7. Cross-Attention Visualization Upgrade

### Why the upgrade

The initial cross-attention outputs were raw JSON files and a generic patch importance map. For a 139-token × 49-patch matrix, the full heatmap is unreadable. The upgrade implements a **Top-K** visualization strategy: instead of showing all 6811 pairs, it highlights only the most informative token-patch relationships with thesis-quality figures.

### Newly added visualizations

| Function | Output | Description |
|---|---|---|
| `plot_token_to_patches()` | `token_overlay_{rank}_{token}.png` + grid | For each top token, shows which patches it attends to with zoomed crops |
| `plot_patch_to_tokens()` | `patch_{idx}_token_explanation.png` | For each top patch: highlighted patch, zoom, top-token bar chart |
| `plot_topk_heatmap()` | `topk_token_patch_heatmap.png` | Top-10×10 readable annotated heatmap |
| `plot_bipartite_graph()` | `token_patch_bipartite_graph.png` | Tokens (left) ↔ Patches (right) with weighted edges |

### Top-K strategy

- Full T×P heatmaps are skipped for T>60 (unreadable at 139×49)
- All thesis figures use Top-K selection (default K=5 for tokens/patches, K=15 for edges)
- Raw matrices are always saved in `.npz` for any downstream analysis

### Interpretation guidelines

| Figure | Question answered |
|---|---|
| Token→Patch overlay | "Which image regions does the word 'ngon' focus on?" |
| Patch→Token explanation | "Which words describe this food region?" |
| Top-K heatmap | "What are the strongest token-patch associations?" |
| Bipartite graph | "What are the dominant cross-modal connections?" |

### Patch visualization rule

Whenever a patch index is mentioned visually, it is always shown highlighted on the original image with a zoomed crop. No figure requires the reader to mentally locate a patch.

---

## 8. Batch Cross-Attention Visualization for 15 Samples

### Why the previous notebook only generated full visuals for sample_0000

The upgraded visualization functions (`plot_token_to_patches`, `plot_patch_to_tokens`, `plot_topk_heatmap`, `plot_bipartite_graph`) were called in standalone notebook cells using the demo sample variable (`sample`/`demo_idx`) — which is always sample_0000. The batch loop (Step 14d) called `CrossAttentionExplainer.explain_sample()` which only generated the basic outputs (raw .npz, summary JSON, patch importance).

### What was changed

1. **`CrossAttentionExplainer.explain_sample()`** now calls all upgraded visualization functions internally (token→patch, patch→token, top-K heatmap, bipartite graph) with per-function try/except for resilience.

2. **Notebook restructured**: The standalone demo cells (14c-2, 14c-3) were replaced with a unified batch cell that processes all 15 samples. Inline display is limited to `NUM_DISPLAY_SAMPLES = 3`.

3. **Validation cell** added: checks that all 15 sample folders contain the expected files and prints a pass/fail table.

### How the sample loop works

```python
for batch_idx, sidx in enumerate(SAMPLE_INDICES):
    r = ca_explainer.explain_sample(sample=s, sample_id=sid)
    # explain_sample now generates ALL visualizations internally
    if batch_idx < NUM_DISPLAY_SAMPLES:
        # display inline
    else:
        # print summary only
```

### Files generated per sample

Each `cross_attention/{sample_id}/` folder contains ~15 files:
- `cross_attention_raw.npz`, `cross_attention_summary.json`
- `token_patch_heatmap.png`, `patch_importance.png`, `token_patch_topk.json`
- `topk_token_patch_heatmap.png`, `token_patch_bipartite_graph.png`
- `top_tokens_patch_overlay_grid.png`
- `token_overlay_1_*.png` through `token_overlay_5_*.png`
- `patch_*_token_explanation.png` (×5 patches)

### Validation strategy

Expected core files are checked per sample. Missing files are reported with full paths. Each sample is marked PASS or FAIL.

---

## 9. Summary

### Existing outputs preserved
All PhoBERT self-attention outputs in `attention/` are unchanged. Cross-attention outputs are backward compatible.
