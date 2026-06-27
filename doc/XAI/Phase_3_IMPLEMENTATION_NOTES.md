# Phase 3: Attention Visualization — Implementation Notes (V2)

Updated for the **token × patch CrossAttentionFusion** architecture on branch `xai-v3`.

---

## 1. Proposal Compliance

| Item | Status | Notes |
|---|---|---|
| `xai/attention_explainer.py` | Updated | Module docstring and metadata `cross_attention_note` updated |
| `extract_phobert_attention()` | Unchanged | Bypasses TextModel.forward(), calls encoder directly |
| `aggregate_attention()` — 3 strategies | Unchanged | last_layer_mean, last_4_layers_mean, attention_rollout |
| `cls_token_importance()` | Unchanged | Extracts CLS row, excludes special tokens |
| `merge_subword_attention()` | Unchanged | Handles both @@ and space-prefix conventions |
| `plot_attention_heatmap()` | Unchanged | seaborn/matplotlib with Vietnamese support |
| `plot_cls_importance_bar()` | Unchanged | Horizontal bar chart, top-k, sorted |
| `compute_attention_sink_ratio()` | Unchanged | Warns if >0.3 |
| `AttentionExplainer` class | Updated | Metadata cross_attention_note reflects new architecture |
| `inspect_tokenization()` | Unchanged | Debugging utility for PhoBERT tokenization |
| Cross-attention verification | **Rewritten** | Notebook Step 14 now verifies token×patch weights are NOT trivial |
| Phase 3 notebook | **Updated V2** | Branch → `xai-v3`, header updated, Step 14 rewritten |

---

## 2. Differences from V1

### 2.1 Module docstring updated

**V1:** "Cross-attention projects to single vectors [B, 1, 512]... trivially 1.0... completely uninformative"
**V2:** "Cross-attention now uses token-level × patch-level... informative and visualizable"

### 2.2 Metadata field updated

**V1:** `cross_attention_note: "Cross-attention weights are trivially 1.0 (Q=[B,1,512], K=[B,1,512])..."`
**V2:** `cross_attention_note: "Cross-attention now uses token×patch attention (Q=[B,T,512], K=[B,P,512])..."`

### 2.3 Notebook Step 14 rewritten

**V1:** Manually replicated old architecture with `unsqueeze(1)`, proved trivially 1.0, asserted `all_ones = True`
**V2:** Uses new APIs (`return_tokens=True`, `forward_features()`), extracts real T×P attention weights, verifies NOT trivially 1.0

### 2.4 Final summary check updated

**V1:** Check `'Cross-attn trivial' → all_ones`
**V2:** Check `'Cross-attn informative' → not all_ones`

---

## 3. Cross-Attention V3 Compatibility

### What changed

The cross-attention is now token × patch, producing `[B, 8, T, P]` attention weights (or `[B, T, P]` head-averaged). These weights are real and informative.

### What did NOT change

- PhoBERT self-attention extraction: completely unchanged (encoder is called directly)
- Attention aggregation strategies: all 3 strategies work identically
- CLS importance: identical (CLS token is at position 0 regardless of architecture)
- Subword merging: identical (depends on tokenizer, not architecture)
- Attention sink detection: identical

### Why `attention_explainer.py` needed minimal changes

PhoBERT self-attention is computed **inside the text encoder**, before any fusion happens. The architecture redesign only changed the fusion layer (cross-attention), not the text encoder internals. Therefore all attention extraction, aggregation, and visualization code works identically.

---

## 4. Engineering Decisions

### 4.1 Cross-attention visualization deferred

V13 in Phase 1 and Step 14 in Phase 3 both verify that cross-attention weights are extractable and non-trivial. Actual cross-attention visualization (e.g., "which image patch does the word 'ngon' attend to?") is a significant new feature deferred to a future Phase 3 update.

### 4.2 No code duplication with Phase 1

Phase 1 V13 and Phase 3 Step 14 both extract cross-attention weights using the same pattern. This is intentional — Phase 1 verifies infrastructure, Phase 3 uses it for analysis.

---

## 5. Compatibility with Future Phases

| Phase | Status | Notes |
|---|---|---|
| Phase 6 (Case Study) | Compatible | Artifact format unchanged |
| Phase 7 (Report) | Compatible | Metadata format unchanged |
| Phase 8 (Thesis Viz) | Compatible | Figure format unchanged |

---

## 6. Remaining Limitations

1. **Target-agnostic:** PhoBERT self-attention is the same for all 5 targets.
2. **Attention ≠ explanation:** Attention shows information flow, not causal importance.
3. **Cross-attention visualization not yet implemented:** Verified extractable but not visualized.
4. **Subword merging heuristic:** Rule-based, may have edge cases.

---

## 7. Summary

### V2 changes
- `attention_explainer.py`: docstring + metadata updated (2 lines)
- Notebook: header updated (V2, architecture note, branch `xai-v3`)
- Notebook: Step 14 rewritten (cross-attention verification using new APIs)
- Notebook: Final summary check inverted (`informative` instead of `trivial`)

### Ready for use?
**Yes.** PhoBERT self-attention is architecture-independent. Cross-attention weights are verified extractable for future visualization.
