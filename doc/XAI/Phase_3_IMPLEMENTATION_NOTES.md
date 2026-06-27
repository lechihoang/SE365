# Phase 3: Attention Visualization — Implementation Notes

## 1. Proposal Compliance

| Item | Status |
|---|---|
| `xai/attention_explainer.py` created | Implemented |
| `extract_phobert_attention()` | Implemented — bypasses TextModel.forward(), calls encoder directly |
| `aggregate_attention()` — 3 strategies | Implemented: last_layer_mean, last_4_layers_mean, attention_rollout |
| `cls_token_importance()` | Implemented — extracts CLS row, excludes special tokens |
| `merge_subword_attention()` | Implemented — handles both @@ and space-prefix conventions |
| `plot_attention_heatmap()` | Implemented — seaborn/matplotlib with Vietnamese support |
| `plot_cls_importance_bar()` | Implemented — horizontal bar chart, top-k, sorted |
| `compute_attention_sink_ratio()` | Implemented — warns if >0.3 |
| `AttentionExplainer` class | Implemented — generates all artifacts per sample |
| `inspect_tokenization()` | Implemented — debugging utility for PhoBERT tokenization |
| Cross-attention triviality documented | Documented in notebook with mathematical proof |
| Phase 3 notebook | Implemented with 15-sample batch processing |

## 2. Proposal Deviations

### 2.1 No separate `attention_runner.py`

- **Proposal:** Create `xai/attention_runner.py` as a separate orchestration module.
- **Actual:** Runner logic is handled by `AttentionExplainer.explain_sample()` and notebook batch processing, matching Phase 2's pattern where `GradCAMExplainer` + notebook batch loop replaced the separate runner.
- **Why:** Phase 2 established the pattern of putting orchestration in the explainer class + notebook. A separate runner module would be inconsistent with the established codebase pattern.

### 2.2 Eager attention patching done in notebook, not in module

- **Proposal:** Implies attention extraction should "just work."
- **Actual:** The sdpa→eager patching is done inline in the notebook (Step 6), exactly matching Phase 1 and Phase 2 patterns.
- **Why:** The patching modifies the model's internal modules in-place. Doing this inside a library function would be a side-effect that's hard to reason about. The notebook pattern makes it explicit and visible.

### 2.3 Attention stored as float16 .npz

- **Proposal:** Store as float32 .npy files.
- **Actual:** Store as float16 .npz (compressed) to save storage. For L=50 tokens: float32 = 12*12*50*50*4 = 1.4MB, float16 = 0.7MB.
- **Why:** Phase 2 uses .npz for raw CAMs. Matching format. Float16 is sufficient for attention weights (values 0-1, softmax output).

## 3. Engineering Decisions

### 3.1 Padding trimming before storage
The attention tensor is trimmed from [12, 12, 256, 256] to [12, 12, seq_len, seq_len] before saving. Most reviews are 20-60 tokens, reducing storage by 85-97%.

### 3.2 Subword merging strategy
PhoBERT v2 uses a BPE tokenizer. The implementation checks both `@@` continuation markers and space-prefix (`Ġ`) convention, since the exact behavior depends on the tokenizer version. The default merge strategy is `mean`.

### 3.3 Heatmap readability thresholds
- ≤30 tokens: full annotated heatmap with cell values
- 31-60 tokens: heatmap without annotations (color only)
- >60 tokens: CLS bar chart as primary, heatmap saved but not displayed in notebook

### 3.4 Cross-attention triviality
Not implemented as a separate analysis module. Instead, demonstrated inline in the notebook with a mathematical explanation. The result is always [B, 8, 1, 1] = 1.0 because softmax over a single element is trivially 1.0.

## 4. Assumptions

- Model is CrossAttentionFusion with PhoBERT as text encoder
- PhoBERT has 12 layers, 12 heads (verified from config constants)
- Eager attention has been patched before calling extraction functions
- Tokenizer is `vinai/phobert-base-v2` (loaded via `get_tokenizer()`)
- Attention weights are softmax outputs: values in [0, 1], rows sum to ~1.0
- CLS token `<s>` is at position 0

## 5. Known Limitations

1. **Target-agnostic:** PhoBERT self-attention is computed once, independent of which target score is predicted. Same attention for all 5 targets. For target-specific text importance, use LIME Text (Phase 5).
2. **Attention ≠ explanation:** Attention shows information flow, not causal importance. Must be framed carefully in thesis.
3. **Cross-attention trivial:** The architecture's cross-attention is uninformative (always 1.0). Documented as a finding.
4. **Subword merging heuristic:** The merging logic is rule-based and may have edge cases with unusual Vietnamese text.

## 6. Files Created

| File | Description |
|---|---|
| `xai/attention_explainer.py` | Core module: extraction, aggregation, visualization, explainer class |
| `xai/notebooks/Phase3_Attention.ipynb` | Interactive notebook with 15-sample processing |
| `doc/XAI/Phase_3_IMPLEMENTATION_NOTES.md` | This document |

## 7. Files Modified

None. No existing files were modified.

## 8. Reusable Components for Later Phases

| Component | Reusable by |
|---|---|
| `extract_phobert_attention()` | Phase 6 (Case Studies) — combined text+image explanations |
| `merge_subword_attention()` | Phase 5 (LIME Text) — word-level perturbation alignment |
| `cls_token_importance()` | Phase 6, Phase 8 — text evidence in combined figures |
| `AttentionExplainer` | Phase 6 — batch attention extraction for case studies |
| `inspect_tokenization()` | Phase 5 — verify LIME text splitting matches PhoBERT tokens |

## 9. Potential Future Improvements

1. **Gradient-weighted attention:** Multiply attention by gradient of target score for target-specific text maps. Would require gradient computation similar to Grad-CAM.
2. **Head specialization analysis:** Automated classification of head types (syntactic, semantic, positional).
3. **Cross-sample attention statistics:** Aggregate attention patterns across the dataset to find common aspect-attention associations.
