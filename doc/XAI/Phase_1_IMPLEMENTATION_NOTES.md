# Phase 1: Implementation Notes (V2)

This document summarizes the V2 implementation of Phase 1 (XAI Infrastructure),
updated for the **token × patch CrossAttentionFusion** architecture on branch `xai-v3`.

---

## 1. Proposal Compliance

| Item | Status | Notes |
|---|---|---|
| `xai/__init__.py` | Unchanged | Exports all public API as specified |
| `xai/config.py` | Unchanged | All constants correct for new architecture (dims unchanged) |
| `xai/utils.py` | Unchanged | All functions compatible with new architecture |
| `get_device()` | Unchanged | |
| `set_seed()` | Unchanged | |
| `get_tokenizer()` | Unchanged | |
| `get_image_processor()` | Unchanged | |
| `load_model()` | Unchanged | Config-driven reconstruction works with new checkpoint |
| `load_single_sample()` | Unchanged | Preprocessing is architecture-independent |
| `get_prediction()` | Unchanged | `model.forward()` API unchanged |
| `save_figure()` | Unchanged | |
| `save_raw_values()` | Unchanged | |
| Verification notebook | **Updated V2** | 13 checks (V1-V13), was 10 (V1-V10) |

---

## 2. Differences from V1

### 2.1 V10 Forward-Pass Consistency — Rewritten

**V1:** Manually replicated the old CrossAttentionFusion forward pass:
```python
t = model.text_proj(tf).unsqueeze(1)          # [B, 1, 512]
i = model.image_proj(imf).unsqueeze(1)        # [B, 1, 512]
t_out, _ = model.cross_attn_t2i(query=t, key=i, value=i)
fused = torch.cat([t_out.squeeze(1), i_out.squeeze(1)], dim=1)
```

**V2:** Replicates the new token × patch architecture:
```python
_, _, text_tokens, text_pad = model.text_model(..., return_tokens=True)
image_patches, patch_mask = model.image_model.forward_features(...)
t = model.text_proj(text_tokens)              # [B, T, 512]
i = model.image_proj(image_patches)           # [B, P, 512]
t_out, _ = model.cross_attn_t2i(query=t, key=i, value=i, key_padding_mask=i_kpm)
# + masked mean pooling
fused = torch.cat([t_pooled, i_pooled], dim=1)
```

### 2.2 Three New Verification Checks Added

| Check | What it verifies |
|---|---|
| **V11** | `TextModel.forward(return_tokens=True)` returns 4 values: preds, features, tokens `[B, T, 768]`, pad_mask `[B, T]` |
| **V12** | `ImageModel.forward_features()` returns patches `[B, P, D]` and patch_mask `[B, P]` |
| **V13** | Cross-attention weights are `[B, T, P]` (or `[B, 8, T, P]`), NOT trivially 1.0 |

### 2.3 Branch Updated

All notebook clone cells use `xai-v3` (was `visualization` in V1).

### 2.4 Notebook Header Updated

Title now says "V2", table shows "token × patch" fusion, check count is 13.

---

## 3. Cross-Attention V3 Compatibility

### What changed in the architecture

| Component | Old | New |
|---|---|---|
| Cross-attention inputs | `[B, 1, 512]` (single vector) | `[B, T, 512]` × `[B, P, 512]` (token × patch) |
| Attention weights | `[B, 8, 1, 1]` = trivially 1.0 | `[B, 8, T, P]` = real attention |
| Pooling | `squeeze(1)` | `masked_mean` with padding masks |
| TextModel API | `forward()` returns 2 values | `forward(return_tokens=True)` returns 4 values |
| ImageModel API | Only `forward()` | New `forward_features()` for patch tokens |

### What did NOT change

- Fused vector: still `[B, 1024]` (512 text-origin + 512 image-origin)
- Prediction head: identical `Sequential(Linear(1024,512), ReLU, DO, Linear(512,256), ReLU, Linear(256,5))`
- Model forward signature: `model(input_ids, attention_mask, pixel_values, num_images)` unchanged
- All `xai/utils.py` functions: no changes needed
- All `xai/config.py` constants: no changes needed

### Why xai/utils.py needed no changes

The utility functions interact with the model through its public forward API, which hasn't changed. `load_model()` loads state dicts generically. `get_prediction()` calls `model(...)` and reads the output. `load_single_sample()` prepares input tensors, not model internals.

---

## 4. Engineering Decisions

### 4.1 V10 replicates `_masked_mean` inline

Rather than importing `CrossAttentionFusion._masked_mean`, V10 replicates the logic inline. This avoids a private method dependency and makes the verification self-documenting.

### 4.2 V13 handles both head-averaged and per-head attention

`nn.MultiheadAttention` returns `[B, T, P]` by default (averaged over heads) or `[B, 8, T, P]` if `average_attn_weights=False`. V13 handles both formats.

### 4.3 No changes to existing xai/ Python modules

`xai/config.py`, `xai/utils.py`, and `xai/__init__.py` required zero modifications. The architecture change only affects:
- How the model internally processes data (token × patch vs single vector)
- The V10 manual forward replication in the notebook
- The cross-attention triviality finding (now reversed — V13 confirms non-trivial)

---

## 5. Compatibility with Phase 2–8

| Phase | Impact | Notes |
|---|---|---|
| Phase 2 (Grad-CAM) | None | Hooks on `encoder.norm` unchanged; gradient flow differs but implementation is correct |
| Phase 3 (Attention) | Module docstring updated | Cross-attention note updated; Phase 3 notebook Step 14 rewritten |
| Phase 4 (SHAP) | None | Fused vector `[B, 1024]` unchanged; hook on `model.head` unchanged |
| Phase 5 (LIME) | None | Black-box wrapper; model forward API unchanged |
| Phase 6 (Case Study) | None | Consumer/orchestrator; no architecture dependency |
| Phase 7 (Report) | Text update needed | Architecture descriptions should reference token × patch |
| Phase 8 (Thesis Viz) | Diagram update needed | Architecture diagrams should show T×P attention |

### Downstream changes already made

- `xai/attention_explainer.py`: Module docstring updated (cross-attention note)
- `xai/attention_explainer.py`: Metadata `cross_attention_note` field updated
- Phase 3 notebook: Step 14 rewritten to verify new cross-attention weights

---

## 6. Remaining Limitations

1. **V10 does not test with `average_attn_weights=False`:** The per-head cross-attention weights `[B, 8, T, P]` could provide head-specific visualization, but V10 uses the default (averaged) mode.

2. **Cross-attention visualization not yet implemented:** V13 verifies that weights are extractable and non-trivial, but actual visualization (e.g., "which image patch does the word 'ngon' attend to?") is deferred to Phase 3 update.

3. **No automated regression test:** The notebook must be run manually on Colab to verify. There is no CI/CD integration.

---

## 7. Summary

### Phase 1 V2 changes
- Notebook: V10 rewritten for token × patch architecture
- Notebook: V11, V12, V13 added (TextModel tokens, ImageModel patches, cross-attention weights)
- Notebook: Branch updated to `xai-v3`
- Notebook: Header updated (V2 title, 13 checks)
- Python modules: Zero changes to `xai/config.py`, `xai/utils.py`, `xai/__init__.py`
- Downstream: `attention_explainer.py` docstring/metadata updated

### Is Phase 1 V2 ready for Phase 2?
**Yes.** All prerequisites verified:
- Model loads correctly (V2)
- Preprocessing matches training pipeline (V6)
- Swin-B spatial feature map layer identified (V9)
- PhoBERT attention extraction verified (V8)
- Token × patch forward-pass decomposition confirmed (V10)
- TextModel `return_tokens` API verified (V11)
- ImageModel `forward_features` API verified (V12)
- Cross-attention weights extractable and non-trivial (V13)
