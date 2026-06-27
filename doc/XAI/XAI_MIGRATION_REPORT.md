# XAI Migration Report — Cross-Attention Architecture Redesign

**Generated:** 2026-06-27
**Scope:** All XAI Phases (1-8) proposal documents

---

## 1. Summary of Architectural Changes

The `CrossAttentionFusion` module has been **fundamentally redesigned** from a degenerate single-vector cross-attention to a proper token-level / patch-level cross-attention architecture.

### What Changed

| Component | Old (Degenerate) | New (Token×Patch) |
|---|---|---|
| **TextModel.forward()** | Returns `(preds, features)` where features = CLS pooled `[B, 768]` | Now supports `return_tokens=True` returning `(preds, features, tokens, pad_mask)` where tokens = `[B, T, 768]` and pad_mask = `[B, T]` |
| **ImageModel** | Only `forward()` returning pooled `[B, 1024]` | New `forward_features()` returning `(patches, patch_mask)` where patches = `[B, P, D]` and patch_mask = `[B, P]` |
| **Cross-attention inputs** | Single-vector: Q=`[B, 1, 512]`, K=`[B, 1, 512]` — **trivially 1.0** | Token-level: Q=`[B, T, 512]`, K=`[B, P, 512]` — **real attention** |
| **Cross-attention outputs** | `[B, 1, 512]` per direction, squeezed to `[B, 512]` | `[B, T, 512]` and `[B, P, 512]`, masked-mean-pooled to `[B, 512]` |
| **Padding masks** | None | `key_padding_mask` used for both directions (text pad + image pad) |
| **Fused vector** | `cat([B,512], [B,512])` = `[B, 1024]` — same dimensions but trivial content | `cat(t_pooled, i_pooled)` = `[B, 1024]` — now contains real cross-modal interactions |
| **Freeze/unfreeze** | Inline per-fusion-module | Centralized in `Models/unfreeze.py` |

### What Remained Unchanged

- `self.head` structure: `Sequential(Linear(1024,512), ReLU, Dropout(0.2), Linear(512,256), ReLU, Linear(256,5))` — **identical**
- Fused vector dimension: still `[B, 1024]` (512 text-origin + 512 image-origin)
- Output: still `[B, 5]` regression scores
- Target names, factor names, all config constants — unchanged
- Other fusion modules (FusionModel, GMUFusion, GatedCrossModalFusion, FiLMFusion) — updated freeze/unfreeze to use shared helpers but forward logic is the same pattern
- Dataset, checkpoint format, training pipeline — unchanged

---

## 2. Cross-Attention Architecture Details

### Old Architecture (Degenerate)

```
TextModel.forward() → features [B, 768] (CLS pooled)
ImageModel.forward() → features [B, 1024] (avg pooled)
    ↓
text_proj(768 → 512).unsqueeze(1) → t [B, 1, 512]
image_proj(1024 → 512).unsqueeze(1) → i [B, 1, 512]
    ↓
cross_attn_t2i(Q=t[B,1,512], K=i[B,1,512], V=i) → t_out [B, 1, 512]
cross_attn_i2t(Q=i[B,1,512], K=t[B,1,512], V=t) → i_out [B, 1, 512]
    ↓
ATTENTION IS ALWAYS 1.0 (softmax over single key = trivially 1.0)
    ↓
cat([t_out.squeeze(1), i_out.squeeze(1)]) → fused [B, 1024]
    ↓
head(fused) → [B, 5]
```

### New Architecture (Token × Patch)

```
TextModel.forward(return_tokens=True)
  → features [B, 768], text_tokens [B, T, 768], text_pad [B, T]
ImageModel.forward_features()
  → image_patches [B, P, D], patch_mask [B, P]
    ↓
text_proj(text_tokens) → t [B, T, 512]      (T tokens projected)
image_proj(image_patches) → i [B, P, 512]   (P patches projected)
    ↓
key_padding_mask for text: text_pad (True = PAD)
key_padding_mask for image: ~patch_mask (True = PAD)
    ↓
cross_attn_t2i(Q=t[B,T,512], K=i[B,P,512], V=i, kpm=i_kpm)
  → t_out [B, T, 512]  (each text token attended to ALL image patches)
cross_attn_i2t(Q=i[B,P,512], K=t[B,T,512], V=t, kpm=t_kpm)
  → i_out [B, P, 512]  (each image patch attended to ALL text tokens)
    ↓
ATTENTION IS NOW MEANINGFUL (T×P matrix, not 1×1)
    ↓
t_pooled = masked_mean(t_out, ~text_pad) → [B, 512]
i_pooled = masked_mean(i_out, patch_mask) → [B, 512]
    ↓
fused = cat(t_pooled, i_pooled) → [B, 1024]
    ↓
head(fused) → [B, 5]
```

### Key Dimensional Summary

| Tensor | Old Shape | New Shape | Notes |
|---|---|---|---|
| Cross-attention Q (text→image) | `[B, 1, 512]` | `[B, T, 512]` | T = text sequence length (~256) |
| Cross-attention K (text→image) | `[B, 1, 512]` | `[B, P, 512]` | P = image patches (49 for Swin-B 7×7) |
| Attention weights (text→image) | `[B, 8, 1, 1]` = always 1.0 | `[B, 8, T, P]` = real attention | **Major change** |
| Cross-attention output | `[B, 1, 512]` | `[B, T, 512]` or `[B, P, 512]` | Pooled via masked_mean to `[B, 512]` |
| Fused vector | `[B, 1024]` | `[B, 1024]` | Same dimension, but richer content |

---

## 3. Impact Analysis per Phase

| Phase | Impact Level | Reason |
|---|---|---|
| **Phase 1 (Infrastructure)** | Minor Update | `load_model()` works the same. But `TextModel.forward()` signature changed — if any verification step calls it directly, it needs `return_tokens=False` (default). Feature shapes are the same for pooled output. |
| **Phase 2 (Grad-CAM)** | Minor Update | Grad-CAM hooks on `image_model.encoder.norm` — unchanged. The image encoder path is the same. But the gradient flow is now through token-level cross-attention instead of single-vector, which may produce slightly different Grad-CAM heatmaps. |
| **Phase 3 (Attention)** | **Major Update** | Cross-attention is **no longer trivially 1.0**. The Phase 3 finding that "cross-attention weights are architecturally trivial" is now **WRONG**. Cross-attention weights `[B, 8, T, P]` are now informative and should be extracted and visualized. PhoBERT self-attention is still available but cross-attention visualization becomes a valuable addition. |
| **Phase 4 (SHAP)** | Minor Update | The fused vector is still `[B, 1024]` with the same 0:512 / 512:1024 split. The head is identical. SHAP on the fusion head works the same. But the fused embedding extraction hook is now after `masked_mean` pooling, not after `squeeze(1)`. The hook on `model.head` still works. |
| **Phase 5 (LIME)** | Minor Update | LIME wraps the entire model as black box. The model's forward signature is the same (`input_ids, attention_mask, pixel_values, num_images`). LIME wrappers work without change. |
| **Phase 6 (Case Study)** | Minor Update | Pure consumer / orchestrator. No direct model architecture dependency. |
| **Phase 7 (Report)** | Minor Update | Architecture description text needs updating. "Degenerate cross-attention" → "Token × Patch cross-attention." |
| **Phase 8 (Thesis Viz)** | Minor Update | Architecture diagrams need updating. |

---

## 4. Proposal Inconsistencies Found

### 4.1 Phase 3 — Cross-Attention Triviality (CRITICAL)

**Old proposal (Phase_3_Attention_Proposal.md, Section 5.2):**
> "SECONDARY (NOT USEFUL): Cross-Attention in CrossAttentionFusion... Both query and key have sequence length 1... attention is trivially 1.0"

**Current implementation:**
Cross-attention now operates on `[B, T, 512]` × `[B, P, 512]` — attention weights are `[B, 8, T, P]`, a real T×P attention matrix that IS useful for visualization.

**Required update:** Remove "NOT USEFUL" classification. Add cross-attention as a PRIMARY visualization target alongside PhoBERT self-attention.

### 4.2 All Proposals — CrossAttentionFusion.forward() Description

**Old proposals describe:**
```python
_, text_feat = self.text_model(input_ids, attention_mask)   # [B, 768]
_, image_feat = self.image_model(pixel_values, num_images)  # [B, 1024]
t = self.text_proj(text_feat).unsqueeze(1)   # [B, 1, 512]
i = self.image_proj(image_feat).unsqueeze(1)  # [B, 1, 512]
```

**Current implementation:**
```python
_, _, text_tokens, text_pad = self.text_model(input_ids, attention_mask, return_tokens=True)
image_patches, patch_mask = self.image_model.forward_features(pixel_values, num_images)
t = self.text_proj(text_tokens)     # (B, T, hidden)
i = self.image_proj(image_patches)  # (B, P, hidden)
```

### 4.3 Phase 2 — Grad-CAM Target Layer

**Old proposal:** States ImageModel only has `forward()`.
**Current:** ImageModel now has both `forward()` and `forward_features()`. The `forward_features()` returns patch-level features `[B, P, D]` with spatial information preserved, which could be an alternative Grad-CAM target.

### 4.4 Phase 4 — Fused Vector Description

**Old proposal:** "dims 0:512 are text features AFTER cross-attending to image (t_out)"
**Current:** Still true dimensionally, but `t_out` is now `masked_mean` of token-level cross-attention output, not a squeezed single vector. The semantic meaning is richer.

---

## 5. XAI Impact

### Grad-CAM
- Attachment point (`encoder.norm`) is unchanged — feature maps are still produced the same way
- Gradient flow now goes through token×patch cross-attention instead of degenerate 1×1 attention
- Heatmaps may differ from old architecture — re-run is needed
- The `forward_features()` method could provide an alternative Grad-CAM target with patch-level spatial features

### Attention Visualization
- **PhoBERT self-attention:** Unchanged — still accessible via `output_attentions=True`
- **Cross-attention:** NOW INFORMATIVE. The `[B, 8, T, P]` attention weights from `cross_attn_t2i` show which image patches each text token attends to. This is the biggest XAI opportunity from the architecture change.
- Implementation: Extract cross-attention weights by modifying the forward pass or using hooks. The `nn.MultiheadAttention` already returns weights when `need_weights=True` (default), but the current code discards them (`_, _ = self.cross_attn_t2i(...)`).

### SHAP
- Fused vector structure unchanged (`[B, 1024]`, first 512 = text-origin, last 512 = image-origin)
- `model.head` is identical
- SHAP analysis works the same way
- The fused embedding is now more expressive (token×patch interaction), but the SHAP methodology is unchanged

### LIME
- Black-box wrapper is architecture-agnostic
- Model's forward signature is unchanged
- LIME works without any modification

---

## 6. Recommended Implementation Order

The recommended order for re-implementing XAI on the new architecture is:

1. **Phase 1 (Infrastructure)** — Update `TextModel.forward()` verification to test `return_tokens=True` path. Verify `ImageModel.forward_features()` produces correct patch tensors. Add cross-attention weight extraction verification.

2. **Phase 2 (Grad-CAM)** — Re-run with the new architecture. Gradient flow through token×patch attention may produce different (potentially more target-specific) heatmaps.

3. **Phase 3 (Attention)** — **PRIORITY UPGRADE.** Add cross-attention visualization (`[B, 8, T, P]` → "which image patches does each text token attend to?"). This is a major new capability.

4. **Phase 4 (SHAP)** — Minimal changes needed. Re-extract fused embeddings from the new architecture and recompute SHAP.

5. **Phase 5 (LIME)** — No changes needed. LIME is architecture-agnostic.

6. **Phase 6 (Case Study)** — Regenerate case studies with new Phase 2-5 outputs.

7. **Phase 7 (Report)** — Update architecture descriptions. Regenerate report.

8. **Phase 8 (Thesis Viz)** — Update architecture diagrams. Regenerate thesis figures.

---

## 7. Cross-Attention Visualization Opportunity

The new architecture provides a unique XAI capability that the old architecture could not offer:

**Text-to-Image Cross-Attention Map:** For each text token `t_i`, the attention weights `attn[b, h, i, :]` show which image patches it attended to. This directly answers:
- "Which part of the image does the word 'ngon' (delicious) focus on?"
- "Does the price-related text ('giá cao') attend to food presentation or restaurant interior?"

**Image-to-Text Cross-Attention Map:** For each image patch `p_j`, the attention weights `attn[b, h, j, :]` show which text tokens it attended to. This directly answers:
- "Which words does the food region of the image focus on?"
- "Does the restaurant interior attend to atmosphere-related words?"

This is a **completely new XAI method** that was impossible with the degenerate 1×1 architecture. It should be a prominent addition to Phase 3.

---

*End of Migration Report*
