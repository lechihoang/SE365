# Phase 2: Grad-CAM — Implementation Notes (V2)

Updated for the **token × patch CrossAttentionFusion** architecture on branch `xai-v3`.

---

## 1. Proposal Compliance

| Item | Status | Notes |
|---|---|---|
| `xai/gradcam_explainer.py` | Unchanged | All classes and functions correct for new architecture |
| `GradCAMExplainer` class | Unchanged | High-level orchestrator works as-is |
| `MultiTargetScoreWrapper` | Unchanged | Wraps full model; forward API unchanged |
| `SwinReshapeTransform` | Unchanged | Feature map format handling unchanged |
| `compute_gradcam_for_image` | Unchanged | Hook-based manual Grad-CAM |
| `overlay_cam_on_image` | Unchanged | Visualization utility |
| `create_5target_comparison` | Unchanged | Comparison figure layout |
| `find_target_layer` | Unchanged | `encoder.norm` still the correct target |
| `diagnose_target_gradients` | Unchanged | Gradient similarity analysis |
| Phase 2 notebook | **Updated V2** | Branch → `xai-v3`, header updated |

---

## 2. Architecture Changes — Grad-CAM Impact

### Why `gradcam_explainer.py` needed no changes

Grad-CAM hooks onto `model.image_model.encoder.norm`, which is **inside the Swin-B encoder** — upstream of the cross-attention fusion layer. The hook captures activations and gradients at the encoder level, before cross-attention operates.

```
pixel_values → Swin-B encoder → encoder.norm [GRAD-CAM HERE] → forward_features()
                                                                       ↓
                                                              image_proj → cross-attention
```

The architecture change (single-vector → token×patch cross-attention) happens **after** the target layer. Therefore hook placement, activation capture, and the implementation itself are unchanged.

### What IS different (behavioral, not code)

The **gradient values** at `encoder.norm` may differ because the gradient flow path now goes through token×patch cross-attention with padding masks instead of trivial 1×1 cross-attention. Heatmaps generated with V2 may show different (potentially more target-specific) patterns.

---

## 3. Grad-CAM Attachment Point

```
CrossAttentionFusion.forward():
  ├── image_model.forward_features(pixel_values, num_images)
  │     ├── encoder.forward_features(pixel_values)
  │     │     ├── patch_embed → layers[0..3] → encoder.norm  ◄── GRAD-CAM TARGET
  │     │           output: [B*N, 7, 7, 1024] (BHWC)
  │     └── reshape + masked average → patches [B, P=49, D=1024]
  │
  ├── text_model(return_tokens=True) → text_tokens [B, T, 768]
  ├── text_proj / image_proj → [B, T, 512] / [B, P, 512]
  ├── cross_attn_t2i / cross_attn_i2t (with key_padding_mask)
  ├── masked_mean pooling → t_pooled, i_pooled
  ├── cat → fused [B, 1024]
  └── head → preds [B, 5]
```

Gradient flow: `preds[0, target_idx] → head → fused → i_pooled → cross_attn → image_proj → encoder.norm`

---

## 4. Generated Artifacts (unchanged)

Per sample: overlays, 5-target comparison, raw CAMs (`.npz`), metadata JSON.

Batch: `gradcam_batch_summary.json`.

---

## 5. Engineering Decisions

- Manual hook-based Grad-CAM (no `pytorch-grad-cam` dependency)
- `normalize_feature_map_to_bchw()` handles BCHW/BHWC/BNC/BCN
- Multi-image context preserved via full forward pass with per-image hook slicing

---

## 6. Compatibility with Future Phases

Phase 6 reads Grad-CAM artifacts for combined figures — artifact format unchanged.
Phase 7 references Grad-CAM PNGs — naming convention unchanged.
All other phases are independent.

---

## 7. Remaining Limitations

1. **7×7 resolution** — coarse but sufficient for region-level localization
2. **Target similarity** — shared encoder dilutes per-target gradients; use SHAP for target-specific analysis
3. **Gradient path changed** — heatmaps should be regenerated on Colab to capture new patterns

---

## 8. Summary

### V2 changes
- Notebook: branch → `xai-v3`, header updated with architecture note
- Python module: **zero changes** — Grad-CAM hooks upstream of cross-attention redesign
- Artifacts: format and naming unchanged, compatible with Phase 6-8

### Ready for use?
**Yes.** Re-run on Colab to generate heatmaps with the new gradient flow patterns.
