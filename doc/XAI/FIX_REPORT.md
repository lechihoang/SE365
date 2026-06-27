# Phase 1 Fix Report — Runtime Errors

## 1. Root Cause

### Error 1: PhoBERT Attention Extraction (`sdpa` backend)

**Symptom:** Notebook prints `[transformers] sdpa attention does not support output_attentions=True` and then `AssertionError: Expected 12 layers, got 0`.

**Root cause:** Recent versions of HuggingFace `transformers` default to `sdpa` (Scaled Dot-Product Attention) backend for RoBERTa-based models like PhoBERT. The `sdpa` backend is faster but does not return attention weight matrices when `output_attentions=True` is requested. Instead it returns an empty tuple, causing the assertion to fail.

**Why this wasn't caught initially:** The `TextModel.py` constructor calls `AutoModel.from_pretrained(model_name)` without specifying `attn_implementation`. At the time the code was written, the default was `eager` attention. The switch to `sdpa` happened in a transformers library update.

### Error 2: Swin-B Feature Map Shape (`[B, H, W, C]` not recognized)

**Symptom:** Notebook prints `encoder.norm: torch.Size([1, 7, 7, 1024])` then `AssertionError: Could not find spatial feature map layer!`.

**Root cause:** The V9 verification cell only checked for two formats:
- `[B, C, H, W]` (4D, dim 1 == 1024) — standard channels-first
- `[B, N, C]` (3D, dim -1 == 1024) — sequence format

But Swin-B in timm outputs `[B, H, W, C]` (4D, dim -1 == 1024) — channels-last. This is a valid spatial feature map but the detection logic didn't handle it.

---

## 2. Files Modified

| File | Changes |
|---|---|
| `xai/utils.py` | Added `enable_eager_attention()`, `normalize_feature_map_to_bchw()`, `xai_mode` parameter to `load_model()` |
| `xai/__init__.py` | Added exports for the two new functions |
| `xai/notebooks/Phase1_Infrastructure_Verification.ipynb` | Rewrote V8 cell (graceful error handling + eager attention), rewrote V9 cell (uses `normalize_feature_map_to_bchw`), updated V2 cell (passes `xai_mode=True`) |

No changes to `Models/TextModel.py`, `Models/ImageModel.py`, `Models/CrossAttentionFusion.py`, or any training code.

---

## 3. Fixes Applied

### Fix 1: `enable_eager_attention(model)` in `xai/utils.py`

Added a function that patches the text encoder's attention layers in-place from `RobertaSdpaSelfAttention` to `RobertaSelfAttention` (eager implementation). The patching:

1. Updates `encoder.config._attn_implementation` and `encoder.config.attn_implementation` to `'eager'`
2. Iterates through all encoder layers
3. For each layer whose attention class name contains `'Sdpa'` or `'Flash'`, creates a new `RobertaSelfAttention` instance with identical weights
4. Replaces the attention module in-place

This approach:
- Does NOT modify `TextModel.py` — backward compatibility preserved
- Does NOT affect model weights — `load_state_dict` from the eager attention module produces identical parameters
- Does NOT affect predictions — eager and sdpa produce the same output, just with different internal computation paths
- Is called automatically by `load_model(xai_mode=True)` (default)

### Fix 2: `normalize_feature_map_to_bchw()` in `xai/utils.py`

Added a function that accepts feature map tensors in any of four formats and converts to `[B, C, H, W]`:

| Input Format | Detection Rule | Conversion |
|---|---|---|
| `[B, C, H, W]` | 4D, dim 1 == expected_channels | No conversion needed |
| `[B, H, W, C]` | 4D, dim 3 == expected_channels | `permute(0, 3, 1, 2)` |
| `[B, N, C]` | 3D, dim 2 == expected_channels, N is perfect square | `permute(0,2,1).reshape(B,C,H,W)` |
| `[B, C, N]` | 3D, dim 1 == expected_channels, N is perfect square | `reshape(B,C,H,W)` |

Returns both the converted tensor and a metadata dict documenting the detected format, reshape rule, and target shape.

### Fix 3: Notebook V8 cell — graceful error handling

The V8 cell now:
- Wraps attention extraction in a try/except
- If attentions are empty (sdpa still active), marks V8 as FAILED with clear reason instead of crashing
- Reads `num_layers` and `num_heads` from the actual output instead of hardcoded constants
- Records detailed diagnostics in `verification_report.json`

### Fix 4: Notebook V9 cell — uses `normalize_feature_map_to_bchw()`

The V9 cell now:
- Uses the reusable helper function instead of inline shape detection
- Records `detected_format` (e.g., `"BHWC"`) and `reshape_rule` (e.g., `"permute(0, 3, 1, 2)"`) in the verification report
- These findings are consumed by Phase 2 Grad-CAM to configure the reshape transform

---

## 4. Compatibility

### Training pipeline: UNCHANGED
Zero modifications to `Models/TextModel.py`, `Models/ImageModel.py`, `Models/CrossAttentionFusion.py`, `Trainer.py`, `test.py`, `main.py`, or `Config.py`.

### Existing experiment notebooks: UNCHANGED
No experiment notebook imports from `xai/`. The `enable_eager_attention` function only affects models loaded via `xai.utils.load_model(xai_mode=True)`.

### `load_model()` backward compatibility
The new `xai_mode` parameter defaults to `True`. If someone calls `load_model(exp_dir, xai_mode=False)`, the model loads identically to the previous version without any attention patching. Predictions are identical regardless of `xai_mode` because eager and sdpa attention produce the same output.

---

## 5. Future Phase Impact

### Phase 2 (Grad-CAM)
- `normalize_feature_map_to_bchw()` can be imported directly: `from xai.utils import normalize_feature_map_to_bchw`
- V9's verification report documents the exact layer name (`encoder.norm`), detected format (`BHWC`), and reshape rule — Phase 2 reads this to configure `pytorch-grad-cam`'s `SwinTransformerReshapeTransform`
- The function handles all timm output formats, so if the backbone changes (e.g., ConvNeXt which uses `BCHW`), Phase 2 code works without modification

### Phase 3 (Attention Visualization)
- `enable_eager_attention()` ensures `output_attentions=True` returns real attention weights
- V8's verification report confirms the exact number of layers and heads — Phase 3 reads this instead of using hardcoded constants
- The function can be imported directly: `from xai.utils import enable_eager_attention`

---

## 6. Remaining Limitations

1. **`enable_eager_attention` relies on internal transformers class names.** If HuggingFace renames `RobertaSdpaSelfAttention` in a future release, the detection logic (`'Sdpa' in cls_name`) may need updating. The function prints how many layers it patched, so a count of 0 is an early warning.

2. **The eager attention patch replaces modules in-place.** If `model.text_model.encoder` is used in a multi-threaded context, the replacement is not thread-safe. This is not an issue for single-sample XAI analysis in a notebook.

3. **Swin-B feature map resolution is 7x7** for 224x224 input. This is a model architecture limitation, not a code issue. Grad-CAM heatmaps will be coarse. Documented in Phase 2 proposal.
