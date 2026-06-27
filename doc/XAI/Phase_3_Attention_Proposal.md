# Phase 3: Attention Visualization for Text Branch

## Implementation Proposal Document

**Phase:** 3 of 8
**XAI Method:** Transformer Self-Attention Visualization
**Target Branch:** Text Branch (PhoBERT)
**Best Model:** Swin-B + PhoBERT (`vinai/phobert-base-v2`) + CrossAttentionFusion + LogCosh
**Status:** Specification Complete --- Ready for Implementation

---

## 1. Purpose

### 1.1 Why This Phase Exists

The multimodal system predicts five quality scores (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`) from Vietnamese restaurant reviews containing both images and text. Phase 2 (Grad-CAM) explains the image branch by highlighting spatial regions. Phase 3 completes the modality-level picture by explaining the text branch.

PhoBERT transforms a sequence of Vietnamese tokens into a single 768-dimensional embedding via 12 layers of self-attention with 12 heads per layer. After training, the text branch is a black box: we know it converts `"Do an ngon nhung gia hoi cao"` into a vector, but we cannot see which tokens interacted, which words the model focused on, or whether aspect-bearing terms like `ngon` (delicious), `gia` (price), or `khong gian` (atmosphere) received meaningful attention.

Attention visualization extracts the internal self-attention weight matrices from PhoBERT and renders them as human-interpretable heatmaps and bar charts. This provides evidence about the information flow inside the text encoder.

### 1.2 Research Motivation

1. **Interpretability of Vietnamese NLP:** PhoBERT is a domain-specific Vietnamese language model. Demonstrating that its attention patterns align with human linguistic intuition about aspect terms strengthens the thesis claim that the model has learned meaningful Vietnamese restaurant review semantics.
2. **Multi-method triangulation:** When combined with LIME Text (Phase 5), attention provides a complementary view. Attention shows internal information flow; LIME shows perturbation-based local importance. Agreement between the two strengthens explanatory claims.
3. **Thesis defense readiness:** Examiners will ask "what did the text branch focus on?" This phase provides concrete, visual answers.

### 1.3 Engineering Motivation

1. **Minimal code change:** PhoBERT (a RoBERTa variant) natively supports `output_attentions=True`. No architectural modification is required.
2. **Reusable infrastructure:** The attention extraction utilities built here will be reused in Phase 6 (Case Studies) and Phase 8 (Thesis Visualization).
3. **Diagnostic value:** Attention patterns can reveal pathological behavior such as attention sinks on special tokens or uniform attention distributions, which are useful for model debugging.

### 1.4 Critical Caveat

**Attention is NOT explanation.** High attention weight between two tokens means they interacted strongly during encoding, but it does not prove that this interaction caused the final prediction. This distinction must be maintained throughout the thesis. Attention visualization is presented as "information flow evidence," never as "causal explanation." Perturbation-based methods (LIME Text, Phase 5) provide the complementary causal perspective.

---

## 2. Objectives

### 2.1 Research Objectives

| ID | Objective | Success Criterion |
|----|-----------|-------------------|
| R1 | Extract and visualize PhoBERT self-attention for Vietnamese restaurant reviews | Heatmaps correctly display token-to-token attention weights with readable Vietnamese tokens |
| R2 | Demonstrate that attention patterns show meaningful information flow for aspect-bearing terms | Qualitative analysis shows `ngon`, `gia`, `cao`, `khong gian`, `nhan vien` receiving contextually appropriate attention |
| R3 | Document that cross-attention weights in CrossAttentionFusion are trivially uninformative | Written analysis with mathematical proof that [B, 8, 1, 1] attention is always 1.0 |
| R4 | Support multi-method triangulation with LIME Text (Phase 5) | Attention-highlighted tokens can be compared with LIME word importance rankings |
| R5 | Frame attention correctly as information flow evidence, not causal explanation | All figures, captions, and text use "information flow" framing |

### 2.2 Engineering Objectives

| ID | Objective | Success Criterion |
|----|-----------|-------------------|
| E1 | Create `xai/attention_explainer.py` with all extraction and aggregation functions | Module passes unit tests, produces correct tensor shapes |
| E2 | Handle PhoBERT BPE subword tokenization with word-level merging | Merged attention produces readable Vietnamese words, not subword fragments |
| E3 | Implement three aggregation strategies: last_layer_mean, last_4_layers_mean, attention_rollout | Each strategy produces a valid [L, L] attention matrix |
| E4 | Generate per-sample artifacts: heatmap PNGs, bar charts, JSON metadata, raw NPY tensors | All artifacts saved to `experiments/EXP_XXX/xai/attention/` |
| E5 | Create reproducible notebook `xai/notebooks/Phase3_Attention.ipynb` | Notebook runs end-to-end with fixed seed, produces identical outputs |
| E6 | Process samples in batch with configurable sample selection | Support for specific sample indices, random sampling, and full-dataset processing |

### 2.3 Expected Contributions

1. First attention-based text explanation for this multimodal Vietnamese review system.
2. Documented finding that cross-attention in the current architecture is architecturally trivial.
3. Subword-to-word merging utility specifically designed for PhoBERT Vietnamese tokenization.
4. Reusable attention extraction pipeline for downstream phases.

---

## 3. Inputs

### 3.1 Model Checkpoint

| Input | Path | Description |
|-------|------|-------------|
| Best model checkpoint | `experiments/EXP_XXX/best_model_train_fusion.pth` | Trained CrossAttentionFusion weights containing `text_model`, `image_model`, and fusion layers |

The checkpoint contains a state dict with keys following the pattern:
- `text_model.encoder.*` -- PhoBERT weights
- `text_model.fc.*` -- Text projection head weights
- `image_model.*` -- Image encoder weights
- `text_proj.*`, `image_proj.*` -- Fusion projection weights
- `cross_attn_t2i.*`, `cross_attn_i2t.*` -- Cross-attention weights
- `head.*` -- Final prediction MLP weights

### 3.2 Dataset Files

| Input | Path | Description |
|-------|------|-------------|
| Test CSV | `data/text/test.csv` | Test split containing `comment_clean` column |
| Validation CSV | `data/text/val.csv` | Validation split (alternative source) |
| Image directory | `data/image/` | Required for full fusion forward pass |

### 3.3 Model Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| `text_model_name` | `vinai/phobert-base-v2` | Experiment config |
| `image_model_name` | `swinv2_base_window12to16_192to256.ms_in22k_ft_in1k` (or equivalent Swin-B) | Experiment config |
| `fusion_type` | `cross_attention` | Experiment config |
| `max_length` | 256 | Tokenizer max sequence length |
| `num_heads` (PhoBERT) | 12 | Fixed by model architecture |
| `num_layers` (PhoBERT) | 12 | Fixed by model architecture |
| `hidden_size` (PhoBERT) | 768 | Fixed by model architecture |

### 3.4 Dependencies on Prior Phases

| Dependency | Phase | What is needed |
|------------|-------|----------------|
| XAI config system | Phase 1 | `xai/config.py` for experiment paths, device selection, seed management |
| XAI utility functions | Phase 1 | `xai/utils.py` for checkpoint loading, figure saving, JSON serialization |

### 3.5 Software Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model inference, attention tensor extraction |
| `transformers` | PhoBERT tokenizer and model, `output_attentions=True` API |
| `numpy` | Array operations, NPY save |
| `matplotlib` | Heatmap and bar chart rendering |
| `seaborn` | Enhanced heatmap styling |
| `json` | Metadata serialization |

---

## 4. Outputs

### 4.1 Per-Sample Artifacts

For each explained sample (identified by dataset index `{id}`):

| Artifact | Format | Shape/Content | Description |
|----------|--------|---------------|-------------|
| `sample_{id}_layer11_mean_heatmap.png` | PNG | Token-by-token heatmap | Last layer (layer 11), mean over 12 heads |
| `sample_{id}_last4layers_mean_heatmap.png` | PNG | Token-by-token heatmap | Mean over layers 8-11, all heads |
| `sample_{id}_cls_importance_bar.png` | PNG | Horizontal bar chart | CLS token's attention to all other tokens (row 0 of attention matrix) |
| `sample_{id}_cls_importance_word_bar.png` | PNG | Horizontal bar chart | Same as above but with subwords merged to words |
| `sample_{id}_topk_tokens.json` | JSON | List of {token, importance, rank} | Top-K most attended tokens from CLS row |
| `sample_{id}_word_importance.json` | JSON | List of {word, importance, rank} | After subword merging, word-level importance |
| `raw/sample_{id}_attention_all_layers.npy` | NPY | `[12, 12, L, L]` float32 | All 12 layers, 12 heads, full attention matrices |
| `raw/sample_{id}_tokens.json` | JSON | List of token strings | Raw subword tokens from PhoBERT tokenizer |
| `metadata/sample_{id}_attention_metadata.json` | JSON | Dict | Review text, predicted scores, true scores, token count, aggregation strategy, timestamp |

### 4.2 Summary Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `attention_summary.json` | JSON | Index of all processed samples, their paths, key statistics |
| `cross_attention_analysis.md` | Markdown | Documented analysis of why cross-attention weights are trivial in this architecture |

### 4.3 Notebook Output

| Artifact | Format | Description |
|----------|--------|-------------|
| `xai/notebooks/Phase3_Attention.ipynb` | Jupyter Notebook | Executed notebook with inline visualizations |

---

## 5. Architecture Attachment Point

### 5.1 PRIMARY: PhoBERT Self-Attention

The primary attachment point is the internal self-attention mechanism of PhoBERT, accessed through the HuggingFace `transformers` API.

**Access path through the model hierarchy:**

```
CrossAttentionFusion
  --> self.text_model (TextModel)
      --> self.encoder (AutoModel, i.e., PhoBERT RobertaModel)
          --> forward(input_ids, attention_mask, output_attentions=True)
          --> outputs.attentions  # tuple of 12 tensors
```

**Concrete traversal:**

```
model.text_model.encoder  -->  this is the PhoBERT RobertaModel
```

Calling `model.text_model.encoder(input_ids, attention_mask, output_attentions=True)` returns a `BaseModelOutputWithPoolingAndCrossAttentions` object. The `.attentions` attribute is a tuple of 12 tensors, one per transformer layer. Each tensor has shape `[B, 12, L, L]` where:

- `B` = batch size
- `12` = number of attention heads
- `L` = sequence length (up to 256 after padding)

**Important:** The current `TextModel.forward()` does NOT pass `output_attentions=True`. For attention extraction, the implementer must call `model.text_model.encoder` directly with this flag, bypassing the `TextModel.forward()` wrapper. This avoids modifying the training code.

**What the attention matrix means:** Element `[b, h, i, j]` is the attention weight that token `i` assigned to token `j` in head `h` of that layer, for sample `b`. Each row sums to 1.0 (after softmax). Row 0 corresponds to the `<s>` (CLS) token's attention distribution over all tokens.

### 5.2 SECONDARY (NOT USEFUL): Cross-Attention in CrossAttentionFusion

**Location:** `Models/CrossAttentionFusion.py`, lines 57-58.

```python
t_out, _ = self.cross_attn_t2i(query=t, key=i, value=i)
i_out, _ = self.cross_attn_i2t(query=i, key=t, value=t)
```

**Why it is NOT useful for visualization:**

The inputs to `nn.MultiheadAttention` are:
- `t = self.text_proj(text_feat).unsqueeze(1)` with shape `[B, 1, 512]`
- `i = self.image_proj(image_feat).unsqueeze(1)` with shape `[B, 1, 512]`

Both query and key have sequence length 1. The attention computation is:

```
attn_weights = softmax(Q @ K^T / sqrt(d_k))
```

where `Q` has shape `[B, 8, 1, 64]` and `K` has shape `[B, 8, 1, 64]`, so `Q @ K^T` has shape `[B, 8, 1, 1]`. After softmax over the last dimension (length 1), the result is always `[B, 8, 1, 1]` with value `1.0`.

**Mathematical proof:** For any scalar `x`, `softmax([x]) = [1.0]`. Since the key sequence length is 1, the query has exactly one key to attend to, so the attention weight is trivially 1.0 regardless of the learned projection weights.

**Decision:** Document this finding explicitly. Do not attempt to visualize cross-attention weights. Explain in the thesis that the current architecture uses pooled single-vector features for cross-modal fusion, which makes cross-attention weights uninformative. This is not a defect -- it is a design choice that trades token-level cross-modal interaction for computational simplicity. For token-level cross-modal attention, the architecture would need to pass full token sequences `[B, L, 512]` as keys and values, which is a different design outside the scope of this thesis.

### 5.3 Architecture Diagram (Text XAI Attachment)

```
Input Review: "Do an ngon nhung gia hoi cao"
        |
        v
[PhoBERT Tokenizer]
        |
        v
input_ids [1, L], attention_mask [1, L]
        |
        v
+--------------------------------------------------+
| PhoBERT RobertaModel (12 layers)                 |
|                                                  |
|  Layer 0:  Self-Attention --> attentions[0]  <----+--- ATTACHMENT POINT
|  Layer 1:  Self-Attention --> attentions[1]  <----+--- (extract all 12 layers)
|  ...                                             |
|  Layer 11: Self-Attention --> attentions[11] <----+--- PRIMARY (last layer)
|                                                  |
|  Output: last_hidden_state [1, L, 768]           |
|          pooler_output [1, 768]                   |
+--------------------------------------------------+
        |
        v
text_feat = pooler_output or CLS token [1, 768]
        |
        v
TextModel.fc: Linear(768, 256) + ReLU + Dropout
        |
        v
features [1, 256]
        |
        v  (to CrossAttentionFusion)
text_proj: Linear(768, 512)   <--- NOTE: uses raw text_feat, not fc output
        |
        v
t = [1, 1, 512]  --> cross_attn_t2i --> trivial [1, 8, 1, 1] weights
```

---

## 6. Detailed Implementation Plan

### 6.1 File: `xai/attention_explainer.py`

This is the core module. It contains all attention extraction, aggregation, merging, and visualization logic.

#### 6.1.1 Function: `extract_phobert_attention`

**Signature:**
```
extract_phobert_attention(model, input_ids, attention_mask, tokenizer) -> dict
```

**Responsibility:** Run a forward pass through PhoBERT with `output_attentions=True` and return all attention tensors plus decoded tokens.

**Step-by-step logic:**

1. Set `model.eval()`.
2. Access the PhoBERT encoder: `encoder = model.text_model.encoder` (for CrossAttentionFusion) or `encoder = model.encoder` (for standalone TextModel).
3. Run `with torch.no_grad(): outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True)`.
4. Extract `attentions = outputs.attentions` -- a tuple of 12 tensors, each `[B, 12, L, L]`.
5. Stack into a single tensor: `attn_tensor = torch.stack(attentions, dim=0)` giving `[12, B, 12, L, L]`.
6. For single-sample analysis, squeeze batch: `attn_tensor[:, 0, :, :, :]` giving `[12, 12, L, L]`.
7. Decode tokens: `tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())`.
8. Determine actual token count (exclude padding): find the last non-pad token using `attention_mask[0]`.
9. Trim attention tensor and token list to actual length (remove padding columns and rows).
10. Return dict: `{"attentions": attn_numpy, "tokens": tokens, "seq_len": actual_len, "input_ids": input_ids_list}`.

**Critical details:**
- Must handle the case where the model is a `CrossAttentionFusion` (access via `model.text_model.encoder`) versus a standalone `TextModel` (access via `model.encoder`).
- Must use `torch.no_grad()` to avoid unnecessary gradient computation.
- Must trim padding tokens -- PhoBERT pads to `max_length=256`, but actual reviews may be 10-50 tokens. The attention matrix for padded positions is meaningless.
- Return NumPy arrays (not tensors) for downstream compatibility with matplotlib and NPY saving.

#### 6.1.2 Function: `aggregate_attention`

**Signature:**
```
aggregate_attention(attentions, strategy='last_layer_mean') -> np.ndarray
```

**Responsibility:** Reduce `[12, 12, L, L]` attention tensor to a single `[L, L]` matrix using the specified aggregation strategy.

**Strategies:**

**Strategy A: `last_layer_mean`** (Primary recommended)
1. Select layer 11 (zero-indexed): `layer_attn = attentions[11]` giving `[12, L, L]`.
2. Average over heads: `result = layer_attn.mean(axis=0)` giving `[L, L]`.
3. Rationale: The last layer is closest to the task-relevant representation. Averaging over heads smooths out head-specific specializations.

**Strategy B: `last_4_layers_mean`** (Secondary recommended)
1. Select layers 8, 9, 10, 11: `layers_attn = attentions[8:12]` giving `[4, 12, L, L]`.
2. Average over both layers and heads: `result = layers_attn.mean(axis=(0, 1))` giving `[L, L]`.
3. Rationale: Captures a broader range of high-level representations. More stable than single-layer aggregation.

**Strategy C: `attention_rollout`** (Advanced extension)
1. Start with identity matrix: `rollout = np.eye(L)`.
2. For each layer `l` from 0 to 11:
   a. Compute mean-head attention: `layer_attn = attentions[l].mean(axis=0)` giving `[L, L]`.
   b. Add residual connection: `layer_attn = 0.5 * layer_attn + 0.5 * np.eye(L)`.
   c. Re-normalize rows: `layer_attn = layer_attn / layer_attn.sum(axis=-1, keepdims=True)`.
   d. Multiply: `rollout = rollout @ layer_attn`.
3. Result: `rollout` is `[L, L]`, representing approximate global information flow.
4. Rationale: Accounts for residual connections and layer composition. More theoretically grounded than single-layer extraction, but involves assumptions about residual stream mixing.

**Return:** NumPy array of shape `[L, L]` with values in [0, 1] range (each row sums to approximately 1.0 for mean strategies, exactly 1.0 for rollout).

#### 6.1.3 Function: `cls_token_importance`

**Signature:**
```
cls_token_importance(attention_matrix, tokens) -> dict
```

**Responsibility:** Extract row 0 of the aggregated attention matrix, which represents how much the CLS token (`<s>`) attends to each other token.

**Step-by-step logic:**

1. Extract row 0: `cls_attn = attention_matrix[0, :]` giving `[L]`.
2. Create list of `(token, importance)` pairs.
3. Exclude special tokens `<s>` (index 0), `</s>`, and `<pad>` from the output.
4. Sort by importance descending.
5. Return dict: `{"importances": [(token, score), ...], "raw_vector": cls_attn, "top_k": top_k_list}`.

**Why CLS row:** In PhoBERT (RoBERTa-based), the CLS token (`<s>` at position 0) is used as the pooled representation for downstream tasks. Row 0 of the attention matrix shows which tokens the CLS token gathered information from. This is the most directly relevant view for understanding what information entered the final text representation.

#### 6.1.4 Function: `merge_subword_attention`

**Signature:**
```
merge_subword_attention(token_importance, tokens, strategy='mean') -> list
```

**Responsibility:** Merge PhoBERT BPE subword tokens back into whole words and aggregate their attention scores.

**PhoBERT tokenization behavior:**

PhoBERT (`vinai/phobert-base-v2`) uses a BPE tokenizer derived from fastBPE. Its tokenization characteristics for Vietnamese:

1. Vietnamese is largely syllable-based, so most common words are single tokens.
2. Uncommon words or compound expressions may be split into subword pieces.
3. PhoBERT uses the `@@` continuation marker appended to a subword piece that is NOT the final piece of a word. Example: if "restaurant" were split, it might become `["rest@@", "aurant"]`.
4. Alternatively, some BPE tokenizers use the `Ġ` (capital G with dot, Unicode character for space-prefixed tokens) convention where tokens starting with a space character indicate the beginning of a new word.

**Step-by-step logic:**

1. Initialize empty word list and current word buffer.
2. Iterate through tokens (excluding `<s>`, `</s>`, `<pad>`):
   a. If the token ends with `@@`, it is a continuation subword:
      - Add to current word buffer.
      - Accumulate its importance score.
   b. If the token does NOT end with `@@`:
      - If there is content in the buffer, append this token to the buffer and finalize the word.
      - If the buffer is empty, this token is a standalone word.
3. For each merged word, aggregate subword importances using `strategy`:
   - `'mean'`: average of subword importance scores.
   - `'max'`: maximum of subword importance scores.
   - `'sum'`: sum of subword importance scores (then re-normalize).
   - `'first'`: take the first subword's importance (linguistically motivated: the first syllable often carries the root meaning).
4. Return list of `{"word": merged_text, "importance": aggregated_score, "num_subwords": count}`.

**IMPORTANT:** The implementer MUST inspect actual PhoBERT tokenization output on sample Vietnamese reviews before finalizing the merging logic. Run the tokenizer on 5-10 example reviews and print the raw token list to determine the exact subword boundary convention. The logic described above covers the two most common conventions; the actual behavior must be empirically verified.

**Fallback:** If the tokenizer does not use `@@` markers, check for the space-prefix convention (tokens starting with a space character indicate a new word, tokens without a leading space are continuations). PhoBERT v2 may differ from v1 in this regard.

#### 6.1.5 Function: `extract_cross_attention_weights`

**Signature:**
```
extract_cross_attention_weights(model, input_ids, attention_mask, pixel_values, num_images) -> dict
```

**Responsibility:** Extract cross-attention weights from CrossAttentionFusion and document that they are trivially 1.0.

**Step-by-step logic:**

1. Run the full forward pass but with `nn.MultiheadAttention` set to return attention weights.
2. Modify the forward call: `t_out, t2i_attn = self.cross_attn_t2i(query=t, key=i, value=i)` (the second return value contains weights when `need_weights=True`, which is the default).
3. Record `t2i_attn` shape: should be `[B, 1, 1]` (or `[B, 8, 1, 1]` if `average_attn_weights=False`).
4. Verify all values are 1.0.
5. Return dict with the weights and a textual explanation.

**Note:** This function exists for documentation purposes only. Its output is not used for visualization. It proves that cross-attention visualization is uninformative in this architecture.

#### 6.1.6 Function: `plot_attention_heatmap`

**Signature:**
```
plot_attention_heatmap(attention_matrix, tokens, title, save_path, figsize=None, cmap='viridis', vmin=0, vmax=None) -> None
```

**Responsibility:** Render a token-by-token attention heatmap using seaborn.

**Step-by-step logic:**

1. Create matplotlib figure with appropriate size. If `figsize` is None, compute from token count: `(max(8, len(tokens)*0.5), max(6, len(tokens)*0.4))`.
2. Use `seaborn.heatmap()` with:
   - `data=attention_matrix`
   - `xticklabels=tokens` (x-axis = key tokens, "attended to")
   - `yticklabels=tokens` (y-axis = query tokens, "attending from")
   - `cmap=cmap`
   - `annot=True` if token count <= 20, else `annot=False`
   - `fmt='.2f'` for annotations
   - `vmin=vmin`, `vmax=vmax`
   - `square=True`
3. Rotate x-axis labels 45 degrees for readability.
4. Set title with aggregation strategy name.
5. Add colorbar label: "Attention Weight".
6. Use `tight_layout()`.
7. Save to `save_path` at 150 DPI.
8. Close figure to free memory.

**Vietnamese text rendering:** Ensure matplotlib is configured with a font that supports Vietnamese characters (diacritics: a, e, o with marks). If the default font fails, fall back to 'DejaVu Sans' or set `matplotlib.rcParams['font.family']` to a Unicode-capable font. The implementer should test this on the target system and add a font configuration block at the top of the module if needed.

#### 6.1.7 Function: `plot_cls_importance_bar`

**Signature:**
```
plot_cls_importance_bar(tokens, importances, title, save_path, top_k=20, figsize=(10, 8)) -> None
```

**Responsibility:** Render a horizontal bar chart showing CLS-to-token importance, sorted descending.

**Step-by-step logic:**

1. Pair tokens with importances, excluding special tokens.
2. Sort by importance descending.
3. Take top `top_k` entries.
4. Create horizontal bar chart with `matplotlib.pyplot.barh()`.
5. Invert y-axis so highest importance is at top.
6. Add value labels at the end of each bar.
7. Set xlabel: "Attention Weight from CLS".
8. Set title.
9. Use `tight_layout()`.
10. Save to `save_path` at 150 DPI.
11. Close figure.

#### 6.1.8 Function: `save_attention_artifacts`

**Signature:**
```
save_attention_artifacts(sample_id, attentions, tokens, metadata, output_dir) -> dict
```

**Responsibility:** Save all raw and processed artifacts to disk in the prescribed folder structure.

**Step-by-step logic:**

1. Create subdirectories: `raw/`, `metadata/` under `output_dir`.
2. Save raw attention tensor: `np.save(raw/sample_{id}_attention_all_layers.npy, attentions)`.
3. Save raw tokens: `json.dump(tokens, raw/sample_{id}_tokens.json)`.
4. Save metadata: `json.dump(metadata, metadata/sample_{id}_attention_metadata.json)`.
5. Return dict of all saved file paths.

### 6.2 File: `xai/attention_runner.py`

**Responsibility:** Orchestrate the full attention extraction pipeline for multiple samples.

#### 6.2.1 Function: `run_attention_analysis`

**Signature:**
```
run_attention_analysis(config) -> None
```

**Step-by-step logic:**

1. Load configuration from `xai/config.py` (experiment path, checkpoint path, sample indices, device, seed).
2. Set random seed for reproducibility.
3. Load the tokenizer: `AutoTokenizer.from_pretrained('vinai/phobert-base-v2')`.
4. Load the image processor (needed for full fusion model forward pass context).
5. Instantiate the model (CrossAttentionFusion with TextModel and ImageModel).
6. Load checkpoint weights.
7. Move model to device, set `model.eval()`.
8. Load the dataset (test or validation split).
9. Create output directory: `experiments/EXP_XXX/xai/attention/`.
10. For each sample index in the configured list:
    a. Load the sample from the dataset.
    b. Move tensors to device.
    c. Call `extract_phobert_attention()` to get raw attention tensors and tokens.
    d. For each aggregation strategy in `['last_layer_mean', 'last_4_layers_mean']`:
       - Call `aggregate_attention()` to get `[L, L]` matrix.
       - Call `plot_attention_heatmap()` to save heatmap PNG.
    e. Call `cls_token_importance()` to get CLS-to-token importance.
    f. Call `plot_cls_importance_bar()` for raw subword tokens.
    g. Call `merge_subword_attention()` to get word-level importance.
    h. Call `plot_cls_importance_bar()` for merged words.
    i. Save top-K tokens JSON.
    j. Save word importance JSON.
    k. Call `save_attention_artifacts()` for raw tensors and metadata.
11. Optionally, call `extract_cross_attention_weights()` for one sample to document the trivial finding.
12. Save `attention_summary.json` with index of all processed samples.
13. Print completion summary.

### 6.3 Cross-Attention Documented Analysis

Create `experiments/EXP_XXX/xai/attention/cross_attention_analysis.md` with the following content:

1. State the architecture: CrossAttentionFusion uses `nn.MultiheadAttention` with 8 heads, hidden=512.
2. State the inputs: `t = text_proj(text_feat).unsqueeze(1)` with shape `[B, 1, 512]`, `i = image_proj(image_feat).unsqueeze(1)` with shape `[B, 1, 512]`.
3. State the computation: `Q @ K^T` produces `[B, 8, 1, 1]` before softmax.
4. State the result: `softmax` over a single-element dimension always yields `1.0`.
5. Conclude: Cross-attention weights in this architecture carry no discriminative information. They cannot distinguish between samples. The architecture uses cross-attention for feature transformation (through the value projection), not for selective attention.
6. State the implication: For text explanation, PhoBERT self-attention is the appropriate tool. For fusion-level explanation, SHAP (Phase 4) quantifies modality contribution.

---

## 7. Required Code Files

| File | Responsibility | New/Modified |
|------|---------------|--------------|
| `xai/attention_explainer.py` | Core attention extraction, aggregation, merging, and visualization functions | **New** |
| `xai/attention_runner.py` | Pipeline orchestration: load model, iterate samples, call explainer, save artifacts | **New** |
| `xai/notebooks/Phase3_Attention.ipynb` | Interactive notebook for exploration, visualization, and thesis figure generation | **New** |
| `xai/__init__.py` | Package init (if not already created in Phase 1) | New or Existing |
| `xai/config.py` | Phase 1 config -- add attention-specific defaults | **Modified** (add section) |
| `xai/utils.py` | Phase 1 utilities -- used as-is | Existing (no change) |

### 7.1 File Responsibility Details

**`xai/attention_explainer.py`** (approximately 350-450 lines):
- `extract_phobert_attention()` -- forward pass with `output_attentions=True`
- `aggregate_attention()` -- three aggregation strategies
- `cls_token_importance()` -- CLS row extraction
- `merge_subword_attention()` -- PhoBERT BPE subword merging
- `extract_cross_attention_weights()` -- trivial cross-attention documentation
- `plot_attention_heatmap()` -- seaborn heatmap rendering
- `plot_cls_importance_bar()` -- matplotlib bar chart rendering
- `save_attention_artifacts()` -- disk I/O for all artifacts
- `inspect_tokenization()` -- utility to print raw tokenization for debugging

**`xai/attention_runner.py`** (approximately 150-200 lines):
- `run_attention_analysis()` -- full pipeline orchestration
- `select_samples()` -- sample selection logic (specific indices, random, stratified)
- `build_attention_config()` -- config builder with defaults

**`xai/notebooks/Phase3_Attention.ipynb`** (approximately 25-30 cells):
- Interactive exploration of attention patterns
- Figure generation for thesis
- Cross-attention triviality demonstration

### 7.2 Additions to `xai/config.py`

Add the following configuration keys for Phase 3:

```
ATTENTION_CONFIG:
  aggregation_strategies: ['last_layer_mean', 'last_4_layers_mean']
  cls_top_k: 15
  subword_merge_strategy: 'mean'
  heatmap_cmap: 'viridis'
  heatmap_dpi: 150
  bar_chart_figsize: [10, 8]
  heatmap_max_tokens_for_annot: 20
  exclude_special_tokens: ['<s>', '</s>', '<pad>']
  font_family: 'DejaVu Sans'
  sample_indices: [0, 1, 2, 3, 4]  # default; override per experiment
```

---

## 8. Folder Structure

### 8.1 Source Code Layout

```
xai/
├── __init__.py
├── config.py                              # Phase 1 (add attention defaults)
├── utils.py                               # Phase 1 (shared utilities)
├── attention_explainer.py                 # Phase 3 (NEW)
├── attention_runner.py                    # Phase 3 (NEW)
└── notebooks/
    └── Phase3_Attention.ipynb             # Phase 3 (NEW)
```

### 8.2 Output Artifact Layout

```
experiments/EXP_XXX/xai/attention/
├── sample_{id}_layer11_mean_heatmap.png
├── sample_{id}_last4layers_mean_heatmap.png
├── sample_{id}_cls_importance_bar.png
├── sample_{id}_cls_importance_word_bar.png
├── sample_{id}_topk_tokens.json
├── sample_{id}_word_importance.json
├── raw/
│   ├── sample_{id}_attention_all_layers.npy    # [12, 12, L, L] float32
│   └── sample_{id}_tokens.json                 # list of subword token strings
├── metadata/
│   ├── sample_{id}_attention_metadata.json     # review text, scores, token count, timestamp
│   └── ...
├── attention_summary.json                      # index of all processed samples
└── cross_attention_analysis.md                 # documented finding about trivial weights
```

### 8.3 Naming Convention

- `sample_{id}` uses the dataset index (integer) as the identifier.
- Strategy names in filenames use snake_case: `layer11_mean`, `last4layers_mean`.
- All PNGs use 150 DPI.
- All JSON files use 2-space indentation.
- All NPY files use float32 dtype.

---

## 9. Notebook Design (Cell by Cell)

### Cell 1: Header and Imports

**Type:** Markdown + Code

**Content:**
- Title: "Phase 3: Attention Visualization for PhoBERT Text Branch"
- Date, experiment ID, model configuration summary
- Import statements: torch, numpy, matplotlib, seaborn, transformers, json, os, sys
- Import from `xai.attention_explainer` and `xai.utils`

### Cell 2: Configuration

**Type:** Code

**Content:**
- Set experiment path: `EXP_DIR = 'experiments/EXP_XXX'`
- Set checkpoint path
- Set device (CUDA / CPU)
- Set random seed (42)
- Set model names: `text_model_name = 'vinai/phobert-base-v2'`, `image_model_name = ...`
- Set sample indices to analyze
- Set aggregation strategies list

### Cell 3: Load Model and Tokenizer

**Type:** Code

**Content:**
- Load PhoBERT tokenizer
- Load image processor (TimmProcessor or AutoImageProcessor)
- Instantiate TextModel + ImageModel + CrossAttentionFusion
- Load checkpoint weights
- Move to device, set eval mode
- Print model summary (parameter count, layer names)

### Cell 4: Tokenization Inspection

**Type:** Markdown + Code

**Content:**
- Markdown: "Before extracting attention, inspect how PhoBERT tokenizes Vietnamese reviews."
- Select 3-5 example review texts from the dataset.
- For each: tokenize, print raw tokens, print token IDs, identify subword boundaries.
- Verify special token positions: `<s>` at index 0, `</s>` at end.
- Document the subword boundary convention observed (e.g., `@@` markers or space-prefix convention).

**Expected output:** Table showing original text, tokens, token IDs, and word boundaries.

### Cell 5: Single Sample Attention Extraction

**Type:** Code

**Content:**
- Load one sample from the dataset.
- Call `extract_phobert_attention()`.
- Print returned tensor shape: should be `[12, 12, L, L]`.
- Print token list.
- Print actual sequence length (non-padded).

**Expected output:** Tensor shape confirmation, token list, sequence length.

### Cell 6: Attention Aggregation -- Last Layer Mean

**Type:** Code

**Content:**
- Call `aggregate_attention(attentions, strategy='last_layer_mean')`.
- Print resulting shape: `[L, L]`.
- Print row sums (should be close to 1.0).
- Display first 5x5 block of the matrix.

### Cell 7: Heatmap Visualization -- Last Layer Mean

**Type:** Code + Output

**Content:**
- Call `plot_attention_heatmap()` inline (display in notebook).
- Use `%matplotlib inline`.
- Show the full token-by-token heatmap.
- Title: "PhoBERT Layer 11 Mean-Head Attention: [review text truncated]"

**Expected output:** Heatmap image displayed inline. Darker cells between semantically related tokens.

### Cell 8: Attention Aggregation -- Last 4 Layers Mean

**Type:** Code + Output

**Content:**
- Call `aggregate_attention(attentions, strategy='last_4_layers_mean')`.
- Call `plot_attention_heatmap()` inline.
- Compare visually with the layer-11-only heatmap.

**Expected output:** Slightly smoother heatmap. Markdown cell below discussing differences.

### Cell 9: CLS Token Importance

**Type:** Code + Output

**Content:**
- Call `cls_token_importance()`.
- Call `plot_cls_importance_bar()` inline.
- Display the top-15 tokens that CLS attends to most.

**Expected output:** Bar chart showing which tokens the CLS representation gathered most information from. Expect aspect terms like `ngon`, `gia`, `cao` to appear in top positions for relevant reviews.

### Cell 10: Subword Merging

**Type:** Markdown + Code

**Content:**
- Markdown: "Merge subword tokens back to Vietnamese words for human readability."
- Call `merge_subword_attention()`.
- Display merged word importance table.
- Call `plot_cls_importance_bar()` with merged words inline.
- Compare subword-level and word-level bar charts side by side.

**Expected output:** Readable Vietnamese words with aggregated importance scores.

### Cell 11: Multiple Samples Comparison

**Type:** Code + Output

**Content:**
- Loop over 3-5 selected samples (choose reviews with diverse aspects: one food-focused, one price-focused, one atmosphere-focused).
- For each sample: generate heatmap and CLS bar chart.
- Display in a grid layout.

**Expected output:** Side-by-side comparison showing that different reviews produce different attention patterns.

### Cell 12: Attention Rollout (Advanced)

**Type:** Markdown + Code

**Content:**
- Markdown: "Attention rollout accounts for residual connections across layers."
- Call `aggregate_attention(attentions, strategy='attention_rollout')`.
- Call `plot_attention_heatmap()` inline.
- Compare with last-layer mean.

**Expected output:** Rollout heatmap. Typically shows more diffuse but globally coherent patterns.

### Cell 13: Special Token Analysis

**Type:** Code + Output

**Content:**
- For the CLS importance vector, compute:
  - Total attention to special tokens (`<s>`, `</s>`).
  - Total attention to content tokens.
  - Ratio of special vs content attention.
- Print summary.
- Flag if special tokens receive more than 30% of total attention (attention sink warning).

**Expected output:** Quantitative breakdown. Discussion of whether special tokens dominate.

### Cell 14: Cross-Attention Triviality Demonstration

**Type:** Markdown + Code

**Content:**
- Markdown: "Demonstrating that cross-attention weights in CrossAttentionFusion are architecturally trivial."
- Run full forward pass with `need_weights=True` on `cross_attn_t2i` and `cross_attn_i2t`.
- Print attention weight shapes and values.
- Assert all values are 1.0 (within numerical tolerance).
- Provide mathematical explanation inline.

**Expected output:** Printed weights showing all 1.0. Assertion passes.

### Cell 15: Head-wise Attention Comparison

**Type:** Code + Output

**Content:**
- For one sample, visualize attention from individual heads in the last layer (layer 11).
- Create a 3x4 grid of heatmaps, one per head.
- Title each: "Head 0", "Head 1", ..., "Head 11".

**Expected output:** 12 small heatmaps showing that different heads capture different patterns (some syntactic, some semantic, some attending to special tokens).

### Cell 16: Layer-wise Attention Evolution

**Type:** Code + Output

**Content:**
- For one sample, visualize CLS importance vector across all 12 layers.
- Create a 3x4 grid of bar charts, one per layer.
- Show how the CLS token's attention focus evolves from early layers (syntactic) to late layers (task-relevant).

**Expected output:** Early layers show relatively uniform attention; late layers show more focused attention on content words.

### Cell 17: Save All Artifacts

**Type:** Code

**Content:**
- Call `save_attention_artifacts()` for all analyzed samples.
- Save heatmap PNGs to disk.
- Save JSON metadata.
- Save raw NPY tensors.
- Print summary of saved files.

### Cell 18: Thesis Figure Generation

**Type:** Markdown + Code

**Content:**
- Markdown: "Generate publication-quality figures for thesis."
- Regenerate the best heatmaps with thesis-quality formatting:
  - Higher DPI (300).
  - Larger font sizes.
  - Consistent color scheme across all Phase 3 figures.
  - Proper axis labels in English.
- Save to a `thesis_figures/` subdirectory.

### Cell 19: Summary and Key Findings

**Type:** Markdown

**Content:**
- Summarize key observations from the attention analysis.
- Note which tokens received highest attention in which review contexts.
- Note whether attention patterns are consistent with human expectations.
- State the cross-attention triviality finding.
- State the attention-is-not-explanation caveat.
- List all generated artifact paths.

---

## 10. Algorithm

### 10.1 Main Pipeline Algorithm

```
ALGORITHM: Attention Visualization Pipeline
INPUT: checkpoint_path, dataset_path, sample_indices, device, seed
OUTPUT: attention artifacts in experiments/EXP_XXX/xai/attention/

1. SET random seed
2. LOAD tokenizer <- AutoTokenizer('vinai/phobert-base-v2')
3. LOAD image_processor <- appropriate processor for Swin-B
4. INSTANTIATE model <- CrossAttentionFusion(TextModel, ImageModel)
5. LOAD model weights from checkpoint_path
6. SET model to eval mode
7. LOAD dataset <- MultimodalDataset(dataset_path, tokenizer, image_processor)
8. CREATE output_dir <- experiments/EXP_XXX/xai/attention/
9. CREATE subdirectories: raw/, metadata/

10. FOR EACH sample_idx IN sample_indices:
    10.1  sample <- dataset[sample_idx]
    10.2  input_ids <- sample['input_ids'].unsqueeze(0).to(device)
    10.3  attention_mask <- sample['attention_mask'].unsqueeze(0).to(device)
    10.4  tokens <- tokenizer.convert_ids_to_tokens(input_ids[0])
    10.5  seq_len <- attention_mask[0].sum().item()
    10.6  tokens <- tokens[:seq_len]  # trim padding

    # --- Extract attention ---
    10.7  WITH torch.no_grad():
            encoder <- model.text_model.encoder
            outputs <- encoder(input_ids, attention_mask, output_attentions=True)
            attentions <- stack(outputs.attentions)[:, 0, :, :seq_len, :seq_len]
            # shape: [12, 12, seq_len, seq_len]

    # --- Aggregate and visualize ---
    10.8  FOR EACH strategy IN ['last_layer_mean', 'last_4_layers_mean']:
            attn_matrix <- aggregate_attention(attentions, strategy)  # [seq_len, seq_len]
            plot_attention_heatmap(attn_matrix, tokens, ..., save_path)

    # --- CLS importance ---
    10.9  cls_importance <- attn_matrix[0, :]  # using last_layer_mean
    10.10 plot_cls_importance_bar(tokens, cls_importance, ..., save_path)

    # --- Subword merging ---
    10.11 word_importance <- merge_subword_attention(cls_importance, tokens, 'mean')
    10.12 plot_cls_importance_bar(words, word_scores, ..., save_path)

    # --- Top-K extraction ---
    10.13 top_k <- sort(word_importance, descending)[:K]
    10.14 SAVE top_k to JSON

    # --- Raw artifacts ---
    10.15 SAVE attentions to NPY
    10.16 SAVE tokens to JSON
    10.17 SAVE metadata to JSON (review_text, true_scores, pred_scores, seq_len, timestamp)

11. DOCUMENT cross-attention triviality finding
12. SAVE attention_summary.json
13. PRINT completion report
```

### 10.2 Attention Rollout Algorithm

```
ALGORITHM: Attention Rollout
INPUT: attentions [12, 12, L, L]
OUTPUT: rollout_matrix [L, L]

1. rollout <- identity_matrix(L)
2. FOR layer_idx FROM 0 TO 11:
    2.1  layer_attn <- mean(attentions[layer_idx], axis=0)  # [L, L] mean over heads
    2.2  layer_attn <- 0.5 * layer_attn + 0.5 * identity_matrix(L)  # residual
    2.3  layer_attn <- layer_attn / row_sum(layer_attn)  # re-normalize
    2.4  rollout <- rollout @ layer_attn  # matrix multiplication
3. RETURN rollout
```

### 10.3 Subword Merging Algorithm

```
ALGORITHM: Subword-to-Word Merging
INPUT: tokens [L], importances [L], merge_strategy
OUTPUT: words [(word_string, aggregated_importance)]

1. REMOVE special tokens (<s>, </s>, <pad>) and their importances
2. word_groups <- []
3. current_group <- []
4. FOR EACH (token, importance) IN zip(tokens, importances):
    4.1  IF token ends with '@@':
           current_group.append((token.rstrip('@@'), importance))
    4.2  ELSE:
           IF current_group is not empty:
             current_group.append((token, importance))
             merged_word <- concatenate all token strings in current_group
             merged_importance <- aggregate(importances in current_group, merge_strategy)
             word_groups.append((merged_word, merged_importance))
             current_group <- []
           ELSE:
             word_groups.append((token, importance))
5. IF current_group is not empty:  # handle trailing subwords
     flush remaining as a word
6. RETURN word_groups
```

---

## 11. Validation

### 11.1 Tensor Shape Validation

| Check | Expected | How to verify |
|-------|----------|---------------|
| Raw attention from PhoBERT | `[12, 12, L, L]` after batch squeeze | Assert shape after `torch.stack(outputs.attentions)[:, 0]` |
| Each row sums to ~1.0 | Sum of each row in attention matrix | `assert np.allclose(attn_matrix.sum(axis=-1), 1.0, atol=1e-5)` |
| Aggregated attention | `[L, L]` | Assert shape after `aggregate_attention()` |
| CLS importance | `[L]` | Assert shape and sum approximately 1.0 |

### 11.2 Tokenization Validation

| Check | Expected | How to verify |
|-------|----------|---------------|
| CLS token at position 0 | `tokens[0] == '<s>'` | Direct string comparison |
| SEP token at end | `tokens[seq_len-1] == '</s>'` | Direct string comparison |
| No pad tokens in trimmed sequence | All tokens are non-`<pad>` | Check `'<pad>' not in tokens[:seq_len]` |
| Subword merging produces valid words | Merged words match original Vietnamese text | Compare concatenated merged words with original `comment_clean` |

### 11.3 Qualitative Validation

| Check | Description |
|-------|-------------|
| Attention on food terms | For reviews discussing food quality, tokens like `ngon`, `do an`, `mon` should receive non-trivial attention |
| Attention on price terms | For reviews discussing price, tokens like `gia`, `cao`, `re`, `dat` should receive non-trivial attention |
| Attention on atmosphere terms | For reviews discussing atmosphere, tokens like `khong gian`, `dep`, `thoai mai` should receive non-trivial attention |
| Attention on service terms | For reviews discussing service, tokens like `nhan vien`, `phuc vu`, `nhiet tinh` should receive non-trivial attention |
| Contrast word attention | Discourse markers like `nhung` (but), `tuy nhien` (however) should show attention connecting the preceding and following clauses |

### 11.4 Cross-Attention Triviality Validation

| Check | Expected | How to verify |
|-------|----------|---------------|
| Cross-attention weight shape | `[B, 1, 1]` or `[B, 8, 1, 1]` | Assert shape |
| Cross-attention weight value | All 1.0 | `assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)` |
| Same result for different samples | Weights identical across samples | Compare weights from 3+ different samples |

### 11.5 Reproducibility Validation

| Check | Description |
|-------|-------------|
| Fixed seed | Run the pipeline twice with seed=42, verify identical outputs |
| Eval mode | Verify `model.training == False` before extraction |
| Deterministic output | Compare NPY files byte-by-byte between runs |
| Artifact completeness | For each sample, verify all expected files exist |

### 11.6 Special Token Attention Sink Check

| Check | Threshold | Action |
|-------|-----------|--------|
| CLS self-attention (attn[0,0]) | > 0.5 | Warning: CLS attending mostly to itself |
| SEP total attention from CLS | > 0.3 | Warning: possible attention sink on SEP |
| Content token attention sum | < 0.3 | Warning: special tokens dominating, content signal weak |

### 11.7 Sanity Check: Untrained vs Trained Model

Run attention extraction on an untrained (randomly initialized) PhoBERT model using the same input. Compare the attention patterns qualitatively. A trained model should show more structured, content-aware attention patterns. An untrained model should show near-uniform or random attention. If there is no visible difference, the attention patterns may not be capturing learned behavior, which would be a significant finding to report.

---

## 12. Risks

### R1: Cross-Attention Weights Are Trivial

**Problem:** The CrossAttentionFusion module uses `nn.MultiheadAttention` where both query and key have sequence length 1 (pooled CLS features projected to `[B, 1, 512]`). The resulting attention weight matrix has shape `[B, 8, 1, 1]`, and softmax over a single element always produces 1.0.

**Why it happens:** The architecture was designed for computational efficiency: instead of passing full token sequences `[B, L, 512]` through cross-attention (which would be O(L^2) per layer), the design pools each modality to a single vector first, then uses cross-attention as a learned projection/transformation mechanism. The attention weight computation becomes degenerate because there is only one key-value pair to attend to.

**Possible implementation strategies:**

- **Strategy A: Visualize the trivial weights anyway.** Generate heatmaps showing 1.0 everywhere. Advantages: complete coverage. Disadvantages: misleading, wastes space, suggests cross-attention is doing something it is not.
- **Strategy B: Document the finding and skip visualization.** Write a clear analysis explaining why the weights are trivial. Focus all text explanation effort on PhoBERT self-attention. Advantages: honest, scientifically correct, saves implementation time. Disadvantages: examiner might ask why cross-attention was not visualized.
- **Strategy C: Modify architecture to use token-level cross-attention.** Change `unsqueeze(1)` to pass full token sequences through cross-attention. Advantages: enables meaningful cross-attention visualization. Disadvantages: violates the constraint of not redesigning the architecture, changes the trained model, requires retraining.

**Engineering trade-offs:** Strategy B requires 1 hour of documentation. Strategy C requires days of implementation, retraining, and re-evaluation.

**Research trade-offs:** Strategy B is the only honest approach. Cross-attention visualization of trivial weights would be misleading in a thesis. An examiner who asks "why not visualize cross-attention?" will be more impressed by a clear mathematical explanation of why it is uninformative than by a meaningless 1x1 heatmap.

**FINAL DECISION: Strategy B.** Document the mathematical proof that single-element softmax is always 1.0. Include this as a "negative finding" in the thesis. Explain that the cross-attention module contributes through its learned value projection (the V transformation), not through selective attention. For cross-modal contribution analysis, direct the reader to SHAP (Phase 4).

**Defense-ready answer:** "Cross-attention weights in our architecture are architecturally trivial because both modalities are pooled to single vectors before cross-attention. The softmax over a single key always produces 1.0. The cross-attention module still contributes through its learned value projection, but the attention weights themselves carry no discriminative information. We document this finding and rely on SHAP for cross-modal contribution analysis."

---

### R2: PhoBERT Subword Tokenization for Vietnamese

**Problem:** PhoBERT uses BPE tokenization. While Vietnamese is largely syllable-based (most common words are single tokens), uncommon words, compound expressions, or words with unusual diacritics may be split into subword pieces. Displaying raw subword tokens in attention visualizations produces unreadable results.

**Why it happens:** BPE tokenization learns a vocabulary of frequent character sequences from a training corpus. Less frequent sequences are decomposed into smaller pieces. Vietnamese has a large number of syllable variants due to tonal diacritics, and compound words (e.g., `khong gian` = "atmosphere/space") may or may not be treated as single tokens depending on the BPE vocabulary.

**Possible implementation strategies:**

- **Strategy A: Display raw subword tokens only.** No merging. Advantages: faithful to what the model actually processes, no information loss. Disadvantages: subword fragments are unreadable for humans; thesis figures become confusing.
- **Strategy B: Merge subwords to words, display only merged view.** Advantages: human-readable, clean figures. Disadvantages: merging may hide interesting subword-level patterns; the merging logic may have edge cases that produce incorrect words.
- **Strategy C: Generate both views. Use merged for main thesis figures, raw for appendix or supplementary materials.** Advantages: complete information at both levels; human-readable primary figures with raw data available for verification. Disadvantages: doubles the number of figures; slightly more implementation effort.

**Engineering trade-offs:** Strategy C requires generating two versions of each bar chart but uses the same underlying data. The additional effort is approximately 20 lines of code.

**Research trade-offs:** Strategy C is the most complete. An examiner who questions the merging can be directed to the raw view. The merged view makes the thesis accessible.

**Recommended implementation:**

1. Implement `merge_subword_attention()` with configurable strategy (mean, max, sum, first).
2. Default to `mean` for the primary merged view.
3. Generate both `_cls_importance_bar.png` (raw subwords) and `_cls_importance_word_bar.png` (merged words) for each sample.
4. In the thesis, use the merged word view for all main figures. Include the raw subword view in the appendix or supplementary materials.

**FINAL DECISION: Strategy C.** Generate both raw subword and merged word visualizations. Use merged words for thesis main body. Use raw subwords for appendix and debugging.

**Important implementation note:** The implementer MUST empirically verify the subword boundary convention by printing tokenization output for 5-10 sample Vietnamese reviews before writing the merging logic. The convention may be `@@` continuation markers (fastBPE style) or space-prefix markers (SentencePiece style). The `merge_subword_attention()` function must handle the actual convention used by `vinai/phobert-base-v2`.

---

### R3: Attention Aggregation Strategy Selection

**Problem:** PhoBERT has 12 layers with 12 heads each, producing 144 attention matrices per sample. There is no single "correct" way to aggregate these into a human-interpretable summary. Different aggregation strategies can produce different (sometimes contradictory) visualizations.

**Why it happens:** Different layers and heads learn different aspects of language. Early layers tend to capture syntactic patterns (e.g., adjacent word dependencies). Late layers tend to capture semantic patterns (e.g., aspect-sentiment relationships). Different heads within the same layer may specialize in different linguistic phenomena. Aggregating across them necessarily loses information.

**Possible implementation strategies:**

| Strategy | Description | Advantages | Disadvantages |
|----------|-------------|------------|---------------|
| Single head, single layer | Choose one specific (layer, head) pair | Most faithful to one specific learned pattern | Arbitrary choice, noisy, not generalizable |
| Last layer mean | Mean of 12 heads in layer 11 | Stable, captures task-relevant late-layer patterns | Ignores potentially useful mid-layer information |
| Last 4 layers mean | Mean across layers 8-11, all 48 heads | Smoother, captures broader high-level semantics | May dilute sharp layer-11 patterns |
| Max over heads | Maximum attention value across heads per layer | Highlights strongest signals | Amplifies noise, loses distributional information |
| Attention rollout | Matrix multiplication across all layers with residual mixing | Most theoretically grounded for global information flow | Computationally more complex, relies on assumptions about residual stream |

**Engineering trade-offs:** All strategies are computationally trivial (matrix operations on `[12, L, L]` tensors). The implementation effort is identical. The difference is purely in interpretation.

**Research trade-offs:** Using multiple strategies provides a more complete picture. Reporting only one strategy is simpler but may miss important patterns or create selection bias.

**FINAL DECISION:**

- **Primary strategy:** `last_layer_mean` -- mean of 12 heads in layer 11. This is the most commonly used aggregation in the attention visualization literature and captures the most task-relevant representations. Use this for all main thesis figures.
- **Secondary strategy:** `last_4_layers_mean` -- mean across layers 8-11. Generate this for comparison. Include in supplementary materials.
- **Extension:** `attention_rollout` -- implement but present as an advanced analysis. Include in the notebook but only in the thesis if it reveals meaningfully different patterns.
- **Diagnostic:** Individual head visualizations (12 heatmaps for layer 11) in the notebook for exploration. Not for thesis main body.

---

### R4: Attention is NOT Explanation

**Problem:** There is a well-documented risk of overclaiming that high attention weight means causal importance. The research literature has established that attention weights are neither necessary nor sufficient for explaining model predictions (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019). Presenting attention as causal explanation would be scientifically incorrect and vulnerable to examiner criticism.

**Why it happens:** Attention heatmaps are visually compelling and appear to "explain" the model. The visual resemblance to human importance judgments creates a strong but potentially misleading intuitive interpretation.

**Possible implementation strategies:**

- **Strategy A: Present attention as explanation.** Advantages: simpler narrative. Disadvantages: scientifically wrong, examiner will attack.
- **Strategy B: Present attention as "information flow evidence" with explicit caveats.** State clearly that attention shows which tokens interacted strongly during encoding, not that these interactions caused the final prediction. Validate attention findings with perturbation-based methods (LIME Text, Phase 5). Advantages: scientifically correct, defensible. Disadvantages: requires more nuanced writing.
- **Strategy C: Do not use attention at all, rely only on perturbation methods.** Advantages: avoids the controversy entirely. Disadvantages: loses a valuable source of evidence about internal model behavior; misses an opportunity to compare complementary methods.

**FINAL DECISION: Strategy B.** Present attention as information flow evidence with explicit caveats at every point where attention is discussed. Specifically:

1. Every attention figure caption must include language like "shows attention weights indicating information flow, not causal importance."
2. The thesis methodology section must cite the attention-is-not-explanation literature.
3. Phase 5 (LIME Text) results must be cross-referenced: "Tokens identified as important by LIME (perturbation-based) overlap with tokens receiving high attention, providing converging evidence."
4. Any claim based on attention must be qualified with "the attention analysis suggests..." rather than "the model relies on..."

**Defense-ready answer:** "Attention weights show which tokens interacted strongly during encoding, which is evidence about information flow but not a causal explanation. We validate attention findings with LIME Text, which provides perturbation-based local importance. When both methods agree, we have stronger evidence. When they disagree, we treat that as diagnostic information about the difference between internal interaction strength and prediction sensitivity."

---

### R5: Special Token Attention Sink

**Problem:** Transformer models are known to exhibit "attention sink" behavior where special tokens (`<s>`, `</s>`) absorb disproportionate attention weight, especially in early layers. This happens because special tokens are present in every input and can serve as "no-op" attention targets when a head has no meaningful content to attend to.

**Why it happens:** During training, if a head learns that attending to a content token would introduce noise for certain inputs, it learns to "dump" attention on the always-present special token instead. This is a well-documented phenomenon in transformer interpretability research.

**Possible implementation strategies:**

- **Strategy A: Include special tokens in all visualizations.** Advantages: faithful to the raw attention data. Disadvantages: special tokens may dominate the heatmap, making content token patterns invisible.
- **Strategy B: Always exclude special tokens.** Advantages: cleaner visualizations focusing on content. Disadvantages: hides potentially informative behavior; attention weights no longer sum to 1.0 after removal.
- **Strategy C: Generate both views. Primary view excludes special tokens (for thesis figures). Diagnostic view includes them (for appendix/debugging). Add a quantitative check that logs the fraction of attention going to special tokens.** Advantages: complete information, clean main figures, quantitative awareness of sink behavior. Disadvantages: slightly more implementation effort.

**Engineering trade-offs:** Strategy C requires one additional filter step and one additional figure per sample. Minimal overhead.

**Research trade-offs:** Strategy C is the most rigorous. An examiner who asks "did you check for attention sinks?" gets a quantitative answer.

**FINAL DECISION: Strategy C.**

1. For CLS importance bar charts (the main thesis figures): exclude `<s>` and `</s>` from the bar chart. Re-normalize the remaining weights to sum to 1.0.
2. For token-to-token heatmaps: include special tokens but use a distinct visual marker (e.g., lighter color or annotation) to flag them.
3. Compute and log the "attention sink ratio" = `(attn_to_special_tokens) / (total_attn)` for each sample. If this ratio exceeds 0.3, flag it in the metadata.
4. Include a paragraph in the thesis discussion section noting the attention sink phenomenon and reporting the typical ratio observed in this dataset.

---

### R6: Target-Agnostic Nature of Self-Attention

**Problem:** PhoBERT self-attention is computed once for the entire input, independent of which of the 5 target scores (`food_score`, `price_score`, `atmosphere_score`, `service_score`, `overall_satisfaction`) is being predicted. The attention matrices are identical regardless of which target head is examined. This means attention cannot provide target-specific text evidence.

**Why it happens:** The architecture computes text encoding before any target-specific processing. The PhoBERT encoder produces a single text representation that is shared across all five prediction targets. Target-specific processing only happens in the downstream MLP heads (in CrossAttentionFusion, the `self.head` module).

**Possible implementation strategies:**

- **Strategy A: Ignore this limitation. Present attention as if it explains individual targets.** Advantages: simpler narrative. Disadvantages: scientifically incorrect.
- **Strategy B: Acknowledge the limitation explicitly. Present attention as text-branch-level evidence. Direct readers to LIME Text (Phase 5) or gradient-based token attribution for target-specific text importance.** Advantages: honest, provides clear guidance for where to find target-specific evidence. Disadvantages: limits the claims that can be made from attention alone.
- **Strategy C: Implement gradient-weighted attention (gradient x attention) to create target-specific attention maps.** Compute gradients of each target score with respect to the attention weights and multiply. Advantages: creates target-specific attention views. Disadvantages: adds implementation complexity, the resulting maps are a hybrid method that is harder to interpret and justify.

**Engineering trade-offs:** Strategy B requires no additional implementation beyond a paragraph in the thesis. Strategy C requires gradient computation infrastructure similar to Grad-CAM, adding approximately 50-100 lines of code.

**Research trade-offs:** Strategy B is the cleanest separation of methods. Strategy C creates a "frankenmethod" that is hard to position in the literature. The thesis already has dedicated methods for target-specific importance: Grad-CAM (Phase 2) for images, LIME (Phase 5) for text, SHAP (Phase 4) for fusion.

**FINAL DECISION: Strategy B.** Acknowledge the limitation. Present attention as "text encoder internal processing evidence" that is shared across all targets. For target-specific text importance, direct readers to:
- LIME Text (Phase 5): "Which words, when removed, change the `price_score` prediction?"
- SHAP on fusion (Phase 4): "Which text embedding dimensions contribute to `food_score`?"

**Defense-ready answer:** "Self-attention in PhoBERT is target-agnostic -- the same attention weights are used regardless of which score is predicted. This is because the text encoder processes the review once, before any target-specific head. For target-specific text evidence, we use LIME Text, which perturbs individual words and measures the effect on each specific target score."

---

### R7: Long Sequence Heatmap Readability

**Problem:** Vietnamese reviews can be up to 256 tokens long (the max_length setting). A 256x256 heatmap with token labels on both axes would be unreadable in a thesis figure.

**Why it happens:** The system uses `max_length=256` for tokenization. While most reviews are shorter, some may contain 50-100+ tokens after BPE tokenization. A heatmap with 100 tokens on each axis becomes a dense, illegible grid.

**Possible implementation strategies:**

- **Strategy A: Always show full heatmap.** Advantages: complete information. Disadvantages: unreadable for long sequences.
- **Strategy B: Truncate to first N tokens for visualization.** Advantages: readable. Disadvantages: may miss important tokens in the latter part of the review.
- **Strategy C: Use two-tier visualization. For short sequences (<=30 tokens), show full annotated heatmap. For longer sequences, show CLS importance bar chart (which is always readable regardless of length) as the primary figure, and save the full heatmap to raw artifacts for reference.** Advantages: readable primary figures, no information loss. Disadvantages: inconsistent figure format between short and long reviews.
- **Strategy D: For long sequences, show only the top-K most important tokens in the heatmap (submatrix).** Extract the K tokens with highest CLS attention, then show the K x K submatrix of their mutual attention. Advantages: focuses on the most relevant interactions. Disadvantages: loses context of how important tokens relate to less important ones.

**FINAL DECISION: Strategy C with elements of D.**

1. For sequences with <=30 tokens: generate full annotated heatmap with cell values displayed.
2. For sequences with 31-60 tokens: generate full heatmap without cell value annotations (color only).
3. For sequences with >60 tokens: generate CLS importance bar chart as the primary figure. Optionally generate a top-20 submatrix heatmap showing mutual attention among the 20 most important tokens.
4. Always save the full raw attention tensor to NPY regardless of sequence length.
5. For thesis figures, select representative samples with moderate length (20-40 tokens) to ensure readability.

---

### R8: Memory Constraints for Raw Attention Storage

**Problem:** Storing raw attention tensors `[12, 12, L, L]` in float32 for many samples can consume significant disk space. For L=256 (max), one sample's attention tensor is `12 * 12 * 256 * 256 * 4 bytes = 37.7 MB`. For 100 samples, this is approximately 3.8 GB.

**Why it happens:** The attention tensor has O(L^2) entries per head per layer, and we store all 144 (12 layers x 12 heads) matrices.

**Possible implementation strategies:**

- **Strategy A: Store all raw tensors.** Advantages: complete data for future analysis. Disadvantages: potentially large disk usage for many samples.
- **Strategy B: Store only the aggregated matrices, not raw tensors.** Advantages: much smaller (a few KB per sample). Disadvantages: cannot recompute different aggregation strategies later.
- **Strategy C: Store raw tensors only for a selected subset (e.g., 10-20 samples for case studies). Store only aggregated matrices for the rest.** Advantages: balanced approach. Disadvantages: slightly more complex logic.
- **Strategy D: Store raw tensors in float16 instead of float32.** Advantages: halves storage. Disadvantages: minor precision loss.

**FINAL DECISION: Strategy C + D.**

1. For the primary sample set (5-20 case study samples): store full raw attention tensors in float16 (`.npy` files). This is approximately 18.8 MB per sample, totaling at most 376 MB for 20 samples.
2. For extended analysis (if running on many samples): store only aggregated `[L, L]` matrices and CLS importance vectors. This is approximately 260 KB per sample for L=256.
3. Trim padding from stored tensors: only store `[12, 12, actual_len, actual_len]`, which is typically much smaller than `[12, 12, 256, 256]` since most reviews are 20-60 tokens.

---

## 13. Best Practices

### 13.1 Deterministic Execution

1. Set `torch.manual_seed(seed)`, `numpy.random.seed(seed)`, `random.seed(seed)` before any inference.
2. Use `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` if using GPU.
3. Always run with `model.eval()` to disable dropout.
4. Use `torch.no_grad()` for all attention extraction (no gradient computation needed for attention weight extraction).

### 13.2 Memory Optimization

1. Process one sample at a time (batch_size=1) for attention extraction. Attention tensors for long sequences are memory-intensive.
2. Move attention tensors to CPU immediately after extraction: `attn = attn.cpu().numpy()`.
3. Close matplotlib figures explicitly after saving (`plt.close()`) to prevent memory accumulation.
4. Delete intermediate tensors after use with `del` and call `torch.cuda.empty_cache()` periodically if on GPU.
5. Trim padding before storing: a 30-token review produces `[12, 12, 30, 30]` instead of `[12, 12, 256, 256]`, saving 98.6% storage.

### 13.3 Figure Quality

1. Use consistent colormap (`viridis`) across all attention heatmaps within this phase.
2. Use consistent figure sizing: heatmaps scale with token count; bar charts use fixed `(10, 8)`.
3. Use 150 DPI for working figures, 300 DPI for thesis-quality figures.
4. Include title, axis labels, and colorbar label on every figure.
5. Use `tight_layout()` to prevent label clipping.
6. Test Vietnamese diacritic rendering: verify that characters like `a`, `e`, `o` display correctly. Configure matplotlib font if needed.

### 13.4 Artifact Naming

1. All filenames use `sample_{id}` prefix where `{id}` is the dataset integer index.
2. Aggregation strategy names use snake_case: `layer11_mean`, `last4layers_mean`, `attention_rollout`.
3. Artifact types use descriptive suffixes: `_heatmap.png`, `_bar.png`, `_tokens.json`, `_metadata.json`.
4. Timestamp format in metadata: ISO 8601 (`YYYY-MM-DDTHH:MM:SS`).

### 13.5 Logging

1. Print a one-line summary for each processed sample: `"[Phase 3] Sample {id}: {seq_len} tokens, saved to {output_dir}"`.
2. Log any warnings: special token attention sink, subword merging edge cases, unexpectedly short/long sequences.
3. At the end, print total processing time and number of samples processed.

### 13.6 Error Handling

1. If a sample has zero valid tokens (empty review), skip it and log a warning.
2. If subword merging produces an empty word list, fall back to raw subword display and log a warning.
3. If Vietnamese font rendering fails, proceed with default font and log the issue.
4. If NPY save fails due to disk space, catch the exception, log it, and continue with the next sample.

### 13.7 Checkpoint Handling

1. Load the checkpoint using `torch.load(path, map_location=device)` to handle CPU/GPU mismatch.
2. Use `model.load_state_dict(ckpt['model_state_dict'])` with strict=True (default) to verify all weights are loaded.
3. Verify the checkpoint matches the expected architecture by checking key counts and names.

### 13.8 Configuration Management

1. All configurable parameters (sample indices, aggregation strategies, top-K, colormap, DPI) are centralized in `xai/config.py`.
2. The notebook can override configuration values in Cell 2 for interactive exploration.
3. The runner script reads configuration from `xai/config.py` for batch processing.

---

## 14. Deliverables

### 14.1 Code Deliverables

| Deliverable | Format | Description |
|-------------|--------|-------------|
| `xai/attention_explainer.py` | Python module | Core attention extraction, aggregation, merging, and visualization functions |
| `xai/attention_runner.py` | Python module | Pipeline orchestration for batch processing |
| `xai/notebooks/Phase3_Attention.ipynb` | Jupyter notebook | Interactive exploration and thesis figure generation |
| Updated `xai/config.py` | Python module | Attention-specific configuration defaults |

### 14.2 Per-Sample Artifact Deliverables

For each analyzed sample (minimum 5 samples, recommended 10-20):

| Deliverable | Format |
|-------------|--------|
| Last-layer-mean attention heatmap | PNG (150 DPI) |
| Last-4-layers-mean attention heatmap | PNG (150 DPI) |
| CLS importance bar chart (subword level) | PNG (150 DPI) |
| CLS importance bar chart (word level) | PNG (150 DPI) |
| Top-K token importance list | JSON |
| Word-level importance list | JSON |
| Raw attention tensors (all 12 layers) | NPY (float16) |
| Raw token list | JSON |
| Sample metadata | JSON |

### 14.3 Summary Deliverables

| Deliverable | Format | Description |
|-------------|--------|-------------|
| `attention_summary.json` | JSON | Index of all processed samples with key statistics |
| `cross_attention_analysis.md` | Markdown | Mathematical proof and discussion of trivial cross-attention weights |

### 14.4 Thesis Figure Deliverables

| Figure | Description | Target thesis section |
|--------|-------------|----------------------|
| Token-to-token heatmap (best sample) | Annotated heatmap showing attention weights for a representative Vietnamese review | Results: Text Branch Explanation |
| CLS importance bar chart (word-level) | Horizontal bar chart showing which words the CLS token attended to | Results: Text Branch Explanation |
| Head-wise comparison (12 heads, layer 11) | 3x4 grid showing attention diversity across heads | Discussion: Attention Head Specialization |
| Layer-wise CLS evolution | Progression of CLS attention from layer 0 to layer 11 | Discussion: Layer-wise Attention Analysis |
| Cross-attention triviality | Table or figure demonstrating 1.0 weights | Discussion: Architectural Analysis |

---

## 15. Thesis Usage

### 15.1 Results Chapter

**Text Branch Explanation Results:**
- Present 3-5 attention heatmaps for representative Vietnamese reviews spanning different aspect categories.
- For a food-focused review (e.g., "Do an ngon, mon nao cung tuoi"), show that tokens `ngon`, `mon`, `tuoi` receive high attention from CLS.
- For a price-focused review (e.g., "Gia hoi cao so voi chat luong"), show that tokens `gia`, `cao`, `chat luong` receive high attention.
- For a mixed review (e.g., "Do an ngon nhung gia hoi cao"), show that both food tokens and price tokens are attended to, with the contrast marker `nhung` connecting the two segments.
- Present word-level CLS importance bar charts as the primary text explanation figures.

**Quantitative Summary:**
- Report the average attention sink ratio across all analyzed samples.
- Report the average number of content tokens receiving >5% of CLS attention.
- Compare attention distributions between correctly predicted and incorrectly predicted samples (if error analysis is available from Phase 2).

### 15.2 Discussion Chapter

**Attention Patterns and Aspect Detection:**
- Discuss whether the attention patterns suggest that PhoBERT has learned aspect-specific representations for Vietnamese restaurant reviews.
- Note that attention is target-agnostic (R6) and cannot differentiate between `food_score` and `price_score` at the text encoder level.

**Attention Head Specialization:**
- If individual head analysis reveals specialization (e.g., one head focuses on sentiment words, another on aspect words), discuss this as evidence of learned linguistic structure.

**Cross-Attention Architectural Insight:**
- Present the cross-attention triviality finding as an architectural analysis contribution.
- Discuss the trade-off between the current design (simple, fast, pooled features) and a hypothetical token-level cross-attention design (more expressive, more computationally expensive).

**Limitations:**
- State explicitly: "Attention visualization provides evidence about information flow within the text encoder but does not constitute causal explanation of the model's predictions."
- Reference the attention-is-not-explanation literature (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).
- Direct readers to LIME Text (Phase 5) for perturbation-based causal evidence.

### 15.3 Case Studies (Phase 6)

Phase 3 artifacts will be combined with Phase 2 (Grad-CAM) and Phase 4 (SHAP) artifacts in Phase 6 case studies. For each case study sample, the text attention heatmap and word importance bar chart will appear alongside the Grad-CAM image overlay and SHAP waterfall plot, providing a multi-method explanation.

### 15.4 Defense Presentation

Prepare the following slides using Phase 3 outputs:
1. **Slide: "What did the text branch focus on?"** -- Show a CLS importance bar chart for a clear, interpretable review. Highlight aspect terms.
2. **Slide: "Attention is not explanation"** -- Two-column slide: left shows attention heatmap, right shows LIME text importance. Discuss agreement and disagreement.
3. **Slide: "Cross-attention architectural finding"** -- Show the mathematical proof and the 1.0 weight demonstration. Position as an honest engineering analysis.

### 15.5 Journal Paper

If adapted for publication:
- Include one representative attention heatmap in the paper body.
- Move all other visualizations to supplementary materials.
- Emphasize the multi-method triangulation narrative (attention + LIME + SHAP).
- Include the cross-attention triviality finding as a discussion point about architectural choices in multimodal fusion.

---

## 16. Phase Completion Checklist

### 16.1 Code Completion

- [ ] `xai/attention_explainer.py` created with all 8 functions documented in Section 6.1
- [ ] `xai/attention_runner.py` created with pipeline orchestration
- [ ] `xai/config.py` updated with attention-specific configuration
- [ ] `xai/notebooks/Phase3_Attention.ipynb` created with all 19 cells documented in Section 9

### 16.2 Functional Verification

- [ ] `extract_phobert_attention()` returns `[12, 12, L, L]` attention tensor (verified)
- [ ] `aggregate_attention()` produces valid `[L, L]` matrices for all three strategies (verified)
- [ ] `cls_token_importance()` extracts CLS row and correctly excludes special tokens (verified)
- [ ] `merge_subword_attention()` produces readable Vietnamese words (verified on 5+ samples)
- [ ] `extract_cross_attention_weights()` confirms all weights are 1.0 (verified)
- [ ] `plot_attention_heatmap()` renders readable heatmaps with correct Vietnamese character display (verified)
- [ ] `plot_cls_importance_bar()` renders correct bar charts with sorted importance (verified)

### 16.3 Artifact Generation

- [ ] At least 5 samples fully processed with all artifacts generated
- [ ] Heatmap PNGs exist for `last_layer_mean` and `last_4_layers_mean` strategies (verified)
- [ ] CLS importance bar charts exist at both subword and word level (verified)
- [ ] Top-K token JSON files exist and contain valid data (verified)
- [ ] Word importance JSON files exist with merged Vietnamese words (verified)
- [ ] Raw NPY attention tensors saved (verified shape and dtype)
- [ ] Metadata JSON files contain review text, scores, token count, timestamp (verified)
- [ ] `attention_summary.json` created with index of all processed samples (verified)
- [ ] `cross_attention_analysis.md` created with mathematical proof (verified)

### 16.4 Validation Checks

- [ ] Attention row sums are approximately 1.0 for all aggregation strategies (verified)
- [ ] CLS token is at position 0, SEP token at end of sequence (verified)
- [ ] No padding tokens in trimmed attention matrices (verified)
- [ ] Subword merging produces words that match the original review text (verified)
- [ ] Special token attention sink ratio computed and logged (verified)
- [ ] Cross-attention weights confirmed as trivially 1.0 for 3+ samples (verified)

### 16.5 Reproducibility

- [ ] Pipeline produces identical outputs when run twice with seed=42 (verified)
- [ ] Model is in eval mode during all attention extraction (verified)
- [ ] `torch.no_grad()` used during all attention extraction (verified)

### 16.6 Documentation

- [ ] Cross-attention triviality finding documented with mathematical proof
- [ ] Attention-is-not-explanation caveat documented
- [ ] Target-agnostic nature of self-attention documented
- [ ] Subword merging convention for PhoBERT v2 empirically verified and documented

### 16.7 Thesis Readiness

- [ ] At least 3 thesis-quality figures generated (300 DPI, proper labels)
- [ ] Figure captions drafted with "information flow" framing (not "causal explanation")
- [ ] Sample methodology paragraph drafted for thesis Methods section
- [ ] Defense slide content identified (3 slides minimum)

---

*Document version: 1.0*
*Generated for: SE365 -- Explainable Multi-modal Deep Learning System for Restaurant Review Quality Assessment*
*Phase: 3 of 8 -- Attention Visualization for Text Branch*
*Best model: Swin-B + PhoBERT (vinai/phobert-base-v2) + CrossAttentionFusion + LogCosh*
