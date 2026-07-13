# AI Agent — Proposal: Hệ thống Giải thích Chất lượng Nhà hàng bằng Ngôn ngữ Tự nhiên

**Phiên bản:** 1.0
**Ngày:** 2026-06-28
**Dự án:** An Explainable Multi-modal Deep Learning System for Restaurant Review Quality Assessment using Image and Text Data

---

## Mục lục

1. [Mục đích](#1-mục-đích)
2. [Phạm vi của Agent](#2-phạm-vi-của-agent)
3. [Đầu vào (Inputs)](#3-đầu-vào-inputs)
4. [Đầu ra (Outputs)](#4-đầu-ra-outputs)
5. [Lựa chọn OpenAI Model](#5-lựa-chọn-openai-model)
6. [Kiến trúc Prompt](#6-kiến-trúc-prompt)
7. [Evidence Grounding](#7-evidence-grounding)
8. [Structured Output Schema](#8-structured-output-schema)
9. [Pipeline Architecture](#9-pipeline-architecture)
10. [Evidence Extraction Layer](#10-evidence-extraction-layer)
11. [Prompt Input Compression](#11-prompt-input-compression)
12. [Xử lý Hình ảnh](#12-xử-lý-hình-ảnh)
13. [Thiết kế API / Module](#13-thiết-kế-api--module)
14. [OpenAI API Integration](#14-openai-api-integration)
15. [Batch Mode](#15-batch-mode)
16. [Đánh giá và Validation](#16-đánh-giá-và-validation)
17. [An toàn và Hạn chế](#17-an-toàn-và-hạn-chế)
18. [Deliverables](#18-deliverables)

---

## 1. Mục đích

### Tại sao cần AI Agent

Hệ thống hiện tại có hai lớp output:

1. **Model predictions** — 5 điểm số hồi quy (food, price, atmosphere, service, overall) trên thang 1–10.
2. **XAI artifacts** — Grad-CAM heatmap, PhoBERT attention token importance, Cross-Attention token×patch visualization, SHAP modality contribution, LIME perturbation evidence.

Tuy nhiên, cả hai lớp này đều ở dạng **kỹ thuật thuần túy**: tensor, JSON, hình ảnh heatmap. Một người dùng cuối (chủ nhà hàng, khách hàng, giảng viên chấm luận văn) không thể đọc trực tiếp `shap_modality_contribution.json` và hiểu ngay tại sao mô hình đánh giá thức ăn 8.2/10.

AI Agent tồn tại để **chuyển đổi** (translate) các output kỹ thuật thành **giải thích bằng ngôn ngữ tự nhiên** — tiếng Việt hoặc tiếng Anh — dựa hoàn toàn trên bằng chứng (evidence) đã có, không bịa đặt thêm.

### Vai trò trong hệ thống

```
Image + Text → Model → Predictions [B, 5]
                ↓
            XAI Pipeline (Phase 2-6)
                ↓
            Structured Evidence (JSON, PNG, NPZ)
                ↓
            AI Agent (OpenAI API)
                ↓
            Natural-Language Explanation (Vietnamese/English)
```

AI Agent **không thay thế** model hay XAI. Nó là lớp cuối cùng chuyển đổi output thành dạng người đọc được.

---

## 2. Phạm vi của Agent

### Agent CÓ THỂ

| Khả năng | Mô tả |
|---|---|
| Tóm tắt predictions | Giải thích 5 điểm số bằng ngôn ngữ tự nhiên |
| Giải thích evidence | Mô tả bằng chứng từ Grad-CAM, Attention, Cross-Attention, SHAP, LIME |
| So sánh modality | Phân tích text-origin vs image-origin contribution từ SHAP |
| Nhận diện conflict | Giải thích khi text và image evidence mâu thuẫn |
| Đề xuất cải thiện | Gợi ý cho nhà hàng dựa trên evidence (ở dạng khuyến nghị, không phải chẩn đoán) |
| Tạo báo cáo | User-friendly report (Vietnamese) và technical report (English) |
| Tạo JSON | Frontend/backend-ready structured output |

### Agent KHÔNG THỂ

| Hạn chế | Lý do |
|---|---|
| Khẳng định nhân quả (causality) | XAI chỉ cho thấy tương quan, không phải nhân quả |
| Bịa evidence không có trong artifacts | Hallucination control — chỉ dùng evidence có sẵn |
| Ghi đè model predictions | Agent chỉ giải thích, không thay đổi điểm số |
| Chẩn đoán chất lượng thực tế | Agent chỉ giải thích **model** nghĩ gì, không phải thực tế |
| Xử lý ảnh trực tiếp | Agent đọc mô tả XAI artifacts, không phải pixel thô (trừ vision mode) |

---

## 3. Đầu vào (Inputs)

### 3.1 Required Inputs

| Field | Type | Source | Mô tả |
|---|---|---|---|
| `sample_id` | string | XAI pipeline | `sample_{idx:04d}` |
| `review_text` | string | Dataset CSV `comment_clean` | Nội dung review tiếng Việt |
| `predictions` | dict | `get_prediction()` hoặc `test_predictions.csv` | `{food_score: float, price_score: float, atmosphere_score: float, service_score: float, overall_satisfaction: float}` |

### 3.2 Optional Inputs (từ XAI artifacts)

| Field | Type | Source File | Mô tả |
|---|---|---|---|
| `ground_truth` | dict | `test_predictions.csv` hoặc `load_single_sample().factor_scores` | Ground truth 5 scores |
| `absolute_errors` | dict | Computed | `|pred - gt|` per target |
| `image_urls` | list[str] | Dataset CSV `image_url` | URLs của ảnh review |
| `num_images` | int | `load_single_sample().num_real_images` | Số ảnh thực |
| `gradcam` | dict | `gradcam/{sample_id}/metadata.json` | Target layer, cam shape, artifact paths |
| `attention_tokens` | dict | `attention/{sample_id}/word_importance.json` | `[{word, importance}]` top tokens |
| `attention_topk` | list | `attention/{sample_id}/topk_tokens.json` | Top-20 `[{token, importance, rank}]` |
| `cross_attention_summary` | dict | `cross_attention/{sample_id}/cross_attention_summary.json` | `{mean_attention, max_attention, mean_token_entropy, top_5_tokens, top_5_patches}` |
| `cross_attention_topk` | list | `cross_attention/{sample_id}/token_patch_topk.json` | Top-20 token-patch pairs `[{token, patch_idx, patch_row, patch_col, attention}]` |
| `shap_contribution` | dict | `shap/{sample_id}/shap_modality_contribution.json` | Per-factor `{text_pct, image_pct, text_signed, image_signed}` |
| `lime_text_weights` | dict | `lime/{sample_id}/{sid}_lime_text_{factor}_weights.json` | Per-factor word weights |
| `lime_image_weights` | dict | `lime/{sample_id}/{sid}_lime_image_{factor}_weights.json` | Per-factor superpixel weights |
| `case_study_metadata` | dict | `case_studies/{case_id}/metadata.json` | Full case study context |
| `case_type` | string | Selection pipeline | `correct`, `high_error`, `conflict`, etc. |

### 3.3 Unavailable Inputs (không có trong hệ thống hiện tại)

| Field | Ghi chú |
|---|---|
| Object detection labels | Không có object detector — Grad-CAM chỉ cho region heatmap |
| Sentiment ground truth | Không có annotation sentiment per-sentence |
| User demographic | Không có thông tin về reviewer |
| Historical comparisons | Không có retrieval system (future work) |

---

## 4. Đầu ra (Outputs)

### 4A. Human-readable Report (Vietnamese)

Báo cáo dạng văn xuôi tiếng Việt, dành cho người dùng cuối (chủ nhà hàng, khách hàng).

```
📋 BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG

Review: "Đồ ăn ngon nhưng giá hơi cao, không gian đẹp..."

🍽️ Thức ăn: 8.2/10 — Mô hình đánh giá cao chất lượng thức ăn. Hình ảnh cho thấy
   món ăn được trình bày hấp dẫn (vùng nổi bật trên Grad-CAM tập trung vào đĩa
   thức ăn). Từ "ngon" trong review cũng hỗ trợ đánh giá này.

💰 Giá cả: 6.1/10 — Điểm giá cả trung bình. Từ "giá hơi cao" trong review text
   ảnh hưởng tiêu cực đến điểm này (LIME text xác nhận từ "cao" giảm điểm).

🏠 Không gian: 7.4/10 — ...
```

### 4B. Technical Report (English)

Báo cáo kỹ thuật cho nghiên cứu / luận văn.

```markdown
## XAI Analysis Report — sample_0042

### Prediction Summary
| Target | Predicted | Ground Truth | Error |
|--------|-----------|--------------|-------|
| Food Score | 8.2 | 8.0 | 0.20 |
| Price Score | 6.1 | 6.0 | 0.10 |
...

### Modality Contribution (SHAP)
Text-origin features contributed 42.3% of the prediction signal,
while image-origin features contributed 57.7%. This indicates the model
relied more on visual evidence...

### Cross-Attention Patterns
The token "ngon" (delicious) attended most strongly to patches (3,4) and (4,4),
which correspond to the food presentation area in the image...
```

### 4C. Structured JSON

```json
{
  "sample_id": "sample_0042",
  "language": "vi",
  "summary": "Mô hình đánh giá nhà hàng này ở mức khá tốt...",
  "scores": {
    "food": {
      "score": 8.2,
      "level": "high",
      "explanation": "Hình ảnh cho thấy món ăn trình bày hấp dẫn...",
      "evidence": {
        "gradcam": "Focus on food dish area (center of image)",
        "attention": ["ngon (0.089)", "đồ ăn (0.045)"],
        "shap": {"text_origin_pct": 38.2, "image_origin_pct": 61.8},
        "lime_text": [{"word": "ngon", "weight": 0.042}],
        "cross_attention": "Token 'ngon' → patches (3,4), (4,4)"
      }
    },
    "price": { ... },
    "atmos": { ... },
    "service": { ... },
    "overall": { ... }
  },
  "modality_contribution": {
    "text_origin_pct": 42.3,
    "image_origin_pct": 57.7,
    "interpretation": "Mô hình dựa nhiều hơn vào bằng chứng hình ảnh..."
  },
  "cross_modal_insights": "Token 'ngon' attends to food presentation patches...",
  "method_agreement": "Grad-CAM và LIME image đều highlight vùng thức ăn...",
  "limitations": [
    "SHAP uses text-origin/image-origin grouping, not pure modality separation",
    "Attention does not prove causality"
  ],
  "recommendations": [
    "Cải thiện presentation thức ăn để tăng food_score",
    "Xem xét giảm giá hoặc tăng giá trị cảm nhận"
  ],
  "confidence": "high",
  "timestamp": "2026-06-28T10:30:00"
}
```

### 4D. Markdown Report

File `.md` cho thesis/case-study export, kết hợp cả Vietnamese summary và English technical detail.

---

## 5. Lựa chọn OpenAI Model

### Chiến lược hai tầng

| Scenario | Model khuyến nghị | Lý do |
|---|---|---|
| **Batch summaries** (15+ samples) | `gpt-4o-mini` | Chi phí thấp, đủ năng lực cho evidence → text |
| **Final report-quality** (case studies) | `gpt-4o` | Chất lượng cao nhất cho thesis/defense |
| **Vision mode** (optional) | `gpt-4o` | Cần vision capability khi gửi ảnh XAI |

### Configuration

```python
AGENT_CONFIG = {
    "batch_model": "gpt-4o-mini",
    "report_model": "gpt-4o",
    "vision_model": "gpt-4o",
    "temperature": 0.3,         # Low temperature cho evidence grounding
    "max_tokens": 2000,         # Đủ cho 1 sample explanation
    "max_tokens_batch": 800,    # Ngắn hơn cho batch mode
}
```

Model name là **configurable**, không hardcode. Khi OpenAI phát hành model mới, chỉ cần thay đổi trong config.

---

## 6. Kiến trúc Prompt

### 6.1 System Prompt

```
You are an AI explanation assistant for a multimodal restaurant review quality 
assessment system. You receive model predictions (5 scores on a 1-10 scale) 
and structured XAI evidence from Grad-CAM, Attention, Cross-Attention, SHAP, 
and LIME analyses.

Your role:
- Translate numerical predictions and XAI evidence into clear natural-language 
  explanations in Vietnamese (or English when specified).
- Every statement must be grounded in the provided evidence.
- Never invent evidence that is not in the input data.
- If evidence for a claim is missing, explicitly say "Không có bằng chứng XAI 
  cho phần này" (Evidence not available for this part).
- Use the term "text-origin" and "image-origin" for SHAP contributions, not 
  "pure text" or "pure image", because cross-attended features contain 
  information from both modalities.
- Do not claim causality — say "mô hình tập trung vào" (the model focused on) 
  rather than "vì" (because).
- Frame recommendations as suggestions based on model evidence, not as 
  definitive quality assessments.

Score interpretation guide:
- 1-3: Low (Thấp)
- 4-5: Below average (Dưới trung bình)  
- 6-7: Average to good (Trung bình - Khá)
- 8-9: Good to excellent (Tốt - Xuất sắc)
- 10: Excellent (Xuất sắc)
```

### 6.2 User Prompt Template

```
## Sample Information
- Sample ID: {sample_id}
- Review text: "{review_text}"
- Number of images: {num_images}

## Model Predictions
{predictions_table}

## Ground Truth (if available)
{ground_truth_table}

## XAI Evidence

### Grad-CAM
{gradcam_evidence}

### PhoBERT Self-Attention
Top tokens: {attention_topk}

### Cross-Attention (Token × Patch)
Top token-patch pairs: {cross_attention_topk}
Summary: {cross_attention_summary}

### SHAP Modality Contribution
{shap_evidence}

### LIME
Text evidence: {lime_text_evidence}
Image evidence: {lime_image_evidence}

## Task
Generate a structured explanation in {language} following this JSON schema:
{output_schema}
```

### 6.3 Style Requirements

- **Vietnamese**: Formal nhưng dễ hiểu. Dùng tiếng Việt phổ thông, tránh từ chuyên môn quá sâu khi viết cho user.
- **English**: Academic style cho technical report. Phù hợp cho thesis/paper.
- **Uncertainty**: Sử dụng "mô hình cho thấy" (the model indicates) thay vì "nhà hàng này có" (this restaurant has).

---

## 7. Evidence Grounding

### Nguyên tắc cốt lõi

Agent phải tuân thủ nguyên tắc **evidence-grounded generation**: mọi khẳng định trong output phải có cơ sở từ input evidence.

### Grounding Rules

| Rule | Ví dụ đúng | Ví dụ sai |
|---|---|---|
| **Chỉ dùng evidence có sẵn** | "Grad-CAM highlight vùng thức ăn (dựa trên artifact gradcam_img0_food.png)" | "Nhà hàng có đồ ăn ngon" (không có nguồn) |
| **Trích dẫn source** | "SHAP cho thấy text-origin đóng góp 42.3%" | "Text quan trọng hơn" (không có số liệu) |
| **Thừa nhận thiếu evidence** | "Không có LIME artifacts cho target này" | Bỏ qua hoàn toàn |
| **Không suy diễn quá mức** | "Attention tập trung vào từ 'giá'" | "Khách hàng phàn nàn về giá" (suy diễn thêm) |

### Validation Pipeline

```python
def validate_grounding(agent_output: dict, evidence_input: dict) -> list[str]:
    """Check every claim in agent output is grounded in evidence."""
    violations = []
    
    # Check: mọi SHAP percentage phải match input
    for factor in FACTOR_NAMES:
        if factor in agent_output['scores']:
            claimed_pct = agent_output['scores'][factor]['evidence'].get('shap', {})
            actual_pct = evidence_input.get('shap_contribution', {}).get(factor, {})
            if claimed_pct and actual_pct:
                if abs(claimed_pct['text_origin_pct'] - actual_pct['text_pct']) > 1.0:
                    violations.append(f"SHAP text_pct mismatch for {factor}")
    
    # Check: mọi token được đề cập phải có trong attention_topk
    # ... additional checks
    
    return violations
```

---

## 8. Structured Output Schema

### JSON Schema đầy đủ

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["sample_id", "language", "summary", "scores", "confidence"],
  "properties": {
    "sample_id": {"type": "string", "pattern": "^sample_\\d{4}$"},
    "language": {"type": "string", "enum": ["vi", "en"]},
    "summary": {
      "type": "string",
      "description": "Tóm tắt 2-3 câu về đánh giá tổng thể"
    },
    "scores": {
      "type": "object",
      "properties": {
        "food": {"$ref": "#/$defs/score_explanation"},
        "price": {"$ref": "#/$defs/score_explanation"},
        "atmos": {"$ref": "#/$defs/score_explanation"},
        "service": {"$ref": "#/$defs/score_explanation"},
        "overall": {"$ref": "#/$defs/score_explanation"}
      }
    },
    "modality_contribution": {
      "type": "object",
      "properties": {
        "text_origin_pct": {"type": "number"},
        "image_origin_pct": {"type": "number"},
        "interpretation": {"type": "string"}
      }
    },
    "cross_modal_insights": {
      "type": "string",
      "description": "Cross-attention token-patch relationship insights"
    },
    "method_agreement": {
      "type": "string",
      "description": "Mô tả mức độ đồng thuận giữa các phương pháp XAI"
    },
    "limitations": {
      "type": "array",
      "items": {"type": "string"}
    },
    "recommendations": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Đề xuất cải thiện dựa trên evidence (optional)"
    },
    "confidence": {
      "type": "string",
      "enum": ["low", "medium", "high"],
      "description": "Confidence based on evidence completeness"
    },
    "timestamp": {"type": "string", "format": "date-time"}
  },
  "$defs": {
    "score_explanation": {
      "type": "object",
      "required": ["score", "level", "explanation"],
      "properties": {
        "score": {"type": "number", "minimum": 1, "maximum": 10},
        "level": {"type": "string", "enum": ["low", "below_average", "average", "good", "excellent"]},
        "explanation": {"type": "string"},
        "evidence": {
          "type": "object",
          "properties": {
            "gradcam": {"type": "string"},
            "attention": {"type": "array", "items": {"type": "string"}},
            "cross_attention": {"type": "string"},
            "shap": {
              "type": "object",
              "properties": {
                "text_origin_pct": {"type": "number"},
                "image_origin_pct": {"type": "number"}
              }
            },
            "lime_text": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "word": {"type": "string"},
                  "weight": {"type": "number"}
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### Confidence Levels

| Level | Điều kiện |
|---|---|
| `high` | Có đủ 5 phương pháp XAI (Grad-CAM + Attention + Cross-Attention + SHAP + LIME) |
| `medium` | Có 3-4 phương pháp |
| `low` | Chỉ có 1-2 phương pháp hoặc evidence mâu thuẫn nghiêm trọng |

---

## 9. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Agent Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INPUT SAMPLE                                            │
│     ├─ sample_id, review_text, image_urls                   │
│     └─ predictions (5 scores)                               │
│                                                             │
│  2. EVIDENCE LOADING                                        │
│     ├─ Load Grad-CAM metadata.json                          │
│     ├─ Load Attention word_importance.json, topk_tokens.json│
│     ├─ Load Cross-Attention cross_attention_summary.json,   │
│     │   token_patch_topk.json                               │
│     ├─ Load SHAP shap_modality_contribution.json            │
│     ├─ Load LIME *_weights.json (per factor)                │
│     └─ Load Case Study metadata.json (if applicable)        │
│                                                             │
│  3. EVIDENCE EXTRACTION                                     │
│     ├─ Extract Top-K tokens from attention                   │
│     ├─ Extract Top-K token-patch pairs from cross-attention  │
│     ├─ Extract SHAP percentages per factor                   │
│     ├─ Extract LIME top positive/negative words              │
│     └─ Compress all into text evidence block                 │
│                                                             │
│  4. PROMPT BUILDING                                         │
│     ├─ Select system prompt                                  │
│     ├─ Fill user prompt template with evidence               │
│     ├─ Set output schema                                     │
│     └─ Select model (batch vs report quality)                │
│                                                             │
│  5. OPENAI API CALL                                         │
│     ├─ Send prompt via openai.chat.completions.create()      │
│     ├─ Request response_format={"type": "json_object"}       │
│     ├─ Handle timeout, retry, rate limit                     │
│     └─ Parse JSON response                                   │
│                                                             │
│  6. VALIDATION                                              │
│     ├─ Validate JSON against schema                          │
│     ├─ Check evidence grounding                              │
│     ├─ Verify SHAP percentages match input                   │
│     └─ Flag any ungrounded claims                            │
│                                                             │
│  7. REPORT GENERATION                                       │
│     ├─ Generate Vietnamese user report                       │
│     ├─ Generate English technical report                     │
│     └─ Save JSON + Markdown outputs                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Evidence Extraction Layer

Mỗi phương pháp XAI cần một hàm extraction chuyên biệt để chuyển artifacts thành text evidence cho LLM.

### 10.1 Grad-CAM Evidence

```python
def extract_gradcam_evidence(sample_id: str, xai_dir: str) -> dict:
    """Load Grad-CAM metadata and summarize."""
    meta_path = f"{xai_dir}/gradcam/{sample_id}/metadata.json"
    # Return: target_layer, which factors have overlays, 
    #         image count, artifact paths
    # Text output: "Grad-CAM attached to encoder.norm (Swin-B).
    #   Overlays available for: food, price, atmos, service, overall.
    #   Image 0 analyzed."
```

### 10.2 Attention Evidence

```python
def extract_attention_evidence(sample_id: str, xai_dir: str) -> dict:
    """Load top tokens and word importance."""
    topk_path = f"{xai_dir}/attention/{sample_id}/topk_tokens.json"
    word_path = f"{xai_dir}/attention/{sample_id}/word_importance.json"
    # Return: top_10_tokens with importance scores,
    #         sink_ratio, seq_len
    # Text output: "Top tokens by CLS attention: ngon (0.089), 
    #   giá (0.045), cao (0.038), ..."
```

### 10.3 Cross-Attention Evidence

```python
def extract_cross_attention_evidence(sample_id: str, xai_dir: str) -> dict:
    """Load token-patch pairs and summary statistics."""
    summary_path = f"{xai_dir}/cross_attention/{sample_id}/cross_attention_summary.json"
    topk_path = f"{xai_dir}/cross_attention/{sample_id}/token_patch_topk.json"
    # Return: top_5_token_patch_pairs, mean_entropy, 
    #         top_5_important_tokens, top_5_important_patches
    # Text output: "Token 'ngon' → Patch(3,4) attn=0.087, 
    #   Patch(4,4) attn=0.065. Token 'giá' → Patch(1,2) attn=0.052"
```

### 10.4 SHAP Evidence

```python
def extract_shap_evidence(sample_id: str, xai_dir: str) -> dict:
    """Load modality contribution percentages."""
    contrib_path = f"{xai_dir}/shap/{sample_id}/shap_modality_contribution.json"
    # Return: per_factor {text_pct, image_pct, text_signed, image_signed}
    # Text output: "Food: text-origin 38.2%, image-origin 61.8%.
    #   Price: text-origin 71.5%, image-origin 28.5%.
    #   Text-origin signed contribution for price is negative (-0.42)."
```

### 10.5 LIME Evidence

```python
def extract_lime_evidence(sample_id: str, xai_dir: str) -> dict:
    """Load top positive/negative words and superpixels."""
    # Per factor, load {sid}_lime_text_{factor}_weights.json
    # Return: per_factor {top_positive_words, top_negative_words}
    # Text output: "food: positive words [ngon (+0.04), thơm (+0.02)],
    #   negative words []. price: negative words [cao (-0.03), đắt (-0.02)]"
```

---

## 11. Prompt Input Compression

### Tại sao cần compression

XAI artifacts thô có thể rất lớn:
- `raw_attention.npz`: `[12, 12, L, L]` float16 — hàng MB
- `raw_shap_values.npz`: `[1024]` × 5 targets
- `cross_attention_raw.npz`: `[T, P]` matrix

LLM không cần (và không thể xử lý) raw tensors. Agent chỉ cần **compressed text evidence**.

### Chiến lược compression

| XAI Method | Raw Data | Compressed Evidence |
|---|---|---|
| Grad-CAM | `raw_cams.npz` [1024, 7, 7] × 5 | "Overlays available for 5 targets. Image 0 analyzed." |
| Attention | `raw_attention.npz` [12, 12, L, L] | Top-10 tokens with importance scores |
| Cross-Attention | `cross_attention_raw.npz` [T, P] | Top-10 token-patch pairs with attention values |
| SHAP | `raw_shap_values.npz` [1024] × 5 | Per-factor `{text_pct, image_pct, text_signed, image_signed}` |
| LIME | Full perturbation results | Top-5 positive and top-5 negative words per factor |

### Token Budget Estimate

| Component | Estimated Tokens |
|---|---|
| System prompt | ~400 |
| Sample info + review text | ~200 |
| Predictions + ground truth | ~100 |
| Grad-CAM evidence | ~50 |
| Attention evidence (top-10) | ~100 |
| Cross-Attention evidence (top-10) | ~150 |
| SHAP evidence (5 factors) | ~200 |
| LIME evidence (5 factors) | ~300 |
| Output schema | ~200 |
| **Total input** | **~1700 tokens** |
| **Expected output** | **~1000-2000 tokens** |

Tổng chi phí ước tính: ~3500 tokens/sample.
- `gpt-4o-mini`: ~$0.001/sample
- `gpt-4o`: ~$0.035/sample

---

## 12. Xử lý Hình ảnh

### Mode A: Text-only Evidence (Recommended Default)

Agent nhận **mô tả text** của XAI artifacts, không nhận ảnh trực tiếp.

**Ưu điểm:**
- Chi phí thấp (không cần vision tokens)
- Nhanh hơn
- Phù hợp với hầu hết use cases
- Đủ thông tin từ compressed evidence

**Nhược điểm:**
- Không thể mô tả visual patterns không có trong metadata
- Không nhìn thấy chất lượng ảnh thực tế

### Mode B: Vision Mode (Optional)

Agent nhận selected visualization images khi sử dụng model có vision capability (`gpt-4o`).

Gửi tối đa 2-3 ảnh nhỏ:
- Original image (resized 224×224)
- Grad-CAM overlay cho target chính
- SHAP modality bar chart

**Ưu điểm:**
- Agent có thể mô tả visual content cụ thể hơn
- Phù hợp cho final thesis-quality reports

**Nhược điểm:**
- Chi phí cao hơn (~10x vision tokens)
- Chậm hơn
- Cần cẩn thận hallucination (LLM có thể mô tả ảnh sai)

### Khuyến nghị

- **Default**: Text-only mode cho batch processing
- **Optional**: Vision mode cho 3-5 case studies quan trọng nhất

---

## 13. Thiết kế API / Module

### Cấu trúc thư mục

```
agent/
├── __init__.py              # Package exports
├── config.py                # Model names, temperature, token limits
├── evidence_loader.py       # Load XAI artifacts from disk
├── evidence_builder.py      # Compress artifacts into text evidence
├── prompt_builder.py        # Build system + user prompts
├── openai_client.py         # OpenAI API wrapper with retry/rate limit
├── output_schema.py         # JSON schema definition
├── report_generator.py      # Generate Vietnamese/English reports
├── validator.py             # Validate grounding + schema compliance
└── notebooks/
    └── AI_Agent_Demo.ipynb  # Interactive demo notebook
```

### Core API

```python
# agent/__init__.py
from agent.config import AgentConfig
from agent.evidence_loader import EvidenceLoader
from agent.evidence_builder import EvidenceBuilder
from agent.prompt_builder import PromptBuilder
from agent.openai_client import OpenAIClient
from agent.report_generator import ReportGenerator
from agent.validator import OutputValidator


class ExplanationAgent:
    """High-level AI Agent for generating natural-language explanations."""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.evidence_loader = EvidenceLoader()
        self.evidence_builder = EvidenceBuilder()
        self.prompt_builder = PromptBuilder(self.config)
        self.client = OpenAIClient(self.config)
        self.validator = OutputValidator()
        self.reporter = ReportGenerator()
    
    def explain_sample(
        self,
        sample_id: str,
        review_text: str,
        predictions: dict,
        xai_dir: str,
        ground_truth: dict = None,
        case_type: str = None,
        language: str = "vi",
        mode: str = "text_only",  # or "vision"
    ) -> dict:
        """Generate explanation for a single sample."""
        # 1. Load evidence
        evidence = self.evidence_loader.load(sample_id, xai_dir)
        
        # 2. Build compressed evidence
        compressed = self.evidence_builder.build(evidence)
        
        # 3. Build prompt
        prompt = self.prompt_builder.build(
            sample_id=sample_id,
            review_text=review_text,
            predictions=predictions,
            ground_truth=ground_truth,
            evidence=compressed,
            case_type=case_type,
            language=language,
        )
        
        # 4. Call OpenAI
        response = self.client.generate(prompt, mode=mode)
        
        # 5. Validate
        violations = self.validator.validate(response, evidence)
        if violations:
            response['validation_warnings'] = violations
        
        return response
    
    def explain_batch(
        self,
        samples: list[dict],
        xai_dir: str,
        language: str = "vi",
    ) -> list[dict]:
        """Batch processing for multiple samples."""
        results = []
        for sample in samples:
            try:
                result = self.explain_sample(
                    sample_id=sample['sample_id'],
                    review_text=sample['review_text'],
                    predictions=sample['predictions'],
                    xai_dir=xai_dir,
                    ground_truth=sample.get('ground_truth'),
                    case_type=sample.get('case_type'),
                    language=language,
                    mode="text_only",  # batch uses cheaper mode
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'sample_id': sample['sample_id'],
                    'status': 'failed',
                    'error': str(e),
                })
        return results
```

### Notebook Demo

```python
# agent/notebooks/AI_Agent_Demo.ipynb

from agent import ExplanationAgent, AgentConfig

config = AgentConfig(
    batch_model="gpt-4o-mini",
    report_model="gpt-4o",
    temperature=0.3,
)

agent = ExplanationAgent(config)

# Single sample
result = agent.explain_sample(
    sample_id="sample_0000",
    review_text="Đồ ăn ngon nhưng giá hơi cao...",
    predictions={"food_score": 8.2, "price_score": 6.1, ...},
    xai_dir="/content/drive/MyDrive/SE365/experiments/EXP_060A/xai",
    language="vi",
)

print(result['summary'])
```

---

## 14. OpenAI API Integration

### 14.1 Environment Variables

```bash
# .env (NEVER commit to git)
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...  # optional
```

```python
# agent/config.py
import os

class AgentConfig:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. "
                "Set it via: export OPENAI_API_KEY=sk-..."
            )
        self.batch_model = "gpt-4o-mini"
        self.report_model = "gpt-4o"
        self.temperature = 0.3
        self.max_tokens = 2000
        self.timeout = 60
        self.max_retries = 3
```

### 14.2 Client Implementation

```python
# agent/openai_client.py
from openai import OpenAI
import time
import json

class OpenAIClient:
    def __init__(self, config):
        self.client = OpenAI(api_key=config.api_key)
        self.config = config
    
    def generate(self, prompt: dict, mode: str = "text_only") -> dict:
        model = self.config.report_model if mode == "vision" else self.config.batch_model
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=prompt['messages'],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                    timeout=self.config.timeout,
                )
                content = response.choices[0].message.content
                return json.loads(content)
            
            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            except openai.APITimeoutError:
                print(f"Timeout on attempt {attempt + 1}")
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                return {"error": "Invalid JSON response", "raw": content}
        
        raise RuntimeError(f"Failed after {self.config.max_retries} attempts")
```

### 14.3 Cost Control

```python
# Logging for cost tracking
import logging
logger = logging.getLogger('agent')

def log_usage(response, model):
    usage = response.usage
    logger.info(
        f"Model: {model} | "
        f"Input: {usage.prompt_tokens} | "
        f"Output: {usage.completion_tokens} | "
        f"Total: {usage.total_tokens}"
    )
```

### 14.4 Security

- **KHÔNG** lưu API key trong code hoặc notebook
- Sử dụng `.env` file (đã có trong `.gitignore`)
- Trên Colab: sử dụng `google.colab.userdata` hoặc Colab secrets
- Log API calls nhưng KHÔNG log API key

---

## 15. Batch Mode

### Input: Case Study Manifest

Đọc từ `case_studies/sample_manifest.json`:

```python
manifest = json.load(open("sample_manifest.json"))
for case in manifest:
    sample_id = case['sample_id']
    # Load review text, predictions, etc.
```

### Output Structure

```
agent_outputs/
├── sample_0000_report.json       # Structured JSON
├── sample_0000_report.md         # Markdown report
├── sample_0000_report_vi.md      # Vietnamese version
├── ...
├── batch_summary.csv             # Summary table
├── batch_summary.json            # Full batch results
└── batch_log.json                # API usage + timing
```

### Batch Summary CSV

| sample_id | case_type | confidence | summary_vi | food | price | atmos | service | overall | api_tokens | elapsed_s |
|---|---|---|---|---|---|---|---|---|---|---|
| sample_0000 | correct | high | Mô hình đánh giá tốt... | 8.2 | 6.1 | 7.4 | 6.8 | 7.5 | 3200 | 2.1 |

---

## 16. Đánh giá và Validation

### 16.1 Automated Validation

| Check | Phương pháp | Pass Criterion |
|---|---|---|
| JSON schema validity | `jsonschema.validate()` | No errors |
| SHAP percentages match | Compare input vs output | Difference < 1% |
| Token mentions grounded | Check tokens exist in `topk_tokens.json` | All mentioned tokens found |
| Score levels consistent | Check `level` matches `score` range | All consistent |
| No fabricated evidence | Check no ungrounded Grad-CAM descriptions | No hallucinated regions |
| Language correct | Check output language matches requested | Match |

### 16.2 Human Review Checklist

Cho mỗi case study quan trọng, reviewer kiểm tra:

- [ ] Summary phản ánh đúng predictions
- [ ] Evidence trích dẫn đúng từ XAI artifacts
- [ ] Không có hallucination (bịa evidence)
- [ ] Vietnamese tự nhiên và dễ hiểu
- [ ] Recommendations hợp lý và có cơ sở
- [ ] Limitations được đề cập
- [ ] Không khẳng định causality

### 16.3 Quality Metrics

| Metric | Target |
|---|---|
| Schema compliance rate | 100% |
| Evidence grounding rate | > 95% |
| Hallucination rate | < 2% |
| Average generation time | < 5s/sample |
| Average cost | < $0.01/sample (batch), < $0.05/sample (report) |

---

## 17. An toàn và Hạn chế

### Hạn chế kỹ thuật

1. **XAI không chứng minh nhân quả.** Grad-CAM cho thấy vùng ảnh mô hình "chú ý", nhưng không chứng minh đó là lý do duy nhất.
2. **Attention không phải explanation hoàn chỉnh.** Attention weights cho thấy information flow, không phải causal importance (Jain & Wallace, 2019).
3. **SHAP dùng "text-origin" / "image-origin" grouping.** Vì cross-attention đã trộn thông tin hai modality, dims 0:512 không phải "pure text" mà là "text features sau khi attend to image".
4. **LIME có tính stochastic.** Kết quả LIME có thể khác nhau giữa các random seeds.
5. **LLM có thể hallucinate.** Dù có grounding rules, LLM vẫn có thể tạo ra claims không có trong evidence. Validation pipeline là bắt buộc.

### Hạn chế ứng dụng

1. **Agent không thay thế human judgment.** Output là gợi ý, không phải chẩn đoán.
2. **Recommendations chỉ dựa trên model evidence.** Mô hình chỉ thấy ảnh và text, không biết thực tế nhà hàng.
3. **Không có historical comparison.** Agent không so sánh với reviews khác (future work: RAG).
4. **OpenAI API dependency.** Agent cần internet và API key. Offline mode không hỗ trợ.

### Safety Measures

- Temperature thấp (0.3) để giảm creativity/hallucination
- Structured output format bắt buộc JSON
- Validation pipeline tự động
- Prompt rules cấm khẳng định nhân quả
- Human review cho thesis-quality outputs

---

## 18. Deliverables

### Files cần tạo

| File | Mô tả |
|---|---|
| `AI_agent_proposal.md` | Tài liệu này |
| `agent/__init__.py` | Package init + `ExplanationAgent` class |
| `agent/config.py` | `AgentConfig` — model names, API key, parameters |
| `agent/evidence_loader.py` | Load XAI artifacts từ disk |
| `agent/evidence_builder.py` | Compress artifacts thành text evidence |
| `agent/prompt_builder.py` | System/user prompt construction |
| `agent/openai_client.py` | OpenAI API wrapper |
| `agent/output_schema.py` | JSON schema definition |
| `agent/report_generator.py` | Generate Markdown/Vietnamese reports |
| `agent/validator.py` | Grounding + schema validation |
| `agent/notebooks/AI_Agent_Demo.ipynb` | Interactive demo (Colab-ready) |

### Output files (generated at runtime)

| File | Mô tả |
|---|---|
| `agent_outputs/{sample_id}_report.json` | Per-sample structured output |
| `agent_outputs/{sample_id}_report.md` | Per-sample Markdown report |
| `agent_outputs/batch_summary.csv` | Batch processing summary |
| `agent_outputs/batch_summary.json` | Full batch results |
| `agent_outputs/batch_log.json` | API usage + cost tracking |

### Dependencies

```
openai>=1.0
jsonschema>=4.0
python-dotenv>=1.0
```

Thêm vào `requirements.txt` hoặc tạo `agent/requirements.txt` riêng.

---

## Phụ lục A — Ví dụ End-to-End

### Input

```json
{
  "sample_id": "sample_0005",
  "review_text": "Đồ ăn ngon, nhân viên phục vụ nhiệt tình. Giá hơi cao nhưng chất lượng xứng đáng. Không gian quán đẹp, thoáng mát.",
  "predictions": {
    "food_score": 8.2,
    "price_score": 6.1,
    "atmosphere_score": 7.8,
    "service_score": 8.5,
    "overall_satisfaction": 7.9
  },
  "ground_truth": {
    "food_score": 8.0,
    "price_score": 6.0,
    "atmosphere_score": 8.0,
    "service_score": 9.0,
    "overall_satisfaction": 8.0
  }
}
```

### Compressed Evidence (text block sent to LLM)

```
### Grad-CAM
Overlays available for all 5 targets on Image 0.
Target layer: encoder.norm (Swin-B).

### Attention (Top-10 tokens)
1. ngon (0.089)
2. phục_vụ (0.052)
3. nhiệt_tình (0.048)
4. không_gian (0.041)
5. đẹp (0.038)
6. giá (0.035)
7. cao (0.032)
8. chất_lượng (0.029)
9. thoáng (0.025)
10. xứng_đáng (0.022)

### Cross-Attention (Top-5 token-patch pairs)
1. ngon → Patch(3,4) attn=0.087
2. ngon → Patch(4,4) attn=0.065
3. đẹp → Patch(1,1) attn=0.058
4. không_gian → Patch(0,3) attn=0.052
5. phục_vụ → Patch(3,3) attn=0.048

### SHAP Modality Contribution
food: text-origin 38.2%, image-origin 61.8%
price: text-origin 71.5%, image-origin 28.5%
atmos: text-origin 32.1%, image-origin 67.9%
service: text-origin 82.3%, image-origin 17.7%
overall: text-origin 48.7%, image-origin 51.3%

### LIME Text (food target)
Positive: ngon (+0.042), chất_lượng (+0.018)
Negative: (none)
### LIME Text (price target)
Positive: xứng_đáng (+0.015)
Negative: cao (-0.031), giá (-0.022)
```

### Expected Output (Vietnamese)

```json
{
  "sample_id": "sample_0005",
  "language": "vi",
  "summary": "Mô hình đánh giá nhà hàng này ở mức khá tốt (overall 7.9/10). Điểm mạnh nổi bật là chất lượng phục vụ (8.5) và thức ăn (8.2). Điểm giá cả thấp hơn (6.1) phản ánh việc review đề cập 'giá hơi cao'.",
  "scores": {
    "food": {
      "score": 8.2,
      "level": "good",
      "explanation": "Mô hình đánh giá thức ăn ở mức tốt. Bằng chứng hình ảnh đóng góp chính (image-origin 61.8% theo SHAP). Từ 'ngon' trong review text được attention mechanism đánh giá quan trọng nhất (importance 0.089). Cross-attention cho thấy từ 'ngon' tập trung vào vùng patch (3,4) và (4,4) trên hình ảnh.",
      "evidence": {
        "gradcam": "Overlay available for food target on Image 0",
        "attention": ["ngon (0.089)", "chất_lượng (0.029)"],
        "cross_attention": "Token 'ngon' → Patch(3,4) attn=0.087",
        "shap": {"text_origin_pct": 38.2, "image_origin_pct": 61.8},
        "lime_text": [{"word": "ngon", "weight": 0.042}]
      }
    }
  },
  "modality_contribution": {
    "text_origin_pct": 48.7,
    "image_origin_pct": 51.3,
    "interpretation": "Tổng thể, mô hình sử dụng cân bằng giữa bằng chứng text-origin và image-origin cho đánh giá overall satisfaction."
  },
  "cross_modal_insights": "Cross-attention cho thấy từ 'ngon' tập trung vào vùng thức ăn trên ảnh (patches 3,4 và 4,4), trong khi từ 'không_gian' và 'đẹp' tập trung vào vùng nội thất (patches 0,3 và 1,1). Điều này cho thấy mô hình kết nối đúng từ khóa với vùng hình ảnh tương ứng.",
  "method_agreement": "Grad-CAM, Attention, và LIME đều cho thấy vùng thức ăn và từ 'ngon' là bằng chứng chính cho food_score cao. SHAP xác nhận image-origin đóng góp nhiều hơn (61.8%) cho food, phù hợp với Grad-CAM.",
  "limitations": [
    "SHAP grouping dùng text-origin/image-origin, không phải pure text/image vì cross-attention đã trộn thông tin",
    "Attention weights không chứng minh nhân quả trực tiếp"
  ],
  "recommendations": [
    "Duy trì chất lượng trình bày thức ăn — mô hình đánh giá cao bằng chứng hình ảnh",
    "Xem xét chiến lược giá — review text ảnh hưởng mạnh đến price_score (text-origin 71.5%)"
  ],
  "confidence": "high",
  "timestamp": "2026-06-28T10:30:00"
}
```

---

## Phụ lục B — Tương thích với Hệ thống Hiện tại

### Artifact Paths (codebase thực tế)

| Phase | Directory Pattern | Key Files |
|---|---|---|
| Grad-CAM | `{xai_dir}/gradcam/{sample_id}/` | `metadata.json`, `gradcam_img0_{factor}.png`, `raw_cams.npz` |
| Attention | `{xai_dir}/attention/{sample_id}/` | `word_importance.json`, `topk_tokens.json`, `tokens.json`, `metadata.json` |
| Cross-Attention | `{xai_dir}/cross_attention/{sample_id}/` | `cross_attention_summary.json`, `token_patch_topk.json`, `cross_attention_raw.npz` |
| SHAP | `{xai_dir}/shap/{sample_id}/` | `shap_modality_contribution.json`, `raw_shap_values.npz`, `metadata.json` |
| LIME | `{xai_dir}/lime/{sample_id}/` | `{sid}_lime_text_{factor}_weights.json`, `{sid}_lime_image_{factor}_weights.json`, `metadata.json` |
| Case Study | `{xai_dir}/case_studies/{case_id}/` | `metadata.json`, `analysis.md`, `combined_figure_target{i}_{factor}.png` |

### Constants (từ `xai/config.py`)

| Constant | Value |
|---|---|
| `FACTOR_NAMES` | `['food', 'price', 'atmos', 'service', 'overall']` |
| `DISPLAY_NAMES` | `['Food Score', 'Price Score', 'Atmosphere Score', 'Service Score', 'Overall Satisfaction']` |
| `SCORE_RANGE` | `(1, 10)` |
| `FUSED_DIM` | `1024` (512 text-origin + 512 image-origin) |
| `CROSS_ATTN_HIDDEN_DIM` | `512` |

### `load_single_sample()` Return Fields

```python
{
    'input_ids': Tensor[1, seq_len],
    'attention_mask': Tensor[1, seq_len],
    'pixel_values': Tensor[1, max_images, C, H, W],
    'num_images': Tensor[1],
    'factor_scores': Tensor[5],        # ground truth
    'text': str,                        # review text
    'image_urls': list[str],
    'loaded_images': list[PIL.Image],
    'num_real_images': int,
    'sample_idx': int,
}
```

### `test_predictions.csv` Columns

```
index, split,
y_true_food, y_true_price, y_true_atmos, y_true_service, y_true_overall,
y_pred_food, y_pred_price, y_pred_atmos, y_pred_service, y_pred_overall,
absolute_error_food, absolute_error_price, absolute_error_atmos,
absolute_error_service, absolute_error_overall
```

---

*Kết thúc AI Agent Proposal — Version 1.0*
