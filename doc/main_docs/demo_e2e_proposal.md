# Proposal: Demo_End_to_End_XAI_AI_Agent.ipynb
## Phiên bản 2.0 — Production-Ready Implementation Proposal

**Tên file notebook đích:** `Demo_End_to_End_XAI_AI_Agent.ipynb`  
**Phiên bản proposal:** 2.0  
**Ngày cập nhật:** 2026-06-30  
**Dự án:** An Explainable Multi-modal Deep Learning System for Restaurant Review Quality Assessment using Image and Text Data  
**Mục tiêu:** Trình bày toàn bộ hệ thống cho hội đồng phản biện luận văn đại học — polished, professional, and impressive.

---

## 1. Mục Đích (Purpose)

Notebook này là **demo trình chiếu** dành riêng cho giảng viên hội đồng phản biện. Nó **không** dùng để nghiên cứu, benchmark hay train lại.

**Thông điệp cốt lõi cần truyền đạt:**

> "Hệ thống không chỉ dự đoán điểm số — nó giải thích BẢO SAO đưa ra dự đoán đó, qua 5 tầng phân tích độc lập, và tổng hợp tất cả bằng chứng thành báo cáo ngôn ngữ tự nhiên."

**Đối tượng:** Giảng viên hội đồng, có thể không có nền tảng kỹ thuật Deep Learning sâu.

**Kết quả cần đạt:**
- Người xem hiểu rõ pipeline từ input → prediction → explanation.
- Người xem thấy 5 phương pháp XAI hỗ trợ và kiểm chéo nhau.
- Người xem đọc được báo cáo AI Agent mà không cần biết kỹ thuật.

**Những gì KHÔNG làm:**
- Train / fine-tune lại model.
- Chạy benchmark trên toàn tập test.
- So sánh nhiều model.
- Ablation study.

---

## 2. Phạm Vi Demo (Demo Scope)

### 2.1 Workflow tổng thể

```
╔══════════════════════════════════════════════════════════╗
║  INPUT: Vietnamese Review Text + 1–4 Restaurant Images   ║
╚══════════════════════╦═══════════════════════════════════╝
                       ↓
╔══════════════════════╩═══════════════════════════════════╗
║  CrossAttentionFusion Model                              ║
║  PhoBERT (text) + Swin-B (image) + Bidirectional CA      ║
║  → 5 Regression Scores (1–10 scale)                      ║
╚══════════════════════╦═══════════════════════════════════╝
                       ↓
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
┌──────────┐   ┌──────────────┐  ┌──────────────────────┐
│ Grad-CAM │   │ PhoBERT Attn │  │ Cross-Attention (T↔P)│
│ (image)  │   │   (text)     │  │   (multimodal)       │
└──────────┘   └──────────────┘  └──────────────────────┘
       ↓               ↓               ↓
┌──────────┐   ┌──────────────┐
│   SHAP   │   │     LIME     │
│(fused)   │   │ (text+image) │
└──────────┘   └──────────────┘
                       ↓
╔══════════════════════╩═══════════════════════════════════╗
║  AI Agent: EvidenceLoader → EvidenceBuilder →             ║
║  ReasoningGraph → GPT-4o → Validator → ReportGenerator   ║
╚══════════════════════╦═══════════════════════════════════╝
                       ↓
    Customer View (Vietnamese)  +  Technical View (JSON/MD)
```

### 2.2 Số lượng mẫu: đúng 3

| Slot | Case Type | Mục đích | Phase 6 ưu tiên |
|---|---|---|---|
| **Sample A** | Accurate prediction | Mô hình hoạt động tốt, XAI nhất quán | `correct`, `agreement` |
| **Sample B** | Prediction error / conflict | Giới hạn mô hình, XAI mâu thuẫn | `high_error`, `conflict` |
| **Sample C** | Multimodal evidence-rich | Cross-Attention + SHAP nổi bật | `text_dominant`, `image_dominant`, `difficult` |

### 2.3 Phạm vi XAI per sample

| Module | Demo Scope | Artifacts Hiển Thị |
|---|---|---|
| Prediction | Tất cả 5 targets | Table + Bar Chart |
| Grad-CAM | `overall_satisfaction` only (lý do xem Mục 7) | Original + Raw Heatmap + Overlay (3-panel) |
| PhoBERT Attention | Word-level + CLS matrix | Highlighted text + Top-K bar + CLS heatmap (top tokens) |
| Cross-Attention | Bidirectional T2I + I2T | T2I heatmap + Overlay + I2T heatmap + Top-K pairs + Bipartite graph |
| SHAP | Overall + per-target | Modality pie + Per-target stacked bar + Waterfall (overall) |
| LIME | Text + Image | Combined 4-panel figure |
| AI Agent | Full pipeline | Evidence Dashboard + Reasoning Graph + Report |

---

## 3. Kiến Trúc Notebook (Notebook Architecture)

### 3.1 Môi trường

- **Google Colab** (GPU T4 hoặc A100). Có thể chạy local nếu VRAM ≥ 8GB.
- Python ≥ 3.9. Branch: `xai-v3`.
- Thời gian chạy ước tính: 20–35 phút (GPU), bao gồm LIME.

### 3.2 Kiến trúc model (tham chiếu)

```
Text:   vinai/phobert-base-v2     → tokens [B, T, 768]  → text_proj  Linear(768→512)
Image:  swin_base_patch4_window7_224 → patches [B,49,1024] → image_proj Linear(1024→512)

CrossAttentionFusion:
  cross_attn_t2i: MultiheadAttention(512, 8, batch_first=True)  → [B,T,512]
  cross_attn_i2t: MultiheadAttention(512, 8, batch_first=True)  → [B,49,512]
  masked mean pool → concat → fused [B, 1024]
    dims 0:512   = text-origin  (text queries attended to image)
    dims 512:1024 = image-origin (image queries attended to text)
  head: Linear(1024→512) → ReLU → Dropout → Linear(512→256) → ReLU → Linear(256→5)

Output: [B, 5]  scale 1–10
        [food_score, price_score, atmosphere_score, service_score, overall_satisfaction]
```

### 3.3 Constants (`xai/config.py`)

```python
TARGET_NAMES   = ['food_score', 'price_score', 'atmosphere_score',
                   'service_score', 'overall_satisfaction']
DISPLAY_NAMES  = ['Food Score', 'Price Score', 'Atmosphere Score',
                   'Service Score', 'Overall Satisfaction']
FACTOR_NAMES   = ['food', 'price', 'atmos', 'service', 'overall']
TEXT_FEATURE_DIM      = 768
IMAGE_FEATURE_DIM     = 1024
CROSS_ATTN_HIDDEN_DIM = 512    # projection dim
FUSED_DIM             = 1024   # 512 text-origin + 512 image-origin
DEFAULT_MAX_LENGTH    = 256
DEFAULT_MAX_IMAGES    = 4
DEFAULT_DPI           = 150
THESIS_DPI            = 300
SCORE_RANGE           = (1, 10)
```

### 3.4 Cấu trúc cell toàn notebook

```
━━━ CELL GROUP 0: Setup & Configuration  ━━━━━━━━━━━━━━━━━━━━━━━━━━
  0.1  Title & Introduction (Markdown)
  0.2  Mount Google Drive
  0.3  Clone repository (xai-v3 branch) + pip install
  0.4  Extract data.zip
  0.5  Path Configuration
  0.6  Import all modules
  0.7  Load Model + Tokenizer + ImageProcessor
  0.8  Cross-Method Summary Table (Markdown — học thuật)

━━━ CELL GROUP 1: Sample Selection  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1.1  Load Phase 6 results + dataset
  1.2  Automatic sample selection (3 mẫu)
  1.3  Selection summary display

━━━ CELL GROUP 2: Load Samples  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  2.1  Load all 3 samples into memory

━━━ CELL GROUP 3: SAMPLE A  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3.0  Sample A Header (Markdown)
  3.1  Sample Overview (text + images + ground truth)
  3.2  Prediction
  3.3  Grad-CAM
  3.4  PhoBERT Attention
  3.5  Cross-Attention
  3.6  SHAP
  3.7  LIME
  3.8  AI Agent
  3.9  Sample A XAI Dashboard (tổng hợp)

━━━ CELL GROUP 4: SAMPLE B  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  (cùng cấu trúc 4.0–4.9)

━━━ CELL GROUP 5: SAMPLE C  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  (cùng cấu trúc 5.0–5.9)

━━━ CELL GROUP 6: Cross-Sample Comparison  ━━━━━━━━━━━━━━━━━━━━━━━━
  6.1  Prediction comparison (3 samples × 5 targets)
  6.2  XAI comparison table (method × sample)
  6.3  SHAP modality profile comparison
  6.4  Conclusion Markdown

━━━ CELL GROUP 7: Manifest & Completion  ━━━━━━━━━━━━━━━━━━━━━━━━━━
  7.1  Save manifest.json
  7.2  Final completion message
```

---

## 4. Chiến Lược Chọn Mẫu (Sample Selection Strategy)

### 4.1 Nguyên tắc: Reuse First

Ưu tiên **tái sử dụng** artifacts từ Phase 6. Chỉ regenerate khi cần.

**Thứ tự tìm kiếm:**
1. Phase 6 `case_study_index.csv` hoặc `sample_manifest.csv` (kết quả đã chọn).
2. Phase 6 `test_predictions.csv` / `predictions.csv` (toàn bộ Phase 6).
3. Full dataset CSV (mở rộng tìm kiếm).

### 4.2 Tiêu chí chọn mẫu

**Hard criteria (bắt buộc):**
```python
HARD_CRITERIA = {
    'text_min_words': 30,       # Review text ≥ 30 từ
    'min_real_images': 2,       # Ít nhất 2 ảnh thực
    'has_ground_truth': True,   # Có đủ 5 ground truth scores
    'no_nan_predictions': True, # Prediction không chứa NaN
    'min_completeness': 0.6,    # check_sample_artifacts() ≥ 0.6
}
```

**Soft criteria (xếp hạng, 0–10 điểm):**
```python
def soft_score(sample_id, sample_data, xai_dir):
    score = 0
    arts = check_sample_artifacts(sample_id, xai_dir)
    score += arts['completeness'] * 4.0    # 0–4 điểm
    if sample_data['num_real_images'] >= 3:
        score += 1.5                        # bonus ảnh nhiều
    if arts.get('lime', False):
        score += 1.0                        # LIME tốn kém, quý
    if arts.get('cross_attention', False):
        score += 1.0                        # Cross-attention bonus
    # SHAP balance: cả 2 modality đóng góp
    shap_contrib = load_shap_contribution(sample_id, xai_dir)
    if shap_contrib and 20 < shap_contrib['overall']['text_pct'] < 80:
        score += 1.0
    # AI Agent validated
    if agent_report_validated(sample_id, xai_dir):
        score += 1.5
    return score  # 0–10
```

### 4.3 Progressive relaxation

```
Round 1: hard criteria đầy đủ + completeness ≥ 0.8
Round 2: completeness ≥ 0.6, num_real_images ≥ 1
Round 3: chỉ cần ground truth + prediction
```

### 4.4 Quyết định Reuse vs Regenerate

| Tình huống | Quyết định | Lý do |
|---|---|---|
| Artifact tồn tại, EXP_ID khớp, completeness ≥ 0.8 | **Reuse** | Không lãng phí thời gian |
| Artifact từ EXP_ID khác | **Regenerate** | Không nhất quán |
| Grad-CAM max-min < 0.05 (đồng đều) | **Regenerate** | Không có giá trị visual |
| File thiếu hoặc JSON corrupt | **Regenerate** | Lỗi dữ liệu |
| AI Agent report validation_passed = False | **Regenerate** | Chất lượng không đủ |
| Artifact tồn tại, parse được, visual đẹp | **Reuse** | OK |

```python
def should_regenerate(artifact_type, sample_id, xai_dir, exp_id):
    """
    Returns (bool_regenerate, reason_str).
    """
    artifact_dir = os.path.join(xai_dir, artifact_type, sample_id)
    if not os.path.isdir(artifact_dir):
        return True, 'directory_missing'
    
    meta_path = os.path.join(artifact_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        stored_exp = meta.get('experiment_id', '')
        if stored_exp and stored_exp != exp_id:
            return True, f'exp_id_mismatch ({stored_exp} != {exp_id})'
    
    arts = check_sample_artifacts(sample_id, xai_dir)
    if arts.get('completeness', 0) < 0.6:
        return True, f'low_completeness ({arts["completeness"]:.2f})'
    
    return False, 'ok_reuse'
```

---

## 5. Cấu Trúc Chi Tiết Notebook

### CELL GROUP 0 — Setup & Configuration

#### Cell 0.1: Title & Introduction (Markdown)

```markdown
# Demo End-to-End: Hệ thống Đánh giá Chất lượng Nhà hàng Đa phương thức
## An Explainable Multi-modal Deep Learning System for Restaurant Review Quality Assessment

**Nhóm:** Nhóm 24  
**Workflow:** Text + Images → Prediction → Grad-CAM → Attention → Cross-Attention → SHAP → LIME → AI Agent → Report

Notebook này trình bày toàn bộ pipeline cho **3 mẫu đại diện** được chọn từ tập test.
```

#### Cell 0.2: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Cell 0.3: Clone + Install

```python
!rm -rf /content/SE365
!git clone -b xai-v3 https://github.com/lechihoang/SE365.git /content/SE365
%cd /content/SE365
!pip install -q -r requirements.txt
!pip install -q shap lime scikit-image openai>=1.0 jsonschema>=4.0
```

#### Cell 0.4: Extract data

```python
!rm -rf ./data
!cp /content/drive/MyDrive/SE365/data.zip ./data.zip
!unzip -q data.zip && rm data.zip
```

#### Cell 0.5: Path Configuration

```python
import os, sys, json, time, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
warnings.filterwarnings('ignore')

DRIVE_ROOT   = '/content/drive/MyDrive/SE365'
PROJECT_ROOT = '/content/SE365'
EXP_ID       = 'EXP_060A_bestsequential_full_configuration'

EXP_DIR      = f'{DRIVE_ROOT}/experiments/{EXP_ID}'
XAI_DIR      = f'{EXP_DIR}/xai'
AGENT_OUT_DIR= f'{EXP_DIR}/agent_outputs'
DATA_DIR     = f'{PROJECT_ROOT}/data/text'
IMAGE_DIR    = f'{PROJECT_ROOT}/data/image'
DEMO_OUT     = f'{DRIVE_ROOT}/demo_e2e'

# OpenAI key — từ Colab Secret
try:
    from google.colab import userdata
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
except Exception:
    pass  # Sẽ warning khi đến AI Agent section

os.makedirs(DEMO_OUT, exist_ok=True)
os.makedirs(f'{DEMO_OUT}/errors', exist_ok=True)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f'EXP_ID   : {EXP_ID}')
print(f'DEMO_OUT : {DEMO_OUT}')
```

#### Cell 0.6: Import All Modules

```python
import torch
from xai.config import (
    TARGET_NAMES, DISPLAY_NAMES, FACTOR_NAMES,
    FUSED_DIM, CROSS_ATTN_HIDDEN_DIM, DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_IMAGES, DEFAULT_DPI, THESIS_DPI, SCORE_RANGE,
)
from xai.utils import (
    load_model, load_single_sample, get_prediction,
    get_tokenizer, get_image_processor,
)
from xai.gradcam_explainer import (
    compute_gradcam_for_image, overlay_cam_on_image,
    find_target_layer, GradCAMExplainer,
)
from xai.attention_explainer import (
    extract_phobert_attention, aggregate_attention,
    cls_token_importance, merge_subword_attention,
    extract_cross_attention, AttentionExplainer,
    CrossAttentionExplainer,
)
from xai.shap_explainer import (
    SHAPExplainer, FusionHeadWrapper, extract_fused_embeddings,
    compute_shap_values, modality_contribution, select_background,
)
from xai.lime_explainer import LIMEExplainer, run_lime_image, run_lime_text
from xai.case_study import check_sample_artifacts

from agent.explanation_agent import ExplanationAgent, AgentConfig
from agent.evidence_loader import EvidenceLoader
from agent.evidence_builder import EvidenceBuilder
from agent.prompt_builder import PromptBuilder
from agent.openai_client import OpenAIClient
from agent.validator import OutputValidator
from agent.report_generator import save_sample_report
# Optional: reasoning graph helper
try:
    from agent.reasoning import build_reasoning_graph
except ImportError:
    build_reasoning_graph = None  # sẽ được xử lý trong Section 12.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
# Error accumulator
DEMO_ERRORS = {}
```

#### Cell 0.7: Load Model

```python
from transformers import AutoTokenizer

model, config = load_model(EXP_DIR, device)
# load_model() returns (model, config) tuple
# load_model() đã gọi enable_eager_attention() — patch sdpa→eager cho PhoBERT
# KHÔNG gọi lại lần thứ 2

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2', use_fast=False)

# image_processor = TimmProcessor (KHÔNG phải HuggingFace AutoImageProcessor)
# Nếu TimmProcessor được export từ xai.utils:
try:
    from xai.utils import TimmProcessor as _TimmProcessor
    image_processor = _TimmProcessor('swin_base_patch4_window7_224')
except (ImportError, AttributeError):
    # Fallback: build timm transform manually
    import timm
    from timm.data import resolve_data_config, create_transform as _create_transform
    _m = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
    image_processor = _create_transform(**resolve_data_config({}, model=_m))

target_layer = find_target_layer(model)
# target_layer = model.image_model.encoder.norm (LayerNorm, output BHWC [B,7,7,1024])

assert model.training == False, 'Model must be in eval mode'
print(f'Model      : {model.__class__.__name__}')
print(f'Target layer: {type(target_layer).__name__}')
print(f'Eager attn : {model.text_model.encoder.config._attn_implementation}')  # expect 'eager'
```

#### Cell 0.8: Cross-Method Summary Table (Markdown học thuật)

```markdown
## Tại Sao Cần 5 Phương Pháp XAI?

Mỗi phương pháp XAI trả lời một câu hỏi khác nhau về cùng một dự đoán:

| Phương pháp | Câu hỏi | Input | Output | Ưu điểm | Giới hạn |
|---|---|---|---|---|---|
| **Grad-CAM** | Mô hình nhìn vào đâu trong ảnh? | Image branch gradient | Heatmap [7×7] | Trực quan, spatial | Không phân biệt tốt 5 targets (shared encoder) |
| **PhoBERT Attention** | Token nào được chú ý khi đọc review? | Self-attention weights [12L×12H] | Word importance | Nhanh, giải thích text | Không phải causal evidence |
| **Cross-Attention** | Token nào liên kết với patch nào? | Bidirectional CA [T×P] | Token↔Patch map | Giải thích multimodal | Chỉ thấy trong fusion layer |
| **SHAP** | Text hay Image đóng góp bao nhiêu %? | Fused embedding [1024] | Attribution % | Mathematically grounded | "text-origin" ≠ pure text |
| **LIME** | Điều gì thay đổi làm prediction thay đổi? | Black-box perturbation | Local weights | Model-agnostic | Stochastic, local only |

**Agreement** giữa các phương pháp = **converging evidence** mạnh mẽ.
**Disagreement** = tín hiệu cần điều tra, không phải lỗi.
```

---

## 6. Trình Bày Dự Đoán (Prediction)

### Mục đích

Cho giảng viên thấy ngay mô hình dự đoán gì và sai bao nhiêu — trước khi giải thích.

### Input cần thiết

```python
# Đã có từ load_single_sample()
pred_result = get_prediction(model, samples['A']['data'])
# pred_result keys: predictions[5], ground_truth[5], absolute_errors[5], mean_mae
```

### Các bước visualization

**Bước 1 — Bảng kết quả:**

```python
df_pred = pd.DataFrame({
    'Score Type':   DISPLAY_NAMES,
    'Ground Truth': [f'{v:.1f}' for v in pred_result['ground_truth']],
    'Prediction':   [f'{v:.2f}' for v in pred_result['predictions']],
    'Abs Error':    [f'{v:.2f}' for v in pred_result['absolute_errors']],
    'Status':       ['✓ OK' if e < 0.5 else '⚠ Check' for e in pred_result['absolute_errors']],
})
display(df_pred.style.applymap(lambda v: 'color: green' if '✓' in str(v) else ('color: orange' if '⚠' in str(v) else '')))
```

**Bước 2 — Bar chart grouped:**

```python
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(5)
w = 0.35
ax.bar(x - w/2, pred_result['ground_truth'], w, label='Ground Truth', color='#2196F3', alpha=0.85)
ax.bar(x + w/2, pred_result['predictions'],  w, label='Prediction',   color='#FF5722', alpha=0.85)
# Thêm error bars hoặc annotation cho abs error
for i, err in enumerate(pred_result['absolute_errors']):
    ax.annotate(f'Δ{err:.2f}', xy=(x[i], max(pred_result['ground_truth'][i], pred_result['predictions'][i]) + 0.15),
                ha='center', fontsize=8, color='gray')
ax.set_xticks(x); ax.set_xticklabels(DISPLAY_NAMES, rotation=20, ha='right')
ax.set_ylim(1, 11); ax.set_ylabel('Score (1-10)'); ax.legend()
ax.set_title(f'Sample {slot} — Prediction vs Ground Truth  (Mean MAE = {pred_result["mean_mae"]:.3f})')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/prediction_chart.png', dpi=DEFAULT_DPI)
plt.show()
```

**Bước 3 — Interpretation Markdown:**

Cell Markdown phân tích kết quả bằng tiếng Việt, ví dụ:
```markdown
**Nhận xét Sample A:** Mô hình dự đoán Overall Satisfaction = 7.82 (GT = 8.0, lệch 0.18 điểm).
Food Score và Service Score được dự đoán chính xác. Price Score có sai lệch cao nhất (0.65).
→ Kết luận: Mô hình hoạt động tốt trên mẫu này.
```

**Bước 4 — Lưu:**

```python
os.makedirs(f'{DEMO_OUT}/sample_{slot}', exist_ok=True)
with open(f'{DEMO_OUT}/sample_{slot}/prediction.json', 'w', encoding='utf-8') as f:
    json.dump({
        'sample_id': selected_samples[slot],
        'predictions': dict(zip(TARGET_NAMES, pred_result['predictions'])),
        'ground_truth': dict(zip(TARGET_NAMES, pred_result['ground_truth'])),
        'absolute_errors': dict(zip(TARGET_NAMES, pred_result['absolute_errors'])),
        'mean_mae': pred_result['mean_mae'],
    }, f, ensure_ascii=False, indent=2)
```

**Lecturer focus:** Cột "Abs Error" và chart bar gap. Mean MAE < 0.5 = excellent.

---

## 7. Trình Bày Grad-CAM

### Mục đích

Cho giảng viên thấy vùng ảnh nào mô hình chú ý khi đánh giá Overall Satisfaction.

### 7.1 Giới hạn kỹ thuật — PHẢI HIỂN THỊ TRƯỚC

```markdown
> ⚠️ **Giới hạn Grad-CAM trong kiến trúc này:**
>
> CrossAttentionFusion dùng chung encoder Swin-B cho tất cả 5 targets.
> Chỉ lớp Linear cuối (256→5) là khác nhau theo target.
> → Gradient của 5 targets đến `encoder.norm` có cosine similarity rất cao (>0.95).
> → 5 Grad-CAM heatmap trông gần như giống nhau về mặt thị giác.
>
> **Giải pháp demo:** Chỉ hiển thị Grad-CAM cho Overall Satisfaction (target quan trọng nhất).
> Để phân tích per-target chính xác → dùng SHAP (xem Mục 10).
```

### 7.2 Target layer

```python
target_layer = find_target_layer(model)
# = model.image_model.encoder.norm
# Output format: BHWC [B, 7, 7, 1024]
# normalize_feature_map_to_bchw() xử lý tự động trong compute_gradcam_for_image()
```

### 7.3 Reuse vs Regenerate Grad-CAM

```python
# Kiểm tra artifact đã có
gradcam_overlay_path = f'{XAI_DIR}/gradcam/{sid}/gradcam_img0_overall.png'
gradcam_raw_path     = f'{XAI_DIR}/gradcam/{sid}/cam_overall_img0.npy'
regen, reason = should_regenerate('gradcam', sid, XAI_DIR, EXP_ID)

if not regen and os.path.exists(gradcam_overlay_path):
    # REUSE — load từ disk
    cam_overlay = np.array(Image.open(gradcam_overlay_path))
    cam_raw = np.load(gradcam_raw_path) if os.path.exists(gradcam_raw_path) else None
    print(f'[REUSE] Grad-CAM from {gradcam_overlay_path}')
else:
    # REGENERATE
    cam_raw = compute_gradcam_for_image(
        model=model,
        sample=samples[slot]['data'],
        target_idx=4,        # overall_satisfaction
        image_idx=0,
        target_layer=target_layer,
        device=device,
    )  # → numpy [H, W] normalized [0, 1], H=W=7 sau đó upsample
    cam_overlay = overlay_cam_on_image(cam_raw, samples[slot]['data']['loaded_images'][0])
    Image.fromarray(cam_overlay).save(f'{DEMO_OUT}/sample_{slot}/gradcam_overlay_overall.png')
    print(f'[REGEN] Grad-CAM generated, reason: {reason}')
```

### 7.4 Three-Panel Visualization: Original + Raw Heatmap + Overlay

```python
orig_img = samples[slot]['data']['loaded_images'][0].resize((224, 224))
orig_arr = np.array(orig_img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Original Image
axes[0].imshow(orig_arr)
axes[0].set_title('① Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Panel 2: Raw Heatmap (cần cam_raw là [H,W] float)
if cam_raw is not None:
    axes[1].imshow(cam_raw, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f'② Raw Grad-CAM Heatmap\n(7×7 patches, upsample→224×224)', fontsize=11)
    cbar = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity')
else:
    axes[1].text(0.5, 0.5, 'Raw heatmap\nnot available', ha='center', va='center')
    axes[1].set_title('② Raw Heatmap', fontsize=11)
axes[1].axis('off')

# Panel 3: Overlay
axes[2].imshow(cam_overlay)
axes[2].set_title('③ Grad-CAM Overlay\n(Vùng đỏ = mô hình chú ý nhiều)', fontsize=11)
axes[2].axis('off')

plt.suptitle(f'Grad-CAM — Overall Satisfaction | Sample {slot} ({sid})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/gradcam_3panel.png', dpi=DEFAULT_DPI)
plt.show()
```

**Mô tả 3 panel cho giảng viên (Markdown cell):**
```markdown
**Đọc 3 panel Grad-CAM:**
- **① Original Image** — ảnh đầu vào (resized 224×224).
- **② Raw Heatmap** — ma trận 7×7 activation strength (màu vàng/đỏ = cao). Đây là gradient signal thuần túy tại `encoder.norm` của Swin-B.
- **③ Overlay** — heatmap được upsample và chồng lên ảnh gốc. Vùng đỏ/cam = mô hình tập trung cao khi đánh giá Overall Satisfaction.

**Giảng viên chú ý:** Vùng nóng có ý nghĩa gì? Thức ăn? Không gian? Bảng hiệu?
```

**Lecturer takeaway:** "Mô hình nhìn đúng chỗ không? Vùng nóng có semantically consistent không?"

---

## 8. Trình Bày PhoBERT Attention

### Mục đích

Cho giảng viên thấy những từ nào trong review được mô hình "chú ý" khi xử lý.

### 8.1 Eager attention note

```python
# enable_eager_attention() đã được gọi trong load_model()
# KHÔNG cần gọi lại — chỉ cần assert để verify
attn_impl = model.text_model.encoder.config._attn_implementation
print(f'Attention implementation: {attn_impl}')
# Expected: 'eager'
```

### 8.2 Extract + Aggregate

```python
# Reuse hoặc regenerate attention artifacts
attn_raw_path  = f'{XAI_DIR}/attention/{sid}/raw_attention.npz'
topk_path      = f'{XAI_DIR}/attention/{sid}/topk_tokens.json'
word_imp_path  = f'{XAI_DIR}/attention/{sid}/word_importance.json'

if all(os.path.exists(p) for p in [attn_raw_path, topk_path, word_imp_path]):
    npz = np.load(attn_raw_path)
    attn_array = npz['attentions']          # [12, 12, L, L]
    tokens = list(np.load(attn_raw_path, allow_pickle=True)['tokens'])
    with open(topk_path) as f: topk_tokens = json.load(f)
    with open(word_imp_path) as f: word_importances_raw = json.load(f)
    print(f'[REUSE] Attention from {sid}')
else:
    # Regenerate
    attn_result = extract_phobert_attention(
        model, samples[slot]['data']['input_ids'],
        samples[slot]['data']['attention_mask'], tokenizer
    )
    # attn_result: {'attentions': [12,12,L,L], 'tokens': list, 'seq_len': int}
    attn_array = attn_result['attentions']
    tokens     = attn_result['tokens']
    print(f'[REGEN] Attention extracted, seq_len={attn_result["seq_len"]}')

# Aggregate: last_layer_mean
agg_matrix = aggregate_attention(attn_array, strategy='last_layer_mean')
# agg_matrix: [L, L]

cls_result     = cls_token_importance(agg_matrix, tokens)
word_importances = merge_subword_attention(
    cls_result['importances'], tokens, strategy='mean'
)
# word_importances: list of (word, score), sorted descending
```

### 8.3 Subword → Word Merging Explanation (Markdown cell bắt buộc)

```markdown
**Tại sao dùng merged words thay vì raw subwords?**

PhoBERT tokenize tiếng Việt thành BPE (Byte-Pair Encoding) subwords.
Ví dụ: "ngon" → `["_ng", "on"]`, "nhiệt_tình" → `["_nhiệt", "_tình"]`.

Subword fragments không có nghĩa riêng lẻ với người đọc.
→ `merge_subword_attention()` gộp các subword fragments về từ gốc, lấy mean score.
→ Kết quả: word-level importance dễ đọc và có ý nghĩa ngữ nghĩa.
```

### 8.4 Visualization 1: Highlighted Review Text

```python
from IPython.display import HTML, display as ipy_display

def highlight_text_html(word_importances, text):
    word_scores = {w: s for w, s in word_importances}
    max_s = max(word_scores.values()) if word_scores else 1.0
    words = text.split()
    parts = []
    for w in words:
        clean = w.strip('.,!?;:')
        s = word_scores.get(clean, word_scores.get(w, 0))
        intensity = max(0, min(255, int(255 * (1 - s / (max_s + 1e-8)))))
        bg = f'rgb(255,{intensity},{intensity})'
        parts.append(f'<span style="background:{bg};padding:2px 4px;border-radius:3px;margin:1px">{w}</span>')
    return '<div style="font-size:14px;line-height:2.0;font-family:sans-serif">' + ' '.join(parts) + '</div>'

ipy_display(HTML(f'<h4>PhoBERT Attention — Highlighted Text (Sample {slot})</h4>'))
ipy_display(HTML(highlight_text_html(word_importances, samples[slot]['data']['text'])))
```

### 8.5 Visualization 2: Top-K Word Importance Bar Chart

```python
TOP_K_WORDS = 15
top_words = word_importances[:TOP_K_WORDS]
words_disp = [w for w, _ in top_words]
scores_disp = [s for _, s in top_words]
colors_disp = plt.cm.YlOrRd(np.array(scores_disp) / (max(scores_disp) + 1e-8))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(words_disp[::-1], scores_disp[::-1], color=colors_disp[::-1], alpha=0.9)
ax.set_xlabel('CLS Attention Score (normalized)', fontsize=11)
ax.set_title(f'Top {TOP_K_WORDS} Important Words — PhoBERT Attention\n(Sample {slot}: {sid})', fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/attention_top_words.png', dpi=DEFAULT_DPI)
plt.show()
```

### 8.6 Visualization 3: CLS Attention Heatmap (Top Tokens)

Hiển thị sub-matrix của attention matrix cho top-20 tokens (để tránh ma trận quá lớn):

```python
TOP_K_TOKENS_HEAT = 20
# Lấy top-K token indices theo CLS attention row
cls_row = agg_matrix[0]  # CLS attention to all tokens
top_idx = np.argsort(cls_row)[::-1][:TOP_K_TOKENS_HEAT]
top_idx_sorted = sorted(top_idx)  # giữ thứ tự trong sequence

sub_matrix = agg_matrix[np.ix_(top_idx_sorted, top_idx_sorted)]
sub_tokens  = [tokens[i] for i in top_idx_sorted]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(sub_matrix, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(sub_tokens)))
ax.set_yticks(range(len(sub_tokens)))
ax.set_xticklabels(sub_tokens, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(sub_tokens, fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title(f'CLS Attention Sub-matrix (Top {TOP_K_TOKENS_HEAT} tokens)\n(Sample {slot})', fontsize=11)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/attention_cls_heatmap.png', dpi=DEFAULT_DPI)
plt.show()
```

### 8.7 Reuse existing artifacts

Nếu file đã có từ Phase 3, ưu tiên load thay vì regenerate:
```python
# Phase 3 artifacts:
# cls_importance_word_bar.png  ← dùng thay thế Visualization 2
# cls_importance_subword_bar.png
# raw_attention.npz
# topk_tokens.json
# word_importance.json
existing_bar = f'{XAI_DIR}/attention/{sid}/cls_importance_word_bar.png'
if os.path.exists(existing_bar) and not regen_attention:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.array(Image.open(existing_bar)))
    ax.axis('off')
    plt.show()
```

### 8.8 Disclaimer bắt buộc

```markdown
**⚠️ Attention ≠ Causality (quan trọng):**

PhoBERT Attention cho thấy token nào được mô hình **tập trung** xử lý.
Tuy nhiên, theo Jain & Wallace (2019) và Wiegreffe & Pinter (2019),
attention weights **không chứng minh nhân quả** — một từ có attention cao
không có nghĩa là nó quyết định duy nhất kết quả prediction.

→ Kết hợp với SHAP và LIME để có góc nhìn toàn diện hơn.
```

**Lecturer takeaway:** "Từ nào được model nhấn mạnh? Có khớp với common sense không?"

---

## 9. Trình Bày Cross-Attention

### Mục đích

Giải thích cơ chế **kết nối ngôn ngữ và hình ảnh** trong CrossAttentionFusion — điểm độc đáo nhất của kiến trúc.

### 9.1 Tại sao Cross-Attention khác Self-Attention? (Markdown)

```markdown
## Cross-Attention vs Self-Attention

**Self-Attention (PhoBERT):** Mỗi token attend to các token KHÁC trong cùng sequence text.
→ Câu hỏi: "Từ này liên quan đến từ nào khác trong câu?"

**Cross-Attention (CrossAttentionFusion):** Text tokens attend to Image patches, và ngược lại.
→ Câu hỏi: "Từ này liên kết với vùng ảnh nào?" / "Vùng ảnh này hỗ trợ từ nào?"

**Bidirectional** = 2 hướng độc lập:
- T2I (Text → Image): text_proj([B,T,512]) làm Query, image_proj([B,49,512]) làm Key/Value
- I2T (Image → Text): image_proj([B,49,512]) làm Query, text_proj([B,T,512]) làm Key/Value

Hai module cross-attention này (cross_attn_t2i và cross_attn_i2t) là KHÔNG phải transpose
của nhau — chúng có trọng số riêng biệt.
```

### 9.2 Extract Cross-Attention

```python
# Reuse hoặc regenerate
ca_summary_path = f'{XAI_DIR}/cross_attention/{sid}/cross_attention_summary.json'
ca_topk_path    = f'{XAI_DIR}/cross_attention/{sid}/token_patch_topk.json'
ca_raw_path     = f'{XAI_DIR}/cross_attention/{sid}/cross_attention_raw.npz'

if all(os.path.exists(p) for p in [ca_summary_path, ca_topk_path]):
    with open(ca_summary_path) as f: ca_summary = json.load(f)
    with open(ca_topk_path)    as f: ca_topk    = json.load(f)
    if os.path.exists(ca_raw_path):
        ca_npz = np.load(ca_raw_path)
        t2i_attn = ca_npz['t2i_attn']  # [T, 49]
        i2t_attn = ca_npz['i2t_attn']  # [49, T]
        ca_tokens = list(ca_npz.get('tokens', np.array([])))
    print(f'[REUSE] Cross-Attention from {sid}')
else:
    ca_result = extract_cross_attention(model, samples[slot]['data'], tokenizer)
    # ca_result: {'t2i_attn': [T,49], 'i2t_attn': [49,T], 'tokens': list,
    #             'seq_len': T, 'num_patches': 49}
    t2i_attn  = ca_result['t2i_attn']
    i2t_attn  = ca_result['i2t_attn']
    ca_tokens = ca_result['tokens']
    print(f'[REGEN] Cross-Attention extracted, T={ca_result["seq_len"]}, P=49')
```

### 9.3 Visualization 1: T2I Heatmap (Token → Patch)

```python
# Hiển thị top-20 tokens để tránh ma trận quá lớn
TOP_K_T2I = 20
t2i_arr = np.array(t2i_attn) if not isinstance(t2i_attn, np.ndarray) else t2i_attn
top_token_idx = np.argsort(t2i_arr.max(axis=1))[-TOP_K_T2I:][::-1]
t2i_display   = t2i_arr[top_token_idx, :]
tokens_disp   = [ca_tokens[i] if i < len(ca_tokens) else f'tok_{i}' for i in top_token_idx]

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(t2i_display, cmap='YlOrRd', aspect='auto')
ax.set_yticks(range(len(tokens_disp))); ax.set_yticklabels(tokens_disp, fontsize=9)
ax.set_xlabel('Image Patches (49 = 7×7 grid)', fontsize=11)
ax.set_title(f'Cross-Attention T2I: Token → Patch\n"Khi đọc từ này, mô hình nhìn vùng ảnh nào?"  (Sample {slot})', fontsize=12)
plt.colorbar(im, ax=ax)
# Thêm grid 7×7 để dễ đọc patch coordinates
for x_line in range(7, 49, 7):
    ax.axvline(x_line - 0.5, color='white', lw=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/cross_attn_t2i_heatmap.png', dpi=DEFAULT_DPI)
plt.show()
```

### 9.4 Visualization 2: Token Overlay on Image (Best Token)

```python
t2i_arr = np.array(t2i_attn)
best_token_idx = int(np.argmax(t2i_arr.max(axis=1)))
best_token     = ca_tokens[best_token_idx] if best_token_idx < len(ca_tokens) else f'tok_{best_token_idx}'
patch_weights  = t2i_arr[best_token_idx, :].reshape(7, 7)

import cv2
heat = cv2.resize(patch_weights.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
orig_arr_224 = np.array(samples[slot]['data']['loaded_images'][0].resize((224, 224)))
colormap     = (plt.cm.hot(heat)[:, :, :3] * 255).astype(np.uint8)
overlay      = (0.5 * orig_arr_224 + 0.5 * colormap).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(orig_arr_224); axes[0].set_title('Original'); axes[0].axis('off')
axes[1].imshow(heat, cmap='hot', vmin=0, vmax=1); axes[1].set_title(f'Patch Weights\nfor "{best_token}"'); axes[1].axis('off')
axes[2].imshow(overlay); axes[2].set_title(f'Overlay\n(Token: "{best_token}" attends here)'); axes[2].axis('off')
plt.suptitle(f'Cross-Attention Overlay — T2I  (Sample {slot})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/cross_attn_t2i_overlay.png', dpi=DEFAULT_DPI)
plt.show()
```

### 9.5 Visualization 3: I2T Heatmap (Patch → Token)

```python
i2t_arr = np.array(i2t_attn)  # [49, T]
TOP_K_TOKENS_I2T = 20
# Lấy top tokens theo max patch attention
top_t_idx    = np.argsort(i2t_arr.max(axis=0))[-TOP_K_TOKENS_I2T:][::-1]
i2t_display  = i2t_arr[:, top_t_idx]
t_labels     = [ca_tokens[i] if i < len(ca_tokens) else f'tok_{i}' for i in top_t_idx]

fig, ax = plt.subplots(figsize=(16, 7))
im = ax.imshow(i2t_display.T, cmap='Blues', aspect='auto')
ax.set_xticks(range(49)); ax.set_xticklabels([f'P{i}' for i in range(49)], fontsize=6, rotation=45)
ax.set_yticks(range(len(t_labels))); ax.set_yticklabels(t_labels, fontsize=9)
ax.set_xlabel('Image Patches (P0–P48)', fontsize=11)
ax.set_title(f'Cross-Attention I2T: Patch → Token\n"Khi nhìn vùng ảnh này, mô hình liên kết từ nào?"  (Sample {slot})', fontsize=12)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/cross_attn_i2t_heatmap.png', dpi=DEFAULT_DPI)
plt.show()
```

### 9.6 Visualization 4: Top-K Token-Patch Pairs

```python
TOP_K_PAIRS = 10
flat_t2i = t2i_arr.flatten()
top_k_flat = np.argsort(flat_t2i)[-TOP_K_PAIRS:][::-1]
rows_disp = []
for idx in top_k_flat:
    t_idx = int(idx // 49)
    p_idx = int(idx % 49)
    row, col = divmod(p_idx, 7)
    tok = ca_tokens[t_idx] if t_idx < len(ca_tokens) else f'tok_{t_idx}'
    rows_disp.append({'Token': tok, 'Patch': f'[{row},{col}]', 'Score': f'{flat_t2i[idx]:.4f}'})

df_pairs = pd.DataFrame(rows_disp)
print(f'Top {TOP_K_PAIRS} Token→Patch Attention Pairs (Sample {slot}):')
display(df_pairs)
```

### 9.7 Reuse Existing Phase 3 Artifacts

Phase 3 CrossAttentionExplainer đã tạo sẵn các file sau — ưu tiên load chúng:

```python
# Thư mục: xai/cross_attention/{sid}/
existing_artifacts = {
    'topk_heatmap':     f'{XAI_DIR}/cross_attention/{sid}/topk_token_patch_heatmap.png',
    'bipartite':        f'{XAI_DIR}/cross_attention/{sid}/token_patch_bipartite_graph.png',
    'overlay_grid':     f'{XAI_DIR}/cross_attention/{sid}/top_tokens_patch_overlay_grid.png',
    'patch_importance': f'{XAI_DIR}/cross_attention/{sid}/patch_importance.png',
    # token-specific overlays:
    'token_overlay_X':  f'{XAI_DIR}/cross_attention/{sid}/token_overlay_<token>.png',
}
# Hiển thị các file đã có:
for name, path in existing_artifacts.items():
    if os.path.exists(path):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(np.array(Image.open(path))); ax.axis('off')
        ax.set_title(f'{name} (Phase 3 artifact)', fontsize=11)
        plt.tight_layout(); plt.show()
```

**Đặc biệt:** `top_tokens_patch_overlay_grid.png` là grid nhiều ảnh overlay cho top tokens — rất ấn tượng, nên ưu tiên hiển thị.

### 9.8 Bidirectional Interpretation Summary (Markdown)

```markdown
**Kết quả Cross-Attention (Bidirectional):**

- **T→I (Token→Patch):** Từ `"ngon"` attend mạnh vào patches [3,4] và [4,4]
  → Khi đọc từ "ngon", mô hình nhìn vào vùng thức ăn ở giữa/dưới ảnh.

- **I→T (Patch→Token):** Patch [1,1] (góc trên-trái) attend mạnh vào từ `"không_gian"`
  → Khi "nhìn" vào vùng nội thất ở góc trên, mô hình liên kết với từ "không_gian".

→ **Cross-Attention alignment cho thấy mô hình đã học được kết nối đúng giữa ngôn ngữ và hình ảnh.**
```

**Lecturer takeaway:** "Model liên kết từ ngữ với vùng ảnh đúng không? Đây là bằng chứng mô hình hiểu multimodal."

---

## 10. Trình Bày SHAP

### Mục đích

Trả lời câu hỏi định lượng: **Text-origin hay Image-origin đóng góp bao nhiêu % cho mỗi target?**

### 10.1 Chuẩn bị Background Embeddings

```python
# Ưu tiên load background đã có từ Phase 4
bg_path = f'{XAI_DIR}/shap/raw/background_fused.pt'
if os.path.exists(bg_path):
    background = torch.load(bg_path, map_location='cpu')
    print(f'[REUSE] Background: {background.shape}')
else:
    # Fallback: extract từ val set (cần DataLoader)
    # background = extract_fused_embeddings(model, val_loader, device, max_samples=100)[0]
    print('[WARNING] Background not found — SHAP will be skipped.')
    background = None
```

### 10.2 Extract Fused Embedding cho Sample

```python
# Hook model.head để capture fused embedding [1, 1024]
sample_fused_emb = None
captured = {}

def hook_fn(module, args):
    captured['fused'] = args[0].detach().cpu() if isinstance(args, tuple) else args.detach().cpu()

hook_handle = model.head.register_forward_pre_hook(hook_fn)
try:
    with torch.no_grad():
        model(**{k: v for k, v in samples[slot]['data'].items()
                 if k in ['input_ids', 'attention_mask', 'pixel_values', 'num_images']})
    sample_fused_emb = captured.get('fused')  # [1, 1024]
finally:
    hook_handle.remove()
    captured.clear()
print(f'Fused embedding shape: {sample_fused_emb.shape}')
```

### 10.3 Reuse SHAP Artifacts

```python
shap_contrib_path = f'{XAI_DIR}/shap/{sid}/shap_modality_contribution.json'
shap_raw_path     = f'{XAI_DIR}/shap/{sid}/raw_shap_values.npz'
shap_chart_path   = f'{XAI_DIR}/shap/{sid}/shap_modality_contribution.png'

regen_shap, _ = should_regenerate('shap', sid, XAI_DIR, EXP_ID)

if not regen_shap and os.path.exists(shap_contrib_path) and background is not None:
    with open(shap_contrib_path) as f:
        shap_contrib = json.load(f)
    shap_raw = np.load(shap_raw_path) if os.path.exists(shap_raw_path) else None
    print(f'[REUSE] SHAP from {sid}')
elif background is not None and sample_fused_emb is not None:
    # Regenerate
    shap_contrib = {}
    shap_vals_all = {}
    for t_idx, fname in enumerate(FACTOR_NAMES):
        wrapper = FusionHeadWrapper(model.head, score_index=t_idx)
        sv, base_val = compute_shap_values(
            wrapper, background.to(device), sample_fused_emb.to(device)
        )
        contrib = modality_contribution(sv[0], text_dim=CROSS_ATTN_HIDDEN_DIM)
        shap_contrib[fname] = {**contrib, 'base_value': base_val,
                                'predicted': float(pred_result['predictions'][t_idx])}
        shap_vals_all[fname] = sv[0]
    print(f'[REGEN] SHAP computed for all 5 targets')
else:
    shap_contrib = None
    print('[SKIP] SHAP skipped: background or embedding not available.')
```

### 10.4 Visualization 1: Modality Contribution Pie + Per-Target Stacked Bar

```python
if shap_contrib:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Pie chart: Overall Satisfaction ---
    overall = shap_contrib.get('overall', {})
    text_pct  = overall.get('text_pct', 0)
    image_pct = overall.get('image_pct', 0)
    axes[0].pie(
        [text_pct, image_pct],
        labels=['Text-origin\n(dims 0:512)', 'Image-origin\n(dims 512:1024)'],
        colors=['#4ECDC4', '#FF6B6B'], autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 12}
    )
    axes[0].set_title(f'Modality Contribution\nOverall Satisfaction (|SHAP| sum)', fontsize=12)

    # --- Stacked bar: 5 targets ---
    x = np.arange(5)
    text_pcts  = [shap_contrib[f]['text_pct']  for f in FACTOR_NAMES]
    image_pcts = [shap_contrib[f]['image_pct'] for f in FACTOR_NAMES]
    b1 = axes[1].bar(x, text_pcts,  label='Text-origin', color='#4ECDC4', alpha=0.85)
    b2 = axes[1].bar(x, image_pcts, bottom=text_pcts, label='Image-origin', color='#FF6B6B', alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(DISPLAY_NAMES, rotation=20, ha='right', fontsize=10)
    axes[1].set_ylabel('Contribution (%)'); axes[1].set_ylim(0, 110)
    axes[1].legend(); axes[1].set_title('Modality Contribution per Target', fontsize=12)
    axes[1].axhline(50, color='black', lw=1, ls='--', alpha=0.4)
    axes[1].grid(axis='y', alpha=0.3)
    for i, (tp, ip) in enumerate(zip(text_pcts, image_pcts)):
        axes[1].text(i, tp/2, f'{tp:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        axes[1].text(i, tp + ip/2, f'{ip:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    plt.suptitle(f'SHAP Modality Contribution — Sample {slot} ({sid})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DEMO_OUT}/sample_{slot}/shap_modality.png', dpi=DEFAULT_DPI)
    plt.show()
```

### 10.5 Visualization 2: SHAP Waterfall Chart (Overall Satisfaction)

Waterfall chart phân tích cụ thể dims đóng góp positive/negative cho Overall Satisfaction:

```python
if shap_contrib and shap_vals_all:
    sv_overall = shap_vals_all.get('overall')  # [1024]
    if sv_overall is not None:
        # Top-10 positive dims + top-10 negative dims
        base_val = shap_contrib.get('overall', {}).get('base_value', 0)
        pred_val = shap_contrib.get('overall', {}).get('predicted', 0)

        top_pos_idx = np.argsort(sv_overall)[::-1][:10]
        top_neg_idx = np.argsort(sv_overall)[:10]
        combined_idx = np.concatenate([top_pos_idx, top_neg_idx])
        combined_vals = sv_overall[combined_idx]

        # Phân nhóm text-origin / image-origin
        labels = []
        colors = []
        for i, (idx, val) in enumerate(zip(combined_idx, combined_vals)):
            origin = 'Text' if idx < 512 else 'Image'
            labels.append(f'{"+" if val>0 else ""}{val:.3f}\n[{origin} dim {idx}]')
            colors.append('#4ECDC4' if idx < 512 else '#FF6B6B')

        fig, ax = plt.subplots(figsize=(14, 6))
        y_pos = np.arange(len(combined_idx))
        bars = ax.barh(y_pos, combined_vals, color=colors, alpha=0.85)
        ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel('SHAP Value')
        ax.set_title(
            f'SHAP Waterfall — Overall Satisfaction\n'
            f'Base={base_val:.2f} → Prediction={pred_val:.2f}  (Sample {slot})\n'
            f'🔵 Text-origin dims (0:512)   🔴 Image-origin dims (512:1024)',
            fontsize=11
        )
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{DEMO_OUT}/sample_{slot}/shap_waterfall_overall.png', dpi=DEFAULT_DPI)
        plt.show()
```

### 10.6 SHAP Disclaimer bắt buộc

```markdown
**⚠️ Hiểu đúng "text-origin" và "image-origin" trong SHAP:**

SHAP được tính trên **fused embedding [1024 dims]** — đầu ra của CrossAttentionFusion.

- **Dims 0:512** (text-origin): kết quả của text queries attended to image patches
  → Không phải "pure text" mà là text features ĐÃ ĐƯỢC ENRICHED bởi image context.
- **Dims 512:1024** (image-origin): kết quả của image queries attended to text tokens
  → Không phải "pure image" mà là image features ĐÃ ĐƯỢC ENRICHED bởi text context.

**Kết luận:** "Text-origin đóng góp 60%" có nghĩa là "nhánh bắt nguồn từ text
(nhưng đã được trộn với thông tin ảnh qua cross-attention) đóng góp 60% vào decision."
```

**Lecturer takeaway:** "Target nào phụ thuộc nhiều vào ảnh, target nào phụ thuộc vào text?"

---

## 11. Trình Bày LIME

### Mục đích

Kiểm tra cục bộ: **Nếu xóa từ X hay che vùng Y, dự đoán thay đổi thế nào?**

### 11.1 Reuse Phase 5 Artifacts

```python
# Khởi tạo tất cả variables để tránh NameError trong visualization
lime_text_weights = {}
lime_text_exp     = None
lime_image_exp    = None

lime_text_path    = f'{XAI_DIR}/lime/{sid}/{sid}_lime_text_overall_weights.json'
lime_img_pos_path = f'{XAI_DIR}/lime/{sid}/{sid}_lime_image_overall_positive.png'

regen_lime, _ = should_regenerate('lime', sid, XAI_DIR, EXP_ID)

if not regen_lime and os.path.exists(lime_text_path):
    with open(lime_text_path) as f:
        lime_text_weights = json.load(f)
    print(f'[REUSE] LIME from {sid}')
else:
    # Regenerate — cảnh báo: LIME text ~500 perturbs, image ~1000 perturbs (~5-10 phút)
    print('[REGEN] Generating LIME (this may take 5–10 minutes)...')
    lime_text_exp, ok_lt = run_safe(
        f'LIME_text_{slot}', run_lime_text,
        model=model, sample=samples[slot]['data'],
        score_index=4, tokenizer=tokenizer, device=device,
        num_features=15, num_samples=500, seed=42,
    )
    lime_image_exp, ok_li = run_safe(
        f'LIME_image_{slot}', run_lime_image,
        model=model, sample=samples[slot]['data'],
        score_index=4, image_processor=image_processor, device=device,
        num_samples=1000, seed=42,
    )
    if lime_text_exp is not None:
        lime_text_weights = dict(lime_text_exp.as_list(label=1))
```

### 11.2 Combined 4-Panel Figure

Đây là điểm nhấn của LIME section — một figure duy nhất show tất cả:

```python
from lime.lime_image import mark_boundaries

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 4, fig, wspace=0.3, hspace=0.35)

# Panel 1 (top-left): Original Image
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(samples[slot]['data']['loaded_images'][0].resize((224, 224)))
ax1.set_title('Original Image', fontsize=11, fontweight='bold'); ax1.axis('off')

# Panel 2 (top-center): LIME Image — Positive Superpixels
ax2 = fig.add_subplot(gs[0, 1:3])
if os.path.exists(lime_img_pos_path):
    ax2.imshow(np.array(Image.open(lime_img_pos_path)))
elif lime_image_exp is not None:
    temp, mask = lime_image_exp.get_image_and_mask(
        label=1, positive_only=True, num_features=5, hide_rest=False
    )
    img_show = temp / 255.0 if temp.max() > 1 else temp
    ax2.imshow(mark_boundaries(img_show, mask))
ax2.set_title('LIME Image: Overall Satisfaction\n(Viền xanh = Superpixels quan trọng)', fontsize=11); ax2.axis('off')

# Panel 3 (top-right): Prediction Summary
ax3 = fig.add_subplot(gs[0, 3])
pred_text = '\n'.join([
    f'{dn}: {p:.2f} (GT={g:.1f})' 
    for dn, p, g in zip(DISPLAY_NAMES, pred_result['predictions'], pred_result['ground_truth'])
])
ax3.text(0.05, 0.95, f'Predictions\n\n{pred_text}', 
         transform=ax3.transAxes, fontsize=9, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F0F4FF', alpha=0.8))
ax3.axis('off'); ax3.set_title('Prediction Summary', fontsize=11, fontweight='bold')

# Panel 4 (bottom): LIME Text Weights Bar Chart
ax4 = fig.add_subplot(gs[1, :])
if lime_text_weights:
    top_lime = sorted(lime_text_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    lw_words  = [w for w, _ in top_lime]
    lw_scores = [s for _, s in top_lime]
    lw_colors = ['#2ECC71' if s > 0 else '#E74C3C' for s in lw_scores]
    ax4.barh(lw_words[::-1], lw_scores[::-1], color=lw_colors[::-1], alpha=0.85)
    ax4.axvline(0, color='black', lw=1)
    ax4.set_xlabel('LIME Weight (positive = tăng score, negative = giảm score)', fontsize=10)
    ax4.set_title('LIME Text — Overall Satisfaction (xanh = giúp tăng điểm, đỏ = kéo giảm điểm)', fontsize=11)
    ax4.grid(axis='x', alpha=0.3)
elif 'lime_text_exp' in dir() and lime_text_exp is not None:
    top_lime_raw = lime_text_exp.as_list(label=1)[:15]
    lw_words  = [w for w, _ in top_lime_raw]
    lw_scores = [s for _, s in top_lime_raw]
    lw_colors = ['#2ECC71' if s > 0 else '#E74C3C' for s in lw_scores]
    ax4.barh(lw_words[::-1], lw_scores[::-1], color=lw_colors[::-1], alpha=0.85)
    ax4.axvline(0, color='black', lw=1)
    ax4.set_xlabel('LIME Weight', fontsize=10)
    ax4.set_title('LIME Text — Overall Satisfaction', fontsize=11)
    ax4.grid(axis='x', alpha=0.3)

plt.suptitle(f'LIME Local Explanation — Sample {slot} ({sid})\n'
             f'Overall Satisfaction: Pred={pred_result["predictions"][4]:.2f}, GT={pred_result["ground_truth"][4]:.1f}',
             fontsize=13, fontweight='bold')
plt.savefig(f'{DEMO_OUT}/sample_{slot}/lime_combined.png', dpi=DEFAULT_DPI, bbox_inches='tight')
plt.show()
```

### 11.3 LIME Disclaimer bắt buộc

```markdown
**⚠️ LIME — Local Explanation Only:**

LIME tạo **stochastic perturbations** xung quanh sample cụ thể này.
→ Kết quả có thể khác nhau giữa các lần chạy (khác random seed).
→ Đây là **giải thích cục bộ** — không phản ánh hành vi toàn cục của mô hình.

**Pseudo-classification:** LIME cần classification output [p_low, p_high].
Hệ thống dùng sigmoid(score) → [1-sigmoid, sigmoid] để tương thích API.
Đây là kỹ thuật engineering, không thay đổi bản chất bài toán regression.

→ Kết hợp LIME với SHAP để phân biệt global feature importance vs local sensitivity.
```

**Lecturer takeaway:** "Từ nào, nếu xóa đi, làm thay đổi dự đoán nhiều nhất? Điều này có hợp lý không?"

---

## 12. Trình Bày AI Agent

### 12.1 Architecture Explanation (Markdown)

```markdown
## AI Agent — Lớp Verbalization Cuối Cùng

```
Prediction Model (5 scores cố định)
        ↓
    XAI Artifacts (Grad-CAM + Attention + Cross-Attention + SHAP + LIME)
        ↓
┌─────────────────────────────────────────────────────────┐
│  DETERMINISTIC LAYER (không có LLM)                     │
│  1. EvidenceLoader  — load tất cả artifacts từ disk     │
│  2. EvidenceBuilder — compress thành text evidence      │
│  3. ReasoningGraph  — pre-LLM reasoning (support/       │
│                       conflict/missing/strength)        │
│  4. PromptBuilder   — build messages list for OpenAI    │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│  GPT-4o (chỉ verbalization — KHÔNG dự đoán lại)        │
│  • Nhận structured evidence + reasoning graph           │
│  • Tạo natural-language explanation trong JSON format    │
│  • 10 anti-hallucination rules trong system prompt      │
└─────────────────────────────────────────────────────────┘
        ↓
    OutputValidator (schema + SHAP grounding ±5% tolerance)
        ↓
    ReportGenerator → JSON + Markdown
        ↓
    Customer View (đơn giản) + Technical View (đầy đủ)
```

**GPT-4o KHÔNG thay đổi scores. KHÔNG tạo evidence mới. Chỉ diễn đạt bằng ngôn ngữ.**
```

### 12.2 Step 1: Evidence Loading + Dashboard

```python
# AgentConfig (imports already done in Cell 0.6)
agent_config = AgentConfig(
    batch_model='gpt-4o',
    report_model='gpt-4o',
    temperature=0.3,
    language='vi',
)

loader  = EvidenceLoader()
builder = EvidenceBuilder(agent_config)

case_dir = f'{XAI_DIR}/case_studies/case_{sid}'  # Phase 6 case study dir
if not os.path.isdir(case_dir):
    case_dir = XAI_DIR  # fallback

evidence       = loader.load(sid, XAI_DIR, sid)
built_evidence = builder.build(evidence)

# Evidence Completeness Dashboard
METHODS = ['gradcam', 'attention', 'cross_attention', 'shap', 'lime']
METHOD_LABELS = ['Grad-CAM', 'Attention', 'Cross-Attn', 'SHAP', 'LIME']
completeness = [1.0 if evidence.get(m) else 0.0 for m in METHODS]
missing = built_evidence.get('missing_summary', 'None')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Bar chart
colors_comp = ['#2ECC71' if c > 0 else '#E74C3C' for c in completeness]
axes[0].bar(METHOD_LABELS, completeness, color=colors_comp, alpha=0.9, width=0.6)
axes[0].set_ylim(0, 1.3); axes[0].set_ylabel('Available (1=Yes, 0=No)')
axes[0].set_title('Evidence Completeness Dashboard', fontsize=12, fontweight='bold')
for i, (m, c) in enumerate(zip(METHOD_LABELS, completeness)):
    axes[0].text(i, c + 0.05, '✓' if c > 0 else '✗', ha='center', fontsize=16,
                 color='green' if c > 0 else 'red')
axes[0].grid(axis='y', alpha=0.3)

# Text summary
total_pct = sum(completeness) / len(completeness) * 100
confidence_lvl = 'High' if total_pct >= 80 else ('Medium' if total_pct >= 60 else 'Low')
summary_text = (
    f'Sample: {sid}\n'
    f'Evidence completeness: {total_pct:.0f}%\n'
    f'Agent confidence: {confidence_lvl}\n'
    f'Missing: {missing or "None"}\n\n'
    + '\n'.join([f'  {"✓" if c>0 else "✗"} {m}' for m, c in zip(METHOD_LABELS, completeness)])
)
axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
             fontsize=11, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F0F4FF', alpha=0.9))
axes[1].axis('off'); axes[1].set_title('Evidence Summary', fontsize=12, fontweight='bold')

plt.suptitle(f'AI Agent — Evidence Loading  (Sample {slot})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/agent_evidence_dashboard.png', dpi=DEFAULT_DPI)
plt.show()
```

### 12.3 Step 2: Reasoning Graph Visualization

```python
if build_reasoning_graph is not None:
    reasoning = build_reasoning_graph(
        predictions=pred_result['predictions'],
        evidence=evidence,
        review_text=samples[slot]['data']['text'],
    )
else:
    # Fallback: dùng reasoning data từ evidence nếu EvidenceBuilder đã build sẵn
    reasoning = built_evidence.get('reasoning_graph', {'targets': {}, 'agreement_matrix': {}})

# Bảng Reasoning Graph
rows_rg = []
for t, tinfo in reasoning.get('targets', {}).items():
    rows_rg.append({
        'Target': DISPLAY_NAMES[TARGET_NAMES.index(t)] if t in TARGET_NAMES else t,
        'Strength': tinfo.get('evidence_strength', '?'),
        'Supporting': len(tinfo.get('supporting_evidence', [])),
        'Contradicting': len(tinfo.get('contradicting_evidence', [])),
        'Missing': len(tinfo.get('missing_evidence', [])),
        'Hint': tinfo.get('interpretation_hint', '')[:50],
    })
df_rg = pd.DataFrame(rows_rg)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
tbl = ax.table(
    cellText=df_rg.values, colLabels=df_rg.columns,
    cellLoc='left', loc='center',
    colWidths=[0.22, 0.10, 0.10, 0.12, 0.08, 0.38],
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1, 1.8)
# Color-code by strength
for i, row in enumerate(rows_rg):
    color = {'high': '#D5F5E3', 'medium': '#FEF9E7', 'low': '#FADBD8'}.get(row['Strength'], 'white')
    for j in range(len(df_rg.columns)):
        tbl[i+1, j].set_facecolor(color)
plt.title(f'Reasoning Graph — Evidence per Target  (Sample {slot})', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_{slot}/agent_reasoning_graph.png', dpi=DEFAULT_DPI)
plt.show()

# Agreement matrix
if 'agreement_matrix' in reasoning:
    ag_mat = pd.DataFrame(reasoning['agreement_matrix'])
    print('\nAgreement Matrix (XAI methods × targets):')
    display(ag_mat.style.background_gradient(cmap='Greens'))
```

### 12.4 Step 3: GPT-4o Call + Stats

```python
api_key = os.environ.get('OPENAI_API_KEY', '')
if not api_key:
    print('[WARNING] OPENAI_API_KEY not set. AI Agent will be SKIPPED.')
    agent_output = None
    agent_ok = False
else:
    prompt_builder = PromptBuilder()
    openai_client  = OpenAIClient(api_key=api_key)

    messages = prompt_builder.build(
        sample_id=sid,
        review_text=samples[slot]['data']['text'],
        predictions=pred_result['predictions'],
        ground_truth=pred_result['ground_truth'],
        built_evidence=built_evidence,
        reasoning_graph=reasoning,
        language='vi',
    )

    # Prompt statistics
    total_tokens_est = sum(len(m.get('content', '').split()) * 1.3 for m in messages)
    print(f'Prompt: {len(messages)} messages, ~{total_tokens_est:.0f} tokens estimated')
    print('Calling GPT-4o...')

    t0_api = time.time()
    try:
        agent_output = openai_client.generate_json(
            messages=messages, model='gpt-4o', temperature=0.3
        )
        elapsed_api = time.time() - t0_api
        print(f'GPT-4o responded in {elapsed_api:.1f}s')
        print(f'Confidence: {agent_output.get("confidence", "?")}')
        agent_ok = True
    except Exception as e:
        print(f'[WARNING] GPT-4o call failed: {e}')
        agent_output = None
        agent_ok = False
```

### 12.5 Step 4: Output Validation Dashboard

```python
if agent_output and agent_ok:
    validator = OutputValidator()
    val_result = validator.validate(agent_output, pred_result['predictions'])

    val_checks = val_result.get('checks', {})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Validation checkmarks
    check_names = list(val_checks.keys()) if val_checks else ['schema', 'all_targets', 'score_levels', 'shap_grounding']
    check_vals  = [val_checks.get(k, True) for k in check_names]
    check_colors = ['#2ECC71' if v else '#E74C3C' for v in check_vals]
    axes[0].barh(check_names, [1]*len(check_names), color=check_colors, alpha=0.85)
    for i, (cn, cv) in enumerate(zip(check_names, check_vals)):
        axes[0].text(0.5, i, '✓ PASS' if cv else '✗ FAIL', ha='center', va='center',
                     fontsize=11, color='white', fontweight='bold')
    axes[0].set_xlim(0, 1.5); axes[0].axis('off')
    axes[0].set_title(f'Validation Checks — {"PASSED ✓" if val_result.get("passed") else "FAILED ✗"}',
                      fontsize=12, fontweight='bold',
                      color='#27AE60' if val_result.get('passed') else '#E74C3C')

    # Output statistics
    stats_text = (
        f'Confidence: {agent_output.get("confidence", "?")}\n'
        f'Language: {agent_output.get("language", "?")}\n'
        f'Scores present: {len(agent_output.get("scores", {}))}/5\n'
        f'Limitations listed: {len(agent_output.get("limitations", []))}\n'
        f'Recommendations: {len(agent_output.get("recommendations", []))}\n'
        f'Validation: {"PASSED ✓" if val_result.get("passed") else "FAILED ✗"}\n'
        + (f'Issues: {", ".join(val_result.get("issues", []))}' if not val_result.get('passed') else '')
    )
    axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                 fontsize=11, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#F0F4FF', alpha=0.9))
    axes[1].axis('off'); axes[1].set_title('Output Statistics', fontsize=12, fontweight='bold')

    plt.suptitle(f'AI Agent — Output Validation  (Sample {slot})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DEMO_OUT}/sample_{slot}/agent_validation.png', dpi=DEFAULT_DPI)
    plt.show()
```

### 12.6 Step 5: Generate + Display Report

```python
if agent_output and agent_ok:
    agent_dir = f'{DEMO_OUT}/sample_{slot}/agent'
    os.makedirs(agent_dir, exist_ok=True)

    report_paths = save_sample_report(
        output=agent_output,
        output_dir=agent_dir,
        review_text=samples[slot]['data']['text'],
        predictions=pred_result['predictions'],
        ground_truth=pred_result['ground_truth'],
    )
    # report_paths: {'json': ..., 'markdown': ...}

    # Load và hiển thị report
    with open(report_paths['markdown'], 'r', encoding='utf-8') as f:
        report_md = f.read()

    # Tách Customer View và Technical View
    if '## Phần B' in report_md or '## Part B' in report_md:
        split_key = '## Phần B' if '## Phần B' in report_md else '## Part B'
        part_a = report_md.split(split_key)[0]
        part_b = split_key + report_md.split(split_key)[1]
    else:
        part_a, part_b = report_md, ''

    from IPython.display import Markdown
    print('\n' + '='*65)
    print('PHẦN A — CUSTOMER VIEW (Vietnamese, non-technical)')
    print('='*65)
    ipy_display(Markdown(part_a))

    print('\n' + '='*65)
    print('PHẦN B — TECHNICAL VIEW (Full XAI evidence)')
    print('='*65)
    ipy_display(Markdown(part_b))

    # Display Agent Summary (from JSON)
    with open(report_paths['json'], 'r', encoding='utf-8') as f:
        agent_json = json.load(f)
    print(f'\nAgent Summary: {agent_json.get("summary", "")}')
    print(f'Confidence: {agent_json.get("confidence", "?")}')
    for rec in agent_json.get('recommendations', []):
        print(f'  → {rec}')
```

---

## 13. Sample XAI Dashboard (Tổng hợp per-sample)

Sau khi hoàn thành tất cả 8 bước, tạo **1 dashboard tổng hợp** cho mỗi sample:

```python
def create_sample_dashboard(slot, sid, pred_result, demo_out):
    """Tạo 1 figure tổng hợp 8 panels cho 1 sample."""
    fig = plt.figure(figsize=(24, 18))
    gs  = gridspec.GridSpec(3, 4, fig, wspace=0.25, hspace=0.30)

    artifact_map = {
        (0,0): (f'{demo_out}/sample_{slot}/overview_images.png',     f'① Input Image(s)\nSample {slot}: {sid}'),
        (0,1): (f'{demo_out}/sample_{slot}/prediction_chart.png',    '② Prediction vs GT'),
        (0,2): (f'{demo_out}/sample_{slot}/gradcam_3panel.png',      '③ Grad-CAM (Overall)'),
        (0,3): (f'{demo_out}/sample_{slot}/attention_top_words.png', '④ PhoBERT Attention'),
        (1,0): (f'{demo_out}/sample_{slot}/cross_attn_t2i_overlay.png', '⑤ Cross-Attention T2I'),
        (1,1): (f'{demo_out}/sample_{slot}/shap_modality.png',       '⑥ SHAP Modality'),
        (1,2): (f'{demo_out}/sample_{slot}/lime_combined.png',       '⑦ LIME (Text + Image)'),
        (1,3): (f'{demo_out}/sample_{slot}/agent_evidence_dashboard.png', '⑧ AI Agent Evidence'),
        (2,0): (f'{demo_out}/sample_{slot}/agent_reasoning_graph.png', '⑨ Reasoning Graph'),
        (2,1): (f'{demo_out}/sample_{slot}/agent_validation.png',    '⑩ Validation'),
        (2,2): None,  # Customer View text
        (2,3): None,  # Technical View text
    }

    for (row, col), entry in artifact_map.items():
        ax = fig.add_subplot(gs[row, col])
        if entry is None:
            ax.axis('off')
            continue
        path, title = entry
        if os.path.exists(path):
            ax.imshow(np.array(Image.open(path)))
        else:
            ax.text(0.5, 0.5, f'{title}\n(Not available)', ha='center', va='center',
                    fontsize=9, color='gray', transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    # Customer View + Technical View text panels
    if 'agent_output' in dir() and agent_output:
        ax_cv = fig.add_subplot(gs[2, 2])
        ax_cv.text(0.05, 0.95, f'Customer View:\n{agent_output.get("summary", "N/A")[:200]}...',
                   transform=ax_cv.transAxes, fontsize=8, va='top', wrap=True,
                   bbox=dict(boxstyle='round', facecolor='#E8F8E8', alpha=0.9))
        ax_cv.axis('off'); ax_cv.set_title('⑪ Customer View', fontsize=10, fontweight='bold')

        ax_tv = fig.add_subplot(gs[2, 3])
        mc = agent_output.get('modality_contribution', {})
        tv_text = (f'Technical View:\n'
                   f'Text-origin: {mc.get("text_origin_pct", "?")}%\n'
                   f'Image-origin: {mc.get("image_origin_pct", "?")}%\n'
                   f'Confidence: {agent_output.get("confidence", "?")}')
        ax_tv.text(0.05, 0.95, tv_text, transform=ax_tv.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='#EEF2FF', alpha=0.9))
        ax_tv.axis('off'); ax_tv.set_title('⑫ Technical View', fontsize=10, fontweight='bold')

    plt.suptitle(f'XAI Dashboard — Sample {slot} ({sid})\nCase Type: {case_types.get(slot, "?")}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(f'{demo_out}/sample_{slot}/xai_dashboard.png', dpi=DEFAULT_DPI, bbox_inches='tight')
    plt.show()
    print(f'Dashboard saved: {demo_out}/sample_{slot}/xai_dashboard.png')

create_sample_dashboard(slot, sid, pred_result, DEMO_OUT)
```

---

## 14. Cấu Trúc Thư Mục Output

```
demo_e2e/
├── manifest.json                      ← Metadata toàn demo
├── summary_comparison.png             ← Cross-sample comparison chart
├── demo_summary.json                  ← Tóm tắt toàn bộ
│
├── sample_A/
│   ├── overview_images.png            ← Input images display
│   ├── prediction.json                ← Prediction + GT + errors
│   ├── prediction_chart.png           ← Bar chart Pred vs GT
│   ├── gradcam_3panel.png             ← Original + Raw + Overlay
│   ├── gradcam_overlay_overall.png    ← Overlay only (save riêng)
│   ├── attention_top_words.png        ← Top-K word bar chart
│   ├── attention_cls_heatmap.png      ← CLS sub-matrix heatmap
│   ├── attention_highlighted.html     ← Highlighted text
│   ├── cross_attn_t2i_heatmap.png    ← T2I heatmap
│   ├── cross_attn_t2i_overlay.png    ← Best token overlay
│   ├── cross_attn_i2t_heatmap.png    ← I2T heatmap
│   ├── shap_modality.png             ← Pie + stacked bar
│   ├── shap_waterfall_overall.png    ← Waterfall chart
│   ├── lime_combined.png             ← 4-panel LIME figure
│   ├── agent_evidence_dashboard.png  ← Completeness dashboard
│   ├── agent_reasoning_graph.png     ← Reasoning graph table
│   ├── agent_validation.png          ← Validation dashboard
│   ├── xai_dashboard.png             ← Full 12-panel summary
│   └── agent/
│       ├── report.json               ← Structured agent output
│       ├── report.md                 ← Markdown report
│       └── validation_result.json    ← Validation details
│
├── sample_B/                         ← Same structure
├── sample_C/                         ← Same structure
│
└── errors/
    ├── sample_A_errors.json
    ├── sample_B_errors.json
    └── sample_C_errors.json
```

### manifest.json format

```json
{
  "demo_version": "2.0",
  "created_at": "2026-06-30T...",
  "exp_id": "EXP_060A_bestsequential_full_configuration",
  "samples": {
    "A": {
      "sample_id": "sample_XXXX",
      "case_type": "correct",
      "completeness": 0.95,
      "reused_from_phase6": true,
      "regenerated_artifacts": [],
      "steps_completed": ["prediction", "gradcam", "attention", "cross_attention", "shap", "lime", "agent"],
      "steps_failed": []
    },
    "B": { "...": "..." },
    "C": { "...": "..." }
  }
}
```

---

## 15. Cross-Sample Comparison (Cell Group 6)

### 15.1 Prediction Comparison Chart

```python
fig, axes = plt.subplots(1, 5, figsize=(22, 6), sharey=True)
colors_by_slot = {'A': '#2196F3', 'B': '#FF5722', 'C': '#4CAF50'}

for i, (target, disp) in enumerate(zip(TARGET_NAMES, DISPLAY_NAMES)):
    ax = axes[i]
    for j, slot in enumerate(['A', 'B', 'C']):
        if slot in all_pred_results:
            p = all_pred_results[slot]['predictions'][i]
            g = all_pred_results[slot]['ground_truth'][i]
            x_base = j * 2
            ax.bar(x_base,     g, 0.8, color=colors_by_slot[slot], alpha=0.4)
            ax.bar(x_base+0.8, p, 0.8, color=colors_by_slot[slot], alpha=0.9)
            ax.text(x_base+0.4, max(g, p)+0.1, f'Δ{abs(g-p):.1f}', ha='center', fontsize=7, color='gray')
    ax.set_title(disp, fontsize=10, fontweight='bold')
    ax.set_xticks([0.8, 2.8, 4.8]); ax.set_xticklabels(['A', 'B', 'C'])
    ax.set_ylim(0, 11); ax.grid(axis='y', alpha=0.3)
    if i == 0:
        ax.set_ylabel('Score (1–10)')
        for slot, color in colors_by_slot.items():
            ax.bar([], [], color=color, label=f'Sample {slot}')
        ax.legend(loc='upper left', fontsize=8)

plt.suptitle('Cross-Sample Comparison: Prediction vs Ground Truth\n(light = GT, dark = Prediction)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/summary_prediction_comparison.png', dpi=DEFAULT_DPI)
plt.show()
```

### 15.2 XAI Method Summary Table

Sau khi chạy xong tất cả 3 samples, tổng hợp kết quả:

```python
summary_rows = []
for slot in ['A', 'B', 'C']:
    sid_s  = selected_samples[slot]
    arts   = check_sample_artifacts(sid_s, XAI_DIR)
    shap_c = all_shap_contribs.get(slot, {}).get('overall', {})
    preds  = all_pred_results.get(slot, {})
    summary_rows.append({
        'Sample': f'Sample {slot}',
        'Case Type': case_types.get(slot, '?'),
        'Mean MAE': f'{preds.get("mean_mae", float("nan")):.3f}',
        'Grad-CAM': '✓' if arts.get('gradcam') else '✗',
        'Attention': '✓' if arts.get('attention') else '✗',
        'Cross-Attn': '✓' if arts.get('cross_attention') else '✗',
        'SHAP': '✓' if arts.get('shap') else '✗',
        'LIME': '✓' if arts.get('lime') else '✗',
        'Text%': f'{shap_c.get("text_pct", "?"):.0f}%' if isinstance(shap_c.get("text_pct"), (int, float)) else '?',
        'Image%': f'{shap_c.get("image_pct", "?"):.0f}%' if isinstance(shap_c.get("image_pct"), (int, float)) else '?',
        'AI Agent': '✓' if all_agent_validated.get(slot) else '✗',
    })

df_summary = pd.DataFrame(summary_rows)
display(df_summary.style.applymap(
    lambda v: 'color: green; font-weight: bold' if v == '✓' else ('color: red' if v == '✗' else '')
).set_caption('Cross-Sample XAI Coverage Summary'))
```

### 15.3 SHAP Modality Profile Comparison

```python
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(5)
w = 0.25

for j, slot in enumerate(['A', 'B', 'C']):
    text_pcts = []
    for f in FACTOR_NAMES:
        contrib = all_shap_contribs.get(slot, {}).get(f, {})
        text_pcts.append(contrib.get('text_pct', 0))
    ax.bar(x + j * w, text_pcts, w,
           label=f'Sample {slot}',
           color=colors_by_slot[slot], alpha=0.8)

ax.axhline(50, color='black', lw=1.5, ls='--', alpha=0.5, label='50% balance')
ax.set_xticks(x + w); ax.set_xticklabels(DISPLAY_NAMES, rotation=20, ha='right')
ax.set_ylabel('Text-origin Contribution (%)'); ax.set_ylim(0, 110)
ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_title('SHAP Modality Profile Comparison — 3 Samples × 5 Targets', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/summary_shap_comparison.png', dpi=DEFAULT_DPI)
plt.show()
```

### 15.4 Conclusion Markdown Cell

```markdown
## Kết Luận Demo

| Tiêu chí | Sample A | Sample B | Sample C |
|---|---|---|---|
| **Case Type** | Correct/Agreement | High Error/Conflict | Multimodal-rich |
| **Prediction Quality** | MAE < 0.5 | MAE > 1.0 | Varies |
| **XAI Consistency** | High (methods agree) | Low (methods conflict) | Medium |
| **Dominant Modality** | Balanced | One-sided | SHAP reveals split |
| **AI Agent Confidence** | High | Low–Medium | Medium–High |
| **Cross-Attention** | Clear alignment | Misalignment visible | Strong alignment |

**Key Insights từ Demo:**

1. **CrossAttentionFusion hoạt động:** Cross-Attention alignment (Section 9) cho thấy model học được
   kết nối text↔image có ý nghĩa ngữ nghĩa.

2. **Giới hạn Grad-CAM được giải quyết:** Grad-CAM hạn chế do shared encoder, nhưng SHAP (Section 10)
   cung cấp target-specific attribution chính xác hơn.

3. **XAI methods bổ sung cho nhau:** Không phương pháp nào toàn diện một mình. Agreement = converging
   evidence. Disagreement = tín hiệu cần điều tra (thấy rõ nhất ở Sample B).

4. **AI Agent là cầu nối:** GPT-4o chuyển đổi complex XAI artifacts thành language giải thích
   cho end-users (Customer View) và cho researchers (Technical View).
```

---

## 16. Xử Lý Lỗi (Error Handling)

### 16.1 Nguyên tắc: Never Stop

```python
DEMO_ERRORS = {}  # Global error accumulator — reset sau mỗi sample

def run_safe(step_name, fn, *args, fallback=None, **kwargs):
    """Chạy một bước XAI an toàn. Không bao giờ raise exception."""
    try:
        result = fn(*args, **kwargs)
        print(f'  ✓ {step_name}')
        return result, True
    except Exception as e:
        tb = traceback.format_exc()
        msg = f'{type(e).__name__}: {e}'
        print(f'  ⚠ [WARNING] {step_name}: {msg}')
        DEMO_ERRORS[step_name] = {'error': msg, 'traceback': tb[:500]}
        return fallback, False

# Ví dụ sử dụng:
cam_raw, ok_gradcam = run_safe(
    f'GradCAM_Sample_{slot}',
    compute_gradcam_for_image,
    model, samples[slot]['data'], 4, 0, target_layer, device,
    fallback=None,
)
```

### 16.2 Placeholder cho Visualization

```python
def show_placeholder(ax, message='Not available'):
    ax.set_facecolor('#F8F8F8')
    ax.text(0.5, 0.5, f'⚠\n{message}', ha='center', va='center',
            fontsize=11, color='#AAAAAA', transform=ax.transAxes)
    ax.axis('off')
```

### 16.3 Error Log per Sample

```python
# Sau khi chạy xong sample, lưu error log
error_path = f'{DEMO_OUT}/errors/sample_{slot}_errors.json'
os.makedirs(os.path.dirname(error_path), exist_ok=True)
with open(error_path, 'w', encoding='utf-8') as f:
    json.dump(DEMO_ERRORS, f, ensure_ascii=False, indent=2)

if DEMO_ERRORS:
    n = len(DEMO_ERRORS)
    print(f'\n[Summary] {n} warning(s) for Sample {slot}:')
    for step, info in DEMO_ERRORS.items():
        print(f'  • {step}: {info["error"]}')
else:
    print(f'\n[Summary] Sample {slot}: All steps completed successfully.')

DEMO_ERRORS = {}  # Reset cho sample tiếp theo
```

### 16.4 Bảng tình huống lỗi đặc biệt

| Tình huống | Xử lý |
|---|---|
| `best_model_train_fusion.pth` thiếu | **RAISE** — lỗi fatal, không thể tiếp tục |
| Dataset CSV thiếu | **RAISE** — không có data thì không demo được |
| `load_single_sample()` fail cho 1 sample | **WARN** + skip sample đó, tiếp tục 2 còn lại |
| Grad-CAM dead gradient (max−min < 0.01) | **WARN** + show original image + annotation |
| PhoBERT attention returns None | **CHECK** eager attention → retry once, rồi skip |
| SHAP background thiếu | **SKIP** SHAP + log warning |
| LIME timeout (>15 phút) | **REDUCE** num_samples → 200/300, retry |
| OpenAI API key thiếu | **SKIP** AI Agent + hiển thị warning prominently |
| GPT-4o rate limit | **RETRY** exponential backoff (max 3 lần) |
| Agent validation fails | **DISPLAY** raw JSON + note "Validation failed" |
| File PNG corrupt/empty | **REGENERATE** artifact |

---

## 17. Validation Checklist

### 17.1 Setup

- [ ] Google Drive mount thành công, `EXP_DIR` trỏ đúng.
- [ ] `best_model_train_fusion.pth` tồn tại trong `EXP_DIR`.
- [ ] Tất cả imports OK — không `ModuleNotFoundError`.
- [ ] `device = cuda` (kiểm tra GPU khả dụng).
- [ ] `model.training == False` (eval mode).
- [ ] Eager attention active: `model.text_model.encoder.config._attn_implementation == 'eager'`.
- [ ] 3 sample IDs được in ra — đủ 3 loại case type khác nhau.
- [ ] OpenAI API key set (cảnh báo nếu không có — AI Agent section sẽ skip).

### 17.2 Per-Sample (lặp lại cho A, B, C)

- [ ] Review text hiển thị đầy đủ (≥ 30 từ, tiếng Việt có nghĩa).
- [ ] Ít nhất 2 ảnh thực hiển thị (không phải all black images).
- [ ] Prediction table: đủ 5 rows, không có NaN.
- [ ] Bar chart Pred vs GT render đúng.
- [ ] **Grad-CAM:** Grad-CAM limitation disclaimer hiển thị TRƯỚC figure.
- [ ] **Grad-CAM:** 3-panel figure (Original + Raw Heatmap + Overlay) hiển thị.
- [ ] **Grad-CAM:** Raw heatmap không constant (max ≠ min, gradient có).
- [ ] **PhoBERT:** Highlighted text HTML hiển thị màu sắc gradient.
- [ ] **PhoBERT:** Top-K word bar chart có từ có nghĩa (không phải special tokens).
- [ ] **PhoBERT:** CLS sub-matrix heatmap hiển thị (đủ ≥ 5 token rows/cols).
- [ ] **PhoBERT:** "Attention ≠ Causality" disclaimer hiển thị.
- [ ] **Cross-Attention:** T2I heatmap hiển thị (≥ 5 token rows).
- [ ] **Cross-Attention:** Overlay (best token → patches) hiển thị gradient.
- [ ] **Cross-Attention:** I2T heatmap hiển thị.
- [ ] **Cross-Attention:** Top-K pairs table có ít nhất 5 rows.
- [ ] **SHAP:** Pie chart tổng text_pct + image_pct ≈ 100%.
- [ ] **SHAP:** Stacked bar hiển thị đủ 5 targets.
- [ ] **SHAP:** Waterfall chart có cả positive (xanh) và negative (đỏ) dims.
- [ ] **SHAP:** "text-origin ≠ pure text" disclaimer hiển thị.
- [ ] **LIME:** 4-panel combined figure hiển thị đầy đủ.
- [ ] **LIME:** Superpixel overlay có màu boundary rõ ràng.
- [ ] **LIME:** "Local explanation only" disclaimer hiển thị.
- [ ] **AI Agent:** Evidence completeness dashboard hiển thị.
- [ ] **AI Agent:** Reasoning graph table có đủ 5 target rows.
- [ ] **AI Agent:** (nếu có API key) GPT-4o response nhận được, confidence hiển thị.
- [ ] **AI Agent:** Validation dashboard hiển thị (pass hoặc fail với lý do).
- [ ] **AI Agent:** Report Part A (Customer View) hiển thị.
- [ ] **AI Agent:** Report Part B (Technical View) hiển thị.
- [ ] **Dashboard:** XAI 12-panel dashboard render và save thành công.

### 17.3 Cross-Sample

- [ ] Prediction comparison: 5 charts, mỗi chart có 3 sample bars.
- [ ] XAI summary table: 3 rows, đủ cột.
- [ ] SHAP modality comparison: 3 bar groups per target.
- [ ] `manifest.json` được tạo tại `{DEMO_OUT}/manifest.json`.

### 17.4 Presentation Quality

- [ ] Tất cả figures có title rõ ràng (kể tên sample + mô tả).
- [ ] Tất cả bar charts có axis labels (xlabel, ylabel).
- [ ] Tất cả Markdown disclaimer cells có nội dung đầy đủ.
- [ ] Không có cell nào hiển thị traceback (tất cả wrapped trong `run_safe`).
- [ ] Thời gian chạy ước tính < 35 phút trên GPU T4/A100.
- [ ] File `demo_e2e/` được save trên Google Drive (persist sau khi Colab reset).

---

## 18. Phụ Lục Kỹ Thuật

### A. Technical Notes Quan Trọng

**A.1 TimmProcessor:**
`get_image_processor` không phải `AutoImageProcessor`. Image processor cho Swin-B phải được tạo qua `timm.data.create_transform()` với config của model. Xem Cell 0.7 cho pattern chính xác.

**A.2 Multi-image padding:**
Mỗi sample có tối đa `DEFAULT_MAX_IMAGES=4`. Slots thừa được padding bằng black tensor. Khi hiển thị ảnh:
```python
real_imgs = samples[slot]['data']['loaded_images'][:samples[slot]['data']['num_real_images']]
```
KHÔNG hiển thị padding images.

**A.3 Eager attention — một lần duy nhất:**
`enable_eager_attention()` được gọi bên trong `load_model()` trong codebase xai-v3.
**KHÔNG** gọi lại thủ công sau đó.

**A.4 FUSED_DIM=1024, split tại dim 512:**
```
fused[0:512]    = text-origin   (text Q attended to image K/V → mixed)
fused[512:1024] = image-origin  (image Q attended to text K/V → mixed)
```
"text-origin" và "image-origin" là tên kỹ thuật, không phải "pure text" hay "pure image".

**A.5 BHWC format của target layer:**
`model.image_model.encoder.norm` output: `[B, H, W, C]` (BHWC), KHÔNG phải `[B, C, H, W]`.
Hàm `normalize_feature_map_to_bchw()` trong gradcam_explainer.py xử lý conversion tự động.

**A.6 LIME pseudo-classification label index:**
LIME API dùng `label=1` = high score class. `as_list(label=1)` trả về list `(word, weight)`.
Weight dương = từ đó tăng score prediction; weight âm = giảm score.

**A.7 Target index mapping:**
```python
0: 'food_score'          ← FACTOR_NAMES[0] = 'food'
1: 'price_score'         ← FACTOR_NAMES[1] = 'price'
2: 'atmosphere_score'    ← FACTOR_NAMES[2] = 'atmos'
3: 'service_score'       ← FACTOR_NAMES[3] = 'service'
4: 'overall_satisfaction'← FACTOR_NAMES[4] = 'overall'  ← dùng trong demo
```

### B. Catalog Toàn Bộ Phase 2-6 Artifacts

```
EXP_DIR/xai/
│
├── gradcam/{sid}/
│   ├── gradcam_img{i}_{factor}.png         ← Overlay per image per target
│   ├── cam_{factor}_img{i}.npy             ← Raw CAM array [H,W] (nếu có)
│   ├── gradient_diagnostics.png            ← Cosine similarity matrix 5×5
│   └── metadata.json                       ← exp_id, timestamp, params
│
├── attention/{sid}/
│   ├── raw_attention.npz                   ← {'attentions':[12,12,L,L], 'tokens', 'seq_len'}
│   ├── cls_importance_word_bar.png         ← Word-level importance bar
│   ├── cls_importance_subword_bar.png      ← Subword-level bar
│   ├── topk_tokens.json                    ← Top-20 tokens list
│   └── word_importance.json               ← {word: score} merged
│
├── cross_attention/{sid}/
│   ├── cross_attention_raw.npz             ← {'t2i_attn':[T,49], 'i2t_attn':[49,T], 'tokens'}
│   ├── cross_attention_summary.json        ← Stats, top pairs summary
│   ├── token_patch_topk.json               ← Top-K token-patch pairs
│   ├── topk_token_patch_heatmap.png        ← Top-K heatmap vis
│   ├── token_patch_bipartite_graph.png     ← Bipartite graph vis
│   ├── patch_importance.png                ← Aggregated patch importance
│   ├── top_tokens_patch_overlay_grid.png   ← Grid overlay (HIGHLIGHT!)
│   └── token_overlay_{token}.png           ← Per-token patch overlay
│
├── shap/{sid}/
│   ├── raw_shap_values.npz                 ← {'overall': [1024], 'food': [1024], ...}
│   ├── shap_modality_contribution.json     ← {factor: {text_pct, image_pct, text_abs, ...}}
│   └── shap_modality_contribution.png      ← Phase 4 chart (có thể reuse)
│
├── shap/raw/
│   └── background_fused.pt                 ← [100, 1024] background embeddings
│
├── lime/{sid}/
│   ├── {sid}_lime_text_{factor}_weights.json    ← {word: weight} per factor
│   ├── {sid}_lime_image_{factor}_positive.png   ← Positive superpixels
│   ├── {sid}_lime_image_{factor}_negative.png   ← Negative superpixels
│   ├── {sid}_lime_image_{factor}_combined.png   ← Combined visualization
│   └── metadata.json
│
└── case_studies/
    ├── sample_manifest.csv                 ← Phase 6 selected samples
    ├── case_{sid}/
    │   ├── metadata.json                   ← Case type, predictions, errors
    │   ├── analysis.md                     ← Phase 6 auto-analysis
    │   ├── combined_figure_target{i}_{factor}.png ← Phase 6 combined fig
    │   └── combined_cross_attention_figure.png
    └── ...

EXP_DIR/agent_outputs/
└── {sid}/
    ├── {sid}_report.json                   ← Structured agent output
    └── {sid}_report.md                     ← Markdown report
```

### C. Pre-Implementation Checklist

Trước khi implement notebook, verify trên Google Drive:

```
1. ✓ EXP_DIR/best_model_train_fusion.pth     ← Model checkpoint
2. ✓ EXP_DIR/xai/                            ← Ít nhất 1/5 subdirs
3. ✓ data/text/test.csv (hoặc val.csv)       ← Dataset
4. ✓ data/image/                             ← Image files
5. ? EXP_DIR/xai/shap/raw/background_fused.pt  ← Optional nhưng nên có
6. ? EXP_DIR/xai/case_studies/               ← Phase 6 results (nếu có)
7. ✓ Colab Secret 'OPENAI_API_KEY'           ← Cho AI Agent
```

---

*Proposal v2.0 — Production-ready implementation guide.*
*Notebook `Demo_End_to_End_XAI_AI_Agent.ipynb` có thể được implement ngay từ proposal này.*
