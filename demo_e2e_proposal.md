# Proposal: Demo_End_to_End_XAI_AI_Agent.ipynb

**Tên file notebook đích:** `Demo_End_to_End_XAI_AI_Agent.ipynb`  
**Phiên bản proposal:** 1.0  
**Ngày:** 2026-06-30  
**Dự án:** An Explainable Multi-modal Deep Learning System for Restaurant Review Quality Assessment using Image and Text Data  

---

## 1. Mục Đích (Purpose)

Notebook này được thiết kế để trình bày toàn bộ pipeline của hệ thống — từ đầu vào thô (văn bản đánh giá + ảnh nhà hàng) đến kết quả giải thích đa tầng (Grad-CAM, PhoBERT Attention, Cross-Attention, SHAP, LIME) và cuối cùng là báo cáo ngôn ngữ tự nhiên do AI Agent tổng hợp — theo cách dễ hiểu cho **giảng viên hội đồng phản biện**.

Notebook **không** dùng để benchmark, không dùng để train lại, và không tạo ra pipeline mới. Toàn bộ các module XAI đã được implement xong; notebook này chỉ orchestrate lại chúng theo một trình tự trình bày logic.

**Đối tượng:** Giảng viên hội đồng luận văn, không nhất thiết có nền tảng kỹ thuật sâu về Deep Learning.

**Kết quả chính cần đạt:**
- Người xem hiểu được mô hình đưa ra dự đoán bằng cách nào.
- Người xem thấy được từng phương pháp XAI đóng góp góc nhìn gì khác nhau.
- AI Agent tổng hợp mọi bằng chứng thành báo cáo đọc được.

---

## 2. Phạm Vi Demo (Demo Scope)

### 2.1 Workflow tổng thể

```
[Input: Text + Images]
        ↓
[CrossAttentionFusion Model → Prediction (5 scores)]
        ↓
[Prediction Visualization: Table + Bar Chart + Error Analysis]
        ↓
[Grad-CAM: Overall Satisfaction heatmap overlay]
        ↓
[PhoBERT Attention: Word-level importance]
        ↓
[Cross-Attention: Bidirectional Token ↔ Patch]
        ↓
[SHAP: Modality contribution (text-origin vs image-origin)]
        ↓
[LIME: Local text + image explanation]
        ↓
[AI Agent: EvidenceLoader → EvidenceBuilder → ReasoningGraph → GPT-4o → OutputValidator → ReportGenerator]
        ↓
[Final Output: Customer View (Markdown) + Technical View (JSON)]
```

### 2.2 Số lượng mẫu

Đúng **3 mẫu**. Mỗi mẫu đại diện cho một loại case khác nhau (xem Mục 4).

### 2.3 Phạm vi XAI

| Module | Scope trong Demo |
|---|---|
| Prediction | Tất cả 5 targets |
| Grad-CAM | Chỉ `overall_satisfaction` (xem lý do tại Mục 7) |
| PhoBERT Attention | Word-level, merged subwords, top-K |
| Cross-Attention | Bidirectional T2I + I2T, top-K pairs, overlay |
| SHAP | Modality contribution + per-target chart |
| LIME Text | Top important words |
| LIME Image | Superpixel explanation |
| AI Agent | Full pipeline: Reasoning Graph + Customer View + Technical View |

### 2.4 Những gì KHÔNG demo

- Train / fine-tune lại mô hình.
- Toàn bộ tập test (chỉ 3 mẫu đại diện).
- So sánh nhiều model khác nhau.
- Ablation study.

---

## 3. Kiến Trúc Notebook (Notebook Architecture)

### 3.1 Môi trường thực thi

Notebook thiết kế để chạy trên **Google Colab** với GPU T4/A100. Có thể chạy local nếu đủ VRAM (≥8GB). Python ≥ 3.9.

### 3.2 Kiến trúc model tham chiếu

```
Text branch:    vinai/phobert-base-v2  →  [B, T, 768]  →  text_proj Linear(768→512)
Image branch:   swin_base_patch4_window7_224  →  [B, 49, 1024]  →  image_proj Linear(1024→512)

CrossAttentionFusion:
  cross_attn_t2i: MultiheadAttention(512, 8, batch_first=True)   [B,T,512] ↔ [B,49,512]
  cross_attn_i2t: MultiheadAttention(512, 8, batch_first=True)
  masked mean pool → concat → [B, 1024]   (FUSED_DIM=1024)
  head: Linear(1024→512) → ReLU → Dropout → Linear(512→256) → ReLU → Linear(256→5)

Output: [B, 5]  — food_score, price_score, atmosphere_score, service_score, overall_satisfaction  (scale 1-10)
```

### 3.3 Checkpoint và config

```python
EXP_ID   = 'EXP_060A_bestsequential_full_configuration'
EXP_DIR  = f'{PROJECT_ROOT}/experiments/{EXP_ID}'
CKPT     = f'{EXP_DIR}/best_model_train_fusion.pth'
XAI_OUT_DIR = f'{PROJECT_ROOT}/xai_outputs/{EXP_ID}'
```

Checkpoint được load qua `xai.utils.load_model(exp_dir, device)`. Hàm này đã tự gọi `enable_eager_attention()` (patch sdpa→eager cho PhoBERT để `output_attentions=True` hoạt động).

### 3.4 Constants từ `xai/config.py`

```python
TARGET_NAMES   = ['food_score', 'price_score', 'atmosphere_score', 'service_score', 'overall_satisfaction']
DISPLAY_NAMES  = ['Food Score', 'Price Score', 'Atmosphere Score', 'Service Score', 'Overall Satisfaction']
FACTOR_NAMES   = ['food', 'price', 'atmos', 'service', 'overall']
TEXT_FEATURE_DIM   = 768
IMAGE_FEATURE_DIM  = 1024
CROSS_ATTN_HIDDEN_DIM = 512
FUSED_DIM      = 1024   # text-origin: dims 0:512, image-origin: dims 512:1024
DEFAULT_MAX_LENGTH  = 256
DEFAULT_MAX_IMAGES  = 4
DEFAULT_DPI    = 150
THESIS_DPI     = 300
```

### 3.5 Cấu trúc cell tổng thể

```
[CELL GROUP 0] Setup & Configuration
[CELL GROUP 1] Sample Selection (3 mẫu)
[CELL GROUP 2] Load Model & Tokenizer
─────────────────────────────────────────
[CELL GROUP 3] SAMPLE A — Overview → Prediction → Grad-CAM → Attention → Cross-Attention → SHAP → LIME → AI Agent
[CELL GROUP 4] SAMPLE B — (tương tự)
[CELL GROUP 5] SAMPLE C — (tương tự)
─────────────────────────────────────────
[CELL GROUP 6] Cross-Sample Summary
[CELL GROUP 7] Manifest & Completion
```

---

## 4. Chiến Lược Chọn Mẫu (Sample Selection Strategy)

### 4.1 Nguyên tắc

Ưu tiên **tái sử dụng (reuse)** các mẫu đã có đầy đủ artifact từ Phase 6 CaseStudy. Chỉ tìm kiếm và regenerate khi mẫu Phase 6 không đủ tiêu chí chất lượng demo.

### 4.2 Ba loại case cần chọn

| Slot | Case Type | Mục đích demo | Case types từ Phase 6 ưu tiên |
|---|---|---|---|
| **Sample A** | Dự đoán chính xác (accurate) | Mô hình hoạt động tốt, XAI coherent | `correct`, `agreement` |
| **Sample B** | Lỗi dự đoán thú vị (error/conflict) | Hiểu giới hạn mô hình | `high_error`, `conflict` |
| **Sample C** | Bằng chứng đa phương thức phong phú | Showcase Cross-Attention + SHAP | `text_dominant`, `image_dominant`, `difficult` |

### 4.3 Tiêu chí đánh giá chất lượng từng mẫu

**Bắt buộc (hard criteria):**
- Review text: ≥ 30 từ, có thể đọc được, không phải noise.
- Số ảnh thực (num_real_images): ≥ 2 (tốt nhất là 3-4).
- Ground truth: có đủ 5 scores.
- Prediction: đã được tính và không phải NaN.
- XAI artifacts: có ít nhất 3 trong 5 loại (gradcam, attention, cross_attention, shap, lime).

**Tiêu chí tối ưu (soft criteria, xếp hạng):**
- Grad-CAM: heatmap có vùng nóng (hot region) phân bố rõ ràng, không phải heatmap đồng đều trắng/xám.
- Attention: top-K words có nghĩa ngữ nghĩa (không phải stopwords).
- Cross-Attention: có ít nhất 1 pair (token, patch) có score > 0.3.
- SHAP: `|text_pct - image_pct|` < 70 (cả 2 modality đều đóng góp).
- LIME Text: ít nhất 3 words quan trọng.
- AI Agent: report đã được validate thành công (validation_passed = True).
- completeness (từ `check_sample_artifacts`) ≥ 0.8.

### 4.4 Thuật toán chọn mẫu tự động

```python
def select_demo_samples(phase6_pred_df, dataset_df, xai_dir, output_dir):
    """
    Trả về list 3 sample_id đã được validate.
    
    Bước 1: Đọc phase6_pred_df (kết quả Phase 6 CaseStudy).
    Bước 2: Với mỗi case type theo thứ tự ưu tiên, tìm mẫu đạt hard criteria.
    Bước 3: Trong các mẫu đạt hard criteria, sort theo soft score.
    Bước 4: Chọn top-1 cho mỗi slot A, B, C.
    Bước 5: Nếu không tìm đủ từ Phase 6, mở rộng tìm kiếm toàn dataset.
    Bước 6: Với mỗi mẫu được chọn, kiểm tra artifacts và regenerate nếu thiếu.
    """
```

**Thứ tự ưu tiên tìm kiếm:**
1. Phase 6 selected cases (file `phase6_cases.json` hoặc tương đương).
2. Phase 6 prediction dataframe toàn bộ.
3. Full dataset CSV.

**Progressive relaxation:** Nếu không tìm được mẫu đạt tất cả hard criteria, giảm dần yêu cầu:
- Vòng 1: num_real_images ≥ 2, completeness ≥ 0.8.
- Vòng 2: num_real_images ≥ 1, completeness ≥ 0.6.
- Vòng 3: Bất kỳ mẫu nào có ground truth và prediction.

### 4.5 Quyết định Reuse vs Regenerate

| Tình huống | Quyết định | Hành động |
|---|---|---|
| Artifact tồn tại, completeness ≥ 0.8, visual quality tốt | **Reuse** | Load từ `xai_outputs/` |
| Artifact tồn tại nhưng từ EXP_ID khác | **Regenerate** | Xóa và tạo lại |
| Artifact tồn tại nhưng Grad-CAM hoàn toàn đồng đều (max-min < 0.05) | **Regenerate** | Tạo lại Grad-CAM |
| Artifact bị thiếu (file not found) | **Regenerate** | Tạo mới |
| AI Agent report thiếu hoặc validation_passed = False | **Regenerate** | Chạy lại agent |
| Artifact tồn tại, file JSON parse được, visual file tồn tại | **Reuse** | Không làm gì |

```python
def should_regenerate(artifact_type, sample_id, xai_dir, exp_id):
    """Trả về (bool, reason_string)"""
    # Kiểm tra file tồn tại
    # Kiểm tra EXP_ID trong metadata nếu có
    # Kiểm tra quality score nếu có thể
    # Trả về True nếu cần regenerate, kèm lý do
```

---

## 5. Cấu Trúc Notebook (Notebook Structure)

### CELL GROUP 0 — Setup & Configuration

**Cell 0.1: Title & Description (Markdown)**
```markdown
# Demo End-to-End: XAI + AI Agent
## Hệ thống đánh giá chất lượng đánh giá nhà hàng đa phương thức có giải thích được
...mô tả ngắn gọn...
```

**Cell 0.2: Mount Google Drive (Code)**
```python
from google.colab import drive
drive.mount('/content/drive')
```
*Ghi chú: Nếu chạy local, bỏ cell này.*

**Cell 0.3: Install Dependencies (Code)**
```python
# Chỉ install những gì chưa có. Không install lại toàn bộ.
import subprocess, sys
packages = ['timm', 'shap', 'lime', 'openai', 'transformers', 'torch']
# ... kiểm tra và install có điều kiện
```

**Cell 0.4: Path Configuration (Code)**
```python
import os, sys
DRIVE_ROOT   = '/content/drive/MyDrive'
PROJECT_ROOT = f'{DRIVE_ROOT}/SE365'  # Điều chỉnh nếu cần
sys.path.insert(0, PROJECT_ROOT)

EXP_ID      = 'EXP_060A_bestsequential_full_configuration'
EXP_DIR     = f'{PROJECT_ROOT}/experiments/{EXP_ID}'
XAI_OUT_DIR = f'{PROJECT_ROOT}/xai_outputs/{EXP_ID}'
DATA_DIR    = f'{PROJECT_ROOT}/data'
IMAGE_DIR   = f'{DATA_DIR}/images'
DEMO_OUT    = f'{PROJECT_ROOT}/demo_e2e'

os.makedirs(DEMO_OUT, exist_ok=True)
OPENAI_API_KEY = 'sk-...'  # Hoặc từ Colab Secret / environment variable
```

**Cell 0.5: Import Core Modules (Code)**
```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import json, warnings, traceback
from pathlib import Path

from xai.config import (TARGET_NAMES, DISPLAY_NAMES, FACTOR_NAMES,
                         FUSED_DIM, DEFAULT_MAX_LENGTH, DEFAULT_MAX_IMAGES,
                         DEFAULT_DPI, THESIS_DPI)
from xai.utils import (load_model, load_single_sample, get_prediction,
                        get_tokenizer, get_image_processor)
from xai.gradcam_explainer import GradCAMExplainer, compute_gradcam_for_image, overlay_cam_on_image
from xai.attention_explainer import AttentionExplainer, CrossAttentionExplainer
from xai.shap_explainer import SHAPExplainer, extract_fused_embeddings
from xai.lime_explainer import LIMEExplainer
from xai.case_study import check_sample_artifacts

from agent.evidence_loader import EvidenceLoader
from agent.evidence_builder import EvidenceBuilder
from agent.reasoning import build_reasoning_graph
from agent.prompt_builder import PromptBuilder
from agent.openai_client import OpenAIClient
from agent.output_schema import AGENT_OUTPUT_SCHEMA
from agent.validator import OutputValidator
from agent.report_generator import save_sample_report, ReportGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
```

**Cell 0.6: Load Model (Code)**
```python
model = load_model(EXP_DIR, device)
# load_model đã gọi enable_eager_attention() — không cần gọi lại
tokenizer        = get_tokenizer('vinai/phobert-base-v2')
image_processor  = get_image_processor('swin_base_patch4_window7_224')
# image_processor là TimmProcessor (không phải HuggingFace AutoImageProcessor)
print('Model loaded. Eager attention enabled.')
```

---

### CELL GROUP 1 — Sample Selection

**Cell 1.1: Load Phase 6 Results (Code)**
```python
# Đọc kết quả Phase 6 và dataset
phase6_csv   = f'{XAI_OUT_DIR}/phase6_predictions.csv'  # Điều chỉnh tên nếu khác
dataset_csv  = f'{DATA_DIR}/processed_reviews.csv'       # Điều chỉnh tên nếu khác

phase6_df   = pd.read_csv(phase6_csv) if os.path.exists(phase6_csv) else None
dataset_df  = pd.read_csv(dataset_csv)
```

**Cell 1.2: Automatic Sample Selection (Code)**
```python
def score_sample(sample_id, xai_dir):
    """Tính điểm chất lượng 0-10 cho demo."""
    arts = check_sample_artifacts(sample_id, xai_dir)
    score = arts['completeness'] * 5  # 0-5 điểm từ completeness
    # Cộng điểm nếu num_real_images >= 3
    # Cộng điểm nếu AI Agent report validated
    # Trừ điểm nếu Grad-CAM đồng đều
    return score

# Hàm chọn mẫu theo từng slot
SLOT_PRIORITY = {
    'A_accurate':    ['correct', 'agreement'],
    'B_error':       ['high_error', 'conflict'],
    'C_multimodal':  ['text_dominant', 'image_dominant', 'difficult'],
}

selected_samples = {}  # {'A': sample_id, 'B': sample_id, 'C': sample_id}
# ... logic chọn mẫu với progressive relaxation
```

**Cell 1.3: Display Selection Summary (Code + Markdown)**
```python
# Hiển thị bảng tóm tắt 3 mẫu đã chọn
print('='*60)
print('Demo Samples Selected:')
for slot, sid in selected_samples.items():
    arts = check_sample_artifacts(sid, XAI_OUT_DIR)
    print(f'  Sample {slot}: {sid} — completeness={arts["completeness"]:.2f}')
```

---

### CELL GROUP 2 — Load Samples

**Cell 2.1: Load All 3 Samples (Code)**
```python
samples = {}
for slot, sid in selected_samples.items():
    try:
        sample_data = load_single_sample(
            csv_path=dataset_csv,
            idx=sid,  # idx có thể là integer index hoặc string sample_id
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_dir=IMAGE_DIR,
            device=device
        )
        # sample_data keys: input_ids[1,256], attention_mask[1,256],
        #   pixel_values[1,4,C,H,W], num_images[1], factor_scores[5],
        #   text, loaded_images (list PIL), num_real_images
        samples[slot] = {'id': sid, 'data': sample_data}
        print(f'Loaded Sample {slot}: {sid}')
    except Exception as e:
        print(f'[WARNING] Cannot load Sample {slot} ({sid}): {e}')
        # Ghi log và tiếp tục, không raise
```

---

### CELL GROUP 3-5 — Per-Sample Demo (lặp lại cho A, B, C)

Mỗi sample group gồm các cell con sau:

#### 3.1 Sample Overview

**Cell X.1: Sample Header (Markdown)**
```markdown
---
## Sample A — [Case Type]: [Sample ID]
**Loại case:** Accurate Prediction  
**Mục đích:** Minh họa khi mô hình dự đoán chính xác, XAI nhất quán với ground truth.
```

**Cell X.2: Display Review Text (Code)**
```python
text = samples['A']['data']['text']
print('📝 Review Text:')
print(text)
print(f'\nĐộ dài: {len(text.split())} từ')
```

**Cell X.3: Display Images (Code)**
```python
loaded_images   = samples['A']['data']['loaded_images']
num_real_images = samples['A']['data']['num_real_images']

fig, axes = plt.subplots(1, min(num_real_images, 4), figsize=(4*min(num_real_images,4), 4))
if num_real_images == 1:
    axes = [axes]
for i, img in enumerate(loaded_images[:num_real_images]):
    axes[i].imshow(img)
    axes[i].set_title(f'Image {i+1}')
    axes[i].axis('off')
plt.suptitle(f'Sample A — {num_real_images} ảnh đầu vào')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/overview_images.png', dpi=DEFAULT_DPI)
plt.show()
```

---

## 6. Trình Bày Dự Đoán (Prediction Demonstration)

### 6.1 Thực hiện dự đoán

```python
pred_result = get_prediction(model, samples['A']['data'])
# pred_result keys: predictions[5], ground_truth[5], absolute_errors[5], mean_mae
```

### 6.2 Bảng kết quả (Code)

```python
df_pred = pd.DataFrame({
    'Score Type':    DISPLAY_NAMES,
    'Ground Truth':  pred_result['ground_truth'],
    'Prediction':    [f'{p:.2f}' for p in pred_result['predictions']],
    'Abs Error':     [f'{e:.2f}' for e in pred_result['absolute_errors']],
})
print(df_pred.to_string(index=False))
print(f'\nMean MAE: {pred_result["mean_mae"]:.3f}')
```

### 6.3 Bar chart (Code)

```python
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(DISPLAY_NAMES))
width = 0.35
bars_gt   = ax.bar(x - width/2, pred_result['ground_truth'], width, label='Ground Truth', color='steelblue', alpha=0.8)
bars_pred = ax.bar(x + width/2, pred_result['predictions'],  width, label='Prediction',   color='coral',     alpha=0.8)
ax.set_ylabel('Score (1-10)')
ax.set_title(f'Sample A — Prediction vs Ground Truth')
ax.set_xticks(x)
ax.set_xticklabels(DISPLAY_NAMES, rotation=20, ha='right')
ax.set_ylim(1, 10)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/prediction_chart.png', dpi=DEFAULT_DPI)
plt.show()
```

### 6.4 Diễn giải ngắn (Markdown cell sau chart)

Notebook phải có một Markdown cell giải thích kết quả bằng tiếng Việt đơn giản. Ví dụ:

```markdown
**Nhận xét:** Mô hình dự đoán Overall Satisfaction = 7.8, Ground Truth = 8.0 (sai lệch 0.2 điểm).
Food Score và Service Score được dự đoán khá chính xác. Price Score có sai lệch cao nhất (0.6 điểm).
```

### 6.5 Lưu kết quả

```python
os.makedirs(f'{DEMO_OUT}/sample_A', exist_ok=True)
with open(f'{DEMO_OUT}/sample_A/prediction.json', 'w', encoding='utf-8') as f:
    json.dump(pred_result, f, ensure_ascii=False, indent=2)
```

---

## 7. Trình Bày Grad-CAM

### 7.1 Giới hạn đã biết (Known Limitation) — PHẢI HIỂN THỊ

> ⚠️ **Giới hạn kỹ thuật:** Hệ thống CrossAttentionFusion dùng chung encoder Swin-B cho tất cả 5 targets. Chỉ có lớp Linear cuối (256→5) là khác nhau theo target. Do đó, gradient của 5 targets có cosine similarity rất cao (thường > 0.95), và 5 Grad-CAM heatmap trông **gần như giống nhau về mặt thị giác**. Đây là đặc tính của kiến trúc fusion, không phải lỗi.

Notebook phải có một Markdown cell nêu rõ giới hạn này **trước khi** hiển thị Grad-CAM.

### 7.2 Chỉ hiển thị Overall Satisfaction

Vì lý do trên, demo chỉ hiển thị Grad-CAM cho `overall_satisfaction` (index 4). Điều này:
- Tránh lặp lại 5 ảnh gần giống nhau gây nhàm chán.
- Tập trung vào target quan trọng nhất.

### 7.3 Target layer

```python
from xai.gradcam_explainer import find_target_layer
target_layer = find_target_layer(model)
# → model.image_model.encoder.norm  (LayerNorm, output BHWC [B,7,7,1024])
```

`normalize_feature_map_to_bchw()` trong `xai/utils.py` xử lý format BHWC → BCHW tự động.

### 7.4 Reuse vs Regenerate Grad-CAM

```python
gradcam_path = f'{XAI_OUT_DIR}/gradcam/sample_{sid}/cam_overall_img0.png'
if os.path.exists(gradcam_path) and should_reuse('gradcam', sid, XAI_OUT_DIR):
    cam_overlay = np.array(Image.open(gradcam_path))
else:
    # Regenerate
    cam = compute_gradcam_for_image(
        model=model,
        sample=samples['A']['data'],
        target_idx=4,  # overall_satisfaction
        image_idx=0,   # ảnh đầu tiên
        target_layer=target_layer,
        device=device
    )  # → numpy [H,W] normalized [0,1]
    cam_overlay = overlay_cam_on_image(cam, samples['A']['data']['loaded_images'][0])
    Image.fromarray(cam_overlay).save(f'{DEMO_OUT}/sample_A/gradcam_overall.png')
```

### 7.5 Hiển thị side-by-side

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(samples['A']['data']['loaded_images'][0])
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(cam_overlay)
axes[1].set_title('Grad-CAM — Overall Satisfaction\n(Vùng đỏ = mô hình chú ý nhiều)')
axes[1].axis('off')
plt.suptitle('Grad-CAM Explanation — Overall Satisfaction')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/gradcam_overlay.png', dpi=DEFAULT_DPI)
plt.show()
```

### 7.6 Diễn giải (Markdown cell)

```markdown
**Đọc kết quả Grad-CAM:** Vùng màu đỏ/cam cho biết phần ảnh mà mô hình tập trung khi
đánh giá Overall Satisfaction. Ví dụ: nếu vùng nóng rơi vào khu vực thực phẩm trên bàn,
mô hình đang chú ý đến chất lượng món ăn nhìn qua ảnh.

*Lưu ý: Do kiến trúc CrossAttentionFusion dùng chung encoder, Grad-CAM của 5 targets 
có xu hướng giống nhau. Điều này là đặc tính kiến trúc, không phải lỗi của mô hình.*
```

---

## 8. Trình Bày PhoBERT Attention

### 8.1 Extract attention

```python
from xai.attention_explainer import (extract_phobert_attention, aggregate_attention,
                                      cls_token_importance, merge_subword_attention)

attn_result = extract_phobert_attention(
    model=model,
    input_ids=samples['A']['data']['input_ids'],
    attention_mask=samples['A']['data']['attention_mask'],
    tokenizer=tokenizer
)
# attn_result: {'attentions': [12, 12, L, L], 'tokens': list[str], 'seq_len': int}
```

### 8.2 Aggregate và merge subwords

```python
agg_matrix = aggregate_attention(
    attn_result['attentions'],
    strategy='last_layer_mean'  # Dùng mean của layer cuối cùng
)
# agg_matrix: [L, L]

token_importances = cls_token_importance(agg_matrix, attn_result['tokens'])
# token_importances: list of (token, score), sorted descending

word_importances = merge_subword_attention(
    token_importances,
    attn_result['tokens'],
    strategy='mean'
)
# word_importances: list of (word, score) — đã merge BPE subwords
```

**Quan trọng:** Luôn dùng `word_importances` (merged) thay vì `token_importances` (raw) khi hiển thị. Subword như `"_ăn", "##g"` không có nghĩa với người xem.

### 8.3 Highlighted text (Code)

```python
# Tô màu text theo attention score
# Dùng matplotlib hoặc HTML display trong Colab
from IPython.display import HTML

def highlight_text(word_importances, text):
    """Tô màu từ trong text theo attention score."""
    word_scores = dict(word_importances)
    max_score = max(word_scores.values()) if word_scores else 1
    words = text.split()
    html_parts = []
    for w in words:
        score = word_scores.get(w, 0)
        intensity = int(255 * (1 - score / max_score))
        color = f'rgb(255, {intensity}, {intensity})'
        html_parts.append(f'<span style="background-color:{color}; padding:2px">{w}</span>')
    return ' '.join(html_parts)

display(HTML(highlight_text(word_importances, samples['A']['data']['text'])))
```

### 8.4 Top-K word importance bar chart

```python
TOP_K = 15
top_words = word_importances[:TOP_K]
words_disp = [w for w, _ in top_words]
scores     = [s for _, s in top_words]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(words_disp[::-1], scores[::-1], color='tomato', alpha=0.8)
ax.set_xlabel('Attention Score (normalized)')
ax.set_title(f'Top {TOP_K} Important Words — PhoBERT Attention')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/attention_top_words.png', dpi=DEFAULT_DPI)
plt.show()
```

### 8.5 Disclaimer (Markdown cell bắt buộc)

```markdown
**Lưu ý quan trọng:** PhoBERT Attention cho thấy những từ mà mô hình **tập trung** khi
xử lý văn bản, nhưng attention weight **không phải** bằng chứng nhân quả (causal evidence).
Một từ có attention cao không nhất thiết có ý nghĩa quyết định đến output.
Đây là điểm yếu đã được ghi nhận trong tài liệu học thuật về interpretability.
Kết hợp với SHAP để có góc nhìn toàn diện hơn.
```

---

## 9. Trình Bày Cross-Attention

### 9.1 Extract cross-attention

```python
from xai.attention_explainer import extract_cross_attention

cross_attn = extract_cross_attention(
    model=model,
    sample=samples['A']['data'],
    tokenizer=tokenizer
)
# cross_attn: {
#   't2i_attn': [T, P],   T = seq_len, P = 49 patches (7×7)
#   'i2t_attn': [P, T],
#   'tokens': list[str],
#   'seq_len': T,
#   'num_patches': 49
# }
```

### 9.2 Heatmap T2I (Token → Patch)

```python
t2i = cross_attn['t2i_attn']  # [T, 49]
tokens = cross_attn['tokens']

# Chỉ hiển thị top 20 tokens để tránh ma trận quá lớn
top_token_idx = np.argsort(t2i.max(axis=1))[-20:][::-1]
t2i_display = t2i[top_token_idx, :]
tokens_display = [tokens[i] for i in top_token_idx]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(t2i_display, cmap='hot', aspect='auto')
ax.set_yticks(range(len(tokens_display)))
ax.set_yticklabels(tokens_display, fontsize=8)
ax.set_xlabel('Image Patches (7×7 = 49 patches)')
ax.set_title('Cross-Attention: Token → Patch (T2I)\n"Khi đọc từ này, mô hình nhìn vào vùng ảnh nào?"')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/cross_attn_t2i.png', dpi=DEFAULT_DPI)
plt.show()
```

### 9.3 Patch overlay trên ảnh gốc

```python
# Reshape patch attention thành 7×7 và overlay lên ảnh gốc
# Chọn 1 token quan trọng nhất để demo
best_token_idx = int(np.argmax(t2i.max(axis=1)))
best_token = tokens[best_token_idx]
patch_weights = t2i[best_token_idx, :].reshape(7, 7)  # [7,7]

# Upsample lên 224×224 và overlay
from PIL import Image as PILImage
import cv2
heat = cv2.resize(patch_weights.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
orig = np.array(samples['A']['data']['loaded_images'][0].resize((224, 224)))
heat_rgb = (plt.cm.hot(heat)[:, :, :3] * 255).astype(np.uint8)
overlay = (0.5 * orig + 0.5 * heat_rgb).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(orig); axes[0].set_title('Original'); axes[0].axis('off')
axes[1].imshow(heat, cmap='hot'); axes[1].set_title(f'Patch Weights for\n"{best_token}"'); axes[1].axis('off')
axes[2].imshow(overlay); axes[2].set_title('Overlay'); axes[2].axis('off')
plt.suptitle(f'Cross-Attention Overlay — Token: "{best_token}"')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/cross_attn_overlay_{best_token}.png', dpi=DEFAULT_DPI)
plt.show()
```

### 9.4 Top-K pairs (bidirectional)

```python
# Top-K (token, patch) pairs theo T2I score
TOP_K_PAIRS = 10
flat_scores = t2i.flatten()
top_k_flat  = np.argsort(flat_scores)[-TOP_K_PAIRS:][::-1]
top_pairs   = [(tokens[i // 49], i % 49, flat_scores[i]) for i in top_k_flat]

print('Top Cross-Attention Pairs (Token → Patch):')
for tok, patch_id, score in top_pairs:
    row, col = divmod(patch_id, 7)
    print(f'  "{tok}" → Patch [{row},{col}] (score={score:.3f})')
```

### 9.5 Diễn giải (Markdown)

```markdown
**Cross-Attention (Bidirectional):**
- **Token → Patch (T2I):** Khi mô hình đọc từ X, nó nhìn vào vùng ảnh nào?
- **Patch → Token (I2T):** Khi mô hình "nhìn" vào vùng ảnh Y, nó liên kết với từ nào?

Cross-Attention là cơ chế kết nối ngôn ngữ và hình ảnh trong CrossAttentionFusion.
Ảnh và văn bản được chiếu xuống không gian chung 512 chiều trước khi thực hiện attention.
```

---

## 10. Trình Bày SHAP

### 10.1 Chuẩn bị background và fused embeddings

```python
from xai.shap_explainer import SHAPExplainer, extract_fused_embeddings, FusionHeadWrapper
from torch.utils.data import DataLoader

# Background: cần một batch dataloader (ít nhất 50 mẫu ngẫu nhiên từ test set)
# Nếu đã có background từ Phase 4, load lại để tiết kiệm thời gian
background_path = f'{XAI_OUT_DIR}/shap/background_embeddings.npy'
if os.path.exists(background_path) and should_reuse('shap_background', None, XAI_OUT_DIR):
    background = np.load(background_path)
    print(f'Reusing background embeddings: {background.shape}')
else:
    # Tạo dataloader từ dataset
    # background = extract_fused_embeddings(model, background_loader, device)
    # np.save(background_path, background)
    pass

# background shape: [N, 1024] — FUSED_DIM
```

### 10.2 Compute SHAP values

```python
explainer_shap = SHAPExplainer(
    model=model,
    background=torch.tensor(background, dtype=torch.float32),
    device=device,
    output_dir=f'{DEMO_OUT}/sample_A'
)

shap_result = explainer_shap.explain_sample(
    sample_fused=sample_A_fused_embedding,  # [1, 1024] tensor
    sample_id=selected_samples['A'],
    background=torch.tensor(background, dtype=torch.float32)
)
# shap_result per target: {'shap_values': [1024], 'base_value': float, 'modality_contribution': dict}
```

### 10.3 Modality contribution chart

```python
from xai.shap_explainer import modality_contribution

# Với Overall Satisfaction (index 4)
shap_vals_overall = shap_result[4]['shap_values']  # [1024]
contrib = modality_contribution(shap_vals_overall, text_dim=512)
# contrib: {text_pct, image_pct, text_abs, image_abs, text_signed, image_signed}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
axes[0].pie(
    [contrib['text_pct'], contrib['image_pct']],
    labels=['Text-origin (dims 0:512)', 'Image-origin (dims 512:1024)'],
    colors=['#4ECDC4', '#FF6B6B'],
    autopct='%1.1f%%', startangle=90
)
axes[0].set_title('Modality Contribution\n(SHAP |value| sum)')

# Per-target bar chart
text_pcts  = [modality_contribution(shap_result[i]['shap_values'], 512)['text_pct']  for i in range(5)]
image_pcts = [modality_contribution(shap_result[i]['shap_values'], 512)['image_pct'] for i in range(5)]
x = np.arange(5)
axes[1].bar(x, text_pcts,  label='Text-origin', color='#4ECDC4', alpha=0.8)
axes[1].bar(x, image_pcts, bottom=text_pcts, label='Image-origin', color='#FF6B6B', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(DISPLAY_NAMES, rotation=20, ha='right')
axes[1].set_ylabel('Contribution (%)')
axes[1].set_title('Modality Contribution per Target')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/shap_modality.png', dpi=DEFAULT_DPI)
plt.show()
```

### 10.4 Disclaimer bắt buộc (Markdown)

```markdown
**Lưu ý về SHAP trong hệ thống này:**

SHAP được tính trên **fused embedding** [1024 dims] — đầu ra của CrossAttentionFusion 
trước khi đi vào prediction head.

- Dims 0:512 được gọi là "text-origin" (bắt nguồn từ text projection).
- Dims 512:1024 được gọi là "image-origin" (bắt nguồn từ image projection).

Tuy nhiên, do Cross-Attention, text-origin dims đã bị ảnh hưởng bởi image context 
và ngược lại. Do đó, đây **không** phải contribution thuần túy của text hay image —
mà là contribution của **vùng feature space** sau cross-attention.

Kết quả SHAP phản ánh tầm quan trọng tương đối của hai nhánh trong không gian fused.
```

---

## 11. Trình Bày LIME

### 11.1 LIME Text

```python
from xai.lime_explainer import run_lime_text

lime_text_result = run_lime_text(
    model=model,
    sample=samples['A']['data'],
    score_index=4,          # overall_satisfaction
    tokenizer=tokenizer,
    device=device,
    num_samples=500         # Số perturbation (500 đủ nhanh cho demo)
)
# lime_text_result: LimeTextExplanation object

# Lấy top-K words
top_words_lime = lime_text_result.as_list()[:15]  # [(word, weight), ...]

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
words   = [w for w, _ in top_words_lime]
weights = [s for _, s in top_words_lime]
colors  = ['#2ECC71' if w > 0 else '#E74C3C' for w in weights]
ax.barh(words[::-1], weights[::-1], color=colors[::-1], alpha=0.85)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('LIME Weight')
ax.set_title('LIME Text — Overall Satisfaction\n(Xanh = tích cực, Đỏ = tiêu cực)')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/lime_text.png', dpi=DEFAULT_DPI)
plt.show()
```

### 11.2 LIME Image

```python
from xai.lime_explainer import run_lime_image

lime_image_result = run_lime_image(
    model=model,
    sample=samples['A']['data'],
    score_index=4,           # overall_satisfaction
    image_processor=image_processor,
    device=device,
    num_samples=200          # Ít hơn text vì image perturbation tốn kém hơn
)
# lime_image_result: LimeImageExplanation object

# Hiển thị superpixel
from lime.lime_image import mark_boundaries
temp, mask = lime_image_result.get_image_and_mask(
    label=1,       # 1 = "high score" class trong pseudo-binary setup
    positive_only=True,
    num_features=5,
    hide_rest=False
)
img_boundary = mark_boundaries(temp / 255.0, mask)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(samples['A']['data']['loaded_images'][0])
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(img_boundary)
axes[1].set_title('LIME Image — Vùng quan trọng\n(Viền xanh = superpixels quan trọng)')
axes[1].axis('off')
plt.tight_layout()
plt.savefig(f'{DEMO_OUT}/sample_A/lime_image.png', dpi=DEFAULT_DPI)
plt.show()
```

### 11.3 Disclaimer bắt buộc (Markdown)

```markdown
**LIME — Local Explanation:**

LIME giải thích dự đoán cho **mẫu cụ thể này** bằng cách tạo perturbations xung quanh 
input và fit một mô hình linear đơn giản.

- **LIME Text:** Từ nào, nếu bị loại bỏ, làm thay đổi dự đoán nhiều nhất?
- **LIME Image:** Superpixel nào ảnh hưởng nhất đến điểm số?

Lưu ý: LIME là **giải thích cục bộ** (local) — kết quả chỉ đúng cho mẫu này,
không phản ánh hành vi toàn cục của mô hình.

Pseudo-classification được dùng (sigmoid của score → [p_low, p_high]) để tương thích
với API của LIME. Đây là một kỹ thuật kỹ thuật, không phải thay đổi bài toán regression.
```

---

## 12. Trình Bày AI Agent

### 12.1 Kiến trúc Agent (Markdown giải thích)

```markdown
## AI Agent Pipeline

```
EvidenceLoader  →  EvidenceBuilder  →  ReasoningGraph  →  PromptBuilder
     ↓                   ↓                   ↓                  ↓
Load artifacts    Compress evidence    Pre-LLM reasoning    GPT-4o prompt
     ↓
OpenAIClient (GPT-4o)
     ↓
OutputValidator  →  ReportGenerator
     ↓                    ↓
Validate schema     JSON + Markdown report
```

GPT-4o chỉ **ngôn ngữ hóa** bằng chứng. Nó **KHÔNG** dự đoán lại scores hay 
thay đổi kết quả XAI.
```

### 12.2 Load evidence

```python
from agent.evidence_loader import EvidenceLoader
from agent.evidence_builder import EvidenceBuilder

loader  = EvidenceLoader()
builder = EvidenceBuilder()

evidence     = loader.load(
    sample_id=selected_samples['A'],
    xai_dir=XAI_OUT_DIR,
    case_id=selected_samples['A']
)
built_evidence = builder.build(evidence)

print('Evidence loaded:')
print(f'  Grad-CAM: {"✓" if evidence.get("gradcam") else "✗"}')
print(f'  Attention: {"✓" if evidence.get("attention") else "✗"}')
print(f'  Cross-Attention: {"✓" if evidence.get("cross_attention") else "✗"}')
print(f'  SHAP: {"✓" if evidence.get("shap") else "✗"}')
print(f'  LIME: {"✓" if evidence.get("lime") else "✗"}')
if built_evidence.get('missing_summary'):
    print(f'  Missing: {built_evidence["missing_summary"]}')
```

### 12.3 Reasoning Graph

```python
from agent.reasoning import build_reasoning_graph

pred_result_A = ...  # Từ bước prediction
reasoning = build_reasoning_graph(
    predictions=pred_result_A['predictions'],
    evidence=evidence,
    review_text=samples['A']['data']['text']
)

# Hiển thị reasoning graph dạng bảng
print('\nReasoning Graph:')
print(f'{"Target":<25} {"Evidence Strength":<20} {"Interpretation Hint"}')
print('-'*70)
for t, info in reasoning['targets'].items():
    print(f'{t:<25} {info["evidence_strength"]:<20} {info["interpretation_hint"]}')

# Hiển thị agreement matrix
print('\nAgreement Matrix (XAI methods ↔ targets):')
agg_matrix_df = pd.DataFrame(reasoning['agreement_matrix'])
display(agg_matrix_df)
```

### 12.4 Gọi GPT-4o

```python
from agent.prompt_builder import PromptBuilder
from agent.openai_client import OpenAIClient

prompt_builder = PromptBuilder()
client         = OpenAIClient(api_key=OPENAI_API_KEY)

messages = prompt_builder.build(
    sample_id=selected_samples['A'],
    review_text=samples['A']['data']['text'],
    predictions=pred_result_A['predictions'],
    ground_truth=pred_result_A['ground_truth'],
    built_evidence=built_evidence,
    reasoning_graph=reasoning,
    language='vi'
)

print('Calling GPT-4o...')
agent_output = client.generate_json(
    messages=messages,
    model='gpt-4o',
    temperature=0.3
)
print('GPT-4o response received.')
```

### 12.5 Validate output

```python
from agent.validator import OutputValidator

validator = OutputValidator()
validation_result = validator.validate(agent_output, pred_result_A['predictions'])

print(f'\nValidation: {"PASSED ✓" if validation_result["passed"] else "FAILED ✗"}')
if not validation_result['passed']:
    print('Issues:')
    for issue in validation_result.get('issues', []):
        print(f'  - {issue}')
```

### 12.6 Generate và hiển thị report

```python
from agent.report_generator import save_sample_report

agent_out_dir = f'{DEMO_OUT}/sample_A/agent'
os.makedirs(agent_out_dir, exist_ok=True)

report_paths = save_sample_report(
    output=agent_output,
    output_dir=agent_out_dir,
    review_text=samples['A']['data']['text'],
    predictions=pred_result_A['predictions'],
    ground_truth=pred_result_A['ground_truth']
)
# report_paths: {'json': path, 'markdown': path}

# Hiển thị Customer View (Phần A)
print('\n' + '='*60)
print('PHẦN A — CUSTOMER VIEW')
print('='*60)
with open(report_paths['markdown'], 'r', encoding='utf-8') as f:
    report_md = f.read()
# Trích phần A
part_a = report_md.split('## Phần B')[0] if '## Phần B' in report_md else report_md
from IPython.display import Markdown
display(Markdown(part_a))

# Hiển thị Technical View (Phần B)
print('\n' + '='*60)
print('PHẦN B — TECHNICAL VIEW')
print('='*60)
if '## Phần B' in report_md:
    part_b = '## Phần B' + report_md.split('## Phần B')[1]
    display(Markdown(part_b))
```

### 12.7 Hiển thị visual artifacts inline

Trước khi hiển thị report text, notebook nên hiển thị lại các ảnh artifact đã tạo:

```python
# Hiển thị tổng hợp artifacts cho 1 sample
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 4, fig)

# Row 1: Original images + Grad-CAM
# Row 2: LIME image + SHAP chart + Agent summary panel
# ... load và hiển thị từng file PNG từ demo_e2e/sample_A/
```

---

## 13. Cấu Trúc Thư Mục Output (Output Folder Structure)

```
demo_e2e/
├── manifest.json                  # Danh sách 3 sample IDs + selection metadata
├── summary_comparison.png         # Cross-sample comparison chart
├── demo_summary.json             # Tóm tắt toàn bộ demo
│
├── sample_A/                     # Sample loại: accurate prediction
│   ├── overview_images.png
│   ├── prediction.json
│   ├── prediction_chart.png
│   ├── gradcam_overall.png       # Grad-CAM numpy array
│   ├── gradcam_overlay.png       # Overlay visualization
│   ├── attention_top_words.png
│   ├── attention_highlighted.png
│   ├── cross_attn_t2i.png
│   ├── cross_attn_i2t.png
│   ├── cross_attn_overlay_<token>.png
│   ├── shap_modality.png
│   ├── shap_per_target.png
│   ├── lime_text.png
│   ├── lime_image.png
│   └── agent/
│       ├── report.json
│       ├── report.md
│       └── validation_result.json
│
├── sample_B/                     # Sample loại: prediction error/conflict
│   └── (cùng cấu trúc như sample_A)
│
├── sample_C/                     # Sample loại: multimodal evidence-rich
│   └── (cùng cấu trúc như sample_A)
│
└── errors/
    ├── sample_A_errors.json      # Log lỗi cho từng bước của Sample A
    ├── sample_B_errors.json
    └── sample_C_errors.json
```

### 13.1 manifest.json format

```json
{
  "demo_version": "1.0",
  "created_at": "2026-...",
  "exp_id": "EXP_060A_bestsequential_full_configuration",
  "samples": {
    "A": {
      "sample_id": "...",
      "case_type": "correct",
      "completeness": 0.95,
      "reused_from_phase6": true,
      "regenerated_artifacts": []
    },
    "B": { "..." },
    "C": { "..." }
  }
}
```

---

## 14. Xử Lý Lỗi (Error Handling)

### 14.1 Nguyên tắc cốt lõi

**Notebook KHÔNG BAO GIỜ dừng hoàn toàn vì một bước thất bại.**

Mỗi bước XAI và AI Agent được bọc trong `try/except`. Khi lỗi xảy ra:
1. In cảnh báo màu vàng (`[WARNING]`).
2. Ghi lỗi vào `demo_e2e/errors/sample_X_errors.json`.
3. Hiển thị placeholder (ví dụ: ảnh màu xám với text "Không có dữ liệu").
4. Tiếp tục cell tiếp theo.

### 14.2 Pattern xử lý lỗi chuẩn

```python
import traceback

errors_log = {}  # Tích lũy trong suốt notebook

def run_step_safe(step_name, fn, *args, **kwargs):
    """
    Wrapper chạy một bước XAI an toàn.
    Trả về (result, success_bool).
    """
    try:
        result = fn(*args, **kwargs)
        print(f'✓ {step_name}')
        return result, True
    except Exception as e:
        msg = f'[WARNING] {step_name} failed: {type(e).__name__}: {e}'
        tb  = traceback.format_exc()
        print(msg)
        errors_log[step_name] = {'error': str(e), 'traceback': tb}
        return None, False

# Ví dụ sử dụng:
gradcam_result, ok = run_step_safe(
    'GradCAM_Sample_A',
    compute_gradcam_for_image,
    model, samples['A']['data'], 4, 0, target_layer, device
)
if not ok:
    # Hiển thị placeholder
    print('[SKIP] Grad-CAM visualization skipped.')
```

### 14.3 Lưu error log

```python
# Cuối mỗi sample, lưu error log
with open(f'{DEMO_OUT}/errors/sample_A_errors.json', 'w', encoding='utf-8') as f:
    json.dump(errors_log, f, ensure_ascii=False, indent=2)
```

### 14.4 Các tình huống lỗi đặc biệt

| Tình huống | Xử lý |
|---|---|
| Model checkpoint không tồn tại | Raise exception rõ ràng ở Cell 0.6 — đây là lỗi setup nghiêm trọng, không bỏ qua |
| Dataset CSV không tồn tại | Raise exception ở Cell 1.1 — không có data thì không demo được |
| Sample ID không tìm thấy trong CSV | Log warning, thử sample ID khác từ danh sách backup |
| Grad-CAM gradient bằng 0 (dead gradient) | Log warning, hiển thị ảnh gốc với text "Grad-CAM không khả dụng cho mẫu này" |
| PhoBERT attention extraction lỗi (SDPA not patched) | Check `enable_eager_attention()` đã được gọi chưa, retry |
| SHAP background không đủ mẫu | Giảm background xuống 20 mẫu tối thiểu, log warning |
| LIME timeout | Set `num_samples` thấp hơn (100 text, 50 image), retry |
| OpenAI API key không hợp lệ | Log error rõ ràng, skip AI Agent section, tiếp tục XAI display |
| GPT-4o rate limit | Retry với exponential backoff (đã implement trong `OpenAIClient`) |
| Agent output không qua validation | Hiển thị raw JSON, note "Validation failed" |

---

## 15. Danh Sách Kiểm Tra (Validation Checklist)

Sau khi implement notebook, thực hiện kiểm tra các mục sau trước khi demo trước hội đồng:

### 15.1 Setup

- [ ] Google Drive mount thành công.
- [ ] Checkpoint file tồn tại tại đúng đường dẫn.
- [ ] Tất cả import không báo lỗi.
- [ ] `device = cuda` (nếu có GPU).
- [ ] Model load thành công, `enable_eager_attention()` đã được gọi.
- [ ] 3 sample IDs đã được chọn và in ra màn hình.

### 15.2 Per-Sample

Với mỗi sample (A, B, C):
- [ ] Review text hiển thị đúng, đọc được.
- [ ] Ít nhất 2 ảnh hiển thị (không phải black padding image).
- [ ] Prediction table đúng, không có NaN.
- [ ] Bar chart prediction vs ground truth render đúng.
- [ ] Grad-CAM overlay hiển thị (không phải ảnh trắng đồng đều).
- [ ] Attention bar chart có top-K words có nghĩa (không phải stopwords hay BPE fragments).
- [ ] Cross-attention heatmap T2I render được.
- [ ] Cross-attention overlay chọn được ít nhất 1 token có nghĩa.
- [ ] SHAP modality pie chart tổng 100%.
- [ ] LIME text chart có ít nhất 3 words.
- [ ] LIME image superpixel hiển thị (không crash).
- [ ] AI Agent: evidence loaded.
- [ ] AI Agent: GPT-4o response nhận được.
- [ ] AI Agent: validation_passed = True (hoặc note validation failed).
- [ ] Report Markdown hiển thị Phần A và Phần B.

### 15.3 Output Files

- [ ] `demo_e2e/manifest.json` tồn tại.
- [ ] Mỗi sample folder có đủ PNG files.
- [ ] Agent report JSON và Markdown tồn tại cho cả 3 samples.
- [ ] Error logs được lưu (dù không có lỗi).

### 15.4 Consistency Check

- [ ] Sample A, B, C là 3 loại case khác nhau.
- [ ] Không có sample nào bị skip hoàn toàn.
- [ ] Cross-sample comparison chart có đủ 3 samples.
- [ ] Summary JSON có đủ metadata.

### 15.5 Presentation Check

- [ ] Tất cả chart có title và xlabel/ylabel rõ ràng.
- [ ] Tất cả Markdown explanation cell có nội dung (không bỏ trống).
- [ ] Disclaimer về Grad-CAM limitation hiển thị rõ.
- [ ] Disclaimer về SHAP/LIME hiển thị rõ.
- [ ] AI Agent disclaimer ("GPT-4o không dự đoán lại scores") hiển thị rõ.
- [ ] Thời gian chạy toàn bộ notebook < 30 phút (với GPU).

---

## 16. Bàn Giao Cuối (Final Deliverables)

### 16.1 File cần tạo

| File | Mô tả |
|---|---|
| `Demo_End_to_End_XAI_AI_Agent.ipynb` | Notebook chính — chạy được từ đầu đến cuối |
| `demo_e2e/manifest.json` | Tự động tạo khi chạy notebook |
| `demo_e2e/sample_A/` ... | Tự động tạo khi chạy notebook |
| `demo_e2e/summary_comparison.png` | Tự động tạo khi chạy notebook |

### 16.2 Cross-sample comparison (Cell Group 6)

Cell cuối cùng trước manifest tạo một bảng so sánh 3 mẫu:

```python
# So sánh predictions của 3 samples
comparison_data = {}
for slot in ['A', 'B', 'C']:
    comparison_data[slot] = {
        'sample_id': selected_samples[slot],
        'predictions': all_predictions[slot],
        'ground_truth': all_ground_truths[slot],
        'mean_mae': all_maes[slot],
        'shap_text_pct': all_shap_text[slot],
        'agent_validated': all_agent_validated[slot],
    }

# Bar chart so sánh predictions của 3 samples trên cùng 1 figure
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, target in enumerate(DISPLAY_NAMES):
    for j, slot in enumerate(['A', 'B', 'C']):
        # Plot bars...
    axes[i].set_title(target)
plt.suptitle('Cross-Sample Comparison — Prediction vs Ground Truth')
plt.savefig(f'{DEMO_OUT}/summary_comparison.png', dpi=DEFAULT_DPI)
plt.show()
```

### 16.3 Final manifest (Cell Group 7)

```python
final_manifest = {
    'demo_version': '1.0',
    'created_at': pd.Timestamp.now().isoformat(),
    'exp_id': EXP_ID,
    'total_samples': 3,
    'samples': {
        slot: {
            'sample_id': selected_samples[slot],
            'case_type': case_types[slot],
            'completeness': completeness_scores[slot],
            'reused_from_phase6': reuse_flags[slot],
            'regenerated_artifacts': regen_lists[slot],
            'steps_completed': steps_ok[slot],
            'steps_failed': steps_failed[slot],
        }
        for slot in ['A', 'B', 'C']
    },
    'output_dir': DEMO_OUT
}
with open(f'{DEMO_OUT}/manifest.json', 'w', encoding='utf-8') as f:
    json.dump(final_manifest, f, ensure_ascii=False, indent=2)

print('\n' + '='*60)
print('DEMO HOÀN THÀNH')
print(f'Tất cả outputs đã được lưu vào: {DEMO_OUT}')
print('='*60)
```

---

## Phụ Lục A — Lưu Ý Kỹ Thuật Quan Trọng

### A.1 TimmProcessor vs HuggingFace AutoImageProcessor

`get_image_processor('swin_base_patch4_window7_224')` trong `xai/utils.py` trả về `TimmProcessor` (không phải `AutoImageProcessor`). Không tự ý thay thế. Điều này là cố ý vì Swin-B được load qua `timm`.

### A.2 Multi-image padding

Mỗi sample có tối đa 4 ảnh. Nếu num_real_images < 4, các slot còn lại được padding bằng black image tensor. Khi hiển thị, chỉ hiển thị `loaded_images[:num_real_images]`.

### A.3 Eager attention patch

`enable_eager_attention(model)` phải được gọi một lần sau khi load model. Hàm này patch RobertaSelfAttention để thay `scaled_dot_product_attention` bằng eager mode, cho phép `output_attentions=True`. Đã được gọi trong `load_model()`.

### A.4 FUSED_DIM phân chia

- Dims `0:512` = text-origin (sau `text_proj` + cross-attended).
- Dims `512:1024` = image-origin (sau `image_proj` + cross-attended).
- Cả hai đã bị ảnh hưởng lẫn nhau qua cross-attention.

### A.5 Grad-CAM target layer format

`model.image_model.encoder.norm` output shape là `[B, H, W, C]` (BHWC format), KHÔNG phải `[B, C, H, W]`. Hàm `normalize_feature_map_to_bchw()` xử lý tự động. Không cần transpose thủ công.

### A.6 LIME pseudo-classification

LIME yêu cầu output dạng classification probabilities. `ImageLimePredictFn` và `TextLimePredictFn` chuyển đổi regression score thành `[p_low, p_high]` qua sigmoid. Đây là kỹ thuật engineer, không làm thay đổi bản chất bài toán regression.

### A.7 Thứ tự index targets

```python
TARGET_NAMES[0] = 'food_score'         # index 0
TARGET_NAMES[1] = 'price_score'        # index 1
TARGET_NAMES[2] = 'atmosphere_score'   # index 2
TARGET_NAMES[3] = 'service_score'      # index 3
TARGET_NAMES[4] = 'overall_satisfaction' # index 4  ← Grad-CAM demo này
```

---

## Phụ Lục B — Kiểm Tra Nhanh Trước Khi Implement

Trước khi implement notebook, AI model (hoặc engineer) thực hiện xác nhận nhanh:

1. **`experiments/EXP_060A_bestsequential_full_configuration/best_model_train_fusion.pth` tồn tại?**  
   Nếu không → hỏi lại đường dẫn đúng.

2. **`xai_outputs/EXP_060A_bestsequential_full_configuration/` tồn tại?**  
   Nếu có → có thể reuse artifacts từ Phase 1-6.

3. **Phase 6 predictions CSV tên gì và nằm ở đâu?**  
   Kiểm tra trong `xai_outputs/.../phase6/` hoặc `xai_outputs/.../case_study/`.

4. **OPENAI_API_KEY có sẵn không?**  
   Nếu không → AI Agent section sẽ skip, còn lại vẫn chạy.

5. **Dataset CSV và image folder nằm ở đâu?**  
   Mặc định giả định `data/processed_reviews.csv` và `data/images/`.

---

*Proposal này đủ để một AI model (hoặc engineer) implement `Demo_End_to_End_XAI_AI_Agent.ipynb` mà không cần hỏi thêm về kiến trúc, luồng dữ liệu, hay yêu cầu demo. Mọi giới hạn kỹ thuật đã được ghi nhận. Mọi decision về reuse vs regenerate đã được định nghĩa rõ ràng.*
