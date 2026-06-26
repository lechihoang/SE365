# Đề xuất Triển khai Explainable AI (XAI) cho Hệ thống Đánh giá Nhà hàng Đa phương thức

> Tài liệu này là lộ trình thực thi XAI cho luận văn đánh giá chất lượng nhà hàng dựa trên review đa phương thức (ảnh + văn bản tiếng Việt). Mọi chi tiết kỹ thuật được trích xuất trực tiếp từ codebase thực tế.

---

## 1. Bối cảnh Dự án (Project Context)

### 1.1 Trạng thái hiện tại

Dự án đã hoàn thành **21 thí nghiệm** qua **7 phase** theo phương pháp ablation tuần tự có kiểm soát:

| Phase | Nội dung | Số thí nghiệm |
|-------|----------|----------------|
| Phase 1 | Baselines (Text-only, Image-only, Multimodal) | 3 |
| Phase 2 | Image Backbone Ablation (Swin-B, EfficientNet-B3, SigLIP) | 3 |
| Phase 3 | Text Backbone Ablation (PhoBERT, ViSoBERT) | 2 |
| Phase 4 | Fusion Upgrades (GMU, Gated Cross, FiLM, Cross-Attention) | 4 |
| Phase 5 | Loss Function (Huber, Log-Cosh, Uncertainty Weighted) | 3 |
| Phase 6 | Promising Combinations | 5 |
| Phase 7 | Multi-Seed Stability | 1 |

**XAI chưa được triển khai.** Tài liệu này đề xuất lộ trình XAI hoàn chỉnh.

### 1.2 Năm target output (chính xác từ `src/dataset.py`)

```python
factor_scores = torch.tensor([
    row['food_score'],           # index 0
    row['price_score'],          # index 1
    row['atmosphere_score'],     # index 2
    row['service_score'],        # index 3
    row['overall_satisfaction']  # index 4
], dtype=torch.float)
```

Mỗi phương pháp XAI phải được chạy riêng cho **từng target** (ví dụ: Grad-CAM cho `food_score` khác với Grad-CAM cho `atmosphere_score`).

### 1.3 Kiến trúc mô hình tốt nhất

**Best Sequential Configuration** (từ Phase 5 → Phase 6):

| Thành phần | Lựa chọn | Nguồn code |
|-----------|----------|------------|
| Image Backbone | `swin_base_patch4_window7_224` (1024-dim features) | `Models/ImageModel.py` |
| Text Backbone | `vinai/phobert-base-v2` (768-dim features) | `Models/TextModel.py` |
| Fusion | Cross-Attention (8 heads, hidden=512) | `Models/CrossAttentionFusion.py` |
| Loss | LogCosh | `Trainer.py` (class `LogCoshLoss`) |
| **Val Mean MAE** | **1.1079** | |
| **Val Overall MAE** | **0.9130** | |
| **Val R²** | **0.6335** | |

**Chi tiết kiến trúc Cross-Attention Fusion** (từ `Models/CrossAttentionFusion.py`):

```
TextModel.forward() → (factor_head_output, features)  # features: [B, 768]
ImageModel.forward() → (factor_head_output, features)  # features: [B, 1024]
     ↓                          ↓
text_proj: Linear(768→512)    image_proj: Linear(1024→512)
     ↓                          ↓
   t: [B, 1, 512]            i: [B, 1, 512]
     ↓                          ↓
cross_attn_t2i(Q=t, K=i, V=i)    cross_attn_i2t(Q=i, K=t, V=t)
     ↓                          ↓
   t_out: [B, 512]           i_out: [B, 512]
     ↓────────────────────────↓
         cat → [B, 1024]
              ↓
         head: Linear(1024→512→256→5)
              ↓
         5 regression scores
```

### 1.4 Xử lý Multi-image

`ImageModel` trong `Models/ImageModel.py` xử lý tối đa **4 ảnh** mỗi review:
- Input: `pixel_values` shape `[B, N, C, H, W]` (N tối đa 4)
- Masked average pooling dựa trên `num_images`
- Ảnh thiếu được padding bằng ảnh đen `(224, 224)`

Điều này ảnh hưởng trực tiếp đến Grad-CAM: cần quyết định chạy Grad-CAM trên từng ảnh riêng lẻ hay trên feature đã pooled.

### 1.5 Tại sao cần XAI?

1. **Yêu cầu học thuật:** Luận văn cần chứng minh model học đúng tín hiệu chất lượng nhà hàng, không phải artifact hay bias
2. **Đa phương thức:** Model kết hợp ảnh + text → cần giải thích từng modality đóng góp gì
3. **5 target khác nhau:** `food_score` và `atmosphere_score` có thể dựa vào ảnh; `service_score` và `price_score` có thể dựa vào text → XAI kiểm chứng giả thuyết này
4. **Bảo vệ luận văn:** Cần trả lời "model nhìn vào đâu?" và "text nào ảnh hưởng nhất?"

---

## 2. Mục tiêu XAI (XAI Objectives)

### Mục tiêu 1: Giải thích bằng chứng hình ảnh (Image Evidence)
Xác định vùng ảnh nào model tập trung khi dự đoán từng target. Ví dụ: `food_score` → model có nhìn vào món ăn không? `atmosphere_score` → model có nhìn vào nội thất quán không?

### Mục tiêu 2: Giải thích bằng chứng văn bản (Text Evidence)
Xác định token/từ nào trong review tiếng Việt ảnh hưởng mạnh đến từng dự đoán. Ví dụ: "ngon", "tươi" → `food_score`; "đắt", "hợp lý" → `price_score`.

### Mục tiêu 3: Đóng góp của từng modality (Modality Contribution)
Lượng hóa tỷ lệ đóng góp image vs. text tại tầng fusion. Trả lời: target nào chủ yếu dựa vào ảnh? Target nào chủ yếu dựa vào text?

### Mục tiêu 4: Giải thích cục bộ từng mẫu (Local Sample Explanation)
Với một review cụ thể, cung cấp giải thích đầy đủ: ảnh nào, từ nào, modality nào, và kết quả dự đoán → phục vụ case study trong luận văn.

### Mục tiêu 5: Hỗ trợ bảo vệ luận văn (Thesis Defense Support)
Chuẩn bị sẵn hình ảnh minh họa, bảng số liệu, và câu trả lời cho các câu hỏi phản biện thường gặp về XAI.

---

## 3. XAI cho từng Target Output

### 3.1 `food_score` (index 0)

**Câu hỏi XAI cần trả lời:**
- Grad-CAM có highlight vùng món ăn trên bàn không?
- Token nào ảnh hưởng mạnh? (kỳ vọng: "ngon", "tươi", "nhạt", "mặn", "dai", "giòn")
- Modality nào chiếm ưu thế? (kỳ vọng: cả image và text đều quan trọng -- ảnh cho thấy hình thức món ăn, text mô tả hương vị)

**Đặc thù:** Food score là target dễ giải thích nhất vì cả ảnh (hình món ăn) lẫn text (mô tả vị) đều chứa tín hiệu rõ ràng.

### 3.2 `price_score` (index 1)

**Câu hỏi XAI cần trả lời:**
- Text branch có tập trung vào từ liên quan giá không? (kỳ vọng: "đắt", "rẻ", "giá", "hợp lý", "xứng đáng", "chất lượng", con số tiền)
- Image branch đóng góp gì? (kỳ vọng: ít hơn text vì giá hiếm khi thể hiện qua ảnh, trừ khi ảnh menu/bill)
- SHAP có cho thấy text-dominant không?

**Đặc thù:** Price score kỳ vọng **text-dominant** rõ rệt. Nếu XAI cho thấy image đóng góp lớn cho `price_score`, cần kiểm tra lại -- có thể model đang dùng proxy (ví dụ: ảnh sang trọng → giá cao).

### 3.3 `atmosphere_score` (index 2)

**Câu hỏi XAI cần trả lời:**
- Grad-CAM có highlight vùng nội thất, bàn ghế, ánh sáng, trang trí không?
- Token nào ảnh hưởng? (kỳ vọng: "đẹp", "sang", "ấm cúng", "chật", "ồn", "view", "không gian")
- Image branch có đóng góp mạnh hơn text không?

**Đặc thù:** Atmosphere score kỳ vọng **image-dominant** hoặc cân bằng. Ảnh nhà hàng thể hiện trực tiếp không gian, trong khi text thường chỉ mô tả ngắn gọn.

### 3.4 `service_score` (index 3)

**Câu hỏi XAI cần trả lời:**
- Text branch tập trung vào đâu? (kỳ vọng: "nhân viên", "phục vụ", "nhanh", "chậm", "thái độ", "niềm nở", "chờ")
- Image branch đóng góp gì? (kỳ vọng: rất ít vì dịch vụ hiếm khi thể hiện qua ảnh)
- SHAP có xác nhận text-dominant không?

**Đặc thù:** Service score là target **khó giải thích nhất qua ảnh** vì dịch vụ nhà hàng gần như không thể nhìn thấy trong ảnh review. Nếu Grad-CAM cho kết quả rõ ràng cho service, đó có thể là tín hiệu giả (spurious signal). Đây là case study thú vị cho luận văn.

### 3.5 `overall_satisfaction` (index 4)

**Câu hỏi XAI cần trả lời:**
- Model kết hợp bằng chứng từ cả hai modality như thế nào?
- Attention weights tại cross-attention layer phân bổ ra sao giữa text và image?
- Overall có tương quan với trung bình 4 aspect khác không hay model học pattern riêng?

**Đặc thù:** Overall satisfaction là target tổng hợp, kỳ vọng sử dụng cả hai modality. XAI ở đây cần cho thấy model tổng hợp thông tin từ nhiều nguồn, không chỉ dựa vào một modality.

---

## 4. Các phương pháp XAI được đề xuất

### 4.1 Grad-CAM (Gradient-weighted Class Activation Mapping)

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục đích** | Tạo heatmap cho thấy vùng ảnh nào model tập trung khi dự đoán một target cụ thể |
| **Input** | Một ảnh review (hoặc từng ảnh trong nhóm tối đa 4 ảnh), checkpoint model, target index (0-4) |
| **Output** | Heatmap overlay lên ảnh gốc, ma trận raw activation values |
| **Gắn vào đâu trong kiến trúc** | Layer cuối có spatial feature map trong Swin-B **trước** global average pooling. Cụ thể: `model.image_model.encoder` — cần hook vào layer cuối của `stages[-1]` (Swin-B có 4 stages, mỗi stage có các Swin Transformer blocks). Output tại đây có shape `[B, H, W, C]` hoặc `[B, C, H, W]` tùy cấu hình timm. **Cần kiểm tra trong codebase** shape chính xác bằng cách chạy `model.image_model.encoder.stages[-1]` với input dummy |
| **Target nào giải thích** | Chạy riêng cho mỗi target: backprop gradient từ `output[:, target_idx]` về feature map |
| **Artifact kỳ vọng** | `gradcam_{sample_id}_target{idx}.png` (heatmap overlay), `gradcam_{sample_id}_target{idx}_raw.npy` (raw values) |
| **Hạn chế** | (1) Swin-B là Vision Transformer, spatial feature map có thể không rõ ràng như CNN thuần túy — cần kiểm tra resolution. (2) Multi-image: Grad-CAM chạy trên feature đã pooled sẽ mất thông tin spatial → nên chạy riêng từng ảnh trước pooling. (3) Grad-CAM chỉ cho thấy tương quan, **không chứng minh nhân quả** |

**Lưu ý quan trọng cho Swin-B:** Swin Transformer tổ chức feature theo window, spatial map có thể có resolution thấp hơn CNN (ví dụ: 7x7 cho input 224x224). Heatmap sau khi upsample vẫn hợp lệ nhưng sẽ thô hơn so với ConvNeXt.

### 4.2 Attention Visualization

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục đích** | Hiển thị token nào trong review tiếng Việt có attention weight cao, cho thấy text branch xử lý thông tin như thế nào |
| **Input** | Review text đã tokenize, checkpoint model, attention layer cần trích xuất |
| **Output** | Token-level attention heatmap, bar chart top-K token quan trọng |
| **Gắn vào đâu trong kiến trúc** | Hai vị trí: (1) **Bên trong PhoBERT** — `model.text_model.encoder` output `attentions` khi gọi với `output_attentions=True`. PhoBERT có 12 layers, mỗi layer 12 heads. Thường dùng attention ở layer cuối hoặc trung bình nhiều layer. (2) **Tại Cross-Attention fusion** — `model.cross_attn_t2i` và `model.cross_attn_i2t` (trong `CrossAttentionFusion.py`), hiện tại forward pass gọi `cross_attn_t2i(query=t, key=i, value=i)` nhưng **không lưu attention weights** (`_` bị discard ở dòng 57-58). Cần sửa code để lưu attention weights |
| **Target nào giải thích** | Attention weights trong PhoBERT là **target-agnostic** (không phụ thuộc target cụ thể). Cross-attention weights cũng target-agnostic vì nằm trước prediction head. Tuy nhiên, có thể kết hợp attention với gradient (Attention Rollout + gradient) để phân biệt theo target |
| **Artifact kỳ vọng** | `attention_{sample_id}_layer{L}.png` (heatmap), `attention_{sample_id}_topk.json` (top-K tokens + scores), `cross_attention_{sample_id}_weights.npy` (raw cross-attention weights) |
| **Hạn chế** | (1) **Attention KHONG phai la explanation** — attention weights cho thấy information flow, không chứng minh token đó là nguyên nhân dự đoán. (2) PhoBERT có nhiều layer x head → phải chọn hoặc aggregate. (3) Token-level attention cần map ngược về word-level (subword → word) vì PhoBERT tokenize tiếng Việt thành subwords |

### 4.3 SHAP (SHapley Additive exPlanations)

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục đích** | Lượng hóa đóng góp của từng feature (hoặc từng modality) vào dự đoán, dựa trên lý thuyết Shapley values |
| **Input** | Fused feature vector (hoặc modality-level features), checkpoint model, background samples (20-50 mẫu đại diện từ tập train/val) |
| **Output** | SHAP values cho từng feature dimension, summary plot, modality contribution bar chart |
| **Gắn vào đâu trong kiến trúc** | **Tại tầng fusion**, trên vector đã fused trước khi đi vào `self.head`. Trong `CrossAttentionFusion.py`, vector fused là `torch.cat([t_out.squeeze(1), i_out.squeeze(1)], dim=1)` có shape `[B, 1024]` (512 từ text branch + 512 từ image branch). Xây wrapper function: input = fused vector [1024], output = predicted score cho target cụ thể. Dùng `shap.KernelExplainer` hoặc `shap.DeepExplainer` |
| **Target nào giải thích** | Chạy riêng cho từng target: wrapper function trả về `output[:, target_idx]` |
| **Artifact kỳ vọng** | `shap_{sample_id}_target{idx}_beeswarm.png`, `shap_{sample_id}_target{idx}_waterfall.png`, `shap_modality_contribution_target{idx}.png` (tổng SHAP values cho 512 dims text vs 512 dims image), `shap_values_{sample_id}_target{idx}.npy` |
| **Hạn chế** | (1) **Chậm**: KernelExplainer với 1024 features rất chậm → nên giảm dimensionality (PCA) hoặc nhóm features thành 2 super-features: text_block [0:512] và image_block [512:1024]. (2) Background samples phải đại diện tốt cho phân phối dữ liệu. (3) SHAP giả định feature independence, nhưng cross-attention đã tạo dependency giữa text/image features |

**Chiến lược giảm chi phí tính toán:**
- **Modality-level SHAP:** Chỉ dùng 2 super-features (text_contribution, image_contribution) thay vì 1024 features riêng lẻ → cực nhanh, trả lời "image hay text quan trọng hơn?"
- **Feature-group SHAP:** Nhóm 1024 features thành 8-16 clusters bằng PCA → vừa nhanh vừa chi tiết hơn

### 4.4 LIME (Local Interpretable Model-agnostic Explanations)

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục đích** | Giải thích cục bộ bằng cách perturbation: loại bỏ từ/vùng ảnh và quan sát thay đổi dự đoán |
| **Input** | Một mẫu review (text + images), model checkpoint, target index, số lượng perturbation (khuyến nghị 1000-5000) |
| **Output** | Top-K từ quan trọng nhất (text LIME), top superpixel regions (image LIME) |
| **Gắn vào đâu trong kiến trúc** | **Model-agnostic** — gọi model end-to-end như black box. Wrapper function nhận (text, images) → score cho target cụ thể. Đối với text LIME: perturbation = bật/tắt từng từ. Đối với image LIME: perturbation = bật/tắt superpixel regions |
| **Target nào giải thích** | Chạy riêng cho từng target |
| **Artifact kỳ vọng** | `lime_text_{sample_id}_target{idx}.html` hoặc `.png`, `lime_image_{sample_id}_target{idx}.png`, `lime_weights_{sample_id}_target{idx}.json` |
| **Hạn chế** | (1) **Không ổn định**: chạy lại với random seed khác có thể cho kết quả khác → cần chạy nhiều lần hoặc fix seed. (2) **Rất chậm** nếu dùng 5000 perturbation cho mỗi mẫu. (3) Multi-image review: phải quyết định perturbation trên từng ảnh hay trên toàn bộ set ảnh. (4) Tiếng Việt tokenization: LIME mặc định split bằng space, nhưng tiếng Việt có từ ghép → cần custom tokenizer |

---

## 5. Lộ trình Triển khai theo Phase

### Phase 0: XAI Infrastructure Setup

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Chuẩn bị hạ tầng code, load model, verify forward pass, tạo utility functions |
| **Tại sao cần** | Mọi phase XAI sau đều phụ thuộc vào khả năng load checkpoint, chạy inference đúng, và lưu kết quả nhất quán |
| **Input files** | `Models/CrossAttentionFusion.py`, `Models/TextModel.py`, `Models/ImageModel.py`, `main.py`, `test.py`, `Config.py`, `src/dataset.py` |
| **Code files cần tạo** | `xai/utils.py` [proposed], `xai/__init__.py` [proposed] |
| **Expected outputs** | Loaded model object, successful inference on 1 sample, verified output shape `[1, 5]` |
| **Success criteria** | (1) Load checkpoint `best_model_train_fusion.pth` thành công. (2) Forward pass trên 1 sample cho output shape `[1, 5]`. (3) Output match với dự đoán trong `predictions.csv` (tolerance < 1e-4). (4) `model.eval()` và `torch.no_grad()` hoạt động đúng |
| **Risks** | Checkpoint path sai, thiếu dependency, mismatch kiến trúc khi load state_dict |

**Chi tiết `xai/utils.py` [proposed]:**
```python
# Các function cần có:
# - load_model(exp_dir, device) → model (eval mode)
# - load_single_sample(dataset, idx) → dict (input tensors)
# - get_prediction(model, sample, device) → np.array [5]
# - save_figure(fig, path)
# - save_raw_values(values, path)  # .npy format
# - TARGET_NAMES = ['food_score', 'price_score', 'atmosphere_score',
#                    'service_score', 'overall_satisfaction']
# - TARGET_INDICES = {name: idx for idx, name in enumerate(TARGET_NAMES)}
```

### Phase 1: Single-sample Demo

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Chạy full XAI pipeline trên **1 mẫu duy nhất** để verify toàn bộ flow |
| **Tại sao cần** | Phát hiện bugs sớm trước khi chạy hàng loạt. Tạo prototype cho báo cáo. Đảm bảo mỗi phương pháp XAI chạy đúng trên kiến trúc thực tế |
| **Input files** | Checkpoint từ best experiment, 1 mẫu từ test set |
| **Code files cần tạo** | `xai/demo_single_sample.py` [proposed] hoặc notebook `xai/demo_single_sample.ipynb` [proposed] |
| **Expected outputs** | 1 Grad-CAM heatmap, 1 Attention visualization, 1 SHAP waterfall, 1 LIME explanation — tất cả cho cùng 1 mẫu, cùng 1 target |
| **Success criteria** | Tất cả 4 phương pháp chạy không lỗi trên 1 mẫu cho 1 target |
| **Risks** | Swin-B spatial map shape không như kỳ vọng, PhoBERT attention extraction lỗi, SHAP quá chậm trên 1024 dims |

### Phase 2: Grad-CAM for Image Branch

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Triển khai Grad-CAM đầy đủ cho image branch, chạy trên tập mẫu đã chọn |
| **Tại sao cần** | Grad-CAM là phương pháp trực quan nhất, dễ giải thích trong bảo vệ luận văn, và phù hợp nhất cho image branch |
| **Input files** | Checkpoint, test set samples, `Models/ImageModel.py` |
| **Code files cần tạo** | `xai/gradcam.py` [proposed] |
| **Expected outputs** | Heatmap overlay cho mỗi sample x mỗi target, raw activation values |
| **Success criteria** | (1) Heatmap có vùng highlight hợp lý (không phải noise đều). (2) `food_score` highlight vùng món ăn. (3) `atmosphere_score` highlight vùng nội thất. (4) Chạy được cho cả 5 targets |
| **Risks** | (1) Swin-B feature map resolution thấp (7x7) → heatmap thô. (2) Multi-image: cần quyết định Grad-CAM trên ảnh nào (ảnh đầu tiên? tất cả? ảnh có contribution cao nhất?). (3) Gradient có thể vanish qua nhiều layer Swin-B |

**Quyết định thiết kế cho multi-image:**
- Chạy Grad-CAM trên **từng ảnh riêng lẻ** (trước masked average pooling)
- Forward pass từng ảnh qua `model.image_model.encoder` riêng, lấy gradient riêng
- Hiển thị heatmap cho tất cả ảnh thực của review (bỏ qua ảnh padding đen)

### Phase 3: Attention Visualization for Text Branch

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Trích xuất và hiển thị attention weights từ PhoBERT và Cross-Attention |
| **Tại sao cần** | Cho thấy text branch xử lý review tiếng Việt như thế nào, token nào được "chú ý" |
| **Input files** | Checkpoint, test set samples, `Models/TextModel.py`, `Models/CrossAttentionFusion.py` |
| **Code files cần tạo** | `xai/attention_viz.py` [proposed] |
| **Expected outputs** | Token-level attention heatmap, top-K token bar chart, cross-attention weight matrix |
| **Success criteria** | (1) Token mapping subword → word chính xác cho tiếng Việt. (2) Attention scores hợp lý (không đều hết). (3) Cross-attention weights được lưu thành công |
| **Risks** | (1) PhoBERT subword tokenization phức tạp cho tiếng Việt → cần merge subword scores. (2) Cần sửa `CrossAttentionFusion.forward()` để trả về attention weights (hiện tại bị discard ở `_, _ = self.cross_attn_t2i(...)`). (3) 12 layers x 12 heads = 144 attention matrices → phải chọn cách aggregate |

**Sửa đổi code cần thiết trong `CrossAttentionFusion.py`:**
```python
# Hiện tại (dòng 57-58):
t_out, _ = self.cross_attn_t2i(query=t, key=i, value=i)
i_out, _ = self.cross_attn_i2t(query=i, key=t, value=t)

# Cần sửa thành (cho XAI mode):
t_out, attn_t2i = self.cross_attn_t2i(query=t, key=i, value=i)
i_out, attn_i2t = self.cross_attn_i2t(query=i, key=t, value=t)
# Lưu vào self hoặc trả về
```

**Lưu ý:** Vì cross-attention trong kiến trúc hiện tại chỉ có 1 query token (CLS) attend to 1 key token (pooled feature), attention weight chỉ là scalar 1.0. Nên tập trung vào attention bên trong PhoBERT encoder thay vì cross-attention weights.

### Phase 4: Fusion-level SHAP

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Tính SHAP values tại tầng fusion để lượng hóa đóng góp text vs. image |
| **Tại sao cần** | Đây là phương pháp duy nhất cho phép so sánh **định lượng** đóng góp giữa hai modality |
| **Input files** | Checkpoint, test set, background samples từ train/val set |
| **Code files cần tạo** | `xai/shap_fusion.py` [proposed] |
| **Expected outputs** | SHAP summary plots, modality contribution charts, raw SHAP values |
| **Success criteria** | (1) SHAP values sum gần bằng `f(x) - E[f(x)]`. (2) Modality contribution ratio hợp lý cho từng target. (3) Chạy xong trong thời gian chấp nhận được (< 1 giờ cho 50 mẫu) |
| **Risks** | (1) KernelExplainer trên 1024 dims quá chậm → dùng modality-level (2 super-features). (2) DeepExplainer có thể không hỗ trợ đầy đủ kiến trúc cross-attention. (3) Background distribution phải đại diện |

**Chiến lược triển khai:**
1. **Bước 1: Modality-level SHAP** — Wrapper function nhận 2 inputs (text_feat_pooled, image_feat_pooled), trả về score → SHAP trên 2 features → nhanh, trả lời câu hỏi chính
2. **Bước 2 (nếu đủ thời gian): Feature-level SHAP** — Wrapper function nhận fused vector 1024-dim, trả về score → SHAP trên 1024 features (hoặc PCA-reduced) → chi tiết hơn

### Phase 5: LIME Local Explanation

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Tạo local explanation bằng perturbation cho text và image riêng biệt |
| **Tại sao cần** | LIME là phương pháp perturbation-based, bổ sung cho Grad-CAM (gradient-based) và SHAP (game-theory-based). Dùng làm sanity check |
| **Input files** | Checkpoint, test set samples |
| **Code files cần tạo** | `xai/lime_explain.py` [proposed] |
| **Expected outputs** | LIME text explanations (top features), LIME image explanations (superpixel highlights) |
| **Success criteria** | (1) Text LIME: top words align với domain knowledge (ví dụ: "ngon" cho food). (2) Image LIME: highlighted regions hợp lý. (3) Kết quả ổn định khi chạy lại (fix seed) |
| **Risks** | (1) Chậm nhất trong 4 phương pháp. (2) Không ổn định giữa các lần chạy. (3) Tiếng Việt word segmentation cần custom. (4) Multi-image perturbation phức tạp |

**Ưu tiên thấp hơn Phase 2-4.** Nếu thiếu thời gian, có thể bỏ qua LIME mà không ảnh hưởng đáng kể đến chất lượng luận văn.

### Phase 6: Case Study Selection

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Chọn mẫu đại diện cho các case study trong luận văn |
| **Tại sao cần** | Case study cụ thể thuyết phục hơn số liệu trung bình. Hội đồng bảo vệ thường hỏi "cho xem ví dụ cụ thể" |
| **Input files** | `predictions.csv` hoặc `test_predictions.csv` từ best experiment, XAI results từ Phase 2-5 |
| **Code files cần tạo** | `xai/select_cases.py` [proposed] |
| **Expected outputs** | Danh sách case studies với sample IDs, phân loại theo case type, lý do chọn |
| **Success criteria** | Có ít nhất 1 case cho mỗi loại: correct, high-error, text-dominant, image-dominant, conflict |
| **Risks** | Không tìm đủ mẫu đại diện → mở rộng phạm vi tìm kiếm |

### Phase 7: XAI Report Generation

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Tự động tạo báo cáo tổng hợp XAI results |
| **Tại sao cần** | Đảm bảo nhất quán format, dễ cập nhật khi chạy lại |
| **Input files** | Tất cả XAI artifacts từ Phase 2-6 |
| **Code files cần tạo** | `xai/generate_report.py` [proposed] |
| **Expected outputs** | `xai/xai_report.md` [proposed] hoặc HTML summary |
| **Success criteria** | Report chứa đầy đủ figures + analysis cho mỗi case study |
| **Risks** | Thiếu artifact → cần fallback gracefully |

### Phase 8: Thesis-ready Visualization

| Thuộc tính | Chi tiết |
|-----------|---------|
| **Mục tiêu** | Polish figures cho chất lượng xuất bản: đúng font, DPI cao, legend rõ ràng, consistent style |
| **Tại sao cần** | Figures trong luận văn phải chuyên nghiệp, consistent, dễ đọc khi in |
| **Input files** | Raw XAI values từ Phase 2-5 |
| **Code files cần tạo** | `xai/thesis_figures.py` [proposed] |
| **Expected outputs** | High-resolution figures (300 DPI), consistent color scheme, proper Vietnamese labels |
| **Success criteria** | Figures đạt chất lượng in ấn, consistent style xuyên suốt luận văn |
| **Risks** | Font tiếng Việt không render đúng trong matplotlib → cần config font |

---

## 6. Thí nghiệm nào cần giải thích?

### Khuyến nghị: KHÔNG chạy XAI cho tất cả 21 thí nghiệm

**Lý do:**
1. **Chi phí tính toán:** SHAP trên 50 mẫu x 5 targets = 250 SHAP runs/experiment. Nhân 21 = 5,250 SHAP runs → không khả thi
2. **Không cần thiết:** Mục đích XAI là giải thích **model tốt nhất**, không phải replay toàn bộ hành trình thử nghiệm
3. **Giá trị giảm dần:** XAI cho EXP_020D (EfficientNet-B3, đã bị loại ở Phase 2) không đóng góp gì cho kết luận luận văn
4. **Rủi ro overclaim:** Chạy XAI quá nhiều mà phân tích sơ sài → hội đồng sẽ hỏi "em hiểu gì từ kết quả này?"

### Danh sách thí nghiệm nên chạy XAI

#### 1. Best Multimodal Baseline — `EXP_012` (ConvNeXt + XLM-R + Concat + MSE)
- **Lý do:** Điểm so sánh "trước khi tối ưu". XAI trên baseline cho thấy model ban đầu nhìn vào đâu → so sánh với best model
- **Giá trị:** Chứng minh rằng kiến trúc tối ưu (Cross-Attention) không chỉ tốt hơn về số, mà còn "nhìn" đúng hơn

#### 2. Best Sequential Model — `EXP_050C` hoặc `EXP_060A` (Swin-B + PhoBERT + Cross-Attention + LogCosh)
- **Lý do:** Đây là model chính của luận văn, XAI đầy đủ nhất phải chạy trên model này
- **Giá trị:** Toàn bộ case study, modality analysis, thesis figures đều lấy từ đây

#### 3. Một Promising Combination khác (nếu khác biệt đáng kể)
- **Ứng viên:** `EXP_060B`, `EXP_060C`, `EXP_060D`, hoặc `EXP_060E` — chọn cấu hình có kết quả khác biệt nhất so với best sequential
- **Lý do:** So sánh XAI giữa 2 cấu hình khác nhau → chứng minh kiến trúc ảnh hưởng đến cách model "nhìn" dữ liệu
- **Giá trị:** Bổ sung depth cho phần thảo luận

#### 4. Failure cases (3-5 mẫu từ best model)
- **Nguồn:** Chọn từ `test_predictions.csv` — mẫu có absolute error > 2.0 trên bất kỳ target nào
- **Lý do:** Giải thích tại sao model sai → hiểu giới hạn
- **Giá trị:** Thể hiện tính trung thực khoa học, hội đồng đánh giá cao

#### 5. High-confidence correct cases (3-5 mẫu)
- **Nguồn:** Mẫu có absolute error < 0.3 trên tất cả 5 targets
- **Lý do:** XAI cho thấy model sử dụng bằng chứng đúng → validate
- **Giá trị:** Positive evidence cho model

#### 6. Modality-conflict cases (2-3 mẫu)
- **Nguồn:** Mẫu mà text sentiment tích cực nhưng ảnh kém (hoặc ngược lại). Cần kiểm tra trong codebase — có thể cần sentiment analysis phụ trợ để tìm
- **Lý do:** Xem model xử lý xung đột giữa hai modality như thế nào
- **Giá trị:** Case study thú vị nhất cho phần thảo luận về fusion

### Tổng kết

| Loại | Số lượng mẫu XAI | Thí nghiệm |
|------|------------------|-------------|
| Full XAI (4 phương pháp, 5 targets) | 20-30 mẫu | Best model (EXP_050C/060A) |
| Grad-CAM + Attention only | 10 mẫu | Baseline (EXP_012) |
| Comparison XAI | 5-10 mẫu | 1 promising combination |
| **Tổng cộng** | **~40-50 mẫu** | **2-3 thí nghiệm** |

---

## 7. Chiến lược Case Study

### 7.1 Các loại Case Study

#### Case Type 1: Correct Prediction (Dự đoán chính xác)
- **Định nghĩa:** Absolute error < 0.3 trên tất cả 5 targets
- **Giá trị cho luận văn:** Chứng minh model hoạt động đúng. XAI cho thấy model dùng bằng chứng hợp lý (ví dụ: nhìn vào món ăn cho food_score, đọc "ngon" cho food_score)
- **Số lượng khuyến nghị:** 2-3 mẫu

#### Case Type 2: High-error (Sai số lớn)
- **Định nghĩa:** Absolute error > 2.0 trên ít nhất 1 target
- **Giá trị cho luận văn:** Thể hiện giới hạn model. XAI giải thích tại sao model sai — có thể do ảnh không liên quan, text mâu thuẫn, hoặc thiếu bằng chứng
- **Số lượng khuyến nghị:** 2-3 mẫu

#### Case Type 3: Image-text Agreement (Hai modality đồng thuận)
- **Định nghĩa:** Cả Grad-CAM và Attention đều highlight bằng chứng phù hợp, SHAP cho thấy cả hai modality đóng góp
- **Giá trị cho luận văn:** Chứng minh fusion hoạt động đúng khi hai modality bổ trợ nhau
- **Số lượng khuyến nghị:** 1-2 mẫu

#### Case Type 4: Image-text Conflict (Xung đột giữa modality)
- **Định nghĩa:** Ảnh cho thấy tín hiệu tích cực nhưng text tiêu cực (hoặc ngược lại). Ví dụ: ảnh món ăn đẹp nhưng review viết "dở"
- **Giá trị cho luận văn:** Case study thú vị nhất — cho thấy model giải quyết xung đột như thế nào (ưu tiên text? image? trung bình hóa?)
- **Số lượng khuyến nghị:** 2-3 mẫu

#### Case Type 5: Text-dominant (Text chiếm ưu thế)
- **Định nghĩa:** SHAP cho thấy text features đóng góp > 70% vào dự đoán
- **Giá trị cho luận văn:** Xác nhận giả thuyết rằng `service_score` và `price_score` chủ yếu dựa vào text
- **Số lượng khuyến nghị:** 1-2 mẫu (1 cho service, 1 cho price)

#### Case Type 6: Image-dominant (Image chiếm ưu thế)
- **Định nghĩa:** SHAP cho thấy image features đóng góp > 60% vào dự đoán
- **Giá trị cho luận văn:** Xác nhận giả thuyết rằng `atmosphere_score` có thể dựa nhiều vào ảnh
- **Số lượng khuyến nghị:** 1-2 mẫu

#### Case Type 7: Service/Price Difficult Sample (Mẫu khó cho target text-dependent)
- **Định nghĩa:** Review không nhắc đến service/price nhưng model vẫn phải dự đoán
- **Giá trị cho luận văn:** Cho thấy model xử lý khi thiếu tín hiệu rõ ràng — có dùng context clues không? Có fallback sang image không?
- **Số lượng khuyến nghị:** 1-2 mẫu

### 7.2 Tổng kết Case Study

| Case Type | Số mẫu | Target chính | XAI methods |
|-----------|--------|-------------|-------------|
| Correct Prediction | 2-3 | Tất cả | Grad-CAM, Attention, SHAP |
| High-error | 2-3 | Target có error cao nhất | Grad-CAM, Attention, SHAP |
| Image-text Agreement | 1-2 | food_score, atmosphere_score | Grad-CAM, Attention, SHAP |
| Image-text Conflict | 2-3 | overall_satisfaction | Tất cả 4 phương pháp |
| Text-dominant | 1-2 | service_score, price_score | Attention, SHAP |
| Image-dominant | 1-2 | atmosphere_score | Grad-CAM, SHAP |
| Difficult Sample | 1-2 | service_score | Tất cả 4 phương pháp |
| **Tổng** | **10-17** | | |

---

## 8. Cấu trúc Artifact XAI

### 8.1 Folder structure đề xuất

```
experiments/
└── EXP_XXX/                           # Thí nghiệm gốc (đã có)
    ├── best_model_train_fusion.pth     # Checkpoint (đã có)
    ├── metrics.json                    # Metrics (đã có)
    ├── predictions.csv                 # Predictions (đã có)
    ├── config.yaml                     # Config (đã có)
    ├── train.log                       # Log (đã có)
    └── xai/                           # [proposed] Thư mục XAI mới
        ├── gradcam/
        │   ├── sample_{id}_target0_food.png
        │   ├── sample_{id}_target0_food_img0.png      # Per-image heatmap
        │   ├── sample_{id}_target0_food_img1.png
        │   ├── sample_{id}_target2_atmosphere.png
        │   └── ...
        ├── attention/
        │   ├── sample_{id}_phobert_layer11.png
        │   ├── sample_{id}_topk_tokens.json
        │   ├── sample_{id}_cross_attn_weights.png
        │   └── ...
        ├── shap/
        │   ├── sample_{id}_target0_waterfall.png
        │   ├── modality_contribution_target0.png
        │   ├── modality_contribution_summary.png       # All 5 targets
        │   └── ...
        ├── lime/
        │   ├── sample_{id}_target0_text.png
        │   ├── sample_{id}_target0_image.png
        │   └── ...
        ├── case_studies/
        │   ├── case_correct_001/
        │   │   ├── overview.png                        # Combined figure
        │   │   ├── metadata.json                       # Sample info + scores
        │   │   └── analysis.txt                        # Brief analysis text
        │   ├── case_higherror_001/
        │   ├── case_conflict_001/
        │   └── ...
        ├── raw_values/
        │   ├── gradcam_sample_{id}_target0.npy
        │   ├── shap_values_sample_{id}_target0.npy
        │   ├── attention_weights_sample_{id}.npy
        │   ├── lime_weights_sample_{id}_target0.json
        │   └── ...
        └── README.md                                   # [proposed] Mô tả nội dung thư mục
```

### 8.2 Quy ước đặt tên

- `sample_{id}` — ID từ dataset (index trong test set)
- `target{idx}` — index của target (0-4)
- `target0_food` — kết hợp index và tên cho dễ đọc
- `img{k}` — index ảnh trong multi-image review (0-3)
- Format ảnh: PNG cho figures, NPY cho raw values, JSON cho metadata

---

## 9. File Output XAI

### 9.1 Figures (hình ảnh minh họa)

| File | Mô tả | Phase |
|------|--------|-------|
| `gradcam_sample_{id}_target{idx}_{name}.png` | Heatmap overlay Grad-CAM lên ảnh gốc | Phase 2 |
| `gradcam_sample_{id}_target{idx}_img{k}.png` | Grad-CAM cho từng ảnh riêng (multi-image) | Phase 2 |
| `gradcam_comparison_5targets.png` | So sánh heatmap cùng 1 ảnh cho 5 targets | Phase 2 |
| `attention_sample_{id}_layer{L}_head{H}.png` | Attention heatmap cho 1 layer/head | Phase 3 |
| `attention_sample_{id}_topk_bar.png` | Bar chart top-K tokens | Phase 3 |
| `attention_sample_{id}_aggregated.png` | Attention trung bình qua heads | Phase 3 |
| `shap_sample_{id}_target{idx}_waterfall.png` | SHAP waterfall plot | Phase 4 |
| `shap_sample_{id}_target{idx}_beeswarm.png` | SHAP beeswarm (nhiều mẫu) | Phase 4 |
| `shap_modality_contribution_target{idx}.png` | Bar chart text vs image contribution | Phase 4 |
| `shap_modality_summary_all_targets.png` | Tổng hợp modality contribution cho 5 targets | Phase 4 |
| `lime_text_sample_{id}_target{idx}.png` | LIME text explanation | Phase 5 |
| `lime_image_sample_{id}_target{idx}.png` | LIME image explanation (superpixels) | Phase 5 |
| `case_study_{type}_{id}_combined.png` | Figure tổng hợp cho case study | Phase 6 |

### 9.2 Raw values (dữ liệu số)

| File | Format | Nội dung |
|------|--------|---------|
| `gradcam_raw_sample_{id}_target{idx}.npy` | NumPy | Activation map trước upsample |
| `attention_weights_sample_{id}.npy` | NumPy | Full attention tensor [layers, heads, seq, seq] |
| `cross_attn_weights_sample_{id}.npy` | NumPy | Cross-attention weights t2i và i2t |
| `shap_values_sample_{id}_target{idx}.npy` | NumPy | SHAP values cho 1024 fused dims |
| `shap_modality_summary.csv` | CSV | Tổng SHAP values text vs image, mỗi mẫu 1 dòng |
| `lime_weights_sample_{id}_target{idx}.json` | JSON | LIME feature weights + intercept |
| `case_study_metadata.json` | JSON | Thông tin mẫu, scores, error, case type |

---

## 10. Ghi chú Triển khai

### 10.1 Thiết lập chung

1. **Luôn dùng `model.eval()`** — BatchNorm và Dropout phải ở eval mode khi chạy XAI. Trong codebase hiện tại, `test.py` dòng 97 đã set `model.eval()`, XAI code nên làm tương tự

2. **Dùng checkpoint cố định** — Không train thêm, load đúng checkpoint từ `best_model_train_fusion.pth`. Sử dụng function `load_ckpt()` trong `test.py` (dòng 26-29) làm reference

3. **Cố định random seed** — Dùng `set_seed(42)` từ `main.py` (dòng 24-34) để đảm bảo reproducibility. SHAP và LIME cũng cần fix seed

4. **Một target mỗi lần** — Mỗi lần chạy XAI, chỉ backprop/tính cho 1 target index. Không tính tất cả 5 cùng lúc

5. **Giữ modality khác cố định** — Khi giải thích image branch (Grad-CAM), text input phải cố định. Khi giải thích text branch (Attention), image input phải cố định

### 10.2 Tái sử dụng code hiện có

| Utility cần dùng | Nguồn trong codebase |
|------------------|---------------------|
| Load checkpoint | `test.py` → `load_ckpt()` (dòng 26-29) |
| Build model architecture | `test.py` → dòng 69-87 (fusion type switch) |
| Image preprocessing | `main.py` → `TimmProcessor` class (dòng 13-22) |
| Text tokenization | `AutoTokenizer.from_pretrained(args.text_model_name)` |
| Dataset loading | `src/dataset.py` → `MultimodalDataset` |
| Seed setting | `main.py` → `set_seed()` (dòng 24-34) |
| Device detection | `test.py` → dòng 35-38 |
| Factor names mapping | `test.py` → dòng 125: `['food', 'price', 'atmos', 'service', 'overall']` |

### 10.3 Lưu ý cho Swin-B Grad-CAM

Swin-B trong timm (`swin_base_patch4_window7_224`) tổ chức feature maps khác CNN:
- Feature map cuối cùng có shape `[B, H*W, C]` (sequence format) hoặc `[B, C, H, W]` sau reshape
- Spatial resolution: 7x7 cho input 224x224
- Cần dùng hook vào layer đúng:
  ```python
  # Cần kiểm tra trong codebase: shape chính xác
  # Gợi ý: model.image_model.encoder.layers[-1].blocks[-1] hoặc
  #         model.image_model.encoder.norm  (trước pooling)
  ```
- Sau khi lấy activation map (7x7), upsample lên 224x224 bằng bilinear interpolation

### 10.4 Lưu ý cho PhoBERT Attention

PhoBERT (`vinai/phobert-base-v2`) tokenize tiếng Việt thành subword units:
- "nhà hàng" có thể thành `["nhà", "@@hàng"]` hoặc tương tự
- Cần merge attention scores của subwords thuộc cùng 1 từ (sum hoặc mean)
- Thêm `output_attentions=True` khi gọi encoder:
  ```python
  outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         output_attentions=True)
  attentions = outputs.attentions  # tuple of [B, num_heads, seq_len, seq_len]
  ```

### 10.5 Lưu figures VÀ raw values

**Luôn lưu cả hai:**
- Figures (PNG, 150-300 DPI) → cho luận văn và trình bày
- Raw values (NPY, JSON) → cho tái tạo, phân tích thêm, so sánh số

---

## 11. Rủi ro và Giải pháp

| # | Rủi ro | Mức độ | Giải pháp |
|---|--------|--------|-----------|
| 1 | **Attention overclaiming** — Kết luận rằng attention weight cao = từ đó "gây ra" dự đoán | Cao | Luôn ghi rõ: "Attention shows information flow, not causal importance." Dùng LIME/SHAP để cross-validate |
| 2 | **Wrong Grad-CAM layer** — Hook vào pooled embedding [B, 1024] thay vì spatial feature map [B, C, H, W] | Cao | Verify shape của activation bằng dummy forward pass. Shape phải có spatial dimensions (H, W > 1) |
| 3 | **SHAP quá chậm** — KernelExplainer trên 1024 dims, mỗi mẫu mất hàng giờ | Trung bình | Dùng modality-level SHAP (2 features) thay vì feature-level. Giới hạn background samples (20-50 mẫu) |
| 4 | **LIME không ổn định** — Kết quả thay đổi giữa các lần chạy | Trung bình | Fix seed, tăng số perturbation (≥ 1000), chạy 3 lần và lấy consensus |
| 5 | **Multimodal confounding** — Không tách được ảnh hưởng của image vs text vì cross-attention tạo dependency | Trung bình | Acknowledge trong luận văn. Dùng ablation: chạy model chỉ với text (mask image = zero), chỉ với image (mask text = PAD), so sánh prediction |
| 6 | **Wrong target** — Chạy Grad-CAM cho `food_score` nhưng backprop từ `atmosphere_score` | Thấp | Double-check `target_idx` trong mọi script. Thêm assert: `assert target_idx == 0, "Expecting food_score"` |
| 7 | **Swin-B heatmap quá thô** — Resolution 7x7 sau upsample vẫn không chi tiết | Trung bình | Chấp nhận hạn chế, ghi rõ trong luận văn. Có thể thử hook vào stage sớm hơn (resolution cao hơn nhưng abstract hơn) |
| 8 | **PhoBERT subword mapping sai** — Attention scores gán nhầm cho từ sai | Trung bình | Viết unit test cho subword-to-word mapping. Test với các câu tiếng Việt mẫu |
| 9 | **Multi-image Grad-CAM** — Không rõ chạy trên ảnh nào trong 4 ảnh | Thấp | Chạy trên tất cả ảnh thực (num_images), bỏ qua ảnh padding đen. Report kết quả per-image |
| 10 | **Out-of-memory** — SHAP/LIME cần nhiều forward passes | Trung bình | Giảm batch size, dùng CPU nếu GPU OOM, giảm perturbation count |

---

## 12. Câu hỏi Bảo vệ Luận văn

### Q1: Tại sao làm XAI sau khi training xong, không phải trong quá trình training?

**A:** XAI là phương pháp phân tích hậu huấn luyện (post-hoc analysis). Mục đích là giải thích model đã học, không phải thay đổi quá trình học. Grad-CAM, SHAP, LIME đều thiết kế để áp dụng lên model đã train xong với checkpoint cố định. Nếu chạy XAI trong training, kết quả sẽ thay đổi mỗi epoch và không có ý nghĩa giải thích.

### Q2: Tại sao không chạy XAI cho tất cả 21 thí nghiệm?

**A:** Vì (1) chi phí tính toán không khả thi — SHAP cho 21 thí nghiệm x 50 mẫu x 5 targets = 5,250 SHAP runs; (2) giá trị giảm dần — XAI trên thí nghiệm đã bị loại ở Phase 2 không đóng góp cho kết luận; (3) trọng tâm luận văn là giải thích model tốt nhất, không phải giải thích quá trình loại bỏ. Tuy nhiên, em có chạy XAI trên baseline (EXP_012) để so sánh.

### Q3: Tại sao dùng nhiều phương pháp XAI thay vì chỉ một?

**A:** Mỗi phương pháp quan sát một tầng khác nhau trong kiến trúc: Grad-CAM quan sát image features (gradient-based), Attention quan sát token interactions (architecture-based), SHAP lượng hóa đóng góp tại fusion layer (game-theory-based), LIME kiểm tra bằng perturbation (model-agnostic). Không có phương pháp nào đầy đủ một mình. Sức mạnh đến từ bằng chứng nhất quán ở nhiều tầng.

### Q4: Attention có chứng minh nhân quả (causality) không?

**A:** Không. Attention weights cho thấy cường độ tương tác (interaction strength) giữa tokens bên trong transformer, không phải bằng chứng nhân quả rằng token đó gây ra dự đoán. Đây là hạn chế đã biết (Jain & Wallace, 2019). Em sử dụng attention như bằng chứng hỗ trợ (supporting evidence) ở tầng text branch, và cross-validate với SHAP/LIME là phương pháp contribution-based/perturbation-based.

### Q5: Làm sao biết heatmap Grad-CAM có ý nghĩa?

**A:** Bằng cách kiểm tra: (1) domain consistency — heatmap cho `food_score` có highlight vùng món ăn không? (2) target specificity — heatmap khác nhau cho `food_score` vs `atmosphere_score` trên cùng 1 ảnh? (3) sanity check — nếu đổi target mà heatmap không đổi, đó là tín hiệu sai. (4) cross-validation — so sánh với LIME image: vùng highlight có overlap không?

### Q6: Làm sao đo lường image vs text contribution?

**A:** Bằng SHAP tại tầng fusion. Fused vector có 1024 dims (512 từ text branch + 512 từ image branch). Tổng absolute SHAP values của 512 dims đầu (text) vs 512 dims sau (image) cho tỷ lệ đóng góp. Bổ sung bằng ablation test: chạy model chỉ với text (zero-out image features), chỉ với image (PAD text), so sánh prediction thay đổi.

### Q7: Nếu Grad-CAM và SHAP cho kết quả mâu thuẫn thì sao?

**A:** Đây là kết quả hợp lệ, không phải lỗi. Grad-CAM và SHAP đo những thứ khác nhau: Grad-CAM đo spatial saliency tại image encoder, SHAP đo feature contribution tại fusion layer. Mâu thuẫn có thể xảy ra khi: (1) image encoder tập trung đúng vùng nhưng fusion layer vẫn ưu tiên text; (2) Grad-CAM highlight vùng rộng nhưng SHAP cho thấy image contribution thấp. Em sẽ report cả hai kết quả và phân tích lý do mâu thuẫn trong phần thảo luận.

---

## 13. Kế hoạch XAI Tối thiểu (Minimum Viable XAI)

**Điều kiện áp dụng:** Thiếu thời gian, chỉ cần đủ XAI cho bảo vệ luận văn.

### Phạm vi

| Phương pháp | Scope | Thời gian ước tính |
|-------------|-------|-------------------|
| Single-sample demo | 1 mẫu, 1 target, 4 phương pháp | 1-2 ngày |
| Grad-CAM | 10-15 mẫu, 5 targets | 1-2 ngày |
| Attention Visualization | 10-15 mẫu, aggregated attention | 1 ngày |
| SHAP (modality-level) | 20-50 mẫu, 5 targets, 2 super-features | 1-2 ngày |
| Case study selection + analysis | 3-5 case studies | 1 ngày |
| Thesis-ready figures | Polish 10-15 figures | 1 ngày |
| **Tổng** | | **6-10 ngày** |

### Deliverables tối thiểu

1. **1 Grad-CAM comparison figure:** Cùng 1 ảnh, 5 heatmaps cho 5 targets khác nhau
2. **1 Attention visualization figure:** Top-10 tokens cho 1 mẫu
3. **1 SHAP modality contribution chart:** Bar chart image vs text cho 5 targets (trung bình 20-50 mẫu)
4. **3-5 case study combined figures:** Mỗi case study 1 figure tổng hợp (ảnh gốc + heatmap + top tokens + modality ratio)
5. **1 summary table:** Tỷ lệ đóng góp trung bình image/text cho 5 targets

### Có thể bỏ qua

- LIME (thay bằng Grad-CAM + SHAP, đủ để bảo vệ)
- Feature-level SHAP (modality-level đủ cho kết luận chính)
- XAI cho baseline experiment (chỉ report số liệu, không cần hình)
- Automated report generation (làm tay cho 5 case studies)

---

## 14. Kế hoạch XAI Đầy đủ (Full XAI Plan)

**Điều kiện áp dụng:** Đủ thời gian, muốn thesis chất lượng cao.

### Phạm vi mở rộng

| Phương pháp | Scope mở rộng | Thời gian thêm |
|-------------|---------------|----------------|
| LIME text + image | 20-30 mẫu, 5 targets | 2-3 ngày |
| Dataset-level SHAP | 100+ mẫu, phân phối SHAP values | 2 ngày |
| Failure clustering | Nhóm mẫu lỗi cao theo pattern XAI | 1-2 ngày |
| Target-wise comparison | So sánh XAI patterns giữa 5 targets | 1 ngày |
| Modality-conflict analysis | Phân tích chi tiết 5-10 mẫu conflict | 2 ngày |
| Baseline vs Best comparison | XAI side-by-side cho EXP_012 vs best | 2 ngày |
| Alternative fusion comparison | XAI cho 1 promising combination khác | 1-2 ngày |
| **Tổng thêm** | | **11-17 ngày** |

### Deliverables bổ sung

1. **LIME vs Grad-CAM comparison:** So sánh vùng highlight giữa 2 phương pháp → cross-validation
2. **Dataset-level SHAP summary:** Beeswarm plot cho 100+ mẫu → pattern toàn dataset
3. **Failure mode taxonomy:** Phân loại lỗi thành nhóm: image-noise, text-ambiguous, modality-conflict, outlier-score
4. **Target-wise XAI profile:** Mỗi target có "profile" riêng: food=balanced, service=text-heavy, atmosphere=image-heavy
5. **Baseline vs Best XAI comparison:** Chứng minh Cross-Attention fusion "nhìn" tốt hơn Concat
6. **Modality necessity analysis:** Cho từng target, chạy ablation (zero-out 1 modality) → đo drop in performance
7. **Complete case study portfolio:** 10-17 case studies đầy đủ, mỗi case có figure tổng hợp 2-3 trang

### Tổng thời gian ước tính

| Kế hoạch | Thời gian | Đủ cho luận văn? |
|----------|-----------|-----------------|
| Tối thiểu | 6-10 ngày | Co, du cho bao ve |
| Đầy đủ | 17-27 ngày | Co, chat luong cao |

---

## Phụ lục A: Mapping kiến trúc → XAI attachment points

```
┌─────────────────────────────────────────────────────────────────┐
│                    CrossAttentionFusion                          │
│                                                                 │
│  ┌──────────────────┐       ┌──────────────────┐               │
│  │    TextModel      │       │   ImageModel      │              │
│  │  (PhoBERT-base)   │       │   (Swin-B)        │              │
│  │                   │       │                   │              │
│  │  encoder          │       │  encoder           │             │
│  │  ├── embeddings   │       │  ├── patch_embed   │             │
│  │  └── encoder      │       │  ├── stages[0..3]  │             │
│  │      └── layer[0] │       │  │   └── blocks[]  │◄── Grad-CAM│
│  │          ...      │       │  └── norm          │    (hook    │
│  │      └── layer[11]│◄──    │                   │     here)   │
│  │              Attention    │  forward():        │             │
│  │              Viz here     │   pixel_values     │             │
│  │                   │       │   [B,N,C,H,W]      │             │
│  │  forward():       │       │   → features       │             │
│  │   → features[B,768]       │   [B,1024]          │            │
│  └──────────────────┘       └──────────────────┘               │
│           │                          │                          │
│     text_proj                  image_proj                       │
│     768→512                    1024→512                         │
│           │                          │                          │
│     t [B,1,512]              i [B,1,512]                       │
│           │                          │                          │
│     cross_attn_t2i(Q=t,K=i,V=i)    cross_attn_i2t(Q=i,K=t,V=t)│
│           │                          │                          │
│     t_out [B,512]            i_out [B,512]                     │
│           │                          │                          │
│           └──────── cat ─────────────┘                         │
│                     │                                           │
│              fused [B,1024]  ◄──── SHAP attaches here           │
│                     │              (512 text + 512 image)       │
│                     │                                           │
│              head: MLP                                          │
│              1024→512→256→5                                     │
│                     │                                           │
│              output [B,5]    ◄──── Target selection (index 0-4) │
│              ┌──┬──┬──┬──┬──┐                                  │
│              │f │p │a │s │o │                                   │
│              └──┴──┴──┴──┴──┘                                  │
│              LIME wraps entire model as black box               │
└─────────────────────────────────────────────────────────────────┘
```

## Phụ lục B: Checklist trước khi chạy XAI

- [ ] Checkpoint `best_model_train_fusion.pth` tồn tại và load được
- [ ] `model.eval()` đã được gọi
- [ ] `set_seed(42)` đã được gọi
- [ ] Forward pass trên 1 mẫu cho output shape `[1, 5]`
- [ ] Output match với `predictions.csv` (tolerance < 1e-4)
- [ ] Xác nhận `target_idx` đúng cho target muốn giải thích
- [ ] Swin-B hook layer đã verified có spatial dimensions
- [ ] PhoBERT `output_attentions=True` hoạt động
- [ ] Background samples cho SHAP đã được chọn từ train/val set
- [ ] Thư mục output `xai/` đã được tạo
- [ ] GPU memory đủ (hoặc fallback sang CPU)
- [ ] Figures lưu ở cả PNG (visualization) lẫn NPY/JSON (raw values)
