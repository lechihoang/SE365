# ĐỀ XUẤT BÀI TRÌNH BÀY CUỐI KỲ

## Explainable Multimodal Deep Learning for Vietnamese Restaurant Review Quality Assessment

## Tổng quan bài trình bày

- **Quy mô đề xuất:** 20 slide chính, 5 Backup Slide, thời lượng khoảng 16–18 phút chưa tính Q&A.
- **Mạch kể chuyện:** Bài toán → năm đóng góp → kiến trúc hiện hành → kết quả Validation → XAI Demo → AI Agent Demo → giới hạn khoa học → lộ trình → kết luận.
- **Thông điệp trung tâm:** Đề tài không chỉ tạo một Multimodal Model, mà xây dựng một Vietnamese Multimodal Research Pipeline có khả năng truy vết từ Dataset, Label Engineering, Controlled Experiments, XAI Evidence đến Human-readable Explanation.
- **Metric chính:** Mean MAE trên năm Regression target; toàn bộ số liệu thực nghiệm đã commit trong `metrics/` là **Validation**, không phải Locked Test.
- **Nguyên tắc thiết kế:** Một slide — một thông điệp — một Visual chính; số lớn, ít chữ, màu nhất quán cho Text, Image, Fusion, XAI và AI Agent.

### Quy ước khoa học dùng chung

- Các slide kết quả phải ghi rõ: **Validation Mean MAE — lower is better**.
- Số liệu Cross-Attention và Loss hiện có thuộc kiến trúc lịch sử trước token–patch refactor; không gán các số này cho current implementation.
- Demo dùng duy nhất `SAMPLE_INDEX = 0`, `SAMPLE_ID = sample_0000`, nội dung về **bánh canh cua**.
- Chỉ đưa XAI Artifact và AI Agent report vào PowerPoint sau khi notebook đã chạy thật trên Colab; repository hiện chưa commit Runtime output.

---

## Slide 1 — Từ review nhà hàng đến Prediction có Evidence

### Thông điệp chính

Hệ thống biến một review tiếng Việt và tối đa bốn ảnh thành năm quality score, sau đó giải thích Prediction bằng Evidence có thể truy vết.

### Nội dung trên slide

- Input: một Vietnamese review + tối đa bốn ảnh.
- Output: Food, Price, Atmosphere, Service, Overall Satisfaction.
- Ba tầng tách biệt: Prediction → XAI Evidence → AI Agent report.
- Nhóm 24 — SE365 — Final Presentation.

### Visual chính

Hero diagram: review và ảnh ở trái, năm score gauge ở giữa, explanation card ở phải; dùng ít chữ và một đường Pipeline duy nhất.

### Dữ liệu / Artifact

- `Figures/Figure_1_1_Research_Value_Chain.png`
- `Figures/Figure_4_10_Prediction_XAI_AI_Agent_Sequence.png`
- Tên nhóm, thành viên, môn học, giảng viên và ngày trình bày.

### Gợi ý thuyết trình

Mở đầu bằng toàn bộ research value chain, không bắt đầu từ tên Backbone. Nhấn mạnh Prediction, Evidence và diễn đạt ngôn ngữ là ba trách nhiệm khác nhau.

---

## Slide 2 — Bài toán: Multi-target Regression từ hai nguồn Evidence không hoàn hảo

### Thông điệp chính

Text và Image bổ sung cho nhau, nhưng mỗi modality chỉ phản ánh một phần trải nghiệm nhà hàng và có thể xung đột.

### Nội dung trên slide

- Text thể hiện nhận xét, giá, dịch vụ và ngữ cảnh trải nghiệm.
- Image thể hiện món ăn, không gian và chất lượng thị giác.
- Output là vector năm score liên tục trên thang 0–10.
- Một review có nhiều ảnh; loader sử dụng tối đa bốn ảnh.

### Visual chính

Input–Output diagram dùng đúng `sample_0000`: review bánh canh cua + các ảnh thật → năm Prediction và Ground Truth.

### Dữ liệu / Artifact

- Target order từ `src/dataset.py` và `xai/config.py`.
- `data/text/test.csv` tại Runtime, dòng `SAMPLE_INDEX = 0`.
- Ảnh thật của `sample_0000` từ cache `data/image/` tại Runtime.

### Gợi ý thuyết trình

Định nghĩa đây là Multi-target Regression, không phải sentiment classification. Chỉ dùng nội dung và ảnh thật được notebook tải từ test row 0.

---

## Slide 3 — Đóng góp I: Từ Foody nhiễu đến Vietnamese Multimodal Dataset có thể huấn luyện

### Thông điệp chính

Đề tài chuyển dữ liệu web nhiều nhiễu thành Dataset tiếng Việt có cleaning trail, liên kết review–image và đơn vị huấn luyện ở mức review.

### Nội dung trên slide

- Raw crawl: **300 nhà hàng, 11.111 review, 24.599 ảnh**.
- Sau Data Cleaning: **9.946 valid review, 22.150 cleaned image record**.
- **6.082 review có ảnh**, tương đương **61,15%** coverage.
- Current preprocessing: **22.150 → 22.146 record → 6.080 review-level sample**.
- Gói dữ liệu dùng cho các experiment đã ghi nhận: **4.800 / 600 / 600**.

### Visual chính

Data funnel hai tầng: tầng trên thể hiện raw → cleaned; tầng dưới thể hiện image record → review-level grouping → experiment split.

### Dữ liệu / Artifact

- `data_raw/cleaning_report.json`
- `data_raw/multimodal_reviews.csv`
- `data_processed/reviews_clean_enhanced.csv`
- `preprocess_data.py`
- `doc/changelog.md`
- Output huấn luyện trong các notebook `EXP_*` với 150 batch × 32 hoặc 300 batch × 16.

### Gợi ý thuyết trình

22.150 là cleaned image record; 22.146 là số còn lại sau required-field filtering; 6.080 là số nhóm review đầy đủ theo current preprocessing. Các metrics đã commit được tạo trên gói experiment 6.000 mẫu; CSV snapshot và hash của split này chưa được commit.

---

## Slide 4 — Đóng góp II: Từ bốn Aspect Rating đến Overall Satisfaction có Evidence

### Thông điệp chính

Overall Satisfaction là Weak Label có khả năng truy vết, kết hợp bốn Aspect Rating với tín hiệu hài lòng tiếng Việt thay vì dùng một trung bình bất biến.

### Nội dung trên slide

- Base score: trung bình Food, Service, Atmosphere và Price.
- **14 rule category:** 8 positive, 6 negative.
- **3.263 / 9.946 review (32,81%)** có adjustment khác 0.
- Mỗi adjustment lưu rule, polarity, score và matched Evidence.
- Final label được clip trên thang **0–10**.

### Visual chính

Một formula card: bốn Aspect Rating + Vietnamese satisfaction signal → Overall Satisfaction; kèm một Evidence dương và một Evidence âm thật từ rule analysis.

### Dữ liệu / Artifact

- `data_processed/overall_satisfaction_rules.json`
- `data_processed/overall_satisfaction_rule_analysis.md`
- `data_processed/reviews_clean_enhanced.csv`

### Gợi ý thuyết trình

Giá trị nghiên cứu nằm ở provenance: mọi thay đổi label đều có thể lần ngược về rule và câu chữ kích hoạt. Việc loại Position khỏi base score làm ít nhất 0,5 điểm thay đổi ở 1.044 review, giúp target gần hơn với trải nghiệm được mô tả và quan sát.

---

## Slide 5 — Đóng góp III: Từ training run rời rạc đến Controlled Experimental Evidence

### Thông điệp chính

Đề tài tổ chức model selection thành chuỗi Controlled Sequential Ablation, cho phép quy kết tác động của từng quyết định kiến trúc.

### Nội dung trên slide

- **20 Validation metric artifact** theo cùng schema năm target.
- Năm câu hỏi: modality, Image Backbone, Text Encoder, Fusion, Loss.
- Mỗi phase giữ winner trước đó và chỉ thay một thành phần chính.
- Mean MAE là criterion chọn model; MAE, RMSE và R² theo từng target.
- Các tổ hợp mở rộng được kiểm tra riêng và chuyển xuống Backup.

### Visual chính

Controlled Sequential Ablation ladder: Baseline → Image → Text → Fusion → Loss, mỗi bậc chỉ tô sáng biến được thay đổi.

### Dữ liệu / Artifact

- `metrics/*.json`
- `Trainer.py`
- `notebook/EXP_010_text_only_xlmr_mse.ipynb` đến các notebook Phase 5.
- `Figures/Figure_5_1_Controlled_Sequential_Ablation_Phases.png`

### Gợi ý thuyết trình

Điểm mạnh không chỉ là số lượng experiment, mà là khả năng trả lời từng research question bằng một comparison có kiểm soát. Các tổ hợp mở rộng được chuyển xuống Backup để mạch chính tập trung vào các quyết định có khả năng diễn giải rõ nhất.

---

## Slide 6 — Đóng góp IV: Mở “hộp đen” Multimodal Model ở nhiều tầng

### Thông điệp chính

Năm XAI method được gắn vào đúng tầng biểu diễn để trả lời năm câu hỏi khác nhau về Prediction.

### Nội dung trên slide

- Grad-CAM: model tập trung vào vùng ảnh nào?
- PhoBERT Attention: từ nào được chú ý trong review?
- Cross-Attention: Token ↔ Patch tương tác ra sao?
- SHAP: Text-origin hay Image-origin đóng góp nhiều hơn sau Fusion?
- LIME: Prediction nhạy thế nào với perturbation cục bộ?

### Visual chính

Architecture-aligned XAI map: mỗi method nối đúng vào Image Encoder, Text Encoder, Fusion representation hoặc prediction function.

### Dữ liệu / Artifact

- `xai/gradcam_explainer.py`
- `xai/attention_explainer.py`
- `xai/shap_explainer.py`
- `xai/lime_explainer.py`
- `Figures/Figure_4_5_Multi_Level_XAI_Pipeline.png`

### Gợi ý thuyết trình

Không có một XAI method duy nhất giải thích được toàn hệ thống. Giá trị của Pipeline là triangulation: các Evidence có thể hỗ trợ, bổ sung hoặc mâu thuẫn với nhau.

---

## Slide 7 — Đóng góp V: Từ XAI Artifact kỹ thuật đến Explanation có thể kiểm tra

### Thông điệp chính

Evidence-grounded AI Agent chuyển XAI Artifact thành hai lớp báo cáo dễ sử dụng mà không tạo hoặc chỉnh sửa Prediction.

### Nội dung trên slide

- Evidence Loader nạp Artifact và ghi nhận phần còn thiếu.
- Reasoning Graph tổ chức support, conflict và missing Evidence.
- Evidence Builder nén raw output thành Top-K theo target.
- Validator kiểm tra schema, grounding, target coverage và warning.
- Report Generator tạo Customer View và Technical View.

### Visual chính

Pipeline: fixed Prediction + XAI Evidence → Reasoning Graph → GPT-4o → Validator → Customer View / Technical View.

### Dữ liệu / Artifact

- `agent/evidence_loader.py`, `agent/evidence_builder.py`
- `agent/reasoning.py`, `agent/prompt_builder.py`
- `agent/openai_client.py`, `agent/validator.py`
- `agent/report_generator.py`
- `Figures/Figure_4_7_Evidence_Based_AI_Agent_Pipeline.png`

### Gợi ý thuyết trình

AI Agent nằm sau Prediction và XAI, nên không có quyền thay đổi năm score. GPT-4o chỉ chạy khi có `OPENAI_API_KEY`; thiếu key thì notebook skip phần Agent nhưng không làm hỏng XAI Pipeline.

---

## Slide 8 — Kiến trúc Multimodal hiện hành: Bidirectional Cross-Attention token–patch

### Thông điệp chính

Current implementation thực hiện Cross-Attention hai chiều thực sự ở mức Token → Patch và Patch → Token, thay cho tương tác pooled-vector 1×1 trước đây.

### Nội dung trên slide

- PhoBERT tạo contextual token sequence.
- Swin-B tạo spatial patch sequence, aggregate qua các ảnh thật.
- Hai khối 8-head Cross-Attention với hidden dimension 512.
- Masked mean pooling → fused representation 1.024 chiều.
- Shared MLP → năm Regression score.

### Visual chính

Sơ đồ đối xứng: token sequence ở trái, patch grid ở phải, hai mũi tên Token → Patch và Patch → Token ở trung tâm.

### Dữ liệu / Artifact

- `Models/TextModel.py`
- `Models/ImageModel.py`
- `Models/CrossAttentionFusion.py`
- `Figures/Figure_4_2_Cross_Attention.png`

### Gợi ý thuyết trình

Đây là kiến trúc hiện hành trong source code và là attachment point cho improved Cross-Attention visualization. Phần kết quả tiếp theo vẫn phải được đọc theo provenance của từng metric artifact.

---

## Slide 9 — Kết quả I: Multimodal có tốt hơn single modality?

### Thông điệp chính

Multimodal Baseline tốt nhất, nhưng text vẫn là nguồn tín hiệu chính và ảnh chỉ tạo thêm một cải thiện aggregate nhỏ.

### Nội dung trên slide

- Text-only XLM-R: **1,2434 Mean MAE**.
- Image-only ConvNeXt: **1,4949**.
- Multimodal ConvNeXt + XLM-R: **1,2385**.
- Multimodal giảm **0,40%** so với Text-only.
- Multimodal giảm **17,15%** so với Image-only.

### Visual chính

Bar chart ba cột, sắp xếp từ thấp đến cao; highlight Multimodal và ghi rõ **Validation Mean MAE — lower is better**.

### Dữ liệu / Artifact

- `metrics/metrics_EXP_010_text_only_xlmr_mse.json`
- `metrics/metrics_EXP_011_image_only_convnext_meanpool_mse.json`
- `metrics/metrics_EXP_012_multimodal_convnext_xlmr_concat_mse.json`

### Gợi ý thuyết trình

Đây là câu trả lời thực nghiệm trực tiếp cho giá trị của Multimodal input. Kết luận cần cân bằng: Image có Evidence bổ sung, nhưng aggregate gain trên Text-only chỉ 0,40%.

---

## Slide 10 — Kết quả II: Swin-B là Image Backbone tốt nhất trong comparison

### Thông điệp chính

Khi giữ XLM-R, Concatenation và MSE cố định, Swin-B cho Validation Mean MAE thấp nhất trong ba Image Backbone.

### Nội dung trên slide

- **Swin-B: 1,2169 Mean MAE**.
- SigLIP: 1,2296.
- EfficientNet-B3: 1,2800.
- Swin-B tốt hơn SigLIP **1,04%**.
- Swin-B tốt hơn EfficientNet-B3 **4,93%**.

### Visual chính

Ranked horizontal bar chart; Swin-B dùng accent color, hai Backbone còn lại dùng màu trung tính; ghi **Validation Mean MAE — lower is better**.

### Dữ liệu / Artifact

- `metrics/metrics_EXP_020B_swinb_xlmr_concat_mse.json`
- `metrics/metrics_EXP_020E_siglip_xlmr_concat_mse.json`
- `metrics/metrics_EXP_020D_efficientnetb3_xlmr_concat_mse.json`

### Gợi ý thuyết trình

Chỉ Image Backbone thay đổi trong comparison này. Không đưa toàn bộ target-wise table lên slide; giữ phần đó cho Backup hoặc Q&A.

---

## Slide 11 — Kết quả III: PhoBERT tạo bước cải thiện lớn nhất

### Thông điệp chính

Text Encoder phù hợp tiếng Việt mang lại tác động lớn hơn rõ rệt so với các tinh chỉnh Fusion và Loss phía sau.

### Nội dung trên slide

- **PhoBERT: 1,1145 Mean MAE**.
- XLM-R: 1,2169.
- ViSoBERT: 1,2328.
- PhoBERT giảm **8,41%** so với XLM-R.
- PhoBERT giảm **9,59%** so với ViSoBERT.

### Visual chính

Lollipop chart ba Text Encoder với callout lớn “−8,41%”; ghi **Validation Mean MAE — lower is better**.

### Dữ liệu / Artifact

- `metrics/metrics_EXP_030B_bestimage_phobert_concat_mse.json`
- `metrics/metrics_EXP_020B_swinb_xlmr_concat_mse.json`
- `metrics/metrics_EXP_030D_bestimage_visobert_concat_mse.json`

### Gợi ý thuyết trình

Đây là kết quả mạnh nhất trong Controlled Sequential Ablation. Nó cho thấy domain-language alignment quan trọng hơn việc chỉ tăng độ phức tạp của Fusion.

---

## Slide 12 — Kết quả IV: Fusion cải thiện nhẹ, top methods gần như hòa

### Thông điệp chính

Historical Cross-Attention đạt Mean MAE thấp nhất, nhưng khoảng cách với Gated Cross-Modal chỉ 0,03%.

### Nội dung trên slide

- **Cross-Attention: 1,1079 Mean MAE**.
- Gated Cross-Modal: 1,1082; Concatenation: 1,1145.
- GMU: 1,1160; FiLM: 1,1195.
- Cross-Attention tốt hơn Concatenation **0,60%**.
- Giá trị chính của Cross-Attention là interaction modeling và XAI.

### Visual chính

Zoomed dot plot cho năm Fusion strategy, highlight khoảng cách rất nhỏ giữa hai vị trí đầu; ghi **Validation Mean MAE — lower is better**.

### Dữ liệu / Artifact

- `metrics/metrics_EXP_030B_bestimage_phobert_concat_mse.json`
- `metrics/metrics_EXP_040B_bestimage_besttext_gmu_mse.json`
- `metrics/metrics_EXP_040C_bestimage_besttext_gatedcrossmodal_mse.json`
- `metrics/metrics_EXP_041A_bestimage_besttext_film_mse.json`
- `metrics/metrics_EXP_041B_bestimage_besttext_crossattention_mse.json`

### Gợi ý thuyết trình

**Metric lịch sử được đo trước token–patch refactor; current implementation cần controlled rerun.** Không diễn giải chênh lệch 0,03% như một improvement có statistical significance.

---

## Slide 13 — Kết quả V: Loss không làm thay đổi đáng kể Mean MAE

### Thông điệp chính

MSE, Log-Cosh, Uncertainty Weighting và Huber cho aggregate performance gần như tương đương.

### Nội dung trên slide

- **MSE: 1,1079 Mean MAE** — thấp nhất theo aggregate criterion.
- Log-Cosh: 1,1080 — **Overall MAE 0,9130**.
- Uncertainty Weighting: 1,1080 — **Overall R² 0,6337**.
- Huber: 1,1085.
- Toàn bộ Mean MAE chỉ chênh **0,0007**.

### Visual chính

Compact heatmap table với ba cột Mean MAE, Overall MAE, Overall R²; ghi **Validation — lower is better cho MAE, higher is better cho R²**.

### Dữ liệu / Artifact

- `metrics/metrics_EXP_041B_bestimage_besttext_crossattention_mse.json`
- `metrics/metrics_EXP_050B_bestfusion_huber.json`
- `metrics/metrics_EXP_050C_bestfusion_logcosh.json`
- `metrics/metrics_EXP_051D_bestfusion_uncertaintyweighted.json`

### Gợi ý thuyết trình

Không so sánh raw Loss value giữa các objective khác nhau. Kết luận trung thực là không có large winner; Multi-seed mới cho biết ordering có ổn định hay không.

---

## Slide 14 — Cấu hình tốt nhất theo Validation khác Demo reference checkpoint

### Thông điệp chính

“Tốt nhất” phụ thuộc criterion: EXP_041B thắng Mean MAE, còn demo dùng EXP_060A/EXP_050C Log-Cosh làm reference checkpoint.

### Nội dung trên slide

- **Best aggregate Validation:** EXP_041B + MSE, Mean MAE **1,1079**.
- Target-wise MAE: Food 1,1024; Price 1,1728; Atmosphere 1,1743; Service 1,1756; Overall 0,9143.
- **Demo/reference:** EXP_060A sao chép checkpoint EXP_050C + Log-Cosh.
- Demo metric: Mean MAE **1,1080**; Overall MAE **0,9130**.
- `BEST_EXP_ID` và notebook cố định Runtime vào EXP_060A.

### Visual chính

Một comparison dashboard: bên trái per-target bar chart của EXP_041B; bên phải card “Demo checkpoint lineage” EXP_050C → EXP_060A → `sample_0000`.

### Dữ liệu / Artifact

- `metrics/metrics_EXP_041B_bestimage_besttext_crossattention_mse.json`
- `metrics/metrics_EXP_050C_bestfusion_logcosh.json`
- `metrics/metrics_EXP_060A_bestsequential_full_configuration.json`
- `notebook/EXP_060A_bestsequential_full_configuration.ipynb`
- `xai/config.py`
- `Success_End_to_End_XAI_AI_Agent_Sample_0000_Improved_CrossAttention.ipynb`

### Gợi ý thuyết trình

EXP_060A copy toàn bộ training artifact từ EXP_050C và được chọn làm reference cho XAI Pipeline; nó không phải winner theo Mean MAE. Checkpoint này được huấn luyện với historical Cross-Attention, dù Runtime hiện dựng current token–patch class để tạo visualization.

---

## Slide 15 — XAI Demo I: Model nhìn đâu và chú ý từ nào trong review bánh canh cua?

### Thông điệp chính

Trên cùng `sample_0000`, Grad-CAM và PhoBERT Attention cho hai góc nhìn bổ sung về vùng ảnh và từ ngữ được model sử dụng.

### Nội dung trên slide

- Cố định `SAMPLE_INDEX = 0`, `SAMPLE_ID = sample_0000`.
- Hiển thị review, ảnh thật và Prediction vs Ground Truth.
- Grad-CAM chỉ cho target **Overall Satisfaction**.
- PhoBERT Attention hiển thị word-level đã merge subword.

### Visual chính

Một dashboard ba vùng: Input + Prediction ở trái; Overall Grad-CAM overlay ở giữa; Top word-level PhoBERT Attention bar ở phải.

### Dữ liệu / Artifact

- Notebook: `Success_End_to_End_XAI_AI_Agent_Sample_0000_Improved_CrossAttention.ipynb`
- `/content/drive/MyDrive/SE365/demo_e2e/sample_0000/sample_0000_prediction.png`
- `/content/drive/MyDrive/SE365/demo_e2e/sample_0000/sample_0000_gradcam_3panel.png`
- `/content/drive/MyDrive/SE365/experiments/EXP_060A_bestsequential_full_configuration/xai/attention/sample_0000/cls_importance_word_bar.png`
- **Trạng thái:** Runtime required; các file trên chưa được commit trong repository.

### Gợi ý thuyết trình

Chỉ mô tả các vùng và từ thật xuất hiện sau khi notebook chạy thành công. Không dùng raw subword, không dùng Grad-CAM comparison năm target và không suy diễn vùng nóng thành nguyên nhân.

---

## Slide 16 — XAI Demo II: Token ↔ Patch interaction, SHAP và LIME trên cùng một sample

### Thông điệp chính

Improved Cross-Attention visualization làm rõ cả Text → Image và Image → Text, trong khi SHAP và LIME bổ sung góc nhìn contribution và local sensitivity.

### Nội dung trên slide

- **Text → Image:** important review word liên hệ với vùng ảnh nào?
- **Image → Text:** selected visual patch liên hệ mạnh với từ nào?
- SHAP: tỷ lệ Text-origin / Image-origin sau Cross-Attention.
- LIME: Local Explanation cho đúng `sample_0000`.

### Visual chính

Hai hình Cross-Attention chiếm khoảng 70% slide: `top_tokens_patch_overlay_grid.png` và `top_patches_token_rankings.png`; SHAP và LIME chỉ là hai summary card nhỏ ở hàng dưới.

### Dữ liệu / Artifact

- `/content/drive/MyDrive/SE365/experiments/EXP_060A_bestsequential_full_configuration/xai/cross_attention/sample_0000/top_tokens_patch_overlay_grid.png`
- `/content/drive/MyDrive/SE365/experiments/EXP_060A_bestsequential_full_configuration/xai/cross_attention/sample_0000/top_patches_token_rankings.png`
- `/content/drive/MyDrive/SE365/demo_e2e/sample_0000/sample_0000_shap_analysis.png`
- `/content/drive/MyDrive/SE365/demo_e2e/sample_0000/sample_0000_lime_4panel.png`
- **Trạng thái:** Runtime required; không dùng raw matrix hoặc bipartite graph cũ làm Visual chính.

### Gợi ý thuyết trình

Cross-Attention thể hiện internal association, không phải causal proof. Text-origin và Image-origin là hai nửa cross-attended representation sau Fusion, không phải hai modality thuần; LIME chỉ giải thích cục bộ sample này.

---

## Slide 17 — AI Agent Demo: Từ fixed Prediction và XAI Evidence đến hai lớp báo cáo

### Thông điệp chính

AI Agent verbalize Evidence của `sample_0000`, đồng thời giữ nguyên Prediction và công khai warning hoặc Evidence còn thiếu.

### Nội dung trên slide

- Input cố định: năm Prediction + XAI Evidence package của `sample_0000`.
- Reasoning Graph phân loại support, conflict và missing Evidence.
- Customer View: ngắn, dễ đọc; Technical View: đầy đủ provenance.
- Validator hiển thị target coverage, grounding và warning.
- Output vẫn cần grounding validation và Human Review.

### Visual chính

Split-screen report thật: Customer View bên trái; Technical View, Evidence completeness và validation warning bên phải; một dải nhỏ phía trên ghi “Prediction unchanged”.

### Dữ liệu / Artifact

- `/content/drive/MyDrive/SE365/demo_e2e/agent_reports/sample_0000/sample_0000_report_vi.md`
- `/content/drive/MyDrive/SE365/demo_e2e/agent_reports/sample_0000/sample_0000_report.json`
- `agent/reasoning.py`, `agent/validator.py`, `agent/report_generator.py`
- **Trạng thái:** Runtime/API key required; repository chưa có report thật để chụp.

### Gợi ý thuyết trình

AI Agent không generate Prediction, không sửa năm score và không bịa Evidence còn thiếu. Nếu `OPENAI_API_KEY` không tồn tại, notebook skip Agent an toàn; chỉ dùng screenshot sau khi report thật đã được tạo và kiểm tra.

---

## Slide 18 — Giới hạn khoa học của kết quả hiện tại (Limitations)

### Thông điệp chính

Các giới hạn được xác định rõ để khoanh vùng điều dự án đã chứng minh và điều còn cần Evidence mạnh hơn.

### Nội dung trên slide

- Một platform/domain; review và ảnh không phải lúc nào cũng đồng nhất.
- Weak Label có provenance nhưng vẫn chịu lỗi phrase, negation và discourse.
- Chưa có Multi-seed, Locked Test package và controlled rerun cho token–patch.
- Attention không chứng minh causality; SHAP và LIME là approximation.
- AI Agent vẫn cần grounding validation và Human Review.

### Visual chính

Limitation–impact matrix gồm năm hàng: nguồn giới hạn, tác động có thể có và phạm vi kết luận được phép.

### Dữ liệu / Artifact

- `data_processed/overall_satisfaction_rule_analysis.md`
- `notebook/EXP_050C_truecrossattn_logcosh.ipynb` — chưa chạy, không có output.
- Git history của `Models/CrossAttentionFusion.py`.
- Metadata của notebook demo: 0 executed cell, 0 output block.

### Gợi ý thuyết trình

Trình bày đây là scientific boundary, không phải danh sách lỗi. Hệ thống đã hoàn thiện về thiết kế; phần còn thiếu chủ yếu là mức độ xác nhận thực nghiệm và Human Evaluation.

---

## Slide 19 — Lộ trình tiếp theo theo ba mức ưu tiên (Future Work)

### Thông điệp chính

Lộ trình tiếp theo ưu tiên củng cố Evidence trước, đánh giá trustworthiness sau, rồi mới mở rộng hệ thống.

### Nội dung trên slide

- **P0 — Experimental Evidence:** version split; rerun token–patch; Multi-seed; Locked Test; lưu prediction, config và hash.
- **P1 — Trustworthiness:** human audit Weak Label; Human Evaluation XAI/Agent; stability, faithfulness, uncertainty và conflict detection.
- **P2 — Expansion:** thêm thành phố/platform; vision–language pretraining; target-conditioned Cross-Attention; tối ưu inference.
- Đích đến: web demo có Evidence traceability từ report về Artifact và Checkpoint.

### Visual chính

Three-horizon roadmap P0 → P1 → P2, mỗi horizon có một deliverable đo được và một completion gate.

### Dữ liệu / Artifact

- `Figures/Figure_7_1_Proposed_Deployment_Architecture.png`
- `Trainer.py`, `test.py`
- Artifact cần tạo: split manifest, `test_metrics.json`, prediction CSV, Checkpoint/config hash, Human Evaluation form.

### Gợi ý thuyết trình

P0 là điều kiện trước khi đưa ra final generalization claim. P1 kiểm tra explanation có hữu ích và faithful hay không; P2 chỉ bắt đầu khi Evidence contract đã ổn định.

---

## Slide 20 — Thông điệp kết luận: Một Research Pipeline có khả năng truy vết

### Thông điệp chính

Đề tài kết nối Dataset, Controlled Experiments, Multi-level XAI và Evidence-grounded AI Agent thành một hệ thống nghiên cứu thống nhất.

### Nội dung trên slide

- **Dataset:** 9.946 valid review, 22.150 cleaned image record.
- **Model & Experiments:** 20 Validation artifact; best recorded Mean MAE **1,1079**.
- **Multi-level XAI:** năm method từ image region đến local perturbation.
- **AI Agent:** Reasoning Graph + Customer View + Technical View có validation warning.

### Visual chính

Bốn pillar — Dataset, Model & Experiments, Multi-level XAI, Evidence-grounded AI Agent — hội tụ vào một trục “Traceable Vietnamese Multimodal Research Pipeline”.

### Dữ liệu / Artifact

- `data_raw/cleaning_report.json`
- `data_processed/overall_satisfaction_rule_analysis.md`
- `metrics/*.json`
- `Figures/Figure_1_1_Research_Value_Chain.png`

### Gợi ý thuyết trình

Kết thúc bằng system-level contribution, không bằng chênh lệch metric rất nhỏ. Câu chốt: **“Prediction có giá trị hơn khi đường đi từ dữ liệu, model, Evidence đến explanation đều có thể kiểm tra.”**

---

## Tóm tắt Metric chính

Tất cả giá trị dưới đây là **Validation Mean MAE — lower is better**; làm tròn bốn chữ số thập phân.

| Research question | Comparison chính | Kết luận | Metric source |
|---|---|---|---|
| Modality | Text 1,2434; Image 1,4949; Multimodal 1,2385 | Multimodal tốt nhất; gain so với Text là 0,40% | `metrics_EXP_010`, `011`, `012` |
| Image Backbone | Swin-B 1,2169; SigLIP 1,2296; EfficientNet-B3 1,2800 | Swin-B đứng đầu comparison | `metrics_EXP_020B`, `020E`, `020D` |
| Text Encoder | PhoBERT 1,1145; XLM-R 1,2169; ViSoBERT 1,2328 | PhoBERT giảm 8,41% so với XLM-R | `metrics_EXP_030B`, `020B`, `030D` |
| Fusion | Cross-Attention 1,1079; Gated 1,1082; Concat 1,1145; GMU 1,1160; FiLM 1,1195 | Historical Cross-Attention thấp nhất; top two gần như hòa | `metrics_EXP_041B`, `040C`, `030B`, `040B`, `041A` |
| Loss | MSE 1,1079; Log-Cosh 1,1080; Uncertainty 1,1080; Huber 1,1085 | Không có material difference về aggregate | `metrics_EXP_041B`, `050C`, `051D`, `050B` |
| Best configuration | EXP_041B: Mean 1,1079; Overall MAE 0,9143; Overall R² 0,6335 | Best recorded aggregate Validation profile | `metrics_EXP_041B_bestimage_besttext_crossattention_mse.json` |

Metric file đầy đủ nằm trong `metrics/`. Không dùng bảng Test từ các progress report cũ vì Backbone, target definition và protocol không đồng nhất với Validation series này.

---

## Danh sách Visual Asset

| Slide | Asset cần dùng | Source path | Trạng thái |
|---:|---|---|---|
| 1 | Research value chain | `Figures/Figure_1_1_Research_Value_Chain.png` | Available |
| 2 | Input–Output của `sample_0000` | Runtime `data/text/test.csv` + `data/image/` | Runtime required |
| 3 | Dataset funnel | `cleaning_report.json`, source CSV, `preprocess_data.py` | Must generate |
| 4 | Label formula + Evidence example | `overall_satisfaction_rule_analysis.md` | Must generate |
| 5 | Controlled Sequential Ablation ladder | `Figures/Figure_5_1_Controlled_Sequential_Ablation_Phases.png` | Available; simplify |
| 6 | Multi-level XAI map | `Figures/Figure_4_5_Multi_Level_XAI_Pipeline.png` | Available |
| 7 | AI Agent Evidence Pipeline | `Figures/Figure_4_7_Evidence_Based_AI_Agent_Pipeline.png` | Available |
| 8 | Current token–patch architecture | `Figures/Figure_4_2_Cross_Attention.png` | Available |
| 9 | Modality bar chart | EXP_010/011/012 metric JSON | Must generate |
| 10 | Image Backbone ranked chart | EXP_020B/020D/020E metric JSON | Must generate |
| 11 | Text Encoder lollipop chart | EXP_020B/030B/030D metric JSON | Must generate |
| 12 | Fusion dot plot | EXP_030B/040B/040C/041A/041B metric JSON | Must generate |
| 13 | Loss heatmap table | EXP_041B/050B/050C/051D metric JSON | Must generate |
| 14 | Best metric vs demo checkpoint dashboard | EXP_041B, EXP_050C, EXP_060A + notebook lineage | Must generate |
| 15 | Prediction, Grad-CAM, word Attention | `demo_e2e/sample_0000/` và `xai/.../sample_0000/` | Runtime required |
| 16 | Hai improved Cross-Attention figure + SHAP/LIME summary | Exact Runtime paths trong Slide 16 | Runtime required |
| 17 | Customer View / Technical View screenshot | `agent_reports/sample_0000/` | Runtime/API key required |
| 18 | Limitation–impact matrix | Evidence trong Limitation section | Must generate |
| 19 | P0/P1/P2 roadmap | Future Work priorities | Must generate |
| 20 | Four-pillar closing visual | Headline statistics + research value chain | Must generate |

---

## Các Backup Slide

## Backup Slide B1 — Phase 6: Component tốt không tự động tạo tổ hợp tốt nhất

### Thông điệp chính

EXP_060A vẫn dẫn đầu Phase 6; thay nhiều component cùng lúc không bảo đảm improvement cộng dồn.

### Nội dung trên slide

- EXP_060A: **1,1080** Mean MAE.
- EXP_060E: 1,1248; EXP_060C: 1,1256.
- EXP_060B: 1,2300; EXP_060D: 1,2829.
- Kết luận: component interaction quan trọng hơn phép cộng winner đơn giản.

### Visual chính

Ranked horizontal bar chart năm EXP_060 configuration; ghi **Validation Mean MAE — lower is better**.

### Dữ liệu / Artifact

- `metrics/metrics_EXP_060A_bestsequential_full_configuration.json`
- `metrics/metrics_EXP_060B_swinb_visobert_gmu_uncertainty.json`
- `metrics/metrics_EXP_060C_efficientnetb3_phobert_film_huber.json`
- `metrics/metrics_EXP_060D_efficientnetb3_visobert_crossattention_logcosh.json`
- `metrics/metrics_EXP_060E_convnext_phobert_gatedcrossmodal_autoweight.json`

### Gợi ý thuyết trình

Dùng slide này khi giảng viên hỏi về Promising Combination Validation. Không đưa vào main deck vì nó không thay đổi chuỗi quyết định chính.

---

## Backup Slide B2 — Bảng đầy đủ Validation Metric

### Thông điệp chính

Toàn bộ 20 metric artifact dùng cùng schema năm target và được sắp theo Mean MAE.

### Nội dung trên slide

| Rank | Experiment | Mean MAE | Overall MAE | Overall R² |
|---:|---|---:|---:|---:|
| 1 | EXP_041B Cross-Attention MSE | **1,1079** | 0,9143 | 0,6335 |
| 2 | EXP_050C Log-Cosh | 1,1080 | **0,9130** | 0,6312 |
| 3 | EXP_051D Uncertainty Weighting | 1,1080 | 0,9144 | **0,6337** |
| 4 | EXP_060A Best Sequential | 1,1080 | 0,9130 | 0,6312 |
| 5 | EXP_040C Gated Cross-Modal | 1,1082 | 0,9198 | 0,6309 |
| 6 | EXP_050B Huber | 1,1085 | 0,9131 | 0,6308 |
| 7 | EXP_030B PhoBERT + Concat | 1,1145 | 0,9300 | 0,6220 |
| 8 | EXP_040B GMU | 1,1160 | 0,9289 | 0,6246 |
| 9 | EXP_041A FiLM | 1,1195 | 0,9278 | 0,6215 |
| 10 | EXP_060E | 1,1248 | 0,9435 | 0,6125 |
| 11 | EXP_060C | 1,1256 | 0,9354 | 0,6123 |
| 12 | EXP_020B Swin-B | 1,2169 | 1,0667 | 0,4874 |
| 13 | EXP_020E SigLIP | 1,2296 | 1,0703 | 0,4715 |
| 14 | EXP_060B | 1,2300 | 1,0864 | 0,4608 |
| 15 | EXP_030D ViSoBERT | 1,2328 | 1,0923 | 0,4589 |
| 16 | EXP_012 Multimodal Baseline | 1,2385 | 1,0876 | 0,4461 |
| 17 | EXP_010 Text-only | 1,2434 | 1,0880 | 0,4620 |
| 18 | EXP_020D EfficientNet-B3 | 1,2800 | 1,1296 | 0,4236 |
| 19 | EXP_060D | 1,2829 | 1,1353 | 0,4259 |
| 20 | EXP_011 Image-only | 1,4949 | 1,3808 | 0,0525 |

### Visual chính

Một ranked table duy nhất; dùng màu nhấn cho best Mean MAE, best Overall MAE và best Overall R².

### Dữ liệu / Artifact

- Toàn bộ `metrics/*.json`.

### Gợi ý thuyết trình

Chỉ mở khi cần tra cứu số cụ thể. Không suy ra statistical significance từ thứ hạng một seed.

---

## Backup Slide B3 — Dataset filtering và hai phiên bản split

### Thông điệp chính

Các con số Dataset không mâu thuẫn; chúng thuộc các stage và version khác nhau của Data Pipeline.

### Nội dung trên slide

- Raw: 24.599 ảnh từ 11.111 review.
- Cleaned: 22.150 image record, 6.082 review có ảnh.
- Required-field filtering: 22.146 image record.
- Current grouping: 6.080 review-level sample → 4.864/608/608.
- Recorded experiment package: 6.000 sample → **4.800/600/600**.

### Visual chính

Versioned provenance timeline, tách rõ “current preprocessing output” và “dataset package used by recorded experiments”.

### Dữ liệu / Artifact

- `data_raw/cleaning_report.json`
- `data_raw/multimodal_reviews.csv`
- `data_processed/reviews_clean_enhanced.csv`
- `preprocess_data.py`
- `doc/changelog.md`
- Training outputs trong các experiment notebook.

### Gợi ý thuyết trình

Main deck dùng 4.800/600/600 khi mô tả experiment protocol. Current 4.864/608/608 chỉ mô tả logic mới; cả hai split CSV snapshot đều không có trong repository hiện tại. Trước khi tạo Runtime Artifact, cần đồng bộ `xai/config.py` từ `SCORE_RANGE=(1,10)` về miền label thực tế 0–10.

---

## Backup Slide B4 — Cross-Attention migration và checkpoint lineage

### Thông điệp chính

Source code hiện đã là token–patch, nhưng committed metrics và demo checkpoint vẫn có provenance từ pooled-vector Cross-Attention lịch sử.

### Nội dung trên slide

- Historical: pooled text/image vector → attention matrix 1×1.
- Current: text token sequence ↔ image patch sequence.
- Refactor commit `1f00508` ngày 27/06/2026.
- `EXP_050C_truecrossattn_logcosh.ipynb` chưa chạy, không có metric mới.
- Lineage demo: EXP_050C checkpoint → EXP_060A folder → current Runtime loader.

### Visual chính

Before/After diagram kết hợp checkpoint lineage; dùng màu khác nhau cho “training provenance” và “Runtime implementation”.

### Dữ liệu / Artifact

- Git history của `Models/CrossAttentionFusion.py`
- `Models/CrossAttentionFusion.py`
- `notebook/EXP_050C_truecrossattn_logcosh.ipynb`
- `notebook/EXP_060A_bestsequential_full_configuration.ipynb`
- `xai/utils.py`

### Gợi ý thuyết trình

`load_model()` dựng current class và load state dict với `strict=True`, nhưng điều đó không biến checkpoint cũ thành checkpoint đã train token–patch. Cần controlled rerun để gắn metric cho current architecture.

---

## Backup Slide B5 — XAI Artifact contract mở rộng cho `sample_0000`

### Thông điệp chính

Demo lưu cả lecturer-facing visual và machine-readable Evidence để mọi report có thể truy vết về Artifact gốc.

### Nội dung trên slide

- Grad-CAM: per-image Overall overlay + `metadata.json`.
- Attention: word bar + `word_importance.json` + raw tensor.
- Cross-Attention: hai improved figure + Top-K JSON + raw NPZ.
- SHAP: per-target contribution JSON + raw values; LIME: text/image output.
- Agent: JSON report + Vietnamese Markdown report + validation warning.

### Visual chính

Artifact tree cho duy nhất `sample_0000`, phân biệt PNG dùng trình bày và JSON/NPZ dùng audit.

### Dữ liệu / Artifact

- `Success_End_to_End_XAI_AI_Agent_Sample_0000_Improved_CrossAttention.ipynb`
- `xai/case_study.py`
- Runtime folders dưới `EXP_060A.../xai/` và `demo_e2e/`.

### Gợi ý thuyết trình

Chỉ hiển thị Artifact đã được tạo thật. Repository hiện mới có code sinh Artifact, chưa có Runtime package của `sample_0000`.

---

## Danh sách kiểm tra ngắn

- [ ] Đúng 20 slide chính; Slide 18, 19 và 20 lần lượt là Giới hạn, Lộ trình tiếp theo và Thông điệp kết luận.
- [ ] Toàn bộ proposal viết bằng tiếng Việt; technical term, model name, Metric và filename giữ nguyên khi cần.
- [ ] Mỗi slide có tối đa 3–5 bullet trên slide và chỉ một Visual chính.
- [ ] Tất cả result chart ghi **Validation Mean MAE — lower is better**; không gọi là Test.
- [ ] EXP_041B được gọi là best aggregate Validation; EXP_060A/EXP_050C được gọi là demo/reference checkpoint.
- [ ] Comparison EXP_060A–EXP_060E chỉ nằm trong Backup Slide B1, không nằm trong main deck.
- [ ] Dataset stage được phân biệt rõ: 24.599 → 22.150 → 22.146 → 6.080; experiment split là 4.800/600/600; score scale là 0–10.
- [ ] Demo chỉ dùng một mẫu cố định: `sample_0000`, test index 0, review bánh canh cua.
- [ ] Slide 16 dùng `top_tokens_patch_overlay_grid.png` và `top_patches_token_rankings.png` làm Visual focus.
- [ ] Không gán historical metric cho current token–patch implementation; không nói Attention chứng minh causality.
- [ ] SHAP dùng Text-origin/Image-origin sau Fusion; LIME được mô tả là Local Explanation.
- [ ] Không dùng XAI screenshot hoặc AI Agent report trước khi Runtime artifact thật được tạo và Human Review.
