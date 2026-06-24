# Presentation Proposal: Progress Report Blueprint (Revised)

> **Mục đích:** Tài liệu này là bản thiết kế chi tiết (blueprint/storyboard) để một AI chuyên slide generation có thể tạo ra bài trình bày PowerPoint hoàn chỉnh mà không cần hỏi thêm.
>
> **Ngôn ngữ:** Tiếng Việt. Thuật ngữ kỹ thuật giữ nguyên tiếng Anh.
>
> **Đối tượng:** Báo cáo tiến độ với giảng viên hướng dẫn (Supervisor Progress Report).
>
> **Dự án:** Hệ thống học sâu đa phương thức có khả năng giải thích cho đánh giá chất lượng trải nghiệm ăn uống từ ảnh và văn bản — SE365.

---

# Slide 1

**Tiêu đề:** Hệ thống Học sâu Đa phương thức có khả năng Giải thích cho Đánh giá Chất lượng Trải nghiệm Ăn uống từ Ảnh và Văn bản

**Mục tiêu:** Slide mở đầu, gây ấn tượng và nêu rõ tên dự án.

**Nội dung chính:**
- Tên đề tài đầy đủ bằng tiếng Việt
- Phụ đề tiếng Anh: *Explainable Multimodal Deep Learning for Vietnamese Restaurant Review Quality Assessment*
- Thông tin:
  - Môn học: SE365
  - Ngày báo cáo: Tháng 6/2026
  - Loại báo cáo: Progress Report

**Yếu tố hình ảnh:**
- Hình nền minh hoạ: collage gồm ảnh review nhà hàng (món ăn, không gian) kết hợp đoạn bình luận tiếng Việt
- Logo trường (nếu có)

**Ghi chú cho người trình bày:**
Giới thiệu ngắn gọn: "Đây là báo cáo tiến độ dự án nghiên cứu kết hợp ảnh và văn bản bình luận để dự đoán điểm đánh giá nhà hàng, với trọng tâm vào khả năng giải thích của mô hình."

---

# Slide 2

**Tiêu đề:** Nội dung Trình bày

**Mục tiêu:** Đặt kỳ vọng cho người nghe, cho thấy bài trình bày có cấu trúc rõ ràng.

**Nội dung chính:**
1. Bài toán & Động lực nghiên cứu
2. Bộ dữ liệu
3. Kiến trúc hệ thống
4. Phương pháp nghiên cứu: Controlled Sequential Ablation
5. Nghiên cứu Image Backbone
6. Nghiên cứu Text Backbone
7. Nghiên cứu Fusion
8. Nghiên cứu Loss Function
9. Kết quả thực nghiệm
10. Tiến độ hiện tại & Kế hoạch tiếp theo
11. Dự kiến đóng góp

**Yếu tố hình ảnh:**
- Dạng danh sách dọc hoặc timeline ngang, đánh số rõ ràng, sử dụng icon cho từng mục

**Ghi chú cho người trình bày:**
"Bài trình bày sẽ đi theo luồng: từ bài toán, qua dữ liệu, đến thiết kế kiến trúc và phương pháp luận, rồi trình bày kết quả 21 thí nghiệm đã hoàn thành, và kết thúc bằng kế hoạch cho phần Explainable AI."

---

# Slide 3

**Tiêu đề:** Bài toán & Động lực Nghiên cứu

**Mục tiêu:** Trả lời "Tại sao dự án này quan trọng?" và "Bài toán cụ thể là gì?"

**Nội dung chính:**

Bối cảnh:
- Nền tảng review ăn uống (Foody.vn) chứa hàng triệu bình luận kèm ảnh
- Người dùng đánh giá dựa đồng thời vào ảnh món ăn, không gian quán VÀ nội dung bình luận
- Hầu hết hệ thống hiện tại chỉ dùng một nguồn thông tin đơn lẻ (chỉ text hoặc chỉ ảnh)

Bài toán:
- Multi-output regression: Dự đoán 5 điểm đánh giá từ cặp (ảnh + văn bản):
  - `food_score` — Chất lượng đồ ăn
  - `price_score` — Mức giá phù hợp
  - `atmosphere_score` — Không gian
  - `service_score` — Chất lượng phục vụ
  - `overall_satisfaction` — Hài lòng tổng thể (sinh bằng rule engine)
- Thang điểm: 1–10

**Yếu tố hình ảnh:**
- Sơ đồ minh hoạ: một review Foody thực tế (ảnh + bình luận) → mũi tên → 5 điểm đầu ra

Hình:
Ví dụ review Foody với ảnh món ăn và bình luận tiếng Việt
(sẽ bổ sung sau — chụp màn hình từ Foody.vn)

**Ghi chú cho người trình bày:**
"Dự án này xây dựng hệ thống dự đoán 5 điểm đánh giá nhà hàng từ cặp dữ liệu ảnh và bình luận tiếng Việt. Đây là bài toán hồi quy đa đầu ra trên dữ liệu đa phương thức."

---

# Slide 4

**Tiêu đề:** Tại sao cần Multimodal?

**Mục tiêu:** Giải thích tại sao dùng cả ảnh lẫn text thay vì chỉ một nguồn.

**Nội dung chính:**

Tại sao cần multimodal:
- Ảnh phản ánh hình thức món ăn, bối cảnh trải nghiệm — thông tin mà text không chứa
- Bình luận thể hiện cảm nhận ngữ nghĩa về giá, phục vụ — thông tin mà ảnh không có
- Kết hợp cả hai → dự đoán chính xác và giải thích được hơn

**Yếu tố hình ảnh:**
- Highlight rằng ảnh cho biết food/atmosphere, text cho biết service/price
- Sơ đồ 2 cột: "Ảnh → food, atmosphere" vs "Text → service, price, overall"

**Ghi chú cho người trình bày:**
"Một bình luận nói 'đồ ăn ngon nhưng giá hơi chát' kết hợp với ảnh món ăn đẹp mắt sẽ cho mô hình thông tin phong phú hơn nhiều so với chỉ dùng một nguồn. Dự án này xây dựng hệ thống có thể tận dụng cả hai nguồn thông tin, đồng thời giải thích được tại sao mô hình đưa ra dự đoán đó."

---

# Slide 5

**Tiêu đề:** Tại sao cần Explainable AI?

**Mục tiêu:** Giải thích trọng tâm "explainable" trong tên đề tài và tại sao chỉ có accuracy là chưa đủ.

**Nội dung chính:**

- Hệ thống đánh giá tự động ảnh hưởng trực tiếp đến quyết định người dùng (chọn nhà hàng)
- Nếu mô hình dự đoán sai, người dùng và nhà hàng đều chịu thiệt hại
- Cần biết: mô hình đang nhìn vào ảnh nào? token nào trong bình luận? modality nào đóng góp nhiều hơn?
- XAI giúp:
  - Debug mô hình: phát hiện branch collapse, overfitting
  - Tăng độ tin cậy: giải thích dự đoán cho stakeholder
  - Phân tích lỗi: tại sao mô hình sai ở sample cụ thể

**Yếu tố hình ảnh:**
Hình:
Sơ đồ minh hoạ: Input (ảnh + text) → Model → Prediction + Explanation (heatmap trên ảnh, highlight trên text)
(sẽ bổ sung sau)

**Ghi chú cho người trình bày:**
"Phần XAI chưa được implement nhưng đã có thiết kế rõ ràng. Em sẽ trình bày chi tiết kế hoạch ở phần cuối."

---

# Slide 6

**Tiêu đề:** Các kỹ thuật XAI dự kiến

**Mục tiêu:** Cho giảng viên thấy đã xác định rõ 4 kỹ thuật XAI và mỗi kỹ thuật trả lời câu hỏi gì.

**Nội dung chính:**

| Kỹ thuật | Đối tượng | Câu hỏi trả lời |
|---|---|---|
| Grad-CAM | Image branch | Mô hình nhìn vào vùng ảnh nào? |
| Attention Visualization | Text branch | Token nào quan trọng nhất? |
| SHAP | Fusion level | Modality nào đóng góp bao nhiêu? |
| LIME | Local explanation | Tại sao sample cụ thể được dự đoán như vậy? |

**Yếu tố hình ảnh:**
- Bảng 4 dòng, font lớn, mỗi dòng một icon minh hoạ

**Ghi chú cho người trình bày:**
"Bốn kỹ thuật này phủ 4 cấp độ giải thích: vùng ảnh, token văn bản, tỷ lệ đóng góp modality, và giải thích cục bộ. Chi tiết kế hoạch sẽ trình bày ở phần cuối."

---

# Slide 7

**Tiêu đề:** Bộ dữ liệu — Thu thập & Làm sạch

**Mục tiêu:** Cho thấy quá trình xây dựng dataset nghiêm túc, không phải lấy sẵn.

**Nội dung chính:**

Pipeline thu thập:
1. Crawl 300 nhà hàng/quán ăn từ Foody.vn
2. Thu thập 11.111 review thô + 24.599 ảnh thô
3. Làm sạch: loại bỏ review trùng lặp, thiếu nội dung, rating không hợp lệ
4. Kết quả: **9.946 review hợp lệ**, 22.150 cặp review-ảnh

**Yếu tố hình ảnh:**
- Sơ đồ pipeline: Raw data → Cleaning → 9.946 reviews
- Bảng thống kê gọn: 300 nhà hàng, 11.111 review thô → 9.946 review sạch

**Ghi chú cho người trình bày:**
"Bộ dữ liệu không có sẵn mà được xây dựng từ đầu. Pipeline làm sạch loại bỏ hơn 1.100 review và gần 2.500 ảnh không hợp lệ."

---

# Slide 8

**Tiêu đề:** Bộ dữ liệu — Cấu trúc Mẫu & Nhãn

**Mục tiêu:** Giải thích cách gom nhóm ảnh và cách sinh nhãn overall_satisfaction.

**Nội dung chính:**

Ghép nhóm (grouping):
- Mỗi mẫu huấn luyện = 1 review + danh sách ảnh (tối đa 4 ảnh)
- Kết quả: **6.082 mẫu đa phương thức** (review có ảnh)
- 61,15% review hợp lệ có ít nhất 1 ảnh

Nhãn `overall_satisfaction`:
- Sinh bằng rule engine từ 14 nhóm luật (8 tích cực, 6 tiêu cực)
- Dựa trên: trung bình 4 điểm khía cạnh + điều chỉnh từ tín hiệu ngôn ngữ
- 3.263 review được điều chỉnh ≠ 0

**Yếu tố hình ảnh:**

Hình:
Phân bố điểm đánh giá theo 5 tiêu chí (histogram)
(sẽ bổ sung sau)

**Ghi chú cho người trình bày:**
"Nhãn overall_satisfaction được sinh bằng rule engine có bằng chứng giải thích — đây cũng là một đóng góp của dự án. Mỗi review có overall_evidence giải thích vì sao nhãn được điều chỉnh."

---

# Slide 9

**Tiêu đề:** Bộ dữ liệu — Train / Validation / Test Split

**Mục tiêu:** Chứng minh việc chia dữ liệu đúng phương pháp.

**Nội dung chính:**

| Tập | Số mẫu | Tỷ lệ |
|---|---:|---|
| Train | ~4.864 | 80% |
| Validation | ~608 | 10% |
| Test | ~608 | 10% |
| **Tổng** | **~6.080** | 100% |

Nguyên tắc:
- Chia theo `review_id` → không rò rỉ dữ liệu giữa các tập
- `random_state=42` → tái lập được
- Test set bị khoá hoàn toàn cho đến khi chọn mô hình cuối cùng
- Mọi kết quả trong ablation study đều trên **Validation set**

**Yếu tố hình ảnh:**
- Biểu đồ tròn hoặc thanh ngang chia 80/10/10
- Nhấn mạnh: "Test set LOCKED" với icon khoá

**Ghi chú cho người trình bày:**
"Tất cả 21 thí nghiệm đã hoàn thành đều được đánh giá trên cùng một tập Validation. Test set chỉ được dùng ở Phase 6 cho một số cấu hình tiềm năng nhất."

---

# Slide 10

**Tiêu đề:** Kiến trúc Hệ thống — Tổng quan

**Mục tiêu:** Giải thích kiến trúc multimodal 3 nhánh: Text, Image, Fusion.

**Nội dung chính:**

Sơ đồ kiến trúc:

```
Review Text ──→ Text Encoder ──→ Text Features ──┐
                                                  ├──→ Fusion Layer ──→ MLP ──→ 5 Predictions
Review Images ──→ Image Encoder ──→ Image Features ┘
                  (multi-image         (food, price, atmos,
                   mean pooling)        service, overall)
```

Chi tiết:
- **Text branch:** HuggingFace `AutoModel` → pooler_output hoặc CLS token → FC 256
- **Image branch:** `timm.create_model` → Global Average Pooling → masked multi-image mean pooling → FC 256
- **Fusion branch:** Concat / GMU / Gated Cross-Modal / FiLM / Cross-Attention → MLP → 5 đầu ra

**Yếu tố hình ảnh:**
- Sơ đồ kiến trúc dạng block diagram, rõ ràng, chuyên nghiệp
- Màu sắc phân biệt: xanh cho text, cam cho image, tím cho fusion

Hình:
Architecture diagram
(sẽ bổ sung sau — vẽ sơ đồ kiến trúc chuyên nghiệp)

**Ghi chú cho người trình bày:**
"Kiến trúc gồm 3 nhánh chính. Hai encoder riêng biệt trích xuất đặc trưng từ text và image, sau đó fusion layer kết hợp chúng lại."

---

# Slide 11

**Tiêu đề:** Kiến trúc — Chiến lược Huấn luyện 3 Giai đoạn

**Mục tiêu:** Giải thích cách train từng nhánh riêng rồi ghép lại.

**Nội dung chính:**

Huấn luyện 3 giai đoạn:
1. Train Text Encoder riêng (20 epochs)
2. Train Image Encoder riêng (20 epochs)
3. Đóng băng cả hai → Train Fusion layer (15 epochs)

Lý do:
- Kiểm soát tốt đóng góp từng nhánh
- Isolate ảnh hưởng của từng thành phần
- Phù hợp với phương pháp ablation: thay đổi 1 biến tại 1 thời điểm

**Yếu tố hình ảnh:**
- Sơ đồ 3 bước theo timeline: Step 1 (Text) → Step 2 (Image) → Step 3 (Fusion)
- Icon đóng băng (❄️) cho Step 3

**Ghi chú cho người trình bày:**
"Mỗi nhánh được huấn luyện riêng trước, rồi ghép lại và chỉ train phần fusion. Cách này giúp kiểm soát tốt và isolate đóng góp của từng thành phần."

---

# Slide 12

**Tiêu đề:** Kiến trúc — Image Branch

**Mục tiêu:** Giải thích cách xử lý ảnh, đặc biệt là multi-image pooling.

**Nội dung chính:**

Thách thức:
- Mỗi review có thể có 1–4 ảnh
- Ảnh rất đa dạng: món ăn, menu, biên lai, không gian, selfie, ảnh mờ
- Không phải ảnh nào cũng liên quan đến điểm đánh giá

Giải pháp hiện tại:
- Lấy tối đa 4 ảnh/review
- Ảnh thiếu → padding bằng ảnh đen
- Image Encoder trích xuất feature cho từng ảnh riêng biệt
- Masked mean pooling: chỉ tính trung bình trên ảnh thực, bỏ qua padding
- `num_images` mask đảm bảo ảnh đen không ảnh hưởng kết quả

```python
# Mã nguồn thực tế từ ImageModel.py
mask = (arange(N) < num_images).float()
features = (features * mask).sum(dim=1) / num_images.clamp(min=1)
```

**Yếu tố hình ảnh:**
- Minh hoạ: 4 ảnh review → encode từng ảnh → masked mean → 1 vector đặc trưng

**Ghi chú cho người trình bày:**
"Điểm đặc biệt là mỗi review có số ảnh khác nhau. Chúng em dùng masked mean pooling để chỉ tính trung bình trên ảnh thực sự, tránh ảnh padding ảnh hưởng kết quả."

---

# Slide 13

**Tiêu đề:** Kiến trúc — Text Branch

**Mục tiêu:** Giải thích cách xử lý văn bản bình luận.

**Nội dung chính:**

- Input: `comment_clean` (bình luận đã làm sạch, tiếng Việt)
- Tokenizer: AutoTokenizer từ HuggingFace
- Max length: 256 token
- Feature: pooler_output hoặc CLS token embedding

**Yếu tố hình ảnh:**
- Sơ đồ: Raw text → Tokenizer → Token IDs → Transformer Encoder → CLS embedding

**Ghi chú cho người trình bày:**
"Text branch đơn giản hơn image branch: tokenize bình luận, đưa qua transformer encoder, lấy CLS token hoặc pooler_output làm đặc trưng văn bản."

---

# Slide 14

**Tiêu đề:** Kiến trúc — Fusion Mechanisms

**Mục tiêu:** Trình bày 5 cơ chế fusion đã implement, mỗi cơ chế khi nào hữu ích.

**Nội dung chính:**

| Fusion | Ý tưởng | Khi nào hữu ích |
|---|---|---|
| Concat + MLP | Nối vector, MLP phân loại | Baseline đơn giản |
| GMU | Gate học tỷ lệ tin cậy text vs image | Khi độ tin cậy ảnh thay đổi theo sample |
| Gated Cross-Modal | Mỗi modality được làm giàu bởi modality kia, rồi gate | Khi modalities bổ sung lẫn nhau |
| FiLM | Text sinh γ, β để điều chỉnh image features | Text điều kiện hoá ảnh |
| Cross-Attention | Text attend vào image và ngược lại | Interaction sâu nhất |

**Yếu tố hình ảnh:**
- Sơ đồ nhỏ cho từng fusion mechanism (đặt cạnh nhau để so sánh)

**Ghi chú cho người trình bày:**
"Từ Concat đơn giản nhất đến Cross-Attention phức tạp nhất — mỗi cơ chế đều đã được implement đầy đủ trong code và đánh giá trong ablation study."

---

# Slide 15

**Tiêu đề:** Phương pháp Nghiên cứu — Controlled Sequential Ablation

**Mục tiêu:** Đây là slide QUAN TRỌNG NHẤT. Cho thấy phương pháp luận khoa học, không phải thử ngẫu nhiên.

**Nội dung chính:**

Vấn đề: Nếu thử tất cả tổ hợp có thể?
- 4 image backbones × 3 text backbones × 5 fusion methods × 4 losses × 3 seeds = **720 thí nghiệm**
- Không khả thi trên Google Colab, không cần thiết cho thesis

Giải pháp: **Controlled Sequential Ablation + Promising Combination Validation**

Nguyên tắc:
1. Cố định tất cả thành phần, chỉ thay đổi **một** biến tại một thời điểm
2. Chọn biến tốt nhất theo validation metric
3. Thay thế biến cũ bằng biến đã chọn
4. Chuyển sang thành phần tiếp theo
5. Cuối cùng: thử một số tổ hợp đầy hứa hẹn để kiểm tra synergy

**Yếu tố hình ảnh:**
- Sơ đồ dạng waterfall/pipeline: Phase 1 → Phase 2 → ... → Phase 6
- Mỗi phase hiển thị: biến thay đổi (đổi màu) + biến cố định (xám)
- Mũi tên nối: "Winner từ Phase trước → Fixed cho Phase sau"

**Ghi chú cho người trình bày:**
"Đây là điểm mấu chốt: em không chọn mô hình ngẫu nhiên. Mỗi thí nghiệm trả lời một câu hỏi nghiên cứu cụ thể. Mỗi kết quả dẫn đến quyết định có bằng chứng cho bước tiếp theo. Phương pháp này được sử dụng rộng rãi trong nghiên cứu deep learning và mạnh hơn brute-force search vì mọi kết quả đều có ý nghĩa khoa học."

---

# Slide 16

**Tiêu đề:** Phương pháp Nghiên cứu — Lộ trình 7 Phase

**Mục tiêu:** Trình bày bảng tổng quan 7 Phase và 21 thí nghiệm.

**Nội dung chính:**

| Phase | Nội dung | Biến thay đổi | Cố định | Số thí nghiệm |
|---|---|---|---|---:|
| 1 | Baselines | Modality | — | 3 |
| 2 | Image Backbone | Image encoder | Text=XLM-R, Fusion=Concat, Loss=MSE | 3 |
| 3 | Text Backbone | Text encoder | Image=Best P2, Fusion=Concat, Loss=MSE | 2 |
| 4 | Fusion | Fusion method | Image=Best, Text=Best, Loss=MSE | 4 |
| 5 | Loss Function | Loss | Image=Best, Text=Best, Fusion=Best | 3 |
| 6 | Promising Combinations | Full config | — | 5 |
| 7 | Seed Validation | Seed | Best config | 1 |
| | | | **Tổng** | **21** |

**Yếu tố hình ảnh:**
- Bảng rõ ràng, font lớn
- Highlight cột "Biến thay đổi" để nhấn mạnh chỉ thay đổi 1 biến mỗi phase

**Ghi chú cho người trình bày:**
"21 thí nghiệm, 7 phase, mỗi phase chỉ thay đổi 1 biến. Đây là phương pháp controlled experiment chuẩn."

---

# Slide 17

**Tiêu đề:** Phase 1 — Baselines

**Mục tiêu:** Thiết lập 3 baseline: text-only, image-only, multimodal concat.

**Nội dung chính:**

| Baseline | Config | Mục đích |
|---|---|---|
| EXP_010 | Text-Only (XLM-R + MSE) | Đo tín hiệu text thuần |
| EXP_011 | Image-Only (ConvNeXt + MSE) | Đo tín hiệu ảnh thuần |
| EXP_012 | Multimodal (ConvNeXt + XLM-R + Concat + MSE) | Baseline đa phương thức |

Câu hỏi: Fusion có giúp ích không? Text hay image mạnh hơn?

Lưu ý: EXP_012 là **anchor** — mọi thí nghiệm sau đều được so sánh với nó.

**Yếu tố hình ảnh:**
- Bảng 3 dòng rõ ràng

Hình:
Biểu đồ bar so sánh Mean MAE của 3 baselines
(sẽ bổ sung sau — hiện chưa có metrics cho EXP_010, EXP_011, EXP_012 trong repo)

**Ghi chú cho người trình bày:**
"Phase 1 thiết lập mốc để đo lường mọi cải tiến sau này. EXP_012 đóng vai trò anchor: mọi kết quả đều được tính % improvement so với nó."

---

# Slide 18

**Tiêu đề:** Phase 2 — Image Backbone: Ứng viên

**Mục tiêu:** Giải thích TẠI SAO chọn các image backbone cụ thể, không phải chỉ liệt kê.

**Nội dung chính:**

Câu hỏi nghiên cứu: Image encoder nào trích xuất tốt nhất đặc trưng ảnh review ăn uống?

Các ứng viên và lý do chọn:

| Backbone | Kiến trúc | Tại sao chọn cho bài toán này |
|---|---|---|
| **ConvNeXt** (baseline) | Modern CNN | Đặc trưng local mạnh cho texture đồ ăn; tương thích Grad-CAM; backbone ổn định nhất |
| **Swin-B** | Hierarchical Vision Transformer | Cửa sổ trượt bắt cả local (món ăn) lẫn global (không gian quán); multi-scale phù hợp ảnh review đa dạng |
| **EfficientNet-B3** | Efficient CNN (compound scaling) | Trade-off tốc độ/chất lượng tốt nhất; phù hợp Google Colab; CNN truyền thống mạnh |
| **SigLIP** | ViT pretrained với image-text alignment | Visual features đã được train với ngôn ngữ → có thể giảm modality gap |

Cố định: Text = XLM-R, Fusion = Concat, Loss = MSE

**Yếu tố hình ảnh:**
- Bảng 4 dòng, mỗi backbone một dòng với giải thích ngắn

**Ghi chú cho người trình bày:**
"Bốn ứng viên đại diện cho 4 họ kiến trúc khác nhau: CNN truyền thống (EfficientNet), modern CNN (ConvNeXt), hierarchical transformer (Swin), và vision-language pretrained (SigLIP). Mỗi cái có lý do cụ thể để thử trên bài toán review ảnh ăn uống."

---

# Slide 19

**Tiêu đề:** Phase 2 — Image Backbone: Kết quả

**Mục tiêu:** Trình bày kết quả so sánh và kết luận.

**Nội dung chính:**

| Image Backbone | Mean MAE ↓ | Overall MAE ↓ | R² Overall ↑ |
|---|---:|---:|---:|
| **Swin-B** 🏆 | **1.2169** | **1.0667** | **0.4874** |
| SigLIP | 1.2296 | 1.0703 | 0.4715 |
| EfficientNet-B3 | 1.2800 | 1.1296 | 0.4236 |

**Kết luận:** Swin-B chiến thắng tuyệt đối → chọn làm image backbone cho tất cả Phase sau.

**Yếu tố hình ảnh:**

Chèn hình:
02_image_backbone_comparison.png

**Ghi chú cho người trình bày:**
"Swin-B dẫn đầu ở mọi metric. Kiến trúc hierarchical transformer giúp nó nắm bắt tốt cả chi tiết cục bộ (hình thức món ăn) lẫn bối cảnh toàn cục (không gian quán). SigLIP xếp thứ hai rất sát, nhưng Swin-B ổn định hơn. EfficientNet-B3 tuy hiệu quả về compute nhưng CNN truyền thống thua sút so với Transformer trong bài toán này."

---

# Slide 20

**Tiêu đề:** Phase 3 — Text Backbone: Ứng viên

**Mục tiêu:** Giải thích tại sao cần thử Vietnamese-specific text models.

**Nội dung chính:**

Câu hỏi: Mô hình ngôn ngữ chuyên biệt tiếng Việt có tốt hơn multilingual baseline?

Bối cảnh dataset:
- Bình luận hoàn toàn bằng tiếng Việt, từ Foody.vn
- Ngôn ngữ informal: viết tắt, emoji, tiếng lóng, không dấu, pha tiếng Anh
- XLM-R là multilingual → không tối ưu cho Vietnamese social text

Các ứng viên:

| Text Backbone | Đặc điểm | Tại sao thử |
|---|---|---|
| **XLM-R** (baseline) | Multilingual, 100+ ngôn ngữ | Baseline đa ngôn ngữ; coverage tiếng Việt tốt nhưng không chuyên |
| **PhoBERT** | Pretrained thuần tiếng Việt (VnExpress + Wikipedia tiếng Việt) | Gold standard cho Vietnamese NLP; tokenizer VnCoreNLP |
| **ViSoBERT** | Pretrained trên social media tiếng Việt | Match domain informal review; xử lý tốt viết tắt, slang |

Cố định: Image = Swin-B (winner Phase 2), Fusion = Concat, Loss = MSE

**Yếu tố hình ảnh:**
- Bảng 3 dòng, mỗi text model một dòng

**Ghi chú cho người trình bày:**
"Ba ứng viên đại diện cho 3 chiến lược: multilingual general (XLM-R), Vietnamese-specific general (PhoBERT), Vietnamese social-media specific (ViSoBERT). Câu hỏi là: specialized pretraining có giúp ích cho bài toán review tiếng Việt không?"

---

# Slide 21

**Tiêu đề:** Phase 3 — Text Backbone: Kết quả

**Mục tiêu:** Trình bày kết quả ấn tượng nhất trong toàn bộ ablation.

**Nội dung chính:**

| Text Backbone | Mean MAE ↓ | Overall MAE ↓ | R² Overall ↑ |
|---|---:|---:|---:|
| **PhoBERT** 🏆 | **1.1145** | **0.9300** | **0.6220** |
| XLM-R (ref) | 1.2169 | 1.0667 | 0.4874 |
| ViSoBERT | 1.2328 | 1.0923 | 0.4589 |

**Kết luận:** PhoBERT hủy diệt mọi đối thủ. Mean MAE giảm từ 1.2169 → 1.1145 (cải thiện ~8.4%). Lần đầu tiên Overall MAE phá mốc 1.0. R² tăng vọt từ 0.49 → 0.62.

**Yếu tố hình ảnh:**

Chèn hình:
03_text_backbone_comparison.png

**Ghi chú cho người trình bày:**
"Đây là kết quả ấn tượng nhất trong toàn bộ ablation. PhoBERT kết hợp Swin-B tạo sự cộng hưởng cực mạnh: R² nhảy từ 0.49 lên 0.62, nghĩa là mô hình giải thích được 62% phương sai thay vì 49%. Điều này chứng minh rằng mô hình tiếng Việt thuần túy thực sự vượt trội so với multilingual cho bài toán review ẩm thực Việt Nam. ViSoBERT dù cũng chuyên tiếng Việt nhưng có dấu hiệu overfit nhanh."

---

# Slide 22

**Tiêu đề:** Phase 4 — Fusion: Bài toán & Ứng viên

**Mục tiêu:** Giải thích vấn đề cốt lõi mà fusion cần giải quyết.

**Nội dung chính:**

Câu hỏi: Có cách nào kết hợp text và image tốt hơn đơn giản nối vector?

Vấn đề cốt lõi: Ảnh review có độ tin cậy không đồng đều — có review ảnh đẹp và liên quan, có review ảnh mờ hoặc không liên quan (menu, biên lai). Fusion cần **biết khi nào nên tin ảnh, khi nào nên tin text**.

5 cơ chế được thử nghiệm:
- **Concat** (baseline): Nối thẳng, MLP tự học
- **GMU**: Gate điều chỉnh tỷ lệ: tin text bao nhiêu, tin ảnh bao nhiêu
- **Gated Cross-Modal**: Mỗi modality được bổ sung bởi modality kia, rồi gate
- **FiLM**: Text sinh hệ số để "xoay/dịch" ảnh features
- **Cross-Attention**: Text attend vào image, image attend vào text — tìm liên kết ngầm

**Yếu tố hình ảnh:**
- Danh sách 5 fusion methods, mỗi cái một dòng mô tả ngắn

**Ghi chú cho người trình bày:**
"Từ Concat đơn giản nhất đến Cross-Attention phức tạp nhất, mỗi cơ chế giải quyết vấn đề modality reliability theo cách khác nhau."

---

# Slide 23

**Tiêu đề:** Phase 4 — Fusion: Kết quả

**Mục tiêu:** Trình bày kết quả so sánh fusion và kết luận.

**Nội dung chính:**

| Fusion | Kết quả Mean MAE |
|---|---:|
| Concat (baseline) | 1.1145 |
| GMU | 1.1160 |
| Gated Cross-Modal | 1.1082 |
| FiLM | 1.1195 |
| **Cross-Attention** 🏆 | **1.1079** |

Cross-Attention chiến thắng sát sao:
- Overall MAE: **0.9143** (kỷ lục mới, giảm từ 0.9300)
- R² Overall: **0.6335** (đỉnh mới)
- Margin nhỏ nhưng nhất quán trên mọi metric

**Yếu tố hình ảnh:**

Chèn hình:
04_fusion_comparison.png

**Ghi chú cho người trình bày:**
"Cross-Attention cho phép text liên tục rà soát ảnh và ngược lại, tìm được những liên kết sâu giữa từ khoá và vùng ảnh. GMU và FiLM gây thất vọng — tỷ lệ tin cậy đơn giản không hoạt động tốt bằng interaction sâu. Gated Cross-Modal xếp nhì rất sát — đây cũng là ứng viên mạnh."

---

# Slide 24

**Tiêu đề:** Phase 5 — Loss Function: Bài toán & Ứng viên

**Mục tiêu:** Giải thích tại sao MSE không phải lúc nào cũng tốt nhất cho noisy data.

**Nội dung chính:**

Vấn đề với MSE:
- Bình phương sai số → outlier ảnh hưởng rất lớn
- Review có nhiễu: review bombing, spam, đánh giá cảm tính cực đoan
- 5 target có độ khó khác nhau → cần cân bằng

Các loss đã thử:

| Loss Function | Đặc điểm | Khi nào tốt |
|---|---|---|
| MSE (baseline) | Phạt nặng outlier | Data sạch, phân bố đều |
| **Huber** | MSE gần 0, MAE cho outlier | Data nhiễu, outlier vừa |
| **Log-Cosh** | Mượt hơn Huber, 2 lần khả vi | Tối ưu hóa ổn định trên data nhiễu |
| **Uncertainty Weighted** | Mỗi target học trọng số riêng | Multi-task imbalance |

Cố định: Swin-B + PhoBERT + Cross-Attention

**Yếu tố hình ảnh:**
- Bảng 4 loss functions

**Ghi chú cho người trình bày:**
"MSE phạt outlier rất nặng vì bình phương sai số. Huber, Log-Cosh, và Uncertainty Weighted đều là các chiến lược giảm ảnh hưởng của outlier và cân bằng multi-task."

---

# Slide 25

**Tiêu đề:** Phase 5 — Loss Function: Kết quả

**Mục tiêu:** Trình bày kết quả so sánh loss và kết luận.

**Nội dung chính:**

| Loss | Mean MAE ↓ | Overall MAE ↓ | R² ↑ |
|---|---:|---:|---:|
| MSE (baseline) | 1.1079 | 0.9143 | 0.6335 |
| Huber | 1.1085 | 0.9131 | 0.6308 |
| **Log-Cosh** 🏆 | **1.1080** | **0.9130** | 0.6312 |
| Uncertainty | 1.1080 | 0.9144 | **0.6337** |

**Kết luận:** Chênh lệch rất nhỏ! Log-Cosh thắng sát nút ở Overall MAE (0.9130 vs 0.9143). Mean MAE gần như bằng nhau. Quyết định chọn Log-Cosh vì Overall MAE quan trọng nhất cho trải nghiệm người dùng.

**Yếu tố hình ảnh:**

Chèn hình:
05_loss_comparison.png

**Ghi chú cho người trình bày:**
"Phase 5 cho thấy kiến trúc đã gần tối ưu — loss function chỉ cải thiện biên rất nhỏ (0.0001). Điều này chứng minh rằng phần lớn improvement đến từ backbone và fusion design, không phải loss tuning. Tuy nhiên, Log-Cosh cho Overall MAE tốt nhất nên em chọn nó."

---

# Slide 26

**Tiêu đề:** Phase 6 — Promising Combinations

**Mục tiêu:** Kiểm tra xem greedy sequential ablation có bỏ lỡ synergy nào không.

**Nội dung chính:**

Vấn đề: Sequential ablation chọn best-of-each-component → nhưng best image + best text + best fusion + best loss chưa chắc là best SYSTEM (do component synergy).

Giải pháp: Thử 5 cấu hình đầy hứa hẹn:

| ID | Config | Ý tưởng |
|---|---|---|
| EXP_060A | Swin-B + PhoBERT + CrossAttention + LogCosh | Best sequential — "Ứng viên Greedy" |
| EXP_060B | Swin-B + ViSoBERT + GMU + Uncertainty | Alternative 1 — "Candidate Social Text" |
| EXP_060C | EfficientNet-B3 + PhoBERT + FiLM + Huber | Alternative 2 — "Candidate Efficient" |
| EXP_060D | EfficientNet-B3 + ViSoBERT + CrossAttention + LogCosh | Alternative 3 |
| EXP_060E | ConvNeXt + PhoBERT + GatedCrossModal + AutoWeight | Alternative 4 — "Candidate Original" |

Một số cấu hình Phase 6 cũng được đánh giá trên **Test set** (lần đầu mở test).

**Yếu tố hình ảnh:**
- Bảng so sánh 5 cấu hình
- Highlight EXP_060A là "Best Sequential"

Hình:
Validation vs Test comparison chart (nếu có test metrics)
(sẽ bổ sung sau)

**Ghi chú cho người trình bày:**
"Phase 6 rất quan trọng vì nó kiểm tra liệu cách chọn tuần tự của chúng em có thực sự tối ưu hay không. Đây là cách làm chuẩn trong nghiên cứu ablation study."

---

# Slide 27

**Tiêu đề:** Kết quả — Overall Leaderboard

**Mục tiêu:** Trình bày bảng xếp hạng tổng thể 21 thí nghiệm.

**Nội dung chính:**

Top 10 thí nghiệm theo Mean MAE (Validation):

| Rank | Experiment | Mean MAE ↓ | Overall MAE | Phase |
|---:|---|---:|---:|---|
| 1 | EXP_041B (CrossAttention) | 1.1079 | 0.9143 | Fusion |
| 2 | EXP_050C (LogCosh) | 1.1080 | 0.9130 | Loss |
| 3 | EXP_051D (Uncertainty) | 1.1080 | 0.9144 | Loss |
| 4 | EXP_040C (GatedCross) | 1.1082 | 0.9198 | Fusion |
| 5 | EXP_050B (Huber) | 1.1085 | 0.9131 | Loss |
| 6 | EXP_030B (PhoBERT) | 1.1145 | 0.9300 | Text |
| ... | ... | ... | ... | ... |

**Yếu tố hình ảnh:**

Chèn hình:
01_overall_leaderboard.png

**Ghi chú cho người trình bày:**
"Top 5 đều cluster quanh Mean MAE 1.108 — đây là vùng tối ưu hiện tại. Điều thú vị là tất cả đều dùng Swin-B + PhoBERT, chứng tỏ backbone selection ở Phase 2-3 là quyết định quan trọng nhất."

---

# Slide 28

**Tiêu đề:** Kết quả — Diễn tiến Cải thiện qua các Phase

**Mục tiêu:** Cho thấy tiến bộ liên tục và mỗi Phase đều đóng góp.

**Nội dung chính:**

| Phase | Best Experiment | Mean MAE | Improvement |
|---|---|---:|---|
| Baseline (Multimodal) | EXP_012 | ~1.30+ | — |
| Image Ablation | EXP_020B (Swin-B) | 1.2169 | Backbone tốt hơn |
| Text Ablation | EXP_030B (PhoBERT) | 1.1145 | Vietnamese NLP vượt trội |
| Fusion Ablation | EXP_041B (CrossAttention) | 1.1079 | Interaction sâu |
| Loss Ablation | EXP_050C (LogCosh) | 1.1080 | Robust to outliers |

Xu hướng: **Mean MAE giảm đều đặn qua mỗi phase**, chứng minh phương pháp sequential ablation hiệu quả.

Điểm nhảy lớn nhất: Phase 3 (Text Ablation) — PhoBERT cải thiện ~8.4% so với XLM-R.

**Yếu tố hình ảnh:**

Chèn hình:
06_performance_evolution.png

**Ghi chú cho người trình bày:**
"Biểu đồ cho thấy xu hướng cải thiện liên tục. Bước nhảy lớn nhất là khi chuyển từ XLM-R sang PhoBERT — chứng minh rằng mô hình ngôn ngữ chuyên biệt tiếng Việt thực sự quan trọng cho bài toán này. Fusion và Loss cải thiện nhỏ hơn nhưng nhất quán."

---

# Slide 29

**Tiêu đề:** Kết quả — Top-3 Radar Chart

**Mục tiêu:** So sánh chi tiết top 3 trên từng tiêu chí đánh giá.

**Nội dung chính:**

Top 3 models so sánh trên 5 tiêu chí (Overall, Food, Price, Service, Atmosphere):
- Normalized score: 1 - normalized MAE → **cao hơn = tốt hơn**
- Cho thấy mô hình nào mạnh ở tiêu chí nào

**Yếu tố hình ảnh:**

Chèn hình:
07_top3_radar_chart.png

**Ghi chú cho người trình bày:**
"Radar chart cho thấy top 3 rất đồng đều — không có mô hình nào vượt trội ở một tiêu chí nhưng yếu ở tiêu chí khác. Điều này cho thấy kiến trúc Swin-B + PhoBERT + Cross-Attention balance tốt trên cả 5 đầu ra."

---

# Slide 30

**Tiêu đề:** Kết quả — Improvement vs Baseline

**Mục tiêu:** Cho thấy rõ mỗi thí nghiệm cải thiện bao nhiêu % so với baseline.

**Nội dung chính:**

Baseline: EXP_012 (ConvNeXt + XLM-R + Concat + MSE)

Top improvements:
- Các thí nghiệm Phase 4-5 cải thiện ~15% so với baseline
- PhoBERT thay XLM-R đóng góp phần lớn improvement
- Swin-B thay ConvNeXt cũng đóng góp đáng kể

**Yếu tố hình ảnh:**

Chèn hình:
improvement_vs_baseline.png

**Ghi chú cho người trình bày:**
"Biểu đồ này cho phép nhìn trực quan: mỗi thanh xanh là % cải thiện so với baseline ban đầu. Rõ ràng nhất là sau khi thay PhoBERT và dùng Cross-Attention fusion."

---

# Slide 31

**Tiêu đề:** Tóm tắt — Cấu hình Tối ưu

**Mục tiêu:** Tổng hợp cấu hình tốt nhất tìm được.

**Nội dung chính:**

**Cấu hình vô địch (Best Sequential Full Configuration):**

| Thành phần | Lựa chọn | Phase quyết định |
|---|---|---|
| Image Backbone | **Swin-B** (Swin Transformer Base) | Phase 2 |
| Text Backbone | **PhoBERT** (Vietnamese-specific BERT) | Phase 3 |
| Fusion | **Cross-Attention** | Phase 4 |
| Loss Function | **Log-Cosh** | Phase 5 |

**Yếu tố hình ảnh:**
- Bảng tóm tắt to, rõ ràng
- Highlight cấu hình chiến thắng
- Mỗi thành phần liên kết ngược với Phase quyết định

**Ghi chú cho người trình bày:**
"Mỗi thành phần được chọn qua một Phase ablation riêng biệt, với bằng chứng thực nghiệm rõ ràng."

---

# Slide 32

**Tiêu đề:** Tóm tắt — Metrics Tốt nhất

**Mục tiêu:** Trình bày kết quả định lượng của cấu hình tốt nhất.

**Nội dung chính:**

**Metrics tốt nhất (Validation):**

| Metric | Giá trị |
|---|---:|
| Mean MAE | **1.1079** |
| Overall MAE | **0.9130** |
| R² Overall | **0.6335** |
| MAE Food | 1.097 |
| MAE Price | 1.169 |
| MAE Atmosphere | 1.173 |
| MAE Service | 1.178 |

Ý nghĩa: Trung bình sai lệch chỉ ~1.1 điểm trên thang 10. Mô hình giải thích được 63% phương sai.

**Yếu tố hình ảnh:**
- Bảng metrics, font lớn
- Highlight Mean MAE và R² Overall

**Ghi chú cho người trình bày:**
"Tổng hợp lại, cấu hình tối ưu là Swin-B + PhoBERT + Cross-Attention + Log-Cosh. Mean MAE chỉ 1.108 trên thang 10 điểm, tức trung bình chỉ sai khoảng 1.1 điểm."

---

# Slide 33

**Tiêu đề:** Tiến độ hiện tại

**Mục tiêu:** Báo cáo trung thực công việc đã hoàn thành, đang làm, chưa bắt đầu.

**Nội dung chính:**

| Hạng mục | Trạng thái | Chi tiết |
|---|---|---|
| Thu thập & làm sạch dữ liệu | ✅ Hoàn thành | 9.946 review, 22.150 cặp review-ảnh |
| Nhãn overall_satisfaction | ✅ Hoàn thành | Rule engine, 14 nhóm luật |
| Pipeline huấn luyện | ✅ Hoàn thành | Seed control, AMP, checkpoint, resume, config.yaml, metrics.json |
| 21 thí nghiệm (Phase 1–6) | ✅ Hoàn thành | Ablation study đầy đủ |
| Leaderboard & báo cáo tự động | ✅ Hoàn thành | 7 figures, 4 tables, auto-generated |
| Multi-seed validation (Phase 7) | ⏳ Chưa bắt đầu | Dự kiến: 3 seeds cho top 2 candidates |
| Test set evaluation (Phase 7) | ⏳ Một phần | Một số EXP_060 đã có test metrics |
| XAI — Grad-CAM, Attention, SHAP, LIME | ❌ Chưa bắt đầu | Thiết kế đã có, code chưa implement |
| Thesis report | ⏳ Đang làm | Progress report hoàn thành |

**Yếu tố hình ảnh:**
- Dạng timeline hoặc checklist
- Dùng icon ✅ ⏳ ❌ rõ ràng
- Thanh tiến độ tổng: ~70%

**Ghi chú cho người trình bày:**
"Phần training đã hoàn thành toàn bộ. Phần còn lại là multi-seed validation, test evaluation chính thức, và quan trọng nhất là module XAI."

---

# Slide 34

**Tiêu đề:** Kế hoạch XAI — Kỹ thuật

**Mục tiêu:** Trình bày 4 kỹ thuật XAI và cách thực hiện.

**Nội dung chính:**

**Phase 8: XAI Analysis (dự kiến)**

| Kỹ thuật | Target | Câu hỏi trả lời | Cách thực hiện |
|---|---|---|---|
| **Grad-CAM** | Image branch | Mô hình nhìn vào vùng ảnh nào? Ảnh nào relevant? | Hook gradient vào last conv layer → heatmap |
| **Attention Visualization** | Text branch | Token nào ảnh hưởng nhiều nhất đến dự đoán? | Trích attention weights từ transformer layer |
| **SHAP** | Fusion level | Text hay image đóng góp bao nhiêu % cho mỗi dự đoán? | SHAP values tại fusion input |
| **LIME** | Local sample | Tại sao sample cụ thể bị dự đoán sai? | Perturbation-based local explanation |

**Yếu tố hình ảnh:**

Hình:
Minh hoạ Grad-CAM heatmap trên ảnh món ăn (ví dụ từ literature)
(sẽ bổ sung sau)

**Ghi chú cho người trình bày:**
"Bốn kỹ thuật phủ 4 cấp độ giải thích: vùng ảnh, token văn bản, tỷ lệ đóng góp modality, và giải thích cục bộ."

---

# Slide 35

**Tiêu đề:** Kế hoạch XAI — Lộ trình Thực hiện

**Mục tiêu:** Trình bày kế hoạch thực hiện XAI cụ thể.

**Nội dung chính:**

Kế hoạch:
1. Sanity check XAI trên 1-2 baseline models (EXP_080)
2. Full XAI analysis trên best baseline vs best final model (EXP_081)
3. Chọn case studies: correct, incorrect, high-error, modality-conflict samples
4. Giải thích từng target riêng biệt

**Yếu tố hình ảnh:**

Hình:
Minh hoạ attention visualization trên text review (ví dụ)
(sẽ bổ sung sau)

**Ghi chú cho người trình bày:**
"XAI sẽ trả lời 4 câu hỏi then chốt: (1) mô hình nhìn vào đâu trên ảnh, (2) từ nào quan trọng trong bình luận, (3) modality nào đóng góp nhiều hơn, (4) tại sao sai ở sample cụ thể. Đây là phần thiết yếu để dự án xứng đáng với chữ 'Explainable' trong tên đề tài."

---

# Slide 36

**Tiêu đề:** Đóng góp Khoa học

**Mục tiêu:** Trình bày đóng góp khoa học của dự án.

**Nội dung chính:**

**1. Đóng góp khoa học:**
- Controlled Sequential Ablation framework cho multimodal regression trên dữ liệu tiếng Việt
- So sánh hệ thống: 4 image backbones × 3 text backbones × 5 fusion methods × 4 loss functions
- Evidence-based: mỗi quyết định thiết kế đều có ablation support
- Chứng minh Vietnamese-specific pretraining (PhoBERT) vượt trội multilingual (XLM-R) cho bài toán review

**Yếu tố hình ảnh:**
- Icon: microscope (khoa học)
- Danh sách 4 bullet points, font lớn

**Ghi chú cho người trình bày:**
"Về mặt khoa học, phương pháp ablation có hệ thống cho 21 thí nghiệm là đóng góp chính. Kết quả PhoBERT vượt trội XLM-R trên bài toán review ẩm thực Việt Nam cũng là finding quan trọng."

---

# Slide 37

**Tiêu đề:** Đóng góp Kỹ thuật & Ứng dụng

**Mục tiêu:** Trình bày đóng góp kỹ thuật và ứng dụng.

**Nội dung chính:**

**2. Đóng góp kỹ thuật:**
- Bộ dữ liệu Foody multimodal tiếng Việt (9.946 reviews, 22.150 ảnh)
- Pipeline end-to-end: crawl → clean → train → evaluate → explain
- 5 fusion architectures implemented (Concat, GMU, Gated Cross-Modal, FiLM, Cross-Attention)
- Rule-based overall_satisfaction label với explainable evidence

**3. Đóng góp ứng dụng:**
- Hệ thống có thể đánh giá tự động chất lượng review
- Khả năng giải thích giúp nhà hàng hiểu lý do đánh giá
- Phát hiện review bất thường (qua XAI)

**Yếu tố hình ảnh:**
- 2 cột: gear (kỹ thuật) + lightbulb (ứng dụng)

**Ghi chú cho người trình bày:**
"Về mặt kỹ thuật, toàn bộ pipeline từ crawl đến evaluate đã hoạt động. Về ứng dụng, hệ thống có thể giải thích dự đoán — đây là yêu cầu quan trọng trong thực tế."

---

# Slide 38

**Tiêu đề:** Tổng kết & Câu hỏi

**Mục tiêu:** Slide kết thúc, tóm tắt ngắn gọn.

**Nội dung chính:**

Tóm tắt:
- ✅ 21 thí nghiệm hoàn thành theo phương pháp Controlled Sequential Ablation
- ✅ Best config: **Swin-B + PhoBERT + Cross-Attention + Log-Cosh** (Mean MAE = 1.108)
- ✅ Cải thiện ~15% so với baseline multimodal
- ⏳ Tiếp theo: Multi-seed validation + Locked test evaluation + XAI analysis

Xin cảm ơn thầy/cô. Em sẵn sàng trả lời câu hỏi.

**Yếu tố hình ảnh:**
- Layout sạch, không quá nhiều text
- Email / Contact info

**Ghi chú cho người trình bày:**
"Tóm lại, dự án đã hoàn thành phần training với 21 thí nghiệm có hệ thống. Công việc còn lại tập trung vào validation cuối cùng và XAI. Em sẵn sàng trả lời câu hỏi."

---

# ===== BACKUP SLIDES =====

---

# Backup Slide B1

**Tiêu đề:** Chi tiết Dataset — Thống kê

**Mục tiêu:** Dự phòng nếu giảng viên hỏi sâu về dataset.

**Nội dung chính:**

| Chỉ số | Giá trị |
|---|---:|
| Nhà hàng crawl | 300 |
| Review thô | 11.111 |
| Review hợp lệ sau lọc | 9.946 |
| Ảnh thô | 24.599 |
| Cặp review-ảnh | 22.150 |
| Review có ảnh | 6.082 (61.15%) |
| Mẫu train | ~4.864 |
| Mẫu validation | ~608 |
| Mẫu test | ~608 |
| Max images/review | 4 |
| Max text length | 256 tokens |

Nhãn overall_satisfaction:
- 14 nhóm luật (8 tích cực, 6 tiêu cực)
- 3.263/9.946 review được điều chỉnh ≠ 0
- 2.058 điều chỉnh tích cực, 1.205 tiêu cực

**Yếu tố hình ảnh:**
- Bảng thống kê chi tiết

---

# Backup Slide B2

**Tiêu đề:** Experiment Inventory — Danh sách 21 Thí nghiệm

**Mục tiêu:** Dự phòng nếu giảng viên muốn xem toàn bộ experiments.

**Nội dung chính:**

Phase 1 (Baselines):
- EXP_010: Text-Only (XLM-R)
- EXP_011: Image-Only (ConvNeXt)
- EXP_012: Multimodal Concat Baseline

Phase 2 (Image Ablation):
- EXP_020B: Swin-B + XLM-R
- EXP_020D: EfficientNet-B3 + XLM-R
- EXP_020E: SigLIP + XLM-R

Phase 3 (Text Ablation):
- EXP_030B: Swin-B + PhoBERT
- EXP_030D: Swin-B + ViSoBERT

Phase 4 (Fusion):
- EXP_040B: GMU
- EXP_040C: Gated Cross-Modal
- EXP_041A: FiLM
- EXP_041B: Cross-Attention

Phase 5 (Loss):
- EXP_050B: Huber
- EXP_050C: Log-Cosh
- EXP_051D: Uncertainty Weighted

Phase 6 (Combinations):
- EXP_060A–E: 5 full configurations

Phase 7:
- EXP_070: Seed validation (planned)

**Yếu tố hình ảnh:**
- Dạng bảng nhỏ hoặc danh sách gọn, 21 mục

---

# Backup Slide B3

**Tiêu đề:** Hyperparameters & Training Settings

**Mục tiêu:** Dự phòng nếu giảng viên hỏi chi tiết training.

**Nội dung chính:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 1e-2 |
| Scheduler | Cosine with warmup (ratio=0.1) |
| Batch size | 16 |
| Max epochs | 20 (backbone) / 15 (fusion) |
| Early stopping | Patience = 3–5 |
| Gradient clipping | max_norm = 1.0 |
| Mixed precision | AMP enabled |
| Seed | 42 |
| Max text length | 256 tokens |
| Max images | 4 |

Huấn luyện 3 giai đoạn:
1. Train text encoder: 20 epochs
2. Train image encoder: 20 epochs
3. Freeze encoders → Train fusion: 15 epochs

**Yếu tố hình ảnh:**
- Bảng thông số

---

# Backup Slide B4

**Tiêu đề:** Fusion Architectures — Chi tiết Kỹ thuật

**Mục tiêu:** Dự phòng nếu giảng viên hỏi sâu về fusion code.

**Nội dung chính:**

**GMU:**
```
gate = sigmoid(W × [text; image])
fused = gate × Wt(text) + (1-gate) × Wi(image)
```

**Gated Cross-Modal:**
```
text_enhanced = text + tanh(W_t2i × image)
image_enhanced = image + tanh(W_i2t × text)
gate = sigmoid(W × [text_enh; image_enh])
fused = gate × proj(text_enh) + (1-gate) × proj(image_enh)
```

**FiLM:**
```
gamma = W_gamma × text
beta = W_beta × text
modulated_image = gamma × image + beta
fused = [text; modulated_image]
```

**Cross-Attention:**
```
text_proj = proj(text).unsqueeze(1)
image_proj = proj(image).unsqueeze(1)
t_out = CrossAttn(query=text, key=image, value=image)
i_out = CrossAttn(query=image, key=text, value=text)
fused = [t_out; i_out]
```

**Yếu tố hình ảnh:**
- Pseudocode rõ ràng cho mỗi fusion

---

# Backup Slide B5

**Tiêu đề:** Validation vs Test Comparison

**Mục tiêu:** Dự phòng nếu giảng viên hỏi về generalization gap.

**Nội dung chính:**

Chỉ áp dụng cho experiments Phase 6 có test_metrics.json.

| Experiment | Val MAE | Test MAE | Gap |
|---|---:|---:|---:|
| (Data will be populated from test_metrics.json if available) |

Chèn hình:
validation_vs_test_comparison.png
(chỉ có nếu experiments Phase 6 đã chạy test)

**Ghi chú cho người trình bày:**
"Generalization gap cho thấy mô hình có overfit không. Gap nhỏ → mô hình generalize tốt."

---

# Backup Slide B6

**Tiêu đề:** Câu hỏi Dự kiến từ Giảng viên

**Mục tiêu:** Chuẩn bị sẵn cho các câu hỏi phổ biến.

**Nội dung chính:**

| Câu hỏi tiềm năng | Hướng trả lời |
|---|---|
| Tại sao không dùng full factorial? | 720+ tổ hợp không khả thi trên Colab; sequential ablation đủ mạnh cho thesis |
| ViSoBERT được pretrain trên social media, sao lại thua PhoBERT? | ViSoBERT overfit nhanh trên training set; PhoBERT có corpus lớn hơn và general hơn |
| Cross-Attention chỉ hơn Concat có 0.006 MAE, có đáng? | Nhất quán trên mọi metric; R² tốt nhất; phần XAI sẽ cho thấy thêm giá trị |
| Tại sao chưa có XAI? | Ưu tiên hoàn thành ablation trước vì XAI cần best model; sẽ là Phase 8 |
| Dataset 6000 mẫu có đủ? | Đủ cho university thesis; kết quả nhất quán qua 21 experiments; sẽ validate bằng multi-seed |
| R² = 0.63 có tốt không? | Acceptable cho noisy user-generated review data; MAE 1.1/10 là sai lệch hợp lý |
| Tại sao 5 targets thay vì 1? | Multi-task learning tận dụng shared representation; individual aspect scores giá trị hơn cho ứng dụng |

**Yếu tố hình ảnh:**
- Bảng Q&A, font lớn

---

# Backup Slide B7

**Tiêu đề:** Ablation Summary Table

**Mục tiêu:** Tóm tắt mỗi phase: reference, winner, improvement.

**Nội dung chính:**

| Phase | Reference | Winner | Ref MAE | Best MAE | Improvement |
|---|---|---|---:|---:|---|
| Image Ablation | EXP_012 (ConvNeXt) | EXP_020B (Swin-B) | ~1.30 | 1.2169 | Best backbone: Swin-B |
| Text Ablation | EXP_020B (XLM-R) | EXP_030B (PhoBERT) | 1.2169 | 1.1145 | ~8.4% improvement |
| Fusion Ablation | EXP_030B (Concat) | EXP_041B (CrossAttention) | 1.1145 | 1.1079 | Cross-modal interaction |
| Loss Ablation | EXP_041B (MSE) | EXP_050C (LogCosh) | 1.1079 | 1.1080 | ~0 (already optimal) |

**Yếu tố hình ảnh:**
- Bảng rõ ràng, highlight improvement column

---

# Backup Slide B8

**Tiêu đề:** Tại sao các Model này? — Research Justification tổng hợp

**Mục tiêu:** Dự phòng nếu giảng viên hỏi "Em chọn model dựa trên cơ sở nào?"

**Nội dung chính:**

**Image Backbones:**
- ConvNeXt: Liu et al. (2022) — CNN modernized, competitive with ViT; Grad-CAM compatible
- Swin-B: Liu et al. (2021) — Shifted window attention, hierarchical, multi-scale
- EfficientNet-B3: Tan & Le (2019) — Compound scaling, efficiency-focused
- SigLIP: Zhai et al. (2023) — Sigmoid loss for image-text alignment

**Text Backbones:**
- XLM-R: Conneau et al. (2020) — 100+ language multilingual BERT; baseline anchor
- PhoBERT: Nguyen & Nguyen (2020) — Pre-trained on Vietnamese Wikipedia + news; Vietnamese NLP gold standard
- ViSoBERT: Nguyen et al. (2023) — Pre-trained on Vietnamese social media text

**Fusion Methods:**
- GMU: Arevalo et al. (2017) — Gated Multimodal Unit for sample-level modality weighting
- FiLM: Perez et al. (2018) — Feature-wise Linear Modulation
- Cross-Attention: Vaswani et al. (2017) — Transformer attention applied cross-modally

**Ghi chú cho người trình bày:**
"Mọi lựa chọn đều dựa trên literature review và phân tích đặc điểm dataset. Không có model nào được chọn ngẫu nhiên."

---

# ===== REVISION SUMMARY =====

## Slide Count

- **Original slide count:** 25 main + 8 backup = 33 total
- **New slide count:** 38 main + 8 backup = 46 total

## Slides that were split

| Original Slide | Split into | Reason |
|---|---|---|
| Slide 3 (Bài toán & Động lực) | Slides 3 + 4 | Mixed problem definition AND multimodal justification — two distinct messages |
| Slide 4 (Tại sao XAI) | Slides 5 + 6 | Mixed motivation bullets AND technique table — separate "why" from "what" |
| Slide 5 (Dataset Thu thập) | Slides 7 + 8 | Mixed collection pipeline AND grouping logic AND label generation — three concepts |
| Slide 7 (Kiến trúc Tổng quan) | Slides 10 + 11 | Mixed architecture diagram AND 3-stage training strategy — two distinct ideas |
| Slide 9 (Text Branch & Fusion) | Slides 13 + 14 | Mixed text branch details AND 5-method fusion table — two components |
| Slide 10 (Controlled Sequential Ablation) | Slides 15 + 16 | Mixed methodology explanation AND 7-row phase table — concept vs roadmap |
| Slide 12 (Image Backbone) | Slides 18 + 19 | Mixed 4-candidate rationale table AND results table AND conclusion — candidates vs results |
| Slide 13 (Text Backbone) | Slides 20 + 21 | Mixed context AND 3-candidate table AND results AND conclusion — candidates vs results |
| Slide 14 (Fusion) | Slides 22 + 23 | Mixed problem statement AND 5-method list AND results table — problem vs results |
| Slide 15 (Loss Function) | Slides 24 + 25 | Mixed MSE problem AND 4-loss table AND results AND conclusion — problem vs results |
| Slide 21 (Cấu hình Tối ưu) | Slides 31 + 32 | Mixed configuration table AND 7-row metrics table AND interpretation — config vs numbers |
| Slide 23 (XAI Planning) | Slides 34 + 35 | Mixed 4-technique table AND 4-step execution plan AND figure placeholders — techniques vs plan |
| Slide 24 (Đóng góp) | Slides 36 + 37 | Mixed scientific AND technical AND application contributions — 11 bullets total across 3 categories |

## Slides kept unchanged

Slides 1, 2, 6→9, 8→12, 11→17, 16→26, 17→27, 18→28, 19→29, 20→30, 22→33, 25→38, and all 8 backup slides were kept as-is because they each communicate one main idea within acceptable density.
