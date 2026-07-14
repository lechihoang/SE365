# Multimodal Aspect-Based Sentiment Analysis for Foody Reviews

Dự án xây dựng mô hình Phân tích Cảm xúc Đa phương thức (Multimodal Aspect-Based Sentiment Analysis) từ các bài đánh giá nhà hàng trên Foody. Mô hình kết hợp cả văn bản (text) và hình ảnh (image) để dự đoán điểm số của 5 khía cạnh (aspects):

- **food** (Đồ ăn)
- **service** (Phục vụ)
- **atmosphere** (Không gian)
- **price** (Giá cả)
- **overall** (Tổng quan)

---

## Dataset

Dữ liệu được thu thập từ Foody thông qua [crawl_data_from_foody.ipynb](notebook/crawl_data_from_foody.ipynb) và làm sạch trong [clean_foody_dataset.ipynb](notebook/clean_foody_dataset.ipynb). Nhãn `overall_satisfaction` được sinh thêm từ [01_generate_overall_satisfaction.ipynb](notebook/01_generate_overall_satisfaction.ipynb). Dữ liệu chia thành 3 tập: Train, Validation, Test.

---

## Hướng dẫn chạy

Chạy lần lượt theo thứ tự sau (khuyến nghị dùng GPU và môi trường `.venv`):

1. [crawl_data_from_foody.ipynb](notebook/crawl_data_from_foody.ipynb) — Thu thập dữ liệu từ Foody.
2. [clean_foody_dataset.ipynb](notebook/clean_foody_dataset.ipynb) — Làm sạch dữ liệu, chia split Train/Val/Test.
3. [01_generate_overall_satisfaction.ipynb](notebook/01_generate_overall_satisfaction.ipynb) — Tạo nhãn `overall_satisfaction`.
4. Chạy tuần tự các nhóm notebook `EXP_01*` đến `EXP_05*` để tái hiện quá trình lựa chọn kiến trúc (chỉ đánh giá trên tập Validation).
5. Chạy các notebook `EXP_060*` để huấn luyện và đánh giá toàn diện trên cả Validation và Test.
6. [generate_experiment_leaderboard.ipynb](notebook/generate_experiment_leaderboard.ipynb) — Tổng hợp toàn bộ kết quả thành bảng xếp hạng.
7. [demo_single_sample_exp060A.ipynb](notebook/demo_single_sample_exp060A.ipynb) — Demo dự đoán với mô hình tốt nhất.

---

## Quá trình thử nghiệm và kết quả

Các phase từ 1 đến 5 chỉ đánh giá trên tập **Validation** để tìm kiếm kiến trúc tốt nhất và tránh rò rỉ thông tin từ tập Test. Phase 6 đánh giá đầy đủ trên cả Validation và Test.

Tổng quan từng phase:

- **Phase 1**: So sánh 3 phương thức cơ bản: Text-only, Image-only, và Multimodal (Concat). Mục tiêu xác định liệu kết hợp đa phương thức có ý nghĩa không.
- **Phase 2**: Cố định Text Encoder (XLM-R) và Fusion (Concat), thay đổi Image Encoder để tìm backbone trích xuất ảnh tốt nhất. So sánh: Swin-B, EfficientNet-B3, SigLIP.
- **Phase 3**: Cố định Image Encoder tốt nhất từ Phase 2 (Swin-B) và Fusion (Concat), thay đổi Text Encoder. So sánh: PhoBERT và ViSoBERT — các mô hình được pretrain đặc thù cho tiếng Việt.
- **Phase 4**: Cố định cả hai encoder tốt nhất (Swin-B + PhoBERT), thay đổi cơ chế Fusion. So sánh: GMU, Gated Cross-Modal, FiLM, Cross-Attention.
- **Phase 5**: Cố định toàn bộ kiến trúc tốt nhất từ các Phase trước (Swin-B + PhoBERT + Cross-Attention), thay đổi hàm Loss. So sánh: Huber, Log-Cosh, Uncertainty Weighted.
- **Phase 6**: Lấy cấu hình tốt nhất từ Phase 1-5 (EXP_060A) và so sánh với một số tổ hợp thay thế khác. Đây là phase đánh giá cuối cùng trên tập Test.

---

### Phase 1 — Baseline (So sánh phương thức)

So sánh 3 phương thức: Text-only (XLM-R), Image-only (ConvNeXt), và Multimodal kết hợp cả hai bằng Concat, với Loss = MSE.

**Kết quả (Validation):**

| Experiment | Cấu hình | Loss (val) | MAE Food | MAE Price | MAE Atmos | MAE Service | MAE Overall | Mean MAE | Aspect MAE | RMSE Food | RMSE Price | RMSE Atmos | RMSE Service | RMSE Overall | R2 Food | R2 Price | R2 Atmos | R2 Service | R2 Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP_010 | Text-only (XLM-R) | 2.8301 | 1.2717 | 1.2928 | 1.2510 | 1.3135 | 1.0880 | 1.2434 | 1.2823 | 1.7472 | 1.7583 | 1.6438 | 1.7583 | 1.4800 | 0.4200 | 0.3078 | 0.3037 | 0.3970 | 0.4620 |
| EXP_011 | Image-only (ConvNeXt) | 4.3926 | 1.6013 | 1.4951 | 1.4166 | 1.5809 | 1.3808 | 1.4949 | 1.5235 | 2.2630 | 2.0552 | 1.9152 | 2.2231 | 1.9642 | 0.0270 | 0.0543 | 0.0549 | 0.0361 | 0.0525 |
| EXP_012 | Multimodal (ConvNeXt + XLM-R + Concat) | 2.8651 | 1.2640 | 1.2890 | 1.2423 | 1.3096 | 1.0876 | 1.2385 | 1.2762 | 1.7669 | 1.7672 | 1.6446 | 1.7773 | 1.5017 | 0.4068 | 0.3008 | 0.3031 | 0.3839 | 0.4461 |

**Kết luận:** Image-only cho kết quả rất kém (R2 gần 0). Text-only nhỉnh hơn Multimodal một chút do ConvNeXt chưa đủ mạnh. Hướng tiếp theo: thay Image Encoder mạnh hơn.

---

### Phase 2 — Image Encoder Selection

Cố định Text Encoder (XLM-R) và Fusion (Concat, MSE). Thay đổi Image Encoder để tìm backbone tốt nhất.

**Kết quả (Validation):**

| Experiment | Image Encoder | Loss (val) | MAE Food | MAE Price | MAE Atmos | MAE Service | MAE Overall | Mean MAE | Aspect MAE | RMSE Food | RMSE Price | RMSE Atmos | RMSE Service | RMSE Overall | R2 Food | R2 Price | R2 Atmos | R2 Service | R2 Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP_020B | **Swin-B** | **2.7093** | 1.2371 | 1.2639 | 1.2249 | 1.2920 | 1.0667 | **1.2169** | **1.2545** | 1.7155 | 1.7202 | 1.6135 | 1.7290 | 1.4447 | 0.4408 | 0.3375 | 0.3292 | 0.4170 | 0.4874 |
| EXP_020D | EfficientNet-B3 | 2.9468 | 1.3174 | 1.3356 | 1.2705 | 1.3469 | 1.1296 | 1.2800 | 1.3176 | 1.8024 | 1.7977 | 1.6746 | 1.7865 | 1.5320 | 0.3828 | 0.2765 | 0.2774 | 0.3776 | 0.4236 |
| EXP_020E | SigLIP | 2.7929 | 1.2522 | 1.2804 | 1.2357 | 1.3095 | 1.0703 | 1.2296 | 1.2695 | 1.7402 | 1.7489 | 1.6307 | 1.7634 | 1.4669 | 0.4246 | 0.3152 | 0.3148 | 0.3935 | 0.4715 |

**Kết luận:** **Swin-B** cho Val Loss và Mean MAE tốt nhất. Chọn Swin-B làm Image Encoder.

---

### Phase 3 — Text Encoder Selection

Cố định Image Encoder (Swin-B) và Fusion (Concat, MSE). Thay đổi Text Encoder, so sánh PhoBERT và ViSoBERT.

**Kết quả (Validation):**

| Experiment | Text Encoder | Loss (val) | MAE Food | MAE Price | MAE Atmos | MAE Service | MAE Overall | Mean MAE | Aspect MAE | RMSE Food | RMSE Price | RMSE Atmos | RMSE Service | RMSE Overall | R2 Food | R2 Price | R2 Atmos | R2 Service | R2 Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP_030B | **PhoBERT** | **2.2034** | 1.1134 | 1.1711 | 1.1767 | 1.1814 | 0.9300 | **1.1145** | **1.1607** | 1.5072 | 1.5631 | 1.5249 | 1.5674 | 1.2406 | 0.5684 | 0.4529 | 0.4009 | 0.5209 | 0.6220 |
| EXP_030D | ViSoBERT | 2.8100 | 1.2576 | 1.2736 | 1.2459 | 1.2945 | 1.0923 | 1.2328 | 1.2679 | 1.7451 | 1.7649 | 1.6386 | 1.7339 | 1.4843 | 0.4214 | 0.3026 | 0.3081 | 0.4137 | 0.4589 |

**Kết luận:** **PhoBERT** vượt trội hoàn toàn so với ViSoBERT. Chọn PhoBERT làm Text Encoder.

---

### Phase 4 — Fusion Method Selection

Cố định Swin-B (image) + PhoBERT (text) + MSE. Thay đổi cơ chế Fusion, so sánh: GMU, Gated Cross-Modal, FiLM, Cross-Attention.

**Kết quả (Validation):**

| Experiment | Fusion Method | Loss (val) | MAE Food | MAE Price | MAE Atmos | MAE Service | MAE Overall | Mean MAE | Aspect MAE | RMSE Food | RMSE Price | RMSE Atmos | RMSE Service | RMSE Overall | R2 Food | R2 Price | R2 Atmos | R2 Service | R2 Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP_040B | GMU | 2.2047 | 1.1136 | 1.1756 | 1.1813 | 1.1808 | 0.9289 | 1.1160 | 1.1628 | 1.5068 | 1.5698 | 1.5245 | 1.5671 | 1.2364 | 0.5686 | 0.4483 | 0.4012 | 0.5210 | 0.6246 |
| EXP_040C | Gated Cross-Modal | 2.1740 | 1.0963 | 1.1713 | 1.1765 | 1.1772 | 0.9198 | 1.1082 | 1.1553 | 1.4810 | 1.5658 | 1.5171 | 1.5618 | 1.2259 | 0.5832 | 0.4510 | 0.4070 | 0.5243 | 0.6309 |
| EXP_041A | FiLM | 2.2243 | 1.1238 | 1.1783 | 1.1789 | 1.1890 | 0.9278 | 1.1195 | 1.1675 | 1.5133 | 1.5741 | 1.5302 | 1.5787 | 1.2415 | 0.5649 | 0.4452 | 0.3966 | 0.5139 | 0.6215 |
| EXP_041B | **Cross-Attention** | **2.1750** | 1.1024 | 1.1728 | 1.1743 | 1.1756 | 0.9143 | **1.1079** | **1.1563** | 1.4887 | 1.5633 | 1.5172 | 1.5613 | 1.2215 | 0.5789 | 0.4528 | 0.4069 | 0.5246 | 0.6335 |

**Kết luận:** **Cross-Attention** cho Mean MAE thấp nhất và R2 Overall cao nhất. Chọn Cross-Attention làm Fusion method.

---

### Phase 5 — Loss Function Selection

Cố định Swin-B + PhoBERT + Cross-Attention. Thay đổi hàm Loss, so sánh: Huber, Log-Cosh, Uncertainty Weighted.

**Kết quả (Validation):**

| Experiment | Loss Function | Loss (val) | MAE Food | MAE Price | MAE Atmos | MAE Service | MAE Overall | Mean MAE | Aspect MAE | RMSE Food | RMSE Price | RMSE Atmos | RMSE Service | RMSE Overall | R2 Food | R2 Price | R2 Atmos | R2 Service | R2 Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP_050B | Huber | 0.7127 | 1.1097 | 1.1685 | 1.1734 | 1.1779 | 0.9131 | 1.1085 | 1.1574 | 1.5054 | 1.5671 | 1.5261 | 1.5722 | 1.2261 | 0.5694 | 0.4501 | 0.3999 | 0.5179 | 0.6308 |
| EXP_050C | **Log-Cosh** | **0.6413** | 1.1066 | 1.1694 | 1.1739 | 1.1770 | 0.9130 | **1.1080** | **1.1567** | 1.5006 | 1.5671 | 1.5250 | 1.5697 | 1.2254 | 0.5722 | 0.4502 | 0.4008 | 0.5194 | 0.6312 |
| EXP_051D | Uncertainty Weighted | 2.1737 | 1.1024 | 1.1727 | 1.1742 | 1.1764 | 0.9144 | 1.1080 | 1.1564 | 1.4885 | 1.5632 | 1.5175 | 1.5623 | 1.2213 | 0.5790 | 0.4529 | 0.4067 | 0.5240 | 0.6337 |

**Kết luận:** **Log-Cosh** đạt Val Loss thấp nhất (0.6413). Uncertainty Weighted có MAE aspect nhỉnh hơn đôi chút nhưng Loss value không so sánh trực tiếp được do thang đo khác. Chọn Log-Cosh cho Phase 6.

---

### Phase 6 — Final Evaluation (Validation + Test)

Lấy cấu hình tốt nhất tích lũy từ Phase 1-5 (EXP_060A: Swin-B + PhoBERT + Cross-Attention + Log-Cosh) và so sánh với một số tổ hợp thay thế khác. Đây là phase duy nhất đánh giá trên tập Test.

#### Kết quả — Validation Set

| Experiment | Cấu hình | Loss (val) | MAE Food | MAE Price | MAE Atmos | MAE Service | MAE Overall | Mean MAE | Aspect MAE | RMSE Food | RMSE Price | RMSE Atmos | RMSE Service | RMSE Overall | R2 Food | R2 Price | R2 Atmos | R2 Service | R2 Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_060A** | **Swin-B + PhoBERT + Cross-Attention + Log-Cosh** | **0.6413** | **1.1066** | **1.1694** | **1.1739** | **1.1770** | **0.9130** | **1.1080** | **1.1567** | **1.5006** | **1.5671** | **1.5250** | **1.5697** | **1.2254** | **0.5722** | **0.4502** | **0.4008** | **0.5194** | **0.6312** |
| EXP_060C | EfficientNet-B3 + PhoBERT + FiLM + Huber | 0.7267 | 1.1382 | 1.1812 | 1.1749 | 1.1985 | 0.9354 | 1.1256 | 1.1732 | 1.5304 | 1.5895 | 1.5450 | 1.5830 | 1.2565 | 0.5550 | 0.4344 | 0.3849 | 0.5112 | 0.6123 |
| EXP_060D | EfficientNet-B3 + ViSoBERT + Cross-Attention + Log-Cosh | 0.7925 | 1.3120 | 1.3630 | 1.2749 | 1.3296 | 1.1353 | 1.2829 | 1.3199 | 1.7891 | 1.8138 | 1.6758 | 1.7647 | 1.5289 | 0.3918 | 0.2634 | 0.2764 | 0.3926 | 0.4259 |
| EXP_060E | ConvNeXt + PhoBERT + Gated Cross-Modal + Auto Weight | 2.2396 | 1.1205 | 1.1838 | 1.1908 | 1.1858 | 0.9435 | 1.1248 | 1.1702 | 1.5029 | 1.5899 | 1.5375 | 1.5803 | 1.2561 | 0.5708 | 0.4341 | 0.3909 | 0.5129 | 0.6125 |
| EXP_060B | Swin-B + ViSoBERT + GMU + Uncertainty Weighted | 2.8061 | 1.2569 | 1.2702 | 1.2446 | 1.2922 | 1.0864 | 1.2300 | 1.2660 | 1.7459 | 1.7682 | 1.6361 | 1.7320 | 1.4817 | 0.4209 | 0.3000 | 0.3103 | 0.4149 | 0.4608 |

#### Kết quả — Test Set

| Experiment | Cấu hình | MAE Food | MAE Price | MAE Atmos | MAE Service | MAE Overall | Mean MAE | Aspect MAE | RMSE Food | RMSE Price | RMSE Atmos | RMSE Service | RMSE Overall | R2 Food | R2 Price | R2 Atmos | R2 Service | R2 Overall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP_060A** | **Swin-B + PhoBERT + Cross-Attention + Log-Cosh** | **1.0471** | **1.1201** | 1.1754 | **1.0867** | **0.8819** | **1.0622** | **1.1073** | **1.4692** | **1.5015** | 1.5565 | **1.4890** | **1.1772** | **0.6093** | **0.4628** | 0.3598 | **0.5326** | **0.6479** |
| EXP_060E | ConvNeXt + PhoBERT + Gated Cross-Modal + Auto Weight | 1.0525 | 1.1212 | 1.1935 | 1.0998 | 0.9098 | 1.0754 | 1.1168 | 1.4617 | 1.4863 | 1.5665 | 1.4976 | 1.2019 | 0.6133 | 0.4736 | 0.3516 | 0.5271 | 0.6330 |
| EXP_060C | EfficientNet-B3 + PhoBERT + FiLM + Huber | 1.1080 | 1.1674 | 1.2107 | 1.1150 | 0.9171 | 1.1036 | 1.1503 | 1.5101 | 1.5356 | 1.5974 | 1.5062 | 1.2185 | 0.5873 | 0.4381 | 0.3257 | 0.5217 | 0.6227 |
| EXP_060B | Swin-B + ViSoBERT + GMU + Uncertainty Weighted | 1.2117 | 1.1964 | 1.2522 | 1.1949 | 1.0185 | 1.1747 | 1.2138 | 1.6469 | 1.5779 | 1.6140 | 1.6348 | 1.3408 | 0.5091 | 0.4068 | 0.3117 | 0.4366 | 0.5432 |
| EXP_060D | EfficientNet-B3 + ViSoBERT + Cross-Attention + Log-Cosh | 1.2764 | 1.2672 | 1.3083 | 1.2477 | 1.0787 | 1.2357 | 1.2749 | 1.7052 | 1.6440 | 1.6600 | 1.6629 | 1.4067 | 0.4738 | 0.3560 | 0.2719 | 0.4170 | 0.4972 |

**Kết luận:** Cấu hình **EXP_060A** (Swin-B + PhoBERT + Cross-Attention + Log-Cosh) là kết quả tổng hợp tốt nhất từ toàn bộ quá trình tìm kiếm Phase 1-5. Mô hình đạt Mean MAE = 1.0622 và R2 Overall = 0.6479 trên tập Test, vượt trội hầu hết các tiêu chí so với các tổ hợp thay thế.

---

## Hướng dẫn huấn luyện lại

Toàn bộ pipeline huấn luyện chạy qua hai script chính: `main.py` (train) và `test.py` (đánh giá trên test set).

### Cài đặt môi trường

```bash
# 1. Clone repo
lechihoang/SE365 https://github.com/lechihoang/SE365
cd SE365

# 2. Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# hoặc: .venv\Scripts\activate   # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

### Các tham số chính

| Tham số | Ý nghĩa | Các giá trị hợp lệ |
| :--- | :--- | :--- |
| `--mode` | Chế độ huấn luyện | `train_text`, `train_image`, `train_fusion` |
| `--text_model_name` | Tên model text encoder trên HuggingFace | `xlm-roberta-base`, `vinai/phobert-base-v2`, `uitnlp/visobert` |
| `--image_model_name` | Tên model image encoder (timm) | `convnext_base_in22k`, `swin_base_patch4_window7_224`, `efficientnet_b3`, `vit_base_patch16_siglip_256` |
| `--fusion_type` | Cơ chế kết hợp 2 luồng đặc trưng | `concat`, `gmu`, `gated_cross`, `film`, `cross_attention` |
| `--loss_fn` | Hàm loss | `mse`, `huber`, `logcosh`, `auto_weight` |
| `--epochs` | Số epoch tối đa | số nguyên, ví dụ `15`, `20` |
| `--batch_size` | Batch size | `16`, `32` |
| `--lr` | Learning rate | `1e-5`, `2e-5` |
| `--grad_accum_steps` | Số bước tích lũy gradient (effective batch = batch_size × grad_accum_steps) | `1`, `2` |
| `--patience` | Số epoch chờ trước khi early stopping | `5` |
| `--unfreeze_text_layers` | Số layer cuối của text encoder được unfreeze khi train fusion | `1` |
| `--unfreeze_image_layers` | Số layer cuối của image encoder được unfreeze khi train fusion | `1` |
| `--seed` | Random seed để tái tạo kết quả | `42` |
| `--use_amp` | Bật Automatic Mixed Precision (tăng tốc train trên GPU) | flag, không cần giá trị |
| `--exp_id` | Tên định danh experiment, dùng để lưu checkpoint | chuỗi bất kỳ |
| `--exp_dir` | Thư mục lưu kết quả experiment | đường dẫn, ví dụ `./experiments` |
| `--save_path` | Đường dẫn load checkpoint khi chạy `test.py` | đường dẫn đến thư mục chứa checkpoint |

---

### Cách train

Quy trình train cho một experiment multimodal gồm 3 bước:

**Bước 1: Train text encoder riêng**

```bash
python main.py \
  --mode train_text \
  --text_model_name vinai/phobert-base-v2 \
  --epochs 20 --batch_size 16 --lr 1e-5 \
  --loss_fn mse --seed 42 --use_amp \
  --exp_dir ./experiments
```

**Bước 2: Train image encoder riêng**

```bash
python main.py \
  --mode train_image \
  --image_model_name swin_base_patch4_window7_224 \
  --epochs 20 --batch_size 16 --lr 1e-5 \
  --loss_fn mse --seed 42 --use_amp \
  --exp_dir ./experiments
```

**Bước 3: Train fusion model (load checkpoint từ 2 bước trên)**

```bash
python main.py \
  --mode train_fusion \
  --fusion_type cross_attention \
  --text_model_name vinai/phobert-base-v2 \
  --image_model_name swin_base_patch4_window7_224 \
  --loss_fn logcosh \
  --epochs 15 --batch_size 16 --lr 1e-5 \
  --grad_accum_steps 2 --patience 5 \
  --unfreeze_text_layers 1 --unfreeze_image_layers 1 \
  --seed 42 --use_amp \
  --exp_id MY_EXPERIMENT --exp_dir ./experiments
```

Thay `--fusion_type` và `--loss_fn` theo cấu hình muốn thử nghiệm. Thay `--text_model_name` và `--image_model_name` để thử encoder khác.

---

### Cách test

Sau khi train xong, chạy `test.py` để đánh giá trên tập Test. Script này load checkpoint tốt nhất từ thư mục experiment và in ra toàn bộ metrics (MAE, RMSE, R2 cho từng aspect).

```bash
python test.py \
  --mode train_fusion \
  --fusion_type cross_attention \
  --text_model_name vinai/phobert-base-v2 \
  --image_model_name swin_base_patch4_window7_224 \
  --loss_fn logcosh \
  --exp_id MY_EXPERIMENT \
  --exp_dir ./experiments \
  --save_path ./experiments/MY_EXPERIMENT
```

`--save_path` trỏ đến thư mục chứa file checkpoint (`.pth`) được lưu từ bước train. Các tham số kiến trúc (`--fusion_type`, `--text_model_name`, `--image_model_name`, `--loss_fn`) phải khớp với lúc train.

