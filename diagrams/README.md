# 📁 Diagrams — SE365 Multimodal Restaurant Review

Thư mục này chứa tất cả sơ đồ của dự án theo yêu cầu báo cáo.

---

## Cấu trúc file

| File | Loại | Nội dung |
|---|---|---|
| `01_system_architecture.drawio` | draw.io XML | Kiến trúc hệ thống tổng thể |
| `02_data_pipeline.drawio` | draw.io XML | Pipeline xử lý dữ liệu (5 bước) |
| `03_model_architecture.drawio` | draw.io XML | Kiến trúc 5 fusion models chi tiết |
| `04_xai_architecture.drawio` | draw.io XML | Pipeline XAI (Grad-CAM, Attention, SHAP, LIME) |
| `mermaid_diagrams.md` | Mermaid | Code dự phòng, render trực tiếp trong Markdown |
| `export_png/` | PNG | Ảnh PNG xuất từ draw.io (cần export thủ công) |

---

## Cách mở file draw.io

### Cách 1: Trực tuyến (khuyên dùng)
1. Truy cập [https://app.diagrams.net](https://app.diagrams.net)
2. Chọn **File → Open from → Device**
3. Chọn file `.drawio` tương ứng
4. Diagram sẽ hiện đầy đủ với màu sắc và layout

### Cách 2: VS Code Extension
- Cài extension **hediet.vscode-drawio** từ VS Code Marketplace
- Mở file `.drawio` trực tiếp trong VS Code

---

## Cách Export PNG từ draw.io (cho báo cáo)

1. Mở file trong [app.diagrams.net](https://app.diagrams.net)
2. **File → Export as → PNG**
3. Thiết lập:
   - Scale: **200%** (để có độ phân giải cao)
   - Border: **10px**
   - Background: **White**
4. Lưu vào thư mục `diagrams/export_png/`

### Tên file PNG gợi ý

```
export_png/
├── 01_system_architecture.png
├── 02_data_pipeline.png
├── 03_model_architecture.png
└── 04_xai_architecture.png
```

---

## Chèn vào report.md

Sau khi export PNG, thêm vào `report.md` tại vị trí phù hợp:

```markdown
### Hình X: Kiến trúc hệ thống tổng thể
![Kiến trúc hệ thống](diagrams/export_png/01_system_architecture.png)

### Hình X+1: Pipeline xử lý dữ liệu
![Data Pipeline](diagrams/export_png/02_data_pipeline.png)

### Hình X+2: Kiến trúc mô hình
![Model Architecture](diagrams/export_png/03_model_architecture.png)

### Hình X+3: Kiến trúc XAI
![XAI Architecture](diagrams/export_png/04_xai_architecture.png)
```

---

## Yêu cầu nộp báo cáo

Theo yêu cầu của giảng viên:
- ✅ File draw.io đã được lưu trong thư mục này
- ⬜ Cần upload thư mục `diagrams/` lên Google Drive folder **"IMAGE REPORT"**
- ⬜ Cần export PNG và chèn vào report trước khi xuất PDF

---

## Mô tả từng sơ đồ

### 01 — Kiến trúc hệ thống tổng thể
Toàn bộ pipeline từ đầu vào (text + ảnh) qua preprocessing, hai nhánh encoder (PhoBERT / Swin-B), tầng Cross-Attention Fusion, đến 5 đầu ra dự đoán. Bao gồm loss function, XAI module, và rule engine cho overall_satisfaction.

### 02 — Pipeline xử lý dữ liệu
5 bước tuần tự: Thu thập từ Foody.vn → Làm sạch → Sinh nhãn (rule engine) → Phân chia train/val/test → Tải và xử lý ảnh. Kèm thống kê số liệu tại mỗi bước.

### 03 — Kiến trúc mô hình chi tiết
Text Branch và Image Branch với các lựa chọn encoder. 5 fusion architectures: Concat, GMU, Gated Cross-Modal, FiLM, Cross-Attention (best). Loss functions và metrics. Unfreeze strategy.

### 04 — Kiến trúc XAI
4 kỹ thuật giải thích: Grad-CAM cho image branch, Attention Visualization cho text, SHAP cho modality contribution, LIME cho local explanation. Shared infrastructure (xai/utils.py) và Case Study pipeline.
