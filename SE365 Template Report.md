**ĐẠI HỌC QUỐC GIA THÀNH PHỐ HỒ CHÍ MINH**

**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN**

**KHOA CÔNG NGHỆ PHẦN MỀM**

![Untitled 1](data:image/jpeg;base64...)

**MÔN HỌC SÂU ỨNG DỤNG**

**TRONG PHÁT TRIỂN PHẦN MỀM**

**ĐỀ TÀI: TÊN ĐỀ TÀI**

|  |  |
| --- | --- |
| GVHD: | TS. Đỗ Trọng Hợp  ThS. Nguyễn Ngọc Quí |

**Sinh viên thực hiện**

|  |  |  |
| --- | --- | --- |
| **Nhóm: X** | | |
| **STT** | **Họ và tên** | **MSSV** |
| 1 | Nguyễn Văn A |  |
| 2 | Trần Thị B |  |
| 3 |  |  |

🙡🙢 Tp. Hồ Chí Minh, 06/2026 🙠🙣

**MỤC LỤC**

[TÓM TẮT TIẾN ĐỘ 3](#_Toc233118629)

[TÓM TẮT 4](#_Toc233118630)

[DANH MỤC BẢNG 5](#_Toc233118631)

[DANH MỤC HÌNH ẢNH 6](#_Toc233118632)

[Chương 1: TỔNG QUAN ĐỀ TÀI 7](#_Toc233118633)

[Chương 2: CÔNG TRÌNH NGHIÊN CỨU LIÊN QUAN 8](#_Toc233118634)

[Chương 3: ĐỊNH NGHĨA BÀI TOÁN VÀ XÂY DỰNG BỘ DỮ LIỆU 9](#_Toc233118635)

[Chương 4: PHƯƠNG PHÁP ĐỀ XUẤT 10](#_Toc233118636)

[Chương 5: THỰC NGHIỆM 11](#_Toc233118637)

[Chương 6: KẾT QUẢ VÀ BÀN LUẬN 12](#_Toc233118638)

[Chương 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN 13](#_Toc233118639)

[TÀI LIỆU THAM KHẢO 14](#_Toc233118640)

# TÓM TẮT TIẾN ĐỘ

|  |  |  |
| --- | --- | --- |
| **Khía cạnh** | **Các công việc đã hoàn thành** | **Các công việc tiếp theo** |
| Bộ dữ liệu |  |  |
| Thực nghiệm |  |  |
| Báo cáo |  |  |

# TÓM TẮT

* Background/problem (1–2 câu)
* Gap (1 câu) — cái chưa ai làm
* Proposed method (2–3 câu) — pipeline + model
* Dataset (1 câu) — quy mô, nguồn
* Kết quả định lượng (1–2 câu) — số liệu cụ thể (F1, accuracy, ...)
* Contribution/impact (1 câu)

# DANH MỤC BẢNG

# DANH MỤC HÌNH ẢNH

# TỔNG QUAN ĐỀ TÀI

* Bối cảnh & động lực — tại sao bài toán quan trọng (du lịch VN, big data từ social media...)
* Thách thức hiện tại — dữ liệu tiếng Việt ít, mô hình đa nhiệm khó, xử lý real-time khó
* Research gap — nêu rõ 2–3 gap cụ thể (không model nào kết hợp đa tác vụ + streaming cho tiếng Việt...)
* Contributions — liệt kê dạng bullet (3–5 điểm), mỗi điểm 1 câu, càng cụ thể & đo lường được càng tốt
* Cấu trúc báo cáo gồm những phần

**Lưu ý: các trích dẫn cần phải đúng, không dùng trích dẫn giả và có cite cụ thể vào từng dòng của bài. Ví dụ: Hop et al. (2026) [1] đã giới thiệu bộ dữ liệu …**

**Mỗi nhận định hay mỗi câu khẳng định cần phải có minh chứng cụ thể từ các nghiên cứu trước không tự dưng khẳng định vô căng cứ. Ví dụ: Bộ dữ liệu này là bộ dữ liệu đầu tiên ở Việt Nam về chủ đề này (phải có minh chứng là chỉ có dataset ở nước ngoài và trích dẫn các bộ dataset đó). Đây là một đề tài cấp thiết của xã hội (Đưa minh chứng cấp thiết từ các nghiên cứu hoặc từ báo chí gì đó đề cập đến, báo chí phải uy tín không dùng báo lá cải và không rõ nguồn gốc)**

# CÔNG TRÌNH NGHIÊN CỨU LIÊN QUAN

Chia theo trục chủ đề hoặc nhánh kỹ thuật, không liệt kê rời rạc từng paper:

* Theo hướng tiếp cận (ví dụ: rule-based vs statistical vs deep learning vs LLM-based)
* Theo bài toán con liên quan
* Theo các bộ dataset trước đó và so sánh với bộ dataset mà mình xây dựng
* Cuối cùng suy ra được Research Gap Summary (đoạn cuối, đối chiếu trực tiếp các công trình đã nêu với contribution của bài)

**Lưu ý: các trích dẫn cần phải đúng, không dùng trích dẫn giả và có cite cụ thể vào từng dòng của bài. Ví dụ: Hop et al. (2026) [1] đã giới thiệu bộ dữ liệu …**

# ĐỊNH NGHĨA BÀI TOÁN VÀ XÂY DỰNG BỘ DỮ LIỆU

* Định nghĩa hình thức bài toán bằng ký hiệu toán học (input, output, không gian giả thuyết ...)
* Mô tả trực tiếp bằng hình ảnh và text với ngắn gọn đầu vào và đầu ra của bài toán.
* Mô tả cụ thể quy trình xây dựng bộ dữ liệu từ thu thập tiền xử lý và gán nhãn, đánh giá quy trình gán nhãn, huấn luyện đội ngũ gán nhãn.
* Định nghĩa bộ dataset, có các cột nào (thuộc tính), nhãn như thế nào, định nghĩa nhãn như thế nào. Phân tích bộ dataset sau hoàn thành xây dựng, thống kê và phân tích bộ dataset, có bị mất cân bằng hay không đưa ra hướng xử lý để thực nghiệm về sau.

# PHƯƠNG PHÁP ĐỀ XUẤT

* Tổng quan kiến trúc/pipeline (sơ đồ tổng thể)
* Mô tả từng thành phần/module (mỗi module một subsection, có công thức toán nếu cần)
* Thuật toán (pseudocode nếu cần thể hiện quy trình huấn luyện/suy luận)
* Phân tích độ phức tạp/lý thuyết (nếu phù hợp — vd. complexity analysis, theoretical guarantees)

# THỰC NGHIỆM

* Evaluation metrics (Accuracy, F1-macro/micro, Precision/Recall — viết công thức)
* Baselines (liệt kê model so sánh)
* Implementation details (hyperparameters, hardware, framework versions)

# KẾT QUẢ VÀ BÀN LUẬN

* Biểu đồ thể hiện loss của việc train và val
* Đánh giá trên nhiều độ đo như F1 weighted, micro, macro, Recall weighted, micro, macro, .... AUC ROC, PR ROC, chạy cross validation ít nhất 5 fold.
* Main results (bảng so sánh các model, biểu đồ)
* Ablation study (bắt buộc bỏ từng module xem ảnh hưởng)
* Error analysis (case study định tính — ví dụ comment bị phân loại sai, vì sao)
* Thời gian huấn luyện, thời gian dự đoán tập test, 1 câu.
* So sánh với các nghiên cứu trước cùng có bộ dataset để đánh giá mức độ adapt với mô hình của mình trên dataset của nghiên cứu trước.
* Discussion — diễn giải ý nghĩa kết quả, liên hệ lại RQ/gap ở phần Introduction.

**Lưu ý: cần lưu lại mô hình khi huấn luyện và ghi chú cẩn thận.**

# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

* Những gì đóng góp được thì tổng kết lại
* Hạn chế dữ liệu (kích thước, domain bias)
* Hạn chế phương pháp (chưa generalize sang ngôn ngữ khác)
* Hạn chế đánh giá (chưa test human evaluation quy mô lớn...)
* Hướng phát triển tương lai (cần cite 1-2 bài báo để người ta biết được mình có tìm đến hướng phát triển)

# TÀI LIỆU THAM KHẢO