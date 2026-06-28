# Danh mục Tài liệu Tham khảo (References)

Dưới đây là danh sách các công trình nghiên cứu khoa học liên quan trực tiếp đến kiến trúc và phương pháp của dự án, có thể sử dụng đưa vào phần **Tài liệu tham khảo** hoặc trích dẫn trong báo cáo:

### 1. Kiến trúc Trích xuất Đặc trưng và Đánh giá Backbone (Backbones & Ablation Studies)

**[1] Swin Transformer cho hình ảnh:**
* **Tên bài báo:** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
* **Tác giả:** Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. & Guo, B. (2021)
* **Link:** https://arxiv.org/abs/2103.14030
* **Ứng dụng trong đồ án:** Chứng minh tính hiệu quả của Swin-B so với CNN truyền thống trong việc biểu diễn đặc trưng phân cấp cho hình ảnh.

**[2] PhoBERT cho văn bản tiếng Việt:**
* **Tên bài báo:** PhoBERT: Pre-trained language models for Vietnamese
* **Tác giả:** Nguyen, D. Q., & Nguyen, A. T. (2020)
* **Link:** https://arxiv.org/abs/2003.00744
* **Ứng dụng trong đồ án:** Khẳng định PhoBERT (họ RoBERTa) là State-of-the-Art cho xử lý văn bản tiếng Việt.

**[3] Sự kết hợp Swin + RoBERTa trong phân tích cảm xúc:**
* **Tên bài báo:** Enhancing multimodal sentiment analysis reliability: SentiGuard+ with Dirichlet evidence and selective prediction
* **Tác giả:** Tạp chí Journal of King Saud University Computer and Information Sciences (2025/2026)
* **Link:** https://link.springer.com/article/10.1007/s44443-025-00447-y
* **Ứng dụng trong đồ án:** Làm cơ sở bảo vệ kiến trúc sử dụng luồng Swin Transformer (Image) + RoBERTa/PhoBERT (Text).

**[4] Xu hướng sử dụng nhiều đặc trưng (Swin + BERT vs Pre-trained OCR/ResNet):**
* **Tên bài báo:** Holistic Visual-Textual Sentiment Analysis with Prior Models
* **Tác giả:** arXiv preprint (2022)
* **Link:** https://arxiv.org/html/2211.12981v2
* **Ứng dụng trong đồ án:** So sánh hiệu năng của nhánh "Swin Transformer + BERT" với các baseline dùng đặc trưng pre-trained cũ (như ResNet, VGG). Bài báo kết luận multimodal input vượt trội hơn hẳn unimodal, đồng thời khẳng định Swin + họ BERT là một nhánh huấn luyện (trainable branch) mang lại sức mạnh vượt trội cho bài toán.

**[5] So sánh thực nghiệm các Backbone (ResNet, ViT) vs BERT:**
* **Tên bài báo:** RethinkingTMSC: An Empirical Study for Target-Oriented Multimodal Sentiment Classification
* **Tác giả:** Junjie Ye et al. (2023)
* **Link:** https://doi.org/10.48550/arxiv.2310.09596
* **Ứng dụng trong đồ án:** Cung cấp cơ sở thực nghiệm so sánh các vision encoder khác nhau (ResNet, ViT, Faster R-CNN) khi đi kèm với BERT. Bài báo chỉ ra rằng Text-only luôn chạy rất tốt, nhưng Visual-only thường kém, nhấn mạnh tầm quan trọng của việc phải chọn đúng kiến trúc Image Backbone phù hợp để bổ trợ cho Text.

**[6] So sánh kết hợp EfficientNet và họ Transformer (RoBERTa/BERT):**
* **Tên bài báo:** Multimodal Sentiment Analysis using Deep Learning Fusion Techniques and Transformers
* **Tác giả:** Muhaimin Bin Habib et al. (2024)
* **Link:** http://dx.doi.org/10.14569/IJACSA.2024.0150686
* **Ứng dụng trong đồ án:** Bài báo nghiên cứu sự kết hợp giữa các kiến trúc Image tiên tiến với Text (RoBERTa + EfficientNet-b3, RoBERTa + ResNet50, BERT + MobileNetV2). Kết quả chứng minh EfficientNet-b3 + RoBERTa cho độ chính xác cao nhất (75%). Điều này củng cố lý do tại sao dự án của bạn cần thực nghiệm và so sánh EfficientNet với Swin Transformer.

**[7] Tích hợp mô hình tiền huấn luyện CLIP và Cross-Attention:**
* **Tên bài báo:** Multimodal Sentiment Analysis Model Based on CLIP and Cross-attention
* **Tác giả:** Chen Yan et al. (Journal of Zhengzhou University, 2024)
* **Link:** http://gxb.zzu.edu.cn/en/oa/darticle.aspx?id=202306065
* **Ứng dụng trong đồ án:** Trực tiếp sử dụng Cross-attention để giao tiếp giữa image và text. Đặc biệt, bài báo cũng áp dụng cơ chế Uncertainty Loss để dung hợp đặc trưng, hoàn toàn khớp với cách xử lý loss đa khía cạnh (Kendall Uncertainty) đang làm trong đồ án.

**[8] Giới thiệu và Ứng dụng SigLIP cho hiểu ngữ nghĩa hình ảnh:**
* **Tên bài báo:** SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features
* **Tác giả:** Michael Tschannen et al. (2025)
* **Link:** https://arxiv.org/abs/2502.14786
* **Ứng dụng trong đồ án:** SigLIP (đặc biệt là phiên bản mới) cải tiến đáng kể so với CLIP nhờ thay đổi hàm loss (pairwise sigmoid loss thay vì contrastive loss toàn cục), cho phép huấn luyện hiệu quả hơn và khả năng bắt ngữ nghĩa (Semantic Understanding) tốt hơn. Điều này cung cấp bằng chứng cực kỳ vững chắc để giải thích việc đưa SigLIP vào đồ án dưới dạng một backbone thay thế cho các kiến trúc cũ như ResNet hay CLIP thông thường.

### 2. Cơ chế Dung hợp (Fusion Mechanism)

**[9] Sử dụng Cross-Attention cho Multimodal:**
* **Tên bài báo:** Multi-Modal Sentiment Analysis Based on Image and Text Fusion Based on Cross-Attention Mechanism
* **Nơi xuất bản:** MDPI Electronics (2024)
* **Link:** https://www.mdpi.com/2079-9292/13/11/2069
* **Ứng dụng trong đồ án:** Chứng minh Cross-Attention dung hợp đặc trưng ảnh và chữ tốt hơn các phương pháp cộng/nối thông thường.

**[10] Lấy Text làm trung tâm truy vấn Image:**
* **Tên bài báo:** Text-Anchored Residual Cross-Modal Fusion for Multimodal Sentiment Analysis
* **Nơi xuất bản:** MDPI Applied Sciences (2024)
* **Link:** https://www.mdpi.com/2076-3417/16/9/4514
* **Ứng dụng trong đồ án:** Giải thích tại sao Cross-Attention hiệu quả với review nhà hàng (văn bản chứa thông tin chính, ảnh làm bằng chứng bổ trợ).

### 3. Phương pháp Phân tích Đa khía cạnh & XAI (MABSA & Explainable AI)

**[11] Khung phân tích cảm xúc đa phương thức theo khía cạnh (MABSA):**
* **Tên bài báo:** Multimodal Aspect-Based Sentiment Analysis: A survey of tasks
* **Tác giả:** Fanghua Ye, et al. (Tạp chí Information Fusion, 2024)
* **Link:** https://doi.org/10.1016/j.inffus.2024.102552
* **Ứng dụng trong đồ án:** Khẳng định tính cấp thiết và xu hướng của bài toán phân tích đánh giá nhà hàng/sản phẩm theo từng khía cạnh cụ thể (Food, Price, Service, Atmosphere) dựa trên cả ảnh và văn bản, thay vì chỉ phân tích cảm xúc chung chung.

**[12] Trí tuệ nhân tạo có thể giải thích (XAI) cho mô hình đa phương thức:**
* **Tên bài báo:** Multimodal Explainable Artificial Intelligence: A Comprehensive Review of Methodological Advances and Future Research Directions
* **Tác giả:** M. Asif et al. (2024)
* **Link:** https://arxiv.org/abs/2402.05154
* **Ứng dụng trong đồ án:** Cung cấp cơ sở lý thuyết vững chắc cho toàn bộ Pipeline XAI (Phase 2-6) trong đồ án. Bài báo ủng hộ việc sử dụng kết hợp các phương pháp giải thích (LIME, SHAP, Grad-CAM, Attention Weights) để đánh giá mức độ đóng góp của từng nhánh dữ liệu (Modality Contribution) và tính minh bạch của mô hình Hộp đen (Black-box).

### 4. Tối ưu hóa và Đánh giá (Optimization)

**[13] Uncertainty Weighting cho Multi-task Learning:**
* **Tên bài báo:** Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
* **Tác giả:** Kendall, A., Gal, Y., & Cipolla, R. (2018)
* **Link:** https://arxiv.org/abs/1705.07115
* **Ứng dụng trong đồ án:** Giải thích công thức và lý do sử dụng `HomoscedasticUncertaintyLoss` (0.5 * e^-log_var * loss + 0.5 * log_var) để cân bằng trọng số giữa 5 khía cạnh.
