# Dữ liệu crawl trong thư mục `data/`

Thư mục `data/` hiện có 2 batch crawl review chính để phục vụ phân tích review có ảnh trên Tiki.

## 1. Batch `bach_hoa_top10_deep`

Đường dẫn:
- `/Users/abc/Documents/SE/data/bach_hoa_top10_deep`

Mô tả:
- Đây là batch thuộc nhóm Bách Hóa Online.
- Batch được tạo bằng cách quét sâu nhiều trang listing của category, gom sản phẩm, sắp xếp theo số lượng review, lấy top 10 rồi crawl theo chế độ `review-only`.
- Dữ liệu phù hợp để phân tích review có ảnh trong nhóm hàng bách hóa, thực phẩm đóng gói và đồ tiêu dùng nhanh.

Thống kê:
- Số sản phẩm đã crawl: `10`
- Tổng số review đã lưu: `2674`
- Tổng số ảnh review đã tải: `3278`

Cấu trúc dữ liệu:
- `data/bach_hoa_top10_deep/{product_id}/reviews.csv`
- `data/bach_hoa_top10_deep/{product_id}/customers.csv`
- `data/bach_hoa_top10_deep/{product_id}/buy_historys.csv`
- `data/bach_hoa_top10_deep/{product_id}/reviews_checkpoint.json`
- `data/bach_hoa_top10_deep/{product_id}/review_images/{product_id}/{review_id}/*`

## 2. Batch `binh_ly_giu_nhiet`

Đường dẫn:
- `/Users/abc/Documents/SE/data/binh_ly_giu_nhiet`

Mô tả:
- Đây là batch thuộc nhóm bình và ly giữ nhiệt.
- Mỗi sản phẩm được crawl vào một thư mục riêng theo `product_id`.
- Dữ liệu phù hợp để phân tích các aspect như giữ nhiệt, rò nước, nắp, trầy xước, màu sắc thực tế và chất lượng hoàn thiện.

Thống kê:
- Số sản phẩm đã crawl: `10`
- Tổng số review đã lưu: `3362`
- Tổng số ảnh review đã tải: `4470`

Cấu trúc dữ liệu:
- `data/binh_ly_giu_nhiet/{product_id}/reviews.csv`
- `data/binh_ly_giu_nhiet/{product_id}/customers.csv`
- `data/binh_ly_giu_nhiet/{product_id}/buy_historys.csv`
- `data/binh_ly_giu_nhiet/{product_id}/reviews_checkpoint.json`
- `data/binh_ly_giu_nhiet/{product_id}/review_images/{product_id}/{review_id}/*`

## 3. Các file chính trong mỗi thư mục sản phẩm

`reviews.csv`
- Chứa dữ liệu review.
- Các cột quan trọng thường dùng: `id`, `product_id`, `content`, `rating`, `images`, `review_image_count`, `has_review_images`, `first_review_image_url`.

`customers.csv`
- Chứa thông tin người dùng đã để lại review.
- Các cột thường dùng: `id`, `full_name`, `avatar_url`, `total_review`, `total_thank`, `purchased`.

`buy_historys.csv`
- Chứa dữ liệu lịch sử mua hàng được map từ payload review.
- Các cột thường dùng: `customer_id`, `product_id`, `seller_id`, `seller_name`, `review_id`, `purchased_at`, `delivery_date`.

`reviews_checkpoint.json`
- Dùng để đánh dấu trạng thái crawl review của sản phẩm trong thư mục output tương ứng.

`review_images/...`
- Chứa ảnh review đã tải về máy.
- Thư mục được tổ chức theo `product_id/review_id`.

## 4. Lưu ý sử dụng

- Hai batch này được lưu theo kiểu mỗi sản phẩm một thư mục riêng, không gộp toàn bộ review vào một file tổng.
- Khi bật tải ảnh review, dữ liệu được lưu tập trung vào các review có ảnh.
- Nếu cần phân tích toàn batch, cần đọc tất cả `reviews.csv` trong các thư mục con rồi gộp lại.
