# Fashion MNIST Classification

Dự án AI phân loại quần áo thời trang sử dụng CNN và dataset Fashion MNIST.

## Các bước chính

1. Chuẩn bị dữ liệu (Fashion MNIST từ CSV Kaggle)
2. Tiền xử lý dữ liệu: chuẩn hóa, reshape, chia train/val/test
3. Xây dựng mô hình CNN với Keras
4. Huấn luyện với Adam, sparse categorical crossentropy, accuracy, early stopping
5. Đánh giá: test accuracy, classification report, confusion matrix
6. Trực quan hóa: biểu đồ accuracy/loss, hiển thị ảnh dự đoán
7. Lưu mô hình
8. So sánh với 2 thuật toán khác (KNN và Logistic Regression): huấn luyện, đánh giá accuracy, vẽ biểu đồ so sánh

## Lưu ý

- Không commit thư mục `venv/` lên Git. Nó đã được thêm vào `.gitignore`.
- Dataset CSV có thể lớn, cân nhắc không commit nếu repository cần nhẹ.

## Huấn luyện mô hình

```bash
python train.py
```

## Chạy web app

```bash
streamlit run app.py
```
