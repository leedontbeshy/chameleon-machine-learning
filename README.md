# BankChurners Preprocessing for Chameleon

Tiền xử lí dữ liệu

## Cài thư viện

```bash
pip install -r requirements.txt
```

## Chạy tiền xử lí

```bash
python src/preprocessing.py
```

Sau khi chạy, script tạo/cập nhật các file trong `data/processed`:

- `bank_churners_features.csv`: dữ liệu đã xử lí, toàn số, dùng để chạy Chameleon.
- `bank_churners_labels.csv`: nhãn churn giữ riêng, chỉ dùng để đánh giá/giải thích cụm sau khi phân cụm.
- `preprocessing_metadata.json`: thông tin về các bước xử lí và danh sách cột.

## Dùng cho Chameleon

```python
import pandas as pd

features = pd.read_csv("data/processed/bank_churners_features.csv")
X = features.to_numpy()
```

Không đưa `bank_churners_labels.csv` vào lúc phân cụm. File label chỉ dùng sau khi có kết quả cụm.

Chi tiết phần tiền xử lí nằm trong `docs/preprocessing.md`.
