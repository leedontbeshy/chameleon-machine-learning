# Báo cáo tiền xử lí dữ liệu BankChurners cho thuật toán Chameleon

## 1. Mục tiêu

Phần tiền xử lí này chuẩn bị dữ liệu `BankChurners.csv` để dùng cho thuật toán phân cụm Chameleon. Chameleon là thuật toán phân cụm không giám sát và thường làm việc dựa trên độ tương đồng/khoảng cách giữa các điểm dữ liệu, nên dữ liệu đầu vào cần:

- không chứa cột định danh hoặc nhãn đáp án;
- toàn bộ đặc trưng ở dạng số;
- các cột số được đưa về thang đo hợp lý;
- các cột phân loại được mã hóa thành dạng máy học có thể xử lí.

Phần này không triển khai thuật toán Chameleon, mà chỉ tạo dữ liệu đầu vào sạch cho bước phân cụm phía sau.

## 2. Mô tả dataset

Dataset sử dụng là Credit Card Customers trên Kaggle: <https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers>.

File gốc trong project:

```text
data/BankChurners.csv
```

Thông tin chính:

- Số dòng: 10,127 khách hàng.
- Số cột ban đầu: 23 cột.
- Nhãn churn: `Attrition_Flag`.
- Phân bố nhãn trong dữ liệu gốc:
  - `Existing Customer`: 8,500 dòng.
  - `Attrited Customer`: 1,627 dòng.

Các cột phân loại quan trọng:

- `Gender`
- `Education_Level`
- `Marital_Status`
- `Income_Category`
- `Card_Category`

Một số cột có giá trị `Unknown`, ví dụ `Education_Level`, `Marital_Status`, `Income_Category`. Trong pipeline này, `Unknown` được giữ lại như một nhóm riêng thay vì tự đoán hoặc xóa dòng.

## 3. Các quyết định tiền xử lí

### 3.1. Loại cột không dùng làm feature

Các cột sau không được đưa vào ma trận đặc trưng dùng cho Chameleon:

- `CLIENTNUM`: đây là mã định danh khách hàng, không thể hiện hành vi hay đặc điểm cụm.
- `Attrition_Flag`: đây là nhãn churn. Vì Chameleon là thuật toán không giám sát, không đưa nhãn vào feature để tránh thuật toán "nhìn đáp án".
- Hai cột bắt đầu bằng `Naive_Bayes_Classifier_`: đây là kết quả mô hình có sẵn trong dataset Kaggle, có nguy cơ gây rò rỉ thông tin.
- `Avg_Open_To_Buy`: cột này gần như trùng thông tin với `Credit_Limit` (`corr ≈ 0.996`), nên bỏ để tránh nhóm đặc trưng tín dụng bị tính lặp trong khoảng cách.

`Attrition_Flag` vẫn được lưu riêng trong file label để sau khi phân cụm có thể đánh giá hoặc diễn giải cụm nào có nhiều khách hàng rời bỏ.

### 3.2. Xử lý cột phân loại

Các cột phân loại được chuyển thành số bằng one-hot encoding. Ví dụ, `Gender` có thể được tách thành các cột như:

```text
Gender_F
Gender_M
```

Cách này giúp thuật toán tính khoảng cách trên dữ liệu số mà không gán thứ tự giả cho các nhóm phân loại.

Giá trị `Unknown` được giữ lại thành một nhóm riêng, ví dụ:

```text
Education_Level_Unknown
Income_Category_Unknown
Marital_Status_Unknown
```

Lý do: nếu tự thay `Unknown` bằng giá trị phổ biến nhất, dữ liệu có thể bị sai lệch vì ta đang tự đoán thông tin không có thật.

### 3.3. Chuẩn hóa cột số

Các cột số được chuẩn hóa bằng `RobustScaler` của scikit-learn.

RobustScaler dùng median và IQR, phù hợp với dữ liệu tài chính/giao dịch vì các cột như `Credit_Limit`, `Total_Trans_Amt`, `Total_Revolving_Bal` có độ lệch lớn và có thể có giá trị rất cao so với phần còn lại.

Các cột one-hot giữ nguyên giá trị 0/1.

## 4. File đầu ra

Sau khi chạy:

```bash
python src/preprocessing.py
```

Pipeline tạo các file sau trong `data/processed`:

```text
data/processed/bank_churners_features.csv
data/processed/bank_churners_labels.csv
data/processed/preprocessing_metadata.json
```

Ý nghĩa từng file:

- `bank_churners_features.csv`: ma trận đặc trưng toàn số, dùng trực tiếp làm đầu vào cho Chameleon.
- `bank_churners_labels.csv`: chứa `CLIENTNUM`, `Attrition_Flag`, `Attrition_Label`; chỉ dùng để đánh giá hoặc giải thích kết quả phân cụm.
- `preprocessing_metadata.json`: lưu thông tin về số dòng/cột, cột bị loại, cột được scale, cột được one-hot, phân bố nhãn, số lượng `Unknown`, và thống kê cơ bản.

## 5. Cách dùng cho bước Chameleon

Ví dụ đọc dữ liệu đã xử lí:

```python
import pandas as pd

features = pd.read_csv("data/processed/bank_churners_features.csv")
labels = pd.read_csv("data/processed/bank_churners_labels.csv")

X = features.to_numpy()
```

`X` là ma trận số có thể dùng để xây đồ thị k-láng giềng gần nhất hoặc làm đầu vào cho bước phân cụm Chameleon.

Không dùng `Attrition_Flag` hoặc `Attrition_Label` để phân cụm. Chỉ dùng chúng sau khi có kết quả cụm, ví dụ để xem cụm nào có tỷ lệ khách hàng rời bỏ cao.

## 6. Kiểm tra sau tiền xử lí

Script `src/preprocessing.py` tự kiểm tra các điều kiện chính:

- Số dòng feature bằng số dòng dữ liệu gốc.
- Số dòng label bằng số dòng dữ liệu gốc.
- Feature không còn giá trị thiếu.
- Feature chỉ chứa dữ liệu số.
- Feature không chứa các cột bị loại như `CLIENTNUM`, `Attrition_Flag`, `Avg_Open_To_Buy`, hoặc cột `Naive_Bayes_Classifier_...`.
- Phân bố nhãn vẫn khớp dữ liệu gốc: 8,500 khách hiện tại và 1,627 khách đã rời bỏ.

