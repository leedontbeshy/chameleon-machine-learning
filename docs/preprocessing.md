# Tiền xử lí dữ liệu BankChurners cho thuật toán Chameleon

## 1. Mục tiêu 

Dữ liệu gốc là file `data/BankChurners.csv`, lấy từ bộ Credit Card Customers trên Kaggle: <https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers>.

## 2. Dữ liệu ban đầu 

File `BankChurners.csv` có:

- 10,127 dòng, mỗi dòng tương ứng với một khách hàng.
- 23 cột ban đầu.
- Một cột nhãn là `Attrition_Flag`, cho biết khách hàng còn sử dụng dịch vụ hay đã rời bỏ.

Phân bố nhãn trong dữ liệu gốc:

| Nhãn | Số lượng |
|---|---:|
| `Existing Customer` | 8,500 |
| `Attrited Customer` | 1,627 |

Trong dữ liệu có cả cột số và cột phân loại. Ví dụ:

- Cột số: `Customer_Age`, `Credit_Limit`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Avg_Utilization_Ratio`.
- Cột phân loại: `Gender`, `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category`.

Một số cột phân loại có giá trị `Unknown`. Cụ thể:

| Cột | Số dòng `Unknown` | Tỉ lệ xấp xỉ |
|---|---:|---:|
| `Education_Level` | 1,519 | 15.0% |
| `Marital_Status` | 749 | 7.4% |
| `Income_Category` | 1,112 | 11.0% |

Dataset không có ô trống thật theo kiểu `NaN`, nhưng có các giá trị `Unknown`. Vì vậy, vấn đề chính không phải là thiếu dữ liệu dạng rỗng, mà là có những thông tin chưa biết được ghi thành một nhóm riêng.

## 3. Các vấn đề cần xử lí trước khi dùng Chameleon

Vấn đề đầu tiên là dữ liệu có cột định danh `CLIENTNUM`. Đây chỉ là mã số của khách hàng. Hai khách hàng có mã gần nhau không có nghĩa là họ giống nhau về hành vi, thu nhập hay giao dịch. Nếu đưa `CLIENTNUM` vào thuật toán, khoảng cách giữa các khách hàng sẽ bị ảnh hưởng bởi một con số không có ý nghĩa phân cụm.

Vấn đề thứ hai là cột `Attrition_Flag`. Đây là nhãn churn, tức là thông tin cho biết khách hàng đã rời bỏ hay chưa. Vì Chameleon là thuật toán không giám sát, ta không được đưa nhãn này vào feature. Nếu đưa vào, thuật toán có thể phân cụm dựa theo chính đáp án, làm kết quả nhìn có vẻ tốt nhưng không còn đúng bản chất phân cụm.

Vấn đề thứ ba là dataset có hai cột bắt đầu bằng `Naive_Bayes_Classifier_...`. Đây là hai cột kết quả mô hình có sẵn trong bộ dữ liệu Kaggle. Chúng không phải đặc trưng gốc của khách hàng, mà là thông tin đã được một mô hình khác tính ra. Nếu giữ lại, dữ liệu đầu vào có nguy cơ bị rò rỉ thông tin và làm sai ý nghĩa bài toán.

Vấn đề thứ tư là cột `Avg_Open_To_Buy` gần như trùng thông tin với `Credit_Limit`. Khi kiểm tra tương quan, hai cột này có hệ số tương quan xấp xỉ `0.996`, tức là gần như đi cùng nhau. Nếu giữ cả hai, nhóm thông tin liên quan đến hạn mức tín dụng sẽ bị tính lặp trong khoảng cách, làm thuật toán chú ý quá nhiều vào một loại đặc trưng.

Vấn đề cuối cùng là dữ liệu có cả số và chữ. Thuật toán tính khoảng cách không thể trực tiếp xử lí các giá trị như `Graduate`, `Married`, `$60K - $80K`. Vì vậy, các cột phân loại cần được chuyển sang dạng số.

## 4. Cách xử lí dữ liệu

### 4.1. Tách nhãn churn ra khỏi feature

Pipeline tạo riêng file label từ hai cột:

- `CLIENTNUM`
- `Attrition_Flag`

Sau đó tạo thêm cột `Attrition_Label`:

| Giá trị gốc | Giá trị mã hóa |
|---|---:|
| `Existing Customer` | 0 |
| `Attrited Customer` | 1 |

File label này không dùng để phân cụm. Nó chỉ dùng sau khi đã có kết quả cụm, ví dụ để kiểm tra cụm nào có tỉ lệ khách hàng rời bỏ cao hơn.

### 4.2. Loại các cột không phù hợp khỏi feature

Các cột bị loại khỏi ma trận feature gồm:

- `CLIENTNUM`
- `Attrition_Flag`
- Hai cột `Naive_Bayes_Classifier_...`
- `Avg_Open_To_Buy`

Lý do chung là các cột này hoặc không có ý nghĩa cho khoảng cách, hoặc chứa thông tin nhãn/mô hình có sẵn, hoặc bị trùng thông tin quá mạnh với cột khác.

### 4.3. Giữ `Unknown` như một nhóm riêng

Với các cột như `Education_Level`, `Marital_Status`, `Income_Category`, pipeline không thay `Unknown` bằng giá trị phổ biến nhất và cũng không xóa các dòng đó.

Cách làm được chọn là giữ `Unknown` như một nhóm riêng. Ví dụ sau khi one-hot encoding, ta có thể có các cột như:

```text
Education_Level_Unknown
Marital_Status_Unknown
Income_Category_Unknown
```

Cách này hợp lý hơn vì ta không tự đoán thông tin của khách hàng. Nếu thay `Unknown` bằng giá trị phổ biến nhất, dữ liệu sẽ sạch hơn về mặt hình thức nhưng lại có thể làm sai sự thật. Nếu xóa toàn bộ dòng có `Unknown`, ta sẽ mất khá nhiều dữ liệu, đặc biệt là cột `Education_Level` có khoảng 15% dòng là `Unknown`.

### 4.4. One-hot encoding các cột phân loại

Các cột phân loại được chuyển thành số bằng one-hot encoding:

- `Gender`
- `Education_Level`
- `Marital_Status`
- `Income_Category`
- `Card_Category`

Ví dụ, cột `Gender` ban đầu có hai giá trị `F` và `M`. Sau khi mã hóa, nó trở thành các cột nhị phân như:

```text
Gender_F
Gender_M
```

Nếu một khách hàng là nữ, `Gender_F = 1` và `Gender_M = 0`. Nếu là nam thì ngược lại.

Không dùng cách mã hóa 0, 1, 2, 3 trực tiếp cho các nhóm phân loại vì cách đó vô tình tạo ra thứ tự giả. Ví dụ, nếu mã hóa `Education_Level` thành `Graduate = 1`, `Doctorate = 2`, `High School = 3`, thuật toán có thể hiểu nhầm rằng giá trị 3 lớn hơn 1 theo nghĩa khoảng cách số học. One-hot encoding tránh vấn đề này.

### 4.5. Chuẩn hóa các cột số bằng RobustScaler

Các cột số trong dataset có thang đo rất khác nhau. Ví dụ:

- `Customer_Age` nằm khoảng vài chục.
- `Credit_Limit` có thể lên tới hàng chục nghìn.
- `Avg_Utilization_Ratio` nằm trong khoảng từ 0 đến gần 1.

Nếu để nguyên, các cột có giá trị lớn như `Credit_Limit` hoặc `Total_Trans_Amt` sẽ chi phối khoảng cách giữa các khách hàng. Khi đó, thuật toán có thể gần như chỉ nhìn vào tiền/hạn mức mà bỏ qua các đặc trưng nhỏ hơn nhưng vẫn quan trọng.

Pipeline dùng `RobustScaler` để chuẩn hóa cột số. RobustScaler dùng median và IQR thay vì trung bình và độ lệch chuẩn. Cách này phù hợp với dữ liệu tài chính/giao dịch vì dữ liệu thường lệch và có những khách hàng có giá trị giao dịch hoặc hạn mức rất cao.

Các cột one-hot sau khi mã hóa vẫn giữ giá trị 0/1, không cần scale lại.

## 5. File đầu ra sau tiền xử lí

Chạy lệnh:

```bash
python src/preprocessing.py
```

Script sẽ tạo các file trong thư mục `data/processed`:

```text
data/processed/bank_churners_features.csv
data/processed/bank_churners_labels.csv
data/processed/preprocessing_metadata.json
```

Ý nghĩa từng file:

| File | Ý nghĩa |
|---|---|
| `bank_churners_features.csv` | Ma trận đặc trưng toàn số, dùng làm đầu vào cho Chameleon. |
| `bank_churners_labels.csv` | Lưu `CLIENTNUM`, `Attrition_Flag`, `Attrition_Label` để đánh giá cụm sau này. |
| `preprocessing_metadata.json` | Lưu thông tin về các bước xử lí, danh sách cột, thống kê dữ liệu và file được tạo. |

Sau khi xử lí, file feature có:

- 10,127 dòng.
- 36 cột feature.
- Không có giá trị thiếu.
- Tất cả các cột đều là dạng số.
- Không chứa `CLIENTNUM`, `Attrition_Flag`, `Avg_Open_To_Buy` hoặc các cột `Naive_Bayes_Classifier_...`.

## 6. Cách dùng dữ liệu cho bước Chameleon

Ở bước cài đặt hoặc chạy Chameleon, chỉ đọc file feature:

```python
import pandas as pd

features = pd.read_csv("data/processed/bank_churners_features.csv")
X = features.to_numpy()
```

`X` là ma trận số có thể dùng để xây đồ thị k-láng giềng gần nhất hoặc đưa vào bước phân cụm của Chameleon.

Nếu muốn đánh giá ý nghĩa cụm sau khi phân cụm xong, đọc thêm file label:

```python
labels = pd.read_csv("data/processed/bank_churners_labels.csv")
```

Ví dụ sau khi có cột kết quả cụm, ta có thể ghép với `labels` để xem mỗi cụm có bao nhiêu khách `Existing Customer` và bao nhiêu khách `Attrited Customer`. Việc này giúp diễn giải cụm, nhưng không được dùng nhãn churn trong lúc phân cụm.

## 7. Các kiểm tra trong script

Script `src/preprocessing.py` có các bước kiểm tra để tránh xuất dữ liệu sai:

- Kiểm tra file input có tồn tại không.
- Kiểm tra các cột bắt buộc có trong dataset không.
- Kiểm tra đúng 2 cột `Naive_Bayes_Classifier_...`.
- Kiểm tra nhãn churn chỉ có các giá trị đã biết.
- Kiểm tra feature không có missing thật.
- Kiểm tra số dòng feature và label bằng số dòng dữ liệu gốc.
- Kiểm tra feature toàn là số.
- Kiểm tra các cột bị loại không xuất hiện lại trong feature.
- Kiểm tra phân bố nhãn vẫn là 8,500 `Existing Customer` và 1,627 `Attrited Customer`.

Các kiểm tra này giúp pipeline đáng tin cậy hơn. Nếu sau này dataset bị thay đổi hoặc thiếu cột, script sẽ báo lỗi thay vì âm thầm tạo output sai.


