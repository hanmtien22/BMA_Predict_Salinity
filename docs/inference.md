# Dự báo độ mặn từ đặc trưng Landsat 8

Chạy dự báo độ mặn (‰) theo lô cho từng điểm quan trắc bằng mô hình cây đã huấn luyện và Gaussian Process (GP) hiệu chỉnh phần dư theo không gian, mùa và thời gian. Mỗi dòng đầu vào mô tả một điểm vào một ngày. Người chạy cần có quyền truy cập bộ model riêng và bảng đặc trưng đã xử lý theo quy trình huấn luyện.

## Chạy lần đầu

Môi trường huấn luyện dùng Python **3.12.13**. Trên Windows, mở PowerShell tại thư mục chứa `predict.py` và chạy các lệnh sau với Python 3.12 đã cài.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py --model-dir ..\model-paramater --input observations.csv --output predictions.csv
```

Trên Linux hoặc macOS, tạo môi trường bằng `python3.12 -m venv .venv` rồi dùng `.venv/bin/python` cho hai lệnh tiếp theo. Trên macOS, LightGBM còn cần OpenMP (`libomp`).

Trỏ `--model-dir` đến thư mục riêng chứa **cả** `parameters.json` và `fitted_models.pkl` từ cùng một lần xuất. Chỉ nạp pickle từ nguồn bạn tin cậy vì quá trình giải tuần tự có thể thực thi mã. CLI đối chiếu SHA-256 với JSON trước khi nạp, rồi kiểm tra mã bộ model và danh sách đặc trưng. Mỗi lần chạy thành công sẽ in tên model, mã bộ model và số dòng đã xử lý.

## Hợp đồng dữ liệu

CSV dùng UTF-8, dấu phẩy và dòng tiêu đề. Đầu vào gồm `date` theo dạng `YYYY-MM-DD` và 35 cột số hữu hạn dưới đây. Thứ tự cột đầu vào tùy ý.

```text
DEM,Lat,Long,EVI,NDBI,NDMI,NDSI,NDVI,NDWI,VSSI,SAVI,COSRI,EVI2,MSI,ND23,ND47,SI1,SI2,SI3,SI4,SI5,SI6,SI7,SI8,SI9,CA,B,G,R,NIR,PCA1,PCA2,PCA3,PCA4,PCA5
```

`Lat` và `Long` là vĩ độ và kinh độ theo độ. Các băng phổ, chỉ số và điểm PCA phải dùng cùng thang đo và phép biến đổi như khi huấn luyện. Đặc biệt, nếu tính PCA cho ảnh mới, cần dùng phép chiếu PCA gốc. Bộ model nhận các đặc trưng đã tính sẵn. CLI tự tạo ba biến thời gian từ ngày quan trắc, đồng thời dùng tâm và độ lệch chuẩn đã lưu để chuẩn hóa tọa độ cho GP.

File kết quả giữ thứ tự dòng và các cột đầu vào, thêm `salinity_pred_per_mille`. Bạn có thể giữ một cột mã trạm để ghép kết quả với dữ liệu khác. Hãy chọn đường dẫn đầu ra riêng trong thư mục có sẵn. Khi chạy thành công, CLI thay thế file ở đường dẫn đầu ra. Nếu một dòng không hợp lệ, lệnh trả mã lỗi và giữ kết quả của lần chạy trước.

## Mô hình và phạm vi đánh giá

Mặc định là **PSO LightGBM + GP**. Với bản huấn luyện có 359 mẫu trước 01/03/2020, mô hình đạt RMSE **3,654 ‰**, MAE **2,739 ‰** và R² **0,815** trên 151 quan trắc từ tháng 3 đến tháng 6/2020, theo bảng đánh giá đi kèm bộ model. Đây là kết quả trên giai đoạn lịch sử đó. Khi đưa vào vận hành cho mùa hoặc khu vực mới, nên đối chiếu dự báo với độ mặn đo thực địa của giai đoạn và khu vực tương ứng.

GP học phần dư từ các dự báo ngoài fold tiến theo thời gian của mô hình cây. Khi suy luận, CLI chặn dự báo cây ở mức tối thiểu 0, cộng hiệu chỉnh GP, rồi chặn kết quả ở mức tối thiểu 0, đúng theo notebook huấn luyện. Dùng `--model "PSO RF"` để chọn cây khác. Bản xuất này có các tên `Decision Tree`, `Bagging`, `PSO Bagging`, `Random Forest`, `PSO RF`, `PSO XGB` và `PSO LightGBM`; mỗi tên dùng GP tương ứng.

## Kiểm thử trước khi phát hành

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GitHub Actions chạy các test với dữ liệu nhân tạo. Để tái hiện kết quả lịch sử, đặt thêm hai biến môi trường rồi chạy lại cùng lệnh. Bước này cần `test_predictions.csv` bên cạnh hai file model riêng.

```powershell
$env:SALINITY_MODEL_DIR = (Resolve-Path ..\model-paramater).Path
$env:SALINITY_SOURCE_CSV = (Resolve-Path ..\result_csv\Check_BMA_Chuanhoa_LS8_20192020_159.csv).Path
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_reference.py -v
```

Test đối chiếu hash của CSV gốc, dựng lại tập test theo ngày và chạy CLI cho từng model. Mỗi giá trị dự báo phải khớp với cột `pred_hybrid` đã xuất trong dung sai tuyệt đối `0.000001 ‰`.
