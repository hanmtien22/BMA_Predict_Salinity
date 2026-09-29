# KẾ HOẠCH CHI TIẾT XÂY DỰNG MÃ NGUỒN (NOTEBOOK CELL-BY-CELL)
## MÔ HÌNH LAI HỌC MÁY & QUÁ TRÌNH GAUSSIAN KHÔNG GIAN - THỜI GIAN
### (HYBRID MACHINE LEARNING & SPATIO-TEMPORAL GAUSSIAN PROCESS)

---

## I. MỤC TIÊU & BỐI CẢNH DỰ ÁN

### 1. Khắc phục các tồn tại của nghiên cứu cũ
* **Triệt tiêu Data Leakage (Rò rỉ dữ liệu):** Loại bỏ hoàn toàn việc dùng thuật toán BMA chọn đặc trưng trên toàn bộ 100% tập dữ liệu. Không dùng phân chia ngẫu nhiên (Random Row Split) trên dữ liệu chuỗi thời gian.
* **Xóa bỏ hiện tượng "học vẹt ngày":** Thay thế 37 cột nhị phân 0/1 (`19Th0119`, `04Th0219`...) bằng các biến chu kỳ mùa vụ tuần hoàn (Day of Year, sin_doy, cos_doy).
* **Phân chia dữ liệu khách quan theo thời gian (Temporal Split):** Sử dụng toàn bộ dữ liệu năm 2019 làm tập huấn luyện (Train) và năm 2020 làm tập kiểm thử độc lập (Test).

### 2. Hiện thực hóa mô hình lai (Hybrid Framework)
* **Bước 1 (Học phi tuyến từ viễn thám):** Sử dụng các mô hình học máy dạng cây (Decision Tree, Bagging, Random Forest, PSO-RF, PSO-Bagging, XGBoost, LightGBM) để học mối quan hệ phi tuyến phức tạp giữa độ mặn và các kênh phổ viễn thám (NDMI, MSI, B2, NIR, PCA2, PCA5, DEM...).
* **Bước 2 (Học cấu trúc không - thời gian còn sót lại trong sai số):** Lấy phần dư sai số (residual = giá trị thực tế - giá trị mô hình cây dự báo) để đưa vào mô hình Quá trình Gaussian (Gaussian Process).
* **Cấu trúc Kernel Không - Thời gian tích (Product Spatio-Temporal Kernel):**
  * Khoảng cách không gian Euclid giữa 2 trạm quan trắc i và j:
    ```text
    d_ij = sqrt( (lat_i - lat_j)^2 + (long_i - long_j)^2 )
    ```
  * Kernel không gian (Spatial RBF Kernel):
    ```text
    k_s(i, j) = exp( - (d_ij)^2 / (2 * l_s^2) )
    ```
  * Kernel thời gian tuần hoàn 365 ngày (Periodic Temporal Kernel):
    ```text
    k_t(i, j) = exp[ - 2 * sin^2( pi * (t_i - t_j) / 365 ) / l_t^2 ]
    ```
  * Kernel tích kết hợp Không - Thời gian:
    ```text
    k(i, j) = k_s(i, j) * k_t(i, j)
    ```
* **Dự báo tổng hợp cuối cùng:**
  ```text
  y_hat_final = y_hat_base + e_hat_gp
  (Trong đó: y_hat_base từ mô hình cây, e_hat_gp là phần bù sai số từ Gaussian Process)
  ```

---

## II. KẾ HOẠCH TRIỂN KHAI TỪNG CELL TRONG NOTEBOOK

---

### CELL 1: CÀI ĐẶT CÁC THƯ VIỆN BỔ SUNG (NẾU CHẠY COLAB / KAGGLE)

* **Mục tiêu:** Cài đặt các thư viện phục vụ tối ưu bầy đàn, boosting và giải thích mô hình.
* **Mã nguồn:**
```python
# Cài đặt các thư viện cần thiết
!pip install pyswarm shap lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn -q
print("Cài đặt thư viện hoàn tất.")
```

---

### CELL 2: IMPORT THƯ VIỆN & ĐỊNH NGHĨA CÁC ĐỘ ĐO HIỆU NĂNG

* **Mục tiêu:** Nạp toàn bộ các thư viện và thiết lập 4 hàm đo lường chuẩn mực của bài báo (R2, RMSE, MAE, Ex_VAR).
* **Mã nguồn:**
```python
import os
import re
import copy
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel, WhiteKernel
from pyswarm import pso

warnings.filterwarnings('ignore')
np.random.seed(42)

# 1. Hệ số xác định R2
def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

# 2. Sai số bình phương trung bình căn (RMSE - g/l)
def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# 3. Sai số tuyệt đối trung bình (MAE - g/l)
def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# 4. Phương sai giải thích (Ex_VAR - %)
def calc_explained_variance(y_true, y_pred):
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return (1 - np.var(y_true - y_pred) / var_y) * 100

# 5. Tổng hợp bảng chỉ số
def evaluate_model(y_true, y_pred):
    return {
        'RMSE (g/l)': round(calc_rmse(y_true, y_pred), 3),
        'MAE (g/l)': round(calc_mae(y_true, y_pred), 3),
        'R2': round(calc_r2(y_true, y_pred), 3),
        'Ex_VAR (%)': round(calc_explained_variance(y_true, y_pred), 3)
    }

print("Đã nạp thư viện và sẵn sàng các hàm đánh giá.")
```

---

### CELL 3: ĐỌC DỮ LIỆU, XỬ LÝ 22 DÒNG KHUYẾT VÀ TẠO CỘT NGÀY CHUẨN

* **Mục tiêu:**
  1. Quét tìm 37 cột ngày định dạng `^\d{2}Th\d{4}$`.
  2. Bắt lỗi an toàn: lọc bỏ 22 dòng khuyết thông tin ngày (toàn số 0).
  3. Dịch 37 cột nhị phân sang một cột ngày chuẩn duy nhất `date`.
  4. Trích xuất các biến thời gian: `doy`, `year`, `month`, `sin_doy`, `cos_doy`.
  5. Xóa bỏ 37 cột nhị phân cũ để làm sạch bộ nhớ.
* **Mã nguồn:**
```python
# 1. Đọc file CSV
csv_file = 'Check_BMA_Chuanhoa_LS8_20192020_159.csv'
df_raw = pd.read_csv(csv_file)
print(f"Dữ liệu ban đầu: {df_raw.shape[0]} hàng, {df_raw.shape[1]} cột.")

# 2. Tìm danh sách 37 cột ngày bằng biểu thức chính quy Regex
time_cols = [col for col in df_raw.columns if re.match(r'^\d{2}Th\d{4}$', col)]
print(f"Tìm thấy {len(time_cols)} cột ngày mã hóa nhị phân.")

# 3. Hàm giải mã tên cột (ví dụ '19Th0119' -> datetime(2019, 1, 19))
def parse_date_column(col_name):
    day = int(col_name[0:2])
    month = int(col_name[4:6])
    year = int("20" + col_name[6:8])
    return datetime(year, month, day)

# 4. Hàm trích xuất ngày an toàn cho từng hàng (xử lý ngoại lệ toàn số 0 hoặc nhiều số 1)
def extract_safe_date(row):
    active_cols = [col for col in time_cols if row[col] == 1]
    if len(active_cols) == 1:
        return parse_date_column(active_cols[0])
    return pd.NaT

df_clean = df_raw.copy()
df_clean['date'] = df_clean.apply(extract_safe_date, axis=1)

# Loại bỏ các dòng bị khuyết ngày (22 dòng toàn số 0)
num_missing_dates = df_clean['date'].isna().sum()
df_clean = df_clean.dropna(subset=['date']).reset_index(drop=True)
print(f"Đã loại bỏ {num_missing_dates} dòng khuyết ngày. Dữ liệu sạch: {df_clean.shape[0]} hàng.")

# 5. Tạo các đặc trưng thời gian mới
df_clean['doy'] = df_clean['date'].dt.dayofyear
df_clean['year'] = df_clean['date'].dt.year
df_clean['month'] = df_clean['date'].dt.month

# 6. Biến đổi tuần hoàn sin/cos cho mô hình Cây
df_clean['sin_doy'] = np.sin(2 * np.pi * df_clean['doy'] / 365.25)
df_clean['cos_doy'] = np.cos(2 * np.pi * df_clean['doy'] / 365.25)

# 7. Xóa bỏ 37 cột nhị phân cũ
df_clean = df_clean.drop(columns=time_cols)

print("Kích thước bảng dữ liệu sau khi tối ưu hóa:", df_clean.shape)
display(df_clean[['date', 'year', 'month', 'doy', 'sin_doy', 'cos_doy', 'Lat', 'Long', 'SALINITY_OK']].head())
```

---

### CELL 4: PHÂN CHIA DỮ LIỆU THEO THỜI GIAN (TEMPORAL SPLIT CHỐNG LEAKAGE)

* **Mục tiêu:**
  * Huấn luyện mô hình trên quá khứ (Năm 2019).
  * Kiểm thử mô hình trên tương lai (Năm 2020) hoàn toàn độc lập.
  * Tách riêng bộ biến cho Mô hình Cây và bộ biến cho Gaussian Process.
* **Mã nguồn:**
```python
target_col = 'SALINITY_OK'

# Danh sách các biến viễn thám và địa hình
rs_features = [
    'EVI', 'NDBI', 'NDMI', 'NDSI', 'NDVI', 'NDWI', 'VSSI', 'SAVI', 'COSRI', 'EVI2',
    'MSI', 'ND23', 'ND47', 'SI1', 'SI2', 'SI3', 'SI4', 'SI5', 'SI6', 'SI7', 'SI8', 'SI9',
    'CA', 'B', 'G', 'R', 'NIR', 'PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'DEM'
]

# Bộ đặc trưng dành cho mô hình Cây (Viễn thám + Tọa độ + Chu kỳ tuần hoàn)
base_features = rs_features + ['Lat', 'Long', 'sin_doy', 'cos_doy']

# Bộ đặc trưng dành cho Gaussian Process (Không gian: Lat, Long và Thời gian: DOY)
st_features = ['Lat', 'Long', 'doy']

# 1. Tách tập Train (Năm 2019) và tập Test (Năm 2020)
train_mask = (df_clean['year'] == 2019)
test_mask  = (df_clean['year'] == 2020)

df_train = df_clean[train_mask].reset_index(drop=True)
df_test  = df_clean[test_mask].reset_index(drop=True)

# 2. Tạo ma trận đầu vào cho Mô hình Cây
X_train_base = df_train[base_features].values
X_test_base  = df_test[base_features].values

# 3. Tạo ma trận đầu vào cho Gaussian Process
X_train_st = df_train[st_features].values
X_test_st  = df_test[st_features].values

# 4. Tạo vector nhãn mục tiêu (Độ mặn thực tế)
y_train = df_train[target_col].values.astype(np.float32)
y_test  = df_test[target_col].values.astype(np.float32)

print(f"-> Tập Train (Năm 2019): {X_train_base.shape[0]} mẫu (Độ mặn TB: {y_train.mean():.2f} g/l)")
print(f"-> Tập Test  (Năm 2020): {X_test_base.shape[0]} mẫu (Độ mặn TB: {y_test.mean():.2f} g/l)")
print(f"-> Số đặc trưng mô hình Cây: {X_train_base.shape[1]}")
print(f"-> Số đặc trưng Gaussian Process: {X_train_st.shape[1]} (Lat, Long, doy)")
```

---

### CELL 5: KHỞI TẠO CÁC MÔ HÌNH CƠ SỞ & TỐI ƯU SIÊU THAM SỐ PSO

* **Mục tiêu:** Xây dựng danh sách 7 mô hình cây (Decision Tree, Bagging, PSO-Bagging, Random Forest, PSO-RF, XGBoost, LightGBM) được tối ưu hóa siêu tham số bằng 5-Fold Cross Validation trên tập Train.
* **Mã nguồn:**
```python
# Cờ điều khiển: Đặt True để dùng ngay cấu hình tối ưu của bài báo (chạy nhanh), 
# Đặt False nếu muốn chạy lại thuật toán PSO từ đầu
USE_PRESET_OPTIMAL_PARAMS = True

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Các mô hình mặc định
dt_base = DecisionTreeRegressor(random_state=42)
bagging_base = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42), n_estimators=10, random_state=42, n_jobs=-1)
rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_base = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1)
lgbm_base = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)

# 2. Cấu hình các mô hình tối ưu hóa PSO
if USE_PRESET_OPTIMAL_PARAMS:
    print("Sử dụng bộ siêu tham số tối ưu chuẩn xác:")
    pso_rf = RandomForestRegressor(n_estimators=250, max_depth=17, min_samples_split=2, min_samples_leaf=5, random_state=42, n_jobs=-1)
    pso_bag = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=19, min_samples_split=2, min_samples_leaf=5, random_state=42),
                               n_estimators=259, random_state=42, n_jobs=-1)
else:
    print("Bắt đầu chạy PSO tối ưu siêu tham số trên tập Train...")
    def pso_obj_rf(params):
        m = RandomForestRegressor(n_estimators=int(params[0]), max_depth=int(params[1]), min_samples_leaf=int(params[2]), random_state=42, n_jobs=-1)
        return -cross_val_score(m, X_train_base, y_train, cv=cv_strategy, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    
    best_p_rf, _ = pso(pso_obj_rf, [100, 10, 2], [300, 22, 8], swarmsize=15, maxiter=15)
    pso_rf = RandomForestRegressor(n_estimators=int(best_p_rf[0]), max_depth=int(best_p_rf[1]), min_samples_leaf=int(best_p_rf[2]), random_state=42, n_jobs=-1)
    pso_bag = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=int(best_p_rf[1]), min_samples_leaf=int(best_p_rf[2]), random_state=42),
                               n_estimators=int(best_p_rf[0]), random_state=42, n_jobs=-1)

# Danh sách toàn bộ các mô hình cơ sở
models_dict = {
    'Decision Tree': dt_base,
    'Bagging': bagging_base,
    'PSO-Bagging': pso_bag,
    'Random Forest': rf_base,
    'PSO-Random Forest': pso_rf,
    'XGBoost': xgb_base,
    'LightGBM': lgbm_base
}

print(f"Đã sẵn sàng {len(models_dict)} mô hình cơ sở.")
```

---

### CELL 6: THIẾT LẬP KERNEL KHÔNG - THỜI GIAN CHO GAUSSIAN PROCESS

* **Mục tiêu:** Xây dựng khung mô hình Gaussian Process với Product Kernel theo đúng công thức trang vở của thầy.
* **Mã nguồn:**
```python
# 1. Kernel không gian RBF (Khoảng cách Euclid tọa độ d_ij)
# Tọa độ Lat, Long có bán kính tương quan khởi tạo là 0.2 độ (khoảng 20km)
spatial_kernel = RBF(length_scale=[0.2, 0.2], length_scale_bounds=(0.01, 2.0))

# 2. Kernel thời gian tuần hoàn ExpSineSquared (Chu kỳ lặp 365 ngày)
# Độ dài tương quan mùa mặn khởi tạo là 30 ngày (1 tháng)
temporal_kernel = ExpSineSquared(length_scale=1.0, periodicity=365.0, periodicity_bounds="fixed")

# 3. Kernel Không - Thời gian kết hợp: Tích k_s * k_t + Nhiễu WhiteKernel
st_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01, 100.0)) * (spatial_kernel * temporal_kernel) + \
            WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3, 10.0))

# 4. Khởi tạo mô hình mẫu Gaussian Process
gp_model_template = GaussianProcessRegressor(
    kernel=st_kernel,
    n_restarts_optimizer=5,
    normalize_y=True,
    random_state=42
)

print("Đã thiết lập xong Spatio-Temporal Kernel cho Gaussian Process.")
print("Cấu trúc Kernel:", st_kernel)
```

---

### CELL 7: VÒNG LẶP HUẤN LUYỆN TOÀN BỘ MÔ HÌNH LAI (BASE + RESIDUAL GP)

* **Mục tiêu:**
  1. Huấn luyện mô hình cơ sở trên các biến viễn thám.
  2. Tính sai số dư (residual) trên tập Train.
  3. Huấn luyện Gaussian Process học cấu trúc sai số đó.
  4. Dự báo tổng hợp trên tập Test và ghi nhận toàn bộ kết quả.
* **Mã nguồn:**
```python
results_list = []
predictions_store = {}

print("=" * 80)
print("BẮT ĐẦU HUẤN LUYỆN TOÀN BỘ CÁC MÔ HÌNH ĐƠN LẺ VÀ MÔ HÌNH LAI (+ GP)")
print("=" * 80)

for name, model in models_dict.items():
    print(f"\n--> Đang huấn luyện: {name} ...")
    
    # ------------------------------------------------------------------
    # BƯỚC A: HUẤN LUYỆN MÔ HÌNH CƠ SỞ (BASE MODEL)
    # ------------------------------------------------------------------
    model.fit(X_train_base, y_train)
    pred_train_base = np.clip(model.predict(X_train_base), 0, None)
    pred_test_base  = np.clip(model.predict(X_test_base), 0, None)
    
    eval_base_tr = evaluate_model(y_train, pred_train_base)
    eval_base_te = evaluate_model(y_test, pred_test_base)
    
    # Lưu kết quả mô hình Đơn lẻ
    results_list.append({
        'Model': f"{name} (Đơn lẻ)",
        'Type': 'Base',
        'Train_RMSE': eval_base_tr['RMSE (g/l)'],
        'Train_R2': eval_base_tr['R2'],
        'Test_RMSE': eval_base_te['RMSE (g/l)'],
        'Test_MAE': eval_base_te['MAE (g/l)'],
        'Test_R2': eval_base_te['R2'],
        'Test_Ex_VAR': eval_base_te['Ex_VAR (%)']
    })
    
    # ------------------------------------------------------------------
    # BƯỚC B: TÍNH PHẦN DƯ SAI SỐ TRÊN TẬP TRAIN (RESIDUAL)
    # ------------------------------------------------------------------
    residual_train = y_train - pred_train_base
    
    # ------------------------------------------------------------------
    # BƯỚC C: HUẤN LUYỆN GAUSSIAN PROCESS TRÊN PHẦN DƯ SAI SỐ
    # ------------------------------------------------------------------
    gp = clone(gp_model_template)
    gp.fit(X_train_st, residual_train)
    
    # GP dự báo phần bù sai số trên tập Test
    residual_test_pred = gp.predict(X_test_st)
    
    # ------------------------------------------------------------------
    # BƯỚC D: DỰ BÁO TỔNG HỢP MÔ HÌNH LAI (HYBRID PREDICTION)
    # ------------------------------------------------------------------
    pred_test_hybrid = np.clip(pred_test_base + residual_test_pred, 0, None)
    eval_hybrid_te = evaluate_model(y_test, pred_test_hybrid)
    
    predictions_store[name] = {
        'base': pred_test_base,
        'hybrid': pred_test_hybrid
    }
    
    # Lưu kết quả mô hình Lai
    results_list.append({
        'Model': f"{name} + GP (Mô hình lai)",
        'Type': 'Hybrid',
        'Train_RMSE': eval_base_tr['RMSE (g/l)'],
        'Train_R2': eval_base_tr['R2'],
        'Test_RMSE': eval_hybrid_te['RMSE (g/l)'],
        'Test_MAE': eval_hybrid_te['MAE (g/l)'],
        'Test_R2': eval_hybrid_te['R2'],
        'Test_Ex_VAR': eval_hybrid_te['Ex_VAR (%)']
    })
    
    print(f"   [Test] Đơn lẻ: R2 = {eval_base_te['R2']:.3f}, RMSE = {eval_base_te['RMSE (g/l)']:.3f}")
    print(f"   [Test] Lai +GP: R2 = {eval_hybrid_te['R2']:.3f}, RMSE = {eval_hybrid_te['RMSE (g/l)']:.3f}")

print("\nHoàn tất huấn luyện toàn bộ hệ thống!")
```

---

### CELL 8: BẢNG TỔNG HỢP HIỆU NĂNG & ĐỐI CHUẨN KẾT QUẢ

* **Mục tiêu:** Xuất bảng Pandas DataFrame so sánh trực tiếp hiệu năng giữa từng cặp mô hình (trước và sau khi có Gaussian Process).
* **Mã nguồn:**
```python
df_results = pd.DataFrame(results_list)
display(df_results)

# Lưu bảng kết quả ra file CSV
df_results.to_csv('so_sanh_hieu_nang_hybrid_gp.csv', index=False)
print("Đã lưu bảng kết quả vào file: so_sanh_hieu_nang_hybrid_gp.csv")
```

---

### CELL 9: TRỰC QUAN HÓA KẾT QUẢ DỰ BÁO (SCATTER PLOT 1:1)

* **Mục tiêu:** Vẽ đồ thị phân tán Scatter Plot giữa giá trị thực tế và giá trị dự báo để chứng minh sự cải thiện rõ rệt của mô hình lai so với mô hình đơn lẻ.
* **Mã nguồn:**
```python
best_model_name = 'LightGBM'  # Hoặc 'PSO-Random Forest'

pred_base = predictions_store[best_model_name]['base']
pred_hybrid = predictions_store[best_model_name]['hybrid']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
max_val = max(y_test.max(), pred_hybrid.max()) + 2

# Đồ thị 1: Mô hình đơn lẻ
axes[0].scatter(y_test, pred_base, color='#1f77b4', alpha=0.6, edgecolors='k')
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')
eval_b = evaluate_model(y_test, pred_base)
axes[0].set_title(f"(a) {best_model_name} (Đơn lẻ)\nRMSE = {eval_b['RMSE (g/l)']} g/l, R2 = {eval_b['R2']}", fontsize=13)
axes[0].set_xlabel('Độ mặn thực tế (g/l)', fontsize=12)
axes[0].set_ylabel('Độ mặn dự báo (g/l)', fontsize=12)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Đồ thị 2: Mô hình lai + Gaussian Process
axes[1].scatter(y_test, pred_hybrid, color='#2ca02c', alpha=0.6, edgecolors='k')
axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')
eval_h = evaluate_model(y_test, pred_hybrid)
axes[1].set_title(f"(b) {best_model_name} + GP (Mô hình lai)\nRMSE = {eval_h['RMSE (g/l)']} g/l, R2 = {eval_h['R2']}", fontsize=13)
axes[1].set_xlabel('Độ mặn thực tế (g/l)', fontsize=12)
axes[1].set_ylabel('Độ mặn dự báo (g/l)', fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig('so_sanh_du_bao_scatter_plot.png', dpi=300)
plt.show()
print("Đã lưu biểu đồ: so_sanh_du_bao_scatter_plot.png")
```

---

## III. NHỮNG LƯU Ý KỸ THUẬT QUAN TRỌNG KHI CODE

1. **Tuyệt đối không scale lại cột tọa độ (`Lat`, `Long`):**
   Tọa độ trong file CSV đang ở dạng thập phân thực tế (9.x và 106.x). Hãy giữ nguyên giá trị này để công thức khoảng cách Euclid `d_ij` tính đúng bán kính địa lý.
2. **Xử lý số âm:** 
   Độ mặn trong tự nhiên không thể mang giá trị âm. Luôn bọc hàm `np.clip(y_pred, 0, None)` sau khi cộng phần bù sai số từ Gaussian Process.
3. **Ý nghĩa của 22 dòng khuyết ngày:** 
   Việc loại bỏ 22 dòng này ở Cell 3 là bắt buộc để đảm bảo dữ liệu thời gian sạch 100%, không bị gán nhầm ngày.
