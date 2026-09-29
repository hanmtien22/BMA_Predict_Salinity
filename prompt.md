Bạn là một AI Coding Assistant chuyên nghiệp về Data Science và Deep Learning. 

Hãy viết cho tôi toàn bộ mã nguồn của một file Google Colab Notebook (.ipynb) để huấn luyện và đánh giá các mô hình học máy dự báo độ mặn nước bề mặt theo đúng phương pháp luận của bài báo nghiên cứu.

---
### 1. YÊU CẦU CẤU TRÚC FILE NOTEBOOK (COLAB CELL-BY-CELL)
Mã nguồn phải được chia thành các Cell rõ ràng bằng Markdown và Python Code block, có thể copy và chạy tuần tự từ trên xuống dưới trên Google Colab mà không phát sinh lỗi:

- Cell 1 (Markdown + Code): Cài đặt thư viện (pyswarm, shap, torch, scikit-learn, pandas, numpy, matplotlib, seaborn).
- Cell 2 (Code): Import thư viện và định nghĩa hàm tính 4 độ đo đánh giá theo đúng công thức bài báo:
  1. R² (Coefficient of Determination)
  2. RMSE (Root Mean Square Error - g/l)
  3. MAE (Mean Absolute Error - g/l)
  4. Ex_VAR (Explained Variance - %)
- Cell 3 (Code): Tải dữ liệu thực tế (đọc từ file CSV) HOẶC tự động sinh tập dữ liệu giả lập (Mock Dataset) gồm 535 mẫu và 52 biến đầu vào chuẩn cấu trúc nếu không tìm thấy file CSV.
- Cell 4 (Code): Tiền xử lý dữ liệu & Lựa chọn 24 biến đầu vào tối ưu (Feature Selection):
  * 2 biến tọa độ: Latitude, Longitude
  * 6 chỉ số viễn thám: NDMI, MSI, B2, B5, PCA2, PCA5
  * 16 biến thời gian mã hóa nhị phân
  * Chia tập Train/Test theo tỷ lệ 70/30 và thực hiện Chuẩn hóa dữ liệu (StandardScaler).
- Cell 5 (Code): Lập trình 3 mô hình cơ sở (Base Models):
  1. Bagging Regressor
  2. Random Forest Regressor (RF)
  3. 1D-Convolutional Neural Network (1D-CNN dùng PyTorch)
- Cell 6 (Code): Lập trình thuật toán Tối ưu hóa bầy đàn (Particle Swarm Optimization - PSO) bằng `pyswarm` để tìm siêu tham số tối ưu cho Bagging và RF:
  * Siêu tham số tối ưu: n_estimators (10-300), max_depth (1-20), min_samples_split (2-10), min_samples_leaf (1-10).
  * Tạo ra 2 mô hình lai: PSO-Bagging và PSO-Random Forest (PSO-RF).
- Cell 7 (Code): Huấn luyện toàn bộ 5 mô hình và Tổng hợp bảng so sánh hiệu năng (R², RMSE, MAE, Ex_VAR) trên cả 2 tập Training set và Testing set dưới dạng Pandas DataFrame.
- Cell 8 (Code): Vẽ biểu đồ dự báo: Biểu đồ Scatter Plot giữa giá trị thực tế vs dự báo của mô hình tốt nhất (PSO-RF) và Biểu đồ SHAP summary plot đánh giá độ quan trọng của các biến.

---
### 2. YÊU CẦU KỸ THUẬT VỀ CODE
1. Code viết hoàn chỉnh, không dùng placeholder (không dùng `pass` hoặc `# TODO`), chạy trực tiếp ra kết quả.
2. Code tối ưu, sạch sẽ, có comment giải thích rõ ràng từng khối lệnh.
3. Đảm bảo mô hình 1D-CNN tương thích với dữ liệu đầu vào dạng bảng (Reshape dữ liệu về dạng [Samples, Channels, Features]).

Hãy sinh ra toàn bộ file Notebook theo cấu trúc trên.