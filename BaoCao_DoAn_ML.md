# BÁO CÁO ĐỒ ÁN HỌC MÁY: DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ (CUSTOMER CHURN PREDICTION)

---

## CHỈ TIÊU 1: TRÌNH BÀY BÀI TOÁN

### 1. Giới thiệu bài toán
Trong bối cảnh cạnh tranh khốc liệt của ngành viễn thông (Telco), việc thu hút một khách hàng mới thường tốn kém hơn rất nhiều (từ 5 đến 25 lần) so với việc giữ chân một khách hàng hiện hữu. Bài toán "Dự đoán khách hàng rời bỏ" (Customer Churn Prediction) ra đời nhằm mục đích sử dụng các kỹ thuật Học Máy (Machine Learning) để phân tích hành vi và thông tin của khách hàng trong quá khứ, từ đó dự báo liệu một khách hàng có khả năng ngưng sử dụng dịch vụ trong tương lai gần hay không.

### 2. Xác định Inputs và Outputs
Dựa trên tập dữ liệu chuẩn **Telco Customer Churn** từ Kaggle:

**Inputs (Đầu vào)**
Bao gồm 20 đặc trưng (features) mô tả khách hàng, chia làm 3 nhóm chính:
- **Thông tin nhân khẩu học:** Giới tính (Gender), Độ tuổi (SeniorCitizen có phải người cao tuổi không), Tình trạng hôn nhân/đối tác (Partner), Người phụ thuộc (Dependents).
- **Thông tin dịch vụ & tài khoản:** Số tháng đã sử dụng (Tenure), Dịch vụ điện thoại (PhoneService), Đường dây ưu tiên (MultipleLines), Loại dịch vụ Internet (DSL, Fiber optic...), Bảo mật trực tuyến (OnlineSecurity), Lưu trữ trực tuyến (OnlineBackup), Bảo vệ thiết bị (DeviceProtection), Hỗ trợ kỹ thuật (TechSupport), Dịch vụ truyền hình (StreamingTV, StreamingMovies).
- **Thông tin thanh toán:** Hợp đồng (Contract: Month-to-month, One year, Two year), Thanh toán không giấy tờ (PaperlessBilling), Phương thức thanh toán (PaymentMethod), Chi phí hàng tháng (MonthlyCharges), Tổng chi phí (TotalCharges).

**Outputs (Đầu ra)**
- Biến mục tiêu (Target variable): `Churn` (Có/Không rời bỏ). 
- Trạng thái: Đây là một bài toán **Phân loại nhị phân (Binary Classification)**, trong đó mô hình sẽ trả về 1 (Yes - Khách hàng sẽ rời bỏ) hoặc 0 (No - Khách hàng sẽ tiếp tục sử dụng dịch vụ).

### 3. Ý nghĩa của bài toán
**Đối với tổ chức (Doanh nghiệp Viễn thông)**
- **Tối ưu hóa chi phí Marketing:** Thay vì tung ra các chiến dịch khuyến mãi dàn trải, doanh nghiệp có thể tập trung ngân sách để chăm sóc chính xác nhóm khách hàng có nguy cơ rời bỏ cao.
- **Cải thiện sản phẩm/dịch vụ:** Thông qua việc phân tích các "đặc trưng quan trọng" (Feature Importance), doanh nghiệp biết được lý do gốc rễ khiến khách hàng rời bỏ (ví dụ: cước phí quá cao, hỗ trợ kỹ thuật kém, v.v.) để đưa ra cải tiến hệ thống kịp thời.

**Đối với cá nhân (Khách hàng)**
- **Trải nghiệm cá nhân hóa tốt hơn:** Những khách hàng có bất mãn tiềm ẩn chưa nói ra có thể sẽ được nhận các khoản giảm giá, gói cước phù hợp hơn hoặc dịch vụ chăm sóc tận tình hơn từ nhà mạng, giúp gia tăng mức độ hài lòng.

**Đối với xã hội (Mức độ địa phương và toàn cầu)**
- **Hỗ trợ phát triển bền vững:** Việc dự đoán và giữ chân người dùng giúp các tập đoàn duy trì được chuỗi cung ứng dịch vụ viễn thông ổn định, tránh lãng phí chi phí vận hành quảng cáo thừa, đóng góp vào sự ổn định thị trường kinh tế địa phương.
- Cung cấp một bộ khung (framework) chuẩn có thể dễ dàng mở rộng sang các lĩnh vực kinh doanh khác như Ngân hàng (dự đoán nợ xấu/thẻ tín dụng), Bảo hiểm, Bán lẻ thương mại điện tử trên toàn cầu.

---

## CHỈ TIÊU 3: CÁC PHƯƠNG PHÁP HỌC MÁY SỬ DỤNG

Trong dự án này, để giải quyết bài toán dự đoán khách hàng rời bỏ (Phân loại nhị phân), chúng tôi áp dụng và so sánh 3 phương pháp học máy (Machine Learning) tiêu biểu, từ cơ bản đến phức tạp.

### 1. Hồi quy Logistic (Logistic Regression)
**Tổng quan:**
Logistic Regression là một mô hình phân loại tuyến tính cơ bản nhưng cực kỳ hiệu quả. Mặc dù có tên là "regression" (hồi quy), phương pháp này thường được dùng để giải quyết bài toán phân loại.

**Cơ chế hoạt động:**
- Thuật toán cố gắng tìm ra một siêu phẳng (hyperplane) phân chia hai lớp (Có Churn và Không Churn) dựa trên phương trình tổ hợp tuyến tính của các features đầu vào.
- Đầu ra của hàm tuyến tính được đưa qua một hàm kích hoạt Sigmoid (Logistic function) để bóp các giá trị về khoảng (0, 1). Giá trị này đại diện cho *xác suất* khách hàng thuộc về nhóm "Churn" (Thường lấy ngưỡng 0.5 để quyết định).

**Ưu điểm:**
- Tính minh bạch cao (Interpretability): Rất dễ giải thích mức độ ảnh hưởng của từng đặc trưng dựa vào trọng số (coefficients).
- Đóng vai trò là mô hình cơ sở (Baseline model) vững chắc để làm thước đo so sánh.

### 2. Rừng ngẫu nhiên (Random Forest)
**Tổng quan:**
Random Forest là một thuật toán Học tăng cường nhóm (Ensemble Learning) dựa trên kỹ thuật Bagging (Bootstrap Aggregating).

**Cơ chế hoạt động:**
- Trái ngược với việc chỉ dùng 1 bộ quy tắc, Random Forest xây dựng hàng trăm đến hàng ngàn cây quyết định (Decision Trees) khác nhau.
- Mỗi cây được huấn luyện độc lập trên một tập dữ liệu ngẫu nhiên (lấy mẫu có hoàn lại từ tập gốc) và với một nhóm nhỏ đặc trưng (features) ngẫu nhiên.
- Khi dự đoán, mô hình lấy kết quả bằng phương pháp "Bầu chọn đa số" (Majority voting) từ tất cả các cây.

**Ưu điểm:**
- Giải quyết rất tốt bài toán phi tuyến (Non-linear boundaries) mà Logistic Regression gặp khó khăn.
- Ít bị mất cân bằng bởi các ngoại lệ (Outliers) và giảm hiện tượng quá mức (Overfitting) nhờ việc trung bình hóa nhiều cây con.
- Cung cấp biểu đồ Độ quan trọng của thuộc tính (Feature Importances) chân thực.

### 3. XGBoost (Extreme Gradient Boosting)
**Tổng quan:**
XGBoost là một trong những thuật toán phát triển từ Gradient Boosting, được coi là phương pháp thuộc hàng hiện đại và mạnh mẽ nhất cho dữ liệu dạng bảng (tabular data).

**Cơ chế hoạt động:**
- Tương tự Random Forest, XGBoost cũng kết hợp nhiều cây quyết định. Tuy nhiên, nó áp dụng kỹ thuật Boosting thay vì Bagging.
- Các cây được xây dựng một cách **tuần tự (sequential)**. Cây quyết định thế hệ sau sẽ tập trung học từ "phần sai sót" (residual errors) của toàn bộ các cây thế hệ trước kết hợp lại.
- Tích hợp thêm các kỹ thuật điều chuẩn (L1, L2 Regularization) để hạn chế tối đa việc học thuật toán thuộc lòng (overfitting) tập train.

**Ưu điểm:**
- Hiệu suất (Performance) cực kỳ ấn tượng, thường dẫn đầu và chiến thắng trong hầu hết các cuộc thi học máy trên Kaggle liên quan đến tabular data.
- Tối ưu hóa hệ thống máy tính cực tốt, chạy nhanh nhờ kiến trúc lập trình tối ưu.
- Rất mạnh mẽ khi xử lý tập dữ liệu có sự mất cân bằng về class (Imbalanced data) nhờ tính năng tối ưu hướng đối tượng linh hoạt (scale_pos_weight).

---

## CHỈ TIÊU 5: THẢO LUẬN CÁC KẾT QUẢ ĐẠT ĐƯỢC

Trong phần thực nghiệm (thể hiện tại tệp `.ipynb`), chúng ta đã tiền xử lý dữ liệu và trải qua quy trình đánh giá 3 mô hình học máy: Logistic Regression, Random Forest, và XGBoost. Dưới đây là những thảo luận và kết luận quan trọng được rút ra.

### 1. Đánh giá hiệu suất của các mô hình (Model Performance)
Dựa trên kết quả chạy chéo (Cross-validation) và tập dữ liệu kiểm thử (Test set):

- **Logistic Regression (Mô hình tuyến tính):** 
  Hoàn thành xuất sắc nhiệm vụ của một mô hình cơ sở (baseline). Accuracy thu được thường nằm ở mức ~80%. Do tập dữ liệu Telco có một số đặc trưng tương quan tuyến tính tương đối tốt (chẳng hạn như MonthlyCharges và TotalCharges), mô hình này hoạt động hiệu quả bất ngờ so với sự đơn giản của nó.

- **Random Forest:** 
  Có khả năng học được các biểu diễn phức tạp. Tuy nhiên, do đặc thù phân nhánh và dễ bị nhiễu trên dữ liệu Categorical đã được One-Hot-Encoding, mô hình đôi khi bị overfitting nếu depth quá lớn, hoặc accuracy chỉ tương đương Logistic Regression (khoảng 79-80%) mặc dù quá trình train mất nhiều tài nguyên hơn. 

- **XGBoost (Best Model):** 
  Thể hiện khả năng mạnh mẽ nhất với bài toán phân loại này. Nhờ khả năng huấn luyện các trọng số lỗi một cách tinh tế và thuật toán Regularization, XGBoost cho số điểm **ROC-AUC** ổn định và nhỉnh hơn mặt bằng chung. Thuật toán này đã được lựa chọn để xuất ra file model phục vụ cho việc tích hợp vào Ứng dụng Web.

### 2. Thảo luận về các Chỉ số đo lường (Metrics)
Trong bài toán Churn Prediction, **Accuracy (Độ chính xác tổng thể)** không phải là chỉ số duy nhất để nhìn nhận, bởi vì bộ dữ liệu bị mất cân bằng (Ngưởi ở lại chiếm khoảng ~73%, Rời bỏ chiếm ~27%). Do đó, chúng ta cần tập trung vào:

- **Recall (Độ phủ):** Cực kỳ quan trọng. Chúng ta thà dự đoán nhầm (False Positive) một khách hàng sẽ rời bỏ để tặng họ voucher, còn hơn bỏ sót (False Negative) một khách hàng thực sự khó chịu rồi để họ sang đối thủ. Bằng cách thiết lập `class_weight='balanced'` hoặc `scale_pos_weight` trong mã nguồn, Recall của lớp Churn đã được cải thiện rõ rệt so với chạy mặc định.
- **ROC-AUC Score:** Là thang đo chuẩn nhất để biết mô hình có khả năng phân biệt tốt giữa 2 lớp (Khách hàng trung thành và Khách hàng muốn rời) hay không, bất chấp việc tỷ lệ phân lớp 73/27.

### 3. Các đặc trưng quan trọng phân tích từ mô hình (Feature Importances)
Từ biểu đồ Feature Importances trích xuất bởi mô hình XGBoost/Random Forest, những yếu tố then chốt nhất ảnh hưởng đến quyết định rời đi của khách hàng là:

1. **Hợp đồng (Contract):** Đặc trưng có trọng số lớn nhất. Khách hàng sử dụng gói cước **Month-to-month (Từng tháng)** có xác suất rời bỏ cao gấp nhiều lần so với gói cước hợp đồng cố định 1-2 năm. 
2. **Thời gian gắn bó (Tenure):** Số tháng tiếp tục đồng hành có tỷ lệ nghịch rất mạnh với Churn. Khách hàng mới (dưới 6 tháng) là đối tượng dễ rớt đài nhất.
3. **Loại Dịch vụ Internet (Internet Service - Fiber Optic):** Khách hàng dùng Cáp quang (Fiber optic) có tỷ lệ rời bỏ cao bất thường. Phân tích này là một insight kinh doanh để tổ chức nên kiểm tra lại đường truyền, tốc độ, hoặc giá trị/chi phí của mạng Cáp quang này.
4. **Chi phí cấu thành (MonthlyCharges / TotalCharges):** Cước phí gia tăng tạo ra sự không hài lòng nếu chất lượng không tương xứng.

### 4. Ứng dụng thực tiễn
Từ các kết quả đạt được, doanh nghiệp có thể:
- Xác định những khách hàng có xác suất (Churn Probability) > 50% bằng ứng dụng Web giao diện người dùng để nhân viên hỗ trợ trực tiếp.
- Ra mắt các gói kích cầu cho khách tháng chuyển sang hợp đồng 1 năm thay vì tháng-một nhằm giữ chân khách lâu hơn (Tăng Tenure).
