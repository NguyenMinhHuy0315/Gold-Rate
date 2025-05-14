# Gold-Rate
• Mô tả:

  - Xây dựng mô hình hồi quy tuyến tính phân tán trên Apache Spark để dự đoán giá vàng lịch sử và điều chỉnh dự báo so với giá spot thực tế.

  - Triển khai ứng dụng web với Flask cho phép người dùng nhập ngày, nhận kết quả dự đoán, sai số tuyệt đối và phần trăm sai số, đồng thời hiển thị dữ liệu lịch sử.

• Công nghệ & công cụ đã sử dụng:
  - Python, Flask, Jinja2
  - ndspark, PySpark (SparkSession, MLlib: LinearRegression, VectorAssembler)
  - Pandas, Requests, logging, datetime, time, os
  - Unit tests với unittest

• Description:

- Built a distributed linear regression model using Apache Spark to predict historical gold prices and adjust forecasts based on actual spot prices.

- Deployed a web application with Flask that allows users to input a date and receive the predicted price, absolute error, and percentage error, while also displaying historical data.

• Technologies & Tools Used:
  - Python, Flask, Jinja2
  - ndspark, PySpark (SparkSession, MLlib: LinearRegression, VectorAssembler)
  - Pandas, Requests, logging, datetime, time, os
  - Unit testing with unittest
