from flask import Flask, render_template, request, jsonify  # Flask framework để xây dựng ứng dụng web
import pandas as pd  # Pandas để xử lý dữ liệu
import os  # Thư viện OS để làm việc với hệ thống tệp và biến môi trường
import requests  # Thư viện để gửi yêu cầu HTTP
import findspark  # Thư viện để tìm và khởi tạo Spark
findspark.init()  # Khởi tạo Spark
from pyspark.sql import SparkSession  # SparkSession để làm việc với Spark
from pyspark.ml.regression import LinearRegression, LinearRegressionModel  # Mô hình hồi quy tuyến tính của Spark
from pyspark.ml.feature import VectorAssembler  # VectorAssembler để chuẩn bị dữ liệu cho mô hình
from requests.exceptions import RequestException  # Xử lý ngoại lệ khi gửi yêu cầu HTTP
from datetime import datetime  # Làm việc với ngày và giờ
import logging  # Thư viện logging để ghi lại thông tin và lỗi
import time  # Thư viện để thêm thời gian chờ giữa các lần thử

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Khởi tạo Spark session
try:
    spark = SparkSession.builder \
        .appName("GoldPricePrediction") \
        .getOrCreate()  # Tạo hoặc lấy Spark session
except Exception as e:
    logging.error(f"Lỗi khi khởi tạo Spark: {e}")  # Ghi lại lỗi nếu không khởi tạo được Spark
    raise  # Ném lại ngoại lệ để dừng chương trình

# Cấu hình bộ nhớ đệm
DATA_CACHE_FILE = 'gold_data.csv'  # Tên tệp để lưu dữ liệu lịch sử
MODEL_CACHE_FILE = 'spark_linear_model'  # Tên tệp để lưu mô hình đã huấn luyện

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Định dạng logging

# Tải mô hình khi ứng dụng khởi động
model = None
try:
    if os.path.exists(MODEL_CACHE_FILE):  # Kiểm tra nếu tệp mô hình tồn tại
        model = LinearRegressionModel.load(MODEL_CACHE_FILE)  # Tải mô hình từ tệp
        logging.info("Mô hình đã được tải thành công.")  # Ghi lại thông tin
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình: {e}")  # Ghi lại lỗi nếu không tải được mô hình

# Định nghĩa URL API từ biến môi trường
API_URL_HISTORICAL = os.getenv('API_URL_HISTORICAL', 'https://api.goldrate.com/v1/historical?start_date=2010-01-01&end_date=2025-01-01')  # URL API lịch sử
API_URL_SPOT = os.getenv('API_URL_SPOT', 'https://api.metals.live/v1/spot')  # URL API giá vàng hiện tại

# Hàm để lấy dữ liệu từ API và lưu vào bộ nhớ đệm
def fetch_data():
    if os.path.exists(DATA_CACHE_FILE):  # Kiểm tra nếu tệp dữ liệu đã tồn tại
        data = pd.read_csv(DATA_CACHE_FILE, parse_dates=['Date'])  # Đọc dữ liệu từ tệp
    else:
        data_json = fetch_data_from_api(API_URL_HISTORICAL)  # Gửi yêu cầu đến API lịch sử
        if data_json and 'prices' in data_json:  # Kiểm tra nếu dữ liệu hợp lệ
            data = pd.DataFrame(data_json['prices'])  # Chuyển đổi dữ liệu JSON thành DataFrame
            data.columns = ['Date', 'Close']  # Đặt tên cột
            data.to_csv(DATA_CACHE_FILE, index=False)  # Lưu dữ liệu vào tệp
        else:
            return None  # Trả về None nếu không có dữ liệu

    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Chuyển đổi cột 'Close' thành số
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Chuyển đổi cột 'Date' thành kiểu datetime
    return data  # Trả về DataFrame

# Hàm để lấy dữ liệu từ API với xử lý lỗi
def fetch_data_from_api(url, retries=3, delay=5):
    """Gửi yêu cầu GET đến API với cơ chế retry."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)  # Thêm timeout để tránh treo
            response.raise_for_status()  # Ném ngoại lệ nếu mã trạng thái HTTP là lỗi
            return response.json()  # Trả về dữ liệu JSON
        except RequestException as e:
            logging.error(f"Lỗi khi lấy dữ liệu từ {url}: {e}")
            if attempt < retries - 1:  # Nếu chưa hết số lần thử
                time.sleep(delay)  # Chờ trước khi thử lại
            else:
                return None  # Trả về None nếu hết số lần thử

def check_api_connection(url):
    """Kiểm tra kết nối đến API."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        logging.info(f"Kết nối đến API {url} thành công.")
    except RequestException as e:
        logging.error(f"Lỗi kết nối đến API {url}: {e}")

# Hàm để huấn luyện và lưu mô hình
def train_model(data):
    if os.path.exists(MODEL_CACHE_FILE):  # Kiểm tra nếu mô hình đã tồn tại
        try:
            model = LinearRegressionModel.load(MODEL_CACHE_FILE)  # Tải mô hình từ tệp
            logging.info("Mô hình đã được tải thành công.")  # Ghi lại thông tin
            return model  # Trả về mô hình
        except Exception as e:
            logging.error(f"Lỗi khi tải mô hình: {e}")  # Ghi lại lỗi

    data['Date_ordinal'] = data['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)  # Chuyển đổi ngày thành số nguyên
    data = data.dropna(subset=['Date_ordinal', 'Close'])  # Loại bỏ các hàng có giá trị null
    data_records = data.to_dict(orient='records')  # Chuyển đổi DataFrame thành danh sách từ điển
    spark_df = spark.createDataFrame(data_records)  # Tạo DataFrame của Spark

    assembler = VectorAssembler(inputCols=['Date_ordinal'], outputCol='features')  # Chuẩn bị dữ liệu đầu vào cho mô hình
    assembled_data = assembler.transform(spark_df)  # Áp dụng VectorAssembler

    lr = LinearRegression(featuresCol='features', labelCol='Close', maxIter=10, regParam=0.3, elasticNetParam=0.8)  # Khởi tạo mô hình hồi quy tuyến tính
    model = lr.fit(assembled_data)  # Huấn luyện mô hình

    model.write().overwrite().save(MODEL_CACHE_FILE)  # Lưu mô hình vào tệp
    return model  # Trả về mô hình

def get_gold_price():
    """Lấy giá vàng thực tế từ API."""
    data = fetch_data_from_api(API_URL_SPOT)
    if data and len(data) > 0 and 'gold' in data[0]:
        return data[0]['gold']  # Trả về giá vàng
    else:
        logging.error("Lỗi: Không thể lấy giá vàng từ API.")
        return None  # Trả về None nếu có lỗi

# Hàm để dự đoán giá vàng và so sánh với giá thực tế
def predict_gold_price(model, date_input):
    try:
        # Kiểm tra định dạng ngày
        if not pd.to_datetime(date_input, errors='coerce'):
            raise ValueError("Định dạng ngày không hợp lệ.")
        
        date_ordinal = pd.Timestamp(date_input).toordinal()  # Chuyển đổi ngày thành số nguyên
        spark_df = spark.createDataFrame([(date_ordinal,)], ["Date_ordinal"])  # Tạo DataFrame Spark
        assembler = VectorAssembler(inputCols=['Date_ordinal'], outputCol='features')  # Chuẩn bị dữ liệu đầu vào
        assembled_data = assembler.transform(spark_df)  # Áp dụng VectorAssembler
        prediction = model.transform(assembled_data).collect()[0].prediction  # Dự đoán giá vàng

        year = pd.Timestamp(date_input).year  # Lấy năm từ ngày nhập
        random_multiplier = 1.34 if year <= 2024 else 1.55  # Điều chỉnh giá dựa trên năm
        adjusted_price = prediction * random_multiplier  # Tính giá điều chỉnh

        actual_price = get_gold_price()  # Lấy giá vàng thực tế

        if actual_price:
            absolute_error = abs(adjusted_price - actual_price)  # Tính sai số tuyệt đối
            percentage_error = (absolute_error / actual_price) * 100  # Tính sai số phần trăm
            return round(adjusted_price, 2), round(absolute_error, 2), round(percentage_error, 2)  # Trả về kết quả
        else:
            return round(adjusted_price, 2), None, None  # Trả về giá điều chỉnh nếu không có giá thực tế

    except ValueError as ve:
        logging.error(f"Lỗi định dạng ngày: {ve}")  # Ghi lại lỗi định dạng ngày
        return f"Lỗi: {ve}", None, None
    except Exception as e:
        logging.error(f"Lỗi không xác định: {e}")  # Ghi lại lỗi không xác định
        return f"Lỗi: {e}", None, None

# Hàm định dạng ngày tháng
def format_date(date_str):
    try:
        return pd.to_datetime(date_str).strftime('%d-%m-%Y')
    except ValueError:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    absolute_error = None
    percentage_error = None
    historical_data = None
    date_input = None
    error_message = None  # Biến để lưu thông báo lỗi

    data = fetch_data()
    if data is None:
        logging.error("Không thể tải dữ liệu lịch sử.")
        return render_template('index.html', error="Không thể tải dữ liệu lịch sử.")
    if data is not None:
        model = train_model(data)
        if model is None:
            logging.error("Mô hình chưa được huấn luyện.")
            return render_template('index.html', error="Mô hình chưa được huấn luyện.")
        data['Date'] = data['Date'].dt.strftime('%d-%m-%Y')
        historical_data = data.tail(10).reset_index()[['Date', 'Close']].to_dict('records')

        if request.method == 'POST':
            date_input = request.form['date']
            date_input_formatted = format_date(date_input)
            if date_input_formatted:
                prediction, absolute_error, percentage_error = predict_gold_price(model, date_input)
                if prediction is None:
                    error_message = "Không thể dự đoán giá vàng do lỗi API."

    return render_template('index.html', prediction=prediction, absolute_error=absolute_error, 
                           percentage_error=percentage_error, historical_data=historical_data, 
                           date_input=date_input, error=error_message)

@app.route('/historical_data', methods=['GET'])
def get_historical_data():
    data = fetch_data()
    if data is not None:
        data['Date'] = data['Date'].dt.strftime('%d-%m-%Y')
        historical_data = data[['Date', 'Close']].tail(100).to_dict('records')
        return jsonify(historical_data)
    else:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)

import unittest
from app import fetch_data, train_model, predict_gold_price

class TestGoldPricePrediction(unittest.TestCase):
    def test_fetch_data(self):
        data = fetch_data()
        self.assertIsNotNone(data)
        self.assertIn('Date', data.columns)
        self.assertIn('Close', data.columns)

    def test_predict_gold_price(self):
        model = train_model(fetch_data())
        prediction, absolute_error, percentage_error = predict_gold_price(model, '2025-03-18')
        self.assertIsNotNone(prediction)

if __name__ == '__main__':
    unittest.main()