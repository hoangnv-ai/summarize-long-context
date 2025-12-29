1.Cài môi trường
conda create -n llm_sum_long_context python==3.11.14

2.Cài thư viện
pip install -r requirements.txt

3.Cung cấp dữ liệu cần tóm tắt trong thư mục ./raw_text. File dạng .txt

4.Chạy code 
python main.py

5.Xem kết quả đầu ra ở file ./output/result.json