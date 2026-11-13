import pandas as pd
import numpy as np
from my_holt_winter  import algo_holt_winter_define
from my_arima_family import algo_auto_arima_define
from my_prophet      import algo_prophet_define
from my_tool         import check_noise

def main():
    np.random.seed(36)
    time = np.arange(36)
    dates = pd.date_range(start='2021-01-01', periods=36, freq='MS') # 36 tháng

    data = []
    for cat in ['C1', 'C2']:
        # Phát sinh ngẫu nhiên 36 giá trị doanh thu cho mỗi Category
        revenues = 10 + 0.5 * time + 5 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 1, 36)
        
        # Ghép từng ngày (d) và doanh thu (rev) thành dict rồi thêm vào list data
        for d, rev in zip(dates, revenues):
            data.append({
                'Category': cat,
                'Period': d.strftime('%Y-%m-%d'),
                'Revenue': round(float(rev), 2)
            })

    ns_option = {'p': '2','d':'1', 'q': '2'}
    s_option = {'p': '2','d':'1', 'q': '2'}

    data = []
    msg, predict_data = my_algo_holt_winter_define(data= data, time_forecast= 12, is_read= True)
    # msg, predict_data = my_algo_arima_define(data= data, time_forecast= 12, ns_option= ns_option)
    # msg, predict_data = my_algo_sarima_define(data= data, time_forecast= 12, ns_option= ns_option, s_option= s_option, s_length= 12)
    # msg, predict_data = my_algo_prophet_define(data= data, time_forecast= 12 )
    print(predict_data)
"""
    Thống nhất chung cho data input đầu vào:
    1. data: cấu trúc data 'Category', 'Period', 'Revenue'
    2. operator: ADD, MUL nếu rỗng -> ADD
    3. smooth: cấu trúc data 'Category', 'Smooth_type'
        --- Hãy cứ xêm nó là 1 ẩn số không nên thiết lập sớm nó đang rất khủng khiếp về toán về sau    

    4. debug: True/False - y/c ghi file để check data
    5. holiday, x_past, x_future: cấu trúc data 'Category', 'Event_name', 'Period', 'Khuyen_mai'(dạng số 1/0)
    6. ns_option: có dạng tuple (p,d,q) - Trọng số cho dòng họ ARIMA không mùa vụ
    7. s_option:  có dạng tuple (p,d,q) - Trong số cho dòng họ ARIMA có mùa vụ
    8. s_length: Độ dài của 1 mùa vụ(nếu theo tháng: 12, quý: 4, tuần: 52)
        Chú ý: chỉ khi bật s_length thì s_option mới đc kích hoạt
    9. all_cpu: True/False - kích hoạt cho phép giải giải thuật lấy toàn bộ tài nguyên CPU để chạy
"""

def my_algo_holt_winter_define(data, time_forecast: int, s_length: int = 12, operator='ADD', is_check_noise: bool = False, 
                               is_debug= False, is_read= False):
    err_msg: pd.DataFrame = None

    # Thực hiện kiểm tra dữ liệu nhiễu
    if is_check_noise:
        err_msg = check_noise(data= data)
        if isinstance(err_msg, pd.DataFrame) and not err_msg.empty:
            err_msg = err_msg[err_msg['Type'].isin(['Error', 'Warning'])]

    predict_data = algo_holt_winter_define( data= data, time_forecast= time_forecast, s_length= 12, operator= operator,
                                            is_debug= is_debug, is_read= is_read)

    # Kết quả trả về
    return err_msg, predict_data

def my_algo_prophet_define(data, time_forecast: int, holiday=None, s_holiday= None, operator='ADD', is_smooth= None, is_check_noise: bool = False,
                           is_debug= False, is_read= False):
    err_msg: pd.DataFrame = None

    # Thực hiện kiểm tra dữ liệu nhiễu
    if is_check_noise:
        err_msg = check_noise(data= data)
        if isinstance(err_msg, pd.DataFrame) and not err_msg.empty:
            err_msg = err_msg[err_msg['Type'].isin(['Error', 'Warning'])]

    predict_data = algo_prophet_define( data= data, time_forecast= time_forecast, 
                                        holiday= holiday, s_holiday= s_holiday, is_smooth= is_smooth, operator=operator, 
                                        is_debug= is_debug)

    # Kết quả trả về
    return err_msg, predict_data

def my_algo_arima_define(data, time_forecast: int, ns_option, is_check_noise: bool = False, all_cpu= False, 
                         is_debug=False, is_read= False):
    err_msg: pd.DataFrame = None

    # Thực hiện kiểm tra dữ liệu nhiễu
    if is_check_noise:
        err_msg = check_noise(data= data)
        if isinstance(err_msg, pd.DataFrame) and not err_msg.empty:
            err_msg = err_msg[err_msg['Type'].isin(['Error', 'Warning'])]

    sts, msg, predict_data, warning_datas = algo_auto_arima_define( data= data,
                                                                    time_forecast= time_forecast,
                                                                    ns_option= ns_option,
                                                                    all_cpu= all_cpu,
                                                                    is_debug= is_debug,
                                                                    is_read= is_read)

    # Kiểm soát dữ liệu sau khi chạy AutoArima
    if sts == 'E':
        new_df = pd.DataFrame([{'Type': 'Error', 'Message': msg}])
        err_msg = pd.concat([err_msg, new_df], ignore_index=True)

    if isinstance(warning_datas, pd.DataFrame) and not warning_datas.empty:
        warn_rows = warning_datas[warning_datas['Warning'] == True]
        if not warn_rows.empty:
            warn_df = pd.DataFrame({
                'Type': 'Warning',
                'Message': "aicc của " + warn_rows['Category'].astype(str) + " < 0.05"
            })
            err_msg = pd.concat([err_msg, warn_df], ignore_index= True)

    # Kết quả trả về
    return err_msg, predict_data

def my_algo_sarima_define(data, time_forecast: int, ns_option, s_option, s_length: int, is_check_noise: bool = False, 
                          all_cpu= False, is_debug=False, is_read= False ):
    err_msg: pd.DataFrame = None

    # Thực hiện kiểm tra dữ liệu nhiễu
    if is_check_noise:
        err_msg = check_noise(data= data)
        if isinstance(err_msg, pd.DataFrame) and not err_msg.empty:
            err_msg = err_msg[err_msg['Type'].isin(['Error', 'Warning'])]

    sts, msg, predict_data, warning_datas = algo_auto_arima_define( data= data,
                                                                    time_forecast= time_forecast,
                                                                    ns_option= ns_option, 
                                                                    s_option= s_option,
                                                                    s_length= s_length,
                                                                    all_cpu= all_cpu,
                                                                    is_debug= is_debug,
                                                                    is_read= is_read)

    # Kiểm soát dữ liệu sau khi chạy AutoArima - với tham số season
    if sts == 'E':
        new_df = pd.DataFrame([{'Type': 'Error', 'Message': msg}])
        err_msg = pd.concat([err_msg, new_df], ignore_index=True)

    if isinstance(warning_datas, pd.DataFrame) and not warning_datas.empty:
            warn_rows = warning_datas[warning_datas['Warning'] == True]
            if not warn_rows.empty:
                warn_df = pd.DataFrame({
                    'Type': 'Warning',
                    'Message': "aicc của " + warn_rows['Category'].astype(str) + " < 0.05"
                })
                err_msg = pd.concat([err_msg, warn_df], ignore_index= True)

    # Kết quả trả về
    return err_msg, predict_data

def my_algo_arimax_define(data, time_forecast: int, ns_option, x_past, x_future, is_check_noise: bool = False, 
                          all_cpu= False, is_debug= False, is_read= False ):
    err_msg: pd.DataFrame = None

    # Thực hiện kiểm tra dữ liệu nhiễu
    if is_check_noise:
        err_msg = check_noise(data= data)
        if isinstance(err_msg, pd.DataFrame) and not err_msg.empty:
            err_msg = err_msg[err_msg['Type'].isin(['Error', 'Warning'])]

    sts, msg, predict_data, warning_datas = algo_auto_arima_define( data= data,
                                                                    time_forecast= time_forecast,
                                                                    ns_option= ns_option,
                                                                    x_past= x_past, 
                                                                    x_future= x_future,
                                                                    all_cpu= all_cpu,
                                                                    is_debug= is_debug,
                                                                    is_read= is_read)

    # Kiểm soát dữ liệu sau khi chạy AutoArima - với tham số 'X' - Exogenous
    if sts == 'E':
            new_df = pd.DataFrame([{'Type': 'Error', 'Message': msg}])
            err_msg = pd.concat([err_msg, new_df], ignore_index=True)
    
    if isinstance(warning_datas, pd.DataFrame) and not warning_datas.empty:
            warn_rows = warning_datas[warning_datas['Warning'] == True]
            if not warn_rows.empty:
                warn_df = pd.DataFrame({
                    'Type': 'Warning',
                    'Message': "aicc của " + warn_rows['Category'].astype(str) + " < 0.05"
                })
                err_msg = pd.concat([err_msg, warn_df], ignore_index= True)

    # Kết quả trả về
    return err_msg, predict_data

def my_algo_sarimax_define(data, time_forecast: int, ns_option, s_option, s_length: int, x_past, x_future, is_check_noise: bool = False, 
                           all_cpu= False, is_debug= False,  is_read= False ):
    err_msg: pd.DataFrame = None

    # Thực hiện kiểm tra dữ liệu nhiễu
    if is_check_noise:
        err_msg = check_noise(data= data)
        if isinstance(err_msg, pd.DataFrame) and not err_msg.empty:
            err_msg = err_msg[err_msg['Type'].isin(['Error', 'Warning'])]

    sts, msg, predict_data, warning_datas = algo_auto_arima_define( data= data,
                                                                    time_forecast= time_forecast,
                                                                    ns_option= ns_option,
                                                                    s_option= s_option,
                                                                    s_length= s_length,
                                                                    x_past= x_past, 
                                                                    x_future= x_future,
                                                                    all_cpu= all_cpu,
                                                                    is_debug= is_debug,
                                                                    is_read= is_read)

    # Kiểm soát dữ liệu sau khi chạy AutoArima - với tham số season và 'X' - Exogenous
    if sts == 'E':
        new_df = pd.DataFrame([{'Type': 'Error', 'Message': msg}])
        err_msg = pd.concat([err_msg, new_df], ignore_index= True)

    if isinstance(warning_datas, pd.DataFrame) and not warning_datas.empty:
        warn_rows = warning_datas[warning_datas['Warning'] == True]
        if not warn_rows.empty:
            warn_df = pd.DataFrame({
                'Type': 'Warning',
                'Message': "aicc của " + warn_rows['Category'].astype(str) + " < 0.05"
            })
            err_msg = pd.concat([err_msg, warn_df], ignore_index= True)

    # Kết quả trả về
    return err_msg, predict_data

if __name__ == "__main__":
    main()