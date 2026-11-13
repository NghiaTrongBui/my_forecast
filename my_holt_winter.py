import pandas as pd
import numpy as np                                              # Thư viện tạo random số
from statsmodels.tsa.holtwinters import ExponentialSmoothing    # type: ignore
from datetime import datetime
from my_tool import get_cwf
from my_tool import join_path
from my_tool import write_csv
from my_tool import read_csv
from my_tool import clear_file

# Function này dùng để test code
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

    list_smooth = [
        {'Category': 'C1', 'Smooth_type': '1'}
    ]

    forecast_data = algo_holt_winter_define(data, 12)
    print(forecast_data)

def algo_holt_winter_define(data, time_forecast: int, s_length: int = 12, smooth= None, operator=None, is_debug=None, is_read= None):
    """
        Giải thích tham số:
            data: data input dạng category, period, revenue
            time_forecast: khoảng thời gian dự đoán
            smooth: sẽ là 1 số giải thuật tự suy nghĩ làm mịn data cho data gốc
                Có dạng dữ liệu kiểu như category, smooth type: theo kiểu nào
            operator: Gắn phương thức nào add, mul cho giải thuật holt-winter
            debug: y/c ghi file để dev vào xem dữ liệu dưới các file trên server
    """
    tmp = pd.DataFrame(data= data)
    if ( data is None  or tmp.empty ) and is_read :
        cwf = get_cwf()
        r_file = join_path(cwf, 'data/original/original_holt_winter.csv')
        data = read_csv(fname= r_file)

    df: pd.DataFrame = None
    if data is not None:
        df = pd.DataFrame(data= data)
    else:
        return False, 'Không có dữ liệu truyền vào.', None, None
    
    # Xóa file dữ liệu trước khi ghi
    if is_debug == True:
        cwf = get_cwf()
        original_file = join_path(path= cwf, file= r'data/original/original_holt_winter.csv')
        smooth_file   = join_path(path= cwf, file= r'data/smooth/smooth_holt_winter.csv')
        param_file    = join_path(path= cwf, file= r'data/param/param_holt_winter.csv')
        forecast_file = join_path(path= cwf, file= r'data/forecast/forecast_holt_winter.csv')

        clear_file(original_file)
        clear_file(smooth_file)
        clear_file(param_file)
        clear_file(forecast_file)

    # Chuẩn hóa dữ liệu thành data frame
    df = pd.DataFrame(data= data)

    # Thực hiện ghi file khi muốn check trọng số
    if is_debug == True:
        write_csv(data= data, fname=original_file)

    # 2. Chuẩn hóa biến smooth thành dictionary (nếu người dùng truyền vào dạng list)
    smooth_dict = {}
    if smooth:
        if isinstance(smooth, dict):
            smooth_dict = smooth
        elif isinstance(smooth, list):
            for item in smooth:
                key = item['Category']
                value = item.get('Smooth_type')
                smooth_dict[key] = value

    # khai báo dạng list of dict về sau
    all_forecasts = [] 
    # 3. Gom nhóm theo Category
    for category, group_df in df.groupby('Category'):
        # 3.1. Chuẩn bị dữ liệu cho category gom nhóm cho tính toán
        # Tách bản sao
        group_df = group_df.copy()

        # Format cột period và sort dữ liệu
        group_df['Period'] = pd.to_datetime(group_df['Period'])
        group_df = group_df.sort_values('Period')

        # 3.2. Xử lý data original về smooth để làm đẹp dữ liệu
        if category in smooth_dict:
            smooth_type = smooth_dict[category]
            match smooth_type:
                case '1':
                    pass

            # Ghi nhận kết quả smooth data vào file    

        # 3.3. Chuyển group_df thành pd.Series để đưa vào algo_holt_winter 
        # --> Chưa nói tới tính toán smooth data
        series_data = pd.Series( data=group_df['Revenue'].values,
                                 index=group_df['Period'] )

        # 3.4. Chạy mô hình Holt-Winters cho nhóm category này
        forecast_series = algo_holt_winter( category=category,
                                            data=series_data,
                                            time_forecast=time_forecast,
                                            s_length= s_length,
                                            operator=operator,
                                            mode_write= 'A' if is_debug == True else None )

        # 3.5. Chuyển kết quả dự báo về chuẩn DataFrame ['Category', 'Period', 'Revenue']
        df_sub_forecast = forecast_series.reset_index()
        df_sub_forecast.columns = ['Period', 'Revenue']
        df_sub_forecast['Category'] = category
        df_sub_forecast['Period'] = df_sub_forecast['Period'].dt.strftime('%Y-%m-%d')
        df_sub_forecast = df_sub_forecast[['Category', 'Period', 'Revenue']]
    
        all_forecasts.append(df_sub_forecast)

    # 4. Gộp toàn bộ kết quả dự báo của các category lại
    if all_forecasts:
        final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
        if is_debug == True:
            write_csv(fname=forecast_file, data=final_forecast_df,option='W')

    return final_forecast_df

# Giải thuật holt-winter
def algo_holt_winter(category, data, time_forecast: int, s_length: int = 12, operator=None, mode_write=None):
    # 1. Khoi tao data model
    match operator:
        case 'ADD':
            model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods= s_length, freq='MS')     
        case 'MUL':
            model = ExponentialSmoothing(data, trend='mul', seasonal='mul', seasonal_periods= s_length, freq='MS')
        case _ :    # Tương tự như else
            model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods= s_length, freq='MS')

    # 2. Lay thong so model => viết vào file để check về sau
    fitted_model = model.fit()

    # 2.1. Ghi nhận tham số va ghi vào file
    params = {
        'Category': category,
        "Level": fitted_model.params['smoothing_level'],
        "Trend": fitted_model.params['smoothing_trend'],
        "Season": fitted_model.params['smoothing_seasonal'],
        "SSE": fitted_model.sse
    }

    if mode_write is not None:
        cwf = get_cwf()
        file = join_path(path= cwf, file=r'data/param/param_hotl_winter.csv')
        status, msg = write_csv(fname=file, data=params, option=mode_write)

    # 3. Dự báo tương lai (ví dụ: dự báo thêm 12 tháng tới)
    data_forecast = fitted_model.forecast(time_forecast)

    return data_forecast

if __name__ == "__main__":
    main()