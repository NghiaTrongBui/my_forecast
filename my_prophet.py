import pandas as pd
import numpy as np
from prophet import Prophet 
from prophet.utilities import regressor_coefficients
from my_tool import get_cwf
from my_tool import join_path
from my_tool import write_csv
from my_tool import clear_file
from my_tool import smooth_zero_to_nan
from my_tool import read_csv

def main():
    np.random.seed(36)
    time = np.arange(36)
    dates = pd.date_range(start='2021-01-01', periods= 36, freq='MS') # 48 tháng

    data = []
    for cat in ['C1']:
        # Phát sinh ngẫu nhiên 36 giá trị doanh thu cho mỗi Category
        revenues = 10 + 0.5 * time + 5 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 1, 36)
        
        # Ghép từng ngày (d) và doanh thu (rev) thành dict rồi thêm vào list data
        for d, rev in zip(dates, revenues):
            data.append({
                'Category': cat,
                'Period': d.strftime('%Y-%m-%d'),
                'Revenue': round(float(rev), 2)
            })

    holiday = [ { "Category": "C1", "Period":"2021-02-01", "Event_name":"TET"},
                { "Category": "C1", "Period":"2022-03-01", "Event_name":"TET"},
                { "Category": "C1", "Period":"2023-02-01", "Event_name":"TET"}
              ]

    s_holiday = { 'Lower': -1, 'Upper': 0 }

    forecast_data = algo_prophet_define( data= data, time_forecast= 12, is_smooth= True, holiday= holiday, s_holiday= s_holiday)

    print(forecast_data)

def algo_prophet_define(data, time_forecast: int, holiday= None, s_holiday= None, is_smooth= None, operator= None, is_debug= None, is_read= None):
    """
        Giải thích tham số:
            data: data input dạng category, period, revenue
            time_forecast: khoảng thời gian dự đoán
            holiday: có dạng là catgory, period, event_name
            smooth: sẽ là 1 số giải thuật tự suy nghĩ làm mịn data cho data gốc
                Có dạng dữ liệu kiểu như category, smooth type: theo kiểu nào
            operator: Gắn phương thức nào add, mul cho giải thuật holt-winter
            is_debug: y/c ghi file để dev vào xem dữ liệu dưới các file trên server
    """
    tmp = pd.DataFrame(data= data)
    if ( data is None  or tmp.empty ) and is_read :
        cwf = get_cwf()
        r_file = join_path(cwf, 'data/original/original_holt_winter.csv')
        data = read_csv(fname= r_file)

    # 1. Xóa file dữ liệu trước khi ghi
    if is_debug == True:
        cwf = get_cwf()
        original_file = join_path(path= cwf, file= r'data/original/original_prophet.csv')
        holiday_file  = join_path(path= cwf, file= r'data/smooth/holiday_prophet.csv')
        smooth_file   = join_path(path= cwf, file= r'data/smooth/smooth_prophet.csv')
        param_file    = join_path(path= cwf, file= r'data/param/param_prophet.csv')
        forecast_file = join_path(path= cwf, file= r'data/forecast/forecast_prophet.csv')

        clear_file(original_file)
        clear_file(holiday_file)
        clear_file(smooth_file)
        clear_file(param_file)
        clear_file(forecast_file)
    
    # 2. Chuẩn hóa dữ liệu thành data frame - data chuẩn dự báo + data holiday
    original_df = pd.DataFrame(data= data)
    holiday_df = pd.DataFrame(data= holiday)

    # Kiểm tra smooth dữ liệu
    if is_smooth == True:
        smooth_df = smooth_zero_to_nan(data= data)

    # Thực hiện ghi file khi muốn check trọng số
    if is_debug == True:
        write_csv(data= original_df, fname= original_file)
        write_csv(data= holiday_df, fname= holiday_file)
        write_csv(data= smooth_df, fname= smooth_file)
        
    if is_smooth == True:
        original_df = smooth_df

    # Khởi tạo data dự báo kết quả cuối cùng 
    all_forecasts = []
    all_params    = []
    for category, group_data_df in original_df.groupby('Category'):
        # 3.1. Chuẩn bị dữ liệu cho category gom nhóm cho tính toán
        # Tách bản sao
        group_df = group_data_df.copy()

        # Format cột period và sort dữ liệu
        group_df['Period'] = pd.to_datetime(group_df['Period'])
        group_df = group_df.sort_values('Period')

        # 3.2. Chuẩn bị data holiday nếu có
        group_holiday_df = None
        if not holiday_df.empty and 'Category' in holiday_df.columns:
            # Lệnh lấy tất cả các Category có cùng với original trong holiday_df
            matched_holiday = holiday_df[holiday_df['Category'] == category].copy()
            if not matched_holiday.empty:
                group_holiday_df = matched_holiday

        # 3.3. Gọi hàm algo_prophet để dự báo cho category này
        status, model, forecast_data = algo_prophet(
            data=group_df,
            time_forecast=time_forecast,
            holiday=group_holiday_df,
            s_holiday= s_holiday,
            operator=operator
        )

        # 3.4. Chuẩn hóa kết quả dự báo trả về theo chuẩn ['Category', 'Period', 'Revenue']
        if status and forecast_data is not None:
            # Lấy kế quả dự báo
            df_sub_forecast = forecast_data[['ds', 'yhat']].tail(time_forecast).copy()

            # Format về dạng cột Period, Revenue
            df_sub_forecast.columns = ['Period', 'Revenue']

            # Gắn thêm cột Category, format date, định dạng cột kết quả dự báo
            df_sub_forecast['Category'] = category
            df_sub_forecast['Period'] = df_sub_forecast['Period'].dt.strftime('%Y-%m-%d')
            df_sub_forecast = df_sub_forecast[['Category', 'Period', 'Revenue']]

            all_forecasts.append(df_sub_forecast)

            # 1. K: Tốc độ tăng trưởng cơ bản ban đầu của đường xu hướng (Base growth rate)
            base_k = float(model.params['k'][0][0]) if model.mcmc_samples == 0 else float(np.mean(model.params['k']))

            # 2. m: Tham số điểm cắt / độ lệch gốc (Offset parameter)
            offset_m = float(model.params['m'][0][0]) if model.mcmc_samples == 0 else float(np.mean(model.params['m']))

            # 3. delta: Mảng trọng số điều chỉnh tốc độ tăng trưởng tại các điểm gãy xu hướng (tương ứng với model.changepoints)
            delta_weights = model.params['delta'][0] if model.mcmc_samples == 0 else np.mean(model.params['delta'], axis=0)

            # 4. Beta: Mảng chứa toàn bộ trọng số của chuỗi Fourier (tính mùa vụ), ngày lễ (holidays) và các biến ngoại sinh
            beta_weights = model.params['beta'][0] if model.mcmc_samples == 0 else np.mean(model.params['beta'], axis=0)

            # 5. sigma_obs: Độ lệch chuẩn của sai số nhiễu
            sigma_obs = float(model.params['sigma_obs'][0][0]) if model.mcmc_samples == 0 else float(np.mean(model.params['sigma_obs']))

            # Đóng gói các thông số ẩn sau khi fit() vào 1 biến dict
            prophet_params = {
                'Category': category,
                'K': base_k,
                'm': offset_m,
                'delta': delta_weights,
                'Beta': beta_weights,
                'sigma_obs': sigma_obs
            }

            all_params.append(prophet_params)

    # Write file cho forecast, 
    if not is_debug is None:
        cwf = get_cwf()
        forecast_file = join_path(path= cwf, file= r'data/forecast/forecast_prophet.csv')
        forecast_df = pd.DataFrame({
                        'Category': category,
                        'Period': forecast_data['ds'].tail(time_forecast), 
                        'forecast': forecast_data['yhat'].tail(time_forecast), 
                        'forecast_lower': forecast_data['yhat_lower'].tail(time_forecast), 
                        'forecast_upper': forecast_data['yhat_upper'].tail(time_forecast)                        
                    })
        write_csv(data= forecast_df, fname= forecast_file)

    # 4. Gom tất cả toàn bộ bản dự đoán về 1 thể thống nhất
    final_forecast_df: pd.DataFrame = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()
    final_params_df: pd.DataFrame = pd.DataFrame(data=all_params)
    if is_debug == True:
        write_csv(fname= forecast_file, data= final_forecast_df, option= 'W')
        write_csv(fname= param_file, data= final_params_df, option= 'W')

    return final_forecast_df

# Đặt dấu ??? tại chỗ khi truyền holiday nó sẽ đổ lượng của tháng đó về tháng trước/sau và trọng lượng ảnh hưởng nó sẽ 
# được tính toán bao nhiêu
def algo_prophet( data, time_forecast: int, holiday= None, s_holiday= None, operator= None):
    # Tính toán độ nở rộng/thu hẹp của dư đoán theo dạng nhân hay cộng
    match operator:
            case 'ADD':
                season_mode = 'additive'
            case 'MUL':
                season_mode = 'multiplicative'
            case _:
                season_mode = 'additive'

    # Convert data history
    if data is not None:
        history_df = pd.DataFrame( {
            'ds':pd.to_datetime(data['Period']),
            'y':pd.to_numeric(data['Revenue'])
         })
    else:
         return False, None, None

    # Convert data History
    low_w: int = 0
    upp_w: int = 0
    if s_holiday is not None:
        if isinstance(s_holiday, tuple):
            low_w = s_holiday[0]
            upp_w = s_holiday[1]
        elif isinstance(s_holiday, dict):
            low_w = s_holiday['Lower']
            upp_w = s_holiday['Upper']

    holiday_df = None
    if holiday is not None:
        raw_holiday = pd.DataFrame(holiday)

        holiday_df = pd.DataFrame({
            'ds': pd.to_datetime(raw_holiday['Period']),
            'holiday': raw_holiday['Event_name'],
            'lower_window': low_w,                      # Số lượng kỳ trước ảnh hưởng   ==> Muốn chỉnh cái này phải biết nguyên tắc kd thực tế
            'upper_window': upp_w,                      # Số lượng kỳ sau ảnh hưởng     ==> Muốn chỉnh cái này phải biết nguyên tắc kd thực tế
            'prior_scale': 10.0                         # Tỉ lệ co giản biên độ trend của khuyến mãi
        })

    """
        Lý giải tham số:
            - changepoint_range: lấy tỉ lệ % data train để tính toán công thức
            - yearly_seasonality: Xác định data truyền vào theo dạng năm
            - changepoint_prior_scale: Tỉ lệ cho vẽ đường cong fit với các data 
                => Càng cao càng ngu -> overfit lịch sữ 
            - seasonality_mode: Dự đoán sự phình to/sụt doanh số nhanh hay chậm
            => Muốn điều chỉnh sơ đồ này thì chỉ có thể diều chỉnh 2 tham số này
                changepoint_range và changepoint_prior_scale
    """
    model = Prophet( changepoint_range= 0.8,             
                     yearly_seasonality= True, 
                     changepoint_prior_scale= 0.05,      # Tỉ lệ co giản biên độ trend của toàn sơ đồ trừ kỳ khuyến mãi
                     changepoints= None, 
                     seasonality_mode= season_mode,
                     holidays= holiday_df
                    )
    model.fit(history_df)

    future = model.make_future_dataframe(periods=time_forecast, freq='MS')
    forecast_data = model.predict(future)

    return True, model, forecast_data

if __name__ == "__main__":
    main()