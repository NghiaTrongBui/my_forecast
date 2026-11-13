import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import ARIMASummary
from statsmodels.stats.diagnostic import acorr_ljungbox
from utilsforecast.processing import make_future_dataframe
from my_tool import get_cwf
from my_tool import join_path
from my_tool import read_csv
from my_tool import write_csv
from my_tool import clear_file

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
    s_option =  {'p': '2','d':'1', 'q': '2'}

    x_past = []
    sale = pd.Series({ 'Category': 'C1',
                        'Period': '2021-05-01',
                        'Khuyen_mai': 1 })
    x_past.append(sale)

    sale = pd.Series({ 'Category': 'C2',
                       'Period': '2021-05-01',
                       'Khuyen_mai': 1 })
    x_past.append(sale)
    x_past = pd.DataFrame(x_past)

    x_future = []
    f_sale = pd.Series({ 'Category': 'C1',
                         'Period': '2024-05-01',
                         'Khuyen_mai': 1 })
    x_future.append(f_sale)

    f_sale = pd.Series({ 'Category': 'C2',
                         'Period': '2024-05-01',
                         'Khuyen_mai': 1 })
    x_future.append(f_sale)
    x_future = pd.DataFrame(x_future)

    sts, msg, predict_data, warning_datas = algo_auto_arima_define( data= data, time_forecast= 12, 
                                                                    ns_option= ns_option, s_option= s_option, s_length= 12, 
                                                                    x_past= x_past, x_future= x_future )

def algo_auto_arima_define(data, time_forecast: int, ns_option, s_option= None, s_length= None, x_past= None, x_future= None, 
                           all_cpu= False, is_debug= False, is_read= False ):
    cwf = ''
    tmp = pd.DataFrame(data= data)
    if ( data is None  or tmp.empty ) and is_read :
        cwf = get_cwf()
        r_file = join_path(cwf, 'data/original/original_sarima.csv')
        data = read_csv(fname= r_file)

    df: pd.DataFrame = None
    if data is not None:
        df = pd.DataFrame(data= data)
    else:
        return False, 'Không có dữ liệu truyền vào.', None, None

    # Thực hiện ghi data input
    if is_debug:
        cwf = get_cwf()
        if s_length is not None or s_length:
            if x_past is not None or x_past:
                original_file = join_path(cwf, 'data/original/original_sarimax.csv')
                holiday_file  = join_path(cwf, 'data/holiday/holiday_sarimax.csv')

                clear_file(holiday_file)
                write_csv(holiday_file)
            else:
                original_file = join_path(cwf, 'data/original/original_sarima.csv')
        else:
            if x_past is not None or x_past:
                original_file = join_path(cwf, 'data/original/original_arimax.csv')
                holiday_file  = join_path(cwf, 'data/holiday/holiday_arimax.csv')

                clear_file(holiday_file)
                write_csv(holiday_file)
            else:
                original_file = join_path(cwf, 'data/original/original_arima.csv')

        clear_file(original_file)
        write_csv(df, original_file)

    # Mapping data event sale to data origin 
    # Sử dụng cho 'X' - Exogenous
    X_future_df = None
    if x_past is not None and isinstance(x_past, (list, dict, pd.DataFrame)):
        envent_df = pd.DataFrame( data= x_past)
        exclude_col = ['Event_name']
        if not envent_df.empty:
            df = pd.merge(df, envent_df.drop(columns=exclude_col, errors= 'ignore'), on=['Category', 'Period'], how='left' )
            df = df.fillna(0)                   # không sài type bool có chuyển sang 1 và 0

        # 1. Tạo khung ngày tương lai (Category, Period) cho 12 tháng tới
        df['Period'] = pd.to_datetime(df['Period'])

        # Lấy ngày cuối cùng truyền data vào trong original
        last_dates = df.groupby('Category')['Period'].max()
        X_future_df = make_future_dataframe(
            uids= last_dates.index,
            last_times= pd.DatetimeIndex(last_dates.values),
            freq= 'MS',
            h= time_forecast,
            id_col= 'Category',
            time_col= 'Period'
        )
    
        # 2. Nếu có truyền x_future vào thì ghép vào khung tương lai
        if x_future is not None and isinstance(x_future, (list, dict, pd.DataFrame)):
            future_df = pd.DataFrame(data= x_future)
            if not future_df.empty:
                future_df['Period'] = pd.to_datetime(future_df['Period'])
                X_future_df = pd.merge(X_future_df, future_df, on=['Category', 'Period'], how='left')
                X_future_df = X_future_df.fillna(0)     # không sài type bool có chuyển sang 1 và 0

    # Thực hiện chạy dự đoán
    status, msg, models, predict_data = algo_auto_arima(
                                            data= df,
                                            time_forecast= time_forecast,
                                            ns_option= ns_option,
                                            s_option= s_option,
                                            s_length= s_length,
                                            x_future= X_future_df,
                                            all_cpu= True if all_cpu else False)

    # Kiểm tra kết quả trả về
    sts = 'S' if status else 'E'

    all_model_params = []
    warning_datas = []
    if status:
        arma_prefixes = ('ar', 'ma', 'sar', 'sma', 'intercept', 'drift')
        for idx, cat_id in enumerate(models.uids):
            sf = models.fitted_[idx, 0].model_
            
            # 1. Lấy thông tin bậc mô hình: arma lưu theo thứ tự [p, q, P, Q, m, d, D]
            order_str = str(ARIMASummary(sf))          # Công thức ARIMA đã chạy là gì
            arma_order = sf.get("arma")                # Get property với tên là arma
            p, q, P, Q, m, d, D = arma_order

            # 2. Tách hệ số ARMA (ar, ma, sar, sma, intercept, drift) và Regression (biến X)
            coefs = sf.get("coef", {}) or {}
            arma_dict = {k: v for k, v in coefs.items() if k.startswith(arma_prefixes)}
            regr_dict = {k: v for k, v in coefs.items() if not k.startswith(arma_prefixes)}

            # Tách riêng từng nhóm hệ số từ coefs (đã làm tròn 3 chữ số)
            ar_vals  = [round(float(v), 3) for k, v in coefs.items() if k.startswith('ar')]
            ma_vals  = [round(float(v), 3) for k, v in coefs.items() if k.startswith('ma')]
            sar_vals = [round(float(v), 3) for k, v in coefs.items() if k.startswith('sar')]
            sma_vals = [round(float(v), 3) for k, v in coefs.items() if k.startswith('sma')]
            x_regr = {k: round(float(v), 3) for k, v in regr_dict.items()}

            arma = pd.Series(arma_dict, name="ARMA Coefficients", dtype=float).round(3)

            # 3. Các chỉ số đánh giá mô hình (Metrics)
            metrics = pd.Series({
                "sigma^2": sf.get("sigma2", np.nan),
                "loglik": sf.get("loglik", np.nan),
                "aic": sf.get("aic", np.nan),
                "aicc": sf.get("aicc", np.nan),
                "bic": sf.get("bic", np.nan),
            }, name="Metrics", dtype=float).round(2)

            # 4. Phần dư (Residuals) & Kiểm định Ljung-Box 
            # =>  Kiểm tra nhiễu trắng cho phần Residuals để tự tin kết quả dự báo
            # Luôn luôn nhớ bộ trọng số của dòng họ ARIMA: (p,d,q)(P,D,Q)m
            m = s_length
            if s_length is None:
                m = 0
            resids = pd.Series(sf["residuals"]).dropna()
            T = len(resids)
            n_params = p + q + P + Q
            target_lag = min(2 * m if m > 1 else 10, T // 5)
            safe_lag = max(n_params + 1, min(target_lag, T - 1))

            ljung_box = acorr_ljungbox(resids, lags=[safe_lag], model_df=n_params)

            model_param = pd.Series({
                'Category': cat_id,
                'algo': order_str,
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q, 'm': m,
                'ar': ar_vals if ar_vals else None,
                'ma': ma_vals if ma_vals else None,
                'sar': sar_vals if sar_vals else None,
                'sma': sma_vals if sma_vals else None,
                'intercept': arma.get('intercept', None),
                'drift': arma.get('drift', None),
                'X': x_regr if x_regr else None,
                'sigma^2': metrics.get('sigma^2', None),
                'loglik': metrics.get('loglik', None),
                'aic': metrics.get('aic', None),
                'aicc': metrics.get('aicc', None),
                'bic': metrics.get('bic', None),
                'ljung_box': ljung_box
            })

            all_model_params.append(model_param)

            # Kiểm soát các bộ data predict lỗi => thông báo
            warning_data = pd.Series({
                'Category': cat_id, 
                'Warning': True if round(float(ljung_box.get('lb_pvalue').values[0]), 2) < 0.05 else False
            })
            warning_datas.append(warning_data)

        all_model_params = pd.DataFrame(all_model_params)   
        warning_datas = pd.DataFrame(warning_datas) 

    # Kiểm soát có nhóm data phân tich lỗi aicc quá thấp
    if not warning_data.empty:
        sts = 'W'

    if is_debug:
        if s_length is not None or s_length:
            if x_past is not None or x_past:
                forecast_file = join_path(cwf, r'data/forecast/forecast_sarimax.csv')
                param_file    = join_path(cwf, r'data/param/param_sarimax.csv')
            else:
                forecast_file = join_path(cwf, r'data/forecast/forecast_sarima.csv')
                param_file    = join_path(cwf, r'data/param/param_sarima.csv')
        else:
            forecast_file = join_path(cwf, r'data/forecast/forecast_arima.csv')
            param_file    = join_path(cwf, r'data/param/param_arima.csv')

        clear_file(forecast_file)
        clear_file(param_file)

        write_csv(fname= forecast_file, data= predict_data)
        write_csv(fname= param_file, data= all_model_params)

    return sts, msg, predict_data, warning_datas

"""
    Chú ý 8 tham số: 
        p, d, q: của None seasonality - ARIMA
        P, D, Q: của seasonality - SARIMA, SARIMAX
        m: lượng cột roll back để các mùa vụ các nắm trừ nhau - thuật ngữ là  Backshift notation
        X: trong SARIMAX sự kiện bất khả kháng để giải thuật chặn độ nhiếu(noise) -> trọng số nhiễu lỗi
    data: Dữ liệu quá khứ để tính toán trị số master => dự đoán tương lai
        có dạng list of dict: [{Category, Period, Revenue}]
    ns_option: None seasonality option - thiết lập trọng số cho ARIMA chạy không có seasonality - mùa vụ
        có dạng tuple hoặc dict (p,d,q)
    s_option: seasonality option - thiết lập trọng số cho SARIMA, SARIMAX chạy có seasonality
        có dạng tuple hoặc dict (P, D, Q)
    s_length: Season length - bắt buộc thiết lập khi chạy giải thuật SARIMA, SARIMAX
        có dạng: int(nó là tham số m)
"""
def algo_auto_arima(data, time_forecast: int, ns_option, s_option= None, s_length: int = None, x_future= None, all_cpu= False):
    msg = ''
    df = None
    if isinstance(data, (pd.DataFrame)) and not data.empty:
        df = pd.DataFrame(data= data)
        df = df.rename(columns={ "Category": "unique_id",   
                                 "Period": "ds" })
    else:
        msg = 'Không có data training'
        return False, msg, None, None

    #  Future data frame
    f_df: pd.DataFrame = None
    if isinstance(x_future, (pd.DataFrame)) and not x_future.empty:
        f_df = pd.DataFrame(x_future)
        f_df = f_df.rename(columns={ "Category": "unique_id",   
                                     "Period": "ds" })

    # Xử lý dữ liệu cho non-seasonality
    max_d = None
    max_p = None
    max_q = None
    if isinstance(ns_option, dict):
            max_p = pd.to_numeric( ns_option['p'] )
            max_d = pd.to_numeric( ns_option['d'] )
            max_q = pd.to_numeric( ns_option['q'] )
    elif isinstance(ns_option, tuple):
        max_p, d, max_q = ns_option
    else:
        msg = 'Chưa set hoặc data không thuộc (dict/tuple) tham số cho None seasonality'
        return False, msg, None, None

    # Xử lý dữ liệu cho seasonality
    max_D = None
    max_P = None
    max_Q = None

    if s_option is not None:
        if isinstance(s_option, dict):
                max_P = pd.to_numeric( s_option['p'] )
                max_D = pd.to_numeric( s_option['d'] )
                max_Q = pd.to_numeric( s_option['q'] )
        elif isinstance(s_option, tuple):
            max_P, D, max_Q = s_option
        else:
            msg = 'Chưa set hoặc data không thuộc (dict/tuple) tham số cho Seasonality'
            return False, msg, None, None

    models = []
    # Xác nhận giải thuật nào ARIMA hay SARIMA
    if s_length is None:
        max_D = 0
        max_Q = 0
        max_P = 0
        s_length = 0

    models = [ AutoARIMA(   max_d= max_d, max_p= max_p, max_q= max_q, start_p= 0, start_q= 0,
                            max_D= max_D, max_P= max_P, max_Q= max_Q, start_P= 0, start_Q= 0, 
                                seasonal= ( True if s_length is not None else False ), 
                                season_length= s_length,
                            stepwise= True , allowmean= True if x_future is not None else False,
                            allowdrift= True if x_future is not None else False ) ]

    sf = StatsForecast(models= models, freq='MS', n_jobs= -1 if all_cpu else 1)

    # Chạy ARIMA, ARIMAX, SARIMA, SARIMAX
    #   Set target_col -> cột dự đoán
    #   Trừ unique_id và ds ra thì tất cả còn lại là tham số X
    models_fit = None
    models_fit = sf.fit(df= df, target_col='Revenue')

    if models_fit is not None:
        data_forecast = models_fit.predict(h= time_forecast, X_df= f_df)
        return True, msg, models_fit, data_forecast
    else:
        msg = 'Không thể chạy đc mô hình Forecast.'
        return False, msg, None, None

if __name__ == "__main__":
    main()