import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import quote_plus
from sqlalchemy import create_engine        # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore
import matplotlib.pyplot as plt

### Khai báo cấu trúc chuẩn dữ liệu
# Data type import phân tích
std_type_imprt = ['Category','Period', 'Revenue']

col_chk_zero = ['Revenue']
col_chk_key  = ['Category', 'Period']

### Data type import sự kiện(nếu có)
## Data type mapping cột
# csv
map_f_csv = {}
# xml
map_f_xml = {}

# MS sql
map_f_sql = {}

### Ham để gắn test code
def main():
    np.random.seed(42)  # Cố định seed để dễ tái lập kết quả test

    categories = ['C1', 'C2']
    # Tạo chuỗi 36 tháng tăng dần bắt đầu từ tháng 01/2021 (2021-01 -> 2023-12)
    periods = pd.date_range(start='2021-01-01', periods=36, freq='MS').strftime('%Y-%m').tolist()

    s_option = (15, 7)  # Kích thước cửa sổ biểu đồ (rộng 15 inch, cao 7 inch)
    l_option = [{'Lable' : 'T_C1', "Category":"C1", "Color": "red"}]
    
    data = []
    for cat in categories:
        # Đặt mức doanh thu cơ sở khác nhau cho từng Category
        base_revenue = 1000.0 if cat == 'C1' else 1500.0

        for period in periods:
            # Giá trị ngẫu nhiên cơ bản + phần nhiễu (phân phối chuẩn N(0, 150))
            random_val = np.random.uniform(800, 1200)
            noise = np.random.normal(loc=0, scale=150)

            # Xác suất nhỏ (10%) tạo nhiễu đột biến (outlier) để test các hàm kiểm tra nhiễu
            if np.random.rand() < 0.1:
                noise += np.random.choice([-1, 1]) * np.random.uniform(500, 900)

            revenue = round(max(0.0, base_revenue + random_val + noise), 2)

            data.append({
                'Category': cat,
                'Period': period,
                'Revenue': revenue
            })

    my_paint(title= 'Test paint', data= data, l_option= l_option, sc_option= s_option )

### Định nghĩa hàm sử dụng
# Get current file
def get_cwf():
    crfile = Path(__file__).resolve()
    return crfile.parent

# Gán chính xác vị trí + tên file
def join_path(path, file):
    return Path.joinpath(path, file)

# Đọc file CSV
def read_csv(fname):
    if map_f_csv:
       # 1. Chỉ đọc đúng các cột có trong map_f_csv (dùng map_f_csv.keys())
        cols_to_read = list(map_f_csv.keys())
        df = pd.read_csv(fname, usecols=cols_to_read)
        
        # 2. Đổi tên cột từ CSV sang tên chuẩn trong std_type_imprt
        df = df.rename(columns=map_f_csv)
        
        # 3. Sắp xếp lại đúng thứ tự các cột chuẩn
        df = df[std_type_imprt]
    else:
        df = pd.read_csv(fname, usecols= lambda col: col in std_type_imprt)

    return df

# Thực hiện write file csv
def write_csv(fname, data, option='W'):
    """
    Ghi dữ liệu ra file CSV.
    
    :param fname: Đường dẫn / Tên file CSV (vd: 'data.csv' hoặc Path('data.csv'))
    :param data: Dữ liệu cần ghi (có thể là pd.DataFrame, list of dicts, hoặc 1 dict)
    :param option: 
        - 'W' (Overwrite): Ghi đè (xóa nội dung cũ nếu file đã tồn tại).
        - 'A' (Append): Ghi nối tiếp vào cuối file (không ghi lặp lại tiêu đề header).
    :return: tuple (bool: Thành công hay thất bại, str: Thông báo chi tiết)
    """
    try:
        option = option.strip().upper()
        if option not in ['A', 'W']:
            return False, f"Tùy chọn option='{option}' không hợp lệ! Chỉ chấp nhận 'A' (Append) hoặc 'W' (Overwrite)."
        
        # 1. Chuyển đổi data thành DataFrame nếu data là list/dict
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, (list, tuple)):
                df = pd.DataFrame(data)
            else:
                return False, "Dữ liệu 'data' phải là DataFrame, list of dicts hoặc dict!"
        else:
            df = data

        # 2. Tự động tạo thư mục chứa nếu chưa tồn tại
        file_path = Path(fname)
        if file_path.parent and not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = file_path.exists()
        # 3. Xử lý các kịch bản ghi file
        if not file_exists:
            # TH 1: File chưa tồn tại -> Luôn tạo mới và có ghi tiêu đề cột (header=True)
            df.to_csv(file_path, mode='w', index=False, header=True, encoding='utf-8-sig')
            msg = f"Tạo mới file '{file_path.name}' và ghi thành công {len(df)} dòng!"
        else:
            # TH 2: File đã tồn tại
            match option:
                case 'W':
                    # 2.1: Ghi đè (Overwrite) -> Xóa toàn bộ nội dung cũ, ghi lại từ đầu có header
                    df.to_csv(file_path, mode='w', index=False, header=True, encoding='utf-8-sig')
                    msg = f"Ghi đè (Overwrite) thành công {len(df)} dòng vào '{file_path.name}'!"
                case 'A':
                    # 2.2: Thực hiện ghi data vào dòng tiếp theo
                    # Kiểm tra nếu file rỗng (0 bytes) thì vẫn cần ghi header, ngược lại thì bỏ header
                    is_empty = file_path.stat().st_size == 0
                    df.to_csv(file_path, mode='a', index=False, header=is_empty, encoding='utf-8-sig')
                    msg = f"Ghi nối tiếp (Append) thành công {len(df)} dòng vào '{file_path.name}'!"
                
        return True, msg
    except Exception as e:
        err_msg = f"Lỗi khi ghi file CSV: {str(e)}"
        return False, err_msg
    
# Đọc file XML
def read_xml(fname):
    # Đọc toàn bộ bảng XML vào DataFrame
    df = pd.read_xml(fname)
    
    if map_f_xml:
        # 1. Chỉ lấy các thẻ có trong map_f_xml
        cols_to_keep = list(map_f_xml.keys())
        df = df[cols_to_keep]
        
        # 2. Đổi tên thẻ thành tên cột chuẩn
        df = df.rename(columns=map_f_xml)
        
        # 3. Sắp xếp lại thứ tự cột
        df = df[std_type_imprt]
    else:
        # Lọc các cột trùng với std_type_imprt
        df = df[[col for col in std_type_imprt if col in df.columns]]
        
    return df

# Xóa trắng dữ liệu file tồn tại
def clear_file(fname):
    path = Path(fname)
    if  path.exists():
        open(path, 'w').close()

# Tạo engine cũng như tạo connection - C/C++ hoặc C#(Thuật ngữ)
def engine_create(server=None, database=None, username=None, password=None, driver='ODBC Driver 17 for SQL Server'):
    # 1. Định nghĩa dữ liệu db Server
    server = server or r'NghiaBui'  # hoặc 'TÊN_MÁY_TÍNH' hoặc IP '192.168.1.x'
    database = database or r'nb_test'
    # username='sa'
    # password='your_password'

    if username and password:
        odbc_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};"
    else:
        odbc_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
    
    # 2. Tạo connection string cho SQLAlchemy thông qua mssql+pyodbc
    message = ''
    try:
        engine_url = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"
        engine = create_engine(engine_url)

        with engine.connect() as conn:
            message = 'Success'
            return True, message, engine
    except Exception as e:
        message = f"Không thể kết nối tới SQL Server ({server}/{database}). Chi tiết lỗi: {e}"
        return False, message, None

# Tạo câu truy vấn cho Insert, Select
def query_create(option, tbl, field):
     # 1. Chuẩn hóa tham số field thành danh sách các cột (list)
    if isinstance(field, str):
        if field.strip() == "*":
            field_list = ["*"]
        else:
            # Tách chuỗi theo dấu phẩy và xóa khoảng trắng thừa
            field_list = [f.strip() for f in field.split(",") if f.strip()]
    elif isinstance(field, (list, tuple, set)):
        field_list = [str(f).strip() for f in field]
    else:
        raise ValueError("Tham số 'field' phải là danh sách (list/tuple) hoặc chuỗi (str)!")

    # 2. Phân tách cho select, insert
    match option:
        case 'I':
            # Create insert query
            if field_list == ["*"]:
                raise ValueError("Lệnh INSERT không hỗ trợ field='*'. Cần chỉ định danh sách tên cột cụ thể!")
        
            cols = ", ".join(field_list)
            # Sinh các tham số dạng :col_name để truyền dữ liệu an toàn trong SQLAlchemy
            placeholders = ", ".join([f":{col}" for col in field_list])
        
            query = f"INSERT INTO {tbl} ({cols}) VALUES ({placeholders})"
        case 'S':
            # Create select query
            if field_list == ["*"]:
                cols = "*"
            else:
                cols = ", ".join(field_list)
                
            query =  f"SELECT {cols} FROM {tbl}"

    return query

# Thực hiện Select
def select_sql(query): 
    status, message, engine = engine_create()
    
    # Đọc kết quả câu query thẳng vào Pandas DataFrame
    if status == False:
        return message, None

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if map_f_sql:
        # 1. Chỉ lấy đúng các cột có trong map_f_sql
        cols_to_keep = list(map_f_sql.keys())
        df = df[cols_to_keep]
        
        # 2. Đổi tên cột từ DB sang tên chuẩn trong std_type_imprt
        df = df.rename(columns=map_f_sql)
        
        # 3. Sắp xếp lại đúng thứ tự các cột chuẩn
        df = df[std_type_imprt]
    else:
        # Nếu không có mapping, chỉ giữ các cột trùng tên với std_type_imprt
        df = df[[col for col in std_type_imprt if col in df.columns]]
        
    return message, df

# Thực hiện Insert
def insert_sql(query, data): 
    status, message, engine = engine_create()

    if status == False:
        return False, message

    # Thực thi và bắt lỗi
    try:
        # Nếu data là 1 dict đơn lẻ, bọc vào list để execute đồng nhất
        params = [data] if isinstance(data, dict) else data

        with engine.connect() as conn:
            conn.execute(query, params)
            conn.commit()  
        
        msg = f"Insert thành công {len(params)} dòng dữ liệu!"
        return True, msg
    except SQLAlchemyError as e: 
        conn.rollback()  
        err_msg = f"Lỗi SQL Server: {str(e.orig) if hasattr(e, 'orig') else str(e)}"
        return False, err_msg
    except Exception as e:
        conn.rollback()
        err_msg = f"Lỗi không xác định: {str(e)}"
        return False, err_msg

# Hàm chuyển đổi dữ liệu 0 thành nan cho thuật toán prophet
#   tránh cho prophet nó ngáo dữ liệu
def smooth_zero_to_nan(data):
    if data is None:
        return None
    
    # 1. Copy hoặc khởi tạo data frame 
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    if df.empty or 'Revenue' not in df.columns:
        return df
    
    # 2. Ép kiểu cột Revenue sang số (đề phòng dữ liệu truyền vào là chuỗi '0')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')

    # 3. Chuyển đổi 0 -> nan
    df['Revenue'] = df['Revenue'].replace(0, np.nan)
    return df

# Hàm vẽ biểu đồ đường (Line Chart) trực quan hóa dữ liệu thực tế, tập test và kết quả dự báo
def my_paint(title: str, data: list | pd.DataFrame , test_data: list | pd.DataFrame = None, predict_data: list | pd.DataFrame = None, 
             sc_option: tuple = None, l_option: list | pd.DataFrame  = None):
    """
    Vẽ biểu đồ đường (Line Chart) so sánh dữ liệu gốc (data), dữ liệu kiểm thử (test_data) 
    và dữ liệu dự báo (predict_data) theo từng nhóm Category qua các kỳ (Period).

    :param title: Tiêu đề của biểu đồ.
    :param data: Dữ liệu chính/thực tế cần vẽ (list of dicts, dict hoặc pd.DataFrame) chứa các cột 'Category', 'Period', 'Revenue'.
    :param test_data: (Tùy chọn) Dữ liệu tập test để đối chiếu (vẽ bằng nét đứt '--').
    :param predict_data: (Tùy chọn) Dữ liệu dự báo tương lai (vẽ bằng nét chấm gạch '-.').
    :param sc_option: (Tùy chọn) Tuple quy định kích thước khung hình (width, height). Mặc định là (10, 5).
    :param l_option: (Tùy chọn) Cấu hình hiển thị cho từng đường theo Category (list of dicts hoặc DataFrame),
                     gồm các khóa/cột: 'Category' (mã nhóm), 'Lable' (tên hiển thị trên chú thích), 'Color' (màu sắc đường).
    :return: None (Hiển thị trực tiếp cửa sổ biểu đồ Matplotlib).
    """
    # 1. Xử lý kích thước màn hình hiển thị (figure size tính theo đơn vị inch, mặc định 1 inch = 100 pixels)
    # Nếu truyền sc_option hợp lệ thì lấy theo cấu hình, ngược lại dùng kích thước mặc định (14, 7)
    if sc_option is not None and len(sc_option) >= 2:
        # Nếu truyền giá trị lớn theo pixel (vd: 1400, 700) thì tự động chia 100 để quy đổi sang inch
        fig_w = sc_option[0] / 100 if sc_option[0] > 50 else sc_option[0]
        fig_h = sc_option[1] / 100 if sc_option[1] > 50 else sc_option[1]
        plt.figure(figsize=(fig_w, fig_h))
    else:
        plt.figure(figsize=(14, 7))

    # 2. Xử lý cấu hình định dạng đường vẽ (nhãn hiển thị và màu sắc cho từng Category)
    # Cấu trúc mẫu: l_option = [{"Category": "C1", "Lable": "T_C1", "Color": "red"}]
    l_df: pd.DataFrame = None
    if l_option is not None:
        if isinstance(l_option, (list, pd.DataFrame)):
            # Chuyển cấu hình về DataFrame và đặt cột 'Category' làm index để tra cứu nhanh
            l_df = pd.DataFrame(l_option)
            l_df = l_df.set_index('Category')

    # 3. Xử lý và vẽ đường dữ liệu chính (data - nét liền '-')
    if isinstance(data, (list, dict, pd.DataFrame)) and data is not None:
        # Chuẩn hóa dữ liệu đầu vào sang DataFrame và ép kiểu dữ liệu cho các cột tọa độ
        df = pd.DataFrame(data= data)
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')  # Trục Y: Doanh thu (kiểu số)
        df['Period'] = df['Period'].astype(str)                        # Trục X: Kỳ thời gian (kiểu chuỗi)

        # Gom nhóm theo từng Category để vẽ từng đường dữ liệu riêng biệt lên biểu đồ
        for Category, group_df in df.groupby('Category'):
            group_df = group_df.copy()
            # Tạo Series với index là Period (trục X) và values là Revenue (trục Y)
            plt_df = pd.Series( data=group_df['Revenue'].values, index=group_df['Period'] )

            # Khởi tạo nhãn và màu sắc mặc định cho từng nhóm Category
            label = f'Đường data {Category}'
            color = 'lightblue'

            # Nếu Category có trong bảng cấu hình l_df thì lấy tên nhãn (Lable) và màu sắc (Color) tương ứng
            if l_df is not None and not l_df.empty and Category in l_df.index:
                label = f"Đường data {l_df['Lable'].get(Category, Category)}"
                color = l_df['Color'].get(Category, 'lightblue')

            # Vẽ đường dữ liệu chính với điểm đánh dấu hình tròn (marker='o')
            plt.plot(plt_df, label= label, marker='o', color= color)
    
    else:
        raise Exception('Check lại data input.')

    # 4. Xử lý và vẽ đường dữ liệu kiểm thử (test_data - nét đứt '--')
    if test_data is not None:
        if isinstance( test_data, (list, dict, pd.DataFrame)):
            # Giải phóng biến df cũ và chuẩn hóa lại dữ liệu test_data
            del df
            df = pd.DataFrame(data= test_data)
            df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
            df['Period'] = df['Period'].astype(str)

            # Duyệt qua từng nhóm Category trong tập test để vẽ đường đối chiếu
            for Category, group_df in df.groupby('Category'):
                group_df = group_df.copy()
                plt_df = pd.Series( data=group_df['Revenue'].values, index=group_df['Period'] )

                label = f'Đường test {Category}'
                color = 'lightblue'

                # Áp dụng cấu hình tên nhãn và màu sắc nếu Category có trong bảng cấu hình l_df
                if l_df is not None and not l_df.empty and Category in l_df.index:
                    label = f"Đường test {l_df['Lable'].get(Category, Category)}"
                    color = l_df['Color'].get(Category, 'lightblue')

                # Vẽ đường test bằng nét đứt (linestyle='--') để phân biệt với dữ liệu gốc
                plt.plot( plt_df, label= label, linestyle= '--', marker='o', color= color )
        else:
            raise Exception('Check lại data test input.')

    # 5. Xử lý và vẽ đường dữ liệu dự báo (predict_data - nét chấm gạch '-.')
    if predict_data is not None:
        if isinstance( predict_data, (list, dict, pd.DataFrame)):
            # Giải phóng biến df cũ và chuẩn hóa lại dữ liệu predict_data
            del df
            df = pd.DataFrame(data= predict_data)
            df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
            df['Period'] = df['Period'].astype(str)

            # Duyệt qua từng nhóm Category trong tập dự báo để vẽ đường dự đoán
            for Category, group_df in df.groupby('Category'):
                group_df = group_df.copy()
                plt_df = pd.Series( data=group_df['Revenue'].values, index=group_df['Period'] )

                label = f'Đường predict {Category}'
                color = 'lightblue'

                # Áp dụng cấu hình tên nhãn và màu sắc nếu Category có trong bảng cấu hình l_df
                if l_df is not None and not l_df.empty and Category in l_df.index:
                    label = f"Đường predict {l_df['Lable'].get(Category, Category)}"
                    color = l_df['Color'].get(Category, 'lightblue')

                # Vẽ đường dự báo bằng nét chấm gạch (linestyle='-.')
                plt.plot(plt_df, label= label, linestyle='-.', marker='o', color= color)
            
        else:
            raise Exception('Check lại data test input.')

    # 6. Thiết lập thông tin chung cho biểu đồ, mở rộng khung vẽ và hiển thị
    plt.title(title)                        # Đặt tiêu đề biểu đồ
    plt.xticks(rotation=45, ha='right')     # Xoay nghiêng nhãn trục X 45 độ để không bị đè chữ
    plt.legend(loc = 'upper left')          # Hiển thị bảng chú thích (legend) ở góc trên bên trái
    plt.grid(True)                          # Bật lưới tọa độ để dễ gióng giá trị

    # Mở rộng khung biểu đồ (khuôn đỏ) ra sát 4 mép cửa sổ, loại bỏ khoảng trắng thừa
    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.12)

    plt.show()                              # Hiển thị cửa sổ biểu đồ

# Relative Standard Deviation (RSD)
def check_rsd(data: list | pd.DataFrame):
    """
    Kiểm tra độ nhiễu (noise/outlier) của dữ liệu theo quy tắc phân phối chuẩn (68 - 95):
    - Vùng ± 1 STD: 'lệch 68'
    - Vùng ± 2 STD: 'lệch 95'
    - Vùng > ± 2 STD: 'quá lệch'
    """
    # 1. Kiểm tra và chuẩn hóa dữ liệu đầu vào
    if not isinstance(data, (list, pd.DataFrame)) or data is None:
        return False, "Lỗi: Dữ liệu đầu vào phải là list hoặc DataFrame!", None, None

    df = pd.DataFrame(data=data).copy()
    if df.empty or 'Category' not in df.columns or 'Revenue' not in df.columns:
        return False, "Lỗi: Dữ liệu phải chứa cột 'Category' và 'Revenue'!", None, None

    # Ép kiểu cột Revenue về dạng số
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')

    result_list = []
    summary_list = []

    # 2. Xử lý tính toán theo từng nhóm Category
    for category, g_df in df.groupby('Category', sort=False):
        g_df = g_df.copy()

        # Trung bình cộng của tất cả các điểm dữ liệu trong Category
        mean_val = g_df['Revenue'].mean()

        # Độ lệch giữa từng điểm Y(x) và trung bình cộng
        deviation = g_df['Revenue'] - mean_val

        # STD = Căn bậc hai của trung bình cộng của tổng bình phương độ lệch từng tháng
        std_val = np.sqrt((deviation ** 2).mean())

        # Ghi nhận các chỉ số thống kê vào bảng kết quả (làm tròn 2 chữ số thập phân)
        g_df['Mean'] = round(mean_val, 2)
        g_df['Deviation'] = round(deviation, 2)
        g_df['STD'] = round(std_val, 2)

        # 3. Phân loại từng điểm theo vùng ± 1 STD, ± 2 STD, > 2 STD
        abs_dev = deviation.abs()
        conditions = [
            abs_dev <= (1 * std_val),
            (abs_dev > (1 * std_val)) & (abs_dev <= (2 * std_val)),
            abs_dev > (2 * std_val)
        ]
        choices = ['deviation_68', 'deviation_95', 'over_deviation']

        g_df['Noise_Level'] = np.select(conditions, choices, default='không xác định')
        result_list.append(g_df)

        # 4. Tính tỷ lệ over_deviation so với tổng số dòng của từng Category
        total_rows = len(g_df)
        over_rows = (g_df['Noise_Level'] == 'over_deviation').sum()
        per_95_val = round((over_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0
        summary_list.append({
            'Category': category,
            'per_95': per_95_val
        })

    # Gộp kết quả của tất cả Category lại thành 1 DataFrame
    final_df = pd.concat(result_list, ignore_index=True)
    summary_df = pd.DataFrame(summary_list)
    
    return True, "", final_df, summary_df

# Column: Cần lấy cột nào để kiểm tra
# key: xác định các cột nào là khóa để viết trực tiếp line nào đang lỗi
def check_zero(data, column, key=None):
    errors = []
    if data is None or not column:
        return True, None

    # 1. Chuẩn hóa data về dạng list of dict
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return False, None
        
        data_list = data.to_dict(orient='records')
    elif isinstance(data, dict):
        data_list = [data]
    elif isinstance(data, list):
        data_list = data
    else:
        errors.append("Định dạng data không hợp lệ!")
        err_df = pd.DataFrame(errors)
        return False, err_df

    # 2. Chuyển tham số column từ list (hoặc str) về dict
    if isinstance(column, list):
        col_dict = dict.fromkeys(column)
    elif isinstance(column, str):
        col_dict = {column: None}
    elif isinstance(column, dict):
        col_dict = column
    else:
        errors.append("Tham số column phải là list, dict hoặc str!")
        err_df = pd.DataFrame(errors)
        return False, err_df

    # 3. Chuyển tham số key từ list (hoặc str) về dict
    full_col = True
    if key is not None:
        if isinstance(key, list):
            key_dict = dict.fromkeys(key)
        elif isinstance(key, str):
            key_dict = {key: None}
        elif isinstance(key, dict):
            key_dict = key
        else:
            errors.append("Tham số  key phải là list, dict hoặc str!")
            err_df = pd.DataFrame(errors)
            return False, err_df

        # Kiểm tra column key tồn tại tất cả trong header data
        col_header = data_list[0].keys()
        for col_key in key_dict:
            if col_key not in col_header:
                full_col  = False
                break

    # 3. Duyệt qua từng dòng dữ liệu và kiểm tra các cột trong col_dict có giá trị = 0 không
    for line_idx, row in enumerate(data_list, start=1):
        if not isinstance(row, dict):
            continue

        # Lấy thông tin theo key xác nhận line đó có giá trị = 0   
        k_val = None 
        if full_col and key is not None:
            for k_col in key_dict.keys():
                k_val =  ( k_val + ' - ' if k_val is not None else '' ) + row[k_col]

        if k_val is not None:
            line_idx = k_val

        # Kiểm tra nhóm column đưa vào có trong row và check giá trị tại cột = 0
        for col in col_dict.keys():
            if col in row and row[col] is not None:
                val = row[col]
                # Kiểm tra nếu giá trị bằng 0 (hỗ trợ cả số 0, 0.0 và chuỗi '0', bỏ qua kiểu bool)
                if not isinstance(val, bool) and (val == 0 or (isinstance(val, str) and val.strip() in ('0', '0.0'))):
                    errors.append(f"Dòng {line_idx}: Cột '{col}' có giá trị = 0")

    if errors:
        err_df = pd.DataFrame(errors, columns=['Message'])
        return False, err_df

    return True, ""

def check_noise(data: list | pd.DataFrame):
    all_msg: pd.DataFrame = None

    # Check zero in data - Xử lý độ nhiễu khủng khiếp nhất
    sts, err_zero = check_zero(data= data, column= col_chk_zero, key= col_chk_key )

    if not sts and isinstance(err_zero, pd.DataFrame) and not err_zero.empty:
        new_df = pd.DataFrame({
            'Type': 'Error',
            'Message': err_zero['Message'] if 'Message' in err_zero.columns else err_zero.iloc[:, 0]
        })
        all_msg = pd.concat([all_msg, new_df], ignore_index=True)

    sts, msg, final_df, summary_df = check_rsd(data= data)
    if not sts:
        new_df = pd.DataFrame([{'Type': 'Noise_Error', 'Message': msg}])
        all_msg = pd.concat([all_msg, new_df], ignore_index=True)
    else:
        if isinstance(summary_df, (pd.DataFrame)):
            summary_df['per_95'] = pd.to_numeric(summary_df['per_95'], errors='coerce')
            noise_msgs = []
            for _, row in summary_df.iterrows():
                cat = row['Category']
                per_95 = row['per_95']
                if per_95 < 5:
                    msg_type = 'Success'
                    msg_text = f"Success {cat}: tỉ lệ nhiễu {per_95} - tỉ lệ nhiễu ổn định"
                elif per_95 <= 15:
                    msg_type = 'Warning'
                    msg_text = f"Warning {cat}: tỉ lệ nhiễu {per_95} - tỉ lệ nhiễu cần đánh giá"
                else:
                    msg_type = 'Error'
                    msg_text = f"Error {cat}: tỉ lệ nhiễu {per_95} - tỉ lệ nhiễu quá lớn"

                noise_msgs.append({'Type': msg_type, 'Message': msg_text})

            if noise_msgs:
                all_msg = pd.concat([all_msg, pd.DataFrame(noise_msgs)], ignore_index=True)

    return all_msg

if __name__ == "__main__":
    main()