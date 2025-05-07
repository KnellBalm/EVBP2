import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import json
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

# Read Config File
with open('./db_conn_info.json', 'r') as f:
    conn_info = json.load(f)

# PostgreSQL 연결을 위한 엔진 생성 함수
def get_db_engine(conn_info):
    conn_dict = {
        'host': conn_info['host'],
        'port': conn_info['port'],
        'dbname': conn_info['dbname'],
        'user': conn_info['user'],
        'password': conn_info['password']
    }
    connection_string = f"postgresql+psycopg2://{conn_dict['user']}:{conn_dict['password']}@{conn_dict['host']}:{conn_dict['port']}/{conn_dict['dbname']}"
    return create_engine(connection_string)

# 데이터 불러올 때 충전유형에 맞는 충전기 번호만 취득하는 함수
def get_charger_id_list(busi_id,sta_id,chrgr_typ):
    df = pd.read_json('./ev_charge_station_info.json')
        # 조건 필터링
    mask = (
        (df['busi_id'] == busi_id) &
        (df['sta_id'] == sta_id) &
        (df['chrgr_typ'] == chrgr_typ)
    )
    filtered = df.loc[mask, 'chrgr_list']
    if not filtered.empty:
        # 정상일 경우 리스트 문자열 처리
        charger_list = ",".join([f"'{x.strip()}'" for x in filtered.iloc[-1].split(',')])
        return charger_list
    else:
        # 없을 경우 기본값 반환 or 예외 처리
        logger.warning(f"[WARN] No data found for: {busi_id}, {sta_id}, {chrgr_typ}")
        return ""  # 또는 None, raise Exception 등
    charger_list = df.loc[(df['busi_id'] == busi_id)&(df['sta_id'] == sta_id)&(df['chrgr_typ'] == chrgr_typ),'chrgr_list'].iloc[-1]
    charger_list = ",".join([f"'{x.strip()}'" for x in charger_list.split(',')])
    return charger_list

# 1. 데이터베이스에서 데이터를 불러오는 함수
def load_data_from_db(busi_id,sta_id,chrgr_typ, month=None):
    '''
    Parameter : 
        busi_id(str): 충전사업자ID
        sta_id(str): 충전소ID
        chrgr_typ(str): 충전기 유형('fast'/'slow')
        month(int) : 활용 데이터 범위 한정, 입력 월의 말일까지로 데이터 입력하게 설정됨
    Returns:
        df(DataFrame) : 결과 DataFrame
    '''
    charger_list = get_charger_id_list(busi_id,sta_id,chrgr_typ)
    
    # 말일 구하는 함수
    def get_last_day_of_month(month, year=2025):
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        return last_day.day
    
    ### 월받아서 쿼리에 전월 말일 23시까지 데이터만 땡겨오는 것
    try:
        engine = get_db_engine(conn_info)
        if month == None: # 월을 입력하지 않았을때 > Target의 Last Update까지 전부 사용
            query = f"""
                select 
                     busi_id || sta_id as station_id, crtr_ymd as std_date, dd_charg_hr,
                     h00,h01,h02,h03,h04,h05,h06,h07,h08,h09,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20,h21,h22,h23
                from evbp_dm.chr_charg_use_info 
                where 1=1 
                and busi_id = '{busi_id}' 
                and sta_id = '{sta_id}' 
                and chrgr_id in ({charger_list})
                ; 
                """
        else: #조회를 시도하려는 월의 전월 말일 23시까지 데이터 가져오도록 설정
            now_date = datetime.now() # 오늘 날짜
            now_year = now_date.year # 오늘의 연도
            input_month = month
            
            if input_month == 1:
                last_month = 12
                target_year = now_year - 1
            else:
                last_month = input_month - 1
                target_year = now_year

            last_day = get_last_day_of_month(last_month, year=target_year)
            limit_date = f'{now_year}{last_month:02d}{last_day}'
            
            query = f"""
            select 
                 busi_id || sta_id as station_id, crtr_ymd as std_date, dd_charg_hr, 
                 h00,h01,h02,h03,h04,h05,h06,h07,h08,h09,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20,h21,h22,h23
            from evbp_dm.chr_charg_use_info 
            where 1=1 
            and busi_id = '{busi_id}' 
            and sta_id = '{sta_id}' 
            and chrgr_id in ({charger_list})
            and crtr_ymd <= '{limit_date}'; 
            """
        df = pd.read_sql_query(query, engine)
        logger.info("Data successfully loaded from the database!")
        return df
    except Exception as e:
        logger.warning(f"Error loading data from the database: {e}")
        return None

# 2. DataFrame을 PostgreSQL 테이블에 INSERT하는 함수
def insert_dataframe_to_db(df, table_name, schema="evbp_dm"):
    try:
        engine = get_db_engine(conn_info)
        df.to_sql(table_name, engine, schema=schema, if_exists='append', index=False)  # 데이터를 추가
        logger.info(f"{len(df)} rows Data successfully inserted into {table_name}!")
    except Exception as e:
        logger.warning(f"Error inserting DataFrame into {table_name}: {e}")

# 3. DataFrame을 기준으로 PostgreSQL 데이터 UPDATE하는 함수 -- 필요없음
# def update_db_with_dataframe(df, table_name, condition_col, update_col):
#     conn = psycopg2.connect(
#         host = conn_info['host'],
#         port = conn_info['port'],
#         dbname = conn_info['dbname'],
#         user = conn_info['user'],
#         password = conn_info['password']
#     )
#     cursor = conn.cursor()
    
#     try:
#         for index, row in df.iterrows():
#             # 예시: 특정 조건을 만족하는 행을 업데이트
#             update_query = f"""
#             UPDATE {table_name}
#             SET {update_col} = %s
#             WHERE {condition_col} = %s;
#             """
#             cursor.execute(update_query, (row[update_col], row[condition_col]))  # 값을 튜플로 전달
#         conn.commit()  # 변경 사항 커밋
#         logger.info(f"Successfully updated {len(df)} rows in {table_name}!")
#     except Exception as e:
#         conn.rollback()  # 예외 발생 시 롤백
#         logger.warning(f"Error updating {table_name}: {e}")
#     finally:
#         cursor.close()
#         conn.close()
