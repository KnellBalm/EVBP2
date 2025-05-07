# 내장
import pandas as pd
import numpy as np
from prophet import Prophet
from holidays import CountryHoliday
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import datetime
import argparse

# 모듈
from Charge_History_Data_Control import *
from Demand_Predict_Modeling import *

# 모델 실행 함수 -- busi_id, sta_id, chrgr_typ, month 를 받게끔 조정
def run_model(busi_id=None,sta_id=None, chrgr_typ=None, n_trials=50, n_jobs = 16):
    
    busi_id, sta_id, chrgr_type = busi_id, sta_id, chrgr_type
    use_info = load_data_from_db(busi_id=busi_id, sta_id=sta_id, chrgr_typ=None) # 데이터 불러오기
    use_df = preprocessing_with_time(use_info, std_date_col_name=std_date_col_name, target_col_name=target_col_name, use_full_range=use_full_range, start_date=start_date) # 데이터 전처리   
    holidays_df = generate_holidays(use_df) # 공휴일 DataFrame 생성
    charger_cnt = get_charger_count(busi_id,sta_id)
    best_model, best_params, best_rmse, best_mae, best_r2, metrics = make_prophet_model(
        use_df, target_column='y', date_column='ds', feature_cols=['prev_day','prev_week'], 
        holiday_df=holidays_df, optimize=True, n_trials=n_trials, n_jobs = n_jobs, charger_count=charger_cnt ) # 모델 학습 및 최적화
    forecast, predict_result = forecast_prophet_model(best_model, use_df, feature_cols=['prev_day','prev_week'], predict_freq='H', charger_count=charger_cnt) # Prophet 모델 예측
    return forecast, predict_result # 예측 결과 반환 (또는 필요에 따라 다른 작업 추가 가능)

if __name__ == "__main__": 
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="예측모델 실행을 위한 스크립트") 

    # 명령줄 인자 추가
    parser.add_argument("--busi_id", type=str, help="충전사업자", required=True)  # 필수 인자
    parser.add_argument("--sta_id", type=str, help="충전소ID", required=True)  # 필수 인자
    parser.add_argument("--chrgr_typ", type=str, help="충전기타입", required=True)  # 필수 인자
    parser.add_argument("--std_date_col_name", type=str, default="std_date", help="표준 날짜 컬럼 이름")  # 선택적 인자, 기본값은 "std_date"
    parser.add_argument("--target_col_name", type=str, default="charg_time", help="목표 컬럼 이름")  # 선택적 인자, 기본값은 "charg_time"
    parser.add_argument("--use_full_range", action='store_true', default = True, help="전체 범위 데이터를 사용할지 여부")  # 선택적 플래그, 기본값은 True
    parser.add_argument("--feature_cols", type=str, nargs='+', help="특징 컬럼들", default=['prev_day','prev_week'])  # 선택적 인자, 기본값은 ['prev_day','prev_week']
    parser.add_argument("--n_trials", type=int, nargs='+', help="최적화 횟수", default=50)  # 선택적 인자, 기본값은 100
    parser.add_argument("--n_jobs", type=int, nargs='+', help="최적화 시 사용할 CPU 갯수", default=6)  # 선택적 인자, 기본값은 16
    
    # 인자 파싱
    args = parser.parse_args()
    
    # run_model 함수 호출
    forecast, predict_result = run_model(
        run_for_one_station
        busi_id=args.busi_id, 
        sta_id=args.sta_id,
        chrgr_typ=args.chrgr_typ,
        
)
    # 결과 데이터 생성 후 DB Insert
    result_df = make_predict_result_data(
        busi_id=args.busi_id, 
        sta_id=args.sta_id,
        table_name = args.table_name
    )
    
