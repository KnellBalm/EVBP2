# 내장
import pandas as pd
import numpy as np
from prophet import Prophet
from holidays import CountryHoliday
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import datetime
import argparse
import logging
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #FutureWarning 제거

# 모듈
from Charge_History_Data_Control import *
from Demand_Predict_Modeling import *

############################ Logger ########################
# 로거 객체 생성
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 포맷 정의
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

# RotatingFileHandler: 5MB 넘으면 자동 순환 (최대 3개 보존)
file_handler = RotatingFileHandler(
    "predict_model_running.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
file_handler.setFormatter(formatter)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 기존 핸들러 제거 후 새 핸들러 추가 (중복 방지)
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(file_handler)
logger.addHandler(console_handler)
##########################################################

# 모델 실행 함수 -- busi_id, sta_id, chrgr_typ, month 를 받게끔 조정
def run_model(busi_id=None,sta_id=None, chrgr_typ=None, month=None, n_trials=50, n_jobs = 6):

    if month is not None : 
        use_info = load_data_from_db(busi_id=busi_id, sta_id=sta_id, chrgr_typ=chrgr_typ, month=month) # 전월말 데이터까지 불러오기
    else :
        use_info = load_data_from_db(busi_id=busi_id, sta_id=sta_id, chrgr_typ=chrgr_typ) # 데이터 전체 불러오기

    ### 전처리 > 공휴일 > 충전기 수 ###    
    use_df = preprocessing_with_time(use_info) # 데이터 전처리   
    holidays_df = generate_holidays(use_df) # 공휴일 DataFrame 생성
    charger_cnt = get_charger_count(busi_id=busi_id, sta_id=sta_id, chrgr_typ=chrgr_typ)

    logger.info("========= Finish Preprocessing =========")
    ### Prophet 모델 생성 > optuna최적화 > Best model out 
    best_model = make_prophet_model(use_df, holiday_df = holidays_df,
                                    charger_count = charger_cnt, n_trials=n_trials, n_jobs = n_jobs) # 모델 학습 및 최적화
    logger.info("========= Finish Modeling =========")
    ### Prophet 모델 예측
    forecast, predict_result = forecast_prophet_model(best_model, use_df, charger_count=charger_cnt, predict_freq='H') 
    return forecast, predict_result # 예측 결과 반환 
    logger.info("========= Finish Predict =========")
    
if __name__ == "__main__": 
    logger.info("========= Predict Model Start =========")
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description="예측모델 실행을 위한 스크립트") 

    # 명령줄 인자 추가
    ## 필수 인자
    parser.add_argument("--busi_id", type=str, help="충전사업자", required=True) 
    parser.add_argument("--sta_id", type=str, help="충전소ID", required=True)
    parser.add_argument("--chrgr_typ", type=str, help="충전기타입", required=True) 
    parser.add_argument("--month", type=int, help="조회희망월", required=True) 
    parser.add_argument("--table_name", type=str, help="예측결과 테이블명",required=True)

    ## 선택 인자
    parser.add_argument("--schema", type=str, help="예측결과 테이블 스키마명", default="evbp_dm") # 기본값은 evbp_dm
    parser.add_argument("--n_trials", type=int, help="최적화 횟수", default=50)  # 기본값은 50
    parser.add_argument("--n_jobs", type=int, help="최적화 시 사용할 CPU 갯수", default=6)  # 기본값은 6
    
    # 인자 파싱
    args = parser.parse_args()
    
    logger.info(f"Input Params: busi_id={args.busi_id}, sta_id={args.sta_id}, chrgr_typ={args.chrgr_typ}, month={args.month}")
    # run_model 함수 호출
    forecast, predict_result = run_model(
        busi_id=args.busi_id, 
        sta_id=args.sta_id,
        chrgr_typ=args.chrgr_typ,
        month=args.month,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )
    logger.info("========= Finish Run Model =========")
    # 결과 데이터 생성 후 DB Insert
    result_df = make_predict_result_data(
        predict_result = predict_result,
        busi_id=args.busi_id, 
        sta_id=args.sta_id,
        charger_type=args.chrgr_typ,
    )
    logger.info(f"Finish Make Result. Start DB Insert. Target Table : {args.schema}.{args.table_name}")

    # 결과 데이터 생성 후 DB Insert
    insert_dataframe_to_db(
        result_df,
        table_name = args.table_name,
        schema= args.schema
    )
    logger.info("========= Finish Module =========")
    logger.info(f"End process: busi_id={args.busi_id}, sta_id={args.sta_id}, chrgr_typ={args.chrgr_typ}, month={args.month}")