import psycopg2
import pandas as pd
import numpy as np
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation
from holidays import CountryHoliday
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt
import optuna
import datetime
import json

def preprocessing_with_time(df, std_date_col_name, target_col_name, use_full_range = True, start_date = None,):
    """
    Params
        - df : station_id, std_date <- crtr_ymd, dd_charg_hr, 00H~23H 로 구성되어있음
        - std_date_col_name : crtr_ymd를 가져온 column name
        - target_col_name : target_var로 쓸 column name
        - use_full_range : 전체 기간 데이터를 보간하려고 하는 경우 True, 아닌 경우 False와 start_time 지정
        - start_time : str으로 지정 'yyyy-mm-dd'
    Returns: final_df
        - ds : datetime columns (시간대 기준일자 yyyymmdd hh)
        - y : target column(충전시간)
        - dd_charg_hr : 해당하는 일자의 총 충전시간
        - weekday : 요일 (숫자)
        - IsWeekend : 주말(토,일)
    """
    # 기준일자 컬럼을 datetime으로 만들기
    df[std_date_col_name] = pd.to_datetime(df[std_date_col_name])
    # 0~23시 충전시간 0으로 채우기
    df.fillna(0,inplace=True)
    # 충전소 일자별 집계
    df = df.groupby([std_date_col_name],as_index=False).agg(
        {'dd_charg_hr' : 'sum', 
         '00H' : 'sum', '01H' : 'sum','02H' : 'sum','03H' : 'sum','04H' : 'sum','05H' : 'sum','06H' : 'sum','07H' : 'sum',
         '08H' : 'sum', '09H' : 'sum','10H' : 'sum','11H' : 'sum','12H' : 'sum','13H' : 'sum','14H' : 'sum','15H' : 'sum',
         '16H' : 'sum', '17H' : 'sum','18H' : 'sum','19H' : 'sum','20H' : 'sum','21H' : 'sum','22H' : 'sum','23H' : 'sum',})
    # 집계 테이블 unpivot
    df = pd.melt(df, id_vars=[std_date_col_name, 'dd_charg_hr'], 
                    value_vars=[f'{i:02d}H' for i in range(24)], 
                    var_name='hour', value_name='charg_time')
    # 시간순서 정렬
    df = df.sort_values([std_date_col_name,'hour']).reset_index(drop=True)
    # YYYY-MM- DD HH:00:00 형식으로 만들기
    df['datetime'] = pd.to_datetime(df[std_date_col_name]) + pd.to_timedelta(df['hour'].str.replace('H','').astype(int),unit='h')
    
    # 보간 여부 결정
    if use_full_range == True : #보간을 할 경우
        min_date, max_date = df['datetime'].min(), df['datetime'].max()
        full_time_range = pd.date_range(start=min_date, end = max_date, freq='h')
        full_time_df = pd.DataFrame(full_time_range, columns=['datetime'])
        final_df = pd.merge(full_time_df, df[['datetime','charg_time']], on='datetime', how='left')
        final_df['charg_time'] = final_df['charg_time'].fillna(0)

    else: #보간을 하지 않을 경우
        final_df = df
    # 데이터 전체 기간 or 특정 일자부터 시작 결정
    if start_date is not None: # 시작일자를 지정한 경우
        final_df = final_df[final_df['datetime'] >= start_date].reset_index(drop=True)

    # Prophet에 필요한 형식으로 컬럼명을 변경
    final_df.rename(columns={'datetime': 'ds', target_col_name: 'y'}, inplace=True)
    
    # 주중/주말/시간/월 변수 추가
    final_df['Weekday'] = final_df['ds'].dt.weekday
    final_df['IsWeekend'] = final_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    final_df['Hour'] = final_df['ds'].dt.hour
    final_df['Month'] = final_df['ds'].dt.month

    # 자기 상관 변수 추가
    final_df['prev_day'] = final_df['y'].shift(24)
    final_df['prev_day'] = final_df['prev_day'].fillna(0)
    
    final_df['prev_week'] = final_df['y'].shift(168)
    final_df['prev_week'] = final_df['prev_week'].fillna(0)
    
    # fillna 결측값 처리 (1차 : 3개 기준, 2차 : 2개 기준 , 3차 : 3개 기준, 4차 : 0으로 채워넣음)
    final_df['y'] = final_df.groupby(['Month','Weekday','Hour'])['y'].transform(lambda x: x.fillna(x.mean())) # 월 X 요일 X 시간대의 평균값으로 대체
    final_df['y'] = final_df.groupby(['Weekday','Hour'])['y'].transform(lambda x: x.fillna(x.mean())) # 요일 X 시간대의 평균값으로 대체
    final_df['y'] = final_df.groupby(['Hour'])['y'].transform(lambda x: x.fillna(x.mean())) # 시간대의 평균값으로 대체
    final_df['y'] = final_df['y'].fillna(0) # 이 경우, 없는 시간대의 값은 충전이 일어나지 않은 것으로 간주하여 0으로 채움 
    
    return final_df

def generate_holidays(df, years=None):
    """
    한국의 공휴일을 추가하는 함수.
    Parameters:
        years (list): 공휴일을 포함할 연도 리스트 (기본값: None이면 데이터에서 자동으로 추출)
    Returns:
        holidays_df (DataFrame): 공휴일 데이터프레임
    """
    # 한국 공휴일을 가져오기
    if years is None:
        years = list(pd.to_datetime(df['ds']).dt.year.unique())

    # 한국 공휴일 가져오기
    kr_holidays = CountryHoliday('KR', years=years)  # holidays 라이브러리에서 공휴일을 추출
    holidays_df = pd.DataFrame(list(kr_holidays.items()), columns=['ds', 'holiday'])

    # `lower_window`와 `upper_window` 값을 동적으로 설정
    def set_windows(row):
        holidays_date = row['ds']
        if holidays_date.weekday() == 4:      # 금요일
            return pd.Series([-1, 2])         # 전날(목요일)부터 일요일까지
        elif holidays_date.weekday() == 5:    # 토요일
            return pd.Series([-1, 1])         # 전날(금요일)부터 일요일까지
        elif holidays_date.weekday() == 6:    # 일요일
            return pd.Series([-1, 0])         # 전날(토요일)부터 당일(일요일)까지
        elif holidays_date.weekday() == 0:    # 월요일
            return pd.Series([-2, 0])         # 토요일부터 당일(월요일)까지
        else:                                # 화요일 ~ 목요일: 당일만 적용
            return pd.Series([0, 0])
    
    holidays_df[['lower_window', 'upper_window']] = holidays_df.apply(set_windows, axis=1)
    return holidays_df

def get_charger_count(busi_id, sta_id, chrgr_typ):
    """
    충전기 개수를 구하는 함수
    
    Parameters:
        busi_id (str): 충전사업자 ID
        sta_id (str): 충전소 ID
        chrgr_typ (str): 충전기 유형 (fast/slow)
    Returns:
        int or str: 충전기 수 또는 오류 메시지
    """
    
    df = pd.read_json('./EVBP_deploy/ev_charge_station_info.json')
    value = df.loc[(df['busi_id'] == busi_id)&(df['sta_id'] == sta_id)&(df['chrgr_typ'] == chrgr_typ),'chrgr_cnt'].iloc[0]
    return int(value)

    # 조건에 맞는 데이터가 없는 경우 예외 발생
    raise ValueError(f"There is no station: busi_id={busi_id}, sta_id={sta_id}, chrgr_typ={chrgr_typ}")


def make_prophet_model(df, holiday_df=None, target_column='y', date_column='ds', 
                       changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, holidays_prior_scale=10.0, 
                       changepoint_range=0.8, n_changepoints=25, seasonality_mode='additive', yearly_seasonality=True, 
                       weekly_seasonality=True, daily_seasonality=False, feature_cols=None, add_seasonality=False, 
                       seasonality_dict=None, optimize=True, n_trials=30, n_jobs = 16, charger_count=1):
    """
    Prophet 모델을 학습하고, Optuna를 통해 하이퍼파라미터를 최적화하는 함수.
    
    Parameters:
        df (DataFrame): 'ds' (날짜), 'y' (값)을 포함하는 시계열 데이터
        target_column (str): 예측할 변수 (기본값: 'y')
        date_column (str): 날짜 컬럼명 (기본값: 'ds')
        changepoint_prior_scale (float): 추세 변화점 민감도
        seasonality_prior_scale (float): 계절성 민감도
        holidays_prior_scale (float): 공휴일 민감도
        changepoint_range (float): 추세 변화점의 범위
        n_changepoints (int): 추세 변화점 개수
        seasonality_mode (str): 계절성 모드 (additive 또는 multiplicative)
        yearly_seasonality (bool): 연간 계절성 사용 여부
        weekly_seasonality (bool): 주간 계절성 사용 여부
        daily_seasonality (bool): 일간 계절성 사용 여부
        feature_cols (list): Prophet 모델에 사용할 회귀 변수들 (기본값: None)
        add_seasonality (bool): 추가 계절성 여부 (기본값: False)
        seasonality_dict (dict): 계절성에 필요한 정보 (기본값: None)
        holiday_df (DataFrame or None): 공휴일 데이터프레임 (기본값: None)
        optimize (bool): Optuna 최적화를 수행할지 여부 (기본값: True)
        charger_count(int) : 충전소 내 충전기 개수(예측값의 상한성을 측정하기 위함*60)
    Returns:
        best_model (Prophet): 학습된 Prophet 모델
        best_params (dict): 최적의 하이퍼파라미터 값
        best_rmse (float): 최적화된 모델의 RMSE 값
    """
    
    # Optuna 최적화 함수
    def objective(trial):
        """
        Optuna 최적화 함수 - Prophet 모델 하이퍼파라미터 최적화
        """
        # global 없이 closure로 접근
        cap = 60 * charger_count
        # 하이퍼파라미터 탐색 공간 정의
        changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.1)
        seasonality_prior_scale = trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10.0)
        holidays_prior_scale = trial.suggest_loguniform('holidays_prior_scale', 0.01, 10.0)
        changepoint_range = trial.suggest_uniform('changepoint_range', 0.7, 0.95)
        n_changepoints = trial.suggest_int('n_changepoints', 2, 30)
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])
        weekly_seasonality = trial.suggest_categorical('weekly_seasonality', [True, False])
        daily_seasonality = trial.suggest_categorical('daily_seasonality', [True, False])
        # 최적화된 하이퍼파라미터로 Prophet 모델 학습
        model = Prophet(
            growth='linear',
            changepoint_prior_scale=changepoint_prior_scale, 
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_range=changepoint_range,
            n_changepoints=n_changepoints,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holiday_df,
            #backend='pystan'
        )
        # 상/하한선 설정
        df['cap'] = cap
        df['floor'] = 0         # floor 설정(하한선 설정)
        if feature_cols:
            for col in feature_cols:
                model.add_regressor(col)
        
        if add_seasonality and seasonality_dict:
            model.add_seasonality(
                name=seasonality_dict['name'], 
                period=seasonality_dict['period'], 
                fourier_order=seasonality_dict['fourier_order']
            )
        
        # 모델 학습
        model.fit(df)
        
        # 예측
        forecast = model.predict(df)
        
        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(df[target_column], forecast['yhat']))
        return rmse

    # 'prophet' 및 'cmdstanpy' 로거의 로그 레벨을 WARNING으로 설정
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    # 로그 레벨을 WARNING으로 설정하여 INFO 수준의 로그를 숨깁니다.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    #상/하한선 설정
    df['floor'] = 0 #하한선 0 설정
    if charger_count >= 1:
        cap = 60 * charger_count
    else :
        raise ValueError("Cannot receive charger count")
    df['cap'] = cap
    
    # Optuna 최적화 실행
    if optimize:
        study = optuna.create_study(direction='minimize')  # RMSE를 최소화하려는 목표
        study.optimize(objective, n_trials=n_trials, n_jobs = n_jobs)
        
        # 최적의 하이퍼파라미터로 모델 학습
        best_params = study.best_params
        best_model = Prophet(growth='linear',
            changepoint_prior_scale=best_params['changepoint_prior_scale'], 
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            holidays_prior_scale=best_params['holidays_prior_scale'],
            changepoint_range=best_params['changepoint_range'],
            n_changepoints=best_params['n_changepoints'],
            seasonality_mode=best_params['seasonality_mode'],
            yearly_seasonality=best_params['yearly_seasonality'],
            weekly_seasonality=best_params['weekly_seasonality'],
            daily_seasonality=best_params['daily_seasonality'],
            holidays=holiday_df,
            #backend='pystan'
        )
        
        if feature_cols:
            for col in feature_cols:
                best_model.add_regressor(col)
        
        if add_seasonality and seasonality_dict:
            best_model.add_seasonality(
                name=seasonality_dict['name'], 
                period=seasonality_dict['period'], 
                fourier_order=seasonality_dict['fourier_order']
            )
        
        best_model.fit(df)
        
        # 최적화된 모델에서 예측
        forecast = best_model.predict(df)

        # # 교차검증 및 성능 평가
        # cv_results = cross_validation(best_model, initial='365 days', period='1 hours', horizon='60 days')
        # metrics = performance_metrics(cv_results)
        
        #검증지표
        best_rmse = np.sqrt(mean_squared_error(df[target_column], forecast['yhat']))
        best_mae = mean_absolute_error(df[target_column], forecast['yhat'])
        best_r2 = r2_score(df[target_column], forecast['yhat'])
        return best_model, best_params, best_rmse, best_mae, best_r2, #metrics  # 최적화된 모델, 파라미터, RMSE 값 반환

    else:
        # 최적화하지 않으면 기본 모델을 학습하고 반환
        model = Prophet(growth='logistic',
            changepoint_prior_scale=changepoint_prior_scale, 
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_range=changepoint_range,
            n_changepoints=n_changepoints,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holiday_df,
            #backend='pystan'
        )
        
        if feature_cols:
            for col in feature_cols:
                model.add_regressor(col)
        
        if add_seasonality and seasonality_dict:
            model.add_seasonality(
                name=seasonality_dict['name'], 
                period=seasonality_dict['period'], 
                fourier_order=seasonality_dict['fourier_order']
            )
        
        model.fit(df)
        
        return model  # 최적화 없이 기본 모델 반환
        
def forecast_prophet_model(model, df, feature_cols=None, predict_periods=None, predict_freq='H', charger_count = 1):
    """
    학습된 Prophet 모델을 사용하여 예측하는 함수.
    
    Parameters:
        model (Prophet): 학습된 Prophet 모델
        df (DataFrame): 예측에 사용할 데이터프레임 ('ds' 컬럼 포함)
        feature_cols (list): Prophet 모델에 사용할 회귀 변수들 (기본값: None)
        predict_periods (int): 예측할 기간 (기본값: 30일)
        predict_freq (str): 예측 주기 (기본값: 'H', 시간 단위)
    
    Returns:
        forecast (DataFrame): 예측 결과
    """
    # 데이터의 마지막 날짜
    last_date = model.history['ds'].max()
    
    # 현재 날짜 (실행 연도)
    current_date = pd.to_datetime(datetime.datetime.now())
    
    # 실행 연도의 마지막 날짜 (12월 31일)
    last_day_of_year = pd.to_datetime(f'{current_date.year}-12-31 23')

    # 예측할 기간 계산 (실행 연도의 12월 31일까지)
    predict_periods = int((last_day_of_year - last_date).total_seconds() / 3600)  # 예측 기간을 시간 단위로 변환 (1일 = 24시간)
    
    # 미래 데이터프레임 생성
    future = model.make_future_dataframe(periods=predict_periods, freq=predict_freq)
    # df와 future 데이터프레임을 left join하여 feature_cols 값 추가
    future = pd.merge(future,df, how='left', on='ds')
    # 결측치 채우기
    future['Weekday'] = future['ds'].dt.weekday
    future['IsWeekend'] = future['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    future['Hour'] = future['ds'].dt.hour
    future['Month'] = future['ds'].dt.month
    # lag feature
    future['prev_day'] = future['y'].shift(24).fillna(0)  #하루전 lag feature
    future['prev_week'] = future['y'].shift(168).fillna(0) #일주일전 lag feature
    future = future.drop(columns=['y'])
    # cap과 floor 적용
    if charger_count >= 1:
        cap = 60 * charger_count
    else:
        raise ValueError('Cannot Receive Charger Count')
    floor = 0  # 하한선 설정 (필요시 다른 값으로 설정)
    future['cap'] = cap
    future['floor'] = floor
    # 예측 수행
    forecast = model.predict(future)
    predict_result = forecast[forecast['ds'] > df['ds'].max()]
    return forecast, predict_result

# 최종 결과 DataFrame 만드는 함수
def make_predict_result_data(predict_result=None, busi_id=None, sta_id=None, charger_type=None) : 
    '''
    Parameter:
        predict_result : forecast_prophet_model에서 나온 predict_result 받아오기
        busi_id(str) : Data Import 할 때 받아온 busi_id
        sta_id(str)  : Data Import 할 때 받아온 sta_id
        charger_type(str)  : Data Import 할 때 받아온 charger_type (fast/slow)
    Returns:
        result_df(DF) : CRTR_YMD, BUSI_ID, STA_ID, CHARG_HR, REG_DT 
    '''
    result_df = predict_result
    result_df['CRTR_YMD'] = result_df['ds'].dt.date
    result_df['CRTR_HR'] = result_df['ds'].dt.hour.astype('str').str.zfill(2)
    result_df['BUSI_ID'] = busi_id
    result_df['STA_ID'] = sta_id
    result_df['CHARG_TYP'] = charger_type
    result_df['CHARG_HR'] = result_df['yhat']
    result_df['REG_DT'] = datetime.datetime.now().strftime('%Y-%m-%d')
    result_df = result_df[['CRTR_YMD','CRTR_HR','BUSI_ID','STA_ID','CHARG_TYP','CHARG_HR','REG_DT']].reset_index(drop=True)
    return result_df