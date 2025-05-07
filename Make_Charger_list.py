import psycopg2
import pandas as pd
import json
import argparse
import os
from sqlalchemy import create_engine

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

# 데이터베이스에서 데이터를 불러오는 함수
def load_charger_list_from_db(engine):
    '''
    충전소/충전기유형별 Master Table을 DataFrame으로 저장
    '''
    try:
        query = """
        SELECT 
            busi_id,
            sta_id,
            CASE 
                WHEN chrgr_typ_cd IN ('02', '08') THEN 'S'
                ELSE 'F'
            END AS chrgr_typ,
            COUNT(*) AS chrgr_cnt,
            STRING_AGG(chrgr_id, ',' ORDER BY chrgr_id) AS chrgr_list
        FROM 
            evbp_dm.chr_charger
        GROUP BY 
            busi_id,
            sta_id,
            CASE 
                WHEN chrgr_typ_cd IN ('02', '08') THEN 'S'
                ELSE 'F'
            END
        ORDER BY 
            busi_id, sta_id, chrgr_typ;
        ;
        """
        df = pd.read_sql_query(query, engine)
        print("Success!! DB Read Complete")
        return df
    except Exception as e:
        print(f" Error : DB READ Fail: {e}")
        return None

# JSON 업데이트 함수
def make_or_update_json(df, json_path):
    new_data = df.to_dict(orient='records')
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        df_old = pd.DataFrame(old_data)

        # 우선 기준 키 설정
        key_cols = ["busi_id", "sta_id", "chrgr_typ"]

        # 기존 데이터를 기준으로 갱신 (중복 키는 새로운 값으로 덮어쓰기)
        df_combined = pd.concat([df_old, df], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=key_cols, keep='last')

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(df_combined.to_dict(orient='records'), f, ensure_ascii=False, indent=4)
        print("Success! JSON Update&Merge")

    else:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
        print("Success! Make New File")

# 실행 함수
def main(config_path, output_json_path):
    '''
    run this module : python3 Make_Charger_list.py --config ./db_conn_info.json --output ./charger_type_list.json
    '''
    with open(config_path, 'r', encoding='utf-8') as f:
        conn_info = json.load(f)

    engine = get_db_engine(conn_info)
    df = load_charger_list_from_db(engine)

    if df is not None:
        make_or_update_json(df, output_json_path)

# 명령행 인자 처리
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PostgreSQL 데이터를 JSON으로 저장 또는 업데이트합니다.")
    parser.add_argument("--config", type=str, default="./db_conn_info.json", help="DB 연결 정보를 담은 JSON 파일 경로")
    parser.add_argument("--output", type=str, default="./ev_charge_station_info.json", help="출력할 JSON 파일 경로")

    args = parser.parse_args()
    main(args.config, args.output)
