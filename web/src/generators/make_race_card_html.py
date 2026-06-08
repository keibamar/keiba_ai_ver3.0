import sys
import os
from datetime import date, timedelta, datetime

# pycache を生成しない
sys.dont_write_bytecode = True

# web/src を import パスに追加
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# libs を追加
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

# src/Datasets を import パスに追加
DATAS_PATH = os.path.join(PROJECT_ROOT, "src/Datasets")
if DATAS_PATH not in sys.path:
    sys.path.insert(0, DATAS_PATH)

# src/RacePrediction を import パスに追加
RACEPRED_PATH = os.path.join(PROJECT_ROOT, "src/RacePrediction")
if RACEPRED_PATH not in sys.path:
    sys.path.insert(0, RACEPRED_PATH)

from generators.date_index import add_race_day    
from generators.daily_index import make_daily_index_page
from generators.race_pages import make_daily_race_card_html

# libs 配下のモジュールは sys.path に追加後にインポート
import analysis_race_info
import make_time_id_list
import race_card
import daily_race_results
import calc_returns

def make_html_prev_day(race_day = date.today() + timedelta(days=1)):
    time_id_list = make_time_id_list.get_time_id_list(race_day)
    if any(time_id_list):
        # 全開催日インデックスページを更新
        add_race_day(race_day)
        # 日付別インデックスページを生成
        make_daily_index_page(race_day)
        # 過去一週間のindexを再作成（リンクの生成）
        for delta_day in range(1, 8):
            past_day = race_day - timedelta(days=delta_day)
            make_daily_index_page(past_day)

        while(any(time_id_list)):
            race_id = time_id_list[0][1]
            print("make_race_card:", race_id)
            try:
                # 予想の更新
                race_card_df, race_info_df = race_card.make_race_card(race_id)
                # csvファイルで出力
                race_card.save_race_cards(race_card_df, race_day, race_id)
                race_card.save_race_info_df(race_info_df, race_day, race_id)
                analysis_race_info.update_horse_name_id_map(race_card_df)
            except :
                print("Make RaceCard Error:", race_id)
            (time_id_list).pop(0)

        # 各レースのHTMLを生成
        make_daily_race_card_html(race_day)
        
        # 日付別インデックスページを生成
        make_daily_index_page(race_day)
        # (リンク生成のため暫定で二度生成)各レースのHTMLを生成
        make_daily_race_card_html(race_day)

def update_daily_html(race_day = date.today()):
    time_id_list = make_time_id_list.get_time_id_list(race_day)
    while(any(time_id_list)):
        race_id = time_id_list[0][1]
        print("update_results_df:",race_id)
        # レース結果の取得
        try:
            results_df = daily_race_results.get_each_race_results(race_id)
            if not results_df.empty:
                daily_race_results.save_each_race_result_csv(race_id, results_df)
        except :
            print("Miss Make Results : ", race_id)
        # 配当結果の取得
        try :
            df_return = calc_returns.get_race_return(race_id)
            if not df_return.empty:
                calc_returns.save_each_race_return_csv(race_id, df_return)
        except :
            print("Miss Make Returns : ", race_id)
        time_id_list.pop(0)
    make_daily_race_card_html(race_day)

# 使用例
if __name__ == "__main__":
    make_html_prev_day()