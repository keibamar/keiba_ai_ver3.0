import os
import sys

import pandas as pd
import datetime
from datetime import date, timedelta
from time import sleep
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\web\src")
import name_header
import scraping
import get_race_id

def save_each_race_result_csv(race_id, df_results):
    """各レースのレース結果を保存
        Args:
            race_id(str) : race_id
            df_results(pd.DataFrame) : 配当結果
    """
    try:
        place_id = int(race_id[4:6])
        year = int(race_id[0:4])
        folder_path = name_header.DATA_PATH + "RaceResults/" +name_header.PLACE_LIST[place_id - 1] + "/" + str(year)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        csv_data_path = folder_path + "//" + str(race_id) + ".csv"
        df_results.to_csv(csv_data_path)
    except Exception as e:
        print(e)

def save_day_race_result_each(race_day = date.today()):
    """指定日のレース結果を保存
    """ 
    # 今日のid_listを取得
    race_id_list = get_race_id.get_daily_id(race_day = race_day)
    if not race_id_list:
        return
    for race_id in race_id_list:
        results_df = scraping.scrape_day_race_results(race_id)
        if not results_df.empty:
            # 各レースの配当結果を保存
            save_each_race_result_csv(race_id, results_df)
        else:   
            print("not_race_result:", race_id)

def get_each_race_results(race_id):
    results_df = scraping.scrape_day_race_results(race_id)
    if not results_df.empty:
        # 各レースの配当結果を保存
        return results_df
    else :
        return pd.DataFrame()
