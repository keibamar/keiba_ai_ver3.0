import os
import re
import sys

from datetime import date, timedelta
import pandas as pd
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header

import make_text
import race_card

def save_time_id_list(race_day = date.today(), time_id_list = list()):
    """time_id_listをCSVファイルで保存
        Args:
            race_day(date) : レース開催日(初期値:今日)
            time_id_list(list) : レース開催日の[race_time, race_id]のリスト
    """
    try:
        str_day = race_day.strftime("%Y%m%d")
        folder_path = name_header.TEXT_PATH + "race_calendar/race_time_id_list"
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        if any(time_id_list):
            # DataFrameを作成
            time_id_list_df = pd.DataFrame(time_id_list, columns=["race_time", "race_id", "race_name"])
            time_id_list_df.to_csv(folder_path + "//" + str(str_day) + ".csv")
    except Exception as e:
        make_time_id_list_error(e)

def get_time_id_list(race_day = date.today()):
    """ time_id_listを取得
        Args:
            race_day(date) : レース開催日(初期値:今日)
        Returns:
            time_id_list(list) : レース開催日の[race_time, race_id]のリスト
    """
    time_id_list = []
    try:
        # CSVファイルからデータを読み込む
        str_day = race_day.strftime("%Y%m%d")
        folder_path = name_header.TEXT_PATH + "race_calendar/race_time_id_list"
        filename = folder_path + "//" + str(str_day) + ".csv"
        # ファイルの存在を確認
        if os.path.exists(filename):
            df = pd.read_csv(filename, dtype=str)
            # データフレームをリストに変換
            time_id_list = df[["race_time", "race_id"]].values.tolist()
        return time_id_list
    except Exception as e:
        make_time_id_list_error(e)
        return time_id_list

def make_time_id_list_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def get_race_time(race_info):
    """ 発走時間に取得
        Args:
            race_info(list) : レース情報
        Returns:
            race_time(str) : 発走時間
    """
    race_time =  str(race_info[1]) + str(race_info[2][0]) + str(race_info[2][1])
    race_time = re.findall(r'\d+', race_time)  # 文字列から数字にマッチするものをリストとして取得
    race_time = ''.join(race_time)
    return race_time

def get_race_name(race_info):
    """ 発走時間に取得
        Args:
            race_info(list) : レース情報
        Returns:
            race_name(str) : レース名
    """
    race_name =  str(race_info[0])
    return race_name

def sort_time(time_id_list):
    """発走時間リストを発走時間順にソート(コース順から発走時間順へソート)
        Args:
            time_id_list(list) : [race_time, race_id]
        Returns:
            time_id_list(list) : ソートされたtime_id_list
    """
    for i in range(len(time_id_list)):
        for l in range(i + 1, len(time_id_list)):
            if (int(time_id_list[i][0]) > int(time_id_list[l][0])):
               temp = time_id_list[i]
               time_id_list[i] = time_id_list[l]
               time_id_list[l] = temp
    
    return time_id_list


def make_time_id_list(race_day = date.today()):
    """time_id_listの作成
        Args:
            race_day(date) : レース開催日(初期値:今日)
        Returns:
            time_id_list(list) : レース開催日の[race_time, race_id]のリスト
    """
    print(race_day)
    time_id_list = []
    race_id_list = get_race_id.get_daily_id(race_day = race_day)
    for race_id in race_id_list:
        race_info = race_card.get_race_info(race_id)
        if any(race_info):
            race_time = get_race_time(race_info)
            race_name = get_race_name(race_info)
            time_id_list.append([race_time, race_id, race_name])
    
    time_id_list = sort_time(time_id_list)
    print(time_id_list)

    return time_id_list

def extract_race_info(text):
    # 地名＋レース番号（例: "東京1R"）とレース名（例: "2歳未勝利"）の抽出
    race_info_match = re.search(r'(\S+?\d+R)\s+(.*?)\s+\d{2}:\d{2}', text)
    race_code = race_info_match.group(1) if race_info_match else None
    race_name = race_info_match.group(2) if race_info_match else None

    # 発送時刻の抽出（hhmm形式）
    time_match = re.search(r'(\d{2}):(\d{2})発走', text)
    time_str = f"{time_match.group(1)}{time_match.group(2)}" if time_match else None

    return race_code, race_name, time_str

def make_time_id_list_from_prediction_txt(race_day):
    print(race_day)
    time_id_list = []
    race_id_list = get_race_id.get_daily_id(race_day = race_day)
    formatted_date = race_day.strftime("%Y%m%d")
    for race_id in race_id_list:
        txt_path = "../../texts/race_prediction/" + str(formatted_date) + "/" + str(race_id) +  ".txt"
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"ファイルが見つかりませんでした: {txt_path}")
            continue
        race_code, race_name, race_time = extract_race_info(text)
        if race_name is not None and race_time is not None:
            time_id_list.append([race_time, race_id, race_name])
    print(time_id_list)
    time_id_list = sort_time(time_id_list)
    print(time_id_list)

    return time_id_list

# 指定日から今日までの日付を順に処理する関数
def run_from_date_to_today(start_date):
    today = date.today()
    race_day = start_date

    while race_day <= today:
        print("Execute:" + str(race_day))
        time_id_list = make_time_id_list_from_prediction_txt(race_day)
        if any(time_id_list):
            save_time_id_list(race_day, time_id_list)
        race_day += timedelta(days=1)


if __name__ == '__main__':

    base_day = date.today()
    # 先、1週間分の開催日を取得
    for i in range(7):
        race_day = base_day + timedelta(days = (7 - i))
        time_id_list = make_time_id_list(race_day)
        if any(time_id_list):
            save_time_id_list(race_day, time_id_list)