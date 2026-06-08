import argparse
import os
import sys

from datetime import date, timedelta, datetime
import pandas as pd
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import scraping
import string_format
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import horse_peds
import past_performance

import day_race_prediction

def make_race_card_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def save_race_cards(race_card_df, race_day, race_id):
    """出馬表を保存
        Args:
            race_card_df(pd.DataFrame) : 出馬表データ
            race_day(date) : レース開催日
            race_id(int) : race_id
    """
    try:
        str_day = race_day.strftime("%Y%m%d")
        folder_path = name_header.DATA_PATH + "RaceCards//" + str_day
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        race_card_df.to_csv(folder_path + "//" + str(race_id) + ".csv")
    except Exception as e:
        make_race_card_error(e)

def save_race_info_df(race_info_df, race_day, race_id):
    """レース情報データセットの保存
        Args:
            race_info_df(pd.DataFrame) : レース情報データ
            race_day(date) : レース開催日
            race_id(int) : race_id
    """
    try:
        str_day = race_day.strftime("%Y%m%d")
        year = str_day[:4]
        place_id = int(str(race_id)[4] + str(race_id)[5])
        folder_path = name_header.DATA_PATH + "RaceInfo//" + name_header.PLACE_LIST[place_id - 1] + "//" + year  + "//"
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        race_info_df.to_csv(folder_path + "//" + str(race_id) + ".csv")
    except Exception as e:
        make_race_card_error(e)

def get_race_info(race_id):
    """レース情報(レース名・発走時刻)を取得
        Args:
            race_id(int) : race_id
        Returns:
            race_info(list) : レース情報
    """
    race_info, race_info_df, shutubahyou_df = scraping.scrape_race_card(race_id)
    return race_info

def get_race_cards(race_day, race_id):
    """出馬表を取得
        Args:
            race_day(date) : レース開催日
            race_id(int) : race_id
        Returns:
            race_card_df(pd.DataFrame) : 
    """
    try:
        str_day = race_day.strftime("%Y%m%d")
        folder_path = name_header.DATA_PATH + "RaceCards/" + str_day + "//" + str(race_id) + ".csv"
        if not os.path.isfile(folder_path):
            print("not pred_data")  
            race_card_df = pd.DataFrame()
        else:
            race_card_df = pd.read_csv(folder_path, index_col = 0)
        
        return race_card_df
    except Exception as e:
        make_race_card_error(e)
        return pd.DataFrame()

def extract_peds_for_display(horse_peds):
    """出馬表用に父，母，母父のみ抽出
        Args:
            horse_peds(pd.DataFrame) : 血統データ
        Returns:
            horse_peds_display(pd.DataFrame) : 表示用血統データ
    """
    horse_peds_display = horse_peds.iloc[:5]
    horse_peds_display = horse_peds_display.drop(horse_peds_display.index[[2,3]])
    horse_peds_display = horse_peds_display.rename(index = {0: '父'})
    horse_peds_display = horse_peds_display.rename(index = {1: '母'})
    horse_peds_display = horse_peds_display.rename(index = {4: '母父'})    
    return horse_peds_display

def make_race_card(race_id):
    """出馬表を出力
        Args:
            race_id (int) : race_id
        Returns:
            race_card_df(pd.DataFrame) : 出馬表データセット
    """
    # 出馬表を取得
    race_info, race_info_df, race_card_df = scraping.scrape_race_card(race_id)
    if race_info_df.empty or race_card_df.empty:
        print("Miss Make Race card")
        return pd.DataFrame()
    # 欠けている列を追加してデフォルト値を設定
    for col, default in {"weather": "-", "ground_state": "良"}.items():
        if col not in race_info_df.columns:
            race_info_df[col] = default
        else:
            race_info_df[col] = race_info_df[col].fillna(default)
    # 出走馬の過去成績と血統情報を取得
    horse_results = []
    horse_peds_df = pd.DataFrame()
    horse_ids = race_card_df.at[str(race_id),"horse_id"]
    for horse_id in horse_ids:
        # 過去成績を取得
        horse_result = past_performance.get_past_performance_dataset(horse_id)
        if horse_result.empty:
            print("make past performance data:", horse_id)
            horse_result = past_performance.make_past_performance_dataset_from_race_results(horse_id)
            past_performance.save_past_performance_dataset(horse_id,horse_result)
        horse_results.append(horse_result)
        # 血統情報を取得
        horse_ped = horse_peds.get_horse_peds_csv(horse_id)
        if horse_ped.empty:
            print("make horse_ped data:", horse_id)
            horse_ped = horse_peds.make_horse_peds_dataset(horse_id)
            horse_ped = horse_peds.delete_invalid_strings(horse_ped)
            horse_peds.save_horse_peds_dataset(horse_id, horse_ped)
        horse_peds_df = pd.concat([horse_peds_df,horse_ped], axis = 1)

    ###################  Todo : LightGBM用になっている  #############################      
    # 枠順、馬番を取得
    waku_df = pd.concat([race_card_df["枠"].reset_index(drop = True),race_card_df["馬番"].reset_index(drop = True)], axis = 1)
    # AI予想
    rank_df = day_race_prediction.rank_prediction(race_id, horse_ids, race_info_df, waku_df)
    #################################################################################   

    # 父，母，母父のみ抽出
    horse_peds_display = extract_peds_for_display(horse_peds_df) 
    # データセットの統合
    race_card_df = pd.concat([race_card_df.reset_index(drop = True), horse_peds_display.T.reset_index(drop = True)], axis = 1)
    race_card_df = pd.concat([race_card_df, rank_df.reset_index(drop=True)], axis = 1)
    
    return race_card_df, race_info_df

def daily_race_card(place_id = 0, race_day = date.today()):
    """出馬表を出力
        Args:
            place_id (int) : 開催コースid (初期値0=全コース)
            race_day(Date) : 日（初期値：今日）
    """
    # 今日のid_listを取得
    race_id_list = get_race_id.get_daily_id(place_id, race_day)
    print("MakeRaceCards")
    print(race_id_list)
    # レース情報を取得
    for race_id in tqdm(race_id_list):
        res = make_race_card(race_id)
        # make_race_card は (race_card_df, race_info_df) を返す場合と
        # 空の DataFrame を返す場合があるため両方に対応する
        if isinstance(res, tuple):
            race_card_df, race_info_df = res
        else:
            race_card_df = res
            race_info_df = pd.DataFrame()
        # csvファイルで出力
        save_race_cards(race_card_df, race_day, race_id)
        # レース情報もあれば保存
        try:
            save_race_info_df(race_info_df, race_day, race_id)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='指定日の予想を作成')
    parser.add_argument('number', nargs='?', default='A')

    args = parser.parse_args()
    race_day = date.today()

    # デフォルト(今日)
    if args.number == 'A':
        race_day = date.today()
        print("本日の予想結果を出力します。")
    # 日付指定（〇日後)
    elif string_format.is_valid_date_format(args.number):
         # 年、月、日を抽出
        year = int(args.number[:4])
        month = int(args.number[4:6])
        day = int(args.number[6:8])
        race_day = datetime(year, month, day)
        print(args.number, "の予想結果を出力します。")
    # 数字指定（〇日後/1週間以内)
    elif args.number.isdigit() and  0 < int(args.number) < 8:
        race_day = date.today() + timedelta(days = int(args.number))
        print(args.number, "日後の予想結果を出力します。")
    else:
        string_format.format_error(args.number)
        print("本日の予想結果を出力します。")

    daily_race_card(race_day = race_day)

    # race_id = "202405050511"
    # race_card_df = make_race_card(race_id)
    # # csvファイルで出力
    # save_race_cards(race_card_df, date.today(), race_id)

