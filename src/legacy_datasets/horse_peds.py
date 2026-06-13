import os
import re
import sys

from datetime import date, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import get_race_id
import name_header
import scraping
import past_performance

def horse_peds_dataset_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def make_horse_peds_dataset(horse_id):
    """  horse_pedsのDataFrameをscrapingして作成
        Args:
            horse_id (int) : horse_id
        Returns:
            DataFrame: horse_idの血統データ    
    """
    # 血統データの取得(スクレイピング)
    url = "https://db.netkeiba.com/horse/ped/" + horse_id
    peds_df =  scraping.scrape_df(url)
    peds_df = peds_df[0]
    try:
        #重複を削除して1列のSeries型データに直す
        generations = {}
        for i in reversed(range(5)):
            generations[i] = peds_df[i]
            peds_df.drop([i], axis=1, inplace=True)
            peds_df = peds_df.drop_duplicates()
        ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
        #列名をpeds_0, ..., peds_61にする
        peds_df = ped.reset_index(drop=True).T.add_prefix('peds_')
    except Exception as e:
        horse_peds_dataset_error(e)
        return pd.DataFrame

    return peds_df

def save_horse_peds_dataset(horse_id, peds_df):
    """ horse_pedsのDataFrameを保存 
        Args:
            horse_id (int) :horse_id
            past_performance_df.DataFrame） : past_performanceのデータセット
    """
    try:
        if any(peds_df):
            # str型に変更
            peds_df = peds_df.astype(str)
            # csv/pickleに保存
            peds_df.to_csv(name_header.DATA_PATH + "/HorsePeds/" + str(horse_id) + ".csv")
            peds_df.to_pickle(name_header.DATA_PATH + "/HorsePeds/" + str(horse_id) + ".pickle")
    except Exception as e:
            horse_peds_dataset_error(e)

def normalize_horse_name_strings(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)

    # カタカナで始まる場合（全角カタカナ＋長音記号＋中点など）
    if re.match(r'^[ァ-ヶー・ヴ]', s):
        # 先頭のカタカナ部分だけ抽出
        m = re.match(r'^([ァ-ヶー・ヴ]+)', s)
        if m:
            s = m.group(1)

    # 英語で始まる場合（A-Z/a-z）
    elif re.match(r'^[A-Za-z]', s):
        # 括弧と中身を削除
        s = re.sub(r'\([^)]*\)', '', s)

    # 行末の空白を削除
    s = s.strip()
    return s

def get_horse_peds_csv(horse_id):
    """ horse_idのhorse_pedsデータcsvを取得 
        Args:
            horse_id (int) : horse_id
    """
    # csvを読み込む 
    path = name_header.DATA_PATH + "/HorsePeds/" + str(horse_id) + '.csv'
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0, dtype = str)
        df.fillna(np.nan)
        col = df.columns[0]
        df[col] = df[col].apply(normalize_horse_name_strings)
    else :
        df = pd.DataFrame()

    return df

def get_peds_info(horse_id):
    """血統情報(父・母父)を取得
        Args:
            horse_id(int) : horse_id
        Returns:
            [父、母父](str) : 父と母父
    """
    try:
        peds_datas = get_horse_peds_csv(horse_id)
        # データセットがない場合は新規に作成
        if peds_datas.empty:
            peds_datas = make_horse_peds_dataset(horse_id)
            peds_datas = delete_invalid_strings(peds_datas)
            save_horse_peds_dataset(horse_id, peds_datas)
            peds_datas = get_horse_peds_csv(horse_id)
        peds_datas = peds_datas[str(horse_id)].tolist()
        return [peds_datas[0], peds_datas[4]]
    except Exception as e:
        horse_peds_dataset_error(e)
        return [np.nan, np.nan]

def is_horse_peds_dataset(horse_id):
    """ hotse_idの血統データが既に存在するかをチェックする 
        Args:
            horse_id (str) : horse_id
        Returns:
            Bool: hotse_idの血統データが既存 True / 未作成 False   
    """
    path = name_header.DATA_PATH + "/HorsePeds/" + str(horse_id) + ".csv"
    if os.path.isfile(path):
        return True
    else:
        return False

def delete_invalid_strings(peds_df):
    """ DataFrameの生年以降の文字列を消去 
        Args:
            peds_df (DataFrame) : 血統データのデータフレーム
        Returns:
            peds_df (DataFrame) : 不要な文字列を消去した血統データのデータフレーム 
    """
    for l in range(len(peds_df)):
        pattern =  re.findall(r'\d+', peds_df.iloc[l]) 
        if pattern:
            # 生年以降を消去
            pos = str(peds_df.iloc[l]).find(pattern[0])
            temp = str(peds_df.iloc[l][:pos])
            peds_df.iloc[l] = temp
        # 前後の空白を消去
        peds_df.iloc[l] = str(peds_df.iloc[l]).strip()
    
    return peds_df

def make_horse_peds_datasets_from_horse_id_list(horse_id_list):
    """ horse_id_listからhorse_pedsのDatasetを作成 
        Args:
            horse_id_list : horse_idのリスト
    """
    try:
        for horse_id in tqdm(horse_id_list):
            # データセットが既存かチェック
            if not is_horse_peds_dataset(horse_id):
                # ない場合は新しく作成
                peds_df = make_horse_peds_dataset(horse_id)
                peds_df = delete_invalid_strings(peds_df)
                save_horse_peds_dataset(horse_id, peds_df)
    except Exception as e:
        horse_peds_dataset_error(e)
      
def weekly_update_horse_peds(day = date.today()):
    """ 指定した日にちから、１週間分のデータセットを更新  
        Args:
         day(Date) : 日（初期値：今日）
    """ 
    for place_id in tqdm(range(1, len(name_header.PLACE_LIST) + 1)):
        print("[WeeklyUpdate]" + name_header.PLACE_LIST[place_id -1] + "HorsePeds")
        race_id_list = get_race_id.get_past_weekly_id(place_id, day)
        horse_id_list = past_performance.get_horse_id_list_from_race_id_list(race_id_list)
        make_horse_peds_datasets_from_horse_id_list(horse_id_list)
        
def monthly_update_horse_peds(day = date.today()):
    """ 指定した日にちまでのその年のデータセットを更新  
        Args:
            day(Date) : 日（初期値：今日）
    """ 
    for place_id in tqdm(range(1, len(name_header.PLACE_LIST) + 1)):
        print("[MonthlyUpdate]" + name_header.PLACE_LIST[place_id -1] + "HorsePeds")
        race_id_list = get_race_id.get_past_year_id(place_id, day)
        horse_id_list = past_performance.get_horse_id_list_from_race_id_list(race_id_list)
        make_horse_peds_datasets_from_horse_id_list(horse_id_list)

def make_all_horse_peds(year = date.today().year):
    """ 指定した年までの、すべてのデータセットを作成 
        Args:
            day(Date) : 日（初期値：今日）
    """ 
    for y in range(2019, year + 1):
        for place_id in range(1, len(name_header.PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + name_header.PLACE_LIST[place_id -1] + " HorsePeds")
            race_id_list = get_race_id.get_year_id_all(place_id,y)
            horse_id_list = past_performance.get_horse_id_list_from_race_id_list(race_id_list)
            make_horse_peds_datasets_from_horse_id_list(horse_id_list)

if __name__ == "__main__":

    race_id_list = get_race_id.get_year_id_all(8,2024)
    horse_id_list = past_performance.get_horse_id_list_from_race_id_list(race_id_list)
    make_horse_peds_datasets_from_horse_id_list(horse_id_list)
    