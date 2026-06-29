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
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import name_header
import get_race_id

import horse_peds
import race_results
import peds_results
import past_performance
import average_time

# 学習データセット（peds/過去成績から作る特徴量CSV）はver3.0側に保存する。
# 旧実装はname_header.DATA_PATH（ver2.0側）に保存していたが、git管理（バックアップ）
# できるようにver3.0のsrc.config.pathsへ向ける。学習済みモデルの保存先を
# ver3.0へ向けた変更（prediction.py）と同じ理由。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.config import paths as paths_v3  # noqa: E402

index = [ 
        # racce_id
        "race_id",
        # 血統情報
            # コース着度数
            "f_course1", "f_course2", "f_course3", "f_courseout",
            "mf_course1", "mf_course2", "mf_course3", "mf_courseout",
            "f_mf_course1", "f_mf_course2", "f_mf_course3", "f_mf_courseout", 
            # キョリ着度数
            "f_len1", "f_len2", "f_len3","f_lenout",
            "mf_len1", "mf_len2", "mf_len3", "mf_classout",
            "f_mf_len1", "f_mf_len2", "f_mf_len3", "f_mf_classout", 
            # クラス着度数
            "f_class1", "f_class2", "f_class3", "f_classout", 
            "mf_class1", "mf_class2", "mf_class3", "mf_classout", 
            "f_mf_class1", "f_mf_class2", "f_mf_class3", "f_mf_classout", 
        # 過去レース情報
        "time_df_course1", "time_df_class1", "ninki_1", "result_1", 
        "time_df_course2", "time_df_class2", "ninki_2", "result_2", 
        "time_df_course3", "time_df_class3", "ninki_3", "result_3", 
        # 枠順
        "waku", "umaban",
        ]

# 回収率重視モデル（サブB）用の列。的中率重視モデル（index）に「そのレース自身の人気」を1列追加したもの。
index_value = index + ["self_popularity"]

def make_dataset_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def save_LightGBM_dataset_csv(place_id, year, type, length, df_dataset, suffix=""):
    """ LightGBM_datasetのDataFrameを保存
        Args:
            place_id(int) : place_id
            year (int) : 年
            type(str) : レースタイプ
            length(int) : キョリ
            df_dataset(pd.DataFrame） : lightGBM用のデータセット
            suffix(str) : ファイル名サフィックス（回収率重視モデル(サブB)用は"_value"）
    """
    try:
        if any(df_dataset):
            out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{year}_{type}{length}_ai_dataset_for_rank{suffix}.csv")
            df_dataset = df_dataset.reset_index(drop = True)
            # ローカル保存
            df_dataset.to_csv(path)
    except Exception as e:
            make_dataset_error(e)

def sava_LightGBM_dataset_flag_csv(place_id, year, type, length, flag_list, suffix=""):
    """ LightGBM_datasetのDataFrameを保存
        Args:
            place_id(int) : place_id
            year (int) : 年
            type(str) : レースタイプ
            length(int) : キョリ
            flag_list(list） : lightGBM用のflagデータセット
            suffix(str) : ファイル名サフィックス（回収率重視モデル(サブB)用は"_value"）
    """
    try:
        if flag_list:
            df_flag = pd.DataFrame(flag_list)
            df_flag.columns = ["result_flag"]
            out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
            os.makedirs(out_dir, exist_ok=True)
            flag_path = os.path.join(out_dir, f"{year}_{type}{length}_ai_dataset_flag{suffix}.csv")
            df_flag.to_csv(flag_path)
    except Exception as e:
            make_dataset_error(e)

def get_LightGBM_dataset_csv(place_id, year, type, length, suffix=""):
    """ LightGBM_datasetのDataFrameを保存
        Args:
            place_id(int) : place_id
            year (int) : 年
            type(str) : レースタイプ
            length(int) : キョリ
            suffix(str) : ファイル名サフィックス（回収率重視モデル(サブB)用は"_value"）
        Returns:
            df(pd.DataFrame） : lightGBM用のデータセット
    """
    # csvを読み込む
    path = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1], f"{year}_{type}{length}_ai_dataset_for_rank{suffix}.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0, dtype = float)
    else :
        df = pd.DataFrame()
    return df

def get_LightGBM_dataset_flag_csv(place_id, year, type, length, suffix=""):
    """ LightGBM_datasetのDataFrameを保存
        Args:
            place_id(int) : place_id
            year (int) : 年
            type(str) : レースタイプ
            length(int) : キョリ
            suffix(str) : ファイル名サフィックス（回収率重視モデル(サブB)用は"_value"）
        Returns:
            df(pd.DataFrame） : lightGBM用のflagデータセット
    """
    # csvを読み込む
    path = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1], f"{year}_{type}{length}_ai_dataset_flag{suffix}.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col = 0, dtype = int)
    else :
        df = pd.DataFrame()
    return df

def relevance_grade(df_result, race_id):
    """ 着順を段階的な関連度スコア（LambdaRankの目的変数）に変換する
        1着=4, 2着=3, 3着=2, 4〜5着=1, 6着以降・出走取消等=0。
        旧実装（is_in_show）は「3着以内か否か」の二値だったため、
        1着と3着の強さの差をモデルが学習できなかった。段階的なラベルにすることで、
        LambdaRankが本来持つ「順位の強さの差」を学習する力を活かす。
        Args:
            df_result(pd.DataFrame） : lightGBM用のデータセット
            race_id(int) : race_id
        Returns:
            int : 関連度スコア（0〜4）
    """
    rank = df_result.at[race_id,"着順"]
    rank_str = str(rank)
    if not rank_str.isdigit():
        return 0
    try:
        rank_val = int(rank_str)
    except Exception:
        return 0
    if rank_val == 1:
        return 4
    if rank_val == 2:
        return 3
    if rank_val == 3:
        return 2
    if rank_val <= 5:
        return 1
    return 0

def value_grade(df_result, race_id):
    """ 回収率重視モデル（サブB）用の関連度スコア（LambdaRankの目的変数）
        着順だけでなく単勝オッズ（穴度）を反映し、人気馬の好走より穴馬の好走を
        高く評価する。「市場（人気）より強い馬」を見つけるモデルにするための目的変数。
        着外は0。1〜3着は (着順ごとの基礎点) + (オッズに応じたボーナス、5倍ごとに+1、最大+7)。
        例: 1番人気の1着→3点、10倍の馬が1着→3+2=5点、40倍の大穴の1着→3+7=10点（上限）。
        ラベルはLightGBMのlambdarank既定のgainテーブル範囲に収まるよう0〜10の整数にしている。
        Args:
            df_result(pd.DataFrame） : lightGBM用のデータセット
            race_id(int) : race_id
        Returns:
            int : 関連度スコア（0〜10）
    """
    rank = df_result.at[race_id,"着順"]
    rank_str = str(rank)
    if not rank_str.isdigit():
        return 0
    try:
        rank_val = int(rank_str)
    except Exception:
        return 0
    if rank_val not in (1, 2, 3):
        return 0
    base = {1: 3, 2: 2, 3: 1}[rank_val]
    try:
        odds_val = float(df_result.at[race_id, "単勝"])
    except Exception:
        odds_val = 1.0
    odds_val = max(1.0, odds_val)
    bonus = min(int(odds_val // 5), 7)
    return base + bonus

def get_self_popularity(df_result, race_id):
    """ そのレース自身の確定人気（回収率重視モデル(サブB)の特徴量）を取得する
        Args:
            df_result(pd.DataFrame） : lightGBM用のデータセット
            race_id(int) : race_id
        Returns:
            float : 人気（取得失敗時はnp.nan）
    """
    try:
        return float(df_result.at[race_id, "人気"])
    except Exception:
        return np.nan

def get_past_race_info_data(race_info_df):
    """ 過去レースのタイム指数、着順、人気を取得 
        Args:
            race_info_df(pd.DataFrame）: lightGBM用のデータセット
        Returns:
            race_score_list(list) : race_scoreのlist
    """
    race_score_list = []
    for i in range(3):
        # レース結果がある場合
        if i < len(race_info_df.index):
            # レース結果を取得(time_df(コース、距離、馬場), time_df(同クラス),"course_flag1", "len_flag1", "type_flag1", "state_flag1", "class_flag1" )
            df_time = get_time_info(race_info_df.iloc[i])

            # 着順・人気は欠損や特殊文字が混じるため文字列化して安全に扱う
            raw_rank = race_info_df.at[i, "着順"]
            raw_pop = race_info_df.at[i, "人気"]
            rank_str = str(raw_rank)

            if "除" in rank_str or "取" in rank_str:
                df_time.append(np.nan)
                df_time.append(np.nan)
            elif "中" in rank_str or "失" in rank_str:
                # 人気だけは数値抽出を試みる（失格等で空の場合あり）
                try:
                    pop_val = float(re.sub(r"\D", "", str(raw_pop)))
                except Exception:
                    pop_val = np.nan
                df_time.append(pop_val)
                df_time.append(np.nan)
            else:
                # 人気・着順を数値化（失敗したら NaN）
                try:
                    pop_val = float(re.sub(r"\D", "", str(raw_pop)))
                except Exception:
                    pop_val = np.nan
                try:
                    rank_val = float(re.sub(r"\D", "", rank_str))
                except Exception:
                    rank_val = np.nan
                df_time.append(pop_val)
                df_time.append(rank_val)
            race_score_list.append(df_time)
            
        else:
            race_score_list.append([np.nan, np.nan, np.nan, np.nan])
    
    # リストの１次元化
    race_score_list = sum(race_score_list, [])

    return race_score_list

def get_time_info(race_info):
    """ 過去のレース結果よりタイム差を取得(タイム差/平均タイム[ms]) 
        Args:
            race_info(pd.DataFrame) : race結果のdataframe
        Returns:
            diff_time_list(list) : タイム差リスト
    """
    # 過去レースのコース情報を取得
    course_info = race_results.get_course_info(race_info)
    # JRA以外のコースの場合0を返す
    if  course_info[0] < 0:
        return [0,0]
    # 走破タイムをmsecに変換
    race_time = average_time.get_race_time_msec(race_info["タイム"])
    # タイム差の計算
    diff_time_list = average_time.get_time_diff(race_time, course_info)
    return diff_time_list

def make_dataset_for_train(place_id, year = date.today().year):
    """ LightGBM_datasetのDataFrameを作成
        Args:
            place_id(int) : place_id
            year (int) : 年
    """
    # 過去のレース結果をデータフレームで取得
    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print("not data:", year, name_header.PLACE_LIST[place_id - 1])
        return
    
    # コースごとにデータセットを分割
    for type, length in name_header.COURSE_LISTS[place_id - 1]:
        print(type,length)
        df_results_course = df_results[df_results["race_type"] == type]
        df_results_course = df_results_course[df_results_course["course_len"] == length]
        if df_results_course.empty:
            continue
        # race_idの取得
        race_id_list = pd.DataFrame(df_results_course.index, columns = ["race_id"])
        
        # datasetを格納するDataFrame
        df_dataset = pd.DataFrame()
        # 着順の関連度スコア（0〜4）
        flag_list = []

        # 各馬のデータセットを作成
        for i in tqdm(range(len(df_results_course))):
            df_result = df_results_course[i:i+1]
            race_id = int(df_result.index.values)

            # 着順を関連度スコアに変換
            flag_list.append(relevance_grade(df_result, race_id))
            
            # レース情報の取得
            course_info = [place_id, df_result.at[race_id,"race_type"], df_result.at[race_id,"course_len"], df_result.at[race_id,"ground_state"],df_result.at[race_id,"class"] ]
            horse_id = df_result.at[race_id, "horse_id"]
            
            # データセットの作成
            df_index = make_dataset_for_lightGBM(race_id, course_info, horse_id)
            df_dataset = pd.concat([df_dataset.reset_index(drop = True), df_index.reset_index(drop = True)])

        # race_idの結合
        df_dataset = pd.concat([race_id_list, df_dataset.reset_index(drop = True)], axis = 1)
        # 枠番・馬番の結合
        df_dataset = pd.concat([df_dataset, df_results_course["枠番"].reset_index(drop = True),df_results_course["馬番"].reset_index(drop = True)], axis = 1).reset_index(drop = True)
        df_dataset.columns = index

        # csvでデータセットを出力
        save_LightGBM_dataset_csv(place_id, year, type, length, df_dataset)
        # csvでフラグデータセットを出力
        sava_LightGBM_dataset_flag_csv(place_id, year, type, length, flag_list)

def make_dataset_for_train_value(place_id, year = date.today().year):
    """ 回収率重視モデル（サブB）用のLightGBM_datasetのDataFrameを作成
        的中率重視モデル（make_dataset_for_train）と同じ特徴量に「そのレース自身の
        確定人気」を1列追加し、目的変数も着順だけでなくオッズ（穴度）を反映した
        value_gradeにする。学習データの人気・オッズは既存のレース結果CSVに
        既に含まれているため、新規スクレイピングは不要。
        Args:
            place_id(int) : place_id
            year (int) : 年
    """
    # 過去のレース結果をデータフレームで取得
    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print("not data:", year, name_header.PLACE_LIST[place_id - 1])
        return

    # コースごとにデータセットを分割
    for type, length in name_header.COURSE_LISTS[place_id - 1]:
        print(type, length, "(value)")
        df_results_course = df_results[df_results["race_type"] == type]
        df_results_course = df_results_course[df_results_course["course_len"] == length]
        if df_results_course.empty:
            continue
        # race_idの取得
        race_id_list = pd.DataFrame(df_results_course.index, columns = ["race_id"])

        # datasetを格納するDataFrame
        df_dataset = pd.DataFrame()
        # 回収率重視の関連度スコア（0〜10）
        flag_list = []
        # そのレース自身の確定人気（特徴量）
        self_pop_list = []

        # 各馬のデータセットを作成
        for i in tqdm(range(len(df_results_course))):
            df_result = df_results_course[i:i+1]
            race_id = int(df_result.index.values)

            # 着順・オッズを回収率重視の関連度スコアに変換
            flag_list.append(value_grade(df_result, race_id))
            self_pop_list.append(get_self_popularity(df_result, race_id))

            # レース情報の取得
            course_info = [place_id, df_result.at[race_id,"race_type"], df_result.at[race_id,"course_len"], df_result.at[race_id,"ground_state"],df_result.at[race_id,"class"] ]
            horse_id = df_result.at[race_id, "horse_id"]

            # データセットの作成（的中率重視モデルと同じ特徴量を再利用）
            df_index = make_dataset_for_lightGBM(race_id, course_info, horse_id)
            df_dataset = pd.concat([df_dataset.reset_index(drop = True), df_index.reset_index(drop = True)])

        # race_idの結合
        df_dataset = pd.concat([race_id_list, df_dataset.reset_index(drop = True)], axis = 1)
        # 枠番・馬番・自レース人気の結合
        df_dataset = pd.concat(
            [
                df_dataset,
                df_results_course["枠番"].reset_index(drop = True),
                df_results_course["馬番"].reset_index(drop = True),
                pd.Series(self_pop_list, name = "self_popularity"),
            ],
            axis = 1,
        ).reset_index(drop = True)
        df_dataset.columns = index_value

        # csvでデータセットを出力（"_value"サフィックスで的中率重視モデルのデータと分離）
        save_LightGBM_dataset_csv(place_id, year, type, length, df_dataset, suffix="_value")
        # csvでフラグデータセットを出力
        sava_LightGBM_dataset_flag_csv(place_id, year, type, length, flag_list, suffix="_value")

def make_dataset_for_lightGBM(race_id, course_info, horse_id):
    """lightGBM用のデータセットを作成
        Args:
            race_id(int) : race_id
            course_info(list) : コース情報(place_id, 芝/ダ, キョリ, 馬場状態, クラス)
            horse_id(int) : horse_id
        Returns:
            df_lightGBM(pd.DataFrame) : lightGBM用データセット
    """
    try:
        race_year = int(str(race_id)[0] + str(race_id)[1] + str(race_id)[2] + str(race_id)[3])
        # 血統情報の取得
        peds_info = horse_peds.get_peds_info(horse_id)

        # 血統データの取得
        df_peds = peds_results.peds_index(peds_info[0], peds_info[1], course_info, race_year)
        # リストの１次元化
        df_peds = sum(df_peds.T.values.tolist(), [])

        # 過去3レースの情報を取得
        race_info = past_performance.get_past_race_info(horse_id, race_id, race_num = 3)
        # 過去3レース分のタイムデータを取得
        df_race = get_past_race_info_data(race_info)

        # データセットの結合
        df_lightGBM = df_peds + df_race

        # データフレームに格納
        df_lightGBM = pd.DataFrame(df_lightGBM).T

        return df_lightGBM
    except Exception as e:
        make_dataset_error(e)
        return pd.DataFrame()

def weekly_update_dataset_for_train(day = date.today()):
    """１週間に開催のあった開催場のデータセットを更新
        Args:
            day(date) : 日（初期値：今日）
    """
    try:
        place_id_list = get_race_id.get_past_weekly_place_id(day)
        for place_id in place_id_list:
            print("[Update]", name_header.PLACE_LIST[place_id - 1], "LightGBM Dataset")
            make_dataset_for_train(place_id, day.year)
    except Exception as e:
        make_dataset_error(e)

def make_annual_dataset(year = date.today().year):
    """指定年の開催場のデータセットを更新
        Args:
            year(int) : 年（初期値：今年）
    """
    try:
        for place_id in range(1, len(name_header.PLACE_LIST) + 1):
            print("[Make Dataset]", name_header.PLACE_LIST[place_id - 1], "LightGBM Dataset")
            make_dataset_for_train(place_id, year)
    except Exception as e:
        make_dataset_error(e)

def make_annual_dataset_value(year = date.today().year):
    """指定年の開催場の回収率重視モデル（サブB）用データセットを更新
        Args:
            year(int) : 年（初期値：今年）
    """
    try:
        for place_id in range(1, len(name_header.PLACE_LIST) + 1):
            print("[Make Dataset(value)]", name_header.PLACE_LIST[place_id - 1], "LightGBM Dataset")
            make_dataset_for_train_value(place_id, year)
    except Exception as e:
        make_dataset_error(e)

    
if __name__ == "__main__":
   print("analysis_datasets")
   for year in range(2020, 2025):
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        print(year, name_header.PLACE_LIST[place_id - 1])
        make_dataset_for_train(place_id, year)