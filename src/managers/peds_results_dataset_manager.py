"""peds_results（血統別レース結果集計）データセットのManager層

CSVの読み込み・書き込み・永続化・集計を担う。
集計の純粋ロジック自体は src/datasets/horse/transform.py に切り出している。

旧 src/legacy_datasets/peds_results.py からの移植。
"""

import os
from datetime import date
from glob import glob

import numpy as np
import pandas as pd

from src.config import paths
from src.config.constants import PLACE_LIST
from src.datasets.horse import model, transform
from src.managers import (
    horse_peds_dataset_manager,
    past_performance_dataset_manager,
    race_result_dataset_manager,
    race_schedule_dataset_manager,
)
from src.utils.file_utils import read_csv_or_empty

PEDS_RESULTS_DATA_PATH = os.path.join(paths.HORSE_DATA_PATH, "peds_results")


def make_peds_dataset_from_race_results(place_id, year):
    """race_resultsからpeds_datasetを作成する

    Args:
        place_id (int): 開催コースid
        year (int): 開催年
    """
    df_course = race_result_dataset_manager.get_race_results_csv(place_id, year)

    if not df_course.empty:
        horse_id_list = df_course["horse_id"].to_list()
        df_peds = get_peds_dataset_from_horse_id_list(horse_id_list)
        save_peds_dataset(df_peds, place_id, year)


def save_peds_dataset(peds_df, place_id, year):
    """peds_datasetのDataFrameを保存する

    Args:
        peds_df (pd.DataFrame): peds_dataset
        place_id (int): 開催コースid
        year (int): 開催年
    """
    if peds_df is None or peds_df.empty:
        return

    peds_df = peds_df.drop_duplicates(keep="first")
    peds_df = peds_df.astype(str)

    out_dir = os.path.join(PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    peds_df.to_csv(os.path.join(out_dir, f"{year}_peds.csv"))
    peds_df.to_pickle(os.path.join(out_dir, f"{year}_peds.pickle"))


def get_peds_dataset_csv(place_id, year):
    """pedsのcsvを取得する

    Args:
        place_id (int): 開催コースid
        year (int): 開催年

    Returns:
        pd.DataFrame: pedsデータセット（存在しない場合は空のDataFrame）
    """
    path = os.path.join(PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_peds.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_peds_data_dataset_csv(place_id, year):
    """peds_dataのcsvを取得する

    Args:
        place_id (int): 開催コースid
        year (int): 開催年

    Returns:
        pd.DataFrame: peds_dataデータセット（存在しない場合は空のDataFrame）
    """
    path = os.path.join(PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_peds_data.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_peds_dataset_from_horse_id_list(horse_id_list):
    """horse_id_listからpeds_datasetを作成する

    Args:
        horse_id_list (list): horse_idのリスト

    Returns:
        pd.DataFrame: peds_dataset
    """
    peds_df = pd.DataFrame()
    for horse_id in horse_id_list:
        peds_df = pd.concat([peds_df, horse_peds_dataset_manager.get_horse_peds_csv(horse_id).T], axis=0)
    return peds_df


def merge_pedsdata_with_race_results(place_id, year):
    """血統とレース結果の統合データを作成・保存する

    Args:
        place_id (int): 開催コースid
        year (int): 開催年
    """
    df_course = race_result_dataset_manager.get_race_results_csv(place_id, year)
    if df_course.empty:
        return

    # レース結果から"着順","race_type","course_len","ground_state","class"を抽出
    df_result = df_course["着順"]
    df_result = pd.concat([df_result, df_course["race_type"]], axis=1)
    df_result = pd.concat([df_result, df_course["course_len"]], axis=1)
    df_result = pd.concat([df_result, df_course["ground_state"]], axis=1)
    df_result = pd.concat([df_result, df_course["class"]], axis=1)

    # 血統データの取得
    df_peds = get_peds_dataset_csv(place_id, year)
    # horse_idに一致する血統情報を抽出
    df_peds_result = pd.DataFrame()
    for i in range(len(df_result.index)):
        row = df_course.iloc[i, :]
        horse_id = int(row["horse_id"])
        if horse_id in df_peds.index:
            df_peds_result = pd.concat([df_peds_result, df_peds.loc[horse_id, :]], axis=1)
        else:
            horse_peds_data = horse_peds_dataset_manager.get_horse_peds_csv(str(horse_id))
            if horse_peds_data.empty:
                horse_peds_data = pd.DataFrame(np.nan, index=df_peds_result.index, columns=[horse_id])
            df_peds_result = pd.concat([df_peds_result, horse_peds_data], axis=1)

    df_peds_result = df_peds_result.reset_index(drop=True).T

    # index/columnsの整理
    df_peds_result = df_peds_result.reset_index(drop=True)
    df_peds_result.columns = [f"peds_{i}" for i in range(len(df_peds_result.columns))]
    # 血統情報を結合
    df_result = pd.concat([df_result.reset_index(drop=True), df_peds_result], axis=1)
    df_result.index = df_course.index
    df_result.index.name = "index"

    # データセットの保存
    out_dir = os.path.join(PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    df_result.to_csv(os.path.join(out_dir, f"{year}_peds_data.csv"))
    df_result.to_pickle(os.path.join(out_dir, f"{year}_peds_data.pickle"))


def peds_index(father, mother_father, course_info, year):
    """血統情報の着度数を抽出する

    Args:
        father (str): 父
        mother_father (str): 母父
        course_info (list): [place_id, race_type, course_len, ground_state, race_class]
        year (int): 年

    Returns:
        pd.DataFrame: 血統の着度数データ
    """
    place_id, race_type, course_len, ground_state, race_class = course_info[:5]

    df_peds = pd.DataFrame()
    # 各年度のデータを抽出
    for y in range(2019, int(year)):
        df_peds = pd.concat([df_peds, get_peds_data_dataset_csv(place_id, y)])
    df_peds = df_peds.reset_index(drop=True)

    return_df = pd.DataFrame()

    # 父情報を抽出
    peds_data_father = df_peds[df_peds["peds_0"] == father]
    peds_data_father = transform.get_race_type_data(peds_data_father, race_type, ground_state)
    return_df = pd.concat([return_df, transform.calc_peds_data(peds_data_father, course_len, race_class)], axis=0)

    # 母父情報を抽出
    peds_data_mother_father = df_peds[df_peds["peds_4"] == mother_father]
    peds_data_mother_father = transform.get_race_type_data(peds_data_mother_father, race_type, ground_state)
    return_df = pd.concat(
        [return_df, transform.calc_peds_data(peds_data_mother_father, course_len, race_class)], axis=0
    )

    # 父×母父情報を抽出
    peds_data = df_peds[df_peds["peds_0"] == father]
    peds_data = peds_data[peds_data["peds_4"] == mother_father]
    peds_data = transform.get_race_type_data(peds_data, race_type, ground_state)
    return_df = pd.concat([return_df, transform.calc_peds_data(peds_data, course_len, race_class)], axis=0)

    return return_df


def aggregate_peds_results(place_id, year):
    """コース（芝/ダート・距離）×馬場状態×クラスごとに血統成績を集計して保存する

    全馬場状態(all)、全クラス(all)の集計も同時に出力する。

    Args:
        place_id (int): 開催コースid
        year (int): 開催年
    """
    df = get_peds_data_dataset_csv(place_id, year)
    if df.empty:
        print("not PedsResultData:", PLACE_LIST[place_id - 1], year)
        return

    output_dir = os.path.join(PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1], str(year))
    os.makedirs(output_dir, exist_ok=True)

    # 着順を数値化
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")

    # race_type, course_lenごとに処理
    for (race_type, course_len), df_group in df.groupby(["race_type", "course_len"]):
        ground_list = sorted(df_group["ground_state"].dropna().unique())

        # 各馬場状態ごと
        for ground_state in ground_list + ["all"]:
            if ground_state == "all":
                df_ground = df_group
            else:
                df_ground = df_group[df_group["ground_state"] == ground_state]

            if df_ground.empty:
                continue

            # 全クラスまとめ(all)
            result_all_classes = []
            result_all = transform.output_results(df_ground)
            if not result_all.empty:
                result_all.insert(0, "クラス", "all")
                result_all_classes.append(result_all)

            # クラス別処理
            for class_name, df_class in df_ground.groupby("class"):
                result_df = transform.output_results(df_class)
                if not result_df.empty:
                    result_df.insert(0, "クラス", class_name)
                    result_all_classes.append(result_df)

            # 全クラス結合
            if result_all_classes:
                final_df = pd.concat(result_all_classes, ignore_index=True)
            else:
                continue

            # 出力
            file_name = f"{race_type}_{course_len}m_{ground_state}.csv"
            output_path = os.path.join(output_dir, file_name)
            final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print("make peds_result", PLACE_LIST[place_id - 1], year, race_type, f"{course_len}m")


def aggregate_total_peds_results(place_id, start_year=2019, end_year=date.today().year):
    """各年度のpeds結果csvを統合し、Totalディレクトリに合計結果を出力する

    Args:
        place_id (int): 開催コースid
        start_year (int): 集計開始年
        end_year (int): 集計終了年（含む）
    """
    base_dir = os.path.join(PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1])
    output_dir = os.path.join(base_dir, "Total")
    os.makedirs(output_dir, exist_ok=True)

    # 全ファイル名の集合を取得
    all_files = set()
    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.exists(year_dir):
            continue
        csv_files = [os.path.basename(p) for p in glob(os.path.join(year_dir, "*.csv"))]
        all_files.update(csv_files)

    print(f"集計対象ファイル数: {len(all_files)} ({start_year}-{end_year})")

    # 各ファイル名ごとに集計
    for file_name in sorted(all_files):
        merged_df_list = []

        for year in range(start_year, end_year + 1):
            csv_path = os.path.join(base_dir, str(year), file_name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if not df.empty:
                    df["year"] = year
                    merged_df_list.append(df)

        if not merged_df_list:
            continue

        df_all = pd.concat(merged_df_list, ignore_index=True)

        # 集計
        agg_df = (
            df_all.groupby(["クラス", "血統"], as_index=False)[["1着", "2着", "3着", "着外"]]
            .sum()
        )

        # クラス順序をカテゴリ型で保持
        agg_df["クラス"] = pd.Categorical(agg_df["クラス"], categories=model.CLASS_ORDER, ordered=True)

        # クラス → 着順 の順でソート
        agg_df = agg_df.sort_values(
            by=["クラス", "1着", "2着", "3着"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

        # 出力
        output_path = os.path.join(output_dir, file_name)
        agg_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print("Total集計完了:", file_name)

    print("すべてのTotal集計が完了しました ->", output_dir)


def get_total_peds_results_csv(place_id, race_type, course_len, ground_state):
    """data/horse/peds_results/{place}/Total/{race_type}_{course_len}m_{ground_state}.csv を取得する"""
    path = os.path.join(
        PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1], "Total", f"{race_type}_{course_len}m_{ground_state}.csv"
    )
    return read_csv_or_empty(path, dtype=None)


def get_annual_peds_results_csv(place_id, year, race_type, course_len, ground_state):
    """data/horse/peds_results/{place}/{year}/{race_type}_{course_len}m_{ground_state}.csv を取得する"""
    path = os.path.join(
        PEDS_RESULTS_DATA_PATH, PLACE_LIST[place_id - 1], str(year), f"{race_type}_{course_len}m_{ground_state}.csv"
    )
    return read_csv_or_empty(path, dtype=None)


def update_peds_dataset(place_id, day=date.today()):
    """指定したコースの指定日から、1週間分のpeds_datasetを更新する

    Args:
        place_id (int): 開催コースid
        day (date): 日（初期値：今日）
    """
    race_id_list = race_schedule_dataset_manager.get_past_weekly_id(place_id, day)
    horse_id_list = past_performance_dataset_manager.get_horse_id_list_from_race_id_list(race_id_list)
    if horse_id_list:
        new_peds_df = get_peds_dataset_from_horse_id_list(horse_id_list)
        old_peds_df = get_peds_dataset_csv(place_id, day.year)
        new_peds_df = pd.concat([old_peds_df, new_peds_df], axis=0)
        save_peds_dataset(new_peds_df, place_id, day.year)
