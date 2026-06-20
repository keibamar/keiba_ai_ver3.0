"""race_info（レース基本情報・集計）データセットのManager層

CSVの読み込み・書き込み・永続化を担う。
変換ロジック自体は src/datasets/race_info/transform.py に切り出している。

旧 src/legacy_datasets/analysis_race_info.py, analysis_race_time.py, average_time.py
からの移植。race_resultsデータは src/managers/race_result_dataset_manager.py から取得する。

出力先は data/race_info/ 配下のスネークケースのサブディレクトリに配置する
（旧実装の AveragePops/AverageWeights/AverageFrames/AverageTimes、data/horse_id_map.csv に相当）。

旧実装の update_average_pops / update_average_frame_and_horse は、top3用・全期間用の
保存判定に誤って result_top.empty（1年分・勝ち馬のみの結果）を使っていたが、
新実装ではそれぞれの結果（result_top3 / total_top / total_top3）の空判定に修正している。

旧実装の winners_time_update(place_id, year) は引数を無視して全開催場・全年度
（2019〜今年）を処理していたが、新実装の update_winner_time(place_id, year) は
引数で指定した開催場・年のみを処理する。

旧 src/legacy_datasets/race_returns.py のうち、配当結果のフォーマット・永続化
（get_race_returns_csv, save_race_returns_dataset, split_race_returns_csv）を移植。
スクレイピング・更新オーケストレーション（scrape_race_returns_dataframe 以降）は別フェーズで対応する。

旧 src/RacePrediction/calc_returns.py の save_each_race_return_csv を
save_race_return_for_race_id として移植。1レース分の配当結果を
data/race_info/race_returns/{place}/{year}/{race_id}.csv に保存する
（get_race_return_csv_for_race と同じ構成のper-race CSV）。
"""

import os
from datetime import date

import pandas as pd

from src.config import paths
from src.config.constants import PLACE_LIST
from src.config.lists import COURSE_LISTS
from src.datasets.race_info import model, transform
from src.logic.calculators import average_calculator
from src.managers import race_result_dataset_manager
from src.utils.file_utils import read_csv_or_empty

HORSE_ID_MAP_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "horse_id_map.csv")
AVERAGE_POPS_DATA_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "average_pops")
AVERAGE_WEIGHTS_DATA_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "average_weights")
AVERAGE_RETURNS_DATA_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "average_returns")
CHAKUDO_DATA_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "chakudo")
AVERAGE_FRAMES_DATA_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "average_frames")
AVERAGE_TIMES_DATA_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "average_times")
RACE_RETURNS_DATA_PATH = os.path.join(paths.RACE_INFO_DATA_PATH, "race_returns")


def _get_race_results_with_race_id(place_id, year):
    """race_resultsを取得し、"race_id"列を持つDataFrameにして返す"""
    df_raw = race_result_dataset_manager.get_race_results_csv(place_id, year)
    if df_raw.empty:
        return df_raw
    return df_raw.reset_index().rename(columns={"index": "race_id"})


# --- 勝ち馬の平均馬体重 -----------------------------------------------------


def analyze_winner_weights(place_id, year):
    """勝ち馬の「馬体重」平均を race_type, course_len, ground_state, class ごとに算出する"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_winner_weights(df_raw, COURSE_LISTS[place_id - 1])


def analyze_winner_weights_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について勝ち馬の平均馬体重を算出し、全期間の平均を返す"""
    results_by_year = {}
    for year in range(start_year, current_year + 1):
        df_year = analyze_winner_weights(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year
    return transform.aggregate_winner_weights(results_by_year)


# --- 着度数（人気別・枠番別・馬番別） -------------------------------------------------


def analyze_pop_chakudo(place_id, year):
    """人気別の着度数を算出する（全馬が対象、勝ち馬限定ではない）"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_pop_chakudo(df_raw, COURSE_LISTS[place_id - 1])


def analyze_frame_chakudo(place_id, year):
    """枠番別の着度数を算出する（全馬が対象、勝ち馬限定ではない）"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_frame_chakudo(df_raw, COURSE_LISTS[place_id - 1])


def analyze_horse_chakudo(place_id, year):
    """馬番別の着度数を算出する（全馬が対象、勝ち馬限定ではない）"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_horse_chakudo(df_raw, COURSE_LISTS[place_id - 1])


def _analyze_chakudo_multi_years(analyze_fn, aggregate_fn, place_id, start_year, current_year):
    results_by_year = {}
    for year in range(start_year, current_year + 1):
        df_year = analyze_fn(place_id, year)
        if not df_year.empty:
            results_by_year[year] = df_year
    return aggregate_fn(results_by_year)


def analyze_pop_chakudo_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について人気別着度数を算出し、全期間の合計を返す"""
    return _analyze_chakudo_multi_years(analyze_pop_chakudo, transform.aggregate_pop_chakudo, place_id, start_year, current_year)


def analyze_frame_chakudo_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について枠番別着度数を算出し、全期間の合計を返す"""
    return _analyze_chakudo_multi_years(analyze_frame_chakudo, transform.aggregate_frame_chakudo, place_id, start_year, current_year)


def analyze_horse_chakudo_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について馬番別着度数を算出し、全期間の合計を返す"""
    return _analyze_chakudo_multi_years(analyze_horse_chakudo, transform.aggregate_horse_chakudo, place_id, start_year, current_year)


def update_chakudo(place_id, year):
    """指定の開催場・年について、人気別・枠番別・馬番別の着度数（全期間合計）を更新する"""
    out_dir = os.path.join(CHAKUDO_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)

    for kind, multi_fn in [
        ("pop", analyze_pop_chakudo_multi_years),
        ("frame", analyze_frame_chakudo_multi_years),
        ("horse", analyze_horse_chakudo_multi_years),
    ]:
        total = multi_fn(place_id, start_year=2019, current_year=year)
        if not total.empty:
            total.to_csv(os.path.join(out_dir, f"total_{kind}_chakudo.csv"), index=False)


def get_total_pop_chakudo_csv(place_id):
    """data/race_info/chakudo/{place}/total_pop_chakudo.csv を取得する"""
    path = os.path.join(CHAKUDO_DATA_PATH, PLACE_LIST[place_id - 1], "total_pop_chakudo.csv")
    return read_csv_or_empty(path, dtype=str)


def get_total_frame_chakudo_csv(place_id):
    """data/race_info/chakudo/{place}/total_frame_chakudo.csv を取得する"""
    path = os.path.join(CHAKUDO_DATA_PATH, PLACE_LIST[place_id - 1], "total_frame_chakudo.csv")
    return read_csv_or_empty(path, dtype=str)


def get_total_horse_chakudo_csv(place_id):
    """data/race_info/chakudo/{place}/total_horse_chakudo.csv を取得する"""
    path = os.path.join(CHAKUDO_DATA_PATH, PLACE_LIST[place_id - 1], "total_horse_chakudo.csv")
    return read_csv_or_empty(path, dtype=str)


# --- 平均配当（勝ち馬の単勝オッズ） -------------------------------------------------


def analyze_average_returns(place_id, year):
    """勝ち馬の単勝配当（オッズ×100円）の平均を race_type, course_len, ground_state, class ごとに算出する"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_average_returns(df_raw, COURSE_LISTS[place_id - 1])


def analyze_average_returns_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について平均配当を算出し、全期間の平均を返す"""
    results_by_year = {}
    for year in range(start_year, current_year + 1):
        df_year = analyze_average_returns(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year
    return transform.aggregate_average_returns(results_by_year)


def update_average_returns(place_id, year):
    """指定の開催場・年について、平均配当の集計結果を更新する"""
    out_dir = os.path.join(AVERAGE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)

    result = analyze_average_returns(place_id, year)
    if not result.empty:
        result.to_csv(os.path.join(out_dir, f"{year}_average_returns.csv"))

    total = analyze_average_returns_multi_years(place_id, start_year=2019, current_year=year)
    if not total.empty:
        total.to_csv(os.path.join(out_dir, "total_average_returns.csv"))


def get_annual_average_returns_csv(place_id, year):
    """data/race_info/average_returns/{place}/{year}_average_returns.csv を取得する"""
    path = os.path.join(AVERAGE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_average_returns.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_total_average_returns_csv(place_id):
    """data/race_info/average_returns/{place}/total_average_returns.csv を取得する"""
    path = os.path.join(AVERAGE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1], "total_average_returns.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


# --- 人気 -------------------------------------------------------------------


def analyze_average_pops(place_id, year, top3=False):
    """勝ち馬または3着内馬の平均人気と人気別勝利数を集計する"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_average_pops(df_raw, COURSE_LISTS[place_id - 1], top3=top3)


def analyze_average_pop_multi_years(place_id, start_year=2019, current_year=date.today().year, top3=False):
    """各年度（start_year〜current_year）について人気データを集計し、全期間の平均を返す"""
    results_by_year = {}
    for year in range(start_year, current_year + 1):
        df_year = analyze_average_pops(place_id, year, top3=top3)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year
    return transform.aggregate_average_pops(results_by_year)


# --- 枠番・馬番（勝ち馬） -----------------------------------------------------


def analyze_average_frame_and_horse(place_id, year):
    """勝ち馬の「平均枠番」「平均馬番」「勝利数」および各枠・馬番の勝ち数を集計する"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_average_frame_and_horse(df_raw, COURSE_LISTS[place_id - 1])


def analyze_frame_and_horse_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について枠・馬番の平均と勝ち数を算出し、全期間の平均・合計を返す"""
    results_by_year = {}
    for year in range(start_year, current_year + 1):
        df_year = analyze_average_frame_and_horse(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year
    return transform.aggregate_average_frame_and_horse(results_by_year)


# --- 枠番・馬番（3着内） ------------------------------------------------------


def analyze_average_frame_and_horse_top3(place_id, year):
    """3着以内馬の「平均枠番」「平均馬番」「3着内馬数」および各枠・馬番ごとの3着内回数を集計する"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_average_frame_and_horse_top3(df_raw, COURSE_LISTS[place_id - 1])


def analyze_frame_and_horse_top3_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について3着以内馬の枠・馬番分析を算出し、全期間の平均・合計を返す"""
    results_by_year = {}
    for year in range(start_year, current_year + 1):
        df_year = analyze_average_frame_and_horse_top3(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year
    return transform.aggregate_average_frame_and_horse_top3(results_by_year)


# --- horse_id_map ------------------------------------------------------------


def get_horse_id_list():
    """data/race_info/horse_id_map.csv から horse_id のリストを作成して返す"""
    df = read_csv_or_empty(HORSE_ID_MAP_PATH, dtype=str)
    if df.empty or "horse_id" not in df.columns:
        return []
    horse_ids = df["horse_id"].str.strip()
    return horse_ids.dropna().unique().tolist()


def update_horse_name_id_map_from_results(place_id, year):
    """指定の開催場・年のrace_resultsから、horse_id_map.csv を更新する"""
    if os.path.isfile(HORSE_ID_MAP_PATH):
        existing_df = pd.read_csv(HORSE_ID_MAP_PATH, dtype=str)
    else:
        existing_df = pd.DataFrame(columns=model.HORSE_ID_MAP_COLUMNS)

    df = race_result_dataset_manager.get_race_results_csv(place_id, year)
    if df.empty or not {"馬名", "horse_id"}.issubset(df.columns):
        return

    new_df = df[["馬名", "horse_id"]].dropna()

    merged_df = transform.merge_horse_id_map(existing_df, new_df)

    os.makedirs(paths.RACE_INFO_DATA_PATH, exist_ok=True)
    merged_df.to_csv(HORSE_ID_MAP_PATH, index=False, encoding="utf-8-sig")


def add_horse_name_id_map(horse_id, horse_name):
    """horse_id_map.csv に1件追加する（horse_id・馬名どちらかが既存と重複する場合はスキップ）"""
    if not os.path.isfile(HORSE_ID_MAP_PATH):
        return

    existing_df = pd.read_csv(HORSE_ID_MAP_PATH, dtype=str)
    updated_df = transform.add_horse_id_map_entry(existing_df, horse_id, horse_name)
    if updated_df is None:
        return

    updated_df.to_csv(HORSE_ID_MAP_PATH, index=False, encoding="utf-8-sig")


def update_horse_name_id_map(race_card_df):
    """出馬表DataFrame（馬名・horse_id列を持つ）から horse_id_map.csv を更新する"""
    if not {"馬名", "horse_id"}.issubset(race_card_df.columns):
        return

    for _, row in race_card_df.iterrows():
        horse_name = str(row["馬名"]).strip()
        horse_id = str(row["horse_id"]).strip()

        if horse_name == "" or horse_id == "" or horse_id.lower() == "nan":
            continue

        add_horse_name_id_map(horse_id, horse_name)


# --- 集計結果の更新 -----------------------------------------------------------


def update_average_pops(place_id, year):
    """指定の開催場・年について、平均人気の集計結果を更新する"""
    out_dir = os.path.join(AVERAGE_POPS_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)

    result_top = analyze_average_pops(place_id, year, top3=False)
    if not result_top.empty:
        result_top.to_csv(os.path.join(out_dir, f"{year}_average_pops.csv"))

    result_top3 = analyze_average_pops(place_id, year, top3=True)
    if not result_top3.empty:
        result_top3.to_csv(os.path.join(out_dir, f"{year}_average_pops_top3.csv"))

    total_top = analyze_average_pop_multi_years(place_id, start_year=2019, current_year=year, top3=False)
    if not total_top.empty:
        total_top.to_csv(os.path.join(out_dir, "total_average_pops.csv"))

    total_top3 = analyze_average_pop_multi_years(place_id, start_year=2019, current_year=year, top3=True)
    if not total_top3.empty:
        total_top3.to_csv(os.path.join(out_dir, "total_average_pops_top3.csv"))


def get_annual_average_pops_csv(place_id, year, top3=False):
    """data/race_info/average_pops/{place}/{year}_average_pops[_top3].csv を取得する"""
    suffix = "_top3" if top3 else ""
    path = os.path.join(AVERAGE_POPS_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_average_pops{suffix}.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_total_average_pops_csv(place_id, top3=False):
    """data/race_info/average_pops/{place}/total_average_pops[_top3].csv を取得する"""
    suffix = "_top3" if top3 else ""
    path = os.path.join(AVERAGE_POPS_DATA_PATH, PLACE_LIST[place_id - 1], f"total_average_pops{suffix}.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def update_winners_weight(place_id, year):
    """指定の開催場・年について、勝ち馬の平均馬体重の集計結果を更新する"""
    out_dir = os.path.join(AVERAGE_WEIGHTS_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)

    result_winner = analyze_winner_weights(place_id, year)
    if not result_winner.empty:
        result_winner.to_csv(os.path.join(out_dir, f"{year}_winner_weight.csv"))

    total_winner = analyze_winner_weights_multi_years(place_id, start_year=2019, current_year=year)
    if not total_winner.empty:
        total_winner.to_csv(os.path.join(out_dir, "total_winner_weight.csv"))


def get_annual_winner_weight_csv(place_id, year):
    """data/race_info/average_weights/{place}/{year}_winner_weight.csv を取得する"""
    path = os.path.join(AVERAGE_WEIGHTS_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_winner_weight.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_total_winner_weight_csv(place_id):
    """data/race_info/average_weights/{place}/total_winner_weight.csv を取得する"""
    path = os.path.join(AVERAGE_WEIGHTS_DATA_PATH, PLACE_LIST[place_id - 1], "total_winner_weight.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def update_average_frame_and_horse(place_id, year):
    """指定の開催場・年について、枠番・馬番の集計結果を更新する"""
    out_dir = os.path.join(AVERAGE_FRAMES_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)

    result_top = analyze_average_frame_and_horse(place_id, year)
    if not result_top.empty:
        result_top.to_csv(os.path.join(out_dir, f"{year}_average_frames.csv"))

    result_top3 = analyze_average_frame_and_horse_top3(place_id, year)
    if not result_top3.empty:
        result_top3.to_csv(os.path.join(out_dir, f"{year}_average_frames_top3.csv"))

    total_top = analyze_frame_and_horse_multi_years(place_id, start_year=2019, current_year=year)
    if not total_top.empty:
        total_top.to_csv(os.path.join(out_dir, "total_average_frames.csv"))

    total_top3 = analyze_frame_and_horse_top3_multi_years(place_id, start_year=2019, current_year=year)
    if not total_top3.empty:
        total_top3.to_csv(os.path.join(out_dir, "total_average_frames_top3.csv"))


def get_annual_average_frames_csv(place_id, year, top3=False):
    """data/race_info/average_frames/{place}/{year}_average_frames[_top3].csv を取得する"""
    suffix = "_top3" if top3 else ""
    path = os.path.join(AVERAGE_FRAMES_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_average_frames{suffix}.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_total_average_frames_csv(place_id, top3=False):
    """data/race_info/average_frames/{place}/total_average_frames[_top3].csv を取得する"""
    suffix = "_top3" if top3 else ""
    path = os.path.join(AVERAGE_FRAMES_DATA_PATH, PLACE_LIST[place_id - 1], f"total_average_frames{suffix}.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


# --- 勝ち馬の上り/通過 --------------------------------------------------------


def analyze_winners(place_id, year):
    """勝ち馬の「上り」と「通過1〜4」を race_type, course_len, ground_state, class ごとに算出する"""
    df_raw = _get_race_results_with_race_id(place_id, year)
    if df_raw.empty:
        return pd.DataFrame()
    return transform.analyze_winners(df_raw, COURSE_LISTS[place_id - 1])


def analyze_winners_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """各年度（start_year〜current_year）について勝ち馬の上り/通過を算出し、全期間の平均を返す"""
    results_by_year = {}
    for year in range(start_year, current_year + 1):
        df_year = analyze_winners(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year
    return transform.aggregate_winner_times(results_by_year)


def update_winner_time(place_id, year):
    """指定の開催場・年について、勝ち馬の上り/通過の集計結果を更新する"""
    out_dir = os.path.join(AVERAGE_TIMES_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)

    result = analyze_winners(place_id, year)
    if not result.empty:
        result.to_csv(os.path.join(out_dir, f"{year}_winner_time.csv"))

    total_df = analyze_winners_multi_years(place_id, start_year=2019, current_year=year)
    if not total_df.empty:
        total_df.to_csv(os.path.join(out_dir, "total_winner_time.csv"))


def get_annual_winner_time_csv(place_id, year):
    """data/race_info/average_times/{place}/{year}_winner_time.csv を取得する"""
    path = os.path.join(AVERAGE_TIMES_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_winner_time.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_total_winner_time_csv(place_id):
    """data/race_info/average_times/{place}/total_winner_time.csv を取得する"""
    path = os.path.join(AVERAGE_TIMES_DATA_PATH, PLACE_LIST[place_id - 1], "total_winner_time.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


# --- 平均タイム ---------------------------------------------------------------


def get_annual_average_time_csv(place_id, year):
    """data/race_info/average_times/{place}/{year}_avg_time.csv を取得する"""
    path = os.path.join(AVERAGE_TIMES_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_avg_time.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_total_average_time_csv(place_id):
    """data/race_info/average_times/{place}/total_avg_time.csv を取得する"""
    path = os.path.join(AVERAGE_TIMES_DATA_PATH, PLACE_LIST[place_id - 1], "total_avg_time.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def get_time_diff(race_time, course_info):
    """走破タイムと平均タイムとの差を算出する

    Args:
        race_time (float): 走破タイム(msec)
        course_info (list): [place_id, race_type, course_len, ground_state, race_class]

    Returns:
        list: [time_diff, time_diff_class]（同コースの全体平均との差・同クラス平均との差）
    """
    place_id, race_type, course_len, ground_state, race_class = course_info[:5]

    df_time = get_total_average_time_csv(place_id)
    if df_time.empty:
        return [0, 0]

    df_time = df_time[df_time["race_type"] == race_type]
    df_time = df_time[df_time["course_len"] == course_len]
    df_time = df_time[df_time["ground_state"] == ground_state]

    if df_time.empty:
        return [0, 0]

    avg_time = df_time[df_time["class"] == "all"].loc[:, ["avg_time"]].reset_index(drop=True).at[0, "avg_time"]
    time_diff = average_calculator.calc_time_diff(avg_time, race_time)

    avg_time_class = df_time[df_time["class"] == race_class].loc[:, ["avg_time"]].reset_index(drop=True).at[0, "avg_time"]
    time_diff_class = average_calculator.calc_time_diff(avg_time_class, race_time)

    return [time_diff, time_diff_class]


def update_annual_average_time(place_id, year):
    """指定の開催場・年について、平均タイムの集計結果を更新する"""
    df_race_results = race_result_dataset_manager.get_race_results_csv(place_id, year)
    if df_race_results.empty:
        return

    df_avg_time = average_calculator.make_average_time_datasets(df_race_results, COURSE_LISTS[place_id - 1])

    out_dir = os.path.join(AVERAGE_TIMES_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    df_avg_time.to_csv(os.path.join(out_dir, f"{year}_avg_time.csv"))
    df_avg_time.to_pickle(os.path.join(out_dir, f"{year}_avg_time.pickle"))


def update_total_average_time(place_id, year=date.today().year):
    """開催場について、2019年〜指定年までのrace_resultsを集計した平均タイムを更新する"""
    df_race_results_all = pd.DataFrame()
    for y in range(2019, year + 1):
        df_race_results_all = pd.concat(
            [df_race_results_all, race_result_dataset_manager.get_race_results_csv(place_id, y)]
        )

    if df_race_results_all.empty:
        return

    df_avg_time = average_calculator.make_average_time_datasets(df_race_results_all, COURSE_LISTS[place_id - 1])

    out_dir = os.path.join(AVERAGE_TIMES_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    df_avg_time.to_csv(os.path.join(out_dir, "total_avg_time.csv"))
    df_avg_time.to_pickle(os.path.join(out_dir, "total_avg_time.pickle"))


# --- 配当結果（race_returns） --------------------------------------------------


def get_race_returns_csv(place_id, year):
    """data/race_info/race_returns/{place}/{year}_race_returns.csv を取得する"""
    path = os.path.join(RACE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_race_returns.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def save_race_returns_dataset(place_id, year, race_returns_df):
    """race_returnsのDataFrameを重複排除のうえcsv/pickleに保存する"""
    if race_returns_df.empty:
        return

    race_returns_df = race_returns_df.drop_duplicates(keep="first")

    out_dir = os.path.join(RACE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    race_returns_df.to_csv(os.path.join(out_dir, f"{year}_race_returns.csv"))
    race_returns_df.to_pickle(os.path.join(out_dir, f"{year}_race_returns.pickle"))


def split_race_returns_csv(place_id, year):
    """race_returnsのcsvをrace_id（インデックス）ごとに分割して保存する"""
    df = get_race_returns_csv(place_id, year)
    if df.empty:
        return

    out_dir = os.path.join(RACE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1], str(year))
    os.makedirs(out_dir, exist_ok=True)
    for race_id, group in df.groupby(df.index):
        group.to_csv(os.path.join(out_dir, f"{race_id}.csv"))


def get_race_return_csv_for_race(race_id):
    """data/race_info/race_returns/{place}/{year}/{race_id}.csv を取得する"""
    year = str(race_id)[:4]
    place_id = int(str(race_id)[4:6])
    path = os.path.join(RACE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1], year, f"{race_id}.csv")
    df = read_csv_or_empty(path, dtype=str, index_col=0)
    if not df.empty:
        df.index.name = "race_id"
    return df


def save_race_return_for_race_id(race_id, race_returns_df):
    """1レース分の配当結果を data/race_info/race_returns/{place}/{year}/{race_id}.csv に保存する

    旧 src/RacePrediction/calc_returns.py の save_each_race_return_csv を移植したもの。

    Args:
        race_id (str): race_id
        race_returns_df (pd.DataFrame): race_idの配当結果（列はmodel.RACE_RETURNS_COLUMNS）
    """
    if race_returns_df is None or race_returns_df.empty:
        return

    place_id = int(str(race_id)[4:6])
    year = int(str(race_id)[:4])

    out_dir = os.path.join(RACE_RETURNS_DATA_PATH, PLACE_LIST[place_id - 1], str(year))
    os.makedirs(out_dir, exist_ok=True)
    race_returns_df.to_csv(os.path.join(out_dir, f"{race_id}.csv"))
