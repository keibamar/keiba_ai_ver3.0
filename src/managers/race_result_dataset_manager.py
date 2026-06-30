"""race_result（レース結果）データセットのManager層

CSVの読み込み・書き込み・永続化を担う。
変換ロジック自体は src/datasets/race_result/transform.py に切り出している。

旧 src/legacy_datasets/race_results.py からの移植。
export_race_info_per_race は race_result のデータからrace_info(data/race_info)を
出力するため、現時点ではこのManagerに置いている。
"""

import os

import pandas as pd

from src.config import paths
from src.config.constants import PLACE_LIST
from src.datasets.race_result import transform
from src.utils.file_utils import read_csv_or_empty


def get_race_results_csv(place_id, year):
    """race_resultsのcsvを取得する

    Args:
        place_id (int): 開催コースid
        year (int): 開催年

    Returns:
        pd.DataFrame: race_resultsデータセット（存在しない場合は空のDataFrame）
    """
    path = os.path.join(paths.RACE_RESULT_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_race_results.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def save_race_results_dataset(place_id, year, race_results_df):
    """race_resultsのDataFrameを保存

    Args:
        place_id (int): 開催コースid
        year (int): 開催年
        race_results_df (pd.DataFrame): race_resultデータセット
    """
    if race_results_df is None or race_results_df.empty:
        return

    race_results_df = transform.format_race_results_dataframe(race_results_df)

    place_dir = os.path.join(paths.RACE_RESULT_DATA_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(place_dir, exist_ok=True)
    race_results_df.to_csv(os.path.join(place_dir, f"{year}_race_results.csv"))
    race_results_df.to_pickle(os.path.join(place_dir, f"{year}_race_results.pickle"))


def save_race_result_for_race_id(race_id, df_results):
    """1レース分のレース結果を data/race_result/{place}/{year}/{race_id}.csv に保存する

    split_race_results_by_yearの出力先と同じ構成のper-race CSV。
    旧 src/RacePrediction/daily_race_results.py の save_each_race_result_csv を移植したもの。

    race.netkeiba.comの速報ページ（scrape_day_race_result）にはrace_type/course_len/
    ground_state/classが含まれないため、これらは本来週次バッチ（年間まとめファイルへの
    統合時）で初めて付与されていた。レースカード作成時に保存済みのレース情報
    （race_card_dataset_manager.get_race_info_csv）があれば、ここで先に付与しておく
    ことで、週次バッチを待たずに開催別成績ページ等にコース・馬場・クラスが反映される
    ようにする（無ければ何もしない＝従来通り週次バッチでの統合を待つ）。

    Args:
        race_id (str): race_id
        df_results (pd.DataFrame): race_idのレース結果
    """
    if df_results is None or df_results.empty:
        return

    from src.managers import race_card_dataset_manager

    cached_info_df = race_card_dataset_manager.get_race_info_csv(race_id)
    if not cached_info_df.empty:
        for col in ("race_type", "course_len", "ground_state", "class"):
            if col in cached_info_df.columns:
                df_results[col] = cached_info_df.iloc[0][col]

    place_id = int(str(race_id)[4:6])
    year = int(str(race_id)[:4])

    out_dir = os.path.join(paths.RACE_RESULT_DATA_PATH, PLACE_LIST[place_id - 1], str(year))
    os.makedirs(out_dir, exist_ok=True)
    df_results.to_csv(os.path.join(out_dir, f"{race_id}.csv"))


def get_race_id_result(race_id):
    """race_idのレース結果を取得

    data/race_result/{place}/{year}_race_results.csv（週次バッチ更新）に
    まだ反映されていない直近のレース（save_race_result_for_race_idが書き込む
    data/race_result/{place}/{year}/{race_id}.csvの方が新しい）の場合は、
    そちらにフォールバックする（出馬表ページに結果が反映されないまま週次更新を
    待ってしまう不具合の対策）。

    Args:
        race_id (str): race_id

    Returns:
        pd.DataFrame: race_idのレース結果
    """
    year = int(str(race_id)[:4])
    place_id = int(str(race_id)[4:6])

    df_results = get_race_results_csv(place_id, year)
    df_results.index = df_results.index.astype(str)
    df_result = df_results[race_id:race_id]
    df_result = df_result.reset_index(drop=True)

    if df_result.empty:
        per_race_path = os.path.join(paths.RACE_RESULT_DATA_PATH, PLACE_LIST[place_id - 1], str(year), f"{race_id}.csv")
        df_result = read_csv_or_empty(per_race_path, dtype=str, index_col=0).reset_index(drop=True)

    return df_result


def split_race_results_by_year(place_id, year):
    """place_idのrace_resultsを指定年でレース単位に分割して保存する

    Args:
        place_id (int): 開催コースid
        year (int): 開催年
    """
    df = get_race_results_csv(place_id, year)

    output_dir = os.path.join(paths.RACE_RESULT_DATA_PATH, PLACE_LIST[place_id - 1], str(year))
    os.makedirs(output_dir, exist_ok=True)

    for race_id, group in df.groupby(df.index):
        try:
            output_path = os.path.join(output_dir, f"{race_id}.csv")
            group = group.drop_duplicates(subset="馬名", keep="first")
            group.to_csv(output_path, index=True)
        except Exception:
            print("レース結果の分割に失敗しました。", race_id)
    print(year, PLACE_LIST[place_id - 1], "CompleteResultsSplit")


def convert_course_len_csv(file_path):
    """指定したCSVファイルのcourse_len列を整数型に変換して上書き保存する

    Args:
        file_path (str): 対象のCSVファイルパス

    Returns:
        bool: 変換と保存が成功したらTrue、ファイルが存在しない・列がない等の場合はFalse
    """
    try:
        if not os.path.isfile(file_path):
            print("convert_course_len_csv: file not exists:", file_path)
            return False

        df = pd.read_csv(file_path, dtype=str)
        if "course_len" not in df.columns:
            return False

        df["course_len"] = df["course_len"].apply(transform.to_int)

        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        return False


def export_race_info_per_race(place_id, year):
    """各race_idごとに基本情報(race_type, course_len, weather, ground_state, class)を抽出して
    data/race_info/{place}/{year}/{race_id}.csv に保存する

    Args:
        place_id (int): 開催コースid
        year (int): 開催年
    """
    input_csv = os.path.join(paths.RACE_RESULT_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_race_results.csv")

    if not os.path.exists(input_csv):
        print("not file exists:", input_csv)
        return
    df = pd.read_csv(input_csv, dtype=str)
    if df.empty:
        print("入力CSVが空です。処理を終了します。")
        return

    out_dir = os.path.join(paths.RACE_INFO_DATA_PATH, PLACE_LIST[place_id - 1], str(year))
    os.makedirs(out_dir, exist_ok=True)

    # --- race_id 列の特定 ---
    if "race_id" in df.columns:
        pass
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "race_id"})
    else:
        df = df.reset_index().rename(columns={"index": "race_id"})
    df["race_id"] = df.index if "race_id" not in df.columns else df["race_id"]

    unique_race_ids = df["race_id"].unique()

    for race_id in unique_race_ids:
        sub = df[df["race_id"] == race_id].iloc[0]

        out_path = os.path.join(out_dir, f"{race_id}.csv")

        record = {
            "race_type": sub.get("race_type", ""),
            "course_len": sub.get("course_len", ""),
            "weather": sub.get("weather", ""),
            "ground_state": sub.get("ground_state", ""),
            "class": sub.get("class", ""),
        }

        out_df = pd.DataFrame([record])
        try:
            out_df["course_len"] = int(out_df["course_len"].astype(float))
        except Exception:
            out_df["course_len"] = out_df["course_len"]

        out_df.to_csv(out_path, index_label="", encoding="utf-8-sig")

    print(PLACE_LIST[place_id - 1], str(year), "レース情報出力完了")
