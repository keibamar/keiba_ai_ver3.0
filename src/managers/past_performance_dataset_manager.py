"""past_performance（過去成績）データセットのManager層

CSVの読み込み・書き込み・永続化を担う。
変換ロジック自体は src/datasets/horse/transform.py に切り出している。

旧 src/legacy_datasets/past_performance.py からの移植。
過去成績の作成方式は data/race_result（race_resultドメイン）から組み立てる方式
（旧実装の make_past_performance_dataset_from_race_results / update_past_performance系）
に統一し、netkeiba.comの馬個別ページを直接スクレイピングする旧方式は移植していない。
"""

import os

import numpy as np
import pandas as pd

from src.config import paths
from src.config.constants import NAME_LIST
from src.datasets.horse import model, transform
from src.managers import race_result_dataset_manager
from src.utils.file_utils import read_csv_or_empty

PAST_PERFORMANCE_DATA_PATH = os.path.join(paths.HORSE_DATA_PATH, "past_performance")


def get_past_performance_dataset(horse_id):
    """past_performanceのデータセットを取得する

    Args:
        horse_id (str): horse_id

    Returns:
        pd.DataFrame: past_performanceデータセット（存在しない場合は空のDataFrame）
    """
    path = os.path.join(PAST_PERFORMANCE_DATA_PATH, f"{horse_id}.csv")
    return read_csv_or_empty(path, dtype=str, index_col=0)


def save_past_performance_dataset(horse_id, past_performance_df):
    """past_performanceのDataFrameを保存する

    Args:
        horse_id (str): horse_id
        past_performance_df (pd.DataFrame): past_performanceのデータセット
    """
    if past_performance_df is None or past_performance_df.empty:
        return

    os.makedirs(PAST_PERFORMANCE_DATA_PATH, exist_ok=True)
    past_performance_df.to_csv(os.path.join(PAST_PERFORMANCE_DATA_PATH, f"{horse_id}.csv"))
    past_performance_df.to_pickle(os.path.join(PAST_PERFORMANCE_DATA_PATH, f"{horse_id}.pickle"))


def get_horse_id_from_race_id(race_id):
    """race_idから出走馬のhorse_idを取得する

    Args:
        race_id (str): race_id

    Returns:
        list: horse_idのリスト
    """
    year = str(race_id)[:4]
    place_id = int(str(race_id)[4:6])

    df = race_result_dataset_manager.get_race_results_csv(place_id, year)
    if df.empty:
        return []

    df.index = df.index.astype(str)
    df = df[race_id:race_id]
    return df.loc[:, "horse_id"].to_list()


def get_horse_id_list_from_race_id_list(race_id_list):
    """race_id_listから出走馬のhorse_idを取得する

    Args:
        race_id_list (list): race_idのリスト

    Returns:
        list: horse_idのリスト
    """
    horse_id_list = []
    try:
        for race_id in race_id_list:
            horse_id_list.append(get_horse_id_from_race_id(race_id))
        return sum(horse_id_list, [])
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        return horse_id_list


def make_past_performance_dataset_from_race_results(horse_id):
    """data/race_result全体からhorse_idの過去成績をpast_performance形式で作成する

    Args:
        horse_id (str): horse_id

    Returns:
        pd.DataFrame: horse_idの過去成績データセット
    """
    all_race_df = []
    base_dir = paths.RACE_RESULT_DATA_PATH

    # --- race_result全体を走査して全レース情報を収集 ---
    for place in sorted(os.listdir(base_dir)):
        place_path = os.path.join(base_dir, place)
        if not os.path.isdir(place_path):
            continue

        for file in sorted(os.listdir(place_path)):
            if not file.endswith("_race_results.csv"):
                continue
            csv_path = os.path.join(place_path, file)
            try:
                df = pd.read_csv(csv_path, dtype=str)
                if "horse_id" not in df.columns:
                    continue
                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "race_id"})
                df["place_id"] = place
                all_race_df.append(df)
            except Exception as e:
                print(f"読み込みエラー: {csv_path}: {e}")

    if not all_race_df:
        print("race_resultが見つかりません。")
        return pd.DataFrame()

    full_df = pd.concat(all_race_df, ignore_index=True)
    full_df = full_df.fillna("")

    # 対象馬のみ抽出
    df = full_df[full_df["horse_id"] == str(horse_id)].copy()
    if df.empty:
        print(f"horse_id={horse_id} のデータが見つかりません。")
        return pd.DataFrame()

    # --- race_idごとに他馬データを参照して派生情報を生成 ---
    records = []
    for _, row in df.iterrows():
        race_id = row["race_id"]
        same_race = full_df[full_df["race_id"] == race_id].copy()

        # 除外・取消は頭数から除外（中止は残す）
        def valid_starter(r):
            txt = str(r["着順"])
            if any(w in txt for w in ["除", "取", "取消"]):
                return False
            return True

        valid_starters = same_race[same_race.apply(valid_starter, axis=1)]
        headcount = len(valid_starters)

        # 勝ち馬・2着馬
        first = valid_starters[valid_starters["着順"] == "1"]
        second = valid_starters[valid_starters["着順"] == "2"]
        first_name = first.iloc[0]["馬名"] if not first.empty else ""
        second_name = second.iloc[0]["馬名"] if not second.empty else ""

        # --- タイム差計算処理 ---
        if any(x in str(row.get("着順", "")) for x in ["除", "取", "中", "失"]) or not row.get("タイム"):
            # 中止レース、またはタイム未記録 → 着差はNaN
            margin = np.nan
        else:
            this_time = transform.to_seconds(row.get("タイム", ""))
            first_time = transform.to_seconds(first.iloc[0]["タイム"]) if not first.empty else np.nan
            second_time = transform.to_seconds(second.iloc[0]["タイム"]) if not second.empty else np.nan

            margin = ""
            if not np.isnan(this_time) and not np.isnan(first_time):
                diff = round(this_time - first_time, 1)
                if str(row["着順"]) == "1":
                    # 1着の場合：2着との差をマイナス値で表示
                    if not np.isnan(second_time):
                        diff_to_second = round(second_time - this_time, 1)
                        margin = f"-{diff_to_second:.1f}" if diff_to_second > 0 else "-0.0"
                    else:
                        margin = "-0.0"
                else:
                    margin = f"{diff:.1f}" if diff >= 0 else ""

        # 勝ち馬欄
        if str(row["着順"]) == "1":
            winner_text = f"({second_name})" if second_name else ""
        else:
            winner_text = first_name

        # --- 統一フォーマット化 ---
        record = {
            "race_id": race_id,
            "日付": row["date"].replace("年", "/").replace("月", "/").replace("日", ""),
            "開催": NAME_LIST[int(row["place_id"][:2]) - 1] if str(row["place_id"][:2]).isdigit() else "",
            "天気": row.get("weather", ""),
            "R": str(race_id)[-2:],
            "レース名": "",  # 後でrace_name_mapで補完
            "class": row.get("class", ""),
            "頭数": headcount,
            "枠番": row.get("枠番", ""),
            "馬番": row.get("馬番", ""),
            "オッズ": row.get("単勝", ""),
            "人気": row.get("人気", ""),
            "着順": row.get("着順", ""),
            "騎手": row.get("騎手", ""),
            "斤量": row.get("斤量", ""),
            "race_type": row.get("race_type", ""),
            "course_len": row.get("course_len", ""),
            "ground_state": row.get("ground_state", ""),
            "タイム": row.get("タイム", ""),
            "着差": margin,
            "通過": row.get("通過", ""),
            "上り": row.get("上り", ""),
            "馬体重": row.get("馬体重", ""),
            "勝ち馬 (2着馬)": winner_text,
        }
        records.append(record)

    result_df = pd.DataFrame(records)

    # --- レース名補完 ---
    race_name_map = {}
    race_time_dir = paths.RACE_TIME_ID_LIST_PATH
    for file in os.listdir(race_time_dir):
        if not file.endswith(".csv"):
            continue
        try:
            df_info = pd.read_csv(os.path.join(race_time_dir, file), dtype=str)
            for _, r in df_info.iterrows():
                race_name_map[str(r["race_id"])] = str(r.get("race_name", ""))
        except Exception:
            continue
    result_df["レース名"] = result_df["race_id"].map(race_name_map).fillna("")

    # --- カラム整列 ---
    for c in model.PAST_PERFORMANCE_COLUMNS:
        if c not in result_df.columns:
            result_df[c] = ""
    result_df = result_df[model.PAST_PERFORMANCE_COLUMNS]

    # 重複レースの削除
    result_df = result_df.drop_duplicates(subset="race_id", keep="last")
    # --- ソート（日付新→古） ---
    try:
        result_df["日付_dt"] = pd.to_datetime(result_df["日付"], errors="coerce")
        result_df = result_df.sort_values("日付_dt", ascending=False).drop(columns=["日付_dt"])
    except Exception:
        pass

    return result_df.reset_index(drop=True)


def update_past_performance(horse_id):
    """horse_idの過去成績をdata/race_resultから再構築し、既存データと統合して保存する

    Args:
        horse_id (str): horse_id

    Returns:
        pd.DataFrame: 更新後のpast_performanceデータセット
    """
    past_path = os.path.join(PAST_PERFORMANCE_DATA_PATH, f"{horse_id}.csv")
    if os.path.exists(past_path):
        past_df = pd.read_csv(past_path, dtype=str)
    else:
        past_df = pd.DataFrame()

    # --- 両方を共通フォーマットに揃える ---
    existing_df_norm = transform.normalize_past_performance_format(past_df)

    # --- 追加データをマージ ---
    add_df = make_past_performance_dataset_from_race_results(horse_id)
    if not add_df.empty:
        updated_df = pd.concat([existing_df_norm, add_df], ignore_index=True)
        updated_df = updated_df.drop_duplicates(subset=["race_id"], keep="first")
    else:
        updated_df = existing_df_norm

    # --- 日付順に並べ替え ---
    if "日付" in updated_df.columns:
        updated_df["日付"] = pd.to_datetime(updated_df["日付"], errors="coerce")
        updated_df = updated_df.sort_values("日付", ascending=False).reset_index(drop=True)

    # 日付を正規化（例：2025-09-07 → 2025/09/07）
    updated_df["日付"] = pd.to_datetime(updated_df["日付"], errors="coerce").dt.strftime("%Y/%m/%d")

    # クラス表記を正規化
    updated_df["class"] = updated_df["class"].apply(transform.normalize_class_text)

    # レース名のクラス表記を削除
    if "レース名" in updated_df.columns:
        updated_df["レース名"] = updated_df["レース名"].apply(transform.clean_race_name)

    # --- 列の並び統一 ---
    for col in model.PAST_PERFORMANCE_COLUMNS:
        if col not in updated_df.columns:
            updated_df[col] = pd.NA
    updated_df = updated_df[model.PAST_PERFORMANCE_COLUMNS]

    # --- 保存 ---
    os.makedirs(PAST_PERFORMANCE_DATA_PATH, exist_ok=True)
    updated_df.to_csv(past_path, index=False, encoding="utf-8-sig")

    return updated_df


def get_past_race_info(horse_id, race_id, race_num):
    """当該レースより過去の指定レース数を取得する

    Args:
        horse_id (str): horse_id
        race_id (str): race_id
        race_num (int): 抽出するレース数

    Returns:
        pd.DataFrame: 指定レース数の過去レース結果
    """
    try:
        horse_result = get_past_performance_dataset(horse_id)
        if horse_result.empty:
            horse_results_df = make_past_performance_dataset_from_race_results(str(horse_id))
            save_past_performance_dataset(str(horse_id), horse_results_df)
            horse_result = get_past_performance_dataset(horse_id)

        horse_result = transform.reset_horse_result(horse_result, race_id)

        if len(horse_result.index) > race_num:
            horse_result = horse_result[0:race_num]

        return horse_result
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
        return pd.DataFrame()
