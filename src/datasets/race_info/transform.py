"""race_infoデータセットの変換ロジック（純粋関数のみ・I/Oなし）

旧 src/legacy_datasets/analysis_race_info.py, analysis_race_time.py, average_time.py
の集計ロジックを移植したもの。
race_resultsのDataFrame（"race_id"列を持つ、index_col=0でCSVを読み込んでreset_indexしたもの）
を入力として、各種集計結果のDataFrameを返す。
"""

import datetime
import re

import numpy as np
import pandas as pd

from src.config.constants import GROUND_STATE_LIST
from src.datasets.race_info import model


def _sort_and_reindex(df_result, columns):
    df_result["race_type"] = pd.Categorical(df_result["race_type"], categories=["芝", "ダート"], ordered=True)
    df_result["class"] = pd.Categorical(df_result["class"], categories=model.CLASSES, ordered=True)
    df_result["ground_state"] = pd.Categorical(df_result["ground_state"], categories=model.GROUNDS, ordered=True)
    df_result = df_result.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)
    return df_result.reindex(columns=columns)


# --- 勝ち馬の平均馬体重 -----------------------------------------------------


def analyze_winner_weights(df_raw, courses):
    """勝ち馬の「馬体重」平均を race_type, course_len, ground_state, class ごとに算出する"""
    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")
    df["馬体重"] = pd.to_numeric(df["馬体重"], errors="coerce")

    winners = df[df["着順"] == 1].copy()
    if winners.empty:
        return pd.DataFrame()

    all_results = []
    for race_type, course_len in courses:
        base_data = winners[
            (winners["race_type"] == race_type) & (winners["course_len"] == float(course_len))
        ]

        for cls in model.CLASSES:
            for grd in model.GROUNDS:
                tmp = base_data.copy()
                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                avg_weight = tmp["馬体重"].mean() if not tmp.empty else None

                all_results.append({
                    "race_type": race_type,
                    "course_len": int(course_len),
                    "ground_state": grd,
                    "class": cls,
                    "馬体重": round(avg_weight, 1) if avg_weight is not None else None,
                })

    df_result = pd.DataFrame(all_results).round(1)
    return _sort_and_reindex(df_result, model.WINNER_WEIGHT_COLUMNS)


def aggregate_winner_weights(results_by_year):
    """年度別の analyze_winner_weights 結果を結合し、全期間の平均を算出する"""
    if not results_by_year:
        return pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    group_cols = ["race_type", "course_len", "ground_state", "class"]
    total_df = (
        combined_df.groupby(group_cols, dropna=False)["馬体重"]
        .mean()
        .round(1)
        .reset_index()
    )
    return _sort_and_reindex(total_df, model.WINNER_WEIGHT_COLUMNS)


# --- 人気 -------------------------------------------------------------------


def analyze_average_pops(df_raw, courses, top3=False):
    """勝ち馬または3着内馬の平均人気と人気別勝利数を集計する

    top3=True の場合は3着内を対象。
    """
    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["人気"] = pd.to_numeric(df["人気"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    # 出走頭数を算出（race_id単位）
    df["頭数"] = df.groupby("race_id")["馬番"].transform("count")

    all_results = []
    for race_type, course_len in courses:
        base_data = df[(df["race_type"] == race_type) & (df["course_len"] == float(course_len))]

        for cls in model.CLASSES:
            for grd in model.GROUNDS:
                tmp = base_data.copy()
                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                if top3:
                    tmp = tmp[tmp["着順"].isin([1, 2, 3])]
                else:
                    tmp = tmp[tmp["着順"] == 1]

                if tmp.empty:
                    result = {
                        "race_type": race_type,
                        "course_len": int(course_len),
                        "ground_state": grd,
                        "class": cls,
                        "avg_pop": None,
                    }
                    for i in range(1, 19):
                        result[f"pop_{i}_count"] = 0
                    all_results.append(result)
                    continue

                # 平均人気（18頭立て換算）
                tmp["norm_pop"] = tmp["人気"] * (18 / tmp["頭数"])
                avg_pop = tmp["norm_pop"].mean()

                # 人気別勝利数カウント
                pop_counts = tmp["人気"].value_counts().to_dict()

                result = {
                    "race_type": race_type,
                    "course_len": int(course_len),
                    "ground_state": grd,
                    "class": cls,
                    "avg_pop": round(avg_pop, 2) if avg_pop else None,
                }
                for i in range(1, 19):
                    result[f"pop_{i}_count"] = int(pop_counts.get(i, 0))

                all_results.append(result)

    df_result = pd.DataFrame(all_results)
    return _sort_and_reindex(df_result, model.AVERAGE_POPS_COLUMNS)


def aggregate_average_pops(results_by_year):
    """年度別の analyze_average_pops 結果を結合し、全期間の平均・合計を算出する"""
    if not results_by_year:
        return pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    group_cols = ["race_type", "course_len", "ground_state", "class"]

    avg_df = (
        combined_df.groupby(group_cols, dropna=False)["avg_pop"]
        .mean()
        .round(2)
        .reset_index()
    )

    pop_cols = [f"pop_{i}_count" for i in range(1, 19)]
    sum_df = (
        combined_df.groupby(group_cols, dropna=False)[pop_cols]
        .sum()
        .reset_index()
    )

    total_df = pd.merge(avg_df, sum_df, on=group_cols, how="left")
    return _sort_and_reindex(total_df, group_cols + ["avg_pop"] + pop_cols)


# --- 枠番・馬番（勝ち馬） -----------------------------------------------------


def _empty_frame_and_horse_record(race_type, course_len, grd, cls):
    """勝ち馬データがない場合の空データ行のテンプレート"""
    record = {
        "race_type": race_type,
        "course_len": int(course_len),
        "ground_state": grd,
        "class": cls,
        "avg_frame": None,
        "avg_horse": None,
        "total_winners": 0,
    }
    for i in range(1, 9):
        record[f"frame_{i}_wins"] = 0
    for j in range(1, 19):
        record[f"horse_{j}_wins"] = 0
    return record


def analyze_average_frame_and_horse(df_raw, courses):
    """勝ち馬の「平均枠番」「平均馬番」「勝利数」および
    各枠・馬番の勝ち数を race_type, course_len, ground_state, class ごとに算出する
    """
    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    all_results = []
    for race_type, course_len in courses:
        for cls in model.CLASSES:
            for grd in model.GROUNDS:
                tmp = df[(df["race_type"] == race_type) & (df["course_len"] == float(course_len))]

                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                if tmp.empty:
                    all_results.append(_empty_frame_and_horse_record(race_type, course_len, grd, cls))
                    continue

                winners = tmp[tmp["着順"] == 1].copy()
                if winners.empty:
                    all_results.append(_empty_frame_and_horse_record(race_type, course_len, grd, cls))
                    continue

                avg_frame = winners["枠番"].mean()
                avg_horse = winners["馬番"].mean()
                frame_counts = winners["枠番"].value_counts().reindex(range(1, 9), fill_value=0)
                horse_counts = winners["馬番"].value_counts().reindex(range(1, 19), fill_value=0)

                result = {
                    "race_type": race_type,
                    "course_len": int(course_len),
                    "ground_state": grd,
                    "class": cls,
                    "avg_frame": round(avg_frame, 2) if not pd.isna(avg_frame) else None,
                    "avg_horse": round(avg_horse, 2) if not pd.isna(avg_horse) else None,
                    "total_winners": len(winners),
                }
                for i in range(1, 9):
                    result[f"frame_{i}_wins"] = frame_counts[i]
                for j in range(1, 19):
                    result[f"horse_{j}_wins"] = horse_counts[j]

                all_results.append(result)

    df_result = pd.DataFrame(all_results)
    return _sort_and_reindex(df_result, model.AVERAGE_FRAMES_COLUMNS)


def aggregate_average_frame_and_horse(results_by_year):
    """年度別の analyze_average_frame_and_horse 結果を結合し、全期間の平均・合計を算出する"""
    if not results_by_year:
        return pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    group_cols = ["race_type", "course_len", "ground_state", "class"]
    mean_cols = ["avg_frame", "avg_horse"]
    sum_cols = [c for c in combined_df.columns if c.startswith("frame_") or c.startswith("horse_") or c == "total_winners"]

    agg_dict = {c: "mean" for c in mean_cols}
    agg_dict.update({c: "sum" for c in sum_cols})

    total_df = (
        combined_df.groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .round(2)
        .reset_index()
    )

    total_df["race_type"] = pd.Categorical(total_df["race_type"], categories=["芝", "ダート"], ordered=True)
    total_df["class"] = pd.Categorical(total_df["class"], categories=model.CLASSES, ordered=True)
    total_df["ground_state"] = pd.Categorical(total_df["ground_state"], categories=model.GROUNDS, ordered=True)

    return total_df.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)


# --- 枠番・馬番（3着内） ------------------------------------------------------


def make_empty_record(race_type, course_len, grd, cls):
    """3着内集計で対象データがない場合の空データ行のテンプレート"""
    record = {
        "race_type": race_type,
        "course_len": int(course_len),
        "ground_state": grd,
        "class": cls,
        "avg_frame": None,
        "avg_horse": None,
        "total_top3": 0,
    }
    for i in range(1, 9):
        record[f"frame_{i}_top3"] = 0
    for j in range(1, 19):
        record[f"horse_{j}_top3"] = 0
    return record


def analyze_average_frame_and_horse_top3(df_raw, courses):
    """3着以内馬の「平均枠番」「平均馬番」「3着内馬数」および
    各枠・馬番ごとの3着内回数を race_type, course_len, ground_state, class ごとに算出する
    """
    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    all_results = []
    for race_type, course_len in courses:
        for cls in model.CLASSES:
            for grd in model.GROUNDS:
                tmp = df[(df["race_type"] == race_type) & (df["course_len"] == float(course_len))]

                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                if tmp.empty:
                    all_results.append(make_empty_record(race_type, course_len, grd, cls))
                    continue

                top3 = tmp[tmp["着順"] <= 3].copy()
                if top3.empty:
                    all_results.append(make_empty_record(race_type, course_len, grd, cls))
                    continue

                avg_frame = top3["枠番"].mean()
                avg_horse = top3["馬番"].mean()
                frame_counts = top3["枠番"].value_counts().reindex(range(1, 9), fill_value=0)
                horse_counts = top3["馬番"].value_counts().reindex(range(1, 19), fill_value=0)

                result = {
                    "race_type": race_type,
                    "course_len": int(course_len),
                    "ground_state": grd,
                    "class": cls,
                    "avg_frame": round(avg_frame, 2) if not pd.isna(avg_frame) else None,
                    "avg_horse": round(avg_horse, 2) if not pd.isna(avg_horse) else None,
                    "total_top3": len(top3),
                }
                for i in range(1, 9):
                    result[f"frame_{i}_top3"] = frame_counts[i]
                for j in range(1, 19):
                    result[f"horse_{j}_top3"] = horse_counts[j]

                all_results.append(result)

    df_result = pd.DataFrame(all_results)
    return _sort_and_reindex(df_result, model.AVERAGE_FRAMES_TOP3_COLUMNS)


def aggregate_average_frame_and_horse_top3(results_by_year):
    """年度別の analyze_average_frame_and_horse_top3 結果を結合し、全期間の平均・合計を算出する"""
    if not results_by_year:
        return pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    group_cols = ["race_type", "course_len", "ground_state", "class"]
    mean_cols = ["avg_frame", "avg_horse"]
    sum_cols = [c for c in combined_df.columns if c.startswith("frame_") or c.startswith("horse_") or c == "total_top3"]

    agg_dict = {c: "mean" for c in mean_cols}
    agg_dict.update({c: "sum" for c in sum_cols})

    total_df = (
        combined_df.groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .round(2)
        .reset_index()
    )

    total_df["race_type"] = pd.Categorical(total_df["race_type"], categories=["芝", "ダート"], ordered=True)
    total_df["class"] = pd.Categorical(total_df["class"], categories=model.CLASSES, ordered=True)
    total_df["ground_state"] = pd.Categorical(total_df["ground_state"], categories=model.GROUNDS, ordered=True)

    return total_df.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)


# --- horse_id_map ------------------------------------------------------------


def merge_horse_id_map(existing_df, new_df):
    """既存の horse_id_map と新規データをマージし、horse_id重複を除去してソートする"""
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["horse_id"], keep="first")
    return merged_df.sort_values(by="horse_id").reset_index(drop=True)


def add_horse_id_map_entry(existing_df, horse_id, horse_name):
    """horse_id_map に1件追加する

    horse_idまたは馬名が空、もしくは既存データと重複する場合はNoneを返す（変更なし）。
    追加可能な場合は horse_id 順にソートした新しいDataFrameを返す。
    """
    horse_id = str(horse_id).strip()
    horse_name = str(horse_name).strip()

    if horse_id == "" or horse_name == "":
        return None

    id_exists = horse_id in existing_df["horse_id"].values
    name_exists = horse_name in existing_df["馬名"].values

    if id_exists or name_exists:
        return None

    new_row = pd.DataFrame([[horse_name, horse_id]], columns=model.HORSE_ID_MAP_COLUMNS)
    updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    return updated_df.sort_values(by="horse_id", ascending=True, ignore_index=True)


# --- 勝ち馬の上り/通過 --------------------------------------------------------


def _parse_passages(pass_str):
    """通過文字列から整数のリストを返す（例 "1-1-1-1" -> [1,1,1,1]）。空・NaN・不正は空リストを返す"""
    if not isinstance(pass_str, str) or pass_str.strip() == "":
        return []
    nums = re.findall(r"\d+", pass_str)
    return [int(n) for n in nums] if nums else []


def analyze_winners(df_raw, courses):
    """勝ち馬の「上り」と「通過1〜4」を race_type, course_len, ground_state, class ごとに算出する"""
    runners_count = df_raw.groupby("race_id").apply(lambda g: g[g["着順"] != "除"].shape[0]).to_dict()

    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["上り"] = pd.to_numeric(df["上り"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    winners = df[df["着順"] == 1].copy()
    if winners.empty:
        return pd.DataFrame()

    all_results = []
    for race_type, course_len in courses:
        base_data = winners[
            (winners["race_type"] == race_type) & (winners["course_len"] == float(course_len))
        ]

        for cls in model.CLASSES:
            for grd in model.GROUNDS:
                tmp = base_data.copy()
                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                avg_last = tmp["上り"].mean() if not tmp.empty else None

                stage_vals = {1: [], 2: [], 3: [], 4: []}
                for _, winner_row in tmp.iterrows():
                    race_id = str(winner_row["race_id"])
                    num_runners = runners_count.get(int(race_id), None)
                    pass_str = df_raw.loc[df_raw["race_id"] == int(race_id), "通過"]
                    pass_val = ""
                    if not pass_str.empty:
                        for v in pass_str.values:
                            if isinstance(v, str) and v.strip() != "":
                                pass_val = v
                                break
                    passages = _parse_passages(pass_val)
                    if not passages or num_runners is None or num_runners == 0:
                        continue
                    for idx, pos in enumerate(passages[:4], start=1):
                        normalized = (pos / num_runners) * 18.0
                        stage_vals[idx].append(normalized)

                avg_stage = {}
                for i in range(1, 5):
                    avg_stage[i] = round(float(np.mean(stage_vals[i])), 2) if stage_vals[i] else None

                all_results.append({
                    "race_type": race_type,
                    "course_len": int(course_len),
                    "ground_state": grd,
                    "class": cls,
                    "上り": round(avg_last, 2) if avg_last is not None else None,
                    "通過1": avg_stage[1],
                    "通過2": avg_stage[2],
                    "通過3": avg_stage[3],
                    "通過4": avg_stage[4],
                })

    df_result = pd.DataFrame(all_results).round(1)
    return _sort_and_reindex(df_result, model.WINNER_TIME_COLUMNS)


def aggregate_winner_times(results_by_year):
    """年度別の analyze_winners 結果を結合し、全期間の平均を算出する"""
    if not results_by_year:
        return pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    numeric_cols = ["上り", "通過1", "通過2", "通過3", "通過4"]
    group_cols = ["race_type", "course_len", "ground_state", "class"]
    total_df = (
        combined_df.groupby(group_cols, dropna=False)[numeric_cols]
        .mean()
        .round(1)
        .reset_index()
    )
    return _sort_and_reindex(total_df, model.WINNER_TIME_COLUMNS)


# --- 平均タイム ---------------------------------------------------------------


def extract_course_race_results(race_type, course_len, race_results_df):
    """race_resultsデータセットから該当コース・距離の勝ち馬の行を抽出する"""
    course_race_results = race_results_df[race_results_df["race_type"] == str(race_type)]
    course_race_results = course_race_results[course_race_results["course_len"] == str(course_len)]
    return course_race_results[course_race_results["着順"] == "1"]


def calc_avg_time(time_data):
    """走破時計（文字列）のSeriesから平均タイム(ms)を算出する。データが無い場合はNaTを返す"""
    if len(time_data) == 0:
        return np.timedelta64("NaT")

    time_format = "%H:%M:%S.%f"
    for i in range(len(time_data)):
        time_data[i] = datetime.datetime.strptime(time_data.iloc[i], time_format)

    time_data = time_data.astype("datetime64[ms]").to_numpy()
    base_time = np.datetime64(0, "ms")
    avg_time = ((time_data - base_time) % np.timedelta64(1, "D")).mean()
    return avg_time.astype(int)


def get_avg_time_list_from_race_results_df(df_course_race_results):
    """全馬場状態・各馬場状態（GROUND_STATE_LIST）ごとの平均タイムのリストを作成する"""
    avg_time_list = []

    all_time = df_course_race_results["タイム"].reset_index(drop=True)
    avg_time_list.append(calc_avg_time(all_time))

    for ground_state in GROUND_STATE_LIST:
        df_temp = df_course_race_results[df_course_race_results["ground_state"] == ground_state]
        time_temp = df_temp["タイム"].reset_index(drop=True)
        avg_time_list.append(calc_avg_time(time_temp))

    return avg_time_list


def make_avg_time_dataset(race_type, course_len, class_name, avg_time_list):
    """平均タイムのリストから avg_time データセットを作成する

    ground_state列は model.GROUNDS（"全"+GROUND_STATE_LIST）に対応する。
    旧実装ではこの列が ["全","良","稍重","重","不"] とハードコードされ"不良"であるべき
    箇所が"不"になっていたが、新実装ではGROUNDSに合わせて"不良"に修正している。
    """
    avg_time = pd.DataFrame({"avg_time": avg_time_list})
    course_data = pd.DataFrame({
        "race_type": [str(race_type)] * len(model.GROUNDS),
        "course_len": [str(course_len)] * len(model.GROUNDS),
        "ground_state": model.GROUNDS,
        "class": [class_name] * len(model.GROUNDS),
    })
    return pd.concat([course_data, avg_time], axis=1)


def type_check(return_df, bet_type):
    """配当結果のDataFrameに指定の式別(bet_type)の行があるかを判定する"""
    for betting_type in return_df.index:
        if str(betting_type) == bet_type:
            return True
    return False


def extract_race_return_table(tables):
    """scrape_df(url)が返すテーブル群から配当結果のテーブルを抽出する

    Args:
        tables (list[pd.DataFrame]): レース結果ページのテーブル一覧
            （tables[1]=単勝/複勝/枠連/馬連、tables[2]=ワイド/馬単/三連複/三連単）

    Returns:
        pd.DataFrame: tables[1]とtables[2]を結合した配当結果テーブル
            （取得に失敗した場合はtablesをそのまま返す）
    """
    try:
        win_place_table = tables[1]
        exotic_table = tables[2]
        return pd.concat([win_place_table, exotic_table]).reset_index(drop=True)
    except Exception:
        return tables


def format_race_return_dataframe(race_return_df):
    """配当結果のテーブルのフォーマットを整える

    各セルの半角スペースを"br"に置き換え、配当列(2)から"円"と","を取り除いたうえで、
    式別の列(0)をインデックスにする。

    Args:
        race_return_df (pd.DataFrame): extract_race_return_tableの戻り値

    Returns:
        pd.DataFrame: 式別をインデックスにした配当結果テーブル（失敗時は空のDataFrame）
    """
    try:
        for i in range(len(race_return_df.index)):
            race_return_df.at[i, 1] = race_return_df.at[i, 1].replace(" ", "br")
            race_return_df.at[i, 2] = race_return_df.at[i, 2].replace(" ", "br")
            race_return_df.at[i, 3] = race_return_df.at[i, 3].replace(" ", "br")
            race_return_df.at[i, 2] = race_return_df.at[i, 2].replace("円", "")
            race_return_df.at[i, 2] = race_return_df.at[i, 2].replace(",", "")
        return race_return_df.set_index(0)
    except Exception:
        return pd.DataFrame()


def format_type_returns_dataframe(return_df, bet_type):
    """配当結果のテーブルから、指定の式別(bet_type)の行を抽出して整形する

    旧実装では三連複・三連単の判定にアラビア数字表記("3連複"/"3連単")を使っていたため、
    src.config.constants.BETTING_TYPE_LISTの漢数字表記("三連複"/"三連単")と一致せず、
    これら2式別は常に空の結果になっていた。新実装では漢数字表記に統一して正しく解析する。

    Args:
        return_df (pd.DataFrame): format_race_return_dataframeの戻り値
        bet_type (str): 抽出する式別（src.config.constants.BETTING_TYPE_LISTの要素）

    Returns:
        pd.DataFrame: model.RACE_RETURNS_COLUMNS（式別/馬番/配当/人気）の行（複数件の場合あり）
    """
    try:
        if return_df.index.name is not None or return_df.index[0] in [
            "単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"
        ]:
            return_df = return_df.reset_index().rename(columns={"index": "式別"})

        cols = list(return_df.columns)
        if len(cols) >= 4:
            return_df.columns = ["式別", "col1", "col2", "col3"]
        else:
            return_df.columns = ["式別", "col1", "col2", "col3"][:len(cols)]

        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text)
            text = re.sub(r"<br\s*/?>", " ", text)
            text = text.replace("br", " ")
            text = text.replace("　", " ")
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        return_df = return_df.map(clean_text)

        bet_type = str(bet_type).strip()
        temp = return_df[return_df["式別"] == bet_type]
        if temp.empty:
            print(f"⚠️ 式別 '{bet_type}' が見つかりませんでした")
            return pd.DataFrame(columns=model.RACE_RETURNS_COLUMNS)

        row = temp.iloc[0, 1:].astype(str).tolist()
        row = [r for r in row if r]

        results = []

        # === 複勝 ===
        if bet_type == "複勝":
            horses = re.findall(r"\d+", row[0])
            pays = re.findall(r"\d+", row[1])
            pops = re.findall(r"\d+", row[2]) if len(row) > 2 else []
            for i in range(len(horses)):
                results.append([
                    bet_type,
                    horses[i],
                    pays[i] if i < len(pays) else None,
                    pops[i] if i < len(pops) else None
                ])

        # === ワイド ===
        elif bet_type == "ワイド":
            horses = re.findall(r"\d+", row[0])
            pays = re.findall(r"\d+", row[1])
            pops = re.findall(r"\d+", row[2]) if len(row) > 2 else []
            pairs = [f"{horses[i]}-{horses[i+1]}" for i in range(0, len(horses), 2)]
            for i in range(len(pairs)):
                results.append([
                    bet_type,
                    pairs[i],
                    pays[i] if i < len(pays) else None,
                    pops[i] if i < len(pops) else None
                ])

        # === 単勝・枠連・馬連・馬単 ===
        elif bet_type in ["単勝", "枠連", "馬連", "馬単"]:
            nums = re.findall(r"\d+", row[0])
            pays = re.findall(r"\d+", row[1])
            pops = re.findall(r"\d+", row[2]) if len(row) > 2 else []
            for i in range(len(nums) // 2 if bet_type != "単勝" else len(nums)):
                if bet_type == "単勝":
                    results.append([bet_type, nums[i], pays[i], pops[i]])
                else:
                    sep = "→" if bet_type == "馬単" else "-"
                    pair = f"{nums[i*2]}{sep}{nums[i*2+1]}"
                    results.append([bet_type, pair, pays[i], pops[i]])

        # === 三連複 ===
        elif bet_type == "三連複":
            nums = re.findall(r"\d+", row[0])
            pays = re.findall(r"\d+", row[1])
            pops = re.findall(r"\d+", row[2]) if len(row) > 2 else []
            combos = [f"{nums[i]}-{nums[i+1]}-{nums[i+2]}" for i in range(0, len(nums), 3)]
            for i, combo in enumerate(combos):
                results.append([
                    bet_type,
                    combo,
                    pays[i] if i < len(pays) else None,
                    pops[i] if i < len(pops) else None
                ])

        # === 三連単 ===
        elif bet_type == "三連単":
            nums = re.findall(r"\d+", row[0])
            pays = re.findall(r"\d+", row[1])
            pops = re.findall(r"\d+", row[2]) if len(row) > 2 else []
            combos = [f"{nums[i]}→{nums[i+1]}→{nums[i+2]}" for i in range(0, len(nums), 3)]
            for i, combo in enumerate(combos):
                results.append([
                    bet_type,
                    combo,
                    pays[i] if i < len(pays) else None,
                    pops[i] if i < len(pops) else None
                ])

        return pd.DataFrame(results, columns=model.RACE_RETURNS_COLUMNS)

    except Exception as e:
        print(f"Error in format_type_returns_dataframe({bet_type}): {repr(e)}")
        return pd.DataFrame(columns=model.RACE_RETURNS_COLUMNS)


def make_average_time_datasets(df_race_results, courses):
    """race_resultsからaverage_timeデータセットを作成する"""
    df_avg_time = pd.DataFrame()
    for race_type, course_len in courses:
        df_course = extract_course_race_results(race_type, course_len, df_race_results)

        all_avg_time = get_avg_time_list_from_race_results_df(df_course)
        df_avg_time = pd.concat([df_avg_time, make_avg_time_dataset(race_type, course_len, "all", all_avg_time)])

        for class_name in model.CLASSES[1:]:
            df_class = df_course[df_course["class"] == class_name]
            class_avg_time = get_avg_time_list_from_race_results_df(df_class)
            df_avg_time = pd.concat(
                [df_avg_time, make_avg_time_dataset(race_type, course_len, class_name, class_avg_time)]
            )

    return df_avg_time.reset_index(drop=True)
