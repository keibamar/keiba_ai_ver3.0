import os
import sys
import re

from datetime import date
from glob import glob
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import name_header

# # --- 固定リスト ---
CLASSES = ["all", "未勝利","新馬", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"]
GROUNDS = ["全", "良", "稍重", "重", "不良"]

def analyze_winner_weights_multi_years(place_id,  start_year=2019, current_year = date.today().year):
    """
    各年度（start_year〜今年）について勝ち馬の平均馬体重を算出し、
    年ごとの結果 + 全期間平均の DataFrame を返す。

    Parameters
    ----------
    base_dir : str
        CSV ファイルのディレクトリパス（例: "./data/"）
        → ファイル名は "{year}_race_results.csv" のようにする前提。
    place_id : int
        競馬場ID (1〜10)
    start_year : int
        集計を始める年（例: 2020）

    Returns
    -------
    results_by_year : dict[int, pd.DataFrame]
        年度別の結果を格納した辞書。
    total_df : pd.DataFrame
        全期間の平均結果。
    """
    base_dir = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//"
    results_by_year = {}

    for year in range(start_year, current_year + 1):
        csv_path = os.path.join(base_dir, f"{year}_race_results.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} が見つかりません。スキップします。")
            continue
        # print(f"📘 {year}年のデータを処理中 ...")
        df_year = analyze_winner_weights(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year

    if not results_by_year:
        print("❌ 有効なデータがありません。")
        return {}, pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    group_cols = ["race_type", "course_len", "ground_state", "class"]
    total_df = (
        combined_df.groupby(group_cols, dropna=False)["馬体重"]
        .mean()
        .round(1)
        .reset_index()
    )

    total_df["race_type"] = pd.Categorical(total_df["race_type"], categories=["芝", "ダート"], ordered=True)
    total_df["class"] = pd.Categorical(total_df["class"], categories=CLASSES, ordered=True)
    total_df["ground_state"] = pd.Categorical(total_df["ground_state"], categories=GROUNDS, ordered=True)
    total_df = total_df.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)
    total_df = total_df.reindex(columns=["race_type", "course_len", "ground_state", "class", "馬体重"])

    print(f"✅ {name_header.PLACE_LIST[int(place_id) - 1]} 馬体重 全期間平均（{start_year}〜{current_year}）を作成しました。")
    return total_df 

def analyze_winner_weights(place_id, year):
    """
    勝ち馬の「馬体重」平均を race_type, course_len, ground_state, class ごとに算出する。
    """
    csv_path = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_race_results.csv"
    if os.path.isfile(csv_path):
        df_raw = pd.read_csv(csv_path, dtype=str, index_col=0).reset_index().rename(columns={"index": "race_id"})
    else:
        return pd.DataFrame()

    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")
    df["馬体重"] = pd.to_numeric(df["馬体重"], errors="coerce")

    winners = df[df["着順"] == 1].copy()
    if winners.empty:
        return pd.DataFrame()

    all_results = []
    courses = name_header.COURSE_LISTS[place_id - 1]

    for race_type, course_len in courses:
        base_data = winners[
            (winners["race_type"] == race_type) &
            (winners["course_len"] == float(course_len))
        ]

        for cls in CLASSES:
            for grd in GROUNDS:
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
    df_result["race_type"] = pd.Categorical(df_result["race_type"], categories=["芝", "ダート"], ordered=True)
    df_result["class"] = pd.Categorical(df_result["class"], categories=CLASSES, ordered=True)
    df_result["ground_state"] = pd.Categorical(df_result["ground_state"], categories=GROUNDS, ordered=True)
    df_result = df_result.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)

    df_result = df_result.reindex(columns=["race_type", "course_len", "ground_state", "class", "馬体重"])

    return df_result

def analyze_average_pop_multi_years(place_id, start_year=2019, current_year=date.today().year, top3=False):
    """
    各年度（start_year〜今年）について、人気データを集計。
    勝ち馬（top3=False）または3着内（top3=True）を対象に、
    全期間平均（TOTAL）を計算して返す。
    """
    base_dir = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//"
    results_by_year = {}

    for year in range(start_year, current_year + 1):
        csv_path = os.path.join(base_dir, f"{year}_race_results.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} が見つかりません。スキップします。")
            continue

        print(f"📘 {year}年の人気データを処理中 ...")
        df_year = analyze_average_pops(place_id, year, top3=top3)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year

    if not results_by_year:
        print("❌ 有効な人気データがありません。")
        return {}, pd.DataFrame()

    # --- 全期間結合 ---
    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    # --- 平均値＆合計を集計 ---
    group_cols = ["race_type", "course_len", "ground_state", "class"]

    # 平均人気
    avg_df = (
        combined_df.groupby(group_cols, dropna=False)["avg_pop"]
        .mean()
        .round(2)
        .reset_index()
    )

    # 人気別勝利数の合計
    pop_cols = [f"pop_{i}_count" for i in range(1, 19)]
    sum_df = (
        combined_df.groupby(group_cols, dropna=False)[pop_cols]
        .sum()
        .reset_index()
    )

    # 結合
    total_df = pd.merge(avg_df, sum_df, on=group_cols, how="left")

    # --- 並び順を整える ---
    total_df["race_type"] = pd.Categorical(total_df["race_type"], categories=["芝", "ダート"], ordered=True)
    total_df["class"] = pd.Categorical(total_df["class"], categories=CLASSES, ordered=True)
    total_df["ground_state"] = pd.Categorical(total_df["ground_state"], categories=GROUNDS, ordered=True)

    total_df = total_df.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)
    total_df = total_df.reindex(columns=group_cols + ["avg_pop"] + pop_cols)

    print(f"✅ {name_header.PLACE_LIST[int(place_id) - 1]} 人気 全期間平均（{start_year}〜{current_year}）を作成しました。対象: {'3着内' if top3 else '勝ち馬'}")
    return total_df

def analyze_average_pops(place_id, year, top3=False):
    """
    勝ち馬または3着内馬の平均人気と人気別勝利数を集計。
    top3=True の場合は3着内を対象。
    """
    csv_path = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_race_results.csv"
    if os.path.isfile(csv_path):
        df_raw = pd.read_csv(csv_path, dtype=str, index_col=0).reset_index().rename(columns={"index": "race_id"})
    else:
        return pd.DataFrame()

    df = df_raw.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["人気"] = pd.to_numeric(df["人気"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    # 出走頭数を算出（race_id単位）
    df["頭数"] = df.groupby("race_id")["馬番"].transform("count")

    all_results = []
    courses = name_header.COURSE_LISTS[place_id - 1]

    for race_type, course_len in courses:
        base_data = df[
            (df["race_type"] == race_type) &
            (df["course_len"] == float(course_len))
        ]

        for cls in CLASSES:
            for grd in GROUNDS:
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
                    # データなしでも全人気列を埋める
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
    df_result["race_type"] = pd.Categorical(df_result["race_type"], categories=["芝", "ダート"], ordered=True)
    df_result["class"] = pd.Categorical(df_result["class"], categories=CLASSES, ordered=True)
    df_result["ground_state"] = pd.Categorical(df_result["ground_state"], categories=GROUNDS, ordered=True)

    df_result = df_result.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)

    cols = ["race_type", "course_len", "ground_state", "class", "avg_pop"] + [f"pop_{i}_count" for i in range(1, 19)]
    df_result = df_result.reindex(columns=cols)
    return df_result

def analyze_frame_and_horse_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """
    各年度（start_year〜今年）について、枠・馬番の平均と勝ち数を算出。
    全年度を結合し、全期間の平均（平均値＋合計勝利数）を生成。
    """
    base_dir = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//"
    results_by_year = {}

    for year in range(start_year, current_year + 1):
        csv_path = os.path.join(base_dir, f"{year}_race_results.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} が見つかりません。スキップします。")
            continue

        print(f"📘 {year}年の枠・馬番データを処理中 ...")
        df_year = analyze_average_frame_and_horse(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year

    if not results_by_year:
        print("❌ 有効な枠・馬番データがありません。")
        return pd.DataFrame()

    combined_df = pd.concat(results_by_year.values(), ignore_index=True)

    # 平均値カラムと合計カラムを判別
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
    total_df["class"] = pd.Categorical(total_df["class"], categories=CLASSES, ordered=True)
    total_df["ground_state"] = pd.Categorical(total_df["ground_state"], categories=GROUNDS, ordered=True)

    total_df = total_df.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)

    print(f"✅ {name_header.PLACE_LIST[int(place_id) - 1]} 全期間平均（{start_year}〜{current_year}）の枠・馬番データを作成しました。")
    return total_df

def analyze_average_frame_and_horse(place_id, year):
    """
    勝ち馬の「平均枠番」「平均馬番」「勝利数」および
    各枠・馬番の勝ち数を race_type, course_len, ground_state, class ごとに算出。
    （人気分析と同じ構造・並び順で全条件生成）
    """
    csv_path = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_race_results.csv"
    if not os.path.isfile(csv_path):
        return pd.DataFrame()

    df_raw = pd.read_csv(csv_path, dtype=str, index_col=0).reset_index().rename(columns={"index": "race_id"})
    df = df_raw.copy()

    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    all_results = []
    courses = name_header.COURSE_LISTS[place_id - 1]

    for race_type, course_len in courses:
        for cls in CLASSES:
            for grd in GROUNDS:
                tmp = df[
                    (df["race_type"] == race_type) &
                    (df["course_len"] == float(course_len))
                ]

                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                if tmp.empty:
                    # データがない場合も空行を埋めておく（人気版と同じ）
                    result = {
                        "race_type": race_type,
                        "course_len": int(course_len),
                        "ground_state": grd,
                        "class": cls,
                        "avg_frame": None,
                        "avg_horse": None,
                        "total_winners": 0
                    }
                    for i in range(1, 9):
                        result[f"frame_{i}_wins"] = 0
                    for j in range(1, 19):
                        result[f"horse_{j}_wins"] = 0
                    all_results.append(result)
                    continue

                winners = tmp[tmp["着順"] == 1].copy()
                if winners.empty:
                    result = {
                        "race_type": race_type,
                        "course_len": int(course_len),
                        "ground_state": grd,
                        "class": cls,
                        "avg_frame": None,
                        "avg_horse": None,
                        "total_winners": 0
                    }
                    for i in range(1, 9):
                        result[f"frame_{i}_wins"] = 0
                    for j in range(1, 19):
                        result[f"horse_{j}_wins"] = 0
                    all_results.append(result)
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
                    "total_winners": len(winners)
                }

                for i in range(1, 9):
                    result[f"frame_{i}_wins"] = frame_counts[i]
                for j in range(1, 19):
                    result[f"horse_{j}_wins"] = horse_counts[j]

                all_results.append(result)

    df_result = pd.DataFrame(all_results)

    # --- カテゴリ順を指定（人気集計と同じ並び）---
    df_result["race_type"] = pd.Categorical(df_result["race_type"], categories=["芝", "ダート"], ordered=True)
    df_result["class"] = pd.Categorical(df_result["class"], categories=CLASSES, ordered=True)
    df_result["ground_state"] = pd.Categorical(df_result["ground_state"], categories=GROUNDS, ordered=True)

    df_result = df_result.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)

    base_cols = ["race_type", "course_len", "ground_state", "class", "avg_frame", "avg_horse", "total_winners"]
    frame_cols = [f"frame_{i}_wins" for i in range(1, 9)]
    horse_cols = [f"horse_{j}_wins" for j in range(1, 19)]

    df_result = df_result.reindex(columns=base_cols + frame_cols + horse_cols)

    return df_result

def analyze_frame_and_horse_top3_multi_years(place_id, start_year=2019, current_year=date.today().year):
    """
    各年度（start_year〜今年）について、3着以内馬の枠・馬番分析を算出。
    全年度を結合して平均化し、全期間の平均と合計を生成。
    """
    base_dir = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//"
    results_by_year = {}

    for year in range(start_year, current_year + 1):
        csv_path = os.path.join(base_dir, f"{year}_race_results.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} が見つかりません。スキップします。")
            continue

        # print(f"📘 {year}年の3着内データを処理中 ...")
        df_year = analyze_average_frame_and_horse_top3(place_id, year)
        if not df_year.empty:
            df_year["year"] = year
            results_by_year[year] = df_year

    if not results_by_year:
        print("❌ 有効な3着内データがありません。")
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
    total_df["class"] = pd.Categorical(total_df["class"], categories=CLASSES, ordered=True)
    total_df["ground_state"] = pd.Categorical(total_df["ground_state"], categories=GROUNDS, ordered=True)

    total_df = total_df.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)

    print(f"✅ {name_header.PLACE_LIST[int(place_id) - 1]} 全期間平均（{start_year}〜{current_year}）の3着内枠・馬番データを作成しました。")
    return total_df

def analyze_average_frame_and_horse_top3(place_id, year):
    """
    3着以内馬の「平均枠番」「平均馬番」「3着内馬数」および
    各枠・馬番ごとの3着内回数を race_type, course_len, ground_state, class ごとに算出。
    """
    csv_path = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_race_results.csv"
    if not os.path.isfile(csv_path):
        return pd.DataFrame()

    df_raw = pd.read_csv(csv_path, dtype=str, index_col=0).reset_index().rename(columns={"index": "race_id"})
    df = df_raw.copy()

    # 数値変換
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["course_len"] = pd.to_numeric(df["course_len"], errors="coerce")

    all_results = []
    courses = name_header.COURSE_LISTS[place_id - 1]

    for race_type, course_len in courses:
        for cls in CLASSES:
            for grd in GROUNDS:
                tmp = df[
                    (df["race_type"] == race_type) &
                    (df["course_len"] == float(course_len))
                ]

                if cls != "all":
                    tmp = tmp[tmp["class"] == cls]
                if grd != "全":
                    tmp = tmp[tmp["ground_state"] == grd]

                if tmp.empty:
                    result = make_empty_record(race_type, course_len, grd, cls)
                    all_results.append(result)
                    continue

                top3 = tmp[tmp["着順"] <= 3].copy()
                if top3.empty:
                    result = make_empty_record(race_type, course_len, grd, cls)
                    all_results.append(result)
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
                    "total_top3": len(top3)
                }

                for i in range(1, 9):
                    result[f"frame_{i}_top3"] = frame_counts[i]
                for j in range(1, 19):
                    result[f"horse_{j}_top3"] = horse_counts[j]

                all_results.append(result)

    df_result = pd.DataFrame(all_results)

    # --- 並び順を統一 ---
    df_result["race_type"] = pd.Categorical(df_result["race_type"], categories=["芝", "ダート"], ordered=True)
    df_result["class"] = pd.Categorical(df_result["class"], categories=CLASSES, ordered=True)
    df_result["ground_state"] = pd.Categorical(df_result["ground_state"], categories=GROUNDS, ordered=True)

    df_result = df_result.sort_values(["race_type", "course_len", "class", "ground_state"]).reset_index(drop=True)

    base_cols = ["race_type", "course_len", "ground_state", "class", "avg_frame", "avg_horse", "total_top3"]
    frame_cols = [f"frame_{i}_top3" for i in range(1, 9)]
    horse_cols = [f"horse_{j}_top3" for j in range(1, 19)]

    df_result = df_result.reindex(columns=base_cols + frame_cols + horse_cols)

    return df_result


def make_empty_record(race_type, course_len, grd, cls):
    """空データ行のテンプレート"""
    record = {
        "race_type": race_type,
        "course_len": int(course_len),
        "ground_state": grd,
        "class": cls,
        "avg_frame": None,
        "avg_horse": None,
        "total_top3": 0
    }
    for i in range(1, 9):
        record[f"frame_{i}_top3"] = 0
    for j in range(1, 19):
        record[f"horse_{j}_top3"] = 0
    return record

def get_horse_id_list():
    """
    horse_id_map.csv から horse_id_list を作成して返す関数
    Returns
    -------
    list[str]
        horse_idのリスト
    """
    horse_id_map_path = os.path.join(name_header.DATA_PATH, "horse_id_map.csv")
    try:
        df = pd.read_csv(horse_id_map_path, dtype=str)
        # 不要な空白などを除去
        df["horse_id"] = df["horse_id"].str.strip()
        horse_id_list = df["horse_id"].dropna().unique().tolist()
        return horse_id_list
    except Exception as e:
        print(f"❌ horse_id_list の作成に失敗しました: {e}")
        return []

def update_horse_name_id_map_from_results(place_id, year):
    """
    指定ディレクトリ内のレース結果CSVをすべて読み込み、
    既存の horse_name_id_map.csv にマージして更新する。

    Parameters
    ----------
    input_dir : str
        レース結果CSVの格納ディレクトリ（例: './RaceResults/'）
    output_path : str
        出力ファイルパス（例: './horse_name_id_map.csv'）
    """
    csv_file_path = os.path.join(name_header.DATA_PATH, "horse_id_map.csv")
    # --- 既存データの読み込み ---
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path, dtype=str)
        print(f"📘 既存の対応表を読み込みました ({len(existing_df)}件)")
    else:
        existing_df = pd.DataFrame(columns=["馬名", "horse_id"])
        print("⚠️ 既存の対応表が見つかりません。新規作成します。")

    # --- 新規データの読み込み ---
    input_files = os.path.join(name_header.DATA_PATH, "RaceResults", name_header.PLACE_LIST[place_id - 1], f"{year}_race_results.csv")

    # --- 入力ファイル読み込み ---
    print()
    all_data = []
    if not os.path.exists(input_files):
        print(f"❌ ファイルが見つかりません: {input_files}")
    else:
        try:
            df = pd.read_csv(input_files, dtype=str)
            if "馬名" in df.columns and "horse_id" in df.columns:
                all_data.append(df[["馬名", "horse_id"]].dropna())
                print(f"📗 読み込み成功: {input_files} ({len(df)}件)")
            else:
                print(f"⚠️ カラム不足のためスキップ: {input_files}")
        except Exception as e:
            print(f"[WARN] {input_files} 読み込み中にエラー: {e}")

    if not all_data:
        print("❌ 有効なCSVデータがありません。")
        return

     # --- 結合 ---
    new_df = pd.concat(all_data, ignore_index=True)
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)

    # --- 重複削除 ---
    merged_df = merged_df.drop_duplicates(subset=["horse_id"], keep="first")
    # merged_df = merged_df.drop_duplicates(subset=["馬名"], keep="first")

    # --- horse_id順にソート ---
    merged_df = merged_df.sort_values(by="horse_id").reset_index(drop=True)

    # --- 保存 ---
    merged_df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

    print(f"✅ 馬名・horse_id対応表を更新しました ({len(merged_df)}件): {csv_file_path}")

def add_horse_name_id_map(horse_id, horse_name):
    csv_file_path = os.path.join(name_header.DATA_PATH, "horse_id_map.csv")
    # --- 既存データの読み込み ---
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path, dtype=str)
    else:
        print("Error Get HorseIDMap")
        return
    
    # --- 入力値の前処理 ---
    horse_id = str(horse_id).strip()
    horse_name = str(horse_name).strip()

    if horse_id == "" or horse_name == "":
        print("⚠️ horse_id または horse_name が空です。追加をスキップします。")
        return

    # --- 重複チェック（IDまたは馬名が既に存在するか） ---
    id_exists = horse_id in existing_df["horse_id"].values
    name_exists = horse_name in existing_df["馬名"].values

    if id_exists and name_exists:
        # print(f"✅ 既に登録済み: {horse_name} ({horse_id})")
        return
    elif id_exists and not name_exists:
        print(f"⚠️ ID重複: {horse_id} が既に存在します（別名）。登録をスキップします。")
        return
    elif name_exists and not id_exists:
        print(f"⚠️ 馬名重複: {horse_name} が既に存在します（別ID）。登録をスキップします。")
        return

    # --- 新規行を追加 ---
    new_row = pd.DataFrame([[horse_name, horse_id]], columns=["馬名", "horse_id"])
    updated_df = pd.concat([existing_df, new_row], ignore_index=True)

    # --- ソート（horse_id順に整理） ---
    updated_df = updated_df.sort_values(by="horse_id", ascending=True, ignore_index=True)

    # --- 保存 ---
    updated_df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

def update_horse_name_id_map(race_card_df):
    # 必要な列があるか確認
    if not {"馬名", "horse_id"}.issubset(race_card_df.columns):
        print("⚠️ DataFrame に '馬名' または 'horse_id' 列がありません。")
        return

    # 行ごとに追加処理
    for _, row in race_card_df.iterrows():
        horse_name = str(row["馬名"]).strip()
        horse_id = str(row["horse_id"]).strip()

        # 無効な値はスキップ
        if horse_name == "" or horse_id == "" or horse_id.lower() == "nan":
            continue

        # 1件ずつ更新
        try:
            add_horse_name_id_map(horse_id, horse_name)
        except Exception as e:
            print(f"⚠️ 追加エラー: {horse_name} ({horse_id}) → {e}")

def update_average_pops(place_id, year):
    result_top = analyze_average_pops(place_id, year, False)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AveragePops//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_average_pops.csv"
        result_top.to_csv(output_path)
    result_top3 = analyze_average_pops(place_id, year, True)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AveragePops//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_average_pops_top3.csv"
        result_top3.to_csv(output_path)
    total_top = analyze_average_pop_multi_years(place_id, start_year=2019, current_year=year, top3=False)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AveragePops//" + name_header.PLACE_LIST[place_id -1] + "//" + f"total_average_pops.csv"
        total_top.to_csv(output_path)
    total_top3 = analyze_average_pop_multi_years(place_id, start_year=2019, current_year=year, top3=True)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AveragePops//" + name_header.PLACE_LIST[place_id -1] + "//" + f"total_average_pops_top3.csv"
        total_top3.to_csv(output_path)

def update_winners_weight(place_id, year):
    result_winner =  analyze_winner_weights(place_id, year)
    if not result_winner.empty:
        output_path = name_header.DATA_PATH + "//AverageWeights//" + name_header.PLACE_LIST[place_id -1] + "//" +f"{year}_winner_weight.csv"
        result_winner.to_csv(output_path)
    total_winner = analyze_winner_weights_multi_years(place_id, start_year=2019, current_year=year)
    if not total_winner.empty:
        output_path = name_header.DATA_PATH + "//AverageWeights//" + name_header.PLACE_LIST[place_id -1] + "//" + f"total_winner_weight.csv"
        total_winner.to_csv(output_path)

def update_average_frame_and_horse(place_id,year):
    result_top = analyze_average_frame_and_horse(place_id, year)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AverageFrames//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_average_frames.csv"
        result_top.to_csv(output_path)
    result_top3 = analyze_average_frame_and_horse_top3(place_id, year)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AverageFrames//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_average_frames_top3.csv"
        result_top3.to_csv(output_path)
    total_top = analyze_frame_and_horse_multi_years(place_id, start_year=2019, current_year=year)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AverageFrames//" + name_header.PLACE_LIST[place_id -1] + "//" + f"total_average_frames.csv"
        total_top.to_csv(output_path)
    total_top3 = analyze_frame_and_horse_top3_multi_years(place_id, start_year=2019, current_year=year)
    if not result_top.empty:
        output_path = name_header.DATA_PATH + "//AverageFrames//" + name_header.PLACE_LIST[place_id -1] + "//" + f"total_average_frames_top3.csv"
        total_top3.to_csv(output_path)

def weekly_update_horse_name_id_map(year = date.today().year):
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        update_horse_name_id_map_from_results(place_id,year)

def weekly_update_average_pops(year = date.today().year):
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        update_average_pops(place_id,year)

def weekly_update_winners_weight(year = date.today().year):
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        update_winners_weight(place_id,year)

def weekly_update_average_frame_and_horse(year = date.today().year):
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        update_average_frame_and_horse(place_id,year)

if __name__ == '__main__':
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        for year in range(2019, 2026):
            update_horse_name_id_map_from_results(place_id,year)

    # for place_id in range(1, len(name_header.PLACE_LIST) + 1):
    #     # 各年の記録を計算
    #     for year in range(2019,date.today().year + 1):
    #         csv_path = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_race_results.csv"
    #         result = analyze_average_pops(csv_path, place_id, False)
    #         if not result.empty:
    #             output_path = name_header.DATA_PATH + "//AveragePops//" + name_header.PLACE_LIST[place_id -1] + "//" + f"{year}_average_pops.csv"
    #             result.to_csv(output_path)
    #     # totalの記録を計算
    #     base_dir = name_header.DATA_PATH + "//RaceResults//" + name_header.PLACE_LIST[place_id -1] + "//"
    #     total_df = analyze_average_pop_multi_years(base_dir, place_id, 2019, False)
    #     total_ouutput_path = name_header.DATA_PATH + "//AveragePops//" + name_header.PLACE_LIST[place_id -1] + "//" + "total_average_pops.csv"
    #     total_df.to_csv(total_ouutput_path)