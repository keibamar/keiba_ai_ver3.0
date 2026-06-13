"""race_infoデータセットの変換ロジック（純粋関数のみ・I/Oなし）

旧 src/legacy_datasets/analysis_race_info.py の集計ロジックを移植したもの。
race_resultsのDataFrame（"race_id"列を持つ、index_col=0でCSVを読み込んでreset_indexしたもの）
を入力として、各種集計結果のDataFrameを返す。
"""

import pandas as pd

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
