"""レース傾向分析ロジック

race_result と ai_performance.csv を集計して、
日次・週次の傾向分析に必要な統計を計算する。
"""

import os
from datetime import date, timedelta

import pandas as pd

from src.config.constants import NAME_LIST, PLACE_LIST
from src.config import paths
from src.managers import race_result_dataset_manager as rr_mgr
from src.managers import race_card_dataset_manager as rc_mgr
from src.managers import ai_performance_dataset_manager as ap_mgr


# 馬場状態の重さ順マップ（数値が大きいほど悪い馬場）
GROUND_STATE_RANK = {"良": 0, "稍重": 1, "重": 2, "不良": 3}


def _place_id_from_race_id(race_id: str) -> int:
    """race_id文字列（12桁）から place_id（1-10）を返す"""
    return int(race_id[4:6])


def _load_time_id_list(target_date: date) -> list:
    """race_time_id_listから指定日のrace_idリストを返す"""
    date_str = target_date.strftime("%Y%m%d")
    path = os.path.join(paths.RACE_TIME_ID_LIST_PATH, f"{date_str}.csv")
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path, dtype=str)
    if "race_id" not in df.columns:
        return []
    return df["race_id"].dropna().tolist()


def _load_results_for_date(target_date: date) -> pd.DataFrame:
    """指定日付のレース結果を全開催場から読み込んで結合する。

    まず年次統合CSV（weekly_update後に最新）を試み、当該日付の行が
    見つからない場合は race_time_id_list 経由でper-race CSVにフォールバックする。
    """
    # race_result の date 列は "2026年07月25日" 形式
    date_str = target_date.strftime("%Y年%m月%d日")
    year = target_date.year
    dfs = []

    # --- 試み1: 年次統合CSV ---
    for place_id in range(1, len(PLACE_LIST) + 1):
        df = rr_mgr.get_race_results_csv(place_id, year)
        if df.empty or "date" not in df.columns:
            continue
        df_day = df[df["date"] == date_str].copy()
        if df_day.empty:
            continue
        df_day["place_id"] = place_id
        df_day.index.name = "race_id"
        df_day = df_day.reset_index()
        dfs.append(df_day)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    # --- 試み2: time_id_list → per-race CSVフォールバック ---
    race_ids = _load_time_id_list(target_date)
    if not race_ids:
        return pd.DataFrame()

    per_race_dfs = []
    for race_id in race_ids:
        df_r = rr_mgr.get_race_id_result(race_id)
        if df_r.empty:
            continue
        # per-race ファイルには date/weather がない場合があるので race_info で補完
        if "weather" not in df_r.columns or df_r["weather"].isna().all():
            info = rc_mgr.get_race_info_csv(race_id)
            if not info.empty and "weather" in info.columns:
                df_r["weather"] = info.iloc[0]["weather"]
        df_r["race_id"] = race_id
        df_r["place_id"] = int(race_id[4:6])
        if "date" not in df_r.columns:
            df_r["date"] = date_str
        per_race_dfs.append(df_r)

    if not per_race_dfs:
        return pd.DataFrame()
    return pd.concat(per_race_dfs, ignore_index=True)


COL_RANK = "着順"
COL_TUKA = "通過"
COL_AGARI = "上り"
COL_TANSHO = "単勝"
COL_NINKI = "人気"


def _compute_upset_index(df: pd.DataFrame) -> dict:
    """1着馬の単勝オッズ分布から荒れ度を計算する

    Returns:
        dict: upset_ratio(float), label(str), high_odds_count(int), race_count(int)
    """
    # race_count はユニークなレース数（同着行の重複を除く）
    if "race_id" in df.columns:
        race_count = int(df["race_id"].nunique())
    else:
        race_count = 0

    winners = df[df[COL_RANK] == "1"].copy()
    if winners.empty:
        return {"upset_ratio": 0.0, "label": "データなし", "high_odds_count": 0, "race_count": race_count}

    def to_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    # 同着対策: race_id ごとに1行だけ取る
    if "race_id" in winners.columns:
        winners = winners.drop_duplicates(subset=["race_id"])

    winners["odds_float"] = winners[COL_TANSHO].apply(to_float)
    valid = winners.dropna(subset=["odds_float"])
    high_odds = int((valid["odds_float"] >= 10.0).sum())
    upset_ratio = high_odds / race_count if race_count > 0 else 0.0

    if upset_ratio >= 0.4:
        label = "荒れ"
    elif upset_ratio >= 0.2:
        label = "やや荒れ"
    else:
        label = "堅い"

    return {
        "upset_ratio": round(upset_ratio, 3),
        "label": label,
        "high_odds_count": int(high_odds),
        "race_count": int(race_count),
    }


def _compute_up3f_stats(df: pd.DataFrame) -> dict:
    """上り3F平均をコース別（芝/ダート）に集計する"""
    def to_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    df = df.copy()
    df["上り_f"] = df[COL_AGARI].apply(to_float)
    result = {}
    for rt in ["芝", "ダート"]:
        sub = df[df["race_type"] == rt]["上り_f"].dropna()
        if len(sub) >= 3:
            result[rt] = round(float(sub.mean()), 2)
    return result


def _parse_time_to_sec(t: str) -> float | None:
    """'0:1:10.0' 形式のタイム文字列を秒に変換する"""
    try:
        parts = str(t).strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except (ValueError, TypeError):
        return None
    return None


def _compute_win_time_stats(df: pd.DataFrame) -> dict:
    """勝ち時計を開催場・クラス・コース種別・距離・馬場状態ごとに集計する

    Returns:
        dict: {
            place_name: {
                race_type: {
                    course_len: {
                        class_ground_key: {"avg_sec": float, "race_count": int, "class": str, "ground": str}
                    }
                }
            }
        }
    """
    required = {"タイム", "place_id", "race_type", "course_len", "class", "ground_state"}
    if not required.issubset(df.columns):
        return {}

    winners = df[df[COL_RANK] == "1"].copy()
    if winners.empty:
        return {}

    winners["time_sec"] = winners["タイム"].apply(_parse_time_to_sec)
    winners = winners.dropna(subset=["time_sec"])

    result = {}
    group_keys = ["place_id", "race_type", "course_len", "class", "ground_state"]
    for keys, grp in winners.groupby(group_keys):
        place_id, race_type, course_len, cls, ground = keys
        place_name = NAME_LIST[int(place_id) - 1]
        times = grp["time_sec"].tolist()
        if not times:
            continue
        cg_key = f"{cls}_{ground}"
        (result
         .setdefault(place_name, {})
         .setdefault(race_type, {})
         .setdefault(str(course_len), {})
         )[cg_key] = {
            "avg_sec": round(sum(times) / len(times), 2),
            "race_count": len(times),
            "class": cls,
            "ground": ground,
        }
    return result


def _compute_pace_bias(df: pd.DataFrame) -> dict:
    """通過順データから前残り率・差し馬勝率を計算する（週次用）

    通過列 format: "1-1-1-1" (コーナー通過順位)
    4角位置が低く（前）勝った → 前残り
    4角位置が高く（後ろ）勝った → 差し/追込
    """
    winners = df[df[COL_RANK] == "1"].copy()
    if winners.empty or COL_TUKA not in winners.columns:
        return {}

    front_count = 0
    closer_count = 0
    valid = 0

    for _, row in winners.iterrows():
        tuka = str(row.get(COL_TUKA, ""))
        if not tuka or tuka in ("", "nan"):
            continue
        parts = tuka.replace(" ", "").split("-")
        if len(parts) < 2:
            continue
        try:
            # 最終コーナー通過順
            last_corner = int(parts[-1])
            valid += 1
            if last_corner <= 3:
                front_count += 1
            elif last_corner >= 6:
                closer_count += 1
        except (ValueError, TypeError):
            continue

    if valid == 0:
        return {}

    return {
        "front_rate": round(front_count / valid, 3),
        "closer_rate": round(closer_count / valid, 3),
        "valid_races": valid,
    }


def _compute_ground_summary(df: pd.DataFrame) -> dict:
    """馬場状態・天気・上り3Fのサマリを開催場別・コース種別別に集計する"""

    def to_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    summary = {}
    for place_id, group in df.groupby("place_id"):
        place_name = NAME_LIST[place_id - 1]
        ground_counts = group["ground_state"].value_counts()
        weather_counts = group["weather"].value_counts()
        race_types = [rt for rt in ["芝", "ダート"] if rt in group["race_type"].values]

        # 種別ごとの上り3F・荒れ度
        course_stats = {}
        for rt in race_types:
            sub = group[group["race_type"] == rt].copy()
            sub["上り_f"] = sub[COL_AGARI].apply(to_float)
            up3f_vals = sub["上り_f"].dropna()
            up3f_avg = round(float(up3f_vals.mean()), 2) if len(up3f_vals) >= 2 else None

            winners = sub[sub[COL_RANK] == "1"].copy()
            winners["odds_f"] = winners[COL_TANSHO].apply(to_float)
            valid_w = winners.dropna(subset=["odds_f"])
            upset_count = int((valid_w["odds_f"] >= 10.0).sum())
            race_count = int(len(sub["race_id"].unique()) if "race_id" in sub.columns else len(sub) // 18)

            course_stats[rt] = {
                "up3f": up3f_avg,
                "race_count": race_count,
                "upset_count": upset_count,
            }

        summary[place_name] = {
            "ground_state": ground_counts.index[0] if len(ground_counts) > 0 else "不明",
            "weather": weather_counts.index[0] if len(weather_counts) > 0 else "不明",
            "race_types": race_types,
            "race_count": len(group["race_id"].unique()) if "race_id" in group.columns else len(group),
            "course_stats": course_stats,
        }
    return summary


def get_day_stats(target_date: date) -> dict:
    """指定日の傾向分析統計を計算して返す

    Args:
        target_date: 集計対象の日付

    Returns:
        dict:
            date: str (YYYYMMDD)
            ground_by_place: dict  {開催場名: {ground_state, weather, ...}}
            upset: dict            {upset_ratio, label, ...}
            up3f: dict             {芝: float, ダート: float}
            ai_perf: dict          {win_hit, win_return, place_hit, ...} 当日分
            race_count: int
    """
    df = _load_results_for_date(target_date)
    if df.empty:
        return {"date": target_date.strftime("%Y%m%d"), "error": "レース結果なし"}

    # 開催場別サマリ（馬場・天気）
    ground_by_place = _compute_ground_summary(df)

    # 荒れ度
    upset = _compute_upset_index(df)

    # 上り3F統計
    up3f = _compute_up3f_stats(df)

    # AI成績
    ai_df = ap_mgr.get_ai_performance_dataset()
    day_str = target_date.strftime("%Y-%m-%d")
    ai_day = ai_df[ai_df["race_day"] == day_str] if not ai_df.empty else pd.DataFrame()

    ai_perf = {}
    if not ai_day.empty:
        for bet in ["win", "place", "trio_box"]:
            hit_col = f"{bet}_hit"
            ret_col = f"{bet}_return"
            if hit_col in ai_day.columns:
                valid = pd.to_numeric(ai_day[hit_col], errors="coerce").dropna()
                ai_perf[f"{bet}_hit"] = round(float(valid.mean()), 3) if len(valid) > 0 else None
            if ret_col in ai_day.columns:
                valid = pd.to_numeric(ai_day[ret_col], errors="coerce").dropna()
                ai_perf[f"{bet}_return"] = round(float(valid.mean()), 3) if len(valid) > 0 else None

    win_times = _compute_win_time_stats(df)

    return {
        "date": target_date.strftime("%Y%m%d"),
        "ground_by_place": ground_by_place,
        "upset": upset,
        "up3f": up3f,
        "win_times": win_times,
        "ai_perf": ai_perf,
        "race_count": int(df["race_id"].nunique()) if "race_id" in df.columns else len(df),
    }


def get_week_stats(sat_date: date, sun_date: date) -> dict:
    """土日両日の週次統計を計算する（展開傾向を含む）

    Args:
        sat_date: 土曜の日付
        sun_date: 日曜の日付

    Returns:
        dict: sat_stats, sun_stats, combined の3キー
    """
    sat_stats = get_day_stats(sat_date)
    sun_stats = get_day_stats(sun_date)

    # 土日合算DataFrame（展開分析用）
    sat_df = _load_results_for_date(sat_date)
    sun_df = _load_results_for_date(sun_date)
    combined_df = pd.concat([sat_df, sun_df], ignore_index=True) if not sat_df.empty or not sun_df.empty else pd.DataFrame()

    combined = {}
    if not combined_df.empty:
        combined["pace_bias"] = _compute_pace_bias(combined_df)
        combined["up3f"] = _compute_up3f_stats(combined_df)
        combined["upset"] = _compute_upset_index(combined_df)

    return {
        "sat": sat_stats,
        "sun": sun_stats,
        "combined": combined,
        "sat_date": sat_date.strftime("%Y%m%d"),
        "sun_date": sun_date.strftime("%Y%m%d"),
    }
