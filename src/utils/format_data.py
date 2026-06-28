"""HTML生成（Forge）で使う表データの整形ヘルパー

旧 web/src/utils/format_data.py からの移植（ロジック変更なし）。
"""

import jpholiday
import pandas as pd


def format_date(date_str):
    """YYYYMMDD → YYYY/M/D"""
    if not date_str:
        return ""
    return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"


def weekday_css_class(day):
    """土曜=weekday-sat（青）、日曜・祝日=weekday-sun（赤）、平日=空文字を返す

    home_generator（今週のメインレース・先週の結果）・ai_performance_report_generator
    （開催別成績のレース詳細）など、日付に曜日を付けて見せる複数のページで共通して使う。
    """
    if day.weekday() == 5:
        return "weekday-sat"
    if day.weekday() == 6 or jpholiday.is_holiday(day):
        return "weekday-sun"
    return ""


def weekday_label_html(day, fmt="%m/%d"):
    """日付に曜日ラベル（土曜=青、日曜・祝日=赤）を付けたHTMLを返す（例: "06/20(土)"）"""
    weekday_kanji = "月火水木金土日"[day.weekday()]
    css_class = weekday_css_class(day)
    class_attr = f' class="{css_class}"' if css_class else ""
    return f'{day.strftime(fmt)}<span{class_attr}>({weekday_kanji})</span>'


def extract_entry_sub(df_analysis):
    """df_analysis から 馬番, rank, score（あればscore_hitrateも）を抽出して統合する

    rank/score が存在しない場合は空列にする。score_hitrate（的中率重視モデルの
    生スコア。バランス型ブレンドの正規化前の値）があれば一緒に運び、
    表示側でScore列をそちらにフォールバックできるようにする。
    """
    df = df_analysis.copy()

    if "馬番" not in df.columns:
        df["馬番"] = None

    if {"rank", "score"}.issubset(df.columns):
        cols = ["馬番", "rank", "score"]
        if "score_hitrate" in df.columns:
            cols.append("score_hitrate")
        entry_sub = df[cols].copy()
        entry_sub["rank_score"] = entry_sub["rank"].astype(str) + "_" + entry_sub["score"].astype(str)
    else:
        entry_sub = df[["馬番"]].copy()
        entry_sub["rank"] = None
        entry_sub["score"] = None
        entry_sub["rank_score"] = None

    return entry_sub


def merge_rank_score(df_race, df_analysis):
    """df_race（出馬表）にdf_analysis（rank/score）を馬番で結合する"""
    df_race["馬番"] = df_race["馬番"].astype(str)
    df_analysis["馬番"] = df_analysis["馬番"].astype(str)

    entry_sub = extract_entry_sub(df_analysis)

    return pd.merge(df_race, entry_sub, on="馬番", how="left")
