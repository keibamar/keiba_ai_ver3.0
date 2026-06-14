"""HTML生成（Forge）で使う表データの整形ヘルパー

旧 web/src/utils/format_data.py からの移植（ロジック変更なし）。
"""

import pandas as pd


def format_date(date_str):
    """YYYYMMDD → YYYY/M/D"""
    if not date_str:
        return ""
    return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"


def extract_entry_sub(df_analysis):
    """df_analysis から 馬番, rank, score を抽出して統合する

    rank/score が存在しない場合は空列にする。
    """
    df = df_analysis.copy()

    if "馬番" not in df.columns:
        df["馬番"] = None

    if {"rank", "score"}.issubset(df.columns):
        entry_sub = df[["馬番", "rank", "score"]].copy()
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
