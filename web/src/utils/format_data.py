import os
import re
import pandas as pd


def format_date(date_str):
    """YYMMDD → YYYY/M/D"""
    if not date_str:
        return ""
    dt = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
    return dt

def extract_entry_sub(df_analysis: pd.DataFrame):
    """
    df_analysis から 馬番, rank, score を抽出して統合。
    rank/score が存在しない場合は空列にする。
    """
    df = df_analysis.copy()

    # 馬番列がない場合はエラー防止のため作成
    if "馬番" not in df.columns:
        df["馬番"] = None

    # rank, score が存在する場合
    if {"rank", "score"}.issubset(df.columns):
        entry_sub = df[["馬番", "rank", "score"]].copy()
        # 統合列を追加（必要なら）
        entry_sub["rank_score"] = entry_sub["rank"].astype(str) + "_" + entry_sub["score"].astype(str)
    else:
        # 存在しない場合 → 空の列を追加
        entry_sub = df[["馬番"]].copy()
        entry_sub["rank"] = None
        entry_sub["score"] = None
        entry_sub["rank_score"] = None

    return entry_sub

def merge_rank_score(df_race, df_analysis):
    # 数値型に揃える
    df_race["馬番"] = df_race["馬番"].astype(str)
    df_analysis["馬番"] = df_analysis["馬番"].astype(str)

    # 必要列だけ抽出
    entry_sub = extract_entry_sub(df_analysis)

    # 結合
    merged = pd.merge(df_race, entry_sub, on="馬番", how="left")
    return merged

def get_subfolders(path):
    """
    指定したフォルダ内にあるフォルダ名一覧を取得する関数
    :param path: 親フォルダのパス
    :return: サブフォルダ名のリスト
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"指定されたパスが存在しません: {path}")
    
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]