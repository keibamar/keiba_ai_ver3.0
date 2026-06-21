"""race_resultデータセットの変換ロジック（純粋関数のみ・I/Oなし）"""

import re

import numpy as np
import pandas as pd

from src.config.constants import NAME_LIST
from src.datasets.race_result import model


def clean_columns(df):
    """重複列がある場合にカウント付きでリネームする

    Args:
        df (pd.DataFrame): 対象のDataFrame

    Returns:
        pd.DataFrame: 重複列をリネームしたDataFrame
    """
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index
        for i, idx in enumerate(dup_idx):
            cols[idx] = f"{dup}_{i + 1}"
    df.columns = cols
    return df


def format_race_results_dataframe(race_results_df):
    """race_resultsのフォーマットを整える

    Args:
        race_results_df (pd.DataFrame): race_resultデータセット

    Returns:
        pd.DataFrame: フォーマットされたrace_resultデータセット
    """
    # 欠損値を先に空文字列にしてからastype(str)する（先にastype(str)すると、
    # NaNがそのまま文字列"nan"になってCSVに保存されてしまうため）
    race_results_df = race_results_df.fillna("")
    race_results_df = race_results_df.astype(str)
    race_results_df = race_results_df.drop_duplicates(keep="first")
    return race_results_df


def to_int(val):
    """文字列等から整数を抽出する（例: '1400m' -> 1400, '1400.0' -> 1400）

    Args:
        val: 変換対象の値

    Returns:
        変換できた場合はint、できない場合は元の値
    """
    if pd.isna(val) or val == "":
        return val
    s = str(val)
    matched = re.search(r"(\d+)", s)
    if matched:
        try:
            return int(matched.group(1))
        except Exception:
            pass
    try:
        return int(float(s))
    except Exception:
        return val


def clean_day_race_results(df):
    """当日のレース結果テーブル（速報ページ）の体裁を整える

    旧 libs/scraping.py の clean_race_results を移植したもの。

    Args:
        df (pd.DataFrame): scrape_day_race_resultで取得した生のテーブル

    Returns:
        pd.DataFrame: 列をリネーム・並び替えしたレース結果
    """
    df = df.copy()

    for i in range(len(df)):
        if df.notnull().at[i, "タイム"]:
            df.at[i, "タイム"] = "0:" + df.at[i, "タイム"]

    df = df.rename(columns={
        "後3F": "上り",
        "コーナー通過順": "通過",
        "単勝オッズ": "単勝",
        "馬体重(増減)": "馬体重",
    })

    reorder_cols = [
        "着順", "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差",
        "人気", "単勝", "上り", "通過", "厩舎", "馬体重",
    ]
    return df[[c for c in reorder_cols if c in df.columns]]


def parse_race_info_tokens(tokens):
    """レース概要のテキストから抽出したトークン列から
    race_type, course_len, class, ground_state, weather, date を求める

    Args:
        tokens (list[str]): レース概要テキストを re.findall(r'\\w+', texts) した結果

    Returns:
        dict: 判定できた項目のみを含む辞書
              (race_type, course_len, class, ground_state, weather, date)
    """
    info = {}
    for text in tokens:
        if text in ["芝", "ダート"]:
            info["race_type"] = text
        if "障" in text:
            info["race_type"] = "障害"
        if "m" in text:
            info["course_len"] = int(re.findall(r"\d+", text)[-1])
        for race_class, texts in model.CLASS_TEXT_MAP.items():
            if text in texts:
                info["class"] = race_class
        if text in model.GROUND_STATE_LIST:
            info["ground_state"] = text
        if text in model.WEATHER_LIST:
            info["weather"] = text
        if "年" in text:
            info["date"] = text
    return info


def get_course_info(result):
    """レース情報を取得

    Args:
        result (pd.Series): 開催, course_len, ground_state, class を含むレース情報

    Returns:
        list: [place_id, race_type, course_len, ground_state, race_class]
    """
    # 開催競馬場を取得 (JRA以外は-1)
    place_id = -1
    course = result["開催"]
    for i in range(len(NAME_LIST)):
        if NAME_LIST[i] in course:
            place_id = i + 1
    if place_id < 0:
        return [place_id, " ", " ", " ", " "]

    # レース情報を取得
    race_info = result["course_len"]
    if "芝" in race_info:
        race_type = "芝"
    elif "ダ" in race_info:
        race_type = "ダート"
    else:
        race_type = "障害"
    course_len = re.sub(r"\D", "", race_info)

    # 馬場状態を取得
    ground_state = result["ground_state"]
    if ground_state in "稍":
        ground_state = "稍重"
    if ground_state in "不":
        ground_state = "不良"

    # クラスを取得
    race_class = result["class"]

    return [place_id, race_type, course_len, ground_state, race_class]
