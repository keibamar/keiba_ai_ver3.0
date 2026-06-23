"""race_cardデータセットの変換ロジック（純粋関数のみ・I/Oなし）

旧 src/RacePrediction/race_card.py の make_race_card 内で使われていた
純粋関数（DataFrame加工・トークン解析）を移植したもの。
"""

import re

import pandas as pd


def is_waku_decided(race_card_df):
    """枠順（枠番）がまだ確定していない出馬表かどうかを判定する

    レース発走の数日前は、netkeiba出馬表ページに馬名等は掲載されるが
    枠番抽せん前のため「枠」列が存在しないか全てNaNになる。この場合は
    AI予想（rank_prediction）の入力に使えないため、呼び出し側で
    予想をスキップする判定に使う。

    Args:
        race_card_df (pd.DataFrame): 出馬表

    Returns:
        bool: 「枠」列が存在し、1件以上値が入っていればTrue
    """
    return "枠" in race_card_df.columns and bool(race_card_df["枠"].notna().any())


def blank_rank_df(n):
    """枠順未確定で予想をスキップする場合のscore/rank列（すべてNaN）を返す

    Args:
        n (int): 出走馬数（race_card_dfの行数と揃える）

    Returns:
        pd.DataFrame: score, rank列を持つDataFrame（全行NaN）
    """
    return pd.DataFrame({"score": [pd.NA] * n, "rank": [pd.NA] * n})


def parse_race_card_info_tokens(info):
    """出馬表ページのレース情報トークン列から
    race_type, course_len, class, ground_state, weather を求める

    Args:
        info (list[str]): レース情報テキストを re.findall(r'\\w+', texts) した結果

    Returns:
        dict: 判定できた項目のみを含む辞書
              (race_type, course_len, class, ground_state, weather)
    """
    df_info = {}
    for text in info:
        if text in ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "１勝クラス", "２勝クラス", "３勝クラス"]:
            df_info["class"] = text
        if "芝" in text:
            df_info["race_type"] = "芝"
        if "ダ" in text:
            df_info["race_type"] = "ダート"
        if "障" in text:
            df_info["race_type"] = "障害"
        if "m" in text:
            df_info["course_len"] = int(re.findall(r"\d+", text)[-1])
        if text in ["良", "重"]:
            df_info["ground_state"] = text
        if text in ["稍重", "稍"]:
            df_info["ground_state"] = "稍重"
        if text in ["不良", "不"]:
            df_info["ground_state"] = "不良"
        if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
            df_info["weather"] = text
    return df_info


def fill_race_info_defaults(race_info_df):
    """race_info_dfに欠けているweather/ground_state列をデフォルト値で補完する

    Args:
        race_info_df (pd.DataFrame): レース情報

    Returns:
        pd.DataFrame: weather/ground_state列を補完したレース情報
    """
    for col, default in {"weather": "-", "ground_state": "良"}.items():
        if col not in race_info_df.columns:
            race_info_df[col] = default
        else:
            race_info_df[col] = race_info_df[col].fillna(default)
    return race_info_df


def extract_peds_for_display(horse_peds):
    """出馬表用に父，母，母父のみ抽出する

    Args:
        horse_peds (pd.DataFrame): 血統データ（行 = peds_0..peds_61）

    Returns:
        pd.DataFrame: 表示用血統データ（行 = peds_0, peds_1, peds_4）
    """
    horse_peds_display = horse_peds.iloc[:5]
    horse_peds_display = horse_peds_display.drop(horse_peds_display.index[[2, 3]])
    horse_peds_display = horse_peds_display.rename(index={0: "父"})
    horse_peds_display = horse_peds_display.rename(index={1: "母"})
    horse_peds_display = horse_peds_display.rename(index={4: "母父"})
    return horse_peds_display
