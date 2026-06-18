"""AI予想成績（的中率・回収率）の集計

Oracleの予想（race_card_dataset_manager.get_race_cards のrank列、rank=1を本命とする）と
確定配当（race_info_dataset_manager.get_race_return_csv_for_race）を照合し、
単勝・複勝・三連複BOXの的中・回収を1レース単位、または複数レースまとめて集計する。

判定ロジック自体は src.output.return_report.py の get_win_result / get_place_result /
get_trio_box_result（1レース・1日単位の集計）と同じ考え方を、任意のレース集合
（年別・開催別・コース別等）に対して再利用できる形に切り出したもの。

過去の予想データ（出馬表+score/rank）は date/RaceCards/ から data/race_card/ に
移行済み（2024-10-20〜、161日分）。一方、配当の的中判定に使う
race_info_dataset_manager.get_race_return_csv_for_race（1レース単位の分割保存）は
現時点では限られた範囲しか蓄積されていない。本モジュールは「予想と確定配当の両方が
存在するレース」のみを対象に集計する。
"""

import math
import os
import re
from datetime import datetime

import pandas as pd

from src.config import paths
from src.managers import race_card_dataset_manager, race_info_dataset_manager

BET_TYPES = ("win", "place", "trio_box")


def parse_race_id(race_id):
    """race_id（YYYYCCTTDDNN形式）を年・開催場・開催回・開催日目・レース番号に分解する"""
    race_id = str(race_id)
    return {
        "year": int(race_id[0:4]),
        "place_id": int(race_id[4:6]),
        "times": int(race_id[6:8]),
        "days": int(race_id[8:10]),
        "race_num": int(race_id[10:12]),
    }


def filter_by_year(race_day_race_id_pairs, year):
    """指定した年のレースのみに絞り込む"""
    return [(day, rid) for day, rid in race_day_race_id_pairs if parse_race_id(rid)["year"] == year]


def filter_by_meeting(race_day_race_id_pairs, year, place_id, times):
    """指定した年・開催場・開催回のレースのみに絞り込む"""
    result = []
    for day, rid in race_day_race_id_pairs:
        parsed = parse_race_id(rid)
        if parsed["year"] == year and parsed["place_id"] == place_id and parsed["times"] == times:
            result.append((day, rid))
    return result


def filter_by_course(race_day_race_id_pairs, place_id, race_type, course_len):
    """指定した開催場・race_type・距離のレースのみに絞り込む

    race_idからは開催場のみ判定できるため、race_type・距離は
    race_card_dataset_manager.get_race_info_csv（保存済みのレース情報）を参照する。
    レース情報が無いレースは対象外とする。
    """
    result = []
    for day, rid in race_day_race_id_pairs:
        if parse_race_id(rid)["place_id"] != place_id:
            continue
        info_df = race_card_dataset_manager.get_race_info_csv(rid)
        if info_df.empty:
            continue
        row = info_df.iloc[0]
        if row.get("race_type") == race_type and str(row.get("course_len")) == str(course_len):
            result.append((day, rid))
    return result


def list_predicted_races():
    """data/race_card/ 配下に保存されている予想済みの (race_day, race_id) を全て列挙する

    Returns:
        list[tuple[date, str]]: (race_day, race_id) のリスト（日付昇順）
    """
    pairs = []
    if not os.path.isdir(paths.RACE_CARD_DATA_PATH):
        return pairs

    for date_str in sorted(os.listdir(paths.RACE_CARD_DATA_PATH)):
        date_dir = os.path.join(paths.RACE_CARD_DATA_PATH, date_str)
        if not os.path.isdir(date_dir):
            continue
        try:
            race_day = datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue
        for filename in sorted(os.listdir(date_dir)):
            if filename.endswith(".csv"):
                pairs.append((race_day, filename[:-4]))
    return pairs


def calc_race_hit_returns(race_day, race_id, box_num=5):
    """1レース分の的中・回収額を、予想rank1位の馬を基準に計算する

    Args:
        race_day (date): レース開催日
        race_id (str): race_id
        box_num (int): 三連複BOXの頭数（初期値5、上位5位までを軸とする）
    Returns:
        dict | None: {"win": (hit, return), "place": (hit, return), "trio_box": (hit, return)}
            （hitは0/1、returnは100円あたりの配当額）。予想または確定配当が
            存在しない場合はNoneを返す。
    """
    pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if "rank" not in pred_df.columns:
        return None

    returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if returns_df.empty:
        return None

    result = {bet_type: (0, 0.0) for bet_type in BET_TYPES}

    win_df = returns_df[returns_df["式別"] == "単勝"].reset_index(drop=True)
    for i in range(len(win_df)):
        num = int(win_df.at[i, "馬番"])
        if pred_df.at[num - 1, "rank"] == 1:
            result["win"] = (1, float(win_df.at[i, "配当"]))

    place_df = returns_df[returns_df["式別"] == "複勝"].reset_index(drop=True)
    for i in range(len(place_df)):
        num = int(place_df.at[i, "馬番"])
        if pred_df.at[num - 1, "rank"] == 1:
            payout = place_df.at[i, "配当"]
            if isinstance(payout, str):
                payout = re.sub(r"\D", "", payout)
            result["place"] = (1, float(payout))

    trio_df = returns_df[returns_df["式別"] == "三連複"].reset_index(drop=True)
    for i in range(len(trio_df)):
        num_list = re.findall(r"\d+", trio_df.at[i, "馬番"])
        ranks = [
            pred_df[pred_df["馬番"] == int(num)].reset_index(drop=True).at[0, "rank"] for num in num_list
        ]
        if all(rank <= box_num for rank in ranks):
            payout = trio_df.at[i, "配当"]
            if isinstance(payout, str):
                payout = re.sub(r"\D", "", payout)
            result["trio_box"] = (1, float(payout) / math.comb(box_num, 3))

    return result


def aggregate_ai_performance(race_day_race_id_pairs, box_num=5):
    """複数レースについて的中率・回収率を集計する

    Args:
        race_day_race_id_pairs (list[tuple[date, str]]): (race_day, race_id) のリスト
        box_num (int): 三連複BOXの頭数
    Returns:
        dict: {"win": {"hit_rate": float, "return_rate": float, "n": int}, "place": {...}, "trio_box": {...}}
            的中率・回収率は%表記（回収率100.0が損益分岐点）。
            n は予想・確定配当の両方が存在し集計対象になったレース数。
    """
    totals = {bet_type: [0, 0.0] for bet_type in BET_TYPES}
    valid_count = 0

    for race_day, race_id in race_day_race_id_pairs:
        result = calc_race_hit_returns(race_day, race_id, box_num=box_num)
        if result is None:
            continue
        valid_count += 1
        for bet_type in BET_TYPES:
            hit, payout = result[bet_type]
            totals[bet_type][0] += hit
            totals[bet_type][1] += payout

    if valid_count == 0:
        return {bet_type: {"hit_rate": 0.0, "return_rate": 0.0, "n": 0} for bet_type in BET_TYPES}

    return {
        bet_type: {
            "hit_rate": totals[bet_type][0] / valid_count * 100,
            "return_rate": totals[bet_type][1] / valid_count,
            "n": valid_count,
        }
        for bet_type in BET_TYPES
    }
