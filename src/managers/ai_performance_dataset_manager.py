"""AI予想成績データセットの永続化（Manager層）

src.logic.calculators.ai_performance_calculator が計算する1レース単位の
的中・回収結果（calc_race_hit_returns）を、data/ai_performance/ai_performance.csv
に永続化する。ページ生成のたびに data/race_card/ を全件スキャンして都度計算する
（現状の ai_performance_report_generator / home_generator の挙動）のは遅いため、
update_ai_performance_dataset で事前に計算・保存し、get_ai_performance_dataset で
高速に読み込めるようにする。

予想または確定配当のいずれかが無いレース（calc_race_hit_returns が None を返す
レース）はデータセットに含めない。配当が後から確定したら次回の更新で追加される。
"""

import os

import pandas as pd

from src.config import paths
from src.logic.calculators import ai_performance_calculator
from src.utils.file_utils import read_csv_or_empty

AI_PERFORMANCE_DATASET_PATH = os.path.join(paths.AI_PERFORMANCE_DATA_PATH, "ai_performance.csv")

COLUMNS = [
    "race_day",
    "place_id",
    "win_hit",
    "win_return",
    "place_hit",
    "place_return",
    "trio_box_hit",
    "trio_box_return",
]


def get_ai_performance_dataset():
    """data/ai_performance/ai_performance.csv を取得する（race_idがindex）"""
    df = read_csv_or_empty(AI_PERFORMANCE_DATASET_PATH, dtype=str, index_col=0)
    if not df.empty:
        df.index = df.index.astype(str)
    return df


def save_ai_performance_dataset(df):
    """AI予想成績データセットをCSVに保存する"""
    os.makedirs(paths.AI_PERFORMANCE_DATA_PATH, exist_ok=True)
    df.to_csv(AI_PERFORMANCE_DATASET_PATH)


def _row_from_result(race_day, place_id, result):
    win_hit, win_return = result["win"]
    place_hit, place_return = result["place"]
    trio_box_hit, trio_box_return = result["trio_box"]
    return {
        "race_day": race_day,
        "place_id": place_id,
        "win_hit": win_hit,
        "win_return": win_return,
        "place_hit": place_hit,
        "place_return": place_return,
        "trio_box_hit": trio_box_hit,
        "trio_box_return": trio_box_return,
    }


def update_ai_performance_dataset():
    """予想済みレースのうち未集計のものについて的中・回収を計算し、データセットに追加する

    Returns:
        int: 新たに追加した行数
    """
    existing_df = get_ai_performance_dataset()
    existing_race_ids = set(existing_df.index.astype(str)) if not existing_df.empty else set()

    new_rows = {}
    for race_day, race_id in ai_performance_calculator.list_predicted_races():
        race_id = str(race_id)
        if race_id in existing_race_ids:
            continue

        result = ai_performance_calculator.calc_race_hit_returns(race_day, race_id)
        if result is None:
            continue

        place_id = ai_performance_calculator.parse_race_id(race_id)["place_id"]
        new_rows[race_id] = _row_from_result(race_day, place_id, result)

    if not new_rows:
        return 0

    new_df = pd.DataFrame.from_dict(new_rows, orient="index", columns=COLUMNS)
    combined_df = pd.concat([existing_df, new_df]) if not existing_df.empty else new_df
    save_ai_performance_dataset(combined_df)
    return len(new_rows)
