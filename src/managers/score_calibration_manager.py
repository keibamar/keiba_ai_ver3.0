"""AI指数（score_to_index）キャリブレーションテーブルのManager層

指数帯（10点刻み）ごとの単勝/複勝的中率の実績を保存・読み込みする。
テーブル自体は不定期に再生成するバッチ処理（過去レースのscoreと確定結果を
突き合わせて集計）で作るため、ここでは永続化済みCSVの読み書きのみを担う。
"""

import os

import pandas as pd

from src.config import paths
from src.utils.file_utils import read_csv_or_empty


def save_score_calibration(calibration_df):
    """指数帯ごとのキャリブレーションテーブルを保存する

    Args:
        calibration_df (pd.DataFrame): ["band_min", "band_max", "n", "win_rate",
            "place_rate"]列を持つDataFrame（band_min昇順）。
    """
    os.makedirs(os.path.dirname(paths.SCORE_CALIBRATION_PATH), exist_ok=True)
    calibration_df.to_csv(paths.SCORE_CALIBRATION_PATH, index=False)


def get_score_calibration():
    """指数帯ごとのキャリブレーションテーブルを取得する

    band_min/band_max/n/win_rate/place_rateはいずれも数値列のため、
    find_bandでの数値比較ができるようdtype=Noneでpandasに型推定させる
    （read_csv_or_emptyの既定dtype=strだと文字列のまま読み込まれてしまう）。

    Returns:
        pd.DataFrame: 保存済みテーブル（未生成の場合は空のDataFrame）
    """
    return read_csv_or_empty(paths.SCORE_CALIBRATION_PATH, dtype=None)


def find_band(index_value):
    """AI指数（0〜100の整数）に対応する指数帯の行を返す

    Args:
        index_value (int | None): race_prediction_engine.score_to_indexの戻り値。

    Returns:
        dict | None: {"band_min", "band_max", "n", "win_rate", "place_rate"}
            （該当する指数帯が無い、またはindex_valueがNoneの場合はNone）
    """
    if index_value is None:
        return None
    calibration_df = get_score_calibration()
    if calibration_df.empty:
        return None
    match = calibration_df[
        (calibration_df["band_min"] <= index_value) & (index_value <= calibration_df["band_max"])
    ]
    if match.empty:
        return None
    return match.iloc[0].to_dict()
