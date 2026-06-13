"""horseデータセットの検証ロジック"""

from src.datasets.horse import model


def has_past_performance_columns(past_performance_df):
    """past_performance_dfが新フォーマットの列を全て持っているか確認する"""
    return all(column in past_performance_df.columns for column in model.PAST_PERFORMANCE_COLUMNS)
