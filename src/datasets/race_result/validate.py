"""race_resultデータセットの検証ロジック"""

from src.datasets.race_result import model


def has_raw_columns(race_results_df):
    """race_results_dfがnetkeibaスクレイピング直後の生データとして
    必要な列を全て持っているか確認する

    Args:
        race_results_df (pd.DataFrame): 検証対象のDataFrame

    Returns:
        bool: RAW_COLUMNSを全て持っているか
    """
    return all(column in race_results_df.columns for column in model.RAW_COLUMNS)
