"""race_infoデータセットの検証ロジック"""

from src.datasets.race_info import model


def has_horse_id_map_columns(horse_id_map_df):
    """horse_id_map_dfが必要な列を全て持っているか確認する"""
    return all(column in horse_id_map_df.columns for column in model.HORSE_ID_MAP_COLUMNS)
