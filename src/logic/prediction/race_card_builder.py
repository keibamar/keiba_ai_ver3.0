"""出馬表(race_card)の作成ロジック

旧 src/RacePrediction/race_card.py の make_race_card を移植したもの。
スクレイピングは src.logic.scraping.netkeiba_scraper.scrape_race_card、
レース情報トークンの解析・血統表示データの抽出は
src.datasets.race_card.transform に切り出している。
"""

import pandas as pd

from src.datasets.race_card import transform as race_card_transform
from src.logic.prediction import race_prediction_engine
from src.logic.scraping import netkeiba_scraper
from src.managers import horse_peds_dataset_manager, past_performance_dataset_manager


def make_race_card(race_id):
    """出馬表を作成する

    Args:
        race_id (str): race_id

    Returns:
        tuple または pd.DataFrame:
            (race_card_df, race_info_df): 出馬表データセット（過去成績に基づくAI予想・
            血統情報を含む）とレース情報のタプル。
            出馬表の取得に失敗した場合は pd.DataFrame()。
    """
    race_info, race_info_df, race_card_df = netkeiba_scraper.scrape_race_card(race_id)
    if race_info_df.empty or race_card_df.empty:
        print("Miss Make Race card")
        return pd.DataFrame()

    race_info_df = race_card_transform.fill_race_info_defaults(race_info_df)

    # 出走馬の過去成績と血統情報を取得
    horse_ids = race_card_df.at[str(race_id), "horse_id"]
    horse_peds_df = pd.DataFrame()
    for horse_id in horse_ids:
        past_performance_dataset_manager.ensure_past_performance_dataset(horse_id)
        horse_ped = horse_peds_dataset_manager.get_horse_peds_dataset(horse_id)
        horse_peds_df = pd.concat([horse_peds_df, horse_ped], axis=1)

    # 枠番、馬番を取得してAI予想
    waku_df = pd.concat(
        [race_card_df["枠"].reset_index(drop=True), race_card_df["馬番"].reset_index(drop=True)], axis=1
    )
    rank_df = race_prediction_engine.rank_prediction(race_id, horse_ids, race_info_df, waku_df)

    # 父，母，母父のみ抽出してデータセットを統合
    horse_peds_display = race_card_transform.extract_peds_for_display(horse_peds_df)
    race_card_df = pd.concat(
        [race_card_df.reset_index(drop=True), horse_peds_display.T.reset_index(drop=True)], axis=1
    )
    race_card_df = pd.concat([race_card_df, rank_df.reset_index(drop=True)], axis=1)

    return race_card_df, race_info_df
