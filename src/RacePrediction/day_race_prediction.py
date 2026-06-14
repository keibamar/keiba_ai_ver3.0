from src.logic.prediction import race_prediction_engine


def day_race_prediction_error(e):
    """ エラー時動作を記載する
        Args:
            e (Exception) : エラー内容
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")


def rank_prediction(race_id, horse_ids, race_info_df, waku_df):
    """AI予想のランキングを計算
        Args:
            race_id(int) : race_id
            horse_ids(int) : レース出走馬のhorse_id
            race_info_df(pd.DataFrame) : レース情報(place_id, 芝/ダ, キョリ, 馬場状態, クラス)
            waku_df(pd.DataFrame) : 枠順・馬番のデータセット

        Returns:
            rank_df(pd.DataFrame) : 予想結果データセット
    """
    return race_prediction_engine.rank_prediction(race_id, horse_ids, race_info_df, waku_df)
