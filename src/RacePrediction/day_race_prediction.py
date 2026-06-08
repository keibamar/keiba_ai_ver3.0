import sys

import pandas as pd

import warnings
warnings.simplefilter('ignore')

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")
import make_dataset
import prediction

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
    try:
        # datasetを格納するDataFrame
        race_dataset = pd.DataFrame()
        # レース情報の取得
        place_id = int(str(race_id)[4] + str(race_id)[5])
        course_info = [place_id, race_info_df.at[0,"race_type"], race_info_df.at[0,"course_len"], race_info_df.at[0,"ground_state"],race_info_df.at[0, "class"] ]
        #データセットの作成
        for horse_id in horse_ids:
            df_result = make_dataset.make_dataset_for_lightGBM(race_id, course_info, horse_id)
            race_dataset = pd.concat([race_dataset.reset_index(drop = True), df_result.reset_index(drop = True)])
    
        # 枠番・馬番の結合
        race_dataset = pd.concat([race_dataset.reset_index(drop = True), waku_df.reset_index(drop = True)], axis = 1)
       
        # レーススコアの予想
        rank_df =  prediction.prediction_race_score(place_id, course_info[1], course_info[2], race_dataset)
        
        return rank_df
    except Exception as e:
        day_race_prediction_error(e)
        return pd.DataFrame()

