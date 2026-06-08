import itertools
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from datetime import date, timedelta
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import name_header

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import race_results

import make_dataset

def prediction_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def get_lightGBM_model(place_id, type, length):
    """ lightGBMモデルの取得
        Args:
            place_id(int) : 開催コースid
            typer(str) : レースタイプ
            length(int) : キョリ
        Returns:
            model : lightGBMモデル
    """
    if type == "芝":
        type_str = "turf"
    else :
        type_str = "dirt"
    model_path = name_header.DATA_PATH + "/PredictionModels/LightGBM/Models/" + name_header.PLACE_LIST[place_id - 1] + '//' + str(type_str) + str(length) + "_lambdarank_model.txt"
    model = lgb.Booster(model_file = model_path)
    return model

def save_lightGBM_model(model, place_id, type, length):
    """学習済モデルの保存
        Args:
            model : lightGBMモデル
            place_id(int) : 開催コースid
            type(str) : レースタイプ
            length(int) : キョリ
    """
    try:
        if type == "芝":
                type = "turf"
        else :
            type = "dirt"
        model_path = name_header.DATA_PATH + "/PredictionModels/LightGBM/Models/" + name_header.PLACE_LIST[place_id - 1] + '//' + str(type) + str(length) + "_lambdarank_model.txt"
        model.booster_.save_model(model_path)
    except Exception as e:
        prediction_error(e)

def save_pred_rank(rank_df, place_id, year, type, length):
    """予想結果の保存
        Args:
            rank_df : 予想結果
            place_id(int) : 開催コースid
            year(int) : 年
            type(str) : レースタイプ
            length(int) : キョリ
    """
    rank_path = name_header.DATA_PATH + "/PredictionModels/LightGBM/Models/" + name_header.PLACE_LIST[place_id - 1] + "//"  + str(year) + "_" + str(type) + str(length) + "_pred_rank.csv"
    rank_df.to_csv(rank_path)

def rank_index(index_list):
    """スコアの順位を計算
        Args:
            index_list(list) : 指数リスト
        Returns:
            list : 降順にソートした指数リスト
    """
    rank = rankdata(index_list)
    # 降順の計算
    rank = (len(rank) - rank + 1).astype(int)
    return list(rank)

def merge_pred_rank(score_list, rank_list, result_df, type, length):
    """予想結果データセット(コース毎)のマージ
        Args:
            score_list(list) : 予想スコアリスト
            rank_list(list) : 予想順位リスト
            result_df(pd.DataFrame) : レース結果データセット
            type(str) : レースタイプ
            length(int) : コースのキョリ
        Returns:
            pred_rank(pd.DataFrame) : 予想結果データセット
    """
    # リストの１次元化
    score_list = list(itertools.chain.from_iterable(score_list))
    rank_list = list(itertools.chain.from_iterable(rank_list))     
    # 各レース結果の着順を取得
    result_df_course = result_df[result_df["race_type"] == type]
    result_df_course = result_df_course[result_df_course["course_len"] == str(length)]
    # レースidの取得
    race_id_list = pd.DataFrame(result_df_course.index, columns = ["race_id"])
    result_df_course = result_df_course[["着順","馬番"]].reset_index(drop = True)

    rank_df = pd.concat([race_id_list, result_df_course, pd.DataFrame(score_list, columns = ["score"]), pd.DataFrame(rank_list, columns = ["pred_rank"])], axis = 1)

    return rank_df

def data_group(df):
    """データのグループを作成
        Args:
            df(pd.DataFrame) : LightGBM用データセット
        Returns:
            x(pd.DataFrame) : 説明変数
            group(pd.DataFrame) :  各レースのデータ数(出走頭数)
    """
    # 説明変数の項目名を取得
    feature = list(df.drop(["race_id"], axis=1).columns)

    group = df['race_id'].value_counts()
    df = df.set_index(['race_id'])
    group = group.sort_index()

    # 説明変数(x)と目的変数(y)をインデックスでソート
    df.sort_index(inplace=True)
    # 説明変数(x)と目的変数(y)を設定
    x = df[feature]

    return x, group

def split_dataframe(race_data, race_flag):
    """訓練データとテストデータを分割
        Args:
            race_data(pd.DataFrame) : raceデータセット
            race_flag(pd.DataFrame) : flagデータセット
        Returns:
            data_train(pd.DataFrame) : raceデータの訓練データ
            data_test(pd.DataFrame) : raceデータのテストデータ
            flag_train(pd.DataFrame) : 訓練データのフラグ
            flag_test(pd.DataFrame) : テストデータのフラグ
    """
    test_ratio = 0.2
    test_size = int(len(race_data.index) * test_ratio)
    test_flag = True
    # 同レース内で分割されない用に調整
    while race_data.at[test_size - 1, "race_id"] == race_data.at[test_size, "race_id"]:
        test_size = test_size + 1
        if test_size == len(race_data.index):
            test_flag = False
            break
    # 訓練データとテストデータが分割できない場合はテストデータを空で返す
    if test_flag == False:
        data_train = race_data
        data_test = pd.DataFrame()
        flag_train = race_flag
        flag_test = pd.DataFrame()
    else :
        data_test = race_data.iloc[0 : test_size, :]
        data_train = race_data.iloc[test_size : len(race_data.index), :]
        flag_test = race_flag.iloc[0 : test_size, :]
        flag_train = race_flag.iloc[test_size : len(race_data.index), :]

    return data_train, data_test, flag_train, flag_test

def lightGBM_rank_train(place_id, year = date.today().year):
    """コースごとにモデルの学習（指定した年までのデータセットを利用)
        Args:
            place_id (int) : 開催コースid
            year(int) : 年（初期値：今年）
    """
    for type, length  in name_header.COURSE_LISTS[place_id - 1]:
        print(type, length)
        race_data, race_flag = pd.DataFrame(), pd.DataFrame()
        # データの読み込み
        for y in range(2020, int(year) + 1):
            race_data = pd.concat([race_data, make_dataset.get_LightGBM_dataset_csv(place_id, y, type, length)])
            race_flag = pd.concat([race_flag, make_dataset.get_LightGBM_dataset_flag_csv(place_id, y, type, length)])
        # nanを-1に変換
        race_data = race_data.fillna(-1)
        race_data = race_data.reset_index(drop = True)
        race_flag = race_flag.reset_index(drop = True)
        if race_data.empty or race_flag.empty:
            print(type, length, "no_dataset")
            continue
        # 訓練データとテストデータを分割
        data_train, data_test, flag_train, flag_test = split_dataframe(race_data, race_flag)
        # テストデータが空の場合学習をスキップ
        if data_test.empty or flag_test.empty :
            print(type, length, "train_skip")
            continue

        # データのグループを作成
        data_train, train_group = data_group(data_train)
        data_test, test_group = data_group(data_test)

        # モデルの学習
        model = lgb.LGBMRanker(random_state=0, force_col_wise=True)
        model.fit(data_train, flag_train, group = train_group, eval_set = [(data_test, flag_test)], eval_group = [list(test_group)])
        
        # モデルの保存
        save_lightGBM_model(model, place_id, type, length)

def prediction_rank(place_id, year = date.today().year):
    """ランキングの推定
        Args:
            place_id (int) : 開催コースid
            year(int) : 年（初期値：今年）
    """
    # ranking結果データセットの作成
    result_df = race_results.get_race_results_csv(place_id, year)

    for type, length  in name_header.COURSE_LISTS[place_id - 1]:
        print(type, length)
        # データの読み込み
        race_datas = make_dataset.get_LightGBM_dataset_csv(place_id, year, type, length)
        race_flags = make_dataset.get_LightGBM_dataset_flag_csv(place_id, year, type, length)
        if race_datas.empty or race_flags.empty:
            continue

        # データのグループを作成
        data_eval, eval_group = data_group(race_datas)
        # モデル読み込み
        model = get_lightGBM_model(place_id, type, length)

        start_index = 0
        rank_list = []
        score_list = []
        for race_num in range(len(eval_group)):
            # 1レース分のデータを取得
            data_race = data_eval.iloc[start_index : start_index + eval_group.iloc[race_num], : ]
            flag_race = race_flags.iloc[start_index : start_index + eval_group.iloc[race_num], : ].reset_index(drop = True)
            start_index += eval_group.iloc[race_num]
            
            # テストデータの予測
            pred_data = model.predict(data_race, num_iteration = model.best_iteration)
            rank = rank_index(pred_data)

            # 予想順位を保存
            score_list.append(pred_data)
            rank_list.append(rank)
            
            # 予想順位の評価
            row = pd.concat([pd.DataFrame(pred_data, columns = ["score"]), flag_race], axis = 1)
            row = row.sort_values(by="score", ascending=False).reset_index(drop = True)
        
        rank_df = merge_pred_rank(score_list, rank_list, result_df, type, length)
        
        # ranking結果データセットの出力
        save_pred_rank(rank_df, place_id, year, type, length)

def prediction_race_score(place_id, type, length, race_dataset):
    """レースのスコアを推定
        Args:
            place_id (int) : 開催コースid
            type(str) : 芝/ダート
            length(int) : キョリ
            race_dataset(pd.DataFrame)
    """
    try:
        race_dataset = race_dataset.fillna(-1)

        # モデル読み込み
        model = get_lightGBM_model(place_id, type, length)

        # テストデータの予測
        y_pred = model.predict(race_dataset, num_iteration = model.best_iteration)
        
        rank = rank_index(y_pred)
        # スコアとランキングを結合
        result_df = pd.concat([pd.DataFrame(y_pred, columns = ["score"]), pd.DataFrame(rank, columns = ["rank"])], axis = 1)

        return result_df
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()

if __name__ == "__main__":
    for place_id in range(1, len(name_header.PLACE_LIST) + 1):
        print(name_header.PLACE_LIST[place_id - 1])
        lightGBM_rank_train(place_id, 2023)
        prediction_rank(place_id, 2023)