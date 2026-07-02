import itertools
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from datetime import date, timedelta
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
import name_header

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import race_results

import make_dataset

# 学習済みモデルの保存先（Webサイトのライブ予想エンジンが読みに行くパス）・
# データセットキャッシュの保存先は、いずれもver3.0のsrc.config.pathsを使う。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.config import paths as paths_v3  # noqa: E402

# 距離帯（同じ帯に入る距離同士は学習データを合算する）。コース・距離ごとに
# モデルを分けると、特に開催数の少ない競馬場では学習データが薄く、木をあまり
# 分岐できず「特徴量が違うのに同じスコアになる馬」が増えてしまう
# （的中率重視・回収率重視どちらのモデルでも起きていた問題）。近い距離同士は
# レースの性質も近いため、データを合算して学習データ量を増やす
# （学習済みモデルは引き続きコース・距離ごとに別ファイルとして保存し、
# ライブ予想エンジン側の読み込み方法は変更しない）。
# 旧: [(0,1400),(1400,1800),(1800,2200),...] → 1800mと1900mが同じ帯に入り、
# 両方全く同じ学習データ(1800+1900合算)で訓練されて同一モデルになっていた。
# 境界を1900mに移動することで 1800m→(1400,1900)帯(1400+1800合算)、
# 1900m→(1900,2200)帯(単独)に分けて別モデルにする。
DISTANCE_BANDS = [(0, 1400), (1400, 1900), (1900, 2200), (2200, 2800), (2800, 99999)]


def _distance_band(length):
    length = int(length)
    for low, high in DISTANCE_BANDS:
        if low <= length < high:
            return (low, high)
    return DISTANCE_BANDS[-1]


def _band_lengths(place_id, type_jp, length):
    """同じ競馬場・芝ダートで、lengthと同じ距離帯に入る距離の一覧を返す（length自身も含む）"""
    band = _distance_band(length)
    return [
        course_length for course_type, course_length in name_header.COURSE_LISTS[place_id - 1]
        if course_type == type_jp and _distance_band(course_length) == band
    ]

def prediction_error(e):
    """ エラー時動作を記載する 
        Args:
            e (Exception) : エラー内容 
    """
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")

def get_lightGBM_model(place_id, type, length, model_suffix=""):
    """ lightGBMモデルの取得
        Args:
            place_id(int) : 開催コースid
            typer(str) : レースタイプ
            length(int) : キョリ
            model_suffix(str) : モデルファイル名サフィックス（回収率重視モデル(サブB)は"_value"）
        Returns:
            model : lightGBMモデル
    """
    if type == "芝":
        type_str = "turf"
    else :
        type_str = "dirt"
    model_path = os.path.join(paths_v3.PREDICTION_MODEL_PATH, name_header.PLACE_LIST[place_id - 1], f"{type_str}{length}_lambdarank_model{model_suffix}.txt")
    model = lgb.Booster(model_file = model_path)
    return model

def save_lightGBM_model(model, place_id, type, length, model_suffix=""):
    """学習済モデルの保存
        Args:
            model : lightGBMモデル
            place_id(int) : 開催コースid
            type(str) : レースタイプ
            length(int) : キョリ
            model_suffix(str) : モデルファイル名サフィックス（回収率重視モデル(サブB)は"_value"）
    """
    try:
        if type == "芝":
                type = "turf"
        else :
            type = "dirt"
        model_dir = os.path.join(paths_v3.PREDICTION_MODEL_PATH, name_header.PLACE_LIST[place_id - 1])
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{type}{length}_lambdarank_model{model_suffix}.txt")
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

def _fit_ranker(params, data_train, flag_train, train_group, data_test, flag_test, test_group):
    """指定パラメータでLGBMRankerを学習する（early stoppingあり・Optuna探索用）
        Args:
            params(dict) : LGBMRankerに渡す追加パラメータ
            data_train, flag_train, train_group : 訓練データ
            data_test, flag_test, test_group : 評価データ
        Returns:
            model : 学習済みLGBMRanker
    """
    from src.config.constants import USE_GPU_TRAINING, LIGHTGBM_DEVICE
    device_kwargs = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}
    model = lgb.LGBMRanker(random_state=0, verbosity=-1, **device_kwargs, **params)
    model.fit(
        data_train, flag_train, group=train_group,
        eval_set=[(data_test, flag_test)], eval_group=[list(test_group)], eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)],
    )
    return model


def _fit_ranker_final(params, data_train, flag_train, train_group):
    """指定パラメータでLGBMRankerを学習する（early stopping なし・最終モデル保存用）

    Optunaで探索したパラメータ（特にn_estimators）を使い、early stoppingなしで
    必ずn_estimators本の木を学習する。NDCG@1はNDCG@1が木1本でピークになりやすく、
    early stoppingにより常にnum_trees=1になる問題を避けるための関数。

        Args:
            params(dict) : LGBMRankerに渡す追加パラメータ（n_estimatorsを含む）
            data_train, flag_train, train_group : 訓練データ（テストデータは使わない）
        Returns:
            model : 学習済みLGBMRanker
    """
    from src.config.constants import USE_GPU_TRAINING, LIGHTGBM_DEVICE
    device_kwargs = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}
    model = lgb.LGBMRanker(random_state=0, verbosity=-1, **device_kwargs, **params)
    model.fit(
        data_train, flag_train, group=train_group,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def tune_hyperparameters(data_train, flag_train, train_group, data_test, flag_test, test_group, n_trials=20):
    """OptunaでLGBMRankerのハイパーパラメータを探索する
        本命馬（rank=1の予想）を正しく当てられているかを示すNDCG@1を最大化する。
        小規模コースではmin_child_samples/reg_alpha/reg_lambdaの探索範囲が
        学習データ件数に対して大きすぎると、木がほとんど分岐できず多くの馬が
        同一の葉＝同一スコアになってしまう（「同じスコアの馬ばかりになる」原因）。
        これを避けるため、min_child_samplesの上限を学習データ件数に応じて絞り、
        正則化の探索範囲も狭め、さらに検証データでのスコアの一意性（ユニーク比率）
        にもボーナスを与えて、的中精度を保ったまま差別化されたスコアを優先する。
        Args:
            data_train, flag_train, train_group : 訓練データ
            data_test, flag_test, test_group : 評価データ
            n_trials(int) : 探索回数（多いほど精度は上がるが時間がかかる）
        Returns:
            dict : 最良パラメータ
    """
    min_child_samples_low = 2
    min_child_samples_high = max(min_child_samples_low, min(30, len(data_train) // 10))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", min_child_samples_low, min_child_samples_high
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        }
        model = _fit_ranker(params, data_train, flag_train, train_group, data_test, flag_test, test_group)
        ndcg1 = model.evals_result_["valid_0"]["ndcg@1"][model.best_iteration_ - 1]

        val_scores = model.predict(data_test, num_iteration=model.best_iteration_)
        unique_ratio = (
            len(np.unique(np.round(val_scores, 6))) / len(val_scores) if len(val_scores) > 0 else 1.0
        )

        # スコアの一意性のボーナスをndcg1（0〜1）に対してより強く効かせる
        # （0.15では同スコアの馬がまだ多く残っていたため引き上げ）
        return ndcg1 + 0.4 * unique_ratio

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def lightGBM_rank_train(place_id, year = date.today().year, n_trials = 20):
    """コースごとにモデルの学習（指定した年までのデータセットを利用)
        Args:
            place_id (int) : 開催コースid
            year(int) : 年（初期値：今年）
            n_trials(int) : Optunaのハイパーパラメータ探索回数
    """
    for type, length  in name_header.COURSE_LISTS[place_id - 1]:
        print(type, length)
        race_data, race_flag = pd.DataFrame(), pd.DataFrame()
        # データの読み込み（同じ距離帯の距離も合算して学習データ量を増やす）
        band_lengths = _band_lengths(place_id, type, length)
        for y in range(2020, int(year) + 1):
            for band_length in band_lengths:
                race_data = pd.concat([race_data, make_dataset.get_LightGBM_dataset_csv(place_id, y, type, band_length)])
                race_flag = pd.concat([race_flag, make_dataset.get_LightGBM_dataset_flag_csv(place_id, y, type, band_length)])
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

        # ハイパーパラメータをOptunaで探索し、最良パラメータで学習
        # Optuna探索はearly stoppingありで効率的に行い、
        # 最終モデルはearly stoppingなしでn_estimators本を確実に学習する
        best_params = tune_hyperparameters(data_train, flag_train, train_group, data_test, flag_test, test_group, n_trials=n_trials)
        model = _fit_ranker_final(best_params, data_train, flag_train, train_group)

        # モデルの保存
        save_lightGBM_model(model, place_id, type, length)

def lightGBM_rank_train_value(place_id, year = date.today().year, n_trials = 20):
    """コースごとに回収率重視モデル（サブB）を学習する（指定した年までのデータセットを利用)
        的中率重視モデル（lightGBM_rank_train）と学習の流れは同じだが、
        make_dataset.make_dataset_for_train_valueが作る「自レースの人気」付き
        データセットと、payoutを反映したvalue_gradeラベルを使う。
        Args:
            place_id (int) : 開催コースid
            year(int) : 年（初期値：今年）
            n_trials(int) : Optunaのハイパーパラメータ探索回数
    """
    for type, length  in name_header.COURSE_LISTS[place_id - 1]:
        print(type, length, "(value)")
        race_data, race_flag = pd.DataFrame(), pd.DataFrame()
        # データの読み込み（同じ距離帯の距離も合算して学習データ量を増やす）
        band_lengths = _band_lengths(place_id, type, length)
        for y in range(2020, int(year) + 1):
            for band_length in band_lengths:
                race_data = pd.concat([race_data, make_dataset.get_LightGBM_dataset_csv(place_id, y, type, band_length, suffix="_value")])
                race_flag = pd.concat([race_flag, make_dataset.get_LightGBM_dataset_flag_csv(place_id, y, type, band_length, suffix="_value")])
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

        # ハイパーパラメータをOptunaで探索し、最良パラメータで学習
        best_params = tune_hyperparameters(data_train, flag_train, train_group, data_test, flag_test, test_group, n_trials=n_trials)
        model = _fit_ranker(best_params, data_train, flag_train, train_group, data_test, flag_test, test_group)

        # モデルの保存（"_value"サフィックスで的中率重視モデルと分離）
        save_lightGBM_model(model, place_id, type, length, model_suffix="_value")

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