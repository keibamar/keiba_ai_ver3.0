"""LightGBM学習データセット生成 v5

v4 の特徴量に血統カテゴリカル特徴量を追加:
  - father_cat        : 父名の整数ID（0=不明）
  - mother_father_cat : 母父名の整数ID（0=不明）
  - paternal_gf_cat   : 父父名の整数ID（0=不明）

血統名→整数IDのマッピング（vocab）は pedigree_vocab_v5.json として
PREDICTION_MODEL_PATH に保存し、推論時も同じファイルを参照する。
LightGBM の categorical_feature 指定により、モデルが種牡馬ごとの
傾向を直接学習できる（現行の着度数統計と相補的な情報）。

v1=51列、v2=62列、v3=65列、v4=67列、v5=70列。
データは "_v5" サフィックスで保存。
"""

import glob
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import name_header
import race_results
import horse_peds as horse_peds_lib

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.make_dataset_v3 import (
    make_dataset_for_lightGBM_v3,
    build_jockey_course_stats,
)
from src.PredictionModels.LightGBM.make_dataset_v4 import (
    index_v4,
    _parse_odds,
    _parse_popularity,
)

# v5 特徴量列名（v4の67列 + 3列）
CAT_COLS = ["father_cat", "mother_father_cat", "paternal_gf_cat"]
index_v5 = index_v4 + CAT_COLS

VOCAB_PATH = os.path.join(paths_v3.PREDICTION_MODEL_PATH, "pedigree_vocab_v5.json")
_HORSE_PEDS_DIR = os.path.join(name_header.DATA_PATH, "HorsePeds")

# horse_peds CSV内でのインデックス位置（peds_0=父, peds_2=父父, peds_4=母父）
_PEDS_IDX = {"father": 0, "paternal_gf": 2, "mother_father": 4}


# ---------- vocab 管理 ----------

def build_pedigree_vocab(force_rebuild: bool = False) -> dict:
    """全HorsePedsファイルをスキャンして血統名→IDのvocabを構築・保存する。

    既にVOCAB_PATHが存在しforce_rebuild=Falseなら既存vocabを読み込んで返す。
    IDs は 1 始まり（0=不明/未知）で、名前のアルファベット順に固定する。
    """
    if not force_rebuild and os.path.isfile(VOCAB_PATH):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"vocab読み込み済み: {len(vocab)}種類 ({VOCAB_PATH})")
        return vocab

    csv_files = glob.glob(os.path.join(_HORSE_PEDS_DIR, "*.csv"))
    all_names: set = set()

    for csv_path in tqdm(csv_files, desc="pedigree vocab構築"):
        try:
            df = pd.read_csv(csv_path, index_col=0, dtype=str)
            if df.empty or df.shape[1] == 0:
                continue
            data = df[df.columns[0]].tolist()
            for idx in _PEDS_IDX.values():
                if len(data) > idx:
                    raw = data[idx]
                    if pd.notna(raw) and str(raw) not in ("nan", ""):
                        name = horse_peds_lib.normalize_horse_name_strings(str(raw))
                        if name:
                            all_names.add(name)
        except Exception:
            continue

    sorted_names = sorted(all_names)
    vocab = {name: i + 1 for i, name in enumerate(sorted_names)}

    os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"vocab構築完了: {len(vocab)}種類 → {VOCAB_PATH}")
    return vocab


def load_vocab() -> dict:
    """保存済みvocabをロードする。未生成なら空dictを返す。"""
    if not os.path.isfile(VOCAB_PATH):
        return {}
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------- 血統エンコード ----------

def encode_ancestor(raw_name, vocab: dict) -> int:
    """血統名を整数IDに変換する。不明・未知は0を返す。"""
    if pd.isna(raw_name) or str(raw_name) in ("nan", ""):
        return 0
    name = horse_peds_lib.normalize_horse_name_strings(str(raw_name))
    return vocab.get(name, 0)


def get_pedigree_cats(horse_id, vocab: dict) -> tuple:
    """馬IDから (father_cat, mother_father_cat, paternal_gf_cat) を返す。

    v2ライブラリ（C:/keiba_ai/keiba_ai_ver2.0/data/HorsePeds/）にデータがない場合は
    v3 horse_peds_dataset_manager 経由でスクレイピングして取得する。
    """
    df = horse_peds_lib.get_horse_peds_csv(str(horse_id))
    if df.empty or df.shape[1] == 0:
        try:
            from src.managers import horse_peds_dataset_manager as _hpdm
            df = _hpdm.get_horse_peds_dataset(str(horse_id))
        except Exception:
            pass
    if df.empty or df.shape[1] == 0:
        return 0, 0, 0
    data = df[df.columns[0]].tolist()
    father      = data[_PEDS_IDX["father"]]      if len(data) > _PEDS_IDX["father"]      else None
    paternal_gf = data[_PEDS_IDX["paternal_gf"]] if len(data) > _PEDS_IDX["paternal_gf"] else None
    mother_father = data[_PEDS_IDX["mother_father"]] if len(data) > _PEDS_IDX["mother_father"] else None
    return (
        encode_ancestor(father, vocab),
        encode_ancestor(mother_father, vocab),
        encode_ancestor(paternal_gf, vocab),
    )


# ---------- データセット保存・ロード ----------

def save_dataset_v5(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v5.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v5.csv")


def load_dataset_v5(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v5.csv"
    flag_path = base + "_flag_v5.csv"
    df   = pd.read_csv(df_path,   index_col=0)          if os.path.isfile(df_path)   else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int) if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


# ---------- データセット生成 ----------

def make_dataset_for_train_v5(place_id, year=date.today().year, vocab=None, course_filter=None):
    """指定競馬場・年の v5 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

    if vocab is None:
        vocab = build_pedigree_vocab()

    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print(f"  データなし: {year} {name_header.PLACE_LIST[place_id - 1]}")
        return

    courses = name_header.COURSE_LISTS[place_id - 1]
    if course_filter is not None:
        courses = [(t, l) for t, l in courses if (t, l) in course_filter]

    print(f"  騎手×コース成績テーブル構築中...")
    jockey_lookup = build_jockey_course_stats(place_id, year)
    print(f"  ルックアップエントリ数: {len(jockey_lookup)}")

    for race_type, length in courses:
        df_course = df_results[
            (df_results["race_type"] == race_type) & (df_results["course_len"] == length)
        ]
        if df_course.empty:
            continue

        print(f"  {race_type}{length}m ({len(df_course)}走)")
        race_id_list = df_course.index.tolist()
        df_dataset = pd.DataFrame()
        flag_list = []
        jockey_wins, jockey_places = [], []
        odds_list, popularity_list = [], []
        father_ids, mf_ids, pgf_ids = [], [], []

        for i in tqdm(range(len(df_course))):
            df_result = df_course.iloc[i:i + 1]
            race_id   = int(df_result.index[0])
            flag_list.append(relevance_grade(df_result, race_id))

            course_info = [
                place_id,
                df_result.iloc[0]["race_type"],
                df_result.iloc[0]["course_len"],
                df_result.iloc[0]["ground_state"],
                df_result.iloc[0]["class"],
            ]
            horse_id = df_result.iloc[0]["horse_id"]
            row = make_dataset_for_lightGBM_v3(race_id, course_info, horse_id)
            df_dataset = pd.concat([df_dataset, row])

            key = (str(race_id), str(horse_id))
            jw, jp = jockey_lookup.get(key, (np.nan, np.nan))
            jockey_wins.append(jw)
            jockey_places.append(jp)

            odds_list.append(_parse_odds(df_result.iloc[0].get("単勝", np.nan)))
            popularity_list.append(_parse_popularity(df_result.iloc[0].get("人気", "")))

            f_cat, mf_cat, pgf_cat = get_pedigree_cats(horse_id, vocab)
            father_ids.append(f_cat)
            mf_ids.append(mf_cat)
            pgf_ids.append(pgf_cat)

        df_dataset = pd.concat([
            pd.DataFrame(race_id_list,        columns=["race_id"]),
            df_dataset.reset_index(drop=True),
            pd.Series(jockey_wins,            name="jockey_win_rate"),
            pd.Series(jockey_places,          name="jockey_place_rate"),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
            pd.Series(odds_list,              name="current_odds"),
            pd.Series(popularity_list,        name="current_popularity"),
            pd.Series(father_ids,             name="father_cat"),
            pd.Series(mf_ids,                 name="mother_father_cat"),
            pd.Series(pgf_ids,                name="paternal_gf_cat"),
        ], axis=1)
        df_dataset.columns = index_v5

        save_dataset_v5(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
