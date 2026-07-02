"""raw_score_distribution.csv の再構築スクリプト

LightGBMモデルを再訓練した後、コース別の生スコア分布テーブルを最新のモデルで
再計算して上書き保存する。

使い方:
    python scripts/rebuild_raw_score_distribution.py

place_id を指定して1競馬場だけ更新したい場合:
    python scripts/rebuild_raw_score_distribution.py --place_id 7

"""

import argparse
import os
import sys
import warnings

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ver2.0のlibsが必要（name_header / make_datasetが依存）
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")

import numpy as np
import pandas as pd

from src.config import paths

# LightGBM側のモジュール（ver2.0のlibs等に依存するのでここで遅延import）
sys.path.append(os.path.join(PROJECT_ROOT, "src", "PredictionModels", "LightGBM"))
import make_dataset
import name_header
from prediction import get_lightGBM_model

COURSE_LISTS = name_header.COURSE_LISTS
PLACE_LIST = name_header.PLACE_LIST
TRAIN_YEARS = list(range(2020, 2026))


def _collect_raw_scores(place_id: int, race_type: str, course_len: str) -> list[float]:
    """指定コース・距離の全年分データをモデルで予測し、生スコアを返す。"""
    all_scores: list[float] = []
    try:
        model = get_lightGBM_model(place_id, race_type, int(course_len))
    except Exception as e:
        print(f"  モデル読み込み失敗 ({race_type}{course_len}m): {e}")
        return []

    for year in TRAIN_YEARS:
        df = make_dataset.get_LightGBM_dataset_csv(place_id, year, race_type, int(course_len))
        if df.empty:
            continue
        df = df.fillna(-1)
        # 訓練時と同様に race_id をインデックス列から除外
        if "race_id" in df.columns:
            df = df.drop("race_id", axis=1)
        try:
            scores = model.predict(df, num_iteration=model.best_iteration)
            all_scores.extend(scores.tolist())
        except Exception as e:
            print(f"  predict失敗 ({year} {race_type}{course_len}m): {e}")

    return all_scores


def rebuild(target_place_ids: list[int] | None = None) -> None:
    # 既存テーブルを読み込んで、更新対象以外の行は保持する
    dist_path = paths.RAW_SCORE_DISTRIBUTION_PATH
    if os.path.isfile(dist_path):
        existing_df = pd.read_csv(dist_path)
    else:
        existing_df = pd.DataFrame(
            columns=["place_id", "race_type", "course_len",
                     "n_hitrate", "mean_hitrate", "std_hitrate",
                     "n_value", "mean_value", "std_value"]
        )

    rows: list[dict] = []

    all_place_ids = list(range(1, len(PLACE_LIST) + 1))
    place_ids = target_place_ids if target_place_ids else all_place_ids

    for place_id in all_place_ids:
        for race_type, course_len in COURSE_LISTS[place_id - 1]:
            if place_id in place_ids:
                # 再計算
                print(f"計算中: place_id={place_id} {race_type}{course_len}m", end=" ... ", flush=True)
                raw_scores = _collect_raw_scores(place_id, race_type, course_len)
                if raw_scores:
                    arr = np.array(raw_scores)
                    row = {
                        "place_id": place_id,
                        "race_type": race_type,
                        "course_len": course_len,
                        "n_hitrate": len(arr),
                        "mean_hitrate": float(arr.mean()),
                        "std_hitrate": float(arr.std()),
                        "n_value": 0,
                        "mean_value": "",
                        "std_value": "",
                    }
                    print(f"n={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}")
                else:
                    print("データなし、スキップ")
                    row = None
            else:
                # 更新対象外 → 既存の行を再利用
                mask = (
                    (existing_df["place_id"].astype(str) == str(place_id)) &
                    (existing_df["race_type"].astype(str) == str(race_type)) &
                    (existing_df["course_len"].astype(str) == str(course_len))
                )
                matched = existing_df[mask]
                if matched.empty:
                    row = None
                else:
                    row = matched.iloc[0].to_dict()

            if row is not None:
                rows.append(row)

    out_df = pd.DataFrame(rows, columns=[
        "place_id", "race_type", "course_len",
        "n_hitrate", "mean_hitrate", "std_hitrate",
        "n_value", "mean_value", "std_value",
    ])
    os.makedirs(os.path.dirname(dist_path), exist_ok=True)
    out_df.to_csv(dist_path, index=False)
    print(f"\n保存完了: {dist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="raw_score_distribution.csv の再構築")
    parser.add_argument(
        "--place_id", type=int, nargs="+", default=None,
        help="更新対象の place_id（省略時は全競馬場）"
    )
    args = parser.parse_args()
    rebuild(target_place_ids=args.place_id)
