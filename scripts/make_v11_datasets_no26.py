"""make_v11_datasets_no26.py

v11 データセット（114列: v10+6列）を全競馬場・全コース・2020〜2025年分作成する。

追加特徴量:
  - trainer_win_rate   : 調教師×コース勝率
  - trainer_place_rate : 調教師×コース複勝率
  - same_cond_cnt5     : 過去5走中の同条件(コース×距離×馬場状態)出走数
  - same_cond_pr5      : 過去5走中の同条件複勝率
  - f_ground_delta     : 父の道悪適性指数（重+稍重+不良複勝率 − 良複勝率）
  - mf_ground_delta    : 母父の道悪適性指数

実行: python scripts/make_v11_datasets_no26.py
"""

import gc
import os
import sys
import time
import traceback
import warnings
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")

import name_header
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab
from src.PredictionModels.LightGBM.make_dataset_v11 import (
    make_dataset_for_train_v11, load_dataset_v11,
)

_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "prediction", "datasets")

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))


def _missing_courses(place_id, year):
    """未作成のコース (race_type, length) リストを返す。"""
    out_dir = os.path.join(_DATASET_DIR, name_header.PLACE_LIST[place_id - 1])
    missing = []
    for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
        flag_path = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset_flag_v11.csv")
        if not os.path.isfile(flag_path):
            missing.append((race_type, length))
    return missing


def main():
    print("=" * 60)
    print(f"v11 データセット作成  年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("特徴量: v10(108列) + 6列 = 114列")
    print("  追加: trainer_win/place_rate, same_cond_cnt5/pr5, f/mf_ground_delta")
    print("=" * 60)

    print("\n[Step 1] 血統 vocab をロード...")
    vocab = build_pedigree_vocab()
    print(f"  vocab_size: {len(vocab)}")

    t0 = time.time()
    total_saved = 0
    total_skipped = 0

    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*55}\n[{place_name}]\n{'='*55}")

        for year in TRAIN_YEARS:
            missing = _missing_courses(place_id, year)
            if not missing:
                print(f"\n  {year}年 ... スキップ（全コース作成済み）")
                continue
            print(f"\n  {year}年 ... 未作成コース: {len(missing)}件")
            for course in missing:
                try:
                    make_dataset_for_train_v11(
                        place_id, year=year, vocab=vocab, course_filter=[course]
                    )
                    total_saved += 1
                except Exception:
                    print(f"    ERROR: {place_name} {year}年 {course}")
                    traceback.print_exc()
                    total_skipped += 1
                gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完了  保存: {total_saved}件  スキップ: {total_skipped}件")
    print(f"所要時間: {elapsed/3600:.1f}時間")


if __name__ == "__main__":
    main()
