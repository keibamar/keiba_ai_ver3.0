"""make_v9_datasets_no26.py

v9 データセット（105列: v8+ペース適性7列）を全競馬場・全コース・2020〜2025年分作成する。

実行: python scripts/make_v9_datasets_no26.py
"""

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

import gc
import name_header
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab
from src.PredictionModels.LightGBM.make_dataset_v9 import (
    make_dataset_for_train_v9, load_dataset_v9,
)

_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "prediction", "datasets")


def _missing_courses(place_id, year):
    """未作成のコース (race_type, length) リストを返す。"""
    out_dir = os.path.join(_DATASET_DIR, name_header.PLACE_LIST[place_id - 1])
    missing = []
    for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
        flag_path = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset_flag_v9.csv")
        if not os.path.isfile(flag_path):
            missing.append((race_type, length))
    return missing

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))


def main():
    print("=" * 60)
    print(f"v9 データセット作成  年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("特徴量: v8(98列) + ペース適性7列 = 105列")
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
            # コース単位で1件ずつ処理してメモリを解放
            for course in missing:
                try:
                    make_dataset_for_train_v9(place_id, year=year, vocab=vocab, course_filter=[course])
                    total_saved += 1
                except Exception:
                    print(f"    ERROR: {place_name} {year}年 {course}")
                    traceback.print_exc()
                    total_skipped += 1
                gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完了  経過時間: {elapsed/60:.1f}分")
    print(f"処理: {total_saved}件 / スキップ: {total_skipped}件")
    print("=" * 60)


if __name__ == "__main__":
    main()
