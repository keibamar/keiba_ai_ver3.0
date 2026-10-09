"""make_v12_datasets_no26.py

v12 データセット（115列: v11+birth_month）を全競馬場・全コース・2020〜2026年分作成する。

v11 データセットCSVを読み込んで birth_month を付加する高速拡張方式。
  所要時間: 約30〜60分（v11フルビルドの1/10以下）

前提: v11 データセットが全コース作成済みであること
  → scripts/make_v11_datasets_no26.py を先に実行

実行: python scripts/make_v12_datasets_no26.py
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

import name_header
from src.PredictionModels.LightGBM.make_dataset_v12 import make_dataset_for_train_v12

_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "prediction", "datasets")

TRAIN_YEARS   = list(range(2020, 2027))
TARGET_VENUES = list(range(1, 11))


def _missing_courses(place_id, year):
    """未作成のコース (race_type, length) リストを返す。"""
    out_dir = os.path.join(_DATASET_DIR, name_header.PLACE_LIST[place_id - 1])
    missing = []
    for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
        flag_path = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset_for_rank_v12.csv")
        if not os.path.isfile(flag_path):
            missing.append((race_type, length))
    return missing


def main():
    print("=" * 60)
    print(f"v12 データセット作成  年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("特徴量: v11(114列) + birth_month(1列) = 115列")
    print("方式: v11 CSV 拡張（高速）")
    print("=" * 60)

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
                total_skipped += len(name_header.COURSE_LISTS[place_id - 1])
                continue
            print(f"\n  {year}年 ... 未作成コース: {len(missing)}件")
            try:
                make_dataset_for_train_v12(place_id, year, course_filter=missing)
                total_saved += len(missing)
            except Exception as e:
                print(f"  ERROR {place_name} {year}: {e}")
                traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完了: 保存={total_saved}コース / スキップ={total_skipped}コース")
    print(f"所要時間: {elapsed/60:.1f}分")
    print("次のステップ:")
    print("  python scripts/train_v12birth_no26.py")
    print("  python scripts/train_v12birth_ev_no26.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
