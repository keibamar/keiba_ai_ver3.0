"""2026年 v11 データセット再生成スクリプト

make_dataset_v11 の修正（horse_id保存追加）後、
2026年分の v11 を再生成して _horseid_v11.csv を出力する。

実行:
    python scripts/make_v11_datasets_2026.py
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
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab
from src.PredictionModels.LightGBM.make_dataset_v11 import make_dataset_for_train_v11

TARGET_YEAR   = 2026
TARGET_VENUES = list(range(1, 11))


def main():
    print("=" * 60)
    print(f"v11 データセット再生成  年: {TARGET_YEAR}")
    print("(horse_id保存対応版)")
    print("=" * 60)

    print("\n[Step 1] 血統 vocab をロード...")
    vocab = build_pedigree_vocab()
    print(f"  vocab_size: {len(vocab)}")

    t0 = time.time()
    total_saved = total_skipped = 0

    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*55}\n[{place_name}]\n{'='*55}")
        try:
            make_dataset_for_train_v11(place_id, year=TARGET_YEAR, vocab=vocab)
            total_saved += 1
        except Exception:
            print(f"  ERROR: {place_name}")
            traceback.print_exc()
            total_skipped += 1

    elapsed = time.time() - t0
    print(f"\n完了: {total_saved}場  エラー: {total_skipped}場  経過: {elapsed/60:.1f}分")


if __name__ == "__main__":
    main()
