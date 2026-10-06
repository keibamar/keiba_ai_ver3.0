"""新馬戦対象馬のプロフィール（生年月日）を一括スクレイピングするスクリプト

既存の horse_peds キャッシュに含まれる horse_id を対象に、
プロフィールキャッシュが未取得の馬だけ netkeiba からスクレイピングして保存する。

実行:
    python scripts/fetch_horse_profiles.py

オプション:
    --shinba_only   新馬戦出走歴のある馬のみ対象（デフォルト: 全馬）
    --force         既存キャッシュを上書きして再取得
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import paths
from src.managers.horse_profile_dataset_manager import fetch_profiles_for_horse_ids

HORSE_PEDS_DATA_PATH = os.path.join(paths.HORSE_DATA_PATH, "horse_peds")
RACE_RESULT_DATA_PATH = paths.RACE_RESULT_DATA_PATH


def get_all_peds_horse_ids():
    """horse_peds キャッシュに存在する全 horse_id を返す"""
    files = glob.glob(os.path.join(HORSE_PEDS_DATA_PATH, "*.csv"))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def get_shinba_horse_ids():
    """新馬戦出走歴のある horse_id を返す"""
    import pandas as pd
    files = glob.glob(os.path.join(RACE_RESULT_DATA_PATH, "*", "*_race_results.csv"))
    ids = set()
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, index_col=0)
            if "class" in df.columns and "horse_id" in df.columns:
                ids.update(df[df["class"] == "新馬"]["horse_id"].dropna().tolist())
        except Exception:
            pass
    return list(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shinba_only", action="store_true",
                        help="新馬戦出走馬のみ対象")
    parser.add_argument("--force", action="store_true",
                        help="既存キャッシュを上書きして再取得")
    args = parser.parse_args()

    if args.shinba_only:
        print("新馬戦出走馬の horse_id を収集中...")
        horse_ids = get_shinba_horse_ids()
    else:
        print("全 horse_peds キャッシュから horse_id を収集中...")
        horse_ids = get_all_peds_horse_ids()

    print(f"対象: {len(horse_ids)}頭")
    fetch_profiles_for_horse_ids(horse_ids, skip_existing=not args.force)


if __name__ == "__main__":
    main()
