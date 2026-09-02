"""スコア列が欠損しているレースカードを再予想するスクリプト

NO_SCORE_COL（score列なし）または ALL_NAN（score列が全NaN）の
race_card CSV を対象に make_race_card を再実行してスコアを補完する。

Usage:
    python scripts/run_repatch_missing_scores.py
"""

import os
import sys
import warnings
import glob
import pandas as pd
from datetime import datetime, date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.logic.prediction.race_card_builder import make_race_card
from src.managers import race_card_dataset_manager


def _needs_repatch(csv_path):
    """score列がない、または全NaNならTrue"""
    try:
        df = pd.read_csv(csv_path, index_col=0)
        if "score" not in df.columns:
            return True
        if df["score"].isna().all():
            return True
        return False
    except Exception:
        return False


def _save_race_card(date_str, race_id, df):
    """race_card_dataset_manager 経由で保存"""
    race_day = datetime.strptime(date_str, "%Y%m%d").date()
    race_card_dataset_manager.save_race_cards(df, race_day, race_id)


def main():
    base = paths.RACE_CARD_DATA_PATH
    target_dates = []
    date_dirs = sorted(os.listdir(base))
    for d in date_dirs:
        if len(d) == 8 and d.isdigit() and d >= "20260725":
            target_dates.append(d)

    missing = []
    for date_dir in target_dates:
        csvs = sorted(glob.glob(os.path.join(base, date_dir, "*.csv")))
        for csv_path in csvs:
            if _needs_repatch(csv_path):
                race_id = os.path.splitext(os.path.basename(csv_path))[0]
                missing.append((date_dir, race_id, csv_path))

    print(f"スコア欠損レース: {len(missing)} 件")
    for date_dir, race_id, _ in missing:
        print(f"  {date_dir} {race_id}")

    ok_count = 0
    fail_list = []

    for date_dir, race_id, csv_path in missing:
        print(f"\n[再予想] {date_dir} {race_id}")
        try:
            result = make_race_card(race_id)
            if isinstance(result, pd.DataFrame) and result.empty:
                print(f"  → make_race_card 空DataFrame: {race_id}")
                fail_list.append((date_dir, race_id, "empty DataFrame"))
                continue
            if isinstance(result, tuple):
                race_card_df = result[0] if len(result) > 1 else result
            else:
                race_card_df = result

            if isinstance(race_card_df, pd.DataFrame) and not race_card_df.empty:
                if "score" in race_card_df.columns and not race_card_df["score"].isna().all():
                    _save_race_card(date_dir, race_id, race_card_df)
                    print(f"  → 保存完了 (score sample: {race_card_df['score'].head(3).tolist()})")
                    ok_count += 1
                else:
                    print(f"  → 再予想後もscoreがNaN: {race_id}")
                    fail_list.append((date_dir, race_id, "score still NaN after repatch"))
            else:
                print(f"  → 予想結果が空: {race_id}")
                fail_list.append((date_dir, race_id, "empty result"))
        except Exception as e:
            print(f"  → エラー: {e}")
            fail_list.append((date_dir, race_id, str(e)))

    print(f"\n=== 完了 ===")
    print(f"  成功: {ok_count} / {len(missing)}")
    if fail_list:
        print(f"  失敗 ({len(fail_list)} 件):")
        for d, r, reason in fail_list:
            print(f"    {d} {r}: {reason}")


if __name__ == "__main__":
    main()
