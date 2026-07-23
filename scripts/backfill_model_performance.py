"""MAR-hit / MAR-val の成績を既存 ai_performance.csv に遡り追加するスクリプト

既存行の hit_*/val_* 列が空のレースを対象に race_card CSV を読み直して計算する。
既に値が入っている行はスキップする（冪等）。

実行例:
  python scripts/backfill_model_performance.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.simplefilter("ignore")

import pandas as pd
from datetime import datetime

from src.managers import ai_performance_dataset_manager as m
from src.logic.calculators import ai_performance_calculator as calc


def main():
    df = m.get_ai_performance_dataset()
    if df.empty:
        print("ai_performance.csv が空です")
        return

    # hit_win_hit 列がない or 空の行を対象とする
    if "hit_win_hit" not in df.columns:
        df["hit_win_hit"]      = ""
        df["hit_win_return"]   = ""
        df["hit_place_hit"]    = ""
        df["hit_place_return"] = ""
        df["hit_trio_box_hit"]    = ""
        df["hit_trio_box_return"] = ""
        df["val_win_hit"]      = ""
        df["val_win_return"]   = ""
        df["val_place_hit"]    = ""
        df["val_place_return"] = ""
        df["val_trio_box_hit"]    = ""
        df["val_trio_box_return"] = ""

    target = df[df["hit_win_hit"].astype(str).isin(["", "nan"])]
    print(f"対象レース数: {len(target)} / {len(df)}")

    updated = 0
    skipped = 0
    for race_id, row in target.iterrows():
        race_id = str(race_id)
        try:
            race_day = datetime.strptime(str(row["race_day"])[:10], "%Y-%m-%d").date()
        except Exception:
            skipped += 1
            continue

        all_results = calc.calc_race_hit_returns_all_models(race_day, race_id)
        if all_results["mar"] is None:
            skipped += 1
            continue

        def _set(prefix, res):
            if res is None:
                return
            df.at[race_id, f"{prefix}win_hit"]      = res["win"][0]
            df.at[race_id, f"{prefix}win_return"]   = res["win"][1]
            df.at[race_id, f"{prefix}place_hit"]    = res["place"][0]
            df.at[race_id, f"{prefix}place_return"] = res["place"][1]
            df.at[race_id, f"{prefix}trio_box_hit"]    = res["trio_box"][0]
            df.at[race_id, f"{prefix}trio_box_return"] = res["trio_box"][1]

        _set("hit_", all_results["hit"])
        _set("val_", all_results["val"])
        updated += 1

        if updated % 100 == 0:
            print(f"  {updated}件処理中...")

    print(f"完了: {updated}件更新 / {skipped}件スキップ")
    m.save_ai_performance_dataset(df)
    print("ai_performance.csv を保存しました")


if __name__ == "__main__":
    main()
