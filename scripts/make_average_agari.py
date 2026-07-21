"""make_average_agari.py

race_result から race_type × course_len × ground_state 別の平均上り3Fを集計し
data/race_info/average_agari/avg_agari.csv に保存する。

実行: python scripts/make_average_agari.py
"""

import glob
import os
import sys
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_PATH = os.path.join(PROJECT_ROOT, "data", "race_info", "average_agari", "avg_agari.csv")

GROUND_STATES = ["全", "良", "稍重", "重", "不良"]


def load_all_race_results():
    files = glob.glob(
        os.path.join(PROJECT_ROOT, "data", "race_result", "**", "*.csv"),
        recursive=True,
    )
    print(f"race_result ファイル数: {len(files)}")
    dfs = []
    skipped = 0
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, encoding="utf-8-sig")
            if df.empty or "race_type" not in df.columns or "course_len" not in df.columns:
                skipped += 1; continue
            agari_col = df.columns[-1]
            vals = pd.to_numeric(df[agari_col], errors="coerce").dropna()
            if vals.empty:
                skipped += 1; continue
            # 上り3Fタイムの妥当範囲: 30〜50秒。範囲外なら別のカラムなのでスキップ
            median_val = vals.median()
            if not (28 <= median_val <= 55):
                skipped += 1; continue
            sub = df[["race_type", "course_len", "ground_state"]].copy()
            sub["agari"] = vals.reindex(df.index)
            dfs.append(sub)
        except Exception as e:
            skipped += 1
    print(f"  有効: {len(dfs)}ファイル / スキップ: {skipped}ファイル")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def calc_avg_agari(df_all):
    """race_type × course_len × ground_state 別の平均上り3Fを計算する。
    ground_state='全' は馬場状態を問わない全平均。
    """
    rows = []
    for (rt, cl), grp in df_all.dropna(subset=["agari"]).groupby(["race_type", "course_len"]):
        avg_all = grp["agari"].mean()
        rows.append({
            "race_type": rt, "course_len": int(cl),
            "ground_state": "全", "avg_agari": round(avg_all, 4),
        })
        for gs in ["良", "稍重", "重", "不良"]:
            sub = grp[grp["ground_state"] == gs]
            avg_gs = sub["agari"].mean() if not sub.empty else avg_all
            rows.append({
                "race_type": rt, "course_len": int(cl),
                "ground_state": gs, "avg_agari": round(avg_gs, 4),
            })
    return pd.DataFrame(rows)


def main():
    print("race_result ロード中...")
    df_all = load_all_race_results()
    if df_all.empty:
        print("データなし"); return

    print(f"  総行数: {len(df_all)}, 上りデータ有: {df_all['agari'].notna().sum()}")

    print("集計中...")
    df_avg = calc_avg_agari(df_all)
    print(f"  集計行数: {len(df_avg)}")
    print(df_avg.head(10).to_string())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_avg.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
