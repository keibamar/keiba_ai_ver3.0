"""backfill_multimodel_cols_2026.py

2026年全レースの race_card CSV に3戦略マルチモデル指数列を追加し、
HTMLを再生成する。

処理の流れ:
  1. race_records_unified_2026.pkl / race_records_nodds_2026.pkl を読み込む
  2. data/race_card/2026*/ を全スキャンして CSV を処理
     - 各CSVに idx_hitrate / rank_hitrate / idx_value /
       rank_value / idx_mar / rank_mar 列を追加して上書き保存
  3. 2026年の全日付で make_daily_race_card_html を実行してHTML再生成

実行:
  python scripts/backfill_multimodel_cols_2026.py
"""

import glob
import os
import pickle
import sys
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.logic.html_generator import race_page_generator
from src.logic.prediction.race_prediction_engine import score_to_index

EVAL_YEAR = 2026


# ── ユーティリティ ──────────────────────────────────────────────────────────

def _norm(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-12:
        return np.zeros(len(arr))
    return (arr - mn) / (mx - mn)


def _blend(s_vx, s_v7, alpha):
    return (1.0 - alpha) * _norm(s_vx) + alpha * _norm(s_v7)


def _strategy_score(s_tan, s_san):
    """本命（argmax s_tan）を s_san 最大値より上に引き上げる"""
    s_tan_n = _norm(s_tan)
    s_san_n = _norm(s_san)
    honmei = int(np.argmax(s_tan_n))
    combined = s_san_n.copy()
    spread = combined.max() - combined.min() + 1e-6
    combined[honmei] = combined.max() + spread
    return combined


def _rank_index(score_arr):
    rank = rankdata(score_arr)
    return list((len(rank) - rank + 1).astype(int))


def _zscore_to_idx(combined):
    s = pd.Series(combined)
    std = s.std()
    if not std or pd.isna(std):
        z = pd.Series([0.0] * len(s))
    else:
        z = (s - s.mean()) / std
    return [score_to_index(float(v)) for v in z]


def compute_multi_cols(ur, nr):
    """unified / nodds レコードから6列を計算する。エラー時は None を返す"""
    if ur is None:
        return None

    s_v7  = ur.get("s_v7")
    s_v11 = ur.get("s_v11")
    s_v12 = ur.get("s_v12")
    s_v15 = ur.get("s_v15")
    s_v11n = nr.get("s_v11n") if nr else None
    s_v12n = nr.get("s_v12n") if nr else None

    if s_v7 is None or s_v11 is None or s_v12 is None or s_v15 is None:
        return None

    # ② 的中率重視: 単複=v11α0.6 / 3連複=v15α0.5
    comb_hr  = _strategy_score(_blend(s_v11, s_v7, 0.6), _blend(s_v15, s_v7, 0.5))

    # ①③ 回収率重視: 単複=v12n / 3連複=v12n（nodds なければ v12 で代替）
    sv = _norm(s_v12n) if s_v12n is not None else _norm(s_v12)
    comb_val = _strategy_score(sv, sv)

    # ④ MAR推奨: 単複=v11n / 3連複=v12α0.4
    sm = _norm(s_v11n) if s_v11n is not None else _norm(s_v11)
    comb_mar = _strategy_score(sm, _blend(s_v12, s_v7, 0.4))

    return {
        "idx_hitrate":  _zscore_to_idx(comb_hr),
        "rank_hitrate": _rank_index(comb_hr),
        "idx_value":    _zscore_to_idx(comb_val),
        "rank_value":   _rank_index(comb_val),
        "idx_mar":      _zscore_to_idx(comb_mar),
        "rank_mar":     _rank_index(comb_mar),
    }


# ── STEP 1: CSV 更新 ────────────────────────────────────────────────────────

def step1_update_csvs(unified_map, nodds_map):
    rc_root = paths.RACE_CARD_DATA_PATH
    pattern = os.path.join(rc_root, f"{EVAL_YEAR}*", "*.csv")
    all_csvs = sorted(glob.glob(pattern))
    print(f"  対象CSV: {len(all_csvs)}件")

    updated = skipped = 0
    for csv_path in all_csvs:
        race_id = os.path.basename(csv_path).replace(".csv", "")
        ur = unified_map.get(race_id)
        if ur is None:
            skipped += 1
            continue

        nr = nodds_map.get(race_id)
        cols_data = compute_multi_cols(ur, nr)
        if cols_data is None:
            skipped += 1
            continue

        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception as e:
            print(f"  ⚠️ 読み込み失敗: {race_id} ({e})")
            skipped += 1
            continue

        if "馬番" not in df.columns:
            skipped += 1
            continue

        # pkl の umabans リスト → 馬番: index マップ
        umabans = [str(int(float(u))) if str(u).replace(".", "").isdigit() else str(u)
                   for u in ur.get("umabans", [])]
        if len(umabans) == 0 or len(umabans) != len(cols_data["idx_hitrate"]):
            skipped += 1
            continue
        ub_to_idx = {ub: i for i, ub in enumerate(umabans)}

        # 各行の馬番でマッチング
        new_cols = {c: [None] * len(df) for c in cols_data}
        for i in range(len(df)):
            try:
                ub = str(int(float(df.iloc[i]["馬番"])))
            except Exception:
                continue
            pkl_i = ub_to_idx.get(ub)
            if pkl_i is None:
                continue
            for col, vals in cols_data.items():
                new_cols[col][i] = vals[pkl_i]

        for col, vals in new_cols.items():
            df[col] = vals

        try:
            df.to_csv(csv_path)
            updated += 1
        except Exception as e:
            print(f"  ⚠️ 保存失敗: {race_id} ({e})")
            skipped += 1

        if updated % 100 == 0 and updated > 0:
            print(f"    {updated}R 更新済み...")

    print(f"  → 更新: {updated}R  /  スキップ: {skipped}R")
    return updated


# ── STEP 2: HTML 再生成 ──────────────────────────────────────────────────────

def step2_regenerate_html():
    rc_root = paths.RACE_CARD_DATA_PATH
    date_dirs = sorted(
        d for d in os.listdir(rc_root)
        if d.startswith(str(EVAL_YEAR)) and len(d) == 8 and d.isdigit()
    )
    print(f"  対象日付: {len(date_dirs)}日")

    for date_str in date_dirs:
        try:
            d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
        except ValueError:
            continue
        try:
            race_page_generator.make_daily_race_card_html(d)
        except Exception as e:
            print(f"  ⚠️ HTML生成エラー: {date_str} ({e})")

    print(f"  → {len(date_dirs)}日分のHTML生成完了")


# ── エントリーポイント ────────────────────────────────────────────────────────

def main():
    print(f"=== backfill_multimodel_cols_{EVAL_YEAR}.py ===\n")

    print("キャッシュ読み込み中...")
    with open(os.path.join(PROJECT_ROOT, "logs", f"race_records_unified_{EVAL_YEAR}.pkl"), "rb") as f:
        unified_list = pickle.load(f)
    try:
        with open(os.path.join(PROJECT_ROOT, "logs", f"race_records_nodds_{EVAL_YEAR}.pkl"), "rb") as f:
            nodds_list = pickle.load(f)
    except FileNotFoundError:
        nodds_list = []
        print("  ⚠️ nodds pkl なし")

    unified_map = {r["race_id"]: r for r in unified_list}
    nodds_map   = {r["race_id"]: r for r in nodds_list}
    print(f"  unified: {len(unified_map)}R  /  nodds: {len(nodds_map)}R\n")

    print("[STEP 1] race_card CSV に指数列を追加...")
    step1_update_csvs(unified_map, nodds_map)

    print("\n[STEP 2] HTML 再生成...")
    step2_regenerate_html()

    print("\n=== 完了! ===")


if __name__ == "__main__":
    main()
