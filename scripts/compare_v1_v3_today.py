"""v1 vs v3 本日予想の的中率比較スクリプト

レース終了後に実行する。
  - data/race_card_v3_test/20260704/ の v3 予想と v1 予想を読み込み
  - netkeiba 速報ページからレース結果をスクレイピング
  - 単勝的中（予想1位が1着）・複勝的中（予想1位が3着内）を集計して比較

実行例:
    python scripts/compare_v1_v3_today.py
    python scripts/compare_v1_v3_today.py 20260704    # 日付を指定する場合
"""

import os
import re
import sys
import warnings
warnings.simplefilter("ignore")

from datetime import date

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.scraping import netkeiba_scraper

RACE_DAY_STR = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y%m%d")
V3_TEST_DIR = os.path.join(paths.RACE_CARD_DATA_PATH + "_v3_test", RACE_DAY_STR)
TARGET_PLACE_IDS = [2, 3, 10]  # 函館, 福島, 小倉


def _to_rank(v):
    s = re.sub(r"\D", "", str(v))
    return int(s) if s else None


def fetch_result(race_id):
    """速報ページから着順を {馬番(str): 着順(int)} で返す"""
    df = netkeiba_scraper.scrape_day_race_result(race_id)
    if df is None or df.empty:
        return {}
    rank_map = {}
    for _, row in df.iterrows():
        umaban = str(row.get("馬番", "")).strip()
        rank = _to_rank(row.get("着順", ""))
        if umaban and rank is not None:
            rank_map[umaban] = rank
    return rank_map


def evaluate_race(pred_df, rank_map):
    """
    pred_df: race_card_v3_test の1レース分（馬番, rank_hitrate, v3_rank 列を含む）
    rank_map: {馬番(str): 着順(int)}

    Returns: dict with keys
        v1_tan_hit, v1_fuku_hit, v3_tan_hit, v3_fuku_hit  (bool)
        v1_pick, v3_pick  (予想1位の実際の着順)
    """
    if pred_df.empty or not rank_map:
        return None

    # v1 の予想1位馬を特定
    v1_col = "rank_hitrate" if "rank_hitrate" in pred_df.columns else "rank"
    v1_top = pred_df.loc[pred_df[v1_col].astype(float) == 1.0]
    v3_top = pred_df.loc[pred_df["v3_rank"].astype(float) == 1.0]

    def get_actual_rank(top_df):
        if top_df.empty:
            return None
        umaban = str(int(float(top_df.iloc[0]["馬番"]))).strip()
        return rank_map.get(umaban)

    v1_actual = get_actual_rank(v1_top)
    v3_actual = get_actual_rank(v3_top)

    return {
        "v1_tan_hit":  v1_actual == 1,
        "v1_fuku_hit": v1_actual is not None and v1_actual <= 3,
        "v3_tan_hit":  v3_actual == 1,
        "v3_fuku_hit": v3_actual is not None and v3_actual <= 3,
        "v1_pick_rank": v1_actual,
        "v3_pick_rank": v3_actual,
    }


def run():
    if not os.path.isdir(V3_TEST_DIR):
        print(f"v3 テスト予想フォルダが見つかりません: {V3_TEST_DIR}")
        sys.exit(1)

    print("=" * 65)
    print(f"v1 vs v3 的中率比較: {RACE_DAY_STR}")
    print("=" * 65)

    total = {"v1_tan": 0, "v1_fuku": 0, "v3_tan": 0, "v3_fuku": 0, "races": 0}

    for place_id in TARGET_PLACE_IDS:
        place_name = PLACE_LIST[place_id - 1]
        place_prefix = RACE_DAY_STR[:4] + f"{place_id:02d}"
        files = sorted([
            f for f in os.listdir(V3_TEST_DIR)
            if f.startswith(place_prefix) and f.endswith(".csv")
        ])
        if not files:
            continue

        venue_total = {"v1_tan": 0, "v1_fuku": 0, "v3_tan": 0, "v3_fuku": 0, "races": 0}

        print(f"\n{'='*65}")
        print(f"  {place_name}")
        print(f"{'='*65}")
        print(f"  {'R':>2}  {'v1本命':>5} {'v1単':>4} {'v1複':>4}  |  {'v3本命':>5} {'v3単':>4} {'v3複':>4}")
        print(f"  {'-'*50}")

        for fname in files:
            race_id = int(fname.replace(".csv", ""))
            race_num = int(fname.replace(".csv", "")[-2:])

            pred_df = pd.read_csv(os.path.join(V3_TEST_DIR, fname), index_col=0)
            rank_map = fetch_result(race_id)

            if not rank_map:
                print(f"  {race_num:>2}R  結果未取得")
                continue

            res = evaluate_race(pred_df, rank_map)
            if res is None:
                continue

            v1_p = res["v1_pick_rank"]
            v3_p = res["v3_pick_rank"]
            v1_t = "○" if res["v1_tan_hit"] else "×"
            v1_f = "○" if res["v1_fuku_hit"] else "×"
            v3_t = "○" if res["v3_tan_hit"] else "×"
            v3_f = "○" if res["v3_fuku_hit"] else "×"

            v1_str = f"{v1_p}着" if v1_p else "-"
            v3_str = f"{v3_p}着" if v3_p else "-"
            print(f"  {race_num:>2}R  {v1_str:>5}  {v1_t:>3}  {v1_f:>3}  |  {v3_str:>5}  {v3_t:>3}  {v3_f:>3}")

            venue_total["v1_tan"]  += res["v1_tan_hit"]
            venue_total["v1_fuku"] += res["v1_fuku_hit"]
            venue_total["v3_tan"]  += res["v3_tan_hit"]
            venue_total["v3_fuku"] += res["v3_fuku_hit"]
            venue_total["races"]   += 1

        n = venue_total["races"]
        if n > 0:
            print(f"  {'-'*50}")
            print(f"  合計({n}R)  v1: 単{venue_total['v1_tan']}/{n}  複{venue_total['v1_fuku']}/{n}"
                  f"  |  v3: 単{venue_total['v3_tan']}/{n}  複{venue_total['v3_fuku']}/{n}")
            for k in ("v1_tan", "v1_fuku", "v3_tan", "v3_fuku", "races"):
                total[k] += venue_total[k]

    n = total["races"]
    if n > 0:
        print(f"\n{'='*65}")
        print(f"  3場合計({n}R)")
        print(f"  v1:  単勝 {total['v1_tan']}/{n} ({100*total['v1_tan']/n:.1f}%)  "
              f"複勝 {total['v1_fuku']}/{n} ({100*total['v1_fuku']/n:.1f}%)")
        print(f"  v3:  単勝 {total['v3_tan']}/{n} ({100*total['v3_tan']/n:.1f}%)  "
              f"複勝 {total['v3_fuku']}/{n} ({100*total['v3_fuku']/n:.1f}%)")
        print("=" * 65)


if __name__ == "__main__":
    run()
