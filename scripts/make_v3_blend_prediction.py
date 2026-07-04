"""v3スコア + 今日のオッズ を確率ベースでブレンドした予想を作成する

ブレンド方式:
  p_ai     = softmax(v3_score)          ← AI の相対勝利確率
  p_market = (1/odds) / sum(1/odds)     ← 市場の含意確率（オーバーラウンド除去済み）
  p_combined = AI_WEIGHT * p_ai + (1 - AI_WEIGHT) * p_market  ← 高い順に良い

両方の信号を同じ確率スケールに揃えることで、整数順位平均の情報損失・タイを解消する。

実行:
    python scripts/make_v3_blend_prediction.py
    python scripts/make_v3_blend_prediction.py 20260704  # 日付を明示する場合
"""

import os
import sys
import warnings
warnings.simplefilter("ignore")

from datetime import date

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.scraping import netkeiba_scraper
from src.managers import race_card_dataset_manager

RACE_DAY_STR = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y%m%d")
RACE_DAY = date(int(RACE_DAY_STR[:4]), int(RACE_DAY_STR[4:6]), int(RACE_DAY_STR[6:8]))
V3_TEST_DIR = os.path.join(paths.RACE_CARD_DATA_PATH + "_v3_test", RACE_DAY_STR)
TARGET_PLACE_IDS = [2, 3, 10]  # 函館, 福島, 小倉

# AI スコアの重み（市場確率の重みは 1 - AI_WEIGHT）
AI_WEIGHT = 0.5


def softmax(x):
    """数値安定版 softmax"""
    x = np.array(x, dtype=float)
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / np.nansum(e)


def blend_proba(v3_scores, odds_vals):
    """
    v3_score（生スコア, 高い=良）と オッズ（倍, 低い=市場有力）を
    確率ベースでブレンドして合成確率を返す（高い=良い）。

    片方だけ欠損の馬は欠損側を相手に合わせる（欠損なし前提で rank を付ける）。

    Returns
    -------
    p_combined : np.ndarray  各馬の合成確率（高い=有力）
    p_ai       : np.ndarray  AI 確率
    p_market   : np.ndarray  市場含意確率
    """
    n = len(v3_scores)
    v3 = pd.to_numeric(v3_scores, errors="coerce").values
    od = pd.to_numeric(odds_vals,  errors="coerce").values

    # --- AI 確率: softmax(v3_score) ---
    valid_v3 = ~np.isnan(v3)
    p_ai = np.full(n, np.nan)
    if valid_v3.any():
        p_ai[valid_v3] = softmax(v3[valid_v3])

    # --- 市場確率: 1/odds を正規化 ---
    valid_od = (~np.isnan(od)) & (od > 0)
    p_market = np.full(n, np.nan)
    if valid_od.any():
        inv = 1.0 / od[valid_od]
        p_market[valid_od] = inv / inv.sum()

    # --- ブレンド ---
    both    = ~np.isnan(p_ai) & ~np.isnan(p_market)
    ai_only = ~np.isnan(p_ai) & np.isnan(p_market)
    mk_only = np.isnan(p_ai)  & ~np.isnan(p_market)

    p_combined = np.full(n, np.nan)
    p_combined[both]    = AI_WEIGHT * p_ai[both]    + (1 - AI_WEIGHT) * p_market[both]
    p_combined[ai_only] = p_ai[ai_only]
    p_combined[mk_only] = p_market[mk_only]

    return p_combined, p_ai, p_market


def run():
    if not os.path.isdir(V3_TEST_DIR):
        print(f"v3 テスト予想フォルダが見つかりません: {V3_TEST_DIR}")
        sys.exit(1)

    print("=" * 65)
    print(f"v3 確率ブレンド予想: {RACE_DAY_STR}")
    print(f"AI_WEIGHT={AI_WEIGHT:.0%}  方式: softmax(v3_score) + 1/odds 正規化")
    print("=" * 65)

    for place_id in TARGET_PLACE_IDS:
        place_name = PLACE_LIST[place_id - 1]
        place_prefix = RACE_DAY_STR[:4] + f"{place_id:02d}"
        files = sorted([
            f for f in os.listdir(V3_TEST_DIR)
            if f.startswith(place_prefix) and f.endswith(".csv")
        ])
        if not files:
            continue

        print(f"\n{'='*65}")
        print(f"  {place_name}")
        print(f"{'='*65}")

        for fname in files:
            race_id  = int(fname.replace(".csv", ""))
            race_num = int(fname.replace(".csv", "")[-2:])
            race_info_df = race_card_dataset_manager.get_race_info_csv(race_id)
            if race_info_df.empty:
                continue

            race_type  = race_info_df.at[0, "race_type"]
            course_len = str(race_info_df.at[0, "course_len"])
            ground     = race_info_df.at[0, "ground_state"]
            cls        = race_info_df.at[0, "class"]

            # v3 予想を読み込む（再実行時に古いブレンド列を削除）
            pred_df = pd.read_csv(os.path.join(V3_TEST_DIR, fname), index_col=0)
            for _col in ["現在オッズ", "現在人気", "p_ai", "p_market",
                         "blended_score", "blended_rank"]:
                if _col in pred_df.columns:
                    pred_df = pred_df.drop(columns=[_col])

            # 現在のオッズをネットから取得
            odds_df = netkeiba_scraper.scrape_race_card_odds(race_id)

            if not odds_df.empty and "馬番" in odds_df.columns:
                odds_df["馬番"] = odds_df["馬番"].astype(str).str.strip()
                pred_df["馬番"] = pred_df["馬番"].astype(str).str.strip()
                pred_df = pred_df.merge(
                    odds_df[["馬番", "オッズ", "人気"]].rename(
                        columns={"オッズ": "現在オッズ", "人気": "現在人気"}
                    ),
                    on="馬番", how="left"
                )
            else:
                pred_df["現在オッズ"] = np.nan
                pred_df["現在人気"]   = np.nan

            # 確率ベースのブレンド
            p_comb, p_ai, p_mkt = blend_proba(
                pred_df["v3_score"], pred_df["現在オッズ"]
            )
            pred_df["p_ai"]         = p_ai
            pred_df["p_market"]     = p_mkt
            pred_df["blended_score"] = p_comb
            # 合成確率が高い順に blended_rank（1=最有力）
            valid_mask = ~pd.isna(pred_df["blended_score"])
            pred_df["blended_rank"] = np.nan
            pred_df.loc[valid_mask, "blended_rank"] = (
                pred_df.loc[valid_mask, "blended_score"]
                .rank(ascending=False, method="min")
                .astype("Int64")
            )

            # 上書き保存
            pred_df.to_csv(os.path.join(V3_TEST_DIR, fname))

            # 表示
            print(f"\n  第{race_num}R  {race_type}{course_len}m  {ground}  {cls}")
            print(f"  {'馬番':>3} {'馬名':<16} {'騎手':<6} | {'v1':>3} | {'v3':>3} | {'人気':>3} | {'ブレンド':>5} | {'p_ai':>6} {'p_市場':>6} {'p_合成':>6}")
            print(f"  {'-'*75}")

            v1_col = "rank_hitrate" if "rank_hitrate" in pred_df.columns else "rank"
            disp = pred_df.sort_values("blended_rank")
            for _, row in disp.iterrows():
                umaban  = row.get("馬番", "?")
                name    = row.get("馬名", "?")
                jockey  = row.get("騎手", "?")
                v1_r    = row.get(v1_col, "?")
                v3_r    = row.get("v3_rank", "?")
                ninki   = row.get("現在人気", "?")
                blended = row.get("blended_rank", "?")
                odds_v  = row.get("現在オッズ", np.nan)
                pai     = row.get("p_ai", np.nan)
                pmkt    = row.get("p_market", np.nan)
                pcmb    = row.get("blended_score", np.nan)

                v1_s  = f"{int(v1_r)}"   if pd.notna(v1_r)   else "-"
                v3_s  = f"{int(v3_r)}"   if pd.notna(v3_r)   else "-"
                nk_s  = f"{int(float(ninki))}" if pd.notna(ninki) and str(ninki) not in ("**","","nan") else "-"
                bl_s  = f"{int(blended)}" if pd.notna(blended) else "-"
                od_s  = f"({odds_v}倍)"  if pd.notna(odds_v) and str(odds_v) not in ("---.-","","nan") else "        "
                pai_s = f"{pai:.3f}"     if pd.notna(pai)    else "  -  "
                pmk_s = f"{pmkt:.3f}"    if pd.notna(pmkt)   else "  -  "
                pcb_s = f"{pcmb:.3f}"    if pd.notna(pcmb)   else "  -  "
                print(f"  {umaban:>3} {str(name):<16} {str(jockey):<6} | {v1_s:>3} | {v3_s:>3} | {nk_s:>3} | {bl_s:>5} | {pai_s} {pmk_s} {pcb_s}  {od_s}")

    print(f"\n{'='*65}")
    print(f"完了。保存先: {V3_TEST_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
