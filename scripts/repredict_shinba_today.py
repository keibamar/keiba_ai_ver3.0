"""本日の新馬戦を新モデルで再予想・CSV更新・HTML再生成

horse_weight を削除した新モデルで既存 race_card CSV のスコアを上書きする。
"""

import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from datetime import date

from src.managers import race_card_dataset_manager
from src.logic.prediction import race_prediction_engine
from src.logic.html_generator import race_page_generator, daily_index_generator

today = date(2026, 10, 24)

time_id_list = race_card_dataset_manager.get_time_id_list(today)

shinba_ids = []
for _, rid in time_id_list:
    info = race_card_dataset_manager.get_race_info_csv(str(rid))
    if not info.empty and info.at[0, "class"] == "新馬":
        shinba_ids.append(str(rid))

print(f"新馬戦: {shinba_ids}")

updated = 0
for race_id in shinba_ids:
    print(f"\n--- {race_id} ---")
    try:
        # 既存 race_card CSV を読む
        card_df = race_card_dataset_manager.get_race_cards(today, race_id)
        info_df = race_card_dataset_manager.get_race_info_csv(race_id)
        if card_df.empty or info_df.empty:
            print("  CSV なし → スキップ")
            continue

        # 予想に必要な列を抽出
        horse_ids = card_df["horse_id"].tolist() if "horse_id" in card_df.columns else []
        if not horse_ids:
            print("  horse_id なし → スキップ")
            continue

        waku_df = card_df[["枠", "馬番"]].copy() if "枠" in card_df.columns and "馬番" in card_df.columns else card_df.copy()
        if "馬体重(増減)" in card_df.columns:
            waku_df["馬体重(増減)"] = card_df["馬体重(増減)"]
        if "性齢" in card_df.columns:
            waku_df["性齢"] = card_df["性齢"]
        if "調教師" in card_df.columns:
            waku_df["調教師"] = card_df["調教師"]

        kinryo_series = card_df["斤量"].reset_index(drop=True) if "斤量" in card_df.columns else None
        jockey_ids = card_df["jockey_id"].reset_index(drop=True).tolist() if "jockey_id" in card_df.columns else None
        odds_series = pd.to_numeric(card_df["オッズ"], errors="coerce").reset_index(drop=True) if "オッズ" in card_df.columns else None

        waku_df = waku_df.reset_index(drop=True)

        # 新馬専用モデルで予想
        result = race_prediction_engine.shinba_rank_prediction(
            race_id=race_id,
            horse_ids=horse_ids,
            race_info_df=info_df,
            waku_df=waku_df,
            kinryo_series=kinryo_series,
            jockey_ids=jockey_ids,
            odds_series=odds_series,
        )

        if result is None or result.empty:
            print("  予想失敗 → スキップ")
            continue

        print(f"  予想完了 {len(result)}頭")
        print(result[["idx_hitrate", "rank_hitrate", "idx_value", "rank_value", "idx_mar", "rank_mar"]].to_string())

        # race_card CSV に上書き保存
        card_df_reset = card_df.reset_index(drop=True)
        for col in ["idx_hitrate", "rank_hitrate", "idx_value", "rank_value", "idx_mar", "rank_mar",
                    "score", "rank", "score_hitrate", "score_value", "p_ai", "p_market"]:
            if col in result.columns:
                card_df_reset[col] = result[col].values

        card_df_reset.index = [race_id] * len(card_df_reset)
        race_card_dataset_manager.save_race_cards(card_df_reset, today, race_id)
        print(f"  CSV 保存完了")
        updated += 1

    except Exception as e:
        import traceback
        print(f"  エラー: {e}")
        traceback.print_exc()

print(f"\n{updated}/{len(shinba_ids)} レース更新完了")

# HTML 再生成
print("\nHTML 再生成中...")
race_page_generator.make_daily_race_card_html(today)
daily_index_generator.make_daily_index_page(today)
print("HTML 再生成完了")
