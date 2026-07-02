"""過去の中京ダート1800/1900mレースを新モデルで再予測してrace_card CSVを更新する

再訓練後に既存の出馬表CSVのスコアを新モデルの出力で上書きする。
その後、該当日の出馬表HTMLとAI成績HTMLを再生成する。

実行例:
    python scripts/repred_chukyo_dirt_1800_1900.py
"""

import os
import sys
import warnings

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from datetime import datetime

from src.config import paths
from src.managers import race_card_dataset_manager
from src.logic.prediction import race_prediction_engine
from src.logic.html_generator import (
    ai_performance_report_generator,
    daily_index_generator,
    race_page_generator,
)
from src.managers import ai_performance_dataset_manager

PLACE_ID = 7  # 中京
TARGET_LENS = {"1400", "1800", "1900"}


def collect_target_races():
    """保存済みrace_card CSVから対象レースを収集する"""
    race_card_dir = paths.RACE_CARD_DATA_PATH
    target = []
    for date_str in sorted(os.listdir(race_card_dir)):
        date_path = os.path.join(race_card_dir, date_str)
        if not os.path.isdir(date_path):
            continue
        for fname in os.listdir(date_path):
            if not fname.endswith(".csv"):
                continue
            race_id = fname.replace(".csv", "")
            if len(race_id) < 6 or race_id[4:6] != f"{PLACE_ID:02d}":
                continue
            info = race_card_dataset_manager.get_race_info_csv(race_id)
            if info.empty:
                continue
            rt = str(info.iloc[0].get("race_type", "") or "")
            cl = str(info.iloc[0].get("course_len", "") or "")
            if rt == "ダート" and cl in TARGET_LENS:
                target.append((date_str, race_id))
    return target


def repred_race(date_str, race_id):
    """1レースを再予測してrace_card CSVを更新する。成功ならTrueを返す"""
    race_day = datetime.strptime(date_str, "%Y%m%d").date()

    df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if df.empty:
        print(f"  [スキップ] race_card なし: {race_id}")
        return False

    race_info_df = race_card_dataset_manager.get_race_info_csv(race_id)
    if race_info_df.empty:
        print(f"  [スキップ] race_info なし: {race_id}")
        return False

    if "枠" not in df.columns or "馬番" not in df.columns or "horse_id" not in df.columns:
        print(f"  [スキップ] 必須列なし: {race_id} cols={list(df.columns)}")
        return False

    df = df.reset_index(drop=True)
    horse_ids = df["horse_id"].tolist()
    waku_df = pd.DataFrame({
        "枠": df["枠"].reset_index(drop=True),
        "馬番": df["馬番"].reset_index(drop=True),
    })
    popularity_series = (
        pd.to_numeric(df["人気"], errors="coerce").reset_index(drop=True)
        if "人気" in df.columns
        else None
    )

    try:
        rank_df = race_prediction_engine.blended_rank_prediction(
            race_id, horse_ids, race_info_df, waku_df, popularity_series=popularity_series
        )
    except Exception as e:
        print(f"  [エラー] blended_rank_prediction: {race_id}: {e}")
        return False

    if rank_df.empty:
        print(f"  [スキップ] 予測結果空: {race_id}")
        return False

    rank_df = rank_df.reset_index(drop=True)
    for col in ["score", "rank", "score_hitrate", "rank_hitrate", "score_value", "rank_value"]:
        if col in rank_df.columns:
            df[col] = rank_df[col]

    race_card_dataset_manager.save_race_cards(df, race_day, race_id)
    return True


if __name__ == "__main__":
    print("=== 対象レース収集 ===")
    target_races = collect_target_races()
    print(f"対象レース数: {len(target_races)}")

    print("\n=== race_card CSV 再予測・更新 ===")
    updated_dates = set()
    ok_count, skip_count = 0, 0
    for date_str, race_id in target_races:
        rt_info = race_card_dataset_manager.get_race_info_csv(race_id)
        cl = str(rt_info.iloc[0].get("course_len", "")) if not rt_info.empty else "?"
        print(f"  {date_str} {race_id} (ダート{cl}m) ...", end=" ")
        if repred_race(date_str, race_id):
            print("OK")
            ok_count += 1
            updated_dates.add(date_str)
        else:
            skip_count += 1

    print(f"\n再予測完了: {ok_count}件更新 / {skip_count}件スキップ")

    # AI成績データセットを再構築（更新したrace_cardを反映させる）
    # update_ai_performance_dataset は未集計レースのみ追加するため、
    # 対象レースを一旦削除してから再計算する
    print("\n=== AI成績データセット再構築 ===")
    updated_race_ids = {race_id for _, race_id in target_races if _ in updated_dates}
    perf_df = ai_performance_dataset_manager.get_ai_performance_dataset()
    if not perf_df.empty:
        before = len(perf_df)
        perf_df = perf_df[~perf_df.index.astype(str).isin(updated_race_ids)]
        removed = before - len(perf_df)
        ai_performance_dataset_manager.save_ai_performance_dataset(perf_df)
        print(f"  既存エントリ削除: {removed}件")
    added = ai_performance_dataset_manager.update_ai_performance_dataset()
    print(f"  再計算・追加: {added}件")

    # 出馬表HTML再生成（更新があった日付のみ）
    print(f"\n=== 出馬表HTML再生成 ({len(updated_dates)}日分) ===")
    for date_str in sorted(updated_dates):
        race_day = datetime.strptime(date_str, "%Y%m%d").date()
        print(f"  {date_str} 再生成...")
        race_page_generator.make_daily_race_card_html(race_day)
        daily_index_generator.make_daily_index_page(race_day)

    # AI成績HTML全再生成
    print("\n=== AI成績HTML再生成 ===")
    ai_performance_report_generator.make_ai_performance_index_page()
    ai_performance_report_generator.make_all_annual_performance_pages()
    ai_performance_report_generator.make_all_course_performance_pages()
    ai_performance_report_generator.make_all_meeting_performance_pages()

    print("\n完了")
