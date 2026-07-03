"""夜間バッチ: 最適特徴量バージョンを選択して全コース学習

処理フロー:
  1. Phase 1/2 実験結果ファイルを読んで v2 vs v3 の AUC を比較
  2. より良い方（または改善幅が閾値未満なら v2）を採用
  3. 全 10 競馬場 × 全コース × 2020-2025 年のデータセットを生成（未生成のみ）
  4. 全コースのモデルを学習・保存（サフィックス _v2 or _v3）

実行:
    python scripts/overnight_train_all_courses.py [v2|v3|auto]
      auto (デフォルト) : 実験結果ファイルから自動判定
      v2 / v3           : バージョンを強制指定
"""

import os
import re
import sys
import time
import traceback
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

import name_header
from src.PredictionModels.LightGBM.prediction import (
    _band_lengths, data_group, split_dataframe,
    tune_hyperparameters, _fit_ranker_final, save_lightGBM_model,
)
from src.PredictionModels.LightGBM.make_dataset_v2 import (
    load_dataset_v2, make_dataset_for_train_v2,
)
from src.PredictionModels.LightGBM.make_dataset_v3 import (
    load_dataset_v3, make_dataset_for_train_v3,
)

TRAIN_YEARS = list(range(2020, date.today().year + 1))
N_TRIALS = 20

# Phase2実験の結果ファイル（experiment_v3_chukyo.py が書くこのパス）
# スクリプト自体は stdout に出力するのみなので、タスク出力ファイルから読む
V3_EXPERIMENT_OUTPUT = None  # None の場合は引数またはデフォルトで判断


# ---------- バージョン判定 ----------

def _parse_auc_summary(text):
    """
    実験出力テキストからサマリ行を解析して {コース: (v2_auc, v3_auc)} を返す。
    サマリ行フォーマット: "ダート1400m    0.6254   0.6309  +0.0055 ↑"
    """
    result = {}
    for line in text.splitlines():
        m = re.match(
            r"(\S+)\s+([\d.]+)\s+([\d.]+)\s+([+\-][\d.]+)\s+([↑↓→])",
            line.strip()
        )
        if m:
            course, v2, v3, diff, sym = m.groups()
            result[course] = (float(v2), float(v3))
    return result


def choose_version(v3_output_path=None, force=None):
    """
    実験結果を見て使用するバージョン ('v2' or 'v3') を決定する。
    v3 が v2 より平均 0.002 以上改善している場合のみ v3 を採用。
    """
    if force in ("v2", "v3"):
        print(f"[バージョン選択] 強制指定: {force}")
        return force

    if v3_output_path and os.path.isfile(v3_output_path):
        text = open(v3_output_path, encoding="utf-8", errors="replace").read()
        summary = _parse_auc_summary(text)
        if summary:
            diffs = [v3 - v2 for v2, v3 in summary.values()]
            avg_diff = sum(diffs) / len(diffs)
            print(f"[バージョン選択] v3 平均 AUC 差 = {avg_diff:+.4f}")
            if avg_diff >= 0.002:
                print(f"  → v3 を採用（{avg_diff:+.4f} 改善）")
                return "v3"
            else:
                print(f"  → 改善幅が小さいため v2 を採用")
                return "v2"

    # デフォルト: v2（Phase1は明確に改善済み）
    print("[バージョン選択] 実験結果ファイルなし → v2 を採用")
    return "v2"


# ---------- データセット生成 ----------

def generate_missing_datasets(version, years=TRAIN_YEARS):
    """全競馬場・全コース・全年のデータセットを未生成のみ生成する。"""
    load_fn = load_dataset_v2 if version == "v2" else load_dataset_v3
    gen_fn  = make_dataset_for_train_v2 if version == "v2" else make_dataset_for_train_v3

    total_venues = len(name_header.PLACE_LIST)
    for place_id in range(1, total_venues + 1):
        place_name = name_header.PLACE_LIST[place_id - 1]
        courses = name_header.COURSE_LISTS[place_id - 1]

        for year in years:
            missing = [
                (rt, ln) for rt, ln in courses
                if load_fn(place_id, year, rt, ln)[0].empty
            ]
            if not missing:
                continue

            print(f"\n[データ生成] {place_name} {year}年 ({len(missing)}コース欠損)")
            try:
                gen_fn(place_id, year, course_filter=missing if version == "v2" else missing)
            except Exception:
                print(f"  ERROR: {place_name} {year}")
                traceback.print_exc()


# ---------- モデル学習 ----------

def train_all_courses(version, years=TRAIN_YEARS):
    """全競馬場・全コースをまとめて学習して保存する。"""
    load_fn = load_dataset_v2 if version == "v2" else load_dataset_v3
    model_suffix = f"_{version}"

    total_venues = len(name_header.PLACE_LIST)
    for place_id in range(1, total_venues + 1):
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*50}")
        print(f"[学習] {place_name} (place_id={place_id})")
        print(f"{'='*50}")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            print(f"\n  {race_type}{length}m ...")
            try:
                race_data = pd.DataFrame()
                race_flag = pd.DataFrame()

                # 距離帯で近いコースも合算
                band_lengths = _band_lengths(place_id, race_type, length)
                for year in years:
                    for band_length in band_lengths:
                        df, flag = load_fn(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty or race_flag.empty:
                    print(f"    データなし: スキップ")
                    continue

                race_data = race_data.fillna(-1).reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                data_train, data_test, flag_train, flag_test = split_dataframe(race_data, race_flag)
                if data_test.empty:
                    print(f"    テストデータなし: スキップ")
                    continue

                data_train, train_group = data_group(data_train)
                data_test,  test_group  = data_group(data_test)

                print(f"    訓練 {len(data_train)}行 / テスト {len(data_test)}行")
                best_params = tune_hyperparameters(
                    data_train, flag_train, train_group,
                    data_test, flag_test, test_group,
                    n_trials=N_TRIALS,
                )
                model = _fit_ranker_final(best_params, data_train, flag_train, train_group)
                save_lightGBM_model(model, place_id, race_type, length, model_suffix=model_suffix)
                print(f"    保存完了: {race_type}{length}_lambdarank_model{model_suffix}.txt")

            except Exception:
                print(f"    ERROR: {place_name} {race_type}{length}")
                traceback.print_exc()


# ---------- メイン ----------

def _wait_for_experiment(output_path, timeout_sec=7200):
    """
    v3 実験結果ファイルが '===サマリ===' または 'サマリ' を含むまで待機する。
    timeout_sec 秒経過したら諦めて None を返す。
    """
    if not output_path or not os.path.isfile(output_path):
        return None

    print(f"[待機] 実験結果を待っています: {output_path}")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        text = open(output_path, encoding="utf-8", errors="replace").read()
        if "サマリ" in text or "======" in text:
            print("[待機] 実験完了を検知しました")
            return text
        remaining = int(deadline - time.time())
        print(f"  まだ実行中... (残り最大 {remaining//60}分)")
        time.sleep(60)

    print("[待機] タイムアウト: 実験完了を確認できず")
    return None


if __name__ == "__main__":
    force_version = sys.argv[1] if len(sys.argv) > 1 else "auto"

    # v3 実験出力ファイルを引数またはデフォルトで決定
    v3_out = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print("夜間バッチ: 全コース学習（最適特徴量バージョン選択）")
    print("=" * 60)
    print(f"対象年度: {TRAIN_YEARS[0]} ～ {TRAIN_YEARS[-1]}")
    print(f"Optuna 試行数: {N_TRIALS}")

    # バージョン決定
    if force_version == "auto":
        # 実験が実行中の場合は最大2時間待つ
        if v3_out:
            _wait_for_experiment(v3_out)
        version = choose_version(v3_out, force=None)
    else:
        version = choose_version(force=force_version)

    print(f"\n採用バージョン: {version}")
    print(f"モデルサフィックス: _{version}")

    # Step 1: データセット生成
    print("\n[Step 1] データセット生成（未生成のみ）")
    generate_missing_datasets(version)

    # Step 2: 全コース学習
    print("\n[Step 2] 全コース学習")
    train_all_courses(version)

    print("\n" + "=" * 60)
    print("夜間バッチ完了")
    print("=" * 60)
    print(f"モデルは _{version} サフィックスで保存されました。")
    print("本番予測エンジンへの切り替えは別途行ってください。")
