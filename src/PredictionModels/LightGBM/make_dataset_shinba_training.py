"""新馬戦 + 調教データ特徴量の生成

make_dataset_shinba.py の SHINBA_FEATURES に追加する調教特徴量 (8列):
    train_has_data       : 調教データあり=1 / なし=0
    train_last_lap       : 最終追い切りの最終200mタイム
    train_last_400m      : 最終追い切りの400mタイム (区間数<2 の場合はNaN)
    train_n_poles        : 最終追い切りの区間数 (走行距離の目安)
    train_grade_code     : 最終追い切りのグレード A=3/B=2/C=1/不明=0
    train_leg_code       : 最終追い切りの強度 馬也=0/強め=1/一杯=2
    train_sessions_count : そのレースに向けた追い切り本数
    train_best_grade     : 全セッション中の最高グレード (A=3/B=2/C=1/不明=0)
"""

import ast
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import paths

TRAINING_FEATURES = [
    "train_has_data",
    "train_last_lap",
    "train_last_400m",
    "train_n_poles",
    "train_grade_code",
    "train_leg_code",
    "train_sessions_count",
    "train_best_grade",
]

_GRADE_MAP = {"A": 3.0, "B": 2.0, "C": 1.0}
_LEG_MAP   = {
    "馬也": 0.0, "馬なり": 0.0,
    "強め": 1.0, "仕掛": 1.0, "末強め": 1.0,
    "一杯": 2.0, "追い切り": 2.0,
}


def _parse_times(times_str) -> list[float]:
    """'[56.3, 41.3, 26.8, 13.3]' → [56.3, 41.3, 26.8, 13.3]"""
    if not times_str or pd.isna(times_str):
        return []
    try:
        val = ast.literal_eval(str(times_str))
        return [float(v) for v in val if v is not None]
    except Exception:
        return []


def _grade_code(g) -> float:
    return _GRADE_MAP.get(str(g).strip().upper(), 0.0)


def _leg_code(leg) -> float:
    s = str(leg).strip()
    for key, val in _LEG_MAP.items():
        if key in s:
            return val
    return 0.0


def extract_training_features(horse_id: str, race_id: str) -> dict:
    """horse_id + race_id に対する調教特徴量を返す。

    data/training/{horse_id}.csv を読み込み、race_id に対応する
    セッション群から特徴量を抽出する。
    データが存在しない場合は全 NaN (train_has_data=0) を返す。
    """
    default = {col: np.nan for col in TRAINING_FEATURES}
    default["train_has_data"] = 0.0

    csv_path = os.path.join(paths.TRAINING_DATA_PATH, f"{horse_id}.csv")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return default

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return default

    if df.empty or "race_id" not in df.columns:
        return default

    # 当該レースに向けたセッションを抽出
    df["race_id"] = df["race_id"].astype(str).str.strip()
    sessions = df[df["race_id"] == str(race_id)].copy()

    if sessions.empty:
        return default

    # training_date でソートして最終追い切りを取得
    sessions["training_date"] = pd.to_datetime(sessions["training_date"], errors="coerce")
    sessions = sessions.dropna(subset=["training_date"]).sort_values("training_date")

    if sessions.empty:
        return default

    last = sessions.iloc[-1]

    times = _parse_times(last.get("times_list", ""))
    last_lap  = float(times[-1]) if len(times) >= 1 else np.nan
    last_400m = float(times[-2]) if len(times) >= 2 else np.nan
    n_poles   = float(len(times))

    grade_code = _grade_code(last.get("grade", ""))
    leg_code   = _leg_code(last.get("leg_type", ""))

    sessions_count = float(len(sessions))

    best_grade = 0.0
    for g in sessions["grade"].fillna("").astype(str):
        gc = _grade_code(g)
        if gc > best_grade:
            best_grade = gc

    return {
        "train_has_data":       1.0,
        "train_last_lap":       last_lap,
        "train_last_400m":      last_400m,
        "train_n_poles":        n_poles,
        "train_grade_code":     grade_code,
        "train_leg_code":       leg_code,
        "train_sessions_count": sessions_count,
        "train_best_grade":     best_grade,
    }


def add_training_features(shinba_df: pd.DataFrame) -> pd.DataFrame:
    """shinba DataFrame に TRAINING_FEATURES 列を追加して返す。

    shinba_df には 'horse_id' と 'race_id' 列が必要。
    行数が多い場合は tqdm で進捗表示する。
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(shinba_df.iterrows(), total=len(shinba_df), desc="調教特徴量抽出")
    except ImportError:
        iterator = shinba_df.iterrows()

    records = []
    for _, row in iterator:
        feats = extract_training_features(str(row["horse_id"]), str(row["race_id"]))
        records.append(feats)

    feat_df = pd.DataFrame(records, index=shinba_df.index)
    return pd.concat([shinba_df, feat_df], axis=1)
