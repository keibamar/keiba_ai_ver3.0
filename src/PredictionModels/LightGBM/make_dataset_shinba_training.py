"""新馬戦 + 調教データ特徴量の生成

make_dataset_shinba.py の SHINBA_FEATURES に追加する調教特徴量 (11列):
    --- 絶対値系 (8列) ---
    train_has_data       : 調教データあり=1 / なし=0
    train_last_lap       : 最終追い切りの最終200mタイム
    train_last_400m      : 最終追い切りの400mタイム (区間数<2 の場合はNaN)
    train_n_poles        : 最終追い切りの区間数 (走行距離の目安)
    train_grade_code     : 最終追い切りのグレード A=3/B=2/C=1/不明=0
    train_leg_code       : 最終追い切りの強度 馬也=0/強め=1/一杯=2
    train_sessions_count : そのレースに向けた追い切り本数
    train_best_grade     : 全セッション中の最高グレード (A=3/B=2/C=1/不明=0)

    --- コース相対値系 (3列) ---
    train_last_lap_z     : 最終200mタイムのコース別zスコア (速いほど負=有利)
    train_last_400m_z    : 400mタイムのコース別zスコア
    train_course_cat     : コース種別 0=坂路 / 1=W系 / 2=その他
"""

import ast
import json
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
    # コース相対値系
    "train_last_lap_z",
    "train_last_400m_z",
    "train_course_cat",
]

_GRADE_MAP = {"A": 3.0, "B": 2.0, "C": 1.0}
_LEG_MAP   = {
    "馬也": 0.0, "馬なり": 0.0,
    "強め": 1.0, "仕掛": 1.0, "末強め": 1.0,
    "一杯": 2.0, "追い切り": 2.0,
}

# コース種別マッピング
_COURSE_CAT_MAP = {}  # 動的に構築 (下の _course_cat 関数)


def _course_cat(course_str: str) -> float:
    """コース文字列 → 0=坂路 / 1=W系 / 2=その他"""
    s = str(course_str).strip()
    if "坂" in s:
        return 0.0
    if "Ｗ" in s or "W" in s or "ＣＷ" in s or "CW" in s or "北Ｃ" in s or "北C" in s:
        return 1.0
    return 2.0


# コース別タイム統計 (遅延ロード)
_COURSE_STATS: dict | None = None
_STATS_PATH = os.path.join(paths.TRAINING_DATA_PATH, "course_stats.json")


def _get_course_stats() -> dict:
    global _COURSE_STATS
    if _COURSE_STATS is None:
        if os.path.exists(_STATS_PATH):
            with open(_STATS_PATH, encoding="utf-8") as f:
                _COURSE_STATS = json.load(f)
        else:
            _COURSE_STATS = {}
    return _COURSE_STATS


def _zscore(value: float, mean: float | None, std: float | None) -> float:
    """z スコアを計算。std が 0 または None の場合は NaN。"""
    if mean is None or std is None or std <= 0 or np.isnan(value):
        return np.nan
    return (value - mean) / std


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

    df["race_id"] = df["race_id"].astype(str).str.strip()
    sessions = df[df["race_id"] == str(race_id)].copy()

    if sessions.empty:
        return default

    sessions["training_date"] = pd.to_datetime(sessions["training_date"], errors="coerce")
    sessions = sessions.dropna(subset=["training_date"]).sort_values("training_date")

    if sessions.empty:
        return default

    last = sessions.iloc[-1]
    course_str = str(last.get("course", "")).strip()

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

    # コース相対値系
    stats = _get_course_stats()
    st = stats.get(course_str, {})
    ll_z  = _zscore(last_lap,  st.get("ll_mean"), st.get("ll_std"))
    l4_z  = _zscore(last_400m, st.get("l4_mean"), st.get("l4_std"))
    course_cat = _course_cat(course_str)

    return {
        "train_has_data":       1.0,
        "train_last_lap":       last_lap,
        "train_last_400m":      last_400m,
        "train_n_poles":        n_poles,
        "train_grade_code":     grade_code,
        "train_leg_code":       leg_code,
        "train_sessions_count": sessions_count,
        "train_best_grade":     best_grade,
        # コース相対値系
        "train_last_lap_z":     ll_z,
        "train_last_400m_z":    l4_z,
        "train_course_cat":     course_cat,
    }


def add_training_features(shinba_df: pd.DataFrame) -> pd.DataFrame:
    """shinba DataFrame に TRAINING_FEATURES 列を追加して返す。

    shinba_df には 'horse_id' と 'race_id' 列が必要。
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
