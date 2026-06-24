"""AI予想成績データセットの永続化（Manager層）

src.logic.calculators.ai_performance_calculator が計算する1レース単位の
的中・回収結果（calc_race_hit_returns）を、data/ai_performance/ai_performance.csv
に永続化する。ページ生成のたびに data/race_card/ を全件スキャンして都度計算する
（旧 ai_performance_report_generator / home_generator の挙動）のは遅いため、
update_ai_performance_dataset で事前に計算・保存し、get_ai_performance_dataset で
取得したデータセットを本モジュールの aggregate / filter_by_* / group_breakdown で
高速に（ファイルI/Oなしで）集計できるようにする。

予想または確定配当のいずれかが無いレース（calc_race_hit_returns が None を返す
レース）はデータセットに含めない。配当が後から確定したら次回の更新で追加される。

race_type/course_len/ground_state/class は出馬表側のrace_info
（race_card_dataset_manager.get_race_info_csv）ではなく確定結果側
（race_result_dataset_manager.get_race_results_csv）から取得する。
予想済み4742件のうちサンプル確認した限り、race_infoのカバー率は1.8%しかないが、
確定結果のカバー率は82.6%あるため。取得できないレースはこれらの列を空にし、
win/place/trio_boxの集計には含めるが、クラス別・馬場別等の内訳（group_breakdown）
には出てこない。
"""

import os

import pandas as pd

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.calculators import ai_performance_calculator
from src.logic.calculators.ai_performance_calculator import BET_TYPES
from src.managers import race_result_dataset_manager
from src.utils.file_utils import read_csv_or_empty

AI_PERFORMANCE_DATASET_PATH = os.path.join(paths.AI_PERFORMANCE_DATA_PATH, "ai_performance.csv")

COLUMNS = [
    "race_day",
    "year",
    "place_id",
    "times",
    "race_type",
    "course_len",
    "ground_state",
    "class",
    "win_hit",
    "win_return",
    "place_hit",
    "place_return",
    "trio_box_hit",
    "trio_box_return",
]


def get_ai_performance_dataset():
    """data/ai_performance/ai_performance.csv を取得する（race_idがindex）"""
    df = read_csv_or_empty(AI_PERFORMANCE_DATASET_PATH, dtype=str, index_col=0)
    if not df.empty:
        df.index = df.index.astype(str)
        # race_type等が空のレースはCSV往復でNaNになるため、空文字列に統一する
        df = df.fillna("")
    return df


def save_ai_performance_dataset(df):
    """AI予想成績データセットをCSVに保存する"""
    os.makedirs(paths.AI_PERFORMANCE_DATA_PATH, exist_ok=True)
    df.to_csv(AI_PERFORMANCE_DATASET_PATH)


def _build_race_conditions(race_ids_by_place_year):
    """(place_id, year)ごとにrace_results_csvを1回だけ読み込み、
    race_id -> {"race_type","course_len","ground_state","class"} の辞書を作る

    race_result_dataset_manager.get_race_id_result(race_id) を予想済み全件に
    そのまま呼ぶと、同じ年間結果CSVを何百回も読み直して遅いため、
    (place_id, year)単位でまとめて1回ずつ読み込む。
    """
    conditions = {}
    for (place_id, year), race_ids in race_ids_by_place_year.items():
        if not (1 <= place_id <= len(PLACE_LIST)):
            continue
        df = race_result_dataset_manager.get_race_results_csv(place_id, year)
        if df.empty:
            continue
        df.index = df.index.astype(str)
        matched = df[df.index.isin(set(race_ids))]
        for race_id in matched.index.unique():
            row = matched.loc[[race_id]].iloc[0]
            conditions[race_id] = {
                "race_type": row.get("race_type", ""),
                "course_len": row.get("course_len", ""),
                "ground_state": row.get("ground_state", ""),
                "class": row.get("class", ""),
            }
    return conditions


def _row_from_result(race_day, parsed, condition, result):
    win_hit, win_return = result["win"]
    place_hit, place_return = result["place"]
    trio_box_hit, trio_box_return = result["trio_box"]
    return {
        "race_day": race_day,
        "year": parsed["year"],
        "place_id": parsed["place_id"],
        "times": parsed["times"],
        "race_type": condition.get("race_type", ""),
        "course_len": condition.get("course_len", ""),
        "ground_state": condition.get("ground_state", ""),
        "class": condition.get("class", ""),
        "win_hit": win_hit,
        "win_return": win_return,
        "place_hit": place_hit,
        "place_return": place_return,
        "trio_box_hit": trio_box_hit,
        "trio_box_return": trio_box_return,
    }


def update_ai_performance_dataset():
    """予想済みレースのうち未集計のものについて的中・回収・分類を計算し、データセットに追加する

    Returns:
        int: 新たに追加した行数
    """
    existing_df = get_ai_performance_dataset()
    existing_race_ids = set(existing_df.index.astype(str)) if not existing_df.empty else set()

    new_pairs = [
        (race_day, str(race_id))
        for race_day, race_id in ai_performance_calculator.list_predicted_races()
        if str(race_id) not in existing_race_ids
    ]
    if not new_pairs:
        return 0

    race_ids_by_place_year = {}
    for _, race_id in new_pairs:
        parsed = ai_performance_calculator.parse_race_id(race_id)
        key = (parsed["place_id"], parsed["year"])
        race_ids_by_place_year.setdefault(key, []).append(race_id)
    conditions = _build_race_conditions(race_ids_by_place_year)

    new_rows = {}
    for race_day, race_id in new_pairs:
        result = ai_performance_calculator.calc_race_hit_returns(race_day, race_id)
        if result is None:
            continue
        parsed = ai_performance_calculator.parse_race_id(race_id)
        condition = conditions.get(race_id, {})
        new_rows[race_id] = _row_from_result(race_day, parsed, condition, result)

    if not new_rows:
        return 0

    new_df = pd.DataFrame.from_dict(new_rows, orient="index", columns=COLUMNS)
    combined_df = pd.concat([existing_df, new_df]) if not existing_df.empty else new_df
    save_ai_performance_dataset(combined_df)
    return len(new_rows)


# --- 高速集計・フィルタ（永続化済みデータセットに対してpandasのみで処理する） -------------


def aggregate(df):
    """per-raceの行（データセットのDataFrame、または絞り込んだ部分集合）から
    的中率・回収率・件数を集計する

    Returns:
        dict: {"win": {"hit_rate": float, "return_rate": float, "n": int}, "place": {...}, "trio_box": {...}}
    """
    if df.empty:
        return {bet_type: {"hit_rate": 0.0, "return_rate": 0.0, "n": 0} for bet_type in BET_TYPES}

    n = len(df)
    return {
        bet_type: {
            "hit_rate": df[f"{bet_type}_hit"].astype(float).mean() * 100,
            "return_rate": df[f"{bet_type}_return"].astype(float).mean(),
            "n": n,
        }
        for bet_type in BET_TYPES
    }


def filter_by_place(df, place_id):
    """指定した開催場のみに絞り込む"""
    if df.empty:
        return df
    return df[df["place_id"].astype(str) == str(place_id)]


def filter_by_year(df, year):
    """指定した年のみに絞り込む"""
    if df.empty:
        return df
    return df[df["year"].astype(str) == str(year)]


def filter_by_course(df, place_id, race_type, course_len):
    """指定した開催場・race_type・距離のみに絞り込む"""
    if df.empty:
        return df
    return df[
        (df["place_id"].astype(str) == str(place_id))
        & (df["race_type"] == race_type)
        & (df["course_len"].astype(str) == str(course_len))
    ]


def filter_by_meeting(df, year, place_id, times):
    """指定した年・開催場・開催回のみに絞り込む"""
    if df.empty:
        return df
    return df[
        (df["year"].astype(str) == str(year))
        & (df["place_id"].astype(str) == str(place_id))
        & (df["times"].astype(str) == str(times))
    ]


def filter_by_date_range(df, start_day, end_day):
    """race_dayが指定範囲（両端含む）に収まる行のみに絞り込む

    race_dayはISO形式（YYYY-MM-DD）の文字列のため、辞書式比較で日付範囲を判定できる。
    """
    if df.empty:
        return df
    start_str, end_str = str(start_day), str(end_day)
    return df[(df["race_day"] >= start_str) & (df["race_day"] <= end_str)]


def group_breakdown(df, column):
    """指定列（class/race_type/ground_state/year等）でグループ化し、
    各グループの集計結果を返す

    値が空・欠損の行（race_resultが見つからずrace_type等が未設定のレース）は
    内訳から除外する。

    Returns:
        list[dict]: [{"value": ..., "performance": aggregate(...)}, ...]
    """
    if df.empty or column not in df.columns:
        return []

    result = []
    for value, group_df in df.groupby(column):
        if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
            continue
        result.append({"value": value, "performance": aggregate(group_df)})
    return result


def cross_breakdown(df, column_a, column_b):
    """2列（例: ground_state, class）の組み合わせごとに集計する

    group_breakdownと同様、値が空・欠損の行は内訳から除外する。

    Returns:
        dict[tuple, dict]: {(column_aの値, column_bの値): aggregate(...), ...}
    """
    if df.empty or column_a not in df.columns or column_b not in df.columns:
        return {}

    result = {}
    for (value_a, value_b), group_df in df.groupby([column_a, column_b]):
        if value_a in (None, "") or value_b in (None, ""):
            continue
        result[(value_a, value_b)] = aggregate(group_df)
    return result


def group_breakdown_by_week(df):
    """race_dayから算出した「週の開始日（土曜）」でグループ化し、週ごとの集計を返す

    開催（重賞・通常開催とも）は基本的に土日に行われるため、土曜始まりの週で
    グループ化すると同じ開催（土日2日分）が1グループにまとまる。
    年間ページ・HOME等で「開催週ごとの傾向・推移」を見せる用途に使う。

    Returns:
        list[dict]: [{"value": 週開始日（YYYY-MM-DD、土曜）, "performance": aggregate(...)}, ...]
            週開始日の昇順（古い→新しい）で返す。
    """
    if df.empty:
        return []

    race_day = pd.to_datetime(df["race_day"])
    # weekday(): 月=0,...,土=5,日=6。直前（当日含む）の土曜までの日数 = (weekday - 5) % 7
    days_since_saturday = (race_day.dt.weekday - 5) % 7
    week_start = race_day - pd.to_timedelta(days_since_saturday, unit="D")
    result = [
        {"value": value, "performance": aggregate(group_df)}
        for value, group_df in df.groupby(week_start.dt.strftime("%Y-%m-%d"))
    ]
    result.sort(key=lambda item: item["value"])
    return result
