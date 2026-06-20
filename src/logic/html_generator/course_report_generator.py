"""コース詳細データページ（Forge: HTMLFactory）のHTML生成

各開催場・コース（芝/ダート×距離）について、race_info_dataset_manager /
peds_results_dataset_manager の週次更新で既に集計済みのコース別データ
（平均勝ち時計・平均人気・勝ち馬の平均馬体重・枠番馬番傾向・平均配当・通過順・
血統別成績）をページとして表示する。集計済みCSVは race_type/course_len/
ground_state/class の組み合わせごとに行を持つため、本モジュールは「全体」行だけでなく
クラス別・馬場別・年度別の内訳もこのCSVから直接フィルタして表示する
（新しい統計ロジックは追加しない、既存データの読み取り・整形のみ）。

開催場別のコース一覧ページ（courses/{place}/index.html）はあくまで代表的な
データ（コースごとの平均勝ち時計・平均人気、芝/ダート別の血統トップ10）の表示に留め、
クラス別・馬場別・年度別・通過順・人気/枠順の分布といった詳細データは
個別コースページ（courses/{place}/{race_type}-{course_len}.html）でのみ確認できるようにする。

血統別成績は、コース（distance）単位の集計（peds_results_dataset_manager）を
芝/ダート単位（その開催場の全距離を合算）に集約して場別コース一覧ページに表示する。

走破時計はmsec単位で保存されているため、表示時は「分:秒.コンマ」形式
（例: 81655 -> "1:21.7"）に変換する。

旧ver2.0の web/site/performance/course/ 相当（ただしver2.0でも的中率・回収率以外の
コース詳細データ自体は専用ページとして仕上がっていなかった）。
"""

from datetime import date

import pandas as pd

from src.config.constants import NAME_LIST, PLACE_LIST
from src.config.lists import COURSE_LISTS
from src.logic.calculators import ai_performance_calculator as calc
from src.managers import html_manager, peds_results_dataset_manager, race_info_dataset_manager

ANNUAL_START_YEAR = 2019
GROUND_STATE_ORDER = ["良", "稍重", "重", "不良"]


def _format_time(value):
    """msec単位の走破/勝ち時計を「分:秒.コンマ」形式に変換する（例: "81655" -> "1:21.7"）"""
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return "データなし"
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return "データなし"
    minutes = int(ms // 60000)
    seconds = (ms % 60000) / 1000
    return f"{minutes}:{seconds:04.1f}"


def _fmt(row, key, suffix=""):
    if row is None or key not in row.index:
        return "データなし"
    value = row[key]
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return "データなし"
    return f"{value}{suffix}"


def _fmt_time(row, key="avg_time"):
    if row is None or key not in row.index:
        return "データなし"
    return _format_time(row[key])


def _filter_rows(df, race_type, course_len, ground_state=None, class_name=None):
    """集計CSVから race_type/course_len（+任意でground_state/class）に絞り込む"""
    if df.empty:
        return df
    cond = (df["race_type"] == race_type) & (df["course_len"].astype(str) == str(course_len))
    if ground_state is not None:
        cond &= df["ground_state"] == ground_state
    if class_name is not None:
        cond &= df["class"] == class_name
    return df[cond]


def _filter_overall_row_by(df, race_type, course_len, ground_state, class_name):
    """集計CSVから指定条件に一致する1行を取り出す（無ければNone）"""
    sub = _filter_rows(df, race_type, course_len, ground_state, class_name)
    if sub.empty:
        return None
    return sub.iloc[0]


def _filter_overall_row(df, race_type, course_len, ground_state_label="全"):
    """集計CSVから、指定race_type/course_lenの「全体(ground_state=全, class=all)」行を1件取り出す"""
    return _filter_overall_row_by(df, race_type, course_len, ground_state_label, "all")


def build_course_report(place_id, race_type, course_len):
    """1コース条件（開催場×race_type×距離）分のレポートデータを辞書で返す"""
    avg_time = _filter_overall_row(
        race_info_dataset_manager.get_total_average_time_csv(place_id), race_type, course_len
    )
    avg_pop = _filter_overall_row(
        race_info_dataset_manager.get_total_average_pops_csv(place_id), race_type, course_len
    )
    winner_weight = _filter_overall_row(
        race_info_dataset_manager.get_total_winner_weight_csv(place_id), race_type, course_len
    )
    avg_frame_and_horse = _filter_overall_row(
        race_info_dataset_manager.get_total_average_frames_csv(place_id), race_type, course_len
    )
    win_return = _filter_overall_row(
        race_info_dataset_manager.get_total_average_returns_csv(place_id), race_type, course_len
    )
    peds_df = peds_results_dataset_manager.get_total_peds_results_csv(place_id, race_type, course_len, "all")

    return {
        "place_id": place_id,
        "race_type": race_type,
        "course_len": course_len,
        "avg_time": avg_time,
        "avg_pop": avg_pop,
        "winner_weight": winner_weight,
        "avg_frame_and_horse": avg_frame_and_horse,
        "win_return": win_return,
        "peds_df": peds_df,
    }


def _breakdown_row(place_id, race_type, course_len, value_label, ground_state, class_name, year=None):
    """クラス別・馬場別・年度別の内訳1行分（平均勝ち時計・人気・体重・枠番・馬番・平均配当）を集計する"""
    if year is None:
        time_row = _filter_overall_row_by(
            race_info_dataset_manager.get_total_average_time_csv(place_id), race_type, course_len, ground_state, class_name
        )
        pop_row = _filter_overall_row_by(
            race_info_dataset_manager.get_total_average_pops_csv(place_id), race_type, course_len, ground_state, class_name
        )
        weight_row = _filter_overall_row_by(
            race_info_dataset_manager.get_total_winner_weight_csv(place_id), race_type, course_len, ground_state, class_name
        )
        frame_row = _filter_overall_row_by(
            race_info_dataset_manager.get_total_average_frames_csv(place_id), race_type, course_len, ground_state, class_name
        )
        return_row = _filter_overall_row_by(
            race_info_dataset_manager.get_total_average_returns_csv(place_id), race_type, course_len, ground_state, class_name
        )
    else:
        time_row = _filter_overall_row_by(
            race_info_dataset_manager.get_annual_average_time_csv(place_id, year),
            race_type, course_len, ground_state, class_name,
        )
        pop_row = _filter_overall_row_by(
            race_info_dataset_manager.get_annual_average_pops_csv(place_id, year),
            race_type, course_len, ground_state, class_name,
        )
        weight_row = _filter_overall_row_by(
            race_info_dataset_manager.get_annual_winner_weight_csv(place_id, year),
            race_type, course_len, ground_state, class_name,
        )
        frame_row = _filter_overall_row_by(
            race_info_dataset_manager.get_annual_average_frames_csv(place_id, year),
            race_type, course_len, ground_state, class_name,
        )
        return_row = _filter_overall_row_by(
            race_info_dataset_manager.get_annual_average_returns_csv(place_id, year),
            race_type, course_len, ground_state, class_name,
        )

    if time_row is None:
        return None

    return {
        "value": value_label,
        "avg_time": _fmt_time(time_row),
        "avg_pop": _fmt(pop_row, "avg_pop"),
        "weight": _fmt(weight_row, "馬体重", "kg"),
        "avg_frame": _fmt(frame_row, "avg_frame"),
        "avg_horse": _fmt(frame_row, "avg_horse"),
        "win_return": _fmt(return_row, "win_return", "円"),
    }


def build_class_breakdown(place_id, race_type, course_len):
    """クラス別の内訳（馬場状態は全体「全」で固定）を返す"""
    time_df = race_info_dataset_manager.get_total_average_time_csv(place_id)
    sub = _filter_rows(time_df, race_type, course_len, ground_state="全")
    classes = [c for c in sub["class"].unique() if c != "all"] if not sub.empty else []

    rows = [
        _breakdown_row(place_id, race_type, course_len, class_name, "全", class_name) for class_name in classes
    ]
    return [r for r in rows if r is not None]


def build_ground_state_breakdown(place_id, race_type, course_len):
    """馬場別の内訳（クラスは全体「all」で固定）を返す（良→稍重→重→不良の順）"""
    rows = [
        _breakdown_row(place_id, race_type, course_len, ground_state, ground_state, "all")
        for ground_state in GROUND_STATE_ORDER
    ]
    return [r for r in rows if r is not None]


def build_year_breakdown(place_id, race_type, course_len, start_year=ANNUAL_START_YEAR, current_year=None):
    """年度別の内訳（馬場状態「全」・クラス「all」で固定）を新しい年から順に返す"""
    current_year = current_year or date.today().year
    rows = [
        _breakdown_row(place_id, race_type, course_len, year, "全", "all", year=year)
        for year in range(start_year, current_year + 1)
    ]
    rows = [r for r in rows if r is not None]
    rows.reverse()
    return rows


def _passage_breakdown_row(place_id, race_type, course_len, value_label, ground_state, class_name, year=None):
    """クラス別・馬場別・年度別の通過順内訳1行分（上り・通過1〜4）を集計する"""
    if year is None:
        row = _filter_overall_row_by(
            race_info_dataset_manager.get_total_winner_time_csv(place_id), race_type, course_len, ground_state, class_name
        )
    else:
        row = _filter_overall_row_by(
            race_info_dataset_manager.get_annual_winner_time_csv(place_id, year),
            race_type, course_len, ground_state, class_name,
        )

    if row is None:
        return None

    return {
        "value": value_label,
        "agari": _fmt(row, "上り", "秒"),
        "passage1": _fmt(row, "通過1"),
        "passage2": _fmt(row, "通過2"),
        "passage3": _fmt(row, "通過3"),
        "passage4": _fmt(row, "通過4"),
    }


def build_class_passage_breakdown(place_id, race_type, course_len):
    """クラス別の通過順内訳（馬場状態は全体「全」で固定）を返す"""
    time_df = race_info_dataset_manager.get_total_average_time_csv(place_id)
    sub = _filter_rows(time_df, race_type, course_len, ground_state="全")
    classes = [c for c in sub["class"].unique() if c != "all"] if not sub.empty else []

    rows = [
        _passage_breakdown_row(place_id, race_type, course_len, class_name, "全", class_name)
        for class_name in classes
    ]
    return [r for r in rows if r is not None]


def build_ground_state_passage_breakdown(place_id, race_type, course_len):
    """馬場別の通過順内訳（クラスは全体「all」で固定）を返す（良→稍重→重→不良の順）"""
    rows = [
        _passage_breakdown_row(place_id, race_type, course_len, ground_state, ground_state, "all")
        for ground_state in GROUND_STATE_ORDER
    ]
    return [r for r in rows if r is not None]


def build_year_passage_breakdown(place_id, race_type, course_len, start_year=ANNUAL_START_YEAR, current_year=None):
    """年度別の通過順内訳（馬場状態「全」・クラス「all」で固定）を新しい年から順に返す"""
    current_year = current_year or date.today().year
    rows = [
        _passage_breakdown_row(place_id, race_type, course_len, year, "全", "all", year=year)
        for year in range(start_year, current_year + 1)
    ]
    rows = [r for r in rows if r is not None]
    rows.reverse()
    return rows


def _passage_breakdown_table_html(rows, value_label, title):
    if not rows:
        body = "<p>対象データがありません。</p>"
    else:
        trs = "".join(
            f"<tr><td>{r['value']}</td><td>{r['agari']}</td><td>{r['passage1']}</td>"
            f"<td>{r['passage2']}</td><td>{r['passage3']}</td><td>{r['passage4']}</td></tr>\n"
            for r in rows
        )
        body = f"""<table>
    <thead><tr><th>{value_label}</th><th>上り(勝ち馬)</th><th>通過1</th><th>通過2</th><th>通過3</th><th>通過4</th></tr></thead>
    <tbody>
      {trs}
    </tbody>
  </table>"""
    return f"<h3>{title}</h3>\n  {body}"


ADVANTAGE_THRESHOLD_POINTS = 8.0  # 複勝率(着内率)が平均よりこの値(%pt)以上離れていたら有利/不利と判定する


def _chakudo_rows(df, race_type, course_len, rank_column, rank_range):
    """rank_range分の着度数（1着/2着/3着/着外・着内率）の行を返す（データなしは0で埋める）"""
    sub = df[(df["race_type"] == race_type) & (df["course_len"].astype(str) == str(course_len))] if not df.empty else df
    rows_by_rank = {str(row[rank_column]): row for _, row in sub.iterrows()} if not sub.empty else {}

    result = []
    for rank in rank_range:
        row = rows_by_rank.get(str(rank))
        counts = {key: int(row[key]) for key in ["1着", "2着", "3着", "着外"]} if row is not None else {
            "1着": 0, "2着": 0, "3着": 0, "着外": 0
        }
        total = sum(counts.values())
        top3_rate = (counts["1着"] + counts["2着"] + counts["3着"]) / total * 100 if total else None
        result.append({"rank": rank, **counts, "total": total, "top3_rate": top3_rate})
    return result


def _label_advantage(rows, exclude_ranks=(), high_label="◎ 有利", low_label="▲ 不利"):
    """着内率(top3_rate)が他より明確に高い/低いランクに有利/不利のラベルを付ける

    対象（exclude_ranksを除く、出走実績のあるランク）の平均着内率からの差が
    ADVANTAGE_THRESHOLD_POINTS以上の場合のみラベルを付ける。差が小さい（フラットな）
    場合や対象が1件以下の場合は何も付けない。low_labelを空文字にすると不利側は
    付けない（人気データのように「人気が低いほど不利なのは当然」で不要な場合）。
    """
    candidates = [r for r in rows if r["total"] > 0 and r["rank"] not in exclude_ranks]
    if len(candidates) < 2:
        for r in rows:
            r["note"] = ""
        return rows

    avg = sum(r["top3_rate"] for r in candidates) / len(candidates)
    for r in rows:
        if r["total"] == 0 or r["rank"] in exclude_ranks:
            r["note"] = ""
            continue
        diff = r["top3_rate"] - avg
        if diff >= ADVANTAGE_THRESHOLD_POINTS:
            r["note"] = high_label
        elif low_label and diff <= -ADVANTAGE_THRESHOLD_POINTS:
            r["note"] = low_label
        else:
            r["note"] = ""
    return rows


def _chakudo_table_html(df, race_type, course_len, rank_column, rank_range, title, exclude_ranks=(), high_label="◎ 有利", low_label="▲ 不利"):
    """人気別・枠番別・馬番別の着度数（1着/2着/3着/着外）をテーブルHTMLにする

    rank_range（人気順位/枠番/馬番の番号）は省略せず全件表示する。着内率が他より
    明確に高い/低いランクには「傾向」列に有利/不利（または任意のラベル）を表示し、
    差が小さい（フラットな）場合は何も表示しない。
    """
    if df.empty:
        return f"<h3>{title}</h3>\n  <p>対象データがありません。</p>"

    rows = _chakudo_rows(df, race_type, course_len, rank_column, rank_range)
    rows = _label_advantage(rows, exclude_ranks=exclude_ranks, high_label=high_label, low_label=low_label)

    trs = "".join(
        f"<tr><td>{r['rank']}</td><td>{r['1着']}</td><td>{r['2着']}</td>"
        f"<td>{r['3着']}</td><td>{r['着外']}</td><td>{r['note']}</td></tr>\n"
        for r in rows
    )

    return f"""<h3>{title}</h3>
  <table>
    <thead><tr><th>順位/番号</th><th>1着</th><th>2着</th><th>3着</th><th>着外</th><th>傾向</th></tr></thead>
    <tbody>
      {trs}
    </tbody>
  </table>"""


def _breakdown_table_html(rows, value_label, title):
    if not rows:
        body = "<p>対象データがありません。</p>"
    else:
        trs = "".join(
            f"<tr><td>{r['value']}</td><td>{r['avg_time']}</td><td>{r['avg_pop']}</td>"
            f"<td>{r['weight']}</td><td>{r['avg_frame']}</td><td>{r['avg_horse']}</td><td>{r['win_return']}</td></tr>\n"
            for r in rows
        )
        body = f"""<table>
    <thead><tr><th>{value_label}</th><th>平均勝ち時計</th><th>平均人気</th><th>勝ち馬平均体重</th><th>平均枠番</th><th>平均馬番</th><th>平均配当(単勝)</th></tr></thead>
    <tbody>
      {trs}
    </tbody>
  </table>"""
    return f"<h3>{title}</h3>\n  {body}"


def _peds_table_html(peds_df, title, heading_level="h3", top_n=10):
    if peds_df is None or peds_df.empty:
        rows = "<tr><td colspan='5'>データなし</td></tr>"
    else:
        rows = "".join(
            f"<tr><td>{row['血統']}</td><td>{row['1着']}</td><td>{row['2着']}</td>"
            f"<td>{row['3着']}</td><td>{row['着外']}</td></tr>\n"
            for _, row in peds_df.head(top_n).iterrows()
        )
    return f"""<{heading_level}>{title}</{heading_level}>
  <table>
    <thead><tr><th>血統(父)</th><th>1着</th><th>2着</th><th>3着</th><th>着外</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>"""


def build_peds_class_breakdown(peds_df, top_n=5):
    """このコースの血統別成績Totalから、クラス別（"all"を除く）上位top_n件を返す

    Returns:
        list[dict]: [{"class": クラス名, "peds_df": そのクラスの上位top_n件}, ...]
            （peds_df側で既にクラス→1着数の順に整理されているため、その順序のまま）
    """
    if peds_df is None or peds_df.empty or "クラス" not in peds_df.columns:
        return []

    result = []
    for class_name in peds_df["クラス"].unique():
        if class_name == "all":
            continue
        sub = peds_df[peds_df["クラス"] == class_name].copy()
        for col in ["1着", "2着", "3着", "着外"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
        sub = sub.sort_values("1着", ascending=False).head(top_n)
        if not sub.empty:
            result.append({"class": class_name, "peds_df": sub})
    return result


def build_peds_year_breakdown(place_id, race_type, course_len, start_year=ANNUAL_START_YEAR, current_year=None, top_n=5):
    """このコースの血統別成績を年度別（"all"クラスのみ）上位top_n件で、新しい年から順に返す"""
    current_year = current_year or date.today().year

    result = []
    for year in range(start_year, current_year + 1):
        df = peds_results_dataset_manager.get_annual_peds_results_csv(place_id, year, race_type, course_len, "all")
        if df.empty or "クラス" not in df.columns:
            continue
        sub = df[df["クラス"] == "all"].copy()
        if sub.empty:
            continue
        for col in ["1着", "2着", "3着", "着外"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
        sub = sub.sort_values("1着", ascending=False).head(top_n)
        if not sub.empty:
            result.append({"year": year, "peds_df": sub})

    result.reverse()
    return result


def _peds_breakdown_html(breakdown, label_key, title):
    if not breakdown:
        return f"<h3>{title}</h3>\n  <p>対象データがありません。</p>"
    sections = "\n  ".join(
        _peds_table_html(item["peds_df"], f"{item[label_key]}（上位{len(item['peds_df'])}件）", heading_level="h4")
        for item in breakdown
    )
    return f"<h3>{title}</h3>\n  {sections}"


def aggregate_peds_by_race_type(place_id, race_type):
    """その開催場の指定race_type（芝/ダート）について、全距離を合算した血統別成績を返す

    1着数の多い順に並べる。距離別の集計済みCSV（クラス=all・馬場=all行）を
    合算するのみで、新しい統計ロジックは追加しない。
    """
    course_lens = [cl for rt, cl in COURSE_LISTS[place_id - 1] if rt == race_type]
    frames = []
    for course_len in course_lens:
        df = peds_results_dataset_manager.get_total_peds_results_csv(place_id, race_type, course_len, "all")
        if df.empty or "クラス" not in df.columns:
            continue
        df = df[df["クラス"] == "all"]
        if not df.empty:
            frames.append(df[["血統", "1着", "2着", "3着", "着外"]])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    for col in ["1着", "2着", "3着", "着外"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)

    result = combined.groupby("血統", as_index=False)[["1着", "2着", "3着", "着外"]].sum()
    return result.sort_values("1着", ascending=False).reset_index(drop=True)


def course_report_to_html(report):
    place_name = NAME_LIST[report["place_id"] - 1]
    place_id = report["place_id"]
    race_type = report["race_type"]
    course_len = report["course_len"]

    winner_weight_html = _fmt(report["winner_weight"], "馬体重", "kg")
    avg_frame_html = _fmt(report["avg_frame_and_horse"], "avg_frame")
    avg_horse_html = _fmt(report["avg_frame_and_horse"], "avg_horse")
    win_return_html = _fmt(report["win_return"], "win_return", "円")

    class_breakdown = build_class_breakdown(place_id, race_type, course_len)
    ground_state_breakdown = build_ground_state_breakdown(place_id, race_type, course_len)
    year_breakdown = build_year_breakdown(place_id, race_type, course_len)

    class_passage_breakdown = build_class_passage_breakdown(place_id, race_type, course_len)
    ground_state_passage_breakdown = build_ground_state_passage_breakdown(place_id, race_type, course_len)
    year_passage_breakdown = build_year_passage_breakdown(place_id, race_type, course_len)

    pop_chakudo_table = _chakudo_table_html(
        race_info_dataset_manager.get_total_pop_chakudo_csv(place_id),
        race_type, course_len, "人気", range(1, 19), "人気データ（人気別着度数、全体）",
        # 1〜3番人気が強いのは当然のため評価対象から除外し、それ以外で着内率が
        # 目立って高いランクのみ「ねらい目」として示す（人気が低いほど不利、は表示しない）
        exclude_ranks={1, 2, 3}, high_label="★ ねらい目", low_label="",
    )
    frame_chakudo_table = _chakudo_table_html(
        race_info_dataset_manager.get_total_frame_chakudo_csv(place_id),
        race_type, course_len, "枠番", range(1, 9), "枠順データ（枠番別着度数、全体）",
    )
    horse_chakudo_table = _chakudo_table_html(
        race_info_dataset_manager.get_total_horse_chakudo_csv(place_id),
        race_type, course_len, "馬番", range(1, 19), "枠順データ（馬番別着度数、全体）",
    )

    peds_df = report["peds_df"]
    peds_total_df = peds_df[peds_df["クラス"] == "all"] if not peds_df.empty and "クラス" in peds_df.columns else peds_df
    peds_class_breakdown = build_peds_class_breakdown(peds_df)
    peds_year_breakdown = build_peds_year_breakdown(place_id, race_type, course_len)

    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{place_name} {race_type}{course_len}m コース詳細</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
</head>
<body>
  <p><a href="index.html">&larr; {place_name}のコース一覧へ</a></p>
  <h1>{place_name} {race_type}{course_len}m コース詳細</h1>

  <div class="summary-stats">
    <div class="summary-stat">
      <span class="value">{_fmt_time(report["avg_time"])}</span>
      <span class="label">平均勝ち時計</span>
    </div>
    <div class="summary-stat">
      <span class="value">{_fmt(report["avg_pop"], "avg_pop")}</span>
      <span class="label">平均人気</span>
    </div>
    <div class="summary-stat">
      <span class="value">{winner_weight_html}</span>
      <span class="label">勝ち馬の平均馬体重</span>
    </div>
    <div class="summary-stat">
      <span class="value">{avg_frame_html} / {avg_horse_html}</span>
      <span class="label">勝ち馬の平均枠番 / 平均馬番</span>
    </div>
    <div class="summary-stat">
      <span class="value">{win_return_html}</span>
      <span class="label">平均配当（単勝）</span>
    </div>
  </div>

  <h2>クラス別・馬場別・年度別データ</h2>
  {_breakdown_table_html(class_breakdown, "クラス", "クラス別")}
  {_breakdown_table_html(ground_state_breakdown, "馬場状態", "馬場別")}
  {_breakdown_table_html(year_breakdown, "年度", "年度別")}

  <h2>通過順（クラス別・馬場別・年度別）</h2>
  {_passage_breakdown_table_html(class_passage_breakdown, "クラス", "クラス別")}
  {_passage_breakdown_table_html(ground_state_passage_breakdown, "馬場状態", "馬場別")}
  {_passage_breakdown_table_html(year_passage_breakdown, "年度", "年度別")}

  <h2>人気・枠順データ</h2>
  {pop_chakudo_table}
  {frame_chakudo_table}
  {horse_chakudo_table}

  <h2>血統別成績</h2>
  {_peds_table_html(peds_total_df, "TOTAL（上位10件）")}
  {_peds_breakdown_html(peds_class_breakdown, "class", "クラス別")}
  {_peds_breakdown_html(peds_year_breakdown, "year", "年度別")}

  <p><a href="../../performance/course/{PLACE_LIST[place_id - 1]}/{race_type}-{course_len}.html">&larr; このコースのAI成績を見る</a></p>
  <p><a href="../../index.html">&larr; HOMEへ戻る</a></p>
</body>
</html>
"""


def make_course_index_page():
    """全開催場一覧ページ（public_html/courses/index.html）を生成する

    開催中の競馬場（ai_performance_calculator.get_current_meetings）は
    大きいタイル（.course-tile.active）、それ以外は小さいリンク
    （.course-tile.inactive、アクセスは可能）で表示する。
    """
    current_place_ids = {m["place_id"] for m in calc.get_current_meetings()}
    current_meetings_by_place = {m["place_id"]: m for m in calc.get_current_meetings()}

    active_tiles = ""
    inactive_tiles = ""
    for i in range(len(PLACE_LIST)):
        place_id = i + 1
        place_name = NAME_LIST[i]
        place_key = PLACE_LIST[i]
        if place_id in current_place_ids:
            meeting = current_meetings_by_place[place_id]
            active_tiles += f"""<div class="course-tile active">
        <a href="{place_key}/index.html"><span class="place-name">{place_name}</span></a>
        <div class="meeting-info">{meeting['times']}回 開催中（{meeting['first_day'].strftime('%m/%d')}〜{meeting['last_day'].strftime('%m/%d')}）</div>
      </div>\n"""
        else:
            inactive_tiles += f'<div class="course-tile inactive"><a href="{place_key}/index.html">{place_name}</a></div>\n'

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>コース詳細データ</title>
  <link rel="stylesheet" href="../assets/css/styles.css">
</head>
<body>
  <h1>コース詳細データ</h1>

  <h2>開催中の競馬場</h2>
  <div class="course-tile-grid">
    {active_tiles if active_tiles else "<p>現在開催中の競馬場はありません。</p>"}
  </div>

  <h2>その他の競馬場</h2>
  <div class="course-tile-grid">
    {inactive_tiles}
  </div>

  <p><a href="../index.html">&larr; HOMEへ戻る</a></p>
</body>
</html>
"""
    html_manager.save_course_index_html(html)


def make_track_page(place_id):
    """場別のコース×距離一覧ページ（public_html/courses/{place}/index.html）を生成する

    単なるリンク一覧ではなく、各コースの平均勝ち時計・平均人気を一覧表示することで
    クリックせずに概要を把握できるようにする。あわせて、芝/ダート別に全距離を
    合算した血統別成績も表示する。
    """
    place_name = NAME_LIST[place_id - 1]
    course_list = COURSE_LISTS[place_id - 1]

    time_df = race_info_dataset_manager.get_total_average_time_csv(place_id)
    pop_df = race_info_dataset_manager.get_total_average_pops_csv(place_id)

    rows = ""
    for race_type, course_len in course_list:
        time_row = _filter_overall_row(time_df, race_type, course_len)
        pop_row = _filter_overall_row(pop_df, race_type, course_len)
        rows += (
            f'<tr><td><a href="{race_type}-{course_len}.html">{race_type}{course_len}m</a></td>'
            f"<td>{_fmt_time(time_row)}</td><td>{_fmt(pop_row, 'avg_pop')}</td></tr>\n"
        )

    peds_sections = "\n  ".join(
        _peds_table_html(
            aggregate_peds_by_race_type(place_id, race_type), f"{race_type}（全距離合算・上位10件）"
        )
        for race_type in sorted({race_type for race_type, _ in course_list})
    )

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{place_name} コース一覧</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
</head>
<body>
  <h1>{place_name} コース一覧</h1>

  <table>
    <thead><tr><th>コース</th><th>平均勝ち時計</th><th>平均人気</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <h2>血統別成績（芝/ダート別）</h2>
  {peds_sections}

  <p><a href="../../performance/course/{PLACE_LIST[place_id - 1]}/index.html">&larr; このコースのAI成績を見る</a></p>
  <p><a href="../index.html">&larr; コース詳細データ一覧へ</a></p>
</body>
</html>
"""
    html_manager.save_track_index_html(place_id, html)


def make_course_detail_page(place_id, race_type, course_len):
    """個別コース詳細ページ（public_html/courses/{place}/{race_type}-{course_len}.html）を生成する"""
    report = build_course_report(place_id, race_type, course_len)
    html_manager.save_course_detail_html(place_id, race_type, course_len, course_report_to_html(report))


def make_all_course_pages():
    """全開催場・全コースのコース詳細データページを一括生成する"""
    make_course_index_page()
    for place_id in range(1, len(PLACE_LIST) + 1):
        make_track_page(place_id)
        for race_type, course_len in COURSE_LISTS[place_id - 1]:
            make_course_detail_page(place_id, race_type, course_len)
