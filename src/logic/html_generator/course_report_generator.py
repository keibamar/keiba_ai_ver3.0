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
from src.logic.calculators import average_calculator
from src.logic.html_generator.race_type_badge_html import course_label_html, race_type_span_html
from src.logic.html_generator.site_nav_html import (
    AD_SLOT_IN_CONTENT_1,
    SITE_URL,
    ad_unit_html,
    adsense_script_html,
    breadcrumb_html,
    ga4_script_html,
    meta_tags_html,
    site_footer_html,
    site_nav_html,
)
from src.logic.html_generator.sparkline_html import single_line_trend_svg
from src.managers import (
    html_manager,
    peds_results_dataset_manager,
    race_info_dataset_manager,
    race_result_dataset_manager,
)

ANNUAL_START_YEAR = 2019
GROUND_STATE_ORDER = ["良", "稍重", "重", "不良"]

# 馬場×クラス×年度パネル（cross-filter）の「開始年」は、本来は自由に選べる方が
# 柔軟だが、開始年ごとに全パネル（人気・枠番・馬番・馬体重・血統の各チャートを含む）
# を事前生成する構成のため、選択肢を増やすほど生成HTMLが線形に増える
# （開始年を2019〜最新年の全パターン用意していたところ、コースページ1件が
# 最大13MBに達していた）。「全期間・直近3年・今年」の3パターンに絞り、
# ページサイズを抑える。


def _cross_filter_start_years(oldest_year, current_year):
    """馬場×クラス×年度パネルの開始年を「全期間・直近3年・今年」の3パターンに絞って返す"""
    candidates = [oldest_year, max(oldest_year, current_year - 2), current_year]
    return sorted(set(candidates))


def _cross_filter_year_label(start_year, oldest_year, current_year):
    """開始年の表示ラベルを返す（全期間/直近N年/今年）"""
    if start_year <= oldest_year:
        return "全期間"
    if start_year == current_year:
        return f"今年（{current_year}年〜）"
    return f"直近{current_year - start_year + 1}年（{start_year}年〜）"

# race_info_dataset_manager.update_chakudo（馬体重帯別着度数）と同じ刻み・範囲
# （src/datasets/race_info/transform.pyのWEIGHT_BUCKET_SIZE/WEIGHT_BUCKET_LOW/HIGH/RANGEと揃える）
WEIGHT_BUCKET_SIZE = 10
WEIGHT_BUCKET_LOW = 390
WEIGHT_BUCKET_HIGH = 550
WEIGHT_BUCKET_RANGE = range(WEIGHT_BUCKET_LOW - WEIGHT_BUCKET_SIZE, WEIGHT_BUCKET_HIGH + 1, WEIGHT_BUCKET_SIZE)


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


def _raw_float(row, key):
    """rowからkeyの値を生のfloatとして取り出す（欠損・変換不可ならNone）"""
    if row is None or key not in row.index:
        return None
    value = row[key]
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    avg_time_fmt = _fmt_time(time_row)
    if avg_time_fmt == "データなし":
        # 年度別CSVは存在するすべての馬場×クラスの組み合わせ分の行を持つため、
        # 行自体はマッチしても実際の出走が無い（平均値が欠損の）組み合わせがある。
        # 3軸（馬場×クラス×年度）まで掛け合わせるとこのケースが大幅に増えるため、
        # 行が見つかったかどうかだけでなく、実際にデータがあるかどうかも確認して除外する。
        return None

    return {
        "value": value_label,
        "avg_time": avg_time_fmt,
        "avg_pop": _fmt(pop_row, "avg_pop"),
        "weight": _fmt(weight_row, "馬体重", "kg"),
        "avg_frame": _fmt(frame_row, "avg_frame"),
        "avg_horse": _fmt(frame_row, "avg_horse"),
        "win_return": _fmt(return_row, "win_return", "円"),
        "win_return_raw": _raw_float(return_row, "win_return"),
    }


def build_venue_comparison_breakdown(place_id, race_type, course_len, ground_state="全", class_name="all"):
    """同じrace_type×course_lenを持つ他競馬場との比較を内訳行として返す

    平均勝ち時計・人気・体重・枠番・配当は競馬場ごとに傾向が異なるため、同条件
    （同じ芝/ダート・同じ距離）の競馬場を横に並べることで「このコースらしさ」を
    相対的に把握できるようにする。対象は実際にその距離のコースを持つ競馬場のみ
    （無理に全10場を並べない）。現在表示中の競馬場は強調表示、他競馬場はその
    競馬場自身の同条件ページへのリンクにする（_breakdown_rowのvalue_labelに
    そのままHTMLを渡せるため、表自体は既存の_breakdown_table_htmlを再利用できる）。
    ground_state/class_nameを指定すると、その条件のみに絞り込んだ比較になる
    （_venue_comparison_cross_filter_htmlの組み合わせ絞り込みUIから使う）。
    """
    rows = []
    for pid in range(1, len(PLACE_LIST) + 1):
        if [race_type, str(course_len)] not in COURSE_LISTS[pid - 1]:
            continue
        place_name = NAME_LIST[pid - 1]
        if pid == place_id:
            value_label = f'<strong class="venue-compare-current">{place_name}（現在）</strong>'
        else:
            value_label = f'<a href="../{PLACE_LIST[pid - 1]}/{race_type}-{course_len}.html">{place_name}</a>'
        row = _breakdown_row(pid, race_type, course_len, value_label, ground_state, class_name)
        if row is not None:
            rows.append(row)
    return rows


def _venue_comparison_panel_html(place_id, race_type, course_len, ground_state, class_name):
    """他競馬場比較の馬場状態×クラス1組み合わせ分のパネルを返す"""
    rows = build_venue_comparison_breakdown(place_id, race_type, course_len, ground_state, class_name)
    table_html = _breakdown_table_html(rows, "競馬場", title="")
    return f"""<div class="cross-filter-panel" data-ground-state="{ground_state}" data-class="{class_name}" hidden>
    {table_html}
  </div>"""


def _venue_comparison_cross_filter_html(place_id, race_type, course_len, ground_states, classes):
    """他競馬場比較を、馬場状態×クラスで絞り込めるインタラクティブUIとして返す

    同一コース内の馬場×クラス×年度の絞り込み（_cross_filter_html）と同じ
    「サーバー側で全組み合わせぶん事前計算し、JS（assets/js/cross-filter.js）は
    表示/非表示の切り替えのみ行う」方式を、ここでは「クラス・馬場の各組み合わせに
    ついて他競馬場と比較する」という軸で再利用する。
    """
    ground_options = "".join(f'<option value="{gs}">{gs}</option>\n' for gs in ground_states)
    class_options = "".join(f'<option value="{cls}">{cls}</option>\n' for cls in classes)

    panels = [_venue_comparison_panel_html(place_id, race_type, course_len, "全", "all")]
    panels += [_venue_comparison_panel_html(place_id, race_type, course_len, gs, "all") for gs in ground_states]
    panels += [_venue_comparison_panel_html(place_id, race_type, course_len, "全", cls) for cls in classes]
    panels += [
        _venue_comparison_panel_html(place_id, race_type, course_len, gs, cls)
        for gs in ground_states
        for cls in classes
    ]

    return f"""<div class="cross-filter">
    <div class="cross-filter-controls">
      <label>馬場状態:
        <select class="cross-filter-ground-state">
          <option value="全">全て</option>
          {ground_options}
        </select>
      </label>
      <label>クラス:
        <select class="cross-filter-class">
          <option value="all">全て</option>
          {class_options}
        </select>
      </label>
    </div>
    {"".join(panels)}
    <p class="cross-filter-empty" hidden>対象データがありません。</p>
  </div>"""


def _load_course_winners(place_id, race_type, course_len, start_year=ANNUAL_START_YEAR, current_year=None):
    """このコース（race_type×course_len）の勝ち馬の行を、生のrace_resultsから年をまたいで一括ロードする

    平均勝ち時計・平均人気・体重・枠番・馬番・配当は、いずれも既存の年度別/合計CSV
    （race_info_dataset_manager）では「勝ち馬」だけを対象に集計されている
    （src/logic/calculators/average_calculator.pyのextract_course_race_results等を参照）。
    年度別CSVには件数（n）が無く、複数年をまとめる際に年ごとの単純平均ではレース数の
    違いを無視してしまうため、開始年（年度フィルタ）を選べる集計はここで生データから
    直接再集計する。馬場×クラス×開始年の組み合わせごとにCSVを読み直すと遅いため、
    1コースぶん（全年×全馬場×全クラス）を一度だけロードしてキャッシュし、各組み合わせの
    集計はメモリ上のDataFrameを絞り込むだけで済ませる。

    Returns:
        pd.DataFrame: 勝ち馬の行（年を表す"年"列・レースを特定する"race_id"列を含む）。
            対象データが無ければ空。
    """
    current_year = current_year or date.today().year
    frames = []
    for year in range(start_year, current_year + 1):
        df = race_result_dataset_manager.get_race_results_csv(place_id, year)
        if df.empty:
            continue
        df = df[
            (df["race_type"] == str(race_type))
            & (df["course_len"] == str(course_len))
            & (df["着順"] == "1")
        ]
        if df.empty:
            continue
        # race_resultsはrace_idをインデックスとして保存されているため、複勝配当との
        # 紐付け（_load_course_place_returns参照）に使えるよう列として取り出す。
        df = df.reset_index().rename(columns={"index": "race_id"})
        df["年"] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_course_place_returns(place_id, start_year=ANNUAL_START_YEAR, current_year=None):
    """開始年〜最新年の複勝配当（race_id×馬番ごと）を、年をまたいで一括ロードする

    平均複勝配当は、勝ち馬（1着馬）の複勝配当の平均として算出する。複勝配当は
    レース単位の配当結果（race_info_dataset_manager.get_race_returns_csv、
    consolidate_race_returnsで個別レースファイルから1年分にまとめ済み）にしかなく、
    race_resultsには含まれないため、_load_course_winnersのrace_id・馬番で
    紐付ける（_breakdown_row_from_winners参照）。

    Returns:
        pd.DataFrame: ["race_id", "馬番", "配当"]の列を持つ複勝配当の行。対象データが
            無ければ空。
    """
    current_year = current_year or date.today().year
    frames = []
    for year in range(start_year, current_year + 1):
        df = race_info_dataset_manager.get_race_returns_csv(place_id, year)
        if df.empty:
            continue
        df = df[df["式別"] == "複勝"]
        if df.empty:
            continue
        df = df.copy()
        # get_race_returns_csvのインデックス名は、consolidate_race_returns経由（名前あり）
        # か週次更新のスクレイピング経由（名前なし）かで変わるため、reset_index前に
        # 明示的に"race_id"へ統一する（インデックスの値自体はどちらもrace_id）。
        df.index.name = "race_id"
        df = df.reset_index()[["race_id", "馬番", "配当"]]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _breakdown_row_from_winners(winners_df, place_returns_df, value_label, ground_state, class_name, start_year=None):
    """ロード済みの勝ち馬DataFrame（_load_course_winners）から内訳1行分を直接集計する

    _breakdown_rowと同じ統計フィールド（avg_time/avg_pop/weight/avg_frame/avg_horse/
    win_return）に加え、勝ち馬の平均複勝配当（place_return）も返す。年度別/合計CSVの
    事前集計を経由せず、対象の勝ち馬の行を直接平均する（複数年をまとめてもレース数の
    違いを正しく反映できる）。複勝配当はrace_resultsには無いため、place_returns_df
    （_load_course_place_returns）とrace_id・馬番で紐付けて取得する。
    """
    sub = winners_df
    if start_year is not None:
        sub = sub[sub["年"] >= start_year]
    if ground_state != "全":
        sub = sub[sub["ground_state"] == ground_state]
    if class_name != "all":
        sub = sub[sub["class"] == class_name]
    if sub.empty:
        return None

    avg_time_ms = average_calculator.calc_avg_time(sub["タイム"].reset_index(drop=True))
    avg_time_fmt = _format_time(avg_time_ms) if pd.notna(avg_time_ms) else "データなし"
    if avg_time_fmt == "データなし":
        return None

    def _mean_or_none(series, ndigits=2):
        value = series.mean()
        return None if pd.isna(value) else round(float(value), ndigits)

    avg_pop = _mean_or_none(pd.to_numeric(sub["人気"], errors="coerce"))
    avg_weight = _mean_or_none(pd.to_numeric(sub["馬体重"], errors="coerce"), ndigits=1)
    avg_frame = _mean_or_none(pd.to_numeric(sub["枠番"], errors="coerce"))
    avg_horse = _mean_or_none(pd.to_numeric(sub["馬番"], errors="coerce"))
    win_return_raw = _mean_or_none(pd.to_numeric(sub["単勝"], errors="coerce") * 100, ndigits=1)

    place_return_raw = None
    if place_returns_df is not None and not place_returns_df.empty:
        merged = sub[["race_id", "馬番"]].merge(place_returns_df, on=["race_id", "馬番"], how="inner")
        if not merged.empty:
            place_return_raw = _mean_or_none(pd.to_numeric(merged["配当"], errors="coerce"), ndigits=1)

    return {
        "value": value_label,
        "avg_time": avg_time_fmt,
        "avg_pop": "データなし" if avg_pop is None else str(avg_pop),
        "avg_pop_raw": avg_pop,
        "weight": "データなし" if avg_weight is None else f"{avg_weight}kg",
        "weight_raw": avg_weight,
        "avg_frame": "データなし" if avg_frame is None else str(avg_frame),
        "avg_frame_raw": avg_frame,
        "avg_horse": "データなし" if avg_horse is None else str(avg_horse),
        "avg_horse_raw": avg_horse,
        "win_return": "データなし" if win_return_raw is None else f"{win_return_raw}円",
        "win_return_raw": win_return_raw,
        "place_return": "データなし" if place_return_raw is None else f"{place_return_raw}円",
        "place_return_raw": place_return_raw,
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


def build_cross_breakdown(place_id, race_type, course_len, oldest_year=ANNUAL_START_YEAR, current_year=None):
    """馬場状態×クラス×開始年の組み合わせごとの内訳（平均勝ち時計・人気・体重・枠番・配当）を返す

    build_class_breakdown/build_ground_state_breakdown/build_year_breakdownはそれぞれ
    1軸だけを動かすが、ここでは3軸を組み合わせて集計し、ユーザーがどの軸も選んで
    絞り込めるようにする。年度軸は単年ではなく「開始年〜最新年」の範囲集計とし、
    開始年=oldest_year（最古年）を選ぶとデフォルトで全期間になる。各軸の「全て」
    （馬場状態="全"・クラス="all"）も組み合わせに含めるため、全体合計・各軸単体の
    内訳・完全な組み合わせのすべてがこの1つの辞書に入る。年度別CSVには件数（n）が
    無く複数年の単純平均ではレース数の違いを無視してしまうため、年度軸は生データ
    （race_results）から直接re集計する（_load_course_winners/_breakdown_row_from_winners）。
    データが存在しない組み合わせは除外する。

    Returns:
        dict[tuple, dict]: {(馬場状態, クラス, 開始年): _breakdown_row_from_winnersの戻り値, ...}
            （開始年はint。開始年=oldest_yearが全期間に対応する）
    """
    current_year = current_year or date.today().year
    time_df = race_info_dataset_manager.get_total_average_time_csv(place_id)
    sub = _filter_rows(time_df, race_type, course_len, ground_state="全")
    classes = [c for c in sub["class"].unique() if c != "all"] if not sub.empty else []

    winners_df = _load_course_winners(place_id, race_type, course_len, start_year=oldest_year, current_year=current_year)
    if winners_df.empty:
        return {}
    place_returns_df = _load_course_place_returns(place_id, start_year=oldest_year, current_year=current_year)

    result = {}
    for ground_state in ["全"] + GROUND_STATE_ORDER:
        for class_name in ["all"] + classes:
            for start_year in _cross_filter_start_years(oldest_year, current_year):
                label = f"{ground_state}×{class_name}×{_cross_filter_year_label(start_year, oldest_year, current_year)}"
                row = _breakdown_row_from_winners(
                    winners_df, place_returns_df, label, ground_state, class_name, start_year=start_year,
                )
                if row is not None:
                    result[(ground_state, class_name, start_year)] = row

    baseline = result.get(("全", "all", oldest_year))
    if baseline is not None:
        for row in result.values():
            _apply_summary_stat_trends(row, baseline)
    return result


def _apply_summary_stat_trends(row, baseline):
    """平均人気・配当(単勝/複勝)・体重・枠番/馬番が、コース全体（baseline）の値より
    SUMMARY_STAT_ADVANTAGE_THRESHOLDS以上離れている場合、rowに{key}_trend（"high"/"low"/""）
    を追加する（概要カードの色付けに使う。サマリー・カードの表示用なので、対象自身が
    baselineの場合は常に差0で何も付かない）。
    """
    for key, threshold in SUMMARY_STAT_ADVANTAGE_THRESHOLDS.items():
        value = row.get(key)
        base_value = baseline.get(key)
        trend = ""
        if value is not None and base_value is not None:
            diff = value - base_value
            if diff >= threshold:
                trend = "high"
            elif diff <= -threshold:
                trend = "low"
        row[f"{key}_trend"] = trend


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


PASSAGE_KEYS = ["passage1", "passage2", "passage3", "passage4"]
PASSAGE_LABELS = {"passage1": "通過1", "passage2": "通過2", "passage3": "通過3", "passage4": "通過4"}


def available_passage_keys(place_id, race_type, course_len):
    """そのコースで実際に通過データが存在する列（通過1〜4のうち）を返す

    コースの距離によって通過のタイミング計測ポイント数が異なり（例: 短距離は
    通過1・2のみ）、無いものは表に列ごと出さない（「データなし」で埋めない）。
    全体（馬場=全・クラス=all）の行を基準に、その列が一度も記録されていなければ
    そのコースには存在しない通過ポイントと判断する。
    """
    row = _filter_overall_row_by(
        race_info_dataset_manager.get_total_winner_time_csv(place_id), race_type, course_len, "全", "all"
    )
    if row is None:
        return PASSAGE_KEYS

    keys = []
    for i, key in enumerate(PASSAGE_KEYS, start=1):
        value = row.get(f"通過{i}")
        if value is not None and value == value and str(value).strip() != "":
            keys.append(key)
    return keys


def _passage_breakdown_table_html(rows, value_label, title, passage_keys=PASSAGE_KEYS):
    if not rows:
        body = "<p>対象データがありません。</p>"
    else:
        header_cells = "".join(f"<th>{PASSAGE_LABELS[key]}</th>" for key in passage_keys)
        trs = "".join(
            f"<tr><td>{r['value']}</td><td>{r['agari']}</td>"
            + "".join(f"<td>{r[key]}</td>" for key in passage_keys)
            + "</tr>\n"
            for r in rows
        )
        body = f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>{value_label}</th><th>上り(勝ち馬)</th>{header_cells}</tr></thead>
    <tbody>
      {trs}
    </tbody>
  </table>
  </div>"""
    return f"<h3>{title}</h3>\n  {body}"


ADVANTAGE_THRESHOLD_POINTS = 8.0  # 複勝率(着内率)が平均よりこの値(%pt)以上離れていたら有利/不利と判定する

# 馬場×クラス×年度パネルの概要カード（平均人気・配当・体重・枠番/馬番）について、
# そのコース全体（全×all×全期間）の値からどれだけ離れていれば「傾向がある」と見るかの
# 閾値。全コースの馬場別・クラス別の内訳とコース全体平均との差（849件）を実データで
# 集計し、上位10%程度（ノイズではなくはっきりした差と言えるライン）を採用している。
SUMMARY_STAT_ADVANTAGE_THRESHOLDS = {
    "avg_pop_raw": 1.0,
    "win_return_raw": 500.0,
    "place_return_raw": 100.0,
    "weight_raw": 15.0,
    "avg_frame_raw": 0.9,
    "avg_horse_raw": 1.7,
}


def _chakudo_rows(df, race_type, course_len, rank_column, rank_range, ground_state="全", class_name="all"):
    """rank_range分の着度数（1着/2着/3着/着外・着内率）の行を返す（データなしは0で埋める）

    着度数CSVはrace_type/course_len/ground_state/classの組み合わせごとに行を持つため、
    既定（ground_state="全", class="all"）では全体の集計を、クラス別・馬場別の内訳を
    見たい場合はground_state/class_nameを指定して絞り込む。
    """
    sub = (
        df[
            (df["race_type"] == race_type)
            & (df["course_len"].astype(str) == str(course_len))
            & (df["ground_state"] == ground_state)
            & (df["class"] == class_name)
        ]
        if not df.empty
        else df
    )
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


def _advantage_labels(items, exclude=(), high_label="◎ 有利", low_label="▲ 不利", threshold=ADVANTAGE_THRESHOLD_POINTS):
    """(id, value)のリストから、有利/不利のラベルをidごとに算出する（共通ロジック）

    対象（valueがNoneでなくexcludeに含まれないid）の平均からの差がthreshold以上の
    場合のみラベルを付ける。差が小さい（フラットな）場合や対象が1件以下の場合は
    何も付けない。low_labelを空文字にすると不利側のラベルは付けない（人気データのように
    「人気が低いほど不利なのは当然」で不要な場合）。

    Returns:
        dict[Any, tuple[str, str]]: {id: (note_text, note_kind)}（note_kindは"high"/"low"/""）
    """
    candidates = [(i, v) for i, v in items if v is not None and i not in exclude]
    if len(candidates) < 2:
        return {i: ("", "") for i, _ in items}

    avg = sum(v for _, v in candidates) / len(candidates)
    labels = {}
    for i, v in items:
        if v is None or i in exclude:
            labels[i] = ("", "")
            continue
        diff = v - avg
        if diff >= threshold:
            labels[i] = (high_label, "high")
        elif low_label and diff <= -threshold:
            labels[i] = (low_label, "low")
        else:
            labels[i] = ("", "")
    return labels


def _label_advantage(rows, exclude_ranks=(), high_label="◎ 有利", low_label="▲ 不利"):
    """着度数の各行（rank/total/top3_rate）に有利/不利のラベル（note_text/note_kind）を付与する"""
    items = [(r["rank"], r["top3_rate"] if r["total"] > 0 else None) for r in rows]
    labels = _advantage_labels(items, exclude=exclude_ranks, high_label=high_label, low_label=low_label)
    for r in rows:
        r["note_text"], r["note_kind"] = labels[r["rank"]]
    return rows


def _advantage_badge_html(note_text, note_kind):
    """有利/不利のラベルを色付きバッジHTMLにする（ラベルが無ければ空文字）"""
    if not note_text:
        return ""
    return f'<span class="advantage-badge {note_kind}">{note_text}</span>'


def _feature_callout_html(race_type, course_len, ground_state, class_name, frame_chakudo_df):
    """この馬場×クラスの組み合わせに、外枠/内枠の有利・不利がはっきり出ていれば一言で表示する

    枠番別着度数から内枠（1〜4枠）・外枠（5〜8枠）それぞれの3着内率を比較し、
    差がADVANTAGE_THRESHOLD_POINTS（%pt）以上あるときだけ「外枠優位」「内枠優位」を
    表示する。差が小さい・サンプルが無い場合は何も表示しない
    （無理に特徴を出そうとはせず、はっきりした傾向だけを伝える）。
    """
    rows = [
        r for r in _chakudo_rows(frame_chakudo_df, race_type, course_len, "枠番", range(1, 9), ground_state=ground_state, class_name=class_name)
        if r["total"] > 0
    ]
    if not rows:
        return ""

    inner_rows = [r for r in rows if r["rank"] <= 4]
    outer_rows = [r for r in rows if r["rank"] >= 5]
    inner_total = sum(r["total"] for r in inner_rows)
    outer_total = sum(r["total"] for r in outer_rows)
    if inner_total == 0 or outer_total == 0:
        return ""

    inner_top3 = sum(r["1着"] + r["2着"] + r["3着"] for r in inner_rows) / inner_total * 100
    outer_top3 = sum(r["1着"] + r["2着"] + r["3着"] for r in outer_rows) / outer_total * 100
    diff = outer_top3 - inner_top3

    if diff >= ADVANTAGE_THRESHOLD_POINTS:
        return (
            '<p class="feature-callout">外枠優位の傾向があります'
            f'（外枠3着内率 {outer_top3:.1f}% / 内枠 {inner_top3:.1f}%）</p>'
        )
    if diff <= -ADVANTAGE_THRESHOLD_POINTS:
        return (
            '<p class="feature-callout">内枠優位の傾向があります'
            f'（内枠3着内率 {inner_top3:.1f}% / 外枠 {outer_top3:.1f}%）</p>'
        )
    return ""


CHAKUDO_SEGMENT_LABELS = ["1着", "2着", "3着", "着外"]
CHAKUDO_SEGMENT_CLASSES = {"1着": "seg-1st", "2着": "seg-2nd", "3着": "seg-3rd", "着外": "seg-out"}


def _waku_label_class(waku):
    """枠番（1〜8）を、出馬表ページと同じ枠色のCSSクラス名（waku-1〜waku-8）に変換する"""
    return f"waku-{waku}"


def _chakudo_chart_html(df, race_type, course_len, rank_column, rank_range, title, exclude_ranks=(), high_label="◎ 有利", low_label="▲ 不利", show_advantage=True, ground_state="全", class_name="all", heading_level="h3", label_formatter=None, label_class_formatter=None, compact=False):
    """人気別・枠番別・馬番別の着度数を、1着/2着/3着/着外の積み上げ横バーチャートで表示する

    各ランクの全出走数を100%として、1着/2着/3着/着外の内訳を色分けした帯で示す
    （単一の％バーよりも結果の分布がひと目で分かる）。帯が細い場合も件数は省略せず
    表示する（マウスホバーのツールチップでは割合・2着・3着は1着からの累積割合も
    確認できる）。出走自体が無いランクは行ごと表示しない（「データなし」とは書かない）。

    有利/不利のバッジは固定幅の枠（.chakudo-badge-slot）に入れ、バッジの有無に
    関わらずバーの開始位置が行ごとにズレないようにする。show_advantage=Trueの
    ときのみ、着内率が他より明確に高い/低いランクにバッジを付ける（人気データは
    「人気が高いほど強いのは当然」で意味が薄いため、呼び出し側でshow_advantage=False
    にして無効化する）。ground_state/class_nameを指定すると、その条件のみに
    絞り込んだ内訳（クラス別・馬場別の個別チャート）を描ける。label_formatterを
    指定すると、表示用ラベルだけをrank値から変換する（例: 馬体重帯450→"450kg台"）。
    一致判定自体は元のrank値（rank_range）で行うため、表示の変換は結果に影響しない。
    label_class_formatterを指定すると、rank値からCSSクラス名を生成しchakudo-labelに
    付与する（枠番別チャートで出馬表ページと同じ枠色を付ける_waku_label_class用）。

    compact=Trueにすると、ツールチップ・凡例文を省いた軽量版になる（同じ内容の
    バーチャートを馬場×クラス×年度の組み合わせぶん何十回も並べるcross-filter
    パネルで使う。組み合わせ数に比例してページサイズが膨らむのを抑えつつ、
    見た目はバーチャートのまま保つ）。
    """
    if df.empty:
        return f"<{heading_level}>{title}</{heading_level}>\n  <p>対象データがありません。</p>"

    rows = [
        r for r in _chakudo_rows(df, race_type, course_len, rank_column, rank_range, ground_state=ground_state, class_name=class_name)
        if r["total"] > 0
    ]
    if not rows:
        return f"<{heading_level}>{title}</{heading_level}>\n  <p>対象データがありません。</p>"

    if show_advantage:
        rows = _label_advantage(rows, exclude_ranks=exclude_ranks, high_label=high_label, low_label=low_label)
    else:
        for r in rows:
            r["note_text"], r["note_kind"] = "", ""

    bar_rows = ""
    for r in rows:
        segments = ""
        cumulative = 0.0
        for label in CHAKUDO_SEGMENT_LABELS:
            count = r[label]
            width = count / r["total"] * 100
            cumulative += width
            if compact:
                tooltip_html = ""
            elif label in ("2着", "3着"):
                tooltip_html = f'<span class="chakudo-tooltip">{label}: {width:.1f}%（累積{cumulative:.1f}%）</span>'
            else:
                tooltip_html = f'<span class="chakudo-tooltip">{label}: {width:.1f}%</span>'
            segments += (
                f'<span class="chakudo-segment {CHAKUDO_SEGMENT_CLASSES[label]}" '
                f'style="width: {width:.2f}%">{count}{tooltip_html}</span>'
            )
        badge = _advantage_badge_html(r["note_text"], r["note_kind"])
        label = label_formatter(r["rank"]) if label_formatter else r["rank"]
        label_class = f" {label_class_formatter(r['rank'])}" if label_class_formatter else ""
        bar_rows += f"""<div class="chakudo-row">
      <span class="chakudo-label{label_class}">{label}</span>
      <span class="chakudo-badge-slot">{badge}</span>
      <span class="chakudo-bar-track">{segments}</span>
      <span class="chakudo-value">n={r['total']}</span>
    </div>\n"""

    legend_html = (
        ""
        if compact
        else '<p class="chakudo-legend">着度数 (1着,2着,3着,着外) ／ '
        "バーにマウスを合わせると内訳の割合（2着・3着は累積割合も）を確認できます</p>\n  "
    )
    return f"""<{heading_level}>{title}</{heading_level}>
  {legend_html}<div class="chakudo-chart">
{bar_rows}  </div>"""


def _weight_bucket_label(bucket_start):
    """馬体重帯の下限値（例: 450）を表示用ラベル（例: "450kg台"）にする

    両端（サンプル数が極端に少ない390kg未満・550kg以上）は1つの帯にまとめているため、
    "kg台"ではなく"未満"/"以上"の開放区間として表示する。
    """
    bucket_start = int(bucket_start)
    if bucket_start < WEIGHT_BUCKET_LOW:
        return f"{WEIGHT_BUCKET_LOW}kg未満"
    if bucket_start >= WEIGHT_BUCKET_HIGH:
        return f"{WEIGHT_BUCKET_HIGH}kg以上"
    return f"{bucket_start}kg台"


def _weight_trend_chart_html(df, race_type, course_len, ground_state="全", class_name="all"):
    """馬体重帯ごとの3着内率を、バーチャートより滑らかな折れ線グラフ（実数値ベース）で表示する

    着度数の積み上げ横バーチャート（_chakudo_chart_html）は帯ごとの内訳がよく分かる
    一方、帯と帯の間の変化（傾向の形）はバーの並びだけでは読み取りにくい。
    同じ集計データから3着内率を1本の折れ線にすることで、どの体重帯あたりに
    ピークがあるか等の傾向をより詳細に確認できるようにする。
    """
    rows = [
        r for r in _chakudo_rows(df, race_type, course_len, "体重帯", WEIGHT_BUCKET_RANGE, ground_state=ground_state, class_name=class_name)
        if r["total"] > 0
    ]
    if len(rows) < 2:
        return ""
    labels = [_weight_bucket_label(r["rank"]) for r in rows]
    values = [r["top3_rate"] for r in rows]
    counts = [r["total"] for r in rows]
    return single_line_trend_svg(labels, values, counts=counts)


def _available_chakudo_classes(df, race_type, course_len):
    """着度数CSVから、そのコースで実際に存在するクラス（"all"を除く）を返す"""
    if df.empty:
        return []
    sub = df[
        (df["race_type"] == race_type) & (df["course_len"].astype(str) == str(course_len)) & (df["ground_state"] == "全")
    ]
    return [c for c in sub["class"].unique() if c != "all"]


def _available_chakudo_ground_states(df, race_type, course_len):
    """着度数CSVから、そのコースで実際に存在する馬場状態（"全"を除く）を良→稍重→重→不良の順で返す"""
    if df.empty:
        return []
    sub = df[
        (df["race_type"] == race_type) & (df["course_len"].astype(str) == str(course_len)) & (df["class"] == "all")
    ]
    found = set(sub["ground_state"].unique())
    return [g for g in GROUND_STATE_ORDER if g in found]


def _chakudo_class_breakdown_html(df, race_type, course_len, rank_column, rank_range, exclude_ranks=(), high_label="◎ 有利", low_label="▲ 不利", show_advantage=True, label_formatter=None, label_class_formatter=None):
    """クラスごとの着度数チャートを連結して返す（h4見出し、馬場状態は全体「全」で固定）"""
    classes = _available_chakudo_classes(df, race_type, course_len)
    if not classes:
        return "<p>対象データがありません。</p>"
    return "\n  ".join(
        _chakudo_chart_html(
            df, race_type, course_len, rank_column, rank_range, class_name,
            exclude_ranks=exclude_ranks, high_label=high_label, low_label=low_label, show_advantage=show_advantage,
            ground_state="全", class_name=class_name, heading_level="h4", label_formatter=label_formatter,
            label_class_formatter=label_class_formatter,
        )
        for class_name in classes
    )


def _chakudo_ground_state_breakdown_html(df, race_type, course_len, rank_column, rank_range, exclude_ranks=(), high_label="◎ 有利", low_label="▲ 不利", show_advantage=True, label_formatter=None, label_class_formatter=None):
    """馬場状態ごとの着度数チャートを連結して返す（h4見出し、クラスは全体「all」で固定、良→稍重→重→不良の順）"""
    ground_states = _available_chakudo_ground_states(df, race_type, course_len)
    if not ground_states:
        return "<p>対象データがありません。</p>"
    return "\n  ".join(
        _chakudo_chart_html(
            df, race_type, course_len, rank_column, rank_range, ground_state,
            exclude_ranks=exclude_ranks, high_label=high_label, low_label=low_label, show_advantage=show_advantage,
            ground_state=ground_state, class_name="all", heading_level="h4", label_formatter=label_formatter,
            label_class_formatter=label_class_formatter,
        )
        for ground_state in ground_states
    )


def _breakdown_table_html(rows, value_label, title):
    """クラス別・馬場別・年度別の内訳表（平均タイム/人気/体重/枠番/配当）をHTMLにする

    平均配当（win_return_raw）が他の行より明確に高い/低い行には「傾向」列に
    色付きバッジで注目（高配当）/堅め（低配当）を表示する（差が小さければ何も表示しない）。
    """
    if not rows:
        body = "<p>対象データがありません。</p>"
    else:
        # 平均配当は円単位（数百〜数千円）で、的中率等とは値のスケールが異なるため、
        # ADVANTAGE_THRESHOLD_POINTS（%pt用）ではなくここでは円単位の固定閾値を使う
        labels = _advantage_labels(
            [(r["value"], r["win_return_raw"]) for r in rows],
            high_label="◎ 注目（高配当）", low_label="▲ 堅め（低配当）", threshold=100.0,
        )
        trs = "".join(
            f"<tr><td>{r['value']}</td><td>{r['avg_time']}</td><td>{r['avg_pop']}</td>"
            f"<td>{r['weight']}</td><td>{r['avg_frame']}</td><td>{r['avg_horse']}</td><td>{r['win_return']}</td>"
            f"<td>{_advantage_badge_html(*labels[r['value']])}</td></tr>\n"
            for r in rows
        )
        body = f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>{value_label}</th><th>平均勝ち時計</th><th>平均人気</th><th>勝ち馬平均体重</th><th>平均枠番</th><th>平均馬番</th><th>平均配当(単勝)</th><th>傾向</th></tr></thead>
    <tbody>
      {trs}
    </tbody>
  </table>
  </div>"""
    return f"<h3>{title}</h3>\n  {body}"


def _peds_table_for_combo(
    place_id, race_type, course_len, ground_state, class_name, year=None, current_year=None, top_n=5,
):
    """馬場状態×クラス×開始年（〜最新年）の組み合わせに対応する血統別成績（上位top_n件）を返す

    peds_results_dataset_manager.get_total/annual_peds_results_csvのground_state引数は
    「全体」をsentinel "全"ではなく"all"で表す（馬場別ファイルが
    Total/{race_type}_{course_len}m_{ground_state}.csvとして個別に存在し、
    全体合計分は"_all.csv"という別ファイルのため）。クラス側のsentinelは
    他の集計と同じ"all"で共通のため変換不要。

    yearを指定すると、その年〜current_year（省略時は今年）までの年度別ファイルを
    合算する（単年ならそのまま、複数年なら血統名でグループ化して件数を合計する。
    着度数は単純な件数のため、年度別CSVを足し合わせるだけで正しく集計できる）。
    """
    peds_ground_state = "all" if ground_state == "全" else ground_state
    if year is None:
        df = peds_results_dataset_manager.get_total_peds_results_csv(place_id, race_type, course_len, peds_ground_state)
    else:
        current_year = current_year or date.today().year
        frames = [
            year_df
            for y in range(year, current_year + 1)
            for year_df in [
                peds_results_dataset_manager.get_annual_peds_results_csv(place_id, y, race_type, course_len, peds_ground_state)
            ]
            if not year_df.empty
        ]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty or "クラス" not in df.columns:
        return None
    sub = df[df["クラス"] == class_name].copy()
    if sub.empty:
        return None
    for col in ["1着", "2着", "3着", "着外"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
    if year is not None and year != current_year:
        sub = sub.groupby("血統", as_index=False)[["1着", "2着", "3着", "着外"]].sum()
    sub = sub.sort_values("1着", ascending=False).head(top_n)
    return sub if not sub.empty else None


def _cross_filter_panel_html(
    place_id, race_type, course_len, ground_state, class_name, start_year, oldest_year, current_year, row,
    pop_chakudo_df, frame_chakudo_df, horse_chakudo_df, weight_chakudo_df,
):
    """馬場状態×クラス×開始年1組み合わせ分のパネルを返す

    平均勝ち時計・平均人気等の概要は、表（数字の羅列）ではなくページ上部の
    summary-statsと同じ「大きな数字+ラベル」のカードで表示し、データの羅列に
    見えないようにする。最も重要な指標である平均勝ち時計だけ大きく
    （summary-stat-primary）、それ以外（平均人気・平均配当(単勝/複勝)・
    勝ち馬の平均体重・平均枠番/馬番）はサブ（summary-stat-secondary）として
    一回り小さく表示し、主従関係を一目で分かるようにする。どの組み合わせを
    見ているかが選択肢だけでは分かりにくいため、見出しに馬場状態×クラス×年度
    （開始年〜最新年）を明示する。

    詳細データは、血統データ→枠番別→馬番別→馬体重別→人気別の順に折りたたみで並べる
    （この順は人気・枠順タブ等、他の参考データの並びとは独立に決めている）。
    """
    year_label = _cross_filter_year_label(start_year, oldest_year, current_year)
    heading = f"{ground_state} × {class_name} × {year_label}"
    if row is None:
        body = "<p>対象データがありません。</p>"
    else:
        def trend_span(text, trend_key):
            trend = row.get(trend_key, "")
            cls = f" value-trend-{trend}" if trend else ""
            return f'<span class="{cls.strip()}">{text}</span>' if cls else text

        body = f"""<div class="summary-stats">
    <div class="summary-stat summary-stat-primary">
      <span class="value">{row['avg_time']}</span>
      <span class="label">平均勝ち時計</span>
    </div>
    <div class="summary-stat summary-stat-secondary">
      <span class="value">{trend_span(row['avg_pop'], 'avg_pop_raw_trend')}</span>
      <span class="label">平均人気</span>
    </div>
    <div class="summary-stat summary-stat-secondary">
      <span class="value">{trend_span(row['win_return'], 'win_return_raw_trend')}</span>
      <span class="label">平均配当（単勝）</span>
    </div>
    <div class="summary-stat summary-stat-secondary">
      <span class="value">{trend_span(row['place_return'], 'place_return_raw_trend')}</span>
      <span class="label">平均配当（複勝）</span>
    </div>
    <div class="summary-stat summary-stat-secondary">
      <span class="value">{trend_span(row['weight'], 'weight_raw_trend')}</span>
      <span class="label">勝ち馬平均体重</span>
    </div>
    <div class="summary-stat summary-stat-secondary">
      <span class="value">{trend_span(row['avg_frame'], 'avg_frame_raw_trend')} / {trend_span(row['avg_horse'], 'avg_horse_raw_trend')}</span>
      <span class="label">平均枠番 / 平均馬番</span>
    </div>
  </div>"""

    # cross-filterは組み合わせ数（馬場×クラス×年度）が多いため、常に見える「全体」の
    # 表とは別に、ツールチップ・凡例文を省いた軽量版（compact=True）のバーチャートを
    # 使う。バー自体は通常表示と同じ見た目を保ちつつ、組み合わせ数に比例した
    # ページサイズの肥大化はある程度抑える。
    pop_chakudo_html = _chakudo_chart_html(
        pop_chakudo_df, race_type, course_len, "人気", range(1, 19), "人気別着度数",
        show_advantage=False, ground_state=ground_state, class_name=class_name, heading_level="h4", compact=True,
    )
    frame_chakudo_html = _chakudo_chart_html(
        frame_chakudo_df, race_type, course_len, "枠番", range(1, 9), "枠番別着度数",
        ground_state=ground_state, class_name=class_name, heading_level="h4",
        label_class_formatter=_waku_label_class, compact=True,
    )
    horse_chakudo_html = _chakudo_chart_html(
        horse_chakudo_df, race_type, course_len, "馬番", range(1, 19), "馬番別着度数",
        ground_state=ground_state, class_name=class_name, heading_level="h4", compact=True,
    )
    weight_chakudo_html = _chakudo_chart_html(
        weight_chakudo_df, race_type, course_len, "体重帯", WEIGHT_BUCKET_RANGE, "馬体重帯別着度数",
        ground_state=ground_state, class_name=class_name, heading_level="h4", label_formatter=_weight_bucket_label,
        compact=True,
    )
    peds_html = _peds_chart_html(
        _peds_table_for_combo(
            place_id, race_type, course_len, ground_state, class_name,
            year=None if start_year <= oldest_year else start_year, current_year=current_year,
        ),
        "血統別成績（上位5件）", heading_level="h4", compact=True,
    )
    feature_html = _feature_callout_html(race_type, course_len, ground_state, class_name, frame_chakudo_df)

    return f"""<div class="cross-filter-panel" data-ground-state="{ground_state}" data-class="{class_name}" data-year="{start_year}" hidden>
    <h3>{heading}</h3>
    {feature_html}
    {body}
    <details class="breakdown" open>
      <summary>血統データを表示</summary>
      {peds_html}
    </details>
    <details class="breakdown" open>
      <summary>枠番データを表示</summary>
      {frame_chakudo_html}
    </details>
    <details class="breakdown" open>
      <summary>馬番データを表示</summary>
      {horse_chakudo_html}
    </details>
    <details class="breakdown" open>
      <summary>馬体重データを表示</summary>
      {weight_chakudo_html}
    </details>
    <details class="breakdown" open>
      <summary>人気データを表示</summary>
      {pop_chakudo_html}
    </details>
  </div>"""


def _cross_filter_html(
    place_id, race_type, course_len, cross_breakdown,
    pop_chakudo_df, frame_chakudo_df, horse_chakudo_df, weight_chakudo_df, class_names, oldest_year, current_year,
):
    """馬場状態×クラス×開始年を3つのセレクトボックスで選び、平均勝ち時計・人気・体重・
    枠番・配当に加え、人気・枠順データ（着度数）・血統データもその組み合わせに
    絞り込んで表示するインタラクティブなUIを返す

    年度は単年ではなく「開始年〜最新年」の範囲集計とし、開始年=oldest_year
    （最古年）を選ぶとデフォルトで全期間になる（開始年・終了年を両方自由に選べる
    ようにすると組み合わせ数が階乗的に増えるため、終了年は常に最新年に固定する）。
    ai_performance_report_generator._cross_filter_htmlと同じ「サーバー側で全組み合わせ
    ぶん事前計算し、JS（assets/js/cross-filter.js）は表示/非表示の切り替えのみ行う」
    方針を踏襲する（同じJSファイルをそのまま再利用できる。年度セレクトの有無は
    JS側で動的に検出するため、2軸のみのページにも影響しない）。
    """
    ground_options = "".join(f'<option value="{gs}">{gs}</option>\n' for gs in GROUND_STATE_ORDER)
    class_options = "".join(f'<option value="{cls}">{cls}</option>\n' for cls in class_names)
    year_options = "".join(
        f'<option value="{y}">{_cross_filter_year_label(y, oldest_year, current_year)}</option>\n'
        for y in _cross_filter_start_years(oldest_year, current_year)
        if y > oldest_year
    )

    panels = [
        _cross_filter_panel_html(
            place_id, race_type, course_len, gs, cls, start_year, oldest_year, current_year, row,
            pop_chakudo_df, frame_chakudo_df, horse_chakudo_df, weight_chakudo_df,
        )
        for (gs, cls, start_year), row in cross_breakdown.items()
    ]

    return f"""<div class="cross-filter">
    <div class="cross-filter-controls">
      <label>馬場状態:
        <select class="cross-filter-ground-state">
          <option value="全">全て</option>
          {ground_options}
        </select>
      </label>
      <label>クラス:
        <select class="cross-filter-class">
          <option value="all">全て</option>
          {class_options}
        </select>
      </label>
      <label>開始年（〜{current_year}年）:
        <select class="cross-filter-year">
          <option value="{oldest_year}">全期間</option>
          {year_options}
        </select>
      </label>
    </div>
    {"".join(panels)}
    <p class="cross-filter-empty" hidden>対象データがありません。</p>
  </div>"""


def _peds_chart_html(peds_df, title, heading_level="h3", top_n=10, compact=False):
    """血統別成績（1着/2着/3着/着外）を、着度数と同じ積み上げ横バーチャートで表示する

    血統（種牡馬）ごとに総戦数が大きく異なるため、表（数字の羅列）よりも分布の
    良し悪しが一目で分かるよう、人気/枠番/馬番と同じチャート（_chakudo_chart_html）の
    見た目を再利用する。バーの幅は血統ごとの総戦数を100%とした内訳の割合にし、
    末尾にn=総戦数を表示することで、割合（成績の良し悪し）と総数（サンプルの
    信頼度）の両方を一目で確認できるようにする。

    compact=Trueの場合は_chakudo_chart_htmlと同様、ツールチップ・凡例文を省いた
    軽量版になる（cross-filterパネル用）。
    """
    if peds_df is None or peds_df.empty:
        return f"<{heading_level}>{title}</{heading_level}>\n  <p>対象データがありません。</p>"

    bar_rows = ""
    for _, row in peds_df.head(top_n).iterrows():
        counts = {label: int(row[label]) for label in CHAKUDO_SEGMENT_LABELS}
        total = sum(counts.values())
        if total == 0:
            continue
        segments = ""
        cumulative = 0.0
        for label in CHAKUDO_SEGMENT_LABELS:
            count = counts[label]
            width = count / total * 100
            cumulative += width
            if compact:
                tooltip_html = ""
            elif label in ("2着", "3着"):
                tooltip_html = f'<span class="chakudo-tooltip">{label}: {width:.1f}%（累積{cumulative:.1f}%）</span>'
            else:
                tooltip_html = f'<span class="chakudo-tooltip">{label}: {width:.1f}%</span>'
            segments += (
                f'<span class="chakudo-segment {CHAKUDO_SEGMENT_CLASSES[label]}" '
                f'style="width: {width:.2f}%">{count}{tooltip_html}</span>'
            )
        bar_rows += f"""<div class="chakudo-row">
      <span class="chakudo-label peds-label">{row['血統']}</span>
      <span class="chakudo-bar-track">{segments}</span>
      <span class="chakudo-value">n={total}</span>
    </div>\n"""

    if not bar_rows:
        return f"<{heading_level}>{title}</{heading_level}>\n  <p>対象データがありません。</p>"

    legend_html = (
        ""
        if compact
        else '<p class="chakudo-legend">着度数 (1着,2着,3着,着外) ／ '
        "バーにマウスを合わせると内訳の割合（2着・3着は累積割合も）を確認できます</p>\n  "
    )
    return f"""<{heading_level}>{title}</{heading_level}>
  {legend_html}<div class="chakudo-chart">
{bar_rows}  </div>"""


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


def build_peds_ground_state_breakdown(place_id, race_type, course_len, top_n=5):
    """このコースの血統別成績を馬場別（"all"クラスのみ）上位top_n件で返す（良→稍重→重→不良の順）

    Total/{race_type}_{course_len}m_{ground_state}.csv は馬場状態ごとに個別のファイルとして
    既に集計済みのため、新しい統計ロジックは追加せずそれぞれ読み取るのみ。
    """
    result = []
    for ground_state in GROUND_STATE_ORDER:
        df = peds_results_dataset_manager.get_total_peds_results_csv(place_id, race_type, course_len, ground_state)
        if df.empty or "クラス" not in df.columns:
            continue
        sub = df[df["クラス"] == "all"].copy()
        if sub.empty:
            continue
        for col in ["1着", "2着", "3着", "着外"]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
        sub = sub.sort_values("1着", ascending=False).head(top_n)
        if not sub.empty:
            result.append({"ground_state": ground_state, "peds_df": sub})
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
        _peds_chart_html(item["peds_df"], f"{item[label_key]}（上位{len(item['peds_df'])}件）", heading_level="h4")
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

    # クラス名の一覧は、馬場×クラス×年度の絞り込みUI（メインタブ）のセレクトボックスの
    # 選択肢として使う（単体の内訳表は廃止し、この絞り込みUIに統合した）。
    current_year = date.today().year
    class_breakdown = build_class_breakdown(place_id, race_type, course_len)
    cross_breakdown = build_cross_breakdown(place_id, race_type, course_len, current_year=current_year)

    class_passage_breakdown = build_class_passage_breakdown(place_id, race_type, course_len)
    ground_state_passage_breakdown = build_ground_state_passage_breakdown(place_id, race_type, course_len)
    year_passage_breakdown = build_year_passage_breakdown(place_id, race_type, course_len)
    passage_keys = available_passage_keys(place_id, race_type, course_len)

    pop_chakudo_df = race_info_dataset_manager.get_total_pop_chakudo_csv(place_id)
    frame_chakudo_df = race_info_dataset_manager.get_total_frame_chakudo_csv(place_id)
    horse_chakudo_df = race_info_dataset_manager.get_total_horse_chakudo_csv(place_id)
    weight_chakudo_df = race_info_dataset_manager.get_total_weight_chakudo_csv(place_id)

    venue_comparison_ground_states = _available_chakudo_ground_states(frame_chakudo_df, race_type, course_len)
    venue_comparison_classes = [r["value"] for r in class_breakdown]
    venue_comparison_html = _venue_comparison_cross_filter_html(
        place_id, race_type, course_len, venue_comparison_ground_states, venue_comparison_classes,
    )

    # 人気は「ねらい目」がだいたい3〜6番人気あたりに集中し、あまり意味のある
    # 情報にならないため、有利/不利のバッジ付けは行わない（クラス別・馬場別も同様）
    pop_chakudo_chart = _chakudo_chart_html(
        pop_chakudo_df, race_type, course_len, "人気", range(1, 19), "人気データ（人気別着度数、全体）",
        show_advantage=False,
    )
    pop_chakudo_class_html = _chakudo_class_breakdown_html(
        pop_chakudo_df, race_type, course_len, "人気", range(1, 19), show_advantage=False,
    )
    pop_chakudo_ground_html = _chakudo_ground_state_breakdown_html(
        pop_chakudo_df, race_type, course_len, "人気", range(1, 19), show_advantage=False,
    )

    frame_chakudo_chart = _chakudo_chart_html(
        frame_chakudo_df, race_type, course_len, "枠番", range(1, 9), "枠順データ（枠番別着度数、全体）",
        label_class_formatter=_waku_label_class,
    )
    frame_chakudo_class_html = _chakudo_class_breakdown_html(
        frame_chakudo_df, race_type, course_len, "枠番", range(1, 9), label_class_formatter=_waku_label_class,
    )
    frame_chakudo_ground_html = _chakudo_ground_state_breakdown_html(
        frame_chakudo_df, race_type, course_len, "枠番", range(1, 9), label_class_formatter=_waku_label_class,
    )

    horse_chakudo_chart = _chakudo_chart_html(
        horse_chakudo_df, race_type, course_len, "馬番", range(1, 19), "枠順データ（馬番別着度数、全体）",
    )
    horse_chakudo_class_html = _chakudo_class_breakdown_html(horse_chakudo_df, race_type, course_len, "馬番", range(1, 19))
    horse_chakudo_ground_html = _chakudo_ground_state_breakdown_html(horse_chakudo_df, race_type, course_len, "馬番", range(1, 19))

    weight_chakudo_chart = _chakudo_chart_html(
        weight_chakudo_df, race_type, course_len, "体重帯", WEIGHT_BUCKET_RANGE, "馬体重データ（馬体重帯別着度数、全体）",
        label_formatter=_weight_bucket_label,
    )
    weight_chakudo_class_html = _chakudo_class_breakdown_html(
        weight_chakudo_df, race_type, course_len, "体重帯", WEIGHT_BUCKET_RANGE,
        label_formatter=_weight_bucket_label,
    )
    weight_chakudo_ground_html = _chakudo_ground_state_breakdown_html(
        weight_chakudo_df, race_type, course_len, "体重帯", WEIGHT_BUCKET_RANGE,
        label_formatter=_weight_bucket_label,
    )
    weight_trend_chart = _weight_trend_chart_html(weight_chakudo_df, race_type, course_len)

    cross_filter_html = _cross_filter_html(
        place_id, race_type, course_len, cross_breakdown,
        pop_chakudo_df, frame_chakudo_df, horse_chakudo_df, weight_chakudo_df,
        [item["value"] for item in class_breakdown],
        ANNUAL_START_YEAR, current_year,
    )

    peds_df = report["peds_df"]
    peds_total_df = peds_df[peds_df["クラス"] == "all"] if not peds_df.empty and "クラス" in peds_df.columns else peds_df
    peds_class_breakdown = build_peds_class_breakdown(peds_df)
    peds_ground_state_breakdown = build_peds_ground_state_breakdown(place_id, race_type, course_len)
    peds_year_breakdown = build_peds_year_breakdown(place_id, race_type, course_len)

    current_label_html = course_label_html(race_type, course_len)
    breadcrumb_items = [
        ("コース詳細データ", "courses/index.html"),
        (place_name, f"courses/{PLACE_LIST[place_id - 1]}/index.html"),
        (current_label_html, None),
    ]
    breadcrumb = breadcrumb_html(breadcrumb_items, base_path="../../")

    # 右側タブでは、横並びのbreadcrumbとは別に「全競馬場」「この競馬場の全コース」も
    # 兄弟項目として並べて表示し、どの競馬場・コースへもタブから直接遷移できるようにする。
    # コースのラベルは芝/ダートで色分けし、一覧の中でも区別しやすくする。
    venue_siblings = [(NAME_LIST[i], f"courses/{PLACE_LIST[i]}/index.html") for i in range(len(PLACE_LIST))]
    course_siblings = [
        (
            course_label_html(rt, cl),
            None if (rt, cl) == (race_type, course_len) else f"courses/{PLACE_LIST[place_id - 1]}/{rt}-{cl}.html",
        )
        for rt, cl in COURSE_LISTS[place_id - 1]
    ]
    tab_hierarchy_items = [
        ("コース詳細データ", "courses/index.html"),
        (place_name, f"courses/{PLACE_LIST[place_id - 1]}/index.html", venue_siblings),
        (current_label_html, None, course_siblings),
    ]

    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_tags_html(
      f"{place_name} {race_type}{course_len}m コース詳細",
      f"{place_name}競馬場 {race_type}{course_len}mのコース傾向・血統・枠番データ分析。",
      f"{SITE_URL}/courses/{PLACE_LIST[place_id - 1]}/{race_type}-{course_len}.html",
  )}
  {adsense_script_html()}
  {ga4_script_html()}
  <title>{place_name} {race_type}{course_len}m コース詳細</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="../../assets/favicon.svg">
</head>
<body class="section-courses">
  {site_nav_html(base_path="../../", breadcrumb_items=tab_hierarchy_items)}
  {breadcrumb}
  <p><a href="index.html">&larr; {place_name}のコース一覧へ</a></p>
  <h1>{place_name} {current_label_html} コース詳細</h1>

  <!-- 常時同じ「全期間トータル」を表示していたsummary-statsは、馬場×クラス×年度の
       メインタブで同じ情報（より柔軟な絞り込み付き）が見られるため廃止した。
       将来的にはここにこのコースの解説・特徴（手動執筆）を表示する想定。 -->

  <div class="tabbed-section">
    <div class="section-tabs">
      <button class="tab-main" data-target="cross" aria-selected="true">馬場×クラス×年度（メイン）</button>
      <span class="section-tabs-sub-label">参考データ:</span>
      <button data-target="passage" aria-selected="false">通過順</button>
      <button data-target="chakudo" aria-selected="false">人気・枠順</button>
      <button data-target="weight" aria-selected="false">馬体重別成績</button>
      <button data-target="peds" aria-selected="false">血統別成績</button>
      <button data-target="venue-compare" aria-selected="false">他競馬場比較</button>
    </div>

    <div class="section-panel" data-section="cross">
      <p class="section-panel-intro">
        馬場状態・クラス・年度を選び、必要な組み合わせの成績だけを表示します
        （いずれも「全て」を選ぶと単体の内訳・全体の成績になります）。
      </p>
      {cross_filter_html}
    </div>

    <div class="section-panel" data-section="passage" hidden>
      {_passage_breakdown_table_html(class_passage_breakdown, "クラス", "クラス別", passage_keys)}
      {_passage_breakdown_table_html(ground_state_passage_breakdown, "馬場状態", "馬場別", passage_keys)}
      <details class="breakdown">
        <summary>年度別を表示</summary>
        {_passage_breakdown_table_html(year_passage_breakdown, "年度", "年度別", passage_keys)}
      </details>
    </div>

    <div class="section-panel" data-section="chakudo" hidden>
      {frame_chakudo_chart}
      <details class="breakdown">
        <summary>枠番データ：クラス別を表示</summary>
        {frame_chakudo_class_html}
      </details>
      <details class="breakdown">
        <summary>枠番データ：馬場別を表示</summary>
        {frame_chakudo_ground_html}
      </details>
      {horse_chakudo_chart}
      <details class="breakdown">
        <summary>馬番データ：クラス別を表示</summary>
        {horse_chakudo_class_html}
      </details>
      <details class="breakdown">
        <summary>馬番データ：馬場別を表示</summary>
        {horse_chakudo_ground_html}
      </details>
      {pop_chakudo_chart}
      <details class="breakdown">
        <summary>人気データ：クラス別を表示</summary>
        {pop_chakudo_class_html}
      </details>
      <details class="breakdown">
        <summary>人気データ：馬場別を表示</summary>
        {pop_chakudo_ground_html}
      </details>
    </div>

    <div class="section-panel" data-section="weight" hidden>
      <p class="section-panel-intro">
        馬体重を{WEIGHT_BUCKET_SIZE}kg刻みの帯に分け、帯ごとの着度数（バー）と
        3着内率の傾向（折れ線）を表示します。バーは帯ごとの内訳の割合、
        折れ線は帯から帯への変化（傾向の形）を見るのに向いています。
      </p>
      <h3>馬体重帯別の3着内率の傾向（全体）</h3>
      {weight_trend_chart}
      {weight_chakudo_chart}
      <details class="breakdown">
        <summary>馬体重データ：クラス別を表示</summary>
        {weight_chakudo_class_html}
      </details>
      <details class="breakdown">
        <summary>馬体重データ：馬場別を表示</summary>
        {weight_chakudo_ground_html}
      </details>
    </div>

    <div class="section-panel" data-section="peds" hidden>
      {_peds_chart_html(peds_total_df, "TOTAL（上位10件）")}
      {_peds_breakdown_html(peds_class_breakdown, "class", "クラス別")}
      {_peds_breakdown_html(peds_ground_state_breakdown, "ground_state", "馬場別")}
      <details class="breakdown">
        <summary>年度別を表示</summary>
        {_peds_breakdown_html(peds_year_breakdown, "year", "年度別")}
      </details>
    </div>

    <div class="section-panel" data-section="venue-compare" hidden>
      <p class="section-panel-intro">
        同じ{race_type}{course_len}mを持つ他競馬場との比較です。馬場状態・クラスを
        選ぶと、その条件に絞り込んだ比較になります（いずれも「全て」を選ぶと全期間
        トータルの比較になります）。競馬場名をクリックすると、その競馬場の同条件
        ページに移動します。
      </p>
      {venue_comparison_html}
    </div>
  </div>

  {ad_unit_html(AD_SLOT_IN_CONTENT_1)}

  <p><a href="../../performance/course/{PLACE_LIST[place_id - 1]}/{race_type}-{course_len}.html">&larr; このコースのAI成績を見る</a></p>
  <p><a href="../../">&larr; HOMEへ戻る</a></p>
  {site_footer_html(base_path="../../")}
  <script src="../../assets/js/sortable-table.js"></script>
  <script src="../../assets/js/section-tabs.js"></script>
  <script src="../../assets/js/cross-filter.js"></script>
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

    breadcrumb_items = [("コース詳細データ", None)]

    # 右側タブでは、コース詳細データの直下に全競馬場を選択肢として表示し、
    # トップページからもどの競馬場へも直接遷移できるようにする。
    venue_siblings = [(NAME_LIST[i], f"courses/{PLACE_LIST[i]}/index.html") for i in range(len(PLACE_LIST))]
    tab_hierarchy_items = [("コース詳細データ", None), ("競馬場", None, venue_siblings)]

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_tags_html(
      "コース詳細データ",
      "全国10競馬場のコース別データ分析。平均勝ち時計・人気・血統傾向を競馬場・距離別に掲載。",
      f"{SITE_URL}/courses/",
  )}
  {adsense_script_html()}
  {ga4_script_html()}
  <title>コース詳細データ</title>
  <link rel="stylesheet" href="../assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="../assets/favicon.svg">
</head>
<body class="section-courses">
  {site_nav_html(base_path="../", breadcrumb_items=tab_hierarchy_items)}
  {breadcrumb_html(breadcrumb_items, base_path="../")}
  <h1>コース詳細データ</h1>

  <h2>開催中の競馬場</h2>
  <div class="course-tile-grid">
    {active_tiles if active_tiles else "<p>現在開催中の競馬場はありません。</p>"}
  </div>

  <h2>その他の競馬場</h2>
  <div class="course-tile-grid">
    {inactive_tiles}
  </div>

  <p><a href="../">&larr; HOMEへ戻る</a></p>
  {site_footer_html(base_path="../")}
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
            f'<tr><td><a href="{race_type}-{course_len}.html">{course_label_html(race_type, course_len)}</a></td>'
            f"<td>{_fmt_time(time_row)}</td><td>{_fmt(pop_row, 'avg_pop')}</td></tr>\n"
        )

    peds_sections = "\n  ".join(
        _peds_chart_html(
            aggregate_peds_by_race_type(place_id, race_type),
            f"{race_type_span_html(race_type)}（全距離合算・上位10件）",
        )
        for race_type in sorted({race_type for race_type, _ in course_list})
    )

    breadcrumb_items = [("コース詳細データ", "courses/index.html"), (place_name, None)]
    breadcrumb = breadcrumb_html(breadcrumb_items, base_path="../../")

    # 右側タブでは、横並びのbreadcrumbとは別に「全競馬場」「この競馬場の全コース」も
    # 兄弟項目として並べて表示し、どの競馬場・コースへもタブから直接遷移できるようにする
    # （横並びのbreadcrumbは現在地までの単純な経路のみで、兄弟一覧は出さない）。
    venue_siblings = [(NAME_LIST[i], f"courses/{PLACE_LIST[i]}/index.html") for i in range(len(PLACE_LIST))]
    venue_siblings[place_id - 1] = (place_name, None)
    course_siblings = [
        (course_label_html(rt, cl), f"courses/{PLACE_LIST[place_id - 1]}/{rt}-{cl}.html") for rt, cl in course_list
    ]
    tab_hierarchy_items = [
        ("コース詳細データ", "courses/index.html"),
        (place_name, None, venue_siblings),
        (f"{place_name}のコース", None, course_siblings),
    ]

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {meta_tags_html(
      f"{place_name} コース一覧",
      f"{place_name}競馬場の全コースの平均勝ち時計・平均人気・血統傾向データ一覧。",
      f"{SITE_URL}/courses/{PLACE_LIST[place_id - 1]}/",
  )}
  {adsense_script_html()}
  {ga4_script_html()}
  <title>{place_name} コース一覧</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="../../assets/favicon.svg">
</head>
<body class="section-courses">
  {site_nav_html(base_path="../../", breadcrumb_items=tab_hierarchy_items)}
  {breadcrumb}
  <h1>{place_name} コース一覧</h1>

  <div class="card-grid">
    <div class="card" style="flex-basis: 100%;">
      <h2>コース別データ</h2>
      <div class="table-wrap">
      <table class="sortable">
        <thead><tr><th>コース</th><th>平均勝ち時計</th><th>平均人気</th></tr></thead>
        <tbody>
          {rows}
        </tbody>
      </table>
      </div>
    </div>

    <div class="card" style="flex-basis: 100%;">
      <h2>血統別成績（芝/ダート別）</h2>
      {peds_sections}
    </div>
  </div>

  <p><a href="../../performance/course/{PLACE_LIST[place_id - 1]}/index.html">&larr; このコースのAI成績を見る</a></p>
  <p><a href="../index.html">&larr; コース詳細データ一覧へ</a></p>
  {site_footer_html(base_path="../../")}
  <script src="../../assets/js/sortable-table.js"></script>
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
