"""src/logic/html_generator/course_report_generator.py のテスト（オフライン）。

race_info_dataset_manager / peds_results_dataset_manager に既に集計済みの実データ
（data/race_info/, data/horse/peds_results/）を使って、コース詳細データページの
生成を検証する。
"""

from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.logic.html_generator import course_report_generator as c

SAMPLE_PLACE_ID = 5  # 05_tokyo
SAMPLE_RACE_TYPE = "芝"
SAMPLE_COURSE_LEN = "1400"


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_COURSES_PATH", str(tmp_path / "public_html" / "courses"))
    return tmp_path


def test_build_course_report_returns_real_data():
    report = c.build_course_report(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_course_report(place_id={SAMPLE_PLACE_ID}, {SAMPLE_RACE_TYPE}{SAMPLE_COURSE_LEN}m) ---")
    print(f"  avg_time: {report['avg_time']['avg_time']}")
    print(f"  avg_pop: {report['avg_pop']['avg_pop']}")
    print(f"  winner_weight: {report['winner_weight']['馬体重']}")
    print(f"  avg_frame/horse: {report['avg_frame_and_horse']['avg_frame']} / {report['avg_frame_and_horse']['avg_horse']}")
    print(f"  peds_df shape: {report['peds_df'].shape}")

    # 2026/6/20-21分のレース結果が追加されたため、値は以前の集計から更新されている
    assert report["avg_time"] is not None
    assert report["avg_time"]["avg_time"] == "81662"
    assert report["avg_pop"]["avg_pop"] == "4.2"
    assert report["winner_weight"]["馬体重"] == "467.2"
    assert report["avg_frame_and_horse"]["avg_frame"] == "4.93"
    assert not report["peds_df"].empty
    assert report["peds_df"].iloc[0]["血統"] == "ロードカナロア"


def test_build_course_report_returns_none_for_unknown_condition():
    report = c.build_course_report(SAMPLE_PLACE_ID, "芝", "9999")

    assert report["avg_time"] is None
    assert report["avg_pop"] is None
    assert report["winner_weight"] is None
    assert report["avg_frame_and_horse"] is None


def test_course_report_to_html_structure():
    report = c.build_course_report(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)
    html = c.course_report_to_html(report)

    print(f"\n--- course_report_to_html(東京, 芝1400m) ---")
    print(html)

    assert '<h1>東京 <span class="race-type-turf">芝1400m</span> コース詳細</h1>' in html
    # 走破時計は msec の生値ではなく「分:秒.コンマ」形式で表示する
    assert "1:21.7" in html
    assert "ロードカナロア" in html

    # 概要はタブではなくページ上部に表示される
    assert html.index("summary-stats") < html.index('<div class="tabbed-section">')
    assert '<button data-target="overview"' not in html

    # 馬場×クラス×年度がメインタブ（最初から表示）、他は参考データタブ（hidden）に分かれている
    assert '<div class="tabbed-section">' in html
    assert '<div class="section-tabs">' in html
    assert '<button class="tab-main" data-target="cross" aria-selected="true">馬場×クラス×年度（メイン）</button>' in html
    assert '<span class="section-tabs-sub-label">参考データ:</span>' in html
    for target, label in [
        ("passage", "通過順"),
        ("chakudo", "人気・枠順"),
        ("peds", "血統別成績"),
    ]:
        assert f'<button data-target="{target}" aria-selected="false">{label}</button>' in html
    # クラス別・馬場別・年度別の単体タブは廃止し、馬場×クラス×年度に統合した
    assert 'data-target="breakdown"' not in html
    # メインタブ（馬場×クラス×年度）は初期表示、参考データはhidden属性で初期非表示
    assert '<div class="section-panel" data-section="cross">' in html
    assert '<div class="section-panel" data-section="passage" hidden>' in html
    assert '<div class="section-panel" data-section="chakudo" hidden>' in html
    assert '<div class="section-panel" data-section="peds" hidden>' in html
    # 馬場×クラス×年度の絞り込みUI: 3つのセレクトと、組み合わせごとの事前計算パネル
    assert '<div class="cross-filter">' in html
    assert 'class="cross-filter-ground-state"' in html
    assert 'class="cross-filter-class"' in html
    assert 'class="cross-filter-year"' in html
    assert '<option value="良">良</option>' in html
    # 開始年は「全期間/直近3年/今年」の3パターンに絞る（組み合わせが増えすぎて
    # ページサイズが肥大化する（実測で1ページ最大13MB）のを防ぐため）
    assert '<option value="2019">全期間</option>' in html
    assert '<option value="2024">直近3年（2024年〜）</option>' in html
    assert '<option value="2026">今年（2026年〜）</option>' in html
    assert '<div class="cross-filter-panel" data-ground-state="全" data-class="all" data-year="2019" hidden>' in html
    assert '<div class="cross-filter-panel" data-ground-state="良" data-class="all" data-year="2019" hidden>' in html
    assert '<div class="cross-filter-panel" data-ground-state="全" data-class="未勝利" data-year="2019" hidden>' in html
    # どの組み合わせを見ているかパネルの見出しで分かる（開始年=最古年は「全期間」と表示する）
    assert "<h3>全 × all × 全期間</h3>" in html
    # 概要は表ではなく、ページ上部と同じ大きな数字のカード（summary-stats）で表示する
    # （データの羅列に見えないようにする）
    assert html.count('<div class="summary-stats">') >= 2
    # 平均勝ち時計は最重要指標として大きく(summary-stat-primary)、それ以外は
    # サブ(summary-stat-secondary)として一回り小さく表示する
    assert '<div class="summary-stat summary-stat-primary">' in html
    assert '<div class="summary-stat summary-stat-secondary">' in html
    # 各組み合わせのパネルには、平均成績のカードに加えて血統→枠番別→馬番別→
    # 馬体重別→人気別の順で折りたたみが表示される
    cross_idx = html.index('<div class="section-panel" data-section="cross">')
    peds_summary_idx = html.index("<summary>血統データを表示</summary>", cross_idx)
    frame_summary_idx = html.index("<summary>枠番データを表示</summary>", cross_idx)
    horse_summary_idx = html.index("<summary>馬番データを表示</summary>", cross_idx)
    weight_summary_idx = html.index("<summary>馬体重データを表示</summary>", cross_idx)
    pop_summary_idx = html.index("<summary>人気データを表示</summary>", cross_idx)
    assert peds_summary_idx < frame_summary_idx < horse_summary_idx < weight_summary_idx < pop_summary_idx
    assert "<h4>人気別着度数</h4>" in html
    assert "<h4>枠番別着度数</h4>" in html
    assert "<h4>馬番別着度数</h4>" in html
    assert "<h4>馬体重帯別着度数</h4>" in html
    assert "<h4>血統別成績（上位5件）</h4>" in html
    # 枠番別チャートには、出馬表ページと同じ枠色のCSSクラスが付く
    assert 'class="chakudo-label waku-1"' in html
    assert 'class="chakudo-label waku-8"' in html

    # 平均配当（単勝・複勝、勝ち馬のオッズ×100円／勝ち馬の複勝配当の平均）が
    # サマリー・カードに追加されている
    assert "平均配当（単勝）" in html
    assert "平均配当（複勝）" in html
    assert "円" in html
    # 通過順データが追加されている。東京芝1400mは通過1・2のみ記録されているコースのため、
    # 存在しない通過3・4は「データなし」で埋めず、列ごと出さない
    assert "上り(勝ち馬)" in html
    assert "<th>通過1</th>" in html
    assert "<th>通過2</th>" in html
    assert "<th>通過3</th>" not in html
    assert "<th>通過4</th>" not in html
    # 人気データ・枠順データは表ではなく1着/2着/3着/着外の積み上げ横バーチャートで表示し、
    # 出走自体が無いランクは省略するが「11以降」のようにまとめない
    assert "<h3>人気データ（人気別着度数、全体）</h3>" in html
    assert "<h3>枠順データ（枠番別着度数、全体）</h3>" in html
    assert "<h3>枠順データ（馬番別着度数、全体）</h3>" in html
    assert '<div class="chakudo-chart">' in html
    assert '<span class="chakudo-segment seg-1st"' in html
    assert "以降" not in html
    # 人気・枠順データにも、クラス別・馬場別の内訳が折りたたみで追加されている
    assert "<summary>人気データ：クラス別を表示</summary>" in html
    assert "<summary>人気データ：馬場別を表示</summary>" in html
    assert "<summary>枠番データ：クラス別を表示</summary>" in html
    assert "<summary>枠番データ：馬場別を表示</summary>" in html
    assert "<summary>馬番データ：クラス別を表示</summary>" in html
    assert "<summary>馬番データ：馬場別を表示</summary>" in html
    # 血統別成績はTOTAL（既存）に加え、クラス別・馬場別・年度別の内訳も追加されている
    # （表ではなく着度数と同じ積み上げ横バーチャートで表示する）
    assert "<h3>TOTAL（上位10件）</h3>" in html
    assert "<h3>クラス別</h3>" in html
    assert "<h3>馬場別</h3>" in html
    assert "<h3>年度別</h3>" in html
    assert "<h4>未勝利" in html
    assert "<h4>良" in html
    assert 'class="chakudo-label peds-label"' in html
    # 馬体重別成績タブ：10kg刻みの帯別着度数（バー）と3着内率の折れ線傾向グラフがある
    assert '<button data-target="weight" aria-selected="false">馬体重別成績</button>' in html
    assert '<div class="section-panel" data-section="weight" hidden>' in html
    assert "<h3>馬体重データ（馬体重帯別着度数、全体）</h3>" in html
    assert "kg台" in html
    # サンプル数が極端に少ない両端（390kg未満・550kg以上）は10kg刻みのままにせず、
    # 1つの開放区間の帯にまとめて表示する
    assert "390kg未満" in html
    assert "550kg以上" in html
    assert "380kg台" not in html
    assert "560kg台" not in html
    assert '<polyline points=' in html
    assert "<summary>馬体重データ：クラス別を表示</summary>" in html
    assert "<summary>馬体重データ：馬場別を表示</summary>" in html
    # 個別コースのAI成績ページへの相互リンクが追加されている
    assert '<a href="../../performance/course/05_tokyo/芝-1400.html">&larr; このコースのAI成績を見る</a>' in html
    assert '<a href="../../">&larr; HOMEへ戻る</a>' in html
    # サイト共通ナビゲーション・列ソートJS・タブJS・年度別の折りたたみが追加されている
    assert '<nav class="site-nav">' in html
    assert "pagead2.googlesyndication.com" in html
    assert 'rel="icon"' in html
    assert "googletagmanager.com/gtag/js?id=G-DNC949064T" in html
    assert '<a href="../../performance/index.html">AI成績</a>' in html
    assert '<script src="../../assets/js/sortable-table.js"></script>' in html
    assert '<script src="../../assets/js/section-tabs.js"></script>' in html
    assert '<script src="../../assets/js/cross-filter.js"></script>' in html
    # 通過順・peds各1つの年度別折りたたみ + 人気/枠番/馬番のクラス別・馬場別の折りたたみ6つ
    # + 馬体重のクラス別・馬場別の折りたたみ2つ
    # + 馬場×クラス×年度の組み合わせごとに5つ（血統/枠番/馬番/馬体重/人気データ）の折りたたみ
    # （メインタブの5つはデフォルトで展開済み(open属性)、それ以外は閉じた状態）
    cross_combo_count = len(c.build_cross_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN))
    assert html.count('<details class="breakdown">') == 10
    assert html.count('<details class="breakdown" open>') == cross_combo_count * 5
    assert html.count("<summary>年度別を表示</summary>") == 2
    # ブレッドクラム（現在地の階層）が追加されている
    assert '<p class="breadcrumb">' in html
    assert '<a href="../../">HOME</a>' in html
    assert '<a href="../../courses/index.html">コース詳細データ</a>' in html
    assert '<a href="../../courses/05_tokyo/index.html">東京</a>' in html
    assert '<span class="breadcrumb-current"><span class="race-type-turf">芝1400m</span></span>' in html
    # 旧来の右サイドバー（競馬場・コース選択）は、右側タブの階層表示と重複するため廃止した
    assert '<aside class="page-sidebar">' not in html
    # 右側タブに同じ階層（他の競馬場・このコースの他の距離）が表示される
    assert '<a href="../../courses/06_nakayama/index.html">中山</a>' in html
    assert '<a href="../../courses/05_tokyo/芝-1600.html"><span class="race-type-turf">芝1600m</span></a>' in html
    assert '<span class="page-calendar-tab-current"><span class="race-type-turf">芝1400m</span></span>' in html


def test_build_class_passage_breakdown_returns_per_class_rows():
    rows = c.build_class_passage_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_class_passage_breakdown(東京, 芝1400m) ---")
    for r in rows:
        print(f"  {r}")

    values = {r["value"] for r in rows}
    assert "未勝利" in values
    assert all(r["agari"].endswith("秒") for r in rows)


def test_build_ground_state_passage_breakdown_returns_ordered_rows():
    rows = c.build_ground_state_passage_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)
    assert [r["value"] for r in rows] == ["良", "稍重", "重", "不良"]


def test_build_year_passage_breakdown_returns_years_newest_first():
    rows = c.build_year_passage_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)
    years = [r["value"] for r in rows]
    assert years == sorted(years, reverse=True)


def test_available_passage_keys_omits_missing_checkpoints_for_short_course():
    # 東京芝1400mは通過1・2のみ記録されている短距離コース
    keys = c.available_passage_keys(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)
    assert keys == ["passage1", "passage2"]


def test_available_passage_keys_includes_all_checkpoints_for_long_course():
    # 東京芝2400mは通過1〜4まで記録されている長距離コース
    keys = c.available_passage_keys(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, "2400")
    assert keys == ["passage1", "passage2", "passage3", "passage4"]


def test_passage_breakdown_table_html_omits_missing_passage_columns():
    rows = [{"value": "未勝利", "agari": "34.0秒", "passage1": "6.0", "passage2": "5.0", "passage3": "データなし", "passage4": "データなし"}]

    html = c._passage_breakdown_table_html(rows, "クラス", "クラス別", ["passage1", "passage2"])

    assert "<th>通過1</th>" in html
    assert "<th>通過2</th>" in html
    assert "<th>通過3</th>" not in html
    assert "<td>34.0秒</td><td>6.0</td><td>5.0</td>" in html


def test_build_peds_ground_state_breakdown_returns_ordered_rows():
    breakdown = c.build_peds_ground_state_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_peds_ground_state_breakdown(東京, 芝1400m) ---")
    for item in breakdown:
        print(f"  {item['ground_state']}: {item['peds_df']['血統'].tolist()}")

    ground_states = [item["ground_state"] for item in breakdown]
    assert ground_states == [g for g in c.GROUND_STATE_ORDER if g in ground_states]
    assert "良" in ground_states


def test_chakudo_chart_html_shows_only_ranks_with_data_as_stacked_bars():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "芝", "芝"],
            "course_len": ["1400", "1400", "1400"],
            "ground_state": ["全", "全", "全"],
            "class": ["all", "all", "all"],
            "人気": [1, 2, 11],
            "1着": [10, 3, 0],
            "2着": [4, 5, 1],
            "3着": [2, 2, 0],
            "着外": [1, 6, 8],
        }
    )

    html = c._chakudo_chart_html(df, "芝", "1400", "人気", range(1, 12), "人気データ")

    print(f"\n--- _chakudo_chart_html(積み上げバー表示) ---\n{html}")

    # 1着/2着/3着/着外を内訳とした積み上げバー（合計100%）になっている
    # ツールチップは件数ではなく割合（2着・3着は累積割合も）を示すカスタムツールチップ
    # （ネイティブのtitle属性ではなく、CSSで大きく表示する.chakudo-tooltip）
    assert '<span class="chakudo-label">1</span>' in html
    assert (
        '<span class="chakudo-segment seg-1st" style="width: 58.82%">10'
        '<span class="chakudo-tooltip">1着: 58.8%</span></span>' in html
    )
    assert (
        '<span class="chakudo-segment seg-2nd" style="width: 23.53%">4'
        '<span class="chakudo-tooltip">2着: 23.5%（累積82.4%）</span></span>' in html
    )
    assert (
        '<span class="chakudo-segment seg-3rd" style="width: 11.76%">2'
        '<span class="chakudo-tooltip">3着: 11.8%（累積94.1%）</span></span>' in html
    )
    # 1人気は着内率(94.1%)が他より明確に高いため「有利」バッジが付く。バッジは
    # バーより前（固定幅の枠の中）に表示され、バーの開始位置はズレない
    assert html.index('<span class="advantage-badge high">◎ 有利</span>') < html.index('chakudo-bar-track')
    assert '<span class="chakudo-badge-slot"><span class="advantage-badge high">◎ 有利</span></span>' in html
    # 件数（n=17）も表示される
    assert '<span class="chakudo-value">n=17</span>' in html
    # 出走自体が無い順位（3〜10）は行ごと表示しない（「データなし」とは書かない）
    assert '<span class="chakudo-label">3</span>' not in html
    assert "データなし" not in html
    # 11位は著しく低いため「不利」バッジが付く。「11以降」のようにまとめない
    assert '<span class="chakudo-label">11</span>' in html
    assert '<span class="advantage-badge low">▲ 不利</span>' in html
    assert "以降" not in html
    # 表ではなく積み上げ横バーチャートになっている
    assert "<table" not in html
    assert '<div class="chakudo-chart">' in html


def test_chakudo_chart_html_always_shows_segment_numbers_even_when_small():
    df = pd.DataFrame(
        {
            "race_type": ["芝"], "course_len": ["1400"], "ground_state": ["全"], "class": ["all"],
            "人気": [1], "1着": [90], "2着": [5], "3着": [2], "着外": [3],
        }
    )

    html = c._chakudo_chart_html(df, "芝", "1400", "人気", range(1, 2), "人気データ")

    # 幅が小さい（2%の）セグメントでも数値を省略しない
    assert (
        '<span class="chakudo-segment seg-3rd" style="width: 2.00%">2'
        '<span class="chakudo-tooltip">3着: 2.0%（累積97.0%）</span></span>' in html
    )
    assert (
        '<span class="chakudo-segment seg-1st" style="width: 90.00%">90'
        '<span class="chakudo-tooltip">1着: 90.0%</span></span>' in html
    )


def test_chakudo_chart_html_reserves_badge_slot_even_without_badge():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "芝"], "course_len": ["1400", "1400"], "ground_state": ["全", "全"], "class": ["all", "all"],
            "人気": [1, 2], "1着": [50, 49], "2着": [10, 10], "3着": [5, 5], "着外": [5, 6],
        }
    )

    html = c._chakudo_chart_html(df, "芝", "1400", "人気", range(1, 3), "人気データ", show_advantage=False)

    # バッジが無い場合でも.chakudo-badge-slot自体は出力され、バーの開始位置を揃える
    assert html.count('<span class="chakudo-badge-slot"></span>') == 2
    assert "advantage-badge" not in html


def test_chakudo_chart_html_disables_advantage_when_show_advantage_false():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "芝"], "course_len": ["1400", "1400"], "ground_state": ["全", "全"], "class": ["all", "all"],
            "人気": [1, 2], "1着": [50, 1], "2着": [10, 1], "3着": [5, 1], "着外": [5, 47],
        }
    )

    html = c._chakudo_chart_html(df, "芝", "1400", "人気", range(1, 3), "人気データ", show_advantage=False)

    assert "advantage-badge" not in html


def test_chakudo_chart_html_handles_missing_data():
    html = c._chakudo_chart_html(pd.DataFrame(), "芝", "1400", "人気", range(1, 3), "人気データ")
    assert "対象データがありません。" in html


def test_chakudo_chart_html_handles_all_ranks_without_data():
    df = pd.DataFrame(
        {
            "race_type": ["芝"], "course_len": ["1400"], "ground_state": ["全"], "class": ["all"],
            "人気": [5], "1着": [0], "2着": [0], "3着": [0], "着外": [0],
        }
    )
    html = c._chakudo_chart_html(df, "芝", "1400", "人気", range(1, 3), "人気データ")
    assert "対象データがありません。" in html


def test_chakudo_class_breakdown_html_returns_per_class_charts():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "芝"], "course_len": ["1400", "1400"], "ground_state": ["全", "全"],
            "class": ["未勝利", "1勝クラス"], "人気": [1, 1], "1着": [10, 8], "2着": [4, 3], "3着": [2, 2], "着外": [4, 7],
        }
    )

    html = c._chakudo_class_breakdown_html(df, "芝", "1400", "人気", range(1, 19))

    assert "<h4>未勝利</h4>" in html
    assert "<h4>1勝クラス</h4>" in html
    # "all"（全クラス合算）は内訳の対象外
    assert "<h4>all</h4>" not in html


def test_chakudo_ground_state_breakdown_html_returns_ordered_charts():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "芝", "芝"], "course_len": ["1400", "1400", "1400"],
            "ground_state": ["不良", "良", "稍重"], "class": ["all", "all", "all"],
            "人気": [1, 1, 1], "1着": [3, 10, 5], "2着": [1, 4, 2], "3着": [1, 2, 1], "着外": [2, 4, 3],
        }
    )

    html = c._chakudo_ground_state_breakdown_html(df, "芝", "1400", "人気", range(1, 19))

    # 良→稍重→重→不良の順（このサンプルには「重」が無いため省略される）
    assert html.index("<h4>良</h4>") < html.index("<h4>稍重</h4>") < html.index("<h4>不良</h4>")
    assert "<h4>重</h4>" not in html


def test_chakudo_class_breakdown_html_handles_no_data():
    html = c._chakudo_class_breakdown_html(pd.DataFrame(), "芝", "1400", "人気", range(1, 19))
    assert "対象データがありません。" in html


def _row(rank, hit1, hit2, hit3, out):
    total = hit1 + hit2 + hit3 + out
    top3_rate = (hit1 + hit2 + hit3) / total * 100 if total else None
    return {"rank": rank, "1着": hit1, "2着": hit2, "3着": hit3, "着外": out, "total": total, "top3_rate": top3_rate}


def test_label_advantage_marks_nothing_when_flat():
    # 全ランクの着内率がほぼ同じ（フラット）場合は何もラベル付けしない
    rows = [_row(1, 30, 30, 30, 60), _row(2, 28, 32, 30, 60), _row(3, 32, 28, 32, 58)]

    result = c._label_advantage(rows)

    assert all(r["note_text"] == "" and r["note_kind"] == "" for r in result)


def test_label_advantage_marks_high_and_low_when_notably_different():
    rows = [_row(1, 50, 10, 5, 5), _row(2, 25, 10, 10, 25), _row(3, 1, 1, 1, 47)]

    result = c._label_advantage(rows)

    assert result[0]["note_text"] == "◎ 有利"
    assert result[0]["note_kind"] == "high"
    assert result[2]["note_text"] == "▲ 不利"
    assert result[2]["note_kind"] == "low"


def test_label_advantage_excludes_ranks_and_suppresses_low_label():
    # 1〜3番人気は除外し、不利ラベルは出さない設定（人気データ向けの使い方）
    rows = [_row(1, 50, 10, 5, 5), _row(2, 25, 10, 10, 25), _row(3, 1, 1, 1, 47), _row(4, 20, 15, 10, 5)]

    result = c._label_advantage(rows, exclude_ranks={1, 2, 3}, high_label="★ ねらい目", low_label="")

    assert result[0]["note_text"] == ""
    assert result[2]["note_text"] == ""
    # 4番人気は対象（1〜3を除いた候補は4のみ）だが、候補が1件のためラベルなし
    assert result[3]["note_text"] == ""


def test_advantage_badge_html_renders_span_or_empty():
    assert c._advantage_badge_html("◎ 有利", "high") == '<span class="advantage-badge high">◎ 有利</span>'
    assert c._advantage_badge_html("", "") == ""


def _breakdown_row_stub(value, win_return_raw):
    return {
        "value": value,
        "avg_time": "1:21.0",
        "avg_pop": "4.0",
        "weight": "460.0kg",
        "avg_frame": "4.5",
        "avg_horse": "8.0",
        "win_return": f"{win_return_raw}円" if win_return_raw is not None else "データなし",
        "win_return_raw": win_return_raw,
    }


def test_breakdown_table_html_highlights_notably_high_and_low_return():
    rows = [
        _breakdown_row_stub("未勝利", 150.0),
        _breakdown_row_stub("1勝クラス", 160.0),
        _breakdown_row_stub("オープン", 400.0),
    ]

    html = c._breakdown_table_html(rows, "クラス", "クラス別")

    print(f"\n--- _breakdown_table_html(平均配当の偏差) ---\n{html}")

    assert '<th>傾向</th>' in html
    assert '<span class="advantage-badge high">◎ 注目（高配当）</span>' in html
    # 未勝利・1勝クラスは差が小さいため、傾向は付かない
    assert html.count('<span class="advantage-badge') == 1
    assert '<div class="table-wrap">' in html
    assert '<table class="sortable">' in html


def test_breakdown_table_html_handles_empty_rows():
    html = c._breakdown_table_html([], "クラス", "クラス別")
    assert "対象データがありません。" in html


def test_build_peds_class_breakdown_excludes_all_class():
    peds_df = pd.DataFrame(
        {
            "クラス": ["all", "all", "未勝利", "未勝利", "1勝クラス"],
            "血統": ["A", "B", "C", "D", "E"],
            "1着": [10, 8, 5, 3, 4],
            "2着": [1, 1, 1, 1, 1],
            "3着": [1, 1, 1, 1, 1],
            "着外": [1, 1, 1, 1, 1],
        }
    )

    breakdown = c.build_peds_class_breakdown(peds_df, top_n=5)

    print(f"\n--- build_peds_class_breakdown ---")
    for item in breakdown:
        print(f"  {item['class']}: {item['peds_df']['血統'].tolist()}")

    classes = [item["class"] for item in breakdown]
    assert "all" not in classes
    assert "未勝利" in classes
    assert "1勝クラス" in classes


def test_build_peds_year_breakdown_returns_years_newest_first():
    breakdown = c.build_peds_year_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_peds_year_breakdown(東京, 芝1400m) ---")
    for item in breakdown:
        print(f"  {item['year']}: {item['peds_df']['血統'].tolist()}")

    years = [item["year"] for item in breakdown]
    assert years == sorted(years, reverse=True)


def test_build_class_breakdown_returns_per_class_rows():
    rows = c.build_class_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_class_breakdown(東京, 芝1400m) ---")
    for r in rows:
        print(f"  {r}")

    values = {r["value"] for r in rows}
    assert "未勝利" in values
    assert "all" not in values


def test_build_ground_state_breakdown_returns_ordered_rows():
    rows = c.build_ground_state_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_ground_state_breakdown(東京, 芝1400m) ---")
    for r in rows:
        print(f"  {r}")

    assert [r["value"] for r in rows] == ["良", "稍重", "重", "不良"]
    assert "全" not in {r["value"] for r in rows}


def test_build_cross_breakdown_returns_rows_keyed_by_ground_state_class_and_start_year():
    oldest_year = c.ANNUAL_START_YEAR
    cross = c.build_cross_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_cross_breakdown(東京, 芝1400m) ---")
    for key, row in cross.items():
        print(f"  {key}: {row}")

    assert cross  # 実データなので少なくとも1組み合わせは存在するはず
    # 各軸の「全て」（馬場状態="全"・クラス="all"）と、開始年=oldest_year（全期間）を
    # 含む全組み合わせが入っている。開始年は常にint（"開始年〜最新年"の範囲集計）
    for ground_state, class_name, start_year in cross:
        assert ground_state in ["全"] + c.GROUND_STATE_ORDER
        assert class_name == "all" or class_name != ""
        assert isinstance(start_year, int)
    assert ("全", "all", oldest_year) in cross  # 全体合計（全期間）
    assert any(gs != "全" and cls == "all" and y == oldest_year for gs, cls, y in cross)  # 馬場のみ
    assert any(gs == "全" and cls != "all" and y == oldest_year for gs, cls, y in cross)  # クラスのみ
    assert any(gs == "全" and cls == "all" and y != oldest_year for gs, cls, y in cross)  # 開始年のみ
    assert any(gs != "全" and cls != "all" and y != oldest_year for gs, cls, y in cross)  # 完全な組み合わせ
    # データが存在しない組み合わせ（実際の出走が無い期間）は除外されている
    for row in cross.values():
        assert row["avg_time"] != "データなし"
    # 各行は build_class_breakdown/build_ground_state_breakdown と同じ統計フィールドを持つ
    sample_row = next(iter(cross.values()))
    assert {"avg_time", "avg_pop", "weight", "avg_frame", "avg_horse", "win_return"} <= sample_row.keys()


def test_build_cross_breakdown_start_year_narrows_to_recent_races_only():
    oldest_year = c.ANNUAL_START_YEAR
    current_year = date.today().year
    cross = c.build_cross_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, current_year=current_year)

    full_period = cross.get(("全", "all", oldest_year))
    latest_year_only = cross.get(("全", "all", current_year))

    print(f"\n--- 全期間 vs {current_year}年〜 ---")
    print(f"  全期間: {full_period}")
    print(f"  {current_year}年〜: {latest_year_only}")

    assert full_period is not None
    # 開始年を最新年にすると対象が狭まるため、平均配当（数値）が全期間とは異なりうる
    # （少なくとも両方データが取れることを確認する）
    if latest_year_only is not None:
        assert latest_year_only["win_return_raw"] is not None


def test_peds_table_for_combo_returns_top_n_sorted_by_first_place():
    df = c._peds_table_for_combo(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, "全", "未勝利", top_n=5)

    print(f"\n--- _peds_table_for_combo(東京, 芝1400m, 全, 未勝利) ---")
    print(df)

    assert df is not None
    assert len(df) <= 5
    counts = pd.to_numeric(df["1着"], errors="coerce").tolist()
    assert counts == sorted(counts, reverse=True)


def test_peds_table_for_combo_translates_total_ground_state_sentinel():
    # get_total_peds_results_csvは全体合計を"全"ではなく"all"で表すため、
    # 馬場状態="全"を渡しても正しく変換されて結果が取れることを確認する
    total_df = c._peds_table_for_combo(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, "全", "all")
    direct_df = c.peds_results_dataset_manager.get_total_peds_results_csv(
        SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, "all"
    )
    assert total_df is not None
    assert not direct_df.empty


def test_peds_table_for_combo_returns_none_for_unknown_combination():
    assert c._peds_table_for_combo(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, "不良", "存在しないクラス") is None


def test_peds_table_for_combo_filters_by_year_when_given():
    total_df = c._peds_table_for_combo(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, "全", "all")
    year_df = c._peds_table_for_combo(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, "全", "all", year=2026)

    print(f"\n--- _peds_table_for_combo(東京, 芝1400m, 全, all, year=2026) ---\n{year_df}")

    assert year_df is not None
    # 年度別ファイルはTotalファイルとは別物のため、内容が異なりうる（少なくとも両方取得できる）
    assert total_df is not None


def test_peds_chart_html_shows_stacked_bar_with_total_count():
    df = c._peds_table_for_combo(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN, "全", "未勝利", top_n=3)

    html = c._peds_chart_html(df, "血統別成績（上位3件）", heading_level="h4")

    print(f"\n--- _peds_chart_html ---\n{html}")

    assert "<h4>血統別成績（上位3件）</h4>" in html
    assert '<div class="chakudo-chart">' in html
    assert '<span class="chakudo-label peds-label">' in html
    assert '<span class="chakudo-segment seg-1st"' in html
    # 血統ごとに総戦数が異なるため、割合（幅）に加えてn=総数を表示する
    assert "n=" in html


def test_peds_chart_html_handles_empty_data():
    html = c._peds_chart_html(None, "TOTAL（上位10件）")
    assert "<h3>TOTAL（上位10件）</h3>" in html
    assert "対象データがありません。" in html


def test_build_year_breakdown_returns_years_newest_first():
    rows = c.build_year_breakdown(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    print(f"\n--- build_year_breakdown(東京, 芝1400m) ---")
    for r in rows:
        print(f"  {r}")

    years = [r["value"] for r in rows]
    assert years == sorted(years, reverse=True)
    assert all(2019 <= y <= date.today().year for y in years)


def test_format_time_converts_msec_to_minutes_seconds():
    assert c._format_time("81655") == "1:21.7"
    assert c._format_time("59999") == "0:60.0"
    assert c._format_time(None) == "データなし"
    assert c._format_time(float("nan")) == "データなし"


def test_aggregate_peds_by_race_type_combines_all_distances():
    df = c.aggregate_peds_by_race_type(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE)

    print(f"\n--- aggregate_peds_by_race_type(東京, 芝) ---")
    print(df.head(5).to_string())

    assert not df.empty
    assert list(df.columns) == ["血統", "1着", "2着", "3着", "着外"]
    # 1着数が降順に並んでいる
    assert df["1着"].is_monotonic_decreasing


def test_make_course_index_page_generates_html(new_roots, monkeypatch):
    from datetime import date

    monkeypatch.setattr(
        c.calc, "get_current_meetings",
        lambda: [{"place_id": 5, "times": 3, "first_day": date(2026, 6, 6), "last_day": date(2026, 6, 21)}],
    )

    c.make_course_index_page()

    out_file = new_roots / "public_html" / "courses" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_course_index_page() ---")
    print(html_content)

    assert "<h1>コース詳細データ</h1>" in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<a href="../performance/index.html">AI成績</a>' in html_content
    # トップ階層はブレッドクラムのみ（サイドバーは追加しない）
    assert '<p class="breadcrumb">' in html_content
    assert '<span class="breadcrumb-current">コース詳細データ</span>' in html_content
    assert '<aside class="page-sidebar">' not in html_content
    # 右側タブには、コース詳細データの直下に全競馬場が選択肢として表示される
    assert '<a href="../courses/05_tokyo/index.html">東京</a>' in html_content
    assert '<a href="../courses/06_nakayama/index.html">中山</a>' in html_content
    # 開催中（東京）は大きいタイル
    assert '<div class="course-tile active">' in html_content
    assert '<a href="05_tokyo/index.html"><span class="place-name">東京</span></a>' in html_content
    assert "3回 開催中（06/06〜06/21）" in html_content
    # 非開催（例: 中山）は小さいリンク
    assert '<div class="course-tile inactive"><a href="06_nakayama/index.html">中山</a></div>' in html_content


def test_make_track_page_generates_html(new_roots):
    c.make_track_page(SAMPLE_PLACE_ID)

    out_file = new_roots / "public_html" / "courses" / "05_tokyo" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_track_page(東京) ---")
    print(html_content)

    # コース詳細データのページ全体に専用のサブカラー（section-courses）を適用する
    assert '<body class="section-courses">' in html_content
    assert "<h1>東京 コース一覧</h1>" in html_content
    assert '<a href="芝-1400.html"><span class="race-type-turf">芝1400m</span></a>' in html_content
    # 単なるリンクではなく、平均勝ち時計・人気が一覧表で見える
    assert "1:21.7" in html_content
    # 芝/ダート別に全距離合算した血統別成績が表示される（見出しも芝/ダートで色分けする）
    assert '<h3><span class="race-type-turf">芝</span>（全距離合算・上位10件）</h3>' in html_content
    assert '<h3><span class="race-type-dirt">ダート</span>（全距離合算・上位10件）</h3>' in html_content
    assert '<a href="../../performance/course/05_tokyo/index.html">&larr; このコースのAI成績を見る</a>' in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<div class="table-wrap">' in html_content
    assert '<table class="sortable">' in html_content
    assert '<script src="../../assets/js/sortable-table.js"></script>' in html_content
    # ブレッドクラム（現在地）が追加されている
    assert '<p class="breadcrumb">' in html_content
    assert '<a href="../../courses/index.html">コース詳細データ</a>' in html_content
    assert '<span class="breadcrumb-current">東京</span>' in html_content
    # 旧来の右サイドバー（他の競馬場一覧）は、右側タブの階層表示と重複するため廃止した
    assert '<aside class="page-sidebar">' not in html_content
    # 右側タブに同じ階層（他の競馬場・東京の全コース）が表示される
    assert '<a href="../../courses/06_nakayama/index.html">中山</a>' in html_content
    assert '<span class="page-calendar-tab-current">東京</span>' in html_content
    assert '<a href="../../courses/05_tokyo/芝-1600.html"><span class="race-type-turf">芝1600m</span></a>' in html_content


def test_make_course_detail_page_generates_html(new_roots):
    c.make_course_detail_page(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    out_file = new_roots / "public_html" / "courses" / "05_tokyo" / "芝-1400.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "コース詳細" in html_content
