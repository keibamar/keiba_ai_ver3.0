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

    assert report["avg_time"] is not None
    assert report["avg_time"]["avg_time"] == "81655"
    assert report["avg_pop"]["avg_pop"] == "4.25"
    assert report["winner_weight"]["馬体重"] == "467.2"
    assert report["avg_frame_and_horse"]["avg_frame"] == "4.92"
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

    assert "<h1>東京 芝1400m コース詳細</h1>" in html
    # 走破時計は msec の生値ではなく「分:秒.コンマ」形式で表示する
    assert "1:21.7" in html
    assert "ロードカナロア" in html
    # クラス別・馬場別・年度別の内訳テーブルが追加されている
    assert "<h3>クラス別</h3>" in html
    assert "<h3>馬場別</h3>" in html
    assert "<h3>年度別</h3>" in html
    # 平均配当（単勝、勝ち馬のオッズ×100円）がサマリー・内訳テーブルに追加されている
    assert "平均配当（単勝）" in html
    assert "平均配当(単勝)" in html
    assert "円" in html
    # 通過順（クラス別・馬場別・年度別）が追加されている
    assert "<h2>通過順（クラス別・馬場別・年度別）</h2>" in html
    assert "上り(勝ち馬)" in html
    # 人気データ・枠順データは着度数（1着/2着/3着/着外）で、馬番は11以降にまとめず全件表示する
    assert "<h3>人気データ（人気別着度数、全体）</h3>" in html
    assert "<h3>枠順データ（枠番別着度数、全体）</h3>" in html
    assert "<h3>枠順データ（馬番別着度数、全体）</h3>" in html
    assert "<th>1着</th><th>2着</th><th>3着</th><th>着外</th><th>傾向</th>" in html
    assert "<tr><td>18</td>" in html
    assert "以降" not in html
    # 血統別成績はTOTAL（既存）に加え、クラス別・年度別の内訳も追加されている
    assert "<h3>TOTAL（上位10件）</h3>" in html
    assert "<h3>クラス別</h3>" in html
    assert "<h3>年度別</h3>" in html
    assert "<h4>未勝利" in html
    # 個別コースのAI成績ページへの相互リンクが追加されている
    assert '<a href="../../performance/course/05_tokyo/芝-1400.html">&larr; このコースのAI成績を見る</a>' in html
    assert '<a href="../../index.html">&larr; HOMEへ戻る</a>' in html


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


def test_chakudo_table_html_shows_all_ranks_without_grouping():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "芝", "芝"],
            "course_len": ["1400", "1400", "1400"],
            "人気": [1, 2, 11],
            "1着": [10, 3, 0],
            "2着": [4, 5, 1],
            "3着": [2, 2, 0],
            "着外": [1, 6, 8],
        }
    )

    html = c._chakudo_table_html(df, "芝", "1400", "人気", range(1, 12), "人気データ")

    print(f"\n--- _chakudo_table_html(全件表示) ---\n{html}")

    # 1人気は着内率(94.1%)が他より明確に高いため「有利」、11人気は著しく低いため「不利」
    assert "<tr><td>1</td><td>10</td><td>4</td><td>2</td><td>1</td><td>◎ 有利</td></tr>" in html
    # 2人気は他との差が小さいため、傾向は付かない
    assert "<tr><td>2</td><td>3</td><td>5</td><td>2</td><td>6</td><td></td></tr>" in html
    # データがない順位（3〜10）も省略せず0で表示する
    assert "<tr><td>3</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td></tr>" in html
    # 11位も「11以降」のようにまとめず、そのまま表示する
    assert "<tr><td>11</td><td>0</td><td>1</td><td>0</td><td>8</td><td>▲ 不利</td></tr>" in html
    assert "以降" not in html


def test_chakudo_table_html_handles_missing_data():
    html = c._chakudo_table_html(pd.DataFrame(), "芝", "1400", "人気", range(1, 3), "人気データ")
    assert "対象データがありません。" in html


def _row(rank, hit1, hit2, hit3, out):
    total = hit1 + hit2 + hit3 + out
    top3_rate = (hit1 + hit2 + hit3) / total * 100 if total else None
    return {"rank": rank, "1着": hit1, "2着": hit2, "3着": hit3, "着外": out, "total": total, "top3_rate": top3_rate}


def test_label_advantage_marks_nothing_when_flat():
    # 全ランクの着内率がほぼ同じ（フラット）場合は何もラベル付けしない
    rows = [_row(1, 30, 30, 30, 60), _row(2, 28, 32, 30, 60), _row(3, 32, 28, 32, 58)]

    result = c._label_advantage(rows)

    assert all(r["note"] == "" for r in result)


def test_label_advantage_marks_high_and_low_when_notably_different():
    rows = [_row(1, 50, 10, 5, 5), _row(2, 25, 10, 10, 25), _row(3, 1, 1, 1, 47)]

    result = c._label_advantage(rows)

    assert result[0]["note"] == "◎ 有利"
    assert result[2]["note"] == "▲ 不利"


def test_label_advantage_excludes_ranks_and_suppresses_low_label():
    # 1〜3番人気は除外し、不利ラベルは出さない設定（人気データ向けの使い方）
    rows = [_row(1, 50, 10, 5, 5), _row(2, 25, 10, 10, 25), _row(3, 1, 1, 1, 47), _row(4, 20, 15, 10, 5)]

    result = c._label_advantage(rows, exclude_ranks={1, 2, 3}, high_label="★ ねらい目", low_label="")

    assert result[0]["note"] == ""
    assert result[2]["note"] == ""
    # 4番人気は対象（1〜3を除いた候補は4のみ）だが、候補が1件のためラベルなし
    assert result[3]["note"] == ""


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

    assert "<h1>東京 コース一覧</h1>" in html_content
    assert '<a href="芝-1400.html">芝1400m</a>' in html_content
    # 単なるリンクではなく、平均勝ち時計・人気が一覧表で見える
    assert "1:21.7" in html_content
    # 芝/ダート別に全距離合算した血統別成績が表示される
    assert "<h3>芝（全距離合算・上位10件）</h3>" in html_content
    assert "<h3>ダート（全距離合算・上位10件）</h3>" in html_content
    assert '<a href="../../performance/course/05_tokyo/index.html">&larr; このコースのAI成績を見る</a>' in html_content


def test_make_course_detail_page_generates_html(new_roots):
    c.make_course_detail_page(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    out_file = new_roots / "public_html" / "courses" / "05_tokyo" / "芝-1400.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "コース詳細" in html_content
