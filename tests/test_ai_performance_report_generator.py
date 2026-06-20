"""src/logic/html_generator/ai_performance_report_generator.py のテスト（オフライン）。

ai_performance_calculator の集計結果をmonkeypatchし、ページ生成（出力先・内容）を検証する。
"""

import pytest

from src.config import paths
from src.logic.html_generator import ai_performance_report_generator as r

SAMPLE_PERFORMANCE = {
    "win": {"hit_rate": 33.3, "return_rate": 120.5, "n": 3},
    "place": {"hit_rate": 50.0, "return_rate": 105.0, "n": 3},
    "trio_box": {"hit_rate": 10.0, "return_rate": 80.0, "n": 3},
}


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_PERFORMANCE_PATH", str(tmp_path / "public_html" / "performance"))
    return tmp_path


@pytest.fixture
def fake_aggregate(monkeypatch):
    monkeypatch.setattr(r.calc, "list_predicted_races", lambda: [])
    monkeypatch.setattr(r.calc, "filter_by_year", lambda pairs, year: pairs)
    monkeypatch.setattr(r.calc, "filter_by_meeting", lambda pairs, year, place_id, times: pairs)
    monkeypatch.setattr(r.calc, "filter_by_course", lambda pairs, place_id, race_type, course_len: pairs)
    monkeypatch.setattr(r.calc, "aggregate_ai_performance", lambda pairs: SAMPLE_PERFORMANCE)


def test_make_ai_performance_index_page_generates_html(new_roots):
    r.make_ai_performance_index_page(year=2026)

    out_file = new_roots / "public_html" / "performance" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_ai_performance_index_page(2026) ---")
    print(html_content)

    assert "<h1>AI予想成績</h1>" in html_content
    assert '<a href="annual/2026.html">2026年の成績を見る</a>' in html_content
    assert '<a href="course/05_tokyo/index.html">東京</a>' in html_content


def test_make_annual_performance_page_generates_html(new_roots, fake_aggregate):
    r.make_annual_performance_page(2026)

    out_file = new_roots / "public_html" / "performance" / "annual" / "2026.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_annual_performance_page(2026) ---")
    print(html_content)

    assert "<h1>2026年 AI予想成績</h1>" in html_content
    assert "<td>単勝</td><td>33.3%</td><td>120.5%</td><td>3</td>" in html_content
    assert "<td>複勝</td><td>50.0%</td><td>105.0%</td><td>3</td>" in html_content
    assert "<td>三連複(5頭BOX)</td><td>10.0%</td><td>80.0%</td><td>3</td>" in html_content


def test_make_meeting_performance_page_generates_html(new_roots, fake_aggregate):
    r.make_meeting_performance_page(2026, place_id=5, times=1)

    out_file = new_roots / "public_html" / "performance" / "meeting" / "2026" / "05_tokyo-1th.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "<h1>2026年 東京1回 AI予想成績</h1>" in html_content


def test_make_course_performance_page_generates_html(new_roots, fake_aggregate):
    r.make_course_performance_page(place_id=5, race_type="芝", course_len="1400")

    out_file = new_roots / "public_html" / "performance" / "course" / "05_tokyo" / "芝-1400.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "<h1>東京 芝1400m AI予想成績</h1>" in html_content


def test_make_course_performance_index_page_generates_html(new_roots):
    # make_ai_performance_index_pageが各place_idについてリンクするcourse/{place}/index.html
    # を生成する関数（以前は存在せず404になっていた）
    r.make_course_performance_index_page(place_id=5)

    out_file = new_roots / "public_html" / "performance" / "course" / "05_tokyo" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_course_performance_index_page(place_id=5) ---")
    print(html_content)

    assert "<h1>東京 AI予想成績</h1>" in html_content
    assert '<a href="芝-1400.html">芝1400m</a>' in html_content
    assert '<a href="../../index.html">&larr; AI成績トップへ</a>' in html_content
