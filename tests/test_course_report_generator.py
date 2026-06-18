"""src/logic/html_generator/course_report_generator.py のテスト（オフライン）。

race_info_dataset_manager / peds_results_dataset_manager に既に集計済みの実データ
（data/race_info/, data/horse/peds_results/）を使って、コース詳細データページの
生成を検証する。
"""

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

    assert "<h1>東京 芝1400m コース詳細</h1>" in html
    assert "81655ms" in html
    assert "ロードカナロア" in html
    assert '<a href="../../index.html">&larr; HOMEへ戻る</a>' in html


def test_make_course_index_page_generates_html(new_roots):
    c.make_course_index_page()

    out_file = new_roots / "public_html" / "courses" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "<h1>コース詳細データ</h1>" in html_content
    assert '<a href="05_tokyo/index.html">東京</a>' in html_content


def test_make_track_page_generates_html(new_roots):
    c.make_track_page(SAMPLE_PLACE_ID)

    out_file = new_roots / "public_html" / "courses" / "05_tokyo" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "<h1>東京 コース一覧</h1>" in html_content
    assert '<a href="芝-1400.html">芝1400m</a>' in html_content


def test_make_course_detail_page_generates_html(new_roots):
    c.make_course_detail_page(SAMPLE_PLACE_ID, SAMPLE_RACE_TYPE, SAMPLE_COURSE_LEN)

    out_file = new_roots / "public_html" / "courses" / "05_tokyo" / "芝-1400.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")
    assert "コース詳細" in html_content
