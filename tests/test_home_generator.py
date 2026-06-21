"""src/logic/html_generator/home_generator.py のテスト（オフライン）。

Homeページ（public_html/index.html）が、レースカレンダー・AI成績・コース詳細データへの
リンクを含む形で生成されることを確認する。
"""

import pytest

from src.config import paths
from src.logic.html_generator import home_generator as h
from src.managers import ai_performance_dataset_manager as dataset_manager


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_PATH", str(tmp_path / "public_html"))
    monkeypatch.setattr(paths, "AI_PERFORMANCE_DATA_PATH", str(tmp_path / "ai_performance"))
    monkeypatch.setattr(
        dataset_manager, "AI_PERFORMANCE_DATASET_PATH", str(tmp_path / "ai_performance" / "ai_performance.csv")
    )
    return tmp_path


def test_make_home_page_generates_index_html(new_roots):
    h.make_home_page()

    out_file = new_roots / "public_html" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_home_page() ---")
    print(f"  出力先: {out_file}")
    print(html_content)

    assert "<h1>MAR(まーる）|競馬AIデータサイト</h1>" in html_content
    assert '<a href="races/index.html">レースカレンダー</a>' in html_content
    assert '<a href="performance/index.html">AI成績</a>' in html_content
    assert '<a href="courses/index.html">コース詳細データ</a>' in html_content

    # カレンダーwidget（base_path=""で埋め込まれている）
    assert 'window.CALENDAR_BASE_PATH = "";' in html_content
    assert '<script src="assets/js/raceDays.js"></script>' in html_content

    # AI成績サマリーカード
    assert "<h3>AI予想成績</h3>" in html_content
    assert "単勝的中率（全期間）" in html_content
    assert "週別推移（単勝回収率、直近8週）" in html_content
    assert "開催中の競馬場の成績" in html_content
    assert '<a class="card-link" href="performance/index.html">AI成績の詳細を見る &rarr;</a>' in html_content

    # 先週の結果カード
    assert "<h3>先週の結果（メインレース）</h3>" in html_content

    # サイト共通ナビゲーション・列ソートJSが追加されている
    assert '<nav class="site-nav">' in html_content
    assert '<script src="assets/js/sortable-table.js"></script>' in html_content


def test_weekly_trend_html_uses_return_rate_gauge():
    from datetime import date

    from src.logic.html_generator.rate_gauge_html import return_rate_gauge_html

    trend = [
        {
            "week_start": date(2026, 6, 1),
            "week_end": date(2026, 6, 7),
            "performance": {"win": {"hit_rate": 13.0, "return_rate": 523.9, "n": 23}},
        },
        {
            "week_start": date(2026, 6, 8),
            "week_end": date(2026, 6, 14),
            "performance": {"win": {"hit_rate": 0.0, "return_rate": 0.0, "n": 0}},
        },
    ]

    html_content = h._weekly_trend_html(trend)

    print(f"\n--- _weekly_trend_html(回収率523.9%/0.0%) ---")
    print(html_content)

    # 各週の回収率ゲージ（ai_performance_report_generatorと共通の色付きゲージ）が使われる
    assert return_rate_gauge_html(523.9) in html_content
    assert return_rate_gauge_html(0.0) in html_content
