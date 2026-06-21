"""src/logic/html_generator/home_generator.py のテスト（オフライン）。

Homeページ（public_html/index.html）が、レースカレンダー・AI成績・コース詳細データへの
リンクを含む形で生成されることを確認する。
"""

from datetime import date

import pandas as pd
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


def test_make_home_page_generates_index_html(new_roots, monkeypatch):
    # get_week_main_races_with_course・get_weekend_main_race_detailsは出走馬一覧
    # ページのスクレイピング・複数データセットの参照を伴うため、オフラインテストでは
    # 固定値に差し替える
    week_race = {
        "race_id": "202605030611",
        "place_id": 5,
        "race_name": "府中牝馬S",
        "race_time": "1545",
        "race_type": "芝",
        "course_len": 1800,
        "race_day": date(2026, 6, 21),
    }
    monkeypatch.setattr(h.calc, "get_week_main_races_with_course", lambda today: [week_race])
    monkeypatch.setattr(
        h.calc,
        "get_weekend_main_race_details",
        lambda weekend_end: [
            {
                "race_day": weekend_end,
                "place_id": 5,
                "race_name": "ジューンS",
                "winner_name": "カネラフィーナ",
                "pick_name": "カネラフィーナ",
                "pick_finish": "1",
                "win_hit": True,
                "win_payout": 510.0,
                "place_hit": True,
                "place_payout": 210.0,
            }
        ],
    )

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

    # メインレースカード（今週のメインレース + 先週の結果）
    assert "<h3>メインレース</h3>" in html_content
    assert "<h4>今週のメインレース</h4>" in html_content
    assert "<h4>先週の結果</h4>" in html_content
    # 今週のメインレースは、コース詳細データへ直接リンクする
    assert html_content.count('<a href="courses/05_tokyo/芝-1800.html">東京 芝1800m</a>') == 1
    # 先週の結果は、的中率ではなく的中時の配当そのものを表示する
    assert "カネラフィーナ" in html_content
    assert "510円" in html_content

    # サイト共通ナビゲーション・列ソートJSが追加されている
    assert '<nav class="site-nav">' in html_content
    assert '<script src="assets/js/sortable-table.js"></script>' in html_content


def test_home_template_applies_weekend_gating_to_last_week_results(monkeypatch):
    # 「先週の結果」は、current_results_weekend_endが返す週末を基準にする
    captured = []
    monkeypatch.setattr(h.calc, "get_week_main_races_with_course", lambda today: [])
    monkeypatch.setattr(
        h.calc, "get_weekend_main_race_details", lambda weekend_end: captured.append(weekend_end) and []
    )
    monkeypatch.setattr(h.calc, "current_results_weekend_end", lambda today: date(2026, 6, 14))

    h.home_template()

    assert captured == [date(2026, 6, 14)]


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


def test_current_meetings_html_links_place_name_to_course_detail_data():
    meetings = [{"place_id": 5, "first_day": date(2026, 6, 20), "times": 2}]
    df = pd.DataFrame(
        columns=["race_day", "year", "place_id", "times", "win_hit", "win_return", "place_hit", "place_return", "trio_box_hit", "trio_box_return"]
    )

    html_content = h._current_meetings_html(meetings, df)

    print(f"\n--- _current_meetings_html ---\n{html_content}")

    # 開催中の競馬場名から、そのコースのコース詳細データへ直接アクセスできる
    assert '<a href="courses/05_tokyo/index.html">東京</a>' in html_content
    assert "2回" in html_content


def test_current_meetings_html_handles_no_current_meetings():
    assert "現在開催中の競馬場はありません。" in h._current_meetings_html([], pd.DataFrame())


def test_week_main_races_html_shows_date_and_links_to_course():
    races = [
        {
            "race_id": "202602010411",
            "place_id": 2,
            "race_name": "UHB杯",
            "race_time": "1520",
            "race_type": "芝",
            "course_len": 1200,
            "race_day": date(2026, 6, 27),
        },
        {
            "race_id": "202605030711",
            "place_id": 5,
            "race_name": "七夕賞",
            "race_time": "1545",
            "race_type": None,
            "course_len": None,
            "race_day": date(2026, 6, 28),
        },
    ]

    html_content = h._week_main_races_html(races)

    print(f"\n--- _week_main_races_html ---\n{html_content}")

    # 土・日それぞれの日付・発走時刻が表示される
    assert "06/27 15:20" in html_content
    assert "06/28 15:45" in html_content
    assert '<a href="courses/02_hakodate/芝-1200.html">函館 芝1200m</a>' in html_content
    # コース情報が取得できなかった場合は競馬場のコース一覧へリンクする
    assert '<a href="courses/05_tokyo/index.html">東京</a>' in html_content


def test_week_main_races_html_handles_no_main_races():
    assert "今週のメインレース（11R）はありません。" in h._week_main_races_html([])


def test_bet_result_cell_html_shows_payout_instead_of_rate_when_hit():
    html_content = h._bet_result_cell_html(True, 510.0)

    assert '<span class="hit-badge win">的中</span>' in html_content
    assert "510円" in html_content
    # 的中率(%)としては表示しない
    assert "%" not in html_content


def test_bet_result_cell_html_shows_only_miss_badge_when_not_hit():
    html_content = h._bet_result_cell_html(False, None)

    assert html_content == '<span class="hit-badge miss">不的中</span>'


def test_weekend_results_html_shows_winner_pick_and_payout_on_hit():
    races = [
        {
            "race_day": date(2026, 6, 13),
            "place_id": 5,
            "race_name": "ジューンS",
            "winner_name": "カネラフィーナ",
            "pick_name": "カネラフィーナ",
            "pick_finish": "1",
            "win_hit": True,
            "win_payout": 510.0,
            "place_hit": True,
            "place_payout": 210.0,
        },
        {
            "race_day": date(2026, 6, 13),
            "place_id": 9,
            "race_name": "三宮S",
            "winner_name": "グランドプラージュ",
            "pick_name": "メイショウズイウン",
            "pick_finish": "9",
            "win_hit": False,
            "win_payout": None,
            "place_hit": False,
            "place_payout": None,
        },
    ]

    html_content = h._weekend_results_html(races)

    print(f"\n--- _weekend_results_html ---\n{html_content}")

    assert "06/13" in html_content
    assert "東京11R ジューンS" in html_content
    assert "カネラフィーナ" in html_content
    assert "1着" in html_content
    assert "510円" in html_content
    assert "210円" in html_content
    # 不的中のレースは本命馬の着順も表示する
    assert "阪神11R 三宮S" in html_content
    assert "メイショウズイウン" in html_content
    assert "9着" in html_content
    assert '<span class="hit-badge miss">不的中</span>' in html_content


def test_weekend_results_html_handles_no_races():
    assert "対象レースのデータがありません。" in h._weekend_results_html([])
