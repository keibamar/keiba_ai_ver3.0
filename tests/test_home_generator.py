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
        "race_day": date(2026, 6, 20),
    }
    monkeypatch.setattr(h.calc, "get_week_main_races_with_course", lambda today: [week_race])
    monkeypatch.setattr(
        h.calc,
        "get_current_meeting_summaries",
        lambda today: [
            {
                "place_id": 5,
                "times": 2,
                "days": [
                    {"day_date": date(2026, 6, 20), "day_number": 7},
                    {"day_date": date(2026, 6, 21), "day_number": 8},
                ],
            }
        ],
    )
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

    # Homeにも、site_nav_html経由で右側の小さなカレンダータブ（矢印で前後月へ移動可能）
    # ＋現在地（HOME）の強調表示が常に表示される
    assert '<aside class="page-calendar-tab">' in html_content
    assert '<div class="page-calendar-tab-calendar">' in html_content
    assert '<button id="prevMonth">&larr;</button>' in html_content
    assert '<button id="nextMonth">&rarr;</button>' in html_content
    assert '<span class="page-calendar-tab-current">HOME</span>' in html_content

    # Homeの最上部（タイトルの下）に今週の開催情報を、土曜・日曜それぞれ表示する
    assert "<h2>今週の開催</h2>" in html_content
    assert '<span class="main"><a href="courses/05_tokyo/index.html">東京</a> 第2回</span>' in html_content
    # 出馬表一覧ページが生成済みの06-20/06-21は、その日の開催一覧へのリンクになる
    assert '<a href="races/20260620/index.html">06/20(土) 7日目</a>' in html_content
    assert '<a href="races/20260621/index.html">06/21(日) 8日目</a>' in html_content
    assert html_content.index("<h1>") < html_content.index("<h2>今週の開催</h2>")

    # 今週のメインレースは、出馬表生成済みのレース（06-20の東京11R）はレース名から出馬表へ飛べる
    assert '<a href="races/20260620/05_tokyoR11.html">府中牝馬S</a>' in html_content

    assert '<a href="races/index.html">レースカレンダー</a>' in html_content
    assert '<a href="performance/index.html">AI成績</a>' in html_content
    assert '<a href="courses/index.html">コース詳細データ</a>' in html_content

    # カレンダーミニタブ（base_path=""で埋め込まれている）
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
    assert html_content.count('<a href="courses/05_tokyo/芝-1800.html">東京<br><span class="race-type-turf">芝1800m</span></a>') == 1
    # 先週の結果は、的中率ではなく的中時の配当そのものを表示する
    assert "カネラフィーナ" in html_content
    assert "510円" in html_content

    # サイト共通ナビゲーション・列ソートJSが追加されている
    assert '<nav class="site-nav">' in html_content
    assert "pagead2.googlesyndication.com" in html_content
    assert '<script src="assets/js/sortable-table.js"></script>' in html_content
    # https://mar-keiba.com/ と https://mar-keiba.com/index.html の重複URLによる
    # SEO上の評価分散を避けるため、正規URLを明示する
    assert '<link rel="canonical" href="https://mar-keiba.com/">' in html_content
    # HOMEへのリンクはindex.htmlを明示しない（同じ理由）
    assert '<a class="site-brand" href="./">' in html_content


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
    # 競馬場名がメイン、開催回数がサブの2行表示
    assert '<span class="main"><a href="courses/05_tokyo/index.html">東京</a></span><span class="sub">2回</span>' in html_content


def test_current_meetings_html_handles_no_current_meetings():
    assert "現在開催中の競馬場はありません。" in h._current_meetings_html([], pd.DataFrame())


def test_date_with_weekday_html_colors_saturday_blue_and_sunday_red():
    assert h._date_with_weekday_html(date(2026, 6, 20)) == '06/20<span class="weekday-sat">(土)</span>'
    assert h._date_with_weekday_html(date(2026, 6, 21)) == '06/21<span class="weekday-sun">(日)</span>'


def test_date_with_weekday_html_colors_holiday_red():
    # 2026-07-20(月)は海の日(祝日)
    assert h._date_with_weekday_html(date(2026, 7, 20)) == '07/20<span class="weekday-sun">(月)</span>'


def test_date_with_weekday_html_weekday_has_no_color_class():
    assert h._date_with_weekday_html(date(2026, 6, 22)) == "06/22<span>(月)</span>"


def test_weekly_meeting_summary_html_shows_place_main_and_each_day_sub():
    # 06-20(土)は出馬表一覧ページが無く、06-21(日)はある状態を想定する
    summaries = [
        {
            "place_id": 5,
            "times": 2,
            "days": [
                {"day_date": date(2026, 6, 20), "day_number": 7},
                {"day_date": date(2026, 6, 21), "day_number": 8},
            ],
        }
    ]

    html_content = h._weekly_meeting_summary_html(summaries)

    print(f"\n--- _weekly_meeting_summary_html ---\n{html_content}")

    # 競馬場名はコース詳細データへ、メインで大きく表示する
    assert '<span class="main"><a href="courses/05_tokyo/index.html">東京</a> 第2回</span>' in html_content
    # 土曜・日曜それぞれの日付・開催日目を表示し、出馬表一覧ページが生成済みならそこへリンクする
    assert '<a href="races/20260620/index.html">06/20(土) 7日目</a>' in html_content
    assert '<a href="races/20260621/index.html">06/21(日) 8日目</a>' in html_content


def test_weekly_meeting_summary_html_shows_plain_label_when_day_index_missing():
    summaries = [
        {
            "place_id": 5,
            "times": 99,
            "days": [{"day_date": date(2020, 1, 1), "day_number": 1}],
        }
    ]

    html_content = h._weekly_meeting_summary_html(summaries)

    # 出馬表一覧ページが無い日はリンクにせず、日付・開催日目のみ表示する
    assert "<span>01/01(水) 1日目</span>" in html_content


def test_weekly_meeting_summary_html_handles_no_meetings():
    assert "今週開催中の競馬場はありません。" in h._weekly_meeting_summary_html([])


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

    # 土・日それぞれの日付（曜日付き）・発走時刻が表示される（土曜は青、日曜は赤）
    assert '06/27<span class="weekday-sat">(土)</span> 15:20' in html_content
    assert '06/28<span class="weekday-sun">(日)</span> 15:45' in html_content
    # 「函館芝1200m」のように続けて並ぶと読みにくいため、競馬場名とコースの間で改行する
    assert '<a href="courses/02_hakodate/芝-1200.html">函館<br><span class="race-type-turf">芝1200m</span></a>' in html_content
    # コース情報が取得できなかった場合は競馬場のコース一覧へリンクする
    assert '<a href="courses/05_tokyo/index.html">東京</a>' in html_content
    # レース名がメイン、開催場・レース番号がサブの2行表示
    assert '<span class="main">UHB杯</span><span class="sub">函館11R</span>' in html_content
    assert '<span class="main">七夕賞</span><span class="sub">東京11R</span>' in html_content


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

    # 2026-06-13は土曜なので青色の曜日表示になる
    assert '06/13<span class="weekday-sat">(土)</span>' in html_content
    # レース名は、その日の出馬表ページ（races/{date}/{place}R11.html）が生成済みなら
    # そこへリンクする（結果が出た後でも出走馬・オッズ等を確認できるようにする）
    assert '<span class="main"><a href="races/20260613/05_tokyoR11.html">ジューンS</a></span><span class="sub">東京11R</span>' in html_content
    assert "カネラフィーナ" in html_content
    assert "1着" in html_content
    assert "510円" in html_content
    assert "210円" in html_content
    # 不的中のレースは本命馬の着順も表示する
    assert '<span class="main"><a href="races/20260613/09_hanshinR11.html">三宮S</a></span><span class="sub">阪神11R</span>' in html_content
    assert "メイショウズイウン" in html_content
    assert "9着" in html_content
    assert '<span class="hit-badge miss">不的中</span>' in html_content


def test_weekend_results_html_handles_no_races():
    assert "対象レースのデータがありません。" in h._weekend_results_html([])
